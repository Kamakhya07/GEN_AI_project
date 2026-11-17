import os
import glob
import argparse
import cv2
from tqdm import tqdm

from src.data.rdd_parser import parse_rdd_annotation
from src.data.patch_extractor import PatchExtractor
from src.data.split_manager import SplitManager

def process_rdd_directory(
    raw_dir: str,
    output_dir: str,
    patch_size: int = 128,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    os.makedirs(output_dir, exist_ok=True)
    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    xml_files = glob.glob(os.path.join(raw_dir, "**", "*.xml"), recursive=True)
    if not xml_files:
        print(f"No XML annotation files found in {raw_dir}")
        return

    print(f"Found {len(xml_files)} annotation files in {raw_dir}")

    extractor = PatchExtractor(
        target_size=(patch_size, patch_size),
        padding_ratio=0.15,
        normal_patches_per_image=2
    )

    all_records = []
    for xml_path in tqdm(xml_files, desc="Processing annotations"):
        img_dir = os.path.join(os.path.dirname(os.path.dirname(xml_path)), "images")
        if not os.path.exists(img_dir):
            img_dir = os.path.dirname(xml_path)

        anno = parse_rdd_annotation(xml_path, image_dir=img_dir)
        if anno is None:
            continue

        if not os.path.exists(anno.image_path):
            continue

        img_bgr = cv2.imread(anno.image_path)
        if img_bgr is None:
            continue

        # Extract damage defect patches
        damage_patches = extractor.extract_damage_patches(img_bgr, anno, patches_dir)
        # Extract normal road patches
        normal_patches = extractor.extract_normal_patches(img_bgr, anno, patches_dir, seed=seed)

        all_records.extend(damage_patches)
        all_records.extend(normal_patches)

    print(f"Extracted {len(all_records)} total patches.")

    # Split by source image
    sm = SplitManager(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    splits = sm.create_source_level_splits(all_records)

    manifest_path = os.path.join(output_dir, "splits_manifest.json")
    counts_path = os.path.join(output_dir, "class_counts.json")
    sm.save_manifest(splits, manifest_path, counts_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, help="Path to RDD2022 subset")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--patch_size", type=int, default=128)
    args = parser.parse_args()

    process_rdd_directory(args.raw_dir, args.output_dir, args.patch_size)
