import os
import glob
import random
import json
import cv2
import numpy as np
from typing import List, Dict, Any

from src.data.split_manager import SplitManager

def process_real_road_data(
    raw_data_dir: str = "data/raw/road_damage_real/data",
    output_patch_dir: str = "data/processed/real_patches",
    splits_output_path: str = "data/processed/splits_real.json",
    samples_output_dir: str = "data/samples/real",
    target_size: tuple = (128, 128),
    max_patches_per_class: int = 250,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_patch_dir, exist_ok=True)
    os.makedirs(samples_output_dir, exist_ok=True)

    images_dir = os.path.join(raw_data_dir, "images")
    labels_dir = os.path.join(raw_data_dir, "labels-YOLO")

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    print(f"Found {len(image_paths)} real vehicle camera images.")

    class_counts = {0: 0, 1: 0, 4: 0} # 0: Normal, 1: Crack, 4: Pothole
    all_records: List[Dict[str, Any]] = []

    # Map dataset class IDs (0: Pothole, 1: Crack, 2: Manhole) to our project classes:
    # 4: Pothole
    # 1: Crack (Longitudinal/Surface Crack)
    # 0: Normal (extracted from clear pavement background)
    LABEL_MAP = {
        0: (4, "Pothole"),
        1: (1, "Longitudinal Crack")
    }

    sample_saved = {"Normal": False, "Pothole": False, "Crack": False}

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        xc = float(parts[1]) * w
                        yc = float(parts[2]) * h
                        bw = float(parts[3]) * w
                        bh = float(parts[4]) * h
                        xmin = int(xc - bw / 2)
                        ymin = int(yc - bh / 2)
                        xmax = int(xc + bw / 2)
                        ymax = int(yc + bh / 2)
                        boxes.append((cid, xmin, ymin, xmax, ymax))

        # Extract defect patches
        for idx, (cid, xmin, ymin, xmax, ymax) in enumerate(boxes):
            if cid not in LABEL_MAP:
                continue
            
            proj_cid, proj_cname = LABEL_MAP[cid]
            if class_counts[proj_cid] >= max_patches_per_class:
                continue

            # Add context padding
            bw = xmax - xmin
            bh = ymax - ymin
            pad_x = int(bw * 0.2)
            pad_y = int(bh * 0.2)

            crop_xmin = max(0, xmin - pad_x)
            crop_ymin = max(0, ymin - pad_y)
            crop_xmax = min(w, xmax + pad_x)
            crop_ymax = min(h, ymax + pad_y)

            crop = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            patch = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
            patch_name = f"real_{base_name}_d{idx}_c{proj_cid}.jpg"
            patch_path = os.path.join(output_patch_dir, patch_name)
            cv2.imwrite(patch_path, patch)

            class_counts[proj_cid] += 1
            all_records.append({
                "patch_id": patch_name,
                "source_image_id": base_name,
                "patch_path": patch_path,
                "class_id": proj_cid,
                "class_name": proj_cname,
                "is_normal": False,
                "bbox": [xmin, ymin, xmax, ymax]
            })

            # Save high-res sample for UI
            if proj_cid == 4 and not sample_saved["Pothole"]:
                sample_pothole_path = os.path.join(samples_output_dir, "real_pothole_sample.jpg")
                cv2.imwrite(sample_pothole_path, cv2.resize(crop, (256, 256)))
                sample_saved["Pothole"] = True
            elif proj_cid == 1 and not sample_saved["Crack"]:
                sample_crack_path = os.path.join(samples_output_dir, "real_crack_sample.jpg")
                cv2.imwrite(sample_crack_path, cv2.resize(crop, (256, 256)))
                sample_saved["Crack"] = True

        # Extract normal patch from lower half (road surface) if far from defect boxes
        if class_counts[0] < max_patches_per_class and len(boxes) < 4:
            # Try random crop in road zone (y in [int(h*0.5), h])
            crop_size = min(150, int(h * 0.4), int(w * 0.4))
            for _ in range(5):
                rx = random.randint(0, max(0, w - crop_size))
                ry = random.randint(int(h * 0.45), max(int(h * 0.45), h - crop_size))
                rx2 = rx + crop_size
                ry2 = ry + crop_size

                # Check IoU / overlap with defect boxes
                overlaps = False
                for (_, bx1, by1, bx2, by2) in boxes:
                    if not (rx2 < bx1 or rx > bx2 or ry2 < by1 or ry > by2):
                        overlaps = True
                        break

                if not overlaps:
                    normal_crop = img[ry:ry2, rx:rx2]
                    if normal_crop.size > 0 and normal_crop.shape[0] >= 20 and normal_crop.shape[1] >= 20:
                        patch = cv2.resize(normal_crop, target_size, interpolation=cv2.INTER_AREA)
                        patch_name = f"real_{base_name}_norm.jpg"
                        patch_path = os.path.join(output_patch_dir, patch_name)
                        cv2.imwrite(patch_path, patch)

                        class_counts[0] += 1
                        all_records.append({
                            "patch_id": patch_name,
                            "source_image_id": base_name,
                            "patch_path": patch_path,
                            "class_id": 0,
                            "class_name": "Normal",
                            "is_normal": True,
                            "bbox": [rx, ry, rx2, ry2]
                        })

                        if not sample_saved["Normal"]:
                            sample_norm_path = os.path.join(samples_output_dir, "real_normal_sample.jpg")
                            cv2.imwrite(sample_norm_path, cv2.resize(normal_crop, (256, 256)))
                            sample_saved["Normal"] = True
                        break

        # Stop early once we reach desired quota
        if all(count >= max_patches_per_class for count in class_counts.values()):
            break

    print(f"\nExtracted Patches Summary:")
    for cid, cnt in class_counts.items():
        cname = "Normal" if cid == 0 else ("Crack" if cid == 1 else "Pothole")
        print(f"  Class {cid} ({cname}): {cnt} patches")

    # Split using SplitManager to guarantee leak-free splits
    split_mgr = SplitManager(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed)
    splits = split_mgr.create_source_level_splits(all_records)
    class_counts_path = os.path.splitext(splits_output_path)[0] + "_class_distribution.json"
    split_mgr.save_manifest(splits, splits_output_path, class_counts_path)

    print(f"\nSplits successfully saved to: {splits_output_path}")
    print(f"  Train: {len(splits['train'])} patches")
    print(f"  Val:   {len(splits['val'])} patches")
    print(f"  Test:  {len(splits['test'])} patches")
    return splits

if __name__ == "__main__":
    process_real_road_data()
