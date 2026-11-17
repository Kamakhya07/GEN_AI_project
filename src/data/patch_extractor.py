import os
import random
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image

from .rdd_parser import RDDAnnotation, BoundingBox

class PatchExtractor:
    def __init__(self, target_size: Tuple[int, int] = (128, 128), padding_ratio: float = 0.15, normal_patches_per_image: int = 2):
        self.target_size = target_size
        self.padding_ratio = padding_ratio
        self.normal_patches_per_image = normal_patches_per_image

    def extract_damage_patches(self, image: np.ndarray, annotation: RDDAnnotation, output_dir: str) -> List[Dict[str, Any]]:
        """
        Crops labeled road defects from an image with padding and saves them.
        Returns list of metadata dicts.
        """
        records = []
        h, w = image.shape[:2]

        for i, box in enumerate(annotation.boxes):
            bw = box.xmax - box.xmin
            bh = box.ymax - box.ymin

            # Context padding
            pad_x = int(bw * self.padding_ratio)
            pad_y = int(bh * self.padding_ratio)

            crop_xmin = max(0, box.xmin - pad_x)
            crop_ymin = max(0, box.ymin - pad_y)
            crop_xmax = min(w, box.xmax + pad_x)
            crop_ymax = min(h, box.ymax + pad_y)

            crop = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_AREA)

            patch_filename = f"{annotation.image_id}_defect_{i}_class_{box.class_id}.jpg"
            patch_path = os.path.join(output_dir, patch_filename)
            cv2.imwrite(patch_path, resized)

            records.append({
                "patch_id": patch_filename,
                "source_image_id": annotation.image_id,
                "patch_path": patch_path,
                "class_id": box.class_id,
                "class_name": box.class_name,
                "is_normal": False,
                "bbox": [box.xmin, box.ymin, box.xmax, box.ymax],
                "crop_coords": [crop_xmin, crop_ymin, crop_xmax, crop_ymax]
            })

        return records

    def extract_normal_patches(self, image: np.ndarray, annotation: RDDAnnotation, output_dir: str, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Samples background road-surface patches from non-defective regions.
        """
        records = []
        h, w = image.shape[:2]
        pw, ph = self.target_size

        rng = random.Random(seed + hash(annotation.image_id) % 100000)

        # Create defect mask
        defect_mask = np.zeros((h, w), dtype=np.uint8)
        for box in annotation.boxes:
            # Mask out defect region with buffer
            x0 = max(0, box.xmin - 10)
            y0 = max(0, box.ymin - 10)
            x1 = min(w, box.xmax + 10)
            y1 = min(h, box.ymax + 10)
            defect_mask[y0:y1, x0:x1] = 1

        attempts = 0
        extracted = 0
        max_attempts = 40

        while extracted < self.normal_patches_per_image and attempts < max_attempts:
            attempts += 1
            # Road surfaces are typically in lower two-thirds of dashcam/CCTV frames
            y_start = int(h * 0.3)
            if y_start + ph >= h or w <= pw:
                break

            ry = rng.randint(y_start, h - ph)
            rx = rng.randint(0, w - pw)

            # Check if candidate overlaps any defect
            if np.sum(defect_mask[ry:ry+ph, rx:rx+pw]) == 0:
                crop = image[ry:ry+ph, rx:rx+pw]
                resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_AREA)

                patch_filename = f"{annotation.image_id}_normal_{extracted}.jpg"
                patch_path = os.path.join(output_dir, patch_filename)
                cv2.imwrite(patch_path, resized)

                records.append({
                    "patch_id": patch_filename,
                    "source_image_id": annotation.image_id,
                    "patch_path": patch_path,
                    "class_id": 0,
                    "class_name": "Normal",
                    "is_normal": True,
                    "bbox": None,
                    "crop_coords": [rx, ry, rx + pw, ry + ph]
                })
                extracted += 1

        return records
