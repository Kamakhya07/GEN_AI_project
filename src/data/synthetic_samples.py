import os
import random
import cv2
import numpy as np
from typing import Tuple

def generate_synthetic_road_patch(class_id: int, size: Tuple[int, int] = (128, 128), seed: int = 42) -> np.ndarray:
    """
    Generates a realistic synthetic road-surface patch for reproducible smoke testing and testing pipeline flows.
    Classes:
    0: Normal road (asphalt grain and slight color variations)
    1: Longitudinal crack (vertical crack lines)
    2: Transverse crack (horizontal crack lines)
    3: Alligator crack (interconnected polygon/mesh cracks)
    4: Pothole (dark elliptical cavity with rough perimeter)
    """
    np.random.seed(seed)
    rng = random.Random(seed)
    h, w = size

    # Base asphalt texture (gray base with Gaussian noise grain)
    base_intensity = np.random.randint(90, 130)
    base = np.full((h, w, 3), base_intensity, dtype=np.float32)
    grain = np.random.normal(0, 12, (h, w, 3)).astype(np.float32)
    patch = np.clip(base + grain, 0, 255).astype(np.uint8)

    # Class 0 is clean/normal asphalt
    if class_id == 0:
        return patch

    # Draw damage features
    # Class 1: Longitudinal crack (mostly vertical with random walk)
    if class_id == 1:
        x = rng.randint(int(w * 0.3), int(w * 0.7))
        points = []
        for y in range(0, h, 4):
            x += rng.randint(-2, 2)
            x = max(5, min(w - 6, x))
            points.append((x, y))
        for p1, p2 in zip(points[:-1], points[1:]):
            cv2.line(patch, p1, p2, (30, 30, 30), thickness=rng.randint(2, 3))
            # subtle edge shading
            cv2.line(patch, (p1[0] + 1, p1[1]), (p2[0] + 1, p2[1]), (50, 50, 50), thickness=1)

    # Class 2: Transverse crack (mostly horizontal with random walk)
    elif class_id == 2:
        y = rng.randint(int(h * 0.3), int(h * 0.7))
        points = []
        for x in range(0, w, 4):
            y += rng.randint(-2, 2)
            y = max(5, min(h - 6, y))
            points.append((x, y))
        for p1, p2 in zip(points[:-1], points[1:]):
            cv2.line(patch, p1, p2, (30, 30, 30), thickness=rng.randint(2, 3))
            cv2.line(patch, (p1[0], p1[1] + 1), (p2[0], p2[1] + 1), (50, 50, 50), thickness=1)

    # Class 3: Alligator crack (mesh/polygonal interconnected cracks)
    elif class_id == 3:
        # Create Voronoi-like crack network
        num_cells = rng.randint(6, 10)
        centers = [(rng.randint(15, w - 15), rng.randint(15, h - 15)) for _ in range(num_cells)]
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                if dist < 45:
                    cv2.line(patch, centers[i], centers[j], (35, 35, 35), thickness=rng.randint(1, 2))

    # Class 4: Pothole (dark irregular hollow with rough edges)
    elif class_id == 4:
        center = (rng.randint(int(w * 0.35), int(w * 0.65)), rng.randint(int(h * 0.35), int(h * 0.65)))
        axes = (rng.randint(18, 30), rng.randint(14, 24))
        angle = rng.randint(0, 180)
        # Inner dark depression
        cv2.ellipse(patch, center, axes, angle, 0, 360, (25, 25, 25), -1)
        # Rough border
        cv2.ellipse(patch, center, (axes[0] + 3, axes[1] + 3), angle, 0, 360, (45, 45, 45), 2)
        # Add internal rough texture
        noise_mask = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
        patch = cv2.add(patch, noise_mask // 3)

    return patch

def generate_synthetic_road_dataset(output_dir: str, num_images: int = 100, seed: int = 42):
    """
    Creates a full mock dataset containing sharp patches, manifest, and source-level splits for smoke tests and initial offline training.
    """
    os.makedirs(output_dir, exist_ok=True)
    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    class_names = {
        0: "Normal",
        1: "Longitudinal Crack",
        2: "Transverse Crack",
        3: "Alligator Crack",
        4: "Pothole"
    }

    records = []
    rng = random.Random(seed)

    # Simulate num_images distinct road source photos
    for img_idx in range(num_images):
        source_id = f"mock_road_{img_idx:04d}"
        
        # Each road image has 1-3 patches (normal patches and damage patches)
        num_patches = rng.randint(1, 3)
        for p_idx in range(num_patches):
            class_id = rng.choice([0, 0, 1, 2, 3, 4]) # realistic distribution with more normal
            patch_img = generate_synthetic_road_patch(class_id=class_id, seed=(seed + img_idx * 10 + p_idx))
            
            patch_name = f"{source_id}_patch_{p_idx}_cls_{class_id}.jpg"
            patch_path = os.path.join(patches_dir, patch_name)
            cv2.imwrite(patch_path, patch_img)

            records.append({
                "patch_id": patch_name,
                "source_image_id": source_id,
                "patch_path": patch_path,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "is_normal": (class_id == 0),
                "bbox": None
            })

    # Split using SplitManager
    from .split_manager import SplitManager
    sm = SplitManager(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed)
    splits = sm.create_source_level_splits(records)
    
    manifest_path = os.path.join(output_dir, "splits_manifest.json")
    counts_path = os.path.join(output_dir, "class_counts.json")
    sm.save_manifest(splits, manifest_path, counts_path)

    print(f"Generated synthetic dataset with {len(records)} patches across {num_images} source images at {output_dir}")
    return manifest_path, counts_path
