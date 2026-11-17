import json
import os
from typing import Dict, Any, List, Optional, Callable, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class RoadPatchClassificationDataset(Dataset):
    """
    PyTorch Dataset for patch-level road damage classification.
    Returns: (image_tensor, label_int)
    """
    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        degradation_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        target_size: Tuple[int, int] = (128, 128)
    ):
        self.manifest_path = manifest_path
        self.split = split
        self.transform = transform
        self.degradation_fn = degradation_fn
        self.target_size = target_size

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if split not in manifest:
            raise ValueError(f"Split '{split}' not found in manifest at {manifest_path}. Available: {list(manifest.keys())}")

        self.records: List[Dict[str, Any]] = manifest[split]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        record = self.records[idx]
        img_path = record["patch_path"]

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image at {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply degradation function (e.g., motion or defocus blur) if provided
        if self.degradation_fn is not None:
            img_rgb = self.degradation_fn(img_rgb)

        if (img_rgb.shape[0], img_rgb.shape[1]) != self.target_size:
            img_rgb = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_AREA)

        label = int(record["class_id"])

        if self.transform is not None:
            img_pil = Image.fromarray(img_rgb)
            img_tensor = self.transform(img_pil)
        else:
            # Default normalization to [0, 1] range: (C, H, W)
            img_norm = img_rgb.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1)

        patch_id = record.get("patch_id", "")
        return img_tensor, label, patch_id

class PairedDeblurDataset(Dataset):
    """
    Paired blurry-sharp patch dataset for Conditional GAN (U-Net + PatchGAN) training.
    Returns: (blurry_tensor, sharp_tensor) normalized to [-1, 1] for Tanh generator activation.
    """
    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        degradation_pipeline: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        target_size: Tuple[int, int] = (128, 128)
    ):
        self.manifest_path = manifest_path
        self.split = split
        self.degradation_pipeline = degradation_pipeline
        self.target_size = target_size

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.records = manifest[split]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        img_path = record["patch_path"]

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")

        sharp_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if (sharp_rgb.shape[0], sharp_rgb.shape[1]) != self.target_size:
            sharp_rgb = cv2.resize(sharp_rgb, self.target_size, interpolation=cv2.INTER_AREA)

        # Apply synthetic degradation
        if self.degradation_pipeline is not None:
            blurry_rgb = self.degradation_pipeline(sharp_rgb)
        else:
            # Fallback simple blur if no pipeline passed
            blurry_rgb = cv2.GaussianBlur(sharp_rgb, (11, 11), 3.0)

        # Normalize images to [-1, 1] for Pix2Pix / cGAN training
        blurry_norm = (blurry_rgb.astype(np.float32) / 127.5) - 1.0
        sharp_norm = (sharp_rgb.astype(np.float32) / 127.5) - 1.0

        blurry_tensor = torch.from_numpy(blurry_norm).permute(2, 0, 1)
        sharp_tensor = torch.from_numpy(sharp_norm).permute(2, 0, 1)

        return blurry_tensor, sharp_tensor
