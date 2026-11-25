"""
Member 1: DeblurGAN-v2 Inference Engine
=======================================
Coordinates official DeblurGAN-v2 road surface restoration with real weights.
Automatically detects checkpoint presence and handles padding to block size 32.

Reference:
    Orest Kupyn, Tetiana Martyniuk, Junru Wu, Zhangyang Wang.
    "DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better." (ICCV 2019)
    GitHub: https://github.com/VITA-Group/DeblurGANv2
    License: BSD 2-Clause / BSD 3-Clause
"""

import os
import time
from typing import Optional, Tuple
import cv2
import numpy as np
import torch

from .model import OfficialDeblurGANv2

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, "..", "..", ".."))

# Candidate checkpoint search paths (priority order)
GAN_CANDIDATE_PATHS = [
    os.path.join(_DIR, "checkpoints", "fpn_mobilenet.h5"),
    os.path.join(_DIR, "checkpoints", "best_gan.pt"),
    os.path.join(_ROOT, "fpn_mobilenet.h5")
]

MEMBER1_CHECKPOINT_PATH = GAN_CANDIDATE_PATHS[0]


def get_gan_checkpoint_path() -> Optional[str]:
    """Finds the first existing valid DeblurGAN checkpoint file."""
    for p in GAN_CANDIDATE_PATHS:
        if os.path.exists(p) and os.path.getsize(p) > 1024:
            return p
    return None


def is_member1_model_available() -> bool:
    """Returns True if DeblurGAN-v2 weights are available."""
    return get_gan_checkpoint_path() is not None


def load_member1_gan(device: Optional[torch.device] = None) -> Optional[OfficialDeblurGANv2]:
    """
    Loads Member 1's DeblurGAN-v2 model from checkpoints.
    Returns None if no compatible checkpoint is found.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = get_gan_checkpoint_path()
    if ckpt_path is None:
        return None

    try:
        model = OfficialDeblurGANv2()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        if isinstance(ckpt, dict):
            if "model" in ckpt:
                raw_state = ckpt["model"]
            elif "generator_state_dict" in ckpt:
                raw_state = ckpt["generator_state_dict"]
            elif "state_dict" in ckpt:
                raw_state = ckpt["state_dict"]
            else:
                raw_state = ckpt
        else:
            raw_state = ckpt

        # Strip 'module.' prefix if wrapped by DataParallel in original training
        clean_state = {}
        for k, v in raw_state.items():
            new_k = k[7:] if k.startswith("module.") else k
            # Strip model. prefix if nested
            if new_k.startswith("generator."):
                new_k = new_k[10:]
            clean_state[new_k] = v

        model.model.load_state_dict(clean_state, strict=True)
        model.to(device)
        model.eval()
        print(f"[DeblurGAN-v2] Successfully loaded official weights from: {ckpt_path}")
        return model
    except Exception as e:
        print(f"[DeblurGAN-v2 Integration Error] Failed to load checkpoint: {e}")
        return None


def run_member1_deblur(
    model: OfficialDeblurGANv2,
    degraded_rgb: np.ndarray,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, float]:
    """
    Runs DeblurGAN-v2 deblurring on a degraded RGB road image.
    
    Args:
        model: Loaded OfficialDeblurGANv2 instance.
        degraded_rgb: Input RGB image (H, W, 3) uint8.
        device: PyTorch device.
        
    Returns:
        (restored_rgb, latency_ms)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()

    h, w = degraded_rgb.shape[:2]
    # DeblurGAN-v2 FPN architecture requires dimensions to be multiples of 32
    block_size = 32
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size

    if pad_h > 0 or pad_w > 0:
        padded = cv2.copyMakeBorder(degraded_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    else:
        padded = degraded_rgb

    # Normalize to [-1.0, 1.0]
    norm_in = ((padded.astype(np.float32) / 127.5) - 1.0)
    tensor_in = torch.from_numpy(norm_in).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor_in)
        out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        restored_padded = ((out_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        restored = restored_padded[:h, :w]

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return restored, latency_ms
