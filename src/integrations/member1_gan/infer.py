"""
Member 1: GAN Inference Engine
===============================
Provides decoupled inference hooks for Member 1's GAN restoration pipeline.
Automatically detects checkpoint presence and handles device mapping.
"""

import os
import time
from typing import Optional, Tuple
import numpy as np
import torch

from .model import Member1GAN

_DIR = os.path.dirname(os.path.abspath(__file__))
MEMBER1_CHECKPOINT_PATH = os.path.join(_DIR, "checkpoints", "best_gan.pt")


def is_member1_model_available() -> bool:
    """Returns True if Member 1 has provided their trained model checkpoint."""
    return os.path.exists(MEMBER1_CHECKPOINT_PATH) and os.path.getsize(MEMBER1_CHECKPOINT_PATH) > 1024


def load_member1_gan(device: Optional[torch.device] = None) -> Optional[Member1GAN]:
    """
    Loads Member 1's GAN model from checkpoints/best_gan.pt.
    Returns None if the checkpoint is not yet integrated.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not is_member1_model_available():
        return None

    model = Member1GAN()
    try:
        ckpt = torch.load(MEMBER1_CHECKPOINT_PATH, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if "generator_state_dict" in ckpt:
                model.generator.load_state_dict(ckpt["generator_state_dict"])
            elif "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            elif "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            else:
                model.load_state_dict(ckpt)
        else:
            model = ckpt
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[Member 1 Integration Error] Failed to load checkpoint: {e}")
        return None


def run_member1_deblur(
    model: Member1GAN,
    degraded_rgb: np.ndarray,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, float]:
    """
    Executes Member 1's GAN deblurring model on a degraded input RGB image.
    
    Args:
        model: Member1GAN instance
        degraded_rgb: HxWxC uint8 numpy array in RGB format
        device: Torch device
        
    Returns:
        (restored_rgb, latency_ms)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()
    with torch.no_grad():
        # Normalize [0, 255] -> [-1, 1]
        norm_in = ((degraded_rgb.astype(np.float32) / 127.5) - 1.0)
        tensor_in = torch.from_numpy(norm_in).permute(2, 0, 1).unsqueeze(0).to(device)
        out = model(tensor_in)
        restored = ((out.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        lat_ms = (time.perf_counter() - t0) * 1000.0

    return restored, lat_ms
