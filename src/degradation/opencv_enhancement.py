"""
OpenCV Road Surface Image Enhancement Module
============================================
Provides non-generative, deterministic image restoration and enhancement for road
surfaces degraded by optical blur, low contrast, and sensor noise.

Implements three classical restoration stages:
1. Denoising: Bilateral filtering or Fast Non-Local Means to reduce asphalt sensor grain
2. CLAHE (Contrast Limited Adaptive Histogram Equalization): Enhances micro-crack and
   pothole depth contrast in the LAB luminance (L) channel without distorting road hue
3. Image Sharpening: Unsharp masking with Gaussian spatial frequency subtraction
   to recover sharp perimeter boundaries and high-frequency defect textures.
"""

import time
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


def denoise_road_image(
    image_rgb: np.ndarray,
    method: str = "bilateral",
    diameter: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0
) -> np.ndarray:
    """
    Suppresses sensor grain and high-frequency road surface noise while preserving edges.
    
    Args:
        image_rgb: RGB input image (H, W, 3) uint8.
        method: 'bilateral' (fast, edge-preserving) or 'fast_nl' (Non-Local Means).
        diameter: Pixel neighborhood diameter for bilateral filter.
        sigma_color: Filter sigma in the color space.
        sigma_space: Filter sigma in the coordinate space.
        
    Returns:
        Denoised RGB image.
    """
    if method == "fast_nl":
        # Fast Non-Local Means (higher quality, slightly higher latency)
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        denoised_bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)
        return cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    else:
        # Bilateral filter preserves sharp crack boundaries while smoothing asphalt noise
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        denoised_bgr = cv2.bilateralFilter(bgr, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        return cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)


def clahe_contrast_enhancement(
    image_rgb: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) in LAB space.
    Operating exclusively on the luminance (L) channel prevents chromatic aberration
    and color shifts on asphalt road textures.
    
    Args:
        image_rgb: RGB input image (H, W, 3) uint8.
        clip_limit: Threshold for contrast limiting. Higher values increase local contrast.
        tile_grid_size: Number of tiles along rows and columns for local equalization.
        
    Returns:
        Contrast-enhanced RGB image.
    """
    # Convert RGB to LAB color space
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to the Luminance (L) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    # Re-merge channels and convert back to RGB
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def sharpen_road_image(
    image_rgb: np.ndarray,
    amount: float = 1.3,
    sigma: float = 2.0
) -> np.ndarray:
    """
    Applies an unsharp mask filter to emphasize high-frequency asphalt defect contours.
    Formula: sharpened = original * (1 + amount) - blurred * amount
    
    Args:
        image_rgb: RGB input image (H, W, 3) uint8.
        amount: Sharpening gain multiplier.
        sigma: Standard deviation of Gaussian blur used for unsharp mask.
        
    Returns:
        Sharpened RGB image.
    """
    blurred = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(
        image_rgb.astype(np.float32), 1.0 + amount,
        blurred.astype(np.float32), -amount,
        0
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_road_opencv(
    image_rgb: np.ndarray,
    apply_denoise: bool = True,
    apply_clahe: bool = True,
    apply_sharpen: bool = True,
    clahe_clip: float = 2.5,
    sharpen_amount: float = 1.3
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Sequential OpenCV Road Enhancement Pipeline:
    Input ➔ Denoise ➔ CLAHE Contrast ➔ Unsharp Sharpen ➔ Output
    
    Args:
        image_rgb: Input RGB road image (H, W, 3).
        apply_denoise: Whether to run denoising stage.
        apply_clahe: Whether to run CLAHE contrast equalization stage.
        apply_sharpen: Whether to run unsharp mask sharpening stage.
        clahe_clip: CLAHE clip limit.
        sharpen_amount: Sharpening intensity.
        
    Returns:
        (enhanced_rgb, latency_ms, telemetry_dict)
    """
    t0 = time.perf_counter()
    current = image_rgb.copy()
    steps_applied = []

    # Stage 1: Denoising
    if apply_denoise:
        current = denoise_road_image(current, method="bilateral")
        steps_applied.append("Bilateral Denoising (d=7, sigma=50)")

    # Stage 2: CLAHE Contrast
    if apply_clahe:
        current = clahe_contrast_enhancement(current, clip_limit=clahe_clip)
        steps_applied.append(f"LAB-CLAHE Contrast (clip={clahe_clip}, grid=8x8)")

    # Stage 3: Sharpening
    if apply_sharpen:
        current = sharpen_road_image(current, amount=sharpen_amount)
        steps_applied.append(f"Unsharp Mask Sharpening (amount={sharpen_amount})")

    latency_ms = (time.perf_counter() - t0) * 1000.0

    telemetry = {
        "pipeline": "OpenCV Classical Road Enhancement",
        "steps_applied": steps_applied,
        "latency_ms": latency_ms,
        "input_resolution": list(image_rgb.shape[:2]),
        "output_resolution": list(current.shape[:2])
    }

    return current, latency_ms, telemetry
