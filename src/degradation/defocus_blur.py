import random
import cv2
import numpy as np
from typing import Optional

def generate_disk_kernel(radius: int) -> np.ndarray:
    """
    Creates a circular pillbox (disk) defocus kernel of diameter 2*radius + 1.
    """
    kernel_size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[mask] = 1.0
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel

def apply_defocus_blur(
    image: np.ndarray,
    kernel_size: int = 11,
    sigma: Optional[float] = None,
    mode: str = "gaussian",
    rng: Optional[random.Random] = None
) -> np.ndarray:
    """
    Applies defocus blur to simulate lens defocus or aperture blur.
    mode can be "gaussian" or "disk".
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    if mode == "disk":
        radius = max(1, kernel_size // 2)
        kernel = generate_disk_kernel(radius)
        return cv2.filter2D(image, -1, kernel)
    else:
        # Gaussian blur
        if sigma is None:
            if rng is not None:
                sigma = rng.uniform(1.5, 3.5)
            else:
                sigma = random.uniform(1.5, 3.5)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
