import math
import random
import cv2
import numpy as np
from typing import Optional, Tuple

def generate_motion_blur_kernel(kernel_size: int, angle_degrees: float) -> np.ndarray:
    """
    Creates a linear motion blur convolution kernel of size (kernel_size, kernel_size)
    oriented at angle_degrees.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size - 1) / 2.0

    angle_rad = math.radians(angle_degrees)
    cos_val = math.cos(angle_rad)
    sin_val = math.sin(angle_rad)

    half_len = (kernel_size - 1) / 2.0
    for l in np.linspace(-half_len, half_len, num=kernel_size * 2):
        x = int(round(center + l * cos_val))
        y = int(round(center + l * sin_val))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    else:
        kernel[int(center), int(center)] = 1.0

    return kernel

def apply_motion_blur(
    image: np.ndarray,
    kernel_size: int = 9,
    angle_degrees: Optional[float] = None,
    rng: Optional[random.Random] = None
) -> np.ndarray:
    """
    Applies synthetic linear motion blur to an RGB or grayscale image.
    """
    if angle_degrees is None:
        if rng is not None:
            angle_degrees = rng.uniform(0.0, 180.0)
        else:
            angle_degrees = random.uniform(0.0, 180.0)

    kernel = generate_motion_blur_kernel(kernel_size, angle_degrees)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred
