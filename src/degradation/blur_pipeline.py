import random
from typing import Optional, Dict, Any, List
import numpy as np
from .motion_blur import apply_motion_blur
from .defocus_blur import apply_defocus_blur

class BlurDegradationPipeline:
    """
    Unified degradation generator supporting:
    - mild_motion (kernel sizes 5, 7, 9)
    - strong_motion (kernel sizes 13, 17, 21)
    - defocus (Gaussian sigma 1.5 - 3.5, disk radius 3-7)
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None, fixed_seed: Optional[int] = None):
        self.config = config or {}
        self.fixed_seed = fixed_seed
        self.rng = random.Random(fixed_seed) if fixed_seed is not None else None

    def sample_degradation(self, blur_type: Optional[str] = None) -> str:
        types = ["mild_motion", "strong_motion", "defocus"]
        if blur_type is not None:
            return blur_type
        if self.rng:
            return self.rng.choice(types)
        return random.choice(types)

    def apply(self, image: np.ndarray, blur_type: Optional[str] = None) -> np.ndarray:
        chosen_type = self.sample_degradation(blur_type)

        if chosen_type == "mild_motion":
            k_sizes = self.config.get("mild_motion", {}).get("kernel_sizes", [5, 7, 9])
            k = self.rng.choice(k_sizes) if self.rng else random.choice(k_sizes)
            return apply_motion_blur(image, kernel_size=k, rng=self.rng)

        elif chosen_type == "strong_motion":
            k_sizes = self.config.get("strong_motion", {}).get("kernel_sizes", [13, 17, 21])
            k = self.rng.choice(k_sizes) if self.rng else random.choice(k_sizes)
            return apply_motion_blur(image, kernel_size=k, rng=self.rng)

        elif chosen_type == "defocus":
            k_sizes = self.config.get("defocus", {}).get("kernel_sizes", [7, 11, 15])
            k = self.rng.choice(k_sizes) if self.rng else random.choice(k_sizes)
            return apply_defocus_blur(image, kernel_size=k, rng=self.rng)

        else:
            return image

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)
