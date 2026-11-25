"""
Official DeblurGAN-v2 Architecture Package
==========================================
Reference:
    Orest Kupyn, Tetiana Martyniuk, Junru Wu, Zhangyang Wang.
    "DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better."
    IEEE/CVF International Conference on Computer Vision (ICCV 2019), pp. 8878-8887.
    Repository: https://github.com/VITA-Group/DeblurGANv2
    License: BSD 2-Clause / BSD 3-Clause (see LICENSE)

Copyright (c) 2019, Orest Kupyn, Tetiana Martyniuk, Junru Wu and Zhangyang Wang
All rights reserved.
"""

from .fpn_mobilenet import FPNMobileNet
from .mobilenet_v2 import MobileNetV2

__all__ = ["FPNMobileNet", "MobileNetV2"]
