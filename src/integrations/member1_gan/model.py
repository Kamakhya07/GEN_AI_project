"""
Member 1: Official DeblurGAN-v2 Generator Integration
=====================================================
Integrates the official DeblurGAN-v2 Feature Pyramid Network (FPN) generator with
MobileNetV2 backbone for edge-optimized road deblurring.

Reference:
    Orest Kupyn, Tetiana Martyniuk, Junru Wu, Zhangyang Wang.
    "DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better."
    IEEE/CVF International Conference on Computer Vision (ICCV 2019), pp. 8878-8887.
    Official GitHub: https://github.com/VITA-Group/DeblurGANv2
    License: BSD 2-Clause / BSD 3-Clause

Copyright (c) 2019, Orest Kupyn, Tetiana Martyniuk, Junru Wu and Zhangyang Wang
All rights reserved.
"""

import functools
import torch
import torch.nn as nn
from src.models.deblurgan.fpn_mobilenet import FPNMobileNet


class OfficialDeblurGANv2(nn.Module):
    """
    Official DeblurGAN-v2 Generator (FPN-MobileNetV2).
    Pretrained on GoPro/blur benchmarks by VITA-Group.
    """
    def __init__(self):
        super().__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
        self.model = FPNMobileNet(norm_layer=norm_layer, pretrained=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 3, H, W) normalized to [-1, 1].
        Returns:
            Tensor of shape (B, 3, H, W) restored road surface in [-1, 1].
        """
        return self.model(x)


# Alias for backward compatibility
Member1GAN = OfficialDeblurGANv2
