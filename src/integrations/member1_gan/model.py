"""
Member 1: GAN Model Architecture Specification
================================================
Defines the deep generative network for restoring motion-blurred and degraded
road surface imagery. Member 1 may deploy DeblurGAN-v2 (FPN backbone) or a custom
Generative Adversarial Network generator.
"""

import torch
import torch.nn as nn
from src.models.deblurgan_v2 import DeblurGANv2Generator


class Member1GAN(nn.Module):
    """
    Standard interface wrapper for Member 1's Deblurring GAN Generator.
    
    Default architecture: DeblurGAN-v2 with FPN-MobileNetV2 feature pyramid
    channels = 128 (Edge-optimized, 3.16M params).
    Member 1 can replace or extend this with their custom architecture.
    """
    def __init__(self, fpn_channels: int = 128, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.generator = DeblurGANv2Generator(
            fpn_channels=fpn_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            pretrained_backbone=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 3, H, W) normalized to [-1, 1].
        Returns:
            Tensor of shape (B, 3, H, W) restored road surface normalized to [-1, 1].
        """
        return self.generator(x)
