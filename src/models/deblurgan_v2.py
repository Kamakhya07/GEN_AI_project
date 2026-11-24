"""
DeblurGAN-v2 Feature Pyramid Network (FPN) Generator with MobileNetV2 Backbone.

Reference:
    Kupyn, O., Martyniuk, T., Wu, J., & Wang, Z. (2019).
    "DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better."
    IEEE/CVF International Conference on Computer Vision (ICCV 2019), pp. 8878-8887.
    Repository: https://github.com/VITA-Group/DeblurGANv2
    License: MIT License

Architectural Characteristics:
    - Backbone: MobileNetV2 feature extractor extracting multi-scale representations at strides 4, 8, 16, and 32.
    - Top-Down Pathway: Lateral 1x1 convolutions connected via bilinear upsampling and feature summation.
    - Multi-Scale Feature Aggregation: Combines all pyramid levels at stride 4, followed by refinement convolution.
    - Residual Learning: Predicts high-frequency residual correction added to degraded input with Tanh/Clamp activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class ConvBlock(nn.Module):
    """Standard Conv2d + BatchNorm2d + ReLU block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeblurGANv2Generator(nn.Module):
    """
    Feature Pyramid Network (FPN) Generator with MobileNetV2 Backbone for fast edge deblurring.
    Adapted for lightweight real-time highway distress restoration.
    """
    def __init__(self, fpn_channels: int = 128, pretrained_backbone: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained_backbone else None
        base = mobilenet_v2(weights=weights).features

        # MobileNetV2 layer extraction points:
        # Layer 3: Stride 4  (24 channels)
        # Layer 6: Stride 8  (32 channels)
        # Layer 13: Stride 16 (96 channels)
        # Layer 18: Stride 32 (320 channels)
        self.layer1 = nn.Sequential(*base[:4])    # stride 4,  24 channels
        self.layer2 = nn.Sequential(*base[4:7])   # stride 8,  32 channels
        self.layer3 = nn.Sequential(*base[7:14])  # stride 16, 96 channels
        self.layer4 = nn.Sequential(*base[14:18]) # stride 32, 320 channels

        # Lateral 1x1 convolutions
        self.lateral4 = nn.Conv2d(320, fpn_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(96, fpn_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(32, fpn_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(24, fpn_channels, kernel_size=1)

        # Smooth 3x3 convolutions per pyramid level
        self.smooth4 = ConvBlock(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth3 = ConvBlock(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth2 = ConvBlock(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth1 = ConvBlock(fpn_channels, fpn_channels, kernel_size=3, padding=1)

        # Multi-scale aggregation head
        # Concatenates 4 pyramid levels -> 4 * fpn_channels
        self.agg_conv = ConvBlock(fpn_channels * 4, fpn_channels, kernel_size=3, padding=1)

        # Output head: upsamples by 4x back to input resolution and outputs 3-channel residual
        self.upsample4x = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(fpn_channels, fpn_channels // 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(fpn_channels // 2, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual skip from degraded input to restored output.
        x: Normalized image tensor in range [-1.0, 1.0] of shape (B, 3, H, W).
        """
        # Bottom-up pathway through MobileNetV2 backbone
        c1 = self.layer1(x)   # (B, 24, H/4, W/4)
        c2 = self.layer2(c1)  # (B, 32, H/8, W/8)
        c3 = self.layer3(c2)  # (B, 96, H/16, W/16)
        c4 = self.layer4(c3)  # (B, 320, H/32, W/32)

        # Top-down pathway with lateral connections
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, scale_factor=2, mode="bilinear", align_corners=False)
        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=False)
        p1 = self.lateral1(c1) + F.interpolate(p2, scale_factor=2, mode="bilinear", align_corners=False)

        # Smooth pyramid representations
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        # Resample all pyramid feature maps to highest resolution level (H/4, W/4)
        target_size = p1.shape[2:]
        p2_up = F.interpolate(p2, size=target_size, mode="bilinear", align_corners=False)
        p3_up = F.interpolate(p3, size=target_size, mode="bilinear", align_corners=False)
        p4_up = F.interpolate(p4, size=target_size, mode="bilinear", align_corners=False)

        # Concatenate multi-scale representations and aggregate
        fused = torch.cat([p1, p2_up, p3_up, p4_up], dim=1)
        fused = self.agg_conv(fused)

        # Generate residual and add to input
        residual = self.upsample4x(fused)
        restored = torch.clamp(x + residual, -1.0, 1.0)
        return restored


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cpu")
    net = DeblurGANv2Generator(fpn_channels=128, pretrained_backbone=False).to(device)
    dummy_input = torch.randn(2, 3, 128, 128, device=device)
    out = net(dummy_input)
    print("DeblurGAN-v2 FPN-MobileNet Generator initialized successfully!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Trainable parameters: {count_parameters(net):,}")
