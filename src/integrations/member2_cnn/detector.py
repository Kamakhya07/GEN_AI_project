"""
Member 2: Pothole Detection Model Architecture
==============================================
Defines the deep CNN architecture for localizing distress regions and potholes
on restored road surfaces.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class PotholeDetector(nn.Module):
    """
    Pothole and Road Distress Localization Network.
    
    Architecture:
    Backbone: MobileNetV2 feature extractor
    Detection Head: Multi-scale spatial convolution head predicting bounding box
    coordinates [x1, y1, x2, y2] and pothole confidence score.
    """
    def __init__(self, pretrained: bool = False, num_anchors: int = 4):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = mobilenet_v2(weights=weights)
        self.features = base.features
        
        # Spatial pooling and detection regression/classification head
        self.head = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            # Outputs: [conf, x1, y1, x2, y2] for detected pothole
            nn.Linear(128, 5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 3, H, W) normalized image
        Returns:
            Tensor of shape (B, 5): [pothole_score_logits, x1, y1, x2, y2]
        """
        feats = self.features(x)
        out = self.head(feats)
        return out
