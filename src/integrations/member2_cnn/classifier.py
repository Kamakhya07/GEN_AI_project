"""
Member 2: Severity Classification Model Architecture
====================================================
Defines the deep CNN architecture for classifying road distress severity:
Low, Moderate, and Critical hazard levels.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


SEVERITY_CLASSES = ["Low", "Moderate", "Critical"]


class SeverityClassifier(nn.Module):
    """
    Road Distress Severity Assessment Classifier.
    
    Predicts hazard tier:
    0: Low Severity (Surface discoloration, micro-fissures)
    1: Moderate Severity (Longitudinal / Transverse fatigue cracking)
    2: Critical Hazard (Deep potholes, alligator web failure)
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = False):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = mobilenet_v2(weights=weights)
        self.features = base.features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base.last_channel, 128),
            nn.ReLU6(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 3, H, W) normalized image
        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
