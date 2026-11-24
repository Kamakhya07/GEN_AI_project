"""
Member 2: CNN Detection & Severity Training Pipeline (Colab / Local Standalone)
==============================================================================
Template training script for Member 2 to train the Pothole Detector and Severity Classifier.
Saves checkpoints to:
- checkpoints/detector_best.pt
- checkpoints/severity_best.pt
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms

import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.integrations.member2_cnn.detector import PotholeDetector
from src.integrations.member2_cnn.classifier import SeverityClassifier, SEVERITY_CLASSES

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DETECTOR_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "detector_best.pt")
SEVERITY_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "severity_best.pt")


def export_starter_models():
    """Demonstrates checkpoint structure for Member 2."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = PotholeDetector(pretrained=False).to(device)
    classifier = SeverityClassifier(num_classes=len(SEVERITY_CLASSES), pretrained=False).to(device)

    print(f"Target paths:")
    print(f"  Detector: {DETECTOR_SAVE_PATH}")
    print(f"  Severity: {SEVERITY_SAVE_PATH}")


if __name__ == "__main__":
    export_starter_models()
