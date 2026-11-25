"""
Member 2: Road Distress & Pothole Severity Classification
=========================================================
Integrates the trained CNN Severity Classifier (Severity_classifier.pkl) pushed by Member 2.
Predicts road damage severity into:
- Minor: Early superficial cracking or shallow surface indentation
- Medium: Moderate pothole / fatigue cracking requiring scheduled maintenance
- Major: Critical deep pothole / base course failure requiring urgent repair
Also retains the standalone MobileNetV2 severity classifier for backward compatibility.
"""

import os
import sys
import types
import pathlib
from typing import Dict, Any, List, Tuple, Union, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Canonical severity classes
SEVERITY_CLASSES = ["Minor", "Medium", "Major"]

# Raw FastAI class vocabulary to clean display names
FASTAI_SEVERITY_MAP = {
    "minor_pothole": "Minor",
    "medium_pothole": "Medium",
    "major_pothole": "Major"
}


def _apply_unpickle_shims():
    """
    Applies cross-platform and cross-Python-version shims before unpickling:
    1. Maps pathlib.PosixPath to pathlib.WindowsPath on Windows systems.
    2. Maps pathlib._local (Python 3.13+) back to pathlib in Python 3.11 environments.
    """
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath

    if "pathlib._local" not in sys.modules:
        m = types.ModuleType("pathlib._local")
        for k, v in pathlib.__dict__.items():
            setattr(m, k, v)
        if os.name == "nt":
            m.PosixPath = pathlib.WindowsPath
        sys.modules["pathlib._local"] = m


class TrainedSeverityClassifier:
    """
    Loads and runs inference with Member 2's trained Severity_classifier.pkl model.
    Extracts the underlying PyTorch neural network for fast, native PyTorch inference.
    """
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Severity classifier weights not found at: {model_path}")

        _apply_unpickle_shims()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model_path = model_path

        from fastai.vision.all import load_learner
        self.learn = load_learner(model_path, cpu=(device.type == "cpu"))
        self.raw_vocab = list(self.learn.dls.vocab) if hasattr(self.learn, "dls") else ["major_pothole", "medium_pothole", "minor_pothole"]
        self.classes = [FASTAI_SEVERITY_MAP.get(c, c) for c in self.raw_vocab]

        # Extract pure PyTorch model to ensure maximum throughput and stability
        self.pytorch_model = self.learn.model.to(self.device)
        self.pytorch_model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Runs severity classification on an image.
        
        Args:
            image_input: File path (str), RGB numpy array (H, W, 3), or PIL Image.
            
        Returns:
            Dict containing:
            - 'severity': 'Minor', 'Medium', or 'Major'
            - 'confidence': float (0.0 to 1.0)
            - 'probabilities': Dict[str, float] for all tiers
            - 'raw_class': original model vocab string
        """
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor_in = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.pytorch_model(tensor_in)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        predicted_severity = self.classes[pred_idx]
        predicted_conf = float(probs[pred_idx])
        raw_pred = self.raw_vocab[pred_idx]

        probs_dict = {
            self.classes[i]: float(probs[i])
            for i in range(len(self.classes))
        }

        return {
            "severity": predicted_severity,
            "confidence": predicted_conf,
            "probabilities": probs_dict,
            "raw_class": raw_pred
        }


class SeverityClassifier(nn.Module):
    """
    Legacy MobileNetV2 Severity Assessment Classifier.
    Preserved for standalone CNN benchmarks.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = False):
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
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
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
