"""
Member 2: Detection & Severity Inference Engine
===============================================
Coordinates the pothole detection and severity classification pipeline.
Provides explicit pending-integration state tracking when checkpoints are awaiting.
"""

import os
import time
from typing import Optional, Dict, Any, List
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from .detector import PotholeDetector
from .classifier import SeverityClassifier, SEVERITY_CLASSES

_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_CHECKPOINT_PATH = os.path.join(_DIR, "checkpoints", "detector_best.pt")
SEVERITY_CHECKPOINT_PATH = os.path.join(_DIR, "checkpoints", "severity_best.pt")


def is_detector_available() -> bool:
    """Returns True if Member 2 has placed detector_best.pt in checkpoints/."""
    return os.path.exists(DETECTOR_CHECKPOINT_PATH) and os.path.getsize(DETECTOR_CHECKPOINT_PATH) > 1024


def is_severity_available() -> bool:
    """Returns True if Member 2 has placed severity_best.pt in checkpoints/."""
    return os.path.exists(SEVERITY_CHECKPOINT_PATH) and os.path.getsize(SEVERITY_CHECKPOINT_PATH) > 1024


def load_member2_models(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Loads Member 2's detector and severity classifier if available.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {
        "detector": None,
        "classifier": None,
        "detector_available": is_detector_available(),
        "severity_available": is_severity_available(),
    }

    if models["detector_available"]:
        try:
            det = PotholeDetector()
            ckpt = torch.load(DETECTOR_CHECKPOINT_PATH, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                det.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                det.load_state_dict(ckpt["state_dict"])
            else:
                det.load_state_dict(ckpt)
            det.to(device)
            det.eval()
            models["detector"] = det
        except Exception as e:
            print(f"[Member 2 Detector Load Error]: {e}")

    if models["severity_available"]:
        try:
            clf = SeverityClassifier(num_classes=len(SEVERITY_CLASSES))
            ckpt = torch.load(SEVERITY_CHECKPOINT_PATH, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                clf.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                clf.load_state_dict(ckpt["state_dict"])
            else:
                clf.load_state_dict(ckpt)
            clf.to(device)
            clf.eval()
            models["classifier"] = clf
        except Exception as e:
            print(f"[Member 2 Severity Classifier Load Error]: {e}")

    return models


def run_member2_pipeline(
    models: Dict[str, Any],
    image_rgb: np.ndarray,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Runs Member 2's detection and severity classification pipeline.
    
    If models are not yet integrated, returns explicit pending integration metadata.
    Does NOT return fake or simulated predictions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = {
        "detection_status": "pending" if models.get("detector") is None else "ready",
        "severity_status": "pending" if models.get("classifier") is None else "ready",
        "detections": [],
        "severity": None,
        "severity_confidence": None,
        "class_probabilities": None,
        "latency_ms": 0.0,
        "pending_message_detector": (
            "Detector weights not found at src/integrations/member2_cnn/checkpoints/detector_best.pt. "
            "Awaiting Member 2 checkpoint commit."
        ),
        "pending_message_severity": (
            "Severity classifier weights not found at src/integrations/member2_cnn/checkpoints/severity_best.pt. "
            "Awaiting Member 2 checkpoint commit."
        )
    }

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pil_img = Image.fromarray(image_rgb)
    t_in = tf(pil_img).unsqueeze(0).to(device)

    t0 = time.perf_counter()

    # 1. Real Detector Inference (if loaded)
    if models.get("detector") is not None:
        with torch.no_grad():
            out = models["detector"](t_in)  # shape (1, 5): [score, x1, y1, x2, y2]
            score = torch.sigmoid(out[0, 0]).item()
            if score > 0.40:
                h, w = image_rgb.shape[:2]
                box = out[0, 1:].clamp(0.0, 1.0).cpu().numpy()
                x1 = int(box[0] * w)
                y1 = int(box[1] * h)
                x2 = int(box[2] * w)
                y2 = int(box[3] * h)
                result["detections"].append({
                    "label": "Pothole",
                    "confidence": float(score),
                    "box": [x1, y1, x2, y2]
                })

    # 2. Real Severity Classifier Inference (if loaded)
    if models.get("classifier") is not None:
        with torch.no_grad():
            logits = models["classifier"](t_in)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))
            result["severity"] = SEVERITY_CLASSES[pred_idx]
            result["severity_confidence"] = float(probs[pred_idx])
            result["class_probabilities"] = {
                SEVERITY_CLASSES[i]: float(probs[i]) for i in range(len(SEVERITY_CLASSES))
            }

    result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
    return result
