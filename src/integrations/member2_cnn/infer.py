"""
Member 2: Detection & Severity Inference Engine
===============================================
Coordinates the trained YOLOv8 road damage detection and CNN severity classification pipeline.
Provides seamless execution with actual trained weights and explicit fallback metadata.
"""

import os
import time
from typing import Optional, Dict, Any, List
import numpy as np
import torch
from PIL import Image

from .detector import YOLORoadDamageDetector, PotholeDetector
from .classifier import TrainedSeverityClassifier, SeverityClassifier, SEVERITY_CLASSES

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, "..", "..", ".."))

# Candidate checkpoint search paths (priority order)
DETECTOR_CANDIDATE_PATHS = [
    os.path.join(_DIR, "checkpoints", "best.pt"),
    os.path.join(_ROOT, "best.pt"),
    os.path.join(_DIR, "checkpoints", "detector_best.pt")
]

SEVERITY_CANDIDATE_PATHS = [
    os.path.join(_DIR, "checkpoints", "Severity_classifier.pkl"),
    os.path.join(_ROOT, "Severity_classifier.pkl"),
    os.path.join(_DIR, "checkpoints", "severity_best.pt")
]


def get_detector_path() -> Optional[str]:
    """Finds the first existing valid detector checkpoint file."""
    for p in DETECTOR_CANDIDATE_PATHS:
        if os.path.exists(p) and os.path.getsize(p) > 1024:
            return p
    return None


def get_severity_path() -> Optional[str]:
    """Finds the first existing valid severity classifier checkpoint file."""
    for p in SEVERITY_CANDIDATE_PATHS:
        if os.path.exists(p) and os.path.getsize(p) > 1024:
            return p
    return None


def is_detector_available() -> bool:
    """Returns True if a valid detector checkpoint is found."""
    return get_detector_path() is not None


def is_severity_available() -> bool:
    """Returns True if a valid severity classifier checkpoint is found."""
    return get_severity_path() is not None


# Legacy constants for dashboard compatibility
DETECTOR_CHECKPOINT_PATH = get_detector_path() or DETECTOR_CANDIDATE_PATHS[0]
SEVERITY_CHECKPOINT_PATH = get_severity_path() or SEVERITY_CANDIDATE_PATHS[0]


def load_member2_models(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Loads Member 2's trained YOLO detector and Severity Classifier if available.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    det_path = get_detector_path()
    sev_path = get_severity_path()

    models = {
        "detector": None,
        "classifier": None,
        "detector_available": det_path is not None,
        "severity_available": sev_path is not None,
        "detector_path": det_path,
        "severity_path": sev_path
    }

    # 1. Load YOLO Road Damage Detector
    if det_path is not None:
        try:
            print(f"[Member 2] Loading YOLO detector from: {det_path}")
            det = YOLORoadDamageDetector(det_path)
            models["detector"] = det
            print("[Member 2] YOLO detector loaded successfully.")
        except Exception as e:
            print(f"[Member 2 Detector Load Error]: {e}")

    # 2. Load Severity Classifier
    if sev_path is not None:
        try:
            print(f"[Member 2] Loading Severity Classifier from: {sev_path}")
            clf = TrainedSeverityClassifier(sev_path, device=device)
            models["classifier"] = clf
            print("[Member 2] Severity Classifier loaded successfully.")
        except Exception as e:
            print(f"[Member 2 Severity Classifier Load Error]: {e}")

    return models


def run_member2_pipeline(
    models: Dict[str, Any],
    image_rgb: np.ndarray,
    device: Optional[torch.device] = None,
    conf_thresh: float = 0.15
) -> Dict[str, Any]:
    """
    Runs Member 2's end-to-end detection and severity classification pipeline
    on the input road image.
    
    Returns:
        Dict with detection list, severity tier (Minor/Medium/Major),
        confidence scores, annotated image with bounding boxes, and latency.
    """
    t0 = time.perf_counter()

    result = {
        "detection_status": "ready" if models.get("detector") is not None else "pending",
        "severity_status": "ready" if models.get("classifier") is not None else "pending",
        "detections": [],
        "severity": None,
        "severity_confidence": None,
        "class_probabilities": None,
        "annotated_image": image_rgb.copy(),
        "latency_ms": 0.0,
        "detector_name": "YOLOv8m Road Damage Detector (RDD2022 India)",
        "classifier_name": "CNN Pothole Severity Classifier (Minor/Medium/Major)",
        "pending_message_detector": (
            "Detector weights not found at src/integrations/member2_cnn/checkpoints/best.pt. "
            "Awaiting Member 2 checkpoint."
        ),
        "pending_message_severity": (
            "Severity classifier weights not found at src/integrations/member2_cnn/checkpoints/Severity_classifier.pkl. "
            "Awaiting Member 2 checkpoint."
        )
    }

    detector: Optional[YOLORoadDamageDetector] = models.get("detector")
    classifier: Optional[TrainedSeverityClassifier] = models.get("classifier")

    # 1. Run YOLO Road Damage Detection
    if detector is not None:
        try:
            detections = detector.detect(image_rgb, conf=conf_thresh)
            result["detections"] = detections
            if detections:
                result["annotated_image"] = detector.draw_detections(image_rgb, detections)
        except Exception as e:
            print(f"[Member 2 Detection Runtime Error]: {e}")

    # 2. Run Severity Classification & Crop Extraction
    result["crops"] = []
    if classifier is not None:
        try:
            h, w = image_rgb.shape[:2]
            crops_list = []
            for idx, d in enumerate(result["detections"]):
                x1, y1, x2, y2 = d["box"]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    defect_crop = image_rgb[y1:y2, x1:x2].copy()
                    try:
                        crop_pred = classifier.predict(defect_crop)
                        crops_list.append({
                            "crop_id": idx + 1,
                            "label": d["label"],
                            "confidence": d["confidence"],
                            "box": [x1, y1, x2, y2],
                            "crop_image": defect_crop,
                            "severity": crop_pred["severity"],
                            "severity_confidence": crop_pred["confidence"],
                            "probabilities": crop_pred["probabilities"]
                        })
                    except Exception as ce:
                        print(f"[Crop Classification Error on #{idx+1}]: {ce}")
            result["crops"] = crops_list

            # If a specific pothole bounding box was detected and cropped, prioritize its severity
            pothole_crops = [c for c in crops_list if "pothole" in c["label"].lower()]
            if pothole_crops:
                best_crop = max(pothole_crops, key=lambda x: x["confidence"])
                result["severity"] = best_crop["severity"]
                result["severity_confidence"] = best_crop["severity_confidence"]
                result["class_probabilities"] = best_crop["probabilities"]
            elif crops_list:
                best_crop = max(crops_list, key=lambda x: x["confidence"])
                result["severity"] = best_crop["severity"]
                result["severity_confidence"] = best_crop["severity_confidence"]
                result["class_probabilities"] = best_crop["probabilities"]
            else:
                pred = classifier.predict(image_rgb)
                result["severity"] = pred["severity"]
                result["severity_confidence"] = pred["confidence"]
                result["class_probabilities"] = pred["probabilities"]
                result["raw_class"] = pred.get("raw_class")
        except Exception as e:
            print(f"[Member 2 Severity Runtime Error]: {e}")

    result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
    return result
