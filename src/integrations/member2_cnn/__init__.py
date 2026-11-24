"""
Member 2: CNN Pothole Detection & Severity Classification Module
================================================================
Contains model definitions, inference hooks, and checkpoints for Member 2's
pothole detector and severity classifier.
"""

from .detector import PotholeDetector
from .classifier import SeverityClassifier
from .infer import (
    is_detector_available,
    is_severity_available,
    load_member2_models,
    run_member2_pipeline,
    DETECTOR_CHECKPOINT_PATH,
    SEVERITY_CHECKPOINT_PATH
)

__all__ = [
    "PotholeDetector",
    "SeverityClassifier",
    "is_detector_available",
    "is_severity_available",
    "load_member2_models",
    "run_member2_pipeline",
    "DETECTOR_CHECKPOINT_PATH",
    "SEVERITY_CHECKPOINT_PATH"
]
