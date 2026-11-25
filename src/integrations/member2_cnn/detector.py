"""
Member 2: Pothole & Road Damage Detection Model Architecture
============================================================
Integrates the trained YOLOv8 road damage detection model pushed by Member 2.
Supports 5 damage classes:
- Longitudinal_Crack (D00)
- Transverse_Crack (D10)
- Alligator_Crack (D20)
- Other_Corruption (D30)
- Pothole (D40)
Also retains the standalone MobileNetV2 spatial detection network for backward compatibility.
"""

import os
from typing import List, Dict, Any, Union, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# 5 Road Damage Classes from YOLOv8 trained model
YOLO_DAMAGE_CLASSES = [
    "Longitudinal_Crack",
    "Transverse_Crack",
    "Alligator_Crack",
    "Other_Corruption",
    "Pothole"
]

# Color map for bounding boxes (BGR format for OpenCV)
CLASS_COLORS = {
    "Pothole": (244, 63, 94),          # Red/Rose
    "Alligator_Crack": (245, 158, 11),  # Orange/Amber
    "Longitudinal_Crack": (56, 189, 248),# Sky Blue
    "Transverse_Crack": (45, 212, 191), # Teal
    "Other_Corruption": (168, 85, 247)  # Purple
}


class YOLORoadDamageDetector:
    """
    YOLOv8 Road Damage & Pothole Detector.
    Wraps the trained best.pt model committed by Member 2.
    """
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model weights not found at: {model_path}")
        from ultralytics import YOLO
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.names = self.model.names if hasattr(self.model, "names") else {i: c for i, c in enumerate(YOLO_DAMAGE_CLASSES)}

    def detect(
        self,
        image_input: Union[str, np.ndarray],
        conf: float = 0.15,
        iou: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Runs YOLO detection on an image path or RGB numpy array.
        
        Args:
            image_input: File path (str) or RGB numpy array (H, W, 3).
            conf: Confidence detection threshold (default 0.15 for high sensitivity).
            iou: NMS IoU threshold.
            
        Returns:
            List of detection dicts:
            [{
                "class_id": int,
                "label": str,
                "confidence": float,
                "box": [x1, y1, x2, y2]
            }]
        """
        # If RGB numpy array, YOLO expects RGB or BGR - passing numpy array directly works
        results = self.model.predict(
            source=image_input,
            conf=conf,
            iou=iou,
            verbose=False
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        if not hasattr(first_res, "boxes") or first_res.boxes is None:
            return detections

        for box in first_res.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            raw_label = self.names.get(cls_id, f"Damage_{cls_id}")
            # Format label cleanly (e.g., 'Longitudinal_Crack' -> 'Longitudinal Crack')
            clean_label = raw_label.replace("_", " ")

            detections.append({
                "class_id": cls_id,
                "raw_label": raw_label,
                "label": clean_label,
                "confidence": confidence,
                "box": [x1, y1, x2, y2]
            })

        return detections

    def draw_detections(
        self,
        image_rgb: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Overlays bounding boxes, category labels, and confidence tags on the image.
        """
        annotated = image_rgb.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            conf = det["confidence"]
            raw_lbl = det.get("raw_label", label)
            color = CLASS_COLORS.get(raw_lbl, CLASS_COLORS.get(label, (45, 212, 191)))

            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw filled header tag
            tag = f"{label}: {conf:.1%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(tag, font, font_scale, thickness)
            
            # Position tag above box, or below if too near the top
            tag_y1 = max(0, y1 - th - 6)
            tag_y2 = y1
            tag_x2 = min(annotated.shape[1], x1 + tw + 6)
            
            cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
            cv2.putText(
                annotated,
                tag,
                (x1 + 3, tag_y2 - 3),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        return annotated


class PotholeDetector(nn.Module):
    """
    Legacy MobileNetV2 Pothole & Distress Localization Network.
    Preserved for standalone CNN benchmarks.
    """
    def __init__(self, pretrained: bool = False, num_anchors: int = 4):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = mobilenet_v2(weights=weights)
        self.features = base.features
        
        self.head = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        out = self.head(feats)
        return out
