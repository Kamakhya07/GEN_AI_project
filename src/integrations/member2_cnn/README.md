# Member 2: CNN Pothole Detection & Severity Classification

Welcome to the **RoadVision AI Member 2 CNN Subsystem**. This folder is completely decoupled so you can independently develop, train, and commit your CNN models without causing merge conflicts or breaking the main dashboard.

---

## 🎯 Role & Objective
- **Subsystem:** Deep CNN for Pothole Detection & Distress Severity Classification.
- **Workflow Position:** Stages 3 & 4:
  `Road Image Upload → GAN Deblurring → **Pothole Detection** → **Severity Classification** → Final Assessment`.
- **Goal:**
  1. Detect and localize potholes and road damage on restored road images.
  2. Grade road distress into severity categories (**Low**, **Moderate**, **Critical**) to drive municipal maintenance dispatch.

---

## 📁 Directory Structure
```
src/integrations/member2_cnn/
├── __init__.py           # Package exports
├── detector.py           # Pothole detection model (localization head)
├── classifier.py         # Severity classification model (Low/Moderate/Critical)
├── infer.py              # Combined inference engine consumed by dashboard
├── README.md             # This guide
├── checkpoints/          # Put your trained weights here
│   ├── .gitkeep
│   ├── detector_best.pt  # <-- TARGET DETECTOR CHECKPOINT
│   └── severity_best.pt  # <-- TARGET SEVERITY CHECKPOINT
└── notebooks/            # Put your Jupyter / Google Colab notebooks here
    └── .gitkeep
```

---

## 🚀 Integration Steps

### Step 1: Model Architectures (`detector.py` & `classifier.py`)
- `PotholeDetector`: MobileNetV2 backbone + localization regression/classification head predicting `[score, x1, y1, x2, y2]`.
- `SeverityClassifier`: MobileNetV2 backbone predicting 3 severity tiers: `["Low", "Moderate", "Critical"]`.
- If you use custom architectures, edit `detector.py` or `classifier.py` accordingly.

### Step 2: Training & Checkpoint Export
Save your trained PyTorch weights to:
1. `src/integrations/member2_cnn/checkpoints/detector_best.pt`
2. `src/integrations/member2_cnn/checkpoints/severity_best.pt`

Save format:
```python
# Detector
torch.save({
    "model_state_dict": detector.state_dict(),
    "epoch": epoch,
    "val_mAP": val_mAP
}, "src/integrations/member2_cnn/checkpoints/detector_best.pt")

# Severity Classifier
torch.save({
    "model_state_dict": classifier.state_dict(),
    "epoch": epoch,
    "val_acc": val_acc
}, "src/integrations/member2_cnn/checkpoints/severity_best.pt")
```

### Step 3: Verify Inference
Run a quick test from the workspace root:
```powershell
python -c "from src.integrations.member2_cnn import is_detector_available, is_severity_available; print('Detector ready:', is_detector_available()); print('Severity ready:', is_severity_available())"
```

### Step 4: Live Dashboard Verification
- When checkpoints are placed in `checkpoints/`, the dashboard's Stage 3 (Pothole Detection) and Stage 4 (Severity Classification) panels will automatically switch from `[Pending Integration: Awaiting Member 2 Checkpoints]` to live inference mode, rendering real bounding boxes and severity metrics!

---

## ⚙️ Specifications
- **Input Image:** $224 \times 224$ RGB image, ImageNet normalized (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Pothole Detection Output:** Bounding box coordinates normalized to `[0.0, 1.0]`.
- **Severity Output:** Softmax probabilities over `["Low", "Moderate", "Critical"]`.
