# Member 2: CNN Road Damage Detection & Severity Classification

Welcome to the **RoadVision AI Member 2 CNN Subsystem**. This module houses Member 2's trained computer vision models for road damage detection and pothole severity classification.

---

## 🎯 Role & Objective
- **Subsystem:** Deep CNN & YOLO for Road Damage Detection & Distress Severity Assessment.
- **Workflow Position:** Stages 3 & 4:
  `Road Image Upload → GAN Deblurring → **Pothole Detection** → **Severity Classification** → Final Assessment`.
- **Goal:**
  1. Detect and localize 5 road damage categories (`Longitudinal_Crack`, `Transverse_Crack`, `Alligator_Crack`, `Other_Corruption`, `Pothole`) with spatial bounding boxes.
  2. Grade road distress into severity categories (**Minor**, **Medium**, **Major**) to drive municipal maintenance dispatch.

---

## 📁 Integrated Models & Artifacts

| Component | File Path | Model Family | Description | Classes |
| :--- | :--- | :--- | :--- | :--- |
| **Road Damage Detector** | `checkpoints/best.pt` (or `best.pt`) | YOLOv8m | Trained on RDD2022 India (50 epochs) | Longitudinal Crack, Transverse Crack, Alligator Crack, Other Corruption, Pothole |
| **Severity Classifier** | `checkpoints/Severity_classifier.pkl` (or `Severity_classifier.pkl`) | FastAI / PyTorch CNN | Deep vision severity model | `['minor_pothole', 'medium_pothole', 'major_pothole']` $\rightarrow$ `['Minor', 'Medium', 'Major']` |
| **Validation Metrics** | `metrics.txt` | Text Report | Precision: 0.492, Recall: 0.412, mAP50: 0.404 | Per-class AP: Alligator Crack (0.623), Pothole (0.443) |

---

## 📁 Directory Structure
```
src/integrations/member2_cnn/
├── __init__.py           # Package exports (YOLORoadDamageDetector, TrainedSeverityClassifier, etc.)
├── detector.py           # YOLOv8 road damage detection wrapper + bounding box renderer
├── classifier.py         # CNN Severity Classifier wrapper with cross-platform unpickler
├── infer.py              # End-to-end inference engine consumed by Streamlit dashboard
├── README.md             # This documentation
├── checkpoints/          # Trained weights directory
│   ├── .gitkeep
│   ├── best.pt                   # YOLOv8m road damage model (52 MB)
│   └── Severity_classifier.pkl   # Severity classifier CNN model (87.5 MB)
└── notebooks/            # Training scripts & notebooks
    ├── .gitkeep
    └── train_cnn_starter.py
```

---

## 🚀 Running Verification

You can verify both models from Python or terminal:

```powershell
python -c "from src.integrations.member2_cnn import is_detector_available, is_severity_available, load_member2_models; print('Detector:', is_detector_available()); print('Severity:', is_severity_available()); m = load_member2_models(); print('Loaded successfully:', m['detector'] is not None and m['classifier'] is not None)"
```

The Streamlit dashboard (`app/streamlit_app.py`) automatically connects image uploads to these models when **"Member 2 CNN Module"** is selected in the sidebar, displaying live bounding boxes, confidence tags, and severity bars.
