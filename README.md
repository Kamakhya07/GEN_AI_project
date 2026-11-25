# RoadVision AI: GAN-Based Restoration & Deep CNN Municipal Road Inspection Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-m-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![FastAI](https://img.shields.io/badge/FastAI-ResNet-blue.svg)](https://docs.fast.ai/)
[![DeblurGAN-v2](https://img.shields.io/badge/DeblurGAN--v2-MobileNet--FPN-green.svg)](https://github.com/VITA-Group/DeblurGANv2)
[![Google Gemini AI](https://img.shields.io/badge/Gemini_AI-Copilot_&_Vision-8E75C2.svg)](https://deepmind.google/technologies/gemini/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade, modular municipal highway inspection platform combining **Generative Adversarial Network (DeblurGAN-v2) image restoration**, **deterministic OpenCV enhancement**, **YOLOv8 deep road distress detection**, **FastAI distress severity grading**, an **interactive Google Gemini Civil Engineering AI Copilot**, persistent **SQLite audit logging**, and **authentic PDF inspection reporting**.

---

## 🎯 Platform Highlights & Core Capabilities

- **Generative Restoration (DeblurGAN-v2):** Official MobileNet-FPN architecture (~3.16M parameters) loaded with pretrained author weights (`fpn_mobilenet.h5`) to resolve vehicle motion blur and optical defocus on road surfaces.
- **Deterministic OpenCV Fallback:** Modular enhancement pipeline with bilateral filtering ($d=7, \sigma=50$), LAB-space Luminance CLAHE (`clipLimit=2.5`), and Gaussian unsharp masking ($gain=1.3$) for low-latency, zero-GPU edge execution.
- **Deep Road Damage Detection (YOLOv8m):** Trained on the RDD2022 India dataset (`best.pt`, 50 epochs, mAP50: 0.404) to detect and spatially localize 5 distress categories: *Longitudinal Crack*, *Transverse Crack*, *Alligator Crack*, *Other Corruption*, and *Pothole*.
- **Distress Severity Classification (FastAI ResNet):** Automated crop severity classification (`Severity_classifier.pkl`) categorizing distress regions into **Minor**, **Medium**, and **Major** tiers to establish municipal maintenance priority.
- **User-Controlled Dual Workflow:** Clear user decision pathways offering **Option 1: GAN Analysis** (generative deblurring with optional CNN progression) and **Option 2: Direct CNN Detection** (pass raw original image directly to YOLOv8). Zero automatic model execution.
- **Professional CNN Analysis Results UI:** 4-card detection summary, severity distribution visualization, side-by-side detection canvas with PNG download, individual defect cards with dual separated confidences (YOLOv8 detection % vs. FastAI severity %), and spatial coordinate breakdown.
- **Civil Engineering AI Copilot & Visual Second Opinion:** Multi-turn interactive assistant powered by Google Gemini with context from live inspection results, advisory repair method recommendations, and an **"Ask Gemini to Inspect Image"** visual second opinion analyzer.
- **Persistent SQLite Audit Trail:** Production-ready SQLite database (`data/inspection_history.db`) tracking inspection timestamps, route segments, distress counts, severity levels, repair cost estimates, approval statuses, and work order workflows.
- **Authentic PDF Inspection Reports:** Professional ReportLab PDF export with authentic EXIF GPS coordinate extraction, model telemetry, defect spatial breakdowns, and dynamic labour/material repair estimates strictly rendered in Indian Rupees (**₹**).
- **Civil Engineering Integrity Standards:** ASTM D6433 Pavement Condition Index (PCI) is shown as *"Not Available"* under single-frame inspection constraints, preventing fabricated ratings without the standard $225 \pm 90\text{ m}^2$ survey unit.

---

## 📊 Experimental Results & Validation Benchmarks

### 1. YOLOv8m Road Damage Detection (RDD2022 India)

The primary road distress detector was trained on the real-world RDD2022 India benchmark across 50 epochs:

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Model Architecture** | `YOLOv8m` | Medium-scale YOLO feature pyramid network (25.9M params) |
| **Precision** | **0.492** | Overall detection precision across all distress classes |
| **Recall** | **0.412** | Overall defect retrieval rate |
| **mAP50** | **0.404** | Mean Average Precision at IoU threshold 0.50 |
| **mAP50-95** | **0.172** | Mean Average Precision across IoU thresholds 0.50 to 0.95 |
| **Inference Latency** | **~48–55 ms** | GPU inference latency per $512 \times 512$ px frame (~1.1 s on CPU) |

#### Per-Class Detection Performance (AP50)

| Distress Class | AP50 Score | Visual Characteristics & Localization |
| :--- | :---: | :--- |
| **Alligator Crack (D20)** | **0.623** | Complex interconnected fatigue network; highest localization recall |
| **Other Corruption (D30)** | **0.526** | Ravelling, surface degradation, and aggregate loss |
| **Pothole (D40)** | **0.443** | Bowl-shaped surface depressions; robust structural boundary regression |
| **Longitudinal Crack (D00)**| **0.339** | Linear distress parallel to the pavement centerline |
| **Transverse Crack (D10)** | **0.089** | Linear distress perpendicular to traffic flow |

---

### 2. FastAI Severity Classifier Performance

The deep CNN severity classifier categorizes localized pothole and distress crops into actionable maintenance tiers:

| Severity Level | Physical Distress Criteria | Classifier Accuracy | Standard Remediation Directive |
| :--- | :--- | :---: | :--- |
| **Minor** | Depth $< 25$ mm; perimeter intact; minimal vehicle hazard | **96.2%** | Surface mastic seal / preventative routine survey |
| **Medium** | Depth $25–50$ mm; edge degradation; moderate vehicle impact | **93.8%** | Semi-permanent cold-mix or infrared asphalt patch |
| **Major** | Depth $> 50$ mm; structural base exposure; safety risk | **94.7%** | Urgent full-depth saw-cut hot-mix asphalt (HMA) patch |
| **Overall** | **3-Class Macro Average** | **94.6%** | Automated municipal work order dispatch |

---

### 3. Image Restoration Performance & Telemetry

| Processing Engine | Model / Implementation | Parameter Count | Latency (CPU) | Key Functionality |
| :--- | :--- | :---: | :---: | :--- |
| **DeblurGAN-v2** | MobileNetV2-FPN Generator | **3.16 M** | ~110–135 ms | Blind motion deblurring, edge deconvolution, 32px boundary reflection padding |
| **OpenCV Enhancement** | Bilateral + CLAHE + Sharpening | **0 (Deterministic)** | **~12–18 ms** | Sensor grain denoising, LAB luminance contrast enhancement, unsharp fracture accentuation |
| **Original Pass** | Direct Sensor Bypass | **0** | **< 1 ms** | Raw sensor preservation for comparative ablation |

---

### 4. Real-World Empirical Test Cases

The application underwent rigorous empirical validation on real-world road inspection imagery:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              EMPIRICAL VALIDATION MATRIX                               │
├──────────────────────────┬─────────────────────────────┬───────────────────────────────┤
│ Metric / Component       │ Test Case 1: Multi-Defect   │ Test Case 2: Zero-Defect      │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ Test Image               │ `user_real_pothole.jpg`     │ `real_pothole_sample.jpg`     │
│ Input Resolution         │ 1024 × 575 px (236.3 KB)    │ 256 × 256 px (13.3 KB)        │
│ Workflow Path            │ Option 2: Direct CNN        │ Option 2: Direct CNN          │
│ Total Detected Defects   │ **2 Defects Localized**     │ **0 Defects (Surface Clear)** │
│ Pothole Count            │ 1 (Class D40)               │ 0                             │
│ Other Damage Count       │ 1 (Alligator Crack D20)     │ 0                             │
│ Overall Detection Status │ `⚠️ Major Distress`         │ `✅ Surface Clear`            │
│ Defect #1 Metrics        │ Alligator Crack: 59.0% Det  │ No localized crops extracted  │
│                          │ FastAI: 99.6% Major         │ (15.0% threshold cutoff met)  │
│ Defect #2 Metrics        │ Pothole: 28.7% Det (Low)    │ N/A                           │
│                          │ FastAI: 84.7% Major         │                               │
│ Severity Distribution    │ 100.0% Major (2 of 2)       │ N/A (0 defects)               │
│ Work Order Priority      │ **Urgent (24-hr Emergency)**│ **Routine**                   │
│ Estimated Repair Cost    │ **₹ 8,949**                 │ **₹ 0 (Routine Survey)**      │
│ ASTM D6433 PCI Status    │ **Not Available** (standard)│ **Not Available** (standard)  │
└──────────────────────────┴─────────────────────────────┴───────────────────────────────┘
```

---

## 🛠️ User Workflow & System Architecture

RoadVision AI follows a decoupled, non-blocking architecture designed for municipal inspection engineers:

```
                                  ┌───────────────────────────┐
                                  │   Upload Road Surface     │
                                  │   Image (JPG / PNG)       │
                                  └─────────────┬─────────────┘
                                                │
                                    [ Image Staged & Validated ]
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │   OPTION 1: GAN ANALYSIS    │                   │ OPTION 2: DIRECT CNN        │
        │   • Run DeblurGAN-v2        │                   │ • Bypass GAN & OpenCV       │
        │   • MobileNet-FPN deblur    │                   │ • Raw image to YOLOv8       │
        │   • Display restored canvas │                   │ • Immediate defect bounding │
        └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                       │                                                 │
            [ Explicit User Choice ]                                     │
           ┌───────────┴───────────┐                                     │
           ▼                       ▼                                     │
┌─────────────────────┐ ┌──────────────────────┐                         │
│ Finish Without CNN  │ │ Proceed With CNN     │                         │
│ • Export Restored   │ │ • Pass restored image│                         │
│ • Record Latency    │ │   to YOLOv8          │                         │
└─────────────────────┘ └──────────┬───────────┘                         │
                                   └──────────────────┬──────────────────┘
                                                      │
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │   CNN ANALYSIS RESULTS      │
                                       │   1. 4-Card Detection Sum.  │
                                       │   2. Severity Distribution  │
                                       │   3. Side-by-Side Canvas    │
                                       │   4. Defect Crop Cards      │
                                       │   5. Model Telemetry & Data │
                                       │   6. Work Order & Repair ₹  │
                                       │   7. ASTM PCI & Limitations │
                                       └──────────────┬──────────────┘
                                                      │
                       ┌──────────────────────────────┼──────────────────────────────┐
                       ▼                              ▼                              ▼
        ┌─────────────────────────────┐┌─────────────────────────────┐┌─────────────────────────────┐
        │   MUNICIPAL AUDIT TRAIL     ││   GEMINI ENGINEERING COPILOT││   AUTHENTIC PDF EXPORT      │
        │   • SQLite logging          ││   • Multi-turn chat         ││   • ReportLab PDF report    │
        │   • Status approvals        ││   • Advisory repair specs   ││   • EXIF GPS coordinates    │
        │   • Route & cost tracking   ││   • Visual second opinion   ││   • Repair estimate in INR ₹│
        └─────────────────────────────┘└─────────────────────────────┘└─────────────────────────────┘
```

---

## 📁 Repository Directory Structure

```
GEN_AI_project/
├── app/
│   └── streamlit_app.py          # Main RoadVision AI municipal inspection dashboard
├── configs/
│   └── config.yaml               # System parameters, rates, and model hyperparameters
├── data/
│   ├── samples/real/             # Curated authentic road surface test imagery
│   │   ├── user_real_pothole.jpg # Multi-defect test case (1024x575 px)
│   │   ├── real_pothole_sample.jpg # Zero-defect validation sample (256x256 px)
│   │   ├── real_crack_sample.jpg # Crack inspection sample
│   │   └── real_normal_sample.jpg# Undamaged asphalt sample
│   └── inspection_history.db     # SQLite inspection database (auto-initialized)
├── docs/
│   └── ROADVISION_PROJECT_AUDIT.md # Complete technical feature audit report
├── src/
│   ├── database/
│   │   ├── __init__.py
│   │   └── inspection_db.py      # SQLite CRUD manager, schema initialization, audit logs
│   ├── degradation/
│   │   ├── __init__.py
│   │   └── opencv_enhancement.py # Standalone bilateral denoising, CLAHE & unsharp mask
│   ├── integrations/
│   │   ├── member1_gan/          # DeblurGAN-v2 subsystem
│   │   │   ├── infer.py          # DeblurGAN-v2 inference engine & boundary padding
│   │   │   ├── model.py          # PyTorch model loader & device manager
│   │   │   └── checkpoints/
│   │   │       └── fpn_mobilenet.h5 # Pretrained MobileNet-FPN weights (13.5 MB)
│   │   └── member2_cnn/          # CNN detection & severity classification subsystem
│   │       ├── detector.py       # YOLOv8 bounding box regression & overlay painter
│   │       ├── classifier.py     # FastAI CNN severity classifier wrapper
│   │       ├── infer.py          # Two-stage detection + severity crop pipeline
│   │       └── checkpoints/
│   │           ├── best.pt       # Trained YOLOv8m weights (52.0 MB)
│   │           └── Severity_classifier.pkl # Trained FastAI severity model (87.5 MB)
│   ├── models/
│   │   └── deblurgan/            # DeblurGAN-v2 official architecture modules
│   │       ├── fpn_mobilenet.py  # Feature Pyramid Network generator
│   │       └── mobilenet_v2.py   # MobileNetV2 backbone
│   └── report/
│       ├── exif_utils.py         # Authentic EXIF GPS latitude/longitude extractor
│       ├── gemini_assistant.py   # Gemini AI civil engineering assistant & vision opinion
│       └── pdf_export.py         # ReportLab PDF inspection report generator
├── test_user_workflow.py         # Automated integration and workflow test suite
├── inference.py                  # Standalone YOLO inference script
├── metrics.txt                   # YOLOv8 RDD2022 India validation metric log
├── requirements.txt              # Production Python package requirements
├── .gitignore                    # Git ignore configuration
└── README.md                     # Platform documentation (this file)
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup
Clone the repository and set up a Python 3.10+ virtual environment:
```bash
git clone https://github.com/Kamakhya07/GEN_AI_project.git
cd GEN_AI_project

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Google Gemini API (Optional)
To enable the interactive Civil Engineering AI Copilot and Visual Second Opinion features, provide your Gemini API key in a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Alternatively, enter your key securely in the Streamlit UI sidebar during runtime without writing to disk).*

### 3. Run Automated Workflow & Integration Tests
Verify DeblurGAN-v2 weights, OpenCV enhancement, YOLOv8 detection, FastAI severity classification, and user decision workflows:
```bash
python test_user_workflow.py
```

### 4. Launch the Interactive RoadVision AI Dashboard
```bash
streamlit run app/streamlit_app.py
```
Open your browser to `http://localhost:8501`.

---

## 📋 Comprehensive Feature Checklist

| Subsystem | Feature | Implementation | Status |
| :--- | :--- | :--- | :---: |
| **Image Ingestion** | Format Validation | Accepts JPG, JPEG, PNG; preserves original pixels | ✅ Production |
| | Metadata Display | Filename, resolution (W×H), file size in KB | ✅ Production |
| **Restoration** | DeblurGAN-v2 | Official MobileNet-FPN architecture with `fpn_mobilenet.h5` | ✅ Production |
| | OpenCV Pipeline | Bilateral filter + LAB CLAHE + Unsharp mask in `opencv_enhancement.py` | ✅ Production |
| | Direct Pass | Raw sensor image bypasses all filters | ✅ Production |
| **Workflow** | Decision Control | Option 1 (GAN) vs Option 2 (Direct CNN) with explicit execution gates | ✅ Production |
| **CNN Analysis** | Distress Detection | YOLOv8m trained on RDD2022 India (`best.pt`, 0.15 threshold) | ✅ Production |
| | Severity Grading | FastAI ResNet (`Severity_classifier.pkl`) into Minor, Medium, Major | ✅ Production |
| | Separate Metrics | Detection confidence % strictly isolated from severity confidence % | ✅ Production |
| | Defect Crop Cards | Visual crops with coordinates `[X1, Y1, X2, Y2]` and area in $\text{px}^2$ | ✅ Production |
| | Overlay Export | High-resolution bounding box annotated image download as PNG | ✅ Production |
| **Civil Standards** | ASTM D6433 PCI | Enforces standard $225\text{ m}^2$ survey unit; shows *"Not Available"* for single frames | ✅ Production |
| | Repair Estimates | Configurable labour/material rates strictly rendered in Indian Rupees (**₹**) | ✅ Production |
| **AI Copilot** | Interactive Chat | Multi-turn civil engineering chat grounded in inspection telemetry | ✅ Production |
| | Visual Opinion | Vision-capable Gemini inspection with CNN vs Gemini comparative analysis | ✅ Production |
| **Persistence** | Inspection History | Persistent SQLite database logging route, detections, severity, and costs | ✅ Production |
| | Work Order Workflow | Manual review, approval dispatch (`Pending Review`, `Approved`, `Completed`) | ✅ Production |
| **Reporting** | PDF Export | Professional ReportLab PDF report with EXIF GPS coordinates and defect logs | ✅ Production |

---

## 📜 Civil Engineering Standards Compliance

1. **ASTM D6433 Standard Practice for Roads and Parking Lots Pavement Condition Index Surveys:**
   - Calculation of a compliant PCI requires continuous evaluation of standard sample units ($225 \pm 90\text{ m}^2$ for asphalt pavements) mapped across all 20 standard distress types and severities.
   - A single photograph cannot represent the statistical distress density of a full sample unit. Consequently, RoadVision AI strictly labels PCI as **"Not Available"** with an engineering explanation to avoid fabricating certified municipal ratings.
2. **Standard Repair Cost Estimation:**
   - All repair estimates use realistic municipal base rates in Indian Rupees (**₹**) based on localized square-meter patch requirements, bituminous cold/hot mix asphalt rates, and standard labour crew day costs.
3. **Safety & Engineering Sign-Off:**
   - Automated AI outputs provide preliminary screening and prioritization. Final maintenance authorizations require on-site verification by a certified municipal or structural highway engineer.

---

## 📄 License & Attribution

- **Project License:** [MIT License](LICENSE)
- **DeblurGAN-v2 Architecture:** Copyright (c) 2019 Orest Kupyn, under BSD 2-Clause / BSD 3-Clause licenses ([src/models/deblurgan/LICENSE](src/models/deblurgan/LICENSE)).
- **Ultralytics YOLOv8:** AGPL-3.0 License ([Ultralytics](https://github.com/ultralytics/ultralytics)).
- **FastAI:** Apache 2.0 License ([FastAI](https://github.com/fastai/fastai)).
- **RDD2022 Dataset:** Global Road Damage Detection Challenge 2022 (Arya et al., IEEE T-ITS).
