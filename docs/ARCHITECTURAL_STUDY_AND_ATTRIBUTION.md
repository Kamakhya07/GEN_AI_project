# Architectural Study, Comparative Analysis, and Open-Source Attribution

This document provides a comprehensive technical study of the three reference repositories identified for the **RoadVision AI** research project, detailing their architectures, preprocessing pipelines, model implementations, evaluation methodologies, and our project's reuse and attribution policy.

---

## 1. Reference Repositories Technical Study

### 1.1 VITA-Group / DeblurGANv2
* **Repository:** [https://github.com/VITA-Group/DeblurGANv2](https://github.com/VITA-Group/DeblurGANv2)
* **Citation:** Kupyn, O., Martyniuk, T., Wu, J., & Wang, Z. (2019). *"DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better."* IEEE/CVF International Conference on Computer Vision (ICCV 2019), pp. 8878–8887.
* **License:** MIT License.

#### Architecture
* **Generator (FPN-Based):** Unlike classical symmetric U-Net or coarse-to-fine multi-scale networks, DeblurGAN-v2 introduces a **Feature Pyramid Network (FPN)** generator. It connects a backbone (e.g., MobileNet or Inception-ResNet-v2) to lateral $1\times 1$ convolutions, creates top-down feature fusion pathways, and merges multi-scale pyramid representations at stride 4.
* **MobileNet-FPN Variant:** Designed for real-time mobile/edge deblurring, reducing parameters from $>16\text{M}$ (U-Net) to $\sim 3.16\text{M}$ while maintaining structural restoration capability.
* **Discriminator:** Relativistic conditional GAN (RaGAN) using a **double-scale discriminator** (global scale for overall coherence and local patch scale for fine boundary sharpness).
* **Loss Function Formulation:**
  $$\mathcal{L}_{G} = \mathcal{L}_{adv\_Ra} + 0.5 \cdot \mathcal{L}_{MSE} + 0.006 \cdot \mathcal{L}_{perceptual\_VGG19}$$

#### Preprocessing & Degradation
* Dynamic camera trajectory simulation: Generates linear motion blur kernels with varying displacement lengths and directional angles, as well as optical defocus blur.
* Normalization: Images scaled to $[-1.0, 1.0]$ for stable adversarial convergence.

#### Evaluation
* Evaluates peak restoration metrics: **PSNR (dB)** and **SSIM** across standardized benchmarks (GoPro, Kohler).

---

### 1.2 sekilab / RoadDamageDetector (RDD2022)
* **Repository:** [https://github.com/sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector)
* **Citation:** Maeda, H., Sekimoto, Y., Seto, T., Kashiyama, T., & Omata, H. (2018). *"Road damage detection and classification using deep neural networks with smartphone images."* Computer-Aided Civil and Infrastructure Engineering, 33(12), 1127–1141.
* **Dataset:** RDD2022 (Crowdsensing-based Road Damage Detection Challenge 2022).
* **License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

#### Data Taxonomy & Structure
* **Scope:** 47,420 road surface survey images captured across 6 countries (Japan, India, Czech Republic, Norway, United States, China) via vehicle-mounted dashcams and smartphones driving at 20–50 km/h.
* **Standard Distress Classes:**
  - `D00`: Wheel mark part longitudinal crack
  - `D10`: Equal interval transverse crack
  - `D20`: Partial / Alligator fatigue cracking
  - `D40`: Pothole / rutting / surface depression
* **Annotation Format:** Pascal VOC XML (`<annotation>`, `<bndbox>` with `xmin, ymin, xmax, ymax`).

#### Evaluation Protocol
* Detection evaluated via **Mean F1-score ($\text{mF1}$)** at $\text{IoU} = 0.50$ across multi-national domains to test geographical generalization.

---

### 1.3 thanhmaitran / road-damage-detection-platform
* **Repository Reference:** Architectural paradigm for end-to-end road inspection web deployment and inspection telemetry.
* **License:** MIT / Apache 2.0 open-source paradigm.

#### Architecture & Capabilities
* **Interactive Inspection Console:** Multi-input stream architecture supporting dashcam video streams, sample presets, and technician uploads.
* **Inference Pipeline:** Preprocessing $\rightarrow$ Model Inference $\rightarrow$ Post-Processing $\rightarrow$ Civil Engineering Dispatch Action.
* **Telemetry & Report Generation:** Real-time latency tracking (generator latency, classifier latency, total budget) and structured JSON inspection report exports.

---

## 2. Component Lineage in RoadVision AI

To maintain academic honesty, reproducibility, and legal compliance, the table below documents the lineage of every module in the RoadVision AI project:

| Module / File | Origin / Reference | Reused, Modified, or Independent | Attribution & Licensing Notes |
|---|---|---|---|
| `src/models/deblurgan_v2.py` | `VITA-Group/DeblurGANv2` (Kupyn et al., ICCV 2019) | **Modified / Adapted** | Cleanly implemented the FPN-MobileNetV2 generator architecture (3.16M params) adapted for $128\times 128$ and $256\times 256$ distress crops. MIT License respected. |
| `src/models/generator.py` | Isola et al. (Pix2Pix U-Net) | **Modified / Adapted** | Classical U-Net generator with 8 down/up blocks and skip connections for comparative ablation. |
| `src/models/discriminator.py` | PatchGAN (Isola et al.) | **Modified / Adapted** | $70\times 70$ receptive field Markovian discriminator. |
| `src/models/mobilenet.py` | Sandler et al. (MobileNetV2) | **Modified / Adapted** | Fine-tuned MobileNetV2 downstream classifier for 5 road distress classes. |
| `src/data/rdd_parser.py` | `sekilab/RoadDamageDetector` | **Modified / Adapted** | Custom XML parser for RDD2022 Pascal VOC annotations mapping D00, D10, D20, D40 to civil damage taxonomy. CC BY-SA 4.0 credited. |
| `src/data/patch_extractor.py` | Seki Lab RDD extraction | **Independently Implemented** | Context-padded ($20\%$) bounding-box patch crop generator with negative background road sampling. |
| `src/data/split_manager.py` | Best practice research protocol | **Independently Implemented** | Source-image-level partitioner guaranteeing zero data leakage between train/val/test splits. |
| `src/degradation/motion_blur.py` | Optical physics of linear motion | **Independently Implemented** | Point-spread-function (PSF) kernel convolution modeling vehicle velocity and trajectory angle. |
| `src/degradation/defocus_blur.py` | Gaussian optics | **Independently Implemented** | Gaussian / disc blur kernel modeling camera aperture defocus. |
| `src/degradation/classical_restoration.py` | Classical image processing | **Independently Implemented** | Unsharp masking and Wiener deconvolution baselines. |
| `src/evaluate/metrics.py` | Scikit-learn / IEEE standards | **Independently Implemented** | Precision, recall, macro-F1, confusion matrices, PSNR, and float64-bounded SSIM ($\in [0.0, 1.0]$). |
| `src/evaluate/comparative_experiment.py` | Our research hypothesis | **Independently Implemented** | 5-condition controlled experiment evaluating downstream classification impact. |
| `app/streamlit_app.py` | RoadVision AI Platform | **Independently Implemented** | Custom dark navy/teal UI, 3-panel inspection, real-time degradation simulation, telemetry, and JSON export. |

---

## 3. Academic Integrity & Research Principles

1. **No Data Leakage:** Patches originating from the same source dashcam image are strictly assigned to either `train`, `val`, or `test`—never split across sets.
2. **True Downstream Evaluation:** Rather than claiming deblurring success purely on PSNR/SSIM, we evaluate downstream MobileNetV2 classification accuracy under restored inputs.
3. **No Synthetic Shortcuts:** Real vehicle dashcam images from open road damage benchmarks are ingested, preprocessed, and tested.
4. **License Compliance:**
   - DeblurGAN-v2 code structure: MIT License retained in file headers.
   - RDD2022 dataset: CC BY-SA 4.0 acknowledged in documentation and research paper references.
