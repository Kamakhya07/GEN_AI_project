# GAN-Based Image Deblurring for Robust CNN-Based Road Damage Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/Colab_End_to_End_Pipeline.ipynb)

A rigorous, reproducible research implementation investigating whether **Generative Adversarial Network (GAN) image deblurring** improves lightweight CNN classification of road-surface distress under optical and motion blur.

---

## 🎯 Research Overview

- **Core Problem:** Road images captured by dashcams, drones, and CCTV cameras suffer from vehicle motion blur, vibration, and focal defocus. This obscures crack boundaries and pothole perimeters, degrading automated pavement inspection.
- **Central Research Question:** *Does GAN-based image deblurring restore discriminative semantic cues to improve lightweight CNN road-damage classification under blur, and are any gains worth the added inference latency and computational overhead?*
- **Important Scientific Scope:** This project focuses on **Patch-Level Classification** of road distress (Normal, Longitudinal Crack, Transverse Crack, Alligator Crack, Pothole) to isolate image restoration effects from bounding-box detection dynamics.

---

## 📁 Repository Structure

```
├── configs/
│   └── config.yaml               # Experiment hyperparameters and paths
├── data/
│   ├── raw/                      # Raw dataset directory (RDD2022)
│   ├── processed/                # Cropped patches, manifests, counts
│   └── README.md                 # Dataset download and annotation guide
├── notebooks/
│   └── Colab_End_to_End_Pipeline.ipynb  # One-click Google Colab pipeline
├── src/
│   ├── data/                     # XML parsing, patch extraction, split manager
│   ├── degradation/              # Motion blur, defocus blur, classical unsharp masking
│   ├── models/                   # U-Net Generator, PatchGAN, MobileNetV2, Custom CNN
│   ├── train/                    # Classifier training & Paired cGAN training loops
│   └── evaluate/                 # Comparative benchmarks, metrics, latency profiling
├── app/
│   └── streamlit_app.py          # Interactive patch deblur & classification demo
├── results/
│   ├── checkpoints/              # Saved model weights
│   ├── figures/                  # Confusion matrices, sample comparisons
│   └── comparative_summary_table.md # Tabular experimental results
├── paper/
│   ├── research_paper_draft.md   # Complete academic paper draft
│   └── references.bib            # BibTeX citations
├── requirements.txt
└── README.md
```

---

## 🧪 Experimental Design: The 5 Controlled Conditions

To guarantee fairness and eliminate confounding variables, the **exact same fixed MobileNetV2 checkpoint** is evaluated across identical held-out test patches under:

1. **Experiment A (Clean Reference):** Sharp ground-truth patches (measures upper-bound clean performance).
2. **Experiment B (Blurred):** Patches degraded with synthetic motion & defocus blur (measures performance drop).
3. **Experiment C (Classical Restoration):** Blurred patches restored via unsharp masking deconvolution.
4. **Experiment D (GAN Restoration - Proposed):** Blurred patches restored by trained U-Net cGAN generator.
5. **Experiment E (Robustness Baseline):** MobileNetV2 trained with stochastic blur augmentation.

### 🛡️ Data Leakage Prevention Guarantee
Patches are split strictly by **unique source image ID** (70% train, 15% validation, 15% test). No crops, normal patches, or blur variants from the same source photograph ever cross split boundaries.

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install requirements in your virtual environment:
```bash
git clone https://github.com/your-username/road-damage-deblur-classification.git
cd road-damage-deblur-classification
pip install -r requirements.txt
```

### 2. Prepare Data (Synthetic / Smoke Test or RDD2022)
For instant local verification without waiting for 30GB RDD downloads, generate synthetic road patches:
```bash
python -m src.data.synthetic_samples
```
To process official RDD2022 dataset downloads:
```bash
python -m src.data.process_rdd --raw_dir data/raw/RDD2022/Japan --output_dir data/processed
```

### 3. Train MobileNetV2 Classifier (Clean Reference)
```bash
python -m src.train.train_classifier --arch mobilenet_v2 --epochs 20
```

### 4. Train Conditional Deblurring GAN (U-Net + PatchGAN)
```bash
python -m src.train.train_gan --epochs 30
```

### 5. (Optional) Train Blur-Augmented Classifier
```bash
python -m src.train.train_classifier --arch mobilenet_v2 --blur_aug --epochs 20
```

### 6. Run the Controlled Comparative Experiment
```bash
python -m src.evaluate.comparative_experiment
```
This automatically produces:
- Top-1 Accuracy and Macro-F1 across all 5 conditions.
- High-resolution confusion matrix heatmaps in `results/figures/confusion_matrices/`.
- Restoration quality metrics (PSNR & SSIM) on test patches.
- Inference latency benchmarks (ms per patch) on declared hardware.
- Consolidated results table in `results/comparative_summary_table.md`.

---

## 🖥️ Interactive Web Demonstration

Launch the local Streamlit visualizer to inspect patches, toggle blur models, and view real-time deblurring and classification predictions:
```bash
streamlit run app/streamlit_app.py
```

---

## 📄 Research Paper Draft

The complete paper manuscript draft with formal academic structure, mathematical formulations, literature review, and experimental analysis is available at:
- [paper/research_paper_draft.md](paper/research_paper_draft.md)
- [paper/references.bib](paper/references.bib)

---

## 📜 Citation & References
- Kupyn et al. (2018), *DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks*, CVPR.
- Sandler et al. (2018), *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR.
- Arya et al. (2022), *Global Road Damage Detection: State-of-the-art Solutions and Open Challenges*, IEEE T-ITS.
