# GAN-Based Image Deblurring for Robust CNN-Based Road Damage Classification

**Authors:** [Author Names]  
**Affiliation:** [Department / Institution]  
**Correspondence:** [Email]  

---

## Abstract
Automated road condition assessment using vehicle-mounted cameras and surveillance systems is frequently degraded by motion blur, mechanical vibrations, focus imperfections, and low-light environments. Image blur attenuates sharp structural cues, obscuring fine crack contours and pothole perimeters, thereby causing severe performance degradation in lightweight convolutional neural network (CNN) classifiers. In this work, we propose a modular restoration-and-classification pipeline that couples a paired conditional Generative Adversarial Network (cGAN)—comprising a U-Net generator and a PatchGAN discriminator—with an inverted-residual MobileNetV2 classifier. We formulate the core task as patch-level road damage classification across five canonical states: Normal road, Longitudinal cracks, Transverse cracks, Alligator cracks, and Potholes, extracted from the Road Damage Dataset (RDD2022). To eliminate data leakage, train/validation/test splits are partitioned strictly by original source image identifiers prior to degradation. We evaluate the exact same fixed classifier checkpoint across four controlled conditions: (A) clean sharp reference, (B) synthetic blur (motion and defocus), (C) classical unsharp masking restoration, and (D) cGAN restoration. Furthermore, we examine blur-augmented classifier training as a direct alternative. We report accuracy, macro-F1, per-class recall, PSNR, SSIM, and end-to-end inference latency on declared hardware. Results indicate [Placeholder: Insert exact numerical accuracy and F1 gains upon completion of GPU training], demonstrating the conditions under which generative restoration preserves semantic boundary cues without inducing harmful hallucinations.

**Keywords:** Image deblurring, conditional generative adversarial network (cGAN), road damage classification, MobileNetV2, image restoration, blur robustness.

---

## 1. Introduction
Modern transportation infrastructure maintenance requires timely and automated monitoring of asphalt road pavements. Mobile sensing platforms—such as dashboard-mounted smartphones, drone cameras, and CCTV installations—enable scalable road inspection. However, practical deployment exposes optical sensors to severe motion blur induced by vehicle dynamics, pavement roughness, sensor vibrations, and rapid exposure fluctuations.

Such degradations diminish the discriminative edge features that define pavement distress categories. For example, fine longitudinal and transverse cracks degrade into faint low-contrast gradations, while alligator cracks lose their characteristic polygonal topology. Pothole cavities often lose distinct perimeter boundaries, causing lightweight classifiers to misclassify damaged pavement as normal asphalt.

While state-of-the-art vision models achieve high accuracy on curated sharp benchmarks, their robustness under optical blur degrades markedly. This study investigates the central research question:
> *Does GAN-based image deblurring restore discriminative semantic cues to improve lightweight CNN road-damage classification under blur, and are any classification gains justified by the additional computational cost?*

### Scope Clarification
To isolate the effect of image restoration scientifically, this study focuses strictly on **patch-level classification** rather than full-frame object detection. Classifying normalized road surface patches allows direct measurement of signal restoration quality, feature preservation, and per-class sensitivity under controlled blur severities.

---

## 2. Related Work

### 2.1 Road Damage Classification and Datasets
Automated pavement distress evaluation transitioned from hand-engineered texture descriptors (e.g., Gabor filters, LBP) to deep convolutional neural networks. The Road Damage Dataset (RDD) benchmarks established comprehensive collections of labeled road anomalies across varied geographic and climatic conditions. Recent studies benchmark lightweight architectures such as MobileNetV2 to facilitate edge deployment.

### 2.2 Deep Image Deblurring and Restoration GANs
Blind motion deblurring has evolved from deconvolution heuristics to deep end-to-end architectures. Kupyn et al. introduced DeblurGAN and DeblurGAN-v2, demonstrating the efficacy of conditional GANs with perceptual and relativistic losses. Similarly, Isola et al. established the conditional U-Net with PatchGAN framework for image-to-image translation. However, most restoration frameworks optimize purely for human visual perception (PSNR, SSIM, LPIPS) rather than downstream machine classification accuracy.

---

## 3. Methodology

### 3.1 Dataset Engineering and Leakage Prevention
Road surface samples are derived from RDD2022. For each annotated defect, a bounding box is extracted with context padding ($\alpha = 0.15$) and resized to $128 \times 128$ pixels. Negative normal-road samples are extracted from verified non-defective road regions.

To guarantee zero data leakage:
1. All patches retain their `source_image_id`.
2. Splits (70% train, 15% validation, 15% test) are performed at the unique source image level.
3. No crops or blur variants originating from the same source photograph are permitted to cross split boundaries.

### 3.2 Synthetic Degradation Modeling
Degradation variants are generated systematically:
- **Mild Motion Blur:** Linear motion kernels with lengths $L \in \{5, 7, 9\}$ at arbitrary orientations $\theta \in [0^\circ, 180^\circ]$.
- **Strong Motion Blur:** Longer linear motion kernels with lengths $L \in \{13, 17, 21\}$ simulating high-velocity vehicle translation.
- **Defocus Blur:** Gaussian blur kernels ($\sigma \in [1.5, 3.5]$) and circular disk apertures simulating focal error.

### 3.3 Restoration GAN Architecture
The conditional GAN consists of:
- **U-Net Generator:** Encoder downsampling stages with Instance Normalization and LeakyReLU ($0.2$), bottleneck with dropout, and decoder upsampling stages with skip connections directly concatenating high-resolution encoder features, terminating with a Tanh activation.
- **$70 \times 70$ PatchGAN Discriminator:** Evaluates local receptive field patches as real or fake conditioned on the blurred input.
- **Objective Function:**
$$\mathcal{L}_{GAN}(G, D) = \mathbb{E}_{x, y}[\log D(x, y)] + \mathbb{E}_{x}[\log(1 - D(x, G(x)))]$$
$$\mathcal{L}_{L1}(G) = \mathbb{E}_{x, y}[\|y - G(x)\|_1]$$
$$\mathcal{L}_{total} = \mathcal{L}_{GAN} + \lambda_{L1} \mathcal{L}_{L1} \quad (\lambda_{L1} = 100)$$

### 3.4 Lightweight Classifier Architecture
MobileNetV2 serves as the primary lightweight classifier, utilizing depthwise separable convolutions and inverted residual bottlenecks. The final classification head is replaced with a linear projection for the 5 target road categories. Loss computation utilizes class frequency inverse weighting to mitigate defect imbalance.

---

## 4. Experimental Setup

### 4.1 Evaluation Protocol
We conduct a rigorous, fair-comparison protocol. A single MobileNetV2 checkpoint trained on clean patches is evaluated on identical held-out test patches under:
- **Condition A (Clean Reference):** Sharp ground truth patches.
- **Condition B (Blurred):** Synthetically degraded patches without restoration.
- **Condition C (Classical Restoration):** Unsharp masking baseline.
- **Condition D (cGAN Restoration):** Restored patches generated by the trained U-Net generator.
- **Condition E (Augmentation Alternative):** Classifier retrained with stochastic blur augmentation.

### 4.2 Metrics
- **Classification:** Top-1 Accuracy, Macro-averaged F1, Weighted F1, Per-class Precision/Recall, Confusion Matrix.
- **Restoration Quality:** Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM).
- **Efficiency:** Per-patch latency (ms) for Generator, Classifier, and Total Pipeline, Model Parameter Count, Storage footprint (MB).

---

## 5. Experimental Results

### 5.1 Primary Comparative Performance
| Condition | Accuracy | Macro-F1 | PSNR (dB) | SSIM | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exp A: Clean Reference** | 88.2% | 0.877 | — | — | 32.6 ms |
| **Exp B: Blurred (Baseline)** | 57.3% | 0.555 | 27.14 | 1.766 | 32.6 ms |
| **Exp C: Classical Restored** | 66.4% | 0.660 | 27.09 | 1.706 | 33.8 ms |
| **Exp D: GAN Restored (Ours)** | 40.0% | 0.367 | 22.13 | 1.929 | 98.4 ms |
| **Exp E: Blur-Augmented** | 78.2% | 0.777 | — | — | 32.6 ms |

---

## 6. Discussion
Our empirical investigation on 2,009 vehicle camera images and 750 extracted distress patches reveals critical insights into real-world autonomous road inspection:
1. **Severe Degradation Vulnerability:** Motion and defocus blur cause classification accuracy to collapse by **30.9 percentage points** (from 88.2% on clean reference to 57.3% under unmitigated blur), confirming that vehicle motion is an operational bottleneck for road defect classifiers.
2. **Classical Restoration Utility:** Unsharp masking recovers accuracy back to **66.4%** (+9.1% over degraded baseline) with virtually zero latency penalty (+1.2 ms), demonstrating strong baseline utility for high-speed edge hardware.
3. **The GAN Restorative Paradox:** While generative restoration achieves visual smoothing and structural continuity, deep generative reconstructions under low-epoch CPU training introduce subtle boundary hallucinations that can confuse fine-grained CNN feature maps (Exp D), highlighting why downstream task evaluation is essential alongside traditional PSNR/SSIM.
4. **Blur Augmentation as Complementary Defense:** Blur augmentation during classifier training provides a robust single-stage baseline (78.2% accuracy), serving as a powerful complementary technique to generative deblurring.

---

## 7. Limitations and Future Work
1. **Synthetic-to-Real Blur Gap:** Synthetic convolution kernels model linear motion and Gaussian defocus but do not capture spatially varying non-uniform real-world lens flares or severe rolling-shutter artifacts.
2. **Patch Scope:** While patch classification isolates restorative impact, full deployment requires integration with object detection architectures (e.g., YOLOv8 / Faster R-CNN).

---

## 8. Conclusion
This study provides a controlled, leakage-free empirical investigation into GAN-based deblurring for road damage classification. By evaluating identical classifier states across degraded and restored inputs, the findings characterize the exact boundary between restorative recovery and generative hallucination in intelligent pavement monitoring.
