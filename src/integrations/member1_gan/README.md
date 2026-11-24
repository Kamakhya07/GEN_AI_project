# Member 1: GAN Road Deblurring Integration

Welcome to the **RoadVision AI Member 1 GAN Subsystem**. This folder is completely decoupled so you can independently develop, train, and commit your Generative Adversarial Network without causing merge conflicts or breaking the main dashboard.

---

## 🎯 Role & Objective
- **Subsystem:** Generative Adversarial Network (GAN) for Road Surface Deblurring.
- **Workflow Position:** Stage 2 (`Road Image Upload → **GAN Deblurring** → Pothole Detection → Severity Classification → Final Assessment`).
- **Goal:** Ingest blurred/motion-degraded road images captured by high-speed survey vehicles and reconstruct sharp, high-fidelity pavement textures for downstream defect classification.

---

## 📁 Directory Structure
```
src/integrations/member1_gan/
├── __init__.py           # Package exports
├── model.py              # Generator architecture (DeblurGAN-v2 / Custom GAN)
├── infer.py              # Inference hook consumed by dashboard
├── README.md             # This guide
├── checkpoints/          # Put your trained weights here
│   ├── .gitkeep
│   └── best_gan.pt       # <-- TARGET CHECKPOINT FILE
└── notebooks/            # Put your Jupyter / Google Colab notebooks here
    └── .gitkeep
```

---

## 🚀 Integration Steps

### Step 1: Model Architecture (`model.py`)
- By default, `Member1GAN` wraps the lightweight **DeblurGAN-v2 FPN-MobileNetV2** generator (3.16M parameters, fast edge inference).
- If you use a custom architecture, edit `Member1GAN` in `model.py` to match your PyTorch generator class.

### Step 2: Training & Checkpoint Export
1. Train your model on blurred/sharp road image pairs (e.g., using RDD2022 or synthetic motion blur pipeline).
2. Save your best model checkpoint to:
   ```
   src/integrations/member1_gan/checkpoints/best_gan.pt
   ```
3. PyTorch save format:
   ```python
   torch.save({
       "generator_state_dict": model.state_dict(),
       "epoch": epoch,
       "val_psnr": val_psnr,
       "val_ssim": val_ssim
   }, "src/integrations/member1_gan/checkpoints/best_gan.pt")
   ```

### Step 3: Verify Inference
Run a quick test from the workspace root:
```powershell
python -c "from src.integrations.member1_gan import is_member1_model_available, load_member1_gan; print('Available:', is_member1_model_available()); model = load_member1_gan(); print('Loaded:', model is not None)"
```

### Step 4: Live Dashboard Verification
- As soon as `best_gan.pt` is placed in `checkpoints/`, the dashboard's Stage 2 panel will switch from `[Pending Integration: Awaiting Member 1 Checkpoint]` to `[Active: Member 1 DeblurGAN Loaded]` and run real-time inference on the uploaded road images.

---

## ⚙️ Specifications
- **Input Tensor:** Shape `(B, 3, H, W)` normalized to `[-1.0, 1.0]`.
- **Output Tensor:** Shape `(B, 3, H, W)` restored RGB normalized to `[-1.0, 1.0]`.
- **Target Resolution:** $224 \times 224$ or $256 \times 256$ pixels.
- **Licensing Compliance:** If using DeblurGAN-v2 code, maintain attribution to Kupyn et al. (ICCV 2019 / BSD-2 Clause).
