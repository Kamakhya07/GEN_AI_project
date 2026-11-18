import os
import json
import argparse
import yaml
from typing import Optional, Dict, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.data.dataset import PairedDeblurDataset
from src.degradation.blur_pipeline import BlurDegradationPipeline
from src.models.generator import UNetGenerator
from src.models.discriminator import PatchGANDiscriminator

def calculate_psnr_ssim_batch(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Computes PSNR and SSIM on normalized tensors in range [-1, 1].
    Converts to [0, 255] uint8 for precise metrics.
    """
    preds_np = ((preds.detach().cpu().numpy().transpose(0, 2, 3, 1) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    targets_np = ((targets.detach().cpu().numpy().transpose(0, 2, 3, 1) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    psnr_vals = []
    ssim_vals = []
    for p, t in zip(preds_np, targets_np):
        mse = np.mean((p.astype(np.float32) - t.astype(np.float32)) ** 2)
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))
        psnr_vals.append(psnr)

        # SSIM calculation per channel
        gray_p = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY).astype(np.float64)
        gray_t = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY).astype(np.float64)
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        mu_p = cv2.GaussianBlur(gray_p, (11, 11), 1.5)
        mu_t = cv2.GaussianBlur(gray_t, (11, 11), 1.5)
        mu_p_sq = mu_p ** 2
        mu_t_sq = mu_t ** 2
        mu_pt = mu_p * mu_t
        sigma_p_sq = np.maximum(0, cv2.GaussianBlur(gray_p ** 2, (11, 11), 1.5) - mu_p_sq)
        sigma_t_sq = np.maximum(0, cv2.GaussianBlur(gray_t ** 2, (11, 11), 1.5) - mu_t_sq)
        sigma_pt = cv2.GaussianBlur(gray_p * gray_t, (11, 11), 1.5) - mu_pt

        ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / ((mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2))
        ssim_vals.append(float(np.clip(np.mean(ssim_map), 0.0, 1.0)))

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals))
    }

def train_gan_pipeline(
    config_path: str = "configs/config.yaml",
    override_epochs: Optional[int] = None,
    output_dir: str = "results/checkpoints/gan",
    samples_dir: str = "results/figures/restoration_samples"
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = "cuda" if torch.cuda.is_available() and cfg["project"].get("device", "cuda") == "cuda" else "cpu"
    device = torch.device(device_str)
    print(f"[GAN Training] Initializing U-Net Generator + PatchGAN Discriminator on {device}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    manifest_path = cfg["data"]["manifest_path"]
    patch_size = cfg["data"]["patch_size"][0]

    degradation_pipe = BlurDegradationPipeline(cfg.get("degradation", {}))

    train_dataset = PairedDeblurDataset(
        manifest_path=manifest_path,
        split="train",
        degradation_pipeline=degradation_pipe,
        target_size=(patch_size, patch_size)
    )
    val_dataset = PairedDeblurDataset(
        manifest_path=manifest_path,
        split="val",
        degradation_pipeline=degradation_pipe,
        target_size=(patch_size, patch_size)
    )

    batch_size = cfg["models"]["gan"].get("batch_size", 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    ngf = cfg["models"]["gan"].get("generator_features", 64)
    ndf = cfg["models"]["gan"].get("discriminator_features", 64)

    net_G = UNetGenerator(in_channels=3, out_channels=3, ngf=ngf).to(device)
    net_D = PatchGANDiscriminator(in_channels=6, ndf=ndf).to(device)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    lambda_l1 = cfg["models"]["gan"].get("lambda_l1", 100.0)

    lr = cfg["models"]["gan"].get("lr", 0.0002)
    beta1 = cfg["models"]["gan"].get("beta1", 0.5)
    beta2 = cfg["models"]["gan"].get("beta2", 0.999)

    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=lr, betas=(beta1, beta2))

    epochs = override_epochs or cfg["models"]["gan"].get("epochs", 30)
    best_psnr = -1.0
    history = []

    best_checkpoint_path = os.path.join(output_dir, "unet_generator_best.pt")

    for epoch in range(1, epochs + 1):
        net_G.train()
        net_D.train()

        running_loss_G = 0.0
        running_loss_D = 0.0
        running_l1 = 0.0

        for blurry, sharp in train_loader:
            blurry, sharp = blurry.to(device), sharp.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real loss
            pred_real = net_D(blurry, sharp)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

            # Fake loss
            fake_sharp = net_G(blurry)
            pred_fake = net_D(blurry, fake_sharp.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            pred_fake_for_G = net_D(blurry, fake_sharp)
            loss_G_adv = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            loss_G_l1 = criterion_L1(fake_sharp, sharp)

            loss_G = loss_G_adv + lambda_l1 * loss_G_l1
            loss_G.backward()
            optimizer_G.step()

            running_loss_D += loss_D.item() * blurry.size(0)
            running_loss_G += loss_G.item() * blurry.size(0)
            running_l1 += loss_G_l1.item() * blurry.size(0)

        epoch_loss_D = running_loss_D / len(train_dataset)
        epoch_loss_G = running_loss_G / len(train_dataset)
        epoch_l1 = running_l1 / len(train_dataset)

        # ---------------------
        #  Validation (PSNR/SSIM)
        # ---------------------
        net_G.eval()
        val_psnrs, val_ssims = [], []

        with torch.no_grad():
            for idx, (blurry, sharp) in enumerate(val_loader):
                blurry, sharp = blurry.to(device), sharp.to(device)
                fake_sharp = net_G(blurry)
                metrics = calculate_psnr_ssim_batch(fake_sharp, sharp)
                val_psnrs.append(metrics["psnr"])
                val_ssims.append(metrics["ssim"])

                # Save comparison images for the first batch
                if idx == 0 and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
                    # Concatenate horizontally: Blurry | Restored | Sharp
                    grid = torch.cat([blurry[:4], fake_sharp[:4], sharp[:4]], dim=0)
                    grid = (grid + 1.0) / 2.0  # Bring from [-1, 1] to [0, 1]
                    sample_img_path = os.path.join(samples_dir, f"epoch_{epoch:03d}_comparison.png")
                    save_image(grid, sample_img_path, nrow=4)

        mean_psnr = float(np.mean(val_psnrs))
        mean_ssim = float(np.mean(val_ssims))

        epoch_log = {
            "epoch": epoch,
            "loss_D": float(epoch_loss_D),
            "loss_G": float(epoch_loss_G),
            "l1_loss": float(epoch_l1),
            "val_psnr": mean_psnr,
            "val_ssim": mean_ssim
        }
        history.append(epoch_log)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss D: {epoch_loss_D:.4f} | Loss G: {epoch_loss_G:.4f} (L1: {epoch_l1:.4f}) | Val PSNR: {mean_psnr:.2f} dB, SSIM: {mean_ssim:.4f}")

        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            torch.save({
                "epoch": epoch,
                "generator_state_dict": net_G.state_dict(),
                "discriminator_state_dict": net_D.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
                "val_psnr": mean_psnr,
                "val_ssim": mean_ssim
            }, best_checkpoint_path)

    # Save training history
    log_path = os.path.join(output_dir, "gan_training_history.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best U-Net Generator checkpoint saved to: {best_checkpoint_path} (Best Val PSNR: {best_psnr:.2f} dB)")
    return best_checkpoint_path, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    train_gan_pipeline(config_path=args.config, override_epochs=args.epochs)
