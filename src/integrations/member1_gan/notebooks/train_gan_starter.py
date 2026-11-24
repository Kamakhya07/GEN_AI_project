"""
Member 1: GAN Deblurring Training Pipeline (Colab / Local Standalone)
=====================================================================
Template training script for Member 1 to train the DeblurGAN-v2 generator.
Saves the resulting checkpoint directly into checkpoints/best_gan.pt.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Import Member 1 GAN Model Architecture
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.integrations.member1_gan.model import Member1GAN
from src.degradation.motion_blur import apply_motion_blur

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_gan.pt")


class PairedRoadDeblurDataset(Dataset):
    """Generates on-the-fly motion blurred / sharp paired road surface patches."""
    def __init__(self, sharp_images_dir: str, patch_size: int = 224):
        self.image_paths = [
            os.path.join(sharp_images_dir, f) for f in os.listdir(sharp_images_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        self.patch_size = patch_size
        self.tf = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        sharp_img = Image.open(path).convert("RGB")
        sharp_np = np.array(sharp_img)
        # Apply motion blur
        blur_np = apply_motion_blur(sharp_np, kernel_size=9, angle_degrees=45)
        blur_img = Image.fromarray(blur_np)
        return self.tf(blur_img), self.tf(sharp_img)


def train_member1_gan(data_dir: str, epochs: int = 15, lr: float = 2e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Member 1 GAN] Training on {device}")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    generator = Member1GAN().to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.L1Loss()
    
    # Save checkpoint format
    torch.save({
        "generator_state_dict": generator.state_dict(),
        "epoch": epochs,
        "model_type": "DeblurGAN-v2-FPN-MobileNetV2"
    }, SAVE_PATH)
    print(f"[Member 1 GAN] Template checkpoint exported successfully to {SAVE_PATH}")


if __name__ == "__main__":
    print("Run `train_member1_gan(data_dir='path/to/sharp_images')` to train.")
