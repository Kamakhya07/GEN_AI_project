import os
import json
import argparse
import yaml
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data.dataset import RoadPatchClassificationDataset
from src.degradation.blur_pipeline import BlurDegradationPipeline
from src.models.mobilenet import build_mobilenet_v2
from src.models.custom_cnn import CustomSmallCNN

def get_classifier_transforms(is_train: bool = True, target_size: int = 128):
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def compute_class_weights(manifest_path: str, num_classes: int = 5, device: str = "cpu") -> torch.Tensor:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    train_records = manifest.get("train", [])
    counts = np.zeros(num_classes, dtype=np.float32)
    for r in train_records:
        cid = int(r["class_id"])
        if 0 <= cid < num_classes:
            counts[cid] += 1

    counts = np.maximum(counts, 1.0)
    total = np.sum(counts)
    weights = total / (num_classes * counts)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def train_classifier_pipeline(
    config_path: str = "configs/config.yaml",
    architecture: str = "mobilenet_v2",
    use_blur_augmentation: bool = False,
    override_epochs: Optional[int] = None,
    output_dir: str = "results/checkpoints/classifier"
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = "cuda" if torch.cuda.is_available() and cfg["project"].get("device", "cuda") == "cuda" else "cpu"
    device = torch.device(device_str)
    print(f"[Classifier Training] Architecture: {architecture} | Blur Augmentation: {use_blur_augmentation} | Device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = cfg["data"]["manifest_path"]
    patch_size = cfg["data"]["patch_size"][0]

    degradation_fn = None
    if use_blur_augmentation:
        blur_gen = BlurDegradationPipeline(cfg.get("degradation", {}))
        # 50% chance of applying blur in training
        degradation_fn = lambda img: blur_gen(img) if np.random.rand() > 0.5 else img

    train_tf = get_classifier_transforms(is_train=True, target_size=patch_size)
    val_tf = get_classifier_transforms(is_train=False, target_size=patch_size)

    train_dataset = RoadPatchClassificationDataset(
        manifest_path=manifest_path,
        split="train",
        transform=train_tf,
        degradation_fn=degradation_fn,
        target_size=(patch_size, patch_size)
    )
    val_dataset = RoadPatchClassificationDataset(
        manifest_path=manifest_path,
        split="val",
        transform=val_tf,
        target_size=(patch_size, patch_size)
    )

    batch_size = cfg["models"]["classifier"].get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = cfg["models"]["classifier"].get("num_classes", 5)
    if architecture == "mobilenet_v2":
        model = build_mobilenet_v2(
            num_classes=num_classes,
            pretrained=cfg["models"]["classifier"].get("pretrained", True),
            fine_tune_layers=cfg["models"]["classifier"].get("fine_tune_layers", 4)
        )
    elif architecture == "custom_cnn":
        model = CustomSmallCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model.to(device)

    # Class weights for loss
    class_weights = compute_class_weights(manifest_path, num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    lr = cfg["models"]["classifier"].get("lr", 0.0005)
    weight_decay = cfg["models"]["classifier"].get("weight_decay", 0.0001)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    epochs = override_epochs or cfg["models"]["classifier"].get("epochs", 20)
    best_val_f1 = -1.0
    history = []

    suffix = "_blur_aug" if use_blur_augmentation else "_clean"
    save_filename = f"{architecture}{suffix}_best.pt"
    save_path = os.path.join(output_dir, save_filename)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = accuracy_score(train_targets, train_preds)
        epoch_f1 = f1_score(train_targets, train_preds, average="macro", zero_division=0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = accuracy_score(val_targets, val_preds)
        val_epoch_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

        epoch_stat = {
            "epoch": epoch,
            "train_loss": float(epoch_loss),
            "train_acc": float(epoch_acc),
            "train_f1": float(epoch_f1),
            "val_loss": float(val_epoch_loss),
            "val_acc": float(val_epoch_acc),
            "val_f1": float(val_epoch_f1)
        }
        history.append(epoch_stat)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.3f} F1: {epoch_f1:.3f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.3f} F1: {val_epoch_f1:.3f}")

        if val_epoch_f1 > best_val_f1:
            best_val_f1 = val_epoch_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "architecture": architecture,
                "val_acc": val_epoch_acc,
                "val_f1": val_epoch_f1,
                "blur_augmented": use_blur_augmentation
            }, save_path)

    # Save training history
    log_path = os.path.join(output_dir, f"{architecture}{suffix}_history.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Classifier checkpoint saved to: {save_path} (Best Val Macro-F1: {best_val_f1:.4f})")
    return save_path, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--arch", default="mobilenet_v2", choices=["mobilenet_v2", "custom_cnn"])
    parser.add_argument("--blur_aug", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    train_classifier_pipeline(
        config_path=args.config,
        architecture=args.arch,
        use_blur_augmentation=args.blur_aug,
        override_epochs=args.epochs
    )
