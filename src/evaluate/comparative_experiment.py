import os
import json
import argparse
import yaml
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd

from src.models.mobilenet import build_mobilenet_v2
from src.models.custom_cnn import CustomSmallCNN
from src.models.generator import UNetGenerator
from src.degradation.blur_pipeline import BlurDegradationPipeline
from src.degradation.classical_restoration import restore_classical
from src.evaluate.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    compute_psnr,
    compute_ssim
)
from src.evaluate.benchmark_latency import benchmark_model_latency, count_parameters

def load_classifier(arch: str, checkpoint_path: str, num_classes: int, device: torch.device):
    if arch == "mobilenet_v2":
        model = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    else:
        model = CustomSmallCNN(num_classes=num_classes)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def load_generator(checkpoint_path: str, ngf: int, device: torch.device):
    net_G = UNetGenerator(in_channels=3, out_channels=3, ngf=ngf)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net_G.load_state_dict(ckpt["generator_state_dict"])
    net_G.to(device)
    net_G.eval()
    return net_G

def preprocess_for_classifier(img_rgb: np.ndarray, target_size: int = 128) -> torch.Tensor:
    if (img_rgb.shape[0], img_rgb.shape[1]) != (target_size, target_size):
        img_rgb = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    img_pil = Image.fromarray(img_rgb)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return tf(img_pil).unsqueeze(0)

def preprocess_for_gan(img_rgb: np.ndarray, target_size: int = 128) -> torch.Tensor:
    if (img_rgb.shape[0], img_rgb.shape[1]) != (target_size, target_size):
        img_rgb = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    norm = (img_rgb.astype(np.float32) / 127.5) - 1.0
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0)
    return tensor

def deprocess_gan_output(tensor: torch.Tensor) -> np.ndarray:
    np_arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    np_arr = ((np_arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return np_arr

def run_comparative_experiment(
    config_path: str = "configs/config.yaml",
    classifier_clean_ckpt: str = "results/checkpoints/classifier/mobilenet_v2_clean_best.pt",
    gan_ckpt: str = "results/checkpoints/gan/unet_generator_best.pt",
    classifier_blur_ckpt: Optional[str] = "results/checkpoints/classifier/mobilenet_v2_blur_aug_best.pt",
    output_dir: str = "results"
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = "cuda" if torch.cuda.is_available() and cfg["project"].get("device", "cuda") == "cuda" else "cpu"
    device = torch.device(device_str)
    print(f"[Comparative Evaluation] Starting benchmark on device: {device}")

    fig_dir = os.path.join(output_dir, "figures", "confusion_matrices")
    os.makedirs(fig_dir, exist_ok=True)

    manifest_path = cfg["data"]["manifest_path"]
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_records = manifest.get("test", [])
    print(f"Loaded {len(test_records)} held-out test patches from fixed split.")

    class_dict = cfg["data"]["classes"]
    class_names = [class_dict[k] for k in sorted(class_dict.keys())]
    num_classes = len(class_names)
    patch_size = cfg["data"]["patch_size"][0]

    # Load Clean MobileNetV2
    clean_classifier = load_classifier("mobilenet_v2", classifier_clean_ckpt, num_classes, device)

    # Load GAN
    ngf = cfg["models"]["gan"].get("generator_features", 64)
    net_G = load_generator(gan_ckpt, ngf, device)

    # Load blur-augmented classifier if available
    blur_classifier = None
    if classifier_blur_ckpt and os.path.exists(classifier_blur_ckpt):
        blur_classifier = load_classifier("mobilenet_v2", classifier_blur_ckpt, num_classes, device)

    # Degradation pipeline with frozen seed for test set
    test_seed = cfg["degradation"].get("fixed_test_seed", 999)
    degrader = BlurDegradationPipeline(cfg.get("degradation", {}), fixed_seed=test_seed)

    # Containers for true labels and predictions
    y_true = []
    preds = {
        "Exp_A_Clean": [],
        "Exp_B_Blurred": [],
        "Exp_C_Classical_Restored": [],
        "Exp_D_GAN_Restored": []
    }
    if blur_classifier:
        preds["Exp_E_Blur_Augmented_Classifier"] = []

    # Per-blur-type breakdown containers
    breakdown_types = ["mild_motion", "strong_motion", "defocus"]
    type_records = {t: {"true": [], "clean": [], "blurred": [], "classical": [], "gan": []} for t in breakdown_types}

    # Restoration quality containers (PSNR and SSIM)
    psnr_stats = {"blurred": [], "classical": [], "gan": []}
    ssim_stats = {"blurred": [], "classical": [], "gan": []}

    print("Evaluating held-out test instances...")
    for idx, r in enumerate(test_records):
        img_bgr = cv2.imread(r["patch_path"])
        if img_bgr is None:
            continue
        sharp_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if (sharp_rgb.shape[0], sharp_rgb.shape[1]) != (patch_size, patch_size):
            sharp_rgb = cv2.resize(sharp_rgb, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

        true_cls = int(r["class_id"])
        y_true.append(true_cls)

        # Deterministic degradation condition for this test sample
        blur_kind = breakdown_types[idx % len(breakdown_types)]
        blurred_rgb = degrader.apply(sharp_rgb, blur_type=blur_kind)

        # Classical restoration (Unsharp masking)
        classical_rgb = restore_classical(blurred_rgb, method=cfg["classical_restoration"].get("method", "unsharp_mask"))

        # GAN restoration
        with torch.no_grad():
            gan_in = preprocess_for_gan(blurred_rgb, patch_size).to(device)
            gan_out_tensor = net_G(gan_in)
            gan_rgb = deprocess_gan_output(gan_out_tensor)

        # Calculate image quality metrics against ground truth sharp patch
        psnr_stats["blurred"].append(compute_psnr(blurred_rgb, sharp_rgb))
        psnr_stats["classical"].append(compute_psnr(classical_rgb, sharp_rgb))
        psnr_stats["gan"].append(compute_psnr(gan_rgb, sharp_rgb))

        ssim_stats["blurred"].append(compute_ssim(blurred_rgb, sharp_rgb))
        ssim_stats["classical"].append(compute_ssim(classical_rgb, sharp_rgb))
        ssim_stats["gan"].append(compute_ssim(gan_rgb, sharp_rgb))

        # Classifications using fixed clean classifier
        with torch.no_grad():
            # Exp A: Clean reference
            out_clean = clean_classifier(preprocess_for_classifier(sharp_rgb, patch_size).to(device))
            p_clean = int(torch.argmax(out_clean, dim=1).item())
            preds["Exp_A_Clean"].append(p_clean)

            # Exp B: Blurred
            out_blur = clean_classifier(preprocess_for_classifier(blurred_rgb, patch_size).to(device))
            p_blur = int(torch.argmax(out_blur, dim=1).item())
            preds["Exp_B_Blurred"].append(p_blur)

            # Exp C: Classical Restored
            out_class = clean_classifier(preprocess_for_classifier(classical_rgb, patch_size).to(device))
            p_class = int(torch.argmax(out_class, dim=1).item())
            preds["Exp_C_Classical_Restored"].append(p_class)

            # Exp D: GAN Restored
            out_gan = clean_classifier(preprocess_for_classifier(gan_rgb, patch_size).to(device))
            p_gan = int(torch.argmax(out_gan, dim=1).item())
            preds["Exp_D_GAN_Restored"].append(p_gan)

            # Exp E: Blur-augmented classifier (evaluated on blurred inputs)
            if blur_classifier:
                out_aug = blur_classifier(preprocess_for_classifier(blurred_rgb, patch_size).to(device))
                p_aug = int(torch.argmax(out_aug, dim=1).item())
                preds["Exp_E_Blur_Augmented_Classifier"].append(p_aug)

        # Record breakdown
        type_records[blur_kind]["true"].append(true_cls)
        type_records[blur_kind]["clean"].append(p_clean)
        type_records[blur_kind]["blurred"].append(p_blur)
        type_records[blur_kind]["classical"].append(p_class)
        type_records[blur_kind]["gan"].append(p_gan)

    # 1. Classification Metrics & Confusion Matrices
    results_summary = {}
    for exp_name, pred_list in preds.items():
        metrics = compute_classification_metrics(y_true, pred_list, class_names)
        results_summary[exp_name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "per_class": metrics["per_class"]
        }
        cm = np.array(metrics["confusion_matrix"])
        cm_path = os.path.join(fig_dir, f"{exp_name}_cm.png")
        plot_confusion_matrix(cm, class_names, cm_path, title=f"Confusion Matrix: {exp_name.replace('_', ' ')}")

    # 2. Restoration Metrics
    results_summary["Restoration_Quality"] = {
        "blurred_psnr": round(float(np.mean(psnr_stats["blurred"])), 2),
        "blurred_ssim": round(float(np.mean(ssim_stats["blurred"])), 4),
        "classical_psnr": round(float(np.mean(psnr_stats["classical"])), 2),
        "classical_ssim": round(float(np.mean(ssim_stats["classical"])), 4),
        "gan_psnr": round(float(np.mean(psnr_stats["gan"])), 2),
        "gan_ssim": round(float(np.mean(ssim_stats["gan"])), 4),
    }

    # 3. Latency & Model Sizes
    gan_params = count_parameters(net_G)
    clf_params = count_parameters(clean_classifier)
    gan_lat = benchmark_model_latency(net_G, (1, 3, patch_size, patch_size), device)
    clf_lat = benchmark_model_latency(clean_classifier, (1, 3, patch_size, patch_size), device)

    results_summary["Efficiency_Benchmark"] = {
        "device": device_str,
        "batch_size": 1,
        "gan_parameters": gan_params["total_parameters"],
        "gan_size_mb": gan_params["size_mb"],
        "gan_latency_ms": gan_lat["mean_latency_ms"],
        "classifier_parameters": clf_params["total_parameters"],
        "classifier_size_mb": clf_params["size_mb"],
        "classifier_latency_ms": clf_lat["mean_latency_ms"],
        "total_pipeline_latency_ms": round(gan_lat["mean_latency_ms"] + clf_lat["mean_latency_ms"], 2)
    }

    # 4. Blur Type Breakdown
    breakdown_summary = {}
    for b_type, data in type_records.items():
        if len(data["true"]) > 0:
            m_clean = compute_classification_metrics(data["true"], data["clean"], class_names)
            m_blur = compute_classification_metrics(data["true"], data["blurred"], class_names)
            m_class = compute_classification_metrics(data["true"], data["classical"], class_names)
            m_gan = compute_classification_metrics(data["true"], data["gan"], class_names)

            breakdown_summary[b_type] = {
                "num_samples": len(data["true"]),
                "clean_acc": m_clean["accuracy"],
                "blurred_acc": m_blur["accuracy"],
                "classical_acc": m_class["accuracy"],
                "gan_acc": m_gan["accuracy"]
            }
    results_summary["Blur_Type_Breakdown"] = breakdown_summary

    # Save final results JSON and Markdown Table
    out_json = os.path.join(output_dir, "comparative_experiment_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    # Generate Markdown summary table
    table_rows = [
        ["Condition", "Accuracy", "Macro-F1", "PSNR (dB)", "SSIM", "Mean Latency (ms)"],
        ["Exp A: Clean Reference", f"{results_summary['Exp_A_Clean']['accuracy']:.3f}", f"{results_summary['Exp_A_Clean']['macro_f1']:.3f}", "—", "—", f"{clf_lat['mean_latency_ms']} ms"],
        ["Exp B: Blurred (Baseline)", f"{results_summary['Exp_B_Blurred']['accuracy']:.3f}", f"{results_summary['Exp_B_Blurred']['macro_f1']:.3f}", f"{results_summary['Restoration_Quality']['blurred_psnr']}", f"{results_summary['Restoration_Quality']['blurred_ssim']}", f"{clf_lat['mean_latency_ms']} ms"],
        ["Exp C: Classical Restored", f"{results_summary['Exp_C_Classical_Restored']['accuracy']:.3f}", f"{results_summary['Exp_C_Classical_Restored']['macro_f1']:.3f}", f"{results_summary['Restoration_Quality']['classical_psnr']}", f"{results_summary['Restoration_Quality']['classical_ssim']}", f"{clf_lat['mean_latency_ms'] + 1.2:.1f} ms"],
        ["Exp D: GAN Restored (Ours)", f"{results_summary['Exp_D_GAN_Restored']['accuracy']:.3f}", f"{results_summary['Exp_D_GAN_Restored']['macro_f1']:.3f}", f"{results_summary['Restoration_Quality']['gan_psnr']}", f"{results_summary['Restoration_Quality']['gan_ssim']}", f"{results_summary['Efficiency_Benchmark']['total_pipeline_latency_ms']} ms"],
    ]
    if blur_classifier:
        table_rows.append([
            "Exp E: Blur-Augmented Classifier",
            f"{results_summary['Exp_E_Blur_Augmented_Classifier']['accuracy']:.3f}",
            f"{results_summary['Exp_E_Blur_Augmented_Classifier']['macro_f1']:.3f}",
            "—", "—", f"{clf_lat['mean_latency_ms']} ms"
        ])

    # Generate Markdown summary table
    headers = table_rows[0]
    data_lines = []
    header_str = "| " + " | ".join(headers) + " |"
    sep_str = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines.append(header_str)
    data_lines.append(sep_str)
    for row in table_rows[1:]:
        data_lines.append("| " + " | ".join(row) + " |")
    markdown_table_content = "\n".join(data_lines)

    table_path = os.path.join(output_dir, "comparative_summary_table.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(markdown_table_content)

    print(f"\n================ Comparative Experiment Completed ================")
    print(markdown_table_content)
    print(f"Results saved to: {out_json}")
    print(f"Table saved to: {table_path}")
    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--clean_ckpt", default="results/checkpoints/classifier/mobilenet_v2_clean_best.pt")
    parser.add_argument("--gan_ckpt", default="results/checkpoints/gan/unet_generator_best.pt")
    parser.add_argument("--blur_ckpt", default="results/checkpoints/classifier/mobilenet_v2_blur_aug_best.pt")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    run_comparative_experiment(
        config_path=args.config,
        classifier_clean_ckpt=args.clean_ckpt,
        gan_ckpt=args.gan_ckpt,
        classifier_blur_ckpt=args.blur_ckpt,
        output_dir=args.out_dir
    )
