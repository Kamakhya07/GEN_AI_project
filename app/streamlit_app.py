import os
import sys
import time
import json
import yaml
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import streamlit as st
import pandas as pd

# Add workspace root to path
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_APP_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.mobilenet import build_mobilenet_v2
from src.models.generator import UNetGenerator
from src.degradation.blur_pipeline import BlurDegradationPipeline
from src.degradation.motion_blur import apply_motion_blur
from src.degradation.defocus_blur import apply_defocus_blur
from src.degradation.classical_restoration import restore_classical
from src.evaluate.metrics import compute_psnr, compute_ssim
from src.data.synthetic_samples import generate_synthetic_road_patch

# -----------------
# Page Configuration & Styling
# -----------------
st.set_page_config(
    page_title="RoadPatch GAN Deblur & Classification",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Scientific Dashboard Theme
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .metric-val {
        font-size: 22px;
        font-weight: 700;
        color: #1a1a2e;
        margin-top: 4px;
    }
    .metric-lbl {
        font-size: 12px;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Damage Badges */
    .badge-normal { background-color: #28a745; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 13px; }
    .badge-longitudinal { background-color: #007bff; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 13px; }
    .badge-transverse { background-color: #6f42c1; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 13px; }
    .badge-alligator { background-color: #fd7e14; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 13px; }
    .badge-pothole { background-color: #dc3545; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 13px; }

    /* Pipeline Step Box */
    .step-box {
        border-radius: 12px;
        border: 1px solid #dee2e6;
        padding: 12px;
        background: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

# -----------------
# Cache Model & Pipeline Loading
# -----------------
@st.cache_resource
def load_resources():
    config_path = os.path.join(_PROJECT_ROOT, "configs", "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg["models"]["classifier"].get("num_classes", 5)
    class_map = cfg["data"]["classes"]
    class_names = [class_map[k] for k in sorted(class_map.keys())]

    # Load MobileNetV2 (Clean baseline)
    clf_path = os.path.join(_PROJECT_ROOT, "results", "checkpoints", "classifier", "mobilenet_v2_clean_best.pt")
    clean_classifier = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    if os.path.exists(clf_path):
        ckpt = torch.load(clf_path, map_location=device, weights_only=False)
        clean_classifier.load_state_dict(ckpt["model_state_dict"])
    clean_classifier.to(device)
    clean_classifier.eval()

    # Load MobileNetV2 (Blur-Augmented)
    blur_clf_path = os.path.join(_PROJECT_ROOT, "results", "checkpoints", "classifier", "mobilenet_v2_blur_aug_best.pt")
    blur_classifier = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    if os.path.exists(blur_clf_path):
        ckpt_b = torch.load(blur_clf_path, map_location=device, weights_only=False)
        blur_classifier.load_state_dict(ckpt_b["model_state_dict"])
    blur_classifier.to(device)
    blur_classifier.eval()

    # Load U-Net Generator
    gan_path = os.path.join(_PROJECT_ROOT, "results", "checkpoints", "gan", "unet_generator_best.pt")
    ngf = cfg["models"]["gan"].get("generator_features", 64)
    net_G = UNetGenerator(in_channels=3, out_channels=3, ngf=ngf)
    if os.path.exists(gan_path):
        ckpt_g = torch.load(gan_path, map_location=device, weights_only=False)
        net_G.load_state_dict(ckpt_g["generator_state_dict"])
    net_G.to(device)
    net_G.eval()

    degrader = BlurDegradationPipeline(cfg.get("degradation", {}))

    return clean_classifier, blur_classifier, net_G, degrader, class_names, device, cfg

clean_classifier, blur_classifier, net_G, degrader, class_names, device, cfg = load_resources()

CLASS_COLORS = {
    "Normal": "#28a745",
    "Longitudinal Crack": "#007bff",
    "Transverse Crack": "#6f42c1",
    "Alligator Crack": "#fd7e14",
    "Pothole": "#dc3545"
}

def get_class_badge(cls_name: str) -> str:
    key = cls_name.lower().split()[0]
    badge_map = {
        "normal": "badge-normal",
        "longitudinal": "badge-longitudinal",
        "transverse": "badge-transverse",
        "alligator": "badge-alligator",
        "pothole": "badge-pothole"
    }
    cls_tag = badge_map.get(key, "badge-normal")
    return f'<span class="{cls_tag}">{cls_name}</span>'

def compute_edge_map(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.hypot(sobelx, sobely)
    grad = (grad / (grad.max() + 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(grad, cv2.COLORMAP_MAGMA)

def run_classifier(model, img_rgb: np.ndarray):
    pil_img = Image.fromarray(img_rgb)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    t_in = tf(pil_img).unsqueeze(0).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(t_in)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    lat = (time.perf_counter() - t0) * 1000.0
    return probs, lat

# -----------------
# Header Navigation
# -----------------
st.title("🛣️ RoadPatch GAN Deblur & Robust Classification")
st.markdown("""
<div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 14px 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
    <h4 style="margin: 0; color: white;">Research Prototype: Paired cGAN Deblurring + Lightweight MobileNetV2</h4>
    <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 14px;">
        Empirical evaluation of deep generative restoration on 128×128 road distress patches under optical & motion blur.
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "🔬 Interactive Restoration & Inference Lab",
    "📈 Comparative Experiments & Results",
    "🗂️ Held-Out Test Split Explorer",
    "📜 Research Paper & Scientific Blueprint"
])

# ==========================================
# TAB 1: Interactive Lab
# ==========================================
with tabs[0]:
    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        st.subheader("⚙️ Patch & Degradation")
        src_mode = st.radio("Patch Source", ["Synthetic Generator", "Upload Image", "Preset RDD Samples"])

        target_size = cfg["data"]["patch_size"][0]
        input_image = None

        if src_mode == "Synthetic Generator":
            sample_cls = st.selectbox("Damage Class", [
                (0, "Normal Road (Clean Asphalt)"),
                (1, "Longitudinal Crack (Wheelpath)"),
                (2, "Transverse Crack (Thermal)"),
                (3, "Alligator Crack (Fatigue Grid)"),
                (4, "Pothole (Cavity Depression)")
            ], format_func=lambda x: x[1])
            seed_val = st.slider("Texture Seed", 1, 150, 42)
            input_image = generate_synthetic_road_patch(class_id=sample_cls[0], size=(target_size, target_size), seed=seed_val)

        elif src_mode == "Upload Image":
            up_file = st.file_uploader("Upload Patch (JPEG/PNG)", type=["jpg", "jpeg", "png"])
            if up_file:
                pil_raw = Image.open(up_file).convert("RGB")
                input_image = cv2.resize(np.array(pil_raw), (target_size, target_size), interpolation=cv2.INTER_AREA)
            else:
                st.info("Upload an image patch to begin.")

        else: # Preset RDD Samples
            manifest_file = os.path.join(_PROJECT_ROOT, cfg["data"]["manifest_path"])
            if os.path.exists(manifest_file):
                with open(manifest_file, "r") as f:
                    m = json.load(f)
                test_list = m.get("test", [])
                if test_list:
                    sample_idx = st.selectbox(
                        "Select Held-Out Test Patch",
                        range(len(test_list)),
                        format_func=lambda i: f"#{i+1}: {test_list[i]['class_name']} ({os.path.basename(test_list[i]['patch_path'])})"
                    )
                    r_path = test_list[sample_idx]["patch_path"]
                    if os.path.exists(r_path):
                        img_bgr = cv2.imread(r_path)
                        input_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        input_image = cv2.resize(input_image, (target_size, target_size))

        st.markdown("---")
        st.subheader("🌪️ Blur Degradation")
        blur_mode = st.selectbox("Blur Type", ["Mild Motion Blur", "Strong Motion Blur", "Defocus Blur", "No Degradation (Clean)"])
        
        kernel_sz = 9
        angle_deg = 45.0
        if blur_mode == "Mild Motion Blur":
            kernel_sz = st.slider("Motion Kernel Length", 3, 11, 7, step=2)
            angle_deg = st.slider("Motion Angle (°)", 0, 180, 35)
        elif blur_mode == "Strong Motion Blur":
            kernel_sz = st.slider("Motion Kernel Length", 13, 25, 17, step=2)
            angle_deg = st.slider("Motion Angle (°)", 0, 180, 60)
        elif blur_mode == "Defocus Blur":
            kernel_sz = st.slider("Gaussian Kernel Size", 5, 19, 11, step=2)
            sigma_val = st.slider("Defocus Sigma", 1.0, 5.0, 2.5)

        show_edges = st.checkbox("Show Crack Edge Maps (Sobel)", value=False)

    with col_main:
        if input_image is not None:
            # Generate Degradation
            if blur_mode == "Mild Motion Blur":
                blurry_img = apply_motion_blur(input_image.copy(), kernel_size=kernel_sz, angle_degrees=angle_deg)
            elif blur_mode == "Strong Motion Blur":
                blurry_img = apply_motion_blur(input_image.copy(), kernel_size=kernel_sz, angle_degrees=angle_deg)
            elif blur_mode == "Defocus Blur":
                blurry_img = apply_defocus_blur(input_image.copy(), kernel_size=kernel_sz, sigma=sigma_val)
            else:
                blurry_img = input_image.copy()

            # Classical Restoration (Unsharp)
            t_c0 = time.perf_counter()
            classical_img = restore_classical(blurry_img, method="unsharp_mask", amount=1.6)
            t_classical = (time.perf_counter() - t_c0) * 1000.0

            # GAN Restoration (U-Net)
            with torch.no_grad():
                t_g0 = time.perf_counter()
                norm_in = ((blurry_img.astype(np.float32) / 127.5) - 1.0)
                tensor_in = torch.from_numpy(norm_in).permute(2, 0, 1).unsqueeze(0).to(device)
                gan_out = net_G(tensor_in)
                gan_img = ((gan_out.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                t_gan = (time.perf_counter() - t_g0) * 1000.0

            # Run MobileNetV2 on all 4 images
            probs_sharp, lat_sharp = run_classifier(clean_classifier, input_image)
            probs_blur, lat_blur = run_classifier(clean_classifier, blurry_img)
            probs_class, lat_class = run_classifier(clean_classifier, classical_img)
            probs_gan, lat_gan = run_classifier(clean_classifier, gan_img)

            # Secondary: Blur-augmented classifier
            probs_aug, _ = run_classifier(blur_classifier, blurry_img)

            pred_sharp = class_names[int(np.argmax(probs_sharp))]
            pred_blur = class_names[int(np.argmax(probs_blur))]
            pred_class = class_names[int(np.argmax(probs_class))]
            pred_gan = class_names[int(np.argmax(probs_gan))]
            pred_aug = class_names[int(np.argmax(probs_aug))]

            # Compute PSNR & SSIM
            psnr_b = compute_psnr(blurry_img, input_image)
            ssim_b = compute_ssim(blurry_img, input_image)
            psnr_c = compute_psnr(classical_img, input_image)
            ssim_c = compute_ssim(classical_img, input_image)
            psnr_g = compute_psnr(gan_img, input_image)
            ssim_g = compute_ssim(gan_img, input_image)

            # -----------------
            # 4-Stage Comparison Display
            # -----------------
            st.subheader("🖼️ Comparative Image Processing Pipeline")
            c1, c2, c3, c4 = st.columns(4)

            display_sharp = compute_edge_map(input_image) if show_edges else input_image
            display_blur = compute_edge_map(blurry_img) if show_edges else blurry_img
            display_class = compute_edge_map(classical_img) if show_edges else classical_img
            display_gan = compute_edge_map(gan_img) if show_edges else gan_img

            with c1:
                st.markdown("**1. Original Sharp (Reference)**")
                st.image(display_sharp, use_container_width=True)
                st.markdown(f"Prediction: {get_class_badge(pred_sharp)} ({probs_sharp.max()*100:.1f}%)", unsafe_allow_html=True)
                st.caption("Ground Truth Target")

            with c2:
                st.markdown(f"**2. Blurred ({blur_mode})**")
                st.image(display_blur, use_container_width=True)
                st.markdown(f"Prediction: {get_class_badge(pred_blur)} ({probs_blur.max()*100:.1f}%)", unsafe_allow_html=True)
                st.caption(f"PSNR: {psnr_b:.1f} dB | SSIM: {ssim_b:.3f}")

            with c3:
                st.markdown("**3. Classical (Unsharp Mask)**")
                st.image(display_class, use_container_width=True)
                st.markdown(f"Prediction: {get_class_badge(pred_class)} ({probs_class.max()*100:.1f}%)", unsafe_allow_html=True)
                st.caption(f"PSNR: {psnr_c:.1f} dB | SSIM: {ssim_c:.3f}")

            with c4:
                st.markdown("**4. GAN Restored (Ours)**")
                st.image(display_gan, use_container_width=True)
                st.markdown(f"Prediction: {get_class_badge(pred_gan)} ({probs_gan.max()*100:.1f}%)", unsafe_allow_html=True)
                st.caption(f"PSNR: {psnr_g:.1f} dB | SSIM: {ssim_g:.3f}")

            st.markdown("---")

            # -----------------
            # Performance Cards
            # -----------------
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-lbl">MobileNetV2 Latency</div>
                    <div class="metric-val">{:.1f} ms</div>
                </div>
                """.format(lat_blur), unsafe_allow_html=True)

            with m2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-lbl">GAN Deblur Latency</div>
                    <div class="metric-val">{:.1f} ms</div>
                </div>
                """.format(t_gan), unsafe_allow_html=True)

            with m3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-lbl">Total Restored Pipeline</div>
                    <div class="metric-val">{:.1f} ms</div>
                </div>
                """.format(t_gan + lat_gan), unsafe_allow_html=True)

            with m4:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-lbl">Augmented Model Pred</div>
                    <div class="metric-val" style="font-size: 16px;">{} ({:.1f}%)</div>
                </div>
                """.format(pred_aug, probs_aug.max()*100), unsafe_allow_html=True)

            # -----------------
            # Probability Distributions
            # -----------------
            st.subheader("📊 Damage Probability Distribution Across Conditions")
            df_probs = pd.DataFrame({
                "Class": class_names,
                "Original Sharp": probs_sharp,
                "Blurred Input": probs_blur,
                "Classical Restored": probs_class,
                "GAN Restored": probs_gan,
                "Blur-Augmented Model": probs_aug
            }).set_index("Class")
            st.bar_chart(df_probs, height=280)

# ==========================================
# TAB 2: Experiments & Benchmark
# ==========================================
with tabs[1]:
    st.header("📈 Primary Experimental Benchmark (Held-Out Test Set)")
    st.markdown("""
    In accordance with the experimental design protocol, the **same fixed MobileNetV2 checkpoint** is evaluated across identical test instances
    under five strictly controlled conditions.
    """)

    # Load Results JSON
    res_json_path = os.path.join(_PROJECT_ROOT, "results", "comparative_experiment_results.json")
    if os.path.exists(res_json_path):
        with open(res_json_path, "r") as f:
            res_data = json.load(f)

        table_md_path = os.path.join(_PROJECT_ROOT, "results", "comparative_summary_table.md")
        if os.path.exists(table_md_path):
            with open(table_md_path, "r") as f:
                st.markdown(f.read())

        st.markdown("---")

        # Confusion Matrix Heatmap Viewer
        st.subheader("🗺️ Confusion Matrix Analysis")
        col_cm_select, col_cm_img = st.columns([1, 2])

        with col_cm_select:
            cm_options = [
                ("Exp_A_Clean", "Experiment A: Clean Reference"),
                ("Exp_B_Blurred", "Experiment B: Blurred (Baseline)"),
                ("Exp_C_Classical_Restored", "Experiment C: Classical Restored"),
                ("Exp_D_GAN_Restored", "Experiment D: GAN Restored (Ours)"),
                ("Exp_E_Blur_Augmented_Classifier", "Experiment E: Blur-Augmented Classifier")
            ]
            selected_cm = st.radio("Select Condition Heatmap", cm_options, format_func=lambda x: x[1])

            exp_key = selected_cm[0]
            if exp_key in res_data:
                st.markdown("#### Per-Class Recall & Precision")
                per_class = res_data[exp_key].get("per_class", {})
                pc_df = pd.DataFrame(per_class).T
                st.dataframe(pc_df.style.format("{:.2f}"))

        with col_cm_img:
            cm_img_path = os.path.join(_PROJECT_ROOT, "results", "figures", "confusion_matrices", f"{selected_cm[0]}_cm.png")
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, caption=f"Confusion Matrix: {selected_cm[1]}", use_container_width=True)

        st.markdown("---")

        # Efficiency & Latency Summary
        st.subheader("⚡ Computational Efficiency & Hardware Profile")
        eff = res_data.get("Efficiency_Benchmark", {})
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Execution Device", str(eff.get("device", "cpu")).upper())
        ec2.metric("U-Net Generator Size", f"{eff.get('gan_size_mb', 0):.1f} MB", f"{eff.get('gan_parameters', 0):,} params")
        ec3.metric("MobileNetV2 Size", f"{eff.get('classifier_size_mb', 0):.1f} MB", f"{eff.get('classifier_parameters', 0):,} params")
        ec4.metric("Total End-to-End Latency", f"{eff.get('total_pipeline_latency_ms', 0):.1f} ms", "per patch")

# ==========================================
# TAB 3: Held-Out Test Explorer
# ==========================================
with tabs[2]:
    st.header("🗂️ Explore the Zero-Leakage Test Split")
    st.markdown("""
    All test patches were partitioned at the unique source image ID level to guarantee zero data leakage.
    Browse individual test instances and verify model performance.
    """)

    manifest_file = os.path.join(_PROJECT_ROOT, cfg["data"]["manifest_path"])
    if os.path.exists(manifest_file):
        with open(manifest_file, "r") as f:
            manifest_data = json.load(f)
        test_patches = manifest_data.get("test", [])
        st.write(f"Total Held-Out Test Patches: **{len(test_patches)}**")

        # Grid of test samples
        cols_per_row = 4
        for row_idx in range(0, min(16, len(test_patches)), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for c_idx in range(cols_per_row):
                patch_idx = row_idx + c_idx
                if patch_idx < len(test_patches):
                    p_info = test_patches[patch_idx]
                    with row_cols[c_idx]:
                        if os.path.exists(p_info["patch_path"]):
                            t_img = Image.open(p_info["patch_path"])
                            st.image(t_img, use_container_width=True)
                            st.markdown(f"**#{patch_idx+1}**: {get_class_badge(p_info['class_name'])}", unsafe_allow_html=True)
                            st.caption(f"Source: `{p_info['source_image_id']}`")

# ==========================================
# TAB 4: Paper & Scientific Blueprint
# ==========================================
with tabs[3]:
    st.header("📜 Research Paper & Scientific Blueprint")
    st.markdown("""
    ### GAN-Based Image Deblurring for Robust CNN-Based Road Damage Classification
    **Authors:** [Research Team] | **Target Venue:** IEEE Transactions on Intelligent Transportation Systems / CVPR Workshop

    #### 1. Central Research Question
    > *Does GAN-based image deblurring restore discriminative semantic cues to improve lightweight CNN road-damage classification under blur, and are any gains worth the added inference latency and computational overhead?*

    #### 2. Scope & Research Guardrails
    - **Patch-Level Focus:** Classification of fixed $128 \\times 128$ road distress patches, isolating image restoration from object bounding-box localization.
    - **Zero Data Leakage:** Partitioning is performed strictly at the source photograph level before degradation.
    - **Controlled Fairness:** The primary classifier checkpoint is held constant across all 4 restoration conditions.

    #### 3. Starter Literature
    1. **Kupyn et al. (CVPR 2018):** *DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks.*
    2. **Kupyn et al. (ICCV 2019):** *DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better.*
    3. **Sandler et al. (CVPR 2018):** *MobileNetV2: Inverted Residuals and Linear Bottlenecks.*
    4. **Arya et al. (IEEE T-ITS 2022):** *Global Road Damage Detection: State-of-the-art Solutions and Open Challenges (RDD2022).*
    """)

    paper_draft_path = os.path.join(_PROJECT_ROOT, "paper", "research_paper_draft.md")
    if os.path.exists(paper_draft_path):
        with st.expander("📄 View Full Research Paper Draft"):
            with open(paper_draft_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
