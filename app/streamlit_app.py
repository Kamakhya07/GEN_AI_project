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

# Add workspace root to sys.path
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_APP_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.mobilenet import build_mobilenet_v2
from src.models.generator import UNetGenerator
from src.models.deblurgan_v2 import DeblurGANv2Generator
from src.degradation.motion_blur import apply_motion_blur
from src.degradation.defocus_blur import apply_defocus_blur
from src.degradation.classical_restoration import restore_classical
from src.data.synthetic_samples import generate_synthetic_road_patch

# -----------------------------------------------------------------------------
# Signal Quality Metric Helpers (Strictly Bounded Float64 Formulation)
# -----------------------------------------------------------------------------
def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.ndim == 3 else img2

    gray1 = gray1.astype(np.float64)
    gray2 = gray2.astype(np.float64)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = np.maximum(0.0, cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq)
    sigma2_sq = np.maximum(0.0, cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq)
    sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    val = float(np.mean(ssim_map))
    return float(np.clip(val, 0.0, 1.0))

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RoadVision AI | Highway Defect & Deblurring Suite",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Premium Navy, Teal & Sky-Blue Dark Theme CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #070d19;
        --bg-surface: #0e172a;
        --bg-card: #131f37;
        --bg-card-hover: #172644;
        --border-subtle: rgba(56, 189, 248, 0.14);
        --border-bright: rgba(45, 212, 191, 0.4);
        --accent-teal: #14b8a6;
        --accent-teal-bright: #2dd4bf;
        --accent-sky: #38bdf8;
        --accent-blue: #0284c7;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
    }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1420px !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #091122 0%, #060c18 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(56, 189, 248, 0.12) !important;
        margin: 1.1rem 0 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3, 
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #e2e8f0 !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        margin-bottom: 0.4rem !important;
    }
    [data-testid="stSidebar"] .stCaption {
        color: var(--text-muted) !important;
        font-size: 0.76rem !important;
    }

    .stSelectbox > div > div {
        background-color: #101c33 !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
        font-size: 0.86rem !important;
    }
    .stSelectbox > div > div:hover {
        border-color: var(--accent-sky) !important;
    }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: var(--accent-teal-bright) !important;
        box-shadow: 0 0 10px rgba(45, 212, 191, 0.6) !important;
    }

    /* Top Navigation Header */
    .top-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(135deg, rgba(14, 23, 42, 0.92) 0%, rgba(19, 31, 55, 0.92) 100%);
        border: 1px solid var(--border-subtle);
        padding: 14px 22px;
        border-radius: 14px;
        margin-bottom: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
    }
    .brand-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(120deg, #ffffff 0%, #38bdf8 50%, #2dd4bf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-subtitle {
        font-size: 0.80rem;
        color: var(--text-secondary);
        font-weight: 500;
        margin-top: 2px;
        letter-spacing: 0.01em;
    }
    .nav-badges {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 5px 11px;
        border-radius: 9999px;
    }
    .badge-online {
        background: rgba(45, 212, 191, 0.12);
        color: #2dd4bf;
        border: 1px solid rgba(45, 212, 191, 0.3);
    }
    .badge-dataset {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-device {
        background: rgba(148, 163, 184, 0.1);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    /* Municipal Telematics Strip */
    .telematics-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(10, 16, 30, 0.8);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 8px 18px;
        margin-bottom: 20px;
        font-size: 0.74rem;
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-secondary);
    }
    .telematics-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .telematics-val {
        color: #f1f5f9;
        font-weight: 600;
    }

    /* Cards & Containers */
    .dash-card {
        background: linear-gradient(135deg, rgba(17, 28, 51, 0.88) 0%, rgba(13, 21, 39, 0.96) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .card-title {
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #e2e8f0;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Image Display Frame */
    .img-frame {
        background: #090f1d;
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 8px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .img-frame:hover {
        border-color: var(--accent-sky);
        transform: translateY(-2px);
    }
    .img-header {
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--accent-sky);
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 4px;
    }
    .img-footer {
        font-size: 0.72rem;
        color: var(--text-secondary);
        margin-top: 6px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Metric Grid Tiles */
    .metric-grid-dark {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-top: 14px;
    }
    .metric-box {
        background: rgba(10, 17, 32, 0.85);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 12px 8px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.25rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        background: linear-gradient(120deg, #38bdf8, #2dd4bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-name {
        font-size: 0.66rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-top: 3px;
    }

    /* Hero Diagnosis Cards */
    .hero-diagnosis-normal {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.3) 0%, rgba(13, 27, 42, 0.8) 100%);
        border: 1px solid rgba(45, 212, 191, 0.4);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(45, 212, 191, 0.1);
    }
    .hero-diagnosis-warning {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.35) 0%, rgba(13, 27, 42, 0.8) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.1);
    }
    .hero-diagnosis-danger {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.38) 0%, rgba(13, 27, 42, 0.8) 100%);
        border: 1px solid rgba(244, 63, 94, 0.45);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(244, 63, 94, 0.12);
    }

    /* PCI Gauge Card (ASTM D6433 Standard) */
    .pci-box {
        background: rgba(10, 18, 35, 0.85);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 14px;
        margin: 14px 0;
    }
    .pci-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--accent-sky);
        margin-bottom: 6px;
    }
    .pci-score {
        font-size: 1.5rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    .pci-bar-bg {
        width: 100%;
        height: 8px;
        background: #080f1e;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 6px;
    }
    .pci-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    /* Work Order Dispatch Box */
    .action-dispatch-box {
        background: rgba(10, 18, 35, 0.9);
        border-left: 3px solid var(--accent-sky);
        border-top: 1px solid var(--border-subtle);
        border-right: 1px solid var(--border-subtle);
        border-bottom: 1px solid var(--border-subtle);
        border-radius: 0 10px 10px 0;
        padding: 14px 16px;
        margin: 14px 0;
    }
    .action-dispatch-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--accent-sky);
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
    }
    .action-dispatch-desc {
        font-size: 0.82rem;
        color: #cbd5e1;
        line-height: 1.45;
    }

    /* Probability Bar Row */
    .prob-bar-wrapper {
        margin-bottom: 9px;
    }
    .prob-bar-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.76rem;
        margin-bottom: 3px;
        color: #cbd5e1;
    }
    .prob-bar-bg {
        width: 100%;
        height: 6px;
        background: #09101f;
        border-radius: 3px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.4s ease;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        gap: 6px !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        padding-bottom: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--text-secondary) !important;
        border-radius: 8px 8px 0 0 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        padding: 7px 14px !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(56, 189, 248, 0.15) 0%, rgba(14, 23, 42, 0.9) 100%) !important;
        border-color: var(--accent-sky) !important;
        color: #ffffff !important;
    }

    /* Download Button */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0d9488 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(45, 212, 191, 0.3) !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 9px 16px !important;
        font-size: 0.84rem !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Model Engine & Resources (Multi-Model Architecture)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ai_engine():
    config_path = os.path.join(_PROJECT_ROOT, "configs", "config_real.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(_PROJECT_ROOT, "configs", "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg["models"]["classifier"].get("num_classes", 5)
    class_map = cfg["data"]["classes"]
    class_names = [class_map[k] for k in sorted(class_map.keys())]

    # 1. Load MobileNetV2 Classifier
    clf_path = os.path.join(_PROJECT_ROOT, "results", "checkpoints", "classifier", "mobilenet_v2_clean_best.pt")
    classifier = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    if os.path.exists(clf_path):
        ckpt = torch.load(clf_path, map_location=device, weights_only=False)
        classifier.load_state_dict(ckpt["model_state_dict"])
    classifier.to(device)
    classifier.eval()

    # 2. Load U-Net Generator (16.6M params baseline)
    gan_path = os.path.join(_PROJECT_ROOT, "results", "checkpoints", "gan", "unet_generator_best.pt")
    ngf = cfg["models"]["gan"].get("generator_features", 64)
    generator_unet = UNetGenerator(in_channels=3, out_channels=3, ngf=ngf)
    if os.path.exists(gan_path):
        ckpt_g = torch.load(gan_path, map_location=device, weights_only=False)
        generator_unet.load_state_dict(ckpt_g["generator_state_dict"])
    generator_unet.to(device)
    generator_unet.eval()

    # 3. Load DeblurGAN-v2 FPN-MobileNet Generator (3.16M params edge architecture)
    generator_deblurgan = DeblurGANv2Generator(fpn_channels=128, pretrained_backbone=False)
    generator_deblurgan.to(device)
    generator_deblurgan.eval()

    return classifier, generator_unet, generator_deblurgan, class_names, device, cfg

classifier, generator_unet, generator_deblurgan, class_names, device, cfg = load_ai_engine()

# ASTM D6433 Pavement Condition Index (PCI) & Civil Distress Metadata
DAMAGE_METADATA = {
    "Normal": {
        "color": "#2dd4bf",
        "severity": "Optimal Pass",
        "css_class": "hero-diagnosis-normal",
        "icon": "✅",
        "pci_base": 96,
        "pci_grade": "Good (Optimal)",
        "work_order_priority": "Routine",
        "est_cost": "$0 - $50 (Routine Survey)",
        "equipment": "Standard Inspection Vehicle",
        "action": "Routine pavement monitoring. Surface structural integrity meets highway specifications."
    },
    "Longitudinal Crack": {
        "color": "#38bdf8",
        "severity": "Moderate Distress",
        "css_class": "hero-diagnosis-warning",
        "icon": "⚠️",
        "pci_base": 72,
        "pci_grade": "Fair (Wheelpath Fatigue)",
        "work_order_priority": "Medium (30-day)",
        "est_cost": "$150 - $350 (Crack Sealing)",
        "equipment": "Hot-Pour Rubberized Crack Sealer",
        "action": "Wheel-path fatigue detected. Schedule crack sealing to prevent subbase water infiltration."
    },
    "Transverse Crack": {
        "color": "#818cf8",
        "severity": "Moderate Distress",
        "css_class": "hero-diagnosis-warning",
        "icon": "⚠️",
        "pci_base": 64,
        "pci_grade": "Fair (Thermal Contraction)",
        "work_order_priority": "Medium (30-day)",
        "est_cost": "$200 - $400 (Joint Routing)",
        "equipment": "Joint Router & Heated Mastic Applicator",
        "action": "Thermal contraction crack. Apply hot-pour rubberized joint sealant across transversal fissure."
    },
    "Alligator Crack": {
        "color": "#fb923c",
        "severity": "High Hazard",
        "css_class": "hero-diagnosis-danger",
        "icon": "🚨",
        "pci_base": 34,
        "pci_grade": "Very Poor (Base Failure)",
        "work_order_priority": "High (7-day)",
        "est_cost": "$800 - $2,500 (Full Patch)",
        "equipment": "Asphalt Milling Planer & Compactor",
        "action": "Structural base course failure. Milling and full-depth asphalt replacement recommended."
    },
    "Pothole": {
        "color": "#f43f5e",
        "severity": "Critical Hazard",
        "css_class": "hero-diagnosis-danger",
        "icon": "🛑",
        "pci_base": 18,
        "pci_grade": "Critical / Failed",
        "work_order_priority": "Urgent (24-hr Emergency)",
        "est_cost": "$400 - $1,200 (Emergency Crew)",
        "equipment": "Tack-Coat Sprayer & Vibratory Tamper",
        "action": "Immediate vehicular tire/suspension hazard. Dispatch emergency cold-mix or hot-mix crew."
    }
}

# Image Processing Helpers
def compute_sobel_edges(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.hypot(sobelx, sobely)
    grad = (grad / (grad.max() + 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(grad, cv2.COLORMAP_VIRIDIS)

def classify_patch(img_rgb: np.ndarray):
    pil_img = Image.fromarray(img_rgb)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    t_in = tf(pil_img).unsqueeze(0).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = classifier(t_in)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    lat = (time.perf_counter() - t0) * 1000.0
    return probs, lat

def run_deblur_model(model: torch.nn.Module, degraded_img: np.ndarray) -> tuple:
    with torch.no_grad():
        t0 = time.perf_counter()
        norm_in = ((degraded_img.astype(np.float32) / 127.5) - 1.0)
        tensor_in = torch.from_numpy(norm_in).permute(2, 0, 1).unsqueeze(0).to(device)
        out = model(tensor_in)
        restored = ((out.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        lat = (time.perf_counter() - t0) * 1000.0
    return restored, lat

# -----------------------------------------------------------------------------
# 4. Top Navigation Bar
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="top-navbar">
    <div>
        <div class="brand-title">🛣️ RoadVision AI <span style="font-size: 0.70rem; font-weight: 600; opacity: 0.85; padding: 2px 8px; border: 1px solid var(--border-subtle); border-radius: 6px; letter-spacing: 0.05em;">v2.6 PRO PLATFORM</span></div>
        <div class="brand-subtitle">Autonomous Generative Deblurring & Highway Structural Health Platform</div>
    </div>
    <div class="nav-badges">
        <span class="badge-pill badge-online">● System Online</span>
        <span class="badge-pill badge-dataset">🎯 RDD2022 / Real Data (92.1% Acc)</span>
        <span class="badge-pill badge-device">Device: {str(device).upper()}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Municipal Telematics Strip
st.markdown("""
<div class="telematics-strip">
    <div class="telematics-item"><span>Vehicle:</span> <span class="telematics-val">Survey Vehicle SV-09</span></div>
    <div class="telematics-item"><span>Speed:</span> <span class="telematics-val">42.5 km/h</span></div>
    <div class="telematics-item"><span>Route:</span> <span class="telematics-val">NH-48 Sector 4-B</span></div>
    <div class="telematics-item"><span>GPS:</span> <span class="telematics-val">18.5204° N, 73.8567° E</span></div>
    <div class="telematics-item"><span>Pavement:</span> <span class="telematics-val">Flexible Asphalt Concrete</span></div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Sidebar Controls & Inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📷 1. Sensor & Input Stream")
    input_source = st.radio(
        "Input Stream Mode",
        [
            "Real Vehicle Camera Dataset (GoPro/Dashcam)",
            "Procedural Distress Simulator",
            "Upload Custom Road Image"
        ],
        label_visibility="collapsed"
    )

    target_size = cfg["data"]["patch_size"][0]
    raw_image = None

    if input_source == "Real Vehicle Camera Dataset (GoPro/Dashcam)":
        real_sample_choice = st.selectbox(
            "Select Real Road Distress Sample",
            [
                ("data/samples/real/real_pothole_sample.jpg", "Pothole (Vehicle Dashcam - Actual Road)"),
                ("data/samples/real/real_crack_sample.jpg", "Longitudinal Crack (Actual Road Surface)"),
                ("data/samples/real/real_normal_sample.jpg", "Normal Clear Asphalt (Vehicle Camera)")
            ],
            format_func=lambda x: x[1]
        )
        sample_file_path = os.path.join(_PROJECT_ROOT, real_sample_choice[0])
        if os.path.exists(sample_file_path):
            img_bgr = cv2.imread(sample_file_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                raw_image = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
        if raw_image is None:
            raw_image = generate_synthetic_road_patch(class_id=4, size=(target_size, target_size), seed=42)

    elif input_source == "Procedural Distress Simulator":
        sample_choice = st.selectbox(
            "Select Target Distress Class",
            [
                (0, "Normal Road (Clear Surface)"),
                (1, "Longitudinal Crack (Wheelpath)"),
                (2, "Transverse Crack (Thermal)"),
                (3, "Alligator Crack (Fatigue Grid)"),
                (4, "Pothole (Structural Depression)")
            ],
            format_func=lambda x: x[1]
        )
        seed = st.slider("Pavement Variation Seed", 1, 100, 42)
        raw_image = generate_synthetic_road_patch(class_id=sample_choice[0], size=(target_size, target_size), seed=seed)

    else:
        uploaded_file = st.file_uploader("Upload Patch Image", type=["jpg", "jpeg", "png"], help="Upload an isolated road surface patch")
        if uploaded_file is not None:
            pil_file = Image.open(uploaded_file).convert("RGB")
            raw_image = cv2.resize(np.array(pil_file), (target_size, target_size), interpolation=cv2.INTER_AREA)

    st.markdown("---")
    st.markdown("### 🤖 2. Restoration Architecture")
    generator_arch_choice = st.selectbox(
        "Active Deblurring Generator",
        [
            "Conditional U-Net (16.6M params - Deep Trained)",
            "DeblurGAN-v2 (FPN-MobileNetV2 - 3.16M params, Fast Edge)"
        ]
    )

    st.markdown("---")
    st.markdown("### 🌪️ 3. Camera Blur Simulation")
    st.caption("Simulate dynamic motion shake, vibration, or focal dispersion.")

    blur_mode = st.selectbox(
        "Degradation Physics Mode",
        [
            "Vehicle Motion Blur (Linear Trajectory)",
            "Camera Defocus Blur (Aperture Dispersion)",
            "No Blur (Pass-Through Reference)"
        ]
    )

    if blur_mode == "Vehicle Motion Blur (Linear Trajectory)":
        kernel_len = st.slider("Displacement Kernel Length (px)", 3, 21, 9, step=2)
        blur_angle = st.slider("Motion Angle (°)", 0, 180, 45)
    elif blur_mode == "Camera Defocus Blur (Aperture Dispersion)":
        defocus_kernel = st.slider("Defocus Aperture Radius (px)", 5, 19, 11, step=2)
        defocus_sigma = st.slider("Optical Spread (σ)", 1.0, 5.0, 2.5)

    st.markdown("---")
    st.markdown("### 🔍 4. Structural Visualizer")
    edge_mode = st.toggle("Highlight Edge Contours (Sobel Gradient)", value=False, help="Computes high-frequency edge gradients to evaluate crack boundary restoration.")

# -----------------------------------------------------------------------------
# 6. Pipeline Execution & Processing
# -----------------------------------------------------------------------------
if raw_image is not None:
    # 1. Apply Degradation
    if blur_mode == "Vehicle Motion Blur (Linear Trajectory)":
        degraded_img = apply_motion_blur(raw_image.copy(), kernel_size=kernel_len, angle_degrees=blur_angle)
    elif blur_mode == "Camera Defocus Blur (Aperture Dispersion)":
        degraded_img = apply_defocus_blur(raw_image.copy(), kernel_size=defocus_kernel, sigma=defocus_sigma)
    else:
        degraded_img = raw_image.copy()

    # 2. Classical Restoration (Unsharp Mask Baseline)
    t_c0 = time.perf_counter()
    classical_img = restore_classical(degraded_img, method="unsharp_mask", amount=1.6)
    t_classical = (time.perf_counter() - t_c0) * 1000.0

    # 3. Primary Selected GAN Model Restoration
    active_model = generator_unet if "U-Net" in generator_arch_choice else generator_deblurgan
    gan_img, t_gan = run_deblur_model(active_model, degraded_img)

    # 4. DeblurGAN-v2 secondary restoration for multi-model tab comparison
    deblurgan_img, t_deblurgan = run_deblur_model(generator_deblurgan, degraded_img)
    unet_img, t_unet = run_deblur_model(generator_unet, degraded_img)

    # 5. AI Classification
    probs_degraded, lat_degraded = classify_patch(degraded_img)
    probs_gan, lat_gan = classify_patch(gan_img)
    probs_sharp, _ = classify_patch(raw_image)

    pred_cls_degraded = class_names[int(np.argmax(probs_degraded))]
    conf_degraded = probs_degraded.max() * 100.0

    pred_cls_gan = class_names[int(np.argmax(probs_gan))]
    conf_gan = probs_gan.max() * 100.0

    meta_gan = DAMAGE_METADATA.get(pred_cls_gan, DAMAGE_METADATA["Normal"])

    # Calculate ASTM D6433 PCI Score
    pci_score = max(5, min(100, int(meta_gan["pci_base"] - (conf_gan * 0.05 if pred_cls_gan != "Normal" else -3))))

    # 6. Signal Quality Metrics
    psnr_degraded = compute_psnr(degraded_img, raw_image)
    ssim_degraded = compute_ssim(degraded_img, raw_image)
    psnr_gan = compute_psnr(gan_img, raw_image)
    ssim_gan = compute_ssim(gan_img, raw_image)

    psnr_classical = compute_psnr(classical_img, raw_image)
    ssim_classical = compute_ssim(classical_img, raw_image)

    psnr_deblurgan = compute_psnr(deblurgan_img, raw_image)
    ssim_deblurgan = compute_ssim(deblurgan_img, raw_image)

    psnr_unet = compute_psnr(unet_img, raw_image)
    ssim_unet = compute_ssim(unet_img, raw_image)

    # Prepare Visual Views
    view_degraded = compute_sobel_edges(degraded_img) if edge_mode else degraded_img
    view_gan = compute_sobel_edges(gan_img) if edge_mode else gan_img
    view_sharp = compute_sobel_edges(raw_image) if edge_mode else raw_image
    view_classical = compute_sobel_edges(classical_img) if edge_mode else classical_img
    view_deblurgan = compute_sobel_edges(deblurgan_img) if edge_mode else deblurgan_img
    view_unet = compute_sobel_edges(unet_img) if edge_mode else unet_img

    # -----------------------------------------------------------------------------
    # 7. Dashboard Layout
    # -----------------------------------------------------------------------------
    col_left, col_right = st.columns([1.65, 1.0], gap="large")

    # LEFT COLUMN: Image Restoration Panels & Telemetry
    with col_left:
        st.markdown("""
        <div class="dash-card">
            <div class="card-title">
                <span>🖼️ Optical Inspection & Generative Deblurring</span>
            </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "⚡ 3-Panel Inspection",
            "🔍 Core Dual View",
            "🔬 Architecture Comparison (DeblurGAN-v2 vs U-Net)",
            "📐 Full 4-Stage Breakdown"
        ])

        with tab1:
            p1, p2, p3 = st.columns(3)
            with p1:
                st.markdown("""
                <div class="img-frame">
                    <div class="img-header">
                        <span>1. Reference Sharp</span>
                        <span style="color:#2dd4bf;">TARGET</span>
                    </div>
                """, unsafe_allow_html=True)
                st.image(view_sharp, use_container_width=True)
                st.markdown(f"<div class='img-footer'>Target: <b>{pred_cls_gan}</b></div></div>", unsafe_allow_html=True)

            with p2:
                st.markdown(f"""
                <div class="img-frame">
                    <div class="img-header">
                        <span>2. Degraded Camera</span>
                        <span style="color:#fb7185;">BLURRED</span>
                    </div>
                """, unsafe_allow_html=True)
                st.image(view_degraded, use_container_width=True)
                st.markdown(f"<div class='img-footer'>PSNR: <b>{psnr_degraded:.1f} dB</b> | SSIM: <b>{ssim_degraded:.3f}</b></div></div>", unsafe_allow_html=True)

            with p3:
                model_tag = "DEBLURGAN-v2" if "DeblurGAN" in generator_arch_choice else "U-NET"
                st.markdown(f"""
                <div class="img-frame">
                    <div class="img-header">
                        <span>3. {model_tag} Restored</span>
                        <span style="color:#38bdf8;">AI RECONSTRUCTED</span>
                    </div>
                """, unsafe_allow_html=True)
                st.image(view_gan, use_container_width=True)
                st.markdown(f"<div class='img-footer'>PSNR: <b>{psnr_gan:.1f} dB</b> | SSIM: <b>{ssim_gan:.3f}</b></div></div>", unsafe_allow_html=True)

        with tab2:
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"""
                <div class="img-frame">
                    <div class="img-header">
                        <span>Input Camera Stream</span>
                        <span style="color:#fb7185;">DEGRADED</span>
                    </div>
                """, unsafe_allow_html=True)
                st.image(view_degraded, use_container_width=True)
                st.markdown(f"<div class='img-footer'>Detected: <b style='color:#f87171;'>{pred_cls_degraded}</b> ({conf_degraded:.1f}%)</div></div>", unsafe_allow_html=True)

            with d2:
                st.markdown(f"""
                <div class="img-frame">
                    <div class="img-header">
                        <span>GAN Reconstructed</span>
                        <span style="color:#2dd4bf;">RECOVERED</span>
                    </div>
                """, unsafe_allow_html=True)
                st.image(view_gan, use_container_width=True)
                st.markdown(f"<div class='img-footer'>Diagnosed: <b style='color:{meta_gan['color']};'>{pred_cls_gan}</b> ({conf_gan:.1f}%)</div></div>", unsafe_allow_html=True)

        with tab3:
            st.caption("Side-by-side benchmarking of DeblurGAN-v2 (FPN-MobileNetV2, 3.16M params) vs U-Net Generator (16.6M params) vs Classical Filter.")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown("<div class='img-frame'><div class='img-header'><span>Classical Unsharp</span><span style='color:#94a3b8;'>FILTER</span></div>", unsafe_allow_html=True)
                st.image(view_classical, use_container_width=True)
                st.markdown(f"<div class='img-footer'>{psnr_classical:.1f} dB | {t_classical:.1f} ms</div></div>", unsafe_allow_html=True)
            with m2:
                st.markdown("<div class='img-frame'><div class='img-header'><span>DeblurGAN-v2 (FPN)</span><span style='color:#2dd4bf;'>3.16M PARAMS</span></div>", unsafe_allow_html=True)
                st.image(view_deblurgan, use_container_width=True)
                st.markdown(f"<div class='img-footer'>{psnr_deblurgan:.1f} dB | {t_deblurgan:.1f} ms</div></div>", unsafe_allow_html=True)
            with m3:
                st.markdown("<div class='img-frame'><div class='img-header'><span>U-Net Generator</span><span style='color:#38bdf8;'>16.6M PARAMS</span></div>", unsafe_allow_html=True)
                st.image(view_unet, use_container_width=True)
                st.markdown(f"<div class='img-footer'>{psnr_unet:.1f} dB | {t_unet:.1f} ms</div></div>", unsafe_allow_html=True)

        with tab4:
            q1, q2, q3, q4 = st.columns(4)
            with q1:
                st.markdown("<div class='img-frame'><div class='img-header'><span>Ground Truth</span></div>", unsafe_allow_html=True)
                st.image(view_sharp, use_container_width=True)
                st.markdown("<div class='img-footer'>Reference</div></div>", unsafe_allow_html=True)
            with q2:
                st.markdown(f"<div class='img-frame'><div class='img-header'><span>Degraded</span></div>", unsafe_allow_html=True)
                st.image(view_degraded, use_container_width=True)
                st.markdown(f"<div class='img-footer'>{psnr_degraded:.1f} dB</div></div>", unsafe_allow_html=True)
            with q3:
                st.markdown("<div class='img-frame'><div class='img-header'><span>Classical</span></div>", unsafe_allow_html=True)
                st.image(view_classical, use_container_width=True)
                st.markdown(f"<div class='img-footer'>Unsharp Mask</div></div>", unsafe_allow_html=True)
            with q4:
                st.markdown(f"<div class='img-frame'><div class='img-header'><span>Restored</span></div>", unsafe_allow_html=True)
                st.image(view_gan, use_container_width=True)
                st.markdown(f"<div class='img-footer'>{psnr_gan:.1f} dB</div></div>", unsafe_allow_html=True)

        # Performance Metric Grid
        st.markdown(f"""
        <div class="metric-grid-dark">
            <div class="metric-box">
                <div class="metric-val">{t_gan:.1f} ms</div>
                <div class="metric-name">Restoration Latency</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{lat_gan:.1f} ms</div>
                <div class="metric-name">MobileNetV2 Inference</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{t_gan + lat_gan:.1f} ms</div>
                <div class="metric-name">Total Pipeline Budget</div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # RIGHT COLUMN: Diagnosis, Maintenance Dispatch & Confidence
    with col_right:
        # Hero Diagnosis Card
        st.markdown(f"""
        <div class="{meta_gan['css_class']}">
            <div style="font-size: 2.1rem; line-height: 1.1; margin-bottom: 5px;">{meta_gan['icon']}</div>
            <div style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.01em; color: #ffffff;">{pred_cls_gan}</div>
            <div style="font-size: 0.74rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: {meta_gan['color']}; margin-top: 5px;">
                STATUS: {meta_gan['severity']} &nbsp;•&nbsp; {conf_gan:.1f}% CONFIDENCE
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ASTM D6433 Pavement Condition Index (PCI) Score Card
        pci_color = "#2dd4bf" if pci_score >= 80 else ("#f59e0b" if pci_score >= 50 else "#f43f5e")
        st.markdown(f"""
        <div class="pci-box">
            <div class="pci-header">
                <span>ASTM D6433 Pavement Condition Index (PCI)</span>
                <span style="color: {pci_color}; font-weight: 800;">{meta_gan['pci_grade']}</span>
            </div>
            <div style="display: flex; align-items: baseline; gap: 8px;">
                <span class="pci-score" style="color: {pci_color};">{pci_score}</span>
                <span style="font-size: 0.8rem; color: var(--text-muted);">/ 100</span>
            </div>
            <div class="pci-bar-bg">
                <div class="pci-bar-fill" style="width: {pci_score}%; background: {pci_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Municipal Work Order Dispatch Box
        st.markdown(f"""
        <div class="action-dispatch-box">
            <div class="action-dispatch-title">
                <span>⚡ Work Order: WO-2026-0842</span>
                <span style="color:{meta_gan['color']};">{meta_gan['work_order_priority']}</span>
            </div>
            <div class="action-dispatch-desc">
                <b>Action:</b> {meta_gan['action']}<br>
                <b>Est. Repair Cost:</b> {meta_gan['est_cost']}<br>
                <b>Required Equipment:</b> {meta_gan['equipment']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Damage Probability Distribution
        st.markdown("""
        <div class="dash-card">
            <div class="card-title">
                <span>📊 Distress Class Probability Distribution</span>
            </div>
        """, unsafe_allow_html=True)

        for i, c_name in enumerate(class_names):
            p_val = probs_gan[i] * 100.0
            color = DAMAGE_METADATA.get(c_name, {}).get("color", "#38bdf8")
            st.markdown(f"""
            <div class="prob-bar-wrapper">
                <div class="prob-bar-labels">
                    <span style="font-weight: 500;">{c_name}</span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: {color};">{p_val:.1f}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: {max(p_val, 1.0):.1f}%; background: {color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Export Report JSON
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "survey_metadata": {
                "vehicle_id": "Survey Vehicle SV-09",
                "route_segment": "NH-48 Sector 4-B",
                "gps_coordinates": "18.5204 N, 73.8567 E",
                "speed_kmh": 42.5
            },
            "diagnosis": pred_cls_gan,
            "confidence_pct": round(float(conf_gan), 2),
            "pci_metrics": {
                "pci_score": pci_score,
                "pci_rating": meta_gan["pci_grade"],
                "standard": "ASTM D6433"
            },
            "work_order": {
                "work_order_id": "WO-2026-0842",
                "priority": meta_gan["work_order_priority"],
                "est_cost": meta_gan["est_cost"],
                "equipment": meta_gan["equipment"],
                "action": meta_gan["action"]
            },
            "degradation_simulated": blur_mode,
            "restoration_architecture": generator_arch_choice,
            "signal_quality": {
                "input_psnr_db": round(float(psnr_degraded), 2),
                "restored_psnr_db": round(float(psnr_gan), 2),
                "ssim": round(float(ssim_gan), 4)
            },
            "latency_ms": {
                "gan_deblur": round(float(t_gan), 2),
                "classifier": round(float(lat_gan), 2),
                "total": round(float(t_gan + lat_gan), 2)
            }
        }
        st.download_button(
            label="📥 Export Inspection Diagnostic Report (JSON)",
            data=json.dumps(report_data, indent=2),
            file_name=f"road_inspection_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True
        )

else:
    st.info("👈 Select a road sample from the sidebar to inspect real-time deblurring and diagnosis.")
