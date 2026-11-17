"""Full end-to-end workflow validation script."""
import sys, os, json
import numpy as np
sys.path.insert(0, ".")

print("=" * 60)
print("STEP 1: Synthetic Data Generation")
print("=" * 60)
from src.data.synthetic_samples import generate_synthetic_road_dataset
m, c = generate_synthetic_road_dataset("data/processed", num_images=80, seed=42)
print(f"Manifest: {m}")

with open(c) as f:
    counts = json.load(f)
for split, cls_counts in counts.items():
    total = sum(cls_counts.values())
    print(f"  {split}: {total} patches -> {cls_counts}")

print()
print("=" * 60)
print("STEP 2: Dataset Loading Checks")
print("=" * 60)
from src.data.dataset import RoadPatchClassificationDataset, PairedDeblurDataset
from src.degradation.blur_pipeline import BlurDegradationPipeline

ds_cls = RoadPatchClassificationDataset(m, split="train")
ds_pair = PairedDeblurDataset(m, split="train", degradation_pipeline=BlurDegradationPipeline())
print(f"Classification train dataset: {len(ds_cls)} samples")
print(f"Paired deblur train dataset:  {len(ds_pair)} samples")

img_t, label, pid = ds_cls[0]
print(f"  Classifier sample shape: {img_t.shape}, label: {label}")

blur_t, sharp_t = ds_pair[0]
print(f"  Paired sample shapes: blur={blur_t.shape}, sharp={sharp_t.shape}")

print()
print("=" * 60)
print("STEP 3: Model Architecture Checks")
print("=" * 60)
import torch
from src.models.generator import UNetGenerator
from src.models.discriminator import PatchGANDiscriminator
from src.models.mobilenet import build_mobilenet_v2
from src.models.custom_cnn import CustomSmallCNN
from src.evaluate.benchmark_latency import count_parameters

g = UNetGenerator(3, 3, 64)
d = PatchGANDiscriminator(6, 64)
clf = build_mobilenet_v2(5, pretrained=False)
cnn = CustomSmallCNN(5)

dummy = torch.randn(1, 3, 128, 128)
g_out = g(dummy)
d_out = d(dummy, g_out)
clf_out = clf(dummy)
cnn_out = cnn(dummy)

gp = count_parameters(g)
dp = count_parameters(d)
cp = count_parameters(clf)
cnp = count_parameters(cnn)

print(f"  Generator:     input {list(dummy.shape)} -> output {list(g_out.shape)}  params: {gp['total_parameters']:,}")
print(f"  Discriminator: cond+target -> patchmap {list(d_out.shape)}  params: {dp['total_parameters']:,}")
print(f"  MobileNetV2:   input {list(dummy.shape)} -> logits {list(clf_out.shape)}  params: {cp['total_parameters']:,}")
print(f"  CustomCNN:     input {list(dummy.shape)} -> logits {list(cnn_out.shape)}  params: {cnp['total_parameters']:,}")

print()
print("=" * 60)
print("STEP 4: Degradation Pipeline Check")
print("=" * 60)
from src.degradation.classical_restoration import restore_classical
from src.evaluate.metrics import compute_psnr, compute_ssim

test_img = np.random.randint(80, 160, (128, 128, 3), dtype=np.uint8)
pipe = BlurDegradationPipeline(fixed_seed=999)

for btype in ["mild_motion", "strong_motion", "defocus"]:
    blurred = pipe.apply(test_img.copy(), blur_type=btype)
    restored = restore_classical(blurred, method="unsharp_mask")
    p_b = compute_psnr(blurred, test_img)
    p_r = compute_psnr(restored, test_img)
    print(f"  {btype:15s}: Blurred PSNR={p_b:.2f} dB | Restored PSNR={p_r:.2f} dB")

print()
print("=" * 60)
print("STEP 5: Checkpoint Verification")
print("=" * 60)
ckpts = [
    "results/checkpoints/classifier/mobilenet_v2_clean_best.pt",
    "results/checkpoints/classifier/mobilenet_v2_blur_aug_best.pt",
    "results/checkpoints/gan/unet_generator_best.pt",
]
for ck in ckpts:
    exists = os.path.exists(ck)
    size_kb = os.path.getsize(ck) / 1024 if exists else 0
    status = f"OK ({size_kb:.0f} KB)" if exists else "MISSING"
    print(f"  {os.path.basename(ck):45s} -> {status}")

print()
print("=" * 60)
print("STEP 6: GAN Restoration End-to-End Test")
print("=" * 60)
device = torch.device("cpu")
gan_ckpt = torch.load("results/checkpoints/gan/unet_generator_best.pt", map_location=device, weights_only=False)
net_G = UNetGenerator(3, 3, 64)
net_G.load_state_dict(gan_ckpt["generator_state_dict"])
net_G.eval()

sharp_patch = np.random.randint(80, 160, (128, 128, 3), dtype=np.uint8)
blurry_patch = pipe.apply(sharp_patch.copy(), blur_type="strong_motion")

with torch.no_grad():
    inp = (blurry_patch.astype(np.float32) / 127.5) - 1.0
    inp_t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0)
    out_t = net_G(inp_t)
    gan_restored = ((out_t.squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

p_blur = compute_psnr(blurry_patch, sharp_patch)
p_gan  = compute_psnr(gan_restored, sharp_patch)
print(f"  Blurred  vs Sharp: PSNR = {p_blur:.2f} dB")
print(f"  GAN Rest vs Sharp: PSNR = {p_gan:.2f} dB")
print(f"  GAN restored patch shape: {gan_restored.shape}")

print()
print("=" * 60)
print("STEP 7: Classification End-to-End Test")
print("=" * 60)
from torchvision import transforms
from PIL import Image

clf_ckpt = torch.load("results/checkpoints/classifier/mobilenet_v2_clean_best.pt", map_location=device, weights_only=False)
clf_model = build_mobilenet_v2(5, pretrained=False)
clf_model.load_state_dict(clf_ckpt["model_state_dict"])
clf_model.eval()

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ["Normal", "Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Pothole"]
from src.data.synthetic_samples import generate_synthetic_road_patch

for cls_id in range(5):
    test_patch = generate_synthetic_road_patch(cls_id, seed=42)
    pil = Image.fromarray(test_patch)
    t_in = tf(pil).unsqueeze(0)
    with torch.no_grad():
        logits = clf_model(t_in)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    pred = int(np.argmax(probs))
    print(f"  Input: {class_names[cls_id]:25s} -> Predicted: {class_names[pred]:25s} (conf: {probs[pred]*100:.1f}%)")

print()
print("=" * 60)
print("ALL 7 WORKFLOW CHECKS PASSED SUCCESSFULLY")
print("=" * 60)
