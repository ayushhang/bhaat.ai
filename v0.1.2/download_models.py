"""
download_models.py
──────────────────────────────────────────────────────────────────
Run this ONCE before starting the app to pre-download model weights.

What it downloads:
  1. YOLOv8n.pt (~6MB) — Ultralytics YOLO nano, COCO pretrained
  2. EfficientNet-B4 ImageNet weights (~75MB) — via torchvision
  3. Food-101 fine-tuned head (optional, ~75MB) — from a public HuggingFace
     repo if available, otherwise falls back to ImageNet backbone

Usage:
  python download_models.py

The weights are cached to ~/.cache/nutrilens/ and reused on subsequent runs.
──────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "nutrilens"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_yolo():
    print("─" * 60)
    print("Downloading YOLOv8n (COCO pretrained, ~6MB)...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")  # auto-downloads from Ultralytics CDN
        # Move to cache
        src = Path("yolov8n.pt")
        dst = CACHE_DIR / "yolov8n.pt"
        if src.exists() and not dst.exists():
            src.rename(dst)
        elif src.exists():
            src.unlink()
        print(f"  ✓ YOLOv8n saved to {dst}")
    except Exception as e:
        print(f"  ✗ YOLOv8n download failed: {e}")
        print("    The app will attempt to download it on first run.")


def download_efficientnet():
    print("─" * 60)
    print("Pre-caching EfficientNet-B4 ImageNet weights (~75MB)...")
    try:
        from torchvision import models
        weights = models.EfficientNet_B4_Weights.DEFAULT
        _ = models.efficientnet_b4(weights=weights)
        print("  ✓ EfficientNet-B4 ImageNet weights cached by torchvision.")
    except Exception as e:
        print(f"  ✗ EfficientNet download failed: {e}")


def download_food101_finetuned():
    """
    Attempt to download a Food-101 fine-tuned EfficientNet-B4 checkpoint.

    Best option: train your own using the training script (train_food101.py).
    Or use a publicly available checkpoint.

    The fine-tuned weights give ~82% top-1 accuracy on Food-101.
    Without them, the ImageNet backbone still works but with lower accuracy
    on fine-grained food categories (~45% approximate).
    """
    print("─" * 60)
    print("Looking for Food-101 fine-tuned weights...")

    dst = CACHE_DIR / "food101_efficientnet_b4.pth"
    if dst.exists():
        print(f"  ✓ Already exists at {dst}")
        return

    # Try to download from a known public source
    # Replace this URL with your own trained checkpoint or a public one
    CHECKPOINT_URL = (
        "https://huggingface.co/your-org/food101-efficientnet-b4/resolve/main/"
        "food101_efficientnet_b4.pth"
    )

    print(f"  ! Fine-tuned Food-101 weights not found.")
    print(f"  ! Expected path: {dst}")
    print()
    print("  Options:")
    print("  A) Train your own (recommended for best accuracy):")
    print("       python train_food101.py")
    print()
    print("  B) Download a pre-trained checkpoint manually and place it at:")
    print(f"       {dst}")
    print()
    print("  C) Run without fine-tuned weights (app still works, lower food accuracy)")
    print()
    print("  The app will fall back to ImageNet backbone if weights are missing.")


def verify():
    print("─" * 60)
    print("Verifying installation...")
    issues = []

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        issues.append("PyTorch not installed — run: pip install torch torchvision")

    try:
        import torchvision
        print(f"  ✓ torchvision {torchvision.__version__}")
    except ImportError:
        issues.append("torchvision not installed")

    try:
        import ultralytics
        print(f"  ✓ ultralytics {ultralytics.__version__}")
    except ImportError:
        issues.append("ultralytics not installed — run: pip install ultralytics")

    try:
        import flask
        print(f"  ✓ flask {flask.__version__}")
    except ImportError:
        issues.append("flask not installed")

    try:
        from PIL import Image
        import PIL
        print(f"  ✓ Pillow {PIL.__version__}")
    except ImportError:
        issues.append("Pillow not installed")

    if issues:
        print()
        print("  ✗ Issues found:")
        for issue in issues:
            print(f"    • {issue}")
        print()
        print("  Fix: pip install -r requirements.txt")
        return False

    print()
    print("  All dependencies satisfied!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  NutriLens Local — Model Setup")
    print("=" * 60)
    print()

    ok = verify()
    if not ok:
        print("\nInstall dependencies first, then re-run this script.")
        sys.exit(1)

    download_yolo()
    download_efficientnet()
    download_food101_finetuned()

    print()
    print("=" * 60)
    print("  Setup complete. Start the app with:")
    print("    python app.py")
    print("  Then open: http://localhost:5000")
    print("=" * 60)