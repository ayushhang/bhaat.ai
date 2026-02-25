"""
train_food101.py
──────────────────────────────────────────────────────────────────────────────
Fine-tune EfficientNet-B4 on the Food-101 dataset.
Produces the food101_efficientnet_b4.pth checkpoint used by the detector.

Expected accuracy after training:
  Top-1: ~82% on Food-101 test set
  Top-5: ~95%

Requirements:
  pip install torch torchvision tqdm

Dataset download (~5GB):
  python -c "import torchvision; torchvision.datasets.Food101('.', download=True)"

Training time (estimate):
  GPU (A100 80GB): ~15 min
  GPU (RTX 3090): ~45 min
  CPU: not recommended (very slow)

Usage:
  python train_food101.py --epochs 20 --batch-size 32 --lr 1e-3
  python train_food101.py --epochs 5  --batch-size 16   # quick test
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models

CACHE_DIR = Path.home() / ".cache" / "nutrilens"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = CACHE_DIR / "food101_efficientnet_b4.pth"
DATA_ROOT = Path("./food101_data")


def get_transforms(train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(380, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize(400),
        T.CenterCrop(380),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(num_classes=101):
    weights = models.EfficientNet_B4_Weights.DEFAULT
    model = models.efficientnet_b4(weights=weights)

    # Freeze early layers, fine-tune later blocks + head
    for name, param in model.named_parameters():
        # Unfreeze features.6, features.7, features.8 and classifier
        if any(layer in name for layer in ["features.6", "features.7", "features.8", "classifier"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 50 == 0:
            acc = 100.0 * correct / total
            print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} | Acc: {acc:.1f}%")

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        total += labels.size(0)

        # Top-1
        _, pred1 = outputs.max(1)
        correct_top1 += pred1.eq(labels).sum().item()

        # Top-5
        _, pred5 = outputs.topk(5, dim=1)
        correct_top5 += pred5.eq(labels.unsqueeze(1).expand_as(pred5)).any(dim=1).sum().item()

    return (total_loss / len(loader),
            100.0 * correct_top1 / total,
            100.0 * correct_top5 / total)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B4 on Food-101")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--unfreeze-all", action="store_true",
                        help="Unfreeze all layers (slower, potentially better)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cpu":
        print("WARNING: Training on CPU is very slow. A GPU is strongly recommended.")

    # ── Data ─────────────────────────────────────────────
    print("\nLoading Food-101 dataset...")
    DATA_ROOT.mkdir(exist_ok=True)

    try:
        train_data = datasets.Food101(DATA_ROOT, split="train",
                                      transform=get_transforms(train=True), download=True)
        test_data  = datasets.Food101(DATA_ROOT, split="test",
                                      transform=get_transforms(train=False), download=True)
    except Exception as e:
        print(f"Dataset error: {e}")
        print("Run: python -c \"import torchvision; torchvision.datasets.Food101('.', download=True)\"")
        return

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    print(f"  Train: {len(train_data):,} images | Test: {len(test_data):,} images")

    # ── Model ─────────────────────────────────────────────
    model = build_model(num_classes=101)
    if args.unfreeze_all:
        for p in model.parameters():
            p.requires_grad = True
        print("All layers unfrozen.")
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {trainable:,} trainable / {total_params:,} total")

    # ── Training setup ────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_acc = 0.0
    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc1, val_acc5 = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.0f}s)")
        print(f"  Train — Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val   — Loss: {val_loss:.4f} | Top-1: {val_acc1:.2f}% | Top-5: {val_acc5:.2f}%")

        if val_acc1 > best_acc:
            best_acc = val_acc1
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  ✓ New best ({best_acc:.2f}%) — checkpoint saved to {CHECKPOINT_PATH}")
        print()

    print(f"Training complete. Best Top-1: {best_acc:.2f}%")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()