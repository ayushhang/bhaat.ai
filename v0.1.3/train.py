"""
Training script for the Food Recognition Model.

Uses the Indian Food Images dataset from the archive folder.
Fine-tunes a pretrained ResNet-18 on 80 Indian food categories.

Usage:
    python train.py                         # train with defaults
    python train.py --epochs 20             # custom epoch count
    python train.py --batch-size 16         # smaller batch for low RAM
"""

import ssl
import certifi
import os

# Fix SSL certificate issue on macOS
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import build_model


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.join(
    os.path.dirname(__file__), "..", "archive",
    "Indian Food Images", "Indian Food Images"
)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(epochs=10, batch_size=32, lr=0.001, val_split=0.2):
    # Check dataset exists
    if not os.path.isdir(DATASET_DIR):
        print(f"ERROR: Dataset not found at {DATASET_DIR}")
        print("Make sure the 'archive/Indian Food Images/Indian Food Images/' folder exists.")
        return

    # Load full dataset (just to read class names and split)
    full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    print(f"Found {len(full_dataset)} images across {num_classes} classes")
    print(f"Classes: {', '.join(c.replace('_', ' ').title() for c in class_names[:10])}...")

    # Train / validation split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Override transform for validation subset
    # (random_split returns Subset objects that still use the parent transform,
    #  so we rebuild the val set with val_transform)
    val_dataset_proper = datasets.ImageFolder(DATASET_DIR, transform=val_transform)
    val_dataset_proper = torch.utils.data.Subset(val_dataset_proper, val_dataset.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset_proper, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = build_model(num_classes).to(device)

    # Freeze backbone initially for 2 epochs, then unfreeze
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Unfreeze backbone after epoch 2
        if epoch == 2:
            print(">> Unfreezing backbone layers for fine-tuning")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=lr * 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # --- Train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        train_acc = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "food_resnet18.pth"))
            with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
                json.dump(class_names, f)
            print(f"  >> Saved best model (Val Acc: {val_acc:.2f}%)")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {MODEL_DIR}/food_resnet18.pth")
    print(f"Class names saved to {MODEL_DIR}/class_names.json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train food recognition model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")

    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, val_split=args.val_split)
