"""
Food Recognition Model - ResNet-18 based CNN
Uses a pretrained ResNet-18 backbone with a custom classification head
fine-tuned on the Indian Food dataset (80 classes).
"""

import ssl
import certifi
import os
import json

# Fix SSL certificate issue on macOS
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# Image preprocessing pipeline (must match training)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def build_model(num_classes, pretrained=True):
    """
    Build a ResNet-18 model with a custom classification head.

    - Loads pretrained ImageNet weights for the CNN backbone (if pretrained=True)
    - Replaces the final fully-connected layer to match our food classes
    """
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)

    # Replace the final FC layer
    # ResNet-18 fc layer input: 512 features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


def load_class_names(model_dir="models"):
    """Load the class name mapping saved during training."""
    path = os.path.join(model_dir, "class_names.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_trained_model(model_dir="models", device=None):
    """
    Load a trained model from disk.

    Returns (model, class_names) or (None, None) if no trained model found.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_class_names(model_dir)
    if class_names is None:
        return None, None

    weights_path = os.path.join(model_dir, "food_resnet18.pth")
    if not os.path.exists(weights_path):
        return None, None

    num_classes = len(class_names)
    model = build_model(num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model, class_names


def predict(image_path, model, class_names, device=None, top_k=5):
    """
    Run inference on a single image.

    Args:
        image_path: path to the image file
        model: trained PyTorch model
        class_names: list of class name strings
        device: torch device
        top_k: number of top predictions to return

    Returns:
        list of (class_name, confidence) tuples, sorted by confidence desc
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = TRANSFORM(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, min(top_k, len(class_names)))
    top_probs = top_probs.squeeze().cpu().tolist()
    top_indices = top_indices.squeeze().cpu().tolist()

    # Handle single prediction case
    if not isinstance(top_probs, list):
        top_probs = [top_probs]
        top_indices = [top_indices]

    results = []
    for prob, idx in zip(top_probs, top_indices):
        name = class_names[idx].replace("_", " ").title()
        results.append((name, round(prob * 100, 2)))

    return results
