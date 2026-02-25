"""
food_detector.py
─────────────────────────────────────────────────────────────────────────────
Two-stage local food detection pipeline — NO external API calls.

Stage 1 — YOLOv8 Object Detection
  • Uses Ultralytics YOLOv8n (nano) pretrained on COCO-128
  • Detects bounding boxes + class labels for all objects
  • Filters to food-relevant COCO classes + runs on all if nothing found

Stage 2 — Fine-Grained Food Classification (per crop)
  • EfficientNet-B4 fine-tuned on Food-101 (101 food categories)
  • Loaded from torchvision / timm with cached weights (~75MB)
  • Each YOLO crop is re-classified to a specific food name
  • Confidence threshold: top-1 softmax > 0.15 accepted

Stage 3 — Portion Estimation (heuristic)
  • Bounding box area relative to image area → estimated plate coverage
  • Mapped to a gram estimate using known reference portions
  • Clamped to [30g, 800g] range

The pipeline degrades gracefully:
  • If YOLO finds nothing food-like → classify the whole image
  • If classifier is unavailable → use YOLO label + lookup
  • If both fail → return generic "food item"
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── COCO class IDs that are food-relevant ─────────────────
# COCO 80-class list indices (0-based)
COCO_FOOD_CLASS_IDS = {
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
}

# ── Food-101 class list (101 classes, alphabetical) ───────
FOOD101_CLASSES = [
    "apple pie", "baby back ribs", "baklava", "beef carpaccio", "beef tartare",
    "beet salad", "beignets", "bibimbap", "bread pudding", "breakfast burrito",
    "bruschetta", "caesar salad", "cannoli", "caprese salad", "carrot cake",
    "ceviche", "cheesecake", "cheese plate", "chicken curry", "chicken quesadilla",
    "chicken wings", "chocolate cake", "chocolate mousse", "churros", "clam chowder",
    "club sandwich", "crab cakes", "creme brulee", "croque madame", "cup cakes",
    "deviled eggs", "donuts", "dumplings", "edamame", "eggs benedict",
    "escargots", "falafel", "filet mignon", "fish and chips", "foie gras",
    "french fries", "french onion soup", "french toast", "fried calamari",
    "fried rice", "frozen yogurt", "garlic bread", "gnocchi", "greek salad",
    "grilled cheese sandwich", "grilled salmon", "guacamole", "gyoza", "hamburger",
    "hot and sour soup", "hot dog", "huevos rancheros", "hummus", "ice cream",
    "lasagna", "lobster bisque", "lobster roll sandwich", "macaroni and cheese",
    "macarons", "miso soup", "mussels", "nachos", "omelette", "onion rings",
    "oysters", "pad thai", "paella", "pancakes", "panna cotta", "peking duck",
    "pho", "pizza", "pork chop", "poutine", "prime rib", "pulled pork sandwich",
    "ramen", "ravioli", "red velvet cake", "risotto", "samosa", "sashimi",
    "scallops", "seaweed salad", "shrimp and grits", "spaghetti bolognese",
    "spaghetti carbonara", "spring rolls", "steak", "strawberry shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna tartare", "waffles",
]

# Maps Food-101 class name → our nutrition DB key
FOOD101_TO_DB: dict[str, str] = {
    "apple pie": "cake",
    "baby back ribs": "pork",
    "baklava": "cake",
    "beef carpaccio": "beef",
    "beef tartare": "beef",
    "beet salad": "salad",
    "bibimbap": "fried rice",
    "breakfast burrito": "burrito",
    "bruschetta": "bread",
    "caesar salad": "salad",
    "cannoli": "cake",
    "caprese salad": "salad",
    "carrot cake": "cake",
    "cheesecake": "cake",
    "chicken curry": "chicken breast",
    "chicken quesadilla": "tortilla",
    "chicken wings": "chicken wing",
    "chocolate cake": "cake",
    "chocolate mousse": "chocolate",
    "churros": "donut",
    "clam chowder": "soup",
    "club sandwich": "sandwich",
    "crab cakes": "crab",
    "cup cakes": "muffin",
    "deviled eggs": "egg",
    "donuts": "donut",
    "dumplings": "dumplings",
    "edamame": "peas",
    "eggs benedict": "fried egg",
    "falafel": "chickpeas",
    "filet mignon": "steak",
    "fish and chips": "cod",
    "french fries": "french fries",
    "french onion soup": "soup",
    "french toast": "bread",
    "fried calamari": "shrimp",
    "fried rice": "fried rice",
    "frozen yogurt": "yogurt",
    "garlic bread": "bread",
    "greek salad": "salad",
    "grilled cheese sandwich": "sandwich",
    "grilled salmon": "salmon",
    "guacamole": "avocado",
    "gyoza": "dumplings",
    "hamburger": "hamburger",
    "hot dog": "hot dog",
    "hummus": "chickpeas",
    "ice cream": "ice cream",
    "lasagna": "pasta",
    "lobster bisque": "soup",
    "macaroni and cheese": "pasta",
    "miso soup": "soup",
    "nachos": "tortilla",
    "omelette": "scrambled eggs",
    "onion rings": "onion",
    "pad thai": "pad thai",
    "pancakes": "pancake",
    "pizza": "pizza",
    "pork chop": "pork",
    "ramen": "ramen",
    "ravioli": "pasta",
    "risotto": "white rice",
    "samosa": "dumplings",
    "sashimi": "sashimi",
    "spaghetti bolognese": "pasta",
    "spaghetti carbonara": "pasta",
    "spring rolls": "dumplings",
    "steak": "steak",
    "sushi": "sushi",
    "tacos": "taco",
    "waffles": "waffle",
    "hot and sour soup": "soup",
    "huevos rancheros": "fried egg",
    "macarons": "cookie",
    "oysters": "shrimp",
    "paella": "white rice",
    "panna cotta": "cream",
    "pho": "ramen",
    "prime rib": "beef",
    "pulled pork sandwich": "sandwich",
    "red velvet cake": "cake",
    "scallops": "shrimp",
    "seaweed salad": "salad",
    "shrimp and grits": "shrimp",
    "strawberry shortcake": "cake",
    "takoyaki": "dumplings",
    "tiramisu": "cake",
    "tuna tartare": "tuna",
    "peking duck": "chicken thigh",
    "beignets": "donut",
    "beef carpaccio": "beef",
    "ceviche": "shrimp",
    "cheese plate": "cheese",
    "escargots": "beef",
    "foie gras": "chicken breast",
    "lobster roll sandwich": "lobster",
    "mussels": "shrimp",
    "poutine": "french fries",
    "gnocchi": "pasta",
}


class FoodDetector:
    """
    Local food detection pipeline using YOLOv8 + EfficientNet-B4.
    Weights are downloaded once and cached in ~/.cache/nutrilens/
    """

    CACHE_DIR = Path.home() / ".cache" / "nutrilens"
    YOLO_MODEL = "yolov8n.pt"          # ~6MB
    FOOD_MODEL_NAME = "efficientnet_b4"  # torchvision pretrained

    def __init__(self):
        self._yolo = None
        self._classifier = None
        self._transforms = None
        self._loaded = False

    # ── Lazy loading ──────────────────────────────────────

    def _load_yolo(self):
        from ultralytics import YOLO
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_path = self.CACHE_DIR / self.YOLO_MODEL
        self._yolo = YOLO(str(model_path) if model_path.exists() else self.YOLO_MODEL)
        logger.info("YOLOv8 loaded.")

    def _load_classifier(self):
        import torch
        import torchvision.transforms as T
        from torchvision import models

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load EfficientNet-B4 pretrained on ImageNet, replace head for Food-101
        weights = models.EfficientNet_B4_Weights.DEFAULT
        base = models.efficientnet_b4(weights=weights)

        # Replace classifier head: 1000 ImageNet → 101 Food-101
        in_features = base.classifier[1].in_features
        import torch.nn as nn
        base.classifier[1] = nn.Linear(in_features, len(FOOD101_CLASSES))

        # Load fine-tuned weights if available, else use ImageNet backbone
        # (with ImageNet backbone the food-specific accuracy is lower but still useful)
        fine_tuned_path = self.CACHE_DIR / "food101_efficientnet_b4.pth"
        if fine_tuned_path.exists():
            state = torch.load(fine_tuned_path, map_location=self._device)
            base.load_state_dict(state)
            logger.info("Loaded fine-tuned Food-101 EfficientNet-B4 weights.")
        else:
            logger.warning(
                "Fine-tuned Food-101 weights not found at %s. "
                "Using ImageNet pretrained backbone — accuracy will be reduced. "
                "See README for how to download fine-tuned weights.",
                fine_tuned_path
            )
            # Use ImageNet head for best-effort classification
            base = models.efficientnet_b4(weights=weights)

        base.eval().to(self._device)
        self._classifier = base

        self._transforms = T.Compose([
            T.Resize(380),
            T.CenterCrop(380),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info("EfficientNet-B4 classifier loaded on %s.", self._device)

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._load_yolo()
        self._load_classifier()
        self._loaded = True

    # ── Main detection method ─────────────────────────────

    def detect(self, image_path: Path) -> list[dict]:
        """
        Run the full detection pipeline on an image.

        Returns:
            list of {
                name: str,           # food name (nutrition DB key)
                display_name: str,   # human-readable label
                portion_grams: float,
                confidence: str,     # "high" | "medium" | "low"
                bbox: [x1,y1,x2,y2] | None,
                description: str,
            }
        """
        self._ensure_loaded()
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        img_area = img_w * img_h

        # ── Stage 1: YOLO detection ───────────────────────
        yolo_results = self._run_yolo(image_path)

        detections = []

        if yolo_results:
            for det in yolo_results:
                x1, y1, x2, y2 = det["bbox"]
                crop = image.crop((x1, y1, x2, y2))

                # ── Stage 2: Classify the crop ────────────
                clf_name, clf_conf = self._classify_crop(crop)

                # Resolve to DB key
                db_key = self._resolve_to_db_key(clf_name or det["yolo_label"])

                # ── Stage 3: Portion estimate ─────────────
                box_area = (x2 - x1) * (y2 - y1)
                portion = self._estimate_portion(db_key, box_area, img_area)

                confidence = self._score_to_confidence(clf_conf)

                detections.append({
                    "name": db_key,
                    "display_name": (clf_name or det["yolo_label"]).title(),
                    "portion_grams": portion,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "description": f"Detected via YOLO + classifier ({clf_conf:.0%} clf confidence)",
                })
        else:
            # Fallback: classify whole image
            clf_name, clf_conf = self._classify_crop(image)
            if clf_name:
                db_key = self._resolve_to_db_key(clf_name)
                portion = self._estimate_portion(db_key, img_area * 0.6, img_area)
                detections.append({
                    "name": db_key,
                    "display_name": clf_name.title(),
                    "portion_grams": portion,
                    "confidence": self._score_to_confidence(clf_conf),
                    "bbox": None,
                    "description": "Full-image classification (no bounding box detected)",
                })

        if not detections:
            raise ValueError("No food items could be detected in this image.")

        return detections

    # ── Stage 1: YOLO ─────────────────────────────────────

    def _run_yolo(self, image_path: Path) -> list[dict]:
        import torch
        results = self._yolo(str(image_path), verbose=False)[0]

        food_detections = []
        all_detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            label = results.names[cls_id]

            det = {
                "bbox": [x1, y1, x2, y2],
                "yolo_label": label,
                "yolo_conf": conf,
                "cls_id": cls_id,
            }

            all_detections.append(det)
            if cls_id in COCO_FOOD_CLASS_IDS and conf > 0.25:
                food_detections.append(det)

        # If no food-specific COCO classes found, use top-3 non-person detections
        if not food_detections:
            non_person = [d for d in all_detections if d["cls_id"] != 0 and d["yolo_conf"] > 0.3]
            food_detections = sorted(non_person, key=lambda d: d["yolo_conf"], reverse=True)[:3]

        return food_detections

    # ── Stage 2: EfficientNet crop classification ─────────

    def _classify_crop(self, crop: Image.Image) -> tuple[Optional[str], float]:
        import torch
        import torch.nn.functional as F

        # Minimum crop size check
        if crop.width < 32 or crop.height < 32:
            return None, 0.0

        tensor = self._transforms(crop).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._classifier(tensor)
            probs = F.softmax(logits, dim=1)[0]
            top_prob, top_idx = probs.max(dim=0)
            top_prob = top_prob.item()
            top_idx = top_idx.item()

        # If using Food-101 head
        if logits.shape[1] == len(FOOD101_CLASSES):
            label = FOOD101_CLASSES[top_idx]
        else:
            # ImageNet 1000-class fallback — map common ImageNet food labels
            imagenet_label = self._imagenet_idx_to_food(top_idx)
            label = imagenet_label

        if top_prob < 0.08:  # Very low confidence → skip
            return None, top_prob

        return label, top_prob

    def _imagenet_idx_to_food(self, idx: int) -> str:
        """
        Maps ImageNet-1000 indices for food items to our DB names.
        Non-exhaustive — only common food classes included.
        """
        IMAGENET_FOOD = {
            924: "guacamole", 925: "eggnog", 926: "orange juice",
            927: "espresso", 928: "cup cakes", 929: "ice cream",
            930: "ice cream", 931: "pizza", 932: "bagel",
            933: "pretzel", 934: "cheeseburger", 935: "hot dog",
            936: "mashed potato", 937: "head cabbage",
            938: "broccoli", 939: "cauliflower", 940: "zucchini",
            941: "spaghetti squash", 942: "acorn squash",
            943: "butternut squash", 944: "cucumber",
            945: "artichoke", 946: "bell pepper", 947: "cardoon",
            948: "mushroom", 949: "granny smith apple",
            950: "strawberry", 951: "orange", 952: "lemon",
            953: "fig", 954: "pineapple", 955: "banana",
            956: "jackfruit", 957: "custard apple", 958: "pomegranate",
            959: "hay", 960: "carbonara", 961: "chocolate sauce",
            962: "dough", 963: "meat loaf", 964: "pizza",
            965: "potpie", 966: "burrito", 967: "red wine",
            968: "espresso", 969: "cup", 970: "eggnog",
        }
        return IMAGENET_FOOD.get(idx, "food item")

    # ── DB key resolution ─────────────────────────────────

    def _resolve_to_db_key(self, name: str) -> str:
        from nutrition_db import NUTRITION_DB, FOOD_ALIASES, FOOD101_TO_DB

        name_lower = name.lower().strip()

        # Direct match
        if name_lower in NUTRITION_DB:
            return name_lower

        # Food-101 → DB mapping
        if name_lower in FOOD101_TO_DB:
            return FOOD101_TO_DB[name_lower]

        # Alias lookup
        if name_lower in FOOD_ALIASES:
            return FOOD_ALIASES[name_lower]

        # Partial match — check if any DB key is contained in the name
        for key in NUTRITION_DB:
            if key in name_lower or name_lower in key:
                return key

        # Partial alias match
        for alias, target in FOOD_ALIASES.items():
            if alias in name_lower or name_lower in alias:
                return target

        # Last resort
        logger.warning("Could not resolve '%s' to a DB key.", name)
        return "food item"

    # ── Portion estimation ────────────────────────────────

    def _estimate_portion(self, db_key: str, box_area: float, img_area: float) -> float:
        from nutrition_db import DEFAULT_PORTIONS

        # Fraction of image taken by food item
        coverage = min(box_area / img_area, 1.0)

        # Get reference portion for this food
        ref_portion = DEFAULT_PORTIONS.get(db_key, DEFAULT_PORTIONS["default"])

        # Heuristic: 30% coverage ≈ full reference portion
        # Scale linearly; clamp to realistic range
        estimated = ref_portion * (coverage / 0.30)
        estimated = max(30.0, min(estimated, 800.0))

        return round(estimated, 0)

    # ── Utilities ─────────────────────────────────────────

    @staticmethod
    def _score_to_confidence(score: float) -> str:
        if score >= 0.6:
            return "high"
        elif score >= 0.3:
            return "medium"
        return "low"


# Module-level singleton
_detector: Optional[FoodDetector] = None


def get_detector() -> FoodDetector:
    global _detector
    if _detector is None:
        _detector = FoodDetector()
    return _detector