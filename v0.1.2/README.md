# NutriLens Local 🍽️🔬

**Fully offline food analysis — no API keys, no cloud calls.**

Uses a local two-stage computer vision pipeline:
1. **YOLOv8** — detects and localises food items with bounding boxes
2. **EfficientNet-B4** — classifies each crop against 101 food categories (Food-101)
3. **Hardcoded nutrition DB** — 200+ foods with calories, protein, carbs, fat, fiber, sugar, sodium

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU recommended** — YOLO + EfficientNet runs fine on CPU (~3–5s/image) but GPU is faster (~0.5s/image). CUDA 11.8+ or MPS (Apple Silicon) supported automatically.

### 2. Download model weights
```bash
python download_models.py
```

This downloads:
- `yolov8n.pt` (~6MB) — YOLO object detector
- EfficientNet-B4 ImageNet backbone (~75MB) — via torchvision cache

### 3. (Recommended) Fine-tune on Food-101 for better accuracy
```bash
# Download Food-101 dataset (~5GB) and train — takes ~45 min on RTX 3090
python train_food101.py --epochs 20 --batch-size 32
```

Without this, the app uses the ImageNet backbone (works but lower food-specific accuracy).
With fine-tuned weights, expect ~82% Top-1 accuracy on Food-101 classes.

### 4. Run
```bash
python app.py
```

Open **http://localhost:5000**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (HTML/JS)                        │
│        Upload → Preview → Analyze → Results Dashboard       │
└────────────────────────┬────────────────────────────────────┘
                         │ POST /analyze
┌────────────────────────▼────────────────────────────────────┐
│                  Flask Backend (app.py)                      │
│                                                              │
│  1. Save image → uploads/{uuid}.jpg                         │
│                                                              │
│  2. YOLOv8n (food_detector.py)                              │
│     → Detect objects + bounding boxes                       │
│     → Filter to food-relevant COCO classes                  │
│     → Fallback: all detected objects if no food classes     │
│                                                              │
│  3. EfficientNet-B4 (per bounding box crop)                 │
│     → Fine-grained food classification (Food-101)           │
│     → Top-1 class + confidence score                        │
│                                                              │
│  4. Portion estimation (heuristic)                          │
│     → BBox area / image area → coverage fraction            │
│     → Scaled from reference portion sizes                   │
│     → Clamped: 30g – 800g                                   │
│                                                              │
│  5. Nutrition lookup (nutrition_db.py)                      │
│     → 200+ food entries, all values per 100g                │
│     → Scaled to estimated portion                           │
│     → Optional: USDA API override if USDA_API_KEY set       │
│                                                              │
│  6. Aggregate + return JSON                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
nutrilens-local/
├── app.py               # Flask server + nutrition pipeline
├── food_detector.py     # YOLOv8 + EfficientNet-B4 detection
├── nutrition_db.py      # 200+ food nutrition entries + aliases
├── download_models.py   # One-time model weight downloader
├── train_food101.py     # Fine-tune EfficientNet-B4 on Food-101
├── requirements.txt
├── uploads/             # Auto-created, stores analysed images
└── templates/
    └── index.html       # Frontend SPA
```

---

## Model Details

### Stage 1 — YOLOv8 (Object Detection)
| Property | Value |
|----------|-------|
| Model | YOLOv8n (nano) |
| Weight size | ~6MB |
| Dataset | COCO 80-class |
| Food classes (COCO) | apple, banana, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake |
| Inference time (CPU) | ~200ms |
| Inference time (GPU) | ~10ms |

When no COCO food classes are detected, the top non-person objects are passed to the classifier anyway — so non-COCO foods (e.g. salmon, ramen) can still be detected via the classifier stage.

### Stage 2 — EfficientNet-B4 (Food Classifier)
| Property | Value |
|----------|-------|
| Base model | EfficientNet-B4 |
| Weight size | ~75MB |
| Training dataset | Food-101 (101,000 images, 101 classes) |
| Top-1 accuracy (fine-tuned) | ~82% |
| Top-1 accuracy (ImageNet only) | ~45% on food |
| Inference time (CPU) | ~150ms per crop |
| Inference time (GPU) | ~8ms per crop |

### Portion Estimation
Uses a geometric heuristic:
```
coverage = bbox_area / image_area
estimated_grams = reference_portion × (coverage / 0.30)
clamped to [30g, 800g]
```
Where `reference_portion` is a per-food baseline (e.g. chicken breast = 150g, pizza slice = 285g).

---

## Nutrition Database

`nutrition_db.py` contains:
- **200+ food entries** — values per 100g (USDA SR Legacy sourced)
- **Calories, protein, carbs, fat, fiber, sugar, sodium, saturated fat**
- **200+ aliases** — e.g. "spaghetti" → "pasta", "prawns" → "shrimp"
- **Food-101 → DB mappings** — bridges classifier output to nutrition entries
- **Default portion table** — reference sizes for portion estimation

---

## Accuracy Notes

| Scenario | Expected Accuracy |
|----------|-----------------|
| Common foods (pizza, burger, salad) | High — both YOLO + classifier agree |
| Foods in COCO classes | Very high — YOLO detects, classifier refines |
| Fine-grained foods (sashimi vs sushi) | Moderate — depends on fine-tuned weights |
| Mixed plate meals | Good — multiple bounding boxes per image |
| Portion estimation | ±30% — heuristic only, no depth data |
| Nutrition values | ±10% — standardised DB values, not recipe-specific |

---

## Configuration

`.env` (optional):
```
USDA_API_KEY=your_key_here   # enables USDA API as primary nutrition source
```

Without `USDA_API_KEY`, the local `nutrition_db.py` is used exclusively.

---

## Hardware Requirements

| Setup | Min RAM | Storage | Notes |
|-------|---------|---------|-------|
| CPU-only | 4GB | 500MB | Slower (~3–5s/image) |
| NVIDIA GPU | 4GB VRAM | 500MB | Fast (~0.5s/image) |
| Apple Silicon | 8GB | 500MB | Uses MPS backend automatically |