"""
app.py — NutriLens Local
────────────────────────────────────────────────────────────────────
Flask server for the fully-local food analysis pipeline.
No Anthropic API. No external calls for detection.
Optional: USDA API for richer nutrition data (falls back to local DB).
────────────────────────────────────────────────────────────────────
"""

import os
import uuid
import logging
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request, send_from_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

# Optional: USDA API for richer data (set in env, falls back to local DB)
USDA_API_KEY = os.getenv("USDA_API_KEY", "")


# ── Lazy-loaded detector ──────────────────────────────────────────
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        from food_detector import FoodDetector
        _detector = FoodDetector()
        logger.info("FoodDetector initialised.")
    return _detector


# ── Nutrition helpers ─────────────────────────────────────────────

def lookup_local_nutrition(db_key: str, portion_grams: float) -> dict:
    """Return nutrition from the hardcoded local DB, scaled to portion."""
    from nutrition_db import NUTRITION_DB

    entry = NUTRITION_DB.get(db_key)
    if entry is None:
        # Try partial match
        for key, val in NUTRITION_DB.items():
            if key in db_key or db_key in key:
                entry = val
                break

    if entry is None:
        logger.warning("'%s' not found in local nutrition DB.", db_key)
        return _empty_nutrition(source="Not found in local DB")

    scale = portion_grams / 100.0
    return {
        "calories":       round(entry.get("calories", 0)       * scale, 1),
        "protein":        round(entry.get("protein", 0)        * scale, 1),
        "carbs":          round(entry.get("carbs", 0)          * scale, 1),
        "fat":            round(entry.get("fat", 0)            * scale, 1),
        "fiber":          round(entry.get("fiber", 0)          * scale, 1),
        "sugar":          round(entry.get("sugar", 0)          * scale, 1),
        "sodium":         round(entry.get("sodium", 0)         * scale, 1),
        "saturated_fat":  round(entry.get("saturated_fat", 0)  * scale, 1),
        "source":         "Local nutrition database",
        "db_key":         db_key,
    }


def lookup_usda_nutrition(food_name: str, portion_grams: float) -> dict | None:
    """Optional: query USDA FoodData Central for richer data."""
    if not USDA_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={"query": food_name, "api_key": USDA_API_KEY, "pageSize": 3,
                    "dataType": "Foundation,SR Legacy"},
            timeout=6,
        )
        resp.raise_for_status()
        foods = resp.json().get("foods", [])
        if not foods:
            return None

        nutrients = {n["nutrientName"]: n["value"] for n in foods[0].get("foodNutrients", [])}
        scale = portion_grams / 100.0

        def g(key):
            for name, val in nutrients.items():
                if key.lower() in name.lower():
                    return round(val * scale, 1)
            return 0

        return {
            "calories": g("Energy"),
            "protein": g("Protein"),
            "carbs": g("Carbohydrate"),
            "fat": g("Total lipid"),
            "fiber": g("Fiber"),
            "sugar": g("Sugars"),
            "sodium": g("Sodium"),
            "saturated_fat": g("Fatty acids, total saturated"),
            "source": "USDA FoodData Central",
            "db_key": foods[0].get("description", food_name),
        }
    except Exception as e:
        logger.warning("USDA lookup failed: %s", e)
        return None


def _empty_nutrition(source="unavailable") -> dict:
    return {
        "calories": 0, "protein": 0, "carbs": 0, "fat": 0,
        "fiber": 0, "sugar": 0, "sodium": 0, "saturated_fat": 0,
        "source": source, "db_key": "unknown",
    }


def calculate_totals(items: list[dict]) -> dict:
    keys = ["calories", "protein", "carbs", "fat", "fiber", "sugar", "sodium", "saturated_fat"]
    totals = {k: 0.0 for k in keys}
    for item in items:
        n = item.get("nutrition", {})
        for k in keys:
            totals[k] = round(totals[k] + n.get(k, 0), 1)
    return totals


def macro_percentages(totals: dict) -> dict:
    p = totals["protein"] * 4
    c = totals["carbs"] * 4
    f = totals["fat"] * 9
    total = p + c + f or 1
    return {
        "protein": round(p / total * 100),
        "carbs": round(c / total * 100),
        "fat": round(f / total * 100),
    }


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/analyze", methods=["POST"])
def analyze():
    # ── Validate upload ───────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_BYTES:
        return jsonify({"error": "File too large (max 10MB)"}), 413

    # ── Save image ────────────────────────────────────────
    ext = Path(file.filename).suffix.lower() or ".jpg"
    filename = f"{uuid.uuid4()}{ext}"
    image_path = UPLOAD_FOLDER / filename
    file.save(image_path)
    logger.info("Image saved: %s", image_path)

    try:
        # ── Stage 1+2+3: Detect food items locally ────────
        detector = get_detector()
        detected = detector.detect(image_path)
        logger.info("Detected %d items: %s", len(detected), [d["name"] for d in detected])

        # ── Stage 4: Nutrition lookup ─────────────────────
        enriched = []
        for det in detected:
            db_key = det["name"]
            portion = det["portion_grams"]

            # Try USDA first (optional), fallback to local DB
            nutrition = lookup_usda_nutrition(det["display_name"], portion) or \
                        lookup_local_nutrition(db_key, portion)

            enriched.append({
                "name": det["display_name"],
                "db_key": db_key,
                "portion": f"{int(portion)}g",
                "portion_grams": portion,
                "confidence": det["confidence"],
                "description": det.get("description", ""),
                "bbox": det.get("bbox"),
                "nutrition": nutrition,
            })

        # ── Stage 5: Aggregate ────────────────────────────
        totals = calculate_totals(enriched)
        pcts = macro_percentages(totals)

        return jsonify({
            "success": True,
            "image_path": f"uploads/{filename}",
            "items": enriched,
            "totals": totals,
            "macro_percentages": pcts,
            "item_count": len(enriched),
            "pipeline": "local-yolo-efficientnet",
        })

    except ValueError as e:
        logger.warning("Detection error: %s", e)
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.exception("Unexpected error during analysis")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "pipeline": "local-yolo-efficientnet"})


if __name__ == "__main__":
    logger.info("Starting NutriLens Local on http://localhost:5000")
    logger.info("Pipeline: YOLOv8 + EfficientNet-B4 (Food-101) — NO external API calls")
    app.run(debug=False, port=5000)
    