import os
import uuid
import json
import base64
import requests
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template
from dotenv import load_dotenv
import anthropic

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
USDA_API_KEY = os.getenv("USDA_API_KEY", "DEMO_KEY")

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def analyze_image_with_claude(image_path: Path) -> list[dict]:
    """
    Send image to Claude Vision and get structured food item list
    with estimated portions.
    Returns list of dicts: [{name, portion_grams, confidence}, ...]
    """
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Detect media type
    suffix = image_path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")

    prompt = """You are a precise food detection and nutrition AI. Analyze this food image carefully.

Identify EVERY distinct food item visible in the image. For each item:
1. Name it specifically (e.g., "white rice" not just "rice", "grilled salmon fillet" not just "fish")
2. Estimate the portion size in grams based on visual cues
3. Rate your detection confidence as "high", "medium", or "low"

Return ONLY a valid JSON array, no other text. Format:
[
  {
    "name": "food item name",
    "portion_grams": 150,
    "confidence": "high",
    "description": "brief description for clarity"
  }
]

If you cannot identify any food in the image, return: {"error": "No food items detected in this image"}

Be thorough - include sauces, dressings, garnishes, and side items if visible."""

    message = anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )

    response_text = message.content[0].text.strip()
    
    # Extract JSON from response
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])
    
    parsed = json.loads(response_text)
    
    if isinstance(parsed, dict) and "error" in parsed:
        raise ValueError(parsed["error"])
    
    return parsed


def get_nutrition_from_usda(food_name: str, portion_grams: float) -> dict | None:
    """
    Query USDA FoodData Central API for nutrition data.
    Returns scaled nutrition per portion_grams.
    """
    search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": food_name,
        "api_key": USDA_API_KEY,
        "pageSize": 5,
        "dataType": "Foundation,SR Legacy,Survey (FNDDS)",
    }
    
    try:
        resp = requests.get(search_url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        
        if not data.get("foods"):
            return None
        
        food = data["foods"][0]
        nutrients = {n["nutrientName"]: n["value"] for n in food.get("foodNutrients", [])}
        
        # USDA values are per 100g — scale to portion
        scale = portion_grams / 100.0
        
        def get_nutrient(*keys, default=0):
            for key in keys:
                for name, val in nutrients.items():
                    if key.lower() in name.lower():
                        return round(val * scale, 1)
            return default
        
        return {
            "calories": get_nutrient("Energy"),
            "protein": get_nutrient("Protein"),
            "carbs": get_nutrient("Carbohydrate"),
            "fat": get_nutrient("Total lipid", "Fat"),
            "fiber": get_nutrient("Fiber"),
            "sugar": get_nutrient("Sugars"),
            "sodium": get_nutrient("Sodium"),
            "saturated_fat": get_nutrient("Fatty acids, total saturated"),
            "source": "USDA FoodData Central",
            "usda_food_name": food.get("description", food_name),
        }
    except Exception as e:
        print(f"USDA lookup failed for '{food_name}': {e}")
        return None


def get_nutrition_from_claude(food_name: str, portion_grams: float) -> dict:
    """
    Fallback: use Claude's nutrition knowledge when USDA lookup fails.
    """
    prompt = f"""Provide approximate nutrition facts for {portion_grams}g of {food_name}.
Return ONLY valid JSON, no other text:
{{
  "calories": 0,
  "protein": 0,
  "carbs": 0,
  "fat": 0,
  "fiber": 0,
  "sugar": 0,
  "sodium": 0,
  "saturated_fat": 0,
  "source": "AI estimate",
  "usda_food_name": "{food_name}"
}}"""

    message = anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    
    response_text = message.content[0].text.strip()
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])
    
    return json.loads(response_text)


def calculate_totals(items: list[dict]) -> dict:
    totals = {
        "calories": 0, "protein": 0, "carbs": 0, "fat": 0,
        "fiber": 0, "sugar": 0, "sodium": 0, "saturated_fat": 0,
    }
    for item in items:
        if "nutrition" in item:
            for key in totals:
                totals[key] = round(totals[key] + item["nutrition"].get(key, 0), 1)
    return totals


def calculate_macro_percentages(totals: dict) -> dict:
    protein_cals = totals["protein"] * 4
    carb_cals = totals["carbs"] * 4
    fat_cals = totals["fat"] * 9
    total_macro_cals = protein_cals + carb_cals + fat_cals
    
    if total_macro_cals == 0:
        return {"protein": 0, "carbs": 0, "fat": 0}
    
    return {
        "protein": round(protein_cals / total_macro_cals * 100),
        "carbs": round(carb_cals / total_macro_cals * 100),
        "fat": round(fat_cals / total_macro_cals * 100),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Save image
    ext = Path(file.filename).suffix.lower() or ".jpg"
    filename = f"{uuid.uuid4()}{ext}"
    image_path = UPLOAD_FOLDER / filename
    file.save(image_path)
    
    try:
        # Step 1: Detect food items via Claude Vision
        detected_items = analyze_image_with_claude(image_path)
        
        # Step 2: Get nutrition for each item
        enriched_items = []
        for item in detected_items:
            food_name = item["name"]
            portion_grams = float(item.get("portion_grams", 100))
            
            nutrition = get_nutrition_from_usda(food_name, portion_grams)
            
            if nutrition is None:
                # Fallback to Claude's knowledge
                try:
                    nutrition = get_nutrition_from_claude(food_name, portion_grams)
                    nutrition["source"] = "AI estimate (fallback)"
                except Exception:
                    nutrition = {
                        "calories": 0, "protein": 0, "carbs": 0, "fat": 0,
                        "fiber": 0, "sugar": 0, "sodium": 0, "saturated_fat": 0,
                        "source": "unavailable",
                        "usda_food_name": food_name,
                    }
            
            enriched_items.append({
                "name": food_name,
                "portion": f"{portion_grams}g",
                "portion_grams": portion_grams,
                "confidence": item.get("confidence", "medium"),
                "description": item.get("description", ""),
                "nutrition": nutrition,
            })
        
        # Step 3: Aggregate
        totals = calculate_totals(enriched_items)
        macro_pct = calculate_macro_percentages(totals)
        
        return jsonify({
            "success": True,
            "image_path": f"uploads/{filename}",
            "items": enriched_items,
            "totals": totals,
            "macro_percentages": macro_pct,
            "item_count": len(enriched_items),
        })
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    if not ANTHROPIC_API_KEY:
        print("WARNING: ANTHROPIC_API_KEY not set in .env")
    app.run(debug=True, port=5000)