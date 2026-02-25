# NutriLens 🍽️

AI-powered food photo analyzer. Upload a meal photo → get instant calorie and macronutrient breakdown per food item.

## Quick Start

### 1. Clone & install dependencies
```bash
cd nutrilens
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
```
Edit `.env` and add your keys:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
USDA_API_KEY=DEMO_KEY   # or get a free key at api.nal.usda.gov
```

**Get API keys:**
- **Anthropic:** https://console.anthropic.com/
- **USDA FoodData Central (free):** https://fdc.nal.usda.gov/api-guide.html

### 3. Run
```bash
python app.py
```

Open http://localhost:5000

## How It Works

1. **Upload** a photo of any meal (drag & drop or click)
2. **Claude Vision AI** scans the image and identifies every food item with estimated portion sizes
3. **USDA FoodData Central API** retrieves precise nutrition data for each item
4. **Results dashboard** shows per-item breakdown + total meal macros with a visual donut chart

## Features

- ✅ Detects single and multiple food items
- ✅ Estimates portion sizes from visual context
- ✅ Per-item: calories, protein, carbs, fat, fiber, sugar, sodium
- ✅ Confidence scoring per detected item
- ✅ Fallback to AI nutrition knowledge when USDA DB has no match
- ✅ Images stored in `/uploads/` folder automatically
- ✅ Responsive web interface

## File Structure

```
nutrilens/
├── app.py              # Flask backend + analysis pipeline
├── templates/
│   └── index.html      # Full frontend (dark editorial design)
├── uploads/            # Auto-created, stores user images
├── requirements.txt
├── .env.example
└── README.md
```

## API Reference

### POST `/analyze`
Upload an image for analysis.

**Request:** `multipart/form-data` with `image` field

**Response:**
```json
{
  "success": true,
  "image_path": "uploads/uuid.jpg",
  "items": [
    {
      "name": "Grilled Chicken Breast",
      "portion": "150g",
      "confidence": "high",
      "nutrition": {
        "calories": 248, "protein": 46.5, "carbs": 0, "fat": 5.4,
        "fiber": 0, "sugar": 0, "sodium": 112, "saturated_fat": 1.5,
        "source": "USDA FoodData Central"
      }
    }
  ],
  "totals": { "calories": 248, "protein": 46.5, ... },
  "macro_percentages": { "protein": 75, "carbs": 0, "fat": 25 },
  "item_count": 1
}
```