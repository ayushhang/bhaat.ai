# NutriLens — AI Food Nutrition Analyzer
## Product Definition Document v1.0

---

## 1. Product Overview

**NutriLens** is a web application that allows users to photograph their meal and instantly receive a detailed nutritional breakdown — calories, macronutrients, and per-item analysis — powered by AI vision and a nutrition database.

**Core Value Proposition:** Take a photo of any meal, get instant, itemized nutrition data. No barcode scanning, no manual logging, no guessing.

---

## 2. User Flow

```
User uploads food photo
        ↓
Image stored in /uploads directory
        ↓
Claude Vision API analyzes image
→ Identifies every distinct food item
→ Estimates portion sizes where possible
        ↓
For each identified item:
→ Query nutrition database (USDA FoodData Central API)
→ Retrieve: calories, protein, carbs, fat, fiber, sugar, sodium
        ↓
Aggregate all items into a total meal summary
        ↓
Display results on a clean, visual dashboard
```

---

## 3. Capabilities

### 3.1 Food Detection
- Identifies **single or multiple food items** in one image
- Handles complex plated meals (e.g., "grilled chicken breast, steamed broccoli, white rice, olive oil drizzle")
- Estimates **portion sizes** using visual cues (plate size, context)
- Handles packaged foods, raw ingredients, restaurant dishes, home-cooked meals
- Fallback: if portion size is ambiguous, uses a standard serving size and flags it

### 3.2 Nutrition Analysis
Per food item retrieved:
- **Calories** (kcal)
- **Protein** (g)
- **Carbohydrates** (g) — total, fiber, sugar
- **Fat** (g) — total, saturated
- **Sodium** (mg)

Aggregated meal totals for all of the above.

### 3.3 Data Display
- Per-item breakdown cards
- Total meal summary bar
- Macro distribution donut chart (protein / carbs / fat %)
- Confidence indicator per food item detection
- Portion size caveat flagging

### 3.4 Image Handling
- Accepted formats: JPEG, PNG, WEBP, HEIC
- Max file size: 10MB
- Image stored in `/uploads/` with a UUID filename
- Timestamp and metadata logged alongside image

---

## 4. Technical Architecture

```
┌─────────────────────────────────────────┐
│              Frontend (HTML/JS)         │
│  - Drag & drop / click-to-upload UI     │
│  - Image preview                        │
│  - Results dashboard                    │
└────────────────┬────────────────────────┘
                 │ POST /analyze (multipart/form-data)
┌────────────────▼────────────────────────┐
│           Flask Backend (Python)        │
│                                         │
│  1. Save image → /uploads/{uuid}.jpg    │
│  2. Send image → Claude Vision API      │
│     → Parse food items + portions       │
│  3. For each item → USDA FoodData API   │
│     → Retrieve nutrition data           │
│  4. Aggregate + return JSON response    │
└─────────────────────────────────────────┘
```

### 4.1 APIs Used
| API | Purpose | Auth |
|-----|---------|------|
| Anthropic Claude Vision (claude-opus-4-6) | Food detection from image | `ANTHROPIC_API_KEY` |
| USDA FoodData Central | Nutrition lookup | `USDA_API_KEY` (free, public) |

### 4.2 Backend Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend HTML |
| `/analyze` | POST | Accept image, run full pipeline, return nutrition JSON |
| `/uploads/<filename>` | GET | Serve uploaded images |

---

## 5. Data Model

### Request
```json
{ "image": "<multipart file>" }
```

### Response
```json
{
  "image_path": "uploads/uuid.jpg",
  "items": [
    {
      "name": "Grilled Chicken Breast",
      "portion": "150g (estimated)",
      "confidence": "high",
      "nutrition": {
        "calories": 248,
        "protein": 46.5,
        "carbs": 0,
        "fat": 5.4,
        "fiber": 0,
        "sugar": 0,
        "sodium": 112
      }
    }
  ],
  "totals": {
    "calories": 580,
    "protein": 52,
    "carbs": 45,
    "fat": 18,
    "fiber": 6,
    "sugar": 4,
    "sodium": 890
  },
  "macro_percentages": {
    "protein": 36,
    "carbs": 31,
    "fat": 28
  }
}
```

---

## 6. Edge Cases & Handling

| Scenario | Handling |
|----------|---------|
| Unrecognizable image (not food) | Return error: "No food items detected" |
| Partial recognition | Return detected items, flag unrecognized regions |
| Ambiguous portion size | Use standard serving, label as "estimated (standard serving)" |
| Food not in USDA DB | Use Claude's built-in nutrition knowledge as fallback, flagged |
| Network failure to USDA | Graceful error per item, still return other items |
| Image too large | Frontend rejects before upload |
| No API key configured | Clear setup instructions returned |

---

## 7. Configuration

Environment variables required in `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
USDA_API_KEY=DEMO_KEY   # or register free at api.nal.usda.gov
```

---

## 8. File Structure

```
nutrilens/
├── app.py                  # Flask backend
├── .env                    # API keys (gitignored)
├── requirements.txt        # Python dependencies
├── uploads/                # Stored user images (auto-created)
└── templates/
    └── index.html          # Full frontend SPA
```

---

## 9. Non-Goals (v1)

- User accounts / history tracking
- Barcode scanning
- Restaurant menu lookup
- Dietary goal tracking
- Mobile app (web-responsive only)
- Real-time video analysis

---

## 10. Success Metrics

- Food detection accuracy: >85% items correctly identified
- Nutrition retrieval rate: >90% items successfully matched in USDA DB
- Response time: <8 seconds end-to-end for typical meal photo
- Error rate: <5% complete failures

---

*Document version: 1.0 | NutriLens*