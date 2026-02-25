"""
Bhaat.AI v0.1.3 - Food Recognition Web App

Simple Flask app that:
1. Serves a frontend for uploading food images
2. Saves uploads to the uploads/ folder
3. Runs the image through a ResNet-18 model trained on Indian food
4. Prints detection results to the terminal

Usage:
    1. Train the model first:   python train.py
    2. Run the server:          python app.py
    3. Open http://localhost:8000 in your browser
    4. Upload a food image and check the terminal for results
"""

import os
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

from model import load_trained_model, predict, TRANSFORM

# ---------------------------------------------------------------------------
# Flask setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
model, class_names = load_trained_model(MODEL_DIR)

if model is not None:
    print(f"[OK] Model loaded — {len(class_names)} food classes")
else:
    print("[WARNING] No trained model found.")
    print("  Run 'python train.py' first to train the model.")
    print("  The server will still accept uploads but won't classify food.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def format_size(size_bytes):
    for unit in ["B", "KB", "MB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} GB"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    has_model = model is not None
    return render_template("index.html", has_model=has_model)


@app.route("/upload", methods=["POST"])
def upload_file():
    # Validate request
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No file in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "File type not allowed. Use JPG or PNG."}), 400

    # Save file
    original = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(original)
    filename = f"{name}_{timestamp}{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    file_size = os.path.getsize(filepath)

    # ----- Print to terminal -----
    print("\n" + "=" * 60)
    print(f"  NEW UPLOAD: {filename}")
    print(f"  Size: {format_size(file_size)}")
    print(f"  Saved to: {filepath}")
    print("=" * 60)

    # ----- Run food recognition -----
    predictions = []
    if model is not None:
        try:
            results = predict(filepath, model, class_names)

            print("\n  FOOD RECOGNITION RESULTS:")
            print("  " + "-" * 40)
            for rank, (food_name, confidence) in enumerate(results, 1):
                print(f"  {rank}. {food_name} — {confidence}% confidence")
                predictions.append({
                    "rank": rank,
                    "name": food_name,
                    "confidence": confidence,
                })
            print("  " + "-" * 40)
            print()
        except Exception as e:
            print(f"  ERROR during recognition: {e}")
    else:
        print("  [SKIPPED] No model loaded — run 'python train.py' first")
        print()

    return jsonify({
        "success": True,
        "message": "File uploaded and processed",
        "filename": filename,
        "size": format_size(file_size),
        "predictions": predictions,
    }), 200


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n  Bhaat.AI v0.1.3")
    print("  http://localhost:8000\n")
    app.run(host="0.0.0.0", port=8000, debug=True)
