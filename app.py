import os

import cv2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from utils.classification import load_cnn_model, predict_cancer
from utils.enhancement import enhance_image, load_gan_model
# ---------------------------
# Import utility modules
# ---------------------------
from utils.preprocessing import preprocess_image
from utils.visualization import (apply_contrast_enhancement,
                                 generate_binarized_image,
                                 generate_zoomed_view, save_visual_outputs)

# ---------------------------
# Flask App Setup
# ---------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/outputs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------
# Load Models Once
# ---------------------------
print("🔄 Loading models...")
load_gan_model("models/gan_generator.pth")
load_cnn_model("models/cancer_classifier.pth")
print("✅ Models loaded successfully")


# ---------------------------
# Landing Page (GET)
# ---------------------------
@app.route("/", methods=["GET"])
def landing():
    return render_template("index.html")


# ---------------------------
# Analyze Image (POST)
# ---------------------------
@app.route("/analyze", methods=["POST"])
def analyze():

    file = request.files.get("image")

    if not file or file.filename == "":
        return render_template("index.html")

    if not allowed_file(file.filename):
        return render_template("index.html")

    # Save uploaded image
    filename = secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    # Read original image
    original = cv2.imread(upload_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        return render_template("index.html")

    original = cv2.resize(original, (128, 128))

    # ---------------------------
    # AI PIPELINE
    # ---------------------------
    preprocessed = preprocess_image(upload_path)
    enhanced = enhance_image(preprocessed)

    contrast = apply_contrast_enhancement(enhanced)
    zoomed = generate_zoomed_view(enhanced)
    binary = generate_binarized_image(enhanced)

    image_paths = save_visual_outputs(
        original,
        enhanced,
        contrast,
        zoomed,
        binary,
        OUTPUT_FOLDER
    )

    label, confidence = predict_cancer(enhanced)

    results = {
        "images": image_paths,
        "label": label,
        "confidence": confidence
    }

    return render_template("result.html", results=results)


# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    app.run(debug=True)
