"""
app.py - SmartCrop AI Flask web application.
Start with:  python app.py
Model must be trained first:  python train_model.py
"""

import os
import json
import pickle
import numpy as np
from flask import Flask, render_template, request, url_for, flash

app = Flask(__name__)
app.secret_key = "smartcrop-secret-key"

MODEL_PATH   = os.path.join("models", "best_model.pkl")
METRICS_PATH = os.path.join("models", "metrics.json")

# ── Crop information dictionary ────────────────────────────────────────────
CROP_INFO = {
    "rice":         {"season": "Kharif",  "water": "High",   "desc": "A staple cereal crop grown in waterlogged paddy fields."},
    "maize":        {"season": "Kharif",  "water": "Medium", "desc": "A versatile cereal used for food, feed, and industrial products."},
    "chickpea":     {"season": "Rabi",    "water": "Low",    "desc": "A protein-rich pulse crop suited to cool, dry climates."},
    "kidneybeans":  {"season": "Kharif",  "water": "Medium", "desc": "A nutritious legume that also fixes atmospheric nitrogen."},
    "pigeonpeas":   {"season": "Kharif",  "water": "Low",    "desc": "A drought-tolerant pulse widely grown in tropical regions."},
    "mothbeans":    {"season": "Kharif",  "water": "Low",    "desc": "A highly drought-resistant legume for arid areas."},
    "mungbean":     {"season": "Kharif",  "water": "Low",    "desc": "A quick-maturing pulse with high protein content."},
    "blackgram":    {"season": "Kharif",  "water": "Low",    "desc": "A pulse crop rich in proteins and essential amino acids."},
    "lentil":       {"season": "Rabi",    "water": "Low",    "desc": "A cool-season legume and important dietary protein source."},
    "pomegranate":  {"season": "Perennial","water": "Low",   "desc": "A fruit shrub prized for its antioxidant-rich seeds."},
    "banana":       {"season": "Perennial","water": "High",  "desc": "A tropical fruit crop requiring warm temperatures and moisture."},
    "mango":        {"season": "Perennial","water": "Medium","desc": "A tropical tree fruit requiring a warm, dry flowering season."},
    "grapes":       {"season": "Perennial","water": "Medium","desc": "A fruit vine cultivated in well-drained, sunny conditions."},
    "watermelon":   {"season": "Kharif",  "water": "Medium", "desc": "A warm-season vine fruit with high water content."},
    "muskmelon":    {"season": "Kharif",  "water": "Medium", "desc": "A sweet melon crop that thrives in warm, sandy soils."},
    "apple":        {"season": "Rabi",    "water": "Medium", "desc": "A temperate fruit requiring cold winters for proper dormancy."},
    "orange":       {"season": "Perennial","water": "Medium","desc": "A citrus fruit tree suited to subtropical climates."},
    "papaya":       {"season": "Perennial","water": "Medium","desc": "A fast-growing tropical fruit rich in vitamins A and C."},
    "coconut":      {"season": "Perennial","water": "High",  "desc": "A multipurpose palm tree native to tropical coastal areas."},
    "cotton":       {"season": "Kharif",  "water": "Medium", "desc": "A fibre crop requiring warm temperatures and adequate rainfall."},
    "jute":         {"season": "Kharif",  "water": "High",   "desc": "A long, soft plant fibre grown in humid tropical climates."},
    "coffee":       {"season": "Perennial","water": "Medium","desc": "A shade-loving shrub producing one of the world's most traded commodities."},
}

# ── Load model & metrics at startup ────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["label_encoder"]

def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)

MODEL, ENCODER = load_model()
METRICS        = load_metrics()

# ── Input validation ────────────────────────────────────────────────────────
FIELD_RANGES = {
    "N":           (0,   200, "Nitrogen (N)"),
    "P":           (0,   200, "Phosphorus (P)"),
    "K":           (0,   200, "Potassium (K)"),
    "temperature": (-10,  55, "Temperature (°C)"),
    "humidity":    (0,   100, "Humidity (%)"),
    "ph":          (0,    14, "pH"),
    "rainfall":    (0,  4000, "Rainfall (mm)"),
}

def validate_inputs(form):
    """Returns (values_dict, error_message). error_message is None on success."""
    values = {}
    for field, (lo, hi, label) in FIELD_RANGES.items():
        raw = form.get(field, "").strip()
        if raw == "":
            return None, f"{label} is required."
        try:
            val = float(raw)
        except ValueError:
            return None, f"{label} must be a number."
        if not (lo <= val <= hi):
            return None, f"{label} must be between {lo} and {hi}."
        values[field] = val
    return values, None

# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    model_ready = MODEL is not None
    return render_template("home.html", model_ready=model_ready)

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if MODEL is None:
        flash("Model not found. Please run  python train_model.py  first.", "error")
        return render_template("recommend.html", error=True)

    if request.method == "POST":
        values, err = validate_inputs(request.form)
        if err:
            flash(err, "error")
            return render_template("recommend.html", form_data=request.form, error=False)

        features = np.array([[
            values["N"], values["P"], values["K"],
            values["temperature"], values["humidity"],
            values["ph"], values["rainfall"]
        ]])
        prediction_enc = MODEL.predict(features)[0]
        crop           = ENCODER.inverse_transform([prediction_enc])[0]
        info           = CROP_INFO.get(crop, {"season": "—", "water": "—", "desc": "No additional information available."})
        return render_template("result.html", crop=crop, values=values, info=info)

    return render_template("recommend.html", form_data=None, error=False)

@app.route("/performance")
def performance():
    if METRICS is None:
        flash("Metrics not found. Please run  python train_model.py  first.", "error")
        return render_template("performance.html", metrics=None)
    return render_template("performance.html", metrics=METRICS)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    if MODEL is None:
        print("Model not found. Run  python train_model.py  before starting the app.")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
