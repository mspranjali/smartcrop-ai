"""
train_model.py
Run once to train and save the best ML model for SmartCrop AI.
Usage: python train_model.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

DATASET_PATH = os.path.join("dataset", "Crop_recommendation.csv")
MODEL_PATH   = os.path.join("models", "best_model.pkl")
METRICS_PATH = os.path.join("models", "metrics.json")

# ── 1. Load & validate dataset ─────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATASET_PATH)
required = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
missing  = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

print(f"  Rows: {len(df)}  |  Classes: {df['label'].nunique()}  |  Nulls: {df.isnull().sum().sum()}")

# ── 2. Encode labels ────────────────────────────────────────────────────────
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df["label"])
classes = le.classes_.tolist()

X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].values
y = df["label_enc"].values

# ── 3. Stratified train/test split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── 4. Train & evaluate three classifiers ─────────────────────────────────
candidates = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN":           KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    results[name] = {
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1_score":  round(float(f1),   4),
        "confusion_matrix": cm,
    }
    print(f"  {name:15s}  acc={acc:.4f}  f1={f1:.4f}")

# ── 5. Select best model ────────────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["f1_score"])
best_model = candidates[best_name]
print(f"\nBest model: {best_name} (F1={results[best_name]['f1_score']:.4f})")

# ── 6. Persist model + metadata ────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": best_model, "label_encoder": le}, f)

# Per-class metrics for the best model
y_pred_best  = best_model.predict(X_test)
per_class_f1 = f1_score(y_test, y_pred_best, average=None, zero_division=0)
per_class    = {cls: round(float(s), 4) for cls, s in zip(classes, per_class_f1)}

# Feature importances (RF/DT only; None for KNN)
importances = None
if hasattr(best_model, "feature_importances_"):
    feat_names   = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    importances  = {f: round(float(v), 4) for f, v in zip(feat_names, best_model.feature_importances_)}

metrics = {
    "best_model":        best_name,
    "classes":           classes,
    "comparison":        results,
    "best_model_detail": results[best_name],
    "per_class_f1":      per_class,
    "feature_importances": importances,
    "train_size":        int(len(X_train)),
    "test_size":         int(len(X_test)),
    "total_samples":     int(len(df)),
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Model saved   -> {MODEL_PATH}")
print(f"Metrics saved -> {METRICS_PATH}")
print("\nTraining complete. Run  python app.py  to start the web application.")
