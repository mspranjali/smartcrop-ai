#!/usr/bin/env bash
# build.sh - Render build script
# Installs dependencies and trains the ML model if not already present.
set -o errexit

pip install -r requirements.txt

if [ ! -f "models/best_model.pkl" ]; then
  echo "Training ML model..."
  python train_model.py
else
  echo "Model already present, skipping training."
fi
