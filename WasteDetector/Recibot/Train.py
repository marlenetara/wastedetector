import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

# ============================================================
# CONFIGURATION & PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
YAML_PATH = BASE_DIR / "Recibot" / "dataset.yaml"

# Choose device: Automatically detect CUDA GPU, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("============================================================")
print(f" GERMAN WASTE DETECTOR - MODEL TRAINING (YOLOv8)")
print(f" Computing Device : {device.upper()}")
print(f" Config File      : {YAML_PATH}")
print("============================================================")

# ============================================================
# 1. LOAD PRETRAINED OBJECT DETECTION MODEL
# ============================================================
# IMPORTANT:
# We use 'yolov8n.pt' (Object Detection with Bounding Boxes),
# NOT 'yolov8n-cls.pt' (Classification).
# This allows the AI to draw boxes around held items automatically!
print("\n[1/3] Loading YOLOv8 Object Detection base model...")
model = YOLO("yolov8n.pt")

# ============================================================
# 2. DATASET VERIFICATION
# ============================================================
print("\n[2/3] Checking dataset configuration...")
if not YAML_PATH.exists():
    print(f"Error: {YAML_PATH} not found.")
    sys.exit(1)

# ============================================================
# 3. TRAIN THE MODEL
# ============================================================
# Hyperparameters are tuned for high accuracy and CPU compatibility:
# - imgsz: 416 (Fast & lightweight on CPU, excellent accuracy)
# - epochs: 20-30 (Sufficient for transfer learning convergence)
# - batch: 8 (Low memory footprint)
# - workers: 2 (Stable on Windows)
print("\n[3/3] Starting training...")
print("The model will learn to locate objects and predict their exact German bin.\n")

try:
    results = model.train(
        data=str(YAML_PATH),
        epochs=25,
        imgsz=416,
        batch=8,
        workers=2,
        device=device,
        project=str(BASE_DIR / "runs" / "detect"),
        name="german_waste_model",
        save=True,
        verbose=True,
        plots=True
    )
    print("\n============================================================")
    print(" TRAINING COMPLETE!")
    print(f" Best model weights saved to:")
    print(f" {BASE_DIR / 'runs' / 'detect' / 'german_waste_model' / 'weights' / 'best.pt'}")
    print("============================================================")
except Exception as e:
    print(f"\n[Training note / Error]: {e}")
    print("\nIf you need to download a pre-annotated dataset first:")
    print("Use Roboflow Universe or TACO detection datasets exported as YOLOv8 PyTorch.")
