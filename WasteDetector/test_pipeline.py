import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

print("Testing YOLO Object Detection Pipeline...")

# Test 1: Load base detector
model = YOLO("yolov8n.pt")
print(f"[OK] Model loaded: {type(model)}")
print(f"[OK] Total classes: {len(model.names)}")

# Test 2: Dummy image prediction
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
results = model.predict(source=dummy, conf=0.25, verbose=False)
print(f"[OK] Prediction ran successfully! Found {len(results[0].boxes)} boxes on blank frame.")

# Test 3: Check German waste mapping
from app import GERMAN_WASTE_RULES, select_held_object
print(f"[OK] Loaded {len(GERMAN_WASTE_RULES)} German waste rules.")
for key in ["plastic_bottle", "can_metal", "banana", "paper_cardboard", "glass_container", "sanitary_waste"]:
    rule = GERMAN_WASTE_RULES.get(key)
    assert rule is not None, f"Missing rule for {key}"
    print(f"   - {key:20s} -> {rule['bin']} ({rule['name']})")

print("\nALL PIPELINE CHECKS PASSED!")

