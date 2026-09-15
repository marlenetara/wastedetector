import cv2
import numpy as np
import torch
from ultralytics import YOLO
from app import select_held_object, GERMAN_WASTE_RULES, load_detection_model

print("Testing select_held_object with real YOLO predictions...")
model = YOLO("yolov8n.pt")

# Create dummy frame with some synthetic drawing
test_img = np.ones((720, 1280, 3), dtype=np.uint8) * 200
# Draw a rectangle in center resembling a book/paper
cv2.rectangle(test_img, (400, 200), (880, 550), (255, 255, 255), -1)
cv2.putText(test_img, "SAMPLE PAPER / BOOK", (450, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

results = model.predict(source=test_img, conf=0.15, imgsz=640, verbose=False)[0]
boxes = results.boxes

print(f"Detected {len(boxes)} boxes on test image.")
held = select_held_object(boxes, 1280, 720)
print(f"select_held_object returned: {held}")
print("NO RUNTIME ERRORS IN select_held_object!")

