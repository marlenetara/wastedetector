from ultralytics import YOLO


# ============================================================
# LOAD PRETRAINED CLASSIFICATION MODEL
# ============================================================

model = YOLO(
    "yolov8n-cls.pt"
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.train(

    data=r"C:\Users\Marlene Tara\WasteDetector\WasteDetector\dataset_16class",

    epochs=50,

    imgsz=224,

    batch=16,

    workers=4,

    verbose=True
)
