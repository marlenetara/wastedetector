# ♻️ WasteDetect AI — Real-Time German Waste Vision & Classification System

**WasteDetect AI** is an automated, real-time computer vision system built to detect and classify waste objects according to the Official German Dual System (*Duales System Deutschland*) recycling regulations to assist you. 

Designed for low-latency CPU edge inference, the application utilizes object detection neural networks (YOLOv8/v10) with ONNX runtime acceleration, coupled with a Tkinter GUI that provides visual feedback and sorting rules compliant with the **Verpackungsgesetz (VerpackG)**.

<img width="1598" height="963" alt="Waste Detector Image" src="https://github.com/user-attachments/assets/8910d5a6-f73f-4ea9-802a-276a8b74096d" />

---

## 📸 Key Architecture & Features

### 1. 🧠 Dynamic Model Pipeline & CPU Acceleration
* **Custom Model Auto-Discovery**: Automatically searches for domain-specific custom-trained weights (`best.pt`) across project directories.
* **CPU Fallback & ONNX Optimization**: If custom weights are missing, it degrades to pre-trained architectures (`yolov10s.pt`, `yolov8s.pt`, `yolov10n.pt`, `yolov8n.pt`).
* **Real-Time ONNX Export**: Automatically converts PyTorch models to ONNX runtime format (`imgsz=416`) on initialization for reduced frame latency on CPU hardware.

### 2. 🎯 Subject-Focus Tracking (`select_held_object`)
Instead of displaying all objects in a room, WasteDetect AI employs a spatial scoring algorithm to isolate objects held directly up to the webcam and creates a frame around those:
* **Spatial Proximity Scoring**: Prioritizes objects presented near the center of the camera frame.
* **Area Ratio Thresholding**: Filters out tiny specks (<0.2%) and full-frame background predictions (>85%).
* **Desk Filtering Heuristics**: Suppresses static desk items (e.g., stationary mice or keyboards resting at y > 78% of frame height).
* **Background Filtering**: Excludes non-waste entities such as humans, faces, hands, furniture, and home spaces by using `IGNORED_CLASSES`.
* **Memory Decay Buffer**: Uses a frame decay system (`MAX_MEMORY_DECAY = 4`) to eliminate visual flickering or lagging during short detection drops.

### 3. 🇩🇪 Official German Recycling Logic Matrix (`match_waste_rule`)
Objects detected by the visual model are mapped to official German disposal categories aligned with the bins' official colors:

| Disposal Stream | Bin Designation | Color Code (BGR) | Example Mapped Items |
| :--- | :--- | :--- | :--- |
| **Gelber Sack / Yellow Bin** | Lightweight Packaging | `(0, 215, 255)` | Plastics, cans, composite packaging, Tetra Pak |
| **Altglas** | Glass Container | `(129, 185, 16)` | Bottles, jars, preserve containers |
| **Biomüll** | Compost / Organic | `(34, 139, 34)` | Food scraps, fruits, compostable waste |
| **Altpapier** | Blue Paper Bin | `(246, 130, 59)` | Paper, cardboard, flattened shipping boxes |
| **Elektroschrott** | E-Waste / Recycling Center | `(0, 0, 230)` | Batteries, electronics, computer peripherals |
| **Pfandautomat** | Deposit Return Machine | `(0, 0, 230)` | Bottles and cans with the *Pfand* mark (€0.25 / €0.15) |
| **Restmüll** | Residual Waste (Default) | `(80, 80, 80)` | Non-recyclables, drinking glasses, sanitary items |

---

## 🛠️ System Requirements & Dependencies

* **OS**: Windows, macOS, or Linux
* **Python**: 3.9+
* **Webcam**: Standard USB Webcam

### Required Libraries
```bash
pip install ultralytics opencv-python pillow numpy
