# ============================================================
# INTELLIGENTE DEUTSCHE MÜLLTRENNUNG (GERMAN WASTE DETECTOR)
# Real-Time Automatic Object Detection & Bin Classification
# Strictly Aligned with German Packaging Law (VerpackG)
# ============================================================

from tkinter import Tk, Label, PhotoImage
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import math

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SETUP_DIR = BASE_DIR / "Recibot" / "setUp"

DISPLAY_WIDTH = 640
SCAN_DELAY_MS = 30

# Detection Confidence Threshold
MIN_CONFIDENCE = 0.15

# Resolution optimized for Ryzen 5 CPU performance (416x416)
INFERENCE_SIZE = 416

# Custom trained model paths
MODEL_CANDIDATES = [
    BASE_DIR / "runs" / "detect" / "german_waste_model" / "weights" / "best.pt",
    BASE_DIR / "Recibot" / "runs" / "detect" / "german_waste_model" / "weights" / "best.pt",
]

# Fast, accurate CPU fallback models
ACCURATE_CPU_MODELS = ["yolov10s.pt", "yolov8s.pt", "yolov10n.pt", "yolov8n.pt"]

# ============================================================
# GERMAN WASTE CLASSIFICATION DEFINITIONS (DUALES SYSTEM)
# ============================================================

# DEFAULT RULE IS STRICTLY RESTMÜLL (NON-RECYCLABLE)
RESTMUELL_DEFAULT_RULE = {
    "bin": "Restmüll (Graue Tonne)",
    "color": (80, 80, 80),  # Dark Grey BGR
    "tip": "Nicht eindeutig zuordenbar. Gehört zur Sicherheit in den Restmüll.",
    "group": "residual"
}

GELBER_SACK_RULE = {
    "bin": "Gelber Sack / Gelbe Tonne",
    "color": (0, 215, 255),  # Yellow BGR
    "tip": "Leichtverpackungen aus Kunststoff, Metall & Verbundstoffen (z.B. Tetra Pak).",
    "group": "yellow_bin"
}

ALTGLAS_RULE = {
    "bin": "Altglascontainer (Nach Farbe: Weiß, Braun, Grün)",
    "color": (0, 200, 0),  # Green BGR
    "tip": "Nur Behälterglas! Deckel vorher in den Gelben Sack werfen.",
    "group": "glass"
}

ALTPAPIER_RULE = {
    "bin": "Altpapier (Blaue Tonne)",
    "color": (235, 128, 41),  # Blue BGR
    "tip": "Nur sauberes, trockenes Papier und flachgefaltete Kartons.",
    "group": "paper"
}

BIOMUELL_RULE = {
    "bin": "Biomüll (Braune Tonne)",
    "color": (42, 42, 165),  # Brown BGR
    "tip": "Organische Abfälle. KEINE Plastiktüten oder Biomüll-Plastikbeutel!",
    "group": "organic"
}

PFAND_RULE = {
    "bin": "Pfandautomat (€0.25 / €0.15)",
    "color": (0, 0, 230),  # Red BGR
    "tip": "Einweg- & Mehrweg-Getränkeverpackungen mit DPG-Logo im Supermarkt abgeben.",
    "group": "deposit"
}

# ============================================================
# COMPREHENSIVE GERMAN WASTE MAPPING TABLE
# ============================================================

GERMAN_WASTE_RULES = {
    # --------------------------------------------------------
    # 1. GELBER SACK / GELBE TONNE / WERTSTOFFTONNE (Yellow)
    # --------------------------------------------------------
    "plastic_bottle": GELBER_SACK_RULE,
    "plastic_packaging": GELBER_SACK_RULE,
    "plastic_bag": GELBER_SACK_RULE,
    "plastic_container": GELBER_SACK_RULE,
    "wrapper": GELBER_SACK_RULE,
    "chips_packet": GELBER_SACK_RULE,
    "plastic_wrap": GELBER_SACK_RULE,
    "tetra_pak": GELBER_SACK_RULE,
    "can_metal": GELBER_SACK_RULE,
    "can": PFAND_RULE,
    "cup": GELBER_SACK_RULE,

    # Specific COCO shapes detected for plastics & flexible packaging
    "handbag": GELBER_SACK_RULE,     # Plastic carrier bags / pouches
    "backpack": GELBER_SACK_RULE,    # Large plastic sacks / chip bags
    "cell phone": GELBER_SACK_RULE,  # Metallic foil wrappers / chip packets
    "remote": GELBER_SACK_RULE,      # Small plastic blister packaging
    "mouse": GELBER_SACK_RULE,       # Small plastic food containers

    # --------------------------------------------------------
    # 2. ALTGLAS (Containers & Jars)
    # --------------------------------------------------------
    "glass_container": ALTGLAS_RULE,
    "glass_bottle": ALTGLAS_RULE,

    # WINE GLASS / DRINKING GLASS ARE RESTMÜLL (Special melting point!)
    "wine glass": {
        "bin": "Restmüll! (KEIN Altglas!)",
        "color": (70, 70, 70),
        "tip": "Trinkgläser und Bleikristall haben einen anderen Schmelzpunkt als Altglas!",
        "group": "residual"
    },

    # --------------------------------------------------------
    # 3. BIOMÜLL (Braune Tonne)
    # --------------------------------------------------------
    "organic_waste": BIOMUELL_RULE,
    "banana": BIOMUELL_RULE,
    "apple": BIOMUELL_RULE,
    "sandwich": BIOMUELL_RULE,
    "orange": BIOMUELL_RULE,
    "broccoli": BIOMUELL_RULE,
    "carrot": BIOMUELL_RULE,
    "pizza": {"bin": "Biomüll (Speisereste)", "color": (42, 42, 165), "tip": "Essensreste in Biomüll, fettigen Karton in Restmüll!", "group": "organic"},

    # --------------------------------------------------------
    # 4. ALTPAPIER (Blaue Tonne)
    # --------------------------------------------------------
    "paper_cardboard": ALTPAPIER_RULE,
    "book": ALTPAPIER_RULE,
    "paper": ALTPAPIER_RULE,
    "cardboard": ALTPAPIER_RULE,
    "box": ALTPAPIER_RULE,

    # --------------------------------------------------------
    # 5. RESTMÜLL (Schwarze / Graue Tonne)
    # --------------------------------------------------------
    "sanitary_waste": RESTMUELL_DEFAULT_RULE,
    "ceramic_rubbish": {
        "bin": "Restmüll (Graue Tonne)",
        "color": (60, 60, 60),
        "tip": "Keramik, Geschirr und Porzellan niemals in den Altglascontainer!",
        "group": "residual"
    },
    "toothbrush": {
        "bin": "Restmüll (Graue Tonne)",
        "color": (60, 60, 60),
        "tip": "Gebruachte Zahnbürsten gehören in den Restmüll.",
        "group": "residual"
    },

    # --------------------------------------------------------
    # 6. SONDERMÜLL / E-SCHROTT
    # --------------------------------------------------------
    "battery_electronic": {
        "bin": "Batterie-Sammelbox / Wertstoffhof",
        "color": (0, 0, 230),
        "tip": "Gefahrgut! Im Supermarkt in die Sammelbox geben.",
        "group": "deposit"
    },
}

PLASTIC_KEYWORDS = ["plastic", "plastico", "pet", "poly", "wrapper", "foil", "chip", "pack"]
GLASS_KEYWORDS = ["glass", "jar", "bottle"]

# ============================================================
# APPLICATION STATE
# ============================================================

cap = None
model = None
model_name = "YOLO"

lblVideo = None
lblimg = None
lblimgtxt = None
lblClase = None
lblTip = None

class_images = {}
text_images = {}

last_held_object = None
memory_decay = 0
MAX_MEMORY_DECAY = 8


# ============================================================
# MODEL LOADER (OPTIMIZED FOR RYZEN CPU)
# ============================================================

def load_detection_model():
    """
    Loads custom waste model if available. Otherwise loads YOLOv10s/v8s
    and exports to ONNX format for AVX CPU acceleration on Ryzen 5.
    """
    global model, model_name

    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            print(f"[Model Loader] Found custom German waste model: {candidate}")
            model = YOLO(str(candidate))
            model_name = "Custom Waste Model"
            return

    for fallback in ACCURATE_CPU_MODELS:
        try:
            print(f"[Model Loader] Loading CPU model ({fallback})...")
            pt_model = YOLO(fallback)
            
            onnx_path = BASE_DIR / fallback.replace(".pt", ".onnx")
            if not onnx_path.exists():
                print(f"[Model Loader] Accelerating model for Ryzen CPU (Exporting {fallback} -> ONNX)...")
                pt_model.export(format="onnx", imgsz=INFERENCE_SIZE, simplify=True)

            if onnx_path.exists():
                model = YOLO(str(onnx_path), task="detect")
                print(f"[Model Loader] ONNX acceleration active: {onnx_path.name}")
            else:
                model = pt_model

            model_name = f"{fallback} (CPU Accelerated)"
            return
        except Exception as e:
            print(f"[Model Loader] Could not load {fallback}: {e}")

    print("[Model Loader] Final fallback to yolov8n.pt")
    model = YOLO("yolov8n.pt")
    model_name = "YOLOv8n"


# ============================================================
# GUI HELPERS
# ============================================================

def clean_labels():
    if lblimg is not None:
        lblimg.config(image='')
    if lblimgtxt is not None:
        lblimgtxt.config(image='')


def show_category_graphics(group_name):
    try:
        if group_name in class_images and lblimg is not None:
            photo1 = class_images[group_name]
            lblimg.configure(image=photo1)
            lblimg.image = photo1

        if group_name in text_images and lblimgtxt is not None:
            photo2 = text_images[group_name]
            lblimgtxt.configure(image=photo2)
            lblimgtxt.image = photo2
    except Exception as e:
        print(f"Error rendering category graphics: {e}")


def load_gui_assets():
    global class_images, text_images
    groups = ["yellow_bin", "organic", "paper", "glass", "residual", "deposit"]

    for group in groups:
        img_path = SETUP_DIR / f"{group}.png"
        txt_path = SETUP_DIR / f"{group}txt.png"

        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                class_images[group] = ImageTk.PhotoImage(image=img)

        if txt_path.exists():
            txt = cv2.imread(str(txt_path))
            if txt is not None:
                txt = cv2.cvtColor(txt, cv2.COLOR_BGR2RGB)
                txt = Image.fromarray(txt)
                text_images[group] = ImageTk.PhotoImage(image=txt)


# ============================================================
# OPENCV MATERIAL ANALYZER (GLASS VS PLASTIC DISAMBIGUATION)
# ============================================================

def detect_container_material(crop):
    """
    Analyzes crop optical attributes (Specular Reflection & Color Variance)
    to differentiate Glass Containers from Plastic Packaging.
    """
    if crop is None or crop.size == 0:
        return "plastic"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 1. Specular Highlight Detection (Glass has sharp bright reflections)
    _, bright_spots = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    highlight_ratio = np.sum(bright_spots > 0) / float(gray.size)

    # 2. Edge crispness & internal reflections
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if highlight_ratio > 0.015 and laplacian_var > 300:
        return "glass"
    return "plastic"


# ============================================================
# GERMAN WASTE RULE MATCHING (STRICT RESTMÜLL DEFAULT)
# ============================================================

IGNORED_CLASSES = {
    "person", "human", "man", "woman", "child", "people",
    "hand", "face", "arm", "body", "hair", "head",
    "chair", "couch", "sofa", "bed", "dining table", "tv", "laptop",
    "refrigerator", "oven", "sink", "microwave", "toilet", "clock", "potted plant"
}


def match_waste_rule(class_name, crop=None):
    """
    Evaluates detected object against official German waste laws.
    STRICT REQUIREMENT: Defaults to Restmüll if not recognized!
    """
    class_name = class_name.lower().strip()

    # Generic "bottle" class resolution using OpenCV optical analysis
    if class_name == "bottle" and crop is not None:
        material = detect_container_material(crop)
        if material == "glass":
            return ALTGLAS_RULE
        else:
            return GELBER_SACK_RULE

    # Direct Dictionary Lookup
    if class_name in GERMAN_WASTE_RULES:
        return GERMAN_WASTE_RULES[class_name]

    # Substring Keyword Match for Plastic or Glass
    if any(k in class_name for k in PLASTIC_KEYWORDS):
        return GELBER_SACK_RULE

    if any(k in class_name for k in GLASS_KEYWORDS):
        return ALTGLAS_RULE

    # STRICT DEFAULT: RESTMÜLL (GRAUE TONNE)
    return RESTMUELL_DEFAULT_RULE


def select_held_object(boxes, frame_width, frame_height):
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0
    max_dist = math.sqrt(center_x**2 + center_y**2)

    best_box = None
    best_score = -1.0

    for box in boxes:
        cls_id = int(box.cls[0].cpu().numpy())
        class_name = model.names.get(cls_id, str(cls_id)).lower()

        # Strict exclusion of humans, hands, and background furniture
        if class_name in IGNORED_CLASSES or "person" in class_name or "hand" in class_name:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())

        box_w = max(0, x2 - x1)
        box_h = max(0, y2 - y1)
        area_ratio = (box_w * box_h) / (frame_width * frame_height)

        if area_ratio < 0.0005:
            continue

        bx = (x1 + x2) / 2.0
        by = (y1 + y2) / 2.0
        dist = math.sqrt((bx - center_x)**2 + (by - center_y)**2)
        norm_dist = dist / max_dist

        score = conf * (1.0 + 1.5 * area_ratio) * (1.0 - 0.15 * norm_dist)

        if score > best_score:
            best_score = score
            best_box = (int(x1), int(y1), int(x2), int(y2), conf, cls_id)

    return best_box


def scanning():
    global cap, lblVideo, lblClase, lblTip
    global last_held_object, memory_decay

    if cap is None or not cap.isOpened():
        return

    ret, frame = cap.read()
    if not ret:
        lblVideo.after(SCAN_DELAY_MS, scanning)
        return

    orig_h, orig_w = frame.shape[:2]

    # Execute CPU inference at 416x416
    results = model.predict(source=frame, conf=MIN_CONFIDENCE, imgsz=INFERENCE_SIZE, verbose=False)[0]
    boxes = results.boxes

    held_object = None
    if boxes is not None and len(boxes) > 0:
        held_object = select_held_object(boxes, orig_w, orig_h)

    # Motion smoothing memory
    if held_object is not None:
        last_held_object = held_object
        memory_decay = MAX_MEMORY_DECAY
    elif memory_decay > 0 and last_held_object is not None:
        held_object = last_held_object
        memory_decay -= 1
    else:
        last_held_object = None

    display_frame = frame.copy()

    if held_object is not None:
        x1, y1, x2, y2, conf, cls_id = held_object
        class_name = model.names.get(cls_id, str(cls_id)).lower()

        # Extract crop for optical material analysis
        crop = frame[max(0, y1):min(orig_h, y2), max(0, x1):min(orig_w, x2)]

        rule = match_waste_rule(class_name, crop)

        box_color = rule["color"]
        german_bin = rule["bin"]
        tip_text = rule["tip"]
        image_group = rule["group"]

        # Draw bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 4)

        # Draw classification label
        bin_short = german_bin.split("(")[0].strip()
        label_text = f"{bin_short} ({int(conf * 100)}%)"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(display_frame, (x1, max(0, y1 - 32)), (x1 + tw + 12, y1), box_color, -1)
        cv2.putText(display_frame, label_text, (x1 + 6, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        accent_color = (
            "#FBBF24" if ("Gelb" in german_bin or "Wertstoff" in german_bin)
            else "#34D399" if "Bio" in german_bin
            else "#60A5FA" if "Papier" in german_bin
            else "#2DD4BF" if "Glas" in german_bin
            else "#F87171" if ("Pfand" in german_bin or "Sonder" in german_bin)
            else "#94A3B8"
        )

        lblClase.config(
            text=f"➔  {german_bin.upper()}   ({int(conf * 100)}%)",
            fg=accent_color,
            bg="#111827",
            font=("Segoe UI", 16, "bold")
        )

        if lblTip is not None:
            lblTip.config(
                text=f"Tipp: {tip_text}",
                fg="#CBD5E1",
                bg="#111827"
            )

        show_category_graphics(image_group)

    else:
        lblClase.config(
            text="Gegenstand in die Kamera halten...",
            fg="#94A3B8",
            bg="#111827",
            font=("Segoe UI", 15, "bold")
        )
        if lblTip is not None:
            lblTip.config(
                text="Halte einen Abfallgegenstand in die Mitte der Kamera • Hände & Personen werden gefiltert",
                fg="#64748B",
                bg="#111827"
            )
        clean_labels()

    # Render frame on Tkinter GUI
    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (640, 480))

    image = Image.fromarray(frame_resized)
    image_tk = ImageTk.PhotoImage(image=image)

    lblVideo.configure(image=image_tk)
    lblVideo.image = image_tk

    lblVideo.after(SCAN_DELAY_MS, scanning)


# ============================================================
# MAIN WINDOW & STARTUP
# ============================================================

def on_close(pantalla):
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    pantalla.destroy()


def ventana_principal():
    global cap, lblVideo, lblimg, lblimgtxt, lblClase, lblTip

    pantalla = Tk()
    pantalla.title("Intelligente Deutsche Mülltrennung - AI Waste Detector")
    pantalla.geometry("1280x740")
    pantalla.resizable(False, False)
    pantalla.configure(bg="#0B0F19")

    # Background Canvas
    bg_path = SETUP_DIR / "Canva.png"
    if bg_path.exists():
        try:
            fondo_img = PhotoImage(file=str(bg_path))
            background = Label(pantalla, image=fondo_img, bg="#0B0F19", bd=0)
            background.place(x=0, y=0, relwidth=1, relheight=1)
            background.image = fondo_img
        except Exception as e:
            print(f"Background notice: {e}")

    # Camera View
    lblVideo = Label(pantalla, bg="#0D121E", bd=0, highlightthickness=0)
    lblVideo.place(x=320, y=115, width=640, height=480)

    # Category Graphics
    lblimg = Label(pantalla, bg="#0F172A", bd=0, highlightthickness=0)
    lblimg.place(x=60, y=260, width=200, height=200)

    lblimgtxt = Label(pantalla, bg="#0F172A", bd=0, highlightthickness=0)
    lblimgtxt.place(x=1020, y=260, width=200, height=200)

    # Result Banner
    lblClase = Label(
        pantalla,
        text="Gegenstand in die Kamera halten...",
        font=("Segoe UI", 16, "bold"),
        justify="center",
        fg="#94A3B8",
        bg="#111827",
        bd=0,
        highlightthickness=0
    )
    lblClase.place(x=320, y=22, width=640, height=42)

    # Recycling Tip Label
    lblTip = Label(
        pantalla,
        text="Automatische Mülltrennung aktiv • Hände & Personen werden gefiltert",
        font=("Segoe UI", 10),
        fg="#64748B",
        bg="#111827",
        bd=0,
        highlightthickness=0
    )
    lblTip.place(x=320, y=66, width=640, height=24)

    # Load Model & GUI Assets
    load_detection_model()
    load_gui_assets()

    # Start Webcam
    print("Starte Webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        lblClase.config(text="Kamera konnte nicht geöffnet werden!", fg="red")
        print("Fehler: Kamera konnte nicht geöffnet werden.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    pantalla.protocol("WM_DELETE_WINDOW", lambda: on_close(pantalla))

    # Start main scanning loop
    scanning()
    pantalla.mainloop()


if __name__ == "__main__":
    ventana_principal()