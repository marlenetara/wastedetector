# ============================================================
# INTELLIGENTE DEUTSCHE MÜLLTRENNUNG (GERMAN WASTE DETECTOR)
# Real-Time Automatic Object Detection & Bin Classification
# Configured for ISY IW-1000-1 HD USB Webcam
# ============================================================

from tkinter import Tk, Label, PhotoImage, Frame
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

# Confidence Threshold (Tuned to 0.35 to eliminate ghost detections and false electro)
MIN_CONFIDENCE = 0.35

# Resolution optimized for CPU performance (416x416)
INFERENCE_SIZE = 416

# Custom trained model paths
MODEL_CANDIDATES = [
    BASE_DIR / "runs" / "detect" / "german_waste_model" / "weights" / "best.pt",
    BASE_DIR / "Recibot" / "runs" / "detect" / "german_waste_model" / "weights" / "best.pt",
    BASE_DIR / "Recibot" / "runs" / "train" / "classify" / "train7" / "weights" / "best.pt",
    BASE_DIR / "weights" / "best.pt",
    BASE_DIR / "best.pt",
    BASE_DIR.parent / "best.pt",
]

# Fast, accurate CPU fallback models
ACCURATE_CPU_MODELS = ["yolov10s.pt", "yolov8s.pt", "yolov10n.pt", "yolov8n.pt"]

# ============================================================
# OFFICIAL GERMAN WASTE DEFINITIONS (DUALES SYSTEM - ENGLISH TIPS)
# ============================================================

RESTMUELL_DEFAULT_RULE = {
    "bin": "Restmüll (Grey Bin)",
    "color": (80, 80, 80),        # Dark Grey BGR
    "hex": "#94A3B8",
    "tip": "Item cannot be uniquely categorized. When in doubt, dispose in general waste.",
    "group": "residual"
}

GELBER_SACK_RULE = {
    "bin": "Gelber Sack / Yellow Bin",
    "color": (0, 215, 255),       # Yellow BGR
    "hex": "#FBBF24",
    "tip": "Lightweight packaging made of plastic, metal & composite materials (e.g., Tetra Pak). Note: If this bottle is GLASS, dispose in Altglas.",
    "group": "yellow_bin"
}

ALTGLAS_UNIFIED_RULE = {
    "bin": "Altglas (Green / Brown / White)",
    "color": (129, 185, 16),      # Emerald Green BGR
    "hex": "#10B981",
    "tip": "Dispose in matching glass container (Green, Brown, or White). Blue glass goes to Green glass!",
    "group": "glass"
}

ALTPAPIER_RULE = {
    "bin": "Altpapier (Blue Bin)",
    "color": (246, 130, 59),      # Royal Blue BGR
    "hex": "#3B82F6",
    "tip": "Clean and dry paper or flattened cardboard containers only.",
    "group": "paper"
}

BIOMUELL_RULE = {
    "bin": "Biomüll (Brown Bin)",
    "color": (34, 139, 34),       # Brown BGR
    "hex": "#34D399",
    "tip": "Organic compostable waste. NO plastic bags or compostable plastic bags!",
    "group": "organic"
}

PFAND_RULE = {
    "bin": "Pfandautomat (€0.25 / €0.15)",
    "color": (0, 0, 230),         # Red BGR
    "hex": "#F87171",
    "tip": "Return deposit bottles & cans with the Pfand logo to the supermarket deposit machine.",
    "group": "deposit"
}

ELEKTRO_RULE = {
    "bin": "Elektroschrott / Recycling Center",
    "color": (0, 0, 230),         # Red BGR
    "hex": "#F87171",
    "tip": "Electronics & batteries must NEVER be thrown into household trash!",
    "group": "deposit"
}

# ============================================================
# GERMAN WASTE MAPPING TABLE & KEYWORDS
# ============================================================

GERMAN_WASTE_RULES = {
    # --------------------------------------------------------
    # 1. GELBER SACK (Lightweight packaging, plastic & metal)
    # Corresponding to dataset: plastic_bottle, can_metal, plastic_packaging, tetra_pak
    # --------------------------------------------------------
    "plastic_bottle": GELBER_SACK_RULE,
    "can_metal": GELBER_SACK_RULE,
    "plastic_packaging": GELBER_SACK_RULE,
    "tetra_pak": GELBER_SACK_RULE,
    "plastic_bag": GELBER_SACK_RULE,
    "plastic_container": GELBER_SACK_RULE,
    "wrapper": GELBER_SACK_RULE,
    "chips_packet": GELBER_SACK_RULE,
    "plastic_wrap": GELBER_SACK_RULE,
    "bottle": GELBER_SACK_RULE,
    "cup": GELBER_SACK_RULE,
    "can": PFAND_RULE,

    # --------------------------------------------------------
    # 2. ALTGLAS (Glass containers & jars)
    # Corresponding to dataset: glass_container
    # --------------------------------------------------------
    "glass_container": ALTGLAS_UNIFIED_RULE,
    "glass_bottle": ALTGLAS_UNIFIED_RULE,
    "jar": ALTGLAS_UNIFIED_RULE,
    # General bottles default to Yellow Bin/Pfand
    
    # --------------------------------------------------------
    # 3. BIOMÜLL (Organic food & compostable waste)
    # Corresponding to dataset: organic_waste
    # --------------------------------------------------------
    "organic_waste": BIOMUELL_RULE,
    "banana": BIOMUELL_RULE,
    "apple": BIOMUELL_RULE,
    "sandwich": BIOMUELL_RULE,
    "orange": BIOMUELL_RULE,
    "broccoli": BIOMUELL_RULE,
    "carrot": BIOMUELL_RULE,
    "pizza": {
        "bin": "Biomüll (Food Waste)",
        "color": (42, 42, 165),
        "hex": "#34D399",
        "tip": "Food scraps go into organic waste, greasy pizza boxes go into general waste!",
        "group": "organic"
    },

    # --------------------------------------------------------
    # 4. ALTPAPIER (Paper & Cardboard)
    # Corresponding to dataset: paper_cardboard
    # --------------------------------------------------------
    "paper_cardboard": ALTPAPIER_RULE,
    "paper": ALTPAPIER_RULE,
    "cardboard": ALTPAPIER_RULE,
    "box": ALTPAPIER_RULE,
    "book": ALTPAPIER_RULE,

    # --------------------------------------------------------
    # 5. RESTMÜLL (Strict German Default - General Residual Waste)
    # Corresponding to dataset: sanitary_waste, ceramic_rubbish
    # --------------------------------------------------------
    "sanitary_waste": RESTMUELL_DEFAULT_RULE,
    "ceramic_rubbish": RESTMUELL_DEFAULT_RULE,
    "toothbrush": {
        "bin": "Restmüll (Grey Bin)",
        "color": (60, 60, 60),
        "hex": "#94A3B8",
        "tip": "Old toothbrushes belong in general residual waste.",
        "group": "residual"
    },
    "scissors": RESTMUELL_DEFAULT_RULE,
    "wine glass": {
        "bin": "Restmüll! (NOT Altglas!)",
        "color": (70, 70, 70),
        "hex": "#F87171",
        "tip": "Drinking glasses belong in general waste due to a different melting point!",
        "group": "residual"
    },

    # --------------------------------------------------------
    # 6. ELEKTROSCHROTT (Batteries, Mouse, Keyboard, Laptop)
    # Corresponding to dataset: battery_electronic
    # --------------------------------------------------------
    "battery_electronic": ELEKTRO_RULE,
    "battery": ELEKTRO_RULE,
    "mouse": ELEKTRO_RULE,
    "keyboard": ELEKTRO_RULE,
    "laptop": ELEKTRO_RULE,
}

ORGANIC_KEYWORDS = ["banana", "apple", "sandwich", "orange", "food", "fruit", "vegetable", "broccoli", "carrot", "pizza"]
PAPER_KEYWORDS = ["paper", "cardboard", "box", "carton", "book"]
GLASS_KEYWORDS = ["glass", "jar", "glass_bottle", "wine", "glass_container"]
PLASTIC_KEYWORDS = ["plastic", "pet", "wrapper", "chips_packet", "tetra_pak", "tetra"]

# Verified electronic classes only (from training datase)
VERIFIED_ELEKTRO = {"battery", "battery_electronic", "mouse", "keyboard", "laptop"}

# ============================================================
# STRICT RULE MATCHING LOGIC
# ============================================================

def match_waste_rule(class_name, conf=1.0):
    """
    Evaluates detected object against strict German waste separation rules.
    If uncertain, low confidence, or unmapped: ALWAYS defaults to Restmüll (Grey Bin).
    Elektro is strictly reserved for verified electronics (batteries, mouse, keyboard, laptop).
    """
    class_name = class_name.lower().strip()

    # 1. Direct rule match
    if class_name in GERMAN_WASTE_RULES:
        rule = GERMAN_WASTE_RULES[class_name]
        # For electronic items, ensure minimum certainty before accepting
        if rule == ELEKTRO_RULE and conf < 0.40:
            return RESTMUELL_DEFAULT_RULE
        return rule

    # 2. Electronics verification (Batteries, Mouse, Keyboard, Laptop)
    # In COCO models, cell phone / remote are frequently hallucinated to other objects.
    # Therefore, cell phone / remote require high certainty (>= 0.65). If not sure, default to Restmüll!
    if class_name in ("cell phone", "mobile phone", "remote"):
        if conf >= 0.65:
            return ELEKTRO_RULE
        else:
            return RESTMUELL_DEFAULT_RULE

    if class_name in VERIFIED_ELEKTRO:
        if conf >= 0.40:
            return ELEKTRO_RULE
        else:
            return RESTMUELL_DEFAULT_RULE

    # 3. Organic keywords
    if any(k == class_name or k in class_name for k in ORGANIC_KEYWORDS):
        return BIOMUELL_RULE

    # 4. Paper keywords
    if any(k == class_name or k in class_name for k in PAPER_KEYWORDS):
        return ALTPAPIER_RULE

    # 5. Glass keywords
    if any(k == class_name or k in class_name for k in GLASS_KEYWORDS):
        return ALTGLAS_UNIFIED_RULE

    # 6. Plastic keywords
    if any(k == class_name or k in class_name for k in PLASTIC_KEYWORDS):
        return GELBER_SACK_RULE

    # 7. STRICT GERMAN DEFAULT: When in doubt, it is Restmüll (never electro!)
    return RESTMUELL_DEFAULT_RULE

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
MAX_MEMORY_DECAY = 4


# ============================================================
# MODEL LOADER
# ============================================================

def load_detection_model():
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
                print(f"[Model Loader] Accelerating model for CPU (Exporting {fallback} -> ONNX)...")
                pt_model.export(format="onnx", imgsz=INFERENCE_SIZE, simplify=True)

            if onnx_path.exists():
                model = YOLO(str(onnx_path))
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
        lblimg.config(image='', text="Waiting for item...", fg="#475569", font=("Segoe UI", 9))
    if lblimgtxt is not None:
        lblimgtxt.config(image='', text="Category Details\nwill appear here", fg="#475569", font=("Segoe UI", 9))


def show_category_graphics(group_name):
    try:
        if group_name in class_images and lblimg is not None:
            photo1 = class_images[group_name]
            lblimg.configure(image=photo1, text="")
            lblimg.image = photo1

        if group_name in text_images and lblimgtxt is not None:
            photo2 = text_images[group_name]
            lblimgtxt.configure(image=photo2, text="")
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
# RULE MATCHING LOGIC & OBJECT FILTERING
# ============================================================

IGNORED_CLASSES = {
    "person", "human", "man", "woman", "child", "people",
    "hand", "face", "arm", "body", "hair", "head", "eyeglasses", 
    "chair", "couch", "sofa", "bed", "dining table", "tv",
    "refrigerator", "oven", "sink", "microwave", "toilet", "clock", "potted plant"
}


def select_held_object(boxes, frame_width, frame_height):
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0
    max_dist = math.sqrt(center_x**2 + center_y**2)

    best_box = None
    best_score = -1.0

    for box in boxes:
        cls_id = int(box.cls[0].cpu().numpy())
        class_name = model.names.get(cls_id, str(cls_id)).lower()

        if class_name in IGNORED_CLASSES or "person" in class_name or "hand" in class_name:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())

        box_w = max(0, x2 - x1)
        box_h = max(0, y2 - y1)
        area_ratio = (box_w * box_h) / (frame_width * frame_height)

        # Ignore tiny specks (<0.2%) or massive full-screen bounding boxes (>85%)
        if area_ratio < 0.002 or area_ratio > 0.85:
            continue

        bx = (x1 + x2) / 2.0
        by = (y1 + y2) / 2.0

        # Desk filter: A stationary mouse or keyboard resting at the bottom of the frame
        # on the desk is not considered held up unless brought towards the center.
        if class_name in ("mouse", "keyboard") and by > frame_height * 0.78 and y2 > frame_height * 0.90:
            continue

        dist = math.sqrt((bx - center_x)**2 + (by - center_y)**2)
        norm_dist = dist / max_dist

        # Favor objects presented centrally to the webcam
        score = conf * (1.0 + 1.2 * area_ratio) * (1.0 - 0.45 * norm_dist)

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
    if not ret or frame is None:
        lblVideo.after(SCAN_DELAY_MS, scanning)
        return

    orig_h, orig_w = frame.shape[:2]

    results = model.predict(source=frame, conf=MIN_CONFIDENCE, imgsz=INFERENCE_SIZE, verbose=False)[0]
    boxes = results.boxes

    held_object = None
    if boxes is not None and len(boxes) > 0:
        held_object = select_held_object(boxes, orig_w, orig_h)

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

        rule = match_waste_rule(class_name, conf=conf)

        box_color = rule["color"]
        german_bin = rule["bin"]
        tip_text = rule["tip"]
        image_group = rule["group"]
        accent_color = rule.get("hex", "#94A3B8")

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 4)

        bin_short = german_bin.split("(")[0].strip()
        label_text = f"{bin_short} ({int(conf * 100)}%)"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(display_frame, (x1, max(0, y1 - 32)), (x1 + tw + 12, y1), box_color, -1)
        cv2.putText(display_frame, label_text, (x1 + 6, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        lblClase.config(
            text=f"➔  {german_bin.upper()}   ({int(conf * 100)}%)",
            fg=accent_color,
            bg="#111827",
            font=("Segoe UI", 15, "bold")
        )
        lblClase.lift()

        # Update Right Panel Tip Text
        if lblTip is not None:
            lblTip.config(
                text=f"{tip_text}",
                fg="#F8FAFC",
                bg="#131C31"
            )
            lblTip.lift()

        show_category_graphics(image_group)

    else:
        lblClase.config(
            text="Hold an item in front of the camera...",
            fg="#94A3B8",
            bg="#111827",
            font=("Segoe UI", 15, "bold")
        )
        lblClase.lift()

        if lblTip is not None:
            lblTip.config(
                text="Hold a waste item in front of the camera to view specific German recycling regulations.",
                fg="#94A3B8",
                bg="#131C31"
            )
            lblTip.lift()

        clean_labels()

    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (640, 480))

    image = Image.fromarray(frame_resized)
    image_tk = ImageTk.PhotoImage(image=image)

    lblVideo.configure(image=image_tk)
    lblVideo.image = image_tk

    lblVideo.after(SCAN_DELAY_MS, scanning)


# ============================================================
# MAIN WINDOW & WEBCAM INITIALIZATION
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

    # ============================================================
    # TOP HEADER: BRANDING & LIVE STATUS
    # ============================================================
    lblBrand = Label(pantalla, text="WASTEDETECT", font=("Segoe UI", 16, "bold"), fg="#F8FAFC", bg="#0B0F19")
    lblBrand.place(x=38, y=16)
    lblAI = Label(pantalla, text="AI", font=("Segoe UI", 16, "bold"), fg="#38BDF8", bg="#0B0F19")
    lblAI.place(x=196, y=16)
    lblSub = Label(pantalla, text="GERMAN WASTE VISION & SORTING SYSTEM", font=("Segoe UI", 8, "bold"), fg="#64748B", bg="#0B0F19")
    lblSub.place(x=38, y=46)

    # Top Center Classification Banner
    frmBanner = Frame(pantalla, bg="#111827", highlightbackground="#1E293B", highlightthickness=1)
    frmBanner.place(x=315, y=16, width=650, height=52)

    lblClase = Label(
        frmBanner,
        text="Hold an item in front of the camera...",
        font=("Segoe UI", 14, "bold"),
        justify="center",
        fg="#94A3B8",
        bg="#111827",
        bd=0,
        highlightthickness=0
    )
    lblClase.place(relx=0.5, rely=0.5, anchor="center")

    # Top Right Live Camera Status Pill
    frmLive = Frame(pantalla, bg="#0F172A", highlightbackground="#166534", highlightthickness=1)
    frmLive.place(x=1065, y=22, width=175, height=38)

    lblDot = Label(frmLive, text="●", font=("Segoe UI", 11), fg="#22C55E", bg="#0F172A")
    lblDot.place(x=14, y=7)
    lblLiveTxt = Label(frmLive, text="AI CAMERA ACTIVE", font=("Segoe UI", 9, "bold"), fg="#F8FAFC", bg="#0F172A")
    lblLiveTxt.place(x=34, y=9)

    # ============================================================
    # LEFT PANEL: ACTIVE DISPOSAL STREAM
    # ============================================================
    frmLeft = Frame(pantalla, bg="#0F172A", highlightbackground="#1E293B", highlightthickness=1)
    frmLeft.place(x=35, y=82, width=250, height=630)

    lblLeftHeader = Label(
        frmLeft,
        text="DISPOSAL STREAM",
        font=("Segoe UI", 11, "bold"),
        fg="#E2E8F0",
        bg="#1E293B"
    )
    lblLeftHeader.place(x=12, y=12, width=224, height=34)

    # Left Icon Slot Box
    frmLeftSlot = Frame(frmLeft, bg="#131C31", highlightbackground="#1E293B", highlightthickness=1)
    frmLeftSlot.place(x=19, y=155, width=210, height=210)

    lblimg = Label(
        frmLeftSlot,
        bg="#131C31",
        bd=0,
        highlightthickness=0,
        text="Waiting for item...",
        fg="#475569",
        font=("Segoe UI", 9)
    )
    lblimg.place(x=5, y=5, width=200, height=200)

    # Left Footer Info
    lblLeftFtr1 = Label(frmLeft, text="OFFICIAL GERMAN BIN", font=("Segoe UI", 10, "bold"), fg="#94A3B8", bg="#0F172A")
    lblLeftFtr1.place(x=12, y=430, width=224)
    lblLeftFtr2 = Label(frmLeft, text="MÜLLTRENNUNG CATEGORY", font=("Segoe UI", 8), fg="#64748B", bg="#0F172A")
    lblLeftFtr2.place(x=12, y=456, width=224)

    # ============================================================
    # CENTER PANEL: CAMERA VIEWPORT & TELEMETRY
    # ============================================================
    frmCenter = Frame(pantalla, bg="#0D121E", highlightbackground="#1E293B", highlightthickness=1)
    frmCenter.place(x=315, y=82, width=650, height=630)

    lblVideo = Label(frmCenter, bg="#000000", bd=0, highlightthickness=0)
    lblVideo.place(x=5, y=8, width=640, height=480)

    # Center Telemetry Bar
    frmTele = Frame(frmCenter, bg="#111827", highlightbackground="#1E293B", highlightthickness=1)
    frmTele.place(x=10, y=500, width=630, height=115)

    lblT1 = Label(frmTele, text="TARGET TRACKING:", font=("Segoe UI", 9, "bold"), fg="#94A3B8", bg="#111827")
    lblT1.place(x=18, y=14)
    lblT1V = Label(frmTele, text="ACTIVE (HELD OBJECT ONLY)", font=("Segoe UI", 9, "bold"), fg="#22C55E", bg="#111827")
    lblT1V.place(x=155, y=14)

    lblT2 = Label(frmTele, text="FILTER APPLIED:", font=("Segoe UI", 9, "bold"), fg="#94A3B8", bg="#111827")
    lblT2.place(x=18, y=44)
    lblT2V = Label(frmTele, text="PEOPLE, HANDS & ROOM BACKGROUND EXCLUDED", font=("Segoe UI", 9), fg="#38BDF8", bg="#111827")
    lblT2V.place(x=155, y=44)

    lblT3 = Label(frmTele, text="DETECTION ENGINE:", font=("Segoe UI", 9, "bold"), fg="#94A3B8", bg="#111827")
    lblT3.place(x=18, y=74)
    lblT3V = Label(frmTele, text="YOLO REAL-TIME VISION • CONFIDENCE >= 20%", font=("Segoe UI", 9), fg="#64748B", bg="#111827")
    lblT3V.place(x=155, y=74)

    # ============================================================
    # RIGHT PANEL: RECYCLING TIPS & REGULATIONS
    # ============================================================
    frmRight = Frame(pantalla, bg="#0F172A", highlightbackground="#1E293B", highlightthickness=1)
    frmRight.place(x=995, y=82, width=250, height=630)

    # Right Panel Header
    lblRightHeader = Label(
        frmRight,
        text="RECYCLING TIP",
        font=("Segoe UI", 11, "bold"),
        fg="#E2E8F0",
        bg="#1E293B"
    )
    lblRightHeader.place(x=12, y=12, width=224, height=34)

    # BOX 1: TIP BOX (Top portion of right card)
    frmTipBox = Frame(frmRight, bg="#131C31", highlightbackground="#1E293B", highlightthickness=1)
    frmTipBox.place(x=19, y=58, width=210, height=185)

    lblTipTitle = Label(
        frmTipBox,
        text="💡 TIP",
        font=("Segoe UI", 10, "bold"),
        fg="#38BDF8",
        bg="#1E293B"
    )
    lblTipTitle.place(x=8, y=8, width=192, height=26)

    lblTip = Label(
        frmTipBox,
        text="Hold a waste item in front of the camera to view specific German recycling regulations.",
        font=("Segoe UI", 10),
        fg="#94A3B8",
        bg="#131C31",
        bd=0,
        highlightthickness=0,
        wraplength=188,
        justify="center"
    )
    lblTip.place(x=8, y=40, width=192, height=135)

    # BOX 2: CATEGORY GRAPHIC CARD (Bottom portion of right card, perfectly aligned!)
    frmCatBox = Frame(frmRight, bg="#131C31", highlightbackground="#1E293B", highlightthickness=1)
    frmCatBox.place(x=19, y=256, width=210, height=210)

    lblimgtxt = Label(
        frmCatBox,
        bg="#131C31",
        bd=0,
        highlightthickness=0,
        text="Category Details\nwill appear here",
        fg="#475569",
        font=("Segoe UI", 9)
    )
    lblimgtxt.place(x=5, y=5, width=200, height=200)

    # Right Footer Info
    lblRightFtr1 = Label(frmRight, text="OFFICIAL TIP", font=("Segoe UI", 10, "bold"), fg="#94A3B8", bg="#0F172A")
    lblRightFtr1.place(x=12, y=498, width=224)
    lblRightFtr2 = Label(frmRight, text="VERPACKUNGSGESETZ COMPLIANT", font=("Segoe UI", 8), fg="#64748B", bg="#0F172A")
    lblRightFtr2.place(x=12, y=524, width=224)
    lblRightFtr3 = Label(frmRight, text="DUALES SYSTEM DEUTSCHLAND", font=("Segoe UI", 8), fg="#475569", bg="#0F172A")
    lblRightFtr3.place(x=12, y=548, width=224)

    # Load Model & Assets
    load_detection_model()
    load_gui_assets()

    # ============================================================
    # PHYSICAL USB WEBCAM INITIALIZATION (SEARCHES INDEX 2 FIRST, SKIPS DROIDCAM)
    # ============================================================
    print("[Kamera] Connecting physical ISY USB Webcam...")
    cap = None

    camera_indices_to_try = [2, 0, 1, 3]
    backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for dev_idx in camera_indices_to_try:
        for backend in backends_to_try:
            try:
                temp_cap = cv2.VideoCapture(dev_idx, backend)
                if temp_cap.isOpened():
                    temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    ret, test_frame = temp_cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        cap = temp_cap
                        print(f"[Kamera] Connected successfully on Index {dev_idx}")
                        break
                temp_cap.release()
            except Exception:
                pass
        if cap is not None and cap.isOpened():
            break

    if cap is None or not cap.isOpened():
        lblClase.config(text="Camera could not be opened!", fg="red")
        print("[Kamera] Error: No working camera found. Check Windows Privacy Settings.")
        return

    # Set buffer size to 1 to eliminate frame delay
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    pantalla.protocol("WM_DELETE_WINDOW", lambda: on_close(pantalla))

    # Start loop
    scanning()
    pantalla.mainloop()


if __name__ == "__main__":
    ventana_principal()