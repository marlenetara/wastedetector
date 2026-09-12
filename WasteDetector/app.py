# ============================================================
# REQUIRED LIBRARIES
# ============================================================

from tkinter import *
from PIL import Image, ImageTk
import imutils
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

# Folder where this app.py file is located
BASE_DIR = Path(__file__).resolve().parent

# Automatically find the newest trained YOLO model
MODEL_SEARCH_DIR = BASE_DIR / "runs" / "classify"

model_files = list(
    MODEL_SEARCH_DIR.glob("**/weights/best.pt")
)

if model_files:
    MODEL_PATH = max(
        model_files,
        key=lambda p: p.stat().st_mtime
    )
else:
    MODEL_PATH = None

# Folder containing the interface images
SETUP_DIR = BASE_DIR / "Recibot" / "setUp"


# ============================================================
# GLOBAL VARIABLES
# ============================================================

cap = None
model = None
clsName = []

lblVideo = None
lblimg = None
lblimgtxt = None
lblClase = None

class_images = []
text_images = []

# ------------------------------------------------------------
# Variables for selecting an object with the mouse
# ------------------------------------------------------------

selecting = False

start_x = 0
start_y = 0

end_x = 0
end_y = 0

selection_box = None


# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================

def clean_lbl():

    lblimg.config(image='')
    lblimgtxt.config(image='')


def show_images(img, imgtxt):

    # Displays the images corresponding to the waste type

    try:

        for source_img, label in [
            (img, lblimg),
            (imgtxt, lblimgtxt)
        ]:

            source_img = cv2.cvtColor(
                source_img,
                cv2.COLOR_BGR2RGB
            )

            source_img = Image.fromarray(
                source_img
            )

            photo = ImageTk.PhotoImage(
                image=source_img
            )

            label.configure(
                image=photo
            )

            label.image = photo

    except Exception as e:

        print(
            f"Error displaying images: {e}"
        )

        clean_lbl()


# ============================================================
# MOUSE SELECTION
# ============================================================

def mouse_down(event):

    global selecting
    global start_x
    global start_y
    global end_x
    global end_y

    # Start drawing the selection
    selecting = True

    start_x = event.x
    start_y = event.y

    end_x = event.x
    end_y = event.y


def mouse_move(event):

    global end_x
    global end_y

    if selecting:

        end_x = event.x
        end_y = event.y


def mouse_up(event):

    global selecting
    global end_x
    global end_y

    selecting = False

    end_x = event.x
    end_y = event.y

    print(
        f"Selected area: "
        f"({start_x}, {start_y}) -> "
        f"({end_x}, {end_y})"
    )


def clear_selection():

    global start_x
    global start_y
    global end_x
    global end_y

    start_x = 0
    start_y = 0
    end_x = 0
    end_y = 0

    lblClase.config(
        text="Select an object",
        fg="black"
    )

    clean_lbl()


# ============================================================
# GET SELECTED OBJECT
# ============================================================

def get_selected_crop(frame):

    """
    Converts the mouse selection from the displayed
    640-pixel-wide image back to the original camera
    resolution and returns only that area.
    """

    global start_x
    global start_y
    global end_x
    global end_y

    # Display dimensions
    display_width = 640

    # Original camera dimensions
    original_height, original_width = frame.shape[:2]

    # Calculate how the displayed image was scaled
    scale = display_width / original_width

    # Avoid division by zero
    if scale <= 0:
        return None

    # Convert displayed coordinates to original coordinates
    x1 = int(
        min(start_x, end_x) / scale
    )

    y1 = int(
        min(start_y, end_y) / scale
    )

    x2 = int(
        max(start_x, end_x) / scale
    )

    y2 = int(
        max(start_y, end_y) / scale
    )

    # Keep coordinates inside the image
    x1 = max(
        0,
        min(x1, original_width)
    )

    x2 = max(
        0,
        min(x2, original_width)
    )

    y1 = max(
        0,
        min(y1, original_height)
    )

    y2 = max(
        0,
        min(y2, original_height)
    )

    # Make sure there is actually an area selected
    if x2 <= x1 or y2 <= y1:

        return None

    # Crop ONLY the selected object
    crop = frame[
        y1:y2,
        x1:x2
    ]

    return crop


# ============================================================
# CAMERA + AI CLASSIFICATION
# ============================================================

def scanning():

    if cap is not None and cap.isOpened():

        ret, frame = cap.read()

        if not ret:

            print(
                "Error capturing frame"
            )

            cap.release()

            return

        # ----------------------------------------------------
        # Convert image for display
        # ----------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Resize camera image to 640 px wide
        # ----------------------------------------------------

        frame_resized = imutils.resize(
            frame_rgb,
            width=640
        )

        # ----------------------------------------------------
        # Draw selection rectangle
        # ----------------------------------------------------

        if (
            start_x != end_x
            and start_y != end_y
        ):

            cv2.rectangle(
                frame_resized,
                (start_x, start_y),
                (end_x, end_y),
                (255, 0, 0),
                3
            )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        try:

            # Only classify when the user has selected
            # an actual area
            if (
                abs(end_x - start_x) > 10
                and abs(end_y - start_y) > 10
            ):

                # Get ONLY the selected object
                selected_object = get_selected_crop(
                    frame
                )

                if selected_object is not None:

                    # ----------------------------------------
                    # Send ONLY selected object to AI
                    # ----------------------------------------

                    result = model.predict(
                        source=selected_object,
                        save=False,
                        verbose=False
                    )[0]

                    probs = result.probs

                    if probs is not None:

                        # Get probabilities
                        probabilities = (
                            probs.data.cpu().numpy()
                        )

                        # Highest probability class
                        class_index = int(
                            np.argmax(probabilities)
                        )

                        # Confidence
                        confidence = float(
                            np.max(probabilities)
                        )

                        # Class name
                        label_text = (
                            f"{clsName[class_index]} "
                            f"{int(confidence * 100)}%"
                        )

                        # ------------------------------------
                        # Update GUI text
                        # ------------------------------------

                        lblClase.config(
                            text=label_text,
                            fg="green",
                            font=(
                                "Arial",
                                18,
                                "bold"
                            )
                        )

                        # ------------------------------------
                        # Display result over camera
                        # ------------------------------------

                        cv2.putText(
                            frame_resized,
                            label_text,
                            (
                                10,
                                40
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (255, 0, 0),
                            3
                        )

                        # ------------------------------------
                        # Show category images
                        # ------------------------------------

                        if (
                            class_index <
                            len(class_images)
                        ):

                            show_images(
                                class_images[
                                    class_index
                                ],
                                text_images[
                                    class_index
                                ]
                            )

                    else:

                        clean_lbl()

            else:

                lblClase.config(
                    text="Select an object",
                    fg="black",
                    font=(
                        "Arial",
                        18,
                        "bold"
                    )
                )

        except Exception as e:

            print(
                f"Error in prediction: {e}"
            )

            clean_lbl()

        # ----------------------------------------------------
        # Display camera image
        # ----------------------------------------------------

        im = Image.fromarray(
            frame_resized
        )

        img = ImageTk.PhotoImage(
            image=im
        )

        lblVideo.configure(
            image=img
        )

        lblVideo.image = img

        # ----------------------------------------------------
        # Continue scanning
        # ----------------------------------------------------

        lblVideo.after(
            30,
            scanning
        )


# ============================================================
# MAIN INTERFACE
# ============================================================

def ventana_principal():

    global cap
    global lblVideo
    global model
    global clsName

    global lblimg
    global lblimgtxt
    global class_images
    global text_images
    global lblClase


    # ========================================================
    # CREATE WINDOW
    # ========================================================

    pantalla = Tk()

    pantalla.title(
        "RECICLAJE INTELIGENTE"
    )

    pantalla.geometry(
        "1280x720"
    )


    # ========================================================
    # BACKGROUND IMAGE
    # ========================================================

    try:

        background_path = (
            SETUP_DIR / "Canva.png"
        )

        fondo_img = PhotoImage(
            file=str(background_path)
        )

        background = Label(
            pantalla,
            image=fondo_img
        )

        background.place(
            x=0,
            y=0,
            relwidth=1,
            relheight=1
        )

        # Keep image from being deleted
        background.image = fondo_img

    except Exception as e:

        print(
            f"Error loading background: {e}"
        )


    # ========================================================
    # VIDEO
    # ========================================================

    lblVideo = Label(
        pantalla
    )

    lblVideo.place(
        x=317,
        y=127
    )


    # ========================================================
    # LEFT CATEGORY IMAGE
    # ========================================================

    lblimg = Label(
        pantalla
    )

    lblimg.place(
        x=75,
        y=260
    )


    # ========================================================
    # RIGHT CATEGORY TEXT IMAGE
    # ========================================================

    lblimgtxt = Label(
        pantalla
    )

    lblimgtxt.place(
        x=995,
        y=310
    )


    # ========================================================
    # CLASSIFICATION TEXT
    # ========================================================

    lblClase = Label(
        pantalla,
        text="Select an object",
        font=(
            "Arial",
            18,
            "bold"
        )
    )

    lblClase.place(
        x=318,
        y=90
    )


    # ========================================================
    # LOAD TRAINED MODEL
    # ========================================================

    try:

        if MODEL_PATH is None:

            raise FileNotFoundError(
                "No trained best.pt model was found."
            )

        print(
            "\n========================================"
        )

        print(
            "Loading trained model..."
        )

        print(
            f"Model path:\n{MODEL_PATH}"
        )

        print(
            "========================================"
        )

        model = YOLO(
            str(MODEL_PATH)
        )

        # Get class names directly from model
        clsName = list(
            model.names.values()
        )

        print(
            "Model loaded successfully!"
        )

        print(
            f"Classes: {clsName}"
        )

        print(
            "========================================\n"
        )

    except Exception as e:

        print(
            f"\nERROR loading model:\n{e}"
        )

        print(
            "\nThe program searched for best.pt inside:"
        )

        print(
            MODEL_SEARCH_DIR
        )

        return


    # ========================================================
    # LOAD CATEGORY IMAGES
    # ========================================================

    rutas_imgs = [
        "Hazardous",
        "Non-Recyclable",
        "Organic",
        "Recyclable"
    ]

    class_images = []
    text_images = []


    for clase in rutas_imgs:

        try:

            image_path = (
                SETUP_DIR /
                f"{clase}.png"
            )

            text_path = (
                SETUP_DIR /
                f"{clase}txt.png"
            )

            print(
                f"Loading category images: {clase}"
            )

            img = cv2.imread(
                str(image_path)
            )

            txt = cv2.imread(
                str(text_path)
            )

            if img is None or txt is None:

                raise ValueError(
                    f"Images not found for: {clase}"
                )

            class_images.append(
                img
            )

            text_images.append(
                txt
            )

        except Exception as e:

            print(
                f"Error loading images for {clase}: {e}"
            )

            # Placeholder
            class_images.append(
                np.zeros(
                    (100, 100, 3),
                    dtype=np.uint8
                )
            )

            text_images.append(
                np.zeros(
                    (100, 100, 3),
                    dtype=np.uint8
                )
            )


    # ========================================================
    # MOUSE CONTROLS
    # ========================================================

    # Bind mouse events to camera image
    lblVideo.bind(
        "<Button-1>",
        mouse_down
    )

    lblVideo.bind(
        "<B1-Motion>",
        mouse_move
    )

    lblVideo.bind(
        "<ButtonRelease-1>",
        mouse_up
    )


    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    instructions = Label(
        pantalla,
        text=(
            "Click and drag around the object "
            "you want to classify"
        ),
        font=(
            "Arial",
            12
        ),
        bg="white"
    )

    instructions.place(
        x=440,
        y=690
    )


    # ========================================================
    # START CAMERA
    # ========================================================

    print(
        "Starting camera..."
    )

    cap = cv2.VideoCapture(0)


    if not cap.isOpened():

        print(
            "Could not open camera."
        )

        return


    print(
        "Camera started successfully!"
    )


    # ========================================================
    # START SCANNING
    # ========================================================

    scanning()


    # ========================================================
    # START GUI
    # ========================================================

    pantalla.mainloop()


# ============================================================
# RUN APPLICATION
# ============================================================

ventana_principal()
