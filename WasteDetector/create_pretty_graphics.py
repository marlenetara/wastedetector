import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SETUP_DIR = Path("Recibot/setUp")
SETUP_DIR.mkdir(parents=True, exist_ok=True)

# Font selection
FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_REGULAR = "C:/Windows/Fonts/arial.ttf"

def get_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except:
        return ImageFont.load_default()

CATEGORIES = {
    "yellow_bin": {
        "title": "YELLOW BIN",
        "subtitle": "Gelber Sack",
        "detail": "Plastics, Cans & Tubs",
        "primary": (255, 193, 7),       # Amber / Yellow
        "dark": (218, 140, 0),
        "light": (255, 248, 225),
        "symbol": "recycle"
    },
    "organic": {
        "title": "ORGANIC WASTE",
        "subtitle": "Biomüll (Brown Bin)",
        "detail": "Food, Peels & Garden",
        "primary": (76, 175, 80),       # Vibrant Green
        "dark": (46, 125, 50),
        "light": (232, 245, 233),
        "symbol": "leaf"
    },
    "paper": {
        "title": "PAPER & BOARD",
        "subtitle": "Altpapier (Blue Bin)",
        "detail": "Cardboard, Paper & Boxes",
        "primary": (33, 150, 243),      # Blue
        "dark": (21, 101, 192),
        "light": (227, 242, 253),
        "symbol": "paper"
    },
    "glass": {
        "title": "GLASS BOTTLES",
        "subtitle": "Altglas (Containers)",
        "detail": "Jars & Bottles by Color",
        "primary": (0, 150, 136),       # Teal / Glass
        "dark": (0, 105, 92),
        "light": (224, 242, 241),
        "symbol": "glass"
    },
    "residual": {
        "title": "RESIDUAL WASTE",
        "subtitle": "Restmüll (Black Bin)",
        "detail": "General Non-Recyclable",
        "primary": (97, 97, 97),        # Dark Charcoal
        "dark": (46, 46, 46),
        "light": (238, 238, 238),
        "symbol": "trash"
    },
    "deposit": {
        "title": "DEPOSIT / PFAND",
        "subtitle": "Supermarket Return",
        "detail": "25¢ / 15¢ Refund & Batteries",
        "primary": (244, 67, 54),       # Red
        "dark": (198, 40, 40),
        "light": (255, 235, 238),
        "symbol": "deposit"
    }
}

def draw_card(draw, box, fill_color, border_color, radius=18):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill_color, outline=border_color, width=3)

def create_icon_image(key, data):
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Outer card
    draw_card(draw, (8, 8, 192, 192), data["light"], data["primary"], radius=24)

    # Center circle badge
    draw.ellipse([34, 34, 166, 166], fill=data["primary"], outline=data["dark"], width=3)

    # Inner decorative rings
    draw.ellipse([42, 42, 158, 158], outline=(255, 255, 255, 140), width=2)

    # Center Icon Graphics
    symbol = data["symbol"]
    if symbol == "recycle":
        # Draw clean recycling triangle arrows / symbols
        pts = [(100, 60), (135, 125), (65, 125)]
        draw.polygon(pts, fill=None, outline=(255, 255, 255), width=6)
        draw.ellipse([90, 85, 110, 105], fill=(255, 255, 255))
    elif symbol == "leaf":
        # Draw organic leaf shape
        draw.pieslice([65, 60, 135, 140], 45, 225, fill=(255, 255, 255))
        draw.line([75, 130, 125, 70], fill=data["dark"], width=3)
    elif symbol == "paper":
        # Draw folded paper / document icon
        draw.rectangle([75, 60, 125, 130], fill=(255, 255, 255), outline=data["dark"], width=2)
        draw.line([85, 80, 115, 80], fill=data["dark"], width=3)
        draw.line([85, 95, 115, 95], fill=data["dark"], width=3)
        draw.line([85, 110, 115, 110], fill=data["dark"], width=3)
    elif symbol == "glass":
        # Draw bottle silhouette
        draw.rectangle([94, 55, 106, 75], fill=(255, 255, 255))
        draw.rounded_rectangle([82, 75, 118, 140], radius=8, fill=(255, 255, 255))
        draw.rectangle([90, 95, 110, 120], fill=data["primary"])
    elif symbol == "trash":
        # Draw wheelie bin silhouette
        draw.rectangle([80, 75, 120, 135], fill=(255, 255, 255))
        draw.rectangle([72, 68, 128, 75], fill=(255, 255, 255))
        draw.line([90, 85, 90, 125], fill=data["dark"], width=2)
        draw.line([100, 85, 100, 125], fill=data["dark"], width=2)
        draw.line([110, 85, 110, 125], fill=data["dark"], width=2)
    elif symbol == "deposit":
        # Draw Euro / Pfand refund symbol
        font_eur = get_font(FONT_BOLD, 46)
        draw.text((100, 96), "€", fill=(255, 255, 255), font=font_eur, anchor="mm")

    # Bottom small tag
    font_tag = get_font(FONT_BOLD, 12)
    draw.rounded_rectangle([45, 160, 155, 185], radius=10, fill=data["dark"])
    draw.text((100, 172), data["title"].split()[0], fill=(255, 255, 255), font=font_tag, anchor="mm")

    return img

def create_text_badge(key, data):
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Outer card
    draw_card(draw, (8, 8, 192, 192), (255, 255, 255), data["primary"], radius=20)

    # Top accent color header bar
    draw.rounded_rectangle([8, 8, 192, 48], radius=14, fill=data["primary"])
    draw.rectangle([8, 30, 192, 48], fill=data["primary"])

    font_header = get_font(FONT_BOLD, 13)
    draw.text((100, 28), data["title"], fill=(255, 255, 255), font=font_header, anchor="mm")

    # German Bin Subtitle
    font_sub = get_font(FONT_BOLD, 15)
    draw.text((100, 80), data["subtitle"].split("(")[0].strip(), fill=data["dark"], font=font_sub, anchor="mm")

    if "(" in data["subtitle"]:
        font_sub2 = get_font(FONT_REGULAR, 12)
        extra = "(" + data["subtitle"].split("(")[1]
        draw.text((100, 102), extra, fill=(100, 100, 100), font=font_sub2, anchor="mm")

    # Divider line
    draw.line([30, 124, 170, 124], fill=(220, 220, 220), width=2)

    # Bottom English Detail pill
    draw.rounded_rectangle([18, 140, 182, 180], radius=12, fill=data["light"], outline=data["primary"], width=1)
    font_det = get_font(FONT_BOLD, 11)
    draw.text((100, 160), data["detail"], fill=data["dark"], font=font_det, anchor="mm")

    return img

# Generate all 12 graphics
for key, data in CATEGORIES.items():
    icon_img = create_icon_image(key, data)
    icon_path = SETUP_DIR / f"{key}.png"
    icon_img.save(str(icon_path))
    print(f"Created {icon_path}")

    text_img = create_text_badge(key, data)
    text_path = SETUP_DIR / f"{key}txt.png"
    text_img.save(str(text_path))
    print(f"Created {text_path}")

# Remove old Spanish files
old_files = [
    "metal.png", "metaltxt.png",
    "organico.png", "organicotxt.png",
    "papel_y_carton.png", "papel_y_cartontxt.png",
    "plastico.png", "plasticotxt.png",
    "vidrio.png", "vidriotxt.png"
]

for old in old_files:
    target = SETUP_DIR / old
    if target.exists():
        try:
            target.unlink()
            print(f"Removed old Spanish graphic: {old}")
        except Exception as e:
            print(f"Notice: {e}")

print("\nALL MODERN ENGLISH GRAPHICS CREATED AND OLD SPANISH FILES REMOVED!")

