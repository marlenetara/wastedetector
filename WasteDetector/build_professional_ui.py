import os
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SETUP_DIR = Path("Recibot/setUp")
SETUP_DIR.mkdir(parents=True, exist_ok=True)

FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_REGULAR = "C:/Windows/Fonts/segoeui.ttf"

def get_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", size)
        except:
            return ImageFont.load_default()

# ============================================================
# 1. BUILD LUXURY DASHBOARD BACKGROUND (Canva.png)
# ============================================================
def build_canvas():
    width, height = 1280, 740
    img = Image.new("RGB", (width, height), (11, 15, 25))
    draw = ImageDraw.Draw(img)

    # Smooth dark vertical gradient
    for y in range(height):
        factor = y / height
        r = int(11 + factor * 8)
        g = int(15 + factor * 14)
        b = int(25 + factor * 25)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Subtle ambient glow in top center
    glow_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_overlay)
    for rad in range(350, 0, -25):
        alpha = int((1 - rad / 350) * 18)
        glow_draw.ellipse([640 - rad, 100 - rad // 2, 640 + rad, 100 + rad // 2], fill=(30, 58, 138, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), glow_overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Helper for frosted glass card
    def draw_glass_card(box, radius=18, border_color=(40, 53, 80), fill_color=(17, 24, 39)):
        draw.rounded_rectangle(box, radius=radius, fill=fill_color, outline=border_color, width=2)
        # Inner soft highlight line
        x0, y0, x1, y1 = box
        draw.line([(x0 + radius, y0 + 1), (x1 - radius, y0 + 1)], fill=(70, 85, 120), width=1)

    # --------------------------------------------------------
    # TOP BRANDING HEADER
    # --------------------------------------------------------
    font_brand = get_font(FONT_BOLD, 22)
    font_sub = get_font(FONT_REGULAR, 12)
    font_live = get_font(FONT_BOLD, 11)

    # Brand Title
    draw.text((42, 32), "WASTEDETECT", fill=(248, 250, 252), font=font_brand)
    draw.text((215, 33), "AI", fill=(56, 189, 248), font=font_brand)
    draw.text((42, 62), "GERMAN WASTE SEPARATION & VISION SYSTEM", fill=(148, 163, 184), font=font_sub)

    # Right Live Status Pill
    draw.rounded_rectangle([1060, 35, 1235, 68], radius=16, fill=(15, 23, 42), outline=(34, 197, 94), width=1)
    draw.ellipse([1075, 47, 1087, 59], fill=(34, 197, 94))
    draw.text((1097, 43), "AI CAMERA LIVE", fill=(248, 250, 252), font=font_live)

    # --------------------------------------------------------
    # LEFT PANEL: ACTIVE DISPOSAL STREAM
    # --------------------------------------------------------
    left_card = (35, 95, 285, 705)
    draw_glass_card(left_card, radius=20, border_color=(38, 50, 75), fill_color=(15, 23, 42))

    # Left Panel Header
    draw.rounded_rectangle([45, 105, 275, 142], radius=12, fill=(30, 41, 59))
    font_panel_h = get_font(FONT_BOLD, 13)
    draw.text((160, 123), "DISPOSAL STREAM", fill=(226, 232, 240), font=font_panel_h, anchor="mm")

    # Left slot guide box for the icon (centered at x: 160, y: 360, size: 200x200)
    # lblimg is placed at x=60, y=260
    draw.rounded_rectangle([55, 255, 265, 465], radius=16, fill=(11, 15, 25), outline=(30, 41, 59), width=1)

    font_slot = get_font(FONT_REGULAR, 11)
    draw.text((160, 520), "OFFICIAL GERMAN BIN", fill=(100, 116, 139), font=font_slot, anchor="mm")
    draw.text((160, 540), "MÜLLTRENNUNG CATEGORY", fill=(71, 85, 105), font=font_slot, anchor="mm")

    # --------------------------------------------------------
    # RIGHT PANEL: GUIDELINES & RECYCLING RULES
    # --------------------------------------------------------
    right_card = (995, 95, 1245, 705)
    draw_glass_card(right_card, radius=20, border_color=(38, 50, 75), fill_color=(15, 23, 42))

    # Right Panel Header
    draw.rounded_rectangle([1005, 105, 1235, 142], radius=12, fill=(30, 41, 59))
    draw.text((1120, 123), "SORTING RULES", fill=(226, 232, 240), font=font_panel_h, anchor="mm")

    # Right slot guide box for the text badge (placed at x: 1020, y: 260)
    draw.rounded_rectangle([1015, 255, 1225, 465], radius=16, fill=(11, 15, 25), outline=(30, 41, 59), width=1)

    draw.text((1120, 520), "REGULATORY GUIDANCE", fill=(100, 116, 139), font=font_slot, anchor="mm")
    draw.text((1120, 540), "VERPACKUNGSGESETZ COMPLIANT", fill=(71, 85, 105), font=font_slot, anchor="mm")

    # --------------------------------------------------------
    # CENTER CAMERA FRAME
    # --------------------------------------------------------
    cam_card = (305, 95, 975, 705)
    draw_glass_card(cam_card, radius=20, border_color=(45, 60, 90), fill_color=(13, 18, 30))

    # Camera Viewport Boundary (x: 320, y: 115, w: 640, h: 480)
    vx0, vy0, vx1, vy1 = 320, 115, 960, 595
    draw.rectangle([vx0, vy0, vx1, vy1], outline=(30, 41, 59), width=2)

    # High-Tech Corner HUD Reticles
    reticle_len = 25
    reticle_color = (56, 189, 248)
    # Top-Left
    draw.line([(vx0 - 4, vy0 - 4), (vx0 + reticle_len, vy0 - 4)], fill=reticle_color, width=3)
    draw.line([(vx0 - 4, vy0 - 4), (vx0 - 4, vy0 + reticle_len)], fill=reticle_color, width=3)
    # Top-Right
    draw.line([(vx1 + 4, vy0 - 4), (vx1 - reticle_len, vy0 - 4)], fill=reticle_color, width=3)
    draw.line([(vx1 + 4, vy0 - 4), (vx1 + 4, vy0 + reticle_len)], fill=reticle_color, width=3)
    # Bottom-Left
    draw.line([(vx0 - 4, vy1 + 4), (vx0 + reticle_len, vy1 + 4)], fill=reticle_color, width=3)
    draw.line([(vx0 - 4, vy1 + 4), (vx0 - 4, vy1 - reticle_len)], fill=reticle_color, width=3)
    # Bottom-Right
    draw.line([(vx1 + 4, vy1 + 4), (vx1 - reticle_len, vy1 + 4)], fill=reticle_color, width=3)
    draw.line([(vx1 + 4, vy1 + 4), (vx1 + 4, vy1 - reticle_len)], fill=reticle_color, width=3)

    # Bottom Telemetry Bar below camera
    draw.rounded_rectangle([320, 615, 960, 685], radius=12, fill=(18, 26, 43), outline=(35, 48, 75), width=1)

    font_tele = get_font(FONT_BOLD, 11)
    font_tele_val = get_font(FONT_REGULAR, 11)

    draw.text((340, 635), "TARGET TRACKING:", fill=(148, 163, 184), font=font_tele)
    draw.text((460, 635), "ACTIVE (HELD OBJECT ONLY)", fill=(34, 197, 94), font=font_tele)

    draw.text((340, 660), "FILTER APPLIED:", fill=(148, 163, 184), font=font_tele)
    draw.text((460, 660), "PEOPLE, HANDS & ROOM BACKGROUND EXCLUDED", fill=(56, 189, 248), font=font_tele_val)

    draw.text((760, 647), "CONFIDENCE THRESHOLD: 20%", fill=(100, 116, 139), font=font_tele_val)

    # Save Canva.png
    canva_path = SETUP_DIR / "Canva.png"
    img.save(str(canva_path), quality=95)
    print(f"[OK] Created luxury dashboard background: {canva_path}")

# ============================================================
# 2. BUILD 3D-STYLED SOPHISTICATED CATEGORY BADGES
# ============================================================
CATEGORIES_DATA = {
    "yellow_bin": {
        "title": "YELLOW BIN",
        "subtitle": "Gelber Sack / Tonne",
        "detail": "Plastics, Cans & Tubs",
        "accent": (245, 158, 11),       # Vibrant Amber
        "gradient_top": (251, 191, 36),
        "gradient_bot": (180, 83, 9),
        "symbol_type": "recycle"
    },
    "organic": {
        "title": "ORGANIC WASTE",
        "subtitle": "Biomüll (Brown Bin)",
        "detail": "Food, Peels & Garden",
        "accent": (16, 185, 129),       # Emerald
        "gradient_top": (52, 211, 153),
        "gradient_bot": (4, 120, 87),
        "symbol_type": "leaf"
    },
    "paper": {
        "title": "PAPER & BOARD",
        "subtitle": "Altpapier (Blue Bin)",
        "detail": "Boxes, Paper & Cardboard",
        "accent": (59, 130, 246),       # Royal Blue
        "gradient_top": (96, 165, 250),
        "gradient_bot": (29, 78, 216),
        "symbol_type": "paper"
    },
    "glass": {
        "title": "GLASS BOTTLES",
        "subtitle": "Altglas (Containers)",
        "detail": "Bottles & Jars by Color",
        "accent": (20, 184, 166),       # Teal Glass
        "gradient_top": (45, 212, 191),
        "gradient_bot": (15, 118, 110),
        "symbol_type": "glass"
    },
    "residual": {
        "title": "RESIDUAL WASTE",
        "subtitle": "Restmüll (Black Bin)",
        "detail": "Non-Recyclable Trash",
        "accent": (148, 163, 184),      # Slate Platinum
        "gradient_top": (203, 213, 225),
        "gradient_bot": (71, 85, 105),
        "symbol_type": "trash"
    },
    "deposit": {
        "title": "DEPOSIT / PFAND",
        "subtitle": "Supermarket Return",
        "detail": "25¢ / 15¢ Refund & Batteries",
        "accent": (239, 68, 68),        # Crimson
        "gradient_top": (248, 113, 113),
        "gradient_bot": (185, 28, 28),
        "symbol_type": "deposit"
    }
}

def create_sophisticated_icon(key, data):
    size = 200
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Outer Glassmorphic Card (Matching Canva dark background #151F33)
    card_box = [6, 6, 194, 194]
    draw.rounded_rectangle(card_box, radius=24, fill=(18, 26, 44, 240), outline=(40, 56, 85), width=2)

    # Subtle top inner highlight
    draw.line([(24, 7), (176, 7)], fill=(80, 100, 140, 150), width=1)

    # 3D Shield Base (Radial / Spherical feel)
    cx, cy, radius = 100, 95, 58
    # Outer glow ring
    accent = data["accent"]
    draw.ellipse([cx - radius - 5, cy - radius - 5, cx + radius + 5, cy + radius + 5],
                 outline=(accent[0], accent[1], accent[2], 90), width=3)

    # Multi-layer 3D Gradient Circle
    top_c = data["gradient_top"]
    bot_c = data["gradient_bot"]
    for r in range(radius, 0, -1):
        ratio = (radius - r) / radius
        cr = int(bot_c[0] + (top_c[0] - bot_c[0]) * (1 - ratio * 0.7))
        cg = int(bot_c[1] + (top_c[1] - bot_c[1]) * (1 - ratio * 0.7))
        cb = int(bot_c[2] + (top_c[2] - bot_c[2]) * (1 - ratio * 0.7))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(cr, cg, cb))

    # Specular Gloss Highlight (Curved glass reflection at top)
    draw.pieslice([cx - radius + 8, cy - radius + 6, cx + radius - 8, cy + radius - 20], 200, 340,
                  fill=(255, 255, 255, 80))

    # Crisp Vector Icon
    st = data["symbol_type"]
    if st == "recycle":
        # Professional 3-arrow Mobius Loop
        pts = [(100, 68), (128, 118), (72, 118)]
        draw.polygon(pts, fill=None, outline=(255, 255, 255), width=6)
        draw.ellipse([92, 88, 108, 104], fill=(255, 255, 255))
    elif st == "leaf":
        # 3D Leaf
        draw.pieslice([72, 65, 128, 128], 45, 225, fill=(255, 255, 255))
        draw.line([82, 122, 118, 72], fill=data["gradient_bot"], width=3)
    elif st == "paper":
        # Clean folded cardboard / paper sheets
        draw.rounded_rectangle([78, 68, 122, 122], radius=4, fill=(255, 255, 255), outline=data["gradient_bot"], width=2)
        draw.line([88, 85, 112, 85], fill=data["gradient_bot"], width=3)
        draw.line([88, 98, 112, 98], fill=data["gradient_bot"], width=3)
        draw.line([88, 110, 104, 110], fill=data["gradient_bot"], width=3)
    elif st == "glass":
        # Sleek Glass Bottle
        draw.rounded_rectangle([95, 60, 105, 75], radius=2, fill=(255, 255, 255))
        draw.rounded_rectangle([84, 75, 116, 128], radius=8, fill=(255, 255, 255))
        draw.rectangle([90, 90, 110, 115], fill=data["gradient_bot"])
    elif st == "trash":
        # Modern Wheelie Bin
        draw.rounded_rectangle([76, 68, 124, 76], radius=3, fill=(255, 255, 255))
        draw.rounded_rectangle([82, 77, 118, 126], radius=4, fill=(255, 255, 255))
        draw.line([92, 86, 92, 116], fill=data["gradient_bot"], width=2)
        draw.line([100, 86, 100, 116], fill=data["gradient_bot"], width=2)
        draw.line([108, 86, 108, 116], fill=data["gradient_bot"], width=2)
    elif st == "deposit":
        # Glowing 25c / Euro refund symbol
        font_c = get_font(FONT_BOLD, 42)
        draw.text((100, 93), "€", fill=(255, 255, 255), font=font_c, anchor="mm")
        # Surrounding circular arrows
        draw.arc([68, 63, 132, 127], 30, 310, fill=(255, 255, 255), width=3)

    # Bottom Pill Label
    font_lbl = get_font(FONT_BOLD, 11)
    draw.rounded_rectangle([35, 160, 165, 185], radius=8, fill=(26, 36, 60), outline=accent, width=1)
    draw.text((100, 172), data["title"].split()[0], fill=(248, 250, 252), font=font_lbl, anchor="mm")

    return img

def create_sophisticated_text_badge(key, data):
    size = 200
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    accent = data["accent"]

    # Outer Glassmorphic Card
    draw.rounded_rectangle([6, 6, 194, 194], radius=24, fill=(18, 26, 44, 240), outline=(40, 56, 85), width=2)
    draw.line([(24, 7), (176, 7)], fill=(80, 100, 140, 150), width=1)

    # Top Accent Color Header
    draw.rounded_rectangle([6, 6, 194, 48], radius=20, fill=accent)
    draw.rectangle([6, 25, 194, 48], fill=accent)

    font_top = get_font(FONT_BOLD, 12)
    draw.text((100, 26), data["title"], fill=(255, 255, 255), font=font_top, anchor="mm")

    # German Classification Stream
    font_de = get_font(FONT_BOLD, 15)
    clean_sub = data["subtitle"].split("(")[0].strip()
    draw.text((100, 80), clean_sub, fill=(248, 250, 252), font=font_de, anchor="mm")

    if "(" in data["subtitle"]:
        font_par = get_font(FONT_REGULAR, 11)
        sub_par = "(" + data["subtitle"].split("(")[1]
        draw.text((100, 102), sub_par, fill=(148, 163, 184), font=font_par, anchor="mm")

    # Divider
    draw.line([(30, 124), (170, 124)], fill=(40, 56, 85), width=1)

    # Details Pill
    draw.rounded_rectangle([16, 138, 184, 182], radius=12, fill=(11, 15, 25), outline=accent, width=1)
    font_dt = get_font(FONT_BOLD, 10)
    draw.text((100, 160), data["detail"], fill=accent, font=font_dt, anchor="mm")

    return img

# Execute Generation
build_canvas()

for key, data in CATEGORIES_DATA.items():
    icon_img = create_sophisticated_icon(key, data)
    icon_path = SETUP_DIR / f"{key}.png"
    icon_img.save(str(icon_path))

    text_img = create_sophisticated_text_badge(key, data)
    text_path = SETUP_DIR / f"{key}txt.png"
    text_img.save(str(text_path))

print("\n[SUCCESS] All high-end luxury graphics generated successfully!")

