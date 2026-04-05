"""
techslide.py — Creates a 5-second tech-stack outro slide for a given sport.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


TECH_STACK = [
    ("Language",       "Python 3.11"),
    ("CV Framework",   "OpenCV 4.x"),
    ("Detection",      "YOLOv8 (Ultralytics)"),
    ("Tracking",       "ByteTrack (Supervision)"),
    ("Rendering",      "PIL / Pillow"),
    ("Video I/O",      "OpenCV VideoCapture / VideoWriter"),
    ("Commentary",     "Rule-based NLP Templates"),
    ("Event Logic",    "Geometric Heuristics (sport-specific)"),
]

SPORT_EMOJI = {
    "basketball": "🏀",
    "volleyball": "🏐",
    "football":   "⚽",
}

ACCENT_COLORS = {
    "basketball": (255, 140, 0),
    "volleyball": (0, 200, 255),
    "football":   (50, 220, 100),
}


def _try_font(size):
    for path in [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_slide_frames(sport: str, width: int, height: int,
                      fps: float = 30.0, duration_sec: float = 5.0) -> list:
    """
    Returns a list of numpy BGR frames representing the tech-stack slide.
    """
    n_frames = int(fps * duration_sec)
    accent = ACCENT_COLORS.get(sport.lower(), (255, 200, 0))
    frames = []

    font_title = _try_font(52)
    font_sub   = _try_font(30)
    font_row   = _try_font(24)
    font_small = _try_font(18)

    for i in range(n_frames):
        t = i / n_frames  # 0 → 1

        # ---- Dark gradient background ----
        img = Image.new("RGB", (width, height), (10, 12, 20))
        draw = ImageDraw.Draw(img)

        # Subtle radial glow
        for r in range(min(width, height) // 2, 0, -20):
            alpha = int(12 * (1 - r / (min(width, height) // 2)))
            draw.ellipse(
                [width // 2 - r, height // 2 - r, width // 2 + r, height // 2 + r],
                fill=(*accent, alpha),
            )

        # ---- Title ----
        title = f"Tech Stack — {sport.capitalize()} CV Analyzer"
        tb = draw.textbbox((0, 0), title, font=font_title)
        tx = (width - (tb[2] - tb[0])) // 2
        draw.text((tx + 2, 52), title, font=font_title, fill=(30, 30, 30))
        draw.text((tx, 50), title, font=font_title, fill=accent)

        # ---- Divider ----
        draw.line([(60, 120), (width - 60, 120)], fill=accent, width=2)

        # ---- Table rows ----
        row_y = 145
        row_h = 46
        for j, (layer, tech) in enumerate(TECH_STACK):
            fade = min(1.0, max(0.0, t * len(TECH_STACK) - j))
            if fade <= 0:
                continue
            gray = int(40 + 30 * (j % 2))
            draw.rounded_rectangle(
                [60, row_y, width - 60, row_y + row_h - 4],
                radius=6,
                fill=(gray, gray + 5, gray + 15),
            )
            draw.text((80, row_y + 10), layer, font=font_row,
                      fill=(*accent, int(255 * fade)))
            draw.text((340, row_y + 10), tech, font=font_row,
                      fill=(220, 220, 220))
            row_y += row_h

        # ---- Footer ----
        footer = "CSE AI/ML — Computer Vision Assignment"
        fb = draw.textbbox((0, 0), footer, font=font_small)
        fx = (width - (fb[2] - fb[0])) // 2
        draw.text((fx, height - 40), footer, font=font_small, fill=(120, 120, 140))

        frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        frames.append(frame_bgr)

    return frames
