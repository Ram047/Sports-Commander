"""
overlay.py — Draws bounding boxes, player labels, ball trails, and a premium scoreboard.
Uses PIL for anti-aliased text rendering and a modern tech aesthetic.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Premium Neon Color Palette (BGR)
COLORS = {
    "neon_blue":   (255, 180, 50),
    "neon_green":  (100, 255, 100),
    "neon_red":    (80, 80, 255),
    "neon_yellow": (50, 255, 255),
    "neon_cyan":   (255, 255, 80),
    "neon_purple": (255, 80, 255),
    "glass_bg":    (20, 20, 25, 160), # RGBA glass effect
    "white":       (240, 240, 240),
}

PLAYER_COLORS = [
    COLORS["neon_blue"],
    COLORS["neon_green"],
    COLORS["neon_red"],
    COLORS["neon_cyan"],
    COLORS["neon_purple"],
    COLORS["neon_yellow"],
]

def _label_color(label: str) -> tuple:
    idx = ord(label[0]) - ord("A")
    return PLAYER_COLORS[idx % len(PLAYER_COLORS)]

def _get_font(size: int):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", size) # Bold Segoe
    except:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            return ImageFont.load_default()

def draw_players(frame: np.ndarray, players: list) -> np.ndarray:
    """Draw refined bounding boxes and player labels."""
    for p in players:
        x1, y1, x2, y2 = p["bbox"]
        label = p["label"]
        color = _label_color(label)

        # Subtle thin box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        
        # Corner accents (gives it a 'target' look)
        length = 15
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, 2)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, 2)
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, 2)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, 2)
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, 2)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, 2)
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, 2)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, 2)

        # Label pill
        text = f"P-{label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return frame

def draw_all_overlays(frame: np.ndarray, sport: str, frame_idx: int, stats: dict, 
                      balls: list, commentary_text: str, commentary_alpha: float) -> np.ndarray:
    """
    Combined PIL-based renderer for superior performance.
    Avoids multiple conversions between OpenCV and PIL.
    """
    # 1. Start with the OpenCV frame (previously drawn with boxes)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Helper to convert BGR color constant to RGB for PIL
    def bgr_to_rgb(c): return (c[2], c[1], c[0])
    
    w, h = img.size

    # --- Draw Scoreboard ---
    bar_h = 50
    draw.rectangle([0, 0, w, bar_h], fill=COLORS["glass_bg"])
    draw.line([0, bar_h, w, bar_h], fill=(255, 255, 255, 40), width=1)
    
    font_main = _get_font(20)
    font_sub = _get_font(14)
    
    sport_text = f"CV ANALYZER: {sport.upper()}"
    draw.text((25, 12), sport_text, font=font_main, fill=bgr_to_rgb(COLORS["neon_yellow"]))
    
    total_seconds = int(frame_idx // 30)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    time_text = f"MATCH TIME: {minutes:02d}:{seconds:02d}"
    draw.text((w // 2 - 60, 12), time_text, font=font_main, fill=(255, 255, 255))
    
    stats_text = " | ".join([f"{k.upper()}: {v}" for k, v in stats.items()])
    st_bbox = draw.textbbox((0, 0), stats_text, font=font_sub)
    draw.text((w - (st_bbox[2]-st_bbox[0]) - 25, 16), stats_text, font=font_sub, fill=(200, 200, 200))

    # --- Draw Ball Outlines ---
    for b in balls:
        x1, y1, x2, y2 = b["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        r = max((x2 - x1), (y2 - y1)) // 2 + 2
        
        c_rgb = bgr_to_rgb(COLORS["neon_yellow"])
        # Outer neon glow (semi-transparent border)
        draw.ellipse([cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4], 
                     outline=(c_rgb + (100,)), width=1)
        # Main neon circle
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], 
                     outline=(c_rgb + (255,)), width=2)

    # --- Draw Commentary ---
    if commentary_alpha > 0 and commentary_text:
        font_comm = _get_font(28)
        bbox = draw.textbbox((0, 0), commentary_text, font=font_comm)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        tx = (w - tw) // 2
        ty = h - th - 70
        padding = 15
        
        draw.rounded_rectangle(
            [tx - padding, ty - padding, tx + tw + padding, ty + th + padding],
            radius=12, fill=(0, 0, 0, int(200 * commentary_alpha))
        )
        draw.rounded_rectangle(
            [tx - padding, ty - padding, tx + tw + padding, ty + th + padding],
            radius=12, outline=(255, 255, 255, int(100 * commentary_alpha)), width=1
        )
        draw.text((tx, ty), commentary_text, font=font_comm, 
                  fill=(255, 255, 255, int(255 * commentary_alpha)))

    # --- Composite and return ---
    img = Image.alpha_composite(img, overlay)
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

# Keep individual functions for backward compatibility or simple use, 
# but they are no longer the primary way to draw the main pipeline frame.
# Empty draw_ball_trail to satisfy user request to remove yellow line.
def draw_ball_trail(frame: np.ndarray, history: list) -> np.ndarray:
    return frame # No-op (removed yellow line)

def draw_players(frame, players):
    """Draw refined bounding boxes and player labels (efficient OpenCV)."""
    for p in players:
        x1, y1, x2, y2 = p["bbox"]
        label = p["label"]
        color = _label_color(label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        length = 15
        cv2.line(frame, (x1, y1), (x1+length, y1), color, 2)
        cv2.line(frame, (x1, y1), (x1, y1+length), color, 2)
        cv2.line(frame, (x2, y1), (x2-length, y1), color, 2)
        cv2.line(frame, (x2, y1), (x2, y1+length), color, 2)
        cv2.line(frame, (x1, y2), (x1+length, y2), color, 2)
        cv2.line(frame, (x1, y2), (x1, y2-length), color, 2)
        cv2.line(frame, (x2, y2), (x2-length, y2), color, 2)
        cv2.line(frame, (x2, y2), (x2, y2-length), color, 2)
        text = f"P-{label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
        cv2.putText(frame, text, (x1+5, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
    return frame
