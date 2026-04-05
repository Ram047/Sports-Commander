"""
pipeline.py — Main orchestrator for the Sports CV Commentary System.

Usage:
  python src/pipeline.py --input input/video.mp4 --output output/out.mp4 --sport basketball

Supports: basketball | volleyball | football

Note: 4K input is automatically downscaled to 1080p for fast inference
      then overlays are upscaled back to original resolution for output.
"""

import argparse
import sys
import os
import cv2
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import Detector
from src.tracker import Tracker
from src.event_engine import get_engine
from src import commentary
from src.overlay import draw_players, draw_all_overlays
from src.techslide import make_slide_frames

# ── Commentary display state ──────────────────────────────────────────────────
FADE_IN_FRAMES  = 9   # ~0.3s at 30fps
HOLD_FRAMES     = 75  # ~2.5s
FADE_OUT_FRAMES = 15  # ~0.5s
TOTAL_FRAMES    = FADE_IN_FRAMES + HOLD_FRAMES + FADE_OUT_FRAMES


class CommentaryDisplay:
    def __init__(self):
        self._text: str = ""
        self._counter: int = 0

    def push(self, text: str):
        """Queue new commentary (replaces current if new one arrives)."""
        if text:
            self._text = text
            self._counter = 0

    def get_state(self) -> tuple[str, float]:
        """Returns (text, alpha) for current frame."""
        if not self._text or self._counter >= TOTAL_FRAMES:
            return "", 0.0

        c = self._counter
        if c < FADE_IN_FRAMES:
            alpha = c / FADE_IN_FRAMES
        elif c < FADE_IN_FRAMES + HOLD_FRAMES:
            alpha = 1.0
        else:
            alpha = 1.0 - (c - FADE_IN_FRAMES - HOLD_FRAMES) / FADE_OUT_FRAMES

        self._counter += 1
        return self._text, max(0.0, min(1.0, alpha))


# ─────────────────────────────────────────────────────────────────────────────

INFER_MAX_WIDTH = 1280   # Downscale 4K→1280p for fast YOLO inference


def process_video(input_path: str, output_path: str, sport: str,
                  conf: float = 0.35, device: str = "cpu") -> None:
    print(f"\n{'='*60}")
    print(f"  Sport  : {sport.upper()}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"{'='*60}\n")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Resolution: {width}x{height} @ {fps:.1f} fps  |  Total frames: {total}")

    # Scale factor for inference (downscale 4K to 1280p for speed)
    scale = min(1.0, INFER_MAX_WIDTH / width)
    infer_w = int(width * scale)
    infer_h = int(height * scale)
    if scale < 1.0:
        print(f"  Inference scale: {scale:.2f}x  ({infer_w}x{infer_h})  — 4K downscaled for speed")
    print()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = Detector(conf=conf)
    tracker  = Tracker()
    engine   = get_engine(sport)
    disp     = CommentaryDisplay()

    # --- Tracking state ---
    ball_history = []
    static_blacklist = {} # (x, y) -> frames_remaining
    match_stats = {"events": 0}
    players_cache = [] # To pass to detector on next frame for proximity filtering

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update static blacklist
        static_blacklist = {k: v - 1 for k, v in static_blacklist.items() if v > 1}

        # ── Resize for fast inference ────────────────────────────────────────
        if scale < 1.0:
            # INTER_NEAREST is much faster for downscaling during inference
            small = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_NEAREST)
        else:
            small = frame

        # ── Detect ──────────────────────────────────────────────────────────
        # Fix: Downscale player coordinates for detection proximity filtering
        players_inf = []
        for p in players_cache:
            p_inf = {"bbox": [int(v * scale) for v in p["bbox"]]}
            players_inf.append(p_inf)

        detections = detector.detect(small, players=players_inf, sport=sport)

        # Scale detections back to full resolution
        if scale < 1.0:
            inv = 1.0 / scale
            for lst in (detections["persons"], detections["balls"]):
                for d in lst:
                    d["bbox"] = [int(v * inv) for v in d["bbox"]]

        # ── Track — use full-res frame shape ────────────────────────────────
        players = tracker.update(detections["persons"], frame.shape)
        players_cache = players # Cache for next frame's ball filtering
        
        # ── Ball Filtering (New Static Noise Zone logic) ────────────────────
        balls = detections["balls"]
        valid_balls = []

        for b_obj in balls:
            b = b_obj["bbox"]
            cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
            
            # Check against blacklist
            is_blacklisted = False
            for (bx, by) in static_blacklist.keys():
                if abs(cx - bx) < 25 and abs(cy - by) < 25: # Increased radius
                    is_blacklisted = True
                    break
            
            if is_blacklisted:
                continue

            # Motion check: if ball is static for >3 frames, blacklist it (more aggressive)
            is_static = False
            static_count = 0
            for h_pos in reversed(ball_history[-10:]):
                if h_pos and abs(h_pos[0]-cx) < 3 and abs(h_pos[1]-cy) < 3:
                    static_count += 1
            
            if static_count > 3:
                static_blacklist[(cx, cy)] = 300 # Ignore this spot for 300 frames (~10s)
                continue
            
            valid_balls.append(b_obj)
        
        balls = valid_balls
        if balls:
            b = balls[0]["bbox"]
            ball_history.append(((b[0] + b[2]) // 2, (b[1] + b[3]) // 2))
        else:
            ball_history.append(None)
            
        if len(ball_history) > 15: ball_history.pop(0)

        # ── Event Analysis (Pass tracker for universal move detection) ──────
        kwargs = {
            "frame_idx": frame_idx,
            "players": players,
            "balls": balls,
            "tracker": tracker
        }
        if sport in ["volleyball", "football"]:
            kwargs.update({"frame_w": width, "frame_h": int(height)})

        events = engine.analyze(**kwargs)

        # ── Commentary ──────────────────────────────────────────────────────
        for ev in events:
            match_stats["events"] += 1
            text = commentary.generate(ev)
            if text:
                print(f"  [{frame_idx:05d}] {text}")
                disp.push(text)

        # ── Overlay (Consolidated for efficiency) ───────────────────────────
        frame = draw_players(frame, players)
        
        # Get commentary state
        txt, alpha = disp.get_state()
        
        # Combined PIL-pass for balls, scoreboard, and commentary
        frame = draw_all_overlays(
            frame, 
            sport=sport, 
            frame_idx=frame_idx, 
            stats=match_stats,
            balls=balls, 
            commentary_text=txt, 
            commentary_alpha=alpha
        )

        writer.write(frame)
        frame_idx += 1

        # Progress
        if frame_idx % 150 == 0:
            elapsed = time.time() - t0
            pct = (frame_idx / total * 100) if total else 0
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  Progress: {pct:.1f}%  |  {fps_proc:.1f} fps processing")

    cap.release()

    # ── Tech-stack slide ────────────────────────────────────────────────────
    print("\n  Appending tech-stack slide …")
    slide_frames = make_slide_frames(sport, width, height, fps=fps, duration_sec=5.0)
    for sf in slide_frames:
        writer.write(sf)

    writer.release()
    elapsed = time.time() - t0
    print(f"\n  [DONE] Output saved to: {output_path}")
    print(f"  Total time: {elapsed:.1f}s  |  Frames processed: {frame_idx}\n")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sports CV Commentary System — generates annotated output video"
    )
    parser.add_argument("--input",  required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output video file")
    parser.add_argument("--sport",  required=True,
                        choices=["basketball", "volleyball", "football"],
                        help="Sport type")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO confidence threshold")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    process_video(args.input, args.output, args.sport, conf=args.conf)


if __name__ == "__main__":
    main()
