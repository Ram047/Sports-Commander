"""
test_run.py — Quick 5-second pipeline test on volleyball video.
Updated to test new blinking-ball fix and enhanced UI.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from src.detector import Detector
from src.tracker import Tracker
from src.event_engine import get_engine
from src import commentary

cap = cv2.VideoCapture("input/20260402_175953.mp4")
fps   = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {width}x{height} @ {fps:.0f}fps")

infer_w = 1280
infer_h = int(height * infer_w / width)
inv = width / infer_w

detector = Detector(conf=0.25)
tracker  = Tracker()
engine   = get_engine("volleyball")

players_cache = []
events_found = 0

for i in range(150):
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (infer_w, infer_h))
    
    # Pass cached players for proximity-based filtering
    dets = detector.detect(small, players=players_cache)

    for lst in (dets["persons"], dets["balls"]):
        for d in lst:
            d["bbox"] = [int(v * inv) for v in d["bbox"]]

    players = tracker.update(dets["persons"], frame.shape)
    players_cache = players # Cache for next frame
    
    events  = engine.analyze(
        frame_idx=i, players=players, balls=dets["balls"],
        frame_w=width, frame_h=height
    )

    for ev in events:
        text = commentary.generate(ev)
        print(f"  [frame {i:03d}] {text}")
        events_found += 1

    if i % 30 == 0:
        n_balls = len(dets["balls"])
        n_players = len(players)
        print(f"  Frame {i:03d}: {n_players} players, {n_balls} balls detected")

cap.release()
print(f"\nTest complete — {events_found} events in 150 frames")
