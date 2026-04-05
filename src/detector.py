"""
detector.py — YOLOv8 person + sports ball detection.
Includes HSV-based color validation for basketballs to prevent blinking.
"""

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class IDs
PERSON_CLASS = 0
SPORTS_BALL_CLASS = 32


class Detector:
    def __init__(self, model_path: str = "yolov8s.pt", conf: float = 0.25):
        self.model = YOLO(model_path)
        # Use a low base confidence globally to capture all potential balls
        self.conf = 0.25

    def detect(self, frame: np.ndarray, players: list = None, sport: str = None) -> dict:
        """
        Returns dict with keys:
          'persons': list of {'bbox': [x1,y1,x2,y2], 'conf': float}
          'balls':   list of {'bbox': [x1,y1,x2,y2], 'conf': float}
        """
        results = self.model(frame, conf=self.conf, verbose=False)[0]

        persons, balls = [], []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            entry = {"bbox": [x1, y1, x2, y2], "conf": conf}

            # Strategy: Be strict for persons, relaxed for balls
            if cls == PERSON_CLASS:
                if conf >= 0.45: # Filter out unsure players
                    persons.append(entry)
            elif cls == SPORTS_BALL_CLASS:
                # Always keep YOLO balls if they pass 0.25
                balls.append(entry)

        # Fallback: Hough-circle ball detection for volleyball & basketball
        # We re-enabled basketball and added an Orange Color Filter below.
        if not balls and sport in ["volleyball", "basketball"]:
            balls = self._hough_ball(frame, players, sport)

        return {"persons": persons, "balls": balls}

    def _hough_ball(self, frame: np.ndarray, players: list = None, sport: str = None) -> list:
        """
        Detect circular ball using HoughCircles with Color Filtering.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Sport thresholds
        min_intensity = 165
        param2 = 50
        min_dist_divisor = 10
        if sport == "basketball":
            min_intensity = 80  # Lowered for dark orange
            param2 = 30         # Very sensitive
            min_dist_divisor = 15

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=w // min_dist_divisor,
            param1=60, param2=param2,
            minRadius=10, maxRadius=45
        )

        balls = []
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            valid_circles = []
            for (cx, cy, r) in circles:
                # 1. Intensity Check
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (cx, cy), r, 255, -1)
                mean_val = cv2.mean(gray, mask=mask)[0]
                
                # 2. Color Calibration (Anti-Blinking)
                if sport == "basketball":
                    # Orange/Brown range in HSV
                    lower_orange = np.array([0, 70, 50])
                    upper_orange = np.array([25, 255, 255])
                    
                    roi_hsv = hsv[max(0, cy-r):min(h, cy+r), max(0, cx-r):min(w, cx+r)]
                    if roi_hsv.size == 0: continue
                    
                    color_mask = cv2.inRange(roi_hsv, lower_orange, upper_orange)
                    orange_ratio = np.sum(color_mask > 0) / (roi_hsv.shape[0] * roi_hsv.shape[1])
                    
                    # Accept if it's intensely orange OR if it passes basic intensity
                    if orange_ratio > 0.4:
                        valid_circles.append((cx, cy, r))
                else:
                    # Volleyball (just intensity)
                    if mean_val > min_intensity:
                        valid_circles.append((cx, cy, r))

            if not valid_circles: return []

            # Best candidate near players
            best_circle = None
            if players:
                min_dist = float("inf")
                for (cx, cy, r) in valid_circles:
                    for p in players:
                        px, py = ((p["bbox"][0] + p["bbox"][2]) // 2,
                                  (p["bbox"][1] + p["bbox"][3]) // 2)
                        d = ((cx - px)**2 + (cy - py)**2)**0.5
                        if d < min_dist:
                            min_dist = d
                            best_circle = (cx, cy, r)
                if min_dist > 500: best_circle = None
            else:
                best_circle = valid_circles[0]

            if best_circle:
                cx, cy, r = best_circle
                x1, y1 = max(0, cx-r), max(0, cy-r)
                x2, y2 = min(w, cx+r), min(h, cy+r)
                balls.append({"bbox": [x1, y1, x2, y2], "conf": 0.5})

        return balls
