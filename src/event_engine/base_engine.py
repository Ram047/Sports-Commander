"""
event_engine/base_engine.py — Abstract base for sport event engines.
"""

import numpy as np
from abc import ABC, abstractmethod


def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


class BaseEventEngine(ABC):
    def __init__(self, cooldown_frames: int = 45):
        self._last_event_frame: dict[str, int] = {}
        self._last_overall_event_frame: int = -9999
        self.cooldown = cooldown_frames  # ~1.5 s at 30fps
        self._ball_history: list = []  # list of (cx, cy) or None
        self._possession: str | None = None  # label of player with ball

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _on_cooldown(self, event_key: str, frame_idx: int) -> bool:
        last = self._last_event_frame.get(event_key, -9999)
        return (frame_idx - last) < self.cooldown

    def _register(self, event_key: str, frame_idx: int):
        self._last_event_frame[event_key] = frame_idx
        # Update overall excitement tracker unless it's a SPRINT or FILLER
        if "sprint" not in event_key and "filler" not in event_key:
            self._last_overall_event_frame = frame_idx

    def _closest_player(self, pos, players):
        """Return (label, distance) of player whose center is closest to pos."""
        best_label, best_d = None, float("inf")
        for p in players:
            d = dist(pos, p["center"])
            if d < best_d:
                best_d = d
                best_label = p["label"]
        return best_label, best_d

    def _vector_to_dir(self, dx: float, dy: float) -> str:
        """Converts normalized (dx,dy) to a technical direction string."""
        if abs(dx) > abs(dy):
            return "to the left" if dx < 0 else "to the right"
        else:
            return "up the court" if dy < 0 else "back to defense"

    def _update_ball_history(self, ball_pos):
        self._ball_history.append(ball_pos)
        if len(self._ball_history) > 30:
            self._ball_history.pop(0)

    def _ball_velocity(self):
        """Returns (vx, vy) over last 10 frames."""
        hist = [p for p in self._ball_history[-10:] if p is not None]
        if len(hist) < 4:
            return (0.0, 0.0)
        p1, p2 = np.array(hist[0]), np.array(hist[-1])
        diff = (p2 - p1) / len(hist)
        return float(diff[0]), float(diff[1])

    @abstractmethod
    def analyze(self, frame_idx: int, players: list, balls: list) -> list[dict]:
        """
        Returns list of event dicts: {'event': str, 'players': list[str], 'frame': int}
        """
        ...

    # ------------------------------------------------------------------ #
    # Common cross-sport events
    # ------------------------------------------------------------------ #
    def _detect_sprint(self, frame_idx, players, tracker):
        events = []
        SPRINT_THRESH = 12  # pixels/frame
        for p in players:
            v = tracker.get_velocity(p["label"])
            if v > SPRINT_THRESH:
                key = f"sprint_{p['label']}"
                if not self._on_cooldown(key, frame_idx):
                    events.append({"event": "SPRINT", "players": [p["label"]], "frame": frame_idx})
                    self._register(key, frame_idx)
        return events
