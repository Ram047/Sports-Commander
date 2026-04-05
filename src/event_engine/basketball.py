"""
event_engine/basketball.py — Basketball-specific event rules.
"""

import numpy as np
from .base_engine import BaseEventEngine, dist


class BasketballEngine(BaseEventEngine):
    BALL_POSSESSION_DIST = 180   # px — ball within this range = in possession
    PASS_MIN_DIST = 150          # px — minimum distance for a pass
    SHOT_UPWARD_SPEED = -8       # vy < this = rapid upward movement
    DRIBBLE_OSC_FRAMES = 6       # frames to detect oscillation

    def __init__(self):
        super().__init__(cooldown_frames=30)
        self._dribble_buffer: dict[str, list] = {}

    def analyze(self, frame_idx: int, players: list, balls: list, tracker=None) -> list[dict]:
        events = []
        if not tracker: return events

        # ---- Ball position ----
        ball_pos = None
        if balls:
            b = balls[0]["bbox"]
            ball_pos = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
        self._update_ball_history(ball_pos)
        if not players: return events

        # ---- Detect possession ----
        prev_possession = self._possession
        if ball_pos:
            label, d = self._closest_player(ball_pos, players)
            self._possession = label if d < self.BALL_POSSESSION_DIST else None
        else:
            self._possession = None

        vx, vy = self._ball_velocity()
        ball_speed = (vx**2 + vy**2) ** 0.5

        # ---- PASS ----
        if (prev_possession and self._possession and
                prev_possession != self._possession and
                ball_speed > 4):
            key = f"pass_{prev_possession}_{self._possession}"
            if not self._on_cooldown(key, frame_idx):
                events.append({"event": "PASS", "players": [prev_possession, self._possession], "frame": frame_idx})
                self._register(key, frame_idx)

        # ---- INTERCEPT ----
        elif (prev_possession and self._possession and
              prev_possession != self._possession and
              ball_speed < 4):
            key = f"intercept_{self._possession}"
            if not self._on_cooldown(key, frame_idx):
                events.append({"event": "INTERCEPT", "players": [self._possession], "frame": frame_idx})
                self._register(key, frame_idx)

        # ---- SHOT ----
        if ball_pos and vy < self.SHOT_UPWARD_SPEED and ball_speed > 12:
            shooter = prev_possession or self._possession
            key = f"shot_{shooter or '?'}"
            if not self._on_cooldown(key, frame_idx):
                events.append({"event": "SHOT", "players": [shooter], "frame": frame_idx})
                self._register(key, frame_idx)

        # ---- DRIBBLE ----
        if self._possession and ball_pos:
            buf = self._dribble_buffer.setdefault(self._possession, [])
            buf.append(ball_pos[1])
            if len(buf) > self.DRIBBLE_OSC_FRAMES:
                buf.pop(0)
            if len(buf) == self.DRIBBLE_OSC_FRAMES:
                diffs = [buf[i+1] - buf[i] for i in range(len(buf)-1)]
                sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i]*diffs[i+1] < 0)
                if sign_changes >= 3:
                    key = f"dribble_{self._possession}"
                    if not self._on_cooldown(key, frame_idx):
                        events.append({"event": "DRIBBLE", "players": [self._possession], "frame": frame_idx})
                        self._register(key, frame_idx)

        # ---- MOVES & POSITIONING (Universal Logic) ----
        for p in players:
            label = p["label"]
            vel = tracker.get_velocity(label)
            dx, dy = tracker.get_direction_vector(label)
            
            # 1. Directional Move
            if vel > 8:
                dir_str = self._vector_to_dir(dx, dy)
                event_type = "MOVE"
                if label == self._possession and dy < -0.6: # Significant upward movement with ball
                    event_type = "DRIVE"
                
                key = f"{event_type}_{label}_{dir_str}"
                if not self._on_cooldown(key, frame_idx):
                    events.append({"event": event_type, "players": [label], "dir": dir_str, "frame": frame_idx})
                    self._register(key, frame_idx)

            # 2. Defensive Pressure
            if self._possession and label != self._possession:
                carrier_pos = [pl["center"] for pl in players if pl["label"]==self._possession]
                if carrier_pos:
                    d_to_carrier = dist(p["center"], carrier_pos[0])
                    if d_to_carrier < 100:
                        key = f"defend_{label}_{self._possession}"
                        if not self._on_cooldown(key, frame_idx):
                            events.append({"event": "DEFEND", "players": [self._possession, label], "frame": frame_idx})
                            self._register(key, frame_idx)

        # Static filler if nothing for a while
        if (frame_idx - self._last_overall_event_frame) > 150:
            key = f"filler_{frame_idx // 200}"
            if not self._on_cooldown(key, frame_idx):
                events.append({"event": "FILLER", "players": [self._possession] if self._possession else [], "frame": frame_idx})
                self._register(key, frame_idx)

        return events
