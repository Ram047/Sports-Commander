"""
event_engine/football.py — Football/Soccer-specific event rules.
"""

import numpy as np
from .base_engine import BaseEventEngine, dist


class FootballEngine(BaseEventEngine):
    POSSESSION_DIST = 150
    PASS_MIN_SPEED = 6
    SHOOT_MIN_SPEED = 18
    DRIBBLE_SPEED = 5

    def __init__(self):
        super().__init__(cooldown_frames=30)
        self._stationary_timer: dict = {}

    def analyze(self, frame_idx: int, players: list, balls: list, 
                frame_w: int = 1280, frame_h: int = 720, tracker=None) -> list[dict]:
        events = []

        ball_pos = None
        if balls:
            b = balls[0]["bbox"]
            ball_pos = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
        self._update_ball_history(ball_pos)

        if not players:
            return events

        vx, vy = self._ball_velocity()
        ball_speed = (vx**2 + vy**2) ** 0.5

        prev_possession = self._possession
        if ball_pos:
            label, d = self._closest_player(ball_pos, players)
            self._possession = label if d < self.POSSESSION_DIST else None
        else:
            self._possession = None

        # ---- PASS ----
        if (prev_possession and self._possession and
                prev_possession != self._possession and
                ball_speed >= self.PASS_MIN_SPEED):
            key = f"pass_{prev_possession}_{self._possession}"
            if not self._on_cooldown(key, frame_idx):
                events.append({"event": "PASS", "players": [prev_possession, self._possession], "frame": frame_idx})
                self._register(key, frame_idx)

        # ---- TACKLE ----
        elif (prev_possession and self._possession and
              prev_possession != self._possession and
              ball_speed < self.PASS_MIN_SPEED):
            key = f"tackle_{self._possession}"
            if not self._on_cooldown(key, frame_idx):
                events.append({"event": "TACKLE", "players": [self._possession], "frame": frame_idx})
                self._register(key, frame_idx)

        # ---- SHOOT ----
        if ball_speed >= self.SHOOT_MIN_SPEED and self._possession is None:
            shooter = prev_possession
            key = f"shoot_{shooter or '?'}"
            if not self._on_cooldown(key, frame_idx):
                events.append({"event": "SHOOT", "players": [shooter], "frame": frame_idx})
                self._register(key, frame_idx)

        # ---- MOVES & POSITIONING (Universal Logic) ----
        if tracker:
            for p in players:
                label = p["label"]
                vel = tracker.get_velocity(label)
                dx, dy = tracker.get_direction_vector(label)
                
                # Directional Move
                if vel > 8:
                    dir_str = self._vector_to_dir(dx, dy)
                    key = f"move_{label}_{dir_str}"
                    if not self._on_cooldown(key, frame_idx):
                        events.append({"event": "MOVE", "players": [label], "dir": dir_str, "frame": frame_idx})
                        self._register(key, frame_idx)

        # ---- FILLER ----
        if (frame_idx - self._last_overall_event_frame) > 150:
            key = f"filler_{frame_idx // 200}"
            if not self._on_cooldown(key, frame_idx):
                events.append({"event": "FILLER", "players": [self._possession] if self._possession else [], "frame": frame_idx})
                self._register(key, frame_idx)

        return events
