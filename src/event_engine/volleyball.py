"""
event_engine/volleyball.py — Volleyball-specific event rules.

Events:
  SERVE       — ball moves fast from near baseline/player upward
  SPIKE       — ball moves very fast downward
  SET         — soft upward ball movement near a player (setter)
  BLOCK       — ball moving downward is deflected upward near the net
  RALLY       — ball crosses net area (horizontal movement + mid-court)
  DIG         — low player + ball moving upward
"""

import numpy as np
from .base_engine import BaseEventEngine, dist


class VolleyballEngine(BaseEventEngine):
    BALL_NEAR_PLAYER = 130
    SPIKE_DOWN_SPEED = 10    # vy > this = fast downward
    SERVE_SPEED = 12
    SET_SPEED_MAX = 7
    NET_ZONE_FRAC = 0.48     # approx net x-fraction of frame width

    def __init__(self):
        super().__init__(cooldown_frames=40)

    def analyze(self, frame_idx: int, players: list, balls: list,
                frame_w: int = 1280, frame_h: int = 720) -> list[dict]:
        events = []

        ball_pos = None
        if balls:
            b = balls[0]["bbox"]
            ball_pos = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
        self._update_ball_history(ball_pos)

        if not players or not ball_pos:
            return events

        vx, vy = self._ball_velocity()
        ball_speed = (vx**2 + vy**2) ** 0.5
        label_near, d_near = self._closest_player(ball_pos, players)

        # ---- SPIKE ----
        if vy > self.SPIKE_DOWN_SPEED and ball_speed > 10 and ball_pos[1] < frame_h * 0.55:
            key = f"spike_{label_near}"
            if not self._on_cooldown(key, frame_idx):
                events.append({"event": "SPIKE", "players": [label_near], "frame": frame_idx})
                self._register(key, frame_idx)

        # ---- SERVE ----
        if ball_speed > self.SERVE_SPEED and vy < -5 and d_near < self.BALL_NEAR_PLAYER:
            key = f"serve_{label_near}"
            if not self._on_cooldown(key, frame_idx):
                events.append({"event": "SERVE", "players": [label_near], "frame": frame_idx})
                self._register(key, frame_idx)

        # ---- SET ----
        if (ball_speed < self.SET_SPEED_MAX and vy < -2 and
                d_near < self.BALL_NEAR_PLAYER and ball_speed > 1):
            key = f"set_{label_near}"
            if not self._on_cooldown(key, frame_idx):
                events.append({"event": "SET", "players": [label_near], "frame": frame_idx})
                self._register(key, frame_idx)

        # ---- BLOCK ----
        if vy < -6 and ball_speed > 8 and ball_pos[1] < frame_h * 0.45:
            key = f"block_{label_near}"
            if not self._on_cooldown(key, frame_idx):
                events.append({"event": "BLOCK", "players": [label_near], "frame": frame_idx})
                self._register(key, frame_idx)

        # ---- RALLY (ball crosses net zone horizontally) ----
        hist = [p for p in self._ball_history if p is not None]
        if len(hist) >= 6:
            net_x = int(frame_w * self.NET_ZONE_FRAC)
            xs = [p[0] for p in hist[-6:]]
            if min(xs) < net_x < max(xs):
                if not self._on_cooldown("rally", frame_idx):
                    events.append({"event": "RALLY", "players": [], "frame": frame_idx})
                    self._register("rally", frame_idx)

        return events
