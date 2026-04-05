"""
tracker.py — Multi-object player tracking using supervision ByteTrack.
Assigns stable letter IDs (A, B, C, …) to tracked players.
"""

import numpy as np
import supervision as sv
from collections import defaultdict


class Tracker:
    def __init__(self, max_age: int = 30, trajectory_len: int = 20):
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.35,
            lost_track_buffer=max_age,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )
        self.trajectory_len = trajectory_len

        # Map numeric track_id → letter label (A, B, C, ...)
        self._id_map: dict[int, str] = {}
        self._next_letter = 0
        self._trajectories: dict[str, list] = defaultdict(list)

    def _get_label(self, track_id: int) -> str:
        if track_id not in self._id_map:
            letter = chr(ord("A") + self._next_letter % 26)
            if self._next_letter >= 26:
                letter = chr(ord("A") + (self._next_letter // 26) - 1) + letter
            self._id_map[track_id] = letter
            self._next_letter += 1
        return self._id_map[track_id]

    def update(self, persons: list, frame_shape: tuple) -> list:
        """
        persons: list of {'bbox': [x1,y1,x2,y2], 'conf': float}
        Returns list of {'label': str, 'bbox': [x1,y1,x2,y2], 'center': (cx,cy)}
        """
        if not persons:
            return []

        xyxy = np.array([p["bbox"] for p in persons], dtype=float)
        confs = np.array([p["conf"] for p in persons], dtype=float)
        class_ids = np.zeros(len(persons), dtype=int)

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=class_ids,
        )

        tracked = self.byte_tracker.update_with_detections(detections)

        results = []
        for i in range(len(tracked)):
            bbox = tracked.xyxy[i].tolist()
            x1, y1, x2, y2 = map(int, bbox)
            track_id = int(tracked.tracker_id[i])
            label = self._get_label(track_id)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Record trajectory
            traj = self._trajectories[label]
            traj.append((cx, cy))
            if len(traj) > self.trajectory_len:
                traj.pop(0)

            results.append({
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "track_id": track_id,
            })

        return results

    def get_trajectory(self, label: str) -> list:
        return list(self._trajectories.get(label, []))

    def get_velocity(self, label: str) -> float:
        """Returns pixel-per-frame speed over last 5 frames."""
        traj = self.get_trajectory(label)
        if len(traj) < 5:
            return 0.0
        p1 = np.array(traj[-5])
        p2 = np.array(traj[-1])
        return float(np.linalg.norm(p2 - p1) / 4)

    def get_direction_vector(self, label: str, window: int = 10) -> tuple[float, float]:
        """Returns (dx, dy) normalized direction over last X frames."""
        traj = self.get_trajectory(label)
        if len(traj) < window:
            return (0.0, 0.0)
        p_old = np.array(traj[-window])
        p_new = np.array(traj[-1])
        diff = p_new - p_old
        norm = np.linalg.norm(diff)
        if norm < 2: # No movement
            return (0.0, 0.0)
        return float(diff[0] / norm), float(diff[1] / norm)
