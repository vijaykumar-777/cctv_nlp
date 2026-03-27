"""Multi-object tracker — wraps BoxMOT's ByteTrack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from boxmot import ByteTrack

from src.detector.yolo_detector import Detection


@dataclass
class Track:
    """A tracked object with persistent identity."""
    track_id: int
    bbox: list[float]       # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


class BoxMOTTracker:
    """ByteTrack-based multi-object tracker.

    Wraps boxmot.BYTETracker to assign persistent track IDs
    across frames.
    """

    def __init__(self) -> None:
        self._tracker = ByteTrack()

    def update(
        self, detections: list[Detection], frame: np.ndarray
    ) -> list[Track]:
        """Feed detections for a frame and return active tracks.

        Args:
            detections: List of Detection objects from YOLODetector.
            frame: The current BGR frame (passed to tracker for features).

        Returns:
            List of Track objects with stable track_id values.
        """
        if not detections:
            dets = np.empty((0, 6), dtype=np.float64)
        else:
            rows = []
            for d in detections:
                rows.append([
                    d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
                    d.confidence, float(d.class_id),
                ])
            dets = np.array(rows, dtype=np.float64)

        # BoxMOT update returns array with columns:
        # [x1, y1, x2, y2, track_id, confidence, class_id, index]
        tracked = self._tracker.update(dets, frame)

        tracks: list[Track] = []
        if tracked is None or len(tracked) == 0:
            return tracks

        for row in tracked:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            track_id = int(row[4])
            conf = float(row[5])
            cls_id = int(row[6])

            # Resolve class name from detections if possible
            class_name = str(cls_id)
            for d in detections:
                if d.class_id == cls_id:
                    class_name = d.class_name
                    break

            tracks.append(
                Track(
                    track_id=track_id,
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    class_id=cls_id,
                    class_name=class_name,
                )
            )

        return tracks
