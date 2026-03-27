"""Event builder — constructs enriched event records from tracked objects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.tracker.boxmot_tracker import Track


@dataclass
class EventRecord:
    """A fully enriched event ready for persistence and indexing."""
    camera_id: str
    track_id: int | None
    timestamp_utc: datetime
    frame_number: int
    class_name: str
    attributes: dict
    bbox: dict
    confidence: float
    caption: str
    video_path: str
    frame_crop: np.ndarray


class EventBuilder:
    """Builds EventRecord objects from tracked detections."""

    def build(
        self,
        camera_id: str,
        track: Track,
        attributes: dict,
        timestamp_utc: datetime,
        frame_number: int,
        frame: np.ndarray,
        video_path: str,
    ) -> EventRecord:
        """Construct a complete EventRecord.

        Generates a human-readable caption and crops the object
        from the frame.
        """
        bbox_dict = {
            "x1": track.bbox[0],
            "y1": track.bbox[1],
            "x2": track.bbox[2],
            "y2": track.bbox[3],
        }

        caption = self._generate_caption(track.class_name, attributes, camera_id)
        crop = self._crop_frame(frame, track.bbox)

        return EventRecord(
            camera_id=camera_id,
            track_id=track.track_id,
            timestamp_utc=timestamp_utc,
            frame_number=frame_number,
            class_name=track.class_name,
            attributes=attributes,
            bbox=bbox_dict,
            confidence=track.confidence,
            caption=caption,
            video_path=video_path,
            frame_crop=crop,
        )

    @staticmethod
    def _generate_caption(
        class_name: str, attributes: dict, camera_id: str
    ) -> str:
        """Create a human-readable caption for the event."""
        color = attributes.get("color", "")

        if class_name.lower() == "person":
            clothing_color = color if color else "unknown"
            return f"Person in {clothing_color} clothing on {camera_id}"

        prefix = f"{color} " if color else ""
        return f"{prefix}{class_name} detected on {camera_id}"

    @staticmethod
    def _crop_frame(frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        """Crop the object region from the frame, clipped to bounds."""
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        crop = frame[y1:y2, x1:x2]

        # Return a 1×1 black pixel if crop is empty
        if crop.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        return crop.copy()
