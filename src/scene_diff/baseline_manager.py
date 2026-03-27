"""Baseline manager — registers and retrieves scene baselines for diff detection."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import cv2
import numpy as np

from src.detector.yolo_detector import Detection


@dataclass
class BaselineObject:
    """A single object in the baseline scene."""
    label: str
    bbox: list[float]
    class_id: int


@dataclass
class BaselineState:
    """Snapshot of the scene used as a reference for diff detection."""
    camera_id: str
    objects: list[BaselineObject]
    frame_path: str
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BaselineManager:
    """Manages scene baselines — register, retrieve, and clear.

    Stores baseline frames as PNG images and state as JSON files.
    """

    def __init__(self, baselines_dir: str = "./clips/baselines") -> None:
        self._baselines_dir = baselines_dir
        os.makedirs(baselines_dir, exist_ok=True)

    def register(
        self,
        camera_id: str,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> BaselineState:
        """Register a new baseline for a camera.

        Saves the frame as PNG and the object list as JSON.
        """
        frame_path = os.path.join(self._baselines_dir, f"{camera_id}_baseline.png")
        state_path = os.path.join(self._baselines_dir, f"{camera_id}_state.json")

        cv2.imwrite(frame_path, frame)

        objects = [
            BaselineObject(label=d.class_name, bbox=d.bbox, class_id=d.class_id)
            for d in detections
        ]

        state = BaselineState(
            camera_id=camera_id,
            objects=objects,
            frame_path=frame_path,
        )

        # Serialize to JSON
        state_dict = {
            "camera_id": state.camera_id,
            "objects": [asdict(o) for o in state.objects],
            "frame_path": state.frame_path,
            "registered_at": state.registered_at.isoformat(),
        }
        with open(state_path, "w") as f:
            json.dump(state_dict, f, indent=2)

        return state

    def get(self, camera_id: str) -> BaselineState | None:
        """Load a previously registered baseline for a camera.

        Returns None if no baseline exists.
        """
        state_path = os.path.join(self._baselines_dir, f"{camera_id}_state.json")
        if not os.path.exists(state_path):
            return None

        with open(state_path) as f:
            data = json.load(f)

        objects = [
            BaselineObject(label=o["label"], bbox=o["bbox"], class_id=o["class_id"])
            for o in data["objects"]
        ]

        return BaselineState(
            camera_id=data["camera_id"],
            objects=objects,
            frame_path=data["frame_path"],
            registered_at=datetime.fromisoformat(data["registered_at"]),
        )

    def clear(self, camera_id: str) -> None:
        """Remove a camera's baseline files."""
        frame_path = os.path.join(self._baselines_dir, f"{camera_id}_baseline.png")
        state_path = os.path.join(self._baselines_dir, f"{camera_id}_state.json")

        for path in (frame_path, state_path):
            if os.path.exists(path):
                os.remove(path)
