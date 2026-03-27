"""Diff detector — compares current scene against a registered baseline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from src.detector.yolo_detector import Detection
from src.scene_diff.baseline_manager import BaselineManager


@dataclass
class SceneDiffEvent:
    """An event representing a change between baseline and current scene."""
    camera_id: str
    object_label: str
    event_type: str          # "removed" | "appeared"
    timestamp_utc: datetime
    frame_number: int
    baseline_frame_path: str
    diff_frame_path: str | None


class DiffDetector:
    """Detects objects that have been removed from a baseline scene.

    Compares current detections against the registered baseline using
    IoU overlap and class matching.
    """

    _DIFFS_DIR = "./clips/diffs"

    def __init__(self, baseline_manager: BaselineManager) -> None:
        self._baseline_manager = baseline_manager
        os.makedirs(self._DIFFS_DIR, exist_ok=True)

    def check(
        self,
        camera_id: str,
        frame: np.ndarray,
        current_detections: list[Detection],
        timestamp_utc: datetime,
        frame_number: int,
    ) -> list[SceneDiffEvent]:
        """Compare current detections against the baseline.

        For each baseline object not found in the current frame
        (IoU > 0.3 with same class_id), emit a 'removed' event.

        Returns:
            List of SceneDiffEvent objects (may be empty).
        """
        baseline = self._baseline_manager.get(camera_id)
        if baseline is None:
            return []

        events: list[SceneDiffEvent] = []

        for baseline_obj in baseline.objects:
            # Check if any current detection overlaps this baseline object
            found = False
            for det in current_detections:
                if det.class_id != baseline_obj.class_id:
                    continue
                if self._iou(baseline_obj.bbox, det.bbox) > 0.3:
                    found = True
                    break

            if not found:
                # Object was removed — save diff frame
                diff_frame_path = os.path.join(
                    self._DIFFS_DIR, f"{camera_id}_{frame_number}.png"
                )
                cv2.imwrite(diff_frame_path, frame)

                events.append(
                    SceneDiffEvent(
                        camera_id=camera_id,
                        object_label=baseline_obj.label,
                        event_type="removed",
                        timestamp_utc=timestamp_utc,
                        frame_number=frame_number,
                        baseline_frame_path=baseline.frame_path,
                        diff_frame_path=diff_frame_path,
                    )
                )

        return events

    @staticmethod
    def _iou(box1: list[float], box2: list[float]) -> float:
        """Compute Intersection over Union between two bounding boxes.

        Each box is [x1, y1, x2, y2].
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        intersection = inter_w * inter_h

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union
