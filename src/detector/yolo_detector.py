"""YOLO object detector — wraps Ultralytics YOLOv8."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from src.config import settings


@dataclass
class Detection:
    """A single object detection result."""
    bbox: list[float]       # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


class YOLODetector:
    """Runs YOLOv8 inference on frames.

    Loads the model once on init.  Uses GPU when available.
    """

    def __init__(self, model_path: str = settings.yolo_model) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = YOLO(model_path)
        self._model.to(self._device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single frame.

        Args:
            frame: BGR numpy array (H, W, 3).

        Returns:
            List of Detection objects (may be empty).
        """
        results = self._model(frame, conf=0.4, verbose=False)

        detections: list[Detection] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        data = boxes.data.cpu().numpy()  # (N, 6) — x1,y1,x2,y2,conf,cls

        for row in data:
            x1, y1, x2, y2, conf, cls_id = row
            cls_id = int(cls_id)
            class_name = self._model.names.get(cls_id, str(cls_id))
            detections.append(
                Detection(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=class_name,
                )
            )

        return detections
