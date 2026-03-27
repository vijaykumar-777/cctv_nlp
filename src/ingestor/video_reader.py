"""Video reader — iterates frames from a video file via OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


class VideoReader:
    """Read frames from a video file.

    Yields (frame_number, timestamp_sec, frame) tuples.
    Supports context-manager protocol for automatic resource cleanup.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")

    # ── Properties ──────────────────────────────────────────────

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Iterator ────────────────────────────────────────────────

    def __iter__(self):
        frame_number = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if frame is None:
                frame_number += 1
                continue  # skip None frames silently
            timestamp_sec = frame_number / self.fps if self.fps > 0 else 0.0
            yield frame_number, timestamp_sec, frame
            frame_number += 1

    # ── Context manager ─────────────────────────────────────────

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def release(self) -> None:
        """Release the underlying VideoCapture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
