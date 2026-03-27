"""Frame sampler — decimates video frames to a target FPS."""

from __future__ import annotations

import numpy as np

from src.ingestor.video_reader import VideoReader


class FrameSampler:
    """Wraps VideoReader and yields frames at a target FPS.

    Skips frames to achieve the desired sampling rate.
    Yields the same (frame_number, timestamp_sec, frame) tuple as VideoReader.
    """

    def __init__(self, video_path: str, target_fps: float = 2.0) -> None:
        self._video_path = video_path
        self._target_fps = target_fps
        self._reader = VideoReader(video_path)

    @property
    def reader(self) -> VideoReader:
        return self._reader

    def __iter__(self):
        source_fps = self._reader.fps
        if source_fps <= 0:
            # Can't compute skip interval — yield everything
            yield from self._reader
            return

        # How many source frames to skip between samples
        skip_interval = max(1.0, source_fps / self._target_fps)
        next_sample_at = 0.0

        for frame_number, timestamp_sec, frame in self._reader:
            if frame_number >= next_sample_at:
                yield frame_number, timestamp_sec, frame
                next_sample_at += skip_interval
