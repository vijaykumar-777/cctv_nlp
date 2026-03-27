"""Tests for BaselineManager and DiffDetector."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from src.scene_diff.baseline_manager import BaselineManager
from src.scene_diff.diff_detector import DiffDetector


def test_no_diff_no_baseline(tmp_path, sample_frame):
    """DiffDetector returns [] when no baseline is registered."""
    mgr = BaselineManager(baselines_dir=str(tmp_path))
    detector = DiffDetector(mgr)

    result = detector.check(
        camera_id="cam_test",
        frame=sample_frame,
        current_detections=[],
        timestamp_utc=datetime.now(timezone.utc),
        frame_number=0,
    )
    assert result == []


def test_removal_detected(tmp_path, sample_frame, sample_detection):
    """DiffDetector emits a 'removed' event when a baseline object disappears."""
    baselines_dir = str(tmp_path / "baselines")
    diffs_dir = str(tmp_path / "diffs")

    mgr = BaselineManager(baselines_dir=baselines_dir)
    detector = DiffDetector(mgr)
    detector._DIFFS_DIR = diffs_dir

    import os
    os.makedirs(diffs_dir, exist_ok=True)

    # Register baseline with one detection
    mgr.register("cam_test", sample_frame, [sample_detection])

    # Check with NO current detections → object removed
    result = detector.check(
        camera_id="cam_test",
        frame=sample_frame,
        current_detections=[],
        timestamp_utc=datetime.now(timezone.utc),
        frame_number=100,
    )

    assert len(result) == 1
    assert result[0].event_type == "removed"
    assert result[0].object_label == "car"
    assert result[0].camera_id == "cam_test"


def test_no_removal_when_object_present(tmp_path, sample_frame, sample_detection):
    """DiffDetector returns [] when baseline objects are still present."""
    baselines_dir = str(tmp_path / "baselines")
    mgr = BaselineManager(baselines_dir=baselines_dir)
    detector = DiffDetector(mgr)

    # Register baseline with one detection
    mgr.register("cam_test", sample_frame, [sample_detection])

    # Check with the SAME detection still present → no removal
    result = detector.check(
        camera_id="cam_test",
        frame=sample_frame,
        current_detections=[sample_detection],
        timestamp_utc=datetime.now(timezone.utc),
        frame_number=100,
    )

    assert result == []
