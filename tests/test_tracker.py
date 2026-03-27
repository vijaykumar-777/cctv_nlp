"""Tests for BoxMOTTracker."""

from __future__ import annotations

import numpy as np

from src.tracker.boxmot_tracker import BoxMOTTracker


def test_tracker_empty_input(sample_frame):
    """Tracker returns a list (empty) when given no detections."""
    tracker = BoxMOTTracker()
    result = tracker.update([], sample_frame)
    assert isinstance(result, list)


def test_tracker_with_detection(sample_frame, sample_detection):
    """Tracker returns a list when given a detection."""
    tracker = BoxMOTTracker()
    result = tracker.update([sample_detection], sample_frame)
    assert isinstance(result, list)
