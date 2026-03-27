"""Tests for YOLODetector and AttributeExtractor."""

from __future__ import annotations

import numpy as np

from src.detector.attribute_extractor import AttributeExtractor
from src.detector.yolo_detector import YOLODetector


def test_detection_is_list(sample_frame):
    """YOLODetector.detect() returns a list (possibly empty on a black frame)."""
    detector = YOLODetector()
    result = detector.detect(sample_frame)
    assert isinstance(result, list)


def test_attribute_color_red(red_frame):
    """AttributeExtractor identifies a pure-red frame as 'red'."""
    extractor = AttributeExtractor()
    color = extractor.extract_color(red_frame, [0, 0, 640, 480])
    assert color == "red"


def test_attribute_size_small(sample_frame):
    """A tiny bbox relative to the frame should be classified as 'small'."""
    extractor = AttributeExtractor()
    # 10×10 box in a 640×480 frame → (100 / 307200) ≈ 0.03% → small
    size = extractor.extract_size([0, 0, 10, 10], sample_frame.shape)
    assert size == "small"


def test_attribute_size_large(sample_frame):
    """A bbox covering most of the frame should be 'large'."""
    extractor = AttributeExtractor()
    size = extractor.extract_size([0, 0, 600, 450], sample_frame.shape)
    assert size == "large"
