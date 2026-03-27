"""Attribute extractor — color and size from detected bounding boxes."""

from __future__ import annotations

import cv2
import numpy as np


class AttributeExtractor:
    """Extracts visual attributes (color, size) from object crops."""

    # Hue ranges (OpenCV hue is 0-179)
    _COLOR_MAP = [
        (  0,  10, "red"),
        ( 10,  25, "orange"),
        ( 25,  35, "yellow"),
        ( 35,  85, "green"),
        ( 85, 125, "blue"),
        (125, 155, "purple"),
        (155, 180, "red"),  # wraps around
    ]

    # Skin-tone HSV range to mask out
    _SKIN_LOW = np.array([0, 20, 70], dtype=np.uint8)
    _SKIN_HIGH = np.array([20, 255, 255], dtype=np.uint8)

    def extract_color(self, frame: np.ndarray, bbox: list) -> str:
        """Determine the dominant color of the object inside *bbox*.

        Crops the region, converts to HSV, masks out skin tones,
        and maps the dominant hue to a color name.
        """
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Mask out skin tones
        skin_mask = cv2.inRange(hsv, self._SKIN_LOW, self._SKIN_HIGH)
        non_skin_mask = cv2.bitwise_not(skin_mask)

        # Check for achromatic pixels (low saturation)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        non_skin_sat = sat[non_skin_mask > 0] if np.any(non_skin_mask > 0) else sat.flatten()
        non_skin_val = val[non_skin_mask > 0] if np.any(non_skin_mask > 0) else val.flatten()

        if len(non_skin_sat) == 0:
            return "unknown"

        median_sat = float(np.median(non_skin_sat))
        median_val = float(np.median(non_skin_val))

        # Achromatic detection
        if median_sat < 40:
            if median_val < 60:
                return "black"
            elif median_val > 200:
                return "white"
            elif median_val > 140:
                return "silver"
            else:
                return "gray"

        # Chromatic — find dominant hue
        hue = hsv[:, :, 0]
        non_skin_hue = hue[non_skin_mask > 0] if np.any(non_skin_mask > 0) else hue.flatten()
        if len(non_skin_hue) == 0:
            return "unknown"

        dominant_hue = float(np.median(non_skin_hue))

        for low, high, color_name in self._COLOR_MAP:
            if low <= dominant_hue < high:
                return color_name

        return "unknown"

    def extract_size(self, bbox: list, frame_shape: tuple) -> str:
        """Classify object size relative to the frame.

        Returns 'small' (<2%), 'medium' (2-15%), or 'large' (>15%).
        """
        x1, y1, x2, y2 = bbox[:4]
        obj_area = abs((x2 - x1) * (y2 - y1))
        frame_area = frame_shape[0] * frame_shape[1]  # H * W

        if frame_area == 0:
            return "unknown"

        ratio = obj_area / frame_area

        if ratio < 0.02:
            return "small"
        elif ratio <= 0.15:
            return "medium"
        else:
            return "large"
