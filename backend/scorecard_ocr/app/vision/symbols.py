from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def detect_symbol(gray_cell: np.ndarray) -> Optional[str]:
    thresh = cv2.adaptiveThreshold(
        gray_cell,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        2,
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 30:
        return None

    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    if width == 0 or height == 0:
        return None

    aspect = max(width, height) / min(width, height)
    if aspect > 5:
        return "/"

    if len(contour) >= 4:
        return "checkmark"

    return None
