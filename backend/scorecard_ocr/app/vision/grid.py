from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class CellBox:
    x: int
    y: int
    w: int
    h: int


def normalize_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    resized = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray, resized


def detect_grid(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = gray.shape[:2]
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        2,
    )

    horizontal_kernel_width = max(20, width // 18)
    vertical_kernel_height = max(20, height // 18)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_height))

    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    horizontal_lines = cv2.dilate(horizontal_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)

    grid_mask = cv2.add(horizontal_lines, vertical_lines)
    return grid_mask, thresh, horizontal_lines, vertical_lines


def _cluster_line_positions(mask: np.ndarray, axis: int, min_gap: int = 3) -> List[int]:
    sums = np.sum(mask, axis=axis)
    indices = np.where(sums > 0)[0]
    if len(indices) == 0:
        return []
    clusters: List[List[int]] = [[int(indices[0])]]
    for idx in indices[1:]:
        if int(idx) - clusters[-1][-1] <= min_gap:
            clusters[-1].append(int(idx))
        else:
            clusters.append([int(idx)])
    return [int(np.mean(cluster)) for cluster in clusters]


def extract_cells(horizontal_lines: np.ndarray, vertical_lines: np.ndarray) -> List[CellBox]:
    y_positions = _cluster_line_positions(horizontal_lines, axis=1)
    x_positions = _cluster_line_positions(vertical_lines, axis=0)
    boxes: List[CellBox] = []

    if len(x_positions) < 2 or len(y_positions) < 2:
        return boxes

    x_positions.sort()
    y_positions.sort()

    for y0, y1 in zip(y_positions[:-1], y_positions[1:]):
        for x0, x1 in zip(x_positions[:-1], x_positions[1:]):
            w = x1 - x0
            h = y1 - y0
            if w < 30 or h < 20:
                continue
            boxes.append(CellBox(x0, y0, w, h))

    return boxes


def sort_cells_into_grid(cells: List[CellBox], row_tolerance: int = 15) -> List[List[CellBox]]:
    cells_sorted = sorted(cells, key=lambda c: (c.y, c.x))
    rows: List[List[CellBox]] = []

    for cell in cells_sorted:
        placed = False
        for row in rows:
            if abs(row[0].y - cell.y) <= row_tolerance:
                row.append(cell)
                placed = True
                break
        if not placed:
            rows.append([cell])

    for row in rows:
        row.sort(key=lambda c: c.x)

    return rows
