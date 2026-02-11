from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import logging

import re
import cv2
import numpy as np

from app.models import ScorecardHole, ScorecardMetadata, ScorecardTotals, PlayerHole, ScorecardPlayer
from app.ocr.gemini_api import parse_scorecard_with_gemini
from app.vision.grid import CellBox, detect_grid, extract_cells, normalize_image, sort_cells_into_grid


def _crop_cell(gray: np.ndarray, cell: CellBox, padding: int = 1) -> np.ndarray:
    x0 = max(cell.x + padding, 0)
    y0 = max(cell.y + padding, 0)
    x1 = min(cell.x + cell.w - padding, gray.shape[1])
    y1 = min(cell.y + cell.h - padding, gray.shape[0])
    return gray[y0:y1, x0:x1]


def _enhance_cell_for_ocr(gray_cell: np.ndarray) -> np.ndarray:
    if gray_cell.size == 0:
        return gray_cell
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray_cell)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    enhanced = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        5,
    )
    return enhanced


def _classify_cell(gray_cell: np.ndarray) -> str:
    if gray_cell.size == 0:
        return "empty"
    _, thresh = cv2.threshold(gray_cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = float(np.count_nonzero(thresh)) / float(thresh.size)
    contrast = float(np.std(gray_cell))
    if ink_ratio < 0.01 and contrast < 12:
        return "empty"
    return "printed"


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split()).lower()


def _extract_int(value: str) -> Optional[int]:
    match = re.search(r"\d+", value)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _get_row_label(row: List[str]) -> str:
    parts: List[str] = []
    for cell in row[:4]:
        cell_text = cell.strip()
        if not cell_text:
            continue
        if cell_text[:1].isdigit():
            break
        parts.append(cell_text)
    return " ".join(parts).strip()


def _build_header_map(
    rows_with_text: List[List[Dict[str, object]]],
) -> Tuple[Dict[int, Tuple[float, float]], List[Tuple[float, float]]]:
    totals_labels = {"out", "in", "int", "tot", "total", "hcp", "net"}
    best_row: Optional[List[Dict[str, object]]] = None
    best_count = 0

    for row in rows_with_text:
        hole_cells: List[Tuple[int, Dict[str, object]]] = []
        for cell in row:
            text = str(cell["text"]).strip()
            try:
                num = int(text)
            except (ValueError, TypeError):
                continue
            if 1 <= num <= 18:
                hole_cells.append((num, cell))
        if len(hole_cells) > best_count:
            best_count = len(hole_cells)
            best_row = row

    if not best_row:
        return _build_header_map_fallback(rows_with_text)

    header_cells: List[Tuple[int, float, float]] = []
    total_ranges: List[Tuple[float, float]] = []
    for cell in best_row:
        text = str(cell["text"]).strip()
        lower = _normalize_text(text)
        box = cell["box"]
        x0 = float(box.x)
        x1 = float(box.x + box.w)
        if lower in totals_labels:
            total_ranges.append((x0, x1))
            continue
        try:
            num = int(text)
        except (ValueError, TypeError):
            continue
        if 1 <= num <= 18:
            header_cells.append((num, x0, x1))

    header_cells.sort(key=lambda item: item[1])
    if not header_cells:
        return _build_header_map_fallback(rows_with_text)

    centers = [((x0 + x1) / 2.0) for _, x0, x1 in header_cells]
    boundaries: List[float] = []
    for idx, center in enumerate(centers):
        if idx == 0:
            boundaries.append(center - (centers[1] - center) / 2 if len(centers) > 1 else center - 20)
        else:
            prev_center = centers[idx - 1]
            boundaries.append((prev_center + center) / 2)
    boundaries.append(centers[-1] + (centers[-1] - centers[-2]) / 2 if len(centers) > 1 else centers[-1] + 20)

    hole_ranges: Dict[int, Tuple[float, float]] = {}
    for idx, (hole, _, _) in enumerate(header_cells):
        hole_ranges[hole] = (boundaries[idx], boundaries[idx + 1])

    return hole_ranges, total_ranges


def _build_header_map_fallback(
    rows_with_text: List[List[Dict[str, object]]],
) -> Tuple[Dict[int, Tuple[float, float]], List[Tuple[float, float]]]:
    totals_labels = {"out", "in", "int", "tot", "total", "hcp", "net"}
    label_keywords = {"hole", "par", "handicap", "hcp", "men", "women", "ladies"}

    if not rows_with_text:
        return {}, []

    widest_row = max(rows_with_text, key=len)
    if not widest_row:
        return {}, []

    # Build column centers from the widest row geometry.
    column_cells = sorted(widest_row, key=lambda cell: cell["box"].x)
    column_centers = [float(cell["box"].x + cell["box"].w / 2) for cell in column_cells]
    if not column_centers:
        return {}, []

    # Aggregate column stats by assigning each cell to nearest column center.
    col_values: List[List[int]] = [[] for _ in column_centers]
    col_texts: List[List[str]] = [[] for _ in column_centers]

    for row in rows_with_text:
        for cell in row:
            text = str(cell["text"]).strip()
            if not text:
                continue
            center = float(cell["box"].x + cell["box"].w / 2)
            nearest_idx = min(range(len(column_centers)), key=lambda idx: abs(column_centers[idx] - center))
            col_texts[nearest_idx].append(_normalize_text(text))
            try:
                value = int(text)
            except (ValueError, TypeError):
                continue
            col_values[nearest_idx].append(value)

    total_ranges: List[Tuple[float, float]] = []
    candidate_indices: List[int] = []

    for idx, center in enumerate(column_centers):
        texts = col_texts[idx]
        values = col_values[idx]
        numeric_ratio = len(values) / max(1, len(texts))

        has_total_label = any(label in text for text in texts for label in totals_labels)
        has_label_keyword = any(label in text for text in texts for label in label_keywords)
        median_value = np.median(values) if values else 0
        looks_like_total = bool(values) and median_value >= 700

        if has_total_label or looks_like_total:
            # Treat as OUT/IN/TOT columns.
            total_ranges.append((center - 10, center + 10))
            continue
        if has_label_keyword:
            continue
        if numeric_ratio < 0.3:
            continue
        candidate_indices.append(idx)

    # If we still don't have enough, fall back to all columns that are not totals/labels.
    if len(candidate_indices) < 9:
        non_total_indices: List[int] = []
        for idx, center in enumerate(column_centers):
            texts = col_texts[idx]
            values = col_values[idx]
            has_total_label = any(label in text for text in texts for label in totals_labels)
            has_label_keyword = any(label in text for text in texts for label in label_keywords)
            median_value = np.median(values) if values else 0
            looks_like_total = bool(values) and median_value >= 700
            if has_total_label or looks_like_total or has_label_keyword:
                continue
            non_total_indices.append(idx)
        candidate_indices = non_total_indices

    # Use left-to-right ordering and take first 18 columns as holes.
    candidate_indices = sorted(candidate_indices, key=lambda idx: column_centers[idx])
    hole_indices = candidate_indices[:18]
    if len(hole_indices) < 9:
        return {}, total_ranges

    hole_indices.sort(key=lambda idx: column_centers[idx])
    boundaries: List[float] = []
    for pos, idx in enumerate(hole_indices):
        center = column_centers[idx]
        if pos == 0:
            next_center = column_centers[hole_indices[pos + 1]] if len(hole_indices) > 1 else center + 20
            boundaries.append(center - (next_center - center) / 2)
        else:
            prev_center = column_centers[hole_indices[pos - 1]]
            boundaries.append((prev_center + center) / 2)
    last_center = column_centers[hole_indices[-1]]
    prev_center = column_centers[hole_indices[-2]] if len(hole_indices) > 1 else last_center - 20
    boundaries.append(last_center + (last_center - prev_center) / 2)

    hole_ranges: Dict[int, Tuple[float, float]] = {}
    for pos, idx in enumerate(hole_indices):
        hole_ranges[pos + 1] = (boundaries[pos], boundaries[pos + 1])

    return hole_ranges, total_ranges


def _cluster_columns_from_cells(
    rows_with_text: List[List[Dict[str, object]]], tolerance: float = 12.0
) -> List[Tuple[float, float, float]]:
    centers: List[Tuple[float, float, float]] = []
    for row in rows_with_text:
        for cell in row:
            box = cell["box"]
            center = float(box.x + box.w / 2)
            centers.append((center, float(box.x), float(box.x + box.w)))

    if not centers:
        return []

    centers.sort(key=lambda item: item[0])
    clusters: List[List[Tuple[float, float, float]]] = []
    for center, left, right in centers:
        if not clusters:
            clusters.append([(center, left, right)])
            continue
        last_cluster = clusters[-1]
        last_center = sum(item[0] for item in last_cluster) / len(last_cluster)
        if abs(center - last_center) <= tolerance:
            last_cluster.append((center, left, right))
        else:
            clusters.append([(center, left, right)])

    ranges: List[Tuple[float, float, float]] = []
    for cluster in clusters:
        min_left = min(item[1] for item in cluster)
        max_right = max(item[2] for item in cluster)
        center = (min_left + max_right) / 2.0
        ranges.append((center, min_left, max_right))
    return ranges


def _total_ranges_from_text(rows_with_text: List[List[Dict[str, object]]]) -> List[Tuple[float, float]]:
    totals_labels = {"out", "in", "int", "tot", "total", "hcp", "net"}
    ranges: List[Tuple[float, float]] = []
    for row in rows_with_text:
        for cell in row:
            text = _normalize_text(str(cell["text"]))
            if not text:
                continue
            if text in totals_labels:
                box = cell["box"]
                ranges.append((float(box.x), float(box.x + box.w)))
    return ranges


def _infer_hole_ranges_geometry(
    rows_with_text: List[List[Dict[str, object]]],
) -> Tuple[Dict[int, Tuple[float, float]], List[Tuple[float, float]]]:
    column_ranges = _cluster_columns_from_cells(rows_with_text)
    if len(column_ranges) < 9:
        return {}, []

    total_ranges = _total_ranges_from_text(rows_with_text)
    filtered_columns: List[Tuple[float, float, float]] = []
    for center, left, right in column_ranges:
        if any(start <= center <= end for start, end in total_ranges):
            continue
        filtered_columns.append((center, left, right))

    if not filtered_columns:
        return {}, total_ranges

    widths = [right - left for _, left, right in filtered_columns]
    median_width = float(np.median(widths)) if widths else 0.0

    candidates: List[Tuple[float, float, float]] = []
    for center, left, right in filtered_columns:
        width = right - left
        if median_width > 0 and width > 1.6 * median_width:
            total_ranges.append((left, right))
            continue
        if median_width > 0 and 0.6 * median_width <= width <= 1.6 * median_width:
            candidates.append((center, left, right))

    if len(candidates) < 9:
        candidates = filtered_columns

    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 18:
        candidates = sorted(
            candidates,
            key=lambda item: abs((item[2] - item[1]) - median_width),
        )[:18]
        candidates.sort(key=lambda item: item[0])

    hole_ranges: Dict[int, Tuple[float, float]] = {}
    for idx, (_, left, right) in enumerate(candidates[:18], start=1):
        hole_ranges[idx] = (left, right)

    if len(hole_ranges) < 9:
        return {}, total_ranges

    return hole_ranges, total_ranges


def _row_label_from_cells(row: List[Dict[str, object]], first_hole_start: float) -> str:
    label_parts: List[str] = []
    for cell in row:
        box = cell["box"]
        if box.x >= first_hole_start - 5:
            continue
        text = str(cell["text"]).strip()
        if not text:
            continue
        if text[:1].isdigit():
            continue
        label_parts.append(text)
    return " ".join(label_parts).strip()


def _map_cells_to_holes(
    row: List[Dict[str, object]],
    hole_ranges: Dict[int, Tuple[float, float]],
    total_ranges: List[Tuple[float, float]],
) -> Dict[int, int]:
    values: Dict[int, int] = {}
    for cell in row:
        text = str(cell["text"]).strip()
        if not text:
            continue
        value = _extract_int(text)
        if value is None:
            continue
        box = cell["box"]
        center = float(box.x + box.w / 2)
        if any(start <= center <= end for start, end in total_ranges):
            continue
        for hole, (start, end) in hole_ranges.items():
            if start <= center <= end:
                values[hole] = value
                break
    return values


def _map_cells_to_holes_nearest(
    row: List[Dict[str, object]],
    hole_ranges: Dict[int, Tuple[float, float]],
    total_ranges: List[Tuple[float, float]],
) -> Dict[int, int]:
    values: Dict[int, int] = {}
    hole_centers = {hole: (start + end) / 2.0 for hole, (start, end) in hole_ranges.items()}
    for cell in row:
        text = str(cell["text"]).strip()
        if not text:
            continue
        value = _extract_int(text)
        if value is None:
            continue
        box = cell["box"]
        center = float(box.x + box.w / 2)
        if any(start <= center <= end for start, end in total_ranges):
            continue
        if not hole_centers:
            continue
        nearest_hole = min(hole_centers.keys(), key=lambda hole: abs(hole_centers[hole] - center))
        if nearest_hole not in values:
            values[nearest_hole] = value
    return values


def _row_numeric_values(row: List[Dict[str, object]]) -> List[int]:
    values: List[int] = []
    for cell in row:
        text = str(cell["text"]).strip()
        if not text:
            continue
        value = _extract_int(text)
        if value is None:
            continue
        values.append(value)
    return values


def _map_cells_by_order(
    row: List[Dict[str, object]],
    hole_ranges: Dict[int, Tuple[float, float]],
    total_ranges: List[Tuple[float, float]],
) -> Dict[int, int]:
    numeric_cells: List[Tuple[float, int]] = []
    for cell in row:
        text = str(cell["text"]).strip()
        if not text:
            continue
        value = _extract_int(text)
        if value is None:
            continue
        box = cell["box"]
        center = float(box.x + box.w / 2)
        if any(start <= center <= end for start, end in total_ranges):
            continue
        numeric_cells.append((center, value))

    if not numeric_cells:
        return {}

    numeric_cells.sort(key=lambda item: item[0])
    holes_sorted = sorted(hole_ranges.keys())
    mapped: Dict[int, int] = {}
    for idx, (_, value) in enumerate(numeric_cells):
        if idx >= len(holes_sorted):
            break
        mapped[holes_sorted[idx]] = value
    return mapped


def _map_cells_partial(
    row: List[Dict[str, object]],
    hole_ranges: Dict[int, Tuple[float, float]],
    total_ranges: List[Tuple[float, float]],
) -> Dict[int, int]:
    numeric_cells: List[Tuple[float, int]] = []
    for cell in row:
        text = str(cell["text"]).strip()
        if not text:
            continue
        value = _extract_int(text)
        if value is None:
            continue
        box = cell["box"]
        center = float(box.x + box.w / 2)
        if any(start <= center <= end for start, end in total_ranges):
            continue
        numeric_cells.append((center, value))

    if len(numeric_cells) < 6:
        return {}

    numeric_cells.sort(key=lambda item: item[0])
    hole_centers = {hole: (start + end) / 2.0 for hole, (start, end) in hole_ranges.items()}
    if not hole_centers:
        return {}

    mid_x = (hole_centers.get(9, 0.0) + hole_centers.get(10, hole_centers.get(9, 0.0))) / 2.0
    avg_center = sum(center for center, _ in numeric_cells) / len(numeric_cells)
    start_hole = 1 if avg_center <= mid_x else 10
    holes_sorted = sorted(hole_ranges.keys())
    if start_hole == 1:
        target_holes = [hole for hole in holes_sorted if hole <= 9]
    else:
        target_holes = [hole for hole in holes_sorted if hole >= 10]

    mapped: Dict[int, int] = {}
    for idx, (_, value) in enumerate(numeric_cells):
        if idx >= len(target_holes):
            break
        mapped[target_holes[idx]] = value
    return mapped


def _filter_values(values: Dict[int, int], min_value: int, max_value: int) -> Dict[int, int]:
    return {hole: value for hole, value in values.items() if min_value <= value <= max_value}


def _looks_like_par(values: Dict[int, int]) -> bool:
    if len(values) < 6:
        return False
    valid = [value for value in values.values() if 3 <= value <= 5]
    return len(valid) >= len(values) * 0.7


def _looks_like_yardage(values: Dict[int, int]) -> bool:
    if len(values) < 6:
        return False
    yardages = [value for value in values.values() if 70 <= value <= 700]
    return len(yardages) >= len(values) * 0.7


def _parse_rating_pairs(row_text: str) -> List[Tuple[float, int]]:
    pairs: List[Tuple[float, int]] = []
    for match in re.finditer(r"([0-9]{2,3}\.?[0-9]?)\s*/\s*([0-9]{2,3})", row_text):
        rating = float(match.group(1))
        slope = int(match.group(2))
        pairs.append((rating, slope))
    return pairs


def parse_scorecard_rows(rows_with_text: List[List[Dict[str, object]]]) -> Dict[str, object]:
    hole_ranges, total_ranges = _build_header_map(rows_with_text)
    inferred_columns = False

    if len(hole_ranges) < 6:
        hole_ranges, total_ranges = _infer_hole_ranges_geometry(rows_with_text)
        inferred_columns = True

    if not hole_ranges:
        hole_ranges = {hole: (0.0, 0.0) for hole in range(1, 19)}
        inferred_columns = True

    first_hole_start = min(start for start, _ in hole_ranges.values())

    tee_colors = ["black", "blue", "white", "gold", "red", "green", "silver", "bronze"]
    tee_aliases = {
        "black": ["black", "blk", "championship", "tips"],
        "blue": ["blue", "blu", "back"],
        "white": ["white", "wht", "middle", "mid"],
        "gold": ["gold", "gld", "w/g", "wg"],
        "red": ["red"],
        "green": ["green", "grn"],
        "silver": ["silver", "silv"],
        "bronze": ["bronze", "brnz"],
    }

    def tee_label(label: str, row_text: str) -> Optional[str]:
        base = label or row_text
        cleaned = re.sub(r"\d+(\.\d+)?\s*/\s*\d+", " ", base)
        cleaned = re.sub(r"\d+", " ", cleaned)
        cleaned = re.sub(r"[^A-Za-z\s]", " ", cleaned)
        cleaned = " ".join(cleaned.split()).strip()
        if not cleaned or len(cleaned) > 40:
            return None
        lower = _normalize_text(cleaned)
        for color in tee_colors:
            if any(alias in lower for alias in tee_aliases[color]):
                return f"{color.title()} tees"
        if "tee" in lower:
            return f"{cleaned} tees"
        return f"{cleaned} tees"

    par_by_hole = {hole: 0 for hole in hole_ranges}
    hcp_m_by_hole = {hole: 0 for hole in hole_ranges}
    hcp_w_by_hole = {hole: 0 for hole in hole_ranges}
    yardage_by_tee: Dict[str, Dict[int, int]] = {}
    rating_m: Dict[str, float] = {}
    slope_m: Dict[str, int] = {}
    rating_w: Dict[str, float] = {}
    slope_w: Dict[str, int] = {}
    par_rows: List[Dict[int, int]] = []
    hcp_m_rows: List[Dict[int, int]] = []
    hcp_w_rows: List[Dict[int, int]] = []
    yardage_rows: Dict[str, List[Dict[int, int]]] = {}

    for row_index, row in enumerate(rows_with_text):
        row_text = " ".join(str(cell["text"]).strip() for cell in row if str(cell["text"]).strip()).strip()
        if not row_text:
            continue
        lower = _normalize_text(row_text)
        label = _row_label_from_cells(row, first_hole_start)
        values_by_hole = _map_cells_to_holes(row, hole_ranges, total_ranges)
        raw_values = _row_numeric_values(row)
        if len(values_by_hole) < 6 and raw_values:
            values_by_hole = _map_cells_to_holes_nearest(row, hole_ranges, total_ranges)
        if len(values_by_hole) < 6 and raw_values:
            values_by_hole = _map_cells_by_order(row, hole_ranges, total_ranges)
        if len(values_by_hole) < 6 and raw_values:
            values_by_hole = _map_cells_partial(row, hole_ranges, total_ranges)
        par_like = raw_values and len([v for v in raw_values if 3 <= v <= 5]) >= 6
        hcp_like = raw_values and len([v for v in raw_values if 1 <= v <= 18]) >= 6
        yardage_like = raw_values and len([v for v in raw_values if 80 <= v <= 700]) >= 6

        print(
            "row_stats",
            {
                "row": row_index,
                "label": label,
                "raw_count": len(raw_values),
                "mapped_count": len(values_by_hole),
                "par_like": bool(par_like),
                "hcp_like": bool(hcp_like),
                "yardage_like": bool(yardage_like),
            },
        )

        if "par" in lower or _looks_like_par(values_by_hole) or par_like:
            par_values = _filter_values(values_by_hole, 3, 5)
            if len(par_values) < 4 and par_like:
                par_values = _filter_values(_map_cells_partial(row, hole_ranges, total_ranges), 3, 5)
            if len(par_values) < 4:
                continue
            par_rows.append(par_values)
            continue

        if "handicap" in lower or "hcp" in lower or "hdcp" in lower or hcp_like:
            hcp_values = _filter_values(values_by_hole, 1, 18)
            if len(hcp_values) < 4 and hcp_like:
                hcp_values = _filter_values(_map_cells_partial(row, hole_ranges, total_ranges), 1, 18)
            if len(hcp_values) < 4:
                continue
            if "women" in lower or "ladies" in lower:
                hcp_w_rows.append(hcp_values)
            else:
                hcp_m_rows.append(hcp_values)
            continue

        candidate_label = tee_label(label, row_text)
        yardage_values = _filter_values(values_by_hole, 80, 700)
        if not candidate_label and yardage_like and len(yardage_values) >= 6:
            candidate_label = tee_label("", row_text)
        if candidate_label and len(yardage_values) >= 6:
            yardage_rows.setdefault(candidate_label, []).append(yardage_values)

            pairs = _parse_rating_pairs(row_text)
            if pairs:
                if "women" in lower and "men" in lower and len(pairs) >= 2:
                    rating_m[candidate_label], slope_m[candidate_label] = pairs[0]
                    rating_w[candidate_label], slope_w[candidate_label] = pairs[1]
                elif "women" in lower or "ladies" in lower:
                    rating_w[candidate_label], slope_w[candidate_label] = pairs[0]
                else:
                    rating_m[candidate_label], slope_m[candidate_label] = pairs[0]

    def _merge_maps(target: Dict[int, int], maps: List[Dict[int, int]]) -> None:
        for values in maps:
            for hole, value in values.items():
                if target[hole] == 0:
                    target[hole] = value

    _merge_maps(par_by_hole, par_rows)
    _merge_maps(hcp_m_by_hole, hcp_m_rows)
    _merge_maps(hcp_w_by_hole, hcp_w_rows)

    for tee_label, maps in yardage_rows.items():
        yardage_by_tee.setdefault(tee_label, {})
        for values in maps:
            for hole, value in values.items():
                if yardage_by_tee[tee_label].get(hole, 0) == 0:
                    yardage_by_tee[tee_label][hole] = value

    return {
        "hole_columns": {hole: idx for idx, hole in enumerate(sorted(hole_ranges.keys()))},
        "par": par_by_hole,
        "handicap_men": hcp_m_by_hole,
        "handicap_women": hcp_w_by_hole,
        "yardage_by_tee": yardage_by_tee,
        "rating_men_by_tee": rating_m,
        "slope_men_by_tee": slope_m,
        "rating_women_by_tee": rating_w,
        "slope_women_by_tee": slope_w,
        "_hole_columns_inferred": inferred_columns,
        "_hole_ranges": hole_ranges,
        "_total_ranges": total_ranges,
    }


def build_holes_from_grid(parsed: Dict[str, object]) -> List[ScorecardHole]:
    holes: List[ScorecardHole] = []
    hole_columns = parsed.get("hole_columns", {})
    par = parsed.get("par", {})
    handicap_men = parsed.get("handicap_men", {})
    handicap_women = parsed.get("handicap_women", {})
    yardage_by_tee = parsed.get("yardage_by_tee", {})

    for hole in sorted(hole_columns.keys()):
        yardages: Dict[str, Optional[int]] = {}
        for tee, values in yardage_by_tee.items():
            yardages[tee] = values.get(hole)
        holes.append(
            ScorecardHole(
                hole=hole,
                par=par.get(hole) or None,
                handicap_men=handicap_men.get(hole) or None,
                handicap_women=handicap_women.get(hole) or None,
                yardages_by_tee=yardages,
            )
        )
    return holes


def _normalize_list(values: Optional[List[Optional[int]]], length: int = 18) -> List[Optional[int]]:
    if not values:
        return [None] * length
    normalized = list(values[:length])
    if len(normalized) < length:
        normalized.extend([None] * (length - len(normalized)))
    return normalized


def parse_scorecard_image(image_bytes: bytes, debug: bool = False, mode: str = "course") -> Dict[str, object]:
    logger = logging.getLogger("app.pipeline")
    logger.info("parse_scorecard_image start mode=%s bytes=%s debug=%s", mode, len(image_bytes), debug)
    parsed = parse_scorecard_with_gemini(image_bytes, mode=mode)
    if "error" in parsed:
        logger.warning("gemini_error mode=%s error=%s", mode, parsed.get("error"))
        return {
            "confidence": 0.0,
            "holes": [],
            "totals": ScorecardTotals().dict(by_alias=True),
            "flags": [parsed.get("error", "gemini_failed")],
            "metadata": ScorecardMetadata().dict(),
        }

    guard_flags: List[str] = []

    def _ensure_list(value: object, name: str) -> List[object]:
        if value is None:
            return []
        if not isinstance(value, list):
            guard_flags.append(f"gemini_schema_{name}")
            return []
        return value

    par_raw = _ensure_list(parsed.get("par"), "par")
    handicap_men_raw = _ensure_list(parsed.get("handicapMen"), "handicapMen")
    handicap_women_raw = _ensure_list(parsed.get("handicapWomen"), "handicapWomen")
    tee_boxes = _ensure_list(parsed.get("teeBoxes"), "teeBoxes")

    par = _normalize_list([value if isinstance(value, (int, type(None))) else None for value in par_raw])
    handicap_men = _normalize_list(
        [value if isinstance(value, (int, type(None))) else None for value in handicap_men_raw]
    )
    handicap_women = _normalize_list(
        [value if isinstance(value, (int, type(None))) else None for value in handicap_women_raw]
    )

    yardage_by_tee: Dict[str, List[Optional[int]]] = {}
    rating_m: Dict[str, Optional[float]] = {}
    slope_m: Dict[str, Optional[int]] = {}
    rating_w: Dict[str, Optional[float]] = {}
    slope_w: Dict[str, Optional[int]] = {}

    for tee in tee_boxes:
        if not isinstance(tee, dict):
            guard_flags.append("gemini_schema_tee_entry")
            continue
        name = str(tee.get("name") or "").strip()
        if not name:
            name = "Unknown tee"
        yardage_by_tee[name] = _normalize_list(tee.get("yardages"))
        rating_m[name] = tee.get("ratingMen")
        slope_m[name] = tee.get("slopeMen")
        rating_w[name] = tee.get("ratingWomen")
        slope_w[name] = tee.get("slopeWomen")

    holes: List[ScorecardHole] = []
    for idx in range(18):
        yardages = {tee: yards[idx] for tee, yards in yardage_by_tee.items()}
        holes.append(
            ScorecardHole(
                hole=idx + 1,
                par=par[idx],
                handicap_men=handicap_men[idx],
                handicap_women=handicap_women[idx],
                yardages_by_tee=yardages,
            )
        )

    populated = 0
    expected = 18 * (1 + 1 + max(1, len(yardage_by_tee)))
    for idx in range(18):
        if par[idx] is not None:
            populated += 1
        if handicap_men[idx] is not None:
            populated += 1
        for tee in yardage_by_tee.values():
            if tee[idx] is not None:
                populated += 1
    confidence = round(populated / expected, 4) if expected else 0.0

    flags: List[str] = []
    if not any(par):
        flags.append("par_missing")
    if not yardage_by_tee:
        flags.append("yardages_missing")
    flags.extend(guard_flags)

    def _clean_numeric_map(values: Dict[str, Optional[float]]) -> Dict[str, float]:
        return {key: value for key, value in values.items() if value is not None}

    metadata = ScorecardMetadata(
        tee_boxes=sorted(yardage_by_tee.keys()),
        rating_men_by_tee=_clean_numeric_map(rating_m),
        slope_men_by_tee=_clean_numeric_map(slope_m),
        rating_women_by_tee=_clean_numeric_map(rating_w),
        slope_women_by_tee=_clean_numeric_map(slope_w),
    )

    player_payload = parsed.get("player") if isinstance(parsed.get("player"), dict) else None
    def _player_has_values(payload: Optional[Dict[str, object]]) -> bool:
        if not payload:
            return False
        holes_payload = payload.get("holes") if isinstance(payload.get("holes"), list) else []
        for entry in holes_payload:
            if not isinstance(entry, dict):
                continue
            if any(entry.get(key) is not None for key in ("score", "putts", "fairway", "green", "upDown", "penalties")):
                return True
        return False

    if mode == "completed" and not _player_has_values(player_payload):
        player_fallback = parse_scorecard_with_gemini(image_bytes, mode="player")
        if isinstance(player_fallback, dict):
            fallback_player = player_fallback.get("player") if isinstance(player_fallback.get("player"), dict) else None
            if fallback_player:
                player_payload = fallback_player
    player_data = None
    if player_payload:
        holes_payload = player_payload.get("holes") if isinstance(player_payload.get("holes"), list) else []
        player_holes: List[PlayerHole] = []
        for hole_index in range(1, 19):
            entry = next((h for h in holes_payload if isinstance(h, dict) and h.get("hole") == hole_index), None)
            par_value = None
            if len(par) >= hole_index:
                par_value = par[hole_index - 1]
            if entry is None:
                player_holes.append(PlayerHole(hole=hole_index))
                continue
            fairway_value = entry.get("fairway")
            if par_value == 3:
                # FIR does not apply on par 3s, always null it out
                fairway_value = None
            player_holes.append(
                PlayerHole(
                    hole=hole_index,
                    score=entry.get("score"),
                    putts=entry.get("putts"),
                    fairway=fairway_value,
                    green=entry.get("green"),
                    up_down=entry.get("upDown"),
                    penalties=entry.get("penalties"),
                )
            )
        player_data = ScorecardPlayer(
            name=player_payload.get("name"),
            date=player_payload.get("date"),
            holes=player_holes,
        )

    par_count = sum(1 for value in par if value is not None)
    yardage_tees = list(yardage_by_tee.keys())
    player_scores = 0
    if player_payload and isinstance(player_payload, dict):
        holes_payload = player_payload.get("holes") if isinstance(player_payload.get("holes"), list) else []
        for entry in holes_payload:
            if isinstance(entry, dict) and entry.get("score") is not None:
                player_scores += 1
    logger.info(
        "parse_scorecard_image summary mode=%s par_filled=%s yardage_tees=%s player_scores=%s confidence=%s flags=%s",
        mode,
        par_count,
        yardage_tees,
        player_scores,
        confidence,
        flags,
    )

    return {
        "confidence": confidence,
        "holes": [hole.dict() for hole in holes],
        "totals": ScorecardTotals().dict(by_alias=True),
        "flags": flags,
        "metadata": metadata.dict(),
        "player": player_data.dict() if player_data else None,
        "debug_image_base64": None,
    }
