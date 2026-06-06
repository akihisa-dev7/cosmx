"""Extract per-cell image crops from full-resolution images."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .image_loader import norm_uint8
from .overlay_utils import compose_overlay


def get_cell_crop(
    raw: np.ndarray,
    mask: np.ndarray,
    cell_id: int,
    padding: int = 25,
) -> dict | None:
    """Extract a padded crop of a single cell.

    Args:
        raw:     2D grayscale uint16 image (full resolution).
        mask:    2D int32 label mask (full resolution).
        cell_id: Target cell label (1-indexed).
        padding: Extra pixels around the bounding box.

    Returns:
        Dict with keys: cell_id, raw_crop (uint8), overlay_crop (RGB uint8),
        bbox (y0,x0,y1,x1), area, centroid_y, centroid_x, est_diam_px.
        None if cell_id not found.
    """
    ys, xs = np.where(mask == cell_id)
    if len(ys) == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    # Add padding, clamp to image
    H, W = mask.shape
    py0 = max(0, y0 - padding)
    px0 = max(0, x0 - padding)
    py1 = min(H, y1 + padding + 1)
    px1 = min(W, x1 + padding + 1)

    raw_crop  = norm_uint8(raw[py0:py1, px0:px1])
    mask_crop = mask[py0:py1, px0:px1].copy()
    # Dim other cells in the crop
    other = (mask_crop != 0) & (mask_crop != cell_id)
    mask_highlight = mask_crop.copy()
    mask_highlight[other] = 0

    ov_crop = compose_overlay(raw_crop, mask_highlight, alpha=0.5)

    area = int(len(ys))
    cy   = float(ys.mean())
    cx   = float(xs.mean())

    return {
        "cell_id":    cell_id,
        "raw_crop":   raw_crop,
        "overlay_crop": ov_crop,
        "bbox":       (y0, x0, y1, x1),
        "area":       area,
        "centroid_y": cy,
        "centroid_x": cx,
        "est_diam_px": 2 * np.sqrt(area / np.pi),
    }


def get_random_cells(
    raw: np.ndarray,
    mask: np.ndarray,
    n: int = 20,
    seed: int | None = None,
    min_area: int = 50,
    padding: int = 25,
) -> list[dict]:
    """Return crops of n random cells from the mask.

    Filters out cells smaller than min_area (likely segmentation artefacts).
    """
    unique = np.unique(mask)
    unique = unique[unique > 0]  # exclude background

    # Filter by area
    if min_area > 0:
        areas = np.array([int((mask == lbl).sum()) for lbl in unique])
        unique = unique[areas >= min_area]

    if len(unique) == 0:
        return []

    rng = np.random.default_rng(seed)
    chosen = rng.choice(unique, size=min(n, len(unique)), replace=False)

    crops = []
    for cid in chosen:
        crop = get_cell_crop(raw, mask, int(cid), padding=padding)
        if crop is not None:
            crops.append(crop)
    return crops


def get_cells_by_ids(
    raw: np.ndarray,
    mask: np.ndarray,
    cell_ids: list[int],
    padding: int = 25,
) -> list[dict]:
    """Return crops for a specific list of cell IDs."""
    return [
        c for cid in cell_ids
        if (c := get_cell_crop(raw, mask, cid, padding)) is not None
    ]


def find_cell_at_xy(mask: np.ndarray, x: int, y: int) -> int | None:
    """Return cell ID at pixel coordinates (x, y), or None if background."""
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        val = int(mask[y, x])
        return val if val > 0 else None
    return None
