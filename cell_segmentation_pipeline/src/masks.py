"""Mask generation, expansion, and cell-table extraction."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels, watershed
from skimage.morphology import disk, white_tophat


# ── Mask expansion ────────────────────────────────────────────────────────────

def expand_mask(mask: np.ndarray, expand_px: int = 5, min_area_px: int = 50) -> np.ndarray:
    """Expand nuclear mask by expand_px pixels without overlap.

    Uses skimage.segmentation.expand_labels which performs a watershed-like
    expansion and guarantees no two cells overlap in the result.

    Args:
        mask:       Integer label mask (0 = background).
        expand_px:  Dilation distance in pixels.
        min_area_px: Remove cells smaller than this after expansion.

    Returns:
        Expanded label mask (same shape, same dtype).
    """
    if expand_px <= 0:
        return mask.copy()

    expanded = expand_labels(mask, distance=expand_px)

    # Remove cells that are too small even after expansion
    if min_area_px > 0:
        props = regionprops_table(expanded, properties=["label", "area"])
        small = set(
            int(lbl)
            for lbl, area in zip(props["label"], props["area"])
            if area < min_area_px
        )
        if small:
            # Use lookup table instead of np.isin to avoid a NumPy off-by-one
            # bug in in1d that triggers on large arrays with many test elements.
            lookup = np.zeros(int(expanded.max()) + 1, dtype=bool)
            for lbl in small:
                lookup[lbl] = True
            rm = lookup[expanded]
            expanded = expanded.copy()
            expanded[rm] = 0

    return expanded


def expand_mask_membrane_guided(
    nuclear_mask: np.ndarray,
    membrane_raw: np.ndarray,
    max_expand_px: int = 10,
    tophat_radius: int = 20,
    min_area_px: int = 50,
) -> np.ndarray:
    """Expand nuclear mask using Membrane channel as a watershed barrier.

    Instead of simple Voronoi expansion, this uses the Membrane signal
    as a topographic map: high Membrane intensity = cell boundary (barrier),
    low intensity = cell interior (flow-through).

    Process
    -------
    1. TopHat-filter Membrane channel to remove background.
    2. Invert signal: high_membrane → deep valley (watershed flows away).
    3. Run watershed from nuclear seeds, bounded by max_expand_px dilation.
    4. Remove cells below min_area_px.

    Args:
        nuclear_mask:  Integer label mask from cpsam (0 = background).
        membrane_raw:  Raw Membrane channel (H, W) uint16.
        max_expand_px: Hard cap on expansion distance (prevents runaway growth).
        tophat_radius: TopHat filter radius for Membrane preprocessing.
        min_area_px:   Minimum cell area after expansion.

    Returns:
        Expanded cell label mask (same shape as nuclear_mask).
    """
    # ── Preprocess Membrane channel ───────────────────────────────────────────
    mem_f = membrane_raw.astype(np.float32)
    th_mem = white_tophat(mem_f, disk(tophat_radius))

    # Normalise to [0, 1]
    mx = float(th_mem.max())
    if mx > 0:
        th_mem_n = th_mem / mx
    else:
        th_mem_n = th_mem

    # Invert: high Membrane (boundary) → high value → watershed stops here
    # watershed minimises the "elevation" surface, so barriers need to be high
    barrier = th_mem_n  # high = hard to cross

    # ── Build expansion region (max distance from any nucleus) ───────────────
    from scipy.ndimage import distance_transform_edt
    fg = nuclear_mask > 0
    dist_from_nuc = distance_transform_edt(~fg)
    expansion_mask = dist_from_nuc <= max_expand_px  # only expand within radius

    # ── Watershed from nuclear seeds ─────────────────────────────────────────
    # watershed image: low inside cells, high at boundaries
    # We use the barrier as the elevation surface
    cell_mask = watershed(
        image=barrier,
        markers=nuclear_mask,
        mask=expansion_mask,
        watershed_line=False,
        compactness=0.01,  # small compactness keeps cells compact
    )

    # ── Remove tiny cells ─────────────────────────────────────────────────────
    if min_area_px > 0:
        props = regionprops_table(cell_mask, properties=["label", "area"])
        small = {
            int(lbl) for lbl, area in zip(props["label"], props["area"])
            if area < min_area_px
        }
        if small:
            rm = np.isin(cell_mask, list(small))
            cell_mask = cell_mask.copy()
            cell_mask[rm] = 0

    return cell_mask.astype(nuclear_mask.dtype)


# ── Cell table ────────────────────────────────────────────────────────────────

def mask_to_cell_table(
    mask: np.ndarray,
    raw_image: np.ndarray,
    fov_meta: dict | None = None,
) -> pd.DataFrame:
    """Extract per-cell morphology metrics from a label mask.

    Output columns match the existing outputs/pilot_4fov/cell_tables/*.csv:
        label, area, centroid_y, centroid_x, bbox_y0, bbox_x0, bbox_y1, bbox_x1,
        mean_intensity, max_intensity, eccentricity, solidity,
        FOV, condition, region, sample_id, gender, age, est_diam_px

    Args:
        mask:      Integer label mask (0 = background).
        raw_image: 2D grayscale image (used for intensity metrics).
        fov_meta:  Dict from get_fov_meta(); keys: fov, condition, region,
                   sample_id, gender, age.

    Returns:
        DataFrame with one row per cell.
    """
    if fov_meta is None:
        fov_meta = {}

    props = regionprops_table(
        mask,
        intensity_image=raw_image.astype(np.float32),
        properties=[
            "label",
            "area",
            "centroid",
            "bbox",
            "mean_intensity",
            "max_intensity",
            "eccentricity",
            "solidity",
        ],
    )

    df = pd.DataFrame(props).rename(columns={
        "centroid-0": "centroid_y",
        "centroid-1": "centroid_x",
        "bbox-0": "bbox_y0",
        "bbox-1": "bbox_x0",
        "bbox-2": "bbox_y1",
        "bbox-3": "bbox_x1",
    })

    # Estimated diameter assuming circular nucleus
    df["est_diam_px"] = 2.0 * np.sqrt(df["area"] / math.pi)

    # Attach biological metadata
    df["FOV"]       = fov_meta.get("fov", "")
    df["condition"] = fov_meta.get("condition", "")
    df["region"]    = fov_meta.get("region", "")
    df["sample_id"] = fov_meta.get("sample_id", "")
    df["gender"]    = fov_meta.get("gender", "")
    df["age"]       = fov_meta.get("age", "")

    # Reorder to match reference schema
    col_order = [
        "label", "area", "centroid_y", "centroid_x",
        "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1",
        "mean_intensity", "max_intensity",
        "eccentricity", "solidity",
        "FOV", "condition", "region", "sample_id", "gender", "age",
        "est_diam_px",
    ]
    return df[[c for c in col_order if c in df.columns]]


# ── Coloured mask PNG ─────────────────────────────────────────────────────────

def save_colored_mask_png(mask: np.ndarray, path: str | Path) -> None:
    """Render label mask as a random-colour RGB image and save as PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_labels = int(mask.max())
    rng = np.random.default_rng(seed=42)
    # colour table: 0 = black background, 1..N = random colours
    colours = np.zeros((n_labels + 1, 3), dtype=np.float32)
    colours[1:] = rng.random((n_labels, 3))

    rgb = colours[mask]          # (H, W, 3)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(f"{n_labels} cells", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
