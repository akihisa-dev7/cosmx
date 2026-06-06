"""Spatial registration utilities for alignment mismatch verification."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as nd_shift
from skimage.morphology import binary_dilation, disk

from .image_loader import norm_uint8, downsample_img, downsample_mask


# ── Spatial transforms ────────────────────────────────────────────────────────

def apply_spatial_transform(
    mask: np.ndarray,
    dx: int = 0,
    dy: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    swap_xy: bool = False,
) -> np.ndarray:
    """Apply flip / axis-swap / shift to a label mask.

    Order: swap_xy → flip_x → flip_y → shift(dy, dx).
    All operations preserve label values (nearest-neighbour, cval=0).
    """
    m = mask.copy()
    if swap_xy:
        m = m.T
    if flip_x:
        m = np.fliplr(m)
    if flip_y:
        m = np.flipud(m)
    if dx != 0 or dy != 0:
        m = nd_shift(m.astype(np.float32), shift=(dy, dx),
                     order=0, mode="constant", cval=0).astype(mask.dtype)
    return m


# ── Boundary extraction ───────────────────────────────────────────────────────

def _boundary(binary: np.ndarray, dilate: int = 1) -> np.ndarray:
    from skimage.morphology import binary_erosion
    bm = binary.astype(bool)
    bd = bm & ~binary_erosion(bm)
    if dilate > 0:
        bd = binary_dilation(bd, disk(dilate))
    return bd


# ── Alignment debug RGB overlay ───────────────────────────────────────────────

def build_alignment_overlay(
    gray: np.ndarray,
    cosmx_nuc_mask: np.ndarray,
    cosmx_cell_mask: np.ndarray,
    custom_mask: np.ndarray,
    alpha: float = 0.7,
    dilate: int = 1,
    max_dim: int = 1024,
) -> np.ndarray:
    """Composite overlay for registration debugging.

    Colours
    -------
    Blue   : CosMx nucleus boundary
    Green  : CosMx cell boundary
    Red    : custom segmentation boundary
    Yellow : overlap between custom boundary and any CosMx boundary
    """
    # Downsample everything together
    scale = max_dim / max(gray.shape)
    if scale < 1.0:
        from PIL import Image
        new_h = int(gray.shape[0] * scale)
        new_w = int(gray.shape[1] * scale)
        size  = (new_w, new_h)
        gray          = np.array(Image.fromarray(norm_uint8(gray)).resize(size, Image.LANCZOS))
        cosmx_nuc_mask  = np.array(Image.fromarray(cosmx_nuc_mask.astype(np.uint8)).resize(size, Image.NEAREST))
        cosmx_cell_mask = np.array(Image.fromarray(cosmx_cell_mask.astype(np.uint8)).resize(size, Image.NEAREST))
        custom_mask   = np.array(Image.fromarray((custom_mask > 0).astype(np.uint8)).resize(size, Image.NEAREST))
    else:
        gray = norm_uint8(gray)

    bnd_nuc    = _boundary(cosmx_nuc_mask  > 0, dilate)
    bnd_cell   = _boundary(cosmx_cell_mask > 0, dilate)
    bnd_custom = _boundary(custom_mask     > 0, dilate)

    bnd_cosmx_any = bnd_nuc | bnd_cell
    overlap = bnd_custom & bnd_cosmx_any

    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    out  = base.copy()

    C = {
        "nuc":    np.array([70,  130, 230], dtype=np.float32),  # blue
        "cell":   np.array([50,  200,  80], dtype=np.float32),  # green
        "custom": np.array([230,  57,  70], dtype=np.float32),  # red
        "overlap":np.array([255, 215,   0], dtype=np.float32),  # yellow
    }

    for px_mask, colour in [
        (bnd_nuc,    C["nuc"]),
        (bnd_cell,   C["cell"]),
        (bnd_custom, C["custom"]),
        (overlap,    C["overlap"]),
    ]:
        if px_mask.any():
            out[px_mask] = (1 - alpha) * out[px_mask] + alpha * colour

    return np.clip(out, 0, 255).astype(np.uint8)


# ── Grid search for best offset ───────────────────────────────────────────────

def grid_search_best_offset(
    custom_mask: np.ndarray,
    ref_mask: np.ndarray,
    range_px: int = 30,
    downsample: int = 4,
    metric: str = "iou",
    progress_cb=None,
) -> dict:
    """Brute-force search for (dx, dy) that maximises IoU/Dice/BoundaryF1.

    Parameters
    ----------
    custom_mask  : int32 label mask to shift
    ref_mask     : reference binary mask (CosMx nucleus)
    range_px     : search range in original-resolution pixels
    downsample   : internal downscale factor for speed
    metric       : "iou" | "dice" | "boundary_f1"
    progress_cb  : optional callback(frac: float) for Streamlit progress bar

    Returns dict with best_dx, best_dy, best_score, score_grid (ndarray).
    """
    from PIL import Image

    h, w = custom_mask.shape
    ds   = downsample
    nh, nw = h // ds, w // ds

    def _ds(arr, nearest=True):
        mode = Image.NEAREST if nearest else Image.LANCZOS
        return np.array(Image.fromarray(arr).resize((nw, nh), mode))

    custom_bool = _ds((custom_mask > 0).astype(np.uint8)).astype(bool)
    ref_bool    = _ds((ref_mask    > 0).astype(np.uint8)).astype(bool)

    if metric == "boundary_f1":
        from skimage.morphology import binary_erosion, binary_dilation
        ref_bnd = ref_bool & ~binary_erosion(ref_bool)
        ref_dil = binary_dilation(ref_bnd, disk(max(1, 3 // ds)))
    else:
        ref_bnd = ref_dil = None

    r    = max(1, range_px // ds)
    steps = range(-r, r + 1)
    total = len(steps) ** 2
    grid  = np.full((len(steps), len(steps)), np.nan)
    best_score = -1.0
    best_dx = best_dy = 0

    for i, dy in enumerate(steps):
        for j, dx in enumerate(steps):
            shifted = nd_shift(custom_bool.astype(np.uint8), (dy, dx),
                               order=0, mode="constant", cval=0).astype(bool)

            if metric == "iou":
                inter = int((shifted & ref_bool).sum())
                union = int((shifted | ref_bool).sum())
                score = inter / union if union > 0 else 0.0

            elif metric == "dice":
                inter = int((shifted & ref_bool).sum())
                denom = int(shifted.sum()) + int(ref_bool.sum())
                score = 2 * inter / denom if denom > 0 else 0.0

            else:  # boundary_f1
                from skimage.morphology import binary_erosion
                bnd_sh = shifted & ~binary_erosion(shifted)
                bnd_dil_sh = binary_dilation(bnd_sh, disk(max(1, 3 // ds)))
                tp_p = int((bnd_sh  & ref_dil).sum())
                tp_r = int((ref_bnd & bnd_dil_sh).sum())
                prec = tp_p / int(bnd_sh.sum())  if bnd_sh.any()  else 0.0
                rec  = tp_r / int(ref_bnd.sum()) if ref_bnd.any() else 0.0
                score = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

            grid[i, j] = score
            if score > best_score:
                best_score = score
                best_dx = dx * ds
                best_dy = dy * ds

        if progress_cb is not None:
            progress_cb((i + 1) / len(steps))

    return {
        "best_dx":    best_dx,
        "best_dy":    best_dy,
        "best_score": best_score,
        "metric":     metric,
        "score_grid": grid,
        "steps":      [s * ds for s in steps],
    }


# ── Coordinate system log ─────────────────────────────────────────────────────

def compute_coordinate_log(
    fov_id: str,
    custom_mask: np.ndarray,
    cosmx_nuc_mask: np.ndarray,
    cosmx_cell_mask: np.ndarray,
    dapi_shape: tuple | None = None,
    matched_pairs_df=None,
) -> dict:
    """Compute coordinate system diagnostics and centroid displacement stats."""
    from skimage.measure import regionprops_table

    def _centroids(mask):
        if mask.max() == 0:
            return np.array([]), np.array([])
        props = regionprops_table(mask.astype(np.int32),
                                  properties=["label", "centroid"])
        return props["centroid-1"], props["centroid-0"]  # x, y

    cust_x, cust_y = _centroids(custom_mask)
    nuc_x,  nuc_y  = _centroids((cosmx_nuc_mask > 0).astype(np.uint8)
                                 * np.arange(1, cosmx_nuc_mask.size + 1,
                                             dtype=np.int32).reshape(cosmx_nuc_mask.shape)
                                 if False else cosmx_nuc_mask.astype(np.int32))

    # CosMx cell labels → use CellLabels for centroid (not compartment)
    cell_x, cell_y = _centroids(cosmx_cell_mask.astype(np.int32))

    log = {
        "fov_id":               fov_id,
        "dapi_shape":           str(dapi_shape) if dapi_shape else "N/A",
        "custom_mask_shape":    str(custom_mask.shape),
        "cosmx_nuc_shape":      str(cosmx_nuc_mask.shape),
        "cosmx_cell_shape":     str(cosmx_cell_mask.shape),
        "shapes_match":         custom_mask.shape == cosmx_nuc_mask.shape,
        # custom centroids
        "custom_n_cells":       int(custom_mask.max()),
        "custom_mean_x":        float(cust_x.mean()) if len(cust_x) > 0 else float("nan"),
        "custom_mean_y":        float(cust_y.mean()) if len(cust_y) > 0 else float("nan"),
        "custom_x_range":       f"[{cust_x.min():.0f}, {cust_x.max():.0f}]" if len(cust_x) > 0 else "N/A",
        "custom_y_range":       f"[{cust_y.min():.0f}, {cust_y.max():.0f}]" if len(cust_y) > 0 else "N/A",
        # cosmx cell centroids
        "cosmx_n_cells":        int(cosmx_cell_mask.max()) if cosmx_cell_mask.dtype != np.uint8 else "N/A",
        "cosmx_mean_x":         float(cell_x.mean()) if len(cell_x) > 0 else float("nan"),
        "cosmx_mean_y":         float(cell_y.mean()) if len(cell_y) > 0 else float("nan"),
        # coordinate system checks
        "index_convention":     "image[row=y, col=x]  — row=0 is top",
        "centroid_convention":  "centroid-0=y(row), centroid-1=x(col)",
        "zero_indexed":         True,
    }

    # Displacement stats from matched pairs if available
    if matched_pairs_df is not None and len(matched_pairs_df) > 0:
        m = matched_pairs_df
        if "dx" in m.columns and "dy" in m.columns:
            matched = m[m.get("matched", True)]
            log["matched_n"]     = int(len(matched))
            log["mean_dx"]       = float(matched["dx"].mean())
            log["mean_dy"]       = float(matched["dy"].mean())
            log["median_dx"]     = float(matched["dx"].median())
            log["median_dy"]     = float(matched["dy"].median())
            log["std_dx"]        = float(matched["dx"].std())
            log["std_dy"]        = float(matched["dy"].std())

    return log


# ── Centroid displacement DataFrame ───────────────────────────────────────────

def compute_centroid_displacement(
    custom_mask: np.ndarray,
    cosmx_cell_labels: np.ndarray,
    max_dist_px: float = 50.0,
) -> "pd.DataFrame":
    """Return per-matched-pair displacement (dx, dy) between centroids."""
    import pandas as pd
    from skimage.measure import regionprops_table
    from scipy.spatial import KDTree

    def _cents(mask):
        if mask.max() == 0:
            return pd.DataFrame(columns=["label", "cx", "cy"])
        p = regionprops_table(mask.astype(np.int32), properties=["label", "centroid"])
        return pd.DataFrame({"label": p["label"],
                              "cx": p["centroid-1"],
                              "cy": p["centroid-0"]})

    cust  = _cents(custom_mask)
    cosmx = _cents(cosmx_cell_labels.astype(np.int32))

    if len(cust) == 0 or len(cosmx) == 0:
        return pd.DataFrame()

    tree = KDTree(cosmx[["cx", "cy"]].values)
    dists, idxs = tree.query(cust[["cx", "cy"]].values)

    rows = []
    for i, (d, idx) in enumerate(zip(dists, idxs)):
        if d > max_dist_px:
            continue
        rows.append({
            "custom_label":  int(cust["label"].iloc[i]),
            "cosmx_label":   int(cosmx["label"].iloc[idx]),
            "custom_cx":     float(cust["cx"].iloc[i]),
            "custom_cy":     float(cust["cy"].iloc[i]),
            "cosmx_cx":      float(cosmx["cx"].iloc[idx]),
            "cosmx_cy":      float(cosmx["cy"].iloc[idx]),
            "dx":            float(cust["cx"].iloc[i] - cosmx["cx"].iloc[idx]),
            "dy":            float(cust["cy"].iloc[i] - cosmx["cy"].iloc[idx]),
            "dist":          float(d),
        })

    return pd.DataFrame(rows)
