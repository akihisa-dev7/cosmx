"""DAPI vs Mask Segmentation Accuracy QC module.

Pure Python / numpy / scipy / scikit-image.  No Streamlit dependency.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects


# ── Colour palette ─────────────────────────────────────────────────────────────
_COLOR_MATCHED  = np.array([0, 200, 0],   dtype=np.uint8)   # #00C800 green
_COLOR_MISSED   = np.array([230, 57, 70],  dtype=np.uint8)   # #E63946 red
_COLOR_LOW_DAPI = np.array([255, 215, 0],  dtype=np.uint8)   # #FFD700 yellow
_COLOR_EXTRA    = np.array([69, 117, 180], dtype=np.uint8)   # #4575B4 blue


# ── Helpers ────────────────────────────────────────────────────────────────────

def _norm8(img: np.ndarray) -> np.ndarray:
    mn, mx = float(img.min()), float(img.max())
    if mx == mn:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def _downsample(img: np.ndarray, max_dim: int) -> np.ndarray:
    from PIL import Image as _PIL
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    if img.ndim == 2:
        pil = _PIL.fromarray(img if img.dtype == np.uint8 else _norm8(img))
        return np.array(pil.resize((nw, nh), _PIL.LANCZOS))
    else:
        pil = _PIL.fromarray(img)
        return np.array(pil.resize((nw, nh), _PIL.LANCZOS))


def _downsample_mask(mask: np.ndarray, max_dim: int) -> np.ndarray:
    from PIL import Image as _PIL
    h, w = mask.shape
    if max(h, w) <= max_dim:
        return mask
    scale = max_dim / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    pil = _PIL.fromarray(mask.astype(np.int32), mode="I")
    return np.array(pil.resize((nw, nh), _PIL.NEAREST))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DAPI coverage
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dapi_coverage(
    raw: np.ndarray,
    mask: np.ndarray,
    sigma_multiplier: float = 1.5,
) -> dict:
    """Compute mean DAPI intensity per mask cell.

    Parameters
    ----------
    raw:  2D grayscale image (any dtype).
    mask: 2D integer label mask (0 = background).

    Returns
    -------
    dict with keys:
        cell_count, mean_dapi_per_cell, median_dapi_per_cell,
        background_mean, background_std,
        signal_to_background_ratio,
        per_cell_df (DataFrame: label, mean_intensity, area, is_low_dapi)
    """
    raw_f = raw.astype(np.float64)

    # Background stats (mask == 0)
    bg_pixels = raw_f[mask == 0]
    if len(bg_pixels) == 0:
        bg_mean = 0.0
        bg_std  = 1.0
    else:
        # Use median/IQR for robustness against outlier bright spots in background
        bg_mean = float(np.median(bg_pixels))
        q25, q75 = float(np.percentile(bg_pixels, 25)), float(np.percentile(bg_pixels, 75))
        bg_std  = (q75 - q25) / 1.35  # IQR → sigma equivalent (normal distribution)
        if bg_std <= 0:
            bg_std = float(bg_pixels.std()) or 1.0

    low_dapi_threshold = bg_mean + sigma_multiplier * bg_std

    labels = np.unique(mask)
    labels = labels[labels > 0]

    records = []
    for lbl in labels:
        px = raw_f[mask == lbl]
        area = int(len(px))
        if area == 0:
            continue
        mean_int = float(px.mean())
        is_low   = bool(mean_int < low_dapi_threshold)
        records.append({
            "label":          int(lbl),
            "mean_intensity": mean_int,
            "area":           area,
            "is_low_dapi":    is_low,
        })

    per_cell_df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["label", "mean_intensity", "area", "is_low_dapi"]
    )

    cell_count = len(per_cell_df)
    if cell_count > 0:
        mean_dapi   = float(per_cell_df["mean_intensity"].mean())
        median_dapi = float(per_cell_df["mean_intensity"].median())
    else:
        mean_dapi = median_dapi = float("nan")

    s2b = mean_dapi / bg_mean if bg_mean > 0 else float("nan")

    return {
        "cell_count":               cell_count,
        "mean_dapi_per_cell":       mean_dapi,
        "median_dapi_per_cell":     median_dapi,
        "background_mean":          bg_mean,
        "background_std":           bg_std,
        "signal_to_background_ratio": s2b,
        "per_cell_df":              per_cell_df,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DAPI object detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_dapi_objects(
    enhanced: np.ndarray,
    min_area_px: int = 50,
    gaussian_sigma: float = 1.0,
) -> np.ndarray:
    """Threshold enhanced DAPI image with Otsu and label connected components.

    Parameters
    ----------
    enhanced:       2D enhanced DAPI image.
    min_area_px:    Minimum object size in pixels.
    gaussian_sigma: Sigma for Gaussian pre-smoothing.

    Returns
    -------
    Integer label array (0 = background).
    """
    smoothed = gaussian(enhanced.astype(np.float64), sigma=gaussian_sigma)
    thresh   = threshold_otsu(smoothed)
    binary   = smoothed > thresh
    binary   = remove_small_objects(binary, max_size=min_area_px)
    labeled, _ = ndi.label(binary)
    return labeled.astype(np.int32)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Match DAPI objects to mask cells
# ═══════════════════════════════════════════════════════════════════════════════

def match_objects_to_masks(
    dapi_labels: np.ndarray,
    mask: np.ndarray,
    iou_threshold: float = 0.3,
    overlap_threshold: float = 0.3,
) -> dict:
    """Match DAPI detected objects to segmentation mask cells by IoU.

    Parameters
    ----------
    dapi_labels:   Integer label array from detect_dapi_objects.
    mask:          Segmentation mask (integer labels, 0 = background).
    iou_threshold: Minimum IoU for a match.

    Returns
    -------
    dict with keys:
        dapi_object_count, mask_cell_count,
        matched_count, missed_dapi_count, extra_mask_count,
        estimated_recall, estimated_precision,
        per_dapi_df, per_mask_df
    """
    dapi_uniq = np.unique(dapi_labels)
    dapi_uniq = dapi_uniq[dapi_uniq > 0]
    mask_uniq = np.unique(mask)
    mask_uniq = mask_uniq[mask_uniq > 0]

    if len(dapi_uniq) > 2000:
        dapi_uniq = dapi_uniq[:2000]

    n_dapi = len(dapi_uniq)
    n_mask = len(mask_uniq)

    # ── Pass 1: DAPI→mask matching (for standard IoU recall/precision) ────────
    per_dapi_records: list[dict] = []
    matched_mask_labels: set[int] = set()

    for d_lbl in dapi_uniq:
        d_px   = dapi_labels == d_lbl
        d_area = int(d_px.sum())
        overlap_lbls = np.unique(mask[d_px])
        overlap_lbls = overlap_lbls[overlap_lbls > 0]

        best_iou, best_m = 0.0, 0
        for m_lbl in overlap_lbls:
            m_px     = mask == m_lbl
            intersec = int((d_px & m_px).sum())
            union    = d_area + int(m_px.sum()) - intersec
            iou      = intersec / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_m = iou, int(m_lbl)

        matched = best_iou >= iou_threshold
        per_dapi_records.append({
            "dapi_label":         int(d_lbl),
            "matched_mask_label": best_m if matched else 0,
            "iou":                best_iou,
            "matched":            matched,
        })
        if matched:
            matched_mask_labels.add(best_m)

    per_dapi_df = pd.DataFrame(per_dapi_records) if per_dapi_records else pd.DataFrame(
        columns=["dapi_label", "matched_mask_label", "iou", "matched"]
    )

    # ── Pass 2: mask→DAPI matching (overlap coefficient from mask perspective)
    # "coverage": what fraction of each mask cell area lies inside ANY DAPI region?
    # This handles the case where one large DAPI blob covers multiple nuclei.
    per_mask_records: list[dict] = []
    covered_mask_labels: set[int] = set()

    dapi_binary = dapi_labels > 0  # any DAPI-positive region

    for m_lbl in mask_uniq:
        m_px     = mask == m_lbl
        m_area   = int(m_px.sum())
        # Overlap with any DAPI region
        in_dapi  = int((m_px & dapi_binary).sum())
        coverage = in_dapi / m_area if m_area > 0 else 0.0

        # Best IoU with individual DAPI objects
        overlap_dapi_lbls = np.unique(dapi_labels[m_px])
        overlap_dapi_lbls = overlap_dapi_lbls[overlap_dapi_lbls > 0]
        best_iou_m, best_d = 0.0, 0
        for d_lbl in overlap_dapi_lbls:
            d_px_m   = dapi_labels == d_lbl
            intersec = int((m_px & d_px_m).sum())
            union    = m_area + int(d_px_m.sum()) - intersec
            iou      = intersec / union if union > 0 else 0.0
            if iou > best_iou_m:
                best_iou_m, best_d = iou, int(d_lbl)

        # A mask cell is "covered" if ≥ overlap_threshold of it lies in DAPI
        covered = coverage >= overlap_threshold
        per_mask_records.append({
            "mask_label":         int(m_lbl),
            "matched_dapi_label": best_d,
            "iou":                best_iou_m,
            "coverage":           coverage,
            "matched":            covered,
        })
        if covered:
            covered_mask_labels.add(int(m_lbl))

    per_mask_df = pd.DataFrame(per_mask_records) if per_mask_records else pd.DataFrame(
        columns=["mask_label", "matched_dapi_label", "iou", "coverage", "matched"]
    )

    # ── Summary metrics ────────────────────────────────────────────────────────
    # IoU-based (DAPI perspective)
    matched_dapi_cnt = int(per_dapi_df["matched"].sum()) if len(per_dapi_df) else 0
    missed_dapi      = n_dapi - matched_dapi_cnt

    # Coverage-based (mask perspective) — the more meaningful metric
    covered_mask_cnt = len(covered_mask_labels)
    extra_mask       = n_mask - covered_mask_cnt

    # estimated_recall: of DAPI objects, how many have an IoU-match?
    recall    = matched_dapi_cnt / n_dapi if n_dapi > 0 else float("nan")
    # estimated_precision: of mask cells, how many are covered by DAPI?
    precision = covered_mask_cnt / n_mask if n_mask > 0 else float("nan")

    return {
        "dapi_object_count":   n_dapi,
        "mask_cell_count":     n_mask,
        "matched_count":       matched_dapi_cnt,
        "missed_dapi_count":   missed_dapi,
        "extra_mask_count":    extra_mask,
        "covered_mask_count":  covered_mask_cnt,
        "estimated_recall":    recall,
        "estimated_precision": precision,
        "per_dapi_df":         per_dapi_df,
        "per_mask_df":         per_mask_df,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Size QC
# ═══════════════════════════════════════════════════════════════════════════════

def compute_size_qc(
    mask: np.ndarray,
    min_area_px: int = 50,
    max_area_px: int = 10000,
) -> dict:
    """Compute size-based QC statistics for mask cells.

    Returns
    -------
    dict with cell_count, median_area_px, mean_area_px,
    small_mask_count, large_mask_count, small_mask_fraction, large_mask_fraction,
    area_p25, area_p75, area_p95
    """
    labels = np.unique(mask)
    labels = labels[labels > 0]

    if len(labels) == 0:
        return {
            "cell_count":           0,
            "median_area_px":       float("nan"),
            "mean_area_px":         float("nan"),
            "small_mask_count":     0,
            "large_mask_count":     0,
            "small_mask_fraction":  float("nan"),
            "large_mask_fraction":  float("nan"),
            "area_p25":             float("nan"),
            "area_p75":             float("nan"),
            "area_p95":             float("nan"),
        }

    # Compute areas efficiently using bincount
    flat = mask.ravel()
    counts = np.bincount(flat)
    areas = counts[labels]  # areas for each label

    n = len(areas)
    small_count = int((areas < min_area_px).sum())
    large_count = int((areas > max_area_px).sum())

    return {
        "cell_count":           n,
        "median_area_px":       float(np.median(areas)),
        "mean_area_px":         float(np.mean(areas)),
        "small_mask_count":     small_count,
        "large_mask_count":     large_count,
        "small_mask_fraction":  small_count / n,
        "large_mask_fraction":  large_count / n,
        "area_p25":             float(np.percentile(areas, 25)),
        "area_p75":             float(np.percentile(areas, 75)),
        "area_p95":             float(np.percentile(areas, 95)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Full FOV QC
# ═══════════════════════════════════════════════════════════════════════════════

def run_fov_dapi_qc(
    raw: np.ndarray,
    enhanced: np.ndarray,
    mask: np.ndarray,
    fov_id: str = "",
    model_name: str = "",
    min_area_px: int = 50,
    max_area_px: int = 10000,
    min_dapi_sigma_multiplier: float = 1.5,
    iou_threshold: float = 0.3,
    dapi_object_min_area: int = 200,
) -> dict:
    """Run all DAPI QC analyses for one FOV.

    Parameters
    ----------
    raw:       Full-resolution raw DAPI image.
    enhanced:  Enhanced DAPI image.
    mask:      Segmentation mask (0 = background).
    fov_id:    FOV identifier string.
    model_name: Segmentation model name.
    min_area_px / max_area_px: Size thresholds.
    min_dapi_sigma_multiplier: Low-DAPI threshold = bg_mean + N * bg_std.
    iou_threshold: Minimum IoU for DAPI<->mask match.
    dapi_object_min_area: Minimum DAPI object size in pixels.

    Returns
    -------
    dict with all scalar metrics + per_cell_df (merged per-cell DataFrame).
    """
    # ── 1. Coverage (use enhanced for better SNR; raw as fallback) ───────────
    cov = compute_dapi_coverage(
        enhanced if enhanced is not None else raw,
        mask,
        sigma_multiplier=min_dapi_sigma_multiplier,
    )

    # ── 2. Detect DAPI objects ────────────────────────────────────────────────
    # Use raw DAPI (not tophat-enhanced) to avoid detecting RNA noise spots.
    # Larger sigma (3.0) smooths sub-nuclear bright spots before thresholding.
    dapi_labels = detect_dapi_objects(
        raw,
        min_area_px=dapi_object_min_area,
        gaussian_sigma=3.0,
    )

    # ── 3. Match ──────────────────────────────────────────────────────────────
    match = match_objects_to_masks(dapi_labels, mask, iou_threshold=iou_threshold)

    # ── 4. Size QC ────────────────────────────────────────────────────────────
    size = compute_size_qc(mask, min_area_px=min_area_px, max_area_px=max_area_px)

    # ── 5. Merge per-cell data ────────────────────────────────────────────────
    per_cell_df   = cov["per_cell_df"].copy()        # label, mean_intensity, area, is_low_dapi
    per_mask_df   = match["per_mask_df"].copy()      # mask_label, matched_dapi_label, iou, matched

    # Join coverage + match data on label
    merged = per_cell_df.merge(
        per_mask_df.rename(columns={"mask_label": "label"}),
        on="label",
        how="outer",
    )

    # auto_qc_label
    def _auto_label(row):
        if row.get("matched", False) and not row.get("is_low_dapi", False):
            return "good"
        if row.get("is_low_dapi", False):
            return "low_dapi"
        return "extra"  # mask with no matching DAPI object

    merged["auto_qc_label"] = merged.apply(_auto_label, axis=1)

    # IoU-based aggregates (only over matched cells)
    matched_iou = per_mask_df.loc[per_mask_df["matched"], "iou"]
    mean_iou    = float(matched_iou.mean())   if len(matched_iou) > 0 else float("nan")
    median_iou  = float(matched_iou.median()) if len(matched_iou) > 0 else float("nan")
    low_iou_cnt = int((matched_iou < iou_threshold * 1.5).sum()) if len(matched_iou) > 0 else 0

    # Low-DAPI derived metrics
    n_cells = cov["cell_count"]
    low_dapi_count = int(per_cell_df["is_low_dapi"].sum()) if len(per_cell_df) > 0 else 0
    dapi_pos_frac  = (n_cells - low_dapi_count) / n_cells if n_cells > 0 else float("nan")
    low_dapi_frac  = low_dapi_count / n_cells if n_cells > 0 else float("nan")

    result: dict = {
        # identifiers
        "fov_id":                     fov_id,
        "model_name":                 model_name,
        # coverage
        "cell_count":                 cov["cell_count"],
        "mean_dapi_per_cell":         cov["mean_dapi_per_cell"],
        "median_dapi_per_cell":       cov["median_dapi_per_cell"],
        "background_mean":            cov["background_mean"],
        "background_std":             cov["background_std"],
        "signal_to_background_ratio": cov["signal_to_background_ratio"],
        # dapi positive / low dapi
        "dapi_positive_mask_fraction": dapi_pos_frac,
        "low_dapi_mask_count":         low_dapi_count,
        "low_dapi_mask_fraction":      low_dapi_frac,
        # match metrics
        "dapi_object_count":           match["dapi_object_count"],
        "mask_cell_count":             match["mask_cell_count"],
        "matched_count":               match["matched_count"],
        "missed_dapi_count":           match["missed_dapi_count"],
        "extra_mask_count":            match["extra_mask_count"],
        "estimated_recall":            match["estimated_recall"],
        "estimated_precision":         match["estimated_precision"],
        # IoU
        "mean_iou":                    mean_iou,
        "median_iou":                  median_iou,
        "low_iou_cell_count":          low_iou_cnt,
        # size QC
        "size_cell_count":             size["cell_count"],
        "size_median_area_px":         size["median_area_px"],
        "size_mean_area_px":           size["mean_area_px"],
        "small_mask_count":            size["small_mask_count"],
        "large_mask_count":            size["large_mask_count"],
        "small_mask_fraction":         size["small_mask_fraction"],
        "large_mask_fraction":         size["large_mask_fraction"],
        "area_p25":                    size["area_p25"],
        "area_p75":                    size["area_p75"],
        "area_p95":                    size["area_p95"],
        # DataFrames
        "per_cell_df":                 merged,
        "per_dapi_df":                 match["per_dapi_df"],
        "dapi_labels":                 dapi_labels,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Save QC visualizations
# ═══════════════════════════════════════════════════════════════════════════════

def save_qc_visualizations(
    raw: np.ndarray,
    enhanced: np.ndarray,
    mask: np.ndarray,
    dapi_labels: np.ndarray,
    qc_result: dict,
    output_dir: str,
    fov_id: str = "",
    model_name: str = "",
    max_dim: int = 1024,
) -> None:
    """Save 5 QC PNG files to output_dir.

    Files saved:
      1. {prefix}dapi_mask_overlay.png
      2. {prefix}dapi_objects_vs_masks.png
      3. {prefix}low_confidence_masks.png
      4. {prefix}missed_dapi_objects.png
      5. {prefix}mask_quality_histograms.png

    Color coding:
      green  (#00C800) = matched good mask
      red    (#E63946) = missed DAPI object
      yellow (#FFD700) = low DAPI signal mask
      blue   (#4575B4) = extra mask (no DAPI object)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    prefix = f"{fov_id}_" if fov_id else ""
    if model_name:
        prefix += f"{model_name}_"

    per_cell_df  = qc_result.get("per_cell_df",  pd.DataFrame())
    per_dapi_df  = qc_result.get("per_dapi_df",  pd.DataFrame())

    # Build lookup sets for fast coloring
    matched_mask_labels: set[int] = set()
    low_dapi_mask_labels: set[int] = set()
    extra_mask_labels: set[int] = set()
    missed_dapi_labels: set[int] = set()

    if len(per_cell_df) > 0:
        for _, row in per_cell_df.iterrows():
            lbl = int(row.get("label", 0))
            ql  = row.get("auto_qc_label", "extra")
            if ql == "good":
                matched_mask_labels.add(lbl)
            elif ql == "low_dapi":
                low_dapi_mask_labels.add(lbl)
            else:
                extra_mask_labels.add(lbl)

    if len(per_dapi_df) > 0:
        for _, row in per_dapi_df.iterrows():
            if not row.get("matched", False):
                missed_dapi_labels.add(int(row.get("dapi_label", 0)))

    # Downsample all images
    raw_ds      = _downsample(_norm8(raw), max_dim)
    enh_ds      = _downsample(_norm8(enhanced), max_dim)
    mask_ds     = _downsample_mask(mask, max_dim)
    dapi_ds     = _downsample_mask(dapi_labels, max_dim)

    # ── 1. DAPI + mask overlay ─────────────────────────────────────────────────
    _save_overlay_colored(
        raw_ds, mask_ds,
        matched_mask_labels, low_dapi_mask_labels, extra_mask_labels,
        str(out_path / f"{prefix}dapi_mask_overlay.png"),
        title=f"DAPI + Mask Overlay — {fov_id} {model_name}",
    )

    # ── 2. DAPI objects vs masks side-by-side ─────────────────────────────────
    _save_side_by_side(
        enh_ds, dapi_ds, enh_ds, mask_ds,
        str(out_path / f"{prefix}dapi_objects_vs_masks.png"),
        left_title="DAPI Objects",
        right_title="Segmentation Mask",
    )

    # ── 3. Low confidence masks ────────────────────────────────────────────────
    _save_low_conf(
        raw_ds, mask_ds, low_dapi_mask_labels,
        str(out_path / f"{prefix}low_confidence_masks.png"),
        title=f"Low DAPI Signal Masks — {fov_id}",
    )

    # ── 4. Missed DAPI objects ─────────────────────────────────────────────────
    _save_missed_dapi(
        enh_ds, dapi_ds, missed_dapi_labels,
        str(out_path / f"{prefix}missed_dapi_objects.png"),
        title=f"Missed DAPI Objects — {fov_id}",
    )

    # ── 5. Histograms ─────────────────────────────────────────────────────────
    _save_histograms(
        qc_result,
        str(out_path / f"{prefix}mask_quality_histograms.png"),
        title=f"{fov_id} {model_name}",
    )


# ── Visualization helpers ──────────────────────────────────────────────────────

def _label_set_to_rgb(
    mask_ds: np.ndarray,
    matched: set[int],
    low_dapi: set[int],
    extra: set[int],
) -> np.ndarray:
    """Return (H, W, 3) uint8 RGB from mask_ds with QC color coding."""
    rgb = np.zeros((*mask_ds.shape, 3), dtype=np.uint8)
    for lbl in np.unique(mask_ds):
        if lbl == 0:
            continue
        px = mask_ds == lbl
        if lbl in matched:
            rgb[px] = _COLOR_MATCHED
        elif lbl in low_dapi:
            rgb[px] = _COLOR_LOW_DAPI
        elif lbl in extra:
            rgb[px] = _COLOR_EXTRA
        else:
            rgb[px] = _COLOR_EXTRA
    return rgb


def _save_overlay_colored(
    gray: np.ndarray,
    mask_ds: np.ndarray,
    matched: set[int],
    low_dapi: set[int],
    extra: set[int],
    path: str,
    title: str = "",
    alpha: float = 0.45,
) -> None:
    rgb_mask = _label_set_to_rgb(mask_ds, matched, low_dapi, extra)
    base     = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    fg       = mask_ds > 0
    blended  = base.copy()
    blended[fg] = (1 - alpha) * base[fg] + alpha * rgb_mask[fg].astype(np.float32)
    blended  = np.clip(blended, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.imshow(blended, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=_COLOR_MATCHED  / 255, label="Matched"),
        mpatches.Patch(color=_COLOR_LOW_DAPI / 255, label="Low DAPI"),
        mpatches.Patch(color=_COLOR_EXTRA    / 255, label="Extra mask"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7,
              framealpha=0.7)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _save_side_by_side(
    img_left: np.ndarray,
    mask_left: np.ndarray,
    img_right: np.ndarray,
    mask_right: np.ndarray,
    path: str,
    left_title: str = "Left",
    right_title: str = "Right",
) -> None:
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=120)

    axes[0].imshow(img_left, cmap="gray", interpolation="nearest")
    # overlay DAPI objects with a light color
    if mask_left.max() > 0:
        n = int(mask_left.max())
        rng = np.random.default_rng(0)
        colors = [(0, 0, 0, 0)] + [(*rng.random(3), 0.6) for _ in range(n)]
        cmap = ListedColormap(colors)
        axes[0].imshow(mask_left, cmap=cmap, vmin=0, vmax=n, interpolation="nearest")
    axes[0].set_title(left_title, fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(img_right, cmap="gray", interpolation="nearest")
    if mask_right.max() > 0:
        n2 = int(mask_right.max())
        rng2 = np.random.default_rng(1)
        colors2 = [(0, 0, 0, 0)] + [(*rng2.random(3), 0.6) for _ in range(n2)]
        cmap2 = ListedColormap(colors2)
        axes[1].imshow(mask_right, cmap=cmap2, vmin=0, vmax=n2, interpolation="nearest")
    axes[1].set_title(right_title, fontsize=10)
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _save_low_conf(
    gray: np.ndarray,
    mask_ds: np.ndarray,
    low_dapi_labels: set[int],
    path: str,
    title: str = "",
    alpha: float = 0.5,
) -> None:
    rgb_overlay = np.zeros((*mask_ds.shape, 3), dtype=np.uint8)
    for lbl in np.unique(mask_ds):
        if lbl == 0:
            continue
        px = mask_ds == lbl
        if lbl in low_dapi_labels:
            rgb_overlay[px] = _COLOR_LOW_DAPI
        else:
            rgb_overlay[px] = np.array([60, 60, 60], dtype=np.uint8)

    base    = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    fg      = mask_ds > 0
    blended = base.copy()
    blended[fg] = (1 - alpha) * base[fg] + alpha * rgb_overlay[fg].astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.imshow(blended, interpolation="nearest")
    ax.set_title(f"{title}\n({len(low_dapi_labels)} low-DAPI cells)", fontsize=9)
    ax.axis("off")
    legend_patches = [
        mpatches.Patch(color=_COLOR_LOW_DAPI / 255, label="Low DAPI signal"),
        mpatches.Patch(color=np.array([60, 60, 60]) / 255, label="Normal"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.7)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _save_missed_dapi(
    gray: np.ndarray,
    dapi_ds: np.ndarray,
    missed_labels: set[int],
    path: str,
    title: str = "",
    alpha: float = 0.5,
) -> None:
    rgb_overlay = np.zeros((*dapi_ds.shape, 3), dtype=np.uint8)
    for lbl in np.unique(dapi_ds):
        if lbl == 0:
            continue
        px = dapi_ds == lbl
        if lbl in missed_labels:
            rgb_overlay[px] = _COLOR_MISSED
        else:
            rgb_overlay[px] = _COLOR_MATCHED

    base    = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    fg      = dapi_ds > 0
    blended = base.copy()
    blended[fg] = (1 - alpha) * base[fg] + alpha * rgb_overlay[fg].astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.imshow(blended, interpolation="nearest")
    ax.set_title(f"{title}\n({len(missed_labels)} missed DAPI objects)", fontsize=9)
    ax.axis("off")
    legend_patches = [
        mpatches.Patch(color=_COLOR_MISSED  / 255, label="Missed DAPI"),
        mpatches.Patch(color=_COLOR_MATCHED / 255, label="Matched DAPI"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.7)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _save_histograms(
    qc_result: dict,
    path: str,
    title: str = "",
) -> None:
    per_cell_df = qc_result.get("per_cell_df", pd.DataFrame())
    per_mask_df = qc_result.get("per_mask_df", per_cell_df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)
    fig.suptitle(f"Mask Quality Histograms — {title}", fontsize=10)

    # Panel 1: DAPI intensity distribution
    ax = axes[0]
    if len(per_cell_df) > 0 and "mean_intensity" in per_cell_df.columns:
        intensities = per_cell_df["mean_intensity"].dropna()
        if len(intensities) > 0:
            ax.hist(intensities, bins=40, color="#4575B4", edgecolor="none", alpha=0.75)
            bg_mean = qc_result.get("background_mean", 0)
            bg_std  = qc_result.get("background_std", 0)
            thresh  = bg_mean + 2 * bg_std
            ax.axvline(thresh, color="#E63946", linestyle="--", linewidth=1.5,
                       label=f"Low-DAPI threshold\n({thresh:.0f})")
            ax.legend(fontsize=7)
    ax.set_xlabel("Mean DAPI intensity")
    ax.set_ylabel("Cell count")
    ax.set_title("DAPI Intensity Distribution")

    # Panel 2: Cell area distribution
    ax = axes[1]
    if len(per_cell_df) > 0 and "area" in per_cell_df.columns:
        areas = per_cell_df["area"].dropna()
        if len(areas) > 0:
            ax.hist(areas, bins=40, color="#00C800", edgecolor="none", alpha=0.75)
            min_a = qc_result.get("small_mask_count", None)
    ax.set_xlabel("Cell area (px²)")
    ax.set_ylabel("Cell count")
    ax.set_title("Cell Area Distribution")

    # Panel 3: IoU distribution (matched cells)
    ax = axes[2]
    if "per_mask_df" in qc_result:
        pm_df = qc_result["per_mask_df"]
    elif "per_mask_df" in per_mask_df.columns if hasattr(per_mask_df, "columns") else False:
        pm_df = per_mask_df
    else:
        pm_df = pd.DataFrame()

    # Try from per_cell_df
    if len(per_cell_df) > 0 and "iou" in per_cell_df.columns:
        ious = per_cell_df.loc[per_cell_df["matched"] == True, "iou"].dropna() \
            if "matched" in per_cell_df.columns else per_cell_df["iou"].dropna()
        if len(ious) > 0:
            ax.hist(ious, bins=30, color="#FFD700", edgecolor="none", alpha=0.85)
            ax.axvline(0.3, color="#E63946", linestyle="--", linewidth=1.5,
                       label="IoU=0.3 threshold")
            ax.legend(fontsize=7)
    ax.set_xlabel("IoU")
    ax.set_ylabel("Cell count")
    ax.set_title("IoU Distribution (matched cells)")

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
