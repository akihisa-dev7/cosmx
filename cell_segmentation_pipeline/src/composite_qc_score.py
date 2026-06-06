"""Composite QC scoring and FOV overview generation.

Replaces "CosMx IoU only" with a multi-axis evaluation:
  Axis 1 — DAPI nucleus quality
  Axis 2 — CosMx native label spatial correspondence
  Axis 3 — Cell count / area consistency

Key design: Population IoU is intentionally low when CosMx uses cell-boundary
segmentation and custom uses DAPI-nucleus segmentation.  This is expected and
should NOT be used as the primary quality gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Weights for composite score ───────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "w_dapi_snr":           0.30,
    "w_cell_count":         0.20,
    "w_area_dist":          0.15,
    "w_one_to_one":         0.20,
    "w_matched_iou":        0.10,
    "w_overseg_penalty":    0.15,
    "w_underseg_penalty":   0.15,
}


# ── Extended boundary metrics ─────────────────────────────────────────────────

def compute_extended_match_metrics(
    per_cell_df: pd.DataFrame,
    custom_centroids: pd.DataFrame,
    cosmx_centroids: pd.DataFrame,
    match_dist_px: float = 30.0,
) -> dict:
    """Compute extended matching statistics from per_cell_df.

    Requires columns: query_label, ref_label, centroid_dist, matched, iou
    """
    if per_cell_df is None or per_cell_df.empty:
        return _empty_extended_metrics()

    matched = per_cell_df[per_cell_df["matched"]].copy()
    n_custom = len(per_cell_df)
    n_cosmx  = len(cosmx_centroids) if cosmx_centroids is not None else 0

    if n_custom == 0 or n_cosmx == 0:
        return _empty_extended_metrics()

    # ── One-to-one vs multi-match ──────────────────────────────────────────────
    # Count how many custom cells map to each CosMx cell
    if len(matched) > 0:
        ref_counts = matched.groupby("ref_label").size()
        one_to_one_count = int((ref_counts == 1).sum())
        multi_match_count = int((ref_counts > 1).sum())
    else:
        one_to_one_count = 0
        multi_match_count = 0

    # ── Zero-match CosMx fraction ─────────────────────────────────────────────
    cosmx_labels = set(cosmx_centroids["label"].astype(int).tolist()) if cosmx_centroids is not None else set()
    matched_cosmx = set(matched["ref_label"].astype(int).tolist()) if len(matched) > 0 else set()
    zero_match_cosmx = len(cosmx_labels - matched_cosmx)

    # ── Registration stats (mean dx, dy of matched pairs) ─────────────────────
    mean_dx, mean_dy = float("nan"), float("nan")
    std_dx,  std_dy  = float("nan"), float("nan")

    if len(matched) > 0 and custom_centroids is not None and cosmx_centroids is not None:
        cust_map  = custom_centroids.set_index("label")[["centroid_x", "centroid_y"]].to_dict("index")
        cosmx_map = cosmx_centroids.set_index("label")[["centroid_x", "centroid_y"]].to_dict("index")
        dxs, dys  = [], []
        for _, row in matched.iterrows():
            ql, rl = int(row["query_label"]), int(row["ref_label"])
            if ql in cust_map and rl in cosmx_map:
                dxs.append(cust_map[ql]["centroid_x"] - cosmx_map[rl]["centroid_x"])
                dys.append(cust_map[ql]["centroid_y"] - cosmx_map[rl]["centroid_y"])
        if dxs:
            mean_dx = float(np.mean(dxs))
            mean_dy = float(np.mean(dys))
            std_dx  = float(np.std(dxs))
            std_dy  = float(np.std(dys))

    # ── Area ratio ────────────────────────────────────────────────────────────
    area_ratios = []
    if len(matched) > 0 and custom_centroids is not None and cosmx_centroids is not None:
        cust_area  = custom_centroids.set_index("label")["area"].to_dict()
        cosmx_area = cosmx_centroids.set_index("label")["area"].to_dict()
        for _, row in matched.iterrows():
            ql, rl = int(row["query_label"]), int(row["ref_label"])
            ca = cust_area.get(ql, np.nan)
            ra = cosmx_area.get(rl, np.nan)
            if not (np.isnan(ca) or np.isnan(ra)) and ra > 0:
                area_ratios.append(ca / ra)

    area_ratio_median = float(np.median(area_ratios)) if area_ratios else float("nan")
    area_ratio_mean   = float(np.mean(area_ratios))   if area_ratios else float("nan")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_matched = len(matched)
    matched_iou = matched["iou"].dropna()

    return {
        "n_custom":                 n_custom,
        "n_cosmx":                  n_cosmx,
        "n_matched":                n_matched,
        "match_rate_custom":        n_matched / n_custom if n_custom > 0 else float("nan"),
        "match_rate_cosmx":         n_matched / n_cosmx  if n_cosmx  > 0 else float("nan"),
        "one_to_one_match_count":   one_to_one_count,
        "one_to_one_match_rate":    one_to_one_count / n_cosmx if n_cosmx > 0 else float("nan"),
        "multi_match_cosmx_count":  multi_match_count,
        "multi_match_cosmx_fraction": multi_match_count / n_cosmx if n_cosmx > 0 else float("nan"),
        "zero_match_cosmx_count":   zero_match_cosmx,
        "zero_match_cosmx_fraction": zero_match_cosmx / n_cosmx if n_cosmx > 0 else float("nan"),
        "mean_dx":                  mean_dx,
        "mean_dy":                  mean_dy,
        "std_dx":                   std_dx,
        "std_dy":                   std_dy,
        "registration_shift_px":    float(np.sqrt(mean_dx**2 + mean_dy**2)) if not np.isnan(mean_dx) else float("nan"),
        "area_ratio_median":        area_ratio_median,
        "area_ratio_mean":          area_ratio_mean,
        "matched_pair_iou_mean":    float(matched_iou.mean())   if len(matched_iou) > 0 else float("nan"),
        "matched_pair_iou_median":  float(matched_iou.median()) if len(matched_iou) > 0 else float("nan"),
        "custom_to_cosmx_ratio":    n_custom / n_cosmx if n_cosmx > 0 else float("nan"),
    }


def _empty_extended_metrics() -> dict:
    keys = [
        "n_custom", "n_cosmx", "n_matched",
        "match_rate_custom", "match_rate_cosmx",
        "one_to_one_match_count", "one_to_one_match_rate",
        "multi_match_cosmx_count", "multi_match_cosmx_fraction",
        "zero_match_cosmx_count", "zero_match_cosmx_fraction",
        "mean_dx", "mean_dy", "std_dx", "std_dy", "registration_shift_px",
        "area_ratio_median", "area_ratio_mean",
        "matched_pair_iou_mean", "matched_pair_iou_median",
        "custom_to_cosmx_ratio",
    ]
    return {k: float("nan") for k in keys}


# ── DAPI quality classification ───────────────────────────────────────────────

def classify_dapi_quality(snr: float) -> tuple[str, str]:
    """Return (status_label, color) for a given SNR value."""
    if np.isnan(snr):
        return "Unknown", "gray"
    if snr >= 3.0:
        return "Good", "green"
    if snr >= 2.0:
        return "Acceptable", "orange"
    return "Warning", "red"


# ── Over / under segmentation detection ──────────────────────────────────────

def detect_segmentation_issues(
    custom_n: int,
    cosmx_n: int,
    match_rate_custom: float,
    one_to_one_rate: float,
    multi_match_fraction: float,
) -> list[dict]:
    """Return list of warning dicts with keys: type, severity, message."""
    warnings = []

    ratio = custom_n / cosmx_n if cosmx_n > 0 else float("nan")

    if not np.isnan(ratio):
        if ratio >= 2.5:
            warnings.append({
                "type": "over_segmentation",
                "severity": "high",
                "message": (
                    f"Possible over-segmentation: custom has {custom_n} cells "
                    f"vs CosMx {cosmx_n} (ratio {ratio:.1f}×). "
                    "Review in Model Comparison page."
                ),
            })
        elif ratio >= 2.0:
            warnings.append({
                "type": "over_segmentation",
                "severity": "medium",
                "message": (
                    f"Possible over-segmentation: custom/CosMx ratio = {ratio:.1f}×. "
                    "Check if one nucleus is split into multiple masks."
                ),
            })
        elif ratio <= 0.4:
            warnings.append({
                "type": "under_segmentation",
                "severity": "high",
                "message": (
                    f"Possible under-segmentation: custom has {custom_n} cells "
                    f"vs CosMx {cosmx_n} (ratio {ratio:.2f}×). "
                    "Review in Model Comparison page."
                ),
            })
        elif ratio <= 0.5:
            warnings.append({
                "type": "under_segmentation",
                "severity": "medium",
                "message": (
                    f"Possible under-segmentation: custom/CosMx ratio = {ratio:.2f}×."
                ),
            })

    if not np.isnan(match_rate_custom) and match_rate_custom < 0.10:
        warnings.append({
            "type": "low_match_rate",
            "severity": "medium",
            "message": (
                f"Low match rate: only {match_rate_custom*100:.1f}% of custom cells "
                "correspond to a CosMx native cell within the distance threshold. "
                "This may indicate a definition difference, not necessarily an error."
            ),
        })

    if not np.isnan(multi_match_fraction) and multi_match_fraction > 0.20:
        warnings.append({
            "type": "multi_match",
            "severity": "medium",
            "message": (
                f"{multi_match_fraction*100:.1f}% of CosMx cells are matched "
                "by multiple custom cells, suggesting local over-segmentation."
            ),
        })

    return warnings


# ── Registration status ───────────────────────────────────────────────────────

def classify_registration(
    mean_dx: float,
    mean_dy: float,
    std_dx: float,
    std_dy: float,
) -> tuple[str, str]:
    """Return (status, message) for registration quality."""
    if np.isnan(mean_dx):
        return "Unknown", "No matched pairs available for registration check."

    shift = float(np.sqrt(mean_dx**2 + mean_dy**2))

    if shift < 5.0:
        status = "OK"
        msg = (
            f"Registration OK — mean shift: dx={mean_dx:+.1f}px, dy={mean_dy:+.1f}px "
            f"(|shift|={shift:.1f}px). "
            "Low Population IoU is due to segmentation definition difference, "
            "not coordinate misalignment."
        )
    elif shift < 20.0:
        status = "Minor offset"
        msg = (
            f"Minor coordinate offset detected: dx={mean_dx:+.1f}px, dy={mean_dy:+.1f}px "
            f"(|shift|={shift:.1f}px). "
            "Within acceptable range; unlikely to affect cell type assignment."
        )
    else:
        status = "Warning"
        msg = (
            f"Significant coordinate shift: dx={mean_dx:+.1f}px, dy={mean_dy:+.1f}px "
            f"(|shift|={shift:.1f}px). "
            "Check if the correct FOV images are being used."
        )

    return status, msg


# ── Composite score ───────────────────────────────────────────────────────────

def compute_composite_score(
    dapi_snr: float,
    custom_n: int,
    cosmx_n: int,
    area_ratio_median: float,
    one_to_one_rate: float,
    matched_pair_iou_median: float,
    weights: Optional[dict] = None,
) -> dict:
    """Compute weighted composite QC score (0–1).

    Intentionally does NOT include Population IoU.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    def _safe(v, lo=0.0, hi=1.0):
        return float(np.clip(v, lo, hi)) if not np.isnan(v) else 0.5

    # Axis 1: DAPI SNR score (3.0 = perfect, 1.0 = noise)
    snr_score = _safe((dapi_snr - 1.0) / 3.0)

    # Axis 2: Cell count reasonableness (ratio close to 1 is good)
    ratio = custom_n / cosmx_n if cosmx_n > 0 else 1.0
    count_score = _safe(1.0 - abs(np.log(ratio)) / np.log(3.0))

    # Axis 3: Area distribution (ratio close to 0.4–0.8 expected for nucleus vs cell)
    # cpsam nucleus ≈ 0.4–0.7× CosMx cell area
    target_ratio = 0.55
    area_score = _safe(1.0 - abs(area_ratio_median - target_ratio) / target_ratio) if not np.isnan(area_ratio_median) else 0.5

    # Axis 4: One-to-one match rate
    oto_score = _safe(one_to_one_rate)

    # Axis 5: Matched-pair IoU
    iou_score = _safe(matched_pair_iou_median)

    # Penalties
    overseg_penalty  = max(0.0, (ratio - 2.0) / 2.0) if ratio > 2.0 else 0.0
    underseg_penalty = max(0.0, (0.5 - ratio) / 0.5) if ratio < 0.5 else 0.0

    score = (
        w["w_dapi_snr"]        * snr_score
        + w["w_cell_count"]    * count_score
        + w["w_area_dist"]     * area_score
        + w["w_one_to_one"]    * oto_score
        + w["w_matched_iou"]   * iou_score
        - w["w_overseg_penalty"]  * overseg_penalty
        - w["w_underseg_penalty"] * underseg_penalty
    )

    return {
        "composite_score":   round(float(np.clip(score, 0, 1)), 4),
        "snr_component":     round(snr_score, 4),
        "count_component":   round(count_score, 4),
        "area_component":    round(area_score, 4),
        "oto_component":     round(oto_score, 4),
        "iou_component":     round(iou_score, 4),
        "overseg_penalty":   round(overseg_penalty, 4),
        "underseg_penalty":  round(underseg_penalty, 4),
    }


# ── Recommended action ────────────────────────────────────────────────────────

def recommend_action(
    composite_score: float,
    dapi_status: str,
    seg_warnings: list[dict],
    match_rate_custom: float,
) -> str:
    warn_types = {w["type"] for w in seg_warnings}

    if composite_score >= 0.65 and not warn_types:
        return "Good — proceed to Cell Type Validation"

    actions = []
    if dapi_status == "Warning":
        actions.append("Check DAPI quality (SNR < 2.0)")
    if "over_segmentation" in warn_types:
        actions.append("Check over-segmentation in Model Comparison")
    if "under_segmentation" in warn_types:
        actions.append("Check under-segmentation in Model Comparison")
    if "low_match_rate" in warn_types:
        actions.append("Note: low match rate expected due to definition difference")
    if not np.isnan(match_rate_custom) and match_rate_custom < 0.05:
        actions.append("Review manually in Random Cell Review")

    if not actions:
        actions.append("Do not optimize by CosMx Population IoU alone")

    return " | ".join(actions) if actions else "Review recommended"


# ── FOV overview row ──────────────────────────────────────────────────────────

def build_fov_overview_row(
    fov_name: str,
    model_name: str,
    dapi_summary: Optional[pd.Series],
    label_summary: Optional[pd.Series],
    extended_metrics: Optional[dict],
) -> dict:
    """Build one row of fov_qc_overview.csv."""
    row: dict = {"fov": fov_name, "model": model_name}

    # DAPI metrics
    if dapi_summary is not None:
        snr = float(dapi_summary.get("signal_to_background_ratio", float("nan")))
        row["custom_cell_count"]     = int(dapi_summary.get("cell_count", 0))
        row["dapi_snr"]              = round(snr, 3)
        row["dapi_positive_fraction"] = round(float(dapi_summary.get("dapi_positive_mask_fraction", float("nan"))), 3)
        row["low_dapi_cell_fraction"] = round(float(dapi_summary.get("low_dapi_mask_fraction", float("nan"))), 3)
        row["median_mask_area_px"]   = round(float(dapi_summary.get("size_median_area_px", float("nan"))), 1)
        row["small_object_fraction"] = round(float(dapi_summary.get("small_mask_fraction", float("nan"))), 3)
        row["large_object_fraction"] = round(float(dapi_summary.get("large_mask_fraction", float("nan"))), 3)
        dapi_status, _ = classify_dapi_quality(snr)
        row["dapi_status"] = dapi_status
    else:
        snr = float("nan")
        dapi_status = "Unknown"
        row["dapi_status"] = dapi_status

    # Label metrics
    if label_summary is not None:
        row["cosmx_cell_count"]      = int(label_summary.get("cosmx_n_cells", 0))
        row["population_iou"]        = round(float(label_summary.get("nucleus_iou", float("nan"))), 4)
        row["nucleus_dice"]          = round(float(label_summary.get("nucleus_dice", float("nan"))), 4)
        row["boundary_f1"]           = round(float(label_summary.get("boundary_f1_vs_nucleus", float("nan"))), 4)
    else:
        row["cosmx_cell_count"] = 0

    # Extended metrics
    if extended_metrics is not None:
        for k in [
            "matched_pair_iou_mean", "matched_pair_iou_median",
            "match_rate_custom", "match_rate_cosmx",
            "one_to_one_match_rate", "multi_match_cosmx_fraction",
            "zero_match_cosmx_fraction", "area_ratio_median",
            "mean_dx", "mean_dy", "registration_shift_px",
            "custom_to_cosmx_ratio",
        ]:
            v = extended_metrics.get(k, float("nan"))
            row[k] = round(float(v), 4) if not np.isnan(float(v)) else float("nan")

        n_custom = int(extended_metrics.get("n_custom", row.get("custom_cell_count", 0)))
        n_cosmx  = int(extended_metrics.get("n_cosmx", row.get("cosmx_cell_count", 0)))

        # Segmentation warnings
        warnings = detect_segmentation_issues(
            n_custom, n_cosmx,
            extended_metrics.get("match_rate_custom", float("nan")),
            extended_metrics.get("one_to_one_match_rate", float("nan")),
            extended_metrics.get("multi_match_cosmx_fraction", float("nan")),
        )
        warn_types = {w["type"] for w in warnings}
        row["oversegmentation_flag"]  = "over_segmentation"  in warn_types
        row["undersegmentation_flag"] = "under_segmentation" in warn_types
        row["low_match_flag"]         = "low_match_rate"      in warn_types

        # Registration
        reg_status, _ = classify_registration(
            extended_metrics.get("mean_dx", float("nan")),
            extended_metrics.get("mean_dy", float("nan")),
            extended_metrics.get("std_dx", float("nan")),
            extended_metrics.get("std_dy", float("nan")),
        )
        row["registration_status"] = reg_status

        # Composite score
        score_dict = compute_composite_score(
            dapi_snr=snr,
            custom_n=n_custom,
            cosmx_n=n_cosmx,
            area_ratio_median=extended_metrics.get("area_ratio_median", float("nan")),
            one_to_one_rate=extended_metrics.get("one_to_one_match_rate", float("nan")),
            matched_pair_iou_median=extended_metrics.get("matched_pair_iou_median", float("nan")),
        )
        row["composite_score"] = score_dict["composite_score"]

        # Recommended action
        row["recommended_action"] = recommend_action(
            score_dict["composite_score"],
            dapi_status,
            warnings,
            extended_metrics.get("match_rate_custom", float("nan")),
        )
    else:
        row["oversegmentation_flag"]  = False
        row["undersegmentation_flag"] = False
        row["low_match_flag"]         = False
        row["registration_status"]    = "Unknown"
        row["composite_score"]        = float("nan")
        row["recommended_action"]     = "Run QC pipeline first"

    return row
