"""CosMx native-label based segmentation QC.

Loads CompartmentLabels / CellLabels from per_fov_decoded and compares
them against custom segmentation masks.  Complements (does not replace)
the Otsu-based QC in dapi_qc.py.

CompartmentLabels pixel values
  0 = background
  1 = nucleus
  2 = cytoplasm
  3 = membrane
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from skimage.measure import regionprops_table
from skimage.morphology import binary_erosion, binary_dilation, disk
from scipy.spatial import KDTree


# ── Path helpers ───────────────────────────────────────────────────────────────

def find_per_fov_dir(
    fov_name: str,
    per_fov_root: str,
) -> Optional[Path]:
    """Return the per_fov_decoded sub-directory for a given FOV name."""
    d = Path(per_fov_root) / fov_name
    return d if d.exists() else None


def find_per_fov_file(fov_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(fov_dir.glob(pattern))
    return matches[0] if matches else None


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_compartment_labels(fov_dir: Path) -> Optional[np.ndarray]:
    f = find_per_fov_file(fov_dir, "CompartmentLabels_*.tif")
    if f is None:
        return None
    import tifffile
    return tifffile.imread(str(f)).astype(np.uint8)


def load_cell_labels(fov_dir: Path) -> Optional[np.ndarray]:
    f = find_per_fov_file(fov_dir, "CellLabels_*.tif")
    if f is None:
        return None
    import tifffile
    return tifffile.imread(str(f)).astype(np.int32)


def load_cell_stats(fov_dir: Path) -> Optional[pd.DataFrame]:
    f = find_per_fov_file(fov_dir, "Run_*_Cell_Stats_*.csv")
    if f is None:
        return None
    return pd.read_csv(str(f))


# ── Boundary extraction ────────────────────────────────────────────────────────

def extract_boundary(binary_mask: np.ndarray, dilate: int = 0) -> np.ndarray:
    """Extract boundary pixels: mask XOR erosion(mask).

    Parameters
    ----------
    binary_mask : bool/uint8 2D array
    dilate      : optional dilation radius for display (0 = analysis-safe)

    Returns boolean boundary mask.
    """
    bm = binary_mask.astype(bool)
    eroded   = binary_erosion(bm)
    boundary = bm & ~eroded
    if dilate > 0:
        boundary = binary_dilation(boundary, disk(dilate))
    return boundary


# ── Label-based masks ─────────────────────────────────────────────────────────

class CosMxLabelData:
    """Container for CosMx native segmentation data for one FOV."""

    def __init__(
        self,
        compartment: np.ndarray,
        cell_labels: np.ndarray,
        cell_stats: Optional[pd.DataFrame] = None,
        fov_name: str = "",
    ):
        self.compartment  = compartment   # (H,W) uint8  0/1/2/3
        self.cell_labels  = cell_labels   # (H,W) int32  0..N
        self.cell_stats   = cell_stats    # DataFrame or None
        self.fov_name     = fov_name

    @property
    def nucleus_mask(self) -> np.ndarray:
        return (self.compartment == 1).astype(np.uint8)

    @property
    def cytoplasm_mask(self) -> np.ndarray:
        return (self.compartment == 2).astype(np.uint8)

    @property
    def membrane_mask(self) -> np.ndarray:
        return (self.compartment == 3).astype(np.uint8)

    @property
    def cell_mask(self) -> np.ndarray:
        return (self.cell_labels > 0).astype(np.uint8)

    @property
    def n_cells(self) -> int:
        return int(self.cell_labels.max())

    def sanity_check(self, custom_mask: np.ndarray) -> dict:
        """Check shape/dtype consistency and return a report dict."""
        h, w = self.compartment.shape
        issues = []

        if self.compartment.shape != self.cell_labels.shape:
            issues.append("CompartmentLabels shape != CellLabels shape")
        if custom_mask.shape != self.compartment.shape:
            issues.append(
                f"custom_mask shape {custom_mask.shape} != "
                f"compartment shape {self.compartment.shape}"
            )
        if h != 4256 or w != 4256:
            issues.append(f"Expected 4256×4256, got {h}×{w}")

        uniq = set(np.unique(self.compartment).tolist())
        expected = {0, 1, 2, 3}
        if not uniq.issubset(expected):
            issues.append(f"Unexpected compartment values: {uniq - expected}")

        return {
            "fov_name":          self.fov_name,
            "shape_ok":          len(issues) == 0,
            "issues":            issues,
            "compartment_shape": self.compartment.shape,
            "custom_mask_shape": custom_mask.shape,
            "cosmx_n_cells":     self.n_cells,
            "custom_n_cells":    int(custom_mask.max()),
            "compartment_vals":  sorted(uniq),
        }


# ── Cell matching (spatial nearest-centroid) ───────────────────────────────────

def compute_centroids(mask: np.ndarray) -> pd.DataFrame:
    """Return DataFrame with label, centroid_y, centroid_x for all labels > 0."""
    props = regionprops_table(mask, properties=["label", "centroid", "area"])
    return pd.DataFrame({
        "label":      props["label"],
        "centroid_y": props["centroid-0"],
        "centroid_x": props["centroid-1"],
        "area":       props["area"],
    })


def match_cells_by_centroid(
    ref_centroids: pd.DataFrame,
    query_centroids: pd.DataFrame,
    max_dist_px: float = 30.0,
) -> pd.DataFrame:
    """Match query cells to reference cells by nearest centroid.

    Returns DataFrame with columns:
      query_label, ref_label, centroid_dist, matched
    """
    if len(ref_centroids) == 0 or len(query_centroids) == 0:
        return pd.DataFrame(
            columns=["query_label", "ref_label", "centroid_dist", "matched"]
        )

    ref_xy    = ref_centroids[["centroid_x", "centroid_y"]].values
    query_xy  = query_centroids[["centroid_x", "centroid_y"]].values
    ref_labels = ref_centroids["label"].values

    tree  = KDTree(ref_xy)
    dists, idxs = tree.query(query_xy)

    records = []
    for i, (d, idx) in enumerate(zip(dists, idxs)):
        records.append({
            "query_label":   int(query_centroids["label"].iloc[i]),
            "ref_label":     int(ref_labels[idx]) if d <= max_dist_px else 0,
            "centroid_dist": float(d),
            "matched":       d <= max_dist_px,
        })
    return pd.DataFrame(records)


# ── Per-cell IoU & Dice ────────────────────────────────────────────────────────

def compute_pairwise_iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    match_df: pd.DataFrame,
    label_col_a: str = "query_label",
    label_col_b: str = "ref_label",
) -> pd.DataFrame:
    """Compute IoU and Dice for matched cell pairs.

    Appends 'iou' and 'dice' columns to match_df.
    """
    ious, dices = [], []
    for _, row in match_df.iterrows():
        if not row["matched"]:
            ious.append(float("nan"))
            dices.append(float("nan"))
            continue
        la, lb = int(row[label_col_a]), int(row[label_col_b])
        pa = mask_a == la
        pb = mask_b == lb
        inter = int((pa & pb).sum())
        union = int((pa | pb).sum())
        aa, ab = int(pa.sum()), int(pb.sum())
        ious.append(inter / union if union > 0 else float("nan"))
        dices.append(2 * inter / (aa + ab) if (aa + ab) > 0 else float("nan"))

    out = match_df.copy()
    out["iou"]  = ious
    out["dice"] = dices
    return out


# ── Boundary F1 score ─────────────────────────────────────────────────────────

def boundary_f1(
    pred_mask: np.ndarray,
    ref_mask:  np.ndarray,
    tolerance_px: int = 3,
) -> dict:
    """Population-level boundary F1 score.

    A predicted boundary pixel is 'correct' if there is a reference boundary
    pixel within tolerance_px distance, and vice versa.
    """
    pred_bnd = extract_boundary(pred_mask > 0)
    ref_bnd  = extract_boundary(ref_mask  > 0)

    # Dilate ref boundary by tolerance → any pred pixel in this region is TP
    if tolerance_px > 0:
        ref_dilated  = binary_dilation(ref_bnd,  disk(tolerance_px))
        pred_dilated = binary_dilation(pred_bnd, disk(tolerance_px))
    else:
        ref_dilated  = ref_bnd
        pred_dilated = pred_bnd

    pred_bnd_px = int(pred_bnd.sum())
    ref_bnd_px  = int(ref_bnd.sum())
    tp_pred     = int((pred_bnd & ref_dilated).sum())   # predicted within tolerance of ref
    tp_ref      = int((ref_bnd  & pred_dilated).sum())  # ref within tolerance of pred

    precision = tp_pred / pred_bnd_px if pred_bnd_px > 0 else float("nan")
    recall    = tp_ref  / ref_bnd_px  if ref_bnd_px  > 0 else float("nan")
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    return {
        "boundary_precision": precision,
        "boundary_recall":    recall,
        "boundary_f1":        f1,
        "pred_boundary_px":   pred_bnd_px,
        "ref_boundary_px":    ref_bnd_px,
    }


# ── Full comparison ────────────────────────────────────────────────────────────

def compare_custom_vs_cosmx(
    custom_mask:  np.ndarray,
    cosmx_data:   CosMxLabelData,
    match_dist_px: float = 30.0,
    boundary_tol:  int   = 3,
) -> dict:
    """Full comparison between custom segmentation and CosMx native labels.

    Returns a dict with:
      - scalar summary metrics
      - per_cell_df: matched pairs with IoU/Dice/centroid_dist
      - sanity_check: coordinate/shape validation
    """
    sane = cosmx_data.sanity_check(custom_mask)

    # ── Population-level mask overlap ────────────────────────────────────────
    cosmx_nuc   = cosmx_data.nucleus_mask.astype(bool)
    cosmx_cell  = cosmx_data.cell_mask.astype(bool)
    custom_bool = (custom_mask > 0)

    def _iou_binary(a, b):
        inter = int((a & b).sum())
        union = int((a | b).sum())
        return inter / union if union > 0 else float("nan")

    def _dice_binary(a, b):
        inter = int((a & b).sum())
        denom = int(a.sum()) + int(b.sum())
        return 2 * inter / denom if denom > 0 else float("nan")

    nucleus_iou    = _iou_binary(cosmx_nuc,  custom_bool)
    nucleus_dice   = _dice_binary(cosmx_nuc, custom_bool)
    cell_iou       = _iou_binary(cosmx_cell, custom_bool)
    cell_dice      = _dice_binary(cosmx_cell, custom_bool)

    # ── Boundary F1 (custom vs cosmx nucleus boundary) ────────────────────────
    bf1_nuc  = boundary_f1(custom_mask > 0, cosmx_data.nucleus_mask > 0, boundary_tol)
    bf1_cell = boundary_f1(custom_mask > 0, cosmx_data.cell_mask > 0, boundary_tol)

    # ── Per-cell centroid matching ─────────────────────────────────────────────
    custom_cents  = compute_centroids(custom_mask)
    cosmx_cents   = compute_centroids(cosmx_data.cell_labels)

    match_df = match_cells_by_centroid(cosmx_cents, custom_cents, match_dist_px)
    match_df = compute_pairwise_iou(custom_mask, cosmx_data.cell_labels, match_df)

    n_custom     = len(custom_cents)
    n_cosmx      = len(cosmx_cents)
    n_matched    = int(match_df["matched"].sum())
    matched_iou  = match_df.loc[match_df["matched"], "iou"].dropna()
    matched_dist = match_df.loc[match_df["matched"], "centroid_dist"]

    # ── Area difference (matched pairs) ──────────────────────────────────────
    area_diffs = []
    if n_matched > 0:
        cosmx_area_map = cosmx_cents.set_index("label")["area"].to_dict()
        custom_area_map = custom_cents.set_index("label")["area"].to_dict()
        for _, row in match_df[match_df["matched"]].iterrows():
            ca = custom_area_map.get(int(row["query_label"]), float("nan"))
            ra = cosmx_area_map.get(int(row["ref_label"]), float("nan"))
            if not (np.isnan(ca) or np.isnan(ra)) and ra > 0:
                area_diffs.append((ca - ra) / ra)

    mean_area_diff = float(np.mean(area_diffs))  if area_diffs else float("nan")
    median_area_diff= float(np.median(area_diffs)) if area_diffs else float("nan")

    # ── False positive / negative area ────────────────────────────────────────
    fp_area = int((custom_bool & ~cosmx_cell).sum())   # custom, not in CosMx cell
    fn_area = int((cosmx_nuc   & ~custom_bool).sum())  # CosMx nucleus, not in custom

    return {
        # sanity
        "sanity": sane,
        # cell counts
        "custom_n_cells":     n_custom,
        "cosmx_n_cells":      n_cosmx,
        "matched_cells":      n_matched,
        "match_rate_custom":  n_matched / n_custom if n_custom > 0 else float("nan"),
        "match_rate_cosmx":   n_matched / n_cosmx  if n_cosmx  > 0 else float("nan"),
        # population IoU / Dice
        "nucleus_iou":        nucleus_iou,
        "nucleus_dice":       nucleus_dice,
        "cell_iou":           cell_iou,
        "cell_dice":          cell_dice,
        # per-cell statistics
        "mean_iou_matched":   float(matched_iou.mean())   if len(matched_iou) > 0 else float("nan"),
        "median_iou_matched": float(matched_iou.median()) if len(matched_iou) > 0 else float("nan"),
        "mean_centroid_dist": float(matched_dist.mean())   if len(matched_dist) > 0 else float("nan"),
        "median_centroid_dist":float(matched_dist.median()) if len(matched_dist) > 0 else float("nan"),
        # boundary F1
        "boundary_f1_vs_nucleus": bf1_nuc["boundary_f1"],
        "boundary_precision_vs_nucleus": bf1_nuc["boundary_precision"],
        "boundary_recall_vs_nucleus":    bf1_nuc["boundary_recall"],
        "boundary_f1_vs_cell": bf1_cell["boundary_f1"],
        # area difference
        "mean_area_diff_frac":   mean_area_diff,
        "median_area_diff_frac": median_area_diff,
        # false positive/negative
        "false_positive_px":  fp_area,
        "false_negative_px":  fn_area,
        # DataFrames
        "per_cell_df":        match_df,
        "custom_centroids":   custom_cents,
        "cosmx_centroids":    cosmx_cents,
    }
