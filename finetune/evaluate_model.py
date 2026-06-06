"""
Evaluate fine-tuned StarDist model against ground truth annotations.

Computes F1 / Precision / Recall at IoU ≥ 0.5 threshold.
Produces a comparison image: DAPI + GT overlay vs. prediction overlay.

Usage (from CosMx_2026/):
    /home/fujiyamaakihisa/miniconda3/envs/cosmx_annotation/bin/python \
        finetune/evaluate_model.py \
        --model_path finetune/models/stardist_brain_v1

Options:
    --model_path   Path to fine-tuned model directory
    --gt_dir       Ground truth masks dir (default: annotation_tools/ground_truth/)
    --img_dir      Enhanced DAPI dir (default: outputs/pilot_4fov/enhanced/)
    --out_dir      Where to save comparison images (default: finetune/eval_results/)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_GT_DIR  = PROJECT_ROOT / "annotation_tools" / "ground_truth"
DEFAULT_IMG_DIR = PROJECT_ROOT / "outputs" / "pilot_4fov" / "enhanced"
DEFAULT_OUT_DIR = Path(__file__).parent / "eval_results"

GAUSSIAN_SIGMA = 5.0


def preprocess_dapi(img: np.ndarray) -> np.ndarray:
    blurred = gaussian_filter(img.astype(np.float32), sigma=GAUSSIAN_SIGMA)
    lo, hi = np.percentile(blurred, 1), np.percentile(blurred, 99)
    if hi == lo:
        return np.zeros_like(blurred)
    return np.clip((blurred - lo) / (hi - lo), 0, 1).astype(np.float32)


def match_iou(gt: np.ndarray, pred: np.ndarray, iou_thresh: float = 0.5):
    """Compute TP, FP, FN by matching GT and pred instances at given IoU."""
    gt_ids = np.unique(gt[gt > 0])
    pred_ids = np.unique(pred[pred > 0])

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 0, 0, 0

    matched_gt = set()
    matched_pred = set()

    for g in gt_ids:
        gt_mask = gt == g
        best_iou, best_p = 0.0, None
        for p in pred_ids:
            if p in matched_pred:
                continue
            pred_mask = pred == p
            inter = np.logical_and(gt_mask, pred_mask).sum()
            if inter == 0:
                continue
            union = np.logical_or(gt_mask, pred_mask).sum()
            iou = inter / union
            if iou > best_iou:
                best_iou, best_p = iou, p
        if best_iou >= iou_thresh:
            matched_gt.add(g)
            matched_pred.add(best_p)

    tp = len(matched_gt)
    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)
    return tp, fp, fn


def scores(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def make_overlay(dapi: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Create RGB overlay: grayscale DAPI + colored label borders."""
    from skimage.segmentation import find_boundaries
    gray = (np.clip(dapi, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    borders = find_boundaries(labels, mode="inner")
    # random color per label
    rng = np.random.default_rng(42)
    colors = {lbl: rng.integers(50, 255, 3) for lbl in np.unique(labels) if lbl > 0}
    for lbl, col in colors.items():
        mask = (labels == lbl)
        border = borders & mask
        rgb[border] = col
    return rgb


def _parse_fov(filename: str) -> str:
    m = re.match(r"(FOV\d{5})", filename)
    return m.group(1) if m else ""


def _parse_crop_tag(filename: str) -> tuple | None:
    m = re.search(r"_crop(\d+)-(\d+)-(\d+)-(\d+)", filename)
    if m:
        return tuple(int(x) for x in m.groups())
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned StarDist model")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--gt_dir",  type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--img_dir", type=Path, default=DEFAULT_IMG_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.model_path.exists():
        sys.exit(f"Model not found: {args.model_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from stardist.models import StarDist2D
    print(f"Loading model: {args.model_path}")
    model = StarDist2D(None, name=args.model_path.name, basedir=str(args.model_path.parent))

    gt_paths = sorted(args.gt_dir.glob("*_gt_mask*.tif"))
    if not gt_paths:
        sys.exit(f"No ground truth in {args.gt_dir}")

    all_tp, all_fp, all_fn = 0, 0, 0

    for gt_path in gt_paths:
        fov = _parse_fov(gt_path.name)
        crop = _parse_crop_tag(gt_path.name)

        dapi_matches = list(args.img_dir.glob(f"{fov}_*_tophat_enhanced.tif"))
        if not dapi_matches:
            continue
        dapi_raw = tifffile.imread(str(dapi_matches[0])).astype(np.float32)
        gt_mask  = tifffile.imread(str(gt_path)).astype(np.int32)

        if crop:
            y0, x0, h, w = crop
            dapi_raw = dapi_raw[y0:y0+h, x0:x0+w]

        dapi_proc = preprocess_dapi(dapi_raw)

        # Predict
        pred_labels, _ = model.predict_instances(
            dapi_proc,
            prob_thresh=model.thresholds.prob,
            nms_thresh=model.thresholds.nms,
            scale=0.25,
        )

        tp, fp, fn = match_iou(gt_mask, pred_labels.astype(np.int32))
        prec, rec, f1 = scores(tp, fp, fn)
        all_tp += tp; all_fp += fp; all_fn += fn

        print(f"\n{gt_path.name}")
        print(f"  GT cells: {gt_mask.max():4d}  Pred cells: {pred_labels.max():4d}")
        print(f"  TP={tp} FP={fp} FN={fn}  →  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

        # Save comparison image
        try:
            import imageio
            gt_overlay   = make_overlay(dapi_proc, gt_mask)
            pred_overlay = make_overlay(dapi_proc, pred_labels.astype(np.int32))
            comparison = np.concatenate([gt_overlay, pred_overlay], axis=1)
            out_path = args.out_dir / f"{fov}_comparison.png"
            imageio.imwrite(str(out_path), comparison)
            print(f"  Saved: {out_path.name}")
        except Exception as e:
            print(f"  (Could not save overlay: {e})")

    # Overall
    p, r, f1 = scores(all_tp, all_fp, all_fn)
    print(f"\n=== Overall ===")
    print(f"TP={all_tp} FP={all_fp} FN={all_fn}")
    print(f"Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")


if __name__ == "__main__":
    main()
