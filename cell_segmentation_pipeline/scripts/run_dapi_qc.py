#!/usr/bin/env python3
"""CLI: DAPI vs Mask Segmentation Accuracy QC.

Example::

    python cell_segmentation_pipeline/scripts/run_dapi_qc.py \\
        --output-root outputs/pilot_4fov \\
        --fovs FOV00001 FOV00007 FOV00037 FOV00043 \\
        --model cpsam \\
        --qc-output outputs/cell_segmentation_pipeline/qc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add CosMx_2026/ to path (2 levels up from scripts/)
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PIPELINE_DIR = _SCRIPT_DIR.parent          # cell_segmentation_pipeline/
_PROJECT_ROOT = _PIPELINE_DIR.parent        # CosMx_2026/

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PIPELINE_DIR / "app"))  # for utils

import numpy as np
import pandas as pd
import tifffile

from cell_segmentation_pipeline.src.dapi_qc import (
    run_fov_dapi_qc,
    save_qc_visualizations,
)


# ── Path helpers ──────────────────────────────────────────────────────────────

def _find_raw_dapi(fov_id: str) -> "np.ndarray | None":
    """Load raw DAPI from raw_data/pilot_4fov/slide1_RNA/morphology_images/."""
    num = fov_id[-5:]   # e.g. "00001"
    img_dir = _PROJECT_ROOT / "raw_data" / "pilot_4fov" / "slide1_RNA" / "morphology_images"
    candidates = list(img_dir.glob(f"*_F{num}.TIF")) + list(img_dir.glob(f"*_F{num}.tif"))
    if not candidates:
        return None
    img = tifffile.imread(str(candidates[0]))
    if img.ndim == 2:
        return img.astype(np.float32)
    # Multi-channel: take channel 0 (DAPI)
    return img[0].astype(np.float32)


def _find_enhanced(fov_id: str, output_root: str) -> "np.ndarray | None":
    root = Path(output_root)
    candidates = sorted((root / "enhanced").glob(f"{fov_id}*enhanced*.tif"))
    if not candidates:
        return None
    img = tifffile.imread(str(candidates[0]))
    return (img[0] if img.ndim == 3 else img).astype(np.float32)


def _find_mask(fov_id: str, model: str, output_root: str) -> "np.ndarray | None":
    root = Path(output_root)
    candidates = sorted((root / "masks").glob(f"{fov_id}*{model}*nuclear_mask.tif"))
    if not candidates:
        return None
    return tifffile.imread(str(candidates[0])).astype(np.int32)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DAPI vs Mask Segmentation Accuracy QC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-root",
        default="outputs/pilot_4fov",
        help="Root directory for enhanced/ and masks/ subdirectories",
    )
    p.add_argument(
        "--fovs",
        nargs="+",
        default=["FOV00001", "FOV00007", "FOV00037", "FOV00043"],
        help="List of FOV IDs to process",
    )
    p.add_argument(
        "--model",
        default="cpsam",
        help="Model key used in mask filenames (e.g. cpsam, instanseg, stardist)",
    )
    p.add_argument(
        "--qc-output",
        default="outputs/cell_segmentation_pipeline/qc",
        help="Output directory for QC results",
    )
    p.add_argument("--min-area",    type=int,   default=50,    help="Min mask cell area (px²)")
    p.add_argument("--max-area",    type=int,   default=10000, help="Max mask cell area (px²)")
    p.add_argument("--iou-thresh",  type=float, default=0.3,   help="IoU threshold for match")
    p.add_argument("--max-dim",     type=int,   default=1024,  help="Max image dim for visualizations")
    p.add_argument("--no-vis",      action="store_true",        help="Skip saving visualizations")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Resolve paths relative to project root
    output_root = str(_PROJECT_ROOT / args.output_root) \
        if not Path(args.output_root).is_absolute() else args.output_root
    qc_output   = str(_PROJECT_ROOT / args.qc_output) \
        if not Path(args.qc_output).is_absolute() else args.qc_output

    qc_out_path = Path(qc_output)
    qc_out_path.mkdir(parents=True, exist_ok=True)

    print(f"Output root : {output_root}")
    print(f"QC output   : {qc_output}")
    print(f"FOVs        : {args.fovs}")
    print(f"Model       : {args.model}")
    print()

    summary_rows: list[dict] = []
    per_cell_dfs: list[pd.DataFrame] = []

    for fov_id in args.fovs:
        print(f"[{fov_id}] Loading images...")

        raw = _find_raw_dapi(fov_id)
        if raw is None:
            print(f"  WARNING: raw DAPI not found for {fov_id}, skipping.")
            continue

        enhanced = _find_enhanced(fov_id, output_root)
        if enhanced is None:
            print(f"  WARNING: enhanced image not found for {fov_id}, skipping.")
            continue

        mask = _find_mask(fov_id, args.model, output_root)
        if mask is None:
            print(f"  WARNING: mask not found for {fov_id} (model={args.model}), skipping.")
            continue

        print(f"  raw={raw.shape}, enhanced={enhanced.shape}, mask={mask.shape}, "
              f"mask_cells={int(mask.max())}")

        print(f"  Running QC...")
        qc = run_fov_dapi_qc(
            raw=raw,
            enhanced=enhanced,
            mask=mask,
            fov_id=fov_id,
            model_name=args.model,
            min_area_px=args.min_area,
            max_area_px=args.max_area,
            iou_threshold=args.iou_thresh,
        )

        # Summary row (scalar metrics only)
        row = {k: v for k, v in qc.items()
               if not isinstance(v, (pd.DataFrame, np.ndarray))}
        summary_rows.append(row)

        # Per-cell DataFrame
        pcdf = qc["per_cell_df"].copy()
        pcdf.insert(0, "fov_id", fov_id)
        pcdf.insert(1, "model_name", args.model)
        per_cell_dfs.append(pcdf)

        # Visualizations
        if not args.no_vis:
            print(f"  Saving visualizations...")
            vis_dir = str(qc_out_path / "visualizations")
            save_qc_visualizations(
                raw=raw,
                enhanced=enhanced,
                mask=mask,
                dapi_labels=qc["dapi_labels"],
                qc_result=qc,
                output_dir=vis_dir,
                fov_id=fov_id,
                model_name=args.model,
                max_dim=args.max_dim,
            )

        # Print per-FOV summary
        prec = qc.get("estimated_precision", float("nan"))
        rec  = qc.get("estimated_recall",    float("nan"))
        miou = qc.get("mean_iou",            float("nan"))
        ldf  = qc.get("low_dapi_mask_fraction", float("nan"))
        print(
            f"  cells={qc['cell_count']}  "
            f"precision={prec:.3f}  recall={rec:.3f}  "
            f"mean_iou={miou:.3f}  low_dapi={ldf:.3f}"
        )

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = qc_out_path / "dapi_mask_qc_summary.csv"
        summary_df.to_csv(str(summary_csv), index=False)
        print(f"\nSummary CSV  -> {summary_csv}")

        # Print table
        display_cols = [
            "fov_id", "model_name", "cell_count", "dapi_object_count",
            "estimated_precision", "estimated_recall", "mean_iou",
            "low_dapi_mask_fraction", "extra_mask_count",
        ]
        show = [c for c in display_cols if c in summary_df.columns]
        print()
        print(summary_df[show].to_string(index=False))

    if per_cell_dfs:
        per_cell_df = pd.concat(per_cell_dfs, ignore_index=True)
        per_cell_csv = qc_out_path / "dapi_mask_qc_per_cell.csv"
        per_cell_df.to_csv(str(per_cell_csv), index=False)
        print(f"Per-cell CSV -> {per_cell_csv}")

    if not summary_rows:
        print("ERROR: No FOVs processed successfully.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
