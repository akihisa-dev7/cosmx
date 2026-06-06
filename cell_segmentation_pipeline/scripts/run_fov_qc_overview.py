#!/usr/bin/env python3
"""Generate fov_qc_overview.csv/json from pre-computed QC CSVs.

Usage:
  python cell_segmentation_pipeline/scripts/run_fov_qc_overview.py \
    --qc-dir outputs/cell_segmentation_pipeline/qc \
    --per-fov-root raw_data/pilot_4fov/slide1_RNA/per_fov_decoded \
    --seg-root outputs/pilot_4fov \
    --model cpsam
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import tifffile
import glob as _glob

from cell_segmentation_pipeline.src.composite_qc_score import (
    compute_extended_match_metrics,
    build_fov_overview_row,
)
from cell_segmentation_pipeline.src.label_based_qc import (
    CosMxLabelData,
    compute_centroids,
    find_per_fov_dir,
    load_compartment_labels,
    load_cell_labels,
    load_cell_stats,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--qc-dir",       default="outputs/cell_segmentation_pipeline/qc")
    p.add_argument("--per-fov-root", default="raw_data/pilot_4fov/slide1_RNA/per_fov_decoded")
    p.add_argument("--seg-root",     default="outputs/pilot_4fov")
    p.add_argument("--model",        default="cpsam")
    p.add_argument("--fovs", nargs="+", default=["FOV00001","FOV00007","FOV00037","FOV00043"])
    p.add_argument("--match-dist-px", type=float, default=30.0)
    return p.parse_args()


def main():
    args = parse_args()
    qc_dir = Path(args.qc_dir)

    # Load pre-computed summaries
    dapi_summary_df  = None
    label_summary_df = None
    label_per_cell_df = None

    dapi_csv  = qc_dir / "dapi_mask_qc_summary.csv"
    label_csv = qc_dir / "label_qc_summary.csv"
    per_cell_csv = qc_dir / "label_qc_per_cell.csv"

    if dapi_csv.exists():
        dapi_summary_df  = pd.read_csv(dapi_csv)
        # Normalise FOV column name
        if "fov_id" in dapi_summary_df.columns:
            dapi_summary_df = dapi_summary_df.rename(columns={"fov_id": "fov"})
    if label_csv.exists():
        label_summary_df = pd.read_csv(label_csv)
    if per_cell_csv.exists():
        label_per_cell_df = pd.read_csv(per_cell_csv)

    rows = []
    for fov_name in args.fovs:
        print(f"Processing {fov_name}...")

        dapi_row  = None
        label_row = None

        if dapi_summary_df is not None:
            sub = dapi_summary_df[dapi_summary_df["fov"] == fov_name]
            if not sub.empty:
                dapi_row = sub.iloc[0]

        if label_summary_df is not None:
            sub = label_summary_df[label_summary_df["fov"] == fov_name]
            if not sub.empty:
                label_row = sub.iloc[0]

        # Compute extended metrics from per-cell data + centroid positions
        extended = None
        if label_per_cell_df is not None:
            pc = label_per_cell_df[label_per_cell_df["fov"] == fov_name].copy()

            # Load centroids from masks
            custom_cents = None
            cosmx_cents  = None

            mask_files = _glob.glob(
                f"{args.seg_root}/masks/{fov_name}*{args.model}*nuclear_mask.tif"
            )
            if mask_files:
                custom_mask = tifffile.imread(mask_files[0]).astype("int32")
                custom_cents = compute_centroids(custom_mask)

            fov_dir = find_per_fov_dir(fov_name, args.per_fov_root)
            if fov_dir:
                cell_lbl = load_cell_labels(fov_dir)
                if cell_lbl is not None:
                    cosmx_cents = compute_centroids(cell_lbl)

            extended = compute_extended_match_metrics(
                pc, custom_cents, cosmx_cents, args.match_dist_px
            )
            print(
                f"  Extended: matched={extended['n_matched']} "
                f"reg_shift={extended['registration_shift_px']:.1f}px "
                f"ratio={extended['custom_to_cosmx_ratio']:.2f}"
            )

        row = build_fov_overview_row(
            fov_name=fov_name,
            model_name=args.model,
            dapi_summary=dapi_row,
            label_summary=label_row,
            extended_metrics=extended,
        )
        rows.append(row)
        print(
            f"  {fov_name}: DAPI={row.get('dapi_status')} "
            f"score={row.get('composite_score','?')} "
            f"action={row.get('recommended_action','?')}"
        )

    overview_df = pd.DataFrame(rows)
    csv_path  = qc_dir / "fov_qc_overview.csv"
    json_path = qc_dir / "fov_qc_overview.json"

    overview_df.to_csv(csv_path, index=False)
    overview_df.to_json(json_path, orient="records", indent=2, force_ascii=False)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print()
    print("=== FOV QC Overview ===")
    disp_cols = [
        "fov", "dapi_status", "custom_cell_count", "cosmx_cell_count",
        "custom_to_cosmx_ratio", "registration_status",
        "matched_pair_iou_median", "one_to_one_match_rate",
        "composite_score", "recommended_action",
    ]
    show_cols = [c for c in disp_cols if c in overview_df.columns]
    print(overview_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
