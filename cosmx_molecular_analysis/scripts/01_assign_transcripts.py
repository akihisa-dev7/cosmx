#!/usr/bin/env python3
"""Step 1: Assign RNA transcripts to custom segmentation cells.

For FOVs covered by tx_file: coordinate-based mask lookup (true custom assignment).
For FOVs NOT in tx_file (e.g. FOV37/43): native exprMat + centroid mapping fallback.

Usage:
  python cosmx_molecular_analysis/scripts/01_assign_transcripts.py \\
    --tx-file raw_data/pilot_4fov/slide1_RNA/flatfiles/*_tx_file.csv.gz \\
    --exprmat raw_data/pilot_4fov/slide1_RNA/flatfiles/*_exprMat_file.csv.gz \\
    --metadata raw_data/pilot_4fov/slide1_RNA/flatfiles/*_metadata_file.csv.gz \\
    --mask-dir outputs/pilot_4fov/expanded_masks \\
    --cell-table-dir outputs/pilot_4fov/cell_tables \\
    --output outputs/cosmx_molecular_analysis/transcript_assignment \\
    --fovs FOV00001 FOV00007 FOV00037 FOV00043
"""
import argparse
import glob
import gzip
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from cosmx_molecular_analysis.src.io import (
    fov_name_to_int,
    load_tx_file,
    load_config,
    find_mask_file,
    load_mask,
)
from cosmx_molecular_analysis.src.transcript_assignment import (
    assign_all_fovs,
    build_counts_matrix,
    build_assignment_summary,
    assign_native_cells_to_custom_mask,
)


def parse_args():
    p = argparse.ArgumentParser(description="Assign RNA transcripts to segmentation mask")
    p.add_argument("--tx-file", nargs="+", required=True, help="*_tx_file.csv.gz")
    p.add_argument("--exprmat", nargs="+", default=None, help="*_exprMat_file.csv.gz (fallback)")
    p.add_argument("--metadata", nargs="+", default=None, help="*_metadata_file.csv.gz (fallback)")
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--cell-table-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--fovs", nargs="+", default=["FOV00001", "FOV00007", "FOV00037", "FOV00043"]
    )
    p.add_argument("--min-qv", type=float, default=None)
    p.add_argument("--exclude-prefixes", nargs="+", default=["NegPrb", "FalseCode"])
    p.add_argument("--config", default=None)
    return p.parse_args()


def resolve_glob(patterns):
    paths = []
    for pat in (patterns or []):
        expanded = glob.glob(pat)
        paths.extend(expanded if expanded else [pat])
    return paths


def load_gzip_csv(path, fovs=None):
    """Load CSV with optional FOV row filter using chunked reading."""
    if fovs is None:
        return pd.read_csv(path, compression="infer", low_memory=False)

    frames = []
    for chunk in pd.read_csv(path, compression="infer", chunksize=5000, low_memory=False):
        if "fov" in chunk.columns:
            chunk = chunk[chunk["fov"].isin(fovs)]
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    args = parse_args()

    if args.config:
        cfg = load_config(args.config)
        asgn = cfg.get("assignment", {})
        exclude = asgn.get("exclude_prefixes", args.exclude_prefixes)
        min_qv = asgn.get("min_qv", args.min_qv)
    else:
        exclude = args.exclude_prefixes
        min_qv = args.min_qv

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    fov_names = args.fovs
    fov_ints = [fov_name_to_int(f) for f in fov_names]

    # ── Step A: Load tx_file and determine which FOVs it covers ──
    tx_paths = resolve_glob(args.tx_file)
    print(f"Loading tx_file(s): {tx_paths}")
    tx_frames = [
        load_tx_file(p, fovs=fov_ints, exclude_prefixes=exclude, min_qv=min_qv)
        for p in tx_paths
    ]
    tx_df = pd.concat(tx_frames, ignore_index=True) if tx_frames else pd.DataFrame()

    tx_fov_ints = sorted(tx_df["fov"].unique()) if not tx_df.empty else []
    print(f"  tx_file FOVs available: {tx_fov_ints}  ({len(tx_df):,} transcripts)")

    tx_fov_names = [f for f, i in zip(fov_names, fov_ints) if i in tx_fov_ints]
    fallback_fov_names = [f for f, i in zip(fov_names, fov_ints) if i not in tx_fov_ints]

    count_frames = []
    assigned_frames = []

    # ── Step B: Custom tx assignment for FOVs with tx_file data ──
    if tx_fov_names:
        print(f"\n[Custom tx assignment] FOVs: {tx_fov_names}")
        assigned_df = assign_all_fovs(tx_df, args.mask_dir, tx_fov_names)
        assigned_frames.append(assigned_df)
        counts_tx = build_counts_matrix(assigned_df)
        count_frames.append(counts_tx)
    else:
        assigned_df = pd.DataFrame()

    # ── Step C: Native exprMat fallback for FOVs without tx_file ──
    if fallback_fov_names:
        print(f"\n[Native exprMat fallback] FOVs: {fallback_fov_names}")

        exprmat_paths = resolve_glob(args.exprmat)
        meta_paths = resolve_glob(args.metadata)

        if not exprmat_paths:
            print("  [WARN] No exprMat paths provided, skipping fallback")
        else:
            fallback_ints = [fov_name_to_int(f) for f in fallback_fov_names]

            native_counts_frames = [load_gzip_csv(p, fovs=fallback_ints) for p in exprmat_paths]
            native_counts = pd.concat(native_counts_frames, ignore_index=True)

            native_meta = pd.DataFrame()
            if meta_paths:
                meta_frames = [load_gzip_csv(p, fovs=fallback_ints) for p in meta_paths]
                native_meta = pd.concat(meta_frames, ignore_index=True)

            for fov_name in fallback_fov_names:
                fov_int = fov_name_to_int(fov_name)
                mask_path = find_mask_file(args.mask_dir, fov_name)
                if mask_path is None:
                    print(f"  [WARN] No mask for {fov_name}")
                    continue
                mask = load_mask(mask_path)
                fallback_counts = assign_native_cells_to_custom_mask(
                    native_meta, native_counts, mask, fov_int, fov_name
                )
                if not fallback_counts.empty:
                    count_frames.append(fallback_counts)

    # ── Step D: Merge all count frames ──
    if not count_frames:
        print("\n[ERROR] No counts produced for any FOV.")
        sys.exit(1)

    # Align columns (union of all genes)
    all_genes = sorted(set().union(*[set(df.columns) for df in count_frames]))
    aligned = [df.reindex(columns=all_genes, fill_value=0) for df in count_frames]
    counts_matrix = pd.concat(aligned, ignore_index=False)
    counts_matrix.index.name = "cell_id"
    print(f"\nFinal counts matrix: {counts_matrix.shape[0]} cells × {counts_matrix.shape[1]} genes")

    # ── Step E: Build summary ──
    if not assigned_frames:
        # No tx-based assignment → simple summary
        summary_rows = []
        for fov_name in fov_names:
            cells_in_fov = [c for c in counts_matrix.index if c.startswith(fov_name + "_")]
            mat = counts_matrix.loc[cells_in_fov] if cells_in_fov else pd.DataFrame()
            summary_rows.append({
                "fov_name": fov_name,
                "method": "native_fallback",
                "n_cells": len(cells_in_fov),
                "transcripts_per_cell_median": mat.sum(axis=1).median() if not mat.empty else 0,
                "genes_per_cell_median": (mat > 0).sum(axis=1).median() if not mat.empty else 0,
            })
        summary = pd.DataFrame(summary_rows)
    else:
        summary = build_assignment_summary(assigned_df, counts_matrix)
        # Add fallback rows
        for fov_name in fallback_fov_names:
            cells_in_fov = [c for c in counts_matrix.index if c.startswith(fov_name + "_")]
            mat = counts_matrix.loc[cells_in_fov] if cells_in_fov else pd.DataFrame()
            summary = pd.concat([
                summary,
                pd.DataFrame([{
                    "fov_name": fov_name,
                    "method": "native_fallback",
                    "n_cells": len(cells_in_fov),
                    "transcripts_per_cell_median": mat.sum(axis=1).median() if not mat.empty else 0,
                    "genes_per_cell_median": (mat > 0).sum(axis=1).median() if not mat.empty else 0,
                }])
            ], ignore_index=True)

    # ── Step F: Save ──
    counts_path = out_dir / "rna_counts_matrix.csv.gz"
    summary_path = out_dir / "assignment_summary.csv"

    print(f"\nSaving to {out_dir} ...")
    counts_matrix.to_csv(counts_path, compression="gzip")
    summary.to_csv(summary_path, index=False)

    if not assigned_frames:
        # Also save an empty assigned_transcripts for schema consistency
        assigned_skeleton = pd.DataFrame(columns=["fov", "target", "x_local_px", "y_local_px",
                                                    "fov_name", "custom_cell_id", "assigned"])
        assigned_skeleton.to_csv(out_dir / "assigned_transcripts.csv.gz", index=False, compression="gzip")
    else:
        assigned_df.to_csv(out_dir / "assigned_transcripts.csv.gz", index=False, compression="gzip")

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nOutputs:")
    print(f"  {counts_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
