#!/usr/bin/env python3
"""Step 2: Build AnnData from RNA counts matrix and cell metadata.

Usage:
  python cosmx_molecular_analysis/scripts/02_build_anndata.py \\
    --counts outputs/cosmx_molecular_analysis/transcript_assignment/rna_counts_matrix.csv.gz \\
    --cell-table-dir outputs/pilot_4fov/cell_tables \\
    --metadata raw_data/pilot_4fov/slide1_RNA/flatfiles/*_metadata_file.csv.gz \\
    --output outputs/cosmx_molecular_analysis/anndata/rna_custom_segmentation.h5ad \\
    --fovs FOV00001 FOV00007 FOV00037 FOV00043
"""
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from cosmx_molecular_analysis.src.anndata_builder import (
    load_cell_tables_for_fovs,
    build_anndata,
)
from cosmx_molecular_analysis.src.io import fov_name_to_int


def parse_args():
    p = argparse.ArgumentParser(description="Build AnnData from RNA counts")
    p.add_argument("--counts", required=True, help="Path to rna_counts_matrix.csv.gz")
    p.add_argument("--cell-table-dir", required=True, help="Directory with *_cells.csv")
    p.add_argument(
        "--metadata", nargs="+", default=None,
        help="Optional: path(s) to *_metadata_file.csv.gz"
    )
    p.add_argument(
        "--output", required=True, help="Output .h5ad path"
    )
    p.add_argument(
        "--fovs", nargs="+", default=["FOV00001", "FOV00007", "FOV00037", "FOV00043"]
    )
    p.add_argument("--px-size-um", type=float, default=0.12, help="Pixel size in µm")
    return p.parse_args()


def main():
    args = parse_args()
    fov_names = args.fovs

    print(f"Loading counts matrix: {args.counts}")
    counts = pd.read_csv(args.counts, index_col=0, compression="infer")
    print(f"  Shape: {counts.shape} ({counts.shape[0]} cells × {counts.shape[1]} genes)")

    print(f"Loading cell tables from: {args.cell_table_dir}")
    cell_tables = load_cell_tables_for_fovs(args.cell_table_dir, fov_names)
    print(f"  Cell tables: {len(cell_tables)} cells")

    if args.metadata:
        meta_paths = []
        for pat in args.metadata:
            expanded = glob.glob(pat)
            meta_paths.extend(expanded if expanded else [pat])

        fov_ints = [fov_name_to_int(f) for f in fov_names]
        meta_frames = []
        for mp in meta_paths:
            try:
                mf = pd.read_csv(mp, compression="infer")
                if "fov" in mf.columns:
                    mf = mf[mf["fov"].isin(fov_ints)]
                meta_frames.append(mf)
            except Exception as e:
                print(f"  [WARN] Could not load metadata {mp}: {e}")
        if meta_frames:
            metadata_df = pd.concat(meta_frames, ignore_index=True)
            print(f"  Native metadata: {len(metadata_df)} rows")
        else:
            metadata_df = pd.DataFrame()
    else:
        metadata_df = pd.DataFrame()

    print("\nBuilding AnnData...")
    adata = build_anndata(
        counts_matrix=counts,
        cell_tables=cell_tables,
        fov_names=fov_names,
        px_size_um=args.px_size_um,
    )
    print(f"  AnnData: {adata.n_obs} cells × {adata.n_vars} genes")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"\nSaved: {out_path}")
    print(f"  obs columns: {list(adata.obs.columns)}")


if __name__ == "__main__":
    main()
