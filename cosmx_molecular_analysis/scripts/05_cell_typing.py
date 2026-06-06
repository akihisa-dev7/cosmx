#!/usr/bin/env python3
"""Step 5: Marker-based cell type annotation.

Usage:
  python cosmx_molecular_analysis/scripts/05_cell_typing.py \\
    --input outputs/cosmx_molecular_analysis/anndata/rna_clustered.h5ad \\
    --markers cosmx_molecular_analysis/config/marker_genes.yaml \\
    --output outputs/cosmx_molecular_analysis/anndata/rna_annotated.h5ad
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anndata as ad

from cosmx_molecular_analysis.src.cell_typing import (
    annotate_cell_types,
    build_cell_type_table,
    save_cell_type_plots,
)
from cosmx_molecular_analysis.src.io import load_marker_genes, load_config


def parse_args():
    p = argparse.ArgumentParser(description="Cell type annotation")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--markers",
        default="cosmx_molecular_analysis/config/marker_genes.yaml",
    )
    p.add_argument("--min-score-margin", type=float, default=0.15)
    p.add_argument("--low-confidence-label", default="LowConfidence")
    p.add_argument("--config", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    min_margin = args.min_score_margin
    lc_label = args.low_confidence_label

    if args.config:
        cfg = load_config(args.config)
        ct_cfg = cfg.get("cell_typing", {})
        min_margin = ct_cfg.get("min_score_margin", min_margin)
        lc_label = ct_cfg.get("low_confidence_label", lc_label)
        if "marker_genes_file" in ct_cfg:
            args.markers = ct_cfg["marker_genes_file"]

    print(f"Loading marker genes: {args.markers}")
    marker_genes = load_marker_genes(args.markers)
    print(f"  Cell types: {list(marker_genes.keys())}")

    print(f"Loading: {args.input}")
    adata = ad.read_h5ad(args.input)
    print(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    print(f"\nAnnotating cell types (min_margin={min_margin})...")
    adata = annotate_cell_types(adata, marker_genes, min_margin, lc_label)

    counts = adata.obs["predicted_cell_type"].value_counts()
    print("\n=== Cell type distribution ===")
    for ct, cnt in counts.items():
        print(f"  {ct}: {cnt} ({100*cnt/adata.n_obs:.1f}%)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Cell type table
    ct_table = build_cell_type_table(adata)
    ct_table_path = out_path.parent.parent / "cell_type_table.csv"
    ct_table.to_csv(ct_table_path, index=False)

    # Plots
    plot_dir = out_path.parent.parent / "cell_typing_plots"
    print(f"\nSaving plots to: {plot_dir}")
    save_cell_type_plots(adata, str(plot_dir), marker_genes)

    adata.write_h5ad(out_path)
    print(f"\nSaved: {out_path}")
    print(f"       {ct_table_path}")


if __name__ == "__main__":
    main()
