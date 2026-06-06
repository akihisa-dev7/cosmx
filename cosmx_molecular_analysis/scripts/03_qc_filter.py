#!/usr/bin/env python3
"""Step 3: QC filtering of AnnData.

Usage:
  python cosmx_molecular_analysis/scripts/03_qc_filter.py \\
    --input outputs/cosmx_molecular_analysis/anndata/rna_custom_segmentation.h5ad \\
    --output outputs/cosmx_molecular_analysis/anndata/rna_qc_filtered.h5ad
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anndata as ad
import pandas as pd

from cosmx_molecular_analysis.src.qc import filter_cells, save_qc_plots, QC_DEFAULTS
from cosmx_molecular_analysis.src.io import load_config


def parse_args():
    p = argparse.ArgumentParser(description="QC filter AnnData")
    p.add_argument("--input", required=True, help="Input .h5ad")
    p.add_argument("--output", required=True, help="Output .h5ad")
    p.add_argument("--config", default=None)
    p.add_argument("--min-transcripts", type=int, default=QC_DEFAULTS["min_transcripts_per_cell"])
    p.add_argument("--max-transcripts", type=int, default=QC_DEFAULTS["max_transcripts_per_cell"])
    p.add_argument("--min-genes", type=int, default=QC_DEFAULTS["min_genes_per_cell"])
    p.add_argument("--max-neg-fraction", type=float, default=QC_DEFAULTS["max_negative_probe_fraction"])
    p.add_argument("--min-area-um2", type=float, default=QC_DEFAULTS["min_cell_area_um2"])
    p.add_argument("--max-area-um2", type=float, default=QC_DEFAULTS["max_cell_area_um2"])
    return p.parse_args()


def main():
    args = parse_args()

    qc_params = {
        "min_transcripts": args.min_transcripts,
        "max_transcripts": args.max_transcripts,
        "min_genes": args.min_genes,
        "max_neg_fraction": args.max_neg_fraction,
        "min_area_um2": args.min_area_um2,
        "max_area_um2": args.max_area_um2,
    }

    if args.config:
        cfg = load_config(args.config)
        qc_cfg = cfg.get("qc", {})
        qc_params["min_transcripts"] = qc_cfg.get("min_transcripts_per_cell", args.min_transcripts)
        qc_params["max_transcripts"] = qc_cfg.get("max_transcripts_per_cell", args.max_transcripts)
        qc_params["min_genes"] = qc_cfg.get("min_genes_per_cell", args.min_genes)
        qc_params["max_neg_fraction"] = qc_cfg.get("max_negative_probe_fraction", args.max_neg_fraction)
        qc_params["min_area_um2"] = qc_cfg.get("min_cell_area_um2", args.min_area_um2)
        qc_params["max_area_um2"] = qc_cfg.get("max_cell_area_um2", args.max_area_um2)

    print(f"Loading: {args.input}")
    adata = ad.read_h5ad(args.input)
    print(f"  Input: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"  QC params: {qc_params}")

    adata_filt, summary = filter_cells(adata, **qc_params)
    print(f"  After QC: {adata_filt.n_obs} cells")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save QC table
    qc_cell_path = out_path.parent / "qc_cells.csv"
    adata_filt.obs.to_csv(qc_cell_path)

    qc_summary_path = out_path.parent / "qc_summary.csv"
    summary.to_csv(qc_summary_path, index=False)

    # Save QC plots
    plot_dir = out_path.parent.parent / "qc_plots"
    print(f"\nSaving QC plots to: {plot_dir}")
    save_qc_plots(adata, str(plot_dir))

    # Save filtered AnnData
    adata_filt.write_h5ad(out_path)

    print(f"\n=== QC Summary ===")
    print(summary.to_string(index=False))
    print(f"\nOutputs:")
    print(f"  {out_path}")
    print(f"  {qc_cell_path}")
    print(f"  {qc_summary_path}")


if __name__ == "__main__":
    main()
