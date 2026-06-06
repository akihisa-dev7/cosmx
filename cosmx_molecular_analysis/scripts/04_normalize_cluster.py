#!/usr/bin/env python3
"""Step 4: Normalize and cluster RNA AnnData.

Usage:
  python cosmx_molecular_analysis/scripts/04_normalize_cluster.py \\
    --input outputs/cosmx_molecular_analysis/anndata/rna_qc_filtered.h5ad \\
    --output outputs/cosmx_molecular_analysis/anndata/rna_clustered.h5ad
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anndata as ad

from cosmx_molecular_analysis.src.normalization import normalize_rna
from cosmx_molecular_analysis.src.clustering import run_clustering, save_umap_plots
from cosmx_molecular_analysis.src.io import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Normalize and cluster AnnData")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--rna-method", default="log1p")
    p.add_argument("--n-top-genes", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--resolution", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    params = {
        "rna_method": args.rna_method,
        "n_top_genes": args.n_top_genes,
        "n_pcs": args.n_pcs,
        "n_neighbors": args.n_neighbors,
        "resolution": args.resolution,
    }

    if args.config:
        cfg = load_config(args.config)
        norm_cfg = cfg.get("normalization", {})
        cl_cfg = cfg.get("clustering", {})
        params.update({
            "rna_method": norm_cfg.get("rna_method", args.rna_method),
            "n_top_genes": cl_cfg.get("n_top_genes", args.n_top_genes),
            "n_pcs": cl_cfg.get("n_pcs", args.n_pcs),
            "n_neighbors": cl_cfg.get("n_neighbors", args.n_neighbors),
            "resolution": cl_cfg.get("resolution", args.resolution),
        })

    print(f"Loading: {args.input}")
    adata = ad.read_h5ad(args.input)
    print(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    print(f"\nNormalizing ({params['rna_method']})...")
    adata = normalize_rna(adata, method=params["rna_method"])

    print(f"\nClustering (n_pcs={params['n_pcs']}, resolution={params['resolution']})...")
    adata = run_clustering(
        adata,
        n_top_genes=params["n_top_genes"],
        n_pcs=params["n_pcs"],
        n_neighbors=params["n_neighbors"],
        resolution=params["resolution"],
    )
    print(f"  Leiden clusters: {sorted(adata.obs['leiden'].unique())}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_dir = out_path.parent.parent / "clustering_plots"
    print(f"\nSaving UMAP plots to: {plot_dir}")
    save_umap_plots(adata, str(plot_dir))

    adata.write_h5ad(out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
