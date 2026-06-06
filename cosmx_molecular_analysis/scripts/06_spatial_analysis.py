#!/usr/bin/env python3
"""Step 6: Spatial analysis – cell type maps, density, composition.

Usage:
  python cosmx_molecular_analysis/scripts/06_spatial_analysis.py \\
    --input outputs/cosmx_molecular_analysis/anndata/rna_annotated.h5ad \\
    --output outputs/cosmx_molecular_analysis/spatial
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anndata as ad
import pandas as pd

from cosmx_molecular_analysis.src.spatial_analysis import (
    save_spatial_cell_type_maps,
    compute_cell_type_composition_by_fov,
    compute_cell_density_by_fov,
    compute_neighborhood_summary,
)
from cosmx_molecular_analysis.src.io import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Spatial analysis")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--neighbor-radius-px", type=float, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    radius = args.neighbor_radius_px
    if args.config:
        cfg = load_config(args.config)
        sp_cfg = cfg.get("spatial", {})
        radius = sp_cfg.get("neighbor_radius_px", radius)

    print(f"Loading: {args.input}")
    adata = ad.read_h5ad(args.input)
    print(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    # 1. Spatial cell type maps per FOV
    fov_names = None
    if "fov_name" in adata.obs.columns:
        fov_names = sorted(adata.obs["fov_name"].unique())

    print(f"\nSaving spatial cell type maps ({fov_names})...")
    save_spatial_cell_type_maps(adata, str(out_dir), fov_names=fov_names)

    # 2. Cell type composition by FOV
    print("\nComputing cell type composition by FOV...")
    composition = compute_cell_type_composition_by_fov(adata)
    if not composition.empty:
        comp_path = out_dir / "cell_type_composition_by_fov.csv"
        composition.to_csv(comp_path, index=False)
        print(f"  Saved: {comp_path}")

    # 3. Cell density by FOV
    print("\nComputing cell density by FOV...")
    density = compute_cell_density_by_fov(adata)
    if not density.empty:
        density_path = out_dir / "cell_density_by_fov.csv"
        density.to_csv(density_path, index=False)
        print(f"  Saved: {density_path}")

    # 4. Neighborhood summary
    print(f"\nComputing neighborhood summary (radius={radius}px)...")
    neighborhood = compute_neighborhood_summary(adata, radius_px=radius)
    if not neighborhood.empty:
        nbr_path = out_dir / "neighborhood_summary.csv"
        neighborhood.to_csv(nbr_path, index=False)
        print(f"  Saved: {nbr_path}")

    # 5. AD vs Control composition
    if "condition" in adata.obs.columns and "predicted_cell_type" in adata.obs.columns:
        print("\nComputing condition-wise composition...")
        cond_comp = (
            adata.obs.groupby(["condition", "predicted_cell_type"])
            .size()
            .unstack(fill_value=0)
        )
        cond_comp = cond_comp.div(cond_comp.sum(axis=1), axis=0)
        cond_path = out_dir / "cell_type_composition_by_condition.csv"
        cond_comp.to_csv(cond_path)
        print(f"  Saved: {cond_path}")

    print("\nSpatial analysis complete.")


if __name__ == "__main__":
    main()
