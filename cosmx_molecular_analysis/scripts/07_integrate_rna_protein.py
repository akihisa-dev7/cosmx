#!/usr/bin/env python3
"""Step 7: FOV-level RNA–Protein integration.

Usage:
  python cosmx_molecular_analysis/scripts/07_integrate_rna_protein.py \\
    --rna outputs/cosmx_molecular_analysis/anndata/rna_annotated.h5ad \\
    --protein raw_data/pilot_4fov/slide1_protein/flatfiles/*_exprMat_file.csv.gz \\
    --output outputs/cosmx_molecular_analysis/integration
"""
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anndata as ad
import pandas as pd

from cosmx_molecular_analysis.src.integration import (
    build_rna_fov_summary,
    build_protein_fov_summary,
    merge_rna_protein_fov,
    compute_rna_protein_correlation,
    save_correlation_plots,
)
from cosmx_molecular_analysis.src.io import (
    load_marker_genes,
    fov_name_to_int,
    fov_int_to_name,
)


def parse_args():
    p = argparse.ArgumentParser(description="RNA–Protein FOV-level integration")
    p.add_argument("--rna", required=True, help="rna_annotated.h5ad")
    p.add_argument("--protein", required=True, nargs="+", help="*_exprMat_file.csv.gz")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--markers",
        default="cosmx_molecular_analysis/config/marker_genes.yaml",
    )
    p.add_argument("--fovs", nargs="+", default=["FOV00001", "FOV00007", "FOV00037", "FOV00043"])
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    fov_names = args.fovs
    fov_ints = [fov_name_to_int(f) for f in fov_names]
    fov_int_name_map = {i: n for i, n in zip(fov_ints, fov_names)}

    marker_genes = load_marker_genes(args.markers)

    print(f"Loading RNA AnnData: {args.rna}")
    adata = ad.read_h5ad(args.rna)
    print(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    print("\nBuilding RNA FOV summary...")
    rna_fov = build_rna_fov_summary(adata, marker_genes)
    print(f"  {len(rna_fov)} FOVs")

    # Load protein
    prot_paths = []
    for pat in args.protein:
        expanded = glob.glob(pat)
        prot_paths.extend(expanded if expanded else [pat])

    print(f"\nLoading protein data: {prot_paths}")
    prot_frames = [pd.read_csv(p, compression="infer") for p in prot_paths]
    prot_df = pd.concat(prot_frames, ignore_index=True)
    prot_df = prot_df[prot_df["fov"].isin(fov_ints)]
    print(f"  Protein cells: {len(prot_df)} across FOVs {sorted(prot_df['fov'].unique())}")

    print("\nBuilding protein FOV summary...")
    protein_fov = build_protein_fov_summary(prot_df, fov_int_name_map, fovs=fov_ints)

    print("\nMerging RNA–Protein FOV summaries...")
    joint = merge_rna_protein_fov(rna_fov, protein_fov)
    joint_path = out_dir / "joint_fov_summary.csv"
    joint.to_csv(joint_path, index=False)
    print(f"  Saved: {joint_path}")

    print("\nComputing RNA–Protein correlations...")
    corr_df = compute_rna_protein_correlation(joint)
    corr_path = out_dir / "rna_protein_correlation.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"  Saved: {corr_path}")
    print(corr_df[["rna_marker", "protein_marker", "pearson_r", "n_fovs"]].to_string(index=False))

    print("\nSaving correlation scatter plots...")
    save_correlation_plots(joint, corr_df, str(out_dir))

    print("\nIntegration complete.")


if __name__ == "__main__":
    main()
