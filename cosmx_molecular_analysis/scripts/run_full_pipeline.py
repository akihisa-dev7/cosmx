#!/usr/bin/env python3
"""Run the full CosMx molecular analysis pipeline.

Usage:
  python cosmx_molecular_analysis/scripts/run_full_pipeline.py \\
    --config cosmx_molecular_analysis/config/default_config.yaml
"""
import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path


def run_step(cmd: list, step_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {step_name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")


def parse_args():
    p = argparse.ArgumentParser(description="Run full CosMx molecular analysis pipeline")
    p.add_argument(
        "--config",
        default="cosmx_molecular_analysis/config/default_config.yaml",
    )
    p.add_argument(
        "--fovs", nargs="+",
        default=["FOV00001", "FOV00007", "FOV00037", "FOV00043"],
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip-steps", nargs="+", default=[], type=int,
                   help="Step numbers to skip (1-7)")
    return p.parse_args()


def main():
    args = parse_args()
    py = args.python
    cfg = args.config
    fovs = args.fovs
    skip = set(args.skip_steps)

    import yaml
    with open(cfg) as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})
    seg_root = paths.get("segmentation_root", "outputs/pilot_4fov")
    rna_flat = paths.get("rna_flatfiles", "raw_data/pilot_4fov/slide1_RNA/flatfiles")
    prot_flat = paths.get("protein_flatfiles", "raw_data/pilot_4fov/slide1_protein/flatfiles")
    out_root = paths.get("output_root", "outputs/cosmx_molecular_analysis")

    # Resolve glob patterns
    def first_glob(pattern):
        matches = glob.glob(pattern)
        return matches[0] if matches else pattern

    tx_file = first_glob(f"{rna_flat}/*_tx_file.csv.gz")
    prot_expr = first_glob(f"{prot_flat}/*_exprMat_file.csv.gz")

    mask_dir = f"{seg_root}/expanded_masks"
    cell_table_dir = f"{seg_root}/cell_tables"
    assign_dir = f"{out_root}/transcript_assignment"
    anndata_dir = f"{out_root}/anndata"
    spatial_dir = f"{out_root}/spatial"
    integ_dir = f"{out_root}/integration"
    markers_path = "cosmx_molecular_analysis/config/marker_genes.yaml"

    steps = {
        1: (
            "Step 1: Transcript assignment",
            [py, "cosmx_molecular_analysis/scripts/01_assign_transcripts.py",
             "--tx-file", tx_file,
             "--mask-dir", mask_dir,
             "--cell-table-dir", cell_table_dir,
             "--output", assign_dir,
             "--fovs"] + fovs + ["--config", cfg],
        ),
        2: (
            "Step 2: Build AnnData",
            [py, "cosmx_molecular_analysis/scripts/02_build_anndata.py",
             "--counts", f"{assign_dir}/rna_counts_matrix.csv.gz",
             "--cell-table-dir", cell_table_dir,
             "--output", f"{anndata_dir}/rna_custom_segmentation.h5ad",
             "--fovs"] + fovs,
        ),
        3: (
            "Step 3: QC filtering",
            [py, "cosmx_molecular_analysis/scripts/03_qc_filter.py",
             "--input", f"{anndata_dir}/rna_custom_segmentation.h5ad",
             "--output", f"{anndata_dir}/rna_qc_filtered.h5ad",
             "--config", cfg],
        ),
        4: (
            "Step 4: Normalize & cluster",
            [py, "cosmx_molecular_analysis/scripts/04_normalize_cluster.py",
             "--input", f"{anndata_dir}/rna_qc_filtered.h5ad",
             "--output", f"{anndata_dir}/rna_clustered.h5ad",
             "--config", cfg],
        ),
        5: (
            "Step 5: Cell type annotation",
            [py, "cosmx_molecular_analysis/scripts/05_cell_typing.py",
             "--input", f"{anndata_dir}/rna_clustered.h5ad",
             "--markers", markers_path,
             "--output", f"{anndata_dir}/rna_annotated.h5ad",
             "--config", cfg],
        ),
        6: (
            "Step 6: Spatial analysis",
            [py, "cosmx_molecular_analysis/scripts/06_spatial_analysis.py",
             "--input", f"{anndata_dir}/rna_annotated.h5ad",
             "--output", spatial_dir,
             "--config", cfg],
        ),
        7: (
            "Step 7: RNA–Protein integration",
            [py, "cosmx_molecular_analysis/scripts/07_integrate_rna_protein.py",
             "--rna", f"{anndata_dir}/rna_annotated.h5ad",
             "--protein", prot_expr,
             "--output", integ_dir,
             "--markers", markers_path,
             "--fovs"] + fovs,
        ),
    }

    print("CosMx Molecular Analysis Pipeline")
    print(f"Config: {cfg}")
    print(f"FOVs: {fovs}")

    t_start = time.time()
    for step_num in sorted(steps.keys()):
        if step_num in skip:
            print(f"\n[Skip] Step {step_num}")
            continue
        name, cmd = steps[step_num]
        run_step(cmd, name)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Outputs in: {out_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
