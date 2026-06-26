#!/usr/bin/env python3
"""CLI: segment enhanced CosMx images and generate masks + cell tables.

Example::

    python cell_segmentation_pipeline/scripts/run_segmentation.py \\
        --input  outputs/cell_segmentation_pipeline/enhanced \\
        --output outputs/cell_segmentation_pipeline \\
        --model  cellpose_cpsam \\
        --diameter 25 --cellprob_threshold -1.0 --flow_threshold 0.6 \\
        --expand_px 5

Notes:
    - --input can be either a directory of enhanced TIFFs (from run_preprocess.py)
      OR the original morphology_images directory (raw DAPI used directly).
    - --manifest can point to the project sample_manifest.csv for metadata tagging.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cell_segmentation_pipeline.src import io as _io
from cell_segmentation_pipeline.src.masks import (
    expand_mask,
    expand_mask_membrane_guided,
    mask_to_cell_table,
    save_colored_mask_png,
)
from cell_segmentation_pipeline.src.qc import (
    compute_fov_qc,
    save_qc_panel,
    plot_size_distribution,
    aggregate_qc,
    plot_cellcount_comparison,
)
from cell_segmentation_pipeline.src.segmentation import get_segmenter
from cell_segmentation_pipeline.src.visualization import save_overlay_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segment CosMx DAPI images.")
    p.add_argument("--input",   required=True, help="Directory of (enhanced) TIFF images")
    p.add_argument("--output",  required=True, help="Root output directory")
    p.add_argument("--model",   default="cellpose_cpsam",
                   help="Segmentation model (cellpose_cpsam | cellpose_nuclei | "
                        "cellpose_cyto3 | stardist | instanseg)")
    p.add_argument("--diameter",           type=float, default=25.0)
    p.add_argument("--cellprob_threshold", type=float, default=-1.0)
    p.add_argument("--flow_threshold",     type=float, default=0.6)
    p.add_argument("--expand_px",          type=int,   default=5,
                   help="Mask expansion distance in px (default: 5)")
    p.add_argument("--min_area_px",        type=int,   default=50,
                   help="Minimum cell area in px² (default: 50)")
    p.add_argument("--channel",            type=int,   default=0,
                   help="Input channel (default: 0 = DAPI). "
                        "Ignored if --input contains single-channel TIFFs.")
    p.add_argument("--membrane_channel",   type=int,   default=None,
                   help="Membrane channel index for guided expansion (default: None=disabled). "
                        "Use 3 for CosMx 5-channel TIFs. Requires raw morphology_images input.")
    p.add_argument("--manifest", default=None,
                   help="Path to sample_manifest.csv for metadata tagging")
    p.add_argument("--fov", default=None, help="Single FOV ID to process (default: all)")
    # Baysor-specific
    p.add_argument("--tx_csv", default=None,
                   help="Transcript CSV (.csv or .csv.gz) — required for --model baysor")
    p.add_argument("--prior_mask", default=None,
                   help="Prior nuclear mask TIFF for Baysor (auto-detected from cpsam output if omitted)")
    p.add_argument("--scale_um", type=float, default=10.0,
                   help="Expected cell radius in µm (Baysor, default: 10.0)")
    p.add_argument("--min_molecules", type=int, default=15,
                   help="Min transcripts per cell (Baysor, default: 15)")
    p.add_argument("--prior_confidence", type=float, default=0.5,
                   help="DAPI prior weight 0–1 (Baysor, default: 0.5)")
    p.add_argument("--max_transcripts", type=int, default=500_000,
                   help="Max transcripts per FOV passed to Baysor (0=no limit). "
                        "Reduces Julia GC segfaults on large datasets (default: 500000)")
    p.add_argument("--baysor_refine", action="store_true",
                   help="After DAPI segmentation, refine with Baysor RNA-guided segmentation")
    return p.parse_args()


def _baysor_refine_fov(
    args, cfg: dict, fov_id: str, fov_num: int,
    prior_mask_path: Path, image_shape: tuple, out: Path,
) -> "np.ndarray | None":
    """Run Baysor RNA-guided refinement on one FOV using an existing DAPI mask as prior.

    Returns the refined integer-label mask, or None if refinement fails.
    """
    import tifffile as _tifffile
    from cell_segmentation_pipeline.src.segmentation.baysor_segmenter import BaysorSegmenter

    tx_path = Path(args.tx_csv) if args.tx_csv else None
    if tx_path is None or not tx_path.is_file():
        print(f"  [{fov_id}] WARNING: --tx_csv not a valid file, skipping Baysor refinement",
              file=sys.stderr)
        return None

    b_cfg = cfg["segmentation"]["baysor"]

    # Use DAPI cell count as n_cells_init so EM starts close to the expected
    # number of cells.  n_cells_init=500 with 1.5M transcripts → 3000 tx/cell
    # average → Baysor aggressively splits cells every iteration → split_ids
    # crashes in Julia at bmm_algorithm.jl:184 with exit code 1.
    # Starting with ~8400 components (matching the prior) → ~178 tx/cell →
    # far fewer splits → stable EM convergence.
    try:
        _prior_arr = _tifffile.imread(str(prior_mask_path))
        _n_cells_prior = int(_prior_arr.max())
        del _prior_arr
        print(f"  [{fov_id}] Prior mask: {_n_cells_prior} DAPI cells → n_cells_init={_n_cells_prior}")
    except Exception:
        _n_cells_prior = 500

    seg = BaysorSegmenter(
        scale_um=b_cfg["scale_um"],
        min_molecules=b_cfg["min_molecules"],
        prior_confidence=b_cfg["prior_confidence"],
        pixel_size_um=b_cfg.get("pixel_size_um", 0.12028),
        max_transcripts=b_cfg.get("max_transcripts", 500_000),
        n_cells_init=_n_cells_prior,
    )

    # Free PyTorch / CUDA memory before loading 4M+ transcript rows so that
    # the large pandas allocation doesn't trigger a SIGSEGV in the Python process.
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass

    # Include the segmenter name in the output directory so that StarDist and
    # Cellpose Baysor refinements don't share the same directory.  Sharing
    # caused exit code 1 crashes: Baysor found stale segmentation.csv /
    # segmentation_params.dump.toml from the previous model's run and tried
    # to use them, creating a state mismatch with the new n_cells_init.
    _prior_stem = prior_mask_path.stem  # e.g. "FOV00001_AD_F_stardist_g5_s025_nuclear_mask"
    baysor_out = out / "baysor" / fov_id / _prior_stem
    print(f"  [{fov_id}] Baysor RNA-guided refinement …")
    try:
        mask_path = seg.run_fov(
            tx_csv=tx_path,
            prior_mask=prior_mask_path,
            out_dir=baysor_out,
            fov_col="fov",
            fov_id=fov_num,
            image_shape=image_shape,
        )
        return _tifffile.imread(str(mask_path)).astype(np.int32)
    except Exception as exc:
        msg = f"  [{fov_id}] WARNING: Baysor refinement failed: {exc}"
        print(msg, file=sys.stderr)
        # Also print to stdout so the Runner progress panel shows the failure
        print(f"[BAYSOR_FAILED] {fov_id}: {exc}")
        return None


def _run_baysor(args, cfg, out: Path, tif_paths: list, manifest) -> None:
    """Baysor RNA-guided segmentation pipeline (transcript CSV → mask → QC)."""
    import tifffile as _tifffile
    from cell_segmentation_pipeline.src.segmentation.baysor_segmenter import BaysorSegmenter

    if not args.tx_csv:
        print("ERROR: --tx_csv is required for --model baysor", file=sys.stderr)
        sys.exit(1)
    tx_path = Path(args.tx_csv)
    if not tx_path.exists() or not tx_path.is_file():
        print(f"ERROR: tx_csv not found or is not a file: {tx_path}", file=sys.stderr)
        sys.exit(1)

    b_cfg = cfg.get("segmentation", {}).get("baysor", {})
    seg = BaysorSegmenter(
        scale_um=b_cfg.get("scale_um", 10.0),
        min_molecules=b_cfg.get("min_molecules", 15),
        prior_confidence=b_cfg.get("prior_confidence", 0.5),
        pixel_size_um=b_cfg.get("pixel_size_um", 0.12028),
        max_transcripts=b_cfg.get("max_transcripts", 500_000),
    )
    print(f"Segmenter: baysor  (scale_um={seg.scale_um}, min_mol={seg.min_molecules}, "
          f"prior_conf={seg.prior_confidence}, max_tx={seg.max_transcripts:,})")

    print(f"Processing {len(tif_paths)} FOV(s) with Baysor …")
    qc_results = []

    for tif_path in tif_paths:
        fov_id  = _io.parse_fov_id(tif_path)
        img_all = _io.load_tiff(tif_path)
        raw     = img_all[args.channel] if img_all.ndim == 3 else img_all

        fov_num = int("".join(c for c in fov_id if c.isdigit()) or "0")

        fov_meta = {}
        if manifest is not None:
            fov_meta = _io.get_fov_meta(manifest, fov_id)
        tag = fov_id
        if fov_meta.get("condition"):
            tag += f"_{fov_meta['condition']}"
        if fov_meta.get("region"):
            tag += f"_{fov_meta['region']}"

        # Resolve prior mask: explicit arg → auto-detect cpsam (DAPI-only) → None
        prior_mask_path = None
        if args.prior_mask and Path(args.prior_mask).exists():
            prior_mask_path = Path(args.prior_mask)
        else:
            hits = [
                p for p in (out / "masks").glob(f"{fov_id}*cpsam*nuclear_mask.tif")
                if "baysor" not in p.name
            ]
            if hits:
                prior_mask_path = hits[0]
                print(f"  [{fov_id}] Prior mask: {prior_mask_path.name}")
            else:
                print(f"  [{fov_id}] No prior mask found — running without DAPI prior")

        baysor_out = out / "baysor" / fov_id
        print(f"  [{fov_id}] Baysor RNA-guided segmentation …")
        mask_path = seg.run_fov(
            tx_csv=tx_path,
            prior_mask=prior_mask_path,
            out_dir=baysor_out,
            fov_col="fov",
            fov_id=fov_num,
            image_shape=raw.shape[:2],
        )

        mask = _tifffile.imread(str(mask_path)).astype(np.int32)
        n_cells = int(mask.max())
        print(f"  [{fov_id}] → {n_cells} cells")

        _io.save_mask_tiff(
            mask.astype(np.uint16),
            out / "masks" / f"{tag}_baysor_nuclear_mask.tif",
        )

        exp_mask = expand_mask(mask, expand_px=args.expand_px, min_area_px=args.min_area_px)
        _io.save_mask_tiff(
            exp_mask.astype(np.uint16),
            out / "expanded_masks" / f"{tag}_baysor_expanded_mask.tif",
        )

        cell_table = mask_to_cell_table(mask, raw, fov_meta)
        csv_path   = out / "cell_tables" / f"{tag}_baysor_cells.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        cell_table.to_csv(str(csv_path), index=False)

        save_overlay_png(raw, mask, out / "overlays" / f"{tag}_baysor_overlay.png")
        save_colored_mask_png(mask, out / "overlays" / f"{tag}_baysor_colored.png")

        save_qc_panel(
            raw, raw, mask, exp_mask,
            out / "qc" / "panels" / f"{tag}_baysor_qc_panel.png",
            fov_id, "baysor",
        )
        plot_size_distribution(
            cell_table,
            out / "qc" / "panels" / f"{tag}_baysor_size_dist.png",
            fov_id, "baysor",
        )

        qc = compute_fov_qc(cell_table, cfg, fov_id=fov_id, model_name="baysor")
        qc_results.append(qc)
        med = qc.get("median_diam_px") if qc else None
        print(f"  [{fov_id}] done  n_cells={n_cells}  median_diam="
              f"{f'{med:.1f}' if med is not None else 'N/A'}px")

    summary = aggregate_qc(qc_results)
    qc_dir  = out / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(str(qc_dir / "qc_summary_baysor.csv"), index=False)
    plot_cellcount_comparison(summary, qc_dir / "cellcount_comparison_baysor.png")
    print(f"\nBaysor summary:\n{summary.to_string(index=False)}")
    print(f"\nQC summary → {qc_dir / 'qc_summary_baysor.csv'}")


def main() -> None:
    args   = parse_args()
    in_dir = Path(args.input)
    out    = Path(args.output)

    manifest = None
    if args.manifest and Path(args.manifest).exists():
        manifest = _io.load_manifest(args.manifest)
    elif Path("config/sample_manifest.csv").exists():
        manifest = _io.load_manifest("config/sample_manifest.csv")

    # Build a minimal config to reuse get_segmenter
    cfg = {
        "input": {"channel": args.channel},
        "segmentation": {
            "default_model": args.model,
            "cellpose": {
                "cpsam":   {"diameter": args.diameter, "cellprob_threshold": args.cellprob_threshold, "flow_threshold": args.flow_threshold},
                "nuclei":  {"diameter": args.diameter, "cellprob_threshold": args.cellprob_threshold, "flow_threshold": args.flow_threshold},
                "cyto3":   {"diameter": args.diameter, "cellprob_threshold": args.cellprob_threshold, "flow_threshold": args.flow_threshold},
            },
            "stardist": {},
            "instanseg": {},
            "baysor": {
                "scale_um":          args.scale_um,
                "min_molecules":     args.min_molecules,
                "prior_confidence":  args.prior_confidence,
                "max_transcripts":   args.max_transcripts,
            },
            "expansion": {"expand_px": args.expand_px, "min_area_px": args.min_area_px},
        },
        "qc": {"min_cell_area_px": args.min_area_px, "max_cell_area_px": 10000},
    }

    tif_paths = _io.list_fov_tiffs(in_dir)
    fov_filter = [args.fov] if args.fov else None
    tif_paths  = _io.filter_fovs(tif_paths, fov_filter)

    if not tif_paths:
        print(f"ERROR: No TIF files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Baysor RNA-guided segmentation (separate pipeline) ────────────────────
    if args.model == "baysor":
        _run_baysor(args, cfg, out, tif_paths, manifest)
        return

    segmenter = get_segmenter(args.model, cfg)
    print(f"Segmenter: {segmenter.name}")

    print(f"Processing {len(tif_paths)} FOV(s) …")
    qc_results = []

    # Multi-channel mode flag: cpsam_membrane uses DAPI(ch0) + Membrane(ch3)
    _MULTICHANNEL_MODELS = {"cellpose_cpsam_membrane"}
    is_multichannel = args.model in _MULTICHANNEL_MODELS

    for tif_path in tif_paths:
        fov_id  = _io.parse_fov_id(tif_path)
        img_all = _io.load_tiff(tif_path)

        if is_multichannel and img_all.ndim == 3:
            # DAPI(ch0) + Membrane(ch3) → TopHat each → normalise → stack (H,W,2)
            from skimage.morphology import disk, white_tophat
            ch_nuc = img_all[0].astype(np.float32)   # DAPI
            ch_mem = img_all[3].astype(np.float32)   # Membrane

            th_nuc = white_tophat(ch_nuc, disk(20))
            th_mem = white_tophat(ch_mem, disk(20))

            def _norm01(x):
                mn, mx = float(x.min()), float(x.max())
                return (x - mn) / (mx - mn + 1e-8)

            # Cellpose channels=[1,2]: ch-index-0=cytoplasm/membrane, ch-index-1=nucleus
            raw = np.stack([_norm01(th_mem), _norm01(th_nuc)], axis=-1)
        elif img_all.ndim == 3:
            raw = _io.extract_channel(img_all, args.channel)
        else:
            raw = img_all

        fov_meta = {}
        if manifest is not None:
            fov_meta = _io.get_fov_meta(manifest, fov_id)

        tag = f"{fov_id}"
        if fov_meta.get("condition"):
            tag += f"_{fov_meta['condition']}"
        if fov_meta.get("region"):
            tag += f"_{fov_meta['region']}"

        print(f"  [{fov_id}] segmenting …")
        mask = segmenter.segment(raw)
        n_cells = int(mask.max())
        print(f"  [{fov_id}] → {n_cells} nuclei (DAPI-only)")

        std_mask_path = out / "masks" / f"{tag}_{segmenter.name}_nuclear_mask.tif"
        _io.save_mask_tiff(mask, std_mask_path)

        # Optional Baysor RNA-guided refinement (uses DAPI mask as prior)
        if args.baysor_refine:
            fov_num = int("".join(c for c in fov_id if c.isdigit()) or "0")
            refined = _baysor_refine_fov(
                args, cfg, fov_id, fov_num,
                prior_mask_path=std_mask_path,
                image_shape=raw.shape[:2],
                out=out,
            )
            if refined is not None:
                print(f"  [{fov_id}] → {int(refined.max())} cells (Baysor RNA-guided)")
                baysor_mask_path = out / "masks" / f"{tag}_{segmenter.name}_baysor_nuclear_mask.tif"
                _io.save_mask_tiff(refined, baysor_mask_path)
                mask = refined  # use refined for downstream QC/overlay
        n_cells = int(mask.max())  # reflect final mask (DAPI or Baysor-refined)

        # Membrane-guided expansion if --membrane_channel is given and raw TIF has it
        if args.membrane_channel is not None and img_all.ndim == 3 and img_all.shape[0] > args.membrane_channel:
            mem_raw = img_all[args.membrane_channel]
            print(f"  [{fov_id}] expanding with Membrane ch{args.membrane_channel} guidance …")
            exp_mask = expand_mask_membrane_guided(
                mask, mem_raw,
                max_expand_px=args.expand_px * 2,   # allow up to 2× the requested distance
                tophat_radius=20,
                min_area_px=args.min_area_px,
            )
        else:
            exp_mask = expand_mask(mask, expand_px=args.expand_px, min_area_px=args.min_area_px)

        _io.save_mask_tiff(exp_mask, out / "expanded_masks" / f"{tag}_{segmenter.name}_expanded_mask.tif")

        # For cell_table intensity metrics, use raw DAPI (not the stacked multi-ch array)
        raw_for_table = img_all[args.channel] if img_all.ndim == 3 else raw
        cell_table = mask_to_cell_table(mask, raw_for_table, fov_meta)
        csv_path   = out / "cell_tables" / f"{tag}_{segmenter.name}_cells.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        cell_table.to_csv(str(csv_path), index=False)

        save_overlay_png(raw_for_table, mask, out / "overlays" / f"{tag}_{segmenter.name}_overlay.png")
        save_colored_mask_png(mask,   out / "overlays" / f"{tag}_{segmenter.name}_colored.png")

        save_qc_panel(
            raw_for_table, raw_for_table, mask, exp_mask,
            out / "qc" / "panels" / f"{tag}_{segmenter.name}_qc_panel.png",
            fov_id, segmenter.name,
        )
        plot_size_distribution(
            cell_table,
            out / "qc" / "panels" / f"{tag}_{segmenter.name}_size_dist.png",
            fov_id, segmenter.name,
        )

        qc = compute_fov_qc(cell_table, cfg, fov_id=fov_id, model_name=segmenter.name)
        qc_results.append(qc)
        med = qc.get("median_diam_px") if qc else None
        med_str = f"{med:.1f}" if med is not None else "N/A"
        print(f"  [{fov_id}] done  n_cells={n_cells}  median_diam={med_str}px")

        # Release GPU memory between FOVs to prevent OOM on multi-FOV runs
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Aggregate summary
    summary = aggregate_qc(qc_results)
    qc_dir  = out / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(str(qc_dir / "qc_summary.csv"), index=False)
    plot_cellcount_comparison(summary, qc_dir / "cellcount_comparison.png")

    print(f"\nSummary:\n{summary.to_string(index=False)}")
    print(f"\nQC summary → {qc_dir / 'qc_summary.csv'}")


if __name__ == "__main__":
    main()
