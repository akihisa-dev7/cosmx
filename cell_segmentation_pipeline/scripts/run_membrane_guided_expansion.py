#!/usr/bin/env python3
"""Apply membrane-guided expansion to existing cpsam nuclear masks.

Loads existing nuclear masks from outputs/pilot_4fov/masks/*_cpsam_nuclear_mask.tif,
loads ch3 (Membrane) from raw morphology TIFs, and produces:
  - membrane-guided expanded masks   → outputs/pilot_4fov/expanded_masks_membrane/
  - simple expanded masks (baseline) → same dir (for comparison)
  - side-by-side QC PNG panels       → outputs/pilot_4fov/qc/membrane_expansion/

Example::

    python cell_segmentation_pipeline/scripts/run_membrane_guided_expansion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.morphology import disk, white_tophat
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cell_segmentation_pipeline.src.masks import (
    expand_mask,
    expand_mask_membrane_guided,
    save_colored_mask_png,
)

# ── Config ────────────────────────────────────────────────────────────────────

RAW_DIR     = Path("raw_data/pilot_4fov/slide1_RNA/morphology_images")
MASK_DIR    = Path("outputs/pilot_4fov/masks")
OUT_DIR     = Path("outputs/pilot_4fov/expanded_masks_membrane")
QC_DIR      = Path("outputs/pilot_4fov/qc/membrane_expansion")

MEM_CH      = 3       # Membrane channel index in raw TIF (0-indexed)
DAPI_CH     = 0       # DAPI channel index
EXPAND_PX   = 5       # simple expansion distance
MAX_MEM_EXP = 10      # max expansion for membrane-guided (2× simple)
TOPHAT_R    = 20      # TopHat radius for membrane preprocessing
MIN_AREA    = 50      # min cell area px²

# FOV → raw TIF mapping (filename pattern: *_F{fov}.TIF)
FOV_MANIFEST = {
    "FOV00001": {"tag": "FOV00001_AD_F",      "condition": "AD",      "region": "F"},
    "FOV00007": {"tag": "FOV00007_AD_H",      "condition": "AD",      "region": "H"},
    "FOV00037": {"tag": "FOV00037_Control_F", "condition": "Control", "region": "F"},
    "FOV00043": {"tag": "FOV00043_Control_H", "condition": "Control", "region": "H"},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_display(ch: np.ndarray, pct: float = 99.5) -> np.ndarray:
    p = float(np.percentile(ch, pct))
    return np.clip(ch.astype(np.float32) / max(p, 1.0), 0.0, 1.0)


def _bdry_rgb(base_ch: np.ndarray, mask: np.ndarray,
              color=(1.0, 0.35, 0.0)) -> np.ndarray:
    rgb = np.stack([_norm_display(base_ch)] * 3, axis=-1)
    rgb[find_boundaries(mask, mode="outer")] = color
    return rgb


def _tophat_norm(ch: np.ndarray) -> np.ndarray:
    th = white_tophat(ch.astype(np.float32), disk(TOPHAT_R))
    mx = float(th.max())
    return th / mx if mx > 0 else th


def save_qc_panel(
    dapi: np.ndarray,
    mem: np.ndarray,
    nuc_mask: np.ndarray,
    simple_exp: np.ndarray,
    mem_exp: np.ndarray,
    out_path: Path,
    fov_id: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(28, 14), dpi=120)

    th_m = _tophat_norm(mem)

    def _show(ax, img, title, cmap="gray"):
        ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    n_nuc    = int(nuc_mask.max())
    n_simple = int(simple_exp.max())
    n_mem    = int(mem_exp.max())

    _show(axes[0, 0], _norm_display(dapi),         "DAPI")
    _show(axes[0, 1], th_m,                         "Membrane (TopHat)", cmap="magma")
    _show(axes[0, 2], _bdry_rgb(dapi, nuc_mask),    f"Nuclear mask  n={n_nuc}")
    _show(axes[0, 3], _bdry_rgb(dapi, simple_exp),  f"Simple expand +{EXPAND_PX}px  n={n_simple}")
    _show(axes[1, 0], _bdry_rgb(mem, nuc_mask,  color=(0.0, 1.0, 1.0)),
          "Membrane + nuclear boundaries")
    _show(axes[1, 1], _bdry_rgb(mem, mem_exp, color=(1.0, 0.5, 0.0)),
          f"Membrane + guided boundaries  n={n_mem}")

    # Overlay comparison: simple=blue, guided=orange
    comp = np.stack([_norm_display(dapi)] * 3, axis=-1)
    comp[find_boundaries(simple_exp, mode="outer")] = [0.2, 0.4, 1.0]
    comp[find_boundaries(mem_exp,    mode="outer")] = [1.0, 0.5, 0.1]
    _show(axes[1, 2], comp, "Blue=simple  Orange=membrane-guided")

    # Difference: membrane-guided covers more / less area
    simple_fg = (simple_exp > 0).astype(np.int8)
    mem_fg    = (mem_exp    > 0).astype(np.int8)
    diff = simple_fg.astype(np.int16) - mem_fg.astype(np.int16)
    _show(axes[1, 3], diff, "Simple-only(+1) vs Guided-only(-1)\n(white=both)", cmap="RdBu")

    simple_cov = simple_fg.mean() * 100
    mem_cov    = mem_fg.mean()    * 100

    fig.suptitle(
        f"{fov_id}  —  Nuclear cpsam + Membrane-guided expansion\n"
        f"Nuclear: {n_nuc}  |  Simple expand: {n_simple} ({simple_cov:.1f}% coverage)"
        f"  |  Membrane-guided: {n_mem} ({mem_cov:.1f}% coverage)",
        fontsize=11,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    print(f"    QC panel → {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    # Find raw TIFs
    raw_tifs = {p.stem.split("_")[-1].upper(): p
                for p in RAW_DIR.glob("*.TIF")}

    results = []

    for fov_id, meta in FOV_MANIFEST.items():
        tag = meta["tag"]
        print(f"\n[{fov_id}]")

        # ── Load nuclear mask ──────────────────────────────────────────────────
        nuc_path = MASK_DIR / f"{tag}_cpsam_nuclear_mask.tif"
        if not nuc_path.exists():
            print(f"  SKIP – nuclear mask not found: {nuc_path}")
            continue
        nuc_mask = tifffile.imread(str(nuc_path)).astype(np.int32)
        print(f"  Nuclear mask: {nuc_mask.max()} cells  shape={nuc_mask.shape}")

        # ── Load raw TIF for DAPI + Membrane ──────────────────────────────────
        fov_num = fov_id.replace("FOV", "")  # e.g. '00001'
        raw_key = f"F{fov_num}"
        # filename ends with _F{fov_num}
        raw_path = next(
            (p for p in RAW_DIR.glob("*.TIF") if f"_F{fov_num}" in p.stem),
            None,
        )
        if raw_path is None:
            print(f"  SKIP – raw TIF not found for {fov_id}")
            continue
        img = tifffile.imread(str(raw_path))   # (5, H, W) uint16
        dapi_ch = img[DAPI_CH]
        mem_ch  = img[MEM_CH]
        print(f"  Raw TIF: {img.shape}  mem_ch max={mem_ch.max()}")

        # ── Simple expansion (baseline) ───────────────────────────────────────
        simple_exp = expand_mask(nuc_mask, expand_px=EXPAND_PX, min_area_px=MIN_AREA)

        # ── Membrane-guided expansion ─────────────────────────────────────────
        print(f"  Running membrane-guided expansion (max={MAX_MEM_EXP}px) …")
        mem_exp = expand_mask_membrane_guided(
            nuc_mask,
            mem_ch,
            max_expand_px=MAX_MEM_EXP,
            tophat_radius=TOPHAT_R,
            min_area_px=MIN_AREA,
        )

        # ── Coverage stats ────────────────────────────────────────────────────
        n_nuc    = int(nuc_mask.max())
        n_simple = int(simple_exp.max())
        n_mem    = int(mem_exp.max())
        cov_s    = (simple_exp > 0).mean() * 100
        cov_m    = (mem_exp    > 0).mean() * 100

        print(f"  Simple  expand: {n_simple} cells, {cov_s:.1f}% coverage")
        print(f"  Membrane-guided: {n_mem} cells, {cov_m:.1f}% coverage")

        # ── Save masks ────────────────────────────────────────────────────────
        tifffile.imwrite(
            str(OUT_DIR / f"{tag}_cpsam_simple_expanded_mask.tif"),
            simple_exp.astype(np.uint16),
        )
        tifffile.imwrite(
            str(OUT_DIR / f"{tag}_cpsam_membrane_guided_expanded_mask.tif"),
            mem_exp.astype(np.uint16),
        )
        save_colored_mask_png(
            simple_exp, OUT_DIR / f"{tag}_cpsam_simple_expanded_colored.png"
        )
        save_colored_mask_png(
            mem_exp, OUT_DIR / f"{tag}_cpsam_membrane_guided_expanded_colored.png"
        )

        # ── QC panel ──────────────────────────────────────────────────────────
        save_qc_panel(
            dapi_ch, mem_ch,
            nuc_mask, simple_exp, mem_exp,
            QC_DIR / f"{tag}_membrane_expansion_qc.png",
            fov_id,
        )

        results.append({
            "FOV": fov_id,
            "condition": meta["condition"],
            "region": meta["region"],
            "n_nuclear": n_nuc,
            "n_simple_exp": n_simple,
            "cov_simple_pct": round(cov_s, 2),
            "n_membrane_guided": n_mem,
            "cov_membrane_guided_pct": round(cov_m, 2),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = OUT_DIR / "membrane_expansion_summary.csv"
        df.to_csv(str(csv_path), index=False)
        print(f"\n{'='*60}")
        print(df.to_string(index=False))
        print(f"\nSummary CSV → {csv_path}")


if __name__ == "__main__":
    main()
