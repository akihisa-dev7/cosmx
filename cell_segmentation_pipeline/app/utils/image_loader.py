"""Cached image / mask / table loaders for the QC GUI."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
# image_loader.py: CosMx_2026/cell_segmentation_pipeline/app/utils/image_loader.py
# parents[0]=utils  [1]=app  [2]=cell_segmentation_pipeline  [3]=CosMx_2026
PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR    = PROJECT_ROOT / "raw_data" / "pilot_4fov" / "slide1_RNA" / "morphology_images"
MANIFEST_PATH = PROJECT_ROOT / "config" / "sample_manifest.csv"

# Known output locations (searched in order)
DEFAULT_OUTPUT_ROOTS = [
    str(PROJECT_ROOT / "outputs" / "pilot_4fov"),
    str(PROJECT_ROOT / "outputs" / "cell_segmentation_pipeline"),
]

CHANNEL_NAMES = {0: "DAPI", 1: "PanCK", 2: "G (autofluorescence)", 3: "Membrane", 4: "CD45"}

# ── Lazy streamlit import (utils can be imported outside Streamlit context) ──
def _cache(fn):
    """Wrap with st.cache_data if streamlit is available, else identity."""
    try:
        import streamlit as st
        return st.cache_data(show_spinner=False)(fn)
    except Exception:
        return fn


# ── Manifest ──────────────────────────────────────────────────────────────────

def _load_manifest_raw() -> "pd.DataFrame | None":
    if MANIFEST_PATH.exists():
        df = pd.read_csv(str(MANIFEST_PATH))
        if "fov" in df.columns:
            df["fov"] = df["fov"].astype(str).str.upper()
        return df
    return None

load_manifest = _cache(_load_manifest_raw)


def get_fov_meta(fov_id: str) -> dict:
    df = load_manifest()
    if df is None:
        return {}
    rows = df[df["fov"] == fov_id.upper()]
    return rows.iloc[0].to_dict() if not rows.empty else {}


# ── TIFF loading ──────────────────────────────────────────────────────────────

def _load_raw_dapi(fov_id: str, channel: int = 0) -> "np.ndarray | None":
    num = fov_id[-5:]
    candidates = list(IMAGE_DIR.glob(f"*_F{num}.TIF")) + \
                 list(IMAGE_DIR.glob(f"*_F{num}.tif"))
    if not candidates:
        return None
    img = tifffile.imread(str(candidates[0]))
    if img.ndim == 2:
        return img
    return img[channel] if channel < img.shape[0] else img[0]

load_raw_dapi = _cache(_load_raw_dapi)


def _load_enhanced(fov_id: str, output_root: str) -> "np.ndarray | None":
    root = Path(output_root)
    candidates = sorted((root / "enhanced").glob(f"{fov_id}*enhanced*.tif"))
    if not candidates:
        return None
    img = tifffile.imread(str(candidates[0]))
    return img[0] if img.ndim == 3 else img

load_enhanced = _cache(_load_enhanced)


def _load_mask(fov_id: str, model_key: str, output_root: str) -> "np.ndarray | None":
    root = Path(output_root)
    # Use exact model-key match to avoid "cpsam" glob hitting "cellpose_cpsam"
    candidates = [
        p for p in sorted((root / "masks").glob(f"{fov_id}*nuclear_mask.tif"))
        if _parse_model(p.name) == model_key
    ]
    if not candidates:
        return None
    return tifffile.imread(str(candidates[0])).astype(np.int32)

load_mask = _cache(_load_mask)


def _load_expanded_mask(fov_id: str, model_key: str, output_root: str) -> "np.ndarray | None":
    root = Path(output_root)
    candidates = [
        p for p in sorted((root / "expanded_masks").glob(f"{fov_id}*expanded_mask.tif"))
        if _parse_model(p.name.replace("_expanded_mask", "_nuclear_mask")) == model_key
    ]
    if not candidates:
        return None
    return tifffile.imread(str(candidates[0])).astype(np.int32)

load_expanded_mask = _cache(_load_expanded_mask)


def _load_cell_table(fov_id: str, model_key: str, output_root: str) -> "pd.DataFrame | None":
    root = Path(output_root)
    candidates = [
        p for p in sorted((root / "cell_tables").glob(f"{fov_id}*cells.csv"))
        if _parse_model(p.name.replace("_cells", "_nuclear_mask")) == model_key
    ]
    if not candidates:
        return None
    return pd.read_csv(str(candidates[0]))

load_cell_table = _cache(_load_cell_table)


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_fovs(output_root: str) -> list[str]:
    root = Path(output_root)
    mask_dir = root / "masks"
    if not mask_dir.exists():
        return []
    fovs = sorted({
        p.name.split("_")[0]
        for p in mask_dir.glob("*nuclear_mask.tif")
        if p.name.startswith("FOV")
    })
    return fovs


def discover_models(fov_id: str, output_root: str) -> list[str]:
    root = Path(output_root)
    mask_dir = root / "masks"
    if not mask_dir.exists():
        return []
    models = []
    for p in sorted(mask_dir.glob(f"{fov_id}*nuclear_mask.tif")):
        model = _parse_model(p.name)
        if model and model not in models:
            models.append(model)
    return models


def _parse_model(filename: str) -> str:
    """Extract model key from mask filename."""
    stem = Path(filename).stem.replace("_nuclear_mask", "")
    stem = re.sub(r"^FOV\d+_?", "", stem)
    stem = re.sub(r"^(AD|Control)_?", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"^[FH]_", "", stem)
    return stem.strip("_")


# ── Display helpers ───────────────────────────────────────────────────────────

def norm_uint8(img: np.ndarray) -> np.ndarray:
    mn, mx = float(img.min()), float(img.max())
    if mx == mn:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def downsample_img(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale    = max_dim / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    if img.ndim == 2:
        pil = Image.fromarray(img, mode="L")
    else:
        pil = Image.fromarray(img)
    return np.array(pil.resize(new_size, Image.LANCZOS))


def downsample_mask(mask: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    h, w = mask.shape
    if max(h, w) <= max_dim:
        return mask
    scale    = max_dim / max(h, w)
    new_size = (int(w * scale), int(h * scale))
    pil = Image.fromarray(mask.astype(np.int32), mode="I")
    return np.array(pil.resize(new_size, Image.NEAREST))
