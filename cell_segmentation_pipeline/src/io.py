"""TIFF I/O, FOV discovery, and manifest utilities."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml


# ── Config ──────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── TIFF I/O ─────────────────────────────────────────────────────────────────

def load_tiff(path: str | Path) -> np.ndarray:
    """Load multi-channel TIFF. Returns (C, H, W) uint16 array."""
    img = tifffile.imread(str(path))
    if img.ndim == 2:
        img = img[np.newaxis, ...]   # add channel dim
    elif img.ndim == 3:
        # tifffile returns (pages, H, W) for multi-page TIFFs
        pass
    return img


def extract_channel(img: np.ndarray, ch: int) -> np.ndarray:
    """Extract single channel from (C, H, W) array → (H, W)."""
    return img[ch]


def save_tiff(arr: np.ndarray, path: str | Path, dtype=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if dtype is not None:
        arr = arr.astype(dtype)
    tifffile.imwrite(str(path), arr)


def save_mask_tiff(mask: np.ndarray, path: str | Path) -> None:
    """Save integer label mask. Uses uint16 unless cell count exceeds 65535."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_labels = int(mask.max())
    dtype = np.uint32 if n_labels > 65535 else np.uint16
    tifffile.imwrite(str(path), mask.astype(dtype))


# ── FOV discovery ─────────────────────────────────────────────────────────────

_FOV_PATTERN = re.compile(r"_F(\d{5})", re.IGNORECASE)
_FOV_PREFIX  = re.compile(r"^FOV(\d{5})", re.IGNORECASE)


def parse_fov_id(path: str | Path) -> str:
    """Extract FOV ID from filename.

    Handles two formats:
      - '*_F00001.TIF'          (raw morphology images)
      - 'FOV00001_*_enhanced.tif' (preprocessed enhanced TIFFs)
    """
    stem = Path(path).stem
    m = _FOV_PATTERN.search(stem)
    if m:
        return f"FOV{m.group(1)}"
    m = _FOV_PREFIX.match(stem)
    if m:
        return f"FOV{m.group(1)}"
    raise ValueError(f"Cannot parse FOV ID from: {path}")


def list_fov_tiffs(image_dir: str | Path) -> list[Path]:
    """Return sorted list of morphology TIFF paths."""
    image_dir = Path(image_dir)
    tifs = sorted(image_dir.glob("*.TIF")) + sorted(image_dir.glob("*.tif"))
    # deduplicate (case-insensitive filesystems)
    seen: set[str] = set()
    unique = []
    for p in tifs:
        key = p.name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def filter_fovs(tif_paths: list[Path], fov_ids: list[str] | None) -> list[Path]:
    """Filter TIFF list to only the requested FOV IDs (None = all)."""
    if fov_ids is None:
        return tif_paths
    wanted = {f.upper() for f in fov_ids}
    return [p for p in tif_paths if parse_fov_id(p) in wanted]


# ── Manifest ──────────────────────────────────────────────────────────────────

def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalise fov column to uppercase FOV00001 format
    if "fov" in df.columns:
        df["fov"] = df["fov"].str.upper()
    return df


def get_fov_meta(manifest: pd.DataFrame, fov_id: str) -> dict:
    """Return metadata dict for a FOV. Falls back to empty dict if not found."""
    rows = manifest[manifest["fov"] == fov_id.upper()]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()
