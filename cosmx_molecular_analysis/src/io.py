"""I/O utilities for reading CosMx flatfiles and segmentation outputs."""
import glob
import gzip
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import yaml


# ---------------------------------------------------------------------------
# FOV name ↔ integer helpers
# ---------------------------------------------------------------------------

def fov_name_to_int(fov_name: str) -> int:
    """Convert 'FOV00001' → 1."""
    return int(re.sub(r"[^0-9]", "", fov_name))


def fov_int_to_name(fov_int: int) -> str:
    """Convert 1 → 'FOV00001'."""
    return f"FOV{fov_int:05d}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_marker_genes(marker_path: str) -> Dict[str, List[str]]:
    with open(marker_path) as f:
        data = yaml.safe_load(f)
    return data.get("cell_types", data)


# ---------------------------------------------------------------------------
# Segmentation outputs
# ---------------------------------------------------------------------------

def find_mask_file(mask_dir: str, fov_name: str) -> Optional[str]:
    """Find expanded mask tif for a given FOV (e.g. FOV00001)."""
    pattern = os.path.join(mask_dir, f"{fov_name}_*expanded*.tif")
    matches = glob.glob(pattern)
    if not matches:
        pattern2 = os.path.join(mask_dir, f"*{fov_name}*expanded*.tif")
        matches = glob.glob(pattern2)
    return matches[0] if matches else None


def load_mask(mask_path: str) -> np.ndarray:
    """Load segmentation mask as uint32 array."""
    return tifffile.imread(mask_path).astype(np.uint32)


def find_cell_table_file(cell_table_dir: str, fov_name: str) -> Optional[str]:
    """Find cell table CSV for a given FOV."""
    pattern = os.path.join(cell_table_dir, f"{fov_name}_*cells.csv")
    matches = glob.glob(pattern)
    if not matches:
        pattern2 = os.path.join(cell_table_dir, f"*{fov_name}*cells.csv")
        matches = glob.glob(pattern2)
    return matches[0] if matches else None


def load_cell_table(cell_table_path: str) -> pd.DataFrame:
    """Load cell morphology table from segmentation pipeline."""
    df = pd.read_csv(cell_table_path)
    df.columns = df.columns.str.strip()
    return df


# ---------------------------------------------------------------------------
# RNA flatfiles
# ---------------------------------------------------------------------------

def find_flatfile(directory: str, suffix: str) -> Optional[str]:
    """Find a flatfile matching *{suffix} in directory."""
    for ext in [".csv.gz", ".csv"]:
        matches = glob.glob(os.path.join(directory, f"*{suffix}{ext}"))
        if matches:
            return matches[0]
    return None


def load_tx_file(
    tx_path: str,
    fovs: Optional[List[int]] = None,
    exclude_prefixes: Optional[List[str]] = None,
    min_qv: Optional[float] = None,
) -> pd.DataFrame:
    """Load RNA transcript file with optional filtering.

    Parameters
    ----------
    fovs:
        Integer FOV IDs to keep (e.g. [1, 7, 37, 43]).
    exclude_prefixes:
        Gene name prefixes to exclude (e.g. ['NegPrb', 'FalseCode']).
    min_qv:
        Minimum quality value threshold (skipped if column absent).
    """
    # Use awk to pre-filter by FOV (column 1 = fov), then parse with pandas.
    # This avoids loading all 49 M rows into Python memory at once.
    import subprocess
    import io as _io
    import tempfile
    import os as _os

    # Determine the FOV column index from the header
    header_proc = subprocess.run(
        ["zcat", tx_path],
        capture_output=True,
        text=True,
    )
    # Only grab first line for the header
    header_lines = header_proc.stdout.split("\n", 2)
    header = header_lines[0] if header_lines else ""

    if fovs is not None:
        # Build awk filter: print header + rows where fov field matches
        fov_values_str = "|".join(str(v) for v in fovs)
        # fov is typically the first column; find its index dynamically
        cols = header.split(",")
        fov_col_idx = 1  # default: 1-indexed column 1
        for i, c in enumerate(cols):
            if c.strip().lower() == "fov":
                fov_col_idx = i + 1
                break

        # Use awk to print header + matching rows
        awk_script = (
            f'NR==1 || (${{fov_col_idx}} ~ /^({fov_values_str})$/)'
        ).replace("{fov_col_idx}", str(fov_col_idx))

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            zcat = subprocess.Popen(
                ["zcat", tx_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            awk = subprocess.Popen(
                ["awk", "-F,", awk_script],
                stdin=zcat.stdout,
                stdout=open(tmp_path, "wb"),
                stderr=subprocess.DEVNULL,
            )
            zcat.stdout.close()
            awk.wait()
            zcat.wait()
            df = pd.read_csv(tmp_path, on_bad_lines="skip", low_memory=False)
        finally:
            _os.unlink(tmp_path)
    else:
        # No FOV filter – read in chunks
        zcat = subprocess.Popen(
            ["zcat", tx_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        frames = []
        for chunk in pd.read_csv(
            zcat.stdout, chunksize=500_000, on_bad_lines="skip", low_memory=False
        ):
            frames.append(chunk)
        zcat.wait()
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Normalise column names
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("gene", "target", "gene_name"):
            col_map[c] = "target"
        elif cl in ("x_local_px", "x_local", "x"):
            col_map[c] = "x_local_px"
        elif cl in ("y_local_px", "y_local", "y"):
            col_map[c] = "y_local_px"
        elif cl == "qv":
            col_map[c] = "qv"
    df = df.rename(columns=col_map)

    if min_qv is not None and "qv" in df.columns:
        df = df[df["qv"] >= min_qv]

    if exclude_prefixes:
        pattern = "|".join(f"^{p}" for p in exclude_prefixes)
        mask_exclude = df["target"].str.match(pattern, na=False)
        df = df[~mask_exclude]

    return df.reset_index(drop=True)


def load_rna_exprmat(exprmat_path: str, fovs: Optional[List[int]] = None) -> pd.DataFrame:
    """Load RNA expression matrix (fov × cell_ID × genes)."""
    df = pd.read_csv(exprmat_path, compression="infer")
    if fovs is not None:
        df = df[df["fov"].isin(fovs)]
    return df.reset_index(drop=True)


def load_rna_metadata(metadata_path: str, fovs: Optional[List[int]] = None) -> pd.DataFrame:
    """Load RNA cell metadata."""
    df = pd.read_csv(metadata_path, compression="infer")
    if fovs is not None:
        df = df[df["fov"].isin(fovs)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Protein flatfiles
# ---------------------------------------------------------------------------

def load_protein_exprmat(
    exprmat_path: str, fovs: Optional[List[int]] = None
) -> pd.DataFrame:
    """Load protein expression matrix."""
    df = pd.read_csv(exprmat_path, compression="infer")
    if fovs is not None:
        df = df[df["fov"].isin(fovs)]
    return df.reset_index(drop=True)


def load_protein_metadata(metadata_path: str, fovs: Optional[List[int]] = None) -> pd.DataFrame:
    """Load protein cell metadata."""
    df = pd.read_csv(metadata_path, compression="infer")
    if fovs is not None:
        df = df[df["fov"].isin(fovs)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sample manifest
# ---------------------------------------------------------------------------

def load_sample_manifest(manifest_path: str) -> pd.DataFrame:
    return pd.read_csv(manifest_path)
