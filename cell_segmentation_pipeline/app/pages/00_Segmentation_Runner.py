"""Segmentation Runner — input rawdata, run models, compare results."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# pages/ -> app/ -> cell_segmentation_pipeline/ -> CosMx_2026/
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st

from utils.image_loader import (
    PROJECT_ROOT,
    load_raw_dapi,
    norm_uint8,
    downsample_img,
    downsample_mask,
)
from utils.overlay_utils import colorize_mask, compose_overlay

st.set_page_config(page_title="Segmentation Runner", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────

SEG_SCRIPT   = Path(__file__).resolve().parents[3] / "scripts" / "run_segmentation.py"
DEFAULT_IMG  = str(PROJECT_ROOT / "raw_data" / "pilot_4fov" / "slide1_RNA" / "morphology_images")
DEFAULT_OUT  = str(PROJECT_ROOT / "outputs" / "cell_segmentation_pipeline")
DEFAULT_MANIFEST = str(PROJECT_ROOT / "config" / "sample_manifest.csv")

MODELS = {
    "cellpose_cpsam":   "Cellpose cpsam",
    "cellpose_nuclei":  "Cellpose nuclei",
    "stardist":         "StarDist",
    "instanseg":        "InstanSeg",
}

STATUS_ICON = {"waiting": "⏳", "running": "🔄", "done": "✅", "error": "❌"}

# ── Session state ─────────────────────────────────────────────────────────────

if "seg_jobs"  not in st.session_state:
    st.session_state["seg_jobs"]  = {}   # key="{model}_{fov}" → status str
if "seg_procs" not in st.session_state:
    st.session_state["seg_procs"] = {}   # key → subprocess.Popen

# ── Helpers ───────────────────────────────────────────────────────────────────

def _discover_fovs(image_dir: str) -> list[str]:
    d = Path(image_dir)
    if not d.exists():
        return []
    fovs = sorted({
        p.stem.split("_")[-1]          # e.g. "CellComposite_F00001" → "F00001"
        for p in d.glob("*.TIF")
    })
    # Normalise F00001 → FOV00001
    result = []
    for f in fovs:
        num = f.lstrip("FOVf0") or "0"
        result.append(f"FOV{int(num):05d}")
    return result or ["FOV00001", "FOV00007", "FOV00037", "FOV00043"]


def _mask_path(model: str, fov_id: str, output_dir: str) -> Path | None:
    root = Path(output_dir) / "masks"
    hits = [
        p for p in root.glob(f"{fov_id}*nuclear_mask.tif")
        if model.replace("cellpose_", "") in p.name or model in p.name
    ]
    return hits[0] if hits else None


def _load_mask_array(model: str, fov_id: str, output_dir: str) -> np.ndarray | None:
    import tifffile
    p = _mask_path(model, fov_id, output_dir)
    if p and p.exists():
        return tifffile.imread(str(p)).astype(np.int32)
    return None


def _poll_jobs(output_dir: str) -> None:
    """Check running subprocesses and update status based on mask file existence."""
    for key, proc in list(st.session_state["seg_procs"].items()):
        if proc.poll() is not None:          # process finished
            model, fov = key.rsplit("_", 1)
            mask = _mask_path(model, fov, output_dir)
            st.session_state["seg_jobs"][key] = "done" if (mask and mask.exists()) else "error"
            del st.session_state["seg_procs"][key]


def _build_cmd(
    model: str,
    fov_id: str,
    image_dir: str,
    output_dir: str,
    manifest: str,
    params: dict,
) -> list[str]:
    cmd = [
        sys.executable, str(SEG_SCRIPT),
        "--input",   image_dir,
        "--output",  output_dir,
        "--model",   model,
        "--fov",     fov_id,
        "--diameter",           str(params.get("diameter", 25.0)),
        "--cellprob_threshold", str(params.get("cellprob_threshold", -1.0)),
        "--flow_threshold",     str(params.get("flow_threshold", 0.6)),
        "--expand_px",          str(params.get("expand_px", 5)),
        "--min_area_px",        str(params.get("min_area_px", 50)),
    ]
    if Path(manifest).exists():
        cmd += ["--manifest", manifest]
    return cmd


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Input")

    image_dir = st.text_input("Raw image directory", value=DEFAULT_IMG)
    output_dir = st.text_input("Output directory",   value=DEFAULT_OUT)
    manifest   = st.text_input("Manifest CSV (optional)", value=DEFAULT_MANIFEST)

    st.divider()
    st.subheader("FOV selection")

    discovered = _discover_fovs(image_dir)
    sel_fovs = st.multiselect("FOVs to process", discovered, default=discovered[:1])

    st.divider()
    st.subheader("Models & parameters")

    model_params: dict[str, dict] = {}
    sel_models: list[str] = []

    for model_key, model_label in MODELS.items():
        enabled = st.checkbox(model_label, value=(model_key == "cellpose_cpsam"))
        if enabled:
            sel_models.append(model_key)
            with st.expander(f"{model_label} parameters"):
                p: dict = {}
                if model_key.startswith("cellpose"):
                    p["diameter"]           = st.number_input(f"{model_key} diameter",           value=25.0, step=1.0, key=f"{model_key}_d")
                    p["cellprob_threshold"] = st.number_input(f"{model_key} cellprob_threshold", value=-1.0, step=0.1, key=f"{model_key}_cp")
                    p["flow_threshold"]     = st.number_input(f"{model_key} flow_threshold",     value=0.6,  step=0.05, key=f"{model_key}_ft")
                p["expand_px"]   = st.number_input(f"{model_key} expand_px",   value=5,  step=1, key=f"{model_key}_ep")
                p["min_area_px"] = st.number_input(f"{model_key} min_area_px", value=50, step=5, key=f"{model_key}_ma")
                model_params[model_key] = p

    st.divider()

    run_btn = st.button("▶ Run segmentation", type="primary", use_container_width=True)

    # Progress
    jobs = st.session_state["seg_jobs"]
    if jobs:
        st.subheader("Progress")
        for key, status in jobs.items():
            model_k, fov_k = key.rsplit("_", 1)
            label = MODELS.get(model_k, model_k)
            st.write(f"{STATUS_ICON.get(status, '?')} {label} / {fov_k}")


# ── Launch jobs ───────────────────────────────────────────────────────────────

if run_btn:
    if not sel_fovs:
        st.sidebar.warning("Select at least one FOV.")
    elif not sel_models:
        st.sidebar.warning("Select at least one model.")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for model in sel_models:
            for fov in sel_fovs:
                key = f"{model}_{fov}"
                cmd = _build_cmd(
                    model, fov, image_dir, output_dir, manifest,
                    model_params.get(model, {}),
                )
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                st.session_state["seg_jobs"][key]  = "running"
                st.session_state["seg_procs"][key] = proc
        st.rerun()

# Poll running jobs
_poll_jobs(output_dir)

# Auto-refresh while jobs are running
if st.session_state["seg_procs"]:
    time.sleep(2)
    st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Segmentation Runner")
st.caption("Input rawdata → select models & parameters → Run → compare results")

# FOV viewer selector
view_fov = st.selectbox(
    "FOV to view",
    discovered if discovered else ["FOV00001"],
    index=0,
)

raw_dapi = load_raw_dapi(view_fov)
raw_ds   = downsample_img(norm_uint8(raw_dapi), 512) if raw_dapi is not None else None

# ── Comparison grid ───────────────────────────────────────────────────────────

all_models_to_show = [m for m in MODELS if _mask_path(m, view_fov, output_dir) is not None] or sel_models

if not all_models_to_show:
    st.info("No segmentation results yet. Select models and click **Run segmentation**.")
    st.stop()

# Header row
hdr = st.columns([1.2, 2, 2, 2])
for col, label in zip(hdr, ["Model", "Raw DAPI", "Seg mask", "Overlay"]):
    col.markdown(f"**{label}**")
st.divider()

cell_counts: list[dict] = []

for model_key in MODELS:          # fixed display order
    mask = _load_mask_array(model_key, view_fov, output_dir)
    model_label = MODELS[model_key]

    # Show running indicator even before mask exists
    job_key = f"{model_key}_{view_fov}"
    job_status = st.session_state["seg_jobs"].get(job_key, "")

    if mask is None and not job_status:
        continue                  # not selected and no result — hide row

    n_cells = int(mask.max()) if mask is not None else 0
    cell_counts.append({"Model": model_label, "Cells": n_cells})

    col_lbl, col_raw, col_mask, col_ov = st.columns([1.2, 2, 2, 2])

    with col_lbl:
        st.markdown(f"**{model_label}**")
        if job_status == "running":
            st.caption(f"{STATUS_ICON['running']} running…")
        elif mask is not None:
            st.caption(f"{n_cells} cells")
        else:
            st.caption("—")

    with col_raw:
        if raw_ds is not None:
            st.image(raw_ds, use_container_width=True, clamp=True)
        else:
            st.info("DAPI not found")

    with col_mask:
        if mask is not None:
            st.image(colorize_mask(downsample_mask(mask, 512)), use_container_width=True)
        elif job_status == "running":
            st.info("Running…")
        elif job_status == "error":
            st.error("Error — check terminal logs")
        else:
            st.info("Run to generate")

    with col_ov:
        if mask is not None and raw_dapi is not None:
            gray_ds = downsample_img(norm_uint8(raw_dapi), 512)
            st.image(
                compose_overlay(gray_ds, downsample_mask(mask, 512)),
                use_container_width=True,
            )
        else:
            st.info("Run to generate")

    st.divider()


# ── Cell count bar chart ──────────────────────────────────────────────────────

if cell_counts:
    import plotly.express as px
    df = pd.DataFrame([r for r in cell_counts if r["Cells"] > 0])
    if not df.empty:
        st.subheader(f"Cell counts — {view_fov}")
        fig = px.bar(
            df, x="Model", y="Cells",
            color="Model",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
