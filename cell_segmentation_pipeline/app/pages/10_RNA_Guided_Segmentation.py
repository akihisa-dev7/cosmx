"""RNA-Guided Segmentation (Baysor) — compare DAPI-only vs RNA-informed masks."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# pages/ -> app/ -> cell_segmentation_pipeline/ -> CosMx_2026/
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.express as px
import streamlit as st
import tifffile

from utils.image_loader import (
    PROJECT_ROOT,
    load_raw_dapi,
    load_mask,
    norm_uint8,
    downsample_img,
    downsample_mask,
)
from utils.overlay_utils import colorize_mask, compose_overlay

st.set_page_config(page_title="RNA-Guided Segmentation", layout="wide")

PILOT_FOVS    = ["FOV00001", "FOV00007", "FOV00037", "FOV00043"]
PRIOR_MODELS  = ["cellpose", "stardist", "instanseg"]
SCRIPT_PATH   = Path(__file__).resolve().parents[3] / "scripts" / "run_baysor.py"

STATUS_ICON = {
    "waiting": "⏳",
    "running": "🔄",
    "done":    "✅",
    "error":   "❌",
}

# ── Baysor mask loader ─────────────────────────────────────────────────────────

def load_baysor_mask(fov_id: str, prior_model: str, output_dir: str) -> np.ndarray | None:
    path = Path(output_dir) / prior_model / fov_id / "baysor_mask.tif"
    if path.exists():
        return tifffile.imread(str(path)).astype(np.int32)
    return None


def _find_dapi_mask(fov_id: str, prior_dir: str, prior_model: str) -> np.ndarray | None:
    keyword_map = {
        "cellpose":  ["cpsam_nuclear", "cpsam", "cellpose"],
        "stardist":  ["stardist_nuclear", "stardist"],
        "instanseg": ["instanseg_nuclear", "instanseg"],
    }
    keywords = keyword_map.get(prior_model, [prior_model])
    candidates = sorted(Path(prior_dir).glob(f"{fov_id}*nuclear_mask.tif"))
    for kw in keywords:
        for c in candidates:
            if kw in c.name:
                return tifffile.imread(str(c)).astype(np.int32)
    return None


def _find_tx_file(fov_id: str, tx_dir: str) -> Path | None:
    d = Path(tx_dir)
    for pat in [f"{fov_id}_tx.csv.gz", f"{fov_id}_tx.csv",
                f"*{fov_id}*tx*.csv.gz", f"*{fov_id}*tx*.csv"]:
        hits = list(d.glob(pat))
        if hits:
            return hits[0]
    return None


def _build_run_cmd(
    mode: str,
    fov_id: str,
    prior_model: str,
    tx_dir: str,
    prior_dir: str,
    output_dir: str,
    scale_um: float,
    min_molecules: int,
    prior_confidence: float,
) -> list[str]:
    out_root = str(Path(output_dir) / prior_model)
    return [
        sys.executable, str(SCRIPT_PATH),
        "--mode",             mode,
        "--tx_dir",           tx_dir,
        "--prior_dir",        prior_dir,
        "--output",           out_root,
        "--prior_model",      prior_model,
        "--fovs",             fov_id,
        "--scale",            str(scale_um),
        "--min_molecules",    str(min_molecules),
        "--prior_confidence", str(prior_confidence),
    ]


# ── Session state ──────────────────────────────────────────────────────────────

if "rna_jobs" not in st.session_state:
    st.session_state["rna_jobs"] = {}

if "prepare_outputs" not in st.session_state:
    st.session_state["prepare_outputs"] = {}   # key -> stdout text


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Baysor Configuration")

    tx_dir = st.text_input(
        "Transcript dir",
        value=str(PROJECT_ROOT / "outputs_ssd" / "proseg_pilot"),
    )
    prior_dir = st.text_input(
        "Prior mask dir",
        value=str(PROJECT_ROOT / "outputs" / "pilot_4fov" / "masks"),
    )
    output_dir = st.text_input(
        "Output dir",
        value=str(PROJECT_ROOT / "outputs" / "baysor"),
    )

    st.divider()

    sel_fovs = st.multiselect(
        "FOVs to process",
        PILOT_FOVS,
        default=PILOT_FOVS,
    )

    sel_models = st.multiselect(
        "Prior models",
        PRIOR_MODELS,
        default=PRIOR_MODELS,
        format_func=lambda m: {"cellpose": "Cellpose", "stardist": "StarDist", "instanseg": "InstanSeg"}[m],
    )

    with st.expander("Advanced parameters"):
        scale_um         = st.number_input("scale_um",          value=10.0, step=0.5)
        min_molecules    = st.number_input("min_molecules",      value=15,   step=1)
        prior_confidence = st.slider("prior_confidence",         0.0, 1.0, 0.5, 0.05)
        st.info("prior_dilation_px: auto (Cellpose/InstanSeg=15 px, StarDist=25 px)")

    st.divider()

    run_prepare = st.button("Run Baysor (prepare stage)", type="primary", use_container_width=True)
    run_post    = st.button("Postprocess",                use_container_width=True)

    # ── Progress tracker ───────────────────────────────────────────────────────
    jobs = st.session_state["rna_jobs"]
    if jobs:
        st.subheader("Job status")
        for key, status in jobs.items():
            st.write(f"{STATUS_ICON.get(status, '?')} `{key}` — {status}")


# ── Run prepare ───────────────────────────────────────────────────────────────

if run_prepare:
    if not sel_fovs or not sel_models:
        st.sidebar.warning("Select at least one FOV and one model.")
    else:
        docker_cmds: list[str] = []
        for model in sel_models:
            for fov in sel_fovs:
                key = f"{model}_{fov}"
                tx_path = _find_tx_file(fov, tx_dir)
                if tx_path is None:
                    st.sidebar.warning(f"No transcript file for {fov} — skipping.")
                    st.session_state["rna_jobs"][key] = "error"
                    continue

                prior_mask = _find_dapi_mask(fov, prior_dir, model)
                if prior_mask is None:
                    st.sidebar.warning(f"No {model} mask for {fov} — skipping.")
                    st.session_state["rna_jobs"][key] = "error"
                    continue

                cmd = _build_run_cmd(
                    "prepare", fov, model, tx_dir, prior_dir, output_dir,
                    scale_um, min_molecules, prior_confidence,
                )
                st.session_state["rna_jobs"][key] = "running"
                try:
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=300,
                    )
                    out_text = proc.stdout + proc.stderr
                    st.session_state["prepare_outputs"][key] = out_text
                    if proc.returncode == 0:
                        st.session_state["rna_jobs"][key] = "done"
                    else:
                        st.session_state["rna_jobs"][key] = "error"
                except Exception as exc:
                    st.session_state["rna_jobs"][key] = "error"
                    st.session_state["prepare_outputs"][key] = str(exc)

        st.warning(
            "Prepare stage complete. A Baysor TOML config and transcript CSV have been written "
            "for each FOV. To continue:\n\n"
            "1. Run the Docker command shown in the output below for each FOV.\n"
            "2. Once Baysor finishes, click **Postprocess** to build label masks and AnnData."
        )


# ── Show prepare output / docker commands ─────────────────────────────────────

prep_outputs = st.session_state.get("prepare_outputs", {})
if prep_outputs:
    with st.expander("Prepare stage output (Docker commands)", expanded=True):
        for key, text in prep_outputs.items():
            st.markdown(f"**{key}**")
            st.code(text, language="bash")


# ── Run postprocess ───────────────────────────────────────────────────────────

if run_post:
    if not sel_fovs or not sel_models:
        st.sidebar.warning("Select at least one FOV and one model.")
    else:
        for model in sel_models:
            for fov in sel_fovs:
                key = f"{model}_{fov}"
                cmd = _build_run_cmd(
                    "postprocess", fov, model, tx_dir, prior_dir, output_dir,
                    scale_um, min_molecules, prior_confidence,
                )
                st.session_state["rna_jobs"][key] = "running"
                with st.spinner(f"Postprocessing {key} …"):
                    try:
                        proc = subprocess.run(
                            cmd, capture_output=True, text=True, timeout=600,
                        )
                        baysor_mask = load_baysor_mask(fov, model, output_dir)
                        st.session_state["rna_jobs"][key] = (
                            "done" if baysor_mask is not None else "error"
                        )
                        if proc.returncode != 0:
                            st.error(f"{key}: {proc.stderr[-500:]}")
                    except Exception as exc:
                        st.session_state["rna_jobs"][key] = "error"
                        st.error(f"{key}: {exc}")
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("RNA-Guided Segmentation (Baysor)")
st.caption(
    "Compare DAPI-only segmentation masks against Baysor RNA-guided masks "
    "for each prior model."
)

view_fov = st.selectbox("Select FOV to view", PILOT_FOVS, index=0)

raw_dapi = load_raw_dapi(view_fov)

if raw_dapi is None:
    st.warning(f"Raw DAPI image not found for {view_fov}.")
    raw_dapi_display = np.zeros((512, 512), dtype=np.uint8)
    raw_dapi_ds = raw_dapi_display
else:
    raw_dapi_ds = downsample_img(norm_uint8(raw_dapi), 512)

# ── Comparison grid ───────────────────────────────────────────────────────────

if not sel_models:
    st.info("Select at least one prior model in the sidebar.")
    st.stop()

model_labels = {"cellpose": "Cellpose", "stardist": "StarDist", "instanseg": "InstanSeg"}

HEADER_COLS = ["Model", "Raw DAPI", "DAPI-only mask", "Baysor mask", "DAPI + Baysor overlay"]
hdr = st.columns([1, 2, 2, 2, 2])
for col, label in zip(hdr, HEADER_COLS):
    col.markdown(f"**{label}**")

st.divider()

cell_count_rows: list[dict] = []

for model in sel_models:
    dapi_mask  = _find_dapi_mask(view_fov, prior_dir, model)
    baysor_mask = load_baysor_mask(view_fov, model, output_dir)

    n_dapi   = int(dapi_mask.max())   if dapi_mask is not None   else 0
    n_baysor = int(baysor_mask.max()) if baysor_mask is not None else 0

    cell_count_rows.append({
        "model":   model_labels[model],
        "DAPI-only":  n_dapi,
        "Baysor": n_baysor,
    })

    col_label, col_raw, col_dapi, col_bay, col_overlay = st.columns([1, 2, 2, 2, 2])

    with col_label:
        st.markdown(f"**{model_labels[model]}**")
        st.caption(f"DAPI-only: {n_dapi} cells")
        st.caption(f"Baysor: {n_baysor} cells")

    with col_raw:
        st.image(raw_dapi_ds, use_container_width=True, clamp=True)

    with col_dapi:
        if dapi_mask is not None:
            dm_ds  = downsample_mask(dapi_mask, 512)
            st.image(colorize_mask(dm_ds), use_container_width=True)
        else:
            st.info("DAPI mask not found")

    with col_bay:
        if baysor_mask is not None:
            bm_ds = downsample_mask(baysor_mask, 512)
            st.image(colorize_mask(bm_ds), use_container_width=True)
        else:
            st.info("Run Baysor to generate")

    with col_overlay:
        if baysor_mask is not None and raw_dapi is not None:
            gray_ds = downsample_img(norm_uint8(raw_dapi), 512)
            bm_ds   = downsample_mask(baysor_mask, 512)
            st.image(compose_overlay(gray_ds, bm_ds), use_container_width=True)
        else:
            st.info("Run Baysor to generate")

    st.divider()

# ── Cell count bar chart ──────────────────────────────────────────────────────

if cell_count_rows:
    st.subheader(f"Cell count comparison — {view_fov}")
    import pandas as pd
    df_counts = pd.DataFrame(cell_count_rows).melt(
        id_vars="model", var_name="Source", value_name="Cell count"
    )
    fig = px.bar(
        df_counts,
        x="model",
        y="Cell count",
        color="Source",
        barmode="group",
        labels={"model": "Prior model"},
        color_discrete_map={"DAPI-only": "#4C78A8", "Baysor": "#F58518"},
    )
    fig.update_layout(margin=dict(t=20, b=20), height=350)
    st.plotly_chart(fig, use_container_width=True)
