"""Segmentation Runner — input rawdata, run models, compare results."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# parents[0]=pages  [1]=app  [2]=cell_segmentation_pipeline  [3]=CosMx_2026
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # app utils

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

SEG_SCRIPT       = Path(__file__).resolve().parents[2] / "scripts" / "run_segmentation.py"
DEFAULT_IMG      = str(PROJECT_ROOT / "raw_data" / "pilot_4fov" / "slide1_RNA" / "morphology_images")
DEFAULT_OUT      = str(PROJECT_ROOT / "outputs" / "cell_segmentation_pipeline")
DEFAULT_MANIFEST = str(PROJECT_ROOT / "config" / "sample_manifest.csv")

MODELS = {
    "cellpose_cpsam":  "Cellpose cpsam",
    "cellpose_nuclei": "Cellpose nuclei",
    "stardist":        "StarDist",
    "instanseg":       "InstanSeg",
}

STATUS_COLOR = {
    "waiting": "#888888",
    "running": "#f0a500",
    "done":    "#2ecc71",
    "error":   "#e74c3c",
}
STATUS_LABEL = {
    "waiting": "待機中",
    "running": "実行中",
    "done":    "完了",
    "error":   "エラー",
}
STATUS_ICON = {"waiting": "⏳", "running": "🔄", "done": "✅", "error": "❌"}

# ── Session state ─────────────────────────────────────────────────────────────

for _k, _v in [
    ("seg_jobs",   {}),
    ("seg_procs",  {}),
    ("seg_errors", {}),
    ("seg_starts", {}),   # key → start timestamp (float)
    ("seg_logs",   {}),   # key → last N lines of stdout
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Helpers ───────────────────────────────────────────────────────────────────

def _discover_fovs(image_dir: str) -> list[str]:
    d = Path(image_dir)
    if not d.exists():
        return ["FOV00001", "FOV00007", "FOV00037", "FOV00043"]
    fovs = sorted({p.stem.split("_")[-1] for p in d.glob("*.TIF")})
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
    return tifffile.imread(str(p)).astype(np.int32) if (p and p.exists()) else None


def _elapsed(key: str) -> str:
    t0 = st.session_state["seg_starts"].get(key)
    if t0 is None:
        return ""
    secs = int(time.time() - t0)
    return f"{secs // 60}:{secs % 60:02d}"


def _poll_jobs(output_dir: str) -> None:
    for key, proc in list(st.session_state["seg_procs"].items()):
        if proc.poll() is not None:
            model, fov = key.rsplit("_", 1)
            mask = _mask_path(model, fov, output_dir)
            st.session_state["seg_jobs"][key] = "done" if (mask and mask.exists()) else "error"
            err_file = Path(output_dir) / f".err_{key}.txt"
            if err_file.exists():
                st.session_state["seg_errors"][key] = err_file.read_text()[-2000:]
                err_file.unlink(missing_ok=True)
            out_file = Path(output_dir) / f".out_{key}.txt"
            if out_file.exists():
                lines = out_file.read_text().splitlines()
                st.session_state["seg_logs"][key] = "\n".join(lines[-20:])
                out_file.unlink(missing_ok=True)
            del st.session_state["seg_procs"][key]


def _build_cmd(model, fov_id, image_dir, output_dir, manifest, params) -> list[str]:
    cmd = [
        sys.executable, str(SEG_SCRIPT),
        "--input",   image_dir, "--output", output_dir,
        "--model",   model,     "--fov",    fov_id,
        "--diameter",           str(params.get("diameter", 25.0)),
        "--cellprob_threshold", str(params.get("cellprob_threshold", -1.0)),
        "--flow_threshold",     str(params.get("flow_threshold", 0.6)),
        "--expand_px",          str(params.get("expand_px", 5)),
        "--min_area_px",        str(params.get("min_area_px", 50)),
    ]
    if Path(manifest).exists():
        cmd += ["--manifest", manifest]
    return cmd


def _status_badge(status: str) -> str:
    color = STATUS_COLOR.get(status, "#888")
    label = STATUS_LABEL.get(status, status)
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:0.8em;font-weight:bold">{label}</span>'
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Input")
    image_dir  = st.text_input("Raw image directory", value=DEFAULT_IMG)
    output_dir = st.text_input("Output directory",    value=DEFAULT_OUT)
    manifest   = st.text_input("Manifest CSV",        value=DEFAULT_MANIFEST)

    st.divider()
    st.subheader("FOV")
    discovered = _discover_fovs(image_dir)
    sel_fovs   = st.multiselect("FOVs to process", discovered, default=discovered[:1])

    st.divider()
    st.subheader("Models & parameters")
    model_params: dict[str, dict] = {}
    sel_models:   list[str]       = []
    for model_key, model_label in MODELS.items():
        enabled = st.checkbox(model_label, value=(model_key == "cellpose_cpsam"))
        if enabled:
            sel_models.append(model_key)
            with st.expander(f"{model_label} parameters"):
                p: dict = {}
                if model_key.startswith("cellpose"):
                    p["diameter"]           = st.number_input("diameter",           value=25.0, step=1.0,  key=f"{model_key}_d")
                    p["cellprob_threshold"] = st.number_input("cellprob_threshold", value=-1.0, step=0.1,  key=f"{model_key}_cp")
                    p["flow_threshold"]     = st.number_input("flow_threshold",     value=0.6,  step=0.05, key=f"{model_key}_ft")
                p["expand_px"]   = st.number_input("expand_px",   value=5,  step=1, key=f"{model_key}_ep")
                p["min_area_px"] = st.number_input("min_area_px", value=50, step=5, key=f"{model_key}_ma")
                model_params[model_key] = p

    st.divider()
    run_btn   = st.button("▶ Run segmentation", type="primary", use_container_width=True)
    reset_btn = st.button("🗑 Clear status",    use_container_width=True)
    if reset_btn:
        for k in ["seg_jobs", "seg_procs", "seg_errors", "seg_starts", "seg_logs"]:
            st.session_state[k] = {}
        st.rerun()


# ── Launch jobs ───────────────────────────────────────────────────────────────

if run_btn:
    # Guard: don't launch if jobs are already running
    if st.session_state["seg_procs"]:
        st.sidebar.warning("実行中のジョブがあります。完了を待つか「🗑 Clear status」で停止してください。")
    elif not SEG_SCRIPT.exists():
        st.error(f"Script not found: {SEG_SCRIPT}")
    elif not sel_fovs:
        st.sidebar.warning("FOVを1つ以上選択してください。")
    elif not sel_models:
        st.sidebar.warning("モデルを1つ以上選択してください。")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for model in sel_models:
            for fov in sel_fovs:
                key = f"{model}_{fov}"
                st.session_state["seg_errors"].pop(key, None)
                st.session_state["seg_logs"].pop(key, None)
        for model in sel_models:
            for fov in sel_fovs:
                key = f"{model}_{fov}"
                cmd      = _build_cmd(model, fov, image_dir, output_dir, manifest, model_params.get(model, {}))
                err_file = Path(output_dir) / f".err_{key}.txt"
                out_file = Path(output_dir) / f".out_{key}.txt"
                proc = subprocess.Popen(cmd, stdout=open(out_file, "w"), stderr=open(err_file, "w"))
                st.session_state["seg_jobs"][key]   = "running"
                st.session_state["seg_procs"][key]  = proc
                st.session_state["seg_starts"][key] = time.time()
        st.rerun()

# Poll & auto-refresh
_poll_jobs(output_dir)
if st.session_state["seg_procs"]:
    time.sleep(2)
    st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Segmentation Runner")

# ── Progress panel (shown whenever there are jobs) ────────────────────────────

jobs = st.session_state["seg_jobs"]

if jobs:
    total   = len(jobs)
    n_done  = sum(1 for s in jobs.values() if s == "done")
    n_err   = sum(1 for s in jobs.values() if s == "error")
    n_run   = sum(1 for s in jobs.values() if s == "running")
    n_wait  = sum(1 for s in jobs.values() if s == "waiting")
    pct     = n_done / total

    st.subheader("実行状況")

    # Overall progress bar
    prog_col, stat_col = st.columns([3, 1])
    with prog_col:
        st.progress(pct, text=f"{n_done} / {total} 完了")
    with stat_col:
        parts = []
        if n_run:  parts.append(f"🔄 実行中 {n_run}")
        if n_done: parts.append(f"✅ 完了 {n_done}")
        if n_err:  parts.append(f"❌ エラー {n_err}")
        if n_wait: parts.append(f"⏳ 待機 {n_wait}")
        st.markdown("  ".join(parts))

    # Per-job table
    rows_html = ""
    for key, status in sorted(jobs.items()):
        model_k, fov_k = key.rsplit("_", 1)
        label   = MODELS.get(model_k, model_k)
        elapsed = _elapsed(key) if status in ("running", "done", "error") else "—"
        badge   = _status_badge(status)
        log_preview = st.session_state["seg_logs"].get(key, "")
        last_line = log_preview.splitlines()[-1] if log_preview else ""

        rows_html += (
            f"<tr>"
            f"<td style='padding:6px 12px'>{label}</td>"
            f"<td style='padding:6px 12px;color:#888'>{fov_k}</td>"
            f"<td style='padding:6px 12px'>{badge}</td>"
            f"<td style='padding:6px 12px;font-family:monospace;color:#aaa;font-size:0.8em'>{elapsed}</td>"
            f"<td style='padding:6px 12px;font-family:monospace;font-size:0.75em;color:#ccc;"
            f"max-width:300px;overflow:hidden;white-space:nowrap'>{last_line}</td>"
            f"</tr>"
        )

    st.markdown(
        f"""
        <table style='width:100%;border-collapse:collapse;background:#1e1e1e;
                      border-radius:8px;overflow:hidden'>
          <thead>
            <tr style='background:#2a2a2a;color:#aaa;font-size:0.85em'>
              <th style='padding:8px 12px;text-align:left'>モデル</th>
              <th style='padding:8px 12px;text-align:left'>FOV</th>
              <th style='padding:8px 12px;text-align:left'>状態</th>
              <th style='padding:8px 12px;text-align:left'>経過時間</th>
              <th style='padding:8px 12px;text-align:left'>最終ログ</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    # Show error details if any
    for key, err_text in st.session_state["seg_errors"].items():
        if err_text:
            model_k, fov_k = key.rsplit("_", 1)
            with st.expander(f"❌ {MODELS.get(model_k, model_k)} / {fov_k} — エラー詳細"):
                st.code(err_text, language="bash")

    st.divider()

# ── FOV selector ──────────────────────────────────────────────────────────────

view_fov = st.selectbox("比較するFOV", discovered if discovered else ["FOV00001"])

raw_dapi = load_raw_dapi(view_fov)
raw_ds   = downsample_img(norm_uint8(raw_dapi), 512) if raw_dapi is not None else None

# ── Comparison grid ───────────────────────────────────────────────────────────

has_any = any(
    _mask_path(m, view_fov, output_dir) is not None or
    st.session_state["seg_jobs"].get(f"{m}_{view_fov}")
    for m in MODELS
)

if not has_any:
    st.info("セグメンテーション結果がありません。左サイドバーでモデルを選択して **▶ Run segmentation** を押してください。")
    st.stop()

# Header
hdr = st.columns([1.2, 2, 2, 2])
for col, label in zip(hdr, ["モデル", "Raw DAPI", "Seg mask", "Overlay"]):
    col.markdown(f"**{label}**")
st.divider()

cell_counts: list[dict] = []

for model_key, model_label in MODELS.items():
    mask      = _load_mask_array(model_key, view_fov, output_dir)
    job_key   = f"{model_key}_{view_fov}"
    job_status = st.session_state["seg_jobs"].get(job_key, "")

    if mask is None and not job_status:
        continue

    n_cells = int(mask.max()) if mask is not None else 0
    if mask is not None:
        cell_counts.append({"Model": model_label, "Cells": n_cells})

    col_lbl, col_raw, col_mask, col_ov = st.columns([1.2, 2, 2, 2])

    with col_lbl:
        st.markdown(f"**{model_label}**")
        if job_status:
            st.markdown(_status_badge(job_status), unsafe_allow_html=True)
        if mask is not None:
            st.caption(f"{n_cells:,} cells")
        if job_status in ("running", "done", "error"):
            elapsed = _elapsed(job_key)
            if elapsed:
                st.caption(f"⏱ {elapsed}")

    with col_raw:
        if raw_ds is not None:
            st.image(raw_ds, use_container_width=True, clamp=True)
        else:
            st.caption("DAPI not found")

    with col_mask:
        if mask is not None:
            st.image(colorize_mask(downsample_mask(mask, 512)), use_container_width=True)
        elif job_status == "running":
            st.markdown(
                "<div style='background:#1e1e1e;border:1px solid #f0a500;border-radius:8px;"
                "padding:40px;text-align:center;color:#f0a500'>🔄 実行中…</div>",
                unsafe_allow_html=True,
            )
        elif job_status == "error":
            st.markdown(
                "<div style='background:#1e1e1e;border:1px solid #e74c3c;border-radius:8px;"
                "padding:40px;text-align:center;color:#e74c3c'>❌ エラー</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("—")

    with col_ov:
        if mask is not None and raw_dapi is not None:
            st.image(
                compose_overlay(downsample_img(norm_uint8(raw_dapi), 512), downsample_mask(mask, 512)),
                use_container_width=True,
            )
        elif job_status == "running":
            st.markdown(
                "<div style='background:#1e1e1e;border:1px dashed #555;border-radius:8px;"
                "padding:40px;text-align:center;color:#555'>マスク生成後に表示</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("—")

    st.divider()

# ── Cell count chart ──────────────────────────────────────────────────────────

if cell_counts:
    import plotly.express as px
    df = pd.DataFrame(cell_counts)
    st.subheader(f"細胞数比較 — {view_fov}")
    fig = px.bar(
        df, x="Model", y="Cells", color="Model",
        color_discrete_sequence=px.colors.qualitative.Set2,
        text="Cells",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=320, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
