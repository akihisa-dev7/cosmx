"""Segmentation Runner — input rawdata, run models, compare results."""
from __future__ import annotations

import json
import os
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
    load_enhanced,
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
DEFAULT_TX_CSV   = str(PROJECT_ROOT / "outputs_ssd" / "proseg_pilot" / "pilot_tx.csv.gz")
STATE_FILE       = PROJECT_ROOT / ".seg_state.json"
HISTORY_FILE     = PROJECT_ROOT / "outputs" / "run_history.jsonl"

# Directories searched (in order) for enhanced DAPI images
_ENHANCED_ROOTS = [
    PROJECT_ROOT / "outputs" / "pilot_4fov",
    PROJECT_ROOT / "outputs" / "enhanced_images",
]

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

# ── Session state init ────────────────────────────────────────────────────────

for _k, _v in [
    ("seg_jobs",       {}),
    ("seg_procs",      {}),   # key → Popen  (in-memory only, lost on reload)
    ("seg_pids",       {}),   # key → int PID  (serializable, saved to JSON)
    ("seg_params",     {}),   # key → params dict (saved for history)
    ("seg_errors",     {}),
    ("seg_starts",     {}),   # key → start timestamp (float)
    ("seg_logs",       {}),   # key → last N lines of stdout
    ("seg_is_running", False),
    ("_state_loaded",  False),
    ("detail_target",  None),  # {fov, model, type, output_dir} — drives detail viewer
    ("annotate_target", None), # passed to 01_Annotate.py via switch_page
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── State persistence helpers ─────────────────────────────────────────────────

def _save_state(output_dir: str) -> None:
    # Merge Popen PIDs into seg_pids
    merged_pids = dict(st.session_state.get("seg_pids", {}))
    for k, proc in st.session_state["seg_procs"].items():
        merged_pids[k] = proc.pid
    state = {
        "seg_jobs":       st.session_state["seg_jobs"],
        "seg_starts":     st.session_state["seg_starts"],
        "seg_errors":     st.session_state["seg_errors"],
        "seg_logs":       st.session_state["seg_logs"],
        "seg_is_running": st.session_state["seg_is_running"],
        "seg_pids":       merged_pids,
        "output_dir":     output_dir,
    }
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


def _load_state_from_file(output_dir: str) -> None:
    if not STATE_FILE.exists():
        return
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except Exception:
        return

    st.session_state["seg_jobs"]   = state.get("seg_jobs", {})
    st.session_state["seg_starts"] = state.get("seg_starts", {})
    st.session_state["seg_errors"] = state.get("seg_errors", {})
    st.session_state["seg_logs"]   = state.get("seg_logs", {})

    saved_pids   = state.get("seg_pids", {})
    saved_outdir = state.get("output_dir", output_dir)
    active_pids: dict[str, int] = {}

    for key, pid in saved_pids.items():
        try:
            os.kill(int(pid), 0)   # 0 = just probe, don't send a signal
            active_pids[key] = int(pid)
        except (ProcessLookupError, PermissionError, OSError):
            # Process is gone — finalise the job status
            if st.session_state["seg_jobs"].get(key) == "running":
                model, fov = key.rsplit("_", 1)
                mask = _mask_path(model, fov, saved_outdir)
                st.session_state["seg_jobs"][key] = (
                    "done" if (mask and mask.exists()) else "error"
                )

    st.session_state["seg_pids"]       = active_pids
    st.session_state["seg_is_running"] = bool(active_pids)


def _delete_state_file() -> None:
    try:
        STATE_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _append_run_history(key: str, status: str, output_dir: str) -> None:
    """Append one completed job record to the JSONL history file."""
    import datetime
    model, fov = key.rsplit("_", 1)

    # n_cells from DAPI mask file
    n_cells = 0
    mask = _mask_path(model, fov, output_dir)
    if mask and mask.exists():
        try:
            import tifffile
            n_cells = int(tifffile.imread(str(mask)).max())
        except Exception:
            pass

    # Baysor status: check mask file and stdout log for failure marker
    baysor_mask = _mask_path(model, fov, output_dir, baysor=True)
    baysor_n_cells = 0
    stdout_log = st.session_state.get("seg_logs", {}).get(key, "")
    if baysor_mask and baysor_mask.exists():
        baysor_status = "ok"
        try:
            import tifffile as _tf
            baysor_n_cells = int(_tf.imread(str(baysor_mask)).max())
        except Exception:
            pass
    elif "[BAYSOR_FAILED]" in stdout_log:
        baysor_status = "failed"
    else:
        baysor_status = "not_run"

    # elapsed time
    t0 = st.session_state["seg_starts"].get(key)
    elapsed_secs = int(time.time() - t0) if t0 else 0

    record = {
        "started_at":    datetime.datetime.fromtimestamp(t0).isoformat() if t0 else "",
        "finished_at":   datetime.datetime.now().isoformat(),
        "elapsed_secs":  elapsed_secs,
        "model":         model,
        "fov":           fov,
        "status":        status,
        "n_cells":       n_cells,
        "baysor_status": baysor_status,
        "baysor_n_cells": baysor_n_cells,
        "output_dir":    output_dir,
        "params":        st.session_state["seg_params"].get(key, {}),
    }
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _load_run_history() -> pd.DataFrame:
    """Load run_history.jsonl and return as DataFrame (newest first)."""
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    records = []
    try:
        with open(HISTORY_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        return pd.DataFrame()
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["finished_at"] = pd.to_datetime(df["finished_at"], errors="coerce")
    return df.sort_values("started_at", ascending=False).reset_index(drop=True)


def _detect_orphan_runs(output_dir: str) -> None:
    """Recover runs that were started without a state file (old code / hot-reload reset)."""
    # Scan for running run_segmentation.py processes
    try:
        result = subprocess.run(["pgrep", "-a", "python"], capture_output=True, text=True)
    except Exception:
        return

    running: dict[str, int] = {}   # key → pid
    for line in result.stdout.splitlines():
        if "run_segmentation.py" not in line:
            continue
        parts = line.split()
        try:
            pid = int(parts[0])
        except (ValueError, IndexError):
            continue
        model = fov = None
        for i, p in enumerate(parts):
            if p == "--model" and i + 1 < len(parts):
                model = parts[i + 1]
            if p == "--fov" and i + 1 < len(parts):
                fov = parts[i + 1]
        if model and fov:
            running[f"{model}_{fov}"] = pid

    if not running:
        return

    out_dir = Path(output_dir)
    for key, pid in running.items():
        if key in st.session_state["seg_jobs"]:
            continue   # already tracked
        st.session_state["seg_jobs"][key]  = "running"
        st.session_state["seg_pids"][key]  = pid
        st.session_state["seg_is_running"] = True
        # Estimate start time from temp file mtime
        out_file = out_dir / f".out_{key}.txt"
        if out_file.exists():
            st.session_state["seg_starts"][key] = out_file.stat().st_mtime
        else:
            st.session_state["seg_starts"][key] = time.time()

    # Persist so subsequent reruns don't need to re-scan
    _save_state(output_dir)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_display_dapi(fov_id: str, output_dir: str) -> tuple[np.ndarray | None, str]:
    """Return (image, label): enhanced DAPI when available, raw DAPI as fallback."""
    import tifffile
    search_dirs = list(_ENHANCED_ROOTS) + [Path(output_dir)]
    for root in search_dirs:
        enhanced_dir = root / "enhanced"
        if enhanced_dir.exists():
            candidates = sorted(enhanced_dir.glob(f"{fov_id}*enhanced*.tif"))
            if candidates:
                img = tifffile.imread(str(candidates[0]))
                return (img[0] if img.ndim == 3 else img), "Enhanced DAPI"
        # Also check for enhanced images directly under root/tif/ (enhanced_images layout)
        tif_dir = root / "tif"
        if tif_dir.exists():
            candidates = sorted(tif_dir.glob(f"{fov_id}*enhanced*.tif"))
            if candidates:
                img = tifffile.imread(str(candidates[0]))
                return (img[0] if img.ndim == 3 else img), "Enhanced DAPI"
    return load_raw_dapi(fov_id), "Raw DAPI"


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


def _mask_path(model: str, fov_id: str, output_dir: str, baysor: bool = False) -> Path | None:
    root = Path(output_dir) / "masks"
    hits = []
    for p in root.glob(f"{fov_id}*nuclear_mask.tif"):
        name = p.name
        has_model = (model.replace("cellpose_", "") in name or model in name)
        has_baysor = "baysor" in name
        if has_model and has_baysor == baysor:
            hits.append(p)
    return hits[0] if hits else None


def _load_mask_array(model: str, fov_id: str, output_dir: str, baysor: bool = False) -> np.ndarray | None:
    import tifffile
    p = _mask_path(model, fov_id, output_dir, baysor=baysor)
    return tifffile.imread(str(p)).astype(np.int32) if (p and p.exists()) else None


def _elapsed(key: str) -> str:
    t0 = st.session_state["seg_starts"].get(key)
    if t0 is None:
        return ""
    secs = int(time.time() - t0)
    return f"{secs // 60}:{secs % 60:02d}"


def _finalize_job(key: str, output_dir: str) -> None:
    model, fov = key.rsplit("_", 1)
    mask = _mask_path(model, fov, output_dir)
    status = "done" if (mask and mask.exists()) else "error"
    st.session_state["seg_jobs"][key] = status
    err_file = Path(output_dir) / f".err_{key}.txt"
    if err_file.exists():
        st.session_state["seg_errors"][key] = err_file.read_text()[-2000:]
        err_file.unlink(missing_ok=True)
    out_file = Path(output_dir) / f".out_{key}.txt"
    if out_file.exists():
        lines = out_file.read_text().splitlines()
        st.session_state["seg_logs"][key] = "\n".join(lines[-20:])
        out_file.unlink(missing_ok=True)
    _append_run_history(key, status, output_dir)


def _read_live_logs(output_dir: str) -> None:
    """Read .out / .err temp files for all currently-running jobs."""
    for key, status in st.session_state["seg_jobs"].items():
        if status != "running":
            continue
        out_file = Path(output_dir) / f".out_{key}.txt"
        err_file = Path(output_dir) / f".err_{key}.txt"
        if out_file.exists():
            try:
                lines = out_file.read_text(errors="replace").splitlines()
                if lines:
                    st.session_state["seg_logs"][key] = "\n".join(lines[-40:])
            except Exception:
                pass
        if err_file.exists():
            try:
                err_text = err_file.read_text(errors="replace")
                if err_text.strip():
                    st.session_state["seg_errors"][key] = err_text[-4000:]
            except Exception:
                pass


def _poll_jobs(output_dir: str) -> None:
    changed = False

    # Popen objects (same browser session, not persisted)
    for key, proc in list(st.session_state["seg_procs"].items()):
        if proc.poll() is not None:
            # Close file handles kept on the proc object
            for attr in ("_stdout_fh", "_stderr_fh"):
                fh = getattr(proc, attr, None)
                if fh:
                    try:
                        fh.close()
                    except Exception:
                        pass
            _finalize_job(key, output_dir)
            del st.session_state["seg_procs"][key]
            st.session_state["seg_pids"].pop(key, None)
            changed = True

    # PID-only tracking (after page reload — no Popen available)
    for key in list(st.session_state.get("seg_pids", {})):
        if key in st.session_state["seg_procs"]:
            continue   # already handled above
        pid = st.session_state["seg_pids"][key]
        try:
            os.kill(int(pid), 0)
        except (ProcessLookupError, PermissionError, OSError):
            _finalize_job(key, output_dir)
            del st.session_state["seg_pids"][key]
            changed = True

    # Release lock when nothing is left running
    if not st.session_state["seg_procs"] and not st.session_state.get("seg_pids"):
        if st.session_state["seg_is_running"]:
            st.session_state["seg_is_running"] = False
            changed = True

    # Always refresh live logs for running jobs
    _read_live_logs(output_dir)

    if changed:
        _save_state(output_dir)


def _build_cmd(
    model, fov_id, image_dir, output_dir, manifest, params,
    use_baysor: bool = False, baysor_params: dict | None = None,
) -> list[str]:
    cmd = [
        sys.executable, str(SEG_SCRIPT),
        "--input",   image_dir, "--output", output_dir,
        "--model",   model,     "--fov",    fov_id,
        "--expand_px",          str(params.get("expand_px", 5)),
        "--min_area_px",        str(params.get("min_area_px", 50)),
        "--diameter",           str(params.get("diameter", 25.0)),
        "--cellprob_threshold", str(params.get("cellprob_threshold", -1.0)),
        "--flow_threshold",     str(params.get("flow_threshold", 0.6)),
    ]
    if use_baysor and baysor_params:
        tx_csv = baysor_params.get("tx_csv", "").strip()
        if tx_csv:
            cmd += [
                "--baysor_refine",
                "--tx_csv",            tx_csv,
                "--scale_um",          str(baysor_params.get("scale_um", 10.0)),
                "--min_molecules",     str(baysor_params.get("min_molecules", 15)),
                "--prior_confidence",  str(baysor_params.get("prior_confidence", 0.5)),
                "--max_transcripts",   str(int(baysor_params.get("max_transcripts", 500_000))),
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


# ── Scale bar helpers ─────────────────────────────────────────────────────────

_PIXEL_SIZE_UM = 0.12028   # CosMx calibration: µm per full-resolution pixel
_FONT_PATH     = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _nice_bar_um(target_um: float) -> float:
    """Return a round µm value closest to target_um for a readable scale bar."""
    for v in [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500]:
        if v >= target_um * 0.5:
            return float(v)
    return 500.0


def _draw_scale_bar(
    img: np.ndarray,
    original_h: int = 4256,
    bar_um: float = 50.0,
) -> np.ndarray:
    """Overlay a white scale bar + label in the bottom-right corner."""
    from PIL import Image as _PImg, ImageDraw as _PDraw, ImageFont as _PFont

    h, w = img.shape[:2]
    scale = h / original_h                          # display / full-res
    bar_px = max(5, int(round(bar_um / _PIXEL_SIZE_UM * scale)))

    rgb = np.stack([img, img, img], axis=-1) if img.ndim == 2 else img.copy()
    pil  = _PImg.fromarray(rgb.astype(np.uint8))
    draw = _PDraw.Draw(pil)

    margin, thick = 8, 4
    x2, y_bar = w - margin, h - margin - thick
    x1 = x2 - bar_px

    draw.rectangle([x1 - 1, y_bar - 1, x2 + 1, y_bar + thick + 1], fill=(0, 0, 0))
    draw.rectangle([x1, y_bar, x2, y_bar + thick],                   fill=(255, 255, 255))

    label = f"{int(bar_um)} µm"
    try:
        font  = _PFont.truetype(_FONT_PATH, 11)
        bbox  = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        font  = None
        tw, th = len(label) * 6, 10

    tx = (x1 + x2) // 2 - tw // 2
    ty = y_bar - th - 3
    draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0),       font=font)
    draw.text((tx,     ty),     label, fill=(255, 255, 255),  font=font)

    return np.array(pil)


# ── Load persisted state (once per browser session) ───────────────────────────

if not st.session_state["_state_loaded"]:
    _load_state_from_file(DEFAULT_OUT)
    # Fallback: detect runs not covered by state file (old code / hot-reload reset)
    _detect_orphan_runs(DEFAULT_OUT)
    st.session_state["_state_loaded"] = True

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Input")
    is_running = st.session_state["seg_is_running"]

    if is_running:
        st.info("実行中のため設定は変更できません")

    image_dir  = st.text_input("Raw image directory", value=DEFAULT_IMG,      disabled=is_running)
    output_dir = st.text_input("Output directory",    value=DEFAULT_OUT,      disabled=is_running)
    manifest   = st.text_input("Manifest CSV",        value=DEFAULT_MANIFEST, disabled=is_running)

    st.divider()
    st.subheader("FOV")
    discovered = _discover_fovs(image_dir)
    sel_fovs   = st.multiselect("FOVs to process", discovered, default=discovered[:1], disabled=is_running)

    st.divider()
    st.subheader("Models & parameters")
    model_params: dict[str, dict] = {}
    sel_models:   list[str]       = []
    for model_key, model_label in MODELS.items():
        enabled = st.checkbox(model_label, value=(model_key == "cellpose_cpsam"), disabled=is_running)
        if enabled:
            sel_models.append(model_key)
            with st.expander(f"{model_label} parameters"):
                p: dict = {}
                if model_key.startswith("cellpose"):
                    # diameter=75px ≈ 9µm at CosMx 0.12µm/px (brain nuclei median)
                    p["diameter"]           = st.number_input("diameter (px)",       value=75.0, step=5.0,  key=f"{model_key}_d",  disabled=is_running,
                                                              help="脳核の中央値は約 9µm = 75px (CosMx 0.12µm/px)")
                    p["cellprob_threshold"] = st.number_input("cellprob_threshold", value=-0.5, step=0.1,  key=f"{model_key}_cp", disabled=is_running,
                                                              help="低いほど多く検出。-0.5 が脳核の推奨値")
                    p["flow_threshold"]     = st.number_input("flow_threshold",     value=0.4,  step=0.05, key=f"{model_key}_ft", disabled=is_running,
                                                              help="高いほど厳格。0.4 で断片を除去")
                p["expand_px"]   = st.number_input("expand_px",   value=5,  step=1, key=f"{model_key}_ep", disabled=is_running)
                p["min_area_px"] = st.number_input("min_area_px", value=200, step=50, key=f"{model_key}_ma", disabled=is_running,
                                                   help="脳核の最小面積。200px²≈直径16px≈1.9µm")
                model_params[model_key] = p

    st.divider()
    st.subheader("RNA誘導 精度向上")
    use_baysor = st.checkbox(
        "Baysor RNA-guided refinement",
        value=False, disabled=is_running,
        help="各モデルの DAPI マスクを prior として Baysor で RNA 誘導精度向上を実行",
    )
    baysor_params: dict = {}
    if use_baysor:
        with st.expander("Baysor パラメータ"):
            baysor_params["tx_csv"] = st.text_input(
                "Transcript CSV (tx_file)", value=DEFAULT_TX_CSV,
                key="baysor_tx", disabled=is_running,
                help="CosMx tx_file CSV/CSV.gz（x_local_px, y_local_px, target 列必須）",
            )
            baysor_params["scale_um"]         = st.number_input("scale_um (µm)",        value=10.0, step=1.0,  key="baysor_su", disabled=is_running)
            baysor_params["min_molecules"]    = st.number_input("min_molecules",         value=15,   step=1,    key="baysor_mm", disabled=is_running)
            baysor_params["prior_confidence"] = st.number_input("prior_confidence (0–1)", value=0.5, step=0.05, key="baysor_pc", min_value=0.0, max_value=1.0, disabled=is_running)
            baysor_params["max_transcripts"]  = st.number_input(
                "max_transcripts (0=無制限)",
                value=500_000, step=100_000, min_value=0, key="baysor_mt", disabled=is_running,
                help="FOVあたりの最大転写産物数。Julia 1.10 GC segfaultを防ぐためサブサンプリング。"
                     "推奨: 500,000（クラッシュ時は自動3回リトライ）。0=制限なし（クラッシュリスク高）",
            )

    st.divider()

    is_running = st.session_state["seg_is_running"]

    run_btn  = st.button(
        "▶ Run segmentation",
        type="primary",
        use_container_width=True,
        disabled=is_running,
    )
    stop_btn = st.button(
        "⏹ 停止",
        use_container_width=True,
        disabled=not is_running,
        type="secondary",
    )
    reset_btn = st.button("🗑 Clear status", use_container_width=True)

    if stop_btn:
        for proc in st.session_state["seg_procs"].values():
            try:
                proc.kill()
            except Exception:
                pass
        # Also kill PID-tracked processes (post-reload)
        for pid in st.session_state["seg_pids"].values():
            try:
                os.kill(int(pid), 9)
            except Exception:
                pass
        for k in ["seg_jobs", "seg_procs", "seg_pids", "seg_errors", "seg_starts", "seg_logs"]:
            st.session_state[k] = {}
        st.session_state["seg_is_running"] = False
        _delete_state_file()
        st.rerun()

    if reset_btn:
        for k in ["seg_jobs", "seg_procs", "seg_pids", "seg_errors", "seg_starts", "seg_logs"]:
            st.session_state[k] = {}
        st.session_state["seg_is_running"] = False
        _delete_state_file()
        st.rerun()


# ── Launch jobs ───────────────────────────────────────────────────────────────

if run_btn and not st.session_state["seg_is_running"]:
    if not SEG_SCRIPT.exists():
        st.error(f"Script not found: {SEG_SCRIPT}")
    elif not sel_fovs:
        st.sidebar.warning("FOVを1つ以上選択してください。")
    elif not sel_models:
        st.sidebar.warning("モデルを1つ以上選択してください。")
    elif use_baysor and not baysor_params.get("tx_csv", "").strip():
        st.sidebar.warning("Baysor RNA誘導には Transcript CSV パスを入力してください。")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for model in sel_models:
            for fov in sel_fovs:
                key = f"{model}_{fov}"
                st.session_state["seg_errors"].pop(key, None)
                st.session_state["seg_logs"].pop(key, None)
        for model in sel_models:
            for fov in sel_fovs:
                key      = f"{model}_{fov}"
                cmd      = _build_cmd(model, fov, image_dir, output_dir, manifest, model_params.get(model, {}), use_baysor, baysor_params)
                err_file = Path(output_dir) / f".err_{key}.txt"
                out_file = Path(output_dir) / f".out_{key}.txt"
                fout = open(out_file, "w")
                ferr = open(err_file, "w")
                proc = subprocess.Popen(cmd, stdout=fout, stderr=ferr)
                # Keep file handles on the proc object so they stay open while the
                # child writes, and are closed when the Popen is garbage-collected.
                proc._stdout_fh = fout
                proc._stderr_fh = ferr
                st.session_state["seg_jobs"][key]   = "running"
                st.session_state["seg_procs"][key]  = proc
                st.session_state["seg_pids"][key]   = proc.pid
                st.session_state["seg_starts"][key] = time.time()
                st.session_state["seg_params"][key] = model_params.get(model, {})
        st.session_state["seg_is_running"] = True
        _save_state(output_dir)
        st.rerun()

# Poll & auto-refresh
_poll_jobs(output_dir)
if st.session_state["seg_procs"] or st.session_state.get("seg_pids"):
    time.sleep(2)
    st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Segmentation Runner")

# ── Environment status ────────────────────────────────────────────────────────

with st.expander("環境ステータス", expanded=False):
    try:
        import torch as _torch
        _gpu_ok  = _torch.cuda.is_available()
        _gpu_name = _torch.cuda.get_device_name(0) if _gpu_ok else "N/A"
        _vram_free  = _torch.cuda.mem_get_info()[0] / 1024**3 if _gpu_ok else 0
        _vram_total = _torch.cuda.get_device_properties(0).total_memory / 1024**3 if _gpu_ok else 0
    except Exception:
        _gpu_ok = False; _gpu_name = "torch not installed"; _vram_free = 0; _vram_total = 0

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        if _gpu_ok:
            st.success(f"GPU: {_gpu_name}")
        else:
            st.error("GPU 利用不可 — CPU で実行されます（非常に低速）")
    with ec2:
        if _gpu_ok:
            st.metric("VRAM 空き", f"{_vram_free:.1f} GB", f"/ {_vram_total:.1f} GB 合計")
        else:
            st.metric("VRAM", "—")
    with ec3:
        if SEG_SCRIPT.exists():
            st.success(f"スクリプト: OK")
        else:
            st.error(f"スクリプト未検出: {SEG_SCRIPT}")

    if not _gpu_ok:
        st.warning(
            "PyTorch が GPU を認識していません。以下を実行して CUDA 対応版に再インストールしてください:\n\n"
            "```bash\n"
            "pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 "
            "--index-url https://download.pytorch.org/whl/cu121\n"
            "```"
        )

# ── Progress panel ────────────────────────────────────────────────────────────

jobs = st.session_state["seg_jobs"]

if jobs:
    total  = len(jobs)
    n_done = sum(1 for s in jobs.values() if s == "done")
    n_err  = sum(1 for s in jobs.values() if s == "error")
    n_run  = sum(1 for s in jobs.values() if s == "running")
    n_wait = sum(1 for s in jobs.values() if s == "waiting")
    pct    = n_done / total

    st.subheader("実行状況")

    # Overall progress bar
    prog_col, stat_col = st.columns([3, 1])
    with prog_col:
        st.progress(pct, text=f"{n_done} / {total} 完了")
    with stat_col:
        parts = []
        if n_run:  parts.append(f"🔄 {n_run}")
        if n_done: parts.append(f"✅ {n_done}")
        if n_err:  parts.append(f"❌ {n_err}")
        if n_wait: parts.append(f"⏳ {n_wait}")
        st.markdown("  ".join(parts))

    # Per-job table
    rows_html = ""
    for key, status in sorted(jobs.items()):
        model_k, fov_k = key.rsplit("_", 1)
        label   = MODELS.get(model_k, model_k)
        elapsed = _elapsed(key) if status in ("running", "done", "error") else "—"
        badge   = _status_badge(status)

        # Last stdout line
        log_preview = st.session_state["seg_logs"].get(key, "")
        last_line   = log_preview.splitlines()[-1] if log_preview else ""
        # Truncate and HTML-escape
        last_line_disp = last_line[:80].replace("<", "&lt;").replace(">", "&gt;")

        # Show first line of stderr when errored
        err_preview = st.session_state["seg_errors"].get(key, "")
        err_first = ""
        if status == "error" and err_preview:
            for ln in reversed(err_preview.splitlines()):
                ln = ln.strip()
                if ln and not ln.startswith("Warning") and "UserWarning" not in ln:
                    err_first = ln[:100].replace("<", "&lt;").replace(">", "&gt;")
                    break

        last_col = (
            f"<span style='color:#e74c3c'>{err_first}</span>"
            if err_first else last_line_disp
        )

        rows_html += (
            f"<tr>"
            f"<td style='padding:6px 12px'>{label}</td>"
            f"<td style='padding:6px 12px;color:#888'>{fov_k}</td>"
            f"<td style='padding:6px 12px'>{badge}</td>"
            f"<td style='padding:6px 12px;font-family:monospace;color:#aaa;font-size:0.8em'>{elapsed}</td>"
            f"<td style='padding:6px 12px;font-family:monospace;font-size:0.75em;color:#ccc;"
            f"max-width:400px;overflow:hidden;white-space:nowrap'>{last_col}</td>"
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
              <th style='padding:8px 12px;text-align:left'>最終ログ / エラー</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.write("")  # spacer

    # ── Per-job log / error panels ────────────────────────────────────────────
    any_log = any(
        st.session_state["seg_logs"].get(k) or st.session_state["seg_errors"].get(k)
        for k in jobs
    )
    if any_log:
        for key, status in sorted(jobs.items()):
            model_k, fov_k = key.rsplit("_", 1)
            label = MODELS.get(model_k, model_k)

            stdout_text = st.session_state["seg_logs"].get(key, "")
            stderr_text = st.session_state["seg_errors"].get(key, "")

            # Determine expander icon and whether to auto-expand
            if status == "error":
                icon = "❌"
                auto_open = True
            elif status == "running":
                icon = "🔄"
                auto_open = True
            elif status == "done":
                icon = "✅"
                auto_open = False
            else:
                continue

            if not stdout_text and not stderr_text:
                continue

            with st.expander(f"{icon} {label} / {fov_k}", expanded=auto_open):
                tabs = []
                if stdout_text:
                    tabs.append("stdout")
                if stderr_text:
                    tabs.append("stderr")
                if not tabs:
                    continue

                tab_objects = st.tabs(tabs)
                idx = 0
                if stdout_text:
                    with tab_objects[idx]:
                        st.code(stdout_text, language="bash")
                    idx += 1
                if stderr_text:
                    with tab_objects[idx]:
                        # Highlight error lines in red via st.error if they contain ERROR/Exception
                        lines = stderr_text.splitlines()
                        error_lines = [l for l in lines if any(
                            kw in l for kw in ("Error", "Exception", "Traceback", "CUDA", "OOM", "killed")
                        )]
                        if error_lines and status == "error":
                            st.error("\n".join(error_lines[-5:]))
                        st.code(stderr_text, language="bash")

    # ── Global error summary when any jobs failed ─────────────────────────────
    failed_keys = [k for k, s in jobs.items() if s == "error"]
    if failed_keys:
        lines = []
        for key in failed_keys:
            model_k, fov_k = key.rsplit("_", 1)
            label    = MODELS.get(model_k, model_k)
            err_text = st.session_state["seg_errors"].get(key, "")
            # Extract the most meaningful error line
            best = ""
            for ln in reversed(err_text.splitlines()):
                ln = ln.strip()
                if ln and not ln.startswith("#") and "UserWarning" not in ln and ln != "":
                    best = ln
                    break
            lines.append(f"**{label} / {fov_k}**: {best[:120] if best else '(詳細はログを参照)'}")
        st.error("セグメンテーションエラー\n\n" + "\n\n".join(lines))

    st.divider()

# Baysor failure notice (job finished OK but Baysor sub-step failed)
_baysor_failures = []
for _key, _log in st.session_state.get("seg_logs", {}).items():
    if "[BAYSOR_FAILED]" in _log:
        _model_k, _fov_k = _key.rsplit("_", 1)
        _reason = next(
            (ln.split("[BAYSOR_FAILED]", 1)[1].strip() for ln in _log.splitlines() if "[BAYSOR_FAILED]" in ln),
            "(詳細はログを参照)",
        )
        _baysor_failures.append(f"**{MODELS.get(_model_k, _model_k)} / {_fov_k}**: {_reason[:120]}")
if _baysor_failures:
    st.warning(
        "⚠️ Baysor RNA誘導精度向上が失敗しました（DAPI-onlyマスクを使用中）\n\n"
        + "\n\n".join(_baysor_failures)
        + "\n\n詳細はログパネルのstderr/stdoutを確認してください。"
    )

# ── FOV selector ──────────────────────────────────────────────────────────────

view_fov = st.selectbox("比較するFOV", discovered if discovered else ["FOV00001"])

display_dapi, dapi_label = _load_display_dapi(view_fov, output_dir)
raw_ds = downsample_img(norm_uint8(display_dapi), 512) if display_dapi is not None else None

# ── Comparison grid ───────────────────────────────────────────────────────────

has_any = any(
    _mask_path(m, view_fov, output_dir) is not None or
    st.session_state["seg_jobs"].get(f"{m}_{view_fov}")
    for m in MODELS
)

if not has_any:
    st.info("セグメンテーション結果がありません。左サイドバーでモデルを選択して **▶ Run segmentation** を押してください。")
    st.stop()

# Detect if any Baysor-refined masks exist for this FOV
show_baysor_col = any(
    _mask_path(m, view_fov, output_dir, baysor=True) is not None
    for m in MODELS
)

if show_baysor_col:
    hdr = st.columns([1.2, 1.5, 1.5, 1.5, 1.5])
    for col, label in zip(hdr, ["モデル", dapi_label, "DAPI mask", "Baysor mask", "Baysor Overlay"]):
        col.markdown(f"**{label}**")
else:
    hdr = st.columns([1.2, 2, 2, 2])
    for col, label in zip(hdr, ["モデル", dapi_label, "Seg mask", "Overlay"]):
        col.markdown(f"**{label}**")
st.divider()

cell_counts: list[dict] = []

_RUNNING_DIV  = ("<div style='background:#1e1e1e;border:1px solid #f0a500;border-radius:8px;"
                 "padding:30px;text-align:center;color:#f0a500'>🔄 実行中…</div>")
_ERROR_DIV    = ("<div style='background:#1e1e1e;border:1px solid #e74c3c;border-radius:8px;"
                 "padding:30px;text-align:center;color:#e74c3c'>❌ エラー</div>")
_PENDING_DIV  = ("<div style='background:#1e1e1e;border:1px dashed #555;border-radius:8px;"
                 "padding:30px;text-align:center;color:#555'>マスク生成後に表示</div>")

for model_key, model_label in MODELS.items():
    mask        = _load_mask_array(model_key, view_fov, output_dir, baysor=False)
    baysor_mask = _load_mask_array(model_key, view_fov, output_dir, baysor=True) if show_baysor_col else None
    job_key     = f"{model_key}_{view_fov}"
    job_status  = st.session_state["seg_jobs"].get(job_key, "")

    if mask is None and not job_status:
        continue

    n_cells = int(mask.max()) if mask is not None else 0
    if mask is not None:
        cell_counts.append({"Model": model_label, "Type": "DAPI", "Cells": n_cells})
    if baysor_mask is not None:
        cell_counts.append({"Model": model_label, "Type": "Baysor", "Cells": int(baysor_mask.max())})

    if show_baysor_col:
        col_lbl, col_raw, col_mask, col_baysor, col_ov = st.columns([1.2, 1.5, 1.5, 1.5, 1.5])
    else:
        col_lbl, col_raw, col_mask, col_ov = st.columns([1.2, 2, 2, 2])

    with col_lbl:
        st.markdown(f"**{model_label}**")
        if job_status:
            st.markdown(_status_badge(job_status), unsafe_allow_html=True)
        if mask is not None:
            st.caption(f"DAPI: {n_cells:,} cells")
        if baysor_mask is not None:
            st.caption(f"Baysor: {int(baysor_mask.max()):,} cells")
        if job_status in ("running", "done", "error"):
            elapsed = _elapsed(job_key)
            if elapsed:
                st.caption(f"⏱ {elapsed}")

    with col_raw:
        if raw_ds is not None:
            st.image(_draw_scale_bar(raw_ds), use_container_width=True, clamp=True)
            if st.button("🔍 詳細", key=f"det_dapi_{model_key}", use_container_width=True):
                st.session_state["detail_target"] = {
                    "fov": view_fov, "model": model_key,
                    "type": "dapi", "output_dir": output_dir,
                }
                st.switch_page("pages/02_ImageViewer.py")
        else:
            st.caption("DAPI not found")

    with col_mask:
        if mask is not None:
            st.image(colorize_mask(downsample_mask(mask, 512)), use_container_width=True)
            if st.button("🔍 詳細", key=f"det_mask_{model_key}", use_container_width=True):
                st.session_state["detail_target"] = {
                    "fov": view_fov, "model": model_key,
                    "type": "mask", "output_dir": output_dir,
                }
                st.switch_page("pages/02_ImageViewer.py")
        elif job_status == "running":
            st.markdown(_RUNNING_DIV, unsafe_allow_html=True)
        elif job_status == "error":
            st.markdown(_ERROR_DIV, unsafe_allow_html=True)
        else:
            st.caption("—")

    if show_baysor_col:
        with col_baysor:
            if baysor_mask is not None:
                st.image(colorize_mask(downsample_mask(baysor_mask, 512)), use_container_width=True)
                if st.button("🔍 詳細", key=f"det_baysor_{model_key}", use_container_width=True):
                    st.session_state["detail_target"] = {
                        "fov": view_fov, "model": model_key,
                        "type": "baysor_mask", "output_dir": output_dir,
                    }
                    st.switch_page("pages/02_ImageViewer.py")
            elif job_status == "running":
                st.markdown(_RUNNING_DIV, unsafe_allow_html=True)
            else:
                st.caption("—")

    with col_ov:
        ov_mask = baysor_mask if (show_baysor_col and baysor_mask is not None) else mask
        if ov_mask is not None and display_dapi is not None:
            ov_img = compose_overlay(
                downsample_img(norm_uint8(display_dapi), 512),
                downsample_mask(ov_mask, 512),
            )
            st.image(_draw_scale_bar(ov_img), use_container_width=True)
            if st.button("✏️ Annotate", key=f"ann_{model_key}", use_container_width=True, type="primary"):
                st.session_state["annotate_target"] = {
                    "fov": view_fov, "model": model_key, "output_dir": output_dir,
                    "use_baysor": show_baysor_col and baysor_mask is not None,
                }
                st.switch_page("pages/01_Annotate.py")
        elif job_status == "running":
            st.markdown(_PENDING_DIV, unsafe_allow_html=True)
        else:
            st.caption("—")

    st.divider()

# ── Cell count chart ──────────────────────────────────────────────────────────

if cell_counts:
    import plotly.express as px
    df = pd.DataFrame(cell_counts)
    df["Label"] = df["Model"] + " (" + df["Type"] + ")"
    st.subheader(f"細胞数比較 — {view_fov}")
    fig = px.bar(
        df, x="Label", y="Cells", color="Type",
        color_discrete_map={"DAPI": "#4e9af1", "Baysor": "#2ecc71"},
        text="Cells",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=True, height=320, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ── Run History ───────────────────────────────────────────────────────────────

st.divider()
st.header("実行履歴")

hist_df = _load_run_history()

if hist_df.empty:
    st.info("まだ実行履歴がありません。セグメンテーションを実行すると自動で記録されます。")
else:
    import plotly.express as px

    # ── Filters ──────────────────────────────────────────────────────────────
    hf1, hf2, hf3 = st.columns([2, 2, 1])
    with hf1:
        all_models = sorted(hist_df["model"].dropna().unique().tolist())
        sel_hist_models = st.multiselect(
            "モデルで絞り込み", all_models, default=all_models, key="hist_model_filter"
        )
    with hf2:
        all_fovs = sorted(hist_df["fov"].dropna().unique().tolist())
        sel_hist_fovs = st.multiselect(
            "FOVで絞り込み", all_fovs, default=all_fovs, key="hist_fov_filter"
        )
    with hf3:
        sel_hist_status = st.multiselect(
            "ステータス", ["done", "error"], default=["done", "error"], key="hist_status_filter"
        )

    filt = hist_df.copy()
    if sel_hist_models:
        filt = filt[filt["model"].isin(sel_hist_models)]
    if sel_hist_fovs:
        filt = filt[filt["fov"].isin(sel_hist_fovs)]
    if sel_hist_status:
        filt = filt[filt["status"].isin(sel_hist_status)]

    st.caption(f"{len(filt)} / {len(hist_df)} 件表示")

    # ── Table ─────────────────────────────────────────────────────────────────
    _has_baysor_col = "baysor_status" in filt.columns

    _disp_cols = ["started_at", "model", "fov", "status", "n_cells"]
    _disp_names = ["開始時刻", "モデル", "FOV", "DAPIセグ", "DAPI細胞数"]
    if _has_baysor_col:
        _disp_cols += ["baysor_status", "baysor_n_cells"]
        _disp_names += ["Baysorセグ", "Baysor細胞数"]
    _disp_cols.append("elapsed_secs")
    _disp_names.append("実行時間(秒)")

    disp = filt[_disp_cols].copy()
    disp.columns = _disp_names
    disp["開始時刻"] = disp["開始時刻"].dt.strftime("%Y-%m-%d %H:%M:%S")

    if _has_baysor_col:
        # Format Baysor status: ok → ✅ N cells, failed → ❌ failed, not_run → —
        def _fmt_baysor_status(row):
            bs = row.get("Baysorセグ", "not_run") or "not_run"
            bn = row.get("Baysor細胞数", 0) or 0
            if bs == "ok":
                return f"✅ {int(bn):,} cells"
            elif bs == "failed":
                return "❌ failed"
            else:
                return "—"
        disp["Baysorセグ"] = disp[["Baysorセグ", "Baysor細胞数"]].apply(_fmt_baysor_status, axis=1)
        disp = disp.drop(columns=["Baysor細胞数"])

    def _color_status(val: str) -> str:
        if val in ("done", "✅"):
            return "font-weight:bold;color:#2ecc71"
        if val in ("error", "❌ failed"):
            return "font-weight:bold;color:#e74c3c"
        return ""

    style_cols = ["DAPIセグ"]
    if _has_baysor_col:
        style_cols.append("Baysorセグ")

    st.dataframe(
        disp.style.applymap(_color_status, subset=style_cols),
        use_container_width=True,
        height=min(400, 40 + len(disp) * 35),
        hide_index=True,
    )

    # CSV download
    st.download_button(
        "CSV ダウンロード",
        data=filt.drop(columns=["params"], errors="ignore").to_csv(index=False),
        file_name="run_history.csv",
        mime="text/csv",
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    done_df = filt[filt["status"] == "done"].copy()
    if not done_df.empty:
        done_df["run_label"] = (
            done_df["started_at"].dt.strftime("%m/%d %H:%M") + " " + done_df["fov"]
        )

        tab_bar, tab_trend = st.tabs(["モデル別 細胞数", "時系列トレンド"])

        with tab_bar:
            # Group by model × fov: latest run per combo
            latest = (
                done_df.sort_values("started_at")
                .groupby(["model", "fov"], as_index=False)
                .last()
            )

            # Build long-form data: DAPI row + Baysor row (if available) per model×fov
            _bar_rows = []
            for _, r in latest.iterrows():
                _bar_rows.append({
                    "fov":    r["fov"],
                    "series": f"{MODELS.get(r['model'], r['model'])} (DAPI)",
                    "cells":  int(r.get("n_cells", 0) or 0),
                })
                _bs = r.get("baysor_status", "not_run") or "not_run"
                _bn = int(r.get("baysor_n_cells", 0) or 0)
                if _bs == "ok" and _bn > 0:
                    _bar_rows.append({
                        "fov":    r["fov"],
                        "series": f"{MODELS.get(r['model'], r['model'])} (Baysor)",
                        "cells":  _bn,
                    })
            import pandas as _pd2
            _bar_df = _pd2.DataFrame(_bar_rows)

            fig_bar = px.bar(
                _bar_df, x="fov", y="cells", color="series", barmode="group",
                text="cells",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"cells": "細胞数", "fov": "FOV", "series": "モデル / 手法"},
                title="最新 run のモデル別細胞数（DAPI・Baysor）",
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(height=380, margin=dict(t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab_trend:
            if len(done_df) < 2:
                st.info("トレンドには 2 件以上の完了履歴が必要です。")
            else:
                fig_trend = px.line(
                    done_df.sort_values("started_at"),
                    x="started_at", y="n_cells",
                    color="model", symbol="fov",
                    markers=True,
                    labels={"started_at": "実行日時", "n_cells": "細胞数", "model": "モデル"},
                    title="細胞数の時系列変化（パラメータ調整の効果確認）",
                )
                fig_trend.update_layout(height=400, margin=dict(t=40, b=20))
                st.plotly_chart(fig_trend, use_container_width=True)

    # ── Params diff (compare runs of same model/fov) ──────────────────────────
    if "params" in filt.columns and not done_df.empty:
        with st.expander("パラメータ詳細 / 差分を確認"):
            combo_options = (
                done_df.assign(combo=done_df["model"] + " / " + done_df["fov"])
                ["combo"].unique().tolist()
            )
            sel_combo = st.selectbox("モデル / FOV を選択", combo_options, key="hist_combo")
            if sel_combo:
                sel_m, sel_f = sel_combo.split(" / ", 1)
                combo_rows = filt[
                    (filt["model"] == sel_m) & (filt["fov"] == sel_f)
                ].sort_values("started_at", ascending=False)

                params_records = []
                for _, row in combo_rows.iterrows():
                    p = row.get("params") or {}
                    if isinstance(p, dict):
                        rec = {"開始時刻": str(row["started_at"])[:19],
                               "ステータス": row["status"],
                               "細胞数": row["n_cells"]}
                        rec.update(p)
                        params_records.append(rec)

                if params_records:
                    st.dataframe(
                        pd.DataFrame(params_records),
                        use_container_width=True,
                        hide_index=True,
                    )
