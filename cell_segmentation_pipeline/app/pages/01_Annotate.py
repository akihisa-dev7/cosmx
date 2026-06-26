"""Annotation launcher — starts web_annotator/server.py and opens it in the browser."""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # project root

from utils.image_loader import PROJECT_ROOT

st.set_page_config(page_title="Cell Annotator", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────

SERVER_PORT = 8790
SERVER_HOST = "127.0.0.1"
SERVER_URL  = f"http://{SERVER_HOST}:{SERVER_PORT}"
SERVER_PY   = PROJECT_ROOT / "annotation_tools" / "web_annotator" / "server.py"
PYTHON_BIN  = "/home/fujiyamaakihisa/miniconda3/envs/cosmx_annotation/bin/python"
GT_DIR      = PROJECT_ROOT / "annotation_tools" / "ground_truth"

_ENHANCED_ROOTS = [
    PROJECT_ROOT / "outputs" / "pilot_4fov" / "enhanced",
    PROJECT_ROOT / "outputs" / "enhanced_images" / "tif",
]

# ── Session state ─────────────────────────────────────────────────────────────

if "annotator_pid" not in st.session_state:
    st.session_state["annotator_pid"] = None

# ── Helpers ───────────────────────────────────────────────────────────────────

def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((SERVER_HOST, port)) == 0


def _start_server(masks_dir: Path, enhanced_dir: Path) -> bool:
    if _port_in_use(SERVER_PORT):
        return True

    env = os.environ.copy()
    env["ANNOTATOR_MASKS_DIR"]    = str(masks_dir)
    env["ANNOTATOR_ENHANCED_DIR"] = str(enhanced_dir)
    env["ANNOTATOR_GT_DIR"]       = str(GT_DIR)

    proc = subprocess.Popen(
        [PYTHON_BIN, str(SERVER_PY),
         "--port", str(SERVER_PORT),
         "--host", SERVER_HOST],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    st.session_state["annotator_pid"] = proc.pid

    for _ in range(12):
        if _port_in_use(SERVER_PORT):
            return True
        time.sleep(0.5)
    return False


def _stop_server() -> None:
    pid = st.session_state.get("annotator_pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        st.session_state["annotator_pid"] = None

# ── Check annotate_target ─────────────────────────────────────────────────────

target = st.session_state.get("annotate_target")

if target is None:
    st.warning("Annotate対象が設定されていません。Segmentation Runner で ✏️ Annotate を押してください。")
    if st.button("← Segmentation Runner に戻る"):
        st.switch_page("pages/00_Segmentation_Runner.py")
    st.stop()

fov_id     = target["fov"]
model_key  = target["model"]
output_dir = target["output_dir"]
use_baysor = target.get("use_baysor", False)

masks_dir = Path(output_dir) / "masks"

enhanced_dir = _ENHANCED_ROOTS[0]
for root in _ENHANCED_ROOTS:
    if root.exists() and any(root.glob(f"{fov_id}*enhanced*.tif")):
        enhanced_dir = root
        break

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("✏️ Cell Annotator")
    st.caption(f"FOV: **{fov_id}**")
    st.caption(f"Model: **{model_key}**")
    if use_baysor:
        st.caption("🧬 Baysor mask 使用中")

    st.divider()

    if st.button("← Runner に戻る", use_container_width=True):
        st.switch_page("pages/00_Segmentation_Runner.py")

    if st.button("🔍 Image Viewer に切り替え", use_container_width=True):
        st.session_state["detail_target"] = {
            "fov": fov_id, "model": model_key,
            "type": "baysor_mask" if use_baysor else "mask",
            "output_dir": output_dir,
        }
        st.switch_page("pages/02_ImageViewer.py")

    st.divider()
    st.subheader("サーバー情報")
    running = _port_in_use(SERVER_PORT)
    if running:
        st.success(f"稼働中 — port {SERVER_PORT}")
    else:
        st.warning("停止中")

    if st.button("🔄 再起動", use_container_width=True):
        _stop_server()
        time.sleep(0.5)
        _start_server(masks_dir, enhanced_dir)
        st.rerun()

    if st.button("⏹ 停止", use_container_width=True):
        _stop_server()
        st.rerun()

    st.divider()
    st.caption(f"Masks: `{masks_dir.name}`")
    st.caption(f"Enhanced: `{enhanced_dir.name}`")
    st.caption(f"保存先: `annotation_tools/ground_truth/`")

# ── Main ──────────────────────────────────────────────────────────────────────

st.title(f"✏️ Cell Annotator — {fov_id}")

# Start server if not already running
running = _port_in_use(SERVER_PORT)
if not running:
    with st.spinner(f"Web Annotatorサーバーを起動中…"):
        running = _start_server(masks_dir, enhanced_dir)

if not running:
    st.error(f"サーバー起動失敗。ポート {SERVER_PORT} が使用中か、または Python 環境を確認してください。")
    st.code(
        f"# 手動で起動する場合:\n"
        f"ANNOTATOR_MASKS_DIR='{masks_dir}' \\\n"
        f"ANNOTATOR_ENHANCED_DIR='{enhanced_dir}' \\\n"
        f"{PYTHON_BIN} {SERVER_PY} --port {SERVER_PORT}",
        language="bash",
    )
    st.stop()

# Show open button and instructions
st.success(f"✅ Web Annotator 起動済み — port **{SERVER_PORT}**")

open_col, info_col = st.columns([1, 2])

with open_col:
    # JavaScript で新規タブを開く
    st.markdown(
        f"""
        <a href="{SERVER_URL}" target="_blank">
            <button style="
                background:#2ecc71;color:white;border:none;padding:14px 28px;
                font-size:1.1em;font-weight:bold;border-radius:8px;cursor:pointer;
                width:100%;margin-top:8px
            ">🌐 Annotatorを新規タブで開く</button>
        </a>
        """,
        unsafe_allow_html=True,
    )

with info_col:
    st.subheader("操作手順")
    st.markdown(
        f"""
1. 上のボタンでアノテーターを **新しいタブ** で開く
2. FOV セレクタで **{fov_id}** を選択
3. タイルをクリックして編集領域を選択
4. **ブラシ / 消去** で核を編集
5. **Save** ボタンで保存
   - 保存先: `annotation_tools/ground_truth/{fov_id}_gt_mask.tif`
        """,
    )

st.divider()

# Saved ground truth files summary
st.subheader("保存済みアノテーション")
gt_files = sorted(GT_DIR.glob("*.tif")) if GT_DIR.exists() else []
if gt_files:
    import tifffile, numpy as np
    rows = []
    for p in gt_files:
        try:
            m = tifffile.imread(str(p)).astype(np.int32)
            n = int(np.unique(m[m > 0]).shape[0])
        except Exception:
            n = "?"
        rows.append({"ファイル": p.name, "細胞数": n, "サイズ": f"{p.stat().st_size // 1024} KB"})
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
else:
    st.info("まだ保存済みのアノテーションはありません。")
