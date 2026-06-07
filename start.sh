#!/usr/bin/env bash
# ── CosMx 2026 — start all services ──────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"
LOG_DIR="$SCRIPT_DIR/.logs"
mkdir -p "$PID_DIR" "$LOG_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

info()    { echo -e "${CYAN}[CosMx]${NC} $*"; }
success() { echo -e "${GREEN}[  OK  ]${NC} $*"; }
warn()    { echo -e "${YELLOW}[ WARN ]${NC} $*"; }
error()   { echo -e "${RED}[ERROR ]${NC} $*"; }

cd "$SCRIPT_DIR"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║    CosMx 2026  —  Starting all services      ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. Streamlit QC Workbench (port 8501) ────────────────────────────────────
STREAMLIT_PID_FILE="$PID_DIR/streamlit.pid"
STREAMLIT_LOG="$LOG_DIR/streamlit.log"
STREAMLIT_PORT=8501
VENV_STREAMLIT="cell_segmentation_pipeline/.venv/bin/streamlit"

if [[ -f "$STREAMLIT_PID_FILE" ]] && kill -0 "$(cat "$STREAMLIT_PID_FILE")" 2>/dev/null; then
    warn "Streamlit QC Workbench already running (PID=$(cat "$STREAMLIT_PID_FILE"))"
elif [[ -f "$VENV_STREAMLIT" ]]; then
    info "Starting Streamlit QC Workbench on port $STREAMLIT_PORT ..."
    "$VENV_STREAMLIT" run cell_segmentation_pipeline/app/streamlit_app.py \
        --server.port "$STREAMLIT_PORT" \
        --server.address 0.0.0.0 \
        --server.headless true \
        > "$STREAMLIT_LOG" 2>&1 &
    echo $! > "$STREAMLIT_PID_FILE"
    sleep 2
    if kill -0 "$(cat "$STREAMLIT_PID_FILE")" 2>/dev/null; then
        success "Streamlit QC Workbench  →  http://localhost:$STREAMLIT_PORT"
    else
        error "Streamlit failed to start. Check $STREAMLIT_LOG"
    fi
else
    warn "Streamlit venv not found at $VENV_STREAMLIT — skipping"
fi

# ── 2. Annotation Web Server (port 8000, FastAPI) ────────────────────────────
ANNOT_PID_FILE="$PID_DIR/annotation.pid"
ANNOT_LOG="$LOG_DIR/annotation.log"
ANNOT_PORT=8000
CONDA_PYTHON="/home/fujiyamaakihisa/miniconda3/envs/cosmx_annotation/bin/python"

if [[ -f "$ANNOT_PID_FILE" ]] && kill -0 "$(cat "$ANNOT_PID_FILE")" 2>/dev/null; then
    warn "Annotation server already running (PID=$(cat "$ANNOT_PID_FILE"))"
elif [[ -f "$CONDA_PYTHON" ]]; then
    info "Starting Annotation Web Server on port $ANNOT_PORT ..."
    "$CONDA_PYTHON" annotation_tools/web_annotator/server.py \
        > "$ANNOT_LOG" 2>&1 &
    echo $! > "$ANNOT_PID_FILE"
    sleep 2
    if kill -0 "$(cat "$ANNOT_PID_FILE")" 2>/dev/null; then
        success "Annotation Server          →  http://localhost:$ANNOT_PORT"
    else
        error "Annotation server failed to start. Check $ANNOT_LOG"
    fi
else
    warn "conda env 'cosmx_annotation' not found — skipping annotation server"
fi

# ── 3. Docker Compose services ────────────────────────────────────────────────
if docker info > /dev/null 2>&1; then
    info "Docker is running. Starting Compose services ..."
    if docker compose up -d 2>&1 | tee "$LOG_DIR/docker.log" | grep -qE "Started|Running|healthy|done"; then
        success "Docker Compose services started"
    else
        docker compose up -d > "$LOG_DIR/docker.log" 2>&1 && success "Docker Compose services started" \
            || warn "Docker Compose may have had issues. Check $LOG_DIR/docker.log"
    fi
    docker compose ps
else
    warn "Docker daemon not running — skipping Docker Compose"
    warn "To start Docker:  sudo systemctl start docker"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              Service URLs                    ║${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║${NC}  QC Workbench    http://localhost:8501       ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  Annotation      http://localhost:8000       ${CYAN}║${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║${NC}  Logs: .logs/   PIDs: .pids/                 ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  Stop: ./stop.sh                             ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""
