#!/usr/bin/env bash
# ── CosMx 2026 — stop all services ───────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

info()    { echo -e "${CYAN}[CosMx]${NC} $*"; }
success() { echo -e "${GREEN}[  OK  ]${NC} $*"; }
warn()    { echo -e "${YELLOW}[ WARN ]${NC} $*"; }

cd "$SCRIPT_DIR"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║    CosMx 2026  —  Stopping all services      ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── Helper: stop a process by PID file ───────────────────────────────────────
stop_pid() {
    local name="$1"
    local pid_file="$PID_DIR/$2.pid"

    if [[ ! -f "$pid_file" ]]; then
        warn "$name: no PID file found — already stopped?"
        return
    fi

    local pid
    pid=$(cat "$pid_file")

    if kill -0 "$pid" 2>/dev/null; then
        info "Stopping $name (PID=$pid) ..."
        kill "$pid" 2>/dev/null || true
        # Wait up to 5s for graceful exit, then force
        local i=0
        while kill -0 "$pid" 2>/dev/null && (( i < 5 )); do
            sleep 1; (( i++ ))
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
        success "$name stopped"
    else
        warn "$name was not running (stale PID=$pid)"
    fi

    rm -f "$pid_file"
}

# ── 1. Streamlit QC Workbench ─────────────────────────────────────────────────
stop_pid "Streamlit QC Workbench" "streamlit"

# ── 2. Annotation Web Server ─────────────────────────────────────────────────
stop_pid "Annotation Server" "annotation"

# ── 3. Docker Compose services ────────────────────────────────────────────────
if docker info > /dev/null 2>&1; then
    info "Stopping Docker Compose services ..."
    docker compose down && success "Docker Compose stopped" \
        || warn "docker compose down had issues (services may already be stopped)"
else
    warn "Docker daemon not running — skipping Docker Compose"
fi

echo ""
success "All services stopped."
echo ""
