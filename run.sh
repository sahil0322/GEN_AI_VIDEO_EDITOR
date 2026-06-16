#!/usr/bin/env bash
# ==============================================================================
# run.sh — Start FlowEdit backend + frontend together
#
# Usage:
#   ./run.sh            # starts both on default ports
#   ./run.sh --api-only # backend only (no frontend server)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

API_ONLY=false
[[ "${1:-}" == "--api-only" ]] && API_ONLY=true

# Load PORT from .env if present, fallback to 8000
API_PORT=$(grep -E '^PORT=' .env 2>/dev/null | cut -d= -f2 | tr -d ' ' || echo 8000)
FE_PORT=5500

# Cleanup on exit — kill both background processes
cleanup() {
    echo ""
    echo "[run] Shutting down…"
    [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null || true
    [[ -n "${FRONTEND_PID:-}" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  FlowEdit"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Backend ───────────────────────────────────────────────────────────────────
echo "[run] Starting FastAPI backend on port $API_PORT…"
uvicorn main:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --reload \
    --log-level info &
BACKEND_PID=$!
echo "[run] Backend PID: $BACKEND_PID"

# Wait for backend to be ready (polls /health for up to 30s)
echo "[run] Waiting for backend to be ready…"
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${API_PORT}/health" &>/dev/null; then
        echo "[run] Backend ready ✓"
        break
    fi
    sleep 1
    [[ $i -eq 30 ]] && { echo "[run] Backend did not start in 30s. Check logs."; exit 1; }
done

# ── Frontend ──────────────────────────────────────────────────────────────────
if [[ "$API_ONLY" == false ]]; then
    echo "[run] Starting frontend server on port $FE_PORT…"
    cd frontend
    python3 -m http.server "$FE_PORT" &>/dev/null &
    FRONTEND_PID=$!
    cd "$SCRIPT_DIR"
    echo "[run] Frontend PID: $FRONTEND_PID"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
[[ "$API_ONLY" == false ]] && echo "  UI:       http://localhost:${FE_PORT}"
echo "  API:      http://localhost:${API_PORT}"
echo "  Docs:     http://localhost:${API_PORT}/docs"
echo "  Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Block until Ctrl+C
wait
