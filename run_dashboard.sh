#!/usr/bin/env bash
# ============================================================
# run_dashboard.sh — Start the Network Intrusion Detection Dashboard
# Phase 8 of the UNSW-NB15 ML Project
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_ROOT/.venv/bin"
BACKEND_DIR="$PROJECT_ROOT/dashboard/backend"
FRONTEND_DIR="$PROJECT_ROOT/dashboard/frontend"
BACKEND_PORT=8000
FRONTEND_PORT=5173

echo "========================================================"
echo "  Network Intrusion Detection Dashboard — Phase 8"
echo "========================================================"

# 1. Check Python virtual environment
if [ ! -f "$VENV/python" ]; then
  echo "ERROR: Python virtual environment not found at $VENV"
  echo "  To setup: python3 -m venv .venv && .venv/bin/pip install -r dashboard/backend/requirements.txt"
  exit 1
fi

# 2. Check Node.js and npm
if ! command -v node &>/dev/null; then
  echo "ERROR: Node.js is required but not installed or not in PATH."
  echo "  Please install Node.js (v18+) from https://nodejs.org"
  exit 1
fi

if ! command -v npm &>/dev/null; then
  echo "ERROR: npm is required but not installed or not in PATH."
  exit 1
fi

# 3. Check / install backend dependencies
if ! "$VENV/python" -c "import fastapi, uvicorn, multipart" &>/dev/null; then
  echo "[backend] Installing required Python dependencies..."
  "$VENV/pip" install -r "$BACKEND_DIR/requirements.txt"
fi

# 4. Check / install frontend dependencies
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "[frontend] Installing npm packages in $FRONTEND_DIR..."
  (cd "$FRONTEND_DIR" && npm install)
fi

# 5. Check if target ports are already bound
check_port() {
  local port=$1
  if lsof -i :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "WARNING: Port $port is already in use by process $(lsof -i :"$port" -sTCP:LISTEN -t | tr '\n' ' ')"
    echo "Please free port $port or terminate the conflicting process before starting the dashboard."
    exit 1
  fi
}

check_port "$BACKEND_PORT"
check_port "$FRONTEND_PORT"

# 6. Cleanup handler: safely terminate only child processes started by this script
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  echo "Shutting down dashboard processes..."
  if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "Stopping FastAPI backend (PID: $BACKEND_PID)..."
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
  if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "Stopping Vite frontend (PID: $FRONTEND_PID)..."
    kill "$FRONTEND_PID" 2>/dev/null || true
    wait "$FRONTEND_PID" 2>/dev/null || true
  fi
  echo "Dashboard servers stopped gracefully."
}
trap cleanup SIGINT SIGTERM EXIT

# 7. Start FastAPI backend
echo ""
echo "[backend] Starting FastAPI server on http://localhost:$BACKEND_PORT ..."
(cd "$BACKEND_DIR" && "$VENV/uvicorn" app.main:app --host 127.0.0.1 --port "$BACKEND_PORT") &
BACKEND_PID=$!

# Wait for backend to report healthy
echo "[backend] Waiting for backend readiness..."
BACKEND_READY=0
for i in $(seq 1 30); do
  if curl -s "http://127.0.0.1:$BACKEND_PORT/api/health" | grep -q '"status":"ok"' 2>/dev/null; then
    echo "[backend] ✓ Backend ready (health check passed)."
    BACKEND_READY=1
    break
  fi
  sleep 1
done

if [ "$BACKEND_READY" -ne 1 ]; then
  echo "ERROR: Backend failed to start within 30 seconds."
  exit 1
fi

# 8. Start Vite frontend
echo ""
echo "[frontend] Starting Vite development server on http://localhost:$FRONTEND_PORT ..."
(cd "$FRONTEND_DIR" && npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT") &
FRONTEND_PID=$!

echo ""
echo "========================================================"
echo "  Dashboard running successfully:"
echo "  → Frontend UI:  http://localhost:$FRONTEND_PORT"
echo "  → Backend API:  http://localhost:$BACKEND_PORT"
echo "  → OpenAPI Docs: http://localhost:$BACKEND_PORT/docs"
echo ""
echo "  Sample traffic files available in: dashboard/sample_data/"
echo "  Press Ctrl+C to cleanly stop all servers."
echo "========================================================"

wait
