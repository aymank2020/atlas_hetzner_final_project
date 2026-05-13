#!/usr/bin/env bash
# ========================================================================
# Run Atlas Solver on Hetzner (headless with Xvfb + Chrome CDP + VNC)
# Usage: sudo -u atlas bash /srv/atlas/OCR_Atlas_annotation_final_project/deploy/run_solver.sh
# ========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
cd "$APP_DIR"

CONFIG="${CONFIG:-configs/config_hetzner_production.yaml}"
LOG_FILE="${LOG_FILE:-$APP_DIR/outputs/solver_service.log}"
PYTHON_BIN="${PYTHON_BIN:-$APP_DIR/.venv/bin/python}"
ENABLE_XVFB="${ENABLE_XVFB:-1}"
XVFB_DISPLAY="${XVFB_DISPLAY:-:99}"
XVFB_ARGS="${XVFB_ARGS:--screen 0 1920x1080x24 -ac +extension RANDR}"
CHROME_CDP_PORT="${CHROME_CDP_PORT:-9222}"
ENABLE_VNC="${ENABLE_VNC:-1}"
VNC_PORT="${VNC_PORT:-5900}"
VNC_PASSWORD="${VNC_PASSWORD:-}"
if [[ "$ENABLE_VNC" == "1" && -z "$VNC_PASSWORD" ]]; then
  echo "[solver] ERROR: VNC_PASSWORD must be set when ENABLE_VNC=1" >&2
  exit 1
fi

# Suppress Node deprecation warnings from Playwright
export NODE_OPTIONS="--no-warnings"

mkdir -p "$(dirname "$LOG_FILE")" "$APP_DIR/.state/gemini_chat_user_data"

# --- Load .env ---
if [[ -f "$APP_DIR/.env" ]]; then
  set -a
  source <(
    python3 - <<'PY'
from pathlib import Path
for line in Path(".env").read_text(encoding="utf-8-sig").splitlines():
    stripped = line.strip()
    if stripped and not stripped.startswith("#"):
        print(stripped)
PY
  )
  set +a
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[solver] ERROR: missing python: $PYTHON_BIN" >&2
  exit 1
fi

# --- Start Xvfb if enabled ---
XVFB_PID=""
if [[ "$ENABLE_XVFB" == "1" ]]; then
  echo "[solver] starting Xvfb on ${XVFB_DISPLAY}..."
  Xvfb "$XVFB_DISPLAY" $XVFB_ARGS &
  XVFB_PID=$!
  export DISPLAY="$XVFB_DISPLAY"
  sleep 1
  echo "[solver] Xvfb started (PID=$XVFB_PID, DISPLAY=$DISPLAY)"
fi

# --- Start VNC if enabled ---
VNC_PID=""
if [[ "$ENABLE_VNC" == "1" ]] && command -v x11vnc >/dev/null 2>&1; then
  echo "[solver] starting VNC on port ${VNC_PORT}..."
  x11vnc -storepasswd "$VNC_PASSWORD" "$APP_DIR/.state/vnc_passwd" 2>/dev/null
  chmod 600 "$APP_DIR/.state/vnc_passwd"
  x11vnc -display "$XVFB_DISPLAY" -rfbauth "$APP_DIR/.state/vnc_passwd" -rfbport "$VNC_PORT" \
    -shared -forever -noxdamage -bg -o /tmp/x11vnc.log 2>/dev/null
  VNC_PID=$(pgrep -f "x11vnc.*$VNC_PORT" | head -1 || true)
  echo "[solver] VNC started (PID=${VNC_PID:-unknown}, port=$VNC_PORT)"
fi

# --- Start Chrome with CDP ---
CHROME_PID=""
echo "[solver] starting Chrome with CDP on port ${CHROME_CDP_PORT}..."
google-chrome-stable \
  --remote-debugging-port="$CHROME_CDP_PORT" \
  --no-first-run \
  --no-default-browser-check \
  --no-sandbox \
  --disable-dev-shm-usage \
  --disable-gpu \
  --disable-software-rasterizer \
  --disable-background-timer-throttling \
  --disable-backgrounding-occluded-windows \
  --disable-renderer-backgrounding \
  --user-data-dir="$APP_DIR/.state/gemini_chat_user_data" \
  --window-size=1920,1080 \
  "about:blank" &
CHROME_PID=$!
sleep 3
echo "[solver] Chrome started (PID=$CHROME_PID)"

# --- Cleanup handler ---
cleanup() {
  echo "[solver] cleaning up..."
  [[ -n "${CHROME_PID:-}" ]] && kill "$CHROME_PID" 2>/dev/null || true
  [[ -n "${VNC_PID:-}" ]] && kill "$VNC_PID" 2>/dev/null || true
  [[ -n "${XVFB_PID:-}" ]] && kill "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- Run solver ---
echo "[solver] starting solver at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "[solver] config: $CONFIG"
echo "[solver] log: $LOG_FILE"

# Run solver as a regular command (not exec) so the cleanup trap fires on exit.
# Direct append to LOG_FILE instead of tee; systemd captures stdout to journal.
"$PYTHON_BIN" -u atlas_web_auto_solver.py --config "$CONFIG" --execute >> "$LOG_FILE" 2>&1
EXIT_CODE=$?
echo "[solver] solver exited with code $EXIT_CODE at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_FILE"
exit $EXIT_CODE
