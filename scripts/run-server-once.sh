#!/usr/bin/env bash
# ========================================================================
# run-server-once.sh - Single production run (headless server)
# ========================================================================
# Usage:   ./scripts/run-server-once.sh
# Optional: ./scripts/run-server-once.sh --config configs/config_hetzner_production.yaml
# ========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo -e "\033[36m[run-server] Starting production solver at $TIMESTAMP\033[0m"

# Load .env if present
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    echo -e "\033[32m[run-server] Loaded .env\033[0m"
fi

# Default to production config unless overridden
CONFIG_PATH="configs/config_hetzner_production.yaml"
if [[ " $* " != *" --config "* ]]; then
    set -- --config "$CONFIG_PATH" "$@"
fi

# Ensure logs directory exists
mkdir -p logs

LOG_FILE="logs/solver_${TIMESTAMP}.log"
echo -e "\033[32m[run-server] Logging to $LOG_FILE\033[0m"

# Run with unbuffered output, tee to log file
python -u atlas_web_auto_solver.py "$@" 2>&1 | tee "$LOG_FILE"
exit_code=${PIPESTATUS[0]}

if [ $exit_code -eq 0 ]; then
    echo -e "\033[32m[run-server] Solver exited successfully at $(date +'%H:%M:%S')\033[0m"
else
    echo -e "\033[33m[run-server] Solver exited with code $exit_code at $(date +'%H:%M:%S')\033[0m"
fi
exit $exit_code
