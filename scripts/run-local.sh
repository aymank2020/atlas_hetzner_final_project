#!/usr/bin/env bash
# ========================================================================
# run-local.sh - Single-episode dry-run on local machine (Linux/macOS)
# ========================================================================
# Usage:   ./scripts/run-local.sh
# Requires: Python 3.10+, Playwright browsers installed, .env populated.
# ========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "\033[36m[run-local] Starting dry-run solver from: $PROJECT_ROOT\033[0m"

# Load .env if present
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    echo -e "\033[32m[run-local] Loaded .env\033[0m"
fi

# Unbuffered mode for live log output
python -u atlas_web_auto_solver.py --config configs/config_local_dev.yaml "$@"
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo -e "\033[33m[run-local] Solver exited with code $exit_code\033[0m"
fi
exit $exit_code
