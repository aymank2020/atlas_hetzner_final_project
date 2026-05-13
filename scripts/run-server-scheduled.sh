#!/usr/bin/env bash
# ========================================================================
# run-server-scheduled.sh - Continuous loop: run solver, pause, repeat.
# ========================================================================
# Usage:   ./scripts/run-server-scheduled.sh
# Ctrl+C to stop gracefully between cycles.
# Options:
#   --pause SEC       Pause between runs (default: 120)
#   --max-runs N      Max runs, 0=unlimited (default: 0)
#   --config PATH     Config file (default: configs/config_hetzner_production.yaml)
# ========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults
PAUSE_BETWEEN_RUNS=120
MAX_RUNS=0
CONFIG="configs/config_hetzner_production.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pause)
            PAUSE_BETWEEN_RUNS="$2"
            shift 2
            ;;
        --max-runs)
            MAX_RUNS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Load .env if present
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

mkdir -p logs

run_count=0
consecutive_failures=0
max_consecutive_failures=5

echo -e "\033[36m[scheduler] Starting continuous solver loop\033[0m"
echo -e "\033[36m[scheduler] Config: $CONFIG | Pause: ${PAUSE_BETWEEN_RUNS}s | MaxRuns: $([ $MAX_RUNS -eq 0 ] && echo 'unlimited' || echo $MAX_RUNS)\033[0m"

while true; do
    ((run_count++)) || true
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    LOG_FILE="logs/solver_${TIMESTAMP}.log"

    echo -e "\n\033[36m[scheduler] === Run #$run_count at $TIMESTAMP ===\033[0m"

    python -u atlas_web_auto_solver.py --config "$CONFIG" 2>&1 | tee "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        consecutive_failures=0
        echo -e "\033[32m[scheduler] Run #$run_count completed successfully\033[0m"
    else
        ((consecutive_failures++)) || true
        echo -e "\033[33m[scheduler] Run #$run_count failed (exit $exit_code, consecutive failures: $consecutive_failures)\033[0m"
        if [ $consecutive_failures -ge $max_consecutive_failures ]; then
            echo -e "\033[31m[scheduler] Too many consecutive failures ($max_consecutive_failures). Stopping.\033[0m"
            exit 1
        fi
    fi

    if [ $MAX_RUNS -gt 0 ] && [ $run_count -ge $MAX_RUNS ]; then
        echo -e "\033[32m[scheduler] Reached max runs ($MAX_RUNS). Stopping.\033[0m"
        exit 0
    fi

    # Backoff on failure: 1.5x, max 600s
    if [ $consecutive_failures -gt 0 ]; then
        pause=$(echo "$PAUSE_BETWEEN_RUNS * (1.5 ^ $consecutive_failures)" | bc -l | cut -c1-8)
        pause=$(echo "$pause" | awk '{if ($1 > 600) print 600; else print $1}')
    else
        pause=$PAUSE_BETWEEN_RUNS
    fi

    echo -e "\033[90m[scheduler] Pausing ${pause}s before next run...\033[0m"
    sleep "$pause"
done
