#!/usr/bin/env bash
# ========================================================================
# setup-local.sh - One-time setup for a local development machine.
# ========================================================================
# Usage:   ./scripts/setup-local.sh
# Requires: Python 3.10+, internet connection.
# ========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "\033[36m=== Atlas Solver Local Setup ===\033[0m"
echo -e "\033[36mProject root: $PROJECT_ROOT\033[0m"

# -------------------------------------------------------------------
# 1. Check Python
# -------------------------------------------------------------------
echo -e "\n\033[33m[1/6] Checking Python...\033[0m"
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    echo -e "  \033[32mFound: $PY_VERSION\033[0m"
else
    echo -e "  \033[31mERROR: Python3 not found. Install Python 3.10+ from https://python.org\033[0m"
    exit 1
fi

# -------------------------------------------------------------------
# 2. Create virtual environment (recommended)
# -------------------------------------------------------------------
echo -e "\n\033[33m[2/6] Setting up virtual environment...\033[0m"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "  \033[32mCreated .venv\033[0m"
else
    echo -e "  \033[90m.venv already exists\033[0m"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# -------------------------------------------------------------------
# 3. Install Python packages
# -------------------------------------------------------------------
echo -e "\n\033[33m[3/6] Installing Python packages...\033[0m"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo -e "  \033[32mPython packages installed.\033[0m"

# -------------------------------------------------------------------
# 4. Install Playwright browsers
# -------------------------------------------------------------------
echo -e "\n\033[33m[4/6] Installing Playwright browsers (Chromium)...\033[0m"
python -m playwright install chromium
echo -e "  \033[32mPlaywright Chromium installed.\033[0m"

# -------------------------------------------------------------------
# 5. Check ffmpeg
# -------------------------------------------------------------------
echo -e "\n\033[33m[5/6] Checking ffmpeg...\033[0m"
if command -v ffmpeg &> /dev/null; then
    FF_VERSION=$(ffmpeg -version 2>&1 | head -1)
    echo -e "  \033[32mFound: $FF_VERSION\033[0m"
else
    echo -e "  \033[33mWARNING: ffmpeg not found on PATH.\033[0m"
    echo -e "  \033[33mVideo optimization will fall back to OpenCV (lower quality).\033[0m"
    if [[ "$(uname)" == "Darwin" ]]; then
        echo -e "  \033[33mInstall via: brew install ffmpeg\033[0m"
    elif [[ "$(uname)" == "Linux" ]]; then
        echo -e "  \033[33mInstall via: sudo apt install ffmpeg  OR  sudo yum install ffmpeg\033[0m"
    fi
fi

# -------------------------------------------------------------------
# 6. Create runtime directories + .env
# -------------------------------------------------------------------
echo -e "\n\033[33m[6/6] Creating runtime directories...\033[0m"
for dir in .state outputs logs; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "  \033[32mCreated: $dir\033[0m"
    else
        echo -e "  \033[90mExists:  $dir\033[0m"
    fi
done

if [ -f .env ]; then
    echo -e "  \033[32m.env found.\033[0m"
elif [ -f .env.example ]; then
    cp .env.example .env
    echo -e "  \033[33mCreated .env from .env.example. Edit it with your credentials.\033[0m"
else
    echo -e "  \033[33mWARNING: No .env or .env.example found. Create .env with credentials.\033[0m"
fi

echo -e "\n\033[36m=== Setup Complete ===\033[0m"
echo -e "\033[36mNext steps:\033[0m"
echo -e "  1. Edit .env with your Atlas/Gemini/Gmail credentials"
echo -e "  2. Test dry-run:  ./scripts/run-local.sh"
echo -e "  3. Production:    ./scripts/run-server-once.sh"
echo -e "  4. Continuous:    ./scripts/run-server-scheduled.sh"
