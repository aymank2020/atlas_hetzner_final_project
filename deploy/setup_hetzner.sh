#!/usr/bin/env bash
# ========================================================================
# Hetzner CX43 Server Setup Script
# Run as root: bash setup_hetzner.sh
# ========================================================================
set -euo pipefail

APP_USER="${APP_USER:-atlas}"
APP_DIR="${APP_DIR:-/srv/atlas/atlas_hetzner_final_project}"
REPO_URL="${REPO_URL:-https://github.com/aymank2020/atlas_hetzner_final_project.git}"
BRANCH="${BRANCH:-main}"
SWAP_SIZE_GB="${SWAP_SIZE_GB:-2}"

echo "============================================"
echo " Atlas Solver - Hetzner CX43 Setup"
echo "============================================"

# --- Must be root ---
if [[ "${EUID}" -ne 0 ]]; then
  echo "[setup] ERROR: please run as root or via sudo" >&2
  exit 1
fi

# --- Create atlas user ---
echo "[setup] ensuring user '${APP_USER}' exists..."
if ! id "$APP_USER" >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash "$APP_USER"
  echo "[setup] created user '${APP_USER}'"
fi

# --- Swap ---
echo "[setup] configuring ${SWAP_SIZE_GB}G swap..."
if swapon --show | grep -q '/swapfile'; then
  echo "[setup] swapfile already active"
else
  fallocate -l "${SWAP_SIZE_GB}G" /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  echo "[setup] swap activated"
fi

# --- System packages ---
echo "[setup] installing system packages..."
apt-get update -y
apt-get install -y \
  git \
  ffmpeg \
  curl \
  wget \
  ca-certificates \
  python3 \
  python3-venv \
  python3-pip \
  xvfb \
  fonts-liberation \
  libnss3 \
  libxss1 \
  libasound2 \
  libatk-bridge2.0-0 \
  libgtk-3-0 \
  libgbm1 \
  libdrm2 \
  libx11-xcb1 \
  libxcomposite1 \
  libxdamage1 \
  libxrandr2 \
  libpango-1.0-0 \
  libcairo2 \
  libcups2 \
  libdbus-1-3 \
  libexpat1 \
  libfontconfig1 \
  libgcc-s1 \
  libglib2.0-0 \
  libnspr4 \
  libpangocairo-1.0-0 \
  libstdc++6 \
  libxcb1 \
  libxext6 \
  libxfixes3 \
  libxi6 \
  libxrender1 \
  libxtst6 \
  lsb-release \
  xdg-utils \
  unzip

# --- Install Google Chrome ---
echo "[setup] installing Google Chrome..."
if command -v google-chrome >/dev/null 2>&1 || command -v google-chrome-stable >/dev/null 2>&1; then
  echo "[setup] Google Chrome already installed"
else
  local_deb="/tmp/google-chrome-stable_current_amd64.deb"
  curl -fsSL "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb" -o "$local_deb"
  apt-get install -y "$local_deb" || apt-get -f install -y
  rm -f "$local_deb"
  echo "[setup] Google Chrome installed"
fi

# --- Clone repository ---
echo "[setup] setting up repository..."
mkdir -p "$(dirname "$APP_DIR")"
if [[ ! -d "${APP_DIR}/.git" ]]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
  echo "[setup] cloned repo"
else
  git -C "$APP_DIR" fetch origin
  git -C "$APP_DIR" checkout "$BRANCH"
  git -C "$APP_DIR" pull --ff-only origin "$BRANCH"
  echo "[setup] updated repo"
fi

# --- Fix ownership ---
chown -R "${APP_USER}:${APP_USER}" "$(dirname "$APP_DIR")"

# --- Create directories ---
sudo -u "$APP_USER" mkdir -p \
  "$APP_DIR/outputs" \
  "$APP_DIR/logs" \
  "$APP_DIR/.state/accounts" \
  "$APP_DIR/.state/gemini_chat_user_data"

# --- Python virtual environment ---
echo "[setup] setting up Python venv..."
sudo -u "$APP_USER" bash -lc "
  cd '$APP_DIR'
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip wheel setuptools
  pip install -r requirements.txt
  python -m playwright install chromium
"

# Install Playwright system deps
"$APP_DIR/.venv/bin/python" -m playwright install-deps chromium

# --- Verify ---
echo ""
echo "[setup] verifying installation..."
echo -n "  Python: " && "$APP_DIR/.venv/bin/python" --version
echo -n "  ffmpeg: " && ffmpeg -version 2>&1 | head -1
echo -n "  Chrome: " && google-chrome --version 2>/dev/null || google-chrome-stable --version 2>/dev/null || echo "not found"
echo -n "  Xvfb:   " && which Xvfb && echo "OK" || echo "not found"

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo " Next steps:"
echo "   1. Create .env:   nano ${APP_DIR}/.env"
echo "   2. Install service: bash ${APP_DIR}/deploy/install_service.sh"
echo "   3. Start solver:  systemctl start atlas-solver"
echo ""
echo " Or run manually:"
echo "   sudo -u ${APP_USER} bash ${APP_DIR}/deploy/run_solver.sh"
echo ""
