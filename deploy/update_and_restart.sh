#!/usr/bin/env bash
# ========================================================================
# Pull latest code and restart the solver service
# Run as root: bash update_and_restart.sh
# ========================================================================
set -euo pipefail

APP_DIR="${APP_DIR:-/srv/atlas/OCR_Atlas_annotation_final_project}"
APP_USER="${APP_USER:-atlas}"

echo "[update] pulling latest code..."
sudo -u "$APP_USER" git -C "$APP_DIR" pull --ff-only origin main

echo "[update] updating pip dependencies..."
sudo -u "$APP_USER" bash -lc "
  cd '$APP_DIR'
  source .venv/bin/activate
  pip install -r requirements.txt --quiet
"

echo "[update] restarting atlas-solver..."
systemctl restart atlas-solver.service
sleep 2
systemctl status atlas-solver.service --no-pager -l

echo "[update] done"
