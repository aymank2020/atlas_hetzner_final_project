#!/usr/bin/env bash
# ========================================================================
# Install systemd services for Atlas Solver + Monitor + Update Checker
# Run as root: bash deploy/install_service.sh
# ========================================================================
set -euo pipefail

APP_DIR="${APP_DIR:-/srv/atlas/OCR_Atlas_annotation_final_project}"

echo "========================================"
echo " Atlas Service Installer"
echo "========================================"
echo "APP_DIR: $APP_DIR"
echo ""

# Ensure output directories exist
mkdir -p "$APP_DIR/outputs/hetzner_production"

# --- 1. Install Atlas Solver Service ---
echo "[1/5] Installing atlas-solver.service..."
cp "$APP_DIR/deploy/atlas-solver.service" /etc/systemd/system/atlas-solver.service

# --- 2. Install Atlas Monitor Service ---
echo "[2/5] Installing atlas-monitor.service..."
cp "$APP_DIR/deploy/atlas-monitor.service" /etc/systemd/system/atlas-monitor.service

# --- 3. Install Atlas Update Checker Service ---
echo "[3/5] Installing atlas-updater.service..."
cp "$APP_DIR/deploy/atlas-updater.service" /etc/systemd/system/atlas-updater.service

# --- 4. Reload and enable ---
echo "[4/5] Reloading systemd and enabling services..."
systemctl daemon-reload
systemctl enable atlas-solver.service
systemctl enable atlas-monitor.service
systemctl enable atlas-updater.service

# --- 5. Summary ---
echo "[5/5] Done!"
echo ""
echo "========================================"
echo " Services installed successfully"
echo "========================================"
echo ""
echo "Solver service:"
echo "  Start:   systemctl start atlas-solver"
echo "  Stop:    systemctl stop atlas-solver"
echo "  Status:  systemctl status atlas-solver"
echo "  Logs:    journalctl -u atlas-solver -f"
echo ""
echo "Monitor daemon:"
echo "  Start:   systemctl start atlas-monitor"
echo "  Stop:    systemctl stop atlas-monitor"
echo "  Status:  systemctl status atlas-monitor"
echo "  Logs:    tail -f $APP_DIR/outputs/monitor_daemon.log"
echo ""
echo "Update checker (every 6h):"
echo "  Start:   systemctl start atlas-updater"
echo "  Stop:    systemctl stop atlas-updater"
echo "  Status:  systemctl status atlas-updater"
echo "  Logs:    tail -f $APP_DIR/outputs/update_checker.log"
echo ""
echo "Quick commands:"
echo "  Start all:    systemctl start atlas-solver atlas-monitor atlas-updater"
echo "  Stop all:     systemctl stop atlas-solver atlas-monitor atlas-updater"
echo "  Restart all:  systemctl restart atlas-solver atlas-monitor atlas-updater"
echo ""
echo "NOTE: Stop any manually-running solver before starting the service:"
echo "  pkill -f atlas_web_auto_solver; pkill -f google-chrome; pkill -f Xvfb"
echo "  rm -f /tmp/.X99-lock /tmp/.X11-unix/X99"
