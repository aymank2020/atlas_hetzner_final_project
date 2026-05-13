#!/usr/bin/env python3
"""Atlas Solver Monitor Daemon

Continuously watches the solver process and logs.
Detects failures, auto-restarts, sends Telegram alerts,
cleans up old output files, and sends periodic status reports.

Usage:
    python3 deploy/monitor_daemon.py

Environment variables (loaded from .env):
    TELEGRAM_BOT_TOKEN      - Telegram bot token from @BotFather
    TELEGRAM_ALLOWED_CHATS  - Comma-separated chat IDs
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_DIR = Path(os.environ.get(
    "ATLAS_APP_DIR",
    "/srv/atlas/OCR_Atlas_annotation_final_project",
))
OUTPUT_DIR = APP_DIR / "outputs" / "hetzner_production"
LOG_FILE = APP_DIR / "outputs" / "solver_service.log"
WRAPPER_LOG = APP_DIR / "outputs" / "solver_service_wrapper.log"
MONITOR_STATE_FILE = APP_DIR / "outputs" / "monitor_state.json"
MONITOR_LOG_FILE = APP_DIR / "outputs" / "monitor_daemon.log"

SOLVER_SERVICE = "atlas-solver.service"
SOLVER_SCRIPT = APP_DIR / "deploy" / "run_solver.sh"

CHECK_INTERVAL_SEC = 60
STALE_LOG_THRESHOLD_SEC = 1800          # 30 min no new log output
STALE_LOG_RESTART_COOLDOWN_SEC = 300    # min 5 min between stale restarts
STATUS_REPORT_INTERVAL_SEC = 21600      # 6 hours
DISK_CLEANUP_INTERVAL_SEC = 3600        # 1 hour
DISK_LOW_THRESHOLD_PERCENT = 85         # alert when disk > 85% used
OUTPUT_MAX_AGE_HOURS = 48               # delete output files older than this
LOG_TAIL_CHARS = 16000                  # how much of the log to scan

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_IDS: list[int] = []

# ---------------------------------------------------------------------------
# Issue patterns
# ---------------------------------------------------------------------------

ISSUE_PATTERNS: list[tuple[str, str]] = [
    ("crash_traceback",       "Traceback (most recent call last):"),
    ("policy_gate_blocked",   "[run] policy gate blocked apply for this episode."),
    ("playwright_epipe",      "Error: write EPIPE"),
    ("target_closed",         "TargetClosedError"),
    ("quota_exceeded",        "GEMINI_QUOTA_EXCEEDED"),
    ("rooms_unavailable",     "Rooms are unavailable"),
    ("browser_disconnected",  "Browser has been disconnected"),
    ("cdp_connection_failed", "CDP connection"),
    ("session_expired",       "requires authenticated session"),
    ("no_episodes",           "No episodes available"),
    ("xvfb_crash",            "Xvfb process died"),
]

# These warrant an automatic restart
RESTART_WORTHY = {
    "crash_traceback",
    "playwright_epipe",
    "target_closed",
    "browser_disconnected",
    "xvfb_crash",
}

# These are informational (send alert but don't restart)
INFO_ONLY = {
    "policy_gate_blocked",
    "quota_exceeded",
    "rooms_unavailable",
    "no_episodes",
}

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_running = True


def _signal_handler(sig, frame):
    global _running
    _running = False


if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        MONITOR_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MONITOR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


def _send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        return False
    payload = json.dumps({
        "text": text[:4000],
        "disable_web_page_preview": True,
        "parse_mode": "HTML",
    }).encode("utf-8")
    sent = 0
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            body = json.dumps({
                "chat_id": chat_id,
                "text": text[:4000],
                "disable_web_page_preview": True,
            }).encode("utf-8")
            req = Request(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=20) as resp:
                if resp.status == 200:
                    sent += 1
        except Exception as e:
            _log(f"[telegram] send failed to {chat_id}: {e}")
    return sent > 0


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def _load_state() -> dict:
    try:
        if MONITOR_STATE_FILE.exists():
            return json.loads(MONITOR_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_state(state: dict) -> None:
    try:
        MONITOR_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        MONITOR_STATE_FILE.write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )
    except Exception as e:
        _log(f"[state] save failed: {e}")


# ---------------------------------------------------------------------------
# Solver process checks
# ---------------------------------------------------------------------------


def _is_solver_running() -> bool:
    """Check if the solver python process is alive."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "atlas_web_auto_solver"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _is_service_active() -> bool:
    """Check if the systemd service is active."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", SOLVER_SERVICE],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() == "active"
    except Exception:
        return False


def _get_solver_pid() -> int | None:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "atlas_web_auto_solver"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split()
            return int(pids[0]) if pids else None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Log analysis
# ---------------------------------------------------------------------------


def _read_log_tail(max_chars: int = LOG_TAIL_CHARS) -> str:
    try:
        if not LOG_FILE.exists():
            return ""
        size = LOG_FILE.stat().st_size
        if size == 0:
            return ""
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            if size > max_chars:
                f.seek(size - max_chars)
                f.readline()  # skip partial line
            return f.read()
    except Exception:
        return ""


def _log_mtime() -> float:
    try:
        if LOG_FILE.exists():
            return LOG_FILE.stat().st_mtime
    except Exception:
        pass
    return 0.0


def _detect_issues(text: str) -> list[str]:
    issues = []
    for code, marker in ISSUE_PATTERNS:
        if marker in text:
            if code == "crash_traceback":
                if not _has_real_traceback(text):
                    continue
            issues.append(code)
    return issues


def _has_real_traceback(text: str) -> bool:
    marker = "Traceback (most recent call last):"
    for chunk in text.split(marker)[1:]:
        snippet = chunk[:2000]
        if "KeyboardInterrupt" in snippet:
            continue
        return True
    return False


def _count_episodes(text: str) -> tuple[int, int]:
    """Count submitted and failed episodes in log tail."""
    submitted = len(re.findall(r"submit VERIFIED", text))
    failed = len(re.findall(r"failure_class=(?!none)", text))
    return submitted, failed


def _extract_last_episode_id(text: str) -> str:
    matches = re.findall(r"episode[_=]([0-9a-f]{24})", text)
    return matches[-1] if matches else ""


# ---------------------------------------------------------------------------
# Auto-recovery
# ---------------------------------------------------------------------------


def _kill_stale_processes() -> None:
    """Kill orphaned Chrome/Xvfb processes before restart."""
    for pattern in [
        "atlas_web_auto_solver",
        "google-chrome",
        "x11vnc",
        "Xvfb",
        "websockify",
    ]:
        try:
            subprocess.run(
                ["pkill", "-f", pattern],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
    time.sleep(2)
    # Clean lock files
    for f in ["/tmp/.X99-lock", "/tmp/.X11-unix/X99"]:
        try:
            Path(f).unlink(missing_ok=True)
        except Exception:
            pass


def _restart_solver(reason: str) -> bool:
    """Restart the solver process."""
    _log(f"[restart] attempting restart: {reason}")

    # Always kill stale processes first to avoid orphan accumulation
    _kill_stale_processes()

    # Try systemd first (timeout=120 because TimeoutStopSec=90 in the unit)
    if _is_service_active() or _service_exists():
        try:
            subprocess.run(
                ["systemctl", "restart", SOLVER_SERVICE],
                capture_output=True, timeout=120,
            )
            time.sleep(10)
            if _is_solver_running():
                _log("[restart] systemd restart succeeded")
                return True
            else:
                _log("[restart] systemd restart returned but solver not running yet, waiting...")
                time.sleep(15)
                if _is_solver_running():
                    _log("[restart] systemd restart succeeded (delayed)")
                    return True
        except Exception as e:
            _log(f"[restart] systemd restart failed: {e}")

    # Fallback: kill everything and re-launch via script
    _log("[restart] falling back to manual restart")
    _kill_stale_processes()

    try:
        subprocess.Popen(
            ["nohup", "bash", str(SOLVER_SCRIPT)],
            stdout=open(WRAPPER_LOG, "a"),
            stderr=subprocess.STDOUT,
            cwd=str(APP_DIR),
            start_new_session=True,
        )
        time.sleep(15)
        alive = _is_solver_running()
        if alive:
            _log("[restart] manual restart succeeded")
        else:
            _log("[restart] manual restart FAILED - solver not running")
        return alive
    except Exception as e:
        _log(f"[restart] manual restart error: {e}")
        return False


def _service_exists() -> bool:
    try:
        result = subprocess.run(
            ["systemctl", "cat", SOLVER_SERVICE],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Disk management
# ---------------------------------------------------------------------------


def _disk_usage_percent() -> float:
    try:
        usage = shutil.disk_usage("/")
        return (usage.used / usage.total) * 100.0
    except Exception:
        return 0.0


def _cleanup_old_outputs() -> int:
    """Remove output files older than OUTPUT_MAX_AGE_HOURS. Returns count."""
    if not OUTPUT_DIR.exists():
        return 0
    cutoff = time.time() - (OUTPUT_MAX_AGE_HOURS * 3600)
    removed = 0
    patterns = ["video_*.mp4", "*_chatchunk_*.mp4", "*_upload_opt.mp4"]
    for pattern in patterns:
        for f in OUTPUT_DIR.glob(pattern):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except Exception:
                pass
    # Clean chat_only subdirectories
    chat_dir = OUTPUT_DIR / "_chat_only"
    if chat_dir.exists():
        for d in chat_dir.iterdir():
            try:
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
            except Exception:
                pass
    # Clean step screenshots older than cutoff
    ss_dir = OUTPUT_DIR / "step_screenshots"
    if ss_dir.exists():
        for d in ss_dir.iterdir():
            try:
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
            except Exception:
                pass
    return removed


# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------


def _build_status_report(state: dict) -> str:
    uptime_sec = time.time() - state.get("monitor_start_epoch", time.time())
    uptime_h = uptime_sec / 3600.0
    total_submitted = state.get("total_submitted", 0)
    total_failed = state.get("total_failed", 0)
    total_restarts = state.get("total_restarts", 0)
    solver_alive = _is_solver_running()
    disk_pct = _disk_usage_percent()
    last_episode = state.get("last_episode_id", "N/A")

    rate = total_submitted / max(uptime_h, 0.1)

    lines = [
        "--- Atlas Solver Status Report ---",
        f"Solver: {'RUNNING' if solver_alive else 'DOWN'}",
        f"Monitor uptime: {uptime_h:.1f}h",
        f"Episodes submitted: {total_submitted}",
        f"Episodes failed: {total_failed}",
        f"Submit rate: {rate:.1f}/hour",
        f"Auto-restarts: {total_restarts}",
        f"Disk usage: {disk_pct:.0f}%",
        f"Last episode: {last_episode}",
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _load_env():
    """Load .env file for Telegram credentials."""
    global TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS
    env_file = APP_DIR / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key, val)
        except Exception:
            pass
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    raw_chats = os.environ.get("TELEGRAM_ALLOWED_CHATS", "").strip()
    if raw_chats:
        TELEGRAM_CHAT_IDS = [
            int(c.strip()) for c in raw_chats.split(",")
            if c.strip().isdigit()
        ]


def main() -> None:
    _load_env()
    _log("[monitor] Atlas Monitor Daemon starting")
    _log(f"[monitor] app_dir={APP_DIR}")
    _log(f"[monitor] log_file={LOG_FILE}")
    _log(f"[monitor] telegram={'configured' if TELEGRAM_BOT_TOKEN else 'NOT configured'}")
    _log(f"[monitor] chat_ids={TELEGRAM_CHAT_IDS}")

    state = _load_state()
    state.setdefault("monitor_start_epoch", time.time())
    state.setdefault("total_submitted", 0)
    state.setdefault("total_failed", 0)
    state.setdefault("total_restarts", 0)
    state.setdefault("last_alert_signature", "")
    state.setdefault("last_status_report_epoch", time.time())
    state.setdefault("last_cleanup_epoch", 0)
    state.setdefault("last_restart_epoch", 0)
    state.setdefault("last_log_mtime", 0.0)
    state.setdefault("last_episode_id", "")
    state.setdefault("consecutive_stale_checks", 0)

    # Send startup notification
    _send_telegram(
        "Atlas Monitor Daemon started\n"
        f"Server: Hetzner CX43\n"
        f"Solver: {'RUNNING' if _is_solver_running() else 'NOT RUNNING'}\n"
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    while _running:
        try:
            _check_cycle(state)
        except Exception as e:
            _log(f"[monitor] check cycle error: {e}")
        _save_state(state)
        # Sleep in small increments to respond to signals
        for _ in range(CHECK_INTERVAL_SEC):
            if not _running:
                break
            time.sleep(1)

    _log("[monitor] shutting down")
    _send_telegram("Atlas Monitor Daemon stopped")


def _check_cycle(state: dict) -> None:
    now = time.time()

    # --- 1. Check solver process ---
    solver_alive = _is_solver_running()

    if not solver_alive:
        _log("[check] solver is NOT running!")
        # Cooldown: don't restart too frequently
        last_restart = state.get("last_restart_epoch", 0)
        if now - last_restart > STALE_LOG_RESTART_COOLDOWN_SEC:
            success = _restart_solver("solver process not running")
            state["last_restart_epoch"] = now
            state["total_restarts"] = state.get("total_restarts", 0) + 1
            _send_telegram(
                f"ALERT: Solver was not running!\n"
                f"Auto-restart: {'SUCCESS' if success else 'FAILED'}\n"
                f"Total restarts: {state['total_restarts']}"
            )
            if not success:
                _send_telegram(
                    "CRITICAL: Solver restart FAILED!\n"
                    "Manual intervention required.\n"
                    f"SSH: ssh root@{os.environ.get('SERVER_IPV4', 'SERVER_IPV4_NOT_SET')}"
                )
        return

    # --- 2. Check log freshness ---
    log_mtime = _log_mtime()
    log_age = now - log_mtime if log_mtime > 0 else 0

    if log_mtime > 0 and log_age > STALE_LOG_THRESHOLD_SEC:
        state["consecutive_stale_checks"] = state.get("consecutive_stale_checks", 0) + 1
        _log(f"[check] log stale: age={log_age:.0f}s consecutive={state['consecutive_stale_checks']}")

        # Only restart after 2 consecutive stale checks (avoid false positives)
        if state["consecutive_stale_checks"] >= 2:
            last_restart = state.get("last_restart_epoch", 0)
            if now - last_restart > STALE_LOG_RESTART_COOLDOWN_SEC:
                _send_telegram(
                    f"ALERT: Solver log stale for {log_age:.0f}s\n"
                    "Restarting solver..."
                )
                success = _restart_solver(f"stale log ({log_age:.0f}s)")
                state["last_restart_epoch"] = now
                state["total_restarts"] = state.get("total_restarts", 0) + 1
                state["consecutive_stale_checks"] = 0
                _send_telegram(
                    f"Solver restart: {'SUCCESS' if success else 'FAILED'}"
                )
    else:
        state["consecutive_stale_checks"] = 0

    # --- 3. Analyze log for issues ---
    log_text = _read_log_tail()
    if log_text:
        issues = _detect_issues(log_text)
        submitted, failed = _count_episodes(log_text)
        last_ep = _extract_last_episode_id(log_text)

        # Update counters (only add new ones since last check)
        prev_submitted = state.get("_prev_cycle_submitted", 0)
        prev_failed = state.get("_prev_cycle_failed", 0)
        if submitted > prev_submitted:
            state["total_submitted"] = state.get("total_submitted", 0) + (submitted - prev_submitted)
        if failed > prev_failed:
            state["total_failed"] = state.get("total_failed", 0) + (failed - prev_failed)
        state["_prev_cycle_submitted"] = submitted
        state["_prev_cycle_failed"] = failed

        if last_ep:
            state["last_episode_id"] = last_ep

        # Check for new issues
        if issues:
            sig = hashlib.md5(
                json.dumps(sorted(issues)).encode()
            ).hexdigest()[:12]
            if sig != state.get("last_alert_signature", ""):
                state["last_alert_signature"] = sig

                # Categorize issues
                restart_issues = [i for i in issues if i in RESTART_WORTHY]
                info_issues = [i for i in issues if i in INFO_ONLY]
                other_issues = [i for i in issues if i not in RESTART_WORTHY and i not in INFO_ONLY]

                alert_lines = ["Atlas Solver Issues Detected:"]
                if restart_issues:
                    alert_lines.append(f"CRITICAL: {', '.join(restart_issues)}")
                if info_issues:
                    alert_lines.append(f"Info: {', '.join(info_issues)}")
                if other_issues:
                    alert_lines.append(f"Warning: {', '.join(other_issues)}")
                alert_lines.append(f"Last episode: {last_ep or 'N/A'}")

                _send_telegram("\n".join(alert_lines))

                # Auto-restart for critical issues (solver handles most internally)
                # Only restart if solver actually died - the solver's own retry
                # logic handles most transient errors

    # --- 4. Disk cleanup ---
    if now - state.get("last_cleanup_epoch", 0) > DISK_CLEANUP_INTERVAL_SEC:
        disk_pct = _disk_usage_percent()
        if disk_pct > DISK_LOW_THRESHOLD_PERCENT:
            _log(f"[cleanup] disk at {disk_pct:.0f}% - running cleanup")
            removed = _cleanup_old_outputs()
            _log(f"[cleanup] removed {removed} old files")
            if removed > 0:
                new_pct = _disk_usage_percent()
                _send_telegram(
                    f"Disk cleanup: removed {removed} files\n"
                    f"Disk: {disk_pct:.0f}% -> {new_pct:.0f}%"
                )
        else:
            removed = _cleanup_old_outputs()
            if removed > 0:
                _log(f"[cleanup] routine cleanup: removed {removed} old files")
        state["last_cleanup_epoch"] = now

    # --- 5. Periodic status report ---
    if now - state.get("last_status_report_epoch", 0) > STATUS_REPORT_INTERVAL_SEC:
        report = _build_status_report(state)
        _log(f"[status] sending periodic report")
        _send_telegram(report)
        state["last_status_report_epoch"] = now


if __name__ == "__main__":
    main()
