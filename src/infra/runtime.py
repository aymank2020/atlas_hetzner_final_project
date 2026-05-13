"""Runtime lifecycle helpers for graceful Atlas solver shutdown."""

from __future__ import annotations

import signal
import threading
import time
from typing import Any, Callable, Optional

_shutdown_requested = threading.Event()


def _request_shutdown(signum: int, frame: Any) -> None:
    """Signal handler: set shutdown flag for clean exit."""
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    print(f"\n[shutdown] received {sig_name}; shutting down gracefully after current step...")
    _shutdown_requested.set()


def _install_signal_handlers() -> None:
    """Register SIGTERM/SIGINT handlers for clean VPS shutdown."""
    try:
        signal.signal(signal.SIGINT, _request_shutdown)
        signal.signal(signal.SIGTERM, _request_shutdown)
        print("[init] graceful shutdown handlers installed (SIGINT/SIGTERM).")
    except (OSError, ValueError):
        # signal handlers can only be set in the main thread
        pass


def _sleep_with_shutdown_heartbeat(
    wait_sec: float,
    *,
    heartbeat_sec: float = 30.0,
    on_heartbeat: Optional[Callable[[], None]] = None,
) -> None:
    """
    Sleep in short chunks so long intentional waits do not look like silent hangs.

    This keeps watchdogs and outer supervisors informed that the process is alive
    during quota backoff, idle keep-alive loops, and other deliberate pauses.
    """
    remaining = max(0.0, float(wait_sec))
    if remaining <= 0.0:
        return

    heartbeat = max(0.01, float(heartbeat_sec))
    while remaining > 0.0:
        if _shutdown_requested.is_set():
            break
        chunk = min(remaining, heartbeat)
        time.sleep(chunk)
        remaining -= chunk
        if on_heartbeat is not None:
            try:
                on_heartbeat()
            except Exception:
                pass


__all__ = [
    "_shutdown_requested",
    "_request_shutdown",
    "_install_signal_handlers",
    "_sleep_with_shutdown_heartbeat",
]
