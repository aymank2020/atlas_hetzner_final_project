"""Append-only JSONL execution journal for episode-level tracing."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.infra.solver_config import _cfg_get

_JOURNAL_LOCK = threading.Lock()
_PEAK_RSS_MB = 0.0


def _utc_now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _journal_root(cfg: Dict[str, Any]) -> Path:
    out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs") or "outputs"))
    root = out_dir / "_journal"
    root.mkdir(parents=True, exist_ok=True)
    return root


def journal_path(cfg: Dict[str, Any], episode_id: str) -> Path:
    safe_episode = str(episode_id or "").strip() or "unknown_episode"
    return _journal_root(cfg) / f"{safe_episode}.jsonl"


def _current_rss_mb() -> Optional[float]:
    try:
        import psutil  # type: ignore

        return round(float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024), 2)
    except Exception:
        return None


def _rss_snapshot_mb() -> tuple[Optional[float], Optional[float]]:
    global _PEAK_RSS_MB
    current = _current_rss_mb()
    if current is not None:
        _PEAK_RSS_MB = max(float(_PEAK_RSS_MB), float(current))
    peak = round(float(_PEAK_RSS_MB), 2) if _PEAK_RSS_MB > 0 else None
    return current, peak


def _compact_submit_outcome(payload: Any) -> Dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    return {
        "submit_verified": bool(data.get("submit_verified", False)),
        "submit_verification_reason": str(data.get("submit_verification_reason", "") or "").strip(),
        "terminal_failure": bool(data.get("terminal_failure", False)),
        "page_url_before_submit": str(data.get("page_url_before_submit", "") or "").strip(),
        "page_url_after_submit": str(data.get("page_url_after_submit", "") or "").strip(),
    }


def _compact_apply_budget(payload: Any) -> Dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    out: Dict[str, Any] = {}
    for key in (
        "target_count",
        "applied_count",
        "skipped_count",
        "consecutive_failures",
        "deadline_at",
        "budget_extensions",
        "status",
        "elapsed_sec",
        "remaining_sec",
        "stalled_for_sec",
    ):
        if key in data:
            out[key] = data.get(key)
    return out


def append_execution_journal_event(
    cfg: Dict[str, Any],
    *,
    episode_id: str,
    event_type: str,
    stage: str = "",
    reason: str = "",
    task_state: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
    run_id: str = "",
    context_id: str = "",
    request_id: str = "",
    mode: str = "",
    baseline_message_count: Optional[int] = None,
    segments_checksum: str = "",
    repair_round: Optional[int] = None,
    page_url: str = "",
) -> str:
    state = task_state if isinstance(task_state, dict) else {}
    extra = payload if isinstance(payload, dict) else {}
    resolved_request_id = str(request_id or state.get("gemini_last_request_id", "") or "").strip()
    resolved_context_id = str(context_id or state.get("context_id", "") or "").strip()
    resolved_run_id = str(run_id or resolved_context_id or state.get("run_id", "") or "").strip()
    resolved_mode = str(mode or state.get("gemini_last_request_mode", "") or "").strip()
    resolved_baseline = (
        int(baseline_message_count)
        if baseline_message_count is not None
        else int(state.get("gemini_last_baseline_message_count", 0) or 0)
    )
    resolved_checksum = str(
        segments_checksum
        or state.get("gemini_last_segments_checksum", "")
        or state.get("segment_checksum", "")
        or ""
    ).strip()
    resolved_page_url = str(
        page_url
        or state.get("submit_page_url_after", "")
        or state.get("submit_page_url_before", "")
        or state.get("page_url", "")
        or ""
    ).strip()

    rss_mb, peak_rss_mb = _rss_snapshot_mb()
    event = {
        "timestamp": _utc_now_text(),
        "episode_id": str(episode_id or "").strip(),
        "run_id": resolved_run_id,
        "context_id": resolved_context_id,
        "stage": str(stage or state.get("stage", "") or state.get("current_stage", "") or "").strip(),
        "status": str(state.get("status", "") or state.get("current_stage_status", "") or "").strip(),
        "event_type": str(event_type or "").strip(),
        "request_id": resolved_request_id,
        "mode": resolved_mode,
        "baseline_message_count": max(0, resolved_baseline),
        "segments_checksum": resolved_checksum,
        "repair_round": int(repair_round or state.get("repair_round", 0) or 0),
        "apply_budget_state": _compact_apply_budget(
            extra.get("apply_budget_state", state.get("apply_budget_state", {}))
        ),
        "submit_outcome_excerpt": _compact_submit_outcome(
            extra.get("submit_outcome", extra.get("submit_status", state.get("submit_outcome", state.get("submit_status", {}))))
        ),
        "page_url": resolved_page_url,
        "reason": str(reason or extra.get("reason", "") or "").strip(),
        "rss_mb": rss_mb,
        "peak_rss_mb": peak_rss_mb,
    }
    for key, value in extra.items():
        if key in event:
            continue
        event[key] = value

    target = journal_path(cfg, episode_id)
    line = json.dumps(event, ensure_ascii=False)
    with _JOURNAL_LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return str(target)


__all__ = [
    "append_execution_journal_event",
    "journal_path",
]
