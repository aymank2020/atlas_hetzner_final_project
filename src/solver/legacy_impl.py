"""
Atlas browser auto-solver:
1) Login to audit.atlascapture.io
2) Auto-read OTP from Gmail (IMAP)
3) Open task room and extract segments
4) Send segments to Gemini API
5) Optionally write labels back into Atlas
"""

from __future__ import annotations

import argparse
import base64
import html
import imaplib
import json
import math
import random
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from email import message_from_bytes
from email.header import decode_header
from email.message import Message
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import requests
import yaml
from dotenv import load_dotenv
from playwright.sync_api import Locator, Page, sync_playwright
from src.infra.execution_journal import append_execution_journal_event
from src.infra.gemini_economics import (
    cost_guard_enforcement_enabled,
    estimate_minimum_episode_cost_usd,
)
from src.infra import artifacts as _artifacts
from src.infra import browser_auth as _browser_auth
from src.infra import runtime as _runtime
from src.infra import solver_config as _solver_config
from src.infra import utils as _utils
from src.policy import context_manager as policy_context
from src.rules import labels as _labels
from src.rules import policy_gate as _policy_gate
from src.solver import browser as _browser
from src.solver.desync import build_segment_checksum
from src.solver.episode_runtime import EpisodeRuntime
from src.solver import gemini as _gemini
from src.solver.live_validation import ValidationTracker
from src.solver import orchestrator as _orchestrator
from src.solver import pre_submit_compare as _pre_submit_compare
from src.solver import prompting as _prompting
from src.solver.reliability import EpisodeReport, FailureClass

# Load .env file if present
load_dotenv()

# ?????? Graceful Shutdown Handler ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
# Enables clean context close on SIGTERM/SIGINT (Windows Service, Task Scheduler)
_shutdown_requested = _runtime._shutdown_requested
_request_shutdown = _runtime._request_shutdown
_install_signal_handlers = _runtime._install_signal_handlers
_sleep_with_shutdown_heartbeat = _runtime._sleep_with_shutdown_heartbeat
_ACTIVE_HEARTBEAT_CALLBACK: Optional[Callable[[], None]] = None


def _should_clear_blocked_tasks_before_idle_retry(
    *,
    clear_blocked_tasks_every_retry: bool,
    blocked_task_ids: set[str],
    open_status: Optional[Dict[str, Any]] = None,
) -> bool:
    """Never clear blocked tasks during a plain idle retry.

    The ``clear_blocked_tasks_every_retry`` parameter is intentionally unused:
    blocked tasks must remain sticky across plain idle retries to avoid
    re-reserving the same blocked episode.  Clearing is handled by the
    separate ``all_visible_blocked`` path instead.
    """
    if not blocked_task_ids:
        return False
    # Intentionally always False: idle retry must not clear blocked tasks.
    return False


def _persist_task_state_fields(
    cfg: Dict[str, Any],
    task_id: str,
    task_state: Optional[Dict[str, Any]] = None,
    **updates: Any,
) -> Dict[str, Any]:
    task_id = str(task_id or "").strip()
    if not task_id:
        return task_state if isinstance(task_state, dict) else {}

    existing: Dict[str, Any] = {}
    try:
        loaded = _load_task_state(cfg, task_id)
        if isinstance(loaded, dict):
            existing = dict(loaded)
    except Exception:
        existing = {}

    merged = dict(existing)
    if isinstance(task_state, dict):
        merged.update(task_state)
    merged["task_id"] = task_id
    merged.update(updates)

    _save_task_state(cfg, task_id, merged)
    if isinstance(task_state, dict):
        task_state.clear()
        task_state.update(merged)
        return task_state
    return merged


def _maybe_clear_sticky_resume_targets(
    cfg: Dict[str, Any],
    target_task_urls: Sequence[str],
    target_task_ids: Sequence[str],
    open_status: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str], bool]:
    status = open_status if isinstance(open_status, dict) else {}
    sticky_exhausted = bool(status.get("sticky_resume_exhausted"))
    normalized_urls = [str(url or "").strip() for url in target_task_urls if str(url or "").strip()]
    normalized_ids = [str(task_id or "").strip() for task_id in target_task_ids if str(task_id or "").strip()]
    if not sticky_exhausted or not normalized_urls:
        return list(normalized_urls), list(normalized_ids), False

    stale_ids: List[str] = []
    for stale_url in normalized_urls:
        stale_task_id = _task_id_from_url(stale_url)
        if not stale_task_id:
            continue
        stale_ids.append(stale_task_id)
        _persist_task_state_fields(
            cfg,
            stale_task_id,
            None,
            episode_locked=False,
            last_error="sticky_resume_exhausted",
            terminal_failure_kind="sticky_resume_exhausted",
        )
    print("[run] sticky resume target became unavailable; cleared sticky lock and reverted to queue mode.")
    return [], [], True


def _can_resume_sticky_task_state(sticky_state: Any) -> bool:
    if not isinstance(sticky_state, dict):
        return False
    sticky_locked = bool(sticky_state.get("episode_locked", False))
    sticky_status = str(sticky_state.get("status", "") or "").strip().lower()
    sticky_episode_status = str(sticky_state.get("episode_status", "") or "").strip().lower()
    sticky_submitted = bool(sticky_state.get("episode_submitted", False))
    terminal_failure_kind = str(sticky_state.get("terminal_failure_kind", "") or "").strip().lower()
    last_error = str(sticky_state.get("last_error", "") or "").strip().lower()
    sticky_url = str(sticky_state.get("task_url", "") or "").strip()
    sticky_task_id = str(sticky_state.get("task_id", "") or "").strip()

    if not sticky_locked:
        return False
    if not sticky_url or not sticky_task_id:
        return False
    if sticky_submitted:
        return False
    if sticky_status in {"completed", "failed_terminal"}:
        return False
    if sticky_episode_status in {"completed", "failed"}:
        return False
    if terminal_failure_kind or last_error == "sticky_resume_exhausted":
        return False
    return True


def _utc_now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _canonical_task_stage(stage: str, status: str, *, terminal_failure_kind: str = "") -> str:
    stage_name = str(stage or "").strip().lower()
    status_name = str(status or "").strip().lower()
    if status_name == "completed" and stage_name == "submit":
        return "completed"
    if status_name == "failed" and str(terminal_failure_kind or "").strip():
        return "failed_terminal"
    if "submit" in stage_name:
        return "submit_verifying"
    if "apply" in stage_name or "timestamp" in stage_name:
        return "applying"
    if "repair" in stage_name or "policy" in stage_name:
        return "repairing"
    if "chat" in stage_name or "gemini" in stage_name:
        return "waiting_for_gemini"
    return "running"


def _canonical_task_status(status: str, *, terminal_failure_kind: str = "", stage: str = "") -> str:
    status_name = str(status or "").strip().lower() or "running"
    if status_name == "completed" and str(stage or "").strip().lower() == "submit":
        return "completed"
    if status_name == "failed" and str(terminal_failure_kind or "").strip():
        return "failed_terminal"
    if status_name == "completed":
        return "running"
    return "running"


def _stage_watchdog_timeout_hint_sec(
    cfg: Dict[str, Any],
    *,
    stage: str,
    base_timeout_sec: float,
    progress_current: int = 0,
    progress_total: int = 0,
) -> float:
    stage_name = str(stage or "").strip().lower()
    total_units = max(int(progress_total or 0), int(progress_current or 0), 0)
    cap_sec = max(
        float(base_timeout_sec),
        float(_cfg_get(cfg, "run.watchdog_dynamic_timeout_cap_sec", max(2400.0, base_timeout_sec))),
    )
    per_unit_defaults = {
        "extract_segments": 1.5,
        "chat_labels": 1.0,
        "apply_labels": 0.35,
        "submit": 0.15,
    }
    per_unit_sec = per_unit_defaults.get(stage_name, 0.0)
    extra_sec = min(max(0.0, cap_sec - float(base_timeout_sec)), total_units * per_unit_sec)
    return max(float(base_timeout_sec), min(cap_sec, float(base_timeout_sec) + extra_sec))


def _persist_task_stage_status(
    cfg: Dict[str, Any],
    task_id: str,
    task_state: Optional[Dict[str, Any]],
    *,
    stage: str,
    status: str,
    progress_current: Optional[int] = None,
    progress_total: Optional[int] = None,
    detail: str = "",
    last_error: str = "",
    watchdog_timeout_sec: Optional[float] = None,
    terminal_failure_kind: str = "",
) -> Dict[str, Any]:
    task = str(task_id or "").strip()
    stage_name = str(stage or "").strip()
    state = task_state if isinstance(task_state, dict) else {}
    if not task or not stage_name:
        return state

    status_value = str(status or "").strip().lower() or "running"
    now_utc = _utc_now_text()
    previous_stage = str(state.get("current_stage", "") or "").strip()
    previous_status = str(state.get("current_stage_status", "") or "").strip().lower()
    started_at = str(state.get("current_stage_started_at_utc", "") or "").strip()
    if stage_name != previous_stage or status_value == "running" and previous_status != "running":
        started_at = now_utc

    episode_status = str(state.get("episode_status", "") or "").strip().lower()
    if status_value == "failed":
        episode_status = "failed"
    elif stage_name == "submit" and status_value == "completed":
        episode_status = "completed"
    elif status_value == "running":
        episode_status = "running"
    elif not episode_status or episode_status == "running":
        episode_status = "running"

    updates: Dict[str, Any] = {
        "episode_status": episode_status,
        "current_stage": stage_name,
        "current_stage_status": status_value,
        "stage": _canonical_task_stage(
            stage_name,
            status_value,
            terminal_failure_kind=terminal_failure_kind,
        ),
        "status": _canonical_task_status(
            status_value,
            terminal_failure_kind=terminal_failure_kind,
            stage=stage_name,
        ),
        "terminal_failure_kind": str(terminal_failure_kind or "").strip(),
        "current_stage_started_at_utc": started_at,
        "current_stage_last_progress_at_utc": now_utc,
        "current_stage_detail": str(detail or "").strip(),
        "current_stage_error": str(last_error or "").strip(),
    }
    if progress_current is not None:
        updates["current_stage_progress_current"] = int(progress_current)
    if progress_total is not None:
        updates["current_stage_progress_total"] = int(progress_total)
    if watchdog_timeout_sec is not None:
        updates["current_stage_watchdog_timeout_sec"] = round(float(watchdog_timeout_sec), 1)
    if status_value in {"completed", "failed"}:
        updates["current_stage_completed_at_utc"] = now_utc
    elif status_value == "running":
        updates["current_stage_completed_at_utc"] = ""
    if status_value == "failed" and str(last_error or "").strip():
        updates["last_error"] = str(last_error).strip()
    elif status_value == "running":
        updates["last_error"] = ""
    persisted = _persist_task_state_fields(cfg, task, task_state, **updates)
    append_execution_journal_event(
        cfg,
        episode_id=task,
        event_type="task_stage_update",
        stage=str(persisted.get("stage", "") or "").strip(),
        reason=str(detail or last_error or "").strip(),
        task_state=persisted,
        payload={
            "detailed_stage": stage_name,
            "detailed_status": status_value,
            "progress_current": int(progress_current or 0) if progress_current is not None else None,
            "progress_total": int(progress_total or 0) if progress_total is not None else None,
            "terminal_failure_kind": str(terminal_failure_kind or "").strip(),
        },
        run_id=str(persisted.get("run_id", persisted.get("context_id", "")) or "").strip(),
        context_id=str(persisted.get("context_id", "") or "").strip(),
        page_url=str(persisted.get("page_url", "") or "").strip(),
    )
    return persisted


def _episode_model_state_updates(
    cfg: Dict[str, Any],
    labels_payload: Optional[Dict[str, Any]],
    task_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = labels_payload.get("_meta", {}) if isinstance(labels_payload, dict) else {}
    primary_model = str(_cfg_get(cfg, "gemini.model", "gemini-3.1-pro-preview") or "").strip()
    active_model = str(
        meta.get("episode_active_model")
        or meta.get("model")
        or (task_state.get("episode_active_model", "") if isinstance(task_state, dict) else "")
        or primary_model
    ).strip()
    if not active_model:
        return {}
    escalated = bool(meta.get("episode_model_escalated", False))
    if not escalated and primary_model:
        escalated = active_model.lower() != primary_model.lower()
    reason = str(
        meta.get("episode_fallback_reason")
        or (task_state.get("episode_fallback_reason", "") if isinstance(task_state, dict) else "")
        or ""
    ).strip()
    updates = {
        "episode_active_model": active_model,
        "episode_model_escalated": escalated,
        "episode_fallback_reason": reason,
    }
    key_class_used = str(
        meta.get("api_key_class")
        or meta.get("episode_key_class_used")
        or (task_state.get("episode_key_class_used", "") if isinstance(task_state, dict) else "")
        or ""
    ).strip()
    key_class_pin = str(
        meta.get("episode_key_class_pin")
        or (task_state.get("episode_key_class_pin", "") if isinstance(task_state, dict) else "")
        or ""
    ).strip()
    if key_class_used:
        updates["episode_key_class_used"] = key_class_used
    if key_class_pin:
        updates["episode_key_class_pin"] = key_class_pin
    solve_backend = str(meta.get("solve_backend") or "").strip()
    if solve_backend:
        updates["solve_backend"] = solve_backend
    if "chat_only_mode" in meta:
        updates["chat_only_mode"] = bool(meta.get("chat_only_mode", False))
    if "chat_compare_skipped" in meta:
        updates["chat_compare_skipped"] = bool(meta.get("chat_compare_skipped", False))
    if "chat_ops_attempted" in meta:
        updates["chat_ops_attempted"] = bool(meta.get("chat_ops_attempted", False))
    if "chat_ops_planned" in meta:
        updates["chat_ops_planned"] = int(meta.get("chat_ops_planned", 0) or 0)
    if "chat_labels_chunk_count" in meta:
        updates["chat_labels_chunk_count"] = int(meta.get("chat_labels_chunk_count", 0) or 0)
    chat_response_json_path = str(meta.get("chat_response_json_path") or "").strip()
    if chat_response_json_path:
        updates["chat_response_json_path"] = chat_response_json_path
    chat_prompt_path = str(meta.get("chat_prompt_path") or "").strip()
    if chat_prompt_path:
        updates["chat_prompt_path"] = chat_prompt_path
    return updates


def _submit_state_updates(
    result: Optional[Dict[str, Any]] = None,
    *,
    episode_submitted: bool,
    last_error: str,
    debug_screenshot: str = "",
    debug_html: str = "",
) -> Dict[str, Any]:
    payload = result if isinstance(result, dict) else {}
    submit_status = payload.get("submit_status", {}) if isinstance(payload.get("submit_status"), dict) else {}
    guard_reasons = payload.get("submit_guard_reasons", []) if isinstance(payload.get("submit_guard_reasons"), list) else []

    updates: Dict[str, Any] = {
        "episode_submitted": bool(episode_submitted),
        "last_error": str(last_error or "").strip(),
        "episode_status": "completed" if episode_submitted else ("failed" if str(last_error or "").strip() else "running"),
        "current_stage": "submit",
        "current_stage_status": "completed" if episode_submitted else ("failed" if str(last_error or "").strip() else "running"),
        "stage": "completed" if episode_submitted else "failed_terminal",
        "status": "completed" if episode_submitted else "failed_terminal",
        "terminal_failure_kind": "" if episode_submitted else "terminal_submit_failure",
        "current_stage_last_progress_at_utc": _utc_now_text(),
        "submit_attempted": bool(submit_status.get("submit_attempted", False)),
        "submit_verified": bool(submit_status.get("submit_verified", False)),
        "submit_verification_reason": str(submit_status.get("submit_verification_reason", "") or "").strip(),
        "submit_page_url_before": str(submit_status.get("page_url_before_submit", "") or "").strip(),
        "submit_page_url_after": str(submit_status.get("page_url_after_submit", "") or "").strip(),
        "submit_complete_button_clicked": bool(submit_status.get("complete_button_clicked", False)),
        "submit_complete_button_retried": bool(submit_status.get("complete_button_retried", False)),
        "submit_modal_already_open": bool(submit_status.get("submit_modal_already_open", False)),
        "submit_saw_no_edits_modal": bool(submit_status.get("saw_no_edits_modal", False)),
        "submit_saw_quality_review_modal": bool(submit_status.get("saw_quality_review_modal", False)),
        "submit_saw_post_submit_transition": bool(submit_status.get("saw_post_submit_transition", False)),
        "submit_no_edits_confirmed": bool(submit_status.get("no_edits_confirmed", False)),
        "submit_quality_review_confirmed": bool(submit_status.get("quality_review_confirmed", False)),
        "submit_manual_watch_used": bool(submit_status.get("manual_submit_watch_used", False)),
        "submit_manual_submit_detected": bool(submit_status.get("manual_submit_detected", False)),
        "submit_manual_watch_reason": str(submit_status.get("manual_submit_watch_reason", "") or "").strip(),
        "submit_manual_watch_signal": str(submit_status.get("manual_submit_watch_signal", "") or "").strip(),
        "submit_manual_watch_timed_out": bool(submit_status.get("manual_submit_watch_timed_out", False)),
        "submit_manual_watch_elapsed_sec": float(submit_status.get("manual_submit_watch_elapsed_sec", 0.0) or 0.0),
        "submit_terminal_failure": bool(
            submit_status.get("terminal_failure", False)
            or (submit_status.get("submit_attempted", False) and not episode_submitted)
        ),
        "submit_guard_blocked": bool(payload.get("submit_guard_blocked", False)),
        "submit_guard_reasons": [str(item) for item in guard_reasons[:20]],
        "submit_debug_screenshot": str(debug_screenshot or "").strip(),
        "submit_debug_html": str(debug_html or "").strip(),
        "submit_outcome": dict(submit_status) if isinstance(submit_status, dict) else {},
        "episode_locked": not bool(episode_submitted),
    }
    apply_budget_state = payload.get("apply_budget_state", {}) if isinstance(payload.get("apply_budget_state"), dict) else {}
    if apply_budget_state:
        updates["apply_budget_state"] = {
            key: apply_budget_state.get(key)
            for key in (
                "target_count",
                "applied_count",
                "consecutive_failures",
                "budget_extensions",
                "elapsed_sec",
                "remaining_sec",
                "stalled_for_sec",
            )
            if key in apply_budget_state
        }
    if episode_submitted:
        updates["submitted_at_utc"] = _utc_now_text()
        updates["current_stage_completed_at_utc"] = updates["submitted_at_utc"]
    elif str(last_error or "").strip():
        updates["current_stage_completed_at_utc"] = _utc_now_text()
    return updates


def _persist_submit_outcome(
    cfg: Dict[str, Any],
    task_id: str,
    task_state: Optional[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    *,
    page: Optional[Page],
    episode_submitted: bool,
    last_error: str,
) -> Dict[str, Any]:
    debug_screenshot = ""
    debug_html = ""
    submit_status = result.get("submit_status", {}) if isinstance(result, dict) else {}
    submit_attempted = bool(submit_status.get("submit_attempted", False)) if isinstance(submit_status, dict) else False
    if not episode_submitted and submit_attempted and page is not None:
        snap_path, html_path = _capture_debug_artifacts(page, cfg, prefix="debug_submit_unverified")
        debug_screenshot = str(snap_path) if snap_path else ""
        debug_html = str(html_path) if html_path else ""
    updates = _submit_state_updates(
        result,
        episode_submitted=episode_submitted,
        last_error=last_error,
        debug_screenshot=debug_screenshot,
        debug_html=debug_html,
    )
    persisted = _persist_task_state_fields(
        cfg,
        task_id,
        task_state,
        **updates,
    )
    append_execution_journal_event(
        cfg,
        episode_id=task_id,
        event_type="submit_outcome_recorded",
        stage=str(persisted.get("stage", "") or "submit_verifying").strip(),
        reason=str(updates.get("submit_verification_reason", "") or last_error or "").strip(),
        task_state=persisted,
        payload={
            "submit_status": dict(submit_status) if isinstance(submit_status, dict) else {},
            "apply_budget_state": dict((result or {}).get("apply_budget_state", {}) or {}),
        },
        run_id=str(persisted.get("run_id", persisted.get("context_id", "")) or "").strip(),
        context_id=str(persisted.get("context_id", "") or "").strip(),
        page_url=str(
            persisted.get("submit_page_url_after", "")
            or persisted.get("submit_page_url_before", "")
            or ""
        ).strip(),
    )
    return persisted


def _capture_episode_step(
    cfg: Dict[str, Any],
    page: Optional[Page],
    task_id: str,
    task_state: Optional[Dict[str, Any]],
    step_name: str,
    *,
    include_html: bool = False,
) -> Dict[str, Any]:
    task = str(task_id or "").strip()
    step = str(step_name or "").strip()
    if page is None or not task or not step:
        return task_state if isinstance(task_state, dict) else {}

    artifact = _capture_step_artifacts(
        page,
        cfg,
        task,
        step,
        include_html=include_html,
    )
    if not artifact:
        return task_state if isinstance(task_state, dict) else {}

    history_limit = max(1, int(_cfg_get(cfg, "run.capture_step_history_limit", 16)))
    history: list[dict[str, Any]] = []
    if isinstance(task_state, dict):
        existing = task_state.get("step_artifacts")
        if isinstance(existing, list):
            history = [dict(item) for item in existing if isinstance(item, dict)]
    history.append(artifact)
    return _persist_task_state_fields(
        cfg,
        task,
        task_state,
        last_step_name=step,
        last_step_screenshot=str(artifact.get("screenshot", "") or ""),
        last_step_html=str(artifact.get("html", "") or ""),
        step_artifacts=history[-history_limit:],
    )


def _runtime_event_logger(event: str, payload: Dict[str, Any]) -> None:
    try:
        body = {"event": str(event or "").strip(), **dict(payload or {})}
        print(f"[runtime] {json.dumps(body, ensure_ascii=False, sort_keys=True)}")
    except Exception:
        print(f"[runtime] event={event} payload={payload}")


def _default_bootstrap_queue_url(page_url: str) -> str:
    parsed = urlparse(str(page_url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}/tasks"


def _episode_reports_dir(cfg: Dict[str, Any]) -> Path:
    out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs")))
    path = out_dir / "episode_reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _emit_episode_report(
    cfg: Dict[str, Any],
    report: EpisodeReport,
    *,
    task_state: Optional[Dict[str, Any]] = None,
    lifecycle_events: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Path]:
    if not bool(_cfg_get(cfg, "run.structured_episode_reports", False)):
        return None

    out_dir = _episode_reports_dir(cfg)
    payload = report.to_dict()
    if isinstance(task_state, dict):
        payload["task_state_excerpt"] = {
            "last_error": str(task_state.get("last_error", "") or "").strip(),
            "submit_verified": bool(task_state.get("submit_verified", False)),
            "submit_guard_blocked": bool(task_state.get("submit_guard_blocked", False)),
            "submit_terminal_failure": bool(task_state.get("submit_terminal_failure", False)),
            "submit_verification_reason": str(task_state.get("submit_verification_reason", "") or "").strip(),
            "episode_submitted": bool(task_state.get("episode_submitted", False)),
            "live_validation_report": str(task_state.get("live_validation_report", "") or "").strip(),
        }
    if lifecycle_events:
        payload["lifecycle_events"] = [dict(item) for item in lifecycle_events]

    report_path = out_dir / f"episode_{str(report.episode_id or 'unknown').strip() or 'unknown'}.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(
        f"[monitor] episode report saved: episode={report.episode_id or 'unknown'} "
        f"context_id={report.context_id or 'bootstrap'} failure_class={report.failure_class or 'none'}"
    )
    return report_path


def _is_authenticated_gemini_page(page: Any) -> bool:
    try:
        url = str(getattr(page, "url", "") or "").strip().lower()
    except Exception:
        url = ""
    try:
        title = str(page.title() or "").strip().lower()
    except Exception:
        title = ""
    try:
        body = re.sub(r"\s+", " ", str(page.locator("body").inner_text(timeout=2000) or "")).strip().lower()
    except Exception:
        body = ""
    sign_in_markers = (
        "sign in gemini",
        "get access to all gemini models",
        "meet gemini, your personal ai assistant",
    )
    if "accounts.google.com" in url:
        return False
    if any(marker in body for marker in sign_in_markers):
        return False
    if "sign in" in title and "gemini" in title:
        return False
    try:
        composer = page.locator('div[contenteditable="true"], textarea').first
        if composer.is_visible(timeout=1200) and not any(marker in body for marker in sign_in_markers):
            return True
    except Exception:
        pass
    return "gemini.google.com/app" in url and "sign in" not in body


def _acquire_gemini_probe_page(
    gemini_context: Any,
    *,
    gemini_chat_url: str,
) -> tuple[Any, bool]:
    fallback_page = None
    for page in reversed(list(getattr(gemini_context, "pages", []) or [])):
        try:
            page_url = str(getattr(page, "url", "") or "").strip().lower()
        except Exception:
            page_url = ""
        if "gemini.google.com" not in page_url:
            continue
        if _is_authenticated_gemini_page(page):
            return page, False
        if fallback_page is None:
            fallback_page = page
    if fallback_page is not None:
        return fallback_page, False

    probe_page = gemini_context.new_page()
    target_url = str(gemini_chat_url or "").strip()
    if target_url:
        try:
            probe_page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
            probe_page.wait_for_timeout(5000)
        except Exception:
            pass
    return probe_page, True


def _activate_episode_runtime_v2(
    *,
    browser: Any,
    gemini_browser: Any = None,
    bootstrap_context: Any,
    bootstrap_page: Page,
    state_path: Path,
    cfg: Dict[str, Any],
    task_id: str,
) -> tuple[Optional[EpisodeRuntime], Page, Any]:
    use_v2 = bool(_cfg_get(cfg, "run.use_episode_runtime_v2", False))
    force_isolation = bool(_cfg_get(cfg, "run.force_episode_browser_isolation", False))
    strict_single_session = bool(_cfg_get(cfg, "run.strict_single_chat_session", False))
    single_window_two_tabs = bool(_cfg_get(cfg, "run.single_window_two_tabs", False))
    single_window_single_tab = bool(_cfg_get(cfg, "run.single_window_single_tab", False))
    if not use_v2 or not force_isolation or browser is None:
        return None, bootstrap_page, bootstrap_context

    task = str(task_id or "").strip()
    if not task:
        return None, bootstrap_page, bootstrap_context

    gemini_state_path = Path(
        str(_cfg_get(cfg, "gemini.chat_web_storage_state", ".state/gemini_chat_storage_state.json") or "").strip()
        or ".state/gemini_chat_storage_state.json"
    )
    gemini_chat_url = str(
        _cfg_get(cfg, "gemini.chat_web_url", "https://gemini.google.com/app") or ""
    ).strip() or "https://gemini.google.com/app"
    reuse_gemini_cdp_context = bool(_cfg_get(cfg, "gemini.chat_web_reuse_cdp_context", False))
    require_authenticated_gemini = bool(
        _cfg_get(cfg, "gemini.chat_web_require_authenticated_session", False)
    )
    gemini_existing_context = None

    try:
        if _is_authenticated_page(bootstrap_page):
            _ensure_parent(state_path)
            bootstrap_context.storage_state(path=str(state_path))
    except Exception as exc:
        print(f"[runtime] warning: could not refresh storage state before isolation: {exc}")

    if strict_single_session and gemini_browser is not None:
        try:
            gemini_contexts = list(getattr(gemini_browser, "contexts", []) or [])
            if gemini_contexts:
                gemini_context = gemini_contexts[0]
                gemini_probe_page, created_probe_page = _acquire_gemini_probe_page(
                    gemini_context,
                    gemini_chat_url=gemini_chat_url,
                )
                gemini_authenticated = _is_authenticated_gemini_page(gemini_probe_page)
                if gemini_authenticated:
                    _ensure_parent(gemini_state_path)
                    gemini_context.storage_state(path=str(gemini_state_path))
                    if reuse_gemini_cdp_context:
                        gemini_existing_context = gemini_context
                    elif created_probe_page:
                        try:
                            gemini_probe_page.close()
                        except Exception:
                            pass
                else:
                    if created_probe_page:
                        try:
                            gemini_probe_page.close()
                        except Exception:
                            pass
                    print(
                        "[runtime] warning: Gemini CDP context is not authenticated; "
                        "preserving existing storage_state fallback and skipping borrowed Gemini reuse."
                    )
        except Exception as exc:
            print(f"[runtime] warning: could not refresh Gemini storage state before isolation: {exc}")

    runtime = EpisodeRuntime(task)
    _gemini_cdp = str(
        _cfg_get(cfg, "gemini.chat_web_connect_over_cdp_url", "")
        or os.environ.get("GEMINI_CHAT_CONNECT_OVER_CDP_URL", "")
        or ""
    ).strip()
    runtime.gemini_cdp_url = _gemini_cdp
    try:
        atlas_existing_context = None
        atlas_existing_page = None
        atlas_page_url = str(getattr(bootstrap_page, "url", "") or "")
        bootstrap_queue_url = str(_cfg_get(cfg, "atlas.room_url", "") or "").strip()
        if not bootstrap_queue_url:
            bootstrap_queue_url = _default_bootstrap_queue_url(atlas_page_url)
        if single_window_single_tab:
            atlas_existing_context = bootstrap_context
            atlas_existing_page = bootstrap_page
        elif single_window_two_tabs:
            atlas_existing_context = bootstrap_context

        runtime.open(
            atlas_browser=None if atlas_existing_context is not None else browser,
            atlas_existing_context=atlas_existing_context,
            atlas_existing_page=atlas_existing_page,
            atlas_storage_state_path=str(state_path) if state_path.exists() else "",
            atlas_page_url=atlas_page_url,
            gemini_browser=gemini_browser if strict_single_session and gemini_browser is not None else (browser if strict_single_session and not require_authenticated_gemini else None),
            gemini_existing_context=gemini_existing_context if strict_single_session else None,
            gemini_storage_state_path=str(gemini_state_path) if strict_single_session and gemini_state_path.exists() else "",
            gemini_page_url=gemini_chat_url if strict_single_session else "",
            logger=_runtime_event_logger,
        )
        if runtime.atlas_page is None or runtime.atlas_context is None:
            raise RuntimeError("EpisodeRuntime did not create an Atlas page/context.")
        if strict_single_session and require_authenticated_gemini and runtime.gemini_page is None:
            raise RuntimeError("Authenticated Gemini session is required, but no Gemini page is available.")
        try:
            runtime.atlas_page.wait_for_timeout(1200)
        except Exception:
            pass
        new_task_id = _task_id_from_url(str(getattr(runtime.atlas_page, "url", "") or ""))
        if new_task_id and new_task_id != task:
            raise RuntimeError(
                f"isolated episode page opened unexpected task: expected={task} got={new_task_id}"
            )
        if (
            bootstrap_page is not None
            and bootstrap_page is not runtime.atlas_page
            and bootstrap_queue_url
        ):
            current_bootstrap_url = str(getattr(bootstrap_page, "url", "") or "").strip()
            if current_bootstrap_url != bootstrap_queue_url:
                try:
                    _goto_with_retry(
                        bootstrap_page,
                        bootstrap_queue_url,
                        wait_until="domcontentloaded",
                        timeout_ms=45000,
                        cfg=cfg,
                        reason="runtime-bootstrap-return",
                    )
                    bootstrap_page.wait_for_timeout(800)
                    print(f"[runtime] returned bootstrap page to queue: {bootstrap_queue_url}")
                except Exception as exc:
                    print(f"[runtime] warning: failed to return bootstrap page to queue: {exc}")
        print(
            f"[runtime] activated Atlas episode context: task={task} "
            f"context_id={runtime.context_id} page={getattr(runtime.atlas_page, 'url', '')}"
        )
        if strict_single_session and runtime.gemini_page is not None:
            try:
                from src.solver import chat_only as _chat_only

                session = _chat_only.register_episode_gemini_session(
                    episode_id=task,
                    runtime=runtime,
                    cfg=cfg,
                )
                if session is not None:
                    runtime.task_state["gemini_session_id"] = session.session_id
                    print(
                        f"[runtime] activated Gemini episode session: task={task} "
                        f"context_id={runtime.context_id} session_id={session.session_id}"
                    )
            except Exception as exc:
                print(f"[runtime] Gemini session fallback: {exc}")
        return runtime, runtime.atlas_page, runtime.atlas_context
    except Exception as exc:
        print(f"[runtime] episode isolation fallback: {exc}")
        try:
            runtime.close(logger=_runtime_event_logger)
        except Exception:
            pass
        return None, bootstrap_page, bootstrap_context


def _cleanup_episode_runtime_v2(
    *,
    runtime: Optional[EpisodeRuntime],
    bootstrap_page: Optional[Page],
    bootstrap_context: Any,
    room_url: str,
    cfg: Dict[str, Any],
    reason: str,
) -> tuple[Optional[Page], Any]:
    if runtime is None:
        return bootstrap_page, bootstrap_context

    try:
        from src.solver import chat_only as _chat_only

        _chat_only.unregister_episode_gemini_session(runtime.episode_id)
    except Exception:
        pass
    try:
        runtime.close(logger=_runtime_event_logger)
    except Exception as exc:
        print(f"[runtime] warning: episode runtime close failed: {exc}")

    if bootstrap_page is not None and room_url:
        try:
            _goto_with_retry(
                bootstrap_page,
                room_url,
                wait_until="domcontentloaded",
                timeout_ms=45000,
                cfg=cfg,
                reason=reason,
            )
            bootstrap_page.wait_for_timeout(1200)
        except Exception as exc:
            print(f"[runtime] warning: failed to restore bootstrap room page: {exc}")
    return bootstrap_page, bootstrap_context


def _finalize_current_episode_v2(
    *,
    cfg: Dict[str, Any],
    report: EpisodeReport,
    task_state: Optional[Dict[str, Any]],
    runtime: Optional[EpisodeRuntime],
    bootstrap_page: Optional[Page],
    bootstrap_context: Any,
    room_url: str,
    page: Optional[Page] = None,
    segments: Optional[List[Dict[str, Any]]] = None,
    validation_report: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    validation_tracker: Optional[ValidationTracker] = None,
    reason: str = "room-after-episode-v2",
) -> tuple[Optional[Page], Any]:
    if page is not None:
        report.page_url = str(getattr(page, "url", "") or report.page_url or "")
    if runtime is not None and isinstance(runtime.task_state, dict):
        report.gemini_session_id = str(runtime.task_state.get("gemini_session_id", report.gemini_session_id) or report.gemini_session_id)
        report.request_id = str(runtime.task_state.get("gemini_last_request_id", report.request_id) or report.request_id)
        report.retry_stage = str(runtime.task_state.get("gemini_last_retry_stage", report.retry_stage) or report.retry_stage)
        report.retry_reason = str(runtime.task_state.get("gemini_last_retry_reason", report.retry_reason) or report.retry_reason)
        try:
            report.gemini_latency_ms = int(runtime.task_state.get("gemini_last_latency_ms", report.gemini_latency_ms) or report.gemini_latency_ms or 0)
        except Exception:
            pass
        validation_errors = runtime.task_state.get("gemini_last_validation_errors", [])
        if isinstance(validation_errors, list):
            report.notes.extend(str(item).strip() for item in validation_errors[:5] if str(item).strip())
        if bool(runtime.task_state.get("gemini_session_restarted", False)):
            report.notes.append("gemini session restarted during episode")
    if segments:
        report.segment_count = len(segments)
        report.segment_checksum = build_segment_checksum(segments)
    if isinstance(result, dict):
        report.submit_blocked = bool(result.get("submit_guard_blocked", False))
        submit_status = result.get("submit_status", {}) if isinstance(result.get("submit_status"), dict) else {}
        report.submit_verification_reason = str(
            submit_status.get("submit_verification_reason", report.submit_verification_reason) or report.submit_verification_reason
        ).strip()
    if isinstance(validation_report, dict):
        warnings = [str(item).strip() for item in (validation_report.get("warnings", []) or []) if str(item).strip()]
        errors = [str(item).strip() for item in (validation_report.get("errors", []) or []) if str(item).strip()]
        if any("desync" in item.lower() for item in warnings + errors):
            report.desync_detected = True
        report.notes.extend(warnings[:10])
        report.notes.extend(errors[:10])
    if validation_tracker is not None:
        try:
            final_overlong_indices: List[int] = []
            if segments:
                final_overlong_indices, _ = validation_tracker.overlong_snapshot(segments)
            live_validation_report = validation_tracker.finalize(
                total_segments=len(segments or []),
                final_overlong_count=len(final_overlong_indices),
                final_overlong_indices=final_overlong_indices,
            )
            live_validation_path = validation_tracker.save_report(live_validation_report)
            report.live_validation_report_path = str(live_validation_path)
            if isinstance(task_state, dict):
                task_state["live_validation_report"] = report.live_validation_report_path
            if live_validation_report.failure_summary:
                report.notes.append(live_validation_report.failure_summary)
        except Exception as exc:
            print(f"[validation] warning: failed to save live validation report: {exc}")
            report.notes.append(f"live validation save failed: {exc}")
    _emit_episode_report(
        cfg,
        report,
        task_state=task_state,
        lifecycle_events=runtime.lifecycle_events if runtime is not None else None,
    )
    return _cleanup_episode_runtime_v2(
        runtime=runtime,
        bootstrap_page=bootstrap_page,
        bootstrap_context=bootstrap_context,
        room_url=room_url,
        cfg=cfg,
        reason=reason,
    )


DEFAULT_CONFIG = _solver_config.DEFAULT_CONFIG
_load_selectors_yaml = _solver_config._load_selectors_yaml
_deep_merge = _solver_config._deep_merge
_cfg_get = _solver_config._cfg_get
_task_id_from_url = _artifacts._task_id_from_url
_task_scoped_artifact_paths = _artifacts._task_scoped_artifact_paths
_load_task_state = _artifacts._load_task_state
_save_task_state = _artifacts._save_task_state
_load_cached_segments = _artifacts._load_cached_segments
_save_cached_segments = _artifacts._save_cached_segments
_save_task_text_files = _artifacts._save_task_text_files
_labels_cache_path = _artifacts._labels_cache_path
_load_cached_labels = _artifacts._load_cached_labels
_save_cached_labels = _artifacts._save_cached_labels
_invalidate_cached_labels = _artifacts._invalidate_cached_labels
_clear_episode_state = _artifacts._clear_episode_state
_save_validation_report = _artifacts._save_validation_report
_save_outputs = _artifacts._save_outputs
_capture_debug_artifacts = _artifacts._capture_debug_artifacts
_capture_step_artifacts = _artifacts._capture_step_artifacts
_default_chrome_user_data_dir = _browser_auth._default_chrome_user_data_dir
_looks_like_profile_dir_name = _browser_auth._looks_like_profile_dir_name
_is_direct_profile_path = _browser_auth._is_direct_profile_path
_resolve_atlas_email = _browser_auth._resolve_atlas_email
_detect_chrome_profile_for_email = _browser_auth._detect_chrome_profile_for_email
_count_site_cookies_in_profile = _browser_auth._count_site_cookies_in_profile
_detect_chrome_profile_for_site_cookie = _browser_auth._detect_chrome_profile_for_site_cookie
_otp_provider = _browser_auth._otp_provider
_otp_is_manual = _browser_auth._otp_is_manual
_ensure_parent = _browser_auth._ensure_parent
_is_authenticated_page = _browser_auth._is_authenticated_page
_restore_storage_state = _browser_auth._restore_storage_state
_is_too_many_redirects_error = _browser_auth._is_too_many_redirects_error
_clear_atlas_site_session = _browser_auth._clear_atlas_site_session
_close_chrome_processes = _browser_auth._close_chrome_processes
_prepare_chrome_profile_clone = _browser_auth._prepare_chrome_profile_clone
_decode_mime_header = _browser_auth._decode_mime_header
_message_to_text = _browser_auth._message_to_text
_extract_otp_from_messages = _browser_auth._extract_otp_from_messages
_imap_login_from_cfg = _browser_auth._imap_login_from_cfg
_get_gmail_uid_watermark = _browser_auth._get_gmail_uid_watermark
_extract_mailbox_name_from_list_line = _browser_auth._extract_mailbox_name_from_list_line
_select_imap_mailbox = _browser_auth._select_imap_mailbox
_fetch_otp_gmail_imap = _browser_auth._fetch_otp_gmail_imap
_resolve_otp_code = _browser_auth._resolve_otp_code
_body_has_rate_limit = _browser_auth._body_has_rate_limit
_wait_until_authenticated = _browser_auth._wait_until_authenticated
ensure_logged_in = _browser_auth.ensure_logged_in
_selector_variants = _browser._selector_variants
_goto_with_retry = _browser._goto_with_retry
_any_locator_exists = _browser._any_locator_exists
_first_visible_locator = _browser._first_visible_locator
_safe_locator_click = _browser._safe_locator_click
_safe_fill = _browser._safe_fill
_safe_locator_text = _browser._safe_locator_text
_first_href_from_selector = _browser._first_href_from_selector
_all_task_label_hrefs_from_page = _browser._all_task_label_hrefs_from_page
_first_task_label_href_from_html = _browser._first_task_label_href_from_html
_is_label_page_not_found = _browser._is_label_page_not_found
_is_label_page_internal_error = _browser._is_label_page_internal_error
_try_go_back_from_label_error = _browser._try_go_back_from_label_error
_is_label_page_actionable = _browser._is_label_page_actionable
_is_room_access_disabled = _browser._is_room_access_disabled
_recover_room_access_disabled = _browser._recover_room_access_disabled
_wait_for_any = _browser._wait_for_any

_LAST_RESERVE_REQUEST_TS = 0.0
_LAST_GEMINI_REQUEST_TS = 0.0
_GEMINI_REQUEST_TIMESTAMPS: List[float] = []
_GEMINI_QUOTA_COOLDOWN_UNTIL_TS = 0.0
_GEMINI_FALLBACK_USES = 0
_SCRIPT_BUILD = "2026-03-28.1730"


def _resolve_secret(explicit: str, env_names: List[str]) -> str:
    return _solver_config._resolve_secret(explicit, env_names)


GeminiKeyPool = _solver_config.GeminiKeyPool

_global_solver_key_pool: Optional[GeminiKeyPool] = None

def _get_global_solver_key_pool(
    explicit_key: str,
    fallback_key: str,
    dotenv: Dict[str, str],
    cfg_api_keys: Optional[List[str]] = None,
    rotation_policy: str = "sticky",
) -> GeminiKeyPool:
    global _global_solver_key_pool
    _global_solver_key_pool = _solver_config._get_global_solver_key_pool(
        explicit_key=explicit_key,
        fallback_key=fallback_key,
        dotenv=dotenv,
        cfg_api_keys=cfg_api_keys,
        rotation_policy=rotation_policy,
    )
    return _global_solver_key_pool

def _resolve_gemini_key(explicit: str) -> str:
    return _solver_config._resolve_gemini_key(explicit)


def _resolve_gemini_fallback_key(explicit: str) -> str:
    return _solver_config._resolve_gemini_fallback_key(explicit)


def _looks_like_video_url(url: str) -> bool:
    raw = html.unescape((url or "").strip())
    if not raw:
        return False
    u = raw.lower()
    if u.startswith("blob:"):
        return False
    parsed = urlparse(u)
    path = parsed.path or ""
    if re.search(r"\.(mp4|webm|mov|m4v|m3u8)$", path, flags=re.I):
        return True
    if re.search(r"\.(woff2?|ttf|otf|css|js|map|png|jpe?g|gif|svg|ico)$", path, flags=re.I):
        return False
    return ("video" in path) or ("video" in parsed.query)


def _collect_video_url_candidates(page: Page, cfg: Dict[str, Any]) -> List[str]:
    selectors = _cfg_get(cfg, "atlas.selectors", {})
    video_sel = str(selectors.get("video_element", "video"))
    source_sel = str(selectors.get("video_source", "video source"))
    base_url = page.url

    seen: set[str] = set()
    out: List[str] = []

    def add(raw: str) -> None:
        raw = html.unescape((raw or "").strip())
        if not raw:
            return
        if raw.startswith("blob:"):
            return
        if raw.startswith("//"):
            raw = f"https:{raw}"
        elif raw.startswith("/"):
            raw = urljoin(base_url, raw)
        parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"}:
            return
        norm = parsed._replace(fragment="").geturl().strip()
        if not _looks_like_video_url(norm):
            return
        if norm in seen:
            return
        seen.add(norm)
        out.append(norm)

    # Video/source elements.
    for sel in _selector_variants(video_sel):
        try:
            loc = page.locator(sel)
            for i in range(min(loc.count(), 6)):
                item = loc.nth(i)
                add(item.get_attribute("src") or "")
                add(item.get_attribute("data-src") or "")
                try:
                    cur = item.evaluate("el => (el.currentSrc || '')")
                    if isinstance(cur, str):
                        add(cur)
                except Exception:
                    pass
        except Exception:
            continue

    for sel in _selector_variants(source_sel):
        try:
            loc = page.locator(sel)
            for i in range(min(loc.count(), 10)):
                item = loc.nth(i)
                add(item.get_attribute("src") or "")
        except Exception:
            continue

    # HTML links/resources.
    try:
        html_doc = page.content()
    except Exception:
        html_doc = ""
    for match in re.findall(r'["\']([^"\']+\.(?:mp4|webm|mov|m4v|m3u8)(?:\?[^"\']*)?)["\']', html_doc, flags=re.I):
        add(match)
    for match in re.findall(r'https?://[^\s"\'<>]+', html_doc):
        if _looks_like_video_url(match):
            add(match)

    # Performance resources.
    try:
        entries = page.evaluate("() => performance.getEntriesByType('resource').map(e => e.name)")
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, str) and _looks_like_video_url(item):
                    add(item)
    except Exception:
        pass

    # Prefer URLs with explicit video extensions first.
    out.sort(
        key=lambda u: (
            0 if re.search(r"\.(mp4|webm|mov|m4v|m3u8)(\?|$)", u, flags=re.I) else 1,
            0 if "atlascapture" in u.lower() else 1,
            len(u),
        )
    )
    return out


def _download_video_via_playwright_request(
    page: Page,
    context: Any,
    video_url: str,
    out_path: Path,
    timeout_sec: int,
) -> Path:
    headers = {
        "Accept": "*/*",
        "Referer": page.url,
    }
    try:
        ua = page.evaluate("() => navigator.userAgent")
        if isinstance(ua, str) and ua.strip():
            headers["User-Agent"] = ua.strip()
    except Exception:
        pass

    req_ctx = getattr(context, "request", None)
    if req_ctx is None:
        raise RuntimeError("playwright request context is unavailable")

    resp = req_ctx.get(
        video_url,
        headers=headers,
        timeout=max(15000, int(timeout_sec * 1000)),
        fail_on_status_code=False,
    )
    status = int(resp.status)
    if status not in {200, 206}:
        raise RuntimeError(f"playwright fallback status={status}")

    body = resp.body() or b""
    if not body:
        raise RuntimeError("playwright fallback returned empty body")

    if status == 206:
        cr = str((resp.headers or {}).get("content-range", "")).strip()
        m = re.search(r"/(\d+)$", cr)
        if m:
            try:
                total = int(m.group(1))
            except Exception:
                total = 0
            if total > 0 and len(body) < total:
                raise RuntimeError(
                    f"playwright fallback returned partial body ({len(body)}/{total})"
                )

    _ensure_parent(out_path)
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    part_path.write_bytes(body)
    try:
        out_path.unlink(missing_ok=True)
    except Exception:
        pass
    part_path.replace(out_path)
    return out_path


def _download_video_from_page_context(
    page: Page,
    context: Any,
    video_url: str,
    out_path: Path,
    timeout_sec: int,
    cfg: Optional[Dict[str, Any]] = None,
) -> Path:
    sess = requests.Session()
    headers = {
        "Accept": "*/*",
        "Referer": page.url,
    }
    try:
        ua = page.evaluate("() => navigator.userAgent")
        if isinstance(ua, str) and ua.strip():
            headers["User-Agent"] = ua.strip()
    except Exception:
        pass

    try:
        cookies = context.cookies([video_url]) or context.cookies()
    except Exception:
        cookies = []
    for c in cookies:
        try:
            sess.cookies.set(
                c.get("name", ""),
                c.get("value", ""),
                domain=c.get("domain"),
                path=c.get("path", "/"),
            )
        except Exception:
            continue

    _ensure_parent(out_path)
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    max_retries = max(0, int(_cfg_get(cfg or {}, "gemini.video_download_retries", 5)))
    chunk_bytes = max(64 * 1024, int(_cfg_get(cfg or {}, "gemini.video_download_chunk_bytes", 1024 * 1024)))
    retry_base = max(0.2, float(_cfg_get(cfg or {}, "gemini.video_download_retry_base_sec", 1.2)))
    use_playwright_fallback = bool(
        _cfg_get(cfg or {}, "gemini.video_download_use_playwright_fallback", True)
    )
    last_err: Optional[Exception] = None

    def _content_range_total(content_range: str) -> int:
        m = re.search(r"/(\d+)$", content_range or "")
        if not m:
            return 0
        try:
            return int(m.group(1))
        except Exception:
            return 0

    for attempt in range(max_retries + 1):
        resume_from = 0
        try:
            if part_path.exists():
                resume_from = int(part_path.stat().st_size)
        except Exception:
            resume_from = 0

        req_headers = dict(headers)
        if resume_from > 0:
            req_headers["Range"] = f"bytes={resume_from}-"

        try:
            with sess.get(
                video_url,
                headers=req_headers,
                timeout=(20, timeout_sec),
                stream=True,
                allow_redirects=True,
            ) as resp:
                status = int(resp.status_code)
                if status not in {200, 206}:
                    resp.raise_for_status()

                if resume_from > 0 and status == 200:
                    # Server ignored Range; restart clean download.
                    try:
                        part_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    resume_from = 0

                expected_total = 0
                cr = resp.headers.get("Content-Range", "")
                if cr:
                    expected_total = _content_range_total(cr)
                if expected_total <= 0:
                    cl = resp.headers.get("Content-Length", "")
                    try:
                        content_len = int(cl)
                    except Exception:
                        content_len = 0
                    if content_len > 0:
                        expected_total = resume_from + content_len

                mode = "ab" if (resume_from > 0 and status == 206 and part_path.exists()) else "wb"
                written_this_attempt = 0
                with part_path.open(mode) as f:
                    for chunk in resp.iter_content(chunk_size=chunk_bytes):
                        if not chunk:
                            continue
                        f.write(chunk)
                        written_this_attempt += len(chunk)

                current_size = int(part_path.stat().st_size) if part_path.exists() else 0
                if current_size <= 0:
                    raise RuntimeError("Downloaded video file is empty.")
                if expected_total > 0 and current_size < expected_total:
                    raise RuntimeError(
                        f"Incomplete download ({current_size}/{expected_total} bytes)"
                    )

                try:
                    out_path.unlink(missing_ok=True)
                except Exception:
                    pass
                part_path.replace(out_path)
                return out_path
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                delay = retry_base * (2**attempt)
                try:
                    partial = int(part_path.stat().st_size) if part_path.exists() else 0
                except Exception:
                    partial = 0
                print(
                    f"[video] download retry {attempt + 1}/{max_retries} "
                    f"(partial={partial} bytes) in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            break

    if use_playwright_fallback:
        try:
            return _download_video_via_playwright_request(
                page=page,
                context=context,
                video_url=video_url,
                out_path=out_path,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            if last_err is not None:
                raise RuntimeError(f"{last_err}; playwright fallback failed: {exc}") from exc
            raise RuntimeError(f"playwright fallback failed: {exc}") from exc

    raise RuntimeError(str(last_err) if last_err else "video download failed")


def _is_probably_mp4(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(64)
    except Exception:
        return False
    if len(head) < 12:
        return False
    # Common MP4 signature: box size then 'ftyp' within first 12 bytes.
    return b"ftyp" in head[:16]


def _is_video_decodable(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import cv2  # type: ignore
    except Exception:
        # If OpenCV is unavailable, do not block on decode probing.
        return True

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        probe_positions = [0]
        if frame_count > 2:
            probe_positions.append(max(0, frame_count // 2))
            probe_positions.append(max(0, frame_count - 2))
        for pos in probe_positions:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(pos))
            except Exception:
                pass
            ok, _ = cap.read()
            if not ok:
                return False
        return True
    finally:
        cap.release()


def _probe_video_stream_meta(path: Path) -> Tuple[int, int, float, int]:
    try:
        import cv2  # type: ignore
    except Exception:
        return 0, 0, 0.0, 0

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0, 0.0, 0
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return width, height, fps, frames
    finally:
        cap.release()


def _quality_preserving_scale_candidates(
    scales: List[float],
    src_w: int,
    src_h: int,
    min_width: int,
    min_short_side: int,
) -> List[float]:
    if src_w <= 0 or src_h <= 0:
        return scales

    min_width = max(2, int(min_width))
    min_short_side = max(2, int(min_short_side))
    short_side = min(src_w, src_h)

    width_floor = min_width / float(src_w)
    short_floor = min_short_side / float(short_side) if short_side > 0 else 0.0
    scale_floor = max(0.1, min(1.0, max(width_floor, short_floor)))

    filtered: List[float] = []
    for raw in scales:
        s = max(0.1, min(1.0, float(raw)))
        if s + 1e-6 >= scale_floor:
            filtered.append(s)

    if not filtered:
        filtered = [scale_floor]
    elif all(abs(s - scale_floor) > 1e-3 for s in filtered):
        filtered.append(scale_floor)

    # Keep largest scales first to preserve detail while meeting size target.
    uniq = sorted({round(s, 4) for s in filtered}, reverse=True)
    return [float(s) for s in uniq]


def _extract_reference_frame_inline_parts(
    video_file: Path,
    cfg: Dict[str, Any],
    trigger_video_mb: float,
) -> Tuple[List[Dict[str, Any]], int]:
    enabled = bool(_cfg_get(cfg, "gemini.reference_frames_enabled", True))
    if not enabled or video_file is None or not video_file.exists():
        return [], 0

    always = bool(_cfg_get(cfg, "gemini.reference_frames_always", False))
    trigger_mb = max(0.1, float(_cfg_get(cfg, "gemini.reference_frame_attach_when_video_mb_le", 2.5)))
    if not always and trigger_video_mb > trigger_mb:
        return [], 0

    try:
        import cv2  # type: ignore
    except Exception:
        return [], 0

    frame_count = max(1, int(_cfg_get(cfg, "gemini.reference_frame_count", 2)))
    max_side = max(240, int(_cfg_get(cfg, "gemini.reference_frame_max_side", 960)))
    jpeg_quality = max(50, min(95, int(_cfg_get(cfg, "gemini.reference_frame_jpeg_quality", 82))))
    max_total_bytes = max(64 * 1024, int(float(_cfg_get(cfg, "gemini.reference_frame_max_total_kb", 420)) * 1024))

    raw_positions = _cfg_get(cfg, "gemini.reference_frame_positions", [0.2, 0.55, 0.85])
    pos_list: List[float] = []
    if isinstance(raw_positions, list):
        for raw in raw_positions:
            try:
                v = float(raw)
            except Exception:
                continue
            if 0.0 <= v <= 1.0:
                pos_list.append(v)
    if not pos_list:
        step = 1.0 / float(frame_count + 1)
        pos_list = [step * (i + 1) for i in range(frame_count)]

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return [], 0

    try:
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frames_total <= 0:
            return [], 0

        indices: List[int] = []
        for p in pos_list:
            idx = int(round((frames_total - 1) * max(0.0, min(1.0, p))))
            if idx not in indices:
                indices.append(idx)
            if len(indices) >= frame_count:
                break
        if not indices:
            return [], 0

        parts: List[Dict[str, Any]] = []
        total_bytes = 0
        for idx in indices:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            except Exception:
                pass
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if h <= 0 or w <= 0:
                continue
            largest = max(h, w)
            if largest > max_side:
                scale = max_side / float(largest)
                nw = max(2, int(round(w * scale)))
                nh = max(2, int(round(h * scale)))
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            ok_enc, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if not ok_enc:
                continue
            data = bytes(enc.tobytes())
            if not data:
                continue
            if total_bytes + len(data) > max_total_bytes:
                break
            total_bytes += len(data)
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(data).decode("ascii"),
                    }
                }
            )
        return parts, total_bytes
    finally:
        cap.release()


def _ensure_even(value: int, minimum: int = 2) -> int:
    v = max(int(minimum), int(value))
    return v if v % 2 == 0 else v - 1


def _parse_float_list(value: Any, fallback: List[float]) -> List[float]:
    if isinstance(value, list):
        out: List[float] = []
        for item in value:
            try:
                n = float(item)
                if n > 0:
                    out.append(n)
            except Exception:
                continue
        if out:
            return out
        return list(fallback)
    if isinstance(value, str):
        out = []
        for raw in value.split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                n = float(raw)
                if n > 0:
                    out.append(n)
            except Exception:
                continue
        if out:
            return out
    return list(fallback)


def _opencv_available() -> bool:
    try:
        import cv2  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_ffmpeg_binary() -> Optional[str]:
    local_app_data = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
    user_profile = str(os.environ.get("USERPROFILE", "") or "").strip()
    # Build the candidate list with platform-specific absolute paths first,
    # so they take priority over bare names resolved via shutil.which().
    # On Windows (LOCALAPPDATA set), WinGet/scoop paths come first.
    # On Linux, /usr/bin/ffmpeg and /usr/local/bin/ffmpeg come first.
    # Bare names like "ffmpeg" are tried last as a PATH fallback.
    candidates: list[str] = []
    if local_app_data:
        candidates.extend(
            [
                str(Path(local_app_data) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"),
                str(
                    Path(local_app_data)
                    / "Microsoft"
                    / "WinGet"
                    / "Packages"
                    / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
                    / "ffmpeg-8.0-full_build"
                    / "bin"
                    / "ffmpeg.exe"
                ),
            ]
        )
    if user_profile:
        candidates.append(str(Path(user_profile) / "scoop" / "apps" / "ffmpeg" / "current" / "bin" / "ffmpeg.exe"))
    if not local_app_data and not user_profile:
        candidates.extend(["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"])
    candidates.extend(
        [
            "C:\\ffmpeg\\bin\\ffmpeg.exe",
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
        ]
    )
    candidates.extend(["ffmpeg", "ffmpeg.exe"])
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        try:
            p = Path(candidate)
            if p.exists() and p.is_file():
                return str(p)
        except Exception:
            continue
    return None


def _resolve_ffprobe_binary(ffmpeg_bin: Optional[str] = None) -> Optional[str]:
    if ffmpeg_bin:
        try:
            ffmpeg_path = Path(ffmpeg_bin)
            probe_name = "ffprobe.exe" if ffmpeg_path.suffix.lower() == ".exe" else "ffprobe"
            sibling = ffmpeg_path.with_name(probe_name)
            if sibling.exists() and sibling.is_file():
                return str(sibling)
        except Exception:
            pass
    local_app_data = str(os.environ.get("LOCALAPPDATA", "") or "").strip()
    user_profile = str(os.environ.get("USERPROFILE", "") or "").strip()
    candidates = [
        "ffprobe",
        "ffprobe.exe",
        "/usr/bin/ffprobe",
        "/usr/local/bin/ffprobe",
        "C:\\ffmpeg\\bin\\ffprobe.exe",
    ]
    if local_app_data:
        candidates.extend(
            [
                str(Path(local_app_data) / "Microsoft" / "WinGet" / "Links" / "ffprobe.exe"),
                str(
                    Path(local_app_data)
                    / "Microsoft"
                    / "WinGet"
                    / "Packages"
                    / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
                    / "ffmpeg-8.0-full_build"
                    / "bin"
                    / "ffprobe.exe"
                ),
            ]
        )
    if user_profile:
        candidates.append(str(Path(user_profile) / "scoop" / "apps" / "ffmpeg" / "current" / "bin" / "ffprobe.exe"))
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        try:
            p = Path(candidate)
            if p.exists() and p.is_file():
                return str(p)
        except Exception:
            continue
    return None


def _probe_video_duration_seconds(video_file: Path, ffmpeg_bin: Optional[str] = None) -> float:
    if video_file is None or not video_file.exists():
        return 0.0

    try:
        _, _, fps, frames = _probe_video_stream_meta(video_file)
        if fps > 0 and frames > 0:
            duration = float(frames) / float(fps)
            if duration > 0.2:
                return duration
    except Exception:
        pass

    ffprobe_bin = _resolve_ffprobe_binary(ffmpeg_bin=ffmpeg_bin)
    if not ffprobe_bin:
        return 0.0
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_file),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            return 0.0
        out = (proc.stdout or "").strip().splitlines()
        if not out:
            return 0.0
        duration = float(out[-1].strip())
        if duration > 0.2:
            return duration
    except Exception:
        return 0.0
    return 0.0


def _split_video_for_upload(video_file: Path, cfg: Dict[str, Any]) -> List[Path]:
    if video_file is None or not video_file.exists():
        return []
    if not bool(_cfg_get(cfg, "gemini.split_upload_enabled", True)):
        return []

    size_bytes = int(video_file.stat().st_size)
    size_mb = size_bytes / (1024 * 1024)
    trigger_mb = max(1.0, float(_cfg_get(cfg, "gemini.split_upload_only_if_larger_mb", 14.0)))
    if size_mb <= trigger_mb:
        return []

    chunk_max_mb = max(2.0, float(_cfg_get(cfg, "gemini.split_upload_chunk_max_mb", 6.0)))
    max_chunks = max(2, int(_cfg_get(cfg, "gemini.split_upload_max_chunks", 4)))
    split_count = int(math.ceil(size_mb / chunk_max_mb))
    split_count = max(2, min(max_chunks, split_count))
    if split_count <= 1:
        return []

    ffmpeg_bin = _resolve_ffmpeg_binary()
    if not ffmpeg_bin:
        print("[video] split upload skipped: ffmpeg not available.")
        return []

    duration_sec = _probe_video_duration_seconds(video_file, ffmpeg_bin=ffmpeg_bin)
    if duration_sec <= 0.2:
        print("[video] split upload skipped: could not determine video duration.")
        return []

    stem = video_file.stem
    parent = video_file.parent
    out_files = [parent / f"{stem}_upload_part{i + 1:02d}.mp4" for i in range(split_count)]
    if all(p.exists() and p.stat().st_size > 0 and _is_probably_mp4(p) for p in out_files):
        total_mb = sum(float(p.stat().st_size) for p in out_files) / (1024 * 1024)
        print(
            f"[video] using cached split upload parts: {len(out_files)} parts "
            f"({total_mb:.1f} MB total)."
        )
        return out_files

    # Remove stale parts before generating a fresh set.
    for stale in parent.glob(f"{stem}_upload_part*.mp4"):
        try:
            stale.unlink(missing_ok=True)
        except Exception:
            pass

    chunk_duration = duration_sec / float(split_count)
    use_reencode_on_copy_fail = bool(_cfg_get(cfg, "gemini.split_upload_reencode_on_copy_fail", True))
    print(
        f"[video] splitting upload video into {split_count} parts "
        f"(source={size_mb:.1f} MB, duration={duration_sec:.1f}s)."
    )
    produced: List[Path] = []
    for idx, out_path in enumerate(out_files):
        start_sec = idx * chunk_duration
        if idx == split_count - 1:
            dur_sec = max(0.2, duration_sec - start_sec)
        else:
            dur_sec = max(0.2, chunk_duration)

        cmd_copy = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-t",
            f"{dur_sec:.3f}",
            "-i",
            str(video_file),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        ok = False
        try:
            proc = subprocess.run(
                cmd_copy,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
                timeout=240,
            )
            ok = proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0 and _is_probably_mp4(out_path)
        except Exception:
            ok = False

        if not ok and use_reencode_on_copy_fail:
            cmd_enc = [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start_sec:.3f}",
                "-t",
                f"{dur_sec:.3f}",
                "-i",
                str(video_file),
                "-an",
                "-sn",
                "-dn",
                "-c:v",
                "libx264",
                "-preset",
                "faster",
                "-crf",
                "21",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
            try:
                proc = subprocess.run(
                    cmd_enc,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    text=True,
                    timeout=240,
                )
                ok = proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0 and _is_probably_mp4(out_path)
            except Exception:
                ok = False

        if not ok:
            print(f"[video] split chunk failed at part {idx + 1}; falling back to single-file flow.")
            for p in out_files:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            return []

        produced.append(out_path)
        try:
            part_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"[video] split part {idx + 1}/{split_count}: {out_path.name} ({part_mb:.1f} MB)")
        except Exception:
            pass

    if len(produced) != split_count:
        return []
    return produced


def _segment_chunks(
    segments: List[Dict[str, Any]],
    max_per_chunk: int,
    *,
    max_window_sec: float = 0.0,
) -> List[List[Dict[str, Any]]]:
    if not segments:
        return []
    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_start: Optional[float] = None
    window_limit = max(0.0, float(max_window_sec or 0.0))

    for seg in segments:
        seg_start = _safe_float(seg.get("start_sec"), 0.0)
        seg_end = max(seg_start, _safe_float(seg.get("end_sec"), seg_start))
        if not current:
            current = [seg]
            current_start = seg_start
            continue

        exceeds_count = len(current) >= max(1, int(max_per_chunk))
        exceeds_window = False
        if window_limit > 0.0 and current_start is not None:
            exceeds_window = (seg_end - current_start) > window_limit

        if exceeds_count or exceeds_window:
            chunks.append(current)
            current = [seg]
            current_start = seg_start
            continue

        current.append(seg)

    if current:
        chunks.append(current)
    return chunks


def _extract_video_window(
    src_video: Path,
    out_video: Path,
    start_sec: float,
    end_sec: float,
    ffmpeg_bin: Optional[str] = None,
) -> bool:
    ffmpeg_path = ffmpeg_bin or _resolve_ffmpeg_binary()
    if not ffmpeg_path:
        return False
    if src_video is None or not src_video.exists():
        return False
    if end_sec <= start_sec:
        return False

    duration = max(0.2, float(end_sec - start_sec))
    try:
        _ensure_parent(out_video)
        if out_video.exists():
            out_video.unlink()
    except Exception:
        pass

    copy_cmd = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(src_video),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "copy",
        "-movflags",
        "+faststart",
        str(out_video),
    ]
    try:
        proc = subprocess.run(
            copy_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=240,
        )
        if proc.returncode == 0 and out_video.exists() and out_video.stat().st_size > 0 and _is_probably_mp4(out_video):
            return True
    except Exception:
        pass

    # Fallback re-encode for better cut accuracy when stream-copy fails.
    encode_cmd = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(src_video),
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "22",
        "-movflags",
        "+faststart",
        str(out_video),
    ]
    try:
        proc = subprocess.run(
            encode_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=300,
        )
        return bool(
            proc.returncode == 0
            and out_video.exists()
            and out_video.stat().st_size > 0
            and _is_probably_mp4(out_video)
        )
    except Exception:
        return False


def _transcode_video_ffmpeg(
    src: Path,
    dst: Path,
    scale: float,
    target_fps: float,
    min_width: int,
    ffmpeg_bin: Optional[str] = None,
) -> Tuple[bool, str]:
    ffmpeg_path = ffmpeg_bin or _resolve_ffmpeg_binary()
    if not ffmpeg_path:
        return False, "ffmpeg binary not found"

    # Keep width even and avoid going below min_width to keep decoder compatibility.
    vf = (
        f"scale=max({min_width}\\,trunc(iw*{float(scale):.4f}/2)*2):-2,"
        f"fps={max(1.0, float(target_fps)):.2f}"
    )
    codec_attempts: List[List[str]] = [
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30"],
        ["-c:v", "mpeg4", "-q:v", "10"],
    ]
    last_err = ""
    for codec_opts in codec_attempts:
        try:
            _ensure_parent(dst)
            if dst.exists():
                dst.unlink()
        except Exception:
            pass
        cmd = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            vf,
            "-an",
            "-sn",
            "-dn",
            *codec_opts,
            "-movflags",
            "+faststart",
            str(dst),
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
                timeout=420,
            )
        except Exception as exc:
            last_err = str(exc)
            continue
        if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0 and _is_probably_mp4(dst):
            return True, ""
        stderr_snippet = (proc.stderr or "").strip()
        if stderr_snippet:
            stderr_snippet = stderr_snippet.splitlines()[-1]
        last_err = stderr_snippet or f"ffmpeg exit code {proc.returncode}"
    return False, last_err


def _transcode_video_cv2(
    src: Path,
    dst: Path,
    scale: float,
    target_fps: float,
    min_width: int,
) -> bool:
    try:
        import cv2  # type: ignore
    except Exception:
        return False

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    try:
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_w <= 0 or src_h <= 0:
            return False
        if src_fps <= 0.1 or src_fps > 240:
            src_fps = 24.0

        scaled_w = max(min_width, int(round(src_w * float(scale))))
        scaled_w = _ensure_even(scaled_w, minimum=min_width)
        scaled_h = int(round(src_h * (scaled_w / float(src_w))))
        scaled_h = _ensure_even(scaled_h, minimum=2)

        target_fps = max(1.0, min(float(target_fps), src_fps))
        frame_interval = max(1, int(round(src_fps / target_fps)))
        out_fps = max(1.0, src_fps / frame_interval)

        _ensure_parent(dst)
        if dst.exists():
            dst.unlink()

        writer = None
        for codec in ("mp4v", "avc1", "H264", "XVID"):
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                candidate = cv2.VideoWriter(str(dst), fourcc, out_fps, (scaled_w, scaled_h))
                if candidate.isOpened():
                    writer = candidate
                    break
                candidate.release()
            except Exception:
                continue
        if writer is None:
            return False

        frame_idx = 0
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_interval > 1 and (frame_idx % frame_interval) != 0:
                frame_idx += 1
                continue
            if frame.shape[1] != scaled_w or frame.shape[0] != scaled_h:
                frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
            frame_idx += 1
        writer.release()
        if written <= 0:
            return False
    finally:
        cap.release()

    return dst.exists() and dst.stat().st_size > 0 and _is_probably_mp4(dst)


def _maybe_optimize_video_for_upload(video_file: Path, cfg: Dict[str, Any]) -> Path:
    if video_file is None or not video_file.exists():
        return video_file

    enabled = bool(_cfg_get(cfg, "gemini.optimize_video_for_upload", True))
    if not enabled:
        return video_file

    size_bytes = int(video_file.stat().st_size)
    size_mb = size_bytes / (1024 * 1024)
    trigger_mb = max(1.0, float(_cfg_get(cfg, "gemini.optimize_video_only_if_larger_mb", 8.0)))
    if size_mb <= trigger_mb:
        return video_file

    target_mb = max(1.0, float(_cfg_get(cfg, "gemini.optimize_video_target_mb", 15.0)))
    target_bytes = int(target_mb * 1024 * 1024)
    target_fps = max(1.0, float(_cfg_get(cfg, "gemini.optimize_video_target_fps", 10.0)))
    min_fps = max(1.0, float(_cfg_get(cfg, "gemini.optimize_video_min_fps", 8.0)))
    target_fps = max(min_fps, target_fps)
    min_width = max(160, int(_cfg_get(cfg, "gemini.optimize_video_min_width", 320)))
    min_short_side = max(160, int(_cfg_get(cfg, "gemini.optimize_video_min_short_side", 320)))
    prefer_ffmpeg = bool(_cfg_get(cfg, "gemini.optimize_video_prefer_ffmpeg", True))
    scales = _parse_float_list(
        _cfg_get(cfg, "gemini.optimize_video_scale_candidates", [0.75, 0.6, 0.5, 0.4, 0.33, 0.25, 0.2]),
        [0.75, 0.6, 0.5, 0.4, 0.33, 0.25, 0.2],
    )
    src_w, src_h, src_fps, _ = _probe_video_stream_meta(video_file)
    scales = _quality_preserving_scale_candidates(
        scales=scales,
        src_w=src_w,
        src_h=src_h,
        min_width=min_width,
        min_short_side=min_short_side,
    )

    out_file = video_file.with_name(f"{video_file.stem}_upload_opt.mp4")
    if out_file.exists():
        try:
            out_size = int(out_file.stat().st_size)
            if out_size > 0 and _is_probably_mp4(out_file) and out_size <= target_bytes:
                print(
                    f"[video] using cached optimized upload file: {out_file} "
                    f"({out_size / (1024 * 1024):.1f} MB)"
                )
                return out_file
        except Exception:
            pass

    src_meta_note = ""
    if src_w > 0 and src_h > 0:
        fps_note = f", {src_fps:.1f}fps" if src_fps > 0 else ""
        src_meta_note = f", source={src_w}x{src_h}{fps_note}"
    print(
        f"[video] optimizing video for upload: {video_file.name} "
        f"({size_mb:.1f} MB -> target <= {target_mb:.1f} MB{src_meta_note})"
    )
    cv2_available = _opencv_available()
    ffmpeg_bin = _resolve_ffmpeg_binary()
    if not cv2_available and ffmpeg_bin:
        print(f"[video] OpenCV unavailable; using ffmpeg optimizer backend: {ffmpeg_bin}")
    elif not cv2_available and not ffmpeg_bin:
        print("[video] OpenCV and ffmpeg are unavailable; cannot optimize upload video.")
    elif prefer_ffmpeg and ffmpeg_bin and cv2_available:
        print(f"[video] preferring ffmpeg optimizer backend: {ffmpeg_bin}")
    candidates: List[Path] = []
    best_path: Optional[Path] = None
    best_size = size_bytes
    ffmpeg_last_error = ""

    for scale in scales:
        scale = max(0.1, min(1.0, float(scale)))
        suffix = int(round(scale * 100))
        cand = video_file.with_name(f"{video_file.stem}_upload_opt_s{suffix}.mp4")
        ok = False
        backend_used: Optional[str] = None
        backend_order: List[str] = []
        if prefer_ffmpeg and ffmpeg_bin:
            backend_order.append("ffmpeg")
        if cv2_available:
            backend_order.append("cv2")
        if ffmpeg_bin and "ffmpeg" not in backend_order:
            backend_order.append("ffmpeg")

        for backend_name in backend_order:
            if backend_name == "cv2":
                try:
                    ok = _transcode_video_cv2(
                        src=video_file,
                        dst=cand,
                        scale=scale,
                        target_fps=target_fps,
                        min_width=min_width,
                    )
                except Exception:
                    ok = False
                if ok:
                    backend_used = "cv2"
                    break
                continue

            try:
                ok, ffmpeg_err = _transcode_video_ffmpeg(
                    src=video_file,
                    dst=cand,
                    scale=scale,
                    target_fps=target_fps,
                    min_width=min_width,
                    ffmpeg_bin=ffmpeg_bin,
                )
                if ok:
                    backend_used = "ffmpeg"
                    break
                if ffmpeg_err:
                    ffmpeg_last_error = ffmpeg_err
            except Exception as exc:
                ffmpeg_last_error = str(exc)
                ok = False
        if not ok:
            continue
        candidates.append(cand)
        try:
            cand_size = int(cand.stat().st_size)
        except Exception:
            continue
        backend_note = f" ({backend_used})" if backend_used else ""
        print(f"[video] optimized candidate scale={scale:.2f}: {cand_size / (1024 * 1024):.1f} MB{backend_note}")
        if cand_size < best_size:
            best_size = cand_size
            best_path = cand
        if cand_size <= target_bytes:
            break

    if best_path is None:
        if ffmpeg_last_error:
            print(f"[video] ffmpeg optimizer failed: {ffmpeg_last_error}")
        print("[video] upload optimization not available; using original video.")
        return video_file

    try:
        if out_file.exists():
            out_file.unlink()
        best_path.replace(out_file)
    except Exception:
        out_file = best_path

    for cand in candidates:
        if cand == out_file:
            continue
        try:
            cand.unlink(missing_ok=True)
        except Exception:
            continue

    out_size = int(out_file.stat().st_size) if out_file.exists() else size_bytes
    if out_size >= size_bytes:
        print("[video] optimization did not reduce size enough; using original video.")
        return video_file
    if out_size > target_bytes:
        print(
            f"[video] optimized upload video remains above target: "
            f"{out_size / (1024 * 1024):.1f} MB > {target_mb:.1f} MB "
            "(quality-preserving floor/backends prevented further reduction)."
        )
    print(
        f"[video] optimized upload video ready: {out_file} "
        f"({out_size / (1024 * 1024):.1f} MB)"
    )
    return out_file


def _prepare_video_for_gemini(
    page: Page,
    context: Any,
    cfg: Dict[str, Any],
    task_id: str = "",
) -> Optional[Path]:
    attach_video = bool(_cfg_get(cfg, "gemini.attach_video", True))
    if not attach_video:
        return None

    out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    use_task_scoped = bool(_cfg_get(cfg, "run.use_task_scoped_artifacts", True))
    if use_task_scoped and task_id:
        video_name = f"video_{task_id}.mp4"
    else:
        video_name = str(_cfg_get(cfg, "run.video_dump", "atlas_task_video.mp4"))
    timeout_sec = int(_cfg_get(cfg, "gemini.video_download_timeout_sec", 180))
    require_video = bool(_cfg_get(cfg, "gemini.require_video", False))
    min_video_bytes = int(_cfg_get(cfg, "gemini.min_video_bytes", 500000))
    validate_video_decode = bool(_cfg_get(cfg, "gemini.validate_video_decode", True))
    scan_attempts = max(1, int(_cfg_get(cfg, "gemini.video_candidate_scan_attempts", 4)))
    scan_wait_ms = max(200, int(_cfg_get(cfg, "gemini.video_candidate_scan_wait_ms", 1200)))
    resume_from_artifacts = bool(_cfg_get(cfg, "run.resume_from_artifacts", True))

    primary_target = out_dir / video_name
    if resume_from_artifacts and primary_target.exists():
        try:
            size_bytes = primary_target.stat().st_size
            if size_bytes >= min_video_bytes and _is_probably_mp4(primary_target):
                if validate_video_decode and not _is_video_decodable(primary_target):
                    print(f"[video] cached file looks corrupted; re-downloading: {primary_target}")
                    try:
                        primary_target.unlink(missing_ok=True)
                    except Exception:
                        pass
                else:
                    size_mb = size_bytes / (1024 * 1024)
                    print(f"[video] reusing existing file: {primary_target} ({size_mb:.1f} MB)")
                    return primary_target
        except Exception:
            pass

    def _nudge_video_network() -> None:
        try:
            page.evaluate(
                """() => {
                    const v = document.querySelector('video');
                    if (!v) return;
                    try { v.muted = true; v.play(); } catch (e) {}
                }"""
            )
            page.wait_for_timeout(900)
            page.evaluate(
                """() => {
                    const v = document.querySelector('video');
                    if (!v) return;
                    try {
                        if (Number.isFinite(v.currentTime)) {
                            v.currentTime = Math.max(0, Number(v.currentTime || 0) + 0.05);
                        }
                        v.pause();
                    } catch (e) {}
                }"""
            )
        except Exception:
            pass

    network_seen: set[str] = set()
    network_candidates: List[str] = []

    def _remember_network_video_url(raw_url: str, content_type: str = "") -> None:
        try:
            raw = html.unescape((raw_url or "").strip())
            if not raw:
                return
            low_ct = (content_type or "").lower()
            if "video" not in low_ct and not _looks_like_video_url(raw):
                return
            if raw in network_seen:
                return
            network_seen.add(raw)
            network_candidates.append(raw)
        except Exception:
            return

    response_listener = None
    try:
        def _on_response(resp: Any) -> None:
            try:
                headers = resp.headers or {}
            except Exception:
                headers = {}
            try:
                content_type = str(headers.get("content-type", "") or "")
            except Exception:
                content_type = ""
            try:
                _remember_network_video_url(str(resp.url or ""), content_type=content_type)
            except Exception:
                return
        response_listener = _on_response
        page.on("response", response_listener)
    except Exception:
        response_listener = None

    def _rank_video_url(url: str) -> Tuple[int, int, int]:
        low = (url or "").lower()
        return (
            0 if re.search(r"\.(mp4|webm|mov|m4v|m3u8)(\?|$)", low, flags=re.I) else 1,
            0 if "atlascapture" in low or "cloudflarestorage.com" in low else 1,
            len(url or ""),
        )

    page.wait_for_timeout(1500)
    _dismiss_blocking_modals(page)
    candidates: List[str] = []
    for scan_idx in range(scan_attempts):
        if scan_idx > 0:
            page.wait_for_timeout(scan_wait_ms)
            _dismiss_blocking_modals(page)
        _nudge_video_network()
        candidates = _collect_video_url_candidates(page, cfg)
        for from_net in network_candidates:
            if from_net not in candidates:
                candidates.append(from_net)
        candidates.sort(key=_rank_video_url)
        if candidates:
            if scan_idx > 0:
                print(f"[video] candidate urls resolved after retry {scan_idx + 1}/{scan_attempts}.")
            break
        if scan_idx < scan_attempts - 1:
            print(f"[video] no candidate urls yet ({scan_idx + 1}/{scan_attempts}); retrying...")

    if response_listener is not None:
        try:
            page.remove_listener("response", response_listener)
        except Exception:
            pass

    print(f"[video] candidate urls found: {len(candidates)}")
    for u in candidates[:5]:
        print(f"[video] candidate: {u}")

    last_err: Optional[Exception] = None
    for idx, url in enumerate(candidates[:20], start=1):
        target = primary_target
        if idx > 1:
            stem = Path(video_name).stem
            suffix = Path(video_name).suffix or ".mp4"
            target = out_dir / f"{stem}_{idx}{suffix}"
        try:
            _download_video_from_page_context(
                page=page,
                context=context,
                video_url=url,
                out_path=target,
                timeout_sec=timeout_sec,
                cfg=cfg,
            )
            size_bytes = target.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            if size_bytes < min_video_bytes:
                print(f"[video] skip candidate (too small {size_bytes} bytes): {url}")
                continue
            if not _is_probably_mp4(target):
                print(f"[video] skip candidate (not mp4 signature): {url}")
                continue
            if validate_video_decode and not _is_video_decodable(target):
                print(f"[video] skip candidate (decode check failed): {url}")
                continue
            print(f"[video] downloaded: {target} ({size_mb:.1f} MB)")
            return target
        except Exception as exc:
            last_err = exc
            continue

    if require_video:
        if last_err is None:
            last_err = RuntimeError("no candidate video URLs discovered")
        raise RuntimeError(f"Could not download task video from page. Last error: {last_err}")
    print("[video] no downloadable video found; proceeding with text-only prompt.")
    return None


def _dismiss_blocking_modals(page: Page, cfg: Optional[Dict[str, Any]] = None) -> None:
    modal_buttons = [
        'button:has-text("I Understand")',
        'button:has-text("Understand")',
        'button:has-text("OK")',
        'button:has-text("Okay")',
        'button:has-text("Got It")',
        'button:has-text("Accept")',
        'button:has-text("Dismiss")',
        'text=/I\\s*Understand/i',
        'text=/\\bunderstand\\b/i',
        'text=/\\bok\\b/i',
        'text=/\\bokay\\b/i',
        '[role="button"]:has-text("I Understand")',
        '[role="button"]:has-text("Understand")',
        '[role="button"]:has-text("OK")',
        '[role="button"]:has-text("Okay")',
        'button:has-text("Close")',
        'text=/\\bClose\\b/i',
        'button:has-text("Got it")',
        'text=/Got\\s*it/i',
        'button:has-text("Continue")',
        'text=/\\bContinue\\b/i',
    ]
    passes = max(1, int(_cfg_get(cfg, "run.modal_dismiss_passes", 2) if cfg else 2))
    click_timeout_ms = max(
        50,
        int(_cfg_get(cfg, "run.modal_dismiss_timeout_ms", 120) if cfg else 120),
    )
    post_click_wait_ms = max(
        0,
        int(_cfg_get(cfg, "run.modal_dismiss_post_click_wait_ms", 180) if cfg else 180),
    )

    for _ in range(passes):
        clicked_any = False
        seen_any = False
        for sel in modal_buttons:
            loc = _first_visible_locator(page, sel, timeout_ms=click_timeout_ms)
            if loc is None:
                continue
            seen_any = True
            try:
                loc.click(timeout=click_timeout_ms, force=True)
                clicked_any = True
            except Exception:
                continue
        if not clicked_any:
            # Fallback JS click by visible text content.
            try:
                clicked_any = bool(
                    page.evaluate(
                        """() => {
                            const nodes = Array.from(document.querySelectorAll(
                                '[role="dialog"] button,[role="dialog"] [role="button"],' +
                                '[role="dialog"] a,[role="dialog"] div,' +
                                'button,[role="button"],a,div'
                            ));
                            for (const n of nodes) {
                                const t = (n.innerText || n.textContent || '').trim().toLowerCase();
                                if (!t) continue;
                                if (
                                    t.includes('understand') ||
                                    t === 'ok' ||
                                    t === 'okay' ||
                                    t === 'close' ||
                                    t.includes('got it') ||
                                    t === 'continue' ||
                                    t === 'accept' ||
                                    t === 'dismiss'
                                ) {
                                    n.click();
                                    return true;
                                }
                            }
                            return false;
                        }"""
                    )
                )
            except Exception:
                clicked_any = False
        if clicked_any:
            if post_click_wait_ms > 0:
                page.wait_for_timeout(post_click_wait_ms)
            try:
                body = (page.inner_text("body") or "").lower()
            except Exception:
                body = ""
            if (
                "probation period" in body
                or "t3 trainee" in body
                or "welcome to the t3 trainee program" in body
            ):
                try:
                    page.evaluate(
                        """() => {
                            const nodes = Array.from(document.querySelectorAll('[role="dialog"] button,[role="dialog"] [role="button"],button,[role="button"]'));
                            for (const n of nodes) {
                                const t = (n.innerText || n.textContent || '').trim().toLowerCase();
                                if (t.includes('understand') || t === 'continue') {
                                    n.click();
                                }
                            }
                        }"""
                    )
                except Exception:
                    pass
                page.wait_for_timeout(max(post_click_wait_ms, 450))
        else:
            if not seen_any:
                break
            break


def _dismiss_blocking_side_panel(page: Page, cfg: Dict[str, Any], aggressive: bool = False) -> bool:
    panel_sel = str(
        _cfg_get(
            cfg,
            "atlas.selectors.blocking_side_panel",
            'div[class*="fixed"][class*="right-4"][class*="z-50"][class*="slide-in-from-right"] || '
            'div[class*="fixed"][class*="right-4"][class*="z-50"][class*="shadow-2xl"]',
        )
    )
    close_sel = str(
        _cfg_get(
            cfg,
            "atlas.selectors.blocking_side_panel_close",
            'button:has-text("Close") || button:has-text("Dismiss") || button:has-text("Done") || '
            'button:has-text("Cancel") || [role="button"]:has-text("Close") || '
            'button[aria-label*="close" i] || button[title*="close" i]',
        )
    )
    changed = False
    panel_variants = _selector_variants(panel_sel)
    close_variants = _selector_variants(close_sel)

    for panel_variant in panel_variants:
        try:
            panel_loc = page.locator(panel_variant)
            count = min(panel_loc.count(), 4)
        except Exception:
            continue
        for i in range(count):
            panel = panel_loc.nth(i)
            try:
                if not panel.is_visible():
                    continue
            except Exception:
                continue
            for close_variant in close_variants:
                try:
                    btn = panel.locator(close_variant).first
                    if btn.count() <= 0 or not btn.is_visible():
                        continue
                    btn.click(timeout=700)
                    changed = True
                except Exception:
                    continue

    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(80)
    except Exception:
        pass

    if changed:
        page.wait_for_timeout(150)
        return True

    panel_present = False
    for panel_variant in panel_variants:
        try:
            loc = page.locator(panel_variant)
            if loc.count() <= 0:
                continue
            scan = min(loc.count(), 3)
            for i in range(scan):
                if loc.nth(i).is_visible():
                    panel_present = True
                    break
            if panel_present:
                break
        except Exception:
            continue

    if aggressive or panel_present:
        try:
            hidden = int(
                page.evaluate(
                    """() => {
                        let hidden = 0;
                        const nodes = Array.from(document.querySelectorAll('div,aside,section'));
                        for (const el of nodes) {
                            if (!el || typeof el.className !== 'string') continue;
                            const cls = el.className;
                            if (!cls.includes('fixed') || !cls.includes('right-4') || !cls.includes('z-50')) continue;
                            if (!(cls.includes('slide-in-from-right') || cls.includes('shadow-2xl'))) continue;
                            const style = window.getComputedStyle(el);
                            if ((style.position || '') !== 'fixed') continue;
                            if (style.display === 'none' || style.visibility === 'hidden') continue;
                            const rect = el.getBoundingClientRect();
                            if (rect.width < 260 || rect.width > 520 || rect.height < 120) continue;
                            if (rect.left < window.innerWidth * 0.45) continue;
                            el.setAttribute('data-codex-hidden-overlay', '1');
                            el.style.pointerEvents = 'none';
                            el.style.display = 'none';
                            hidden += 1;
                        }
                        return hidden;
                    }"""
                )
            )
            if hidden > 0:
                print(f"[atlas] neutralized blocking side panel(s): {hidden}")
                return True
        except Exception:
            pass
    return changed


def _click_segment_row_with_recovery(page: Page, rows: Locator, idx: int, cfg: Dict[str, Any]) -> None:
    last_exc: Exception | None = None
    for attempt in range(4):
        row = rows.nth(idx - 1)
        try:
            _dismiss_blocking_side_panel(page, cfg, aggressive=True)
            row.scroll_into_view_if_needed()
            row.click(timeout=2200, no_wait_after=True)
            return
        except Exception as exc:
            last_exc = exc
            _dismiss_blocking_modals(page)
            _dismiss_blocking_side_panel(page, cfg, aggressive=(attempt >= 1))
            try:
                row = rows.nth(idx - 1)
                row.click(timeout=1200, force=True, no_wait_after=True)
                return
            except Exception as force_exc:
                last_exc = force_exc
                try:
                    row.evaluate(
                        """(el) => {
                            if (!el) return false;
                            const evt = { bubbles: true, cancelable: true, view: window };
                            el.dispatchEvent(new MouseEvent('mousedown', evt));
                            el.dispatchEvent(new MouseEvent('mouseup', evt));
                            el.dispatchEvent(new MouseEvent('click', evt));
                            if (typeof el.click === 'function') el.click();
                            return true;
                        }"""
                    )
                    return
                except Exception as js_exc:
                    last_exc = js_exc
                page.wait_for_timeout(120 + attempt * 120)
    raise RuntimeError(str(last_exc) if last_exc else "failed to focus segment row")


_respect_reserve_cooldown = _browser._respect_reserve_cooldown
_respect_reserve_min_interval = _browser._respect_reserve_min_interval
_mark_reserve_request = _browser._mark_reserve_request
_click_reserve_button_dynamic = _browser._click_reserve_button_dynamic
_extract_wait_seconds_from_page = _browser._extract_wait_seconds_from_page
_reserve_rate_limited = _browser._reserve_rate_limited
_room_has_no_reserved_episodes = _browser._room_has_no_reserved_episodes
_release_all_reserved_episodes = _browser._release_all_reserved_episodes


def _respect_episode_delay(cfg: Dict[str, Any]) -> None:
    min_delay = float(_cfg_get(cfg, "run.min_delay_between_episodes_sec", 0.0) or 0.0)
    max_delay = float(_cfg_get(cfg, "run.max_delay_between_episodes_sec", 0.0) or 0.0)
    if max_delay < min_delay:
        min_delay, max_delay = max_delay, min_delay
    min_delay = max(0.0, min_delay)
    max_delay = max(0.0, max_delay)
    if max_delay <= 0:
        return
    delay = min_delay if max_delay == min_delay else random.uniform(min_delay, max_delay)
    print(f"[run] waiting {delay:.1f}s before next episode.")
    time.sleep(delay)


def _respect_major_step_pause(
    cfg: Dict[str, Any],
    step_name: str,
    *,
    heartbeat: Optional[Callable[[], None]] = None,
) -> None:
    if not bool(_cfg_get(cfg, "run.major_step_pause_enabled", False)):
        return
    min_delay = float(_cfg_get(cfg, "run.major_step_pause_min_sec", 0.0) or 0.0)
    max_delay = float(_cfg_get(cfg, "run.major_step_pause_max_sec", 0.0) or 0.0)
    if max_delay < min_delay:
        min_delay, max_delay = max_delay, min_delay
    min_delay = max(0.0, min_delay)
    max_delay = max(0.0, max_delay)
    if max_delay <= 0.0:
        return
    delay = min_delay if max_delay == min_delay else random.uniform(min_delay, max_delay)
    label = str(step_name or "next step").strip() or "next step"
    print(f"[run] pacing pause before {label}: {delay:.1f}s")
    _sleep_with_shutdown_heartbeat(
        delay,
        heartbeat_sec=min(10.0, max(2.0, delay / 4.0)),
        on_heartbeat=heartbeat,
    )


def _compute_backoff_delay(cfg: Dict[str, Any], attempt: int) -> float:
    base_delay = max(0.2, float(_cfg_get(cfg, "gemini.retry_base_delay_sec", 2.0)))
    jitter_max = max(0.0, float(_cfg_get(cfg, "gemini.retry_jitter_sec", 0.8)))
    max_backoff = max(base_delay, float(_cfg_get(cfg, "gemini.max_backoff_sec", 30.0)))
    delay = min(max_backoff, base_delay * (2**attempt))
    if jitter_max > 0:
        delay += random.uniform(0.0, jitter_max)
    return delay


_extract_retry_seconds_from_text = _gemini._extract_retry_seconds_from_text
_extract_retry_seconds_from_response = _gemini._extract_retry_seconds_from_response
_set_gemini_quota_cooldown = _gemini._set_gemini_quota_cooldown
_respect_gemini_quota_cooldown = _gemini._respect_gemini_quota_cooldown
_respect_gemini_rate_limit = _gemini._respect_gemini_rate_limit
_is_non_retriable_gemini_error = _gemini._is_non_retriable_gemini_error


def _ensure_loop_off(page: Page, cfg: Dict[str, Any]) -> None:
    loop_sel = str(_cfg_get(cfg, "atlas.selectors.loop_toggle_button", "")).strip()
    if loop_sel:
        loop_loc = _first_visible_locator(page, loop_sel, timeout_ms=2200)
        if loop_loc is not None:
            try:
                txt = (_safe_locator_text(loop_loc, timeout_ms=700) or "").lower()
                title = (loop_loc.get_attribute("title") or "").lower()
                classes = (loop_loc.get_attribute("class") or "").lower()
                aria_pressed = (loop_loc.get_attribute("aria-pressed") or "").lower()
                should_toggle = False
                if "loop on" in txt:
                    should_toggle = True
                elif "toggle segment loop" in title and ("bg-primary" in classes or aria_pressed == "true"):
                    should_toggle = True
                if should_toggle:
                    loop_loc.click()
                    print("[video] loop toggled OFF.")
            except Exception:
                pass
    try:
        page.evaluate(
            """() => {
                const v = document.querySelector('video');
                if (v) v.loop = false;
            }"""
        )
    except Exception:
        pass


def _play_full_video_once(page: Page, cfg: Dict[str, Any]) -> None:
    if not bool(_cfg_get(cfg, "run.play_full_video_before_labeling", False)):
        return
    max_wait_sec = max(10, int(_cfg_get(cfg, "run.play_full_video_max_wait_sec", 900)))
    _ensure_loop_off(page, cfg)
    try:
        st = page.evaluate(
            """() => {
                const v = document.querySelector('video');
                if (!v) return null;
                try { v.loop = false; } catch (e) {}
                try { v.muted = true; } catch (e) {}
                try { v.playbackRate = 1; } catch (e) {}
                try { v.play(); } catch (e) {}
                return {
                    current: Number(v.currentTime || 0),
                    duration: Number(v.duration || 0)
                };
            }"""
        )
    except Exception:
        st = None
    if not st:
        print("[video] video element not found; skipping full-video playback step.")
        return

    current = float(st.get("current", 0) or 0)
    duration = float(st.get("duration", 0) or 0)
    if duration <= 0:
        print("[video] unknown duration; skipping full-video playback wait.")
        return

    wait_budget = min(max_wait_sec, max(5, int(duration - current + 3)))
    print(f"[video] playing video to end (duration={duration:.1f}s, wait_budget={wait_budget}s).")
    start = time.time()
    last_log = -999.0
    while time.time() - start < wait_budget:
        try:
            state = page.evaluate(
                """() => {
                    const v = document.querySelector('video');
                    if (!v) return null;
                    return {
                        ended: !!v.ended,
                        current: Number(v.currentTime || 0),
                        duration: Number(v.duration || 0)
                    };
                }"""
            )
        except Exception:
            state = None
        if not state:
            break
        cur = float(state.get("current", 0) or 0)
        dur = float(state.get("duration", 0) or 0)
        ended = bool(state.get("ended", False))
        if ended or (dur > 0 and cur >= dur - 0.2):
            print(f"[video] playback reached end at {cur:.1f}/{dur:.1f}s.")
            break
        elapsed = time.time() - start
        if elapsed - last_log >= 15:
            last_log = elapsed
            print(f"[video] playback progress: {cur:.1f}/{dur:.1f}s")
        page.wait_for_timeout(1000)
    try:
        page.evaluate(
            """() => {
                const v = document.querySelector('video');
                if (v) v.pause();
            }"""
        )
    except Exception:
        pass
    _ensure_loop_off(page, cfg)


def _normalize_target_task_urls(raw: Any) -> List[str]:
    out: List[str] = []
    if not raw:
        return out

    if isinstance(raw, list):
        candidates = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return out
        candidates = [part for part in re.split(r"[\r\n,;]+", text) if str(part or "").strip()]

    for item in candidates:
        value = str(item or "").strip()
        if not value:
            continue
        if re.fullmatch(r"[A-Za-z0-9]{12,}", value):
            value = f"https://audit.atlascapture.io/tasks/room/normal/label/{value}"
        elif value.startswith("/"):
            value = urljoin("https://audit.atlascapture.io", value)
        elif not value.startswith("http"):
            continue
        task_id = _task_id_from_url(value)
        if not task_id:
            continue
        normalized = f"https://audit.atlascapture.io/tasks/room/normal/label/{task_id}"
        if normalized not in out:
            out.append(normalized)
    return out


goto_task_room = _browser.goto_task_room


def _parse_mmss_to_seconds(token: str) -> float:
    token = token.strip()
    if not token:
        return 0.0
    if ":" not in token:
        try:
            return float(token)
        except ValueError:
            return 0.0
    left, right = token.split(":", 1)
    try:
        return int(left) * 60 + float(right)
    except ValueError:
        return 0.0


def _extract_start_end_from_text(text: str) -> Tuple[float, float]:
    text = (text or "").replace("\u2192", "->").replace("\u2013", "-")
    matches = re.findall(r"\b\d+:\d{2}(?:\.\d+)?\b", text)
    if len(matches) >= 2:
        return _parse_mmss_to_seconds(matches[0]), _parse_mmss_to_seconds(matches[1])
    if len(matches) == 1:
        start = _parse_mmss_to_seconds(matches[0])
        # Atlas may show "(6.0s)" duration while end timestamp isn't directly extracted.
        dur_match = re.search(r"\((\d+(?:\.\d+)?)\s*s\)", text, flags=re.IGNORECASE)
        if dur_match:
            try:
                dur_sec = float(dur_match.group(1))
                if dur_sec > 0:
                    return start, start + dur_sec
            except ValueError:
                pass
    return 0.0, 0.0


def _resolve_rows_locator(
    page: Page,
    rows_selector: str,
    sample_size: int = 8,
    row_text_timeout_ms: int = 350,
) -> Tuple[str, Locator]:
    best_sel = ""
    best_score = -1
    best_count = 0

    for candidate in _selector_variants(rows_selector):
        try:
            loc = page.locator(candidate)
            count = loc.count()
            if count <= 0:
                continue
            sample = min(count, max(1, sample_size))
            ts_hits = 0
            for i in range(sample):
                text = _safe_locator_text(loc.nth(i), timeout_ms=max(120, row_text_timeout_ms))
                if re.search(r"\b\d+:\d{2}(?:\.\d+)?\b", text):
                    ts_hits += 1
            score = ts_hits * 10 + min(count, 50)
            if score > best_score:
                best_score = score
                best_count = count
                best_sel = candidate
        except Exception:
            continue

    if not best_sel:
        diagnostics: List[str] = []
        diagnostics.append("No segment rows found. Candidate selector counts:")
        for candidate in _selector_variants(rows_selector):
            try:
                c = page.locator(candidate).count()
            except Exception:
                c = -1
            diagnostics.append(f"  - {candidate} => {c}")
        try:
            body = page.inner_text("body")
            body_snippet = (body or "")[:1200].replace("\n", " | ")
            diagnostics.append(f"Body snippet: {body_snippet}")
        except Exception:
            pass
        raise RuntimeError("\n".join(diagnostics))
    print(f"[atlas] using segment rows selector: {best_sel} (count={best_count})")
    return best_sel, page.locator(best_sel)


def _first_text_from_row(row: Locator, selector: str) -> str:
    for candidate in _selector_variants(selector):
        try:
            text = _safe_locator_text(row.locator(candidate).first, timeout_ms=700)
            if text:
                return text
        except Exception:
            continue
    return ""


def _resolve_row_child_selector(
    rows: Locator,
    selector: str,
    *,
    sample_size: int = 4,
    timeout_ms: int = 250,
    require_timestamp: bool = False,
) -> str:
    best_sel = ""
    best_hits = 0
    row_count = min(rows.count(), max(1, sample_size))
    for candidate in _selector_variants(selector):
        hits = 0
        for i in range(row_count):
            try:
                text = _safe_locator_text(
                    rows.nth(i).locator(candidate).first,
                    timeout_ms=timeout_ms,
                )
            except Exception:
                text = ""
            if require_timestamp:
                if re.search(r"\b\d+:\d{2}(?:\.\d+)?\b", text or ""):
                    hits += 1
            elif text:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_sel = candidate
    return best_sel if best_hits > 0 else ""


def extract_segments(
    page: Page,
    cfg: Dict[str, Any],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    max_segments = int(_cfg_get(cfg, "run.max_segments", 0) or 0)
    resolve_attempts = max(1, int(_cfg_get(cfg, "run.segment_resolve_attempts", 24)))
    resolve_retry_ms = max(150, int(_cfg_get(cfg, "run.segment_resolve_retry_ms", 800)))
    resolve_sample_size = max(1, int(_cfg_get(cfg, "run.segment_resolve_sample_size", 8)))
    resolve_row_text_timeout_ms = max(100, int(_cfg_get(cfg, "run.segment_resolve_row_text_timeout_ms", 350)))
    extract_progress_every = max(0, int(_cfg_get(cfg, "run.segment_extract_progress_every", 12) or 12))
    row_scroll_timeout_ms = max(250, int(_cfg_get(cfg, "run.segment_row_scroll_timeout_ms", 1200)))
    rows_sel = str(_cfg_get(cfg, "atlas.selectors.segment_rows", ""))
    label_sel = str(_cfg_get(cfg, "atlas.selectors.segment_label", ""))
    start_sel = str(_cfg_get(cfg, "atlas.selectors.segment_start", ""))
    end_sel = str(_cfg_get(cfg, "atlas.selectors.segment_end", ""))
    heartbeat = _ACTIVE_HEARTBEAT_CALLBACK if callable(_ACTIVE_HEARTBEAT_CALLBACK) else None

    def _touch_heartbeat() -> None:
        if not callable(heartbeat):
            return
        try:
            heartbeat()
        except Exception:
            pass

    last_error: Exception | None = None
    rows = None
    for attempt in range(1, resolve_attempts + 1):
        _touch_heartbeat()
        _dismiss_blocking_modals(page)
        try:
            _, rows = _resolve_rows_locator(
                page,
                rows_sel,
                sample_size=resolve_sample_size,
                row_text_timeout_ms=resolve_row_text_timeout_ms,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt == 1 or attempt % 3 == 0 or attempt == resolve_attempts:
                msg = str(exc).strip().replace("\n", " | ")
                if len(msg) > 220:
                    msg = msg[:220] + "..."
                print(f"[atlas] segment rows not ready (attempt {attempt}/{resolve_attempts}): {msg}")
            page.wait_for_timeout(resolve_retry_ms)
    if rows is None:
        if last_error:
            raise last_error
        raise RuntimeError("Could not resolve segment rows.")

    count = rows.count()
    limit = count if max_segments <= 0 else min(count, max_segments)
    resolved_label_sel = _resolve_row_child_selector(
        rows,
        label_sel,
        sample_size=min(limit, 4),
        timeout_ms=resolve_row_text_timeout_ms,
    )
    resolved_start_sel = _resolve_row_child_selector(
        rows,
        start_sel,
        sample_size=min(limit, 4),
        timeout_ms=resolve_row_text_timeout_ms,
        require_timestamp=True,
    )
    resolved_end_sel = _resolve_row_child_selector(
        rows,
        end_sel,
        sample_size=min(limit, 4),
        timeout_ms=resolve_row_text_timeout_ms,
        require_timestamp=True,
    )
    if label_sel and not resolved_label_sel:
        print("[atlas] segment label child selector missed sample rows; using row-text fallback.")
    if start_sel and not resolved_start_sel:
        print("[atlas] segment start child selector missed sample rows; using row-text fallback.")
    if end_sel and not resolved_end_sel:
        print("[atlas] segment end child selector missed sample rows; using row-text fallback.")

    items: List[Dict[str, Any]] = []
    for i in range(limit):
        _touch_heartbeat()
        row = rows.nth(i)
        label = _first_text_from_row(row, resolved_label_sel)
        start_text = _first_text_from_row(row, resolved_start_sel)
        end_text = _first_text_from_row(row, resolved_end_sel)
        raw_text = _safe_locator_text(row, timeout_ms=2000)
        if not label and not str(start_text or "").strip() and not str(end_text or "").strip() and not raw_text:
            try:
                row.scroll_into_view_if_needed(timeout=row_scroll_timeout_ms)
            except Exception:
                pass
            label = _first_text_from_row(row, resolved_label_sel)
            start_text = _first_text_from_row(row, resolved_start_sel)
            end_text = _first_text_from_row(row, resolved_end_sel)
            raw_text = _safe_locator_text(row, timeout_ms=2000)

        start_sec = _parse_mmss_to_seconds(start_text)
        end_sec = _parse_mmss_to_seconds(end_text)

        # Fallback extraction from row text when either timestamp is missing.
        fb_start, fb_end = _extract_start_end_from_text(raw_text)
        if start_sec <= 0 and fb_start > 0:
            start_sec = fb_start
        if end_sec <= 0 and fb_end > 0:
            end_sec = fb_end

        if not label and raw_text:
            lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            if lines:
                label = lines[-1]

        items.append(
            {
                "segment_index": i + 1,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "current_label": label,
                "raw_text": raw_text,
            }
        )
        if callable(progress_callback) and extract_progress_every > 0 and (
            (i + 1) % extract_progress_every == 0 or (i + 1) == limit
        ):
            try:
                progress_callback(i + 1, limit)
            except Exception:
                pass
    return items


_clean_json_text = _gemini._clean_json_text
_repair_gemini_json_text = _gemini._repair_gemini_json_text
_enforce_gemini_output_contract = _gemini._enforce_gemini_output_contract
_parse_json_text = _gemini._parse_json_text
_parse_gemini_response = _gemini._parse_gemini_response


def _normalize_operation_action(action: str) -> str:
    a = str(action or "").strip().lower()
    aliases = {
        "e": "edit",
        "edit": "edit",
        "s": "split",
        "split": "split",
        "d": "delete",
        "delete": "delete",
        "remove": "delete",
        "m": "merge",
        "merge": "merge",
    }
    return aliases.get(a, "")


def _normalize_operations(payload: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    raw_ops = payload.get("operations", [])
    if not isinstance(raw_ops, list):
        return []
    max_ops = max(0, int(_cfg_get(cfg or {}, "run.max_structural_operations", 12)))
    structural_allow_split = bool(_cfg_get(cfg or {}, "run.structural_allow_split", False))
    structural_allow_merge = bool(_cfg_get(cfg or {}, "run.structural_allow_merge", False))
    structural_allow_delete = bool(_cfg_get(cfg or {}, "run.structural_allow_delete", False))
    out: List[Dict[str, Any]] = []
    for item in raw_ops:
        action = ""
        idx = 0
        if isinstance(item, dict):
            action = _normalize_operation_action(
                str(item.get("action") or item.get("op") or item.get("type") or "")
            )
            idx_raw = item.get("segment_index", item.get("index", item.get("segment", 0)))
            try:
                idx = int(idx_raw)
            except Exception:
                idx = 0
        elif isinstance(item, str):
            token = item.strip().lower()
            # Examples: "split 3", "d 5"
            m = re.match(r"([a-z]+)\s+(\d+)$", token)
            if m:
                action = _normalize_operation_action(m.group(1))
                idx = int(m.group(2))
            else:
                action = _normalize_operation_action(token)
        if not action or idx <= 0:
            continue
        if action == "split" and not structural_allow_split:
            continue
        if action == "merge" and not structural_allow_merge:
            continue
        if action == "delete" and not structural_allow_delete:
            continue
        out.append({"action": action, "segment_index": idx})
        if max_ops and len(out) >= max_ops:
            break
    return out


def build_prompt(
    segments: List[Dict[str, Any]],
    extra_instructions: str,
    allow_operations: bool = True,
    policy_trigger: str = "base",
) -> str:
    header = (
        "You are an Atlas Standard Tier-3 labeling assistant.\n"
        "You may receive the full task video as attached media plus employee segment text.\n"
        "Use the video as source of truth; employee labels may be wrong.\n"
        "Destroy and rebuild: treat draft phrasing as potentially flawed and rewrite from scratch from video evidence.\n"
        "For each segment index, output corrected label and timestamps when needed.\n"
        "Apply one-mental-model policy: one continuous interaction toward one goal per segment.\n"
        "Gripper rule: treat gripper as an extension of hand.\n"
        "Usually do NOT mention the tool in labels; if unavoidable, use only 'gripper'.\n"
        "Never use tool terms like mechanical arm / robotic arm / robot arm / manipulator / claw arm.\n"
        "Split only when goal changes or hands disengage/restart; do not split only for No Action pauses.\n"
        "Continuity rule: if same coarse goal continues without disengaging from the object, keep one coarse segment.\n"
        "CRITICAL continuity: if draft has 3 or more consecutive segments of the same ongoing action "
        "and tool/object is never dropped, you MUST merge them.\n"
        "Coarse-goal verbs: avoid mechanical muscle-motion phrasing (e.g., 'move saw back and forth'). "
        "Use task-goal verbs (e.g., 'cut wood with saw', 'sand board with sandpaper').\n"
        "No token stuttering: never repeat words/phrases like 'detangle detangle' or 'pull loosened pull loosened'.\n"
        "No '-ing' verbs: use imperative commands only (e.g., 'turn mold', not 'turning the mold').\n"
        "Timestamp strictness: describe only what happens inside each exact segment start_sec/end_sec; "
        "do not shift actions into neighboring segments.\n"
        "Use coarse single-goal labels for repetitive actions; use dense labels only when needed.\n"
        "Dense labels may include multiple atomic actions separated by commas/and.\n"
        "Do not exceed 20 words or 2 atomic actions per label (typically one separator: a single comma or one 'and').\n"
        "Do not write narrative filler words like then/another/next/continue/again.\n"
        "For small corrective reorientation/reposition, prefer verb 'adjust'.\n"
        "Avoid forbidden verbs: rotate, inspect, check, look, examine, reach, grab, relocate.\n"
        "Use conservative object names that are directly visible.\n"
        "If object identity is unclear after careful inspection, use safe general nouns (tool/container/item).\n"
        "Do not guess hidden object identities and do not keep placeholder/default labels.\n"
        "If surface type/elevation is unclear (floor mat vs table/shelf), do not guess raised furniture.\n"
        "Use neutral location wording (on surface/on mat/on floor) unless elevation is clearly visible.\n"
        "Use 'place' only with explicit location (on/in/into/onto/at/to/inside/under/over).\n"
        "No-Action pause rule: if ego still holds the task object/tool during a pause, do not use 'No Action'. "
        "Keep/merge it with surrounding action.\n"
        "Attach verbs to objects: do not write 'pick up and place box' or 'pick up box and place under desk'; "
        "write 'pick up box, place box under desk'.\n"
        "If the segment clearly includes lift then placement, include both actions (pick up ..., place ...).\n"
        "Hold First Rule: if a label contains both a 'hold' action and another action, ALWAYS list 'hold' first.\n"
        "Template: hold [object], [action] [object] (e.g., 'hold shoe sole, press shoe sole').\n"
        "Independent rewrite: treat input draft labels as potentially flawed; if a label violates Tier-3 phrasing, "
        "rewrite from scratch rather than patching shorthand.\n"
        "No shortcuts: do not merge distinct physical interactions into one invalid phrase to save words.\n"
        "Avoid body-part wording (hands/fingers/body parts) unless unavoidable.\n"
        "Examples:\n"
        "BAD: pick up and place stack of paper\n"
        "GOOD: pick up stack of paper from desk, place stack of paper into cardboard box\n"
        "BAD: place bag\n"
        "GOOD: place bag in cabinet\n"
        "BAD: seg1='move saw to cut wood board' + seg2='finish cutting wood board' while interaction is continuous\n"
        "GOOD: merge into one segment label 'cut wood board with saw'\n"
        "BAD: seg='No Action' while tool is still held between polish/adjust segments\n"
        "GOOD: merge/relabel pause into surrounding action; do not isolate No Action.\n"
        "BAD: paint chair -> dip paintbrush -> paint chair in separate short consecutive segments\n"
        "GOOD: merge micro-actions into one segment label 'paint chair with paintbrush' when tool is never dropped.\n"
        "BAD: move comb through wig to detangle\n"
        "GOOD: detangle wig with comb\n"
        "BAD: move hair straightener to press wig section\n"
        "GOOD: straighten wig section with hair straightener\n"
        "BAD: press shoe sole, hold shoe sole\n"
        "GOOD: hold shoe sole, press shoe sole\n"
        "If a segment timestamp is wrong, correct start_sec/end_sec.\n"
        "Label rules: imperative style, concise, minimum 2 words, maximum 20 words.\n"
        "Use \"No Action\" only as standalone label.\n"
        "If boundaries are fundamentally wrong, you may request split/merge operations before final labels.\n"
        "Allowed operations: edit, split, merge. Do NOT use delete.\n"
        "Operation segment_index refers to the row index at execution time.\n"
        "Operations must be ordered exactly as they should be executed.\n"
        "Return strict JSON object only:\n"
        "Response must start with '{' and end with '}'.\n"
        "Do not wrap JSON in markdown code fences.\n"
        "{\"operations\":[{\"action\":\"split\",\"segment_index\":3}],"
        "\"segments\":[{\"segment_index\":1,\"start_sec\":0.0,\"end_sec\":1.2,\"label\":\"...\"}]}\n"
        "If no structural change is needed, return \"operations\":[]\n"
        "Keep segment count and indices unchanged unless a justified split/merge is needed; timestamps must stay monotonic.\n"
    )
    if not allow_operations:
        header += (
            "Structural operations are disabled for this pass.\n"
            "Return operations as an empty list.\n"
        )
    header += (
        "CRITICAL: You MUST provide a label for EVERY segment_index listed below. "
        "Do NOT skip any segment. Do NOT request delete operations. "
        "If segments are repetitive, use a coarse single-goal label for each one "
        "(e.g., 'roll dough, place dough in tray') but still output every segment.\n"
    )
    policy_summary = policy_context.build_policy_prompt_summary()
    retrieved_policy_context = policy_context.retrieve_runtime_rules(
        segments,
        trigger=policy_trigger,
        budget={"rules": 6, "examples": 1},
    )
    lines = ["Canonical policy:", policy_summary, "", "Segments input:"]
    for seg in segments:
        lines.append(
            f"- segment_index={seg['segment_index']} start_sec={seg['start_sec']} "
            f"end_sec={seg['end_sec']} current_label={json.dumps(seg.get('current_label', ''), ensure_ascii=False)} "
            f"raw_text={json.dumps(seg.get('raw_text', ''), ensure_ascii=False)}"
        )
    extra_blocks = [block for block in [retrieved_policy_context, extra_instructions.strip()] if block]
    merged_extra_instructions = "\n".join(extra_blocks).strip()
    if merged_extra_instructions:
        lines.append("")
        lines.append("Extra instructions (Filtered for current context):")
        
        # Token Diff Optimization: Subset rules based on segment keywords
        seg_words = " ".join([f"{seg.get('current_label', '')} {seg.get('raw_text', '')}".lower() for seg in segments])
        seg_tokens = set(re.findall(r'[a-z]{3,}', seg_words))
        
        diff_rules = []
        for rule in merged_extra_instructions.split('\n'):
            rule_str = rule.strip()
            if not rule_str:
                continue
            rule_tokens = set(re.findall(r'[a-z]{4,}', rule_str.lower()))
            # Keep if no distinct tokens, rule overlaps with segments, or is small (header)
            if not rule_tokens or any(t in seg_tokens for t in rule_tokens) or len(rule_str) < 40:
                diff_rules.append(rule_str)

        lines.append("\n".join(diff_rules) if diff_rules else merged_extra_instructions)
        
    return header + "\n".join(lines)


def _read_optional_text_file(path_text: str) -> str:
    path_raw = (path_text or "").strip()
    if not path_raw:
        return ""
    try:
        p = Path(path_raw)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _resolve_system_instruction(cfg: Dict[str, Any]) -> str:
    file_text = _read_optional_text_file(str(_cfg_get(cfg, "gemini.system_instruction_file", "")))
    inline_text = str(_cfg_get(cfg, "gemini.system_instruction_text", "")).strip()
    policy_text = policy_context.build_policy_prompt_summary().strip()
    chunks = [txt for txt in [file_text, inline_text, policy_text] if txt]
    return "\n\n".join(chunks).strip()


def _count_atomic_actions_in_label(label: str) -> int:
    text = (label or "").strip()
    if not text:
        return 0
    if text.lower() == "no action":
        return 1
    count = 0
    for part in re.split(r"\s*,\s*", text):
        chunk = part.strip()
        if not chunk:
            continue
        subparts = [p.strip() for p in re.split(r"\band\b", chunk, flags=re.IGNORECASE) if p.strip()]
        if subparts:
            count += len(subparts)
        else:
            count += 1
    return max(1, count)


_DISALLOWED_TOOL_TERMS = (
    "mechanical arm",
    "robotic arm",
    "robot arm",
    "manipulator",
    "robot gripper",
    "claw arm",
)


def _normalize_gripper_terms(text: str) -> str:
    out = text or ""
    for term in _DISALLOWED_TOOL_TERMS:
        out = re.sub(rf"\b{re.escape(term)}\b", "gripper", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def _validate_segment_plan_against_policy(
    cfg: Dict[str, Any],
    source_segments: List[Dict[str, Any]],
    segment_plan: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    min_words = max(1, int(_cfg_get(cfg, "run.min_label_words", 2)))
    max_words = max(min_words, int(_cfg_get(cfg, "run.max_label_words", 20)))
    max_atomic_actions = max(1, int(_cfg_get(cfg, "run.max_atomic_actions_per_label", 2)))
    forbidden_verbs_raw = _cfg_get(cfg, "run.forbidden_label_verbs", [])
    forbidden_verbs = [str(v).strip().lower() for v in forbidden_verbs_raw if str(v).strip()]
    allowed_verb_token_patterns = _allowed_label_start_verb_token_patterns_from_cfg(cfg)
    forbidden_narrative_raw = _cfg_get(cfg, "run.forbidden_narrative_words", [])
    forbidden_narrative_words = [str(v).strip().lower() for v in forbidden_narrative_raw if str(v).strip()]
    skip_unchanged_lexical = bool(
        _cfg_get(cfg, "run.skip_policy_lexical_checks_on_unchanged_labels", False)
    )
    place_location_pattern = re.compile(r"\bplace\b.*\b(on|in|into|onto|at|to|inside|under|over)\b", re.IGNORECASE)
    chained_verb_without_object_pattern = re.compile(
        r"\b(pick up|place|move|adjust|hold|align|relocate)\s+and\s+(pick up|place|move|adjust|hold|align|relocate)\b",
        re.IGNORECASE,
    )
    orphan_second_place_pattern = re.compile(
        r"\band\s+place\s+(on|in|into|onto|at|to|inside|under|over)\b",
        re.IGNORECASE,
    )
    body_part_reference_pattern = re.compile(
        r"\b(hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms|leg|legs|foot|feet|toe|toes)\b",
        re.IGNORECASE,
    )
    token_stuttering_pattern = re.compile(
        r"\b([a-z]+(?:\s+[a-z]+){0,2})\s+\1\b",
        re.IGNORECASE,
    )
    mechanical_motion_pattern = re.compile(
        r"\bmove\s+(?:comb(?:\s+tail)?|hair\s+straightener)\b|"
        r"\bmove\s+\w+\s+back\s+and\s+forth\b",
        re.IGNORECASE,
    )
    
    confidence_threshold = float(_cfg_get(cfg, "run.confidence_threshold", 0.7) or 0.7)
    block_on_escalation = bool(_cfg_get(cfg, "run.block_on_escalation_flag", True))
    block_on_high_audit_risk = bool(_cfg_get(cfg, "run.block_on_high_audit_risk", True))

    source_by_idx: Dict[int, Dict[str, Any]] = {}
    for seg in source_segments:
        try:
            source_by_idx[int(seg.get("segment_index", 0))] = seg
        except Exception:
            continue

    errors: List[str] = []
    warnings: List[str] = []
    prev_start = -1.0
    prev_end = -1.0

    for idx in sorted(segment_plan):
        item = segment_plan[idx]
        label = str(item.get("label", "")).strip()
        label_l = label.lower()
        start = _safe_float(item.get("start_sec"), -1.0)
        end = _safe_float(item.get("end_sec"), -1.0)
        source = source_by_idx.get(idx)
        source_label = str(source.get("current_label", "")).strip() if source is not None else ""
        label_unchanged_from_source = (
            bool(source_label)
            and _normalize_label_for_compare(source_label) == _normalize_label_for_compare(label)
        )

        if not label:
            errors.append(f"segment {idx}: empty label")
        else:
            words = [w for w in re.split(r"\s+", label) if w]
            if label_unchanged_from_source and skip_unchanged_lexical:
                # Avoid blocking whole episodes on legacy/source labels that were not edited now.
                pass
            elif label != "No Action":
                if len(words) < min_words:
                    errors.append(f"segment {idx}: label has fewer than {min_words} words")
                if len(words) > max_words:
                    errors.append(f"segment {idx}: label has more than {max_words} words")
                if not _label_starts_with_allowed_action_verb(label, allowed_verb_token_patterns):
                    errors.append(f"segment {idx}: label must start with an allowed action verb")
                clause_starts_invalid = False
                for clause in _label_action_clauses(label):
                    if not _label_starts_with_allowed_action_verb(clause, allowed_verb_token_patterns):
                        clause_starts_invalid = True
                        break
                if clause_starts_invalid:
                    errors.append(f"segment {idx}: each action clause must start with an allowed action verb")
                for v in forbidden_verbs:
                    if re.search(rf"\b{re.escape(v)}\b", label_l):
                        errors.append(f"segment {idx}: forbidden verb '{v}' found")
                for token in forbidden_narrative_words:
                    if re.search(rf"\b{re.escape(token)}\b", label_l):
                        errors.append(f"segment {idx}: narrative token '{token}' found")
                for term in _DISALLOWED_TOOL_TERMS:
                    if re.search(rf"\b{re.escape(term)}\b", label_l):
                        errors.append(
                            f"segment {idx}: disallowed tool term '{term}' found (use 'gripper' only if unavoidable)"
                        )
                if re.search(r"\bgripper\b", label_l):
                    warnings.append(f"segment {idx}: label mentions 'gripper' (ensure tool mention is unavoidable)")
                if re.search(r"\d", label):
                    errors.append(f"segment {idx}: label contains numerals")
                if body_part_reference_pattern.search(label):
                    errors.append(f"segment {idx}: avoid body-part wording unless unavoidable")
                if token_stuttering_pattern.search(label):
                    errors.append(f"segment {idx}: repeated token/phrase detected (stuttering)")
                if mechanical_motion_pattern.search(label):
                    errors.append(f"segment {idx}: mechanical-motion phrasing detected (use coarse goal verb)")
                if "place" in label_l and not place_location_pattern.search(label):
                    errors.append(f"segment {idx}: 'place' missing explicit location")
                if chained_verb_without_object_pattern.search(label):
                    errors.append(
                        f"segment {idx}: verbs must attach to objects (avoid '<verb> and <verb>' chaining)"
                    )
                if orphan_second_place_pattern.search(label):
                    errors.append(f"segment {idx}: 'place' missing explicit object after conjunction")
                if re.search(r"\bno action\b", label_l) and label_l != "no action":
                    errors.append(f"segment {idx}: 'No Action' must be standalone")
                action_count = _count_atomic_actions_in_label(label)
                if action_count > max_atomic_actions:
                    errors.append(
                        f"segment {idx}: label has more than {max_atomic_actions} atomic actions"
                    )

            # Confidence and Escalation Checks (No-Confidence Exception)
            confidence = _safe_float(item.get("confidence"), 1.0)
            if confidence < confidence_threshold:
                errors.append(f"segment {idx}: low confidence ({confidence:.2f} < {confidence_threshold})")
            
            if block_on_escalation and item.get("escalation_flag"):
                errors.append(f"segment {idx}: escalation_flag is set by model")
                
            audit_risk = item.get("audit_risk", {})
            if block_on_high_audit_risk and isinstance(audit_risk, dict) and audit_risk.get("level") == "high":
                errors.append(f"segment {idx}: high audit risk detected")
            else:
                if "," in label or " and " in label_l:
                    errors.append(f"segment {idx}: 'No Action' must be standalone")

        if start < 0 or end < 0:
            errors.append(f"segment {idx}: invalid timestamp values")
        elif end <= start:
            errors.append(f"segment {idx}: end_sec must be greater than start_sec")

        if prev_start >= 0 and start < prev_start - 0.05:
            errors.append(f"segment {idx}: start_sec is not monotonic")
        if prev_end >= 0 and start < prev_end - 0.05:
            errors.append(f"segment {idx}: overlaps previous segment")
        prev_start = max(prev_start, start)
        prev_end = max(prev_end, end)

        if source is not None:
            src_start = _safe_float(source.get("start_sec"), start)
            src_end = _safe_float(source.get("end_sec"), end)
            if abs(start - src_start) > 12 or abs(end - src_end) > 12:
                warnings.append(f"segment {idx}: large timestamp drift from source")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "segment_count": len(segment_plan),
    }


def _is_timestamp_policy_error(message: str) -> bool:
    m = str(message or "").strip().lower()
    if not m:
        return False
    markers = (
        "invalid timestamp values",
        "end_sec must be greater than start_sec",
        "start_sec is not monotonic",
        "overlaps previous segment",
    )
    return any(token in m for token in markers)


def _is_no_action_policy_error(message: str) -> bool:
    m = str(message or "").strip().lower()
    if not m:
        return False
    return "'no action' must be standalone" in m


_gemini_file_state = _gemini._gemini_file_state
_wait_for_gemini_file_ready = _gemini._wait_for_gemini_file_ready
_normalize_upload_chunk_size = _gemini._normalize_upload_chunk_size
_upload_video_via_gemini_files_api = _gemini._upload_video_via_gemini_files_api
_cleanup_gemini_uploaded_file = _gemini._cleanup_gemini_uploaded_file
_sweep_stale_gemini_files = _gemini._sweep_stale_gemini_files
_is_gemini_quota_error_text = _gemini._is_gemini_quota_error_text
_is_gemini_quota_exceeded_429 = _gemini._is_gemini_quota_exceeded_429
_is_gemini_quota_error = _gemini._is_gemini_quota_error
_build_gemini_generation_config = _gemini._build_gemini_generation_config
call_gemini_labels = _gemini.call_gemini_labels


_CHUNK_CONSISTENCY_VERB_PREFIXES: Tuple[str, ...] = (
    "pick up",
    "place",
    "open",
    "close",
    "pull open",
    "push",
    "adjust",
    "move",
    "drag",
    "tighten",
    "loosen",
    "remove",
    "insert",
    "fold",
    "spread out",
    "sand",
    "twist",
    "pour",
    "scoop",
    "hold",
    "position",
    "align",
    "pry open",
    "drive",
    "set",
    "put",
)
_CHUNK_CONSISTENCY_EQUIVALENCE_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("table", "surface"),
)
_CHUNK_CONSISTENCY_PREPOSITION_RE = re.compile(
    r"\b(from|in|into|on|onto|under|inside|at|to|with|over|near|across|through)\b",
    flags=re.IGNORECASE,
)
_CHUNK_CONSISTENCY_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", flags=re.IGNORECASE)
_CHUNK_CONSISTENCY_STOPWORDS: set[str] = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "with",
    "from",
    "in",
    "into",
    "on",
    "onto",
    "under",
    "inside",
    "at",
    "over",
    "near",
    "across",
    "through",
}


def _consistency_norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip(" ,.;:")).lower()


def _consistency_tokens(text: str) -> List[str]:
    return [t.lower() for t in _CHUNK_CONSISTENCY_TOKEN_RE.findall(text or "")]


def _extract_consistency_terms_from_label(label: str, max_terms: int = 8) -> List[str]:
    if not label:
        return []
    terms: List[str] = []
    clauses = [c.strip().lower() for c in re.split(r",", label) if c and c.strip()]
    for clause in clauses:
        if clause == "no action":
            continue
        rest = clause
        for prefix in _CHUNK_CONSISTENCY_VERB_PREFIXES:
            token = prefix + " "
            if rest.startswith(token):
                rest = rest[len(token) :].strip()
                break
        if not rest:
            continue
        m = _CHUNK_CONSISTENCY_PREPOSITION_RE.search(rest)
        candidates = [rest]
        if m:
            candidates = [rest[: m.start()].strip(), rest[m.end() :].strip()]
        for cand in candidates:
            norm = _consistency_norm(re.sub(r"^(the|a|an)\s+", "", cand))
            if not norm:
                continue
            if norm in _CHUNK_CONSISTENCY_STOPWORDS:
                continue
            if len(_consistency_tokens(norm)) == 0:
                continue
            if norm not in terms:
                terms.append(norm)
                if len(terms) >= max_terms:
                    return terms
    return terms


def _find_equivalent_canonical_term(norm_term: str, canonical_terms: List[str]) -> str:
    if not norm_term:
        return ""
    for existing in canonical_terms:
        if _consistency_norm(existing) == norm_term:
            return existing

    for group in _CHUNK_CONSISTENCY_EQUIVALENCE_GROUPS:
        group_set = {_consistency_norm(x) for x in group}
        if norm_term in group_set:
            for existing in canonical_terms:
                if _consistency_norm(existing) in group_set:
                    return existing

    term_tokens = _consistency_tokens(norm_term)
    if not term_tokens:
        return ""
    term_head = term_tokens[-1]
    term_set = set(term_tokens)
    for existing in canonical_terms:
        existing_norm = _consistency_norm(existing)
        existing_tokens = _consistency_tokens(existing_norm)
        if not existing_tokens:
            continue
        if existing_tokens[-1] != term_head:
            continue
        existing_set = set(existing_tokens)
        overlap = term_set.intersection(existing_set)
        if term_set.issubset(existing_set) or existing_set.issubset(term_set):
            return existing
        if len(overlap) >= max(1, min(len(term_set), len(existing_set)) - 1):
            return existing
    return ""


def _apply_consistency_aliases_to_label(label: str, alias_to_canonical: Dict[str, str]) -> str:
    out = label or ""
    if not out or not alias_to_canonical:
        return out
    replacements = sorted(alias_to_canonical.items(), key=lambda item: len(item[0]), reverse=True)
    for alias_norm, canonical in replacements:
        src = _consistency_norm(alias_norm)
        dst = _consistency_norm(canonical)
        if not src or not dst or src == dst:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(src) + r"(?![a-z0-9])"
        out = re.sub(pattern, dst, out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def _update_chunk_consistency_memory(
    label: str,
    canonical_terms: List[str],
    alias_to_canonical: Dict[str, str],
    memory_limit: int,
) -> str:
    rewritten = _apply_consistency_aliases_to_label(label, alias_to_canonical)
    extracted_terms = _extract_consistency_terms_from_label(rewritten)
    for term in extracted_terms:
        term_norm = _consistency_norm(term)
        if not term_norm:
            continue
        if term_norm in alias_to_canonical:
            continue
        canonical = _find_equivalent_canonical_term(term_norm, canonical_terms)
        if canonical:
            alias_to_canonical[term_norm] = canonical
            rewritten = _apply_consistency_aliases_to_label(rewritten, alias_to_canonical)
            continue
        alias_to_canonical[term_norm] = term
        canonical_terms.append(term)

    if memory_limit > 0 and len(canonical_terms) > memory_limit:
        canonical_terms[:] = canonical_terms[-memory_limit:]
        allowed = {_consistency_norm(term) for term in canonical_terms}
        for alias_key in list(alias_to_canonical.keys()):
            alias_norm = _consistency_norm(alias_key)
            canonical_norm = _consistency_norm(alias_to_canonical.get(alias_key, ""))
            if alias_norm in allowed or canonical_norm in allowed:
                continue
            alias_to_canonical.pop(alias_key, None)
    return rewritten


def _build_chunk_consistency_prompt_hint(canonical_terms: List[str], max_terms: int) -> str:
    if not canonical_terms or max_terms <= 0:
        return ""
    selected = canonical_terms[-max_terms:]
    return (
        "PREFERRED OBJECT/LOCATION TERMS from previous chunks (must keep naming stable for same object): "
        + " | ".join(selected)
    )


def _segment_duration_exceeds_limit(seg: Dict[str, Any], max_duration_sec: float) -> bool:
    start = _safe_float(seg.get("start_sec"), 0.0)
    end = _safe_float(seg.get("end_sec"), start)
    return (end - start) > max_duration_sec + 1e-6


def _collect_chunk_structural_operations(
    *,
    cfg: Dict[str, Any],
    chunk_payload: Dict[str, Any],
    chunk_segments: List[Dict[str, Any]],
    max_segment_duration_sec: float,
    split_only: bool,
) -> List[Dict[str, Any]]:
    raw_ops = _normalize_operations(chunk_payload, cfg=cfg)
    if not raw_ops:
        return []
    segment_by_index: Dict[int, Dict[str, Any]] = {
        int(seg.get("segment_index", 0) or 0): seg for seg in chunk_segments
    }
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()
    for op in raw_ops:
        action = str(op.get("action", "") or "").strip().lower()
        idx = int(op.get("segment_index", 0) or 0)
        if idx <= 0 or idx not in segment_by_index:
            continue
        if split_only and action != "split":
            continue
        if action == "split":
            if max_segment_duration_sec > 0 and not _segment_duration_exceeds_limit(
                segment_by_index[idx],
                max_segment_duration_sec,
            ):
                continue
        key = (action, idx)
        if key in seen:
            continue
        seen.add(key)
        out.append({"action": action, "segment_index": idx})
    out.sort(key=lambda item: (str(item.get("action", "")) != "split", -int(item.get("segment_index", 0) or 0)))
    return out


_request_labels_with_optional_segment_chunking = _gemini._request_labels_with_optional_segment_chunking


_safe_float = _utils._safe_float


def _short_error_text(exc: Exception, max_len: int = 180) -> str:
    raw = str(exc or "").strip()
    if not raw:
        return exc.__class__.__name__
    first = raw.splitlines()[0].strip()
    if len(first) > max_len:
        return first[:max_len] + "..."
    return first


_log_gemini_usage = _gemini._log_gemini_usage


def _label_main_verb(label: str) -> str:
    text = re.sub(r"\s+", " ", (label or "").strip()).lower()
    if not text:
        return ""
    m = re.match(r"([a-z]+)", text)
    return m.group(1) if m else ""


def _is_no_action_label(label: str) -> bool:
    normalized = re.sub(r"[\s_-]+", " ", (label or "").strip()).lower()
    return normalized in {"no action", "noaction"}


_LABEL_TOKEN_RE = re.compile(r"[a-z]+")
_LABEL_OVERLAP_STOPWORDS: set[str] = {
    "no",
    "action",
    "with",
    "on",
    "in",
    "into",
    "onto",
    "at",
    "to",
    "from",
    "under",
    "over",
    "inside",
}


def _label_content_tokens(label: str) -> set[str]:
    text = re.sub(r"\s+", " ", (label or "").strip()).lower()
    if not text:
        return set()
    tokens = set(_LABEL_TOKEN_RE.findall(text))
    return {tok for tok in tokens if tok and tok not in _LABEL_OVERLAP_STOPWORDS}


_AUTOFIX_ALLOWED_LABEL_START_VERB_TOKEN_PATTERNS: Tuple[Tuple[str, ...], ...] = (
    ("pick", "up"),
    ("put", "down"),
    ("place",),
    ("move",),
    ("adjust",),
    ("align",),
    ("hold",),
    ("cut",),
    ("open",),
    ("close",),
    ("peel",),
    ("secure",),
    ("wipe",),
    ("flip",),
    ("pull",),
    ("push",),
    ("insert",),
    ("remove",),
    ("attach",),
    ("detach",),
    ("connect",),
    ("disconnect",),
    ("tighten",),
    ("loosen",),
    ("screw",),
    ("unscrew",),
    ("press",),
    ("twist",),
    ("turn",),
    ("slide",),
    ("lift",),
    ("lower",),
    ("set",),
    ("position",),
    ("straighten",),
    ("comb",),
    ("detangle",),
    ("sand",),
    ("paint",),
    ("clean",),
    ("put",),
    ("stir",),
    ("mix",),
    ("blend",),
    ("pour",),
    ("squeeze",),
    ("fold",),
    ("unfold",),
    ("wrap",),
    ("unwrap",),
    ("tape",),
    ("tie",),
    ("untie",),
    ("thread",),
    ("sew",),
    ("stitch",),
    ("knit",),
    ("weave",),
    ("braid",),
    ("brush",),
    ("scrub",),
    ("sweep",),
    ("mop",),
    ("rinse",),
    ("wash",),
    ("dry",),
    ("iron",),
    ("spray",),
    ("apply",),
    ("spread",),
    ("rub",),
    ("dab",),
    ("tap",),
    ("pat",),
    ("shake",),
    ("roll",),
    ("unroll",),
    ("stack",),
    ("sort",),
    ("arrange",),
    ("gather",),
    ("collect",),
    ("dump",),
    ("empty",),
    ("fill",),
    ("scoop",),
    ("scrape",),
    ("trim",),
    ("clip",),
    ("chop",),
    ("dice",),
    ("slice",),
    ("grate",),
    ("grind",),
    ("crush",),
    ("tear",),
    ("snap",),
    ("bend",),
    ("stretch",),
    ("clamp",),
    ("staple",),
    ("pin",),
    ("nail",),
    ("hammer",),
    ("drill",),
    ("file",),
    ("polish",),
    ("buff",),
    ("sharpen",),
    ("saw",),
    ("chisel",),
    ("carve",),
    ("mark",),
    ("write",),
    ("draw",),
    ("trace",),
    ("measure",),
    ("weigh",),
    ("test",),
    ("lock",),
    ("unlock",),
    ("hook",),
    ("unhook",),
    ("hang",),
    ("mount",),
    ("load",),
    ("unload",),
    ("pack",),
    ("unpack",),
    ("seal",),
    ("unseal",),
    ("cap",),
    ("uncap",),
    ("plug",),
    ("unplug",),
    ("zip",),
    ("unzip",),
    ("fasten",),
    ("unfasten",),
    ("release",),
    ("grip",),
    ("drop",),
    ("toss",),
    ("swap",),
    ("replace",),
    ("transfer",),
    ("drag",),
    ("carry",),
    ("feed",),
    ("dig",),
    ("rake",),
    ("prune",),
    ("assemble",),
    ("disassemble",),
    ("repair",),
    ("fix",),
    ("glue",),
    ("weld",),
    ("solder",),
    ("splice",),
    ("operate",),
    ("activate",),
    ("switch",),
    ("toggle",),
    ("spin",),
    ("wind",),
    ("unwind",),
    ("coil",),
    ("dip",),
    ("soak",),
    ("drain",),
    ("filter",),
    ("sift",),
    ("knead",),
    ("flatten",),
    ("shape",),
    ("mold",),
    ("smooth",),
    ("level",),
    ("balance",),
    ("center",),
    ("tilt",),
    ("prop",),
    ("support",),
    ("guide",),
    ("steer",),
)

_AUTOFIX_OBJECT_EXPECTING_VERBS: set[str] = {
    "pick up",
    "place",
    "move",
    "adjust",
    "align",
    "hold",
    "cut",
    "open",
    "close",
    "peel",
    "secure",
    "wipe",
    "flip",
    "pull",
    "push",
    "insert",
    "remove",
    "attach",
    "detach",
    "connect",
    "disconnect",
    "tighten",
    "loosen",
    "screw",
    "unscrew",
}

_AUTOFIX_VERB_HINT_MAP: Tuple[Tuple[str, str], ...] = (
    ("wire", "connect"),
    ("cable", "connect"),
    ("plug", "connect"),
    ("socket", "connect"),
    ("cloth", "wipe"),
    ("towel", "wipe"),
    ("rag", "wipe"),
    ("screw", "tighten"),
    ("bolt", "tighten"),
    ("nut", "tighten"),
    ("lid", "close"),
    ("door", "close"),
    ("cap", "close"),
    ("switch", "press"),
    ("button", "press"),
    ("paper", "place"),
    ("box", "place"),
)


def _allowed_label_start_verb_token_patterns_from_cfg(cfg: Dict[str, Any]) -> List[Tuple[str, ...]]:
    raw = _cfg_get(cfg, "run.allowed_label_start_verbs", [])
    patterns: List[Tuple[str, ...]] = []
    if isinstance(raw, list):
        for item in raw:
            tokens = tuple(re.findall(r"[a-z]+", str(item).lower()))
            if tokens:
                patterns.append(tokens)
    if not patterns:
        patterns = list(_AUTOFIX_ALLOWED_LABEL_START_VERB_TOKEN_PATTERNS)
    deduped: List[Tuple[str, ...]] = []
    seen: set[Tuple[str, ...]] = set()
    for pattern in patterns:
        if pattern in seen:
            continue
        seen.add(pattern)
        deduped.append(pattern)
    return deduped


def _label_starts_with_allowed_action_verb(
    action_phrase: str,
    allowed_verb_token_patterns: List[Tuple[str, ...]],
) -> bool:
    phrase = re.sub(r"\s+", " ", (action_phrase or "").strip()).lower()
    if not phrase or phrase == "no action":
        return False
    words = re.findall(r"[a-z]+", phrase)
    if not words:
        return False
    for pattern in allowed_verb_token_patterns:
        if not pattern:
            continue
        n = len(pattern)
        if len(words) >= n and tuple(words[:n]) == pattern:
            if any(word.endswith("ing") for word in words[:n]):
                return False
            return True
    return False


def _contains_forbidden_verb_in_label(label: str, forbidden_verbs: List[str]) -> bool:
    text = (label or "").strip().lower()
    if not text:
        return False
    for verb in forbidden_verbs:
        if re.search(rf"\b{re.escape(verb)}\b", text):
            return True
    return False


def _strip_forbidden_verbs_for_autofix(label: str, forbidden_verbs: List[str]) -> str:
    out = label or ""
    for verb in forbidden_verbs:
        if not verb:
            continue
        out = re.sub(rf"\b{re.escape(verb)}\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip(" ,.;:")
    return out


def _action_phrase_missing_object_for_autofix(action_phrase: str) -> bool:
    phrase = re.sub(r"\s+", " ", (action_phrase or "").strip()).lower()
    if not phrase:
        return True
    for verb in sorted(_AUTOFIX_OBJECT_EXPECTING_VERBS, key=len, reverse=True):
        if phrase == verb:
            return True
        if phrase.startswith(verb + " "):
            remaining = phrase[len(verb):].strip()
            if not remaining:
                return True
            if len(re.findall(r"[a-z]+", remaining)) == 0:
                return True
            return False
    return False


def _heuristic_autofix_verb_from_text(text: str) -> str:
    lowered = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if lowered:
        for needle, verb in _AUTOFIX_VERB_HINT_MAP:
            if needle in lowered:
                return verb
    return "pick up"


def _autofix_label_candidate(
    cfg: Dict[str, Any],
    label: str,
    source_label: str,
    forbidden_verbs: List[str],
    allowed_verb_token_patterns: List[Tuple[str, ...]],
) -> str:
    min_words = max(1, int(_cfg_get(cfg, "run.min_label_words", 2)))
    max_words = max(min_words, int(_cfg_get(cfg, "run.max_label_words", 20)))

    def _normalize(x: str) -> str:
        out = _normalize_label_min_safety(x)
        out = _strip_forbidden_verbs_for_autofix(out, forbidden_verbs)
        out = re.sub(r"\b(?:then|another|continue|next)\b", "", out, flags=re.IGNORECASE)
        out = re.sub(r"\s+", " ", out).strip(" ,.;:")
        return out

    def _valid_candidate(x: str) -> bool:
        if not x or x.lower() == "no action":
            return False
        if not _label_starts_with_allowed_action_verb(x, allowed_verb_token_patterns):
            return False
        if _contains_forbidden_verb_in_label(x, forbidden_verbs):
            return False
        first_clause = x.split(",")[0].split(" and ")[0].strip()
        if _action_phrase_missing_object_for_autofix(first_clause):
            return False
        return True

    for base in (label, source_label):
        candidate = _normalize(base)
        if _valid_candidate(candidate):
            words = [w for w in candidate.split() if w]
            if len(words) < min_words:
                candidate = f"{candidate} item".strip()
            words = [w for w in candidate.split() if w]
            if len(words) > max_words:
                if "," in candidate:
                    candidate = candidate.split(",", 1)[0].strip()
                else:
                    candidate = " ".join(words[:max_words])
            return candidate

    base_text = _normalize(label or source_label or "")
    base_tokens = re.findall(r"[a-z]+", base_text.lower())
    object_tokens = list(base_tokens)
    for pattern in allowed_verb_token_patterns:
        n = len(pattern)
        if n > 0 and len(base_tokens) >= n and tuple(base_tokens[:n]) == pattern:
            object_tokens = base_tokens[n:]
            break

    object_tokens = [t for t in object_tokens if t not in {"and", "then"}]
    object_phrase = " ".join(object_tokens).strip() or "item"
    verb = _heuristic_autofix_verb_from_text(base_text)
    candidate = _normalize(f"{verb} {object_phrase}")

    if not _label_starts_with_allowed_action_verb(candidate, allowed_verb_token_patterns):
        candidate = _normalize(f"pick up {object_phrase}")
    if not candidate:
        candidate = "pick up item"

    words = [w for w in candidate.split() if w]
    if len(words) < min_words:
        candidate = f"{candidate} item".strip()
    words = [w for w in candidate.split() if w]
    if len(words) > max_words:
        candidate = " ".join(words[:max_words])
    return candidate


_MICRO_ACTION_VERBS: set[str] = {"dip", "reload", "wet"}


def _label_action_clauses(label: str) -> List[str]:
    text = re.sub(r"\s+", " ", (label or "").strip())
    if not text:
        return []
    parts: List[str] = []
    for chunk in text.split(","):
        subs = [s.strip() for s in re.split(r"\band\b", chunk, flags=re.IGNORECASE) if s.strip()]
        parts.extend(subs)
    return parts


def _label_goal_key(label: str) -> str:
    """
    Build a coarse goal key for continuity merge detection.
    Prefer the last non-micro verb; fallback to last verb.
    """
    if _is_no_action_label(label):
        return ""
    clauses = _label_action_clauses(label)
    if not clauses:
        return ""
    verbs: List[str] = []
    for clause in clauses:
        v = _label_main_verb(clause)
        if v:
            verbs.append(v)
    if not verbs:
        return ""
    non_micro = [v for v in verbs if v not in _MICRO_ACTION_VERBS]
    return (non_micro[-1] if non_micro else verbs[-1]).strip().lower()


def _build_auto_continuity_merge_operations(
    segment_plan: Dict[int, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not bool(_cfg_get(cfg, "run.auto_continuity_merge_enabled", True)):
        return []
    if not bool(_cfg_get(cfg, "run.structural_allow_merge", True)):
        return []

    min_run = max(3, int(_cfg_get(cfg, "run.auto_continuity_merge_min_run_segments", 3)))
    min_overlap = max(0, int(_cfg_get(cfg, "run.auto_continuity_merge_min_token_overlap", 1)))

    ordered = sorted(int(k) for k in segment_plan.keys())
    if len(ordered) < min_run:
        return []

    max_merges = max(0, int(_cfg_get(cfg, "run.auto_continuity_merge_max_ops", 6)))

    def same_goal(i1: int, i2: int) -> bool:
        a = segment_plan.get(i1, {})
        b = segment_plan.get(i2, {})
        la = str(a.get("label", "")).strip()
        lb = str(b.get("label", "")).strip()
        ka = _label_goal_key(la)
        kb = _label_goal_key(lb)
        if not ka or not kb or ka != kb:
            return False
        # Exclude the shared goal verb from overlap so min_overlap is meaningful
        shared_tokens = _label_content_tokens(la).intersection(_label_content_tokens(lb))
        shared_tokens.discard(ka)
        return len(shared_tokens) >= min_overlap

    runs: List[Tuple[int, int]] = []
    run_start = ordered[0]
    run_end = ordered[0]
    for idx in ordered[1:]:
        if idx == run_end + 1 and same_goal(run_end, idx):
            run_end = idx
            continue
        if (run_end - run_start + 1) >= min_run:
            runs.append((run_start, run_end))
        run_start = idx
        run_end = idx
    if (run_end - run_start + 1) >= min_run:
        runs.append((run_start, run_end))

    if not runs:
        return []

    merge_indices: List[int] = []
    for start_idx, end_idx in runs:
        # Descending indices to keep operation row references stable.
        for idx in range(end_idx, start_idx, -1):
            merge_indices.append(idx)

    merge_indices = sorted(set(merge_indices), reverse=True)
    if max_merges and len(merge_indices) > max_merges:
        print(
            f"[policy] continuity merge capped: {len(merge_indices)} -> {max_merges} "
            f"(run.auto_continuity_merge_max_ops)"
        )
        merge_indices = merge_indices[:max_merges]
    return [{"action": "merge", "segment_index": int(idx)} for idx in merge_indices]


def _rewrite_no_action_pauses_in_plan(segment_plan: Dict[int, Dict[str, Any]], cfg: Dict[str, Any]) -> int:
    if not bool(_cfg_get(cfg, "run.no_action_pause_rewrite_enabled", True)):
        return 0
    max_pause_sec = max(0.0, float(_cfg_get(cfg, "run.no_action_pause_rewrite_max_sec", 12.0)))
    min_overlap = max(1, int(_cfg_get(cfg, "run.no_action_pause_rewrite_min_overlap_tokens", 1)))
    prefer_next_adjust = bool(_cfg_get(cfg, "run.no_action_pause_rewrite_prefer_next_adjust", True))

    ordered_indices = sorted(segment_plan.keys())
    rewrites = 0
    for pos, idx in enumerate(ordered_indices):
        item = segment_plan.get(idx, {})
        label = str(item.get("label", "")).strip()
        if not _is_no_action_label(label):
            continue
        start_sec = _safe_float(item.get("start_sec", 0.0), 0.0)
        end_sec = _safe_float(item.get("end_sec", start_sec), start_sec)
        if (end_sec - start_sec) > max_pause_sec:
            continue
        if pos == 0 or pos >= len(ordered_indices) - 1:
            continue

        prev_item = segment_plan.get(ordered_indices[pos - 1], {})
        next_item = segment_plan.get(ordered_indices[pos + 1], {})
        prev_label = str(prev_item.get("label", "")).strip()
        next_label = str(next_item.get("label", "")).strip()
        if not prev_label or not next_label:
            continue
        if _is_no_action_label(prev_label) or _is_no_action_label(next_label):
            continue

        overlap = len(_label_content_tokens(prev_label).intersection(_label_content_tokens(next_label)))
        if overlap < min_overlap:
            continue

        replacement = prev_label
        if prefer_next_adjust and _label_main_verb(next_label) == "adjust":
            replacement = next_label
        elif _label_main_verb(prev_label) == _label_main_verb(next_label):
            replacement = prev_label

        if replacement and replacement != label:
            item["label"] = replacement
            segment_plan[idx] = item
            rewrites += 1
    return rewrites


_ING_TO_BASE_VERB_MAP: Dict[str, str] = {
    "positioning": "position",
    "scraping": "scrape",
    "lifting": "lift",
    "turning": "turn",
    "setting": "set",
    "placing": "place",
    "moving": "move",
    "polishing": "polish",
    "sanding": "sand",
    "leveling": "level",
    "dislodging": "dislodge",
    "adjusting": "adjust",
    "opening": "open",
    "closing": "close",
    "cutting": "cut",
    "pulling": "pull",
    "pushing": "push",
    "holding": "hold",
    "inserting": "insert",
    "removing": "remove",
    "twisting": "twist",
    "pouring": "pour",
    "scooping": "scoop",
    "filling": "fill",
    "compacting": "compact",
}


def _normalize_ing_verbs_to_imperative(text: str) -> str:
    out = text or ""
    if not out:
        return out
    for src, dst in _ING_TO_BASE_VERB_MAP.items():
        out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


_NUM_WORDS_0_TO_19 = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_NUM_TENS_WORDS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]


def _int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n < 20:
        return _NUM_WORDS_0_TO_19[n]
    if n < 100:
        tens, rem = divmod(n, 10)
        return _NUM_TENS_WORDS[tens] if rem == 0 else f"{_NUM_TENS_WORDS[tens]}-{_NUM_WORDS_0_TO_19[rem]}"
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        return (
            f"{_NUM_WORDS_0_TO_19[hundreds]} hundred"
            if rem == 0
            else f"{_NUM_WORDS_0_TO_19[hundreds]} hundred {_int_to_words(rem)}"
        )
    if n < 10000:
        thousands, rem = divmod(n, 1000)
        return (
            f"{_NUM_WORDS_0_TO_19[thousands]} thousand"
            if rem == 0
            else f"{_NUM_WORDS_0_TO_19[thousands]} thousand {_int_to_words(rem)}"
        )
    return str(n)


def _replace_numerals_with_words(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            value = int(token)
        except (TypeError, ValueError):
            return token
        return _int_to_words(value)

    out = re.sub(r"\b\d+\b", repl, text or "")
    return re.sub(r"\s+", " ", out).strip()


def _expand_verb_object_attachment_patterns(text: str) -> str:
    """
    Normalize common chained-verb shorthand into explicit object-attached actions.
    Example: "pick up box and place under desk" -> "pick up box, place box under desk"
    """
    out = text or ""

    def _clean(token: str) -> str:
        return re.sub(r"\s+", " ", (token or "").strip(" ,"))

    def _repl(match: re.Match[str]) -> str:
        obj = _clean(match.group(1))
        prep = _clean(match.group(2)).lower()
        dest = _clean(match.group(3))
        if not obj or not prep or not dest:
            return match.group(0)
        return f"pick up {obj}, place {obj} {prep} {dest}"

    # Case A: object omitted after first verb.
    out = re.sub(
        r"\bpick up\s+and\s+place\s+([^,]+?)\s+(on|in|into|onto|at|to|inside|under|over)\s+([^,]+)",
        _repl,
        out,
        flags=re.IGNORECASE,
    )
    # Case B: object omitted after second verb.
    out = re.sub(
        r"\bpick up\s+([^,]+?)\s+and\s+place\s+(on|in|into|onto|at|to|inside|under|over)\s+([^,]+)",
        _repl,
        out,
        flags=re.IGNORECASE,
    )

    return re.sub(r"\s+", " ", out).strip(" ,")


def _normalize_mechanical_motion_to_goal(text: str) -> str:
    """
    Replace mechanical-motion wording with coarse goal verbs.
    Example: "move handsaw back and forth on wooden board" -> "cut wooden board with handsaw"
    """
    out = text or ""

    def _norm_obj(value: str) -> str:
        obj = re.sub(r"\s+", " ", (value or "").strip(" ,.;:"))
        obj = re.sub(r"^(?:on|onto|across|to|into|in)\s+", "", obj, flags=re.IGNORECASE)
        obj = re.sub(r"^(?:finish|fully)\s+cut(?:ting)?\s+", "", obj, flags=re.IGNORECASE)
        obj = re.sub(r"^cut(?:ting)?\s+", "", obj, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", obj).strip(" ,.;:")

    def _saw_repl(match: re.Match[str]) -> str:
        tool_raw = str(match.group("tool") or "").strip().lower()
        obj = _norm_obj(str(match.group("obj") or ""))
        if not obj:
            return match.group(0)
        tool = "handsaw" if "hand" in tool_raw else "saw"
        return f"cut {obj} with {tool}"

    # move saw/handsaw back and forth [on/across/to cut] <object>
    out = re.sub(
        r"\bmove\s+(?P<tool>hand\s*saw|handsaw|saw)\s+back\s+and\s+forth\s+"
        r"(?:(?:to\s+)?(?:(?:finish|fully)\s+)?cut(?:ting)?\s+)?"
        r"(?:(?:on|onto|across|to|into|in)\s+)?(?P<obj>[^,]+)",
        _saw_repl,
        out,
        flags=re.IGNORECASE,
    )

    def _sand_repl(match: re.Match[str]) -> str:
        obj = _norm_obj(str(match.group("obj") or ""))
        if not obj:
            return match.group(0)
        return f"sand {obj} with sandpaper"

    # move/rub sandpaper [back and forth] on <object>
    out = re.sub(
        r"\b(?:move|rub)\s+sandpaper(?:\s+back\s+and\s+forth)?\s+"
        r"(?:(?:on|onto|across|to|into|in)\s+)?(?P<obj>[^,]+)",
        _sand_repl,
        out,
        flags=re.IGNORECASE,
    )

    def _norm_hair_obj(value: str) -> str:
        obj = _norm_obj(value)
        obj = re.sub(r"\bsection\s+hair\b", "wig", obj, flags=re.IGNORECASE)
        obj = re.sub(r"\bwig\s+hair\b", "wig", obj, flags=re.IGNORECASE)
        obj = re.sub(r"\bhair\b", "wig", obj, flags=re.IGNORECASE)
        obj = re.sub(r"\s+", " ", obj).strip(" ,.;:")
        return obj or "wig"

    def _comb_section_repl(match: re.Match[str]) -> str:
        obj = _norm_hair_obj(str(match.group("obj") or "wig"))
        return f"section {obj} with comb"

    # move comb/tail through wig to section hair -> section wig with comb
    out = re.sub(
        r"\bmove\s+comb(?:\s+tail)?\s+through\s+(?P<obj>[^,]+?)\s+to\s+section(?:\s+hair)?\b",
        _comb_section_repl,
        out,
        flags=re.IGNORECASE,
    )

    def _comb_detangle_repl(match: re.Match[str]) -> str:
        obj = _norm_hair_obj(str(match.group("obj") or "wig"))
        return f"detangle {obj} with comb"

    out = re.sub(
        r"\bmove\s+comb\s+through\s+(?P<obj>[^,]+?)\s+to\s+detangle\b",
        _comb_detangle_repl,
        out,
        flags=re.IGNORECASE,
    )

    def _comb_style_repl(match: re.Match[str]) -> str:
        obj = _norm_hair_obj(str(match.group("obj") or "wig"))
        return f"comb {obj}"

    out = re.sub(
        r"\bmove\s+comb\s+through\s+(?P<obj>[^,]+?)\s+to\s+style\b",
        _comb_style_repl,
        out,
        flags=re.IGNORECASE,
    )

    def _comb_generic_repl(match: re.Match[str]) -> str:
        obj = _norm_hair_obj(str(match.group("obj") or "wig"))
        return f"comb {obj}"

    # move comb through wig -> comb wig
    out = re.sub(
        r"\bmove\s+comb\s+through\s+(?P<obj>[^,]+)\b",
        _comb_generic_repl,
        out,
        flags=re.IGNORECASE,
    )

    def _straightener_repl(match: re.Match[str]) -> str:
        obj = _norm_hair_obj(str(match.group("obj") or "wig"))
        return f"straighten {obj} with hair straightener"

    # move hair straightener to press wig section -> straighten wig with hair straightener
    out = re.sub(
        r"\bmove\s+hair\s+straightener\s+(?:to\s+)?(?:press|straighten)\s+(?P<obj>[^,]+)\b",
        _straightener_repl,
        out,
        flags=re.IGNORECASE,
    )

    return re.sub(r"\s+", " ", out).strip(" ,")


def _collapse_adjacent_duplicate_tokens(text: str) -> str:
    out = re.sub(r"\s+", " ", (text or "").strip())
    if not out:
        return out
    repeated_phrase = re.compile(r"\b([a-z]+(?:\s+[a-z]+){1,2})\s+\1\b", re.IGNORECASE)
    repeated_word = re.compile(r"\b([a-z]+)\s+\1\b", re.IGNORECASE)
    for _ in range(6):
        prev = out
        out = repeated_phrase.sub(r"\1", out)
        out = repeated_word.sub(r"\1", out)
        out = re.sub(r"\s+", " ", out).strip(" ,")
        if out == prev:
            break
    return out


def _rewrite_label_tier3(label: str) -> str:
    text = re.sub(r"\s+", " ", (label or "").strip())
    if not text:
        return text
    if text.lower() == "no action":
        return "No Action"

    # 1. Normalize complex tool/gripper and mechanical terms FIRST
    # (before individual words like 'arm' or 'hand' are stripped)
    text = _normalize_mechanical_motion_to_goal(text)
    text = _normalize_gripper_terms(text)
    
    # 2. Re-standardize spacing
    text = re.sub(r"\s+", " ", text).strip()
    
    # 3. Narrative fillers
    text = re.sub(r"\bthen\b", ",", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnext\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcontinue\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bagain\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\banother\b\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\brotate(?:d|s|ing)?\b", "adjust", text, flags=re.IGNORECASE)
    text = re.sub(r"\bturn(?:ed|s|ing)?\b", "adjust", text, flags=re.IGNORECASE)
    text = re.sub(r"\brelocate(?:d|s|ing)?\b", "move", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgrab(?:bed|s|bing)?\b", "pick up", text, flags=re.IGNORECASE)
    
    # 4. Body-part stripping
    text = re.sub(r"\bwith\s+(?:left|right)?\s*(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\busing\s+(?:left|right)?\s*(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:left|right)\s+(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms|leg|legs|foot|feet|toe|toes)\b", "", text, flags=re.IGNORECASE)
    
    # 5. Other normalizations
    text = _normalize_ing_verbs_to_imperative(text)
    text = _collapse_adjacent_duplicate_tokens(text)
    text = _replace_numerals_with_words(text)
    text = _expand_verb_object_attachment_patterns(text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")

    clauses = [c.strip() for c in text.split(",") if c.strip()]
    if not clauses:
        return text

    # Remove exact duplicate clauses.
    deduped: List[str] = []
    seen: set[str] = set()
    for c in clauses:
        key = re.sub(r"\s+", " ", c).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    clauses = deduped

    return ", ".join(clauses).strip()


def _normalize_label_min_safety(label: str) -> str:
    text = re.sub(r"\s+", " ", (label or "").strip())
    if not text:
        return text
    if text.lower() == "no action":
        return "No Action"
    # Always enforce this minimal safety rewrite before policy gate.
    text = re.sub(r"\brotate(?:d|s|ing)?\b", "adjust", text, flags=re.IGNORECASE)
    text = re.sub(r"\bturn(?:ed|s|ing)?\b", "adjust", text, flags=re.IGNORECASE)
    text = re.sub(r"\brelocate(?:d|s|ing)?\b", "move", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgrab(?:bed|s|bing)?\b", "pick up", text, flags=re.IGNORECASE)
    # Strip body-part references automatically
    text = re.sub(r"\bwith\s+(?:left|right)?\s*(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\busing\s+(?:left|right)?\s*(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:left|right)\s+(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms|leg|legs|foot|feet|toe|toes)\b", "", text, flags=re.IGNORECASE)
    text = _normalize_ing_verbs_to_imperative(text)
    text = _normalize_mechanical_motion_to_goal(text)
    text = _collapse_adjacent_duplicate_tokens(text)
    text = _normalize_gripper_terms(text)
    text = _replace_numerals_with_words(text)
    text = _expand_verb_object_attachment_patterns(text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text


def _normalize_segment_plan(
    payload: Dict[str, Any],
    source_segments: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    items = payload.get("segments")
    if not isinstance(items, list):
        raise ValueError("Gemini payload must contain list at 'segments'")

    effective_cfg = cfg or {}
    forbidden_verbs_raw = _cfg_get(effective_cfg, "run.forbidden_label_verbs", [])
    forbidden_verbs = [str(v).strip().lower() for v in forbidden_verbs_raw if str(v).strip()]
    allowed_verb_token_patterns = _allowed_label_start_verb_token_patterns_from_cfg(effective_cfg)

    source_by_idx: Dict[int, Dict[str, Any]] = {int(seg["segment_index"]): seg for seg in source_segments}
    out: Dict[int, Dict[str, Any]] = {}
    out_idx = 1
    for item in items:
        if not isinstance(item, dict):
            continue
        idx_raw = item.get("segment_index", item.get("index"))
        try:
            idx = int(idx_raw)
        except (TypeError, ValueError):
            idx = out_idx
        source = source_by_idx.get(idx)
        if source is None:
            # Fallback to nearest source or just use the item as is
            source = source_by_idx.get(out_idx, {})
        source_label = str(source.get("current_label", "")).strip()
        label = str(item.get("label", "")).strip() or source_label
        if bool(_cfg_get(effective_cfg, "run.tier3_label_rewrite", True)):
            label = _rewrite_label_tier3(label)
        label = _autofix_label_candidate(
            effective_cfg,
            label,
            source_label,
            forbidden_verbs,
            allowed_verb_token_patterns,
        )
        label = _normalize_label_min_safety(label)
        start_src = _safe_float(source.get("start_sec", 0.0), 0.0)
        end_src = _safe_float(source.get("end_sec", 0.0), 0.0)
        start_sec = _safe_float(item.get("start_sec", start_src), start_src)
        end_sec = _safe_float(item.get("end_sec", end_src), end_src)
        if end_sec <= start_sec:
            start_sec = start_src
            end_sec = end_src
        # Clamp Gemini timestamps to source ?? max_drift to prevent wild spans
        max_drift = 12.0
        if abs(start_sec - start_src) > max_drift:
            start_sec = start_src
        if abs(end_sec - end_src) > max_drift:
            end_sec = end_src
        out[out_idx] = {
            "segment_index": out_idx,
            "label": label,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
        }
        out_idx += 1

    # If Gemini dropped segments, append remaining original segments to keep length consistent
    while out_idx <= len(source_by_idx):
        source = source_by_idx.get(out_idx, {})
        source_label = str(source.get("current_label", "")).strip()
        if bool(_cfg_get(effective_cfg, "run.tier3_label_rewrite", True)):
            source_label = _rewrite_label_tier3(source_label)
        source_label = _autofix_label_candidate(
            effective_cfg,
            source_label,
            source_label,
            forbidden_verbs,
            allowed_verb_token_patterns,
        )
        source_label = _normalize_label_min_safety(source_label)
        out[out_idx] = {
            "segment_index": out_idx,
            "label": source_label,
            "start_sec": round(_safe_float(source.get("start_sec", 0.0), 0.0), 3),
            "end_sec": round(_safe_float(source.get("end_sec", 0.0), 0.0), 3),
        }
        out_idx += 1

    if not out:
        raise ValueError("Gemini returned no usable segment plan")
    return out


def _normalize_label_map_from_plan(segment_plan: Dict[int, Dict[str, Any]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for idx, item in segment_plan.items():
        label = str(item.get("label", "")).strip()
        if label:
            out[idx] = label
    if not out:
        raise ValueError("Segment plan has no usable labels")
    return out


def _first_visible_child_locator(row: Locator, selector: str, max_scan: int = 10) -> Optional[Locator]:
    for candidate in _selector_variants(selector):
        try:
            loc = row.locator(candidate)
            count = min(loc.count(), max_scan)
            for i in range(count):
                item = loc.nth(i)
                if item.is_visible() and item.is_enabled():
                    return item
        except Exception:
            continue
    return None


from src.solver import segments as _segments
from src.solver import video as _video

_looks_like_video_url = _video._looks_like_video_url
_collect_video_url_candidates = _video._collect_video_url_candidates
_download_video_via_playwright_request = _video._download_video_via_playwright_request
_download_video_from_page_context = _video._download_video_from_page_context
_is_probably_mp4 = _video._is_probably_mp4
_is_video_decodable = _video._is_video_decodable
_ensure_loop_off = _video._ensure_loop_off
_play_full_video_once = _video._play_full_video_once
_prepare_video_for_gemini = _video._prepare_video_for_gemini
_parse_mmss_to_seconds = _segments._parse_mmss_to_seconds
_extract_start_end_from_text = _segments._extract_start_end_from_text
_resolve_rows_locator = _segments._resolve_rows_locator
_first_text_from_row = _segments._first_text_from_row
extract_segments = _segments.extract_segments
_normalize_operation_action = _segments._normalize_operation_action
_normalize_operations = _segments._normalize_operations
_normalize_segment_plan = _labels._normalize_segment_plan
_normalize_label_map_from_plan = _labels._normalize_label_map_from_plan
_first_visible_child_locator = _segments._first_visible_child_locator
_respect_major_step_pause = _segments._respect_major_step_pause
_short_error_text = _segments._short_error_text
_submit_transition_observed = _segments._submit_transition_observed
apply_timestamp_adjustments = _segments.apply_timestamp_adjustments
_action_selector_for_row = _segments._action_selector_for_row
_action_hotkey = _segments._action_hotkey
_confirm_action_dialog = _segments._confirm_action_dialog
_wait_rows_delta = _segments._wait_rows_delta
apply_segment_operations = _segments.apply_segment_operations
_fill_input = _segments._fill_input
_normalize_label_for_compare = _segments._normalize_label_for_compare
_filter_unchanged_label_map = _segments._filter_unchanged_label_map
_handle_quality_review_modal = _segments._handle_quality_review_modal
_handle_no_edits_modal = _segments._handle_no_edits_modal
_submit_episode = _segments._submit_episode
apply_labels = _segments.apply_labels


def _apply_global_gemini_video_policy(cfg: Dict[str, Any]) -> None:
    """
    Enforce a low-cost + quality-preserving video policy for all accounts.
    This runs after YAML merge so old/new account files get consistent behavior.
    """
    gem = cfg.setdefault("gemini", {})
    if not isinstance(gem, dict):
        cfg["gemini"] = {}
        gem = cfg["gemini"]

    defaults = {
        "optimize_video_for_upload": True,
        "optimize_video_target_mb": 4.0,
        "optimize_video_target_fps": 10.0,
        "optimize_video_min_fps": 8.0,
        "optimize_video_min_width": 320,
        "optimize_video_min_short_side": 320,
        "split_upload_enabled": True,
        "split_upload_only_if_larger_mb": 8.0,
        "split_upload_chunk_max_mb": 6.0,
        "split_upload_max_chunks": 4,
        "split_upload_reencode_on_copy_fail": True,
        "split_upload_inline_total_max_mb": 12.0,
        "reference_frames_enabled": True,
        "reference_frames_always": False,
        "reference_frame_attach_when_video_mb_le": 2.5,
        "reference_frame_count": 2,
        "reference_frame_positions": [0.2, 0.55, 0.85],
        "reference_frame_max_side": 960,
        "reference_frame_jpeg_quality": 82,
        "reference_frame_max_total_kb": 420,
        "video_transport": "files_api",
        "files_api_fallback_to_inline": False,
        "retry_with_quota_fallback_model": False,
        "quota_fallback_model": "gemini-3.1-pro-preview",
        "quota_fallback_from_models": [],
        "policy_retry_model": "gemini-2.5-flash",
        "retry_with_stronger_model_on_policy_fail": False,
        "stage_models": {
            "labeling": "gemini-2.5-flash",
            "repair": "gemini-2.5-flash",
            "policy_retry": "gemini-2.5-flash",
            "compare_api": "gemini-3.1-pro-preview",
            "compare_chat": "gemini-3.1-pro-preview",
        },
    }
    for key, value in defaults.items():
        gem.setdefault(key, value)

    preferred_model = "gemini-3.1-pro-preview"
    legacy_model_values = {"", "gemini-2.5-pro", "gemini-3-pro-preview"}
    configured_model = str(gem.get("model", "") or "").strip()
    if configured_model.lower() in legacy_model_values:
        if configured_model != preferred_model:
            gem["model"] = preferred_model
            print(f"[policy] gemini.model forced to {preferred_model}.")

    configured_policy_retry_model = str(gem.get("policy_retry_model", "") or "").strip()
    if configured_policy_retry_model.lower() in legacy_model_values:
        gem["policy_retry_model"] = preferred_model

    configured_quota_fallback_model = str(gem.get("quota_fallback_model", "") or "").strip()
    if configured_quota_fallback_model.lower() in legacy_model_values:
        gem["quota_fallback_model"] = preferred_model
    normalized_model = str(gem.get("model", preferred_model) or preferred_model).strip() or preferred_model
    normalized_quota_fallback_model = str(gem.get("quota_fallback_model", preferred_model) or preferred_model).strip() or preferred_model

    if str(gem.get("policy_retry_model", "") or "").strip().lower() == preferred_model.lower():
        gem["retry_with_stronger_model_on_policy_fail"] = False
    quota_fallback_from_models = gem.get("quota_fallback_from_models", [])
    normalized_from_models = {
        str(item or "").strip().lower()
        for item in (quota_fallback_from_models if isinstance(quota_fallback_from_models, list) else [quota_fallback_from_models])
        if str(item or "").strip()
    }
    if (
        not normalized_from_models
        or normalized_from_models == {normalized_quota_fallback_model.lower()}
    ):
        gem["quota_fallback_from_models"] = [normalized_model]
    if normalized_model.lower() == normalized_quota_fallback_model.lower():
        gem["retry_with_quota_fallback_model"] = False

    gem["files_api_fallback_to_inline"] = bool(gem.get("files_api_fallback_to_inline", False))

    # Keep upload cost low while avoiding aggressive visual degradation.
    split_upload_enabled = bool(gem.get("split_upload_enabled", True))
    target_cap_mb = 50.0
    try:
        target_mb = float(gem.get("optimize_video_target_mb", 15.0))
    except Exception:
        target_mb = 15.0
    gem["optimize_video_target_mb"] = min(target_cap_mb, max(1.0, target_mb))

    try:
        target_fps = float(gem.get("optimize_video_target_fps", 10.0))
    except Exception:
        target_fps = 10.0
    try:
        min_fps = float(gem.get("optimize_video_min_fps", 8.0))
    except Exception:
        min_fps = 8.0
    min_fps = max(8.0, min_fps)
    gem["optimize_video_min_fps"] = min_fps
    gem["optimize_video_target_fps"] = max(min_fps, target_fps)

    try:
        min_width = int(gem.get("optimize_video_min_width", 320))
    except Exception:
        min_width = 320
    try:
        min_short = int(gem.get("optimize_video_min_short_side", 320))
    except Exception:
        min_short = 320
    gem["optimize_video_min_width"] = max(320, min_width)
    gem["optimize_video_min_short_side"] = max(320, min_short)

    floor_surface_guard = (
        "If floor mat vs table is unclear, do not guess raised furniture; "
        "use neutral location wording."
    )
    extra = str(gem.get("extra_instructions", "") or "").strip()
    if floor_surface_guard.lower() not in extra.lower():
        gem["extra_instructions"] = (
            f"{extra}\n{floor_surface_guard}".strip() if extra else floor_surface_guard
        )


def _apply_global_run_policy(cfg: Dict[str, Any]) -> None:
    """
    Enforce safe run-level defaults that prevent known quality failures
    across older account YAML files.
    """
    run = cfg.setdefault("run", {})
    if not isinstance(run, dict):
        cfg["run"] = {}
        run = cfg["run"]

    run.setdefault("auto_continuity_merge_enabled", True)
    run.setdefault("auto_continuity_merge_min_run_segments", 3)
    run.setdefault("auto_continuity_merge_min_token_overlap", 1)
    run.setdefault("segment_chunking_min_video_sec", 60.0)
    run["skip_reserve_when_all_visible_blocked"] = False
    run["clear_blocked_tasks_after_all_visible_blocked_hits"] = 1
    run["clear_blocked_tasks_every_retry"] = True
    run["reserve_cooldown_sec"] = 0
    run["reserve_min_interval_sec"] = 0
    run["reserve_wait_only_on_rate_limit"] = True
    run["reserve_attempts_per_visit"] = max(3, int(run.get("reserve_attempts_per_visit", 3) or 3))
    run["reserve_rate_limit_wait_sec"] = 5
    run["release_and_reserve_on_all_visible_blocked"] = True
    run["release_and_reserve_on_submit_unverified"] = True
    run["no_task_retry_delay_sec"] = 5.0
    run["no_task_backoff_factor"] = 1.0
    run["no_task_max_delay_sec"] = 5.0
    run["keep_alive_idle_cycle_pause_sec"] = 5.0
    run["release_all_wait_sec"] = 5.0

    # Respect explicit structural_allow_merge=false from config.
    # Auto-continuity merge has its own enable flag now.


_read_optional_text_file = _prompting._read_optional_text_file
_resolve_system_instruction = _prompting._resolve_system_instruction
build_prompt = _prompting.build_prompt
_consistency_norm = _prompting._consistency_norm
_consistency_tokens = _prompting._consistency_tokens
_extract_consistency_terms_from_label = _prompting._extract_consistency_terms_from_label
_find_equivalent_canonical_term = _prompting._find_equivalent_canonical_term
_apply_consistency_aliases_to_label = _prompting._apply_consistency_aliases_to_label
_update_chunk_consistency_memory = _prompting._update_chunk_consistency_memory
_build_chunk_consistency_prompt_hint = _prompting._build_chunk_consistency_prompt_hint

_count_atomic_actions_in_label = _labels._count_atomic_actions_in_label
_normalize_gripper_terms = _labels._normalize_gripper_terms
_label_main_verb = _labels._label_main_verb
_is_no_action_label = _labels._is_no_action_label
_label_content_tokens = _labels._label_content_tokens
_allowed_label_start_verb_token_patterns_from_cfg = _labels._allowed_label_start_verb_token_patterns_from_cfg
_label_starts_with_allowed_action_verb = _labels._label_starts_with_allowed_action_verb
_contains_forbidden_verb_in_label = _labels._contains_forbidden_verb_in_label
_strip_forbidden_verbs_for_autofix = _labels._strip_forbidden_verbs_for_autofix
_action_phrase_missing_object_for_autofix = _labels._action_phrase_missing_object_for_autofix
_heuristic_autofix_verb_from_text = _labels._heuristic_autofix_verb_from_text
_autofix_label_candidate = _labels._autofix_label_candidate
_label_action_clauses = _labels._label_action_clauses
_label_goal_key = _labels._label_goal_key
_build_auto_continuity_merge_operations = _labels._build_auto_continuity_merge_operations
_rewrite_no_action_pauses_in_plan = _labels._rewrite_no_action_pauses_in_plan
_normalize_ing_verbs_to_imperative = _labels._normalize_ing_verbs_to_imperative
_int_to_words = _labels._int_to_words
_replace_numerals_with_words = _labels._replace_numerals_with_words
_expand_verb_object_attachment_patterns = _labels._expand_verb_object_attachment_patterns
_normalize_mechanical_motion_to_goal = _labels._normalize_mechanical_motion_to_goal
_collapse_adjacent_duplicate_tokens = _labels._collapse_adjacent_duplicate_tokens
_rewrite_label_tier3 = _labels._rewrite_label_tier3
_normalize_label_min_safety = _labels._normalize_label_min_safety

_validate_segment_plan_against_policy = _policy_gate._validate_segment_plan_against_policy
_is_timestamp_policy_error = _policy_gate._is_timestamp_policy_error
_is_no_action_policy_error = _policy_gate._is_no_action_policy_error
_maybe_run_pre_submit_chat_compare = _pre_submit_compare.maybe_run_pre_submit_chat_compare


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be YAML object")
    cfg = _deep_merge(DEFAULT_CONFIG, raw)
    # Wire selectors.yaml overrides (Sprint 10: Adaptive Selector Engine)
    selector_overrides = _load_selectors_yaml(str(path.with_name("selectors.yaml")))
    if selector_overrides:
        atlas_cfg = cfg.setdefault("atlas", {})
        selectors_cfg = atlas_cfg.setdefault("selectors", {})
        if isinstance(selectors_cfg, dict):
            selectors_cfg.update(selector_overrides)
    _apply_global_gemini_video_policy(cfg)
    _apply_global_run_policy(cfg)
    return cfg


def _launch_playwright_browser(
    pw: Any,
    *,
    headless: bool,
    slow_mo: int,
    chrome_channel: str,
    browser_executable_path: str = "",
    browser_proxy: Optional[Dict[str, str]] = None,
    launch_label: str = "browser",
) -> Any:
    launch_kwargs: Dict[str, Any] = {
        "headless": headless,
        "slow_mo": slow_mo,
        "args": ["--disable-blink-features=AutomationControlled"],
    }
    if browser_proxy:
        launch_kwargs["proxy"] = dict(browser_proxy)
    if browser_executable_path:
        launch_kwargs["executable_path"] = browser_executable_path
    else:
        launch_kwargs["channel"] = chrome_channel
    browser = pw.chromium.launch(**launch_kwargs)
    print(f"[browser] launched {launch_label} via Playwright fallback.")
    return browser


def _cdp_endpoint_ready(cdp_url: str, *, timeout_sec: float = 1.5) -> bool:
    target = str(cdp_url or "").strip().rstrip("/")
    if not target:
        return False
    try:
        resp = requests.get(f"{target}/json/version", timeout=max(0.2, float(timeout_sec)))
        if not resp.ok:
            return False
        payload = resp.json()
    except Exception:
        return False
    return isinstance(payload, dict) and bool(str(payload.get("Browser", "")).strip())


def _resolve_local_chrome_executable(
    *,
    browser_executable_path: str,
    chrome_channel: str,
) -> str:
    explicit = str(browser_executable_path or "").strip()
    if explicit:
        resolved = shutil.which(explicit) or explicit
        if Path(resolved).exists():
            return resolved

    candidates: List[str] = []
    channel = str(chrome_channel or "").strip().lower()
    if os.name == "nt":
        candidates.extend(
            [
                str(Path(os.environ.get("ProgramFiles", "")) / "Google" / "Chrome" / "Application" / "chrome.exe"),
                str(Path(os.environ.get("ProgramFiles(x86)", "")) / "Google" / "Chrome" / "Application" / "chrome.exe"),
                str(Path(os.environ.get("LocalAppData", "")) / "Google" / "Chrome" / "Application" / "chrome.exe"),
                str(Path(os.environ.get("ProgramFiles", "")) / "Chromium" / "Application" / "chrome.exe"),
            ]
        )
    else:
        channel_bins: List[str] = []
        if channel:
            channel_bins.append(channel)
        channel_bins.extend(["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"])
        for name in channel_bins:
            resolved = shutil.which(name)
            if resolved:
                candidates.append(resolved)

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return ""


def _launch_local_shared_chrome_cdp(
    *,
    cdp_url: str,
    browser_executable_path: str,
    chrome_channel: str,
    use_chrome_profile: bool,
    chrome_user_data_dir: str,
    chrome_profile_directory: str,
    clone_chrome_profile_to_temp: bool,
    cloned_user_data_dir: str,
    reuse_existing_cloned_profile: bool,
    prefer_profile_with_atlas_cookies: bool,
    atlas_email: str,
    close_chrome_before_profile_launch: bool,
    profile_launch_timeout_ms: int,
    start_urls: List[str],
) -> bool:
    if _cdp_endpoint_ready(cdp_url):
        return False

    chrome_path = _resolve_local_chrome_executable(
        browser_executable_path=browser_executable_path,
        chrome_channel=chrome_channel,
    )
    if not chrome_path:
        print("[browser] shared CDP bootstrap skipped: no local Chrome executable found.")
        return False

    parsed = urlparse(str(cdp_url or "").strip())
    port = parsed.port
    if port is None:
        print(f"[browser] shared CDP bootstrap skipped: invalid CDP url {cdp_url!r}")
        return False

    launch_user_data_dir = str(chrome_user_data_dir or "").strip()
    launch_profile_directory = str(chrome_profile_directory or "").strip()
    if use_chrome_profile and launch_user_data_dir and clone_chrome_profile_to_temp:
        detected_profile = ""
        if atlas_email:
            detected_profile = _detect_chrome_profile_for_email(launch_user_data_dir, atlas_email)
        if prefer_profile_with_atlas_cookies:
            cookie_profile = _detect_chrome_profile_for_site_cookie(launch_user_data_dir)
            if cookie_profile:
                detected_profile = cookie_profile
        if detected_profile:
            launch_profile_directory = detected_profile
        if not launch_profile_directory:
            launch_profile_directory = "Default"
        try:
            launch_user_data_dir = _prepare_chrome_profile_clone(
                launch_user_data_dir,
                launch_profile_directory,
                cloned_user_data_dir,
                reuse_existing=reuse_existing_cloned_profile,
            )
        except Exception as exc:
            print(f"[browser] shared CDP bootstrap clone failed; using source profile directly: {exc}")

    if close_chrome_before_profile_launch:
        _close_chrome_processes()
        time.sleep(1.5)

    urls = [str(item or "").strip() for item in start_urls if str(item or "").strip()]
    if not urls:
        urls = ["about:blank"]

    args = [
        f"--remote-debugging-port={port}",
        "--disable-blink-features=AutomationControlled",
        "--no-first-run",
        "--disable-default-apps",
        "--new-window",
    ]
    if launch_user_data_dir:
        args.append(f"--user-data-dir={launch_user_data_dir}")
    if launch_profile_directory:
        args.append(f"--profile-directory={launch_profile_directory}")
    args.extend(urls)

    popen_kwargs: Dict[str, Any] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        creationflags = 0
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        if creationflags:
            popen_kwargs["creationflags"] = creationflags
    else:
        popen_kwargs["start_new_session"] = True

    subprocess.Popen([chrome_path, *args], **popen_kwargs)
    wait_deadline = time.time() + max(5.0, float(profile_launch_timeout_ms) / 1000.0)
    while time.time() < wait_deadline:
        if _cdp_endpoint_ready(cdp_url, timeout_sec=1.0):
            print(
                "[browser] launched shared Chrome CDP bootstrap."
                f" port={port} user_data_dir={launch_user_data_dir or '(default)'}"
            )
            return True
        time.sleep(1.0)

    print(f"[browser] shared CDP bootstrap launched but {cdp_url} did not become ready in time.")
    return False


def _connect_atlas_browser_context(
    pw: Any,
    *,
    cdp_url: str,
    cdp_connect_timeout_ms: int,
    state_path: Path,
    headless: bool,
    slow_mo: int,
    chrome_channel: str,
    browser_executable_path: str = "",
    browser_proxy: Optional[Dict[str, str]] = None,
) -> tuple[Any, Any, Page, str]:
    def _is_closeable_bootstrap_page(candidate: Any) -> bool:
        try:
            url = str(getattr(candidate, "url", "") or "").strip().lower()
        except Exception:
            return False
        if not url:
            return False
        if "gemini.google.com" in url:
            return False
        return url.startswith("http://") or url.startswith("https://") or url == "about:blank"

    last_conn_exc = None
    for attempt in range(3):
        print(f"[browser] connecting over CDP (attempt {attempt+1}/3): {cdp_url}")
        try:
            browser = pw.chromium.connect_over_cdp(cdp_url, timeout=cdp_connect_timeout_ms)
            last_conn_exc = None
            if browser.contexts:
                context = browser.contexts[0]
            else:
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    viewport={"width": 1920, "height": 1080}
                )
            print("[browser] CDP connection established.")
            if context.pages:
                pages_snapshot = list(context.pages)
                page = None
                for candidate in pages_snapshot:
                    url = str(getattr(candidate, "url", "") or "").strip().lower()
                    if "audit.atlascapture.io" in url:
                        page = candidate
                        break
                if page is None:
                    for candidate in pages_snapshot:
                        url = str(getattr(candidate, "url", "") or "").strip().lower()
                        if "gemini.google.com" in url:
                            continue
                        if url.startswith("http://") or url.startswith("https://"):
                            page = candidate
                            break
                if page is not None:
                    try:
                        # Force Desktop User Agent
                        page.set_extra_http_headers({
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                        })
                        context.set_viewport_size({"width": 1920, "height": 1080})
                        # Force reload to clear mobile overlays with new UA
                        page.reload(wait_until="domcontentloaded")
                        page.wait_for_timeout(1000)
                    except Exception:
                        pass
                    for candidate in list(context.pages):
                        if candidate is page:
                            continue
                        if not _is_closeable_bootstrap_page(candidate):
                            continue
                        try:
                            candidate.close()
                        except Exception:
                            pass
                    print(f"[browser] reused and cleaned up window. Primary tab: {page.url}")
                else:
                    page = context.new_page()
                    try:
                        page.set_viewport_size({"width": 1920, "height": 1080})
                    except Exception:
                        pass
                    for candidate in list(pages_snapshot):
                        if candidate is page:
                            continue
                        if not _is_closeable_bootstrap_page(candidate):
                            continue
                        try:
                            candidate.close()
                        except Exception:
                            pass
                    print("[browser] only internal tabs were present; opened fresh tab in window.")
            else:
                page = context.new_page()
                try:
                    page.set_viewport_size({"width": 1920, "height": 1080})
                except Exception:
                    pass
                print("[browser] opened fresh tab in window.")
            print(f"[browser] attached to page: {page.url}")
            return browser, context, page, "cdp"
        except Exception as exc:
            last_conn_exc = exc
            if attempt < 2:
                print(f"[browser] CDP connection failed: {exc}. Retrying in 5s...")
                time.sleep(5)

    print(
        f"[browser] CDP unavailable after retries; launching Atlas via Playwright fallback: {last_conn_exc}"
    )
    browser = _launch_playwright_browser(
        pw,
        headless=headless,
        slow_mo=slow_mo,
        chrome_channel=chrome_channel,
        browser_executable_path=browser_executable_path,
        browser_proxy=browser_proxy,
        launch_label="Atlas browser",
    )
    context_kwargs: Dict[str, Any] = {}
    if state_path.exists():
        context_kwargs["storage_state"] = str(state_path)
    context = browser.new_context(**context_kwargs)
    page = context.new_page()
    print(
        "[browser] Atlas Playwright fallback ready."
        f" storage_state={'yes' if state_path.exists() else 'no'}"
    )
    return browser, context, page, "playwright_fallback"


def _cleanup_browser_connections(
    *,
    context: Any,
    browser: Any,
    atlas_browser_mode: str,
    gemini_browser: Any,
    owns_gemini_browser: bool,
) -> None:
    shared_atlas_cdp = str(atlas_browser_mode or "").strip().lower() == "cdp"
    if shared_atlas_cdp:
        print("[browser] preserving shared Atlas CDP browser/context during cleanup.")
    else:
        try:
            if context is not None:
                context.close()
        except Exception as exc:
            print(f"[browser] ignore context close error during cleanup: {exc}")
        if browser is not None:
            try:
                browser.close()
            except Exception as exc:
                print(f"[browser] ignore browser close error during cleanup: {exc}")
    if gemini_browser is not None and gemini_browser is not browser and owns_gemini_browser:
        try:
            gemini_browser.close()
        except Exception as exc:
            print(f"[browser] ignore Gemini browser close error during cleanup: {exc}")


def run(cfg: Dict[str, Any], execute: bool) -> None:
    global _GEMINI_FALLBACK_USES
    _GEMINI_FALLBACK_USES = 0
    gemini_uploaded_file_names: List[str] = []
    state_path = Path(str(_cfg_get(cfg, "browser.storage_state_path", ".state/atlas_auth.json")))
    force_login = bool(_cfg_get(cfg, "browser.force_login", False))
    headless = bool(_cfg_get(cfg, "browser.headless", False))
    slow_mo = int(_cfg_get(cfg, "browser.slow_mo_ms", 0))
    use_chrome_profile = bool(_cfg_get(cfg, "browser.use_chrome_profile", False))
    restore_state_in_profile_mode = bool(_cfg_get(cfg, "browser.restore_state_in_profile_mode", False))
    fallback_on_profile_error = bool(
        _cfg_get(cfg, "browser.fallback_to_isolated_context_on_profile_error", True)
    )
    profile_launch_timeout_ms = int(_cfg_get(cfg, "browser.profile_launch_timeout_ms", 30000))
    close_chrome_before_profile_launch = bool(
        _cfg_get(cfg, "browser.close_chrome_before_profile_launch", False)
    )
    profile_launch_retry_count = max(0, int(_cfg_get(cfg, "browser.profile_launch_retry_count", 1)))
    profile_launch_retry_delay_sec = max(0.2, float(_cfg_get(cfg, "browser.profile_launch_retry_delay_sec", 2.0)))
    clone_chrome_profile_to_temp = bool(_cfg_get(cfg, "browser.clone_chrome_profile_to_temp", True))
    reuse_existing_cloned_profile = bool(_cfg_get(cfg, "browser.reuse_existing_cloned_profile", True))
    prefer_profile_with_atlas_cookies = bool(_cfg_get(cfg, "browser.prefer_profile_with_atlas_cookies", True))
    cloned_user_data_dir = str(_cfg_get(cfg, "browser.cloned_user_data_dir", ".state/chrome_user_data_clone")).strip()
    chrome_channel = str(_cfg_get(cfg, "browser.chrome_channel", "chrome")).strip() or "chrome"
    browser_executable_path_raw = (
        str(_cfg_get(cfg, "browser.executable_path", "")).strip()
        or os.environ.get("BROWSER_EXECUTABLE_PATH", "").strip()
    )
    browser_executable_path = ""
    if browser_executable_path_raw:
        browser_executable_path = shutil.which(browser_executable_path_raw) or browser_executable_path_raw
    chrome_user_data_dir = (
        str(_cfg_get(cfg, "browser.chrome_user_data_dir", "")).strip()
        or os.environ.get("CHROME_USER_DATA_DIR", "").strip()
        or _default_chrome_user_data_dir()
    )
    chrome_profile_directory_raw = (
        str(_cfg_get(cfg, "browser.chrome_profile_directory", "Default")).strip()
        or os.environ.get("CHROME_PROFILE_DIRECTORY", "").strip()
    )
    if chrome_profile_directory_raw.lower() in {"none", "direct", "no_profile_arg", "-"}:
        chrome_profile_directory = ""
    else:
        chrome_profile_directory = chrome_profile_directory_raw or "Default"
    proxy_server_raw = (
        str(_cfg_get(cfg, "browser.proxy_server", "")).strip()
        or os.environ.get("ATLAS_PROXY_SERVER", "").strip()
    )
    proxy_username = (
        str(_cfg_get(cfg, "browser.proxy_username", "")).strip()
        or os.environ.get("ATLAS_PROXY_USERNAME", "").strip()
    )
    proxy_password = (
        str(_cfg_get(cfg, "browser.proxy_password", "")).strip()
        or os.environ.get("ATLAS_PROXY_PASSWORD", "").strip()
    )
    proxy_bypass = (
        str(_cfg_get(cfg, "browser.proxy_bypass", "")).strip()
        or os.environ.get("ATLAS_PROXY_BYPASS", "").strip()
    )
    clear_env_proxy_for_backend_requests = bool(
        _cfg_get(cfg, "browser.clear_env_proxy_for_backend_requests", True)
    )
    browser_proxy: Optional[Dict[str, str]] = None
    if proxy_server_raw:
        proxy_server = proxy_server_raw if "://" in proxy_server_raw else f"http://{proxy_server_raw}"
        browser_proxy = {"server": proxy_server}
        if proxy_username:
            browser_proxy["username"] = proxy_username
        if proxy_password:
            browser_proxy["password"] = proxy_password
        if proxy_bypass:
            browser_proxy["bypass"] = proxy_bypass
        print(f"[browser] proxy enabled: {proxy_server_raw} (auth={'yes' if proxy_username else 'no'})")
    if clear_env_proxy_for_backend_requests:
        cleared_proxy_env: List[str] = []
        for env_name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            if os.environ.pop(env_name, None):
                cleared_proxy_env.append(env_name)
        if cleared_proxy_env:
            print(
                "[net] cleared env proxy vars for backend requests: "
                + ", ".join(cleared_proxy_env)
            )
    atlas_email = _resolve_atlas_email(cfg)

    dry_run = bool(_cfg_get(cfg, "run.dry_run", True))
    if execute:
        dry_run = False

    # Production Hardening: Exclusive CDP Connection
    cdp_url = f"http://127.0.0.1:9222"
    cdp_connect_timeout_ms = 45000
    gemini_cdp_url = str(
        _cfg_get(cfg, "gemini.chat_web_connect_over_cdp_url", "")
        or os.environ.get("GEMINI_CHAT_CONNECT_OVER_CDP_URL", "")
        or ""
    ).strip()
    strict_single_session_cfg = bool(_cfg_get(cfg, "run.strict_single_chat_session", False))
    shared_cdp_start_urls: List[str] = []
    room_url_hint = str(_cfg_get(cfg, "atlas.room_url", "") or "").strip()
    if room_url_hint:
        shared_cdp_start_urls.append(room_url_hint)
    if strict_single_session_cfg and gemini_cdp_url and gemini_cdp_url == cdp_url:
        gemini_url_hint = str(
            _cfg_get(cfg, "gemini.chat_web_url", "https://gemini.google.com/app") or ""
        ).strip()
        if gemini_url_hint:
            shared_cdp_start_urls.append(gemini_url_hint)
    if use_chrome_profile and not _cdp_endpoint_ready(cdp_url):
        _launch_local_shared_chrome_cdp(
            cdp_url=cdp_url,
            browser_executable_path=browser_executable_path,
            chrome_channel=chrome_channel,
            use_chrome_profile=use_chrome_profile,
            chrome_user_data_dir=chrome_user_data_dir,
            chrome_profile_directory=chrome_profile_directory,
            clone_chrome_profile_to_temp=clone_chrome_profile_to_temp,
            cloned_user_data_dir=cloned_user_data_dir,
            reuse_existing_cloned_profile=reuse_existing_cloned_profile,
            prefer_profile_with_atlas_cookies=prefer_profile_with_atlas_cookies,
            atlas_email=atlas_email,
            close_chrome_before_profile_launch=close_chrome_before_profile_launch,
            profile_launch_timeout_ms=profile_launch_timeout_ms,
            start_urls=shared_cdp_start_urls,
        )
    
    with sync_playwright() as pw:
        browser = None
        gemini_browser = None
        context = None
        atlas_browser_mode = "unknown"
        owns_gemini_browser = False
        release_all_after_batch = bool(_cfg_get(cfg, "run.release_all_after_batch", True))
        disable_release_all_during_canary = bool(
            _cfg_get(cfg, "run.disable_release_all_during_canary", False)
        )
        sticky_episode_resume = bool(_cfg_get(cfg, "run.sticky_episode_resume", False))
        single_window_two_tabs = bool(_cfg_get(cfg, "run.single_window_two_tabs", False))
        single_window_single_tab = bool(_cfg_get(cfg, "run.single_window_single_tab", False))
        browser, context, page, atlas_browser_mode = _connect_atlas_browser_context(
            pw,
            cdp_url=cdp_url,
            cdp_connect_timeout_ms=cdp_connect_timeout_ms,
            state_path=state_path,
            headless=headless,
            slow_mo=slow_mo,
            chrome_channel=chrome_channel,
            browser_executable_path=browser_executable_path,
            browser_proxy=browser_proxy,
        )
        print(f"[browser] startup mode: {atlas_browser_mode}")
        bootstrap_context = context
        bootstrap_page = page

        if (
            bool(_cfg_get(cfg, "run.use_episode_runtime_v2", False))
            and bool(_cfg_get(cfg, "run.strict_single_chat_session", False))
            and gemini_cdp_url
        ):
            try:
                print(f"[runtime] connecting Gemini over CDP: {gemini_cdp_url}")
                gemini_browser = pw.chromium.connect_over_cdp(gemini_cdp_url, timeout=cdp_connect_timeout_ms)
                print("[runtime] Gemini CDP connection established.")
            except Exception as exc:
                if bool(_cfg_get(cfg, "gemini.chat_web_require_authenticated_session", False)):
                    raise RuntimeError(
                        f"Gemini CDP connection failed and guest fallback is disabled: {exc}"
                    ) from exc
                try:
                    gemini_browser = _launch_playwright_browser(
                        pw,
                        headless=headless,
                        slow_mo=slow_mo,
                        chrome_channel=chrome_channel,
                        browser_executable_path=browser_executable_path,
                        browser_proxy=browser_proxy,
                        launch_label="Gemini browser",
                    )
                    owns_gemini_browser = True
                    print(
                        "[runtime] Gemini CDP unavailable; using Playwright fallback with "
                        "storage-state-backed episode contexts."
                    )
                except Exception as fallback_exc:
                    gemini_browser = None
                    print(
                        "[runtime] warning: Gemini CDP connection failed; "
                        f"falling back to storage-state context only: {exc}; "
                        f"fallback launch error: {fallback_exc}"
                    )

        global _ACTIVE_HEARTBEAT_CALLBACK
        try:
            room_url = str(_cfg_get(cfg, "atlas.room_url", "")).strip()
            target_task_urls = _normalize_target_task_urls(_cfg_get(cfg, "run.target_task_urls", []))
            if sticky_episode_resume and not target_task_urls:
                sticky_state_files = sorted(
                    Path(str(_cfg_get(cfg, "run.output_dir", "outputs"))).glob("task_state_*.json"),
                    key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
                    reverse=True,
                )
                for sticky_path in sticky_state_files:
                    try:
                        sticky_state = json.loads(sticky_path.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if not _can_resume_sticky_task_state(sticky_state):
                        continue
                    sticky_url = str(sticky_state.get("task_url", "") or "").strip()
                    sticky_task_id = str(sticky_state.get("task_id", "") or "").strip()
                    if sticky_url and sticky_task_id:
                        target_task_urls = [sticky_url]
                        print(f"[run] sticky resume active: task={sticky_task_id} url={sticky_url}")
                        break
            if room_url:
                print(f"[run] opening room url: {room_url}")
                try:
                    _goto_with_retry(page, room_url, wait_until="commit", timeout_ms=45000, cfg=cfg, reason="initial-room")
                    print(f"[run] page after room goto: {page.url}")
                except Exception as exc:
                    if _is_too_many_redirects_error(exc):
                        print("[run] room redirect loop detected; clearing Atlas session and retrying room once.")
                        _clear_atlas_site_session(page)
                        try:
                            _goto_with_retry(page, room_url, wait_until="commit", timeout_ms=45000, cfg=cfg, reason="initial-room-after-clear")
                            print(f"[run] page after room retry: {page.url}")
                        except Exception as retry_exc:
                            print(f"[run] room retry after clear failed: {retry_exc}. Continuing with login flow.")
                    else:
                        print(f"[run] room goto failed: {exc}. Continuing with login flow.")

            if "/dashboard" not in page.url.lower() and "/tasks" not in page.url.lower():
                ensure_logged_in(page, cfg)
                if _is_authenticated_page(page):
                    _ensure_parent(state_path)
                    context.storage_state(path=str(state_path))
                    print(f"[auth] saved state: {state_path}")

            max_episodes_per_run = int(_cfg_get(cfg, "run.max_episodes_per_run", 5))
            recycle_after_max_episodes = bool(_cfg_get(cfg, "run.recycle_after_max_episodes", True))
            release_all_wait_sec = max(0.0, float(_cfg_get(cfg, "run.release_all_wait_sec", 5)))
            no_task_retry_count = max(0, int(_cfg_get(cfg, "run.no_task_retry_count", 5)))
            no_task_retry_delay_sec = max(0.0, float(_cfg_get(cfg, "run.no_task_retry_delay_sec", 5.0)))
            no_task_backoff_factor = max(1.0, float(_cfg_get(cfg, "run.no_task_backoff_factor", 1.0)))
            no_task_max_delay_sec = max(
                no_task_retry_delay_sec,
                float(_cfg_get(cfg, "run.no_task_max_delay_sec", max(5.0, no_task_retry_delay_sec))),
            )
            clear_blocked_tasks_every_retry = bool(
                _cfg_get(cfg, "run.clear_blocked_tasks_every_retry", True)
            )
            release_and_reserve_on_all_visible_blocked = bool(
                _cfg_get(cfg, "run.release_and_reserve_on_all_visible_blocked", True)
            )
            release_and_reserve_on_submit_unverified = bool(
                _cfg_get(cfg, "run.release_and_reserve_on_submit_unverified", True)
            )
            strict_terminal_submit_handling = bool(
                _cfg_get(cfg, "run.use_episode_runtime_v2", False)
                or _cfg_get(cfg, "run.strict_single_chat_session", False)
                or _cfg_get(cfg, "run.force_episode_browser_isolation", False)
            )
            if disable_release_all_during_canary:
                release_all_after_batch = False
                recycle_after_max_episodes = False
                release_and_reserve_on_all_visible_blocked = False
                release_and_reserve_on_submit_unverified = False
            if strict_terminal_submit_handling:
                release_and_reserve_on_submit_unverified = False
            keep_alive_when_idle = bool(_cfg_get(cfg, "run.keep_alive_when_idle", True))
            keep_alive_idle_cycle_pause_sec = max(
                0.0, float(_cfg_get(cfg, "run.keep_alive_idle_cycle_pause_sec", 5.0))
            )
            clear_blocked_after_hits = max(
                1, int(_cfg_get(cfg, "run.clear_blocked_tasks_after_all_visible_blocked_hits", 1))
            )
            resume_from_artifacts = bool(_cfg_get(cfg, "run.resume_from_artifacts", True))
            resume_skip_video_steps_when_cached = bool(_cfg_get(cfg, "run.resume_skip_video_steps_when_cached", True))
            resume_skip_apply_steps_when_done = bool(_cfg_get(cfg, "run.resume_skip_apply_steps_when_done", True))
            allow_resume_auto_submit = bool(_cfg_get(cfg, "run.allow_resume_auto_submit", False))
            execute_force_fresh_gemini = bool(_cfg_get(cfg, "run.execute_force_fresh_gemini", True))
            execute_force_live_segments = bool(_cfg_get(cfg, "run.execute_force_live_segments", True))
            execute_require_video_context = bool(_cfg_get(cfg, "run.execute_require_video_context", True))
            skip_duplicate_task_in_run = bool(_cfg_get(cfg, "run.skip_duplicate_task_in_run", True))
            duplicate_task_retry_count = max(0, int(_cfg_get(cfg, "run.duplicate_task_retry_count", 3)))
            duplicate_task_retry_wait_sec = max(0.0, float(_cfg_get(cfg, "run.duplicate_task_retry_wait_sec", 2.0)))
            continue_on_episode_error = bool(_cfg_get(cfg, "run.continue_on_episode_error", True))
            if disable_release_all_during_canary:
                continue_on_episode_error = False
                keep_alive_when_idle = False
            max_episode_failures_per_run = max(0, int(_cfg_get(cfg, "run.max_episode_failures_per_run", 3)))
            episode_failure_retry_delay_sec = max(0.0, float(_cfg_get(cfg, "run.episode_failure_retry_delay_sec", 4.0)))
            gemini_quota_global_pause_min_sec = max(
                1.0, float(_cfg_get(cfg, "run.gemini_quota_global_pause_min_sec", 60.0))
            )
            gemini_quota_global_pause_step_sec = max(
                1.0, float(_cfg_get(cfg, "run.gemini_quota_global_pause_step_sec", 60.0))
            )
            gemini_quota_task_block_max_wait_sec = max(
                5.0, float(_cfg_get(cfg, "run.gemini_quota_task_block_max_wait_sec", 21600.0))
            )
            max_video_prepare_failures_per_task = max(1, int(_cfg_get(cfg, "run.max_video_prepare_failures_per_task", 2)))
            max_gemini_failures_per_task = max(1, int(_cfg_get(cfg, "run.max_gemini_failures_per_task", 1)))
            target_task_ids = [_task_id_from_url(url) for url in target_task_urls if _task_id_from_url(url)]
            if target_task_ids:
                print(
                    "[run] target_task_urls active: "
                    + ", ".join(target_task_ids)
                )
            seed_blocked_task_ids = {
                str(item or "").strip().lower()
                for item in (_cfg_get(cfg, "run.seed_blocked_task_ids", []) or [])
                if str(item or "").strip()
            }
            if seed_blocked_task_ids:
                print(
                    "[run] seeded blocked task ids: "
                    + ", ".join(sorted(seed_blocked_task_ids))
                )
            episode_no = 0
            seen_task_ids: set[str] = set()
            blocked_task_ids: set[str] = set(seed_blocked_task_ids)
            quota_blocked_task_until_ts: Dict[str, float] = {}
            gemini_quota_global_pause_until_ts = 0.0
            video_prepare_failures_by_task: Dict[str, int] = {}
            gemini_failures_by_task: Dict[str, int] = {}
            duplicate_hits = 0
            no_task_hits = 0
            all_visible_blocked_hits = 0
            consecutive_episode_failures = 0
            # ?????? Production Hardening: Watchdog & Duration ??????????????????????????????????????????
            _run_start_ts = time.time()
            _last_activity_ts = time.time()
            max_run_duration_sec = max(
                0.0, float(_cfg_get(cfg, "run.max_run_duration_sec", 0.0))
            )
            watchdog_stale_threshold_base_sec = max(
                60.0, float(_cfg_get(cfg, "run.watchdog_stale_threshold_sec", 600.0))
            )
            watchdog_stale_threshold_sec = watchdog_stale_threshold_base_sec
            # ?????? Startup stale sweep prevent API quota leaks ??????
            cleanup_api_key = _resolve_gemini_key(str(_cfg_get(cfg, "gemini.api_key", "")).strip())
            if not cleanup_api_key:
                cleanup_api_key = _resolve_gemini_fallback_key(
                    str(_cfg_get(cfg, "gemini.fallback_api_key", "")).strip()
                )
            if cleanup_api_key:
                _sweep_stale_gemini_files(cleanup_api_key, cfg)

            # ?????? Backup Watchdog (Process Level) ??????
            def _backup_watchdog():
                if _shutdown_requested.is_set():
                    return
                stale_sec = time.time() - _last_activity_ts
                if stale_sec > watchdog_stale_threshold_sec + 60.0:
                    print(f"[watchdog] FATAL: process hung for {stale_sec:.0f}s. Forcing exit.")
                    os._exit(1)
            
            _backup_timer = None
            def _reset_backup_watchdog():
                nonlocal _backup_timer
                if _backup_timer is not None:
                    _backup_timer.cancel()
                _backup_timer = threading.Timer(watchdog_stale_threshold_sec + 60.0, _backup_watchdog)
                _backup_timer.daemon = True
                _backup_timer.start()

            def _touch_watchdog() -> None:
                nonlocal _last_activity_ts
                _last_activity_ts = time.time()
                _reset_backup_watchdog()

            def _mark_task_stage(
                task_id_value: str,
                task_state_value: Optional[Dict[str, Any]],
                *,
                stage: str,
                status: str,
                progress_current: Optional[int] = None,
                progress_total: Optional[int] = None,
                detail: str = "",
                last_error: str = "",
            ) -> Dict[str, Any]:
                nonlocal watchdog_stale_threshold_sec
                task_key = str(task_id_value or "").strip()
                if not task_key:
                    return task_state_value if isinstance(task_state_value, dict) else {}

                if str(status or "").strip().lower() == "running":
                    watchdog_stale_threshold_sec = _stage_watchdog_timeout_hint_sec(
                        cfg,
                        stage=stage,
                        base_timeout_sec=watchdog_stale_threshold_base_sec,
                        progress_current=int(progress_current or 0),
                        progress_total=int(progress_total or 0),
                    )
                else:
                    watchdog_stale_threshold_sec = watchdog_stale_threshold_base_sec
                _reset_backup_watchdog()
                _touch_watchdog()
                return _persist_task_stage_status(
                    cfg,
                    task_key,
                    task_state_value,
                    stage=stage,
                    status=status,
                    progress_current=progress_current,
                    progress_total=progress_total,
                    detail=detail,
                    last_error=last_error,
                    watchdog_timeout_sec=watchdog_stale_threshold_sec,
                )
            
            _ACTIVE_HEARTBEAT_CALLBACK = _touch_watchdog
            _reset_backup_watchdog()

            def _episode_failure_mode() -> str:
                if not continue_on_episode_error:
                    return "raise"
                if consecutive_episode_failures > max_episode_failures_per_run:
                    return "stop"
                return "continue"

            def _cleanup_expired_quota_blocks() -> None:
                now_ts = time.time()
                expired_ids = [
                    task for task, until_ts in quota_blocked_task_until_ts.items()
                    if until_ts <= now_ts
                ]
                for task in expired_ids:
                    quota_blocked_task_until_ts.pop(task, None)
                if expired_ids:
                    print(f"[run] cleared expired quota task cooldowns: {len(expired_ids)}")

            def _active_quota_blocked_task_ids() -> set[str]:
                _cleanup_expired_quota_blocks()
                now_ts = time.time()
                return {
                    task for task, until_ts in quota_blocked_task_until_ts.items()
                    if until_ts > now_ts
                }

            def _register_quota_failure(task_id: Optional[str], exc: Exception, phase_label: str) -> float:
                nonlocal gemini_quota_global_pause_until_ts
                base_delay_sec = max(1.0, float(_cfg_get(cfg, "run.gemini_quota_retry_delay_sec", 15.0)))
                quota_wait_sec = _extract_retry_seconds_from_text(str(exc or ""), default_wait_sec=base_delay_sec)
                quota_wait_sec = min(gemini_quota_task_block_max_wait_sec, max(base_delay_sec, quota_wait_sec))
                now_ts = time.time()

                if quota_wait_sec >= gemini_quota_global_pause_min_sec:
                    pause_until_ts = now_ts + quota_wait_sec
                    if pause_until_ts > gemini_quota_global_pause_until_ts:
                        gemini_quota_global_pause_until_ts = pause_until_ts
                    print(
                        "[run] Gemini quota cooldown activated globally "
                        f"for {quota_wait_sec:.1f}s ({phase_label})."
                    )

                if task_id:
                    blocked_until_ts = now_ts + quota_wait_sec
                    prev_until_ts = quota_blocked_task_until_ts.get(task_id, 0.0)
                    if blocked_until_ts > prev_until_ts:
                        quota_blocked_task_until_ts[task_id] = blocked_until_ts
                    blocked_task_ids.add(task_id)
                    print(
                        f"[run] gemini quota limit for task {task_id}; "
                        f"task blocked for {quota_wait_sec:.1f}s before retry."
                    )
                else:
                    print(
                        "[run] gemini quota limit encountered without task id; "
                        f"using retry wait {quota_wait_sec:.1f}s."
                    )

                return quota_wait_sec

            def _release_then_reserve_cycle(reason: str) -> bool:
                nonlocal no_task_hits, all_visible_blocked_hits, duplicate_hits
                print(f"[run] {reason}; triggering release-all + reserve-new cycle.")
                released = _release_all_reserved_episodes(page, cfg)
                if not released:
                    print("[run] release-all cycle skipped (button not found).")
                    return False
                blocked_task_ids.clear()
                seen_task_ids.clear()
                video_prepare_failures_by_task.clear()
                gemini_failures_by_task.clear()
                no_task_hits = 0
                all_visible_blocked_hits = 0
                duplicate_hits = 0
                if room_url:
                    try:
                        _goto_with_retry(
                            page,
                            room_url,
                            wait_until="domcontentloaded",
                            timeout_ms=45000,
                            cfg=cfg,
                            reason="room-after-release-reserve-cycle",
                        )
                    except Exception:
                        pass
                try:
                    immediate_status: Dict[str, Any] = {}
                    opened_now = goto_task_room(
                        page,
                        cfg,
                        skip_task_ids=set(),
                        status_out=immediate_status,
                    )
                    if opened_now:
                        print("[run] release+reserve cycle opened a new episode immediately.")
                except Exception:
                    pass
                return True

            while True:
                # ?????? Graceful Shutdown Check ????????????????????????????????????????????????????????????????????????????????????
                if _shutdown_requested.is_set():
                    print("[run] graceful shutdown requested; exiting run loop.")
                    break

                # ?????? Wall-Clock Duration Limit ??????????????????????????????????????????????????????????????????????????????
                if max_run_duration_sec > 0:
                    elapsed_sec = time.time() - _run_start_ts
                    if elapsed_sec >= max_run_duration_sec:
                        print(
                            f"[run] max_run_duration_sec={max_run_duration_sec:.0f} reached "
                            f"(elapsed {elapsed_sec:.0f}s); stopping run."
                        )
                        break

                # ?????? Activity Watchdog ??????????????????????????????????????????????????????????????????????????????????????????????????????
                _watchdog_idle_sec = time.time() - _last_activity_ts
                if _watchdog_idle_sec > watchdog_stale_threshold_sec:
                    print(
                        f"[watchdog] no successful activity for {_watchdog_idle_sec:.0f}s "
                        f"(threshold={watchdog_stale_threshold_sec:.0f}s); "
                        "possible browser hang. Breaking run loop for restart."
                    )
                    break

                if max_episodes_per_run > 0 and episode_no >= max_episodes_per_run:
                    print(f"[run] reached max_episodes_per_run={max_episodes_per_run}.")
                    if recycle_after_max_episodes:
                        if release_all_after_batch:
                            _release_all_reserved_episodes(page, cfg)
                        if release_all_wait_sec > 0:
                            print(f"[run] waiting {release_all_wait_sec:.0f}s before reserving a new episode batch.")
                            _sleep_with_shutdown_heartbeat(
                                release_all_wait_sec,
                                heartbeat_sec=min(30.0, max(5.0, watchdog_stale_threshold_sec / 4.0)),
                                on_heartbeat=_touch_watchdog,
                            )
                        episode_no = 0
                        seen_task_ids.clear()
                        blocked_task_ids.clear()
                        video_prepare_failures_by_task.clear()
                        gemini_failures_by_task.clear()
                        duplicate_hits = 0
                        no_task_hits = 0
                        consecutive_episode_failures = 0

                        print("[run] recycling browser page to prevent chromium memory leak.")
                        try:
                            old_page = page
                            page = context.new_page()
                            old_page.close()
                            bootstrap_page = page
                            bootstrap_context = context
                        except Exception as e:
                            print(f"[run] warning: failed to recycle page: {e}")

                        if room_url:
                            try:
                                _goto_with_retry(
                                    page,
                                    room_url,
                                    wait_until="domcontentloaded",
                                    timeout_ms=45000,
                                    cfg=cfg,
                                    reason="room-after-release-cycle",
                                )
                            except Exception:
                                pass
                        continue
                    break

                if gemini_quota_global_pause_until_ts > time.time():
                    remaining_sec = gemini_quota_global_pause_until_ts - time.time()
                    pause_sec = min(remaining_sec, gemini_quota_global_pause_step_sec)
                    pause_sec = max(1.0, pause_sec)
                    print(
                        "[run] Gemini daily quota pause is active; "
                        f"remaining {remaining_sec:.1f}s, sleeping {pause_sec:.1f}s."
                    )
                    _sleep_with_shutdown_heartbeat(
                        pause_sec,
                        heartbeat_sec=min(30.0, max(5.0, watchdog_stale_threshold_sec / 4.0)),
                        on_heartbeat=_touch_watchdog,
                    )
                    continue

                active_quota_blocked_task_ids = _active_quota_blocked_task_ids()
                if target_task_ids:
                    remaining_target_ids = [
                        tid
                        for tid in target_task_ids
                        if tid
                        and tid not in seen_task_ids
                        and tid not in blocked_task_ids
                        and tid not in active_quota_blocked_task_ids
                    ]
                    if not remaining_target_ids:
                        quota_blocked_target_ids = [
                            tid for tid in target_task_ids
                            if tid
                            and tid not in seen_task_ids
                            and tid in active_quota_blocked_task_ids
                        ]
                        if quota_blocked_target_ids:
                            soonest_target_retry_sec = min(
                                max(0.0, quota_blocked_task_until_ts.get(tid, 0.0) - time.time())
                                for tid in quota_blocked_target_ids
                            )
                            wait_sec = max(1.0, min(soonest_target_retry_sec, gemini_quota_global_pause_step_sec))
                            print(
                                "[run] all remaining target tasks are quota-blocked; "
                                f"retrying targeted queue in {wait_sec:.1f}s."
                            )
                            _sleep_with_shutdown_heartbeat(
                                wait_sec,
                                heartbeat_sec=min(30.0, max(5.0, watchdog_stale_threshold_sec / 4.0)),
                                on_heartbeat=_touch_watchdog,
                            )
                            continue
                        print("[run] target_task_urls exhausted for this run.")
                        break
                skip_task_ids_for_open: set[str] = set(blocked_task_ids)
                skip_task_ids_for_open.update(active_quota_blocked_task_ids)
                if skip_duplicate_task_in_run:
                    skip_task_ids_for_open.update(seen_task_ids)
                open_status: Dict[str, Any] = {}
                blocked_before_open = set(blocked_task_ids)
                opened = goto_task_room(
                    page,
                    cfg,
                    skip_task_ids=skip_task_ids_for_open,
                    preferred_task_urls=target_task_urls,
                    status_out=open_status,
                )
                target_task_urls, target_task_ids, _sticky_resume_cleared = _maybe_clear_sticky_resume_targets(
                    cfg,
                    target_task_urls,
                    target_task_ids,
                    open_status,
                )
                # Important: do not pollute blocked_task_ids with seen_task_ids.
                newly_blocked = {
                    tid for tid in skip_task_ids_for_open
                    if tid not in blocked_before_open
                    and tid not in seen_task_ids
                    and tid not in active_quota_blocked_task_ids
                }
                if newly_blocked:
                    blocked_task_ids.update(newly_blocked)
                if not opened:
                    no_reserved_episodes = bool(open_status.get("no_reserved_episodes"))
                    if _should_clear_blocked_tasks_before_idle_retry(
                        clear_blocked_tasks_every_retry=clear_blocked_tasks_every_retry,
                        blocked_task_ids=blocked_task_ids,
                        open_status=open_status,
                    ):
                        print(
                            "[run] clearing blocked-task list before retry "
                            f"(size={len(blocked_task_ids)})."
                        )
                        blocked_task_ids.clear()
                        all_visible_blocked_hits = 0
                    if bool(open_status.get("all_visible_blocked")):
                        all_visible_blocked_hits += 1
                        if release_and_reserve_on_all_visible_blocked and all_visible_blocked_hits >= 1:
                            if _release_then_reserve_cycle("all visible tasks are blocked"):
                                continue
                        if blocked_task_ids and all_visible_blocked_hits >= clear_blocked_after_hits:
                            print(
                                "[run] all visible tasks stayed blocked across idle checks; "
                                f"clearing blocked-task list (size={len(blocked_task_ids)})."
                            )
                            blocked_task_ids.clear()
                            all_visible_blocked_hits = 0
                    else:
                        all_visible_blocked_hits = 0
                    no_task_hits += 1
                    keep_alive_pause_sec = 0.0
                    if no_task_hits > no_task_retry_count:
                        if keep_alive_when_idle and max_episodes_per_run <= 0:
                            print(
                                f"[run] {'no reserved episodes or label task available right now' if no_reserved_episodes else 'no label task available right now'}; "
                                "retry budget exhausted but keep-alive is enabled, continuing poll loop."
                            )
                            no_task_hits = max(1, no_task_retry_count)
                            keep_alive_pause_sec = keep_alive_idle_cycle_pause_sec
                        else:
                            if no_reserved_episodes:
                                print("[run] no reserved episodes or label task available right now; retry budget exhausted.")
                            else:
                                print("[run] no label task available right now; retry budget exhausted.")
                            break
                    backoff_exp = max(0, no_task_hits - 1)
                    retry_delay_sec = min(
                        no_task_max_delay_sec,
                        no_task_retry_delay_sec * (no_task_backoff_factor**backoff_exp),
                    )
                    if active_quota_blocked_task_ids:
                        soonest_quota_release_sec = min(
                            max(0.0, quota_blocked_task_until_ts.get(tid, 0.0) - time.time())
                            for tid in active_quota_blocked_task_ids
                        )
                        quota_idle_wait_sec = min(
                            soonest_quota_release_sec,
                            gemini_quota_global_pause_step_sec,
                        )
                        if quota_idle_wait_sec > retry_delay_sec:
                            retry_delay_sec = quota_idle_wait_sec
                            print(
                                "[run] quota-task cooldown active while idle; "
                                f"next retry in {retry_delay_sec:.1f}s."
                            )
                    if keep_alive_pause_sec > retry_delay_sec:
                        retry_delay_sec = keep_alive_pause_sec
                    print(
                        f"[run] {'no reserved episodes or label task available right now' if no_reserved_episodes else 'no label task available right now'}; retry "
                        f"{no_task_hits}/{no_task_retry_count} in {retry_delay_sec:.1f}s."
                    )
                    if room_url:
                        try:
                            _goto_with_retry(
                                page,
                                room_url,
                                wait_until="domcontentloaded",
                                timeout_ms=45000,
                                cfg=cfg,
                                reason="room-retry-no-task",
                            )
                        except Exception:
                            pass
                    if retry_delay_sec > 0:
                        _sleep_with_shutdown_heartbeat(
                            retry_delay_sec,
                            heartbeat_sec=min(30.0, max(5.0, watchdog_stale_threshold_sec / 4.0)),
                            on_heartbeat=_touch_watchdog,
                        )
                    continue
                no_task_hits = 0
                all_visible_blocked_hits = 0
                task_id = _task_id_from_url(page.url)
                if target_task_ids and task_id and task_id not in target_task_ids:
                    print(f"[run] opened non-target task while target_task_urls are active; skipping: {task_id}")
                    blocked_task_ids.add(task_id)
                    if room_url:
                        try:
                            _goto_with_retry(
                                page,
                                room_url,
                                wait_until="domcontentloaded",
                                timeout_ms=45000,
                                cfg=cfg,
                                reason="room-after-non-target-open",
                            )
                        except Exception:
                            pass
                    continue
                if task_id and task_id in blocked_task_ids:
                    print(f"[run] opened blocked task again ({task_id}); retrying room selection.")
                    if room_url:
                        try:
                            _goto_with_retry(
                                page,
                                room_url,
                                wait_until="domcontentloaded",
                                timeout_ms=45000,
                                cfg=cfg,
                                reason="room-after-blocked-task",
                            )
                        except Exception:
                            pass
                    if no_task_retry_delay_sec > 0:
                        _sleep_with_shutdown_heartbeat(
                            no_task_retry_delay_sec,
                            heartbeat_sec=min(30.0, max(5.0, watchdog_stale_threshold_sec / 4.0)),
                            on_heartbeat=_touch_watchdog,
                        )
                    continue
                if skip_duplicate_task_in_run and task_id and task_id in seen_task_ids:
                    duplicate_hits += 1
                    print(
                        f"[run] duplicate task reopened in same run: {task_id} "
                        f"(retry {duplicate_hits}/{duplicate_task_retry_count})."
                    )
                    if duplicate_hits > duplicate_task_retry_count:
                        blocked_task_ids.add(task_id)
                        duplicate_hits = 0
                        print(
                            "[run] duplicate task retry budget exhausted; "
                            f"blocking task for this run and continuing: {task_id}"
                        )
                        if room_url:
                            try:
                                _goto_with_retry(
                                    page,
                                    room_url,
                                    wait_until="domcontentloaded",
                                    timeout_ms=45000,
                                    cfg=cfg,
                                    reason="room-after-duplicate-exhausted",
                                )
                            except Exception:
                                pass
                        if duplicate_task_retry_wait_sec > 0:
                            _sleep_with_shutdown_heartbeat(
                                duplicate_task_retry_wait_sec,
                                heartbeat_sec=min(30.0, max(5.0, watchdog_stale_threshold_sec / 4.0)),
                                on_heartbeat=_touch_watchdog,
                            )
                        continue
                    if duplicate_task_retry_wait_sec > 0:
                        _sleep_with_shutdown_heartbeat(
                            duplicate_task_retry_wait_sec,
                            heartbeat_sec=min(30.0, max(5.0, watchdog_stale_threshold_sec / 4.0)),
                            on_heartbeat=_touch_watchdog,
                        )
                    if room_url:
                        try:
                            _goto_with_retry(
                                page,
                                room_url,
                                wait_until="domcontentloaded",
                                timeout_ms=45000,
                                cfg=cfg,
                                reason="room-after-duplicate",
                            )
                        except Exception:
                            pass
                    continue
                duplicate_hits = 0
                episode_no += 1
                print(f"[run] episode {episode_no} opened: {page.url}")
                episode_runtime: Optional[EpisodeRuntime] = None
                episode_report = EpisodeReport(
                    episode_id=str(task_id or f"episode_{episode_no}").strip(),
                    page_url=str(getattr(page, "url", "") or ""),
                )
                validation_tracker: Optional[ValidationTracker] = None
                if bool(_cfg_get(cfg, "run.live_validation_enabled", False)) or bool(
                    _cfg_get(cfg, "run.structured_episode_reports", False)
                ):
                    validation_tracker = ValidationTracker(
                        cfg,
                        str(task_id or f"episode_{episode_no}").strip(),
                    )
                episode_bootstrap_page = bootstrap_page
                episode_bootstrap_context = bootstrap_context
                episode_runtime, page, context = _activate_episode_runtime_v2(
                    browser=browser,
                    gemini_browser=gemini_browser,
                    bootstrap_context=episode_bootstrap_context,
                    bootstrap_page=episode_bootstrap_page,
                    state_path=state_path,
                    cfg=cfg,
                    task_id=task_id,
                )
                episode_report.context_id = (
                    episode_runtime.context_id if episode_runtime is not None else "bootstrap"
                )
                episode_report.page_url = str(getattr(page, "url", "") or "")
                segments: List[Dict[str, Any]] = []
                validation_report: Dict[str, Any] = {}
                result: Dict[str, Any] = {}

                # ── Clean State Protocol: clear stale caches for fresh episode ──
                if task_id:
                    cleared_paths = _clear_episode_state(
                        cfg,
                        task_id,
                        clear_task_state=False,
                        clear_shared_dumps=True,
                    )
                    print(
                        f"[state] clean state: cleared {len(cleared_paths)} artifact(s) "
                        f"for episode {episode_no} (task={task_id[:12]}...)"
                    )

                task_state = _load_task_state(cfg, task_id) if task_id else {}
                scoped_paths = _task_scoped_artifact_paths(cfg, task_id) if task_id else {}
                if task_id:
                    task_state = _persist_task_state_fields(
                        cfg,
                        task_id,
                        task_state,
                        task_url=page.url,
                        last_error="",
                        sticky_task_id=task_id,
                        resume_from_stage=str(task_state.get("current_stage", "") or "episode_opened"),
                        episode_locked=True,
                        browser_topology=(
                            "single_window_single_tab"
                            if single_window_single_tab
                            else ("single_window_two_tabs" if single_window_two_tabs else "isolated_context")
                        ),
                        held_object_context=task_state.get("held_object_context", {}),
                    )
                    task_state = _capture_episode_step(cfg, page, task_id, task_state, "episode_opened")

                _dismiss_blocking_modals(page)
                if bool(_cfg_get(cfg, "run.loop_off_on_episode_open", True)):
                    # Fast-path: toggle loop off right after opening the episode.
                    _ensure_loop_off(page, cfg)
                labels_payload: Optional[Dict[str, Any]] = None
                if task_id:
                    if execute and execute_force_fresh_gemini:
                        print("[gemini] execute mode: forcing fresh Gemini evaluation (ignoring cached labels).")
                    else:
                        labels_payload = _load_cached_labels(cfg, task_id)
                min_video_bytes = int(_cfg_get(cfg, "gemini.min_video_bytes", 500000))
                validate_video_decode = bool(_cfg_get(cfg, "gemini.validate_video_decode", True))
                cached_video_file: Optional[Path] = None
                if task_id:
                    candidate = scoped_paths.get("video")
                    if candidate is not None and candidate.exists():
                        try:
                            if candidate.stat().st_size >= min_video_bytes and _is_probably_mp4(candidate):
                                if not validate_video_decode or _is_video_decodable(candidate):
                                    cached_video_file = candidate
                                else:
                                    print(f"[video] ignoring cached video with failed decode check: {candidate}")
                        except Exception:
                            cached_video_file = None

                skip_video_steps = bool(
                    resume_from_artifacts
                    and resume_skip_video_steps_when_cached
                    and (cached_video_file is not None or labels_payload is not None)
                )
                if skip_video_steps:
                    print(
                        f"[run] resume mode: skipping video playback "
                        f"(cached_video={cached_video_file is not None}, cached_labels={labels_payload is not None})."
                    )
                else:
                    _ensure_loop_off(page, cfg)
                    _play_full_video_once(page, cfg)

                if labels_payload is not None and skip_video_steps and cached_video_file is None:
                    video_file = None
                    print("[video] skipped video preparation (cached labels available).")
                elif cached_video_file is not None:
                    video_file = cached_video_file
                    print(f"[video] using cached task video: {video_file}")
                else:
                    print("[run] preparing task video for Gemini...")
                    try:
                        video_file = _prepare_video_for_gemini(page, context, cfg, task_id=task_id)
                    except Exception as exc:
                        consecutive_episode_failures += 1
                        print(f"[run] episode {episode_no} failed during video preparation: {exc}")
                        _capture_debug_artifacts(page, cfg, prefix="debug_episode_failure")
                        if task_id:
                            current_failures = int(video_prepare_failures_by_task.get(task_id, 0)) + 1
                            video_prepare_failures_by_task[task_id] = current_failures
                            print(
                                f"[run] video prepare failure for task {task_id}: "
                                f"{current_failures}/{max_video_prepare_failures_per_task}"
                            )
                            if current_failures >= max_video_prepare_failures_per_task:
                                blocked_task_ids.add(task_id)
                                print(f"[run] task blocked for this run due to repeated video failures: {task_id}")
                        if task_id:
                            task_state = _persist_task_state_fields(
                                cfg,
                                task_id,
                                task_state,
                                last_error=str(exc),
                            )
                        failure_mode = _episode_failure_mode()
                        if failure_mode == "raise":
                            raise
                        if failure_mode == "stop":
                            print(
                                "[run] episode failure budget exhausted "
                                f"({consecutive_episode_failures}>{max_episode_failures_per_run}); stopping run."
                            )
                            episode_report.failure_class = FailureClass.APPLY_FAILURE
                            episode_report.notes.append("video preparation failed")
                            page, context = _finalize_current_episode_v2(
                                cfg=cfg,
                                report=episode_report,
                                task_state=task_state,
                                runtime=episode_runtime,
                                bootstrap_page=episode_bootstrap_page,
                                bootstrap_context=episode_bootstrap_context,
                                room_url=room_url,
                                page=page,
                                segments=segments,
                                validation_report=validation_report,
                                result=result,
                                validation_tracker=validation_tracker,
                                reason="room-after-episode-failure-video-v2",
                            )
                            break
                        if room_url:
                            print("[run] returning to room page after episode failure.")
                            try:
                                _goto_with_retry(
                                    page,
                                    room_url,
                                    wait_until="domcontentloaded",
                                    timeout_ms=45000,
                                    cfg=cfg,
                                    reason="room-after-episode-failure-video",
                                )
                            except Exception:
                                pass
                        if episode_failure_retry_delay_sec > 0:
                            print(f"[run] waiting {episode_failure_retry_delay_sec:.1f}s before retrying next episode.")
                            _sleep_with_shutdown_heartbeat(
                                episode_failure_retry_delay_sec,
                                heartbeat_sec=min(30.0, max(5.0, watchdog_stale_threshold_sec / 4.0)),
                                on_heartbeat=_touch_watchdog,
                            )
                        episode_report.failure_class = FailureClass.APPLY_FAILURE
                        episode_report.notes.append("video preparation failed")
                        page, context = _finalize_current_episode_v2(
                            cfg=cfg,
                            report=episode_report,
                            task_state=task_state,
                            runtime=episode_runtime,
                            bootstrap_page=episode_bootstrap_page,
                            bootstrap_context=episode_bootstrap_context,
                            room_url=room_url,
                            page=page,
                            segments=segments,
                            validation_report=validation_report,
                            result=result,
                            validation_tracker=validation_tracker,
                            reason="room-after-episode-failure-video-v2",
                        )
                        continue

                if task_id and video_file is not None:
                    task_state = _persist_task_state_fields(
                        cfg,
                        task_id,
                        task_state,
                        video_path=str(video_file),
                        video_ready=True,
                        last_error="",
                    )
                    task_state = _capture_episode_step(cfg, page, task_id, task_state, "video_ready")

                segments = None
                if not (execute and execute_force_live_segments) and resume_from_artifacts and task_id:
                    cached_segments = _load_cached_segments(cfg, task_id)
                    if cached_segments:
                        segments = cached_segments
                        print(f"[atlas] using cached segments for task {task_id}: {len(segments)}")
                        task_state = _mark_task_stage(
                            task_id,
                            task_state,
                            stage="extract_segments",
                            status="completed",
                            progress_current=len(segments),
                            progress_total=len(segments),
                            detail="reused cached segments",
                        )
                if segments is None:
                    print("[run] extracting Atlas segments...")
                    if task_id:
                        task_state = _mark_task_stage(
                            task_id,
                            task_state,
                            stage="extract_segments",
                            status="running",
                            detail="reading Atlas segment rows",
                        )

                    def _extract_progress(current: int, total: int) -> None:
                        nonlocal task_state
                        _touch_watchdog()
                        if task_id:
                            task_state = _mark_task_stage(
                                task_id,
                                task_state,
                                stage="extract_segments",
                                status="running",
                                progress_current=current,
                                progress_total=total,
                                detail=f"read {current}/{total} segment rows",
                            )

                    segments = extract_segments(page, cfg, progress_callback=_extract_progress)
                    print(f"[atlas] extracted {len(segments)} segments")
                    if task_id:
                        append_execution_journal_event(
                            cfg,
                            episode_id=task_id,
                            event_type="segments_snapshot",
                            stage="running",
                            reason="extract_segments_completed",
                            task_state=task_state,
                            payload={
                                "segment_count": len(segments),
                                "segments_checksum": build_segment_checksum(segments),
                            },
                            run_id=str((task_state or {}).get("run_id", (task_state or {}).get("context_id", "")) or "").strip(),
                            context_id=str((task_state or {}).get("context_id", "") or "").strip(),
                            segments_checksum=build_segment_checksum(segments),
                            page_url=str(getattr(page, "url", "") or "").strip(),
                        )
                    if task_id:
                        task_state = _mark_task_stage(
                            task_id,
                            task_state,
                            stage="extract_segments",
                            status="completed",
                            progress_current=len(segments),
                            progress_total=len(segments),
                            detail="segment extraction completed",
                        )
                    if task_id and resume_from_artifacts:
                        _save_cached_segments(cfg, task_id, segments)
                if validation_tracker is not None:
                    validation_tracker.set_initial_state(
                        segments,
                        max(
                            0.1,
                            float(_cfg_get(cfg, "run.max_segment_duration_sec", 20.0) or 20.0),
                        ),
                    )

                enable_structural_actions = bool(_cfg_get(cfg, "run.enable_structural_actions", True))
                requery_after_structural_actions = bool(_cfg_get(cfg, "run.requery_after_structural_actions", True))
                prompt = build_prompt(
                    segments,
                    str(_cfg_get(cfg, "gemini.extra_instructions", "")),
                    allow_operations=True,
                    policy_trigger="base",
                )
                projected_episode_cost = estimate_minimum_episode_cost_usd(cfg, len(segments))
                cost_guards_enabled = cost_guard_enforcement_enabled(cfg)
                projected_over_hard_cap = (
                    str(projected_episode_cost.get("episode_budget_state", "")).strip() == "over_hard_cap"
                )
                if task_id:
                    projected_cost_updates = dict(projected_episode_cost)
                    if not cost_guards_enabled:
                        projected_cost_updates["deferred_due_to_cost_ratio"] = False
                        if str((task_state or {}).get("last_error", "") or "").startswith(
                            "deferred_due_to_cost_ratio:"
                        ):
                            projected_cost_updates["last_error"] = ""
                    task_state = _persist_task_state_fields(
                        cfg,
                        task_id,
                        task_state,
                        **projected_cost_updates,
                    )
                if projected_over_hard_cap and cost_guards_enabled:
                    print(
                        "[economics] projected minimum path exceeds hard cost ratio; deferring episode. "
                        f"cost=${float(projected_episode_cost.get('episode_estimated_cost_usd', 0.0) or 0.0):.4f} "
                        f"ratio={float(projected_episode_cost.get('episode_cost_ratio', 0.0) or 0.0) * 100.0:.1f}%"
                    )
                    if task_id:
                        task_state = _persist_task_state_fields(
                            cfg,
                            task_id,
                            task_state,
                            deferred_due_to_cost_ratio=True,
                            last_error="deferred_due_to_cost_ratio:projected_minimum_path",
                        )
                        task_state = _capture_episode_step(
                            cfg,
                            page,
                            task_id,
                            task_state,
                            "cost_ratio_deferred",
                            include_html=True,
                        )
                        blocked_task_ids.add(task_id)
                        if skip_duplicate_task_in_run:
                            seen_task_ids.add(task_id)
                    if keep_alive_when_idle and max_episodes_per_run <= 0:
                        if room_url:
                            try:
                                _goto_with_retry(
                                    page,
                                    room_url,
                                    wait_until="domcontentloaded",
                                    timeout_ms=45000,
                                    cfg=cfg,
                                    reason="room-after-cost-defer",
                                )
                            except Exception:
                                pass
                    episode_report.failure_class = FailureClass.POLICY_FAILURE
                    episode_report.notes.append("episode deferred due to projected hard cost ratio")
                    page, context = _finalize_current_episode_v2(
                        cfg=cfg,
                        report=episode_report,
                        task_state=task_state,
                        runtime=episode_runtime,
                        bootstrap_page=episode_bootstrap_page,
                        bootstrap_context=episode_bootstrap_context,
                        room_url=room_url,
                        page=page,
                        segments=segments,
                        validation_report=validation_report,
                        result=result,
                        validation_tracker=validation_tracker,
                        reason="room-after-cost-defer-v2",
                    )
                    continue
                if projected_over_hard_cap:
                    print(
                        "[economics] projected minimum path exceeds hard cost ratio, but cost guards are disabled; "
                        f"continuing. cost=${float(projected_episode_cost.get('episode_estimated_cost_usd', 0.0) or 0.0):.4f} "
                        f"ratio={float(projected_episode_cost.get('episode_cost_ratio', 0.0) or 0.0) * 100.0:.1f}%"
                    )
                    episode_report.notes.append(
                        "projected minimum path exceeded hard cost ratio; informational only"
                    )
                if labels_payload is None:
                    print("[run] requesting labels from Gemini...")
                    chat_only_mode = bool(_cfg_get(cfg, "run.chat_only_mode", False))
                    if task_id:
                        task_state = _mark_task_stage(
                            task_id,
                            task_state,
                            stage="chat_labels",
                            status="running",
                            progress_total=len(segments),
                            detail="requesting Gemini labels",
                        )
                    if chat_only_mode and task_id:
                        task_state = _capture_episode_step(cfg, page, task_id, task_state, "before_chat_ops")
                        task_state = _capture_episode_step(cfg, page, task_id, task_state, "before_chat_labels")
                    try:
                        labels_payload = _request_labels_with_optional_segment_chunking(
                            cfg,
                            segments,
                            prompt,
                            video_file,
                            allow_operations=True,
                            task_id=task_id,
                            task_state=task_state,
                            stage_name="labeling",
                        )
                    except Exception as exc:
                        quota_error = _is_gemini_quota_error(exc)
                        quota_wait_sec = 0.0
                        if quota_error:
                            consecutive_episode_failures = 0
                        else:
                            consecutive_episode_failures += 1
                        print(f"[run] episode {episode_no} failed during Gemini request: {exc}")
                        _capture_debug_artifacts(page, cfg, prefix="debug_episode_failure")
                        if task_id:
                            if quota_error:
                                quota_wait_sec = _register_quota_failure(task_id, exc, "gemini-request")
                            else:
                                current_failures = int(gemini_failures_by_task.get(task_id, 0)) + 1
                                gemini_failures_by_task[task_id] = current_failures
                                print(
                                    f"[run] gemini failure for task {task_id}: "
                                    f"{current_failures}/{max_gemini_failures_per_task}"
                                )
                                if current_failures >= max_gemini_failures_per_task:
                                    blocked_task_ids.add(task_id)
                                    print(f"[run] task blocked for this run due to repeated Gemini failures: {task_id}")
                        if task_id:
                            task_state = _persist_task_state_fields(
                                cfg,
                                task_id,
                                task_state,
                                last_error=str(exc),
                            )
                            task_state = _mark_task_stage(
                                task_id,
                                task_state,
                                stage="chat_labels",
                                status="failed",
                                progress_total=len(segments),
                                detail="Gemini labels request failed",
                                last_error=str(exc),
                            )
                        if _is_non_retriable_gemini_error(exc):
                            print("[run] non-retriable Gemini error detected; stopping run.")
                            episode_report.failure_class = FailureClass.GEMINI_TRANSPORT_FAILURE
                            episode_report.notes.append("non-retriable gemini request failure")
                            page, context = _finalize_current_episode_v2(
                                cfg=cfg,
                                report=episode_report,
                                task_state=task_state,
                                runtime=episode_runtime,
                                bootstrap_page=episode_bootstrap_page,
                                bootstrap_context=episode_bootstrap_context,
                                room_url=room_url,
                                page=page,
                                segments=segments,
                                validation_report=validation_report,
                                result=result,
                                validation_tracker=validation_tracker,
                                reason="room-after-episode-failure-gemini-v2",
                            )
                            break
                        failure_mode = _episode_failure_mode()
                        if failure_mode == "raise":
                            raise
                        if failure_mode == "stop":
                            print(
                                "[run] episode failure budget exhausted "
                                f"({consecutive_episode_failures}>{max_episode_failures_per_run}); stopping run."
                            )
                            episode_report.failure_class = FailureClass.GEMINI_TRANSPORT_FAILURE
                            episode_report.notes.append("gemini request failure budget exhausted")
                            page, context = _finalize_current_episode_v2(
                                cfg=cfg,
                                report=episode_report,
                                task_state=task_state,
                                runtime=episode_runtime,
                                bootstrap_page=episode_bootstrap_page,
                                bootstrap_context=episode_bootstrap_context,
                                room_url=room_url,
                                page=page,
                                segments=segments,
                                validation_report=validation_report,
                                result=result,
                                validation_tracker=validation_tracker,
                                reason="room-after-episode-failure-gemini-v2",
                            )
                            break
                        if room_url:
                            print("[run] returning to room page after episode failure.")
                            try:
                                _goto_with_retry(
                                    page,
                                    room_url,
                                    wait_until="domcontentloaded",
                                    timeout_ms=45000,
                                    cfg=cfg,
                                    reason="room-after-episode-failure-gemini",
                                )
                            except Exception:
                                pass
                        retry_delay_sec = episode_failure_retry_delay_sec
                        if quota_error:
                            quota_retry_delay_sec = max(
                                0.0,
                                float(_cfg_get(cfg, "run.gemini_quota_retry_delay_sec", 15.0)),
                            )
                            if quota_wait_sec <= 0:
                                quota_wait_sec = _register_quota_failure(task_id, exc, "gemini-request")
                            retry_delay_sec = max(retry_delay_sec, quota_retry_delay_sec, quota_wait_sec)
                        if retry_delay_sec > 0:
                            print(f"[run] waiting {retry_delay_sec:.1f}s before retrying next episode.")
                            _sleep_with_shutdown_heartbeat(
                                retry_delay_sec,
                                heartbeat_sec=min(30.0, max(5.0, watchdog_stale_threshold_sec / 4.0)),
                                on_heartbeat=_touch_watchdog,
                            )
                        episode_report.failure_class = FailureClass.GEMINI_TRANSPORT_FAILURE
                        episode_report.notes.append("gemini request failed")
                        page, context = _finalize_current_episode_v2(
                            cfg=cfg,
                            report=episode_report,
                            task_state=task_state,
                            runtime=episode_runtime,
                            bootstrap_page=episode_bootstrap_page,
                            bootstrap_context=episode_bootstrap_context,
                            room_url=room_url,
                            page=page,
                            segments=segments,
                            validation_report=validation_report,
                            result=result,
                            validation_tracker=validation_tracker,
                            reason="room-after-episode-failure-gemini-v2",
                        )
                        continue
                    if chat_only_mode and task_id:
                        meta = labels_payload.get("_meta", {}) if isinstance(labels_payload, dict) else {}
                        if bool(meta.get("chat_ops_attempted", False)):
                            task_state = _capture_episode_step(cfg, page, task_id, task_state, "after_chat_ops")
                        task_state = _capture_episode_step(cfg, page, task_id, task_state, "after_chat_labels")
                    if task_id:
                        task_state = _mark_task_stage(
                            task_id,
                            task_state,
                            stage="chat_labels",
                            status="completed",
                            progress_current=len(segments),
                            progress_total=len(segments),
                            detail="Gemini labels ready",
                        )
                    if execute and execute_require_video_context:
                        meta = labels_payload.get("_meta", {}) if isinstance(labels_payload, dict) else {}
                        video_attached = bool(meta.get("video_attached", False))
                        mode = str(meta.get("mode", "unknown"))
                        if not video_attached:
                            raise RuntimeError(
                                "Execute blocked: Gemini response is text-only (no video context). "
                                "Video review is required before apply/complete."
                            )
                        print(f"[gemini] execute guard: video context confirmed ({mode}).")
                    if task_id:
                        _save_cached_labels(cfg, task_id, labels_payload)
                    if task_id:
                        task_state = _persist_task_state_fields(
                            cfg,
                            task_id,
                            task_state,
                            labels_ready=True,
                            last_error="",
                            **_episode_model_state_updates(cfg, labels_payload, task_state),
                        )

                operations = _normalize_operations(labels_payload, cfg=cfg)
                if not operations and execute and enable_structural_actions:
                    try:
                        merge_plan_preview = _normalize_segment_plan(labels_payload, segments, cfg=cfg)
                        auto_merge_ops = _build_auto_continuity_merge_operations(merge_plan_preview, cfg)
                        if auto_merge_ops:
                            operations = auto_merge_ops
                            print(
                                f"[policy] auto-generated merge operations for continuity: {len(auto_merge_ops)}"
                            )
                    except Exception as auto_merge_exc:
                        print(f"[policy] auto continuity-merge skipped: {auto_merge_exc}")
                if operations:
                    ops_text = ", ".join([f"{op['action']}#{op['segment_index']}" for op in operations[:20]])
                    print(f"[gemini] suggested operations ({len(operations)}): {ops_text}")

                if execute and enable_structural_actions and operations:
                    op_result = apply_segment_operations(page, cfg, operations, heartbeat=_touch_watchdog)
                    print(
                        f"[run] operations applied: {op_result['applied']} "
                        f"(structural={op_result['structural_applied']})"
                    )
                    if op_result["failed"]:
                        print("[run] operation failures:")
                        for item in op_result["failed"]:
                            print(f"  - {item}")

                    if op_result["structural_applied"] > 0 and requery_after_structural_actions:
                        print("[run] structural changes detected; refreshing segments and requesting Gemini again...")
                        segments = extract_segments(page, cfg)
                        print(f"[atlas] extracted {len(segments)} segments (post-operations)")
                        if task_id and resume_from_artifacts:
                            _save_cached_segments(cfg, task_id, segments)
                        prompt = build_prompt(
                            segments,
                            str(_cfg_get(cfg, "gemini.extra_instructions", "")),
                            allow_operations=False,
                            policy_trigger="policy_conflict",
                        )
                        if bool(_cfg_get(cfg, "run.chat_only_mode", False)) and task_id:
                            task_state = _capture_episode_step(cfg, page, task_id, task_state, "before_chat_labels")
                        try:
                            labels_payload = _request_labels_with_optional_segment_chunking(
                                cfg,
                                segments,
                                prompt,
                                video_file,
                                allow_operations=False,
                                task_id=task_id,
                                task_state=task_state,
                                stage_name="repair",
                            )
                        except Exception as exc:
                            quota_error = _is_gemini_quota_error(exc)
                            quota_wait_sec = 0.0
                            if quota_error:
                                consecutive_episode_failures = 0
                            else:
                                consecutive_episode_failures += 1
                            print(f"[run] episode {episode_no} failed during Gemini re-query: {exc}")
                            _capture_debug_artifacts(page, cfg, prefix="debug_episode_failure")
                            if task_id:
                                if quota_error:
                                    quota_wait_sec = _register_quota_failure(task_id, exc, "gemini-requery")
                                else:
                                    current_failures = int(gemini_failures_by_task.get(task_id, 0)) + 1
                                    gemini_failures_by_task[task_id] = current_failures
                                    print(
                                        f"[run] gemini re-query failure for task {task_id}: "
                                        f"{current_failures}/{max_gemini_failures_per_task}"
                                    )
                                    if current_failures >= max_gemini_failures_per_task:
                                        blocked_task_ids.add(task_id)
                                        print(f"[run] task blocked for this run due to repeated Gemini failures: {task_id}")
                            if task_id:
                                task_state = _persist_task_state_fields(
                                    cfg,
                                    task_id,
                                    task_state,
                                    last_error=str(exc),
                                )
                            if _is_non_retriable_gemini_error(exc):
                                print("[run] non-retriable Gemini error detected; stopping run.")
                                episode_report.failure_class = FailureClass.GEMINI_TRANSPORT_FAILURE
                                episode_report.notes.append("non-retriable gemini re-query failure")
                                page, context = _finalize_current_episode_v2(
                                    cfg=cfg,
                                    report=episode_report,
                                    task_state=task_state,
                                    runtime=episode_runtime,
                                    bootstrap_page=episode_bootstrap_page,
                                    bootstrap_context=episode_bootstrap_context,
                                    room_url=room_url,
                                    page=page,
                                    segments=segments,
                                    validation_report=validation_report,
                                    result=result,
                                    validation_tracker=validation_tracker,
                                    reason="room-after-episode-failure-gemini-requery-v2",
                                )
                                break
                            failure_mode = _episode_failure_mode()
                            if failure_mode == "raise":
                                raise
                            if failure_mode == "stop":
                                print(
                                    "[run] episode failure budget exhausted "
                                    f"({consecutive_episode_failures}>{max_episode_failures_per_run}); stopping run."
                                )
                                episode_report.failure_class = FailureClass.GEMINI_TRANSPORT_FAILURE
                                episode_report.notes.append("gemini re-query failure budget exhausted")
                                page, context = _finalize_current_episode_v2(
                                    cfg=cfg,
                                    report=episode_report,
                                    task_state=task_state,
                                    runtime=episode_runtime,
                                    bootstrap_page=episode_bootstrap_page,
                                    bootstrap_context=episode_bootstrap_context,
                                    room_url=room_url,
                                    page=page,
                                    segments=segments,
                                    validation_report=validation_report,
                                    result=result,
                                    validation_tracker=validation_tracker,
                                    reason="room-after-episode-failure-gemini-requery-v2",
                                )
                                break
                            if room_url:
                                print("[run] returning to room page after episode failure.")
                                try:
                                    _goto_with_retry(
                                        page,
                                        room_url,
                                        wait_until="domcontentloaded",
                                        timeout_ms=45000,
                                        cfg=cfg,
                                        reason="room-after-episode-failure-gemini-requery",
                                    )
                                except Exception:
                                    pass
                            retry_delay_sec = episode_failure_retry_delay_sec
                            if quota_error:
                                quota_retry_delay_sec = max(
                                    0.0,
                                    float(_cfg_get(cfg, "run.gemini_quota_retry_delay_sec", 15.0)),
                                )
                                if quota_wait_sec <= 0:
                                    quota_wait_sec = _register_quota_failure(task_id, exc, "gemini-requery")
                                retry_delay_sec = max(retry_delay_sec, quota_retry_delay_sec, quota_wait_sec)
                            if retry_delay_sec > 0:
                                print(f"[run] waiting {retry_delay_sec:.1f}s before retrying next episode.")
                                time.sleep(retry_delay_sec)
                            episode_report.failure_class = FailureClass.GEMINI_TRANSPORT_FAILURE
                            episode_report.notes.append("gemini re-query failed")
                            page, context = _finalize_current_episode_v2(
                                cfg=cfg,
                                report=episode_report,
                                task_state=task_state,
                                runtime=episode_runtime,
                                bootstrap_page=episode_bootstrap_page,
                                bootstrap_context=episode_bootstrap_context,
                                room_url=room_url,
                                page=page,
                                segments=segments,
                                validation_report=validation_report,
                                result=result,
                                validation_tracker=validation_tracker,
                                reason="room-after-episode-failure-gemini-requery-v2",
                            )
                            continue
                        if bool(_cfg_get(cfg, "run.chat_only_mode", False)) and task_id:
                            task_state = _capture_episode_step(cfg, page, task_id, task_state, "after_chat_labels")
                        if execute and execute_require_video_context:
                            meta = labels_payload.get("_meta", {}) if isinstance(labels_payload, dict) else {}
                            video_attached = bool(meta.get("video_attached", False))
                            mode = str(meta.get("mode", "unknown"))
                            
                            for fname in meta.get("uploaded_file_names", []):
                                if fname not in gemini_uploaded_file_names:
                                    gemini_uploaded_file_names.append(fname)

                            if not video_attached:
                                raise RuntimeError(
                                    "Execute blocked: Gemini response is text-only (no video context). "
                                    "Video review is required before apply/complete."
                                )
                            print(f"[gemini] execute guard: video context confirmed ({mode}).")
                        post_ops = _normalize_operations(labels_payload, cfg=cfg)
                        if post_ops:
                            print("[run] ignoring operations in second pass (labels-only pass).")
                        if task_id:
                            _save_cached_labels(cfg, task_id, labels_payload)
                        if task_id:
                            task_state = _persist_task_state_fields(
                                cfg,
                                task_id,
                                task_state,
                                labels_ready=True,
                                last_error="",
                                **_episode_model_state_updates(cfg, labels_payload, task_state),
                            )
                elif operations and not execute:
                    print("[run] operations skipped (dry-run mode).")

                _save_outputs(cfg, segments, prompt, labels_payload, task_id=task_id)

                segment_plan = _normalize_segment_plan(labels_payload, segments, cfg=cfg)
                no_action_rewrites = _rewrite_no_action_pauses_in_plan(segment_plan, cfg)
                if no_action_rewrites:
                    print(f"[policy] rewrote short no-action pauses: {no_action_rewrites}")
                if task_id:
                    _save_task_text_files(cfg, task_id, segments, segment_plan)
                    task_state = _capture_episode_step(cfg, page, task_id, task_state, "labels_ready")

                if bool(_cfg_get(cfg, "run.enable_policy_gate", True)):
                    if task_id:
                        task_state = _capture_episode_step(
                            cfg,
                            page,
                            task_id,
                            task_state,
                            "before_policy_gate_compare",
                        )
                    policy_result = _orchestrator._process_policy_gate_and_compare(
                        cfg=cfg,
                        page=page,
                        segments=segments,
                        prompt=prompt,
                        video_file=video_file,
                        labels_payload=labels_payload,
                        segment_plan=segment_plan,
                        episode_no=episode_no,
                        task_id=task_id,
                        execute=execute,
                        execute_require_video_context=execute_require_video_context,
                        gemini_uploaded_file_names=gemini_uploaded_file_names,
                        resume_from_artifacts=resume_from_artifacts,
                        task_state=task_state,
                        heartbeat=_touch_watchdog,
                        enable_structural_actions=enable_structural_actions,
                        requery_after_structural_actions=requery_after_structural_actions,
                        validation_tracker=validation_tracker,
                    )
                    segments = policy_result["segments"]
                    prompt = policy_result["prompt"]
                    labels_payload = policy_result["labels_payload"]
                    segment_plan = policy_result["segment_plan"]
                    validation_report = policy_result["validation_report"]
                    warnings = policy_result["warnings"]
                    errors = policy_result["errors"]
                    report_task_id = policy_result["report_task_id"]
                    repair_skipped_reason = str(policy_result.get("repair_skipped_reason", "") or "").strip()
                    if isinstance(policy_result.get("task_state"), dict):
                        task_state = policy_result["task_state"]
                    if task_id:
                        task_state = _capture_episode_step(
                            cfg,
                            page,
                            task_id,
                            task_state,
                            "after_policy_gate_compare",
                        )
                    if errors:
                        payload_meta = (
                            labels_payload.get("_meta", {})
                            if isinstance(labels_payload, dict)
                            else {}
                        )
                        terminal_repair_failure = bool(
                            isinstance(payload_meta, dict) and payload_meta.get("repair_fail_closed", False)
                        )
                        if repair_skipped_reason:
                            print(f"[policy] auto-repair skipped: {repair_skipped_reason}")
                        print(f"[policy] validation errors: {len(errors)}")
                        for item in errors[:20]:
                            print(f"  - {item}")
                        if bool(_cfg_get(cfg, "run.block_apply_on_validation_fail", True)):
                            print("[run] policy gate blocked apply for this episode.")
                            if task_id:
                                if terminal_repair_failure:
                                    task_state = _persist_task_stage_status(
                                        cfg,
                                        task_id,
                                        task_state,
                                        stage="repair",
                                        status="failed",
                                        detail="targeted repair exhausted before apply",
                                        last_error="terminal_repair_failure",
                                        terminal_failure_kind="terminal_repair_failure",
                                    )
                                task_state = _capture_episode_step(
                                    cfg,
                                    page,
                                    task_id,
                                    task_state,
                                    "policy_gate_blocked",
                                    include_html=True,
                                )
                            if task_id:
                                blocked_task_ids.add(task_id)
                                if skip_duplicate_task_in_run:
                                    seen_task_ids.add(task_id)
                                _invalidate_cached_labels(cfg, task_id)
                            episode_report.failure_class = FailureClass.POLICY_FAILURE
                            if any("low confidence" in str(e).lower() for e in errors):
                                episode_report.failure_class = FailureClass.HALLUCINATION_FAILURE
                                episode_report.retry_reason = RetryReason.LOW_CONFIDENCE
                            if any("escalation_flag" in str(e).lower() for e in errors):
                                episode_report.failure_class = FailureClass.HALLUCINATION_FAILURE
                                episode_report.retry_reason = "model_escalation"
                                
                            episode_report.desync_detected = any(
                                "desync" in str(item).lower()
                                for item in errors + warnings
                            )
                            if keep_alive_when_idle and max_episodes_per_run <= 0 and not terminal_repair_failure:
                                if room_url:
                                    print("[run] returning to room page after policy gate block.")
                                    try:
                                        _goto_with_retry(
                                            page,
                                            room_url,
                                            wait_until="domcontentloaded",
                                            timeout_ms=45000,
                                            cfg=cfg,
                                            reason="room-after-policy-block",
                                        )
                                    except Exception:
                                        pass
                                _respect_episode_delay(cfg)
                                page, context = _finalize_current_episode_v2(
                                    cfg=cfg,
                                    report=episode_report,
                                    task_state=task_state,
                                    runtime=episode_runtime,
                                    bootstrap_page=episode_bootstrap_page,
                                    bootstrap_context=episode_bootstrap_context,
                                    room_url=room_url,
                                    page=page,
                                    segments=segments,
                                    validation_report=validation_report,
                                    result=result,
                                    validation_tracker=validation_tracker,
                                    reason="room-after-policy-block-v2",
                                )
                                continue
                            page, context = _finalize_current_episode_v2(
                                cfg=cfg,
                                report=episode_report,
                                task_state=task_state,
                                runtime=episode_runtime,
                                bootstrap_page=episode_bootstrap_page,
                                bootstrap_context=episode_bootstrap_context,
                                room_url=room_url,
                                page=page,
                                segments=segments,
                                validation_report=validation_report,
                                result=result,
                                validation_tracker=validation_tracker,
                                reason="room-after-policy-block-v2",
                            )
                            break

                label_map = _normalize_label_map_from_plan(segment_plan)
                print(f"[gemini] usable labels: {len(label_map)}")
                pre_skipped_unchanged = 0
                if bool(_cfg_get(cfg, "run.skip_unchanged_labels", True)):
                    label_map, pre_skipped_unchanged = _filter_unchanged_label_map(label_map, segments)
                    if pre_skipped_unchanged:
                        print(f"[run] pre-skip unchanged labels: {pre_skipped_unchanged}")

                if dry_run:
                    print("[run] dry-run enabled; no labels were applied to Atlas")
                    page, context = _finalize_current_episode_v2(
                        cfg=cfg,
                        report=episode_report,
                        task_state=task_state,
                        runtime=episode_runtime,
                        bootstrap_page=episode_bootstrap_page,
                        bootstrap_context=episode_bootstrap_context,
                        room_url=room_url,
                        page=page,
                        segments=segments,
                        validation_report=validation_report,
                        result=result,
                        validation_tracker=validation_tracker,
                        reason="room-after-dry-run-v2",
                    )
                    break

                if resume_from_artifacts and resume_skip_apply_steps_when_done and bool(task_state.get("timestamps_done")):
                    ts_result = {"adjusted": 0, "failed": []}
                    print("[run] resume: skipping timestamp adjustments (already completed previously).")
                else:
                    if task_id:
                        task_state = _mark_task_stage(
                            task_id,
                            task_state,
                            stage="apply_labels",
                            status="running",
                            progress_total=len(label_map),
                            detail="applying labels and submit flow",
                        )
                    ts_result = apply_timestamp_adjustments(page, cfg, segments, segment_plan)
                print(f"[run] timestamp adjustments: {ts_result['adjusted']}")
                if ts_result["failed"]:
                    print("[run] timestamp adjustment failures:")
                    for item in ts_result["failed"]:
                        print(f"  - {item}")
                elif task_id:
                    task_state = _persist_task_state_fields(
                        cfg,
                        task_id,
                        task_state,
                        timestamps_done=True,
                        last_error="",
                    )

                if resume_from_artifacts and resume_skip_apply_steps_when_done and bool(task_state.get("labels_applied")):
                    print("[run] resume: skipping label apply (already completed previously).")
                    completed_from_resume = False
                    submit_status: Dict[str, Any] = {
                        "submit_attempted": False,
                        "submit_verified": False,
                        "submit_verification_reason": "resume_auto_submit_disabled",
                        "page_url_before_submit": "",
                        "page_url_after_submit": "",
                    }
                    if allow_resume_auto_submit:
                        submit_status = _submit_episode(page, cfg, return_details=True)
                        completed_from_resume = bool(submit_status.get("submit_verified"))
                    else:
                        print("[run] resume auto-submit disabled; not clicking Complete from stale state.")
                    result = {
                        "applied": 0,
                        "skipped_unchanged": 0,
                        "failed": [],
                        "completed": completed_from_resume,
                        "submit_status": submit_status,
                    }
                else:
                    if task_id:
                        task_state = _capture_episode_step(
                            cfg,
                            page,
                            task_id,
                            task_state,
                            "before_apply_labels",
                        )
                    _respect_major_step_pause(cfg, "apply_labels", heartbeat=_touch_watchdog)

                    # ── Pre-apply consistency check ──
                    try:
                        from src.rules.consistency import validate_pre_submit_consistency
                    except ImportError:
                        validate_pre_submit_consistency = None
                    if validate_pre_submit_consistency is not None:
                        consist_res = validate_pre_submit_consistency(page, cfg, segment_plan, segments)
                        if not consist_res.get("consistent", True):
                            mismatches = [
                                str(item).strip()
                                for item in (consist_res.get("mismatches", []) or [])
                                if str(item).strip()
                            ]
                            if not mismatches:
                                mismatches = ["pre-apply consistency check failed"]
                            print(f"[run] pre-apply consistency check failed. Mismatches: {mismatches}")
                            consecutive_episode_failures += 1
                            consistency_error_text = "; ".join(mismatches[:10])
                            _capture_debug_artifacts(page, cfg, prefix="debug_episode_failure")
                            if task_id:
                                task_state = _persist_task_state_fields(
                                    cfg,
                                    task_id,
                                    task_state,
                                    last_error=consistency_error_text,
                                )
                                task_state = _capture_episode_step(
                                    cfg,
                                    page,
                                    task_id,
                                    task_state,
                                    "pre_apply_consistency_failed",
                                    include_html=True,
                                )
                            failure_mode = _episode_failure_mode()
                            consistency_failure_class = FailureClass.DESYNC_BLOCK
                            if any(
                                token in consistency_error_text.lower()
                                for token in (
                                    "target page, context or browser has been closed",
                                    "page has been closed",
                                    "targetclosederror",
                                    "live dom unavailable",
                                )
                            ):
                                consistency_failure_class = FailureClass.APPLY_FAILURE
                            if failure_mode == "raise":
                                raise RuntimeError(
                                    "Pre-apply consistency check failed: "
                                    f"{consistency_error_text}"
                                )
                            if failure_mode == "stop":
                                print(
                                    "[run] episode failure budget exhausted "
                                    f"({consecutive_episode_failures}>{max_episode_failures_per_run}); stopping run."
                                )
                                episode_report.failure_class = consistency_failure_class
                                episode_report.notes.append("pre-apply consistency check failed")
                                page, context = _finalize_current_episode_v2(
                                    cfg=cfg,
                                    report=episode_report,
                                    task_state=task_state,
                                    runtime=episode_runtime,
                                    bootstrap_page=episode_bootstrap_page,
                                    bootstrap_context=episode_bootstrap_context,
                                    room_url=room_url,
                                    page=page,
                                    segments=segments,
                                    validation_report=validation_report,
                                    result=result,
                                    validation_tracker=validation_tracker,
                                    reason="room-after-episode-failure-pre-apply-consistency-v2",
                                )
                                break
                            if room_url:
                                print("[run] returning to room page after episode failure.")
                                try:
                                    _goto_with_retry(
                                        page,
                                        room_url,
                                        wait_until="domcontentloaded",
                                        timeout_ms=45000,
                                        cfg=cfg,
                                        reason="room-after-episode-failure-pre-apply-consistency",
                                    )
                                except Exception:
                                    pass
                            if episode_failure_retry_delay_sec > 0:
                                print(f"[run] waiting {episode_failure_retry_delay_sec:.1f}s before retrying next episode.")
                                _sleep_with_shutdown_heartbeat(
                                    episode_failure_retry_delay_sec,
                                    heartbeat_sec=min(30.0, max(5.0, watchdog_stale_threshold_sec / 4.0)),
                                    on_heartbeat=_touch_watchdog,
                                )
                            episode_report.failure_class = consistency_failure_class
                            episode_report.notes.append("pre-apply consistency check failed")
                            page, context = _finalize_current_episode_v2(
                                cfg=cfg,
                                report=episode_report,
                                task_state=task_state,
                                runtime=episode_runtime,
                                bootstrap_page=episode_bootstrap_page,
                                bootstrap_context=episode_bootstrap_context,
                                room_url=room_url,
                                page=page,
                                segments=segments,
                                validation_report=validation_report,
                                result=result,
                                validation_tracker=validation_tracker,
                                reason="room-after-episode-failure-pre-apply-consistency-v2",
                            )
                            continue

                    def _apply_progress_callback(event_type: str, payload: Dict[str, Any]) -> None:
                        nonlocal task_state
                        if not task_id:
                            return
                        apply_budget_state = dict(payload.get("apply_budget_state", {}) or {})
                        task_state = _persist_task_state_fields(
                            cfg,
                            task_id,
                            task_state,
                            apply_budget_state=apply_budget_state,
                            page_url=str(getattr(page, "url", "") or "").strip(),
                        )
                        if event_type in {"apply_start", "submit_verifying"}:
                            task_state = _mark_task_stage(
                                task_id,
                                task_state,
                                stage="submit" if event_type == "submit_verifying" else "apply_labels",
                                status="running",
                                progress_current=int(apply_budget_state.get("applied_count", 0) or 0),
                                progress_total=int(payload.get("total_targets", len(label_map)) or len(label_map)),
                                detail=(
                                    "verifying submit outcome"
                                    if event_type == "submit_verifying"
                                    else f"apply event: {event_type}"
                                ),
                            )
                        append_execution_journal_event(
                            cfg,
                            episode_id=task_id,
                            event_type=str(event_type or "").strip() or "apply_progress",
                            stage="submit_verifying" if event_type == "submit_verifying" else "applying",
                            reason=str(payload.get("message", "") or payload.get("stop_reason", "") or "").strip(),
                            task_state=task_state,
                            payload=dict(payload),
                            run_id=str(task_state.get("run_id", task_state.get("context_id", "")) or "").strip(),
                            context_id=str(task_state.get("context_id", "") or "").strip(),
                            page_url=str(getattr(page, "url", "") or "").strip(),
                        )

                    result = apply_labels(
                        page,
                        cfg,
                        label_map,
                        episode_id=task_id,
                        segment_plan=segment_plan,
                        source_segments=segments,
                        heartbeat=_touch_watchdog,
                        validation_tracker=validation_tracker,
                        progress_callback=_apply_progress_callback if task_id else None,
                    )
                    if task_id:
                        task_state = _mark_task_stage(
                            task_id,
                            task_state,
                            stage="apply_labels",
                            status="completed",
                            progress_current=len(label_map),
                            progress_total=len(label_map),
                            detail="apply_labels finished",
                        )
                if task_id:
                    task_state = _capture_episode_step(
                        cfg,
                        page,
                        task_id,
                        task_state,
                        "after_apply_labels",
                    )
                print(f"[run] applied labels: {result['applied']}")
                skipped_total = int(pre_skipped_unchanged) + int(result.get("skipped_unchanged", 0))
                if skipped_total:
                    print(f"[run] skipped unchanged labels: {skipped_total}")
                if result["failed"]:
                    print("[run] failures:")
                    for item in result["failed"]:
                        print(f"  - {item}")
                elif task_id:
                    task_state = _persist_task_state_fields(
                        cfg,
                        task_id,
                        task_state,
                        labels_applied=True,
                        last_error="",
                    )
                if bool(result.get("submit_guard_blocked")):
                    print("[run] episode submit skipped by submit guard.")
                    for item in result.get("submit_guard_reasons", [])[:10]:
                        print(f"  - {item}")
                    if task_id:
                        guard_reason = "; ".join(str(item) for item in result.get("submit_guard_reasons", [])[:10])
                        task_state = _persist_submit_outcome(
                            cfg,
                            task_id,
                            task_state,
                            result,
                            page=page,
                            episode_submitted=False,
                            last_error=(
                                f"submit_guard_blocked:{guard_reason}"
                                if guard_reason
                                else "submit_guard_blocked"
                            ),
                        )
                elif not result.get("completed"):
                    print("[run] episode submit could not be fully verified (Complete/Quality confirmation not fully observed).")
                    if task_id:
                        submit_status = result.get("submit_status", {}) if isinstance(result, dict) else {}
                        submit_reason = str(
                            submit_status.get("submit_verification_reason", "")
                            if isinstance(submit_status, dict)
                            else ""
                        ).strip() or "unknown"
                        task_state = _persist_submit_outcome(
                            cfg,
                            task_id,
                            task_state,
                            result,
                            page=page,
                            episode_submitted=False,
                            last_error=f"submit_unverified:{submit_reason}",
                        )
                        task_state = _persist_task_stage_status(
                            cfg,
                            task_id,
                            task_state,
                            stage="submit",
                            status="failed",
                            detail="submit verification failed",
                            last_error=f"submit_unverified:{submit_reason}",
                            terminal_failure_kind="terminal_submit_failure",
                        )
                        blocked_task_ids.add(task_id)
                    if task_id:
                        task_state = _capture_episode_step(
                            cfg,
                            page,
                            task_id,
                            task_state,
                            "after_submit_attempt",
                            include_html=True,
                        )
                    if release_and_reserve_on_submit_unverified:
                        _release_then_reserve_cycle("submit could not be fully verified")
                elif task_id:
                    task_state = _persist_submit_outcome(
                        cfg,
                        task_id,
                        task_state,
                        result,
                        page=page,
                        episode_submitted=True,
                        last_error="",
                    )
                if task_id and result.get("completed"):
                    task_state = _capture_episode_step(
                        cfg,
                        page,
                        task_id,
                        task_state,
                        "after_submit_attempt",
                        include_html=True,
                    )
                if bool(result.get("submit_guard_blocked")):
                    episode_report.failure_class = FailureClass.SUBMIT_GUARD_BLOCK
                    episode_report.submit_blocked = True
                elif not result.get("completed"):
                    episode_report.failure_class = FailureClass.SUBMIT_VERIFICATION_FAILURE
                else:
                    episode_report.failure_class = ""

                if task_id:
                    seen_task_ids.add(task_id)
                    if task_id in video_prepare_failures_by_task:
                        video_prepare_failures_by_task.pop(task_id, None)
                    if task_id in gemini_failures_by_task:
                        gemini_failures_by_task.pop(task_id, None)
                consecutive_episode_failures = 0
                _touch_watchdog()

                if room_url and (result.get("completed") or not strict_terminal_submit_handling):
                    print("[run] returning to room page for next episode.")
                    _goto_with_retry(
                        page,
                        room_url,
                        wait_until="domcontentloaded",
                        timeout_ms=45000,
                        cfg=cfg,
                        reason="room-after-episode",
                    )
                    page.wait_for_timeout(1500)
                page, context = _finalize_current_episode_v2(
                    cfg=cfg,
                    report=episode_report,
                    task_state=task_state,
                    runtime=episode_runtime,
                    bootstrap_page=episode_bootstrap_page,
                    bootstrap_context=episode_bootstrap_context,
                    room_url=room_url,
                    page=page,
                    segments=segments,
                    validation_report=validation_report,
                    result=result,
                    validation_tracker=validation_tracker,
                    reason="room-after-episode-v2",
                )
                bootstrap_page = page if page is not None else bootstrap_page
                bootstrap_context = context if context is not None else bootstrap_context
                if not result.get("completed") and strict_terminal_submit_handling:
                    print("[run] terminal submit failure recorded; stopping run for manual review.")
                    break
                _respect_episode_delay(cfg)
        except Exception as exc:
            _capture_debug_artifacts(page, cfg, prefix="debug_failure")
            raise
        finally:
            _ACTIVE_HEARTBEAT_CALLBACK = None
            if release_all_after_batch:
                try:
                    if browser is not None and page is not None and not page.is_closed():
                        released = _release_all_reserved_episodes(page, cfg)
                        if released:
                            print("[run] released reserved episodes during end-of-run cleanup.")
                except Exception as exc:
                    print(f"[run] release-all cleanup skipped: {exc}")
            # ?????? Cleanup: Gemini uploaded files ???????????????????????????????????????????????????????????????
            if gemini_uploaded_file_names:
                print(f"[gemini] cleaning up {len(gemini_uploaded_file_names)} uploaded file(s)...")
                cleanup_api_key = _resolve_gemini_key(str(_cfg_get(cfg, "gemini.api_key", "")).strip())
                if not cleanup_api_key:
                    cleanup_api_key = _resolve_gemini_fallback_key(
                        str(_cfg_get(cfg, "gemini.fallback_api_key", "")).strip()
                    )
                for _fname in gemini_uploaded_file_names:
                    _cleanup_gemini_uploaded_file(cleanup_api_key, _fname, cfg)
                gemini_uploaded_file_names.clear()
            # ?????? Cleanup: Browser state ???????????????????????????????????????????????????????????????????????????????????????
            try:
                if _is_authenticated_page(page):
                    _ensure_parent(state_path)
                    context.storage_state(path=str(state_path))
                    print(f"[auth] saved state: {state_path}")
                else:
                    print("[auth] skip saving state (session not authenticated).")
            except Exception as exc:
                print(f"[auth] skip saving state during cleanup: {exc}")
            _cleanup_browser_connections(
                context=context,
                browser=browser,
                atlas_browser_mode=atlas_browser_mode,
                gemini_browser=gemini_browser,
                owns_gemini_browser=owns_gemini_browser,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Atlas browser auto-solver (Atlas -> Gemini -> optional autofill)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply labels to Atlas. Without this flag, script runs in dry-run mode.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Override run.max_episodes_per_run for this process only.",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="",
        help="Override gemini.model for this process only (e.g., gemini-3.1-pro-preview).",
    )
    parser.add_argument(
        "--use-fallback-key",
        action="store_true",
        help="Use GEMINI_API_KEY_FALLBACK as primary key for this process only.",
    )
    return parser.parse_args()


def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    run_cfg = cfg.setdefault("run", {})
    gem_cfg = cfg.setdefault("gemini", {})
    if not isinstance(run_cfg, dict):
        cfg["run"] = {}
        run_cfg = cfg["run"]
    if not isinstance(gem_cfg, dict):
        cfg["gemini"] = {}
        gem_cfg = cfg["gemini"]

    if args.max_episodes is not None:
        run_cfg["max_episodes_per_run"] = max(1, int(args.max_episodes))
        run_cfg["recycle_after_max_episodes"] = False
        print(f"[cli] override: run.max_episodes_per_run={run_cfg['max_episodes_per_run']}")
        print("[cli] override: run.recycle_after_max_episodes=False")

    model_override = str(args.gemini_model or "").strip()
    if model_override:
        gem_cfg["model"] = model_override
        print(f"[cli] override: gemini.model={model_override}")

    if bool(args.use_fallback_key):
        fallback_key = _resolve_gemini_fallback_key(str(gem_cfg.get("fallback_api_key", "") or ""))
        if not fallback_key:
            raise RuntimeError(
                "--use-fallback-key requested but no fallback key found. "
                "Set GEMINI_API_KEY_FALLBACK (or gemini.fallback_api_key)."
            )
        gem_cfg["api_key"] = fallback_key
        gem_cfg["fallback_api_key"] = ""
        gem_cfg["quota_fallback_enabled"] = False
        print("[cli] override: using fallback Gemini API key as primary for this run.")


def main() -> None:
    print(f"[build] atlas_web_auto_solver {_SCRIPT_BUILD}")
    args = parse_args()
    cfg = load_config(Path(args.config))
    _apply_cli_overrides(cfg, args)
    run(cfg, execute=bool(args.execute))


if __name__ == "__main__":
    main()

