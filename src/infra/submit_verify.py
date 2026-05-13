"""
submit_verify.py — Post-submit verification probe.

After the browser automation clicks Submit, this module verifies the episode
actually appeared on the Audit Dashboard by probing the page.

Design:
- Uses the same Playwright page/browser session to check the dashboard
- Falls back to body-text scanning if selectors are unavailable
- Returns a structured SubmitVerifyResult
- Does NOT block the pipeline — logs and returns status for observability
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.infra.logging_utils import build_print_logger as _build_print_logger
from src.infra.solver_config import _cfg_get

_logger = logging.getLogger(__name__)
print = _build_print_logger(_logger)


def _normalized_episode_tokens(*values: str) -> tuple[list[str], list[str]]:
    full_tokens: list[str] = []
    suffix_tokens: list[str] = []
    seen_full: set[str] = set()
    seen_suffix: set[str] = set()
    for raw in values:
        token = str(raw or "").strip().lower()
        if not token or token == "unknown":
            continue
        if token not in seen_full:
            full_tokens.append(token)
            seen_full.add(token)
        if len(token) >= 8:
            suffix = token[-8:]
            if suffix not in seen_suffix:
                suffix_tokens.append(suffix)
                seen_suffix.add(suffix)
    return full_tokens, suffix_tokens


def _dashboard_body_match_detail(body: str, full_tokens: list[str], suffix_tokens: list[str]) -> tuple[bool, str, str]:
    lower_body = str(body or "").lower()
    for token in full_tokens:
        if token in lower_body:
            return True, "dashboard_scan", "Episode ID found in dashboard body text"
    for suffix in suffix_tokens:
        if re.search(rf"\bepisode\s+{re.escape(suffix)}\b", lower_body):
            return True, "dashboard_scan_suffix", f"Episode suffix {suffix} found in dashboard body text"
    return False, "", ""


def _dashboard_link_match_detail(page: Any, full_tokens: list[str], suffix_tokens: list[str]) -> tuple[bool, str, str]:
    try:
        links = page.locator("a[href]")
        count = min(int(links.count() or 0), 80)
    except Exception:
        return False, "", ""

    for i in range(count):
        try:
            item = links.nth(i)
            href = str(item.get_attribute("href") or "").strip().lower()
            text = str(item.inner_text(timeout=500) or "").strip().lower()
        except Exception:
            continue
        for token in full_tokens:
            if token and token in href:
                return True, "dashboard_link", f"Dashboard link href contains episode id: {href}"
            if token and token in text:
                return True, "dashboard_link_text", f"Dashboard link text contains episode id: {text}"
        for suffix in suffix_tokens:
            if re.search(rf"/feedback/[a-z0-9]*{re.escape(suffix)}\b", href):
                return True, "dashboard_link_suffix", f"Dashboard link href contains episode suffix {suffix}: {href}"
            if re.search(rf"\bepisode\s+{re.escape(suffix)}\b", text):
                return True, "dashboard_link_suffix", f"Dashboard link text contains episode suffix {suffix}: {text}"
    return False, "", ""


@dataclass
class SubmitVerifyResult:
    """Outcome of a post-submit verification probe."""

    episode_id: str
    verified: bool           # True = confirmed on dashboard
    method: str              # "url_check" | "body_scan" | "toast_detected" | "timeout"
    detail: str = ""
    elapsed_sec: float = 0.0

    def summary_line(self) -> str:
        status = "VERIFIED" if self.verified else "UNVERIFIED"
        return (
            f"[submit_verify] {status} episode={self.episode_id} "
            f"method={self.method} elapsed={self.elapsed_sec:.1f}s"
        )


def verify_submit_completion(
    page: Any,
    cfg: Dict[str, Any],
    *,
    episode_id: str = "",
    task_id: str = "",
    timeout_sec: float = 12.0,
) -> SubmitVerifyResult:
    """
    Verify that a submission was actually processed by the Atlas backend.

    Strategy priority:
    1. Detect success toast/modal on current page
    2. Check if URL changed to a post-submit state
    3. Scan page body for confirmation text

    This is a best-effort probe — it does NOT retry or re-submit.
    """
    eid = episode_id or task_id or "unknown"
    start = time.time()
    deadline = start + max(1.0, timeout_sec)

    # ── Strategy 1: success toast / modal detection ──────────────────────
    success_toast_sel = str(_cfg_get(cfg, "atlas.selectors.submit_success_toast", "")).strip()
    success_patterns = [
        r"successfully submitted",
        r"submission complete",
        r"episode submitted",
        r"submitted successfully",
        r"labels saved",
    ]

    while time.time() < deadline:
        # Check for toast/modal via selector
        if success_toast_sel:
            try:
                from src.solver.browser import _first_visible_locator
                loc = _first_visible_locator(page, success_toast_sel, timeout_ms=500)
                if loc is not None:
                    elapsed = time.time() - start
                    print(f"[submit_verify] success toast detected for {eid} ({elapsed:.1f}s)")
                    return SubmitVerifyResult(
                        episode_id=eid,
                        verified=True,
                        method="toast_detected",
                        detail="Success toast/modal visible on page",
                        elapsed_sec=elapsed,
                    )
            except Exception:
                pass

        # Check body text for success patterns
        try:
            body = (page.inner_text("body") or "").lower()
        except Exception:
            body = ""

        for pattern in success_patterns:
            if re.search(pattern, body):
                elapsed = time.time() - start
                print(f"[submit_verify] success text matched for {eid}: '{pattern}' ({elapsed:.1f}s)")
                return SubmitVerifyResult(
                    episode_id=eid,
                    verified=True,
                    method="body_scan",
                    detail=f"Matched pattern: {pattern}",
                    elapsed_sec=elapsed,
                )

        # Check if URL changed away from label page (indicates submit navigated)
        current_url = (page.url or "").lower()
        if "/tasks/room/normal/label/" not in current_url and "/tasks/room/normal" in current_url:
            elapsed = time.time() - start
            print(f"[submit_verify] URL changed to room page for {eid} ({elapsed:.1f}s)")
            return SubmitVerifyResult(
                episode_id=eid,
                verified=True,
                method="url_check",
                detail=f"URL navigated to room: {page.url}",
                elapsed_sec=elapsed,
            )

        # Check for error indicators
        error_patterns = [
            r"submission failed",
            r"error submitting",
            r"failed to submit",
            r"could not submit",
        ]
        for pattern in error_patterns:
            if re.search(pattern, body):
                elapsed = time.time() - start
                print(f"[submit_verify] ERROR detected for {eid}: '{pattern}' ({elapsed:.1f}s)")
                return SubmitVerifyResult(
                    episode_id=eid,
                    verified=False,
                    method="body_scan",
                    detail=f"Error pattern matched: {pattern}",
                    elapsed_sec=elapsed,
                )

        page.wait_for_timeout(400)

    elapsed = time.time() - start
    print(f"[submit_verify] TIMEOUT: no confirmation detected for {eid} ({elapsed:.1f}s)")
    return SubmitVerifyResult(
        episode_id=eid,
        verified=False,
        method="timeout",
        detail=f"No success indicator found within {timeout_sec:.0f}s",
        elapsed_sec=elapsed,
    )


def verify_episode_on_dashboard(
    page: Any,
    cfg: Dict[str, Any],
    *,
    episode_id: str = "",
    task_id: str = "",
    timeout_sec: float = 15.0,
) -> SubmitVerifyResult:
    """
    Navigate to the Audit Dashboard and verify an episode is listed.

    This is a heavier check that navigates away from the current page.
    Only call after submit_completion is confirmed.
    """
    from src.solver.browser import _goto_with_retry

    eid = episode_id or task_id or "unknown"
    full_tokens, suffix_tokens = _normalized_episode_tokens(eid, task_id)
    start = time.time()

    feedback_url = str(_cfg_get(cfg, "atlas.feedback_url", "")).strip()
    if not feedback_url:
        feedback_url = "https://audit.atlascapture.io/feedback"

    try:
        _goto_with_retry(
            page, feedback_url,
            wait_until="domcontentloaded",
            timeout_ms=30000,
            cfg=cfg,
            reason="submit-verify-dashboard",
        )
    except Exception as exc:
        elapsed = time.time() - start
        return SubmitVerifyResult(
            episode_id=eid,
            verified=False,
            method="dashboard_nav_failed",
            detail=f"Could not navigate to dashboard: {str(exc)[:200]}",
            elapsed_sec=elapsed,
        )

    deadline = start + max(3.0, timeout_sec)
    while time.time() < deadline:
        try:
            body = (page.inner_text("body") or "")
        except Exception:
            body = ""

        matched, method, detail = _dashboard_link_match_detail(page, full_tokens, suffix_tokens)
        if matched:
            elapsed = time.time() - start
            print(f"[submit_verify] episode {eid} found on dashboard via {method} ({elapsed:.1f}s)")
            return SubmitVerifyResult(
                episode_id=eid,
                verified=True,
                method=method,
                detail=detail,
                elapsed_sec=elapsed,
            )

        matched, method, detail = _dashboard_body_match_detail(body, full_tokens, suffix_tokens)
        if matched:
            elapsed = time.time() - start
            print(f"[submit_verify] episode {eid} found on dashboard via {method} ({elapsed:.1f}s)")
            return SubmitVerifyResult(
                episode_id=eid,
                verified=True,
                method=method,
                detail=detail,
                elapsed_sec=elapsed,
            )

        page.wait_for_timeout(500)

    elapsed = time.time() - start
    print(f"[submit_verify] episode {eid} NOT found on dashboard ({elapsed:.1f}s)")
    return SubmitVerifyResult(
        episode_id=eid,
        verified=False,
        method="timeout",
        detail=f"Episode not found on dashboard within {timeout_sec:.0f}s",
        elapsed_sec=elapsed,
    )


__all__ = [
    "SubmitVerifyResult",
    "verify_submit_completion",
    "verify_episode_on_dashboard",
]
