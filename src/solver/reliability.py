"""Retry, classification, and per-episode observability helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any, Dict, List, Optional


class RetryStage:
    FULL_GENERATE = "full_generate"
    TARGETED_REPAIR_1 = "targeted_repair_1"
    TARGETED_REPAIR_2 = "targeted_repair_2"
    FULL_RESET_REGENERATE = "full_reset_regenerate"
    TRANSPORT = "transport"


class RetryReason:
    TIMEOUT = "timeout"
    EMPTY_RESPONSE = "empty_response"
    INVALID_JSON = "invalid_json"
    HALLUCINATED_INDICES = "hallucinated_indices"
    NO_NEW_ASSISTANT_MESSAGE = "no_new_assistant_message"
    STALE_RESPONSE = "stale_response"
    THREAD_CONTAMINATION = "thread_contamination"
    POLICY_OVERLONG = "policy_overlong"
    DESYNC = "desync"
    SUBMIT_GUARD = "submit_guard"
    PAGE_CRASH = "page_crash"
    UNKNOWN = "unknown"


class FailureClass:
    DESYNC_BLOCK = "desync_block"
    GEMINI_TRANSPORT_FAILURE = "gemini_transport_failure"
    GEMINI_INTEGRITY_FAILURE = "gemini_integrity_failure"
    POLICY_FAILURE = "policy_failure"
    APPLY_FAILURE = "apply_failure"
    SUBMIT_GUARD_BLOCK = "submit_guard_block"
    SUBMIT_VERIFICATION_FAILURE = "submit_verification_failure"


TRANSPORT_BACKOFF_SECONDS: List[float] = [1.0, 3.0, 7.0]


@dataclass
class EpisodeReport:
    episode_id: str
    context_id: str = ""
    gemini_session_id: str = ""
    request_id: str = ""
    segment_checksum: str = ""
    retry_stage: str = ""
    retry_reason: str = ""
    desync_detected: bool = False
    gemini_latency_ms: int = 0
    segment_count: int = 0
    page_url: str = ""
    submit_blocked: bool = False
    submit_verification_reason: str = ""
    failure_class: str = ""
    live_validation_report_path: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeminiRequestContext:
    request_id: str
    mode: str
    expected_schema: str
    requested_indices: List[int] = field(default_factory=list)
    episode_id: str = ""
    segments_checksum: str = ""
    baseline_message_count: int = 0
    baseline_response_hash: str = ""
    baseline_preview: str = ""
    started_at_utc: str = ""
    prompt_marker: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]] = None) -> "GeminiRequestContext":
        raw = payload if isinstance(payload, dict) else {}
        requested = raw.get("requested_indices", [])
        if not isinstance(requested, list):
            requested = []
        return cls(
            request_id=str(raw.get("request_id", "") or "").strip(),
            mode=str(raw.get("mode", "") or "").strip(),
            expected_schema=str(raw.get("expected_schema", "") or "").strip(),
            requested_indices=[
                int(item) for item in requested if str(item).strip() and int(item or 0) > 0
            ],
            episode_id=str(raw.get("episode_id", "") or "").strip(),
            segments_checksum=str(raw.get("segments_checksum", "") or "").strip(),
            baseline_message_count=max(0, int(raw.get("baseline_message_count", 0) or 0)),
            baseline_response_hash=str(raw.get("baseline_response_hash", "") or "").strip(),
            baseline_preview=str(raw.get("baseline_preview", "") or "").strip(),
            started_at_utc=str(raw.get("started_at_utc", "") or "").strip(),
            prompt_marker=str(raw.get("prompt_marker", "") or "").strip(),
        )


@dataclass
class ApplyBudgetState:
    target_count: int
    applied_count: int = 0
    skipped_count: int = 0
    last_progress_at: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    deadline_at: float = 0.0
    budget_extensions: int = 0
    started_at: float = field(default_factory=time.time)
    status: str = "active"

    def mark_progress(self, applied_delta: int = 1, *, skipped_delta: int = 0) -> None:
        self.applied_count += max(0, int(applied_delta or 0))
        self.skipped_count += max(0, int(skipped_delta or 0))
        self.last_progress_at = time.time()
        self.consecutive_failures = 0
        self.status = "active"

    def mark_failure(self) -> None:
        self.consecutive_failures += 1
        self.status = "active"

    def extend_deadline(self, extra_sec: float) -> None:
        extra = max(0.0, float(extra_sec or 0.0))
        if extra <= 0:
            return
        self.deadline_at += extra
        self.budget_extensions += 1

    def mark_completed(self) -> None:
        self.status = "completed"

    def mark_timed_out(self) -> None:
        self.status = "timed_out"

    def mark_failed(self) -> None:
        self.status = "failed"

    def stalled_for_sec(self, now: Optional[float] = None) -> float:
        current = time.time() if now is None else float(now)
        return max(0.0, current - float(self.last_progress_at or current))

    def elapsed_sec(self, now: Optional[float] = None) -> float:
        current = time.time() if now is None else float(now)
        return max(0.0, current - float(self.started_at or current))

    def remaining_sec(self, now: Optional[float] = None) -> float:
        current = time.time() if now is None else float(now)
        return max(0.0, float(self.deadline_at or 0.0) - current)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["elapsed_sec"] = round(self.elapsed_sec(), 2)
        payload["remaining_sec"] = round(self.remaining_sec(), 2)
        payload["stalled_for_sec"] = round(self.stalled_for_sec(), 2)
        return payload

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]] = None) -> "ApplyBudgetState":
        raw = payload if isinstance(payload, dict) else {}
        return cls(
            target_count=max(0, int(raw.get("target_count", 0) or 0)),
            applied_count=max(0, int(raw.get("applied_count", 0) or 0)),
            skipped_count=max(0, int(raw.get("skipped_count", 0) or 0)),
            last_progress_at=float(raw.get("last_progress_at", time.time()) or time.time()),
            consecutive_failures=max(0, int(raw.get("consecutive_failures", 0) or 0)),
            deadline_at=float(raw.get("deadline_at", 0.0) or 0.0),
            budget_extensions=max(0, int(raw.get("budget_extensions", 0) or 0)),
            started_at=float(raw.get("started_at", time.time()) or time.time()),
            status=str(raw.get("status", "active") or "active").strip() or "active",
        )


@dataclass
class SubmitOutcome:
    submit_attempted: bool = False
    submit_verified: bool = False
    submit_verification_reason: str = ""
    page_url_before_submit: str = ""
    page_url_after_submit: str = ""
    terminal_failure: bool = False
    complete_button_clicked: bool = False
    complete_button_retried: bool = False
    submit_modal_already_open: bool = False
    saw_no_edits_modal: bool = False
    saw_quality_review_modal: bool = False
    saw_post_submit_transition: bool = False
    no_edits_confirmed: bool = False
    quality_review_confirmed: bool = False
    manual_submit_watch_used: bool = False
    manual_submit_detected: bool = False
    manual_submit_watch_reason: str = ""
    manual_submit_watch_signal: str = ""
    manual_submit_watch_timed_out: bool = False
    manual_submit_watch_elapsed_sec: float = 0.0
    last_error: str = ""
    dashboard_verified: bool = False
    dashboard_verify_method: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_status(
        cls,
        status: Optional[Dict[str, Any]] = None,
        *,
        terminal_failure: bool = False,
    ) -> "SubmitOutcome":
        payload = status if isinstance(status, dict) else {}
        field_names = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        kwargs = {
            key: payload.get(key)
            for key in field_names
            if key in payload
        }
        outcome = cls(**kwargs)
        outcome.terminal_failure = bool(terminal_failure or payload.get("terminal_failure", False))
        return outcome


def transport_backoff_seconds(attempt_number: int) -> float:
    index = max(0, int(attempt_number) - 1)
    if index >= len(TRANSPORT_BACKOFF_SECONDS):
        return TRANSPORT_BACKOFF_SECONDS[-1]
    return TRANSPORT_BACKOFF_SECONDS[index]


def classify_transport_failure(message: str) -> str:
    text = str(message or "").strip().lower()
    if not text:
        return RetryReason.UNKNOWN
    if "timed out" in text or "timeout" in text:
        return RetryReason.TIMEOUT
    if "empty output" in text or "empty response" in text:
        return RetryReason.EMPTY_RESPONSE
    if "invalid json" in text or "not json" in text:
        return RetryReason.INVALID_JSON
    if "upload was not confirmed" in text or "attachment required" in text:
        return RetryReason.PAGE_CRASH
    if "no assistant response appeared after baseline" in text:
        return RetryReason.NO_NEW_ASSISTANT_MESSAGE
    if "did not advance baseline" in text:
        return RetryReason.NO_NEW_ASSISTANT_MESSAGE
    if "stale parseable body" in text or "same preview reused after baseline" in text:
        return RetryReason.STALE_RESPONSE
    if "thread contamination" in text or "rejected stale candidates after baseline" in text:
        return RetryReason.THREAD_CONTAMINATION
    if (
        "page crash" in text
        or "target page, context or browser has been closed" in text
        or "chat input not visible on session page" in text
        or "chat input disappeared" in text
        or "transient error response" in text
        or "something went wrong" in text
        or "unable to complete" in text
        or "could you try again" in text
        or "try your request again" in text
    ):
        return RetryReason.PAGE_CRASH
    return RetryReason.UNKNOWN


__all__ = [
    "RetryStage",
    "RetryReason",
    "FailureClass",
    "EpisodeReport",
    "GeminiRequestContext",
    "ApplyBudgetState",
    "SubmitOutcome",
    "TRANSPORT_BACKOFF_SECONDS",
    "transport_backoff_seconds",
    "classify_transport_failure",
]
