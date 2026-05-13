"""Episode-scoped Gemini chat session manager for the v2 runtime path."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import atlas_triplet_compare as _chat

from src.infra.artifacts import _capture_step_artifacts
from src.infra.execution_journal import append_execution_journal_event
from src.infra.gemini_economics import resolve_stage_model
from src.infra.solver_config import _cfg_get
from src.solver.prompts.modes import resolve_mode_contract
from src.solver.desync import SegmentSnapshot
from src.solver.episode_runtime import EpisodeRuntime
from src.solver.reliability import (
    GeminiRequestContext,
    RetryReason,
    RetryStage,
    classify_transport_failure,
    transport_backoff_seconds,
)


@dataclass
class GeminiResult:
    request_id: str
    episode_id: str
    context_id: str
    retry_stage: str
    latency_ms: int
    raw_text: str
    parsed_payload: Dict[str, Any]
    validated_segments: List[Dict[str, Any]] = field(default_factory=list)
    attach_notes: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    session_restarted: bool = False
    raw_response_path: str = ""
    raw_response_meta_path: str = ""
    expected_schema: str = ""
    requested_indices: List[int] = field(default_factory=list)
    started_at_utc: str = ""
    request_context: Optional[GeminiRequestContext] = None
    acceptance_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_payload_segments(payload: Any) -> List[Dict[str, Any]]:
    raw_items: Any = payload
    if isinstance(raw_items, dict):
        raw_items = raw_items.get("segments", raw_items.get("labels", []))
    if not isinstance(raw_items, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("segment_index", item.get("index", 0)) or 0)
        except Exception:
            idx = 0
        if idx <= 0:
            continue
        start_sec = round(_safe_float(item.get("start_sec"), 0.0), 3)
        end_sec = round(_safe_float(item.get("end_sec"), start_sec), 3)
        out.append(
            {
                "segment_index": idx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "label": str(item.get("label", item.get("current_label", "")) or "").strip(),
            }
        )
    return out


def _extract_payload_operations(payload: Any, *, allow_merge: bool) -> List[Dict[str, Any]]:
    raw_ops: Any = payload
    if isinstance(raw_ops, dict):
        raw_ops = raw_ops.get("operations", [])
    if not isinstance(raw_ops, list):
        return []

    out: List[Dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in raw_ops:
        action = ""
        idx = 0
        if isinstance(item, dict):
            action = str(item.get("action", item.get("op", item.get("type", ""))) or "").strip().lower()
            try:
                idx = int(item.get("segment_index", item.get("index", item.get("segment", 0))) or 0)
            except Exception:
                idx = 0
        elif isinstance(item, str):
            parts = str(item or "").strip().lower().split()
            if len(parts) >= 2:
                action = parts[0]
                try:
                    idx = int(parts[-1])
                except Exception:
                    idx = 0
        if action not in {"split", "merge"}:
            continue
        if action == "merge" and not allow_merge:
            continue
        if idx <= 0:
            continue
        key = (action, idx)
        if key in seen:
            continue
        seen.add(key)
        out.append({"action": action, "segment_index": idx})
    out.sort(key=lambda item: (item["action"] != "split", -int(item["segment_index"])))
    return out


def _infer_retry_reason_from_errors(errors: Sequence[str], *, raw_text: str = "") -> str:
    joined = " | ".join(str(item or "").strip() for item in errors if str(item or "").strip()).lower()
    if "unknown segment" in joined or "outside requested scope" in joined or "duplicated segment" in joined:
        return RetryReason.HALLUCINATED_INDICES
    if "invalid json" in joined or "not contain valid json" in joined:
        return RetryReason.INVALID_JSON
    if "timed out" in joined or "timeout" in joined:
        return RetryReason.TIMEOUT
    if "target page, context or browser has been closed" in joined or "chat input not visible on session page" in joined:
        return RetryReason.PAGE_CRASH
    if not str(raw_text or "").strip():
        return RetryReason.EMPTY_RESPONSE
    return RetryReason.UNKNOWN


def validate_normalized_segments(
    normalized_segments: Sequence[Dict[str, Any]],
    source_segments: Sequence[Dict[str, Any]],
    *,
    requested_indices: Optional[Sequence[int]] = None,
    allow_partial: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    source_by_idx = {
        int(seg.get("segment_index", 0) or 0): seg
        for seg in (source_segments or [])
        if int(seg.get("segment_index", 0) or 0) > 0
    }
    requested = {
        int(idx)
        for idx in (requested_indices or [])
        if int(idx or 0) > 0
    }
    if not requested:
        requested = set(source_by_idx)

    validated: List[Dict[str, Any]] = []
    errors: List[str] = []
    seen: set[int] = set()

    for item in normalized_segments or []:
        idx = int(item.get("segment_index", 0) or 0)
        if idx <= 0:
            errors.append("response included non-positive segment index")
            continue
        if idx not in source_by_idx:
            errors.append(f"response referenced unknown segment {idx}")
            continue
        if idx in seen:
            errors.append(f"response duplicated segment {idx}")
            continue
        if requested and idx not in requested:
            errors.append(f"response referenced segment {idx} outside requested scope")
            continue
        start_sec = _safe_float(item.get("start_sec"), 0.0)
        end_sec = _safe_float(item.get("end_sec"), start_sec)
        if end_sec <= start_sec:
            errors.append(f"segment {idx} has invalid timestamps {start_sec:.3f}-{end_sec:.3f}")
            continue
        seen.add(idx)
        source = source_by_idx[idx]
        source_start = round(_safe_float(source.get("start_sec"), 0.0), 3)
        source_end = round(_safe_float(source.get("end_sec"), source_start), 3)
        validated.append(
            {
                "segment_index": idx,
                "start_sec": source_start,
                "end_sec": source_end,
                "label": str(item.get("label", "") or "").strip(),
            }
        )

    missing = sorted(idx for idx in requested if idx not in seen)
    if missing and not allow_partial:
        errors.append(f"response missing segment indices: {missing[:10]}")

    return validated, errors


def validate_payload_schema(
    payload: Any,
    *,
    expected_schema: str,
    requested_indices: Optional[Sequence[int]] = None,
    allow_merge: bool = False,
) -> List[str]:
    schema_name = str(expected_schema or "").strip().lower()
    if not schema_name:
        return []

    errors: List[str] = []
    requested = {
        int(idx)
        for idx in (requested_indices or [])
        if int(idx or 0) > 0
    }
    if not isinstance(payload, dict):
        errors.append(
            f"response must be a JSON object in {schema_name} mode"
        )
        return errors

    if schema_name == "segments_only":
        if "operations" in payload:
            errors.append('response unexpectedly included top-level "operations" in segments-only mode')
        if "segments" not in payload:
            errors.append('response missing top-level key "segments" in segments-only mode')
        elif not isinstance(payload.get("segments"), list):
            errors.append('response key "segments" must be a list in segments-only mode')
        return errors

    if schema_name == "operations_only":
        if "segments" in payload:
            errors.append('response unexpectedly included top-level "segments" in operations-only mode')
        if "operations" not in payload:
            errors.append('response missing top-level key "operations" in operations-only mode')
            return errors
        raw_ops = payload.get("operations")
        if not isinstance(raw_ops, list):
            errors.append('response key "operations" must be a list in operations-only mode')
            return errors
        normalized_ops = _extract_payload_operations(payload, allow_merge=allow_merge)
        if raw_ops and not normalized_ops:
            errors.append("response operations could not be normalized in operations-only mode")
            return errors
        if requested:
            for item in normalized_ops:
                idx = int(item.get("segment_index", 0) or 0)
                if idx not in requested:
                    errors.append(
                        f"response referenced segment {idx} outside requested repair scope"
                    )
        return errors

    errors.append(f"unknown expected schema: {schema_name}")
    return errors


class _HeartbeatGuard:
    def __init__(
        self,
        heartbeat: Optional[Callable[[], None]],
        *,
        interval_sec: float,
    ) -> None:
        self._heartbeat = heartbeat
        self._interval_sec = max(0.5, float(interval_sec or 10.0))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_HeartbeatGuard":
        if not callable(self._heartbeat):
            return self
        try:
            self._heartbeat()
        except Exception:
            pass

        def _worker() -> None:
            while not self._stop.wait(self._interval_sec):
                try:
                    if callable(self._heartbeat):
                        self._heartbeat()
                except Exception:
                    pass

        self._thread = threading.Thread(
            target=_worker,
            name="gemini-session-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(1.0, self._interval_sec))
        if callable(self._heartbeat):
            try:
                self._heartbeat()
            except Exception:
                pass


class GeminiSession:
    def __init__(self, runtime: EpisodeRuntime, cfg: Dict[str, Any]) -> None:
        self.runtime = runtime
        self.cfg = cfg
        self.session_id = uuid.uuid4().hex[:10]
        self._initialized = False
        self._accepted_outputs: List[Dict[str, Any]] = []
        self._last_expected_schema = ""
        if not isinstance(getattr(self.runtime, "task_state", None), dict):
            self.runtime.task_state = {}
        self.runtime.task_state.setdefault("context_id", str(runtime.context_id or "").strip())
        self.runtime.task_state.setdefault("run_id", str(runtime.context_id or "").strip())
        self.runtime.task_state.setdefault("gemini_session_id", self.session_id)

    @classmethod
    def start(cls, runtime: EpisodeRuntime, cfg: Dict[str, Any]) -> "GeminiSession":
        session = cls(runtime, cfg)
        session._ensure_page()
        session._journal_event(
            "gemini_session_started",
            stage="waiting_for_gemini",
            reason="session_initialized",
        )
        return session

    def _journal_event(
        self,
        event_type: str,
        *,
        stage: str = "",
        reason: str = "",
        request_context: Optional[GeminiRequestContext] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        effective_context = request_context if isinstance(request_context, GeminiRequestContext) else None
        append_execution_journal_event(
            self.cfg,
            episode_id=str(self.runtime.episode_id or "").strip(),
            event_type=event_type,
            stage=stage or "waiting_for_gemini",
            reason=reason,
            task_state=self.runtime.task_state,
            payload=payload,
            run_id=str(self.runtime.context_id or "").strip(),
            context_id=str(self.runtime.context_id or "").strip(),
            request_id=str(getattr(effective_context, "request_id", "") or "").strip(),
            mode=str(getattr(effective_context, "mode", "") or "").strip(),
            baseline_message_count=(
                int(getattr(effective_context, "baseline_message_count", 0) or 0)
                if effective_context is not None
                else None
            ),
            segments_checksum=str(getattr(effective_context, "segments_checksum", "") or "").strip(),
            page_url=str(getattr(self.runtime.gemini_page, "url", "") or "").strip(),
        )

    def _chat_url(self) -> str:
        return (
            str(
                _cfg_get(
                    self.cfg,
                    "gemini.chat_web_url",
                    "https://gemini.google.com/app",
                )
                or ""
            ).strip()
            or "https://gemini.google.com/app"
        )

    def _preserve_existing_thread(self, chat_url: str = "") -> bool:
        return bool(
            _cfg_get(
                self.cfg,
                "gemini.chat_web_preserve_existing_thread",
                False,
            )
        )

    def _session_entry_url(self, *, clean_thread: bool = False) -> str:
        chat_url = self._chat_url()
        if clean_thread or not self._preserve_existing_thread(chat_url):
            return _chat._normalize_gemini_chat_entry_url(chat_url, clean_thread=True)
        return _chat._normalize_gemini_chat_entry_url(chat_url, clean_thread=False)

    def _ensure_page(self) -> Any:
        page = self.runtime.gemini_page
        if page is None and self.runtime.gemini_context is None and self.runtime.gemini_browser is not None:
            try:
                page = self.runtime.reopen_gemini()
            except Exception:
                page = None
        if page is None and self.runtime.gemini_context is not None:
            page = self.runtime.gemini_context.new_page()
            self.runtime.gemini_page = page
        if page is None:
            raise RuntimeError("GeminiSession requires runtime.gemini_page or runtime.gemini_context")

        chat_url = self._chat_url()
        clean_thread = (not self._initialized) and self._should_clean_thread_on_session_init(chat_url)
        target_url = self._session_entry_url(clean_thread=clean_thread)
        try:
            current_url = str(getattr(page, "url", "") or "").strip().rstrip("/")
            if target_url.rstrip("/") == "https://gemini.google.com/app":
                needs_nav = current_url != target_url.rstrip("/")
            else:
                needs_nav = not current_url.startswith(target_url)
            if needs_nav:
                page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(2500)
        except Exception:
            pass
        return page

    def _should_clean_thread_on_session_init(self, chat_url: str) -> bool:
        if self._preserve_existing_thread(chat_url):
            return False
        if bool(_cfg_get(self.cfg, "gemini.chat_web_clean_thread_per_episode", False)):
            return True
        preserve_across_episodes = _cfg_get(
            self.cfg,
            "gemini.chat_web_preserve_existing_thread_across_episodes",
            None,
        )
        if preserve_across_episodes is False:
            return True
        if self._preserve_existing_thread(chat_url):
            return False
        if bool(_cfg_get(self.cfg, "gemini.chat_web_clean_thread_on_episode_start", True)):
            return True
        return bool(_cfg_get(self.cfg, "run.strict_single_chat_session", False))

    def restart_with_minimal_history(
        self,
        snapshot: SegmentSnapshot,
        accepted_outputs: Optional[List[Dict[str, Any]]] = None,
        heartbeat: Optional[Callable[[], None]] = None,
    ) -> None:
        if accepted_outputs is not None:
            self._accepted_outputs = [dict(item) for item in accepted_outputs]
        try:
            self.runtime.reopen_gemini()
        except Exception:
            try:
                if self.runtime.gemini_page is not None:
                    self.runtime.gemini_page.close()
            except Exception:
                pass
            try:
                if getattr(self.runtime, "gemini_context", None) is not None:
                    self.runtime.gemini_context.close()
            except Exception:
                pass
            self.runtime.gemini_page = None
            self.runtime.gemini_context = None
            for attr in ("gemini_page_borrowed", "gemini_context_borrowed"):
                if hasattr(self.runtime, attr):
                    try:
                        setattr(self.runtime, attr, False)
                    except Exception:
                        pass
            self.runtime.reopen_gemini()
        self._initialized = False
        self.session_id = uuid.uuid4().hex[:10]
        page = self._ensure_page()
        heartbeat_interval_sec = max(2.0, float(_cfg_get(self.cfg, "run.chat_subprocess_heartbeat_sec", 10) or 10))
        response_stall_sec = max(8.0, float(_cfg_get(self.cfg, "gemini.chat_web_response_stall_sec", 45.0) or 45.0))
        primer_parts = [
            f"Episode ID: {snapshot.episode_id or self.runtime.episode_id}",
            f"Segment checksum: {snapshot.checksum}",
            "Continue this episode in the same context. Do not create a new thread.",
        ]
        if self._accepted_outputs:
            primer_parts.append(
                "Accepted labels so far:\n"
                + json.dumps(self._accepted_outputs[:20], ensure_ascii=False, indent=2)
            )
        try:
            background_interval_sec = max(
                0.5,
                float(
                    _cfg_get(
                        self.cfg,
                        "run.gemini_background_heartbeat_interval_sec",
                        min(10.0, heartbeat_interval_sec),
                    )
                    or min(10.0, heartbeat_interval_sec)
                ),
            )
            with _HeartbeatGuard(heartbeat, interval_sec=background_interval_sec):
                _chat._handle_gemini_consent_if_present(page)
                chat_box = _chat._first_visible_locator(
                    page,
                    str(_cfg_get(self.cfg, "gemini.chat_web_input_selector", 'div[contenteditable="true"] || textarea')),
                    timeout_ms=30000,
                )
                if chat_box is not None:
                    baseline_state = _chat._capture_chat_response_state(page)
                    baseline_candidates = list(baseline_state.get("texts", []) or [])
                    baseline = str(baseline_state.get("latest_text", "") or "")
                    _chat._send_chat_prompt(
                        page=page,
                        chat_box=chat_box,
                        send_selector=str(
                            _cfg_get(
                                self.cfg,
                                "gemini.chat_web_send_selector",
                                'button[aria-label*="Send" i] || button:has-text("Send") || button:has-text("Run")',
                            )
                        ),
                        prompt_text="\n".join(primer_parts),
                    )
                    _chat._wait_for_new_chat_response_text(
                        page,
                        baseline_text=baseline,
                        baseline_candidates=baseline_candidates,
                        baseline_state=baseline_state,
                        timeout_sec=max(20.0, float(_cfg_get(self.cfg, "gemini.chat_web_timeout_sec", 180) or 180) * 0.35),
                        heartbeat=heartbeat,
                        heartbeat_interval_sec=heartbeat_interval_sec,
                        response_stall_sec=response_stall_sec,
                    )
        except Exception:
            pass

    def _should_clean_thread_per_request(self, chat_url: str = "") -> bool:
        if self._preserve_existing_thread(chat_url):
            return False
        gem_cfg = self.cfg.get("gemini", {}) if isinstance(self.cfg.get("gemini"), dict) else {}
        configured = gem_cfg.get("chat_web_clean_thread_per_request", None)
        if configured is None:
            if bool(_cfg_get(self.cfg, "gemini.chat_web_clean_thread_per_episode", False)):
                return False
            preserve_across_episodes = _cfg_get(
                self.cfg,
                "gemini.chat_web_preserve_existing_thread_across_episodes",
                None,
            )
            if preserve_across_episodes is False:
                return False
            if self._preserve_existing_thread(chat_url):
                return False
            return True
        return bool(configured)

    def _response_candidate_matches_request(
        self,
        raw_text: str,
        *,
        snapshot: SegmentSnapshot,
        expected_schema: str,
        requested_indices: Optional[Sequence[int]],
        allow_merge: bool,
    ) -> bool:
        schema_name = str(expected_schema or "").strip().lower()
        requested_scope = self._resolve_requested_indices(snapshot, requested_indices)
        if schema_name != "segments_only":
            return True
        effective_scope = requested_scope
        if not effective_scope:
            effective_scope = [
                int(seg.get("segment_index", 0) or 0)
                for seg in (snapshot.segments or [])
                if int(seg.get("segment_index", 0) or 0) > 0
            ]
        if not effective_scope:
            return True
        try:
            parsed_payload = json.loads(_chat._clean_json_text(raw_text))
        except Exception:
            return False
        schema_errors = validate_payload_schema(
            parsed_payload,
            expected_schema=schema_name,
            requested_indices=effective_scope,
            allow_merge=allow_merge,
        )
        if any(
            ("unknown segment" in str(item or "").lower())
            or ("outside requested scope" in str(item or "").lower())
            or ("duplicated segment" in str(item or "").lower())
            for item in schema_errors
        ):
            return False
        extracted_segments = _extract_payload_segments(parsed_payload)
        validated_segments, validation_errors = validate_normalized_segments(
            extracted_segments,
            snapshot.segments,
            requested_indices=effective_scope,
            allow_partial=True,
        )
        if not validated_segments:
            return False
        return not any(
            ("unknown segment" in str(item or "").lower())
            or ("outside requested scope" in str(item or "").lower())
            or ("duplicated segment" in str(item or "").lower())
            for item in validation_errors
        )

    def _normalize_request_mode(
        self,
        request_mode: str,
        *,
        expected_schema: str,
        requested_indices: Optional[Sequence[int]],
    ) -> str:
        clean_mode = str(request_mode or "").strip().lower()
        if clean_mode in {"labeling", "ops_planner", "repair"}:
            return resolve_mode_contract(clean_mode).mode
        schema_name = str(expected_schema or "").strip().lower()
        if schema_name == "operations_only":
            return resolve_mode_contract("ops_planner").mode
        if requested_indices:
            return resolve_mode_contract("repair").mode
        return resolve_mode_contract("labeling").mode

    def _resolve_request_stage_name(self, normalized_mode: str) -> str:
        clean_mode = str(normalized_mode or "").strip().lower()
        if clean_mode == resolve_mode_contract("repair").mode:
            return "repair"
        if clean_mode == resolve_mode_contract("ops_planner").mode:
            return "labeling"
        return "labeling"

    def _resolve_requested_model_name(self, normalized_mode: str) -> str:
        gem_cfg = self.cfg.get("gemini", {}) if isinstance(self.cfg.get("gemini"), dict) else {}
        stage_name = self._resolve_request_stage_name(normalized_mode)
        fallback = str(gem_cfg.get("model", "") or "").strip()
        resolved = str(resolve_stage_model(self.cfg, stage_name, fallback) or fallback).strip()
        return resolved or fallback

    def _build_same_thread_retry_prompt(
        self,
        prompt: str,
        *,
        expected_schema: str,
        requested_indices: Sequence[int],
    ) -> str:
        schema_name = str(expected_schema or "").strip().lower()
        lines = [
            "Retry the same Atlas request using the same video already present in this Gemini thread.",
            "Do not ask for a new upload and do not re-attach the video.",
        ]
        if schema_name == "segments_only":
            lines.append('Return exactly one JSON object with top-level key "segments".')
            if requested_indices:
                lines.append("Return every requested segment_index exactly once.")
        elif schema_name == "operations_only":
            lines.append('Return exactly one JSON object with top-level key "operations".')
        lines.append("Return only the final complete JSON response with no markdown and no prose.")
        lines.append("")
        lines.append(str(prompt or "").strip())
        return "\n".join(lines).strip()

    def _build_request_context(
        self,
        *,
        request_id: str,
        request_mode: str,
        snapshot: SegmentSnapshot,
        expected_schema: str,
        requested_indices: Sequence[int],
        baseline_state: Optional[Dict[str, Any]],
        episode_id: str,
        started_at_utc: str,
    ) -> GeminiRequestContext:
        baseline = baseline_state if isinstance(baseline_state, dict) else {}
        contract = resolve_mode_contract(request_mode)
        expected_schema_name = str(expected_schema or "").strip() or contract.expected_schema
        marker = (
            "ATLAS_REQUEST_CONTEXT "
            f"request_id={request_id} "
            f"mode={contract.mode} "
            f"episode_id={episode_id} "
            f"segments_checksum={snapshot.checksum}"
        )
        return GeminiRequestContext(
            request_id=str(request_id or "").strip(),
            mode=contract.mode,
            expected_schema=expected_schema_name,
            requested_indices=[int(idx) for idx in requested_indices if int(idx or 0) > 0],
            episode_id=str(episode_id or "").strip(),
            segments_checksum=str(snapshot.checksum or "").strip(),
            baseline_message_count=max(0, int(baseline.get("message_count", 0) or 0)),
            baseline_response_hash=str(baseline.get("response_hash", "") or "").strip(),
            baseline_preview=str(baseline.get("latest_text", "") or "").strip(),
            started_at_utc=str(started_at_utc or "").strip(),
            prompt_marker=marker,
        )

    def _decorate_prompt_with_request_context(
        self,
        prompt: str,
        *,
        request_context: GeminiRequestContext,
    ) -> str:
        lines = [
            "Internal request context for Atlas auditing only.",
            request_context.prompt_marker,
            "Do not repeat this metadata in your answer.",
            "",
            str(prompt or "").strip(),
        ]
        return "\n".join(lines).strip()

    def generate_labels(
        self,
        snapshot: SegmentSnapshot,
        prompt: str,
        video_file: Optional[Path],
        heartbeat: Optional[Callable[[], None]] = None,
    ) -> GeminiResult:
        result = self._request_payload(
            snapshot=snapshot,
            prompt=prompt,
            video_file=video_file,
            retry_stage=RetryStage.FULL_GENERATE,
            expected_schema="segments_only",
            request_mode="labeling",
            heartbeat=heartbeat,
        )
        result = self._maybe_retry_schema_followup(
            snapshot=snapshot,
            result=result,
            expected_schema="segments_only",
            requested_indices=None,
            allow_merge=False,
            retry_stage=RetryStage.FULL_GENERATE,
            heartbeat=heartbeat,
        )
        result = self._maybe_retry_segments_scope_followup(
            snapshot=snapshot,
            result=result,
            requested_indices=None,
            allow_partial=False,
            retry_stage=RetryStage.FULL_GENERATE,
            heartbeat=heartbeat,
        )
        if result.validated_segments:
            self._accepted_outputs = [dict(item) for item in result.validated_segments]
        self._remember_result(result)
        return result

    def repair_failed_segments(
        self,
        snapshot: SegmentSnapshot,
        failing_indices: Sequence[int],
        current_plan: Dict[int, Dict[str, Any]],
        reason: str,
        heartbeat: Optional[Callable[[], None]] = None,
    ) -> GeminiResult:
        scoped_indices = [int(idx) for idx in failing_indices if int(idx or 0) > 0]
        scoped_segments = [
            seg for seg in snapshot.segments if int(seg.get("segment_index", 0) or 0) in set(scoped_indices)
        ]
        prompt_lines = [
            "Fix ONLY the failing segments below.",
            f"Reason: {str(reason or '').strip() or 'policy failure'}",
            "Do not change segments outside the requested list.",
            "Return strict JSON with key \"segments\" only.",
            "",
            "Failing live DOM segments:",
            json.dumps(scoped_segments, ensure_ascii=False, indent=2),
            "",
            "Current plan subset:",
            json.dumps(
                [
                    current_plan[idx]
                    for idx in scoped_indices
                    if idx in current_plan and isinstance(current_plan[idx], dict)
                ],
                ensure_ascii=False,
                indent=2,
            ),
        ]
        retry_stage = RetryStage.TARGETED_REPAIR_1 if len(self._accepted_outputs) == 0 else RetryStage.TARGETED_REPAIR_2
        result = self._request_payload(
            snapshot=snapshot,
            prompt="\n".join(prompt_lines).strip(),
            video_file=None,
            retry_stage=retry_stage,
            expected_schema="segments_only",
            requested_indices=scoped_indices,
            request_mode="repair",
            heartbeat=heartbeat,
        )
        result = self._maybe_retry_schema_followup(
            snapshot=snapshot,
            result=result,
            expected_schema="segments_only",
            requested_indices=scoped_indices,
            allow_merge=False,
            retry_stage=retry_stage,
            heartbeat=heartbeat,
        )
        result = self._maybe_retry_segments_scope_followup(
            snapshot=snapshot,
            result=result,
            requested_indices=scoped_indices,
            allow_partial=True,
            retry_stage=retry_stage,
            heartbeat=heartbeat,
        )
        if result.validated_segments:
            self._accepted_outputs = [dict(item) for item in result.validated_segments]
        self._remember_result(result)
        return result

    def plan_structural_operations(
        self,
        snapshot: SegmentSnapshot,
        prompt: str,
        *,
        allow_merge: bool,
        video_file: Optional[Path],
        heartbeat: Optional[Callable[[], None]] = None,
    ) -> GeminiResult:
        result = self._request_payload(
            snapshot=snapshot,
            prompt=prompt,
            video_file=video_file,
            retry_stage=RetryStage.FULL_GENERATE,
            expected_schema="operations_only",
            requested_indices=[int(seg.get("segment_index", 0) or 0) for seg in snapshot.segments],
            allow_merge=allow_merge,
            request_mode="ops_planner",
            heartbeat=heartbeat,
        )
        result = self._maybe_retry_schema_followup(
            snapshot=snapshot,
            result=result,
            expected_schema="operations_only",
            requested_indices=[int(seg.get("segment_index", 0) or 0) for seg in snapshot.segments],
            allow_merge=allow_merge,
            retry_stage=RetryStage.FULL_GENERATE,
            heartbeat=heartbeat,
        )
        operations = _extract_payload_operations(result.parsed_payload, allow_merge=allow_merge)
        result.parsed_payload = {"operations": operations}
        result.validation_errors.extend(
            validate_payload_schema(
                {"operations": operations},
                expected_schema="operations_only",
                requested_indices=[int(seg.get("segment_index", 0) or 0) for seg in snapshot.segments],
                allow_merge=allow_merge,
            )
        )
        self._remember_result(result)
        return result

    def request_json(
        self,
        *,
        snapshot: SegmentSnapshot,
        prompt: str,
        video_file: Optional[Path],
        retry_stage: str = RetryStage.TRANSPORT,
        requested_indices: Optional[Sequence[int]] = None,
        allow_partial: bool = True,
        expected_schema: str = "",
        allow_merge: bool = False,
        request_mode: str = "",
        heartbeat: Optional[Callable[[], None]] = None,
    ) -> GeminiResult:
        result = self._request_payload(
            snapshot=snapshot,
            prompt=prompt,
            video_file=video_file,
            retry_stage=retry_stage,
            expected_schema=expected_schema,
            requested_indices=requested_indices,
            allow_merge=allow_merge,
            request_mode=request_mode,
            heartbeat=heartbeat,
        )
        result = self._maybe_retry_schema_followup(
            snapshot=snapshot,
            result=result,
            expected_schema=expected_schema,
            requested_indices=requested_indices,
            allow_merge=allow_merge,
            retry_stage=retry_stage,
            heartbeat=heartbeat,
        )
        if str(expected_schema or "").strip().lower() == "segments_only":
            result = self._maybe_retry_segments_scope_followup(
                snapshot=snapshot,
                result=result,
                requested_indices=requested_indices,
                allow_partial=allow_partial,
                retry_stage=retry_stage,
                heartbeat=heartbeat,
            )
            if result.validated_segments:
                self._accepted_outputs = [dict(item) for item in result.validated_segments]
        else:
            extracted_segments = _extract_payload_segments(result.parsed_payload)
            if extracted_segments:
                validated_segments, validation_errors = validate_normalized_segments(
                    extracted_segments,
                    snapshot.segments,
                    requested_indices=requested_indices,
                    allow_partial=allow_partial,
                )
                result.validated_segments = validated_segments
                result.validation_errors.extend(validation_errors)
                if validated_segments:
                    self._accepted_outputs = [dict(item) for item in validated_segments]
        self._remember_result(result)
        return result

    def _segment_scope_followup_attempts(self) -> int:
        return max(0, int(_cfg_get(self.cfg, "run.gemini_scope_followup_attempts", 1) or 1))

    def _schema_followup_attempts(self) -> int:
        return max(0, int(_cfg_get(self.cfg, "run.gemini_schema_followup_attempts", 1) or 1))

    def _resolve_requested_indices(
        self,
        snapshot: SegmentSnapshot,
        requested_indices: Optional[Sequence[int]] = None,
    ) -> List[int]:
        available = [
            int(seg.get("segment_index", 0) or 0)
            for seg in snapshot.segments
            if int(seg.get("segment_index", 0) or 0) > 0
        ]
        available_set = set(available)
        if not requested_indices:
            return list(available)
        out: List[int] = []
        seen: set[int] = set()
        for raw_idx in requested_indices:
            idx = int(raw_idx or 0)
            if idx <= 0 or idx not in available_set or idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
        return out

    def _validate_segments_result(
        self,
        *,
        result: GeminiResult,
        snapshot: SegmentSnapshot,
        requested_indices: Optional[Sequence[int]] = None,
        allow_partial: bool = False,
    ) -> GeminiResult:
        extracted_segments = _extract_payload_segments(result.parsed_payload)
        validated_segments, validation_errors = validate_normalized_segments(
            extracted_segments,
            snapshot.segments,
            requested_indices=requested_indices,
            allow_partial=allow_partial,
        )
        result.validated_segments = validated_segments
        result.validation_errors.extend(validation_errors)
        return result

    def _should_retry_scope_followup(self, errors: Sequence[str]) -> bool:
        for item in errors or []:
            message = str(item or "").strip().lower()
            if not message:
                continue
            if (
                "unknown segment" in message
                or "outside requested scope" in message
                or "duplicated segment" in message
                or "missing segment indices" in message
            ):
                return True
        return False

    def _should_retry_schema_followup(self, *, expected_schema: str, errors: Sequence[str]) -> bool:
        schema_name = str(expected_schema or "").strip().lower()
        if schema_name not in {"segments_only", "operations_only"}:
            return False
        for item in errors or []:
            message = str(item or "").strip().lower()
            if not message:
                continue
            if "top-level key" in message or "unexpectedly included top-level" in message:
                return True
            if schema_name == "segments_only" and '"segments"' in message:
                return True
            if schema_name == "operations_only" and '"operations"' in message:
                return True
        return False

    def _build_scope_followup_prompt(
        self,
        *,
        snapshot: SegmentSnapshot,
        requested_indices: Sequence[int],
        validation_errors: Sequence[str],
        allow_partial: bool,
    ) -> str:
        requested = [int(idx) for idx in requested_indices if int(idx or 0) > 0]
        requested_set = set(requested)
        scope_segments: List[Dict[str, Any]] = []
        for seg in snapshot.segments:
            idx = int(seg.get("segment_index", 0) or 0)
            if idx <= 0 or idx not in requested_set:
                continue
            scope_segments.append(
                {
                    "segment_index": idx,
                    "start_sec": round(_safe_float(seg.get("start_sec"), 0.0), 3),
                    "end_sec": round(_safe_float(seg.get("end_sec"), seg.get("start_sec", 0.0)), 3),
                    "current_label": str(seg.get("current_label", seg.get("label", "")) or "").strip(),
                    "raw_text": str(seg.get("raw_text", "") or "").strip()[:200],
                }
            )
        scope_rule = (
            "Return exactly one segment row for every allowed segment_index listed below."
            if not allow_partial
            else "Return rows only for the allowed segment_index values listed below. Partial output is allowed, but do not include any other segment_index."
        )
        error_lines = [
            f"- {str(item or '').strip()}"
            for item in validation_errors[:8]
            if str(item or "").strip()
        ]
        lines = [
            "Your last JSON response used invalid segment_index values for this request.",
            "Rewrite ONLY your last answer as strict JSON with top-level key \"segments\".",
            scope_rule,
            "Do not invent, renumber, split, merge, or add any segment_index outside the allowed list.",
            "Do not include markdown fences or prose.",
            f"Allowed segment_index values: {requested}.",
        ]
        if error_lines:
            lines.extend(["", "Fix these validation errors:", *error_lines])
        lines.extend(
            [
                "",
                "Authoritative DOM/source segment rows for this request:",
                json.dumps(scope_segments, ensure_ascii=False, indent=2),
            ]
        )
        return "\n".join(lines).strip()

    def _build_schema_followup_prompt(
        self,
        *,
        snapshot: SegmentSnapshot,
        expected_schema: str,
        requested_indices: Sequence[int],
        validation_errors: Sequence[str],
        allow_merge: bool,
    ) -> str:
        schema_name = str(expected_schema or "").strip().lower()
        requested = [int(idx) for idx in requested_indices if int(idx or 0) > 0]
        requested_set = set(requested)
        scope_segments: List[Dict[str, Any]] = []
        for seg in snapshot.segments:
            idx = int(seg.get("segment_index", 0) or 0)
            if idx <= 0:
                continue
            if requested and idx not in requested_set:
                continue
            scope_segments.append(
                {
                    "segment_index": idx,
                    "start_sec": round(_safe_float(seg.get("start_sec"), 0.0), 3),
                    "end_sec": round(_safe_float(seg.get("end_sec"), seg.get("start_sec", 0.0)), 3),
                    "current_label": str(seg.get("current_label", seg.get("label", "")) or "").strip(),
                    "raw_text": str(seg.get("raw_text", "") or "").strip()[:200],
                }
            )
        error_lines = [
            f"- {str(item or '').strip()}"
            for item in validation_errors[:8]
            if str(item or "").strip()
        ]
        if schema_name == "segments_only":
            lines = [
                "Your last JSON response used the wrong top-level schema for this labeling request.",
                'Rewrite ONLY your last answer as strict JSON with top-level key "segments".',
                'Do not include top-level key "operations".',
                "Each item must include segment_index, start_sec, end_sec, and label.",
                "Do not include markdown fences or prose.",
            ]
            if requested:
                lines.append(f"Allowed segment_index values: {requested}.")
                lines.append("Return exactly one row for each allowed segment_index.")
        elif schema_name == "operations_only":
            allowed_actions = ["split"]
            if allow_merge:
                allowed_actions.append("merge")
            lines = [
                "Your last JSON response used the wrong top-level schema for this structural-planner request.",
                'Rewrite ONLY your last answer as strict JSON with top-level key "operations".',
                'Do not include top-level key "segments".',
                f"Allowed actions: {allowed_actions}.",
                "Return [] when no structural operation is needed.",
                "Do not include markdown fences or prose.",
            ]
            if requested:
                lines.append(f"Only reference these segment_index values: {requested}.")
        else:
            return ""
        if error_lines:
            lines.extend(["", "Fix these validation errors:", *error_lines])
        if scope_segments:
            lines.extend(
                [
                    "",
                    "Authoritative DOM/source segment rows for this request:",
                    json.dumps(scope_segments, ensure_ascii=False, indent=2),
                ]
            )
        return "\n".join(lines).strip()

    def _maybe_retry_schema_followup(
        self,
        *,
        snapshot: SegmentSnapshot,
        result: GeminiResult,
        expected_schema: str,
        requested_indices: Optional[Sequence[int]],
        allow_merge: bool,
        retry_stage: str,
        heartbeat: Optional[Callable[[], None]] = None,
    ) -> GeminiResult:
        schema_name = str(expected_schema or "").strip().lower()
        max_followups = self._schema_followup_attempts()
        if max_followups <= 0 or not self._should_retry_schema_followup(expected_schema=schema_name, errors=result.validation_errors):
            return result

        requested_scope = self._resolve_requested_indices(snapshot, requested_indices)
        last_result = result
        for followup_attempt in range(1, max_followups + 1):
            followup_prompt = self._build_schema_followup_prompt(
                snapshot=snapshot,
                expected_schema=schema_name,
                requested_indices=requested_scope,
                validation_errors=last_result.validation_errors,
                allow_merge=allow_merge,
            )
            if not followup_prompt:
                break
            print(
                "[trace] gemini session schema followup: "
                f"episode_id={snapshot.episode_id or self.runtime.episode_id} "
                f"session_id={self.session_id} stage={retry_stage} "
                f"schema={schema_name} attempt={followup_attempt}/{max_followups}"
            , flush=True)
            followup_result = self._request_payload(
                snapshot=snapshot,
                prompt=followup_prompt,
                video_file=None,
                retry_stage=retry_stage,
                expected_schema=schema_name,
                requested_indices=requested_scope,
                allow_merge=allow_merge,
                heartbeat=heartbeat,
                preserve_current_thread=True,
            )
            followup_result.attach_notes = [
                *list(getattr(last_result, "attach_notes", []) or []),
                *list(getattr(followup_result, "attach_notes", []) or []),
            ]
            followup_result.attach_notes.append(f"schema_followup_retry_used:{followup_attempt}")
            if not self._should_retry_schema_followup(
                expected_schema=schema_name,
                errors=followup_result.validation_errors,
            ):
                followup_result.attach_notes.append("schema_followup_retry_recovered")
                return followup_result
            last_result = followup_result
        return last_result

    def _maybe_retry_segments_scope_followup(
        self,
        *,
        snapshot: SegmentSnapshot,
        result: GeminiResult,
        requested_indices: Optional[Sequence[int]],
        allow_partial: bool,
        retry_stage: str,
        heartbeat: Optional[Callable[[], None]] = None,
    ) -> GeminiResult:
        requested_scope = self._resolve_requested_indices(snapshot, requested_indices)
        current_result = self._validate_segments_result(
            result=result,
            snapshot=snapshot,
            requested_indices=requested_scope or requested_indices,
            allow_partial=allow_partial,
        )
        max_followups = self._segment_scope_followup_attempts()
        if max_followups <= 0 or not self._should_retry_scope_followup(current_result.validation_errors):
            return current_result

        last_result = current_result
        for followup_attempt in range(1, max_followups + 1):
            followup_prompt = self._build_scope_followup_prompt(
                snapshot=snapshot,
                requested_indices=requested_scope,
                validation_errors=last_result.validation_errors,
                allow_partial=allow_partial,
            )
            print(
                "[trace] gemini session scope followup: "
                f"episode_id={snapshot.episode_id or self.runtime.episode_id} "
                f"session_id={self.session_id} stage={retry_stage} "
                f"attempt={followup_attempt}/{max_followups} "
                f"requested_indices={requested_scope[:12]}"
            , flush=True)
            followup_result = self._request_payload(
                snapshot=snapshot,
                prompt=followup_prompt,
                video_file=None,
                retry_stage=retry_stage,
                expected_schema="segments_only",
                requested_indices=requested_scope,
                heartbeat=heartbeat,
                preserve_current_thread=True,
            )
            followup_result.attach_notes = [
                *list(getattr(last_result, "attach_notes", []) or []),
                *list(getattr(followup_result, "attach_notes", []) or []),
            ]
            followup_result.attach_notes.append(f"scope_followup_retry_used:{followup_attempt}")
            followup_result = self._validate_segments_result(
                result=followup_result,
                snapshot=snapshot,
                requested_indices=requested_scope or requested_indices,
                allow_partial=allow_partial,
            )
            if followup_result.validated_segments and not followup_result.validation_errors:
                followup_result.attach_notes.append("scope_followup_retry_recovered")
                return followup_result
            last_result = followup_result
            if not self._should_retry_scope_followup(last_result.validation_errors):
                break
        return last_result

    def _persist_raw_response(
        self,
        *,
        request_id: str,
        raw_text: str,
        retry_stage: str,
        started_at_utc: str,
        expected_schema: str,
        requested_indices: Sequence[int],
        request_context: Optional[GeminiRequestContext],
    ) -> Tuple[str, str]:
        out_dir = Path(str(_cfg_get(self.cfg, "run.output_dir", "outputs") or "outputs")).resolve()
        cache_root = out_dir / "_chat_only" / str(self.runtime.episode_id or "episode").strip()
        cache_root.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        raw_path = cache_root / f"raw_gemini_response_{retry_stage}_{request_id}_{stamp}.txt"
        raw_path.write_text(str(raw_text or ""), encoding="utf-8")
        meta_path = cache_root / f"raw_gemini_response_{retry_stage}_{request_id}_{stamp}.json"
        meta_payload = {
            "request_id": str(request_id or "").strip(),
            "episode_id": str(self.runtime.episode_id or "").strip(),
            "context_id": str(self.runtime.context_id or "").strip(),
            "retry_stage": str(retry_stage or "").strip(),
            "expected_schema": str(expected_schema or "").strip(),
            "requested_indices": [int(idx) for idx in requested_indices if int(idx or 0) > 0],
            "started_at_utc": str(started_at_utc or "").strip(),
            "raw_text_path": str(raw_path),
            "raw_text": str(raw_text or ""),
            "request_context": request_context.to_dict() if isinstance(request_context, GeminiRequestContext) else {},
        }
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(raw_path), str(meta_path)

    def _update_raw_response_meta(
        self,
        meta_path: str,
        *,
        acceptance_metadata: Optional[Dict[str, Any]] = None,
        validation_errors: Optional[Sequence[str]] = None,
    ) -> None:
        target = Path(str(meta_path or "").strip())
        if not target.exists():
            return
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        if isinstance(acceptance_metadata, dict):
            payload["acceptance_metadata"] = dict(acceptance_metadata)
        if validation_errors is not None:
            payload["validation_errors"] = [
                str(item).strip()
                for item in validation_errors
                if str(item or "").strip()
            ]
        try:
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _build_acceptance_metadata(
        self,
        *,
        page: Any,
        request_context: GeminiRequestContext,
        raw_text: str,
        parsed_payload: Dict[str, Any],
        schema_errors: Sequence[str],
        snapshot: SegmentSnapshot,
        requested_indices: Sequence[int],
        allow_merge: bool,
    ) -> Dict[str, Any]:
        current_state = _chat._capture_chat_response_state(page)
        current_message_count = max(0, int(current_state.get("message_count", 0) or 0))
        current_response_hash = str(current_state.get("response_hash", "") or "").strip()
        baseline_hash = str(request_context.baseline_response_hash or "").strip()
        baseline_count = max(0, int(request_context.baseline_message_count or 0))
        response_hash_changed = bool(
            current_response_hash and baseline_hash and current_response_hash != baseline_hash
        )
        baseline_advanced = bool(current_message_count > baseline_count or response_hash_changed)
        scope_errors: List[str] = []
        payload_segments = _extract_payload_segments(parsed_payload)
        if payload_segments:
            _, scope_errors = validate_normalized_segments(
                payload_segments,
                snapshot.segments,
                requested_indices=requested_indices,
                allow_partial=True,
            )
        return {
            "accepted": bool(baseline_advanced and not list(schema_errors)),
            "mode": str(request_context.mode or "").strip(),
            "expected_schema": str(request_context.expected_schema or "").strip(),
            "requested_indices": [int(idx) for idx in requested_indices if int(idx or 0) > 0],
            "prompt_marker": str(request_context.prompt_marker or "").strip(),
            "baseline_message_count": baseline_count,
            "current_message_count": current_message_count,
            "baseline_response_hash": baseline_hash,
            "current_response_hash": current_response_hash,
            "response_hash_changed": response_hash_changed,
            "baseline_advanced": baseline_advanced,
            "parseable_json": bool(str(raw_text or "").strip()),
            "schema_errors": [str(item).strip() for item in schema_errors if str(item or "").strip()],
            "scope_errors": [str(item).strip() for item in scope_errors if str(item or "").strip()],
            "segments_checksum": str(request_context.segments_checksum or "").strip(),
            "allow_merge": bool(allow_merge),
        }

    def _attachment_confirmed(self, attach_notes: Sequence[str]) -> bool:
        for note in attach_notes or []:
            lowered = str(note or "").strip().lower()
            if ": attached (" in lowered:
                return True
        return False

    def _assert_request_integrity(
        self,
        *,
        video_file: Optional[Path],
        attach_notes: Sequence[str],
        acceptance_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata = acceptance_metadata if isinstance(acceptance_metadata, dict) else {}
        requires_video = bool(video_file is not None and Path(str(video_file)).exists())
        if requires_video and not self._attachment_confirmed(attach_notes):
            raise RuntimeError(
                "GEMINI_ATTACHMENT_REQUIRED: video upload was not confirmed for current request"
            )
        if not bool(metadata.get("baseline_advanced", False)):
            raise RuntimeError(
                "GEMINI_NO_NEW_ASSISTANT_MESSAGE: assistant response did not advance baseline"
            )

    def _remember_result(self, result: GeminiResult) -> None:
        retry_reason = _infer_retry_reason_from_errors(result.validation_errors, raw_text=result.raw_text)
        state = self.runtime.task_state
        state["gemini_session_id"] = self.session_id
        state["gemini_last_request_id"] = result.request_id
        state["gemini_last_retry_stage"] = result.retry_stage
        state["gemini_last_retry_reason"] = retry_reason
        state["gemini_last_latency_ms"] = int(result.latency_ms or 0)
        state["gemini_last_response_path"] = str(result.raw_response_path or "")
        state["gemini_last_response_meta_path"] = str(result.raw_response_meta_path or "")
        state["gemini_last_validation_errors"] = [str(item) for item in result.validation_errors[:10]]
        state["gemini_last_expected_schema"] = str(result.expected_schema or "")
        state["gemini_last_requested_indices"] = [int(item) for item in result.requested_indices[:20]]
        state["gemini_last_started_at_utc"] = str(result.started_at_utc or "")
        state["gemini_last_acceptance_metadata"] = dict(getattr(result, "acceptance_metadata", {}) or {})
        request_context = getattr(result, "request_context", None)
        if isinstance(request_context, GeminiRequestContext):
            state["gemini_last_request_context"] = request_context.to_dict()
            state["gemini_last_request_mode"] = str(request_context.mode or "").strip()
            state["gemini_last_segments_checksum"] = str(request_context.segments_checksum or "").strip()
            state["gemini_last_baseline_message_count"] = int(request_context.baseline_message_count or 0)
            state["gemini_last_baseline_response_hash"] = str(request_context.baseline_response_hash or "").strip()
        if result.session_restarted:
            state["gemini_session_restarted"] = True
        self._update_raw_response_meta(
            str(result.raw_response_meta_path or "").strip(),
            acceptance_metadata=getattr(result, "acceptance_metadata", {}),
            validation_errors=result.validation_errors,
        )
        self._journal_event(
            "gemini_result_recorded",
            stage="waiting_for_gemini",
            reason=retry_reason,
            request_context=request_context if isinstance(request_context, GeminiRequestContext) else None,
            payload={
                "acceptance_metadata": dict(getattr(result, "acceptance_metadata", {}) or {}),
                "validation_errors": [str(item) for item in result.validation_errors[:8]],
                "raw_response_meta_path": str(result.raw_response_meta_path or "").strip(),
            },
        )

    def _request_payload(
        self,
        *,
        snapshot: SegmentSnapshot,
        prompt: str,
        video_file: Optional[Path],
        retry_stage: str,
        expected_schema: str = "",
        requested_indices: Optional[Sequence[int]] = None,
        allow_merge: bool = False,
        request_mode: str = "",
        heartbeat: Optional[Callable[[], None]] = None,
        preserve_current_thread: bool = False,
    ) -> GeminiResult:
        max_retries = max(1, int(_cfg_get(self.cfg, "run.gemini_transport_max_retries", 3) or 3))
        last_exc: Optional[Exception] = None
        session_restarted = False
        requested_scope = [int(idx) for idx in (requested_indices or []) if int(idx or 0) > 0]
        base_prompt = str(prompt or "").strip()
        reuse_thread_video_only = False

        for attempt in range(1, max_retries + 1):
            request_id = uuid.uuid4().hex[:8]
            started = time.monotonic()
            episode_id = snapshot.episode_id or self.runtime.episode_id
            started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            resolved_mode = self._normalize_request_mode(
                request_mode,
                expected_schema=expected_schema,
                requested_indices=requested_scope,
            )
            prompt_for_attempt = base_prompt
            video_file_for_attempt = video_file
            if reuse_thread_video_only:
                prompt_for_attempt = self._build_same_thread_retry_prompt(
                    base_prompt,
                    expected_schema=expected_schema,
                    requested_indices=requested_scope,
                )
                video_file_for_attempt = None
            progress_capture_interval_sec = max(
                15.0,
                float(_cfg_get(self.cfg, "run.gemini_progress_capture_interval_sec", 30.0) or 30.0),
            )
            last_progress_capture_ts = 0.0

            def _capture_request_step(step_name: str) -> None:
                if not bool(_cfg_get(self.cfg, "run.capture_step_screenshots", False)):
                    return
                if page_ref[0] is None:
                    return
                try:
                    _capture_step_artifacts(
                        page_ref[0],
                        self.cfg,
                        str(episode_id or "").strip(),
                        f"gemini_{request_id}_{step_name}",
                    )
                except Exception:
                    pass

            def _progress_hook(payload: Dict[str, Any]) -> None:
                nonlocal last_progress_capture_ts
                phase = str(payload.get("phase", "progress") or "progress").strip()
                details: List[str] = []
                for key in (
                    "attachment_name",
                    "elapsed_sec",
                    "remaining_sec",
                    "response_chars",
                    "stable_count",
                    "seconds_since_change",
                    "token_count",
                    "page_token_count",
                ):
                    if key in payload and payload.get(key) not in ("", None):
                        details.append(f"{key}={payload.get(key)}")
                for key in (
                    "thinking",
                    "parseable_json",
                    "stop_visible",
                    "send_visible",
                    "evidence_match",
                    "drive_match",
                    "attachment_hint",
                    "attached",
                ):
                    if key in payload:
                        details.append(f"{key}={bool(payload.get(key))}")
                if payload.get("mode"):
                    details.append(f"mode={payload.get('mode')}")
                preview = str(payload.get("preview", "") or "").strip()
                if preview:
                    details.append(f'preview="{preview}"')
                detail = str(payload.get("detail", "") or "").strip()
                if detail:
                    details.append(f'detail="{detail}"')
                print(
                    "[trace] gemini session progress: "
                    f"episode_id={episode_id} session_id={self.session_id} "
                    f"request_id={request_id} stage={retry_stage} phase={phase} "
                    + " ".join(details)
                , flush=True)
                now = time.monotonic()
                if (now - last_progress_capture_ts) < progress_capture_interval_sec:
                    return
                if phase not in {"attach_wait", "response_wait", "attach_done"}:
                    return
                last_progress_capture_ts = now
                elapsed_marker = str(payload.get("elapsed_sec", "0")).replace(".", "p")
                _capture_request_step(f"{phase}_{elapsed_marker}s")

            page_ref: List[Any] = [None]
            print(
                "[trace] gemini session request start: "
                f"episode_id={episode_id} "
                f"session_id={self.session_id} request_id={request_id} "
                f"stage={retry_stage} attempt={attempt}/{max_retries} "
                f"schema={str(expected_schema or 'unspecified').strip() or 'unspecified'} "
                f"segments={int(snapshot.segment_count or len(snapshot.segments or []))} "
                f"video={str(getattr(video_file_for_attempt, 'name', '') or 'none')}"
            , flush=True)
            self._journal_event(
                "gemini_request_started",
                stage="waiting_for_gemini",
                reason=str(retry_stage or "").strip(),
                payload={
                    "retry_stage": str(retry_stage or "").strip(),
                    "expected_schema": str(expected_schema or "").strip(),
                    "segment_count": int(snapshot.segment_count or len(snapshot.segments or [])),
                    "requested_indices": list(requested_scope),
                    "video_file": str(getattr(video_file_for_attempt, "name", "") or "").strip(),
                    "reuse_thread_video_only": bool(reuse_thread_video_only),
                },
            )
            try:
                page = self._ensure_page()
                page_ref[0] = page
                try:
                    page.bring_to_front()
                except Exception:
                    pass
                gem_cfg = self.cfg.get("gemini", {}) if isinstance(self.cfg.get("gemini"), dict) else {}
                chat_url = self._chat_url()
                requested_model_name = self._resolve_requested_model_name(resolved_mode)
                desired_model_mode = _chat._resolve_chat_web_ui_model_mode(
                    gem_cfg,
                    requested_model=requested_model_name,
                )
                allowed_model_modes = _chat._resolve_chat_web_allowed_model_modes(
                    gem_cfg,
                    requested_model=requested_model_name,
                )
                input_sel = str(gem_cfg.get("chat_web_input_selector", 'div[contenteditable="true"] || textarea') or "").strip()
                send_sel = str(
                    gem_cfg.get(
                        "chat_web_send_selector",
                        'button[aria-label*="Send" i] || button:has-text("Send") || button:has-text("Run")',
                    )
                    or ""
                ).strip()
                timeout_sec = max(20.0, float(gem_cfg.get("chat_web_timeout_sec", 180) or 180))
                response_stall_sec = _chat._resolve_chat_web_response_stall_sec(
                    gem_cfg,
                    requested_model=requested_model_name,
                    fallback=45.0,
                )
                heartbeat_interval_sec = max(
                    2.0,
                    float(_cfg_get(self.cfg, "run.chat_subprocess_heartbeat_sec", 10) or 10),
                )
                background_interval_sec = max(
                    0.5,
                    float(
                        _cfg_get(
                            self.cfg,
                            "run.gemini_background_heartbeat_interval_sec",
                            min(10.0, heartbeat_interval_sec),
                        )
                        or min(10.0, heartbeat_interval_sec)
                    ),
                )
                with _HeartbeatGuard(heartbeat, interval_sec=background_interval_sec):
                    if not str(getattr(page, "url", "") or "").startswith(chat_url):
                        page.goto(chat_url, wait_until="domcontentloaded", timeout=60000)
                        page.wait_for_timeout(2500)
                    _chat._handle_gemini_consent_if_present(page)
                    if heartbeat is not None:
                        try:
                            heartbeat()
                        except Exception:
                            pass
                    chat_box = _chat._first_visible_locator(page, input_sel, timeout_ms=30000)
                    if chat_box is None:
                        current_url = str(getattr(page, "url", "") or "").strip()
                        raise RuntimeError(
                            "Gemini chat input not visible on session page."
                            + (f" url={current_url}" if current_url else "")
                        )
                    model_mode_result = _chat._ensure_gemini_chat_model_mode(
                        page,
                        desired_model_mode,
                        allowed_modes=allowed_model_modes,
                        settle_sec=max(1.0, float(gem_cfg.get("chat_web_model_mode_settle_sec", 1.0) or 1.0)),
                    )
                    current_model_mode = str(model_mode_result.get("current_mode", "") or "").strip()
                    response_stall_sec = _chat._resolve_chat_web_response_stall_sec(
                        gem_cfg,
                        requested_model=requested_model_name,
                        current_mode=current_model_mode,
                        fallback=response_stall_sec,
                    )
                    print(
                        "[trace] gemini session model mode: "
                        f"episode_id={episode_id} session_id={self.session_id} "
                        f"request_id={request_id} desired={desired_model_mode} "
                        f"allowed={','.join(allowed_model_modes)} "
                        f"current={str(model_mode_result.get('current_mode', '') or '').strip()} "
                        f"verified={bool(model_mode_result.get('verified', False))} "
                        f"reason={str(model_mode_result.get('reason', '') or '').strip()} "
                        f"response_stall_sec={response_stall_sec:.1f}"
                    , flush=True)
                    self._journal_event(
                        "gemini_model_mode_ready",
                        stage="waiting_for_gemini",
                        reason=str(model_mode_result.get("reason", "") or "").strip(),
                        payload={
                            "desired_mode": str(desired_model_mode or "").strip(),
                            "allowed_modes": list(allowed_model_modes),
                            "current_mode": current_model_mode,
                            "verified": bool(model_mode_result.get("verified", False)),
                            "requested_model": str(requested_model_name or "").strip(),
                            "response_stall_sec": float(response_stall_sec),
                        },
                    )
                    if bool(gem_cfg.get("chat_web_enforce_model_mode", False)) and not bool(
                        model_mode_result.get("verified", False)
                    ):
                        raise RuntimeError(
                            "Gemini chat model mode verification failed: "
                            f"expected one of {allowed_model_modes}, got {current_model_mode or 'unknown'}"
                        )
                    print(
                        "[trace] gemini session page ready: "
                        f"episode_id={episode_id} session_id={self.session_id} "
                        f"request_id={request_id} url={str(getattr(page, 'url', '') or '').strip()}"
                    , flush=True)
                    _capture_request_step("page_ready")

                    clean_thread = False if preserve_current_thread else self._should_clean_thread_per_request(chat_url)
                    if not clean_thread and not self._initialized:
                        clean_thread = self._should_clean_thread_on_session_init(chat_url)
                    memory_primer_text = _chat._load_chat_memory_primer(gem_cfg)
                    seed_context_text = _chat._load_chat_seed_context(gem_cfg)
                    send_seed_context = bool(gem_cfg.get("chat_web_seed_context_send_before_prompt", False))
                    pre_send_ready_timeout_sec = max(
                        8.0,
                        float(gem_cfg.get("chat_web_pre_send_ready_timeout_sec", 18.0) or 18.0),
                    )
                    pre_send_settle_sec = max(
                        1.0,
                        float(gem_cfg.get("chat_web_pre_send_settle_sec", 3.0) or 3.0),
                    )
                    if clean_thread or not self._initialized:
                        print(
                            "[trace] gemini session thread prepare: "
                            f"episode_id={episode_id} session_id={self.session_id} "
                            f"request_id={request_id} clean_thread={clean_thread} "
                            f"seed_context={bool(str(seed_context_text or '').strip()) and send_seed_context}"
                        , flush=True)
                        if clean_thread:
                            chat_box, _ = _chat._prepare_clean_chat_thread(
                                page=page,
                                input_selector=input_sel,
                                send_selector=send_sel,
                                timeout_sec=timeout_sec,
                                memory_primer_text=memory_primer_text,
                                base_url=chat_url,
                                heartbeat=heartbeat,
                                heartbeat_interval_sec=heartbeat_interval_sec,
                                response_stall_sec=response_stall_sec,
                            )
                        if send_seed_context and str(seed_context_text or "").strip():
                            chat_box, _ = _chat._seed_chat_thread(
                                page=page,
                                chat_box=chat_box,
                                input_selector=input_sel,
                                send_selector=send_sel,
                                timeout_sec=timeout_sec,
                                seed_context_text=seed_context_text,
                                heartbeat=heartbeat,
                                heartbeat_interval_sec=heartbeat_interval_sec,
                                response_stall_sec=response_stall_sec,
                            )
                        _capture_request_step("thread_ready")
                    self._initialized = True

                    attach_notes: List[str] = []
                    if video_file_for_attempt is not None and video_file_for_attempt.exists():
                        video_size_mb = float(video_file_for_attempt.stat().st_size) / (1024 * 1024)
                        print(
                            "[trace] gemini session attach begin: "
                            f"episode_id={episode_id} session_id={self.session_id} "
                            f"request_id={request_id} file={video_file_for_attempt.name} size_mb={video_size_mb:.1f}"
                        , flush=True)
                        _capture_request_step("attach_begin")
                        attach_notes.extend(
                            _chat._attach_files_via_chat_ui(
                                page=page,
                                composer_locator=chat_box,
                                attach_candidates=[video_file_for_attempt],
                                episode_id=episode_id,
                                prefer_drive_picker=bool(gem_cfg.get("chat_web_prefer_drive_picker", False)),
                                drive_root_folder_url=str(gem_cfg.get("chat_web_drive_root_folder_url", "") or "").strip(),
                                max_upload_mb=max(50.0, float(gem_cfg.get("chat_web_max_upload_mb", 2048) or 2048)),
                                attach_button_sel=str(
                                    gem_cfg.get(
                                        "chat_web_attach_button_selector",
                                        'button[aria-label*="Open upload file menu" i] || button[aria-label*="Add files" i] || button[aria-label*="Upload" i] || button[aria-label*="Tools" i] || button:has-text("Add files") || button:has-text("Upload") || button:has-text("Tools")',
                                    )
                                ),
                                upload_menu_sel=str(
                                    gem_cfg.get(
                                        "chat_web_upload_menu_selector",
                                        'button[aria-label*="Upload files" i] || [role="menuitem"]:has-text("Upload files") || button:has-text("Upload files") || [role="option"]:has-text("Upload files") || text=/^Upload files$/i',
                                    )
                                ),
                                file_input_sel=str(gem_cfg.get("chat_web_file_input_selector", 'input[type="file"]')),
                                upload_settle_min_sec=max(1.0, float(gem_cfg.get("chat_web_upload_settle_min_sec", 4.0) or 4.0)),
                                upload_settle_sec_per_100mb=max(0.0, float(gem_cfg.get("chat_web_upload_settle_sec_per_100mb", 12.0) or 12.0)),
                                upload_settle_max_sec=max(4.0, float(gem_cfg.get("chat_web_upload_settle_max_sec", 45.0) or 45.0)),
                                heartbeat=heartbeat,
                                heartbeat_interval_sec=heartbeat_interval_sec,
                                progress_hook=_progress_hook,
                            )
                        )
                        print(
                            "[trace] gemini session attach complete: "
                            f"episode_id={episode_id} session_id={self.session_id} "
                            f"request_id={request_id} attach_notes={len(attach_notes)}"
                        , flush=True)
                        _capture_request_step("attach_complete")
                    else:
                        attach_notes = []
                    baseline_state = _chat._capture_chat_response_state(page)
                    baseline_candidates = list(baseline_state.get("texts", []) or [])
                    baseline = str(baseline_state.get("latest_text", "") or "")
                    request_context = self._build_request_context(
                        request_id=request_id,
                        request_mode=resolved_mode,
                        snapshot=snapshot,
                        expected_schema=expected_schema,
                        requested_indices=requested_scope,
                        baseline_state=baseline_state,
                        episode_id=str(episode_id or ""),
                        started_at_utc=started_at_utc,
                    )
                    self._journal_event(
                        "gemini_request_context_ready",
                        stage="waiting_for_gemini",
                        reason=str(retry_stage or "").strip(),
                        request_context=request_context,
                        payload={
                            "expected_schema": str(expected_schema or "").strip(),
                            "requested_indices": list(requested_scope),
                        },
                    )
                    decorated_prompt = self._decorate_prompt_with_request_context(
                        prompt_for_attempt,
                        request_context=request_context,
                    )
                    print(
                        "[trace] gemini session prompt send start: "
                        f"episode_id={episode_id} session_id={self.session_id} "
                        f"request_id={request_id} prompt_chars={len(str(prompt_for_attempt or ''))} "
                        f"baseline_chars={len(str(baseline or ''))}"
                    , flush=True)
                    _chat._send_chat_prompt(
                        page=page,
                        chat_box=chat_box,
                        send_selector=send_sel,
                        prompt_text=decorated_prompt,
                        pre_send_ready_timeout_sec=pre_send_ready_timeout_sec,
                        pre_send_settle_sec=pre_send_settle_sec,
                    )
                    _capture_request_step("prompt_sent")
                    print(
                        "[trace] gemini session prompt sent: "
                        f"episode_id={episode_id} session_id={self.session_id} "
                        f"request_id={request_id}"
                    , flush=True)
                    print(
                        "[trace] gemini session response wait start: "
                        f"episode_id={episode_id} session_id={self.session_id} "
                        f"request_id={request_id} timeout_sec={timeout_sec:.1f} "
                        f"stall_sec={response_stall_sec:.1f}"
                    , flush=True)
                    preferred_key = ""
                    schema_name = str(expected_schema or "").strip().lower()
                    if schema_name == "segments_only":
                        preferred_key = "segments"
                    elif schema_name == "operations_only":
                        preferred_key = "operations"
                    raw_text = _chat._wait_for_new_chat_response_text(
                        page,
                        baseline_text=baseline,
                        baseline_candidates=baseline_candidates,
                        baseline_state=baseline_state,
                        timeout_sec=timeout_sec,
                        heartbeat=heartbeat,
                        heartbeat_interval_sec=heartbeat_interval_sec,
                        response_stall_sec=response_stall_sec,
                        progress_hook=_progress_hook,
                        require_parseable_json=True,
                        preferred_top_level_key=preferred_key,
                        response_candidate_validator=(
                            lambda candidate_text: self._response_candidate_matches_request(
                                candidate_text,
                                snapshot=snapshot,
                                expected_schema=expected_schema,
                                requested_indices=requested_scope,
                                allow_merge=allow_merge,
                            )
                        ),
                    )
                if not raw_text:
                    raise RuntimeError("Timed out waiting for Gemini chat response.")
                print(
                    "[trace] gemini session response received: "
                    f"episode_id={episode_id} session_id={self.session_id} "
                    f"request_id={request_id} chars={len(str(raw_text or ''))}"
                , flush=True)
                _capture_request_step("response_ready")
                raw_response_path, raw_response_meta_path = self._persist_raw_response(
                    request_id=request_id,
                    raw_text=raw_text,
                    retry_stage=retry_stage,
                    started_at_utc=started_at_utc,
                    expected_schema=expected_schema,
                    requested_indices=requested_scope,
                    request_context=request_context,
                )
                if _chat._is_retryable_chat_error_text(raw_text):
                    preview = " ".join(str(raw_text or "").strip().split())
                    if len(preview) > 160:
                        preview = preview[:157] + "..."
                    raise RuntimeError(f"Gemini chat returned transient error response: {preview}")
                try:
                    parsed_payload = json.loads(_chat._clean_json_text(raw_text))
                except Exception as exc:
                    raise RuntimeError(f"Gemini session returned invalid JSON: {exc}") from exc
                schema_errors = validate_payload_schema(
                    parsed_payload,
                    expected_schema=expected_schema,
                    requested_indices=requested_scope,
                    allow_merge=allow_merge,
                )
                acceptance_metadata = self._build_acceptance_metadata(
                    page=page_ref[0],
                    request_context=request_context,
                    raw_text=raw_text,
                    parsed_payload=parsed_payload,
                    schema_errors=schema_errors,
                    snapshot=snapshot,
                    requested_indices=requested_scope,
                    allow_merge=allow_merge,
                )
                self._assert_request_integrity(
                    video_file=video_file_for_attempt,
                    attach_notes=attach_notes,
                    acceptance_metadata=acceptance_metadata,
                )
                latency_ms = int(round((time.monotonic() - started) * 1000))
                result = GeminiResult(
                    request_id=request_id,
                    episode_id=episode_id,
                    context_id=self.runtime.context_id,
                    retry_stage=retry_stage,
                    latency_ms=latency_ms,
                    raw_text=raw_text,
                    parsed_payload=parsed_payload,
                    attach_notes=attach_notes,
                    validation_errors=schema_errors,
                    session_restarted=session_restarted,
                    raw_response_path=raw_response_path,
                    raw_response_meta_path=raw_response_meta_path,
                    expected_schema=str(expected_schema or "").strip(),
                    requested_indices=list(requested_scope),
                    started_at_utc=started_at_utc,
                    request_context=request_context,
                    acceptance_metadata=acceptance_metadata,
                )
                self._last_expected_schema = str(expected_schema or "").strip().lower()
                self.runtime.task_state["gemini_session_id"] = self.session_id
                self._update_raw_response_meta(
                    raw_response_meta_path,
                    acceptance_metadata=acceptance_metadata,
                    validation_errors=schema_errors,
                )
                self._journal_event(
                    "gemini_request_completed",
                    stage="waiting_for_gemini",
                    reason="accepted_response",
                    request_context=request_context,
                    payload={
                        "retry_stage": str(retry_stage or "").strip(),
                        "latency_ms": latency_ms,
                        "expected_schema": str(expected_schema or "").strip(),
                        "requested_indices": list(requested_scope),
                        "acceptance_metadata": dict(acceptance_metadata),
                        "raw_response_meta_path": str(raw_response_meta_path or "").strip(),
                    },
                )
                print(
                    "[trace] gemini session request completed: "
                    f"episode_id={episode_id} "
                    f"session_id={self.session_id} request_id={request_id} "
                    f"stage={retry_stage} latency_ms={latency_ms} "
                    f"attach_notes={len(attach_notes)}"
                , flush=True)
                return result
            except Exception as exc:
                last_exc = exc
                reason = classify_transport_failure(str(exc))
                same_thread_retry = "GEMINI_STALE_RESPONSE" in str(exc)
                if same_thread_retry:
                    reuse_thread_video_only = True
                _capture_request_step("request_failed")
                print(
                    "[trace] gemini session request failed: "
                    f"episode_id={episode_id} "
                    f"session_id={self.session_id} request_id={request_id} "
                    f"stage={retry_stage} attempt={attempt}/{max_retries} "
                    f"retry_reason={reason} detail={str(exc)}"
                , flush=True)
                if reason in {
                    RetryReason.PAGE_CRASH,
                    RetryReason.NO_NEW_ASSISTANT_MESSAGE,
                    RetryReason.STALE_RESPONSE,
                    RetryReason.THREAD_CONTAMINATION,
                } and self.runtime.gemini_browser is not None and not same_thread_retry:
                    print(
                        "[trace] gemini session restart triggered: "
                        f"episode_id={episode_id} "
                        f"request_id={request_id} stage={retry_stage} reason={reason}"
                    , flush=True)
                    self._journal_event(
                        "gemini_session_restart",
                        stage="waiting_for_gemini",
                        reason=reason,
                        payload={
                            "retry_stage": str(retry_stage or "").strip(),
                            "attempt": int(attempt),
                            "request_id": str(request_id or "").strip(),
                        },
                    )
                    self.restart_with_minimal_history(snapshot, self._accepted_outputs, heartbeat=heartbeat)
                    session_restarted = True
                self._journal_event(
                    "gemini_request_failed",
                    stage="waiting_for_gemini",
                    reason=reason,
                    payload={
                        "retry_stage": str(retry_stage or "").strip(),
                        "attempt": int(attempt),
                        "request_id": str(request_id or "").strip(),
                        "error": str(exc),
                    },
                )
                if attempt >= max_retries:
                    break
                backoff_sec = transport_backoff_seconds(attempt)
                print(
                    "[trace] gemini session request retrying: "
                    f"episode_id={snapshot.episode_id or self.runtime.episode_id} "
                    f"request_id={request_id} stage={retry_stage} "
                    f"attempt={attempt + 1}/{max_retries} backoff={backoff_sec:.1f}s"
                , flush=True)
                time.sleep(backoff_sec)

        final_latency_ms = 0
        if last_exc is None:
            last_exc = RuntimeError("Gemini session failed without exception details.")
        return GeminiResult(
            request_id=uuid.uuid4().hex[:8],
            episode_id=snapshot.episode_id or self.runtime.episode_id,
            context_id=self.runtime.context_id,
            retry_stage=retry_stage,
            latency_ms=final_latency_ms,
            raw_text="",
            parsed_payload={},
            validated_segments=[],
            attach_notes=[],
            validation_errors=[str(last_exc)],
            session_restarted=session_restarted,
            raw_response_path="",
        )


__all__ = [
    "GeminiResult",
    "GeminiSession",
    "validate_normalized_segments",
]
