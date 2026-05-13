"""Live validation tracker for overlong repair effectiveness and submit verification.

Hooks into the existing pipeline by wrapping _capture_step_artifacts and the
repair/submit flow with enriched screenshot checkpoints and metrics collection.

Design:
  - ValidationTracker accumulates state across one episode lifecycle
  - Each checkpoint captures screenshot + structured metrics
  - Final report is written as JSON alongside existing step_screenshots/
  - Does NOT change existing pipeline behavior; only observes and records
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.infra.solver_config import _cfg_get

_logger_name = "src.solver.live_validation"

try:
    import logging

    _logger = logging.getLogger(_logger_name)
    from src.infra.logging_utils import build_print_logger as _build_print_logger

    print = _build_print_logger(_logger)
except Exception:
    _logger = None


@dataclass
class OverlongRepairCheckpoint:
    round_no: int
    timestamp_utc: str
    overlong_indices_before: List[int]
    overlong_durations_before: Dict[int, float]
    max_duration_sec: float
    repair_action: str
    split_ops_planned: List[Dict[str, Any]]
    split_ops_applied: int
    split_ops_failed: int
    overlong_indices_after: List[int]
    overlong_durations_after: Dict[int, float]
    segment_count_before: int
    segment_count_after: int
    stagnant: bool
    screenshot_before: str = ""
    screenshot_after: str = ""
    html_before: str = ""
    html_after: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubmitVerificationCheckpoint:
    timestamp_utc: str
    complete_button_selector: str
    complete_button_found: bool
    complete_button_clicked: bool
    complete_button_retried: bool
    modal_already_open: bool
    saw_no_edits_modal: bool
    no_edits_confirmed: bool
    saw_quality_review_modal: bool
    quality_review_confirmed: bool
    saw_post_submit_transition: bool
    submit_verified: bool
    verification_reason: str
    page_url_before: str
    page_url_after: str
    toast_error_detected: bool
    toast_error_text: str
    submit_guard_blocked: bool
    submit_guard_reasons: List[str]
    deep_dashboard_verified: bool
    screenshot_before_click: str = ""
    screenshot_after_click: str = ""
    screenshot_after_verify: str = ""
    html_before_click: str = ""
    html_after_verify: str = ""
    latency_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeValidationReport:
    episode_id: str
    started_at_utc: str
    finished_at_utc: str
    total_segments: int
    initial_overlong_count: int
    initial_overlong_indices: List[int]
    final_overlong_count: int
    final_overlong_indices: List[int]
    repair_rounds: int
    repair_effective: bool
    repair_checkpoints: List[OverlongRepairCheckpoint]
    submit_checkpoint: Optional[SubmitVerificationCheckpoint]
    submit_succeeded: bool
    validation_passed: bool
    failure_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["repair_checkpoints"] = [cp.to_dict() for cp in self.repair_checkpoints]
        if self.submit_checkpoint is not None:
            d["submit_checkpoint"] = self.submit_checkpoint.to_dict()
        return d


class ValidationTracker:
    """Accumulates validation state for one episode.

    Usage:
        tracker = ValidationTracker(cfg, episode_id)
        tracker.record_repair_before(round_no, segments, overlong_indices, ...)
        tracker.record_repair_after(round_no, segments, overlong_indices, ...)
        tracker.record_submit_before(submit_status_snapshot, page)
        tracker.record_submit_after(submit_result, page)
        tracker.finalize(total_segments, ...)
        tracker.save_report()
    """

    def __init__(self, cfg: Dict[str, Any], episode_id: str) -> None:
        self.cfg = cfg
        self.episode_id = str(episode_id or "unknown").strip()
        self.started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.repair_checkpoints: List[OverlongRepairCheckpoint] = []
        self._pending_repair: Optional[OverlongRepairCheckpoint] = None
        self.submit_checkpoint: Optional[SubmitVerificationCheckpoint] = None
        self._submit_start: float = 0.0
        self._initial_overlong_indices: List[int] = []
        self._initial_overlong_durations: Dict[int, float] = {}
        self._initial_segment_count: int = 0
        self._repair_rounds: int = 0
        self._max_duration_sec: float = 0.0

    def _output_dir(self) -> Path:
        out = Path(str(_cfg_get(self.cfg, "run.output_dir", "outputs")))
        val_dir = out / "live_validation"
        val_dir.mkdir(parents=True, exist_ok=True)
        return val_dir

    def _now_utc(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @staticmethod
    def _overlong_from_segments(
        segments: List[Dict[str, Any]],
        max_dur: float,
    ) -> Tuple[List[int], Dict[int, float]]:
        indices = []
        durations = {}
        for seg in segments or []:
            idx = int(seg.get("segment_index", 0) or 0)
            start = float(seg.get("start_sec", 0) or 0)
            end = float(seg.get("end_sec", 0) or 0)
            dur = end - start
            if dur > max_dur and idx > 0:
                indices.append(idx)
                durations[idx] = round(dur, 2)
        indices.sort()
        return indices, durations

    def set_initial_state(
        self,
        segments: List[Dict[str, Any]],
        max_duration_sec: float,
    ) -> None:
        self._max_duration_sec = max_duration_sec
        self._initial_segment_count = len(segments or [])
        indices, durations = self._overlong_from_segments(segments, max_duration_sec)
        self._initial_overlong_indices = indices
        self._initial_overlong_durations = durations
        print(
            f"[validation] initial state: {len(segments or [])} segments, "
            f"{len(indices)} overlong (max={max_duration_sec}s): {indices}"
        )

    def overlong_snapshot(
        self,
        segments: List[Dict[str, Any]],
    ) -> Tuple[List[int], Dict[int, float]]:
        if self._max_duration_sec <= 0:
            return [], {}
        return self._overlong_from_segments(segments, self._max_duration_sec)

    def record_repair_before(
        self,
        round_no: int,
        segments: List[Dict[str, Any]],
        overlong_indices: List[int],
        split_ops_planned: List[Dict[str, Any]],
        *,
        repair_action: str = "",
        screenshot_path: str = "",
        html_path: str = "",
    ) -> None:
        _, durations = self._overlong_from_segments(segments, self._max_duration_sec)
        action = str(repair_action or "").strip()
        if not action:
            action = (
                "chat_targeted"
                if bool(_cfg_get(self.cfg, "run.chat_only_mode", False))
                else "simple_split"
            )
        self._pending_repair = OverlongRepairCheckpoint(
            round_no=round_no,
            timestamp_utc=self._now_utc(),
            overlong_indices_before=sorted(overlong_indices or []),
            overlong_durations_before={
                idx: durations.get(idx, 0.0) for idx in (overlong_indices or [])
            },
            max_duration_sec=self._max_duration_sec,
            repair_action=action,
            split_ops_planned=list(split_ops_planned or []),
            split_ops_applied=0,
            split_ops_failed=0,
            overlong_indices_after=[],
            overlong_durations_after={},
            segment_count_before=len(segments or []),
            segment_count_after=0,
            stagnant=False,
            screenshot_before=screenshot_path,
            html_before=html_path,
        )

    def record_repair_after(
        self,
        segments: List[Dict[str, Any]],
        overlong_indices_after: List[int],
        split_ops_applied: int,
        split_ops_failed: int,
        stagnant: bool = False,
        *,
        screenshot_path: str = "",
        html_path: str = "",
        error: str = "",
    ) -> None:
        if self._pending_repair is None:
            return
        cp = self._pending_repair
        _, durations_after = self._overlong_from_segments(
            segments, self._max_duration_sec
        )
        cp.overlong_indices_after = sorted(overlong_indices_after or [])
        cp.overlong_durations_after = {
            idx: durations_after.get(idx, 0.0) for idx in (overlong_indices_after or [])
        }
        cp.segment_count_after = len(segments or [])
        cp.split_ops_applied = split_ops_applied
        cp.split_ops_failed = split_ops_failed
        cp.stagnant = stagnant
        cp.screenshot_after = screenshot_path
        cp.html_after = html_path
        cp.error = error
        self.repair_checkpoints.append(cp)
        self._pending_repair = None
        self._repair_rounds = max(self._repair_rounds, cp.round_no)

        before_count = len(cp.overlong_indices_before)
        after_count = len(cp.overlong_indices_after)
        improved = after_count < before_count
        print(
            f"[validation] repair round {cp.round_no}: "
            f"overlong {before_count} -> {after_count} "
            f"{'IMPROVED' if improved else 'NO CHANGE' if before_count == after_count else 'WORSE'} "
            f"applied={split_ops_applied} failed={split_ops_failed}"
        )

    def record_submit_before(
        self,
        page_url: str,
        complete_sel: str,
        *,
        screenshot_path: str = "",
        html_path: str = "",
    ) -> None:
        self._submit_start = time.time()
        self.submit_checkpoint = SubmitVerificationCheckpoint(
            timestamp_utc=self._now_utc(),
            complete_button_selector=complete_sel,
            complete_button_found=False,
            complete_button_clicked=False,
            complete_button_retried=False,
            modal_already_open=False,
            saw_no_edits_modal=False,
            no_edits_confirmed=False,
            saw_quality_review_modal=False,
            quality_review_confirmed=False,
            saw_post_submit_transition=False,
            submit_verified=False,
            verification_reason="pending",
            page_url_before=page_url,
            page_url_after="",
            toast_error_detected=False,
            toast_error_text="",
            submit_guard_blocked=False,
            submit_guard_reasons=[],
            deep_dashboard_verified=False,
            screenshot_before_click=screenshot_path,
            html_before_click=html_path,
        )

    def record_submit_after(
        self,
        submit_status: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
        *,
        screenshot_after_click: str = "",
        screenshot_after_verify: str = "",
        html_after_verify: str = "",
    ) -> None:
        if self.submit_checkpoint is None:
            return
        cp = self.submit_checkpoint
        ss = submit_status if isinstance(submit_status, dict) else {}
        cp.complete_button_found = bool(
            ss.get("complete_button_clicked", False)
            or ss.get("submit_modal_already_open", False)
            or ss.get("submit_clicked", False)
        )
        cp.complete_button_clicked = bool(
            ss.get("complete_button_clicked", False) or ss.get("submit_clicked", False)
        )
        cp.complete_button_retried = bool(ss.get("complete_button_retried", False))
        cp.modal_already_open = bool(ss.get("submit_modal_already_open", False))
        cp.saw_no_edits_modal = bool(ss.get("saw_no_edits_modal", False))
        cp.no_edits_confirmed = bool(ss.get("no_edits_confirmed", False))
        cp.saw_quality_review_modal = bool(ss.get("saw_quality_review_modal", False))
        cp.quality_review_confirmed = bool(ss.get("quality_review_confirmed", False))
        cp.saw_post_submit_transition = bool(
            ss.get("saw_post_submit_transition", False)
        )
        cp.submit_verified = bool(
            ss.get("submit_verified", ss.get("verified", False))
        )
        cp.verification_reason = str(
            ss.get("submit_verification_reason", "")
            or ss.get("method", "")
            or ""
        ).strip()
        cp.page_url_after = str(ss.get("page_url_after_submit", "") or "").strip()
        last_error_text = str(ss.get("last_error", "") or "").strip()
        cp.toast_error_detected = bool(last_error_text) and "UI Validation" in last_error_text
        cp.toast_error_text = (
            last_error_text
            if cp.toast_error_detected
            else ""
        )
        if result and isinstance(result, dict):
            cp.submit_guard_blocked = bool(result.get("submit_guard_blocked", False))
            cp.submit_guard_reasons = [
                str(r) for r in (result.get("submit_guard_reasons", []) or [])[:10]
            ]
            cp.deep_dashboard_verified = bool(
                ss.get("dashboard_verified", False)
                or result.get("deep_dashboard_verified", False)
            )
        cp.screenshot_after_click = screenshot_after_click
        cp.screenshot_after_verify = screenshot_after_verify
        cp.html_after_verify = html_after_verify
        if self._submit_start > 0:
            cp.latency_sec = round(time.time() - self._submit_start, 2)

        status = "VERIFIED" if cp.submit_verified else "FAILED"
        print(
            f"[validation] submit {status}: reason={cp.verification_reason} "
            f"clicked={cp.complete_button_clicked} transition={cp.saw_post_submit_transition} "
            f"quality_modal={cp.saw_quality_review_modal} latency={cp.latency_sec:.1f}s"
        )

    def finalize(
        self,
        total_segments: int,
        final_overlong_count: int = 0,
        final_overlong_indices: Optional[List[int]] = None,
    ) -> EpisodeValidationReport:
        repair_effective = len(self._initial_overlong_indices) > 0 and (
            final_overlong_count or 0
        ) < len(self._initial_overlong_indices)
        if len(self._initial_overlong_indices) == 0:
            repair_effective = True

        submit_succeeded = bool(
            self.submit_checkpoint and self.submit_checkpoint.submit_verified
        )
        validation_passed = True
        failure_parts: List[str] = []

        if len(self._initial_overlong_indices) > 0 and not repair_effective:
            validation_passed = False
            failure_parts.append(
                f"overlong_repair_ineffective: {len(self._initial_overlong_indices)} segments "
                f"overlong before, {final_overlong_count} after {self._repair_rounds} rounds"
            )

        if self.submit_checkpoint is not None and not submit_succeeded:
            validation_passed = False
            reason = self.submit_checkpoint.verification_reason or "unknown"
            failure_parts.append(f"submit_failed: {reason}")

        finished_at = self._now_utc()
        report = EpisodeValidationReport(
            episode_id=self.episode_id,
            started_at_utc=self.started_at_utc,
            finished_at_utc=finished_at,
            total_segments=total_segments,
            initial_overlong_count=len(self._initial_overlong_indices),
            initial_overlong_indices=self._initial_overlong_indices,
            final_overlong_count=final_overlong_count or 0,
            final_overlong_indices=final_overlong_indices or [],
            repair_rounds=self._repair_rounds,
            repair_effective=repair_effective,
            repair_checkpoints=self.repair_checkpoints,
            submit_checkpoint=self.submit_checkpoint,
            submit_succeeded=submit_succeeded,
            validation_passed=validation_passed,
            failure_summary="; ".join(failure_parts),
        )
        return report

    def save_report(self, report: EpisodeValidationReport) -> Path:
        out_dir = self._output_dir()
        safe_id = (
            re.sub(r"[^A-Za-z0-9._-]+", "_", self.episode_id).strip("._-") or "unknown"
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = out_dir / f"validation_{safe_id}_{timestamp}.json"
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False, sort_keys=False),
            encoding="utf-8",
        )
        print(f"[validation] report saved: {report_path}")
        return report_path
__all__ = [
    "OverlongRepairCheckpoint",
    "SubmitVerificationCheckpoint",
    "EpisodeValidationReport",
    "ValidationTracker",
]
