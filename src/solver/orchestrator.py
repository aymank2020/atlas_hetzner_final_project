"""Main run-loop orchestration helpers extracted from the legacy solver."""

from __future__ import annotations

import logging
import re
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.infra.execution_journal import append_execution_journal_event
from src.solver.live_validation import ValidationTracker
from src.solver.desync import build_segment_checksum

from src.infra.gemini_economics import (
    cost_guard_enforcement_enabled,
    estimate_cost_usd,
    resolve_stage_model,
    would_exceed_ratio_cap,
)
from src.infra.logging_utils import build_print_logger as _build_print_logger
from src.infra.solver_config import _cfg_get

_logger = logging.getLogger(__name__)
print = _build_print_logger(_logger)

_OVERLONG_DURATION_ERROR_RE = re.compile(
    r"segment\s+(?P<idx>\d+):\s+duration\s+(?P<duration>\d+(?:\.\d+)?)s\s+exceeds\s+max\s+(?P<max>\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)


def _last_step_artifact_paths(task_state: Optional[Dict[str, Any]]) -> tuple[str, str]:
    if not isinstance(task_state, dict):
        return "", ""
    return (
        str(task_state.get("last_step_screenshot", "") or "").strip(),
        str(task_state.get("last_step_html", "") or "").strip(),
    )


def _update_retry_metadata(
    *,
    task_state: Optional[Dict[str, Any]],
    labels_payload: Optional[Dict[str, Any]] = None,
    retry_stage: str = "",
    retry_reason: str = "",
) -> Optional[Dict[str, Any]]:
    stage = str(retry_stage or "").strip()
    reason = str(retry_reason or "").strip()
    if isinstance(labels_payload, dict):
        meta = labels_payload.setdefault("_meta", {})
        if stage:
            meta["retry_stage"] = stage
        if reason:
            meta["retry_reason"] = reason
    if isinstance(task_state, dict):
        if stage:
            task_state["gemini_last_retry_stage"] = stage
        if reason:
            task_state["gemini_last_retry_reason"] = reason
    return task_state


def _maybe_retry_policy_with_stronger_model(
    cfg: Dict[str, Any],
    segments: List[Dict[str, Any]],
    prompt: str,
    video_file: Optional[Path],
    labels_payload: Dict[str, Any],
    segment_plan: Dict[int, Dict[str, Any]],
    validation_report: Dict[str, Any],
    *,
    task_id: str = "",
    execute: bool = False,
    execute_require_video_context: bool = False,
    gemini_uploaded_file_names: Optional[List[str]] = None,
    resume_from_artifacts: bool = False,
    task_state: Optional[Dict[str, Any]] = None,
    heartbeat: Any = None,
) -> Dict[str, Any]:
    legacy = import_module("src.solver.legacy_impl")
    gemini_mod = import_module("src.solver.gemini")

    def _heartbeat() -> None:
        if callable(heartbeat):
            try:
                heartbeat()
            except Exception:
                pass

    raw_errors = [
        str(e).strip() for e in validation_report.get("errors", []) if str(e).strip()
    ]
    chat_only_mode = bool(_cfg_get(cfg, "run.chat_only_mode", False))
    chat_only_policy_retry_enabled = bool(
        _cfg_get(cfg, "run.chat_only_policy_retry_enabled", False)
    )
    retry_with_stronger_model = bool(
        _cfg_get(cfg, "gemini.retry_with_stronger_model_on_policy_fail", False)
    )
    policy_retry_model = str(
        resolve_stage_model(
            cfg, "policy_retry", _cfg_get(cfg, "gemini.policy_retry_model", "")
        )
    ).strip()
    policy_retry_only_if_flash = bool(
        _cfg_get(cfg, "gemini.policy_retry_only_if_flash", True)
    )
    current_model = str(
        (labels_payload.get("_meta", {}) or {}).get(
            "model",
            _cfg_get(cfg, "gemini.model", "gemini-3.1-pro-preview"),
        )
    ).strip()

    can_retry_with_stronger_model = (
        retry_with_stronger_model
        and bool(raw_errors)
        and bool(policy_retry_model)
        and policy_retry_model.lower() != current_model.lower()
    )
    if chat_only_mode:
        policy_retry_model = str(policy_retry_model or current_model).strip()
        can_retry_with_stronger_model = (
            chat_only_policy_retry_enabled
            and bool(raw_errors)
            and bool(policy_retry_model)
        )
    if (
        can_retry_with_stronger_model
        and policy_retry_only_if_flash
        and not chat_only_mode
    ):
        can_retry_with_stronger_model = "flash" in current_model.lower()

    result = {
        "labels_payload": labels_payload,
        "segment_plan": segment_plan,
        "validation_report": validation_report,
        "task_state": task_state,
        "retried": False,
        "adopted_retry": False,
    }

    if not can_retry_with_stronger_model:
        return result
    if not chat_only_mode and bool(
        getattr(gemini_mod, "_is_gemini_model_zero_quota_known")(policy_retry_model)
    ):
        print(
            "[policy] skipping stronger-model retry: zero-quota model cooldown is active "
            f"for {policy_retry_model}."
        )
        return result

    _heartbeat()
    if chat_only_mode:
        print(
            "[policy] validation failed; retrying chat-only solve once "
            f"with {policy_retry_model}..."
        )
    else:
        print(
            "[policy] validation failed; retrying Gemini with stronger model "
            f"({current_model} -> {policy_retry_model})..."
        )
    # Prefer the optimized upload video so uploads stay under the
    # CDP 50 MB transfer limit on remote-browser setups.
    _retry_video = video_file
    if video_file is not None:
        _opt = video_file.with_name(video_file.stem + "_upload_opt.mp4")
        if _opt.exists() and _opt.stat().st_size > 0:
            _retry_video = _opt
    try:
        retry_payload = legacy._request_labels_with_optional_segment_chunking(
            cfg,
            segments,
            prompt,
            _retry_video,
            allow_operations=False,
            model_override=policy_retry_model,
            task_id=task_id,
            task_state=task_state,
            stage_name="policy_retry",
        )
        _heartbeat()
        result["retried"] = True

        if execute and execute_require_video_context:
            retry_meta = (
                retry_payload.get("_meta", {})
                if isinstance(retry_payload, dict)
                else {}
            )
            retry_video_attached = bool(retry_meta.get("video_attached", False))
            retry_mode = str(retry_meta.get("mode", "unknown"))

            if gemini_uploaded_file_names is not None:
                for fname in retry_meta.get("uploaded_file_names", []):
                    if fname not in gemini_uploaded_file_names:
                        gemini_uploaded_file_names.append(fname)

            if not retry_video_attached:
                raise RuntimeError(
                    "Execute blocked: stronger-model retry returned text-only "
                    "(no video context)."
                )
            print(f"[gemini] stronger-model retry has video context ({retry_mode}).")

        retry_plan = legacy._normalize_segment_plan(retry_payload, segments, cfg=cfg)
        retry_no_action_rewrites = legacy._rewrite_no_action_pauses_in_plan(
            retry_plan, cfg
        )
        if retry_no_action_rewrites:
            print(
                "[policy] stronger-model pass rewrote short no-action pauses: "
                f"{retry_no_action_rewrites}"
            )
        retry_validation_report = legacy._validate_segment_plan_against_policy(
            cfg, segments, retry_plan
        )
        _heartbeat()
        retry_raw_errors = [
            str(e).strip()
            for e in retry_validation_report.get("errors", [])
            if str(e).strip()
        ]

        if len(retry_raw_errors) <= len(raw_errors):
            print(
                "[policy] accepted stronger-model retry: "
                f"errors {len(raw_errors)} -> {len(retry_raw_errors)}"
            )
            result["labels_payload"] = retry_payload
            result["segment_plan"] = retry_plan
            result["validation_report"] = retry_validation_report
            result["adopted_retry"] = True
            legacy._save_outputs(cfg, segments, prompt, retry_payload, task_id=task_id)
            if task_id:
                legacy._save_task_text_files(cfg, task_id, segments, retry_plan)
                legacy._save_cached_labels(cfg, task_id, retry_payload)
                if isinstance(task_state, dict):
                    task_state = legacy._persist_task_state_fields(
                        cfg,
                        task_id,
                        task_state,
                        labels_ready=True,
                        last_error="",
                        **legacy._episode_model_state_updates(
                            cfg, retry_payload, task_state
                        ),
                    )
        else:
            print(
                "[policy] kept primary-model output: stronger-model retry was not better "
                f"({len(raw_errors)} -> {len(retry_raw_errors)})."
            )
    except Exception as retry_exc:
        print(f"[policy] stronger-model retry failed: {retry_exc}")

    return result


def _overlong_segment_indices_from_validation_report(
    validation_report: Dict[str, Any],
) -> List[int]:
    ranked: List[tuple[float, int]] = []
    seen: set[int] = set()
    for raw_error in validation_report.get("errors", []) or []:
        text = str(raw_error or "").strip()
        if not text:
            continue
        match = _OVERLONG_DURATION_ERROR_RE.search(text)
        if not match:
            continue
        try:
            idx = int(match.group("idx"))
        except Exception:
            continue
        if idx <= 0 or idx in seen:
            continue
        seen.add(idx)
        try:
            duration = float(match.group("duration"))
        except Exception:
            duration = 0.0
        ranked.append((duration, idx))
    ranked.sort(key=lambda item: (-item[0], -item[1]))
    return [idx for _, idx in ranked]


def _targeted_repair_scope_indices(
    source_segments: Sequence[Dict[str, Any]],
    failing_indices: Sequence[int],
    *,
    neighbor_count: int = 2,
) -> List[int]:
    ordered = [
        int(seg.get("segment_index", 0) or 0)
        for seg in (source_segments or [])
        if int(seg.get("segment_index", 0) or 0) > 0
    ]
    if not ordered:
        return []
    positions = {idx: pos for pos, idx in enumerate(ordered)}
    scope: set[int] = set()
    radius = max(0, int(neighbor_count or 0))
    for raw_idx in failing_indices or []:
        idx = int(raw_idx or 0)
        pos = positions.get(idx)
        if pos is None:
            continue
        start = max(0, pos - radius)
        end = min(len(ordered) - 1, pos + radius)
        scope.update(ordered[start : end + 1])
    return [idx for idx in ordered if idx in scope]


def _expand_contiguous_failure_targets(
    source_segments: Sequence[Dict[str, Any]],
    failing_indices: Sequence[int],
    *,
    base_limit: int,
) -> List[int]:
    ordered = [
        int(seg.get("segment_index", 0) or 0)
        for seg in (source_segments or [])
        if int(seg.get("segment_index", 0) or 0) > 0
    ]
    failing_ordered = [
        idx for idx in ordered if int(idx or 0) in {int(item or 0) for item in (failing_indices or []) if int(item or 0) > 0}
    ]
    if not failing_ordered:
        return []
    selected = list(failing_ordered[: max(1, int(base_limit or 1))])
    selected_set = set(selected)
    if not selected:
        return []
    positions = {idx: pos for pos, idx in enumerate(ordered)}
    failing_set = set(failing_ordered)
    left = min(selected, key=lambda item: positions.get(item, 10**9))
    right = max(selected, key=lambda item: positions.get(item, -1))
    left_pos = positions.get(left)
    right_pos = positions.get(right)
    if left_pos is None or right_pos is None:
        return selected
    start = left_pos
    end = right_pos
    while start > 0 and ordered[start - 1] in failing_set:
        start -= 1
    while end < (len(ordered) - 1) and ordered[end + 1] in failing_set:
        end += 1
    expanded = [
        idx
        for idx in ordered[start : end + 1]
        if idx in failing_set or idx in selected_set
    ]
    return expanded or selected


def _maybe_repair_overlong_segments(
    cfg: Dict[str, Any],
    page: Any,
    segments: List[Dict[str, Any]],
    prompt: str,
    video_file: Optional[Path],
    labels_payload: Dict[str, Any],
    segment_plan: Dict[int, Dict[str, Any]],
    validation_report: Dict[str, Any],
    *,
    task_id: str = "",
    execute: bool = False,
    execute_require_video_context: bool = False,
    gemini_uploaded_file_names: Optional[List[str]] = None,
    resume_from_artifacts: bool = False,
    task_state: Optional[Dict[str, Any]] = None,
    enable_structural_actions: bool = True,
    requery_after_structural_actions: bool = True,
    heartbeat: Any = None,
    validation_tracker: Optional[ValidationTracker] = None,
) -> Dict[str, Any]:
    legacy = import_module("src.solver.legacy_impl")

    def _heartbeat() -> None:
        if callable(heartbeat):
            try:
                heartbeat()
            except Exception:
                pass

    def _capture_step(step_name: str, *, include_html: bool = False) -> None:
        nonlocal task_state
        if not task_id:
            return
        try:
            task_state = legacy._capture_episode_step(
                cfg,
                page,
                task_id,
                task_state,
                str(step_name or "").strip(),
                include_html=include_html,
            )
        except Exception:
            return

    def _journal_repair_event(
        event_type: str,
        *,
        reason: str = "",
        repair_round: int = 0,
        payload: Optional[Dict[str, Any]] = None,
        segments_snapshot: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        if not task_id:
            return
        extra = dict(payload or {})
        if segments_snapshot:
            extra.setdefault("segment_count", len(list(segments_snapshot)))
            extra.setdefault("segments_checksum", build_segment_checksum(list(segments_snapshot)))
        append_execution_journal_event(
            cfg,
            episode_id=task_id,
            event_type=event_type,
            stage="repairing",
            reason=reason,
            task_state=task_state,
            payload=extra,
            run_id=str((task_state or {}).get("run_id", (task_state or {}).get("context_id", "")) or "").strip(),
            context_id=str((task_state or {}).get("context_id", "") or "").strip(),
            repair_round=repair_round,
            segments_checksum=str(extra.get("segments_checksum", "") or "").strip(),
            page_url=str(getattr(page, "url", "") or "").strip(),
        )

    def _record_repair_before(
        round_no: int,
        overlong_indices_before: List[int],
        split_ops_planned: List[Dict[str, Any]],
        *,
        repair_action: str = "",
    ) -> None:
        if validation_tracker is None:
            return
        screenshot_path, html_path = _last_step_artifact_paths(task_state)
        validation_tracker.record_repair_before(
            round_no,
            current_segments,
            overlong_indices_before,
            split_ops_planned,
            repair_action=repair_action,
            screenshot_path=screenshot_path,
            html_path=html_path,
        )

    def _record_repair_after(
        overlong_indices_after: List[int],
        *,
        split_ops_applied: int = 0,
        split_ops_failed: int = 0,
        stagnant: bool = False,
        error: str = "",
    ) -> None:
        if validation_tracker is None:
            return
        screenshot_path, html_path = _last_step_artifact_paths(task_state)
        validation_tracker.record_repair_after(
            current_segments,
            overlong_indices_after,
            split_ops_applied,
            split_ops_failed,
            stagnant=stagnant,
            screenshot_path=screenshot_path,
            html_path=html_path,
            error=error,
        )

    result = {
        "segments": segments,
        "prompt": prompt,
        "labels_payload": labels_payload,
        "segment_plan": segment_plan,
        "validation_report": validation_report,
        "task_state": task_state,
        "repair_rounds": 0,
        "skip_compare": False,
        "retry_stage": "",
        "retry_reason": "",
        "repair_skipped_reason": "",
    }

    if not bool(_cfg_get(cfg, "run.policy_auto_split_repair_enabled", False)):
        result["repair_skipped_reason"] = "run.policy_auto_split_repair_enabled=false"
        return result
    if not execute:
        result["repair_skipped_reason"] = (
            "execute=false (dry-run); policy auto-repair only runs during the execute/apply pass"
        )
        return result
    if not enable_structural_actions:
        result["repair_skipped_reason"] = "run.enable_structural_actions=false"
        return result
    if not bool(_cfg_get(cfg, "run.structural_allow_split", False)):
        result["repair_skipped_reason"] = "run.structural_allow_split=false"
        return result
    if not requery_after_structural_actions:
        result["repair_skipped_reason"] = "run.requery_after_structural_actions=false"
        return result

    max_rounds = max(
        0, int(_cfg_get(cfg, "run.policy_auto_split_repair_max_rounds", 0) or 0)
    )
    if bool(_cfg_get(cfg, "run.chat_only_mode", False)):
        targeted_rounds = max(
            0,
            int(
                _cfg_get(cfg, "run.targeted_repair_max_rounds", max_rounds)
                or max_rounds
            ),
        )
        max_rounds = max(1, targeted_rounds) if max_rounds > 0 else targeted_rounds
    max_splits_per_round = max(
        1,
        int(
            _cfg_get(cfg, "run.policy_auto_split_repair_max_segments_per_round", 3) or 3
        ),
    )
    if max_rounds <= 0:
        result["repair_skipped_reason"] = "run.policy_auto_split_repair_max_rounds<=0"
        return result

    current_segments = segments
    current_prompt = prompt
    current_payload = labels_payload
    current_plan = segment_plan
    current_report = validation_report
    stagnant_rounds = 0
    repair_stage_model = str(
        resolve_stage_model(
            cfg, "repair", _cfg_get(cfg, "gemini.model", "gemini-2.5-flash")
        )
    ).strip()
    projected_repair_cost_usd = estimate_cost_usd(
        cfg,
        repair_stage_model,
        prompt_tokens=3994,
        output_tokens=178,
        total_tokens=4172,
    )

    for round_no in range(1, max_rounds + 1):
        _heartbeat()
        if round_no > 1:
            projected_ratio_exceeded = would_exceed_ratio_cap(
                cfg,
                task_state if isinstance(task_state, dict) else None,
                additional_cost_usd=projected_repair_cost_usd,
                ratio_limit=float(
                    _cfg_get(cfg, "economics.target_cost_ratio", 0.15) or 0.15
                ),
            )
            if projected_ratio_exceeded and cost_guard_enforcement_enabled(cfg):
                print(
                    "[economics] skipping additional overlong repair round: "
                    f"projected delta=${projected_repair_cost_usd:.4f} would exceed target ratio."
                )
                break
            if projected_ratio_exceeded:
                print(
                    "[economics] projected additional overlong repair round exceeds target ratio, "
                    "but cost guards are disabled; continuing."
                )
        overlong_indices = _overlong_segment_indices_from_validation_report(
            current_report
        )
        if not overlong_indices:
            break

        # Prepare a scoped repair video clip covering only the overlong
        # segments so uploads stay under the CDP 50 MB transfer limit.
        from src.solver.video_core import _prepare_repair_video_clip as _make_repair_clip

        _repair_out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs")))
        _repair_episode_id = str(task_id).replace("episode_", "") if task_id else ""
        effective_repair_video = video_file
        if video_file is not None:
            _repair_clip = _make_repair_clip(
                video_file=video_file,
                segments=current_segments,
                target_indices=list(overlong_indices),
                out_dir=_repair_out_dir,
                episode_id=_repair_episode_id,
                pad_sec=max(0.0, float(
                    _cfg_get(cfg, "run.repair_clip_pad_sec", 2.0) or 2.0
                )),
                cdp_max_mb=max(1.0, float(
                    _cfg_get(cfg, "run.repair_clip_cdp_max_mb", 45.0) or 45.0
                )),
            )
            if _repair_clip is not None:
                effective_repair_video = _repair_clip

        chat_only_mode = bool(_cfg_get(cfg, "run.chat_only_mode", False))
        if chat_only_mode:
            retry_stage = "targeted_repair_1" if round_no == 1 else "targeted_repair_2"
            current_failure_signature = tuple(
                sorted(int(idx) for idx in overlong_indices)
            )
            target_indices = [
                int(idx)
                for idx in _expand_contiguous_failure_targets(
                    current_segments,
                    overlong_indices,
                    base_limit=max_splits_per_round,
                )
            ]
            max_segment_duration_sec = max(
                0.1,
                float(_cfg_get(cfg, "run.max_segment_duration_sec", 10.0) or 10.0),
            )
            neighbor_count = max(
                0, int(_cfg_get(cfg, "run.targeted_repair_scope_neighbors", 2) or 2)
            )
            print(
                f"[policy] chat-feedback auto-repairing overlong segments: "
                f"round {round_no}/{max_rounds} targets={target_indices}"
            )
            _journal_repair_event(
                "repair_round_started",
                reason="policy_overlong",
                repair_round=round_no,
                payload={
                    "target_indices": list(target_indices),
                    "neighbor_count": int(neighbor_count),
                    "overlong_indices": list(overlong_indices),
                },
                segments_snapshot=current_segments,
            )
            chat_only = import_module("src.solver.chat_only")
            scoped_indices = _targeted_repair_scope_indices(
                current_segments,
                target_indices,
                neighbor_count=neighbor_count,
            )
            scoped_segments = [
                seg
                for seg in current_segments
                if int(seg.get("segment_index", 0) or 0) in set(scoped_indices)
            ]
            repair_prompt = chat_only.build_targeted_repair_planner_prompt(
                current_segments,
                failing_indices=target_indices,
                allow_merge=False,
                max_segment_duration_sec=max_segment_duration_sec,
                neighbor_count=neighbor_count,
                extra_instructions=(
                    "Only repair the failing overlong rows when the live DOM visibly supports a split. "
                    "Keep all operations inside the provided local scope."
                ),
            )
            _capture_step(f"before_targeted_repair_round_{round_no}", include_html=True)

            try:
                cfg_dir = Path(_cfg_get(cfg, "_meta.config_dir", ".")).resolve()
                cache_dir = (
                    cfg_dir / ".state" / "cache" / "labels" / str(task_id or "unknown")
                )
                episode_id = str(task_id).replace("episode_", "") if task_id else ""

                if video_file is None:
                    raise RuntimeError(
                        "targeted chat repair requires a task video file"
                    )

                chat_repair_res = chat_only.run_structural_planner(
                    cfg=cfg,
                    source_segments=scoped_segments,
                    prompt_text=repair_prompt,
                    cache_dir=cache_dir,
                    episode_id=episode_id,
                    model=repair_stage_model,
                    video_file=effective_repair_video,
                    allow_merge=False,
                    prompt_scope="chat_ops",
                    heartbeat=_heartbeat,
                )
                _capture_step(f"after_targeted_repair_plan_{round_no}")
                split_ops = [
                    {
                        "action": "split",
                        "segment_index": int(item.get("segment_index", 0) or 0),
                    }
                    for item in (chat_repair_res.get("operations", []) or [])
                    if str(item.get("action", "")).strip().lower() == "split"
                    and int(item.get("segment_index", 0) or 0) in set(scoped_indices)
                ]
                _record_repair_before(
                    round_no,
                    list(overlong_indices),
                    split_ops,
                )
                if not split_ops:
                    print(
                        "[policy] chat-repair planner returned no usable split operations."
                    )
                    _capture_step(f"targeted_repair_noop_{round_no}", include_html=True)
                    _record_repair_after(
                        list(overlong_indices),
                        stagnant=True,
                    )
                    _journal_repair_event(
                        "repair_round_noop",
                        reason="planner_returned_no_split_ops",
                        repair_round=round_no,
                        payload={"target_indices": list(target_indices)},
                        segments_snapshot=current_segments,
                    )
                    remaining = list(overlong_indices)
                    stagnant_rounds += 1
                    result["repair_rounds"] = round_no
                    result["retry_stage"] = retry_stage
                    result["retry_reason"] = "policy_overlong"
                    if stagnant_rounds >= 2:
                        print(
                            "[policy] overlong repair stopped after repeated no-op rounds."
                        )
                        break
                    continue

                op_result = legacy.apply_segment_operations(
                    page,
                    cfg,
                    split_ops,
                    source_segments=current_segments,
                    heartbeat=_heartbeat,
                )
                _heartbeat()
                _capture_step(f"after_targeted_repair_apply_{round_no}")
                print(
                    "[policy] chat-repair split operations applied: "
                    f"{op_result['applied']} (structural={op_result['structural_applied']})"
                )
                if op_result["failed"]:
                    print("[policy] chat-repair structural failures:")
                    for item in op_result["failed"]:
                        print(f"  - {item}")
                if op_result["structural_applied"] <= 0:
                    _capture_step(
                        f"targeted_repair_not_applied_{round_no}", include_html=True
                    )
                    _record_repair_after(
                        list(overlong_indices),
                        split_ops_applied=int(op_result.get("structural_applied", 0) or 0),
                        split_ops_failed=len(op_result.get("failed", []) or []),
                        stagnant=True,
                    )
                    _journal_repair_event(
                        "repair_round_not_applied",
                        reason="structural_ops_not_applied",
                        repair_round=round_no,
                        payload={
                            "split_ops_failed": len(op_result.get("failed", []) or []),
                            "split_ops_applied": int(op_result.get("structural_applied", 0) or 0),
                        },
                        segments_snapshot=current_segments,
                    )
                    remaining = list(overlong_indices)
                    stagnant_rounds += 1
                    result["repair_rounds"] = round_no
                    result["retry_stage"] = retry_stage
                    result["retry_reason"] = "policy_overlong"
                    if stagnant_rounds >= 2:
                        print(
                            "[policy] overlong repair stopped after repeated non-applied rounds."
                        )
                        break
                    continue

                current_segments = legacy.extract_segments(page, cfg)
                _heartbeat()
                print(
                    f"[atlas] extracted {len(current_segments)} segments (post-chat overlong repair)"
                )
                _journal_repair_event(
                    "segments_snapshot",
                    reason="post_chat_overlong_repair_extract",
                    repair_round=round_no,
                    payload={"segment_count": len(current_segments)},
                    segments_snapshot=current_segments,
                )
                if task_id and resume_from_artifacts:
                    legacy._save_cached_segments(cfg, task_id, current_segments)

                current_prompt = legacy.build_prompt(
                    current_segments,
                    str(_cfg_get(cfg, "gemini.extra_instructions", "")),
                    allow_operations=False,
                    policy_trigger="policy_conflict",
                )
                current_payload = legacy._request_labels_with_optional_segment_chunking(
                    cfg,
                    current_segments,
                    current_prompt,
                    effective_repair_video,
                    allow_operations=False,
                    task_id=task_id,
                    task_state=task_state,
                    stage_name="repair",
                )
                _update_retry_metadata(
                    task_state=task_state,
                    labels_payload=current_payload,
                    retry_stage=retry_stage,
                    retry_reason="policy_overlong",
                )
                _heartbeat()
                if execute and execute_require_video_context:
                    repair_meta = (
                        current_payload.get("_meta", {})
                        if isinstance(current_payload, dict)
                        else {}
                    )
                    repair_video_attached = bool(
                        repair_meta.get("video_attached", False)
                    )
                    repair_mode = str(repair_meta.get("mode", "unknown"))
                    if gemini_uploaded_file_names is not None:
                        for fname in repair_meta.get("uploaded_file_names", []):
                            if fname not in gemini_uploaded_file_names:
                                gemini_uploaded_file_names.append(fname)
                    if not repair_video_attached:
                        raise RuntimeError(
                            "Execute blocked: overlong chat-repair re-query returned text-only "
                            "(no video context)."
                        )
                    print(
                        f"[gemini] overlong chat-repair re-query has video context ({repair_mode})."
                    )

                post_ops = legacy._normalize_operations(current_payload, cfg=cfg)
                if post_ops:
                    print(
                        "[policy] ignoring operations in chat overlong repair labels-only pass."
                    )
                if task_id:
                    legacy._save_cached_labels(cfg, task_id, current_payload)
                if task_id and isinstance(task_state, dict):
                    task_state = legacy._persist_task_state_fields(
                        cfg,
                        task_id,
                        task_state,
                        labels_ready=True,
                        last_error="",
                        **legacy._episode_model_state_updates(
                            cfg, current_payload, task_state
                        ),
                    )

                legacy._save_outputs(
                    cfg,
                    current_segments,
                    current_prompt,
                    current_payload,
                    task_id=task_id,
                )
                current_plan = legacy._normalize_segment_plan(
                    current_payload, current_segments, cfg=cfg
                )
                no_action_rewrites = legacy._rewrite_no_action_pauses_in_plan(
                    current_plan, cfg
                )
                if no_action_rewrites:
                    print(
                        "[policy] chat overlong repair rewrote short no-action pauses: "
                        f"{no_action_rewrites}"
                    )
                if task_id:
                    legacy._save_task_text_files(
                        cfg, task_id, current_segments, current_plan
                    )
                current_report = legacy._validate_segment_plan_against_policy(
                    cfg, current_segments, current_plan
                )
                remaining = _overlong_segment_indices_from_validation_report(
                    current_report
                )
                print(
                    "[policy] overlong split repair round complete: "
                    f"remaining_overlong={len(remaining)}"
                )
                _capture_step(f"after_targeted_repair_round_{round_no}")
                result["repair_rounds"] = round_no
                result["retry_stage"] = retry_stage
                result["retry_reason"] = "policy_overlong"
                remaining_signature = tuple(sorted(int(idx) for idx in remaining))
                _record_repair_after(
                    list(remaining),
                    split_ops_applied=int(op_result.get("structural_applied", 0) or 0),
                    split_ops_failed=len(op_result.get("failed", []) or []),
                    stagnant=remaining_signature == current_failure_signature,
                )
                _journal_repair_event(
                    "repair_round_completed",
                    reason="policy_overlong",
                    repair_round=round_no,
                    payload={
                        "remaining_overlong": list(remaining),
                        "split_ops_applied": int(op_result.get("structural_applied", 0) or 0),
                        "split_ops_failed": len(op_result.get("failed", []) or []),
                    },
                    segments_snapshot=current_segments,
                )
                if remaining_signature == current_failure_signature:
                    stagnant_rounds += 1
                else:
                    stagnant_rounds = 0
                if not remaining:
                    break
                if stagnant_rounds >= 2:
                    print(
                        "[policy] overlong repair stopped after repeated unchanged failure signatures."
                    )
                    _capture_step(
                        f"targeted_repair_stagnant_{round_no}", include_html=True
                    )
                    break

            except Exception as repair_exc:
                repair_error = (
                    "Policy chat-repair re-query failed: "
                    f"{str(repair_exc).strip() or repair_exc.__class__.__name__}"
                )
                print(f"[policy] {repair_error}")
                _capture_step(f"targeted_repair_failed_{round_no}", include_html=True)
                if validation_tracker is not None:
                    fallback_overlong, _ = validation_tracker.overlong_snapshot(
                        current_segments
                    )
                    _record_repair_after(
                        list(fallback_overlong),
                        stagnant=True,
                        error=repair_error,
                    )
                if task_id and isinstance(task_state, dict):
                    task_state = legacy._persist_task_state_fields(
                        cfg,
                        task_id,
                        task_state,
                        last_error=str(repair_exc),
                    )
                _journal_repair_event(
                    "repair_round_failed",
                    reason="policy_overlong",
                    repair_round=round_no,
                    payload={"error": repair_error},
                    segments_snapshot=current_segments,
                )
                result["segments"] = current_segments
                result["prompt"] = current_prompt
                result["labels_payload"] = current_payload
                result["segment_plan"] = current_plan
                result["validation_report"] = {
                    "errors": [repair_error],
                    "warnings": current_report.get("warnings", []),
                }
                result["task_state"] = task_state
                result["repair_rounds"] = round_no
                result["skip_compare"] = True
                result["retry_stage"] = retry_stage
                result["retry_reason"] = "policy_overlong"
                return result

            continue

        retry_stage = "targeted_repair_1" if round_no == 1 else "targeted_repair_2"
        split_ops = [
            {"action": "split", "segment_index": int(idx)}
            for idx in overlong_indices[:max_splits_per_round]
        ]
        print(
            "[policy] auto-repairing overlong segments via split: "
            f"round {round_no}/{max_rounds} targets="
            + ", ".join(str(op["segment_index"]) for op in split_ops)
        )
        _record_repair_before(
            round_no,
            list(overlong_indices),
            split_ops,
        )
        op_result = legacy.apply_segment_operations(
            page,
            cfg,
            split_ops,
            source_segments=current_segments,
            heartbeat=_heartbeat,
        )
        _heartbeat()
        print(
            "[policy] overlong split repair applied: "
            f"{op_result['applied']} (structural={op_result['structural_applied']})"
        )
        if op_result["failed"]:
            print("[policy] overlong split repair failures:")
            for item in op_result["failed"]:
                print(f"  - {item}")
        if op_result["structural_applied"] <= 0:
            _record_repair_after(
                list(overlong_indices),
                split_ops_applied=int(op_result.get("structural_applied", 0) or 0),
                split_ops_failed=len(op_result.get("failed", []) or []),
                stagnant=True,
            )
            break

        current_segments = legacy.extract_segments(page, cfg)
        _heartbeat()
        print(
            f"[atlas] extracted {len(current_segments)} segments (post-overlong repair)"
        )
        if task_id and resume_from_artifacts:
            legacy._save_cached_segments(cfg, task_id, current_segments)

        current_prompt = legacy.build_prompt(
            current_segments,
            str(_cfg_get(cfg, "gemini.extra_instructions", "")),
            allow_operations=False,
            policy_trigger="policy_conflict",
        )
        try:
            current_payload = legacy._request_labels_with_optional_segment_chunking(
                cfg,
                current_segments,
                current_prompt,
                effective_repair_video,
                allow_operations=False,
                task_id=task_id,
                task_state=task_state,
                stage_name="repair",
            )
            _update_retry_metadata(
                task_state=task_state,
                labels_payload=current_payload,
                retry_stage=retry_stage,
                retry_reason="policy_overlong",
            )
        except Exception as repair_exc:
            repair_error = (
                "Policy auto-repair re-query failed after structural changes: "
                f"{str(repair_exc).strip() or repair_exc.__class__.__name__}"
            )
            print(f"[policy] {repair_error}")
            if validation_tracker is not None:
                fallback_overlong, _ = validation_tracker.overlong_snapshot(
                    current_segments
                )
                _record_repair_after(
                    list(fallback_overlong),
                    split_ops_applied=int(op_result.get("structural_applied", 0) or 0),
                    split_ops_failed=len(op_result.get("failed", []) or []),
                    stagnant=True,
                    error=repair_error,
                )
            if task_id and isinstance(task_state, dict):
                task_state = legacy._persist_task_state_fields(
                    cfg,
                    task_id,
                    task_state,
                    last_error=str(repair_exc),
                )
            result["segments"] = current_segments
            result["prompt"] = current_prompt
            result["labels_payload"] = current_payload
            result["segment_plan"] = current_plan
            result["validation_report"] = {
                "errors": [repair_error],
                "warnings": [
                    str(item).strip()
                    for item in (current_report.get("warnings", []) or [])
                    if str(item).strip()
                ],
            }
            result["task_state"] = task_state
            result["repair_rounds"] = round_no
            result["skip_compare"] = True
            result["retry_stage"] = retry_stage
            result["retry_reason"] = "policy_overlong"
            return result
        _heartbeat()
        if execute and execute_require_video_context:
            repair_meta = (
                current_payload.get("_meta", {})
                if isinstance(current_payload, dict)
                else {}
            )
            repair_video_attached = bool(repair_meta.get("video_attached", False))
            repair_mode = str(repair_meta.get("mode", "unknown"))
            if gemini_uploaded_file_names is not None:
                for fname in repair_meta.get("uploaded_file_names", []):
                    if fname not in gemini_uploaded_file_names:
                        gemini_uploaded_file_names.append(fname)
            if not repair_video_attached:
                raise RuntimeError(
                    "Execute blocked: overlong split repair re-query returned text-only "
                    "(no video context)."
                )
            print(
                f"[gemini] overlong repair re-query has video context ({repair_mode})."
            )

        post_ops = legacy._normalize_operations(current_payload, cfg=cfg)
        if post_ops:
            print("[policy] ignoring operations in overlong repair labels-only pass.")
        if task_id:
            legacy._save_cached_labels(cfg, task_id, current_payload)
        if task_id and isinstance(task_state, dict):
            task_state = legacy._persist_task_state_fields(
                cfg,
                task_id,
                task_state,
                labels_ready=True,
                last_error="",
                **legacy._episode_model_state_updates(cfg, current_payload, task_state),
            )

        legacy._save_outputs(
            cfg, current_segments, current_prompt, current_payload, task_id=task_id
        )
        current_plan = legacy._normalize_segment_plan(
            current_payload, current_segments, cfg=cfg
        )
        no_action_rewrites = legacy._rewrite_no_action_pauses_in_plan(current_plan, cfg)
        if no_action_rewrites:
            print(
                "[policy] overlong repair rewrote short no-action pauses: "
                f"{no_action_rewrites}"
            )
        if task_id:
            legacy._save_task_text_files(cfg, task_id, current_segments, current_plan)
        current_report = legacy._validate_segment_plan_against_policy(
            cfg, current_segments, current_plan
        )
        _heartbeat()
        remaining = _overlong_segment_indices_from_validation_report(current_report)
        print(
            "[policy] overlong split repair round complete: "
            f"remaining_overlong={len(remaining)}"
        )
        result["repair_rounds"] = round_no
        result["retry_stage"] = retry_stage
        result["retry_reason"] = "policy_overlong"
        _record_repair_after(
            list(remaining),
            split_ops_applied=int(op_result.get("structural_applied", 0) or 0),
            split_ops_failed=len(op_result.get("failed", []) or []),
            stagnant=bool(remaining),
        )
        if not remaining:
            break

    remaining_after_repairs = _overlong_segment_indices_from_validation_report(
        current_report
    )
    if remaining_after_repairs:
        repair_error = (
            "Policy targeted repair exhausted without resolving all overlong segments; "
            "full reset regenerate is disabled in production. Failing closed."
        )
        print(f"[policy] {repair_error}")
        _capture_step("targeted_repair_exhausted", include_html=True)
        _journal_repair_event(
            "repair_exhausted",
            reason="policy_overlong",
            repair_round=max_rounds,
            payload={"error": repair_error, "remaining_overlong": list(remaining_after_repairs)},
            segments_snapshot=current_segments,
        )
        _record_repair_after(
            list(remaining_after_repairs),
            stagnant=True,
            error=repair_error,
        )
        if isinstance(current_payload, dict):
            payload_meta = current_payload.setdefault("_meta", {})
            if isinstance(payload_meta, dict):
                payload_meta["retry_stage"] = "targeted_repair_exhausted"
                payload_meta["retry_reason"] = "policy_overlong"
                payload_meta["repair_fail_closed"] = True
        if task_id and isinstance(task_state, dict):
            task_state = legacy._persist_task_state_fields(
                cfg,
                task_id,
                task_state,
                gemini_last_retry_stage="targeted_repair_exhausted",
                gemini_last_retry_reason="policy_overlong",
                last_error=repair_error,
            )
        existing_errors = [
            str(item).strip()
            for item in (current_report.get("errors", []) or [])
            if str(item).strip()
        ]
        if repair_error not in existing_errors:
            existing_errors.append(repair_error)
        current_report = {
            "errors": existing_errors,
            "warnings": [
                str(item).strip()
                for item in (current_report.get("warnings", []) or [])
                if str(item).strip()
            ],
        }
        result["segments"] = current_segments
        result["prompt"] = current_prompt
        result["labels_payload"] = current_payload
        result["segment_plan"] = current_plan
        result["validation_report"] = current_report
        result["task_state"] = task_state
        result["repair_rounds"] = max(result["repair_rounds"], max_rounds)
        result["retry_reason"] = "policy_overlong"
        result["retry_stage"] = "targeted_repair_exhausted"
        result["skip_compare"] = True
        return result

    result["segments"] = current_segments
    result["prompt"] = current_prompt
    result["labels_payload"] = current_payload
    result["segment_plan"] = current_plan
    result["validation_report"] = current_report
    result["task_state"] = task_state
    return result


def _expert_consultation_repair(
    cfg: Dict[str, Any],
    page: Any,
    apply_result: Dict[str, Any],
    source_segments: List[Dict[str, Any]],
    current_payload: Dict[str, Any],
    *,
    task_id: str = "",
    task_state: Optional[Dict[str, Any]] = None,
    heartbeat: Any = None,
    consultation_round: int = 1,
) -> Dict[str, Any]:
    """Consult Gemini for expert advice on a failed label application turn."""
    legacy = import_module("src.solver.legacy_impl")
    chat_only = import_module("src.solver.chat_only")

    def _heartbeat() -> None:
        if callable(heartbeat):
            try:
                heartbeat()
            except Exception:
                pass

    total = apply_result.get("total_targets", 0)
    applied = apply_result.get("applied", 0)
    failed_details = apply_result.get("failed_details", {})
    reasons = "; ".join(str(r) for r in apply_result.get("submit_guard_reasons", []))

    print(
        f"[expert] initiating consultation round {consultation_round} for {task_id or 'episode'}"
    )
    _heartbeat()  # Expert Mode Keep-Alive

    # Construct detailed failure report
    failure_report = "\n".join(
        [f"- Segment {idx}: {err}" for idx, err in failed_details.items()]
    )

    # Get short log context or page status
    status_msg = "The 'Complete' button might be disabled or the page is unresponsive."
    if "budget exceeded" in reasons.lower():
        status_msg = "The application timed out due to slow browser response or high segment count."

    expert_prompt = (
        "EXPERT CONSULTATION - RECOVERY MODE\n"
        "Goal: Achieve 100% quality and successful submission.\n\n"
        f"Current System Status: {status_msg}\n"
        f"Progress: {applied}/{total} segments successfully labeled.\n"
        f"Failures Identified:\n{failure_report}\n\n"
        "INSTRUCTIONS:\n"
        "You MUST respond with a JSON object ONLY. No conversational text.\n"
        "Choose ONE decision from: 'retry', 'force_submit', 'page_reload', 'adjust'.\n\n"
        "JSON Schema Example:\n"
        "{\n"
        '  "decision": "retry",\n'
        '  "updated_payload": { "segments": [...] },\n'
        '  "reasoning": "Brief explanation"\n'
        "}\n"
    )

    repair_stage_model = str(
        resolve_stage_model(
            cfg, "repair", _cfg_get(cfg, "gemini.model", "gemini-3.1-pro-preview")
        )
    ).strip()

    try:
        cfg_dir = Path(_cfg_get(cfg, "_meta.config_dir", ".")).resolve()
        cache_dir = cfg_dir / ".state" / "cache" / "labels" / str(task_id or "unknown")
        episode_id = str(task_id).replace("episode_", "") if task_id else ""

        chat_res = chat_only.run_repair_query(
            cfg=cfg,
            source_segments=source_segments,
            prompt_text=expert_prompt,
            cache_dir=cache_dir,
            episode_id=episode_id,
            model=repair_stage_model,
            video_file=getattr(cfg, "_last_video_file", None),
            heartbeat=_heartbeat,
        )

        # Decision logic based on Gemini's response (Robust Regex Parsing)
        decision = "retry"
        text_res = str(chat_res.get("_meta", {}).get("raw_response", "")).upper()

        # Save Audit Trail (expert_decision.txt)
        try:
            audit_path = cache_dir / f"expert_decision_round_{consultation_round}.txt"
            audit_path.write_text(
                f"QUERY:\n{expert_prompt}\n\nGEMINI RESPONSE:\n{text_res}\n",
                encoding="utf-8",
            )
            print(f"[expert] audit trail saved: {audit_path.name}")
        except Exception as audit_err:
            print(f"[expert] failed to save audit trail: {audit_err}")

        if re.search(r"\bFORCE_SUBMIT\b", text_res):
            print("[expert] Gemini decision: FORCE_SUBMIT (bypassing guards)")
            decision = "force_submit"
        elif re.search(r"\bPAGE_RELOAD\b", text_res):
            print("[expert] Gemini decision: PAGE_RELOAD (refreshing page)")
            decision = "page_reload"
        elif re.search(r"\bADJUST_TIMESTAMPS\b", text_res):
            print("[expert] Gemini decision: ADJUST_TIMESTAMPS")
            decision = "adjust"
        elif re.search(r"\bRETRY_MISSING\b", text_res):
            print("[expert] Gemini decision: RETRY_MISSING")
            decision = "retry"

        return {
            "decision": decision,
            "updated_payload": chat_res,
            "raw_response": text_res,
            "success": True,
        }

    except Exception as exc:
        print(f"[expert] consultation failed: {exc}")
        return {"success": False, "error": str(exc), "decision": "abort"}


def _process_policy_gate_and_compare(
    cfg: Dict[str, Any],
    page: Any,
    segments: List[Dict[str, Any]],
    prompt: str,
    video_file: Optional[Path],
    labels_payload: Dict[str, Any],
    segment_plan: Dict[int, Dict[str, Any]],
    *,
    episode_no: int = 0,
    task_id: str = "",
    execute: bool = False,
    execute_require_video_context: bool = False,
    gemini_uploaded_file_names: Optional[List[str]] = None,
    resume_from_artifacts: bool = False,
    task_state: Optional[Dict[str, Any]] = None,
    heartbeat: Any = None,
    enable_structural_actions: bool = True,
    requery_after_structural_actions: bool = True,
    validation_tracker: Optional[ValidationTracker] = None,
) -> Dict[str, Any]:
    legacy = import_module("src.solver.legacy_impl")

    # ── Desync detection: compare DOM vs Gemini plan durations ──
    source_by_idx = {
        int(s.get("segment_index", 0)): s
        for s in segments
        if int(s.get("segment_index", 0)) > 0
    }
    desync_count = 0
    for idx, plan_item in segment_plan.items():
        src = source_by_idx.get(idx)
        if src is None:
            continue
        from src.infra.utils import _safe_float

        plan_dur = _safe_float(plan_item.get("end_sec"), 0) - _safe_float(
            plan_item.get("start_sec"), 0
        )
        dom_dur = _safe_float(src.get("end_sec"), 0) - _safe_float(
            src.get("start_sec"), 0
        )
        if abs(plan_dur - dom_dur) > 1.0:
            desync_count += 1
            if desync_count <= 5:
                print(
                    f"[desync] segment {idx}: Gemini={plan_dur:.1f}s vs DOM={dom_dur:.1f}s "
                    f"(delta={plan_dur - dom_dur:+.1f}s)"
                )
    if desync_count > 0:
        print(
            f"[desync] TOTAL: {desync_count} segment(s) have duration mismatch (Gemini vs DOM)"
        )
    else:
        print(
            f"[trace] no desync detected: {len(segment_plan)} segments match DOM durations"
        )

    validation_report = legacy._validate_segment_plan_against_policy(
        cfg, segments, segment_plan
    )
    report_task_id = task_id or f"episode_{episode_no}"

    retry_result = _maybe_retry_policy_with_stronger_model(
        cfg=cfg,
        segments=segments,
        prompt=prompt,
        video_file=video_file,
        labels_payload=labels_payload,
        segment_plan=segment_plan,
        validation_report=validation_report,
        task_id=task_id,
        execute=execute,
        execute_require_video_context=execute_require_video_context,
        gemini_uploaded_file_names=gemini_uploaded_file_names,
        resume_from_artifacts=resume_from_artifacts,
        task_state=task_state,
        heartbeat=heartbeat,
    )
    labels_payload = retry_result["labels_payload"]
    segment_plan = retry_result["segment_plan"]
    validation_report = retry_result["validation_report"]
    if isinstance(retry_result.get("task_state"), dict):
        task_state = retry_result["task_state"]

    overlong_repair_result = _maybe_repair_overlong_segments(
        cfg=cfg,
        page=page,
        segments=segments,
        prompt=prompt,
        video_file=video_file,
        labels_payload=labels_payload,
        segment_plan=segment_plan,
        validation_report=validation_report,
        task_id=task_id,
        execute=execute,
        execute_require_video_context=execute_require_video_context,
        gemini_uploaded_file_names=gemini_uploaded_file_names,
        resume_from_artifacts=resume_from_artifacts,
        task_state=task_state,
        enable_structural_actions=enable_structural_actions,
        requery_after_structural_actions=requery_after_structural_actions,
        heartbeat=heartbeat,
        validation_tracker=validation_tracker,
    )
    segments = overlong_repair_result["segments"]
    prompt = overlong_repair_result["prompt"]
    labels_payload = overlong_repair_result["labels_payload"]
    segment_plan = overlong_repair_result["segment_plan"]
    validation_report = overlong_repair_result["validation_report"]
    if isinstance(overlong_repair_result.get("task_state"), dict):
        task_state = overlong_repair_result["task_state"]

    report_path = legacy._save_validation_report(cfg, report_task_id, validation_report)
    if report_path is not None:
        print(f"[out] validation: {report_path}")

    if bool(overlong_repair_result.get("skip_compare")):
        warnings = [
            str(w).strip()
            for w in validation_report.get("warnings", [])
            if str(w).strip()
        ]
        errors = [
            str(e).strip()
            for e in validation_report.get("errors", [])
            if str(e).strip()
        ]
        if task_id and isinstance(task_state, dict):
            task_state = legacy._persist_task_state_fields(
                cfg,
                task_id,
                task_state,
                validation_ok=len(errors) == 0,
                last_error="" if not errors else task_state.get("last_error", ""),
            )
        return {
            "segments": segments,
            "prompt": prompt,
            "labels_payload": labels_payload,
            "segment_plan": segment_plan,
            "validation_report": validation_report,
            "warnings": warnings,
            "errors": errors,
            "task_state": task_state,
            "report_task_id": report_task_id,
            "compare_result": {"decision": "skip_overlong_repair_failure"},
            "repair_skipped_reason": str(
                overlong_repair_result.get("repair_skipped_reason", "") or ""
            ),
        }

    chat_only_mode = bool(_cfg_get(cfg, "run.chat_only_mode", False))
    if chat_only_mode:
        warnings = [
            str(w).strip()
            for w in validation_report.get("warnings", [])
            if str(w).strip()
        ]
        errors = [
            str(e).strip()
            for e in validation_report.get("errors", [])
            if str(e).strip()
        ]
        if task_id and isinstance(task_state, dict):
            task_state = legacy._persist_task_state_fields(
                cfg,
                task_id,
                task_state,
                validation_ok=len(errors) == 0,
                chat_compare_skipped=True,
                last_error="" if not errors else task_state.get("last_error", ""),
            )
        print("[compare] pre-submit compare skipped: chat_only_mode")
        return {
            "segments": segments,
            "prompt": prompt,
            "labels_payload": labels_payload,
            "segment_plan": segment_plan,
            "validation_report": validation_report,
            "warnings": warnings,
            "errors": errors,
            "task_state": task_state,
            "report_task_id": report_task_id,
            "compare_result": {"decision": "skip_chat_only_mode"},
            "repair_skipped_reason": str(
                overlong_repair_result.get("repair_skipped_reason", "") or ""
            ),
        }

    current_model = str(
        (labels_payload.get("_meta", {}) or {}).get(
            "model",
            _cfg_get(cfg, "gemini.model", "gemini-3.1-pro-preview"),
        )
    ).strip()
    legacy._respect_major_step_pause(cfg, "pre_submit_compare", heartbeat=heartbeat)
    compare_result = legacy._maybe_run_pre_submit_chat_compare(
        cfg=cfg,
        source_segments=segments,
        api_segment_plan=segment_plan,
        task_id=report_task_id,
        video_file=video_file,
        api_model=current_model,
        episode_active_model=str(
            (
                task_state.get("episode_active_model", "")
                if isinstance(task_state, dict)
                else ""
            )
            or current_model
        ).strip(),
        task_state=task_state,
        heartbeat=heartbeat,
    )
    if compare_result.get("executed"):
        print(
            "[compare] pre-submit compare: "
            f"winner={compare_result.get('winner', 'api')} "
            f"decision={compare_result.get('decision', 'keep_api')}"
        )
        report_json = str(compare_result.get("json_report_path", "") or "").strip()
        if report_json:
            print(f"[compare] report: {report_json}")
    elif compare_result.get("decision"):
        print(f"[compare] pre-submit compare skipped: {compare_result.get('decision')}")

    if compare_result.get("adopted"):
        adopted_plan = compare_result.get("selected_plan")
        adopted_validation = compare_result.get("selected_validation_report")
        adopted_payload = compare_result.get("selected_payload")
        if isinstance(adopted_plan, dict) and isinstance(adopted_validation, dict):
            segment_plan = adopted_plan
            validation_report = adopted_validation
            if isinstance(adopted_payload, dict):
                labels_payload = adopted_payload
            print("[compare] adopted Chat UI candidate before apply.")
            if task_id:
                legacy._save_task_text_files(cfg, task_id, segments, segment_plan)
    elif compare_result.get("selected_operations"):
        selected_operations = compare_result.get("selected_operations")
        if (
            execute
            and enable_structural_actions
            and isinstance(selected_operations, list)
        ):
            if callable(heartbeat):
                try:
                    heartbeat()
                except Exception:
                    pass
            print(
                "[compare] applying Chat UI split repair operations before apply: "
                f"{len(selected_operations)}"
            )
            op_result = legacy.apply_segment_operations(
                page, cfg, selected_operations, heartbeat=heartbeat
            )
            print(
                f"[compare] split repair applied: {op_result['applied']} "
                f"(structural={op_result['structural_applied']})"
            )
            if op_result["failed"]:
                print("[compare] split repair operation failures:")
                for item in op_result["failed"]:
                    print(f"  - {item}")

            compare_repair_error = ""
            if op_result["structural_applied"] <= 0:
                compare_repair_error = (
                    "Pre-submit compare proposed split repair but no structural changes were "
                    "applied in Atlas; blocking submit."
                )
            elif not requery_after_structural_actions:
                compare_repair_error = (
                    "Pre-submit compare applied split repair but requery_after_structural_actions "
                    "is disabled; refusing to continue without a fresh label pass."
                )
            else:
                try:
                    segments = legacy.extract_segments(page, cfg)
                    print(
                        f"[atlas] extracted {len(segments)} segments (post-compare repair)"
                    )
                    if task_id and resume_from_artifacts:
                        legacy._save_cached_segments(cfg, task_id, segments)
                    prompt = legacy.build_prompt(
                        segments,
                        str(_cfg_get(cfg, "gemini.extra_instructions", "")),
                        allow_operations=False,
                        policy_trigger="policy_conflict",
                    )
                    # Use optimized upload video for post-compare re-query
                    # to stay under CDP 50 MB transfer limit.
                    _compare_repair_video = video_file
                    if video_file is not None:
                        _opt = video_file.with_name(
                            video_file.stem + "_upload_opt.mp4"
                        )
                        if _opt.exists() and _opt.stat().st_size > 0:
                            _compare_repair_video = _opt
                    labels_payload = (
                        legacy._request_labels_with_optional_segment_chunking(
                            cfg,
                            segments,
                            prompt,
                            _compare_repair_video,
                            allow_operations=False,
                            task_id=task_id,
                            task_state=task_state,
                        )
                    )
                    if execute and execute_require_video_context:
                        repair_meta = (
                            labels_payload.get("_meta", {})
                            if isinstance(labels_payload, dict)
                            else {}
                        )
                        repair_video_attached = bool(
                            repair_meta.get("video_attached", False)
                        )
                        repair_mode = str(repair_meta.get("mode", "unknown"))

                        if gemini_uploaded_file_names is not None:
                            for fname in repair_meta.get("uploaded_file_names", []):
                                if fname not in gemini_uploaded_file_names:
                                    gemini_uploaded_file_names.append(fname)

                        if not repair_video_attached:
                            raise RuntimeError(
                                "Execute blocked: Chat-guided split repair re-query returned text-only "
                                "(no video context)."
                            )
                        print(
                            "[gemini] compare repair re-query has video context "
                            f"({repair_mode})."
                        )
                    post_ops = legacy._normalize_operations(labels_payload, cfg=cfg)
                    if post_ops:
                        print(
                            "[compare] ignoring operations in post-repair labels-only pass."
                        )
                    if task_id:
                        legacy._save_cached_labels(cfg, task_id, labels_payload)
                    if task_id and isinstance(task_state, dict):
                        task_state = legacy._persist_task_state_fields(
                            cfg,
                            task_id,
                            task_state,
                            labels_ready=True,
                            last_error="",
                            **legacy._episode_model_state_updates(
                                cfg, labels_payload, task_state
                            ),
                        )

                    legacy._save_outputs(
                        cfg, segments, prompt, labels_payload, task_id=task_id
                    )
                    segment_plan = legacy._normalize_segment_plan(
                        labels_payload, segments, cfg=cfg
                    )
                    no_action_rewrites = legacy._rewrite_no_action_pauses_in_plan(
                        segment_plan, cfg
                    )
                    if no_action_rewrites:
                        print(
                            "[policy] compare repair rewrote short no-action pauses: "
                            f"{no_action_rewrites}"
                        )
                    if task_id:
                        legacy._save_task_text_files(
                            cfg, task_id, segments, segment_plan
                        )

                    validation_report = legacy._validate_segment_plan_against_policy(
                        cfg, segments, segment_plan
                    )
                    report_path = legacy._save_validation_report(
                        cfg, report_task_id, validation_report
                    )
                    if report_path is not None:
                        print(f"[out] validation: {report_path}")
                    print("[compare] applied Chat UI split repair before apply.")
                except Exception as compare_repair_exc:
                    compare_repair_error = (
                        "Pre-submit compare split repair failed during re-query: "
                        f"{compare_repair_exc}"
                    )
                    if task_id and isinstance(task_state, dict):
                        task_state = legacy._persist_task_state_fields(
                            cfg,
                            task_id,
                            task_state,
                            last_error=str(compare_repair_exc),
                        )

            if compare_repair_error:
                validation_errors = list(validation_report.get("errors", []) or [])
                validation_errors.append(compare_repair_error)
                validation_report = dict(validation_report)
                validation_report["errors"] = validation_errors
        else:
            print(
                "[compare] split repair available but skipped "
                "(dry-run or structural actions disabled)."
            )
    elif compare_result.get("block_apply"):
        compare_reason = str(compare_result.get("block_reason", "")).strip()
        if compare_reason:
            validation_errors = list(validation_report.get("errors", []) or [])
            validation_errors.append(compare_reason)
            validation_report = dict(validation_report)
            validation_report["errors"] = validation_errors

    warnings = [
        str(w).strip() for w in validation_report.get("warnings", []) if str(w).strip()
    ]
    errors = [
        str(e).strip() for e in validation_report.get("errors", []) if str(e).strip()
    ]
    ignored_ts_errors: List[str] = []
    if not bool(_cfg_get(cfg, "run.adjust_timestamps", True)) and bool(
        _cfg_get(cfg, "run.ignore_timestamp_policy_errors_when_adjust_disabled", True)
    ):
        ignored_ts_errors = [e for e in errors if legacy._is_timestamp_policy_error(e)]
        if ignored_ts_errors:
            errors = [e for e in errors if not legacy._is_timestamp_policy_error(e)]
            print(
                f"[policy] ignored timestamp errors: {len(ignored_ts_errors)} "
                "(adjust_timestamps=false)"
            )
            for item in ignored_ts_errors[:10]:
                print(f"  - {item}")
    ignored_no_action_errors: List[str] = []
    if bool(_cfg_get(cfg, "run.ignore_no_action_standalone_policy_error", True)):
        ignored_no_action_errors = [
            e for e in errors if legacy._is_no_action_policy_error(e)
        ]
        if ignored_no_action_errors:
            errors = [e for e in errors if not legacy._is_no_action_policy_error(e)]
            print(
                f"[policy] ignored no-action standalone errors: "
                f"{len(ignored_no_action_errors)}"
            )
            for item in ignored_no_action_errors[:10]:
                print(f"  - {item}")
    if warnings:
        print(f"[policy] warnings: {len(warnings)}")
        for item in warnings[:10]:
            print(f"  - {item}")
    if task_id and isinstance(task_state, dict):
        task_state = legacy._persist_task_state_fields(
            cfg,
            task_id,
            task_state,
            validation_ok=len(errors) == 0,
            last_error="" if not errors else task_state.get("last_error", ""),
        )

    return {
        "segments": segments,
        "prompt": prompt,
        "labels_payload": labels_payload,
        "segment_plan": segment_plan,
        "validation_report": validation_report,
        "warnings": warnings,
        "errors": errors,
        "task_state": task_state,
        "report_task_id": report_task_id,
        "compare_result": compare_result,
        "repair_skipped_reason": str(
            overlong_repair_result.get("repair_skipped_reason", "") or ""
        ),
    }


def run(cfg: Dict[str, Any], execute: bool) -> None:
    legacy = import_module("src.solver.legacy_impl")
    legacy.run(cfg, execute)


__all__ = [
    "_maybe_retry_policy_with_stronger_model",
    "_process_policy_gate_and_compare",
    "run",
]
