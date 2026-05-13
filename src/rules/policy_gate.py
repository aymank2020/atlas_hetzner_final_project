"""Policy-gate helpers extracted from the legacy solver."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.infra.solver_config import _cfg_get
from src.infra.utils import _normalize_label_for_compare, _safe_float
from src.rules.labels import (
    _DISALLOWED_TOOL_TERMS,
    _allowed_label_start_verb_token_patterns_from_cfg,
    _count_atomic_actions_in_label,
    _infer_held_object_context,
    _label_action_clauses,
    _label_main_verb,
    _label_starts_with_allowed_action_verb,
)

_logger = logging.getLogger(__name__)


def _validate_segment_plan_against_policy(
    cfg: Dict[str, Any],
    source_segments: List[Dict[str, Any]],
    segment_plan: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    min_words = max(1, int(_cfg_get(cfg, "run.min_label_words", 2)))
    max_words = max(min_words, int(_cfg_get(cfg, "run.max_label_words", 20)))
    max_atomic_actions = max(1, int(_cfg_get(cfg, "run.max_atomic_actions_per_label", 2)))
    max_segment_duration_sec = max(0.1, float(_cfg_get(cfg, "run.max_segment_duration_sec", 10.0)))
    forbidden_verbs_raw = _cfg_get(cfg, "run.forbidden_label_verbs", [])
    forbidden_verbs = [str(item).strip().lower() for item in forbidden_verbs_raw if str(item).strip()]
    allowed_verb_token_patterns = _allowed_label_start_verb_token_patterns_from_cfg(cfg)
    forbidden_narrative_raw = _cfg_get(cfg, "run.forbidden_narrative_words", [])
    forbidden_narrative_words = [str(item).strip().lower() for item in forbidden_narrative_raw if str(item).strip()]
    skip_unchanged_lexical = bool(_cfg_get(cfg, "run.skip_policy_lexical_checks_on_unchanged_labels", False))
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
    guide_as_object_descriptor_pattern = re.compile(
        r"\b(?:pick up|place|align|adjust|pull|move|position|set|smooth|hold)\s+guide\s+"
        r"(?:fabric|cloth|garment|material|item|piece)\b",
        re.IGNORECASE,
    )
    invalid_adjust_over_pattern = re.compile(r"\badjust\s+over\b", re.IGNORECASE)

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
        label_unchanged_from_source = bool(source_label) and _normalize_label_for_compare(source_label) == _normalize_label_for_compare(label)

        if not label:
            errors.append(f"segment {idx}: empty label")
        else:
            words = [word for word in re.split(r"\s+", label) if word]
            if not (label_unchanged_from_source and skip_unchanged_lexical):
                if label != "No Action":
                    if len(words) < min_words:
                        errors.append(f"segment {idx}: label has fewer than {min_words} words")
                    if len(words) > max_words:
                        errors.append(f"segment {idx}: label has more than {max_words} words")
                    if not _label_starts_with_allowed_action_verb(label, allowed_verb_token_patterns):
                        errors.append(f"segment {idx}: label must start with an allowed action verb")
                    if any(
                        not _label_starts_with_allowed_action_verb(clause, allowed_verb_token_patterns)
                        for clause in _label_action_clauses(label)
                    ):
                        errors.append(f"segment {idx}: each action clause must start with an allowed action verb")
                    for verb in forbidden_verbs:
                        if re.search(rf"\b{re.escape(verb)}\b", label_l):
                            errors.append(f"segment {idx}: forbidden verb '{verb}' found")
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
                    if guide_as_object_descriptor_pattern.search(label):
                        errors.append(f"segment {idx}: 'guide' should not be used as an object descriptor")
                    if invalid_adjust_over_pattern.search(label):
                        errors.append(f"segment {idx}: 'adjust over' phrasing is not semantically valid")
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
                        errors.append(f"segment {idx}: label has more than {max_atomic_actions} atomic actions")
                    clauses = _label_action_clauses(label)
                    hold_positions = [
                        pos
                        for pos, clause in enumerate(clauses)
                        if _label_main_verb(clause) == "hold"
                    ]
                    if hold_positions and hold_positions[0] > 0:
                        errors.append(f"segment {idx}: 'hold' must appear before the other action")
                    held_context = _infer_held_object_context(
                        idx,
                        source_segments,
                        normalized_plan=segment_plan,
                        cfg=cfg,
                    )
                    if (
                        held_context
                        and not hold_positions
                        and action_count < max_atomic_actions
                        and held_context.lower() not in label_l
                    ):
                        errors.append(
                            f"segment {idx}: bilateral continuity context requires leading hold clause"
                        )
                elif "," in label or " and " in label_l:
                    errors.append(f"segment {idx}: 'No Action' must be standalone")

        # Duration and timestamp validation.
        if start < 0 or end < 0:
            errors.append(f"segment {idx}: invalid timestamp values")
        elif end <= start:
            errors.append(f"segment {idx}: end_sec must be greater than start_sec")
        else:
            plan_duration = end - start
            if source is not None:
                src_start = _safe_float(source.get("start_sec"), start)
                src_end = _safe_float(source.get("end_sec"), end)
                dom_duration = src_end - src_start if src_end > src_start else plan_duration

                if plan_duration > max_segment_duration_sec + 0.05:
                    errors.append(
                        f"segment {idx}: duration {plan_duration:.1f}s exceeds max "
                        f"{max_segment_duration_sec:.1f}s (MANDATORY SPLIT REQUIRED)"
                    )

                if (
                    plan_duration > max_segment_duration_sec + 0.05
                    and dom_duration <= max_segment_duration_sec + 0.05
                ):
                    warnings.append(
                        f"segment {idx}: DESYNC DETECTED - Gemini duration={plan_duration:.1f}s "
                        f"but DOM duration={dom_duration:.1f}s (using DOM as ground truth)"
                    )
                    _logger.warning(
                        "[policy] DESYNC segment %d: plan=%.1fs vs DOM=%.1fs - "
                        "plan remains blocked despite DOM mismatch",
                        idx,
                        plan_duration,
                        dom_duration,
                    )
                elif dom_duration > max_segment_duration_sec + 0.05 and abs(plan_duration - dom_duration) > 1.0:
                    warnings.append(
                        f"segment {idx}: duration mismatch - "
                        f"plan={plan_duration:.1f}s vs DOM={dom_duration:.1f}s"
                    )
            else:
                if plan_duration > max_segment_duration_sec + 0.05:
                    errors.append(
                        f"segment {idx}: duration {plan_duration:.1f}s exceeds max "
                        f"{max_segment_duration_sec:.1f}s (MANDATORY SPLIT REQUIRED)"
                    )

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


def _save_validation_report(cfg: Dict[str, Any], task_id: str, report: Dict[str, Any]) -> Optional[Path]:
    out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"validation_{task_id}.json" if task_id else "validation_report.json"
    path = out_dir / filename
    try:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except Exception:
        return None


def _is_timestamp_policy_error(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    markers = (
        "invalid timestamp values",
        "end_sec must be greater than start_sec",
        "start_sec is not monotonic",
        "overlaps previous segment",
    )
    return bool(normalized) and any(token in normalized for token in markers)


def _is_no_action_policy_error(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    return bool(normalized) and "'no action' must be standalone" in normalized


__all__ = [
    "_validate_segment_plan_against_policy",
    "_save_validation_report",
    "_is_timestamp_policy_error",
    "_is_no_action_policy_error",
]
