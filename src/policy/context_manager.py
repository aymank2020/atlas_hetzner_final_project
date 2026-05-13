"""
Central Rule Authority (CRA) for Discord-driven policy propagation.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


POLICY_ROOT = Path("data/policy")
CURRENT_POLICY_PATH = POLICY_ROOT / "current_policy.json"
STAGED_RULES_PATH = POLICY_ROOT / "staged_rules.jsonl"
TRUST_CONFIG_PATH = POLICY_ROOT / "trust_config.json"
PROMPT_SUMMARY_PATH = POLICY_ROOT / "generated_prompt_summary.txt"
POLICY_DIFF_PATH = POLICY_ROOT / "policy_diff.json"
GENERATED_CONTEXT_PATH = Path("prompts/generated_policy_context.txt")

LESSONS_PATH = Path("data/knowledge/policy_lessons.jsonl")
GOLDEN_INDEX_PATH = Path("data/knowledge/golden_index.jsonl")

TRIGGERABLE_RETRIEVALS = {
    "low_confidence",
    "validator_error",
    "policy_conflict",
    "repair_needed",
}

DEFAULT_TRUST_CONFIG: Dict[str, Any] = {
    "trusted_authors": ["sentientcake", "durian0", "dana", "danatimmer", "atlascapture"],
    "trusted_channels": [
        "#atlas-rules",
        "atlas-rules",
        "#LEVEL3_ANNOUNCEMENT",
        "#ATLAS_ANNOUNCEMENT",
        "#LEVEL3_QUESTION",
        "LEVEL3_ANNOUNCEMENT",
        "ATLAS_ANNOUNCEMENT",
    ],
    "trusted_roles": ["manager", "trainer", "admin", "staff"],
    "auto_promote_scopes": ["engine", "validator", "global", "prompt_only"],
    "auto_promote_fields": [
        "engine_limits.max_segment_seconds",
        "annotation.preferred_density_sec",
        "annotation.max_atomic_actions",
        "annotation.max_label_words",
        "lexicon.forbidden_verbs",
        "lexicon.forbidden_narrative_words",
    ],
    "conflict_policy": "newest_exact_wins",
}

DEFAULT_POLICY_TEMPLATE: Dict[str, Any] = {
    "policy_version": "atlas-discord-cra-v1",
    "generated_at": "",
    "effective_at": "",
    "authority_mode": "trusted_auto",
    "engine_limits": {"max_segment_seconds": 10.0},
    "annotation": {
        "preferred_density_sec": {"min": 2.0, "max": 5.0},
        "max_atomic_actions": 2,
        "max_label_words": 20,
    },
    "merge_rules": {
        "tolerance_sec": 0.5,
        "max_combined_duration_seconds": 60.0,
    },
    "lexicon": {
        "forbidden_verbs": ["inspect", "check", "reach", "examine"],
        "forbidden_narrative_words": ["then", "another", "continue", "next", "again"],
        "disallowed_tool_terms": [
            "mechanical arm",
            "robotic arm",
            "robot arm",
            "manipulator",
            "robot gripper",
            "claw arm",
        ],
    },
    "behavior": {
        "no_action_standalone": True,
        "no_action_forbidden_while_holding": True,
        "merge_identical_consecutive": True,
        "hold_rule_mode": "task_relevant_only",
        "segment_style": "dense_first",
        "correct_existing_labels_preferred": True,
        "label_task_relevant_pauses_within_segment": True,
    },
    "runtime": {
        "universal_rules": [
            "Use the canonical policy values below over any stale prompt wording.",
            "Shorter segments are the current standard: aim for 2-5 seconds and never exceed 10 seconds.",
            "Prefer dense, granular corrections when the visible actions support it; dense is better than overly coarse rewrites.",
            "Treat current employee labels as drafts to correct, not text to discard by default.",
            "Within each kept segment, capture all task-relevant actions and pauses.",
            "No Action is standalone only and is forbidden while a task-relevant object is still held.",
            "Each label should keep at most 2 atomic actions and 20 words.",
        ]
    },
    "sources": [],
}

FIELD_TYPE_MAP: Dict[str, str] = {
    "engine_limits.max_segment_seconds": "number",
    "annotation.preferred_density_sec": "range",
    "annotation.max_atomic_actions": "integer",
    "annotation.max_label_words": "integer",
    "merge_rules.max_combined_duration_seconds": "number",
    "merge_rules.tolerance_sec": "number",
    "lexicon.forbidden_verbs": "list",
    "lexicon.forbidden_narrative_words": "list",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_policy() -> Dict[str, Any]:
    policy = copy.deepcopy(DEFAULT_POLICY_TEMPLATE)
    now = _utc_now()
    policy["generated_at"] = now
    policy["effective_at"] = now
    return policy


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return copy.deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return copy.deepcopy(default)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                clean = line.strip()
                if not clean:
                    continue
                try:
                    row = json.loads(clean)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        return []
    return rows


def _save_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _merge_defaults(payload: Any, defaults: Any) -> Any:
    if isinstance(defaults, dict):
        base = {} if not isinstance(payload, dict) else dict(payload)
        for key, default_value in defaults.items():
            base[key] = _merge_defaults(base.get(key), default_value)
        return base
    if isinstance(defaults, list):
        if isinstance(payload, list):
            return payload
        return copy.deepcopy(defaults)
    return payload if payload is not None else copy.deepcopy(defaults)


def ensure_policy_files(policy_root: Path | str = POLICY_ROOT) -> Dict[str, Path]:
    root = Path(policy_root)
    current_policy_path = root / CURRENT_POLICY_PATH.name
    staged_rules_path = root / STAGED_RULES_PATH.name
    trust_config_path = root / TRUST_CONFIG_PATH.name
    prompt_summary_path = root / PROMPT_SUMMARY_PATH.name
    policy_diff_path = root / POLICY_DIFF_PATH.name

    root.mkdir(parents=True, exist_ok=True)
    if not current_policy_path.exists():
        _json_dump(current_policy_path, _default_policy())
    if not trust_config_path.exists():
        _json_dump(trust_config_path, DEFAULT_TRUST_CONFIG)
    if not staged_rules_path.exists():
        staged_rules_path.write_text("", encoding="utf-8")
    if not prompt_summary_path.exists():
        prompt_summary_path.write_text("", encoding="utf-8")
    if not policy_diff_path.exists():
        _json_dump(
            policy_diff_path,
            {
                "policy_version": DEFAULT_POLICY_TEMPLATE["policy_version"],
                "generated_at": _utc_now(),
                "changed_fields": [],
                "promoted_rule_ids": [],
            },
        )
    return {
        "root": root,
        "current_policy": current_policy_path,
        "staged_rules": staged_rules_path,
        "trust_config": trust_config_path,
        "prompt_summary": prompt_summary_path,
        "policy_diff": policy_diff_path,
    }


def load_current_policy(current_policy_path: Path | str = CURRENT_POLICY_PATH) -> Dict[str, Any]:
    path = Path(current_policy_path)
    ensure_policy_files(path.parent)
    payload = _load_json(path, _default_policy())
    return _merge_defaults(payload, _default_policy())


def _load_trust_config(trust_config_path: Path | str = TRUST_CONFIG_PATH) -> Dict[str, Any]:
    path = Path(trust_config_path)
    ensure_policy_files(path.parent)
    payload = _load_json(path, DEFAULT_TRUST_CONFIG)
    return _merge_defaults(payload, DEFAULT_TRUST_CONFIG)


def get_policy(
    path: str,
    default: Any = None,
    *,
    policy: Optional[Dict[str, Any]] = None,
    current_policy_path: Path | str = CURRENT_POLICY_PATH,
) -> Any:
    node: Any = load_current_policy(current_policy_path) if policy is None else policy
    for part in (path or "").split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _set_policy_value(policy: Dict[str, Any], path: str, value: Any) -> None:
    parts = [part for part in path.split(".") if part]
    node = policy
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _field_scope(field_path: str) -> str:
    if field_path.startswith("engine_limits."):
        return "engine"
    if field_path.startswith("merge_rules."):
        return "validator"
    if field_path.startswith("annotation."):
        return "global"
    if field_path.startswith("lexicon."):
        return "global"
    return "prompt_only"


def _normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9_#-]+", "", str(value or "").strip().lower())


def _infer_trust_tier(
    author: str,
    channel: str,
    *,
    trust_config: Optional[Dict[str, Any]] = None,
) -> str:
    trust = DEFAULT_TRUST_CONFIG if trust_config is None else trust_config
    author_norm = _normalize_name(author)
    channel_norm = _normalize_name(channel)
    trusted_authors = {_normalize_name(item) for item in trust.get("trusted_authors", [])}
    trusted_channels = {_normalize_name(item) for item in trust.get("trusted_channels", [])}
    if author_norm in trusted_authors or channel_norm in trusted_channels:
        return "trusted_auto"
    return "community"


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _coerce_value(field_path: str, value: Any) -> Any:
    value_type = FIELD_TYPE_MAP.get(field_path)
    if value_type == "integer":
        coerced = _safe_int(value)
        return coerced if coerced is not None else value
    if value_type == "number":
        coerced = _safe_float(value)
        return coerced if coerced is not None else value
    return value


def _format_value_for_summary(value: Any) -> str:
    if isinstance(value, dict) and {"min", "max"}.issubset(set(value)):
        return f"{value['min']}-{value['max']}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _record_id(message_id: str, field_path: str, timestamp: str) -> str:
    base = "|".join([message_id or "unknown", field_path or "unknown", timestamp or "unknown"])
    return re.sub(r"[^a-z0-9_.:-]+", "-", base.lower())


def _segment_text(source_segments: Sequence[Any]) -> str:
    parts: List[str] = []
    for item in source_segments or []:
        if isinstance(item, dict):
            parts.extend(
                [
                    str(item.get("label", "")),
                    str(item.get("current_label", "")),
                    str(item.get("raw_text", "")),
                ]
            )
        else:
            parts.append(str(item))
    return " ".join(parts).strip()


def _extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    stop = {"the", "and", "for", "with", "this", "that", "into", "from", "only", "must"}
    return {word for word in words if len(word) > 2 and word not in stop}


def _score_overlap(text: str, keywords: set[str]) -> int:
    if not text or not keywords:
        return 0
    return len(_extract_keywords(text).intersection(keywords))


def _compatibility_error(field_path: str, value: Any, policy: Dict[str, Any]) -> str:
    if field_path == "engine_limits.max_segment_seconds":
        max_segment = _safe_float(value)
        if max_segment is None or max_segment <= 0 or max_segment > 300:
            return "max_segment_seconds_out_of_range"
    if field_path == "annotation.max_atomic_actions":
        atomic_actions = _safe_int(value)
        if atomic_actions is None or atomic_actions <= 0 or atomic_actions > 6:
            return "max_atomic_actions_out_of_range"
    if field_path == "annotation.max_label_words":
        max_words = _safe_int(value)
        if max_words is None or max_words <= 1 or max_words > 50:
            return "max_label_words_out_of_range"
    if field_path == "annotation.preferred_density_sec":
        if not isinstance(value, dict):
            return "preferred_density_not_a_range"
        minimum = _safe_float(value.get("min"))
        maximum = _safe_float(value.get("max"))
        if minimum is None or maximum is None or minimum <= 0 or maximum < minimum:
            return "preferred_density_invalid_range"
        max_segment = _safe_float(get_policy("engine_limits.max_segment_seconds", 10.0, policy=policy))
        if max_segment is not None and maximum > max_segment:
            return "preferred_density_exceeds_max_segment"
    return ""


def _exact_confidence_ok(record: Dict[str, Any]) -> bool:
    value = record.get("value")
    confidence = _safe_float(record.get("confidence"))
    return value is not None and confidence is not None and confidence >= 0.9


def _is_rule_active(record: Dict[str, Any], *, now: Optional[datetime] = None) -> bool:
    if str(record.get("status", "")).strip().lower() != "promoted":
        return False
    now_utc = now or datetime.now(timezone.utc)
    expires_at = _parse_iso_datetime(record.get("expires_at"))
    if expires_at is not None and expires_at <= now_utc:
        return False
    return True


def _extract_rules_from_text(raw_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    author = str(metadata.get("author", "")).strip()
    channel = str(metadata.get("channel", "")).strip()
    timestamp = str(metadata.get("timestamp", "")).strip() or _utc_now()
    message_id = str(metadata.get("message_id", "")).strip()
    trust_config = metadata.get("trust_config") or DEFAULT_TRUST_CONFIG
    trust_tier = str(metadata.get("trust_tier", "")).strip() or _infer_trust_tier(
        author,
        channel,
        trust_config=trust_config,
    )
    text = str(raw_text or "").strip()
    lowered = text.lower()
    candidates: List[Dict[str, Any]] = []

    def append_candidate(
        *,
        field_path: str,
        value: Any,
        confidence: float,
        summary: str,
    ) -> None:
        candidates.append(
            {
                "rule_id": _record_id(message_id, field_path, timestamp),
                "message_id": message_id,
                "timestamp": timestamp,
                "channel": channel,
                "author": author,
                "trust_tier": trust_tier,
                "scope": _field_scope(field_path),
                "field_path": field_path,
                "value": _coerce_value(field_path, value),
                "value_type": FIELD_TYPE_MAP.get(field_path, "string"),
                "confidence": confidence,
                "summary": summary,
                "raw_text": text,
                "status": "staged",
                "expires_at": metadata.get("expires_at"),
            }
        )

    duration_match = re.search(
        r"max\s+segment\s+duration\s+(?:is\s+now|is|must\s+be|=|:)\s*\**\s*(\d+(?:\.\d+)?)\s*seconds?",
        lowered,
    )
    if duration_match:
        value = _safe_float(duration_match.group(1))
        if value is not None:
            append_candidate(
                field_path="engine_limits.max_segment_seconds",
                value=value,
                confidence=0.99,
                summary=f"Set max segment duration to {value:g} seconds",
            )
    elif "max segment duration" in lowered:
        append_candidate(
            field_path="engine_limits.max_segment_seconds",
            value=None,
            confidence=0.35,
            summary="Possible max segment duration update requires review",
        )

    density_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*seconds?\s+being\s+the\s+sweet\s+spot",
        lowered,
    )
    if density_match:
        minimum = _safe_float(density_match.group(1))
        maximum = _safe_float(density_match.group(2))
        if minimum is not None and maximum is not None:
            append_candidate(
                field_path="annotation.preferred_density_sec",
                value={"min": minimum, "max": maximum},
                confidence=0.98,
                summary=f"Preferred density sweet spot is {minimum:g}-{maximum:g} seconds",
            )

    atomic_actions_match = re.search(r"(?:maximum|max)\s+of\s+(\d+)\s+atomic\s+actions", lowered)
    if atomic_actions_match:
        value = _safe_int(atomic_actions_match.group(1))
        if value is not None:
            append_candidate(
                field_path="annotation.max_atomic_actions",
                value=value,
                confidence=0.98,
                summary=f"Limit each segment to {value} atomic actions",
            )

    label_words_match = re.search(
        r"(?:segment|label).{0,40}never\s+exceed\s+\**\s*(\d+)\s+words",
        lowered,
    )
    if label_words_match:
        value = _safe_int(label_words_match.group(1))
        if value is not None:
            append_candidate(
                field_path="annotation.max_label_words",
                value=value,
                confidence=0.95,
                summary=f"Limit labels to {value} words",
            )

    return candidates


def extract_candidates_from_messages(
    messages: Sequence[Dict[str, Any]],
    *,
    channel: str = "",
    trust_config_path: Path | str = TRUST_CONFIG_PATH,
) -> List[Dict[str, Any]]:
    trust_config = _load_trust_config(trust_config_path)
    candidates: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        author_meta = message.get("author", {})
        if isinstance(author_meta, dict):
            author = author_meta.get("username") or author_meta.get("name") or "unknown"
        else:
            author = "unknown"
        metadata = {
            "author": author,
            "channel": str(message.get("channel", "")).strip() or channel,
            "timestamp": str(message.get("timestamp", "")).strip(),
            "message_id": str(message.get("id", "")).strip(),
            "trust_config": trust_config,
            "expires_at": message.get("expires_at"),
        }
        candidates.extend(_extract_rules_from_text(content, metadata))
    return candidates


def extract_candidates_from_rule_entries(
    rules: Sequence[Dict[str, Any]],
    *,
    channel: str = "prepurified_rules",
    trust_config_path: Path | str = TRUST_CONFIG_PATH,
) -> List[Dict[str, Any]]:
    trust_config = _load_trust_config(trust_config_path)
    candidates: List[Dict[str, Any]] = []
    for rule_obj in rules:
        if not isinstance(rule_obj, dict):
            continue
        rule_text = str(rule_obj.get("rule", "")).strip()
        if not rule_text:
            continue
        metadata = {
            "author": str(rule_obj.get("author", "system")).strip() or "system",
            "channel": str(rule_obj.get("channel", "")).strip() or channel,
            "timestamp": str(rule_obj.get("extracted_at", "")).strip(),
            "message_id": str(rule_obj.get("source_message_id", "")).strip(),
            "trust_config": trust_config,
            "expires_at": rule_obj.get("expires_at"),
        }
        candidates.extend(_extract_rules_from_text(rule_text, metadata))
    return candidates


def _upsert_staged_records(
    candidates: Sequence[Dict[str, Any]],
    *,
    staged_rules_path: Path | str = STAGED_RULES_PATH,
) -> List[Dict[str, Any]]:
    path = Path(staged_rules_path)
    existing = {str(row.get("rule_id", "")): row for row in _load_jsonl(path)}
    for record in candidates:
        rule_id = str(record.get("rule_id", "")).strip()
        if rule_id:
            existing[rule_id] = record
    ordered = sorted(existing.values(), key=lambda row: str(row.get("timestamp", "")))
    _save_jsonl(path, ordered)
    return ordered


def _policy_field_snapshot(policy: Dict[str, Any]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    for field_path in FIELD_TYPE_MAP:
        snapshot[field_path] = get_policy(field_path, None, policy=policy)
    return snapshot


def _active_promoted_records(
    records: Sequence[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    ordered = sorted(records, key=lambda row: str(row.get("timestamp", "")))
    return [record for record in ordered if _is_rule_active(record, now=now)]


def _write_derived_artifacts(
    policy: Dict[str, Any],
    *,
    policy_root: Path,
    changed_fields: Sequence[Dict[str, Any]],
    promoted_rule_ids: Sequence[str],
) -> None:
    summary = build_policy_prompt_summary(policy=policy)
    (policy_root / PROMPT_SUMMARY_PATH.name).write_text(summary + "\n", encoding="utf-8")

    generated_context = "\n".join(
        [
            "# Canonical Atlas Policy Context",
            "",
            summary,
            "",
            "Recent promoted fields:",
            *(f"- {item['field_path']}: {item['new_value']}" for item in changed_fields),
        ]
    ).strip()
    GENERATED_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_CONTEXT_PATH.write_text(generated_context + "\n", encoding="utf-8")

    _json_dump(
        policy_root / POLICY_DIFF_PATH.name,
        {
            "policy_version": str(policy.get("policy_version", "")),
            "generated_at": _utc_now(),
            "changed_fields": list(changed_fields),
            "promoted_rule_ids": list(promoted_rule_ids),
        },
    )


def rebuild_current_policy(
    *,
    current_policy_path: Path | str = CURRENT_POLICY_PATH,
    staged_rules_path: Path | str = STAGED_RULES_PATH,
) -> Dict[str, Any]:
    current_path = Path(current_policy_path)
    staged_path = Path(staged_rules_path)
    paths = ensure_policy_files(current_path.parent)
    previous_policy = load_current_policy(current_path)
    previous_snapshot = _policy_field_snapshot(previous_policy)
    records = _load_jsonl(staged_path)
    now_utc = datetime.now(timezone.utc)

    for record in records:
        record["active"] = _is_rule_active(record, now=now_utc)
        if record.get("status") == "promoted":
            record["inactive_reason"] = "expired" if not record["active"] and record.get("expires_at") else ""
        else:
            record["inactive_reason"] = ""

    rebuilt_policy = _default_policy()
    sources: List[Dict[str, Any]] = []
    active_records = _active_promoted_records(records, now=now_utc)
    latest_rule_by_field: Dict[str, str] = {}

    for record in active_records:
        field_path = str(record.get("field_path", "")).strip()
        if not field_path:
            continue
        value = record.get("value")
        _set_policy_value(rebuilt_policy, field_path, value)
        latest_rule_by_field[field_path] = str(record.get("rule_id", ""))
        source_entry = {
            "rule_id": str(record.get("rule_id", "")),
            "message_id": str(record.get("message_id", "")),
            "timestamp": str(record.get("timestamp", "")),
            "channel": str(record.get("channel", "")),
            "author": str(record.get("author", "")),
            "trust_tier": str(record.get("trust_tier", "")),
            "field_path": field_path,
            "value": value,
        }
        if record.get("expires_at"):
            source_entry["expires_at"] = record.get("expires_at")
        sources.append(source_entry)

    rebuilt_policy["generated_at"] = _utc_now()
    rebuilt_policy["effective_at"] = rebuilt_policy["generated_at"]
    rebuilt_policy["sources"] = sources[-50:]

    current_snapshot = _policy_field_snapshot(rebuilt_policy)
    changed_fields: List[Dict[str, Any]] = []
    all_field_paths = sorted(set(previous_snapshot) | set(current_snapshot))
    for field_path in all_field_paths:
        previous_value = previous_snapshot.get(field_path)
        new_value = current_snapshot.get(field_path)
        if previous_value != new_value:
            changed_fields.append(
                {
                    "field_path": field_path,
                    "old_value": previous_value,
                    "new_value": new_value,
                    "rule_id": latest_rule_by_field.get(field_path, ""),
                }
            )

    _json_dump(current_path, rebuilt_policy)
    _save_jsonl(staged_path, records)
    promoted_rule_ids = [str(record.get("rule_id", "")) for record in active_records]
    _write_derived_artifacts(
        rebuilt_policy,
        policy_root=paths["root"],
        changed_fields=changed_fields,
        promoted_rule_ids=promoted_rule_ids,
    )
    return {
        "policy_version": str(rebuilt_policy.get("policy_version", "")),
        "promoted_rule_ids": promoted_rule_ids,
        "changed_fields": changed_fields,
        "current_policy_path": str(current_path),
        "staged_rules_path": str(staged_path),
    }


def promote_staged_rules(
    *,
    current_policy_path: Path | str = CURRENT_POLICY_PATH,
    staged_rules_path: Path | str = STAGED_RULES_PATH,
    trust_config_path: Path | str = TRUST_CONFIG_PATH,
) -> Dict[str, Any]:
    current_path = Path(current_policy_path)
    staged_path = Path(staged_rules_path)
    trust_path = Path(trust_config_path)
    ensure_policy_files(current_path.parent)
    policy = load_current_policy(current_path)
    trust_config = _load_trust_config(trust_path)
    records = _load_jsonl(staged_path)
    newly_promoted_rule_ids: List[str] = []

    for record in sorted(records, key=lambda row: str(row.get("timestamp", ""))):
        if str(record.get("status", "")).strip().lower() == "promoted":
            continue
        field_path = str(record.get("field_path", "")).strip()
        value_type = str(record.get("value_type", "")).strip()
        scope = str(record.get("scope", "")).strip()
        trust_tier = str(record.get("trust_tier", "")).strip()
        if not field_path or not value_type:
            record["status"] = "rejected"
            record["decision_reason"] = "missing_field_metadata"
            continue
        if scope not in set(trust_config.get("auto_promote_scopes", [])):
            record["status"] = "staged"
            record["decision_reason"] = "scope_not_auto_promotable"
            continue
        if field_path not in set(trust_config.get("auto_promote_fields", [])):
            record["status"] = "staged"
            record["decision_reason"] = "field_not_auto_promotable"
            continue
        if not trust_tier.startswith("trusted"):
            record["status"] = "staged"
            record["decision_reason"] = "untrusted_source"
            continue
        if FIELD_TYPE_MAP.get(field_path) and FIELD_TYPE_MAP[field_path] != value_type:
            record["status"] = "staged"
            record["decision_reason"] = "field_type_conflict"
            continue
        if not _exact_confidence_ok(record):
            record["status"] = "staged"
            record["decision_reason"] = "not_exact_enough"
            continue

        value = record.get("value")
        compatibility_error = _compatibility_error(field_path, value, policy)
        if compatibility_error:
            record["status"] = "staged"
            record["decision_reason"] = compatibility_error
            continue

        record["status"] = "promoted"
        record["decision_reason"] = "trusted_auto_promote"
        newly_promoted_rule_ids.append(str(record.get("rule_id", "")))
    _save_jsonl(staged_path, records)
    rebuild_report = rebuild_current_policy(
        current_policy_path=current_path,
        staged_rules_path=staged_path,
    )
    rebuild_report["newly_promoted_rule_ids"] = newly_promoted_rule_ids
    return rebuild_report


def ingest_message_entries(
    messages: Sequence[Dict[str, Any]],
    *,
    policy_root: Path | str = POLICY_ROOT,
    channel: str = "",
) -> Dict[str, Any]:
    paths = ensure_policy_files(policy_root)
    candidates = extract_candidates_from_messages(
        messages,
        channel=channel,
        trust_config_path=paths["trust_config"],
    )
    _upsert_staged_records(candidates, staged_rules_path=paths["staged_rules"])
    report = promote_staged_rules(
        current_policy_path=paths["current_policy"],
        staged_rules_path=paths["staged_rules"],
        trust_config_path=paths["trust_config"],
    )
    report["candidate_count"] = len(candidates)
    return report


def ingest_rule_entries(
    rules: Sequence[Dict[str, Any]],
    *,
    policy_root: Path | str = POLICY_ROOT,
    channel: str = "prepurified_rules",
) -> Dict[str, Any]:
    paths = ensure_policy_files(policy_root)
    candidates = extract_candidates_from_rule_entries(
        rules,
        channel=channel,
        trust_config_path=paths["trust_config"],
    )
    _upsert_staged_records(candidates, staged_rules_path=paths["staged_rules"])
    report = promote_staged_rules(
        current_policy_path=paths["current_policy"],
        staged_rules_path=paths["staged_rules"],
        trust_config_path=paths["trust_config"],
    )
    report["candidate_count"] = len(candidates)
    return report


def build_policy_prompt_summary(
    *,
    policy: Optional[Dict[str, Any]] = None,
    current_policy_path: Path | str = CURRENT_POLICY_PATH,
) -> str:
    active_policy = load_current_policy(current_policy_path) if policy is None else policy
    max_segment = get_policy("engine_limits.max_segment_seconds", 10.0, policy=active_policy)
    preferred_density = get_policy(
        "annotation.preferred_density_sec",
        {"min": 2.0, "max": 5.0},
        policy=active_policy,
    )
    max_atomic_actions = get_policy("annotation.max_atomic_actions", 2, policy=active_policy)
    max_label_words = get_policy("annotation.max_label_words", 20, policy=active_policy)
    forbidden_verbs = get_policy("lexicon.forbidden_verbs", [], policy=active_policy)
    forbidden_narrative = get_policy("lexicon.forbidden_narrative_words", [], policy=active_policy)
    segment_style = str(get_policy("behavior.segment_style", "dense_first", policy=active_policy) or "").strip()
    correct_existing_labels_preferred = bool(
        get_policy("behavior.correct_existing_labels_preferred", True, policy=active_policy)
    )
    label_task_relevant_pauses = bool(
        get_policy("behavior.label_task_relevant_pauses_within_segment", True, policy=active_policy)
    )
    lines = [
        f"Canonical policy version: {active_policy.get('policy_version', 'atlas-policy')}",
        f"- Maximum segment duration: {max_segment:g} seconds.",
        f"- Preferred density sweet spot: {_format_value_for_summary(preferred_density)} seconds.",
        f"- Maximum atomic actions per label: {max_atomic_actions}.",
        f"- Maximum label words: {max_label_words}.",
        f"- Forbidden verbs: {', '.join(str(v) for v in forbidden_verbs)}.",
        f"- Forbidden narrative words: {', '.join(str(v) for v in forbidden_narrative)}.",
        f"- Segment style preference: {segment_style}.",
        f"- Correct existing labels before rewriting from scratch: {'yes' if correct_existing_labels_preferred else 'no'}.",
        f"- Label task-relevant pauses inside the kept segment: {'yes' if label_task_relevant_pauses else 'no'}.",
        "- No Action is standalone only and must not be used while a task-relevant object remains held.",
    ]
    return "\n".join(lines)


def retrieve_runtime_rules(
    source_segments: Sequence[Any],
    trigger: str,
    budget: Any = None,
    *,
    current_policy_path: Path | str = CURRENT_POLICY_PATH,
    staged_rules_path: Path | str = STAGED_RULES_PATH,
    lessons_path: Path | str = LESSONS_PATH,
    golden_path: Path | str = GOLDEN_INDEX_PATH,
) -> str:
    trigger_norm = str(trigger or "").strip().lower()
    if trigger_norm not in TRIGGERABLE_RETRIEVALS:
        return ""

    if isinstance(budget, int):
        max_rules = max(1, budget)
        max_examples = 1
    elif isinstance(budget, dict):
        max_rules = max(1, int(budget.get("rules", 6)))
        max_examples = max(0, int(budget.get("examples", 1)))
    else:
        max_rules = 6
        max_examples = 1

    active_policy = load_current_policy(current_policy_path)
    keywords = _extract_keywords(_segment_text(source_segments))
    universal_rules = list(get_policy("runtime.universal_rules", [], policy=active_policy))
    sections: List[str] = []

    if universal_rules:
        sections.append("=== CANONICAL AUDIT RULES ===")
        sections.extend(f"- {rule}" for rule in universal_rules[:4])

    candidate_rows: List[Tuple[int, str]] = []
    for record in _load_jsonl(Path(staged_rules_path)):
        if record.get("status") not in {"promoted", "staged"}:
            continue
        if record.get("expires_at") and not _is_rule_active({**record, "status": "promoted"}):
            continue
        text = f"{record.get('summary', '')} {record.get('raw_text', '')}"
        score = _score_overlap(text, keywords)
        if record.get("status") == "promoted":
            score += 5
        if str(record.get("trust_tier", "")).startswith("trusted"):
            score += 3
        if score > 0:
            candidate_rows.append((score, f"- {record.get('summary', '')}"))

    for row in _load_jsonl(Path(lessons_path)):
        text = f"{row.get('question', '')} {row.get('answer', '')}"
        score = _score_overlap(text, keywords)
        category = str(row.get("category", "")).strip().lower()
        if category in {"timestamp_policy", "hallucination", "quality_gate"}:
            score += 2
        if score > 0:
            candidate_rows.append((score, f"- {str(row.get('answer', '')).strip()}"))

    golden_lines: List[str] = []
    if max_examples > 0:
        for row in _load_jsonl(Path(golden_path)):
            if not row.get("has_labels"):
                continue
            score = _score_overlap(" ".join(row.get("keywords", [])), keywords)
            if score <= 0:
                continue
            labels = row.get("labels", {}).get("segments", [])
            preview = "; ".join(str(seg.get("label", "")) for seg in labels[:2] if isinstance(seg, dict))
            if preview:
                golden_lines.append(f"- {row.get('filename', 'example')}: {preview}")
            if len(golden_lines) >= max_examples:
                break

    if candidate_rows:
        sections.append("=== RETRIEVED POLICY LESSONS ===")
        candidate_rows.sort(key=lambda item: item[0], reverse=True)
        sections.extend(line for _score, line in candidate_rows[:max_rules])

    if golden_lines:
        sections.append("=== GOLDEN EXAMPLES ===")
        sections.extend(golden_lines[:max_examples])

    return "\n".join(sections).strip()
