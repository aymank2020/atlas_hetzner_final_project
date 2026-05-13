"""Chat-web only solve helpers for Atlas episodes."""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import fcntl
except ImportError:
    fcntl = None

import atlas_triplet_compare as _chat
from src.infra.solver_config import _cfg_get
from src.solver.desync import build_segment_snapshot
from src.solver.gemini_session import GeminiSession, validate_normalized_segments, validate_payload_schema
from src.solver.reliability import classify_transport_failure, transport_backoff_seconds

_REGISTERED_GEMINI_SESSIONS: Dict[str, GeminiSession] = {}


def _episode_registry_keys(episode_id: str) -> List[str]:
    raw = str(episode_id or "").strip().lower()
    if not raw:
        return []
    candidates = [raw]
    chunk_match = re.match(r"^(?P<base>.+?)_chunk_\d+$", raw)
    if chunk_match:
        candidates.append(str(chunk_match.group("base") or "").strip().lower())
    if raw.startswith("episode_"):
        candidates.append(raw[len("episode_") :].strip())
    else:
        candidates.append(f"episode_{raw}")
    out: List[str] = []
    seen: set[str] = set()
    for item in candidates:
        clean = str(item or "").strip().lower()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def register_episode_gemini_session(
    *,
    episode_id: str,
    runtime: Any,
    cfg: Dict[str, Any],
) -> Optional[GeminiSession]:
    keys = _episode_registry_keys(episode_id)
    if not keys:
        return None
    session = GeminiSession.start(runtime, cfg)
    for key in keys:
        _REGISTERED_GEMINI_SESSIONS[key] = session
    runtime.task_state["gemini_session_id"] = session.session_id
    return session


def unregister_episode_gemini_session(episode_id: str) -> None:
    keys = _episode_registry_keys(episode_id)
    if not keys:
        return
    for key in keys:
        _REGISTERED_GEMINI_SESSIONS.pop(key, None)


def _get_registered_gemini_session(episode_id: str) -> Optional[GeminiSession]:
    for key in _episode_registry_keys(episode_id):
        session = _REGISTERED_GEMINI_SESSIONS.get(key)
        if session is not None:
            return session
    return None


def _use_registered_gemini_session(cfg: Dict[str, Any], episode_id: str) -> Optional[GeminiSession]:
    use_v2 = bool(_cfg_get(cfg, "run.use_episode_runtime_v2", False))
    strict_single_session = bool(_cfg_get(cfg, "run.strict_single_chat_session", False))
    if not use_v2 or not strict_single_session:
        return None
    return _get_registered_gemini_session(episode_id)


def _ensure_registered_gemini_session_ready(session: GeminiSession) -> None:
    runtime = getattr(session, "runtime", None)
    if runtime is None:
        raise RuntimeError("Registered Gemini session is missing runtime.")
    try:
        session._ensure_page()
    except Exception as exc:
        raise RuntimeError(f"Registered Gemini session is not ready: {exc}") from exc


def restart_episode_gemini_session(
    *,
    cfg: Dict[str, Any],
    episode_id: str,
    source_segments: List[Dict[str, Any]],
    heartbeat: Optional[Callable[[], None]] = None,
) -> bool:
    session = _use_registered_gemini_session(cfg, episode_id)
    if session is None:
        return False
    snapshot = _build_runtime_snapshot(session=session, source_segments=source_segments)
    session.restart_with_minimal_history(snapshot, heartbeat=heartbeat)
    _ensure_registered_gemini_session_ready(session)
    return True


def _config_dir(cfg: Dict[str, Any]) -> Path:
    meta = cfg.get("_meta", {}) if isinstance(cfg.get("_meta"), dict) else {}
    raw = str(meta.get("config_dir", "") or "").strip()
    if raw:
        return Path(raw)
    return Path.cwd()


def _config_path(cfg: Dict[str, Any]) -> Path:
    meta = cfg.get("_meta", {}) if isinstance(cfg.get("_meta"), dict) else {}
    raw = str(meta.get("config_path", "") or "").strip()
    if raw:
        return Path(raw)
    return Path()


def _runtime_cfg_for_scope(
    cfg: Dict[str, Any],
    *,
    scope: str,
    model: str,
) -> Dict[str, Any]:
    runtime_cfg = copy.deepcopy(cfg)
    gem_cfg = runtime_cfg.setdefault("gemini", {})
    if not isinstance(gem_cfg, dict):
        runtime_cfg["gemini"] = {}
        gem_cfg = runtime_cfg["gemini"]
    gem_cfg["auth_mode"] = "chat_web"
    if str(model or "").strip():
        gem_cfg["model"] = str(model).strip()
    system_text = _chat._resolve_system_instruction_text(
        gem_cfg,
        cfg_dir=_config_dir(cfg),
        scope=scope,
        alias_text_keys=[
            "timed_labels_system_instruction_text",
            "chat_timed_system_instruction_text",
        ],
        alias_file_keys=[
            "timed_labels_system_instruction_file",
            "chat_timed_system_instruction_file",
        ],
    )
    if system_text:
        gem_cfg["system_instruction_text"] = system_text
    return runtime_cfg


def _segment_labels_response_schema() -> Dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "segments": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "segment_index": {"type": "INTEGER"},
                        "start_sec": {"type": "NUMBER"},
                        "end_sec": {"type": "NUMBER"},
                        "label": {"type": "STRING"},
                    },
                    "required": ["segment_index", "start_sec", "end_sec", "label"],
                },
            }
        },
        "required": ["segments"],
    }


def _structural_operations_response_schema(*, allow_merge: bool) -> Dict[str, Any]:
    allowed_actions = ["split", "merge"] if allow_merge else ["split"]
    return {
        "type": "OBJECT",
        "properties": {
            "operations": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "action": {"type": "STRING", "enum": allowed_actions},
                        "segment_index": {"type": "INTEGER"},
                    },
                    "required": ["action", "segment_index"],
                },
            }
        },
        "required": ["operations"],
    }


_LABELS_PROMPT_STRIP_PREFIXES = (
    "If boundaries are fundamentally wrong,",
    "Allowed operations:",
    "Operation segment_index refers",
    "Operations must be ordered exactly",
    "Return strict JSON object only:",
    "Response must start with '{' and end with '}'.",
    "Do not wrap JSON in markdown code fences.",
    'If no structural change is needed, return "operations":[]',
    "Structural operations are disabled for this pass.",
    "Return operations as an empty list.",
)


def _strip_structural_operations_contract(prompt_text: str) -> str:
    cleaned_lines: List[str] = []
    for raw_line in str(prompt_text or "").splitlines():
        line = str(raw_line or "")
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if any(stripped.startswith(prefix) for prefix in _LABELS_PROMPT_STRIP_PREFIXES):
            continue
        if '"operations":' in stripped or '{"operations":' in stripped:
            continue
        cleaned_lines.append(line)
    collapsed: List[str] = []
    previous_blank = False
    for line in cleaned_lines:
        is_blank = not str(line or "").strip()
        if is_blank and previous_blank:
            continue
        collapsed.append(line)
        previous_blank = is_blank
    return "\n".join(collapsed).strip()


def build_labels_prompt(prompt_text: str) -> str:
    base = _strip_structural_operations_contract(prompt_text)
    return (
        f"{base}\n\n"
        "Chat-only mode override:\n"
        'Return strict JSON object only with key "segments".\n'
        "Each segment item must include segment_index, start_sec, end_sec, and label.\n"
        "Keep one output row for every listed segment_index.\n"
        'Do not return top-level key "operations".\n'
        "Do not add, remove, split, merge, or reindex segments in this pass.\n"
        "Never return a segment_index that is not present in the provided segment list.\n"
        "Do not include markdown fences.\n"
    ).strip()


def build_structural_planner_prompt(
    source_segments: List[Dict[str, Any]],
    *,
    allow_merge: bool,
    max_segment_duration_sec: float,
    extra_instructions: str = "",
) -> str:
    merge_line = (
        "You may propose merge when consecutive segments represent one uninterrupted coarse goal."
        if allow_merge
        else "Merge is disabled; only propose split when clearly necessary."
    )
    lines = [
        "You are an Atlas structural-planning assistant.",
        "Review the video and current segment rows.",
        "Return ONLY structural operations. Do not return labels.",
        "Prefer an empty operations list when uncertain.",
        "Use split only when a segment clearly contains multiple actions/goals or exceeds the duration policy.",
        f"If a row exceeds {max_segment_duration_sec:.1f}s, split it even when the visible action simply continues into the next row.",
        merge_line,
        f"Current max segment duration target: {max_segment_duration_sec:.1f}s.",
        'Return strict JSON object only: {"operations":[{"action":"split","segment_index":3}]}',
        "Allowed actions: split"
        + (", merge" if allow_merge else "")
        + ". Never use delete.",
    ]
    extra = str(extra_instructions or "").strip()
    if extra:
        lines.extend(["", "Extra instructions:", extra])
    lines.extend(["", "Current segments:"])
    for seg in source_segments:
        lines.append(
            "- "
            f"segment_index={int(seg.get('segment_index', 0) or 0)} "
            f"start_sec={float(seg.get('start_sec', 0.0) or 0.0):.3f} "
            f"end_sec={float(seg.get('end_sec', 0.0) or 0.0):.3f} "
            f"current_label={json.dumps(str(seg.get('current_label', '') or ''), ensure_ascii=False)} "
            f"raw_text={json.dumps(str(seg.get('raw_text', '') or ''), ensure_ascii=False)}"
        )
    return "\n".join(lines).strip()


def build_targeted_repair_planner_prompt(
    source_segments: List[Dict[str, Any]],
    *,
    failing_indices: List[int],
    allow_merge: bool,
    max_segment_duration_sec: float,
    neighbor_count: int = 1,
    extra_instructions: str = "",
) -> str:
    failing = {
        int(idx)
        for idx in (failing_indices or [])
        if int(idx or 0) > 0
    }
    if not failing:
        return build_structural_planner_prompt(
            source_segments,
            allow_merge=allow_merge,
            max_segment_duration_sec=max_segment_duration_sec,
            extra_instructions=extra_instructions,
        )

    ordered = [
        int(seg.get("segment_index", 0) or 0)
        for seg in source_segments
        if int(seg.get("segment_index", 0) or 0) > 0
    ]
    scope: set[int] = set(failing)
    extra_neighbors = max(0, int(neighbor_count or 0))
    for idx in ordered:
        if idx not in failing:
            continue
        for delta in range(1, extra_neighbors + 1):
            scope.add(idx - delta)
            scope.add(idx + delta)

    scoped_segments = [
        seg for seg in source_segments if int(seg.get("segment_index", 0) or 0) in scope
    ]
    targeted_lines = [
        "Targeted repair mode: focus only on the failing segments and their immediate neighbors.",
        f"Failing segment indices: {sorted(failing)}.",
        "Return structural operations only.",
        "Prefer split for overlong or multi-action rows.",
        f"Split any row longer than {max_segment_duration_sec:.1f}s even if the action continues in the next row.",
        "Do not rewrite labels here.",
        "Do not propose operations outside the listed scope.",
    ]
    extra = str(extra_instructions or "").strip()
    if extra:
        targeted_lines.append(extra)
    return build_structural_planner_prompt(
        scoped_segments,
        allow_merge=allow_merge,
        max_segment_duration_sec=max_segment_duration_sec,
        extra_instructions="\n".join(targeted_lines),
    )


def _normalize_chat_segment_items(
    parsed: Any,
    source_segments: List[Dict[str, Any]],
    *,
    validation_errors: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    raw_items: Any = parsed
    if isinstance(raw_items, dict):
        raw_items = raw_items.get("segments", raw_items.get("labels", []))
    if not isinstance(raw_items, list):
        raw_items = []

    source_by_idx: Dict[int, Dict[str, Any]] = {
        int(seg.get("segment_index", 0) or 0): seg for seg in source_segments if int(seg.get("segment_index", 0) or 0) > 0
    }
    ordered_indices = [int(seg.get("segment_index", 0) or 0) for seg in source_segments if int(seg.get("segment_index", 0) or 0) > 0]
    used_indices: set[int] = set()
    sequential_index = 0
    out: List[Dict[str, Any]] = []

    def _next_unused_index() -> int:
        nonlocal sequential_index
        while sequential_index < len(ordered_indices):
            idx = ordered_indices[sequential_index]
            sequential_index += 1
            if idx not in used_indices:
                return idx
        return 0

    for item in raw_items:
        if not isinstance(item, dict):
            continue
        parsed_seg = _chat._segment_from_obj(item)
        if not parsed_seg:
            continue
        idx_raw = item.get("segment_index", item.get("index"))
        has_explicit_index = idx_raw is not None and str(idx_raw).strip() != ""
        try:
            idx = int(idx_raw)
        except Exception:
            idx = 0
        if has_explicit_index and idx <= 0:
            if validation_errors is not None:
                validation_errors.append(f"response included invalid explicit segment index: {idx_raw!r}")
            continue
        if has_explicit_index and idx not in source_by_idx:
            if validation_errors is not None:
                validation_errors.append(f"response referenced unknown segment {idx}")
            continue
        if has_explicit_index and idx in used_indices:
            if validation_errors is not None:
                validation_errors.append(f"response duplicated segment {idx}")
            continue
        if not has_explicit_index:
            idx = _next_unused_index()
        if idx <= 0 or idx not in source_by_idx:
            continue
        used_indices.add(idx)
        source = source_by_idx[idx]
        # DOM/source timestamps are authoritative in chat-web label mode.
        # Gemini may hallucinate segment boundaries even when labels are valid.
        start_sec = float(source.get("start_sec", 0.0) or 0.0)
        end_sec = float(source.get("end_sec", start_sec) or start_sec)
        out.append(
            {
                "segment_index": idx,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "label": str(parsed_seg.get("label", "") or "").strip(),
            }
        )

    out.sort(key=lambda item: item["segment_index"])
    return out


def _normalize_structural_operations(
    parsed: Any,
    *,
    allow_merge: bool,
) -> List[Dict[str, Any]]:
    raw_ops: Any = parsed
    if isinstance(raw_ops, dict):
        raw_ops = raw_ops.get("operations", [])
    if not isinstance(raw_ops, list):
        raw_ops = []

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
            token = str(item or "").strip().lower()
            match = re.match(r"([a-z]+)\s+(\d+)$", token)
            if match:
                action = match.group(1)
                idx = int(match.group(2))
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


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_runtime_snapshot(
    *,
    session: GeminiSession,
    source_segments: List[Dict[str, Any]],
) -> Any:
    runtime = session.runtime
    return build_segment_snapshot(
        segments=source_segments,
        episode_id=str(runtime.episode_id or "").strip(),
        context_id=str(runtime.context_id or "").strip(),
        page_url=str(getattr(runtime.atlas_page, "url", "") or getattr(runtime.gemini_page, "url", "") or ""),
        source_kind="extracted_source",
    )


def _session_result_meta(result: Any) -> Dict[str, Any]:
    request_context = getattr(result, "request_context", None)
    return {
        "request_id": str(getattr(result, "request_id", "") or "").strip(),
        "gemini_session_id": str(getattr(result, "session_id", "") or "").strip(),
        "retry_stage": str(getattr(result, "retry_stage", "") or "").strip(),
        "latency_ms": int(getattr(result, "latency_ms", 0) or 0),
        "raw_response_path": str(getattr(result, "raw_response_path", "") or "").strip(),
        "raw_response_meta_path": str(getattr(result, "raw_response_meta_path", "") or "").strip(),
        "request_mode": str(getattr(request_context, "mode", "") or "").strip(),
        "segments_checksum": str(getattr(request_context, "segments_checksum", "") or "").strip(),
        "baseline_message_count": int(getattr(request_context, "baseline_message_count", 0) or 0),
        "baseline_response_hash": str(getattr(request_context, "baseline_response_hash", "") or "").strip(),
        "request_context": (
            request_context.to_dict()
            if hasattr(request_context, "to_dict")
            else {}
        ),
        "acceptance_metadata": dict(getattr(result, "acceptance_metadata", {}) or {}),
    }


def _run_labels_generation_v2(
    *,
    session: GeminiSession,
    source_segments: List[Dict[str, Any]],
    video_file: Path,
    prompt_text: str,
    cache_dir: Path,
    episode_id: str,
    model: str,
    heartbeat: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    cache_root = cache_dir.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    prompt_path = cache_root / "labels_prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(str(prompt_text or "").strip() + "\n", encoding="utf-8")
    snapshot = _build_runtime_snapshot(session=session, source_segments=source_segments)
    result = session.generate_labels(snapshot, prompt_text, video_file, heartbeat=heartbeat)
    result_meta = _session_result_meta(result)
    result_meta["gemini_session_id"] = session.session_id

    if result.validation_errors or not result.validated_segments:
        debug_path = cache_root / f"failed_raw_chat_{episode_id}.txt"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(str(result.raw_text or ""), encoding="utf-8")
        message = "; ".join(result.validation_errors[:8]) if result.validation_errors else "response did not contain any usable segments"
        raise RuntimeError(
            "Chat labels response failed integrity checks: "
            + message
            + f". Raw output dumped to {debug_path}"
        )

    segments = list(result.validated_segments)
    out_txt_path = cache_root / "chat_labels.txt"
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    if not prompt_path.exists():
        prompt_path.write_text(str(prompt_text or "").strip() + "\n", encoding="utf-8")
    out_txt_path.write_text(_chat.segments_to_timed_text(segments) + "\n", encoding="utf-8")
    out_json_path = cache_root / "chat_labels.json"
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    _save_json(
        out_json_path,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "episode_id": str(episode_id or "").strip().lower(),
            "model": str(model or "").strip(),
            "segments": segments,
            "attach_notes": list(result.attach_notes),
            "usage": {},
            "raw_text": str(result.raw_text or ""),
            "raw_response_path": str(result.raw_response_path or ""),
            "raw_response_meta_path": str(getattr(result, "raw_response_meta_path", "") or ""),
            "prompt_path": str(prompt_path),
            **result_meta,
        },
    )
    return {
        "episode_id": str(episode_id or "").strip().lower(),
        "model": str(model or "").strip(),
        "segments": segments,
        "attach_notes": list(result.attach_notes),
        "usage": {},
        "out_json": str(out_json_path),
        "out_txt": str(out_txt_path),
        "prompt_path": str(prompt_path),
        "raw_response_meta_path": str(getattr(result, "raw_response_meta_path", "") or ""),
        **result_meta,
    }


def _run_structural_planner_v2(
    *,
    session: GeminiSession,
    source_segments: List[Dict[str, Any]],
    video_file: Path,
    prompt_text: str,
    cache_dir: Path,
    episode_id: str,
    model: str,
    allow_merge: bool,
    heartbeat: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    cache_root = cache_dir.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    prompt_path = cache_root / "ops_prompt.txt"
    prompt_path.write_text(str(prompt_text or "").strip() + "\n", encoding="utf-8")
    snapshot = _build_runtime_snapshot(session=session, source_segments=source_segments)
    result = session.plan_structural_operations(
        snapshot,
        prompt_text,
        allow_merge=allow_merge,
        video_file=video_file,
        heartbeat=heartbeat,
    )
    if result.validation_errors:
        debug_path = cache_root / f"failed_raw_ops_{episode_id}.txt"
        debug_path.write_text(str(result.raw_text or ""), encoding="utf-8")
        raise RuntimeError(
            "Chat structural planner response failed integrity checks: "
            + "; ".join(result.validation_errors[:8])
            + f". Raw output dumped to {debug_path}"
        )
    result_meta = _session_result_meta(result)
    result_meta["gemini_session_id"] = session.session_id
    operations = list(result.parsed_payload.get("operations", []) if isinstance(result.parsed_payload, dict) else [])
    out_json_path = cache_root / "chat_ops.json"
    _save_json(
        out_json_path,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "episode_id": str(episode_id or "").strip().lower(),
            "model": str(model or "").strip(),
            "operations": operations,
            "attach_notes": list(result.attach_notes),
            "usage": {},
            "raw_text": str(result.raw_text or ""),
            "raw_response_path": str(result.raw_response_path or ""),
            "raw_response_meta_path": str(getattr(result, "raw_response_meta_path", "") or ""),
            "prompt_path": str(prompt_path),
            "source_segment_count": len(source_segments),
            **result_meta,
        },
    )
    return {
        "episode_id": str(episode_id or "").strip().lower(),
        "model": str(model or "").strip(),
        "operations": operations,
        "attach_notes": list(result.attach_notes),
        "usage": {},
        "out_json": str(out_json_path),
        "prompt_path": str(prompt_path),
        "raw_response_meta_path": str(getattr(result, "raw_response_meta_path", "") or ""),
        **result_meta,
    }


def _run_chat_subprocess_once(
    *,
    cfg: Dict[str, Any],
    video_file: Path,
    prompt_text: str,
    cache_dir: Path,
    episode_id: str,
    model: str,
    prompt_scope: str,
    mode: str,
    allow_merge: bool = False,
    response_schema_enabled: bool = True,
    heartbeat: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    config_path = _config_path(cfg)
    if not config_path.exists():
        raise RuntimeError("Chat-only mode requires config_path metadata for subprocess execution.")

    cache_root = cache_dir.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    prompt_path = cache_root / f"{mode}_prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(str(prompt_text or "").strip() + "\n", encoding="utf-8")

    script_path = Path(__file__).resolve().parents[2] / "run_gemini_chat_json.py"
    gem_cfg = cfg.get("gemini", {}) if isinstance(cfg.get("gemini"), dict) else {}
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
    default_timeout_sec = max(60.0, float(gem_cfg.get("chat_web_timeout_sec", 360) or 360))
    if mode == "ops":
        timeout_raw = run_cfg.get("chat_ops_timeout_sec", gem_cfg.get("chat_ops_timeout_sec", 300.0))
        timeout_sec = max(60.0, min(default_timeout_sec, float(timeout_raw or 300.0)))
    else:
        timeout_raw = run_cfg.get("chat_labels_timeout_sec", gem_cfg.get("chat_labels_timeout_sec", default_timeout_sec))
        timeout_sec = max(60.0, float(timeout_raw or default_timeout_sec))

    heartbeat_interval_sec = max(
        5.0, float(run_cfg.get("chat_subprocess_heartbeat_sec", 10) or 10)
    )
    request_id = uuid.uuid4().hex[:8]
    started_at = time.monotonic()
    started_at_utc = datetime.now(timezone.utc).isoformat()

    command = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--video-path",
        str(video_file),
        "--cache-dir",
        str(cache_root),
        "--episode-id",
        str(episode_id or "").strip(),
        "--model",
        str(model or "").strip(),
        "--prompt-file",
        str(prompt_path),
        "--prompt-scope",
        str(prompt_scope or "").strip(),
        "--mode",
        str(mode or "").strip(),
    ]
    if allow_merge:
        command.append("--allow-merge")
    if response_schema_enabled:
        command.append("--response-schema-enabled")

    print(
        f"[trace] chat request start: request_id={request_id} "
        f"episode_id={str(episode_id or '').strip() or 'unknown'} "
        f"mode={mode} model={str(model or '').strip() or 'unknown'} "
        f"timeout={timeout_sec:.0f}s prompt={prompt_path.name}"
    )

    # ── Popen + polling with heartbeat ──────────────────────────────
    deadline = time.monotonic() + timeout_sec + 60
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    last_heartbeat_ts = time.monotonic()
    try:
        while True:
            try:
                stdout_data, stderr_data = proc.communicate(timeout=heartbeat_interval_sec)
                # Process finished
                break
            except subprocess.TimeoutExpired:
                # Still running — send heartbeat
                now = time.monotonic()
                if heartbeat is not None and (now - last_heartbeat_ts) >= heartbeat_interval_sec:
                    try:
                        heartbeat()
                    except Exception:
                        pass
                    last_heartbeat_ts = now
                if now >= deadline:
                    proc.kill()
                    proc.wait(timeout=5)
                    video_mb = 0.0
                    try:
                        video_mb = video_file.stat().st_size / (1024 * 1024)
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Chat {mode} subprocess timed out after {int(timeout_sec + 60)}s "
                        f"(configured timeout={timeout_sec:.0f}s). "
                        f"Video: {video_file.name} ({video_mb:.1f}MB) "
                        f"[request_id={request_id}]"
                    )
    except Exception:
        # Ensure child is cleaned up on any exception
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        raise

    duration_sec = time.monotonic() - started_at
    if proc.returncode != 0:
        stderr = str(stderr_data or "").strip()
        stdout = str(stdout_data or "").strip()
        if stdout:
            try:
                (cache_root / f"chat_{mode}_stdout_{episode_id}_{request_id}.txt").write_text(
                    stdout,
                    encoding="utf-8",
                )
            except Exception:
                pass
        if stderr:
            try:
                (cache_root / f"chat_{mode}_stderr_{episode_id}_{request_id}.txt").write_text(
                    stderr,
                    encoding="utf-8",
                )
            except Exception:
                pass
        detail = stderr or stdout or f"chat subprocess failed with exit code {proc.returncode}"
        print(
            f"[trace] chat request failed: request_id={request_id} "
            f"mode={mode} rc={proc.returncode} duration={duration_sec:.1f}s"
        )
        raise RuntimeError(f"{detail} [request_id={request_id}]")
    raw = str(stdout_data or "").strip()
    if not raw:
        print(
            f"[trace] chat request failed: request_id={request_id} "
            f"mode={mode} empty_output duration={duration_sec:.1f}s"
        )
        raise RuntimeError(f"Chat subprocess returned empty output. [request_id={request_id}]")

    # ── IMMEDIATE SAVE: write raw response BEFORE parsing ──
    # Prevents data loss if JSON parsing fails
    raw_save_path = cache_root / f"raw_gemini_response_{episode_id}_{request_id}.txt"
    raw_meta_path = cache_root / f"raw_gemini_response_{episode_id}_{request_id}.json"
    try:
        raw_save_path.write_text(raw, encoding="utf-8")
    except Exception:
        pass
    try:
        _save_json(
            raw_meta_path,
            {
                "request_id": request_id,
                "episode_id": str(episode_id or "").strip(),
                "model": str(model or "").strip(),
                "mode": str(mode or "").strip(),
                "prompt_scope": str(prompt_scope or "").strip(),
                "started_at_utc": started_at_utc,
                "prompt_path": str(prompt_path),
                "raw_text_path": str(raw_save_path),
                "raw_text": raw,
            },
        )
    except Exception:
        pass
    print(
        f"[trace] chat request raw saved: request_id={request_id} "
        f"file={raw_save_path.name} len={len(raw)} duration={duration_sec:.1f}s"
    )

    try:
        # Diagnostic prints in the subprocess can pollute stdout. Extract the JSON block:
        start_idx = raw.find('{')
        end_idx = raw.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            clean_raw = raw[start_idx:end_idx+1]
        else:
            clean_raw = raw
        
        payload = json.loads(clean_raw)
    except Exception as exc:
        raise RuntimeError(
            f"Chat subprocess returned invalid JSON: {raw[:400]} [request_id={request_id}]"
        ) from exc
    payload["prompt_path"] = str(prompt_path)
    payload["_request_id"] = request_id
    payload["_request_duration_sec"] = round(duration_sec, 3)
    payload["_raw_response_path"] = str(raw_save_path)
    payload["_raw_response_meta_path"] = str(raw_meta_path)
    payload["_started_at_utc"] = started_at_utc
    print(
        f"[trace] chat request completed: request_id={request_id} "
        f"mode={mode} duration={duration_sec:.1f}s"
    )
    return payload


def _run_chat_subprocess(
    *,
    cfg: Dict[str, Any],
    video_file: Path,
    prompt_text: str,
    cache_dir: Path,
    episode_id: str,
    model: str,
    prompt_scope: str,
    mode: str,
    allow_merge: bool = False,
    response_schema_enabled: bool = True,
    heartbeat: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
    retries_raw = (
        run_cfg.get("gemini_transport_max_retries_ops", run_cfg.get("gemini_transport_max_retries", 2))
        if mode == "ops"
        else run_cfg.get("gemini_transport_max_retries", 3)
    )
    max_transport_retries = max(1, int(retries_raw or (2 if mode == "ops" else 3)))
    last_error: Optional[Exception] = None

    for attempt in range(1, max_transport_retries + 1):
        try:
            return _run_chat_subprocess_once(
                cfg=cfg,
                video_file=video_file,
                prompt_text=prompt_text,
                cache_dir=cache_dir,
                episode_id=episode_id,
                model=model,
                prompt_scope=prompt_scope,
                mode=mode,
                allow_merge=allow_merge,
                response_schema_enabled=response_schema_enabled,
                heartbeat=heartbeat,
            )
        except Exception as exc:
            last_error = exc
            reason = classify_transport_failure(str(exc))
            if attempt >= max_transport_retries:
                raise
            backoff_sec = transport_backoff_seconds(attempt)
            print(
                f"[trace] chat request retrying: "
                f"episode_id={str(episode_id or '').strip() or 'unknown'} "
                f"mode={mode} attempt={attempt + 1}/{max_transport_retries} "
                f"retry_reason={reason} backoff={backoff_sec:.1f}s"
            )
            time.sleep(backoff_sec)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Chat subprocess failed without error details.")


def _run_labels_generation_impl(
    *,
    cfg: Dict[str, Any],
    source_segments: List[Dict[str, Any]],
    video_file: Path,
    prompt_text: str,
    cache_dir: Path,
    episode_id: str,
    model: str,
    prompt_scope: str = "chat_labels",
    heartbeat: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    session = _use_registered_gemini_session(cfg, episode_id)
    if session is not None:
        return _run_labels_generation_v2(
            session=session,
            source_segments=source_segments,
            video_file=video_file,
            prompt_text=prompt_text,
            cache_dir=cache_dir,
            episode_id=episode_id,
            model=model,
            heartbeat=heartbeat,
        )

    cache_root = cache_dir.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    gem_cfg = cfg.get("gemini", {}) if isinstance(cfg.get("gemini"), dict) else {}
    result = _run_chat_subprocess(
        cfg=cfg,
        video_file=video_file,
        prompt_text=prompt_text,
        cache_dir=cache_root,
        episode_id=episode_id,
        model=model,
        prompt_scope=prompt_scope,
        mode="labels",
        response_schema_enabled=bool(gem_cfg.get("chat_labels_response_schema_enabled", True)),
        heartbeat=heartbeat,
    )
    validation_errors: List[str] = []
    validation_errors.extend(
        validate_payload_schema(
            result.get("parsed", {}),
            expected_schema="segments_only",
        )
    )
    segments = _normalize_chat_segment_items(
        result.get("parsed", {}),
        source_segments,
        validation_errors=validation_errors,
    )

    validated_segments, integrity_errors = validate_normalized_segments(
        segments,
        source_segments,
        allow_partial=False,
    )
    if integrity_errors:
        validation_errors.extend(integrity_errors)

    if validation_errors:
        debug_path = cache_root / f"failed_raw_chat_{episode_id}.txt"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(str(result.get("raw_text", "")), encoding="utf-8")
        raise RuntimeError(
            "Chat labels response failed integrity checks: "
            + "; ".join(validation_errors[:8])
            + f". Raw output dumped to {debug_path}"
        )

    if not validated_segments:
        # Save raw output to a debug file before crashing
        debug_path = cache_root / f"failed_raw_chat_{episode_id}.txt"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(str(result.get("raw_text", "")), encoding="utf-8")
        raise RuntimeError(f"Chat labels response did not contain any usable segments. Raw output dumped to {debug_path}")
    segments = validated_segments
    out_txt_path = cache_root / "chat_labels.txt"
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path.write_text(_chat.segments_to_timed_text(segments) + "\n", encoding="utf-8")
    out_json_path = cache_root / "chat_labels.json"
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    _save_json(
        out_json_path,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "episode_id": str(episode_id or "").strip().lower(),
            "model": str(model or "").strip(),
            "segments": segments,
            "attach_notes": result.get("attach_notes", []),
              "usage": result.get("usage", {}),
              "raw_text": str(result.get("raw_text", "") or ""),
              "raw_response_path": str(result.get("_raw_response_path", "") or ""),
              "raw_response_meta_path": str(result.get("_raw_response_meta_path", "") or ""),
              "prompt_path": str(result.get("prompt_path", "") or ""),
              "request_id": str(result.get("_request_id", "") or ""),
              "started_at_utc": str(result.get("_started_at_utc", "") or ""),
          },
      )
    return {
        "episode_id": str(episode_id or "").strip().lower(),
        "model": str(model or "").strip(),
        "segments": segments,
        "attach_notes": result.get("attach_notes", []),
        "usage": result.get("usage", {}),
        "out_json": str(out_json_path),
        "out_txt": str(out_txt_path),
        "prompt_path": str(result.get("prompt_path", "") or ""),
        "raw_response_path": str(result.get("_raw_response_path", "") or ""),
        "raw_response_meta_path": str(result.get("_raw_response_meta_path", "") or ""),
        "request_id": str(result.get("_request_id", "") or ""),
        "started_at_utc": str(result.get("_started_at_utc", "") or ""),
    }


def _run_structural_planner_impl(
    *,
    cfg: Dict[str, Any],
    source_segments: List[Dict[str, Any]],
    video_file: Path,
    prompt_text: str,
    cache_dir: Path,
    episode_id: str,
    model: str,
    allow_merge: bool,
    prompt_scope: str = "chat_ops",
    heartbeat: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    session = _use_registered_gemini_session(cfg, episode_id)
    if session is not None:
        return _run_structural_planner_v2(
            session=session,
            source_segments=source_segments,
            video_file=video_file,
            prompt_text=prompt_text,
            cache_dir=cache_dir,
            episode_id=episode_id,
            model=model,
            allow_merge=allow_merge,
            heartbeat=heartbeat,
        )

    cache_root = cache_dir.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    gem_cfg = cfg.get("gemini", {}) if isinstance(cfg.get("gemini"), dict) else {}
    result = _run_chat_subprocess(
        cfg=cfg,
        video_file=video_file,
        prompt_text=prompt_text,
        cache_dir=cache_root,
        episode_id=episode_id,
        model=model,
        prompt_scope=prompt_scope,
        mode="ops",
        allow_merge=allow_merge,
        response_schema_enabled=bool(gem_cfg.get("chat_ops_response_schema_enabled", True)),
        heartbeat=heartbeat,
    )
    validation_errors = validate_payload_schema(
        result.get("parsed", {}),
        expected_schema="operations_only",
        requested_indices=[int(seg.get("segment_index", 0) or 0) for seg in source_segments],
        allow_merge=allow_merge,
    )
    if validation_errors:
        debug_path = cache_root / f"failed_raw_ops_{episode_id}.txt"
        debug_path.write_text(str(result.get("raw_text", "")), encoding="utf-8")
        raise RuntimeError(
            "Chat structural planner response failed integrity checks: "
            + "; ".join(validation_errors[:8])
            + f". Raw output dumped to {debug_path}"
        )
    operations = _normalize_structural_operations(result.get("parsed", {}), allow_merge=allow_merge)
    out_json_path = cache_root / "chat_ops.json"
    _save_json(
        out_json_path,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "episode_id": str(episode_id or "").strip().lower(),
            "model": str(model or "").strip(),
            "operations": operations,
            "attach_notes": result.get("attach_notes", []),
            "usage": result.get("usage", {}),
            "raw_text": str(result.get("raw_text", "") or ""),
            "raw_response_path": str(result.get("_raw_response_path", "") or ""),
            "raw_response_meta_path": str(result.get("_raw_response_meta_path", "") or ""),
            "prompt_path": str(result.get("prompt_path", "") or ""),
            "source_segment_count": len(source_segments),
        },
    )
    return {
        "episode_id": str(episode_id or "").strip().lower(),
        "model": str(model or "").strip(),
        "operations": operations,
        "attach_notes": result.get("attach_notes", []),
        "usage": result.get("usage", {}),
        "out_json": str(out_json_path),
        "prompt_path": str(result.get("prompt_path", "") or ""),
        "raw_response_path": str(result.get("_raw_response_path", "") or ""),
        "raw_response_meta_path": str(result.get("_raw_response_meta_path", "") or ""),
        "request_id": str(result.get("_request_id", "") or ""),
        "started_at_utc": str(result.get("_started_at_utc", "") or ""),
    }


def run_repair_query(
    *,
    cfg: Dict[str, Any],
    source_segments: List[Dict[str, Any]],
    prompt_text: str,
    cache_dir: Path,
    episode_id: str,
    model: str,
    video_file: Optional[Path] = None,
    failing_indices: Optional[List[int]] = None,
    current_plan: Optional[Dict[int, Dict[str, Any]]] = None,
    retry_reason: str = "",
    heartbeat: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Consult Gemini for a specific repair or expert consultation query."""
    cache_root = cache_dir.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    session = _use_registered_gemini_session(cfg, episode_id)
    if session is not None:
        snapshot = _build_runtime_snapshot(session=session, source_segments=source_segments)
        scoped_indices = [int(idx) for idx in (failing_indices or []) if int(idx or 0) > 0]
        if scoped_indices:
            session_result = session.repair_failed_segments(
                snapshot,
                scoped_indices,
                current_plan or {},
                retry_reason or "repair_query",
                heartbeat=heartbeat,
            )
        else:
            session_result = session.request_json(
                snapshot=snapshot,
                prompt=prompt_text,
                video_file=video_file,
                allow_partial=True,
                expected_schema="",
                heartbeat=heartbeat,
            )
        parsed = session_result.parsed_payload if isinstance(session_result.parsed_payload, dict) else {}
        validation_errors: List[str] = list(session_result.validation_errors)
        prompt_path = cache_root / "repair_prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(str(prompt_text or "").strip() + "\n", encoding="utf-8")
        if session_result.validated_segments and not validation_errors:
            segments = list(session_result.validated_segments)
            out_txt_path = cache_root / "chat_labels.txt"
            out_txt_path.parent.mkdir(parents=True, exist_ok=True)
            out_txt_path.write_text(_chat.segments_to_timed_text(segments) + "\n", encoding="utf-8")
            out_json_path = cache_root / "chat_labels.json"
            out_json_path.parent.mkdir(parents=True, exist_ok=True)
            _save_json(
                out_json_path,
                {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "episode_id": str(episode_id or "").strip().lower(),
                    "model": str(model or "").strip(),
                    "segments": segments,
                    "attach_notes": list(getattr(session_result, "attach_notes", []) or []),
                    "usage": {},
                    "raw_text": str(session_result.raw_text or ""),
                    "raw_response_path": str(session_result.raw_response_path or ""),
                    "raw_response_meta_path": str(getattr(session_result, "raw_response_meta_path", "") or ""),
                    "prompt_path": str(prompt_path),
                    "request_id": str(session_result.request_id or ""),
                    "gemini_session_id": session.session_id,
                    "retry_stage": str(session_result.retry_stage or ""),
                    "latency_ms": int(session_result.latency_ms or 0),
                    "retry_reason": str(retry_reason or ""),
                },
            )
            out = {
                "segments": segments,
                "out_txt": str(out_txt_path),
                "out_json": str(out_json_path),
            }
        else:
            out = parsed if isinstance(parsed, dict) else {"parsed": parsed}
        out["_meta"] = {
            "raw_response": session_result.raw_text,
            "raw_response_path": str(session_result.raw_response_path or ""),
            "usage": {},
            "prompt_path": str(prompt_path),
            "validation_errors": validation_errors,
            "request_id": str(session_result.request_id or ""),
            "gemini_session_id": session.session_id,
            "retry_stage": str(session_result.retry_stage or ""),
            "latency_ms": int(session_result.latency_ms or 0),
            "retry_reason": str(retry_reason or ""),
            "raw_response_meta_path": str(getattr(session_result, "raw_response_meta_path", "") or ""),
            "request_context": (
                session_result.request_context.to_dict()
                if getattr(session_result, "request_context", None) is not None
                and hasattr(session_result.request_context, "to_dict")
                else {}
            ),
            "acceptance_metadata": dict(getattr(session_result, "acceptance_metadata", {}) or {}),
        }
        return out

    # If video_file is not provided, try to find it in the cache_dir
    if not video_file:
        print("[repair] WARNING: no video file found for repair query; attempting text-only.")
        candidates = list(cache_root.parent.glob(f"video_{episode_id}.mp4"))
        if candidates:
            video_file = candidates[0]
        else:
            video_file = cache_root.parent / f"video_{episode_id}.mp4"

    result = _run_chat_subprocess(
        cfg=cfg,
        video_file=video_file,
        prompt_text=prompt_text,
        cache_dir=cache_root,
        episode_id=episode_id,
        model=model,
        prompt_scope="chat_repair",
        mode="labels",
        response_schema_enabled=bool(failing_indices),
        heartbeat=heartbeat,
    )
    
    # Repaired result is often free-text or a specific JSON blob.
    # If the response contains usable segments (e.g. for an overlong repair), normalize them.
    parsed = result.get("parsed", {})
    validation_errors: List[str] = []
    segments = _normalize_chat_segment_items(
        parsed,
        source_segments,
        validation_errors=validation_errors,
    )

    if segments:
        segments, integrity_errors = validate_normalized_segments(
            segments,
            source_segments,
            allow_partial=True,
        )
        validation_errors.extend(integrity_errors)
    if segments and not validation_errors:
        print(f"[chat_only] repair query yielded {len(segments)} normalized segments")
        out = {"segments": segments}
    else:
        out = parsed if isinstance(parsed, dict) else {"parsed": parsed}
        
    out["_meta"] = {
        "raw_response": result.get("raw_text", ""),
        "usage": result.get("usage", {}),
        "prompt_path": result.get("prompt_path", ""),
        "validation_errors": validation_errors,
        "raw_response_path": str(result.get("_raw_response_path", "") or ""),
        "raw_response_meta_path": str(result.get("_raw_response_meta_path", "") or ""),
    }
    return out


def run_labels_generation(
    *,
    cfg: Dict[str, Any],
    source_segments: List[Dict[str, Any]],
    video_file: Path,
    prompt_text: str,
    cache_dir: Path,
    episode_id: str,
    model: str,
    prompt_scope: str = "chat_labels",
    heartbeat: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Backward compatibility wrapper for labels generation."""
    return _run_labels_generation_impl(
        cfg=cfg,
        source_segments=source_segments,
        video_file=video_file,
        prompt_text=prompt_text,
        cache_dir=cache_dir,
        episode_id=episode_id,
        model=model,
        prompt_scope=prompt_scope,
        heartbeat=heartbeat,
    )


def run_structural_planner(
    *,
    cfg: Dict[str, Any],
    source_segments: List[Dict[str, Any]],
    video_file: Path,
    prompt_text: str,
    cache_dir: Path,
    episode_id: str,
    model: str,
    allow_merge: bool,
    prompt_scope: str = "chat_ops",
    heartbeat: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Backward compatibility wrapper for structural planner."""
    return _run_structural_planner_impl(
        cfg=cfg,
        source_segments=source_segments,
        video_file=video_file,
        prompt_text=prompt_text,
        cache_dir=cache_dir,
        episode_id=episode_id,
        model=model,
        allow_merge=allow_merge,
        prompt_scope=prompt_scope,
        heartbeat=heartbeat,
    )


__all__ = [
    "build_labels_prompt",
    "build_structural_planner_prompt",
    "build_targeted_repair_planner_prompt",
    "register_episode_gemini_session",
    "restart_episode_gemini_session",
    "run_labels_generation",
    "run_structural_planner",
    "run_repair_query",
    "unregister_episode_gemini_session",
]
