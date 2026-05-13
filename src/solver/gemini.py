"""Gemini transport and response helpers extracted from the legacy solver."""

from __future__ import annotations

import base64
import json
import logging
import re
import shutil
import time
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.infra.gemini_economics import (
    build_episode_cost_updates,
    estimate_cost_from_usage,
    estimate_cost_usd,
    resolve_stage_model,
)
from src.infra.logging_utils import build_print_logger as _build_print_logger
from src.infra.solver_config import (
    GeminiKeyPool,
    _cfg_get,
    _ordered_gen3_gemini_models,
    collect_unique_gemini_keys,
)

_logger = logging.getLogger(__name__)
print = _build_print_logger(_logger)


def _emit_solver_heartbeat() -> None:
    try:
        legacy = import_module("src.solver.legacy_impl")
        callback = getattr(legacy, "_ACTIVE_HEARTBEAT_CALLBACK", None)
        if callable(callback):
            callback()
    except Exception:
        pass


def _chat_phase_watchdog_timeout_hint_sec(
    cfg: Dict[str, Any],
    *,
    phase: str,
    request_scope: str = "",
) -> float:
    phase_name = str(phase or "").strip().lower()
    scope_name = str(request_scope or "").strip().lower()
    base_timeout_sec = max(
        60.0,
        float(_cfg_get(cfg, "run.watchdog_stale_threshold_sec", 600.0) or 600.0),
    )
    cap_timeout_sec = max(
        base_timeout_sec,
        float(
            _cfg_get(
                cfg,
                "run.watchdog_dynamic_timeout_cap_sec",
                max(2400.0, base_timeout_sec),
            )
            or max(2400.0, base_timeout_sec)
        ),
    )
    request_buffer_sec = max(
        60.0,
        float(_cfg_get(cfg, "run.chat_request_watchdog_buffer_sec", 180.0) or 180.0),
    )
    attach_floor_sec = max(
        120.0,
        float(_cfg_get(cfg, "run.chat_attach_watchdog_floor_sec", 300.0) or 300.0),
    )
    dispatch_floor_sec = max(
        120.0,
        float(_cfg_get(cfg, "run.chat_dispatch_watchdog_floor_sec", 300.0) or 300.0),
    )
    default_chat_timeout_sec = max(
        60.0,
        float(_cfg_get(cfg, "gemini.chat_web_timeout_sec", 360.0) or 360.0),
    )
    labels_timeout_sec = max(
        60.0,
        float(_cfg_get(cfg, "run.chat_labels_timeout_sec", default_chat_timeout_sec) or default_chat_timeout_sec),
    )
    ops_timeout_sec = max(
        60.0,
        float(
            _cfg_get(
                cfg,
                "run.chat_ops_timeout_sec",
                _cfg_get(cfg, "gemini.chat_ops_timeout_sec", min(default_chat_timeout_sec, 300.0)),
            )
            or min(default_chat_timeout_sec, 300.0)
        ),
    )

    is_planner_scope = scope_name == "planner" or phase_name.startswith("planner")
    active_request_timeout_sec = ops_timeout_sec if is_planner_scope else labels_timeout_sec

    if phase_name in {"planner", "chunk_request", "single_request", "response_wait"}:
        return min(cap_timeout_sec, max(base_timeout_sec, active_request_timeout_sec + request_buffer_sec))
    if phase_name in {"attach_start", "attach_wait", "attach_done"}:
        return min(cap_timeout_sec, max(base_timeout_sec, attach_floor_sec))
    if phase_name in {"prompt_send", "prompt_dispatch"}:
        return min(cap_timeout_sec, max(base_timeout_sec, dispatch_floor_sec))
    return base_timeout_sec


def _clean_json_text(text: str) -> str:
    clean = re.sub(r"```json|```", "", text or "", flags=re.IGNORECASE).strip()
    object_start = clean.find("{")
    object_end = clean.rfind("}")
    list_start = clean.find("[")
    list_end = clean.rfind("]")
    if list_start >= 0 and list_end > list_start and (object_start < 0 or list_start < object_start):
        return clean[list_start : list_end + 1]
    if object_start >= 0 and object_end > object_start:
        return clean[object_start : object_end + 1]
    return clean


_GEMINI_JSON_INDEX_REPAIR_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r'("segment_index"\s*:\s*)(\d+_\d+)\b'),
    re.compile(r'("index"\s*:\s*)(\d+_\d+)\b'),
    re.compile(r'("segment"\s*:\s*)(\d+_\d+)\b'),
)


def _repair_gemini_json_text(text: str) -> str:
    repaired = _clean_json_text(text)
    for pattern in _GEMINI_JSON_INDEX_REPAIR_PATTERNS:
        repaired = pattern.sub(r'\1"\2"', repaired)
    return repaired


def _enforce_gemini_output_contract(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Gemini output to the expected contract."""
    normalized = dict(payload or {})
    operations = normalized.get("operations", [])
    normalized["operations"] = operations if isinstance(operations, list) else []
    segments = normalized.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Gemini payload must contain list at 'segments'")
    return normalized


def _parse_json_text(text: str) -> Dict[str, Any]:
    cleaned = _clean_json_text(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = _repair_gemini_json_text(cleaned)
        if repaired == cleaned:
            raise
        payload = json.loads(repaired)
    if isinstance(payload, dict):
        return _enforce_gemini_output_contract(payload)
    if isinstance(payload, list):
        return _enforce_gemini_output_contract({"operations": [], "segments": payload})
    raise ValueError("Gemini response is not JSON object/list")


def _parse_gemini_response(data: Dict[str, Any]) -> Dict[str, Any]:
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                try:
                    return _parse_json_text(text)
                except Exception:
                    continue
    raise RuntimeError(f"Could not parse JSON from Gemini response: {data}")


def _gemini_response_has_nonempty_text_candidate(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    for candidate in data.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content", {})
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []):
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return True
    return False


def _merge_usage_metadata(items: List[Dict[str, Any]]) -> Dict[str, int]:
    prompt_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            prompt_tokens += int(item.get("promptTokenCount", item.get("prompt_tokens", 0)) or 0)
        except Exception:
            pass
        try:
            output_tokens += int(item.get("candidatesTokenCount", item.get("output_tokens", 0)) or 0)
        except Exception:
            pass
        try:
            total_tokens += int(item.get("totalTokenCount", item.get("total_tokens", 0)) or 0)
        except Exception:
            pass
    if total_tokens <= 0:
        total_tokens = max(0, prompt_tokens + output_tokens)
    return {
        "promptTokenCount": prompt_tokens,
        "candidatesTokenCount": output_tokens,
        "totalTokenCount": total_tokens,
    }


def _chat_attach_notes_confirm_video(notes: Any) -> bool:
    if not isinstance(notes, list):
        return False
    for item in notes:
        text = str(item or "").strip().lower()
        if not text:
            continue
        if "attached" in text and "skipped" not in text:
            return True
    return False


_LAST_GEMINI_REQUEST_TS = 0.0
_GEMINI_REQUEST_TIMESTAMPS: List[float] = []
_GEMINI_QUOTA_COOLDOWN_UNTIL_TS = 0.0
_GEMINI_ZERO_QUOTA_MODELS: Dict[str, float] = {}
_STRICT_PAID_GEMINI_POOL: Optional[GeminiKeyPool] = None
_STRICT_PAID_GEMINI_POOL_SIGNATURE: Optional[Tuple[Any, ...]] = None


def _extract_retry_seconds_from_text(text: str, default_wait_sec: float = 0.0) -> float:
    body = (text or "").lower()
    if not body:
        return max(0.0, float(default_wait_sec))

    compound_match = re.search(r"(?:retry|try again|please retry)\s+in\s*([^\n\r,;]+)", body)
    if compound_match:
        fragment = compound_match.group(1)[:96]
        total_sec = 0.0
        token_count = 0
        for amount_str, unit in re.findall(
            r"([0-9]+(?:\.[0-9]+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)",
            fragment,
        ):
            try:
                amount = float(amount_str)
            except Exception:
                continue
            if amount <= 0:
                continue
            token_count += 1
            unit_char = unit[0]
            if unit_char == "h":
                total_sec += amount * 3600.0
            elif unit_char == "m":
                total_sec += amount * 60.0
            else:
                total_sec += amount
        if token_count > 0 and total_sec > 0:
            return total_sec

    patterns: List[Tuple[str, float]] = [
        (r"(?:retry|try again|please retry)\s+in\s*([0-9]+(?:\.[0-9]+)?)\s*(?:h|hr|hrs|hour|hours)\b", 3600.0),
        (r"(?:retry|try again|please retry)\s+in\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|secs|second|seconds)\b", 1.0),
        (r"(?:retry|try again|please retry)\s+in\s*([0-9]+(?:\.[0-9]+)?)\s*(?:m|min|mins|minute|minutes)\b", 60.0),
        (r"wait\s*([0-9]+(?:\.[0-9]+)?)\s*(?:h|hr|hrs|hour|hours)\b", 3600.0),
        (r"wait\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|secs|second|seconds)\b", 1.0),
        (r"wait\s*([0-9]+(?:\.[0-9]+)?)\s*(?:m|min|mins|minute|minutes)\b", 60.0),
    ]
    for pattern, multiplier in patterns:
        m = re.search(pattern, body)
        if not m:
            continue
        try:
            amount = float(m.group(1))
        except Exception:
            continue
        if amount > 0:
            return amount * multiplier
    return max(0.0, float(default_wait_sec))


def _extract_retry_seconds_from_response(resp: requests.Response, default_wait_sec: float = 0.0) -> float:
    retry_after = (resp.headers.get("Retry-After") or "").strip()
    if retry_after:
        try:
            value = float(retry_after)
            if value > 0:
                return value
        except Exception:
            pass
    return _extract_retry_seconds_from_text(resp.text or "", default_wait_sec=default_wait_sec)


def _set_gemini_quota_cooldown(wait_sec: float) -> float:
    global _GEMINI_QUOTA_COOLDOWN_UNTIL_TS
    wait_sec = max(0.0, float(wait_sec))
    if wait_sec <= 0:
        return 0.0
    until_ts = time.time() + wait_sec
    if until_ts > _GEMINI_QUOTA_COOLDOWN_UNTIL_TS:
        _GEMINI_QUOTA_COOLDOWN_UNTIL_TS = until_ts
    return wait_sec


def _respect_gemini_quota_cooldown(cfg: Dict[str, Any]) -> None:
    global _GEMINI_QUOTA_COOLDOWN_UNTIL_TS
    now = time.time()
    if _GEMINI_QUOTA_COOLDOWN_UNTIL_TS <= now:
        return
    remaining = _GEMINI_QUOTA_COOLDOWN_UNTIL_TS - now
    max_wait = max(0.0, float(_cfg_get(cfg, "gemini.quota_cooldown_max_wait_sec", 120.0)))
    wait_sec = remaining if max_wait <= 0 else min(remaining, max_wait)
    if wait_sec <= 0:
        return
    print(f"[gemini] quota cooldown active: sleeping {wait_sec:.1f}s.")
    _emit_solver_heartbeat()
    time.sleep(wait_sec)
    _emit_solver_heartbeat()
    if time.time() >= _GEMINI_QUOTA_COOLDOWN_UNTIL_TS - 0.05:
        _GEMINI_QUOTA_COOLDOWN_UNTIL_TS = 0.0


def _extract_zero_quota_model_name(text: str) -> str:
    body = str(text or "")
    match = re.search(r"limit:\s*0\s*,\s*model:\s*([a-zA-Z0-9._-]+)", body, re.IGNORECASE)
    if not match:
        return ""
    return str(match.group(1) or "").strip().lower()


def _zero_quota_model_cache_limit(cfg: Optional[Dict[str, Any]] = None) -> int:
    return max(
        1,
        int(_cfg_get(cfg or {}, "gemini.zero_quota_model_cache_max_entries", 50) or 50),
    )


def _prune_zero_quota_model_cache(
    *,
    now: Optional[float] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> None:
    current_ts = time.time() if now is None else float(now)
    expired = [
        model
        for model, until_ts in list(_GEMINI_ZERO_QUOTA_MODELS.items())
        if float(until_ts or 0.0) <= current_ts
    ]
    for model in expired:
        _GEMINI_ZERO_QUOTA_MODELS.pop(model, None)
    cache_limit = _zero_quota_model_cache_limit(cfg)
    overflow = len(_GEMINI_ZERO_QUOTA_MODELS) - cache_limit
    if overflow <= 0:
        return
    for model, _until_ts in sorted(
        _GEMINI_ZERO_QUOTA_MODELS.items(),
        key=lambda item: (float(item[1] or 0.0), str(item[0] or "")),
    )[:overflow]:
        _GEMINI_ZERO_QUOTA_MODELS.pop(model, None)


def _direct_read_bytes_limit_bytes(cfg: Dict[str, Any]) -> int:
    limit_mb = max(0.0, float(_cfg_get(cfg, "gemini.inline_read_bytes_max_mb", 8.0) or 8.0))
    if limit_mb <= 0:
        return 0
    return int(limit_mb * 1024 * 1024)


def _mark_gemini_model_zero_quota(
    model_name: str,
    wait_sec: float,
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> float:
    model = str(model_name or "").strip().lower()
    if not model:
        return 0.0
    now = time.time()
    _prune_zero_quota_model_cache(now=now, cfg=cfg)
    until_ts = now + max(0.0, float(wait_sec))
    current_until = float(_GEMINI_ZERO_QUOTA_MODELS.get(model, 0.0) or 0.0)
    if until_ts > current_until:
        _GEMINI_ZERO_QUOTA_MODELS[model] = until_ts
    _prune_zero_quota_model_cache(now=now, cfg=cfg)
    return max(0.0, _GEMINI_ZERO_QUOTA_MODELS.get(model, 0.0) - now)


def _is_gemini_model_zero_quota_known(
    model_name: str,
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    model = str(model_name or "").strip().lower()
    if not model:
        return False
    _prune_zero_quota_model_cache(cfg=cfg)
    until_ts = float(_GEMINI_ZERO_QUOTA_MODELS.get(model, 0.0) or 0.0)
    if until_ts <= time.time():
        _GEMINI_ZERO_QUOTA_MODELS.pop(model, None)
        return False
    return True


def _respect_gemini_rate_limit(cfg: Dict[str, Any]) -> None:
    global _LAST_GEMINI_REQUEST_TS, _GEMINI_REQUEST_TIMESTAMPS
    if not bool(_cfg_get(cfg, "gemini.rate_limit_enabled", True)):
        return
    rpm = max(1, int(_cfg_get(cfg, "gemini.rate_limit_requests_per_minute", 9)))
    window_sec = max(5.0, float(_cfg_get(cfg, "gemini.rate_limit_window_sec", 60.0)))
    min_interval_sec = max(0.0, float(_cfg_get(cfg, "gemini.rate_limit_min_interval_sec", 0.0)))

    now = time.time()
    cutoff = now - window_sec
    _GEMINI_REQUEST_TIMESTAMPS = [ts for ts in _GEMINI_REQUEST_TIMESTAMPS if ts >= cutoff]

    wait_sec = 0.0
    if len(_GEMINI_REQUEST_TIMESTAMPS) >= rpm:
        earliest = min(_GEMINI_REQUEST_TIMESTAMPS)
        wait_sec = max(wait_sec, (earliest + window_sec) - now + 0.01)
    if min_interval_sec > 0 and _LAST_GEMINI_REQUEST_TS > 0:
        wait_sec = max(wait_sec, (_LAST_GEMINI_REQUEST_TS + min_interval_sec) - now)

    if wait_sec > 0:
        print(f"[gemini] rate limiter: sleeping {wait_sec:.1f}s (limit={rpm}/{int(window_sec)}s).")
        _emit_solver_heartbeat()
        time.sleep(wait_sec)
        _emit_solver_heartbeat()
        now = time.time()
        cutoff = now - window_sec
        _GEMINI_REQUEST_TIMESTAMPS = [ts for ts in _GEMINI_REQUEST_TIMESTAMPS if ts >= cutoff]

    sent_at = time.time()
    _GEMINI_REQUEST_TIMESTAMPS.append(sent_at)
    _LAST_GEMINI_REQUEST_TS = sent_at


def _normalize_model_name(value: str) -> str:
    model_name = str(value or "").strip().lower()
    if model_name == "gemini-3-pro-preview":
        return "gemini-3.1-pro-preview"
    return model_name


def _model_requires_paid_pool(model_name: str) -> bool:
    return _normalize_model_name(model_name) == "gemini-3.1-pro-preview"


def _model_prefers_free_pool(model_name: str) -> bool:
    return _normalize_model_name(model_name) == "gemini-2.5-flash"


def _is_gen3_model(model_name: str) -> bool:
    normalized = _normalize_model_name(model_name)
    return normalized.startswith("gemini-3")


def _get_strict_paid_solver_key_pool(
    legacy: Any,
    dotenv: Dict[str, str],
    *,
    cfg_api_keys: Optional[List[str]] = None,
    rotation_policy: str = "sticky",
) -> GeminiKeyPool:
    global _STRICT_PAID_GEMINI_POOL, _STRICT_PAID_GEMINI_POOL_SIGNATURE

    def _read(name: str) -> str:
        return str(legacy._solver_config._read_secret(name, dotenv) or "").strip()

    def _append_unique(target: List[str], value: str) -> None:
        text = str(value or "").strip()
        if text and text not in target:
            target.append(text)

    def _append_csv(target: List[str], raw: str) -> None:
        for part in str(raw or "").split(","):
            _append_unique(target, part)

    raw_keys: List[str] = []
    for item in cfg_api_keys or []:
        _append_unique(raw_keys, str(item or "").strip())
    _append_unique(raw_keys, _read("GEMINI_API_KEY_PAID_EPISODE_EVAL"))
    _append_csv(raw_keys, _read("GEMINI_API_KEYS_PAID_POOL"))
    _append_unique(raw_keys, _read("GEMINI_API_KEY_PAID_SECONDARY"))

    if not raw_keys:
        free_keys_present = bool(
            collect_unique_gemini_keys(
                dotenv=dotenv,
                pool_env_names=("GEMINI_API_KEYS_FREE_POOL",),
                single_env_names=(
                    "GEMINI_API_KEY_FREE_OPS",
                    "GEMINI_API_KEY2_FREE_OPS2",
                    "GEMINI_API_KEY_FREE_FALLBACK",
                    "GEMINI_API_KEY_FREE_FALLBACK2",
                ),
            )
        )
        if not free_keys_present:
            _append_unique(raw_keys, _read("GEMINI_API_KEY"))
            _append_unique(raw_keys, _read("GOOGLE_API_KEY"))
            _append_unique(raw_keys, _read("GEMINI_API_KEY_FALLBACK"))
            _append_unique(raw_keys, _read("GOOGLE_API_KEY_FALLBACK"))
            _append_unique(raw_keys, _read("GEMINI_API_KEY_SECONDARY"))
            _append_unique(raw_keys, _read("GOOGLE_API_KEY_SECONDARY"))
    signature: Tuple[Any, ...] = (
        tuple(raw_keys),
        str(rotation_policy or "").strip().lower(),
    )
    if _STRICT_PAID_GEMINI_POOL is None or signature != _STRICT_PAID_GEMINI_POOL_SIGNATURE:
        _STRICT_PAID_GEMINI_POOL = legacy.GeminiKeyPool(
            explicit_key="",
            fallback_key="",
            dotenv={},
            cfg_api_keys=raw_keys,
            rotation_policy=rotation_policy,
            pool_env_names=(),
            single_env_names=(),
        )
        _STRICT_PAID_GEMINI_POOL_SIGNATURE = signature
    else:
        _STRICT_PAID_GEMINI_POOL.set_rotation_policy(rotation_policy)
    return _STRICT_PAID_GEMINI_POOL


def _is_non_retriable_gemini_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    if not msg:
        return False
    fatal_markers = [
        "missing gemini api key",
        "api key not valid",
        "permission denied",
        "unauthorized",
        "forbidden",
    ]
    return any(marker in msg for marker in fatal_markers)


def _gemini_file_state(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    state_obj: Any = payload.get("state", "")
    if isinstance(state_obj, dict):
        for key in ("name", "state"):
            val = state_obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip().upper()
        return ""
    if isinstance(state_obj, str):
        return state_obj.strip().upper()
    return ""


def _wait_for_gemini_file_ready(
    api_key: str,
    file_name: str,
    cfg: Dict[str, Any],
    connect_timeout_sec: int,
    request_timeout_sec: int,
) -> None:
    file_name = (file_name or "").strip()
    if not file_name or not file_name.startswith("files/"):
        return

    timeout_sec = max(5, int(_cfg_get(cfg, "gemini.file_ready_timeout_sec", 120)))
    poll_sec = max(0.5, float(_cfg_get(cfg, "gemini.file_ready_poll_sec", 2.0)))
    deadline = time.time() + timeout_sec
    url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}"

    while True:
        try:
            resp = requests.get(
                url,
                params={"key": api_key},
                timeout=(connect_timeout_sec, request_timeout_sec),
            )
            if resp.status_code == 200:
                payload = resp.json()
                state = _gemini_file_state(payload)
                if not state or state in {"ACTIVE", "READY", "SUCCEEDED"}:
                    return
                if state in {"FAILED", "ERROR", "CANCELLED"}:
                    raise RuntimeError(f"Gemini file processing failed: state={state}")
            elif resp.status_code not in {404, 429, 500, 502, 503, 504}:
                snippet = (resp.text or "")[:200]
                raise RuntimeError(f"Gemini file state check failed HTTP {resp.status_code}: {snippet}")
        except requests.exceptions.RequestException:
            pass

        if time.time() >= deadline:
            raise TimeoutError(f"Gemini file was not ready within {timeout_sec}s: {file_name}")
        time.sleep(poll_sec)


def _normalize_upload_chunk_size(
    requested_chunk_bytes: int,
    size_bytes: int,
    chunk_granularity: int,
) -> int:
    requested = max(64 * 1024, int(requested_chunk_bytes))
    size = max(0, int(size_bytes))
    granularity = max(1, int(chunk_granularity))

    if size <= granularity:
        return size

    chunk = max(requested, granularity)
    if chunk % granularity != 0:
        chunk = (chunk // granularity) * granularity
        if chunk <= 0:
            chunk = granularity
    return chunk


def _legacy_backoff_delay(cfg: Dict[str, Any], attempt: int) -> float:
    legacy = import_module("src.solver.legacy_impl")
    return legacy._compute_backoff_delay(cfg, attempt)


def _upload_video_via_gemini_files_api(
    api_key: str,
    video_file: Path,
    cfg: Dict[str, Any],
    connect_timeout_sec: int,
    request_timeout_sec: int,
) -> Tuple[str, str]:
    if video_file is None or not video_file.exists():
        raise RuntimeError("Video file is missing for Gemini Files API upload.")

    size_bytes = int(video_file.stat().st_size)
    upload_timeout_sec = max(request_timeout_sec, int(_cfg_get(cfg, "gemini.upload_request_timeout_sec", 180)))
    requested_chunk_bytes = max(64 * 1024, int(_cfg_get(cfg, "gemini.upload_chunk_bytes", 8 * 1024 * 1024)))
    chunk_granularity = max(1, int(_cfg_get(cfg, "gemini.upload_chunk_granularity_bytes", 8 * 1024 * 1024)))
    chunk_bytes = _normalize_upload_chunk_size(
        requested_chunk_bytes=requested_chunk_bytes,
        size_bytes=size_bytes,
        chunk_granularity=chunk_granularity,
    )
    if chunk_bytes != requested_chunk_bytes:
        print(
            "[gemini] adjusted upload_chunk_bytes "
            f"from {requested_chunk_bytes} to {chunk_bytes} "
            f"(granularity={chunk_granularity})."
        )
    chunk_retries = max(0, int(_cfg_get(cfg, "gemini.upload_chunk_max_retries", 5)))

    start_url = "https://generativelanguage.googleapis.com/upload/v1beta/files"
    start_headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(size_bytes),
        "X-Goog-Upload-Header-Content-Type": "video/mp4",
        "Content-Type": "application/json",
    }
    start_payload = {"file": {"display_name": video_file.name}}
    start_resp: Optional[requests.Response] = None
    last_start_err = ""
    for attempt in range(chunk_retries + 1):
        try:
            start_resp = requests.post(
                start_url,
                params={"key": api_key},
                headers=start_headers,
                json=start_payload,
                timeout=(connect_timeout_sec, upload_timeout_sec),
            )
            if start_resp.status_code // 100 == 2:
                break
            snippet = (start_resp.text or "")[:300]
            last_start_err = f"HTTP {start_resp.status_code}: {snippet}"
        except requests.exceptions.RequestException as exc:
            last_start_err = str(exc)
        if attempt < chunk_retries:
            delay = _legacy_backoff_delay(cfg, attempt)
            print(f"[gemini] files API start retry {attempt + 1}/{chunk_retries} in {delay:.1f}s")
            time.sleep(delay)
    if start_resp is None or start_resp.status_code // 100 != 2:
        raise RuntimeError(f"Gemini file upload start failed: {last_start_err}")

    upload_url = (
        start_resp.headers.get("X-Goog-Upload-URL")
        or start_resp.headers.get("x-goog-upload-url")
        or ""
    ).strip()
    if not upload_url:
        raise RuntimeError("Gemini file upload start succeeded but upload URL is missing.")

    def _query_uploaded_offset() -> Optional[int]:
        try:
            resp = requests.post(
                upload_url,
                headers={"X-Goog-Upload-Command": "query"},
                timeout=(connect_timeout_sec, upload_timeout_sec),
            )
        except requests.exceptions.RequestException:
            return None
        if resp.status_code // 100 != 2:
            return None
        raw = (
            resp.headers.get("X-Goog-Upload-Size-Received")
            or resp.headers.get("x-goog-upload-size-received")
            or ""
        ).strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    streaming_threshold_mb = max(0.0, float(_cfg_get(cfg, "gemini.streaming_upload_threshold_mb", 0.0)))
    file_size_mb = size_bytes / (1024 * 1024)
    direct_read_limit_bytes = _direct_read_bytes_limit_bytes(cfg)
    direct_read_limit_mb = (
        float(direct_read_limit_bytes) / (1024 * 1024)
        if direct_read_limit_bytes > 0
        else 0.0
    )
    use_streaming = streaming_threshold_mb > 0.0 and file_size_mb >= streaming_threshold_mb
    mem_data: Optional[bytes] = None

    if direct_read_limit_bytes > 0 and size_bytes > direct_read_limit_bytes:
        use_streaming = True
        print(
            f"[gemini] forcing streaming upload mode: {file_size_mb:.1f} MB exceeds "
            f"direct-read threshold {direct_read_limit_mb:.1f} MB."
        )
    elif use_streaming:
        print(
            f"[gemini] streaming upload mode: {file_size_mb:.1f} MB "
            f"(>= {streaming_threshold_mb:.1f} MB threshold). "
            f"RAM per chunk: ~{chunk_bytes / (1024 * 1024):.0f} MB."
        )
    else:
        try:
            mem_data = video_file.read_bytes()
            print(f"[gemini] memory upload mode: {file_size_mb:.1f} MB (fast path).")
        except MemoryError:
            use_streaming = True
            print(
                f"[gemini] MemoryError reading {file_size_mb:.1f} MB into RAM; "
                "falling back to streaming upload."
            )

    def _read_upload_chunk(start: int, end: int) -> bytes:
        if mem_data is not None:
            return mem_data[start:end]
        with open(video_file, "rb") as fh:
            fh.seek(start)
            return fh.read(end - start)

    offset = 0
    upload_resp: Optional[requests.Response] = None
    while offset < size_bytes:
        next_offset = min(size_bytes, offset + chunk_bytes)
        chunk = _read_upload_chunk(offset, next_offset)
        is_final = next_offset >= size_bytes
        command = "upload, finalize" if is_final else "upload"
        sent = False
        resynced = False
        last_chunk_err = ""

        for attempt in range(chunk_retries + 1):
            try:
                resp = requests.post(
                    upload_url,
                    headers={
                        "X-Goog-Upload-Offset": str(offset),
                        "X-Goog-Upload-Command": command,
                        "Content-Type": "video/mp4",
                    },
                    data=chunk,
                    timeout=(connect_timeout_sec, upload_timeout_sec),
                )
                if resp.status_code // 100 == 2:
                    upload_resp = resp if is_final else upload_resp
                    offset = next_offset
                    sent = True
                    break
                snippet = (resp.text or "")[:220]
                last_chunk_err = f"HTTP {resp.status_code}: {snippet}"
            except requests.exceptions.RequestException as exc:
                last_chunk_err = str(exc)

            if attempt < chunk_retries:
                remote_offset = _query_uploaded_offset()
                if remote_offset is not None and remote_offset > offset:
                    offset = min(remote_offset, size_bytes)
                    resynced = True
                    print(f"[gemini] files upload resumed at offset {offset}/{size_bytes}.")
                    break
                delay = _legacy_backoff_delay(cfg, attempt)
                print(f"[gemini] files chunk upload retry {attempt + 1}/{chunk_retries} at offset {offset} in {delay:.1f}s")
                time.sleep(delay)

        if sent or resynced:
            chunk = b""
            continue
        chunk = b""
        raise RuntimeError(f"Gemini file chunk upload failed at offset {offset}: {last_chunk_err}")

    mem_data = None

    if upload_resp is None:
        raise RuntimeError("Gemini file upload finalize response missing.")

    try:
        upload_payload = upload_resp.json()
    except Exception as exc:
        raise RuntimeError("Gemini file upload finalize returned non-JSON response.") from exc

    if isinstance(upload_payload, dict) and isinstance(upload_payload.get("file"), dict):
        file_info = upload_payload["file"]
    elif isinstance(upload_payload, dict):
        file_info = upload_payload
    else:
        raise RuntimeError("Gemini file upload finalize returned unexpected payload shape.")

    file_uri = str(file_info.get("uri", "")).strip()
    file_name = str(file_info.get("name", "")).strip()
    if not file_uri:
        raise RuntimeError("Gemini file upload succeeded but file URI is missing.")

    _wait_for_gemini_file_ready(
        api_key=api_key,
        file_name=file_name,
        cfg=cfg,
        connect_timeout_sec=connect_timeout_sec,
        request_timeout_sec=request_timeout_sec,
    )
    return file_uri, file_name


def _cleanup_gemini_uploaded_file(
    api_key: str,
    file_name: str,
    cfg: Dict[str, Any],
    connect_timeout_sec: int = 10,
) -> bool:
    if not file_name or not api_key:
        return False
    try:
        base_url = str(_cfg_get(cfg, "gemini.base_url", "https://generativelanguage.googleapis.com")).rstrip("/")
        api_version = str(_cfg_get(cfg, "gemini.api_version", "v1beta")).strip()
        url = f"{base_url}/{api_version}/{file_name}"
        resp = requests.delete(
            url,
            params={"key": api_key},
            timeout=(connect_timeout_sec, 30),
        )
        if resp.status_code // 100 == 2:
            print(f"[gemini] cleaned up uploaded file: {file_name}")
            return True
        print(f"[gemini] file cleanup returned HTTP {resp.status_code} for {file_name} (non-fatal).")
        return False
    except Exception as exc:
        print(f"[gemini] file cleanup failed for {file_name}: {exc} (non-fatal).")
        return False


def _sweep_stale_gemini_files(api_key: str, cfg: Dict[str, Any]) -> int:
    _ = api_key, cfg
    return 0


def _is_gemini_quota_error_text(text: str) -> bool:
    body = (text or "").lower()
    quota_markers = (
        "quota exceeded",
        "exceeded your current quota",
        "free_tier",
        "resource_exhausted",
        "generate_content_free_tier_requests",
    )
    return any(marker in body for marker in quota_markers)


def _is_gemini_quota_exceeded_429(resp: requests.Response) -> bool:
    if resp.status_code != 429:
        return False
    return _is_gemini_quota_error_text(resp.text or "")


def _is_gemini_quota_error(exc: Exception) -> bool:
    return _is_gemini_quota_error_text(str(exc or ""))


def _is_gemini_api_key_invalid_text(text: str) -> bool:
    body = (text or "").lower()
    if not body:
        return False
    markers = (
        "api_key_invalid",
        "api key not found",
        "please pass a valid api key",
        "\"reason\": \"api_key_invalid\"",
    )
    return any(marker in body for marker in markers)


def _is_gemini_api_key_invalid_response(resp: requests.Response) -> bool:
    if resp.status_code not in {400, 401, 403}:
        return False
    return _is_gemini_api_key_invalid_text(resp.text or "")


def _is_gemini_availability_error_text(text: str) -> bool:
    body = (text or "").lower()
    if not body:
        return False
    if "high demand" in body:
        return True
    if "status\": \"unavailable" in body or "status': 'unavailable" in body:
        return True
    if "currently experiencing high demand" in body:
        return True
    return "503" in body and "unavailable" in body


def _is_gemini_availability_error(exc: Exception) -> bool:
    return _is_gemini_availability_error_text(str(exc or ""))


def _build_gemini_generation_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    temperature = float(_cfg_get(cfg, "gemini.temperature", 0.0))
    candidate_count = max(1, int(_cfg_get(cfg, "gemini.candidate_count", 1)))
    top_p_raw = _cfg_get(cfg, "gemini.top_p", None)
    top_k_raw = _cfg_get(cfg, "gemini.top_k", None)
    max_output_tokens_raw = _cfg_get(cfg, "gemini.max_output_tokens", None)

    gen_cfg: Dict[str, Any] = {
        "temperature": temperature,
        "responseMimeType": "application/json",
        "candidateCount": candidate_count,
    }
    try:
        if top_p_raw is not None and str(top_p_raw).strip() != "":
            top_p = float(top_p_raw)
            if top_p > 0:
                gen_cfg["topP"] = top_p
    except Exception:
        pass
    try:
        if top_k_raw is not None and str(top_k_raw).strip() != "":
            top_k = int(top_k_raw)
            if top_k > 0:
                gen_cfg["topK"] = top_k
    except Exception:
        pass
    try:
        if max_output_tokens_raw is not None and str(max_output_tokens_raw).strip() != "":
            max_tokens = int(max_output_tokens_raw)
            if max_tokens > 0:
                gen_cfg["maxOutputTokens"] = max_tokens
    except Exception:
        pass
    return gen_cfg


def _log_gemini_usage(
    cfg: Dict[str, Any],
    usage_meta: Dict[str, Any],
    *,
    model: str,
    mode: str,
    key_source: str,
    key_class: str = "",
    stage_name: str = "",
    task_id: str = "",
) -> None:
    try:
        prompt_tokens = int(usage_meta.get("promptTokenCount", 0) or 0)
    except Exception:
        prompt_tokens = 0
    try:
        output_tokens = int(usage_meta.get("candidatesTokenCount", 0) or 0)
    except Exception:
        output_tokens = 0
    try:
        total_tokens = int(usage_meta.get("totalTokenCount", 0) or 0)
    except Exception:
        total_tokens = prompt_tokens + output_tokens

    if prompt_tokens <= 0 and output_tokens <= 0 and total_tokens <= 0:
        return

    est_cost = estimate_cost_usd(
        cfg,
        model,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
    print(
        "[gemini] usage: "
        f"prompt={prompt_tokens} output={output_tokens} total={total_tokens} "
        f"est_cost=${est_cost:.6f}"
    )

    out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    usage_log_name = str(_cfg_get(cfg, "gemini.usage_log_file", "gemini_usage.jsonl")).strip() or "gemini_usage.jsonl"
    usage_log_path = out_dir / usage_log_name
    line = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "mode": mode,
        "key_source": key_source,
        "key_class": str(key_class or "").strip() or "unknown",
        "stage_name": str(stage_name or "").strip() or "unknown",
        "task_id": str(task_id or "").strip(),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(est_cost, 8),
    }
    try:
        with usage_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        pass


def call_gemini_labels(
    cfg: Dict[str, Any],
    prompt: str,
    video_file: Optional[Path] = None,
    segment_count: int = 0,
    model_override: str = "",
    task_id: str = "",
    task_state: Optional[Dict[str, Any]] = None,
    stage_name: str = "labeling",
) -> Dict[str, Any]:
    legacy = import_module("src.solver.legacy_impl")

    model = str(
        model_override
        or resolve_stage_model(cfg, stage_name, _cfg_get(cfg, "gemini.model", "gemini-2.5-flash"))
    ).strip()
    cfg_api_keys_raw = _cfg_get(cfg, "gemini.api_keys", [])
    cfg_api_keys = (
        [str(item or "").strip() for item in cfg_api_keys_raw if str(item or "").strip()]
        if isinstance(cfg_api_keys_raw, list)
        else []
    )
    rotation_policy = str(_cfg_get(cfg, "gemini.rotation_policy", "sticky") or "").strip().lower() or "sticky"
    dotenv = legacy._solver_config._load_dotenv(Path(".env"))
    paid_pool = _get_strict_paid_solver_key_pool(
        legacy,
        dotenv,
        cfg_api_keys=cfg_api_keys,
        rotation_policy=rotation_policy,
    )
    free_pool = legacy._solver_config._get_global_free_solver_key_pool(
        dotenv,
        cfg_api_keys=cfg_api_keys,
        rotation_policy=rotation_policy,
    )
    pinned_key_class = str(
        (task_state.get("episode_key_class_pin", "") if isinstance(task_state, dict) else "")
    ).strip().lower()
    use_paid_only = _model_requires_paid_pool(model)
    prefer_free_pool = _model_prefers_free_pool(model)

    active_pool = paid_pool
    active_key_class = "paid"
    if pinned_key_class == "paid":
        active_pool = paid_pool
        active_key_class = "paid"
    elif pinned_key_class == "free" and free_pool.keys:
        active_pool = free_pool
        active_key_class = "free"
    elif use_paid_only:
        active_pool = paid_pool
        active_key_class = "paid"
    elif prefer_free_pool and free_pool.keys:
        active_pool = free_pool
        active_key_class = "free"
    elif paid_pool.keys:
        active_pool = paid_pool
        active_key_class = "paid"
    elif free_pool.keys:
        active_pool = free_pool
        active_key_class = "free"

    primary_api_key = active_pool.begin_request()
    if not primary_api_key:
        if active_key_class == "paid":
            raise RuntimeError(
                "Missing paid Gemini key for this stage. Set GEMINI_API_KEYS_PAID_POOL or "
                "GEMINI_API_KEY_PAID_EPISODE_EVAL in .env."
            )
        raise RuntimeError(
            "Missing free Gemini key for this stage. Set GEMINI_API_KEYS_FREE_POOL or "
            "GEMINI_API_KEY_FREE_OPS in .env."
        )
    if active_key_class == "paid":
        legacy._global_solver_key_pool = active_pool

    quota_fallback_enabled = bool(_cfg_get(cfg, "gemini.quota_fallback_enabled", False))
    quota_fallback_max_uses_per_run = max(0, int(_cfg_get(cfg, "gemini.quota_fallback_max_uses_per_run", 1)))
    if not quota_fallback_enabled or not paid_pool.keys:
        quota_fallback_max_uses_per_run = 0

    system_instruction = legacy._resolve_system_instruction(cfg)
    max_retries = max(0, int(_cfg_get(cfg, "gemini.max_retries", 3)))
    generation_config_template = legacy._build_gemini_generation_config(cfg)
    connect_timeout_sec = max(5, int(_cfg_get(cfg, "gemini.connect_timeout_sec", 30)))
    request_timeout_sec = max(30, int(_cfg_get(cfg, "gemini.request_timeout_sec", 420)))
    require_video = bool(_cfg_get(cfg, "gemini.require_video", False))
    attach_video = bool(_cfg_get(cfg, "gemini.attach_video", True))
    video_attach_block_reason = ""
    skip_video_when_segments_le = max(0, int(_cfg_get(cfg, "gemini.skip_video_when_segments_le", 0)))
    allow_text_fallback = bool(_cfg_get(cfg, "gemini.allow_text_only_fallback_on_network_error", True))
    video_transport = str(_cfg_get(cfg, "gemini.video_transport", "files_api")).strip().lower() or "files_api"
    files_api_fallback_to_inline = bool(_cfg_get(cfg, "gemini.files_api_fallback_to_inline", False))
    max_inline_video_mb = float(_cfg_get(cfg, "gemini.max_inline_video_mb", 20.0))
    inline_retry_targets_mb = legacy._parse_float_list(
        _cfg_get(cfg, "gemini.inline_retry_target_mb", [4.0, 2.5, 1.5, 1.0]),
        [4.0, 2.5, 1.5, 1.0],
    )
    if require_video and not attach_video:
        raise RuntimeError("Invalid config: gemini.require_video=true but gemini.attach_video=false.")

    if (
        attach_video
        and not require_video
        and skip_video_when_segments_le > 0
        and segment_count > 0
        and segment_count <= skip_video_when_segments_le
    ):
        attach_video = False
        video_attach_block_reason = "short_episode_threshold"
        print(
            "[gemini] skipping video attachment for short episode: "
            f"segment_count={segment_count} <= {skip_video_when_segments_le}."
        )
    elif not attach_video:
        video_attach_block_reason = "disabled_by_config"

    active_api_key = primary_api_key
    active_key_name = f"key_{active_pool.get_current_index() + 1}"
    episode_key_class_pin = "paid" if pinned_key_class == "paid" else ""
    can_use_secondary = (
        active_key_class == "paid"
        and paid_pool.has_multiple_keys()
        and quota_fallback_max_uses_per_run > 0
    )
    retry_switch_key_on_503 = bool(_cfg_get(cfg, "gemini.retry_switch_key_on_503", True))
    retry_switch_key_max_uses_per_request = max(
        0,
        int(_cfg_get(cfg, "gemini.retry_switch_key_max_uses_per_request", 2) or 2),
    )
    retry_switch_key_cooldown_sec = max(
        0.0,
        float(_cfg_get(cfg, "gemini.retry_switch_key_cooldown_sec", 90.0) or 90.0),
    )
    temporary_switch_uses = 0

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    video_parts: List[Dict[str, Any]] = []
    frame_parts: List[Dict[str, Any]] = []
    reference_frame_bytes = 0
    reference_frame_count_used = 0
    video_transport_used = "none"
    uploaded_gemini_file_names: List[str] = []
    if system_instruction:
        print(f"[gemini] using system instruction ({len(system_instruction)} chars).")

    def _rebuild_parts() -> None:
        nonlocal parts
        parts = [{"text": prompt}]
        if video_parts:
            if len(video_parts) > 1:
                parts.append(
                    {
                        "text": (
                            f"Video context is split into {len(video_parts)} sequential chunks. "
                            "Use all chunks together as one continuous episode timeline."
                        )
                    }
                )
            parts.extend(video_parts)
            if frame_parts:
                parts.extend(frame_parts)

    prepared_video_file = video_file
    prepared_video_files: List[Path] = []
    if attach_video and prepared_video_file is not None and prepared_video_file.exists():
        prepared_video_file = legacy._maybe_optimize_video_for_upload(prepared_video_file, cfg)
        split_files = legacy._split_video_for_upload(prepared_video_file, cfg)
        if split_files:
            prepared_video_files = split_files
            print(f"[video] split upload enabled: using {len(prepared_video_files)} video chunks.")
        elif prepared_video_file is not None and prepared_video_file.exists():
            prepared_video_files = [prepared_video_file]
    source_video_for_retry = video_file if (video_file is not None and video_file.exists()) else prepared_video_file
    split_inline_total_max_mb = float(_cfg_get(cfg, "gemini.split_upload_inline_total_max_mb", 12.0))
    inline_retry_target_idx = 0

    def _build_files_api_video_parts(api_key: str) -> List[Dict[str, Any]]:
        built: List[Dict[str, Any]] = []
        total = len(prepared_video_files)
        for idx, vfile in enumerate(prepared_video_files, start=1):
            file_uri, file_name = legacy._upload_video_via_gemini_files_api(
                api_key=api_key,
                video_file=vfile,
                cfg=cfg,
                connect_timeout_sec=connect_timeout_sec,
                request_timeout_sec=request_timeout_sec,
            )
            built.append({"file_data": {"mime_type": "video/mp4", "file_uri": file_uri}})
            if file_name:
                uploaded_gemini_file_names.append(file_name)
            if total > 1:
                print(f"[gemini] attached video chunk {idx}/{total} via Files API: {file_name or file_uri}")
            else:
                print(f"[gemini] attached video via Files API: {file_name or file_uri}")
        return built

    def _build_inline_video_parts(files: List[Path]) -> Optional[List[Dict[str, Any]]]:
        if not files:
            return None
        direct_read_limit_bytes = _direct_read_bytes_limit_bytes(cfg)
        direct_read_limit_mb = (
            float(direct_read_limit_bytes) / (1024 * 1024)
            if direct_read_limit_bytes > 0
            else 0.0
        )
        try:
            total_mb = sum(float(p.stat().st_size) for p in files) / (1024 * 1024)
        except Exception:
            total_mb = 0.0
        if len(files) > 1 and split_inline_total_max_mb > 0 and total_mb > split_inline_total_max_mb:
            return None
        built: List[Dict[str, Any]] = []
        for p in files:
            try:
                part_size_bytes = int(p.stat().st_size)
                part_mb = part_size_bytes / (1024 * 1024)
            except Exception:
                return None
            if part_mb > max_inline_video_mb:
                return None
            if direct_read_limit_bytes > 0 and part_size_bytes > direct_read_limit_bytes:
                print(
                    f"[gemini] inline video skipped for {p.name}: {part_mb:.1f} MB exceeds "
                    f"direct-read threshold {direct_read_limit_mb:.1f} MB."
                )
                return None
            with p.open("rb") as handle:
                raw_bytes = handle.read()
            data = base64.b64encode(raw_bytes).decode("ascii")
            raw_bytes = b""
            built.append({"inline_data": {"mime_type": "video/mp4", "data": data}})
        return built

    def _switch_to_smaller_inline_video() -> bool:
        nonlocal prepared_video_file, prepared_video_files, video_parts, include_video, fallback_used, inline_retry_target_idx
        nonlocal frame_parts, reference_frame_bytes, reference_frame_count_used
        if len(prepared_video_files) != 1:
            return False
        if source_video_for_retry is None or not source_video_for_retry.exists():
            return False
        if prepared_video_file is None or not prepared_video_file.exists():
            return False

        current_size = int(prepared_video_file.stat().st_size)
        while inline_retry_target_idx < len(inline_retry_targets_mb):
            target_mb = float(inline_retry_targets_mb[inline_retry_target_idx])
            inline_retry_target_idx += 1
            current_mb = current_size / (1024 * 1024)
            if current_mb <= target_mb + 0.05:
                continue
            cfg_retry = legacy._deep_merge(
                cfg,
                {
                    "gemini": {
                        "optimize_video_only_if_larger_mb": 0.0,
                        "optimize_video_target_mb": target_mb,
                    }
                },
            )
            candidate = legacy._maybe_optimize_video_for_upload(source_video_for_retry, cfg_retry)
            if candidate is None or not candidate.exists():
                continue
            try:
                candidate_size = int(candidate.stat().st_size)
            except Exception:
                continue
            if candidate_size <= 0 or candidate_size >= current_size:
                continue
            prepared_video_file = candidate
            prepared_video_files = [prepared_video_file]
            built_inline = _build_inline_video_parts(prepared_video_files)
            if not built_inline:
                continue
            video_parts = built_inline
            include_video = True
            fallback_used = True
            frame_source = source_video_for_retry if source_video_for_retry.exists() else prepared_video_file
            frame_parts, reference_frame_bytes = legacy._extract_reference_frame_inline_parts(
                frame_source,
                cfg,
                trigger_video_mb=(candidate_size / (1024 * 1024)),
            )
            reference_frame_count_used = len(frame_parts)
            _rebuild_parts()
            print(
                f"[gemini] retrying with smaller inline video "
                f"({candidate_size / (1024 * 1024):.1f} MB, target<={target_mb:.1f} MB)."
            )
            return True
        return False

    if attach_video and prepared_video_files:
        total_size_mb = 0.0
        try:
            total_size_mb = sum(float(p.stat().st_size) for p in prepared_video_files) / (1024 * 1024)
        except Exception:
            total_size_mb = 0.0
        wants_files_api = video_transport in {"auto", "files_api", "files"}
        inline_allowed = video_transport in {"auto", "inline"} or (wants_files_api and files_api_fallback_to_inline)

        if wants_files_api:
            try:
                video_parts = _build_files_api_video_parts(active_api_key)
                video_transport_used = "files_api-multi" if len(video_parts) > 1 else "files_api"
            except Exception as exc:
                print(f"[gemini] files API upload failed: {exc}")
                if not inline_allowed or not files_api_fallback_to_inline:
                    if require_video:
                        raise
                    print("[gemini] continuing without video after Files API failure.")
                else:
                    print("[gemini] falling back to inline video attachment after Files API failure.")

        if not video_parts and inline_allowed:
            if len(prepared_video_files) == 1 and total_size_mb > max_inline_video_mb:
                msg = (
                    f"Video is {total_size_mb:.1f} MB which exceeds max_inline_video_mb={max_inline_video_mb:.1f}. "
                    "Increase gemini.max_inline_video_mb or provide smaller video."
                )
                if require_video:
                    raise RuntimeError(msg)
                print(f"[video] {msg} Proceeding without attachment.")
            else:
                built_inline = _build_inline_video_parts(prepared_video_files)
                if built_inline:
                    video_parts = built_inline
                    video_transport_used = "inline-multi" if len(video_parts) > 1 else "inline"
                    if len(video_parts) > 1:
                        print(
                            f"[gemini] attached split video inline ({len(video_parts)} parts, "
                            f"{total_size_mb:.1f} MB total)."
                        )
                    else:
                        print(f"[gemini] attached video inline ({total_size_mb:.1f} MB).")
                elif require_video:
                    raise RuntimeError(
                        "Split video inline payload exceeds limits; reduce split chunk size or use Files API."
                    )
                else:
                    print("[gemini] split inline video is too large; continuing without video attachment.")
    else:
        if not attach_video and video_file is not None:
            if video_attach_block_reason == "short_episode_threshold":
                print("[gemini] video attachment skipped for this request due to short-episode threshold.")
            else:
                print("[gemini] video attachment disabled by config (gemini.attach_video=false).")
        elif require_video:
            raise RuntimeError("gemini.require_video=true but no downloadable video file was prepared.")
    include_video = bool(video_parts)
    if include_video:
        try:
            if prepared_video_file is not None and prepared_video_file.exists():
                trigger_mb = prepared_video_file.stat().st_size / (1024 * 1024)
            elif prepared_video_files:
                trigger_mb = sum(float(p.stat().st_size) for p in prepared_video_files) / (1024 * 1024)
            else:
                trigger_mb = 0.0
        except Exception:
            trigger_mb = 0.0
        frame_source = (
            source_video_for_retry if (source_video_for_retry is not None and source_video_for_retry.exists()) else prepared_video_file
        )
        if (frame_source is None or not frame_source.exists()) and prepared_video_files:
            frame_source = prepared_video_files[0]
        if frame_source is not None and frame_source.exists():
            frame_parts, reference_frame_bytes = legacy._extract_reference_frame_inline_parts(
                frame_source,
                cfg,
                trigger_video_mb=trigger_mb,
            )
            reference_frame_count_used = len(frame_parts)
            if reference_frame_count_used > 0:
                print(
                    f"[gemini] attached {reference_frame_count_used} reference frame(s) "
                    f"({reference_frame_bytes / 1024:.0f} KB total)."
                )
    _rebuild_parts()

    last_error = ""
    used_video_in_success = False
    fallback_used = False

    def _rebuild_video_parts_for_active_key(reason_label: str) -> None:
        nonlocal include_video, video_parts, video_transport_used, fallback_used
        nonlocal prepared_video_files
        nonlocal frame_parts, reference_frame_bytes, reference_frame_count_used
        if not include_video or not video_transport_used.startswith("files_api"):
            return
        rebuilt: List[Dict[str, Any]] = []
        try:
            rebuilt = _build_files_api_video_parts(active_api_key)
        except Exception as exc:
            print(f"[gemini] {reason_label} Files API re-upload failed: {exc}")
            rebuilt = []
        if rebuilt:
            video_parts = rebuilt
            include_video = True
            video_transport_used = "files_api-multi" if len(video_parts) > 1 else "files_api"
            _rebuild_parts()
            print(f"[gemini] rebuilt Files API video attachment after {reason_label} key switch.")
            return
        inline_parts = _build_inline_video_parts(prepared_video_files)
        if inline_parts:
            video_parts = inline_parts
            include_video = True
            video_transport_used = "inline-multi" if len(video_parts) > 1 else "inline"
            _rebuild_parts()
            print(f"[gemini] switched to inline video payload after {reason_label} key switch.")
            return
        if not require_video and allow_text_fallback:
            include_video = False
            video_parts = []
            frame_parts = []
            reference_frame_bytes = 0
            reference_frame_count_used = 0
            _rebuild_parts()
            fallback_used = True
            print(f"[gemini] continuing in text-only mode after {reason_label} key switch.")

    def _switch_to_secondary_key_for_quota() -> bool:
        nonlocal active_api_key, active_key_name, active_pool, active_key_class, fallback_used
        if not quota_fallback_enabled or not can_use_secondary or active_key_class != "paid":
            return False
        if legacy._GEMINI_FALLBACK_USES >= quota_fallback_max_uses_per_run:
            return False
        if paid_pool.switch_to_next():
            active_pool = paid_pool
            active_api_key = paid_pool.get_current_key()
            active_key_name = f"key_{paid_pool.get_current_index() + 1}"
            legacy._GEMINI_FALLBACK_USES += 1
            fallback_used = True
            print(
                f"[gemini] quota exhausted; switching to {active_key_name} "
                f"({legacy._GEMINI_FALLBACK_USES}/{quota_fallback_max_uses_per_run}) for this request."
            )
        else:
            return False

        _rebuild_video_parts_for_active_key("quota")
        return True

    def _switch_to_paid_pool_for_quota() -> bool:
        nonlocal active_api_key, active_key_name, active_pool, active_key_class
        nonlocal episode_key_class_pin, can_use_secondary, fallback_used
        if not quota_fallback_enabled or active_key_class != "free":
            return False
        if legacy._GEMINI_FALLBACK_USES >= quota_fallback_max_uses_per_run:
            return False
        next_key = paid_pool.begin_request()
        if not next_key:
            return False
        active_pool = paid_pool
        active_key_class = "paid"
        active_api_key = next_key
        active_key_name = f"key_{paid_pool.get_current_index() + 1}"
        episode_key_class_pin = "paid"
        legacy._GEMINI_FALLBACK_USES += 1
        can_use_secondary = paid_pool.has_multiple_keys() and quota_fallback_max_uses_per_run > 0
        fallback_used = True
        print(
            "[gemini] free-tier quota exhausted; switching this episode request to paid pool "
            f"({legacy._GEMINI_FALLBACK_USES}/{quota_fallback_max_uses_per_run})."
        )
        _rebuild_video_parts_for_active_key("paid-quota-fallback")
        return True

    def _switch_to_next_key_for_unavailable(
        *,
        log_label: str = "temporary 503/unavailable",
        reason_label: str = "temporary-error",
        honor_retry_switch_config: bool = True,
        use_temporary_budget: bool = True,
    ) -> bool:
        nonlocal active_api_key, active_key_name, temporary_switch_uses, active_pool
        if honor_retry_switch_config and not retry_switch_key_on_503:
            return False
        if not active_pool.has_multiple_keys():
            return False
        if use_temporary_budget and temporary_switch_uses >= retry_switch_key_max_uses_per_request:
            return False
        if not active_pool.switch_to_next():
            return False
        active_api_key = active_pool.get_current_key()
        active_key_name = f"key_{active_pool.get_current_index() + 1}"
        if use_temporary_budget:
            temporary_switch_uses += 1
            usage_note = f" ({temporary_switch_uses}/{retry_switch_key_max_uses_per_request})"
        else:
            usage_note = ""
        print(f"[gemini] {log_label}; switching to {active_key_name}{usage_note} for retry.")
        _rebuild_video_parts_for_active_key(reason_label)
        return True

    for attempt in range(max_retries + 1):
        _emit_solver_heartbeat()
        mode = "with-video" if include_video else "text-only"
        print(
            f"[gemini] request attempt {attempt + 1}/{max_retries + 1} "
            f"(model={model}, mode={mode}, key={active_key_name})"
        )
        generation_config = dict(generation_config_template)
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        try:
            legacy._respect_gemini_quota_cooldown(cfg)
            legacy._respect_gemini_rate_limit(cfg)
            headers = {"Content-Type": "application/json", "X-goog-api-key": active_api_key}
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(connect_timeout_sec, request_timeout_sec),
            )
            _emit_solver_heartbeat()
        except requests.exceptions.RequestException as exc:
            _emit_solver_heartbeat()
            last_error = f"Gemini network error: {exc}"
            if include_video and video_transport_used.startswith("inline") and _switch_to_smaller_inline_video():
                continue
            if include_video and not require_video and allow_text_fallback:
                include_video = False
                video_parts = []
                frame_parts = []
                reference_frame_bytes = 0
                reference_frame_count_used = 0
                _rebuild_parts()
                fallback_used = True
                print("[gemini] network error while sending video; switching to text-only fallback.")
                continue
            if attempt < max_retries:
                delay = legacy._compute_backoff_delay(cfg, attempt)
                print(f"[gemini] network error, retrying in {delay:.1f}s")
                _emit_solver_heartbeat()
                time.sleep(delay)
                _emit_solver_heartbeat()
                continue
            break

        if resp.status_code == 200:
            print("[gemini] response received (HTTP 200).")
            used_video_in_success = include_video
            raw_json = resp.json()
            if not _gemini_response_has_nonempty_text_candidate(raw_json):
                last_error = f"Gemini HTTP 200 with empty candidate text: {raw_json}"
                if attempt < max_retries:
                    delay = legacy._compute_backoff_delay(cfg, attempt)
                    print(f"[gemini] empty model text in HTTP 200 response, retrying in {delay:.1f}s")
                    _emit_solver_heartbeat()
                    time.sleep(delay)
                    _emit_solver_heartbeat()
                    continue
                break
            parsed = legacy._parse_gemini_response(raw_json)
            usage_meta = raw_json.get("usageMetadata", {}) if isinstance(raw_json, dict) else {}
            mode_name = "with-video" if include_video else "text-only"
            estimated_cost_usd = estimate_cost_from_usage(cfg, model, usage_meta if isinstance(usage_meta, dict) else {})
            legacy._log_gemini_usage(
                cfg,
                model=model,
                mode=mode_name,
                usage_meta=usage_meta,
                key_source=active_key_name,
                key_class=active_key_class,
                stage_name=stage_name,
                task_id=task_id,
            )
            if isinstance(parsed, dict):
                parsed["_meta"] = {
                    "video_attached": bool(used_video_in_success),
                    "mode": "with-video" if used_video_in_success else "text-only",
                    "fallback_used": bool(fallback_used),
                    "video_transport": video_transport_used,
                    "video_parts_count": int(len(video_parts)) if used_video_in_success else 0,
                    "reference_frames_attached": int(reference_frame_count_used),
                    "reference_frames_total_kb": round(reference_frame_bytes / 1024, 1),
                    "api_key_source": active_key_name,
                    "api_key_index": int(active_pool.get_current_index() + 1),
                    "api_key_class": active_key_class,
                    "rotation_policy": rotation_policy,
                    "model": model,
                    "stage_name": str(stage_name or "").strip() or "labeling",
                    "estimated_cost_usd": round(float(estimated_cost_usd or 0.0), 8),
                    "episode_key_class_pin": episode_key_class_pin,
                    "usage": usage_meta if isinstance(usage_meta, dict) else {},
                    "uploaded_file_names": list(uploaded_gemini_file_names),
                }
            return parsed

        last_error = f"Gemini HTTP {resp.status_code}: {resp.text[:800]}"
        if _is_gemini_api_key_invalid_response(resp) and attempt < max_retries:
            failed_key = str(active_api_key or "").strip()
            invalid_key_cooldown_sec = max(retry_switch_key_cooldown_sec, 21600.0)
            if failed_key:
                active_pool.mark_key_temporarily_unavailable(failed_key, invalid_key_cooldown_sec)
                print(
                    f"[gemini] cooling down invalid API key for {invalid_key_cooldown_sec:.0f}s "
                    f"after HTTP {resp.status_code}."
                )
            if _switch_to_next_key_for_unavailable(
                log_label="invalid API key",
                reason_label="invalid-key",
                honor_retry_switch_config=False,
                use_temporary_budget=False,
            ):
                continue
        if legacy._is_gemini_quota_exceeded_429(resp):
            quota_default_wait = max(1.0, float(_cfg_get(cfg, "gemini.quota_retry_default_wait_sec", 12.0)))
            quota_wait_sec = legacy._extract_retry_seconds_from_response(resp, default_wait_sec=quota_default_wait)
            zero_quota_model = _extract_zero_quota_model_name(resp.text or "") or str(model or "").strip().lower()
            if "limit: 0" in str(resp.text or ""):
                zero_quota_wait = max(
                    quota_wait_sec,
                    float(_cfg_get(cfg, "gemini.zero_quota_model_cooldown_sec", 21600.0) or 21600.0),
                )
                remaining_zero_quota = _mark_gemini_model_zero_quota(
                    zero_quota_model,
                    zero_quota_wait,
                    cfg=cfg,
                )
                print(
                    "[gemini] marking model as zero-quota unavailable: "
                    f"{zero_quota_model or 'unknown'} for ~{remaining_zero_quota:.0f}s."
                )
            if _switch_to_paid_pool_for_quota():
                continue
            if _switch_to_secondary_key_for_quota():
                continue
            cooldown_wait = legacy._set_gemini_quota_cooldown(quota_wait_sec)
            retry_on_quota_429 = bool(_cfg_get(cfg, "gemini.retry_on_quota_429", False))
            if retry_on_quota_429 and attempt < max_retries:
                delay = max(cooldown_wait, legacy._compute_backoff_delay(cfg, attempt))
                print(f"[gemini] quota error 429, retrying in {delay:.1f}s")
                _emit_solver_heartbeat()
                time.sleep(delay)
                _emit_solver_heartbeat()
                continue
            print(
                "[gemini] quota error 429 detected; skipping extra retries for this request "
                f"(cooldown={cooldown_wait:.1f}s)."
            )
            break
        if include_video and not require_video and allow_text_fallback and resp.status_code in {400, 408, 413, 422}:
            include_video = False
            video_parts = []
            frame_parts = []
            reference_frame_bytes = 0
            reference_frame_count_used = 0
            _rebuild_parts()
            fallback_used = True
            print(f"[gemini] HTTP {resp.status_code} while using video; switching to text-only fallback.")
            continue
        if include_video and video_transport_used.startswith("inline") and resp.status_code in {400, 408, 413, 422}:
            if _switch_to_smaller_inline_video():
                continue
        if resp.status_code == 503 and attempt < max_retries:
            resp_text_l = str(resp.text or "").lower()
            if "unavailable" in resp_text_l or "high demand" in resp_text_l:
                failed_key = str(active_api_key or "").strip()
                if failed_key:
                    active_pool.mark_key_temporarily_unavailable(failed_key, retry_switch_key_cooldown_sec)
                    print(
                        f"[gemini] cooling down failed key for {retry_switch_key_cooldown_sec:.0f}s "
                        f"after 503/unavailable."
                    )
                _switch_to_next_key_for_unavailable()
        if resp.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            delay = legacy._compute_backoff_delay(cfg, attempt)
            print(f"[gemini] temporary error {resp.status_code}, retrying in {delay:.1f}s")
            _emit_solver_heartbeat()
            time.sleep(delay)
            _emit_solver_heartbeat()
            continue
        break
    raise RuntimeError(last_error or "Gemini request failed")


def _request_labels_with_optional_segment_chunking(
    cfg: Dict[str, Any],
    segments: List[Dict[str, Any]],
    prompt: str,
    video_file: Optional[Path],
    allow_operations: bool,
    model_override: str = "",
    task_id: str = "",
    task_state: Optional[Dict[str, Any]] = None,
    stage_name: str = "labeling",
) -> Dict[str, Any]:
    legacy = import_module("src.solver.legacy_impl")

    # If the original video file was optimized and deleted to save space,
    # the orchestrator might pass a stale path. Auto-correct to _upload_opt.mp4.
    if video_file is not None and not video_file.exists():
        opt_path = video_file.with_name(video_file.stem + "_upload_opt.mp4")
        if opt_path.exists():
            video_file = opt_path

    chunking_enabled = bool(_cfg_get(cfg, "run.segment_chunking_enabled", True))
    min_segments_for_chunking = max(2, int(_cfg_get(cfg, "run.segment_chunking_min_segments", 16)))
    min_video_sec_for_chunking = max(0.0, float(_cfg_get(cfg, "run.segment_chunking_min_video_sec", 60.0)))
    max_segments_per_chunk = max(2, int(_cfg_get(cfg, "run.segment_chunking_max_segments_per_request", 8)))
    max_window_sec_per_chunk = max(0.0, float(_cfg_get(cfg, "run.segment_chunking_max_window_sec", 0.0)))
    chunking_disable_operations = bool(_cfg_get(cfg, "run.segment_chunking_disable_operations", True))
    force_operations_on_overlong = bool(
        _cfg_get(cfg, "run.segment_chunking_force_operations_on_overlong_segments", True)
    )
    chunk_split_only = bool(_cfg_get(cfg, "run.segment_chunking_collect_split_operations_only", True))
    chunking_video_pad_sec = max(0.0, float(_cfg_get(cfg, "run.segment_chunking_video_pad_sec", 1.0)))
    chunking_keep_temp_files = bool(_cfg_get(cfg, "run.segment_chunking_keep_temp_files", False))
    include_previous_labels_context = bool(
        _cfg_get(cfg, "run.segment_chunking_include_previous_labels_context", True)
    )
    max_previous_labels = max(0, int(_cfg_get(cfg, "run.segment_chunking_max_previous_labels", 12)))
    consistency_memory_enabled = bool(_cfg_get(cfg, "run.segment_chunking_consistency_memory_enabled", True))
    consistency_memory_limit = max(8, int(_cfg_get(cfg, "run.segment_chunking_consistency_memory_limit", 40)))
    consistency_prompt_terms = max(0, int(_cfg_get(cfg, "run.segment_chunking_consistency_prompt_terms", 16)))
    consistency_normalize_labels = bool(
        _cfg_get(cfg, "run.segment_chunking_consistency_normalize_labels", True)
    )
    configured_episode_primary_model = str(
        resolve_stage_model(cfg, stage_name, _cfg_get(cfg, "gemini.model", "gemini-2.5-flash"))
        or "gemini-2.5-flash"
    ).strip() or "gemini-2.5-flash"
    retry_with_quota_fallback_model = bool(_cfg_get(cfg, "gemini.retry_with_quota_fallback_model", False))
    quota_fallback_model = str(_cfg_get(cfg, "gemini.quota_fallback_model", "gemini-3.1-pro-preview") or "").strip()
    quota_fallback_from_models_raw = _cfg_get(
        cfg,
        "gemini.quota_fallback_from_models",
        [configured_episode_primary_model],
    )
    quota_fallback_from_models: set[str] = set()
    if isinstance(quota_fallback_from_models_raw, list):
        for item in quota_fallback_from_models_raw:
            value = str(item or "").strip().lower()
            if value:
                quota_fallback_from_models.add(value)
    else:
        raw_text = str(quota_fallback_from_models_raw or "").strip()
        if raw_text:
            for part in re.split(r"[,\|;]+", raw_text):
                value = str(part or "").strip().lower()
                if value:
                    quota_fallback_from_models.add(value)
    if not quota_fallback_from_models and configured_episode_primary_model:
        quota_fallback_from_models.add(configured_episode_primary_model.lower())
    active_model_override = str(
        model_override or (task_state.get("episode_active_model", "") if isinstance(task_state, dict) else "")
    ).strip() or configured_episode_primary_model
    episode_model_escalated = bool(
        task_state.get("episode_model_escalated", False) if isinstance(task_state, dict) else False
    )
    episode_fallback_reason = str(
        task_state.get("episode_fallback_reason", "") if isinstance(task_state, dict) else ""
    ).strip()
    chat_only_mode = bool(_cfg_get(cfg, "run.chat_only_mode", False)) or (
        str(_cfg_get(cfg, "run.primary_solve_backend", "api") or "api").strip().lower() == "chat_web"
    )
    max_segment_duration_sec = max(0.0, float(_cfg_get(cfg, "run.max_segment_duration_sec", 10.0)))
    source_has_overlong_segments = bool(
        max_segment_duration_sec > 0
        and any(legacy._segment_duration_exceeds_limit(seg, max_segment_duration_sec) for seg in segments)
    )
    synthesize_chat_split_fallback = bool(_cfg_get(cfg, "run.chat_ops_synthesize_split_fallback", True))

    def _apply_episode_model_updates(
        model_name: str,
        *,
        reason: Optional[str] = None,
        persist: bool = False,
    ) -> Dict[str, Any]:
        nonlocal active_model_override, episode_model_escalated, episode_fallback_reason
        active_model_override = str(model_name or active_model_override or configured_episode_primary_model).strip()
        if not active_model_override:
            active_model_override = configured_episode_primary_model
        episode_model_escalated = bool(
            active_model_override
            and configured_episode_primary_model
            and active_model_override.lower() != configured_episode_primary_model.lower()
        )
        if reason is not None:
            episode_fallback_reason = str(reason or "").strip()
        updates = {
            "episode_active_model": active_model_override,
            "episode_model_escalated": episode_model_escalated,
            "episode_fallback_reason": episode_fallback_reason,
        }
        if isinstance(task_state, dict):
            task_state.update(updates)
        if persist and task_id:
            persist_updates = dict(updates)
            if not persist_updates["episode_fallback_reason"]:
                persist_updates.pop("episode_fallback_reason")
            legacy._persist_task_state_fields(
                cfg,
                task_id,
                task_state if isinstance(task_state, dict) else None,
                **persist_updates,
            )
        return updates

    def _persist_cost_updates(payload: Dict[str, Any], request_model: str) -> None:
        if not task_id:
            return
        meta = payload.get("_meta", {}) if isinstance(payload, dict) else {}
        usage_meta = meta.get("usage", {}) if isinstance(meta, dict) else {}
        estimated_cost_usd = float(meta.get("estimated_cost_usd", 0.0) or 0.0)
        if estimated_cost_usd <= 0 and isinstance(usage_meta, dict):
            estimated_cost_usd = estimate_cost_from_usage(cfg, request_model, usage_meta)
        cost_updates = build_episode_cost_updates(
            cfg,
            task_state if isinstance(task_state, dict) else None,
            stage_name=stage_name,
            model_name=request_model,
            cost_usd=estimated_cost_usd,
            key_class=str(meta.get("api_key_class", "") or ""),
        )
        if str(meta.get("episode_key_class_pin", "") or "").strip():
            cost_updates["episode_key_class_pin"] = str(meta.get("episode_key_class_pin", "")).strip()
        if isinstance(task_state, dict):
            task_state.update(cost_updates)
        legacy._persist_task_state_fields(
            cfg,
            task_id,
            task_state if isinstance(task_state, dict) else None,
            **cost_updates,
        )
        print(
            "[gemini] cost ledger: "
            f"stage={stage_name} model={request_model} "
            f"delta=${estimated_cost_usd:.4f} "
            f"total=${float(cost_updates.get('episode_estimated_cost_usd', 0.0) or 0.0):.4f} "
            f"ratio={float(cost_updates.get('episode_cost_ratio', 0.0) or 0.0) * 100.0:.1f}% "
            f"key_class={str(cost_updates.get('episode_key_class_used', '') or meta.get('api_key_class', '') or 'unknown')}"
        )

    def _call_labels(
        prompt_text: str,
        media_file: Optional[Path],
        seg_count: int,
        *,
        effective_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        nonlocal active_model_override, task_state
        request_cfg = effective_cfg or cfg
        requested_model = str(
            active_model_override or resolve_stage_model(request_cfg, stage_name, configured_episode_primary_model)
        ).strip() or configured_episode_primary_model
        model_candidates = [requested_model or configured_episode_primary_model]
        if _is_gen3_model(requested_model):
            model_candidates = _ordered_gen3_gemini_models(
                requested_model,
                _cfg_get(request_cfg, "gemini.gen3_fallback_models", ["gemini-3.1-pro-preview"]),
            )
            if requested_model and requested_model.lower() not in {item.lower() for item in model_candidates}:
                model_candidates.insert(0, requested_model)
            if not model_candidates:
                model_candidates = [requested_model or configured_episode_primary_model]

        candidate_index = 0
        while candidate_index < len(model_candidates):
            request_model = str(model_candidates[candidate_index] or "").strip() or configured_episode_primary_model
            try:
                try:
                    payload = legacy.call_gemini_labels(
                        request_cfg,
                        prompt_text,
                        video_file=media_file,
                        segment_count=seg_count,
                        model_override=request_model,
                        task_id=task_id,
                        task_state=task_state,
                        stage_name=stage_name,
                    )
                except TypeError as exc:
                    if "unexpected keyword argument" not in str(exc):
                        raise
                    payload = legacy.call_gemini_labels(
                        request_cfg,
                        prompt_text,
                        video_file=media_file,
                        segment_count=seg_count,
                        model_override=request_model,
                    )
                updates = _apply_episode_model_updates(request_model, reason=episode_fallback_reason or None)
                if isinstance(payload, dict):
                    meta = payload.setdefault("_meta", {})
                    meta["episode_active_model"] = updates["episode_active_model"]
                    meta["episode_model_escalated"] = bool(updates["episode_model_escalated"])
                    if updates["episode_fallback_reason"]:
                        meta["episode_fallback_reason"] = updates["episode_fallback_reason"]
                    _persist_cost_updates(payload, request_model)
                return payload
            except Exception as exc:
                if legacy._is_gemini_quota_error(exc):
                    fallback_model = quota_fallback_model
                    if not retry_with_quota_fallback_model or not _is_gen3_model(request_model):
                        if task_id and isinstance(task_state, dict) and _model_requires_paid_pool(request_model):
                            task_state = legacy._persist_task_state_fields(
                                cfg,
                                task_id,
                                task_state,
                                deferred_due_to_model_quota=True,
                                last_error=str(exc),
                            )
                        raise
                    if not fallback_model or fallback_model.lower() == request_model.lower():
                        if task_id and isinstance(task_state, dict) and _model_requires_paid_pool(request_model):
                            task_state = legacy._persist_task_state_fields(
                                cfg,
                                task_id,
                                task_state,
                                deferred_due_to_model_quota=True,
                                last_error=str(exc),
                            )
                        raise
                    if quota_fallback_from_models and request_model.lower() not in quota_fallback_from_models:
                        if task_id and isinstance(task_state, dict) and _model_requires_paid_pool(request_model):
                            task_state = legacy._persist_task_state_fields(
                                cfg,
                                task_id,
                                task_state,
                                deferred_due_to_model_quota=True,
                                last_error=str(exc),
                            )
                        raise
                    if fallback_model.lower() not in {item.lower() for item in model_candidates}:
                        model_candidates.append(fallback_model)
                    candidate_index = next(
                        idx for idx, item in enumerate(model_candidates) if item.lower() == fallback_model.lower()
                    )
                    print(
                        "[gemini] quota model fallback engaged: "
                        f"{request_model} -> {fallback_model}"
                    )
                    _apply_episode_model_updates(
                        fallback_model,
                        reason=(
                            f"Quota fallback promoted episode model from {request_model} "
                            f"to {fallback_model}"
                        ),
                        persist=bool(task_id),
                    )
                    continue

                if not _is_gemini_availability_error(exc):
                    raise

                next_index = candidate_index + 1
                if next_index >= len(model_candidates):
                    exhausted_reason = (
                        "Gen3 model fallback exhausted after availability errors: "
                        f"{str(exc).strip() or exc.__class__.__name__}"
                    )
                    _apply_episode_model_updates(request_model, reason=exhausted_reason, persist=bool(task_id))
                    raise

                next_model = str(model_candidates[next_index] or "").strip()
                if not next_model:
                    raise
                print(
                    "[gemini] episode model fallback engaged: "
                    f"{request_model} -> {next_model}"
                )
                _apply_episode_model_updates(
                    next_model,
                    reason=(
                        f"availability fallback after {request_model}: "
                        f"{str(exc).strip() or exc.__class__.__name__}"
                    ),
                    persist=bool(task_id),
                )
                candidate_index = next_index

        raise RuntimeError("Gemini request failed before any Gen3 model candidate could be used.")

    def _persist_chat_only_fields(**updates: Any) -> None:
        if not task_id:
            return
        merged = {
            "solve_backend": "chat_web",
            "chat_only_mode": True,
            "chat_compare_skipped": True,
        }
        merged.update(updates)
        if isinstance(task_state, dict):
            task_state.update(merged)
        legacy._persist_task_state_fields(
            cfg,
            task_id,
            task_state if isinstance(task_state, dict) else None,
            **merged,
        )

    chat_progress_state: Dict[str, Any] = {
        "detail": "requesting Gemini labels",
        "phase": "startup",
        "completed_segments": 0,
        "chunk_index": 0,
        "chunk_total": 0,
        "request_scope": str(stage_name or "").strip() or "labeling",
    }
    last_chat_progress_persist_ts = 0.0

    def _persist_chat_stage_progress(*, force: bool = False) -> None:
        nonlocal task_state, last_chat_progress_persist_ts
        _emit_solver_heartbeat()
        if not task_id:
            return

        now = time.monotonic()
        persist_interval_sec = max(
            5.0,
            float(_cfg_get(cfg, "run.chat_stage_progress_persist_sec", 10.0) or 10.0),
        )
        if not force and (now - last_chat_progress_persist_ts) < persist_interval_sec:
            return

        progress_total = len(segments)
        progress_current = min(
            progress_total,
            max(0, int(chat_progress_state.get("completed_segments", 0) or 0)),
        )
        detail = str(chat_progress_state.get("detail", "") or "").strip() or "requesting Gemini labels"
        phase = str(chat_progress_state.get("phase", "") or "").strip()
        request_scope = str(chat_progress_state.get("request_scope", "") or "").strip()
        watchdog_base_sec = max(
            60.0,
            float(_cfg_get(cfg, "run.watchdog_stale_threshold_sec", 600.0) or 600.0),
        )
        watchdog_hint_sec = max(
            legacy._stage_watchdog_timeout_hint_sec(
                cfg,
                stage="chat_labels",
                base_timeout_sec=watchdog_base_sec,
                progress_current=progress_current,
                progress_total=progress_total,
            ),
            _chat_phase_watchdog_timeout_hint_sec(
                cfg,
                phase=phase,
                request_scope=request_scope,
            ),
        )
        task_state = legacy._persist_task_stage_status(
            cfg,
            task_id,
            task_state if isinstance(task_state, dict) else None,
            stage="chat_labels",
            status="running",
            progress_current=progress_current,
            progress_total=progress_total,
            detail=detail,
            watchdog_timeout_sec=watchdog_hint_sec,
        )
        task_state = legacy._persist_task_state_fields(
            cfg,
            task_id,
            task_state if isinstance(task_state, dict) else None,
            chat_active_phase=phase,
            chat_active_chunk_index=int(chat_progress_state.get("chunk_index", 0) or 0),
            chat_active_chunk_total=int(chat_progress_state.get("chunk_total", 0) or 0),
            chat_active_scope=request_scope,
            chat_completed_segments=int(progress_current),
            chat_active_watchdog_timeout_sec=round(float(watchdog_hint_sec or 0.0), 1),
            chat_failure_phase="",
            chat_failure_reason="",
        )
        last_chat_progress_persist_ts = now

    def _chat_progress_heartbeat() -> None:
        _persist_chat_stage_progress(force=False)

    def _set_chat_progress(
        *,
        detail: str,
        phase: str,
        completed_segments: Optional[int] = None,
        chunk_index: Optional[int] = None,
        chunk_total: Optional[int] = None,
        request_scope: Optional[str] = None,
        force: bool = False,
    ) -> None:
        if completed_segments is not None:
            chat_progress_state["completed_segments"] = max(0, int(completed_segments or 0))
        if chunk_index is not None:
            chat_progress_state["chunk_index"] = max(0, int(chunk_index or 0))
        if chunk_total is not None:
            chat_progress_state["chunk_total"] = max(0, int(chunk_total or 0))
        if request_scope is not None:
            chat_progress_state["request_scope"] = str(request_scope or "").strip()
        chat_progress_state["detail"] = str(detail or "").strip() or str(chat_progress_state.get("detail", "") or "").strip()
        chat_progress_state["phase"] = str(phase or "").strip() or str(chat_progress_state.get("phase", "") or "").strip()
        _persist_chat_stage_progress(force=force)

    def _should_run_chat_structural_planner() -> bool:
        if not allow_operations:
            return False
        if not bool(_cfg_get(cfg, "run.chat_ops_enabled", True)):
            return False
        allow_split = bool(_cfg_get(cfg, "run.structural_allow_split", False))
        allow_merge = bool(_cfg_get(cfg, "run.structural_allow_merge", False))
        if not allow_split and not allow_merge:
            return False
        structural_skip_if_segments_ge = max(0, int(_cfg_get(cfg, "run.structural_skip_if_segments_ge", 0) or 0))
        if structural_skip_if_segments_ge > 0 and len(segments) >= structural_skip_if_segments_ge and not source_has_overlong_segments:
            return False
        if source_has_overlong_segments:
            return True
        if not bool(_cfg_get(cfg, "run.chat_ops_run_without_overlong", False)):
            return False
        return bool(_cfg_get(cfg, "run.auto_continuity_merge_enabled", False)) and allow_merge

    def _request_via_chat_only(
        override_max_segments: Optional[int] = None,
        *,
        target_segments: Optional[List[Dict[str, Any]]] = None,
        seed_collected_labels: Optional[Dict[int, Dict[str, Any]]] = None,
        seed_prior_labels: Optional[List[str]] = None,
        seed_consistency_terms: Optional[List[str]] = None,
        seed_consistency_alias_to_canonical: Optional[Dict[str, str]] = None,
        seed_chunk_manifest: Optional[List[Dict[str, Any]]] = None,
        seed_segments_done: int = 0,
        seed_attached_notes: Optional[List[str]] = None,
        seed_usage_rows: Optional[List[Dict[str, Any]]] = None,
        seed_planner_attempted: bool = False,
        seed_planned_operations: Optional[List[Dict[str, Any]]] = None,
        seed_planner_json_path: str = "",
        seed_planner_prompt_path: str = "",
        skip_planner: bool = False,
    ) -> Dict[str, Any]:
        nonlocal active_model_override, task_state
        out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs") or "outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        episode_token = str(task_id or "episode").strip() or "episode"
        episode_cache_dir = out_dir / "_chat_only" / episode_token
        episode_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_root = episode_cache_dir / (str(stage_name or "labeling").strip() or "labeling")
        cache_root.mkdir(parents=True, exist_ok=True)
        working_segments = list(target_segments or segments)

        def _resolve_prepared_chat_video(raw_video_file: Optional[Path]) -> Optional[Path]:
            seen: set[str] = set()
            ordered_candidates: List[Path] = []

            def _append_candidate(candidate: Optional[Path | str]) -> None:
                if not candidate:
                    return
                try:
                    path_obj = candidate if isinstance(candidate, Path) else Path(str(candidate))
                except Exception:
                    return
                variants = [path_obj]
                if not path_obj.is_absolute():
                    try:
                        variants.append((Path.cwd() / path_obj))
                    except Exception:
                        pass
                for variant in variants:
                    key = str(variant)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    ordered_candidates.append(variant)

            _append_candidate(raw_video_file)
            if isinstance(task_state, dict):
                _append_candidate(str(task_state.get("video_path", "") or "").strip())
            if task_id:
                try:
                    scoped_paths = legacy._task_scoped_artifact_paths(cfg, task_id)
                except Exception:
                    scoped_paths = {}
                _append_candidate(scoped_paths.get("video"))
                _append_candidate(episode_cache_dir / f"video_{episode_token}.mp4")

            expanded_candidates: List[Path] = []
            expanded_seen: set[str] = set()

            def _append_expanded(candidate: Path) -> None:
                key = str(candidate)
                if not key or key in expanded_seen:
                    return
                expanded_seen.add(key)
                expanded_candidates.append(candidate)

            for candidate in ordered_candidates:
                _append_expanded(candidate)
                try:
                    _append_expanded(candidate.with_name(f"{candidate.stem}_upload_opt.mp4"))
                except Exception:
                    pass
                try:
                    if candidate.parent.exists():
                        for opt_candidate in sorted(
                            candidate.parent.glob(f"{candidate.stem}_upload_opt_s*.mp4")
                        ):
                            _append_expanded(opt_candidate)
                except Exception:
                    pass

            for candidate in expanded_candidates:
                try:
                    if candidate.exists():
                        return candidate
                except Exception:
                    continue
            return None

        current_video_file = video_file
        if current_video_file is not None and not isinstance(current_video_file, Path):
            current_video_file = Path(current_video_file)
        current_video_file = _resolve_prepared_chat_video(current_video_file)
        if current_video_file is None:
            failure = f"Chat-only solve requires a prepared task video. (path={video_file})"
            print(f"[gemini] [error] {failure}")
            _persist_chat_only_fields(chat_failure_phase="video", chat_failure_reason=failure, last_error=failure)
            raise RuntimeError(failure)

        chat_only = import_module("src.solver.chat_only")
        optimized_chat_video = legacy._maybe_optimize_video_for_upload(current_video_file, cfg)
        chat_video_file = (
            optimized_chat_video
            if optimized_chat_video is not None and optimized_chat_video.exists()
            else current_video_file
        )
        if chat_video_file != current_video_file:
            try:
                original_mb = float(current_video_file.stat().st_size) / (1024 * 1024)
                optimized_mb = float(chat_video_file.stat().st_size) / (1024 * 1024)
                print(
                    "[chat] using optimized upload video: "
                    f"{current_video_file.name} ({original_mb:.1f} MB) -> {chat_video_file.name} ({optimized_mb:.1f} MB)"
                )
            except Exception:
                print(f"[chat] using optimized upload video: {chat_video_file.name}")
        prepared_video_cache = episode_cache_dir / f"video_{episode_token}{chat_video_file.suffix or '.mp4'}"
        if chat_video_file != prepared_video_cache:
            try:
                needs_refresh = True
                if prepared_video_cache.exists():
                    needs_refresh = prepared_video_cache.stat().st_size != chat_video_file.stat().st_size
                if needs_refresh:
                    shutil.copy2(chat_video_file, prepared_video_cache)
                    print(f"[video] cached prepared chat video: {prepared_video_cache}")
            except Exception as exc:
                print(f"[video] warning: could not cache prepared chat video: {exc}")
        request_model = str(
            active_model_override or resolve_stage_model(cfg, stage_name, configured_episode_primary_model)
        ).strip() or configured_episode_primary_model
        updates = _apply_episode_model_updates(request_model, reason=episode_fallback_reason or None)
        base_prompt_path = cache_root / "chat_base_prompt.txt"
        base_prompt_path.write_text(str(prompt or "").strip() + "\n", encoding="utf-8")

        def _build_payload(
            *,
            segments_payload: List[Dict[str, Any]],
            operations_payload: List[Dict[str, Any]],
            usage_rows: List[Dict[str, Any]],
            attached_notes: List[str],
            response_json_path: str,
            prompt_path: str,
            chunk_count_value: int,
            planner_attempted: bool,
            request_id: str = "",
            gemini_session_id: str = "",
            retry_stage: str = "",
            retry_reason: str = "",
            gemini_latency_ms: int = 0,
            raw_response_path: str = "",
        ) -> Dict[str, Any]:
            usage_meta = _merge_usage_metadata(usage_rows)
            estimated_cost_usd = estimate_cost_from_usage(cfg, request_model, usage_meta)
            payload: Dict[str, Any] = {
                "operations": operations_payload,
                "segments": segments_payload,
                "_meta": {
                    "solve_backend": "chat_web",
                    "chat_only_mode": True,
                    "chat_compare_skipped": True,
                    "video_attached": _chat_attach_notes_confirm_video(attached_notes),
                    "mode": "chat_web",
                    "fallback_used": False,
                    "video_transport": "chat_web",
                    "video_parts_count": 1,
                    "chunked": chunk_count_value > 1,
                    "chunk_count": chunk_count_value,
                    "api_key_source": "chat_web",
                    "api_key_class": "chat_web",
                    "rotation_policy": "chat_web",
                    "model": request_model,
                    "stage_name": str(stage_name or "").strip() or "labeling",
                    "estimated_cost_usd": round(float(estimated_cost_usd or 0.0), 8),
                    "episode_key_class_pin": "",
                    "episode_active_model": updates["episode_active_model"],
                    "episode_model_escalated": bool(updates["episode_model_escalated"]),
                    "usage": usage_meta,
                    "chat_response_json_path": str(response_json_path or ""),
                    "chat_prompt_path": str(prompt_path or ""),
                    "chat_labels_chunk_count": int(chunk_count_value),
                    "chat_ops_attempted": bool(planner_attempted),
                    "chat_ops_planned": len(operations_payload),
                    "request_id": str(request_id or "").strip(),
                    "gemini_session_id": str(gemini_session_id or "").strip(),
                    "retry_stage": str(retry_stage or "").strip(),
                    "retry_reason": str(retry_reason or "").strip(),
                    "gemini_latency_ms": int(gemini_latency_ms or 0),
                    "raw_response_path": str(raw_response_path or "").strip(),
                },
            }
            if updates["episode_fallback_reason"]:
                payload["_meta"]["episode_fallback_reason"] = updates["episode_fallback_reason"]
            return payload

        attached_notes: List[str] = [str(item) for item in (seed_attached_notes or [])]
        usage_rows: List[Dict[str, Any]] = [dict(item) for item in (seed_usage_rows or []) if isinstance(item, dict)]
        planner_attempted = bool(seed_planner_attempted)
        planned_operations: List[Dict[str, Any]] = [dict(item) for item in (seed_planned_operations or []) if isinstance(item, dict)]
        planner_json_path = str(seed_planner_json_path or "")
        planner_prompt_path = str(seed_planner_prompt_path or "")

        def _finalize_chat_labels_result(
            labels_result: Dict[str, Any],
            *,
            chunk_count_value: int,
            response_json_path: str = "",
            prompt_path: str = "",
        ) -> Dict[str, Any]:
            attached_notes.extend([str(item) for item in labels_result.get("attach_notes", []) or []])
            if isinstance(labels_result.get("usage"), dict):
                usage_rows.append(labels_result["usage"])
            result_segments = list(labels_result.get("segments", []) or [])
            if seed_collected_labels:
                merged_by_idx: Dict[int, Dict[str, Any]] = {
                    int(idx): dict(item)
                    for idx, item in (seed_collected_labels or {}).items()
                    if int(idx or 0) > 0 and isinstance(item, dict)
                }
                for item in result_segments:
                    idx = int(item.get("segment_index", 0) or 0)
                    if idx > 0:
                        merged_by_idx[idx] = dict(item)
                merged_segments: List[Dict[str, Any]] = []
                for seg in segments:
                    idx = int(seg.get("segment_index", 0) or 0)
                    item = dict(merged_by_idx.get(idx, {}))
                    merged_segments.append(
                        {
                            "segment_index": idx,
                            "start_sec": round(float(item.get("start_sec", seg.get("start_sec", 0.0)) or 0.0), 3),
                            "end_sec": round(float(item.get("end_sec", seg.get("end_sec", 0.0)) or 0.0), 3),
                            "label": str(item.get("label", seg.get("current_label", "")) or "").strip(),
                        }
                    )
                result_segments = merged_segments
            resolved_response_json = str(
                response_json_path or labels_result.get("out_json", "") or planner_json_path
            )
            resolved_prompt_path = str(
                prompt_path or labels_result.get("prompt_path", "") or planner_prompt_path or base_prompt_path
            )
            payload = _build_payload(
                segments_payload=result_segments,
                operations_payload=planned_operations,
                usage_rows=usage_rows,
                attached_notes=attached_notes,
                response_json_path=resolved_response_json,
                prompt_path=resolved_prompt_path,
                chunk_count_value=chunk_count_value,
                planner_attempted=planner_attempted,
                request_id=str(labels_result.get("request_id", "") or ""),
                gemini_session_id=str(labels_result.get("gemini_session_id", "") or ""),
                retry_stage=str(labels_result.get("retry_stage", "") or ""),
                retry_reason=str(labels_result.get("retry_reason", "") or ""),
                gemini_latency_ms=int(labels_result.get("latency_ms", 0) or 0),
                raw_response_path=str(labels_result.get("raw_response_path", "") or ""),
            )
            _persist_cost_updates(payload, request_model)
            _persist_chat_only_fields(
                chat_ops_attempted=planner_attempted,
                chat_ops_planned=len(planned_operations),
                chat_labels_chunk_count=chunk_count_value,
                chat_response_json_path=resolved_response_json,
                chat_prompt_path=resolved_prompt_path,
                gemini_last_request_id=str(labels_result.get("request_id", "") or ""),
                gemini_session_id=str(labels_result.get("gemini_session_id", "") or ""),
                gemini_last_retry_stage=str(labels_result.get("retry_stage", "") or ""),
                gemini_last_retry_reason=str(labels_result.get("retry_reason", "") or ""),
                gemini_last_latency_ms=int(labels_result.get("latency_ms", 0) or 0),
                last_error="",
            )
            return payload

        def _run_single_request_labels(cache_leaf: str) -> Dict[str, Any]:
            _set_chat_progress(
                detail="waiting for Gemini labels response",
                phase="single_request",
                completed_segments=0,
                chunk_index=1,
                chunk_total=1,
                request_scope=str(stage_name or "").strip() or "labeling",
                force=True,
            )
            labels_prompt = chat_only.build_labels_prompt(prompt)
            labels_result = chat_only.run_labels_generation(
                cfg=cfg,
                source_segments=working_segments,
                video_file=chat_video_file,
                prompt_text=labels_prompt,
                cache_dir=cache_root / cache_leaf,
                episode_id=episode_token,
                model=request_model,
                prompt_scope="chat_labels",
                heartbeat=_chat_progress_heartbeat,
            )
            _set_chat_progress(
                detail="Gemini labels response received",
                phase="single_request_complete",
                completed_segments=min(len(segments), seed_segments_done + len(working_segments)),
                chunk_index=1,
                chunk_total=1,
                request_scope=str(stage_name or "").strip() or "labeling",
                force=True,
            )
            return _finalize_chat_labels_result(labels_result, chunk_count_value=1)

        def _synthesize_split_operations_from_source() -> List[Dict[str, Any]]:
            if not synthesize_chat_split_fallback or max_segment_duration_sec <= 0:
                return []
            out: List[Dict[str, Any]] = []
            seen: set[int] = set()
            for seg in segments:
                idx = int(seg.get("segment_index", 0) or 0)
                if idx <= 0 or idx in seen:
                    continue
                if not legacy._segment_duration_exceeds_limit(seg, max_segment_duration_sec):
                    continue
                seen.add(idx)
                out.append({"action": "split", "segment_index": idx})
            out.sort(key=lambda item: -int(item.get("segment_index", 0) or 0))
            return out

        if (not skip_planner) and _should_run_chat_structural_planner():
            planner_attempted = True
            _set_chat_progress(
                detail="planning structural operations in Gemini",
                phase="planner",
                completed_segments=0,
                chunk_index=0,
                chunk_total=0,
                request_scope="planner",
                force=True,
            )
            planner_prompt = chat_only.build_structural_planner_prompt(
                segments,
                allow_merge=bool(_cfg_get(cfg, "run.structural_allow_merge", False)),
                max_segment_duration_sec=max_segment_duration_sec,
                extra_instructions=str(_cfg_get(cfg, "gemini.extra_instructions", "") or "").strip(),
            )
            try:
                    planner_result = chat_only.run_structural_planner(
                        cfg=cfg,
                        source_segments=segments,
                        video_file=chat_video_file,
                        prompt_text=planner_prompt,
                        cache_dir=cache_root / "planner",
                        episode_id=episode_token,
                    model=request_model,
                    allow_merge=bool(_cfg_get(cfg, "run.structural_allow_merge", False)),
                    prompt_scope="chat_ops",
                    heartbeat=_chat_progress_heartbeat,
                )
            except Exception as exc:
                failure = str(exc).strip() or exc.__class__.__name__
                planner_prompt_path = str((cache_root / "planner" / "ops_prompt.txt").resolve())
                fail_open = bool(_cfg_get(cfg, "run.chat_ops_fail_open", True))
                if not fail_open:
                    _persist_chat_only_fields(
                        chat_failure_phase="chat_ops",
                        chat_failure_reason=failure,
                        chat_ops_attempted=True,
                        last_error=failure,
                    )
                    raise RuntimeError(f"Chat structural planner failed: {failure}") from exc
                attached_notes.append(f"chat_ops_soft_fail:{failure}")
                planned_operations = _synthesize_split_operations_from_source()
                if planned_operations:
                    attached_notes.append(f"chat_ops_split_fallback:{len(planned_operations)}")
                    print(
                        "[chat] structural planner failed; synthesized split fallback from DOM/source: "
                        f"count={len(planned_operations)} reason={failure}"
                    )
                else:
                    print(f"[chat] structural planner failed; continuing without structural ops: {failure}")
                _set_chat_progress(
                    detail=(
                        "structural planner failed; synthesized split fallback from DOM/source"
                        if planned_operations
                        else "structural planner failed; continuing with labels only"
                    ),
                    phase="planner_soft_fail",
                    completed_segments=0,
                    chunk_index=0,
                    chunk_total=0,
                    request_scope="planner",
                    force=True,
                )
                _persist_chat_only_fields(
                    chat_ops_attempted=True,
                    chat_ops_failure_reason=failure,
                    chat_response_json_path=planner_json_path,
                    chat_prompt_path=planner_prompt_path,
                    last_error="",
                )
            else:
                planned_operations = list(planner_result.get("operations", []) or [])
                attached_notes.extend([str(item) for item in planner_result.get("attach_notes", []) or []])
                if isinstance(planner_result.get("usage"), dict):
                    usage_rows.append(planner_result["usage"])
                planner_json_path = str(planner_result.get("out_json", "") or "")
                planner_prompt_path = str(planner_result.get("prompt_path", "") or "")
                _persist_chat_only_fields(
                    chat_ops_attempted=True,
                    chat_ops_failure_reason="",
                    chat_response_json_path=planner_json_path,
                    chat_prompt_path=planner_prompt_path,
                    last_error="",
                )
                _set_chat_progress(
                    detail="structural planner completed; requesting labels",
                    phase="planner_complete",
                    completed_segments=0,
                    chunk_index=0,
                    chunk_total=0,
                    request_scope="labeling",
                    force=True,
                )

        can_chunk_by_shape = bool(
            chunking_enabled
            and len(working_segments) >= min_segments_for_chunking
            and (override_max_segments if override_max_segments is not None else max_segments_per_chunk) < len(working_segments)
        )
        short_video_for_chunking = False
        video_duration_sec_for_chunking = 0.0
        if can_chunk_by_shape and min_video_sec_for_chunking > 0:
            video_duration_sec_for_chunking = legacy._probe_video_duration_seconds(chat_video_file)
            short_video_for_chunking = (
                video_duration_sec_for_chunking > 0.2 and video_duration_sec_for_chunking < min_video_sec_for_chunking
            )
            if short_video_for_chunking:
                print(
                    "[chat] segment chunking skipped: short video "
                    f"({video_duration_sec_for_chunking:.1f}s < {min_video_sec_for_chunking:.1f}s); "
                    "using single-request flow."
                )
        should_chunk = bool(can_chunk_by_shape and not short_video_for_chunking)

        if not should_chunk:
            try:
                return _run_single_request_labels("labels")
            except Exception as exc:
                failure = str(exc).strip() or exc.__class__.__name__
                _persist_chat_only_fields(
                    chat_failure_phase="chat_labels",
                    chat_failure_reason=failure,
                    chat_ops_attempted=planner_attempted,
                    last_error=failure,
                )
                raise RuntimeError(f"Chat labels generation failed: {failure}") from exc

        ffmpeg_bin = legacy._resolve_ffmpeg_binary()
        if not ffmpeg_bin:
            print("[chat] segment chunking skipped: ffmpeg not found; using single-request flow.")
            return _run_single_request_labels("labels_no_ffmpeg")

        chunks = legacy._segment_chunks(
            working_segments,
            override_max_segments if override_max_segments is not None else max_segments_per_chunk,
            max_window_sec=max_window_sec_per_chunk,
        )
        if len(chunks) <= 1:
            return _run_single_request_labels("labels_single_chunk")

        temp_chunk_files: List[Path] = []
        collected_labels: Dict[int, Dict[str, Any]] = {
            int(idx): dict(item)
            for idx, item in (seed_collected_labels or {}).items()
            if int(idx or 0) > 0 and isinstance(item, dict)
        }
        prior_labels: List[str] = [str(item) for item in (seed_prior_labels or []) if str(item or "").strip()]
        consistency_terms: List[str] = [str(item) for item in (seed_consistency_terms or []) if str(item or "").strip()]
        consistency_alias_to_canonical: Dict[str, str] = {
            str(key): str(value)
            for key, value in (seed_consistency_alias_to_canonical or {}).items()
            if str(key or "").strip()
        }
        chunk_manifest: List[Dict[str, Any]] = [dict(item) for item in (seed_chunk_manifest or []) if isinstance(item, dict)]
        extra_base = str(_cfg_get(cfg, "gemini.extra_instructions", "") or "").strip()
        segments_done = max(0, int(seed_segments_done or 0))
        print(
            f"[chat] segment chunking enabled: total_segments={len(working_segments)} "
            f"chunks={len(chunks)} max_per_chunk={override_max_segments if override_max_segments is not None else max_segments_per_chunk} "
            f"seeded_completed={segments_done}"
        )
        try:
            for chunk_idx, chunk_segments in enumerate(chunks, start=1):
                _emit_solver_heartbeat()
                window_start = max(
                    0.0,
                    min(legacy._safe_float(seg.get("start_sec"), 0.0) for seg in chunk_segments) - chunking_video_pad_sec,
                )
                window_end = max(legacy._safe_float(seg.get("end_sec"), 0.0) for seg in chunk_segments) + chunking_video_pad_sec
                if window_end <= window_start:
                    window_end = window_start + 1.0

                chunk_video_path = out_dir / f"video_{episode_token}_chatchunk_{chunk_idx:02d}.mp4"
                clipped = legacy._extract_video_window(
                    src_video=chat_video_file,
                    out_video=chunk_video_path,
                    start_sec=window_start,
                    end_sec=window_end,
                    ffmpeg_bin=ffmpeg_bin,
                )
                effective_video = chunk_video_path if clipped else chat_video_file
                if clipped:
                    temp_chunk_files.append(chunk_video_path)
                elif chunk_video_path.exists():
                    try:
                        chunk_video_path.unlink(missing_ok=True)
                    except Exception:
                        pass

                chunk_extra_parts: List[str] = []
                if extra_base:
                    chunk_extra_parts.append(extra_base)
                chunk_extra_parts.append(
                    f"This clip covers approximately {window_start:.1f}s to {window_end:.1f}s of the full episode timeline."
                )
                chunk_extra_parts.append("Label only the listed segment_index rows in this chunk; do not invent extra rows.")
                if include_previous_labels_context and max_previous_labels > 0 and prior_labels:
                    context_labels = prior_labels[-max_previous_labels:]
                    chunk_extra_parts.append(
                        "Consistency context from previous chunks (keep object naming stable): "
                        + " | ".join(context_labels)
                    )
                if consistency_memory_enabled and consistency_prompt_terms > 0 and consistency_terms:
                    chunk_hint = legacy._build_chunk_consistency_prompt_hint(
                        consistency_terms,
                        max_terms=consistency_prompt_terms,
                    )
                    if chunk_hint:
                        chunk_extra_parts.append(chunk_hint)

                chunk_prompt = legacy.build_prompt(
                    chunk_segments,
                    "\n".join(chunk_extra_parts),
                    allow_operations=False,
                    policy_trigger="policy_conflict" if stage_name != "labeling" else "base",
                )
                print(
                    f"[chat] chunk request {chunk_idx}/{len(chunks)}: "
                    f"segments={len(chunk_segments)} window={window_start:.1f}-{window_end:.1f}s "
                    f"video={effective_video.name}"
                )
                _set_chat_progress(
                    detail=(
                        f"waiting for Gemini chunk {chunk_idx}/{len(chunks)} "
                        f"({segments_done}/{len(segments)} segments completed)"
                    ),
                    phase="chunk_request",
                    completed_segments=segments_done,
                    chunk_index=chunk_idx,
                    chunk_total=len(chunks),
                    request_scope=str(stage_name or "").strip() or "labeling",
                    force=True,
                )
                labels_result = chat_only.run_labels_generation(
                    cfg=cfg,
                    source_segments=chunk_segments,
                    video_file=effective_video,
                    prompt_text=chat_only.build_labels_prompt(chunk_prompt),
                    cache_dir=cache_root / f"chunk_{chunk_idx:02d}",
                    episode_id=f"{episode_token}_chunk_{chunk_idx:02d}",
                    model=request_model,
                    prompt_scope="chat_labels",
                    heartbeat=_chat_progress_heartbeat,
                )
                segments_done += len(chunk_segments)
                _set_chat_progress(
                    detail=(
                        f"Gemini chunk {chunk_idx}/{len(chunks)} completed "
                        f"({segments_done}/{len(segments)} segments completed)"
                    ),
                    phase="chunk_complete",
                    completed_segments=segments_done,
                    chunk_index=chunk_idx,
                    chunk_total=len(chunks),
                    request_scope=str(stage_name or "").strip() or "labeling",
                    force=True,
                )
                attached_notes.extend([str(item) for item in labels_result.get("attach_notes", []) or []])
                if isinstance(labels_result.get("usage"), dict):
                    usage_rows.append(labels_result["usage"])
                chunk_manifest.append(
                    {
                        "chunk_index": chunk_idx,
                        "segment_count": len(chunk_segments),
                        "out_json": str(labels_result.get("out_json", "") or ""),
                        "prompt_path": str(labels_result.get("prompt_path", "") or ""),
                    }
                )
                reset_after_n_chunks = max(
                    0,
                    int(_cfg_get(cfg, "gemini.chat_web_chunk_thread_reset_after_n_chunks", 0) or 0),
                )
                if (
                    chat_only_mode
                    and reset_after_n_chunks > 0
                    and chunk_idx < len(chunks)
                    and chunk_idx % reset_after_n_chunks == 0
                ):
                    chat_only.restart_episode_gemini_session(
                        cfg=cfg,
                        episode_id=episode_token,
                        source_segments=working_segments,
                        heartbeat=_chat_progress_heartbeat,
                    )
                    print(
                        "[chat] reset Gemini thread inside authenticated custom app after "
                        f"chunk {chunk_idx}/{len(chunks)}."
                    )
                for item in list(labels_result.get("segments", []) or []):
                    idx = int(item.get("segment_index", 0) or 0)
                    label = str(item.get("label", "") or "").strip()
                    if not label:
                        continue
                    if consistency_memory_enabled:
                        label = legacy._update_chunk_consistency_memory(
                            label,
                            canonical_terms=consistency_terms,
                            alias_to_canonical=consistency_alias_to_canonical,
                            memory_limit=consistency_memory_limit,
                        )
                    entry = dict(item)
                    entry["label"] = label
                    collected_labels[idx] = entry
                    prior_labels.append(label)
                    if len(prior_labels) > 128:
                        prior_labels = prior_labels[-128:]

            combined_segments: List[Dict[str, Any]] = []
            for seg in segments:
                idx = int(seg.get("segment_index", 0) or 0)
                item = dict(collected_labels.get(idx, {}))
                label = str(item.get("label", seg.get("current_label", "")) or "").strip()
                if consistency_memory_enabled and consistency_normalize_labels:
                    label = legacy._update_chunk_consistency_memory(
                        label,
                        canonical_terms=consistency_terms,
                        alias_to_canonical=consistency_alias_to_canonical,
                        memory_limit=consistency_memory_limit,
                    )
                if bool(_cfg_get(cfg, "run.tier3_label_rewrite", True)):
                    label = legacy._rewrite_label_tier3(label)
                label = legacy._normalize_label_min_safety(label)
                combined_segments.append(
                    {
                        "segment_index": idx,
                        "start_sec": round(float(item.get("start_sec", seg.get("start_sec", 0.0)) or 0.0), 3),
                        "end_sec": round(float(item.get("end_sec", seg.get("end_sec", 0.0)) or 0.0), 3),
                        "label": label,
                    }
                )

            manifest_path = cache_root / "chat_chunk_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "episode_id": episode_token,
                        "model": request_model,
                        "planner_json_path": planner_json_path,
                        "planner_prompt_path": planner_prompt_path,
                        "chunks": chunk_manifest,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            payload = _build_payload(
                segments_payload=combined_segments,
                operations_payload=planned_operations,
                usage_rows=usage_rows,
                attached_notes=attached_notes,
                response_json_path=str(manifest_path),
                prompt_path=str(base_prompt_path),
                chunk_count_value=len(chunks),
                planner_attempted=planner_attempted,
            )
            _persist_cost_updates(payload, request_model)
            _persist_chat_only_fields(
                chat_ops_attempted=planner_attempted,
                chat_ops_planned=len(planned_operations),
                chat_labels_chunk_count=len(chunks),
                chat_response_json_path=str(manifest_path),
                chat_prompt_path=str(base_prompt_path),
                last_error="",
            )
            _set_chat_progress(
                detail=f"merged Gemini chunk results ({len(chunks)} chunks)",
                phase="chunk_merge_complete",
                completed_segments=len(segments),
                chunk_index=len(chunks),
                chunk_total=len(chunks),
                request_scope=str(stage_name or "").strip() or "labeling",
                force=True,
            )
            print(
                f"[chat] chunked labels merged: {len(combined_segments)} segments from {len(chunks)} chunks; "
                f"consistency_terms={len(consistency_terms)}"
            )
            return payload
        except Exception as exc:
            failure = str(exc).strip() or exc.__class__.__name__
            fallback_enabled = bool(_cfg_get(cfg, "run.chat_chunk_fallback_to_single_request", True))
            is_large = len(segments) > 20 or (video_file and video_file.stat().st_size > 50 * 1024 * 1024)
            remaining_segments = [
                dict(seg)
                for seg in segments
                if int(seg.get("segment_index", 0) or 0) not in collected_labels
            ]
            
            if fallback_enabled:
                restart_failed = False
                if chat_only_mode:
                    try:
                        chat_only.restart_episode_gemini_session(
                            cfg=cfg,
                            episode_id=episode_token,
                            source_segments=remaining_segments or working_segments or segments,
                            heartbeat=_chat_progress_heartbeat,
                        )
                        print("[chat] restarted Gemini session before chunk fallback retry.")
                    except Exception as restart_exc:
                        restart_failed = True
                        failure = (
                            f"{failure} | session restart failed before chunk fallback retry: "
                            f"{str(restart_exc).strip() or restart_exc.__class__.__name__}"
                        )
                        print(f"[chat] aborting chunk fallback because Gemini session restart failed: {restart_exc}")
                if not restart_failed and is_large and (override_max_segments is None or override_max_segments > 4):
                    retry_scope = "failed chunk onward" if collected_labels and remaining_segments else "full sequence"
                    print(
                        "[chat] chunk labeling failed for large episode; retrying with STUBBIER chunks "
                        f"(max=4) on {retry_scope}: {failure}"
                    )
                    try:
                        return _request_via_chat_only(
                            override_max_segments=4,
                            target_segments=remaining_segments or working_segments,
                            seed_collected_labels=collected_labels,
                            seed_prior_labels=prior_labels,
                            seed_consistency_terms=consistency_terms,
                            seed_consistency_alias_to_canonical=consistency_alias_to_canonical,
                            seed_chunk_manifest=chunk_manifest,
                            seed_segments_done=segments_done,
                            seed_attached_notes=attached_notes,
                            seed_usage_rows=usage_rows,
                            seed_planner_attempted=planner_attempted,
                            seed_planned_operations=planned_operations,
                            seed_planner_json_path=planner_json_path,
                            seed_planner_prompt_path=planner_prompt_path,
                            skip_planner=True,
                        )
                    except Exception as retry_exc:
                        failure = f"{failure} | smaller-chunk retry failed: {str(retry_exc)}"

                if not restart_failed:
                    print(f"[chat] chunk labeling failed; retrying single-request flow: {failure}")
                    try:
                        return _run_single_request_labels("labels_chunk_fallback")
                    except Exception as fallback_exc:
                        fallback_failure = str(fallback_exc).strip() or fallback_exc.__class__.__name__
                        failure = f"{failure} | single-request fallback failed: {fallback_failure}"
            _persist_chat_only_fields(
                chat_failure_phase="chat_chunking",
                chat_failure_reason=failure,
                chat_ops_attempted=planner_attempted,
                last_error=failure,
            )
            raise RuntimeError(f"Chat chunk labeling failed: {failure}") from exc
        finally:
            if not chunking_keep_temp_files:
                for item in temp_chunk_files:
                    try:
                        item.unlink(missing_ok=True)
                    except Exception:
                        continue

    if chat_only_mode:
        return _request_via_chat_only()

    can_chunk_by_shape = bool(
        chunking_enabled
        and video_file is not None
        and video_file.exists()
        and len(segments) >= min_segments_for_chunking
        and max_segments_per_chunk < len(segments)
    )
    short_video_for_chunking = False
    video_duration_sec_for_chunking = 0.0
    if can_chunk_by_shape and min_video_sec_for_chunking > 0 and video_file is not None and video_file.exists():
        video_duration_sec_for_chunking = legacy._probe_video_duration_seconds(video_file)
        short_video_for_chunking = (
            video_duration_sec_for_chunking > 0.2 and video_duration_sec_for_chunking < min_video_sec_for_chunking
        )
        if short_video_for_chunking:
            print(
                "[gemini] segment chunking skipped: short video "
                f"({video_duration_sec_for_chunking:.1f}s < {min_video_sec_for_chunking:.1f}s); "
                "using single-request flow."
            )

    should_chunk = bool(can_chunk_by_shape and not short_video_for_chunking)
    if not should_chunk:
        return _call_labels(prompt, video_file, len(segments))

    ffmpeg_bin = legacy._resolve_ffmpeg_binary()
    if not ffmpeg_bin:
        print("[gemini] segment chunking skipped: ffmpeg not found; using single-request flow.")
        return _call_labels(prompt, video_file, len(segments))

    chunks = legacy._segment_chunks(
        segments,
        max_segments_per_chunk,
        max_window_sec=max_window_sec_per_chunk,
    )
    if len(chunks) <= 1:
        return _call_labels(prompt, video_file, len(segments))

    extra_base = str(_cfg_get(cfg, "gemini.extra_instructions", "") or "").strip()
    out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_chunk_files: List[Path] = []
    collected_labels: Dict[int, str] = {}
    collected_operations: List[Dict[str, Any]] = []
    prior_labels: List[str] = []
    consistency_terms: List[str] = []
    consistency_alias_to_canonical: Dict[str, str] = {}
    meta_key_sources: List[str] = []
    meta_models: List[str] = []
    meta_key_classes: List[str] = []
    meta_costs: List[float] = []
    chunk_count = len(chunks)
    print(
        f"[gemini] segment chunking enabled: total_segments={len(segments)} "
        f"chunks={chunk_count} max_per_chunk={max_segments_per_chunk}"
    )
    try:
        for chunk_idx, chunk_segments in enumerate(chunks, start=1):
            _emit_solver_heartbeat()
            window_start = max(
                0.0,
                min(legacy._safe_float(seg.get("start_sec"), 0.0) for seg in chunk_segments) - chunking_video_pad_sec,
            )
            window_end = max(legacy._safe_float(seg.get("end_sec"), 0.0) for seg in chunk_segments) + chunking_video_pad_sec
            if window_end <= window_start:
                window_end = window_start + 1.0

            chunk_video_path = out_dir / f"video_{task_id or 'chunked'}_segchunk_{chunk_idx:02d}.mp4"
            clipped = legacy._extract_video_window(
                src_video=video_file,
                out_video=chunk_video_path,
                start_sec=window_start,
                end_sec=window_end,
                ffmpeg_bin=ffmpeg_bin,
            )
            effective_video = chunk_video_path if clipped else video_file
            if clipped:
                temp_chunk_files.append(chunk_video_path)
            elif chunk_video_path.exists():
                try:
                    chunk_video_path.unlink(missing_ok=True)
                except Exception:
                    pass

            chunk_extra_parts: List[str] = []
            if extra_base:
                chunk_extra_parts.append(extra_base)
            chunk_extra_parts.append(
                f"This clip covers approximately {window_start:.1f}s to {window_end:.1f}s of the full episode timeline."
            )
            chunk_extra_parts.append(
                "Label only the listed segment_index rows in this chunk; do not invent extra rows."
            )
            if include_previous_labels_context and max_previous_labels > 0 and prior_labels:
                context_labels = prior_labels[-max_previous_labels:]
                chunk_extra_parts.append(
                    "Consistency context from previous chunks (keep object naming stable): "
                    + " | ".join(context_labels)
                )
            if consistency_memory_enabled and consistency_prompt_terms > 0 and consistency_terms:
                chunk_hint = legacy._build_chunk_consistency_prompt_hint(
                    consistency_terms,
                    max_terms=consistency_prompt_terms,
                )
                if chunk_hint:
                    chunk_extra_parts.append(chunk_hint)

            chunk_allow_operations = bool(
                allow_operations
                and (
                    not chunking_disable_operations
                    or (force_operations_on_overlong and source_has_overlong_segments)
                )
            )
            chunk_prompt = legacy.build_prompt(
                chunk_segments,
                "\n".join(chunk_extra_parts),
                allow_operations=chunk_allow_operations,
            )
            print(
                f"[gemini] chunk request {chunk_idx}/{chunk_count}: "
                f"segments={len(chunk_segments)} window={window_start:.1f}-{window_end:.1f}s "
                f"video={effective_video.name if effective_video is not None else 'none'}"
            )
            chunk_cfg = cfg
            if int(_cfg_get(cfg, "gemini.skip_video_when_segments_le", 0) or 0) > 0:
                chunk_cfg = legacy._deep_merge(cfg, {"gemini": {"skip_video_when_segments_le": 0}})
            chunk_payload = _call_labels(
                chunk_prompt,
                effective_video,
                len(chunk_segments),
                effective_cfg=chunk_cfg,
            )
            _emit_solver_heartbeat()
            if chunk_allow_operations:
                collected_operations.extend(
                    legacy._collect_chunk_structural_operations(
                        cfg=cfg,
                        chunk_payload=chunk_payload,
                        chunk_segments=chunk_segments,
                        max_segment_duration_sec=max_segment_duration_sec,
                        split_only=chunk_split_only,
                    )
                )
            chunk_plan = legacy._normalize_segment_plan(chunk_payload, chunk_segments, cfg=cfg)
            for seg in chunk_segments:
                idx = int(seg.get("segment_index", 0))
                item = chunk_plan.get(idx, {})
                label = str(item.get("label", "")).strip()
                if not label:
                    label = str(seg.get("current_label", "")).strip()
                if label:
                    if consistency_memory_enabled:
                        label = legacy._update_chunk_consistency_memory(
                            label,
                            canonical_terms=consistency_terms,
                            alias_to_canonical=consistency_alias_to_canonical,
                            memory_limit=consistency_memory_limit,
                        )
                    collected_labels[idx] = label
                    prior_labels.append(label)
                    if len(prior_labels) > 128:
                        prior_labels = prior_labels[-128:]

            meta = chunk_payload.get("_meta", {}) if isinstance(chunk_payload, dict) else {}
            key_source = str(meta.get("api_key_source", "")).strip()
            if key_source:
                meta_key_sources.append(key_source)
            model_name = str(meta.get("model", "")).strip()
            if model_name:
                meta_models.append(model_name)
            key_class_name = str(meta.get("api_key_class", "")).strip().lower()
            if key_class_name:
                meta_key_classes.append(key_class_name)
            try:
                meta_costs.append(float(meta.get("estimated_cost_usd", 0.0) or 0.0))
            except Exception:
                meta_costs.append(0.0)

        combined_segments: List[Dict[str, Any]] = []
        for seg in segments:
            idx = int(seg.get("segment_index", 0))
            label = collected_labels.get(idx, str(seg.get("current_label", "")).strip())
            if consistency_memory_enabled and consistency_normalize_labels:
                label = legacy._update_chunk_consistency_memory(
                    label,
                    canonical_terms=consistency_terms,
                    alias_to_canonical=consistency_alias_to_canonical,
                    memory_limit=consistency_memory_limit,
                )
            if bool(_cfg_get(cfg, "run.tier3_label_rewrite", True)):
                label = legacy._rewrite_label_tier3(label)
            label = legacy._normalize_label_min_safety(label)
            combined_segments.append(
                {
                    "segment_index": idx,
                    "start_sec": round(legacy._safe_float(seg.get("start_sec"), 0.0), 3),
                    "end_sec": round(legacy._safe_float(seg.get("end_sec"), 0.0), 3),
                    "label": label,
                }
            )

        key_source_meta = meta_key_sources[-1] if meta_key_sources else "key_1"
        key_class_meta = "paid" if "paid" in meta_key_classes else ("free" if "free" in meta_key_classes else "")
        model_meta = meta_models[-1] if meta_models else str(
            active_model_override or resolve_stage_model(cfg, stage_name, _cfg_get(cfg, "gemini.model", "gemini-2.5-flash"))
        )
        merged_operations: List[Dict[str, Any]] = []
        seen_ops: set[Tuple[str, int]] = set()
        for op in collected_operations:
            key = (str(op.get("action", "") or "").strip().lower(), int(op.get("segment_index", 0) or 0))
            if key[1] <= 0 or key in seen_ops:
                continue
            seen_ops.add(key)
            merged_operations.append({"action": key[0], "segment_index": key[1]})
        merged_operations.sort(
            key=lambda item: (str(item.get("action", "")) != "split", -int(item.get("segment_index", 0) or 0))
        )
        result: Dict[str, Any] = {
            "operations": merged_operations,
            "segments": combined_segments,
            "_meta": {
                "video_attached": True,
                "mode": "with-video",
                "fallback_used": False,
                "video_transport": "chunked-window",
                "video_parts_count": 1,
                "chunked": True,
                "chunk_count": chunk_count,
                "consistency_memory_terms": len(consistency_terms),
                "structural_operations_count": len(merged_operations),
                "api_key_source": key_source_meta,
                "api_key_class": key_class_meta,
                "model": model_meta,
                "stage_name": str(stage_name or "").strip() or "labeling",
                "estimated_cost_usd": round(sum(meta_costs), 8),
                "episode_key_class_pin": "paid" if key_class_meta == "paid" else "",
                "episode_active_model": active_model_override or model_meta,
                "episode_model_escalated": bool(
                    episode_model_escalated
                    or (
                        active_model_override
                        and configured_episode_primary_model
                        and active_model_override.lower() != configured_episode_primary_model.lower()
                    )
                ),
            },
        }
        if episode_fallback_reason:
            result["_meta"]["episode_fallback_reason"] = episode_fallback_reason
        print(
            f"[gemini] chunked labels merged: {len(combined_segments)} segments from {chunk_count} chunks; "
            f"consistency_terms={len(consistency_terms)}"
        )
        return result
    finally:
        if not chunking_keep_temp_files:
            for p in temp_chunk_files:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    continue


_LEGACY_EXPORTS = {
    "_safe_float",
    "_short_error_text",
}


def __getattr__(name: str) -> Any:
    if name in _LEGACY_EXPORTS:
        legacy = import_module("src.solver.legacy_impl")
        return getattr(legacy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "_extract_retry_seconds_from_text",
    "_extract_retry_seconds_from_response",
    "_set_gemini_quota_cooldown",
    "_respect_gemini_quota_cooldown",
    "_respect_gemini_rate_limit",
    "_extract_zero_quota_model_name",
    "_prune_zero_quota_model_cache",
    "_mark_gemini_model_zero_quota",
    "_is_gemini_model_zero_quota_known",
    "_direct_read_bytes_limit_bytes",
    "_is_non_retriable_gemini_error",
    "_clean_json_text",
    "_repair_gemini_json_text",
    "_enforce_gemini_output_contract",
    "_parse_json_text",
    "_parse_gemini_response",
    "_gemini_file_state",
    "_wait_for_gemini_file_ready",
    "_normalize_upload_chunk_size",
    "_upload_video_via_gemini_files_api",
    "_cleanup_gemini_uploaded_file",
    "_sweep_stale_gemini_files",
    "_is_gemini_quota_error_text",
    "_is_gemini_quota_exceeded_429",
    "_is_gemini_quota_error",
    "_is_gemini_availability_error_text",
    "_is_gemini_availability_error",
    "_build_gemini_generation_config",
    "_safe_float",
    "_short_error_text",
    "_log_gemini_usage",
    "call_gemini_labels",
    "_request_labels_with_optional_segment_chunking",
]
