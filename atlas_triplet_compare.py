"""
Compare 4 candidate solutions (Tier2 / Gemini API / Gemini Chat / Vertex Chat) against up to 2 videos.

Supports:
- Local file paths
- Google Drive folder-link + filename references, e.g.
  https://drive.google.com/drive/folders/<FOLDER_ID>?usp=sharing\\video_x.mp4

Auth modes:
- gemini.auth_mode: api_key   (GEMINI_API_KEY / GOOGLE_API_KEY)
- gemini.auth_mode: vertex_ai (Service Account + Vertex endpoint)
- gemini.auth_mode: chat_web  (Playwright on gemini.google.com, no API billing)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.infra.gemini_economics import estimate_cost_usd

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("triplet_compare")


def _clean_json_text(text: str) -> str:
    """Strip markdown fences and extract the outermost JSON structure."""
    clean = re.sub(r"```json|```", "", text or "", flags=re.IGNORECASE).strip()
    obj_start = clean.find("{")
    obj_end = clean.rfind("}")
    arr_start = clean.find("[")
    arr_end = clean.rfind("]")
    has_obj = obj_start >= 0 and obj_end > obj_start
    has_arr = arr_start >= 0 and arr_end > arr_start
    # Prefer array if it starts before any object (Gemini Chat often returns raw [])
    if has_arr and (not has_obj or arr_start < obj_start):
        return clean[arr_start : arr_end + 1]
    if has_obj:
        return clean[obj_start : obj_end + 1]
    return clean

import requests
import yaml
from src.infra.solver_config import _ordered_gen3_gemini_models


def _load_dotenv(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = str(raw or "").strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = str(k or "").strip()
            if not key:
                continue
            out[key] = str(v or "").strip().strip('"').strip("'")
    except Exception as e:
        logger.error(f"Failed to load .env from {path}: {e}")
        return {}
    return out


def _read_secret(name: str, dotenv: Dict[str, str]) -> str:
    env_v = str(os.environ.get(name, "") or "").strip()
    if env_v:
        return env_v
    return str(dotenv.get(name, "") or "").strip()


def _load_chat_memory_primer(gem_cfg: Dict[str, Any]) -> str:
    text = str(gem_cfg.get("chat_web_memory_primer_text", "") or "").strip()
    if text:
        return text
    raw_path = str(gem_cfg.get("chat_web_memory_primer_file", "") or "").strip()
    if not raw_path:
        return ""
    path = Path(raw_path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _load_chat_seed_context(gem_cfg: Dict[str, Any]) -> str:
    text = str(gem_cfg.get("chat_web_seed_context_text", "") or "").strip()
    if text:
        return text
    raw_path = str(gem_cfg.get("chat_web_seed_context_file", "") or "").strip()
    if not raw_path:
        return ""
    path = Path(raw_path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _ordered_gen3_model_candidates(gem_cfg: Dict[str, Any], selected_model: str, *fallback_keys: str) -> List[str]:
    fallback_values: List[str] = []
    for key in fallback_keys:
        value = gem_cfg.get(key, "") if isinstance(gem_cfg, dict) else ""
        text = str(value or "").strip()
        if text:
            fallback_values.append(text)
    raw_fallback_models: Any = gem_cfg.get("gen3_fallback_models", fallback_values) if isinstance(gem_cfg, dict) else fallback_values
    candidates = _ordered_gen3_gemini_models(selected_model, raw_fallback_models)
    if not candidates:
        candidates = [selected_model] if str(selected_model or "").strip() else []
        for item in fallback_values:
            if item and item not in candidates:
                candidates.append(item)
    return candidates or ["gemini-3.1-pro-preview"]


def _infer_chat_web_model_mode_from_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if "show thinking" in text or "hide thinking" in text:
        return ""
    if "thinking" in text or "reasoning" in text:
        return "thinking"
    if "fast" in text or "flash" in text:
        return "fast"
    if "advanced" in text or re.search(r"\bpro\b", text):
        return "pro"
    return ""


def _normalize_chat_web_model_mode(value: Any) -> str:
    return _infer_chat_web_model_mode_from_text(value)


def _resolve_chat_web_ui_model_mode(
    gem_cfg: Dict[str, Any],
    *,
    requested_model: str = "",
) -> str:
    configured = _normalize_chat_web_model_mode(gem_cfg.get("chat_web_ui_model_mode", ""))
    if configured:
        return configured
    inferred = _normalize_chat_web_model_mode(requested_model or gem_cfg.get("model", ""))
    return inferred or "pro"


def _resolve_chat_web_allowed_model_modes(
    gem_cfg: Dict[str, Any],
    *,
    requested_model: str = "",
) -> List[str]:
    raw_value = gem_cfg.get("chat_web_allowed_model_modes", None)
    resolved: List[str] = []
    if isinstance(raw_value, (list, tuple, set)):
        items = list(raw_value)
    elif raw_value in (None, ""):
        items = []
    else:
        items = re.split(r"[\s,|]+", str(raw_value or "").strip())
    for item in items:
        normalized = _normalize_chat_web_model_mode(item)
        if normalized and normalized not in resolved:
            resolved.append(normalized)
    if resolved:
        return resolved
    desired = _resolve_chat_web_ui_model_mode(gem_cfg, requested_model=requested_model)
    if desired == "pro":
        return ["pro", "thinking"]
    return [desired] if desired else ["pro", "thinking"]


def _resolve_chat_web_response_stall_sec(
    gem_cfg: Dict[str, Any],
    *,
    requested_model: str = "",
    current_mode: str = "",
    fallback: float = 45.0,
) -> float:
    desired_mode = _resolve_chat_web_ui_model_mode(gem_cfg, requested_model=str(requested_model or "").strip())
    effective_mode = _normalize_chat_web_model_mode(current_mode) or desired_mode
    mode_specific = gem_cfg.get(f"chat_web_response_stall_sec_{effective_mode}", None)
    if mode_specific not in (None, ""):
        try:
            return max(8.0, float(mode_specific))
        except Exception:
            pass
    explicit = gem_cfg.get("chat_web_response_stall_sec", None)
    if explicit not in (None, ""):
        try:
            return max(8.0, float(explicit))
        except Exception:
            pass
    defaults = {
        "fast": 35.0,
        "pro": 75.0,
        "thinking": 180.0,
    }
    return max(8.0, float(defaults.get(effective_mode, fallback if fallback > 0 else 45.0)))


def _build_chat_web_launch_args(raw_args: Any) -> List[str]:
    args: List[str] = []
    if isinstance(raw_args, list):
        for item in raw_args:
            val = str(item or "").strip()
            if val:
                args.append(val)
    default_args = [
        "--disable-blink-features=AutomationControlled",
    ]
    for item in default_args:
        if item not in args:
            args.append(item)
    return args


def _apply_chat_web_stealth(context: Any) -> None:
    try:
        context.add_init_script(
            """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
window.chrome = window.chrome || { runtime: {} };
"""
        )
    except Exception:
        return


def _resolve_usage_log_path(cfg: Dict[str, Any]) -> Path:
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
    gem_cfg = cfg.get("gemini", {}) if isinstance(cfg.get("gemini"), dict) else {}
    out_dir_raw = str(run_cfg.get("output_dir", "outputs") or "outputs").strip() or "outputs"
    usage_name = str(gem_cfg.get("usage_log_file", "gemini_usage.jsonl") or "gemini_usage.jsonl").strip() or "gemini_usage.jsonl"
    out_dir = Path(out_dir_raw)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / usage_name


def _log_gemini_usage(
    cfg: Dict[str, Any],
    *,
    model: str,
    mode: str,
    usage_meta: Dict[str, Any],
    key_source: str,
) -> None:
    if not isinstance(usage_meta, dict):
        return

    prompt_tokens = _safe_int(
        usage_meta.get("promptTokenCount", usage_meta.get("prompt_tokens", 0)),
        0,
    )
    output_tokens = _safe_int(
        usage_meta.get("candidatesTokenCount", usage_meta.get("output_tokens", 0)),
        0,
    )
    total_tokens = _safe_int(
        usage_meta.get("totalTokenCount", usage_meta.get("total_tokens", 0)),
        0,
    )
    if total_tokens <= 0:
        total_tokens = max(0, prompt_tokens + output_tokens)
    if prompt_tokens <= 0 and output_tokens <= 0 and total_tokens <= 0:
        return

    est_cost = estimate_cost_usd(
        cfg,
        model,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )

    row = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model or "").strip(),
        "mode": str(mode or "").strip() or "triplet_compare",
        "key_source": str(key_source or "").strip() or "unknown",
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(est_cost, 8),
    }

    usage_log_path = _resolve_usage_log_path(cfg)
    try:
        with usage_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        # Cost logging should not break the main workflow.
        return


def _join_text_blocks(*values: Any) -> str:
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text:
            out.append(text)
    return "\n\n".join(out).strip()


def _read_optional_text_file(path_value: Any, *, base_dir: Path) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.exists() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _build_prompt_context(gem_cfg: Dict[str, Any], *, cfg_dir: Path, scope: str) -> str:
    if bool(gem_cfg.get("use_vertex_cached_context_only", False)):
        return ""
    scope_norm = str(scope or "").strip().lower()
    return _join_text_blocks(
        gem_cfg.get("context_text", ""),
        _read_optional_text_file(gem_cfg.get("context_file", ""), base_dir=cfg_dir),
        gem_cfg.get(f"{scope_norm}_context_text", ""),
        _read_optional_text_file(gem_cfg.get(f"{scope_norm}_context_file", ""), base_dir=cfg_dir),
    )


def _resolve_system_instruction_text(
    gem_cfg: Dict[str, Any],
    *,
    cfg_dir: Path,
    scope: str,
    alias_text_keys: Optional[List[str]] = None,
    alias_file_keys: Optional[List[str]] = None,
) -> str:
    scope_norm = str(scope or "").strip().lower()
    alias_text_keys = alias_text_keys or []
    alias_file_keys = alias_file_keys or []

    text_candidates: List[str] = [
        str(gem_cfg.get(f"{scope_norm}_system_instruction_text", "") or "").strip(),
        *[str(gem_cfg.get(k, "") or "").strip() for k in alias_text_keys],
        str(gem_cfg.get("system_instruction_text", "") or "").strip(),
    ]
    for txt in text_candidates:
        if txt:
            return txt

    file_candidates: List[str] = [
        str(gem_cfg.get(f"{scope_norm}_system_instruction_file", "") or "").strip(),
        *[str(gem_cfg.get(k, "") or "").strip() for k in alias_file_keys],
        str(gem_cfg.get("system_instruction_file", "") or "").strip(),
        "prompts/system_prompt.txt",
    ]
    for file_ref in file_candidates:
        txt = _read_optional_text_file(file_ref, base_dir=cfg_dir)
        if txt:
            # AUTO-APPEND DYNAMIC UPDATES (Phase 2 Intelligence)
            dynamic_path = cfg_dir / "prompts" / "atlas_discord_updates.md"
            if dynamic_path.exists():
                try:
                    dynamic_txt = dynamic_path.read_text(encoding="utf-8").strip()
                    if dynamic_txt:
                        txt += "\n\n# 🧠 DYNAMIC DISCORD UPDATES (LATEST INTELLIGENCE):\n" + dynamic_txt
                except Exception:
                    pass
            return txt
    return ""


def _normalize_auth_mode(raw: str) -> str:
    mode = str(raw or "").strip().lower()
    aliases = {
        "": "api_key",
        "api": "api_key",
        "apikey": "api_key",
        "api_key": "api_key",
        "google_api_key": "api_key",
        "vertex": "vertex_ai",
        "vertexai": "vertex_ai",
        "vertex_ai": "vertex_ai",
        "service_account": "vertex_ai",
        "chat": "chat_web",
        "chat_web": "chat_web",
        "gemini_chat": "chat_web",
        "gemini_web": "chat_web",
        "playwright": "chat_web",
    }
    return aliases.get(mode, "api_key")


def _normalize_vertex_model_id(model: str) -> str:
    m = str(model or "").strip()
    aliases = {
        "gemini-3.1-pro-preview": "gemini-3-pro-preview",
        "gemini-3.1-flash-preview": "gemini-3-flash-preview",
    }
    return aliases.get(m, m)


def _extract_folder_id(link: str) -> Optional[str]:
    src = str(link or "")
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", src)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", src)
    if m:
        return m.group(1)
    return None


def _parse_drive_folder_file_ref(ref: str) -> Optional[Tuple[str, str]]:
    src = str(ref or "").strip()
    if "drive.google.com/" not in src or "/folders/" not in src:
        return None
    folder_id = _extract_folder_id(src)
    if not folder_id:
        return None

    normalized = src.replace("\\", "/")
    last = normalized.rsplit("/", 1)[-1]
    filename = last.split("?", 1)[0].strip()
    if not filename or "." not in filename:
        return None
    return folder_id, filename


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, timeout=600)


def _infer_drive_account_subdir(cfg: Dict[str, Any]) -> str:
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
    raw_output = str(run_cfg.get("output_dir", "") or "").strip()
    if raw_output:
        parts = [part for part in Path(raw_output).parts if str(part or "").strip()]
        if "outputs" in parts:
            idx = parts.index("outputs")
            if idx + 1 < len(parts):
                candidate = str(parts[idx + 1] or "").strip()
                if candidate:
                    return candidate
    for env_name in ("ATLAS_ACCOUNT_NAME", "ATLAS_ACCOUNT_SLUG"):
        candidate = str(os.environ.get(env_name, "") or "").strip()
        if candidate:
            return candidate
    return "danatimer"


def _rclone_remote_available(remote_name: str) -> tuple[bool, str]:
    remote = str(remote_name or "").strip()
    if not remote:
        return False, "remote_missing"
    if shutil.which("rclone") is None:
        return False, "rclone_missing"
    try:
        probe = subprocess.run(
            ["rclone", "listremotes"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except Exception as exc:
        return False, f"rclone_probe_failed:{exc}"
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "").strip()
        detail = re.sub(r"\s+", " ", detail)
        return False, f"rclone_probe_failed:{detail[:180]}"
    available = {
        str(line or "").strip().rstrip(":")
        for line in (probe.stdout or "").splitlines()
        if str(line or "").strip()
    }
    if remote not in available:
        return False, f"rclone_remote_missing:{remote}"
    return True, ""


def _should_disable_drive_picker_after_stage(stage_notes: List[str]) -> bool:
    notes = [str(note or "").strip() for note in (stage_notes or []) if str(note or "").strip()]
    if not notes:
        return False
    if any(note.startswith("drive_stage_uploaded=") for note in notes):
        return False
    return any(
        note.startswith("drive_stage_skipped:")
        or note.startswith("drive_stage_failed=")
        or note.startswith("drive_stage_disabled:")
        for note in notes
    )


def _should_skip_chat_web_shutdown(*, raw_text: str, gem_cfg: Dict[str, Any]) -> bool:
    if not str(raw_text or "").strip():
        return False
    return bool(gem_cfg.get("chat_web_skip_shutdown_after_success", True))


def _is_chat_web_boot_error_text(text: str) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    markers = (
        "connect_over_cdp",
        "browsertype.connect_over_cdp",
        "page.goto: timeout",
        'navigating to "https://gemini.google.com/',
        "playwright sync api inside the asyncio loop",
        "please use the async api instead",
        "timed out waiting for chromium to connect",
    )
    return any(marker in low for marker in markers)


def _should_force_storage_state_after_cdp_failure(
    *,
    cdp_launch_error: str,
    storage_state: str,
    user_data_dir: str = "",
) -> bool:
    return bool(
        str(cdp_launch_error or "").strip()
        and str(storage_state or "").strip()
        and not str(user_data_dir or "").strip()
    )


def _effective_timed_labels_retry_attempts(*, requested_attempts: int, auth_mode: str) -> int:
    attempts = max(1, int(requested_attempts or 1))
    if _normalize_auth_mode(auth_mode) == "chat_web":
        return 1
    return attempts


def _stage_episode_artifacts_for_drive_picker(
    cfg: Dict[str, Any],
    *,
    episode_id: str,
    paths: List[Path],
) -> List[str]:
    eid = str(episode_id or "").strip().lower()
    if not eid:
        return []
    remote_name = str(os.environ.get("RCLONE_REMOTE", "gdrive") or "gdrive").strip() or "gdrive"
    remote_ok, remote_reason = _rclone_remote_available(remote_name)
    if not remote_ok:
        return [f"drive_stage_disabled:{remote_reason}"]

    remote_root = str(os.environ.get("RCLONE_PATH", "OCR_annotation_Atlas/vps_outputs") or "").strip().strip("/")
    root_folder_id = str(os.environ.get("GDRIVE_ROOT_FOLDER_ID", "") or "").strip()
    account_subdir = _infer_drive_account_subdir(cfg)
    remote_episode_dir = "/".join(
        part for part in [remote_root, account_subdir, "episodes", eid] if str(part or "").strip()
    )

    notes: List[str] = []
    for path in paths:
        if path is None or not path.exists() or not path.is_file():
            continue
        target = f"{remote_name}:{remote_episode_dir}/{path.name}"
        cmd = [
            "rclone",
            "copyto",
            str(path),
            target,
            "--checkers",
            "4",
            "--transfers",
            "2",
            "--retries",
            "5",
            "--low-level-retries",
            "10",
        ]
        if root_folder_id:
            cmd += ["--drive-root-folder-id", root_folder_id]
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
            if completed.returncode == 0:
                notes.append(f"drive_stage_uploaded={path.name}")
                continue
            detail = (completed.stderr or completed.stdout or f"exit_code={completed.returncode}").strip()
            detail = re.sub(r"\s+", " ", detail)
            notes.append(f"drive_stage_failed={path.name}:{detail[:220]}")
        except Exception as exc:
            notes.append(f"drive_stage_failed={path.name}:{exc}")
    return notes


def _download_from_drive_ref(
    ref: str,
    out_dir: Path,
    remote: str,
) -> Path:
    parsed = _parse_drive_folder_file_ref(ref)
    if not parsed:
        raise RuntimeError(f"Unsupported Drive reference format: {ref}")
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone is required for Drive references but was not found in PATH.")

    folder_id, filename = parsed
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rclone",
        "copy",
        f"{remote}:",
        str(out_dir),
        "--drive-root-folder-id",
        folder_id,
        "--include",
        filename,
        "--checkers",
        "8",
        "--transfers",
        "4",
        "--progress",
    ]
    _run(cmd)
    found = list(out_dir.rglob(filename))
    if not found:
        raise RuntimeError(f"Drive download completed but file was not found locally: {filename}")
    return found[0]


def _resolve_input_path(ref: str, cache_dir: Path, remote: str) -> Path:
    raw = str(ref or "").strip()
    if not raw:
        raise RuntimeError("Empty input reference.")
    p = Path(raw)
    if p.exists():
        return p.resolve()
    if "drive.google.com/" in raw and "/folders/" in raw:
        return _download_from_drive_ref(raw, cache_dir, remote).resolve()
    raise RuntimeError(f"Input path is not local file and not supported Drive reference: {raw}")


def _load_text_or_json(path: Path) -> str:
    if not path.exists():
        return ""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() != ".json":
        return raw
    try:
        payload = json.loads(raw)
    except Exception:
        return raw
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _compress_video_for_inline(src: Path, out_dir: Path) -> Optional[Path]:
    if shutil.which("ffmpeg") is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_inline.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        "scale='min(960,iw)':-2:flags=lanczos,fps=8",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "30",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    try:
        _run(cmd)
    except Exception:
        return None
    if not dst.exists() or dst.stat().st_size <= 0:
        return None
    return dst


def _video_part_for_inline(
    path: Path,
    max_inline_mb: Optional[float] = None,
    cache_dir: Optional[Path] = None,
    **kwargs: Any,
) -> Tuple[Optional[Dict[str, Any]], str]:
    # Backward-compatible kwargs support:
    # - old callsites may pass max_mb=...
    # - newer callsites pass max_inline_mb=...
    max_mb_raw = kwargs.pop("max_mb", None)
    max_mb = float(max_mb_raw if max_mb_raw is not None else (max_inline_mb if max_inline_mb is not None else 20.0))
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "triplet_inline_cache"
    src = path
    size_mb = src.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        compressed = _compress_video_for_inline(src, cache_dir)
        if compressed is None:
            return None, f"{src.name}: skipped (size {size_mb:.1f} MB > {max_mb:.1f} MB and ffmpeg unavailable/failed)"
        src = compressed
        size_mb = src.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        return None, f"{src.name}: skipped after compression ({size_mb:.1f} MB > {max_mb:.1f} MB)"
    data = base64.b64encode(src.read_bytes()).decode("ascii")
    return {"inline_data": {"mime_type": "video/mp4", "data": data}}, f"{src.name}: attached ({size_mb:.1f} MB)"


def _extract_text_from_response_json(data: Dict[str, Any]) -> str:
    for cand in data.get("candidates", []):
        content = cand.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return ""


def _parse_time_like_to_sec(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        sec = float(value)
        if sec >= 0:
            return sec
        return None
    src = str(value or "").strip()
    if not src:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", src):
        try:
            sec = float(src)
            if sec >= 0:
                return sec
        except Exception:
            return None
        return None
    parts = src.split(":")
    if len(parts) not in {2, 3}:
        return None
    nums: List[float] = []
    for part in parts:
        if not re.fullmatch(r"\d+(?:\.\d+)?", part.strip()):
            return None
        try:
            nums.append(float(part.strip()))
        except Exception:
            return None
    if len(nums) == 2:
        mm, ss = nums
        return mm * 60.0 + ss
    hh, mm, ss = nums
    return hh * 3600.0 + mm * 60.0 + ss


def _format_time_sec(value: float) -> str:
    txt = f"{float(value):.3f}".rstrip("0").rstrip(".")
    if "." not in txt:
        txt += ".0"
    return txt


def _segment_from_obj(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    start_raw = (
        item.get("start_sec")
        if isinstance(item, dict)
        else None
    )
    if start_raw is None:
        start_raw = item.get("start") if isinstance(item, dict) else None
    if start_raw is None:
        start_raw = item.get("start_time") if isinstance(item, dict) else None
    if start_raw is None:
        start_raw = item.get("from") if isinstance(item, dict) else None
    if start_raw is None:
        start_raw = item.get("t0") if isinstance(item, dict) else None

    end_raw = (
        item.get("end_sec")
        if isinstance(item, dict)
        else None
    )
    if end_raw is None:
        end_raw = item.get("end") if isinstance(item, dict) else None
    if end_raw is None:
        end_raw = item.get("end_time") if isinstance(item, dict) else None
    if end_raw is None:
        end_raw = item.get("to") if isinstance(item, dict) else None
    if end_raw is None:
        end_raw = item.get("t1") if isinstance(item, dict) else None

    label_raw: Any = ""
    if isinstance(item, dict):
        label_raw = item.get("label")
        if label_raw in (None, ""):
            label_raw = item.get("action")
        if label_raw in (None, ""):
            label_raw = item.get("text")
        if label_raw in (None, "") and isinstance(item.get("actions"), list):
            label_raw = ", ".join(str(x or "").strip() for x in item.get("actions", []) if str(x or "").strip())
    label = str(label_raw or "").strip()

    a = _parse_time_like_to_sec(start_raw)
    b = _parse_time_like_to_sec(end_raw)
    if a is None or b is None:
        return None
    if b <= a:
        return None
    return {"start_sec": a, "end_sec": b, "label": label}


def parse_timed_segments_payload(payload: Any) -> List[Dict[str, Any]]:
    obj = payload
    if isinstance(obj, dict):
        if isinstance(obj.get("segments"), list):
            obj = obj.get("segments")
        elif isinstance(obj.get("labels"), list):
            obj = obj.get("labels")
        elif isinstance(obj.get("data"), dict) and isinstance(obj.get("data", {}).get("segments"), list):
            obj = obj.get("data", {}).get("segments")
    if not isinstance(obj, list):
        return []

    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for item in obj:
        if not isinstance(item, dict):
            continue
        seg = _segment_from_obj(item)
        if not seg:
            continue
        key = (
            _format_time_sec(seg["start_sec"]),
            _format_time_sec(seg["end_sec"]),
            str(seg.get("label") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(seg)
    out.sort(key=lambda s: (float(s.get("start_sec", 0.0)), float(s.get("end_sec", 0.0))))
    return out


def parse_timed_segments_text(raw_text: str) -> List[Dict[str, Any]]:
    src = str(raw_text or "")
    if not src.strip():
        return []

    # JSON first (most reliable).
    try:
        parsed = json.loads(_clean_json_text(src))
        segs = parse_timed_segments_payload(parsed)
        if segs:
            return segs
    except Exception:
        pass

    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    lines = [ln.strip() for ln in src.replace("\r", "\n").split("\n") if str(ln or "").strip()]
    for line in lines:
        # Tab-separated forms:
        # 1\t0.0\t4.2\tlabel
        # 0.0\t4.2\tlabel
        parts = [p.strip() for p in re.split(r"\t+", line) if p.strip()]
        candidates: List[Tuple[str, str, str]] = []
        if len(parts) >= 4:
            candidates.append((parts[1], parts[2], " ".join(parts[3:]).strip()))
        if len(parts) >= 3:
            candidates.append((parts[0], parts[1], " ".join(parts[2:]).strip()))

        # Markdown table-like row:
        # | 1 | 0.0 | 4.2 | label |
        pipe_parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(pipe_parts) >= 4:
            candidates.append((pipe_parts[1], pipe_parts[2], " ".join(pipe_parts[3:]).strip()))
        elif len(pipe_parts) >= 3:
            candidates.append((pipe_parts[0], pipe_parts[1], " ".join(pipe_parts[2:]).strip()))

        # Time range in free text:
        # 00:00.0 -> 00:04.2 label
        m = re.search(
            r"(?P<a>\d{1,2}:\d{1,2}(?::\d{1,2})?(?:\.\d+)?)\s*(?:->|=>|[-–—])\s*(?P<b>\d{1,2}:\d{1,2}(?::\d{1,2})?(?:\.\d+)?)",
            line,
        )
        if m:
            end_idx = m.end("b")
            candidates.append((m.group("a"), m.group("b"), line[end_idx:].strip()))

        for a_raw, b_raw, label_raw in candidates:
            a = _parse_time_like_to_sec(a_raw)
            b = _parse_time_like_to_sec(b_raw)
            if a is None or b is None or b <= a:
                continue
            label = str(label_raw or "").strip()
            key = (_format_time_sec(a), _format_time_sec(b), label)
            if key in seen:
                continue
            seen.add(key)
            out.append({"start_sec": a, "end_sec": b, "label": label})
            break
    out.sort(key=lambda s: (float(s.get("start_sec", 0.0)), float(s.get("end_sec", 0.0))))
    return out


def segments_to_timed_text(segments: List[Dict[str, Any]]) -> str:
    normalized = parse_timed_segments_payload({"segments": segments})
    lines: List[str] = []
    for idx, seg in enumerate(normalized, 1):
        label = str(seg.get("label") or "").strip() or "No Action"
        lines.append(
            f"{idx}\t{_format_time_sec(float(seg['start_sec']))}\t{_format_time_sec(float(seg['end_sec']))}\t{label}"
        )
    return "\n".join(lines).strip()


def _fill_timeline_gaps_with_no_action(
    segments: List[Dict[str, Any]],
    *,
    start_at_zero: bool = True,
    gap_epsilon_sec: float = 0.05,
) -> List[Dict[str, Any]]:
    normalized = parse_timed_segments_payload({"segments": segments})
    if not normalized:
        return []

    out: List[Dict[str, Any]] = []
    cursor = 0.0 if start_at_zero else max(0.0, float(normalized[0].get("start_sec", 0.0)))
    eps = max(0.0, float(gap_epsilon_sec))

    for seg in normalized:
        start = max(0.0, float(seg.get("start_sec", 0.0)))
        end = max(start, float(seg.get("end_sec", start)))
        label = str(seg.get("label") or "").strip() or "No Action"
        if end <= start:
            continue

        if start > cursor + eps:
            out.append(
                {
                    "start_sec": cursor,
                    "end_sec": start,
                    "label": "No Action",
                }
            )
            cursor = start
        elif start < cursor:
            start = cursor
            if end <= start:
                continue

        out.append(
            {
                "start_sec": start,
                "end_sec": end,
                "label": label,
            }
        )
        cursor = end
    return out


def _timed_labels_response_schema() -> Dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "segments": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "start_sec": {"type": "NUMBER"},
                        "end_sec": {"type": "NUMBER"},
                        "label": {"type": "STRING"},
                    },
                    "required": ["start_sec", "end_sec", "label"],
                },
            }
        },
        "required": ["segments"],
    }


def _triplet_compare_response_schema(include_thought_process: bool) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "winner": {"type": "STRING", "enum": ["tier2", "api", "chat", "vertex_chat", "none"]},
        "submit_safe_solution": {"type": "STRING", "enum": ["tier2", "api", "chat", "vertex_chat", "none"]},
        "scores": {
            "type": "OBJECT",
            "properties": {
                "tier2": {"type": "INTEGER"},
                "api": {"type": "INTEGER"},
                "chat": {"type": "INTEGER"},
                "vertex_chat": {"type": "INTEGER"},
            },
            "required": ["tier2", "api", "chat", "vertex_chat"],
        },
        "hallucination": {
            "type": "OBJECT",
            "properties": {
                "tier2": {"type": "BOOLEAN"},
                "api": {"type": "BOOLEAN"},
                "chat": {"type": "BOOLEAN"},
                "vertex_chat": {"type": "BOOLEAN"},
            },
            "required": ["tier2", "api", "chat", "vertex_chat"],
        },
        "major_issues": {
            "type": "OBJECT",
            "properties": {
                "tier2": {"type": "ARRAY", "items": {"type": "STRING"}},
                "api": {"type": "ARRAY", "items": {"type": "STRING"}},
                "chat": {"type": "ARRAY", "items": {"type": "STRING"}},
                "vertex_chat": {"type": "ARRAY", "items": {"type": "STRING"}},
            },
            "required": ["tier2", "api", "chat", "vertex_chat"],
        },
        "best_reason_short": {"type": "STRING"},
        "final_recommendation": {"type": "STRING"},
    }
    required = [
        "winner",
        "submit_safe_solution",
        "scores",
        "hallucination",
        "major_issues",
        "best_reason_short",
        "final_recommendation",
    ]
    if include_thought_process:
        props["thought_process"] = {"type": "STRING"}
        required.insert(0, "thought_process")
    return {"type": "OBJECT", "properties": props, "required": required}


def _validate_triplet_judge_result(parsed: Any, *, require_thought_process: bool) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        raise RuntimeError("judge_result is not a JSON object.")
    allowed = {"tier2", "api", "chat", "vertex_chat", "none"}
    out: Dict[str, Any] = {}

    thought_process = str(parsed.get("thought_process") or "").strip()
    if require_thought_process and not thought_process:
        raise RuntimeError("Missing thought_process.")
    if thought_process:
        out["thought_process"] = thought_process

    winner = str(parsed.get("winner") or "").strip().lower()
    submit_safe = str(parsed.get("submit_safe_solution") or "").strip().lower()
    if winner not in allowed:
        raise RuntimeError(f"Invalid winner={winner!r}")
    if submit_safe not in allowed:
        raise RuntimeError(f"Invalid submit_safe_solution={submit_safe!r}")
    out["winner"] = winner
    out["submit_safe_solution"] = submit_safe

    scores = parsed.get("scores")
    if not isinstance(scores, dict):
        raise RuntimeError("scores must be object.")
    score_out: Dict[str, int] = {}
    for key in ("tier2", "api", "chat", "vertex_chat"):
        raw = scores.get(key)
        if raw is None:
            raise RuntimeError(f"scores.{key} missing.")
        try:
            score_out[key] = max(0, min(100, int(float(raw))))
        except Exception as exc:
            raise RuntimeError(f"scores.{key} invalid numeric value.") from exc
    out["scores"] = score_out

    hallucination = parsed.get("hallucination")
    if not isinstance(hallucination, dict):
        raise RuntimeError("hallucination must be object.")
    out["hallucination"] = {
        k: bool(hallucination.get(k, False))
        for k in ("tier2", "api", "chat", "vertex_chat")
    }

    major_issues = parsed.get("major_issues")
    if not isinstance(major_issues, dict):
        raise RuntimeError("major_issues must be object.")
    issues_out: Dict[str, List[str]] = {}
    for key in ("tier2", "api", "chat", "vertex_chat"):
        raw_list = major_issues.get(key, [])
        if raw_list is None:
            raw_list = []
        if not isinstance(raw_list, list):
            raise RuntimeError(f"major_issues.{key} must be array.")
        issues_out[key] = [str(v).strip() for v in raw_list if str(v).strip()]
    out["major_issues"] = issues_out

    out["best_reason_short"] = str(parsed.get("best_reason_short") or "").strip()
    out["final_recommendation"] = str(parsed.get("final_recommendation") or "").strip()
    return out


def _translate_payload_for_vertex(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            mapped = key
            if key == "inline_data":
                mapped = "inlineData"
            elif key == "mime_type":
                mapped = "mimeType"
            out[mapped] = _translate_payload_for_vertex(item)
        return out
    if isinstance(value, list):
        return [_translate_payload_for_vertex(item) for item in value]
    return value


def _selector_variants(expr: str) -> List[str]:
    out: List[str] = []
    for part in str(expr or "").split("||"):
        candidate = str(part or "").strip()
        if candidate:
            out.append(candidate)
    return out


def _first_visible_locator(page: Any, selector_expr: str, timeout_ms: int = 15000) -> Optional[Any]:
    variants = _selector_variants(selector_expr)
    if not variants:
        return None
    per_variant = max(500, int(timeout_ms / max(1, len(variants))))
    for sel in variants:
        try:
            loc = _first_visible_candidate(page.locator(sel), timeout_ms=per_variant)
            if loc is not None:
                return loc
        except Exception:
            continue
    return None


def _handle_gemini_consent_if_present(page: Any, timeout_ms: int = 12000) -> bool:
    try:
        page.wait_for_timeout(1500)
    except Exception:
        pass

    # Aggressively click known Gemini pop-up blockers before checking main body text (bypasses Shadow DOM limitations)
    aggressive_candidates = [
        '[data-test-id="upload-image-agree-button"]',
        'button:has-text("Agree")',
        'button:has-text("Dismiss")',
        'button:has-text("Got it")',
        '[role="button"]:has-text("Dismiss")',
        '[role="button"]:has-text("Got it")',
    ]
    for selector in aggressive_candidates:
        try:
            loc = page.locator(selector).first
            if loc.is_visible(timeout=1000):
                try:
                    loc.click(timeout=3000)
                    page.wait_for_timeout(1000)
                    return True
                except Exception:
                    pass
        except Exception:
            continue

    current_url = ""
    try:
        current_url = str(page.url or "")
    except Exception:
        current_url = ""
    consent_like = "consent.google.com" in current_url.lower()
    body_text = ""
    if not consent_like:
        try:
            body_text = page.locator("body").inner_text(timeout=2000)
        except Exception:
            body_text = ""
        consent_like = (
            "Before you continue" in body_text
            or "Google services" in body_text
            or "Creating content from images and files" in body_text
            or "Prohibited Use Policy" in body_text
        )
    if not consent_like:
        return False

    button_candidates = [
        '[data-test-id="upload-image-agree-button"]',
        'button:has-text("Agree")',
        'button:has-text("Dismiss")',
        'button:has-text("Got it")',
        'button:has-text("Reject all")',
        'button:has-text("Accept all")',
        'button:has-text("I agree")',
        'button:has-text("Continue")',
        '[role="button"]:has-text("Reject all")',
        '[role="button"]:has-text("Accept all")',
    ]
    for selector in button_candidates:
        try:
            loc = page.locator(selector).first
            if loc.count() and loc.is_visible(timeout=1500):
                try:
                    loc.click(timeout=timeout_ms)
                except Exception:
                    try:
                        loc.click(timeout=timeout_ms, force=True)
                    except Exception:
                        loc.evaluate("(el) => el.click()")
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
                except Exception:
                    pass
                try:
                    page.wait_for_timeout(3000)
                except Exception:
                    pass
                return True
        except Exception:
            continue
    return False


def _last_present_locator(page: Any, selector_expr: str) -> Optional[Any]:
    variants = _selector_variants(selector_expr)
    if not variants:
        return None
    for sel in variants:
        try:
            loc = page.locator(sel)
            count = int(loc.count() or 0)
        except Exception:
            continue
        if count <= 0:
            continue
        try:
            return loc.nth(count - 1)
        except Exception:
            continue
    return None


def _try_set_files_on_locator(file_input: Any, file_path: Path) -> Tuple[bool, str]:
    try:
        file_input.set_input_files(str(file_path))
        return True, "file_input"
    except Exception as exc:
        return False, str(exc)


def _infer_episode_id_from_paths(*paths: Any) -> str:
    for raw in paths:
        if not raw:
            continue
        try:
            name = Path(str(raw)).name
        except Exception:
            name = str(raw or "")
        m = re.search(r"video_([0-9a-f]{16,})", str(name or ""), flags=re.IGNORECASE)
        if m:
            return str(m.group(1) or "").strip().lower()
    return ""


def _safe_bounding_box(locator: Any) -> Optional[Dict[str, float]]:
    try:
        box = locator.bounding_box()
    except Exception:
        return None
    if not isinstance(box, dict):
        return None
    try:
        return {
            "x": float(box.get("x", 0.0) or 0.0),
            "y": float(box.get("y", 0.0) or 0.0),
            "width": float(box.get("width", 0.0) or 0.0),
            "height": float(box.get("height", 0.0) or 0.0),
        }
    except Exception:
        return None


def _has_usable_box(locator: Any, *, min_width: float = 8.0, min_height: float = 8.0) -> bool:
    box = _safe_bounding_box(locator)
    if box is None:
        return False
    return float(box.get("width", 0.0) or 0.0) >= float(min_width) and float(box.get("height", 0.0) or 0.0) >= float(min_height)


def _last_visible_candidate(locator: Any, timeout_ms: int = 1500) -> Optional[Any]:
    try:
        count = int(locator.count() or 0)
    except Exception:
        return None
    if count <= 0:
        return None
    fallback = None
    for idx in range(count - 1, -1, -1):
        try:
            candidate = locator.nth(idx)
            candidate.wait_for(state="visible", timeout=timeout_ms)
            if _has_usable_box(candidate):
                return candidate
            if fallback is None:
                fallback = candidate
        except Exception:
            continue
    return fallback


def _first_visible_candidate(locator: Any, timeout_ms: int = 1500) -> Optional[Any]:
    try:
        count = int(locator.count() or 0)
    except Exception:
        return None
    if count <= 0:
        return None
    fallback = None
    candidate_timeout = max(200, int(timeout_ms / max(1, min(count, 4))))
    for idx in range(count):
        try:
            candidate = locator.nth(idx)
            candidate.wait_for(state="visible", timeout=candidate_timeout)
            if _has_usable_box(candidate):
                return candidate
            if fallback is None:
                fallback = candidate
        except Exception:
            continue
    return fallback


def _first_exact_upload_trigger(page: Any) -> Optional[Any]:
    candidates = [
        lambda: page.locator('button[aria-controls="upload-file-menu"]'),
        lambda: page.locator('button.upload-card-button'),
        lambda: page.locator('uploader button[mat-icon-button]'),
        lambda: page.get_by_role("button", name="Open upload file menu"),
        lambda: page.get_by_role("button", name="Tools"),
        lambda: page.get_by_role("button", name=re.compile(r"Upload|Add files", re.IGNORECASE)),
        lambda: page.locator('button[aria-label*="Open upload file menu" i]'),
        lambda: page.locator('button[aria-label*="Upload files" i]'),
        lambda: page.locator('button[aria-label*="Tools" i]'),
    ]
    for factory in candidates:
        try:
            loc = _last_visible_candidate(factory(), timeout_ms=5000)
            if loc is None:
                continue
            return loc
        except Exception:
            continue
    return None


def _first_exact_upload_item(page: Any) -> Optional[Any]:
    candidates = [
        lambda: page.get_by_role("menuitem", name="Upload files"),
        lambda: page.get_by_text("Upload files", exact=True),
        lambda: page.locator('text=/^Upload files$/i'),
    ]
    for factory in candidates:
        try:
            loc = _last_visible_candidate(factory(), timeout_ms=4500)
            if loc is None:
                continue
            return loc
        except Exception:
            continue
    return None


def _first_exact_drive_item(page: Any) -> Optional[Any]:
    candidates = [
        lambda: page.get_by_role("menuitem", name="Add from Drive"),
        lambda: page.get_by_text("Add from Drive", exact=True),
        lambda: page.locator('text=/^Add from Drive$/i'),
    ]
    for factory in candidates:
        try:
            loc = _last_visible_candidate(factory(), timeout_ms=4500)
            if loc is None:
                continue
            return loc
        except Exception:
            continue
    return None


def _robust_click(locator: Any) -> None:
    try:
        locator.scroll_into_view_if_needed(timeout=1200)
    except Exception:
        pass
    click_errors: List[str] = []
    for clicker in (
        lambda: locator.click(timeout=5000, force=True, no_wait_after=True),
        lambda: locator.click(timeout=5000, no_wait_after=True),
        lambda: locator.evaluate("el => el.click()"),
    ):
        try:
            clicker()
            return
        except Exception as exc:
            click_errors.append(str(exc))
            continue
    raise RuntimeError("; ".join([err for err in click_errors if err]) or "click failed")


def _prepare_chat_composer_for_attach(page: Any, composer_locator: Optional[Any]) -> None:
    if composer_locator is None:
        return
    try:
        page.keyboard.press("Escape")
    except Exception:
        pass
    try:
        page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
    except Exception:
        pass
    try:
        page.wait_for_timeout(250)
    except Exception:
        pass
    for _ in range(2):
        try:
            composer_locator.scroll_into_view_if_needed(timeout=2500)
        except Exception:
            pass
        try:
            composer_locator.click(timeout=3500, force=True)
        except Exception:
            try:
                composer_locator.click(timeout=3500)
            except Exception:
                pass
        try:
            page.wait_for_timeout(300)
        except Exception:
            pass
    try:
        _clear_local_chat_attachments(page, composer_locator)
    except Exception:
        pass


def _wait_for_upload_ready(
    page: Any,
    *,
    attach_button_sel: str,
    file_input_sel: str,
    timeout_ms: int = 6000,
    heartbeat: Optional[Callable[[], None]] = None,
    heartbeat_interval_sec: float = 10.0,
) -> Tuple[Optional[Any], Optional[Any]]:
    deadline = time.monotonic() + max(0.5, float(timeout_ms) / 1000.0)
    last_heartbeat_ts = time.monotonic()
    while True:
        attach_trigger = _first_exact_upload_trigger(page) or _first_visible_locator(
            page, attach_button_sel, timeout_ms=500
        )
        file_input = _last_present_locator(page, file_input_sel)
        if attach_trigger is not None or file_input is not None:
            return attach_trigger, file_input
        now = time.monotonic()
        if now >= deadline:
            return None, None
        if heartbeat is not None and (now - last_heartbeat_ts) >= max(2.0, float(heartbeat_interval_sec or 10.0)):
            try:
                heartbeat()
            except Exception:
                pass
            last_heartbeat_ts = now
        try:
            page.wait_for_timeout(250)
        except Exception:
            pass


def _clear_local_chat_attachments(page: Any, composer_locator: Optional[Any], *, max_passes: int = 4) -> int:
    if composer_locator is None:
        return 0
    composer_box = _safe_bounding_box(composer_locator)
    if composer_box is None:
        return 0
    composer_top = float(composer_box.get("y", 0.0) or 0.0)
    composer_bottom = composer_top + float(composer_box.get("height", 0.0) or 0.0)
    selectors = [
        'button[aria-label*="Remove file" i]',
        'button[aria-label*="Remove attachment" i]',
        '[role="button"][aria-label*="Remove file" i]',
        '[role="button"][aria-label*="Remove attachment" i]',
        '[title*="Remove file" i]',
        '[title*="Remove attachment" i]',
    ]
    removed = 0
    for _ in range(max(1, int(max_passes or 1))):
        clicked = False
        for selector in selectors:
            try:
                loc = page.locator(selector)
                count = min(12, int(loc.count() or 0))
            except Exception:
                continue
            for idx in range(count):
                try:
                    node = loc.nth(idx)
                    if not node.is_visible():
                        continue
                    box = _safe_bounding_box(node)
                    if box is not None:
                        y = float(box.get("y", 0.0) or 0.0)
                        if y < composer_top - 280.0 or y > composer_bottom + 220.0:
                            continue
                    _robust_click(node)
                    removed += 1
                    clicked = True
                    try:
                        page.wait_for_timeout(250)
                    except Exception:
                        pass
                    break
                except Exception:
                    continue
            if clicked:
                break
        if not clicked:
            break
    return removed


def _try_attach_via_file_chooser(page: Any, clickable: Any, file_path: Path) -> Tuple[bool, str]:
    last_error = ""
    set_files_timeout_ms = _chooser_set_files_timeout_ms(file_path)
    for timeout_ms in (3500, 6500):
        try:
            with page.expect_file_chooser(timeout=timeout_ms) as chooser_info:
                _robust_click(clickable)
            chooser = chooser_info.value
            chooser.set_files(str(file_path), timeout=set_files_timeout_ms)
            return True, "file_chooser"
        except Exception as exc:
            last_error = str(exc)
            try:
                if _handle_gemini_consent_if_present(page, timeout_ms=max(timeout_ms, 4000)):
                    last_error = f"{last_error}; upload consent handled"
                    continue
            except Exception:
                pass
    return False, last_error or "file chooser did not appear"


def _chooser_set_files_timeout_ms(file_path: Path) -> int:
    try:
        size_mb = float(file_path.stat().st_size) / (1024 * 1024)
    except Exception:
        size_mb = 0.0
    return int(min(180000, max(60000, 60000 + (size_mb * 5000.0))))


def _reveal_hidden_upload_trigger(trigger: Any) -> None:
    try:
        trigger.evaluate(
            """
            el => {
                el.removeAttribute('aria-hidden');
                el.tabIndex = 0;
                el.style.setProperty('opacity', '1', 'important');
                el.style.setProperty('width', '40px', 'important');
                el.style.setProperty('height', '40px', 'important');
                el.style.setProperty('min-width', '40px', 'important');
                el.style.setProperty('min-height', '40px', 'important');
                el.style.setProperty('position', 'fixed', 'important');
                el.style.setProperty('left', '24px', 'important');
                el.style.setProperty('bottom', '24px', 'important');
                el.style.setProperty('z-index', '2147483647', 'important');
                el.style.setProperty('pointer-events', 'auto', 'important');
                el.style.setProperty('visibility', 'visible', 'important');
                el.style.setProperty('transform', 'none', 'important');
            }
            """
        )
    except Exception:
        pass
    try:
        trigger.scroll_into_view_if_needed(timeout=1500)
    except Exception:
        pass


def _try_hidden_local_file_trigger(page: Any, file_path: Path) -> Tuple[bool, str]:
    candidates = [
        lambda: page.locator('[data-test-id="hidden-local-file-upload-button"]'),
        lambda: page.locator('[data-test-id="hidden-local-image-upload-button"]'),
        lambda: page.locator('button.hidden-local-file-upload-button'),
        lambda: page.locator('button.hidden-local-upload-button'),
        lambda: page.locator('button[xapfileselectortrigger][data-test-id*="file-upload" i]'),
        lambda: page.locator('button[xapfileselectortrigger][data-test-id*="image-upload" i]'),
    ]
    last_error = ""
    set_files_timeout_ms = _chooser_set_files_timeout_ms(file_path)
    for factory in candidates:
        try:
            raw = factory()
        except Exception as exc:
            last_error = str(exc)
            continue
        try:
            count = int(raw.count() or 0)
        except Exception as exc:
            last_error = str(exc)
            continue
        if count <= 0:
            continue
        try:
            trigger = raw.nth(max(0, count - 1))
            _reveal_hidden_upload_trigger(trigger)
            chooser_errors: List[str] = []
            for clicker in (
                lambda: trigger.click(timeout=2500, force=True, no_wait_after=True),
                lambda: trigger.click(timeout=2500, no_wait_after=True),
                lambda: trigger.dispatch_event("click"),
                lambda: trigger.evaluate(
                    "el => { el.click(); el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window })); }"
                ),
            ):
                try:
                    with page.expect_file_chooser(timeout=3500) as chooser_info:
                        clicker()
                    chooser_info.value.set_files(str(file_path), timeout=set_files_timeout_ms)
                    return True, "hidden_file_chooser"
                except Exception as inner_exc:
                    try:
                        if _handle_gemini_consent_if_present(page, timeout_ms=4000):
                            chooser_errors.append(f"{inner_exc}; upload consent handled")
                            continue
                    except Exception:
                        pass
                    chooser_errors.append(str(inner_exc))
                    continue
            last_error = "; ".join([err for err in chooser_errors if err]) or last_error
        except Exception as exc:
            last_error = str(exc)
            try:
                trigger.set_input_files(str(file_path))
                return True, "hidden_file_input"
            except Exception as inner_exc:
                last_error = str(inner_exc or exc)
                continue
    return False, last_error or "hidden local file trigger not found"
def _build_attachment_expected_fragments(*, attachment_path: Path, episode_id: str) -> List[str]:
    effective_episode_id = str(episode_id or "").strip().lower() or _infer_episode_id_from_paths(attachment_path)
    fragments: List[str] = []
    if effective_episode_id:
        fragments.append(effective_episode_id)
    fragments.extend(
        [
            attachment_path.stem.lower(),
            attachment_path.name.lower(),
        ]
    )
    normalized: List[str] = []
    for item in fragments:
        text = str(item or "").strip().lower()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _attachment_tokens_match_expected(tokens: Sequence[str], expected_fragments: Sequence[str]) -> bool:
    token_text = " || ".join(str(item or "").strip().lower() for item in (tokens or []) if str(item or "").strip())
    if not token_text:
        return False
    candidates = [
        str(frag or "").strip().lower()
        for frag in (expected_fragments or [])
        if len(str(frag or "").strip()) >= 12
    ]
    return any(candidate in token_text for candidate in candidates)


def _collect_attachment_tokens(
    page: Any,
    *,
    composer_locator: Optional[Any],
    local_only: bool,
) -> List[str]:
    composer_top = None
    composer_bottom = None
    composer_box = _safe_bounding_box(composer_locator) if composer_locator is not None else None
    if composer_box:
        composer_top = float(composer_box.get("y", 0.0) or 0.0)
        composer_bottom = composer_top + float(composer_box.get("height", 0.0) or 0.0)

    tokens: List[str] = []
    selectors = [
        ".new-file-name",
        ".new-file-type",
        '[aria-label*=".mp4" i]',
        '[aria-label*="google drive" i]',
        '[aria-label*="video_" i]',
        '[title*=".mp4" i]',
        '[title*="google drive" i]',
        '[title*="video_" i]',
        '[data-testid*="attachment" i]',
        '[data-testid*="file" i]',
    ]

    def _looks_like_attachment_text(text: str) -> bool:
        lower = str(text or "").strip().lower()
        if not lower:
            return False
        return any(
            needle in lower
            for needle in (
                ".mp4",
                "google drive",
                "video_",
                "remove file",
                "remove attachment",
                "attached file",
            )
        )

    for sel in selectors:
        try:
            loc = page.locator(sel)
            count = int(loc.count() or 0)
        except Exception:
            continue
        for idx in range(count):
            try:
                node = loc.nth(idx)
                if not node.is_visible():
                    continue
                box = _safe_bounding_box(node)
                if local_only and composer_top is not None and composer_bottom is not None and box is not None:
                    y = float(box.get("y", 0.0) or 0.0)
                    if y < composer_top - 360.0 or y > composer_bottom + 260.0:
                        continue
                values = [
                    str(node.inner_text(timeout=300) or "").strip(),
                    str(node.text_content(timeout=300) or "").strip(),
                    str(node.get_attribute("aria-label", timeout=300) or "").strip(),
                    str(node.get_attribute("title", timeout=300) or "").strip(),
                    str(node.get_attribute("data-testid", timeout=300) or "").strip(),
                ]
                for value in values:
                    if _looks_like_attachment_text(value):
                        tokens.append(f"{sel}:{value}")
            except Exception:
                continue
    try:
        dom_tokens = page.evaluate(
            """
            ({ localOnly, composerTop, composerBottom }) => {
              const out = [];
              const nodes = Array.from(document.querySelectorAll('[aria-label],[title],[data-testid],[role],[class]'));
              for (const el of nodes) {
                const rect = typeof el.getBoundingClientRect === 'function' ? el.getBoundingClientRect() : null;
                if (!rect || rect.width <= 0 || rect.height <= 0) continue;
                if (localOnly && composerTop !== null && composerBottom !== null) {
                  if (rect.top < composerTop - 360 || rect.top > composerBottom + 260) continue;
                }
                const pieces = [
                  el.getAttribute('aria-label') || '',
                  el.getAttribute('title') || '',
                  el.getAttribute('data-testid') || '',
                  (el.textContent || '').trim(),
                ].filter(Boolean);
                if (!pieces.length) continue;
                const joined = pieces.join(' | ');
                const lower = joined.toLowerCase();
                if (
                  lower.includes('.mp4') ||
                  lower.includes('google drive') ||
                  lower.includes('video_') ||
                  lower.includes('remove attachment') ||
                  lower.includes('remove file')
                ) {
                  out.push(joined.slice(0, 240));
                }
              }
              return out.slice(0, 40);
            }
            """,
            {
                "localOnly": bool(local_only),
                "composerTop": composer_top,
                "composerBottom": composer_bottom,
            },
        )
        if isinstance(dom_tokens, list):
            for value in dom_tokens:
                text = str(value or "").strip()
                if _looks_like_attachment_text(text):
                    tokens.append(f"dom:{text}")
    except Exception:
        pass
    return sorted(set(tokens))


def _network_upload_confirmation_ok(
    *,
    attach_mode: str,
    request_log: List[str],
    response_log: List[str],
) -> bool:
    mode = str(attach_mode or "").strip().lower()
    if mode not in {"file_input", "file_chooser"}:
        return False
    request_count = len([row for row in request_log if str(row or "").strip()])
    success_count = 0
    for row in response_log:
        text = str(row or "").strip()
        match = re.match(r"^(\d{3})\b", text)
        if not match:
            continue
        try:
            status = int(match.group(1))
        except Exception:
            continue
        if 200 <= status < 300:
            success_count += 1
    return request_count >= 2 and success_count >= 1


def _wait_for_drive_picker(page: Any, timeout_sec: float = 20.0) -> Optional[Any]:
    deadline = time.time() + max(2.0, float(timeout_sec))
    while time.time() < deadline:
        for fr in page.frames:
            try:
                url = str(fr.url or "")
            except Exception:
                url = ""
            if "docs.google.com/picker" in url:
                return fr
        try:
            page.wait_for_timeout(500)
        except Exception:
            time.sleep(0.5)
    return None


def _picker_search_input(picker: Any) -> Optional[Any]:
    candidates = [
        lambda: picker.get_by_placeholder("Search in Drive or paste URL").first,
        lambda: picker.locator('input[aria-label*="Search in Drive" i]').first,
        lambda: picker.locator('input[placeholder*="Search in Drive" i]').first,
    ]
    for factory in candidates:
        try:
            loc = factory()
            if int(loc.count() or 0) <= 0:
                continue
            loc.wait_for(state="visible", timeout=1500)
            return loc
        except Exception:
            continue
    return None


def _picker_visible_option(picker: Any, text: str) -> Optional[Any]:
    if not text:
        return None
    candidates = [
        lambda: picker.get_by_role("option", name=text).first,
        lambda: picker.get_by_text(text, exact=True).first,
        lambda: picker.locator(f"text={text}").first,
    ]
    for factory in candidates:
        try:
            loc = factory()
            if int(loc.count() or 0) <= 0:
                continue
            loc.wait_for(state="visible", timeout=1500)
            return loc
        except Exception:
            continue
    return None


def _try_attach_via_drive_picker(
    *,
    page: Any,
    attach_trigger: Optional[Any],
    episode_id: str,
    drive_root_folder_url: str,
) -> Tuple[bool, str]:
    eid = str(episode_id or "").strip().lower()
    if not eid:
        return False, "missing episode_id"

    if attach_trigger is None:
        attach_trigger = _first_exact_upload_trigger(page)
    if attach_trigger is None:
        return False, "drive picker trigger not found"

    try:
        _robust_click(attach_trigger)
        page.wait_for_timeout(700)
    except Exception as exc:
        picker = _wait_for_drive_picker(page, timeout_sec=2.0)
        if picker is None:
            return False, f"open upload menu failed: {exc}"

    drive_item = _first_exact_drive_item(page)
    if drive_item is None:
        picker = _wait_for_drive_picker(page, timeout_sec=2.0)
        if picker is None:
            return False, "Add from Drive item not found"
    else:
        try:
            _robust_click(drive_item)
        except Exception as exc:
            picker = _wait_for_drive_picker(page, timeout_sec=3.0)
            if picker is None:
                return False, f"Add from Drive click failed: {exc}"

    picker = _wait_for_drive_picker(page, timeout_sec=25.0)
    if picker is None:
        return False, "Drive picker frame not found"

    search = _picker_search_input(picker)
    if search is None:
        return False, "Drive picker search input not found"

    video_name = f"video_{eid}.mp4"
    search_queries = [video_name, eid]
    root_url = str(drive_root_folder_url or "").strip()
    if root_url:
        search_queries.append(root_url)

    direct_video_locator = None
    for query in search_queries:
        try:
            search.fill("")
            search.fill(query, timeout=8000)
            search.press("Enter")
            page.wait_for_timeout(4000)
        except Exception:
            continue
        try:
            direct_video_locator = _picker_visible_option(picker, video_name)
            if direct_video_locator is None or int(direct_video_locator.count() or 0) <= 0:
                alt = picker.locator(f'text=/{re.escape(video_name)}/i').first
                if int(alt.count() or 0) > 0:
                    alt.wait_for(state="visible", timeout=1500)
                    direct_video_locator = alt
        except Exception:
            direct_video_locator = None
        if direct_video_locator is not None:
            break

    if direct_video_locator is not None:
        try:
            direct_video_locator.click(timeout=5000)
            page.wait_for_timeout(800)
        except Exception as exc:
            return False, f"episode video direct click failed: {exc}"
        insert_btn = None
        candidates = [
            lambda: picker.get_by_role("button", name="Insert").first,
            lambda: picker.get_by_text("Insert", exact=True).first,
            lambda: picker.locator('text=/^Insert$/i').first,
        ]
        for factory in candidates:
            try:
                loc = factory()
                if int(loc.count() or 0) <= 0:
                    continue
                loc.wait_for(state="visible", timeout=1500)
                insert_btn = loc
                break
            except Exception:
                continue
        if insert_btn is None:
            return False, "Drive picker Insert button not found after direct video search"
        try:
            _robust_click(insert_btn)
            return True, "drive_picker"
        except Exception as exc:
            return False, f"Drive picker Insert failed after direct video search: {exc}"

    folder_locator = None
    try:
        search.fill("")
        search.press("Enter")
        page.wait_for_timeout(1800)
    except Exception:
        pass
    for query in search_queries:
        try:
            search.fill("")
            search.fill(query, timeout=8000)
            search.press("Enter")
            page.wait_for_timeout(4000)
        except Exception:
            continue
        folder_locator = _picker_visible_option(picker, eid)
        if folder_locator is not None:
            break

    if folder_locator is None:
        try:
            search.fill("")
            search.press("Enter")
            page.wait_for_timeout(1800)
        except Exception:
            pass
        episodes_folder = _picker_visible_option(picker, "episodes")
        if episodes_folder is not None:
            try:
                episodes_folder.dblclick(timeout=5000)
            except Exception:
                try:
                    episodes_folder.click(timeout=5000)
                    page.wait_for_timeout(400)
                    episodes_folder.click(click_count=2, timeout=5000)
                except Exception as exc:
                    return False, f"episodes folder open failed: {exc}"
            page.wait_for_timeout(3000)
            try:
                search.fill("")
                search.press("Enter")
                page.wait_for_timeout(1800)
            except Exception:
                pass
            try:
                folder_locator = _picker_visible_option(picker, eid)
            except Exception:
                folder_locator = None
            for query in [eid, video_name]:
                try:
                    search.fill("")
                    search.fill(query, timeout=8000)
                    search.press("Enter")
                    page.wait_for_timeout(3000)
                except Exception:
                    continue
                folder_locator = _picker_visible_option(picker, eid)
                if folder_locator is not None:
                    break

    if folder_locator is None:
        return False, f"episode folder not found in Drive picker: {eid}"

    try:
        folder_locator.dblclick(timeout=5000)
    except Exception:
        try:
            folder_locator.click(timeout=5000)
            page.wait_for_timeout(400)
            folder_locator.click(click_count=2, timeout=5000)
        except Exception as exc:
            return False, f"episode folder open failed: {exc}"
    page.wait_for_timeout(4000)
    try:
        search.fill("")
        search.press("Enter")
        page.wait_for_timeout(1800)
    except Exception:
        pass

    video_locator = None
    candidates = [
        lambda: _picker_visible_option(picker, video_name),
        lambda: picker.locator('text=/^video_.*\\.mp4$/i').first,
    ]
    for factory in candidates:
        try:
            loc = factory()
            if loc is None:
                continue
            if int(loc.count() or 0) <= 0:
                continue
            loc.wait_for(state="visible", timeout=2000)
            video_locator = loc
            break
        except Exception:
            continue
    if video_locator is None:
        return False, f"episode video not visible in Drive picker: {video_name}"

    try:
        video_locator.click(timeout=5000)
        page.wait_for_timeout(800)
    except Exception as exc:
        return False, f"episode video click failed: {exc}"

    insert_btn = None
    candidates = [
        lambda: picker.get_by_role("button", name="Insert").first,
        lambda: picker.get_by_text("Insert", exact=True).first,
        lambda: picker.locator('text=/^Insert$/i').first,
    ]
    for factory in candidates:
        try:
            loc = factory()
            if int(loc.count() or 0) <= 0:
                continue
            loc.wait_for(state="visible", timeout=1500)
            insert_btn = loc
            break
        except Exception:
            continue
    if insert_btn is None:
        return False, "Drive picker Insert button not found"

    try:
        _robust_click(insert_btn)
        return True, "drive_picker"
    except Exception as exc:
        return False, f"Drive picker Insert failed: {exc}"


def _attach_files_via_chat_ui(
    *,
    page: Any,
    composer_locator: Optional[Any],
    attach_candidates: List[Path],
    episode_id: str,
    prefer_drive_picker: bool,
    drive_root_folder_url: str,
    max_upload_mb: float,
    attach_button_sel: str,
    upload_menu_sel: str,
    file_input_sel: str,
    upload_settle_min_sec: float,
    upload_settle_sec_per_100mb: float,
    upload_settle_max_sec: float,
    heartbeat: Optional[Callable[[], None]] = None,
    heartbeat_interval_sec: float = 10.0,
    progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_interval_sec: float = 15.0,
) -> List[str]:
    attach_notes: List[str] = []
    if not attach_candidates:
        return attach_notes

    last_heartbeat_ts = time.monotonic()

    def _maybe_emit_heartbeat() -> None:
        nonlocal last_heartbeat_ts
        interval_sec = max(2.0, float(heartbeat_interval_sec or 10.0))
        if heartbeat is None:
            return
        now = time.monotonic()
        if (now - last_heartbeat_ts) < interval_sec:
            return
        try:
            heartbeat()
        except Exception:
            pass
        last_heartbeat_ts = now

    for vid in attach_candidates:
        _maybe_emit_heartbeat()
        _prepare_chat_composer_for_attach(page, composer_locator)
        effective_episode_id = str(episode_id or "").strip().lower() or _infer_episode_id_from_paths(vid)
        expected_fragments = _build_attachment_expected_fragments(
            attachment_path=vid,
            episode_id=effective_episode_id,
        )
        baseline_tokens = _collect_attachment_tokens(page, composer_locator=composer_locator, local_only=True)
        baseline_page_tokens = _collect_attachment_tokens(page, composer_locator=composer_locator, local_only=False)
        request_log: List[str] = []
        response_log: List[str] = []

        def _track_request(req: Any) -> None:
            try:
                url = str(req.url or "")
            except Exception:
                return
            url_l = url.lower()
            if "_/bardchatui/data/batchexecute" in url_l or "upload" in url_l or "attachment" in url_l:
                request_log.append(f"{str(req.method or '').upper()} {url}")

        def _track_response(resp: Any) -> None:
            try:
                url = str(resp.url or "")
                status = int(resp.status)
            except Exception:
                return
            url_l = url.lower()
            if "_/bardchatui/data/batchexecute" in url_l or "upload" in url_l or "attachment" in url_l:
                response_log.append(f"{status} {str(resp.request.method or '').upper()} {url}")

        try:
            page.on("request", _track_request)
            page.on("response", _track_response)
        except Exception:
            pass

        size_mb = float(vid.stat().st_size) / (1024 * 1024)
        print(
            "[trace] gemini attach start: "
            f"file={vid.name} size_mb={size_mb:.1f} "
            f"prefer_drive_picker={bool(prefer_drive_picker)}"
        , flush=True)
        _emit_progress_hook(
            progress_hook,
            {
                "phase": "attach_start",
                "attachment_name": vid.name,
                "size_mb": round(size_mb, 1),
                "prefer_drive_picker": bool(prefer_drive_picker),
            },
        )
        if size_mb > max_upload_mb:
            attach_notes.append(f"{vid.name}: skipped ({size_mb:.1f} MB > {max_upload_mb:.1f} MB)")
            try:
                page.remove_listener("request", _track_request)
                page.remove_listener("response", _track_response)
            except Exception:
                pass
            continue

        attached = False
        attach_mode = ""
        last_error = ""
        try:
            page.wait_for_timeout(600)
        except Exception:
            pass
        _maybe_emit_heartbeat()

        attach_trigger, file_input = _wait_for_upload_ready(
            page,
            attach_button_sel=attach_button_sel,
            file_input_sel=file_input_sel,
            timeout_ms=12000,
            heartbeat=_maybe_emit_heartbeat,
            heartbeat_interval_sec=heartbeat_interval_sec,
        )
        if attach_trigger is None and file_input is None:
            _prepare_chat_composer_for_attach(page, composer_locator)
            attach_trigger, file_input = _wait_for_upload_ready(
                page,
                attach_button_sel=attach_button_sel,
                file_input_sel=file_input_sel,
                timeout_ms=8000,
                heartbeat=_maybe_emit_heartbeat,
                heartbeat_interval_sec=heartbeat_interval_sec,
            )
        if prefer_drive_picker and effective_episode_id:
            attached, attach_mode = _try_attach_via_drive_picker(
                page=page,
                attach_trigger=attach_trigger,
                episode_id=effective_episode_id,
                drive_root_folder_url=drive_root_folder_url,
            )
            if not attached:
                last_error = attach_mode

        if not attached and file_input is not None:
            attached, attach_mode = _try_set_files_on_locator(file_input, vid)
            if not attached:
                last_error = attach_mode

        if not attached:
            if attach_trigger is not None:
                attached, attach_mode = _try_attach_via_file_chooser(page, attach_trigger, vid)
                if attached:
                    last_error = ""
                else:
                    last_error = attach_mode or last_error

                if not attached:
                    upload_item = None
                    try:
                        _robust_click(attach_trigger)
                        page.wait_for_timeout(600)
                        if _handle_gemini_consent_if_present(page, timeout_ms=5000):
                            page.wait_for_timeout(500)
                            attach_trigger = _first_exact_upload_trigger(page) or _first_visible_locator(
                                page, attach_button_sel, timeout_ms=3000
                            )
                            if attach_trigger is not None:
                                _robust_click(attach_trigger)
                                page.wait_for_timeout(600)
                        upload_item = _first_exact_upload_item(page) or _first_visible_locator(
                            page, upload_menu_sel, timeout_ms=5000
                        )
                    except Exception as exc:
                        last_error = str(exc)

                    if upload_item is not None:
                        attached, attach_mode = _try_attach_via_file_chooser(page, upload_item, vid)
                        if not attached:
                            last_error = attach_mode or last_error
                            try:
                                _robust_click(upload_item)
                                page.wait_for_timeout(900)
                            except Exception as exc:
                                last_error = str(exc)

                if not attached:
                    try:
                        page.wait_for_timeout(900)
                    except Exception:
                        pass
                    _maybe_emit_heartbeat()
                    file_input = _last_present_locator(page, file_input_sel)
                    if file_input is not None:
                        attached, attach_mode = _try_set_files_on_locator(file_input, vid)
                        if not attached:
                            last_error = attach_mode

        if not attached:
            attached, attach_mode = _try_hidden_local_file_trigger(page, vid)
            if not attached:
                last_error = attach_mode

        if attached:
            settle_info = _wait_for_chat_upload_settle(
                page,
                composer_locator=composer_locator,
                baseline_tokens=baseline_tokens,
                baseline_page_tokens=baseline_page_tokens,
                expected_fragments=expected_fragments,
                require_google_drive_video=(attach_mode == "drive_picker"),
                size_mb=size_mb,
                min_wait_sec=upload_settle_min_sec,
                sec_per_100mb=upload_settle_sec_per_100mb,
                max_wait_sec=upload_settle_max_sec,
                heartbeat=heartbeat,
                heartbeat_interval_sec=heartbeat_interval_sec,
                attachment_name=vid.name,
                progress_hook=progress_hook,
                progress_interval_sec=progress_interval_sec,
            )
            if bool(settle_info.get("confirmed", False)):
                token_text = ",".join(settle_info.get("tokens", [])[:2])
                attach_notes.append(
                    f"{vid.name}: attached ({size_mb:.1f} MB, settle={float(settle_info.get('wait_sec', 0.0) or 0.0):.1f}s, "
                    f"mode={attach_mode}, reqs={len(request_log)}, resps={len(response_log)}, tokens={token_text})"
                )
            else:
                if _network_upload_confirmation_ok(
                    attach_mode=attach_mode,
                    request_log=request_log,
                    response_log=response_log,
                ):
                    token_text = ",".join(settle_info.get("tokens", [])[:2]) or "network-confirmed"
                    attach_notes.append(
                        f"{vid.name}: attached ({size_mb:.1f} MB, settle={float(settle_info.get('wait_sec', 0.0) or 0.0):.1f}s, "
                        f"mode={attach_mode}, reqs={len(request_log)}, resps={len(response_log)}, tokens={token_text}, "
                        f"confirmation=network)"
                    )
                else:
                    attached = False
                    last_error = (
                        f"attachment chip not confirmed near composer "
                        f"(reqs={len(request_log)}, resps={len(response_log)})"
                    )

        if not attached:
            reason = last_error or "file input not found"
            attach_notes.append(f"{vid.name}: skipped ({reason})")

        print(
            "[trace] gemini attach result: "
            f"file={vid.name} attached={bool(attached)} mode={attach_mode or 'none'} "
            f"detail={attach_notes[-1] if attach_notes else last_error or 'n/a'}"
        , flush=True)
        _emit_progress_hook(
            progress_hook,
            {
                "phase": "attach_done",
                "attachment_name": vid.name,
                "size_mb": round(size_mb, 1),
                "attached": bool(attached),
                "mode": str(attach_mode or "none"),
                "detail": str(attach_notes[-1] if attach_notes else last_error or "n/a"),
            },
        )
        try:
            page.remove_listener("request", _track_request)
            page.remove_listener("response", _track_response)
        except Exception:
            pass

    return attach_notes


def _fill_chat_input(input_box: Any, text: str, page: Any) -> None:
    payload = str(text or "")
    if not payload:
        return

    def _keyboard_insert_payload() -> None:
        insert_text = getattr(page.keyboard, "insert_text", None)
        if callable(insert_text):
            insert_text(payload)
            return
        type_text = getattr(page.keyboard, "type", None)
        if callable(type_text):
            type_text(payload)
            return
        raise RuntimeError("keyboard text insertion unavailable")

    # ---- Strategy 1: click to focus, select-all, then CDP insert_text ----
    # Playwright's keyboard.insert_text() sends a single CDP Input.insertText
    # command which is fast even for multi-KB payloads and properly triggers
    # Angular/React/ProseMirror editing pipelines.  This is the most reliable
    # approach for Gemini's contenteditable composer.
    try:
        is_contenteditable = bool(
            input_box.evaluate(
                """(el) => {
                    if (!el) return false;
                    return el.isContentEditable ||
                           String(el.getAttribute('contenteditable') || '').toLowerCase() === 'true';
                }"""
            )
        )
    except Exception:
        is_contenteditable = False

    if is_contenteditable:
        try:
            input_box.click(timeout=2000, force=True)
            # Move cursor to end WITHOUT selecting all – Ctrl+A would also
            # select (and then delete) file attachment chips in the composer.
            page.keyboard.press("End")
            insert_text = getattr(page.keyboard, "insert_text", None)
            if callable(insert_text):
                insert_text(payload)
                # Nudge the framework: dispatch an InputEvent on the element so
                # Angular/React pick up the content from CDP insert_text.
                try:
                    input_box.evaluate(
                        """(el) => {
                            try {
                                el.dispatchEvent(new InputEvent('input', {
                                    bubbles: true,
                                    inputType: 'insertText',
                                    data: ' ',
                                }));
                            } catch (e) {}
                            try { el.dispatchEvent(new Event('change', {bubbles: true})); } catch (e) {}
                        }"""
                    )
                except Exception:
                    pass
                print(f"[trace] _fill_chat_input: used CDP insert_text ({len(payload)} chars)", flush=True)
                return
        except Exception as exc:
            print(f"[trace] _fill_chat_input: CDP insert_text failed: {exc}", flush=True)

    # ---- Strategy 2: direct DOM assignment for textarea/input elements ----
    try:
        assigned = bool(
            input_box.evaluate(
                """(el, value) => {
                    if (!el) return false;
                    const tag = String(el.tagName || '').toLowerCase();
                    const isTextControl = tag === 'textarea' || tag === 'input';
                    if (!isTextControl) return false;
                    try { el.focus(); } catch (e) {}
                    el.value = value;
                    el.dispatchEvent(new Event('input', {bubbles: true}));
                    el.dispatchEvent(new Event('change', {bubbles: true}));
                    return true;
                }""",
                payload,
            )
        )
        if assigned:
            print(f"[trace] _fill_chat_input: used DOM value assignment ({len(payload)} chars)", flush=True)
            return
    except Exception:
        pass

    # ---- Strategy 3: Playwright .fill() ----
    try:
        try:
            input_box.fill(payload, timeout=1500)
        except TypeError:
            input_box.fill(payload)
        print(f"[trace] _fill_chat_input: used Playwright fill ({len(payload)} chars)", flush=True)
        return
    except Exception:
        pass

    # ---- Strategy 4: click + keyboard insert_text (non-contenteditable) ----
    try:
        input_box.click(timeout=1000, force=True)
        page.keyboard.press("Control+A")
        _keyboard_insert_payload()
        print(f"[trace] _fill_chat_input: used keyboard insert fallback ({len(payload)} chars)", flush=True)
        return
    except Exception:
        pass
    input_box.click(timeout=1000, force=True)
    _keyboard_insert_payload()
    print(f"[trace] _fill_chat_input: used final keyboard fallback ({len(payload)} chars)", flush=True)


def _is_dedicated_gemini_chat_url(url: str) -> bool:
    normalized = str(url or "").strip().rstrip("/")
    return "gemini.google.com/app/" in normalized


def _is_authenticated_gemini_chat_page(page: Any) -> bool:
    try:
        current_url = str(getattr(page, "url", "") or "").strip().lower()
    except Exception:
        current_url = ""
    try:
        title = str(page.title() or "").strip().lower()
    except Exception:
        title = ""
    try:
        body = str(page.locator("body").inner_text(timeout=1500) or "").strip().lower()
    except Exception:
        body = ""
    if "accounts.google.com" in current_url:
        return False
    if "sign in" in title and "gemini" in title:
        return False
    if any(
        marker in body
        for marker in (
            "sign in",
            "meet gemini, your personal ai assistant",
            "get access to all gemini models",
        )
    ):
        return False
    if any(marker in body for marker in ("ask gemini", "upgrade", "plus", "hi ayman")):
        return True
    return "gemini.google.com/app" in current_url and "sign in" not in body


def _pick_existing_gemini_chat_page(context: Any, *, chat_url: str) -> Optional[Any]:
    target = str(chat_url or "").strip()
    fallback = None
    authenticated_fallback = None
    try:
        for candidate in reversed(list(getattr(context, "pages", []) or [])):
            current_url = str(getattr(candidate, "url", "") or "").strip()
            authenticated = _is_authenticated_gemini_chat_page(candidate)
            if target and current_url.startswith(target) and authenticated:
                return candidate
            if target and current_url.startswith(target) and fallback is None:
                fallback = candidate
            if "gemini.google.com" in current_url and authenticated and authenticated_fallback is None:
                authenticated_fallback = candidate
            if fallback is None and "gemini.google.com" in current_url and not _is_dedicated_gemini_chat_url(target):
                fallback = candidate
    except Exception:
        return None
    target_is_dedicated = _is_dedicated_gemini_chat_url(target)
    if target_is_dedicated:
        return fallback
    if authenticated_fallback is not None:
        return authenticated_fallback
    return fallback


def _extract_recent_chat_response_entries(page: Any, *, limit: int = 8) -> List[Dict[str, Any]]:
    selectors = [
        "message-content",
        "model-response",
        "div.markdown",
    ]
    max_items = max(1, int(limit or 1))
    for sel in selectors:
        try:
            loc = page.locator(sel)
        except Exception:
            continue
        try:
            total_count = int(loc.count() or 0)
        except Exception:
            total_count = 0
        if total_count <= 0:
            continue
        values: List[Dict[str, Any]] = []
        start_idx = max(0, total_count - max_items)
        for idx in range(start_idx, total_count):
            txt = ""
            try:
                txt = str(loc.nth(idx).inner_text(timeout=500) or "").strip()
            except Exception:
                txt = ""
            values.append({"message_index": idx, "text": txt})
        if values:
            return values[-max_items:]
    try:
        values = page.evaluate(
            """() => {
                    const selectors = ['message-content', 'model-response', 'div.markdown'];
                    const maxItems = MAX_ITEMS;
                    for (const selector of selectors) {
                        const nodes = Array.from(document.querySelectorAll(selector));
                        const values = [];
                        const start = Math.max(0, nodes.length - maxItems);
                        for (let i = start; i < nodes.length; i += 1) {
                            const value = String(
                                nodes[i]?.innerText || nodes[i]?.textContent || ''
                            ).trim();
                            values.push({ message_index: i, text: value });
                        }
                        if (values.length) return values.slice(-maxItems);
                    }
                    return [];
                }""".replace("MAX_ITEMS", str(max_items))
        )
    except Exception:
        values = []
    if isinstance(values, list):
        clean_values: List[Dict[str, Any]] = []
        for item in values[-max_items:]:
            if not isinstance(item, dict):
                continue
            clean_values.append(
                {
                    "message_index": int(item.get("message_index", 0) or 0),
                    "text": str(item.get("text", "") or "").strip(),
                }
            )
        if clean_values:
            return clean_values
    return []


def _hash_chat_response_entries(entries: Sequence[Dict[str, Any]], *, message_count: int = 0) -> str:
    digest = hashlib.sha1()
    digest.update(str(int(message_count or 0)).encode("utf-8", errors="ignore"))
    for item in entries or []:
        digest.update(b"|")
        digest.update(str(int(item.get("message_index", 0) or 0)).encode("utf-8", errors="ignore"))
        digest.update(b":")
        digest.update(str(item.get("text", "") or "").encode("utf-8", errors="ignore"))
    return digest.hexdigest()[:16]


def _capture_chat_response_state(page: Any, *, limit: int = 8) -> Dict[str, Any]:
    entries = _extract_recent_chat_response_entries(page, limit=limit)
    message_count = 0
    if entries:
        try:
            message_count = max(int(item.get("message_index", 0) or 0) for item in entries) + 1
        except Exception:
            message_count = len(entries)
    texts = [
        str(item.get("text", "") or "").strip()
        for item in entries
        if str(item.get("text", "") or "").strip()
    ]
    return {
        "message_count": int(message_count or 0),
        "response_hash": _hash_chat_response_entries(entries, message_count=message_count),
        "entries": list(entries),
        "texts": texts[-max(1, int(limit or 1)):],
        "latest_text": texts[-1] if texts else "",
    }


def _extract_recent_chat_response_texts(page: Any, *, limit: int = 8) -> List[str]:
    state = _capture_chat_response_state(page, limit=limit)
    texts = state.get("texts", [])
    if isinstance(texts, list):
        return [str(item or "").strip() for item in texts if str(item or "").strip()]
    return []


def _new_chat_response_candidates_after_baseline(
    current_state: Optional[Dict[str, Any]],
    *,
    baseline_state: Optional[Dict[str, Any]] = None,
    baseline_candidates: Optional[Sequence[str]] = None,
) -> List[str]:
    state = current_state if isinstance(current_state, dict) else {}
    baseline = baseline_state if isinstance(baseline_state, dict) else {}
    baseline_history = [
        str(item or "").strip()
        for item in (baseline_candidates or [])
        if str(item or "").strip()
    ]
    baseline_message_count = max(0, int(baseline.get("message_count", 0) or 0))
    current_entries = state.get("entries", []) if isinstance(state.get("entries"), list) else []
    new_entries = [
        str(item.get("text", "") or "").strip()
        for item in current_entries
        if int(item.get("message_index", 0) or 0) >= baseline_message_count
        and str(item.get("text", "") or "").strip()
    ]
    if new_entries:
        return new_entries
    current_candidates = state.get("texts", []) if isinstance(state.get("texts"), list) else []
    if not current_candidates:
        return []
    fallback = _subtract_baseline_response_candidates(current_candidates, baseline_history)
    if not fallback:
        return []
    current_hash = str(state.get("response_hash", "") or "").strip()
    baseline_hash = str(baseline.get("response_hash", "") or "").strip()
    current_message_count = max(0, int(state.get("message_count", 0) or 0))
    if current_message_count > baseline_message_count:
        return fallback
    if current_hash and baseline_hash and current_hash != baseline_hash:
        return fallback
    return []


def _extract_latest_chat_response_text(page: Any) -> str:
    recent = _extract_recent_chat_response_texts(page, limit=1)
    if recent:
        return str(recent[-1] or "").strip()
    return ""


def _parse_chat_response_candidate(text: str) -> Optional[Any]:
    candidate = _clean_json_text(text)
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def _select_preferred_chat_response_candidate(
    candidates: Sequence[str],
    *,
    preferred_top_level_key: str = "",
    require_parseable_json: bool = False,
) -> str:
    preferred = str(preferred_top_level_key or "").strip()
    latest_parseable = ""
    for raw in reversed(list(candidates or [])):
        txt = str(raw or "").strip()
        if not txt:
            continue
        parsed = _parse_chat_response_candidate(txt)
        if isinstance(parsed, dict) and preferred and preferred in parsed:
            return txt
        if parsed is not None and not latest_parseable:
            latest_parseable = txt
        if not require_parseable_json and not preferred:
            return txt
    if latest_parseable:
        return latest_parseable
    if require_parseable_json:
        return ""
    for raw in reversed(list(candidates or [])):
        txt = str(raw or "").strip()
        if txt:
            return txt
    return ""


def _subtract_baseline_response_candidates(
    current_candidates: Sequence[str],
    baseline_candidates: Sequence[str],
) -> List[str]:
    baseline_counts: Counter[str] = Counter(
        str(item or "").strip()
        for item in (baseline_candidates or [])
        if str(item or "").strip()
    )
    running_counts: Counter[str] = Counter()
    out: List[str] = []
    for raw in current_candidates or []:
        txt = str(raw or "").strip()
        if not txt:
            continue
        running_counts[txt] += 1
        if running_counts[txt] > baseline_counts[txt]:
            out.append(txt)
    return out


def _emit_progress_hook(progress_hook: Optional[Callable[[Dict[str, Any]], None]], payload: Dict[str, Any]) -> None:
    if progress_hook is None:
        return
    try:
        progress_hook(dict(payload))
    except Exception:
        pass


def _quick_locator_visible(page: Any, selector_expr: str) -> bool:
    for sel in _selector_variants(selector_expr):
        try:
            loc = page.locator(sel)
            count = min(int(loc.count() or 0), 2)
        except Exception:
            continue
        for idx in range(count):
            try:
                if loc.nth(idx).is_visible(timeout=250):
                    return True
            except Exception:
                continue
    return False


def _build_wait_progress_payload(
    page: Any,
    *,
    phase: str,
    elapsed_sec: float,
    remaining_sec: float,
    latest_text: str,
    baseline_text: str,
    stable_count: int,
    last_change_age_sec: float,
    current_message_count: int = 0,
    baseline_message_count: int = 0,
    response_hash_changed: bool = False,
) -> Dict[str, Any]:
    body_text = ""
    try:
        body_text = str(page.locator("body").inner_text(timeout=400) or "")
    except Exception:
        body_text = ""
    body_lower = body_text.lower()
    latest_clean = _clean_json_text(latest_text) if latest_text else ""
    parseable_json = False
    if latest_clean:
        try:
            parseable_json = isinstance(json.loads(latest_clean), (dict, list))
        except Exception:
            parseable_json = False
    preview = re.sub(r"\s+", " ", str(latest_text or "").strip())
    if len(preview) > 120:
        preview = preview[:117] + "..."
    return {
        "phase": phase,
        "elapsed_sec": round(max(0.0, float(elapsed_sec or 0.0)), 1),
        "remaining_sec": round(max(0.0, float(remaining_sec or 0.0)), 1),
        "baseline_changed": bool(latest_text and latest_text != baseline_text),
        "response_chars": len(str(latest_text or "")),
        "stable_count": int(stable_count or 0),
        "seconds_since_change": round(max(0.0, float(last_change_age_sec or 0.0)), 1),
        "thinking": (
            "thinking" in body_lower
            or "analyzing" in body_lower
            or "reasoning" in body_lower
            or "responding" in body_lower
        ),
        "quota_banner": (
            "reached your pro model limit" in body_lower
            or "reached your gemini advanced limit" in body_lower
        ),
        "error_banner": _is_retryable_chat_error_text(body_text),
        "stop_visible": _quick_locator_visible(
            page,
            'button[aria-label*="Stop" i] || button:has-text("Stop responding") || button:has-text("Stop")',
        ),
        "send_visible": _quick_locator_visible(
            page,
            'button[aria-label*="Send" i] || button:has-text("Send") || button:has-text("Run")',
        ),
        "parseable_json": parseable_json,
        "current_message_count": int(current_message_count or 0),
        "baseline_message_count": int(baseline_message_count or 0),
        "response_hash_changed": bool(response_hash_changed),
        "preview": preview,
    }


def _is_retryable_chat_error_text(text: str) -> bool:
    preview = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not preview:
        return False
    return any(
        needle in preview
        for needle in (
            "something went wrong",
            "unable to complete",
            "could you try again",
            "try your request again",
            "please try again",
            "i encountered an error doing what you asked",
            "try again later",
        )
    )
def _find_visible_button_with_labels(
    page: Any,
    labels: Sequence[str],
    *,
    near_bottom: bool = False,
) -> Optional[Any]:
    normalized_labels = {str(item or "").strip().lower() for item in (labels or []) if str(item or "").strip()}
    if not normalized_labels:
        return None
    viewport_height = 0.0
    if near_bottom:
        try:
            viewport_height = float(
                page.evaluate("() => window.innerHeight || document.documentElement.clientHeight || 0") or 0.0
            )
        except Exception:
            viewport_height = 0.0
    best_node = None
    best_score: Optional[Tuple[float, float]] = None
    for selector in ("button", '[role="button"]', '[role="menuitem"]', '[role="option"]'):
        try:
            loc = page.locator(selector)
            count = min(120, int(loc.count() or 0))
        except Exception:
            continue
        for idx in range(count):
            try:
                node = loc.nth(idx)
                if not node.is_visible():
                    continue
                if near_bottom and viewport_height > 0.0:
                    box = _safe_bounding_box(node)
                    if box is not None:
                        center_y = float(box.get("y", 0.0) or 0.0) + (float(box.get("height", 0.0) or 0.0) / 2.0)
                        if center_y < (viewport_height * 0.55):
                            continue
                raw_values = [
                    str(node.inner_text(timeout=200) or "").strip(),
                    str(node.text_content(timeout=200) or "").strip(),
                    str(node.get_attribute("aria-label", timeout=200) or "").strip(),
                    str(node.get_attribute("title", timeout=200) or "").strip(),
                ]
                for value in raw_values:
                    inferred = _infer_chat_web_model_mode_from_text(value)
                    if inferred and inferred in normalized_labels:
                        box = _safe_bounding_box(node) or {}
                        score = (
                            float(box.get("y", 0.0) or 0.0) + float(box.get("height", 0.0) or 0.0),
                            float(box.get("x", 0.0) or 0.0) + float(box.get("width", 0.0) or 0.0),
                        )
                        if best_score is None or score > best_score:
                            best_node = node
                            best_score = score
                        break
                    if str(value or "").strip().lower() in normalized_labels:
                        box = _safe_bounding_box(node) or {}
                        score = (
                            float(box.get("y", 0.0) or 0.0) + float(box.get("height", 0.0) or 0.0),
                            float(box.get("x", 0.0) or 0.0) + float(box.get("width", 0.0) or 0.0),
                        )
                        if best_score is None or score > best_score:
                            best_node = node
                            best_score = score
                        break
            except Exception:
                continue
    return best_node


def _find_gemini_mode_button(page: Any) -> Optional[Any]:
    selectors = [
        'button[data-test-id="bard-mode-menu-button"]',
        'button[aria-label*="Open mode picker" i]',
        'button:has([data-test-id="logo-pill-label-container"])',
    ]
    for selector in selectors:
        loc = _first_visible_locator(page, selector, timeout_ms=400)
        if loc is not None:
            return loc
    return None


def _detect_gemini_chat_model_mode(page: Any) -> str:
    direct_button = _find_gemini_mode_button(page)
    if direct_button is not None:
        for getter in ("inner_text", "text_content"):
            try:
                value = getattr(direct_button, getter)(timeout=200)
            except TypeError:
                try:
                    value = getattr(direct_button, getter)()
                except Exception:
                    continue
            except Exception:
                continue
            inferred = _infer_chat_web_model_mode_from_text(value)
            if inferred:
                return inferred
        for attr_name in ("aria-label", "title"):
            try:
                value = direct_button.get_attribute(attr_name, timeout=200)
            except TypeError:
                try:
                    value = direct_button.get_attribute(attr_name)
                except Exception:
                    continue
            except Exception:
                continue
            inferred = _infer_chat_web_model_mode_from_text(value)
            if inferred:
                return inferred
    for near_bottom in (True, False):
        button = _find_visible_button_with_labels(page, ("pro", "fast", "thinking"), near_bottom=near_bottom)
        if button is None:
            continue
        for getter in ("inner_text", "text_content"):
            try:
                value = getattr(button, getter)(timeout=200)
            except TypeError:
                try:
                    value = getattr(button, getter)()
                except Exception:
                    continue
            except Exception:
                continue
            inferred = _infer_chat_web_model_mode_from_text(value)
            if inferred:
                return inferred
    try:
        body_text = str(page.locator("body").inner_text(timeout=250) or "")
    except Exception:
        body_text = ""
    return _infer_chat_web_model_mode_from_text(body_text)


def _build_gemini_model_option_selector(mode: str) -> str:
    desired = _normalize_chat_web_model_mode(mode)
    labels_by_mode = {
        "pro": ["Pro", "Gemini Pro", "Gemini Advanced"],
        "fast": ["Fast", "Flash"],
        "thinking": ["Thinking"],
    }
    selectors: List[str] = []
    for label in labels_by_mode.get(desired, []):
        escaped = re.escape(str(label or "").strip())
        selectors.extend(
            [
                f'text=/^{escaped}$/i',
                f'button:has-text("{label}")',
                f'[role="menuitem"]:has-text("{label}")',
                f'[role="option"]:has-text("{label}")',
            ]
        )
    return " || ".join(selectors)


def _normalize_gemini_chat_entry_url(chat_url: str, *, clean_thread: bool = False) -> str:
    normalized = str(chat_url or "").strip().rstrip("/")
    if not normalized:
        normalized = "https://gemini.google.com/app"
    if not clean_thread:
        return normalized
    match = re.match(r"^(https://gemini\.google\.com/app)(?:/[A-Za-z0-9_-]+)?(?:[?#].*)?$", normalized)
    if match:
        return match.group(1)
    return normalized


def _ensure_gemini_chat_model_mode(
    page: Any,
    desired_mode: str,
    *,
    allowed_modes: Optional[Sequence[str]] = None,
    settle_sec: float = 1.0,
) -> Dict[str, Any]:
    desired = _normalize_chat_web_model_mode(desired_mode)
    if not desired:
        return {"desired_mode": "", "current_mode": "", "verified": False, "changed": False, "reason": "no_desired_mode"}
    acceptable_modes = [
        mode
        for mode in (_normalize_chat_web_model_mode(item) for item in (allowed_modes or []))
        if mode
    ]
    if not acceptable_modes:
        acceptable_modes = [desired]
    current_mode = _detect_gemini_chat_model_mode(page)
    if current_mode in acceptable_modes:
        return {
            "desired_mode": desired,
            "current_mode": current_mode,
            "verified": True,
            "changed": False,
            "reason": "already_selected" if current_mode == desired else "already_acceptable",
            "allowed_modes": acceptable_modes,
        }

    opener = _find_gemini_mode_button(page)
    if opener is None:
        opener = _find_visible_button_with_labels(page, ("pro", "fast", "thinking"), near_bottom=True)
    if opener is None:
        opener = _find_visible_button_with_labels(page, ("pro", "fast", "thinking"), near_bottom=False)
    if opener is None:
        return {
            "desired_mode": desired,
            "current_mode": current_mode,
            "verified": current_mode in acceptable_modes,
            "changed": False,
            "reason": "mode_button_not_found",
            "allowed_modes": acceptable_modes,
        }

    try:
        _robust_click(opener)
    except Exception as exc:
        return {
            "desired_mode": desired,
            "current_mode": current_mode,
            "verified": False,
            "changed": False,
            "reason": f"mode_button_click_failed:{exc}",
            "allowed_modes": acceptable_modes,
        }
    try:
        page.wait_for_timeout(400)
    except Exception:
        pass

    option = _first_visible_locator(page, _build_gemini_model_option_selector(desired), timeout_ms=2500)
    if option is None:
        try:
            page.keyboard.press("Escape")
        except Exception:
            pass
        refreshed_mode = _detect_gemini_chat_model_mode(page)
        return {
            "desired_mode": desired,
            "current_mode": refreshed_mode or current_mode,
            "verified": (refreshed_mode or current_mode) in acceptable_modes,
            "changed": False,
            "reason": "mode_option_not_found",
            "allowed_modes": acceptable_modes,
        }

    try:
        _robust_click(option)
    except Exception as exc:
        return {
            "desired_mode": desired,
            "current_mode": current_mode,
            "verified": False,
            "changed": False,
            "reason": f"mode_option_click_failed:{exc}",
            "allowed_modes": acceptable_modes,
        }
    try:
        page.wait_for_timeout(max(250, int(max(0.0, float(settle_sec or 0.0)) * 1000.0)))
    except Exception:
        pass

    refreshed_mode = _detect_gemini_chat_model_mode(page)
    verified = refreshed_mode in acceptable_modes
    return {
        "desired_mode": desired,
        "current_mode": refreshed_mode,
        "verified": verified,
        "changed": verified and current_mode != refreshed_mode,
        "reason": (
            "mode_selected"
            if refreshed_mode == desired
            else ("mode_selected_acceptable" if verified else "mode_verification_failed")
        ),
        "allowed_modes": acceptable_modes,
    }


def _wait_for_new_chat_response_text(
    page: Any,
    baseline_text: str,
    *,
    baseline_candidates: Optional[Sequence[str]] = None,
    baseline_state: Optional[Dict[str, Any]] = None,
    timeout_sec: float,
    heartbeat: Optional[Callable[[], None]] = None,
    heartbeat_interval_sec: float = 10.0,
    response_stall_sec: float = 45.0,
    progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_interval_sec: float = 15.0,
    require_parseable_json: bool = False,
    preferred_top_level_key: str = "",
    response_candidate_validator: Optional[Callable[[str], bool]] = None,
) -> str:
    deadline = time.time() + max(10.0, float(timeout_sec))
    last_text = ""
    stable_count = 0
    last_heartbeat_ts = time.monotonic()
    last_change_ts = time.monotonic()
    started_ts = time.monotonic()
    last_progress_ts = started_ts
    transient_error_grace_sec = min(5.0, max(2.0, float(response_stall_sec or 45.0) * 0.2))
    baseline_history = [
        str(item or "").strip()
        for item in (baseline_candidates or [])
        if str(item or "").strip()
    ]
    if not baseline_history and str(baseline_text or "").strip():
        baseline_history = [str(baseline_text or "").strip()]
    baseline_snapshot = (
        dict(baseline_state)
        if isinstance(baseline_state, dict)
        else {
            "message_count": 0,
            "response_hash": "",
            "latest_text": str(baseline_text or "").strip(),
        }
    )
    baseline_message_count = max(0, int(baseline_snapshot.get("message_count", 0) or 0))
    baseline_response_hash = str(baseline_snapshot.get("response_hash", "") or "").strip()
    if not baseline_response_hash:
        baseline_response_hash = _hash_chat_response_entries([], message_count=baseline_message_count)
        baseline_snapshot["response_hash"] = baseline_response_hash
    if not str(baseline_snapshot.get("latest_text", "") or "").strip() and baseline_history:
        baseline_snapshot["latest_text"] = baseline_history[-1]
    last_message_count = baseline_message_count
    last_response_hash = baseline_response_hash
    stale_parseable_without_new = False
    rejected_candidate_count = 0
    stale_preview_repeat_count = 0
    last_stale_preview = str(baseline_snapshot.get("latest_text", "") or "").strip()
    last_preview_text = last_stale_preview
    saw_partial_preview_after_signal = False
    invalid_preferred_key_started_ts = None
    invalid_preferred_key_cycles = 0
    invalid_preferred_key_preview = ""
    invalid_preferred_key_grace_sec = max(
        8.0,
        min(90.0, max(10.0, float(response_stall_sec or 45.0) * 2.0)),
    )
    preview_accept_grace_sec = max(
        6.0,
        min(30.0, max(4.0, float(response_stall_sec or 45.0) * 0.5)),
    )
    while time.time() < deadline:
        interval_sec = max(2.0, float(heartbeat_interval_sec or 10.0))
        if heartbeat is not None:
            now = time.monotonic()
            if (now - last_heartbeat_ts) >= interval_sec:
                try:
                    heartbeat()
                except Exception:
                    pass
                last_heartbeat_ts = now
        body_txt = ""
        try:
            body_txt = str(page.locator("body").inner_text(timeout=200) or "")
            body_lower = body_txt.lower()
            if "reached your pro model limit" in body_lower or "reached your gemini advanced limit" in body_lower:
                raise RuntimeError("GEMINI_QUOTA_EXCEEDED: Quota limit reached, returning to cooldown.")
        except RuntimeError:
            raise
        except Exception:
            pass

        try:
            current_state = _capture_chat_response_state(
                page,
                limit=max(6, len(baseline_history) + 3),
            )
        except Exception:
            current_state = {
                "message_count": 0,
                "response_hash": "",
                "entries": [],
                "texts": [],
                "latest_text": "",
            }
        current_candidates = current_state.get("texts", []) if isinstance(current_state.get("texts"), list) else []
        current_message_count = max(0, int(current_state.get("message_count", 0) or 0))
        current_response_hash = str(current_state.get("response_hash", "") or "").strip()
        new_candidates = _new_chat_response_candidates_after_baseline(
            current_state,
            baseline_state=baseline_snapshot,
            baseline_candidates=baseline_history,
        )
        rejected_candidates: List[str] = []
        txt = ""
        for raw_candidate in reversed(list(new_candidates or [])):
            candidate_text = str(raw_candidate or "").strip()
            if not candidate_text:
                continue
            if require_parseable_json and candidate_text in baseline_history:
                rejected_candidates.append(candidate_text)
                continue
            parsed_candidate = _parse_chat_response_candidate(candidate_text)
            if preferred_top_level_key and isinstance(parsed_candidate, dict) and preferred_top_level_key not in parsed_candidate:
                rejected_candidates.append(candidate_text)
                continue
            if require_parseable_json and parsed_candidate is None:
                continue
            if response_candidate_validator is not None:
                try:
                    if not bool(response_candidate_validator(candidate_text)):
                        rejected_candidates.append(candidate_text)
                        continue
                except Exception:
                    pass
            txt = candidate_text
            break
        if rejected_candidates:
            baseline_history.extend(rejected_candidates)
            rejected_candidate_count += len(rejected_candidates)
        preview_text = txt or _select_preferred_chat_response_candidate(
            new_candidates if new_candidates else current_candidates,
            preferred_top_level_key=preferred_top_level_key,
            require_parseable_json=False,
        )
        if not preview_text:
            preview_text = str(current_state.get("latest_text", "") or "").strip()
        preview_now = time.monotonic()
        if preview_text != last_preview_text:
            last_preview_text = preview_text
            last_change_ts = preview_now
        hash_changed = bool(
            current_response_hash
            and baseline_response_hash
            and current_response_hash != baseline_response_hash
        )
        has_new_response_signal = current_message_count > baseline_message_count or hash_changed
        stop_visible_now = _quick_locator_visible(
            page,
            'button[aria-label*="Stop" i] || button:has-text("Stop responding") || button:has-text("Stop")',
        )
        send_visible_now = _quick_locator_visible(
            page,
            'button[aria-label*="Send" i] || button:has-text("Send") || button:has-text("Run")',
        )
        response_settled = (not stop_visible_now) or send_visible_now
        preview_parsed = _parse_chat_response_candidate(preview_text) if preview_text else None
        preview_matches_baseline = bool(preview_text and preview_text in baseline_history)
        preview_candidate_valid = False
        if preview_text and preview_parsed is not None:
            preview_candidate_valid = True
            if preferred_top_level_key:
                preview_candidate_valid = (
                    isinstance(preview_parsed, dict) and preferred_top_level_key in preview_parsed
                )
            if preview_matches_baseline:
                preview_candidate_valid = False
            if preview_candidate_valid and response_candidate_validator is not None:
                try:
                    preview_candidate_valid = bool(response_candidate_validator(preview_text))
                except Exception:
                    pass
        preview_has_expected_key = bool(
            preview_parsed is not None
            and (
                not preferred_top_level_key
                or (isinstance(preview_parsed, dict) and preferred_top_level_key in preview_parsed)
            )
            and not preview_matches_baseline
        )
        if has_new_response_signal and preview_text and preview_parsed is None:
            saw_partial_preview_after_signal = True
        if (
            require_parseable_json
            and preferred_top_level_key
            and isinstance(preview_parsed, dict)
            and preferred_top_level_key not in preview_parsed
            and has_new_response_signal
            and not txt
        ):
            invalid_preferred_key_cycles += 1
            if invalid_preferred_key_started_ts is None:
                invalid_preferred_key_started_ts = time.monotonic()
            invalid_preferred_key_preview = preview_text
        if require_parseable_json and not txt:
            if preview_parsed is not None:
                if preview_text and preview_text == last_stale_preview:
                    stale_preview_repeat_count += 1
                else:
                    stale_preview_repeat_count = 1 if preview_text else 0
                    last_stale_preview = preview_text
                if current_message_count <= baseline_message_count and not hash_changed:
                    stale_parseable_without_new = True
        if txt:
            now = time.monotonic()
            invalid_preferred_key_started_ts = None
            invalid_preferred_key_cycles = 0
            invalid_preferred_key_preview = ""
            if txt == last_text:
                stable_count += 1
            else:
                stable_count = 0
                last_text = txt
                last_change_ts = now
            candidate_is_parseable = _parse_chat_response_candidate(txt) is not None
            if candidate_is_parseable:
                if not require_parseable_json:
                    return txt
                candidate_committed = bool(
                    response_settled
                    or (has_new_response_signal and stable_count >= 2)
                )
                if candidate_committed:
                    return txt
            if require_parseable_json and _is_retryable_chat_error_text(txt or body_txt):
                if (now - last_change_ts) >= transient_error_grace_sec:
                    preview = re.sub(r"\s+", " ", str(txt or body_txt or "").strip())
                    if len(preview) > 160:
                        preview = preview[:157] + "..."
                    raise RuntimeError(f"GEMINI_TRANSIENT_ERROR_RESPONSE: {preview}")
                continue
            if (now - last_change_ts) >= max(8.0, float(response_stall_sec or 45.0)):
                return txt
            # Wait a bit for streaming to settle.
            if stable_count >= 2 and not require_parseable_json:
                return txt
        last_message_count = current_message_count
        last_response_hash = current_response_hash or last_response_hash
        now = time.monotonic()
        if (
            require_parseable_json
            and not txt
            and preview_matches_baseline
            and rejected_candidate_count == 0
            and stale_preview_repeat_count >= 2
            and response_settled
            and (now - started_ts) >= preview_accept_grace_sec
        ):
            raise RuntimeError("GEMINI_STALE_RESPONSE: same preview reused after baseline")
        if (
            require_parseable_json
            and not txt
            and has_new_response_signal
            and (
                preview_candidate_valid
                or (preview_has_expected_key and saw_partial_preview_after_signal)
            )
            and rejected_candidate_count == 0
            and stale_preview_repeat_count >= 2
            and response_settled
            and (now - started_ts) >= preview_accept_grace_sec
        ):
            return preview_text
        if (
            require_parseable_json
            and not txt
            and preview_parsed is not None
            and invalid_preferred_key_started_ts is None
            and rejected_candidate_count == 0
            and stale_preview_repeat_count >= 2
            and response_settled
            and (now - last_change_ts) >= max(8.0, float(response_stall_sec or 45.0))
        ):
            if current_message_count <= baseline_message_count and not hash_changed:
                raise RuntimeError("GEMINI_STALE_RESPONSE: same preview reused after baseline")
            raise RuntimeError("GEMINI_STALE_RESPONSE: parseable preview never stabilized into accepted response")
        if (now - last_progress_ts) >= max(5.0, float(progress_interval_sec or 15.0)):
            _emit_progress_hook(
                progress_hook,
                _build_wait_progress_payload(
                    page,
                    phase="response_wait",
                    elapsed_sec=now - started_ts,
                    remaining_sec=max(0.0, deadline - time.time()),
                    latest_text=preview_text,
                    baseline_text=baseline_text,
                    stable_count=stable_count,
                    last_change_age_sec=now - last_change_ts,
                    current_message_count=current_message_count,
                    baseline_message_count=baseline_message_count,
                    response_hash_changed=bool(
                        current_response_hash
                        and baseline_response_hash
                        and current_response_hash != baseline_response_hash
                    ),
                ),
            )
            last_progress_ts = now
        if (
            require_parseable_json
            and rejected_candidate_count > 0
            and not txt
            and preview_parsed is not None
            and has_new_response_signal
            and response_settled
            and stale_preview_repeat_count >= 2
            and (now - started_ts) >= preview_accept_grace_sec
        ):
            raise RuntimeError(
                "GEMINI_THREAD_CONTAMINATION: rejected stale candidates after baseline "
                f"({rejected_candidate_count})"
            )
        if (
            require_parseable_json
            and invalid_preferred_key_started_ts is not None
            and invalid_preferred_key_cycles >= 2
            and not txt
        ):
            invalid_elapsed_sec = time.monotonic() - invalid_preferred_key_started_ts
            if invalid_elapsed_sec >= invalid_preferred_key_grace_sec:
                preview = re.sub(r"\s+", " ", str(invalid_preferred_key_preview or "").strip())
                if len(preview) > 160:
                    preview = preview[:157] + "..."
                raise RuntimeError(
                    "GEMINI_THREAD_CONTAMINATION: repeated invalid top-level responses "
                    f"after baseline (expected {preferred_top_level_key}): {preview}"
                )
        try:
            page.wait_for_timeout(1000)
        except Exception:
            time.sleep(1.0)
    if require_parseable_json and _is_retryable_chat_error_text(last_text):
        preview = re.sub(r"\s+", " ", str(last_text or "").strip())
        if len(preview) > 160:
            preview = preview[:157] + "..."
        raise RuntimeError(f"GEMINI_TRANSIENT_ERROR_RESPONSE: {preview}")
    if require_parseable_json and rejected_candidate_count > 0 and not last_text:
        raise RuntimeError(
            "GEMINI_THREAD_CONTAMINATION: rejected stale candidates after baseline "
            f"({rejected_candidate_count})"
        )
    if require_parseable_json and stale_parseable_without_new and not last_text:
        if stale_preview_repeat_count >= 2:
            raise RuntimeError("GEMINI_STALE_RESPONSE: same preview reused after baseline")
        raise RuntimeError("GEMINI_STALE_RESPONSE: stale parseable body never advanced beyond baseline")
    if require_parseable_json and last_message_count <= baseline_message_count and not last_text:
        raise RuntimeError("GEMINI_NO_NEW_ASSISTANT_MESSAGE: no assistant response appeared after baseline")
    return last_text if last_text else ""


def _page_contains_text(page: Any, needle: str) -> bool:
    target = str(needle or "").strip().lower()
    if not target:
        return False
    try:
        body_text = str(page.locator("body").inner_text(timeout=1500) or "").lower()
    except Exception:
        return False
    return target in body_text


def _chat_response_has_required_fields(parsed: Any, response_schema: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(response_schema, dict):
        return True
    if not isinstance(parsed, dict):
        return False
    required = response_schema.get("required")
    if not isinstance(required, list):
        return True
    for key in required:
        name = str(key or "").strip()
        if not name:
            continue
        if name not in parsed:
            return False
    return True


def _wait_for_chat_upload_settle(
    page: Any,
    *,
    composer_locator: Optional[Any],
    baseline_tokens: Optional[List[str]],
    baseline_page_tokens: Optional[List[str]],
    expected_fragments: Optional[List[str]],
    require_google_drive_video: bool,
    size_mb: float,
    min_wait_sec: float,
    sec_per_100mb: float,
    max_wait_sec: float,
    heartbeat: Optional[Callable[[], None]] = None,
    heartbeat_interval_sec: float = 10.0,
    attachment_name: str = "",
    progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    progress_interval_sec: float = 15.0,
) -> Dict[str, Any]:
    def _specific_expected_fragments(values: Optional[List[str]]) -> List[str]:
        fragments: List[str] = []
        for value in values or []:
            fragment = str(value or "").strip().lower()
            if not fragment:
                continue
            if fragment in {".mp4", "mp4", "video_"}:
                continue
            fragments.append(fragment)
        return fragments

    def _token_text_matches_expected(values: Optional[List[str]], expected: List[str]) -> bool:
        tokens_text = " ".join(str(value or "").strip().lower() for value in (values or []) if str(value or "").strip())
        if not tokens_text:
            return False
        if not expected:
            return True
        return any(fragment in tokens_text for fragment in expected)

    wait_sec = max(
        min_wait_sec,
        min(max_wait_sec, (max(0.0, float(size_mb)) / 100.0) * sec_per_100mb),
    )
    deadline = time.time() + wait_sec
    last_tokens = baseline_tokens or []
    last_page_tokens = baseline_page_tokens or []
    last_heartbeat_ts = time.monotonic()
    started_ts = time.monotonic()
    last_progress_ts = started_ts
    while time.time() < deadline:
        interval_sec = max(2.0, float(heartbeat_interval_sec or 10.0))
        if heartbeat is not None:
            now = time.monotonic()
            if (now - last_heartbeat_ts) >= interval_sec:
                try:
                    heartbeat()
                except Exception:
                    pass
                last_heartbeat_ts = now
        tokens = _collect_attachment_tokens(page, composer_locator=composer_locator, local_only=True)
        page_tokens = _collect_attachment_tokens(page, composer_locator=composer_locator, local_only=False)
        evidence_tokens = [str(v or "").strip().lower() for v in [*tokens, *page_tokens] if str(v or "").strip()]
        body_text = ""
        try:
            body_text = str(page.locator("body").inner_text(timeout=1000) or "").lower()
        except Exception:
            body_text = ""
        expected = [str(v or "").strip().lower() for v in (expected_fragments or []) if str(v or "").strip()]
        specific_expected = _specific_expected_fragments(expected)
        evidence_match = True if not specific_expected else any(
            frag in " ".join(evidence_tokens) or frag in body_text for frag in specific_expected
        )
        drive_match = True
        if require_google_drive_video:
            drive_match = (
                "mp4 (google drive)" in " ".join(evidence_tokens)
                or "mp4 (google drive)" in body_text
                or ("google drive" in body_text and "mp4" in body_text)
            )
        token_delta = [value for value in tokens if value not in (baseline_tokens or [])]
        page_token_delta = [value for value in page_tokens if value not in (baseline_page_tokens or [])]
        token_delta_matches = _token_text_matches_expected(token_delta, specific_expected)
        page_token_delta_matches = _token_text_matches_expected(page_token_delta, specific_expected)
        body_attachment_hint = (
            require_google_drive_video
            and evidence_match
            and drive_match
            and (
                "mp4 (google drive)" in body_text
                or (
                    ("google drive" in body_text and "mp4" in body_text)
                    and ("video_" in body_text or any(frag in body_text for frag in expected))
                )
            )
        )
        if token_delta and token_delta_matches and drive_match:
            return {
                "wait_sec": max(0.0, wait_sec - max(0.0, deadline - time.time())),
                "confirmed": True,
                "tokens": token_delta,
            }
        if page_token_delta and page_token_delta_matches and drive_match:
            return {
                "wait_sec": max(0.0, wait_sec - max(0.0, deadline - time.time())),
                "confirmed": True,
                "tokens": page_token_delta,
            }
        if require_google_drive_video and evidence_match and drive_match and (tokens or page_tokens):
            return {
                "wait_sec": max(0.0, wait_sec - max(0.0, deadline - time.time())),
                "confirmed": True,
                "tokens": tokens or page_tokens,
            }
        if body_attachment_hint:
            return {
                "wait_sec": max(0.0, wait_sec - max(0.0, deadline - time.time())),
                "confirmed": True,
                "tokens": tokens or page_tokens or ["body:mp4 (google drive)"],
            }
        now = time.monotonic()
        if (now - last_progress_ts) >= max(5.0, float(progress_interval_sec or 15.0)):
            _emit_progress_hook(
                progress_hook,
                {
                    "phase": "attach_wait",
                    "attachment_name": str(attachment_name or "").strip(),
                    "elapsed_sec": round(max(0.0, now - started_ts), 1),
                    "remaining_sec": round(max(0.0, deadline - time.time()), 1),
                    "size_mb": round(max(0.0, float(size_mb or 0.0)), 1),
                    "token_count": len(tokens),
                    "page_token_count": len(page_tokens),
                    "evidence_match": bool(evidence_match),
                    "drive_match": bool(drive_match),
                    "attachment_hint": bool(body_attachment_hint),
                    "tokens": [str(item) for item in (tokens or page_tokens)[:3]],
                },
            )
            last_progress_ts = now
        last_tokens = tokens
        last_page_tokens = page_tokens
        try:
            page.wait_for_timeout(1000)
        except Exception:
            time.sleep(1.0)
    return {
        "wait_sec": wait_sec,
        "confirmed": False,
        "tokens": last_tokens or last_page_tokens,
    }


def _stop_active_gemini_response(page: Any, *, timeout_sec: float = 8.0) -> bool:
    stop_selector = 'button[aria-label*="Stop" i] || button:has-text("Stop responding") || button:has-text("Stop")'
    send_selector = 'button[aria-label*="Send" i] || button:has-text("Send") || button:has-text("Run")'
    if not _quick_locator_visible(page, stop_selector):
        return False
    stop_button = _first_visible_locator(page, stop_selector, timeout_ms=600)
    if stop_button is None:
        return False
    try:
        _robust_click(stop_button)
    except Exception:
        try:
            stop_button.click(timeout=1200, force=True)
        except Exception:
            return False
    deadline = time.monotonic() + max(1.0, float(timeout_sec or 0.0))
    while time.monotonic() < deadline:
        if not _quick_locator_visible(page, stop_selector):
            return True
        if _quick_locator_visible(page, send_selector):
            return True
        try:
            page.wait_for_timeout(250)
        except Exception:
            time.sleep(0.25)
    return not _quick_locator_visible(page, stop_selector)


def _send_chat_prompt(
    *,
    page: Any,
    chat_box: Any,
    send_selector: str,
    prompt_text: str,
    pre_send_ready_timeout_sec: float = 12.0,
    pre_send_settle_sec: float = 2.0,
) -> None:
    _stop_active_gemini_response(page, timeout_sec=max(4.0, pre_send_ready_timeout_sec))
    baseline_state = _capture_chat_response_state(page)
    baseline_candidates = list(baseline_state.get("texts", []) or [])
    baseline_text = str(baseline_state.get("latest_text", "") or "")
    attachment_tokens_before_send = _collect_attachment_tokens(
        page,
        composer_locator=chat_box,
        local_only=True,
    )
    attachment_dispatch_required = bool(attachment_tokens_before_send)

    stop_selector = 'button[aria-label*="Stop" i] || button:has-text("Stop responding") || button:has-text("Stop")'
    normalized_prompt = re.sub(r"\s+", " ", str(prompt_text or "").strip()).lower()
    prompt_marker = ""
    try:
        prompt_marker_match = re.search(r"\brequest_id=([a-z0-9_-]+)\b", str(prompt_text or ""), flags=re.IGNORECASE)
    except Exception:
        prompt_marker_match = None
    if prompt_marker_match:
        prompt_marker = f"request_id={prompt_marker_match.group(1)}".lower()
    elif normalized_prompt:
        prompt_marker = normalized_prompt[: min(120, len(normalized_prompt))]
    send_dispatch_timeout_sec = min(
        10.0,
        max(
            2.0,
            2.0 + (len(normalized_prompt) / 2500.0),
        ),
    )
    send_dispatch_stall_grace_sec = min(
        max(0.75, send_dispatch_timeout_sec * 0.6),
        max(1.5, send_dispatch_timeout_sec - 0.5),
    )
    pre_send_ready_timeout_sec = max(4.0, float(pre_send_ready_timeout_sec or 0.0))
    pre_send_settle_sec = max(0.0, float(pre_send_settle_sec or 0.0))

    selectors_to_try = []
    for raw_selector in (
        send_selector,
        "button[aria-label*='Send message' i]",
        "button[aria-label*='Send' i]",
        "button.send-button",
        "button[data-mat-icon-name='send']",
        "button:has-text('Send')",
    ):
        for selector_variant in _selector_variants(str(raw_selector or "")):
            if selector_variant and selector_variant not in selectors_to_try:
                selectors_to_try.append(selector_variant)

    def _composer_attachment_tokens() -> List[str]:
        return _collect_attachment_tokens(
            page,
            composer_locator=chat_box,
            local_only=True,
        )

    attachment_ready_tokens: List[str] = []

    def _body_upload_busy() -> bool:
        try:
            body_text = str(page.locator("body").inner_text(timeout=250) or "").lower()
        except Exception:
            body_text = ""
        return any(
            marker in body_text
            for marker in (
                "uploading",
                "processing",
                "preparing file",
                "adding file",
            )
        )

    def _attachments_still_pending() -> bool:
        if not attachment_dispatch_required:
            return False
        current_tokens = _composer_attachment_tokens()
        if _body_upload_busy():
            return True
        if attachment_ready_tokens:
            return not bool(current_tokens)
        return True

    def _attachment_tokens_look_ready(tokens: Sequence[str]) -> bool:
        token_text = " || ".join(str(item or "").strip().lower() for item in (tokens or []) if str(item or "").strip())
        if not token_text:
            return False
        return any(
            marker in token_text
            for marker in (
                "remove file",
                "remove attachment",
                "already uploaded a file named",
                "google drive",
            )
        )

    def _wait_for_attachment_readiness() -> None:
        nonlocal attachment_ready_tokens
        if not attachment_dispatch_required:
            return
        stable_hits = 0
        last_tokens: List[str] = []
        deadline = time.monotonic() + max(8.0, pre_send_ready_timeout_sec)
        while time.monotonic() < deadline:
            current_tokens = _composer_attachment_tokens()
            upload_busy = _body_upload_busy()
            tokens_look_ready = _attachment_tokens_look_ready(current_tokens)
            if current_tokens and current_tokens == last_tokens and tokens_look_ready and not upload_busy:
                stable_hits += 1
            elif current_tokens and tokens_look_ready and not upload_busy:
                stable_hits = 1
            else:
                stable_hits = 0
            last_tokens = list(current_tokens)
            if stable_hits >= 2:
                attachment_ready_tokens = list(current_tokens)
                return
            try:
                page.wait_for_timeout(500)
            except Exception:
                time.sleep(0.5)
        if last_tokens and _attachment_tokens_look_ready(last_tokens) and not _body_upload_busy():
            attachment_ready_tokens = list(last_tokens)

    _wait_for_attachment_readiness()

    _fill_chat_input(chat_box, prompt_text, page)

    try:
        page.wait_for_timeout(1000)
    except Exception:
        pass

    def _composer_text() -> str:
        for getter in ("inner_text", "text_content", "input_value"):
            try:
                value = getattr(chat_box, getter)()
            except Exception:
                continue
            text = str(value or "").strip()
            if text:
                return text
        return ""

    def _composer_has_prompt() -> bool:
        composer_text = re.sub(r"\s+", " ", _composer_text()).lower()
        if not composer_text:
            return False
        if prompt_marker and prompt_marker in composer_text:
            return True
        if normalized_prompt:
            prompt_prefix = normalized_prompt[: min(160, len(normalized_prompt))]
            prompt_suffix = normalized_prompt[-min(80, len(normalized_prompt)) :]
            if prompt_prefix and prompt_prefix in composer_text:
                return True
            if prompt_suffix and prompt_suffix in composer_text:
                return True
            if composer_text in normalized_prompt and len(composer_text) >= min(120, len(normalized_prompt)):
                return True
        return len(composer_text) >= 40

    def _prompt_posted_to_body(body_text: str) -> bool:
        posted_text = re.sub(r"\s+", " ", str(body_text or "").strip()).lower()
        if not posted_text:
            return False
        if prompt_marker and prompt_marker in posted_text:
            return True
        if normalized_prompt:
            prompt_prefix = normalized_prompt[: min(160, len(normalized_prompt))]
            prompt_suffix = normalized_prompt[-min(80, len(normalized_prompt)) :]
            if prompt_prefix and prompt_prefix in posted_text:
                return True
            if prompt_suffix and prompt_suffix in posted_text:
                return True
        return False

    def _get_send_button() -> Optional[Any]:
        for selector_variant in selectors_to_try:
            send_btn = _first_visible_locator(page, selector_variant, timeout_ms=350)
            if send_btn is not None:
                return send_btn
        return None

    def _wait_for_pre_send_ready() -> None:
        ready_since = 0.0
        deadline = time.monotonic() + pre_send_ready_timeout_sec
        while time.monotonic() < deadline:
            attachments_ready = not _attachments_still_pending()
            prompt_ready = _composer_has_prompt()
            send_ready = _get_send_button() is not None
            if attachments_ready and prompt_ready and send_ready:
                if ready_since <= 0.0:
                    ready_since = time.monotonic()
                if (time.monotonic() - ready_since) >= pre_send_settle_sec:
                    return
            else:
                ready_since = 0.0
            try:
                page.wait_for_timeout(250)
            except Exception:
                time.sleep(0.25)

    _wait_for_pre_send_ready()

    def _did_chat_send_start() -> bool:
        stop_visible = _quick_locator_visible(page, stop_selector)
        if stop_visible:
            return True
        if _attachments_still_pending():
            return False
        send_visible = False
        for selector_variant in selectors_to_try:
            if selector_variant and _quick_locator_visible(page, selector_variant):
                send_visible = True
                break
        try:
            body_text = str(page.locator("body").inner_text(timeout=300) or "").lower()
        except Exception:
            body_text = ""
        composer_text = re.sub(r"\s+", " ", _composer_text()).lower()
        try:
            current_state = _capture_chat_response_state(
                page,
                limit=max(6, len(baseline_candidates) + 3),
            )
        except Exception:
            current_state = {
                "message_count": 0,
                "response_hash": "",
                "entries": [],
                "texts": [],
                "latest_text": "",
            }
        new_candidates = _new_chat_response_candidates_after_baseline(
            current_state,
            baseline_state=baseline_state,
            baseline_candidates=baseline_candidates,
        )
        latest_text = _select_preferred_chat_response_candidate(new_candidates, require_parseable_json=False)
        # A stale or newly rendered assistant preview is not enough to prove
        # dispatch while the send button is still visible. Gemini can keep the
        # old response body on screen until a real send toggles the composer
        # into the "Stop response" state.
        if latest_text and not send_visible:
            return True
        if _prompt_posted_to_body(body_text):
            # Guard: if the send button is still visible AND the prompt is
            # still in the composer, the prompt text in body_text is from
            # the composer itself – not from a submitted message bubble.
            if send_visible and _composer_has_prompt():
                pass  # false positive – prompt is still in the input box
            else:
                return True
        return False

    def _send_dispatch_stalled() -> bool:
        if _quick_locator_visible(page, stop_selector):
            return False
        if attachment_dispatch_required and _attachments_still_pending():
            return True
        send_visible = False
        for selector_variant in selectors_to_try:
            if selector_variant and _quick_locator_visible(page, selector_variant):
                send_visible = True
                break
        if not send_visible:
            return False
        composer_text = re.sub(r"\s+", " ", _composer_text()).lower()
        if not composer_text:
            return False
        if normalized_prompt:
            return composer_text in normalized_prompt or normalized_prompt in composer_text
        return len(composer_text) >= 20

    def _wait_for_chat_send_dispatch(timeout_sec: float = 2.0) -> bool:
        started = time.monotonic()
        deadline = time.monotonic() + max(1.0, float(timeout_sec or 0.0))
        while time.monotonic() < deadline:
            if _did_chat_send_start():
                return True
            if (time.monotonic() - started) >= send_dispatch_stall_grace_sec and _send_dispatch_stalled():
                return False
            try:
                page.wait_for_timeout(250)
            except Exception:
                time.sleep(0.25)
        return _did_chat_send_start()

    def _restore_prompt_if_missing() -> None:
        if _composer_has_prompt():
            return
        _fill_chat_input(chat_box, prompt_text, page)
        try:
            page.wait_for_timeout(350)
        except Exception:
            time.sleep(0.35)
        _wait_for_pre_send_ready()

    sent = False
    _send_attempt_counter = 0

    def _attempt_send_button(action: Callable[[Any], None], label: str = "") -> bool:
        nonlocal _send_attempt_counter
        _send_attempt_counter += 1
        _restore_prompt_if_missing()
        send_btn = _get_send_button()
        if send_btn is None:
            print(f"[trace] send attempt #{_send_attempt_counter} ({label}): no send button found", flush=True)
            return False
        try:
            action(send_btn)
            print(f"[trace] send attempt #{_send_attempt_counter} ({label}): action executed", flush=True)
        except Exception as exc:
            print(f"[trace] send attempt #{_send_attempt_counter} ({label}): action failed: {exc}", flush=True)
            return False
        if _wait_for_chat_send_dispatch(timeout_sec=send_dispatch_timeout_sec):
            print(f"[trace] send attempt #{_send_attempt_counter} ({label}): dispatch confirmed", flush=True)
            return True
        print(f"[trace] send attempt #{_send_attempt_counter} ({label}): dispatch NOT confirmed", flush=True)
        _restore_prompt_if_missing()
        return False

    # Try Enter on the composer FIRST – Gemini's contenteditable composer
    # dispatches the message on plain Enter and this avoids click-target issues
    # with the Send button on the landing page.
    if not sent:
        try:
            _send_attempt_counter += 1
            print(f"[trace] send attempt #{_send_attempt_counter} (enter_first): trying Enter on chat_box", flush=True)
            _restore_prompt_if_missing()
            chat_box.click(timeout=1000, force=True)
            page.wait_for_timeout(300)
            chat_box.press("Enter", timeout=1200)
            sent = _wait_for_chat_send_dispatch(timeout_sec=send_dispatch_timeout_sec)
            print(f"[trace] send attempt #{_send_attempt_counter} (enter_first): sent={sent}", flush=True)
        except Exception as exc:
            print(f"[trace] send attempt #{_send_attempt_counter} (enter_first): failed: {exc}", flush=True)
            sent = False

    if not sent and _attempt_send_button(lambda send_btn: send_btn.click(timeout=2000, force=True), "pw_click_force"):
        sent = True

    if not sent and _attempt_send_button(lambda send_btn: send_btn.evaluate("(el) => el.click()"), "js_click"):
        sent = True

    if not sent:
        def _mouse_click_send(send_btn: Any) -> None:
            box = send_btn.bounding_box()
            if not box:
                raise RuntimeError("send button missing bounding box")
            page.mouse.click(
                float(box.get("x", 0.0)) + (float(box.get("width", 0.0)) / 2.0),
                float(box.get("y", 0.0)) + (float(box.get("height", 0.0)) / 2.0),
            )
        if _attempt_send_button(_mouse_click_send, "mouse_click"):
            sent = True

    if not sent:
        try:
            _send_attempt_counter += 1
            print(f"[trace] send attempt #{_send_attempt_counter} (ctrl_enter): trying Ctrl+Enter on chat_box", flush=True)
            _restore_prompt_if_missing()
            chat_box.click(timeout=1000, force=True)
            chat_box.press("Control+Enter", timeout=1200)
            sent = _wait_for_chat_send_dispatch(timeout_sec=send_dispatch_timeout_sec)
            print(f"[trace] send attempt #{_send_attempt_counter} (ctrl_enter): sent={sent}", flush=True)
        except Exception as exc:
            print(f"[trace] send attempt #{_send_attempt_counter} (ctrl_enter): failed: {exc}", flush=True)
            sent = False

    if not sent:
        try:
            _send_attempt_counter += 1
            print(f"[trace] send attempt #{_send_attempt_counter} (enter): trying Enter on chat_box", flush=True)
            _restore_prompt_if_missing()
            chat_box.click(timeout=1000, force=True)
            chat_box.press("Enter", timeout=1200)
            sent = _wait_for_chat_send_dispatch(timeout_sec=send_dispatch_timeout_sec)
            print(f"[trace] send attempt #{_send_attempt_counter} (enter): sent={sent}", flush=True)
        except Exception as exc:
            print(f"[trace] send attempt #{_send_attempt_counter} (enter): failed: {exc}", flush=True)
            sent = False

    if not sent:
        try:
            _restore_prompt_if_missing()
            page.keyboard.press("Control+Enter")
            sent = _wait_for_chat_send_dispatch(timeout_sec=send_dispatch_timeout_sec)
        except Exception:
            sent = False

    if not sent:
        try:
            _restore_prompt_if_missing()
            page.keyboard.press("Enter")
            sent = _wait_for_chat_send_dispatch(timeout_sec=send_dispatch_timeout_sec)
        except Exception:
            sent = False

    if not sent:
        if _did_chat_send_start():
            sent = True

    if not sent:
        if attachment_dispatch_required and _attachments_still_pending():
            raise RuntimeError(
                "Could not send prompt in Gemini chat (attachment stayed pending in composer)."
            )
        raise RuntimeError("Could not send prompt in Gemini chat (Enter/send button failed).")
        
    try:
        page.wait_for_timeout(1000)
    except Exception:
        pass


def _seed_chat_thread(
    *,
    page: Any,
    chat_box: Any,
    input_selector: str,
    send_selector: str,
    timeout_sec: float,
    seed_context_text: str,
    heartbeat: Optional[Callable[[], None]] = None,
    heartbeat_interval_sec: float = 10.0,
    response_stall_sec: float = 45.0,
) -> Tuple[Any, List[str]]:
    notes: List[str] = []
    seed = str(seed_context_text or "").strip()
    if not seed:
        notes.append("chat_seed_context_empty")
        return chat_box, notes

    marker_hash = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:12]
    marker = f"ATLAS_DISCORD_SYNC_HASH: {marker_hash}"
    if _page_contains_text(page, marker):
        notes.append("chat_seed_context_already_present")
        return chat_box, notes

    baseline_state = _capture_chat_response_state(page)
    baseline_candidates = list(baseline_state.get("texts", []) or [])
    baseline = str(baseline_state.get("latest_text", "") or "")
    seed_prompt = (
        "Store the following Atlas Discord and policy context as authoritative supplemental memory for "
        "subsequent Atlas annotation turns in this chat. If it conflicts with older thread assumptions, "
        "prefer this context. Reply with CONTEXT_SYNCED only.\n\n"
        f"{marker}\n\n"
        f"{seed}"
    )
    _send_chat_prompt(page=page, chat_box=chat_box, send_selector=send_selector, prompt_text=seed_prompt)
    ack_text = _wait_for_new_chat_response_text(
        page,
        baseline,
        baseline_candidates=baseline_candidates,
        baseline_state=baseline_state,
        timeout_sec=min(90.0, max(20.0, timeout_sec)),
        heartbeat=heartbeat,
        heartbeat_interval_sec=heartbeat_interval_sec,
        response_stall_sec=response_stall_sec,
    )
    notes.append("chat_seed_context_sent")
    if ack_text:
        notes.append("chat_seed_context_ack")
    chat_box = _first_visible_locator(page, input_selector, timeout_ms=30000)
    if chat_box is None:
        raise RuntimeError("Gemini chat input disappeared after seed context.")
    _prepare_chat_composer_for_attach(page, chat_box)
    return chat_box, notes


def _prepare_clean_chat_thread(
    *,
    page: Any,
    input_selector: str,
    send_selector: str,
    timeout_sec: float,
    memory_primer_text: str,
    base_url: str = "",
    heartbeat: Optional[Callable[[], None]] = None,
    heartbeat_interval_sec: float = 10.0,
    response_stall_sec: float = 45.0,
) -> Tuple[Any, List[str]]:
    notes: List[str] = []
    clean_input_selector = (
        f"{str(input_selector or '').strip()} || "
        'div[role="textbox"] || '
        'div[aria-label*="Ask Gemini" i] || '
        'textarea[aria-label*="Ask Gemini" i] || '
        'div.ql-editor'
    )
    try:
        target_url = _normalize_gemini_chat_entry_url(
            str(base_url or "https://gemini.google.com/app").strip() or "https://gemini.google.com/app",
            clean_thread=True,
        )
        page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(1800)
    except Exception as exc:
        notes.append(f"clean_thread_base_nav_failed={exc}")

    new_chat_btn = None
    candidates = [
        lambda: _last_visible_candidate(
            page.locator('[role="button"][aria-label*="New chat" i]'),
            timeout_ms=3000,
        ),
        lambda: _last_visible_candidate(
            page.locator('a[role="button"][aria-label*="New chat" i]'),
            timeout_ms=3000,
        ),
        lambda: _last_visible_candidate(page.get_by_role("button", name="New chat"), timeout_ms=3000),
        lambda: _last_visible_candidate(page.get_by_text("New chat", exact=True), timeout_ms=3000),
        lambda: _last_visible_candidate(page.locator('text=/^New chat$/i'), timeout_ms=3000),
    ]
    for factory in candidates:
        try:
            new_chat_btn = factory()
        except Exception:
            new_chat_btn = None
        if new_chat_btn is not None:
            break
    if new_chat_btn is not None:
        try:
            _robust_click(new_chat_btn)
            page.wait_for_timeout(1200)
            notes.append("clean_thread_opened")
        except Exception as exc:
            notes.append(f"clean_thread_open_failed={exc}")
    else:
        notes.append("clean_thread_button_missing")

    chat_box = _first_visible_locator(page, clean_input_selector, timeout_ms=30000)
    if chat_box is None:
        try:
            ask_gemini = _last_visible_candidate(
                page.get_by_text("Ask Gemini", exact=False),
                timeout_ms=2500,
            )
            if ask_gemini is not None:
                _robust_click(ask_gemini)
                page.wait_for_timeout(1200)
        except Exception:
            pass
        chat_box = _first_visible_locator(page, clean_input_selector, timeout_ms=12000)
    if chat_box is None:
        fallback_box = _first_visible_locator(page, input_selector, timeout_ms=4000)
        if fallback_box is not None:
            notes.append("clean_thread_input_fallback_original_selector")
            chat_box = fallback_box
        else:
            raise RuntimeError("Gemini chat input not visible after opening clean thread.")
    try:
        chat_box.scroll_into_view_if_needed(timeout=2000)
    except Exception:
        pass
    try:
        chat_box.click(timeout=2500)
        page.wait_for_timeout(300)
    except Exception:
        pass
    _prepare_chat_composer_for_attach(page, chat_box)

    primer = str(memory_primer_text or "").strip()
    if primer:
        baseline_state = _capture_chat_response_state(page)
        baseline_candidates = list(baseline_state.get("texts", []) or [])
        baseline = str(baseline_state.get("latest_text", "") or "")
        primer_prompt = (
            "Use the following Atlas project memory for all subsequent turns in this chat. "
            "Acknowledge with READY only.\n\n"
            f"{primer}"
        )
        _send_chat_prompt(page=page, chat_box=chat_box, send_selector=send_selector, prompt_text=primer_prompt)
        _wait_for_new_chat_response_text(
            page,
            baseline,
            baseline_candidates=baseline_candidates,
            baseline_state=baseline_state,
            timeout_sec=min(90.0, max(20.0, timeout_sec)),
            heartbeat=heartbeat,
            heartbeat_interval_sec=heartbeat_interval_sec,
            response_stall_sec=response_stall_sec,
        )
        notes.append("clean_thread_memory_primer_sent")
        chat_box = _first_visible_locator(page, input_selector, timeout_ms=30000)
        if chat_box is None:
            raise RuntimeError("Gemini chat input disappeared after memory primer.")
        _prepare_chat_composer_for_attach(page, chat_box)
    else:
        notes.append("clean_thread_memory_primer_empty")
    return chat_box, notes


def _call_gemini_compare_chat_web(
    *,
    cfg: Dict[str, Any],
    prompt: str,
    video_a: Optional[Path],
    video_b: Optional[Path],
    episode_id: str = "",
    response_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError("Playwright is required for chat_web mode.") from exc

    gem = cfg.get("gemini", {}) if isinstance(cfg.get("gemini"), dict) else {}
    chat_url = str(
        gem.get("chat_web_url", "https://gemini.google.com/app") or ""
    ).strip() or "https://gemini.google.com/app"
    
    # Removed active authuser for multi-account rotation (requested by user)
    # The script will rely exclusively on the primary signed-in account

    headless = bool(gem.get("chat_web_headless", True))
    timeout_sec = max(20.0, float(gem.get("chat_web_timeout_sec", 180) or 180))
    max_upload_mb = max(50.0, float(gem.get("chat_web_max_upload_mb", 2048) or 2048))
    attach_secondary = bool(gem.get("chat_web_attach_secondary_video", False))
    input_sel = str(gem.get("chat_web_input_selector", 'div[contenteditable="true"] || textarea') or "").strip()
    send_sel = str(
        gem.get(
            "chat_web_send_selector",
            'button[aria-label*="Send" i] || button:has-text("Send") || button:has-text("Run")',
        )
        or ""
    ).strip()
    file_input_sel = str(gem.get("chat_web_file_input_selector", 'input[type="file"]') or "").strip()
    attach_button_sel = str(
        gem.get(
            "chat_web_attach_button_selector",
            'button[aria-label*="Open upload file menu" i] || button[aria-label*="Add files" i] || button[aria-label*="Upload" i] || button[aria-label*="Tools" i] || button:has-text("Add files") || button:has-text("Upload") || button:has-text("Tools")',
        )
        or ""
    ).strip()
    upload_menu_sel = str(
        gem.get(
            "chat_web_upload_menu_selector",
            'button[aria-label*="Upload files" i] || [role="menuitem"]:has-text("Upload files") || button:has-text("Upload files") || [role="option"]:has-text("Upload files") || text=/^Upload files$/i',
        )
        or ""
    ).strip()
    channel = str(
        gem.get("chat_web_channel", "") or os.environ.get("GEMINI_CHAT_CHROME_CHANNEL", "") or "chrome"
    ).strip()
    cdp_url = str(
        gem.get("chat_web_connect_over_cdp_url", "")
        or os.environ.get("GEMINI_CHAT_CONNECT_OVER_CDP_URL", "")
        or ""
    ).strip()
    cdp_connect_timeout_ms = max(10000, int(gem.get("chat_web_cdp_connect_timeout_ms", 45000) or 45000))
    storage_state = str(
        gem.get("chat_web_storage_state", "") or os.environ.get("GEMINI_CHAT_STORAGE_STATE", "") or ".state/gemini_chat_storage_state.json"
    ).strip()
    user_data_dir = str(
        gem.get("chat_web_user_data_dir", "") or os.environ.get("GEMINI_CHAT_USER_DATA_DIR", "") or ".state/gemini_chat_user_data"
    ).strip()
    ignore_automation = bool(gem.get("chat_web_ignore_automation", True))
    upload_settle_min_sec = max(1.0, float(gem.get("chat_web_upload_settle_min_sec", 4.0) or 4.0))
    upload_settle_sec_per_100mb = max(
        0.0, float(gem.get("chat_web_upload_settle_sec_per_100mb", 12.0) or 12.0)
    )
    upload_settle_max_sec = max(
        upload_settle_min_sec, float(gem.get("chat_web_upload_settle_max_sec", 45.0) or 45.0)
    )
    prefer_drive_picker = bool(gem.get("chat_web_prefer_drive_picker", False))
    drive_root_folder_url = str(gem.get("chat_web_drive_root_folder_url", "") or "").strip()
    clean_thread_fallback_enabled = bool(gem.get("chat_web_clean_thread_fallback_enabled", False))
    force_clean_thread = bool(gem.get("chat_web_force_clean_thread", False))
    allow_text_only_fallback_on_network_error = bool(
        gem.get("allow_text_only_fallback_on_network_error", False)
    )
    memory_primer_text = _load_chat_memory_primer(gem)
    seed_context_text = _load_chat_seed_context(gem)
    send_seed_context = bool(gem.get("chat_web_seed_context_send_before_prompt", False))
    apply_stealth = bool(gem.get("chat_web_apply_stealth", False))
    json_followup_retry = bool(gem.get("chat_web_json_followup_retry", True))
    raw_args = gem.get("chat_web_launch_args", [])
    launch_args: List[str] = _build_chat_web_launch_args(raw_args)
    # Running browser as root on Linux requires --no-sandbox.
    try:
        if hasattr(os, "geteuid") and int(os.geteuid()) == 0:
            if "--no-sandbox" not in launch_args:
                launch_args.append("--no-sandbox")
            if "--disable-dev-shm-usage" not in launch_args:
                launch_args.append("--disable-dev-shm-usage")
    except Exception:
        pass

    attach_candidates: List[Path] = []
    if video_a is not None and video_a.exists():
        attach_candidates.append(video_a)
    if attach_secondary and video_b is not None and video_b.exists():
        attach_candidates.append(video_b)

    raw_text = ""
    attach_notes: List[str] = []
    skip_shutdown_after_success = bool(gem.get("chat_web_skip_shutdown_after_success", True))

    pw_runner = sync_playwright()
    pw = pw_runner.start()
    context = None
    browser = None
    page = None
    owns_context = True
    owns_browser = True
    skip_shutdown = False
    _diag_t0 = time.monotonic()
    def _diag(phase: str) -> None:
        print(f"[chat_web_diag] {phase}: {time.monotonic() - _diag_t0:.1f}s elapsed")
    try:
        try:
            launch_mode = "connect_over_cdp"
            # Default to local CDP if not specified
            if not cdp_url:
                cdp_url = "http://127.0.0.1:9222"
            
            try:
                _diag(f"cdp_connect_start url={cdp_url} timeout={cdp_connect_timeout_ms}ms")
                browser = pw.chromium.connect_over_cdp(cdp_url, timeout=cdp_connect_timeout_ms)
                owns_browser = False
                if browser.contexts:
                    context = browser.contexts[0]
                    owns_context = False
                else:
                    context = browser.new_context()
                    owns_context = True
                _diag("cdp_connect_ok")
            except Exception as exc:
                raise RuntimeError(f"CDP connection failed: {exc}. Ensure 'atlas-chrome-cdp.service' is running.")

            _diag(f"browser_ready launch_mode={launch_mode}")

            if apply_stealth:
                _apply_chat_web_stealth(context)
            
            # Smart Tab Management: prefer the exact configured Gemini thread,
            # and avoid hijacking an unrelated Gemini tab when the URL targets
            # a dedicated custom chat.
            page = None
            matched_page = _pick_existing_gemini_chat_page(context, chat_url=chat_url)
            if matched_page is not None:
                page = matched_page
                print(f"[chat_web] switching to existing Gemini tab: {page.url}")

            # Cleanup extra redundant tabs (especially about:blank)
            while len(context.pages) > 2:
                try:
                    p_to_close = context.pages[-1]
                    close_url = str(getattr(p_to_close, "url", "") or "").strip().lower()
                    if close_url in {"", "about:blank"}:
                        p_to_close.close()
                        continue
                    break
                except:
                    break

            if not page:
                if len(context.pages) > 1 and not _is_dedicated_gemini_chat_url(chat_url):
                    # Reuse the second tab if it's blank or not Atlas
                    cand = context.pages[1]
                    if "atlascapture.io" not in cand.url:
                        page = cand
                        print(f"[chat_web] reusing second tab for Gemini: {page.url}")

            if not page:
                page = context.new_page()
                if _is_dedicated_gemini_chat_url(chat_url):
                    print("[chat_web] opened dedicated Gemini tab in window.")
                else:
                    print("[chat_web] opened new Gemini tab in window.")

            page.goto(chat_url, wait_until="domcontentloaded", timeout=60000)
            _diag("page_loaded")
            
            desired_mode = _resolve_chat_web_ui_model_mode(gem, requested_model=str(gem.get("model", "") or "").strip())
            try:
                page.wait_for_timeout(2000)
            except Exception:
                pass
            _ensure_gemini_chat_model_mode(page, desired_mode, settle_sec=1.0)
            try:
                page.wait_for_timeout(6000)
            except Exception:
                pass
            _handle_gemini_consent_if_present(page)

            chat_box = _first_visible_locator(page, input_sel, timeout_ms=30000)
            if chat_box is None:
                raise RuntimeError("Gemini chat input not visible. Login/session is likely missing. Please login via VNC.")

            if prefer_drive_picker and attach_candidates:
                import threading
                def _background_stage():
                    try:
                        _stage_episode_artifacts_for_drive_picker(
                            cfg,
                            episode_id=episode_id,
                            paths=attach_candidates,
                        )
                    except Exception as e:
                        print(f"[chat_web] async drive stage error: {e}")
                
                threading.Thread(target=_background_stage, daemon=True).start()
                attach_notes.append("drive_stage_started_async")
                print("[chat_web] Deferred Drive upload to background thread.")
                
                # Turn off prefer_drive_picker for the rest of this workflow because
                # the file won't be immediately available for the UI picker.
                prefer_drive_picker = False
            if force_clean_thread:
                chat_box, clean_notes = _prepare_clean_chat_thread(
                    page=page,
                    input_selector=input_sel,
                    send_selector=send_sel,
                    timeout_sec=timeout_sec,
                    memory_primer_text=memory_primer_text,
                )
                attach_notes.extend(clean_notes)
            if send_seed_context:
                chat_box, seed_notes = _seed_chat_thread(
                    page=page,
                    chat_box=chat_box,
                    input_selector=input_sel,
                    send_selector=send_sel,
                    timeout_sec=timeout_sec,
                    seed_context_text=seed_context_text,
                )
                attach_notes.extend(seed_notes)

            attach_notes.extend(
                _attach_files_via_chat_ui(
                page=page,
                composer_locator=chat_box,
                attach_candidates=attach_candidates,
                episode_id=episode_id,
                prefer_drive_picker=prefer_drive_picker,
                drive_root_folder_url=drive_root_folder_url,
                max_upload_mb=max_upload_mb,
                attach_button_sel=attach_button_sel,
                upload_menu_sel=upload_menu_sel,
                file_input_sel=file_input_sel,
                upload_settle_min_sec=upload_settle_min_sec,
                upload_settle_sec_per_100mb=upload_settle_sec_per_100mb,
                upload_settle_max_sec=upload_settle_max_sec,
                )
            )
            attached_any = any("attached" in str(note or "").lower() for note in (attach_notes or []))
            if not attached_any and clean_thread_fallback_enabled:
                chat_box, clean_notes = _prepare_clean_chat_thread(
                    page=page,
                    input_selector=input_sel,
                    send_selector=send_sel,
                    timeout_sec=timeout_sec,
                    memory_primer_text=memory_primer_text,
                )
                attach_notes.extend(clean_notes)
                if send_seed_context:
                    chat_box, seed_notes = _seed_chat_thread(
                        page=page,
                        chat_box=chat_box,
                        input_selector=input_sel,
                        send_selector=send_sel,
                        timeout_sec=timeout_sec,
                        seed_context_text=seed_context_text,
                    )
                    attach_notes.extend(seed_notes)
                attach_notes.extend(
                    _attach_files_via_chat_ui(
                        page=page,
                        composer_locator=chat_box,
                        attach_candidates=attach_candidates,
                        episode_id=episode_id,
                        prefer_drive_picker=prefer_drive_picker,
                        drive_root_folder_url=drive_root_folder_url,
                        max_upload_mb=max_upload_mb,
                        attach_button_sel=attach_button_sel,
                        upload_menu_sel=upload_menu_sel,
                        file_input_sel=file_input_sel,
                        upload_settle_min_sec=upload_settle_min_sec,
                        upload_settle_sec_per_100mb=upload_settle_sec_per_100mb,
                        upload_settle_max_sec=upload_settle_max_sec,
                    )
                )
                attached_any = any("attached" in str(note or "").lower() for note in (attach_notes or []))
            if attach_candidates and not attached_any and not allow_text_only_fallback_on_network_error:
                raise RuntimeError(
                    "Gemini chat video attachment failed; refusing to send prompt without video context."
                )

            _diag("files_attached")
            baseline_state = _capture_chat_response_state(page)
            baseline_candidates = list(baseline_state.get("texts", []) or [])
            baseline_text = str(baseline_state.get("latest_text", "") or "")
            _send_chat_prompt(
                page=page,
                chat_box=chat_box,
                send_selector=send_sel,
                prompt_text=prompt,
            )
            _diag("prompt_sent")

            preferred_key = ""
            if isinstance(response_schema, dict) and isinstance(response_schema.get("required"), list):
                for key in response_schema.get("required", []):
                    clean_key = str(key or "").strip()
                    if clean_key:
                        preferred_key = clean_key
                        break
            raw_text = _wait_for_new_chat_response_text(
                page,
                baseline_text=baseline_text,
                baseline_candidates=baseline_candidates,
                baseline_state=baseline_state,
                timeout_sec=timeout_sec,
                preferred_top_level_key=preferred_key,
            )
            _diag(f"response_received len={len(raw_text or '')}")
            if not raw_text:
                raise RuntimeError("Timed out waiting for Gemini chat response.")

            parsed_preview: Any = None
            try:
                parsed_preview = json.loads(_clean_json_text(raw_text))
            except Exception:
                parsed_preview = None

            # Manual chat often needs a short second turn to force strict JSON.
            if json_followup_retry and not _chat_response_has_required_fields(parsed_preview, response_schema):
                required_keys = []
                if isinstance(response_schema, dict) and isinstance(response_schema.get("required"), list):
                    required_keys = [str(k) for k in response_schema.get("required", []) if str(k or "").strip()]
                followup = (
                    "Rewrite your last answer as strict JSON only with no markdown and no prose. "
                    "Return exactly one JSON object."
                )
                if required_keys:
                    followup += " Required keys: " + ", ".join(required_keys) + "."
                baseline_retry_state = _capture_chat_response_state(page)
                baseline_retry_candidates = list(baseline_retry_state.get("texts", []) or [])
                baseline_retry = str(baseline_retry_state.get("latest_text", "") or "")
                _send_chat_prompt(
                    page=page,
                    chat_box=chat_box,
                    send_selector=send_sel,
                    prompt_text=followup,
                )
                retry_text = _wait_for_new_chat_response_text(
                    page,
                    baseline_text=baseline_retry,
                    baseline_candidates=baseline_retry_candidates,
                    baseline_state=baseline_retry_state,
                    timeout_sec=max(25.0, timeout_sec * 0.6),
                    preferred_top_level_key=preferred_key,
                )
                if retry_text:
                    raw_text = retry_text
                    attach_notes.append("json_followup_retry_used")
            skip_shutdown = _should_skip_chat_web_shutdown(raw_text=raw_text, gem_cfg=gem) and skip_shutdown_after_success
        finally:
            if not skip_shutdown:
                try:
                    if page is not None:
                        page.close()
                except Exception:
                    pass
                try:
                    if context is not None and owns_context:
                        context.close()
                except Exception:
                    pass
                if browser is not None and owns_browser:
                    try:
                        browser.close()
                    except Exception:
                        pass
    finally:
        if not skip_shutdown:
            try:
                pw_runner.stop()
            except Exception:
                pass

    try:
        parsed = json.loads(_clean_json_text(raw_text))
    except Exception:
        parsed = {"raw_text": raw_text}
    return {
        "parsed": parsed,
        "raw_text": raw_text,
        "attach_notes": attach_notes,
        "usage": {},
    }


def _vertex_access_token(credentials_path: Path) -> str:
    try:
        from google.auth.transport.requests import Request as GoogleAuthRequest
        from google.oauth2 import service_account
    except Exception as exc:
        raise RuntimeError("google-auth package is required for vertex_ai mode.") from exc

    creds = service_account.Credentials.from_service_account_file(
        str(credentials_path),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    req = GoogleAuthRequest()
    creds.refresh(req)
    token = str(getattr(creds, "token", "") or "").strip()
    if not token:
        raise RuntimeError("Could not obtain Vertex access token.")
    return token


class GeminiKeyPool:
    def __init__(self, explicit_key: str, dotenv: Dict[str, str]):
        self.keys: List[str] = []
        if explicit_key and explicit_key.strip():
            self.keys.append(explicit_key.strip())
            
        pool_raw = _read_secret("GEMINI_API_KEYS_POOL", dotenv)
        if pool_raw:
            for k in pool_raw.split(","):
                val = k.strip()
                if val and val not in self.keys:
                    self.keys.append(val)
                    
        for k_name in ["GEMINI_API_KEY", "GEMINI_API_KEY2", "GEMINI_API_KEY_FALLBACK", "GOOGLE_API_KEY"]:
            val = _read_secret(k_name, dotenv)
            if val and val not in self.keys:
                self.keys.append(val)
                
        self.current_index = 0

    def get_current_key(self) -> str:
        if not self.keys:
            return ""
        return self.keys[self.current_index]

    def switch_to_next(self) -> bool:
        if len(self.keys) <= 1:
            return False
        self.current_index = (self.current_index + 1) % len(self.keys)
        return True
        
    def has_multiple_keys(self) -> bool:
        return len(self.keys) > 1

_global_key_pool: Optional[GeminiKeyPool] = None

def _get_global_key_pool(explicit_key: str, dotenv: Dict[str, str]) -> GeminiKeyPool:
    global _global_key_pool
    if _global_key_pool is None:
        _global_key_pool = GeminiKeyPool(explicit_key, dotenv)
    return _global_key_pool


def _call_gemini_compare(
    cfg: Dict[str, Any],
    dotenv: Dict[str, str],
    model: str,
    prompt: str,
    video_a: Optional[Path],
    video_b: Optional[Path],
    cache_dir: Path,
    episode_id: str = "",
    response_schema: Optional[Dict[str, Any]] = None,
    usage_mode: str = "triplet_compare",
) -> Dict[str, Any]:
    gem = cfg.get("gemini", {}) if isinstance(cfg.get("gemini"), dict) else {}
    auth_mode = _normalize_auth_mode(gem.get("auth_mode", "api_key"))
    vertex_cached_content_name = str(
        gem.get("vertex_cached_content_name", "")
        or _read_secret("VERTEX_CACHED_CONTENT_NAME", dotenv)
    ).strip()
    if auth_mode == "chat_web":
        return _call_gemini_compare_chat_web(
            cfg=cfg,
            prompt=prompt,
            video_a=video_a,
            video_b=video_b,
            episode_id=episode_id,
            response_schema=response_schema,
        )

    max_inline_mb = float(gem.get("max_inline_video_mb", 20.0) or 20.0)
    connect_timeout_sec = max(5, int(gem.get("connect_timeout_sec", 30) or 30))
    request_timeout_sec = max(30, int(gem.get("request_timeout_sec", 420) or 420))

    parts: list[Dict[str, Any]] = [{"text": prompt}]
    attach_notes: list[str] = []
    for vid in [video_a, video_b]:
        if vid is None:
            continue
        part, note = _video_part_for_inline(vid, max_inline_mb=max_inline_mb, cache_dir=cache_dir)
        attach_notes.append(note)
        if part is not None:
            parts.append(part)

    generation_cfg = {
        "temperature": float(gem.get("temperature", 0.0) or 0.0),
        "responseMimeType": "application/json",
        "candidateCount": 1,
    }
    if isinstance(response_schema, dict) and response_schema:
        generation_cfg["responseSchema"] = response_schema
    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_cfg,
    }

    system_instruction = str(gem.get("system_instruction_text", "") or "").strip()
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    headers: Dict[str, str]
    url: str
    payload_to_send: Dict[str, Any] = payload

    if auth_mode == "vertex_ai":
        project = str(gem.get("vertex_project", "") or "").strip() or _read_secret(
            "GOOGLE_CLOUD_PROJECT", dotenv
        )
        location = str(gem.get("vertex_location", "") or "").strip() or _read_secret(
            "GOOGLE_CLOUD_LOCATION", dotenv
        ) or "us-central1"
        cred_path_raw = str(gem.get("vertex_credentials_path", "") or "").strip() or _read_secret(
            "GOOGLE_APPLICATION_CREDENTIALS", dotenv
        )
        if not project:
            raise RuntimeError("Missing Vertex project (gemini.vertex_project / GOOGLE_CLOUD_PROJECT).")
        if not cred_path_raw:
            raise RuntimeError(
                "Missing Vertex credentials path (gemini.vertex_credentials_path / GOOGLE_APPLICATION_CREDENTIALS)."
            )
        cred_path = Path(cred_path_raw)
        if not cred_path.exists():
            raise RuntimeError(f"Vertex credentials file not found: {cred_path}")
        token = _vertex_access_token(cred_path)
        model_id = _normalize_vertex_model_id(model)
        model_path = model_id if "/" in model_id else f"publishers/google/models/{model_id}"
        host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
        url = f"https://{host}/v1/projects/{project}/locations/{location}/{model_path}:generateContent"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        payload_to_send = _translate_payload_for_vertex(payload)
        if vertex_cached_content_name:
            payload_to_send["cachedContent"] = vertex_cached_content_name
            if bool(gem.get("vertex_cached_content_strip_system_instruction", True)):
                payload_to_send.pop("systemInstruction", None)
    else:
        api_key = str(gem.get("api_key", "") or "").strip()
        pool = _get_global_key_pool(api_key, dotenv)
        api_key = pool.get_current_key()
        if not api_key:
            raise RuntimeError("Missing Gemini API key (GEMINI_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEYS_POOL).")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}

    if auth_mode != "vertex_ai":
        max_attempts = len(_global_key_pool.keys) if _global_key_pool and _global_key_pool.keys else 1
        for attempt in range(max_attempts):
            resp = requests.post(
                url,
                headers=headers,
                json=payload_to_send,
                timeout=(connect_timeout_sec, request_timeout_sec),
            )
            if resp.status_code == 429 and _global_key_pool and _global_key_pool.has_multiple_keys():
                print(f"[GeminiKeyPool] Key {_global_key_pool.current_index + 1}/{len(_global_key_pool.keys)} exhausted (429). Switching...")
                _global_key_pool.switch_to_next()
                headers["X-goog-api-key"] = _global_key_pool.get_current_key()
                # Optional: pause 2s to not hammer immediately.
                time.sleep(2.0)
                continue
            break
    else:
        resp = requests.post(
            url,
            headers=headers,
            json=payload_to_send,
            timeout=(connect_timeout_sec, request_timeout_sec),
        )
    if (
        (resp.status_code == 400 and "does not match the model in the cached content" in str(resp.text or "").lower())
        or (resp.status_code == 404 and "cached content" in str(resp.text or "").lower())
    ) and auth_mode == "vertex_ai" and bool(vertex_cached_content_name):
        # Fallback: when cache is missing (404) or model mismatch (400), retry once without cached content.
        retry_payload = dict(payload_to_send)
        retry_payload.pop("cachedContent", None)
        # Also restore systemInstruction if it was stripped for the cached version
        if system_instruction and "systemInstruction" not in retry_payload:
            retry_payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        retry_resp = requests.post(
            url,
            headers=headers,
            json=retry_payload,
            timeout=(connect_timeout_sec, request_timeout_sec),
        )
        if retry_resp.status_code == 200:
            resp = retry_resp
            attach_notes.append("retried_without_cached_content_due_error")
        else:
            raise RuntimeError(
                f"Gemini compare request failed with cached-content error (HTTP {resp.status_code}); "
                f"retry without cache also failed HTTP {retry_resp.status_code}: {retry_resp.text[:800]}"
            )
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini compare request failed HTTP {resp.status_code}: {resp.text[:800]}")
    data = resp.json()
    usage_meta = data.get("usageMetadata", {}) if isinstance(data, dict) else {}
    _log_gemini_usage(
        cfg,
        model=model,
        mode=usage_mode,
        usage_meta=usage_meta,
        key_source=auth_mode,
    )
    text = _extract_text_from_response_json(data)
    if not text:
        raise RuntimeError("Gemini compare returned empty text response.")
    try:
        parsed = json.loads(_clean_json_text(text))
    except Exception:
        parsed = {"raw_text": text}
    return {
        "parsed": parsed,
        "raw_text": text,
        "attach_notes": attach_notes,
        "usage": usage_meta,
    }


def _build_timed_labels_prompt(
    episode_id: str = "",
    context_text: str = "",
    tier2_draft_text: str = "",
    *,
    repair_from_tier2: bool = True,
    require_gapless_timeline: bool = True,
    strict_action_policy: bool = True,
) -> str:
    eid = str(episode_id or "").strip()
    eid_line = f"Episode ID: {eid}\n" if eid else ""
    draft_text = str(tier2_draft_text or "").strip()
    draft_block = ""
    draft_rules = ""
    if repair_from_tier2 and draft_text:
        draft_rules = """
6) Tier2 draft is provided. Treat it as a noisy draft to repair, not ground truth.
7) Keep useful structure from draft only when video evidence agrees.
8) If draft contradicts video, follow video and correct it.
""".strip()
        draft_block = f"""
[Tier2 Draft To Repair]
{draft_text}
""".strip()

    policy_rules = ""
    if strict_action_policy:
        policy_rules = """
9) Use only visible physical actions, not intent/goal language.
10) Avoid intent/goal phrases like: "to loosen", "to remove", "prepare to", "try to".
11) Each action clause must start with a concrete action verb (e.g. pick up, place, hold, move, turn, pull, push, insert, remove, tighten, loosen, align, open, close, cut, wipe, clean, press, release).
12) If a visible action continues past 10.0 seconds, split it at or before 10.0 seconds and continue the same action in the next segment. Never keep one segment longer than 10.0 seconds just because the action is continuous.
13) If one hand keeps holding object X while the other hand performs another visible action, start the label with "hold X" first (example: "hold nozzle, insert fuel nozzle into tank").
14) Replace "drop" with "place". Avoid vague verbs like inspect/check/reach/work on/tweak.
15) Keep object naming consistent across the full timeline.
16) Use "No Action" only when there is no relevant hand/object interaction.
17) Keep labels concise (prefer <=2 atomic actions per segment).
""".strip()

    timeline_rules = """
4) Ensure segments are chronological and non-overlapping.
""".strip()
    if require_gapless_timeline:
        timeline_rules += """
5) Ensure a gapless timeline: if there is idle time, insert a "No Action" segment instead of leaving gaps.
""".rstrip()
    else:
        timeline_rules += """
5) Use "No Action" only when there is clearly no relevant action.
""".rstrip()

    context_block = ""
    context_clean = str(context_text or "").strip()
    if context_clean:
        context_block = f"""
16) Apply project-specific context below exactly:
[Project Context]
{context_clean}
""".strip()
    return f"""
You are generating Atlas timed action labels from the attached video.
{eid_line}Rules:
1) Output ONLY valid JSON (no markdown, no commentary).
2) JSON schema:
{{
  "segments": [
    {{"start_sec": 0.0, "end_sec": 1.2, "label": "action 1, action 2"}}
  ]
}}
3) Keep timestamps in seconds.
{timeline_rules}
{draft_rules}
{policy_rules}
{context_block}
{draft_block}
""".strip()


def _build_triplet_compare_prompt(
    *,
    tier2_text: str,
    api_text: str,
    chat_text: str,
    vertex_chat_text: str,
    task_state_text: str,
    context_text: str = "",
    include_thought_process: bool = True,
) -> str:
    context_block = ""
    context_clean = str(context_text or "").strip()
    if context_clean:
        context_block = f"""
[Project Context]
{context_clean}
""".strip()
    thought_line = '  "thought_process": "short internal analysis before final verdict",' if include_thought_process else ""
    return f"""
You are a strict Atlas annotation QA judge.
Use attached videos as source of truth.
If OCR text has minor typo but refers to same physical object, prioritize physical consistency and note typo in major_issues.

Compare exactly 4 candidate solutions:
1) Tier2 (employee draft)
2) Gemini API (3.1 pro style)
3) Gemini Chat (3.1 pro style)
4) Vertex Chat (3.1 pro style via Vertex AI)

Decide which solution is best and safest (least hallucination).
If all are bad, choose "none".

Return ONLY valid JSON with this shape:
{{
{thought_line}
  "winner": "tier2|api|chat|vertex_chat|none",
  "submit_safe_solution": "tier2|api|chat|vertex_chat|none",
  "scores": {{"tier2": 0, "api": 0, "chat": 0, "vertex_chat": 0}},
  "hallucination": {{"tier2": false, "api": false, "chat": false, "vertex_chat": false}},
  "major_issues": {{
    "tier2": [],
    "api": [],
    "chat": [],
    "vertex_chat": []
  }},
  "best_reason_short": "",
  "final_recommendation": ""
}}

{context_block}

[Tier2]
{tier2_text}

[Gemini API]
{api_text}

[Gemini Chat]
{chat_text}

[Vertex Chat]
{vertex_chat_text}

[Task State Optional]
{task_state_text}
""".strip()


def generate_gemini_chat_timed_labels(
    *,
    config_path: str,
    video_path: str,
    video_path_limit: str = "",
    remote: str = "",
    cache_dir: str = "tmp/triplet_chat_labels_cache",
    model: str = "",
    out_txt: str = "",
    out_json: str = "",
    episode_id: str = "",
    auth_mode_override: str = "",
    prompt_scope: str = "timed_labels",
    tier2_draft_path: str = "",
    tier2_draft_text: str = "",
) -> Dict[str, Any]:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise RuntimeError(f"Config file not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise RuntimeError("Config root must be a YAML object.")
    cfg_dir = cfg_path.parent.resolve()

    dotenv = _load_dotenv(Path(".env"))
    resolved_remote = str(remote or os.environ.get("RCLONE_REMOTE", "gdrive")).strip() or "gdrive"
    cache_root = Path(cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    video_main = _resolve_input_path(video_path, cache_root / "inputs", resolved_remote)
    video_limit: Optional[Path] = None
    if str(video_path_limit or "").strip():
        video_limit = _resolve_input_path(video_path_limit, cache_root / "inputs", resolved_remote)

    gem_cfg = cfg.get("gemini", {}) if isinstance(cfg.get("gemini"), dict) else {}
    scope = str(prompt_scope or "timed_labels").strip().lower() or "timed_labels"
    prompt_context = _build_prompt_context(gem_cfg, cfg_dir=cfg_dir, scope=scope)
    draft_text = str(tier2_draft_text or "").strip()
    draft_path_resolved = ""
    if not draft_text:
        raw_draft_path = str(tier2_draft_path or "").strip()
        if raw_draft_path:
            try:
                draft_path = _resolve_input_path(raw_draft_path, cache_root / "inputs", resolved_remote)
                draft_path_resolved = str(draft_path)
                draft_text = _load_text_or_json(draft_path)
            except Exception:
                draft_text = ""

    repair_from_tier2 = bool(
        gem_cfg.get(
            f"{scope}_repair_from_tier2",
            gem_cfg.get("timed_labels_repair_from_tier2", True),
        )
    )
    require_gapless_timeline = bool(
        gem_cfg.get(
            f"{scope}_require_gapless_timeline",
            gem_cfg.get("timed_labels_require_gapless_timeline", True),
        )
    )
    strict_action_policy = bool(
        gem_cfg.get(
            f"{scope}_strict_action_policy",
            gem_cfg.get("timed_labels_strict_action_policy", True),
        )
    )
    tier2_max_chars = max(
        0,
        int(
            gem_cfg.get(
                f"{scope}_tier2_max_chars",
                gem_cfg.get("timed_labels_tier2_max_chars", 16000),
            )
            or 0
        ),
    )
    if tier2_max_chars > 0 and len(draft_text) > tier2_max_chars:
        draft_text = draft_text[:tier2_max_chars].rstrip() + "\n...[truncated]"
    if not repair_from_tier2:
        draft_text = ""

    prompt = _build_timed_labels_prompt(
        episode_id=episode_id,
        context_text=prompt_context,
        tier2_draft_text=draft_text,
        repair_from_tier2=repair_from_tier2,
        require_gapless_timeline=require_gapless_timeline,
        strict_action_policy=strict_action_policy,
    )

    selected_model = _first_non_empty(
        model,
        gem_cfg.get("chat_timed_model", ""),
        gem_cfg.get("timed_labels_model", ""),
        gem_cfg.get("model", "gemini-3.1-pro-preview"),
    )
    cfg_for_call = dict(cfg)
    cfg_gem = dict(gem_cfg) if isinstance(gem_cfg, dict) else {}
    timed_temp = gem_cfg.get(f"{scope}_temperature", None)
    if timed_temp is None:
        timed_temp = gem_cfg.get("timed_labels_temperature", gem_cfg.get("chat_timed_temperature", None))
    if timed_temp is not None:
        cfg_gem["temperature"] = timed_temp
    scope_auth_mode = _first_non_empty(
        auth_mode_override,
        gem_cfg.get(f"{scope}_auth_mode", ""),
        gem_cfg.get("auth_mode", ""),
    )
    if scope_auth_mode:
        cfg_gem["auth_mode"] = scope_auth_mode
    timed_system_instruction = _resolve_system_instruction_text(
        gem_cfg,
        cfg_dir=cfg_dir,
        scope=scope,
        alias_text_keys=["timed_labels_system_instruction_text", "chat_timed_system_instruction_text"],
        alias_file_keys=["timed_labels_system_instruction_file", "chat_timed_system_instruction_file"],
    )
    if timed_system_instruction:
        cfg_gem["system_instruction_text"] = timed_system_instruction
    # For chat_web mode: allow secondary attach so upload_opt can be used automatically.
    cfg_gem["chat_web_attach_secondary_video"] = True
    cfg_for_call["gemini"] = cfg_gem
    auth_mode_effective = _normalize_auth_mode(str(cfg_gem.get("auth_mode", "") or ""))
    externalize_chat_web_retries = auth_mode_effective == "chat_web"
    timed_schema_enabled = bool(
        gem_cfg.get(f"{scope}_response_schema_enabled", gem_cfg.get("timed_labels_response_schema_enabled", True))
    )
    timed_response_schema = _timed_labels_response_schema() if timed_schema_enabled else None

    model_candidates = _ordered_gen3_model_candidates(
        gem_cfg,
        selected_model,
        "chat_timed_fallback_model",
        "timed_labels_fallback_model",
        "triplet_fallback_model",
    )
    if externalize_chat_web_retries and model_candidates:
        model_candidates = model_candidates[:1]

    result: Optional[Dict[str, Any]] = None
    used_model = model_candidates[0]
    attempt_errors: List[str] = []
    run_notes: List[str] = []
    if repair_from_tier2:
        if draft_text:
            run_notes.append("repair_from_tier2=on")
        else:
            run_notes.append("repair_from_tier2=on_no_draft")
    else:
        run_notes.append("repair_from_tier2=off")
    if require_gapless_timeline:
        run_notes.append("gapless_timeline=on")
    if strict_action_policy:
        run_notes.append("strict_action_policy=on")
    retry_attempts = max(
        1,
        int(
            gem_cfg.get(
                f"{scope}_retry_attempts",
                gem_cfg.get("timed_labels_retry_attempts", gem_cfg.get("max_retries", 3)),
            )
            or 3
        ),
    )
    retry_attempts = _effective_timed_labels_retry_attempts(
        requested_attempts=retry_attempts,
        auth_mode=auth_mode_effective,
    )
    retry_base_delay = max(0.5, float(gem_cfg.get("retry_base_delay_sec", 2.0) or 2.0))
    if externalize_chat_web_retries:
        run_notes.append("chat_web_retries_externalized_to_subprocess")

    for model_name in model_candidates:
        for attempt in range(1, retry_attempts + 1):
            try:
                result = _call_gemini_compare(
                    cfg=cfg_for_call,
                    dotenv=dotenv,
                    model=model_name,
                    prompt=prompt,
                    video_a=video_main,
                    video_b=video_limit,
                    cache_dir=cache_root / "video_inline",
                    episode_id=episode_id,
                    response_schema=timed_response_schema,
                    usage_mode=f"timed_labels:{scope}",
                )
                used_model = model_name
                notes = result.get("attach_notes", [])
                attached_any = False
                if isinstance(notes, list):
                    attached_any = any("attached" in str(n or "").lower() for n in notes)
                if not attached_any and auth_mode_effective == "chat_web" and not externalize_chat_web_retries:
                    time.sleep(2.0)
                    retry_same = _call_gemini_compare(
                        cfg=cfg_for_call,
                        dotenv=dotenv,
                        model=model_name,
                        prompt=prompt,
                        video_a=video_main,
                        video_b=video_limit,
                        cache_dir=cache_root / "video_inline",
                        episode_id=episode_id,
                        response_schema=timed_response_schema,
                        usage_mode=f"timed_labels:{scope}",
                    )
                    retry_same_notes = retry_same.get("attach_notes", [])
                    retry_same_attached = False
                    if isinstance(retry_same_notes, list):
                        retry_same_attached = any("attached" in str(n or "").lower() for n in retry_same_notes)
                    if retry_same_attached:
                        if isinstance(retry_same_notes, list):
                            retry_same_notes = [*retry_same_notes, "retry_same_video_after_unconfirmed_attach"]
                        else:
                            retry_same_notes = ["retry_same_video_after_unconfirmed_attach"]
                        retry_same["attach_notes"] = retry_same_notes
                        result = retry_same
                        attached_any = True
                # If no video was attached, retry once using upload_opt only (if available).
                if not attached_any and video_limit is not None and not externalize_chat_web_retries:
                    retry = _call_gemini_compare(
                        cfg=cfg_for_call,
                        dotenv=dotenv,
                        model=model_name,
                        prompt=prompt,
                        video_a=video_limit,
                        video_b=None,
                        cache_dir=cache_root / "video_inline",
                        episode_id=episode_id,
                        response_schema=timed_response_schema,
                        usage_mode=f"timed_labels:{scope}",
                    )
                    retry_notes = retry.get("attach_notes", [])
                    if isinstance(retry_notes, list):
                        retry_notes = [*retry_notes, "retry_with_upload_opt_video"]
                    else:
                        retry_notes = ["retry_with_upload_opt_video"]
                    retry["attach_notes"] = retry_notes
                    result = retry
                break
            except Exception as exc:
                msg = str(exc)
                low = msg.lower()
                attempt_errors.append(f"{model_name}#{attempt}: {msg}")
                is_chat_web_boot_error = auth_mode_effective == "chat_web" and _is_chat_web_boot_error_text(low)
                is_transient = any(
                    t in low
                    for t in (
                        "http 429",
                        "resource exhausted",
                        "http 500",
                        "http 502",
                        "http 503",
                        "http 504",
                        "timeout",
                        "timed out",
                    )
                )
                if is_chat_web_boot_error:
                    is_transient = False

                if is_transient and attempt < retry_attempts and not externalize_chat_web_retries:
                    sleep_sec = min(30.0, retry_base_delay * (2 ** (attempt - 1)))
                    run_notes.append(f"timed_transient_retry_{attempt}_sleep_{sleep_sec:.1f}s")
                    time.sleep(sleep_sec)
                    continue

                if video_limit is not None and not is_chat_web_boot_error and not externalize_chat_web_retries:
                    try:
                        retry = _call_gemini_compare(
                            cfg=cfg_for_call,
                            dotenv=dotenv,
                            model=model_name,
                            prompt=prompt,
                            video_a=video_limit,
                            video_b=None,
                            cache_dir=cache_root / "video_inline",
                            episode_id=episode_id,
                            response_schema=timed_response_schema,
                            usage_mode=f"timed_labels:{scope}",
                        )
                        retry_notes = retry.get("attach_notes", [])
                        if isinstance(retry_notes, list):
                            retry_notes = [*retry_notes, "retry_after_error_with_upload_opt_video"]
                        else:
                            retry_notes = ["retry_after_error_with_upload_opt_video"]
                        retry["attach_notes"] = retry_notes
                        result = retry
                        used_model = model_name
                        run_notes.append("fallback_to_upload_opt_after_error")
                        break
                    except Exception as retry_exc:
                        attempt_errors.append(f"{model_name}/upload_opt#{attempt}: {retry_exc}")
                # Move to next model after final attempt for this model.
                break
        if result is not None:
            break

    if result is None:
        tail = " | ".join(attempt_errors[-4:]) if attempt_errors else "unknown timed labels failure"
        raise RuntimeError(tail)

    parsed = result.get("parsed", {})
    raw_text = str(result.get("raw_text") or "")
    segments = parse_timed_segments_payload(parsed)
    if not segments:
        segments = parse_timed_segments_text(raw_text)
    if not segments:
        raise RuntimeError("Gemini timed labels response did not contain parseable segments.")
    if require_gapless_timeline:
        segments = _fill_timeline_gaps_with_no_action(segments, start_at_zero=True)
    timed_text = segments_to_timed_text(segments)
    if not timed_text:
        raise RuntimeError("Gemini timed labels were parsed but empty after normalization.")

    out_txt_path = Path(out_txt).resolve() if str(out_txt or "").strip() else cache_root / "text_chat_generated.txt"
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path.write_text(timed_text + "\n", encoding="utf-8")

    out_json_path: Optional[Path] = None
    if str(out_json or "").strip():
        out_json_path = Path(out_json).resolve()
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(
            json.dumps(
                {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "episode_id": str(episode_id or "").strip().lower(),
                    "model": used_model,
                    "segments": segments,
                    "attach_notes": [*(result.get("attach_notes", []) if isinstance(result.get("attach_notes"), list) else []), *run_notes],
                    "usage": result.get("usage", {}) if isinstance(result.get("usage"), dict) else {},
                    "tier2_draft_used": bool(draft_text),
                    "tier2_draft_path_resolved": draft_path_resolved,
                    "require_gapless_timeline": require_gapless_timeline,
                    "strict_action_policy": strict_action_policy,
                    "raw_text": raw_text,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return {
        "episode_id": str(episode_id or "").strip().lower(),
        "model": used_model,
        "segment_count": len(segments),
        "out_txt": str(out_txt_path),
        "out_json": str(out_json_path) if out_json_path else "",
        "attach_notes": [*(result.get("attach_notes", []) if isinstance(result.get("attach_notes"), list) else []), *run_notes],
        "usage": result.get("usage", {}) if isinstance(result.get("usage"), dict) else {},
    }


def _build_hybrid_refine_prompt(original_prompt: str, draft_text: str) -> str:
    return f"""
{original_prompt}

[Vertex AI Initial Draft]
{draft_text}

[Refinement Task]
The above draft was produced by another model. It might contain minor inaccuracies or formatting issues.
Please review the video carefully and provide a "100% Perfect" version of the labels.
Follow all original constraints but prioritize absolute accuracy against the video evidence.
Return your answer in the same strict JSON format.
""".strip()


def run_triplet_compare(
    *,
    config_path: str,
    video_path: str,
    tier2_path: str,
    api_path: str,
    video_path_limit: str = "",
    chat_path: str = "",
    vertex_chat_path: str = "",
    task_state_path: str = "",
    labels_path: str = "",
    remote: str = "",
    cache_dir: str = "tmp/triplet_compare_cache",
    model: str = "",
    out: str = "outputs/triplet_compare_result.json",
    episode_id: str = "",
) -> Dict[str, Any]:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise RuntimeError(f"Config file not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise RuntimeError("Config root must be a YAML object.")
    cfg_dir = cfg_path.parent.resolve()

    dotenv = _load_dotenv(Path(".env"))
    resolved_remote = str(remote or os.environ.get("RCLONE_REMOTE", "gdrive")).strip() or "gdrive"
    cache_root = Path(cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    video_main = _resolve_input_path(video_path, cache_root / "inputs", resolved_remote)
    video_limit = (
        _resolve_input_path(video_path_limit, cache_root / "inputs", resolved_remote)
        if str(video_path_limit or "").strip()
        else None
    )

    tier2_file = _resolve_input_path(tier2_path, cache_root / "inputs", resolved_remote)
    api_file = _resolve_input_path(api_path, cache_root / "inputs", resolved_remote)

    chat_file: Optional[Path] = None
    if str(chat_path or "").strip():
        chat_file = _resolve_input_path(chat_path, cache_root / "inputs", resolved_remote)
    elif str(labels_path or "").strip():
        chat_file = _resolve_input_path(labels_path, cache_root / "inputs", resolved_remote)

    vertex_chat_file: Optional[Path] = None
    if str(vertex_chat_path or "").strip():
        vertex_chat_file = _resolve_input_path(vertex_chat_path, cache_root / "inputs", resolved_remote)

    task_state_file: Optional[Path] = None
    if str(task_state_path or "").strip():
        task_state_file = _resolve_input_path(task_state_path, cache_root / "inputs", resolved_remote)

    tier2_text = _load_text_or_json(tier2_file)
    api_text = _load_text_or_json(api_file)
    chat_text = _load_text_or_json(chat_file) if chat_file else ""
    vertex_chat_text = _load_text_or_json(vertex_chat_file) if vertex_chat_file else ""
    task_state_text = _load_text_or_json(task_state_file) if task_state_file else ""

    gem_cfg = cfg.get("gemini", {}) if isinstance(cfg.get("gemini"), dict) else {}
    selected_model = _first_non_empty(
        model,
        gem_cfg.get("compare_model", ""),
        gem_cfg.get("triplet_compare_model", ""),
        gem_cfg.get("model", "gemini-3.1-pro-preview"),
    )
    retry_attempts = max(1, int(gem_cfg.get("triplet_retry_attempts", 3) or 3))
    include_thought_process = bool(gem_cfg.get("compare_include_thought_process", True))
    compare_schema_enabled = bool(gem_cfg.get("compare_response_schema_enabled", True))
    compare_fail_on_none = bool(gem_cfg.get("compare_fail_on_none", False))
    prompt_context = _build_prompt_context(gem_cfg, cfg_dir=cfg_dir, scope="compare")
    prompt = _build_triplet_compare_prompt(
        tier2_text=tier2_text,
        api_text=api_text,
        chat_text=chat_text,
        vertex_chat_text=vertex_chat_text,
        task_state_text=task_state_text,
        context_text=prompt_context,
        include_thought_process=include_thought_process,
    )

    cfg_for_call = dict(cfg)
    cfg_gem = dict(gem_cfg) if isinstance(gem_cfg, dict) else {}
    compare_temp = gem_cfg.get("compare_temperature", gem_cfg.get("triplet_compare_temperature", None))
    if compare_temp is not None:
        cfg_gem["temperature"] = compare_temp
    compare_auth_mode = _first_non_empty(
        gem_cfg.get("compare_auth_mode", ""),
        gem_cfg.get("auth_mode", ""),
    )
    if compare_auth_mode:
        cfg_gem["auth_mode"] = compare_auth_mode
    compare_system_instruction = _resolve_system_instruction_text(
        gem_cfg,
        cfg_dir=cfg_dir,
        scope="compare",
        alias_text_keys=["triplet_compare_system_instruction_text"],
        alias_file_keys=["triplet_compare_system_instruction_file"],
    )
    if compare_system_instruction:
        cfg_gem["system_instruction_text"] = compare_system_instruction
    cfg_for_call["gemini"] = cfg_gem
    compare_response_schema = (
        _triplet_compare_response_schema(include_thought_process=include_thought_process)
        if compare_schema_enabled
        else None
    )

    # Robust execution strategy:
    # 1) retry transient HTTP failures
    # 2) retry without video when request is too large / malformed
    # 3) fallback to a stable model if requested model is unavailable
    model_candidates = _ordered_gen3_model_candidates(
        gem_cfg,
        selected_model,
        "compare_fallback_model",
        "triplet_compare_fallback_model",
        "triplet_fallback_model",
    )

    run_notes: list[str] = []
    attempt_errors: list[str] = []
    result: Optional[Dict[str, Any]] = None
    used_model = selected_model

    for model_name in model_candidates:
        use_video = True
        for attempt in range(1, retry_attempts + 1):
            try:
                current_video_a = video_main if use_video else None
                current_video_b = video_limit if use_video else None
                result = _call_gemini_compare(
                    cfg=cfg_for_call,
                    dotenv=dotenv,
                    model=model_name,
                    prompt=prompt,
                    video_a=current_video_a,
                    video_b=current_video_b,
                    cache_dir=cache_root / "video_inline",
                    episode_id=str(episode_id or "").strip() or _infer_episode_id_from_paths(current_video_a, current_video_b),
                    response_schema=compare_response_schema,
                    usage_mode="triplet_compare",
                )
                
                # HYBRID REFINEMENT: If initial pass is API-based AND refinement is enabled,
                # perform a second pass via chat_web to reach "100% quality".
                refine_enabled = bool(gem_cfg.get("hybrid_chat_refine_enabled", False))
                if result and refine_enabled and compare_auth_mode != "chat_web":
                    draft_json = result.get("parsed", {})
                    # Only refine if we have some valid text to refine.
                    winner_key = str(draft_json.get("winner") or "").strip().lower()
                    draft_text = ""
                    if winner_key == "api":
                        draft_text = api_text
                    elif winner_key == "chat":
                        draft_text = chat_text
                    elif winner_key == "vertex_chat":
                        draft_text = vertex_chat_text
                    
                    if draft_text:
                        refine_prompt = _build_hybrid_refine_prompt(
                            original_prompt=prompt,
                            draft_text=draft_text,
                        )
                        print(f"[triplet-compare] info episode={episode_id} triggering_hybrid_refine_pass")
                        refine_result = _call_gemini_compare_chat_web(
                            cfg=cfg_for_call,
                            prompt=refine_prompt,
                            video_a=current_video_a,
                            video_b=current_video_b,
                            episode_id=str(episode_id or "").strip() or _infer_episode_id_from_paths(current_video_a, current_video_b),
                            response_schema=compare_response_schema,
                        )
                        if refine_result and refine_result.get("parsed"):
                            # Update result with refined version
                            result["parsed"] = refine_result["parsed"]
                            result["raw_text"] = refine_result.get("raw_text", "")
                            notes = result.get("attach_notes", [])
                            result["attach_notes"] = [*notes, "hybrid_chat_refine_success"]

                used_model = model_name
                if not use_video:
                    run_notes.append("retried_without_video")
                break
            except Exception as exc:
                msg = str(exc)
                low = msg.lower()
                attempt_errors.append(f"{model_name}#{attempt}: {msg}")
                is_transient = any(t in low for t in ("http 429", "http 500", "http 502", "http 503", "http 504", "timeout", "timed out"))
                is_size_or_payload = any(
                    t in low
                    for t in (
                        "http 400",
                        "http 413",
                        "payload",
                        "request too large",
                        "request entity too large",
                        "inline_data",
                        "inlinedata",
                        "content size",
                    )
                )
                is_model_issue = ("http 404" in low) or (
                    "model" in low and any(t in low for t in ("not found", "unsupported", "unavailable"))
                )

                if use_video and is_size_or_payload:
                    use_video = False
                    run_notes.append(f"retry_no_video_after_error: {msg[:160]}")
                    continue

                if is_transient and attempt < retry_attempts:
                    sleep_sec = min(20, 2 ** attempt)
                    run_notes.append(f"transient_retry_{attempt}_sleep_{sleep_sec}s")
                    time.sleep(sleep_sec)
                    continue

                if is_model_issue:
                    run_notes.append(f"model_issue_on_{model_name}")
                break
        if result is not None:
            break

    if result is None:
        tail = " | ".join(attempt_errors[-4:]) if attempt_errors else "unknown compare failure"
        raise RuntimeError(tail)

    if run_notes:
        existing = result.get("attach_notes", [])
        if not isinstance(existing, list):
            existing = []
        result["attach_notes"] = [*existing, *run_notes]
    judge_valid = _validate_triplet_judge_result(
        result.get("parsed", {}),
        require_thought_process=include_thought_process,
    )
    if compare_fail_on_none and str(judge_valid.get("winner") or "") == "none":
        raise RuntimeError("judge_result winner=none and compare_fail_on_none=true")

    out_path = Path(out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": used_model,
        "video_refs": {
            "video_path": video_path,
            "video_path_limit": video_path_limit,
            "resolved_video_path": str(video_main),
            "resolved_video_path_limit": str(video_limit) if video_limit else "",
        },
        "text_refs": {
            "tier2_path": tier2_path,
            "api_path": api_path,
            "chat_path": chat_path,
            "vertex_chat_path": vertex_chat_path,
            "labels_path": labels_path,
            "task_state_path": task_state_path,
            "resolved_tier2_path": str(tier2_file),
            "resolved_api_path": str(api_file),
            "resolved_chat_path": str(chat_file) if chat_file else "",
            "resolved_vertex_chat_path": str(vertex_chat_file) if vertex_chat_file else "",
            "resolved_task_state_path": str(task_state_file) if task_state_file else "",
        },
        "attach_notes": result.get("attach_notes", []),
        "judge_result": judge_valid,
        "judge_raw_text": result.get("raw_text", ""),
        "usage": result.get("usage", {}),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output_path"] = str(out_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="4-way compare: Tier2 vs Gemini API vs Gemini Chat vs Vertex Chat")
    parser.add_argument("--config", default="sample_web_auto_solver.yaml")
    parser.add_argument("--video-path", required=True, help="Local path or Drive folder-link+filename reference")
    parser.add_argument("--video-path-limit", default="", help="Second video path (optimized)")
    parser.add_argument("--tier2-path", required=True, help="Tier2 text/json path reference")
    parser.add_argument("--api-path", required=True, help="Gemini API output text/json path reference")
    parser.add_argument("--chat-path", default="", help="Gemini Chat output text/json path reference")
    parser.add_argument("--vertex-chat-path", default="", help="Vertex Chat output text/json path reference")
    parser.add_argument("--task-state-path", default="", help="Optional task_state JSON reference")
    parser.add_argument("--labels-path", default="", help="Optional labels JSON reference (used as chat fallback)")
    parser.add_argument("--remote", default=os.environ.get("RCLONE_REMOTE", "gdrive"))
    parser.add_argument("--cache-dir", default="tmp/triplet_compare_cache")
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--out", default="outputs/triplet_compare_result.json")
    args = parser.parse_args()

    payload = run_triplet_compare(
        config_path=args.config,
        video_path=args.video_path,
        video_path_limit=args.video_path_limit,
        tier2_path=args.tier2_path,
        api_path=args.api_path,
        chat_path=args.chat_path,
        vertex_chat_path=args.vertex_chat_path,
        task_state_path=args.task_state_path,
        labels_path=args.labels_path,
        remote=args.remote,
        cache_dir=args.cache_dir,
        model=args.model,
        episode_id=args.video_path,
    )

    judge = payload.get("judge_result", {})
    winner = ""
    if isinstance(judge, dict):
        winner = str(judge.get("winner", "") or "").strip()
    print(f"[triplet-compare] winner: {winner or 'unknown'}")
    print(f"[triplet-compare] output: {payload.get('output_path', '')}")


def _call_gemini_chat_web_on_page(
    cfg: Dict[str, Any],
    prompt: str,
    page_override: Any,
    video_a: Optional[Path] = None,
    video_b: Optional[Path] = None,
    episode_id: str = "",
) -> Dict[str, Any]:
    gem = cfg.get("gemini", {}) if isinstance(cfg.get("gemini"), dict) else {}
    chat_url = str(
        gem.get("chat_web_url", "https://gemini.google.com/app") or ""
    ).strip() or "https://gemini.google.com/app"
    timeout_sec = max(20.0, float(gem.get("chat_web_timeout_sec", 180) or 180))
    max_upload_mb = max(50.0, float(gem.get("chat_web_max_upload_mb", 2048) or 2048))
    attach_secondary = bool(gem.get("chat_web_attach_secondary_video", False))
    input_sel = str(gem.get("chat_web_input_selector", 'div[contenteditable="true"] || textarea') or "").strip()
    send_sel = str(gem.get("chat_web_send_selector", 'button[aria-label*="Send" i] || button:has-text("Send") || button:has-text("Run")') or "").strip()
    file_input_sel = str(gem.get("chat_web_file_input_selector", 'input[type="file"]') or "").strip()
    attach_button_sel = str(gem.get("chat_web_attach_button_selector", 'button[aria-label*="Open upload file menu" i] || button[aria-label*="Add files" i] || button[aria-label*="Upload" i] || button[aria-label*="Tools" i] || button:has-text("Add files") || button:has-text("Upload") || button:has-text("Tools")') or "").strip()
    upload_menu_sel = str(gem.get("chat_web_upload_menu_selector", 'button[aria-label*="Upload files" i] || [role="menuitem"]:has-text("Upload files") || button:has-text("Upload files") || [role="option"]:has-text("Upload files") || text=/^Upload files$/i') or "").strip()
    upload_settle_min_sec = max(1.0, float(gem.get("chat_web_upload_settle_min_sec", 4.0) or 4.0))
    upload_settle_sec_per_100mb = max(0.0, float(gem.get("chat_web_upload_settle_sec_per_100mb", 12.0) or 12.0))
    upload_settle_max_sec = max(upload_settle_min_sec, float(gem.get("chat_web_upload_settle_max_sec", 45.0) or 45.0))
    prefer_drive_picker = bool(gem.get("chat_web_prefer_drive_picker", False))
    drive_root_folder_url = str(gem.get("chat_web_drive_root_folder_url", "") or "").strip()
    clean_thread_fallback_enabled = bool(gem.get("chat_web_clean_thread_fallback_enabled", False))
    force_clean_thread = bool(gem.get("chat_web_force_clean_thread", False))
    allow_text_only_fallback_on_network_error = bool(
        gem.get("allow_text_only_fallback_on_network_error", False)
    )
    memory_primer_text = _load_chat_memory_primer(gem)
    seed_context_text = _load_chat_seed_context(gem)
    send_seed_context = bool(gem.get("chat_web_seed_context_send_before_prompt", False))
    json_followup_retry = bool(gem.get("chat_web_json_followup_retry", True))

    attach_candidates: List[Path] = []
    if video_a is not None and video_a.exists():
        attach_candidates.append(video_a)
    if attach_secondary and video_b is not None and video_b.exists():
        attach_candidates.append(video_b)

    raw_text = ""

    chat_page = page_override.context.new_page()
    try:
        chat_page.goto(chat_url, wait_until="domcontentloaded", timeout=60000)
        try:
            chat_page.wait_for_timeout(6000)
        except Exception:
            pass
        _handle_gemini_consent_if_present(chat_page)
        chat_box = _first_visible_locator(chat_page, input_sel, timeout_ms=30000)
        if chat_box is None:
            raise RuntimeError("Gemini chat input not visible on new tab.")

        attach_notes: List[str] = []
        if prefer_drive_picker and attach_candidates:
            import threading
            def _background_stage_override():
                try:
                    _stage_episode_artifacts_for_drive_picker(
                        cfg,
                        episode_id=episode_id,
                        paths=attach_candidates,
                    )
                except Exception as e:
                    print(f"[chat_web] async drive stage error: {e}")
            
            threading.Thread(target=_background_stage_override, daemon=True).start()
            attach_notes.append("drive_stage_started_async")
            print("[chat_web] Deferred Drive upload to background thread (page override).")
            
            # Turn off prefer_drive_picker for the rest of this workflow because
            # the file won't be immediately available for the UI picker.
            prefer_drive_picker = False
        if force_clean_thread:
            chat_box, clean_notes = _prepare_clean_chat_thread(
                page=chat_page,
                input_selector=input_sel,
                send_selector=send_sel,
                timeout_sec=timeout_sec,
                memory_primer_text=memory_primer_text,
            )
            attach_notes.extend(clean_notes)
        if send_seed_context:
            chat_box, seed_notes = _seed_chat_thread(
                page=chat_page,
                chat_box=chat_box,
                input_selector=input_sel,
                send_selector=send_sel,
                timeout_sec=timeout_sec,
                seed_context_text=seed_context_text,
            )
            attach_notes.extend(seed_notes)

        attach_notes.extend(_attach_files_via_chat_ui(
            page=chat_page,
            composer_locator=chat_box,
            attach_candidates=attach_candidates,
            episode_id=episode_id,
            prefer_drive_picker=prefer_drive_picker,
            drive_root_folder_url=drive_root_folder_url,
            max_upload_mb=max_upload_mb,
            attach_button_sel=attach_button_sel,
            upload_menu_sel=upload_menu_sel,
            file_input_sel=file_input_sel,
            upload_settle_min_sec=upload_settle_min_sec,
            upload_settle_sec_per_100mb=upload_settle_sec_per_100mb,
            upload_settle_max_sec=upload_settle_max_sec,
        ))
        attached_any = any("attached" in str(note or "").lower() for note in (attach_notes or []))
        if not attached_any and clean_thread_fallback_enabled:
            chat_box, clean_notes = _prepare_clean_chat_thread(
                page=chat_page,
                input_selector=input_sel,
                send_selector=send_sel,
                timeout_sec=timeout_sec,
                memory_primer_text=memory_primer_text,
            )
            attach_notes.extend(clean_notes)
            if send_seed_context:
                chat_box, seed_notes = _seed_chat_thread(
                    page=chat_page,
                    chat_box=chat_box,
                    input_selector=input_sel,
                    send_selector=send_sel,
                    timeout_sec=timeout_sec,
                    seed_context_text=seed_context_text,
                )
                attach_notes.extend(seed_notes)
            attach_notes.extend(
                _attach_files_via_chat_ui(
                    page=chat_page,
                    composer_locator=chat_box,
                    attach_candidates=attach_candidates,
                    episode_id=episode_id,
                    prefer_drive_picker=prefer_drive_picker,
                    drive_root_folder_url=drive_root_folder_url,
                    max_upload_mb=max_upload_mb,
                    attach_button_sel=attach_button_sel,
                    upload_menu_sel=upload_menu_sel,
                    file_input_sel=file_input_sel,
                    upload_settle_min_sec=upload_settle_min_sec,
                    upload_settle_sec_per_100mb=upload_settle_sec_per_100mb,
                    upload_settle_max_sec=upload_settle_max_sec,
                )
            )
            attached_any = any("attached" in str(note or "").lower() for note in (attach_notes or []))
        if attach_candidates and not attached_any and not allow_text_only_fallback_on_network_error:
            raise RuntimeError(
                "Gemini chat video attachment failed; refusing to send prompt without video context."
            )

        baseline_state = _capture_chat_response_state(chat_page)
        baseline_candidates = list(baseline_state.get("texts", []) or [])
        baseline_text = str(baseline_state.get("latest_text", "") or "")
        _send_chat_prompt(page=chat_page, chat_box=chat_box, send_selector=send_sel, prompt_text=prompt)

        raw_text = _wait_for_new_chat_response_text(
            chat_page,
            baseline_text=baseline_text,
            baseline_candidates=baseline_candidates,
            baseline_state=baseline_state,
            timeout_sec=timeout_sec,
        )
        if not raw_text:
            raise RuntimeError("Timed out waiting for Gemini chat response.")

        import json
        parsed_preview: Any = None
        try:
            parsed_preview = json.loads(_clean_json_text(raw_text))
        except Exception:
            parsed_preview = None

        if json_followup_retry and not _chat_response_has_required_fields(parsed_preview, None):
            followup = "Rewrite your last answer as strict JSON only with no markdown and no prose. Return exactly one JSON object."
            baseline_retry_state = _capture_chat_response_state(chat_page)
            baseline_retry_candidates = list(baseline_retry_state.get("texts", []) or [])
            baseline_retry = str(baseline_retry_state.get("latest_text", "") or "")
            _send_chat_prompt(page=chat_page, chat_box=chat_box, send_selector=send_sel, prompt_text=followup)
            retry_text = _wait_for_new_chat_response_text(
                chat_page,
                baseline_text=baseline_retry,
                baseline_candidates=baseline_retry_candidates,
                baseline_state=baseline_retry_state,
                timeout_sec=max(25.0, timeout_sec * 0.6),
            )
            if retry_text:
                raw_text = retry_text
                attach_notes.append("json_followup_retry_used")
    finally:
        try:
            chat_page.close()
        except Exception:
            pass

    try:
        parsed = json.loads(_clean_json_text(raw_text))
    except Exception:
        parsed = {"raw_text": raw_text}
    return {
        "parsed": parsed,
        "raw_text": raw_text,
        "attach_notes": attach_notes,
    }

if __name__ == "__main__":
    main()
