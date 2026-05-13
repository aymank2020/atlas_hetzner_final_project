"""Artifact and cache helpers extracted from the legacy solver."""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from playwright.sync_api import Page

from src.infra.solver_config import _cfg_get


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _task_id_from_url(url: str) -> str:
    match = re.search(r"/tasks/room/normal/label/([A-Za-z0-9]+)", url or "")
    return match.group(1) if match else ""


def _task_scoped_artifact_paths(cfg: Dict[str, Any], task_id: str) -> Dict[str, Path]:
    out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    task = (task_id or "").strip() or "unknown_task"
    return {
        "video": out_dir / f"video_{task}.mp4",
        "text_current": out_dir / f"text_{task}_current.txt",
        "text_update": out_dir / f"text_{task}_update.txt",
        "segments_cache": out_dir / f"segments_{task}.json",
        "labels_dump": out_dir / f"labels_{task}.json",
        "prompt_dump": out_dir / f"prompt_{task}.txt",
        "state": out_dir / f"task_state_{task}.json",
    }


def _load_task_state(cfg: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    if not task_id:
        return {}
    state_path = _task_scoped_artifact_paths(cfg, task_id)["state"]
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_task_state(cfg: Dict[str, Any], task_id: str, state: Dict[str, Any]) -> None:
    if not task_id:
        return
    state_path = _task_scoped_artifact_paths(cfg, task_id)["state"]
    try:
        _ensure_parent(state_path)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _load_cached_segments(cfg: Dict[str, Any], task_id: str) -> Optional[List[Dict[str, Any]]]:
    if not task_id:
        return None
    seg_path = _task_scoped_artifact_paths(cfg, task_id)["segments_cache"]
    if not seg_path.exists():
        return None
    try:
        data = json.loads(seg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        return data["segments"]
    if isinstance(data, list):
        return data
    return None


def _save_cached_segments(cfg: Dict[str, Any], task_id: str, segments: List[Dict[str, Any]]) -> None:
    if not task_id:
        return
    seg_path = _task_scoped_artifact_paths(cfg, task_id)["segments_cache"]
    try:
        _ensure_parent(seg_path)
        seg_path.write_text(json.dumps({"segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _save_task_text_files(
    cfg: Dict[str, Any],
    task_id: str,
    segments: List[Dict[str, Any]],
    segment_plan: Dict[int, Dict[str, Any]],
) -> None:
    if not task_id:
        return
    paths = _task_scoped_artifact_paths(cfg, task_id)
    current_lines: List[str] = []
    update_lines: List[str] = []
    by_idx_src: Dict[int, Dict[str, Any]] = {}
    for seg in segments:
        try:
            idx = int(seg.get("segment_index", 0))
        except Exception:
            continue
        by_idx_src[idx] = seg
    for idx in sorted(by_idx_src):
        src = by_idx_src[idx]
        cur_label = str(src.get("current_label", "")).strip()
        cur_start = src.get("start_sec", 0.0)
        cur_end = src.get("end_sec", 0.0)
        current_lines.append(f"{idx}\t{cur_start}\t{cur_end}\t{cur_label}")
        planned = segment_plan.get(idx) or {}
        upd_label = str(planned.get("label", cur_label)).strip()
        upd_start = planned.get("start_sec", cur_start)
        upd_end = planned.get("end_sec", cur_end)
        update_lines.append(f"{idx}\t{upd_start}\t{upd_end}\t{upd_label}")
    try:
        paths["text_current"].write_text("\n".join(current_lines) + ("\n" if current_lines else ""), encoding="utf-8")
        print(f"[out] text current: {paths['text_current']}")
    except Exception:
        pass
    try:
        paths["text_update"].write_text("\n".join(update_lines) + ("\n" if update_lines else ""), encoding="utf-8")
        print(f"[out] text update: {paths['text_update']}")
    except Exception:
        pass


def _labels_cache_path(cfg: Dict[str, Any], task_id: str) -> Optional[Path]:
    if not task_id:
        return None
    out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"gemini_labels_cache_{task_id}.json"


def _load_cached_labels(cfg: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    if not bool(_cfg_get(cfg, "run.reuse_cached_labels", True)):
        return None
    cache_path = _labels_cache_path(cfg, task_id)
    if cache_path is None or not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        print(f"[gemini] using cached labels for task {task_id}: {cache_path}")
        return data
    return None


def _save_cached_labels(cfg: Dict[str, Any], task_id: str, payload: Dict[str, Any]) -> None:
    cache_path = _labels_cache_path(cfg, task_id)
    if cache_path is None:
        return
    try:
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[gemini] cached labels: {cache_path}")
    except Exception:
        pass


def _invalidate_cached_labels(cfg: Dict[str, Any], task_id: str) -> None:
    cache_path = _labels_cache_path(cfg, task_id)
    if cache_path is None or not cache_path.exists():
        return
    try:
        cache_path.unlink()
        print(f"[gemini] invalidated cached labels for task {task_id}: {cache_path}")
    except Exception:
        pass


def _clear_episode_state(
    cfg: Dict[str, Any],
    task_id: str,
    *,
    clear_task_state: bool = False,
    clear_shared_dumps: bool = False,
) -> List[str]:
    """Remove episode-scoped caches/artifacts that can leak stale state across runs."""
    task = str(task_id or "").strip()
    if not task:
        return []

    out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    scoped = _task_scoped_artifact_paths(cfg, task)

    cleared: List[str] = []
    candidates: List[Optional[Path]] = [
        _labels_cache_path(cfg, task),
        scoped.get("segments_cache"),
        scoped.get("labels_dump"),
        scoped.get("prompt_dump"),
    ]
    if clear_task_state:
        candidates.append(scoped.get("state"))

    for path in candidates:
        if path is None or not path.exists():
            continue
        try:
            path.unlink()
            cleared.append(str(path))
        except Exception:
            continue

    chat_cache_dir = out_dir / "_chat_only" / task
    if chat_cache_dir.exists():
        try:
            shutil.rmtree(chat_cache_dir)
            cleared.append(str(chat_cache_dir))
        except Exception:
            pass

    if clear_shared_dumps:
        for raw_name in (
            str(_cfg_get(cfg, "run.segments_dump", "atlas_segments_dump.json") or "").strip(),
            str(_cfg_get(cfg, "run.prompt_dump", "atlas_prompt.txt") or "").strip(),
            str(_cfg_get(cfg, "run.labels_dump", "atlas_labels_from_gemini.json") or "").strip(),
        ):
            if not raw_name:
                continue
            shared_path = out_dir / raw_name
            if not shared_path.exists():
                continue
            try:
                shared_path.unlink()
                cleared.append(str(shared_path))
            except Exception:
                continue

    return cleared


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


def _save_outputs(
    cfg: Dict[str, Any],
    segments: List[Dict[str, Any]],
    prompt: str,
    labels_payload: Dict[str, Any],
    task_id: str = "",
) -> None:
    out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_path = out_dir / str(_cfg_get(cfg, "run.segments_dump", "atlas_segments_dump.json"))
    prompt_path = out_dir / str(_cfg_get(cfg, "run.prompt_dump", "atlas_prompt.txt"))
    labels_path = out_dir / str(_cfg_get(cfg, "run.labels_dump", "atlas_labels_from_gemini.json"))

    _ensure_parent(seg_path)
    _ensure_parent(prompt_path)
    _ensure_parent(labels_path)

    seg_path.write_text(json.dumps({"segments": segments}, indent=2, ensure_ascii=False), encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")
    labels_path.write_text(json.dumps(labels_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[out] segments: {seg_path}")
    print(f"[out] prompt:   {prompt_path}")
    print(f"[out] labels:   {labels_path}")

    if task_id and bool(_cfg_get(cfg, "run.use_task_scoped_artifacts", True)):
        scoped = _task_scoped_artifact_paths(cfg, task_id)
        try:
            scoped["segments_cache"].write_text(
                json.dumps({"segments": segments}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[out] segments task: {scoped['segments_cache']}")
        except Exception:
            pass
        try:
            scoped["prompt_dump"].write_text(prompt, encoding="utf-8")
            print(f"[out] prompt task:   {scoped['prompt_dump']}")
        except Exception:
            pass
        try:
            scoped["labels_dump"].write_text(
                json.dumps(labels_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[out] labels task:   {scoped['labels_dump']}")
        except Exception:
            pass


def _capture_debug_artifacts(page: Page, cfg: Dict[str, Any], prefix: str = "debug_failure") -> Tuple[Optional[Path], Optional[Path]]:
    out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    snap_path = out_dir / f"{prefix}_{timestamp}.png"
    html_path = out_dir / f"{prefix}_{timestamp}.html"

    snap_saved: Optional[Path] = None
    html_saved: Optional[Path] = None
    try:
        page.screenshot(path=str(snap_path), full_page=True)
        snap_saved = snap_path
        print(f"[debug] screenshot saved: {snap_path}")
    except Exception:
        pass
    try:
        html_path.write_text(page.content(), encoding="utf-8")
        html_saved = html_path
        print(f"[debug] html saved: {html_path}")
    except Exception:
        pass
    return snap_saved, html_saved


def _capture_step_artifacts(
    page: Page,
    cfg: Dict[str, Any],
    task_id: str,
    step_name: str,
    *,
    include_html: Optional[bool] = None,
    full_page: Optional[bool] = None,
) -> Dict[str, str]:
    task = str(task_id or "").strip()
    step = str(step_name or "").strip()
    if not task or not step:
        return {}
    if not bool(_cfg_get(cfg, "run.capture_step_screenshots", False)):
        return {}

    out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs")))
    step_dir = out_dir / "step_screenshots" / task
    step_dir.mkdir(parents=True, exist_ok=True)
    safe_step = re.sub(r"[^A-Za-z0-9._-]+", "_", step).strip("._-") or "step"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1.0) * 1000.0)
    base_name = f"{timestamp}_{millis:03d}_{safe_step}"
    snap_path = step_dir / f"{base_name}.png"
    html_path = step_dir / f"{base_name}.html"

    save_html = bool(_cfg_get(cfg, "run.capture_step_html", False)) if include_html is None else bool(include_html)
    full_page_capture = (
        bool(_cfg_get(cfg, "run.capture_step_screenshots_full_page", False))
        if full_page is None
        else bool(full_page)
    )

    snap_saved = ""
    html_saved = ""
    try:
        page.screenshot(path=str(snap_path), full_page=full_page_capture)
        snap_saved = str(snap_path)
        print(f"[debug] step screenshot saved: {snap_path}")
    except Exception:
        pass
    if save_html:
        try:
            html_path.write_text(page.content(), encoding="utf-8")
            html_saved = str(html_path)
            print(f"[debug] step html saved: {html_path}")
        except Exception:
            pass
    if not snap_saved and not html_saved:
        return {}
    return {
        "step": step,
        "screenshot": snap_saved,
        "html": html_saved,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


__all__ = [
    "_task_id_from_url",
    "_task_scoped_artifact_paths",
    "_load_task_state",
    "_save_task_state",
    "_load_cached_segments",
    "_save_cached_segments",
    "_save_task_text_files",
    "_labels_cache_path",
    "_load_cached_labels",
    "_save_cached_labels",
    "_invalidate_cached_labels",
    "_clear_episode_state",
    "_save_validation_report",
    "_save_outputs",
    "_capture_debug_artifacts",
    "_capture_step_artifacts",
]
