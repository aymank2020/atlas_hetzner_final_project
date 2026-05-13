"""Video acquisition and preprocessing helpers extracted from the legacy solver."""

from __future__ import annotations

import html
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from playwright.sync_api import Page

from src.infra.artifacts import _ensure_parent
from src.infra.logging_utils import build_print_logger as _build_print_logger
from src.infra.solver_config import _cfg_get
from src.solver import browser as _browser
from src.solver import video_core as _video_core

_logger = logging.getLogger(__name__)
print = _build_print_logger(_logger)

_probe_video_stream_meta = _video_core._probe_video_stream_meta
_quality_preserving_scale_candidates = _video_core._quality_preserving_scale_candidates
_extract_reference_frame_inline_parts = _video_core._extract_reference_frame_inline_parts
_ensure_even = _video_core._ensure_even
_parse_float_list = _video_core._parse_float_list
_opencv_available = _video_core._opencv_available
_resolve_ffmpeg_binary = _video_core._resolve_ffmpeg_binary
_resolve_ffprobe_binary = _video_core._resolve_ffprobe_binary
_probe_video_duration_seconds = _video_core._probe_video_duration_seconds
_split_video_for_upload = _video_core._split_video_for_upload
_segment_chunks = _video_core._segment_chunks
_extract_video_window = _video_core._extract_video_window
_transcode_video_ffmpeg = _video_core._transcode_video_ffmpeg
_transcode_video_cv2 = _video_core._transcode_video_cv2
_maybe_optimize_video_for_upload = _video_core._maybe_optimize_video_for_upload


def _dismiss_blocking_modals(page: Page, cfg: Optional[Dict[str, Any]] = None) -> None:
    from importlib import import_module

    legacy = import_module("src.solver.legacy_impl")
    legacy._dismiss_blocking_modals(page, cfg)


def _looks_like_video_url(url: str) -> bool:
    return _video_core._looks_like_video_url(url)


def _collect_video_url_candidates(page: Page, cfg: Dict[str, Any]) -> List[str]:
    selectors = _cfg_get(cfg, "atlas.selectors", {})
    video_sel = str(selectors.get("video_element", "video"))
    source_sel = str(selectors.get("video_source", "video source"))
    base_url = page.url

    seen: set[str] = set()
    out: List[str] = []

    def add(raw: str) -> None:
        raw = html.unescape((raw or "").strip())
        if not raw:
            return
        if raw.startswith("blob:"):
            return
        if raw.startswith("//"):
            raw = f"https:{raw}"
        elif raw.startswith("/"):
            raw = urljoin(base_url, raw)
        parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"}:
            return
        normalized = parsed._replace(fragment="").geturl().strip()
        if not _looks_like_video_url(normalized):
            return
        if normalized in seen:
            return
        seen.add(normalized)
        out.append(normalized)

    for sel in _browser._selector_variants(video_sel):
        try:
            loc = page.locator(sel)
            for i in range(min(loc.count(), 6)):
                item = loc.nth(i)
                add(item.get_attribute("src") or "")
                add(item.get_attribute("data-src") or "")
                try:
                    cur = item.evaluate("el => (el.currentSrc || '')")
                    if isinstance(cur, str):
                        add(cur)
                except Exception:
                    pass
        except Exception:
            continue

    for sel in _browser._selector_variants(source_sel):
        try:
            loc = page.locator(sel)
            for i in range(min(loc.count(), 10)):
                item = loc.nth(i)
                add(item.get_attribute("src") or "")
        except Exception:
            continue

    try:
        html_doc = page.content()
    except Exception:
        html_doc = ""
    for match in re.findall(r'["\']([^"\']+\.(?:mp4|webm|mov|m4v|m3u8)(?:\?[^"\']*)?)["\']', html_doc, flags=re.I):
        add(match)
    for match in re.findall(r'https?://[^\s"\'<>]+', html_doc):
        if _looks_like_video_url(match):
            add(match)

    try:
        entries = page.evaluate("() => performance.getEntriesByType('resource').map(e => e.name)")
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, str) and _looks_like_video_url(item):
                    add(item)
    except Exception:
        pass

    out.sort(
        key=lambda url: (
            0 if re.search(r"\.(mp4|webm|mov|m4v|m3u8)(\?|$)", url, flags=re.I) else 1,
            0 if "atlascapture" in url.lower() else 1,
            len(url),
        )
    )
    return out


def _download_video_via_playwright_request(
    page: Page,
    context: Any,
    video_url: str,
    out_path: Path,
    timeout_sec: int,
) -> Path:
    headers = {
        "Accept": "*/*",
        "Referer": page.url,
    }
    try:
        ua = page.evaluate("() => navigator.userAgent")
        if isinstance(ua, str) and ua.strip():
            headers["User-Agent"] = ua.strip()
    except Exception:
        pass

    req_ctx = getattr(context, "request", None)
    if req_ctx is None:
        raise RuntimeError("playwright request context is unavailable")

    resp = req_ctx.get(
        video_url,
        headers=headers,
        timeout=max(15000, int(timeout_sec * 1000)),
        fail_on_status_code=False,
    )
    status = int(resp.status)
    if status not in {200, 206}:
        raise RuntimeError(f"playwright fallback status={status}")

    body = resp.body() or b""
    if not body:
        raise RuntimeError("playwright fallback returned empty body")

    if status == 206:
        content_range = str((resp.headers or {}).get("content-range", "")).strip()
        match = re.search(r"/(\d+)$", content_range)
        if match:
            try:
                total = int(match.group(1))
            except Exception:
                total = 0
            if total > 0 and len(body) < total:
                raise RuntimeError(
                    f"playwright fallback returned partial body ({len(body)}/{total})"
                )

    _ensure_parent(out_path)
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    part_path.write_bytes(body)
    try:
        out_path.unlink(missing_ok=True)
    except Exception:
        pass
    part_path.replace(out_path)
    return out_path


def _download_video_from_page_context(
    page: Page,
    context: Any,
    video_url: str,
    out_path: Path,
    timeout_sec: int,
    cfg: Optional[Dict[str, Any]] = None,
) -> Path:
    sess = requests.Session()
    headers = {
        "Accept": "*/*",
        "Referer": page.url,
    }
    try:
        ua = page.evaluate("() => navigator.userAgent")
        if isinstance(ua, str) and ua.strip():
            headers["User-Agent"] = ua.strip()
    except Exception:
        pass

    try:
        cookies = context.cookies([video_url]) or context.cookies()
    except Exception:
        cookies = []
    for cookie in cookies:
        try:
            sess.cookies.set(
                cookie.get("name", ""),
                cookie.get("value", ""),
                domain=cookie.get("domain"),
                path=cookie.get("path", "/"),
            )
        except Exception:
            continue

    _ensure_parent(out_path)
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    max_retries = max(0, int(_cfg_get(cfg or {}, "gemini.video_download_retries", 5)))
    chunk_bytes = max(64 * 1024, int(_cfg_get(cfg or {}, "gemini.video_download_chunk_bytes", 1024 * 1024)))
    retry_base = max(0.2, float(_cfg_get(cfg or {}, "gemini.video_download_retry_base_sec", 1.2)))
    stalled_partial_retry_limit = max(
        1,
        int(_cfg_get(cfg or {}, "gemini.video_download_stalled_partial_retry_limit", 2)),
    )
    use_playwright_fallback = bool(
        _cfg_get(cfg or {}, "gemini.video_download_use_playwright_fallback", True)
    )
    last_err: Optional[Exception] = None
    stalled_partial_attempts = 0

    def _content_range_total(content_range: str) -> int:
        match = re.search(r"/(\d+)$", content_range or "")
        if not match:
            return 0
        try:
            return int(match.group(1))
        except Exception:
            return 0

    for attempt in range(max_retries + 1):
        resume_from = 0
        try:
            if part_path.exists():
                resume_from = int(part_path.stat().st_size)
        except Exception:
            resume_from = 0

        req_headers = dict(headers)
        if resume_from > 0:
            req_headers["Range"] = f"bytes={resume_from}-"

        try:
            with sess.get(
                video_url,
                headers=req_headers,
                timeout=(20, timeout_sec),
                stream=True,
                allow_redirects=True,
            ) as resp:
                status = int(resp.status_code)
                if status not in {200, 206}:
                    resp.raise_for_status()

                if resume_from > 0 and status == 200:
                    try:
                        part_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    resume_from = 0

                expected_total = 0
                content_range = resp.headers.get("Content-Range", "")
                if content_range:
                    expected_total = _content_range_total(content_range)
                if expected_total <= 0:
                    content_length = resp.headers.get("Content-Length", "")
                    try:
                        content_len = int(content_length)
                    except Exception:
                        content_len = 0
                    if content_len > 0:
                        expected_total = resume_from + content_len

                mode = "ab" if (resume_from > 0 and status == 206 and part_path.exists()) else "wb"
                with part_path.open(mode) as f:
                    for chunk in resp.iter_content(chunk_size=chunk_bytes):
                        if not chunk:
                            continue
                        f.write(chunk)

                current_size = int(part_path.stat().st_size) if part_path.exists() else 0
                if current_size <= 0:
                    raise RuntimeError("Downloaded video file is empty.")
                if expected_total > 0 and current_size < expected_total:
                    raise RuntimeError(f"Incomplete download ({current_size}/{expected_total} bytes)")

                try:
                    out_path.unlink(missing_ok=True)
                except Exception:
                    pass
                part_path.replace(out_path)
                return out_path
        except Exception as exc:
            last_err = exc
            try:
                partial = int(part_path.stat().st_size) if part_path.exists() else 0
            except Exception:
                partial = 0
            made_progress = partial > resume_from
            if partial > 0 and not made_progress:
                stalled_partial_attempts += 1
            elif partial > 0:
                stalled_partial_attempts = 0
            else:
                stalled_partial_attempts = 0
            if (
                use_playwright_fallback
                and partial > 0
                and stalled_partial_attempts >= stalled_partial_retry_limit
            ):
                print(
                    "[video] download stalled with no partial progress "
                    f"({partial} bytes); switching to playwright fallback."
                )
                break
            if attempt < max_retries:
                delay = retry_base * (2**attempt)
                print(
                    f"[video] download retry {attempt + 1}/{max_retries} "
                    f"(partial={partial} bytes) in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            break

    if use_playwright_fallback:
        try:
            return _download_video_via_playwright_request(
                page=page,
                context=context,
                video_url=video_url,
                out_path=out_path,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:
            if last_err is not None:
                raise RuntimeError(f"{last_err}; playwright fallback failed: {exc}") from exc
            raise RuntimeError(f"playwright fallback failed: {exc}") from exc

    raise RuntimeError(str(last_err) if last_err else "video download failed")


def _is_probably_mp4(path: Path) -> bool:
    return _video_core._is_probably_mp4(path)


def _is_video_decodable(path: Path) -> bool:
    return _video_core._is_video_decodable(path)


def _ensure_loop_off(page: Page, cfg: Dict[str, Any]) -> None:
    loop_sel = str(_cfg_get(cfg, "atlas.selectors.loop_toggle_button", "")).strip()
    if loop_sel:
        loop_loc = _browser._first_visible_locator(page, loop_sel, timeout_ms=2200)
        if loop_loc is not None:
            try:
                txt = (_browser._safe_locator_text(loop_loc, timeout_ms=700) or "").lower()
                title = (loop_loc.get_attribute("title") or "").lower()
                classes = (loop_loc.get_attribute("class") or "").lower()
                aria_pressed = (loop_loc.get_attribute("aria-pressed") or "").lower()
                should_toggle = False
                if "loop on" in txt:
                    should_toggle = True
                elif "toggle segment loop" in title and ("bg-primary" in classes or aria_pressed == "true"):
                    should_toggle = True
                if should_toggle:
                    loop_loc.click()
                    print("[video] loop toggled OFF.")
            except Exception:
                pass
    try:
        page.evaluate(
            """() => {
                const v = document.querySelector('video');
                if (v) v.loop = false;
            }"""
        )
    except Exception:
        pass


def _play_full_video_once(page: Page, cfg: Dict[str, Any]) -> None:
    if not bool(_cfg_get(cfg, "run.play_full_video_before_labeling", False)):
        return
    max_wait_sec = max(10, int(_cfg_get(cfg, "run.play_full_video_max_wait_sec", 900)))
    _ensure_loop_off(page, cfg)
    try:
        state = page.evaluate(
            """() => {
                const v = document.querySelector('video');
                if (!v) return null;
                try { v.loop = false; } catch (e) {}
                try { v.muted = true; } catch (e) {}
                try { v.playbackRate = 1; } catch (e) {}
                try { v.play(); } catch (e) {}
                return {
                    current: Number(v.currentTime || 0),
                    duration: Number(v.duration || 0)
                };
            }"""
        )
    except Exception:
        state = None
    if not state:
        print("[video] video element not found; skipping full-video playback step.")
        return

    current = float(state.get("current", 0) or 0)
    duration = float(state.get("duration", 0) or 0)
    if duration <= 0:
        print("[video] unknown duration; skipping full-video playback wait.")
        return

    wait_budget = min(max_wait_sec, max(5, int(duration - current + 3)))
    print(f"[video] playing video to end (duration={duration:.1f}s, wait_budget={wait_budget}s).")
    start = time.time()
    last_log = -999.0
    while time.time() - start < wait_budget:
        try:
            state = page.evaluate(
                """() => {
                    const v = document.querySelector('video');
                    if (!v) return null;
                    return {
                        ended: !!v.ended,
                        current: Number(v.currentTime || 0),
                        duration: Number(v.duration || 0)
                    };
                }"""
            )
        except Exception:
            state = None
        if not state:
            break
        current = float(state.get("current", 0) or 0)
        duration = float(state.get("duration", 0) or 0)
        ended = bool(state.get("ended", False))
        if ended or (duration > 0 and current >= duration - 0.2):
            print(f"[video] playback reached end at {current:.1f}/{duration:.1f}s.")
            break
        elapsed = time.time() - start
        if elapsed - last_log >= 15:
            last_log = elapsed
            print(f"[video] playback progress: {current:.1f}/{duration:.1f}s")
        page.wait_for_timeout(1000)
    try:
        page.evaluate(
            """() => {
                const v = document.querySelector('video');
                if (v) v.pause();
            }"""
        )
    except Exception:
        pass
    _ensure_loop_off(page, cfg)


def _prepare_video_for_gemini(
    page: Page,
    context: Any,
    cfg: Dict[str, Any],
    task_id: str = "",
) -> Optional[Path]:
    attach_video = bool(_cfg_get(cfg, "gemini.attach_video", True))
    if not attach_video:
        return None

    out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    use_task_scoped = bool(_cfg_get(cfg, "run.use_task_scoped_artifacts", True))
    if use_task_scoped and task_id:
        video_name = f"video_{task_id}.mp4"
    else:
        video_name = str(_cfg_get(cfg, "run.video_dump", "atlas_task_video.mp4"))
    timeout_sec = int(_cfg_get(cfg, "gemini.video_download_timeout_sec", 180))
    require_video = bool(_cfg_get(cfg, "gemini.require_video", False))
    min_video_bytes = int(_cfg_get(cfg, "gemini.min_video_bytes", 500000))
    validate_video_decode = bool(_cfg_get(cfg, "gemini.validate_video_decode", True))
    scan_attempts = max(1, int(_cfg_get(cfg, "gemini.video_candidate_scan_attempts", 4)))
    scan_wait_ms = max(200, int(_cfg_get(cfg, "gemini.video_candidate_scan_wait_ms", 1200)))
    resume_from_artifacts = bool(_cfg_get(cfg, "run.resume_from_artifacts", True))

    primary_target = out_dir / video_name
    if resume_from_artifacts and primary_target.exists():
        try:
            size_bytes = primary_target.stat().st_size
            if size_bytes >= min_video_bytes and _is_probably_mp4(primary_target):
                if validate_video_decode and not _is_video_decodable(primary_target):
                    print(f"[video] cached file looks corrupted; re-downloading: {primary_target}")
                    try:
                        primary_target.unlink(missing_ok=True)
                    except Exception:
                        pass
                else:
                    size_mb = size_bytes / (1024 * 1024)
                    print(f"[video] reusing existing file: {primary_target} ({size_mb:.1f} MB)")
                    return primary_target
        except Exception:
            pass

    def _nudge_video_network() -> None:
        try:
            page.evaluate(
                """() => {
                    const v = document.querySelector('video');
                    if (!v) return;
                    try { v.muted = true; v.play(); } catch (e) {}
                }"""
            )
            page.wait_for_timeout(900)
            page.evaluate(
                """() => {
                    const v = document.querySelector('video');
                    if (!v) return;
                    try {
                        if (Number.isFinite(v.currentTime)) {
                            v.currentTime = Math.max(0, Number(v.currentTime || 0) + 0.05);
                        }
                        v.pause();
                    } catch (e) {}
                }"""
            )
        except Exception:
            pass

    network_seen: set[str] = set()
    network_candidates: List[str] = []

    def _remember_network_video_url(raw_url: str, content_type: str = "") -> None:
        try:
            raw = html.unescape((raw_url or "").strip())
            if not raw:
                return
            low_ct = (content_type or "").lower()
            if "video" not in low_ct and not _looks_like_video_url(raw):
                return
            if raw in network_seen:
                return
            network_seen.add(raw)
            network_candidates.append(raw)
        except Exception:
            return

    response_listener = None
    try:
        def _on_response(resp: Any) -> None:
            try:
                headers = resp.headers or {}
            except Exception:
                headers = {}
            try:
                content_type = str(headers.get("content-type", "") or "")
            except Exception:
                content_type = ""
            try:
                _remember_network_video_url(str(resp.url or ""), content_type=content_type)
            except Exception:
                return

        response_listener = _on_response
        page.on("response", response_listener)
    except Exception:
        response_listener = None

    def _rank_video_url(url: str) -> tuple[int, int, int]:
        low = (url or "").lower()
        return (
            0 if re.search(r"\.(mp4|webm|mov|m4v|m3u8)(\?|$)", low, flags=re.I) else 1,
            0 if "atlascapture" in low or "cloudflarestorage.com" in low else 1,
            len(url or ""),
        )

    page.wait_for_timeout(1500)
    _dismiss_blocking_modals(page, cfg)
    candidates: List[str] = []
    for scan_idx in range(scan_attempts):
        if scan_idx > 0:
            page.wait_for_timeout(scan_wait_ms)
            _dismiss_blocking_modals(page, cfg)
        _nudge_video_network()
        candidates = _collect_video_url_candidates(page, cfg)
        for from_net in network_candidates:
            if from_net not in candidates:
                candidates.append(from_net)
        candidates.sort(key=_rank_video_url)
        if candidates:
            if scan_idx > 0:
                print(f"[video] candidate urls resolved after retry {scan_idx + 1}/{scan_attempts}.")
            break
        if scan_idx < scan_attempts - 1:
            print(f"[video] no candidate urls yet ({scan_idx + 1}/{scan_attempts}); retrying...")

    if response_listener is not None:
        try:
            page.remove_listener("response", response_listener)
        except Exception:
            pass

    print(f"[video] candidate urls found: {len(candidates)}")
    for url in candidates[:5]:
        print(f"[video] candidate: {url}")

    last_err: Optional[Exception] = None
    for idx, url in enumerate(candidates[:20], start=1):
        target = primary_target
        if idx > 1:
            stem = Path(video_name).stem
            suffix = Path(video_name).suffix or ".mp4"
            target = out_dir / f"{stem}_{idx}{suffix}"
        try:
            _download_video_from_page_context(
                page=page,
                context=context,
                video_url=url,
                out_path=target,
                timeout_sec=timeout_sec,
                cfg=cfg,
            )
            size_bytes = target.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            if size_bytes < min_video_bytes:
                print(f"[video] skip candidate (too small {size_bytes} bytes): {url}")
                continue
            if not _is_probably_mp4(target):
                print(f"[video] skip candidate (not mp4 signature): {url}")
                continue
            if validate_video_decode and not _is_video_decodable(target):
                print(f"[video] skip candidate (decode check failed): {url}")
                continue
            print(f"[video] downloaded: {target} ({size_mb:.1f} MB)")
            return target
        except Exception as exc:
            last_err = exc
            continue

    if require_video:
        if last_err is None:
            last_err = RuntimeError("no candidate video URLs discovered")
        raise RuntimeError(f"Could not download task video from page. Last error: {last_err}")
    print("[video] no downloadable video found; proceeding with text-only prompt.")
    return None


__all__ = [
    "_looks_like_video_url",
    "_collect_video_url_candidates",
    "_download_video_via_playwright_request",
    "_download_video_from_page_context",
    "_is_probably_mp4",
    "_is_video_decodable",
    "_probe_video_stream_meta",
    "_quality_preserving_scale_candidates",
    "_extract_reference_frame_inline_parts",
    "_ensure_even",
    "_parse_float_list",
    "_opencv_available",
    "_resolve_ffmpeg_binary",
    "_resolve_ffprobe_binary",
    "_probe_video_duration_seconds",
    "_split_video_for_upload",
    "_segment_chunks",
    "_extract_video_window",
    "_transcode_video_ffmpeg",
    "_transcode_video_cv2",
    "_maybe_optimize_video_for_upload",
    "_ensure_loop_off",
    "_prepare_video_for_gemini",
    "_play_full_video_once",
]
