"""Segment extraction and apply/submit helpers extracted from the legacy solver."""

from __future__ import annotations

import logging
import random
import re
import time
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from playwright.sync_api import Locator, Page

from src.infra import logging_utils as _logging_utils
from src.infra import runtime as _runtime
from src.infra import solver_config as _solver_config
from src.infra import utils as _utils
from src.infra.artifacts import _capture_step_artifacts, _task_id_from_url
from src.rules import labels as _labels
from src.solver import browser as _browser
from src.solver.desync import build_segment_checksum
from src.solver.reliability import ApplyBudgetState, SubmitOutcome
from src.solver import video as _video
from src.infra import submit_verify as _submit_verify

_logger = logging.getLogger(__name__)
print = _logging_utils.build_print_logger(_logger)

_cfg_get = _solver_config._cfg_get

_ensure_loop_off = _video._ensure_loop_off
_normalize_label_for_compare = _utils._normalize_label_for_compare

if TYPE_CHECKING:
    from src.solver.live_validation import ValidationTracker


def _capture_submit_step(
    page: Page,
    cfg: Dict[str, Any],
    step_name: str,
    *,
    task_id: str = "",
    include_html: bool = False,
) -> Dict[str, Any]:
    if not bool(_cfg_get(cfg, "run.capture_step_screenshots", False)):
        return {}
    step_task_id = str(task_id or "").strip() or _task_id_from_url(_safe_page_url(page)) or "_submit_flow"
    try:
        artifact = _capture_step_artifacts(
            page,
            cfg,
            step_task_id,
            str(step_name or "").strip() or "submit_step",
            include_html=include_html,
        )
        return artifact if isinstance(artifact, dict) else {}
    except Exception:
        return {}


def _page_body_text_lower(page: Page) -> str:
    try:
        return re.sub(r"\s+", " ", (page.inner_text("body") or "")).strip().lower()
    except Exception:
        return ""


def _active_extract_heartbeat_callback() -> Optional[Callable[[], None]]:
    try:
        legacy = import_module("src.solver.legacy_impl")
    except Exception:
        return None
    callback = getattr(legacy, "_ACTIVE_HEARTBEAT_CALLBACK", None)
    return callback if callable(callback) else None


def _quality_review_prompt_visible(page: Page) -> bool:
    body = _page_body_text_lower(page)
    if not body:
        return False
    markers = (
        "quality review",
        "before submitting, please confirm you've reviewed your work",
        "i verify that i have reviewed every segment",
        "every label is correct to the best of my ability",
    )
    return any(marker in body for marker in markers)


def _submit_confirmation_signal_visible(page: Page) -> bool:
    body = _page_body_text_lower(page)
    if not body:
        return False
    if "no edits made" in body and "labels are correct" in body:
        return True
    return _quality_review_prompt_visible(page)


def _force_primary_submit_click(page: Page, selector: str) -> bool:
    locator = _browser._first_visible_locator(page, selector, timeout_ms=1200)
    if locator is not None:
        for force_click in (False, True):
            try:
                locator.scroll_into_view_if_needed(timeout=800)
            except Exception:
                pass
            try:
                locator.focus(timeout=500)
            except Exception:
                pass
            try:
                locator.click(timeout=1200, force=force_click, no_wait_after=True)
                return True
            except Exception:
                pass
            try:
                locator.evaluate(
                    """(el) => {
                        const target = el.closest('button,[role="button"],a') || el;
                        const fire = (type) => {
                            const ctor = type.startsWith('pointer') ? PointerEvent : MouseEvent;
                            target.dispatchEvent(new ctor(type, {
                                bubbles: true,
                                cancelable: true,
                                composed: true,
                                view: window,
                            }));
                        };
                        try { target.focus(); } catch (err) {}
                        fire('pointerdown');
                        fire('mousedown');
                        fire('pointerup');
                        fire('mouseup');
                        fire('click');
                        return true;
                    }"""
                )
                return True
            except Exception:
                pass
            try:
                box = locator.bounding_box()
            except Exception:
                box = None
            if box and box.get("width", 0) > 0 and box.get("height", 0) > 0:
                try:
                    page.mouse.click(
                        float(box["x"]) + float(box["width"]) / 2.0,
                        float(box["y"]) + float(box["height"]) / 2.0,
                    )
                    return True
                except Exception:
                    pass
        try:
            page.keyboard.press("Enter")
            return True
        except Exception:
            pass

    try:
        return bool(
            page.evaluate(
                """(selector) => {
                    const normalize = (value) => String(value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                    const nodes = Array.from(document.querySelectorAll('button,[role="button"],a'));
                    const wanted = ['complete', 'submit', 'submit task', 'submit episode', 'finish', 'done'];
                    const isVisible = (el) => {
                        if (!el) return false;
                        const st = window.getComputedStyle(el);
                        const r = el.getBoundingClientRect();
                        return st && st.visibility !== 'hidden' && st.display !== 'none' && r.width > 0 && r.height > 0;
                    };
                    const strongClick = (node) => {
                        try { node.scrollIntoView({ block: 'center', inline: 'center' }); } catch (err) {}
                        try { node.focus(); } catch (err) {}
                        try { node.click(); return true; } catch (err) {}
                        const fire = (type) => {
                            try {
                                const ctor = type.startsWith('pointer') ? PointerEvent : MouseEvent;
                                node.dispatchEvent(new ctor(type, {
                                    bubbles: true,
                                    cancelable: true,
                                    composed: true,
                                    view: window,
                                }));
                            } catch (err) {}
                        };
                        fire('pointerdown');
                        fire('mousedown');
                        fire('pointerup');
                        fire('mouseup');
                        fire('click');
                        return true;
                    };
                    let target = null;
                    for (const node of nodes) {
                        if (!isVisible(node)) continue;
                        const disabled = !!node.disabled || normalize(node.getAttribute('aria-disabled')) === 'true';
                        if (disabled) continue;
                        if (selector && node.matches) {
                            try {
                                if (node.matches(selector)) {
                                    target = node;
                                    break;
                                }
                            } catch (err) {}
                        }
                        const text = normalize(node.innerText || node.textContent || '');
                        if (!text) continue;
                        if (wanted.some((candidate) => text === candidate || text.includes(candidate))) {
                            target = node;
                            break;
                        }
                    }
                    if (!target) return false;
                    return strongClick(target);
                }""",
                selector,
            )
        )
    except Exception:
        return False


def _parse_mmss_to_seconds(token: str) -> float:
    token = token.strip()
    if not token:
        return 0.0
    if ":" not in token:
        try:
            return float(token)
        except ValueError:
            return 0.0
    left, right = token.split(":", 1)
    try:
        return int(left) * 60 + float(right)
    except ValueError:
        return 0.0


def _extract_start_end_from_text(text: str) -> Tuple[float, float]:
    text = (text or "").replace("\u2192", "->").replace("\u2013", "-")
    matches = re.findall(r"\b\d+:\d{2}(?:\.\d+)?\b", text)
    if len(matches) >= 2:
        return _parse_mmss_to_seconds(matches[0]), _parse_mmss_to_seconds(matches[1])
    if len(matches) == 1:
        start = _parse_mmss_to_seconds(matches[0])
        dur_match = re.search(r"\((\d+(?:\.\d+)?)\s*s\)", text, flags=re.IGNORECASE)
        if dur_match:
            try:
                dur_sec = float(dur_match.group(1))
                if dur_sec > 0:
                    return start, start + dur_sec
            except ValueError:
                pass
    return 0.0, 0.0


def _resolve_rows_locator(
    page: Page,
    rows_selector: str,
    sample_size: int = 8,
    row_text_timeout_ms: int = 350,
) -> Tuple[str, Locator]:
    best_sel = ""
    best_score = -1
    best_count = 0

    for candidate in _browser._selector_variants(rows_selector):
        try:
            loc = page.locator(candidate)
            count = loc.count()
            if count <= 0:
                continue
            sample = min(count, max(1, sample_size))
            ts_hits = 0
            for i in range(sample):
                text = _browser._safe_locator_text(loc.nth(i), timeout_ms=row_text_timeout_ms)
                if re.search(r"\b\d+:\d{2}(?:\.\d+)?\b", text):
                    ts_hits += 1
            score = ts_hits * 10 + min(count, 50)
            if score > best_score:
                best_score = score
                best_count = count
                best_sel = candidate
        except Exception:
            continue

    if not best_sel:
        diagnostics: List[str] = []
        diagnostics.append("No segment rows found. Candidate selector counts:")
        for candidate in _browser._selector_variants(rows_selector):
            try:
                count = page.locator(candidate).count()
            except Exception:
                count = -1
            diagnostics.append(f"  - {candidate} => {count}")
        try:
            body = page.inner_text("body")
            body_snippet = (body or "")[:1200].replace("\n", " | ")
            diagnostics.append(f"Body snippet: {body_snippet}")
        except Exception:
            pass
        raise RuntimeError("\n".join(diagnostics))
    print(f"[atlas] using segment rows selector: {best_sel} (count={best_count})")
    return best_sel, page.locator(best_sel)


def _first_text_from_row(row: Locator, selector: str, timeout_ms: int = 350) -> str:
    for candidate in _browser._selector_variants(selector):
        try:
            text = _browser._safe_locator_text(
                row.locator(candidate).first,
                timeout_ms=max(100, int(timeout_ms or 350)),
            )
            if text:
                return text
        except Exception:
            continue
    return ""


def _resolve_row_child_selector(
    rows: Locator,
    selector: str,
    *,
    sample_size: int = 4,
    timeout_ms: int = 250,
    require_timestamp: bool = False,
) -> str:
    best_sel = ""
    best_hits = 0
    row_count = min(rows.count(), max(1, sample_size))
    for candidate in _browser._selector_variants(selector):
        hits = 0
        for i in range(row_count):
            try:
                text = _browser._safe_locator_text(
                    rows.nth(i).locator(candidate).first,
                    timeout_ms=timeout_ms,
                )
            except Exception:
                text = ""
            if require_timestamp:
                if re.search(r"\b\d+:\d{2}(?:\.\d+)?\b", text or ""):
                    hits += 1
            elif text:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_sel = candidate
    return best_sel if best_hits > 0 else ""


def _wait_for_segments_stable(
    page: Page,
    rows_selector: str,
    timeout_ms: int = 3000,
    poll_ms: int = 200,
) -> bool:
    """Wait until segment row count and text content stabilize (React render complete).

    Returns True if segments stabilized within timeout, False if timed out.
    """
    prev_hash = ""
    deadline = time.time() + timeout_ms / 1000.0
    stable_count = 0
    required_stable = 2  # Need 2 consecutive identical reads

    while time.time() < deadline:
        try:
            rows = page.locator(rows_selector)
            count = rows.count()
            # Sample first and last few rows for stability check
            sample_indices = list(range(min(count, 3)))
            if count > 3:
                sample_indices.extend(range(max(3, count - 2), count))
            texts = []
            for i in sample_indices:
                try:
                    texts.append(_browser._safe_locator_text(rows.nth(i), timeout_ms=500))
                except Exception:
                    texts.append("")
            current_hash = f"{count}:{hash(tuple(texts))}"
        except Exception:
            current_hash = "error"

        if current_hash == prev_hash and prev_hash and current_hash != "error":
            stable_count += 1
            if stable_count >= required_stable:
                return True
        else:
            stable_count = 0
        prev_hash = current_hash
        page.wait_for_timeout(poll_ms)

    print(f"[trace] _wait_for_segments_stable timed out after {timeout_ms}ms (proceeding anyway)")
    return False


def extract_segments(
    page: Page,
    cfg: Dict[str, Any],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    max_segments = int(_cfg_get(cfg, "run.max_segments", 0) or 0)
    resolve_attempts = max(1, int(_cfg_get(cfg, "run.segment_resolve_attempts", 24)))
    resolve_retry_ms = max(150, int(_cfg_get(cfg, "run.segment_resolve_retry_ms", 800)))
    resolve_sample_size = max(1, int(_cfg_get(cfg, "run.segment_resolve_sample_size", 8)))
    resolve_row_text_timeout_ms = max(100, int(_cfg_get(cfg, "run.segment_resolve_row_text_timeout_ms", 350)))
    extract_row_text_timeout_ms = max(
        100,
        int(_cfg_get(cfg, "run.segment_extract_row_text_timeout_ms", resolve_row_text_timeout_ms)),
    )
    extract_progress_every = max(0, int(_cfg_get(cfg, "run.segment_extract_progress_every", 12) or 12))
    row_scroll_timeout_ms = max(250, int(_cfg_get(cfg, "run.segment_row_scroll_timeout_ms", 1200)))
    rows_sel = str(_cfg_get(cfg, "atlas.selectors.segment_rows", ""))
    label_sel = str(_cfg_get(cfg, "atlas.selectors.segment_label", ""))
    start_sel = str(_cfg_get(cfg, "atlas.selectors.segment_start", ""))
    end_sel = str(_cfg_get(cfg, "atlas.selectors.segment_end", ""))
    heartbeat = _active_extract_heartbeat_callback()

    def _touch_heartbeat() -> None:
        if not callable(heartbeat):
            return
        try:
            heartbeat()
        except Exception:
            pass

    # ── Wait for React render to stabilize before extraction ──
    if rows_sel:
        _wait_for_segments_stable(page, rows_sel, timeout_ms=3000)

    last_error: Optional[Exception] = None
    rows: Optional[Locator] = None
    for attempt in range(1, resolve_attempts + 1):
        _touch_heartbeat()
        _browser._dismiss_blocking_modals(page)
        try:
            _, rows = _resolve_rows_locator(
                page,
                rows_sel,
                sample_size=resolve_sample_size,
                row_text_timeout_ms=resolve_row_text_timeout_ms,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt == 1 or attempt % 3 == 0 or attempt == resolve_attempts:
                msg = str(exc).strip().replace("\n", " | ")
                if len(msg) > 220:
                    msg = msg[:220] + "..."
                print(f"[atlas] segment rows not ready (attempt {attempt}/{resolve_attempts}): {msg}")
            page.wait_for_timeout(resolve_retry_ms)
    if rows is None:
        if last_error:
            raise last_error
        raise RuntimeError("Could not resolve segment rows.")

    count = rows.count()
    limit = count if max_segments <= 0 else min(count, max_segments)
    resolved_label_sel = _resolve_row_child_selector(
        rows,
        label_sel,
        sample_size=min(limit, 4),
        timeout_ms=resolve_row_text_timeout_ms,
    )
    resolved_start_sel = _resolve_row_child_selector(
        rows,
        start_sel,
        sample_size=min(limit, 4),
        timeout_ms=resolve_row_text_timeout_ms,
        require_timestamp=True,
    )
    resolved_end_sel = _resolve_row_child_selector(
        rows,
        end_sel,
        sample_size=min(limit, 4),
        timeout_ms=resolve_row_text_timeout_ms,
        require_timestamp=True,
    )
    if label_sel and not resolved_label_sel:
        print("[atlas] segment label child selector missed sample rows; using row-text fallback.")
    if start_sel and not resolved_start_sel:
        print("[atlas] segment start child selector missed sample rows; using row-text fallback.")
    if end_sel and not resolved_end_sel:
        print("[atlas] segment end child selector missed sample rows; using row-text fallback.")

    items: List[Dict[str, Any]] = []
    for i in range(limit):
        _touch_heartbeat()
        row = rows.nth(i)
        label = _first_text_from_row(row, resolved_label_sel, timeout_ms=resolve_row_text_timeout_ms)
        start_text = _first_text_from_row(row, resolved_start_sel, timeout_ms=resolve_row_text_timeout_ms)
        end_text = _first_text_from_row(row, resolved_end_sel, timeout_ms=resolve_row_text_timeout_ms)
        raw_text = ""
        if not label or ":" not in str(start_text or "") or ":" not in str(end_text or ""):
            raw_text = _browser._safe_locator_text(row, timeout_ms=extract_row_text_timeout_ms)
        if not label and not str(start_text or "").strip() and not str(end_text or "").strip() and not raw_text:
            try:
                row.scroll_into_view_if_needed(timeout=row_scroll_timeout_ms)
            except Exception:
                pass
            label = _first_text_from_row(row, resolved_label_sel, timeout_ms=resolve_row_text_timeout_ms)
            start_text = _first_text_from_row(row, resolved_start_sel, timeout_ms=resolve_row_text_timeout_ms)
            end_text = _first_text_from_row(row, resolved_end_sel, timeout_ms=resolve_row_text_timeout_ms)
            if not label or ":" not in str(start_text or "") or ":" not in str(end_text or ""):
                raw_text = _browser._safe_locator_text(row, timeout_ms=extract_row_text_timeout_ms)
        start_sec = _parse_mmss_to_seconds(start_text)
        end_sec = _parse_mmss_to_seconds(end_text)

        fb_start, fb_end = _extract_start_end_from_text(raw_text)
        if start_sec <= 0 and fb_start > 0:
            start_sec = fb_start
        if end_sec <= 0 and fb_end > 0:
            end_sec = fb_end

        if not label and raw_text:
            lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            if lines:
                label = lines[-1]

        items.append(
            {
                "segment_index": i + 1,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "current_label": label,
                "raw_text": raw_text,
            }
        )
        if extract_progress_every > 0 and ((i + 1) % extract_progress_every == 0 or (i + 1) == limit):
            print(f"[trace] extract_segments progress: {i + 1}/{limit}")
            _touch_heartbeat()
            if callable(progress_callback):
                try:
                    progress_callback(i + 1, limit)
                except Exception:
                    pass

    # ── Structured trace logging ──
    if items:
        durations = [round(seg["end_sec"] - seg["start_sec"], 1) for seg in items]
        max_dur = max(durations) if durations else 0.0
        overlong = [i + 1 for i, d in enumerate(durations) if d > 10.0]
        checksum = build_segment_checksum(items)
        print(
            f"[trace] extract_segments: count={len(items)} "
            f"max_duration={max_dur:.1f}s "
            f"overlong_indices={overlong if overlong else 'none'} "
            f"source=DOM checksum={checksum}"
        )

    return items


def _pre_submit_duration_check(
    page: Page,
    cfg: Dict[str, Any],
    max_dur: float = 10.0,
) -> Dict[str, Any]:
    """Final safety check: re-read DOM and verify ALL segments <= max_dur.

    This is the absolute last line of defense before clicking Submit.
    """
    segments = extract_segments(page, cfg)
    violations: List[str] = []
    for seg in segments:
        dur = seg["end_sec"] - seg["start_sec"]
        if dur > max_dur + 0.05:
            violations.append(
                f"segment {seg['segment_index']}: {dur:.1f}s > {max_dur}s"
            )
    if violations:
        print(f"[guard] PRE-SUBMIT BLOCK: {len(violations)} segment(s) exceed {max_dur}s limit")
        for v in violations:
            print(f"[guard]   - {v}")
    else:
        print(f"[guard] pre-submit check passed: all {len(segments)} segments <= {max_dur}s")
    return {"ok": len(violations) == 0, "violations": violations, "segment_count": len(segments)}


def _normalize_operation_action(action: str) -> str:
    value = str(action or "").strip().lower()
    aliases = {
        "e": "edit",
        "edit": "edit",
        "s": "split",
        "split": "split",
        "d": "delete",
        "delete": "delete",
        "remove": "delete",
        "m": "merge",
        "merge": "merge",
    }
    return aliases.get(value, "")


def _normalize_operations(payload: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    raw_ops = payload.get("operations", [])
    if not isinstance(raw_ops, list):
        return []
    max_ops = max(0, int(_cfg_get(cfg or {}, "run.max_structural_operations", 12)))
    structural_allow_split = bool(_cfg_get(cfg or {}, "run.structural_allow_split", False))
    structural_allow_merge = bool(_cfg_get(cfg or {}, "run.structural_allow_merge", False))
    structural_allow_delete = bool(_cfg_get(cfg or {}, "run.structural_allow_delete", False))
    out: List[Dict[str, Any]] = []
    for item in raw_ops:
        action = ""
        idx = 0
        if isinstance(item, dict):
            action = _normalize_operation_action(
                str(item.get("action") or item.get("op") or item.get("type") or "")
            )
            idx_raw = item.get("segment_index", item.get("index", item.get("segment", 0)))
            try:
                idx = int(idx_raw)
            except Exception:
                idx = 0
        elif isinstance(item, str):
            token = item.strip().lower()
            match = re.match(r"([a-z]+)\s+(\d+)$", token)
            if match:
                action = _normalize_operation_action(match.group(1))
                idx = int(match.group(2))
            else:
                action = _normalize_operation_action(token)
        if not action or idx <= 0:
            continue
        if action == "split" and not structural_allow_split:
            continue
        if action == "merge" and not structural_allow_merge:
            continue
        if action == "delete" and not structural_allow_delete:
            continue
        out.append({"action": action, "segment_index": idx})
        if max_ops and len(out) >= max_ops:
            break
    return out


def _segment_duration_seconds(seg: Dict[str, Any]) -> float:
    start = _utils._safe_float(seg.get("start_sec"), 0.0)
    end = _utils._safe_float(seg.get("end_sec"), start)
    return max(0.0, end - start)


def _filter_structural_operations(
    operations: List[Dict[str, Any]],
    source_segments: Optional[List[Dict[str, Any]]],
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not operations:
        return []
    if not source_segments:
        return list(operations)

    max_segment_duration_sec = max(0.0, float(_cfg_get(cfg or {}, "run.max_segment_duration_sec", 10.0) or 0.0))
    split_min_duration_sec = float(
        _cfg_get(
            cfg or {},
            "run.structural_split_min_duration_sec",
            max_segment_duration_sec or 10.0,
        )
        or 0.0
    )
    if split_min_duration_sec <= 0.0:
        split_min_duration_sec = max_segment_duration_sec or 10.0

    segment_by_index: Dict[int, Dict[str, Any]] = {
        int(seg.get("segment_index", 0) or 0): seg for seg in source_segments
    }
    seen: set[Tuple[str, int]] = set()
    out: List[Dict[str, Any]] = []
    dropped_short: List[str] = []
    dropped_missing: List[int] = []

    for op in operations:
        action = str(op.get("action", "") or "").strip().lower()
        idx = int(op.get("segment_index", 0) or 0)
        if not action or idx <= 0:
            continue

        key = (action, idx)
        if key in seen:
            continue
        seen.add(key)

        if action == "split":
            seg = segment_by_index.get(idx)
            if seg is None:
                dropped_missing.append(idx)
                continue
            duration_sec = _segment_duration_seconds(seg)
            if split_min_duration_sec > 0.0 and duration_sec <= split_min_duration_sec + 1e-6:
                dropped_short.append(f"{idx}({duration_sec:.1f}s)")
                continue

        out.append({"action": action, "segment_index": idx})

    if dropped_short:
        print(
            "[run] filtered split operations on already-short segments: "
            + ", ".join(dropped_short[:12])
        )
    if dropped_missing:
        joined = ", ".join(str(idx) for idx in dropped_missing[:12])
        print(f"[run] dropped structural operations for missing source segments: {joined}")

    return out


_normalize_segment_plan = _labels._normalize_segment_plan
_normalize_label_map_from_plan = _labels._normalize_label_map_from_plan


def _first_visible_child_locator(row: Locator, selector: str, max_scan: int = 10) -> Optional[Locator]:
    for candidate in _browser._selector_variants(selector):
        try:
            loc = row.locator(candidate)
            count = min(loc.count(), max_scan)
            for i in range(count):
                item = loc.nth(i)
                if item.is_visible() and item.is_enabled():
                    return item
        except Exception:
            continue
    return None


def _respect_major_step_pause(
    cfg: Dict[str, Any],
    step_name: str,
    *,
    heartbeat: Optional[Callable[[], None]] = None,
) -> None:
    if not bool(_cfg_get(cfg, "run.major_step_pause_enabled", False)):
        return
    min_delay = float(_cfg_get(cfg, "run.major_step_pause_min_sec", 0.0) or 0.0)
    max_delay = float(_cfg_get(cfg, "run.major_step_pause_max_sec", 0.0) or 0.0)
    if max_delay < min_delay:
        min_delay, max_delay = max_delay, min_delay
    min_delay = max(0.0, min_delay)
    max_delay = max(0.0, max_delay)
    if max_delay <= 0.0:
        return
    delay = min_delay if max_delay == min_delay else random.uniform(min_delay, max_delay)
    label = str(step_name or "next step").strip() or "next step"
    print(f"[run] pacing pause before {label}: {delay:.1f}s")
    _runtime._sleep_with_shutdown_heartbeat(
        delay,
        heartbeat_sec=min(10.0, max(2.0, delay / 4.0)),
        on_heartbeat=heartbeat,
    )


def _short_error_text(exc: Exception, max_len: int = 180) -> str:
    raw = str(exc or "").strip()
    if not raw:
        return exc.__class__.__name__
    first = raw.splitlines()[0].strip()
    if len(first) > max_len:
        return first[:max_len] + "..."
    return first


def apply_timestamp_adjustments(
    page: Page,
    cfg: Dict[str, Any],
    source_segments: List[Dict[str, Any]],
    segment_plan: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    if not bool(_cfg_get(cfg, "run.adjust_timestamps", True)):
        return {"adjusted": 0, "failed": []}
    mode = str(_cfg_get(cfg, "run.timestamp_adjust_mode", "best_effort")).strip().lower() or "best_effort"
    if mode in {"off", "none", "disabled"}:
        return {"adjusted": 0, "failed": []}
    skip_if_segments_ge = max(0, int(_cfg_get(cfg, "run.timestamp_skip_if_segments_ge", 24)))
    if skip_if_segments_ge > 0 and len(segment_plan) >= skip_if_segments_ge and mode != "strict":
        print(
            f"[run] timestamp adjustments skipped: segment count {len(segment_plan)} >= "
            f"{skip_if_segments_ge} (mode={mode})."
        )
        return {"adjusted": 0, "failed": []}

    _browser._dismiss_blocking_modals(page)
    _browser._dismiss_blocking_side_panel(page, cfg, aggressive=True)

    rows_sel = str(_cfg_get(cfg, "atlas.selectors.segment_rows", ""))
    plus_sel = str(_cfg_get(cfg, "atlas.selectors.segment_time_plus_button", 'button:has(svg.lucide-plus)'))
    minus_sel = str(_cfg_get(cfg, "atlas.selectors.segment_time_minus_button", 'button:has(svg.lucide-minus)'))
    step_sec = max(0.01, float(_cfg_get(cfg, "atlas.timestamp_step_sec", 0.1)))
    max_clicks = max(1, int(_cfg_get(cfg, "atlas.timestamp_max_clicks_per_segment", 30)))
    click_timeout_ms = max(120, int(_cfg_get(cfg, "run.timestamp_click_timeout_ms", 350)))
    click_pause_ms = max(0, int(_cfg_get(cfg, "run.timestamp_click_pause_ms", 15)))
    max_failures = max(1, int(_cfg_get(cfg, "run.timestamp_max_failures_per_episode", 10)))
    max_total_clicks = max(1, int(_cfg_get(cfg, "run.timestamp_max_total_clicks", 80)))
    abort_on_first_failure = bool(_cfg_get(cfg, "run.timestamp_abort_on_first_failure", False))
    skip_disabled_buttons = bool(_cfg_get(cfg, "run.timestamp_skip_disabled_buttons", True))

    try:
        best_rows_sel, rows = _resolve_rows_locator(page, rows_sel)
    except Exception:
        return {"adjusted": 0, "failed": ["rows locator unavailable for timestamp adjustment"]}

    def _short_err(exc: Exception, max_len: int = 180) -> str:
        raw = str(exc or "").strip()
        if not raw:
            return exc.__class__.__name__
        first = raw.splitlines()[0].strip()
        if len(first) > max_len:
            return first[:max_len] + "..."
        return first

    source_by_idx: Dict[int, Dict[str, Any]] = {int(seg["segment_index"]): seg for seg in source_segments}
    adjusted = 0
    failed: List[str] = []
    total_clicks_done = 0

    for idx in sorted(segment_plan):
        if total_clicks_done >= max_total_clicks:
            print(
                f"[run] timestamp adjustment budget reached: "
                f"{total_clicks_done}/{max_total_clicks} clicks."
            )
            break
        if len(failed) >= max_failures:
            print(
                f"[run] timestamp adjustments stopped early after {len(failed)} failures "
                f"(limit={max_failures})."
            )
            break
        rows = page.locator(best_rows_sel)
        count = rows.count()
        if idx > count:
            continue
        src = source_by_idx.get(idx)
        if not src:
            continue
        target_end = _utils._safe_float(segment_plan[idx].get("end_sec"), _utils._safe_float(src.get("end_sec"), 0.0))
        current_end = _utils._safe_float(src.get("end_sec"), 0.0)
        diff = target_end - current_end
        if abs(diff) < (step_sec / 2):
            continue

        row = rows.nth(idx - 1)
        clicks = min(max_clicks, int(round(abs(diff) / step_sec)))
        if clicks <= 0:
            continue
        clicks = min(clicks, max_total_clicks - total_clicks_done)
        if clicks <= 0:
            break
        use_plus = diff > 0
        btn_sel = plus_sel if use_plus else minus_sel
        btn = _first_visible_child_locator(row, btn_sel)
        if btn is None:
            failed.append(f"segment {idx}: timestamp {'plus' if use_plus else 'minus'} button not found")
            if abort_on_first_failure:
                break
            continue
        if skip_disabled_buttons:
            try:
                if not btn.is_enabled():
                    failed.append(
                        f"segment {idx}: timestamp {'plus' if use_plus else 'minus'} button disabled"
                    )
                    if abort_on_first_failure:
                        break
                    continue
            except Exception:
                pass
        try:
            _browser._click_segment_row_with_recovery(page, rows, idx, cfg)
            clicked_this_segment = 0
            for _ in range(clicks):
                live_row = page.locator(best_rows_sel).nth(idx - 1)
                live_btn = _first_visible_child_locator(live_row, btn_sel)
                if live_btn is None:
                    raise RuntimeError(
                        f"timestamp {'plus' if use_plus else 'minus'} button disappeared during adjustment"
                    )
                if skip_disabled_buttons:
                    try:
                        if not live_btn.is_enabled():
                            raise RuntimeError(
                                f"timestamp {'plus' if use_plus else 'minus'} button disabled during adjustment"
                            )
                    except RuntimeError:
                        raise
                    except Exception:
                        pass
                try:
                    live_btn.click(timeout=click_timeout_ms, no_wait_after=True)
                except Exception as click_exc:
                    _browser._dismiss_blocking_side_panel(page, cfg, aggressive=True)
                    try:
                        live_btn.click(timeout=click_timeout_ms, force=True, no_wait_after=True)
                    except Exception as force_exc:
                        if mode == "strict":
                            raise force_exc
                        raise RuntimeError(_short_err(click_exc)) from force_exc
                clicked_this_segment += 1
                total_clicks_done += 1
                if click_pause_ms > 0:
                    time.sleep(click_pause_ms / 1000.0)
            if clicked_this_segment > 0:
                adjusted += 1
        except Exception as exc:
            failed.append(f"segment {idx}: {_short_err(exc)}")
            if abort_on_first_failure:
                break

    return {"adjusted": adjusted, "failed": failed}


def _action_selector_for_row(cfg: Dict[str, Any], action: str) -> str:
    if action == "edit":
        return str(_cfg_get(cfg, "atlas.selectors.edit_button_in_row", "")).strip()
    if action == "split":
        return str(_cfg_get(cfg, "atlas.selectors.split_button_in_row", "")).strip()
    if action == "delete":
        return str(_cfg_get(cfg, "atlas.selectors.delete_button_in_row", "")).strip()
    if action == "merge":
        return str(_cfg_get(cfg, "atlas.selectors.merge_button_in_row", "")).strip()
    return ""


def _action_hotkey(action: str) -> str:
    if action == "edit":
        return "e"
    if action == "split":
        return "s"
    if action == "delete":
        return "d"
    if action == "merge":
        return "m"
    return ""


def _action_confirm_selector_candidates(cfg: Dict[str, Any], action: str = "") -> List[str]:
    configured = str(_cfg_get(cfg, "atlas.selectors.action_confirm_button", "")).strip()
    action = str(action or "").strip().lower()
    action_specific: List[str] = []
    if action == "split":
        action_specific = [
            'div[role="dialog"] button:has-text("Split")',
            'div[role="dialog"] [role="button"]:has-text("Split")',
            'button:has-text("Split")',
            '[role="button"]:has-text("Split")',
            'button:has-text("Confirm Split")',
        ]
    elif action == "merge":
        action_specific = [
            'div[role="dialog"] button:has-text("Merge")',
            'div[role="dialog"] [role="button"]:has-text("Merge")',
            'button:has-text("Merge")',
            '[role="button"]:has-text("Merge")',
        ]
    elif action == "delete":
        action_specific = [
            'div[role="dialog"] button:has-text("Delete")',
            'div[role="dialog"] [role="button"]:has-text("Delete")',
            'button:has-text("Delete")',
            '[role="button"]:has-text("Delete")',
        ]

    generic = [
        'div[role="dialog"] button:has-text("Confirm")',
        'div[role="dialog"] [role="button"]:has-text("Confirm")',
        'div[role="dialog"] button:has-text("Apply")',
        'div[role="dialog"] [role="button"]:has-text("Apply")',
        'div[role="dialog"] button:has-text("Continue")',
        'div[role="dialog"] [role="button"]:has-text("Continue")',
        'button:has-text("Confirm")',
        '[role="button"]:has-text("Confirm")',
    ]

    seen: set[str] = set()
    out: List[str] = []
    for raw in action_specific + _browser._selector_variants(configured) + generic:
        selector = str(raw or "").strip()
        if not selector or selector in seen:
            continue
        seen.add(selector)
        out.append(selector)
    return out


def _confirm_action_dialog(page: Page, cfg: Dict[str, Any], action: str = "") -> bool:
    for selector in _action_confirm_selector_candidates(cfg, action):
        clicked = _browser._safe_locator_click(page, selector, timeout_ms=1200)
        if clicked:
            page.wait_for_timeout(250)
            return True
    return False


def _wait_rows_delta(
    page: Page,
    rows_selector: str,
    before_count: int,
    expected_delta: int,
    timeout_ms: int = 4000,
    *,
    mode: str = "exact",
) -> bool:
    if expected_delta == 0:
        return True
    target = max(0, before_count + expected_delta)
    deadline = time.time() + (timeout_ms / 1000.0)
    while time.time() < deadline:
        try:
            current = page.locator(rows_selector).count()
            if mode == "at_least":
                if current >= target:
                    return True
            elif mode == "at_most":
                if current <= target:
                    return True
            elif current == target:
                return True
        except Exception:
            pass
        time.sleep(0.12)
    return False


def _split_seek_target_seconds(seg: Dict[str, Any]) -> Optional[float]:
    start = _utils._safe_float(seg.get("start_sec"), 0.0)
    end = _utils._safe_float(seg.get("end_sec"), start)
    if end <= start + 0.3:
        return None
    midpoint = start + ((end - start) / 2.0)
    target = min(end - 0.15, max(start + 0.15, midpoint))
    if target <= start or target >= end:
        return None
    return round(target, 3)


def _seek_video_to_time(page: Page, cfg: Dict[str, Any], target_sec: float, timeout_ms: int = 2800) -> bool:
    selectors = _cfg_get(cfg, "atlas.selectors", {}) or {}
    video_sel = str(selectors.get("video_element", "video")).strip() or "video"
    try:
        state = page.evaluate(
            """([selector, target]) => {
                const v = document.querySelector(selector) || document.querySelector('video');
                if (!v) return null;
                const duration = Number(v.duration || 0);
                let clamped = Number(target || 0);
                if (Number.isFinite(duration) && duration > 0) {
                    clamped = Math.min(Math.max(0, clamped), Math.max(0, duration - 0.05));
                } else {
                    clamped = Math.max(0, clamped);
                }
                try { v.pause(); } catch (e) {}
                try { v.currentTime = clamped; } catch (e) {}
                return {
                    current: Number(v.currentTime || 0),
                    duration: duration,
                    target: clamped,
                };
            }""",
            [video_sel, float(target_sec)],
        )
    except Exception:
        return False
    if not state:
        return False

    target = float(state.get("target", target_sec) or target_sec)
    deadline = time.time() + (timeout_ms / 1000.0)
    while time.time() < deadline:
        try:
            current = page.evaluate(
                """(selector) => {
                    const v = document.querySelector(selector) || document.querySelector('video');
                    if (!v) return null;
                    return Number(v.currentTime || 0);
                }""",
                video_sel,
            )
        except Exception:
            current = None
        if current is not None and abs(float(current) - target) <= 0.35:
            page.wait_for_timeout(120)
            return True
        page.wait_for_timeout(80)
    return False


def _position_video_for_split(
    page: Page,
    cfg: Dict[str, Any],
    source_segment: Optional[Dict[str, Any]],
    idx: int,
) -> bool:
    if not isinstance(source_segment, dict):
        return False
    target = _split_seek_target_seconds(source_segment)
    if target is None:
        return False
    positioned = _seek_video_to_time(page, cfg, target)
    if positioned:
        print(f"[atlas] positioned playhead for split segment {idx} at {target:.1f}s")
    return positioned


def _structural_candidate_row_indices(action: str, idx: int, count: int) -> List[int]:
    action = str(action or "").strip().lower()
    if idx <= 0 or count <= 0:
        return []
    out: List[int] = [idx]
    # Atlas merge semantics are "merge row N into N-1". Some UI variants expose the
    # merge affordance on the current row, while others expose it on the previous row.
    if action == "merge" and idx > 1:
        out.append(idx - 1)
    deduped: List[int] = []
    seen: set[int] = set()
    for candidate in out:
        if candidate <= 0 or candidate > count or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _should_force_structural_repairs_on_large_episode(
    operations: List[Dict[str, Any]],
    source_segments: Optional[List[Dict[str, Any]]],
    cfg: Dict[str, Any],
) -> bool:
    if not operations or not source_segments:
        return False
    max_duration = max(0.1, float(_cfg_get(cfg, "run.max_segment_duration_sec", 10.0) or 10.0))
    aggressive_actions = {
        str(op.get("action", "")).strip().lower()
        for op in (operations or [])
        if str(op.get("action", "")).strip()
    }
    if not aggressive_actions.intersection({"split", "delete"}):
        return False
    for seg in source_segments or []:
        try:
            start_sec = float(seg.get("start_sec", 0.0) or 0.0)
            end_sec = float(seg.get("end_sec", start_sec) or start_sec)
        except Exception:
            continue
        if (end_sec - start_sec) > (max_duration + 0.05):
            return True
    return False


def apply_segment_operations(
    page: Page,
    cfg: Dict[str, Any],
    operations: List[Dict[str, Any]],
    *,
    source_segments: Optional[List[Dict[str, Any]]] = None,
    heartbeat: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    def _heartbeat() -> None:
        if callable(heartbeat):
            try:
                heartbeat()
            except Exception:
                pass

    if not operations:
        return {"applied": 0, "structural_applied": 0, "failed": []}
    operations = _filter_structural_operations(operations, source_segments, cfg)
    if not operations:
        return {"applied": 0, "structural_applied": 0, "failed": []}
    rows_sel = str(_cfg_get(cfg, "atlas.selectors.segment_rows", ""))
    sample_size = max(1, int(_cfg_get(cfg, "run.segment_resolve_sample_size", 8)))
    row_text_timeout_ms = max(100, int(_cfg_get(cfg, "run.segment_resolve_row_text_timeout_ms", 350)))
    structural_skip_if_segments_ge = max(0, int(_cfg_get(cfg, "run.structural_skip_if_segments_ge", 40)))
    structural_max_failures = max(1, int(_cfg_get(cfg, "run.structural_max_failures_per_episode", 4)))
    structural_wait_rows_delta_timeout_ms = max(
        600, int(_cfg_get(cfg, "run.structural_wait_rows_delta_timeout_ms", 1800))
    )
    failed: List[str] = []
    applied = 0
    structural_applied = 0
    source_by_idx: Dict[int, Dict[str, Any]] = {
        int(seg.get("segment_index", 0) or 0): seg for seg in (source_segments or [])
    }
    force_large_episode_repairs = _should_force_structural_repairs_on_large_episode(
        operations,
        source_segments,
        cfg,
    )

    if structural_skip_if_segments_ge > 0:
        try:
            _, probe_rows = _resolve_rows_locator(
                page,
                rows_sel,
                sample_size=sample_size,
                row_text_timeout_ms=row_text_timeout_ms,
            )
            seg_count = probe_rows.count()
            if seg_count >= structural_skip_if_segments_ge:
                if force_large_episode_repairs:
                    print(
                        f"[run] overriding large-episode structural skip: segment count {seg_count} >= "
                        f"{structural_skip_if_segments_ge}, but source contains overlong segments."
                    )
                else:
                    allow_merge_when_large = bool(_cfg_get(cfg, "run.structural_skip_allow_merge", True))
                    if allow_merge_when_large:
                        retained_ops = [op for op in operations if str(op.get("action", "")).strip().lower() == "merge"]
                    else:
                        retained_ops = []
                    skipped_count = max(0, len(operations) - len(retained_ops))
                    if retained_ops:
                        print(
                            f"[run] structural split/delete skipped on large episode: segment count {seg_count} >= "
                            f"{structural_skip_if_segments_ge}; retaining {len(retained_ops)} merge op(s)."
                        )
                        operations = retained_ops
                    else:
                        print(
                            f"[run] structural operations skipped: segment count {seg_count} >= "
                            f"{structural_skip_if_segments_ge}."
                        )
                        return {"applied": 0, "structural_applied": 0, "failed": []}
                    if skipped_count > 0:
                        failed.append(
                            f"skipped {skipped_count} structural op(s) on large episode "
                            f"(segment count={seg_count})"
                        )
        except Exception:
            pass

    for i, op in enumerate(operations, start=1):
        _heartbeat()
        if len(failed) >= structural_max_failures:
            print(
                f"[run] structural operations stopped after {len(failed)} failures "
                f"(limit={structural_max_failures})."
            )
            break
        action = str(op.get("action", "")).strip().lower()
        idx = int(op.get("segment_index", 0) or 0)
        if action not in {"edit", "split", "delete", "merge"} or idx <= 0:
            failed.append(f"op#{i}: invalid operation payload {op}")
            continue

        _browser._dismiss_blocking_modals(page, cfg)
        _browser._dismiss_blocking_side_panel(page, cfg, aggressive=True)
        _heartbeat()
        try:
            best_rows_sel, rows = _resolve_rows_locator(
                page,
                rows_sel,
                sample_size=sample_size,
                row_text_timeout_ms=row_text_timeout_ms,
            )
        except Exception as exc:
            failed.append(f"op#{i} {action} segment {idx}: rows unavailable ({exc})")
            continue

        count = rows.count()
        if idx > count:
            failed.append(f"op#{i} {action} segment {idx}: row missing (count={count})")
            continue

        row = rows.nth(idx - 1)
        try:
            _browser._click_segment_row_with_recovery(page, rows, idx, cfg)
        except Exception as exc:
            failed.append(f"op#{i} {action} segment {idx}: cannot focus row ({exc})")
            continue

        before_count = count
        btn_sel = _action_selector_for_row(cfg, action)
        expected_delta = 0
        if action == "split":
            expected_delta = 1
        elif action in {"delete", "merge"}:
            expected_delta = -1
        wait_mode = "exact"
        if action == "split":
            wait_mode = "at_least"
        elif action in {"delete", "merge"}:
            wait_mode = "at_most"

        trigger_attempts: List[str] = []
        if btn_sel:
            trigger_attempts.append("button")
        if _action_hotkey(action):
            trigger_attempts.append("hotkey")
        if not trigger_attempts:
            failed.append(f"op#{i} {action} segment {idx}: action trigger failed")
            continue

        succeeded = False
        for trigger_no, trigger_kind in enumerate(trigger_attempts, start=1):
            row_candidates = _structural_candidate_row_indices(action, idx, before_count) or [idx]
            for candidate_idx in row_candidates:
                _heartbeat()
                _browser._dismiss_blocking_modals(page, cfg)
                _browser._dismiss_blocking_side_panel(page, cfg, aggressive=True)
                try:
                    live_rows = page.locator(best_rows_sel)
                    live_row = live_rows.nth(candidate_idx - 1)
                except Exception:
                    live_rows = rows
                    live_row = row
                if action == "split":
                    _position_video_for_split(page, cfg, source_by_idx.get(idx), idx)
                    _heartbeat()
                try:
                    _browser._click_segment_row_with_recovery(page, live_rows, candidate_idx, cfg)
                except Exception:
                    pass
                try:
                    live_row.scroll_into_view_if_needed(timeout=800)
                except Exception:
                    pass
                try:
                    live_row.hover(timeout=900)
                    page.wait_for_timeout(120)
                except Exception:
                    pass

                triggered = False
                if trigger_kind == "button" and btn_sel:
                    for candidate in _browser._selector_variants(btn_sel):
                        try:
                            btn = live_row.locator(candidate).first
                            if btn.count() <= 0:
                                continue
                            try:
                                btn.scroll_into_view_if_needed(timeout=800)
                            except Exception:
                                pass
                            try:
                                if btn.is_visible() and btn.is_enabled():
                                    btn.click(timeout=1200, no_wait_after=True, force=True)
                                    triggered = True
                                    break
                            except Exception:
                                pass
                            try:
                                btn.evaluate("(el) => el.click()")
                                triggered = True
                                break
                            except Exception:
                                continue
                        except Exception:
                            continue
                elif trigger_kind == "hotkey":
                    key = _action_hotkey(action)
                    if key:
                        try:
                            page.keyboard.press(key)
                            triggered = True
                        except Exception:
                            try:
                                page.keyboard.press(key.upper())
                                triggered = True
                            except Exception:
                                triggered = False

                if not triggered:
                    continue

                if action in {"delete", "merge", "split"}:
                    page.wait_for_timeout(220)
                    _heartbeat()
                    _confirm_action_dialog(page, cfg, action=action)
                    _browser._dismiss_blocking_modals(page, cfg)
                    _heartbeat()

                wait_timeout_ms = structural_wait_rows_delta_timeout_ms
                if action in {"split", "merge"}:
                    wait_timeout_ms = max(wait_timeout_ms, 3200)
                if _wait_rows_delta(
                    page,
                    best_rows_sel,
                    before_count,
                    expected_delta,
                    timeout_ms=wait_timeout_ms,
                    mode=wait_mode,
                ):
                    _heartbeat()
                    succeeded = True
                    break

                page.wait_for_timeout(350 + (trigger_no * 120))
                _heartbeat()

            if succeeded:
                break

        if not succeeded:
            try:
                after_count = page.locator(best_rows_sel).count()
            except Exception:
                after_count = before_count
            if expected_delta != 0 and after_count == before_count:
                failed.append(
                    f"op#{i} {action} segment {idx}: no row-count change "
                    f"(expected {before_count + expected_delta}, got {after_count})"
                )
                continue

        applied += 1
        if action in {"split", "delete", "merge"}:
            structural_applied += 1
        print(f"[atlas] operation applied: {action} on segment {idx}")
        if best_rows_sel:
            _wait_for_segments_stable(
                page,
                best_rows_sel,
                timeout_ms=max(1200, min(4000, structural_wait_rows_delta_timeout_ms + 800)),
            )
        page.wait_for_timeout(220)
        _heartbeat()

    return {"applied": applied, "structural_applied": structural_applied, "failed": failed}


def _fill_input(locator: Locator, label: str, page: Page) -> None:
    try:
        locator.scroll_into_view_if_needed(timeout=700)
    except Exception:
        pass
    try:
        locator.click(timeout=900, force=True)
    except Exception:
        try:
            locator.click(timeout=700)
        except Exception:
            pass
    try:
        editable = bool(locator.evaluate("el => !!el.isContentEditable"))
    except Exception:
        editable = False

    try:
        locator.fill(label, timeout=1600)
        return
    except Exception:
        pass

    try:
        locator.evaluate("el => { if (el && typeof el.focus === 'function') el.focus(); }")
    except Exception:
        pass

    if editable:
        page.keyboard.press("Control+A")
        page.keyboard.insert_text(label)
        return
    page.keyboard.press("Control+A")
    page.keyboard.insert_text(label)


def _filter_unchanged_label_map(
    label_map: Dict[int, str],
    source_segments: List[Dict[str, Any]],
) -> Tuple[Dict[int, str], int]:
    source_by_idx: Dict[int, str] = {
        int(seg.get("segment_index", 0)): str(seg.get("current_label", "")).strip()
        for seg in source_segments
    }
    out: Dict[int, str] = {}
    skipped = 0
    for idx, target in label_map.items():
        current = source_by_idx.get(int(idx), "")
        if current and _normalize_label_for_compare(current) == _normalize_label_for_compare(target):
            skipped += 1
            continue
        out[int(idx)] = target
    return out, skipped


def _wait_after_quality_review_accept(
    page: Page,
    cfg: Dict[str, Any],
    *,
    task_id: str = "",
    reason: str = "",
) -> None:
    settle_sec = max(
        0.0,
        float(_cfg_get(cfg, "run.quality_review_submit_settle_sec", 1.3) or 1.3),
    )
    if settle_sec <= 0:
        return
    reason_text = str(reason or "").strip() or "quality_review_accepted"
    print(f"[atlas] waiting {settle_sec:.1f}s after quality review acceptance ({reason_text}).")
    _capture_submit_step(
        page,
        cfg,
        f"quality_review_accept_wait_{reason_text}",
        task_id=task_id,
        include_html=True,
    )
    page.wait_for_timeout(int(settle_sec * 1000.0))


def _handle_quality_review_modal(
    page: Page,
    cfg: Dict[str, Any],
    timeout_ms: int = 8000,
    *,
    task_id: str = "",
    return_details: bool = False,
) -> bool | Tuple[bool, bool]:
    def _done(ok: bool, saw_modal: bool) -> bool | Tuple[bool, bool]:
        if return_details:
            return ok, saw_modal
        return ok

    if not bool(_cfg_get(cfg, "run.enable_quality_review_submit", True)):
        return _done(True, False)

    modal_sel = str(_cfg_get(cfg, "atlas.selectors.quality_review_modal", "")).strip()
    checkbox_sel = str(_cfg_get(cfg, "atlas.selectors.quality_review_checkbox", "")).strip()
    submit_sel = str(_cfg_get(cfg, "atlas.selectors.quality_review_submit_button", "")).strip()
    if not checkbox_sel or not submit_sel:
        return _done(True, False)

    modal = _browser._first_visible_locator(page, modal_sel, timeout_ms=timeout_ms) if modal_sel else None
    page_prompt_visible = _quality_review_prompt_visible(page)
    if modal is None and not page_prompt_visible:
        return _done(True, False)
    try:
        modal_text = (
            re.sub(r"\s+", " ", (modal.inner_text(timeout=1200) or "")).strip().lower()
            if modal is not None
            else _page_body_text_lower(page)
        )
    except Exception:
        modal_text = ""
    if modal_text and not any(
        marker in modal_text
        for marker in (
            "quality",
            "review",
            "reviewed",
            "confirm",
            "checkbox",
            "submit",
            "proceed",
            "continue",
            "done",
            "finish",
        )
    ):
        return _done(True, False)

    saw_modal = True
    print("[atlas] quality review modal detected.")
    _capture_submit_step(page, cfg, "quality_review_modal_detected", task_id=task_id, include_html=True)
    scope = modal if modal is not None else page

    def _force_check_quality_review_checkbox(modal_loc: Optional[Locator]) -> bool:
        phrases = [
            "i verify that i have reviewed every segment",
            "every label is correct to the best of my ability",
            "before submitting, please confirm you've reviewed your work",
        ]
        modal_script = """(modal, phrases) => {
                        const normalize = (value) => String(value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                        const scope = modal || document;
                        const markChecked = (node) => {
                            if (!node) return false;
                            try {
                                if (node instanceof HTMLInputElement && node.type === 'checkbox') {
                                    if (!node.checked) {
                                        node.checked = true;
                                        node.dispatchEvent(new Event('input', { bubbles: true }));
                                        node.dispatchEvent(new Event('change', { bubbles: true }));
                                    }
                                    node.click();
                                    return true;
                                }
                                const role = normalize(node.getAttribute && node.getAttribute('role'));
                                if (role === 'checkbox') {
                                    node.click();
                                    return normalize(node.getAttribute('aria-checked')) === 'true'
                                        || normalize(node.ariaChecked) === 'true'
                                        || true;
                                }
                            } catch (err) {}
                            return false;
                        };

                        const allCheckboxes = Array.from(
                            scope.querySelectorAll('input[type="checkbox"], [role="checkbox"]')
                        );
                        for (const cb of allCheckboxes) {
                            const container = cb.closest('label, div, section, article') || cb.parentElement || cb;
                            const text = normalize(container && container.innerText);
                            if (!phrases.some((phrase) => text.includes(phrase))) {
                                continue;
                            }
                            if (markChecked(cb)) return true;
                            try { container.click(); } catch (err) {}
                            if (markChecked(cb)) return true;
                        }

                        if (allCheckboxes.length === 1) {
                            const only = allCheckboxes[0];
                            if (markChecked(only)) return true;
                            const container = only.closest('label, div, section, article') || only.parentElement || only;
                            try { container.click(); } catch (err) {}
                            if (markChecked(only)) return true;
                        }

                        const textNodes = Array.from(scope.querySelectorAll('label, span, div, p'));
                        for (const node of textNodes) {
                            const text = normalize(node.innerText);
                            if (!phrases.some((phrase) => text.includes(phrase))) {
                                continue;
                            }
                            try { node.click(); } catch (err) {}
                            const container = node.closest('label, div, section, article') || node.parentElement || node;
                            const cb = container.querySelector('input[type="checkbox"], [role="checkbox"]');
                            if (markChecked(cb)) return true;
                        }
                        return false;
                    }"""
        page_script = """(phrases) => {
                        const normalize = (value) => String(value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                        const scope = document;
                        const markChecked = (node) => {
                            if (!node) return false;
                            try {
                                if (node instanceof HTMLInputElement && node.type === 'checkbox') {
                                    if (!node.checked) {
                                        node.checked = true;
                                        node.dispatchEvent(new Event('input', { bubbles: true }));
                                        node.dispatchEvent(new Event('change', { bubbles: true }));
                                    }
                                    node.click();
                                    return true;
                                }
                                const role = normalize(node.getAttribute && node.getAttribute('role'));
                                if (role === 'checkbox') {
                                    node.click();
                                    return normalize(node.getAttribute('aria-checked')) === 'true'
                                        || normalize(node.ariaChecked) === 'true'
                                        || true;
                                }
                            } catch (err) {}
                            return false;
                        };

                        const allCheckboxes = Array.from(
                            scope.querySelectorAll('input[type="checkbox"], [role="checkbox"]')
                        );
                        for (const cb of allCheckboxes) {
                            const container = cb.closest('label, div, section, article') || cb.parentElement || cb;
                            const text = normalize(container && container.innerText);
                            if (!phrases.some((phrase) => text.includes(phrase))) {
                                continue;
                            }
                            if (markChecked(cb)) return true;
                            try { container.click(); } catch (err) {}
                            if (markChecked(cb)) return true;
                        }

                        if (allCheckboxes.length === 1) {
                            const only = allCheckboxes[0];
                            if (markChecked(only)) return true;
                            const container = only.closest('label, div, section, article') || only.parentElement || only;
                            try { container.click(); } catch (err) {}
                            if (markChecked(only)) return true;
                        }

                        const textNodes = Array.from(scope.querySelectorAll('label, span, div, p'));
                        for (const node of textNodes) {
                            const text = normalize(node.innerText);
                            if (!phrases.some((phrase) => text.includes(phrase))) {
                                continue;
                            }
                            try { node.click(); } catch (err) {}
                            const container = node.closest('label, div, section, article') || node.parentElement || node;
                            const cb = container.querySelector('input[type="checkbox"], [role="checkbox"]');
                            if (markChecked(cb)) return true;
                        }
                        return false;
                    }"""
        try:
            forced = bool(
                modal_loc.evaluate(modal_script, phrases) if modal_loc is not None else page.evaluate(page_script, phrases)
            )
        except Exception:
            try:
                forced = bool(page.evaluate(page_script, phrases))
            except Exception:
                forced = False
        return forced

    checked = False
    for candidate in _browser._selector_variants(checkbox_sel):
        try:
            loc = scope.locator(candidate)
            scan = min(loc.count(), 4)
            for i in range(scan):
                cb = loc.nth(i)
                if not cb.is_visible():
                    continue
                try:
                    tag = str(cb.evaluate("el => (el.tagName || '').toLowerCase()"))
                    typ = str(cb.evaluate("el => (el.getAttribute('type') || '').toLowerCase()"))
                except Exception:
                    tag, typ = "", ""
                try:
                    if tag == "input" and typ == "checkbox":
                        cb.check(timeout=1200, force=True)
                    else:
                        cb.click(timeout=1200, force=True, no_wait_after=True)
                    checked = True
                    break
                except Exception:
                    continue
            if checked:
                break
        except Exception:
            continue
    if not checked:
        checked = _force_check_quality_review_checkbox(modal)
    if checked:
        print("[atlas] quality review checkbox checked.")
        _capture_submit_step(page, cfg, "quality_review_checkbox_checked", task_id=task_id)
        page.wait_for_timeout(250)
    else:
        print("[atlas] quality review checkbox could not be checked.")

    def _find_submit_button(modal_loc: Any) -> Optional[Locator]:
        for candidate in _browser._selector_variants(submit_sel):
            try:
                loc = modal_loc.locator(candidate)
                scan = min(loc.count(), 4)
                for i in range(scan):
                    btn = loc.nth(i)
                    if btn.is_visible():
                        return btn
            except Exception:
                continue
        return None

    def _try_click_submit(modal_loc: Any) -> bool:
        submit_btn = _find_submit_button(modal_loc)
        if submit_btn is None:
            return False
        try:
            disabled = bool(
                submit_btn.evaluate(
                    "el => !!el.disabled || String(el.getAttribute('aria-disabled') || '').toLowerCase() === 'true'"
                )
            )
        except Exception:
            disabled = False
        if disabled:
            return False
        try:
            submit_btn.click(timeout=1500, force=True, no_wait_after=True)
            return True
        except Exception:
            return False

    submitted = False
    if not checked and _find_submit_button(scope) is None:
        return _done(True, saw_modal)
    for _ in range(5):
        submit_btn = _find_submit_button(scope)
        if submit_btn is None:
            break
        try:
            disabled = bool(
                submit_btn.evaluate(
                    "el => !!el.disabled || String(el.getAttribute('aria-disabled') || '').toLowerCase() === 'true'"
                )
            )
        except Exception:
            disabled = False
        if disabled:
            page.wait_for_timeout(300)
            continue
        try:
            submit_btn.click(timeout=1500, force=True, no_wait_after=True)
            submitted = True
            print("[atlas] quality review submitted.")
            _capture_submit_step(page, cfg, "quality_review_submitted", task_id=task_id, include_html=True)
            _wait_after_quality_review_accept(
                page,
                cfg,
                task_id=task_id,
                reason="submit_clicked",
            )
            break
        except Exception:
            page.wait_for_timeout(300)
            continue

    if not submitted:
        if checked:
            current_modal = _browser._first_visible_locator(page, modal_sel, timeout_ms=350) if modal_sel else None
            if current_modal is None and not _quality_review_prompt_visible(page):
                print("[atlas] quality review accepted before submit button became clickable.")
                _wait_after_quality_review_accept(
                    page,
                    cfg,
                    task_id=task_id,
                    reason="accepted_before_submit_click",
                )
                return _done(True, saw_modal)
        return _done(False, saw_modal)

    for _ in range(18):
        current_modal = _browser._first_visible_locator(page, modal_sel, timeout_ms=350) if modal_sel else None
        if current_modal is None and not _quality_review_prompt_visible(page):
            if checked:
                print("[atlas] quality review accepted without visible submit button.")
                _wait_after_quality_review_accept(
                    page,
                    cfg,
                    task_id=task_id,
                    reason="auto_accepted",
                )
            return _done(True, saw_modal)
        active_scope = current_modal if current_modal is not None else page
        if _try_click_submit(active_scope):
            page.wait_for_timeout(450)
        else:
            page.wait_for_timeout(450)

    current_modal = _browser._first_visible_locator(page, modal_sel, timeout_ms=350) if modal_sel else None
    if current_modal is None and not _quality_review_prompt_visible(page):
        if checked:
            print("[atlas] quality review accepted after modal disappeared.")
            _wait_after_quality_review_accept(
                page,
                cfg,
                task_id=task_id,
                reason="modal_disappeared",
            )
        return _done(True, saw_modal)
    active_scope = current_modal if current_modal is not None else page
    submit_btn = _find_submit_button(active_scope)
    if submit_btn is None:
        if checked:
            print("[atlas] quality review accepted after submit button disappeared.")
            _wait_after_quality_review_accept(
                page,
                cfg,
                task_id=task_id,
                reason="submit_button_missing",
            )
        return _done(True, saw_modal)
    try:
        disabled = bool(
            submit_btn.evaluate(
                "el => !!el.disabled || String(el.getAttribute('aria-disabled') || '').toLowerCase() === 'true'"
            )
        )
    except Exception:
        disabled = False
    if disabled:
        print("[atlas] quality review submit appears accepted (button disabled; modal still visible).")
        _wait_after_quality_review_accept(
            page,
            cfg,
            task_id=task_id,
            reason="submit_button_disabled",
        )
        return _done(True, saw_modal)
    return _done(False, saw_modal)


def _handle_no_edits_modal(
    page: Page,
    cfg: Optional[Dict[str, Any]] = None,
    timeout_ms: int = 7000,
    *,
    task_id: str = "",
    return_details: bool = False,
) -> bool | Tuple[bool, bool]:
    def _done(ok: bool, saw_modal: bool) -> bool | Tuple[bool, bool]:
        if return_details:
            return ok, saw_modal
        return ok

    start = time.time()
    deadline = start + max(800, timeout_ms) / 1000.0
    confirm_sel = (
        'button:has-text("Yes, Labels Are Correct") '
        '|| [role="button"]:has-text("Yes, Labels Are Correct")'
    )
    saw_modal = False

    def _modal_visible() -> bool:
        try:
            body = (page.inner_text("body") or "").lower()
        except Exception:
            body = ""
        if not body:
            return False
        return "no edits made" in body and "labels are correct" in body

    while time.time() < deadline:
        visible = _modal_visible()
        if not visible:
            if saw_modal or (time.time() - start) >= 1.2:
                return _done(True, saw_modal)
            page.wait_for_timeout(180)
            continue

        saw_modal = True
        _capture_submit_step(page, cfg or {}, "no_edits_modal_detected", task_id=task_id, include_html=True)
        confirm_btn = _browser._first_visible_locator(page, confirm_sel, timeout_ms=400)
        if confirm_btn is None:
            page.wait_for_timeout(250)
            continue

        try:
            disabled = bool(
                confirm_btn.evaluate(
                    "el => !!el.disabled || String(el.getAttribute('aria-disabled') || '').toLowerCase() === 'true'"
                )
            )
        except Exception:
            disabled = False
        if disabled:
            page.wait_for_timeout(350)
            continue

        clicked = False
        try:
            confirm_btn.click(timeout=1500, force=True, no_wait_after=True)
            clicked = True
        except Exception:
            try:
                clicked = bool(
                    page.evaluate(
                        """() => {
                            const nodes = Array.from(document.querySelectorAll('button,[role="button"]'));
                            for (const node of nodes) {
                                const text = (node.innerText || node.textContent || '').trim().toLowerCase();
                                if (!text.startsWith('yes, labels are correct')) continue;
                                const ariaDisabled = String(node.getAttribute('aria-disabled') || '').toLowerCase() === 'true';
                                if (node.disabled || ariaDisabled) continue;
                                node.click();
                                return true;
                            }
                            return false;
                        }"""
                    )
                )
            except Exception:
                clicked = False
        if clicked:
            print("[atlas] confirmed 'No Edits Made' modal.")
            _capture_submit_step(page, cfg or {}, "no_edits_modal_confirmed", task_id=task_id, include_html=True)
            page.wait_for_timeout(700)
        else:
            page.wait_for_timeout(250)

    return _done(not _modal_visible(), saw_modal)


def _submit_transition_observed(page: Page) -> bool:
    try:
        current_url = str(getattr(page, "url", "") or "").strip().lower()
    except Exception:
        current_url = ""

    if current_url and "/label/" not in current_url:
        if "/tasks" in current_url or "/feedback" in current_url or "/disputes" in current_url:
            print(f"[atlas] submit appears accepted after navigation to: {current_url}")
            return True

    try:
        body = (page.inner_text("body") or "").strip().lower()
    except Exception:
        body = ""

    if not body:
        return False
    pending_confirmation_markers = (
        "no edits made",
        "yes, labels are correct",
        "quality review",
        "i verify that i have reviewed every segment",
        "before submitting, please confirm you've reviewed your work",
    )
    if any(marker in body for marker in pending_confirmation_markers):
        return False
    if current_url and "/tasks/room/normal/label/" in current_url:
        stale_label_markers = (
            "label episode",
            "segments",
            "action history",
            "quick guide",
        )
        if any(marker in body for marker in stale_label_markers):
            return False
    success_markers = (
        "campaign audit feedback",
        "your reserved episodes",
        "your journey",
        "reserve 3 episodes",
        "reserve 2 episodes",
        "reserve 1 episode",
        "back to tasks",
        "rooms are unavailable",
        "room access is currently disabled",
        "complete training",
        "do labeling tasks",
        "set up payment",
        "welcome,",
    )
    if any(marker in body for marker in success_markers):
        print("[atlas] submit appears accepted from post-submit page content.")
        return True
    return False


def _wait_for_submit_transition(page: Page, *, timeout_ms: int = 0, poll_ms: int = 500) -> bool:
    deadline = time.time() + max(0.0, float(timeout_ms) / 1000.0)
    if timeout_ms <= 0:
        return _submit_transition_observed(page)
    while time.time() < deadline:
        if _submit_transition_observed(page):
            return True
        try:
            page.wait_for_timeout(max(100, int(poll_ms)))
        except Exception:
            break
    return _submit_transition_observed(page)


def _safe_page_url(page: Page) -> str:
    try:
        return str(getattr(page, "url", "") or "").strip()
    except Exception:
        return ""


def _normalize_post_submit_tasks_view(page: Page, cfg: Dict[str, Any], *, task_id: str = "") -> bool:
    current_url = _safe_page_url(page).lower()
    if not current_url or "/tasks/room/normal" not in current_url or "/label/" in current_url:
        return False
    try:
        body = (page.inner_text("body") or "").strip().lower()
    except Exception:
        body = ""
    if any(
        marker in body
        for marker in (
            "rooms are unavailable",
            "room access is currently disabled",
            "back to tasks",
        )
    ):
        print("[atlas] post-submit landed on room page; returning to tasks root.")
        _capture_submit_step(page, cfg, "post_submit_room_page", task_id=task_id, include_html=True)
        if _browser._recover_room_access_disabled(page, cfg, timeout_ms=3000):
            _capture_submit_step(page, cfg, "post_submit_back_to_tasks_clicked", task_id=task_id)
            return True
    tasks_root_url = _browser._tasks_root_url(cfg)
    if not tasks_root_url:
        return False
    try:
        _browser._goto_with_retry(
            page,
            tasks_root_url,
            wait_until="domcontentloaded",
            timeout_ms=45000,
            cfg=cfg,
            reason="post-submit-back-to-tasks",
        )
        print("[atlas] post-submit normalized back to tasks root.")
        _capture_submit_step(page, cfg, "post_submit_tasks_root", task_id=task_id)
        return True
    except Exception:
        return False


def _new_submit_status(page: Page, *, reason: str = "") -> Dict[str, Any]:
    current_url = _safe_page_url(page)
    return SubmitOutcome(
        submit_attempted=False,
        submit_verified=False,
        submit_verification_reason=str(reason or "").strip(),
        page_url_before_submit=current_url,
        page_url_after_submit=current_url,
    ).to_dict()


def _read_submit_ui_validation_error(page: Page) -> str:
    try:
        toast_err = page.evaluate(
            """() => {
                const t = document.querySelector('#toast');
                if (t && t.classList.contains('err')) {
                    return (t.innerText || t.textContent || '').trim();
                }
                return null;
            }"""
        )
    except Exception:
        toast_err = None
    return str(toast_err or "").strip()


def _watch_for_manual_submit_signal(
    page: Page,
    cfg: Dict[str, Any],
    *,
    episode_id: str = "",
    complete_sel: str = "",
    modal_sel: str = "",
    trigger_reason: str = "",
) -> Dict[str, Any]:
    enabled = bool(_cfg_get(cfg, "run.submit_manual_watch_enabled", False))
    timeout_sec = max(
        0.0, float(_cfg_get(cfg, "run.submit_manual_watch_timeout_sec", 180.0) or 180.0)
    )
    poll_ms = max(
        150, int(_cfg_get(cfg, "run.submit_manual_watch_poll_ms", 500) or 500)
    )
    log_interval_sec = max(
        1.0,
        float(
            _cfg_get(cfg, "run.submit_manual_watch_log_interval_sec", 10.0) or 10.0
        ),
    )
    result = {
        "enabled": enabled,
        "detected": False,
        "timed_out": False,
        "signal": "",
        "last_error": "",
        "elapsed_sec": 0.0,
        "page_url": _safe_page_url(page),
        "trigger_reason": str(trigger_reason or "").strip(),
    }
    if not enabled or timeout_sec <= 0:
        return result

    print(
        "[atlas] auto-submit needs manual confirmation: "
        f"reason={result['trigger_reason'] or 'unknown'}; "
        f"waiting up to {timeout_sec:.1f}s for operator submit click while monitoring the page."
    )
    _capture_submit_step(
        page,
        cfg,
        "manual_submit_watch_started",
        task_id=episode_id,
        include_html=True,
    )

    started_at = time.time()
    deadline = started_at + timeout_sec
    next_log_at = started_at + log_interval_sec

    while time.time() < deadline:
        ui_error = _read_submit_ui_validation_error(page)
        if ui_error:
            result["detected"] = True
            result["signal"] = "ui_validation_error"
            result["last_error"] = ui_error
            result["page_url"] = _safe_page_url(page)
            result["elapsed_sec"] = round(time.time() - started_at, 2)
            print(f"[atlas] manual submit watch detected UI validation error: {ui_error}")
            _capture_submit_step(
                page,
                cfg,
                "manual_submit_watch_ui_error",
                task_id=episode_id,
                include_html=True,
            )
            return result

        if _submit_transition_observed(page):
            result["detected"] = True
            result["signal"] = "post_submit_transition"
            result["page_url"] = _safe_page_url(page)
            result["elapsed_sec"] = round(time.time() - started_at, 2)
            print(
                "[atlas] manual submit watch detected post-submit transition "
                f"after {result['elapsed_sec']:.1f}s."
            )
            _capture_submit_step(
                page,
                cfg,
                "manual_submit_watch_transition",
                task_id=episode_id,
                include_html=True,
            )
            return result

        current_modal = (
            _browser._first_visible_locator(page, modal_sel, timeout_ms=250)
            if modal_sel
            else None
        )
        if current_modal is not None:
            result["detected"] = True
            result["signal"] = "submit_modal_visible"
            result["page_url"] = _safe_page_url(page)
            result["elapsed_sec"] = round(time.time() - started_at, 2)
            print(
                "[atlas] manual submit watch detected submit confirmation modal "
                f"after {result['elapsed_sec']:.1f}s."
            )
            _capture_submit_step(
                page,
                cfg,
                "manual_submit_watch_modal",
                task_id=episode_id,
                include_html=True,
            )
            return result

        if _submit_confirmation_signal_visible(page):
            result["detected"] = True
            result["signal"] = "submit_confirmation_signal"
            result["page_url"] = _safe_page_url(page)
            result["elapsed_sec"] = round(time.time() - started_at, 2)
            print(
                "[atlas] manual submit watch detected confirmation signal "
                f"after {result['elapsed_sec']:.1f}s."
            )
            _capture_submit_step(
                page,
                cfg,
                "manual_submit_watch_confirmation_signal",
                task_id=episode_id,
                include_html=True,
            )
            return result

        now = time.time()
        if now >= next_log_at:
            print(
                "[atlas] waiting for manual submit click... "
                f"elapsed={now - started_at:.1f}s "
                f"url={_safe_page_url(page) or '<unknown>'} "
                f"selector={complete_sel or '<none>'}"
            )
            next_log_at = now + log_interval_sec

        try:
            page.wait_for_timeout(poll_ms)
        except Exception:
            break

    result["timed_out"] = True
    result["page_url"] = _safe_page_url(page)
    result["elapsed_sec"] = round(time.time() - started_at, 2)
    print(
        "[atlas] manual submit watch timed out without a confirmed signal "
        f"after {result['elapsed_sec']:.1f}s."
    )
    _capture_submit_step(
        page,
        cfg,
        "manual_submit_watch_timeout",
        task_id=episode_id,
        include_html=True,
    )
    return result


def _submit_episode(
    page: Page,
    cfg: Dict[str, Any],
    *,
    episode_id: str = "",
    return_details: bool = False,
    _allow_manual_submit_watch: bool = True,
    _skip_initial_click: bool = False,
) -> bool | Dict[str, Any]:
    if not episode_id:
        episode_id = _task_id_from_url(page.url)
    complete_sel = str(_cfg_get(cfg, "atlas.selectors.complete_button", "")).strip()
    modal_sel = str(_cfg_get(cfg, "atlas.selectors.quality_review_modal", "")).strip()
    submit_status = _new_submit_status(page)

    modal_open = bool(modal_sel and _browser._first_visible_locator(page, modal_sel, timeout_ms=900) is not None)
    submit_status["submit_modal_already_open"] = modal_open
    submit_status["submit_attempted"] = bool(modal_open or complete_sel or _skip_initial_click)
    completed = bool(modal_open or _skip_initial_click)
    if _skip_initial_click:
        submit_status["complete_button_clicked"] = True
    transition_observed = False

    if not modal_open and complete_sel and not _skip_initial_click:
        _browser._dismiss_blocking_modals(page)
        _browser._dismiss_blocking_side_panel(page, cfg, aggressive=True)
        completed = _browser._safe_locator_click(page, complete_sel, timeout_ms=7000)
        submit_status["complete_button_clicked"] = bool(completed)
        if not completed:
            completed = _force_primary_submit_click(page, complete_sel)
            submit_status["complete_button_clicked"] = bool(completed)
        if completed:
            print("[atlas] clicked Complete button.")
            _capture_submit_step(page, cfg, "after_complete_click_attempt", task_id=episode_id, include_html=True)
            page.wait_for_timeout(900)
            for _ in range(8):
                # Check for UI Toast Errors (Validation Failures)
                toast_err = _read_submit_ui_validation_error(page)
                if toast_err:
                    print(f"[atlas] UI submission blocked: {toast_err}")
                    submit_status["last_error"] = f"UI Validation: {toast_err}"
                    submit_status["submit_verified"] = False
                    if return_details:
                        return submit_status
                    return False

                if _submit_transition_observed(page):
                    transition_observed = True
                    submit_status["saw_post_submit_transition"] = True
                    submit_status["submit_verified"] = True
                    submit_status["submit_verification_reason"] = "post_submit_transition_observed"
                    _normalize_post_submit_tasks_view(page, cfg, task_id=episode_id)
                    submit_status["page_url_after_submit"] = _safe_page_url(page)
                    print(
                        "[atlas] submit verified: reason=post_submit_transition_observed "
                        f"url={submit_status['page_url_after_submit'] or '<unknown>'}"
                    )
                    if return_details:
                        return submit_status
                    return True
                current_modal = _browser._first_visible_locator(page, modal_sel, timeout_ms=350) if modal_sel else None
                if current_modal is not None or _submit_confirmation_signal_visible(page):
                    break
                page.wait_for_timeout(350)
            if not _submit_transition_observed(page) and not _submit_confirmation_signal_visible(page):
                if _force_primary_submit_click(page, complete_sel):
                    submit_status["complete_button_retried"] = True
                    print("[atlas] retried Complete button after no post-click transition.")
                    _capture_submit_step(page, cfg, "after_complete_retry_attempt", task_id=episode_id, include_html=True)
                    page.wait_for_timeout(1200)
            else:
                transition_observed = True

    if not completed and not modal_open:
        manual_watch = (
            _watch_for_manual_submit_signal(
                page,
                cfg,
                episode_id=episode_id,
                complete_sel=complete_sel,
                modal_sel=modal_sel,
                trigger_reason="complete_click_failed",
            )
            if _allow_manual_submit_watch
            else {"enabled": False, "detected": False, "timed_out": False, "signal": "", "last_error": "", "elapsed_sec": 0.0, "page_url": _safe_page_url(page), "trigger_reason": "complete_click_failed"}
        )
        if manual_watch.get("enabled", False):
            submit_status["manual_submit_watch_used"] = True
            submit_status["manual_submit_watch_reason"] = str(manual_watch.get("trigger_reason", "") or "").strip()
            submit_status["manual_submit_watch_signal"] = str(manual_watch.get("signal", "") or "").strip()
            submit_status["manual_submit_watch_timed_out"] = bool(manual_watch.get("timed_out", False))
            submit_status["manual_submit_watch_elapsed_sec"] = float(manual_watch.get("elapsed_sec", 0.0) or 0.0)
            submit_status["manual_submit_detected"] = bool(manual_watch.get("detected", False))
        if manual_watch.get("last_error"):
            submit_status["last_error"] = f"UI Validation: {manual_watch['last_error']}"
            submit_status["page_url_after_submit"] = str(manual_watch.get("page_url", "") or _safe_page_url(page)).strip()
            submit_status["submit_verification_reason"] = "manual_submit_ui_validation_error"
            if return_details:
                return submit_status
            return False
        if manual_watch.get("detected", False):
            signal = str(manual_watch.get("signal", "") or "").strip()
            submit_status["complete_button_clicked"] = True
            if signal == "post_submit_transition":
                transition_observed = True
                submit_status["saw_post_submit_transition"] = True
                submit_status["submit_verified"] = True
                submit_status["submit_verification_reason"] = "post_submit_transition_observed"
                _normalize_post_submit_tasks_view(page, cfg, task_id=episode_id)
                submit_status["page_url_after_submit"] = _safe_page_url(page)
                if return_details:
                    return submit_status
                return True
            manual_result = _submit_episode(
                page,
                cfg,
                episode_id=episode_id,
                return_details=True,
                _allow_manual_submit_watch=False,
                _skip_initial_click=True,
            )
            if isinstance(manual_result, dict):
                manual_result["manual_submit_watch_used"] = True
                manual_result["manual_submit_detected"] = True
                manual_result["manual_submit_watch_reason"] = str(manual_watch.get("trigger_reason", "") or "").strip()
                manual_result["manual_submit_watch_signal"] = signal
                manual_result["manual_submit_watch_timed_out"] = False
                manual_result["manual_submit_watch_elapsed_sec"] = float(manual_watch.get("elapsed_sec", 0.0) or 0.0)
                if return_details:
                    return manual_result
                return bool(manual_result.get("submit_verified", False))
            if return_details:
                return submit_status
            return bool(manual_result)
        submit_status["page_url_after_submit"] = _safe_page_url(page)
        submit_status["submit_verification_reason"] = (
            "manual_submit_watch_timeout"
            if manual_watch.get("enabled", False) and manual_watch.get("timed_out", False)
            else "complete_click_failed"
        )
        if return_details:
            return submit_status
        return False

    no_edits_result = _handle_no_edits_modal(
        page,
        cfg,
        timeout_ms=8000,
        task_id=episode_id,
        return_details=True,
    )
    no_edits_ok, no_edits_seen = (
        no_edits_result if isinstance(no_edits_result, tuple) else (bool(no_edits_result), False)
    )
    submit_status["saw_no_edits_modal"] = bool(no_edits_seen)
    submit_status["no_edits_confirmed"] = bool(no_edits_ok and no_edits_seen)
    if not no_edits_ok:
        if _submit_transition_observed(page):
            transition_observed = True
            submit_status["saw_post_submit_transition"] = True
            submit_status["submit_verified"] = True
            submit_status["submit_verification_reason"] = "post_submit_transition_observed"
            _normalize_post_submit_tasks_view(page, cfg, task_id=episode_id)
            submit_status["page_url_after_submit"] = _safe_page_url(page)
            print(
                "[atlas] submit verified: reason=post_submit_transition_observed "
                f"url={submit_status['page_url_after_submit'] or '<unknown>'}"
            )
            if return_details:
                return submit_status
            return True
        if _allow_manual_submit_watch:
            manual_watch = _watch_for_manual_submit_signal(
                page,
                cfg,
                episode_id=episode_id,
                complete_sel=complete_sel,
                modal_sel=modal_sel,
                trigger_reason="no_edits_modal_timeout",
            )
            submit_status["manual_submit_watch_used"] = bool(manual_watch.get("enabled", False))
            submit_status["manual_submit_watch_reason"] = str(manual_watch.get("trigger_reason", "") or "").strip()
            submit_status["manual_submit_watch_signal"] = str(manual_watch.get("signal", "") or "").strip()
            submit_status["manual_submit_watch_timed_out"] = bool(manual_watch.get("timed_out", False))
            submit_status["manual_submit_watch_elapsed_sec"] = float(manual_watch.get("elapsed_sec", 0.0) or 0.0)
            submit_status["manual_submit_detected"] = bool(manual_watch.get("detected", False))
            if manual_watch.get("last_error"):
                submit_status["last_error"] = f"UI Validation: {manual_watch['last_error']}"
                submit_status["page_url_after_submit"] = str(manual_watch.get("page_url", "") or _safe_page_url(page)).strip()
                submit_status["submit_verification_reason"] = "manual_submit_ui_validation_error"
                if return_details:
                    return submit_status
                return False
            if manual_watch.get("detected", False):
                signal = str(manual_watch.get("signal", "") or "").strip()
                if signal == "post_submit_transition":
                    transition_observed = True
                    submit_status["saw_post_submit_transition"] = True
                    submit_status["submit_verified"] = True
                    submit_status["submit_verification_reason"] = "post_submit_transition_observed"
                    _normalize_post_submit_tasks_view(page, cfg, task_id=episode_id)
                    submit_status["page_url_after_submit"] = _safe_page_url(page)
                    if return_details:
                        return submit_status
                    return True
                manual_result = _submit_episode(
                    page,
                    cfg,
                    episode_id=episode_id,
                    return_details=True,
                    _allow_manual_submit_watch=False,
                    _skip_initial_click=True,
                )
                if isinstance(manual_result, dict):
                    manual_result["manual_submit_watch_used"] = True
                    manual_result["manual_submit_detected"] = True
                    manual_result["manual_submit_watch_reason"] = str(manual_watch.get("trigger_reason", "") or "").strip()
                    manual_result["manual_submit_watch_signal"] = signal
                    manual_result["manual_submit_watch_timed_out"] = False
                    manual_result["manual_submit_watch_elapsed_sec"] = float(manual_watch.get("elapsed_sec", 0.0) or 0.0)
                    if return_details:
                        return manual_result
                    return bool(manual_result.get("submit_verified", False))
        print("[atlas] timed out waiting for 'No Edits Made' confirmation.")
        submit_status["page_url_after_submit"] = _safe_page_url(page)
        submit_status["submit_verification_reason"] = (
            "manual_submit_watch_timeout"
            if submit_status.get("manual_submit_watch_timed_out", False)
            else "no_edits_modal_timeout"
        )
        if return_details:
            return submit_status
        return False

    quality_result = _handle_quality_review_modal(
        page,
        cfg,
        timeout_ms=9000,
        task_id=episode_id,
        return_details=True,
    )
    reviewed, quality_seen = (
        quality_result if isinstance(quality_result, tuple) else (bool(quality_result), False)
    )
    submit_status["saw_quality_review_modal"] = bool(quality_seen)
    submit_status["quality_review_confirmed"] = bool(reviewed and quality_seen)
    if not reviewed and _submit_transition_observed(page):
        reviewed = True
        transition_observed = True
    transition_observed = transition_observed or _submit_transition_observed(page)
    submit_status["saw_post_submit_transition"] = bool(transition_observed)
    verification_evidence = transition_observed or no_edits_seen or quality_seen
    if not verification_evidence:
        final_grace_sec = max(0.0, float(_cfg_get(cfg, "run.submit_verification_grace_sec", 12.0) or 12.0))
        if final_grace_sec > 0:
            if _wait_for_submit_transition(page, timeout_ms=int(final_grace_sec * 1000.0), poll_ms=500):
                transition_observed = True
                submit_status["saw_post_submit_transition"] = True
                verification_evidence = True
    if transition_observed:
        _normalize_post_submit_tasks_view(page, cfg, task_id=episode_id)
    submit_status["page_url_after_submit"] = _safe_page_url(page)
    if not verification_evidence:
        if _allow_manual_submit_watch:
            manual_watch = _watch_for_manual_submit_signal(
                page,
                cfg,
                episode_id=episode_id,
                complete_sel=complete_sel,
                modal_sel=modal_sel,
                trigger_reason="missing_verification_evidence",
            )
            submit_status["manual_submit_watch_used"] = bool(manual_watch.get("enabled", False))
            submit_status["manual_submit_watch_reason"] = str(manual_watch.get("trigger_reason", "") or "").strip()
            submit_status["manual_submit_watch_signal"] = str(manual_watch.get("signal", "") or "").strip()
            submit_status["manual_submit_watch_timed_out"] = bool(manual_watch.get("timed_out", False))
            submit_status["manual_submit_watch_elapsed_sec"] = float(manual_watch.get("elapsed_sec", 0.0) or 0.0)
            submit_status["manual_submit_detected"] = bool(manual_watch.get("detected", False))
            if manual_watch.get("last_error"):
                submit_status["last_error"] = f"UI Validation: {manual_watch['last_error']}"
                submit_status["page_url_after_submit"] = str(manual_watch.get("page_url", "") or _safe_page_url(page)).strip()
                submit_status["submit_verification_reason"] = "manual_submit_ui_validation_error"
                if return_details:
                    return submit_status
                return False
            if manual_watch.get("detected", False):
                signal = str(manual_watch.get("signal", "") or "").strip()
                if signal == "post_submit_transition":
                    submit_status["saw_post_submit_transition"] = True
                    submit_status["submit_verified"] = True
                    submit_status["submit_verification_reason"] = "post_submit_transition_observed"
                    _normalize_post_submit_tasks_view(page, cfg, task_id=episode_id)
                    submit_status["page_url_after_submit"] = _safe_page_url(page)
                    if return_details:
                        return submit_status
                    return True
                manual_result = _submit_episode(
                    page,
                    cfg,
                    episode_id=episode_id,
                    return_details=True,
                    _allow_manual_submit_watch=False,
                    _skip_initial_click=True,
                )
                if isinstance(manual_result, dict):
                    manual_result["manual_submit_watch_used"] = True
                    manual_result["manual_submit_detected"] = True
                    manual_result["manual_submit_watch_reason"] = str(manual_watch.get("trigger_reason", "") or "").strip()
                    manual_result["manual_submit_watch_signal"] = signal
                    manual_result["manual_submit_watch_timed_out"] = False
                    manual_result["manual_submit_watch_elapsed_sec"] = float(manual_watch.get("elapsed_sec", 0.0) or 0.0)
                    if return_details:
                        return manual_result
                    return bool(manual_result.get("submit_verified", False))
        print("[atlas] submit not verified: no post-submit transition or confirmation modal observed.")
        submit_status["submit_verification_reason"] = (
            "manual_submit_watch_timeout"
            if submit_status.get("manual_submit_watch_timed_out", False)
            else "missing_verification_evidence"
        )
        if return_details:
            return submit_status
        return False

    if transition_observed:
        submit_status["submit_verified"] = bool(completed)
    elif quality_seen and reviewed:
        submit_status["submit_verified"] = bool(completed)
    elif no_edits_seen and no_edits_ok:
        submit_status["submit_verified"] = bool(completed)
    else:
        submit_status["submit_verified"] = bool(completed and reviewed)
    if transition_observed:
        submit_status["submit_verification_reason"] = "post_submit_transition_observed"
    elif quality_seen and reviewed:
        submit_status["submit_verification_reason"] = "quality_review_confirmed"
    elif no_edits_seen and no_edits_ok:
        submit_status["submit_verification_reason"] = "no_edits_modal_confirmed"
    else:
        submit_status["submit_verification_reason"] = "submit_verified"

    if submit_status["submit_verified"]:
        # NEW: Deep verification via Audit Dashboard check
        verify_dashboard = bool(_cfg_get(cfg, "run.submit_deep_verify_dashboard", False))
        if verify_dashboard:
            verify_timeout_sec = max(
                3.0,
                float(_cfg_get(cfg, "run.submit_deep_verify_dashboard_timeout_sec", 15.0) or 15.0),
            )
            print(f"[atlas] deep verification enabled: checking dashboard for {episode_id or 'unknown'}")
            verify_res = _submit_verify.verify_episode_on_dashboard(
                page,
                cfg,
                episode_id=episode_id,
                timeout_sec=verify_timeout_sec,
            )
            submit_status["dashboard_verified"] = verify_res.verified
            submit_status["dashboard_verify_method"] = verify_res.method
            if not verify_res.verified:
                print(f"[atlas] WARNING: submission verified on page but NOT FOUND on dashboard: {verify_res.detail}")

    if return_details:
        return submit_status
    return bool(submit_status["submit_verified"])


def _evaluate_apply_submit_guard(
    *,
    total_targets: int,
    applied: int,
    skipped_unchanged: int,
    failed: List[str],
    submit_guard_enabled: bool,
    submit_guard_max_failure_ratio: float,
    submit_guard_min_applied_ratio: float,
    submit_guard_block_on_budget_exceeded: bool,
) -> List[str]:
    submit_guard_reasons: List[str] = []
    budget_exceeded = any("apply budget exceeded" in str(msg).lower() for msg in failed)
    total_targets_safe = max(1, int(total_targets))
    failure_ratio = float(len(failed)) / float(total_targets_safe)
    covered_ratio = float(applied + skipped_unchanged) / float(total_targets_safe)
    if submit_guard_enabled and total_targets > 0:
        if submit_guard_block_on_budget_exceeded and budget_exceeded:
            submit_guard_reasons.append("apply budget exceeded")
        if failure_ratio > submit_guard_max_failure_ratio:
            submit_guard_reasons.append(
                f"failure ratio {failure_ratio:.1%} > {submit_guard_max_failure_ratio:.1%}"
            )
        if covered_ratio < submit_guard_min_applied_ratio:
            submit_guard_reasons.append(
                f"covered ratio {covered_ratio:.1%} < {submit_guard_min_applied_ratio:.1%}"
            )
    return submit_guard_reasons


def apply_labels(
    page: Page,
    cfg: Dict[str, Any],
    label_map: Dict[int, str],
    *,
    episode_id: str = "",
    segment_plan: Optional[Dict[int, Dict[str, Any]]] = None,
    source_segments: Optional[List[Dict[str, Any]]] = None,
    heartbeat: Optional[Callable[[], None]] = None,
    validation_tracker: Optional["ValidationTracker"] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    def _heartbeat() -> None:
        if callable(heartbeat):
            try:
                heartbeat()
            except Exception:
                pass

    def _artifact_paths(artifact: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        if not isinstance(artifact, dict):
            return "", ""
        return (
            str(artifact.get("screenshot", "") or "").strip(),
            str(artifact.get("html", "") or "").strip(),
        )

    def _emit_progress(event_type: str, **payload: Any) -> None:
        if not callable(progress_callback):
            return
        event_payload = {
            "episode_id": str(episode_id or "").strip(),
            "apply_budget_state": apply_budget.to_dict() if "apply_budget" in locals() else {},
            "failed_count": len(failed) if "failed" in locals() else 0,
            "processed": int(processed) if "processed" in locals() else 0,
            "total_targets": int(total_targets) if "total_targets" in locals() else 0,
            **payload,
        }
        try:
            progress_callback(event_type, event_payload)
        except Exception:
            pass

    _browser._dismiss_blocking_modals(page)
    _browser._dismiss_blocking_side_panel(page, cfg, aggressive=True)
    _heartbeat()

    rows_sel = str(_cfg_get(cfg, "atlas.selectors.segment_rows", ""))
    label_sel = str(_cfg_get(cfg, "atlas.selectors.segment_label", ""))
    edit_sel = str(_cfg_get(cfg, "atlas.selectors.edit_button_in_row", ""))
    input_sel = str(_cfg_get(cfg, "atlas.selectors.label_input", ""))
    save_sel = str(_cfg_get(cfg, "atlas.selectors.save_button", ""))
    complete_sel = str(_cfg_get(cfg, "atlas.selectors.complete_button", ""))
    skip_unchanged = bool(_cfg_get(cfg, "run.skip_unchanged_labels", True))
    progress_every = max(1, int(_cfg_get(cfg, "run.label_apply_progress_every", 5)))
    configured_max_total_sec = max(30.0, float(_cfg_get(cfg, "run.label_apply_max_total_sec", 36000) or 36000))
    dynamic_budget_enabled = bool(_cfg_get(cfg, "run.label_apply_dynamic_budget_enabled", True))
    dynamic_budget_floor_sec = max(
        30.0,
        float(_cfg_get(cfg, "run.label_apply_dynamic_budget_floor_sec", 420.0) or 420.0),
    )
    dynamic_budget_base_sec = max(
        0.0,
        float(_cfg_get(cfg, "run.label_apply_dynamic_budget_base_sec", 120.0) or 120.0),
    )
    dynamic_budget_per_target_sec = max(
        0.0,
        float(_cfg_get(cfg, "run.label_apply_dynamic_budget_per_target_sec", 12.0) or 12.0),
    )
    no_progress_timeout_sec = max(
        10.0,
        float(_cfg_get(cfg, "run.label_apply_no_progress_timeout_sec", 90.0) or 90.0),
    )
    max_consecutive_row_failures = max(
        1,
        int(_cfg_get(cfg, "run.label_apply_max_consecutive_row_failures", 3) or 3),
    )
    max_failures = max(1, int(_cfg_get(cfg, "run.label_apply_max_failures", 18)))
    input_timeout_ms = max(800, int(_cfg_get(cfg, "run.label_apply_input_timeout_ms", 3000)))
    save_timeout_ms = max(300, int(_cfg_get(cfg, "run.label_apply_save_timeout_ms", 1800)))
    edit_click_timeout_ms = max(400, int(_cfg_get(cfg, "run.label_apply_edit_click_timeout_ms", 900)))
    submit_guard_enabled = bool(_cfg_get(cfg, "run.submit_guard_enabled", True))
    submit_guard_max_failure_ratio = min(
        1.0, max(0.0, float(_cfg_get(cfg, "run.submit_guard_max_failure_ratio", 0.25)))
    )
    submit_guard_min_applied_ratio = min(
        1.0, max(0.3, float(_cfg_get(cfg, "run.submit_guard_min_applied_ratio", 0.90)))
    )
    submit_guard_block_on_budget_exceeded = bool(
        _cfg_get(cfg, "run.submit_guard_block_on_budget_exceeded", True)
    )

    best_rows_sel, rows = _resolve_rows_locator(page, rows_sel)
    failed: List[str] = []
    failed_details: Dict[int, str] = {}
    applied = 0
    skipped_unchanged = 0
    total_targets = len(label_map)
    max_total_sec = configured_max_total_sec
    if dynamic_budget_enabled:
        computed_budget_sec = max(
            dynamic_budget_floor_sec,
            dynamic_budget_base_sec + (dynamic_budget_per_target_sec * float(total_targets)),
        )
        max_total_sec = max(configured_max_total_sec, computed_budget_sec)
    started_at = time.time()
    apply_budget = ApplyBudgetState(
        target_count=total_targets,
        applied_count=0,
        last_progress_at=started_at,
        deadline_at=started_at + float(max_total_sec),
        started_at=started_at,
    )
    processed = 0
    _emit_progress(
        "apply_start",
        no_progress_timeout_sec=float(no_progress_timeout_sec),
        budget_sec=float(max_total_sec),
    )
    if total_targets > 0:
        print(
            f"[run] apply labels started: targets={total_targets} "
            f"(budget={max_total_sec:.1f}s, no_progress_timeout={no_progress_timeout_sec:.1f}s)"
        )

    for idx in sorted(label_map):
        _heartbeat()
        processed += 1
        loop_now = time.time()
        elapsed = apply_budget.elapsed_sec(now=loop_now)
        stalled_for_sec = apply_budget.stalled_for_sec(now=loop_now)
        if loop_now > float(apply_budget.deadline_at) and stalled_for_sec >= no_progress_timeout_sec:
            apply_budget.mark_timed_out()
            failed.append(
                f"apply budget exceeded after {elapsed:.1f}s without progress for {stalled_for_sec:.1f}s "
                f"(processed={processed - 1}/{total_targets})"
            )
            print(
                f"[run] apply labels stopped: exceeded {max_total_sec:.1f}s budget "
                f"with no progress for {stalled_for_sec:.1f}s after {processed - 1}/{total_targets} segments."
            )
            _emit_progress(
                "apply_stop",
                stop_reason="budget_exceeded",
                stalled_for_sec=float(stalled_for_sec),
                elapsed_sec=float(elapsed),
            )
            break
        if stalled_for_sec >= no_progress_timeout_sec:
            apply_budget.mark_timed_out()
            failed.append(
                f"apply stalled for {stalled_for_sec:.1f}s without progress "
                f"(processed={processed - 1}/{total_targets})"
            )
            print(
                f"[run] apply labels stopped: no progress for {stalled_for_sec:.1f}s "
                f"after {processed - 1}/{total_targets} segments."
            )
            _emit_progress(
                "apply_stop",
                stop_reason="no_progress_timeout",
                stalled_for_sec=float(stalled_for_sec),
                elapsed_sec=float(elapsed),
            )
            break
        if len(failed) >= max_failures:
            apply_budget.mark_failed()
            print(
                f"[run] apply labels stopped: failure limit {max_failures} reached "
                f"(processed={processed - 1}/{total_targets})."
            )
            _emit_progress(
                "apply_stop",
                stop_reason="max_failures",
                elapsed_sec=float(elapsed),
            )
            break
        if apply_budget.consecutive_failures >= max_consecutive_row_failures:
            apply_budget.mark_failed()
            failed.append(
                f"apply stopped after {apply_budget.consecutive_failures} consecutive row failures "
                f"(processed={processed - 1}/{total_targets})"
            )
            print(
                f"[run] apply labels stopped: consecutive row failure limit "
                f"{max_consecutive_row_failures} reached after {processed - 1}/{total_targets} segments."
            )
            _emit_progress(
                "apply_stop",
                stop_reason="consecutive_failures",
                elapsed_sec=float(elapsed),
            )
            break
        rows = page.locator(best_rows_sel)
        count = rows.count()
        if idx > count:
            message = f"segment {idx}: row missing (count={count})"
            failed.append(message)
            failed_details[idx] = message
            apply_budget.mark_failure()
            _emit_progress(
                "apply_failure",
                segment_index=int(idx),
                message=message,
            )
            if processed % progress_every == 0:
                print(
                    f"[run] apply progress {processed}/{total_targets} "
                    f"(applied={applied}, skipped={skipped_unchanged}, failed={len(failed)})"
                )
                _heartbeat()
            continue
        row = rows.nth(idx - 1)
        label = label_map[idx]
        try:
            _browser._dismiss_blocking_modals(page)
            _browser._dismiss_blocking_side_panel(page, cfg, aggressive=True)
            _heartbeat()
            _browser._click_segment_row_with_recovery(page, rows, idx, cfg)
            row = page.locator(best_rows_sel).nth(idx - 1)
            if skip_unchanged and label_sel:
                current_label = _first_text_from_row(row, label_sel)
                if current_label:
                    if _normalize_label_for_compare(current_label) == _normalize_label_for_compare(label):
                        skipped_unchanged += 1
                        apply_budget.mark_progress(0, skipped_delta=1)
                        _emit_progress(
                            "apply_row",
                            segment_index=int(idx),
                            outcome="skipped_unchanged",
                        )
                        continue

            input_loc = None
            try:
                page.keyboard.press("e")
                input_loc = _browser._first_visible_locator(page, input_sel, timeout_ms=min(1200, input_timeout_ms))
            except Exception:
                input_loc = None

            if input_loc is None:
                try:
                    row.dblclick(timeout=max(500, edit_click_timeout_ms - 300))
                except Exception:
                    pass
                input_loc = _browser._first_visible_locator(page, input_sel, timeout_ms=min(1200, input_timeout_ms))

            if input_loc is None:
                for candidate in _browser._selector_variants(edit_sel):
                    edit_loc = row.locator(candidate).first
                    if edit_loc.count() > 0 and edit_loc.is_visible():
                        try:
                            edit_loc.click(timeout=edit_click_timeout_ms, no_wait_after=True)
                        except Exception:
                            _browser._dismiss_blocking_side_panel(page, cfg, aggressive=True)
                            edit_loc.click(
                                timeout=max(400, edit_click_timeout_ms - 300),
                                force=True,
                                no_wait_after=True,
                            )
                        input_loc = _browser._first_visible_locator(
                            page,
                            input_sel,
                            timeout_ms=min(1800, input_timeout_ms),
                        )
                        if input_loc is not None:
                            break

            if input_loc is None:
                input_loc = _browser._first_visible_locator(page, input_sel, timeout_ms=input_timeout_ms)
            if input_loc is None:
                raise RuntimeError("label input not found")
            _fill_input(input_loc, label, page)
            _heartbeat()

            saved = _browser._safe_locator_click(page, save_sel, timeout_ms=save_timeout_ms) if save_sel else False
            if not saved:
                for candidate in _browser._selector_variants(save_sel):
                    btn = _browser._first_visible_locator(page, candidate, timeout_ms=max(300, save_timeout_ms // 2))
                    if btn is None:
                        continue
                    try:
                        btn.click(timeout=max(300, save_timeout_ms // 2), force=True, no_wait_after=True)
                        saved = True
                        break
                    except Exception:
                        continue
            if not saved:
                page.keyboard.press("Control+Enter")

            applied += 1
            apply_budget.mark_progress(1)
            _emit_progress(
                "apply_row",
                segment_index=int(idx),
                outcome="applied",
            )
            time.sleep(0.15)
            _heartbeat()
        except Exception as exc:
            message = f"segment {idx}: {_short_error_text(exc)}"
            failed.append(message)
            failed_details[idx] = message
            apply_budget.mark_failure()
            _emit_progress(
                "apply_failure",
                segment_index=int(idx),
                message=message,
            )
        if processed % progress_every == 0:
            print(
                f"[run] apply progress {processed}/{total_targets} "
                f"(applied={applied}, skipped={skipped_unchanged}, failed={len(failed)})"
            )
            _heartbeat()
            _emit_progress("apply_progress")

    submit_guard_reasons = _evaluate_apply_submit_guard(
        total_targets=total_targets,
        applied=applied,
        skipped_unchanged=skipped_unchanged,
        failed=failed,
        submit_guard_enabled=submit_guard_enabled,
        submit_guard_max_failure_ratio=submit_guard_max_failure_ratio,
        submit_guard_min_applied_ratio=submit_guard_min_applied_ratio,
        submit_guard_block_on_budget_exceeded=submit_guard_block_on_budget_exceeded,
    )

    if not submit_guard_reasons:
        if isinstance(segment_plan, dict) and isinstance(source_segments, list):
            try:
                from src.rules.consistency import validate_pre_submit_consistency

                consistency = validate_pre_submit_consistency(
                    page,
                    cfg,
                    segment_plan,
                    source_segments,
                )
                for warning in consistency.get("warnings", [])[:10]:
                    print(f"[guard] consistency warning: {warning}")
                if not consistency.get("consistent", True):
                    submit_guard_reasons.append("pre-submit consistency mismatch")
                    for message in consistency.get("mismatches", [])[:10]:
                        submit_guard_reasons.append(message)
            except ImportError:
                pass

        max_segment_duration_sec = max(
            0.1,
            float(_cfg_get(cfg, "run.max_segment_duration_sec", 10.0) or 10.0),
        )
        duration_guard = _pre_submit_duration_check(
            page,
            cfg,
            max_dur=max_segment_duration_sec,
        )
        if not duration_guard.get("ok", True):
            submit_guard_reasons.append("live DOM contains overlong segments before submit")
            submit_guard_reasons.extend(duration_guard.get("violations", [])[:10])

    submit_result_stub = {
        "submit_guard_blocked": bool(submit_guard_reasons),
        "submit_guard_reasons": list(submit_guard_reasons),
    }
    if validation_tracker is not None:
        submit_before_artifact = _capture_submit_step(
            page,
            cfg,
            "before_submit_attempt",
            task_id=episode_id,
            include_html=True,
        )
        before_screenshot, before_html = _artifact_paths(submit_before_artifact)
        validation_tracker.record_submit_before(
            _safe_page_url(page),
            complete_sel,
            screenshot_path=before_screenshot,
            html_path=before_html,
        )

    if submit_guard_reasons:
        print("[run] submit guard blocked auto-submit for this episode:")
        for reason in submit_guard_reasons:
            print(f"  - {reason}")
        submit_status = _new_submit_status(page, reason="submit_guard_blocked")
        submit_status["submit_guard_blocked"] = True
        submit_status["submit_guard_reasons"] = list(submit_guard_reasons)
        completed = False
        apply_budget.mark_failed()
        if validation_tracker is not None:
            blocked_artifact = _capture_submit_step(
                page,
                cfg,
                "submit_guard_blocked",
                task_id=episode_id,
                include_html=True,
            )
            blocked_screenshot, blocked_html = _artifact_paths(blocked_artifact)
            validation_tracker.record_submit_after(
                submit_status,
                submit_result_stub,
                screenshot_after_verify=blocked_screenshot,
                html_after_verify=blocked_html,
            )
    else:
        _emit_progress("submit_verifying")
        _respect_major_step_pause(cfg, "submit_episode", heartbeat=_heartbeat)
        if complete_sel:
            # Shift to robust centralized verification to avoid false successes
            verify_timeout = float(_cfg_get(cfg, "run.submit_verification_timeout_sec", 45.0))
            submit_res = _browser._click_submit_with_verification(
                page, cfg, task_id=episode_id, verify_timeout_sec=verify_timeout
            )
            submit_status = submit_res.get("verification", {})
            completed = bool(
                submit_status.get("verified", submit_status.get("submit_verified", False))
            )
        else:
            submit_status = {"verified": False, "method": "selector_missing", "detail": "Complete selector not found"}
            completed = False
        normalized_submit = SubmitOutcome.from_status(
            submit_status,
            terminal_failure=not bool(completed),
        )
        if "submit_verified" not in submit_status:
            normalized_submit.submit_verified = bool(submit_status.get("verified", False))
        if not normalized_submit.submit_verification_reason:
            normalized_submit.submit_verification_reason = str(
                submit_status.get("submit_verification_reason", "")
                or submit_status.get("method", "")
                or "submit_unverified"
            ).strip()
        if not normalized_submit.last_error and not normalized_submit.submit_verified:
            normalized_submit.last_error = str(submit_status.get("detail", "") or "").strip()
        if not normalized_submit.page_url_before_submit:
            normalized_submit.page_url_before_submit = _safe_page_url(page)
        if not normalized_submit.page_url_after_submit:
            normalized_submit.page_url_after_submit = _safe_page_url(page)
            submit_status = normalized_submit.to_dict()
        if validation_tracker is not None:
            verified_artifact = _capture_submit_step(
                page,
                cfg,
                "after_submit_verification",
                task_id=episode_id,
                include_html=True,
            )
            verified_screenshot, verified_html = _artifact_paths(verified_artifact)
            validation_tracker.record_submit_after(
                submit_status,
                submit_result_stub,
                screenshot_after_verify=verified_screenshot,
                html_after_verify=verified_html,
            )

    normalized_submit = SubmitOutcome.from_status(
        submit_status,
        terminal_failure=bool(submit_guard_reasons) or not bool(completed),
    )
    if "submit_verified" not in submit_status:
        normalized_submit.submit_verified = bool(submit_status.get("verified", False))
    if not normalized_submit.submit_verification_reason:
        normalized_submit.submit_verification_reason = str(
            submit_status.get("submit_verification_reason", "")
            or submit_status.get("method", "")
            or ("submit_guard_blocked" if submit_guard_reasons else "submit_unverified")
        ).strip()
    if not normalized_submit.last_error and not normalized_submit.submit_verified:
        normalized_submit.last_error = str(submit_status.get("detail", "") or "").strip()
    if not normalized_submit.page_url_before_submit:
        normalized_submit.page_url_before_submit = _safe_page_url(page)
    if not normalized_submit.page_url_after_submit:
        normalized_submit.page_url_after_submit = _safe_page_url(page)
    submit_status = normalized_submit.to_dict()
    if processed >= total_targets and apply_budget.status == "active":
        apply_budget.mark_completed()
    if not completed and apply_budget.status == "active":
        apply_budget.mark_failed()
    _emit_progress(
        "apply_complete",
        completed=bool(completed),
        submit_status=dict(submit_status),
        submit_guard_blocked=bool(submit_guard_reasons),
    )

    return {
        "applied": applied,
        "skipped_unchanged": skipped_unchanged,
        "failed": failed,
        "failed_details": failed_details,
        "total_targets": total_targets,
        "apply_budget_sec": float(max_total_sec),
        "elapsed_sec": float(time.time() - started_at),
        "apply_budget_state": apply_budget.to_dict(),
        "completed": completed,
        "submit_guard_blocked": bool(submit_guard_reasons),
        "submit_guard_reasons": submit_guard_reasons,
        "submit_status": submit_status,
    }


__all__ = [
    "_ensure_loop_off",
    "_parse_mmss_to_seconds",
    "_extract_start_end_from_text",
    "_resolve_rows_locator",
    "_first_text_from_row",
    "extract_segments",
    "_normalize_operation_action",
    "_normalize_operations",
    "_segment_duration_seconds",
    "_filter_structural_operations",
    "_normalize_segment_plan",
    "_normalize_label_map_from_plan",
    "_first_visible_child_locator",
    "_respect_major_step_pause",
    "_short_error_text",
    "apply_timestamp_adjustments",
    "_action_selector_for_row",
    "_action_hotkey",
    "_confirm_action_dialog",
    "_wait_rows_delta",
    "_structural_candidate_row_indices",
    "apply_segment_operations",
    "_fill_input",
    "_normalize_label_for_compare",
    "_filter_unchanged_label_map",
    "_handle_quality_review_modal",
    "_handle_no_edits_modal",
    "apply_labels",
]

