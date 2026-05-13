"""Browser navigation and selector helpers extracted from the legacy solver."""

from __future__ import annotations

import logging
import re
import time
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

from playwright.sync_api import Locator, Page

from src.infra.artifacts import _capture_step_artifacts, _task_id_from_url
from src.infra.logging_utils import build_print_logger as _build_print_logger
from src.infra.solver_config import _cfg_get

_logger = logging.getLogger(__name__)
print = _build_print_logger(_logger)

_LAST_RESERVE_REQUEST_TS = 0.0


def _capture_room_flow_step(
    page: Page,
    cfg: Dict[str, Any],
    step_name: str,
    *,
    include_html: bool = False,
) -> None:
    if not bool(_cfg_get(cfg, "run.capture_step_screenshots", False)):
        return
    try:
        _capture_step_artifacts(
            page,
            cfg,
            "_room_flow",
            str(step_name or "").strip() or "room_step",
            include_html=include_html,
        )
    except Exception:
        pass


def _selector_variants(selector: str) -> List[str]:
    if not selector:
        return []
    if "||" in selector:
        return [part.strip() for part in selector.split("||") if part.strip()]
    return [selector.strip()]


def _goto_with_retry(
    page: Page,
    url: str,
    wait_until: str = "domcontentloaded",
    timeout_ms: int = 45000,
    cfg: Optional[Dict[str, Any]] = None,
    reason: str = "",
) -> bool:
    retry_count = max(0, int(_cfg_get(cfg, "run.goto_retry_count", 3) if cfg else 3))
    retry_delay_sec = max(0.2, float(_cfg_get(cfg, "run.goto_retry_delay_sec", 1.2) if cfg else 1.2))
    for attempt in range(retry_count + 1):
        try:
            page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            return True
        except Exception as exc:
            message = str(exc)
            transient = any(
                key in message
                for key in (
                    "ERR_NETWORK_CHANGED",
                    "ERR_INTERNET_DISCONNECTED",
                    "ERR_CONNECTION_RESET",
                    "ERR_TIMED_OUT",
                    "ERR_ABORTED",
                )
            )
            if transient and attempt < retry_count:
                tag = f" ({reason})" if reason else ""
                short = message.splitlines()[0][:200]
                print(
                    f"[nav] transient goto error{tag}; retry {attempt + 1}/{retry_count} "
                    f"in {retry_delay_sec:.1f}s: {short}"
                )
                time.sleep(retry_delay_sec)
                continue
            raise
    return False


def _any_locator_exists(page: Page, selector: str) -> bool:
    for candidate in _selector_variants(selector):
        try:
            if page.locator(candidate).count() > 0:
                return True
        except Exception:
            continue
    return False


def _first_visible_locator(page: Page, selector: str, timeout_ms: int = 4000) -> Locator | None:
    if not selector:
        return None
    deadline = time.time() + timeout_ms / 1000.0
    max_scan = 25
    while time.time() < deadline:
        for candidate in _selector_variants(selector):
            try:
                loc = page.locator(candidate)
                count = loc.count()
                if count <= 0:
                    continue
                scan = min(count, max_scan)
                for i in range(scan):
                    locator = loc.nth(i)
                    if locator.is_visible():
                        return locator
            except Exception:
                continue
        time.sleep(0.1)
    return None


def _safe_locator_click(page: Page, selector: str, timeout_ms: int = 4000) -> bool:
    locator = _first_visible_locator(page, selector, timeout_ms=timeout_ms)
    if locator is None:
        return False
    try:
        click_timeout_ms = max(300, min(timeout_ms, 2000))
        locator.click(timeout=click_timeout_ms, no_wait_after=True)
        return True
    except Exception:
        return False


def _safe_fill(page: Page, selector: str, value: str, timeout_ms: int = 4000) -> bool:
    locator = _first_visible_locator(page, selector, timeout_ms=timeout_ms)
    if locator is None:
        return False
    try:
        locator.fill(value)
        return True
    except Exception:
        try:
            locator.click()
            page.keyboard.press("Control+A")
            page.keyboard.type(value, delay=10)
            return True
        except Exception:
            return False


def _safe_locator_text(locator: Locator, timeout_ms: int = 1200) -> str:
    try:
        return (locator.inner_text(timeout=timeout_ms) or "").strip()
    except Exception:
        return ""


def _first_href_from_selector(page: Page, selector: str, max_scan: int = 40) -> str:
    for candidate in _selector_variants(selector):
        try:
            loc = page.locator(candidate)
            count = min(loc.count(), max_scan)
            for i in range(count):
                href = loc.nth(i).get_attribute("href")
                if href and href.strip():
                    return href.strip()
        except Exception:
            continue
    return ""


def _all_task_label_hrefs_from_page(page: Page) -> List[str]:
    base_url = page.url or "https://audit.atlascapture.io/tasks/room/normal"
    seen: set[str] = set()
    out: List[str] = []

    def _add(raw: str) -> None:
        raw = (raw or "").strip()
        if not raw:
            return
        if raw.startswith("/"):
            raw = urljoin(base_url, raw)
        if "/tasks/room/normal/label/" not in raw:
            return
        if raw in seen:
            return
        seen.add(raw)
        out.append(raw)

    for candidate in _selector_variants('a[href*="/tasks/room/normal/label/"]'):
        try:
            loc = page.locator(candidate)
            count = min(loc.count(), 80)
            for i in range(count):
                href = loc.nth(i).get_attribute("href")
                _add(str(href or ""))
        except Exception:
            continue

    try:
        hrefs_eval = page.evaluate(
            """() => {
                return Array.from(document.querySelectorAll('a[href*="/tasks/room/normal/label/"]'))
                    .map(a => (a.getAttribute('href') || a.href || '').trim())
                    .filter(Boolean);
            }"""
        )
        if isinstance(hrefs_eval, list):
            for item in hrefs_eval:
                if isinstance(item, str):
                    _add(item)
    except Exception:
        pass

    return out


def _first_task_label_href_from_html(page: Page, skip_task_ids: Optional[set[str]] = None) -> str:
    blocked = skip_task_ids if skip_task_ids is not None else set()
    for href in _all_task_label_hrefs_from_page(page):
        task_id = _task_id_from_url(href)
        if task_id and task_id in blocked:
            continue
        return href
    return ""


def _is_label_page_not_found(page: Page) -> bool:
    try:
        body = (page.inner_text("body") or "").lower()
    except Exception:
        body = ""
    if not body:
        return False
    markers = [
        "this page could not be found",
        "404: this page could not be found",
        "next-error-h1",
    ]
    return any(marker in body for marker in markers)


def _is_label_page_internal_error(page: Page) -> bool:
    try:
        body = (page.inner_text("body") or "").lower()
    except Exception:
        body = ""
    if not body:
        return False
    has_error = (
        "error loading episode" in body
        or "an internal error occurred" in body
        or "internal error occurred" in body
    )
    if not has_error:
        return False
    return "go back" in body or "/tasks/room/normal/label/" in (page.url or "").lower()


def _try_go_back_from_label_error(page: Page, cfg: Dict[str, Any], timeout_ms: int = 2500) -> bool:
    go_back_sel = str(_cfg_get(cfg, "atlas.selectors.error_go_back_button", "")).strip()
    if go_back_sel and _safe_locator_click(page, go_back_sel, timeout_ms=timeout_ms):
        page.wait_for_timeout(650)
        return True
    clicked = _safe_locator_click(page, 'button:has-text("Go Back") || a:has-text("Go Back")', timeout_ms=timeout_ms)
    if clicked:
        page.wait_for_timeout(650)
    return clicked


def _is_label_page_actionable(page: Page, cfg: Dict[str, Any], timeout_ms: int = 4500) -> bool:
    from src.solver import legacy_impl as _legacy

    url_l = (page.url or "").lower()
    if "/tasks/room/normal/label/" not in url_l:
        return False

    selectors = _cfg_get(cfg, "atlas.selectors", {})
    rows_sel = str(selectors.get("segment_rows", "")).strip()
    video_sel = str(selectors.get("video_element", "video")).strip() or "video"

    deadline = time.time() + max(300, timeout_ms) / 1000.0
    while time.time() < deadline:
        _legacy._dismiss_blocking_modals(page, cfg)
        if _is_label_page_not_found(page):
            return False
        if _is_label_page_internal_error(page):
            _try_go_back_from_label_error(page, cfg, timeout_ms=800)
            return False

        if video_sel:
            if _first_visible_locator(page, video_sel, timeout_ms=250) is not None:
                return True

        if rows_sel:
            for candidate in _selector_variants(rows_sel):
                try:
                    if page.locator(candidate).count() > 0:
                        return True
                except Exception:
                    continue

        try:
            body = (page.inner_text("body") or "").lower()
        except Exception:
            body = ""
        if "label episode" in body and "segments" in body:
            return True

        page.wait_for_timeout(180)

    return False


def _is_room_access_disabled(page: Page) -> bool:
    try:
        body = (page.inner_text("body") or "").lower()
    except Exception:
        body = ""
    if not body:
        return False
    return "rooms are unavailable" in body or "room access is currently disabled" in body


def _tasks_root_url(cfg: Dict[str, Any]) -> str:
    room_url = str(_cfg_get(cfg, "atlas.room_url", "")).strip()
    if room_url:
        normalized = room_url.rstrip("/")
        if "/tasks/room/" in normalized:
            return normalized.split("/room/", 1)[0]
        if normalized.endswith("/tasks"):
            return normalized
    dashboard_url = str(_cfg_get(cfg, "atlas.dashboard_url", "")).strip()
    if dashboard_url:
        normalized = dashboard_url.rstrip("/")
        if normalized.endswith("/dashboard"):
            return normalized[: -len("/dashboard")] + "/tasks"
    return "https://audit.atlascapture.io/tasks"


def _recover_room_access_disabled(page: Page, cfg: Dict[str, Any], timeout_ms: int = 2500) -> bool:
    if not _is_room_access_disabled(page):
        return False
    clicked = _safe_locator_click(
        page,
        'button:has-text("Back to Tasks") || a:has-text("Back to Tasks")',
        timeout_ms=timeout_ms,
    )
    if clicked:
        page.wait_for_timeout(1200)
        print("[atlas] room access disabled; returned to tasks page.")
        return True
    tasks_url = _tasks_root_url(cfg)
    if not tasks_url:
        return False
    try:
        _goto_with_retry(
            page,
            tasks_url,
            wait_until="domcontentloaded",
            timeout_ms=45000,
            cfg=cfg,
            reason="room-access-disabled",
        )
        print("[atlas] room access disabled; navigated back to tasks root.")
        return True
    except Exception:
        return False


def _is_post_reserve_tasks_onboarding(page: Page) -> bool:
    try:
        body = re.sub(r"\s+", " ", str(page.inner_text("body") or "").lower()).strip()
    except Exception:
        body = ""
    if not body:
        return False
    marker_count = sum(
        1
        for marker in (
            "welcome,",
            "your journey",
            "do labeling tasks",
            "browse tasks",
            "complete training",
            "set up payment",
        )
        if marker in body
    )
    return marker_count >= 2


def _recover_post_reserve_tasks_onboarding(page: Page, cfg: Dict[str, Any], timeout_ms: int = 3000) -> bool:
    if not _is_post_reserve_tasks_onboarding(page):
        return False
    print("[atlas] post-reserve onboarding detected; opening tasks queue.")
    _capture_room_flow_step(page, cfg, "post_reserve_onboarding_detected", include_html=True)

    browse_tasks_selector = (
        'a[href="/tasks"]:has-text("Browse Tasks") || '
        '[role="link"]:has-text("Browse Tasks") || '
        'a:has-text("Browse Tasks") || '
        'button:has-text("Browse Tasks") || '
        '[role="button"]:has-text("Browse Tasks") || '
        'a[href="/tasks"]:has-text("Tasks") || '
        '[data-sidebar="menu-button"][href="/tasks"] || '
        'a[href="/tasks"]'
    )
    if _safe_locator_click(page, browse_tasks_selector, timeout_ms=timeout_ms):
        page.wait_for_timeout(1200)
        _capture_room_flow_step(page, cfg, "post_reserve_onboarding_browse_tasks_clicked", include_html=True)
        return True

    tasks_url = _tasks_root_url(cfg)
    if not tasks_url:
        return False
    try:
        _goto_with_retry(
            page,
            tasks_url,
            wait_until="domcontentloaded",
            timeout_ms=45000,
            cfg=cfg,
            reason="post-reserve-onboarding",
        )
        page.wait_for_timeout(900)
        _capture_room_flow_step(page, cfg, "post_reserve_onboarding_tasks_goto", include_html=True)
        return True
    except Exception:
        return False


def _wait_for_any(page: Page, selector: str, timeout_ms: int = 8000) -> bool:
    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        if _any_locator_exists(page, selector):
            return True
        time.sleep(0.2)
    return False


def _respect_reserve_cooldown(cfg: Dict[str, Any]) -> None:
    global _LAST_RESERVE_REQUEST_TS
    cooldown_sec = max(0, int(_cfg_get(cfg, "run.reserve_cooldown_sec", 120)))
    if cooldown_sec <= 0 or _LAST_RESERVE_REQUEST_TS <= 0:
        return
    elapsed = time.time() - _LAST_RESERVE_REQUEST_TS
    remaining = cooldown_sec - elapsed
    if remaining > 0:
        print(f"[atlas] waiting {int(remaining)}s before reserve request (cooldown).")
        time.sleep(remaining)


def _respect_reserve_min_interval(cfg: Dict[str, Any]) -> None:
    global _LAST_RESERVE_REQUEST_TS
    min_interval_sec = max(0, int(_cfg_get(cfg, "run.reserve_min_interval_sec", 90)))
    if min_interval_sec <= 0 or _LAST_RESERVE_REQUEST_TS <= 0:
        return
    elapsed = time.time() - _LAST_RESERVE_REQUEST_TS
    remaining = min_interval_sec - elapsed
    if remaining > 0:
        print(f"[atlas] waiting {int(remaining)}s before next reserve attempt (min-interval).")
        time.sleep(remaining)


def _mark_reserve_request() -> None:
    global _LAST_RESERVE_REQUEST_TS
    _LAST_RESERVE_REQUEST_TS = time.time()


def _click_reserve_button_dynamic(page: Page, cfg: Dict[str, Any], timeout_ms: int = 2500) -> Tuple[bool, str]:
    reserve_btn = str(_cfg_get(cfg, "atlas.selectors.reserve_episodes_button", "")).strip()
    reserve_loc = _first_visible_locator(page, reserve_btn, timeout_ms=timeout_ms) if reserve_btn else None
    if reserve_loc is not None:
        try:
            txt = (_safe_locator_text(reserve_loc, timeout_ms=600) or "").strip()
            reserve_loc.click(timeout=2000)
            return True, txt
        except Exception:
            pass

    direct_reserve_selector = (
        'button:has-text("Reserve 3 Episodes") || '
        '[role="button"]:has-text("Reserve 3 Episodes") || '
        'button:has-text("Reserve 2 Episodes") || '
        '[role="button"]:has-text("Reserve 2 Episodes") || '
        'button:has-text("Reserve 1 Episode") || '
        '[role="button"]:has-text("Reserve 1 Episode") || '
        'button:has-text("Reserve 1 Episodes") || '
        '[role="button"]:has-text("Reserve 1 Episodes") || '
        'button:has-text("Reserve New Episode") || '
        '[role="button"]:has-text("Reserve New Episode")'
    )
    direct_reserve_loc = _first_visible_locator(page, direct_reserve_selector, timeout_ms=timeout_ms)
    if direct_reserve_loc is not None:
        try:
            txt = (_safe_locator_text(direct_reserve_loc, timeout_ms=600) or "").strip()
            direct_reserve_loc.click(timeout=2000, no_wait_after=True)
            return True, txt
        except Exception:
            pass

    try:
        result = page.evaluate(
            """() => {
                const items = Array.from(document.querySelectorAll('button, [role="button"], a'));
                const isVisible = (el) => {
                    const st = window.getComputedStyle(el);
                    const r = el.getBoundingClientRect();
                    return st && st.visibility !== 'hidden' && st.display !== 'none' && r.width > 0 && r.height > 0;
                };
                const pickScore = (text) => {
                    const t = (text || '').toLowerCase().replace(/\\s+/g, ' ').trim();
                    if (!t.includes('reserve')) return -1;
                    const m = t.match(/reserve\\s*(\\d+)\\s*episodes?/i);
                    if (m) return 100 + parseInt(m[1] || '0', 10);
                    if (t.includes('reserve new episode')) return 60;
                    if (t.includes('reserve') && t.includes('episode')) return 50;
                    return 10;
                };

                let best = null;
                for (const el of items) {
                    if (!isVisible(el)) continue;
                    const text = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
                    const score = pickScore(text);
                    if (score < 0) continue;
                    if (!best || score > best.score) best = { el, text, score };
                }
                if (!best) return { clicked: false, text: '' };
                const target = best.el.closest('button, [role="button"], a') || best.el;
                const clickTarget = (el) => {
                    try {
                        el.click();
                        return true;
                    } catch (e) {}
                    try {
                        el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
                        return true;
                    } catch (e) {}
                    return false;
                };
                return { clicked: clickTarget(target), text: best.text || '' };
            }"""
        )
        if isinstance(result, dict) and bool(result.get("clicked")):
            return True, str(result.get("text", "") or "")
    except Exception:
        pass
    return False, ""


def _page_has_reserve_cta(page: Page, cfg: Dict[str, Any], timeout_ms: int = 900) -> bool:
    reserve_btn_selector = str(_cfg_get(cfg, "atlas.selectors.reserve_episodes_button", "")).strip()
    try:
        if _all_task_label_hrefs_from_page(page):
            return False
    except Exception:
        pass
    if reserve_btn_selector:
        try:
            reserve_loc = _first_visible_locator(page, reserve_btn_selector, timeout_ms=timeout_ms)
            if reserve_loc is not None:
                return True
        except Exception:
            pass
    try:
        body = (page.inner_text("body") or "").lower()
    except Exception:
        body = ""
    if not body:
        return False
    normalized_body = re.sub(r"\s+", " ", body).strip()
    if re.search(r"reserve\s+\d+\s+episodes?", normalized_body):
        return True
    return any(
        marker in normalized_body
        for marker in (
            "no episodes reserved",
            "no episode reserved",
            "reserve a batch of episodes",
            "reserve new episode",
            "you'll reserve",
        )
    )


def _extract_wait_seconds_from_page(page: Page, default_wait_sec: int = 5) -> int:
    try:
        body = (page.inner_text("body") or "").lower()
    except Exception:
        body = ""
    hard_cap = max(5, int(default_wait_sec))
    if not body:
        return hard_cap
    if not re.search(
        r"(too many requests|rate[\s-]?limit|try again in|please wait|temporarily unavailable)",
        body,
    ):
        return hard_cap
    m = re.search(r"(?:try again in|wait|after)\s*(\d+)\s*(seconds?|minutes?|mins?)", body)
    if m:
        try:
            amount = int(m.group(1))
            unit = (m.group(2) or "").lower()
            if amount <= 0:
                return hard_cap
            if unit.startswith("second"):
                return min(hard_cap, max(5, amount))
            return min(hard_cap, amount * 60)
        except Exception:
            pass
    return hard_cap


def _reserve_rate_limited(page: Page) -> bool:
    try:
        body = (page.inner_text("body") or "").lower()
    except Exception:
        body = ""
    if not body:
        return False
    if "reserve" not in body and "episode" not in body:
        return False
    for pat in [
        r"try again in\s*\d+\s*(?:second|seconds|minute|minutes|min)?",
        r"too many requests",
        r"rate[\s-]?limit",
        r"temporarily unavailable",
        r"please wait",
    ]:
        try:
            if re.search(pat, body):
                return True
        except Exception:
            continue
    return False


def _room_has_no_reserved_episodes(page: Page, cfg: Dict[str, Any]) -> bool:
    probe_timeout_ms = max(200, int(_cfg_get(cfg, "run.reserve_no_reserved_probe_timeout_ms", 900)))
    reserve_btn_selector = str(_cfg_get(cfg, "atlas.selectors.reserve_episodes_button", "")).strip()
    try:
        body = (page.inner_text("body") or "").lower()
    except Exception:
        body = ""
    if not body:
        return False
    normalized_body = re.sub(r"\s+", " ", body).strip()
    try:
        if _all_task_label_hrefs_from_page(page):
            return False
    except Exception:
        pass
    if _page_has_reserve_cta(page, cfg, timeout_ms=probe_timeout_ms):
        return True

    no_reserved_markers = (
        "no episodes reserved",
        "no episode reserved",
        "reserve a batch of episodes",
        "you'll reserve",
    )
    likely_no_reserved = any(marker in normalized_body for marker in no_reserved_markers)
    if not likely_no_reserved and not re.search(r"reserve\s+\d+\s+episodes?", normalized_body):
        return False
    if reserve_btn_selector:
        reserve_loc = _first_visible_locator(page, reserve_btn_selector, timeout_ms=probe_timeout_ms)
        if reserve_loc is not None:
            return True
    return bool(re.search(r"reserve\s+\d+\s+episodes?", normalized_body)) or (
        "reserve" in normalized_body and "episode" in normalized_body and likely_no_reserved
    )


def _release_all_reserved_episodes(page: Page, cfg: Dict[str, Any]) -> bool:
    room_url = str(_cfg_get(cfg, "atlas.room_url", "")).strip()
    release_btn = str(_cfg_get(cfg, "atlas.selectors.release_all_button", "")).strip()
    confirm_release_btn = str(_cfg_get(cfg, "atlas.selectors.confirm_release_button", "")).strip()

    if room_url:
        try:
            _goto_with_retry(
                page,
                room_url,
                wait_until="domcontentloaded",
                timeout_ms=45000,
                cfg=cfg,
                reason="room-before-release-all",
            )
        except Exception:
            pass
    _dismiss_blocking_modals(page, cfg)
    if not _safe_locator_click(page, release_btn, timeout_ms=3500):
        print("[atlas] release-all button not found; skipping release cycle.")
        return False
    page.wait_for_timeout(450)

    confirm_clicks = 0
    release_dialog_sel = (
        'div[role="dialog"]:has-text("Release all episodes") '
        '|| div[role="dialog"]:has-text("Release All") '
        '|| [role="dialog"]:has-text("Release all episodes")'
    )
    modal_release_btn = (
        'div[role="dialog"] button:has-text("Release All") '
        '|| div[role="dialog"] [role="button"]:has-text("Release All")'
    )
    try:
        if _wait_for_any(page, release_dialog_sel, timeout_ms=2200):
            if _safe_locator_click(page, modal_release_btn, timeout_ms=2600):
                confirm_clicks += 1
                page.wait_for_timeout(500)
    except Exception:
        pass

    if confirm_clicks == 0 and confirm_release_btn:
        for _ in range(2):
            if _safe_locator_click(page, confirm_release_btn, timeout_ms=2200):
                confirm_clicks += 1
                page.wait_for_timeout(450)
            else:
                break

    page.wait_for_timeout(850)
    total_clicks = 1 + confirm_clicks
    print(f"[atlas] release-all requested for current reserved episodes (clicks={total_clicks}).")
    return True


def goto_task_room(
    page: Page,
    cfg: Dict[str, Any],
    skip_task_ids: Optional[set[str]] = None,
    preferred_task_urls: Optional[List[str]] = None,
    status_out: Optional[Dict[str, Any]] = None,
) -> bool:
    room_url = str(_cfg_get(cfg, "atlas.room_url", "")).strip()
    dashboard_url = str(_cfg_get(cfg, "atlas.dashboard_url", "")).strip()
    wait_sec = float(_cfg_get(cfg, "atlas.wait_before_continue_sec", 5))

    tasks_nav = str(_cfg_get(cfg, "atlas.selectors.tasks_nav", ""))
    enter_workflow = str(_cfg_get(cfg, "atlas.selectors.enter_workflow_button", ""))
    continue_room = str(_cfg_get(cfg, "atlas.selectors.continue_room_button", ""))
    label_button = str(_cfg_get(cfg, "atlas.selectors.label_button", ""))
    label_task_link = str(_cfg_get(cfg, "atlas.selectors.label_task_link", ""))
    confirm_reserve_btn = str(_cfg_get(cfg, "atlas.selectors.confirm_reserve_button", ""))
    blocked_task_ids = skip_task_ids if skip_task_ids is not None else set()
    preferred_targets = preferred_task_urls if preferred_task_urls is not None else []
    preferred_task_ids = {_task_id_from_url(url) for url in preferred_targets if _task_id_from_url(url)}
    if isinstance(status_out, dict):
        status_out["all_visible_blocked"] = False
        status_out["room_access_disabled"] = False
        status_out["no_reserved_episodes"] = False
        status_out["reserve_clicked"] = False
        status_out["label_fast_path"] = False
        status_out["room_disabled_recovery"] = False
        status_out["empty_after_reserve"] = False
    release_all_on_internal_error = bool(_cfg_get(cfg, "run.release_all_on_internal_error", True))
    if bool(_cfg_get(cfg, "run.disable_release_all_during_canary", False)):
        release_all_on_internal_error = False
    reserve_rate_limit_wait_sec = max(5, int(_cfg_get(cfg, "run.reserve_rate_limit_wait_sec", 5)))
    release_requested_by_internal_error = False
    current_url = (page.url or "").strip()
    current_l = current_url.lower()

    if "/tasks/room/normal/label/" in page.url:
        current_task_id = _task_id_from_url(page.url)
        if not preferred_task_ids or current_task_id in preferred_task_ids:
            return True
        print(f"[atlas] current label page is not in target_task_urls; returning to room: {current_task_id}")
        if room_url:
            try:
                _goto_with_retry(
                    page,
                    room_url,
                    wait_until="domcontentloaded",
                    timeout_ms=45000,
                    cfg=cfg,
                    reason="room-after-non-target-label",
                )
            except Exception:
                pass

    if room_url:
        print(f"[atlas] goto room url: {room_url}")
        room_norm = room_url.rstrip("/").lower()
        current_norm = current_url.rstrip("/").lower()
        if current_norm == room_norm or "/tasks/room/normal" in current_l:
            print("[atlas] already on room page; skipping duplicate room navigation.")
        else:
            _goto_with_retry(page, room_url, wait_until="domcontentloaded", timeout_ms=45000, cfg=cfg, reason="goto-room")
    elif dashboard_url:
        print(f"[atlas] goto dashboard url: {dashboard_url}")
        page.goto(dashboard_url, wait_until="domcontentloaded")

    if _is_room_access_disabled(page):
        print("[atlas] room access page is disabled; recovering back to tasks.")
        _recover_room_access_disabled(page, cfg)
        if isinstance(status_out, dict):
            status_out["room_access_disabled"] = True
            status_out["room_disabled_recovery"] = True
        _capture_room_flow_step(page, cfg, "room_disabled_recovery", include_html=True)

    def _recover_standard_workflow_entry() -> None:
        url_l = (page.url or "").lower()
        if "/tasks" not in url_l and "/dashboard" not in url_l:
            return
        if "/tasks/room/normal" in url_l or "/tasks/room/normal/label/" in url_l:
            return
        enter_clicks = max(1, int(_cfg_get(cfg, "run.workflow_reentry_enter_clicks", 2)))
        second_click_delay_sec = max(0.0, float(_cfg_get(cfg, "run.workflow_reentry_second_click_delay_sec", 5.0)))
        clicked_any = False
        for i in range(enter_clicks):
            clicked = _safe_locator_click(page, enter_workflow, timeout_ms=5000)
            if clicked:
                clicked_any = True
                print(f"[atlas] workflow recovery: clicked workflow entry ({i + 1}/{enter_clicks}).")
            if i == 0 and enter_clicks > 1 and second_click_delay_sec > 0:
                print(f"[atlas] workflow recovery: waiting {second_click_delay_sec:.1f}s before second workflow-entry click.")
                time.sleep(second_click_delay_sec)
            page.wait_for_timeout(700)
        if not clicked_any:
            return
        _safe_locator_click(page, continue_room, timeout_ms=4500)
        _safe_locator_click(page, label_button, timeout_ms=4500)
        page.wait_for_timeout(700)

    def _wait_label_page_ready() -> None:
        checks = max(1, int(_cfg_get(cfg, "run.label_open_loading_max_checks", 5)))
        wait_ms = max(120, int(_cfg_get(cfg, "run.label_open_loading_wait_ms", 600)))
        for _ in range(checks):
            _dismiss_blocking_modals(page)
            try:
                body = (page.inner_text("body") or "").lower()
            except Exception:
                body = ""
            if "loading..." not in body:
                break
            page.wait_for_timeout(wait_ms)

    def _handle_internal_error_release_cycle() -> None:
        nonlocal release_requested_by_internal_error
        if not release_requested_by_internal_error:
            return
        _release_all_reserved_episodes(page, cfg)
        release_requested_by_internal_error = False
        if room_url:
            try:
                _goto_with_retry(
                    page,
                    room_url,
                    wait_until="domcontentloaded",
                    timeout_ms=45000,
                    cfg=cfg,
                    reason="room-after-release-internal-error",
                )
            except Exception:
                pass
        page.wait_for_timeout(900)

    def _open_label_target(target: str, reason: str, log_label: str) -> bool:
        nonlocal release_requested_by_internal_error
        _goto_with_retry(page, target, wait_until="commit", timeout_ms=45000, cfg=cfg, reason=reason)
        print(f"[atlas] opened label task by {log_label}: {target}")
        if isinstance(status_out, dict) and "fast-path" in str(log_label or "").lower():
            status_out["label_fast_path"] = True
            _capture_room_flow_step(page, cfg, "label_fast_path_opened")
        _wait_label_page_ready()
        if _is_label_page_actionable(page, cfg, timeout_ms=5000):
            return True
        bad_task_id = _task_id_from_url(page.url) or _task_id_from_url(target)
        if bad_task_id:
            blocked_task_ids.add(bad_task_id)
            print(f"[atlas] label page unavailable; task blocked for this run: {bad_task_id}")
        if _is_label_page_internal_error(page):
            _try_go_back_from_label_error(page, cfg)
            print("[atlas] label page failed with internal error; clicked Go Back.")
            if release_all_on_internal_error:
                release_requested_by_internal_error = True
                print("[atlas] internal error detected; release-all cycle requested.")
        elif _is_label_page_not_found(page):
            print("[atlas] label URL returned not-found page; trying another task.")
        else:
            print("[atlas] label page opened but video/segments are unavailable; trying another task.")
        if room_url:
            try:
                _goto_with_retry(
                    page,
                    room_url,
                    wait_until="domcontentloaded",
                    timeout_ms=45000,
                    cfg=cfg,
                    reason="room-after-invalid-label",
                )
            except Exception:
                pass
        return False

    if preferred_targets:
        for target in preferred_targets:
            tid = _task_id_from_url(target)
            if tid and tid in blocked_task_ids:
                continue
            if _open_label_target(target, reason="open-target-task", log_label="configured target url"):
                return True
            if release_requested_by_internal_error:
                _handle_internal_error_release_cycle()
        if isinstance(status_out, dict):
            status_out["target_tasks_exhausted"] = True
            status_out["sticky_resume_exhausted"] = bool(
                _cfg_get(cfg, "run.sticky_episode_resume", False)
            )
        if bool(_cfg_get(cfg, "run.sticky_episode_resume", False)):
            print("[atlas] preferred sticky target is no longer actionable; falling back to normal queue scan.")
        else:
            return False

    if label_task_link:
        if bool(status_out and status_out.get("room_access_disabled")) and "/tasks" in (page.url or "").lower():
            page.wait_for_timeout(1000)
        for href_from_html in _all_task_label_hrefs_from_page(page):
            tid = _task_id_from_url(href_from_html)
            if tid and tid in blocked_task_ids:
                continue
            target = href_from_html if href_from_html.startswith("http") else f"https://audit.atlascapture.io{href_from_html}"
            if _open_label_target(target, reason="open-label-fast", log_label="html href (fast-path)"):
                return True
            if release_requested_by_internal_error:
                _handle_internal_error_release_cycle()
                break

    direct_reserve_view = False
    if label_task_link:
        late_href_candidates = _all_task_label_hrefs_from_page(page)
        if not late_href_candidates and "/tasks" in ((page.url or "").lower()):
            try:
                page.wait_for_timeout(1500)
            except Exception:
                pass
            late_href_candidates = _all_task_label_hrefs_from_page(page)
        for href_from_html in late_href_candidates:
            tid = _task_id_from_url(href_from_html)
            if tid and tid in blocked_task_ids:
                continue
            target = href_from_html if href_from_html.startswith("http") else f"https://audit.atlascapture.io{href_from_html}"
            if _open_label_target(target, reason="open-label-fast-late", log_label="html href (late fast-path)"):
                return True
            if release_requested_by_internal_error:
                _handle_internal_error_release_cycle()
                break
        direct_reserve_view = _page_has_reserve_cta(page, cfg, timeout_ms=1200)
        if direct_reserve_view:
            print("[atlas] reserve CTA visible on current tasks view; skipping workflow re-entry.")
            if isinstance(status_out, dict):
                status_out["no_reserved_episodes"] = True
            _capture_room_flow_step(page, cfg, "reserve_cta_current_view", include_html=True)

    if not direct_reserve_view:
        _recover_standard_workflow_entry()

        current_l = (page.url or "").lower()
        if tasks_nav and "/tasks/room/normal" not in current_l and "/tasks/room/normal/label/" not in current_l:
            _safe_locator_click(page, tasks_nav, timeout_ms=3000)
        _safe_locator_click(page, enter_workflow, timeout_ms=4000)
        if wait_sec > 0:
            time.sleep(wait_sec)
        _safe_locator_click(page, continue_room, timeout_ms=4000)
        _safe_locator_click(page, label_button, timeout_ms=4000)
        if _is_room_access_disabled(page):
            print("[atlas] room access became disabled after room entry; returning to tasks.")
            _recover_room_access_disabled(page, cfg)
            if isinstance(status_out, dict):
                status_out["room_access_disabled"] = True
                status_out["room_disabled_recovery"] = True
            _capture_room_flow_step(page, cfg, "room_disabled_after_entry", include_html=True)

    if label_task_link:
        page.wait_for_timeout(1000)
        reserve_attempts = max(1, int(_cfg_get(cfg, "run.reserve_attempts_per_visit", 3)))
        label_wait_timeout_ms = max(1500, int(_cfg_get(cfg, "run.reserve_label_wait_timeout_ms", 12000)))
        label_wait_after_reserve_timeout_ms = max(
            1000, int(_cfg_get(cfg, "run.reserve_label_wait_timeout_after_reserve_ms", 3500))
        )
        reserve_refresh_after_click = bool(_cfg_get(cfg, "run.reserve_refresh_after_click", True))
        reserve_wait_only_on_rate_limit = bool(_cfg_get(cfg, "run.reserve_wait_only_on_rate_limit", True))
        reserve_immediate_when_no_reserved = bool(_cfg_get(cfg, "run.reserve_immediate_when_no_reserved", True))
        reserve_skip_initial_label_scan_when_no_reserved = bool(
            _cfg_get(cfg, "run.reserve_skip_initial_label_scan_when_no_reserved", True)
        )
        skip_reserve_when_all_visible_blocked = bool(
            _cfg_get(cfg, "run.skip_reserve_when_all_visible_blocked", False)
        )
        tasks_root_url = _tasks_root_url(cfg)

        def _open_first_label_from_page(reason: str) -> bool:
            nonlocal release_requested_by_internal_error
            href_candidates = _all_task_label_hrefs_from_page(page)
            attempted_unblocked_candidate = False
            for href_from_html in href_candidates:
                tid = _task_id_from_url(href_from_html)
                if tid and tid in blocked_task_ids:
                    continue
                attempted_unblocked_candidate = True
                target = href_from_html if href_from_html.startswith("http") else f"https://audit.atlascapture.io{href_from_html}"
                if _open_label_target(target, reason=reason, log_label="html href"):
                    return True
                if release_requested_by_internal_error:
                    return False

            if href_candidates and blocked_task_ids and not attempted_unblocked_candidate:
                print(f"[atlas] all visible label tasks are blocked in this run ({len(blocked_task_ids)} blocked).")
                if isinstance(status_out, dict):
                    status_out["all_visible_blocked"] = True
                return False

            link_loc = _first_visible_locator(page, label_task_link, timeout_ms=2500)
            if link_loc is None:
                href = _first_href_from_selector(page, label_task_link)
                if href:
                    target = href if href.startswith("http") else f"https://audit.atlascapture.io{href}"
                    return _open_label_target(target, reason=f"{reason}-href", log_label="href")
                return False

            try:
                href = link_loc.get_attribute("href")
                link_loc.click()
                if href:
                    print(f"[atlas] opened label task: {href}")
                _wait_label_page_ready()
            except Exception:
                return False
            if "/tasks/room/normal/label/" in page.url:
                if _is_label_page_actionable(page, cfg, timeout_ms=5000):
                    return True
                bad_task_id = _task_id_from_url(page.url) or _task_id_from_url(href or "")
                if bad_task_id:
                    blocked_task_ids.add(bad_task_id)
                    print(f"[atlas] label page unavailable; task blocked for this run: {bad_task_id}")
                if _is_label_page_internal_error(page):
                    _try_go_back_from_label_error(page, cfg)
                    if release_all_on_internal_error:
                        release_requested_by_internal_error = True
                if room_url:
                    try:
                        _goto_with_retry(
                            page,
                            room_url,
                            wait_until="domcontentloaded",
                            timeout_ms=45000,
                            cfg=cfg,
                            reason="room-after-click-invalid-label",
                        )
                    except Exception:
                        pass
                return False
            href = _first_href_from_selector(page, label_task_link)
            if href:
                target = href if href.startswith("http") else f"https://audit.atlascapture.io{href}"
                return _open_label_target(target, reason=f"{reason}-href-fallback", log_label="href fallback")
            return False

        visible_reserve_cta = _page_has_reserve_cta(page, cfg, timeout_ms=1200)
        visible_label_hrefs = _all_task_label_hrefs_from_page(page)
        no_reserved_episodes = reserve_immediate_when_no_reserved and (
            not bool(visible_label_hrefs)
            and (_room_has_no_reserved_episodes(page, cfg) or visible_reserve_cta or direct_reserve_view)
        )
        if isinstance(status_out, dict):
            status_out["no_reserved_episodes"] = bool(no_reserved_episodes)
        if no_reserved_episodes:
            print("[atlas] no reserved episodes detected; reserving immediately.")
            _capture_room_flow_step(page, cfg, "no_reserved_detected", include_html=True)
        try_initial_label_scan = not (no_reserved_episodes and reserve_skip_initial_label_scan_when_no_reserved)

        if try_initial_label_scan and _open_first_label_from_page("open-label-html"):
            return True
        if (
            try_initial_label_scan
            and isinstance(status_out, dict)
            and bool(status_out.get("all_visible_blocked"))
            and skip_reserve_when_all_visible_blocked
        ):
            print("[atlas] skipping reserve: all visible tasks are blocked in this run.")
            return False
        if release_requested_by_internal_error:
            _handle_internal_error_release_cycle()

        for reserve_attempt in range(1, reserve_attempts + 1):
            reserved = False
            reserve_label = ""
            if not no_reserved_episodes:
                _respect_reserve_min_interval(cfg)
                if not reserve_wait_only_on_rate_limit:
                    _respect_reserve_cooldown(cfg)
            clicked, reserve_label = _click_reserve_button_dynamic(page, cfg, timeout_ms=2500)
            if no_reserved_episodes and not clicked:
                print(
                    f"[atlas] reserve CTA was expected on no-reserved page but click was not confirmed "
                    f"({reserve_attempt}/{reserve_attempts})."
                )
            if clicked:
                try:
                    reserved = True
                    _mark_reserve_request()
                    if isinstance(status_out, dict):
                        status_out["reserve_clicked"] = True
                    if reserve_label:
                        print(f"[atlas] reserve requested: '{reserve_label}' ({reserve_attempt}/{reserve_attempts}).")
                    else:
                        print(f"[atlas] reserve requested ({reserve_attempt}/{reserve_attempts}).")
                    _capture_room_flow_step(page, cfg, "reserve_clicked", include_html=True)
                except Exception:
                    reserved = False

            if reserved:
                _safe_locator_click(page, confirm_reserve_btn, timeout_ms=4500)
                if _reserve_rate_limited(page):
                    rate_wait_sec = _extract_wait_seconds_from_page(page, default_wait_sec=reserve_rate_limit_wait_sec)
                    print(f"[atlas] reserve is rate-limited; waiting {rate_wait_sec}s then retrying reserve.")
                    time.sleep(rate_wait_sec)
                    if room_url:
                        try:
                            _goto_with_retry(
                                page,
                                room_url,
                                wait_until="domcontentloaded",
                                timeout_ms=45000,
                                cfg=cfg,
                                reason="room-after-reserve-rate-limit",
                            )
                        except Exception:
                            pass
                    continue
                reserve_refresh_target = room_url
                reserve_refresh_reason = "room-refresh-after-reserve"
                if direct_reserve_view or ((page.url or "").rstrip("/").lower().endswith("/tasks") and tasks_root_url):
                    reserve_refresh_target = tasks_root_url
                    reserve_refresh_reason = "tasks-refresh-after-reserve"
                if reserve_refresh_after_click and reserve_refresh_target:
                    try:
                        _goto_with_retry(
                            page,
                            reserve_refresh_target,
                            wait_until="domcontentloaded",
                            timeout_ms=45000,
                            cfg=cfg,
                            reason=reserve_refresh_reason,
                        )
                    except Exception:
                        pass
                    if _is_room_access_disabled(page):
                        print("[atlas] room access disabled after reserve refresh; returning to tasks root.")
                        _recover_room_access_disabled(page, cfg)
                        if isinstance(status_out, dict):
                            status_out["room_access_disabled"] = True
                            status_out["room_disabled_recovery"] = True
                        _capture_room_flow_step(page, cfg, "room_disabled_after_reserve_refresh", include_html=True)
                    elif _recover_post_reserve_tasks_onboarding(page, cfg):
                        page.wait_for_timeout(700)
                    if _open_first_label_from_page("open-label-after-reserve-refresh"):
                        return True

            if _is_room_access_disabled(page):
                print("[atlas] room access disabled during reserve/open flow; recovering to tasks root.")
                _recover_room_access_disabled(page, cfg)
                if isinstance(status_out, dict):
                    status_out["room_access_disabled"] = True
                    status_out["room_disabled_recovery"] = True
                _capture_room_flow_step(page, cfg, "room_disabled_during_reserve_flow", include_html=True)
                if _open_first_label_from_page("open-label-after-room-disabled-recover"):
                    return True

            if reserved and _recover_post_reserve_tasks_onboarding(page, cfg):
                if _open_first_label_from_page("open-label-after-post-reserve-onboarding"):
                    return True

            if reserved and _open_first_label_from_page("open-label-after-reserve-direct-scan"):
                return True
            _safe_locator_click(page, label_button, timeout_ms=3500)
            wait_timeout_ms = label_wait_after_reserve_timeout_ms if reserved else label_wait_timeout_ms
            if no_reserved_episodes and not reserved:
                wait_timeout_ms = min(wait_timeout_ms, 2500)
            _wait_for_any(page, label_task_link, timeout_ms=wait_timeout_ms)
            page.wait_for_timeout(700)
            if reserved and _recover_post_reserve_tasks_onboarding(page, cfg):
                page.wait_for_timeout(700)
                if _open_first_label_from_page("open-label-after-post-reserve-onboarding-wait"):
                    return True
            if _open_first_label_from_page("open-label-after-reserve"):
                return True
            if reserved and isinstance(status_out, dict):
                status_out["empty_after_reserve"] = True
                _capture_room_flow_step(page, cfg, "empty_after_reserve", include_html=True)
            if release_requested_by_internal_error:
                _handle_internal_error_release_cycle()
                continue

            if reserve_attempt < reserve_attempts:
                page.wait_for_timeout(250 if no_reserved_episodes else 900)

    return "/tasks/room/normal/label/" in page.url


def _legacy() -> Any:
    return import_module("src.solver.legacy_impl")


def _dismiss_blocking_modals(page: Page, cfg: Optional[Dict[str, Any]] = None) -> None:
    _legacy()._dismiss_blocking_modals(page, cfg)


def _dismiss_blocking_side_panel(page: Page, cfg: Dict[str, Any], aggressive: bool = False) -> bool:
    return _legacy()._dismiss_blocking_side_panel(page, cfg, aggressive=aggressive)


def _click_segment_row_with_recovery(page: Page, rows: Locator, idx: int, cfg: Dict[str, Any]) -> None:
    _legacy()._click_segment_row_with_recovery(page, rows, idx, cfg)


def _respect_episode_delay(cfg: Dict[str, Any]) -> None:
    _legacy()._respect_episode_delay(cfg)


def _compute_backoff_delay(cfg: Dict[str, Any], attempt: int) -> float:
    return _legacy()._compute_backoff_delay(cfg, attempt)

def _verify_submit_and_wait(
    page: Page,
    cfg: Dict[str, Any],
    *,
    task_id: str = "",
    timeout_sec: float = 12.0,
) -> Dict[str, Any]:
    """
    After clicking Submit, wait for confirmation (success toast, URL change, or body text).

    Returns dict with:
      verified: bool
      method: str (toast_detected | url_check | body_scan | timeout)
      detail: str
    """
    start = time.time()
    deadline = start + max(1.0, timeout_sec)

    success_toast_sel = str(_cfg_get(cfg, "atlas.selectors.submit_success_toast", "")).strip()
    error_toast_sel = str(_cfg_get(cfg, "atlas.selectors.error_submit_toast", "")).strip()
    success_patterns = [
        r"successfully submitted",
        r"submission complete",
        r"episode submitted",
        r"submitted successfully",
        r"labels saved",
    ]
    error_patterns = [
        r"submission failed",
        r"error submitting",
        r"failed to submit",
        r"could not submit",
        r"overlaps previous segment",
        r"must be greater than",
        r"still unlabeled",
        r"is not monotonic",
    ]

    while time.time() < deadline:
        # Check for success toast
        if success_toast_sel:
            loc = _first_visible_locator(page, success_toast_sel, timeout_ms=300)
            if loc is not None:
                elapsed = time.time() - start
                print(f"[submit] success toast detected for {task_id} ({elapsed:.1f}s)")
                return {"verified": True, "method": "toast_detected", "detail": "Success toast visible"}

        # Check for error toast
        if error_toast_sel:
            loc = _first_visible_locator(page, error_toast_sel, timeout_ms=200)
            if loc is not None:
                elapsed = time.time() - start
                text = _safe_locator_text(loc, timeout_ms=500)
                print(f"[submit] ERROR toast detected for {task_id}: {text} ({elapsed:.1f}s)")
                return {"verified": False, "method": "error_toast", "detail": f"Error toast: {text}"}

        # Check body text
        try:
            body = (page.inner_text("body") or "").lower()
        except Exception:
            body = ""

        for pattern in success_patterns:
            if re.search(pattern, body):
                elapsed = time.time() - start
                return {"verified": True, "method": "body_scan", "detail": f"Matched: {pattern}"}

        for pattern in error_patterns:
            if re.search(pattern, body):
                elapsed = time.time() - start
                return {"verified": False, "method": "body_scan", "detail": f"Error matched: {pattern}"}

        # Check URL change (navigated away from label page)
        current_url = (page.url or "").lower()
        if "/tasks/room/normal/label/" not in current_url and "/tasks/room/normal" in current_url:
            elapsed = time.time() - start
            print(f"[submit] URL navigated to room for {task_id} ({elapsed:.1f}s)")
            return {"verified": True, "method": "url_check", "detail": f"Navigated: {page.url}"}

        page.wait_for_timeout(350)

    elapsed = time.time() - start
    print(f"[submit] TIMEOUT waiting for confirmation ({task_id}, {elapsed:.1f}s)")
    return {"verified": False, "method": "timeout", "detail": f"No confirmation within {timeout_sec:.0f}s"}


def _click_submit_with_verification(
    page: Page,
    cfg: Dict[str, Any],
    *,
    task_id: str = "",
    verify_timeout_sec: float = 12.0,
) -> Dict[str, Any]:
    """
    Complete the submit flow with post-submit verification:
    1. Click Complete/Submit button
    2. Handle quality review modal if it appears
    3. Wait for submit confirmation (toast, URL change, body text)

    Returns dict with submit_clicked, quality_modal_handled, verification result.
    """
    result: Dict[str, Any] = {
        "submit_clicked": False,
        "quality_modal_handled": False,
        "verification": {"verified": False, "method": "not_attempted", "detail": ""},
    }

    try:
        segments_mod = import_module("src.solver.segments")
        submit_status = segments_mod._submit_episode(
            page,
            cfg,
            episode_id=task_id,
            return_details=True,
        )
        if isinstance(submit_status, dict):
            reason = str(submit_status.get("submit_verification_reason", "") or "").strip()
            result["submit_clicked"] = bool(
                submit_status.get("complete_button_clicked", False)
                or submit_status.get("submit_modal_already_open", False)
            )
            result["quality_modal_handled"] = bool(submit_status.get("quality_review_confirmed", False))
            result["verification"] = {
                "verified": bool(submit_status.get("submit_verified", False)),
                "method": reason or "submit_status",
                "detail": reason,
                **submit_status,
            }
            return result
    except Exception as exc:
        print(f"[submit] robust submit delegation failed for {task_id or '<unknown>'}: {exc}")

    complete_sel = str(_cfg_get(cfg, "atlas.selectors.complete_button", "")).strip()
    if not complete_sel:
        complete_sel = (
            'button:has-text("Complete") || [role="button"]:has-text("Complete") || '
            'button:has-text("Finish") || [role="button"]:has-text("Finish") || '
            'button:has-text("Done") || [role="button"]:has-text("Done") || '
            'button:has-text("Submit Task") || button:has-text("Submit Episode") || '
            'button:has-text("Submit") || [role="button"]:has-text("Submit")'
        )

    if _safe_locator_click(page, complete_sel, timeout_ms=8000):
        result["submit_clicked"] = True
        print(f"[submit] clicked Complete/Submit for {task_id}")
        page.wait_for_timeout(1000)
    else:
        print(f"[submit] ERROR: could not find Complete/Submit button for {task_id}")
        return result

    result["verification"] = _verify_submit_and_wait(
        page,
        cfg,
        task_id=task_id,
        timeout_sec=verify_timeout_sec,
    )
    return result


def _legacy() -> Any:
    return import_module("src.solver.legacy_impl")


def _dismiss_blocking_modals(page: Page, cfg: Optional[Dict[str, Any]] = None) -> None:
    _legacy()._dismiss_blocking_modals(page, cfg)


def _dismiss_blocking_side_panel(page: Page, cfg: Dict[str, Any], aggressive: bool = False) -> bool:
    return _legacy()._dismiss_blocking_side_panel(page, cfg, aggressive=aggressive)


def _click_segment_row_with_recovery(page: Page, rows: Locator, idx: int, cfg: Dict[str, Any]) -> None:
    _legacy()._click_segment_row_with_recovery(page, rows, idx, cfg)


def _respect_episode_delay(cfg: Dict[str, Any]) -> None:
    _legacy()._respect_episode_delay(cfg)


def _compute_backoff_delay(cfg: Dict[str, Any], attempt: int) -> float:
    return _legacy()._compute_backoff_delay(cfg, attempt)

__all__ = [
    "_selector_variants",
    "_goto_with_retry",
    "_any_locator_exists",
    "_first_visible_locator",
    "_safe_locator_click",
    "_safe_fill",
    "_safe_locator_text",
    "_first_href_from_selector",
    "_all_task_label_hrefs_from_page",
    "_first_task_label_href_from_html",
    "_is_label_page_not_found",
    "_is_label_page_internal_error",
    "_try_go_back_from_label_error",
    "_is_label_page_actionable",
    "_wait_for_any",
    "_dismiss_blocking_modals",
    "_dismiss_blocking_side_panel",
    "_click_segment_row_with_recovery",
    "_respect_reserve_cooldown",
    "_respect_reserve_min_interval",
    "_mark_reserve_request",
    "_click_reserve_button_dynamic",
    "_page_has_reserve_cta",
    "_extract_wait_seconds_from_page",
    "_reserve_rate_limited",
    "_room_has_no_reserved_episodes",
    "_release_all_reserved_episodes",
    "_respect_episode_delay",
    "_compute_backoff_delay",
    "_is_room_access_disabled",
    "_recover_room_access_disabled",
    "_verify_submit_and_wait",
    "_click_submit_with_verification",
    "goto_task_room",
]

