from __future__ import annotations

from typing import Any, Dict, List

import atlas_web_auto_solver  # noqa: F401
from src.solver import segments


class _FakeButton:
    def __init__(self, page: "_FakePage", candidate: str) -> None:
        self.page = page
        self.candidate = candidate

    def count(self) -> int:
        return 1

    def scroll_into_view_if_needed(self, timeout: int | None = None) -> None:
        return None

    def is_visible(self) -> bool:
        return True

    def is_enabled(self) -> bool:
        return True

    def click(self, timeout: int | None = None, no_wait_after: bool | None = None, force: bool | None = None) -> None:
        self.page.button_clicks += 1

    def evaluate(self, script: str) -> None:
        self.page.button_evals += 1


class _FakeChildCollection:
    def __init__(self, page: "_FakePage", candidate: str) -> None:
        self.page = page
        self.candidate = candidate

    def count(self) -> int:
        return 1

    def nth(self, index: int) -> _FakeButton:
        return _FakeButton(self.page, self.candidate)

    @property
    def first(self) -> _FakeButton:
        return _FakeButton(self.page, self.candidate)


class _FakeRow:
    def __init__(self, page: "_FakePage") -> None:
        self.page = page

    def scroll_into_view_if_needed(self, timeout: int | None = None) -> None:
        self.page.scroll_timeouts.append(timeout)
        return None

    def click(self, timeout: int | None = None, no_wait_after: bool | None = None, force: bool | None = None) -> None:
        self.page.row_clicks += 1

    def hover(self, timeout: int | None = None) -> None:
        return None

    def locator(self, candidate: str) -> _FakeChildCollection:
        return _FakeChildCollection(self.page, candidate)


class _FakeRows:
    def __init__(self, page: "_FakePage") -> None:
        self.page = page

    def count(self) -> int:
        return self.page.row_count

    def nth(self, index: int) -> _FakeRow:
        return _FakeRow(self.page)


class _FakeKeyboard:
    def __init__(self, page: "_FakePage") -> None:
        self.page = page
        self.presses: List[str] = []

    def press(self, key: str) -> None:
        self.presses.append(key)
        if key.lower() == "s" and not self.page.split_applied:
            self.page.row_count += 1
            self.page.split_applied = True
        if key.lower() == "m" and not self.page.merge_applied and self.page.row_count > 0:
            self.page.row_count -= 1
            self.page.merge_applied = True


class _FakePage:
    def __init__(self) -> None:
        self.row_count = 3
        self.row_clicks = 0
        self.button_clicks = 0
        self.button_evals = 0
        self.split_applied = False
        self.merge_applied = False
        self.keyboard = _FakeKeyboard(self)
        self.waits: List[int] = []
        self.scroll_timeouts: List[int | None] = []

    def locator(self, selector: str) -> _FakeRows:
        return _FakeRows(self)

    def wait_for_timeout(self, ms: int) -> None:
        self.waits.append(ms)


def test_confirm_action_dialog_prefers_split_specific_button(monkeypatch) -> None:
    seen: List[str] = []

    def _fake_safe_locator_click(page: Any, selector: str, timeout_ms: int = 0) -> bool:
        seen.append(selector)
        return 'has-text("Split")' in selector

    monkeypatch.setattr(segments._browser, "_safe_locator_click", _fake_safe_locator_click)

    page = _FakePage()
    cfg: Dict[str, Any] = {
        "atlas": {
            "selectors": {
                "action_confirm_button": 'button:has-text("Confirm")',
            }
        }
    }

    assert segments._confirm_action_dialog(page, cfg, action="split") is True
    assert any('has-text("Split")' in item for item in seen)


def test_apply_segment_operations_retries_split_with_hotkey_and_positions_playhead(monkeypatch) -> None:
    positioned: List[int] = []

    monkeypatch.setattr(
        segments,
        "_resolve_rows_locator",
        lambda page, rows_selector, sample_size=8, row_text_timeout_ms=350: ("rows", page.locator("rows")),
    )
    monkeypatch.setattr(segments._browser, "_dismiss_blocking_modals", lambda page, cfg=None: None)
    monkeypatch.setattr(segments._browser, "_dismiss_blocking_side_panel", lambda page, cfg, aggressive=False: False)
    monkeypatch.setattr(segments._browser, "_click_segment_row_with_recovery", lambda page, rows, idx, cfg: None)
    monkeypatch.setattr(segments._browser, "_selector_variants", lambda selector: [selector] if selector else [])
    monkeypatch.setattr(
        segments,
        "_position_video_for_split",
        lambda page, cfg, source_segment, idx: positioned.append(idx) or True,
    )
    monkeypatch.setattr(segments, "_confirm_action_dialog", lambda page, cfg, action="": False)
    monkeypatch.setattr(
        segments,
        "_wait_rows_delta",
        lambda page, rows_selector, before_count, expected_delta, timeout_ms=0, mode="exact": page.row_count > before_count,
    )

    page = _FakePage()
    cfg: Dict[str, Any] = {
        "run": {
            "structural_skip_if_segments_ge": 40,
            "structural_max_failures_per_episode": 4,
            "structural_wait_rows_delta_timeout_ms": 50,
            "max_segment_duration_sec": 10.0,
        },
        "atlas": {
            "selectors": {
                "segment_rows": "rows",
                "split_button_in_row": 'button[title*="Split"]',
            }
        },
    }
    operations = [{"action": "split", "segment_index": 1}]
    source_segments = [
        {
            "segment_index": 1,
            "start_sec": 0.0,
            "end_sec": 12.0,
            "current_label": "adjust fabric",
        }
    ]

    result = segments.apply_segment_operations(
        page,
        cfg,
        operations,
        source_segments=source_segments,
    )

    assert result["applied"] == 1
    assert result["structural_applied"] == 1
    assert result["failed"] == []
    assert page.keyboard.presses
    assert positioned


def test_apply_segment_operations_allows_merge_on_large_episode(monkeypatch) -> None:
    monkeypatch.setattr(
        segments,
        "_resolve_rows_locator",
        lambda page, rows_selector, sample_size=8, row_text_timeout_ms=350: ("rows", page.locator("rows")),
    )
    monkeypatch.setattr(segments._browser, "_dismiss_blocking_modals", lambda page, cfg=None: None)
    monkeypatch.setattr(segments._browser, "_dismiss_blocking_side_panel", lambda page, cfg, aggressive=False: False)
    monkeypatch.setattr(segments._browser, "_click_segment_row_with_recovery", lambda page, rows, idx, cfg: None)
    monkeypatch.setattr(segments._browser, "_selector_variants", lambda selector: [])
    monkeypatch.setattr(segments, "_confirm_action_dialog", lambda page, cfg, action="": False)
    monkeypatch.setattr(
        segments,
        "_wait_rows_delta",
        lambda page, rows_selector, before_count, expected_delta, timeout_ms=0, mode="exact": page.row_count < before_count,
    )

    page = _FakePage()
    page.row_count = 52
    cfg: Dict[str, Any] = {
        "run": {
            "structural_skip_if_segments_ge": 40,
            "structural_skip_allow_merge": True,
            "structural_allow_merge": True,
            "structural_max_failures_per_episode": 4,
            "structural_wait_rows_delta_timeout_ms": 50,
        },
        "atlas": {"selectors": {"segment_rows": "rows"}},
    }
    operations = [
        {"action": "split", "segment_index": 1},
        {"action": "merge", "segment_index": 2},
    ]

    result = segments.apply_segment_operations(page, cfg, operations, source_segments=[])

    assert result["applied"] == 1
    assert result["structural_applied"] == 1
    assert any("skipped 1 structural op" in item for item in result["failed"])
    assert "m" in [key.lower() for key in page.keyboard.presses]


def test_apply_segment_operations_allows_split_on_large_episode_when_source_is_overlong(monkeypatch) -> None:
    positioned: List[int] = []

    monkeypatch.setattr(
        segments,
        "_resolve_rows_locator",
        lambda page, rows_selector, sample_size=8, row_text_timeout_ms=350: ("rows", page.locator("rows")),
    )
    monkeypatch.setattr(segments._browser, "_dismiss_blocking_modals", lambda page, cfg=None: None)
    monkeypatch.setattr(segments._browser, "_dismiss_blocking_side_panel", lambda page, cfg, aggressive=False: False)
    monkeypatch.setattr(segments._browser, "_click_segment_row_with_recovery", lambda page, rows, idx, cfg: None)
    monkeypatch.setattr(segments._browser, "_selector_variants", lambda selector: [selector] if selector else [])
    monkeypatch.setattr(
        segments,
        "_position_video_for_split",
        lambda page, cfg, source_segment, idx: positioned.append(idx) or True,
    )
    monkeypatch.setattr(segments, "_confirm_action_dialog", lambda page, cfg, action="": False)
    monkeypatch.setattr(
        segments,
        "_wait_rows_delta",
        lambda page, rows_selector, before_count, expected_delta, timeout_ms=0, mode="exact": page.row_count > before_count,
    )

    page = _FakePage()
    page.row_count = 52
    cfg: Dict[str, Any] = {
        "run": {
            "structural_skip_if_segments_ge": 40,
            "structural_skip_allow_merge": True,
            "structural_max_failures_per_episode": 4,
            "structural_wait_rows_delta_timeout_ms": 50,
            "max_segment_duration_sec": 10.0,
        },
        "atlas": {
            "selectors": {
                "segment_rows": "rows",
                "split_button_in_row": 'button[title*="Split"]',
            }
        },
    }
    operations = [{"action": "split", "segment_index": 1}]
    source_segments = [
        {
            "segment_index": 1,
            "start_sec": 0.0,
            "end_sec": 18.0,
            "current_label": "adjust fabric",
        }
    ]

    result = segments.apply_segment_operations(
        page,
        cfg,
        operations,
        source_segments=source_segments,
    )

    assert result["applied"] == 1
    assert result["structural_applied"] == 1
    assert result["failed"] == []
    assert "s" in [key.lower() for key in page.keyboard.presses]
    assert positioned
    assert all(idx == 1 for idx in positioned)


def test_apply_segment_operations_merge_can_use_previous_row_anchor(monkeypatch) -> None:
    monkeypatch.setattr(
        segments,
        "_resolve_rows_locator",
        lambda page, rows_selector, sample_size=8, row_text_timeout_ms=350: ("rows", page.locator("rows")),
    )
    monkeypatch.setattr(segments._browser, "_dismiss_blocking_modals", lambda page, cfg=None: None)
    monkeypatch.setattr(segments._browser, "_dismiss_blocking_side_panel", lambda page, cfg, aggressive=False: False)
    monkeypatch.setattr(segments._browser, "_selector_variants", lambda selector: [])
    monkeypatch.setattr(segments, "_confirm_action_dialog", lambda page, cfg, action="": False)
    monkeypatch.setattr(
        segments,
        "_wait_rows_delta",
        lambda page, rows_selector, before_count, expected_delta, timeout_ms=0, mode="exact": page.row_count < before_count,
    )

    page = _FakePage()
    page.focused_idx = None

    def _focus_row(page_obj: Any, rows: Any, idx: int, cfg: Dict[str, Any]) -> None:
        page_obj.focused_idx = idx

    monkeypatch.setattr(segments._browser, "_click_segment_row_with_recovery", _focus_row)

    def _merge_press(key: str) -> None:
        page.keyboard.presses.append(key)
        if key.lower() == "m" and getattr(page, "focused_idx", None) == 1 and not page.merge_applied:
            page.row_count -= 1
            page.merge_applied = True

    page.keyboard.press = _merge_press
    cfg: Dict[str, Any] = {
        "run": {
            "structural_allow_merge": True,
            "structural_skip_if_segments_ge": 40,
            "structural_wait_rows_delta_timeout_ms": 50,
            "structural_max_failures_per_episode": 4,
        },
        "atlas": {"selectors": {"segment_rows": "rows"}},
    }

    result = segments.apply_segment_operations(
        page,
        cfg,
        [{"action": "merge", "segment_index": 2}],
        source_segments=[],
    )

    assert result["applied"] == 1
    assert result["structural_applied"] == 1
    assert result["failed"] == []
    assert getattr(page, "focused_idx", None) == 1


def test_extract_segments_skips_full_row_text_when_direct_fields_exist(monkeypatch) -> None:
    monkeypatch.setattr(segments, "_wait_for_segments_stable", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        segments,
        "_resolve_rows_locator",
        lambda page, rows_selector, sample_size=8, row_text_timeout_ms=350: ("rows", page.locator("rows")),
    )

    timeouts: List[int] = []

    def _fake_safe_locator_text(locator: Any, timeout_ms: int = 1200) -> str:
        timeouts.append(int(timeout_ms))
        return ""

    def _fake_first_text_from_row(row: Any, selector: str, timeout_ms: int = 350) -> str:
        selector_l = str(selector or "").lower()
        if "label" in selector_l:
            return "pick up cloth"
        if "start" in selector_l:
            return "0:00.0"
        if "end" in selector_l:
            return "0:01.9"
        return ""

    monkeypatch.setattr(segments._browser, "_safe_locator_text", _fake_safe_locator_text)
    monkeypatch.setattr(segments, "_first_text_from_row", _fake_first_text_from_row)
    monkeypatch.setattr(segments, "_resolve_row_child_selector", lambda rows, selector, **kwargs: selector)

    page = _FakePage()
    page.row_count = 4
    cfg: Dict[str, Any] = {
        "run": {
            "segment_resolve_row_text_timeout_ms": 350,
            "segment_extract_row_text_timeout_ms": 777,
            "segment_extract_progress_every": 2,
        },
        "atlas": {
            "selectors": {
                "segment_rows": "rows",
                "segment_label": "label",
                "segment_start": "start",
                "segment_end": "end",
            }
        },
    }

    out = segments.extract_segments(page, cfg)

    assert len(out) == 4
    assert out[0]["current_label"] == "pick up cloth"
    assert out[0]["start_sec"] == 0.0
    assert out[0]["end_sec"] == 1.9
    assert 777 not in timeouts
    assert page.scroll_timeouts == []


def test_extract_segments_touches_heartbeat_and_uses_scroll_timeout(monkeypatch) -> None:
    monkeypatch.setattr(segments, "_wait_for_segments_stable", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        segments,
        "_resolve_rows_locator",
        lambda page, rows_selector, sample_size=8, row_text_timeout_ms=350: ("rows", page.locator("rows")),
    )

    heartbeats: List[str] = []
    monkeypatch.setattr(
        segments,
        "_active_extract_heartbeat_callback",
        lambda: (lambda: heartbeats.append("beat")),
    )
    monkeypatch.setattr(segments._browser, "_safe_locator_text", lambda locator, timeout_ms=1200: "")
    monkeypatch.setattr(segments, "_resolve_row_child_selector", lambda rows, selector, **kwargs: selector)

    def _fake_first_text_from_row(row: Any, selector: str, timeout_ms: int = 350) -> str:
        return ""

    monkeypatch.setattr(segments, "_first_text_from_row", _fake_first_text_from_row)

    page = _FakePage()
    page.row_count = 3
    cfg: Dict[str, Any] = {
        "run": {
            "segment_row_scroll_timeout_ms": 4321,
            "segment_extract_progress_every": 99,
        },
        "atlas": {
            "selectors": {
                "segment_rows": "rows",
                "segment_label": "label",
                "segment_start": "start",
                "segment_end": "end",
            }
        },
    }

    out = segments.extract_segments(page, cfg)

    assert len(out) == 3
    assert page.scroll_timeouts == [4321, 4321, 4321]
    assert len(heartbeats) >= 3
