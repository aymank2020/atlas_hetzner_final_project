"""Exercise corrupt extraction through the real pre-submit consumer, offline."""

from types import SimpleNamespace

import pytest

from src.rules.consistency import validate_pre_submit_consistency
from src.solver import segments


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "invalid"])
@pytest.mark.parametrize("field", ["start_sec", "end_sec"])
def test_matching_invalid_timestamps_block_submit(monkeypatch, bad, field):
    extracted = [{"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0}]
    extracted[0][field] = bad
    monkeypatch.setattr(segments, "extract_segments", lambda *_: extracted)
    result = validate_pre_submit_consistency(SimpleNamespace(url="offline"), {}, {1: {"label": "pick up cup"}}, extracted)
    assert result["consistent"] is False
    assert any("finite numbers" in message for message in result["mismatches"])
    assert result["desync_decision"]["requires_reextract"] is True


@pytest.mark.parametrize("extracted", [
    [],
    [{"segment_index": 0, "start_sec": 0, "end_sec": 4}],
    [{"segment_index": 1, "start_sec": 0, "end_sec": 4}] * 2,
    [{"segment_index": 1, "start_sec": 4, "end_sec": 4}],
    [{"segment_index": 1, "start_sec": 5, "end_sec": 4}],
    [{"segment_index": 1, "start_sec": -1, "end_sec": 4}],
])
def test_matching_unusable_snapshots_block_submit(monkeypatch, extracted):
    monkeypatch.setattr(segments, "extract_segments", lambda *_: extracted)
    result = validate_pre_submit_consistency(SimpleNamespace(url="offline"), {}, {1: {"label": "pick up cup"}} if extracted else {}, extracted)
    assert result["consistent"] is False
    assert result["mismatches"]


def test_valid_dom_remains_authoritative_over_ai_timestamps(monkeypatch):
    extracted = [{"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0}]
    monkeypatch.setattr(segments, "extract_segments", lambda *_: extracted)
    result = validate_pre_submit_consistency(SimpleNamespace(url="offline"), {}, {1: {"start_sec": 0, "end_sec": 99, "label": "pick up cup"}}, extracted)
    assert result["consistent"] is True
    assert result["warnings"]


@pytest.mark.parametrize("existing_url", ["https://gemini.google.com/app/other", "https://gemini.google.com/app/thread-extra"])
def test_episode_runtime_does_not_borrow_another_conversation(existing_url):
    from src.solver.episode_runtime import EpisodeRuntime

    class Page:
        def __init__(self, url=""):
            self.url = url
        def goto(self, url, **kwargs):
            self.url = url

    existing = Page(existing_url)
    context = SimpleNamespace(pages=[existing], new_page=lambda: Page())
    runtime = EpisodeRuntime("offline").open(gemini_existing_context=context, gemini_page_url="https://gemini.google.com/app/thread")
    assert runtime.gemini_page is not existing
    assert runtime.gemini_page.url == "https://gemini.google.com/app/thread"
    assert existing.url == existing_url


def test_apply_labels_blocks_empty_snapshot_before_submit_click(monkeypatch):
    page = SimpleNamespace(url="offline", wait_for_timeout=lambda milliseconds: None)
    monkeypatch.setattr(segments, "_resolve_rows_locator", lambda *_: ("rows", SimpleNamespace(count=lambda: 0)))
    monkeypatch.setattr(segments, "extract_segments", lambda *_: [])
    monkeypatch.setattr(segments._browser, "_dismiss_blocking_modals", lambda *args, **kwargs: None)
    monkeypatch.setattr(segments._browser, "_dismiss_blocking_side_panel", lambda *args, **kwargs: None)
    monkeypatch.setattr(segments, "_pre_submit_duration_check", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(segments._browser, "_click_submit_with_verification", lambda *args, **kwargs: pytest.fail("unsafe submit was attempted"))
    result = segments.apply_labels(page, {"atlas": {"selectors": {"complete_button": "complete"}}}, {}, segment_plan={}, source_segments=[])
    assert result["submit_guard_blocked"] is True
    assert result["completed"] is False
    assert any("contains no segments" in reason for reason in result["submit_guard_reasons"])
