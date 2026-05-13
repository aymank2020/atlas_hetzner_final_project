import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infra import submit_verify
from src.solver import browser, legacy_impl, orchestrator, segments
from src.solver.live_validation import ValidationTracker
from src.solver.reliability import EpisodeReport


def test_process_policy_gate_forwards_validation_tracker(monkeypatch, tmp_path: Path):
    tracker = ValidationTracker(
        {"run": {"output_dir": str(tmp_path / "outputs")}},
        "ep-forward",
    )
    tracker.set_initial_state(
        [{"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0}],
        10.0,
    )
    captured = {}

    monkeypatch.setattr(
        orchestrator,
        "_maybe_retry_policy_with_stronger_model",
        lambda **kwargs: {
            "labels_payload": kwargs["labels_payload"],
            "segment_plan": kwargs["segment_plan"],
            "validation_report": kwargs["validation_report"],
            "task_state": kwargs.get("task_state"),
            "retried": False,
            "adopted_retry": False,
        },
    )

    def fake_repair(**kwargs):
        captured["validation_tracker"] = kwargs.get("validation_tracker")
        return {
            "segments": kwargs["segments"],
            "prompt": kwargs["prompt"],
            "labels_payload": kwargs["labels_payload"],
            "segment_plan": kwargs["segment_plan"],
            "validation_report": kwargs["validation_report"],
            "task_state": kwargs.get("task_state"),
            "repair_rounds": 0,
            "skip_compare": False,
            "retry_stage": "",
            "retry_reason": "",
            "repair_skipped_reason": "",
        }

    monkeypatch.setattr(orchestrator, "_maybe_repair_overlong_segments", fake_repair)
    monkeypatch.setattr(
        legacy_impl,
        "_validate_segment_plan_against_policy",
        lambda cfg, segs, plan: {"errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        legacy_impl,
        "_save_validation_report",
        lambda cfg, task_id, report: None,
    )

    result = orchestrator._process_policy_gate_and_compare(
        cfg={"run": {"chat_only_mode": True}},
        page=object(),
        segments=[{"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0}],
        prompt="prompt",
        video_file=None,
        labels_payload={},
        segment_plan={
            1: {
                "segment_index": 1,
                "start_sec": 0.0,
                "end_sec": 4.0,
                "label": "pick up item",
            }
        },
        task_id="ep-forward",
        execute=False,
        validation_tracker=tracker,
    )

    assert captured["validation_tracker"] is tracker
    assert result["errors"] == []


def test_overlong_repair_records_validation_checkpoint(monkeypatch, tmp_path: Path):
    cfg = {
        "run": {
            "output_dir": str(tmp_path / "outputs"),
            "policy_auto_split_repair_enabled": True,
            "policy_auto_split_repair_max_rounds": 1,
            "policy_auto_split_repair_max_segments_per_round": 1,
            "structural_allow_split": True,
            "requery_after_structural_actions": True,
        }
    }
    initial_segments = [
        {"segment_index": 1, "start_sec": 0.0, "end_sec": 12.0, "current_label": "move item"}
    ]
    repaired_segments = [
        {"segment_index": 1, "start_sec": 0.0, "end_sec": 6.0, "current_label": "move item"},
        {"segment_index": 2, "start_sec": 6.0, "end_sec": 12.0, "current_label": "move item"},
    ]
    tracker = ValidationTracker(cfg, "ep-repair")
    tracker.set_initial_state(initial_segments, 10.0)

    monkeypatch.setattr(
        legacy_impl,
        "apply_segment_operations",
        lambda *args, **kwargs: {"applied": 1, "structural_applied": 1, "failed": []},
    )
    monkeypatch.setattr(legacy_impl, "extract_segments", lambda page, cfg: repaired_segments)
    monkeypatch.setattr(legacy_impl, "build_prompt", lambda *args, **kwargs: "prompt-2")
    monkeypatch.setattr(
        legacy_impl,
        "_request_labels_with_optional_segment_chunking",
        lambda *args, **kwargs: {"_meta": {}, "segments": []},
    )
    monkeypatch.setattr(legacy_impl, "_normalize_operations", lambda payload, cfg=None: [])
    monkeypatch.setattr(legacy_impl, "_save_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(legacy_impl, "_save_task_text_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(legacy_impl, "_save_cached_labels", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        legacy_impl,
        "_episode_model_state_updates",
        lambda cfg, payload, task_state=None: {},
    )
    monkeypatch.setattr(
        legacy_impl,
        "_normalize_segment_plan",
        lambda payload, segs, cfg=None: {
            1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 6.0, "label": "move item"},
            2: {"segment_index": 2, "start_sec": 6.0, "end_sec": 12.0, "label": "move item"},
        },
    )
    monkeypatch.setattr(legacy_impl, "_rewrite_no_action_pauses_in_plan", lambda plan, cfg: 0)
    monkeypatch.setattr(
        legacy_impl,
        "_validate_segment_plan_against_policy",
        lambda cfg, segs, plan: {"errors": [], "warnings": []},
    )

    result = orchestrator._maybe_repair_overlong_segments(
        cfg=cfg,
        page=object(),
        segments=initial_segments,
        prompt="prompt",
        video_file=None,
        labels_payload={},
        segment_plan={1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 12.0, "label": "move item"}},
        validation_report={"errors": ["segment 1: duration 12.0s exceeds max 10.0s"], "warnings": []},
        execute=True,
        execute_require_video_context=False,
        validation_tracker=tracker,
    )

    assert result["repair_rounds"] == 1
    assert len(tracker.repair_checkpoints) == 1
    checkpoint = tracker.repair_checkpoints[0]
    assert checkpoint.overlong_indices_before == [1]
    assert checkpoint.overlong_indices_after == []
    assert checkpoint.split_ops_applied == 1
    assert checkpoint.segment_count_after == 2


def test_apply_labels_records_submit_guard_in_validation_tracker(monkeypatch):
    class DummyPage:
        url = "https://audit.atlascapture.io/tasks/room/normal/label/ep-submit"

        def wait_for_timeout(self, ms):
            return None

    page = DummyPage()
    tracker_calls = {"before": [], "after": []}

    class FakeTracker:
        def record_submit_before(self, *args, **kwargs):
            tracker_calls["before"].append((args, kwargs))

        def record_submit_after(self, *args, **kwargs):
            tracker_calls["after"].append((args, kwargs))

    monkeypatch.setattr(
        segments,
        "_resolve_rows_locator",
        lambda page_obj, rows_selector: ("rows", SimpleNamespace(count=lambda: 0)),
    )
    monkeypatch.setattr(
        segments._browser,
        "_dismiss_blocking_modals",
        lambda page_obj, cfg=None: None,
    )
    monkeypatch.setattr(
        segments._browser,
        "_dismiss_blocking_side_panel",
        lambda page_obj, cfg_obj, aggressive=False: None,
    )
    monkeypatch.setattr(
        segments,
        "_pre_submit_duration_check",
        lambda page_obj, cfg_obj, max_dur=10.0: {
            "ok": False,
            "violations": ["segment 1: 12.0s > 10.0s"],
            "segment_count": 1,
        },
    )

    result = segments.apply_labels(
        page,
        {
            "atlas": {
                "selectors": {
                    "segment_rows": "rows",
                    "segment_label": "label",
                    "edit_button_in_row": "edit",
                    "label_input": "input",
                    "save_button": "save",
                    "complete_button": "complete",
                }
            }
        },
        {},
        episode_id="ep-submit",
        segment_plan=None,
        source_segments=None,
        validation_tracker=FakeTracker(),
    )

    assert result["submit_guard_blocked"] is True
    assert len(tracker_calls["before"]) == 1
    assert len(tracker_calls["after"]) == 1
    submit_status = tracker_calls["after"][0][0][0]
    submit_result = tracker_calls["after"][0][0][1]
    assert submit_status["submit_verification_reason"] == "submit_guard_blocked"
    assert submit_result["submit_guard_blocked"] is True


def test_verify_episode_on_dashboard_matches_feedback_view_link(monkeypatch):
    episode_id = "6976c3568d796cd4d377424b"

    class _LinkItem:
        def __init__(self, href: str, text: str):
            self._href = href
            self._text = text

        def get_attribute(self, name: str):
            if name == "href":
                return self._href
            return None

        def inner_text(self, timeout: int = 0):
            return self._text

    class _LinkCollection:
        def __init__(self, items):
            self._items = items

        def count(self):
            return len(self._items)

        def nth(self, idx: int):
            return self._items[idx]

    class DummyPage:
        def __init__(self):
            self.url = "https://audit.atlascapture.io/tasks/room/normal/label/" + episode_id

        def inner_text(self, selector):
            assert selector == "body"
            return "Feedback Your Episodes Episode d377424b View"

        def wait_for_timeout(self, ms):
            return None

        def locator(self, selector):
            if selector == "a[href]":
                return _LinkCollection([_LinkItem(f"/feedback/{episode_id}", "View")])
            return _LinkCollection([])

    monkeypatch.setattr(browser, "_goto_with_retry", lambda page, url, **kwargs: True)

    result = submit_verify.verify_episode_on_dashboard(
        DummyPage(),
        {},
        episode_id=episode_id,
        timeout_sec=3.0,
    )

    assert result.verified is True
    assert result.method == "dashboard_link"
    assert episode_id in result.detail


def test_verify_episode_on_dashboard_matches_truncated_episode_suffix_in_body(monkeypatch):
    episode_id = "6976c8b52d3b724b89fee00b"

    class _LinkCollection:
        def count(self):
            return 0

        def nth(self, idx: int):
            raise IndexError(idx)

    class DummyPage:
        url = "https://audit.atlascapture.io/feedback"

        def inner_text(self, selector):
            assert selector == "body"
            return "Feedback Your Episodes Episode 89fee00b T2 Awaiting T3 Today View"

        def wait_for_timeout(self, ms):
            return None

        def locator(self, selector):
            assert selector == "a[href]"
            return _LinkCollection()

    monkeypatch.setattr(browser, "_goto_with_retry", lambda page, url, **kwargs: True)

    result = submit_verify.verify_episode_on_dashboard(
        DummyPage(),
        {},
        episode_id=episode_id,
        timeout_sec=3.0,
    )

    assert result.verified is True
    assert result.method == "dashboard_scan_suffix"
    assert "89fee00b" in result.detail


def test_apply_labels_emits_apply_start_progress_without_duplicate_event_type(monkeypatch):
    class DummyPage:
        url = "https://audit.atlascapture.io/tasks/room/normal/label/ep-progress"

        def wait_for_timeout(self, ms):
            return None

    page = DummyPage()
    progress_events = []

    monkeypatch.setattr(
        segments,
        "_resolve_rows_locator",
        lambda page_obj, rows_selector: ("rows", SimpleNamespace(count=lambda: 0)),
    )
    monkeypatch.setattr(
        segments._browser,
        "_dismiss_blocking_modals",
        lambda page_obj, cfg=None: None,
    )
    monkeypatch.setattr(
        segments._browser,
        "_dismiss_blocking_side_panel",
        lambda page_obj, cfg_obj, aggressive=False: None,
    )
    monkeypatch.setattr(
        segments,
        "_pre_submit_duration_check",
        lambda page_obj, cfg_obj, max_dur=10.0: {
            "ok": True,
            "violations": [],
            "segment_count": 0,
        },
    )
    monkeypatch.setattr(
        segments._browser,
        "_click_submit_with_verification",
        lambda *args, **kwargs: {"verification": {"submit_verified": False, "method": "not_clicked"}},
    )

    result = segments.apply_labels(
        page,
        {
            "atlas": {
                "selectors": {
                    "segment_rows": "rows",
                    "segment_label": "label",
                    "edit_button_in_row": "edit",
                    "label_input": "input",
                    "save_button": "save",
                    "complete_button": "complete",
                }
            }
        },
        {},
        episode_id="ep-progress",
        progress_callback=lambda event_type, payload: progress_events.append((event_type, payload)),
    )

    assert result["applied"] == 0
    assert progress_events
    first_event_type, first_payload = progress_events[0]
    assert first_event_type == "apply_start"
    assert first_payload["episode_id"] == "ep-progress"
    assert first_payload["budget_sec"] >= 30.0
    assert "no_progress_timeout_sec" in first_payload


def test_apply_submit_guard_counts_skipped_unchanged_as_covered_work():
    reasons = segments._evaluate_apply_submit_guard(
        total_targets=20,
        applied=17,
        skipped_unchanged=3,
        failed=[],
        submit_guard_enabled=True,
        submit_guard_max_failure_ratio=0.25,
        submit_guard_min_applied_ratio=0.90,
        submit_guard_block_on_budget_exceeded=True,
    )

    assert reasons == []


def test_apply_submit_guard_blocks_when_covered_ratio_is_too_low():
    reasons = segments._evaluate_apply_submit_guard(
        total_targets=20,
        applied=15,
        skipped_unchanged=1,
        failed=[],
        submit_guard_enabled=True,
        submit_guard_max_failure_ratio=0.25,
        submit_guard_min_applied_ratio=0.90,
        submit_guard_block_on_budget_exceeded=True,
    )

    assert reasons == ["covered ratio 80.0% < 90.0%"]


def test_finalize_current_episode_v2_saves_live_validation_report(tmp_path: Path):
    cfg = {"run": {"structured_episode_reports": True, "output_dir": str(tmp_path / "outputs")}}
    tracker = ValidationTracker(cfg, "ep99")
    tracker.set_initial_state(
        [{"segment_index": 1, "start_sec": 0.0, "end_sec": 12.0}],
        10.0,
    )
    tracker.record_submit_before(
        "https://audit.atlascapture.io/tasks/room/normal/label/ep99",
        "button.complete",
    )
    tracker.record_submit_after(
        {
            "submit_verified": False,
            "submit_verification_reason": "submit_guard_blocked",
            "complete_button_clicked": False,
        },
        {
            "submit_guard_blocked": True,
            "submit_guard_reasons": ["live DOM contains overlong segments before submit"],
        },
    )

    report = EpisodeReport(episode_id="ep99", context_id="ctx99")
    page, context = legacy_impl._finalize_current_episode_v2(
        cfg=cfg,
        report=report,
        task_state={},
        runtime=None,
        bootstrap_page=None,
        bootstrap_context=None,
        room_url="",
        page=None,
        segments=[{"segment_index": 1, "start_sec": 0.0, "end_sec": 12.0}],
        validation_report={"errors": ["segment 1: duration 12.0s exceeds max 10.0s"], "warnings": []},
        result={"submit_guard_blocked": True},
        validation_tracker=tracker,
        reason="test-finalize",
    )

    assert page is None
    assert context is None

    episode_report_path = tmp_path / "outputs" / "episode_reports" / "episode_ep99.json"
    payload = json.loads(episode_report_path.read_text(encoding="utf-8"))
    assert payload["live_validation_report_path"]

    live_report_path = Path(payload["live_validation_report_path"])
    assert live_report_path.exists()
    live_payload = json.loads(live_report_path.read_text(encoding="utf-8"))
    assert live_payload["episode_id"] == "ep99"
    assert live_payload["submit_succeeded"] is False
    assert live_payload["validation_passed"] is False
