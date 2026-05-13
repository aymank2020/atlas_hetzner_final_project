import subprocess
import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import atlas_triplet_compare as _chat

from src.infra import execution_journal
from src.infra.solver_config import DEFAULT_CONFIG
from src.solver import chat_only, orchestrator, video_core
from src.solver.desync import build_segment_snapshot, compare_segment_snapshots, warn_on_plan_vs_live
from src.solver.gemini_session import validate_normalized_segments
from src.solver.episode_runtime import EpisodeRuntime
from src.solver.reliability import ApplyBudgetState, EpisodeReport, RetryReason, classify_transport_failure
from src.solver import legacy_impl


def test_v2_default_flags_present():
    run_cfg = DEFAULT_CONFIG["run"]
    gem_cfg = DEFAULT_CONFIG["gemini"]

    assert run_cfg["use_episode_runtime_v2"] is False
    assert run_cfg["strict_single_chat_session"] is False
    assert run_cfg["force_episode_browser_isolation"] is False
    assert run_cfg["gemini_transport_max_retries"] == 3
    assert run_cfg["gemini_transport_max_retries_ops"] == 2
    assert run_cfg["gemini_scope_followup_attempts"] == 1
    assert run_cfg["gemini_schema_followup_attempts"] == 1
    assert run_cfg["chat_ops_synthesize_split_fallback"] is True
    assert run_cfg["targeted_repair_max_rounds"] == 3
    assert run_cfg["targeted_repair_scope_neighbors"] == 2
    assert run_cfg["chat_ops_timeout_sec"] == 300.0
    assert run_cfg["structured_episode_reports"] is False
    assert run_cfg["live_validation_enabled"] is False
    assert run_cfg["submit_manual_watch_enabled"] is False
    assert run_cfg["submit_manual_watch_timeout_sec"] == 180.0
    assert run_cfg["submit_manual_watch_poll_ms"] == 500
    assert run_cfg["submit_manual_watch_log_interval_sec"] == 10.0
    assert run_cfg["desync_snapshot_tolerance_sec"] == 0.25
    assert run_cfg["sticky_episode_resume"] is False
    assert run_cfg["disable_release_all_during_canary"] is False
    assert run_cfg["single_window_two_tabs"] is False
    assert run_cfg["single_window_single_tab"] is False
    assert run_cfg["hold_rule_context_neighbors"] == 2
    assert gem_cfg["chat_web_url"] == "https://gemini.google.com/app"
    assert gem_cfg["chat_web_storage_state"] == ".state/gemini_chat_storage_state.json"
    assert gem_cfg["chat_web_clean_thread_on_episode_start"] is True
    assert gem_cfg["optimize_video_target_mb"] == 15.0
    assert gem_cfg["optimize_video_prefer_ffmpeg"] is True
    assert gem_cfg["inline_read_bytes_max_mb"] == 8.0
    assert gem_cfg["zero_quota_model_cache_max_entries"] == 50


def test_connect_atlas_browser_context_falls_back_to_playwright_launch(monkeypatch, tmp_path: Path):
    state_path = tmp_path / "atlas_state.json"
    state_path.write_text('{"cookies":[],"origins":[]}', encoding="utf-8")

    class FakePage:
        def __init__(self, url: str = "about:blank"):
            self.url = url
            self.closed = False

        def close(self):
            self.closed = True

    class FakeContext:
        def __init__(self):
            self.pages = []
            self.new_page_calls = 0

        def new_page(self):
            self.new_page_calls += 1
            page = FakePage()
            self.pages.append(page)
            return page

    class FakeBrowser:
        def __init__(self):
            self.contexts = []
            self.new_context_kwargs = None

        def new_context(self, **kwargs):
            self.new_context_kwargs = kwargs
            context = FakeContext()
            self.contexts.append(context)
            return context

    class FakeChromium:
        def __init__(self):
            self.launch_kwargs = None
            self.browser = FakeBrowser()

        def connect_over_cdp(self, _url, timeout=None):
            raise RuntimeError("cdp unavailable")

        def launch(self, **kwargs):
            self.launch_kwargs = kwargs
            return self.browser

    class FakePW:
        def __init__(self):
            self.chromium = FakeChromium()

    monkeypatch.setattr(legacy_impl.time, "sleep", lambda _sec: None)

    pw = FakePW()
    browser, context, page, mode = legacy_impl._connect_atlas_browser_context(
        pw,
        cdp_url="http://127.0.0.1:9222",
        cdp_connect_timeout_ms=1234,
        state_path=state_path,
        headless=False,
        slow_mo=40,
        chrome_channel="chrome",
        browser_executable_path="",
        browser_proxy={"server": "http://proxy:8080"},
    )

    assert mode == "playwright_fallback"
    assert browser is pw.chromium.browser
    assert context is pw.chromium.browser.contexts[0]
    assert page is context.pages[0]
    assert pw.chromium.browser.new_context_kwargs == {"storage_state": str(state_path)}
    assert pw.chromium.launch_kwargs["channel"] == "chrome"
    assert pw.chromium.launch_kwargs["proxy"] == {"server": "http://proxy:8080"}
    assert "--disable-blink-features=AutomationControlled" in pw.chromium.launch_kwargs["args"]


def test_launch_local_shared_chrome_cdp_uses_profile_clone_and_both_urls(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {"ready": 0}

    def fake_ready(_cdp_url: str, *, timeout_sec: float = 1.5) -> bool:
        calls["ready"] = int(calls["ready"]) + 1
        return int(calls["ready"]) >= 2

    launched: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(args, **kwargs):
        launched.append((list(args), dict(kwargs)))
        return SimpleNamespace(pid=12345)

    def fake_clone(source_user_data_dir, profile_directory, target_user_data_dir, reuse_existing=True):
        assert source_user_data_dir == str(tmp_path / "source")
        assert profile_directory == "Profile 7"
        assert target_user_data_dir == str(tmp_path / "clone")
        assert reuse_existing is True
        return str(tmp_path / "clone-ready")

    monkeypatch.setattr(legacy_impl, "_cdp_endpoint_ready", fake_ready)
    monkeypatch.setattr(legacy_impl, "_resolve_local_chrome_executable", lambda **_kwargs: "C:/Chrome/chrome.exe")
    monkeypatch.setattr(legacy_impl, "_prepare_chrome_profile_clone", fake_clone)
    monkeypatch.setattr(legacy_impl.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(legacy_impl.time, "sleep", lambda _sec: None)

    launched_ok = legacy_impl._launch_local_shared_chrome_cdp(
        cdp_url="http://127.0.0.1:9222",
        browser_executable_path="",
        chrome_channel="chrome",
        use_chrome_profile=True,
        chrome_user_data_dir=str(tmp_path / "source"),
        chrome_profile_directory="Profile 7",
        clone_chrome_profile_to_temp=True,
        cloned_user_data_dir=str(tmp_path / "clone"),
        reuse_existing_cloned_profile=True,
        prefer_profile_with_atlas_cookies=False,
        atlas_email="",
        close_chrome_before_profile_launch=False,
        profile_launch_timeout_ms=5000,
        start_urls=[
            "https://audit.atlascapture.io/tasks",
            "https://gemini.google.com/app/b3006ba9f325b55c",
        ],
    )

    assert launched_ok is True
    assert len(launched) == 1
    args, kwargs = launched[0]
    assert args[0] == "C:/Chrome/chrome.exe"
    assert "--remote-debugging-port=9222" in args
    assert f"--user-data-dir={tmp_path / 'clone-ready'}" in args
    assert "--profile-directory=Profile 7" in args
    assert "https://audit.atlascapture.io/tasks" in args
    assert "https://gemini.google.com/app/b3006ba9f325b55c" in args
    assert "stdout" in kwargs and "stderr" in kwargs


def test_episode_runtime_reuses_existing_atlas_page_without_opening_new_tab():
    class FakePage:
        def __init__(self, url: str = "https://audit.atlascapture.io/tasks"):
            self.url = url
            self.goto_calls = []

        def goto(self, url: str, wait_until: str = "", timeout: int = 0):
            self.goto_calls.append((url, wait_until, timeout))
            self.url = url

    class FakeContext:
        def __init__(self):
            self.new_page_calls = 0

        def new_page(self):
            self.new_page_calls += 1
            return FakePage()

    context = FakeContext()
    page = FakePage()

    runtime = EpisodeRuntime("task-single-tab").open(
        atlas_existing_context=context,
        atlas_existing_page=page,
        atlas_page_url="https://audit.atlascapture.io/tasks/room/normal/label/task-single-tab",
    )

    assert runtime.atlas_page is page
    assert runtime.atlas_page_borrowed is True
    assert runtime.atlas_context is context
    assert context.new_page_calls == 0
    assert page.goto_calls == [
        ("https://audit.atlascapture.io/tasks/room/normal/label/task-single-tab", "domcontentloaded", 60000)
    ]


def test_episode_runtime_prefers_authenticated_existing_gemini_page(monkeypatch):
    class FakePage:
        def __init__(self, url: str, authenticated: bool):
            self.url = url
            self.authenticated = authenticated

        def title(self):
            return "Google Gemini"

        @property
        def locator(self):
            raise AssertionError("locator should not be used when auth check is monkeypatched")

    class FakeContext:
        def __init__(self, pages):
            self.pages = pages
            self.new_page_calls = 0

        def new_page(self):
            self.new_page_calls += 1
            page = FakePage("https://gemini.google.com/app", False)
            self.pages.append(page)
            return page

    monkeypatch.setattr(
        "src.solver.episode_runtime._is_authenticated_gemini_page",
        lambda page: bool(getattr(page, "authenticated", False)),
    )

    unauthenticated = FakePage("https://gemini.google.com/app", False)
    authenticated = FakePage("https://gemini.google.com/app", True)
    context = FakeContext([unauthenticated, authenticated])

    runtime = EpisodeRuntime("task-auth-page").open(
        gemini_existing_context=context,
        gemini_page_url="https://gemini.google.com/app",
    )

    assert runtime.gemini_page is authenticated
    assert runtime.gemini_page_borrowed is True
    assert context.new_page_calls == 0


def test_gemini_session_cleans_pinned_conversation_on_session_init_by_default():
    runtime = SimpleNamespace(
        episode_id="ep-custom-chat",
        context_id="ctx-custom-chat",
        gemini_page=None,
        gemini_context=None,
        gemini_browser=None,
    )
    session = chat_only.GeminiSession(
        runtime=runtime,
        cfg={
            "gemini": {
                "chat_web_url": "https://gemini.google.com/app/b3006ba9f325b55c",
                "chat_web_clean_thread_on_episode_start": True,
            },
            "run": {
                "strict_single_chat_session": True,
            },
        },
    )

    chat_url = session._chat_url()

    assert session._preserve_existing_thread(chat_url) is False
    assert session._should_clean_thread_on_session_init(chat_url) is True


def test_gemini_session_request_integrity_requires_confirmed_video_upload(tmp_path: Path):
    runtime = SimpleNamespace(
        episode_id="ep-attach",
        context_id="ctx-attach",
        gemini_page=None,
        gemini_context=None,
        gemini_browser=None,
        task_state={},
    )
    session = chat_only.GeminiSession(runtime=runtime, cfg={"gemini": {}, "run": {}})
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"00")

    with pytest.raises(RuntimeError, match="video upload was not confirmed"):
        session._assert_request_integrity(
            video_file=video_file,
            attach_notes=["video.mp4: skipped (file chooser did not appear)"],
            acceptance_metadata={"baseline_advanced": True},
        )


def test_gemini_session_request_integrity_rejects_response_that_did_not_advance_baseline():
    runtime = SimpleNamespace(
        episode_id="ep-baseline",
        context_id="ctx-baseline",
        gemini_page=None,
        gemini_context=None,
        gemini_browser=None,
        task_state={},
    )
    session = chat_only.GeminiSession(runtime=runtime, cfg={"gemini": {}, "run": {}})

    with pytest.raises(RuntimeError, match="did not advance baseline"):
        session._assert_request_integrity(
            video_file=None,
            attach_notes=[],
            acceptance_metadata={"baseline_advanced": False},
        )


def test_apply_budget_state_round_trips_and_execution_journal_tracks_peak_rss(monkeypatch, tmp_path: Path):
    budget = ApplyBudgetState(
        target_count=5,
        applied_count=1,
        skipped_count=1,
        last_progress_at=50.0,
        consecutive_failures=0,
        deadline_at=200.0,
        budget_extensions=1,
        started_at=25.0,
        status="active",
    )
    round_trip = ApplyBudgetState.from_dict(budget.to_dict())

    assert round_trip.target_count == 5
    assert round_trip.applied_count == 1
    assert round_trip.skipped_count == 1
    assert round_trip.deadline_at == 200.0
    assert round_trip.budget_extensions == 1
    assert round_trip.status == "active"

    rss_values = iter([64.0, 61.5])
    monkeypatch.setattr(execution_journal, "_PEAK_RSS_MB", 0.0)
    monkeypatch.setattr(execution_journal, "_current_rss_mb", lambda: next(rss_values))

    cfg = {"run": {"output_dir": str(tmp_path / "outputs")}}
    target = execution_journal.append_execution_journal_event(
        cfg,
        episode_id="ep-journal",
        event_type="apply_progress",
        stage="applying",
        payload={"apply_budget_state": budget.to_dict()},
    )
    execution_journal.append_execution_journal_event(
        cfg,
        episode_id="ep-journal",
        event_type="apply_complete",
        stage="completed",
        payload={"apply_budget_state": budget.to_dict()},
    )

    lines = [
        json.loads(line)
        for line in Path(target).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    assert lines[0]["rss_mb"] == 64.0
    assert lines[0]["peak_rss_mb"] == 64.0
    assert lines[1]["rss_mb"] == 61.5
    assert lines[1]["peak_rss_mb"] == 64.0
    assert lines[0]["apply_budget_state"]["target_count"] == 5


def test_validate_normalized_segments_anchors_timestamps_to_source_dom():
    validated, errors = validate_normalized_segments(
        [
            {
                "segment_index": 1,
                "start_sec": 4.2,
                "end_sec": 9.9,
                "label": "hold nozzle, insert fuel nozzle into tank",
            }
        ],
        [
            {
                "segment_index": 1,
                "start_sec": 0.0,
                "end_sec": 1.2,
                "current_label": "",
            }
        ],
    )

    assert errors == []
    assert validated == [
        {
            "segment_index": 1,
            "start_sec": 0.0,
            "end_sec": 1.2,
            "label": "hold nozzle, insert fuel nozzle into tank",
        }
    ]


def test_persist_task_stage_status_records_terminal_failure_kind_and_canonical_state(monkeypatch):
    persisted = {}
    journal_events = []

    def fake_persist(_cfg, _task, task_state, **updates):
        merged = dict(task_state or {})
        merged.update(updates)
        persisted.update(merged)
        return merged

    monkeypatch.setattr(legacy_impl, "_persist_task_state_fields", fake_persist)
    monkeypatch.setattr(
        legacy_impl,
        "append_execution_journal_event",
        lambda *args, **kwargs: journal_events.append(kwargs) or "journal.jsonl",
    )

    state = legacy_impl._persist_task_stage_status(
        {},
        "task-terminal",
        {},
        stage="submit",
        status="failed",
        detail="submit verification failed",
        last_error="submit_unverified",
        terminal_failure_kind="terminal_submit_failure",
    )

    assert state["stage"] == "failed_terminal"
    assert state["status"] == "failed_terminal"
    assert state["terminal_failure_kind"] == "terminal_submit_failure"
    assert state["current_stage"] == "submit"
    assert journal_events[-1]["stage"] == "failed_terminal"


def test_maybe_clear_sticky_resume_targets_unlocks_stale_task(monkeypatch):
    persisted = []
    task_id = "69112e6b28bb03174e8d3adb"

    def fake_persist(_cfg, task_id, task_state=None, **updates):
        persisted.append((task_id, dict(updates)))
        return {"task_id": task_id, **updates}

    monkeypatch.setattr(legacy_impl, "_persist_task_state_fields", fake_persist)

    urls, ids, cleared = legacy_impl._maybe_clear_sticky_resume_targets(
        {},
        [f"https://audit.atlascapture.io/tasks/room/normal/label/{task_id}"],
        [task_id],
        {"sticky_resume_exhausted": True},
    )

    assert cleared is True
    assert urls == []
    assert ids == []
    assert persisted == [
        (
            task_id,
            {
                "episode_locked": False,
                "last_error": "sticky_resume_exhausted",
                "terminal_failure_kind": "sticky_resume_exhausted",
            },
        )
    ]


def test_can_resume_sticky_task_state_requires_locked_nonterminal_episode():
    assert legacy_impl._can_resume_sticky_task_state(
        {
            "task_id": "69112e6b28bb03174e8d3adb",
            "task_url": "https://audit.atlascapture.io/tasks/room/normal/label/69112e6b28bb03174e8d3adb",
            "episode_locked": True,
            "status": "running",
            "episode_status": "running",
            "episode_submitted": False,
            "terminal_failure_kind": "",
            "last_error": "",
        }
    ) is True

    assert legacy_impl._can_resume_sticky_task_state(
        {
            "task_id": "69112e6b28bb03174e8d3adb",
            "task_url": "https://audit.atlascapture.io/tasks/room/normal/label/69112e6b28bb03174e8d3adb",
            "episode_locked": False,
            "status": "running",
            "episode_status": "running",
        }
    ) is False

    assert legacy_impl._can_resume_sticky_task_state(
        {
            "task_id": "69112e6b28bb03174e8d3adb",
            "task_url": "https://audit.atlascapture.io/tasks/room/normal/label/69112e6b28bb03174e8d3adb",
            "episode_locked": True,
            "status": "running",
            "episode_status": "running",
            "terminal_failure_kind": "terminal_submit_failure",
        }
    ) is False


def test_policy_auto_repair_reports_dry_run_skip_reason():
    result = orchestrator._maybe_repair_overlong_segments(
        cfg={
            "run": {
                "policy_auto_split_repair_enabled": True,
                "structural_allow_split": True,
                "requery_after_structural_actions": True,
                "policy_auto_split_repair_max_rounds": 2,
            }
        },
        page=None,
        segments=[{"segment_index": 1, "start_sec": 0.0, "end_sec": 20.0}],
        prompt="prompt",
        video_file=None,
        labels_payload={},
        segment_plan={1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 20.0, "label": "move item"}},
        validation_report={"errors": ["segment 1: duration 20.0s exceeds max 10.0s"], "warnings": []},
        execute=False,
        enable_structural_actions=True,
        requery_after_structural_actions=True,
    )

    assert "execute=false" in result["repair_skipped_reason"]


def test_persist_task_stage_status_clears_stale_failed_episode_state(monkeypatch):
    captured = {}

    def fake_persist(_cfg, _task, task_state, **updates):
        merged = dict(task_state or {})
        merged.update(updates)
        captured.update(merged)
        return merged

    monkeypatch.setattr(legacy_impl, "_persist_task_state_fields", fake_persist)

    state = legacy_impl._persist_task_stage_status(
        {},
        "task-1",
        {
            "episode_status": "failed",
            "last_error": "old failure",
            "current_stage_completed_at_utc": "2026-04-17T00:00:00Z",
        },
        stage="chat_labels",
        status="running",
        progress_current=2,
        progress_total=10,
        detail="waiting for Gemini chunk 1/5",
    )

    assert state["episode_status"] == "running"
    assert state["last_error"] == ""
    assert state["current_stage_completed_at_utc"] == ""


def test_video_optimizer_prefers_ffmpeg_when_available(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"0" * (9 * 1024 * 1024))
    calls = []

    monkeypatch.setattr(video_core, "_probe_video_stream_meta", lambda _path: (854, 480, 30.0, 300))
    monkeypatch.setattr(video_core, "_opencv_available", lambda: True)
    monkeypatch.setattr(video_core, "_resolve_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr(video_core, "_quality_preserving_scale_candidates", lambda **_kwargs: [0.75])
    monkeypatch.setattr(video_core, "_is_probably_mp4", lambda _path: True)

    def fake_ffmpeg(**kwargs):
        calls.append(("ffmpeg", kwargs["scale"]))
        kwargs["dst"].write_bytes(b"1" * (2 * 1024 * 1024))
        return True, ""

    def fake_cv2(**kwargs):
        calls.append(("cv2", kwargs["scale"]))
        kwargs["dst"].write_bytes(b"2" * (3 * 1024 * 1024))
        return True

    monkeypatch.setattr(video_core, "_transcode_video_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(video_core, "_transcode_video_cv2", fake_cv2)

    result = video_core._maybe_optimize_video_for_upload(
        source,
        cfg={
            "gemini": {
                "optimize_video_for_upload": True,
                "optimize_video_only_if_larger_mb": 8.0,
                "optimize_video_target_mb": 15.0,
                "optimize_video_prefer_ffmpeg": True,
                "optimize_video_target_fps": 10.0,
                "optimize_video_min_fps": 8.0,
                "optimize_video_min_width": 320,
                "optimize_video_min_short_side": 320,
                "optimize_video_scale_candidates": [0.75],
            }
        },
    )

    assert result.name == "source_upload_opt.mp4"
    assert calls[0][0] == "ffmpeg"
    assert all(name != "cv2" for name, _scale in calls)


def test_resolve_ffmpeg_binary_detects_winget_link(monkeypatch, tmp_path: Path):
    local_app_data = tmp_path / "LocalAppData"
    winget_dir = local_app_data / "Microsoft" / "WinGet" / "Links"
    winget_dir.mkdir(parents=True)
    ffmpeg_path = winget_dir / "ffmpeg.exe"
    ffmpeg_path.write_bytes(b"")

    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.delenv("USERPROFILE", raising=False)
    monkeypatch.setattr(video_core.shutil, "which", lambda _candidate: "")

    resolved = video_core._resolve_ffmpeg_binary()

    assert resolved == str(ffmpeg_path)


def test_legacy_impl_resolve_ffmpeg_binary_detects_winget_link(monkeypatch, tmp_path: Path):
    local_app_data = tmp_path / "LocalAppData"
    winget_dir = local_app_data / "Microsoft" / "WinGet" / "Links"
    winget_dir.mkdir(parents=True)
    ffmpeg_path = winget_dir / "ffmpeg.exe"
    ffmpeg_path.write_bytes(b"")

    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.delenv("USERPROFILE", raising=False)
    monkeypatch.setattr(legacy_impl.shutil, "which", lambda _candidate: "")

    resolved = legacy_impl._resolve_ffmpeg_binary()

    assert resolved == str(ffmpeg_path)


def test_legacy_impl_video_optimizer_prefers_ffmpeg_when_available(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"0" * (9 * 1024 * 1024))
    calls = []

    monkeypatch.setattr(legacy_impl, "_probe_video_stream_meta", lambda _path: (854, 480, 30.0, 300))
    monkeypatch.setattr(legacy_impl, "_opencv_available", lambda: True)
    monkeypatch.setattr(legacy_impl, "_resolve_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr(legacy_impl, "_quality_preserving_scale_candidates", lambda **_kwargs: [0.75])
    monkeypatch.setattr(legacy_impl, "_is_probably_mp4", lambda _path: True)

    def fake_ffmpeg(**kwargs):
        calls.append(("ffmpeg", kwargs["scale"]))
        kwargs["dst"].write_bytes(b"1" * (2 * 1024 * 1024))
        return True, ""

    def fake_cv2(**kwargs):
        calls.append(("cv2", kwargs["scale"]))
        kwargs["dst"].write_bytes(b"2" * (3 * 1024 * 1024))
        return True

    monkeypatch.setattr(legacy_impl, "_transcode_video_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(legacy_impl, "_transcode_video_cv2", fake_cv2)

    result = legacy_impl._maybe_optimize_video_for_upload(
        source,
        cfg={
            "gemini": {
                "optimize_video_for_upload": True,
                "optimize_video_only_if_larger_mb": 8.0,
                "optimize_video_target_mb": 15.0,
                "optimize_video_prefer_ffmpeg": True,
                "optimize_video_target_fps": 10.0,
                "optimize_video_min_fps": 8.0,
                "optimize_video_min_width": 320,
                "optimize_video_min_short_side": 320,
                "optimize_video_scale_candidates": [0.75],
            }
        },
    )

    assert result.name == "source_upload_opt.mp4"
    assert calls[0][0] == "ffmpeg"
    assert all(name != "cv2" for name, _scale in calls)


def test_desync_detector_blocks_live_source_drift_and_warns_plan_only():
    source_segments = [
        {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "move item"},
        {"segment_index": 2, "start_sec": 5.0, "end_sec": 9.0, "current_label": "place item"},
    ]
    live_segments = [
        {"segment_index": 1, "start_sec": 0.0, "end_sec": 7.0, "current_label": "move item"},
        {"segment_index": 2, "start_sec": 7.0, "end_sec": 9.0, "current_label": "place item"},
    ]
    plan_segments = {
        1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 15.0, "label": "move item"},
        2: {"segment_index": 2, "start_sec": 5.0, "end_sec": 9.0, "label": "place item"},
    }

    decision = compare_segment_snapshots(
        live_snapshot=build_segment_snapshot(segments=live_segments, source_kind="live_dom"),
        source_snapshot=build_segment_snapshot(segments=source_segments, source_kind="extracted_source"),
        tolerance_sec=0.25,
    )
    warnings = warn_on_plan_vs_live(
        plan_segments=plan_segments,
        live_snapshot=build_segment_snapshot(segments=source_segments, source_kind="live_dom"),
    )

    assert decision.ok is False
    assert decision.desync_detected is True
    assert any("segment 1" in item for item in decision.blocking_mismatches)
    assert any("segment 1" in item for item in warnings)


def test_validate_normalized_segments_rejects_hallucinated_indices():
    source_segments = [
        {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0},
        {"segment_index": 2, "start_sec": 5.0, "end_sec": 9.0},
    ]
    normalized_segments = [
        {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "move item"},
        {"segment_index": 9, "start_sec": 5.0, "end_sec": 9.0, "label": "ghost item"},
    ]

    validated, errors = validate_normalized_segments(normalized_segments, source_segments)

    assert len(validated) == 1
    assert any("unknown segment 9" in item for item in errors)


def test_transport_classifier_treats_missing_chat_input_as_page_crash():
    reason = classify_transport_failure("Gemini chat input not visible on session page. url=https://gemini.google.com/app")
    assert reason == RetryReason.PAGE_CRASH


def test_transport_classifier_treats_retryable_chat_error_as_page_crash():
    reason = classify_transport_failure("Gemini chat returned transient error response: Sorry, something went wrong. Please try your request again.")
    assert reason == RetryReason.PAGE_CRASH


def test_transport_classifier_treats_attachment_and_baseline_failures_as_restartable():
    assert (
        classify_transport_failure(
            "GEMINI_ATTACHMENT_REQUIRED: video upload was not confirmed for current request"
        )
        == RetryReason.PAGE_CRASH
    )
    assert (
        classify_transport_failure(
            "GEMINI_NO_NEW_ASSISTANT_MESSAGE: assistant response did not advance baseline"
        )
        == RetryReason.NO_NEW_ASSISTANT_MESSAGE
    )


def test_chat_only_transport_retry_wrapper_retries_once(monkeypatch, tmp_path: Path):
    calls = {"count": 0}

    def fake_once(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("Chat subprocess returned invalid JSON.")
        return {"parsed": {"segments": []}, "raw_text": "{}", "attach_notes": [], "usage": {}}

    monkeypatch.setattr(chat_only, "_run_chat_subprocess_once", fake_once)
    monkeypatch.setattr(chat_only.time, "sleep", lambda *_args, **_kwargs: None)

    result = chat_only._run_chat_subprocess(
        cfg={"run": {"gemini_transport_max_retries": 2}},
        video_file=tmp_path / "video.mp4",
        prompt_text="prompt",
        cache_dir=tmp_path,
        episode_id="ep1",
        model="gemini-3.1-pro-preview",
        prompt_scope="chat_labels",
        mode="labels",
    )

    assert calls["count"] == 2
    assert result["parsed"] == {"segments": []}


def test_chat_only_transport_retry_wrapper_uses_ops_retry_budget(monkeypatch, tmp_path: Path):
    calls = {"count": 0}

    def fake_once(**kwargs):
        calls["count"] += 1
        raise RuntimeError("Chat ops subprocess timed out after 360s (configured timeout=300s).")

    monkeypatch.setattr(chat_only, "_run_chat_subprocess_once", fake_once)
    monkeypatch.setattr(chat_only.time, "sleep", lambda *_args, **_kwargs: None)

    try:
        chat_only._run_chat_subprocess(
            cfg={"run": {"gemini_transport_max_retries": 5, "gemini_transport_max_retries_ops": 2}},
            video_file=tmp_path / "video.mp4",
            prompt_text="prompt",
            cache_dir=tmp_path,
            episode_id="ep-ops",
            model="gemini-3.1-pro-preview",
            prompt_scope="chat_ops",
            mode="ops",
        )
    except RuntimeError:
        pass

    assert calls["count"] == 2


def test_clean_import_of_legacy_video_segments_modules():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.solver import legacy_impl, video, video_core, segments; print('ok')",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "ok" in (proc.stdout or "")


def test_emit_episode_report_writes_json_and_jsonl(tmp_path: Path):
    cfg = {"run": {"structured_episode_reports": True, "output_dir": str(tmp_path / "outputs")}}
    report = EpisodeReport(
        episode_id="ep42",
        context_id="ctx42",
        segment_checksum="abc123",
        segment_count=3,
        page_url="https://audit.atlascapture.io/tasks/room/normal/label/ep42",
    )

    report_path = legacy_impl._emit_episode_report(
        cfg,
        report,
        task_state={"last_error": "", "episode_submitted": True},
        lifecycle_events=[{"event": "context_created", "context_id": "ctx42"}],
    )

    assert report_path is not None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    jsonl_path = tmp_path / "outputs" / "episode_reports" / "episodes.jsonl"

    assert payload["episode_id"] == "ep42"
    assert payload["context_id"] == "ctx42"
    assert payload["segment_checksum"] == "abc123"
    assert payload["task_state_excerpt"]["episode_submitted"] is True
    assert jsonl_path.exists()


def test_activate_episode_runtime_v2_creates_isolated_context(monkeypatch, tmp_path: Path):
    class FakePage:
        def __init__(self, url: str = ""):
            self.url = url

        def goto(self, url: str, **_kwargs):
            self.url = url

        def wait_for_timeout(self, _ms: int):
            return None

    class FakeContext:
        def __init__(self):
            self.pages = []
            self.storage_state_path = ""

        def new_page(self):
            page = FakePage()
            self.pages.append(page)
            return page

        def storage_state(self, path: str):
            self.storage_state_path = path

        def close(self):
            return None

    class FakeBrowser:
        def __init__(self):
            self.created_contexts = []

        def new_context(self, **_kwargs):
            context = FakeContext()
            self.created_contexts.append(context)
            return context

    state_path = tmp_path / "atlas_auth.json"
    state_path.write_text("{}", encoding="utf-8")
    bootstrap_context = FakeContext()
    bootstrap_page = FakePage("https://audit.atlascapture.io/tasks/room/normal/label/task123")
    browser = FakeBrowser()

    monkeypatch.setattr(legacy_impl, "_is_authenticated_page", lambda _page: True)

    runtime, page, context = legacy_impl._activate_episode_runtime_v2(
        browser=browser,
        bootstrap_context=bootstrap_context,
        bootstrap_page=bootstrap_page,
        state_path=state_path,
        cfg={
            "run": {
                "use_episode_runtime_v2": True,
                "force_episode_browser_isolation": True,
            }
        },
        task_id="task123",
    )

    assert runtime is not None
    assert page is runtime.atlas_page
    assert context is runtime.atlas_context
    assert runtime.context_id
    assert page.url.endswith("task123")


def test_activate_episode_runtime_v2_returns_bootstrap_page_to_queue(monkeypatch, tmp_path: Path):
    class FakePage:
        def __init__(self, url: str = ""):
            self.url = url

        def goto(self, url: str, **_kwargs):
            self.url = url

        def wait_for_timeout(self, _ms: int):
            return None

    class FakeContext:
        def __init__(self):
            self.pages = []
            self.storage_state_path = ""

        def new_page(self):
            page = FakePage()
            self.pages.append(page)
            return page

        def storage_state(self, path: str):
            self.storage_state_path = path

        def close(self):
            return None

    class FakeBrowser:
        def __init__(self):
            self.created_contexts = []

        def new_context(self, **_kwargs):
            context = FakeContext()
            self.created_contexts.append(context)
            return context

    state_path = tmp_path / "atlas_auth.json"
    state_path.write_text("{}", encoding="utf-8")
    bootstrap_context = FakeContext()
    bootstrap_page = FakePage("https://audit.atlascapture.io/tasks/room/normal/label/task321")
    browser = FakeBrowser()

    monkeypatch.setattr(legacy_impl, "_is_authenticated_page", lambda _page: True)

    runtime, page, context = legacy_impl._activate_episode_runtime_v2(
        browser=browser,
        bootstrap_context=bootstrap_context,
        bootstrap_page=bootstrap_page,
        state_path=state_path,
        cfg={
            "run": {
                "use_episode_runtime_v2": True,
                "force_episode_browser_isolation": True,
            },
            "atlas": {"room_url": "https://audit.atlascapture.io/tasks/room/normal"},
        },
        task_id="task321",
    )

    assert runtime is not None
    assert page is runtime.atlas_page
    assert context is runtime.atlas_context
    assert page.url.endswith("task321")
    assert bootstrap_page.url == "https://audit.atlascapture.io/tasks/room/normal"


def test_activate_episode_runtime_v2_registers_gemini_session(monkeypatch, tmp_path: Path):
    class FakePage:
        def __init__(self, url: str = ""):
            self.url = url

        def goto(self, url: str, **_kwargs):
            self.url = url

        def wait_for_timeout(self, _ms: int):
            return None

        def close(self):
            return None

    class FakeContext:
        def __init__(self):
            self.pages = []
            self.saved_paths = []

        def new_page(self):
            page = FakePage()
            self.pages.append(page)
            return page

        def storage_state(self, path: str):
            self.saved_paths.append(path)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("{}", encoding="utf-8")

        def close(self):
            return None

    class FakeBrowser:
        def __init__(self):
            self.created_contexts = []

        def new_context(self, **_kwargs):
            context = FakeContext()
            self.created_contexts.append(context)
            return context

    browser = FakeBrowser()
    bootstrap_context = FakeContext()
    bootstrap_page = FakePage("https://audit.atlascapture.io/tasks/room/normal/label/taskABC")
    state_path = tmp_path / "atlas_auth.json"
    state_path.write_text("{}", encoding="utf-8")
    gemini_state = tmp_path / "gemini_chat_state.json"
    captured = {}

    monkeypatch.setattr(legacy_impl, "_is_authenticated_page", lambda _page: True)

    def fake_register_episode_gemini_session(*, episode_id, runtime, cfg):
        captured["episode_id"] = episode_id
        captured["runtime"] = runtime
        return SimpleNamespace(session_id="sess-v2")

    monkeypatch.setattr(chat_only, "register_episode_gemini_session", fake_register_episode_gemini_session)

    runtime, page, context = legacy_impl._activate_episode_runtime_v2(
        browser=browser,
        bootstrap_context=bootstrap_context,
        bootstrap_page=bootstrap_page,
        state_path=state_path,
        cfg={
            "run": {
                "use_episode_runtime_v2": True,
                "force_episode_browser_isolation": True,
                "strict_single_chat_session": True,
            },
            "gemini": {
                "chat_web_storage_state": str(gemini_state),
                "chat_web_url": "https://gemini.google.com/app/session",
            },
        },
        task_id="taskABC",
    )

    assert runtime is not None
    assert page is runtime.atlas_page
    assert context is runtime.atlas_context
    assert runtime.gemini_context is not None
    assert runtime.gemini_page is not None
    assert runtime.gemini_page.url == "https://gemini.google.com/app/session"
    assert len(browser.created_contexts) == 2
    assert captured["episode_id"] == "taskABC"
    assert captured["runtime"] is runtime
    assert runtime.task_state["gemini_session_id"] == "sess-v2"


def test_activate_episode_runtime_v2_reuses_separate_gemini_cdp_context(monkeypatch, tmp_path: Path):
    class FakePage:
        def __init__(self, url: str = ""):
            self.url = url
            self.goto_calls = []

        def goto(self, url: str, **_kwargs):
            self.url = url
            self.goto_calls.append(url)

        def wait_for_timeout(self, _ms: int):
            return None

        def close(self):
            return None

    class FakeContext:
        def __init__(self, pages=None):
            self.pages = list(pages or [])
            self.saved_paths = []

        def new_page(self):
            page = FakePage()
            self.pages.append(page)
            return page

        def storage_state(self, path: str):
            self.saved_paths.append(path)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text('{"origins":[],"cookies":[]}', encoding="utf-8")

        def close(self):
            return None

    class FakeBrowser:
        def __init__(self, contexts=None):
            self.contexts = list(contexts or [])
            self.created_contexts = []

        def new_context(self, **_kwargs):
            context = FakeContext()
            self.created_contexts.append(context)
            return context

    browser = FakeBrowser()
    bootstrap_context = FakeContext()
    bootstrap_page = FakePage("https://audit.atlascapture.io/tasks/room/normal/label/taskCDP")
    state_path = tmp_path / "atlas_auth.json"
    state_path.write_text("{}", encoding="utf-8")
    gemini_state = tmp_path / "gemini_chat_state.json"
    gemini_page = FakePage("https://gemini.google.com/app/b3006ba9f325b55c")
    gemini_context = FakeContext(pages=[gemini_page])
    gemini_browser = FakeBrowser(contexts=[gemini_context])

    monkeypatch.setattr(legacy_impl, "_is_authenticated_page", lambda _page: True)
    monkeypatch.setattr(
        chat_only,
        "register_episode_gemini_session",
        lambda *, episode_id, runtime, cfg: SimpleNamespace(session_id="sess-cdp"),
    )

    runtime, page, context = legacy_impl._activate_episode_runtime_v2(
        browser=browser,
        gemini_browser=gemini_browser,
        bootstrap_context=bootstrap_context,
        bootstrap_page=bootstrap_page,
        state_path=state_path,
        cfg={
            "run": {
                "use_episode_runtime_v2": True,
                "force_episode_browser_isolation": True,
                "strict_single_chat_session": True,
            },
            "gemini": {
                "chat_web_storage_state": str(gemini_state),
                "chat_web_url": "https://gemini.google.com/app/b3006ba9f325b55c",
                "chat_web_reuse_cdp_context": True,
            },
        },
        task_id="taskCDP",
    )

    assert runtime is not None
    assert page is runtime.atlas_page
    assert context is runtime.atlas_context
    assert runtime.gemini_context is gemini_context
    assert runtime.gemini_page is gemini_page
    assert runtime.gemini_context_borrowed is True
    assert runtime.gemini_page_borrowed is True
    assert gemini_context.saved_paths == [str(gemini_state)]
    assert bootstrap_context.saved_paths == [str(state_path)]


def test_activate_episode_runtime_v2_does_not_hijack_unrelated_gemini_tab(monkeypatch, tmp_path: Path):
    class FakePage:
        def __init__(self, url: str = ""):
            self.url = url
            self.goto_calls = []

        def goto(self, url: str, **_kwargs):
            self.url = url
            self.goto_calls.append(url)

        def wait_for_timeout(self, _ms: int):
            return None

        def close(self):
            return None

    class FakeContext:
        def __init__(self, pages=None):
            self.pages = list(pages or [])
            self.saved_paths = []

        def new_page(self):
            page = FakePage()
            self.pages.append(page)
            return page

        def storage_state(self, path: str):
            self.saved_paths.append(path)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text('{"origins":[],"cookies":[]}', encoding="utf-8")

        def close(self):
            return None

    class FakeBrowser:
        def __init__(self, contexts=None):
            self.contexts = list(contexts or [])
            self.created_contexts = []

        def new_context(self, **_kwargs):
            context = FakeContext()
            self.created_contexts.append(context)
            return context

    browser = FakeBrowser()
    bootstrap_context = FakeContext()
    bootstrap_page = FakePage("https://audit.atlascapture.io/tasks/room/normal/label/taskDEDI")
    state_path = tmp_path / "atlas_auth.json"
    state_path.write_text("{}", encoding="utf-8")
    gemini_state = tmp_path / "gemini_chat_state.json"
    unrelated_gemini_page = FakePage("https://gemini.google.com/app/random-thread")
    gemini_context = FakeContext(pages=[unrelated_gemini_page])
    gemini_browser = FakeBrowser(contexts=[gemini_context])

    monkeypatch.setattr(legacy_impl, "_is_authenticated_page", lambda _page: True)
    monkeypatch.setattr(
        chat_only,
        "register_episode_gemini_session",
        lambda *, episode_id, runtime, cfg: SimpleNamespace(session_id="sess-dedi"),
    )

    runtime, page, context = legacy_impl._activate_episode_runtime_v2(
        browser=browser,
        gemini_browser=gemini_browser,
        bootstrap_context=bootstrap_context,
        bootstrap_page=bootstrap_page,
        state_path=state_path,
        cfg={
            "run": {
                "use_episode_runtime_v2": True,
                "force_episode_browser_isolation": True,
                "strict_single_chat_session": True,
            },
            "gemini": {
                "chat_web_storage_state": str(gemini_state),
                "chat_web_url": "https://gemini.google.com/app/b3006ba9f325b55c",
                "chat_web_reuse_cdp_context": True,
            },
        },
        task_id="taskDEDI",
    )

    assert runtime is not None
    assert page is runtime.atlas_page
    assert context is runtime.atlas_context
    assert runtime.gemini_context is gemini_context
    assert runtime.gemini_page is not unrelated_gemini_page
    assert runtime.gemini_page.url == "https://gemini.google.com/app/b3006ba9f325b55c"
    assert unrelated_gemini_page.url == "https://gemini.google.com/app/random-thread"
    assert runtime.gemini_page_borrowed is False


def test_gemini_session_cleans_pinned_conversation_on_session_init_and_per_request():
    session = chat_only.GeminiSession(
        runtime=SimpleNamespace(
            episode_id="ep1",
            context_id="ctx1",
            gemini_page=None,
            gemini_context=None,
            gemini_browser=None,
        ),
        cfg={
            "run": {"strict_single_chat_session": True},
            "gemini": {},
        },
    )

    assert session._should_clean_thread_on_session_init("https://gemini.google.com/app/b3006ba9f325b55c") is True
    assert session._should_clean_thread_on_session_init("https://gemini.google.com/app") is True
    assert session._should_clean_thread_per_request("https://gemini.google.com/app/b3006ba9f325b55c") is True
    assert session._should_clean_thread_per_request("https://gemini.google.com/app") is True


def test_gemini_session_can_explicitly_preserve_existing_thread():
    session = chat_only.GeminiSession(
        runtime=SimpleNamespace(
            episode_id="ep-preserve",
            context_id="ctx-preserve",
            gemini_page=None,
            gemini_context=None,
            gemini_browser=None,
        ),
        cfg={
            "run": {"strict_single_chat_session": True},
            "gemini": {
                "chat_web_url": "https://gemini.google.com/app/b3006ba9f325b55c",
                "chat_web_preserve_existing_thread": True,
            },
        },
    )

    assert session._preserve_existing_thread(session._chat_url()) is True
    assert session._should_clean_thread_on_session_init(session._chat_url()) is False


def test_gemini_session_initial_ensure_page_keeps_pinned_chat_when_preserving_thread():
    class FakePage:
        def __init__(self, url: str = "about:blank"):
            self.url = url
            self.goto_calls = []
            self.waits = []

        def goto(self, url: str, wait_until: str = "", timeout: int = 0):
            self.goto_calls.append((url, wait_until, timeout))
            self.url = url

        def wait_for_timeout(self, timeout_ms: int):
            self.waits.append(timeout_ms)

    page = FakePage()
    runtime = SimpleNamespace(
        episode_id="ep-preserve-init",
        context_id="ctx-preserve-init",
        gemini_page=page,
        gemini_context=None,
        gemini_browser=None,
        task_state={},
    )
    session = chat_only.GeminiSession(
        runtime=runtime,
        cfg={
            "run": {"strict_single_chat_session": True},
            "gemini": {
                "chat_web_url": "https://gemini.google.com/app/b3006ba9f325b55c",
                "chat_web_preserve_existing_thread": True,
            },
        },
    )

    ensured = session._ensure_page()

    assert ensured is page
    assert page.goto_calls == [("https://gemini.google.com/app/b3006ba9f325b55c", "domcontentloaded", 60000)]
    assert page.waits == [2500]


def test_acquire_gemini_probe_page_prefers_authenticated_existing_page(monkeypatch):
    class FakePage:
        def __init__(self, url: str, authenticated: bool):
            self.url = url
            self.authenticated = authenticated

    class FakeContext:
        def __init__(self, pages):
            self.pages = pages
            self.new_page_calls = 0

        def new_page(self):
            self.new_page_calls += 1
            page = FakePage("https://gemini.google.com/app", False)
            self.pages.append(page)
            return page

    monkeypatch.setattr(
        legacy_impl,
        "_is_authenticated_gemini_page",
        lambda page: bool(getattr(page, "authenticated", False)),
    )

    unauthenticated = FakePage("https://gemini.google.com/app", False)
    authenticated = FakePage("https://gemini.google.com/app", True)
    context = FakeContext([unauthenticated, authenticated])

    page, created = legacy_impl._acquire_gemini_probe_page(
        context,
        gemini_chat_url="https://gemini.google.com/app",
    )

    assert page is authenticated
    assert created is False
    assert context.new_page_calls == 0


def test_gemini_session_rejects_stale_full_label_candidate_without_requested_scope():
    session = chat_only.GeminiSession(
        runtime=SimpleNamespace(
            episode_id="ep-stale",
            context_id="ctx-stale",
            gemini_page=None,
            gemini_context=None,
            gemini_browser=None,
            task_state={},
        ),
        cfg={"run": {}, "gemini": {}},
    )
    snapshot = build_segment_snapshot(
        segments=[
            {"segment_index": idx, "start_sec": float(idx - 1), "end_sec": float(idx), "raw_text": f"segment {idx}"}
            for idx in range(1, 15)
        ],
        episode_id="ep-stale",
        context_id="ctx-stale",
        source_kind="extracted_source",
    )
    stale_raw = json.dumps(
        {
            "segments": [
                {
                    "segment_index": idx,
                    "start_sec": float(idx - 1),
                    "end_sec": float(idx),
                    "label": f"stale label {idx}",
                }
                for idx in range(1, 18)
            ]
        }
    )

    matched = session._response_candidate_matches_request(
        stale_raw,
        snapshot=snapshot,
        expected_schema="segments_only",
        requested_indices=None,
        allow_merge=False,
    )

    assert matched is False


def test_gemini_session_retries_hallucinated_indices_with_scope_followup(monkeypatch):
    runtime = SimpleNamespace(
        episode_id="ep-scope",
        context_id="ctx-scope",
        gemini_page=None,
        gemini_context=None,
        gemini_browser=None,
        task_state={},
    )
    session = chat_only.GeminiSession(
        runtime=runtime,
        cfg={
            "run": {
                "gemini_scope_followup_attempts": 1,
            },
            "gemini": {},
        },
    )
    snapshot = build_segment_snapshot(
        segments=[
            {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "raw_text": "loosen screw"},
            {"segment_index": 2, "start_sec": 4.0, "end_sec": 8.0, "raw_text": "remove cover"},
        ],
        episode_id="ep-scope",
        context_id="ctx-scope",
        source_kind="extracted_source",
    )

    calls = []

    def fake_request_payload(
        *,
        snapshot,
        prompt,
        video_file,
        retry_stage,
        expected_schema="",
        requested_indices=None,
        allow_merge=False,
        request_mode="",
        heartbeat=None,
        preserve_current_thread=False,
    ):
        calls.append(
            {
                "prompt": prompt,
                "video_file": video_file,
                "retry_stage": retry_stage,
                "expected_schema": expected_schema,
                "requested_indices": list(requested_indices or []),
                "preserve_current_thread": preserve_current_thread,
            }
        )
        if len(calls) == 1:
            return SimpleNamespace(
                request_id="req-scope-1",
                episode_id="ep-scope",
                context_id="ctx-scope",
                retry_stage=retry_stage,
                latency_ms=100,
                raw_text='{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":4.0,"label":"loosen screw"},{"segment_index":9,"start_sec":4.0,"end_sec":8.0,"label":"remove cover"}]}',
                parsed_payload={
                    "segments": [
                        {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "label": "loosen screw"},
                        {"segment_index": 9, "start_sec": 4.0, "end_sec": 8.0, "label": "remove cover"},
                    ]
                },
                validated_segments=[],
                attach_notes=["video.mp4: attached"],
                validation_errors=[],
                session_restarted=False,
                raw_response_path="",
                raw_response_meta_path="",
                expected_schema=expected_schema,
                requested_indices=list(requested_indices or []),
                started_at_utc="2026-04-16T00:00:00Z",
            )
        return SimpleNamespace(
            request_id="req-scope-2",
            episode_id="ep-scope",
            context_id="ctx-scope",
            retry_stage=retry_stage,
            latency_ms=120,
            raw_text='{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":4.0,"label":"loosen screw"},{"segment_index":2,"start_sec":4.0,"end_sec":8.0,"label":"remove cover"}]}',
            parsed_payload={
                "segments": [
                    {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "label": "loosen screw"},
                    {"segment_index": 2, "start_sec": 4.0, "end_sec": 8.0, "label": "remove cover"},
                ]
            },
            validated_segments=[],
            attach_notes=[],
            validation_errors=[],
            session_restarted=False,
            raw_response_path="",
            raw_response_meta_path="",
            expected_schema=expected_schema,
            requested_indices=list(requested_indices or []),
            started_at_utc="2026-04-16T00:00:05Z",
        )

    monkeypatch.setattr(session, "_request_payload", fake_request_payload)

    result = session.generate_labels(snapshot, "label prompt", Path("video.mp4"))

    assert len(calls) == 2
    assert calls[0]["video_file"] == Path("video.mp4")
    assert calls[1]["video_file"] is None
    assert calls[1]["preserve_current_thread"] is True
    assert calls[1]["requested_indices"] == [1, 2]
    assert "Allowed segment_index values: [1, 2]." in calls[1]["prompt"]
    assert "unknown segment 9" in calls[1]["prompt"]
    assert result.validation_errors == []
    assert [item["segment_index"] for item in result.validated_segments] == [1, 2]
    assert "scope_followup_retry_used:1" in result.attach_notes
    assert "scope_followup_retry_recovered" in result.attach_notes
    assert "video.mp4: attached" in result.attach_notes
    assert runtime.task_state["gemini_last_request_id"] == "req-scope-2"


def test_gemini_session_retries_missing_segment_indices_with_scope_followup(monkeypatch):
    runtime = SimpleNamespace(
        episode_id="ep-missing",
        context_id="ctx-missing",
        gemini_page=None,
        gemini_context=None,
        gemini_browser=None,
        task_state={},
    )
    session = chat_only.GeminiSession(
        runtime=runtime,
        cfg={
            "run": {
                "gemini_scope_followup_attempts": 1,
            },
            "gemini": {},
        },
    )
    snapshot = build_segment_snapshot(
        segments=[
            {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "raw_text": "coat dough in powder"},
            {"segment_index": 2, "start_sec": 4.0, "end_sec": 8.0, "raw_text": "place dough on tray"},
        ],
        episode_id="ep-missing",
        context_id="ctx-missing",
        source_kind="extracted_source",
    )

    calls = []

    def fake_request_payload(
        *,
        snapshot,
        prompt,
        video_file,
        retry_stage,
        expected_schema="",
        requested_indices=None,
        allow_merge=False,
        request_mode="",
        heartbeat=None,
        preserve_current_thread=False,
    ):
        calls.append(
            {
                "prompt": prompt,
                "video_file": video_file,
                "retry_stage": retry_stage,
                "expected_schema": expected_schema,
                "requested_indices": list(requested_indices or []),
                "preserve_current_thread": preserve_current_thread,
            }
        )
        if len(calls) == 1:
            return SimpleNamespace(
                request_id="req-missing-1",
                episode_id="ep-missing",
                context_id="ctx-missing",
                retry_stage=retry_stage,
                latency_ms=100,
                raw_text='{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":4.0,"label":"coat dough in powder"}]}',
                parsed_payload={
                    "segments": [
                        {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "label": "coat dough in powder"},
                    ]
                },
                validated_segments=[],
                attach_notes=["video.mp4: attached"],
                validation_errors=[],
                session_restarted=False,
                raw_response_path="",
                raw_response_meta_path="",
                expected_schema=expected_schema,
                requested_indices=list(requested_indices or []),
                started_at_utc="2026-04-16T00:00:00Z",
            )
        return SimpleNamespace(
            request_id="req-missing-2",
            episode_id="ep-missing",
            context_id="ctx-missing",
            retry_stage=retry_stage,
            latency_ms=120,
            raw_text='{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":4.0,"label":"coat dough in powder"},{"segment_index":2,"start_sec":4.0,"end_sec":8.0,"label":"place dough on tray"}]}',
            parsed_payload={
                "segments": [
                    {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "label": "coat dough in powder"},
                    {"segment_index": 2, "start_sec": 4.0, "end_sec": 8.0, "label": "place dough on tray"},
                ]
            },
            validated_segments=[],
            attach_notes=[],
            validation_errors=[],
            session_restarted=False,
            raw_response_path="",
            raw_response_meta_path="",
            expected_schema=expected_schema,
            requested_indices=list(requested_indices or []),
            started_at_utc="2026-04-16T00:00:05Z",
        )

    monkeypatch.setattr(session, "_request_payload", fake_request_payload)

    result = session.generate_labels(snapshot, "label prompt", Path("video.mp4"))

    assert len(calls) == 2
    assert calls[0]["video_file"] == Path("video.mp4")
    assert calls[1]["video_file"] is None
    assert calls[1]["preserve_current_thread"] is True
    assert calls[1]["requested_indices"] == [1, 2]
    assert "missing segment indices: [2]" in calls[1]["prompt"]
    assert result.validation_errors == []
    assert [item["segment_index"] for item in result.validated_segments] == [1, 2]
    assert "scope_followup_retry_used:1" in result.attach_notes
    assert "scope_followup_retry_recovered" in result.attach_notes
    assert "video.mp4: attached" in result.attach_notes
    assert runtime.task_state["gemini_last_request_id"] == "req-missing-2"


def test_gemini_session_retries_wrong_schema_with_followup(monkeypatch):
    runtime = SimpleNamespace(
        episode_id="ep-schema",
        context_id="ctx-schema",
        gemini_page=None,
        gemini_context=None,
        gemini_browser=None,
        task_state={},
    )
    session = chat_only.GeminiSession(
        runtime=runtime,
        cfg={
            "run": {
                "gemini_schema_followup_attempts": 1,
            },
            "gemini": {},
        },
    )
    snapshot = build_segment_snapshot(
        segments=[
            {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "raw_text": "pick up bread"},
            {"segment_index": 2, "start_sec": 4.0, "end_sec": 8.0, "raw_text": "slice bread"},
        ],
        episode_id="ep-schema",
        context_id="ctx-schema",
        source_kind="extracted_source",
    )

    calls = []

    def fake_request_payload(
        *,
        snapshot,
        prompt,
        video_file,
        retry_stage,
        expected_schema="",
        requested_indices=None,
        allow_merge=False,
        request_mode="",
        heartbeat=None,
        preserve_current_thread=False,
    ):
        calls.append(
            {
                "prompt": prompt,
                "video_file": video_file,
                "expected_schema": expected_schema,
                "requested_indices": list(requested_indices or []),
                "preserve_current_thread": preserve_current_thread,
            }
        )
        if len(calls) == 1:
            return SimpleNamespace(
                request_id="req-schema-1",
                episode_id="ep-schema",
                context_id="ctx-schema",
                retry_stage=retry_stage,
                latency_ms=100,
                raw_text='{"operations":[{"action":"split","segment_index":2}]}',
                parsed_payload={"operations": [{"action": "split", "segment_index": 2}]},
                validated_segments=[],
                attach_notes=["video.mp4: attached"],
                validation_errors=[
                    'response unexpectedly included top-level "operations" in segments-only mode',
                    'response missing top-level key "segments" in segments-only mode',
                ],
                session_restarted=False,
                raw_response_path="",
                raw_response_meta_path="",
                expected_schema=expected_schema,
                requested_indices=list(requested_indices or []),
                started_at_utc="2026-04-16T00:00:00Z",
            )
        return SimpleNamespace(
            request_id="req-schema-2",
            episode_id="ep-schema",
            context_id="ctx-schema",
            retry_stage=retry_stage,
            latency_ms=120,
            raw_text='{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":4.0,"label":"pick up bread"},{"segment_index":2,"start_sec":4.0,"end_sec":8.0,"label":"slice bread"}]}',
            parsed_payload={
                "segments": [
                    {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "label": "pick up bread"},
                    {"segment_index": 2, "start_sec": 4.0, "end_sec": 8.0, "label": "slice bread"},
                ]
            },
            validated_segments=[],
            attach_notes=[],
            validation_errors=[],
            session_restarted=False,
            raw_response_path="",
            raw_response_meta_path="",
            expected_schema=expected_schema,
            requested_indices=list(requested_indices or []),
            started_at_utc="2026-04-16T00:00:03Z",
        )

    monkeypatch.setattr(session, "_request_payload", fake_request_payload)

    result = session.generate_labels(snapshot, "label prompt", Path("video.mp4"))

    assert len(calls) == 2
    assert calls[0]["video_file"] == Path("video.mp4")
    assert calls[1]["video_file"] is None
    assert calls[1]["preserve_current_thread"] is True
    assert calls[1]["requested_indices"] == [1, 2]
    assert 'top-level key "segments"' in calls[1]["prompt"]
    assert 'Do not include top-level key "operations".' in calls[1]["prompt"]
    assert result.validation_errors == []
    assert [item["segment_index"] for item in result.validated_segments] == [1, 2]
    assert "schema_followup_retry_used:1" in result.attach_notes
    assert "schema_followup_retry_recovered" in result.attach_notes
    assert "video.mp4: attached" in result.attach_notes


def test_episode_runtime_reopen_gemini_falls_back_from_stale_borrowed_context():
    class BrokenBorrowedContext:
        def __init__(self):
            self.pages = []

        def new_page(self):
            raise RuntimeError("BrowserContext.new_page: Target page, context or browser has been closed")

    class FreshPage:
        def __init__(self):
            self.url = ""

        def goto(self, url, wait_until=None, timeout=None):
            self.url = url

    class FreshContext:
        def __init__(self):
            self.page = FreshPage()

        def new_page(self):
            return self.page

    class BrowserStub:
        def __init__(self, borrowed_context, fresh_context):
            self.contexts = [borrowed_context]
            self._fresh_context = fresh_context

        def new_context(self, **kwargs):
            return self._fresh_context

    borrowed_context = BrokenBorrowedContext()
    fresh_context = FreshContext()
    runtime = EpisodeRuntime(
        episode_id="ep-stale-borrowed",
        gemini_browser=BrowserStub(borrowed_context, fresh_context),
        gemini_context=borrowed_context,
        gemini_context_borrowed=True,
        gemini_page_url="https://gemini.google.com/app/b3006ba9f325b55c",
    )

    page = runtime.reopen_gemini()

    assert runtime.gemini_context is fresh_context
    assert runtime.gemini_context_borrowed is False
    assert runtime.gemini_page_borrowed is False
    assert page is fresh_context.page
    assert page.url == "https://gemini.google.com/app/b3006ba9f325b55c"


def test_gemini_session_restart_retries_after_reset_when_first_reopen_fails(monkeypatch):
    closed = {"page": 0, "context": 0, "reopen": 0}

    class Closable:
        def __init__(self, key):
            self.key = key

        def close(self):
            closed[self.key] += 1

    runtime = SimpleNamespace(
        episode_id="ep-restart",
        context_id="ctx-restart",
        gemini_page=Closable("page"),
        gemini_context=Closable("context"),
        gemini_browser=object(),
        gemini_page_borrowed=True,
        gemini_context_borrowed=True,
        reopen_gemini=None,
    )
    session = chat_only.GeminiSession(runtime=runtime, cfg={"gemini": {}, "run": {}})
    snapshot = build_segment_snapshot(
        segments=[{"segment_index": 1, "start_sec": 0.0, "end_sec": 2.0, "raw_text": "turn knob"}],
        episode_id="ep-restart",
        context_id="ctx-restart",
        source_kind="extracted_source",
    )

    def fake_reopen():
        closed["reopen"] += 1
        if closed["reopen"] == 1:
            raise RuntimeError("Target page, context or browser has been closed")
        runtime.gemini_page = None
        runtime.gemini_context = None
        return object()

    monkeypatch.setattr(runtime, "reopen_gemini", fake_reopen)
    monkeypatch.setattr(session, "_ensure_page", lambda: object())
    monkeypatch.setattr(_chat, "_handle_gemini_consent_if_present", lambda page: None)
    monkeypatch.setattr(_chat, "_first_visible_locator", lambda *args, **kwargs: None)

    session.restart_with_minimal_history(snapshot)

    assert closed["reopen"] == 2
    assert closed["page"] == 1
    assert closed["context"] == 1
    assert runtime.gemini_page_borrowed is False
    assert runtime.gemini_context_borrowed is False


def test_chat_only_uses_registered_session_for_labels(tmp_path: Path):
    cfg = {
        "run": {
            "use_episode_runtime_v2": True,
            "strict_single_chat_session": True,
        }
    }
    runtime = SimpleNamespace(
        episode_id="taskxyz",
        context_id="ctxxyz",
        atlas_page=SimpleNamespace(url="https://audit.atlascapture.io/tasks/room/normal/label/taskxyz"),
        gemini_page=SimpleNamespace(url="https://gemini.google.com/app/session"),
        task_state={},
    )
    def _generate_labels(snapshot, prompt, video_file, heartbeat=None):
        return SimpleNamespace(
            request_id="req-1",
            retry_stage="full_generate",
            latency_ms=321,
            raw_text='{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":4.0,"label":"pick up item"}]}',
            raw_response_path=str(tmp_path / "raw.txt"),
            attach_notes=["video.mp4: attached"],
            validation_errors=[],
            validated_segments=[
                {
                    "segment_index": 1,
                    "start_sec": 0.0,
                    "end_sec": 4.0,
                    "label": "pick up item",
                }
            ],
        )

    fake_session = SimpleNamespace(
        session_id="sess-xyz",
        runtime=runtime,
        generate_labels=_generate_labels,
    )
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"00")
    chat_only._REGISTERED_GEMINI_SESSIONS.clear()
    chat_only._REGISTERED_GEMINI_SESSIONS["taskxyz"] = fake_session

    result = chat_only.run_labels_generation(
        cfg=cfg,
        source_segments=[{"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "current_label": "move item"}],
        video_file=video_file,
        prompt_text="prompt",
        cache_dir=tmp_path,
        episode_id="taskxyz_chunk_01",
        model="gemini-3.1-pro-preview",
    )

    assert result["segments"][0]["label"] == "pick up item"
    assert result["gemini_session_id"] == "sess-xyz"
    assert result["request_id"] == "req-1"


def test_finalize_current_episode_v2_includes_gemini_metadata(tmp_path: Path):
    cfg = {"run": {"structured_episode_reports": True, "output_dir": str(tmp_path / "outputs")}}
    runtime = SimpleNamespace(
        episode_id="ep99",
        context_id="ctx99",
        task_state={
            "gemini_session_id": "sess99",
            "gemini_last_request_id": "req99",
            "gemini_last_retry_stage": "targeted_repair_1",
            "gemini_last_retry_reason": "hallucinated_indices",
            "gemini_last_latency_ms": 987,
            "gemini_last_validation_errors": ["response referenced unknown segment 12"],
        },
        lifecycle_events=[{"event": "context_created", "context_id": "ctx99"}],
    )
    report = EpisodeReport(episode_id="ep99", context_id="ctx99")

    page, context = legacy_impl._finalize_current_episode_v2(
        cfg=cfg,
        report=report,
        task_state={},
        runtime=runtime,
        bootstrap_page=None,
        bootstrap_context=None,
        room_url="",
        page=None,
        segments=[{"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0}],
        validation_report={"warnings": [], "errors": []},
        result={},
        reason="test-finalize",
    )

    payload = json.loads((tmp_path / "outputs" / "episode_reports" / "episode_ep99.json").read_text(encoding="utf-8"))

    assert page is None
    assert context is None
    assert payload["gemini_session_id"] == "sess99"
    assert payload["request_id"] == "req99"
    assert payload["retry_stage"] == "targeted_repair_1"
    assert payload["retry_reason"] == "hallucinated_indices"
    assert payload["gemini_latency_ms"] == 987


def test_chat_only_repair_query_uses_registered_session_targeted_repair(tmp_path: Path):
    runtime = SimpleNamespace(
        episode_id="ep-repair",
        context_id="ctx-repair",
        atlas_page=SimpleNamespace(url="https://audit.atlascapture.io/tasks/room/normal/label/ep-repair"),
        gemini_page=SimpleNamespace(url="https://gemini.google.com/app/session"),
        task_state={},
    )
    captured = {}

    class FakeSession:
        session_id = "sess-repair"

        def __init__(self):
            self.runtime = runtime

        def repair_failed_segments(self, snapshot, failing_indices, current_plan, reason, heartbeat=None):
            captured["indices"] = list(failing_indices)
            captured["current_plan"] = current_plan
            captured["reason"] = reason
            captured["heartbeat"] = heartbeat
            return SimpleNamespace(
                request_id="req-repair",
                retry_stage="targeted_repair_1",
                latency_ms=456,
                raw_text='{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":4.0,"label":"pick up item"}]}',
                raw_response_path=str(tmp_path / "raw_repair.txt"),
                parsed_payload={
                    "segments": [
                        {
                            "segment_index": 1,
                            "start_sec": 0.0,
                            "end_sec": 4.0,
                            "label": "pick up item",
                        }
                    ]
                },
                validation_errors=[],
                validated_segments=[
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 4.0,
                        "label": "pick up item",
                    }
                ],
            )

    chat_only._REGISTERED_GEMINI_SESSIONS.clear()
    chat_only._REGISTERED_GEMINI_SESSIONS["ep-repair"] = FakeSession()

    result = chat_only.run_repair_query(
        cfg={"run": {"use_episode_runtime_v2": True, "strict_single_chat_session": True}},
        source_segments=[{"segment_index": 1, "start_sec": 0.0, "end_sec": 12.0}],
        prompt_text="repair prompt",
        cache_dir=tmp_path,
        episode_id="ep-repair",
        model="gemini-3.1-pro-preview",
        failing_indices=[1],
        current_plan={1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 12.0, "label": "too long"}},
        retry_reason="policy_overlong",
    )

    assert captured["indices"] == [1]
    assert captured["reason"] == "policy_overlong"
    assert result["_meta"]["retry_stage"] == "targeted_repair_1"
    assert result["_meta"]["gemini_session_id"] == "sess-repair"


def test_chat_only_registered_session_forwards_heartbeat(tmp_path: Path):
    runtime = SimpleNamespace(
        episode_id="ep-heartbeat",
        context_id="ctx-heartbeat",
        atlas_page=SimpleNamespace(url="https://audit.atlascapture.io/tasks/room/normal/label/ep-heartbeat"),
        gemini_page=SimpleNamespace(url="https://gemini.google.com/app/session"),
        task_state={},
    )
    captured = {}

    class FakeSession:
        session_id = "sess-heartbeat"

        def __init__(self):
            self.runtime = runtime

        def generate_labels(self, snapshot, prompt, video_file, heartbeat=None):
            captured["heartbeat"] = heartbeat
            return SimpleNamespace(
                request_id="req-heartbeat",
                retry_stage="full_generate",
                latency_ms=123,
                raw_text='{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":4.0,"label":"pick up item"}]}',
                raw_response_path=str(tmp_path / "raw.txt"),
                attach_notes=[],
                validation_errors=[],
                validated_segments=[
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 4.0,
                        "label": "pick up item",
                    }
                ],
            )

    heartbeat = lambda: None
    chat_only._REGISTERED_GEMINI_SESSIONS.clear()
    chat_only._REGISTERED_GEMINI_SESSIONS["ep-heartbeat"] = FakeSession()
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"00")

    chat_only.run_labels_generation(
        cfg={"run": {"use_episode_runtime_v2": True, "strict_single_chat_session": True}},
        source_segments=[{"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0}],
        video_file=video_file,
        prompt_text="prompt",
        cache_dir=tmp_path,
        episode_id="ep-heartbeat",
        model="gemini-3.1-pro-preview",
        heartbeat=heartbeat,
    )

    assert captured["heartbeat"] is heartbeat


def test_wait_for_new_chat_response_text_emits_heartbeat(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

        def advance(self, sec):
            self.now += sec

    class FakeBody:
        def inner_text(self, timeout=None):
            return ""

    class FakePage:
        def __init__(self, clock):
            self.clock = clock

        def locator(self, selector):
            assert selector == "body"
            return FakeBody()

        def wait_for_timeout(self, ms):
            self.clock.advance(float(ms) / 1000.0)

    clock = FakeClock()
    page = FakePage(clock)
    responses = iter([[], [], ["ready"], ["ready"], ["ready"]])
    heartbeat_calls = []

    def fake_capture_state(_page, limit=8):
        values = list(next(responses, ["ready"]))
        return {
            "message_count": len(values),
            "response_hash": f"hash-{len(values)}-{'-'.join(values)}",
            "entries": [
                {"message_index": idx, "text": text}
                for idx, text in enumerate(values)
            ],
            "texts": values,
            "latest_text": values[-1] if values else "",
        }

    monkeypatch.setattr(_chat.time, "time", clock.time)
    monkeypatch.setattr(_chat.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(_chat, "_capture_chat_response_state", fake_capture_state)

    out = _chat._wait_for_new_chat_response_text(
        page,
        baseline_text="",
        timeout_sec=20.0,
        heartbeat=lambda: heartbeat_calls.append(clock.monotonic()),
        heartbeat_interval_sec=2.0,
    )

    assert out == "ready"
    assert len(heartbeat_calls) >= 1


def test_wait_for_new_chat_response_text_returns_parseable_json_early(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

        def advance(self, sec):
            self.now += sec

    class FakeBody:
        def inner_text(self, timeout=None):
            return ""

    class FakePage:
        def __init__(self, clock):
            self.clock = clock
            self.wait_calls = 0

        def locator(self, selector):
            assert selector == "body"
            return FakeBody()

        def wait_for_timeout(self, ms):
            self.wait_calls += 1
            self.clock.advance(float(ms) / 1000.0)

    clock = FakeClock()
    page = FakePage(clock)
    responses = iter([['{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":4.0,"label":"pick up item"}]}']])

    def fake_capture_state(_page, limit=8):
        values = list(next(responses, []))
        return {
            "message_count": len(values),
            "response_hash": f"hash-{len(values)}-{'-'.join(values)}",
            "entries": [
                {"message_index": idx, "text": text}
                for idx, text in enumerate(values)
            ],
            "texts": values,
            "latest_text": values[-1] if values else "",
        }

    monkeypatch.setattr(_chat.time, "time", clock.time)
    monkeypatch.setattr(_chat.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(_chat, "_capture_chat_response_state", fake_capture_state)

    out = _chat._wait_for_new_chat_response_text(page, baseline_text="", timeout_sec=20.0)

    assert out.startswith('{"segments"')
    assert page.wait_calls == 0


def test_wait_for_new_chat_response_text_waits_past_retryable_error_until_parseable_json(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

        def advance(self, sec):
            self.now += sec

    class FakeBody:
        def inner_text(self, timeout=None):
            return ""

    class FakePage:
        def __init__(self, clock):
            self.clock = clock
            self.wait_calls = 0

        def locator(self, selector):
            assert selector == "body"
            return FakeBody()

        def wait_for_timeout(self, ms):
            self.wait_calls += 1
            self.clock.advance(float(ms) / 1000.0)

    clock = FakeClock()
    page = FakePage(clock)
    responses = iter(
        [
            ["I encountered an error doing what you asked. Could you try again?"],
            ["I encountered an error doing what you asked. Could you try again?"],
            ['{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":4.0,"label":"pick up item"}]}'],
        ]
    )

    def fake_capture_state(_page, limit=8):
        values = list(next(responses, []))
        return {
            "message_count": len(values),
            "response_hash": f"hash-{len(values)}-{'-'.join(values)}",
            "entries": [
                {"message_index": idx, "text": text}
                for idx, text in enumerate(values)
            ],
            "texts": values,
            "latest_text": values[-1] if values else "",
        }

    monkeypatch.setattr(_chat.time, "time", clock.time)
    monkeypatch.setattr(_chat.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(_chat, "_capture_chat_response_state", fake_capture_state)

    out = _chat._wait_for_new_chat_response_text(
        page,
        baseline_text="",
        timeout_sec=20.0,
        require_parseable_json=True,
    )

    assert out.startswith('{"segments"')


def test_wait_for_chat_upload_settle_emits_heartbeat(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

        def advance(self, sec):
            self.now += sec

    class FakeBody:
        def inner_text(self, timeout=None):
            return ""

    class FakePage:
        def __init__(self, clock):
            self.clock = clock

        def locator(self, selector):
            assert selector == "body"
            return FakeBody()

        def wait_for_timeout(self, ms):
            self.clock.advance(float(ms) / 1000.0)

    clock = FakeClock()
    page = FakePage(clock)
    heartbeat_calls = []
    token_states = [
        ([], []),
        ([], []),
        (["video_ep.mp4"], []),
    ]
    call_state = {"count": 0}

    def fake_collect(_page, *, composer_locator, local_only):
        idx = min(call_state["count"] // 2, len(token_states) - 1)
        current = token_states[idx]
        call_state["count"] += 1
        return current[0] if local_only else current[1]

    monkeypatch.setattr(_chat.time, "time", clock.time)
    monkeypatch.setattr(_chat.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(_chat, "_collect_attachment_tokens", fake_collect)

    result = _chat._wait_for_chat_upload_settle(
        page,
        composer_locator=None,
        baseline_tokens=[],
        baseline_page_tokens=[],
        expected_fragments=["video_ep.mp4"],
        require_google_drive_video=False,
        size_mb=10.0,
        min_wait_sec=4.0,
        sec_per_100mb=12.0,
        max_wait_sec=45.0,
        heartbeat=lambda: heartbeat_calls.append(clock.monotonic()),
        heartbeat_interval_sec=1.0,
    )

    assert result["confirmed"] is True
    assert len(heartbeat_calls) >= 1
