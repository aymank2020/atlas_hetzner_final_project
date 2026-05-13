import time
from pathlib import Path

from src.solver import gemini, gemini_session, legacy_impl


def test_emit_solver_heartbeat_calls_active_callback(monkeypatch):
    calls = []
    monkeypatch.setattr(legacy_impl, "_ACTIVE_HEARTBEAT_CALLBACK", lambda: calls.append("hb"), raising=False)

    gemini._emit_solver_heartbeat()

    assert calls == ["hb"]


def test_chunked_request_emits_solver_heartbeat(monkeypatch, tmp_path):
    heartbeats = []
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"00")
    segments = [
        {"segment_index": 1, "start_sec": 0.0, "end_sec": 1.0, "current_label": "hold fabric"},
        {"segment_index": 2, "start_sec": 1.0, "end_sec": 2.0, "current_label": "move fabric"},
        {"segment_index": 3, "start_sec": 2.0, "end_sec": 3.0, "current_label": "place fabric"},
        {"segment_index": 4, "start_sec": 3.0, "end_sec": 4.0, "current_label": "release fabric"},
    ]

    monkeypatch.setattr(gemini, "_emit_solver_heartbeat", lambda: heartbeats.append("hb"))
    monkeypatch.setattr(legacy_impl, "_probe_video_duration_seconds", lambda _video: 120.0)
    monkeypatch.setattr(legacy_impl, "_resolve_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        legacy_impl,
        "_segment_chunks",
        lambda rows, max_segments_per_chunk, max_window_sec=0.0: [rows[:2], rows[2:]],
    )
    monkeypatch.setattr(legacy_impl, "_extract_video_window", lambda **kwargs: False)
    monkeypatch.setattr(
        legacy_impl,
        "_safe_float",
        lambda value, default=0.0: float(default if value is None else value),
    )
    monkeypatch.setattr(legacy_impl, "build_prompt", lambda *args, **kwargs: "prompt")
    monkeypatch.setattr(legacy_impl, "_collect_chunk_structural_operations", lambda **kwargs: [])
    monkeypatch.setattr(
        legacy_impl,
        "_normalize_segment_plan",
        lambda _payload, chunk_segments, cfg=None: {
            int(seg["segment_index"]): {
                "segment_index": int(seg["segment_index"]),
                "label": f"label {int(seg['segment_index'])}",
            }
            for seg in chunk_segments
        },
    )
    monkeypatch.setattr(legacy_impl, "_update_chunk_consistency_memory", lambda label, **kwargs: label)
    monkeypatch.setattr(legacy_impl, "_rewrite_label_tier3", lambda label: label)
    monkeypatch.setattr(legacy_impl, "_normalize_label_min_safety", lambda label: label)
    monkeypatch.setattr(
        legacy_impl,
        "call_gemini_labels",
        lambda *args, **kwargs: {"_meta": {"model": "gemini-3.1-flash-lite-preview", "api_key_source": "key_1"}},
    )

    result = gemini._request_labels_with_optional_segment_chunking(
        cfg={
            "run": {
                "segment_chunking_enabled": True,
                "segment_chunking_min_segments": 2,
                "segment_chunking_max_segments_per_request": 2,
                "segment_chunking_min_video_sec": 60.0,
                "output_dir": str(tmp_path),
                "tier3_label_rewrite": False,
            },
            "gemini": {},
        },
        segments=segments,
        prompt="prompt",
        video_file=video_path,
        allow_operations=False,
        task_id="ep_hb",
    )

    assert len(heartbeats) >= 2
    assert len(result["segments"]) == 4


def test_background_heartbeat_guard_emits_while_blocked():
    calls = []

    with gemini_session._HeartbeatGuard(lambda: calls.append(time.monotonic()), interval_sec=0.05):
        time.sleep(0.18)

    assert len(calls) >= 2


def test_chat_phase_watchdog_timeout_respects_active_request_budget():
    cfg = {
        "run": {
            "watchdog_stale_threshold_sec": 600.0,
            "watchdog_dynamic_timeout_cap_sec": 2400.0,
            "chat_labels_timeout_sec": 1200.0,
            "chat_ops_timeout_sec": 300.0,
            "chat_request_watchdog_buffer_sec": 180.0,
        },
        "gemini": {
            "chat_web_timeout_sec": 360.0,
        },
    }

    assert gemini._chat_phase_watchdog_timeout_hint_sec(
        cfg,
        phase="chunk_request",
        request_scope="labeling",
    ) == 1380.0
    assert gemini._chat_phase_watchdog_timeout_hint_sec(
        cfg,
        phase="planner",
        request_scope="planner",
    ) == 600.0


def test_chunked_chat_progress_persists_stage_updates(monkeypatch, tmp_path):
    stage_updates = []
    field_updates = []
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"00")
    segments = [
        {"segment_index": 1, "start_sec": 0.0, "end_sec": 1.0, "current_label": "hold fabric"},
        {"segment_index": 2, "start_sec": 1.0, "end_sec": 2.0, "current_label": "move fabric"},
        {"segment_index": 3, "start_sec": 2.0, "end_sec": 3.0, "current_label": "place fabric"},
        {"segment_index": 4, "start_sec": 3.0, "end_sec": 4.0, "current_label": "release fabric"},
    ]
    task_state = {}

    def fake_stage_status(cfg, task_id, task_state_obj=None, **updates):
        merged = dict(task_state_obj or {})
        merged.update(updates)
        stage_updates.append(dict(merged))
        if isinstance(task_state_obj, dict):
            task_state_obj.clear()
            task_state_obj.update(merged)
            return task_state_obj
        return merged

    def fake_persist(cfg, task_id, task_state_obj=None, **updates):
        merged = dict(task_state_obj or {})
        merged.update(updates)
        field_updates.append(dict(merged))
        if isinstance(task_state_obj, dict):
            task_state_obj.clear()
            task_state_obj.update(merged)
            return task_state_obj
        return merged

    def fake_run_labels_generation(**kwargs):
        heartbeat = kwargs.get("heartbeat")
        if callable(heartbeat):
            heartbeat()
        chunk_segments = list(kwargs.get("source_segments", []) or [])
        return {
            "segments": [
                {
                    "segment_index": int(seg["segment_index"]),
                    "start_sec": float(seg["start_sec"]),
                    "end_sec": float(seg["end_sec"]),
                    "label": f"label {int(seg['segment_index'])}",
                }
                for seg in chunk_segments
            ],
            "attach_notes": ["video.mp4: attached"],
            "usage": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            "out_json": str(tmp_path / f"chunk_{len(stage_updates)}.json"),
            "prompt_path": str(tmp_path / f"chunk_{len(stage_updates)}.txt"),
        }

    monkeypatch.setattr(gemini, "_emit_solver_heartbeat", lambda: None)
    monkeypatch.setattr(legacy_impl, "_persist_task_stage_status", fake_stage_status)
    monkeypatch.setattr(legacy_impl, "_persist_task_state_fields", fake_persist)
    monkeypatch.setattr(legacy_impl, "_stage_watchdog_timeout_hint_sec", lambda *args, **kwargs: 777.0)
    monkeypatch.setattr(legacy_impl, "_probe_video_duration_seconds", lambda _video: 120.0)
    monkeypatch.setattr(legacy_impl, "_resolve_ffmpeg_binary", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        legacy_impl,
        "_segment_chunks",
        lambda rows, max_segments_per_chunk, max_window_sec=0.0: [rows[:2], rows[2:]],
    )
    monkeypatch.setattr(legacy_impl, "_extract_video_window", lambda **kwargs: False)
    monkeypatch.setattr(
        legacy_impl,
        "_safe_float",
        lambda value, default=0.0: float(default if value is None else value),
    )
    monkeypatch.setattr(legacy_impl, "build_prompt", lambda *args, **kwargs: "prompt")
    monkeypatch.setattr(legacy_impl, "_update_chunk_consistency_memory", lambda label, **kwargs: label)
    monkeypatch.setattr(legacy_impl, "_rewrite_label_tier3", lambda label: label)
    monkeypatch.setattr(legacy_impl, "_normalize_label_min_safety", lambda label: label)
    monkeypatch.setattr("src.solver.chat_only.run_labels_generation", fake_run_labels_generation)

    result = gemini._request_labels_with_optional_segment_chunking(
        cfg={
            "run": {
                "chat_only_mode": True,
                "primary_solve_backend": "chat_web",
                "segment_chunking_enabled": True,
                "segment_chunking_min_segments": 2,
                "segment_chunking_max_segments_per_request": 2,
                "segment_chunking_min_video_sec": 60.0,
                "output_dir": str(tmp_path),
                "tier3_label_rewrite": False,
                "chat_stage_progress_persist_sec": 0.0,
            },
            "gemini": {},
        },
        segments=segments,
        prompt="prompt",
        video_file=video_path,
        allow_operations=False,
        task_id="ep_progress",
        task_state=task_state,
        stage_name="labeling",
    )

    assert len(result["segments"]) == 4
    assert any("waiting for Gemini chunk 1/2" in str(item.get("detail", "")) for item in stage_updates)
    assert any("Gemini chunk 2/2 completed" in str(item.get("detail", "")) for item in stage_updates)
    assert any(int(item.get("progress_current", 0) or 0) == 4 for item in stage_updates)
    assert any(int(item.get("chat_active_chunk_total", 0) or 0) == 2 for item in field_updates)
