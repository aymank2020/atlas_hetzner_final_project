from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.infra.gemini_economics import estimate_minimum_episode_cost_usd
from src.rules import labels as label_rules
from src.rules import policy_gate
from src.solver import chat_only, gemini, orchestrator
from atlas_triplet_compare import _build_timed_labels_prompt


def test_targeted_repair_scope_indices_use_two_neighbors_by_default() -> None:
    source_segments = [{"segment_index": idx} for idx in range(1, 8)]

    assert orchestrator._targeted_repair_scope_indices(source_segments, [4]) == [2, 3, 4, 5, 6]


def test_expand_contiguous_failure_targets_keeps_adjacent_block() -> None:
    source_segments = [{"segment_index": idx} for idx in range(1, 9)]

    assert orchestrator._expand_contiguous_failure_targets(
        source_segments,
        [3, 4, 5, 8],
        base_limit=1,
    ) == [3, 4, 5]


def test_normalize_ing_verbs_to_imperative_rewrites_ing_forms() -> None:
    assert label_rules._normalize_ing_verbs_to_imperative("moving cloth and placing it") == "move cloth and place it"


def test_hold_rule_reorders_hold_clause_first() -> None:
    label = label_rules._enforce_hold_rule(
        "move spoon into bowl, hold cup",
        segment_index=2,
        source_segments=[
            {"segment_index": 1, "current_label": "hold cup, stir batter", "start_sec": 0.0, "end_sec": 4.0},
            {"segment_index": 2, "current_label": "move spoon into bowl, hold cup", "start_sec": 4.0, "end_sec": 8.0},
        ],
        normalized_plan={},
        cfg={"run": {"max_atomic_actions_per_label": 2, "hold_rule_context_neighbors": 2}},
    )

    assert label == "hold cup, move spoon into bowl"


def test_hold_rule_inserts_contextual_hold_clause() -> None:
    label = label_rules._enforce_hold_rule(
        "move spoon into bowl",
        segment_index=2,
        source_segments=[
            {"segment_index": 1, "current_label": "hold cup, stir batter", "start_sec": 0.0, "end_sec": 4.0},
            {"segment_index": 2, "current_label": "move spoon into bowl", "start_sec": 4.0, "end_sec": 8.0},
            {"segment_index": 3, "current_label": "hold cup, scrape bowl", "start_sec": 8.0, "end_sec": 12.0},
        ],
        normalized_plan={},
        cfg={"run": {"max_atomic_actions_per_label": 2, "hold_rule_context_neighbors": 2}},
    )

    assert label == "hold cup, move spoon into bowl"


def test_normalize_segment_plan_applies_hold_rule_with_future_context() -> None:
    plan = label_rules._normalize_segment_plan(
        {
            "segments": [
                {"segment_index": 11, "start_sec": 50.0, "end_sec": 55.0, "label": "place blue car into white container"},
                {"segment_index": 12, "start_sec": 55.0, "end_sec": 60.0, "label": "lift blue car, place blue car inside model box"},
                {"segment_index": 13, "start_sec": 60.0, "end_sec": 62.9, "label": "hold model box"},
            ]
        },
        [
            {"segment_index": 11, "current_label": "place blue car into white container", "start_sec": 50.0, "end_sec": 55.0},
            {"segment_index": 12, "current_label": "lift blue car, place blue car inside model box", "start_sec": 55.0, "end_sec": 60.0},
            {"segment_index": 13, "current_label": "hold model box", "start_sec": 60.0, "end_sec": 62.9},
        ],
        cfg={"run": {"max_atomic_actions_per_label": 2, "hold_rule_context_neighbors": 2}},
    )

    assert plan[11]["label"] == "hold model box, place blue car into white container"


def test_policy_gate_rejects_hold_after_other_action() -> None:
    report = policy_gate._validate_segment_plan_against_policy(
        {
            "run": {
                "min_label_words": 2,
                "max_label_words": 20,
                "max_atomic_actions_per_label": 2,
                "max_segment_duration_sec": 10.0,
                "allowed_label_start_verbs": ["hold", "move", "place"],
                "forbidden_label_verbs": [],
                "forbidden_narrative_words": [],
                "hold_rule_context_neighbors": 2,
            }
        },
        [
            {"segment_index": 1, "current_label": "hold cup, move spoon into bowl", "start_sec": 0.0, "end_sec": 4.0},
            {"segment_index": 2, "current_label": "move spoon into bowl, hold cup", "start_sec": 4.0, "end_sec": 8.0},
        ],
        {
            1: {"segment_index": 1, "label": "hold cup, move spoon into bowl", "start_sec": 0.0, "end_sec": 4.0},
            2: {"segment_index": 2, "label": "move spoon into bowl, hold cup", "start_sec": 4.0, "end_sec": 8.0},
        },
    )

    assert any("hold' must appear before" in error for error in report["errors"])


def test_timed_labels_prompt_emphasizes_ten_second_split_and_hold_first() -> None:
    prompt = _build_timed_labels_prompt(strict_action_policy=True)

    assert "split it at or before 10.0 seconds" in prompt
    assert "Never keep one segment longer than 10.0 seconds" in prompt
    assert 'start the label with "hold X" first' in prompt
    assert 'hold nozzle, insert fuel nozzle into tank' in prompt


def test_targeted_repair_planner_prompt_requires_split_for_continuous_overlong_rows() -> None:
    prompt = chat_only.build_targeted_repair_planner_prompt(
        [{"segment_index": 1, "start_sec": 0.0, "end_sec": 14.0, "current_label": "hold nozzle, insert fuel nozzle into tank"}],
        failing_indices=[1],
        allow_merge=False,
        max_segment_duration_sec=10.0,
    )

    assert "Split any row longer than 10.0s even if the action continues in the next row." in prompt


def test_normalize_chat_segment_items_anchors_timestamps_to_dom_source() -> None:
    segments = chat_only._normalize_chat_segment_items(
        {
            "segments": [
                {
                    "segment_index": 1,
                    "start_sec": 4.2,
                    "end_sec": 9.9,
                    "label": "hold nozzle, insert fuel nozzle into tank",
                }
            ]
        },
        [{"segment_index": 1, "start_sec": 0.0, "end_sec": 1.2, "current_label": ""}],
        validation_errors=[],
    )

    assert segments == [
        {
            "segment_index": 1,
            "start_sec": 0.0,
            "end_sec": 1.2,
            "label": "hold nozzle, insert fuel nozzle into tank",
        }
    ]


def test_build_labels_prompt_strips_structural_operations_contract() -> None:
    base_prompt = (
        "You are an Atlas assistant.\n"
        "If boundaries are fundamentally wrong, you may request split/merge operations before final labels.\n"
        "Allowed operations: edit, split, merge. Do NOT use delete.\n"
        "Operation segment_index refers to the row index at execution time.\n"
        "Operations must be ordered exactly as they should be executed.\n"
        "Return strict JSON object only:\n"
        "Response must start with '{' and end with '}'.\n"
        "Do not wrap JSON in markdown code fences.\n"
        '{"operations":[{"action":"split","segment_index":3}],"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":1.2,"label":"..."}]}\n'
        'If no structural change is needed, return "operations":[]\n'
        "Segments input:\n"
        "- segment_index=1 start_sec=0.0 end_sec=4.0 current_label=\"wipe frame\"\n"
    )

    prompt = chat_only.build_labels_prompt(base_prompt)

    assert "Allowed operations:" not in prompt
    assert "Operation segment_index refers" not in prompt
    assert "If no structural change is needed" not in prompt
    assert '{"operations":' not in prompt
    assert 'Do not return top-level key "operations".' in prompt
    assert "Never return a segment_index that is not present in the provided segment list." in prompt
    assert 'Return strict JSON object only with key "segments".' in prompt


def test_run_labels_generation_v2_recreates_cache_dir_before_artifact_write(tmp_path: Path) -> None:
    cache_dir = tmp_path / "outputs" / "_chat_only" / "ep1" / "repair" / "labels"
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"00")

    class FakeSession:
        def __init__(self) -> None:
            self.session_id = "sess-one"
            self.runtime = SimpleNamespace(
                episode_id="ep1",
                context_id="ctx-one",
                atlas_page=SimpleNamespace(url="https://audit.atlascapture.io/tasks"),
                gemini_page=SimpleNamespace(url="https://gemini.google.com/app/test"),
            )

        def generate_labels(self, snapshot, prompt_text, video_file, heartbeat=None):
            import shutil

            shutil.rmtree(cache_dir, ignore_errors=True)
            return SimpleNamespace(
                request_id="req-one",
                retry_stage="full_generate",
                latency_ms=15,
                raw_response_path="",
                raw_response_meta_path="",
                validated_segments=[
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 4.0,
                        "label": "pick up item",
                    }
                ],
                validation_errors=[],
                raw_text='{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":4.0,"label":"pick up item"}]}',
                attach_notes=[],
            )

    result = chat_only._run_labels_generation_v2(
        session=FakeSession(),
        source_segments=[{"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0}],
        video_file=video_file,
        prompt_text="prompt",
        cache_dir=cache_dir,
        episode_id="ep1",
        model="gemini-3.1-pro-preview",
    )

    assert Path(result["out_txt"]).exists()
    assert Path(result["out_json"]).exists()


def test_run_repair_query_v2_writes_label_artifacts(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "repair" / "labels"
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"00")

    class FakeSession:
        def __init__(self) -> None:
            self.session_id = "sess-repair"
            self.runtime = SimpleNamespace(
                episode_id="ep1",
                context_id="ctx-repair",
                atlas_page=SimpleNamespace(url="https://audit.atlascapture.io/tasks"),
                gemini_page=SimpleNamespace(url="https://gemini.google.com/app/test"),
            )

        def repair_failed_segments(self, snapshot, scoped_indices, current_plan, reason, heartbeat=None):
            return SimpleNamespace(
                request_id="req-repair",
                retry_stage="targeted_repair_1",
                latency_ms=22,
                raw_response_path="",
                raw_response_meta_path="",
                validated_segments=[
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 4.0,
                        "label": "adjust shirt",
                    }
                ],
                validation_errors=[],
                parsed_payload={"segments": []},
                raw_text='{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":4.0,"label":"adjust shirt"}]}',
                attach_notes=[],
            )

    monkeypatch.setattr(chat_only, "_use_registered_gemini_session", lambda cfg, episode_id: FakeSession())

    result = chat_only.run_repair_query(
        cfg={"run": {"use_episode_runtime_v2": True, "strict_single_chat_session": True}},
        source_segments=[{"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0}],
        prompt_text="repair prompt",
        cache_dir=cache_dir,
        episode_id="ep1",
        model="gemini-3.1-pro-preview",
        video_file=video_file,
        failing_indices=[1],
        current_plan={1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "label": "move item"}},
        retry_reason="policy_overlong",
    )

    assert result["segments"][0]["label"] == "adjust shirt"
    assert Path(result["out_txt"]).exists()
    assert Path(result["out_json"]).exists()


def test_chat_only_request_labels_skips_api_runtime(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "run": {
            "output_dir": str(tmp_path / "outputs"),
            "chat_only_mode": True,
            "primary_solve_backend": "chat_web",
            "segment_chunking_enabled": True,
            "segment_chunking_min_segments": 16,
            "segment_chunking_max_segments_per_request": 8,
            "enable_structural_actions": True,
            "auto_continuity_merge_enabled": False,
            "structural_allow_split": True,
            "structural_allow_merge": False,
            "capture_step_screenshots": False,
        },
        "gemini": {
            "model": "gemini-3.1-pro-preview",
            "stage_models": {"labeling": "gemini-3.1-pro-preview"},
        },
        "_meta": {"config_dir": str(tmp_path)},
    }
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"00")
    task_state: dict[str, object] = {}
    segments = [
        {
            "segment_index": 1,
            "start_sec": 0.0,
            "end_sec": 4.0,
            "current_label": "move item",
            "raw_text": "move item",
        }
    ]
    labels_calls: list[str] = []

    def fake_api_call(*args, **kwargs):
        raise AssertionError("API solve should not be called in chat_only_mode")

    def fake_persist(cfg, task_id, task_state=None, **updates):
        merged = dict(task_state or {})
        merged.update(updates)
        if isinstance(task_state, dict):
            task_state.clear()
            task_state.update(merged)
            return task_state
        return merged

    def fake_run_labels_generation(**kwargs):
        labels_calls.append(str(kwargs.get("model", "")))
        out_json = tmp_path / "chat_labels.json"
        prompt_path = tmp_path / "chat_prompt.txt"
        out_json.write_text("{}", encoding="utf-8")
        prompt_path.write_text("prompt", encoding="utf-8")
        return {
            "segments": [
                {
                    "segment_index": 1,
                    "start_sec": 0.0,
                    "end_sec": 4.0,
                    "label": "pick up item",
                }
            ],
            "attach_notes": ["video.mp4: attached (3.1 MB, settle=4.0s, mode=file_chooser)"],
            "usage": {"promptTokenCount": 100, "candidatesTokenCount": 10, "totalTokenCount": 110},
            "out_json": str(out_json),
            "prompt_path": str(prompt_path),
        }

    monkeypatch.setattr("src.solver.legacy_impl.call_gemini_labels", fake_api_call)
    monkeypatch.setattr("src.solver.legacy_impl._persist_task_state_fields", fake_persist)
    monkeypatch.setattr("src.solver.legacy_impl._resolve_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr("src.solver.legacy_impl._segment_duration_exceeds_limit", lambda seg, max_sec: False)
    monkeypatch.setattr("src.solver.chat_only.run_labels_generation", fake_run_labels_generation)

    result = gemini._request_labels_with_optional_segment_chunking(
        cfg,
        segments,
        "prompt text",
        video_file,
        allow_operations=True,
        task_id="ep_chat_only",
        task_state=task_state,
        stage_name="labeling",
    )

    assert labels_calls == ["gemini-3.1-pro-preview"]
    assert result["_meta"]["mode"] == "chat_web"
    assert result["_meta"]["solve_backend"] == "chat_web"
    assert task_state["solve_backend"] == "chat_web"
    assert task_state["chat_only_mode"] is True
    assert task_state["chat_compare_skipped"] is True


def test_chat_only_chunk_retry_resumes_from_remaining_segments(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "run": {
            "output_dir": str(tmp_path / "outputs"),
            "chat_only_mode": True,
            "primary_solve_backend": "chat_web",
            "segment_chunking_enabled": True,
            "segment_chunking_min_segments": 4,
            "segment_chunking_max_segments_per_request": 5,
            "segment_chunking_min_video_sec": 0,
            "chat_chunk_fallback_to_single_request": True,
            "enable_structural_actions": False,
            "auto_continuity_merge_enabled": False,
            "capture_step_screenshots": False,
        },
        "gemini": {
            "model": "gemini-3.1-pro-preview",
            "stage_models": {"labeling": "gemini-3.1-pro-preview"},
        },
        "_meta": {"config_dir": str(tmp_path)},
    }
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"00")
    task_state: dict[str, object] = {}
    segments = [
        {
            "segment_index": idx,
            "start_sec": float(idx - 1) * 4.0,
            "end_sec": float(idx) * 4.0,
            "current_label": f"label {idx}",
            "raw_text": f"raw {idx}",
        }
        for idx in range(1, 22)
    ]
    chunk_calls: list[list[int]] = []
    restart_scopes: list[list[int]] = []

    def fake_persist(cfg, task_id, task_state=None, **updates):
        merged = dict(task_state or {})
        merged.update(updates)
        if isinstance(task_state, dict):
            task_state.clear()
            task_state.update(merged)
            return task_state
        return merged

    def fake_segment_chunks(source_segments, max_segments, max_window_sec=None):
        max_segments = int(max_segments)
        return [source_segments[i : i + max_segments] for i in range(0, len(source_segments), max_segments)]

    def fake_run_labels_generation(**kwargs):
        source_segments = list(kwargs["source_segments"])
        indices = [int(seg["segment_index"]) for seg in source_segments]
        chunk_calls.append(indices)
        if len(chunk_calls) == 2:
            raise RuntimeError("Chat labels response failed integrity checks: response missing segment indices: [8]")
        return {
            "segments": [
                {
                    "segment_index": int(seg["segment_index"]),
                    "start_sec": float(seg["start_sec"]),
                    "end_sec": float(seg["end_sec"]),
                    "label": f"resolved {int(seg['segment_index'])}",
                }
                for seg in source_segments
            ],
            "attach_notes": [],
            "usage": {},
            "out_json": str(tmp_path / f"chunk_{len(chunk_calls)}.json"),
            "prompt_path": str(tmp_path / f"chunk_{len(chunk_calls)}.txt"),
            "request_id": f"req-{len(chunk_calls)}",
            "gemini_session_id": "sess",
            "retry_stage": "full_generate",
            "retry_reason": "",
            "latency_ms": 1,
            "raw_response_path": "",
        }

    monkeypatch.setattr("src.solver.legacy_impl.call_gemini_labels", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("API solve should not be called")))
    monkeypatch.setattr("src.solver.legacy_impl._persist_task_state_fields", fake_persist)
    monkeypatch.setattr("src.solver.legacy_impl._resolve_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr("src.solver.legacy_impl._segment_chunks", fake_segment_chunks)
    monkeypatch.setattr("src.solver.legacy_impl._probe_video_duration_seconds", lambda _path: 0.0)
    monkeypatch.setattr("src.solver.legacy_impl._maybe_optimize_video_for_upload", lambda path, cfg: None)
    monkeypatch.setattr("src.solver.legacy_impl._extract_video_window", lambda **kwargs: False)
    monkeypatch.setattr("src.solver.chat_only.run_labels_generation", fake_run_labels_generation)
    monkeypatch.setattr(
        "src.solver.chat_only.restart_episode_gemini_session",
        lambda **kwargs: restart_scopes.append([int(seg["segment_index"]) for seg in kwargs["source_segments"]]) or True,
    )

    result = gemini._request_labels_with_optional_segment_chunking(
        cfg,
        segments,
        "prompt text",
        video_file,
        allow_operations=False,
        task_id="ep_chat_retry",
        task_state=task_state,
        stage_name="labeling",
    )

    assert chunk_calls[:2] == [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
    ]
    assert chunk_calls[2:] == [
        [6, 7, 8, 9],
        [10, 11, 12, 13],
        [14, 15, 16, 17],
        [18, 19, 20, 21],
    ]
    assert restart_scopes == [[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
    assert [item["segment_index"] for item in result["segments"]] == list(range(1, 22))
    assert result["segments"][0]["label"] != "label 1"
    assert result["segments"][-1]["label"] != "label 21"


def test_chat_only_chunk_retry_aborts_when_session_restart_fails(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "run": {
            "output_dir": str(tmp_path / "outputs"),
            "chat_only_mode": True,
            "primary_solve_backend": "chat_web",
            "segment_chunking_enabled": True,
            "segment_chunking_min_segments": 4,
            "segment_chunking_max_segments_per_request": 5,
            "segment_chunking_min_video_sec": 0,
            "chat_chunk_fallback_to_single_request": True,
            "enable_structural_actions": False,
            "auto_continuity_merge_enabled": False,
            "capture_step_screenshots": False,
        },
        "gemini": {
            "model": "gemini-3.1-pro-preview",
            "stage_models": {"labeling": "gemini-3.1-pro-preview"},
        },
        "_meta": {"config_dir": str(tmp_path)},
    }
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"00")
    task_state: dict[str, object] = {}
    segments = [
        {
            "segment_index": idx,
            "start_sec": float(idx - 1) * 4.0,
            "end_sec": float(idx) * 4.0,
            "current_label": f"label {idx}",
            "raw_text": f"raw {idx}",
        }
        for idx in range(1, 22)
    ]
    fallback_calls: list[str] = []

    def fake_segment_chunks(source_segments, max_segments, max_window_sec=None):
        max_segments = int(max_segments)
        return [source_segments[i : i + max_segments] for i in range(0, len(source_segments), max_segments)]

    def fake_run_labels_generation(**kwargs):
        source_segments = list(kwargs["source_segments"])
        indices = [int(seg["segment_index"]) for seg in source_segments]
        if indices and indices[0] >= 6:
            raise RuntimeError("Chat labels response failed integrity checks: response missing segment indices: [8]")
        return {
            "segments": [
                {
                    "segment_index": int(seg["segment_index"]),
                    "start_sec": float(seg["start_sec"]),
                    "end_sec": float(seg["end_sec"]),
                    "label": f"resolved {int(seg['segment_index'])}",
                }
                for seg in source_segments
            ],
            "attach_notes": [],
            "usage": {},
            "out_json": str(tmp_path / "chunk.json"),
            "prompt_path": str(tmp_path / "chunk.txt"),
        }

    monkeypatch.setattr("src.solver.legacy_impl.call_gemini_labels", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("API solve should not be called")))
    monkeypatch.setattr("src.solver.legacy_impl._persist_task_state_fields", lambda cfg, task_id, task_state=None, **updates: dict(task_state or {}, **updates))
    monkeypatch.setattr("src.solver.legacy_impl._resolve_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr("src.solver.legacy_impl._segment_chunks", fake_segment_chunks)
    monkeypatch.setattr("src.solver.legacy_impl._probe_video_duration_seconds", lambda _path: 0.0)
    monkeypatch.setattr("src.solver.legacy_impl._maybe_optimize_video_for_upload", lambda path, cfg: None)
    monkeypatch.setattr("src.solver.legacy_impl._extract_video_window", lambda **kwargs: False)
    monkeypatch.setattr("src.solver.chat_only.run_labels_generation", fake_run_labels_generation)
    monkeypatch.setattr(
        "src.solver.chat_only.restart_episode_gemini_session",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("session unavailable")),
    )
    monkeypatch.setattr(
        "src.solver.chat_only.run_labels_generation",
        fake_run_labels_generation,
    )
    monkeypatch.setattr(
        "src.solver.gemini._request_labels_via_gemini",
        lambda *args, **kwargs: fallback_calls.append("single-request") or {"segments": []},
        raising=False,
    )

    try:
        gemini._request_labels_with_optional_segment_chunking(
            cfg,
            segments,
            "prompt text",
            video_file,
            allow_operations=False,
            task_id="ep_chat_retry_abort",
            task_state=task_state,
            stage_name="labeling",
        )
    except RuntimeError as exc:
        assert "session restart failed before chunk fallback retry" in str(exc)
    else:
        raise AssertionError("Expected chunk retry to fail closed when session restart fails")

    assert fallback_calls == []


def test_process_policy_gate_skips_compare_in_chat_only_mode(monkeypatch) -> None:
    cfg = {
        "run": {"chat_only_mode": True, "enable_structural_actions": True},
        "gemini": {"model": "gemini-3.1-pro-preview"},
    }
    segments = [{"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "current_label": "move item"}]
    labels_payload = {
        "segments": [{"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "label": "pick up item"}],
        "_meta": {"model": "gemini-3.1-pro-preview", "solve_backend": "chat_web", "chat_only_mode": True},
    }
    segment_plan = {
        1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "label": "pick up item"}
    }
    task_state: dict[str, object] = {}

    def fake_persist(cfg, task_id, task_state=None, **updates):
        merged = dict(task_state or {})
        merged.update(updates)
        if isinstance(task_state, dict):
            task_state.clear()
            task_state.update(merged)
            return task_state
        return merged

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
    monkeypatch.setattr(
        orchestrator,
        "_maybe_repair_overlong_segments",
        lambda **kwargs: {
            "segments": kwargs["segments"],
            "prompt": kwargs["prompt"],
            "labels_payload": kwargs["labels_payload"],
            "segment_plan": kwargs["segment_plan"],
            "validation_report": kwargs["validation_report"],
            "task_state": kwargs.get("task_state"),
            "repair_rounds": 0,
            "skip_compare": False,
        },
    )
    monkeypatch.setattr("src.solver.legacy_impl._validate_segment_plan_against_policy", lambda cfg, segments, plan: {"errors": [], "warnings": []})
    monkeypatch.setattr("src.solver.legacy_impl._save_validation_report", lambda cfg, task_id, report: None)
    monkeypatch.setattr("src.solver.legacy_impl._persist_task_state_fields", fake_persist)
    monkeypatch.setattr(
        "src.solver.legacy_impl._maybe_run_pre_submit_chat_compare",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("compare should be skipped in chat_only_mode")),
    )

    result = orchestrator._process_policy_gate_and_compare(
        cfg=cfg,
        page=None,
        segments=segments,
        prompt="prompt",
        video_file=None,
        labels_payload=labels_payload,
        segment_plan=segment_plan,
        episode_no=1,
        task_id="ep_chat_only",
        execute=True,
        task_state=task_state,
        enable_structural_actions=True,
        requery_after_structural_actions=True,
    )

    assert result["compare_result"]["decision"] == "skip_chat_only_mode"
    assert task_state["chat_compare_skipped"] is True


def test_label_action_clauses_keeps_object_conjunctions_intact() -> None:
    assert label_rules._label_action_clauses("scrub shoe sole and side with yellow brush") == [
        "scrub shoe sole and side with yellow brush"
    ]
    assert label_rules._label_action_clauses("pick up brush and place brush on sink") == [
        "pick up brush",
        "place brush on sink",
    ]


def test_chat_only_policy_retry_disabled_by_default(monkeypatch) -> None:
    cfg = {
        "run": {"chat_only_mode": True},
        "gemini": {"policy_retry_model": "gemini-3.1-pro-preview", "model": "gemini-3.1-pro-preview"},
    }

    def fail_request(*args, **kwargs):
        raise AssertionError("chat-only policy retry should be skipped by default")

    monkeypatch.setattr(
        "src.solver.legacy_impl._request_labels_with_optional_segment_chunking",
        fail_request,
    )

    result = orchestrator._maybe_retry_policy_with_stronger_model(
        cfg=cfg,
        segments=[{"segment_index": 1, "start_sec": 0.0, "end_sec": 12.0, "current_label": "move item"}],
        prompt="prompt",
        video_file=None,
        labels_payload={"_meta": {"model": "gemini-3.1-pro-preview"}},
        segment_plan={1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 12.0, "label": "move item"}},
        validation_report={"errors": ["segment 1: duration 12.0s exceeds max 10.0s"], "warnings": []},
        task_id="ep_chat_only",
        task_state={},
    )

    assert result["retried"] is False
    assert result["adopted_retry"] is False


def test_estimate_minimum_episode_cost_skips_compare_when_chat_only() -> None:
    cfg = {
        "run": {
            "chat_only_mode": True,
            "pre_submit_chat_compare_enabled": True,
            "segment_chunking_max_segments_per_request": 8,
            "segment_chunking_min_segments": 16,
        },
        "gemini": {
            "model": "gemini-3.1-pro-preview",
            "stage_models": {
                "labeling": "gemini-3.1-pro-preview",
                "compare_chat": "gemini-3.1-pro-preview",
            },
        },
        "economics": {
            "episode_expected_revenue_usd": 0.5,
            "target_cost_ratio": 0.15,
            "hard_cost_ratio": 0.20,
        },
    }

    summary = estimate_minimum_episode_cost_usd(cfg, 20)

    assert summary["minimum_compare_cost_usd"] == 0.0
    assert summary["minimum_compare_model"] == ""


def test_run_structural_planner_uses_subprocess_payload(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "_meta": {"config_path": str(tmp_path / "cfg.yaml"), "config_dir": str(tmp_path)},
        "gemini": {"chat_ops_response_schema_enabled": True, "chat_web_timeout_sec": 180},
    }
    Path(cfg["_meta"]["config_path"]).write_text("gemini: {}\n", encoding="utf-8")
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"00")
    captured: dict[str, object] = {}

    def fake_subprocess_once(**kwargs):
        captured["mode"] = kwargs["mode"]
        captured["prompt_scope"] = kwargs["prompt_scope"]
        return {
            "parsed": {"operations": [{"action": "split", "segment_index": 2}]},
            "attach_notes": ["ok"],
            "usage": {"totalTokenCount": 12},
            "raw_text": "{}",
            "prompt_path": str(tmp_path / "cache" / "ops_prompt.txt"),
        }

    monkeypatch.setattr(chat_only, "_run_chat_subprocess_once", fake_subprocess_once)

    result = chat_only.run_structural_planner(
        cfg=cfg,
        source_segments=[{"segment_index": 2, "start_sec": 0.0, "end_sec": 9.0}],
        video_file=video_file,
        prompt_text="plan",
        cache_dir=tmp_path / "cache",
        episode_id="ep1",
        model="gemini-3.1-pro-preview",
        allow_merge=False,
    )

    assert result["operations"] == [{"action": "split", "segment_index": 2}]
    assert captured["mode"] == "ops"
    assert captured["prompt_scope"] == "chat_ops"


def test_chat_only_soft_fails_planner_and_continues(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "run": {
            "output_dir": str(tmp_path / "outputs"),
            "chat_only_mode": True,
            "primary_solve_backend": "chat_web",
            "chat_ops_fail_open": True,
            "chat_ops_run_without_overlong": False,
            "enable_structural_actions": True,
            "structural_allow_split": True,
            "structural_allow_merge": False,
            "capture_step_screenshots": False,
        },
        "gemini": {
            "model": "gemini-3.1-pro-preview",
            "stage_models": {"labeling": "gemini-3.1-pro-preview"},
        },
        "_meta": {"config_dir": str(tmp_path)},
    }
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"00")
    task_state: dict[str, object] = {}
    segments = [
        {
            "segment_index": 1,
            "start_sec": 0.0,
            "end_sec": 12.0,
            "current_label": "move item",
            "raw_text": "move item",
        }
    ]

    def fake_persist(cfg, task_id, task_state=None, **updates):
        merged = dict(task_state or {})
        merged.update(updates)
        if isinstance(task_state, dict):
            task_state.clear()
            task_state.update(merged)
            return task_state
        return merged

    def fake_run_structural_planner(**kwargs):
        raise RuntimeError("planner timeout")

    def fake_run_labels_generation(**kwargs):
        out_json = tmp_path / "chat_labels.json"
        prompt_path = tmp_path / "chat_prompt.txt"
        out_json.write_text("{}", encoding="utf-8")
        prompt_path.write_text("prompt", encoding="utf-8")
        return {
            "segments": [
                {
                    "segment_index": 1,
                    "start_sec": 0.0,
                    "end_sec": 12.0,
                    "label": "pick up item",
                }
            ],
            "attach_notes": ["ok"],
            "usage": {"promptTokenCount": 100, "candidatesTokenCount": 10, "totalTokenCount": 110},
            "out_json": str(out_json),
            "prompt_path": str(prompt_path),
        }

    monkeypatch.setattr("src.solver.legacy_impl._persist_task_state_fields", fake_persist)
    monkeypatch.setattr("src.solver.legacy_impl._resolve_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr("src.solver.legacy_impl._segment_duration_exceeds_limit", lambda seg, max_sec: True)
    monkeypatch.setattr("src.solver.chat_only.run_structural_planner", fake_run_structural_planner)
    monkeypatch.setattr("src.solver.chat_only.run_labels_generation", fake_run_labels_generation)

    result = gemini._request_labels_with_optional_segment_chunking(
        cfg,
        segments,
        "prompt text",
        video_file,
        allow_operations=True,
        task_id="ep_chat_only",
        task_state=task_state,
        stage_name="labeling",
    )

    assert result["segments"][0]["label"] == "pick up item"
    assert result["operations"] == [{"action": "split", "segment_index": 1}]
    assert task_state["chat_ops_attempted"] is True
    assert task_state["chat_ops_planned"] == 1
    assert task_state["chat_ops_failure_reason"] == "planner timeout"
    assert task_state["last_error"] == ""


def test_chat_only_uses_optimized_video_for_labels(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "run": {
            "output_dir": str(tmp_path / "outputs"),
            "chat_only_mode": True,
            "primary_solve_backend": "chat_web",
            "enable_structural_actions": True,
            "structural_allow_split": True,
            "structural_allow_merge": False,
            "capture_step_screenshots": False,
        },
        "gemini": {
            "model": "gemini-3.1-pro-preview",
            "stage_models": {"labeling": "gemini-3.1-pro-preview"},
        },
        "_meta": {"config_dir": str(tmp_path)},
    }
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"00")
    optimized_file = tmp_path / "video_upload_opt.mp4"
    optimized_file.write_bytes(b"11")
    captured: dict[str, object] = {}

    def fake_persist(cfg, task_id, task_state=None, **updates):
        return updates

    def fake_run_labels_generation(**kwargs):
        captured["video_file"] = str(kwargs.get("video_file"))
        out_json = tmp_path / "chat_labels.json"
        prompt_path = tmp_path / "chat_prompt.txt"
        out_json.write_text("{}", encoding="utf-8")
        prompt_path.write_text("prompt", encoding="utf-8")
        return {
            "segments": [
                {
                    "segment_index": 1,
                    "start_sec": 0.0,
                    "end_sec": 4.0,
                    "label": "pick up item",
                }
            ],
            "attach_notes": ["ok"],
            "usage": {"promptTokenCount": 100, "candidatesTokenCount": 10, "totalTokenCount": 110},
            "out_json": str(out_json),
            "prompt_path": str(prompt_path),
        }

    monkeypatch.setattr("src.solver.legacy_impl._persist_task_state_fields", fake_persist)
    monkeypatch.setattr("src.solver.legacy_impl._resolve_ffmpeg_binary", lambda: "")
    monkeypatch.setattr("src.solver.legacy_impl._segment_duration_exceeds_limit", lambda seg, max_sec: False)
    monkeypatch.setattr("src.solver.legacy_impl._maybe_optimize_video_for_upload", lambda video, cfg: optimized_file)
    monkeypatch.setattr("src.solver.chat_only.run_labels_generation", fake_run_labels_generation)

    gemini._request_labels_with_optional_segment_chunking(
        cfg,
        [{"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "current_label": "move item", "raw_text": "move item"}],
        "prompt text",
        video_file,
        allow_operations=True,
        task_id="ep_chat_only",
        task_state={},
        stage_name="labeling",
    )

    assert captured["video_file"] == str(optimized_file)


def test_chat_only_recovers_prepared_video_from_episode_cache(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "run": {
            "output_dir": str(tmp_path / "outputs"),
            "chat_only_mode": True,
            "primary_solve_backend": "chat_web",
            "enable_structural_actions": True,
            "structural_allow_split": True,
            "structural_allow_merge": False,
            "capture_step_screenshots": False,
        },
        "gemini": {
            "model": "gemini-3.1-pro-preview",
            "stage_models": {"repair": "gemini-3.1-pro-preview"},
        },
        "_meta": {"config_dir": str(tmp_path)},
    }
    missing_video = tmp_path / "missing.mp4"
    cached_video = tmp_path / "outputs" / "_chat_only" / "ep_repair" / "video_ep_repair.mp4"
    cached_video.parent.mkdir(parents=True, exist_ok=True)
    cached_video.write_bytes(b"00")
    captured: dict[str, object] = {}

    def fake_persist(cfg, task_id, task_state=None, **updates):
        return updates

    def fake_run_labels_generation(**kwargs):
        captured["video_file"] = str(kwargs.get("video_file"))
        out_json = tmp_path / "repair_labels.json"
        prompt_path = tmp_path / "repair_prompt.txt"
        out_json.write_text("{}", encoding="utf-8")
        prompt_path.write_text("prompt", encoding="utf-8")
        return {
            "segments": [
                {
                    "segment_index": 1,
                    "start_sec": 0.0,
                    "end_sec": 4.0,
                    "label": "repair item",
                }
            ],
            "attach_notes": ["ok"],
            "usage": {"promptTokenCount": 100, "candidatesTokenCount": 10, "totalTokenCount": 110},
            "out_json": str(out_json),
            "prompt_path": str(prompt_path),
        }

    monkeypatch.setattr("src.solver.legacy_impl._persist_task_state_fields", fake_persist)
    monkeypatch.setattr("src.solver.legacy_impl._resolve_ffmpeg_binary", lambda: "")
    monkeypatch.setattr("src.solver.legacy_impl._segment_duration_exceeds_limit", lambda seg, max_sec: False)
    monkeypatch.setattr("src.solver.legacy_impl._maybe_optimize_video_for_upload", lambda video, cfg: video)
    monkeypatch.setattr("src.solver.chat_only.run_labels_generation", fake_run_labels_generation)

    gemini._request_labels_with_optional_segment_chunking(
        cfg,
        [{"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "current_label": "repair item", "raw_text": "repair item"}],
        "repair prompt",
        missing_video,
        allow_operations=False,
        task_id="ep_repair",
        task_state={},
        stage_name="repair",
    )

    assert captured["video_file"] == str(cached_video)


def test_chat_only_persists_episode_cache_video(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "run": {
            "output_dir": str(tmp_path / "outputs"),
            "chat_only_mode": True,
            "primary_solve_backend": "chat_web",
            "enable_structural_actions": True,
            "structural_allow_split": True,
            "structural_allow_merge": False,
            "capture_step_screenshots": False,
        },
        "gemini": {
            "model": "gemini-3.1-pro-preview",
            "stage_models": {"labeling": "gemini-3.1-pro-preview"},
        },
        "_meta": {"config_dir": str(tmp_path)},
    }
    video_file = tmp_path / "source.mp4"
    video_file.write_bytes(b"00")

    def fake_persist(cfg, task_id, task_state=None, **updates):
        return updates

    def fake_run_labels_generation(**kwargs):
        out_json = tmp_path / "cache_labels.json"
        prompt_path = tmp_path / "cache_prompt.txt"
        out_json.write_text("{}", encoding="utf-8")
        prompt_path.write_text("prompt", encoding="utf-8")
        return {
            "segments": [
                {
                    "segment_index": 1,
                    "start_sec": 0.0,
                    "end_sec": 4.0,
                    "label": "pick up item",
                }
            ],
            "attach_notes": ["ok"],
            "usage": {"promptTokenCount": 100, "candidatesTokenCount": 10, "totalTokenCount": 110},
            "out_json": str(out_json),
            "prompt_path": str(prompt_path),
        }

    monkeypatch.setattr("src.solver.legacy_impl._persist_task_state_fields", fake_persist)
    monkeypatch.setattr("src.solver.legacy_impl._resolve_ffmpeg_binary", lambda: "")
    monkeypatch.setattr("src.solver.legacy_impl._segment_duration_exceeds_limit", lambda seg, max_sec: False)
    monkeypatch.setattr("src.solver.legacy_impl._maybe_optimize_video_for_upload", lambda video, cfg: video)
    monkeypatch.setattr("src.solver.chat_only.run_labels_generation", fake_run_labels_generation)

    gemini._request_labels_with_optional_segment_chunking(
        cfg,
        [{"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "current_label": "move item", "raw_text": "move item"}],
        "prompt text",
        video_file,
        allow_operations=False,
        task_id="ep_cache",
        task_state={},
        stage_name="labeling",
    )

    assert (tmp_path / "outputs" / "_chat_only" / "ep_cache" / "video_ep_cache.mp4").exists()


def test_chat_only_chunking_extracts_chunk_videos(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "run": {
            "output_dir": str(tmp_path / "outputs"),
            "chat_only_mode": True,
            "primary_solve_backend": "chat_web",
            "segment_chunking_enabled": True,
            "segment_chunking_min_segments": 2,
            "segment_chunking_max_segments_per_request": 2,
            "segment_chunking_min_video_sec": 0.0,
            "enable_structural_actions": True,
            "structural_allow_split": True,
            "structural_allow_merge": False,
            "capture_step_screenshots": False,
        },
        "gemini": {
            "model": "gemini-3.1-pro-preview",
            "stage_models": {"labeling": "gemini-3.1-pro-preview"},
        },
        "_meta": {"config_dir": str(tmp_path)},
    }
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"00")
    captured_videos: list[str] = []

    def fake_persist(cfg, task_id, task_state=None, **updates):
        return updates

    def fake_extract_video_window(src_video, out_video, start_sec, end_sec, ffmpeg_bin):
        Path(out_video).write_bytes(b"11")
        return True

    def fake_run_labels_generation(**kwargs):
        source_segments = kwargs["source_segments"]
        captured_videos.append(str(kwargs["video_file"]))
        out_json = tmp_path / f"{len(captured_videos)}_chat_labels.json"
        prompt_path = tmp_path / f"{len(captured_videos)}_chat_prompt.txt"
        out_json.write_text("{}", encoding="utf-8")
        prompt_path.write_text("prompt", encoding="utf-8")
        return {
            "segments": [
                {
                    "segment_index": seg["segment_index"],
                    "start_sec": seg["start_sec"],
                    "end_sec": seg["end_sec"],
                    "label": f"pick up item {seg['segment_index']}",
                }
                for seg in source_segments
            ],
            "attach_notes": ["ok"],
            "usage": {"promptTokenCount": 100, "candidatesTokenCount": 10, "totalTokenCount": 110},
            "out_json": str(out_json),
            "prompt_path": str(prompt_path),
        }

    monkeypatch.setattr("src.solver.legacy_impl._persist_task_state_fields", fake_persist)
    monkeypatch.setattr("src.solver.legacy_impl._resolve_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr("src.solver.legacy_impl._segment_duration_exceeds_limit", lambda seg, max_sec: False)
    monkeypatch.setattr("src.solver.legacy_impl._extract_video_window", fake_extract_video_window)
    monkeypatch.setattr("src.solver.legacy_impl._maybe_optimize_video_for_upload", lambda video, cfg: video_file)
    monkeypatch.setattr("src.solver.chat_only.run_labels_generation", fake_run_labels_generation)

    result = gemini._request_labels_with_optional_segment_chunking(
        cfg,
        [
            {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "current_label": "move item", "raw_text": "move item"},
            {"segment_index": 2, "start_sec": 4.0, "end_sec": 8.0, "current_label": "move item", "raw_text": "move item"},
            {"segment_index": 3, "start_sec": 8.0, "end_sec": 12.0, "current_label": "move item", "raw_text": "move item"},
        ],
        "prompt text",
        video_file,
        allow_operations=True,
        task_id="ep_chunked",
        task_state={},
        stage_name="labeling",
    )

    assert len(captured_videos) == 2
    assert all("chatchunk" in path for path in captured_videos)
    assert len(result["segments"]) == 3


def test_chat_only_chunking_falls_back_to_single_request(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "run": {
            "output_dir": str(tmp_path / "outputs"),
            "chat_only_mode": True,
            "primary_solve_backend": "chat_web",
            "segment_chunking_enabled": True,
            "segment_chunking_min_segments": 2,
            "segment_chunking_max_segments_per_request": 2,
            "segment_chunking_min_video_sec": 0.0,
            "chat_chunk_fallback_to_single_request": True,
            "enable_structural_actions": True,
            "structural_allow_split": True,
            "structural_allow_merge": False,
            "capture_step_screenshots": False,
        },
        "gemini": {
            "model": "gemini-3.1-pro-preview",
            "stage_models": {"labeling": "gemini-3.1-pro-preview"},
        },
        "_meta": {"config_dir": str(tmp_path)},
    }
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"00")
    captured_videos: list[str] = []

    def fake_persist(cfg, task_id, task_state=None, **updates):
        return updates

    def fake_extract_video_window(src_video, out_video, start_sec, end_sec, ffmpeg_bin):
        Path(out_video).write_bytes(b"11")
        return True

    def fake_run_labels_generation(**kwargs):
        source_segments = kwargs["source_segments"]
        video_path = str(kwargs["video_file"])
        captured_videos.append(video_path)
        if "chatchunk" in video_path:
            raise RuntimeError("chunk timeout")
        out_json = tmp_path / "fallback_chat_labels.json"
        prompt_path = tmp_path / "fallback_chat_prompt.txt"
        out_json.write_text("{}", encoding="utf-8")
        prompt_path.write_text("prompt", encoding="utf-8")
        return {
            "segments": [
                {
                    "segment_index": seg["segment_index"],
                    "start_sec": seg["start_sec"],
                    "end_sec": seg["end_sec"],
                    "label": f"pick up item {seg['segment_index']}",
                }
                for seg in source_segments
            ],
            "attach_notes": ["ok"],
            "usage": {"promptTokenCount": 100, "candidatesTokenCount": 10, "totalTokenCount": 110},
            "out_json": str(out_json),
            "prompt_path": str(prompt_path),
        }

    monkeypatch.setattr("src.solver.legacy_impl._persist_task_state_fields", fake_persist)
    monkeypatch.setattr("src.solver.legacy_impl._resolve_ffmpeg_binary", lambda: "ffmpeg")
    monkeypatch.setattr("src.solver.legacy_impl._segment_duration_exceeds_limit", lambda seg, max_sec: False)
    monkeypatch.setattr("src.solver.legacy_impl._extract_video_window", fake_extract_video_window)
    monkeypatch.setattr("src.solver.legacy_impl._maybe_optimize_video_for_upload", lambda video, cfg: video_file)
    monkeypatch.setattr("src.solver.chat_only.run_labels_generation", fake_run_labels_generation)

    result = gemini._request_labels_with_optional_segment_chunking(
        cfg,
        [
            {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "current_label": "move item", "raw_text": "move item"},
            {"segment_index": 2, "start_sec": 4.0, "end_sec": 8.0, "current_label": "move item", "raw_text": "move item"},
            {"segment_index": 3, "start_sec": 8.0, "end_sec": 12.0, "current_label": "move item", "raw_text": "move item"},
        ],
        "prompt text",
        video_file,
        allow_operations=True,
        task_id="ep_chunked_fallback",
        task_state={},
        stage_name="labeling",
    )

    assert any("chatchunk" in path for path in captured_videos)
    assert captured_videos[-1] == str(video_file)
    assert result["_meta"]["chunk_count"] == 1
    assert len(result["segments"]) == 3


def test_chat_only_skips_planner_when_chat_ops_disabled(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "run": {
            "output_dir": str(tmp_path / "outputs"),
            "chat_only_mode": True,
            "primary_solve_backend": "chat_web",
            "chat_ops_enabled": False,
            "enable_structural_actions": True,
            "structural_allow_split": True,
            "structural_allow_merge": True,
            "auto_continuity_merge_enabled": True,
            "capture_step_screenshots": False,
        },
        "gemini": {
            "model": "gemini-3.1-pro-preview",
            "stage_models": {"labeling": "gemini-3.1-pro-preview"},
        },
        "_meta": {"config_dir": str(tmp_path)},
    }
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"00")

    def fake_persist(cfg, task_id, task_state=None, **updates):
        return updates

    def fake_run_structural_planner(**kwargs):
        raise AssertionError("planner should be skipped when chat_ops_enabled=false")

    def fake_run_labels_generation(**kwargs):
        out_json = tmp_path / "chat_labels.json"
        prompt_path = tmp_path / "chat_prompt.txt"
        out_json.write_text("{}", encoding="utf-8")
        prompt_path.write_text("prompt", encoding="utf-8")
        return {
            "segments": [
                {
                    "segment_index": 1,
                    "start_sec": 0.0,
                    "end_sec": 4.0,
                    "label": "pick up item",
                }
            ],
            "attach_notes": ["ok"],
            "usage": {"promptTokenCount": 100, "candidatesTokenCount": 10, "totalTokenCount": 110},
            "out_json": str(out_json),
            "prompt_path": str(prompt_path),
        }

    monkeypatch.setattr("src.solver.legacy_impl._persist_task_state_fields", fake_persist)
    monkeypatch.setattr("src.solver.legacy_impl._resolve_ffmpeg_binary", lambda: "")
    monkeypatch.setattr("src.solver.legacy_impl._segment_duration_exceeds_limit", lambda seg, max_sec: True)
    monkeypatch.setattr("src.solver.chat_only.run_structural_planner", fake_run_structural_planner)
    monkeypatch.setattr("src.solver.chat_only.run_labels_generation", fake_run_labels_generation)

    result = gemini._request_labels_with_optional_segment_chunking(
        cfg,
        [{"segment_index": 1, "start_sec": 0.0, "end_sec": 12.0, "current_label": "move item", "raw_text": "move item"}],
        "prompt text",
        video_file,
        allow_operations=True,
        task_id="ep_chat_only",
        task_state={},
        stage_name="labeling",
    )

    assert result["_meta"]["chat_ops_attempted"] is False


def test_chat_only_payload_meta_carries_session_retry_fields(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "run": {
            "output_dir": str(tmp_path / "outputs"),
            "chat_only_mode": True,
            "primary_solve_backend": "chat_web",
            "enable_structural_actions": True,
            "structural_allow_split": False,
            "structural_allow_merge": False,
            "capture_step_screenshots": False,
        },
        "gemini": {
            "model": "gemini-3.1-pro-preview",
            "stage_models": {"labeling": "gemini-3.1-pro-preview"},
        },
        "_meta": {"config_dir": str(tmp_path)},
    }
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"00")
    task_state: dict[str, object] = {}

    def fake_persist(cfg, task_id, task_state=None, **updates):
        merged = dict(task_state or {})
        merged.update(updates)
        if isinstance(task_state, dict):
            task_state.clear()
            task_state.update(merged)
            return task_state
        return merged

    def fake_run_labels_generation(**kwargs):
        out_json = tmp_path / "chat_labels.json"
        prompt_path = tmp_path / "chat_prompt.txt"
        out_json.write_text("{}", encoding="utf-8")
        prompt_path.write_text("prompt", encoding="utf-8")
        return {
            "segments": [
                {
                    "segment_index": 1,
                    "start_sec": 0.0,
                    "end_sec": 4.0,
                    "label": "pick up item",
                }
            ],
            "attach_notes": ["video.mp4: attached"],
            "usage": {"promptTokenCount": 100, "candidatesTokenCount": 10, "totalTokenCount": 110},
            "out_json": str(out_json),
            "prompt_path": str(prompt_path),
            "request_id": "req-meta",
            "gemini_session_id": "sess-meta",
            "retry_stage": "targeted_repair_1",
            "retry_reason": "policy_overlong",
            "latency_ms": 432,
            "raw_response_path": str(tmp_path / "raw_meta.txt"),
        }

    monkeypatch.setattr("src.solver.legacy_impl._persist_task_state_fields", fake_persist)
    monkeypatch.setattr("src.solver.legacy_impl._resolve_ffmpeg_binary", lambda: "")
    monkeypatch.setattr("src.solver.legacy_impl._segment_duration_exceeds_limit", lambda seg, max_sec: False)
    monkeypatch.setattr("src.solver.chat_only.run_labels_generation", fake_run_labels_generation)

    result = gemini._request_labels_with_optional_segment_chunking(
        cfg,
        [{"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "current_label": "move item", "raw_text": "move item"}],
        "prompt text",
        video_file,
        allow_operations=False,
        task_id="ep_meta",
        task_state=task_state,
        stage_name="labeling",
    )

    meta = result["_meta"]
    assert meta["request_id"] == "req-meta"
    assert meta["gemini_session_id"] == "sess-meta"
    assert meta["retry_stage"] == "targeted_repair_1"
    assert meta["retry_reason"] == "policy_overlong"
    assert meta["gemini_latency_ms"] == 432
    assert task_state["gemini_last_request_id"] == "req-meta"
    assert task_state["gemini_last_retry_stage"] == "targeted_repair_1"


def test_overlong_repair_fails_closed_after_targeted_rounds(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "run": {
            "chat_only_mode": True,
            "policy_auto_split_repair_enabled": True,
            "policy_auto_split_repair_max_rounds": 2,
            "targeted_repair_max_rounds": 2,
            "policy_auto_split_repair_max_segments_per_round": 1,
            "structural_allow_split": True,
        },
        "gemini": {
            "model": "gemini-3.1-pro-preview",
            "stage_models": {"repair": "gemini-3.1-pro-preview"},
        },
        "_meta": {"config_dir": str(tmp_path)},
    }
    task_state: dict[str, object] = {}
    segments = [{"segment_index": 1, "start_sec": 0.0, "end_sec": 12.0, "current_label": "move item"}]
    validation_report = {"errors": ["segment 1: duration 12.0s exceeds max 10.0s"], "warnings": []}
    calls = {"planner": 0, "apply_ops": 0, "label_requery": 0}
    video_file = tmp_path / "repair.mp4"
    video_file.write_bytes(b"00")

    def fake_persist(cfg, task_id, task_state=None, **updates):
        merged = dict(task_state or {})
        merged.update(updates)
        if isinstance(task_state, dict):
            task_state.clear()
            task_state.update(merged)
            return task_state
        return merged

    def fake_run_structural_planner(**kwargs):
        calls["planner"] += 1
        return {
            "operations": [{"action": "split", "segment_index": 1}],
            "_meta": {"retry_stage": f"targeted_repair_{min(calls['planner'], 2)}"},
        }

    def fake_request_labels(*args, **kwargs):
        calls["label_requery"] += 1
        if calls["label_requery"] <= 2:
            return {
                "segments": [
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 12.0,
                        "label": "still too long",
                    }
                ],
                "_meta": {},
            }
    def fake_apply_segment_operations(*args, **kwargs):
        calls["apply_ops"] += 1
        return {"applied": 1, "structural_applied": 1, "failed": []}

    monkeypatch.setattr("src.solver.legacy_impl._persist_task_state_fields", fake_persist)
    monkeypatch.setattr("src.solver.legacy_impl._save_task_text_files", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.solver.legacy_impl._save_cached_segments", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.solver.legacy_impl._save_cached_labels", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.solver.legacy_impl._save_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.solver.legacy_impl._invalidate_cached_labels", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.solver.legacy_impl.extract_segments", lambda page, cfg: segments)
    monkeypatch.setattr("src.solver.legacy_impl.apply_segment_operations", fake_apply_segment_operations)
    monkeypatch.setattr("src.solver.legacy_impl._rewrite_no_action_pauses_in_plan", lambda plan, cfg: 0)
    monkeypatch.setattr(
        "src.solver.legacy_impl.build_prompt",
        lambda segments, extra, allow_operations=False, policy_trigger="": "rebuilt prompt",
    )
    monkeypatch.setattr(
        "src.solver.legacy_impl._normalize_segment_plan",
        lambda payload, current_segments, cfg=None: {
            int(item["segment_index"]): dict(item)
            for item in payload.get("segments", [])
        },
    )
    monkeypatch.setattr(
        "src.solver.legacy_impl._validate_segment_plan_against_policy",
        lambda cfg, current_segments, plan: (
            {"errors": [], "warnings": []}
            if any(float(item.get("end_sec", 0.0)) <= 10.0 for item in plan.values())
            else {"errors": ["segment 1: duration 12.0s exceeds max 10.0s"], "warnings": []}
        ),
    )
    monkeypatch.setattr("src.solver.chat_only.run_structural_planner", fake_run_structural_planner)
    monkeypatch.setattr("src.solver.legacy_impl._request_labels_with_optional_segment_chunking", fake_request_labels)

    result = orchestrator._maybe_repair_overlong_segments(
        cfg=cfg,
        page=None,
        segments=segments,
        prompt="prompt",
        video_file=video_file,
        labels_payload={"segments": [{"segment_index": 1, "start_sec": 0.0, "end_sec": 12.0, "label": "too long"}]},
        segment_plan={1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 12.0, "label": "too long"}},
        validation_report=validation_report,
        task_id="ep-overlong",
        execute=True,
        task_state=task_state,
        enable_structural_actions=True,
        requery_after_structural_actions=True,
    )

    assert calls["planner"] == 2
    assert calls["apply_ops"] == 2
    assert result["retry_stage"] == "targeted_repair_exhausted"
    assert result["retry_reason"] == "policy_overlong"
    assert result["labels_payload"]["_meta"]["retry_stage"] == "targeted_repair_exhausted"
    assert result["labels_payload"]["_meta"]["repair_fail_closed"] is True
    assert any(
        "full reset regenerate is disabled in production" in str(item)
        for item in result["validation_report"]["errors"]
    )
