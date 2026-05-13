import json
import sqlite3
from pathlib import Path

from src.infra import artifacts, browser_auth, runtime, solver_config
from src.rules import consistency, labels, policy_gate
from src.solver import browser, gemini, legacy_impl, orchestrator, prompting, segments, video


def test_legacy_runtime_aliases_bind_to_extracted_module():
    assert legacy_impl._shutdown_requested is runtime._shutdown_requested
    assert legacy_impl._request_shutdown is runtime._request_shutdown
    assert legacy_impl._install_signal_handlers is runtime._install_signal_handlers


def test_legacy_rule_and_prompt_helpers_bind_to_extracted_modules():
    assert legacy_impl.build_prompt is prompting.build_prompt
    assert legacy_impl._rewrite_label_tier3 is labels._rewrite_label_tier3
    assert legacy_impl._validate_segment_plan_against_policy is policy_gate._validate_segment_plan_against_policy


def test_legacy_artifact_helpers_bind_to_extracted_module():
    assert legacy_impl._task_id_from_url is artifacts._task_id_from_url
    assert legacy_impl._task_scoped_artifact_paths is artifacts._task_scoped_artifact_paths
    assert legacy_impl._load_task_state is artifacts._load_task_state
    assert legacy_impl._save_task_state is artifacts._save_task_state
    assert legacy_impl._load_cached_segments is artifacts._load_cached_segments
    assert legacy_impl._save_cached_segments is artifacts._save_cached_segments
    assert legacy_impl._save_task_text_files is artifacts._save_task_text_files
    assert legacy_impl._labels_cache_path is artifacts._labels_cache_path
    assert legacy_impl._load_cached_labels is artifacts._load_cached_labels
    assert legacy_impl._save_cached_labels is artifacts._save_cached_labels
    assert legacy_impl._invalidate_cached_labels is artifacts._invalidate_cached_labels
    assert legacy_impl._save_validation_report is artifacts._save_validation_report
    assert legacy_impl._save_outputs is artifacts._save_outputs
    assert legacy_impl._capture_debug_artifacts is artifacts._capture_debug_artifacts
    assert legacy_impl._capture_step_artifacts is artifacts._capture_step_artifacts


def test_legacy_browser_auth_helpers_bind_to_extracted_module():
    assert legacy_impl._default_chrome_user_data_dir is browser_auth._default_chrome_user_data_dir
    assert legacy_impl._looks_like_profile_dir_name is browser_auth._looks_like_profile_dir_name
    assert legacy_impl._is_direct_profile_path is browser_auth._is_direct_profile_path
    assert legacy_impl._resolve_atlas_email is browser_auth._resolve_atlas_email
    assert legacy_impl._detect_chrome_profile_for_email is browser_auth._detect_chrome_profile_for_email
    assert legacy_impl._count_site_cookies_in_profile is browser_auth._count_site_cookies_in_profile
    assert legacy_impl._detect_chrome_profile_for_site_cookie is browser_auth._detect_chrome_profile_for_site_cookie
    assert legacy_impl._otp_provider is browser_auth._otp_provider
    assert legacy_impl._otp_is_manual is browser_auth._otp_is_manual
    assert legacy_impl._ensure_parent is browser_auth._ensure_parent
    assert legacy_impl._restore_storage_state is browser_auth._restore_storage_state
    assert legacy_impl._is_too_many_redirects_error is browser_auth._is_too_many_redirects_error
    assert legacy_impl._clear_atlas_site_session is browser_auth._clear_atlas_site_session
    assert legacy_impl._close_chrome_processes is browser_auth._close_chrome_processes
    assert legacy_impl._prepare_chrome_profile_clone is browser_auth._prepare_chrome_profile_clone
    assert legacy_impl._decode_mime_header is browser_auth._decode_mime_header
    assert legacy_impl._message_to_text is browser_auth._message_to_text
    assert legacy_impl._extract_otp_from_messages is browser_auth._extract_otp_from_messages
    assert legacy_impl._imap_login_from_cfg is browser_auth._imap_login_from_cfg
    assert legacy_impl._get_gmail_uid_watermark is browser_auth._get_gmail_uid_watermark
    assert legacy_impl._extract_mailbox_name_from_list_line is browser_auth._extract_mailbox_name_from_list_line
    assert legacy_impl._select_imap_mailbox is browser_auth._select_imap_mailbox
    assert legacy_impl._fetch_otp_gmail_imap is browser_auth._fetch_otp_gmail_imap
    assert legacy_impl._resolve_otp_code is browser_auth._resolve_otp_code
    assert legacy_impl._body_has_rate_limit is browser_auth._body_has_rate_limit
    assert legacy_impl._wait_until_authenticated is browser_auth._wait_until_authenticated
    assert legacy_impl.ensure_logged_in is browser_auth.ensure_logged_in


def test_legacy_browser_helpers_bind_to_extracted_module():
    assert legacy_impl._selector_variants is browser._selector_variants
    assert legacy_impl._goto_with_retry is browser._goto_with_retry
    assert legacy_impl._any_locator_exists is browser._any_locator_exists
    assert legacy_impl._first_visible_locator is browser._first_visible_locator
    assert legacy_impl._safe_locator_click is browser._safe_locator_click
    assert legacy_impl._safe_fill is browser._safe_fill
    assert legacy_impl._safe_locator_text is browser._safe_locator_text
    assert legacy_impl._first_href_from_selector is browser._first_href_from_selector
    assert legacy_impl._all_task_label_hrefs_from_page is browser._all_task_label_hrefs_from_page
    assert legacy_impl._first_task_label_href_from_html is browser._first_task_label_href_from_html
    assert legacy_impl._is_label_page_not_found is browser._is_label_page_not_found
    assert legacy_impl._is_label_page_internal_error is browser._is_label_page_internal_error
    assert legacy_impl._try_go_back_from_label_error is browser._try_go_back_from_label_error
    assert legacy_impl._is_label_page_actionable is browser._is_label_page_actionable
    assert legacy_impl._is_room_access_disabled is browser._is_room_access_disabled
    assert legacy_impl._recover_room_access_disabled is browser._recover_room_access_disabled
    assert legacy_impl._wait_for_any is browser._wait_for_any
    assert legacy_impl._respect_reserve_cooldown is browser._respect_reserve_cooldown
    assert legacy_impl._respect_reserve_min_interval is browser._respect_reserve_min_interval
    assert legacy_impl._mark_reserve_request is browser._mark_reserve_request
    assert legacy_impl._click_reserve_button_dynamic is browser._click_reserve_button_dynamic
    assert legacy_impl._extract_wait_seconds_from_page is browser._extract_wait_seconds_from_page
    assert legacy_impl._reserve_rate_limited is browser._reserve_rate_limited
    assert legacy_impl._room_has_no_reserved_episodes is browser._room_has_no_reserved_episodes
    assert legacy_impl._release_all_reserved_episodes is browser._release_all_reserved_episodes
    assert legacy_impl.goto_task_room is browser.goto_task_room


def test_legacy_video_and_segment_parse_helpers_bind_to_extracted_modules():
    assert legacy_impl._looks_like_video_url is video._looks_like_video_url
    assert legacy_impl._collect_video_url_candidates is video._collect_video_url_candidates
    assert legacy_impl._download_video_via_playwright_request is video._download_video_via_playwright_request
    assert legacy_impl._download_video_from_page_context is video._download_video_from_page_context
    assert legacy_impl._is_probably_mp4 is video._is_probably_mp4
    assert legacy_impl._is_video_decodable is video._is_video_decodable
    assert legacy_impl._ensure_loop_off is video._ensure_loop_off
    assert legacy_impl._play_full_video_once is video._play_full_video_once
    assert legacy_impl._prepare_video_for_gemini is video._prepare_video_for_gemini
    assert legacy_impl._parse_mmss_to_seconds is segments._parse_mmss_to_seconds
    assert legacy_impl._extract_start_end_from_text is segments._extract_start_end_from_text
    assert legacy_impl._resolve_rows_locator is segments._resolve_rows_locator
    assert legacy_impl._first_text_from_row is segments._first_text_from_row
    assert legacy_impl.extract_segments is segments.extract_segments
    assert legacy_impl._normalize_operation_action is segments._normalize_operation_action
    assert legacy_impl._normalize_operations is segments._normalize_operations
    assert segments._normalize_segment_plan is labels._normalize_segment_plan
    assert segments._normalize_label_map_from_plan is labels._normalize_label_map_from_plan
    assert legacy_impl._normalize_segment_plan is labels._normalize_segment_plan
    assert legacy_impl._normalize_label_map_from_plan is labels._normalize_label_map_from_plan
    assert legacy_impl._first_visible_child_locator is segments._first_visible_child_locator


def test_legacy_segment_apply_submit_helpers_bind_to_extracted_module():
    assert legacy_impl._respect_major_step_pause is segments._respect_major_step_pause
    assert legacy_impl._short_error_text is segments._short_error_text
    assert legacy_impl.apply_timestamp_adjustments is segments.apply_timestamp_adjustments
    assert legacy_impl._action_selector_for_row is segments._action_selector_for_row
    assert legacy_impl._action_hotkey is segments._action_hotkey
    assert legacy_impl._confirm_action_dialog is segments._confirm_action_dialog
    assert legacy_impl._wait_rows_delta is segments._wait_rows_delta
    assert legacy_impl.apply_segment_operations is segments.apply_segment_operations
    assert legacy_impl._fill_input is segments._fill_input
    assert legacy_impl._filter_unchanged_label_map is segments._filter_unchanged_label_map
    assert legacy_impl._handle_quality_review_modal is segments._handle_quality_review_modal
    assert legacy_impl._handle_no_edits_modal is segments._handle_no_edits_modal
    assert legacy_impl._submit_transition_observed is segments._submit_transition_observed
    assert legacy_impl._submit_episode is segments._submit_episode
    assert legacy_impl.apply_labels is segments.apply_labels


def test_legacy_gemini_parse_helpers_bind_to_extracted_module():
    assert legacy_impl._extract_retry_seconds_from_text is gemini._extract_retry_seconds_from_text
    assert legacy_impl._extract_retry_seconds_from_response is gemini._extract_retry_seconds_from_response
    assert legacy_impl._set_gemini_quota_cooldown is gemini._set_gemini_quota_cooldown
    assert legacy_impl._respect_gemini_quota_cooldown is gemini._respect_gemini_quota_cooldown
    assert legacy_impl._respect_gemini_rate_limit is gemini._respect_gemini_rate_limit
    assert legacy_impl._is_non_retriable_gemini_error is gemini._is_non_retriable_gemini_error
    assert legacy_impl._clean_json_text is gemini._clean_json_text
    assert legacy_impl._repair_gemini_json_text is gemini._repair_gemini_json_text
    assert legacy_impl._enforce_gemini_output_contract is gemini._enforce_gemini_output_contract
    assert legacy_impl._parse_json_text is gemini._parse_json_text
    assert legacy_impl._parse_gemini_response is gemini._parse_gemini_response
    assert legacy_impl._gemini_file_state is gemini._gemini_file_state
    assert legacy_impl._wait_for_gemini_file_ready is gemini._wait_for_gemini_file_ready
    assert legacy_impl._normalize_upload_chunk_size is gemini._normalize_upload_chunk_size
    assert legacy_impl._upload_video_via_gemini_files_api is gemini._upload_video_via_gemini_files_api
    assert legacy_impl._cleanup_gemini_uploaded_file is gemini._cleanup_gemini_uploaded_file
    assert legacy_impl._sweep_stale_gemini_files is gemini._sweep_stale_gemini_files
    assert legacy_impl._is_gemini_quota_error_text is gemini._is_gemini_quota_error_text
    assert legacy_impl._is_gemini_quota_exceeded_429 is gemini._is_gemini_quota_exceeded_429
    assert legacy_impl._is_gemini_quota_error is gemini._is_gemini_quota_error
    assert legacy_impl._build_gemini_generation_config is gemini._build_gemini_generation_config
    assert legacy_impl._log_gemini_usage is gemini._log_gemini_usage
    assert legacy_impl.call_gemini_labels is gemini.call_gemini_labels
    assert (
        legacy_impl._request_labels_with_optional_segment_chunking
        is gemini._request_labels_with_optional_segment_chunking
    )


def test_orchestrator_stronger_model_retry_adopts_better_candidate(monkeypatch):
    cfg = {
        "gemini": {
            "retry_with_stronger_model_on_policy_fail": True,
            "policy_retry_model": "gemini-3.1-pro-preview",
            "policy_retry_only_if_flash": True,
        }
    }
    segments_in = [{"segment_index": 1, "current_label": "bad label"}]
    labels_payload = {"_meta": {"model": "gemini-3.1-flash-lite-preview"}}
    segment_plan = {1: {"segment_index": 1, "label": "bad label"}}
    validation_report = {"errors": ["segment 1: bad"], "warnings": []}
    retry_payload = {
        "_meta": {"model": "gemini-3.1-pro-preview", "video_attached": True, "mode": "with-video"},
        "segments": [{"segment_index": 1, "label": "better label"}],
    }
    task_state = {}
    saved = {}

    monkeypatch.setattr(legacy_impl, "_request_labels_with_optional_segment_chunking", lambda *args, **kwargs: retry_payload)
    monkeypatch.setattr(legacy_impl, "_normalize_segment_plan", lambda payload, segs, cfg=None: {1: {"segment_index": 1, "label": "better label"}})
    monkeypatch.setattr(legacy_impl, "_rewrite_no_action_pauses_in_plan", lambda plan, cfg: 0)
    monkeypatch.setattr(legacy_impl, "_validate_segment_plan_against_policy", lambda cfg, segs, plan: {"errors": [], "warnings": []})
    monkeypatch.setattr(legacy_impl, "_save_outputs", lambda *args, **kwargs: saved.setdefault("outputs", True))
    monkeypatch.setattr(legacy_impl, "_save_task_text_files", lambda *args, **kwargs: saved.setdefault("texts", True))
    monkeypatch.setattr(legacy_impl, "_save_cached_labels", lambda *args, **kwargs: saved.setdefault("cache", True))
    monkeypatch.setattr(legacy_impl, "_save_task_state", lambda cfg, task_id, state: saved.setdefault("state", dict(state)))

    result = orchestrator._maybe_retry_policy_with_stronger_model(
        cfg=cfg,
        segments=segments_in,
        prompt="prompt",
        video_file=None,
        labels_payload=labels_payload,
        segment_plan=segment_plan,
        validation_report=validation_report,
        task_id="task1",
        execute=False,
        execute_require_video_context=False,
        gemini_uploaded_file_names=[],
        resume_from_artifacts=True,
        task_state=task_state,
    )

    assert result["retried"] is True
    assert result["adopted_retry"] is True
    assert result["segment_plan"][1]["label"] == "better label"
    assert result["validation_report"]["errors"] == []
    assert task_state["labels_ready"] is True
    assert saved["outputs"] is True
    assert saved["texts"] is True
    assert saved["cache"] is True


def test_orchestrator_stronger_model_retry_keeps_primary_when_not_better(monkeypatch):
    cfg = {
        "gemini": {
            "retry_with_stronger_model_on_policy_fail": True,
            "policy_retry_model": "gemini-3.1-pro-preview",
            "policy_retry_only_if_flash": True,
        }
    }
    segments_in = [{"segment_index": 1, "current_label": "bad label"}]
    labels_payload = {"_meta": {"model": "gemini-3.1-flash-lite-preview"}}
    segment_plan = {1: {"segment_index": 1, "label": "bad label"}}
    validation_report = {"errors": ["segment 1: bad"], "warnings": []}
    retry_payload = {"_meta": {"model": "gemini-3.1-pro-preview"}, "segments": [{"segment_index": 1, "label": "still bad"}]}

    monkeypatch.setattr(legacy_impl, "_request_labels_with_optional_segment_chunking", lambda *args, **kwargs: retry_payload)
    monkeypatch.setattr(legacy_impl, "_normalize_segment_plan", lambda payload, segs, cfg=None: {1: {"segment_index": 1, "label": "still bad"}})
    monkeypatch.setattr(legacy_impl, "_rewrite_no_action_pauses_in_plan", lambda plan, cfg: 0)
    monkeypatch.setattr(
        legacy_impl,
        "_validate_segment_plan_against_policy",
        lambda cfg, segs, plan: {"errors": ["segment 1: still bad", "segment 1: extra"], "warnings": []},
    )
    monkeypatch.setattr(legacy_impl, "_save_outputs", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not save retry output")))

    result = orchestrator._maybe_retry_policy_with_stronger_model(
        cfg=cfg,
        segments=segments_in,
        prompt="prompt",
        video_file=None,
        labels_payload=labels_payload,
        segment_plan=segment_plan,
        validation_report=validation_report,
        task_id="task1",
        execute=False,
        execute_require_video_context=False,
        gemini_uploaded_file_names=[],
        resume_from_artifacts=False,
        task_state={},
    )

    assert result["retried"] is True
    assert result["adopted_retry"] is False
    assert result["segment_plan"] is segment_plan
    assert result["validation_report"] is validation_report


def test_orchestrator_policy_gate_compare_adopts_chat_candidate(monkeypatch):
    labels_payload = {"_meta": {"model": "gemini-3.1-pro-preview"}}
    segment_plan = {1: {"segment_index": 1, "label": "api label"}}
    validation_report = {"errors": [], "warnings": []}
    adopted_payload = {"_meta": {"model": "gemini-3.1-pro-preview"}}
    adopted_plan = {1: {"segment_index": 1, "label": "chat label"}}
    adopted_validation = {"errors": [], "warnings": ["minor"]}
    saved = {}
    compare_kwargs = {}

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
    monkeypatch.setattr(legacy_impl, "_validate_segment_plan_against_policy", lambda cfg, segs, plan: validation_report)
    monkeypatch.setattr(legacy_impl, "_save_validation_report", lambda cfg, task_id, report: None)
    monkeypatch.setattr(legacy_impl, "_respect_major_step_pause", lambda cfg, step, heartbeat=None: None)
    def fake_compare(**kwargs):
        compare_kwargs.update(kwargs)
        return {
            "executed": True,
            "winner": "chat",
            "decision": "adopt_chat_same_timeline",
            "adopted": True,
            "selected_plan": adopted_plan,
            "selected_validation_report": adopted_validation,
            "selected_payload": adopted_payload,
            "json_report_path": "",
        }

    monkeypatch.setattr(legacy_impl, "_maybe_run_pre_submit_chat_compare", fake_compare)
    monkeypatch.setattr(legacy_impl, "_save_task_text_files", lambda cfg, task_id, segs, plan: saved.setdefault("text", plan))

    result = orchestrator._process_policy_gate_and_compare(
        cfg={"run": {"enable_policy_gate": True}},
        page=object(),
        segments=[{"segment_index": 1}],
        prompt="prompt",
        video_file=None,
        labels_payload=labels_payload,
        segment_plan=segment_plan,
        task_id="task1",
        execute=False,
        resume_from_artifacts=False,
    )

    assert result["segment_plan"] == adopted_plan
    assert result["validation_report"] == adopted_validation
    assert result["labels_payload"] == adopted_payload
    assert result["warnings"] == ["minor"]
    assert result["errors"] == []
    assert saved["text"] == adopted_plan
    assert compare_kwargs["episode_active_model"] == "gemini-3.1-pro-preview"


def test_orchestrator_policy_gate_filters_ignored_timestamp_and_no_action_errors(monkeypatch):
    task_state = {}
    saved = {}
    validation_report = {
        "errors": [
            "segment 1: invalid timestamp values",
            "segment 2: 'No Action' must be standalone",
            "segment 3: bad label",
        ],
        "warnings": ["warn"],
    }

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
    monkeypatch.setattr(legacy_impl, "_validate_segment_plan_against_policy", lambda cfg, segs, plan: validation_report)
    monkeypatch.setattr(legacy_impl, "_save_validation_report", lambda cfg, task_id, report: None)
    monkeypatch.setattr(legacy_impl, "_respect_major_step_pause", lambda cfg, step, heartbeat=None: None)
    monkeypatch.setattr(legacy_impl, "_maybe_run_pre_submit_chat_compare", lambda **kwargs: {"decision": "skip"})
    monkeypatch.setattr(legacy_impl, "_save_task_state", lambda cfg, task_id, state: saved.setdefault("state", dict(state)))

    result = orchestrator._process_policy_gate_and_compare(
        cfg={
            "run": {
                "enable_policy_gate": True,
                "adjust_timestamps": False,
                "ignore_timestamp_policy_errors_when_adjust_disabled": True,
                "ignore_no_action_standalone_policy_error": True,
            }
        },
        page=object(),
        segments=[{"segment_index": 1}],
        prompt="prompt",
        video_file=None,
        labels_payload={"_meta": {"model": "gemini-3.1-pro-preview"}},
        segment_plan={1: {"segment_index": 1, "label": "bad"}},
        task_id="task1",
        execute=False,
        resume_from_artifacts=True,
        task_state=task_state,
    )

    assert result["warnings"] == ["warn"]
    assert result["errors"] == ["segment 3: bad label"]
    assert task_state["validation_ok"] is False
    assert saved["state"]["validation_ok"] is False


def test_orchestrator_policy_gate_auto_repairs_overlong_segments(monkeypatch):
    task_state = {}
    call_counts = {"validate": 0, "apply": 0}
    initial_validation = {
        "errors": ["segment 1: duration 24.0s exceeds max 10.0s"],
        "warnings": [],
    }
    repaired_validation = {"errors": [], "warnings": []}
    repaired_segments = [
        {"segment_index": 1, "start_sec": 0.0, "end_sec": 9.8, "current_label": "position sandal strap"},
        {"segment_index": 2, "start_sec": 9.8, "end_sec": 19.6, "current_label": "stitch sandal strap"},
    ]
    repaired_plan = {
        1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 9.8, "label": "position sandal strap"},
        2: {"segment_index": 2, "start_sec": 9.8, "end_sec": 19.6, "label": "stitch sandal strap"},
    }
    repaired_payload = {"_meta": {"model": "gemini-3.1-flash-lite-preview", "video_attached": True, "mode": "with-video"}}

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

    def fake_validate(cfg, segs, plan):
        call_counts["validate"] += 1
        return initial_validation if call_counts["validate"] == 1 else repaired_validation

    monkeypatch.setattr(legacy_impl, "_validate_segment_plan_against_policy", fake_validate)
    monkeypatch.setattr(legacy_impl, "_save_validation_report", lambda cfg, task_id, report: None)
    monkeypatch.setattr(legacy_impl, "_respect_major_step_pause", lambda cfg, step, heartbeat=None: None)
    monkeypatch.setattr(legacy_impl, "_maybe_run_pre_submit_chat_compare", lambda **kwargs: {"decision": "skip"})
    monkeypatch.setattr(
        legacy_impl,
        "apply_segment_operations",
        lambda page, cfg, operations, source_segments=None, heartbeat=None: (
            call_counts.__setitem__("apply", call_counts["apply"] + 1) or {
                "applied": len(operations),
                "structural_applied": len(operations),
                "failed": [],
            }
        ),
    )
    monkeypatch.setattr(legacy_impl, "extract_segments", lambda page, cfg: repaired_segments)
    monkeypatch.setattr(legacy_impl, "build_prompt", lambda segs, extra, allow_operations=False, policy_trigger="": "prompt2")
    monkeypatch.setattr(legacy_impl, "_request_labels_with_optional_segment_chunking", lambda *args, **kwargs: repaired_payload)
    monkeypatch.setattr(legacy_impl, "_normalize_operations", lambda payload, cfg=None: [])
    monkeypatch.setattr(legacy_impl, "_save_cached_labels", lambda cfg, task_id, payload: None)
    monkeypatch.setattr(legacy_impl, "_save_cached_segments", lambda cfg, task_id, segs: None)
    monkeypatch.setattr(legacy_impl, "_save_outputs", lambda cfg, segs, prompt, payload, task_id="": None)
    monkeypatch.setattr(legacy_impl, "_save_task_text_files", lambda cfg, task_id, segs, plan: None)
    monkeypatch.setattr(legacy_impl, "_save_task_state", lambda cfg, task_id, state: None)
    monkeypatch.setattr(legacy_impl, "_rewrite_no_action_pauses_in_plan", lambda plan, cfg: 0)
    monkeypatch.setattr(legacy_impl, "_normalize_segment_plan", lambda payload, segs, cfg=None: repaired_plan)

    result = orchestrator._process_policy_gate_and_compare(
        cfg={
            "run": {
                "enable_policy_gate": True,
                "policy_auto_split_repair_enabled": True,
                "policy_auto_split_repair_max_rounds": 2,
                "policy_auto_split_repair_max_segments_per_round": 2,
                "structural_allow_split": True,
            }
        },
        page=object(),
        segments=[{"segment_index": 1, "start_sec": 0.0, "end_sec": 24.0, "current_label": "move sandal"}],
        prompt="prompt",
        video_file=None,
        labels_payload={"_meta": {"model": "gemini-3.1-flash-lite-preview"}},
        segment_plan={1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 24.0, "label": "move sandal"}},
        task_id="task1",
        execute=True,
        execute_require_video_context=False,
        resume_from_artifacts=False,
        task_state=task_state,
        enable_structural_actions=True,
        requery_after_structural_actions=True,
    )

    assert call_counts["apply"] == 1
    assert result["segments"] == repaired_segments
    assert result["segment_plan"] == repaired_plan
    assert result["errors"] == []


def test_orchestrator_policy_gate_overlong_repair_gemini_failure_blocks_without_crashing(monkeypatch):
    task_state = {}
    saved_reports = {}
    validation_calls = {"count": 0}

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

    def fake_validate(cfg, segs, plan):
        validation_calls["count"] += 1
        if validation_calls["count"] == 1:
            return {
                "errors": ["segment 1: duration 24.0s exceeds max 10.0s"],
                "warnings": ["warn"],
            }
        raise AssertionError("validate should not be called again after failed re-query")

    monkeypatch.setattr(legacy_impl, "_validate_segment_plan_against_policy", fake_validate)
    monkeypatch.setattr(
        legacy_impl,
        "_save_validation_report",
        lambda cfg, task_id, report: saved_reports.setdefault("report", report) or None,
    )
    monkeypatch.setattr(
        legacy_impl,
        "_respect_major_step_pause",
        lambda cfg, step, heartbeat=None: (_ for _ in ()).throw(AssertionError("compare pause should be skipped")),
    )
    monkeypatch.setattr(
        legacy_impl,
        "_maybe_run_pre_submit_chat_compare",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("compare should be skipped")),
    )
    monkeypatch.setattr(
        legacy_impl,
        "apply_segment_operations",
        lambda page, cfg, operations, source_segments=None, heartbeat=None: {
            "applied": len(operations),
            "structural_applied": len(operations),
            "failed": [],
        },
    )
    monkeypatch.setattr(
        legacy_impl,
        "extract_segments",
        lambda page, cfg: [
            {"segment_index": 1, "start_sec": 0.0, "end_sec": 9.8, "current_label": "position sandal strap"},
            {"segment_index": 2, "start_sec": 9.8, "end_sec": 19.6, "current_label": "stitch sandal strap"},
        ],
    )
    monkeypatch.setattr(legacy_impl, "build_prompt", lambda segs, extra, allow_operations=False, policy_trigger="": "prompt2")
    monkeypatch.setattr(
        legacy_impl,
        "_request_labels_with_optional_segment_chunking",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "Gemini HTTP 503: {\"error\":{\"code\":503,\"message\":\"This model is currently experiencing high demand.\",\"status\":\"UNAVAILABLE\"}}"
            )
        ),
    )
    monkeypatch.setattr(legacy_impl, "_save_cached_segments", lambda cfg, task_id, segs: None)
    monkeypatch.setattr(legacy_impl, "_save_task_state", lambda cfg, task_id, state: None)

    result = orchestrator._process_policy_gate_and_compare(
        cfg={
            "run": {
                "enable_policy_gate": True,
                "policy_auto_split_repair_enabled": True,
                "policy_auto_split_repair_max_rounds": 2,
                "policy_auto_split_repair_max_segments_per_round": 2,
                "structural_allow_split": True,
            }
        },
        page=object(),
        segments=[{"segment_index": 1, "start_sec": 0.0, "end_sec": 24.0, "current_label": "move sandal"}],
        prompt="prompt",
        video_file=None,
        labels_payload={"_meta": {"model": "gemini-3.1-flash-lite-preview"}},
        segment_plan={1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 24.0, "label": "move sandal"}},
        task_id="task1",
        execute=True,
        execute_require_video_context=False,
        resume_from_artifacts=False,
        task_state=task_state,
        enable_structural_actions=True,
        requery_after_structural_actions=True,
    )

    assert result["compare_result"]["decision"] == "skip_overlong_repair_failure"
    assert len(result["errors"]) == 1
    assert "Policy auto-repair re-query failed after structural changes" in result["errors"][0]
    assert "Gemini HTTP 503" in result["errors"][0]
    assert result["warnings"] == ["warn"]
    assert saved_reports["report"]["errors"] == result["errors"]
    assert task_state["validation_ok"] is False


def test_gemini_parse_helpers_repair_json_and_normalize_lists():
    payload = gemini._parse_json_text(
        """```json
        [{"segment_index": 1, "label": "pick up fabric"}]
        ```"""
    )
    assert payload == {
        "operations": [],
        "segments": [{"segment_index": 1, "label": "pick up fabric"}],
    }

    repaired = gemini._parse_json_text(
        '{"operations":"bad","segments":[{"segment_index":1_2,"label":"align fabric"}]}'
    )
    assert repaired["operations"] == []
    assert repaired["segments"][0]["segment_index"] == "1_2"

    parsed = gemini._parse_gemini_response(
        {
            "candidates": [
                {"content": {"parts": [{"text": ""}, {"text": '{"segments":[{"segment_index":1,"label":"fold fabric"}]}'}]}}
            ]
        }
    )
    assert parsed["segments"][0]["label"] == "fold fabric"


def test_gemini_retry_and_generation_helpers_handle_compound_waits_and_optional_fields():
    wait_sec = gemini._extract_retry_seconds_from_text("Please retry in 3h52m42.1s")
    assert round(wait_sec, 1) == round(3 * 3600 + 52 * 60 + 42.1, 1)

    cfg = {
        "gemini": {
            "temperature": 0.2,
            "candidate_count": 2,
            "top_p": "0.85",
            "top_k": "32",
            "max_output_tokens": "4096",
        }
    }
    assert gemini._build_gemini_generation_config(cfg) == {
        "temperature": 0.2,
        "responseMimeType": "application/json",
        "candidateCount": 2,
        "topP": 0.85,
        "topK": 32,
        "maxOutputTokens": 4096,
    }

    assert gemini._gemini_file_state({"state": {"name": "ready"}}) == "READY"
    assert gemini._is_gemini_quota_error_text("RESOURCE_EXHAUSTED free_tier quota exceeded")
    assert gemini._is_non_retriable_gemini_error(RuntimeError("API key not valid for this request"))


def test_artifact_helpers_round_trip_state_and_text_files(tmp_path):
    cfg = {"run": {"output_dir": str(tmp_path), "reuse_cached_labels": True}}
    task_id = "task123"

    segments = [
        {"segment_index": 1, "start_sec": 0.0, "end_sec": 1.5, "current_label": "pick up fabric"},
        {"segment_index": 2, "start_sec": 1.5, "end_sec": 3.0, "current_label": "move fabric"},
    ]
    segment_plan = {
        1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 1.5, "label": "pick up fabric"},
        2: {"segment_index": 2, "start_sec": 1.5, "end_sec": 3.0, "label": "place fabric on table"},
    }
    labels_payload = {"segments": [{"segment_index": 2, "label": "place fabric on table"}]}
    state_payload = {"episode_submitted": False, "task_id": task_id}

    artifacts._save_task_state(cfg, task_id, state_payload)
    artifacts._save_cached_segments(cfg, task_id, segments)
    artifacts._save_task_text_files(cfg, task_id, segments, segment_plan)
    artifacts._save_cached_labels(cfg, task_id, labels_payload)
    validation_path = artifacts._save_validation_report(cfg, task_id, {"ok": True, "errors": [], "warnings": []})
    artifacts._save_outputs(cfg, segments, "prompt text", labels_payload, task_id=task_id)

    assert artifacts._load_task_state(cfg, task_id) == state_payload
    assert artifacts._load_cached_segments(cfg, task_id) == segments
    assert artifacts._load_cached_labels(cfg, task_id) == labels_payload
    assert validation_path is not None and validation_path.exists()
    assert "pick up fabric" in artifacts._task_scoped_artifact_paths(cfg, task_id)["text_current"].read_text(encoding="utf-8")
    assert "place fabric on table" in artifacts._task_scoped_artifact_paths(cfg, task_id)["text_update"].read_text(encoding="utf-8")

    artifacts._invalidate_cached_labels(cfg, task_id)

    assert artifacts._load_cached_labels(cfg, task_id) is None


def test_capture_step_artifacts_saves_task_scoped_screenshot(tmp_path):
    cfg = {
        "run": {
            "output_dir": str(tmp_path),
            "capture_step_screenshots": True,
            "capture_step_screenshots_full_page": False,
            "capture_step_html": True,
        }
    }

    class DummyPage:
        def screenshot(self, path, full_page=False):
            Path(path).write_bytes(b"png")

        def content(self):
            return "<html><body>ok</body></html>"

    payload = artifacts._capture_step_artifacts(
        DummyPage(),
        cfg,
        "task123",
        "before_policy_gate_compare",
    )

    assert payload["step"] == "before_policy_gate_compare"
    assert Path(payload["screenshot"]).exists()
    assert Path(payload["html"]).exists()


def test_request_labels_promotes_flash_lite_to_pro_on_quota(tmp_path, monkeypatch):
    cfg = {
        "run": {
            "output_dir": str(tmp_path),
            "segment_chunking_enabled": False,
        },
        "gemini": {
            "model": "gemini-3.1-flash-lite-preview",
            "gen3_fallback_models": ["gemini-3.1-pro-preview"],
            "retry_with_quota_fallback_model": True,
            "quota_fallback_model": "gemini-3.1-pro-preview",
            "quota_fallback_from_models": ["gemini-3.1-flash-lite-preview"],
        },
    }
    calls = []

    def fake_call_gemini_labels(request_cfg, prompt_text, video_file=None, segment_count=0, model_override=""):
        calls.append(model_override)
        if model_override == "gemini-3.1-flash-lite-preview":
            raise RuntimeError(
                'Gemini HTTP 429: {"error":{"status":"RESOURCE_EXHAUSTED","message":"quota exceeded"}}'
            )
        return {"segments": [{"segment_index": 1, "label": "fold fabric"}]}

    monkeypatch.setattr(legacy_impl, "call_gemini_labels", fake_call_gemini_labels)

    task_state = {}
    payload = gemini._request_labels_with_optional_segment_chunking(
        cfg,
        [{"segment_index": 1, "start_sec": 0.0, "end_sec": 1.0}],
        "prompt",
        None,
        False,
        task_id="task123",
        task_state=task_state,
    )

    assert calls == ["gemini-3.1-flash-lite-preview", "gemini-3.1-pro-preview"]
    assert task_state["episode_active_model"] == "gemini-3.1-pro-preview"
    assert task_state["episode_model_escalated"] is True
    assert "quota" in task_state["episode_fallback_reason"].lower()
    assert payload["_meta"]["episode_active_model"] == "gemini-3.1-pro-preview"
    persisted = artifacts._load_task_state(cfg, "task123")
    assert persisted["episode_active_model"] == "gemini-3.1-pro-preview"


def test_persist_task_state_fields_merges_existing_state_without_resume(tmp_path):
    cfg = {"run": {"output_dir": str(tmp_path), "reuse_cached_labels": False}}
    task_id = "task123"

    artifacts._save_task_state(
        cfg,
        task_id,
        {"task_id": task_id, "episode_submitted": False, "labels_ready": False},
    )

    runtime_state = {}
    persisted = legacy_impl._persist_task_state_fields(
        cfg,
        task_id,
        runtime_state,
        video_ready=True,
        task_url="https://audit.atlascapture.io/tasks/room/normal/label/task123",
        last_error="",
    )

    assert persisted is runtime_state
    assert runtime_state["task_id"] == task_id
    assert runtime_state["episode_submitted"] is False
    assert runtime_state["video_ready"] is True
    assert runtime_state["task_url"].endswith("/task123")
    assert runtime_state["last_error"] == ""
    assert artifacts._load_task_state(cfg, task_id)["video_ready"] is True


def test_browser_auth_profile_cookie_and_restore_helpers(tmp_path, monkeypatch):
    user_data = tmp_path / "User Data"
    default_profile = user_data / "Default"
    default_profile.mkdir(parents=True)
    (default_profile / "Preferences").write_text(
        json.dumps({"account_info": [{"email": "person@example.com"}]}),
        encoding="utf-8",
    )
    network_dir = default_profile / "Network"
    network_dir.mkdir()
    cookies_db = network_dir / "Cookies"
    conn = sqlite3.connect(cookies_db)
    conn.execute("CREATE TABLE cookies (host_key TEXT)")
    conn.execute("INSERT INTO cookies(host_key) VALUES (?)", (".atlascapture.io",))
    conn.commit()
    conn.close()

    cfg = {"atlas": {"email": "person@example.com"}, "otp": {"provider": "gmail_imap", "gmail_email": "person@example.com", "gmail_app_password": "ab cd ef"}}
    monkeypatch.setenv("ATLAS_LOGIN_EMAIL", "")
    monkeypatch.setenv("ATLAS_EMAIL", "")

    assert browser_auth._looks_like_profile_dir_name("Default") is True
    assert browser_auth._is_direct_profile_path(str(default_profile)) is True
    assert browser_auth._resolve_atlas_email(cfg) == "person@example.com"
    assert browser_auth._detect_chrome_profile_for_email(str(user_data), "person@example.com") == "Default"
    assert browser_auth._count_site_cookies_in_profile(default_profile, "atlascapture.io") == 1
    assert browser_auth._detect_chrome_profile_for_site_cookie(str(user_data)) == "Default"
    assert browser_auth._otp_provider(cfg) == "gmail_imap"
    assert browser_auth._otp_is_manual(cfg) is False
    assert browser_auth._imap_login_from_cfg(cfg) == ("imap.gmail.com", 993, "person@example.com", "abcdef")
    assert browser_auth._extract_mailbox_name_from_list_line('(\\HasNoChildren) "/" "[Gmail]/All Mail"') == "[Gmail]/All Mail"

    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "cookies": [{"name": "a", "value": "1", "domain": ".example.com", "path": "/"}],
                "origins": [
                    {"origin": "https://example.com", "localStorage": [{"name": "foo", "value": "bar"}]},
                ],
            }
        ),
        encoding="utf-8",
    )

    class DummyTempPage:
        def __init__(self):
            self.visits = []
            self.payloads = []
            self.closed = False

        def goto(self, url, wait_until=None, timeout=None):
            self.visits.append((url, wait_until, timeout))

        def evaluate(self, script, items):
            self.payloads.append((script, items))

        def close(self):
            self.closed = True

    class DummyContext:
        def __init__(self):
            self.cookies = []
            self.temp_page = DummyTempPage()

        def add_cookies(self, cookies):
            self.cookies.extend(cookies)

        def new_page(self):
            return self.temp_page

    class DummyPage:
        def __init__(self):
            self.navigations = []

        def goto(self, url, wait_until=None, timeout=None):
            self.navigations.append((url, wait_until, timeout))

    context = DummyContext()
    page = DummyPage()
    assert browser_auth._restore_storage_state(context, page, state_path) is True
    assert context.cookies[0]["name"] == "a"
    assert context.temp_page.visits[0][0] == "https://example.com"
    assert page.navigations[-1][0] == "about:blank"


def test_browser_auth_resolve_otp_code_uses_imap_fetch(monkeypatch):
    cfg = {"otp": {"provider": "gmail_imap"}}

    monkeypatch.setattr(browser_auth, "_fetch_otp_gmail_imap", lambda cfg, started_at_unix, min_uid=None: "123456")

    assert browser_auth._resolve_otp_code(cfg, 0.0) == "123456"
    assert browser_auth._resolve_otp_code({"otp": {"provider": "manual"}}, 0.0) == ""


def test_browser_auth_wait_and_rate_limit_helpers(monkeypatch):
    class DummyPage:
        def __init__(self, url, body):
            self.url = url
            self._body = body

        def inner_text(self, selector):
            assert selector == "body"
            return self._body

    assert browser_auth._body_has_rate_limit(DummyPage("https://audit.atlascapture.io/login", "Too many requests have been made")) is True

    page = DummyPage("https://audit.atlascapture.io/tasks", "")
    browser_auth._wait_until_authenticated(page, {"atlas": {"selectors": {"tasks_nav": ""}}}, timeout_sec=1)

    class WaitingPage(DummyPage):
        pass

    waiting_page = WaitingPage("https://audit.atlascapture.io/login", "")
    tick = {"count": 0}

    def fake_any_locator_exists(page, selector):
        tick["count"] += 1
        if tick["count"] >= 2:
            waiting_page.url = "https://audit.atlascapture.io/tasks"
            return True
        return False

    monkeypatch.setattr(legacy_impl, "_any_locator_exists", fake_any_locator_exists)
    browser_auth._wait_until_authenticated(waiting_page, {"atlas": {"selectors": {"tasks_nav": "#tasks"}}}, timeout_sec=2)


def test_browser_selector_and_room_helpers(monkeypatch):
    assert browser._selector_variants('#a || .b') == ["#a", ".b"]
    assert browser._selector_variants("") == []

    class _FakeLocatorItem:
        def __init__(self, visible=True, href="", text=""):
            self._visible = visible
            self._href = href
            self._text = text

        def is_visible(self):
            return self._visible

        def click(self, timeout=None, no_wait_after=None):
            return None

        def fill(self, value):
            self._text = value

        def inner_text(self, timeout=None):
            return self._text

        def get_attribute(self, name):
            if name == "href":
                return self._href
            return None

    class _FakeLocatorCollection:
        def __init__(self, items):
            self.items = items

        def count(self):
            return len(self.items)

        def nth(self, idx):
            return self.items[idx]

    class _Keyboard:
        def press(self, key):
            return None

        def type(self, value, delay=0):
            return None

    class _FakePage:
        def __init__(self):
            self.url = "https://audit.atlascapture.io/tasks/room/normal"
            self.keyboard = _Keyboard()
            self.waits = []
            self.locators = {
                "#email": _FakeLocatorCollection([_FakeLocatorItem()]),
                'a[href*="/tasks/room/normal/label/"]': _FakeLocatorCollection(
                    [_FakeLocatorItem(href="/tasks/room/normal/label/abc123xyz789")]
                ),
            }

        def locator(self, selector):
            return self.locators.get(selector, _FakeLocatorCollection([]))

        def inner_text(self, selector):
            return "Rooms are unavailable\nRoom access is currently disabled."

        def evaluate(self, script):
            return ["/tasks/room/normal/label/xyz987abc654"]

        def wait_for_timeout(self, ms):
            self.waits.append(ms)

        def goto(self, url, wait_until=None, timeout=None):
            self.url = url

    page = _FakePage()

    assert browser._any_locator_exists(page, "#email") is True
    assert browser._first_visible_locator(page, "#email") is not None
    assert browser._safe_locator_click(page, "#email") is True
    assert browser._safe_fill(page, "#email", "hello") is True
    assert browser._safe_locator_text(page.locator("#email").nth(0)) == "hello"
    assert browser._first_href_from_selector(page, 'a[href*="/tasks/room/normal/label/"]') == "/tasks/room/normal/label/abc123xyz789"
    hrefs = browser._all_task_label_hrefs_from_page(page)
    assert any("/tasks/room/normal/label/" in href for href in hrefs)
    assert browser._first_task_label_href_from_html(page) != ""
    assert browser._is_room_access_disabled(page) is True

    clicked = []
    monkeypatch.setattr(browser, "_safe_locator_click", lambda page_obj, selector, timeout_ms=0: clicked.append(selector) or True)
    assert browser._recover_room_access_disabled(page, {"atlas": {"room_url": "https://audit.atlascapture.io/tasks"}}) is True
    assert clicked == ['button:has-text("Back to Tasks") || a:has-text("Back to Tasks")']


def test_validate_pre_submit_consistency_returns_blocking_result_when_live_dom_unavailable(monkeypatch):
    class DummyPage:
        url = "https://audit.atlascapture.io/tasks/room/normal/label/ep-consistency"

    source_segments = [
        {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "current_label": "pick up cup"},
        {"segment_index": 2, "start_sec": 4.0, "end_sec": 8.0, "current_label": "place cup on table"},
    ]
    segment_plan = {
        1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "label": "pick up cup"},
        2: {"segment_index": 2, "start_sec": 4.0, "end_sec": 8.0, "label": "place cup on table"},
    }

    def _boom(page_obj, cfg_obj, progress_callback=None):
        raise RuntimeError("Target page, context or browser has been closed")

    monkeypatch.setattr(segments, "extract_segments", _boom)

    result = consistency.validate_pre_submit_consistency(
        DummyPage(),
        {"run": {"desync_snapshot_tolerance_sec": 0.25}},
        segment_plan,
        source_segments,
    )

    assert result["consistent"] is False
    assert result["live_segments"] == []
    assert "live DOM unavailable during pre-submit consistency" in result["mismatches"][0]
    assert "target page, context or browser has been closed" in result["mismatches"][0].lower()
    assert result["desync_decision"]["ok"] is False
    assert result["live_snapshot"]["segment_count"] == 0
    assert result["source_snapshot"]["segment_count"] == 2


def test_resolve_rows_locator_ignores_debug_stdout_encoding_traps(monkeypatch):
    class _RowItem:
        def __init__(self, text):
            self._text = text

        def inner_text(self, timeout=None):
            return self._text

    class _RowCollection:
        def __init__(self, items):
            self._items = items

        def count(self):
            return len(self._items)

        def nth(self, idx):
            return self._items[idx]

    class _Page:
        def locator(self, selector):
            if selector == "candidate":
                return _RowCollection(
                    [
                        _RowItem("1\n0:00.0\n→\n0:06.3\n(6.3s)\npick up cloth"),
                        _RowItem("2\n0:06.3\n→\n0:13.8\n(7.5s)\nwipe switch"),
                    ]
                )
            return _RowCollection([])

        def inner_text(self, selector):
            return ""

    def _guarded_print(message):
        if str(message).startswith("[debug] sel="):
            raise UnicodeEncodeError("cp1252", "→", 0, 1, "character maps to <undefined>")
        return None

    monkeypatch.setattr(segments, "print", _guarded_print)
    monkeypatch.setattr(browser, "_safe_locator_text", lambda locator, timeout_ms=0: locator.inner_text(timeout=timeout_ms))

    selector, rows = segments._resolve_rows_locator(_Page(), "candidate", sample_size=2, row_text_timeout_ms=250)

    assert selector == "candidate"
    assert rows.count() == 2


def test_goto_task_room_prefers_tasks_root_labels_after_room_disabled(monkeypatch):
    label_url = "https://audit.atlascapture.io/tasks/room/normal/label/690f9dc5example"
    goto_calls = []
    click_selectors = []

    class _RoomDisabledThenTasksPage:
        def __init__(self):
            self.url = "https://audit.atlascapture.io/tasks/room/normal"

        def inner_text(self, selector):
            assert selector == "body"
            if self.url.endswith("/tasks/room/normal"):
                return "Rooms are unavailable\nRoom access is currently disabled."
            if self.url.endswith("/tasks"):
                return "Your Reserved Episodes\nLabel"
            return "Label Episode\nsegments"

        def wait_for_timeout(self, ms):
            return None

    page = _RoomDisabledThenTasksPage()
    cfg = {
        "atlas": {
            "room_url": "https://audit.atlascapture.io/tasks/room/normal",
            "dashboard_url": "https://audit.atlascapture.io/dashboard",
            "wait_before_continue_sec": 0,
            "selectors": {
                "tasks_nav": "tasks-nav",
                "enter_workflow_button": "enter-workflow",
                "continue_room_button": "continue-room",
                "label_button": "label-button",
                "label_task_link": 'a[href*="/tasks/room/normal/label/"]',
                "confirm_reserve_button": "confirm-reserve",
            },
        }
    }

    def _fake_recover(page_obj, cfg_obj, timeout_ms=2500):
        page_obj.url = "https://audit.atlascapture.io/tasks"
        return True

    def _fake_all_hrefs(page_obj):
        if page_obj.url.endswith("/tasks"):
            return [label_url]
        return []

    def _fake_goto(page_obj, url, wait_until="domcontentloaded", timeout_ms=45000, cfg=None, reason=""):
        goto_calls.append((url, reason))
        page_obj.url = url
        return True

    monkeypatch.setattr(browser, "_recover_room_access_disabled", _fake_recover)
    monkeypatch.setattr(browser, "_all_task_label_hrefs_from_page", _fake_all_hrefs)
    monkeypatch.setattr(browser, "_goto_with_retry", _fake_goto)
    monkeypatch.setattr(browser, "_is_label_page_actionable", lambda page_obj, cfg_obj, timeout_ms=0: "/tasks/room/normal/label/" in page_obj.url)
    monkeypatch.setattr(browser, "_dismiss_blocking_modals", lambda page_obj, cfg=None: None)
    monkeypatch.setattr(browser, "_safe_locator_click", lambda page_obj, selector, timeout_ms=0: click_selectors.append(selector) or False)

    status = {}
    opened = browser.goto_task_room(page, cfg, status_out=status)

    assert opened is True
    assert page.url == label_url
    assert goto_calls[-1][0] == label_url
    assert status["room_access_disabled"] is True
    assert "enter-workflow" not in click_selectors


def test_room_has_no_reserved_episodes_detects_tasks_root_reserve_card(monkeypatch):
    class _TasksReservePage:
        url = "https://audit.atlascapture.io/tasks"

        def inner_text(self, selector):
            assert selector == "body"
            return (
                "No Episodes Reserved\n"
                "Reserve a batch of episodes to start labeling.\n"
                "You'll reserve: Tier 3 episodes\n"
                "Reserve 3 Episodes"
            )

    monkeypatch.setattr(browser, "_all_task_label_hrefs_from_page", lambda page_obj: [])
    monkeypatch.setattr(browser, "_first_visible_locator", lambda page_obj, selector, timeout_ms=0: object())

    cfg = {
        "run": {"reserve_no_reserved_probe_timeout_ms": 250},
        "atlas": {"selectors": {"reserve_episodes_button": "reserve-button"}},
    }

    assert browser._room_has_no_reserved_episodes(_TasksReservePage(), cfg) is True


def test_page_has_reserve_cta_detects_tasks_root_reserve_card(monkeypatch):
    class _TasksReservePage:
        url = "https://audit.atlascapture.io/tasks"

        def inner_text(self, selector):
            assert selector == "body"
            return (
                "No Episodes Reserved\n"
                "Reserve a batch of episodes to start labeling.\n"
                "You'll reserve: Tier 3 episodes\n"
                "Reserve 3 Episodes"
            )

    monkeypatch.setattr(browser, "_all_task_label_hrefs_from_page", lambda page_obj: [])
    monkeypatch.setattr(browser, "_first_visible_locator", lambda page_obj, selector, timeout_ms=0: object())

    cfg = {
        "run": {"reserve_no_reserved_probe_timeout_ms": 250},
        "atlas": {"selectors": {"reserve_episodes_button": "reserve-button"}},
    }

    assert browser._page_has_reserve_cta(_TasksReservePage(), cfg) is True


def test_click_reserve_button_dynamic_uses_js_fallback_when_locator_missing():
    class _EmptyLocator:
        def count(self):
            return 0

    class _ReserveJsPage:
        def locator(self, selector):
            return _EmptyLocator()

        def evaluate(self, script):
            return {"clicked": True, "text": "Reserve 3 Episodes"}

    clicked, label = browser._click_reserve_button_dynamic(
        _ReserveJsPage(),
        {"atlas": {"selectors": {"reserve_episodes_button": "reserve-button"}}},
        timeout_ms=10,
    )

    assert clicked is True
    assert label == "Reserve 3 Episodes"


def test_release_all_reserved_episodes_uses_confirm_release_button(monkeypatch):
    clicked_selectors = []

    class _ReleasePage:
        url = "https://audit.atlascapture.io/tasks"

        def wait_for_timeout(self, ms):
            return None

    def _fake_click(page_obj, selector, timeout_ms=0):
        clicked_selectors.append(selector)
        return selector in {"release-btn", "confirm-release"}

    monkeypatch.setattr(browser, "_goto_with_retry", lambda *args, **kwargs: True)
    monkeypatch.setattr(browser, "_dismiss_blocking_modals", lambda page_obj, cfg=None: None)
    monkeypatch.setattr(browser, "_wait_for_any", lambda page_obj, selector, timeout_ms=0: False)
    monkeypatch.setattr(browser, "_safe_locator_click", _fake_click)

    cfg = {
        "atlas": {
            "room_url": "https://audit.atlascapture.io/tasks",
            "selectors": {
                "release_all_button": "release-btn",
                "confirm_release_button": "confirm-release",
            },
        }
    }

    assert browser._release_all_reserved_episodes(_ReleasePage(), cfg) is True
    assert "release-btn" in clicked_selectors
    assert "confirm-release" in clicked_selectors


def test_goto_task_room_reserves_from_tasks_root_when_no_reserved(monkeypatch):
    label_url = "https://audit.atlascapture.io/tasks/room/normal/label/690f9f6dreserve"
    goto_calls = []
    clicked_selectors = []

    class _NoReservedTasksPage:
        def __init__(self):
            self.url = "https://audit.atlascapture.io/tasks"
            self.reserved = False

        def inner_text(self, selector):
            assert selector == "body"
            if not self.reserved:
                return (
                    "No Episodes Reserved\n"
                    "Reserve a batch of episodes to start labeling.\n"
                    "You'll reserve: Tier 3 episodes\n"
                    "Reserve 3 Episodes"
                )
            if self.url.endswith("/tasks"):
                return "Your Reserved Episodes\nLabel"
            return "Label Episode\nsegments"

        def wait_for_timeout(self, ms):
            return None

    page = _NoReservedTasksPage()
    cfg = {
        "run": {
            "reserve_attempts_per_visit": 1,
            "reserve_refresh_after_click": True,
            "reserve_wait_only_on_rate_limit": True,
            "reserve_immediate_when_no_reserved": True,
            "reserve_skip_initial_label_scan_when_no_reserved": True,
        },
        "atlas": {
            "room_url": "https://audit.atlascapture.io/tasks",
            "dashboard_url": "https://audit.atlascapture.io/dashboard",
            "wait_before_continue_sec": 0,
            "selectors": {
                "tasks_nav": "tasks-nav",
                "enter_workflow_button": "enter-workflow",
                "continue_room_button": "continue-room",
                "label_button": "label-button",
                "label_task_link": 'a[href*="/tasks/room/normal/label/"]',
                "confirm_reserve_button": "confirm-reserve",
                "reserve_episodes_button": "reserve-button",
            },
        },
    }

    def _fake_all_hrefs(page_obj):
        if page_obj.reserved and page_obj.url.endswith("/tasks"):
            return [label_url]
        return []

    def _fake_goto(page_obj, url, wait_until="domcontentloaded", timeout_ms=45000, cfg=None, reason=""):
        goto_calls.append((url, reason))
        page_obj.url = url
        return True

    def _fake_click_reserve(page_obj, cfg_obj, timeout_ms=2500):
        page_obj.reserved = True
        return True, "Reserve 3 Episodes"

    monkeypatch.setattr(browser, "_all_task_label_hrefs_from_page", _fake_all_hrefs)
    monkeypatch.setattr(browser, "_goto_with_retry", _fake_goto)
    monkeypatch.setattr(browser, "_click_reserve_button_dynamic", _fake_click_reserve)
    monkeypatch.setattr(browser, "_room_has_no_reserved_episodes", lambda page_obj, cfg_obj: not page_obj.reserved)
    monkeypatch.setattr(browser, "_is_label_page_actionable", lambda page_obj, cfg_obj, timeout_ms=0: "/tasks/room/normal/label/" in page_obj.url)
    monkeypatch.setattr(browser, "_dismiss_blocking_modals", lambda page_obj, cfg=None: None)
    monkeypatch.setattr(browser, "_wait_for_any", lambda page_obj, selector, timeout_ms=0: True)
    monkeypatch.setattr(
        browser,
        "_safe_locator_click",
        lambda page_obj, selector, timeout_ms=0: clicked_selectors.append(selector) or (selector == "confirm-reserve"),
    )

    status = {}
    opened = browser.goto_task_room(page, cfg, status_out=status)

    assert opened is True
    assert page.reserved is True
    assert page.url == label_url
    assert clicked_selectors.count("confirm-reserve") == 1
    assert goto_calls[-1][0] == label_url
    assert status["no_reserved_episodes"] is True


def test_goto_task_room_reserves_from_visible_cta_even_when_no_reserved_probe_misses(monkeypatch):
    label_url = "https://audit.atlascapture.io/tasks/room/normal/label/690f9f6dreserve-visible"
    goto_calls = []
    clicked_selectors = []

    class _ReserveVisiblePage:
        def __init__(self):
            self.url = "https://audit.atlascapture.io/tasks"
            self.reserved = False

        def inner_text(self, selector):
            assert selector == "body"
            if not self.reserved:
                return "Reserve 3 Episodes\nTier 3 episodes"
            if self.url.endswith("/tasks"):
                return "Your Reserved Episodes\nLabel"
            return "Label Episode\nsegments"

        def wait_for_timeout(self, ms):
            return None

    page = _ReserveVisiblePage()
    cfg = {
        "run": {
            "reserve_attempts_per_visit": 1,
            "reserve_refresh_after_click": True,
            "reserve_wait_only_on_rate_limit": True,
            "reserve_immediate_when_no_reserved": True,
            "reserve_skip_initial_label_scan_when_no_reserved": True,
        },
        "atlas": {
            "room_url": "https://audit.atlascapture.io/tasks/room/normal",
            "dashboard_url": "https://audit.atlascapture.io/dashboard",
            "wait_before_continue_sec": 0,
            "selectors": {
                "tasks_nav": "tasks-nav",
                "enter_workflow_button": "enter-workflow",
                "continue_room_button": "continue-room",
                "label_button": "label-button",
                "label_task_link": 'a[href*="/tasks/room/normal/label/"]',
                "confirm_reserve_button": "confirm-reserve",
                "reserve_episodes_button": "reserve-button",
            },
        },
    }

    def _fake_all_hrefs(page_obj):
        if page_obj.reserved and page_obj.url.endswith("/tasks"):
            return [label_url]
        return []

    def _fake_goto(page_obj, url, wait_until="domcontentloaded", timeout_ms=45000, cfg=None, reason=""):
        goto_calls.append((url, reason))
        page_obj.url = url
        return True

    def _fake_click_reserve(page_obj, cfg_obj, timeout_ms=2500):
        page_obj.reserved = True
        return True, "Reserve 3 Episodes"

    monkeypatch.setattr(browser, "_all_task_label_hrefs_from_page", _fake_all_hrefs)
    monkeypatch.setattr(browser, "_goto_with_retry", _fake_goto)
    monkeypatch.setattr(browser, "_click_reserve_button_dynamic", _fake_click_reserve)
    monkeypatch.setattr(browser, "_room_has_no_reserved_episodes", lambda page_obj, cfg_obj: False)
    monkeypatch.setattr(browser, "_page_has_reserve_cta", lambda page_obj, cfg_obj, timeout_ms=0: not page_obj.reserved)
    monkeypatch.setattr(browser, "_is_label_page_actionable", lambda page_obj, cfg_obj, timeout_ms=0: "/tasks/room/normal/label/" in page_obj.url)
    monkeypatch.setattr(browser, "_dismiss_blocking_modals", lambda page_obj, cfg=None: None)
    monkeypatch.setattr(browser, "_wait_for_any", lambda page_obj, selector, timeout_ms=0: True)
    monkeypatch.setattr(
        browser,
        "_safe_locator_click",
        lambda page_obj, selector, timeout_ms=0: clicked_selectors.append(selector) or (selector == "confirm-reserve"),
    )

    status = {}
    opened = browser.goto_task_room(page, cfg, status_out=status)

    assert opened is True
    assert page.reserved is True
    assert page.url == label_url
    assert "enter-workflow" not in clicked_selectors
    assert any(reason == "tasks-refresh-after-reserve" for _, reason in goto_calls)
    assert goto_calls[-1][0] == label_url
    assert status["no_reserved_episodes"] is True


def test_goto_task_room_prefers_late_label_links_over_reserve_cta(monkeypatch):
    label_url = "https://audit.atlascapture.io/tasks/room/normal/label/late-hydrate-task"
    goto_calls = []

    class _LateHydratePage:
        def __init__(self):
            self.url = "https://audit.atlascapture.io/tasks"

        def inner_text(self, selector):
            assert selector == "body"
            if self.url.endswith("/tasks"):
                return "Your Reserved Episodes\nLabel"
            return "Label Episode\nsegments"

        def wait_for_timeout(self, ms):
            return None

    page = _LateHydratePage()
    cfg = {
        "run": {
            "reserve_attempts_per_visit": 1,
            "reserve_immediate_when_no_reserved": True,
            "reserve_skip_initial_label_scan_when_no_reserved": True,
        },
        "atlas": {
            "room_url": "https://audit.atlascapture.io/tasks/room/normal",
            "dashboard_url": "https://audit.atlascapture.io/dashboard",
            "wait_before_continue_sec": 0,
            "selectors": {
                "tasks_nav": "tasks-nav",
                "enter_workflow_button": "enter-workflow",
                "continue_room_button": "continue-room",
                "label_button": "label-button",
                "label_task_link": 'a[href*="/tasks/room/normal/label/"]',
                "confirm_reserve_button": "confirm-reserve",
                "reserve_episodes_button": "reserve-button",
            },
        },
    }

    scan_state = {"count": 0}

    def _fake_all_hrefs(page_obj):
        scan_state["count"] += 1
        if scan_state["count"] == 1:
            return []
        return [label_url]

    def _fake_goto(page_obj, url, wait_until="domcontentloaded", timeout_ms=45000, cfg=None, reason=""):
        goto_calls.append((url, reason))
        page_obj.url = url
        return True

    monkeypatch.setattr(browser, "_all_task_label_hrefs_from_page", _fake_all_hrefs)
    monkeypatch.setattr(browser, "_goto_with_retry", _fake_goto)
    monkeypatch.setattr(browser, "_page_has_reserve_cta", lambda page_obj, cfg_obj, timeout_ms=0: True)
    monkeypatch.setattr(browser, "_room_has_no_reserved_episodes", lambda page_obj, cfg_obj: False)
    monkeypatch.setattr(browser, "_is_label_page_actionable", lambda page_obj, cfg_obj, timeout_ms=0: "/tasks/room/normal/label/" in page_obj.url)
    monkeypatch.setattr(browser, "_dismiss_blocking_modals", lambda page_obj, cfg=None: None)
    monkeypatch.setattr(browser, "_wait_for_any", lambda page_obj, selector, timeout_ms=0: True)
    monkeypatch.setattr(
        browser,
        "_click_reserve_button_dynamic",
        lambda page_obj, cfg_obj, timeout_ms=2500: (_ for _ in ()).throw(AssertionError("reserve flow should not run")),
    )

    opened = browser.goto_task_room(page, cfg, status_out={})

    assert opened is True
    assert page.url == label_url
    assert goto_calls[-1][0] == label_url
    assert scan_state["count"] >= 2


def test_goto_task_room_recovers_tasks_root_labels_after_reserve_refresh_hits_disabled_room(monkeypatch):
    label_url = "https://audit.atlascapture.io/tasks/room/normal/label/690fa5b9796dcf9fcc8fd20c"
    goto_calls = []
    clicked_selectors = []

    class _ReserveThenDisabledRoomPage:
        def __init__(self):
            self.url = "https://audit.atlascapture.io/tasks"
            self.reserved = False

        def inner_text(self, selector):
            assert selector == "body"
            if self.url.endswith("/tasks/room/normal"):
                return "Rooms are unavailable\nRoom access is currently disabled."
            if self.reserved and self.url.endswith("/tasks"):
                return "Your Reserved Episodes\nLabel"
            return (
                "No Episodes Reserved\n"
                "Reserve a batch of episodes to start labeling.\n"
                "You'll reserve: Tier 3 episodes\n"
                "Reserve 3 Episodes"
            )

        def wait_for_timeout(self, ms):
            return None

    page = _ReserveThenDisabledRoomPage()
    cfg = {
        "run": {
            "reserve_attempts_per_visit": 1,
            "reserve_refresh_after_click": True,
            "reserve_wait_only_on_rate_limit": True,
            "reserve_immediate_when_no_reserved": True,
            "reserve_skip_initial_label_scan_when_no_reserved": True,
        },
        "atlas": {
            "room_url": "https://audit.atlascapture.io/tasks/room/normal",
            "dashboard_url": "https://audit.atlascapture.io/dashboard",
            "wait_before_continue_sec": 0,
            "selectors": {
                "tasks_nav": "tasks-nav",
                "enter_workflow_button": "enter-workflow",
                "continue_room_button": "continue-room",
                "label_button": "label-button",
                "label_task_link": 'a[href*="/tasks/room/normal/label/"]',
                "confirm_reserve_button": "confirm-reserve",
                "reserve_episodes_button": "reserve-button",
            },
        },
    }

    def _fake_all_hrefs(page_obj):
        if page_obj.reserved and page_obj.url.endswith("/tasks"):
            return [label_url]
        return []

    def _fake_goto(page_obj, url, wait_until="domcontentloaded", timeout_ms=45000, cfg=None, reason=""):
        goto_calls.append((url, reason))
        page_obj.url = url
        return True

    def _fake_click_reserve(page_obj, cfg_obj, timeout_ms=2500):
        page_obj.reserved = True
        return True, "Reserve 3 Episodes"

    def _fake_recover(page_obj, cfg_obj, timeout_ms=2500):
        page_obj.url = "https://audit.atlascapture.io/tasks"
        return True

    monkeypatch.setattr(browser, "_all_task_label_hrefs_from_page", _fake_all_hrefs)
    monkeypatch.setattr(browser, "_goto_with_retry", _fake_goto)
    monkeypatch.setattr(browser, "_click_reserve_button_dynamic", _fake_click_reserve)
    monkeypatch.setattr(browser, "_recover_room_access_disabled", _fake_recover)
    monkeypatch.setattr(browser, "_room_has_no_reserved_episodes", lambda page_obj, cfg_obj: not page_obj.reserved)
    monkeypatch.setattr(browser, "_is_label_page_actionable", lambda page_obj, cfg_obj, timeout_ms=0: "/tasks/room/normal/label/" in page_obj.url)
    monkeypatch.setattr(browser, "_dismiss_blocking_modals", lambda page_obj, cfg=None: None)
    monkeypatch.setattr(browser, "_wait_for_any", lambda page_obj, selector, timeout_ms=0: True)
    monkeypatch.setattr(
        browser,
        "_safe_locator_click",
        lambda page_obj, selector, timeout_ms=0: clicked_selectors.append(selector) or (selector == "confirm-reserve"),
    )

    status = {}
    opened = browser.goto_task_room(page, cfg, status_out=status)

    assert opened is True
    assert page.reserved is True
    assert page.url == label_url
    assert clicked_selectors.count("confirm-reserve") == 1
    assert status["no_reserved_episodes"] is True
    assert status["room_access_disabled"] is True
    assert any(reason == "tasks-refresh-after-reserve" for _, reason in goto_calls)
    assert goto_calls[-1][0] == label_url


def test_goto_task_room_recovers_post_reserve_onboarding_back_to_tasks(monkeypatch):
    label_url = "https://audit.atlascapture.io/tasks/room/normal/label/690fa5b9796dcf9fcc8fd20d"
    goto_calls = []
    clicked_selectors = []

    class _ReserveThenOnboardingPage:
        def __init__(self):
            self.url = "https://audit.atlascapture.io/tasks"
            self.reserved = False
            self.onboarding = False

        def inner_text(self, selector):
            assert selector == "body"
            if not self.reserved:
                return (
                    "No Episodes Reserved\n"
                    "Reserve a batch of episodes to start labeling.\n"
                    "You'll reserve: Tier 3 episodes\n"
                    "Reserve 3 Episodes"
                )
            if self.onboarding:
                return (
                    "Welcome, Dana!\n"
                    "Your Journey\n"
                    "Complete Training\n"
                    "Do Labeling Tasks\n"
                    "Browse Tasks"
                )
            if self.url.endswith("/tasks"):
                return "Your Reserved Episodes\nLabel"
            return "Label Episode\nsegments"

        def wait_for_timeout(self, ms):
            return None

    page = _ReserveThenOnboardingPage()
    cfg = {
        "run": {
            "reserve_attempts_per_visit": 1,
            "reserve_refresh_after_click": True,
            "reserve_wait_only_on_rate_limit": True,
            "reserve_immediate_when_no_reserved": True,
            "reserve_skip_initial_label_scan_when_no_reserved": True,
        },
        "atlas": {
            "room_url": "https://audit.atlascapture.io/tasks/room/normal",
            "dashboard_url": "https://audit.atlascapture.io/dashboard",
            "wait_before_continue_sec": 0,
            "selectors": {
                "tasks_nav": "tasks-nav",
                "enter_workflow_button": "enter-workflow",
                "continue_room_button": "continue-room",
                "label_button": "label-button",
                "label_task_link": 'a[href*="/tasks/room/normal/label/"]',
                "confirm_reserve_button": "confirm-reserve",
                "reserve_episodes_button": "reserve-button",
            },
        },
    }

    def _fake_all_hrefs(page_obj):
        if page_obj.reserved and not page_obj.onboarding and page_obj.url.endswith("/tasks"):
            return [label_url]
        return []

    def _fake_goto(page_obj, url, wait_until="domcontentloaded", timeout_ms=45000, cfg=None, reason=""):
        goto_calls.append((url, reason))
        page_obj.url = url
        return True

    def _fake_click_reserve(page_obj, cfg_obj, timeout_ms=2500):
        page_obj.reserved = True
        page_obj.onboarding = True
        return True, "Reserve 3 Episodes"

    def _fake_safe_click(page_obj, selector, timeout_ms=0):
        clicked_selectors.append(selector)
        if selector == "confirm-reserve":
            return True
        if "Browse Tasks" in selector and page_obj.onboarding:
            page_obj.onboarding = False
            page_obj.url = "https://audit.atlascapture.io/tasks"
            return True
        return False

    monkeypatch.setattr(browser, "_all_task_label_hrefs_from_page", _fake_all_hrefs)
    monkeypatch.setattr(browser, "_goto_with_retry", _fake_goto)
    monkeypatch.setattr(browser, "_click_reserve_button_dynamic", _fake_click_reserve)
    monkeypatch.setattr(browser, "_room_has_no_reserved_episodes", lambda page_obj, cfg_obj: not page_obj.reserved)
    monkeypatch.setattr(browser, "_is_label_page_actionable", lambda page_obj, cfg_obj, timeout_ms=0: "/tasks/room/normal/label/" in page_obj.url)
    monkeypatch.setattr(browser, "_dismiss_blocking_modals", lambda page_obj, cfg=None: None)
    monkeypatch.setattr(browser, "_wait_for_any", lambda page_obj, selector, timeout_ms=0: True)
    monkeypatch.setattr(browser, "_safe_locator_click", _fake_safe_click)

    status = {}
    opened = browser.goto_task_room(page, cfg, status_out=status)

    assert opened is True
    assert page.reserved is True
    assert page.onboarding is False
    assert page.url == label_url
    assert clicked_selectors.count("confirm-reserve") == 1
    assert any("Browse Tasks" in selector for selector in clicked_selectors)
    assert any(reason == "tasks-refresh-after-reserve" for _, reason in goto_calls)
    assert goto_calls[-1][0] == label_url
    assert status["no_reserved_episodes"] is True
    assert status["empty_after_reserve"] is False


def test_goto_task_room_recovers_late_post_reserve_onboarding_before_empty(monkeypatch):
    label_url = "https://audit.atlascapture.io/tasks/room/normal/label/690fa5b9796dcf9fcc8fd20e"
    goto_calls = []
    clicked_selectors = []

    class _ReserveThenLateOnboardingPage:
        def __init__(self):
            self.url = "https://audit.atlascapture.io/tasks"
            self.reserved = False
            self.onboarding = False
            self.tasks_ready = False

        def inner_text(self, selector):
            assert selector == "body"
            if not self.reserved:
                return (
                    "No Episodes Reserved\n"
                    "Reserve a batch of episodes to start labeling.\n"
                    "You'll reserve: Tier 3 episodes\n"
                    "Reserve 3 Episodes"
                )
            if self.onboarding:
                return (
                    "Welcome, Dana!\n"
                    "Your Journey\n"
                    "Complete Training\n"
                    "Do Labeling Tasks\n"
                    "Browse Tasks"
                )
            if self.url.endswith("/tasks"):
                return "Your Reserved Episodes"
            return "Label Episode\nsegments"

        def wait_for_timeout(self, ms):
            return None

    page = _ReserveThenLateOnboardingPage()
    cfg = {
        "run": {
            "reserve_attempts_per_visit": 1,
            "reserve_refresh_after_click": True,
            "reserve_wait_only_on_rate_limit": True,
            "reserve_immediate_when_no_reserved": True,
            "reserve_skip_initial_label_scan_when_no_reserved": True,
        },
        "atlas": {
            "room_url": "https://audit.atlascapture.io/tasks/room/normal",
            "dashboard_url": "https://audit.atlascapture.io/dashboard",
            "wait_before_continue_sec": 0,
            "selectors": {
                "tasks_nav": "tasks-nav",
                "enter_workflow_button": "enter-workflow",
                "continue_room_button": "continue-room",
                "label_button": "label-button",
                "label_task_link": 'a[href*="/tasks/room/normal/label/"]',
                "confirm_reserve_button": "confirm-reserve",
                "reserve_episodes_button": "reserve-button",
            },
        },
    }

    def _fake_all_hrefs(page_obj):
        if page_obj.tasks_ready and page_obj.url.endswith("/tasks"):
            return [label_url]
        return []

    def _fake_goto(page_obj, url, wait_until="domcontentloaded", timeout_ms=45000, cfg=None, reason=""):
        goto_calls.append((url, reason))
        page_obj.url = url
        return True

    def _fake_click_reserve(page_obj, cfg_obj, timeout_ms=2500):
        page_obj.reserved = True
        return True, "Reserve 3 Episodes"

    def _fake_safe_click(page_obj, selector, timeout_ms=0):
        clicked_selectors.append(selector)
        if selector == "confirm-reserve":
            return True
        if "Browse Tasks" in selector and page_obj.onboarding:
            page_obj.onboarding = False
            page_obj.tasks_ready = True
            page_obj.url = "https://audit.atlascapture.io/tasks"
            return True
        return False

    def _fake_wait_for_any(page_obj, selector, timeout_ms=0):
        if page_obj.reserved and not page_obj.tasks_ready:
            page_obj.onboarding = True
        return False

    monkeypatch.setattr(browser, "_all_task_label_hrefs_from_page", _fake_all_hrefs)
    monkeypatch.setattr(browser, "_goto_with_retry", _fake_goto)
    monkeypatch.setattr(browser, "_click_reserve_button_dynamic", _fake_click_reserve)
    monkeypatch.setattr(browser, "_room_has_no_reserved_episodes", lambda page_obj, cfg_obj: not page_obj.reserved)
    monkeypatch.setattr(browser, "_is_label_page_actionable", lambda page_obj, cfg_obj, timeout_ms=0: "/tasks/room/normal/label/" in page_obj.url)
    monkeypatch.setattr(browser, "_dismiss_blocking_modals", lambda page_obj, cfg=None: None)
    monkeypatch.setattr(browser, "_wait_for_any", _fake_wait_for_any)
    monkeypatch.setattr(browser, "_safe_locator_click", _fake_safe_click)

    status = {}
    opened = browser.goto_task_room(page, cfg, status_out=status)

    assert opened is True
    assert page.reserved is True
    assert page.onboarding is False
    assert page.tasks_ready is True
    assert page.url == label_url
    assert clicked_selectors.count("confirm-reserve") == 1
    assert any("Browse Tasks" in selector for selector in clicked_selectors)
    assert any(reason == "tasks-refresh-after-reserve" for _, reason in goto_calls)
    assert goto_calls[-1][0] == label_url
    assert status["empty_after_reserve"] is False


def test_solver_config_load_config_reads_server_config_file():
    cfg = solver_config.load_config(Path("configs/config_hetzner_production.yaml"))
    assert cfg["browser"]["headless"] is True
    assert "run" in cfg
    assert "gemini" in cfg
