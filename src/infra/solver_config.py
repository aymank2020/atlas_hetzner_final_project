"""Configuration and selector-loading helpers for the Atlas solver."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_SCRIPT_BUILD = "2026-04-03.1335-refactor"


DEFAULT_CONFIG: Dict[str, Any] = {
    "browser": {
        "headless": False,
        "slow_mo_ms": 120,
        "storage_state_path": ".state/atlas_auth.json",
        "force_login": False,
        "restore_state_in_profile_mode": False,
        "use_chrome_profile": False,
        "proxy_server": "",
        "proxy_username": "",
        "proxy_password": "",
        "proxy_bypass": "",
        "clear_env_proxy_for_backend_requests": True,
        "chrome_channel": "chrome",
        "executable_path": "",
        "chrome_user_data_dir": "",
        "chrome_profile_directory": "Default",
        "fallback_to_isolated_context_on_profile_error": True,
        "profile_launch_timeout_ms": 30000,
        "close_chrome_before_profile_launch": False,
        "profile_launch_retry_count": 1,
        "profile_launch_retry_delay_sec": 2.0,
        "clone_chrome_profile_to_temp": True,
        "cloned_user_data_dir": ".state/chrome_user_data_clone",
        "reuse_existing_cloned_profile": True,
        "prefer_profile_with_atlas_cookies": True,
    },
    "run": {
        "dry_run": True,
        "max_segments": 0,
        "max_episodes_per_run": 5,
        "target_task_urls": [],
        "no_task_retry_count": 5,
        "no_task_retry_delay_sec": 5.0,
        "no_task_backoff_factor": 1.0,
        "no_task_max_delay_sec": 5.0,
        "clear_blocked_tasks_every_retry": True,
        "keep_alive_when_idle": True,
        "keep_alive_idle_cycle_pause_sec": 5.0,
        "skip_reserve_when_all_visible_blocked": False,
        "clear_blocked_tasks_after_all_visible_blocked_hits": 1,
        "reserve_cooldown_sec": 0,
        "reserve_min_interval_sec": 0,
        "reserve_wait_only_on_rate_limit": True,
        "reserve_attempts_per_visit": 3,
        "reserve_label_wait_timeout_ms": 12000,
        "reserve_label_wait_timeout_after_reserve_ms": 3500,
        "reserve_immediate_when_no_reserved": True,
        "reserve_skip_initial_label_scan_when_no_reserved": True,
        "reserve_no_reserved_probe_timeout_ms": 900,
        "reserve_refresh_after_click": True,
        "reserve_rate_limit_wait_sec": 5,
        "release_all_on_internal_error": True,
        "release_and_reserve_on_all_visible_blocked": True,
        "release_and_reserve_on_submit_unverified": True,
        "recycle_after_max_episodes": True,
        "release_all_after_batch": True,
        "release_all_wait_sec": 5,
        "goto_retry_count": 3,
        "goto_retry_delay_sec": 1.2,
        "skip_duplicate_task_in_run": True,
        "duplicate_task_retry_count": 3,
        "duplicate_task_retry_wait_sec": 2.0,
        "continue_on_episode_error": True,
        "max_episode_failures_per_run": 3,
        "episode_failure_retry_delay_sec": 4.0,
        "gemini_quota_retry_delay_sec": 15.0,
        "gemini_quota_global_pause_min_sec": 60.0,
        "gemini_quota_global_pause_step_sec": 60.0,
        "gemini_quota_task_block_max_wait_sec": 21600.0,
        "max_video_prepare_failures_per_task": 2,
        "max_gemini_failures_per_task": 1,
        "workflow_reentry_enter_clicks": 2,
        "workflow_reentry_second_click_delay_sec": 5.0,
        "min_delay_between_episodes_sec": 0.0,
        "max_delay_between_episodes_sec": 0.0,
        "reuse_cached_labels": True,
        "skip_unchanged_labels": True,
        "resume_from_artifacts": True,
        "resume_skip_video_steps_when_cached": True,
        "resume_skip_apply_steps_when_done": False,
        "allow_resume_auto_submit": False,
        "execute_force_fresh_gemini": True,
        "execute_force_live_segments": True,
        "execute_require_video_context": True,
        "use_episode_runtime_v2": False,
        "strict_single_chat_session": False,
        "force_episode_browser_isolation": False,
        "sticky_episode_resume": False,
        "disable_release_all_during_canary": False,
        "single_window_two_tabs": False,
        "single_window_single_tab": False,
        "hold_rule_context_neighbors": 2,
        "gemini_transport_max_retries": 3,
        "gemini_transport_max_retries_ops": 2,
        "gemini_scope_followup_attempts": 1,
        "gemini_schema_followup_attempts": 1,
        "targeted_repair_max_rounds": 3,
        "targeted_repair_scope_neighbors": 2,
        "structured_episode_reports": False,
        "live_validation_enabled": False,
        "desync_snapshot_tolerance_sec": 0.25,
        "primary_solve_backend": "api",
        "chat_only_mode": False,
        "chat_ops_enabled": True,
        "chat_ops_fail_open": True,
        "chat_ops_synthesize_split_fallback": True,
        "chat_ops_run_without_overlong": False,
        "chat_ops_timeout_sec": 300.0,
        "chat_only_policy_retry_enabled": False,
        "chat_labels_timeout_sec": 1200.0,
        "chat_request_watchdog_buffer_sec": 180.0,
        "chat_attach_watchdog_floor_sec": 300.0,
        "chat_dispatch_watchdog_floor_sec": 300.0,
        "chat_chunk_fallback_to_single_request": True,
        "segment_chunking_enabled": True,
        "segment_chunking_min_segments": 16,
        "segment_chunking_min_video_sec": 60.0,
        "segment_chunking_max_segments_per_request": 8,
        "segment_chunking_max_window_sec": 20.0,
        "segment_chunking_video_pad_sec": 1.0,
        "segment_chunking_keep_temp_files": False,
        "segment_chunking_include_previous_labels_context": True,
        "segment_chunking_max_previous_labels": 12,
        "segment_chunking_disable_operations": True,
        "segment_chunking_force_operations_on_overlong_segments": True,
        "segment_chunking_collect_split_operations_only": True,
        "segment_chunking_consistency_memory_enabled": True,
        "segment_chunking_consistency_memory_limit": 40,
        "segment_chunking_consistency_prompt_terms": 16,
        "segment_chunking_consistency_normalize_labels": True,
        "auto_continuity_merge_enabled": True,
        "auto_continuity_merge_min_run_segments": 3,
        "auto_continuity_merge_min_token_overlap": 1,
        "auto_continuity_merge_max_combined_duration_sec": 0.0,
        "use_task_scoped_artifacts": True,
        "capture_step_screenshots": False,
        "capture_step_screenshots_full_page": False,
        "capture_step_html": False,
        "capture_step_history_limit": 16,
        "enable_quality_review_submit": True,
        "submit_manual_watch_enabled": False,
        "submit_manual_watch_timeout_sec": 180.0,
        "submit_manual_watch_poll_ms": 500,
        "submit_manual_watch_log_interval_sec": 10.0,
        "quality_review_submit_settle_sec": 1.3,
        "loop_off_on_episode_open": True,
        "enable_policy_gate": True,
        "block_apply_on_validation_fail": True,
        "skip_policy_lexical_checks_on_unchanged_labels": False,
        "ignore_timestamp_policy_errors_when_adjust_disabled": True,
        "ignore_no_action_standalone_policy_error": True,
        "max_segment_duration_sec": 10.0,
        "pre_submit_chat_compare_enabled": False,
        "pre_submit_chat_compare_required": False,
        "pre_submit_chat_compare_model": "",
        "pre_submit_chat_compare_block_on_chat_failure": False,
        "pre_submit_chat_compare_block_when_chat_better": True,
        "pre_submit_chat_compare_auto_adopt_same_timeline": True,
        "pre_submit_chat_compare_auto_repair_split_only": False,
        "pre_submit_chat_compare_retry_on_missing_attachment": True,
        "pre_submit_chat_compare_max_wait_sec": 300.0,
        "pre_submit_chat_compare_same_timeline_epsilon_sec": 0.35,
        "policy_auto_split_repair_enabled": True,
        "policy_auto_split_repair_max_rounds": 3,
        "policy_auto_split_repair_max_segments_per_round": 3,
        "pre_submit_chat_compare_seed_discord_context": False,
        "pre_submit_chat_compare_discord_context_max_messages": 120,
        "pre_submit_chat_compare_discord_context_max_chars_per_message": 900,
        "major_step_pause_enabled": False,
        "major_step_pause_min_sec": 0.0,
        "major_step_pause_max_sec": 0.0,
        "no_action_pause_rewrite_enabled": True,
        "no_action_pause_rewrite_max_sec": 12.0,
        "no_action_pause_rewrite_min_overlap_tokens": 1,
        "no_action_pause_rewrite_prefer_next_adjust": True,
        "min_label_words": 2,
        "max_label_words": 20,
        "max_atomic_actions_per_label": 2,
        "forbidden_label_verbs": ["grasp", "clutch"],
        "forbidden_narrative_words": ["another", "then", "next", "continue", "again", "starts", "begins", "finishes", "ends"],
        "allowed_label_start_verbs": [
            "pick up",
            "place",
            "move",
            "adjust",
            "align",
            "hold",
            "cut",
            "open",
            "close",
            "peel",
            "secure",
            "wipe",
            "flip",
            "pull",
            "push",
            "insert",
            "remove",
            "attach",
            "detach",
            "connect",
            "disconnect",
            "tighten",
            "loosen",
            "screw",
            "unscrew",
            "press",
            "twist",
            "turn",
            "slide",
            "lift",
            "lower",
            "set",
            "position",
            "straighten",
            "comb",
            "detangle",
            "sand",
            "paint",
            "clean",
            "put",
            "put down",
            "stir",
            "mix",
            "blend",
            "pour",
            "squeeze",
            "fold",
            "unfold",
            "wrap",
            "unwrap",
            "tape",
            "untape",
            "tie",
            "untie",
            "knot",
            "lace",
            "thread",
            "sew",
            "stitch",
            "knit",
            "weave",
            "braid",
            "brush",
            "scrub",
            "sweep",
            "mop",
            "rinse",
            "wash",
            "dry",
            "iron",
            "spray",
            "apply",
            "spread",
            "rub",
            "dab",
            "tap",
            "pat",
            "shake",
            "roll",
            "unroll",
            "stack",
            "unstack",
            "sort",
            "arrange",
            "organize",
            "gather",
            "collect",
            "dump",
            "empty",
            "fill",
            "refill",
            "scoop",
            "ladle",
            "spoon",
            "fork",
            "scrape",
            "shave",
            "trim",
            "snip",
            "clip",
            "chop",
            "dice",
            "slice",
            "mince",
            "grate",
            "grind",
            "crush",
            "smash",
            "break",
            "tear",
            "rip",
            "crack",
            "snap",
            "bend",
            "stretch",
            "compress",
            "clamp",
            "crimp",
            "staple",
            "pin",
            "nail",
            "hammer",
            "drill",
            "bore",
            "file",
            "polish",
            "buff",
            "sharpen",
            "hone",
            "saw",
            "plane",
            "chisel",
            "carve",
            "engrave",
            "etch",
            "mark",
            "label",
            "write",
            "draw",
            "trace",
            "measure",
            "weigh",
            "test",
            "calibrate",
            "lock",
            "unlock",
            "latch",
            "hook",
            "unhook",
            "hang",
            "mount",
            "dismount",
            "load",
            "unload",
            "pack",
            "unpack",
            "seal",
            "unseal",
            "cap",
            "uncap",
            "plug",
            "unplug",
            "zip",
            "unzip",
            "button",
            "unbutton",
            "buckle",
            "unbuckle",
            "fasten",
            "unfasten",
            "release",
            "grip",
            "drop",
            "toss",
            "throw",
            "catch",
            "pass",
            "hand",
            "swap",
            "exchange",
            "replace",
            "transfer",
            "drag",
            "carry",
            "transport",
            "deliver",
            "feed",
            "water",
            "plant",
            "dig",
            "rake",
            "hoe",
            "prune",
            "harvest",
            "assemble",
            "disassemble",
            "build",
            "demolish",
            "repair",
            "fix",
            "patch",
            "glue",
            "weld",
            "solder",
            "crimp",
            "splice",
            "operate",
            "activate",
            "deactivate",
            "power",
            "start",
            "stop",
            "reset",
            "switch",
            "toggle",
            "spin",
            "wind",
            "unwind",
            "coil",
            "uncoil",
            "wrap",
            "dip",
            "submerge",
            "soak",
            "drain",
            "filter",
            "sift",
            "sieve",
            "strain",
            "knead",
            "flatten",
            "shape",
            "mold",
            "form",
            "smooth",
            "level",
            "balance",
            "center",
            "rotate",
            "pivot",
            "tilt",
            "lean",
            "prop",
            "support",
            "brace",
            "reinforce",
            "inspect",
            "examine",
            "check",
            "verify",
            "review",
            "scan",
            "read",
            "type",
            "click",
            "swipe",
            "scroll",
            "point",
            "aim",
            "direct",
            "guide",
            "steer",
            "navigate",
            "drive",
            "No Action",
        ],
        "tier3_label_rewrite": True,
        "enable_structural_actions": True,
        "structural_allow_split": True,
        "structural_allow_merge": True,
        "structural_allow_delete": False,
        "requery_after_structural_actions": True,
        "max_structural_operations": 12,
        "structural_skip_if_segments_ge": 40,
        "structural_skip_allow_merge": True,
        "structural_max_failures_per_episode": 4,
        "structural_wait_rows_delta_timeout_ms": 1800,
        "adjust_timestamps": False,
        "timestamp_adjust_mode": "off",
        "timestamp_skip_if_segments_ge": 24,
        "timestamp_click_timeout_ms": 350,
        "timestamp_click_pause_ms": 15,
        "timestamp_max_failures_per_episode": 10,
        "timestamp_max_total_clicks": 80,
        "timestamp_abort_on_first_failure": False,
        "timestamp_skip_disabled_buttons": True,
        "label_apply_progress_every": 5,
        "label_apply_max_total_sec": 600,
        "label_apply_dynamic_budget_enabled": True,
        "label_apply_dynamic_budget_floor_sec": 420.0,
        "label_apply_dynamic_budget_base_sec": 120.0,
        "label_apply_dynamic_budget_per_target_sec": 12.0,
        "label_apply_no_progress_timeout_sec": 90.0,
        "label_apply_max_consecutive_row_failures": 3,
        "label_apply_max_failures": 18,
        "label_apply_input_timeout_ms": 3000,
        "label_apply_save_timeout_ms": 1800,
        "label_apply_edit_click_timeout_ms": 900,
        "submit_guard_enabled": True,
        "submit_guard_max_failure_ratio": 0.25,
        "submit_guard_min_applied_ratio": 0.9,
        "submit_guard_block_on_budget_exceeded": True,
        "submit_manual_watch_enabled": False,
        "submit_manual_watch_timeout_sec": 180.0,
        "submit_manual_watch_poll_ms": 500,
        "submit_manual_watch_log_interval_sec": 10.0,
        "play_full_video_before_labeling": False,
        "play_full_video_max_wait_sec": 900,
        "segment_resolve_attempts": 24,
        "segment_resolve_retry_ms": 800,
        "segment_resolve_sample_size": 8,
        "segment_resolve_row_text_timeout_ms": 350,
        "segment_extract_row_text_timeout_ms": 350,
        "segment_extract_progress_every": 12,
        "segment_row_scroll_timeout_ms": 1200,
        "label_open_loading_max_checks": 5,
        "label_open_loading_wait_ms": 600,
        "modal_dismiss_passes": 2,
        "modal_dismiss_timeout_ms": 120,
        "modal_dismiss_post_click_wait_ms": 180,
        "output_dir": "outputs",
        "segments_dump": "atlas_segments_dump.json",
        "labels_dump": "atlas_labels_from_gemini.json",
        "prompt_dump": "atlas_prompt.txt",
        "video_dump": "atlas_task_video.mp4",
    },
    "atlas": {
        "login_url": "https://audit.atlascapture.io/login?redirect=%2F",
        "dashboard_url": "https://audit.atlascapture.io/dashboard",
        "room_url": "https://audit.atlascapture.io/tasks",
        "email": "",
        "auth_timeout_sec": 180,
        "wait_before_continue_sec": 5,
        "selectors": {
            "email_input": '#email || input#email || input[type="email"] || input[autocomplete="email"] || input[placeholder*="email" i]',
            "start_button": 'button[type="submit"] || button:has-text("Start") || button:has-text("Earning") || button:has-text("Begin") || button:has-text("Join") || button:has-text("Sign") || button:has-text("Log in") || button:has-text("Continue") || form >> button || button.bg-gradient-to-r',
            "otp_input": '#code || input#code || input[inputmode="numeric"] || input[placeholder="000000"] || input[placeholder*="code" i] || input[autocomplete="one-time-code"] || input[maxlength="6"]',
            "verify_button": 'button[type="submit"] || button:has-text("Verify") || button:has-text("Confirm") || button:has-text("Submit") || button:has-text("Continue") || form >> button',
            "tasks_nav": 'a[href*="/tasks"] || a:has-text("Tasks") || button:has-text("Tasks") || [data-testid*="tasks"] || nav >> a:has-text("Task")',
            "enter_workflow_button": 'button:has-text("Enter") || button:has-text("Standard Workflow") || button:has-text("Workflow") || text=/enter\\s+.*workflow/i || button:has-text("Start Workflow") || button:has-text("Begin Workflow") || button:has-text("Browse Tasks") || a:has-text("Browse Tasks")',
            "continue_room_button": 'button:has-text("Continue") || button:has-text("Room") || text=/continue\\s+to\\s+room/i || button:has-text("Proceed") || button:has-text("Go to Room")',
            "label_button": 'button:has-text("Label") || a:has-text("Label") || text=/\\blabel\\b/i || [role="tab"]:has-text("Label") || button:has-text("Annotate")',
            "label_task_link": 'a[href*="/tasks/room/normal/label/"] || a[href*="/label/"] || a[href*="/task/"]',
            "reserve_episodes_button": 'button:has-text("Reserve") || text=/reserve\\s+\\d+\\s+episode/i || text=/reserve\\s+new/i || button[class*="reserve" i] || [data-testid*="reserve"]',
            "confirm_reserve_button": 'div[role="dialog"] button:has-text("Understand") || div[role="dialog"] button:has-text("Confirm") || div[role="dialog"] button:has-text("OK") || div[role="dialog"] button:has-text("Yes") || div[role="dialog"] button:has-text("Accept") || div[role="dialog"] button:has-text("Agree") || button:has-text("I Understand")',
            "release_all_button": 'button:has-text("Release All") || button:has-text("Release all") || button:has-text("Release Episodes") || button:has-text("Release") || text=/release\\s*all/i || [data-testid*="release"]',
            "confirm_release_button": 'div[role="dialog"] button:has-text("Release") || div[role="dialog"] button:has-text("Confirm") || div[role="dialog"] button:has-text("Yes") || div[role="dialog"] button:has-text("OK")',
            "error_go_back_button": 'button:has-text("Go Back") || a:has-text("Go Back") || [role="button"]:has-text("Go Back") || button:has-text("Back") || a:has-text("Back") || button:has-text("Return")',
            "video_element": "video || [data-testid*=\"video\"] || video[src]",
            "video_source": "video source || video > source",
            "loop_toggle_button": 'button:has-text("Loop") || button[title*="loop" i] || button[title*="Toggle segment loop"] || button[aria-label*="loop" i] || text=/loop\\s*(on|off)/i',
            "complete_button": '[data-testid="complete"] || [data-testid="submit"] || button:has-text("Complete") || button:has-text("Finish") || button:has-text("Done") || button:has-text("Submit Task") || button[type="submit"]:visible',
            "quality_review_modal": 'div[role="dialog"]:has-text("Quality") || div[role="alertdialog"]:has-text("Quality") || div[role="dialog"]:has-text("Review") || div[role="alertdialog"]:has-text("Review") || div[role="dialog"]:has(input[type="checkbox"]) || div[role="alertdialog"]:has(input[type="checkbox"])',
            "quality_review_checkbox": 'input[type="checkbox"] || [role="checkbox"] || label:has-text("verify") || label:has-text("reviewed") || label:has-text("confirm") || label:has-text("I have")',
            "quality_review_submit_button": 'div[role="dialog"] button:has-text("Submit") || div[role="alertdialog"] button:has-text("Submit") || div[role="dialog"] button:has-text("Confirm") || div[role="alertdialog"] button:has-text("Confirm") || div[role="dialog"] button:has-text("Continue") || div[role="alertdialog"] button:has-text("Continue") || div[role="dialog"] button:has-text("Proceed") || div[role="alertdialog"] button:has-text("Proceed") || div[role="dialog"] button:has-text("Done") || div[role="alertdialog"] button:has-text("Done") || div[role="dialog"] button:has-text("Finish") || div[role="alertdialog"] button:has-text("Finish") || div[role="dialog"] button[type="submit"] || div[role="alertdialog"] button[type="submit"] || div[role="dialog"] button:has-text("Send") || div[role="alertdialog"] button:has-text("Send")',
            "blocking_side_panel": 'div[class*="fixed"][class*="right-4"][class*="z-50"][class*="slide-in-from-right"] || div[class*="fixed"][class*="right-4"][class*="z-50"][class*="shadow-2xl"] || div[class*="fixed"][class*="z-50"]:visible',
            "blocking_side_panel_close": 'button[aria-label*="close" i] || button[title*="close" i] || button:has-text("Close") || button:has-text("Dismiss") || button:has-text("Done") || button:has-text("Cancel") || [role="button"]:has-text("Close") || button:has(svg.lucide-x)',
            "segment_rows": "div.space-y-1.p-2 > div.rounded-lg.border.p-3 || [data-testid*=\"segment\"] || [data-cy*=\"segment\"] || .segment-row || .seg-item || [class*=\"segment\"][class*=\"row\"]",
            "segment_label": 'p[title*="Double-click"] || p[title*="click to edit" i] || p.text-sm.font-medium.cursor-text || [data-testid*="label"] || [data-cy*="label"] || .segment-label || [class*="label"]',
            "segment_start": 'span.font-mono.text-xs || [data-testid*="start"] || [data-cy*="start"]',
            "segment_end": 'span.font-mono.text-xs.px-1\\.5.py-0\\.5 || [data-testid*="end"] || [data-cy*="end"]',
            "segment_time_plus_button": 'button:has(svg.lucide-plus) || button[aria-label*="increase" i] || button[title*="plus" i]',
            "segment_time_minus_button": 'button:has(svg.lucide-minus) || button[aria-label*="decrease" i] || button[title*="minus" i]',
            "edit_button_in_row": 'button[title*="Edit"] || button:has-text("Edit") || [aria-label*="Edit"] || button:has(svg.lucide-pencil) || button:has(svg.lucide-edit)',
            "split_button_in_row": 'button[title*="Split"] || [aria-label*="Split"] || button:has(svg.lucide-scissors) || button:has-text("Split")',
            "delete_button_in_row": 'button[title*="Delete"] || [aria-label*="Delete"] || button:has(svg.lucide-trash) || button:has(svg.lucide-trash-2) || button:has-text("Delete") || button:has-text("Remove")',
            "merge_button_in_row": 'button[title*="Merge"] || [aria-label*="Merge"] || button:has(svg.lucide-git-merge) || button:has-text("Merge") || button:has-text("Combine")',
            "action_confirm_button": 'button:has-text("Confirm") || button:has-text("Yes") || button:has-text("Delete") || button:has-text("Merge") || button:has-text("Apply") || button:has-text("Continue") || button:has-text("OK")',
            "label_input": 'textarea || [contenteditable="true"] || input[type="text"] || [data-testid*="label-input"]',
            "save_button": 'button:has-text("Save") || button:has-text("Apply") || button:has-text("Done") || button:has-text("Submit") || button:has-text("OK") || button[type="submit"]',
        },
        "timestamp_step_sec": 0.1,
        "timestamp_max_clicks_per_segment": 30,
    },
    "otp": {
        "provider": "gmail_imap",
        "gmail_email": "",
        "gmail_app_password": "",
        "imap_host": "imap.gmail.com",
        "imap_port": 993,
        "mailbox": "[Gmail]/All Mail",
        "sender_hint": "",
        "subject_hint": "",
        "code_regex": "\\b(\\d{6})\\b",
        "timeout_sec": 120,
        "poll_interval_sec": 4,
        "max_messages": 25,
        "unseen_only": False,
        "lookback_sec": 300,
    },
    "economics": {
        "episode_expected_revenue_usd": 0.50,
        "target_cost_ratio": 0.15,
        "hard_cost_ratio": 0.20,
        "enforce_cost_guards": False,
    },
    "gemini": {
        "api_key": "",
        "fallback_api_key": "",
        "api_keys": [],
        "rotation_policy": "sticky",
        "prefer_fallback_key_as_primary": True,
        "quota_fallback_enabled": False,
        "quota_fallback_max_uses_per_run": 1,
        "retry_switch_key_on_503": True,
        "retry_switch_key_max_uses_per_request": 2,
        "retry_switch_key_cooldown_sec": 90.0,
        "auth_mode": "api_key",
        "model": "gemini-2.5-flash",
        "stage_models": {
            "labeling": "gemini-2.5-flash",
            "repair": "gemini-2.5-flash",
            "policy_retry": "gemini-2.5-flash",
            "compare_api": "gemini-3.1-pro-preview",
            "compare_chat": "gemini-3.1-pro-preview",
        },
        "gen3_fallback_models": ["gemini-3.1-pro-preview"],
        "retry_with_quota_fallback_model": False,
        "quota_fallback_model": "gemini-3.1-pro-preview",
        "quota_fallback_from_models": [],
        "policy_retry_model": "gemini-2.5-flash",
        "chat_ops_model": "",
        "chat_labels_model": "",
        "chat_ops_response_schema_enabled": True,
        "chat_labels_response_schema_enabled": True,
        "retry_with_stronger_model_on_policy_fail": False,
        "policy_retry_only_if_flash": True,
        "zero_quota_model_cooldown_sec": 21600.0,
        "system_instruction_file": "",
        "system_instruction_text": "",
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "candidate_count": 1,
        "max_retries": 3,
        "retry_on_quota_429": False,
        "quota_retry_default_wait_sec": 12.0,
        "quota_cooldown_max_wait_sec": 120.0,
        "retry_base_delay_sec": 2.0,
        "retry_jitter_sec": 0.8,
        "max_backoff_sec": 30.0,
        "price_input_per_million": 0.30,
        "price_output_per_million": 2.50,
        "usage_log_file": "gemini_usage.jsonl",
        "rate_limit_enabled": True,
        "rate_limit_requests_per_minute": 6,
        "rate_limit_window_sec": 60.0,
        "rate_limit_min_interval_sec": 10.5,
        "connect_timeout_sec": 30,
        "request_timeout_sec": 1200,
        "chat_web_timeout_sec": 1200,
        "attach_video": True,
        "require_video": False,
        "allow_text_only_fallback_on_network_error": True,
        "skip_video_when_segments_le": 0,
        "video_transport": "files_api",
        "files_api_fallback_to_inline": False,
        "file_ready_timeout_sec": 120,
        "file_ready_poll_sec": 2.0,
        "upload_request_timeout_sec": 180,
        "upload_chunk_bytes": 8388608,
        "upload_chunk_granularity_bytes": 8388608,
        "upload_chunk_max_retries": 5,
        "optimize_video_for_upload": True,
        "optimize_video_only_if_larger_mb": 8.0,
        "optimize_video_target_mb": 15.0,
        "optimize_video_prefer_ffmpeg": True,
        "optimize_video_target_fps": 10.0,
        "optimize_video_min_fps": 8.0,
        "optimize_video_min_width": 320,
        "optimize_video_min_short_side": 320,
        "optimize_video_scale_candidates": [0.75, 0.6, 0.5, 0.4, 0.33, 0.25, 0.2],
        "inline_retry_target_mb": [15.0, 10.0, 8.0, 6.0],
        "max_inline_video_mb": 20.0,
        "inline_read_bytes_max_mb": 8.0,
        "zero_quota_model_cache_max_entries": 50,
        "split_upload_enabled": True,
        "split_upload_only_if_larger_mb": 8.0,
        "split_upload_chunk_max_mb": 6.0,
        "split_upload_max_chunks": 4,
        "split_upload_reencode_on_copy_fail": True,
        "split_upload_inline_total_max_mb": 12.0,
        "reference_frames_enabled": True,
        "reference_frames_always": False,
        "reference_frame_attach_when_video_mb_le": 2.5,
        "reference_frame_count": 2,
        "reference_frame_positions": [0.2, 0.55, 0.85],
        "reference_frame_max_side": 960,
        "reference_frame_jpeg_quality": 82,
        "reference_frame_max_total_kb": 420,
        "video_download_timeout_sec": 180,
        "video_download_retries": 5,
        "video_download_chunk_bytes": 1048576,
        "video_download_retry_base_sec": 1.2,
        "video_download_use_playwright_fallback": True,
        "video_candidate_scan_attempts": 4,
        "video_candidate_scan_wait_ms": 1200,
        "validate_video_decode": True,
        "min_video_bytes": 500000,
        "extra_instructions": "",
        "chat_web_url": "https://gemini.google.com/app",
        "chat_web_connect_over_cdp_url": "",
        "chat_web_reuse_cdp_context": False,
        "chat_web_preserve_existing_thread": False,
        "chat_web_preserve_existing_thread_across_episodes": True,
        "chat_web_clean_thread_on_episode_start": True,
        "chat_web_clean_thread_per_episode": False,
        "chat_web_clean_thread_per_request": True,
        "chat_web_require_authenticated_session": False,
        "chat_web_chunk_thread_reset_after_n_chunks": 0,
        "chat_web_channel": "chrome",
        "chat_web_storage_state": ".state/gemini_chat_storage_state.json",
        "chat_web_user_data_dir": ".state/gemini_chat_user_data",
        "chat_web_ignore_automation": True,
        "chat_web_launch_args": ["--disable-blink-features=AutomationControlled"],
        "chat_web_prefer_drive_picker": False,
        "chat_web_drive_root_folder_url": "",
        "chat_web_clean_thread_fallback_enabled": False,
        "chat_web_memory_primer_file": "",
        "chat_web_memory_primer_text": "",
        "chat_web_seed_context_file": "",
        "chat_web_seed_context_text": "",
        "chat_web_seed_context_send_before_prompt": False,
        "chat_web_apply_stealth": False,
        "chat_web_headless": False,
        "chat_web_upload_settle_min_sec": 12.0,
        "chat_web_upload_settle_sec_per_100mb": 25.0,
        "chat_web_upload_settle_max_sec": 180.0,
    },
}

def _load_selectors_yaml(yaml_path: str = "selectors.yaml") -> Dict[str, str]:
    """Load selector overrides from external YAML file."""
    path = Path(yaml_path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[selectors] warning: failed to parse {yaml_path}: {exc}")
        return {}
    if not isinstance(data, dict):
        return {}
    selectors_section = data.get("selectors", data)
    if not isinstance(selectors_section, dict):
        return {}
    result: Dict[str, str] = {}
    for key, value in selectors_section.items():
        if isinstance(value, dict):
            strategies = value.get("strategies", [])
            if isinstance(strategies, list) and strategies:
                result[key] = " || ".join(str(item) for item in strategies)
        elif isinstance(value, str):
            result[key] = value
        elif isinstance(value, list):
            result[key] = " || ".join(str(item) for item in value)
    if result:
        print(f"[selectors] loaded {len(result)} selector overrides from {yaml_path}")
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _is_gen3_gemini_model_name(model_name: str) -> bool:
    normalized = str(model_name or "").strip().lower()
    return normalized.startswith("gemini-3")


def _normalize_gen3_fallback_models(raw: Any, *, primary_model: str = "") -> List[str]:
    values: List[str] = []
    if isinstance(raw, list):
        source_items = list(raw)
    else:
        raw_text = str(raw or "").strip()
        if not raw_text:
            source_items = []
        else:
            source_items = [part for part in re.split(r"[,\|;]+", raw_text)]

    normalized_primary = str(primary_model or "").strip().lower()
    seen: set[str] = set()
    for item in source_items:
        value = str(item or "").strip()
        if not value or not _is_gen3_gemini_model_name(value):
            continue
        lowered = value.lower()
        if normalized_primary and lowered == normalized_primary:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        values.append(value)
    return values


def _ordered_gen3_gemini_models(primary_model: str, raw_fallback_models: Any) -> List[str]:
    ordered: List[str] = []
    primary = str(primary_model or "").strip()
    if primary and _is_gen3_gemini_model_name(primary):
        ordered.append(primary)
    ordered.extend(_normalize_gen3_fallback_models(raw_fallback_models, primary_model=primary))
    return ordered


def _resolve_secret(explicit: str, env_names: list[str]) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    for name in env_names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _load_dotenv(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        out[key] = value
    return out


def _read_secret(name: str, dotenv: Dict[str, str]) -> str:
    env_value = os.environ.get(name, "").strip()
    if env_value:
        return env_value
    return str(dotenv.get(name, "")).strip()


PAID_GEMINI_POOL_ENV_NAMES: Tuple[str, ...] = (
    "GEMINI_API_KEYS_PAID_POOL",
    "GEMINI_API_KEYS_POOL",
)

PAID_GEMINI_PRIMARY_ENV_NAMES: Tuple[str, ...] = (
    "GEMINI_API_KEY_PAID_EPISODE_EVAL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)

PAID_GEMINI_FALLBACK_ENV_NAMES: Tuple[str, ...] = (
    "GEMINI_API_KEY_PAID_SECONDARY",
    "GEMINI_API_KEY_FALLBACK",
    "GOOGLE_API_KEY_FALLBACK",
    "GEMINI_API_KEY_SECONDARY",
    "GOOGLE_API_KEY_SECONDARY",
)

PAID_GEMINI_SINGLE_ENV_NAMES: Tuple[str, ...] = (
    "GEMINI_API_KEY_PAID_EPISODE_EVAL",
    "GEMINI_API_KEY_PAID_SECONDARY",
    "GEMINI_API_KEY",
    "GEMINI_API_KEY2",
    "GEMINI_API_KEY_FALLBACK",
    "GOOGLE_API_KEY",
    "GOOGLE_API_KEY_FALLBACK",
    "GEMINI_API_KEY_SECONDARY",
    "GOOGLE_API_KEY_SECONDARY",
)

FREE_GEMINI_POOL_ENV_NAMES: Tuple[str, ...] = (
    "GEMINI_API_KEYS_FREE_POOL",
)

FREE_GEMINI_SINGLE_ENV_NAMES: Tuple[str, ...] = (
    "GEMINI_API_KEY_FREE_OPS",
    "GEMINI_API_KEY2_FREE_OPS2",
    "GEMINI_API_KEY_FREE_FALLBACK",
    "GEMINI_API_KEY_FREE_FALLBACK2",
    "GEMINI_API_KEY_OPS",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


def _append_unique_keys(target: List[str], values: List[str]) -> None:
    for candidate in values:
        value = str(candidate or "").strip()
        if value and value not in target:
            target.append(value)


def _split_secret_csv(raw: str) -> List[str]:
    items: List[str] = []
    for part in str(raw or "").split(","):
        value = part.strip()
        if value and value not in items:
            items.append(value)
    return items


def _collect_secret_csv_keys(dotenv: Dict[str, str], env_names: Tuple[str, ...]) -> List[str]:
    keys: List[str] = []
    for env_name in env_names:
        _append_unique_keys(keys, _split_secret_csv(_read_secret(env_name, dotenv)))
    return keys


def _collect_secret_named_keys(dotenv: Dict[str, str], env_names: Tuple[str, ...]) -> List[str]:
    keys: List[str] = []
    for env_name in env_names:
        _append_unique_keys(keys, [_read_secret(env_name, dotenv)])
    return keys


def collect_unique_gemini_keys(
    *,
    dotenv: Optional[Dict[str, str]] = None,
    pool_env_names: Tuple[str, ...] = (),
    single_env_names: Tuple[str, ...] = (),
    explicit_values: Optional[List[str]] = None,
) -> List[str]:
    source = dotenv or {}
    keys: List[str] = []
    _append_unique_keys(keys, list(explicit_values or []))
    _append_unique_keys(keys, _collect_secret_csv_keys(source, pool_env_names))
    _append_unique_keys(keys, _collect_secret_named_keys(source, single_env_names))
    return keys


def choose_rotating_gemini_key(
    keys: List[str],
    *,
    cursor_name: str,
    state_dir: Optional[Path] = None,
) -> str:
    normalized: List[str] = []
    _append_unique_keys(normalized, keys)
    if not normalized:
        return ""
    if len(normalized) == 1:
        return normalized[0]

    cursor_root = state_dir or Path(".state")
    cursor_path = cursor_root / f"{cursor_name}.cursor"
    try:
        cursor_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return normalized[0]

    raw_cursor = ""
    try:
        raw_cursor = cursor_path.read_text(encoding="utf-8").strip()
    except Exception:
        raw_cursor = ""
    try:
        cursor_value = int(raw_cursor)
    except Exception:
        cursor_value = 0
    index = cursor_value % len(normalized)
    next_cursor = (index + 1) % len(normalized)
    try:
        cursor_path.write_text(str(next_cursor), encoding="utf-8")
    except Exception:
        pass
    return normalized[index]


class GeminiKeyPool:
    def __init__(
        self,
        explicit_key: str,
        fallback_key: str,
        dotenv: Dict[str, str],
        cfg_api_keys: Optional[List[str]] = None,
        rotation_policy: str = "sticky",
        pool_env_names: Tuple[str, ...] = PAID_GEMINI_POOL_ENV_NAMES,
        single_env_names: Tuple[str, ...] = PAID_GEMINI_SINGLE_ENV_NAMES,
    ):
        self.keys: List[str] = []
        self.current_index = 0
        self.request_cursor = 0
        self.rotation_policy = "sticky"
        self.key_cooldowns: Dict[str, float] = {}
        self.set_rotation_policy(rotation_policy)

        _append_unique_keys(self.keys, [str(key or "").strip() for key in (cfg_api_keys or [])])
        _append_unique_keys(self.keys, _collect_secret_csv_keys(dotenv, pool_env_names))
        _append_unique_keys(self.keys, [explicit_key, fallback_key])
        _append_unique_keys(self.keys, _collect_secret_named_keys(dotenv, single_env_names))

    def set_rotation_policy(self, value: str) -> None:
        policy = str(value or "").strip().lower()
        self.rotation_policy = policy if policy in {"sticky", "round_robin"} else "sticky"

    def get_current_key(self) -> str:
        if not self.keys:
            return ""
        return self.keys[self.current_index]

    def get_current_index(self) -> int:
        return self.current_index

    def _clear_expired_key_cooldowns(self) -> None:
        if not self.key_cooldowns:
            return
        now = time.time()
        expired = [key for key, until_ts in self.key_cooldowns.items() if until_ts <= now]
        for key in expired:
            self.key_cooldowns.pop(key, None)

    def is_key_temporarily_unavailable(self, key: str) -> bool:
        value = str(key or "").strip()
        if not value:
            return False
        self._clear_expired_key_cooldowns()
        return value in self.key_cooldowns

    def mark_key_temporarily_unavailable(self, key: str, cooldown_sec: float) -> None:
        value = str(key or "").strip()
        if not value or value not in self.keys:
            return
        cooldown_sec = max(0.0, float(cooldown_sec or 0.0))
        if cooldown_sec <= 0:
            return
        self.key_cooldowns[value] = max(self.key_cooldowns.get(value, 0.0), time.time() + cooldown_sec)

    def _find_next_available_index(self, start_index: int, allow_current: bool = True) -> Optional[int]:
        if not self.keys:
            return None
        self._clear_expired_key_cooldowns()
        total = len(self.keys)
        for offset in range(total):
            candidate_index = (start_index + offset) % total
            if not allow_current and candidate_index == self.current_index:
                continue
            candidate_key = self.keys[candidate_index]
            if candidate_key not in self.key_cooldowns:
                return candidate_index
        return None

    def begin_request(self) -> str:
        if not self.keys:
            return ""
        start_index = self.current_index
        if self.rotation_policy == "round_robin":
            start_index = self.request_cursor % len(self.keys)
        next_index = self._find_next_available_index(start_index, allow_current=True)
        if next_index is not None:
            self.current_index = next_index
        elif self.rotation_policy == "round_robin":
            self.current_index = self.request_cursor % len(self.keys)
        if self.rotation_policy == "round_robin":
            self.request_cursor = (self.current_index + 1) % len(self.keys)
        return self.get_current_key()

    def switch_to_next(self) -> bool:
        if len(self.keys) <= 1:
            return False
        next_index = self._find_next_available_index((self.current_index + 1) % len(self.keys), allow_current=False)
        if next_index is None:
            return False
        self.current_index = next_index
        if self.rotation_policy == "round_robin":
            self.request_cursor = (self.current_index + 1) % len(self.keys)
        return True

    def prioritize_key(self, key: str) -> None:
        value = str(key or "").strip()
        if not value or value not in self.keys:
            return
        self.keys.remove(value)
        self.keys.insert(0, value)
        self.current_index = 0
        if self.rotation_policy == "round_robin":
            self.request_cursor = 0

    def has_multiple_keys(self) -> bool:
        return len(self.keys) > 1


_global_solver_key_pool: Optional[GeminiKeyPool] = None
_global_solver_key_pool_signature: Optional[Tuple[Any, ...]] = None
_global_free_solver_key_pool: Optional[GeminiKeyPool] = None
_global_free_solver_key_pool_signature: Optional[Tuple[Any, ...]] = None


def _get_global_solver_key_pool(
    explicit_key: str,
    fallback_key: str,
    dotenv: Dict[str, str],
    cfg_api_keys: Optional[List[str]] = None,
    rotation_policy: str = "sticky",
) -> GeminiKeyPool:
    global _global_solver_key_pool, _global_solver_key_pool_signature

    signature = (
        tuple(str(item or "").strip() for item in (cfg_api_keys or []) if str(item or "").strip()),
        str(explicit_key or "").strip(),
        str(fallback_key or "").strip(),
        *tuple(_read_secret(name, dotenv) for name in PAID_GEMINI_POOL_ENV_NAMES),
        *tuple(_read_secret(name, dotenv) for name in PAID_GEMINI_SINGLE_ENV_NAMES),
        str(rotation_policy or "").strip().lower(),
    )
    if _global_solver_key_pool is None or signature != _global_solver_key_pool_signature:
        _global_solver_key_pool = GeminiKeyPool(
            explicit_key=explicit_key,
            fallback_key=fallback_key,
            dotenv=dotenv,
            cfg_api_keys=cfg_api_keys,
            rotation_policy=rotation_policy,
        )
        _global_solver_key_pool_signature = signature
    else:
        _global_solver_key_pool.set_rotation_policy(rotation_policy)
    return _global_solver_key_pool


def _get_global_free_solver_key_pool(
    dotenv: Dict[str, str],
    *,
    cfg_api_keys: Optional[List[str]] = None,
    rotation_policy: str = "sticky",
) -> GeminiKeyPool:
    global _global_free_solver_key_pool, _global_free_solver_key_pool_signature

    signature = (
        tuple(str(item or "").strip() for item in (cfg_api_keys or []) if str(item or "").strip()),
        *tuple(_read_secret(name, dotenv) for name in FREE_GEMINI_POOL_ENV_NAMES),
        *tuple(_read_secret(name, dotenv) for name in FREE_GEMINI_SINGLE_ENV_NAMES),
        str(rotation_policy or "").strip().lower(),
    )
    if _global_free_solver_key_pool is None or signature != _global_free_solver_key_pool_signature:
        _global_free_solver_key_pool = GeminiKeyPool(
            explicit_key="",
            fallback_key="",
            dotenv=dotenv,
            cfg_api_keys=cfg_api_keys,
            rotation_policy=rotation_policy,
            pool_env_names=FREE_GEMINI_POOL_ENV_NAMES,
            single_env_names=FREE_GEMINI_SINGLE_ENV_NAMES,
        )
        _global_free_solver_key_pool_signature = signature
    else:
        _global_free_solver_key_pool.set_rotation_policy(rotation_policy)
    return _global_free_solver_key_pool


def _resolve_gemini_key(explicit: str) -> str:
    value = _resolve_secret(explicit, list(PAID_GEMINI_PRIMARY_ENV_NAMES))
    if value:
        return value
    for env_name in PAID_GEMINI_POOL_ENV_NAMES:
        pool_value = os.environ.get(env_name, "").strip()
        if pool_value:
            keys = _split_secret_csv(pool_value)
            if keys:
                return keys[0]
    return ""


def _resolve_gemini_fallback_key(explicit: str) -> str:
    return _resolve_secret(explicit, list(PAID_GEMINI_FALLBACK_ENV_NAMES))


def _apply_global_gemini_video_policy(cfg: Dict[str, Any]) -> None:
    """
    Enforce safe, conservative Gemini/video defaults on top of account configs.
    """
    gemini_cfg = cfg.setdefault("gemini", {})
    if not isinstance(gemini_cfg, dict):
        cfg["gemini"] = {}
        gemini_cfg = cfg["gemini"]

    gemini_cfg.setdefault("video_inline_upload_enabled", True)
    gemini_cfg.setdefault("video_inline_prefer_small_segments", True)
    gemini_cfg.setdefault("video_max_total_mb", 45.0)
    gemini_cfg.setdefault("video_inline_small_segment_mb", 20.0)
    gemini_cfg.setdefault("video_inline_small_segment_max_sec", 45.0)
    gemini_cfg.setdefault("video_inline_segment_target_sec", 35.0)
    gemini_cfg.setdefault("video_inline_segment_overlap_sec", 1.0)
    gemini_cfg.setdefault("video_upload_transport", "inline")
    gemini_cfg.setdefault("video_upload_timeout_sec", 180.0)
    gemini_cfg.setdefault("video_upload_poll_interval_sec", 3.0)
    gemini_cfg.setdefault("video_upload_delete_after_use", True)
    gemini_cfg.setdefault("video_upload_max_retries", 4)
    gemini_cfg.setdefault("video_upload_retry_base_sec", 2.0)
    gemini_cfg.setdefault("files_api_enabled", True)
    gemini_cfg.setdefault("files_api_delete_after_use", True)
    gemini_cfg.setdefault("files_api_retry_base_sec", 2.0)
    gemini_cfg.setdefault("files_api_max_retries", 4)

    preferred_model = "gemini-3.1-pro-preview"
    legacy_model_values = {"", "gemini-2.5-pro", "gemini-3-pro-preview"}

    configured_model = str(gemini_cfg.get("model", "") or "").strip()
    if configured_model.lower() in legacy_model_values and configured_model != preferred_model:
        gemini_cfg["model"] = preferred_model

    policy_retry_model = str(gemini_cfg.get("policy_retry_model", "") or "").strip()
    if policy_retry_model.lower() in legacy_model_values and policy_retry_model != preferred_model:
        gemini_cfg["policy_retry_model"] = preferred_model

    quota_fallback_model = str(gemini_cfg.get("quota_fallback_model", "") or "").strip()
    if quota_fallback_model.lower() in legacy_model_values and quota_fallback_model != preferred_model:
        gemini_cfg["quota_fallback_model"] = preferred_model

    normalized_model = str(gemini_cfg.get("model", preferred_model) or preferred_model).strip() or preferred_model
    gemini_cfg["gen3_fallback_models"] = _normalize_gen3_fallback_models(
        gemini_cfg.get("gen3_fallback_models", [preferred_model]),
        primary_model=normalized_model,
    )
    normalized_quota_fallback_model = str(
        gemini_cfg.get("quota_fallback_model", preferred_model) or preferred_model
    ).strip() or preferred_model
    quota_fallback_from_models = gemini_cfg.get("quota_fallback_from_models", [])
    normalized_from_models = {
        str(item or "").strip().lower()
        for item in (quota_fallback_from_models if isinstance(quota_fallback_from_models, list) else [quota_fallback_from_models])
        if str(item or "").strip()
    }
    if (
        not normalized_from_models
        or normalized_from_models == {normalized_quota_fallback_model.lower()}
    ):
        gemini_cfg["quota_fallback_from_models"] = [normalized_model]
    if normalized_model.lower() == normalized_quota_fallback_model.lower():
        gemini_cfg["retry_with_quota_fallback_model"] = False

    if str(gemini_cfg.get("policy_retry_model", "") or "").strip().lower() == str(
        gemini_cfg.get("model", preferred_model) or preferred_model
    ).strip().lower():
        gemini_cfg["retry_with_stronger_model_on_policy_fail"] = False

    video_transport = str(gemini_cfg.get("video_transport", "files_api") or "").strip().lower() or "files_api"
    gemini_cfg["video_transport"] = video_transport
    gemini_cfg["files_api_fallback_to_inline"] = bool(
        gemini_cfg.get("files_api_fallback_to_inline", False)
    )

    video_cfg = cfg.setdefault("video", {})
    if not isinstance(video_cfg, dict):
        cfg["video"] = {}
        video_cfg = cfg["video"]
    video_cfg.setdefault("merge_adjacent_segments", True)
    video_cfg.setdefault("max_clip_sec", 60.0)
    video_cfg.setdefault("reencode_for_gemini", True)
    video_cfg.setdefault("target_fps", 15)
    video_cfg.setdefault("target_max_width", 960)
    video_cfg.setdefault("target_video_bitrate_kbps", 1400)
    video_cfg.setdefault("target_audio_bitrate_kbps", 96)
    video_cfg.setdefault("video_download_timeout_sec", 180)
    video_cfg.setdefault("video_download_retries", 5)
    video_cfg.setdefault("video_download_chunk_bytes", 1048576)
    video_cfg.setdefault("video_download_retry_base_sec", 1.2)
    video_cfg.setdefault("video_download_use_playwright_fallback", True)


def _apply_global_gemini_chat_policy(cfg: Dict[str, Any]) -> None:
    gemini_cfg = cfg.setdefault("gemini", {})
    if not isinstance(gemini_cfg, dict):
        cfg["gemini"] = {}
        gemini_cfg = cfg["gemini"]

    gemini_cfg.setdefault("chat_web_require_authenticated_session", False)
    gemini_cfg.setdefault("chat_web_clean_thread_per_episode", False)
    gemini_cfg.setdefault("chat_web_preserve_existing_thread_across_episodes", True)
    gemini_cfg.setdefault("chat_web_chunk_thread_reset_after_n_chunks", 0)


def _apply_global_run_policy(cfg: Dict[str, Any]) -> None:
    """
    Enforce safe run-level defaults that prevent known quality failures
    across older account YAML files.
    """
    run = cfg.setdefault("run", {})
    if not isinstance(run, dict):
        cfg["run"] = {}
        run = cfg["run"]

    run.setdefault("auto_continuity_merge_enabled", True)
    run.setdefault("auto_continuity_merge_min_run_segments", 3)
    run.setdefault("auto_continuity_merge_min_token_overlap", 1)
    run.setdefault("segment_chunking_min_video_sec", 60.0)
    run.setdefault("primary_solve_backend", "api")
    run.setdefault("chat_only_mode", False)
    run.setdefault("chat_ops_enabled", True)
    run.setdefault("chat_ops_fail_open", True)
    run.setdefault("chat_ops_run_without_overlong", False)
    run.setdefault("gemini_transport_max_retries_ops", 2)
    run.setdefault("gemini_scope_followup_attempts", 1)
    run.setdefault("chat_ops_timeout_sec", 300.0)
    run.setdefault("chat_only_policy_retry_enabled", False)
    run.setdefault("chat_labels_timeout_sec", 420.0)
    run.setdefault("chat_chunk_fallback_to_single_request", True)
    run.setdefault("sticky_episode_resume", False)
    run.setdefault("disable_release_all_during_canary", False)
    run.setdefault("single_window_two_tabs", False)
    run.setdefault("single_window_single_tab", False)
    run.setdefault("hold_rule_context_neighbors", 2)
    if bool(run.get("single_window_single_tab", False)):
        run["single_window_two_tabs"] = False
    run.setdefault("skip_reserve_when_all_visible_blocked", False)
    run.setdefault("clear_blocked_tasks_after_all_visible_blocked_hits", 1)
    run.setdefault("clear_blocked_tasks_every_retry", True)
    run.setdefault("reserve_cooldown_sec", 0)
    run.setdefault("reserve_min_interval_sec", 0)
    run.setdefault("reserve_wait_only_on_rate_limit", True)
    run["reserve_attempts_per_visit"] = max(3, int(run.get("reserve_attempts_per_visit", 3) or 3))
    run.setdefault("reserve_rate_limit_wait_sec", 5)
    run.setdefault("release_and_reserve_on_all_visible_blocked", True)
    run.setdefault("release_and_reserve_on_submit_unverified", True)
    run.setdefault("no_task_retry_delay_sec", 5.0)
    run.setdefault("no_task_backoff_factor", 1.0)
    run.setdefault("no_task_max_delay_sec", 5.0)
    run.setdefault("keep_alive_idle_cycle_pause_sec", 5.0)
    run.setdefault("release_all_wait_sec", 5.0)

    if bool(run.get("disable_release_all_during_canary", False)):
        run["release_all_on_internal_error"] = False
        run["release_all_after_batch"] = False
        run["release_and_reserve_on_all_visible_blocked"] = False
        run["release_and_reserve_on_submit_unverified"] = False
        run["recycle_after_max_episodes"] = False
        run["continue_on_episode_error"] = False
        run["keep_alive_when_idle"] = False

    auto_continuity_merge_enabled = bool(run.get("auto_continuity_merge_enabled", True))
    if auto_continuity_merge_enabled and not bool(run.get("structural_allow_merge", True)):
        run["structural_allow_merge"] = True
        print("[policy] run.structural_allow_merge forced ON for continuity safety.")
    chat_only_mode = bool(run.get("chat_only_mode", False))
    primary_backend = str(run.get("primary_solve_backend", "api") or "api").strip().lower()
    if chat_only_mode or primary_backend == "chat_web":
        run["chat_only_mode"] = True
        run["primary_solve_backend"] = "chat_web"
        run["pre_submit_chat_compare_enabled"] = False
        run["pre_submit_chat_compare_required"] = False


def _apply_global_atlas_policy(cfg: Dict[str, Any]) -> None:
    atlas_cfg = cfg.setdefault("atlas", {})
    if not isinstance(atlas_cfg, dict):
        cfg["atlas"] = {}
        atlas_cfg = cfg["atlas"]

    room_url = str(atlas_cfg.get("room_url", "") or "").strip()
    if room_url.rstrip("/").lower() == "https://audit.atlascapture.io/tasks/room/normal":
        atlas_cfg["room_url"] = "https://audit.atlascapture.io/tasks"


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be YAML object")
    cfg = _deep_merge(DEFAULT_CONFIG, raw)
    selector_overrides = _load_selectors_yaml(str(path.with_name("selectors.yaml")))
    if selector_overrides:
        atlas_cfg = cfg.setdefault("atlas", {})
        selectors_cfg = atlas_cfg.setdefault("selectors", {})
        if isinstance(selectors_cfg, dict):
            selectors_cfg.update(selector_overrides)
    _apply_global_atlas_policy(cfg)
    _apply_global_gemini_video_policy(cfg)
    _apply_global_gemini_chat_policy(cfg)
    _apply_global_run_policy(cfg)
    cfg_meta = cfg.setdefault("_meta", {})
    if isinstance(cfg_meta, dict):
        cfg_meta["config_path"] = str(path.resolve())
        cfg_meta["config_dir"] = str(path.resolve().parent)
    return cfg
__all__ = [
    "_SCRIPT_BUILD",
    "DEFAULT_CONFIG",
    "GeminiKeyPool",
    "_load_selectors_yaml",
    "_deep_merge",
    "_cfg_get",
    "_is_gen3_gemini_model_name",
    "_normalize_gen3_fallback_models",
    "_ordered_gen3_gemini_models",
    "_load_dotenv",
    "_read_secret",
    "_resolve_secret",
    "_get_global_free_solver_key_pool",
    "_get_global_solver_key_pool",
    "_resolve_gemini_key",
    "_resolve_gemini_fallback_key",
    "_apply_global_gemini_video_policy",
    "_apply_global_run_policy",
    "_apply_global_atlas_policy",
    "load_config",
]
