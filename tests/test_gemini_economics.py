from __future__ import annotations

from src.infra.gemini_economics import (
    build_episode_cost_updates,
    cost_guard_enforcement_enabled,
    estimate_cost_usd,
    estimate_minimum_episode_cost_usd,
    resolve_stage_model,
    would_exceed_ratio_cap,
)


def test_estimate_cost_usd_uses_model_specific_pro_pricing() -> None:
    cfg = {"gemini": {}}
    cost = estimate_cost_usd(
        cfg,
        "gemini-3.1-pro-preview",
        prompt_tokens=3994,
        output_tokens=178,
        total_tokens=4172,
    )
    assert round(cost, 6) == 0.010124


def test_estimate_cost_usd_uses_model_specific_flash_pricing() -> None:
    cfg = {"gemini": {}}
    cost = estimate_cost_usd(
        cfg,
        "gemini-2.5-flash",
        prompt_tokens=3994,
        output_tokens=178,
        total_tokens=4172,
    )
    assert round(cost, 6) == 0.001643


def test_resolve_stage_model_prefers_stage_specific_model() -> None:
    cfg = {
        "gemini": {
            "model": "gemini-2.5-flash",
            "stage_models": {
                "compare_chat": "gemini-3.1-pro-preview",
            },
        },
        "run": {"pre_submit_chat_compare_model": "gemini-3.1-pro-preview"},
    }
    assert resolve_stage_model(cfg, "labeling", "") == "gemini-2.5-flash"
    assert resolve_stage_model(cfg, "compare_chat", "") == "gemini-3.1-pro-preview"


def test_estimate_minimum_episode_cost_usd_reports_budget_state() -> None:
    cfg = {
        "gemini": {
            "model": "gemini-2.5-flash",
            "stage_models": {
                "labeling": "gemini-2.5-flash",
                "compare_chat": "gemini-3.1-pro-preview",
            },
        },
        "run": {
            "segment_chunking_min_segments": 16,
            "segment_chunking_max_segments_per_request": 8,
            "pre_submit_chat_compare_model": "gemini-3.1-pro-preview",
        },
        "economics": {
            "episode_expected_revenue_usd": 0.50,
            "target_cost_ratio": 0.15,
            "hard_cost_ratio": 0.20,
        },
    }
    summary = estimate_minimum_episode_cost_usd(cfg, 32)
    assert summary["minimum_labeling_requests"] == 4
    assert summary["minimum_labeling_model"] == "gemini-2.5-flash"
    assert summary["minimum_compare_model"] == "gemini-3.1-pro-preview"
    assert summary["episode_budget_state"] == "within_target"
    assert summary["economics_cost_guards_enabled"] is False


def test_build_episode_cost_updates_accumulates_and_tracks_key_class() -> None:
    cfg = {"economics": {"episode_expected_revenue_usd": 0.50, "target_cost_ratio": 0.15, "hard_cost_ratio": 0.20}}
    task_state = {}
    first = build_episode_cost_updates(
        cfg,
        task_state,
        stage_name="labeling",
        model_name="gemini-2.5-flash",
        cost_usd=0.008,
        key_class="free",
    )
    second = build_episode_cost_updates(
        cfg,
        first,
        stage_name="compare_chat",
        model_name="gemini-3.1-pro-preview",
        cost_usd=0.012,
        key_class="paid",
    )
    assert second["episode_cost_by_stage"]["labeling"] == 0.008
    assert second["episode_cost_by_stage"]["compare_chat"] == 0.012
    assert second["episode_estimated_cost_usd"] == 0.02
    assert second["episode_key_class_used"] == "paid"


def test_would_exceed_ratio_cap_uses_episode_total_cost() -> None:
    cfg = {"economics": {"episode_expected_revenue_usd": 0.50}}
    task_state = {"episode_estimated_cost_usd": 0.08}
    assert would_exceed_ratio_cap(cfg, task_state, additional_cost_usd=0.03, ratio_limit=0.20) is True
    assert would_exceed_ratio_cap(cfg, task_state, additional_cost_usd=0.01, ratio_limit=0.20) is False


def test_cost_guard_enforcement_enabled_defaults_false_and_accepts_true_strings() -> None:
    assert cost_guard_enforcement_enabled({}) is False
    assert cost_guard_enforcement_enabled({"economics": {"enforce_cost_guards": False}}) is False
    assert cost_guard_enforcement_enabled({"economics": {"enforce_cost_guards": "true"}}) is True
