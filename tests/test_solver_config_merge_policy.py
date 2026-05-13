from __future__ import annotations

import tempfile
from pathlib import Path

from src.infra.solver_config import (
    _normalize_gen3_fallback_models,
    _ordered_gen3_gemini_models,
    load_config,
)


def test_load_config_respects_disabled_merge_when_auto_continuity_merge_disabled() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "cfg.yaml"
        cfg_path.write_text(
            "\n".join(
                [
                    "run:",
                    "  auto_continuity_merge_enabled: false",
                    "  structural_allow_merge: false",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        cfg = load_config(cfg_path)

    assert cfg["run"]["auto_continuity_merge_enabled"] is False
    assert cfg["run"]["structural_allow_merge"] is False


def test_normalize_gen3_fallback_models_keeps_only_unique_gen3_entries() -> None:
    models = _normalize_gen3_fallback_models(
        [
            "gemini-3.1-pro-preview",
            "gemini-3.1-pro-preview",
            "gemini-2.5-flash",
            " custom-model ",
            "gemini-3.1-flash-preview",
        ],
        primary_model="gemini-3.1-flash-lite-preview",
    )

    assert models == ["gemini-3.1-pro-preview", "gemini-3.1-flash-preview"]


def test_ordered_gen3_models_starts_with_primary_and_ignores_non_gen3() -> None:
    candidates = _ordered_gen3_gemini_models(
        "gemini-3.1-flash-lite-preview",
        ["gemini-2.5-pro", "gemini-3.1-pro-preview", "gemini-3.1-pro-preview"],
    )

    assert candidates == ["gemini-3.1-flash-lite-preview", "gemini-3.1-pro-preview"]


def test_load_config_keeps_stage_models_and_economics_for_hybrid_capped() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "cfg.yaml"
        cfg_path.write_text(
            "\n".join(
                [
                    "gemini:",
                    "  model: gemini-2.5-flash",
                    "  quota_fallback_enabled: true",
                    "  retry_with_quota_fallback_model: false",
                    "  quota_fallback_model: gemini-3.1-pro-preview",
                    "  stage_models:",
                    "    labeling: gemini-2.5-flash",
                    "    repair: gemini-2.5-flash",
                    "    policy_retry: gemini-2.5-flash",
                    "    compare_api: gemini-3.1-pro-preview",
                    "    compare_chat: gemini-3.1-pro-preview",
                    "run:",
                    "  pre_submit_chat_compare_model: gemini-3.1-pro-preview",
                    "economics:",
                    "  episode_expected_revenue_usd: 0.50",
                    "  target_cost_ratio: 0.15",
                    "  hard_cost_ratio: 0.20",
                    "  enforce_cost_guards: true",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        cfg = load_config(cfg_path)

    assert cfg["gemini"]["quota_fallback_enabled"] is True
    assert cfg["gemini"]["retry_with_quota_fallback_model"] is False
    assert cfg["gemini"]["quota_fallback_model"] == "gemini-3.1-pro-preview"
    assert cfg["gemini"]["stage_models"]["labeling"] == "gemini-2.5-flash"
    assert cfg["gemini"]["stage_models"]["compare_chat"] == "gemini-3.1-pro-preview"
    assert cfg["run"]["pre_submit_chat_compare_model"] == "gemini-3.1-pro-preview"
    assert cfg["economics"]["episode_expected_revenue_usd"] == 0.50
    assert cfg["economics"]["target_cost_ratio"] == 0.15
    assert cfg["economics"]["hard_cost_ratio"] == 0.20
    assert cfg["economics"]["enforce_cost_guards"] is True


def test_load_config_forces_compare_off_in_chat_only_mode() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "cfg.yaml"
        cfg_path.write_text(
            "\n".join(
                [
                    "run:",
                    "  chat_only_mode: true",
                    "  primary_solve_backend: chat_web",
                    "  pre_submit_chat_compare_enabled: true",
                    "  pre_submit_chat_compare_required: true",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        cfg = load_config(cfg_path)

    assert cfg["run"]["chat_only_mode"] is True
    assert cfg["run"]["primary_solve_backend"] == "chat_web"
    assert cfg["run"]["pre_submit_chat_compare_enabled"] is False
    assert cfg["run"]["pre_submit_chat_compare_required"] is False
