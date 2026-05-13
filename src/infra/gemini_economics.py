from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from src.infra.solver_config import _cfg_get


_AVG_PROMPT_TOKENS_PER_REQUEST = 3994
_AVG_OUTPUT_TOKENS_PER_REQUEST = 178
_PRO_CONTEXT_THRESHOLD_TOKENS = 200_000

_MODEL_PRICE_TABLE: Dict[str, Dict[str, float]] = {
    "gemini-3.1-pro-preview": {
        "input_per_million": 2.0,
        "output_per_million": 12.0,
        "input_per_million_over_200k": 4.0,
        "output_per_million_over_200k": 18.0,
    },
    "gemini-2.5-flash": {
        "input_per_million": 0.30,
        "output_per_million": 2.50,
        "input_per_million_over_200k": 0.30,
        "output_per_million_over_200k": 2.50,
    },
}


def normalize_model_name(model_name: str) -> str:
    value = str(model_name or "").strip().lower()
    if value == "gemini-3-pro-preview":
        return "gemini-3.1-pro-preview"
    return value


def resolve_stage_model(cfg: Dict[str, Any], stage_name: str, fallback: str = "") -> str:
    stage = str(stage_name or "").strip().lower()
    configured = str(_cfg_get(cfg, f"gemini.stage_models.{stage}", "") or "").strip()
    if configured:
        return configured
    if stage == "policy_retry":
        configured = str(_cfg_get(cfg, "gemini.policy_retry_model", "") or "").strip()
        if configured:
            return configured
    if stage == "compare_chat":
        configured = str(_cfg_get(cfg, "run.pre_submit_chat_compare_model", "") or "").strip()
        if configured:
            return configured
    if stage == "compare_api":
        configured = str(_cfg_get(cfg, "run.pre_submit_chat_compare_model", "") or "").strip()
        if configured:
            return configured
    return str(fallback or _cfg_get(cfg, "gemini.model", "gemini-2.5-flash") or "gemini-2.5-flash").strip()


def resolve_model_prices(
    cfg: Dict[str, Any],
    model_name: str,
    *,
    total_tokens: int = 0,
) -> Tuple[float, float]:
    normalized = normalize_model_name(model_name)
    table = _cfg_get(cfg, "gemini.model_pricing", {})
    price_row = table.get(normalized, {}) if isinstance(table, dict) else {}
    if not isinstance(price_row, dict):
        price_row = {}
    defaults = _MODEL_PRICE_TABLE.get(normalized, {})
    over_200k = int(total_tokens or 0) > _PRO_CONTEXT_THRESHOLD_TOKENS
    if over_200k:
        in_price = float(
            price_row.get(
                "input_per_million_over_200k",
                defaults.get("input_per_million_over_200k", defaults.get("input_per_million", 0.30)),
            )
        )
        out_price = float(
            price_row.get(
                "output_per_million_over_200k",
                defaults.get("output_per_million_over_200k", defaults.get("output_per_million", 2.50)),
            )
        )
        return in_price, out_price

    in_price = float(
        price_row.get(
            "input_per_million",
            defaults.get("input_per_million", _cfg_get(cfg, "gemini.price_input_per_million", 0.30)),
        )
    )
    out_price = float(
        price_row.get(
            "output_per_million",
            defaults.get("output_per_million", _cfg_get(cfg, "gemini.price_output_per_million", 2.50)),
        )
    )
    return in_price, out_price


def estimate_cost_usd(
    cfg: Dict[str, Any],
    model_name: str,
    *,
    prompt_tokens: int,
    output_tokens: int,
    total_tokens: int = 0,
) -> float:
    in_price, out_price = resolve_model_prices(cfg, model_name, total_tokens=total_tokens)
    return (max(0, int(prompt_tokens or 0)) / 1_000_000.0) * in_price + (
        max(0, int(output_tokens or 0)) / 1_000_000.0
    ) * out_price


def estimate_cost_from_usage(cfg: Dict[str, Any], model_name: str, usage_meta: Dict[str, Any]) -> float:
    if not isinstance(usage_meta, dict):
        return 0.0
    try:
        prompt_tokens = int(usage_meta.get("promptTokenCount", usage_meta.get("prompt_tokens", 0)) or 0)
    except Exception:
        prompt_tokens = 0
    try:
        output_tokens = int(usage_meta.get("candidatesTokenCount", usage_meta.get("output_tokens", 0)) or 0)
    except Exception:
        output_tokens = 0
    try:
        total_tokens = int(usage_meta.get("totalTokenCount", usage_meta.get("total_tokens", 0)) or 0)
    except Exception:
        total_tokens = prompt_tokens + output_tokens
    if total_tokens <= 0:
        total_tokens = prompt_tokens + output_tokens
    return estimate_cost_usd(
        cfg,
        model_name,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def episode_expected_revenue_usd(cfg: Dict[str, Any]) -> float:
    return max(0.0, float(_cfg_get(cfg, "economics.episode_expected_revenue_usd", 0.50) or 0.50))


def cost_guard_enforcement_enabled(cfg: Dict[str, Any]) -> bool:
    raw = _cfg_get(cfg, "economics.enforce_cost_guards", False)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def budget_snapshot(cfg: Dict[str, Any], total_cost_usd: float) -> Dict[str, Any]:
    revenue = episode_expected_revenue_usd(cfg)
    target_ratio = max(0.0, float(_cfg_get(cfg, "economics.target_cost_ratio", 0.15) or 0.15))
    hard_ratio = max(target_ratio, float(_cfg_get(cfg, "economics.hard_cost_ratio", 0.20) or 0.20))
    ratio = (float(total_cost_usd or 0.0) / revenue) if revenue > 0 else 0.0
    state = "within_target"
    if ratio > hard_ratio:
        state = "over_hard_cap"
    elif ratio > target_ratio:
        state = "over_target"
    return {
        "episode_expected_revenue_usd": round(revenue, 8),
        "episode_estimated_cost_usd": round(float(total_cost_usd or 0.0), 8),
        "episode_cost_ratio": round(ratio, 8),
        "episode_budget_state": state,
        "economics_target_cost_ratio": round(target_ratio, 8),
        "economics_hard_cost_ratio": round(hard_ratio, 8),
        "economics_cost_guards_enabled": cost_guard_enforcement_enabled(cfg),
    }


def build_episode_cost_updates(
    cfg: Dict[str, Any],
    task_state: Dict[str, Any] | None,
    *,
    stage_name: str,
    model_name: str,
    cost_usd: float,
    key_class: str = "",
) -> Dict[str, Any]:
    existing = dict(task_state or {})
    by_stage = dict(existing.get("episode_cost_by_stage", {}) or {})
    by_model = dict(existing.get("episode_cost_by_model", {}) or {})
    stage = str(stage_name or "").strip() or "unknown"
    model = str(model_name or "").strip() or "unknown"
    delta = round(float(cost_usd or 0.0), 8)
    by_stage[stage] = round(float(by_stage.get(stage, 0.0) or 0.0) + delta, 8)
    by_model[model] = round(float(by_model.get(model, 0.0) or 0.0) + delta, 8)
    total_cost = sum(float(v or 0.0) for v in by_stage.values())
    updates = {
        "episode_cost_by_stage": by_stage,
        "episode_cost_by_model": by_model,
        "last_cost_stage_name": stage,
        "last_cost_stage_delta_usd": delta,
    }
    if key_class:
        updates["episode_key_class_used"] = str(key_class)
    updates.update(budget_snapshot(cfg, total_cost))
    return updates


def estimate_minimum_episode_cost_usd(cfg: Dict[str, Any], segment_count: int) -> Dict[str, Any]:
    max_segments = max(2, int(_cfg_get(cfg, "run.segment_chunking_max_segments_per_request", 8) or 8))
    min_chunking_segments = max(2, int(_cfg_get(cfg, "run.segment_chunking_min_segments", 16) or 16))
    effective_segment_count = max(1, int(segment_count or 0))
    labeling_requests = 1
    if effective_segment_count >= min_chunking_segments:
        labeling_requests = max(1, math.ceil(effective_segment_count / max_segments))
    labeling_model = resolve_stage_model(cfg, "labeling", _cfg_get(cfg, "gemini.model", "gemini-2.5-flash"))
    chat_only_mode = bool(_cfg_get(cfg, "run.chat_only_mode", False))
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
    compare_enabled_raw = run_cfg.get("pre_submit_chat_compare_enabled", None)
    compare_enabled = not chat_only_mode
    if compare_enabled_raw is None:
        compare_enabled = compare_enabled and bool(
            str(_cfg_get(cfg, "run.pre_submit_chat_compare_model", "") or "").strip()
            or str(_cfg_get(cfg, "gemini.stage_models.compare_chat", "") or "").strip()
        )
    else:
        compare_enabled = compare_enabled and bool(compare_enabled_raw)
    compare_model = ""
    if compare_enabled:
        compare_model = resolve_stage_model(
            cfg,
            "compare_chat",
            _cfg_get(cfg, "run.pre_submit_chat_compare_model", _cfg_get(cfg, "gemini.model", "gemini-3.1-pro-preview")),
        )
    labeling_cost = labeling_requests * estimate_cost_usd(
        cfg,
        labeling_model,
        prompt_tokens=_AVG_PROMPT_TOKENS_PER_REQUEST,
        output_tokens=_AVG_OUTPUT_TOKENS_PER_REQUEST,
        total_tokens=_AVG_PROMPT_TOKENS_PER_REQUEST + _AVG_OUTPUT_TOKENS_PER_REQUEST,
    )
    compare_cost = 0.0
    if compare_model:
        compare_cost = estimate_cost_usd(
            cfg,
            compare_model,
            prompt_tokens=_AVG_PROMPT_TOKENS_PER_REQUEST,
            output_tokens=_AVG_OUTPUT_TOKENS_PER_REQUEST,
            total_tokens=_AVG_PROMPT_TOKENS_PER_REQUEST + _AVG_OUTPUT_TOKENS_PER_REQUEST,
        )
    total = labeling_cost + compare_cost
    summary = budget_snapshot(cfg, total)
    summary.update(
        {
            "minimum_labeling_requests": labeling_requests,
            "minimum_labeling_model": labeling_model,
            "minimum_compare_model": compare_model,
            "minimum_labeling_cost_usd": round(labeling_cost, 8),
            "minimum_compare_cost_usd": round(compare_cost, 8),
        }
    )
    return summary


def would_exceed_ratio_cap(
    cfg: Dict[str, Any],
    task_state: Dict[str, Any] | None,
    *,
    additional_cost_usd: float,
    ratio_limit: float,
) -> bool:
    current_total = float((task_state or {}).get("episode_estimated_cost_usd", 0.0) or 0.0)
    revenue = episode_expected_revenue_usd(cfg)
    if revenue <= 0:
        return False
    projected_ratio = (current_total + float(additional_cost_usd or 0.0)) / revenue
    return projected_ratio > max(0.0, float(ratio_limit or 0.0))


__all__ = [
    "budget_snapshot",
    "build_episode_cost_updates",
    "cost_guard_enforcement_enabled",
    "episode_expected_revenue_usd",
    "estimate_cost_from_usage",
    "estimate_cost_usd",
    "estimate_minimum_episode_cost_usd",
    "normalize_model_name",
    "resolve_model_prices",
    "resolve_stage_model",
    "would_exceed_ratio_cap",
]
