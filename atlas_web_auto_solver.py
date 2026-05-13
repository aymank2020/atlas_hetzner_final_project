"""Compatibility shim for the refactored Atlas auto-solver package layout."""

import sys

from src.infra.runtime import (
    _install_signal_handlers,
    _request_shutdown,
    _shutdown_requested,
    _sleep_with_shutdown_heartbeat,
)
from src.infra.solver_config import DEFAULT_CONFIG, _load_selectors_yaml, load_config
from src.rules.labels import (
    _allowed_label_start_verb_token_patterns_from_cfg,
    _autofix_label_candidate,
    _build_auto_continuity_merge_operations,
    _count_atomic_actions_in_label,
    _label_starts_with_allowed_action_verb,
    _rewrite_label_tier3,
)
from src.rules.policy_gate import _validate_segment_plan_against_policy
from src.solver import legacy_impl as _legacy
from src.solver.browser import _selector_variants
from src.solver.cli import _apply_cli_overrides, main, parse_args
from src.solver.gemini import _normalize_upload_chunk_size, call_gemini_labels
from src.solver.orchestrator import run
from src.solver.prompting import (
    _apply_consistency_aliases_to_label,
    _find_equivalent_canonical_term,
    _update_chunk_consistency_memory,
    build_prompt,
)
from src.solver.segments import _normalize_operations, apply_labels, extract_segments

_SCRIPT_BUILD = "2026-04-03.1200-refactor"

__all__ = [
    "_SCRIPT_BUILD",
    "DEFAULT_CONFIG",
    "_shutdown_requested",
    "_request_shutdown",
    "_install_signal_handlers",
    "_sleep_with_shutdown_heartbeat",
    "_load_selectors_yaml",
    "_selector_variants",
    "_allowed_label_start_verb_token_patterns_from_cfg",
    "_apply_consistency_aliases_to_label",
    "_autofix_label_candidate",
    "_build_auto_continuity_merge_operations",
    "_count_atomic_actions_in_label",
    "_find_equivalent_canonical_term",
    "_label_starts_with_allowed_action_verb",
    "_normalize_operations",
    "_normalize_upload_chunk_size",
    "_rewrite_label_tier3",
    "_update_chunk_consistency_memory",
    "_validate_segment_plan_against_policy",
    "build_prompt",
    "call_gemini_labels",
    "extract_segments",
    "apply_labels",
    "load_config",
    "run",
    "parse_args",
    "_apply_cli_overrides",
    "main",
]


def __getattr__(name: str):
    """Fallback to the legacy implementation during the transition period."""
    return getattr(_legacy, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_legacy)))


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            continue


if __name__ == "__main__":
    _configure_stdio()
    main()
