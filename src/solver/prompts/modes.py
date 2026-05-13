"""Canonical prompt modes used by the Atlas v2 Gemini flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PromptModeContract:
    mode: str
    expected_schema: str
    description: str


MODE_CONTRACTS: Dict[str, PromptModeContract] = {
    "labeling": PromptModeContract(
        mode="labeling",
        expected_schema="segments_only",
        description="Generate labels only for the current DOM-grounded segments.",
    ),
    "ops_planner": PromptModeContract(
        mode="ops_planner",
        expected_schema="operations_only",
        description="Plan structural operations only for the supplied segment scope.",
    ),
    "repair": PromptModeContract(
        mode="repair",
        expected_schema="segments_only",
        description="Repair only the failing local segment scope using authoritative DOM rows.",
    ),
}


def resolve_mode_contract(mode: str) -> PromptModeContract:
    clean = str(mode or "").strip().lower()
    return MODE_CONTRACTS.get(clean, MODE_CONTRACTS["labeling"])


__all__ = [
    "MODE_CONTRACTS",
    "PromptModeContract",
    "resolve_mode_contract",
]
