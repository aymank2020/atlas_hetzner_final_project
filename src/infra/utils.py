"""Small shared utilities used across Atlas solver modules."""

from __future__ import annotations

import re
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_label_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


__all__ = ["_safe_float", "_normalize_label_for_compare"]
