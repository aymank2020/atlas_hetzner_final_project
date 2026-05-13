"""Tests for Sprint 10: Adaptive Selector Engine."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# ── Test _load_selectors_yaml (from auto-solver) ──────────────────

# We import the function directly since it's a module-level function
sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_load_selectors_yaml_missing_file():
    """Should return empty dict when YAML file doesn't exist."""
    from atlas_web_auto_solver import _load_selectors_yaml

    result = _load_selectors_yaml("nonexistent_file.yaml")
    assert result == {}


def test_load_selectors_yaml_with_strategies():
    """Should convert strategy lists to ||-separated strings."""
    from atlas_web_auto_solver import _load_selectors_yaml

    yaml_content = """
selectors:
  start_button:
    strategies:
      - 'button[type="submit"]'
      - 'button:has-text("Start")'
      - 'button:has-text("Begin")'
    intent: "Main CTA button"
  email_input:
    strategies:
      - '#email'
      - 'input[type="email"]'
    intent: "Email field"
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write(yaml_content)
        f.flush()
        path = f.name

    try:
        result = _load_selectors_yaml(path)
        assert len(result) == 2
        assert "start_button" in result
        assert "email_input" in result
        assert "||" in result["start_button"]
        assert 'button[type="submit"]' in result["start_button"]
        assert '#email' in result["email_input"]
    finally:
        os.unlink(path)


def test_load_selectors_yaml_plain_strings():
    """Should accept plain string values (backward compatible)."""
    from atlas_web_auto_solver import _load_selectors_yaml

    yaml_content = """
selectors:
  video_element: "video"
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write(yaml_content)
        f.flush()
        path = f.name

    try:
        result = _load_selectors_yaml(path)
        assert result["video_element"] == "video"
    finally:
        os.unlink(path)


def test_load_selectors_yaml_list_format():
    """Should accept flat list format."""
    from atlas_web_auto_solver import _load_selectors_yaml

    yaml_content = """
selectors:
  otp_input:
    - '#code'
    - 'input[inputmode="numeric"]'
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write(yaml_content)
        f.flush()
        path = f.name

    try:
        result = _load_selectors_yaml(path)
        assert "||" in result["otp_input"]
        assert "#code" in result["otp_input"]
    finally:
        os.unlink(path)


def test_load_selectors_yaml_invalid_yaml():
    """Should return empty dict on invalid YAML."""
    from atlas_web_auto_solver import _load_selectors_yaml

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write("{{invalid: yaml: [[[")
        f.flush()
        path = f.name

    try:
        result = _load_selectors_yaml(path)
        assert result == {}
    finally:
        os.unlink(path)


# ── Test _selector_variants ───────────────────────────────────────

def test_selector_variants_split():
    """Should split selectors by || separator."""
    from atlas_web_auto_solver import _selector_variants

    result = _selector_variants('#email || input[type="email"] || input.email')
    assert len(result) == 3
    assert result[0] == "#email"
    assert result[1] == 'input[type="email"]'
    assert result[2] == "input.email"


def test_selector_variants_single():
    """Should handle single selector without ||."""
    from atlas_web_auto_solver import _selector_variants

    result = _selector_variants("button.primary")
    assert result == ["button.primary"]


def test_selector_variants_empty():
    """Should handle empty string."""
    from atlas_web_auto_solver import _selector_variants

    result = _selector_variants("")
    assert result == []


# ── Test DEFAULT_CONFIG selectors are resilient ───────────────────

def test_default_selectors_have_fallbacks():
    """Every critical selector should have at least 3 fallback strategies."""
    from atlas_web_auto_solver import DEFAULT_CONFIG, _selector_variants

    critical_keys = [
        "start_button", "verify_button", "complete_button",
        "reserve_episodes_button", "confirm_reserve_button",
        "release_all_button", "quality_review_submit_button",
    ]
    selectors = DEFAULT_CONFIG["atlas"]["selectors"]
    for key in critical_keys:
        variants = _selector_variants(selectors[key])
        assert len(variants) >= 3, (
            f"Selector '{key}' has only {len(variants)} variants, expected ≥3"
        )


def test_default_selectors_no_exact_phrases():
    """Critical selectors should NOT rely on exact full phrases."""
    from atlas_web_auto_solver import DEFAULT_CONFIG

    selectors = DEFAULT_CONFIG["atlas"]["selectors"]
    # These exact phrases existed before and should NOT be the only option
    forbidden_only = [
        'button:has-text("Start Earning Today")',
        'button:has-text("Reserve 5 Episodes")',
    ]
    for phrase in forbidden_only:
        for key, value in selectors.items():
            if value == phrase:  # If it's the ONLY selector (no ||)
                pytest.fail(
                    f"Selector '{key}' uses exact phrase '{phrase}' as only option"
                )
