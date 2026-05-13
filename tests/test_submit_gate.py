"""Tests for submit_gate.py — deterministic hard gate before Atlas submit."""

import pytest

from submit_gate import SubmitGateResult, evaluate_submit_safety


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_judge(
    winner="api",
    submit_safe="api",
    score=96,
    hallucination=False,
):
    """Build a minimal judge_result dict."""
    return {
        "winner": winner,
        "submit_safe_solution": submit_safe,
        "scores": {winner: score} if winner else {},
        "hallucination": {winner: hallucination} if winner else {},
    }


DUMMY_INPUTS = {
    "tier2_path": "",
    "api_path": "",
    "chat_path": "",
    "vertex_chat_path": "",
}


# ---------------------------------------------------------------------------
# Check 1: winner validity
# ---------------------------------------------------------------------------

class TestWinnerValidity:
    def test_winner_none_blocks(self):
        result = evaluate_submit_safety(
            episode_id="ep1",
            judge_result=_make_judge(winner="none"),
            inputs=DUMMY_INPUTS,
            run_validator=False,
        )
        assert not result.safe
        assert result.reason == "winner_none_or_invalid"
        assert result.suggested_action == "retry_with_repair"

    def test_winner_empty_blocks(self):
        result = evaluate_submit_safety(
            episode_id="ep2",
            judge_result=_make_judge(winner=""),
            inputs=DUMMY_INPUTS,
            run_validator=False,
        )
        assert not result.safe
        assert result.reason == "winner_none_or_invalid"

    def test_valid_winner_passes(self):
        for w in ("tier2", "api", "chat", "vertex_chat"):
            result = evaluate_submit_safety(
                episode_id=f"ep_{w}",
                judge_result=_make_judge(winner=w, submit_safe=w),
                inputs=DUMMY_INPUTS,
                run_validator=False,
            )
            assert "winner_validity" in result.checks_performed


# ---------------------------------------------------------------------------
# Check 2: submit_safe_solution (RELAXED behavior)
# ---------------------------------------------------------------------------

class TestSubmitSafeRelaxed:
    def test_empty_submit_safe_is_soft_warning_not_block(self):
        """Empty submit_safe_solution should NOT block — this is the critical fix."""
        result = evaluate_submit_safety(
            episode_id="ep3",
            judge_result=_make_judge(winner="api", submit_safe="", score=96),
            inputs=DUMMY_INPUTS,
            run_validator=False,
        )
        # Should pass (all other checks OK)
        assert result.safe, f"Expected safe=True but got reason={result.reason}"
        assert result.submit_safe_mismatch is True  # flagged as warning
        assert "submit_safe_warning" in result.reason_detail

    def test_none_submit_safe_is_soft_warning(self):
        judge = _make_judge(winner="api", score=96)
        judge["submit_safe_solution"] = None
        result = evaluate_submit_safety(
            episode_id="ep4",
            judge_result=judge,
            inputs=DUMMY_INPUTS,
            run_validator=False,
        )
        assert result.safe

    def test_explicit_mismatch_blocks(self):
        """submit_safe=chat but winner=api should hard block."""
        result = evaluate_submit_safety(
            episode_id="ep5",
            judge_result=_make_judge(winner="api", submit_safe="chat", score=96),
            inputs=DUMMY_INPUTS,
            run_validator=False,
        )
        assert not result.safe
        assert result.reason == "submit_safe_explicit_mismatch"
        assert result.suggested_action == "retry_with_repair"

    def test_matching_submit_safe_passes(self):
        result = evaluate_submit_safety(
            episode_id="ep6",
            judge_result=_make_judge(winner="api", submit_safe="api", score=96),
            inputs=DUMMY_INPUTS,
            run_validator=False,
        )
        assert result.safe


# ---------------------------------------------------------------------------
# Check 3: hallucination flag
# ---------------------------------------------------------------------------

class TestHallucinationFlag:
    def test_hallucination_true_blocks(self):
        result = evaluate_submit_safety(
            episode_id="ep7",
            judge_result=_make_judge(hallucination=True),
            inputs=DUMMY_INPUTS,
            run_validator=False,
        )
        assert not result.safe
        assert result.reason == "winner_hallucination"
        assert result.llm_hallucination_flag is True

    def test_hallucination_false_passes(self):
        result = evaluate_submit_safety(
            episode_id="ep8",
            judge_result=_make_judge(hallucination=False),
            inputs=DUMMY_INPUTS,
            run_validator=False,
        )
        assert result.safe


# ---------------------------------------------------------------------------
# Check 5: score threshold (with relaxed threshold when validator passes)
# ---------------------------------------------------------------------------

class TestScoreThreshold:
    def test_score_below_95_blocks_without_validator(self):
        result = evaluate_submit_safety(
            episode_id="ep9",
            judge_result=_make_judge(score=93),
            inputs=DUMMY_INPUTS,
            run_validator=False,
        )
        assert not result.safe
        assert "score_below_threshold" in result.reason

    def test_score_at_95_passes_without_validator(self):
        result = evaluate_submit_safety(
            episode_id="ep10",
            judge_result=_make_judge(score=95),
            inputs=DUMMY_INPUTS,
            run_validator=False,
        )
        assert result.safe

    def test_missing_score_blocks(self):
        judge = _make_judge(score=96)
        judge["scores"] = {}  # no score for winner
        result = evaluate_submit_safety(
            episode_id="ep11",
            judge_result=judge,
            inputs=DUMMY_INPUTS,
            run_validator=False,
        )
        assert not result.safe
        assert result.reason == "missing_winner_score"

    def test_relaxed_threshold_param(self):
        """Score 90 should pass with min_pass_score_validator_ok=88 when validator skipped."""
        # Without validator, the normal threshold applies
        result = evaluate_submit_safety(
            episode_id="ep12",
            judge_result=_make_judge(score=90),
            inputs=DUMMY_INPUTS,
            run_validator=False,
            min_pass_score=95,
            min_pass_score_validator_ok=88,
        )
        # Without validator running, validator_ok is None, so normal threshold (95) applies
        assert not result.safe


# ---------------------------------------------------------------------------
# SubmitGateResult
# ---------------------------------------------------------------------------

class TestSubmitGateResult:
    def test_to_dict_includes_suggested_action(self):
        r = SubmitGateResult(
            episode_id="ep_test",
            safe=False,
            winner="api",
            reason="test_reason",
            suggested_action="retry_with_repair",
        )
        d = r.to_dict()
        assert d["suggested_action"] == "retry_with_repair"

    def test_summary_line_format(self):
        r = SubmitGateResult(
            episode_id="ep_test",
            safe=True,
            winner="api",
            reason="all_checks_passed",
            score_pct=96,
            validator_ok=True,
        )
        line = r.summary_line()
        assert "APPROVED" in line
        assert "ep_test" in line

    def test_missing_judge_result(self):
        result = evaluate_submit_safety(
            episode_id="ep_bad",
            judge_result="not_a_dict",  # type: ignore
            inputs=DUMMY_INPUTS,
            run_validator=False,
        )
        assert not result.safe
        assert result.reason == "missing_judge_result"
        assert result.suggested_action == "manual_review"
