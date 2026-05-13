"""
Tests for Spec-Kit rule enforcement in validator.py.
Tests: RULE-HOLD (no intent), RULE-MERGE (consecutive identical labels).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import validator


class TestCheckHoldIntent:
    """RULE-HOLD: hold [object] must NOT include intent."""

    def test_hold_with_intent_to_check(self):
        assert validator.check_hold_intent("hold phone to check charging status") is True

    def test_hold_with_intent_to_see(self):
        assert validator.check_hold_intent("hold battery to see voltage") is True

    def test_hold_with_intent_to_inspect(self):
        assert validator.check_hold_intent("hold cable to inspect damage") is True

    def test_hold_with_intent_to_verify(self):
        assert validator.check_hold_intent("hold screen to verify alignment") is True

    def test_hold_with_intent_to_examine(self):
        assert validator.check_hold_intent("hold phone assembly to examine connector") is True

    def test_hold_with_intent_for_inspection(self):
        assert validator.check_hold_intent("hold phone for inspection") is True

    def test_hold_clean_simple(self):
        assert validator.check_hold_intent("hold phone assembly") is False

    def test_hold_clean_with_location(self):
        assert validator.check_hold_intent("hold probes on battery terminals") is False

    def test_hold_clean_object_only(self):
        assert validator.check_hold_intent("hold battery") is False

    def test_non_hold_label(self):
        assert validator.check_hold_intent("pick up phone") is False

    def test_non_hold_with_to(self):
        assert validator.check_hold_intent("move cable to battery") is False


class TestCheckMergeableConsecutive:
    """RULE-MERGE: consecutive identical labels under 60s must merge."""

    def test_mergeable_pair(self):
        """Two consecutive 'hold probes' segments totaling 17s should be flagged."""
        segs = [
            {"segment_index": 6, "start_sec": 23.5, "end_sec": 33.5, "label": "hold probes on battery terminals"},
            {"segment_index": 7, "start_sec": 33.5, "end_sec": 40.5, "label": "hold probes on battery terminals"},
        ]
        warnings = validator.check_mergeable_consecutive(segs)
        assert len(warnings) == 1
        assert "mergeable_consecutive" in warnings[0]
        assert "6->7" in warnings[0]

    def test_non_mergeable_different_labels(self):
        """Different labels should NOT be flagged."""
        segs = [
            {"segment_index": 1, "start_sec": 0, "end_sec": 5, "label": "pick up phone"},
            {"segment_index": 2, "start_sec": 5, "end_sec": 10, "label": "place phone on table"},
        ]
        warnings = validator.check_mergeable_consecutive(segs)
        assert len(warnings) == 0

    def test_non_mergeable_over_60s(self):
        """Identical labels exceeding 60s combined should NOT be flagged for merge."""
        segs = [
            {"segment_index": 1, "start_sec": 0, "end_sec": 40, "label": "sew fabric"},
            {"segment_index": 2, "start_sec": 40, "end_sec": 90, "label": "sew fabric"},
        ]
        warnings = validator.check_mergeable_consecutive(segs)
        assert len(warnings) == 0

    def test_mergeable_exactly_60s(self):
        """Identical labels totaling exactly 60s should still be flagged for merge."""
        segs = [
            {"segment_index": 1, "start_sec": 0, "end_sec": 25, "label": "hold wire"},
            {"segment_index": 2, "start_sec": 25, "end_sec": 60, "label": "hold wire"},
        ]
        warnings = validator.check_mergeable_consecutive(segs)
        assert len(warnings) == 1
        assert "1->2" in warnings[0]

    def test_mergeable_triple(self):
        """Three consecutive identical labels totaling 15s should be flagged."""
        segs = [
            {"segment_index": 1, "start_sec": 0, "end_sec": 5, "label": "hold wire"},
            {"segment_index": 2, "start_sec": 5, "end_sec": 10, "label": "hold wire"},
            {"segment_index": 3, "start_sec": 10, "end_sec": 15, "label": "hold wire"},
        ]
        warnings = validator.check_mergeable_consecutive(segs)
        assert len(warnings) == 1
        assert "1->3" in warnings[0]

    def test_gap_beyond_tolerance(self):
        """Segments with gap > 0.5s should NOT be considered consecutive."""
        segs = [
            {"segment_index": 1, "start_sec": 0, "end_sec": 5, "label": "hold wire"},
            {"segment_index": 2, "start_sec": 6.0, "end_sec": 10, "label": "hold wire"},
        ]
        warnings = validator.check_mergeable_consecutive(segs)
        assert len(warnings) == 0

    def test_gap_within_tolerance(self):
        """Segments with gap <= 0.5s should be treated as continuous."""
        segs = [
            {"segment_index": 1, "start_sec": 0, "end_sec": 5, "label": "hold wire"},
            {"segment_index": 2, "start_sec": 5.3, "end_sec": 10, "label": "hold wire"},
        ]
        warnings = validator.check_mergeable_consecutive(segs)
        assert len(warnings) == 1

    def test_no_action_not_merged(self):
        """'No Action' segments should never be flagged for merge."""
        segs = [
            {"segment_index": 1, "start_sec": 0, "end_sec": 5, "label": "No Action"},
            {"segment_index": 2, "start_sec": 5, "end_sec": 10, "label": "No Action"},
        ]
        warnings = validator.check_mergeable_consecutive(segs)
        assert len(warnings) == 0

    def test_empty_segments(self):
        warnings = validator.check_mergeable_consecutive([])
        assert len(warnings) == 0


class TestHoldIntentInValidateSegment:
    """Ensure hold_intent_violation propagates through validate_segment."""

    def test_violation_appears_in_errors(self):
        seg = {
            "segment_index": 1,
            "start_sec": 0.0,
            "end_sec": 5.0,
            "duration_sec": 5.0,
            "label": "hold phone to check charging status",
            "granularity": "coarse",
            "confidence": 0.9,
            "rule_checks": {},
            "audit_risk": {"level": "low", "reasons": []},
        }
        report, errors, _ = validator.validate_segment(seg, 100.0)
        assert "hold_intent_violation" in errors
        assert report["derived_rule_checks"]["no_hold_intent_violation"] is False

    def test_clean_hold_passes(self):
        seg = {
            "segment_index": 1,
            "start_sec": 0.0,
            "end_sec": 5.0,
            "duration_sec": 5.0,
            "label": "hold phone assembly",
            "granularity": "coarse",
            "confidence": 0.9,
            "rule_checks": {},
            "audit_risk": {"level": "low", "reasons": []},
        }
        report, errors, _ = validator.validate_segment(seg, 100.0)
        assert "hold_intent_violation" not in errors
        assert report["derived_rule_checks"]["no_hold_intent_violation"] is True
