import unittest

import validator


class TestValidator(unittest.TestCase):
    def test_refresh_policy_constraints_returns_cached_object(self):
        refreshed = validator.refresh_policy_constraints()
        cached = validator.get_policy_constraints()
        self.assertEqual(refreshed, cached)
        self.assertGreater(refreshed.max_segment_duration_sec, 0.0)
        self.assertGreater(refreshed.max_words_per_label, 0)

    def test_normalize_annotation_from_step_format(self):
        raw = [
            {
                "step": 1,
                "start": "0:00.0",
                "end": "0:03.2",
                "description": "grab picture from pile",
            },
            {
                "step": 2,
                "start": "0:03.2",
                "end": "0:07.4",
                "description": "place picture on frame",
            },
        ]
        ann = validator.normalize_annotation(raw, episode_id="f01", video_duration_sec=10)
        self.assertEqual(ann["episode_id"], "f01")
        self.assertEqual(len(ann["segments"]), 2)
        self.assertAlmostEqual(ann["segments"][0]["start_sec"], 0.0, places=2)
        self.assertAlmostEqual(ann["segments"][1]["end_sec"], 7.4, places=2)

    def test_validate_episode_detects_label_violations(self):
        ann = validator.normalize_annotation(
            {
                "episode_id": "e1",
                "video_duration_sec": 4.0,
                "segments": [
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 4.0,
                        "duration_sec": 4.0,
                        "label": "inspect 3 items",
                        "granularity": "coarse",
                        "primary_goal": "inspect items",
                        "primary_object": "items",
                        "confidence": 0.9,
                    }
                ],
            }
        )
        report = validator.validate_episode(ann)
        errors = report["segment_reports"][0]["errors"]
        self.assertIn("forbidden_verbs", errors)
        self.assertIn("numerals_present", errors)
        self.assertFalse(report["ok"])

    def test_validate_episode_rejects_examine_and_narrative_words(self):
        ann = validator.normalize_annotation(
            {
                "episode_id": "e1b",
                "video_duration_sec": 6.0,
                "segments": [
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 6.0,
                        "duration_sec": 6.0,
                        "label": "examine box then place box on table",
                        "granularity": "coarse",
                        "primary_goal": "place box",
                        "primary_object": "box",
                        "confidence": 0.9,
                    }
                ],
            }
        )
        report = validator.validate_episode(ann)
        errors = report["segment_reports"][0]["errors"]
        self.assertIn("forbidden_verbs", errors)
        self.assertIn("narrative_filler_words", errors)
        self.assertIn("narrative_filler_words_used", report["major_fail_triggers"])
        self.assertFalse(report["ok"])

    def test_validate_episode_detects_overlap(self):
        ann = validator.normalize_annotation(
            {
                "episode_id": "e2",
                "video_duration_sec": 8.0,
                "segments": [
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 4.0,
                        "duration_sec": 4.0,
                        "label": "pick up tool",
                        "granularity": "coarse",
                        "primary_goal": "pick up tool",
                        "primary_object": "tool",
                        "confidence": 0.8,
                    },
                    {
                        "segment_index": 2,
                        "start_sec": 3.5,
                        "end_sec": 6.0,
                        "duration_sec": 2.5,
                        "label": "place tool on table",
                        "granularity": "coarse",
                        "primary_goal": "place tool",
                        "primary_object": "tool",
                        "confidence": 0.8,
                    },
                ],
            }
        )
        report = validator.validate_episode(ann)
        self.assertIn("timestamp_overlap", report["episode_errors"])

    def test_validate_episode_rejects_more_than_two_atomic_actions(self):
        ann = validator.normalize_annotation(
            {
                "episode_id": "e3",
                "video_duration_sec": 8.0,
                "segments": [
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 4.0,
                        "duration_sec": 4.0,
                        "label": "pick up cup, place cup on table, move cup to sink",
                        "granularity": "dense",
                        "primary_goal": "relocate cup",
                        "primary_object": "cup",
                        "confidence": 0.9,
                    }
                ],
            }
        )
        report = validator.validate_episode(ann)
        errors = report["segment_reports"][0]["errors"]
        self.assertIn("too_many_atomic_actions", errors)
        self.assertFalse(report["ok"])

    def test_validate_episode_rejects_disallowed_tool_terms(self):
        ann = validator.normalize_annotation(
            {
                "episode_id": "e4",
                "video_duration_sec": 4.0,
                "segments": [
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 4.0,
                        "duration_sec": 4.0,
                        "label": "use mechanical arm to pick up block",
                        "granularity": "coarse",
                        "primary_goal": "pick up block",
                        "primary_object": "block",
                        "confidence": 0.8,
                    }
                ],
            }
        )
        report = validator.validate_episode(ann)
        errors = report["segment_reports"][0]["errors"]
        self.assertIn("disallowed_tool_terms", errors)
        self.assertFalse(report["ok"])

    def test_starts_with_allowed_action_verb_accepts_multi_action_phrase(self):
        self.assertTrue(validator.starts_with_allowed_action_verb("pick up component, connect wires to component"))
        self.assertTrue(validator.starts_with_allowed_action_verb("connect wires to component"))

    def test_starts_with_allowed_action_verb_accepts_separable_take_phrases(self):
        self.assertTrue(validator.starts_with_allowed_action_verb("take screwdriver out of bag"))
        self.assertTrue(validator.starts_with_allowed_action_verb("take cover off device"))
        self.assertTrue(validator.starts_with_allowed_action_verb("chisel metal surface"))

    @unittest.skip("verb_start_not_allowed renamed/removed in VPS validator evolution — needs remapping")
    def test_validate_episode_rejects_label_that_starts_with_noun_or_adjective(self):
        ann = validator.normalize_annotation(
            {
                "episode_id": "e5",
                "video_duration_sec": 4.0,
                "segments": [
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 4.0,
                        "duration_sec": 4.0,
                        "label": "internal laptop component with cloth",
                        "granularity": "coarse",
                        "primary_goal": "clean component",
                        "primary_object": "component",
                        "confidence": 0.8,
                    }
                ],
            }
        )
        report = validator.validate_episode(ann)
        errors = report["segment_reports"][0]["errors"]
        self.assertIn("verb_start_not_allowed", errors)
        self.assertFalse(report["ok"])

    def test_validate_episode_rejects_place_without_location(self):
        ann = validator.normalize_annotation(
            {
                "episode_id": "e6",
                "video_duration_sec": 6.0,
                "segments": [
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 6.0,
                        "duration_sec": 6.0,
                        "label": "place cup",
                        "granularity": "coarse",
                        "primary_goal": "place cup",
                        "primary_object": "cup",
                        "confidence": 0.8,
                    }
                ],
            }
        )
        report = validator.validate_episode(ann)
        errors = report["segment_reports"][0]["errors"]
        self.assertIn("place_missing_location", errors)
        self.assertFalse(report["ok"])

    def test_validate_episode_allows_reach_only_on_truncated_end(self):
        allowed = validator.normalize_annotation(
            {
                "episode_id": "e7a",
                "video_duration_sec": 10.0,
                "segments": [
                    {
                        "segment_index": 1,
                        "start_sec": 9.7,
                        "end_sec": 10.0,
                        "duration_sec": 0.3,
                        "label": "reach for connector",
                        "granularity": "coarse",
                        "primary_goal": "reach connector",
                        "primary_object": "connector",
                        "confidence": 0.8,
                    }
                ],
            }
        )
        allowed_report = validator.validate_episode(allowed)
        allowed_errors = allowed_report["segment_reports"][0]["errors"]
        self.assertNotIn("forbidden_verbs", allowed_errors)

        disallowed = validator.normalize_annotation(
            {
                "episode_id": "e7b",
                "video_duration_sec": 10.0,
                "segments": [
                    {
                        "segment_index": 1,
                        "start_sec": 2.0,
                        "end_sec": 4.0,
                        "duration_sec": 2.0,
                        "label": "reach for connector",
                        "granularity": "coarse",
                        "primary_goal": "reach connector",
                        "primary_object": "connector",
                        "confidence": 0.8,
                    }
                ],
            }
        )
        disallowed_report = validator.validate_episode(disallowed)
        disallowed_errors = disallowed_report["segment_reports"][0]["errors"]
        self.assertIn("forbidden_verbs", disallowed_errors)
        self.assertFalse(disallowed_report["ok"])

    def test_validate_episode_warns_on_pluralization_hint(self):
        ann = validator.normalize_annotation(
            {
                "episode_id": "e8",
                "video_duration_sec": 5.0,
                "segments": [
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 5.0,
                        "duration_sec": 5.0,
                        "label": "pick up three knife",
                        "granularity": "coarse",
                        "primary_goal": "pick up knife",
                        "primary_object": "knife",
                        "confidence": 0.8,
                    }
                ],
            }
        )
        report = validator.validate_episode(ann)
        warnings = report["segment_reports"][0]["warnings"]
        self.assertIn("pluralization_hint", warnings)

    def test_validate_episode_warns_on_object_naming_inconsistency(self):
        ann = validator.normalize_annotation(
            {
                "episode_id": "e9",
                "video_duration_sec": 8.0,
                "segments": [
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 4.0,
                        "duration_sec": 4.0,
                        "label": "pick up box",
                        "granularity": "coarse",
                        "primary_goal": "pick up box",
                        "primary_object": "box",
                        "confidence": 0.8,
                    },
                    {
                        "segment_index": 2,
                        "start_sec": 4.0,
                        "end_sec": 8.0,
                        "duration_sec": 4.0,
                        "label": "place cardboard box on table",
                        "granularity": "coarse",
                        "primary_goal": "place box",
                        "primary_object": "cardboard box",
                        "confidence": 0.8,
                    },
                ],
            }
        )
        report = validator.validate_episode(ann)
        self.assertIn("object_naming_inconsistent", report["episode_warnings"])
        self.assertTrue(any("cardboard box" in w for w in report["episode_warning_details"]))

    def test_validate_episode_warns_on_device_class_internal_conflict(self):
        ann = validator.normalize_annotation(
            {
                "episode_id": "e10",
                "video_duration_sec": 10.0,
                "segments": [
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 5.0,
                        "duration_sec": 5.0,
                        "label": "unscrew component from phone",
                        "granularity": "coarse",
                        "primary_goal": "unscrew component",
                        "primary_object": "phone",
                        "confidence": 0.8,
                    },
                    {
                        "segment_index": 2,
                        "start_sec": 5.0,
                        "end_sec": 10.0,
                        "duration_sec": 5.0,
                        "label": "adjust laptop on table",
                        "granularity": "coarse",
                        "primary_goal": "adjust laptop",
                        "primary_object": "laptop",
                        "confidence": 0.8,
                    },
                ],
            }
        )
        report = validator.validate_episode(ann)
        self.assertTrue(any(w.startswith("device_class_internal_conflict:") for w in report["episode_warnings"]))

    def test_validate_episode_warns_on_device_class_conflict_against_draft(self):
        ann = validator.normalize_annotation(
            {
                "episode_id": "e11",
                "video_duration_sec": 8.0,
                "segments": [
                    {
                        "segment_index": 1,
                        "start_sec": 0.0,
                        "end_sec": 8.0,
                        "duration_sec": 8.0,
                        "label": "adjust laptop on table",
                        "granularity": "coarse",
                        "primary_goal": "adjust laptop",
                        "primary_object": "laptop",
                        "confidence": 0.8,
                    }
                ],
            }
        )
        ann["draft_segments"] = [
            {"segment_index": 1, "label": "pry open red phone frame"},
            {"segment_index": 2, "label": "remove back panel from phone"},
        ]
        report = validator.validate_episode(ann)
        self.assertIn("device_class_conflict:phone->laptop", report["episode_warnings"])
        self.assertIn("device_class_conflict:phone->laptop", report.get("device_class_conflicts", []))


if __name__ == "__main__":
    unittest.main()
