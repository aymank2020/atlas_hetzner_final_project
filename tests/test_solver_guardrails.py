import unittest

import validator
from src.rules.policy_gate import _validate_segment_plan_against_policy
from src.solver.legacy_impl import _collect_chunk_structural_operations, _parse_json_text
from src.solver.prompting import build_prompt


class TestSolverGuardrails(unittest.TestCase):
    def test_build_prompt_allows_split_consistently(self) -> None:
        prompt = build_prompt(
            [
                {
                    "segment_index": 1,
                    "start_sec": 0.0,
                    "end_sec": 4.0,
                    "current_label": "pick up fabric",
                    "raw_text": "pick up fabric",
                }
            ],
            extra_instructions="",
            allow_operations=True,
        )
        self.assertIn("Allowed operations: edit, split, merge. Do NOT use delete.", prompt)
        self.assertNotIn("Do NOT use delete or split", prompt)

    def test_policy_gate_rejects_overlong_segment(self) -> None:
        cfg = {"run": {"max_segment_duration_sec": 10.0}}
        source_segments = [
            {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"}
        ]
        segment_plan = {
            1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 12.4, "label": "pick up fabric"}
        }
        report = _validate_segment_plan_against_policy(cfg, source_segments, segment_plan)
        self.assertFalse(report["ok"])
        self.assertTrue(any("exceeds max" in item for item in report["errors"]))

    def test_policy_gate_rejects_guide_as_object_descriptor(self) -> None:
        cfg = {"run": {"max_segment_duration_sec": 10.0}}
        source_segments = [
            {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "current_label": "pick up fabric"}
        ]
        segment_plan = {
            1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "label": "pick up guide fabric from pile"}
        }
        report = _validate_segment_plan_against_policy(cfg, source_segments, segment_plan)
        self.assertFalse(report["ok"])
        self.assertTrue(any("object descriptor" in item for item in report["errors"]))

    def test_validator_flags_segment_too_long_and_adjust_over(self) -> None:
        validator.refresh_policy_constraints()
        annotation = {
            "episode_id": "guardrail_demo",
            "video_duration_sec": 15.0,
            "segments": [
                {
                    "segment_index": 1,
                    "start_sec": 0.0,
                    "end_sec": 12.4,
                    "duration_sec": 12.4,
                    "label": "adjust over guide fabric",
                }
            ],
        }
        report = validator.validate_episode(annotation)
        self.assertFalse(report["ok"])
        seg_errors = report["segment_reports"][0]["errors"]
        self.assertIn("segment_too_long", seg_errors)
        self.assertIn("invalid_adjust_over_phrase", seg_errors)

    def test_collect_chunk_structural_operations_keeps_only_overlong_split_ops(self) -> None:
        cfg = {"run": {"max_structural_operations": 12, "structural_allow_split": True}}
        chunk_segments = [
            {"segment_index": 4, "start_sec": 0.0, "end_sec": 12.6},
            {"segment_index": 5, "start_sec": 12.6, "end_sec": 17.0},
        ]
        payload = {
            "operations": [
                {"action": "split", "segment_index": 5},
                {"action": "split", "segment_index": 4},
                {"action": "merge", "segment_index": 4},
                {"action": "split", "segment_index": 4},
            ]
        }
        ops = _collect_chunk_structural_operations(
            cfg=cfg,
            chunk_payload=payload,
            chunk_segments=chunk_segments,
            max_segment_duration_sec=10.0,
            split_only=True,
        )
        self.assertEqual(ops, [{"action": "split", "segment_index": 4}])

    def test_parse_json_text_repairs_split_child_segment_indexes(self) -> None:
        payload = _parse_json_text(
            '{"operations":[{"action":"split","segment_index":1}],'
            '"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":8.0,"label":"drive screw"},'
            '{"segment_index":1_1,"start_sec":8.0,"end_sec":17.8,"label":"drive screw"}]}'
        )
        self.assertEqual(payload["operations"], [{"action": "split", "segment_index": 1}])
        self.assertEqual(payload["segments"][1]["segment_index"], "1_1")


if __name__ == "__main__":
    unittest.main()
