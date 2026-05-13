import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.solver.pre_submit_compare import (
    _clean_chat_subprocess_text,
    _chat_generation_attached_video,
    _derive_split_repair_operations,
    _parse_chat_subprocess_stdout,
    _resolve_pre_submit_chat_timeout_sec,
    _resolve_pre_submit_retry_light_chat_timeout_sec,
    _resolve_pre_submit_retry_light_max_wait_sec,
    _run_chat_generation_subprocess,
    maybe_run_pre_submit_chat_compare,
)


class TestPreSubmitChatCompare(unittest.TestCase):
    def _base_cfg(self, tmp: Path, storage_state: Path) -> dict:
        return {
            "run": {
                "output_dir": str(tmp / "outputs"),
                "pre_submit_chat_compare_enabled": True,
                "pre_submit_chat_compare_required": False,
                "pre_submit_chat_compare_block_on_chat_failure": False,
                "pre_submit_chat_compare_block_when_chat_better": True,
                "pre_submit_chat_compare_auto_adopt_same_timeline": True,
                "pre_submit_chat_compare_auto_repair_split_only": True,
                "pre_submit_chat_compare_same_timeline_epsilon_sec": 0.35,
                "pre_submit_chat_compare_allow_api_fallback_on_chat_failure_when_clean": True,
                "pre_submit_chat_compare_api_fallback_min_validator_score": 95,
                "max_segment_duration_sec": 10.0,
            },
            "gemini": {
                "chat_web_storage_state": str(storage_state),
                "model": "gemini-3.1-pro-preview",
            },
        }

    def test_compare_adopts_chat_when_same_timeline_and_better(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "current_label": "pick up fabric"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 4.0, "label": "adjust over guide fabric"}
            }

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                out_json = Path(cmd[cmd.index("--out-json") + 1])
                out_txt = Path(cmd[cmd.index("--out-txt") + 1])
                out_json.write_text(
                    json.dumps(
                        {
                            "segments": [
                                {"start_sec": 0.0, "end_sec": 4.0, "label": "pick up fabric from pile"}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_txt.write_text("1\t0.0\t4.0\tpick up fabric from pile\n", encoding="utf-8")
                return {
                    "out_json": str(out_json),
                    "out_txt": str(out_txt),
                    "model": "chat_ui",
                    "attach_notes": ["video.mp4: attached (3.1 MB, settle=4.0s, mode=file_chooser)"],
                }

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_same_timeline",
                    video_file=video_file,
                    api_model="gemini-3.1-flash-lite-preview",
                )

            self.assertTrue(result["executed"])
            self.assertEqual(result["winner"], "chat")
            self.assertTrue(result["adopted"])
            self.assertEqual(result["decision"], "adopt_chat_same_timeline")

    def test_compare_blocks_when_chat_is_better_but_changes_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "adjust over guide fabric"}
            }

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                out_json = Path(cmd[cmd.index("--out-json") + 1])
                out_txt = Path(cmd[cmd.index("--out-txt") + 1])
                out_json.write_text(
                    json.dumps(
                        {
                            "segments": [
                                {"start_sec": 0.0, "end_sec": 2.5, "label": "pick up fabric from pile"},
                                {"start_sec": 2.5, "end_sec": 5.0, "label": "place fabric on table"},
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_txt.write_text(
                    "1\t0.0\t2.5\tpick up fabric from pile\n2\t2.5\t5.0\tplace fabric on table\n",
                    encoding="utf-8",
                )
                return {
                    "out_json": str(out_json),
                    "out_txt": str(out_txt),
                    "model": "chat_ui",
                    "attach_notes": ["video.mp4: attached (3.1 MB, settle=4.0s, mode=file_chooser)"],
                }

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_timeline_differs",
                    video_file=video_file,
                    api_model="gemini-3.1-flash-lite-preview",
                )

            self.assertTrue(result["executed"])
            self.assertEqual(result["winner"], "chat")
            self.assertFalse(result["adopted"])
            self.assertTrue(result["block_apply"])
            self.assertEqual(result["decision"], "block_submit_chat_better_timeline_differs")

    def test_compare_ignores_chat_output_when_video_was_not_attached(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
            }

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                out_json = Path(cmd[cmd.index("--out-json") + 1])
                out_txt = Path(cmd[cmd.index("--out-txt") + 1])
                out_json.write_text(
                    json.dumps(
                        {
                            "segments": [
                                {"start_sec": 0.0, "end_sec": 2.5, "label": "pick up fabric from pile"},
                                {"start_sec": 2.5, "end_sec": 5.0, "label": "place fabric on table"},
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_txt.write_text(
                    "1\t0.0\t2.5\tpick up fabric from pile\n2\t2.5\t5.0\tplace fabric on table\n",
                    encoding="utf-8",
                )
                return {
                    "out_json": str(out_json),
                    "out_txt": str(out_txt),
                    "model": "chat_ui",
                    "attach_notes": ["video.mp4: skipped (file input not found)"],
                }

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_missing_chat_video",
                    video_file=video_file,
                    api_model="gemini-3.1-flash-lite-preview",
                )

            self.assertTrue(result["executed"])
            self.assertFalse(result["chat_video_attached"])
            self.assertEqual(result["winner"], "api")
            self.assertEqual(result["decision"], "chat_compare_missing_video_attachment")
            self.assertFalse(result["adopted"])

    def test_compare_retries_without_drive_picker_when_attachment_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            cfg["gemini"]["chat_web_prefer_drive_picker"] = True
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "adjust over guide fabric"}
            }
            calls = []

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                cfg_path = Path(cmd[cmd.index("--config") + 1])
                runtime_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                calls.append(runtime_cfg.get("gemini", {}))
                out_json = Path(cmd[cmd.index("--out-json") + 1])
                out_txt = Path(cmd[cmd.index("--out-txt") + 1])
                out_json.write_text(
                    json.dumps(
                        {
                            "segments": [
                                {"start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_txt.write_text("1\t0.0\t5.0\tpick up fabric from pile\n", encoding="utf-8")
                if len(calls) == 1:
                    return {
                        "out_json": str(out_json),
                        "out_txt": str(out_txt),
                        "model": "chat_ui",
                        "attach_notes": ["video.mp4: skipped (attachment chip not confirmed near composer)"],
                    }
                return {
                    "out_json": str(out_json),
                    "out_txt": str(out_txt),
                    "model": "chat_ui",
                    "attach_notes": ["video.mp4: attached (3.1 MB, settle=4.0s, mode=file_chooser)"],
                }

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_retry_no_drive",
                    video_file=video_file,
                    api_model="gemini-3.1-flash-lite-preview",
                )

            self.assertEqual(len(calls), 2)
            self.assertTrue(result["chat_video_attached"])
            self.assertTrue(result["chat_video_attached_retry"])
            self.assertIsNotNone(result.get("chat_generation_retry"))

    def test_compare_retries_with_lighter_runtime_after_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            cfg["gemini"]["chat_web_prefer_drive_picker"] = True
            cfg["gemini"]["chat_web_force_clean_thread"] = True
            cfg["gemini"]["context_text"] = "shared atlas context"
            cfg["gemini"]["context_file"] = "prompts/system_prompt.txt"
            cfg["gemini"]["timed_labels_context_text"] = "timed scope context"
            cfg["gemini"]["timed_labels_context_file"] = "prompts/generated_policy_context.txt"
            cfg["gemini"]["system_instruction_text"] = "long form system instruction"
            cfg["gemini"]["timed_labels_system_instruction_text"] = "timed labels system instruction"
            cfg["gemini"]["chat_web_seed_context_text"] = "seed context"
            cfg["run"]["pre_submit_chat_compare_max_wait_sec"] = 420.0
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "adjust over guide fabric"}
            }
            runtime_gemini_cfgs = []
            max_wait_values = []

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                cfg_path = Path(cmd[cmd.index("--config") + 1])
                runtime_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                runtime_gemini_cfgs.append(runtime_cfg.get("gemini", {}))
                max_wait_values.append(max_wait_sec)
                if len(runtime_gemini_cfgs) == 1:
                    raise RuntimeError("chat timed-label subprocess timed out")
                out_json = Path(cmd[cmd.index("--out-json") + 1])
                out_txt = Path(cmd[cmd.index("--out-txt") + 1])
                out_json.write_text(
                    json.dumps(
                        {
                            "segments": [
                                {"start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_txt.write_text("1\t0.0\t5.0\tpick up fabric from pile\n", encoding="utf-8")
                return {
                    "out_json": str(out_json),
                    "out_txt": str(out_txt),
                    "model": "chat_ui",
                    "attach_notes": ["video.mp4: attached (3.1 MB, settle=4.0s, mode=file_chooser)"],
                }

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_retry_timeout",
                    video_file=video_file,
                    api_model="gemini-3.1-flash-lite-preview",
                )

            self.assertEqual(len(runtime_gemini_cfgs), 2)
            self.assertTrue(runtime_gemini_cfgs[0]["chat_web_prefer_drive_picker"])
            self.assertTrue(runtime_gemini_cfgs[0]["chat_web_force_clean_thread"])
            self.assertFalse(runtime_gemini_cfgs[1]["chat_web_prefer_drive_picker"])
            self.assertFalse(runtime_gemini_cfgs[1]["chat_web_force_clean_thread"])
            self.assertFalse(runtime_gemini_cfgs[1]["chat_web_clean_thread_fallback_enabled"])
            self.assertEqual(runtime_gemini_cfgs[1]["chat_web_connect_over_cdp_url"], "")
            self.assertFalse(runtime_gemini_cfgs[1]["chat_web_seed_context_send_before_prompt"])
            self.assertEqual(runtime_gemini_cfgs[1]["chat_web_seed_context_file"], "")
            self.assertEqual(runtime_gemini_cfgs[1]["chat_web_seed_context_text"], "")
            self.assertEqual(runtime_gemini_cfgs[1]["chat_web_memory_primer_file"], "")
            self.assertEqual(runtime_gemini_cfgs[1]["context_text"], "")
            self.assertEqual(runtime_gemini_cfgs[1]["context_file"], "")
            self.assertEqual(runtime_gemini_cfgs[1]["timed_labels_context_text"], "")
            self.assertEqual(runtime_gemini_cfgs[1]["timed_labels_context_file"], "")
            self.assertEqual(runtime_gemini_cfgs[1]["system_instruction_text"], "")
            self.assertEqual(runtime_gemini_cfgs[1]["system_instruction_file"], "")
            self.assertEqual(runtime_gemini_cfgs[1]["timed_labels_system_instruction_text"], "")
            self.assertEqual(runtime_gemini_cfgs[1]["timed_labels_system_instruction_file"], "")
            self.assertEqual(
                runtime_gemini_cfgs[1]["chat_web_timeout_sec"],
                _resolve_pre_submit_retry_light_chat_timeout_sec(cfg),
            )
            self.assertEqual(
                runtime_gemini_cfgs[1]["timed_labels_tier2_max_chars"],
                6000,
            )
            self.assertEqual(runtime_gemini_cfgs[1]["chat_web_upload_settle_min_sec"], 6.0)
            self.assertEqual(runtime_gemini_cfgs[1]["chat_web_upload_settle_sec_per_100mb"], 10.0)
            self.assertEqual(runtime_gemini_cfgs[1]["chat_web_upload_settle_max_sec"], 60.0)
            self.assertNotIn("chat_web_user_data_dir", runtime_gemini_cfgs[1])
            self.assertEqual(max_wait_values[0], 420.0)
            self.assertEqual(max_wait_values[1], _resolve_pre_submit_retry_light_max_wait_sec(cfg))
            self.assertTrue(result["chat_generation_recovered_via_retry"])
            self.assertEqual(result["chat_generation_retry_reason"], "retryable_failure")
            self.assertNotIn("chat_generation_error", result)

    def test_compare_keeps_clean_api_when_chat_generation_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            cfg["run"]["pre_submit_chat_compare_required"] = True
            cfg["run"]["pre_submit_chat_compare_block_on_chat_failure"] = True
            cfg["run"]["pre_submit_chat_compare_auto_adopt_same_timeline"] = False
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "place screw on table"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "place screw on table"}
            }

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                raise RuntimeError("chat timed-label subprocess timed out")

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_clean_api_timeout",
                    video_file=video_file,
                    api_model="gemini-3.1-flash-lite-preview",
                )

            self.assertTrue(result["executed"])
            self.assertEqual(result["winner"], "api")
            self.assertEqual(result["decision"], "keep_api_after_chat_failure_clean_api")
            self.assertFalse(result.get("block_apply", False))
            self.assertTrue(result["chat_failure_but_api_clean"])

    def test_compare_retries_lighter_runtime_when_chat_input_not_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            cfg["gemini"]["chat_web_connect_over_cdp_url"] = "http://127.0.0.1:9222"
            cfg["gemini"]["chat_web_user_data_dir"] = "/tmp/gemini_profile"
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "adjust over guide fabric"}
            }
            runtime_gemini_cfgs = []

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                cfg_path = Path(cmd[cmd.index("--config") + 1])
                runtime_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                runtime_gemini_cfgs.append(runtime_cfg.get("gemini", {}))
                if len(runtime_gemini_cfgs) == 1:
                    raise RuntimeError("Gemini chat input not visible. Login/session is likely missing.")
                out_json = Path(cmd[cmd.index("--out-json") + 1])
                out_txt = Path(cmd[cmd.index("--out-txt") + 1])
                out_json.write_text(
                    json.dumps(
                        {
                            "segments": [
                                {"start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_txt.write_text("1\t0.0\t5.0\tpick up fabric from pile\n", encoding="utf-8")
                return {
                    "out_json": str(out_json),
                    "out_txt": str(out_txt),
                    "model": "chat_ui",
                    "attach_notes": ["video.mp4: attached (3.1 MB, settle=4.0s, mode=file_chooser)"],
                }

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_retry_input_missing",
                    video_file=video_file,
                    api_model="gemini-3.1-flash-lite-preview",
                )

            self.assertEqual(len(runtime_gemini_cfgs), 2)
            self.assertEqual(runtime_gemini_cfgs[1]["chat_web_connect_over_cdp_url"], "")
            self.assertEqual(runtime_gemini_cfgs[1]["chat_web_user_data_dir"], "/tmp/gemini_profile")
            self.assertTrue(result["chat_generation_recovered_via_retry"])

    def test_compare_retries_lighter_runtime_when_page_goto_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            cfg["gemini"]["chat_web_connect_over_cdp_url"] = "http://127.0.0.1:9222"
            cfg["gemini"]["chat_web_user_data_dir"] = "/tmp/gemini_profile"
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "adjust over guide fabric"}
            }
            runtime_gemini_cfgs = []

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                cfg_path = Path(cmd[cmd.index("--config") + 1])
                runtime_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                runtime_gemini_cfgs.append(runtime_cfg.get("gemini", {}))
                if len(runtime_gemini_cfgs) == 1:
                    raise RuntimeError(
                        'Page.goto: Timeout 60000ms exceeded. Call log: navigating to "https://gemini.google.com/app/x", waiting until "domcontentloaded"'
                    )
                out_json = Path(cmd[cmd.index("--out-json") + 1])
                out_txt = Path(cmd[cmd.index("--out-txt") + 1])
                out_json.write_text(
                    json.dumps(
                        {
                            "segments": [
                                {"start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_txt.write_text("1\t0.0\t5.0\tpick up fabric from pile\n", encoding="utf-8")
                return {
                    "out_json": str(out_json),
                    "out_txt": str(out_txt),
                    "model": "chat_ui",
                    "attach_notes": ["video.mp4: attached (3.1 MB, settle=4.0s, mode=file_chooser)"],
                }

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_retry_page_goto_timeout",
                    video_file=video_file,
                    api_model="gemini-3.1-flash-lite-preview",
                )

            self.assertEqual(len(runtime_gemini_cfgs), 2)
            self.assertEqual(runtime_gemini_cfgs[1]["chat_web_connect_over_cdp_url"], "")
            self.assertEqual(runtime_gemini_cfgs[1]["chat_web_user_data_dir"], "/tmp/gemini_profile")
            self.assertTrue(result["chat_generation_recovered_via_retry"])

    def test_compare_inherits_episode_active_model_when_episode_was_escalated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            cfg["gemini"]["model"] = "gemini-3.1-flash-lite-preview"
            cfg["gemini"]["gen3_fallback_models"] = ["gemini-3.1-pro-preview"]
            cfg["run"]["pre_submit_chat_compare_model"] = "gemini-3.1-flash-lite-preview"
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "adjust over guide fabric"}
            }
            seen_models = []

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                seen_models.append(cmd[cmd.index("--model") + 1])
                out_json = Path(cmd[cmd.index("--out-json") + 1])
                out_txt = Path(cmd[cmd.index("--out-txt") + 1])
                out_json.write_text(
                    json.dumps(
                        {
                            "segments": [
                                {"start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_txt.write_text("1\t0.0\t5.0\tpick up fabric from pile\n", encoding="utf-8")
                return {
                    "out_json": str(out_json),
                    "out_txt": str(out_txt),
                    "model": "gemini-3.1-pro-preview",
                    "attach_notes": ["video.mp4: attached (3.1 MB, settle=4.0s, mode=file_chooser)"],
                }

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_episode_model_pin",
                    video_file=video_file,
                    api_model="gemini-3.1-pro-preview",
                    episode_active_model="gemini-3.1-pro-preview",
                )

            self.assertEqual(seen_models, ["gemini-3.1-pro-preview"])
            self.assertEqual(result["chat_model"], "gemini-3.1-pro-preview")

    def test_compare_falls_back_to_next_gen3_model_on_availability_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            cfg["gemini"]["model"] = "gemini-3.1-flash-lite-preview"
            cfg["gemini"]["gen3_fallback_models"] = ["gemini-3.1-pro-preview"]
            cfg["run"]["pre_submit_chat_compare_model"] = "gemini-3.1-flash-lite-preview"
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "adjust over guide fabric"}
            }
            seen_models = []

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                model_name = cmd[cmd.index("--model") + 1]
                seen_models.append(model_name)
                if len(seen_models) == 1:
                    raise RuntimeError(
                        'Gemini HTTP 503: {"error":{"code":503,"message":"This model is currently experiencing high demand.","status":"UNAVAILABLE"}}'
                    )
                out_json = Path(cmd[cmd.index("--out-json") + 1])
                out_txt = Path(cmd[cmd.index("--out-txt") + 1])
                out_json.write_text(
                    json.dumps(
                        {
                            "segments": [
                                {"start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_txt.write_text("1\t0.0\t5.0\tpick up fabric from pile\n", encoding="utf-8")
                return {
                    "out_json": str(out_json),
                    "out_txt": str(out_txt),
                    "model": model_name,
                    "attach_notes": ["video.mp4: attached (3.1 MB, settle=4.0s, mode=file_chooser)"],
                }

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_compare_model_fallback",
                    video_file=video_file,
                    api_model="gemini-3.1-flash-lite-preview",
                )

            self.assertEqual(
                seen_models,
                ["gemini-3.1-flash-lite-preview", "gemini-3.1-pro-preview"],
            )
            self.assertEqual(result["chat_model"], "gemini-3.1-pro-preview")
            self.assertEqual(result["chat_model_fallbacks"][0]["to"], "gemini-3.1-pro-preview")

    def test_clean_chat_subprocess_text_removes_benign_node_deprecation_lines(self) -> None:
        text = "\n".join(
            [
                "(node:183088) [DEP0169] DeprecationWarning: `url.parse()` behavior is not standardized and prone to errors that have security implications.",
                "Use the WHATWG URL API instead. CVEs are not issued for `url.parse()` vulnerabilities.",
                "(Use `node --trace-deprecation ...` to show where the warning was created)",
                "RuntimeError: actual failure",
            ]
        )
        self.assertEqual(_clean_chat_subprocess_text(text), "RuntimeError: actual failure")

    def test_parse_chat_subprocess_stdout_reads_last_json_line(self) -> None:
        parsed = _parse_chat_subprocess_stdout("noise\n{\"ok\": true}\n")
        self.assertEqual(parsed, {"ok": True})

    def test_run_chat_generation_subprocess_accepts_json_stdout_when_stderr_has_only_node_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            script = tmp / "emit_json.py"
            script.write_text(
                "\n".join(
                    [
                        "import json, sys",
                        "sys.stderr.write('(node:1) [DEP0169] DeprecationWarning: `url.parse()` behavior is not standardized and prone to errors that have security implications.\\n')",
                        "sys.stderr.write('Use the WHATWG URL API instead. CVEs are not issued for `url.parse()` vulnerabilities.\\n')",
                        "sys.stderr.write('(Use `node --trace-deprecation ...` to show where the warning was created)\\n')",
                        "print(json.dumps({'ok': True, 'attach_notes': ['video.mp4: attached']}))",
                    ]
                ),
                encoding="utf-8",
            )
            result = _run_chat_generation_subprocess(cmd=[sys.executable, str(script)], max_wait_sec=30.0)
            self.assertTrue(result["ok"])

    def test_run_chat_generation_subprocess_prefers_cleaned_stderr_over_warning_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            script = tmp / "fail.py"
            script.write_text(
                "\n".join(
                    [
                        "import sys",
                        "sys.stderr.write('(node:1) [DEP0169] DeprecationWarning: `url.parse()` behavior is not standardized and prone to errors that have security implications.\\n')",
                        "sys.stderr.write('Use the WHATWG URL API instead. CVEs are not issued for `url.parse()` vulnerabilities.\\n')",
                        "sys.stderr.write('Actual compare failure\\n')",
                        "raise SystemExit(1)",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError) as ctx:
                _run_chat_generation_subprocess(cmd=[sys.executable, str(script)], max_wait_sec=30.0)
            self.assertEqual(str(ctx.exception), "Actual compare failure")

    def test_compare_passes_configured_max_wait_to_chat_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            cfg["run"]["pre_submit_chat_compare_max_wait_sec"] = 123.0
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
            }
            seen_waits = []

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                seen_waits.append(max_wait_sec)
                out_json = Path(cmd[cmd.index("--out-json") + 1])
                out_txt = Path(cmd[cmd.index("--out-txt") + 1])
                out_json.write_text(
                    json.dumps(
                        {
                            "segments": [
                                {"start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_txt.write_text("1\t0.0\t5.0\tpick up fabric from pile\n", encoding="utf-8")
                return {
                    "out_json": str(out_json),
                    "out_txt": str(out_txt),
                    "model": "chat_ui",
                    "attach_notes": ["video.mp4: attached (3.1 MB, settle=4.0s, mode=file_chooser)"],
                }

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_max_wait",
                    video_file=video_file,
                    api_model="gemini-3.1-flash-lite-preview",
                )

            self.assertTrue(result["executed"])
            self.assertEqual(seen_waits, [123.0])

    def test_compare_runtime_config_derives_chat_web_timeout_from_max_wait(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            cfg["run"]["pre_submit_chat_compare_max_wait_sec"] = 420.0
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
            }
            seen_chat_timeout = []

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                cfg_path = Path(cmd[cmd.index("--config") + 1])
                runtime_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                seen_chat_timeout.append(runtime_cfg.get("gemini", {}).get("chat_web_timeout_sec"))
                out_json = Path(cmd[cmd.index("--out-json") + 1])
                out_txt = Path(cmd[cmd.index("--out-txt") + 1])
                out_json.write_text(
                    json.dumps(
                        {
                            "segments": [
                                {"start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_txt.write_text("1\t0.0\t5.0\tpick up fabric from pile\n", encoding="utf-8")
                return {
                    "out_json": str(out_json),
                    "out_txt": str(out_txt),
                    "model": "chat_ui",
                    "attach_notes": ["video.mp4: attached (3.1 MB, settle=4.0s, mode=file_chooser)"],
                }

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_chat_timeout",
                    video_file=video_file,
                    api_model="gemini-3.1-flash-lite-preview",
                )

            self.assertTrue(result["executed"])
            self.assertEqual(seen_chat_timeout, [390.0])

    def test_resolve_pre_submit_chat_timeout_sec_prefers_explicit_config(self) -> None:
        cfg = {
            "run": {"pre_submit_chat_compare_max_wait_sec": 420.0},
            "gemini": {"chat_web_timeout_sec": 245.0},
        }
        self.assertEqual(_resolve_pre_submit_chat_timeout_sec(cfg), 245.0)

    def test_chat_attachment_network_fallback_accepts_unconfirmed_chip_when_segments_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_json = tmp / "chat.json"
            out_json.write_text(
                json.dumps(
                    {
                        "segments": [
                            {"start_sec": 0.0, "end_sec": 4.0, "label": "pick up fabric from pile"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                _chat_generation_attached_video(
                    {
                        "out_json": str(out_json),
                        "attach_notes": [
                            "video.mp4: skipped (attachment chip not confirmed near composer (reqs=3, resps=3))"
                        ],
                    }
                )
            )

    def test_chat_attachment_network_fallback_rejects_weak_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_json = tmp / "chat.json"
            out_json.write_text(json.dumps({"segments": []}), encoding="utf-8")
            self.assertFalse(
                _chat_generation_attached_video(
                    {
                        "out_json": str(out_json),
                        "attach_notes": [
                            "video.mp4: skipped (attachment chip not confirmed near composer (reqs=1, resps=1))"
                        ],
                    }
                )
            )

    def test_compare_passes_discord_seed_context_into_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            cfg["run"]["pre_submit_chat_compare_seed_discord_context"] = True
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
            }

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                runtime_cfg = Path(cmd[cmd.index("--config") + 1])
                runtime_payload = json.loads(runtime_cfg.read_text(encoding="utf-8"))
                gem_cfg = runtime_payload.get("gemini", {})
                self.assertTrue(gem_cfg.get("chat_web_seed_context_send_before_prompt"))
                seed_path = Path(str(gem_cfg.get("chat_web_seed_context_file", "")))
                self.assertTrue(seed_path.exists())
                self.assertIn("discord context", seed_path.read_text(encoding="utf-8").lower())
                out_json = Path(cmd[cmd.index("--out-json") + 1])
                out_txt = Path(cmd[cmd.index("--out-txt") + 1])
                out_json.write_text(
                    json.dumps(
                        {
                            "segments": [
                                {"start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_txt.write_text("1\t0.0\t5.0\tpick up fabric from pile\n", encoding="utf-8")
                return {
                    "out_json": str(out_json),
                    "out_txt": str(out_txt),
                    "model": "chat_ui",
                    "attach_notes": ["video.mp4: attached (3.1 MB, settle=4.0s, mode=drive_picker)"],
                }

            with patch(
                "src.solver.pre_submit_compare._build_chat_seed_context_text",
                return_value="Atlas Discord context\n- rule update",
            ), patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_seed_context",
                    video_file=video_file,
                    api_model="gemini-3.1-flash-lite-preview",
                )

            self.assertTrue(result["executed"])
            self.assertTrue(str(result.get("discord_context_path", "")).strip())
            self.assertTrue(Path(result["discord_context_path"]).exists())

    def test_required_compare_blocks_when_storage_state_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "missing_state.json"
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            cfg["run"]["pre_submit_chat_compare_required"] = True
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
            }
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]

            result = maybe_run_pre_submit_chat_compare(
                cfg=cfg,
                source_segments=source_segments,
                api_segment_plan=api_plan,
                task_id="ep_required_missing_state",
                video_file=video_file,
                api_model="gemini-3.1-flash-lite-preview",
            )

            self.assertFalse(result["executed"])
            self.assertEqual(result["decision"], "skipped_missing_chat_storage_state")
            self.assertTrue(result["block_apply"])

    def test_required_compare_allows_user_data_dir_fallback_when_storage_state_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "missing_state.json"
            user_data_dir = tmp / "gemini_chat_user_data"
            user_data_dir.mkdir()
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            cfg["run"]["pre_submit_chat_compare_required"] = True
            cfg["gemini"]["chat_web_user_data_dir"] = str(user_data_dir)
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric"}
            }
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                out_json = Path(cmd[cmd.index("--out-json") + 1])
                out_txt = Path(cmd[cmd.index("--out-txt") + 1])
                out_json.write_text(
                    json.dumps(
                        {
                            "segments": [
                                {"start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_txt.write_text("1\t0.0\t5.0\tpick up fabric from pile\n", encoding="utf-8")
                return {
                    "out_json": str(out_json),
                    "out_txt": str(out_txt),
                    "model": "chat_ui",
                    "attach_notes": ["video.mp4: attached"],
                }

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_required_user_data_dir",
                    video_file=video_file,
                    api_model="gemini-3.1-flash-lite-preview",
                )

            self.assertTrue(result["executed"])
            self.assertFalse(result.get("block_apply", False))
            self.assertNotEqual(result["decision"], "skipped_missing_chat_storage_state")

    def test_compare_persists_cost_ledger_to_task_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            cfg["run"]["pre_submit_chat_compare_model"] = "gemini-3.1-pro-preview"
            cfg["gemini"]["stage_models"] = {"compare_chat": "gemini-3.1-pro-preview"}
            cfg["economics"] = {
                "episode_expected_revenue_usd": 0.50,
                "target_cost_ratio": 0.15,
                "hard_cost_ratio": 0.20,
            }
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
            }
            task_state = {}
            persisted: list[dict] = []

            def fake_persist(cfg, task_id, task_state=None, **updates):
                merged = dict(task_state or {})
                merged.update(updates)
                persisted.append(dict(merged))
                if isinstance(task_state, dict):
                    task_state.clear()
                    task_state.update(merged)
                    return task_state
                return merged

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                out_json = Path(cmd[cmd.index("--out-json") + 1])
                out_txt = Path(cmd[cmd.index("--out-txt") + 1])
                out_json.write_text(
                    json.dumps(
                        {
                            "segments": [
                                {"start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_txt.write_text("1\t0.0\t5.0\tpick up fabric from pile\n", encoding="utf-8")
                return {
                    "out_json": str(out_json),
                    "out_txt": str(out_txt),
                    "model": "gemini-3.1-pro-preview",
                    "usage": {
                        "promptTokenCount": 3994,
                        "candidatesTokenCount": 178,
                        "totalTokenCount": 4172,
                    },
                    "attach_notes": ["video.mp4: attached (3.1 MB, settle=4.0s, mode=file_chooser)"],
                }

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess), patch(
                "src.solver.legacy_impl._persist_task_state_fields",
                side_effect=fake_persist,
            ):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_compare_cost",
                    video_file=video_file,
                    api_model="gemini-2.5-flash",
                    task_state=task_state,
                )

            self.assertTrue(result["executed"])
            self.assertEqual(task_state["episode_key_class_used"], "paid")
            self.assertIn("compare_chat", task_state["episode_cost_by_stage"])
            self.assertGreater(float(task_state["episode_estimated_cost_usd"]), 0.0)
            self.assertTrue(any("compare_chat" in row.get("episode_cost_by_stage", {}) for row in persisted))

    def test_compare_skips_retry_light_when_hard_cost_ratio_would_be_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            cfg["run"]["pre_submit_chat_compare_retry_on_failure"] = True
            cfg["run"]["pre_submit_chat_compare_block_on_chat_failure"] = True
            cfg["run"]["pre_submit_chat_compare_model"] = "gemini-3.1-pro-preview"
            cfg["gemini"]["stage_models"] = {"compare_chat": "gemini-3.1-pro-preview"}
            cfg["economics"] = {
                "episode_expected_revenue_usd": 0.05,
                "target_cost_ratio": 0.15,
                "hard_cost_ratio": 0.20,
                "enforce_cost_guards": True,
            }
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
            }
            task_state = {"episode_estimated_cost_usd": 0.0098}
            call_count = 0

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                nonlocal call_count
                call_count += 1
                raise RuntimeError("chat timed-label subprocess timed out")

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_compare_cost_guard",
                    video_file=video_file,
                    api_model="gemini-2.5-flash",
                    task_state=task_state,
                )

            self.assertEqual(call_count, 1)
            self.assertEqual(result["decision"], "keep_api_after_chat_failure_clean_api")
            self.assertFalse(result.get("block_apply", False))
            self.assertEqual(result["chat_generation_retry_skipped"], "cost_ratio_guard")

    def test_compare_retries_even_when_cost_ratio_would_be_exceeded_if_guards_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            storage_state = tmp / "gemini_chat_storage_state.json"
            storage_state.write_text("{}", encoding="utf-8")
            video_file = tmp / "video.mp4"
            video_file.write_bytes(b"00")
            cfg = self._base_cfg(tmp, storage_state)
            cfg["run"]["pre_submit_chat_compare_retry_on_failure"] = True
            cfg["run"]["pre_submit_chat_compare_block_on_chat_failure"] = True
            cfg["run"]["pre_submit_chat_compare_model"] = "gemini-3.1-pro-preview"
            cfg["gemini"]["stage_models"] = {"compare_chat": "gemini-3.1-pro-preview"}
            cfg["economics"] = {
                "episode_expected_revenue_usd": 0.05,
                "target_cost_ratio": 0.15,
                "hard_cost_ratio": 0.20,
            }
            source_segments = [
                {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "current_label": "pick up fabric"},
            ]
            api_plan = {
                1: {"segment_index": 1, "start_sec": 0.0, "end_sec": 5.0, "label": "adjust over guide fabric"}
            }
            runtime_gemini_cfgs = []

            def fake_subprocess(*, cmd, heartbeat=None, max_wait_sec=None):
                cfg_path = Path(cmd[cmd.index("--config") + 1])
                runtime_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                runtime_gemini_cfgs.append(runtime_cfg.get("gemini", {}))
                if len(runtime_gemini_cfgs) == 1:
                    raise RuntimeError("chat timed-label subprocess timed out")
                out_json = Path(cmd[cmd.index("--out-json") + 1])
                out_txt = Path(cmd[cmd.index("--out-txt") + 1])
                out_json.write_text(
                    json.dumps(
                        {
                            "segments": [
                                {"start_sec": 0.0, "end_sec": 5.0, "label": "pick up fabric from pile"}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                out_txt.write_text("1\t0.0\t5.0\tpick up fabric from pile\n", encoding="utf-8")
                return {
                    "out_json": str(out_json),
                    "out_txt": str(out_txt),
                    "model": "chat_ui",
                    "attach_notes": ["video.mp4: attached (3.1 MB, settle=4.0s, mode=file_chooser)"],
                }

            with patch("src.solver.pre_submit_compare._run_chat_generation_subprocess", side_effect=fake_subprocess):
                result = maybe_run_pre_submit_chat_compare(
                    cfg=cfg,
                    source_segments=source_segments,
                    api_segment_plan=api_plan,
                    task_id="ep_compare_cost_info_only",
                    video_file=video_file,
                    api_model="gemini-2.5-flash",
                )

            self.assertEqual(len(runtime_gemini_cfgs), 2)
            self.assertTrue(result["chat_generation_recovered_via_retry"])
            self.assertTrue(result["chat_generation_retry_cost_ratio_exceeded"])
            self.assertEqual(result["chat_generation_retry_projected_cost_usd"], 0.010124)
            self.assertNotIn("chat_generation_retry_skipped", result)

    def test_derive_split_repair_operations_for_overlong_source(self) -> None:
        source_segments = [
            {"segment_index": 1, "start_sec": 0.0, "end_sec": 18.0, "current_label": "shake bottle"},
            {"segment_index": 2, "start_sec": 18.0, "end_sec": 25.0, "current_label": "place bottle"},
        ]
        candidate_segments = [
            {"segment_index": 1, "start_sec": 0.0, "end_sec": 8.8, "label": "shake bottle"},
            {"segment_index": 2, "start_sec": 8.8, "end_sec": 18.0, "label": "shake bottle"},
            {"segment_index": 3, "start_sec": 18.0, "end_sec": 25.0, "label": "place bottle"},
        ]
        ops = _derive_split_repair_operations(
            source_segments=source_segments,
            candidate_segments=candidate_segments,
            max_duration_sec=10.0,
            epsilon_sec=0.35,
        )
        self.assertEqual(ops, [{"action": "split", "segment_index": 1}])


if __name__ == "__main__":
    unittest.main()
