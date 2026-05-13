import json
from pathlib import Path

import yaml

from src.solver.complex_test_harness import (
    _parse_timed_tsv,
    audit_episode_case,
    build_repair_queue,
    build_rotation_plan,
    load_episode_cases,
    run_complex_test,
)
from src.solver.account_scheduler import load_account_index


def test_parse_timed_tsv_reads_segments(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("1\t0.0\t3.5\tpick up fabric\n2\t3.5\t8.0\tplace fabric on table\n", encoding="utf-8")
    rows = _parse_timed_tsv(path)
    assert len(rows) == 2
    assert rows[1]["label"] == "place fabric on table"


def test_rotation_plan_honors_five_per_account(tmp_path: Path):
    index_path = tmp_path / "index.yaml"
    index_path.write_text(
        yaml.safe_dump(
            {
                "scheduler": {"episodes_per_account_per_turn": 5},
                "accounts": [
                    {"name": "a1", "enabled": True, "config": "a1.yaml"},
                    {"name": "a2", "enabled": True, "config": "a2.yaml"},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    index_cfg = load_account_index(index_path)
    cases = []
    for idx in range(12):
        cases.append(
            type(
                "Case",
                (),
                {"episode_id": f"ep{idx}", "review_status": "submitted", "quality_score": "", "notes": ""},
            )()
        )
    rotation = build_rotation_plan(index_cfg, cases)
    assert [row["account"] for row in rotation] == ["a1", "a2", "a1"]
    assert len(rotation[0]["episodes"]) == 5
    assert len(rotation[1]["episodes"]) == 5
    assert len(rotation[2]["episodes"]) == 2


def test_audit_episode_case_flags_overlong_segments(tmp_path: Path):
    current_path = tmp_path / "current.txt"
    update_path = tmp_path / "update.txt"
    current_path.write_text("1\t0.0\t4.0\tpick up fabric\n", encoding="utf-8")
    update_path.write_text("1\t0.0\t12.5\tpick up fabric from pile\n", encoding="utf-8")
    case = type(
        "EpisodeCaseProxy",
        (),
        {
            "episode_id": "ep1",
            "task_url": "https://example.test/task/ep1",
            "review_status": "submitted",
            "quality_score": "97%",
            "video_path": None,
            "current_text_path": current_path,
            "update_text_path": update_path,
            "validation_path": None,
            "task_state_path": None,
            "notes": "ok",
            "source_row": {},
        },
    )()
    cfg = {"run": {"max_segment_duration_sec": 10.0, "min_label_words": 2, "max_label_words": 20, "max_atomic_actions_per_label": 3}}
    report = audit_episode_case(cfg, case)
    assert report["counts"]["overlong_segments"] == 1
    assert report["quality_bucket"] == "excellent"
    assert report["ready_for_manual_review"] is False


def test_run_complex_test_generates_reports(tmp_path: Path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    current = outputs / "text_epa_current.txt"
    update = outputs / "text_epa_update.txt"
    current.write_text("1\t0.0\t4.0\tpick up fabric\n", encoding="utf-8")
    update.write_text("1\t0.0\t4.0\tpick up fabric from pile\n", encoding="utf-8")
    review_index = outputs / "episodes_review_index.json"
    review_index.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_id": "epa",
                        "task_url": "https://example.test/task/epa",
                        "review_status": "submitted",
                        "tier2_text_path": str(current),
                        "tier3_text_path": str(update),
                        "video_path": "",
                        "validation_path": "",
                        "related_files": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    feedback = outputs / "manual_feedback_snapshot.json"
    feedback.write_text(
        json.dumps({"episodes": [{"episode_id": "epa", "quality_score": "100%", "notes": "great"}]}),
        encoding="utf-8",
    )
    index_path = tmp_path / "index.yaml"
    index_path.write_text(
        yaml.safe_dump(
            {
                "scheduler": {"episodes_per_account_per_turn": 5},
                "accounts": [{"name": "acct1", "enabled": True, "config": "acct1.yaml"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    base_cfg = tmp_path / "base.yaml"
    base_cfg.write_text(yaml.safe_dump({"run": {"max_segment_duration_sec": 10.0}}, sort_keys=False), encoding="utf-8")
    payload = run_complex_test(
        index_path=index_path,
        base_cfg_path=base_cfg,
        review_index_path=review_index,
        manual_feedback_path=feedback,
        output_dir=tmp_path / "report",
        limit=10,
        pause_between_batches_sec=0.0,
    )
    assert payload["summary"]["episodes"] == 1
    assert Path(payload["json_path"]).exists()
    assert Path(payload["md_path"]).exists()
    assert Path(payload["repair_queue_json_path"]).exists()
    assert Path(payload["repair_queue_md_path"]).exists()


def test_build_repair_queue_prioritizes_overlong_episodes_first():
    queue = build_repair_queue(
        [
            {
                "turn": 1,
                "account": "acct1",
                "audits": [
                    {
                        "episode_id": "ep_ready",
                        "task_url": "",
                        "review_status": "submitted",
                        "quality_score": "100%",
                        "counts": {"overlong_segments": 0},
                        "policy_report": {"errors": [], "warnings": []},
                        "ready_for_manual_review": True,
                    },
                    {
                        "episode_id": "ep_overlong",
                        "task_url": "",
                        "review_status": "submitted",
                        "quality_score": "",
                        "counts": {"overlong_segments": 2},
                        "policy_report": {"errors": ["segment 1: duration 12.0s exceeds max 10.0s"], "warnings": []},
                        "ready_for_manual_review": False,
                    },
                    {
                        "episode_id": "ep_wording",
                        "task_url": "",
                        "review_status": "submitted",
                        "quality_score": "",
                        "counts": {"overlong_segments": 0},
                        "policy_report": {"errors": ["segment 2: label must start with an allowed action verb"], "warnings": []},
                        "ready_for_manual_review": False,
                    },
                ],
            }
        ]
    )
    assert queue[0]["episode_id"] == "ep_overlong"
    assert queue[0]["severity"] == "critical"
    assert queue[1]["episode_id"] == "ep_wording"
    assert queue[-1]["episode_id"] == "ep_ready"
