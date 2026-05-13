"""Offline complex-test harness for Atlas solver validation.

This harness replays previously saved episode artifacts without touching the
live Atlas platform. It is meant for high-complexity code testing: multi-account
rotation, quality checks, policy validation, and reporting.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.infra.solver_config import load_config
from src.rules.policy_gate import _validate_segment_plan_against_policy
from src.solver.account_scheduler import _enabled_accounts, load_account_index


_TSV_RE = re.compile(
    r"^\s*(?P<index>\d+)\t(?P<start>-?\d+(?:\.\d+)?)\t(?P<end>-?\d+(?:\.\d+)?)\t(?P<label>.*)\s*$"
)


@dataclass
class EpisodeCase:
    episode_id: str
    task_url: str
    review_status: str
    quality_score: str
    video_path: Optional[Path]
    current_text_path: Optional[Path]
    update_text_path: Optional[Path]
    validation_path: Optional[Path]
    task_state_path: Optional[Path]
    notes: str
    source_row: Dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_timed_tsv(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _TSV_RE.match(line)
        if not match:
            continue
        rows.append(
            {
                "segment_index": int(match.group("index")),
                "start_sec": float(match.group("start")),
                "end_sec": float(match.group("end")),
                "label": match.group("label").strip(),
            }
        )
    return rows


def _source_segments_from_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "segment_index": int(row["segment_index"]),
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "current_label": str(row["label"]),
            }
        )
    return out


def _segment_plan_from_rows(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        idx = int(row["segment_index"])
        out[idx] = {
            "segment_index": idx,
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
            "label": str(row["label"]),
        }
    return out


def _count_changed_segments(current_rows: List[Dict[str, Any]], update_rows: List[Dict[str, Any]]) -> int:
    current_map = {int(row["segment_index"]): str(row["label"]).strip() for row in current_rows}
    changed = 0
    for row in update_rows:
        if str(current_map.get(int(row["segment_index"]), "")).strip() != str(row["label"]).strip():
            changed += 1
    return changed


def _find_overlong_segments(rows: List[Dict[str, Any]], *, limit_sec: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        duration = float(row["end_sec"]) - float(row["start_sec"])
        if duration > limit_sec + 0.05:
            out.append(
                {
                    "segment_index": int(row["segment_index"]),
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "duration_sec": round(duration, 3),
                    "label": str(row["label"]),
                }
            )
    return out


def _quality_score_numeric(raw: str) -> Optional[float]:
    text = str(raw or "").strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _resolve_optional_path(raw: Any) -> Optional[Path]:
    value = str(raw or "").strip()
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _recommend_repair_action(
    *,
    overlong_count: int,
    errors: List[str],
    warnings: List[str],
) -> Tuple[str, str, int]:
    normalized_errors = [str(item or "").strip().lower() for item in errors if str(item or "").strip()]
    normalized_warnings = [str(item or "").strip().lower() for item in warnings if str(item or "").strip()]

    if overlong_count > 0:
        return (
            "critical",
            "split overlong segments first, then re-run policy validation and label wording review",
            100 + overlong_count,
        )
    if any("overlaps previous segment" in item or "start_sec is not monotonic" in item for item in normalized_errors):
        return (
            "high",
            "repair timestamp order/overlap before any wording changes",
            85,
        )
    if any(
        marker in item
        for item in normalized_errors
        for marker in (
            "must start with an allowed action verb",
            "forbidden verb",
            "narrative token",
            "disallowed tool term",
            "'guide' should not be used as an object descriptor",
            "'adjust over' phrasing is not semantically valid",
            "'place' missing explicit location",
            "'no action' must be standalone",
        )
    ):
        return (
            "medium",
            "rewrite labels for policy-safe verbs and object/location phrasing",
            60,
        )
    if normalized_errors:
        return (
            "medium",
            "review policy errors manually and repair affected segments",
            50,
        )
    if normalized_warnings:
        return (
            "low",
            "review timestamp drift and comparison warnings before approval",
            20,
        )
    return ("ready", "ready for manual review", 0)


def build_repair_queue(rotation_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    queue: List[Dict[str, Any]] = []
    for row in rotation_rows:
        account = str(row.get("account", "")).strip()
        turn = int(row.get("turn", 0) or 0)
        for audit in row.get("audits", []):
            errors = list(audit.get("policy_report", {}).get("errors", []) or [])
            warnings = list(audit.get("policy_report", {}).get("warnings", []) or [])
            overlong_count = int(audit.get("counts", {}).get("overlong_segments", 0) or 0)
            severity, action, priority_score = _recommend_repair_action(
                overlong_count=overlong_count,
                errors=errors,
                warnings=warnings,
            )
            quality = _quality_score_numeric(str(audit.get("quality_score", "") or ""))
            queue.append(
                {
                    "turn": turn,
                    "account": account,
                    "episode_id": str(audit.get("episode_id", "")).strip(),
                    "task_url": str(audit.get("task_url", "")).strip(),
                    "review_status": str(audit.get("review_status", "")).strip(),
                    "quality_score": str(audit.get("quality_score", "")).strip(),
                    "quality_numeric": quality,
                    "ready_for_manual_review": bool(audit.get("ready_for_manual_review")),
                    "severity": severity,
                    "priority_score": priority_score,
                    "recommended_action": action,
                    "policy_error_count": len(errors),
                    "policy_warning_count": len(warnings),
                    "overlong_segments": overlong_count,
                    "first_errors": errors[:5],
                    "first_warnings": warnings[:3],
                }
            )

    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3, "ready": 4}
    queue.sort(
        key=lambda item: (
            severity_rank.get(str(item.get("severity", "")), 99),
            -int(item.get("priority_score", 0) or 0),
            999 if item.get("quality_numeric") is None else float(item.get("quality_numeric")),
            str(item.get("episode_id", "")),
        )
    )
    return queue


def load_episode_cases(
    *,
    review_index_path: Path,
    manual_feedback_path: Optional[Path] = None,
    limit: int = 0,
) -> List[EpisodeCase]:
    review_index = _read_json(review_index_path)
    rows = review_index.get("episodes", [])
    if not isinstance(rows, list):
        return []

    quality_by_episode: Dict[str, Tuple[str, str]] = {}
    if manual_feedback_path is not None and manual_feedback_path.exists():
        manual_payload = _read_json(manual_feedback_path)
        manual_rows = manual_payload.get("episodes", [])
        if isinstance(manual_rows, list):
            for row in manual_rows:
                if not isinstance(row, dict):
                    continue
                episode_id = str(row.get("episode_id", "")).strip().lower()
                if not episode_id:
                    continue
                quality_by_episode[episode_id] = (
                    str(row.get("quality_score", "")).strip(),
                    str(row.get("notes", "")).strip(),
                )

    cases: List[EpisodeCase] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        episode_id = str(row.get("episode_id", "")).strip().lower()
        if not episode_id:
            continue
        current_path = _resolve_optional_path(row.get("tier2_text_path"))
        update_path = _resolve_optional_path(row.get("tier3_text_path"))
        video_path = _resolve_optional_path(row.get("video_path"))
        validation_path = _resolve_optional_path(row.get("validation_path"))
        task_state_path = None
        related_files = row.get("related_files", [])
        if isinstance(related_files, list):
            for item in related_files:
                candidate = _resolve_optional_path(item)
                if candidate is not None and candidate.name == f"task_state_{episode_id}.json":
                    task_state_path = candidate
                    break
        if current_path is None and update_path is None:
            continue
        quality_score, notes = quality_by_episode.get(episode_id, ("", ""))
        cases.append(
            EpisodeCase(
                episode_id=episode_id,
                task_url=str(row.get("task_url", "") or row.get("atlas_url", "")).strip(),
                review_status=str(row.get("review_status", "")).strip() or "unknown",
                quality_score=quality_score,
                video_path=video_path,
                current_text_path=current_path,
                update_text_path=update_path,
                validation_path=validation_path,
                task_state_path=task_state_path,
                notes=notes,
                source_row=row,
            )
        )

    def _case_sort_key(item: EpisodeCase) -> Tuple[int, float]:
        quality = _quality_score_numeric(item.quality_score)
        quality_rank = -1 if quality is None else int(math.floor(quality))
        return (0 if item.review_status in {"submitted", "reviewed"} else 1, -quality_rank)

    cases.sort(key=_case_sort_key)
    if limit > 0:
        cases = cases[:limit]
    return cases


def build_rotation_plan(index_cfg: Dict[str, Any], cases: List[EpisodeCase]) -> List[Dict[str, Any]]:
    accounts = _enabled_accounts(index_cfg)
    if not accounts:
        raise ValueError("No enabled accounts found.")
    scheduler = index_cfg.get("scheduler", {})
    per_turn = max(1, int(scheduler.get("episodes_per_account_per_turn", 5) or 5))

    assignments: List[Dict[str, Any]] = []
    case_iter = iter(cases)
    turn_no = 0
    while True:
        consumed_this_turn = 0
        turn_no += 1
        for account in accounts:
            account_name = str(account.get("name", "")).strip()
            batch: List[EpisodeCase] = []
            for _ in range(per_turn):
                try:
                    batch.append(next(case_iter))
                except StopIteration:
                    break
            if not batch:
                continue
            consumed_this_turn += len(batch)
            assignments.append(
                {
                    "turn": turn_no,
                    "account": account_name,
                    "episodes": [item.episode_id for item in batch],
                    "case_batch": batch,
                }
            )
        if consumed_this_turn <= 0:
            break
    return assignments


def audit_episode_case(cfg: Dict[str, Any], case: EpisodeCase) -> Dict[str, Any]:
    current_rows = _parse_timed_tsv(case.current_text_path)
    update_rows = _parse_timed_tsv(case.update_text_path)
    effective_rows = update_rows or current_rows
    source_segments = _source_segments_from_rows(current_rows or effective_rows)
    segment_plan = _segment_plan_from_rows(effective_rows)
    policy_report = _validate_segment_plan_against_policy(cfg, source_segments, segment_plan)
    existing_validation = _read_json(case.validation_path)
    task_state = _read_json(case.task_state_path)

    max_duration_sec = float(cfg.get("run", {}).get("max_segment_duration_sec", 10.0))
    overlong = _find_overlong_segments(effective_rows, limit_sec=max_duration_sec)
    changed_count = _count_changed_segments(current_rows, update_rows) if current_rows and update_rows else 0
    quality_numeric = _quality_score_numeric(case.quality_score)
    quality_bucket = (
        "excellent" if quality_numeric is not None and quality_numeric >= 95 else
        "acceptable" if quality_numeric is not None and quality_numeric >= 90 else
        "needs_review" if quality_numeric is not None else
        "unknown"
    )

    return {
        "episode_id": case.episode_id,
        "task_url": case.task_url,
        "review_status": case.review_status,
        "quality_score": case.quality_score,
        "quality_bucket": quality_bucket,
        "notes": case.notes,
        "paths": {
            "video": str(case.video_path) if case.video_path else "",
            "current": str(case.current_text_path) if case.current_text_path else "",
            "update": str(case.update_text_path) if case.update_text_path else "",
            "validation": str(case.validation_path) if case.validation_path else "",
            "task_state": str(case.task_state_path) if case.task_state_path else "",
        },
        "counts": {
            "current_segments": len(current_rows),
            "update_segments": len(update_rows),
            "effective_segments": len(effective_rows),
            "changed_segments": changed_count,
            "policy_errors": len(policy_report.get("errors", []) or []),
            "policy_warnings": len(policy_report.get("warnings", []) or []),
            "overlong_segments": len(overlong),
        },
        "policy_report": policy_report,
        "existing_validation": existing_validation,
        "task_state": task_state,
        "overlong_segments": overlong,
        "ready_for_manual_review": bool(policy_report.get("ok")) and len(overlong) == 0,
    }


def run_complex_test(
    *,
    index_path: Path,
    base_cfg_path: Path,
    review_index_path: Path,
    manual_feedback_path: Optional[Path],
    output_dir: Path,
    limit: int,
    pause_between_batches_sec: float,
) -> Dict[str, Any]:
    index_cfg = load_account_index(index_path)
    cfg = load_config(base_cfg_path)
    cases = load_episode_cases(
        review_index_path=review_index_path,
        manual_feedback_path=manual_feedback_path,
        limit=limit,
    )
    rotation = build_rotation_plan(index_cfg, cases)
    output_dir.mkdir(parents=True, exist_ok=True)

    account_rows: List[Dict[str, Any]] = []
    summary_counts = defaultdict(int)
    for slot in rotation:
        account_name = str(slot["account"])
        batch_cases: List[EpisodeCase] = list(slot["case_batch"])
        batch_audits = [audit_episode_case(cfg, case) for case in batch_cases]
        ready = sum(1 for item in batch_audits if item["ready_for_manual_review"])
        needs_review = len(batch_audits) - ready
        summary_counts["episodes"] += len(batch_audits)
        summary_counts["ready_for_manual_review"] += ready
        summary_counts["needs_review"] += needs_review
        account_rows.append(
            {
                "turn": int(slot["turn"]),
                "account": account_name,
                "episode_ids": [case.episode_id for case in batch_cases],
                "ready_for_manual_review": ready,
                "needs_review": needs_review,
                "audits": batch_audits,
            }
        )
        if pause_between_batches_sec > 0:
            time.sleep(pause_between_batches_sec)

    payload = {
        "generated_at_utc": _utc_now(),
        "mode": "complex_test_manual_review",
        "index_path": str(index_path),
        "base_cfg_path": str(base_cfg_path),
        "review_index_path": str(review_index_path),
        "manual_feedback_path": str(manual_feedback_path) if manual_feedback_path else "",
        "summary": dict(summary_counts),
        "rotation": account_rows,
    }
    repair_queue = build_repair_queue(account_rows)
    payload["repair_queue"] = repair_queue
    json_path = output_dir / "complex_test_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = output_dir / "complex_test_report.md"
    md_lines = [
        "# Complex Test Report",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        f"- Episodes audited: {summary_counts['episodes']}",
        f"- Ready for manual review: {summary_counts['ready_for_manual_review']}",
        f"- Needs review: {summary_counts['needs_review']}",
        "",
    ]
    for row in account_rows:
        md_lines.append(f"## Turn {row['turn']} - {row['account']}")
        md_lines.append("")
        for audit in row["audits"]:
            md_lines.append(
                f"- `{audit['episode_id']}` status={audit['review_status']} "
                f"quality={audit['quality_score'] or 'n/a'} "
                f"segments={audit['counts']['effective_segments']} "
                f"errors={audit['counts']['policy_errors']} "
                f"warnings={audit['counts']['policy_warnings']} "
                f"overlong={audit['counts']['overlong_segments']} "
                f"ready={audit['ready_for_manual_review']}"
            )
        md_lines.append("")
    md_path.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    queue_json_path = output_dir / "repair_queue.json"
    queue_json_path.write_text(json.dumps(repair_queue, ensure_ascii=False, indent=2), encoding="utf-8")
    queue_md_path = output_dir / "repair_queue.md"
    queue_md_lines = [
        "# Repair Queue",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        f"- Queue items: {len(repair_queue)}",
        "",
    ]
    for item in repair_queue:
        queue_md_lines.append(
            f"- [{item['severity']}] `{item['episode_id']}` account={item['account']} "
            f"errors={item['policy_error_count']} warnings={item['policy_warning_count']} "
            f"overlong={item['overlong_segments']} action={item['recommended_action']}"
        )
    queue_md_path.write_text("\n".join(queue_md_lines).strip() + "\n", encoding="utf-8")

    payload["json_path"] = str(json_path)
    payload["md_path"] = str(md_path)
    payload["repair_queue_json_path"] = str(queue_json_path)
    payload["repair_queue_md_path"] = str(queue_md_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline complex Atlas test harness (manual-review only)")
    parser.add_argument("--index", default="configs/accounts/index.yaml", help="Account index YAML path")
    parser.add_argument("--config", default="configs/production_hetzner.yaml", help="Base config YAML path")
    parser.add_argument("--review-index", default="outputs/episodes_review_index.json", help="Episode review index JSON path")
    parser.add_argument(
        "--manual-feedback",
        default="outputs/gemini_memory_sources/manual_feedback_snapshot.json",
        help="Manual feedback snapshot JSON path",
    )
    parser.add_argument("--output-dir", default="outputs/complex_test/latest", help="Output directory")
    parser.add_argument("--limit", type=int, default=20, help="Maximum episode cases to audit")
    parser.add_argument("--pause-between-batches-sec", type=float, default=0.0, help="Optional pause between account batches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_complex_test(
        index_path=Path(args.index),
        base_cfg_path=Path(args.config),
        review_index_path=Path(args.review_index),
        manual_feedback_path=Path(args.manual_feedback) if str(args.manual_feedback).strip() else None,
        output_dir=Path(args.output_dir),
        limit=max(1, int(args.limit)),
        pause_between_batches_sec=max(0.0, float(args.pause_between_batches_sec)),
    )
    print(f"[complex-test] episodes={payload['summary'].get('episodes', 0)}")
    print(f"[complex-test] json={payload['json_path']}")
    print(f"[complex-test] markdown={payload['md_path']}")
    print(f"[complex-test] repair-queue-json={payload['repair_queue_json_path']}")
    print(f"[complex-test] repair-queue-md={payload['repair_queue_md_path']}")


if __name__ == "__main__":
    main()
