"""Optional pre-submit Chat UI comparison gate for Atlas solver runs."""

from __future__ import annotations

import copy
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import prompts
import validator
import yaml

from atlas_triplet_compare import parse_timed_segments_payload, segments_to_timed_text
from src.infra.gemini_economics import (
    build_episode_cost_updates,
    cost_guard_enforcement_enabled,
    estimate_cost_from_usage,
    estimate_cost_usd,
    resolve_stage_model,
    would_exceed_ratio_cap,
)
from src.infra.solver_config import _cfg_get, _ordered_gen3_gemini_models
from src.rules.policy_gate import _validate_segment_plan_against_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_CONTEXT_PATH = PROJECT_ROOT / "prompts" / "generated_policy_context.txt"
DEFAULT_LIVE_POLICY_PATH = PROJECT_ROOT / "data" / "gemini_policy_discord_live.txt"
DEFAULT_HARVEST_DIR = PROJECT_ROOT / "outputs" / "discord_exports"
DEFAULT_REVIEW_INDEX_PATH = PROJECT_ROOT / "outputs" / "episodes_review_index.json"
DEFAULT_USER_MEMORY_DIR = PROJECT_ROOT / "outputs" / "gemini_memory_sources"
DEFAULT_MANUAL_FEEDBACK_PATH = DEFAULT_USER_MEMORY_DIR / "manual_feedback_snapshot.json"


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _load_json_file(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _latest_discord_export_path() -> Optional[Path]:
    if not DEFAULT_HARVEST_DIR.exists():
        return None
    candidates = sorted(
        DEFAULT_HARVEST_DIR.glob("harvest_*.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _shorten_text(text: str, limit: int) -> str:
    raw = str(text or "").strip()
    if limit <= 0 or len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 1)].rstrip() + "…"


def _load_manual_feedback_snapshot() -> List[Dict[str, Any]]:
    payload = _load_json_file(DEFAULT_MANUAL_FEEDBACK_PATH)
    rows = payload.get("episodes", [])
    return rows if isinstance(rows, list) else []


def _build_feedback_review_memory_section(*, max_items: int = 8, max_chars_per_item: int = 900) -> str:
    sections: List[str] = []

    manual_rows = _load_manual_feedback_snapshot()
    rendered_manual: List[str] = []
    for row in manual_rows[: max(0, int(max_items))]:
        if not isinstance(row, dict):
            continue
        episode_id = str(row.get("episode_id", "") or "").strip().lower()
        if not episode_id:
            continue
        score = str(row.get("quality_score", "") or "").strip()
        notes = _shorten_text(str(row.get("notes", "") or ""), max_chars_per_item)
        lines = [f"- episode={episode_id} quality={score or 'unknown'}"]
        if notes:
            lines.append(f"  notes: {notes}")
        feedback_url = str(row.get("feedback_url", "") or "").strip()
        if feedback_url:
            lines.append(f"  feedback_url: {feedback_url}")
        rendered_manual.append("\n".join(lines))
    if rendered_manual:
        sections.append(
            "[Confirmed Atlas Review Outcomes]\n"
            "Operator-confirmed quality/review outcomes that should influence future decisions.\n"
            + "\n".join(rendered_manual)
        )

    review_index = _load_json_file(DEFAULT_REVIEW_INDEX_PATH)
    episodes = review_index.get("episodes", [])
    rendered_index: List[str] = []
    if isinstance(episodes, list):
        interesting: List[Dict[str, Any]] = []
        for row in episodes:
            if not isinstance(row, dict):
                continue
            review_status = str(row.get("review_status", "") or "").strip().lower()
            disputes_count = int(row.get("disputes_count", 0) or 0)
            if review_status not in {"", "unknown", "pending", "awaiting_t3"} or disputes_count > 0:
                interesting.append(row)
        for row in interesting[: max(0, int(max_items))]:
            episode_id = str(row.get("episode_id", "") or "").strip().lower()
            if not episode_id:
                continue
            review_status = str(row.get("review_status", "") or "").strip() or "unknown"
            disputes_count = int(row.get("disputes_count", 0) or 0)
            lines = [f"- episode={episode_id} review_status={review_status} disputes={disputes_count}"]
            validation = row.get("validation", {})
            if isinstance(validation, dict):
                score = validation.get("score")
                if score not in (None, ""):
                    lines.append(f"  validator_score: {score}")
            tier3_text = _shorten_text(str(row.get("tier3_text", "") or ""), max_chars_per_item)
            if tier3_text:
                lines.append(f"  tier3_text: {tier3_text}")
            rendered_index.append("\n".join(lines))
    if rendered_index:
        sections.append("[Historical Review Index]\n" + "\n".join(rendered_index))

    return "\n\n".join(section.strip() for section in sections if section.strip()).strip()


def _build_drive_workspace_memory_section(*, max_chars: int = 2800) -> str:
    drive_root = DEFAULT_USER_MEMORY_DIR / "drive_imports"
    if not drive_root.exists():
        return ""

    sections: List[str] = []

    conv_candidates = [
        drive_root / "gemini_export_a" / "conversation_1773179135.txt",
        drive_root / "gemini_export_b" / "conversation_1773179135.txt",
    ]
    conversation_payload: Dict[str, Any] = {}
    for candidate in conv_candidates:
        if not candidate.exists():
            continue
        try:
            conversation_payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            conversation_payload = {}
        if conversation_payload:
            break
    if conversation_payload:
        title = str(conversation_payload.get("title", "") or "").strip()
        turns = conversation_payload.get("conversation_turns", [])
        lines: List[str] = []
        if title:
            lines.append(f"title: {title}")
        if isinstance(turns, list):
            for row in turns[:4]:
                if not isinstance(row, dict):
                    continue
                user_turn = row.get("user_turn", {})
                if isinstance(user_turn, dict):
                    prompt = _shorten_text(str(user_turn.get("prompt", "") or ""), max_chars)
                    if prompt:
                        lines.append(f"user_prompt: {prompt}")
                system_turn = row.get("system_turn", {})
                if isinstance(system_turn, dict):
                    text_blocks = system_turn.get("text", [])
                    if isinstance(text_blocks, list):
                        for block in text_blocks[:1]:
                            if isinstance(block, dict):
                                data = _shorten_text(str(block.get("data", "") or ""), max_chars)
                                if data:
                                    lines.append(f"prior_gemini_summary: {data}")
        if lines:
            sections.append("[User Workspace / Prior Gemini Memory]\n" + "\n".join(lines))

    gems_path = drive_root / "gemini_export_a" / "gemini_gems_data.html"
    scheduled_path = drive_root / "gemini_export_a" / "gemini_scheduled_actions_data.html"
    gems_text = _read_text_file(gems_path)
    scheduled_text = _read_text_file(scheduled_path)
    sections.append(
        "[Gemini Workspace Export Status]\n"
        + "\n".join(
            [
                f"gemini_gems_export: {'empty' if gems_text in {'', '<div></div>'} else 'present'}",
                f"gemini_scheduled_actions_export: {'empty' if scheduled_text in {'', '<div></div>'} else 'present'}",
            ]
        )
    )

    if (drive_root / "takeout" / "archive_browser.html").exists():
        sections.append(
            "[Takeout Archive Inventory]\n"
            "A Google Takeout archive browser export is available for this workspace snapshot."
        )

    return "\n\n".join(section.strip() for section in sections if section.strip()).strip()


def _build_chat_memory_bundle_text(
    *,
    cfg: Dict[str, Any],
    base_primer_text: str,
    max_feedback_items: int,
    max_feedback_chars: int,
    max_drive_chars: int,
) -> str:
    sections: List[str] = []

    base = str(base_primer_text or "").strip()
    if base:
        sections.append("[Base Project Memory]\n" + base)

    drive_section = _build_drive_workspace_memory_section(max_chars=max_drive_chars)
    if drive_section:
        sections.append(drive_section)

    feedback_section = _build_feedback_review_memory_section(
        max_items=max_feedback_items,
        max_chars_per_item=max_feedback_chars,
    )
    if feedback_section:
        sections.append(feedback_section)

    policy_context = _read_text_file(DEFAULT_POLICY_CONTEXT_PATH)
    if policy_context and "[Canonical Policy Context]" not in "\n\n".join(sections):
        sections.append("[Canonical Policy Context]\n" + policy_context)

    live_policy = _read_text_file(DEFAULT_LIVE_POLICY_PATH)
    if live_policy and "[Live Discord Rule Sync]" not in "\n\n".join(sections):
        sections.append("[Live Discord Rule Sync]\n" + live_policy)

    if not sections:
        return ""

    header = (
        "Atlas persistent memory bundle for Gemini Chat.\n"
        "Read this once before the video. Treat it as stable project memory, operator context, "
        "review lessons, and policy guidance.\n"
        "If older thread assumptions conflict with this bundle, prefer this bundle.\n"
    )
    return header.strip() + "\n\n" + "\n\n".join(section.strip() for section in sections if section.strip())


def _build_chat_seed_context_text(
    *,
    cfg: Dict[str, Any],
    max_messages: int,
    max_chars_per_message: int,
) -> str:
    sections: List[str] = []

    policy_context = _read_text_file(DEFAULT_POLICY_CONTEXT_PATH)
    if policy_context:
        sections.append("[Canonical Policy Context]\n" + policy_context)

    live_policy = _read_text_file(DEFAULT_LIVE_POLICY_PATH)
    if live_policy:
        sections.append("[Live Discord Rule Sync]\n" + live_policy)

    harvest_path = _latest_discord_export_path()
    harvest_payload = _load_json_file(harvest_path) if harvest_path is not None else {}
    if harvest_payload:
        metadata_lines = [
            f"source_mode: {str(harvest_payload.get('source_mode', '') or '').strip() or 'unknown'}",
            f"scan_window: {str(harvest_payload.get('start_date', '') or '').strip()} -> {str(harvest_payload.get('end_date', '') or '').strip()}",
            f"channel_count: {int(harvest_payload.get('channel_count', 0) or 0)}",
            f"message_count: {int(harvest_payload.get('message_count', 0) or 0)}",
        ]
        sections.append("[Discord Harvest Metadata]\n" + "\n".join(metadata_lines))

        messages = harvest_payload.get("messages", [])
        rendered_messages: List[str] = []
        if isinstance(messages, list):
            for row in messages[: max(0, int(max_messages))]:
                if not isinstance(row, dict):
                    continue
                content = str(row.get("content", "") or "").strip()
                if max_chars_per_message > 0 and len(content) > max_chars_per_message:
                    content = content[: max_chars_per_message - 1].rstrip() + "…"
                rendered = (
                    f"- [{str(row.get('timestamp', '') or '').strip()}] "
                    f"{str(row.get('channel', '') or '').strip()} "
                    f"{str(((row.get('author') or {}) if isinstance(row.get('author'), dict) else {}).get('username', '') or '').strip()}: "
                    f"{content}"
                ).strip()
                attachments = row.get("attachments", [])
                if isinstance(attachments, list) and attachments:
                    rendered += f"\n  attachments: {', '.join(str(v or '').strip() for v in attachments[:3] if str(v or '').strip())}"
                link = str(row.get("message_link", "") or "").strip()
                if link:
                    rendered += f"\n  link: {link}"
                if rendered:
                    rendered_messages.append(rendered)
        if rendered_messages:
            sections.append("[Visible Discord Messages]\n" + "\n".join(rendered_messages))

    if not sections:
        return ""

    header = (
        "Atlas Discord context for Gemini Chat.\n"
        "Treat this as authoritative supplemental policy + recent message context for Atlas annotation work.\n"
        "If older thread assumptions conflict with this context, prefer this context.\n"
    )
    return header.strip() + "\n\n" + "\n\n".join(section.strip() for section in sections if section.strip())


def _sorted_source_segments(source_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(list(source_segments), key=lambda seg: int(seg.get("segment_index", 0) or 0))


def _source_segments_to_draft_text(source_segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for seg in _sorted_source_segments(source_segments):
        idx = int(seg.get("segment_index", 0) or 0)
        start = float(seg.get("start_sec", 0.0) or 0.0)
        end = float(seg.get("end_sec", start) or start)
        lines.append(
            f"{idx}\t{start:.3f}\t{end:.3f}\t{str(seg.get('current_label', '') or '').strip()}"
        )
    return "\n".join(lines).strip()


def _plan_to_segments(plan: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx in sorted(plan):
        item = plan[idx]
        start = float(item.get("start_sec", 0.0) or 0.0)
        end = float(item.get("end_sec", start) or start)
        out.append(
            {
                "segment_index": int(item.get("segment_index", idx) or idx),
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(max(0.0, end - start), 3),
                "label": str(item.get("label", "") or "").strip(),
            }
        )
    return out


def _segments_to_plan(segments: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for idx, seg in enumerate(segments, 1):
        start = float(seg.get("start_sec", 0.0) or 0.0)
        end = float(seg.get("end_sec", start) or start)
        out[idx] = {
            "segment_index": idx,
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "label": str(seg.get("label", "") or "").strip(),
        }
    return out


def _make_validator_annotation(episode_id: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    duration = max((float(seg.get("end_sec", 0.0) or 0.0) for seg in segments), default=0.0)
    return {
        "episode_id": episode_id,
        "video_duration_sec": round(duration, 3),
        "segments": copy.deepcopy(segments),
    }


def _segment_error_count(report: Dict[str, Any]) -> int:
    count = 0
    for item in report.get("segment_reports", []) or []:
        if isinstance(item, dict):
            count += len(item.get("errors", []) or [])
    return count


def _score_validator_report(report: Dict[str, Any]) -> int:
    score = 100
    score -= 25 * len(report.get("episode_errors", []) or [])
    score -= 10 * len(report.get("major_fail_triggers", []) or [])
    score -= 2 * _segment_error_count(report)
    score -= len(report.get("episode_warnings", []) or [])
    return max(0, score)


def _validator_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": bool(report.get("ok", False)),
        "score": _score_validator_report(report),
        "episode_errors": report.get("episode_errors", []) or [],
        "major_fail_triggers": report.get("major_fail_triggers", []) or [],
        "episode_warnings": report.get("episode_warnings", []) or [],
        "segment_error_count": _segment_error_count(report),
    }


def _same_timeline_as_source(
    source_segments: List[Dict[str, Any]],
    candidate_segments: List[Dict[str, Any]],
    *,
    epsilon_sec: float,
) -> bool:
    source_ordered = _sorted_source_segments(source_segments)
    candidate_ordered = sorted(candidate_segments, key=lambda seg: float(seg.get("start_sec", 0.0) or 0.0))
    if len(source_ordered) != len(candidate_ordered):
        return False
    for src, cand in zip(source_ordered, candidate_ordered):
        src_start = float(src.get("start_sec", 0.0) or 0.0)
        src_end = float(src.get("end_sec", src_start) or src_start)
        cand_start = float(cand.get("start_sec", 0.0) or 0.0)
        cand_end = float(cand.get("end_sec", cand_start) or cand_start)
        if abs(src_start - cand_start) > epsilon_sec or abs(src_end - cand_end) > epsilon_sec:
            return False
    return True


def _derive_split_repair_operations(
    *,
    source_segments: List[Dict[str, Any]],
    candidate_segments: List[Dict[str, Any]],
    max_duration_sec: float,
    epsilon_sec: float,
) -> List[Dict[str, Any]]:
    if max_duration_sec <= 0:
        return []
    candidate_ordered = sorted(candidate_segments, key=lambda seg: float(seg.get("start_sec", 0.0) or 0.0))
    out: List[Dict[str, Any]] = []
    for src in _sorted_source_segments(source_segments):
        idx = int(src.get("segment_index", 0) or 0)
        start = float(src.get("start_sec", 0.0) or 0.0)
        end = float(src.get("end_sec", start) or start)
        duration = max(0.0, end - start)
        if idx <= 0 or duration <= max_duration_sec + epsilon_sec:
            continue
        covered = 0
        for cand in candidate_ordered:
            cand_start = float(cand.get("start_sec", 0.0) or 0.0)
            cand_end = float(cand.get("end_sec", cand_start) or cand_start)
            if cand_end <= start + epsilon_sec:
                continue
            if cand_start >= end - epsilon_sec:
                break
            if cand_start >= start - epsilon_sec and cand_end <= end + epsilon_sec:
                covered += 1
        if covered >= 2:
            out.append({"action": "split", "segment_index": idx})
    out.sort(key=lambda item: -int(item.get("segment_index", 0) or 0))
    return out


def _evaluate_candidate(
    *,
    name: str,
    episode_id: str,
    cfg: Dict[str, Any],
    source_segments: List[Dict[str, Any]],
    plan: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    segments = _plan_to_segments(plan)
    policy_report = _validate_segment_plan_against_policy(cfg, source_segments, plan)
    prompts.refresh_policy_assets()
    validator.refresh_policy_constraints()
    validator_report = validator.validate_episode(_make_validator_annotation(episode_id, segments))
    return {
        "name": name,
        "segments": segments,
        "timed_text": segments_to_timed_text(segments),
        "policy_report": policy_report,
        "policy_error_count": len(policy_report.get("errors", []) or []),
        "policy_warning_count": len(policy_report.get("warnings", []) or []),
        "validator_report": validator_report,
        "validator_summary": _validator_summary(validator_report),
    }


def _candidate_rank(summary: Dict[str, Any]) -> Tuple[int, int, int, int]:
    validator_summary = summary.get("validator_summary", {}) or {}
    return (
        int(summary.get("policy_error_count", 0) or 0),
        -int(validator_summary.get("score", 0) or 0),
        int(summary.get("policy_warning_count", 0) or 0),
        int(validator_summary.get("segment_error_count", 0) or 0),
    )


def _chat_generation_attached_video(chat_result: Dict[str, Any]) -> bool:
    notes = chat_result.get("attach_notes", [])
    if not isinstance(notes, list):
        return False
    for note in notes:
        text = str(note or "").strip().lower()
        if "attached" in text and "skipped" not in text:
            return True
    for note in notes:
        text = str(note or "").strip().lower()
        if "attachment chip not confirmed near composer" not in text:
            continue
        match = re.search(r"reqs=(\d+),\s*resps=(\d+)", text)
        if not match:
            continue
        try:
            reqs = int(match.group(1))
            resps = int(match.group(2))
        except Exception:
            continue
        if reqs < 2 or resps < 2:
            continue
        out_json = Path(str(chat_result.get("out_json", "") or "").strip())
        if not out_json.exists():
            continue
        try:
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            segments = parse_timed_segments_payload(payload)
        except Exception:
            segments = []
        if segments:
            return True
    return False


def _blocked_skip_payload(*, decision: str, block: bool, reason: str = "", **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"executed": False, "decision": decision}
    if block:
        payload["block_apply"] = True
        payload["block_reason"] = reason or f"Pre-submit compare blocked apply: {decision}"
    payload.update(extra)
    return payload


def _write_markdown_report(*, report_path: Path, compare_payload: Dict[str, Any]) -> None:
    api_summary = compare_payload.get("api_summary", {}) or {}
    chat_summary = compare_payload.get("chat_summary", {}) or {}
    lines = [
        f"# Pre-Submit Compare: {compare_payload.get('episode_id', 'episode')}",
        "",
        f"- Winner: `{compare_payload.get('winner', 'api')}`",
        f"- Decision: `{compare_payload.get('decision', 'keep_api')}`",
        f"- Chat same timeline as source: `{compare_payload.get('chat_same_timeline', False)}`",
        f"- Chat generation executed: `{compare_payload.get('executed', False)}`",
        "",
        "## API Summary",
        f"- Policy errors: `{api_summary.get('policy_error_count', 0)}`",
        f"- Policy warnings: `{api_summary.get('policy_warning_count', 0)}`",
        f"- Validator score: `{(api_summary.get('validator_summary', {}) or {}).get('score', 0)}`",
        "",
        "## Chat Summary",
        f"- Policy errors: `{chat_summary.get('policy_error_count', 0)}`",
        f"- Policy warnings: `{chat_summary.get('policy_warning_count', 0)}`",
        f"- Validator score: `{(chat_summary.get('validator_summary', {}) or {}).get('score', 0)}`",
        "",
        "## API Output",
        "```text",
        str(api_summary.get("timed_text", "") or "").strip(),
        "```",
        "",
        "## Chat Output",
        "```text",
        str(chat_summary.get("timed_text", "") or "").strip(),
        "```",
    ]
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


_BENIGN_CHAT_SUBPROCESS_LINE_MARKERS: Tuple[str, ...] = (
    "[DEP0169] DeprecationWarning:",
    "`url.parse()` behavior is not standardized",
    "CVEs are not issued for `url.parse()` vulnerabilities.",
    "Use the WHATWG URL API instead.",
    "(Use `node --trace-deprecation ...` to show where the warning was created)",
)


def _clean_chat_subprocess_text(text: str) -> str:
    lines: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = str(raw_line or "")
        if any(marker in line for marker in _BENIGN_CHAT_SUBPROCESS_LINE_MARKERS):
            continue
        if line.strip():
            lines.append(line)
    return "\n".join(lines).strip()


def _parse_chat_subprocess_stdout(stdout_text: str) -> Optional[Dict[str, Any]]:
    text = str(stdout_text or "").strip()
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        parsed = json.loads(lines[-1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _run_chat_generation_subprocess(
    *,
    cmd: List[str],
    heartbeat: Optional[Callable[[], None]] = None,
    heartbeat_sec: float = 10.0,
    max_wait_sec: float = 300.0,
) -> Dict[str, Any]:
    env = dict(os.environ)
    env.setdefault("NODE_NO_WARNINGS", "1")
    popen_kwargs: Dict[str, Any] = {
        "cwd": str(Path.cwd()),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "env": env,
    }
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if creationflags:
            popen_kwargs["creationflags"] = creationflags
    else:
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        cmd,
        **popen_kwargs,
    )
    started_at = time.time()
    while proc.poll() is None:
        if heartbeat is not None:
            try:
                heartbeat()
            except Exception:
                pass
        if time.time() - started_at >= max_wait_sec:
            _terminate_chat_subprocess_tree(proc)
            stdout_text, stderr_text = proc.communicate()
            stderr_clean = _clean_chat_subprocess_text(stderr_text)
            stdout_clean = (stdout_text or "").strip()
            detail = stderr_clean or stdout_clean or "chat timed-label subprocess timed out"
            raise RuntimeError(detail)
        time.sleep(max(1.0, float(heartbeat_sec)))
    stdout_text, stderr_text = proc.communicate()
    parsed_stdout = _parse_chat_subprocess_stdout(stdout_text)
    if proc.returncode != 0:
        stderr = _clean_chat_subprocess_text(stderr_text)
        stdout = (stdout_text or "").strip()
        if parsed_stdout is not None and not stderr:
            return parsed_stdout
        detail = stderr or stdout or f"subprocess exit {proc.returncode}"
        raise RuntimeError(detail)
    if parsed_stdout is not None:
        return parsed_stdout
    stdout_text = (stdout_text or "").strip()
    if not stdout_text:
        raise RuntimeError("chat timed-label subprocess returned empty stdout")
    raise RuntimeError("chat timed-label subprocess did not return parseable JSON on stdout")


def _terminate_chat_subprocess_tree(proc: subprocess.Popen[str]) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _is_chat_compare_availability_error_text(text: str) -> bool:
    body = str(text or "").strip().lower()
    if not body:
        return False
    if "high demand" in body or "currently experiencing high demand" in body:
        return True
    if "status\": \"unavailable" in body or "status': 'unavailable" in body:
        return True
    return "503" in body and "unavailable" in body


def _runtime_model_overrides(model_name: str) -> Dict[str, Any]:
    selected = str(model_name or "").strip()
    return {
        "model": selected,
        "chat_timed_model": selected,
        "timed_labels_model": selected,
        "compare_model": selected,
        "triplet_compare_model": selected,
        "chat_timed_fallback_model": selected,
        "timed_labels_fallback_model": selected,
        "compare_fallback_model": selected,
        "triplet_compare_fallback_model": selected,
        "gen3_fallback_models": [],
    }


def _resolve_pre_submit_chat_timeout_sec(cfg: Dict[str, Any]) -> float:
    configured = float(_cfg_get(cfg, "gemini.chat_web_timeout_sec", 0) or 0)
    if configured > 0:
        return max(20.0, configured)
    max_wait_sec = float(_cfg_get(cfg, "run.pre_submit_chat_compare_max_wait_sec", 300.0) or 300.0)
    # Keep the inner Chat UI timeout slightly below the outer subprocess watchdog.
    derived = max_wait_sec - 30.0
    return max(120.0, min(max_wait_sec - 5.0, derived))


def _resolve_pre_submit_retry_light_max_wait_sec(cfg: Dict[str, Any]) -> float:
    configured = float(_cfg_get(cfg, "run.pre_submit_chat_compare_retry_light_max_wait_sec", 0) or 0)
    if configured > 0:
        return max(60.0, configured)
    primary = float(_cfg_get(cfg, "run.pre_submit_chat_compare_max_wait_sec", 300.0) or 300.0)
    derived = min(240.0, max(120.0, primary / 2.0))
    return max(60.0, min(primary, derived))


def _resolve_pre_submit_retry_light_chat_timeout_sec(cfg: Dict[str, Any]) -> float:
    configured = float(_cfg_get(cfg, "gemini.chat_web_retry_light_timeout_sec", 0) or 0)
    if configured > 0:
        return max(20.0, configured)
    max_wait_sec = _resolve_pre_submit_retry_light_max_wait_sec(cfg)
    derived = max_wait_sec - 20.0
    return max(90.0, min(max_wait_sec - 5.0, derived))


def _retry_light_gemini_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tier2_max_chars = max(
        1000,
        int(_cfg_get(cfg, "run.pre_submit_chat_compare_retry_light_tier2_max_chars", 6000) or 6000),
    )
    return {
        "chat_web_timeout_sec": _resolve_pre_submit_retry_light_chat_timeout_sec(cfg),
        "chat_web_prefer_drive_picker": False,
        "chat_web_force_clean_thread": False,
        "chat_web_clean_thread_fallback_enabled": False,
        "chat_web_connect_over_cdp_url": "",
        "chat_web_seed_context_send_before_prompt": False,
        "chat_web_seed_context_file": "",
        "chat_web_seed_context_text": "",
        "chat_web_memory_primer_file": "",
        "context_text": "",
        "context_file": "",
        "timed_labels_context_text": "",
        "timed_labels_context_file": "",
        "timed_labels_system_instruction_text": "",
        "timed_labels_system_instruction_file": "",
        "system_instruction_text": "",
        "system_instruction_file": "",
        "timed_labels_tier2_max_chars": tier2_max_chars,
        "chat_web_upload_settle_min_sec": 6.0,
        "chat_web_upload_settle_sec_per_100mb": 10.0,
        "chat_web_upload_settle_max_sec": 60.0,
    }


def _api_candidate_is_safe_without_chat_compare(
    *,
    cfg: Dict[str, Any],
    api_summary: Dict[str, Any],
) -> bool:
    if not bool(
        _cfg_get(
            cfg,
            "run.pre_submit_chat_compare_allow_api_fallback_on_chat_failure_when_clean",
            True,
        )
    ):
        return False
    validator_summary = api_summary.get("validator_summary", {}) or {}
    min_score = max(
        0,
        int(_cfg_get(cfg, "run.pre_submit_chat_compare_api_fallback_min_validator_score", 95) or 95),
    )
    if int(api_summary.get("policy_error_count", 0) or 0) != 0:
        return False
    if not bool(validator_summary.get("ok", False)):
        return False
    if int(validator_summary.get("score", 0) or 0) < min_score:
        return False
    if int(validator_summary.get("segment_error_count", 0) or 0) != 0:
        return False
    if validator_summary.get("episode_errors"):
        return False
    if validator_summary.get("major_fail_triggers"):
        return False
    return True


def maybe_run_pre_submit_chat_compare(
    *,
    cfg: Dict[str, Any],
    source_segments: List[Dict[str, Any]],
    api_segment_plan: Dict[int, Dict[str, Any]],
    task_id: str,
    video_file: Optional[Path],
    api_model: str,
    episode_active_model: str = "",
    task_state: Optional[Dict[str, Any]] = None,
    heartbeat: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    if not bool(_cfg_get(cfg, "run.pre_submit_chat_compare_enabled", False)):
        return {"executed": False, "decision": "disabled"}
    require_chat_compare = bool(_cfg_get(cfg, "run.pre_submit_chat_compare_required", False))
    block_on_chat_failure = bool(_cfg_get(cfg, "run.pre_submit_chat_compare_block_on_chat_failure", False))
    block_if_missing = require_chat_compare or block_on_chat_failure
    if not task_id:
        return _blocked_skip_payload(
            decision="skipped_missing_task_id",
            block=block_if_missing,
            reason="Pre-submit compare blocked apply: missing task id for mandatory Chat UI compare.",
        )
    if video_file is None or not Path(video_file).exists():
        return _blocked_skip_payload(
            decision="skipped_missing_video",
            block=block_if_missing,
            reason="Pre-submit compare blocked apply: episode video is missing for mandatory Chat UI compare.",
        )

    chat_state_raw = str(_cfg_get(cfg, "gemini.chat_web_storage_state", ".state/gemini_chat_storage_state.json")).strip()
    chat_state = Path(chat_state_raw)
    if not chat_state.is_absolute():
        chat_state = (Path.cwd() / chat_state).resolve()
    chat_user_data_dir_raw = str(_cfg_get(cfg, "gemini.chat_web_user_data_dir", "") or "").strip()
    chat_user_data_dir = Path(chat_user_data_dir_raw) if chat_user_data_dir_raw else None
    if chat_user_data_dir is not None and not chat_user_data_dir.is_absolute():
        chat_user_data_dir = (Path.cwd() / chat_user_data_dir).resolve()
    has_chat_storage_state = chat_state.exists()
    has_chat_user_data_dir = bool(chat_user_data_dir and chat_user_data_dir.exists())
    if not has_chat_storage_state and not has_chat_user_data_dir:
        return _blocked_skip_payload(
            decision="skipped_missing_chat_storage_state",
            block=block_if_missing,
            reason="Pre-submit compare blocked apply: Gemini Chat session state is missing.",
            chat_storage_state=str(chat_state),
        )

    out_dir = Path(str(_cfg_get(cfg, "run.output_dir", "outputs"))).resolve()
    compare_dir = out_dir / "pre_submit_compare" / str(task_id).strip().lower()
    compare_dir.mkdir(parents=True, exist_ok=True)

    epsilon_sec = max(0.05, float(_cfg_get(cfg, "run.pre_submit_chat_compare_same_timeline_epsilon_sec", 0.35)))
    block_when_chat_better = bool(_cfg_get(cfg, "run.pre_submit_chat_compare_block_when_chat_better", True))
    auto_adopt_same_timeline = bool(_cfg_get(cfg, "run.pre_submit_chat_compare_auto_adopt_same_timeline", True))
    auto_repair_split_only = bool(_cfg_get(cfg, "run.pre_submit_chat_compare_auto_repair_split_only", False))
    chat_model = str(
        episode_active_model
        or resolve_stage_model(
            cfg,
            "compare_chat",
            _cfg_get(cfg, "run.pre_submit_chat_compare_model", "gemini-3.1-pro-preview"),
        )
        or "gemini-3.1-pro-preview"
    ).strip() or "gemini-3.1-pro-preview"
    chat_model_candidates = _ordered_gen3_gemini_models(
        chat_model,
        _cfg_get(cfg, "gemini.gen3_fallback_models", ["gemini-3.1-pro-preview"]),
    )
    if not chat_model_candidates:
        chat_model_candidates = [chat_model]
    max_segment_duration_sec = max(0.0, float(_cfg_get(cfg, "run.max_segment_duration_sec", 10.0)))

    draft_text = _source_segments_to_draft_text(source_segments)
    (compare_dir / f"{task_id}_source_draft.txt").write_text(draft_text + "\n", encoding="utf-8")

    runtime_cfg = copy.deepcopy(cfg)
    gem_cfg = dict(runtime_cfg.get("gemini", {}) or {})
    gem_cfg["chat_web_storage_state"] = str(chat_state)
    gem_cfg["chat_web_headless"] = bool(_cfg_get(cfg, "gemini.chat_web_headless", False))
    gem_cfg["chat_web_timeout_sec"] = _resolve_pre_submit_chat_timeout_sec(cfg)
    gem_cfg["hybrid_chat_refine_enabled"] = False
    base_primer_path = Path(str(gem_cfg.get("chat_web_memory_primer_file", "") or "").strip())
    if str(base_primer_path or "") and not base_primer_path.is_absolute():
        base_primer_path = (Path.cwd() / base_primer_path).resolve()
    base_primer_text = _read_text_file(base_primer_path) if str(base_primer_path or "").strip() else ""
    memory_bundle_text = _build_chat_memory_bundle_text(
        cfg=cfg,
        base_primer_text=base_primer_text,
        max_feedback_items=max(1, int(_cfg_get(cfg, "run.pre_submit_chat_compare_memory_feedback_max_items", 8) or 8)),
        max_feedback_chars=max(
            240,
            int(_cfg_get(cfg, "run.pre_submit_chat_compare_memory_feedback_max_chars_per_item", 900) or 900),
        ),
        max_drive_chars=max(500, int(_cfg_get(cfg, "run.pre_submit_chat_compare_memory_drive_max_chars", 2800) or 2800)),
    )
    if memory_bundle_text:
        memory_bundle_path = compare_dir / f"{task_id}_chat_memory_bundle.txt"
        memory_bundle_path.write_text(memory_bundle_text + "\n", encoding="utf-8")
        gem_cfg["chat_web_memory_primer_file"] = str(memory_bundle_path)
        compare_payload_memory_path = str(memory_bundle_path)
    else:
        compare_payload_memory_path = ""
    if bool(_cfg_get(cfg, "run.pre_submit_chat_compare_seed_discord_context", False)):
        seed_text = _build_chat_seed_context_text(
            cfg=cfg,
            max_messages=max(1, int(_cfg_get(cfg, "run.pre_submit_chat_compare_discord_context_max_messages", 120) or 120)),
            max_chars_per_message=max(
                120,
                int(_cfg_get(cfg, "run.pre_submit_chat_compare_discord_context_max_chars_per_message", 900) or 900),
            ),
        )
        if seed_text:
            seed_path = compare_dir / f"{task_id}_discord_chat_context.txt"
            seed_path.write_text(seed_text + "\n", encoding="utf-8")
            gem_cfg["chat_web_seed_context_file"] = str(seed_path)
            gem_cfg["chat_web_seed_context_send_before_prompt"] = True
            compare_payload_seed_path = str(seed_path)
        else:
            compare_payload_seed_path = ""
    else:
        compare_payload_seed_path = ""
    runtime_cfg["gemini"] = gem_cfg

    def _write_runtime_config(
        tag: str,
        model_name: str,
        gemini_overrides: Optional[Dict[str, Any]] = None,
    ) -> Path:
        payload = copy.deepcopy(runtime_cfg)
        payload_gem = dict(payload.get("gemini", {}) or {})
        for key, value in _runtime_model_overrides(model_name).items():
            payload_gem[key] = value
        for key, value in (gemini_overrides or {}).items():
            payload_gem[key] = value
        payload["gemini"] = payload_gem
        runtime_config_path = compare_dir / f"{task_id}_runtime_config_{tag}.yaml"
        runtime_config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return runtime_config_path

    api_summary = _evaluate_candidate(
        name="api",
        episode_id=task_id,
        cfg=cfg,
        source_segments=source_segments,
        plan=api_segment_plan,
    )
    (compare_dir / f"{task_id}_api.txt").write_text(api_summary["timed_text"] + "\n", encoding="utf-8")
    (compare_dir / f"{task_id}_api.json").write_text(
        json.dumps(
            {
                "episode_id": task_id,
                "model": api_model,
                "segments": api_summary["segments"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    compare_payload: Dict[str, Any] = {
        "executed": True,
        "episode_id": task_id,
        "video_file": str(Path(video_file).resolve()),
        "chat_storage_state": str(chat_state),
        "memory_bundle_path": compare_payload_memory_path,
        "discord_context_path": compare_payload_seed_path,
        "api_model": api_model,
        "chat_model": chat_model,
        "chat_model_candidates": chat_model_candidates,
        "api_summary": api_summary,
    }
    compare_request_estimated_cost_usd = estimate_cost_usd(
        cfg,
        chat_model,
        prompt_tokens=3994,
        output_tokens=178,
        total_tokens=4172,
    )

    def _persist_compare_cost(chat_result: Dict[str, Any], model_name: str) -> None:
        if not task_id:
            return
        usage_meta = chat_result.get("usage", {}) if isinstance(chat_result, dict) else {}
        estimated_cost = estimate_cost_from_usage(cfg, model_name, usage_meta if isinstance(usage_meta, dict) else {})
        if estimated_cost <= 0:
            estimated_cost = compare_request_estimated_cost_usd
        updates = build_episode_cost_updates(
            cfg,
            task_state if isinstance(task_state, dict) else None,
            stage_name="compare_chat",
            model_name=model_name,
            cost_usd=estimated_cost,
            key_class="paid",
        )
        if isinstance(task_state, dict):
            task_state.update(updates)
        legacy = __import__("src.solver.legacy_impl", fromlist=["_persist_task_state_fields"])
        legacy._persist_task_state_fields(
            cfg,
            task_id,
            task_state if isinstance(task_state, dict) else None,
            **updates,
        )
        compare_payload["cost_snapshot"] = {
            "delta_usd": round(float(estimated_cost), 8),
            "total_usd": round(float(updates.get("episode_estimated_cost_usd", 0.0) or 0.0), 8),
            "ratio": round(float(updates.get("episode_cost_ratio", 0.0) or 0.0), 8),
        }

    draft_path = compare_dir / f"{task_id}_source_draft.txt"

    def _run_chat_compare_once(
        config_path: Path,
        tag: str,
        model_name: str,
        *,
        max_wait_sec_override: float = 0.0,
    ) -> Dict[str, Any]:
        cmd = [
            sys.executable,
            "run_gemini_chat_timed_labels.py",
            "--config",
            str(config_path),
            "--video-path",
            str(Path(video_file).resolve()),
            "--cache-dir",
            str(compare_dir / f"cache_chat_{tag}"),
            "--out-txt",
            str(compare_dir / f"{task_id}_{tag}.txt"),
            "--out-json",
            str(compare_dir / f"{task_id}_{tag}.json"),
            "--episode-id",
            task_id,
            "--model",
            model_name,
            "--prompt-scope",
            "timed_labels",
            "--auth-mode-override",
            "chat_web",
        ]
        if draft_path.exists():
            cmd.extend(["--tier2-draft-path", str(draft_path)])
        max_wait_sec = float(max_wait_sec_override or 0.0)
        if max_wait_sec <= 0:
            max_wait_sec = float(_cfg_get(cfg, "run.pre_submit_chat_compare_max_wait_sec", 300.0) or 300.0)
        return _run_chat_generation_subprocess(cmd=cmd, heartbeat=heartbeat, max_wait_sec=max_wait_sec)

    def _should_retry_chat_failure(exc: Exception) -> bool:
        text = str(exc or "").strip().lower()
        if not text:
            return False
        retry_markers = (
            "timed out",
            "timeout",
            "returned empty stdout",
            "did not return parseable json on stdout",
            "connect_over_cdp",
            "page.goto: timeout",
            "chat input not visible",
            "login/session is likely missing",
        )
        return any(marker in text for marker in retry_markers)

    def _apply_chat_compare_result(
        chat_result: Dict[str, Any],
        *,
        allow_missing_attachment_retry: bool,
        model_name: str,
    ) -> None:
        effective_chat_model = str(chat_result.get("model") or model_name or chat_model).strip() or chat_model
        compare_payload["chat_model"] = effective_chat_model
        chat_video_attached = _chat_generation_attached_video(chat_result)
        compare_payload["chat_generation"] = chat_result
        compare_payload.setdefault("chat_generation_initial", chat_result)
        compare_payload["chat_video_attached"] = chat_video_attached
        _persist_compare_cost(chat_result, effective_chat_model)
        if not chat_video_attached:
            retry_on_missing_attachment = allow_missing_attachment_retry and bool(
                _cfg_get(cfg, "run.pre_submit_chat_compare_retry_on_missing_attachment", True)
            )
            chat_retry_result: Optional[Dict[str, Any]] = None
            if retry_on_missing_attachment:
                retry_config_path = _write_runtime_config(
                    "retry_no_drive",
                    effective_chat_model,
                    {
                        "chat_web_prefer_drive_picker": False,
                        "chat_web_force_clean_thread": False,
                        "chat_web_clean_thread_fallback_enabled": True,
                    },
                )
                try:
                    chat_retry_result = _run_chat_compare_once(retry_config_path, "chat_retry", effective_chat_model)
                    compare_payload["chat_generation_retry"] = chat_retry_result
                    retry_attached = _chat_generation_attached_video(chat_retry_result)
                    compare_payload["chat_video_attached_retry"] = retry_attached
                    if retry_attached:
                        chat_result = chat_retry_result
                        chat_video_attached = True
                        compare_payload["chat_generation"] = chat_retry_result
                        compare_payload["chat_video_attached"] = True
                except Exception as retry_exc:
                    compare_payload["chat_generation_retry_error"] = str(retry_exc)
            if not chat_video_attached:
                compare_payload["winner"] = "api"
                compare_payload["decision"] = "chat_compare_missing_video_attachment"
                compare_payload["adopted"] = False
                if block_on_chat_failure:
                    compare_payload["block_apply"] = True
                    compare_payload["block_reason"] = (
                        "Pre-submit compare blocked apply: Chat UI response did not confirm a successful video "
                        "attachment, so the comparison was ignored."
                    )
                return

        chat_json_path = Path(str(chat_result.get("out_json") or compare_dir / f"{task_id}_chat.json"))
        chat_payload = json.loads(chat_json_path.read_text(encoding="utf-8"))
        chat_segments = parse_timed_segments_payload(chat_payload)
        chat_plan = _segments_to_plan(chat_segments)
        chat_summary = _evaluate_candidate(
            name="chat",
            episode_id=task_id,
            cfg=cfg,
            source_segments=source_segments,
            plan=chat_plan,
        )
        chat_same_timeline = _same_timeline_as_source(source_segments, chat_segments, epsilon_sec=epsilon_sec)
        compare_payload["chat_same_timeline"] = chat_same_timeline
        compare_payload["chat_summary"] = chat_summary

        api_rank = _candidate_rank(api_summary)
        chat_rank = _candidate_rank(chat_summary)
        if chat_rank < api_rank:
            winner = "chat"
        elif api_rank < chat_rank:
            winner = "api"
        else:
            winner = "tie"
        compare_payload["winner"] = winner

        if winner == "chat" and chat_same_timeline and auto_adopt_same_timeline:
            compare_payload["decision"] = "adopt_chat_same_timeline"
            compare_payload["selected_plan"] = chat_plan
            compare_payload["selected_validation_report"] = chat_summary["policy_report"]
            compare_payload["selected_payload"] = {
                "segments": chat_summary["segments"],
                "_meta": {
                    "model": effective_chat_model,
                    "mode": "chat_web_pre_submit_compare",
                    "source": "chat_ui_compare",
                },
            }
            compare_payload["adopted"] = True
        elif winner == "chat" and auto_repair_split_only:
            split_repair_ops = _derive_split_repair_operations(
                source_segments=source_segments,
                candidate_segments=chat_segments,
                max_duration_sec=max_segment_duration_sec,
                epsilon_sec=epsilon_sec,
            )
            if split_repair_ops:
                compare_payload["decision"] = "apply_chat_split_repair"
                compare_payload["selected_operations"] = split_repair_ops
                compare_payload["adopted"] = False
            elif block_when_chat_better:
                compare_payload["decision"] = "block_submit_chat_better_timeline_differs"
                compare_payload["block_apply"] = True
                compare_payload["block_reason"] = (
                    "Pre-submit compare blocked apply: Chat UI candidate scored better than API candidate "
                    "but changes the timeline/count beyond the safe auto-adopt threshold."
                )
                compare_payload["adopted"] = False
        elif winner == "chat" and block_when_chat_better:
            compare_payload["decision"] = "block_submit_chat_better_timeline_differs"
            compare_payload["block_apply"] = True
            compare_payload["block_reason"] = (
                "Pre-submit compare blocked apply: Chat UI candidate scored better than API candidate "
                "but changes the timeline/count beyond the safe auto-adopt threshold."
            )
            compare_payload["adopted"] = False
        else:
            compare_payload["decision"] = "keep_api"
            compare_payload["adopted"] = False

    def _run_chat_compare_with_model(model_name: str, tag_prefix: str) -> Dict[str, Any]:
        primary_config_path = _write_runtime_config(f"{tag_prefix}_primary", model_name)
        try:
            return _run_chat_compare_once(primary_config_path, tag_prefix, model_name)
        except Exception as exc:
            compare_payload["chat_generation_initial_error"] = str(exc)
            retry_on_failure = bool(_cfg_get(cfg, "run.pre_submit_chat_compare_retry_on_failure", True))
            if not retry_on_failure or _is_chat_compare_availability_error_text(exc):
                raise
            if not _should_retry_chat_failure(exc):
                raise
            hard_ratio = float(_cfg_get(cfg, "economics.hard_cost_ratio", 0.20) or 0.20)
            retry_light_estimated_cost_usd = estimate_cost_usd(
                cfg,
                model_name,
                prompt_tokens=3994,
                output_tokens=178,
                total_tokens=4172,
            )
            projected_retry_exceeds_ratio = would_exceed_ratio_cap(
                cfg,
                task_state if isinstance(task_state, dict) else None,
                additional_cost_usd=retry_light_estimated_cost_usd,
                ratio_limit=hard_ratio,
            )
            if projected_retry_exceeds_ratio:
                compare_payload["chat_generation_retry_projected_cost_usd"] = round(
                    float(retry_light_estimated_cost_usd or 0.0), 8
                )
                compare_payload["chat_generation_retry_cost_ratio_exceeded"] = True
                if cost_guard_enforcement_enabled(cfg):
                    compare_payload["chat_generation_retry_skipped"] = "cost_ratio_guard"
                    raise RuntimeError(
                        "chat timed-label subprocess timed out and retry_light would exceed hard cost ratio"
                    )
                print(
                    "[economics] pre-submit retry_light would exceed hard cost ratio, "
                    "but cost guards are disabled; continuing retry."
                )
            retry_config_path = _write_runtime_config(
                f"{tag_prefix}_retry_light",
                model_name,
                _retry_light_gemini_overrides(cfg),
            )
            retry_result = _run_chat_compare_once(
                retry_config_path,
                f"{tag_prefix}_retry_light",
                model_name,
                max_wait_sec_override=_resolve_pre_submit_retry_light_max_wait_sec(cfg),
            )
            compare_payload["chat_generation_retry"] = retry_result
            compare_payload["chat_generation_retry_reason"] = "retryable_failure"
            compare_payload["chat_generation_recovered_via_retry"] = True
            return retry_result

    last_exc: Optional[Exception] = None
    compare_payload["chat_model_attempts"] = []
    for model_index, model_name in enumerate(chat_model_candidates, start=1):
        compare_payload["chat_model"] = model_name
        try:
            chat_result = _run_chat_compare_with_model(model_name, f"chat_model_{model_index}")
            compare_payload["chat_model_attempts"].append({"model": model_name, "status": "ok"})
            _apply_chat_compare_result(
                chat_result,
                allow_missing_attachment_retry=True,
                model_name=model_name,
            )
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            compare_payload["chat_model_attempts"].append(
                {
                    "model": model_name,
                    "status": "error",
                    "error": str(exc),
                }
            )
            if model_index < len(chat_model_candidates) and _is_chat_compare_availability_error_text(exc):
                next_model = chat_model_candidates[model_index]
                compare_payload.setdefault("chat_model_fallbacks", []).append(
                    {
                        "from": model_name,
                        "to": next_model,
                        "reason": str(exc),
                    }
                )
                continue
            break

    if last_exc is not None:
        compare_payload["chat_generation_error"] = str(last_exc)
        compare_payload["winner"] = "api"
        compare_payload["decision"] = "chat_compare_failed"
        compare_payload["adopted"] = False
        if _api_candidate_is_safe_without_chat_compare(cfg=cfg, api_summary=api_summary):
            compare_payload["decision"] = "keep_api_after_chat_failure_clean_api"
            compare_payload["adopted"] = False
            compare_payload["block_apply"] = False
            compare_payload["chat_failure_but_api_clean"] = True
            compare_payload["chat_failure_safe_fallback_reason"] = str(last_exc)
        elif block_on_chat_failure:
            compare_payload["block_apply"] = True
            compare_payload["block_reason"] = (
                f"Pre-submit compare blocked apply: Chat UI generation failed: {last_exc}"
            )

    json_report_path = compare_dir / f"{task_id}_pre_submit_compare.json"
    md_report_path = compare_dir / f"{task_id}_pre_submit_compare.md"
    json_report_path.write_text(json.dumps(compare_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown_report(report_path=md_report_path, compare_payload=compare_payload)
    compare_payload["json_report_path"] = str(json_report_path)
    compare_payload["markdown_report_path"] = str(md_report_path)
    return compare_payload


__all__ = ["maybe_run_pre_submit_chat_compare"]
