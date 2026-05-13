"""
submit_gate.py — Deterministic hard gate before any Atlas submit.

Design goals:
- Single responsibility: decide YES or NO for submit, never silently skip.
- Calls validator.py (rule engine) on the ACTUAL winner text, not on LLM's
  self-reported hallucination flags.
- Returns a structured SubmitGateResult so callers can log, block, or queue.

Usage (from atlas_triplet_batch.py or run_single_episode_4way.sh):

    from submit_gate import evaluate_submit_safety, SubmitGateResult

    result = evaluate_submit_safety(
        episode_id=eid,
        judge_result=payload_obj.get("judge_result", {}),
        inputs=inputs,            # dict with tier2_path, api_path, chat_path, vertex_chat_path
        min_pass_score=95,
    )
    if not result.safe:
        # block submit, log result.reason, optionally queue for manual review
        ...
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.infra.logging_utils import build_print_logger as _build_print_logger

_logger = logging.getLogger(__name__)
print = _build_print_logger(_logger)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SubmitGateResult:
    """Full verdict from the submit gate."""

    episode_id: str
    safe: bool                          # True  = approved for submit
    winner: str                         # tier2 | api | chat | vertex_chat | none
    reason: str                         # machine-readable primary reason
    reason_detail: str = ""             # human-readable extended detail
    score_pct: Optional[int] = None
    validator_ok: Optional[bool] = None
    validator_major_fails: List[str] = field(default_factory=list)
    validator_segment_errors: int = 0
    llm_hallucination_flag: bool = False
    submit_safe_mismatch: bool = False
    checks_performed: List[str] = field(default_factory=list)
    suggested_action: str = ""          # retry_with_repair | manual_review | lower_threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "safe": self.safe,
            "winner": self.winner,
            "reason": self.reason,
            "reason_detail": self.reason_detail,
            "score_pct": self.score_pct,
            "validator_ok": self.validator_ok,
            "validator_major_fails": self.validator_major_fails,
            "validator_segment_errors": self.validator_segment_errors,
            "llm_hallucination_flag": self.llm_hallucination_flag,
            "submit_safe_mismatch": self.submit_safe_mismatch,
            "checks_performed": self.checks_performed,
            "suggested_action": self.suggested_action,
        }

    def summary_line(self) -> str:
        status = "APPROVED" if self.safe else "BLOCKED"
        return (
            f"[gate] {status} episode={self.episode_id} winner={self.winner} "
            f"score={self.score_pct} validator_ok={self.validator_ok} "
            f"reason={self.reason}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_score(value: Any) -> Optional[float]:
    try:
        v = float(value)
        if 0.0 <= v <= 100.0:
            return v
        return None
    except Exception:
        return None


def _write_episode_eval_text_file(outputs_dir: Path, episode_id: str, text: str) -> Optional[Path]:
    import re
    eid = str(episode_id or "").strip().lower()
    if not eid or not re.fullmatch(r"[a-z0-9_-]+", eid):
        return None
    
    target_dir = outputs_dir / "chat_reviews" / eid
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "eval.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _parse_timed_segments_from_file(path_str: str) -> List[Dict[str, Any]]:
    """
    Parse a winner candidate file (text or JSON) into a list of segment dicts
    compatible with validator.validate_episode().

    Imports parse_timed_segments_text lazily to avoid circular imports.
    Falls back to empty list on any error.
    """
    raw = str(path_str or "").strip()
    if not raw:
        return []
    p = Path(raw)
    if not p.exists() or not p.is_file():
        return []
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    # Lazy import to keep this module self-contained
    try:
        from atlas_triplet_compare import parse_timed_segments_text  # type: ignore
        segs = parse_timed_segments_text(content)
        if segs:
            return segs
    except Exception:
        pass

    # Fallback: minimal JSON parse
    try:
        payload = json.loads(content)
        if isinstance(payload, list):
            return [s for s in payload if isinstance(s, dict)]
        if isinstance(payload, dict):
            segs = payload.get("segments") or payload.get("labels") or []
            if isinstance(segs, list):
                return [s for s in segs if isinstance(s, dict)]
    except Exception:
        pass

    return []


def _build_annotation_dict(
    episode_id: str,
    segments: List[Dict[str, Any]],
    video_duration_sec: float = 0.0,
) -> Dict[str, Any]:
    """Build a minimal annotation dict accepted by validator.validate_episode()."""
    normalized_segs = []
    for idx, seg in enumerate(segments):
        normalized_segs.append({
            "segment_index": idx,
            "start_sec": float(seg.get("start_sec", seg.get("start", 0.0)) or 0.0),
            "end_sec": float(seg.get("end_sec", seg.get("end", 0.0)) or 0.0),
            "label": str(seg.get("label", "") or "").strip(),
            "duration_sec": max(0.0, float(seg.get("end_sec", seg.get("end", 0.0)) or 0.0)
                                - float(seg.get("start_sec", seg.get("start", 0.0)) or 0.0)),
        })
    return {
        "episode_id": episode_id,
        "video_duration_sec": video_duration_sec,
        "segments": normalized_segs,
    }


def _resolve_winner_path(winner: str, inputs: Dict[str, Any]) -> str:
    """Map winner name → candidate file path from inputs dict."""
    mapping = {
        "tier2": "tier2_path",
        "api": "api_path",
        "chat": "chat_path",
        "vertex_chat": "vertex_chat_path",
    }
    key = mapping.get(str(winner or "").strip().lower(), "")
    if not key:
        return ""
    return str(inputs.get(key) or "").strip()


# ---------------------------------------------------------------------------
# Core gate function
# ---------------------------------------------------------------------------

def evaluate_submit_safety(
    *,
    episode_id: str,
    judge_result: Dict[str, Any],
    inputs: Dict[str, Any],
    min_pass_score: int = 95,
    min_pass_score_validator_ok: int = 95,
    video_duration_sec: float = 0.0,
    run_validator: bool = True,
) -> SubmitGateResult:
    """
    Evaluate whether an episode is safe to submit.

    Check order (each check can independently block):
      1. winner must be a known valid candidate (not "none")
      2. submit_safe_solution — explicit mismatch blocks; empty/missing is
         a soft warning (logged but does not block when other checks pass)
      3. LLM hallucination flag must be False for winner
      4. validator.py must pass on the actual winner text (deterministic)
      5. winner score must be >= min_pass_score (or >= min_pass_score_validator_ok
         when the deterministic validator already passed)

    Returns a SubmitGateResult with safe=True only if ALL checks pass.
    """
    eid = str(episode_id or "").strip().lower()
    checks: List[str] = []

    if not isinstance(judge_result, dict):
        return SubmitGateResult(
            episode_id=eid, safe=False, winner="", reason="missing_judge_result",
            reason_detail="judge_result is missing or not a dict",
            suggested_action="manual_review",
        )

    winner = str(judge_result.get("winner") or "").strip().lower()
    submit_safe = str(judge_result.get("submit_safe_solution") or "").strip().lower()

    scores = judge_result.get("scores")
    winner_score: Optional[float] = None
    if isinstance(scores, dict):
        winner_score = _safe_score(scores.get(winner))

    hall = judge_result.get("hallucination")
    llm_hall = isinstance(hall, dict) and bool(hall.get(winner))

    # ── Check 1: winner validity ──────────────────────────────────────────
    checks.append("winner_validity")
    if winner not in {"tier2", "api", "chat", "vertex_chat"}:
        return SubmitGateResult(
            episode_id=eid, safe=False, winner=winner,
            reason="winner_none_or_invalid",
            reason_detail=f"LLM judge returned winner='{winner}' which is not a valid candidate",
            score_pct=_safe_int(winner_score) if winner_score is not None else None,
            checks_performed=checks,
            suggested_action="retry_with_repair",
        )

    # ── Check 2: submit_safe_solution match ───────────────────────────────
    # Relaxed: empty/missing submit_safe is a soft warning, NOT a hard block.
    # Only an explicit mismatch (submit_safe is set AND differs) blocks.
    checks.append("submit_safe_match")
    submit_safe_mismatch = False
    submit_safe_warning = False
    if not submit_safe:
        # Missing/empty — soft warning, continue to remaining checks
        submit_safe_warning = True
        print(
            f"[gate] WARNING: submit_safe_solution is empty for winner='{winner}' "
            f"(episode={eid}). Proceeding with remaining checks."
        )
    elif submit_safe != winner:
        # Explicit mismatch — hard block
        submit_safe_mismatch = True
        return SubmitGateResult(
            episode_id=eid, safe=False, winner=winner,
            reason="submit_safe_explicit_mismatch",
            reason_detail=(
                f"submit_safe_solution='{submit_safe}' explicitly contradicts winner='{winner}'. "
                "LLM judge chose a different candidate as safe to submit."
            ),
            score_pct=_safe_int(winner_score) if winner_score is not None else None,
            llm_hallucination_flag=llm_hall,
            submit_safe_mismatch=True,
            checks_performed=checks,
            suggested_action="retry_with_repair",
        )

    # ── Check 3: LLM hallucination flag ───────────────────────────────────
    checks.append("hallucination_flag")
    if llm_hall:
        return SubmitGateResult(
            episode_id=eid, safe=False, winner=winner,
            reason="winner_hallucination",
            reason_detail=f"LLM judge flagged hallucination=True for winner='{winner}'",
            score_pct=_safe_int(winner_score) if winner_score is not None else None,
            llm_hallucination_flag=True,
            checks_performed=checks,
            suggested_action="retry_with_repair",
        )

    # ── Check 4: deterministic validator.py ───────────────────────────────
    # Run before score threshold so we capture diagnostics and can use
    # the relaxed threshold when validator passes.
    validator_ok: Optional[bool] = None
    validator_major_fails: List[str] = []
    validator_segment_errors = 0

    if run_validator:
        checks.append("validator_deterministic")
        winner_path = _resolve_winner_path(winner, inputs)
        segments = _parse_timed_segments_from_file(winner_path)

        if not segments:
            # Can't validate without segments — treat as blocking: safer to block than skip
            return SubmitGateResult(
                episode_id=eid, safe=False, winner=winner,
                reason="validator_no_segments_parsed",
                reason_detail=(
                    f"Could not parse any segments from winner file: '{winner_path}'. "
                    "Cannot run deterministic validation. Blocking as precaution."
                ),
                score_pct=int(round(winner_score)) if winner_score is not None else 0,
                validator_ok=False,
                llm_hallucination_flag=llm_hall,
                checks_performed=checks,
                suggested_action="manual_review",
            )

        annotation = _build_annotation_dict(eid, segments, video_duration_sec)
        try:
            import validator as val_module  # type: ignore
            val_report = val_module.validate_episode(annotation)
            validator_ok = bool(val_report.get("ok"))
            validator_major_fails = list(val_report.get("major_fail_triggers") or [])
            seg_reports = val_report.get("segment_reports") or []
            validator_segment_errors = sum(
                1 for r in seg_reports if isinstance(r, dict) and r.get("errors")
            )
        except ImportError:
            checks.append("validator_import_failed")
            return SubmitGateResult(
                episode_id=eid,
                safe=False,
                winner=winner,
                reason="validator_unavailable",
                reason_detail="validator.py import failed. Gate is fail-closed in production.",
                score_pct=int(round(winner_score)) if winner_score is not None else 0,
                validator_ok=False,
                llm_hallucination_flag=llm_hall,
                checks_performed=checks,
                suggested_action="manual_review",
            )
        except Exception as exc:
            checks.append(f"validator_exception:{str(exc)[:80]}")
            return SubmitGateResult(
                episode_id=eid,
                safe=False,
                winner=winner,
                reason="validator_runtime_error",
                reason_detail=f"validator.py raised exception: {str(exc)[:240]}",
                score_pct=int(round(winner_score)) if winner_score is not None else 0,
                validator_ok=False,
                llm_hallucination_flag=llm_hall,
                checks_performed=checks,
                suggested_action="manual_review",
            )

        if validator_ok is False:
            return SubmitGateResult(
                episode_id=eid, safe=False, winner=winner,
                reason="validator_policy_fail",
                reason_detail=(
                    f"validator.py rejected winner='{winner}': "
                    f"major_fail_triggers={validator_major_fails}, "
                    f"segments_with_errors={validator_segment_errors}"
                ),
                score_pct=int(round(winner_score)) if winner_score is not None else 0,
                validator_ok=False,
                validator_major_fails=validator_major_fails,
                validator_segment_errors=validator_segment_errors,
                llm_hallucination_flag=llm_hall,
                checks_performed=checks,
                suggested_action="retry_with_repair",
            )

    # ── Check 5: score threshold ──────────────────────────────────────────
    # Use relaxed threshold when the deterministic validator already passed.
    checks.append("score_threshold")
    if validator_ok:
        effective_threshold = max(0, min(100, int(min_pass_score_validator_ok)))
    else:
        effective_threshold = max(0, min(100, int(min_pass_score)))
    if winner_score is None:
        return SubmitGateResult(
            episode_id=eid, safe=False, winner=winner,
            reason="missing_winner_score",
            reason_detail=f"No score found for winner='{winner}' in judge scores: {scores}",
            validator_ok=validator_ok,
            validator_major_fails=validator_major_fails,
            validator_segment_errors=validator_segment_errors,
            llm_hallucination_flag=llm_hall,
            checks_performed=checks,
            suggested_action="manual_review",
        )
    if winner_score < float(effective_threshold):
        return SubmitGateResult(
            episode_id=eid, safe=False, winner=winner,
            reason=f"score_below_threshold({winner_score:.1f}<{effective_threshold})",
            reason_detail=(
                f"Winner score {winner_score:.1f} is below required threshold {effective_threshold}"
                + (" (relaxed: validator passed)" if validator_ok else "")
            ),
            score_pct=int(round(winner_score)),
            validator_ok=validator_ok,
            validator_major_fails=validator_major_fails,
            validator_segment_errors=validator_segment_errors,
            llm_hallucination_flag=llm_hall,
            checks_performed=checks,
            suggested_action="lower_threshold",
        )

    return SubmitGateResult(
        episode_id=eid, safe=True, winner=winner,
        reason="all_checks_passed",
        reason_detail=(
            f"winner='{winner}' score={winner_score:.1f} "
            f"validator_ok={validator_ok} "
            f"major_fails={validator_major_fails}"
            + (" [submit_safe_warning: field was empty]" if submit_safe_warning else "")
        ),
        score_pct=int(round(winner_score)),
        validator_ok=validator_ok,
        validator_major_fails=validator_major_fails,
        validator_segment_errors=validator_segment_errors,
        llm_hallucination_flag=llm_hall,
        submit_safe_mismatch=submit_safe_warning,
        checks_performed=checks,
    )


# ---------------------------------------------------------------------------
# Manual queue helper
# ---------------------------------------------------------------------------

def append_to_manual_queue(
    outputs_dir: Path,
    result: SubmitGateResult,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write a blocked episode to outputs/manual_queue.jsonl for human review.
    Idempotent: calling multiple times for the same episode appends a new row
    (reviewer can see retry history).
    """
    from datetime import datetime, timezone  # lazy import
    queue_path = outputs_dir / "manual_queue.jsonl"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    row = result.to_dict()
    row["queued_at_utc"] = datetime.now(timezone.utc).isoformat()
    if isinstance(extra_meta, dict):
        row.update(extra_meta)
    with queue_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return queue_path


# ---------------------------------------------------------------------------
# CLI — for direct testing from shell
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Test submit gate on a triplet_compare result JSON"
    )
    parser.add_argument("--result-json", required=True, help="Path to triplet_compare_<eid>.json")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--min-pass-score", type=int, default=95)
    parser.add_argument("--no-validator", action="store_true")
    args = parser.parse_args()

    p = Path(args.result_json).resolve()
    if not p.exists():
        print(f"[gate] error: file not found: {p}")
        sys.exit(2)

    payload = json.loads(p.read_text(encoding="utf-8"))
    judge = payload.get("judge_result", {}) if isinstance(payload, dict) else {}

    # Build inputs from text_refs in the result JSON
    text_refs = payload.get("text_refs", {}) if isinstance(payload, dict) else {}
    inputs = {
        "tier2_path":       str(text_refs.get("resolved_tier2_path") or ""),
        "api_path":         str(text_refs.get("resolved_api_path") or ""),
        "chat_path":        str(text_refs.get("resolved_chat_path") or ""),
        "vertex_chat_path": str(text_refs.get("resolved_vertex_chat_path") or ""),
    }

    eid = str(payload.get("episode_id") or p.stem.replace("triplet_compare_", ""))
    result = evaluate_submit_safety(
        episode_id=eid,
        judge_result=judge,
        inputs=inputs,
        min_pass_score=args.min_pass_score,
        run_validator=not args.no_validator,
    )

    print(result.summary_line())
    import json as _json
    print(_json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    if not result.safe:
        queue_path = append_to_manual_queue(Path(args.outputs_dir), result)
        print(f"[gate] queued_for_manual_review: {queue_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
