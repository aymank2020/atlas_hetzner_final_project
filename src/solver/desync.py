"""Snapshot-based desync detection helpers for episode-scoped validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class SegmentSnapshot:
    episode_id: str
    context_id: str
    segments: List[Dict[str, Any]]
    segment_count: int
    checksum: str
    page_url: str
    created_at_utc: str
    source_kind: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DesyncDecision:
    ok: bool
    desync_detected: bool
    reason: str
    requires_reextract: bool
    requires_cache_invalidation: bool
    blocking_mismatches: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _segment_signature(segment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "segment_index": int(segment.get("segment_index", 0) or 0),
        "start_sec": round(_safe_float(segment.get("start_sec"), 0.0), 3),
        "end_sec": round(_safe_float(segment.get("end_sec"), 0.0), 3),
        "current_label": str(
            segment.get("current_label", segment.get("label", "")) or ""
        ).strip(),
    }


def build_segment_checksum(segments: List[Dict[str, Any]]) -> str:
    payload = [_segment_signature(segment) for segment in (segments or [])]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def build_segment_snapshot(
    *,
    segments: List[Dict[str, Any]],
    episode_id: str = "",
    context_id: str = "",
    page_url: str = "",
    source_kind: str = "dom",
) -> SegmentSnapshot:
    clean_segments = [dict(segment) for segment in (segments or [])]
    return SegmentSnapshot(
        episode_id=str(episode_id or "").strip(),
        context_id=str(context_id or "").strip(),
        segments=clean_segments,
        segment_count=len(clean_segments),
        checksum=build_segment_checksum(clean_segments),
        page_url=str(page_url or "").strip(),
        created_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source_kind=str(source_kind or "dom").strip() or "dom",
    )


def _segments_by_index(segments: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for segment in segments or []:
        idx = int(segment.get("segment_index", 0) or 0)
        if idx > 0:
            out[idx] = segment
    return out


def compare_segment_snapshots(
    *,
    live_snapshot: SegmentSnapshot,
    source_snapshot: SegmentSnapshot,
    last_snapshot: Optional[SegmentSnapshot] = None,
    tolerance_sec: float = 0.25,
) -> DesyncDecision:
    live_by_idx = _segments_by_index(live_snapshot.segments)
    source_by_idx = _segments_by_index(source_snapshot.segments)
    last_by_idx = _segments_by_index(last_snapshot.segments) if last_snapshot is not None else {}

    blocking: List[str] = []
    warnings: List[str] = []

    live_only = sorted(idx for idx in live_by_idx if idx not in source_by_idx)
    source_only = sorted(idx for idx in source_by_idx if idx not in live_by_idx)
    if live_only:
        blocking.append(f"live DOM has unexpected extra segments: {live_only[:10]}")
    if source_only:
        blocking.append(f"source snapshot has segments missing from live DOM: {source_only[:10]}")

    for idx in sorted(set(live_by_idx) & set(source_by_idx)):
        live = live_by_idx[idx]
        source = source_by_idx[idx]
        live_start = _safe_float(live.get("start_sec"), 0.0)
        live_end = _safe_float(live.get("end_sec"), live_start)
        source_start = _safe_float(source.get("start_sec"), 0.0)
        source_end = _safe_float(source.get("end_sec"), source_start)
        if abs(live_start - source_start) > tolerance_sec or abs(live_end - source_end) > tolerance_sec:
            blocking.append(
                f"segment {idx}: live DOM {live_start:.2f}-{live_end:.2f}s "
                f"drifted from source {source_start:.2f}-{source_end:.2f}s"
            )

    if last_snapshot is not None:
        last_only = sorted(idx for idx in last_by_idx if idx not in live_by_idx)
        if last_only:
            warnings.append(f"last snapshot still references missing live segments: {last_only[:10]}")
        if last_snapshot.checksum != live_snapshot.checksum and not blocking:
            warnings.append(
                f"live snapshot checksum changed since last snapshot: "
                f"{last_snapshot.checksum} -> {live_snapshot.checksum}"
            )

    if blocking:
        return DesyncDecision(
            ok=False,
            desync_detected=True,
            reason=blocking[0],
            requires_reextract=True,
            requires_cache_invalidation=True,
            blocking_mismatches=blocking,
            warnings=warnings,
        )
    return DesyncDecision(
        ok=True,
        desync_detected=False,
        reason="live DOM matches extracted source snapshot",
        requires_reextract=False,
        requires_cache_invalidation=False,
        blocking_mismatches=[],
        warnings=warnings,
    )


def warn_on_plan_vs_live(
    *,
    plan_segments: Dict[int, Dict[str, Any]],
    live_snapshot: SegmentSnapshot,
    tolerance_sec: float = 0.5,
) -> List[str]:
    live_by_idx = _segments_by_index(live_snapshot.segments)
    warnings: List[str] = []
    for raw_idx, plan in (plan_segments or {}).items():
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        live = live_by_idx.get(idx)
        if live is None:
            warnings.append(f"policy input references segment {idx} missing from live DOM")
            continue
        plan_start = _safe_float(plan.get("start_sec"), 0.0)
        plan_end = _safe_float(plan.get("end_sec"), plan_start)
        live_start = _safe_float(live.get("start_sec"), 0.0)
        live_end = _safe_float(live.get("end_sec"), live_start)
        if abs((plan_end - plan_start) - (live_end - live_start)) > tolerance_sec:
            warnings.append(
                f"segment {idx}: plan duration={(plan_end - plan_start):.1f}s "
                f"vs live DOM={(live_end - live_start):.1f}s"
            )
    return warnings


__all__ = [
    "SegmentSnapshot",
    "DesyncDecision",
    "build_segment_checksum",
    "build_segment_snapshot",
    "compare_segment_snapshots",
    "warn_on_plan_vs_live",
]
