"""Pre-submit consistency validation layer."""

from typing import Any, Dict, List

from src.infra.solver_config import _cfg_get


def validate_pre_submit_consistency(
    page: Any,
    cfg: Dict[str, Any],
    segment_plan: Dict[int, Dict[str, Any]],
    source_segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare live DOM, extracted source, and policy input before submit.

    Blocking mismatches focus on DOM vs extracted source, because those represent
    actual UI drift/state desync. Plan-vs-DOM timestamp drift is logged as a warning
    unless it also implies missing/extra indices.
    """
    from src.solver.desync import (
        build_segment_snapshot,
        compare_segment_snapshots,
        warn_on_plan_vs_live,
    )

    # Local import to avoid circular dependency
    from src.solver.segments import extract_segments

    tolerance_sec = float(_cfg_get(cfg, "run.desync_snapshot_tolerance_sec", 0.25) or 0.25)
    page_url = str(getattr(page, "url", "") or "")
    source_snapshot = build_segment_snapshot(
        segments=source_segments,
        source_kind="extracted_source",
        page_url=page_url,
    )
    try:
        live_segments = extract_segments(page, cfg)
    except Exception as exc:
        message = f"live DOM unavailable during pre-submit consistency: {exc}"
        live_segments = []
        live_snapshot = build_segment_snapshot(
            segments=live_segments,
            source_kind="live_dom",
            page_url=page_url,
        )
        print("[consistency] PRE-SUBMIT CHECK FAILED: 1 blocking mismatch(es) detected")
        print(f"[consistency]   - {message}")
        return {
            "consistent": False,
            "mismatches": [message],
            "warnings": [],
            "live_segments": live_segments,
            "live_snapshot": live_snapshot.to_dict(),
            "source_snapshot": source_snapshot.to_dict(),
            "desync_decision": {
                "ok": False,
                "desync_detected": True,
                "reason": message,
                "requires_reextract": True,
                "requires_cache_invalidation": False,
                "blocking_mismatches": [message],
                "warnings": [],
            },
        }
    live_snapshot = build_segment_snapshot(
        segments=live_segments,
        source_kind="live_dom",
        page_url=page_url,
    )
    decision = compare_segment_snapshots(
        live_snapshot=live_snapshot,
        source_snapshot=source_snapshot,
        tolerance_sec=tolerance_sec,
    )
    blocking_mismatches: List[str] = list(decision.blocking_mismatches)
    warnings: List[str] = list(decision.warnings)
    warnings.extend(
        warn_on_plan_vs_live(
            plan_segments=segment_plan or {},
            live_snapshot=live_snapshot,
            tolerance_sec=0.5,
        )
    )

    plan_by_idx = {
        int(idx): item
        for idx, item in (segment_plan or {}).items()
        if int(idx or 0) > 0 and isinstance(item, dict)
    }
    live_by_idx = {
        int(seg.get("segment_index", 0)): seg
        for seg in live_segments
        if int(seg.get("segment_index", 0) or 0) > 0
    }
    source_by_idx = {
        int(seg.get("segment_index", 0)): seg
        for seg in source_segments
        if int(seg.get("segment_index", 0) or 0) > 0
    }
    plan_only = sorted(idx for idx in plan_by_idx if idx not in live_by_idx)
    missing_from_plan = sorted(idx for idx in source_by_idx if idx not in plan_by_idx)
    if plan_only:
        blocking_mismatches.append(
            f"policy input references segments missing from live DOM: {plan_only[:10]}"
        )
    if missing_from_plan:
        blocking_mismatches.append(
            f"policy input is missing source segment indices: {missing_from_plan[:10]}"
        )

    if blocking_mismatches:
        print(
            f"[consistency] PRE-SUBMIT CHECK FAILED: "
            f"{len(blocking_mismatches)} blocking mismatch(es) detected"
        )
        for message in blocking_mismatches:
            print(f"[consistency]   - {message}")
    else:
        print(
            "[consistency] pre-submit check passed: live DOM matches extracted source "
            f"({len(live_by_idx)} segments, checksum={live_snapshot.checksum})"
        )
    for warning in warnings[:10]:
        print(f"[consistency] warning: {warning}")

    return {
        "consistent": decision.ok and len(blocking_mismatches) == 0,
        "mismatches": blocking_mismatches,
        "warnings": warnings,
        "live_segments": live_segments,
        "live_snapshot": live_snapshot.to_dict(),
        "source_snapshot": source_snapshot.to_dict(),
        "desync_decision": decision.to_dict(),
    }
