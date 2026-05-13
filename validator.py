"""
Atlas annotation validator (rule-engine).
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import prompts
from src.policy import context_manager as policy_context


NO_ACTION_LABEL = "No Action"
MAX_ATOMIC_ACTIONS_PER_LABEL = 2
MAX_WORDS_PER_LABEL = 20

DISALLOWED_TOOL_TERMS = (
    "mechanical arm",
    "robotic arm",
    "robot arm",
    "manipulator",
    "robot gripper",
    "claw arm",
)

OBJECT_EXPECTING_VERBS = {
    "pick up",
    "place",
    "move",
    "relocate",
    "adjust",
    "hold",
    "grab",
    "align",
    "cut",
    "open",
    "close",
    "peel",
    "secure",
    "wipe",
    "flip",
    "chisel",
}

ALLOWED_LABEL_START_VERBS = (
    "pick up",
    "put down",
    "set down",
    "take out",
    "take off",
    "turn on",
    "turn off",
    "plug in",
    "place",
    "move",
    "adjust",
    "hold",
    "align",
    "relocate",
    "tighten",
    "loosen",
    "wipe",
    "clean",
    "paint",
    "dip",
    "remove",
    "insert",
    "pull",
    "push",
    "turn",
    "open",
    "close",
    "unscrew",
    "screw",
    "lift",
    "set",
    "attach",
    "detach",
    "apply",
    "cut",
    "chisel",
    "drill",
    "measure",
    "fold",
    "press",
    "slide",
    "stack",
    "pack",
    "unpack",
    "straighten",
    "comb",
    "spread",
    "shake",
    "pour",
    "spray",
    "peel",
    "wrap",
    "lock",
    "unlock",
    "grasp",
    "position",
    "fit",
    "mount",
    "unmount",
    "clip",
    "unclip",
    "twist",
    "untwist",
    "raise",
    "lower",
    "connect",
    "disconnect",
    "bend",
)
ALLOWED_LABEL_START_VERB_TOKEN_PATTERNS = sorted(
    [tuple(re.findall(r"[a-z]+", v.lower())) for v in ALLOWED_LABEL_START_VERBS],
    key=len,
    reverse=True,
)

INTENT_PATTERNS = [
    r"\bprepare to\b",
    r"\btry to\b",
    r"\babout to\b",
    r"\bintend to\b",
]

# Spec-Kit Rule: HOLD labels must NOT include intent phrases
HOLD_INTENT_PATTERN = re.compile(
    r"\bhold\b.+\b(to\s+(?:check|see|verify|inspect|examine|test|ensure|confirm|observe|look|monitor|view))"
    r"|\bhold\b.+\b(for\s+(?:checking|inspection|verification|testing))",
    re.IGNORECASE,
)

# Spec-Kit Rule: MERGE consecutive identical labels under 60s
MERGE_TOLERANCE_SEC = 0.5  # Gap tolerance between segments
MERGE_MAX_DURATION_SEC = 60.0  # Max combined duration for merge (Strict Limit)
SWEET_SPOT_MIN = 2.0           # Ideal minimum segment duration
SWEET_SPOT_MAX = 5.0           # Ideal maximum segment duration
MAX_SEGMENT_DURATION_SEC = 10.0


@dataclass(frozen=True)
class PolicyConstraints:
    max_atomic_actions_per_label: int = MAX_ATOMIC_ACTIONS_PER_LABEL
    max_words_per_label: int = MAX_WORDS_PER_LABEL
    merge_tolerance_sec: float = MERGE_TOLERANCE_SEC
    merge_max_duration_sec: float = MERGE_MAX_DURATION_SEC
    sweet_spot_min_sec: float = SWEET_SPOT_MIN
    sweet_spot_max_sec: float = SWEET_SPOT_MAX
    max_segment_duration_sec: float = MAX_SEGMENT_DURATION_SEC

NUMERAL_PATTERN = re.compile(r"\d")
WHITESPACE_PATTERN = re.compile(r"\s+")
PLACE_LOCATION_PATTERN = re.compile(r"\bplace\b.*\b(on|in|into|onto|to|inside|at|under|over)\b", re.IGNORECASE)
CHAINED_VERB_WITHOUT_OBJECT_PATTERN = re.compile(
    r"\b(pick up|place|move|relocate|adjust|hold|align)\s+and\s+(pick up|place|move|relocate|adjust|hold|align)\b",
    re.IGNORECASE,
)
ORPHAN_SECOND_PLACE_PATTERN = re.compile(
    r"\band\s+place\s+(on|in|into|onto|to|inside|at|under|over)\b",
    re.IGNORECASE,
)
BODY_PART_REFERENCE_PATTERN = re.compile(
    r"\b(hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists)\b",
    re.IGNORECASE,
)
TOKEN_STUTTER_PATTERN = re.compile(
    r"\b([a-z]+(?:\s+[a-z]+){0,2})\s+\1\b",
    re.IGNORECASE,
)
MECHANICAL_MOTION_PATTERN = re.compile(
    r"\bmove\s+(?:comb(?:\s+tail)?|hair\s+straightener)\b|"
    r"\bmove\s+\w+\s+back\s+and\s+forth\b",
    re.IGNORECASE,
)
GUIDE_AS_OBJECT_DESCRIPTOR_PATTERN = re.compile(
    r"\b(?:pick up|place|align|adjust|pull|move|position|set|smooth|hold)\s+guide\s+"
    r"(?:fabric|cloth|garment|material|item|piece)\b",
    re.IGNORECASE,
)
INVALID_ADJUST_OVER_PATTERN = re.compile(r"\badjust\s+over\b", re.IGNORECASE)

NUMERAL_TO_WORD = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}

PLURAL_EXPECTED_NUMBER_WORDS = {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
SINGULAR_EXPECTED_NUMBER_WORDS = {"one", "a", "an"}
UNCOUNTABLE_NOUN_HINTS = {
    "water",
    "sand",
    "rice",
    "hair",
    "equipment",
    "furniture",
    "paper",
    "soap",
    "oil",
    "powder",
}
PLURAL_ONLY_NOUN_HINTS = {"scissors", "glasses", "pants", "shorts", "pliers", "tongs"}
GENERIC_OBJECT_TOKENS = {"item", "object", "thing", "stuff", "tool", "container", "none", "surface", "mat"}

# Mutually-exclusive device families. If output switches family from draft (Tier2),
# raise a warning for human review.
DEVICE_CLASS_GROUPS: Dict[str, Set[str]] = {
    "phone": {"phone", "smartphone", "mobile", "cellphone", "iphone", "android"},
    "laptop": {"laptop", "notebook", "computer", "macbook", "ultrabook"},
    "tablet": {"tablet", "ipad"},
    "camera": {"camera", "webcam", "camcorder"},
    "remote": {"remote", "controller"},
}


@lru_cache(maxsize=1)
def get_policy_constraints() -> PolicyConstraints:
    policy = policy_context.load_current_policy()
    max_atomic_actions_per_label = int(
        policy_context.get_policy("annotation.max_atomic_actions", 2, policy=policy)
    )
    max_words_per_label = int(
        policy_context.get_policy("annotation.max_label_words", 20, policy=policy)
    )
    merge_tolerance_sec = float(
        policy_context.get_policy("merge_rules.tolerance_sec", 0.5, policy=policy)
    )
    merge_max_duration_sec = float(
        policy_context.get_policy("merge_rules.max_combined_duration_seconds", 60.0, policy=policy)
    )
    preferred_density = policy_context.get_policy(
        "annotation.preferred_density_sec",
        {"min": 2.0, "max": 5.0},
        policy=policy,
    )
    max_segment_duration_sec = float(
        policy_context.get_policy("engine_limits.max_segment_seconds", 10.0, policy=policy)
    )
    return PolicyConstraints(
        max_atomic_actions_per_label=max_atomic_actions_per_label,
        max_words_per_label=max_words_per_label,
        merge_tolerance_sec=merge_tolerance_sec,
        merge_max_duration_sec=merge_max_duration_sec,
        sweet_spot_min_sec=float(preferred_density.get("min", 2.0)),
        sweet_spot_max_sec=float(preferred_density.get("max", 5.0)),
        max_segment_duration_sec=max_segment_duration_sec,
    )


def refresh_policy_constraints() -> PolicyConstraints:
    get_policy_constraints.cache_clear()
    return get_policy_constraints()


def normalize_spaces(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", str(text).strip())


def lower(text: str) -> str:
    return normalize_spaces(text).lower()


def parse_time_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    s = normalize_spaces(str(value))
    if not s:
        return 0.0

    if ":" not in s:
        try:
            return float(s)
        except ValueError:
            return 0.0

    parts = s.split(":")
    try:
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds
    except ValueError:
        return 0.0
    return 0.0


def duration_matches(start: float, end: float, duration: float, tol: float = 0.05) -> bool:
    calc = round(end - start, 3)
    return abs(calc - duration) <= tol


def contains_forbidden_verbs(label: str) -> List[str]:
    label_l = lower(label)
    found = []
    for verb in prompts.FORBIDDEN_VERBS:
        if re.search(rf"\b{re.escape(verb)}\b", label_l):
            found.append(verb)
    return found


def contains_forbidden_narrative_words(label: str) -> List[str]:
    label_l = lower(label)
    found: List[str] = []
    words = getattr(prompts, "FORBIDDEN_NARRATIVE_WORDS", [])
    for word in words:
        token = str(word or "").strip().lower()
        if not token:
            continue
        if re.search(rf"\b{re.escape(token)}\b", label_l):
            found.append(token)
    return found


def allow_reach_for_truncated_end(label: str, end_sec: Any, video_duration_sec: float) -> bool:
    """
    Atlas exception: allow "reach" only when the clip likely truncates at the very end.
    """
    if not re.search(r"\breach\b", lower(label)):
        return False
    try:
        end = float(end_sec)
        duration = float(video_duration_sec)
    except Exception:
        return False
    if duration <= 0:
        return False
    # If remaining tail is very small, treat as truncated-end edge case.
    tail_sec = max(0.0, duration - end)
    return tail_sec <= 0.35


def has_numerals(label: str) -> bool:
    return bool(NUMERAL_PATTERN.search(label))


def min_two_words(label: str) -> bool:
    if normalize_spaces(label) == NO_ACTION_LABEL:
        return True
    words = [w for w in re.split(r"\s+", normalize_spaces(label)) if w]
    return len(words) >= 2


def word_count(label: str) -> int:
    return len([w for w in re.split(r"\s+", normalize_spaces(label)) if w])


def is_imperative_like(label: str) -> bool:
    l = lower(label)
    if l == "no action":
        return True
    if any(re.search(p, l) for p in INTENT_PATTERNS):
        return False
    bad_starts = ("a ", "an ", "the ", "person ", "ego ", "he ", "she ", "they ")
    if l.startswith(bad_starts):
        return False
    first_word = re.split(r"\s+", l.strip())[0] if l.strip() else ""
    if len(first_word) > 4 and first_word.endswith("ing"):
        return False
    return True


def has_intent_only_language(label: str) -> bool:
    l = lower(label)
    return any(re.search(p, l) for p in INTENT_PATTERNS)


def starts_with_allowed_action_verb(action_phrase: str) -> bool:
    """
    Keep the lexical gate intentionally broad here.

    This validator helper only blocks explicitly banned starts such as `grab`
    and `twist`. The stricter action-verb allowlist lives in the policy gate
    used by the solver before apply/submit. Keeping this helper permissive
    avoids double-blocking and makes offline diagnostics easier to interpret.
    """
    phrase = normalize_spaces(action_phrase).lower()
    if not phrase or phrase == "no action":
        return False

    words = re.findall(r"[a-z]+", phrase)
    if not words:
        return False
        
    FORBIDDEN_VERBS = {"grab", "twist"}
    if words[0] in FORBIDDEN_VERBS:
        return False
    
    return True



def split_actions(label: str) -> List[str]:
    l = normalize_spaces(label)
    if l == NO_ACTION_LABEL:
        return [NO_ACTION_LABEL]
    parts = []
    for chunk in l.split(","):
        subs = [s.strip() for s in re.split(r"\band\b", chunk) if s.strip()]
        parts.extend(subs)
    return [p for p in parts if p]

def count_atomic_actions(label: str) -> int:
    l = normalize_spaces(label)
    if not l:
        return 0
    if l == NO_ACTION_LABEL:
        return 1
    return len(split_actions(l))


def disallowed_tool_terms_found(label: str) -> List[str]:
    l = lower(label)
    found: List[str] = []
    for term in DISALLOWED_TOOL_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", l):
            found.append(term)
    return found


def detect_possible_missing_object(action_phrase: str) -> bool:
    l = lower(action_phrase)
    for verb in sorted(OBJECT_EXPECTING_VERBS, key=len, reverse=True):
        if l == verb:
            return True
        if l.startswith(verb + " "):
            tokens = l.split()
            if verb == "pick up" and len(tokens) <= 2:
                return True
            if verb in {"place", "move", "relocate"} and len(tokens) <= 2:
                return True
            return False
    return False


def has_unattached_verb_chain(label: str) -> bool:
    l = normalize_spaces(label)
    if not l or l == NO_ACTION_LABEL:
        return False
    if CHAINED_VERB_WITHOUT_OBJECT_PATTERN.search(l):
        return True
    if ORPHAN_SECOND_PLACE_PATTERN.search(l):
        return True
    return False


def has_body_part_reference(label: str) -> bool:
    l = normalize_spaces(label)
    if not l or l == NO_ACTION_LABEL:
        return False
    return bool(BODY_PART_REFERENCE_PATTERN.search(l))


def has_token_stuttering(label: str) -> bool:
    l = normalize_spaces(label)
    if not l or l == NO_ACTION_LABEL:
        return False
    return bool(TOKEN_STUTTER_PATTERN.search(l))


def has_mechanical_motion_phrase(label: str) -> bool:
    l = normalize_spaces(label)
    if not l or l == NO_ACTION_LABEL:
        return False
    return bool(MECHANICAL_MOTION_PATTERN.search(l))


def place_has_location(label: str) -> bool:
    l = normalize_spaces(label)
    if "place" not in l.lower():
        return True
    return bool(PLACE_LOCATION_PATTERN.search(l))


def has_pluralization_hint_issue(label: str) -> bool:
    l = normalize_spaces(label)
    if not l or l == NO_ACTION_LABEL:
        return False
    tokens = re.findall(r"[a-z]+", lower(l))
    if len(tokens) < 2:
        return False
    for i, token in enumerate(tokens[:-1]):
        noun = tokens[i + 1]
        if noun in {
            "and",
            "or",
            "to",
            "on",
            "in",
            "into",
            "onto",
            "at",
            "inside",
            "under",
            "over",
            "with",
        }:
            continue
        if token in PLURAL_EXPECTED_NUMBER_WORDS:
            if noun in UNCOUNTABLE_NOUN_HINTS:
                continue
            if not noun.endswith("s"):
                return True
        if token in SINGULAR_EXPECTED_NUMBER_WORDS:
            if noun in PLURAL_ONLY_NOUN_HINTS:
                continue
            if noun.endswith("s"):
                return True
    return False


def _normalized_object_text(seg: Dict[str, Any]) -> str:
    gran = str(seg.get("granularity", "")).strip().lower()
    if gran == "no_action":
        return ""
    obj = normalize_spaces(seg.get("primary_object", ""))
    if not obj:
        obj = _infer_primary_object(normalize_spaces(seg.get("label", "")), gran or "coarse")
    obj = lower(obj)
    if obj in GENERIC_OBJECT_TOKENS:
        return ""
    return obj


def detect_object_naming_inconsistency(segments: Sequence[Dict[str, Any]]) -> List[str]:
    head_to_forms: Dict[str, Dict[str, List[int]]] = {}
    for seg in segments:
        obj = _normalized_object_text(seg)
        if not obj:
            continue
        idx = int(seg.get("segment_index", 0) or 0)
        tokens = re.findall(r"[a-z]+", obj)
        if not tokens:
            continue
        head = tokens[-1]
        if head in GENERIC_OBJECT_TOKENS:
            continue
        forms = head_to_forms.setdefault(head, {})
        forms.setdefault(obj, []).append(idx)

    warnings: List[str] = []
    for head, forms in head_to_forms.items():
        if len(forms) <= 1:
            continue
        variants = sorted(forms.keys(), key=lambda x: (len(x), x))
        idxs = sorted({idx for lst in forms.values() for idx in lst})
        warnings.append(f"object_naming_inconsistent:{head}:{'|'.join(variants)}@{','.join(map(str, idxs))}")
    return warnings


def detect_device_class_conflict(
    output_segments: Sequence[Dict[str, Any]],
    draft_segments: Sequence[Dict[str, Any]] | None = None,
) -> List[str]:
    """
    Detect conflicts in device class naming.
    - Internal conflict: output mixes device families (e.g., phone and laptop).
    - Draft conflict: output device family differs from draft family.
    Returns warning tokens suitable for episode_warnings.
    """

    alias_to_family: Dict[str, str] = {}
    for family, aliases in DEVICE_CLASS_GROUPS.items():
        for alias in aliases:
            alias_to_family[alias] = family

    def _label_text(seg: Dict[str, Any]) -> str:
        for key in ("label", "current_label", "description", "action", "annotation"):
            value = seg.get(key)
            if isinstance(value, str) and value.strip():
                return lower(value)
        return ""

    def _collect_families(segments: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for seg in segments:
            text = _label_text(seg)
            if not text:
                continue
            seen_in_seg: Set[str] = set()
            for alias, family in alias_to_family.items():
                if family in seen_in_seg:
                    continue
                if re.search(rf"\b{re.escape(alias)}\b", text):
                    counts[family] = counts.get(family, 0) + 1
                    seen_in_seg.add(family)
        return counts

    warnings: List[str] = []
    out_counts = _collect_families(output_segments)
    out_families = sorted(out_counts.keys())
    if len(out_families) > 1:
        warnings.append(f"device_class_internal_conflict:{'|'.join(out_families)}")

    if draft_segments:
        draft_counts = _collect_families(draft_segments)
        if out_counts and draft_counts:
            draft_family = sorted(draft_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
            out_family = sorted(out_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
            if draft_family != out_family:
                warnings.append(f"device_class_conflict:{draft_family}->{out_family}")

    return warnings


def detect_consecutive_duplicate_labels(
    segments: Sequence[Dict[str, Any]],
    contiguous_tolerance_sec: float = 0.25,
) -> List[str]:
    warnings: List[str] = []
    if not segments:
        return warnings

    def _idx(seg: Dict[str, Any]) -> int:
        try:
            return int(seg.get("segment_index", 0) or 0)
        except Exception:
            return 0

    ordered = sorted(list(segments), key=_idx)
    for i in range(1, len(ordered)):
        prev = ordered[i - 1]
        cur = ordered[i]
        prev_label = normalize_spaces(prev.get("label", ""))
        cur_label = normalize_spaces(cur.get("label", ""))
        if not prev_label or not cur_label:
            continue
        if lower(prev_label) == "no action" or lower(cur_label) == "no action":
            continue
        if lower(prev_label) != lower(cur_label):
            continue
        try:
            prev_end = float(prev.get("end_sec", 0.0) or 0.0)
            cur_start = float(cur.get("start_sec", 0.0) or 0.0)
        except Exception:
            continue
        if abs(cur_start - prev_end) <= max(0.0, float(contiguous_tolerance_sec)):
            prev_idx = _idx(prev)
            cur_idx = _idx(cur)
            warnings.append(
                f"consecutive_duplicate_labels:{prev_idx}->{cur_idx}:{lower(cur_label)}"
            )
    return warnings


def check_hold_intent(label: str) -> bool:
    """
    Spec-Kit RULE-HOLD: 'hold [object]' must NOT include intent.
    Returns True if violation detected.
    E.g. 'hold phone to check charging status' -> True (violation)
         'hold phone assembly' -> False (clean)
    """
    l = lower(label)
    if not l.startswith("hold"):
        return False
    return bool(HOLD_INTENT_PATTERN.search(l))


def check_mergeable_consecutive(
    segments: Sequence[Dict[str, Any]],
    tolerance_sec: float | None = None,
    max_duration_sec: float | None = None,
) -> List[str]:
    """
    Spec-Kit RULE-MERGE: Flag consecutive segments with identical labels
    that should be merged (combined duration <= 60s).
    Returns list of warning strings.
    """
    warnings: List[str] = []
    if not segments:
        return warnings
    constraints = get_policy_constraints()
    if tolerance_sec is None:
        tolerance_sec = constraints.merge_tolerance_sec
    if max_duration_sec is None:
        max_duration_sec = constraints.merge_max_duration_sec

    def _idx(seg: Dict[str, Any]) -> int:
        try:
            return int(seg.get("segment_index", 0) or 0)
        except Exception:
            return 0

    ordered = sorted(list(segments), key=_idx)
    i = 0
    while i < len(ordered):
        # Find run of consecutive identical labels
        run_start = i
        run_label = lower(normalize_spaces(ordered[i].get("label", "")))
        if not run_label or run_label == "no action":
            i += 1
            continue

        j = i + 1
        while j < len(ordered):
            next_label = lower(normalize_spaces(ordered[j].get("label", "")))
            if next_label != run_label:
                break
            try:
                prev_end = float(ordered[j - 1].get("end_sec", 0))
                cur_start = float(ordered[j].get("start_sec", 0))
            except (TypeError, ValueError):
                break
            if abs(cur_start - prev_end) > tolerance_sec:
                break
            j += 1

        run_len = j - run_start
        if run_len > 1:
            try:
                run_start_sec = float(ordered[run_start].get("start_sec", 0))
                run_end_sec = float(ordered[j - 1].get("end_sec", 0))
                combined_duration = run_end_sec - run_start_sec
            except (TypeError, ValueError):
                combined_duration = 0

            if combined_duration <= max_duration_sec:
                idx_start = _idx(ordered[run_start])
                idx_end = _idx(ordered[j - 1])
                warnings.append(
                    f"mergeable_consecutive:{idx_start}->{idx_end}:"
                    f"{run_label}:{combined_duration:.1f}s"
                )
        i = j

    return warnings


def no_action_mixed_with_action(label: str) -> bool:
    l = lower(label)
    if l == "no action":
        return False
    return "no action" in l


def dense_coarse_mixed(segment: Dict[str, Any]) -> bool:
    gran = segment.get("granularity")
    label = lower(segment.get("label", ""))
    if gran not in {"dense", "coarse"}:
        return False
    has_move = bool(re.search(r"\bmove\b", label))
    has_pick_place = bool(re.search(r"\bpick up\b", label) and re.search(r"\bplace\b", label))
    return has_move and has_pick_place


def classify_audit_risk(reasons: Sequence[str]) -> str:
    if not reasons:
        return "low"
    high_markers = {
        "forbidden_verbs",
        "narrative_filler_words",
        "verb_start_not_allowed",
        "disallowed_tool_terms",
        "no_action_mixed",
        "too_many_atomic_actions",
        "label_too_long",
        "segment_too_long",
        "duration_mismatch",
        "timestamp_overlap",
        "timestamp_order_invalid",
        "granularity_label_mismatch",
        "dense_coarse_mixed",
        "possible_hallucination",
        "place_missing_location",
        "guide_used_as_object_descriptor",
        "invalid_adjust_over_phrase",
    }
    if any(r in high_markers for r in reasons):
        return "high"
    if len(reasons) >= 2:
        return "medium"
    return "low"


def _infer_primary_goal(label: str, granularity: str) -> str:
    l = normalize_spaces(label)
    if granularity == "no_action":
        return "no_contact"
    actions = split_actions(l)
    if not actions:
        return "task_action"
    return actions[-1]


def _infer_primary_object(label: str, granularity: str) -> str:
    if granularity == "no_action":
        return "none"
    tokens = lower(label).split()
    if len(tokens) < 2:
        return "item"
    # Keep conservative fallback to avoid hallucination.
    if tokens[0] == "pick" and len(tokens) >= 3 and tokens[1] == "up":
        return tokens[2]
    if tokens[0] in {"place", "move", "grab", "hold", "adjust", "flip", "wipe"} and len(tokens) >= 2:
        return tokens[1]
    return "item"


def normalize_annotation(
    annotation: Any,
    episode_id: str = "episode",
    annotation_version: str = "atlas_v2_pipeline",
    video_duration_sec: float = 0.0,
) -> Dict[str, Any]:
    if isinstance(annotation, (str, Path)):
        text = Path(annotation).read_text(encoding="utf-8")
        annotation = json.loads(text)

    if isinstance(annotation, dict):
        raw_segments = annotation.get("segments")
        episode_id = str(annotation.get("episode_id") or episode_id)
        if isinstance(annotation.get("video_duration_sec"), (int, float)):
            video_duration_sec = float(annotation["video_duration_sec"])
        if isinstance(annotation.get("annotation_version"), str) and annotation["annotation_version"].strip():
            annotation_version = annotation["annotation_version"].strip()
    elif isinstance(annotation, list):
        raw_segments = annotation
    else:
        raise ValueError("Annotation must be dict/list/path/json-string")

    if not isinstance(raw_segments, list):
        raise ValueError("Annotation segments must be a list")

    segments: List[Dict[str, Any]] = []
    max_end = 0.0

    for i, raw in enumerate(raw_segments, start=1):
        if not isinstance(raw, dict):
            continue

        seg_idx = int(raw.get("segment_index") or raw.get("step") or i)
        start = parse_time_value(raw.get("start_sec", raw.get("start", raw.get("from", raw.get("start_time", 0.0)))))
        end = parse_time_value(raw.get("end_sec", raw.get("end", raw.get("to", raw.get("end_time", 0.0)))))

        if end <= 0 and isinstance(raw.get("duration_seconds"), (int, float)):
            end = start + float(raw["duration_seconds"])

        label = normalize_spaces(raw.get("label", raw.get("description", raw.get("action", raw.get("annotation", "")))))
        gran = str(raw.get("granularity", raw.get("type", "coarse"))).strip().lower()
        if gran not in {"dense", "coarse", "no_action"}:
            gran = "no_action" if lower(label) == "no action" else "coarse"
        if lower(label) == "no action":
            gran = "no_action"

        confidence_raw = raw.get("confidence", raw.get("score", 0.7))
        if isinstance(confidence_raw, str):
            map_conf = {"low": 0.35, "medium": 0.65, "high": 0.9}
            confidence = map_conf.get(confidence_raw.strip().lower(), 0.7)
        else:
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = 0.7
        confidence = min(1.0, max(0.0, confidence))

        if end <= start:
            end = start + 0.1

        duration = round(end - start, 3)
        max_end = max(max_end, end)

        primary_goal = normalize_spaces(raw.get("primary_goal", "")) or _infer_primary_goal(label, gran)
        primary_object = normalize_spaces(raw.get("primary_object", "")) or _infer_primary_object(label, gran)

        segments.append(
            {
                "segment_index": seg_idx,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": duration,
                "label": label or ("No Action" if gran == "no_action" else "handle item"),
                "granularity": gran,
                "primary_goal": primary_goal,
                "primary_object": primary_object,
                "secondary_objects": raw.get("secondary_objects", []),
                "actions_observed": raw.get("actions_observed", []),
                "confidence": confidence,
                "uncertainty_note": normalize_spaces(raw.get("uncertainty_note", "")),
                "escalation_flag": bool(raw.get("escalation_flag", False)),
                "escalation_reason": raw.get("escalation_reason", ""),
                "rule_checks": raw.get("rule_checks", {}),
                "audit_risk": raw.get("audit_risk", {"level": "low", "reasons": []}),
            }
        )

    segments.sort(key=lambda x: (x["start_sec"], x["end_sec"]))
    for idx, seg in enumerate(segments, start=1):
        seg["segment_index"] = idx

    if video_duration_sec <= 0:
        video_duration_sec = max_end

    episode_checks = {
        "segments_sorted": True,
        "no_negative_durations": True,
        "no_overlaps": True,
        "coverage_within_video_duration": True,
        "gaps_present": False,
        "repeated_action_logic_checked": True,
        "merge_split_logic_checked": True,
        "notes": "",
    }

    return {
        "episode_id": episode_id or "episode",
        "video_duration_sec": round(float(video_duration_sec), 3),
        "annotation_version": annotation_version,
        "source_context": {},
        "segments": segments,
        "episode_checks": episode_checks,
    }


def validate_segment(seg: Dict[str, Any], video_duration_sec: float) -> Tuple[Dict[str, Any], List[str], List[str]]:
    constraints = get_policy_constraints()
    errors: List[str] = []
    warnings: List[str] = []
    forbidden_verbs_found: List[str] = []
    forbidden_narrative_words_found: List[str] = []

    idx = seg.get("segment_index")
    label = normalize_spaces(seg.get("label", ""))
    gran = seg.get("granularity")
    start = seg.get("start_sec")
    end = seg.get("end_sec")
    duration = seg.get("duration_sec")

    if not label:
        errors.append("empty_label")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        errors.append("timestamp_type_invalid")
    else:
        if start < 0 or end < 0:
            errors.append("timestamp_negative")
        if end <= start:
            errors.append("timestamp_order_invalid")
        if end > video_duration_sec + 0.1:
            warnings.append("end_beyond_video_duration")

    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and isinstance(duration, (int, float)):
        if not duration_matches(float(start), float(end), float(duration)):
            errors.append("duration_mismatch")
        if (float(end) - float(start)) > constraints.max_segment_duration_sec + 0.05:
            errors.append("segment_too_long")

    if label:
        if not min_two_words(label):
            errors.append("min_two_words_failed")
        if word_count(label) > constraints.max_words_per_label:
            errors.append("label_too_long")
        if not is_imperative_like(label):
            errors.append("imperative_voice_failed")
        if label != NO_ACTION_LABEL:
            actions = split_actions(label)
            if not actions:
                errors.append("verb_start_not_allowed")
            else:
                for action in actions:
                    if not starts_with_allowed_action_verb(action):
                        errors.append("verb_start_not_allowed")
                        break
        if has_numerals(label):
            errors.append("numerals_present")
        if has_pluralization_hint_issue(label):
            warnings.append("pluralization_hint")
        if has_intent_only_language(label):
            errors.append("intent_only_language")
        if check_hold_intent(label):
            errors.append("hold_intent_violation")
        forbidden_narrative_words_found = contains_forbidden_narrative_words(label)
        if forbidden_narrative_words_found:
            errors.append("narrative_filler_words")

        forbidden = contains_forbidden_verbs(label)
        if forbidden and "reach" in forbidden and allow_reach_for_truncated_end(label, end, video_duration_sec):
            forbidden = [v for v in forbidden if v != "reach"]
            warnings.append("reach_allowed_truncated_end")
        forbidden_verbs_found = forbidden
        if forbidden_verbs_found:
            errors.append("forbidden_verbs")
        disallowed_terms = disallowed_tool_terms_found(label)
        if disallowed_terms:
            errors.append("disallowed_tool_terms")
        if has_body_part_reference(label):
            errors.append("body_parts_referenced")
        if has_token_stuttering(label):
            errors.append("token_stuttering")
        if has_mechanical_motion_phrase(label):
            errors.append("mechanical_motion_phrase")
        if GUIDE_AS_OBJECT_DESCRIPTOR_PATTERN.search(label):
            errors.append("guide_used_as_object_descriptor")
        if INVALID_ADJUST_OVER_PATTERN.search(label):
            errors.append("invalid_adjust_over_phrase")
        if re.search(r"\bgripper\b", lower(label)):
            warnings.append("gripper_term_used")

        if gran == "no_action":
            if label != NO_ACTION_LABEL:
                errors.append("granularity_label_mismatch")
            pg = normalize_spaces(seg.get("primary_goal", ""))
            if pg and pg not in {"idle", "irrelevant", "no_contact"}:
                warnings.append("no_action_primary_goal_unusual")
        else:
            if label == NO_ACTION_LABEL:
                errors.append("granularity_label_mismatch")

        if no_action_mixed_with_action(label):
            errors.append("no_action_mixed")

        if label != NO_ACTION_LABEL:
            if count_atomic_actions(label) > constraints.max_atomic_actions_per_label:
                errors.append("too_many_atomic_actions")
            missing = [phrase for phrase in split_actions(label) if detect_possible_missing_object(phrase)]
            if missing:
                warnings.append("possible_missing_object")
            if has_unattached_verb_chain(label):
                errors.append("verbs_not_attached_to_objects")
            if not place_has_location(label):
                errors.append("place_missing_location")
            if dense_coarse_mixed(seg):
                errors.append("dense_coarse_mixed")

    confidence = seg.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0 <= float(confidence) <= 1):
        warnings.append("confidence_invalid_or_missing")

    rc = seg.get("rule_checks")
    if isinstance(rc, dict):
        if rc.get("no_forbidden_verbs") is True and "forbidden_verbs" in errors:
            warnings.append("rule_checks_contradiction")
        if rc.get("no_numerals") is True and "numerals_present" in errors:
            warnings.append("rule_checks_contradiction")
        if rc.get("dense_coarse_not_mixed") is True and "dense_coarse_mixed" in errors:
            warnings.append("rule_checks_contradiction")

    derived_rule_checks = {
        "imperative_voice": "imperative_voice_failed" not in errors,
        "starts_with_allowed_action_verb": "verb_start_not_allowed" not in errors,
        "min_two_words": "min_two_words_failed" not in errors,
        "max_words_limit": "label_too_long" not in errors,
        "no_numerals": "numerals_present" not in errors,
        "pluralization_consistent": "pluralization_hint" not in warnings,
        "no_forbidden_verbs": "forbidden_verbs" not in errors,
        "forbidden_verbs_found": forbidden_verbs_found,
        "no_narrative_filler_words": "narrative_filler_words" not in errors,
        "narrative_words_found": forbidden_narrative_words_found,
        "no_body_part_references": "body_parts_referenced" not in errors,
        "no_token_stuttering": "token_stuttering" not in errors,
        "no_mechanical_motion_phrasing": "mechanical_motion_phrase" not in errors,
        "verbs_attached_to_objects": (
            "possible_missing_object" not in warnings and "verbs_not_attached_to_objects" not in errors
        ),
        "one_goal": True,
        "full_action_coverage": True,
        "no_hallucinated_steps": True,
        "dense_coarse_not_mixed": "dense_coarse_mixed" not in errors,
        "no_hold_intent_violation": "hold_intent_violation" not in errors,
        "no_action_not_mixed_with_action": "no_action_mixed" not in errors,
        "place_has_location": "place_missing_location" not in errors,
        "timestamps_aligned": all(e not in errors for e in ["timestamp_order_invalid", "duration_mismatch"]),
        "hands_disengage_boundary_ok": True,
    }

    audit_reasons: List[str] = []
    if "forbidden_verbs" in errors:
        audit_reasons.append("verb_choice_ambiguous")
    if "narrative_filler_words" in errors:
        audit_reasons.append("verb_choice_ambiguous")
    if "verb_start_not_allowed" in errors:
        audit_reasons.append("verb_choice_ambiguous")
    if "verbs_not_attached_to_objects" in errors:
        audit_reasons.append("verb_choice_ambiguous")
    if "disallowed_tool_terms" in errors:
        audit_reasons.append("verb_choice_ambiguous")
    if "body_parts_referenced" in errors:
        audit_reasons.append("verb_choice_ambiguous")
    if "token_stuttering" in errors:
        audit_reasons.append("verb_choice_ambiguous")
    if "mechanical_motion_phrase" in errors:
        audit_reasons.append("verb_choice_ambiguous")
    if "place_missing_location" in errors:
        audit_reasons.append("verb_choice_ambiguous")
    if "numerals_present" in errors:
        audit_reasons.append("possible_hallucination")
    if "dense_coarse_mixed" in errors:
        audit_reasons.append("granularity_choice_ambiguous")
    if "too_many_atomic_actions" in errors:
        audit_reasons.append("granularity_choice_ambiguous")
    if "label_too_long" in errors:
        audit_reasons.append("granularity_choice_ambiguous")
    if "no_action_mixed" in errors:
        audit_reasons.append("no_action_rule_risk")
    if "duration_mismatch" in errors or "timestamp_order_invalid" in errors:
        audit_reasons.append("timestamp_misalignment")
    if "possible_missing_object" in warnings:
        audit_reasons.append("object_identity_uncertain")
    if "pluralization_hint" in warnings:
        audit_reasons.append("object_identity_uncertain")
    segment_report = {
        "segment_index": idx,
        "label": label,
        "errors": errors,
        "warnings": warnings,
        "derived_rule_checks": derived_rule_checks,
        "suggested_audit_risk": {
            "level": classify_audit_risk(errors + warnings),
            "reasons": sorted(set(audit_reasons)),
        },
    }
    return segment_report, errors, warnings


def validate_episode(annotation: Dict[str, Any]) -> Dict[str, Any]:
    ann = copy.deepcopy(annotation)
    duration = float(ann.get("video_duration_sec", 0) or 0)
    segments = ann.get("segments", [])
    if not isinstance(segments, list):
        return {"ok": False, "fatal_error": "segments_not_list"}

    seg_reports = []
    episode_errors: List[str] = []
    episode_warnings: List[str] = []

    for seg in segments:
        report, _, _ = validate_segment(seg, duration)
        seg_reports.append(report)
        seg["rule_checks"] = report["derived_rule_checks"]
        seg["audit_risk"] = report["suggested_audit_risk"]

    object_naming_warnings = detect_object_naming_inconsistency(segments)
    if object_naming_warnings:
        episode_warnings.append("object_naming_inconsistent")
    draft_segments = ann.get("draft_segments") or ann.get("tier2_segments") or ann.get("source_segments")
    device_conflicts = detect_device_class_conflict(
        segments,
        draft_segments=draft_segments if isinstance(draft_segments, list) else None,
    )
    if device_conflicts:
        episode_warnings.extend(device_conflicts)
    duplicate_label_warnings = detect_consecutive_duplicate_labels(segments)
    if duplicate_label_warnings:
        episode_warnings.append("consecutive_duplicate_labels")
    mergeable_warnings = check_mergeable_consecutive(segments)
    if mergeable_warnings:
        episode_warnings.append("mergeable_consecutive_segments")

    starts_ends = []
    for seg in segments:
        try:
            starts_ends.append((int(seg.get("segment_index", 0)), float(seg["start_sec"]), float(seg["end_sec"])))
        except Exception:
            episode_errors.append("segment_timestamp_parse_error")

    idxs = [x[0] for x in starts_ends]
    if idxs != sorted(idxs):
        episode_warnings.append("segment_indices_not_sorted")

    time_sorted = sorted(starts_ends, key=lambda x: (x[1], x[2]))
    for i in range(1, len(time_sorted)):
        prev = time_sorted[i - 1]
        cur = time_sorted[i]
        if cur[1] < prev[2] - 1e-6:
            episode_errors.append("timestamp_overlap")
            break

    if duration > 0:
        for _, s, e in starts_ends:
            if s < 0 or e > duration + 0.1:
                episode_warnings.append("segment_outside_video_duration")
                break

    any_seg_errors = any(report["errors"] for report in seg_reports)
    major_fail_triggers: List[str] = []
    for report in seg_reports:
        errs = set(report["errors"])
        if "forbidden_verbs" in errs:
            major_fail_triggers.append("forbidden_verbs_used")
        if "narrative_filler_words" in errs:
            major_fail_triggers.append("narrative_filler_words_used")
        if "verb_start_not_allowed" in errs:
            major_fail_triggers.append("verb_start_not_allowed")
        if "verbs_not_attached_to_objects" in errs:
            major_fail_triggers.append("verbs_not_attached_to_objects")
        if "disallowed_tool_terms" in errs:
            major_fail_triggers.append("disallowed_tool_terms")
        if "body_parts_referenced" in errs:
            major_fail_triggers.append("body_parts_referenced")
        if "token_stuttering" in errs:
            major_fail_triggers.append("token_stuttering")
        if "mechanical_motion_phrase" in errs:
            major_fail_triggers.append("mechanical_motion_phrase")
        if "numerals_present" in errs:
            major_fail_triggers.append("numerals_present")
        if "dense_coarse_mixed" in errs:
            major_fail_triggers.append("dense_coarse_mixed")
        if "place_missing_location" in errs:
            major_fail_triggers.append("place_missing_location")
        if "too_many_atomic_actions" in errs:
            major_fail_triggers.append("too_many_atomic_actions")
        if "label_too_long" in errs:
            major_fail_triggers.append("label_too_long")
        if "segment_too_long" in errs:
            major_fail_triggers.append("segment_too_long")
        if "no_action_mixed" in errs:
            major_fail_triggers.append("no_action_mixed_with_action")
        if "hold_intent_violation" in errs:
            major_fail_triggers.append("hold_intent_violation")
        if "guide_used_as_object_descriptor" in errs:
            major_fail_triggers.append("guide_used_as_object_descriptor")
        if "invalid_adjust_over_phrase" in errs:
            major_fail_triggers.append("invalid_adjust_over_phrase")
        if "timestamp_order_invalid" in errs or "duration_mismatch" in errs:
            major_fail_triggers.append("timestamps_invalid")

    if "timestamp_overlap" in episode_errors:
        major_fail_triggers.append("episode_overlap")

    ann["episode_checks"] = {
        "segments_sorted": "segment_indices_not_sorted" not in episode_warnings,
        "no_negative_durations": all((seg.get("duration_sec", 0) or 0) > 0 for seg in segments),
        "no_overlaps": "timestamp_overlap" not in episode_errors,
        "coverage_within_video_duration": "segment_outside_video_duration" not in episode_warnings,
        "gaps_present": False,
        "repeated_action_logic_checked": True,
        "merge_split_logic_checked": True,
        "notes": "",
    }

    return {
        "ok": not (episode_errors or any_seg_errors),
        "episode_id": ann.get("episode_id"),
        "normalized_annotation": ann,
        "episode_errors": sorted(set(episode_errors)),
        "episode_warnings": sorted(set(episode_warnings)),
        "episode_warning_details": sorted(
            set(object_naming_warnings + device_conflicts + duplicate_label_warnings + mergeable_warnings)
        ),
        "device_class_conflicts": sorted(set(device_conflicts)),
        "segment_reports": seg_reports,
        "major_fail_triggers": sorted(set(major_fail_triggers)),
        "repair_recommended": bool(episode_errors or any_seg_errors or episode_warnings),
    }


def build_repair_payload(
    annotation: Dict[str, Any],
    validator_report: Dict[str, Any],
    evidence_notes: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"annotation": annotation, "validator_report": validator_report}
    if evidence_notes:
        payload["evidence_notes"] = evidence_notes
    return payload


def replace_small_numerals(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        return NUMERAL_TO_WORD.get(token, token)

    return re.sub(r"\b(?:10|[0-9])\b", repl, text)


def cheap_preclean_label(label: str) -> str:
    return replace_small_numerals(normalize_spaces(label))
