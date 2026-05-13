"""Label policy helpers extracted from the legacy solver."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from src.infra.solver_config import _cfg_get
from src.infra.utils import _safe_float

_DISALLOWED_TOOL_TERMS = (
    "mechanical arm",
    "robotic arm",
    "robot arm",
    "manipulator",
    "robot gripper",
    "claw arm",
)

_LABEL_TOKEN_RE = re.compile(r"[a-z]+")
_LABEL_OVERLAP_STOPWORDS: set[str] = {
    "no",
    "action",
    "with",
    "on",
    "in",
    "into",
    "onto",
    "at",
    "to",
    "from",
    "under",
    "over",
    "inside",
}

_AUTOFIX_ALLOWED_LABEL_START_VERB_TOKEN_PATTERNS: Tuple[Tuple[str, ...], ...] = (
    ("pick", "up"),
    ("put", "down"),
    ("place",),
    ("move",),
    ("adjust",),
    ("align",),
    ("hold",),
    ("cut",),
    ("open",),
    ("close",),
    ("peel",),
    ("secure",),
    ("wipe",),
    ("flip",),
    ("pull",),
    ("push",),
    ("insert",),
    ("remove",),
    ("attach",),
    ("detach",),
    ("connect",),
    ("disconnect",),
    ("tighten",),
    ("loosen",),
    ("screw",),
    ("unscrew",),
    ("press",),
    ("twist",),
    ("turn",),
    ("slide",),
    ("lift",),
    ("lower",),
    ("set",),
    ("position",),
    ("straighten",),
    ("comb",),
    ("detangle",),
    ("sand",),
    ("paint",),
    ("clean",),
    ("put",),
    ("stir",),
    ("mix",),
    ("blend",),
    ("roll",),
    ("fold",),
    ("spread", "out"),
    ("hang",),
    ("stack",),
    ("pour",),
    ("scoop",),
    ("level",),
    ("pry", "open"),
    ("drive",),
    ("dig",),
    ("brush",),
)

_AUTOFIX_OBJECT_EXPECTING_VERBS: Tuple[str, ...] = (
    "pick up",
    "put down",
    "place",
    "move",
    "adjust",
    "align",
    "hold",
    "cut",
    "open",
    "close",
    "peel",
    "secure",
    "wipe",
    "flip",
    "pull",
    "push",
    "insert",
    "remove",
    "attach",
    "detach",
    "connect",
    "disconnect",
    "tighten",
    "loosen",
    "screw",
    "unscrew",
    "press",
    "twist",
    "turn",
    "slide",
    "lift",
    "lower",
    "set",
    "position",
    "straighten",
    "comb",
    "detangle",
    "sand",
    "paint",
    "clean",
    "put",
    "stir",
    "mix",
    "blend",
    "roll",
    "fold",
    "spread out",
    "hang",
    "stack",
    "pour",
    "scoop",
    "level",
    "pry open",
    "drive",
    "dig",
    "brush",
)

_AUTOFIX_VERB_HINT_MAP: Tuple[Tuple[str, str], ...] = (
    ("door", "open"),
    ("drawer", "open"),
    ("cabinet", "open"),
    ("box", "place"),
    ("paper", "place"),
    ("button", "press"),
    ("switch", "press"),
    ("cap", "close"),
    ("lid", "close"),
    ("door", "close"),
    ("nut", "tighten"),
)

_MICRO_ACTION_VERBS: set[str] = {"dip", "reload", "wet"}

_ING_TO_BASE_VERB_MAP: Dict[str, str] = {
    "applying": "apply",
    "painting": "paint",
    "brushing": "brush",
    "driving": "drive",
    "positioning": "position",
    "scraping": "scrape",
    "lifting": "lift",
    "turning": "turn",
    "setting": "set",
    "placing": "place",
    "moving": "move",
    "polishing": "polish",
    "sanding": "sand",
    "leveling": "level",
    "dislodging": "dislodge",
    "adjusting": "adjust",
    "opening": "open",
    "closing": "close",
    "cutting": "cut",
    "pulling": "pull",
    "pushing": "push",
    "holding": "hold",
    "inserting": "insert",
    "removing": "remove",
    "twisting": "twist",
    "pouring": "pour",
    "scooping": "scoop",
    "filling": "fill",
    "compacting": "compact",
}

_NUM_WORDS_0_TO_19 = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_NUM_TENS_WORDS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]
def _count_atomic_actions_in_label(label: str) -> int:
    text = (label or "").strip()
    if not text:
        return 0
    if text.lower() == "no action":
        return 1
    count = 0
    for part in re.split(r"\s*,\s*", text):
        chunk = part.strip()
        if not chunk:
            continue
        subparts = [p.strip() for p in re.split(r"\band\b", chunk, flags=re.IGNORECASE) if p.strip()]
        count += len(subparts) if subparts else 1
    return max(1, count)


def _normalize_gripper_terms(text: str) -> str:
    out = text or ""
    for term in _DISALLOWED_TOOL_TERMS:
        out = re.sub(rf"\b{re.escape(term)}\b", "gripper", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def _label_main_verb(label: str) -> str:
    text = re.sub(r"\s+", " ", (label or "").strip()).lower()
    if not text:
        return ""
    match = re.match(r"([a-z]+)", text)
    return match.group(1) if match else ""


def _is_no_action_label(label: str) -> bool:
    normalized = re.sub(r"[\s_-]+", " ", (label or "").strip()).lower()
    return normalized in {"no action", "noaction"}


def _label_content_tokens(label: str) -> set[str]:
    text = re.sub(r"\s+", " ", (label or "").strip()).lower()
    if not text:
        return set()
    tokens = set(_LABEL_TOKEN_RE.findall(text))
    return {tok for tok in tokens if tok and tok not in _LABEL_OVERLAP_STOPWORDS}


def _allowed_label_start_verb_token_patterns_from_cfg(cfg: Dict[str, Any]) -> List[Tuple[str, ...]]:
    raw = _cfg_get(cfg, "run.allowed_label_start_verbs", [])
    patterns: List[Tuple[str, ...]] = []
    if isinstance(raw, list):
        for item in raw:
            tokens = tuple(re.findall(r"[a-z]+", str(item).lower()))
            if tokens:
                patterns.append(tokens)
    if not patterns:
        patterns = list(_AUTOFIX_ALLOWED_LABEL_START_VERB_TOKEN_PATTERNS)
    deduped: List[Tuple[str, ...]] = []
    seen: set[Tuple[str, ...]] = set()
    for pattern in patterns:
        if pattern in seen:
            continue
        seen.add(pattern)
        deduped.append(pattern)
    return deduped


def _label_starts_with_allowed_action_verb(
    action_phrase: str,
    allowed_verb_token_patterns: List[Tuple[str, ...]],
) -> bool:
    phrase = re.sub(r"\s+", " ", (action_phrase or "").strip()).lower()
    if not phrase or phrase == "no action":
        return False
    words = re.findall(r"[a-z]+", phrase)
    if not words:
        return False
    for pattern in allowed_verb_token_patterns:
        if not pattern:
            continue
        size = len(pattern)
        if len(words) >= size and tuple(words[:size]) == pattern:
            if any(word.endswith("ing") for word in words[:size]):
                return False
            return True
    return False


def _contains_forbidden_verb_in_label(label: str, forbidden_verbs: List[str]) -> bool:
    text = (label or "").strip().lower()
    if not text:
        return False
    for verb in forbidden_verbs:
        if re.search(rf"\b{re.escape(verb)}\b", text):
            return True
    return False


def _strip_forbidden_verbs_for_autofix(label: str, forbidden_verbs: List[str]) -> str:
    out = label or ""
    for verb in forbidden_verbs:
        if not verb:
            continue
        out = re.sub(rf"\b{re.escape(verb)}\b", "", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip(" ,.;:")


def _action_phrase_missing_object_for_autofix(action_phrase: str) -> bool:
    phrase = re.sub(r"\s+", " ", (action_phrase or "").strip()).lower()
    if not phrase:
        return True
    for verb in sorted(_AUTOFIX_OBJECT_EXPECTING_VERBS, key=len, reverse=True):
        if phrase == verb:
            return True
        if phrase.startswith(verb + " "):
            remaining = phrase[len(verb):].strip()
            if not remaining:
                return True
            return len(re.findall(r"[a-z]+", remaining)) == 0
    return False


def _heuristic_autofix_verb_from_text(text: str) -> str:
    lowered = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if lowered:
        for needle, verb in _AUTOFIX_VERB_HINT_MAP:
            if needle in lowered:
                return verb
    return "pick up"


def _autofix_label_candidate(
    cfg: Dict[str, Any],
    label: str,
    source_label: str,
    forbidden_verbs: List[str],
    allowed_verb_token_patterns: List[Tuple[str, ...]],
) -> str:
    min_words = max(1, int(_cfg_get(cfg, "run.min_label_words", 2)))
    max_words = max(min_words, int(_cfg_get(cfg, "run.max_label_words", 20)))

    def _normalize(value: str) -> str:
        out = _normalize_label_min_safety(value)
        out = _strip_forbidden_verbs_for_autofix(out, forbidden_verbs)
        out = re.sub(r"\b(?:then|another|continue|next)\b", "", out, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", out).strip(" ,.;:")

    def _valid_candidate(value: str) -> bool:
        if not value or value.lower() == "no action":
            return False
        if not _label_starts_with_allowed_action_verb(value, allowed_verb_token_patterns):
            return False
        if _contains_forbidden_verb_in_label(value, forbidden_verbs):
            return False
        clauses = _label_action_clauses(value)
        if not clauses:
            return False
        for clause in clauses:
            if not _label_starts_with_allowed_action_verb(clause, allowed_verb_token_patterns):
                return False
            if _action_phrase_missing_object_for_autofix(clause):
                return False
        return True

    for base in (label, source_label):
        candidate = _normalize(base)
        if _valid_candidate(candidate):
            words = [word for word in candidate.split() if word]
            if len(words) < min_words:
                candidate = f"{candidate} item".strip()
            words = [word for word in candidate.split() if word]
            if len(words) > max_words:
                candidate = candidate.split(",", 1)[0].strip() if "," in candidate else " ".join(words[:max_words])
            return candidate

    base_text = _normalize(label or source_label or "")
    base_tokens = re.findall(r"[a-z]+", base_text.lower())
    object_tokens = list(base_tokens)
    for pattern in allowed_verb_token_patterns:
        size = len(pattern)
        if size > 0 and len(base_tokens) >= size and tuple(base_tokens[:size]) == pattern:
            object_tokens = base_tokens[size:]
            break
    object_tokens = [token for token in object_tokens if token not in {"and", "then"}]
    object_phrase = " ".join(object_tokens).strip() or "item"
    verb = _heuristic_autofix_verb_from_text(base_text)
    candidate = _normalize(f"{verb} {object_phrase}")
    if not _label_starts_with_allowed_action_verb(candidate, allowed_verb_token_patterns):
        candidate = _normalize(f"pick up {object_phrase}")
    if not candidate:
        candidate = "pick up item"
    words = [word for word in candidate.split() if word]
    if len(words) < min_words:
        candidate = f"{candidate} item".strip()
    words = [word for word in candidate.split() if word]
    if len(words) > max_words:
        candidate = " ".join(words[:max_words])
    return candidate


def _starts_with_common_label_verb(text: str) -> bool:
    lower = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if not lower:
        return False
    for verb in _COMMON_LABEL_START_VERBS:
        if lower == verb or lower.startswith(f"{verb} "):
            return True
    return False


def _split_clause_on_action_and(chunk: str) -> List[str]:
    text = re.sub(r"\s+", " ", (chunk or "").strip(" ,"))
    if not text:
        return []

    parts = [text]
    changed = True
    while changed:
        changed = False
        next_parts: List[str] = []
        for part in parts:
            match_found = False
            for match in re.finditer(r"\band\b", part, flags=re.IGNORECASE):
                head = re.sub(r"\s+", " ", part[: match.start()].strip(" ,"))
                tail = re.sub(r"\s+", " ", part[match.end() :].strip(" ,"))
                if not head or not tail:
                    continue
                if not _starts_with_common_label_verb(tail):
                    continue
                next_parts.extend([head, tail])
                changed = True
                match_found = True
                break
            if not match_found:
                next_parts.append(part)
        parts = next_parts
    return [part for part in parts if part]


def _label_action_clauses(label: str) -> List[str]:
    text = re.sub(r"\s+", " ", (label or "").strip())
    if not text:
        return []
    parts: List[str] = []
    for chunk in text.split(","):
        parts.extend(_split_clause_on_action_and(chunk))
    return parts


def _hold_clause_object(clause: str) -> str:
    text = re.sub(r"\s+", " ", (clause or "").strip(" ,.;:"))
    if _label_main_verb(text) != "hold":
        return ""
    match = re.match(r"^hold\s+(.+)$", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return re.sub(r"\s+", " ", str(match.group(1) or "").strip(" ,.;:"))


def _infer_held_object_context(
    segment_index: int,
    source_segments: List[Dict[str, Any]],
    normalized_plan: Optional[Dict[int, Dict[str, Any]]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    neighbors = max(0, int(_cfg_get(cfg or {}, "run.hold_rule_context_neighbors", 2)))
    source_by_idx: Dict[int, Dict[str, Any]] = {}
    for seg in source_segments:
        try:
            source_by_idx[int(seg.get("segment_index", 0) or 0)] = seg
        except Exception:
            continue

    scored: Dict[str, int] = {}
    surface: Dict[str, str] = {}
    for offset in range(0, neighbors + 1):
        candidate_indices = [segment_index - offset]
        if offset > 0:
            candidate_indices.append(segment_index + offset)
        for candidate_idx in candidate_indices:
            labels_to_scan: List[str] = []
            if isinstance(normalized_plan, dict):
                plan_item = normalized_plan.get(candidate_idx, {})
                if isinstance(plan_item, dict):
                    plan_label = str(plan_item.get("label", "") or "").strip()
                    if plan_label:
                        labels_to_scan.append(plan_label)
            source = source_by_idx.get(candidate_idx)
            if isinstance(source, dict):
                source_label = str(source.get("current_label", "") or "").strip()
                if source_label:
                    labels_to_scan.append(source_label)
            for label in labels_to_scan:
                for clause in _label_action_clauses(label):
                    held_object = _hold_clause_object(clause)
                    if not held_object:
                        continue
                    key = held_object.lower()
                    score = max(1, neighbors + 1 - offset)
                    scored[key] = scored.get(key, 0) + score
                    surface.setdefault(key, held_object)
    if not scored:
        return ""
    best_key = sorted(scored.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return surface.get(best_key, best_key)


def _enforce_hold_rule(
    label: str,
    *,
    segment_index: int,
    source_segments: List[Dict[str, Any]],
    normalized_plan: Optional[Dict[int, Dict[str, Any]]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    text = re.sub(r"\s+", " ", (label or "").strip(" ,"))
    if not text or text.lower() == "no action":
        return text
    clauses = [clause for clause in _label_action_clauses(text) if clause]
    if not clauses:
        return text

    hold_clauses = [clause for clause in clauses if _label_main_verb(clause) == "hold"]
    other_clauses = [clause for clause in clauses if _label_main_verb(clause) != "hold"]
    ordered_clauses = list(clauses)
    if hold_clauses and other_clauses:
        ordered_clauses = hold_clauses + other_clauses

    max_atomic_actions = max(1, int(_cfg_get(cfg or {}, "run.max_atomic_actions_per_label", 2)))
    current_text = ", ".join(ordered_clauses).strip(" ,")
    if (
        not hold_clauses
        and other_clauses
        and _count_atomic_actions_in_label(current_text) < max_atomic_actions
    ):
        held_object = _infer_held_object_context(
            segment_index,
            source_segments,
            normalized_plan=normalized_plan,
            cfg=cfg,
        )
        if held_object and held_object.lower() not in current_text.lower():
            candidate_clauses = [f"hold {held_object}"] + ordered_clauses
            candidate_text = ", ".join(candidate_clauses).strip(" ,")
            if _count_atomic_actions_in_label(candidate_text) <= max_atomic_actions:
                ordered_clauses = candidate_clauses

    return re.sub(r"\s+", " ", ", ".join(ordered_clauses)).strip(" ,")


def _label_goal_key(label: str) -> str:
    if _is_no_action_label(label):
        return ""
    clauses = _label_action_clauses(label)
    if not clauses:
        return ""
    verbs: List[str] = []
    for clause in clauses:
        verb = _label_main_verb(clause)
        if verb:
            verbs.append(verb)
    if not verbs:
        return ""
    non_micro = [verb for verb in verbs if verb not in _MICRO_ACTION_VERBS]
    return (non_micro[-1] if non_micro else verbs[-1]).strip().lower()


def _build_auto_continuity_merge_operations(
    segment_plan: Dict[int, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not bool(_cfg_get(cfg, "run.auto_continuity_merge_enabled", True)):
        return []
    if not bool(_cfg_get(cfg, "run.structural_allow_merge", True)):
        return []

    min_run = max(3, int(_cfg_get(cfg, "run.auto_continuity_merge_min_run_segments", 3)))
    min_overlap = max(0, int(_cfg_get(cfg, "run.auto_continuity_merge_min_token_overlap", 1)))
    ordered = sorted(int(key) for key in segment_plan.keys())
    if len(ordered) < min_run:
        return []

    max_merges = max(0, int(_cfg_get(cfg, "run.auto_continuity_merge_max_ops", 6)))

    def same_goal(idx_a: int, idx_b: int) -> bool:
        label_a = str(segment_plan.get(idx_a, {}).get("label", "")).strip()
        label_b = str(segment_plan.get(idx_b, {}).get("label", "")).strip()
        key_a = _label_goal_key(label_a)
        key_b = _label_goal_key(label_b)
        if not key_a or not key_b or key_a != key_b:
            return False
        # Exclude the shared goal verb from overlap so min_overlap is meaningful
        shared_tokens = _label_content_tokens(label_a).intersection(_label_content_tokens(label_b))
        shared_tokens.discard(key_a)
        return len(shared_tokens) >= min_overlap

    max_combined_duration_sec = max(
        0.0,
        float(_cfg_get(cfg, "run.auto_continuity_merge_max_combined_duration_sec", 0.0) or 0.0),
    )
    runs: List[Tuple[int, int]] = []
    run_start = ordered[0]
    run_end = ordered[0]
    for idx in ordered[1:]:
        start_sec = _safe_float(segment_plan.get(run_start, {}).get("start_sec"), 0.0)
        end_sec_candidate = _safe_float(segment_plan.get(idx, {}).get("end_sec"), start_sec)
        
        combined_duration = end_sec_candidate - start_sec
        duration_ok = (
            max_combined_duration_sec <= 0.0
            or combined_duration <= max_combined_duration_sec + 1e-6
        )
        if idx == run_end + 1 and same_goal(run_end, idx) and duration_ok:
            run_end = idx
            continue
        if (run_end - run_start + 1) >= min_run:
            runs.append((run_start, run_end))
        run_start = idx
        run_end = idx
    if (run_end - run_start + 1) >= min_run:
        runs.append((run_start, run_end))

    merge_indices: List[int] = []
    for start_idx, end_idx in runs:
        for idx in range(end_idx, start_idx, -1):
            merge_indices.append(idx)
    merge_indices = sorted(set(merge_indices), reverse=True)
    if max_merges and len(merge_indices) > max_merges:
        print(
            f"[policy] continuity merge capped: {len(merge_indices)} -> {max_merges} "
            f"(run.auto_continuity_merge_max_ops)"
        )
        merge_indices = merge_indices[:max_merges]
    return [{"action": "merge", "segment_index": int(idx)} for idx in merge_indices]


def _rewrite_no_action_pauses_in_plan(segment_plan: Dict[int, Dict[str, Any]], cfg: Dict[str, Any]) -> int:
    if not bool(_cfg_get(cfg, "run.no_action_pause_rewrite_enabled", True)):
        return 0
    max_pause_sec = max(0.0, float(_cfg_get(cfg, "run.no_action_pause_rewrite_max_sec", 12.0)))
    min_overlap = max(1, int(_cfg_get(cfg, "run.no_action_pause_rewrite_min_overlap_tokens", 1)))
    prefer_next_adjust = bool(_cfg_get(cfg, "run.no_action_pause_rewrite_prefer_next_adjust", True))

    ordered_indices = sorted(segment_plan.keys())
    rewrites = 0
    for pos, idx in enumerate(ordered_indices):
        item = segment_plan.get(idx, {})
        label = str(item.get("label", "")).strip()
        if not _is_no_action_label(label):
            continue
        start_sec = _safe_float(item.get("start_sec", 0.0), 0.0)
        end_sec = _safe_float(item.get("end_sec", start_sec), start_sec)
        if (end_sec - start_sec) > max_pause_sec:
            continue
        if pos == 0 or pos >= len(ordered_indices) - 1:
            continue

        prev_label = str(segment_plan.get(ordered_indices[pos - 1], {}).get("label", "")).strip()
        next_label = str(segment_plan.get(ordered_indices[pos + 1], {}).get("label", "")).strip()
        if not prev_label or not next_label:
            continue
        if _is_no_action_label(prev_label) or _is_no_action_label(next_label):
            continue

        overlap = len(_label_content_tokens(prev_label).intersection(_label_content_tokens(next_label)))
        if overlap < min_overlap:
            continue

        replacement = prev_label
        if prefer_next_adjust and _label_main_verb(next_label) == "adjust":
            replacement = next_label
        elif _label_main_verb(prev_label) == _label_main_verb(next_label):
            replacement = prev_label

        if replacement and replacement != label:
            item["label"] = replacement
            segment_plan[idx] = item
            rewrites += 1
    return rewrites


def _normalize_ing_verbs_to_imperative(text: str) -> str:
    out = text or ""
    for src, dst in _ING_TO_BASE_VERB_MAP.items():
        out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def _int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n < 20:
        return _NUM_WORDS_0_TO_19[n]
    if n < 100:
        tens, rem = divmod(n, 10)
        return _NUM_TENS_WORDS[tens] if rem == 0 else f"{_NUM_TENS_WORDS[tens]}-{_NUM_WORDS_0_TO_19[rem]}"
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        return (
            f"{_NUM_WORDS_0_TO_19[hundreds]} hundred"
            if rem == 0
            else f"{_NUM_WORDS_0_TO_19[hundreds]} hundred {_int_to_words(rem)}"
        )
    if n < 10000:
        thousands, rem = divmod(n, 1000)
        return (
            f"{_NUM_WORDS_0_TO_19[thousands]} thousand"
            if rem == 0
            else f"{_NUM_WORDS_0_TO_19[thousands]} thousand {_int_to_words(rem)}"
        )
    return str(n)


def _replace_numerals_with_words(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            value = int(token)
        except (TypeError, ValueError):
            return token
        return _int_to_words(value)

    out = re.sub(r"\b\d+\b", repl, text or "")
    return re.sub(r"\s+", " ", out).strip()


def _expand_verb_object_attachment_patterns(text: str) -> str:
    out = text or ""

    def _clean(token: str) -> str:
        return re.sub(r"\s+", " ", (token or "").strip(" ,"))

    def _repl(match: re.Match[str]) -> str:
        obj = _clean(match.group(1))
        prep = _clean(match.group(2)).lower()
        dest = _clean(match.group(3))
        if not obj or not prep or not dest:
            return match.group(0)
        return f"pick up {obj}, place {obj} {prep} {dest}"

    out = re.sub(
        r"\bpick up\s+and\s+place\s+([^,]+?)\s+(on|in|into|onto|at|to|inside|under|over)\s+([^,]+)",
        _repl,
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\bpick up\s+([^,]+?)\s+and\s+place\s+(on|in|into|onto|at|to|inside|under|over)\s+([^,]+)",
        _repl,
        out,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", out).strip(" ,")


def _normalize_mechanical_motion_to_goal(text: str) -> str:
    out = text or ""

    def _norm_obj(value: str) -> str:
        obj = re.sub(r"\s+", " ", (value or "").strip(" ,.;:"))
        obj = re.sub(r"^(?:on|onto|across|to|into|in)\s+", "", obj, flags=re.IGNORECASE)
        obj = re.sub(r"^(?:finish|fully)\s+cut(?:ting)?\s+", "", obj, flags=re.IGNORECASE)
        obj = re.sub(r"^cut(?:ting)?\s+", "", obj, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", obj).strip(" ,.;:")

    def _saw_repl(match: re.Match[str]) -> str:
        tool_raw = str(match.group("tool") or "").strip().lower()
        obj = _norm_obj(str(match.group("obj") or ""))
        if not obj:
            return match.group(0)
        tool = "handsaw" if "hand" in tool_raw else "saw"
        return f"cut {obj} with {tool}"

    out = re.sub(
        r"\bmove\s+(?P<tool>hand\s*saw|handsaw|saw)\s+back\s+and\s+forth\s+"
        r"(?:(?:to\s+)?(?:(?:finish|fully)\s+)?cut(?:ting)?\s+)?"
        r"(?:(?:on|onto|across|to|into|in)\s+)?(?P<obj>[^,]+)",
        _saw_repl,
        out,
        flags=re.IGNORECASE,
    )

    def _sand_repl(match: re.Match[str]) -> str:
        obj = _norm_obj(str(match.group("obj") or ""))
        return f"sand {obj} with sandpaper" if obj else match.group(0)

    out = re.sub(
        r"\b(?:move|rub)\s+sandpaper(?:\s+back\s+and\s+forth)?\s+"
        r"(?:(?:on|onto|across|to|into|in)\s+)?(?P<obj>[^,]+)",
        _sand_repl,
        out,
        flags=re.IGNORECASE,
    )

    def _norm_hair_obj(value: str) -> str:
        obj = _norm_obj(value)
        obj = re.sub(r"\bsection\s+hair\b", "wig", obj, flags=re.IGNORECASE)
        obj = re.sub(r"\bwig\s+hair\b", "wig", obj, flags=re.IGNORECASE)
        obj = re.sub(r"\bhair\b", "wig", obj, flags=re.IGNORECASE)
        obj = re.sub(r"\s+", " ", obj).strip(" ,.;:")
        return obj or "wig"

    def _comb_section_repl(match: re.Match[str]) -> str:
        return f"section {_norm_hair_obj(str(match.group('obj') or 'wig'))} with comb"

    out = re.sub(
        r"\bmove\s+comb(?:\s+tail)?\s+through\s+(?P<obj>[^,]+?)\s+to\s+section(?:\s+hair)?\b",
        _comb_section_repl,
        out,
        flags=re.IGNORECASE,
    )

    def _comb_detangle_repl(match: re.Match[str]) -> str:
        return f"detangle {_norm_hair_obj(str(match.group('obj') or 'wig'))} with comb"

    out = re.sub(
        r"\bmove\s+comb\s+through\s+(?P<obj>[^,]+?)\s+to\s+detangle\b",
        _comb_detangle_repl,
        out,
        flags=re.IGNORECASE,
    )

    def _comb_style_repl(match: re.Match[str]) -> str:
        return f"comb {_norm_hair_obj(str(match.group('obj') or 'wig'))}"

    out = re.sub(
        r"\bmove\s+comb\s+through\s+(?P<obj>[^,]+?)\s+to\s+style\b",
        _comb_style_repl,
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\bmove\s+comb\s+through\s+(?P<obj>[^,]+)\b",
        _comb_style_repl,
        out,
        flags=re.IGNORECASE,
    )

    def _straightener_repl(match: re.Match[str]) -> str:
        return f"straighten {_norm_hair_obj(str(match.group('obj') or 'wig'))} with hair straightener"

    out = re.sub(
        r"\bmove\s+hair\s+straightener\s+(?:to\s+)?(?:press|straighten)\s+(?P<obj>[^,]+)\b",
        _straightener_repl,
        out,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", out).strip(" ,")


_COMMON_LABEL_START_VERBS = (
    "pick up",
    "put down",
    "move",
    "place",
    "adjust",
    "align",
    "hold",
    "cut",
    "open",
    "close",
    "insert",
    "remove",
    "attach",
    "detach",
    "push",
    "pull",
    "slide",
    "lift",
    "lower",
    "set",
    "position",
    "stack",
    "unstack",
    "sort",
    "arrange",
    "load",
    "unload",
    "transfer",
    "carry",
)

_DANGLING_TRAILING_ACTION_WORDS: set[str] = {
    "apply",
    "applying",
    "brush",
    "brushing",
    "coat",
    "coating",
    "drive",
    "driving",
    "paint",
    "painting",
    "polish",
    "polishing",
    "sand",
    "sanding",
    "scrape",
    "scraping",
    "spread",
    "spreading",
}


def _rewrite_clause_missing_motion_verb(clause: str) -> str:
    text = re.sub(r"\s+", " ", (clause or "").strip(" ,.;:"))
    if not text:
        return text
    lower = text.lower()
    if lower == "no action":
        return "No Action"
    for verb in _COMMON_LABEL_START_VERBS:
        if lower.startswith(f"{verb} "):
            return text
    match = re.match(r"^(?P<obj>.+?)\s+to\s+(?P<dest>.+)$", text, flags=re.IGNORECASE)
    if not match:
        return text
    obj = re.sub(r"\s+", " ", str(match.group("obj") or "").strip(" ,.;:"))
    dest = re.sub(r"\s+", " ", str(match.group("dest") or "").strip(" ,.;:"))
    if not obj or not dest:
        return text
    return f"move {obj} to {dest}"


def _rewrite_embedded_object_action_clause(clause: str) -> str:
    text = re.sub(r"\s+", " ", (clause or "").strip(" ,.;:"))
    if not text or text.lower() == "no action":
        return text

    def _clean(value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip(" ,.;:"))

    match = re.match(
        r"^(?P<object>.+?)\s+with\s+apply\s+(?P<material>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        obj = _clean(str(match.group("object") or ""))
        material = _clean(str(match.group("material") or ""))
        if obj and material:
            if re.search(r"\b(?:on|onto|into|in|inside|to|under|over|across)\b", material, flags=re.IGNORECASE):
                return f"apply {material}"
            return f"apply {material} on {obj}"

    tail_match = re.match(
        r"^(?P<head>.+?)\s+\band\b\s+(?P<tail>[a-z]+)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if tail_match:
        head = _clean(str(tail_match.group("head") or ""))
        tail = str(tail_match.group("tail") or "").strip().lower()
        if head and tail in _DANGLING_TRAILING_ACTION_WORDS:
            return head

    return text


def _collapse_adjacent_duplicate_tokens(text: str) -> str:
    out = re.sub(r"\s+", " ", (text or "").strip())
    if not out:
        return out
    repeated_phrase = re.compile(r"\b([a-z]+(?:\s+[a-z]+){1,2})\s+\1\b", re.IGNORECASE)
    repeated_word = re.compile(r"\b([a-z]+)\s+\1\b", re.IGNORECASE)
    for _ in range(6):
        prev = out
        out = repeated_phrase.sub(r"\1", out)
        out = repeated_word.sub(r"\1", out)
        out = re.sub(r"\s+", " ", out).strip(" ,")
        if out == prev:
            break
    return out


def _rewrite_label_tier3(label: str) -> str:
    text = re.sub(r"\s+", " ", (label or "").strip())
    if not text:
        return text
    if text.lower() == "no action":
        return "No Action"

    text = _normalize_mechanical_motion_to_goal(text)
    text = _normalize_gripper_terms(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bthen\b", ",", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnext\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcontinue\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bagain\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\banother\b\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\brotate(?:d|s|ing)?\b", "adjust", text, flags=re.IGNORECASE)
    text = re.sub(r"\bturn(?:ed|s|ing)?\b", "adjust", text, flags=re.IGNORECASE)
    text = re.sub(r"\brelocate(?:d|s|ing)?\b", "move", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgrab(?:bed|s|bing)?\b", "pick up", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\bwith\s+(?:left|right)?\s*(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\busing\s+(?:left|right)?\s*(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:left|right)\s+(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms|leg|legs|foot|feet|toe|toes)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = _normalize_ing_verbs_to_imperative(text)
    text = _collapse_adjacent_duplicate_tokens(text)
    text = _replace_numerals_with_words(text)
    text = _expand_verb_object_attachment_patterns(text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    clauses = [
        _rewrite_embedded_object_action_clause(clause.strip())
        for clause in text.split(",")
        if clause.strip()
    ]
    if not clauses:
        return text
    deduped: List[str] = []
    seen: set[str] = set()
    for clause in clauses:
        clause = _rewrite_clause_missing_motion_verb(clause)
        key = re.sub(r"\s+", " ", clause).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clause)
    return ", ".join(deduped).strip()


def _normalize_label_min_safety(label: str) -> str:
    text = re.sub(r"\s+", " ", (label or "").strip())
    if not text:
        return text
    if text.lower() == "no action":
        return "No Action"
    text = re.sub(r"\brotate(?:d|s|ing)?\b", "adjust", text, flags=re.IGNORECASE)
    text = re.sub(r"\bturn(?:ed|s|ing)?\b", "adjust", text, flags=re.IGNORECASE)
    text = re.sub(r"\brelocate(?:d|s|ing)?\b", "move", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgrab(?:bed|s|bing)?\b", "pick up", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\bwith\s+(?:left|right)?\s*(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\busing\s+(?:left|right)?\s*(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:left|right)\s+(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:hand|hands|finger|fingers|thumb|thumbs|palm|palms|wrist|wrists|arm|arms|leg|legs|foot|feet|toe|toes)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = _normalize_ing_verbs_to_imperative(text)
    text = _normalize_mechanical_motion_to_goal(text)
    text = _collapse_adjacent_duplicate_tokens(text)
    text = _normalize_gripper_terms(text)
    text = _replace_numerals_with_words(text)
    text = _expand_verb_object_attachment_patterns(text)
    clauses = [
        _rewrite_embedded_object_action_clause(clause.strip())
        for clause in text.split(",")
        if clause.strip()
    ]
    return re.sub(r"\s+", " ", ", ".join(clauses) if clauses else text).strip(" ,")


def _normalize_segment_plan(
    payload: Dict[str, Any],
    source_segments: List[Dict[str, Any]],
    cfg: Dict[str, Any] | None = None,
) -> Dict[int, Dict[str, Any]]:
    items = payload.get("segments")
    if not isinstance(items, list):
        raise ValueError("Gemini payload must contain list at 'segments'")

    effective_cfg = cfg or {}
    forbidden_verbs_raw = _cfg_get(effective_cfg, "run.forbidden_label_verbs", [])
    forbidden_verbs = [str(v).strip().lower() for v in forbidden_verbs_raw if str(v).strip()]
    allowed_verb_token_patterns = _allowed_label_start_verb_token_patterns_from_cfg(effective_cfg)

    source_by_idx: Dict[int, Dict[str, Any]] = {int(seg["segment_index"]): seg for seg in source_segments}
    out: Dict[int, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        idx_raw = item.get("segment_index", item.get("index"))
        try:
            idx = int(idx_raw)
        except (TypeError, ValueError):
            continue
        source = source_by_idx.get(idx)
        if source is None:
            continue
        source_label = str(source.get("current_label", "")).strip()
        label = str(item.get("label", "")).strip() or source_label
        if bool(_cfg_get(effective_cfg, "run.tier3_label_rewrite", True)):
            label = _rewrite_label_tier3(label)
        label = _autofix_label_candidate(
            effective_cfg,
            label,
            source_label,
            forbidden_verbs,
            allowed_verb_token_patterns,
        )
        label = _normalize_label_min_safety(label)
        label = _enforce_hold_rule(
            label,
            segment_index=idx,
            source_segments=source_segments,
            normalized_plan=out,
            cfg=effective_cfg,
        )
        start_src = _safe_float(source.get("start_sec", 0.0), 0.0)
        end_src = _safe_float(source.get("end_sec", 0.0), 0.0)
        start_sec = _safe_float(item.get("start_sec", start_src), start_src)
        end_sec = _safe_float(item.get("end_sec", end_src), end_src)
        if end_sec <= start_sec:
            start_sec = start_src
            end_sec = end_src
        max_drift = 12.0
        if abs(start_sec - start_src) > max_drift:
            start_sec = start_src
        if abs(end_sec - end_src) > max_drift:
            end_sec = end_src
        out[idx] = {
            "segment_index": idx,
            "label": label,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
        }

    for idx, source in source_by_idx.items():
        if idx in out:
            continue
        source_label = str(source.get("current_label", "")).strip()
        if bool(_cfg_get(effective_cfg, "run.tier3_label_rewrite", True)):
            source_label = _rewrite_label_tier3(source_label)
        source_label = _autofix_label_candidate(
            effective_cfg,
            source_label,
            source_label,
            forbidden_verbs,
            allowed_verb_token_patterns,
        )
        source_label = _normalize_label_min_safety(source_label)
        source_label = _enforce_hold_rule(
            source_label,
            segment_index=idx,
            source_segments=source_segments,
            normalized_plan=out,
            cfg=effective_cfg,
        )
        out[idx] = {
            "segment_index": idx,
            "label": source_label,
            "start_sec": round(_safe_float(source.get("start_sec", 0.0), 0.0), 3),
            "end_sec": round(_safe_float(source.get("end_sec", 0.0), 0.0), 3),
        }

    # Re-run hold-rule enforcement against the fully built plan so forward
    # continuity cues from later segments can repair earlier labels.
    full_plan_snapshot = {
        idx: dict(item)
        for idx, item in out.items()
        if isinstance(item, dict)
    }
    for idx in sorted(full_plan_snapshot):
        current = out.get(idx)
        if not isinstance(current, dict):
            continue
        current_label = str(current.get("label", "")).strip()
        repaired_label = _enforce_hold_rule(
            current_label,
            segment_index=idx,
            source_segments=source_segments,
            normalized_plan=full_plan_snapshot,
            cfg=effective_cfg,
        )
        out[idx]["label"] = repaired_label

    if not out:
        raise ValueError("Gemini returned no usable segment plan")
    return out


def _normalize_label_map_from_plan(segment_plan: Dict[int, Dict[str, Any]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for idx, item in segment_plan.items():
        label = str(item.get("label", "")).strip()
        if label:
            out[idx] = label
    if not out:
        raise ValueError("Segment plan has no usable labels")
    return out


__all__ = [
    "_DISALLOWED_TOOL_TERMS",
    "_count_atomic_actions_in_label",
    "_normalize_gripper_terms",
    "_label_main_verb",
    "_is_no_action_label",
    "_label_content_tokens",
    "_allowed_label_start_verb_token_patterns_from_cfg",
    "_label_starts_with_allowed_action_verb",
    "_contains_forbidden_verb_in_label",
    "_strip_forbidden_verbs_for_autofix",
    "_action_phrase_missing_object_for_autofix",
    "_heuristic_autofix_verb_from_text",
    "_autofix_label_candidate",
    "_label_action_clauses",
    "_hold_clause_object",
    "_infer_held_object_context",
    "_enforce_hold_rule",
    "_label_goal_key",
    "_build_auto_continuity_merge_operations",
    "_rewrite_no_action_pauses_in_plan",
    "_normalize_ing_verbs_to_imperative",
    "_int_to_words",
    "_replace_numerals_with_words",
    "_expand_verb_object_attachment_patterns",
    "_normalize_mechanical_motion_to_goal",
    "_collapse_adjacent_duplicate_tokens",
    "_rewrite_label_tier3",
    "_normalize_label_min_safety",
    "_normalize_segment_plan",
    "_normalize_label_map_from_plan",
]
