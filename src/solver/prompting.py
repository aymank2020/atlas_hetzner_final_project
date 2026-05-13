"""Prompt-building and chunk-consistency helpers extracted from the legacy solver."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.infra.solver_config import _cfg_get
from src.policy import context_manager as policy_context

_CHUNK_CONSISTENCY_VERB_PREFIXES: Tuple[str, ...] = (
    "pick up",
    "place",
    "open",
    "close",
    "pull open",
    "push",
    "adjust",
    "move",
    "drag",
    "tighten",
    "loosen",
    "remove",
    "insert",
    "fold",
    "spread out",
    "sand",
    "twist",
    "pour",
    "scoop",
    "hold",
    "position",
    "align",
    "pry open",
    "drive",
    "set",
    "put",
)
_CHUNK_CONSISTENCY_EQUIVALENCE_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("table", "surface"),
)
_CHUNK_CONSISTENCY_PREPOSITION_RE = re.compile(
    r"\b(from|in|into|on|onto|under|inside|at|to|with|over|near|across|through)\b",
    flags=re.IGNORECASE,
)
_CHUNK_CONSISTENCY_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", flags=re.IGNORECASE)
_CHUNK_CONSISTENCY_STOPWORDS: set[str] = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "with",
    "from",
    "in",
    "into",
    "on",
    "onto",
    "under",
    "inside",
    "at",
    "over",
    "near",
    "across",
    "through",
}


def build_prompt(
    segments: List[Dict[str, Any]],
    extra_instructions: str,
    allow_operations: bool = True,
    policy_trigger: str = "base",
) -> str:
    header = (
        "You are an Atlas Standard Tier-3 labeling assistant.\n"
        "You may receive the full task video as attached media plus employee segment text.\n"
        "Use the video as source of truth; employee labels may be wrong.\n"
        "Treat employee labels as drafts: preserve them when they are already correct, and correct them only where the video requires it.\n"
        "Rewrite from scratch only when the draft is wrong, stale, incomplete, or structurally invalid.\n"
        "For each segment index, output corrected label and timestamps when needed.\n"
        "Apply one-mental-model policy: one continuous interaction toward one goal per segment.\n"
        "Gripper rule: treat gripper as an extension of hand.\n"
        "Usually do NOT mention the tool in labels; if unavoidable, use only 'gripper'.\n"
        "Never use tool terms like mechanical arm / robotic arm / robot arm / manipulator / claw arm.\n"
        "Shorter segments are MANDATORY: target 2-5 second segments whenever the video supports it. Segments MUST NEVER exceed 10.0 seconds.\n"
        "MANDATORY SPLIT RULE: If any action or pause exceeds 10 seconds, you MUST split it into multiple smaller sub-segments of 2-5 seconds each.\n"
        "Do not merge for convenience; merge only when segments are truly redundant and the merged result STRICTLY follows the 10-second standard.\n"
        "Coarse-goal verbs: avoid mechanical muscle-motion phrasing (e.g., 'move saw back and forth'). "
        "Use task-goal verbs (e.g., 'cut wood with saw', 'sand board with sandpaper').\n"
        "No token stuttering: never repeat words/phrases like 'detangle detangle' or 'pull loosened pull loosened'.\n"
        "No '-ing' verbs: use imperative commands only (e.g., 'turn mold', not 'turning the mold').\n"
        "Timestamp strictness: describe only what happens inside each exact segment start_sec/end_sec; "
        "do not shift actions into neighboring segments.\n"
        "Prefer dense, granular labels when the actions are visible; dense is better than overly coarse rewrites.\n"
        "Within each kept segment, label all task-relevant visible actions and pauses.\n"
        "Dense labels may include multiple atomic actions separated by commas/and.\n"
        "Do not exceed 20 words or 2 atomic actions per label (typically one separator: a single comma or one 'and').\n"
        "Do not write narrative filler words like then/another/next/continue/again.\n"
        "For small corrective reorientation/reposition, prefer verb 'adjust'.\n"
        "Avoid forbidden verbs: rotate, inspect, check, look, examine, reach, grab, relocate.\n"
        "Use conservative object names that are directly visible.\n"
        "If object identity is unclear after careful inspection, use safe general nouns (tool/container/item).\n"
        "Do not guess hidden object identities and do not keep placeholder/default labels.\n"
        "If surface type/elevation is unclear (floor mat vs table/shelf), do not guess raised furniture.\n"
        "Use neutral location wording (on surface/on mat/on floor) unless elevation is clearly visible.\n"
        "Use 'place' only with explicit location (on/in/into/onto/at/to/inside/under/over).\n"
        "No-Action pause rule: if ego still holds the task object/tool during a pause, do not use 'No Action'. "
        "Keep/merge it with surrounding action.\n"
        "Attach verbs to objects: do not write 'pick up and place box' or 'pick up box and place under desk'; "
        "write 'pick up box, place box under desk'.\n"
        "If the segment clearly includes lift then placement, include both actions (pick up ..., place ...).\n"
        "Hold First Rule: if a label contains both a 'hold' action and another action, ALWAYS list 'hold' first.\n"
        "Template: hold [object], [action] [object] (e.g., 'hold shoe sole, press shoe sole').\n"
        "Correction-first workflow: improve the existing draft before inventing a totally new phrasing.\n"
        "No shortcuts: do not merge distinct physical interactions into one invalid phrase to save words.\n"
        "Avoid body-part wording (hands/fingers/body parts) unless unavoidable.\n"
        "Examples:\n"
        "BAD: paint chair -> dip paintbrush -> paint chair in separate short consecutive segments\n"
        "GOOD: merge micro-actions into one segment label 'paint chair with paintbrush' when tool is never dropped.\n"
        "BAD: move comb through wig to detangle\n"
        "GOOD: detangle wig with comb\n"
        "BAD: move hair straightener to press wig section\n"
        "GOOD: straighten wig section with hair straightener\n"
        "BAD: press shoe sole, hold shoe sole\n"
        "GOOD: hold shoe sole, press shoe sole\n"
        "If a segment timestamp is wrong, correct start_sec/end_sec.\n"
        "Label rules: imperative style, concise, minimum 2 words, maximum 20 words.\n"
        "STRICT VALIDATION RULES (FAILURE TO FOLLOW = SUBMISSION BLOCKED):\n"
        "1. NO DIGITS: NEVER use numerals (1, 2, 3). ALWAYS write numbers as words (one, two, three).\n"
        "2. FORBIDDEN VERBS: NEVER use 'reach', 'inspect', or 'check'. Use 'adjust', 'move', or 'pick up' instead.\n"
        "3. NO ONSET TERMS: NEVER use the word 'start' (e.g., 'start picking up' is WRONG, use 'pick up').\n"
        'Use "No Action" only as standalone label.\n'
        "If boundaries are fundamentally wrong, you may request split/merge operations before final labels.\n"
        "Allowed operations: edit, split, merge. Do NOT use delete.\n"
        "Operation segment_index refers to the row index at execution time.\n"
        "Operations must be ordered exactly as they should be executed.\n"
        "Return strict JSON object only:\n"
        "Response must start with '{' and end with '}'.\n"
        "Do not wrap JSON in markdown code fences.\n"
        '{"operations":[{"action":"split","segment_index":3}],'
        '"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":1.2,"label":"..."}]}\n'
        'If no structural change is needed, return "operations":[]\n'
        "Keep segment count and indices unchanged unless a justified split/merge is needed; timestamps must stay monotonic.\n"
    )
    if not allow_operations:
        header += "Structural operations are disabled for this pass.\nReturn operations as an empty list.\n"
    header += (
        "CRITICAL: You MUST provide a label for EVERY segment_index listed below. "
        "Do NOT skip any segment. Do NOT request delete operations. "
        "If segments are repetitive, use a coarse single-goal label for each one "
        "(e.g., 'roll dough, place dough in tray') but still output every segment.\n"
    )

    policy_summary = policy_context.build_policy_prompt_summary()
    retrieved_policy_context = policy_context.retrieve_runtime_rules(
        segments,
        trigger=policy_trigger,
        budget={"rules": 6, "examples": 1},
    )
    lines = ["Canonical policy:", policy_summary, "", "Segments input:"]
    for seg in segments:
        lines.append(
            f"- segment_index={seg['segment_index']} start_sec={seg['start_sec']} "
            f"end_sec={seg['end_sec']} current_label={json.dumps(seg.get('current_label', ''), ensure_ascii=False)} "
            f"raw_text={json.dumps(seg.get('raw_text', ''), ensure_ascii=False)}"
        )
    extra_blocks = [block for block in [retrieved_policy_context, extra_instructions.strip()] if block]
    merged_extra_instructions = "\n".join(extra_blocks).strip()
    if merged_extra_instructions:
        lines.extend(["", "Extra instructions (Filtered for current context):"])
        seg_words = " ".join(
            f"{seg.get('current_label', '')} {seg.get('raw_text', '')}".lower() for seg in segments
        )
        seg_tokens = set(re.findall(r"[a-z]{3,}", seg_words))
        diff_rules = []
        for rule in merged_extra_instructions.split("\n"):
            rule_str = rule.strip()
            if not rule_str:
                continue
            rule_tokens = set(re.findall(r"[a-z]{4,}", rule_str.lower()))
            if not rule_tokens or any(token in seg_tokens for token in rule_tokens) or len(rule_str) < 40:
                diff_rules.append(rule_str)
        lines.append("\n".join(diff_rules) if diff_rules else merged_extra_instructions)
    return header + "\n".join(lines)


def _read_optional_text_file(path_text: str) -> str:
    path_raw = (path_text or "").strip()
    if not path_raw:
        return ""
    try:
        path = Path(path_raw)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _resolve_system_instruction(cfg: Dict[str, Any]) -> str:
    file_text = _read_optional_text_file(str(_cfg_get(cfg, "gemini.system_instruction_file", "")))
    inline_text = str(_cfg_get(cfg, "gemini.system_instruction_text", "")).strip()
    policy_text = policy_context.build_policy_prompt_summary().strip()
    chunks = [text for text in [file_text, inline_text, policy_text] if text]
    return "\n\n".join(chunks).strip()


def _consistency_norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip(" ,.;:")).lower()


def _consistency_tokens(text: str) -> List[str]:
    return [token.lower() for token in _CHUNK_CONSISTENCY_TOKEN_RE.findall(text or "")]


def _extract_consistency_terms_from_label(label: str, max_terms: int = 8) -> List[str]:
    if not label:
        return []
    terms: List[str] = []
    clauses = [clause.strip().lower() for clause in re.split(r",", label) if clause and clause.strip()]
    for clause in clauses:
        if clause == "no action":
            continue
        rest = clause
        for prefix in _CHUNK_CONSISTENCY_VERB_PREFIXES:
            token = prefix + " "
            if rest.startswith(token):
                rest = rest[len(token):].strip()
                break
        if not rest:
            continue
        match = _CHUNK_CONSISTENCY_PREPOSITION_RE.search(rest)
        candidates = [rest]
        if match:
            candidates = [rest[:match.start()].strip(), rest[match.end():].strip()]
        for candidate in candidates:
            norm = _consistency_norm(re.sub(r"^(the|a|an)\s+", "", candidate))
            if not norm or norm in _CHUNK_CONSISTENCY_STOPWORDS:
                continue
            if len(_consistency_tokens(norm)) == 0:
                continue
            if norm not in terms:
                terms.append(norm)
                if len(terms) >= max_terms:
                    return terms
    return terms


def _find_equivalent_canonical_term(norm_term: str, canonical_terms: List[str]) -> str:
    if not norm_term:
        return ""
    for existing in canonical_terms:
        if _consistency_norm(existing) == norm_term:
            return existing
    for group in _CHUNK_CONSISTENCY_EQUIVALENCE_GROUPS:
        group_set = {_consistency_norm(item) for item in group}
        if norm_term in group_set:
            for existing in canonical_terms:
                if _consistency_norm(existing) in group_set:
                    return existing
    term_tokens = _consistency_tokens(norm_term)
    if not term_tokens:
        return ""
    term_head = term_tokens[-1]
    term_set = set(term_tokens)
    for existing in canonical_terms:
        existing_norm = _consistency_norm(existing)
        existing_tokens = _consistency_tokens(existing_norm)
        if not existing_tokens or existing_tokens[-1] != term_head:
            continue
        existing_set = set(existing_tokens)
        overlap = term_set.intersection(existing_set)
        if term_set.issubset(existing_set) or existing_set.issubset(term_set):
            return existing
        if len(overlap) >= max(1, min(len(term_set), len(existing_set)) - 1):
            return existing
    return ""


def _apply_consistency_aliases_to_label(label: str, alias_to_canonical: Dict[str, str]) -> str:
    out = label or ""
    if not out or not alias_to_canonical:
        return out
    replacements = sorted(alias_to_canonical.items(), key=lambda item: len(item[0]), reverse=True)
    for alias_norm, canonical in replacements:
        src = _consistency_norm(alias_norm)
        dst = _consistency_norm(canonical)
        if not src or not dst or src == dst:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(src) + r"(?![a-z0-9])"
        out = re.sub(pattern, dst, out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def _update_chunk_consistency_memory(
    label: str,
    canonical_terms: List[str],
    alias_to_canonical: Dict[str, str],
    memory_limit: int,
) -> str:
    rewritten = _apply_consistency_aliases_to_label(label, alias_to_canonical)
    for term in _extract_consistency_terms_from_label(rewritten):
        term_norm = _consistency_norm(term)
        if not term_norm or term_norm in alias_to_canonical:
            continue
        canonical = _find_equivalent_canonical_term(term_norm, canonical_terms)
        if canonical:
            alias_to_canonical[term_norm] = canonical
            rewritten = _apply_consistency_aliases_to_label(rewritten, alias_to_canonical)
            continue
        alias_to_canonical[term_norm] = term
        canonical_terms.append(term)
    if memory_limit > 0 and len(canonical_terms) > memory_limit:
        canonical_terms[:] = canonical_terms[-memory_limit:]
        allowed = {_consistency_norm(term) for term in canonical_terms}
        for alias_key in list(alias_to_canonical.keys()):
            alias_norm = _consistency_norm(alias_key)
            canonical_norm = _consistency_norm(alias_to_canonical.get(alias_key, ""))
            if alias_norm in allowed or canonical_norm in allowed:
                continue
            alias_to_canonical.pop(alias_key, None)
    return rewritten


def _build_chunk_consistency_prompt_hint(canonical_terms: List[str], max_terms: int) -> str:
    if not canonical_terms or max_terms <= 0:
        return ""
    selected = canonical_terms[-max_terms:]
    return (
        "PREFERRED OBJECT/LOCATION TERMS from previous chunks (must keep naming stable for same object): "
        + " | ".join(selected)
    )


def _request_labels_with_optional_segment_chunking(
    cfg: Dict[str, Any],
    segments: List[Dict[str, Any]],
    prompt: str,
    video_file: Optional[Path],
    allow_operations: bool,
    model_override: str = "",
    task_id: str = "",
    task_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from src.solver import legacy_impl as _legacy

    return _legacy._request_labels_with_optional_segment_chunking(
        cfg,
        segments,
        prompt,
        video_file,
        allow_operations,
        model_override=model_override,
        task_id=task_id,
        task_state=task_state,
    )


__all__ = [
    "build_prompt",
    "_read_optional_text_file",
    "_resolve_system_instruction",
    "_consistency_norm",
    "_consistency_tokens",
    "_extract_consistency_terms_from_label",
    "_find_equivalent_canonical_term",
    "_apply_consistency_aliases_to_label",
    "_update_chunk_consistency_memory",
    "_build_chunk_consistency_prompt_hint",
    "_request_labels_with_optional_segment_chunking",
]
