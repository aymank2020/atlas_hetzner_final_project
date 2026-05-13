"""Label autofix layer to correct common Gemini policy violations before validation."""

import re
from typing import Any, Dict, List, Tuple
from src.infra.solver_config import _cfg_get
from src.rules.labels import (
    _label_action_clauses,
    _allowed_label_start_verb_token_patterns_from_cfg,
    _contains_forbidden_verb_in_label,
    _label_starts_with_allowed_action_verb,
    _normalize_label_for_compare,
)

# Common mappings for forbidden/bad verbs to allowed ones
_VERB_REPLACEMENTS = {
    "reach": "adjust",
    "inspect": "adjust",
    "check": "adjust",
    "examine": "adjust",
    "look": "adjust",
    "grab": "pick up",
    "relocate": "move",
    "start picking up": "pick up",
    "start placing": "place",
    "start moving": "move",
    "continue to": "",
    "continue": "",
    "proceed to": "",
    "finish": "",
}

# Ensure place has a location
_PLACE_LOCATION_RE = re.compile(r"\bplace\b.*\b(on|in|into|onto|at|to|inside|under|over)\b", re.IGNORECASE)

def _fix_forbidden_verbs(text: str) -> str:
    """Replace forbidden verbs with allowed equivalents."""
    out = text or ""
    for bad_verb, good_verb in _VERB_REPLACEMENTS.items():
        pattern = rf"\b{re.escape(bad_verb)}\b"
        if good_verb == "":
            # Remove verb and cleanup spaces
            out = re.sub(pattern, "", out, flags=re.IGNORECASE)
        else:
            out = re.sub(pattern, good_verb, out, flags=re.IGNORECASE)
    
    # Cleanup extra spaces
    return re.sub(r"\s+", " ", out).strip()

def _fix_missing_place_location(text: str) -> str:
    """If 'place' is used but no location preposition is found, append 'on surface'."""
    out = text or ""
    if "place" in out.lower() and not _PLACE_LOCATION_RE.search(out):
        # We append ' on surface' to the clause that contains 'place'
        clauses = []
        for clause in _label_action_clauses(out):
            if "place" in clause.lower() and not _PLACE_LOCATION_RE.search(clause):
                clauses.append(f"{clause.strip()} on surface")
            else:
                clauses.append(clause.strip())
        out = ", ".join(clauses)
    return out

def _fix_too_many_atomic_actions(text: str, max_actions: int = 2) -> str:
    """If label has more than max_actions clauses, truncate it."""
    out = text or ""
    clauses = _label_action_clauses(out)
    if len(clauses) > max_actions:
        # Keep only the first max_actions clauses
        out = ", ".join(clauses[:max_actions])
    return out

def autofix_segment_label(
    label: str, 
    cfg: Dict[str, Any], 
    allowed_patterns: List[Tuple[str, ...]],
    forbidden_verbs: List[str]
) -> str:
    """Apply all autofix rules to a single label."""
    if not label or label.lower() == "no action":
        return label

    fixed = label

    # Rule 1: Fix forbidden verbs
    fixed = _fix_forbidden_verbs(fixed)

    # Rule 2: Ensure 'place' has a location
    fixed = _fix_missing_place_location(fixed)

    # Rule 3: Enforce max atomic actions
    max_actions = max(1, int(_cfg_get(cfg, "run.max_atomic_actions_per_label", 2)))
    fixed = _fix_too_many_atomic_actions(fixed, max_actions)

    # Rule 4: Clean up any double spaces or bad punctuation left behind
    fixed = re.sub(r"\s+", " ", fixed).strip(" ,.;:")
    
    # Rule 5: Fallback if verb is still invalid after fix
    if not _label_starts_with_allowed_action_verb(fixed, allowed_patterns):
        # Extremely coarse fallback if everything failed
        words = fixed.split()
        if words:
            fixed = f"adjust {' '.join(words)}"
        else:
            fixed = "adjust item"

    return fixed

def apply_autofix_to_plan(
    segment_plan: Dict[int, Dict[str, Any]], 
    cfg: Dict[str, Any]
) -> Dict[int, Dict[str, Any]]:
    """Mutate the segment plan by applying autofix to all labels."""
    if not bool(_cfg_get(cfg, "run.label_autofix_enabled", True)):
        return segment_plan

    forbidden_verbs_raw = _cfg_get(cfg, "run.forbidden_label_verbs", [])
    forbidden_verbs = [str(item).strip().lower() for item in forbidden_verbs_raw if str(item).strip()]
    allowed_patterns = _allowed_label_start_verb_token_patterns_from_cfg(cfg)

    fixes_applied = 0
    for idx, item in segment_plan.items():
        original_label = str(item.get("label", "")).strip()
        fixed_label = autofix_segment_label(original_label, cfg, allowed_patterns, forbidden_verbs)
        
        if _normalize_label_for_compare(original_label) != _normalize_label_for_compare(fixed_label):
            item["label"] = fixed_label
            fixes_applied += 1
            
    if fixes_applied > 0:
        import logging
        _logger = logging.getLogger(__name__)
        _logger.info(f"[autofix] Applied corrections to {fixes_applied} segments.")

    return segment_plan
