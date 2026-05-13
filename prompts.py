"""
Prompt and schema assets for Atlas-style annotation pipelines.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from src.policy import context_manager as policy_context


POLICY_VERSION = "atlas-discord-cra-v1"
POLICY_PROMPT_SUMMARY = ""
FORBIDDEN_VERBS = ["inspect", "check", "reach", "examine"]
FORBIDDEN_NARRATIVE_WORDS = ["then", "another", "continue", "next", "again"]


VIDEO_ANNOTATION_PROMPT = """You are an expert egocentric hand-action annotation assistant for Atlas Capture style labeling.

Analyze the provided video and return segment annotations as strict JSON.

Core objective:
- one segment = one continuous hand-object interaction toward one goal
- start when hands engage toward contact
- end when hands disengage or goal changes

Strict rules:
- label only task-relevant hand-object interactions
- do not label walking/navigation/idle gestures/inspection-only behavior
- treat gripper as an extension of hand
- usually do not mention the tool; if unavoidable, use only "gripper"
- never use tool terms like "mechanical arm", "robotic arm", "robot arm", "manipulator", "claw arm"
- imperative voice only
- forbidden verbs: inspect, check, reach
- reach exception: allow "reach" only when the segment ends at the clipped/truncated end of the video
- forbidden narrative words: then, another, continue, next, again
- no "-ing" verb starts; use imperative commands (e.g., "turn mold", not "turning mold")
- no numerals in labels
- never mix dense/coarse in one segment
- max 2 atomic actions per segment label (usually one comma or one "and")
- No Action only when hands touch nothing or ego is idle/irrelevant
- No Action must be standalone, never mixed with action
- place should include location
- attach every verb to an object (no "pick up and place box"; use "pick up box, place box ...")
- if lift then place both happen in the same segment, include both actions
- if draft phrasing violates Tier-3 rules, rewrite the label from scratch (do not patch bad shorthand)
- no shortcuts: do not merge distinct physical interactions into one invalid phrase
- if two adjacent segments are one continuous coarse action (no disengagement), merge them into one segment
- if 3+ consecutive segments are the same ongoing action and tool/object is never dropped, merge them aggressively
- action preservation: do not erase a clear primary action from Tier-2 draft (e.g., cut/screw/wipe) unless visual evidence proves Tier-2 is wrong
- no back-to-back duplicates: never output two consecutive segments with exactly identical non-"No Action" labels
- treat short tool reloading actions (dip/reload brush, wet sponge, etc.) as micro-actions of the same main goal
- avoid mechanical-motion phrasing like "move saw back and forth"; use coarse goal verbs like "cut wood with saw"
- never use mechanical tool-motion labels like "move comb through wig" or "move hair straightener to press hair";
  use direct task verbs like "detangle wig with comb", "section wig with comb", "straighten wig with hair straightener"
- no token stuttering/repetition like "detangle detangle" or "pull loosened pull loosened"
- if ego is still holding tool/object, "No Action" is forbidden; merge/relabel short pauses into surrounding action
- timestamp strictness: label only what happens inside each exact segment start/end window; do not shift actions
- avoid body-part wording (hands/fingers/body parts) unless unavoidable
- if uncertain, use general nouns (tool/container/cloth/item)
- Object Identity Continuity: if Tier-2 draft consistently identifies a device class (e.g., "phone"),
  do not rename it to a different class (e.g., "laptop") unless visual evidence is unequivocal;
  if uncertain, use generic "device" instead of guessing a different class
- do not keep placeholder/default labels
- do not hallucinate hidden actions
- for retrieval from containers, prefer "remove [item] from [container]" over "take [item] out"
- segment duration strictness: MAXIMUM 60 seconds per segment (No exceptions)
- preferred density: 2-5 seconds per segment (the "Sweet Spot")
- for continuous actions lasting > 60s: split every 60s and repeat the same label
- handling empty draft rows: if draft row is blank/no-label and hands are disengaged, use "No Action" (do not invent an action)

Output requirements:
- JSON only, no markdown
- use the provided schema shape
- include `step_by_step_reasoning` first: max 2 short sentences of chronological analysis before final segments
- include start/end in seconds, label, granularity, confidence, rule checks, audit risk
"""


REPAIR_PROMPT = """You are a strict annotation repair assistant for Atlas-style egocentric hand-action labels.

Input includes:
1) existing annotation JSON
2) validator_report
3) optional evidence_notes

Critical behavior:
- preserve valid fields and segments
- apply minimal edits needed to satisfy rules
- do not invent actions or objects
- do not guess hidden actions
- if uncertain, prefer general objects and coarse abstraction
- if repair requires guessing, set escalation_flag=true

Rules to enforce:
- imperative labels only
- forbidden verbs: inspect, check, reach
- reach exception: allow "reach" only for clipped/truncated video-end segments
- forbidden narrative words: then, another, continue, next, again
- treat gripper as hand extension; avoid tool mention unless unavoidable
- if tool must be named, use only "gripper" (never mechanical/robotic arm wording)
- no numerals in labels
- max 2 atomic actions per segment label (usually one comma or one "and")
- No Action must not be mixed with action
- each segment has one granularity: dense/coarse/no_action
- place should include a location
- verbs must be attached to objects (no "pick up and place box"; use "pick up box, place box ...")
- avoid intent-only wording (prepare to, try to, about to)
- preserve object/device identity continuity unless visual evidence clearly proves draft is wrong

Timestamp handling:
- do not change timestamps unless validator reports boundary/timestamp/coverage issues
- if timestamps change, update duration_sec

Merge/split handling:
- do not merge/split unless validator explicitly requires merge_split or boundary correction
- if split is required, keep order and regenerate segment_index sequence

Return repaired annotation JSON only.
No markdown. No extra text.
"""


AUDIT_JUDGE_PROMPT = """You are an Atlas-style annotation audit judge.

Evaluate the provided annotation JSON against strict egocentric hand-action rules.

Audit each segment for:
1) one continuous interaction toward one goal
2) task-relevant hand-object focus
3) accurate verb/object naming (no guessing)
4) label format: imperative, no numerals, max 2 atomic actions, verbs attached to objects
5) no forbidden verbs: inspect/check/reach (except truncated-end reach)
6) no forbidden narrative words: then/another/continue/next/again
7) dense/coarse not mixed
8) No Action usage and isolation correctness
9) timestamp alignment to engagement/disengagement boundaries
10) merge/split logic consistency
11) no hallucinated or missed major actions

Decision policy:
- FAIL if major fail conditions exist:
  missed major action, hallucination, invalid timestamps,
  forbidden verbs, forbidden narrative words, dense/coarse mix,
  >2 atomic actions, place missing location, No Action mixed with action
- BORDERLINE if no major fail but multiple medium risks
- PASS only when accurate, defensible, and consistent

Return JSON only:
{
  "overall_verdict": "PASS|FAIL|BORDERLINE",
  "overall_score": 0-100,
  "segment_results": [],
  "audit_fail_conditions_triggered": [],
  "high_priority_fixes": [],
  "low_priority_fixes": [],
  "final_notes": ""
}
"""


NORMALIZATION_PROMPT = """You are a strict label normalization assistant.

Input: already-segmented annotation JSON.
Task: normalize wording only, with minimal edits.

Rules:
- imperative labels
- no numerals
- no forbidden verbs: inspect/check/reach (except truncated-end reach)
- no forbidden narrative words: then/another/continue/next/again
- normalize disallowed tool terms to "gripper" when unavoidable
- no intent-only language
- consistent object naming within episode
- verbs clearly attached to objects
- preserve No Action exactly when granularity=no_action
- keep timestamps unchanged unless explicit rule violation requires it
- preserve meaning and segment order

Output JSON only. Minimal edits only.
"""


CONSISTENCY_PROMPT = """Review annotation JSON for consistency only.

Do NOT re-segment and do NOT invent actions.
Normalize only:
- object naming consistency
- verb consistency where meaning is unchanged
- separator consistency (comma / and)
- pluralization if clearly visible and needed
- remove unnecessary adjectives unless needed to disambiguate

Preserve timestamps, segment count, and action meaning.
Return JSON only.
"""


_VIDEO_ANNOTATION_TEMPLATE = VIDEO_ANNOTATION_PROMPT
_REPAIR_TEMPLATE = REPAIR_PROMPT
_AUDIT_JUDGE_TEMPLATE = AUDIT_JUDGE_PROMPT
_NORMALIZATION_TEMPLATE = NORMALIZATION_PROMPT
_CONSISTENCY_TEMPLATE = CONSISTENCY_PROMPT


def _rewrite_policy_text(template: str, *, max_segment_seconds: float, preferred_density: Dict[str, Any], max_atomic_actions: int) -> str:
    text = template
    text = re.sub(
        r"forbidden verbs: [^\n]+",
        f"forbidden verbs: {', '.join(FORBIDDEN_VERBS)}",
        text,
    )
    text = re.sub(
        r"forbidden narrative words: [^\n]+",
        f"forbidden narrative words: {', '.join(FORBIDDEN_NARRATIVE_WORDS)}",
        text,
    )
    text = text.replace("forbidden verbs: inspect/check/reach (except truncated-end reach)", f"forbidden verbs: {'/'.join(FORBIDDEN_VERBS)} (except truncated-end reach)")
    text = text.replace("no forbidden narrative words: then/another/continue/next/again", f"no forbidden narrative words: {'/'.join(FORBIDDEN_NARRATIVE_WORDS)}")
    text = text.replace("no forbidden verbs: inspect/check/reach (except truncated-end reach)", f"no forbidden verbs: {'/'.join(FORBIDDEN_VERBS)} (except truncated-end reach)")
    text = text.replace(
        "max 2 atomic actions per segment label (usually one comma or one \"and\")",
        f"max {max_atomic_actions} atomic actions per segment label (usually one comma or one \"and\")",
    )
    text = text.replace(
        "segment duration strictness: MAXIMUM 60 seconds per segment (No exceptions)",
        f"segment duration strictness: MAXIMUM {max_segment_seconds:.0f} seconds per segment (No exceptions)",
    )
    text = text.replace(
        "preferred density: 2-5 seconds per segment (the \"Sweet Spot\")",
        f"preferred density: {preferred_density['min']:g}-{preferred_density['max']:g} seconds per segment (the \"Sweet Spot\")",
    )
    text = text.replace(
        "for continuous actions lasting > 60s: split every 60s and repeat the same label",
        f"for continuous actions lasting > {max_segment_seconds:.0f}s: split every {max_segment_seconds:.0f}s and repeat the same label",
    )
    return text


def refresh_policy_assets() -> None:
    global POLICY_VERSION
    global POLICY_PROMPT_SUMMARY
    global FORBIDDEN_VERBS
    global FORBIDDEN_NARRATIVE_WORDS
    global VIDEO_ANNOTATION_PROMPT
    global REPAIR_PROMPT
    global AUDIT_JUDGE_PROMPT
    global NORMALIZATION_PROMPT
    global CONSISTENCY_PROMPT

    policy = policy_context.load_current_policy()
    POLICY_VERSION = str(policy.get("policy_version", POLICY_VERSION))
    FORBIDDEN_VERBS = list(
        policy_context.get_policy("lexicon.forbidden_verbs", FORBIDDEN_VERBS, policy=policy)
    )
    FORBIDDEN_NARRATIVE_WORDS = list(
        policy_context.get_policy(
            "lexicon.forbidden_narrative_words",
            FORBIDDEN_NARRATIVE_WORDS,
            policy=policy,
        )
    )
    max_segment_seconds = float(
        policy_context.get_policy("engine_limits.max_segment_seconds", 10.0, policy=policy)
    )
    preferred_density = policy_context.get_policy(
        "annotation.preferred_density_sec",
        {"min": 2.0, "max": 5.0},
        policy=policy,
    )
    max_atomic_actions = int(
        policy_context.get_policy("annotation.max_atomic_actions", 2, policy=policy)
    )
    POLICY_PROMPT_SUMMARY = policy_context.build_policy_prompt_summary(policy=policy)
    VIDEO_ANNOTATION_PROMPT = _rewrite_policy_text(
        _VIDEO_ANNOTATION_TEMPLATE,
        max_segment_seconds=max_segment_seconds,
        preferred_density=preferred_density,
        max_atomic_actions=max_atomic_actions,
    ) + f"\nCanonical policy snapshot:\n{POLICY_PROMPT_SUMMARY}\n"
    REPAIR_PROMPT = _rewrite_policy_text(
        _REPAIR_TEMPLATE,
        max_segment_seconds=max_segment_seconds,
        preferred_density=preferred_density,
        max_atomic_actions=max_atomic_actions,
    ) + f"\nCanonical policy snapshot:\n{POLICY_PROMPT_SUMMARY}\n"
    AUDIT_JUDGE_PROMPT = _rewrite_policy_text(
        _AUDIT_JUDGE_TEMPLATE,
        max_segment_seconds=max_segment_seconds,
        preferred_density=preferred_density,
        max_atomic_actions=max_atomic_actions,
    ) + f"\nCanonical policy snapshot:\n{POLICY_PROMPT_SUMMARY}\n"
    NORMALIZATION_PROMPT = _rewrite_policy_text(
        _NORMALIZATION_TEMPLATE,
        max_segment_seconds=max_segment_seconds,
        preferred_density=preferred_density,
        max_atomic_actions=max_atomic_actions,
    ) + f"\nCanonical policy snapshot:\n{POLICY_PROMPT_SUMMARY}\n"
    CONSISTENCY_PROMPT = _rewrite_policy_text(
        _CONSISTENCY_TEMPLATE,
        max_segment_seconds=max_segment_seconds,
        preferred_density=preferred_density,
        max_atomic_actions=max_atomic_actions,
    ) + f"\nCanonical policy snapshot:\n{POLICY_PROMPT_SUMMARY}\n"


def get_video_annotation_prompt() -> str:
    return VIDEO_ANNOTATION_PROMPT


def get_repair_prompt() -> str:
    return REPAIR_PROMPT


def get_audit_judge_prompt() -> str:
    return AUDIT_JUDGE_PROMPT


refresh_policy_assets()


ANNOTATION_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "EgocentricHandActionAnnotationEpisode",
    "type": "object",
    "required": [
        "step_by_step_reasoning",
        "episode_id",
        "video_duration_sec",
        "annotation_version",
        "segments",
        "episode_checks",
    ],
    "properties": {
        "step_by_step_reasoning": {
            "type": "string",
            "minLength": 1,
            "maxLength": 320,
            "description": (
                "Chronological reasoning summary in at most 2 short sentences: identify key "
                "hand-object interactions, goal transitions, and why split/merge choices were made."
            ),
        },
        "episode_id": {"type": "string", "minLength": 1},
        "video_duration_sec": {"type": "number", "minimum": 0},
        "annotation_version": {"type": "string"},
        "source_context": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "task_title": {"type": "string"},
                "task_summary": {"type": "string"},
                "fps_sampled": {"type": "number", "minimum": 0},
                "model_name": {"type": "string"},
            },
        },
        "segments": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/segment"}},
        "episode_checks": {"$ref": "#/$defs/episodeChecks"},
    },
    "$defs": {
        "segment": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "segment_index",
                "start_sec",
                "end_sec",
                "duration_sec",
                "label",
                "granularity",
                "primary_goal",
                "primary_object",
                "confidence",
                "rule_checks",
                "audit_risk",
            ],
            "properties": {
                "segment_index": {"type": "integer", "minimum": 1},
                "start_sec": {"type": "number", "minimum": 0},
                "end_sec": {"type": "number", "minimum": 0},
                "duration_sec": {"type": "number", "exclusiveMinimum": 0},
                "label": {"type": "string", "minLength": 2},
                "granularity": {"type": "string", "enum": ["dense", "coarse", "no_action"]},
                "primary_goal": {"type": "string", "minLength": 1},
                "primary_object": {"type": "string", "minLength": 1},
                "secondary_objects": {"type": "array", "items": {"type": "string"}, "default": []},
                "actions_observed": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["verb", "object"],
                        "properties": {
                            "verb": {"type": "string", "minLength": 1},
                            "object": {"type": "string", "minLength": 1},
                            "location": {"type": "string"},
                        },
                    },
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "uncertainty_note": {"type": "string"},
                "escalation_flag": {"type": "boolean", "default": False},
                "escalation_reason": {
                    "type": "string",
                    "enum": [
                        "object_unidentifiable",
                        "action_unclear_without_guessing",
                        "cannot_be_accurate_even_with_coarse",
                        "other",
                    ],
                },
                "rule_checks": {"$ref": "#/$defs/ruleChecks"},
                "audit_risk": {"$ref": "#/$defs/auditRisk"},
            },
            "allOf": [
                {
                    "if": {"properties": {"granularity": {"const": "no_action"}}, "required": ["granularity"]},
                    "then": {
                        "properties": {
                            "label": {"enum": ["No Action"]},
                            "primary_goal": {"enum": ["idle", "irrelevant", "no_contact"]},
                        }
                    },
                }
            ],
        },
        "ruleChecks": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "imperative_voice",
                "min_two_words",
                "no_numerals",
                "no_forbidden_verbs",
                "verbs_attached_to_objects",
                "one_goal",
                "full_action_coverage",
                "no_hallucinated_steps",
                "dense_coarse_not_mixed",
                "no_action_not_mixed_with_action",
                "timestamps_aligned",
            ],
            "properties": {
                "imperative_voice": {"type": "boolean"},
                "min_two_words": {"type": "boolean"},
                "no_numerals": {"type": "boolean"},
                "no_forbidden_verbs": {"type": "boolean"},
                "forbidden_verbs_found": {"type": "array", "items": {"type": "string"}},
                "verbs_attached_to_objects": {"type": "boolean"},
                "one_goal": {"type": "boolean"},
                "full_action_coverage": {"type": "boolean"},
                "no_hallucinated_steps": {"type": "boolean"},
                "dense_coarse_not_mixed": {"type": "boolean"},
                "no_action_not_mixed_with_action": {"type": "boolean"},
                "timestamps_aligned": {"type": "boolean"},
                "hands_disengage_boundary_ok": {"type": "boolean"},
            },
        },
        "auditRisk": {
            "type": "object",
            "additionalProperties": False,
            "required": ["level", "reasons"],
            "properties": {
                "level": {"type": "string", "enum": ["low", "medium", "high"]},
                "reasons": {"type": "array", "items": {"type": "string"}},
            },
        },
        "episodeChecks": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "segments_sorted",
                "no_negative_durations",
                "no_overlaps",
                "coverage_within_video_duration",
                "repeated_action_logic_checked",
                "merge_split_logic_checked",
            ],
            "properties": {
                "segments_sorted": {"type": "boolean"},
                "no_negative_durations": {"type": "boolean"},
                "no_overlaps": {"type": "boolean"},
                "coverage_within_video_duration": {"type": "boolean"},
                "gaps_present": {"type": "boolean"},
                "repeated_action_logic_checked": {"type": "boolean"},
                "merge_split_logic_checked": {"type": "boolean"},
                "notes": {"type": "string"},
            },
        },
    },
}


def schema_json(indent: int = 2) -> str:
    return json.dumps(ANNOTATION_SCHEMA, indent=indent, ensure_ascii=False)
