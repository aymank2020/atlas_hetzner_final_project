"""
Build repair payloads from annotation JSON + validator report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import validator


def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(payload: Dict[str, Any], path: str) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_payload_from_annotation(
    annotation: Dict[str, Any],
    evidence_notes: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    normalized = validator.normalize_annotation(annotation)
    report = validator.validate_episode(normalized)
    normalized_annotation = report.get("normalized_annotation", normalized)
    return validator.build_repair_payload(normalized_annotation, report, evidence_notes=evidence_notes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build repair payload for Atlas annotation repair prompt")
    parser.add_argument("--annotation-json", required=True, help="Path to candidate annotation JSON")
    parser.add_argument("--output-json", required=True, help="Path to save repair payload JSON")
    parser.add_argument("--evidence-json", default="", help="Optional evidence notes JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation = load_json(args.annotation_json)
    evidence = load_json(args.evidence_json) if args.evidence_json else None
    payload = build_payload_from_annotation(annotation, evidence_notes=evidence)
    save_json(payload, args.output_json)
    print(f"Saved repair payload: {args.output_json}")


if __name__ == "__main__":
    main()
