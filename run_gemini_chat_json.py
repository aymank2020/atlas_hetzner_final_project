from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

import atlas_triplet_compare as triplet


def _resolve_system_instruction_text(gem_cfg: dict, cfg_dir: Path, scope: str) -> str:
    if scope == "chat_ops":
        alias_text_keys = []
        alias_file_keys = []
    else:
        alias_text_keys = [
            "timed_labels_system_instruction_text",
            "chat_timed_system_instruction_text",
        ]
        alias_file_keys = [
            "timed_labels_system_instruction_file",
            "chat_timed_system_instruction_file",
        ]
    return triplet._resolve_system_instruction_text(
        gem_cfg,
        cfg_dir=cfg_dir,
        scope=scope,
        alias_text_keys=alias_text_keys,
        alias_file_keys=alias_file_keys,
    )


def _labels_schema() -> dict:
    return {
        "type": "OBJECT",
        "properties": {
            "segments": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "segment_index": {"type": "INTEGER"},
                        "start_sec": {"type": "NUMBER"},
                        "end_sec": {"type": "NUMBER"},
                        "label": {"type": "STRING"},
                    },
                    "required": ["segment_index", "start_sec", "end_sec", "label"],
                },
            }
        },
        "required": ["segments"],
    }


def _ops_schema(*, allow_merge: bool) -> dict:
    allowed_actions = ["split", "merge"] if allow_merge else ["split"]
    return {
        "type": "OBJECT",
        "properties": {
            "operations": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "action": {"type": "STRING", "enum": allowed_actions},
                        "segment_index": {"type": "INTEGER"},
                    },
                    "required": ["action", "segment_index"],
                },
            }
        },
        "required": ["operations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a chat-web Gemini JSON request in an isolated subprocess.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--prompt-scope", default="chat_labels")
    parser.add_argument("--mode", choices=["labels", "ops"], required=True)
    parser.add_argument("--allow-merge", action="store_true")
    parser.add_argument("--response-schema-enabled", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise RuntimeError("Config root must be a YAML object.")
    cfg_dir = cfg_path.parent

    runtime_cfg = dict(cfg)
    gem_cfg = dict(runtime_cfg.get("gemini", {}) if isinstance(runtime_cfg.get("gemini"), dict) else {})
    gem_cfg["auth_mode"] = "chat_web"
    gem_cfg["model"] = str(args.model or "").strip()

    # Propagate timeout from run config into gemini config so that
    # atlas_triplet_compare uses the correct timeout for this mode
    run_cfg = runtime_cfg.get("run", {}) if isinstance(runtime_cfg.get("run"), dict) else {}
    if args.mode == "labels":
        labels_timeout = run_cfg.get("chat_labels_timeout_sec")
        if labels_timeout is not None:
            try:
                gem_cfg["chat_web_timeout_sec"] = max(60.0, float(labels_timeout))
            except (TypeError, ValueError):
                pass
    elif args.mode == "ops":
        ops_timeout = run_cfg.get("chat_ops_timeout_sec")
        if ops_timeout is not None:
            try:
                gem_cfg["chat_web_timeout_sec"] = max(60.0, float(ops_timeout))
            except (TypeError, ValueError):
                pass

    system_instruction = _resolve_system_instruction_text(gem_cfg, cfg_dir, str(args.prompt_scope or "").strip())
    if system_instruction:
        gem_cfg["system_instruction_text"] = system_instruction

    # ── Propagate correct timeout to triplet compare ──────────────
    run_cfg = runtime_cfg.get("run", {}) if isinstance(runtime_cfg.get("run"), dict) else {}
    if args.mode == "labels":
        propagated_timeout = run_cfg.get(
            "chat_labels_timeout_sec",
            gem_cfg.get("chat_labels_timeout_sec",
                        gem_cfg.get("chat_web_timeout_sec", 360)),
        )
    else:
        propagated_timeout = run_cfg.get(
            "chat_ops_timeout_sec",
            gem_cfg.get("chat_ops_timeout_sec",
                        gem_cfg.get("chat_web_timeout_sec", 300)),
        )
    gem_cfg["chat_web_timeout_sec"] = float(propagated_timeout or 360)

    runtime_cfg["gemini"] = gem_cfg

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    dotenv = triplet._load_dotenv(Path(".env"))
    response_schema = None
    if args.response_schema_enabled:
        response_schema = _ops_schema(allow_merge=bool(args.allow_merge)) if args.mode == "ops" else _labels_schema()

    result = triplet._call_gemini_compare(
        cfg=runtime_cfg,
        dotenv=dotenv,
        model=str(args.model or "").strip(),
        prompt=prompt,
        video_a=Path(args.video_path).resolve(),
        video_b=None,
        cache_dir=Path(args.cache_dir).resolve() / "video_inline",
        episode_id=str(args.episode_id or "").strip(),
        response_schema=response_schema,
        usage_mode=f"chat_solve:{str(args.prompt_scope or '').strip()}",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
