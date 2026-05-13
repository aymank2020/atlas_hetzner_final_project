from __future__ import annotations

import argparse
import json

from atlas_triplet_compare import generate_gemini_chat_timed_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemini Chat timed-label generation in an isolated subprocess.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-txt", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--prompt-scope", default="timed_labels")
    parser.add_argument("--auth-mode-override", default="chat_web")
    parser.add_argument("--tier2-draft-path", default="")
    args = parser.parse_args()

    result = generate_gemini_chat_timed_labels(
        config_path=args.config,
        video_path=args.video_path,
        cache_dir=args.cache_dir,
        out_txt=args.out_txt,
        out_json=args.out_json,
        episode_id=args.episode_id,
        auth_mode_override=args.auth_mode_override,
        prompt_scope=args.prompt_scope,
        tier2_draft_path=args.tier2_draft_path,
        model=args.model,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
