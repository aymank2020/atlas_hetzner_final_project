#!/usr/bin/env python3
"""Atlas Update Checker Daemon

Checks Discord channels and the Atlas Training Hub for guideline/policy
updates every 6 hours (4 times per day).

Sources monitored:
  1. Discord announcement channels (via Bot API)
  2. Atlas Training Hub /training/hub (via authenticated HTTP or Playwright CDP)

When changes are detected:
  - Sends Telegram alerts with a summary
  - Extracts policy-relevant rules via Gemini API
  - Syncs updates into data/gemini_policy_discord_live.txt

Usage:
    python3 deploy/update_checker.py                 # single check
    python3 deploy/update_checker.py --daemon         # 6-hour loop
    python3 deploy/update_checker.py --interval 3600  # custom interval

Environment variables (from .env):
    DISCORD_BOT_TOKEN       - Discord bot token
    DISCORD_GUILD_ID        - Atlas Discord server ID
    TELEGRAM_BOT_TOKEN      - Telegram alerts
    TELEGRAM_ALLOWED_CHATS  - Telegram chat IDs
    GEMINI_API_KEY          - For policy extraction (optional)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_DIR = Path(os.environ.get(
    "ATLAS_APP_DIR",
    str(Path(__file__).resolve().parent.parent),
))

STATE_FILE = APP_DIR / "outputs" / "update_checker_state.json"
LOG_FILE = APP_DIR / "outputs" / "update_checker.log"
POLICY_OVERRIDE_FILE = APP_DIR / "data" / "gemini_policy_discord_live.txt"
MAIN_POLICY_FILE = APP_DIR / "data" / "gemini_policy_v1.txt"
RUNS_DIR = APP_DIR / "outputs" / "update_checker_runs"

CHECK_INTERVAL_SEC = 6 * 3600  # 6 hours = 4 times per day

# Discord channels to monitor (bot's own server: "Dana Timmer's server")
# The bot only has access to its own server, not the main Atlas Discord.
# Forward important Atlas announcements to #atlas-rules for the bot to pick up.
DISCORD_CHANNELS: Dict[str, str] = {}  # populated in _load_env()

# Fallback: also read Discord export files from this directory
DISCORD_EXPORTS_DIR = APP_DIR / "outputs" / "discord_exports"

# How far back to look for messages (hours)
DISCORD_LOOKBACK_HOURS = 12

# Training hub pages
TRAINING_HUB_URL = "https://audit.atlascapture.io/training/hub"
TRAINING_HUB_TABS = ["guidelines", "references", "training-video", "quizzes"]

# Managed block markers for policy file sync
MANAGED_BLOCK_START = "# BEGIN_DISCORD_LIVE_UPDATES"
MANAGED_BLOCK_END = "# END_DISCORD_LIVE_UPDATES"

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_IDS: list[int] = []

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_running = True


def _signal_handler(sig, frame):
    global _running
    _running = False


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


def _send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        return False
    sent = 0
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            body = json.dumps({
                "chat_id": chat_id,
                "text": text[:4000],
                "disable_web_page_preview": True,
            }).encode("utf-8")
            req = Request(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=20) as resp:
                if resp.status == 200:
                    sent += 1
        except Exception as e:
            _log(f"[telegram] send failed to {chat_id}: {e}")
    return sent > 0


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def _load_state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        _log(f"[state] save failed: {e}")


# ---------------------------------------------------------------------------
# Discord API
# ---------------------------------------------------------------------------


def _discord_api_get(endpoint: str, bot_token: str) -> Any:
    """Make a GET request to the Discord API."""
    url = f"https://discord.com/api/v10{endpoint}"
    req = Request(url, headers={
        "Authorization": f"Bot {bot_token}",
        "User-Agent": "AtlasUpdateChecker/1.0",
    })
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        _log(f"[discord] API error for {endpoint}: {e}")
        return None


def _fetch_discord_messages(
    channel_id: str,
    bot_token: str,
    after_timestamp: Optional[datetime] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Fetch recent messages from a Discord channel."""
    params = f"?limit={limit}"

    # Discord snowflake: convert timestamp to snowflake ID for 'after' param
    if after_timestamp:
        # Discord epoch: 2015-01-01T00:00:00Z
        discord_epoch = 1420070400000
        ts_ms = int(after_timestamp.timestamp() * 1000)
        snowflake = (ts_ms - discord_epoch) << 22
        if snowflake > 0:
            params += f"&after={snowflake}"

    data = _discord_api_get(f"/channels/{channel_id}/messages{params}", bot_token)
    if not isinstance(data, list):
        return []
    return data


def _check_discord(bot_token: str, state: dict) -> Tuple[List[Dict], List[str]]:
    """Check all Discord channels for new messages. Returns (new_messages, changes)."""
    if not bot_token:
        _log("[discord] no bot token configured, skipping")
        return [], []

    lookback = datetime.now(timezone.utc) - timedelta(hours=DISCORD_LOOKBACK_HOURS)
    prev_hashes = state.get("discord_message_hashes", {})
    new_hashes: Dict[str, str] = {}
    all_messages: List[Dict] = []
    changes: List[str] = []

    for channel_name, channel_id in DISCORD_CHANNELS.items():
        if not channel_id or channel_id == "0":
            continue
        _log(f"[discord] checking #{channel_name} ({channel_id})")
        messages = _fetch_discord_messages(channel_id, bot_token, after_timestamp=lookback)
        if not messages:
            continue

        # Filter to relevant messages (from admins/mods or with labeling keywords)
        for msg in messages:
            content = msg.get("content", "")
            author = msg.get("author", {})
            author_name = author.get("username", "") or author.get("global_name", "")
            msg_id = msg.get("id", "")
            timestamp = msg.get("timestamp", "")

            if not content.strip():
                continue

            # Build a hash for change detection
            content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:12]
            new_hashes[msg_id] = content_hash

            # Check if this is a new message we haven't seen
            if msg_id not in prev_hashes:
                all_messages.append({
                    "id": msg_id,
                    "channel": channel_name,
                    "channel_id": channel_id,
                    "author": author_name,
                    "content": content,
                    "timestamp": timestamp,
                    "attachments": [
                        a.get("url", "") for a in msg.get("attachments", [])
                    ],
                })

        # Rate limit courtesy
        time.sleep(1)

    # Detect changes
    new_msg_count = len(all_messages)
    if new_msg_count > 0:
        channels_with_new = set(m["channel"] for m in all_messages)
        changes.append(
            f"Discord: {new_msg_count} new message(s) in {', '.join(sorted(channels_with_new))}"
        )

    # Update state
    state["discord_message_hashes"] = {**prev_hashes, **new_hashes}
    # Trim old hashes (keep last 5000)
    if len(state["discord_message_hashes"]) > 5000:
        items = list(state["discord_message_hashes"].items())
        state["discord_message_hashes"] = dict(items[-5000:])

    # Also check Discord export files (for main Atlas channels the bot can't access)
    export_messages, export_changes = _check_discord_exports(state)
    all_messages.extend(export_messages)
    changes.extend(export_changes)

    return all_messages, changes


def _check_discord_exports(state: dict) -> Tuple[List[Dict], List[str]]:
    """Read Discord export JSON/TXT files for channels the bot can't access."""
    if not DISCORD_EXPORTS_DIR.exists():
        return [], []

    prev_file_hashes = state.get("discord_export_file_hashes", {})
    new_file_hashes: Dict[str, str] = {}
    prev_seen_ids = set(state.get("discord_export_seen_ids", []))
    new_seen_ids: List[str] = list(state.get("discord_export_seen_ids", []))
    all_messages: List[Dict] = []
    changes: List[str] = []

    # Find all JSON/TXT export files
    files: List[Path] = []
    for pattern in ["*.json", "*.txt"]:
        files.extend(DISCORD_EXPORTS_DIR.rglob(pattern))

    for fpath in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
        file_key = str(fpath.resolve())
        file_hash = hashlib.md5(
            f"{fpath.stat().st_size}:{fpath.stat().st_mtime}".encode()
        ).hexdigest()[:12]
        new_file_hashes[file_key] = file_hash

        # Skip unchanged files
        if prev_file_hashes.get(file_key) == file_hash:
            continue

        _log(f"[discord-export] reading changed file: {fpath.name}")
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Parse JSON exports (DiscordChatExporter format)
        if fpath.suffix.lower() == ".json":
            try:
                data = json.loads(text)
                raw_msgs = []
                if isinstance(data, dict) and isinstance(data.get("messages"), list):
                    raw_msgs = data["messages"]
                elif isinstance(data, list):
                    raw_msgs = data
                for msg in raw_msgs:
                    if not isinstance(msg, dict):
                        continue
                    msg_id = str(msg.get("id", ""))
                    if not msg_id or msg_id in prev_seen_ids:
                        continue
                    content = msg.get("content", "")
                    if not content:
                        continue
                    author = msg.get("author", {})
                    author_name = author.get("name", "") if isinstance(author, dict) else str(author)
                    all_messages.append({
                        "id": msg_id,
                        "channel": fpath.stem,
                        "channel_id": "",
                        "author": author_name,
                        "content": content,
                        "timestamp": msg.get("timestamp", ""),
                        "attachments": [],
                        "source": f"export:{fpath.name}",
                    })
                    prev_seen_ids.add(msg_id)
                    new_seen_ids.append(msg_id)
            except json.JSONDecodeError:
                pass

        # Parse TXT exports
        elif fpath.suffix.lower() == ".txt":
            pattern = re.compile(
                r"^\s*(?:\[(?P<ts>[^\]]+)\]\s*)?(?P<author>[^:]{1,80}):\s*(?P<content>.+)",
            )
            for idx, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                m = pattern.match(line)
                if not m:
                    continue
                msg_id = f"txt_{fpath.stem}_{idx}"
                if msg_id in prev_seen_ids:
                    continue
                all_messages.append({
                    "id": msg_id,
                    "channel": fpath.stem,
                    "channel_id": "",
                    "author": m.group("author").strip(),
                    "content": m.group("content").strip(),
                    "timestamp": m.group("ts") or "",
                    "attachments": [],
                    "source": f"export:{fpath.name}",
                })
                prev_seen_ids.add(msg_id)
                new_seen_ids.append(msg_id)

    if all_messages:
        changes.append(f"Discord exports: {len(all_messages)} new message(s) from export files")

    state["discord_export_file_hashes"] = new_file_hashes
    # Trim seen IDs
    if len(new_seen_ids) > 10000:
        new_seen_ids = new_seen_ids[-10000:]
    state["discord_export_seen_ids"] = new_seen_ids

    return all_messages, changes


# ---------------------------------------------------------------------------
# Training Hub scraper (authenticated HTTP via storage state cookies)
# ---------------------------------------------------------------------------


def _load_atlas_cookies() -> Dict[str, str]:
    """Load Atlas auth cookies from the Playwright storage state file."""
    candidates = [
        APP_DIR / ".state" / "gemini_chat_storage_state_danatimer.json",
        APP_DIR / ".state" / "atlas_storage_state.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                cookies = {}
                for c in data.get("cookies", []):
                    domain = c.get("domain", "")
                    if "atlascapture" in domain:
                        cookies[c["name"]] = c["value"]
                if cookies:
                    _log(f"[hub] loaded {len(cookies)} Atlas cookies from {path.name}")
                    return cookies
            except Exception as e:
                _log(f"[hub] failed to load cookies from {path}: {e}")
    return {}


def _fetch_hub_page(url: str, cookies: Dict[str, str]) -> str:
    """Fetch a training hub page with authentication cookies."""
    cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
    req = Request(url, headers={
        "Cookie": cookie_header,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        _log(f"[hub] fetch failed for {url}: {e}")
        return ""


def _extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML, stripping tags."""
    if not html:
        return ""
    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Replace block elements with newlines
    text = re.sub(r"<(?:br|p|div|h[1-6]|li|tr)[^>]*>", "\n", text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _check_training_hub(state: dict) -> Tuple[str, List[str]]:
    """Check the Atlas training hub for content changes.
    Returns (content_text, list_of_changes)."""
    cookies = _load_atlas_cookies()
    if not cookies:
        _log("[hub] no Atlas auth cookies found, skipping training hub check")
        return "", []

    prev_hashes = state.get("hub_content_hashes", {})
    new_hashes: Dict[str, str] = {}
    changes: List[str] = []
    all_content: List[str] = []

    # Fetch the main hub page
    _log("[hub] fetching training hub...")
    html = _fetch_hub_page(TRAINING_HUB_URL, cookies)
    if not html or len(html) < 200:
        _log("[hub] training hub returned empty/short response (auth may have expired)")
        return "", []

    text = _extract_text_from_html(html)
    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
    new_hashes["main"] = content_hash
    all_content.append(f"=== Training Hub Main ===\n{text}")

    if prev_hashes.get("main") != content_hash and prev_hashes.get("main"):
        changes.append("Training Hub: main page content changed")

    # Try to fetch tab-specific URLs if the hub uses them
    # The hub appears to be a SPA with tabs, but let's try common URL patterns
    for tab in TRAINING_HUB_TABS:
        for url_pattern in [
            f"{TRAINING_HUB_URL}?tab={tab}",
            f"{TRAINING_HUB_URL}/{tab}",
        ]:
            html = _fetch_hub_page(url_pattern, cookies)
            if html and len(html) > 200:
                text = _extract_text_from_html(html)
                tab_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
                new_hashes[tab] = tab_hash
                all_content.append(f"=== Training Hub: {tab} ===\n{text}")

                if prev_hashes.get(tab) != tab_hash and prev_hashes.get(tab):
                    changes.append(f"Training Hub: '{tab}' tab content changed")
                break  # Found working URL pattern for this tab
        time.sleep(1)

    state["hub_content_hashes"] = new_hashes
    combined_text = "\n\n".join(all_content)

    # Save raw content for debugging
    try:
        hub_dir = RUNS_DIR / "hub_snapshots"
        hub_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        (hub_dir / f"hub_content_{ts}.txt").write_text(
            combined_text[:500000], encoding="utf-8"
        )
    except Exception:
        pass

    return combined_text, changes


# ---------------------------------------------------------------------------
# Policy extraction and sync
# ---------------------------------------------------------------------------


def _is_policy_relevant(message: Dict) -> bool:
    """Check if a Discord message is likely to contain policy updates."""
    content = (message.get("content", "") or "").lower()
    keywords = [
        "update", "new rule", "guideline", "forbidden", "must", "do not",
        "limit", "gripper", "coarse", "dense", "atomic", "hallucination",
        "label", "segment", "annotation", "hold", "no action", "intent",
        "merge", "split", "duration", "timestamp", "effective immediately",
        "clarification", "example", "correct", "incorrect", "acceptable",
    ]
    hits = sum(1 for kw in keywords if kw in content)
    has_attachment = bool(message.get("attachments"))

    # Admin/mod authors are always relevant
    admin_names = ["frans", "duwop", "admin", "mod"]
    author = (message.get("author", "") or "").lower()
    is_admin = any(a in author for a in admin_names)

    return hits >= 2 or (is_admin and hits >= 1) or (has_attachment and hits >= 1)


def _extract_rules_native(messages: List[Dict]) -> List[str]:
    """Extract explicit rules using regex patterns (no API needed)."""
    rules: List[str] = []
    patterns = [
        re.compile(r"(?i)^(?:golden rule|new rule|update|rule):\s*(.+)", re.MULTILINE),
        re.compile(r"(?i)^(never\s+.{10,})", re.MULTILINE),
        re.compile(r"(?i)^(always\s+.{10,})", re.MULTILINE),
        re.compile(r"(?i)^(do not\s+.{10,})", re.MULTILINE),
        re.compile(r"(?i)^(must\s+.{10,})", re.MULTILINE),
    ]
    seen: set = set()

    for msg in messages:
        content = msg.get("content", "")
        for pattern in patterns:
            for match in pattern.finditer(content):
                rule_text = match.group(1).strip()
                if len(rule_text) < 10:
                    continue
                rule_key = rule_text.lower()[:80]
                if rule_key not in seen:
                    seen.add(rule_key)
                    rules.append(rule_text)
    return rules


def _extract_rules_gemini(messages: List[Dict], hub_content: str) -> List[str]:
    """Use Gemini API to extract policy updates from messages + hub content."""
    # Try to find a Gemini API key
    api_key = ""
    for env_name in [
        "GEMINI_API_KEY", "GEMINI_API_KEY_FREE_OPS",
        "GEMINI_API_KEY_OPS", "GEMINI_API_KEY_FALLBACK",
    ]:
        val = os.environ.get(env_name, "").strip()
        if val:
            api_key = val
            break

    if not api_key:
        _log("[gemini] no API key found, skipping Gemini extraction")
        return []

    # Build prompt
    prompt_parts = [
        "Extract ONLY actionable annotation-policy updates from these messages.",
        "Return a JSON object with key 'policy_updates' containing a list of short imperative rule strings.",
        "If no real updates exist, return {\"policy_updates\": []}.",
        "",
    ]

    if messages:
        prompt_parts.append("=== Discord Messages ===")
        for msg in messages[-50:]:  # limit to 50 most recent
            author = msg.get("author", "unknown")
            channel = msg.get("channel", "")
            content = msg.get("content", "")[:800]
            prompt_parts.append(f"[{author} in #{channel}]: {content}")
        prompt_parts.append("")

    if hub_content:
        # Include a digest of hub content changes
        hub_snippet = hub_content[:3000]
        prompt_parts.append("=== Training Hub Content (excerpt) ===")
        prompt_parts.append(hub_snippet)

    prompt = "\n".join(prompt_parts)

    # Call Gemini API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }

    try:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{url}?key={api_key}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = ""
            try:
                parts = data["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts).strip()
            except (KeyError, IndexError):
                pass
            if text:
                parsed = json.loads(text) if text.startswith("{") else {}
                updates = parsed.get("policy_updates", [])
                if isinstance(updates, list):
                    return [str(u) for u in updates if isinstance(u, str) and len(u) > 5]
    except Exception as e:
        _log(f"[gemini] extraction failed: {e}")

    return []


def _sync_policy_files(policy_updates: List[str], sources_count: int) -> Dict[str, Any]:
    """Write policy updates to the override file and sync into main policy."""
    POLICY_OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Build override text
    lines = [
        "# AUTO-GENERATED FROM DISCORD + TRAINING HUB UPDATES",
        f"# generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"# source_messages: {sources_count}",
        "",
    ]
    if not policy_updates:
        lines.append("- No new actionable policy updates detected in this cycle.")
    else:
        for rule in policy_updates:
            lines.append(f"- {rule}")
    override_text = "\n".join(lines).strip() + "\n"

    POLICY_OVERRIDE_FILE.write_text(override_text, encoding="utf-8")

    # Sync into main policy file if it exists
    main_updated = False
    if MAIN_POLICY_FILE.exists():
        try:
            base_text = MAIN_POLICY_FILE.read_text(encoding="utf-8")
            managed_block = f"{MANAGED_BLOCK_START}\n{override_text.rstrip()}\n{MANAGED_BLOCK_END}\n"

            if MANAGED_BLOCK_START in base_text and MANAGED_BLOCK_END in base_text:
                i = base_text.find(MANAGED_BLOCK_START)
                j = base_text.find(MANAGED_BLOCK_END, i)
                if j >= 0:
                    j_end = j + len(MANAGED_BLOCK_END)
                    while j_end < len(base_text) and base_text[j_end] in "\r\n":
                        j_end += 1
                    new_text = base_text[:i] + managed_block + base_text[j_end:]
                    if new_text != base_text:
                        MAIN_POLICY_FILE.write_text(new_text, encoding="utf-8")
                        main_updated = True
            else:
                # Append managed block
                if not base_text.endswith("\n\n"):
                    base_text = base_text.rstrip("\n") + "\n\n"
                new_text = base_text + managed_block
                MAIN_POLICY_FILE.write_text(new_text, encoding="utf-8")
                main_updated = True
        except Exception as e:
            _log(f"[policy] main policy sync failed: {e}")

    return {
        "override_file": str(POLICY_OVERRIDE_FILE),
        "main_policy_file": str(MAIN_POLICY_FILE),
        "main_policy_updated": main_updated,
    }


# ---------------------------------------------------------------------------
# Main check cycle
# ---------------------------------------------------------------------------


def _run_check(state: dict) -> Dict[str, Any]:
    """Run one complete update check cycle."""
    cycle_start = time.time()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"check_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    discord_token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()

    # 1. Check Discord
    _log("[check] checking Discord channels...")
    discord_messages, discord_changes = _check_discord(discord_token, state)

    # 2. Check Training Hub
    _log("[check] checking Training Hub...")
    hub_content, hub_changes = _check_training_hub(state)

    # 3. Combine changes
    all_changes = discord_changes + hub_changes
    relevant_messages = [m for m in discord_messages if _is_policy_relevant(m)]

    # 4. Extract policy rules
    policy_updates: List[str] = []
    if relevant_messages or hub_changes:
        # Try native extraction first
        native_rules = _extract_rules_native(relevant_messages)
        if native_rules:
            policy_updates = native_rules
            _log(f"[check] native extraction found {len(native_rules)} rules")
        else:
            # Fall back to Gemini
            gemini_rules = _extract_rules_gemini(relevant_messages, hub_content if hub_changes else "")
            if gemini_rules:
                policy_updates = gemini_rules
                _log(f"[check] Gemini extraction found {len(gemini_rules)} rules")

    # 5. Sync policy files
    if policy_updates:
        sync_result = _sync_policy_files(policy_updates, len(relevant_messages))
        _log(f"[check] policy sync: override={sync_result['override_file']} main_updated={sync_result['main_policy_updated']}")
    else:
        sync_result = {"override_file": "", "main_policy_file": "", "main_policy_updated": False}

    # 6. Save run artifacts
    run_index = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "discord_messages_total": len(discord_messages),
        "discord_messages_relevant": len(relevant_messages),
        "hub_content_length": len(hub_content),
        "changes_detected": all_changes,
        "policy_updates": policy_updates,
        "policy_sync": sync_result,
        "duration_sec": round(time.time() - cycle_start, 1),
    }
    (run_dir / "INDEX.json").write_text(
        json.dumps(run_index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if discord_messages:
        (run_dir / "discord_messages.json").write_text(
            json.dumps(discord_messages, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # 7. Send Telegram alert if changes found
    if all_changes:
        alert_lines = [
            "Atlas Update Checker - Changes Detected!",
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "Changes:",
        ]
        for change in all_changes:
            alert_lines.append(f"  - {change}")

        if policy_updates:
            alert_lines.append("")
            alert_lines.append(f"Policy updates extracted: {len(policy_updates)}")
            for rule in policy_updates[:5]:
                alert_lines.append(f"  * {rule[:100]}")
            if len(policy_updates) > 5:
                alert_lines.append(f"  ... and {len(policy_updates) - 5} more")

        if relevant_messages:
            alert_lines.append("")
            alert_lines.append("Recent relevant messages:")
            for msg in relevant_messages[:3]:
                snippet = msg["content"][:120].replace("\n", " ")
                alert_lines.append(f"  [{msg['author']}] {snippet}")

        _send_telegram("\n".join(alert_lines))
        _log(f"[check] Telegram alert sent for {len(all_changes)} change(s)")
    else:
        _log("[check] no changes detected")

    # Update state
    state["last_check_epoch"] = time.time()
    state["last_check_iso"] = datetime.now(timezone.utc).isoformat()
    state["total_checks"] = state.get("total_checks", 0) + 1
    state["total_discord_messages_seen"] = state.get("total_discord_messages_seen", 0) + len(discord_messages)
    state["total_changes_detected"] = state.get("total_changes_detected", 0) + len(all_changes)
    state["total_policy_updates"] = state.get("total_policy_updates", 0) + len(policy_updates)

    return run_index


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------


def _load_env():
    """Load .env file."""
    global TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS
    env_file = APP_DIR / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key, val)
        except Exception:
            pass

    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    raw_chats = os.environ.get("TELEGRAM_ALLOWED_CHATS", "").strip()
    if raw_chats:
        TELEGRAM_CHAT_IDS = [
            int(c.strip()) for c in raw_chats.split(",")
            if c.strip().lstrip("-").isdigit()
        ]

    # Bot's own server channels (the only ones the bot can access)
    global DISCORD_CHANNELS
    DISCORD_CHANNELS = {
        "atlas-rules": os.environ.get("COMMAND_CHANNEL_ID", "1485540323584118877"),
        "general":     "1485540039910883440",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Atlas Update Checker")
    parser.add_argument("--daemon", action="store_true", help="Run continuously (every 6 hours)")
    parser.add_argument("--interval", type=int, default=None, help="Check interval in seconds")
    args = parser.parse_args()

    _load_env()
    _log("[updater] Atlas Update Checker starting")
    _log(f"[updater] app_dir={APP_DIR}")
    _log(f"[updater] telegram={'configured' if TELEGRAM_BOT_TOKEN else 'NOT configured'}")
    _log(f"[updater] discord={'configured' if os.environ.get('DISCORD_BOT_TOKEN') else 'NOT configured'}")
    _log(f"[updater] channels={list(DISCORD_CHANNELS.keys())}")

    state = _load_state()
    interval = args.interval or CHECK_INTERVAL_SEC

    if not args.daemon:
        # Single check
        result = _run_check(state)
        _save_state(state)
        _log(f"[updater] check complete: changes={len(result.get('changes_detected', []))} "
             f"policy_updates={len(result.get('policy_updates', []))} "
             f"duration={result.get('duration_sec', 0)}s")
        return

    # Daemon mode
    _log(f"[updater] daemon mode: interval={interval}s ({interval/3600:.1f}h)")
    _send_telegram(
        f"Atlas Update Checker started (daemon mode)\n"
        f"Check interval: {interval/3600:.1f}h\n"
        f"Discord channels: {len(DISCORD_CHANNELS)}\n"
        f"Training hub: {TRAINING_HUB_URL}"
    )

    while _running:
        try:
            result = _run_check(state)
            _save_state(state)
            _log(f"[updater] cycle complete: changes={len(result.get('changes_detected', []))} "
                 f"duration={result.get('duration_sec', 0)}s")
        except Exception as e:
            _log(f"[updater] check cycle error: {e}")

        # Sleep in small increments
        _log(f"[updater] next check in {interval/3600:.1f}h")
        for _ in range(interval):
            if not _running:
                break
            time.sleep(1)

    _log("[updater] shutting down")
    _send_telegram("Atlas Update Checker stopped")


if __name__ == "__main__":
    main()
