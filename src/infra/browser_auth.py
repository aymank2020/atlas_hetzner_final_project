"""Authentication and browser-profile helpers extracted from the legacy solver."""

from __future__ import annotations

import html
import imaplib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from datetime import datetime, timedelta, timezone
from email import message_from_bytes
from email.header import decode_header
from email.message import Message
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from playwright.sync_api import Page

from src.infra.artifacts import _ensure_parent
from src.infra.solver_config import DEFAULT_CONFIG, _cfg_get, _resolve_secret


def _default_chrome_user_data_dir() -> str:
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return str(Path(local_app_data) / "Google" / "Chrome" / "User Data")
    return ""


def _looks_like_profile_dir_name(name: str) -> bool:
    normalized = (name or "").strip()
    return normalized == "Default" or normalized.startswith("Profile ")


def _is_direct_profile_path(path_value: str) -> bool:
    if not path_value:
        return False
    path = Path(path_value)
    if not path.exists() or not path.is_dir():
        return False
    if not _looks_like_profile_dir_name(path.name):
        return False
    return (path / "Preferences").exists()


def _resolve_atlas_email(cfg: Dict[str, Any]) -> str:
    return (
        str(_cfg_get(cfg, "atlas.email", "")).strip()
        or os.environ.get("ATLAS_LOGIN_EMAIL", "").strip()
        or os.environ.get("ATLAS_EMAIL", "").strip()
    )


def _detect_chrome_profile_for_email(user_data_dir: str, email: str) -> str:
    if not user_data_dir or not email:
        return ""
    root = Path(user_data_dir)
    if not root.exists():
        return ""
    target = email.strip().lower()
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        if not _looks_like_profile_dir_name(child.name):
            continue
        pref = child / "Preferences"
        if not pref.exists():
            continue
        try:
            raw = json.loads(pref.read_text(encoding="utf-8"))
        except Exception:
            continue
        account_info = raw.get("account_info", [])
        if isinstance(account_info, list):
            for acc in account_info:
                if not isinstance(acc, dict):
                    continue
                acc_email = str(acc.get("email", "")).strip().lower()
                if acc_email and acc_email == target:
                    return child.name
    return ""


def _count_site_cookies_in_profile(profile_dir: Path, domain_hint: str) -> int:
    if not profile_dir.exists():
        return 0
    db_candidates = [profile_dir / "Network" / "Cookies", profile_dir / "Cookies"]
    for db_path in db_candidates:
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM cookies WHERE host_key LIKE ?",
                (f"%{domain_hint}%",),
            )
            row = cur.fetchone()
            conn.close()
            return int(row[0]) if row else 0
        except Exception:
            continue
    return 0


def _detect_chrome_profile_for_site_cookie(user_data_dir: str, domain_hint: str = "atlascapture.io") -> str:
    if not user_data_dir:
        return ""
    root = Path(user_data_dir)
    if not root.exists():
        return ""
    best_profile = ""
    best_count = 0
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        if not _looks_like_profile_dir_name(child.name):
            continue
        cookie_count = _count_site_cookies_in_profile(child, domain_hint=domain_hint)
        if cookie_count > best_count:
            best_count = cookie_count
            best_profile = child.name
    if best_profile:
        print(f"[browser] detected profile with atlas cookies: {best_profile} (cookies={best_count})")
    return best_profile


def _otp_provider(cfg: Dict[str, Any]) -> str:
    return str(_cfg_get(cfg, "otp.provider", "gmail_imap")).strip().lower()


def _otp_is_manual(cfg: Dict[str, Any]) -> bool:
    return _otp_provider(cfg) in {"manual", "manual_browser", "browser", "none"}


def _is_authenticated_page(page: Page) -> bool:
    url = (page.url or "").lower()
    if "/login" in url or "/verify" in url:
        return False
    return "/dashboard" in url or "/tasks" in url


def _restore_storage_state(context: Any, page: Page, state_path: Path) -> bool:
    if not state_path.exists():
        return False
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    restored_any = False
    cookies = data.get("cookies")
    if isinstance(cookies, list) and cookies:
        try:
            context.add_cookies(cookies)
            print(f"[auth] restored {len(cookies)} cookies from state: {state_path}")
            restored_any = True
        except Exception:
            pass
    origins = data.get("origins")
    if isinstance(origins, list) and origins:
        for item in origins:
            if not isinstance(item, dict):
                continue
            origin = str(item.get("origin", "")).strip()
            ls_items = item.get("localStorage")
            if not origin or not isinstance(ls_items, list):
                continue
            try:
                temp_page = context.new_page()
                temp_page.goto(origin, wait_until="domcontentloaded", timeout=30000)
                temp_page.evaluate(
                    """(items) => {
                        for (const it of items || []) {
                            if (it && typeof it.name === 'string') {
                                localStorage.setItem(it.name, String(it.value ?? ''));
                            }
                        }
                    }""",
                    ls_items,
                )
                temp_page.close()
                restored_any = True
            except Exception:
                continue
    try:
        if restored_any:
            page.goto("about:blank")
    except Exception:
        pass
    return restored_any


def _is_too_many_redirects_error(exc: Exception) -> bool:
    return "ERR_TOO_MANY_REDIRECTS" in str(exc or "")


def _clear_atlas_site_session(page: Page) -> None:
    try:
        ctx = page.context
    except Exception:
        ctx = None
    if ctx is not None:
        for domain in ("atlascapture.io", ".atlascapture.io", "audit.atlascapture.io"):
            try:
                ctx.clear_cookies(domain=domain)
            except Exception:
                continue
    for origin in ("https://audit.atlascapture.io", "https://atlascapture.io"):
        try:
            page.goto(origin, wait_until="domcontentloaded", timeout=20000)
            page.evaluate(
                """() => {
                    try { localStorage.clear(); } catch (e) {}
                    try { sessionStorage.clear(); } catch (e) {}
                }"""
            )
        except Exception:
            continue
    try:
        page.goto("about:blank", wait_until="commit", timeout=8000)
    except Exception:
        pass


def _close_chrome_processes() -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/IM", "chrome.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        subprocess.run(
            ["pkill", "-f", "chrome"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _prepare_chrome_profile_clone(
    source_user_data_dir: str,
    profile_directory: str,
    target_user_data_dir: str,
    reuse_existing: bool = True,
) -> str:
    src_root = Path(source_user_data_dir)
    if not src_root.exists():
        raise FileNotFoundError(f"Chrome user data dir not found: {source_user_data_dir}")

    src_profile = src_root / profile_directory
    if not src_profile.exists():
        raise FileNotFoundError(f"Chrome profile directory not found: {src_profile}")

    dst_root = Path(target_user_data_dir).resolve()
    if reuse_existing and (dst_root / profile_directory).exists():
        print(f"[browser] reusing existing cloned profile: {dst_root}")
        return str(dst_root)
    if dst_root.exists():
        shutil.rmtree(dst_root, ignore_errors=True)
    dst_root.mkdir(parents=True, exist_ok=True)

    for root_name in ["Local State", "First Run"]:
        src_file = src_root / root_name
        if src_file.exists() and src_file.is_file():
            shutil.copy2(src_file, dst_root / root_name)

    skip_dir_names = {
        "Cache",
        "Code Cache",
        "GPUCache",
        "ShaderCache",
        "GrShaderCache",
        "DawnCache",
        "Service Worker",
        "Media Cache",
        "Crashpad",
    }
    copied = 0
    skipped = 0
    src_root_profile = src_profile.resolve()
    dst_root_profile = (dst_root / profile_directory).resolve()
    dst_root_profile.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(src_root_profile):
        dirs[:] = [d for d in dirs if d not in skip_dir_names]
        rel_root = Path(root).resolve().relative_to(src_root_profile)
        dst_dir = (dst_root_profile / rel_root).resolve()
        dst_dir.mkdir(parents=True, exist_ok=True)
        for file_name in files:
            src_file = Path(root) / file_name
            dst_file = dst_dir / file_name
            try:
                shutil.copy2(src_file, dst_file)
                copied += 1
            except Exception:
                skipped += 1

    print(f"[browser] profile clone done. copied_files={copied}, skipped_files={skipped}")
    return str(dst_root)


def _decode_mime_header(value: str) -> str:
    if not value:
        return ""
    out: List[str] = []
    for chunk, enc in decode_header(value):
        if isinstance(chunk, bytes):
            charset = enc or "utf-8"
            try:
                out.append(chunk.decode(charset, errors="ignore"))
            except Exception:
                out.append(chunk.decode("utf-8", errors="ignore"))
        else:
            out.append(str(chunk))
    return "".join(out).strip()


def _message_to_text(msg: Message) -> str:
    parts: List[str] = []
    walk = msg.walk() if msg.is_multipart() else [msg]

    for part in walk:
        ctype = (part.get_content_type() or "").lower()
        disp = (part.get("Content-Disposition") or "").lower()
        if "attachment" in disp:
            continue
        if ctype not in {"text/plain", "text/html"}:
            continue
        payload = part.get_payload(decode=True) or b""
        charset = part.get_content_charset() or "utf-8"
        try:
            text = payload.decode(charset, errors="ignore")
        except Exception:
            text = payload.decode("utf-8", errors="ignore")
        if ctype == "text/html":
            text = re.sub(r"(?is)<script.*?>.*?</script>|<style.*?>.*?</style>", " ", text)
            text = re.sub(r"(?is)<[^>]+>", " ", text)
            text = html.unescape(text)
        parts.append(text)
    return "\n".join(parts).strip()


def _extract_otp_from_messages(
    rows: List[Tuple[datetime, str, str, str]],
    code_regex: str,
    sender_hint: str,
    subject_hint: str,
    not_before: datetime,
) -> str:
    sender_hint = sender_hint.strip().lower()
    subject_hint = subject_hint.strip().lower()
    regex = re.compile(code_regex)

    for msg_dt, sender, subject, body in rows:
        if msg_dt < not_before:
            continue
        if sender_hint and sender_hint not in sender.lower():
            continue
        if subject_hint and subject_hint not in subject.lower():
            continue
        hay = "\n".join([subject, body])
        match = regex.search(hay)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return ""


def _imap_login_from_cfg(cfg: Dict[str, Any]) -> Tuple[str, int, str, str]:
    otp_cfg = _cfg_get(cfg, "otp", {})
    host = str(otp_cfg.get("imap_host", "imap.gmail.com"))
    port = int(otp_cfg.get("imap_port", 993))
    user = _resolve_secret(str(otp_cfg.get("gmail_email", "")), ["ATLAS_LOGIN_EMAIL", "GMAIL_EMAIL"])
    password = _resolve_secret(
        str(otp_cfg.get("gmail_app_password", "")),
        ["ATLAS_GMAIL_APP_PASSWORD", "GMAIL_APP_PASSWORD"],
    )
    password = re.sub(r"\s+", "", password or "")
    if not user or not password:
        raise RuntimeError(
            "Missing Gmail IMAP credentials. Set otp.gmail_email + otp.gmail_app_password "
            "or env vars GMAIL_EMAIL + GMAIL_APP_PASSWORD."
        )
    return host, port, user, password


def _get_gmail_uid_watermark(cfg: Dict[str, Any]) -> Optional[int]:
    try:
        host, port, user, password = _imap_login_from_cfg(cfg)
    except Exception:
        return None

    imap = imaplib.IMAP4_SSL(host, port)
    try:
        imap.login(user, password)
        otp_cfg = _cfg_get(cfg, "otp", {})
        mailbox = str(otp_cfg.get("mailbox", "[Gmail]/All Mail")).strip() or "[Gmail]/All Mail"
        selected = _select_imap_mailbox(imap, mailbox)
        print(f"[otp] watermark mailbox: {selected}")
        status, data = imap.uid("search", None, "ALL")
        if status != "OK" or not data or not data[0]:
            return None
        parts = [p for p in data[0].split() if p]
        if not parts:
            return None
        try:
            return int(parts[-1].decode("utf-8", errors="ignore"))
        except Exception:
            return None
    except Exception:
        return None
    finally:
        try:
            imap.logout()
        except Exception:
            pass


def _extract_mailbox_name_from_list_line(line: str) -> str:
    line = line.strip()
    match = re.search(r'"([^"]+)"\s*$', line)
    if match:
        return match.group(1).strip()
    parts = line.split(" ")
    if parts:
        return parts[-1].strip('"').strip()
    return ""


def _select_imap_mailbox(imap: imaplib.IMAP4_SSL, preferred: str) -> str:
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["INBOX", "[Gmail]/All Mail", "All Mail", "[Gmail]/Spam", "Spam"])

    status, data = imap.list()
    if status == "OK" and data:
        auto_all: List[str] = []
        for row in data:
            line = row.decode("utf-8", errors="ignore") if isinstance(row, bytes) else str(row)
            name = _extract_mailbox_name_from_list_line(line)
            if not name:
                continue
            if "\\All" in line or "all mail" in name.lower():
                auto_all.append(name)
            candidates.append(name)
        candidates = auto_all + candidates

    seen = set()
    ordered: List[str] = []
    for candidate in candidates:
        key = candidate.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(candidate.strip())

    for mailbox in ordered:
        for attempt in (mailbox, f'"{mailbox}"'):
            try:
                status, _ = imap.select(attempt)
                if status == "OK":
                    return attempt
            except Exception:
                continue
    raise RuntimeError("Could not select a readable mailbox for OTP search.")


def _fetch_otp_gmail_imap(cfg: Dict[str, Any], started_at_unix: float, min_uid: Optional[int] = None) -> str:
    host, port, user, password = _imap_login_from_cfg(cfg)
    otp_cfg = _cfg_get(cfg, "otp", {})
    mailbox = str(otp_cfg.get("mailbox", "[Gmail]/All Mail")).strip() or "[Gmail]/All Mail"

    timeout_sec = int(otp_cfg.get("timeout_sec", 120))
    poll_sec = max(1.0, float(otp_cfg.get("poll_interval_sec", 4)))
    max_messages = max(5, int(otp_cfg.get("max_messages", 25)))
    unseen_only = bool(otp_cfg.get("unseen_only", False))
    sender_hint = str(otp_cfg.get("sender_hint", ""))
    subject_hint = str(otp_cfg.get("subject_hint", ""))
    code_regex = str(otp_cfg.get("code_regex", r"\b(\d{6})\b"))
    lookback_sec = max(0, int(otp_cfg.get("lookback_sec", 300)))

    started_at = datetime.fromtimestamp(started_at_unix, tz=timezone.utc).replace(microsecond=0)
    not_before = started_at - timedelta(seconds=lookback_sec)
    deadline = time.time() + timeout_sec
    print(f"[otp] polling Gmail for OTP (timeout={timeout_sec}s)")

    while time.time() < deadline:
        rows: List[Tuple[datetime, str, str, str]] = []
        imap = imaplib.IMAP4_SSL(host, port)
        try:
            try:
                imap.login(user, password)
            except imaplib.IMAP4.error as exc:
                msg = str(exc)
                lowered = msg.lower()
                if "application-specific password required" in lowered or "app password" in lowered:
                    raise RuntimeError(
                        "Gmail IMAP login failed: App Password required. "
                        "Enable 2-Step Verification on this Google account, then create a 16-char App Password "
                        "and put it in otp.gmail_app_password (or env GMAIL_APP_PASSWORD)."
                    ) from exc
                raise RuntimeError(f"Gmail IMAP login failed: {msg}") from exc
            _select_imap_mailbox(imap, mailbox)
            criteria = "UNSEEN" if unseen_only else "ALL"
            status, data = imap.uid("search", None, criteria)
            if status != "OK":
                raise RuntimeError(f"IMAP search failed: {status}")
            uid_items = [u for u in (data[0].split() if data and data[0] else []) if u]
            uids: List[str] = []
            for raw_uid in uid_items:
                try:
                    uid_int = int(raw_uid.decode("utf-8", errors="ignore"))
                except Exception:
                    continue
                if min_uid is not None and uid_int <= int(min_uid):
                    continue
                uids.append(str(uid_int))
            uids = uids[-max_messages:]

            for uid in reversed(uids):
                status, fetched = imap.uid("fetch", uid, "(RFC822 INTERNALDATE)")
                if status != "OK":
                    continue
                raw_bytes = b""
                msg_dt = datetime.now(timezone.utc)
                for entry in fetched:
                    if not isinstance(entry, tuple):
                        continue
                    if isinstance(entry[1], bytes):
                        raw_bytes = entry[1]
                    header_text = (entry[0] or b"").decode("utf-8", errors="ignore")
                    match = re.search(r'INTERNALDATE \"([^\"]+)\"', header_text)
                    if match:
                        try:
                            dt = parsedate_to_datetime(match.group(1))
                            msg_dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                        except Exception:
                            pass
                if not raw_bytes:
                    continue

                msg = message_from_bytes(raw_bytes)
                subject = _decode_mime_header(msg.get("Subject", ""))
                sender = _decode_mime_header(msg.get("From", ""))
                body = _message_to_text(msg)
                rows.append((msg_dt, sender, subject, body))

            if not rows:
                time.sleep(poll_sec)
                continue

            code = _extract_otp_from_messages(
                rows=rows,
                code_regex=code_regex,
                sender_hint=sender_hint,
                subject_hint=subject_hint,
                not_before=not_before,
            )
            if not code:
                code = _extract_otp_from_messages(
                    rows=rows,
                    code_regex=code_regex,
                    sender_hint="",
                    subject_hint="",
                    not_before=not_before,
                )
            if code:
                print("[otp] OTP found.")
                return code
        finally:
            try:
                imap.logout()
            except Exception:
                pass
        time.sleep(poll_sec)

    raise TimeoutError("OTP not found in Gmail within timeout.")


def _resolve_otp_code(cfg: Dict[str, Any], started_at_unix: float, min_uid: Optional[int] = None) -> str:
    provider = _otp_provider(cfg)
    if provider in {"manual", "manual_browser", "browser", "none"}:
        return ""
    if provider in {"gmail_imap", "imap"}:
        return _fetch_otp_gmail_imap(cfg, started_at_unix, min_uid=min_uid)
    raise ValueError(f"Unsupported otp.provider: {provider}")


def _body_has_rate_limit(page: Page) -> bool:
    try:
        text = (page.inner_text("body") or "").lower()
    except Exception:
        return False
    return "too many request" in text or "rate limit" in text


def _wait_until_authenticated(page: Page, cfg: Dict[str, Any], timeout_sec: int) -> None:
    from src.solver import legacy_impl as _legacy

    tasks_nav = str(_cfg_get(cfg, "atlas.selectors.tasks_nav", ""))
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        current = page.url.lower()
        is_loginish = "/login" in current or "/verify" in current
        if ("/dashboard" in current or "/tasks" in current) and not is_loginish:
            print(f"[auth] authenticated at {page.url}")
            return
        if tasks_nav and _legacy._any_locator_exists(page, tasks_nav) and not is_loginish:
            print(f"[auth] authenticated (tasks nav visible) at {page.url}")
            return
        if _body_has_rate_limit(page):
            raise RuntimeError("Atlas login is rate-limited: too many requests.")
        time.sleep(0.5)
    raise TimeoutError(f"Authentication timeout after {timeout_sec}s.")


def ensure_logged_in(page: Page, cfg: Dict[str, Any]) -> None:
    from src.solver import legacy_impl as _legacy

    email = _resolve_atlas_email(cfg)
    login_url = str(_cfg_get(cfg, "atlas.login_url", DEFAULT_CONFIG["atlas"]["login_url"]))
    room_url = str(_cfg_get(cfg, "atlas.room_url", DEFAULT_CONFIG["atlas"].get("room_url", ""))).strip()
    dashboard_url = str(_cfg_get(cfg, "atlas.dashboard_url", DEFAULT_CONFIG["atlas"].get("dashboard_url", ""))).strip()
    timeout_sec = int(_cfg_get(cfg, "atlas.auth_timeout_sec", 180))

    email_sel = str(_cfg_get(cfg, "atlas.selectors.email_input", ""))
    start_sel = str(_cfg_get(cfg, "atlas.selectors.start_button", ""))
    otp_sel = str(_cfg_get(cfg, "atlas.selectors.otp_input", ""))
    verify_sel = str(_cfg_get(cfg, "atlas.selectors.verify_button", ""))

    print(f"[auth] open login page: {login_url}")
    try:
        page.goto(login_url, wait_until="domcontentloaded")
    except Exception as exc:
        if _is_too_many_redirects_error(exc):
            print("[auth] login redirect loop detected; clearing Atlas session and retrying login page once.")
            _clear_atlas_site_session(page)
            page.goto(login_url, wait_until="domcontentloaded")
        else:
            raise
    if "/dashboard" in page.url.lower() or "/tasks" in page.url.lower():
        print("[auth] already logged in via existing session.")
        return

    if _body_has_rate_limit(page):
        raise RuntimeError("Atlas login is rate-limited: 'Too many requests have been made'.")

    otp_uid_watermark: Optional[int] = None
    if email:
        if not _otp_is_manual(cfg):
            otp_uid_watermark = _get_gmail_uid_watermark(cfg)
            if otp_uid_watermark is not None:
                print(f"[otp] inbox uid watermark before request: {otp_uid_watermark}")
        if not _legacy._safe_fill(page, email_sel, email, timeout_ms=8000):
            raise RuntimeError("Could not fill Atlas email input.")
        if not _legacy._safe_locator_click(page, start_sel, timeout_ms=8000):
            raise RuntimeError("Could not click Atlas start button.")
    else:
        print("[auth] atlas.email not set; relying on existing logged-in session/profile only.")
        if "/login" in page.url.lower() or "/verify" in page.url.lower():
            state_path = Path(
                str(_cfg_get(cfg, "browser.storage_state_path", DEFAULT_CONFIG["browser"]["storage_state_path"]))
            )
            restored = _restore_storage_state(page.context, page, state_path)
            if restored:
                target_url = room_url or dashboard_url or login_url
                try:
                    page.goto(target_url, wait_until="domcontentloaded")
                except Exception:
                    pass
                if "/login" not in page.url.lower() and "/verify" not in page.url.lower():
                    _wait_until_authenticated(page, cfg, timeout_sec=timeout_sec)
                    return
            raise RuntimeError(
                "No authenticated session found and atlas.email is empty. "
                "Set ATLAS_LOGIN_EMAIL (or atlas.email) only when you want to submit a fresh login."
            )
        _wait_until_authenticated(page, cfg, timeout_sec=timeout_sec)
        return

    page.wait_for_timeout(1200)
    body_text = page.inner_text("body").lower()
    if "too many request" in body_text:
        raise RuntimeError("Atlas login is rate-limited: 'Too many requests have been made'.")
    if "applications are not currently open" in body_text or "join waitlist" in body_text:
        raise RuntimeError("Login stopped at waitlist page. This account cannot continue automatically.")

    started_at = time.time()
    if _legacy._wait_for_any(page, otp_sel, timeout_ms=20000) or "/verify" in page.url.lower():
        if _otp_is_manual(cfg):
            print("[otp] manual mode: enter OTP in the opened browser window, then script will continue.")
        else:
            code = _resolve_otp_code(cfg, started_at, min_uid=otp_uid_watermark)
            if not _legacy._safe_fill(page, otp_sel, code, timeout_ms=8000):
                raise RuntimeError("Could not fill OTP code.")
            if not _legacy._safe_locator_click(page, verify_sel, timeout_ms=8000):
                raise RuntimeError("Could not click Verify button.")

    _wait_until_authenticated(page, cfg, timeout_sec=timeout_sec)


__all__ = [
    "_default_chrome_user_data_dir",
    "_looks_like_profile_dir_name",
    "_is_direct_profile_path",
    "_resolve_atlas_email",
    "_detect_chrome_profile_for_email",
    "_count_site_cookies_in_profile",
    "_detect_chrome_profile_for_site_cookie",
    "_otp_provider",
    "_otp_is_manual",
    "_ensure_parent",
    "_is_authenticated_page",
    "_restore_storage_state",
    "_is_too_many_redirects_error",
    "_clear_atlas_site_session",
    "_close_chrome_processes",
    "_prepare_chrome_profile_clone",
    "_decode_mime_header",
    "_message_to_text",
    "_extract_otp_from_messages",
    "_imap_login_from_cfg",
    "_get_gmail_uid_watermark",
    "_extract_mailbox_name_from_list_line",
    "_select_imap_mailbox",
    "_fetch_otp_gmail_imap",
    "_resolve_otp_code",
    "_body_has_rate_limit",
    "_wait_until_authenticated",
    "ensure_logged_in",
]
