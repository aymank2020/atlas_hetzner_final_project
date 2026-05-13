"""Episode-scoped runtime container for isolated Atlas and Gemini resources."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


LoggerFn = Callable[[str, Dict[str, Any]], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _emit(logger: Optional[LoggerFn], event: str, payload: Dict[str, Any]) -> None:
    if callable(logger):
        try:
            logger(event, payload)
        except Exception:
            pass


def _is_dedicated_gemini_chat_url(url: str) -> bool:
    normalized = str(url or "").strip().rstrip("/")
    return "gemini.google.com/app/" in normalized


def _is_authenticated_gemini_page(page: Any) -> bool:
    try:
        current_url = str(getattr(page, "url", "") or "").strip().lower()
    except Exception:
        current_url = ""
    try:
        title = str(page.title() or "").strip().lower()
    except Exception:
        title = ""
    try:
        body = str(page.locator("body").inner_text(timeout=1500) or "").strip().lower()
    except Exception:
        body = ""
    if "accounts.google.com" in current_url:
        return False
    if "sign in" in title and "gemini" in title:
        return False
    if any(
        marker in body
        for marker in (
            "sign in",
            "meet gemini, your personal ai assistant",
            "get access to all gemini models",
        )
    ):
        return False
    if any(marker in body for marker in ("ask gemini", "upgrade", "plus")):
        return True
    return "gemini.google.com/app" in current_url and "sign in" not in body


def _select_existing_gemini_page(context: Any, *, target_url: str = "") -> tuple[Any, bool]:
    target = str(target_url or "").strip()
    target_is_dedicated = _is_dedicated_gemini_chat_url(target)
    fallback_page = None
    fallback_authenticated = None
    try:
        for candidate in reversed(list(getattr(context, "pages", []) or [])):
            current_url = str(getattr(candidate, "url", "") or "").strip()
            authenticated = _is_authenticated_gemini_page(candidate)
            if target and current_url.startswith(target) and authenticated:
                return candidate, True
            if target and current_url.startswith(target) and fallback_page is None:
                fallback_page = candidate
            if "gemini.google.com" in current_url and authenticated and fallback_authenticated is None:
                fallback_authenticated = candidate
            # When target is a dedicated chat URL, do not fall back to an
            # unrelated Gemini page – only exact URL prefix matches qualify.
            if fallback_page is None and "gemini.google.com" in current_url and not target_is_dedicated:
                fallback_page = candidate
    except Exception:
        return None, False
    if target_is_dedicated:
        # Dedicated URLs require an exact match; do not return unrelated pages.
        return fallback_page, fallback_page is not None
    if fallback_authenticated is not None:
        return fallback_authenticated, True
    if fallback_page is not None:
        return fallback_page, True
    return None, False


@dataclass
class EpisodeRuntime:
    episode_id: str
    context_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    atlas_browser: Any = None
    atlas_context: Any = None
    atlas_page: Any = None
    atlas_context_borrowed: bool = False
    atlas_page_borrowed: bool = False
    atlas_storage_state_path: str = ""
    atlas_page_url: str = ""
    gemini_browser: Any = None
    gemini_context: Any = None
    gemini_page: Any = None
    gemini_context_borrowed: bool = False
    gemini_page_borrowed: bool = False
    gemini_storage_state_path: str = ""
    gemini_page_url: str = ""
    gemini_cdp_url: str = ""
    _playwright_instance: Any = None
    task_state: Dict[str, Any] = field(default_factory=dict)
    opened_at: str = field(default_factory=_utc_now)
    lifecycle_events: List[Dict[str, Any]] = field(default_factory=list)

    def open(
        self,
        *,
        atlas_browser: Any = None,
        atlas_existing_context: Any = None,
        atlas_existing_page: Any = None,
        atlas_storage_state_path: str = "",
        atlas_page_url: str = "",
        gemini_browser: Any = None,
        gemini_existing_context: Any = None,
        gemini_storage_state_path: str = "",
        gemini_page_url: str = "",
        logger: Optional[LoggerFn] = None,
    ) -> "EpisodeRuntime":
        if atlas_browser is not None:
            self.atlas_browser = atlas_browser
        if gemini_browser is not None:
            self.gemini_browser = gemini_browser
        if str(atlas_storage_state_path or "").strip():
            self.atlas_storage_state_path = str(atlas_storage_state_path).strip()
        if str(gemini_storage_state_path or "").strip():
            self.gemini_storage_state_path = str(gemini_storage_state_path).strip()
        if str(atlas_page_url or "").strip():
            self.atlas_page_url = str(atlas_page_url).strip()
        if str(gemini_page_url or "").strip():
            self.gemini_page_url = str(gemini_page_url).strip()

        if atlas_existing_page is not None and self.atlas_context is None:
            self.atlas_context = atlas_existing_context
            self.atlas_context_borrowed = atlas_existing_context is not None
            self.atlas_page = atlas_existing_page
            self.atlas_page_borrowed = True
            if str(self.atlas_page_url or "").strip():
                current_url = str(getattr(self.atlas_page, "url", "") or "").strip()
                if not current_url.startswith(str(self.atlas_page_url)):
                    self.atlas_page.goto(str(self.atlas_page_url), wait_until="domcontentloaded", timeout=60000)
            event = {
                "episode_id": self.episode_id,
                "context_id": self.context_id,
                "browser": "atlas",
                "page_url": str(self.atlas_page_url or "").strip(),
                "opened_at": self.opened_at,
                "borrowed_context": bool(self.atlas_context_borrowed),
                "borrowed_page": True,
            }
            self.lifecycle_events.append({"event": "context_created", **event})
            _emit(logger, "context_created", event)
        elif atlas_existing_context is not None and self.atlas_context is None:
            self.atlas_context = atlas_existing_context
            self.atlas_context_borrowed = True
            self.atlas_page = atlas_existing_context.new_page()
            self.atlas_page_borrowed = False
            if str(self.atlas_page_url or "").strip():
                self.atlas_page.goto(str(self.atlas_page_url), wait_until="domcontentloaded", timeout=60000)
            event = {
                "episode_id": self.episode_id,
                "context_id": self.context_id,
                "browser": "atlas",
                "page_url": str(self.atlas_page_url or "").strip(),
                "opened_at": self.opened_at,
                "borrowed_context": True,
            }
            self.lifecycle_events.append({"event": "context_created", **event})
            _emit(logger, "context_created", event)
        elif atlas_browser is not None and self.atlas_context is None:
            atlas_kwargs: Dict[str, Any] = {}
            if str(self.atlas_storage_state_path or "").strip():
                atlas_kwargs["storage_state"] = str(Path(self.atlas_storage_state_path))
            self.atlas_context = atlas_browser.new_context(**atlas_kwargs)
            self.atlas_page = self.atlas_context.new_page()
            if str(self.atlas_page_url or "").strip():
                self.atlas_page.goto(str(self.atlas_page_url), wait_until="domcontentloaded", timeout=60000)
            event = {
                "episode_id": self.episode_id,
                "context_id": self.context_id,
                "browser": "atlas",
                "page_url": str(self.atlas_page_url or "").strip(),
                "opened_at": self.opened_at,
            }
            self.lifecycle_events.append({"event": "context_created", **event})
            _emit(logger, "context_created", event)

        if gemini_existing_context is not None and self.gemini_context is None:
            self.gemini_context = gemini_existing_context
            self.gemini_context_borrowed = True
            page, borrowed = _select_existing_gemini_page(
                gemini_existing_context,
                target_url=self.gemini_page_url,
            )
            self.gemini_page_borrowed = borrowed
            if page is None:
                self.gemini_page = gemini_existing_context.new_page()
                self.gemini_page_borrowed = False
            else:
                self.gemini_page = page
            if str(self.gemini_page_url or "").strip():
                current_url = str(getattr(self.gemini_page, "url", "") or "").strip()
                if not current_url.startswith(self.gemini_page_url):
                    self.gemini_page.goto(str(self.gemini_page_url), wait_until="domcontentloaded", timeout=60000)
            event = {
                "episode_id": self.episode_id,
                "context_id": self.context_id,
                "browser": "gemini",
                "page_url": str(self.gemini_page_url or "").strip(),
                "opened_at": self.opened_at,
            }
            self.lifecycle_events.append({"event": "context_created", **event})
            _emit(logger, "context_created", event)
        elif gemini_browser is not None and self.gemini_context is None:
            gemini_kwargs: Dict[str, Any] = {}
            if str(self.gemini_storage_state_path or "").strip():
                gemini_kwargs["storage_state"] = str(Path(self.gemini_storage_state_path))
            self.gemini_context = gemini_browser.new_context(**gemini_kwargs)
            self.gemini_page = self.gemini_context.new_page()
            if str(self.gemini_page_url or "").strip():
                self.gemini_page.goto(str(self.gemini_page_url), wait_until="domcontentloaded", timeout=60000)
            event = {
                "episode_id": self.episode_id,
                "context_id": self.context_id,
                "browser": "gemini",
                "page_url": str(self.gemini_page_url or "").strip(),
                "opened_at": self.opened_at,
            }
            self.lifecycle_events.append({"event": "context_created", **event})
            _emit(logger, "context_created", event)

        return self

    def reopen_gemini(self, *, logger: Optional[LoggerFn] = None) -> Any:
        if self.gemini_context_borrowed:
            try:
                if self.gemini_page is not None and not self.gemini_page_borrowed:
                    self.gemini_page.close()
            except Exception:
                pass
            finally:
                self.gemini_page = None
                self.gemini_page_borrowed = False

            borrowed_context = self.gemini_context
            if borrowed_context is None and self.gemini_browser is not None:
                contexts = list(getattr(self.gemini_browser, "contexts", []) or [])
                if contexts:
                    borrowed_context = contexts[0]
                    self.gemini_context = borrowed_context
            if borrowed_context is None:
                raise RuntimeError("EpisodeRuntime cannot reopen borrowed Gemini context without an active context.")
            try:
                page, borrowed = _select_existing_gemini_page(
                    borrowed_context,
                    target_url=self.gemini_page_url,
                )
                self.gemini_page_borrowed = borrowed
                if page is None:
                    page = borrowed_context.new_page()
                    self.gemini_page_borrowed = False
                self.gemini_page = page
                if str(self.gemini_page_url or "").strip():
                    current_url = str(getattr(page, "url", "") or "").strip()
                    if not current_url.startswith(self.gemini_page_url):
                        page.goto(str(self.gemini_page_url), wait_until="domcontentloaded", timeout=60000)
            except Exception:
                # A borrowed CDP context can go stale mid-run. Fall back to a fresh
                # isolated Gemini context instead of reusing the broken handle.
                self.gemini_page = None
                self.gemini_context = None
                self.gemini_context_borrowed = False
                self.gemini_page_borrowed = False
                if self.gemini_browser is None:
                    raise
                return self.reopen_gemini(logger=logger)

            opened_event = {
                "episode_id": self.episode_id,
                "context_id": self.context_id,
                "browser": "gemini",
                "page_url": str(self.gemini_page_url or "").strip(),
                "opened_at": _utc_now(),
            }
            self.lifecycle_events.append({"event": "context_created", **opened_event})
            _emit(logger, "context_created", opened_event)
            return self.gemini_page

        if self.gemini_context is not None:
            try:
                if self.gemini_storage_state_path:
                    Path(self.gemini_storage_state_path).parent.mkdir(parents=True, exist_ok=True)
                    self.gemini_context.storage_state(path=self.gemini_storage_state_path)
            except Exception:
                pass
        try:
            if self.gemini_page is not None:
                self.gemini_page.close()
        except Exception:
            pass
        finally:
            self.gemini_page = None
        try:
            if self.gemini_context is not None:
                self.gemini_context.close()
        except Exception:
            pass
        finally:
            self.gemini_context = None

        event = {
            "episode_id": self.episode_id,
            "context_id": self.context_id,
            "browser": "gemini",
            "closed_at": _utc_now(),
        }
        self.lifecycle_events.append({"event": "context_destroyed", **event})
        _emit(logger, "context_destroyed", event)

        if self.gemini_browser is None and not self.gemini_cdp_url:
            raise RuntimeError("EpisodeRuntime cannot reopen Gemini context without gemini_browser.")

        gemini_kwargs: Dict[str, Any] = {}
        if str(self.gemini_storage_state_path or "").strip():
            gemini_kwargs["storage_state"] = str(Path(self.gemini_storage_state_path))

        try:
            self.gemini_context = self.gemini_browser.new_context(**gemini_kwargs)
            self.gemini_page = self.gemini_context.new_page()
        except Exception:
            if self.gemini_cdp_url:
                self.gemini_browser, self.gemini_context, self.gemini_page = (
                    self._reconnect_gemini_cdp(logger=logger)
                )
            else:
                raise

        if str(self.gemini_page_url or "").strip():
            self.gemini_page.goto(str(self.gemini_page_url), wait_until="domcontentloaded", timeout=60000)

        opened_event = {
            "episode_id": self.episode_id,
            "context_id": self.context_id,
            "browser": "gemini",
            "page_url": str(self.gemini_page_url or "").strip(),
            "opened_at": _utc_now(),
        }
        self.lifecycle_events.append({"event": "context_created", **opened_event})
        _emit(logger, "context_created", opened_event)
        return self.gemini_page

    def _reconnect_gemini_cdp(self, *, logger: Optional[LoggerFn] = None) -> tuple:
        """Reconnect to Chrome via CDP when the existing browser handle is stale."""
        _emit(logger, "cdp_reconnect", {
            "episode_id": self.episode_id,
            "cdp_url": self.gemini_cdp_url,
        })
        if self._playwright_instance is None:
            from playwright.sync_api import sync_playwright
            self._playwright_instance = sync_playwright().start()
        pw = self._playwright_instance
        browser = pw.chromium.connect_over_cdp(self.gemini_cdp_url)
        contexts = browser.contexts
        if not contexts:
            raise RuntimeError(f"CDP reconnect succeeded but no contexts found at {self.gemini_cdp_url}")
        context = contexts[0]
        self.gemini_context_borrowed = True
        page, borrowed = _select_existing_gemini_page(context, target_url=self.gemini_page_url)
        self.gemini_page_borrowed = borrowed
        if page is None:
            page = context.new_page()
            self.gemini_page_borrowed = False
        return browser, context, page

    def close(self, *, logger: Optional[LoggerFn] = None) -> None:
        for browser_name, page_attr, context_attr in (
            ("gemini", "gemini_page", "gemini_context"),
            ("atlas", "atlas_page", "atlas_context"),
        ):
            page = getattr(self, page_attr, None)
            context = getattr(self, context_attr, None)
            storage_state_path = ""
            if browser_name == "gemini":
                storage_state_path = str(self.gemini_storage_state_path or "").strip()
            elif browser_name == "atlas":
                storage_state_path = str(self.atlas_storage_state_path or "").strip()
            if context is not None and storage_state_path:
                try:
                    Path(storage_state_path).parent.mkdir(parents=True, exist_ok=True)
                    context.storage_state(path=storage_state_path)
                except Exception:
                    pass
            try:
                is_borrowed_page = (
                    (browser_name == "gemini" and self.gemini_page_borrowed)
                    or (browser_name == "atlas" and self.atlas_page_borrowed)
                )
                if page is not None and not is_borrowed_page:
                    page.close()
            except Exception:
                pass
            finally:
                setattr(self, page_attr, None)
            try:
                is_borrowed_context = (
                    (browser_name == "gemini" and self.gemini_context_borrowed)
                    or (browser_name == "atlas" and self.atlas_context_borrowed)
                )
                if context is not None and not is_borrowed_context:
                    context.close()
            except Exception:
                pass
            finally:
                setattr(self, context_attr, None)
                if browser_name == "gemini":
                    self.gemini_context_borrowed = False
                    self.gemini_page_borrowed = False
                elif browser_name == "atlas":
                    self.atlas_context_borrowed = False
                    self.atlas_page_borrowed = False

            event = {
                "episode_id": self.episode_id,
                "context_id": self.context_id,
                "browser": browser_name,
                "closed_at": _utc_now(),
            }
            self.lifecycle_events.append({"event": "context_destroyed", **event})
            _emit(logger, "context_destroyed", event)


__all__ = ["EpisodeRuntime"]
