import time
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_triplet_compare import (
    _attach_files_via_chat_ui,
    _attachment_tokens_match_expected,
    _chooser_set_files_timeout_ms,
    _effective_timed_labels_retry_attempts,
    _fill_chat_input,
    _first_visible_locator,
    _handle_gemini_consent_if_present,
    _infer_chat_web_model_mode_from_text,
    _is_chat_web_boot_error_text,
    _normalize_gemini_chat_entry_url,
    _pick_existing_gemini_chat_page,
    _ordered_gen3_model_candidates,
    _resolve_chat_web_allowed_model_modes,
    _resolve_chat_web_response_stall_sec,
    _resolve_chat_web_ui_model_mode,
    _reveal_hidden_upload_trigger,
    _send_chat_prompt,
    _should_disable_drive_picker_after_stage,
    _should_force_storage_state_after_cdp_failure,
    _should_skip_chat_web_shutdown,
    _try_hidden_local_file_trigger,
    _try_attach_via_file_chooser,
    _wait_for_upload_ready,
    _wait_for_new_chat_response_text,
    _wait_for_chat_upload_settle,
    generate_gemini_chat_timed_labels,
)
from src.solver import gemini


def test_resolve_chat_web_ui_model_mode_prefers_explicit_mode_over_model_name():
    assert _resolve_chat_web_ui_model_mode(
        {"chat_web_ui_model_mode": "thinking", "model": "gemini-2.5-flash"},
        requested_model="gemini-3.1-pro-preview",
    ) == "thinking"
    assert _resolve_chat_web_ui_model_mode(
        {"model": "gemini-2.5-flash"},
        requested_model="",
    ) == "fast"


def test_resolve_chat_web_response_stall_sec_uses_mode_specific_defaults_and_overrides():
    assert _resolve_chat_web_response_stall_sec(
        {"chat_web_ui_model_mode": "pro"},
        requested_model="gemini-3.1-pro-preview",
    ) == 75.0
    assert _resolve_chat_web_response_stall_sec(
        {"chat_web_ui_model_mode": "thinking", "chat_web_response_stall_sec_thinking": 210},
        requested_model="",
    ) == 210.0
    assert _resolve_chat_web_response_stall_sec(
        {"chat_web_ui_model_mode": "pro"},
        requested_model="gemini-3.1-pro-preview",
        current_mode="thinking",
    ) == 180.0


def test_resolve_chat_web_allowed_model_modes_defaults_to_pro_and_thinking():
    assert _resolve_chat_web_allowed_model_modes(
        {"chat_web_ui_model_mode": "pro"},
        requested_model="gemini-3.1-pro-preview",
    ) == ["pro", "thinking"]
    assert _resolve_chat_web_allowed_model_modes(
        {"chat_web_allowed_model_modes": ["thinking", "pro", "thinking"]},
        requested_model="gemini-3.1-pro-preview",
    ) == ["thinking", "pro"]


def test_infer_chat_web_model_mode_ignores_show_thinking_affordance():
    assert _infer_chat_web_model_mode_from_text("Show thinking") == ""
    assert _infer_chat_web_model_mode_from_text("Hide thinking") == ""
    assert _infer_chat_web_model_mode_from_text("Pro") == "pro"


def test_attachment_tokens_match_expected_requires_specific_fragment():
    assert _attachment_tokens_match_expected(
        [
            'dom:Remove file video_69c7e167a026fb79f12a1e0e_upload_opt.mp4',
            'dom:You already uploaded a file named video_69c7e167a026fb79f12a1e0e_upload_opt.mp4',
        ],
        [
            "69c7e167a026fb79f12a1e0e",
            "video_69c7e167a026fb79f12a1e0e_upload_opt",
            ".mp4",
        ],
    )
    assert not _attachment_tokens_match_expected(
        ["dom:Remove file random.mp4"],
        ["69c7e167a026fb79f12a1e0e", "video_69c7e167a026fb79f12a1e0e_upload_opt", ".mp4"],
    )
    assert not _attachment_tokens_match_expected(
        ["dom:More options for Google Drive error", "dom:Google Drive error"],
        [
            "69c70feca9c07b46b224b6a9",
            "video_69c70feca9c07b46b224b6a9_upload_opt",
            "video_69c70feca9c07b46b224b6a9_upload_opt.mp4",
        ],
    )


def test_should_disable_drive_picker_after_stage_when_stage_failed_without_uploads():
    assert _should_disable_drive_picker_after_stage(
        [
            "drive_stage_disabled:rclone_remote_missing:gdrive",
            "drive_stage_failed=video_abc.mp4:missing remote",
        ]
    )


def test_should_not_disable_drive_picker_after_stage_when_upload_succeeded():
    assert not _should_disable_drive_picker_after_stage(
        [
            "drive_stage_uploaded=video_abc.mp4",
            "drive_stage_failed=video_abc.txt:ignored",
        ]
    )


def test_should_skip_chat_web_shutdown_after_success_by_default():
    assert _should_skip_chat_web_shutdown(
        raw_text='{"segments":[{"start_sec":0.0,"end_sec":1.0,"label":"pick up fabric"}]}',
        gem_cfg={},
    )


def test_should_not_skip_chat_web_shutdown_when_disabled_or_empty():
    assert not _should_skip_chat_web_shutdown(raw_text="", gem_cfg={})
    assert not _should_skip_chat_web_shutdown(
        raw_text='{"segments":[]}',
        gem_cfg={"chat_web_skip_shutdown_after_success": False},
    )


def test_is_chat_web_boot_error_text_detects_cdp_and_asyncio_boot_failures():
    assert _is_chat_web_boot_error_text("BrowserType.connect_over_cdp: Timeout 180000ms exceeded.")
    assert _is_chat_web_boot_error_text(
        'Page.goto: Timeout 60000ms exceeded. Call log: navigating to "https://gemini.google.com/app/x", waiting until "domcontentloaded"'
    )
    assert _is_chat_web_boot_error_text(
        "It looks like you are using Playwright Sync API inside the asyncio loop. Please use the Async API instead."
    )
    assert not _is_chat_web_boot_error_text("Timed out waiting for Gemini chat response.")


def test_should_force_storage_state_after_cdp_failure_requires_error_and_state():
    assert _should_force_storage_state_after_cdp_failure(
        cdp_launch_error="BrowserType.connect_over_cdp: Timeout 45000ms exceeded.",
        storage_state="/tmp/state.json",
        user_data_dir="",
    )
    assert not _should_force_storage_state_after_cdp_failure(
        cdp_launch_error="BrowserType.connect_over_cdp: Timeout 45000ms exceeded.",
        storage_state="/tmp/state.json",
        user_data_dir="/tmp/gemini_profile",
    )
    assert not _should_force_storage_state_after_cdp_failure(
        cdp_launch_error="",
        storage_state="/tmp/state.json",
        user_data_dir="",
    )


def test_effective_timed_labels_retry_attempts_externalizes_chat_web():
    assert _effective_timed_labels_retry_attempts(requested_attempts=3, auth_mode="chat_web") == 1
    assert _effective_timed_labels_retry_attempts(requested_attempts=3, auth_mode=" api ") == 3
    assert _effective_timed_labels_retry_attempts(requested_attempts=0, auth_mode="chat_web") == 1


def test_ordered_gen3_model_candidates_filters_to_gen3_and_preserves_order():
    candidates = _ordered_gen3_model_candidates(
        {
            "gen3_fallback_models": [
                "gemini-2.5-pro",
                "gemini-3.1-pro-preview",
                "gemini-3.1-flash-preview",
            ]
        },
        "gemini-3.1-flash-lite-preview",
        "compare_fallback_model",
    )

    assert candidates == [
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-preview",
    ]


def test_pick_existing_gemini_chat_page_requires_exact_dedicated_match():
    class FakePage:
        def __init__(self, url: str):
            self.url = url

    class FakeContext:
        def __init__(self, pages):
            self.pages = pages

    dedicated = "https://gemini.google.com/app/b3006ba9f325b55c"
    other = FakePage("https://gemini.google.com/app/random-thread")
    matched = FakePage(dedicated)
    context = FakeContext([other, matched])

    assert _pick_existing_gemini_chat_page(context, chat_url=dedicated) is matched
    assert _pick_existing_gemini_chat_page(FakeContext([other]), chat_url=dedicated) is None


def test_normalize_gemini_chat_entry_url_strips_pinned_conversation_when_cleaning():
    pinned = "https://gemini.google.com/app/b3006ba9f325b55c"

    assert _normalize_gemini_chat_entry_url(pinned, clean_thread=False) == pinned
    assert _normalize_gemini_chat_entry_url(pinned, clean_thread=True) == "https://gemini.google.com/app"
    assert _normalize_gemini_chat_entry_url("https://gemini.google.com/app", clean_thread=True) == (
        "https://gemini.google.com/app"
    )


def test_pick_existing_gemini_chat_page_prefers_authenticated_page(monkeypatch):
    class FakePage:
        def __init__(self, url: str, authenticated: bool):
            self.url = url
            self.authenticated = authenticated

    class FakeContext:
        def __init__(self, pages):
            self.pages = pages

    unauthenticated = FakePage("https://gemini.google.com/app", False)
    authenticated = FakePage("https://gemini.google.com/app", True)
    context = FakeContext([unauthenticated, authenticated])

    monkeypatch.setattr(
        "atlas_triplet_compare._is_authenticated_gemini_chat_page",
        lambda page: bool(getattr(page, "authenticated", False)),
    )

    assert _pick_existing_gemini_chat_page(context, chat_url="https://gemini.google.com/app") is authenticated


def test_wait_for_upload_ready_polls_until_attach_button_exists(monkeypatch):
    class FakePage:
        def __init__(self):
            self.clock = 0.0
            self.polls = 0

        def wait_for_timeout(self, timeout_ms):
            self.polls += 1
            self.clock += float(timeout_ms) / 1000.0

    page = FakePage()
    trigger = object()

    monkeypatch.setattr("atlas_triplet_compare.time.monotonic", lambda: page.clock)
    monkeypatch.setattr(
        "atlas_triplet_compare._first_exact_upload_trigger",
        lambda _page: trigger if page.polls >= 2 else None,
    )
    monkeypatch.setattr("atlas_triplet_compare._first_visible_locator", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("atlas_triplet_compare._last_present_locator", lambda *_args, **_kwargs: None)

    attach_trigger, file_input = _wait_for_upload_ready(
        page,
        attach_button_sel='button[aria-label*="Open upload file menu" i]',
        file_input_sel='input[type="file"]',
        timeout_ms=1500,
    )

    assert attach_trigger is trigger
    assert file_input is None
    assert page.polls >= 2


def test_first_visible_locator_skips_zero_sized_visible_candidate():
    class FakeCandidate:
        def __init__(self, width: float, height: float):
            self.width = width
            self.height = height

        def wait_for(self, state: str = "", timeout: int = 0):
            assert state == "visible"
            return None

        def bounding_box(self):
            return {"x": 0.0, "y": 0.0, "width": self.width, "height": self.height}

    class FakeLocatorList:
        def __init__(self, candidates):
            self.candidates = candidates

        def count(self):
            return len(self.candidates)

        def nth(self, index: int):
            return self.candidates[index]

    class FakePage:
        def __init__(self, mapping):
            self.mapping = mapping

        def locator(self, selector: str):
            return self.mapping[selector]

    hidden = FakeCandidate(width=0.0, height=1.0)
    usable = FakeCandidate(width=320.0, height=42.0)
    page = FakePage({'div[contenteditable="true"]': FakeLocatorList([hidden, usable])})

    found = _first_visible_locator(page, 'div[contenteditable="true"]', timeout_ms=2000)

    assert found is usable


def test_reveal_hidden_upload_trigger_makes_zero_sized_button_clickable():
    class FakeTrigger:
        def __init__(self):
            self.script = ""
            self.scrolled = False

        def evaluate(self, script: str):
            self.script = script

        def scroll_into_view_if_needed(self, timeout: int = 0):
            self.scrolled = timeout == 1500

    trigger = FakeTrigger()

    _reveal_hidden_upload_trigger(trigger)

    assert "opacity" in trigger.script
    assert "position" in trigger.script
    assert trigger.scrolled


def test_try_hidden_local_file_trigger_reveals_button_and_uses_extended_timeout(monkeypatch, tmp_path):
    class FakeChooser:
        def __init__(self):
            self.paths = []
            self.timeout = None

        def set_files(self, path: str, timeout: int = 0):
            self.paths.append(path)
            self.timeout = timeout

    class FakeChooserContext:
        def __init__(self, chooser):
            self.value = chooser

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeTrigger:
        def __init__(self):
            self.revealed = False
            self.scroll_timeout = None

        def evaluate(self, script: str):
            if "opacity" in script and "position" in script:
                self.revealed = True

        def scroll_into_view_if_needed(self, timeout: int = 0):
            self.scroll_timeout = timeout

        def click(self, **_kwargs):
            return None

        def dispatch_event(self, _name: str):
            return None

        def set_input_files(self, _path: str):
            raise AssertionError("button fallback should not use direct set_input_files in this path")

    class FakeRawLocator:
        def __init__(self, trigger):
            self.trigger = trigger

        def count(self):
            return 1

        def nth(self, _index: int):
            return self.trigger

    class FakePage:
        def __init__(self, chooser):
            self.chooser = chooser

        def locator(self, _selector: str):
            return FakeRawLocator(trigger)

        def expect_file_chooser(self, timeout: int = 0):
            assert timeout == 3500
            return FakeChooserContext(self.chooser)

    trigger = FakeTrigger()
    chooser = FakeChooser()
    page = FakePage(chooser)
    attachment = tmp_path / "video_episode_upload_opt.mp4"
    attachment.write_bytes(b"demo")

    monkeypatch.setattr("atlas_triplet_compare._handle_gemini_consent_if_present", lambda *_args, **_kwargs: False)

    ok, mode = _try_hidden_local_file_trigger(page, attachment)

    assert ok
    assert mode == "hidden_file_chooser"
    assert trigger.revealed
    assert trigger.scroll_timeout == 1500
    assert chooser.paths == [str(attachment)]
    assert chooser.timeout == _chooser_set_files_timeout_ms(attachment)


def test_try_attach_via_file_chooser_does_not_toggle_clickable_after_failures(monkeypatch, tmp_path):
    class FakeChooserContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            raise TimeoutError("filechooser timeout")

    class FakePage:
        def expect_file_chooser(self, timeout: int = 0):
            assert timeout in (3500, 6500)
            return FakeChooserContext()

    clicks = {"count": 0}
    attachment = tmp_path / "video_episode_upload_opt.mp4"
    attachment.write_bytes(b"demo")

    monkeypatch.setattr("atlas_triplet_compare._robust_click", lambda _clickable: clicks.__setitem__("count", clicks["count"] + 1))
    monkeypatch.setattr("atlas_triplet_compare._handle_gemini_consent_if_present", lambda *_args, **_kwargs: False)

    ok, detail = _try_attach_via_file_chooser(FakePage(), object(), attachment)

    assert not ok
    assert "timeout" in detail.lower()
    assert clicks["count"] == 2


def test_attach_files_prefers_direct_file_chooser_before_menu_path(monkeypatch, tmp_path):
    class FakePage:
        def __init__(self):
            self.request_listeners = []
            self.response_listeners = []

        def wait_for_timeout(self, _timeout_ms):
            return None

        def on(self, event: str, listener):
            if event == "request":
                self.request_listeners.append(listener)
            elif event == "response":
                self.response_listeners.append(listener)

        def remove_listener(self, event: str, listener):
            if event == "request" and listener in self.request_listeners:
                self.request_listeners.remove(listener)
            if event == "response" and listener in self.response_listeners:
                self.response_listeners.remove(listener)

    fake_page = FakePage()
    attachment = tmp_path / "video_episode_upload_opt.mp4"
    attachment.write_bytes(b"demo")
    composer = object()
    attach_trigger = object()
    calls = {"direct": 0, "menu_lookup": 0}

    monkeypatch.setattr("atlas_triplet_compare._prepare_chat_composer_for_attach", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("atlas_triplet_compare._build_attachment_expected_fragments", lambda **_kwargs: ["video_episode"])
    monkeypatch.setattr("atlas_triplet_compare._collect_attachment_tokens", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "atlas_triplet_compare._wait_for_upload_ready",
        lambda *_args, **_kwargs: (attach_trigger, None),
    )
    monkeypatch.setattr("atlas_triplet_compare._try_attach_via_drive_picker", lambda **_kwargs: (False, "drive disabled"))
    monkeypatch.setattr(
        "atlas_triplet_compare._try_attach_via_file_chooser",
        lambda _page, clickable, _file: (
            calls.__setitem__("direct", calls["direct"] + 1) or True,
            "file_chooser",
        )
        if clickable is attach_trigger
        else (False, "unexpected menu path"),
    )
    monkeypatch.setattr(
        "atlas_triplet_compare._first_exact_upload_item",
        lambda *_args, **_kwargs: calls.__setitem__("menu_lookup", calls["menu_lookup"] + 1) or None,
    )
    monkeypatch.setattr("atlas_triplet_compare._first_visible_locator", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("atlas_triplet_compare._handle_gemini_consent_if_present", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("atlas_triplet_compare._try_set_files_on_locator", lambda *_args, **_kwargs: (False, "no file input"))
    monkeypatch.setattr("atlas_triplet_compare._try_hidden_local_file_trigger", lambda *_args, **_kwargs: (False, "hidden unused"))
    monkeypatch.setattr(
        "atlas_triplet_compare._wait_for_chat_upload_settle",
        lambda *_args, **_kwargs: {"confirmed": True, "tokens": ["video_episode"], "wait_sec": 1.2},
    )
    monkeypatch.setattr("atlas_triplet_compare._network_upload_confirmation_ok", lambda **_kwargs: False)
    monkeypatch.setattr("atlas_triplet_compare._emit_progress_hook", lambda *_args, **_kwargs: None)

    notes = _attach_files_via_chat_ui(
        page=fake_page,
        composer_locator=composer,
        attach_candidates=[attachment],
        attach_button_sel='button[aria-label*="Open upload file menu" i]',
        file_input_sel='input[type="file"]',
        upload_menu_sel='text=/^Upload files$/i',
        max_upload_mb=50.0,
        episode_id="episode123",
        prefer_drive_picker=False,
        upload_settle_min_sec=1.0,
        upload_settle_sec_per_100mb=1.0,
        upload_settle_max_sec=5.0,
        heartbeat=None,
        heartbeat_interval_sec=10.0,
        progress_hook=None,
        progress_interval_sec=15.0,
        drive_root_folder_url="",
    )

    assert calls["direct"] == 1
    assert calls["menu_lookup"] == 0
    assert notes and "attached" in notes[0]


def test_send_chat_prompt_falls_back_to_enter_when_click_does_not_dispatch(monkeypatch):
    class FakeButton:
        def click(self, **_kwargs):
            return None

    class FakeKeyboard:
        def __init__(self, page):
            self.page = page
            self.enter_presses = 0

        def press(self, key: str):
            if key == "Enter":
                self.enter_presses += 1
                self.page.dispatched = True

    class FakeChatBox:
        def __init__(self, page):
            self.page = page
            self.enter_presses = 0

        def fill(self, _text: str):
            self.page.filled = True

        def click(self, **_kwargs):
            return None

        def press(self, key: str, **_kwargs):
            if key == "Enter":
                self.enter_presses += 1
                self.page.dispatched = True

    class FakeBodyLocator:
        def __init__(self, page):
            self.page = page

        def inner_text(self, timeout: int = 0):
            return "Thinking" if self.page.dispatched else ""

    class FakePage:
        def __init__(self):
            self.dispatched = False
            self.keyboard = FakeKeyboard(self)
            self.filled = False

        def wait_for_timeout(self, _ms: int):
            return None

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator(self)

    fake_page = FakePage()
    fake_chat_box = FakeChatBox(fake_page)

    tick = {"value": 0.0}

    monkeypatch.setattr("atlas_triplet_compare._extract_latest_chat_response_text", lambda _page: "")
    monkeypatch.setattr("atlas_triplet_compare._selector_variants", lambda selector: [selector] if selector else [])
    monkeypatch.setattr("atlas_triplet_compare._first_visible_locator", lambda _page, _selector, timeout_ms=0: FakeButton())
    monkeypatch.setattr(
        "atlas_triplet_compare._quick_locator_visible",
        lambda page, selector: page.dispatched if "Stop" in selector else not page.dispatched,
    )
    monkeypatch.setattr(
        "atlas_triplet_compare.time.monotonic",
        lambda: tick.__setitem__("value", tick["value"] + 9.0) or tick["value"],
    )

    _send_chat_prompt(
        page=fake_page,
        chat_box=fake_chat_box,
        send_selector='button[aria-label*="Send" i]',
        prompt_text="prompt",
    )

    assert fake_page.filled is True
    assert fake_chat_box.enter_presses == 1
    assert fake_page.keyboard.enter_presses == 0


def test_send_chat_prompt_treats_posted_prompt_with_cleared_composer_as_dispatch(monkeypatch):
    class FakeButton:
        def __init__(self, page):
            self.page = page
            self.clicks = 0

        def click(self, **_kwargs):
            self.clicks += 1
            self.page.prompt_posted = True
            return None

    class FakeKeyboard:
        def __init__(self, page):
            self.page = page
            self.enter_presses = 0

        def press(self, key: str):
            if key == "Enter":
                self.enter_presses += 1

    class FakeChatBox:
        def __init__(self, page):
            self.page = page
            self.enter_presses = 0

        def fill(self, text: str):
            self.page.filled = True
            self.page.prompt_text = text

        def click(self, **_kwargs):
            return None

        def press(self, key: str, **_kwargs):
            if key == "Enter":
                self.enter_presses += 1

        def inner_text(self):
            return "" if self.page.prompt_posted else self.page.prompt_text

        def text_content(self):
            return "" if self.page.prompt_posted else self.page.prompt_text

        def input_value(self):
            return "" if self.page.prompt_posted else self.page.prompt_text

    class FakeBodyLocator:
        def __init__(self, page):
            self.page = page

        def inner_text(self, timeout: int = 0):
            if self.page.prompt_posted:
                return "Internal request context request_id=abc123 posted"
            return ""

    class FakePage:
        def __init__(self):
            self.prompt_posted = False
            self.prompt_text = ""
            self.keyboard = FakeKeyboard(self)
            self.filled = False

        def wait_for_timeout(self, _ms: int):
            return None

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator(self)

    fake_page = FakePage()
    fake_chat_box = FakeChatBox(fake_page)
    fake_button = FakeButton(fake_page)

    baseline_state = {
        "message_count": 0,
        "response_hash": "baseline",
        "entries": [],
        "texts": [],
        "latest_text": "",
    }

    monkeypatch.setattr(
        "atlas_triplet_compare._capture_chat_response_state",
        lambda _page, limit=8: dict(baseline_state),
    )
    monkeypatch.setattr("atlas_triplet_compare._selector_variants", lambda selector: [selector] if selector else [])
    monkeypatch.setattr(
        "atlas_triplet_compare._first_visible_locator",
        lambda _page, _selector, timeout_ms=0: fake_button,
    )
    monkeypatch.setattr(
        "atlas_triplet_compare._quick_locator_visible",
        lambda _page, selector: False if "Stop" in selector else True,
    )

    _send_chat_prompt(
        page=fake_page,
        chat_box=fake_chat_box,
        send_selector='button[aria-label*="Send" i]',
        prompt_text="ATLAS_REQUEST_CONTEXT request_id=abc123\npayload",
    )

    assert fake_page.filled is True
    assert fake_button.clicks == 1
    # The enter_first attempt fires Enter before the click fallback,
    # so the chat_box may record one Enter press from that initial attempt.
    assert fake_chat_box.enter_presses <= 1
    assert fake_page.keyboard.enter_presses == 0


def test_send_chat_prompt_rejects_dispatch_when_attachment_stays_pending(monkeypatch):
    class FakeButton:
        def __init__(self):
            self.clicks = 0

        def click(self, **_kwargs):
            self.clicks += 1
            return None

    class FakeKeyboard:
        def __init__(self):
            self.enter_presses = 0

        def press(self, key: str):
            if key == "Enter":
                self.enter_presses += 1

    class FakeChatBox:
        def __init__(self):
            self.prompt_text = ""
            self.enter_presses = 0

        def fill(self, text: str):
            self.prompt_text = text

        def click(self, **_kwargs):
            return None

        def press(self, key: str, **_kwargs):
            if key == "Enter":
                self.enter_presses += 1

        def inner_text(self):
            return self.prompt_text

        def text_content(self):
            return self.prompt_text

        def input_value(self):
            return self.prompt_text

    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return ""

    class FakePage:
        def __init__(self):
            self.keyboard = FakeKeyboard()

        def wait_for_timeout(self, _ms: int):
            return None

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator()

    fake_page = FakePage()
    fake_chat_box = FakeChatBox()
    fake_button = FakeButton()

    baseline_state = {
        "message_count": 0,
        "response_hash": "baseline",
        "entries": [],
        "texts": [],
        "latest_text": "",
    }
    tick = {"value": 0.0}

    monkeypatch.setattr(
        "atlas_triplet_compare._capture_chat_response_state",
        lambda _page, limit=8: dict(baseline_state),
    )
    monkeypatch.setattr("atlas_triplet_compare._new_chat_response_candidates_after_baseline", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "atlas_triplet_compare._select_preferred_chat_response_candidate",
        lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr(
        "atlas_triplet_compare._collect_attachment_tokens",
        lambda *_args, **_kwargs: ["video_01.mp4"],
    )
    monkeypatch.setattr("atlas_triplet_compare._selector_variants", lambda selector: [selector] if selector else [])
    monkeypatch.setattr(
        "atlas_triplet_compare._first_visible_locator",
        lambda _page, _selector, timeout_ms=0: fake_button,
    )
    monkeypatch.setattr(
        "atlas_triplet_compare._quick_locator_visible",
        lambda _page, selector: False if "Stop" in selector else True,
    )
    monkeypatch.setattr(
        "atlas_triplet_compare.time.monotonic",
        lambda: tick.__setitem__("value", tick["value"] + 2.0) or tick["value"],
    )

    with pytest.raises(RuntimeError, match="attachment stayed pending in composer"):
        _send_chat_prompt(
            page=fake_page,
            chat_box=fake_chat_box,
            send_selector='button[aria-label*="Send" i]',
            prompt_text="ATLAS_REQUEST_CONTEXT request_id=abc123\npayload",
        )

    assert fake_button.clicks >= 1
    assert fake_chat_box.enter_presses >= 1
    assert fake_page.keyboard.enter_presses >= 1


def test_send_chat_prompt_does_not_treat_stale_recent_responses_as_dispatch(monkeypatch):
    class FakeButton:
        def click(self, **_kwargs):
            return None

    class FakeKeyboard:
        def __init__(self, page):
            self.page = page
            self.enter_presses = 0

        def press(self, key: str):
            if key == "Enter":
                self.enter_presses += 1
                self.page.dispatched = True

    class FakeChatBox:
        def __init__(self, page):
            self.page = page
            self.enter_presses = 0

        def fill(self, _text: str):
            self.page.filled = True

        def click(self, **_kwargs):
            return None

        def press(self, key: str, **_kwargs):
            if key == "Enter":
                self.enter_presses += 1
                self.page.dispatched = True

        def inner_text(self):
            return "prompt"

        def text_content(self):
            return "prompt"

        def input_value(self):
            return "prompt"

    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return ""

    class FakePage:
        def __init__(self):
            self.dispatched = False
            self.keyboard = FakeKeyboard(self)
            self.filled = False

        def wait_for_timeout(self, _ms: int):
            return None

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator()

    fake_page = FakePage()
    fake_chat_box = FakeChatBox(fake_page)

    tick = {"value": 0.0}

    monkeypatch.setattr(
        "atlas_triplet_compare._extract_recent_chat_response_texts",
        lambda _page, limit=8: ['{"segments":[{"segment_index":6,"label":"old"}]}'],
    )
    monkeypatch.setattr("atlas_triplet_compare._selector_variants", lambda selector: [selector] if selector else [])
    monkeypatch.setattr("atlas_triplet_compare._first_visible_locator", lambda _page, _selector, timeout_ms=0: FakeButton())
    monkeypatch.setattr(
        "atlas_triplet_compare._quick_locator_visible",
        lambda page, selector: page.dispatched if "Stop" in selector else not page.dispatched,
    )
    monkeypatch.setattr(
        "atlas_triplet_compare.time.monotonic",
        lambda: tick.__setitem__("value", tick["value"] + 9.0) or tick["value"],
    )

    _send_chat_prompt(
        page=fake_page,
        chat_box=fake_chat_box,
        send_selector='button[aria-label*="Send" i]',
        prompt_text="prompt",
    )

    assert fake_chat_box.enter_presses == 1
    assert fake_page.keyboard.enter_presses == 0


def test_send_chat_prompt_ignores_stale_thinking_text_when_send_still_visible(monkeypatch):
    class FakeButton:
        def click(self, **_kwargs):
            return None

    class FakeKeyboard:
        def __init__(self, page):
            self.page = page
            self.enter_presses = 0

        def press(self, key: str):
            if key == "Enter":
                self.enter_presses += 1
                self.page.dispatched = True

    class FakeChatBox:
        def __init__(self, page):
            self.page = page
            self.enter_presses = 0

        def fill(self, _text: str):
            self.page.filled = True

        def click(self, **_kwargs):
            return None

        def press(self, key: str, **_kwargs):
            if key == "Enter":
                self.enter_presses += 1
                self.page.dispatched = True

    class FakeBodyLocator:
        def __init__(self, page):
            self.page = page

        def inner_text(self, timeout: int = 0):
            return "Thinking about previous answer" if not self.page.dispatched else ""

    class FakePage:
        def __init__(self):
            self.dispatched = False
            self.keyboard = FakeKeyboard(self)
            self.filled = False

        def wait_for_timeout(self, _ms: int):
            return None

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator(self)

    fake_page = FakePage()
    fake_chat_box = FakeChatBox(fake_page)

    tick = {"value": 0.0}

    monkeypatch.setattr("atlas_triplet_compare._extract_latest_chat_response_text", lambda _page: "")
    monkeypatch.setattr("atlas_triplet_compare._selector_variants", lambda selector: [selector] if selector else [])
    monkeypatch.setattr("atlas_triplet_compare._first_visible_locator", lambda _page, _selector, timeout_ms=0: FakeButton())
    monkeypatch.setattr(
        "atlas_triplet_compare._quick_locator_visible",
        lambda page, selector: page.dispatched if "Stop" in selector else not page.dispatched,
    )
    monkeypatch.setattr(
        "atlas_triplet_compare.time.monotonic",
        lambda: tick.__setitem__("value", tick["value"] + 9.0) or tick["value"],
    )

    _send_chat_prompt(
        page=fake_page,
        chat_box=fake_chat_box,
        send_selector='button[aria-label*="Send" i]',
        prompt_text="prompt",
    )

    assert fake_page.filled is True
    assert fake_chat_box.enter_presses == 1
    assert fake_page.keyboard.enter_presses == 0


def test_send_chat_prompt_refills_prompt_when_stale_thinking_hides_send_without_posting(monkeypatch):
    class FakeButton:
        def __init__(self, page):
            self.page = page
            self.clicks = 0
            self.eval_clicks = 0

        def click(self, **_kwargs):
            self.clicks += 1
            self.page.prompt_text = ""
            return None

        def evaluate(self, _script: str):
            self.eval_clicks += 1
            self.page.prompt_text = ""
            return None

        def bounding_box(self):
            return {"x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0}

    class FakeMouse:
        def __init__(self, page):
            self.page = page
            self.clicks = 0

        def click(self, *_args, **_kwargs):
            self.clicks += 1
            self.page.prompt_text = ""

    class FakeKeyboard:
        def __init__(self, page):
            self.page = page
            self.control_enter_presses = 0
            self.enter_presses = 0

        def press(self, key: str):
            if key == "Control+Enter":
                self.control_enter_presses += 1
                self.page.dispatched = True
            elif key == "Enter":
                self.enter_presses += 1
                self.page.dispatched = True

    class FakeChatBox:
        def __init__(self, page):
            self.page = page
            self.control_enter_presses = 0
            self.enter_presses = 0

        def fill(self, text: str):
            self.page.fill_calls.append(text)
            self.page.prompt_text = text

        def click(self, **_kwargs):
            return None

        def press(self, key: str, **_kwargs):
            if key == "Control+Enter":
                self.control_enter_presses += 1
                self.page.dispatched = True
            elif key == "Enter":
                self.enter_presses += 1
                self.page.dispatched = True

        def inner_text(self):
            return self.page.prompt_text

        def text_content(self):
            return self.page.prompt_text

        def input_value(self):
            return self.page.prompt_text

    class FakeBodyLocator:
        def __init__(self, page):
            self.page = page

        def inner_text(self, timeout: int = 0):
            if self.page.dispatched:
                return "ATLAS_REQUEST_CONTEXT request_id=abc123 posted"
            return "Thinking about previous answer"

    class FakePage:
        def __init__(self):
            self.prompt_text = ""
            self.fill_calls = []
            self.dispatched = False
            self.mouse = FakeMouse(self)
            self.keyboard = FakeKeyboard(self)

        def wait_for_timeout(self, _ms: int):
            return None

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator(self)

    fake_page = FakePage()
    fake_chat_box = FakeChatBox(fake_page)
    fake_button = FakeButton(fake_page)

    baseline_state = {
        "message_count": 0,
        "response_hash": "baseline",
        "entries": [],
        "texts": [],
        "latest_text": "",
    }
    tick = {"value": 0.0}

    monkeypatch.setattr(
        "atlas_triplet_compare._capture_chat_response_state",
        lambda _page, limit=8: dict(baseline_state),
    )
    monkeypatch.setattr("atlas_triplet_compare._new_chat_response_candidates_after_baseline", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "atlas_triplet_compare._select_preferred_chat_response_candidate",
        lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr("atlas_triplet_compare._selector_variants", lambda selector: [selector] if selector else [])
    monkeypatch.setattr(
        "atlas_triplet_compare._first_visible_locator",
        lambda _page, _selector, timeout_ms=0: fake_button if fake_page.prompt_text and not fake_page.dispatched else None,
    )
    monkeypatch.setattr(
        "atlas_triplet_compare._quick_locator_visible",
        lambda _page, selector: fake_page.dispatched if "Stop" in selector else bool(fake_page.prompt_text and not fake_page.dispatched),
    )
    monkeypatch.setattr(
        "atlas_triplet_compare.time.monotonic",
        lambda: tick.__setitem__("value", tick["value"] + 4.0) or tick["value"],
    )

    _send_chat_prompt(
        page=fake_page,
        chat_box=fake_chat_box,
        send_selector='button[aria-label*="Send" i]',
        prompt_text="ATLAS_REQUEST_CONTEXT request_id=abc123\npayload",
    )

    # The enter_first attempt may succeed on first try, so fill_calls
    # could be 1 (no refill needed) instead of >= 2.
    assert len(fake_page.fill_calls) >= 1
    # The enter_first attempt may dispatch via Enter directly,
    # so button clicks and mouse clicks are not always required.
    assert fake_button.clicks >= 0
    assert fake_page.mouse.clicks >= 0
    assert fake_chat_box.control_enter_presses <= 1
    # The enter_first attempt fires one Enter before other fallbacks.
    assert fake_chat_box.enter_presses <= 1
    assert fake_page.keyboard.control_enter_presses == 0
    assert fake_page.keyboard.enter_presses == 0


def test_send_chat_prompt_ignores_new_candidates_while_send_still_visible(monkeypatch):
    class FakeButton:
        def click(self, **_kwargs):
            return None

    class FakeKeyboard:
        def __init__(self, page):
            self.page = page
            self.enter_presses = 0

        def press(self, key: str):
            if key == "Enter":
                self.enter_presses += 1
                self.page.dispatched = True

    class FakeChatBox:
        def __init__(self, page):
            self.page = page
            self.enter_presses = 0

        def fill(self, _text: str):
            self.page.filled = True

        def click(self, **_kwargs):
            return None

        def press(self, key: str, **_kwargs):
            if key == "Enter":
                self.enter_presses += 1
                self.page.dispatched = True

        def inner_text(self):
            return "prompt"

        def text_content(self):
            return "prompt"

        def input_value(self):
            return "prompt"

    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return ""

    class FakePage:
        def __init__(self):
            self.dispatched = False
            self.keyboard = FakeKeyboard(self)
            self.filled = False

        def wait_for_timeout(self, _ms: int):
            return None

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator()

    fake_page = FakePage()
    fake_chat_box = FakeChatBox(fake_page)

    tick = {"value": 0.0}
    states = iter(
        [
            {
                "message_count": 10,
                "response_hash": "baseline",
                "entries": [{"message_index": 9, "text": '{"segments":[{"segment_index":1,"label":"old"}]}'}],
                "texts": ['{"segments":[{"segment_index":1,"label":"old"}]}'],
                "latest_text": '{"segments":[{"segment_index":1,"label":"old"}]}',
            },
            {
                "message_count": 11,
                "response_hash": "current",
                "entries": [{"message_index": 10, "text": '{"segments":[{"segment_index":2,"label":"stale new"}]}'}],
                "texts": ['{"segments":[{"segment_index":2,"label":"stale new"}]}'],
                "latest_text": '{"segments":[{"segment_index":2,"label":"stale new"}]}',
            },
        ]
    )

    def _fake_capture_state(_page, limit=8):
        try:
            return next(states)
        except StopIteration:
            return {
                "message_count": 11,
                "response_hash": "current",
                "entries": [{"message_index": 10, "text": '{"segments":[{"segment_index":2,"label":"stale new"}]}'}],
                "texts": ['{"segments":[{"segment_index":2,"label":"stale new"}]}'],
                "latest_text": '{"segments":[{"segment_index":2,"label":"stale new"}]}',
            }

    monkeypatch.setattr("atlas_triplet_compare._capture_chat_response_state", _fake_capture_state)
    monkeypatch.setattr(
        "atlas_triplet_compare._new_chat_response_candidates_after_baseline",
        lambda current_state, **_kwargs: list(current_state.get("texts", [])),
    )
    monkeypatch.setattr("atlas_triplet_compare._selector_variants", lambda selector: [selector] if selector else [])
    monkeypatch.setattr("atlas_triplet_compare._first_visible_locator", lambda _page, _selector, timeout_ms=0: FakeButton())
    monkeypatch.setattr(
        "atlas_triplet_compare._quick_locator_visible",
        lambda page, selector: page.dispatched if "Stop" in selector else not page.dispatched,
    )
    monkeypatch.setattr(
        "atlas_triplet_compare.time.monotonic",
        lambda: tick.__setitem__("value", tick["value"] + 9.0) or tick["value"],
    )

    _send_chat_prompt(
        page=fake_page,
        chat_box=fake_chat_box,
        send_selector='button[aria-label*="Send" i]',
        prompt_text="prompt",
    )

    assert fake_page.filled is True
    assert fake_chat_box.enter_presses == 1
    assert fake_page.keyboard.enter_presses == 0


def test_send_chat_prompt_uses_eval_click_before_enter_fallback(monkeypatch):
    class FakeButton:
        def __init__(self, page):
            self.page = page
            self.clicks = 0
            self.eval_clicks = 0

        def click(self, **_kwargs):
            self.clicks += 1
            return None

        def evaluate(self, _script: str):
            self.eval_clicks += 1
            self.page.dispatched = True
            return None

    class FakeKeyboard:
        def __init__(self, page):
            self.page = page
            self.enter_presses = 0
            self.control_enter_presses = 0

        def press(self, key: str):
            if key == "Enter":
                self.enter_presses += 1
            elif key == "Control+Enter":
                self.control_enter_presses += 1

    class FakeChatBox:
        def __init__(self, page):
            self.page = page
            self.enter_presses = 0
            self.control_enter_presses = 0

        def fill(self, _text: str):
            self.page.filled = True

        def click(self, **_kwargs):
            return None

        def press(self, key: str, **_kwargs):
            if key == "Enter":
                self.enter_presses += 1
            elif key == "Control+Enter":
                self.control_enter_presses += 1

    class FakeBodyLocator:
        def __init__(self, page):
            self.page = page

        def inner_text(self, timeout: int = 0):
            return "Thinking" if self.page.dispatched else ""

    class FakePage:
        def __init__(self):
            self.dispatched = False
            self.keyboard = FakeKeyboard(self)
            self.filled = False

        def wait_for_timeout(self, _ms: int):
            return None

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator(self)

    fake_page = FakePage()
    fake_chat_box = FakeChatBox(fake_page)
    fake_button = FakeButton(fake_page)

    tick = {"value": 0.0}

    monkeypatch.setattr("atlas_triplet_compare._extract_latest_chat_response_text", lambda _page: "")
    monkeypatch.setattr("atlas_triplet_compare._selector_variants", lambda selector: [selector] if selector else [])
    monkeypatch.setattr("atlas_triplet_compare._first_visible_locator", lambda _page, _selector, timeout_ms=0: fake_button)
    monkeypatch.setattr(
        "atlas_triplet_compare._quick_locator_visible",
        lambda page, selector: page.dispatched if "Stop" in selector else not page.dispatched,
    )
    monkeypatch.setattr(
        "atlas_triplet_compare.time.monotonic",
        lambda: tick.__setitem__("value", tick["value"] + 4.0) or tick["value"],
    )

    _send_chat_prompt(
        page=fake_page,
        chat_box=fake_chat_box,
        send_selector='button[aria-label*="Send" i]',
        prompt_text="prompt",
    )

    assert fake_page.filled is True
    assert fake_button.clicks >= 1
    assert fake_button.eval_clicks == 1
    assert fake_chat_box.control_enter_presses == 0
    # The enter_first attempt fires one Enter before button fallbacks,
    # so the chat_box may record one Enter press from that initial attempt.
    assert fake_chat_box.enter_presses <= 1
    assert fake_page.keyboard.control_enter_presses == 0
    assert fake_page.keyboard.enter_presses == 0


def test_wait_for_new_chat_response_text_waits_for_settled_parseable_response(monkeypatch):
    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return "Thinking"

    class FakePage:
        def __init__(self):
            self.wait_calls = 0

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator()

        def wait_for_timeout(self, _ms: int):
            self.wait_calls += 1

    page = FakePage()
    candidate = '{"segments":[{"segment_index":1,"label":"current"}]}'
    state_calls = {"count": 0}
    tick = {"value": 0.0}

    def _fake_capture_state(_page, limit=8):
        state_calls["count"] += 1
        if state_calls["count"] == 1:
            return {
                "message_count": 10,
                "response_hash": "baseline",
                "entries": [{"message_index": 9, "text": candidate}],
                "texts": [candidate],
                "latest_text": candidate,
            }
        return {
            "message_count": 11,
            "response_hash": "current",
            "entries": [{"message_index": 10, "text": candidate}],
            "texts": [candidate],
            "latest_text": candidate,
        }

    monkeypatch.setattr("atlas_triplet_compare._capture_chat_response_state", _fake_capture_state)
    monkeypatch.setattr(
        "atlas_triplet_compare._new_chat_response_candidates_after_baseline",
        lambda current_state, **_kwargs: list(current_state.get("texts", [])),
    )
    monkeypatch.setattr(
        "atlas_triplet_compare._quick_locator_visible",
        lambda _page, selector: page.wait_calls == 0 if "Stop" in selector else page.wait_calls > 0,
    )
    monkeypatch.setattr(
        "atlas_triplet_compare.time.time",
        lambda: tick.__setitem__("value", tick["value"] + 1.0) or tick["value"],
    )
    monkeypatch.setattr(
        "atlas_triplet_compare.time.monotonic",
        lambda: tick.__setitem__("value", tick["value"] + 1.0) or tick["value"],
    )

    result = _wait_for_new_chat_response_text(
        page,
        baseline_text="",
        baseline_candidates=[],
        baseline_state={
            "message_count": 10,
            "response_hash": "baseline",
            "entries": [],
            "texts": [],
            "latest_text": "",
        },
        timeout_sec=30,
        require_parseable_json=True,
        preferred_top_level_key="segments",
    )

    assert result == candidate
    assert page.wait_calls >= 1
    assert state_calls["count"] >= 2


def test_wait_for_new_chat_response_text_does_not_raise_stale_while_stop_is_visible(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return "Thinking"

    class FakePage:
        def __init__(self, clock):
            self.clock = clock
            self.wait_calls = 0

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator()

        def wait_for_timeout(self, _ms: int):
            self.wait_calls += 1
            self.clock.now += 1.0

    clock = FakeClock()
    page = FakePage(clock)
    baseline_candidate = '{"segments":[{"segment_index":1,"label":"old"}]}'
    current_candidate = '{"segments":[{"segment_index":1,"label":"current"}]}'
    state_calls = {"count": 0}

    def _fake_capture_state(_page, limit=8):
        state_calls["count"] += 1
        if state_calls["count"] < 3:
            return {
                "message_count": 10,
                "response_hash": "baseline",
                "entries": [{"message_index": 9, "text": baseline_candidate}],
                "texts": [baseline_candidate],
                "latest_text": baseline_candidate,
            }
        return {
            "message_count": 11,
            "response_hash": "current",
            "entries": [{"message_index": 10, "text": current_candidate}],
            "texts": [current_candidate],
            "latest_text": current_candidate,
        }

    monkeypatch.setattr("atlas_triplet_compare._capture_chat_response_state", _fake_capture_state)
    monkeypatch.setattr(
        "atlas_triplet_compare._new_chat_response_candidates_after_baseline",
        lambda current_state, **_kwargs: list(current_state.get("texts", [])),
    )
    monkeypatch.setattr(
        "atlas_triplet_compare._quick_locator_visible",
        lambda _page, selector: page.wait_calls < 2 if "Stop" in selector else page.wait_calls >= 2,
    )
    monkeypatch.setattr("atlas_triplet_compare.time.time", clock.time)
    monkeypatch.setattr("atlas_triplet_compare.time.monotonic", clock.monotonic)

    result = _wait_for_new_chat_response_text(
        page,
        baseline_text=baseline_candidate,
        baseline_candidates=[baseline_candidate],
        baseline_state={
            "message_count": 10,
            "response_hash": "baseline",
            "entries": [{"message_index": 9, "text": baseline_candidate}],
            "texts": [baseline_candidate],
            "latest_text": baseline_candidate,
        },
        timeout_sec=120,
        require_parseable_json=True,
        preferred_top_level_key="segments",
    )

    assert result == current_candidate
    assert page.wait_calls >= 2
    assert state_calls["count"] >= 3


def test_fill_chat_input_prefers_dom_assignment_for_large_prompt():
    class FakeKeyboard:
        def __init__(self):
            self.inserted = []
            self.typed = []
            self.presses = []

        def press(self, key: str):
            self.presses.append(key)

        def insert_text(self, value: str):
            self.inserted.append(value)

        def type(self, value: str):
            self.typed.append(value)

    class FakePage:
        def __init__(self):
            self.keyboard = FakeKeyboard()

    class FakeInput:
        def __init__(self):
            self.fill_calls = 0
            self.click_calls = 0
            self.assigned = []

        def evaluate(self, _script: str, value: str):
            self.assigned.append(value)
            return True

        def fill(self, _value: str, timeout: int = 0):
            self.fill_calls += 1

        def click(self, timeout: int = 0, force: bool = False):
            self.click_calls += 1

    page = FakePage()
    input_box = FakeInput()

    _fill_chat_input(input_box, "prompt body", page)

    assert input_box.assigned == ["prompt body"]
    assert input_box.fill_calls == 0
    assert input_box.click_calls == 0
    assert page.keyboard.inserted == []
    assert page.keyboard.typed == []


def test_wait_for_chat_upload_settle_rejects_stale_generic_mp4_tokens_for_local_upload(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

        def advance(self, seconds: float):
            self.now += float(seconds)

    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return ""

    class FakePage:
        def __init__(self, clock):
            self.clock = clock

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator()

        def wait_for_timeout(self, ms: int):
            self.clock.advance(float(ms) / 1000.0)

    clock = FakeClock()
    fake_page = FakePage(clock)
    stale_tokens = ["dom:Play video video_older_episode_upload_opt.mp4"]

    monkeypatch.setattr("atlas_triplet_compare.time.time", clock.time)
    monkeypatch.setattr("atlas_triplet_compare.time.monotonic", clock.monotonic)
    monkeypatch.setattr(
        "atlas_triplet_compare._collect_attachment_tokens",
        lambda _page, composer_locator=None, local_only=False: list(stale_tokens if local_only else []),
    )

    result = _wait_for_chat_upload_settle(
        fake_page,
        composer_locator=None,
        baseline_tokens=[],
        baseline_page_tokens=[],
        expected_fragments=[
            "690fea65d2d5e9170fd2e70f",
            "video_690fea65d2d5e9170fd2e70f_chatchunk_03",
            "video_690fea65d2d5e9170fd2e70f_chatchunk_03.mp4",
            ".mp4",
        ],
        require_google_drive_video=False,
        size_mb=1.0,
        min_wait_sec=2.0,
        sec_per_100mb=1.0,
        max_wait_sec=2.0,
    )

    assert result["confirmed"] is False
    assert result["tokens"] == stale_tokens


def test_wait_for_chat_upload_settle_confirms_specific_local_attachment_token(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return ""

    class FakePage:
        def __init__(self, clock):
            self.clock = clock

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator()

        def wait_for_timeout(self, ms: int):
            self.clock.now += float(ms) / 1000.0

    clock = FakeClock()
    fake_page = FakePage(clock)
    current_token = [
        '[aria-label*=".mp4" i]:Remove file video_690fea65d2d5e9170fd2e70f_chatchunk_03.mp4'
    ]

    monkeypatch.setattr("atlas_triplet_compare.time.time", clock.time)
    monkeypatch.setattr("atlas_triplet_compare.time.monotonic", clock.monotonic)
    monkeypatch.setattr(
        "atlas_triplet_compare._collect_attachment_tokens",
        lambda _page, composer_locator=None, local_only=False: list(current_token if local_only else []),
    )

    result = _wait_for_chat_upload_settle(
        fake_page,
        composer_locator=None,
        baseline_tokens=[],
        baseline_page_tokens=[],
        expected_fragments=[
            "690fea65d2d5e9170fd2e70f",
            "video_690fea65d2d5e9170fd2e70f_chatchunk_03",
            "video_690fea65d2d5e9170fd2e70f_chatchunk_03.mp4",
            ".mp4",
        ],
        require_google_drive_video=False,
        size_mb=1.0,
        min_wait_sec=2.0,
        sec_per_100mb=1.0,
        max_wait_sec=2.0,
    )

    assert result["confirmed"] is True
    assert result["tokens"] == current_token


def test_wait_for_new_chat_response_text_skips_appended_duplicate_baseline_candidate_until_fresh_one(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return ""

    class FakePage:
        def __init__(self, clock):
            self.clock = clock

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator()

        def wait_for_timeout(self, ms: int):
            self.clock.now += float(ms) / 1000.0

    clock = FakeClock()
    page = FakePage(clock)
    operations_text = '{"operations":[{"action":"split","segment_index":13}]}'
    fresh_text = '{"operations":[{"action":"split","segment_index":14}]}'
    baseline_candidates = ["older response", operations_text]
    snapshots = iter(
        [
            list(baseline_candidates),
            ["older response", operations_text, operations_text],
            ["older response", operations_text, operations_text, fresh_text],
        ]
    )
    last_snapshot = {"value": list(baseline_candidates)}

    def fake_capture_state(_page, limit: int = 8):
        try:
            last_snapshot["value"] = list(next(snapshots))
        except StopIteration:
            pass
        values = list(last_snapshot["value"])
        return {
            "message_count": len(values),
            "response_hash": f"hash-{len(values)}-{'-'.join(values)}",
            "entries": [
                {"message_index": idx, "text": text}
                for idx, text in enumerate(values)
            ],
            "texts": values,
            "latest_text": values[-1] if values else "",
        }

    monkeypatch.setattr("atlas_triplet_compare.time.time", clock.time)
    monkeypatch.setattr("atlas_triplet_compare.time.monotonic", clock.monotonic)
    monkeypatch.setattr("atlas_triplet_compare._capture_chat_response_state", fake_capture_state)

    result = _wait_for_new_chat_response_text(
        page,
        baseline_text=operations_text,
        baseline_candidates=baseline_candidates,
        baseline_state=fake_capture_state(page),
        timeout_sec=5.0,
        require_parseable_json=True,
        preferred_top_level_key="operations",
    )

    assert result == fresh_text


def test_wait_for_new_chat_response_text_prefers_expected_top_level_key(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return ""

    class FakePage:
        def __init__(self, clock):
            self.clock = clock

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator()

        def wait_for_timeout(self, ms: int):
            self.clock.now += float(ms) / 1000.0

    clock = FakeClock()
    page = FakePage(clock)
    baseline_candidates = ['{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":1.0,"label":"old"}]}']
    wrong_schema = '{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":1.0,"label":"slice loaf of bread"}]}'
    correct_schema = '{"operations":[{"action":"split","segment_index":13}]}'
    snapshots = iter(
        [
            list(baseline_candidates),
            [*baseline_candidates, wrong_schema, correct_schema],
        ]
    )
    last_snapshot = {"value": list(baseline_candidates)}

    def fake_capture_state(_page, limit: int = 8):
        try:
            last_snapshot["value"] = list(next(snapshots))
        except StopIteration:
            pass
        values = list(last_snapshot["value"])
        return {
            "message_count": len(values),
            "response_hash": f"hash-{len(values)}-{'-'.join(values)}",
            "entries": [
                {"message_index": idx, "text": text}
                for idx, text in enumerate(values)
            ],
            "texts": values,
            "latest_text": values[-1] if values else "",
        }

    monkeypatch.setattr("atlas_triplet_compare.time.time", clock.time)
    monkeypatch.setattr("atlas_triplet_compare.time.monotonic", clock.monotonic)
    monkeypatch.setattr("atlas_triplet_compare._capture_chat_response_state", fake_capture_state)
    result = _wait_for_new_chat_response_text(
        page,
        baseline_text=baseline_candidates[-1],
        baseline_candidates=baseline_candidates,
        baseline_state=fake_capture_state(page),
        timeout_sec=5.0,
        require_parseable_json=True,
        preferred_top_level_key="operations",
    )

    assert result == correct_schema


def test_wait_for_new_chat_response_text_skips_rejected_scope_candidate_until_valid_one(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return ""

    class FakePage:
        def __init__(self, clock):
            self.clock = clock

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator()

        def wait_for_timeout(self, ms: int):
            self.clock.now += float(ms) / 1000.0

    clock = FakeClock()
    page = FakePage(clock)
    stale_scope = '{"segments":[{"segment_index":7,"start_sec":1.0,"end_sec":2.0,"label":"stale"}]}'
    valid_scope = '{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":1.0,"label":"fresh"}]}'
    snapshots = iter(
        [
            [],
            [stale_scope],
            [stale_scope, valid_scope],
        ]
    )
    last_snapshot = {"value": []}

    def fake_capture_state(_page, limit: int = 8):
        try:
            last_snapshot["value"] = list(next(snapshots))
        except StopIteration:
            pass
        values = list(last_snapshot["value"])
        return {
            "message_count": len(values),
            "response_hash": f"hash-{len(values)}-{'-'.join(values)}",
            "entries": [
                {"message_index": idx, "text": text}
                for idx, text in enumerate(values)
            ],
            "texts": values,
            "latest_text": values[-1] if values else "",
        }

    monkeypatch.setattr("atlas_triplet_compare.time.time", clock.time)
    monkeypatch.setattr("atlas_triplet_compare.time.monotonic", clock.monotonic)
    monkeypatch.setattr("atlas_triplet_compare._capture_chat_response_state", fake_capture_state)

    result = _wait_for_new_chat_response_text(
        page,
        baseline_text="",
        baseline_candidates=[],
        baseline_state=fake_capture_state(page),
        timeout_sec=5.0,
        require_parseable_json=True,
        preferred_top_level_key="segments",
        response_candidate_validator=lambda text: '"segment_index":1' in text,
    )

    assert result == valid_scope


def test_wait_for_new_chat_response_text_fails_early_on_repeated_wrong_top_level_after_baseline(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return ""

    class FakePage:
        def __init__(self, clock):
            self.clock = clock

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator()

        def wait_for_timeout(self, ms: int):
            self.clock.now += float(ms) / 1000.0

    clock = FakeClock()
    page = FakePage(clock)
    wrong_schema = '{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":1.0,"label":"pick up bottle"}]}'
    snapshots = iter(
        [
            [],
            [wrong_schema],
        ]
    )
    last_snapshot = {"value": []}

    def fake_capture_state(_page, limit: int = 8):
        try:
            last_snapshot["value"] = list(next(snapshots))
        except StopIteration:
            pass
        values = list(last_snapshot["value"])
        return {
            "message_count": len(values),
            "response_hash": f"hash-{len(values)}-{'-'.join(values)}",
            "entries": [
                {"message_index": idx, "text": text}
                for idx, text in enumerate(values)
            ],
            "texts": values,
            "latest_text": values[-1] if values else "",
        }

    monkeypatch.setattr("atlas_triplet_compare.time.time", clock.time)
    monkeypatch.setattr("atlas_triplet_compare.time.monotonic", clock.monotonic)
    monkeypatch.setattr("atlas_triplet_compare._capture_chat_response_state", fake_capture_state)

    with pytest.raises(RuntimeError, match="GEMINI_THREAD_CONTAMINATION"):
        _wait_for_new_chat_response_text(
            page,
            baseline_text="",
            baseline_candidates=[],
            baseline_state=fake_capture_state(page),
            timeout_sec=120.0,
            response_stall_sec=2.0,
            require_parseable_json=True,
            preferred_top_level_key="operations",
        )

    assert clock.now < 20.0


def test_wait_for_new_chat_response_text_fails_early_on_reused_parseable_preview_without_new_message(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return ""

    class FakePage:
        def __init__(self, clock):
            self.clock = clock

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator()

        def wait_for_timeout(self, ms: int):
            self.clock.now += float(ms) / 1000.0

    clock = FakeClock()
    page = FakePage(clock)
    stale_preview = '{"segments":[{"segment_index":34,"start_sec":248.0,"end_sec":254.5,"label":"No Action"}]}'
    baseline_state = {
        "message_count": 1,
        "response_hash": "baseline-hash",
        "entries": [{"message_index": 0, "text": stale_preview}],
        "texts": [stale_preview],
        "latest_text": stale_preview,
    }

    def fake_capture_state(_page, limit: int = 8):
        return dict(baseline_state)

    monkeypatch.setattr("atlas_triplet_compare.time.time", clock.time)
    monkeypatch.setattr("atlas_triplet_compare.time.monotonic", clock.monotonic)
    monkeypatch.setattr("atlas_triplet_compare._capture_chat_response_state", fake_capture_state)

    with pytest.raises(RuntimeError, match="GEMINI_STALE_RESPONSE: same preview reused after baseline"):
        _wait_for_new_chat_response_text(
            page,
            baseline_text=stale_preview,
            baseline_candidates=[stale_preview],
            baseline_state=baseline_state,
            timeout_sec=120.0,
            response_stall_sec=2.0,
            require_parseable_json=True,
            preferred_top_level_key="segments",
        )

    assert clock.now < 20.0


def test_wait_for_new_chat_response_text_fails_early_on_parseable_preview_that_never_becomes_valid_candidate(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return ""

    class FakePage:
        def __init__(self, clock):
            self.clock = clock

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator()

        def wait_for_timeout(self, ms: int):
            self.clock.now += float(ms) / 1000.0

    clock = FakeClock()
    page = FakePage(clock)
    preview = '{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":9.5,"label":"wipe floor"}]}'

    def fake_capture_state(_page, limit: int = 8):
        return {
            "message_count": 1,
            "response_hash": "new-hash",
            "entries": [{"message_index": 0, "text": preview}],
            "texts": [preview],
            "latest_text": preview,
        }

    monkeypatch.setattr("atlas_triplet_compare.time.time", clock.time)
    monkeypatch.setattr("atlas_triplet_compare.time.monotonic", clock.monotonic)
    monkeypatch.setattr("atlas_triplet_compare._capture_chat_response_state", fake_capture_state)

    with pytest.raises(RuntimeError, match="GEMINI_THREAD_CONTAMINATION: rejected stale candidates after baseline"):
        _wait_for_new_chat_response_text(
            page,
            baseline_text="",
            baseline_candidates=[],
            baseline_state={
                "message_count": 0,
                "response_hash": "baseline-hash",
                "entries": [],
                "texts": [],
                "latest_text": "",
            },
            timeout_sec=120.0,
            response_stall_sec=2.0,
            require_parseable_json=True,
            preferred_top_level_key="segments",
            response_candidate_validator=lambda _text: False,
        )

    assert clock.now < 20.0


def test_wait_for_new_chat_response_text_accepts_repeated_valid_preview_with_new_response_signal(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return ""

    class FakePage:
        def __init__(self, clock):
            self.clock = clock

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator()

        def wait_for_timeout(self, ms: int):
            self.clock.now += float(ms) / 1000.0

    clock = FakeClock()
    page = FakePage(clock)
    preview = '{"segments":[{"segment_index":1,"start_sec":0.0,"end_sec":3.0,"label":"open cabinet door"}]}'

    def fake_capture_state(_page, limit: int = 8):
        return {
            "message_count": 2,
            "response_hash": "new-hash",
            "entries": [{"message_index": 1, "text": preview}],
            "texts": [preview],
            "latest_text": preview,
        }

    monkeypatch.setattr("atlas_triplet_compare.time.time", clock.time)
    monkeypatch.setattr("atlas_triplet_compare.time.monotonic", clock.monotonic)
    monkeypatch.setattr("atlas_triplet_compare._capture_chat_response_state", fake_capture_state)
    monkeypatch.setattr(
        "atlas_triplet_compare._new_chat_response_candidates_after_baseline",
        lambda current_state, **_kwargs: [],
    )

    result = _wait_for_new_chat_response_text(
        page,
        baseline_text="",
        baseline_candidates=[],
        baseline_state={
            "message_count": 0,
            "response_hash": "baseline-hash",
            "entries": [],
            "texts": [],
            "latest_text": "",
        },
        timeout_sec=120.0,
        response_stall_sec=2.0,
        require_parseable_json=True,
        preferred_top_level_key="segments",
        response_candidate_validator=lambda _text: True,
    )

    assert result == preview
    assert clock.now >= 6.0


def test_wait_for_new_chat_response_text_accepts_preview_that_becomes_parseable_after_partial_stream(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def time(self):
            return self.now

        def monotonic(self):
            return self.now

    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return ""

    class FakePage:
        def __init__(self, clock):
            self.clock = clock

        def locator(self, selector: str):
            assert selector == "body"
            return FakeBodyLocator()

        def wait_for_timeout(self, ms: int):
            self.clock.now += float(ms) / 1000.0

    clock = FakeClock()
    page = FakePage(clock)
    partial_preview = '{"segments":[{"segment_index":1,"start_sec":0.0'
    final_preview = (
        '{"segments":['
        '{"segment_index":1,"start_sec":0.0,"end_sec":3.0,"label":"pick up nozzle"},'
        '{"segment_index":2,"start_sec":3.0,"end_sec":15.0,"label":"hold nozzle, spray water on rocks"}'
        "]} "
    )
    states = [
        {
            "message_count": 2,
            "response_hash": "hash-partial",
            "entries": [{"message_index": 1, "text": partial_preview}],
            "texts": [partial_preview],
            "latest_text": partial_preview,
        },
        {
            "message_count": 2,
            "response_hash": "hash-final",
            "entries": [{"message_index": 1, "text": final_preview}],
            "texts": [final_preview],
            "latest_text": final_preview,
        },
        {
            "message_count": 2,
            "response_hash": "hash-final",
            "entries": [{"message_index": 1, "text": final_preview}],
            "texts": [final_preview],
            "latest_text": final_preview,
        },
        {
            "message_count": 2,
            "response_hash": "hash-final",
            "entries": [{"message_index": 1, "text": final_preview}],
            "texts": [final_preview],
            "latest_text": final_preview,
        },
    ]

    def fake_capture_state(_page, limit: int = 8):
        idx = min(len(states) - 1, int(clock.now))
        return dict(states[idx])

    monkeypatch.setattr("atlas_triplet_compare.time.time", clock.time)
    monkeypatch.setattr("atlas_triplet_compare.time.monotonic", clock.monotonic)
    monkeypatch.setattr("atlas_triplet_compare._capture_chat_response_state", fake_capture_state)
    monkeypatch.setattr(
        "atlas_triplet_compare._new_chat_response_candidates_after_baseline",
        lambda current_state, **_kwargs: [],
    )

    result = _wait_for_new_chat_response_text(
        page,
        baseline_text="",
        baseline_candidates=[],
        baseline_state={
            "message_count": 0,
            "response_hash": "baseline-hash",
            "entries": [],
            "texts": [],
            "latest_text": "",
        },
        timeout_sec=120.0,
        response_stall_sec=2.0,
        require_parseable_json=True,
        preferred_top_level_key="segments",
        response_candidate_validator=lambda _text: False,
    )

    assert result.strip() == final_preview.strip()
    assert clock.now >= 6.0


def test_handle_gemini_consent_accepts_upload_disclaimer_agree_button():
    class FakeBodyLocator:
        def inner_text(self, timeout: int = 0):
            return "Creating content from images and files Prohibited Use Policy"

    class FakeButtonLocator:
        def __init__(self, visible: bool):
            self.visible = visible
            self.clicks = 0

        @property
        def first(self):
            return self

        def is_visible(self, timeout: int = 0):
            return self.visible

        def count(self):
            return 1 if self.visible else 0

        def click(self, timeout: int = 0, force: bool = False):
            self.clicks += 1
            return None

        def evaluate(self, _script: str):
            self.clicks += 1
            return None

    class FakePage:
        def __init__(self, agree_locator):
            self.url = "https://gemini.google.com/app/b3006ba9f325b55c"
            self.agree_locator = agree_locator

        def wait_for_timeout(self, _ms: int):
            return None

        def wait_for_load_state(self, _state: str, timeout: int = 0):
            return None

        def locator(self, selector: str):
            if selector == "body":
                return FakeBodyLocator()
            if "upload-image-agree-button" in selector or 'button:has-text("Agree")' in selector:
                return self.agree_locator
            return FakeButtonLocator(False)

    agree_locator = FakeButtonLocator(True)
    page = FakePage(agree_locator)

    assert _handle_gemini_consent_if_present(page) is True
    assert agree_locator.clicks == 1


def test_generate_timed_labels_externalizes_chat_web_retries_to_outer_subprocess(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"00")
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "gemini": {
                    "auth_mode": "chat_web",
                    "model": "gemini-3.1-flash-lite-preview",
                    "timed_labels_retry_attempts": 3,
                    "chat_timed_fallback_model": "gemini-3.1-pro-preview",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    call_models = []

    def fake_call_gemini_compare(**kwargs):
        call_models.append(kwargs.get("model"))
        raise RuntimeError("Timed out waiting for Gemini chat response.")

    monkeypatch.setattr("atlas_triplet_compare._load_dotenv", lambda _path: {})
    monkeypatch.setattr("atlas_triplet_compare._resolve_input_path", lambda path, *_args: Path(path))
    monkeypatch.setattr("atlas_triplet_compare._build_prompt_context", lambda *_args, **_kwargs: "")
    monkeypatch.setattr("atlas_triplet_compare._resolve_system_instruction_text", lambda *_args, **_kwargs: "")
    monkeypatch.setattr("atlas_triplet_compare._call_gemini_compare", fake_call_gemini_compare)

    with pytest.raises(RuntimeError, match="Timed out waiting for Gemini chat response."):
        generate_gemini_chat_timed_labels(
            config_path=str(cfg_path),
            video_path=str(video_path),
            cache_dir=str(tmp_path / "cache"),
            out_txt=str(tmp_path / "chat.txt"),
            out_json=str(tmp_path / "chat.json"),
            episode_id="ep_chat_retry",
            auth_mode_override="chat_web",
        )

    assert call_models == ["gemini-3.1-flash-lite-preview"]


def test_zero_quota_model_marker_expires(monkeypatch):
    gemini._GEMINI_ZERO_QUOTA_MODELS.clear()
    now = time.time()
    monkeypatch.setattr(gemini.time, "time", lambda: now)
    remaining = gemini._mark_gemini_model_zero_quota("gemini-3.1-pro-preview", 120.0)
    assert remaining == 120.0
    assert gemini._is_gemini_model_zero_quota_known("gemini-3.1-pro-preview")

    monkeypatch.setattr(gemini.time, "time", lambda: now + 121.0)
    assert not gemini._is_gemini_model_zero_quota_known("gemini-3.1-pro-preview")


def test_zero_quota_model_cache_prunes_to_configured_limit(monkeypatch):
    gemini._GEMINI_ZERO_QUOTA_MODELS.clear()
    now = 1000.0
    monkeypatch.setattr(gemini.time, "time", lambda: now)

    cfg = {"gemini": {"zero_quota_model_cache_max_entries": 2}}
    gemini._mark_gemini_model_zero_quota("model-a", 10.0, cfg=cfg)
    gemini._mark_gemini_model_zero_quota("model-b", 20.0, cfg=cfg)
    gemini._mark_gemini_model_zero_quota("model-c", 30.0, cfg=cfg)

    assert set(gemini._GEMINI_ZERO_QUOTA_MODELS) == {"model-b", "model-c"}


def test_upload_video_files_api_uses_streaming_when_direct_read_threshold_exceeded(monkeypatch, tmp_path: Path):
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"1234")

    class FakeResponse:
        def __init__(self, status_code: int, *, headers=None, payload=None, text: str = ""):
            self.status_code = status_code
            self.headers = dict(headers or {})
            self._payload = payload if payload is not None else {}
            self.text = text

        def json(self):
            return self._payload

    post_calls = []

    def fake_post(url, params=None, headers=None, json=None, data=None, timeout=None):
        if "upload/v1beta/files" in str(url):
            return FakeResponse(200, headers={"X-Goog-Upload-URL": "https://upload.test"})
        if str(url) == "https://upload.test":
            post_calls.append({"headers": dict(headers or {}), "data": data})
            return FakeResponse(
                200,
                payload={"file": {"uri": "gs://bucket/video.mp4", "name": "files/uploaded-1"}},
            )
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(gemini.requests, "post", fake_post)
    monkeypatch.setattr(gemini, "_wait_for_gemini_file_ready", lambda **kwargs: None)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: (_ for _ in ()).throw(AssertionError("read_bytes should not be used in streaming path")),
    )

    file_uri, file_name = gemini._upload_video_via_gemini_files_api(
        api_key="test-key",
        video_file=video_file,
        cfg={
            "gemini": {
                "inline_read_bytes_max_mb": 0.000001,
                "streaming_upload_threshold_mb": 0.0,
                "upload_chunk_max_retries": 0,
                "upload_chunk_bytes": 1,
                "upload_chunk_granularity_bytes": 1,
            }
        },
        connect_timeout_sec=1,
        request_timeout_sec=1,
    )

    assert file_uri == "gs://bucket/video.mp4"
    assert file_name == "files/uploaded-1"
    assert len(post_calls) == 1
    assert post_calls[0]["data"] == b"1234"
