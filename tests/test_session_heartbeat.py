from src.infra import session_heartbeat


class _FakePage:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeContext:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeBrowser:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeLocator:
    def __init__(self, *, visible: bool = False) -> None:
        self.visible = visible
        self.clicks = 0

    @property
    def first(self):
        return self

    def is_visible(self, timeout=None):
        return self.visible

    def click(self):
        self.clicks += 1


class _FakeChooserPage:
    def __init__(self, mapping: dict[str, _FakeLocator]) -> None:
        self.url = "https://accounts.google.com/v3/signin/accountchooser"
        self._mapping = mapping
        self.wait_calls: list[int] = []

    def locator(self, selector: str):
        return self._mapping.setdefault(selector, _FakeLocator(visible=False))

    def wait_for_timeout(self, timeout_ms: int) -> None:
        self.wait_calls.append(timeout_ms)


def test_cleanup_probe_keeps_shared_cdp_browser_open() -> None:
    browser = _FakeBrowser()
    context = _FakeContext()
    page = _FakePage()

    session_heartbeat._cleanup_probe(browser, context, page, owns_context=False)

    assert page.closed is True
    assert context.closed is False
    assert browser.closed is False


def test_cleanup_probe_closes_owned_context_only() -> None:
    browser = _FakeBrowser()
    context = _FakeContext()
    page = _FakePage()

    session_heartbeat._cleanup_probe(browser, context, page, owns_context=True)

    assert page.closed is True
    assert context.closed is True
    assert browser.closed is False


def test_handle_google_account_chooser_clicks_matching_account() -> None:
    account_locator = _FakeLocator(visible=True)
    page = _FakeChooserPage(
        {
            "input[type='email']": _FakeLocator(visible=False),
            "text='user@example.com'": account_locator,
            "text='Use another account'": _FakeLocator(visible=True),
        }
    )

    clicked = session_heartbeat._handle_google_account_chooser(page, "user@example.com")

    assert clicked is True
    assert account_locator.clicks == 1
    assert page.wait_calls == [1200]


def test_handle_google_account_chooser_falls_back_to_use_another_account() -> None:
    fallback_locator = _FakeLocator(visible=True)
    page = _FakeChooserPage(
        {
            "input[type='email']": _FakeLocator(visible=False),
            "text='user@example.com'": _FakeLocator(visible=False),
            "text='Use another account'": fallback_locator,
        }
    )

    clicked = session_heartbeat._handle_google_account_chooser(page, "user@example.com")

    assert clicked is True
    assert fallback_locator.clicks == 1
    assert page.wait_calls == [1200]
