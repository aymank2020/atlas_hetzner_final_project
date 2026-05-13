from pathlib import Path

from src.solver import video


class _FakePage:
    def __init__(self) -> None:
        self.url = "https://audit.atlascapture.io/tasks/room/normal/label/example"

    def evaluate(self, script):
        return "Fake UA"


class _FakeContext:
    def __init__(self) -> None:
        self.request = object()

    def cookies(self, *args, **kwargs):
        return []


class _FakeResponse:
    def __init__(self, status_code: int, headers: dict[str, str], chunks: list[bytes]) -> None:
        self.status_code = status_code
        self.headers = headers
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        raise RuntimeError(f"http status {self.status_code}")

    def iter_content(self, chunk_size=0):
        return iter(self._chunks)


class _FakeSession:
    def __init__(self) -> None:
        self.cookies = type(
            "_Cookies",
            (),
            {"set": staticmethod(lambda *args, **kwargs: None)},
        )()
        self.calls: list[dict[str, str]] = []

    def get(self, url, headers=None, timeout=None, stream=None, allow_redirects=None):
        headers = dict(headers or {})
        self.calls.append(headers)
        range_header = headers.get("Range")
        if not range_header:
            return _FakeResponse(
                200,
                {"Content-Length": "20"},
                [b"1234567890"],
            )
        return _FakeResponse(
            206,
            {"Content-Range": "bytes 10-19/20"},
            [],
        )


def test_download_video_from_page_context_uses_playwright_fallback_after_stalled_partial(
    monkeypatch,
    tmp_path,
):
    fake_session = _FakeSession()
    fallback_calls = []
    sleep_calls = []

    monkeypatch.setattr(video.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(video.time, "sleep", lambda delay: sleep_calls.append(delay))
    monkeypatch.setattr(
        video,
        "_download_video_via_playwright_request",
        lambda **kwargs: fallback_calls.append(kwargs) or kwargs["out_path"],
    )

    out_path = tmp_path / "episode.mp4"
    result = video._download_video_from_page_context(
        page=_FakePage(),
        context=_FakeContext(),
        video_url="https://example.com/video.mp4",
        out_path=out_path,
        timeout_sec=30,
        cfg={
            "gemini": {
                "video_download_retries": 8,
                "video_download_retry_base_sec": 0.1,
                "video_download_stalled_partial_retry_limit": 2,
                "video_download_use_playwright_fallback": True,
            }
        },
    )

    assert result == out_path
    assert len(fallback_calls) == 1
    assert len(fake_session.calls) == 3
    assert fake_session.calls[1]["Range"] == "bytes=10-"
    assert fake_session.calls[2]["Range"] == "bytes=10-"
    assert sleep_calls == [0.2, 0.4]
