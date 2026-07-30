from __future__ import annotations

from types import SimpleNamespace

from scripts import monitor_v22_batch_status as monitor


def test_service_snapshot_includes_wilor_28004(monkeypatch) -> None:
    requested_urls: list[str] = []

    class Response:
        status = 200

        def read(self, _size: int) -> bytes:
            return b"ok"

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_urlopen(url: str, *, timeout: float) -> Response:
        assert timeout == 3.0
        requested_urls.append(url)
        return Response()

    monkeypatch.setattr(monitor.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(monitor.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""))

    snapshot = monitor.service_snapshot()

    assert set(snapshot["services"]) == {"28000", "28001", "28002", "28003", "28004"}
    assert "http://127.0.0.1:28004/-/healthz" in requested_urls
