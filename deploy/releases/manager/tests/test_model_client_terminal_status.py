from __future__ import annotations

import sys

from services import model_client


def test_model_client_accepts_partial_camera_coverage_as_terminal_success(monkeypatch, tmp_path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        model_client,
        "upload_and_infer",
        lambda *_args, **_kwargs: {"status": "completed_with_partial_camera_coverage", "output_artifacts": {}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["model_client.py", "--request-json", str(request_path), "--endpoint", "http://resident"],
    )

    model_client.main()
