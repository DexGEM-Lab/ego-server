from __future__ import annotations

from scripts.run_v22_cosmos_captioning_source import parse_args


def test_cosmos_individual_fallback_is_opt_in() -> None:
    args = parse_args(["--video", "/tmp/input.mp4", "--run-root", "/tmp/run", "--case-id", "case_001"])
    assert args.fallback_individual is False

    enabled = parse_args(["--video", "/tmp/input.mp4", "--run-root", "/tmp/run", "--case-id", "case_001", "--fallback-individual"])
    assert enabled.fallback_individual is True
