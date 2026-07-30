"""CLI-contract tests for the real-workload DROID client."""
from __future__ import annotations

from benchmarks.ray_serve import benchmark_droid_continuous as continuous


def test_parser_binds_preserved_payload_count_to_full_video_frames() -> None:
    args = continuous.parse_args([
        "--endpoint", "http://127.0.0.1:32000",
        "--runtime-identity", "/tmp/identity.json",
        "--model-revision", "droid-v1",
        "--preserved-payload-manifest", "/tmp/production720.json",
        "--run-root", "/tmp/rw2-s8",
        "--sessions", "8",
    ])

    assert args.frames == 720
    assert args.payload_count == 720
