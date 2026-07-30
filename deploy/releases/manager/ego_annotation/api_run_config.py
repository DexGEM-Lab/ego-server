"""Frozen single-video API run fixture and CPU preflight contract.

This module defines the next run without launching it. The runtime driver must
supply observations from the A800 filesystem/video probe and a fresh run root.
"""
from __future__ import annotations

from dataclasses import dataclass


class RunPreflightError(ValueError):
    """Input identity/timeline or fresh-root preflight failed."""


@dataclass(frozen=True)
class FullVideoRunFixture:
    case_id: str
    input_path: str
    input_sha256: str
    input_size_bytes: int
    frame_count: int
    duration_s: float
    fps: float
    width_px: int
    height_px: int
    fresh_root_template: str


@dataclass(frozen=True)
class SourceProbe:
    sha256: str
    size_bytes: int
    frame_count: int
    duration_s: float
    fps: float
    width_px: int
    height_px: int


DEFAULT_FROZEN_SINGLE_VIDEO = FullVideoRunFixture(
    case_id="feishu_validation_task10_20260718_attempt2",
    input_path="/vePFS-Mindverse/user/yiwen/user-home/zjh/data/v22_feishu_ray_validation_20260718/attempt_0019_droid_no_grad/input/clips/feishu_validation_task10_20260718_attempt2.mp4",
    input_sha256="6a2e406c4de0886daf9efc4c7c072111e761aaef22d1d81470314c3822e29bfe",
    input_size_bytes=14_700_015,
    frame_count=360,
    duration_s=12.0,
    fps=30.0,
    width_px=1920,
    height_px=1080,
    fresh_root_template="/home/zjh/data/v22_api_backend_frozen_single_feishu_validation_task10_20260720T<HHMMSS>Z",
)


def validate_preflight(fixture: FullVideoRunFixture, observed: SourceProbe, *, run_root: str, run_root_exists: bool, run_root_nonempty: bool) -> None:
    expected = SourceProbe(fixture.input_sha256, fixture.input_size_bytes, fixture.frame_count, fixture.duration_s, fixture.fps, fixture.width_px, fixture.height_px)
    if observed != expected:
        raise RunPreflightError(f"source probe mismatch: expected {expected}, observed {observed}")
    if "<HHMMSS>" in run_root or not run_root.startswith("/home/zjh/data/v22_api_backend_frozen_single_"):
        raise RunPreflightError("run_root must expand the fresh timestamp template")
    if run_root_exists or run_root_nonempty:
        raise RunPreflightError("run_root already exists or is non-empty; prediction output cannot be reused")


__all__ = ["DEFAULT_FROZEN_SINGLE_VIDEO", "FullVideoRunFixture", "RunPreflightError", "SourceProbe", "validate_preflight"]
