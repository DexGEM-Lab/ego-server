"""Full-video typed timeline scheduling for the frozen algorithm DAG.

This module stops at the physical-model intermediate.  It decodes one immutable
source timeline, submits every algorithm work unit through the existing typed
``AlgorithmRequest``/``AlgorithmResult`` seam, and returns an N-frame state for a
later MANO/world/render adapter.  It does not render, replay MANO assets, or write
artifact paths.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, TypeVar

import numpy as np

from ego_annotation.api_backend import ApiBackend
from ego_annotation.api_routes import route_for
from ego_annotation.cosmos_semantics import CosmosSemanticResult, run_cosmos_semantics, validate_semantic_coverage
from ego_annotation.fps_config import DEFAULT_FPS_CONDITION, get_fps_condition
from ego_annotation.scripted.contracts import (
    AlgorithmRequest,
    AlgorithmResult,
    ClientRequestTiming,
    FrameTimelineMetadata,
    NativeWorkDescription,
    StageMetadata,
)
from ego_annotation.scripted.droid_rgbd import (
    CanonicalKAggregation,
    DepthEvidence,
    DroidPixelGeometry,
    IntrinsicsCandidate,
    aggregate_canonical_k,
    pack_native_sensor_depth,
)
from ego_annotation.typed_contracts import (
    CropTransform,
    DroidCapabilities,
    DroidCreateInput,
    DroidCreateOutput,
    DroidFinalizeInput,
    DroidFinalizeOutput,
    DroidPushInput,
    DroidPushOutput,
    HandSide,
    HandsInput,
    HandsOutput,
    HaworObservation,
    HaworTrackInput,
    HaworTrackOutput,
    InfillerFrame,
    InfillerInput,
    InfillerOutput,
    Ownership,
    SpatialTransform,
    TypedContractError,
    TypedTensor,
    UniDepthInput,
    UniDepthOutput,
    WiLoRInput,
    WiLoROutput,
)


class TimelineDriverError(RuntimeError):
    """Base failure for source, scheduling, or result-contract violations."""


class PreflightError(TimelineDriverError):
    """The single-video invocation is invalid before any run root is created."""


class StageConfigurationError(TimelineDriverError):
    """A required typed algorithm stage has no executable backend."""


class StageResultError(TimelineDriverError):
    """A backend changed typed identity, timeline, shape, or capability semantics."""


# Each resident DROID session admits at most 256 selected-frame pushes. A
# 64-frame exact source-ID overlap supplies enough motion for a baseline-checked
# Sim(3) gauge join while keeping a 400-frame source at two sessions.
DROID_SERVICE_PUSH_CAPACITY = 256
DROID_SIM3_OVERLAP_FRAMES = 64
DROID_SIM3_MIN_BASELINE = 1.0e-4
DROID_SIM3_MAX_RELATIVE_SCALE = 1.0e4
DROID_SIM3_MAX_NORMALIZED_TRANSLATION_RESIDUAL = 0.05
DROID_SIM3_MAX_ROTATION_RESIDUAL_RAD = 0.10
DEFAULT_HAWOR_ROOT = "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/HaWoR"
DEFAULT_METRIC3D_CHECKPOINT = DEFAULT_HAWOR_ROOT + "/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth"
DEFAULT_METRIC3D_PYTHON = "/home/zjh/miniconda3/envs/hawor/bin/python"


@dataclass(frozen=True)
class SourceFrame:
    frame_index: int
    timestamp_s: float
    frame_id: str
    spatial: SpatialTransform

    def __post_init__(self) -> None:
        if self.frame_index < 0 or not np.isfinite(self.timestamp_s) or not self.frame_id:
            raise TimelineDriverError("source frame identity/timestamp is invalid")


@dataclass(frozen=True)
class SourceTimeline:
    source_id: str
    source_path: str | None
    source_sha256: str
    source_size_bytes: int
    frame_count: int
    fps: float
    duration_s: float
    width_px: int
    height_px: int
    color_space: str
    frames: tuple[SourceFrame, ...]

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_sha256 or len(self.source_sha256) != 64:
            raise TimelineDriverError("source timeline requires source identity and SHA-256")
        if self.source_size_bytes < 0 or self.frame_count <= 0 or self.fps <= 0 or self.duration_s <= 0:
            raise TimelineDriverError("source size/timeline metadata must be positive")
        if self.width_px <= 0 or self.height_px <= 0 or self.color_space != "RGB":
            raise TimelineDriverError("source dimensions and RGB convention are required")
        if len(self.frames) != self.frame_count:
            raise TimelineDriverError("source frame records must cover the complete timeline")
        expected_indices = tuple(range(self.frame_count))
        if tuple(frame.frame_index for frame in self.frames) != expected_indices:
            raise TimelineDriverError("source timeline frame indices must be contiguous from zero")
        expected_times = np.arange(self.frame_count, dtype=np.float64) / self.fps
        actual_times = np.asarray([frame.timestamp_s for frame in self.frames], dtype=np.float64)
        if not np.allclose(actual_times, expected_times, atol=1e-9, rtol=0.0):
            raise TimelineDriverError("source timestamps must be frame_index / fps")

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(range(self.frame_count))

    @property
    def timestamps_s(self) -> tuple[float, ...]:
        return tuple(frame.timestamp_s for frame in self.frames)

    def metadata(self, indices: Sequence[int] | None = None) -> FrameTimelineMetadata:
        selected = self.frame_indices if indices is None else tuple(int(index) for index in indices)
        return self._metadata_for_indices(selected, timeline_mode="dense")

    def droid_sampled_metadata(self, indices: Sequence[int]) -> FrameTimelineMetadata:
        """Build the explicit sparse envelope reserved for scheduled DROID chunks."""
        selected = tuple(int(index) for index in indices)
        return self._metadata_for_indices(selected, timeline_mode="droid_sampled")

    def _metadata_for_indices(self, selected: tuple[int, ...], *, timeline_mode: str) -> FrameTimelineMetadata:
        if not selected:
            raise TimelineDriverError("request timeline cannot be empty")
        if any(index < 0 or index >= self.frame_count for index in selected):
            raise TimelineDriverError("request timeline index is outside source timeline")
        return FrameTimelineMetadata(
            source_id=self.source_id,
            frame_indices=selected,
            timestamps_s=tuple(self.frames[index].timestamp_s for index in selected),
            source_sha256=self.source_sha256,
            width_px=self.width_px,
            height_px=self.height_px,
            fps=self.fps,
            timeline_mode=timeline_mode,
        )


class FrameSource(Protocol):
    @property
    def timeline(self) -> SourceTimeline:
        ...

    def read_rgb(self, frame_index: int) -> np.ndarray:
        ...

    def iter_rgb(self, frame_indices: Sequence[int]) -> Iterable[tuple[int, np.ndarray]]:
        ...


class InMemoryFrameSource:
    """CPU-test frame source with the same immutable timeline contract as video."""

    def __init__(
        self,
        frames_rgb: Sequence[np.ndarray],
        *,
        fps: float,
        source_id: str = "in-memory-source",
        source_sha256: str | None = None,
    ) -> None:
        if not frames_rgb or fps <= 0:
            raise TimelineDriverError("in-memory source requires frames and positive fps")
        copied: list[np.ndarray] = []
        shape: tuple[int, int, int] | None = None
        digest = hashlib.sha256()
        for frame in frames_rgb:
            value = np.ascontiguousarray(frame)
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                raise TimelineDriverError("in-memory frames must be uint8 RGB HWC")
            if shape is None:
                shape = value.shape
            if value.shape != shape:
                raise TimelineDriverError("all source frames must have one spatial shape")
            value.setflags(write=False)
            copied.append(value)
            digest.update(value.tobytes(order="C"))
        assert shape is not None
        self._frames = tuple(copied)
        sha = source_sha256 or digest.hexdigest()
        height, width, _ = shape
        spatial = _identity_spatial("source_rgb", width, height)
        records = tuple(
            SourceFrame(index, index / fps, f"{source_id}:frame:{index:06d}", spatial)
            for index in range(len(copied))
        )
        self._timeline = SourceTimeline(
            source_id=source_id,
            source_path=None,
            source_sha256=sha,
            source_size_bytes=sum(frame.nbytes for frame in copied),
            frame_count=len(copied),
            fps=float(fps),
            duration_s=len(copied) / float(fps),
            width_px=width,
            height_px=height,
            color_space="RGB",
            frames=records,
        )

    @property
    def timeline(self) -> SourceTimeline:
        return self._timeline

    def read_rgb(self, frame_index: int) -> np.ndarray:
        try:
            return self._frames[frame_index]
        except IndexError as exc:
            raise TimelineDriverError(f"source frame {frame_index} is out of range") from exc

    def iter_rgb(self, frame_indices: Sequence[int]) -> Iterable[tuple[int, np.ndarray]]:
        for frame_index in frame_indices:
            yield frame_index, self.read_rgb(frame_index)


class OpenCvFrameSource:
    """Original-resolution RGB source with an explicit sequential frame store.

    ``build_frame_store`` is the batched-driver path: it traverses the source once
    in ascending order, retains only pre-registered indices, and makes every later
    access a RAM or per-frame NPY lookup.  The small LRU/random-seek path remains
    available only to callers that never build a store.
    """

    DEFAULT_FRAME_STORE_MAX_BYTES = 24 * 1024**3

    def __init__(
        self,
        timeline: SourceTimeline,
        *,
        cache_frames: int = 8,
        frame_store_max_bytes: int | None = None,
    ) -> None:
        if timeline.source_path is None:
            raise TimelineDriverError("OpenCV frame source requires a local source path")
        if cache_frames <= 0:
            raise TimelineDriverError("frame cache size must be positive")
        configured_cap = frame_store_max_bytes
        if configured_cap is None:
            raw_cap = os.environ.get("EGO_FRAME_STORE_MAX_BYTES")
            try:
                configured_cap = self.DEFAULT_FRAME_STORE_MAX_BYTES if raw_cap is None else int(raw_cap)
            except ValueError as exc:
                raise TimelineDriverError("EGO_FRAME_STORE_MAX_BYTES must be an integer") from exc
        if configured_cap < 0:
            raise TimelineDriverError("frame store byte cap must be nonnegative")
        self._timeline = timeline
        self._cache_frames = cache_frames
        self._frame_store_max_bytes = int(configured_cap)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._capture: Any | None = None
        self._lazy_last_requested: int | None = None
        self._store_built = False
        self._store_registered: frozenset[int] = frozenset()
        self._store_ram: dict[int, np.ndarray] = {}
        self._store_disk: dict[int, Path] = {}
        self._store_spill_dir: Path | None = None
        self._store_ram_bytes = 0
        self._store_disk_bytes = 0
        self._capture_read_calls = 0
        self._capture_grab_calls = 0
        self._pos_frames_seek_calls = 0
        self._backward_seek_calls = 0
        self._lookup_calls = 0
        self._missing_lookup_errors = 0
        self._traversal_passes = 0
        self._traversed_frame_count = 0

    @classmethod
    def from_video(
        cls,
        path: str | os.PathLike[str],
        *,
        cache_frames: int = 8,
        frame_store_max_bytes: int | None = None,
    ) -> "OpenCvFrameSource":
        return cls(inspect_video(path), cache_frames=cache_frames, frame_store_max_bytes=frame_store_max_bytes)

    @property
    def timeline(self) -> SourceTimeline:
        return self._timeline

    @property
    def frame_store_built(self) -> bool:
        return self._store_built

    def build_frame_store(
        self,
        frame_indices: Sequence[int],
        *,
        spill_dir: str | os.PathLike[str],
    ) -> None:
        """Decode a pre-registered union in one forward pass without POS_FRAMES seeks."""
        requested = tuple(sorted(set(int(index) for index in frame_indices)))
        if not requested:
            raise TimelineDriverError("frame store requires at least one registered frame")
        if any(index < 0 or index >= self._timeline.frame_count for index in requested):
            raise TimelineDriverError("frame store index is outside source timeline")
        if self._store_built:
            missing = set(requested).difference(self._store_registered)
            if missing:
                raise TimelineDriverError(f"frame store was already built without indices {sorted(missing)[:8]}")
            return
        spill_root = Path(spill_dir).expanduser().resolve()
        spill_root.mkdir(parents=True, exist_ok=False)
        import cv2

        capture = cv2.VideoCapture(str(self._timeline.source_path))
        if not capture.isOpened():
            raise TimelineDriverError(f"cannot open video {self._timeline.source_path}")
        registered = frozenset(requested)
        ram: dict[int, np.ndarray] = {}
        disk: dict[int, Path] = {}
        ram_bytes = 0
        disk_bytes = 0
        read_calls = 0
        max_index = requested[-1]
        try:
            for frame_index in range(max_index + 1):
                ok, bgr = capture.read()
                read_calls += 1
                if not ok or bgr is None:
                    raise TimelineDriverError(f"failed to decode source frame {frame_index}")
                if frame_index not in registered:
                    continue
                rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if rgb.shape != (self._timeline.height_px, self._timeline.width_px, 3):
                    raise TimelineDriverError("decoded frame dimensions changed from preflight metadata")
                rgb.setflags(write=False)
                if ram_bytes + rgb.nbytes <= self._frame_store_max_bytes:
                    ram[frame_index] = rgb
                    ram_bytes += int(rgb.nbytes)
                else:
                    path = spill_root / f"frame_{frame_index:08d}.npy"
                    np.save(path, rgb, allow_pickle=False)
                    disk[frame_index] = path
                    disk_bytes += int(path.stat().st_size)
        except BaseException:
            for path in spill_root.glob("frame_*.npy"):
                path.unlink(missing_ok=True)
            spill_root.rmdir()
            raise
        finally:
            capture.release()
        if set(ram).union(disk) != set(registered):
            raise TimelineDriverError("sequential frame store did not retain every registered frame")
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            self._cache.clear()
            self._store_registered = registered
            self._store_ram = ram
            self._store_disk = disk
            self._store_spill_dir = spill_root
            self._store_ram_bytes = ram_bytes
            self._store_disk_bytes = disk_bytes
            self._capture_read_calls += read_calls
            self._traversal_passes += 1
            self._traversed_frame_count += max_index + 1
            self._store_built = True
        (spill_root.parent / "frame_store_decode_report.json").write_text(
            json.dumps(self.frame_store_report(), indent=2), encoding="utf-8"
        )

    def read_rgb(self, frame_index: int) -> np.ndarray:
        if frame_index < 0 or frame_index >= self._timeline.frame_count:
            raise TimelineDriverError(f"source frame {frame_index} is out of range")
        if self._store_built:
            with self._lock:
                self._lookup_calls += 1
                value = self._store_ram.get(frame_index)
                path = self._store_disk.get(frame_index)
                registered = frame_index in self._store_registered
            if value is not None:
                return value
            if path is not None:
                loaded = np.ascontiguousarray(np.load(path, allow_pickle=False))
                if loaded.dtype != np.uint8 or loaded.shape != (self._timeline.height_px, self._timeline.width_px, 3):
                    raise TimelineDriverError(f"spilled source frame {frame_index} has invalid geometry or dtype")
                loaded.setflags(write=False)
                return loaded
            if not registered:
                with self._lock:
                    self._missing_lookup_errors += 1
                raise TimelineDriverError(f"source frame {frame_index} was not pre-registered in the frame store")
            raise TimelineDriverError(f"registered source frame {frame_index} is absent from the frame store")
        with self._lock:
            cached = self._cache.pop(frame_index, None)
            if cached is not None:
                self._cache[frame_index] = cached
                return cached
            import cv2

            if self._capture is None:
                self._capture = cv2.VideoCapture(str(self._timeline.source_path))
                if not self._capture.isOpened():
                    raise TimelineDriverError(f"cannot open video {self._timeline.source_path}")
            self._pos_frames_seek_calls += 1
            if self._lazy_last_requested is not None and frame_index < self._lazy_last_requested:
                self._backward_seek_calls += 1
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self._lazy_last_requested = frame_index
            ok, bgr = self._capture.read()
            self._capture_read_calls += 1
            if not ok or bgr is None:
                raise TimelineDriverError(f"failed to decode source frame {frame_index}")
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if rgb.shape != (self._timeline.height_px, self._timeline.width_px, 3):
                raise TimelineDriverError("decoded frame dimensions changed from preflight metadata")
            rgb.setflags(write=False)
            self._cache[frame_index] = rgb
            while len(self._cache) > self._cache_frames:
                self._cache.popitem(last=False)
            return rgb

    def iter_rgb(self, frame_indices: Sequence[int]) -> Iterable[tuple[int, np.ndarray]]:
        requested = tuple(int(index) for index in frame_indices)
        if tuple(sorted(set(requested))) != requested:
            raise TimelineDriverError("sequential frame decode requires strictly increasing unique indices")
        if any(index < 0 or index >= self._timeline.frame_count for index in requested):
            raise TimelineDriverError("sequential frame decode index is outside source timeline")
        if self._store_built:
            for frame_index in requested:
                yield frame_index, self.read_rgb(frame_index)
            return
        if not requested:
            return
        import cv2

        capture = cv2.VideoCapture(str(self._timeline.source_path))
        if not capture.isOpened():
            raise TimelineDriverError(f"cannot open video {self._timeline.source_path}")
        wanted = iter(requested)
        target = next(wanted)
        read_calls = 0
        try:
            for frame_index in range(target + 1):
                ok, bgr = capture.read()
                read_calls += 1
                if not ok or bgr is None:
                    raise TimelineDriverError(f"failed to decode source frame {frame_index}")
            while True:
                rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if rgb.shape != (self._timeline.height_px, self._timeline.width_px, 3):
                    raise TimelineDriverError("decoded frame dimensions changed from preflight metadata")
                rgb.setflags(write=False)
                with self._lock:
                    self._cache[target] = rgb
                    while len(self._cache) > self._cache_frames:
                        self._cache.popitem(last=False)
                yield target, rgb
                try:
                    target = next(wanted)
                except StopIteration:
                    return
                for frame_index in range(frame_index + 1, target + 1):
                    ok, bgr = capture.read()
                    read_calls += 1
                    if not ok or bgr is None:
                        raise TimelineDriverError(f"failed to decode source frame {frame_index}")
        finally:
            capture.release()
            with self._lock:
                self._capture_read_calls += read_calls
                self._traversal_passes += 1
                self._traversed_frame_count += read_calls

    def frame_store_report(self) -> dict[str, object]:
        with self._lock:
            return {
                "schema": "ego.annotation.frame_store_decode.v1",
                "status": "built" if self._store_built else "lazy_fallback",
                "source_frame_count": self._timeline.frame_count,
                "source_width_px": self._timeline.width_px,
                "source_height_px": self._timeline.height_px,
                "source_pixel_format": "uint8_rgb_original_resolution",
                "registered_frame_count": len(self._store_registered),
                "registered_max_frame_index": max(self._store_registered) if self._store_registered else None,
                "ram_frame_count": len(self._store_ram),
                "ram_bytes": self._store_ram_bytes,
                "ram_cap_bytes": self._frame_store_max_bytes,
                "spill_frame_count": len(self._store_disk),
                "spill_bytes": self._store_disk_bytes,
                "spill_dir": str(self._store_spill_dir) if self._store_spill_dir is not None else None,
                "capture_read_calls": self._capture_read_calls,
                "capture_grab_calls": self._capture_grab_calls,
                "pos_frames_seek_calls": self._pos_frames_seek_calls,
                "backward_seek_calls": self._backward_seek_calls,
                "traversal_passes": self._traversal_passes,
                "traversed_frame_count": self._traversed_frame_count,
                "unique_decoded_frame_count": self._traversed_frame_count,
                "duplicate_decode_count": max(0, self._capture_read_calls - self._traversed_frame_count),
                "lookup_calls": self._lookup_calls,
                "missing_lookup_errors": self._missing_lookup_errors,
            }


@dataclass(frozen=True)
class ValidatedVideoProfile:
    profile_id: str
    case_id: str
    source_sha256: str
    source_size_bytes: int
    frame_count: int
    fps: float
    duration_s: float
    width_px: int
    height_px: int

    def __post_init__(self) -> None:
        if not self.profile_id or not self.case_id or len(self.source_sha256) != 64:
            raise PreflightError("validated profile identity is invalid")
        if min(self.source_size_bytes, self.frame_count, self.width_px, self.height_px) <= 0:
            raise PreflightError("validated profile sizes/counts must be positive")
        if self.fps <= 0 or self.duration_s <= 0:
            raise PreflightError("validated profile timeline must be positive")


@dataclass(frozen=True)
class SingleVideoPreflight:
    case_id: str
    fresh_root: str
    timeline: SourceTimeline
    profile_id: str | None
    checks: tuple[str, ...]


@dataclass(frozen=True)
class CoveragePolicy:
    chunk_length: int
    stride: int
    tail: str = "pad_unobserved"

    def __post_init__(self) -> None:
        if self.chunk_length <= 0 or self.stride <= 0 or self.stride > self.chunk_length:
            raise TimelineDriverError("coverage length/stride must be positive and stride <= length")
        if self.tail != "pad_unobserved":
            raise TimelineDriverError("only explicit pad_unobserved tail semantics are supported")

    def starts(self, length: int, *, offset: int = 0) -> tuple[int, ...]:
        if length <= 0 or offset < 0:
            return ()
        return tuple(range(offset, offset + length, self.stride))


@dataclass(frozen=True)
class FullVideoDriverConfig:
    item_batch_size: int = 1
    cosmos_enabled: bool = False
    fps_condition: str = DEFAULT_FPS_CONDITION
    unidepth_fps: float | None = None
    droid_fps: float | None = None
    inference_size_yx: tuple[int, int] | None = (540, 960)
    cosmos_gallery_width: int = 960
    hawor_coverage: CoveragePolicy = CoveragePolicy(16, 8)
    infiller_coverage: CoveragePolicy = CoveragePolicy(120, 60)
    max_track_gap: int = 8
    min_hand_score: float = 1e-6
    track_match_threshold: float = 2.25
    crop_scale: float = 1.25
    require_rgbd_capability: bool = True
    allow_monocular_droid_smoke: bool = False
    frame_store_spill_dir: str | None = None
    droid_target_area_px: int = 384 * 512
    # An explicit model input shape is diagnostic-only. The default preserves
    # target-area resizing; a probe may compare one historical model geometry.
    droid_input_shape_yx: tuple[int, int] | None = None
    # A resident DROID session accepts no more than 256 selected frames. Source-keyed
    # sessions use a fixed 16-frame exact-ID overlap for a baseline-checked Sim(3) join.
    droid_keyframe_buffer: int = DROID_SERVICE_PUSH_CAPACITY
    droid_session_workers: int = 2
    # Exact reference scalar path. Metric3D runs in its pinned HaWoR environment;
    # UniDepth remains the wrist-depth/intrinsics source only.
    droid_metric3d_hawor_root: str | None = DEFAULT_HAWOR_ROOT
    droid_metric3d_checkpoint: str | None = DEFAULT_METRIC3D_CHECKPOINT
    droid_metric3d_python: str = DEFAULT_METRIC3D_PYTHON
    droid_metric3d_cuda_visible_devices: str = "1"
    lower_filter_retry_thresh: float | None = 1.2
    max_keyframe_retries: int = 1
    model_revisions: Mapping[str, str] = field(default_factory=lambda: {
        "unidepth.infer": "unidepth-v2-vitl14-corrected",
        "hands.detect": "hands-yolo-v2",
        "wilor.reconstruct": "wilor-final-v1",
        "droid.create_session": "droid-v1",
        "droid.push_frame": "droid-v1",
        "droid.finalize": "droid-v1",
        "hawor.infer_tracks": "hawor-v1",
        "hawor_infiller.fill": "hawor-infiller-v1",
        "cosmos3.reason": "cosmos3-frozen",
    })

    def __post_init__(self) -> None:
        condition = get_fps_condition(self.fps_condition)
        if self.unidepth_fps is None:
            object.__setattr__(self, "unidepth_fps", condition.unidepth_fps)
        if self.droid_fps is None:
            object.__setattr__(self, "droid_fps", condition.droid_fps)
        if self.item_batch_size != 1:
            raise TimelineDriverError("single-video driver requires item_batch_size=1")
        if self.cosmos_gallery_width <= 0:
            raise TimelineDriverError("Cosmos gallery width must be positive")
        if self.require_rgbd_capability == self.allow_monocular_droid_smoke:
            raise TimelineDriverError("select strict RGB-D or explicit diagnostic monocular mode")
        if self.droid_target_area_px <= 0:
            raise TimelineDriverError("DROID target area must be positive")
        if self.droid_keyframe_buffer != DROID_SERVICE_PUSH_CAPACITY:
            raise TimelineDriverError("DROID keyframe buffer is fixed to the 256-push service capacity")
        if self.droid_session_workers <= 0:
            raise TimelineDriverError("DROID session worker bound must be positive")
        if not self.droid_metric3d_python.strip():
            raise TimelineDriverError("Metric3D Python interpreter must be configured")
        if self.droid_input_shape_yx is not None and (
            len(self.droid_input_shape_yx) != 2
            or any(int(value) <= 0 or int(value) % 8 for value in self.droid_input_shape_yx)
        ):
            raise TimelineDriverError("explicit DROID input shape must be positive H,W multiples of 8")
        if self.frame_store_spill_dir is not None and not self.frame_store_spill_dir.strip():
            raise TimelineDriverError("frame store spill directory must be a nonempty path")
        if self.inference_size_yx is not None and (
            len(self.inference_size_yx) != 2 or any(dim <= 0 for dim in self.inference_size_yx)
        ):
            raise TimelineDriverError("inference_size_yx must be positive H,W")
        if self.hawor_coverage.chunk_length != 16:
            raise TimelineDriverError("frozen HaWoR contract fixes chunk_length=16")
        if self.infiller_coverage.chunk_length != 120:
            raise TimelineDriverError("frozen Infiller contract fixes horizon=120")
        if self.max_track_gap < 0 or self.track_match_threshold <= 0 or self.crop_scale <= 0:
            raise TimelineDriverError("tracking/crop policy is invalid")
        if self.max_keyframe_retries != 1:
            raise TimelineDriverError("DROID insufficient-keyframe recovery is bounded to one retry")
        if self.lower_filter_retry_thresh is not None and self.lower_filter_retry_thresh <= 0:
            raise TimelineDriverError("DROID retry filter threshold must be positive")
        required = set(REQUIRED_STAGE_IDS)
        if self.cosmos_enabled:
            required.add("cosmos3.reason")
        missing = required.difference(self.model_revisions)
        if missing:
            raise TimelineDriverError(f"model revisions missing stages: {sorted(missing)}")


class AlgorithmBackend(Protocol):
    def execute(self, request: AlgorithmRequest[Any]) -> AlgorithmResult[Any]:
        ...


class AlgorithmStageClient:
    """Typed adapter over either one generic backend or stage-owned backends.

    Stage-owned scripted workers and the live backend both expose the same generic
    ``execute`` operation. This adapter adds required-stage preflight and
    full-burst ``execute_many`` without introducing another codec or model API. Each stage
    submits its complete logical request set to the manager-owned admission
    boundary; this client deliberately adds no per-video admission layer.
    """

    def __init__(
        self,
        *,
        backend: AlgorithmBackend | None = None,
        stage_backends: Mapping[str, AlgorithmBackend] | None = None,
    ) -> None:
        if (backend is None) == (stage_backends is None):
            raise StageConfigurationError("configure exactly one generic backend or a stage-backend mapping")
        self._backend = backend
        self._stage_backends = dict(stage_backends or {})

    def preflight(self, required_stage_ids: Iterable[str]) -> None:
        required = set(required_stage_ids)
        if self._backend is not None:
            if not callable(getattr(self._backend, "execute", None)):
                raise StageConfigurationError("generic backend has no typed execute method")
            return
        missing = sorted(stage for stage in required if stage not in self._stage_backends)
        invalid = sorted(
            stage for stage, backend in self._stage_backends.items()
            if stage in required and not callable(getattr(backend, "execute", None))
        )
        if missing or invalid:
            raise StageConfigurationError(
                f"script stage configuration incomplete: missing={missing}, invalid={invalid}"
            )

    def execute(self, request: AlgorithmRequest[Any]) -> AlgorithmResult[Any]:
        result, _timing = self.execute_timed(request)
        return result

    def execute_timed(self, request: AlgorithmRequest[Any]) -> tuple[AlgorithmResult[Any], ClientRequestTiming]:
        backend = self._backend or self._stage_backends.get(request.algorithm_id)
        if backend is None:
            raise StageConfigurationError(f"no typed backend configured for {request.algorithm_id!r}")
        started = time.monotonic()
        timed = getattr(backend, "execute_timed", None)
        if callable(timed):
            result, timing = timed(request)
        else:
            result = backend.execute(request)
            timing = ClientRequestTiming(
                total_wall_s=time.monotonic() - started,
                available=False,
                unavailable_reason="backend exposes execute() only; prepare/transport/decode boundaries unavailable",
            )
        _validate_result_envelope(request, result)
        return result, timing

    def execute_many(
        self,
        requests: Sequence[AlgorithmRequest[Any]],
    ) -> tuple[AlgorithmResult[Any], ...]:
        with ThreadPoolExecutor(
            max_workers=_stage_worker_count(requests),
            thread_name_prefix=f"stage-{requests[0].algorithm_id}" if requests else "stage-empty",
        ) as pool:
            futures = [pool.submit(self.execute, request) for request in requests]
            return tuple(future.result() for future in futures)


def _stage_worker_count(requests: Sequence[object]) -> int:
    """Give every logical stage request a worker; empty batches need a valid API value."""

    return len(requests) if requests else 1


class ScriptAlgorithmStageClient(AlgorithmStageClient):
    """Required-stage adapter for script-owned generic ``execute`` workers."""

    def __init__(self, stage_backends: Mapping[str, AlgorithmBackend]) -> None:
        super().__init__(stage_backends=stage_backends)


class LiveFrozenApiStageClient(AlgorithmStageClient):
    """Typed adapter over the existing frozen-route ``ApiBackend`` methods."""

    def __init__(self, backend: ApiBackend) -> None:
        super().__init__(backend=backend)


@dataclass(frozen=True)
class HandDetectionRecord:
    detection_id: str
    frame_index: int
    timestamp_s: float
    side: HandSide
    box_xyxy_source: tuple[float, float, float, float]
    score: float
    visibility: float
    uncertainty: float
    occlusion_state: str
    ownership: Ownership


@dataclass(frozen=True)
class HandTrack:
    track_id: str
    side: HandSide
    start_frame: int
    end_frame: int
    detections: tuple[HandDetectionRecord, ...]
    visibility_by_frame: tuple[str, ...]
    uncertainty_by_frame: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise TimelineDriverError("hand track bounds are invalid")
        if any(record.side is not self.side for record in self.detections):
            raise TimelineDriverError("hand track cannot change side")


@dataclass(frozen=True)
class HaworChunkTrace:
    track_id: str
    side: HandSide
    start_frame: int
    source_frame_slots: tuple[int, ...]
    observed_slots: tuple[bool, ...]
    padded_slots: tuple[bool, ...]
    request_scope: str


@dataclass(frozen=True)
class InfillerWindowTrace:
    window_id: str
    start_frame: int
    source_frame_slots: tuple[int, ...]
    observed_left: tuple[bool, ...]
    observed_right: tuple[bool, ...]
    padded_slots: tuple[bool, ...]
    submitted: bool
    blocker: str | None


@dataclass
class ModuleTimingCollector:
    """Thread-safe per-job accumulator for client and local timing boundaries.

    Request sums intentionally remain additive under concurrency; module wall
    spans are supplied separately from the existing non-overcounting traces.
    """

    _requests: dict[str, list[ClientRequestTiming]] = field(default_factory=dict)
    _local_s: dict[str, float] = field(default_factory=dict)
    _unavailable: dict[str, set[str]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def request(self, stage_id: str, timing: ClientRequestTiming) -> None:
        with self._lock:
            self._requests.setdefault(stage_id, []).append(timing)
            if not timing.available and timing.unavailable_reason:
                self._unavailable.setdefault(stage_id, set()).add(timing.unavailable_reason)

    def local(self, module: str, duration_s: float) -> None:
        with self._lock:
            self._local_s[module] = self._local_s.get(module, 0.0) + max(0.0, float(duration_s))

    def breakdown(self, module_timings_s: Mapping[str, float]) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
        groups = {
            "unidepth": ("unidepth.infer",), "hands": ("hands.detect",),
            "wilor_service": ("wilor.reconstruct",),
            "droid": ("droid.create_session", "droid.push_frame", "droid.finalize"),
            "hawor": ("hawor.infer_tracks",), "infiller": ("hawor_infiller.fill",),
            "cosmos": ("cosmos3.reason",),
        }
        with self._lock:
            requests = {key: tuple(value) for key, value in self._requests.items()}
            local = dict(self._local_s)
            unavailable = {key: set(value) for key, value in self._unavailable.items()}
        result: dict[str, dict[str, object]] = {}
        notes: dict[str, str] = {}
        for module, stages in groups.items():
            values = tuple(timing for stage in stages for timing in requests.get(stage, ()))
            result[module] = {
                "client_prepare_s": sum(item.client_prepare_s for item in values),
                "transport_wait_s": sum(item.transport_wait_s for item in values),
                "client_decode_postprocess_s": sum(item.client_decode_postprocess_s for item in values),
                "local_assembly_write_s": local.get(module, 0.0),
                "total_wall_s": float(module_timings_s.get(module, 0.0)),
                "request_count": len(values),
            }
            reasons = sorted(unavailable.get(next((stage for stage in stages if stage in unavailable), ""), set()))
            if reasons:
                notes[module] = "; ".join(reasons)
            elif not values:
                notes[module] = "no request timing boundary observed for this module"
        for module in ("frame_store", "wilor_build", "render"):
            result.setdefault(module, {
                "client_prepare_s": 0.0, "transport_wait_s": 0.0,
                "client_decode_postprocess_s": 0.0,
                "local_assembly_write_s": local.get(module, 0.0),
                "total_wall_s": float(module_timings_s.get(module, 0.0)), "request_count": 0,
            })
            if module not in notes and module not in local:
                notes[module] = "no timing boundary observed for this module"
        return result, notes


@dataclass(frozen=True)
class RequestBatchTrace:
    stage_id: str
    request_count: int
    submitted_concurrency: int
    native_batch_cap: int
    native_work_shape: tuple[int, ...]
    started_monotonic_s: float
    completed_monotonic_s: float


def _record_service_module_timings(
    module_timings_s: dict[str, float],
    traces: Sequence[RequestBatchTrace],
) -> None:
    """Record stage wall intervals without summing concurrent requests."""

    stage_groups = {
        "unidepth": {"unidepth.infer"},
        "hands": {"hands.detect"},
        "wilor_service": {"wilor.reconstruct"},
        "droid": {"droid.create_session", "droid.push_frame", "droid.finalize"},
        "hawor": {"hawor.infer_tracks"},
        "infiller": {"hawor_infiller.fill"},
        "cosmos": {"cosmos3.reason"},
    }
    for module, stage_ids in stage_groups.items():
        selected = [trace for trace in traces if trace.stage_id in stage_ids]
        module_timings_s[module] = 0.0 if not selected else float(max(trace.completed_monotonic_s for trace in selected) - min(trace.started_monotonic_s for trace in selected))


@dataclass(frozen=True)
class TimelineFrameProvenance:
    frame_index: int
    side: HandSide
    source_stage: str
    source_scope: str
    observed: bool
    inferred: bool
    uncertainty_m: float


@dataclass(frozen=True)
class TimelineInferenceState:
    frame_count: int
    side_order: tuple[HandSide, HandSide]
    root_orient: TypedTensor
    hand_pose: TypedTensor
    betas: TypedTensor
    trans_camera_m: TypedTensor
    vertices_camera_m: TypedTensor
    joints_camera_m: TypedTensor
    vertices_source_px: TypedTensor
    joints_source_px: TypedTensor
    valid: TypedTensor
    observed: TypedTensor
    inferred: TypedTensor
    uncertainty_m: TypedTensor
    visibility_state: tuple[tuple[str, ...], tuple[str, ...]]
    provenance: tuple[TimelineFrameProvenance, ...]
    merge_policy: str

    def __post_init__(self) -> None:
        n = self.frame_count
        if self.side_order != (HandSide.LEFT, HandSide.RIGHT):
            raise TimelineDriverError("timeline side axis must be [left,right]")
        expected = {
            "root": (2, n, 3, 3),
            "pose": (2, n, 15, 3, 3),
            "betas": (2, n, 10),
            "trans": (2, n, 3),
            "vertices": (2, n, 778, 3),
        }
        actual = {
            "root": self.root_orient.shape,
            "pose": self.hand_pose.shape,
            "betas": self.betas.shape,
            "trans": self.trans_camera_m.shape,
            "vertices": self.vertices_camera_m.shape,
        }
        if actual != expected or self.joints_camera_m.shape[:2] != (2, n) or self.joints_camera_m.shape[-1] != 3:
            raise TimelineDriverError("timeline MANO arrays do not cover typed [2,N] state")
        if self.vertices_source_px.shape != (2, n, 778, 2) or self.joints_source_px.shape != self.joints_camera_m.shape[:-1] + (2,):
            raise TimelineDriverError("timeline crop projections do not cover typed [2,N] MANO state")
        for tensor in (self.valid, self.observed, self.inferred, self.uncertainty_m):
            if tensor.shape != (2, n):
                raise TimelineDriverError("timeline masks/uncertainty must be [2,N]")
        if any(len(row) != n for row in self.visibility_state):
            raise TimelineDriverError("timeline visibility must cover every source frame")


@dataclass(frozen=True)
class DroidChunkAttemptOutcome:
    """One primary or bounded-recovery session for a scheduled chunk."""

    attempt: int
    session_id: str
    options: Mapping[str, str | int | float | bool]
    succeeded: bool
    error_type: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.attempt not in {0, 1} or not self.session_id:
            raise TimelineDriverError("DROID chunk attempt identity is invalid")
        if self.succeeded == (self.error is not None or self.error_type is not None):
            raise TimelineDriverError("DROID chunk attempt success/error provenance is inconsistent")

    def to_wire(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "session_id": self.session_id,
            "options": dict(self.options),
            "succeeded": self.succeeded,
            "error_type": self.error_type,
            "error": self.error,
        }


@dataclass(frozen=True)
class DroidChunkOutcome:
    """One promptly-finalized DROID chunk in a stitched sparse schedule."""

    chunk_index: int
    source_indices: tuple[int, ...]
    session_id: str
    keyframe_count: int
    stitch_boundary_translation_error_m: float | None
    stitch_boundary_rotation_error_rad: float | None
    attempts: tuple[DroidChunkAttemptOutcome, ...] = ()


class DroidChunkFinalizeError(StageResultError):
    """Finalize failed after a complete scheduled chunk submission."""

    def __init__(
        self,
        *,
        chunk_index: int,
        attempt: int,
        session_id: str,
        options: Mapping[str, str | int | float | bool],
        create_result: AlgorithmResult[DroidCreateOutput],
        push_results: tuple[AlgorithmResult[DroidPushOutput], ...],
        traces: tuple[RequestBatchTrace, ...],
        cause: Exception,
    ) -> None:
        self.chunk_index = chunk_index
        self.attempt = attempt
        self.session_id = session_id
        self.options = dict(options)
        self.create_result = create_result
        self.push_results = push_results
        self.traces = traces
        self.cause_type = type(cause).__name__
        self.cause_message = str(cause)
        lowered = self.cause_message.lower()
        self.nonfinite_trajectory = "t_world_camera" in lowered and "finite values" in lowered
        super().__init__(
            f"DROID chunk {chunk_index} attempt {attempt} finalize failed for session {session_id}: "
            f"{self.cause_type}: {self.cause_message}"
        )

    def attempt_outcome(self) -> DroidChunkAttemptOutcome:
        return DroidChunkAttemptOutcome(
            self.attempt,
            self.session_id,
            dict(self.options),
            False,
            self.cause_type,
            self.cause_message,
        )


@dataclass(frozen=True)
class DroidExecutionRecords:
    create_results: tuple[AlgorithmResult[DroidCreateOutput], ...]
    push_results_by_attempt: tuple[tuple[AlgorithmResult[DroidPushOutput], ...], ...]
    finalize_results: tuple[AlgorithmResult[DroidFinalizeOutput], ...]
    retries_used: int
    accepted_trajectory: bool
    blocker: str | None
    coverage: DroidCoverage
    chunk_outcomes: tuple[DroidChunkOutcome, ...] = ()

    @property
    def final(self) -> AlgorithmResult[DroidFinalizeOutput]:
        if not self.finalize_results:
            raise TimelineDriverError("DROID execution has no finalize result")
        return self.finalize_results[-1]


@dataclass(frozen=True)
class AlgorithmAcceptance:
    accepted: bool
    diagnostic_only: bool
    scale_mode: str
    reasons: tuple[str, ...]
    physical_adapter_status: str
    render_status: str

    def __post_init__(self) -> None:
        if self.physical_adapter_status != "pending_next_slice" or self.render_status != "pending_next_slice":
            raise TimelineDriverError("this slice may only expose pending physical/render status")
        if self.diagnostic_only and self.accepted:
            raise TimelineDriverError("diagnostic state cannot be accepted")


@dataclass(frozen=True)
class FullVideoAlgorithmState:
    frame_count: int
    source_timeline: SourceTimeline
    canonical_K: CanonicalKAggregation
    hand_detections: tuple[HandDetectionRecord, ...]
    hand_tracks: tuple[HandTrack, ...]
    unidepth_records: tuple[AlgorithmResult[UniDepthOutput], ...]
    hands_records: tuple[AlgorithmResult[HandsOutput], ...]
    wilor_records: tuple[AlgorithmResult[WiLoROutput], ...]
    droid_records: DroidExecutionRecords
    hawor_records: tuple[AlgorithmResult[HaworTrackOutput], ...]
    infiller_records: tuple[AlgorithmResult[InfillerOutput], ...]
    hawor_chunks: tuple[HaworChunkTrace, ...]
    infiller_windows: tuple[InfillerWindowTrace, ...]
    timeline_inference: TimelineInferenceState
    batch_request_traces: tuple[RequestBatchTrace, ...]
    semantic_status: str
    semantic_rows: tuple[Mapping[str, object], ...]
    semantic_request_count: int
    semantic_review: Mapping[str, object]
    acceptance: AlgorithmAcceptance
    module_timings_s: Mapping[str, float] = field(default_factory=dict)
    module_timing_breakdown_s: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    module_timing_breakdown_notes: Mapping[str, str] = field(default_factory=dict)
    render_paths: tuple[str, ...] = ()
    manifest_paths: tuple[str, ...] = ()

    @property
    def hawor_geometry_diagnostics(self) -> Mapping[str, object]:
        return _hawor_geometry_diagnostics(self.hawor_records, self.hawor_chunks)

    def __post_init__(self) -> None:
        if self.frame_count != self.source_timeline.frame_count or self.timeline_inference.frame_count != self.frame_count:
            raise TimelineDriverError("full-video state must cover exactly source N")
        if self.semantic_status == "absent_disabled":
            if self.semantic_rows or self.semantic_request_count != 0 or self.semantic_review:
                raise TimelineDriverError("disabled Cosmos must have no semantic rows, requests, or review")
        elif self.semantic_status in {"enabled", "completed_with_anomalies"}:
            if self.semantic_request_count <= 0 or not self.semantic_review:
                raise TimelineDriverError("completed Cosmos requires truthful request and review evidence")
            try:
                validate_semantic_coverage(self.semantic_rows, self.frame_count)
            except Exception as exc:
                raise TimelineDriverError(str(exc)) from exc
        else:
            raise TimelineDriverError("semantic status must be enabled or absent_disabled")
        if self.render_paths or self.manifest_paths:
            raise TimelineDriverError("timeline-driver slice cannot write render/manifest paths")



@dataclass(frozen=True)
class DroidCoverage:
    """DROID measurement coverage and the source-keyed <=256 session layout."""

    source_frame_count: int
    submitted_count: int  # unique selected source frames
    pose_valid: tuple[bool, ...]
    pose_sampled: tuple[bool, ...] | None = None
    chunk_source_indices: tuple[tuple[int, ...], ...] = ()
    droid_fps: float | None = None

    def __post_init__(self) -> None:
        if self.source_frame_count <= 0 or len(self.pose_valid) != self.source_frame_count:
            raise TimelineDriverError("DROID coverage must provide one validity value per source frame")
        sampled = self.pose_sampled
        if sampled is None:
            raise TimelineDriverError("DROID coverage requires an explicit source-keyed sampled mask")
        if len(sampled) != self.source_frame_count or not all(self.pose_valid):
            raise TimelineDriverError("DROID coverage requires all-valid dense poses and N-frame sampled mask")
        if self.submitted_count != sum(sampled) or self.submitted_count <= 0:
            raise TimelineDriverError("DROID coverage unique count must equal sampled-mask count")
        if not sampled[0] or not sampled[-1] or (self.droid_fps is not None and self.droid_fps <= 0):
            raise TimelineDriverError("DROID coverage requires endpoint source samples and a valid explicit target FPS")
        if self.droid_fps is None and not all(sampled):
            raise TimelineDriverError("droid_fps=None owns every source frame")
        if not self.chunk_source_indices or any(not chunk or len(chunk) > DROID_SERVICE_PUSH_CAPACITY for chunk in self.chunk_source_indices):
            raise TimelineDriverError("DROID sessions must be nonempty and admit at most 256 pushes")
        flattened = tuple(index for n, chunk in enumerate(self.chunk_source_indices) for index in (chunk if n == 0 else chunk[DROID_SIM3_OVERLAP_FRAMES:]))
        selected = tuple(index for index, value in enumerate(sampled) if value)
        if flattened != selected:
            raise TimelineDriverError("DROID sessions must preserve exact source keys with only canonical Sim(3) overlap")
        for left, right in zip(self.chunk_source_indices, self.chunk_source_indices[1:]):
            if len(left) < DROID_SIM3_OVERLAP_FRAMES or len(right) <= DROID_SIM3_OVERLAP_FRAMES or right[:DROID_SIM3_OVERLAP_FRAMES] != left[-DROID_SIM3_OVERLAP_FRAMES:]:
                raise TimelineDriverError("consecutive DROID sessions require the fixed exact canonical Sim(3) overlap")

    @property
    def scheduled(self) -> bool:
        return True

    @property
    def partial(self) -> bool:
        return False

    @property
    def actual_pushed_count(self) -> int:
        return sum(len(chunk) for chunk in self.chunk_source_indices)

    def to_wire(self) -> dict[str, object]:
        return {
            "status": "completed_source_keyed_session_dag",
            "reason": "source_keyed_sessions_fixed_64_frame_overlap_then_sim3_merge",
            "source_frame_count": self.source_frame_count,
            "effective_unique_coverage_count": self.submitted_count,
            "actual_pushed_count": self.actual_pushed_count,
            "overlap_push_count": self.actual_pushed_count - self.submitted_count,
            "sampled_count": self.submitted_count,
            "interpolated_count": self.source_frame_count - self.submitted_count,
            "droid_fps": self.droid_fps,
            "chunk_source_indices": [list(chunk) for chunk in self.chunk_source_indices],
            "droid_pose_valid": list(self.pose_valid),
            "droid_pose_sampled": list(self.pose_sampled or ()),
            "pose_validity": "per-frame source-keyed mask; all true after endpoint-inclusive SE3 interpolation",
            "pose_sampled": "per-frame source-keyed mask; true only where DROID received a source frame",
            "session_policy": "at_most_256_pushes_fixed_64_frame_exact_source_overlap",
            "scale_mode": "up_to_scale_monocular",
            "acceptance": False,
        }


REQUIRED_STAGE_IDS: tuple[str, ...] = (
    "unidepth.infer",
    "hands.detect",
    "wilor.reconstruct",
    "droid.create_session",
    "droid.push_frame",
    "droid.finalize",
    "hawor.infer_tracks",
    "hawor_infiller.fill",
)


@dataclass
class _MutableTrack:
    track_id: str
    side: HandSide
    detections: list[HandDetectionRecord]
    last_center: np.ndarray
    velocity: np.ndarray
    last_box: np.ndarray
    last_frame: int


def _geometry_anomaly_codes(vertices: np.ndarray | None, joints: np.ndarray | None) -> tuple[str, ...]:
    if vertices is None or joints is None:
        return ("missing_surface_geometry",)
    vertex_array = np.asarray(vertices)
    joint_array = np.asarray(joints)
    anomalies: list[str] = []
    vertices_finite = bool(np.isfinite(vertex_array).all())
    joints_finite = bool(np.isfinite(joint_array).all())
    if not vertices_finite:
        anomalies.append("nonfinite_vertices")
    if not joints_finite:
        anomalies.append("nonfinite_joints")
    if vertices_finite:
        if not np.any(vertex_array != 0):
            anomalies.append("all_zero_vertices")
        elif np.max(np.std(vertex_array.astype(np.float64), axis=0)) <= 1e-8:
            anomalies.append("collapsed_vertices")
    if joints_finite and joint_array.size and float(np.max(np.abs(joint_array))) > 100.0:
        anomalies.append("implausible_joint_magnitude_gt_100m")
    return tuple(anomalies)


def _parameter_anomaly_codes(
    root_orient: np.ndarray,
    hand_pose: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    uncertainty: float,
) -> tuple[str, ...]:
    """Check MANO parameter validity for infiller/state eligibility.

    root_orient: [3,3] or [16,3,3] rotation matrix
    hand_pose: [15,3,3] or [16,15,3,3] rotation matrix
    trans: [3] or [16,3] translation in metres
    betas: [10] or [16,10]
    """
    anomalies: list[str] = []

    root = np.asarray(root_orient, dtype=np.float64)
    if not np.isfinite(root).all():
        anomalies.append("nonfinite_root_orient")

    hp = np.asarray(hand_pose, dtype=np.float64)
    if not np.isfinite(hp).all():
        anomalies.append("nonfinite_hand_pose")

    t = np.asarray(trans, dtype=np.float64)
    if not np.isfinite(t).all():
        anomalies.append("nonfinite_translation")
    elif t.size and float(np.max(np.abs(t))) > 100.0:
        anomalies.append("implausible_translation_gt_100m")

    b = np.asarray(betas, dtype=np.float64)
    if not np.isfinite(b).all():
        anomalies.append("nonfinite_betas")

    if not (np.isfinite(uncertainty) and uncertainty >= 0.0):
        anomalies.append("nonfinite_or_negative_uncertainty")

    # Match HaWoR's SO(3) tolerance for root orientation validation.
    if np.isfinite(root).all() and root.ndim >= 2 and root.shape[-1] == 3 and root.shape[-2] == 3:
        root_3d = root.reshape(-1, 3, 3)
        for i in range(root_3d.shape[0]):
            rotation = root_3d[i]
            det = float(np.linalg.det(rotation))
            ortho = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
            if det <= 0 or ortho > 2e-4:
                anomalies.append(f"invalid_root_orient_rotation_frame_{i}")
                break

    return tuple(anomalies)


@dataclass(frozen=True)
class _ManoCandidate:
    frame_index: int
    side: HandSide
    root_orient: np.ndarray
    hand_pose: np.ndarray
    betas: np.ndarray
    trans: np.ndarray
    vertices: np.ndarray | None
    joints: np.ndarray | None
    observed: bool
    inferred: bool
    uncertainty: float
    source_stage: str
    source_scope: str
    vertices_source_px: np.ndarray | None = None
    joints_source_px: np.ndarray | None = None

    @property
    def geometry_anomaly_codes(self) -> tuple[str, ...]:
        return _geometry_anomaly_codes(self.vertices, self.joints)

    @property
    def parameter_anomaly_codes(self) -> tuple[str, ...]:
        return _parameter_anomaly_codes(
            self.root_orient,
            self.hand_pose,
            self.trans,
            self.betas,
            self.uncertainty,
        )

    @property
    def has_valid_parameters(self) -> bool:
        return not self.parameter_anomaly_codes

    @property
    def has_valid_surface_geometry(self) -> bool:
        return not self.geometry_anomaly_codes

    @property
    def has_finite_geometry(self) -> bool:
        return not self.geometry_anomaly_codes

    @property
    def is_eligible(self) -> bool:
        """Hard eligibility requires both valid parameters and surface geometry."""
        return self.has_valid_parameters and self.has_valid_surface_geometry

    @property
    def rank(self) -> tuple[int, int, int, float]:
        # Ineligible candidates remain selectable for diagnostics when no valid
        # candidate exists, but can never displace an eligible candidate.
        return (
            0 if self.is_eligible else 1,
            0 if self.observed else 1,
            0 if self.has_finite_geometry else 1,
            float(self.uncertainty),
        )


def _droid_input_shape_yx(timeline: SourceTimeline, config: FullVideoDriverConfig) -> tuple[int, int]:
    """Return the one model-grid input geometry used by DROID create and pushes."""

    if config.droid_input_shape_yx is not None:
        return tuple(int(value) for value in config.droid_input_shape_yx)
    scale = math.sqrt(config.droid_target_area_px / float(timeline.width_px * timeline.height_px))
    provisional_h = max(8, int(timeline.height_px * scale))
    provisional_w = max(8, int(timeline.width_px * scale))
    return max(8, provisional_h - provisional_h % 8), max(8, provisional_w - provisional_w % 8)


def _droid_session_buffer(submitted_frame_count: int) -> int:
    """Allocate no more than the 256-push session chunk actually needed."""

    if submitted_frame_count <= 0:
        raise TimelineDriverError("DROID submitted schedule must be nonempty")
    return min(DROID_SERVICE_PUSH_CAPACITY, submitted_frame_count)


def _scheduled_droid_options(
    *,
    chunk_index: int,
    submitted_frame_count: int,
    droid_fps: float,
    attempt: int,
    filter_thresh: float | None,
    frontend_thresh: float | None = None,
    backend_thresh: float | None = None,
) -> dict[str, str | int | float | bool]:
    if submitted_frame_count <= 0 or submitted_frame_count > DROID_SERVICE_PUSH_CAPACITY:
        raise TimelineDriverError("scheduled DROID chunk must remain within the 256-push capacity")
    options: dict[str, str | int | float | bool] = {
        "scheduled_chunk": chunk_index,
        "buffer": DROID_SERVICE_PUSH_CAPACITY,
        "droid_fps": float(droid_fps),
        "attempt": attempt,
    }
    if filter_thresh is not None:
        options["filter_thresh"] = float(filter_thresh)
        options["bounded_lower_filter_retry"] = True
    if frontend_thresh is not None:
        options["frontend_thresh"] = float(frontend_thresh)
    if backend_thresh is not None:
        options["backend_thresh"] = float(backend_thresh)
    return options


def _droid_prefix_coverage(frame_count: int) -> DroidCoverage:
    """Compatibility name for complete source-keyed session planning."""
    indices = tuple(range(frame_count))
    chunks = _droid_chunks_with_overlap(indices)
    return DroidCoverage(frame_count, frame_count, (True,) * frame_count, (True,) * frame_count, chunks, None)


def _droid_stride(source_fps: float, droid_fps: float) -> int:
    if source_fps <= 0 or droid_fps <= 0:
        raise TimelineDriverError("source and target DROID FPS must be positive")
    return max(1, int(math.floor(source_fps / droid_fps + 0.5)))


def _droid_sample_source_indices(timeline: SourceTimeline, droid_fps: float) -> tuple[int, ...]:
    """Uniform stride schedule whose final source endpoint is never dropped."""
    stride = _droid_stride(timeline.fps, droid_fps)
    indices = list(range(0, timeline.frame_count, stride))
    if indices[-1] != timeline.frame_count - 1:
        indices.append(timeline.frame_count - 1)
    return tuple(indices)


def _droid_chunks_with_overlap(indices: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    if not indices or tuple(sorted(set(indices))) != tuple(indices):
        raise TimelineDriverError("DROID sampled indices must be strictly increasing and nonempty")
    chunks: list[tuple[int, ...]] = []
    start = 0
    while start < len(indices):
        end = min(len(indices), start + DROID_SERVICE_PUSH_CAPACITY)
        chunks.append(tuple(int(index) for index in indices[start:end]))
        if end == len(indices):
            break
        start = end - DROID_SIM3_OVERLAP_FRAMES
    return tuple(chunks)


def _droid_scheduled_coverage(timeline: SourceTimeline, droid_fps: float) -> DroidCoverage:
    sampled_indices = _droid_sample_source_indices(timeline, droid_fps)
    sampled = np.zeros(timeline.frame_count, dtype=bool)
    sampled[list(sampled_indices)] = True
    return DroidCoverage(
        source_frame_count=timeline.frame_count,
        submitted_count=len(sampled_indices),
        pose_valid=(True,) * timeline.frame_count,
        pose_sampled=tuple(bool(value) for value in sampled),
        chunk_source_indices=_droid_chunks_with_overlap(sampled_indices),
        droid_fps=droid_fps,
    )


def _droid_validity_from_output(output: DroidFinalizeOutput, frame_count: int) -> np.ndarray:
    raw = output.T_world_camera.provenance.get("droid_pose_valid")
    values = np.asarray(raw, dtype=bool) if isinstance(raw, (list, tuple, np.ndarray)) else np.isfinite(output.T_world_camera.array).all(axis=(1, 2))
    if values.shape != (frame_count,):
        raise StageResultError("DROID pose validity mask does not cover the source timeline")
    return values


def _selected_source_indices(timeline: SourceTimeline, target_fps: float | None) -> tuple[int, ...]:
    if target_fps is None:
        return timeline.frame_indices
    if target_fps <= 0:
        raise TimelineDriverError("sampling target FPS must be positive")
    count = max(1, int(math.ceil((timeline.frame_count / timeline.fps) * target_fps - 1e-9)))
    selected: list[int] = []
    for step in range(count):
        timestamp = float(step) / float(target_fps)
        index = min(timeline.frame_count - 1, max(0, int(math.floor(timestamp * timeline.fps + 0.5))))
        if not selected or selected[-1] != index:
            selected.append(index)
    return tuple(selected)


def _nearest_unidepth_record(records: Sequence[AlgorithmResult[UniDepthOutput]], frame_index: int) -> AlgorithmResult[UniDepthOutput]:
    if not records:
        raise StageResultError("at least one UniDepth record is required")
    return min(records, key=lambda result: abs(int(result.output.frame_indices[0]) - int(frame_index)))


def _unidepth_hand_depth_m(
    record: AlgorithmResult[UniDepthOutput],
    source_uv: np.ndarray,
    detection: HandDetectionRecord,
) -> float | None:
    """Robustly sample metric depth around a projected WiLoR wrist."""

    output = record.output
    depth = np.asarray(output.depth_m.array[0], dtype=np.float32)
    confidence = np.asarray(output.confidence.array[0], dtype=np.float32)
    if depth.ndim != 2 or confidence.shape != depth.shape:
        raise StageResultError("UniDepth hand scale source must be aligned 2D depth/confidence")
    source_to_depth = np.linalg.inv(np.asarray(output.spatial.pixel_to_source, dtype=np.float64))
    uv = np.asarray(source_uv, dtype=np.float64).reshape(2)
    q = source_to_depth @ np.asarray([uv[0], uv[1], 1.0], dtype=np.float64)
    q = q[:2] / q[2]
    box = np.asarray(detection.box_xyxy_source, dtype=np.float64)
    box_scale = max(float(box[2] - box[0]), float(box[3] - box[1]), 2.0)
    depth_per_source_x = abs(float(source_to_depth[0, 0]))
    depth_per_source_y = abs(float(source_to_depth[1, 1]))
    radius_x = max(1, int(round(0.08 * box_scale * depth_per_source_x)))
    radius_y = max(1, int(round(0.08 * box_scale * depth_per_source_y)))
    center_x, center_y = int(round(float(q[0]))), int(round(float(q[1])))
    x0, x1 = max(0, center_x - radius_x), min(depth.shape[1], center_x + radius_x + 1)
    y0, y1 = max(0, center_y - radius_y), min(depth.shape[0], center_y + radius_y + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    local_depth = depth[y0:y1, x0:x1]
    local_confidence = confidence[y0:y1, x0:x1]
    valid = np.isfinite(local_depth) & (local_depth > 0.05) & (local_depth < 20.0) & np.isfinite(local_confidence) & (local_confidence > 0.0)
    values = local_depth[valid]
    return None if not values.size else float(np.median(values))


def _camera_points_to_world(points_camera: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    """Apply the raw DROID world-from-camera convention once.

    For row-wise points this is exactly ``P_cam @ R_c2w.T + t_c2w``;
    translation-scale alignment must happen before this join and only on the
    camera pose translation.
    """
    points = np.asarray(points_camera, dtype=np.float32)
    transform = np.asarray(T_world_camera, dtype=np.float32)
    if points.shape[-1] != 3 or transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise StageResultError("world hand transform requires finite points[...,3] and T_world_camera[4,4]")
    return np.ascontiguousarray(points @ transform[:3, :3].T + transform[:3, 3])


def _metric_wilor_geometry(
    vertices_root: np.ndarray,
    joints_root: np.ndarray,
    joints_source_px: np.ndarray,
    depth_m: float | None,
    k_source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Place canonical MANO geometry at UniDepth wrist depth in source camera."""

    if depth_m is None or not np.isfinite(depth_m) or depth_m <= 0.0:
        return None
    vertices = np.asarray(vertices_root, dtype=np.float32)
    joints = np.asarray(joints_root, dtype=np.float32)
    projected = np.asarray(joints_source_px, dtype=np.float32)
    k = np.asarray(k_source, dtype=np.float64)
    if vertices.shape != (778, 3) or joints.ndim != 2 or joints.shape[1] != 3 or projected.shape != (len(joints), 2):
        raise StageResultError("WiLoR metric lift requires matching root-relative MANO and source projections")
    wrist_uv = projected[0]
    wrist_ray = np.linalg.inv(k) @ np.asarray([wrist_uv[0], wrist_uv[1], 1.0], dtype=np.float64)
    if not np.isfinite(wrist_ray).all() or wrist_ray[2] <= 1e-8:
        return None
    wrist_camera = (wrist_ray * (depth_m / wrist_ray[2])).astype(np.float32)
    translation = wrist_camera - joints[0]
    return vertices + translation[None, :], joints + translation[None, :], translation


def _densify_droid_tensor(tensor: TypedTensor, source_indices: tuple[int, ...], frame_count: int) -> TypedTensor:
    array = np.asarray(tensor.array)
    if array.shape[0] != len(source_indices):
        raise StageResultError("DROID sparse output length does not match submitted source schedule")
    selected = np.asarray(source_indices, dtype=np.int64)
    targets = np.arange(frame_count, dtype=np.int64)
    nearest = np.abs(targets[:, None] - selected[None, :]).argmin(axis=1)
    dense = np.ascontiguousarray(array[nearest])
    provenance = dict(tensor.provenance)
    provenance.update({"sampling": "nearest_source_frame_fill", "measured_source_frame_indices": list(source_indices), "inferred_full_timeline": True})
    return TypedTensor(dense, tensor.units, tensor.coordinate_frame, tensor.tensor_index_order, tensor.semantic_tag, provenance, tensor.pixel_transform)



def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return a normalized [x,y,z,w] quaternion for a finite SO(3) matrix."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise StageResultError("DROID rotation is not finite 3x3")
    trace = float(np.trace(matrix))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([(matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s, 0.25 * s])
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array([0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[2, 1] - matrix[1, 2]) / s])
        elif axis == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array([(matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s, (matrix[0, 2] - matrix[2, 0]) / s])
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array([(matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s, (matrix[1, 0] - matrix[0, 1]) / s])
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0 or not np.isfinite(norm):
        raise StageResultError("DROID rotation cannot form a quaternion")
    return quaternion / norm


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _slerp_rotation(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _rotation_to_quaternion(left)
    q1 = _rotation_to_quaternion(right)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        quaternion = q0 + alpha * (q1 - q0)
        quaternion /= np.linalg.norm(quaternion)
    else:
        theta = math.acos(max(-1.0, min(1.0, dot)))
        sine = math.sin(theta)
        quaternion = (math.sin((1.0 - alpha) * theta) / sine) * q0 + (math.sin(alpha * theta) / sine) * q1
    return _quaternion_to_rotation(quaternion)


def _rotation_distance_rad(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left, dtype=np.float64).T @ np.asarray(right, dtype=np.float64)
    return float(math.acos(max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) / 2.0))))



def _interpolate_droid_poses(sampled_indices: Sequence[int], sampled_poses: np.ndarray, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    indices = tuple(int(index) for index in sampled_indices)
    poses = np.asarray(sampled_poses, dtype=np.float64)
    if not indices or indices[0] != 0 or indices[-1] != frame_count - 1 or tuple(sorted(set(indices))) != indices:
        raise StageResultError("DROID interpolation requires increasing endpoint-inclusive source samples")
    if poses.shape != (len(indices), 4, 4) or not np.isfinite(poses).all():
        raise StageResultError("DROID sampled poses do not match sampled source schedule")
    dense = np.empty((frame_count, 4, 4), dtype=np.float32)
    sampled_mask = np.zeros(frame_count, dtype=bool)
    if len(indices) == 1:
        dense[0] = poses[0].astype(np.float32)
    for left_slot, (left_index, right_index) in enumerate(zip(indices[:-1], indices[1:])):
        left, right = poses[left_slot], poses[left_slot + 1]
        for frame_index in range(left_index, right_index + 1):
            alpha = (frame_index - left_index) / float(right_index - left_index)
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = _slerp_rotation(left[:3, :3], right[:3, :3], alpha)
            pose[:3, 3] = (1.0 - alpha) * left[:3, 3] + alpha * right[:3, 3]
            dense[frame_index] = pose.astype(np.float32)
    sampled_mask[list(indices)] = True
    if not np.isfinite(dense).all():
        raise StageResultError("endpoint-inclusive DROID interpolation left nonfinite source poses")
    return dense, sampled_mask




class _TimingClientProxy:
    """Cosmos semantic lane adapter that preserves request timing attribution."""
    def __init__(self, driver: "FullVideoTimelineDriver") -> None:
        self._driver = driver

    def execute(self, request: AlgorithmRequest[Any]) -> AlgorithmResult[Any]:
        return self._driver._execute_timed(request)




def _droid_temporal_qc(world: np.ndarray, valid: np.ndarray) -> Mapping[str, object]:
    finite_indices = [int(i) for i in np.flatnonzero(valid)]
    if len(finite_indices) < 2:
        return {"status": "insufficient_valid_pairs", "max_one_step_rotation_rad": None}

    def rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
        relative = left.T @ right
        cosine = max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) / 2.0))
        return float(math.acos(cosine))

    one_step: list[float] = []
    consecutive_indices: list[int] = []
    for left, right in zip(finite_indices[:-1], finite_indices[1:]):
        if right != left + 1:
            continue
        consecutive_indices.append(left)
        one_step.append(rotation_distance(world[left, :3, :3], world[right, :3, :3]))
    maximum = max(one_step) if one_step else None
    if maximum is not None and maximum > 1.0:
        raise StageResultError(
            f"DROID temporal QC rejected physically impossible source-frame rotation {maximum:.6f} rad"
        )
    parity_run = longest_run = 0
    for frame_index in range(len(world) - 2):
        if not (valid[frame_index] and valid[frame_index + 1] and valid[frame_index + 2]):
            parity_run = 0
            continue
        first = rotation_distance(world[frame_index, :3, :3], world[frame_index + 1, :3, :3])
        second = rotation_distance(world[frame_index + 1, :3, :3], world[frame_index + 2, :3, :3])
        two_step = rotation_distance(world[frame_index, :3, :3], world[frame_index + 2, :3, :3])
        if first > 0.20 and second > 0.20 and two_step < 0.10:
            parity_run += 1
            longest_run = max(longest_run, parity_run)
        else:
            parity_run = 0
    if longest_run >= 4:
        raise StageResultError(
            f"DROID temporal QC rejected sustained parity-interleaved trajectory run={longest_run}"
        )
    return {
        "status": "passed",
        "max_one_step_rotation_rad": maximum,
        "valid_pair_count": len(one_step),
        "longest_parity_return_run": longest_run,
    }

def _droid_keyframe_source_indices(output: DroidFinalizeOutput, selected: tuple[int, ...]) -> tuple[int, ...] | None:
    """Decode service mapping records or numeric string IDs without offset guesses."""
    count = int(output.disparities.shape[0])
    for provenance in (output.disparities.provenance, output.T_world_camera.provenance, output.scale_provenance):
        for key in ("keyframe_mapping", "keyframe_source_indices", "keyframe_frame_indices", "source_frame_indices"):
            mapping = _mapping_source_indices(provenance.get(key), expected_count=count, allowed=selected)
            if mapping is not None:
                return mapping
    return None


def _estimate_metric3d_droid_scale(
    output: DroidFinalizeOutput,
    timeline: SourceTimeline,
    selected: tuple[int, ...],
    dynamic_masks: Mapping[int, np.ndarray],
    source: FrameSource,
    canonical_k: np.ndarray,
    *,
    hawor_root: str | None,
    metric_checkpoint: str | None,
    metric3d_python: str,
    cuda_visible_devices: str,
) -> tuple[float, Mapping[str, object]]:
    if not hawor_root or not metric_checkpoint:
        raise StageResultError("Metric3D DROID scale paths are not configured")
    keyframes = _droid_keyframe_source_indices(output, selected)
    if keyframes is None:
        raise StageResultError("Metric3D DROID scale requires keyframe_mapping source_frame_id rows")
    try:
        import cv2
    except Exception as exc:
        raise StageResultError(f"Metric3D bridge requires OpenCV: {type(exc).__name__}: {exc}") from exc
    helper = Path(__file__).resolve().parents[1] / "scripts" / "estimate_droid_metric3d_scale_bridge.py"
    if not helper.is_file() or not Path(metric3d_python).is_file():
        raise StageResultError("Metric3D bridge script or pinned interpreter is unavailable")
    with tempfile.TemporaryDirectory(prefix="manager-droid-metric3d-") as temp:
        root = Path(temp)
        image_dir = root / "images"
        image_dir.mkdir()
        for frame_index in timeline.frame_indices:
            path = image_dir / f"{frame_index:08d}.jpg"
            rgb = np.asarray(source.read_rgb(frame_index), dtype=np.uint8)
            if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise StageResultError(f"Metric3D could not materialize source frame {frame_index}")
        # Both the reference estimator and DROID BA use positive=dynamic/ignore.
        dynamic_ignore_masks = np.zeros((timeline.frame_count, *next(iter(dynamic_masks.values())).shape), dtype=np.uint8)
        for frame_index, mask in dynamic_masks.items():
            dynamic_ignore_masks[int(frame_index)] = np.asarray(mask, dtype=np.uint8)
        geometry_path = root / "geometry.npz"
        masks_path = root / "dynamic_masks.npy"
        calib_path = root / "calib.npy"
        output_path = root / "scale.json"
        np.savez_compressed(
            geometry_path,
            frame_idx=np.arange(timeline.frame_count, dtype=np.int32),
            tstamp=np.asarray(keyframes, dtype=np.int32),
            disps=np.asarray(output.disparities.array, dtype=np.float32),
        )
        np.save(masks_path, dynamic_ignore_masks, allow_pickle=False)
        K = np.asarray(canonical_k, dtype=np.float32)
        np.save(calib_path, np.asarray([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32), allow_pickle=False)
        command = [
            metric3d_python, str(helper), "--geometry", str(geometry_path),
            "--image-dir", str(image_dir), "--masks", str(masks_path),
            "--calib", str(calib_path), "--hawor-root", hawor_root,
            "--metric-checkpoint", metric_checkpoint, "--output", str(output_path),
        ]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        completed = subprocess.run(command, cwd=str(helper.parents[1]), env=environment, capture_output=True, text=True)
        if completed.returncode != 0 or not output_path.is_file():
            detail = (completed.stderr or completed.stdout or "Metric3D bridge failed")[-4000:]
            raise StageResultError(f"Metric3D DROID scale estimation failed: {detail}")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    scalar = float(payload["scale"])
    if not np.isfinite(scalar) or scalar <= 0:
        raise StageResultError("Metric3D DROID scale must be finite and positive")
    report = dict(payload["report"])
    report["bridge_python"] = metric3d_python
    report["cuda_visible_devices"] = cuda_visible_devices
    report["mask_semantics"] = "positive=dynamic_ignore"
    return scalar, report


def _estimate_metric3d_droid_scales(
    outputs: Sequence[DroidFinalizeOutput],
    chunks: Sequence[tuple[int, ...]],
    timeline: SourceTimeline,
    dynamic_masks: Mapping[int, np.ndarray],
    source: FrameSource,
    canonical_k: np.ndarray,
    *,
    hawor_root: str | None,
    metric_checkpoint: str | None,
    metric3d_python: str,
    cuda_visible_devices: str,
) -> tuple[tuple[float, ...], tuple[Mapping[str, object], ...]]:
    """Estimate all session scales in one shared Metric3D depth pass.

    The bridge receives each session's exact keyframe mapping, but source RGB
    frames, dynamic masks, Metric3D construction, and Metric3D depth inference
    occur once for the video.  A failure is terminal: falling back to one
    subprocess per session would silently reintroduce the long-video cost this
    path exists to avoid.
    """
    if len(outputs) != len(chunks) or not outputs:
        raise StageResultError("Metric3D DROID session output/chunk cardinality is invalid")
    if not hawor_root or not metric_checkpoint:
        raise StageResultError("Metric3D DROID scale paths are not configured")
    session_geometries: list[tuple[tuple[int, ...], np.ndarray]] = []
    for session_index, (output, chunk) in enumerate(zip(outputs, chunks)):
        keyframes = _droid_keyframe_source_indices(output, chunk)
        disparities = np.asarray(output.disparities.array, dtype=np.float32)
        if keyframes is None or disparities.shape[0] != len(keyframes) or not np.isfinite(disparities).all():
            raise StageResultError(f"DROID session {session_index} lacks valid exact keyframe Metric3D evidence")
        session_geometries.append((keyframes, disparities))
    try:
        import cv2
    except Exception as exc:
        raise StageResultError(f"Metric3D bridge requires OpenCV: {type(exc).__name__}: {exc}") from exc
    helper = Path(__file__).resolve().parents[1] / "scripts" / "estimate_droid_metric3d_scale_bridge.py"
    if not helper.is_file() or not Path(metric3d_python).is_file():
        raise StageResultError("Metric3D bridge script or pinned interpreter is unavailable")
    if not set(dynamic_masks).issubset(set(timeline.frame_indices)) or not dynamic_masks:
        raise StageResultError("Metric3D DROID scales require source-keyed dynamic masks")
    with tempfile.TemporaryDirectory(prefix="manager-droid-metric3d-multi-") as temp:
        root = Path(temp)
        image_dir = root / "images"
        image_dir.mkdir()
        for frame_index in timeline.frame_indices:
            path = image_dir / f"{frame_index:08d}.jpg"
            rgb = np.asarray(source.read_rgb(frame_index), dtype=np.uint8)
            if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise StageResultError(f"Metric3D could not materialize source frame {frame_index}")
        mask_shape = np.asarray(next(iter(dynamic_masks.values()))).shape
        dynamic_ignore_masks = np.zeros((timeline.frame_count, *mask_shape), dtype=np.uint8)
        for frame_index, mask in dynamic_masks.items():
            value = np.asarray(mask, dtype=np.uint8)
            if value.shape != mask_shape:
                raise StageResultError("Metric3D DROID dynamic masks have inconsistent source grids")
            dynamic_ignore_masks[int(frame_index)] = value
        if not np.isfinite(dynamic_ignore_masks).all() or not np.all((dynamic_ignore_masks == 0) | (dynamic_ignore_masks == 1)):
            raise StageResultError("Metric3D DROID dynamic masks must be finite binary positive=dynamic_ignore")
        geometry_path = root / "multi_geometry.npz"
        geometry_payload: dict[str, np.ndarray] = {
            "frame_idx": np.asarray(timeline.frame_indices, dtype=np.int32),
            "session_count": np.asarray(len(session_geometries), dtype=np.int32),
        }
        for session_index, (keyframes, disparities) in enumerate(session_geometries):
            geometry_payload[f"session_{session_index}_tstamp"] = np.asarray(keyframes, dtype=np.int32)
            geometry_payload[f"session_{session_index}_disps"] = disparities
        masks_path = root / "dynamic_masks.npy"
        calib_path = root / "calib.npy"
        output_path = root / "scales.json"
        np.savez_compressed(geometry_path, **geometry_payload)
        np.save(masks_path, dynamic_ignore_masks, allow_pickle=False)
        K = np.asarray(canonical_k, dtype=np.float32)
        np.save(calib_path, np.asarray([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32), allow_pickle=False)
        command = [
            metric3d_python, str(helper), "--multi-geometry", str(geometry_path),
            "--image-dir", str(image_dir), "--masks", str(masks_path),
            "--calib", str(calib_path), "--hawor-root", hawor_root,
            "--metric-checkpoint", metric_checkpoint, "--output", str(output_path),
        ]
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        completed = subprocess.run(command, cwd=str(helper.parents[1]), env=environment, capture_output=True, text=True)
        if completed.returncode != 0 or not output_path.is_file():
            detail = (completed.stderr or completed.stdout or "Metric3D multi-session bridge failed")[-4000:]
            raise StageResultError(f"Metric3D DROID multi-session scale estimation failed: {detail}")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    rows = payload.get("sessions") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or len(rows) != len(outputs):
        raise StageResultError("Metric3D DROID multi-session bridge returned invalid session cardinality")
    shared = payload.get("shared_metric3d") if isinstance(payload, Mapping) else None
    if not isinstance(shared, Mapping) or shared.get("metric_model_load_count") != 1 or shared.get("metric_depth_pass_count") != timeline.frame_count:
        raise StageResultError("Metric3D DROID multi-session bridge did not attest one shared model/depth pass")
    scales: list[float] = []
    reports: list[Mapping[str, object]] = []
    for session_index, (row, (keyframes, _)) in enumerate(zip(rows, session_geometries)):
        if not isinstance(row, Mapping) or not isinstance(row.get("report"), Mapping):
            raise StageResultError(f"Metric3D DROID session {session_index} returned malformed evidence")
        scalar = float(row.get("scale", float("nan")))
        report = dict(row["report"])
        if not np.isfinite(scalar) or scalar <= 0 or report.get("exact_keyframe_source_ids") != list(keyframes):
            raise StageResultError(f"Metric3D DROID session {session_index} returned invalid scale or keyframe mapping")
        report.update({
            "bridge_python": metric3d_python,
            "cuda_visible_devices": cuda_visible_devices,
            "mask_semantics": "positive=dynamic_ignore",
            "shared_metric3d": dict(shared),
        })
        scales.append(scalar)
        reports.append(report)
    return tuple(scales), tuple(reports)


def _scale_droid_output_once(
    output: DroidFinalizeOutput,
    timeline: SourceTimeline,
    selected: tuple[int, ...],
    dynamic_masks: Mapping[int, np.ndarray],
    source: FrameSource,
    canonical_k: np.ndarray,
    *,
    hawor_root: str | None,
    metric_checkpoint: str | None,
    metric3d_python: str,
    cuda_visible_devices: str,
    metric_scalar: float | None = None,
    metric_report: Mapping[str, object] | None = None,
) -> DroidFinalizeOutput:
    if output.scale_provenance.get("translation_scale_applied"):
        return output
    if metric_scalar is None:
        scalar, report = _estimate_metric3d_droid_scale(
            output, timeline, selected, dynamic_masks, source, canonical_k,
            hawor_root=hawor_root, metric_checkpoint=metric_checkpoint,
            metric3d_python=metric3d_python, cuda_visible_devices=cuda_visible_devices,
        )
    else:
        scalar = float(metric_scalar)
        report = dict(metric_report or {})
        if not np.isfinite(scalar) or scalar <= 0:
            raise StageResultError("precomputed Metric3D DROID scale must be finite and positive")
    world = np.asarray(output.T_world_camera.array, dtype=np.float32).copy()
    valid_world = np.isfinite(world).all(axis=(1, 2))
    world[valid_world, :3, 3] *= np.float32(scalar)
    camera = np.full_like(world, np.nan)
    for frame_index in np.flatnonzero(valid_world):
        camera[frame_index] = np.linalg.inv(world[frame_index]).astype(np.float32)
    provenance = {
        **dict(output.scale_provenance), "translation_scale_applied": True,
        "scale_source": "Metric3D_est_scale_hybrid", "scale": scalar,
        "scale_provenance": "droid_local_world_scaled_by_metric3d_scalar",
        "scale_report": dict(report), "scale_applied_to": "translation_only_once",
    }
    def dt(t: TypedTensor, arr: np.ndarray) -> TypedTensor:
        return TypedTensor(arr, t.units, t.coordinate_frame, t.tensor_index_order, t.semantic_tag, {**dict(t.provenance), **provenance}, t.pixel_transform)
    return replace(output, T_world_camera=dt(output.T_world_camera, world), T_camera_world=dt(output.T_camera_world, camera), scale_provenance=provenance)


def _mapping_source_indices(raw: object, *, expected_count: int, allowed: Sequence[int]) -> tuple[int, ...] | None:
    """Parse service mapping records or string IDs without inferring slot offsets."""
    if not isinstance(raw, (list, tuple)) or len(raw) != expected_count:
        return None
    values: list[int] = []
    for row in raw:
        value = (row.get("source_frame_id", row.get("source_frame_index")) if isinstance(row, Mapping) else row)
        if isinstance(value, bool):
            return None
        if isinstance(value, str):
            text = value.strip()
            # Services may return either a numeric canonical ID or the exact
            # SourceFrame ``<source>:frame:<zero-padded-index>`` string.
            if ":frame:" in text:
                _, _, text = text.rpartition(":frame:")
            if not text or not text.isdecimal():
                return None
            parsed = int(text)
        elif isinstance(value, (int, np.integer)):
            parsed = int(value)
        else:
            return None
        values.append(parsed)
    result = tuple(values)
    return result if tuple(sorted(set(result))) == result and all(value in allowed for value in result) else None


def _droid_pose_source_indices(output: DroidFinalizeOutput, expected: tuple[int, ...]) -> tuple[int, ...] | None:
    pose_count = int(output.T_world_camera.shape[0])
    for provenance in (output.T_world_camera.provenance, output.T_camera_world.provenance, output.scale_provenance):
        for key in ("dense_mapping", "dense_source_indices", "pose_mapping", "pose_source_indices", "keyframe_mapping", "keyframe_source_indices"):
            mapping = _mapping_source_indices(provenance.get(key), expected_count=pose_count, allowed=expected)
            if mapping is not None:
                return mapping
    return None


def _validate_rigid_pose_array(poses: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(poses, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (4, 4) or not np.isfinite(array).all():
        raise StageResultError(f"{label} poses must be finite [N,4,4]")
    if not np.allclose(array[:, 3, :], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-5, rtol=0.0):
        raise StageResultError(f"{label} poses have invalid homogeneous row")
    for rotation in array[:, :3, :3]:
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0.0) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4, rtol=0.0):
            raise StageResultError(f"{label} poses contain non-SO(3) rotation")
    return array


def _anchor_stitch_drift_diagnostics(
    global_poses: Mapping[int, np.ndarray],
    local_poses: Mapping[int, np.ndarray],
    overlap: tuple[int, ...],
    *,
    anchor_source_id: int,
    session_index: int,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    """Rigidly align at one exact anchor and measure, never fit, overlap drift."""
    if anchor_source_id not in global_poses or anchor_source_id not in local_poses:
        raise StageResultError(f"DROID session {session_index} anchor source ID is unavailable")
    anchor_global, anchor_local = global_poses[anchor_source_id], local_poses[anchor_source_id]
    rotation = anchor_global[:3, :3] @ anchor_local[:3, :3].T
    translation = anchor_global[:3, 3] - rotation @ anchor_local[:3, 3]
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise StageResultError(f"DROID session {session_index} anchor transform is nonfinite")
    global_centers = np.stack([global_poses[key][:3, 3] for key in overlap]).astype(np.float64)
    local_centers = np.stack([local_poses[key][:3, 3] for key in overlap]).astype(np.float64)
    pairs = [(left, right) for left in range(len(overlap)) for right in range(left + 1, len(overlap))]
    local_distances = np.asarray([np.linalg.norm(local_centers[right] - local_centers[left]) for left, right in pairs])
    global_distances = np.asarray([np.linalg.norm(global_centers[right] - global_centers[left]) for left, right in pairs])
    usable = (local_distances > DROID_SIM3_MIN_BASELINE) & (global_distances > DROID_SIM3_MIN_BASELINE)
    if int(np.count_nonzero(usable)) < 2:
        raise StageResultError(f"DROID session {session_index} anchor overlap has degenerate camera-center baseline")
    local_baseline = float(np.median(local_distances[usable]))
    global_baseline = float(np.median(global_distances[usable]))
    predicted_centers = (rotation @ local_centers.T).T + translation
    translation_errors = np.linalg.norm(predicted_centers - global_centers, axis=1)
    aligned_rotations = np.einsum("ij,njk->nik", rotation, np.stack([local_poses[key][:3, :3] for key in overlap]))
    rotation_errors = np.asarray([_rotation_distance_rad(aligned, global_poses[key][:3, :3]) for aligned, key in zip(aligned_rotations, overlap)])
    held_indices = [index for index, key in enumerate(overlap) if key != anchor_source_id]
    held_translation = translation_errors[held_indices]
    held_rotation = rotation_errors[held_indices]
    endpoint_global = global_centers[-1] - global_centers[0]
    endpoint_local = local_centers[-1] - local_centers[0]
    local_norm, global_norm = float(np.linalg.norm(endpoint_local)), float(np.linalg.norm(endpoint_global))
    angle = None
    if local_norm > DROID_SIM3_MIN_BASELINE and global_norm > DROID_SIM3_MIN_BASELINE:
        cosine = float(np.dot(endpoint_local, endpoint_global) / (local_norm * global_norm))
        angle = float(math.acos(max(-1.0, min(1.0, cosine))))
    return rotation, translation, {
        "alignment_method": "exact_final_overlap_anchor_rigid",
        "boundary_source_frame_ids": list(overlap),
        "anchor_source_frame_id": anchor_source_id,
        "overlap_local_baseline": local_baseline,
        "overlap_global_baseline": global_baseline,
        "translation_residual_m": float(np.max(translation_errors)),
        "translation_residual_rms_m": float(np.sqrt(np.mean(translation_errors * translation_errors))),
        "translation_residual_normalized": float(np.max(translation_errors) / global_baseline),
        "rotation_residual_rad": float(np.max(rotation_errors)),
        "overlap_endpoint_residuals": {
            "first_source_frame_id": overlap[0],
            "first_translation_residual_m": float(translation_errors[0]),
            "first_rotation_residual_rad": float(rotation_errors[0]),
            "anchor_source_frame_id": anchor_source_id,
            "anchor_translation_residual_m": float(translation_errors[-1]),
            "anchor_rotation_residual_rad": float(rotation_errors[-1]),
            "local_endpoint_displacement_m": local_norm,
            "global_endpoint_displacement_m": global_norm,
            "global_to_local_endpoint_ratio": None if local_norm <= DROID_SIM3_MIN_BASELINE else global_norm / local_norm,
            "endpoint_displacement_angle_rad": angle,
        },
        "held_out_overlap_drift": {
            "source_frame_ids": [overlap[index] for index in held_indices],
            "translation_max_m": float(np.max(held_translation)),
            "translation_rms_m": float(np.sqrt(np.mean(held_translation * held_translation))),
            "translation_normalized": float(np.max(held_translation) / global_baseline),
            "rotation_max_rad": float(np.max(held_rotation)),
        },
    }


def _validate_anchor_seam(
    merged: Mapping[int, np.ndarray],
    aligned_successor: Mapping[int, np.ndarray],
    *,
    anchor_source_id: int,
    first_unique_source_id: int,
    session_index: int,
) -> Mapping[str, object]:
    """Reject an impossible immediate rigid seam before timeline interpolation."""
    anchor, successor = merged[anchor_source_id], aligned_successor[first_unique_source_id]
    rotation = _rotation_distance_rad(anchor[:3, :3], successor[:3, :3])
    if not np.isfinite(rotation) or rotation > 1.0:
        raise StageResultError(
            f"DROID session {session_index} anchor stitch rejected impossible seam one-step rotation {rotation:.6f} rad"
        )
    return {
        "anchor_source_frame_id": anchor_source_id,
        "first_unique_source_frame_id": first_unique_source_id,
        "one_step_rotation_rad": rotation,
    }


def _merge_source_keyed_droid_sessions(
    session_outputs: Sequence[DroidFinalizeOutput],
    chunks: Sequence[tuple[int, ...]],
) -> tuple[dict[int, np.ndarray], tuple[Mapping[str, object], ...], tuple[tuple[int, ...], ...]]:
    """Anchor-stitch per-session Metric3D-normalized poses by exact source key."""
    if len(session_outputs) != len(chunks) or not chunks:
        raise StageResultError("DROID session output/chunk cardinality is invalid")
    merged: dict[int, np.ndarray] = {}
    reports: list[Mapping[str, object]] = []
    mappings: list[tuple[int, ...]] = []
    for session_index, (output, chunk) in enumerate(zip(session_outputs, chunks)):
        local = _validate_rigid_pose_array(output.T_world_camera.array, label=f"DROID session {session_index}")
        mapping = _droid_pose_source_indices(output, chunk)
        if mapping is None:
            if len(session_outputs) > 1:
                raise StageResultError(f"DROID session {session_index} lacks exact source mapping; positional offsets are forbidden")
            mapping = tuple(chunk)
        if set(mapping) != set(chunk) or len(mapping) != len(chunk):
            raise StageResultError(f"DROID session {session_index} mapping does not cover its exact source-keyed submission")
        local_by_source = {source_id: local[slot] for slot, source_id in enumerate(mapping)}
        if not merged:
            merged.update(local_by_source)
            mappings.append(mapping)
            continue
        overlap = tuple(sorted(set(merged).intersection(local_by_source)))
        expected_overlap = chunks[session_index - 1][-DROID_SIM3_OVERLAP_FRAMES:]
        if overlap != expected_overlap:
            raise StageResultError(f"DROID session {session_index} overlap must equal the fixed exact canonical source-ID window")
        anchor_source_id = expected_overlap[-1]
        rotation, translation, report = _anchor_stitch_drift_diagnostics(
            merged, local_by_source, overlap, anchor_source_id=anchor_source_id, session_index=session_index,
        )
        aligned: dict[int, np.ndarray] = {}
        for source_id, pose in local_by_source.items():
            value = np.eye(4, dtype=np.float64)
            value[:3, :3] = rotation @ pose[:3, :3]
            value[:3, 3] = rotation @ pose[:3, 3] + translation
            aligned[source_id] = value
        _validate_rigid_pose_array(np.stack([aligned[source_id] for source_id in mapping]), label=f"DROID session {session_index} anchored")
        anchor_slot = mapping.index(anchor_source_id)
        if mapping[:anchor_slot + 1] != overlap:
            raise StageResultError(f"DROID session {session_index} anchor does not terminate the exact overlap")
        unique_source_ids = mapping[anchor_slot + 1:]
        if not unique_source_ids:
            raise StageResultError(f"DROID session {session_index} has no unique source frames after anchor")
        report = {**report, "seam": _validate_anchor_seam(
            merged, aligned, anchor_source_id=anchor_source_id,
            first_unique_source_id=unique_source_ids[0], session_index=session_index,
        )}
        for source_id in unique_source_ids:
            if source_id in merged:
                raise StageResultError(f"DROID session {session_index} anchor stitch would overwrite source frame {source_id}")
            merged[source_id] = aligned[source_id]
        reports.append(report)
        mappings.append(mapping)
    return merged, tuple(reports), tuple(mappings)


def _normalize_droid_session_metric_gauge(
    output: DroidFinalizeOutput,
    *,
    relative_gauge: float,
    session_index: int,
) -> DroidFinalizeOutput:
    """Express one raw session in the first session's pre-final Metric3D gauge."""
    gauge = float(relative_gauge)
    if not np.isfinite(gauge) or gauge <= 0:
        raise StageResultError(f"DROID session {session_index} relative Metric3D gauge must be finite and positive")
    world = _validate_rigid_pose_array(output.T_world_camera.array, label=f"DROID session {session_index} raw")
    normalized_world = world.copy()
    normalized_world[:, :3, 3] *= gauge
    normalized_world = _validate_rigid_pose_array(normalized_world, label=f"DROID session {session_index} metric-normalized")
    camera = np.stack([np.linalg.inv(pose) for pose in normalized_world]).astype(np.float32)
    provenance = {
        **dict(output.scale_provenance),
        "per_session_metric3d_relative_gauge": gauge,
        "per_session_metric3d_normalization": "raw_camera_translation_multiplied_by_relative_gauge",
    }
    def tensor(source: TypedTensor, array: np.ndarray) -> TypedTensor:
        return TypedTensor(array.astype(np.float32), source.units, source.coordinate_frame, source.tensor_index_order, source.semantic_tag, {**dict(source.provenance), **provenance}, source.pixel_transform)
    return replace(output, T_world_camera=tensor(output.T_world_camera, normalized_world), T_camera_world=tensor(output.T_camera_world, camera), scale_provenance=provenance)


def _aggregate_sim3_normalized_scale_evidence(
    session_outputs: Sequence[DroidFinalizeOutput],
    chunks: Sequence[tuple[int, ...]],
    session_gauges: Sequence[float],
) -> TypedTensor:
    """Aggregate exact keyframe disparities after per-session Metric3D gauge normalization."""
    if len(session_outputs) != len(chunks) or len(session_gauges) != len(chunks):
        raise StageResultError("DROID scale evidence session cardinality is invalid")
    rows: dict[int, np.ndarray] = {}
    template: TypedTensor | None = None
    provenance_rows: list[Mapping[str, object]] = []
    for session_index, (output, chunk, gauge) in enumerate(zip(session_outputs, chunks, session_gauges)):
        keyframes = _droid_keyframe_source_indices(output, chunk)
        disparities = np.asarray(output.disparities.array, dtype=np.float32)
        if keyframes is None or disparities.shape[0] != len(keyframes) or not np.isfinite(disparities).all() or not np.isfinite(gauge) or gauge <= 0:
            raise StageResultError(f"DROID session {session_index} has invalid Metric3D-gauge scale evidence")
        if template is None:
            template = output.disparities
        elif disparities.shape[1:] != template.shape[1:]:
            raise StageResultError("DROID sessions have incompatible disparity grids for aggregate Metric3D evidence")
        normalized = disparities / np.float32(gauge)
        for source_id, disparity in zip(keyframes, normalized):
            existing = rows.get(source_id)
            if existing is not None and not np.allclose(existing, disparity, rtol=0.05, atol=1e-5):
                raise StageResultError(f"DROID duplicate keyframe {source_id} disagrees after Metric3D gauge normalization")
            rows.setdefault(source_id, np.ascontiguousarray(disparity))
        provenance_rows.append({"session_index": session_index, "relative_gauge": float(gauge), "keyframe_count": len(keyframes)})
    if template is None or not rows:
        raise StageResultError("DROID aggregate Metric3D evidence is empty")
    source_ids = tuple(sorted(rows))
    array = np.stack([rows[source_id] for source_id in source_ids]).astype(np.float32)
    mapping = [{"keyframe_index": index, "source_frame_id": str(source_id)} for index, source_id in enumerate(source_ids)]
    provenance = {
        **dict(template.provenance),
        "keyframe_mapping": mapping,
        "aggregate_source_frame_ids": list(source_ids),
        "aggregate_session_evidence": provenance_rows,
        "disparity_gauge": "inverse_depth_divided_by_per_session_metric3d_relative_gauge",
        "video_level_metric3d_evidence": True,
    }
    return TypedTensor(array, template.units, template.coordinate_frame, template.tensor_index_order, template.semantic_tag, provenance, template.pixel_transform)


def _densify_merged_droid_output(
    output: DroidFinalizeOutput,
    timeline: SourceTimeline,
    coverage: DroidCoverage,
    merged: Mapping[int, np.ndarray],
    session_mappings: Sequence[tuple[int, ...]],
    overlap_reports: Sequence[Mapping[str, object]],
) -> DroidFinalizeOutput:
    selected = tuple(index for index, sampled in enumerate(coverage.pose_sampled or ()) if sampled)
    if tuple(sorted(merged)) != selected:
        raise StageResultError("DROID merged source keys do not equal selected source schedule")
    sampled_poses = np.stack([merged[index] for index in selected]).astype(np.float32)
    world, sampled_mask = _interpolate_droid_poses(selected, sampled_poses, timeline.frame_count)
    temporal_qc = _droid_temporal_qc(world, np.ones(timeline.frame_count, dtype=bool))
    camera = np.linalg.inv(world).astype(np.float32)
    provenance = {
        **dict(output.scale_provenance),
        "sampling": "source_keyed_sessions_per_session_metric3d_then_anchor_stitch_then_endpoint_inclusive_se3_interpolation",
        "measured_source_frame_indices": list(selected),
        "droid_pose_sampled": sampled_mask.tolist(),
        "droid_pose_valid": [True] * timeline.frame_count,
        "droid_session_source_mappings": [list(mapping) for mapping in session_mappings],
        "droid_sim3_overlap_reports": [dict(row) for row in overlap_reports],
        "effective_unique_coverage_count": coverage.submitted_count,
        "actual_pushed_count": coverage.actual_pushed_count,
        "droid_temporal_qc": temporal_qc,
    }
    def tensor(source: TypedTensor, array: np.ndarray) -> TypedTensor:
        return TypedTensor(array, source.units, source.coordinate_frame, source.tensor_index_order, source.semantic_tag, {**dict(source.provenance), **provenance}, source.pixel_transform)
    return replace(output, T_world_camera=tensor(output.T_world_camera, world), T_camera_world=tensor(output.T_camera_world, camera), scale_provenance={**dict(output.scale_provenance), **provenance})


class FullVideoTimelineDriver:
    """Execute the typed algorithm portion for one complete source video."""

    def __init__(self, client: AlgorithmStageClient, config: FullVideoDriverConfig | None = None) -> None:
        self.client = client
        self.config = config or FullVideoDriverConfig()
        self._timing_collector: ModuleTimingCollector | None = None
        required = (*REQUIRED_STAGE_IDS, "cosmos3.reason") if self.config.cosmos_enabled else REQUIRED_STAGE_IDS
        self.client.preflight(required)

    def run(self, source: FrameSource, *, case_id: str, item_id: str | None = None) -> FullVideoAlgorithmState:
        frame_store_started = time.monotonic()
        if isinstance(source, OpenCvFrameSource) and not source.frame_store_built:
            if self.config.frame_store_spill_dir is None:
                raise TimelineDriverError("batched OpenCV driver requires a configured sequential frame-store spill directory")
            # The current DAG's dense hand detector and full-timeline renderers
            # make their union the complete source timeline. UniDepth/DROID,
            # WiLoR detections, HaWoR crops, and Cosmos galleries are subsets.
            # Register this union before any worker thread can request a frame.
            source.build_frame_store(source.timeline.frame_indices, spill_dir=self.config.frame_store_spill_dir)
        module_timings_s = {"frame_store": float(time.monotonic() - frame_store_started)}
        collector = ModuleTimingCollector()
        collector.local("frame_store", module_timings_s["frame_store"])
        self._timing_collector = collector
        semantic_pool: ThreadPoolExecutor | None = None
        semantic_future: Future[tuple[CosmosSemanticResult, RequestBatchTrace]] | None = None
        item = item_id or f"{case_id}:item-0"
        if self.config.cosmos_enabled:
            semantic_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cosmos-independent")
            semantic_future = semantic_pool.submit(self._run_cosmos, source, case_id, item)
        try:
            state = self._run_physical(source, case_id=case_id, item_id=item, semantic_future=semantic_future, module_timings_s=module_timings_s)
            if not hasattr(state, "module_timings_s"):
                return state
            breakdown, notes = collector.breakdown(state.module_timings_s)
            return replace(state, module_timing_breakdown_s=breakdown, module_timing_breakdown_notes=notes)
        finally:
            if semantic_pool is not None:
                semantic_pool.shutdown(wait=True, cancel_futures=True)
            self._timing_collector = None

    def _execute_timed(self, request: AlgorithmRequest[Any]) -> AlgorithmResult[Any]:
        started = time.monotonic()
        timed = getattr(self.client, "execute_timed", None)
        if callable(timed):
            result, timing = timed(request)
        else:
            result = self.client.execute(request)
            timing = ClientRequestTiming(
                total_wall_s=time.monotonic() - started,
                available=False,
                unavailable_reason="client exposes execute() only; prepare/transport/decode boundaries unavailable",
            )
        collector = getattr(self, "_timing_collector", None)
        if collector is not None:
            collector.request(request.algorithm_id, timing)
        return result

    def _run_physical(
        self,
        source: FrameSource,
        *,
        case_id: str,
        item_id: str,
        semantic_future: Future[tuple[CosmosSemanticResult, RequestBatchTrace]] | None,
        module_timings_s: dict[str, float],
    ) -> FullVideoAlgorithmState:
        timeline = source.timeline
        if not case_id:
            raise TimelineDriverError("case_id is required")
        item = item_id
        wave1_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="wave1") as wave:
            unidepth_future = wave.submit(self._run_unidepth, source, case_id, item)
            hands_future = wave.submit(self._run_hands, source, case_id, item)
            unidepth_records, unidepth_trace = unidepth_future.result()
            hands_records, hands_trace = hands_future.result()
        wave1_end = time.monotonic()
        canonical = self._canonical_k(unidepth_records)
        detections = self._hand_detections(hands_records, timeline)
        tracks = self._associate_tracks(detections, timeline.frame_count)
        wilor_build_started = time.monotonic()
        wilor_requests = self._build_wilor_requests(source, case_id, item, canonical, detections)
        module_timings_s["wilor_build"] = float(time.monotonic() - wilor_build_started)
        collector = getattr(self, "_timing_collector", None)
        if collector is not None:
            collector.local("wilor_build", module_timings_s["wilor_build"])

        # DROID must consume masks derived from the completed WiLoR geometry;
        # these stages are intentionally sequential, not concurrent lanes.
        wave2_start = time.monotonic()
        wilor_records, wilor_trace = self._run_many_traced("wilor.reconstruct", wilor_requests)
        droid_records, droid_traces = self._run_droid(
            source, case_id, item, canonical, unidepth_records, detections, tracks, wilor_records,
        )
        wave2_end = time.monotonic()

        base_traces = (
            RequestBatchTrace("wave1.concurrent_lanes", 2, 2, 2, (timeline.frame_count,), wave1_start, wave1_end),
            unidepth_trace,
            hands_trace,
            RequestBatchTrace("wave2.sequential_wilor_then_droid", 1, 1, 1, (timeline.frame_count,), wave2_start, wave2_end),
            *droid_traces,
            wilor_trace,
        )
        if not droid_records.accepted_trajectory:
            semantic_status, semantic_rows, semantic_count, semantic_review, semantic_trace = self._finish_semantic(semantic_future)
            empty = _empty_timeline_state(timeline.frame_count)
            reasons = (droid_records.blocker or "remote_droid_insufficient_keyframes",)
            final = droid_records.final.output
            traces = tuple((*base_traces, *((semantic_trace,) if semantic_trace is not None else ())))
            _record_service_module_timings(module_timings_s, traces)
            return FullVideoAlgorithmState(
                timeline.frame_count,
                timeline,
                canonical,
                detections,
                tracks,
                unidepth_records,
                hands_records,
                wilor_records,
                droid_records,
                (),
                (),
                (),
                (),
                empty,
                traces,
                semantic_status,
                semantic_rows,
                semantic_count,
                semantic_review,
                AlgorithmAcceptance(False, final.diagnostic_only, final.scale_mode, reasons, "pending_next_slice", "pending_next_slice"),
                dict(module_timings_s),
            )

        hawor_build_started = time.monotonic()
        hawor_requests, hawor_chunks = self._build_hawor_requests(
            source, case_id, item, tracks, canonical, unidepth_records, droid_records.final.output
        )
        hawor_build_s = time.monotonic() - hawor_build_started
        collector = getattr(self, "_timing_collector", None)
        if collector is not None:
            collector.local("hawor", hawor_build_s)
        hawor_records, hawor_trace = self._run_many_traced("hawor.infer_tracks", hawor_requests)
        hawor_candidates_started = time.monotonic()
        hawor_candidates = self._hawor_candidates(hawor_records, hawor_chunks)
        collector = getattr(self, "_timing_collector", None)
        if collector is not None:
            collector.local("hawor", time.monotonic() - hawor_candidates_started)
        infiller_build_started = time.monotonic()
        infiller_requests, infiller_windows = self._build_infiller_requests(
            case_id, item, timeline, canonical, droid_records.final.output, hawor_candidates
        )
        collector = getattr(self, "_timing_collector", None)
        if collector is not None:
            collector.local("infiller", time.monotonic() - infiller_build_started)
        infiller_records, infiller_trace = self._run_many_traced("hawor_infiller.fill", infiller_requests)
        candidates_started = time.monotonic()
        final_candidates = list(self._wilor_candidates(wilor_records, detections, unidepth_records, canonical))
        final_candidates.extend(hawor_candidates)
        final_candidates.extend(self._infiller_candidates(infiller_records, tuple(window for window in infiller_windows if window.submitted)))
        inference = _merge_timeline_candidates(timeline.frame_count, final_candidates)
        collector = getattr(self, "_timing_collector", None)
        if collector is not None:
            collector.local("wilor_service", time.monotonic() - candidates_started)

        detected_sides = {record.side for record in detections}
        valid = inference.valid.array
        hand_complete = not detections or all(np.all(valid[0 if side is HandSide.LEFT else 1]) for side in detected_sides)
        final_droid = droid_records.final.output
        reasons: list[str] = []
        if final_droid.diagnostic_only:
            reasons.append("diagnostic_monocular_droid_unaccepted")
        if droid_records.retries_used:
            reasons.append("droid_nonfinite_finalize_recovered")
        if droid_records.coverage.partial:
            reasons.append(f"service_capacity_{DROID_SERVICE_PUSH_CAPACITY}_exceeded")
        if detections and not hand_complete:
            reasons.append("incomplete_timeline_hand_state")
        for window in infiller_windows:
            if not window.submitted and window.blocker:
                reasons.append(window.blocker)
        accepted = final_droid.acceptance and hand_complete and not reasons
        semantic_status, semantic_rows, semantic_count, semantic_review, semantic_trace = self._finish_semantic(semantic_future)
        acceptance = AlgorithmAcceptance(
            accepted=accepted,
            diagnostic_only=final_droid.diagnostic_only,
            scale_mode=final_droid.scale_mode,
            reasons=tuple(dict.fromkeys(reasons)),
            physical_adapter_status="pending_next_slice",
            render_status="pending_next_slice",
        )
        traces = tuple((*base_traces, hawor_trace, infiller_trace, *((semantic_trace,) if semantic_trace is not None else ())))
        _record_service_module_timings(module_timings_s, traces)
        return FullVideoAlgorithmState(
            timeline.frame_count,
            timeline,
            canonical,
            detections,
            tracks,
            unidepth_records,
            hands_records,
            wilor_records,
            droid_records,
            hawor_records,
            infiller_records,
            hawor_chunks,
            infiller_windows,
            inference,
            tuple((*base_traces, hawor_trace, infiller_trace, *((semantic_trace,) if semantic_trace is not None else ()))),
            semantic_status,
            semantic_rows,
            semantic_count,
            semantic_review,
            acceptance,
            dict(module_timings_s),
        )

    def _run_cosmos(self, source: FrameSource, case_id: str, item_id: str) -> tuple[CosmosSemanticResult, RequestBatchTrace]:
        started = time.monotonic()
        result = run_cosmos_semantics(
            _TimingClientProxy(self),
            source,
            case_id=case_id,
            item_id=item_id,
            revision=self.config.model_revisions["cosmos3.reason"],
            gallery_width=self.config.cosmos_gallery_width,
        )
        completed = time.monotonic()
        return result, RequestBatchTrace(
            "cosmos3.reason",
            result.request_count,
            1,
            1,
            (1,),
            started,
            completed,
        )

    @staticmethod
    def _finish_semantic(
        future: Future[tuple[CosmosSemanticResult, RequestBatchTrace]] | None,
    ) -> tuple[str, tuple[Mapping[str, object], ...], int, Mapping[str, object], RequestBatchTrace | None]:
        if future is None:
            return "absent_disabled", (), 0, {}, None
        result, trace = future.result()
        semantic_status = "completed_with_anomalies" if result.anomalies else "enabled"
        review: Mapping[str, object] = {
            "schema": "v22_cosmos_semantic_review.v1",
            "status": "completed_with_anomalies" if result.anomalies else "ok",
            "method": "cosmos_gallery_boundary_video_understanding",
            "sampled_source_indices": list(result.sampled_source_indices),
            "coarse_request_count": result.coarse_request_count,
            "refinement_request_count": result.refinement_request_count,
            "repair_count": result.repair_request_count,
            "request_count": result.request_count,
            "coarse_rows": [dict(row) for row in result.coarse_rows],
            "boundary_refinements": [dict(row) for row in result.refinements],
            "segments": [dict(row) for row in result.segments],
            "semantic_rows": [dict(row) for row in result.rows],
            "summary": {
                "semantic_clip_count": len(result.rows),
                "detected_changes": len(result.refinements),
                "boundary_refinements": len(result.refinements),
                "timeline_coverage_fraction": sum(int(row["end_frame"]) - int(row["start_frame"]) for row in result.rows) / result.rows[-1]["end_frame"],
            },
            "attempts": [dict(attempt) for attempt in result.attempts],
            "anomaly_count": len(result.anomalies),
            "anomaly_ledger": [dict(anomaly) for anomaly in result.anomalies],
            "refinements": [dict(row) for row in result.refinements],
            "responses": [
                {
                    "finish_reason": output.finish_reason,
                    "stop_reason": output.stop_reason,
                    "prompt_tokens": output.prompt_tokens,
                    "completion_tokens": output.completion_tokens,
                    "total_tokens": output.total_tokens,
                    "timings": dict(output.timings),
                    "trace": dict(output.trace),
                    "media_provenance": [dict(item) for item in output.media_provenance],
                    "model_revision": output.model_revision,
                }
                for output in result.outputs
            ],
            "claim_scope": "semantic_only_not_physical_evidence",
        }
        return semantic_status, result.rows, result.request_count, review, trace

    def _run_unidepth(self, source: FrameSource, case_id: str, item_id: str) -> tuple[tuple[AlgorithmResult[UniDepthOutput], ...], RequestBatchTrace]:
        indices = _selected_source_indices(source.timeline, self.config.unidepth_fps)
        requests = tuple(self._frame_request(source, case_id, item_id, index, "unidepth.infer") for index in indices)
        results, trace = self._run_many_traced("unidepth.infer", requests)
        return _typed_results(results, UniDepthOutput), trace

    def _run_hands(self, source: FrameSource, case_id: str, item_id: str) -> tuple[tuple[AlgorithmResult[HandsOutput], ...], RequestBatchTrace]:
        requests = tuple(self._frame_request(source, case_id, item_id, index, "hands.detect") for index in source.timeline.frame_indices)
        results, trace = self._run_many_traced("hands.detect", requests)
        return _typed_results(results, HandsOutput), trace

    def _frame_request(
        self,
        source: FrameSource,
        case_id: str,
        item_id: str,
        frame_index: int,
        stage_id: str,
    ) -> AlgorithmRequest[Any]:
        timeline = source.timeline
        rgb, spatial = _prepare_inference_rgb(source.read_rgb(frame_index), timeline, self.config.inference_size_yx)
        ownership = _ownership(case_id, item_id, timeline.source_id, stage_id, f"frame:{frame_index:06d}")
        tensor = TypedTensor(
            rgb[None],
            units="uint8_rgb",
            coordinate_frame=spatial.grid_id,
            tensor_index_order="thwc",
            semantic_tag="source_backed_rgb_v1",
            provenance={"source_sha256": timeline.source_sha256, "frame_index": frame_index},
            pixel_transform=spatial.pixel_to_source,
        )
        indices = (frame_index,)
        timestamps = (timeline.frames[frame_index].timestamp_s,)
        value: UniDepthInput | HandsInput
        if stage_id == "unidepth.infer":
            value = UniDepthInput(ownership, tensor, indices, timestamps, spatial)
        elif stage_id == "hands.detect":
            value = HandsInput(ownership, tensor, indices, timestamps, spatial)
        else:
            raise TimelineDriverError(f"unsupported frame stage {stage_id}")
        return _request(stage_id, case_id, item_id, timeline.metadata(indices), value, self.config.model_revisions[stage_id], native_shape=(1, *rgb.shape))

    def _canonical_k(self, records: Sequence[AlgorithmResult[UniDepthOutput]]) -> CanonicalKAggregation:
        candidates: list[IntrinsicsCandidate] = []
        for result in records:
            output = result.output
            if len(output.frame_indices) != 1:
                raise StageResultError("driver requires one UniDepth result per source frame")
            confidence = output.confidence.array[0]
            finite = confidence[np.isfinite(confidence)]
            quality = float(np.mean(finite)) if finite.size else 1e-6
            quality = max(quality, 1e-6)
            candidates.append(
                IntrinsicsCandidate(
                    frame_idx=output.frame_indices[0],
                    k_depth=np.asarray(output.K_px.array[0], dtype=np.float64),
                    p_depth_to_source=np.asarray(output.spatial.pixel_to_source, dtype=np.float64),
                    confidence=quality,
                    frame_quality=1.0,
                )
            )
        return aggregate_canonical_k(candidates)

    def _hand_detections(
        self,
        records: Sequence[AlgorithmResult[HandsOutput]],
        timeline: SourceTimeline,
    ) -> tuple[HandDetectionRecord, ...]:
        detections: list[HandDetectionRecord] = []
        for result in records:
            output = result.output
            if len(output.frame_indices) != 1:
                raise StageResultError("driver requires one Hands result per source frame")
            frame_index = output.frame_indices[0]
            boxes = output.detections.boxes_xyxy.array[0]
            scores = output.detections.scores.array[0]
            sides = output.detections.sides.array[0]
            visibility = output.detections.visibility.array[0]
            uncertainty = output.detections.uncertainty.array[0]
            transform = np.asarray(output.spatial.pixel_to_source, dtype=np.float64)
            for slot in range(scores.shape[0]):
                score = float(scores[slot])
                box = np.asarray(boxes[slot], dtype=np.float64)
                if not np.isfinite(score) or score < self.config.min_hand_score or not np.all(np.isfinite(box)):
                    continue
                source_box = _transform_box(box, transform)
                if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
                    continue
                side_code = int(sides[slot])
                if side_code not in (0, 1):
                    raise StageResultError("hand side must be 0=left or 1=right")
                side = HandSide.LEFT if side_code == 0 else HandSide.RIGHT
                vis = float(np.clip(visibility[slot], 0.0, 1.0))
                unc = max(0.0, float(uncertainty[slot]))
                state = "visible" if vis >= 0.75 else "partially_visible" if vis > 0.0 else "occluded"
                detections.append(
                    HandDetectionRecord(
                        detection_id=f"{timeline.source_id}:{frame_index:06d}:{slot:02d}",
                        frame_index=frame_index,
                        timestamp_s=timeline.frames[frame_index].timestamp_s,
                        side=side,
                        box_xyxy_source=tuple(float(value) for value in source_box),
                        score=score,
                        visibility=vis,
                        uncertainty=unc,
                        occlusion_state=state,
                        ownership=output.ownership,
                    )
                )
        return tuple(detections)

    def _associate_tracks(self, detections: Sequence[HandDetectionRecord], frame_count: int) -> tuple[HandTrack, ...]:
        by_frame: dict[int, list[HandDetectionRecord]] = {}
        for detection in detections:
            by_frame.setdefault(detection.frame_index, []).append(detection)
        active: list[_MutableTrack] = []
        completed: list[_MutableTrack] = []
        next_id = 0
        for frame_index in range(frame_count):
            current = by_frame.get(frame_index, [])
            expired = [track for track in active if frame_index - track.last_frame > self.config.max_track_gap]
            for track in expired:
                active.remove(track)
                completed.append(track)
            matches: list[tuple[float, int, int]] = []
            for track_index, track in enumerate(active):
                dt = max(1, frame_index - track.last_frame)
                predicted = track.last_center + track.velocity * dt
                for detection_index, detection in enumerate(current):
                    if detection.side is not track.side:
                        continue
                    box = np.asarray(detection.box_xyxy_source, dtype=np.float64)
                    center = _box_center(box)
                    scale = max(1.0, math.sqrt(_box_area(track.last_box)))
                    cost = (1.0 - _box_iou(track.last_box, box)) + float(np.linalg.norm(center - predicted) / scale) + 0.25 * (1.0 - detection.score)
                    matches.append((cost, track_index, detection_index))
            used_tracks: set[int] = set()
            used_detections: set[int] = set()
            for cost, track_index, detection_index in sorted(matches):
                if cost > self.config.track_match_threshold or track_index in used_tracks or detection_index in used_detections:
                    continue
                track = active[track_index]
                detection = current[detection_index]
                center = _box_center(np.asarray(detection.box_xyxy_source, dtype=np.float64))
                dt = max(1, frame_index - track.last_frame)
                measured_velocity = (center - track.last_center) / dt
                track.velocity = 0.5 * track.velocity + 0.5 * measured_velocity
                track.last_center = center
                track.last_box = np.asarray(detection.box_xyxy_source, dtype=np.float64)
                track.last_frame = frame_index
                track.detections.append(detection)
                used_tracks.add(track_index)
                used_detections.add(detection_index)
            for detection_index, detection in enumerate(current):
                if detection_index in used_detections:
                    continue
                box = np.asarray(detection.box_xyxy_source, dtype=np.float64)
                active.append(
                    _MutableTrack(
                        track_id=f"track-{next_id:04d}",
                        side=detection.side,
                        detections=[detection],
                        last_center=_box_center(box),
                        velocity=np.zeros(2, dtype=np.float64),
                        last_box=box,
                        last_frame=frame_index,
                    )
                )
                next_id += 1
        completed.extend(active)
        result: list[HandTrack] = []
        for track in sorted(completed, key=lambda item: int(item.track_id.split("-")[-1])):
            detections_by_frame = {record.frame_index: record for record in track.detections}
            start = track.detections[0].frame_index
            end = track.detections[-1].frame_index
            visibility: list[str] = []
            uncertainty: list[float] = []
            for frame_index in range(start, end + 1):
                record = detections_by_frame.get(frame_index)
                if record is None:
                    visibility.append("unresolved")
                    uncertainty.append(float("inf"))
                else:
                    visibility.append(record.occlusion_state)
                    uncertainty.append(record.uncertainty)
            result.append(HandTrack(track.track_id, track.side, start, end, tuple(track.detections), tuple(visibility), tuple(uncertainty)))
        return tuple(result)

    def _build_wilor_requests(
        self,
        source: FrameSource,
        case_id: str,
        item_id: str,
        canonical: CanonicalKAggregation,
        detections: Sequence[HandDetectionRecord],
    ) -> tuple[AlgorithmRequest[WiLoRInput], ...]:
        timeline = source.timeline
        k_tensor = _typed_K(canonical.k_canonical, timeline.source_id)
        crop_batch, transforms = _normalized_crop_batch(source, detections, timeline, self.config.crop_scale)
        requests: list[AlgorithmRequest[WiLoRInput]] = []
        for detection_index, (detection, transform) in enumerate(zip(detections, transforms)):
            scope = f"detection:{detection.detection_id}"
            ownership = _ownership(case_id, item_id, timeline.source_id, "wilor.reconstruct", scope)
            crop = crop_batch[detection_index : detection_index + 1]
            value = WiLoRInput(
                ownership,
                TypedTensor(
                    crop,
                    "imagenet_normalized",
                    "wilor_crop",
                    "bcyx",
                    "wilor_real_normalized_crop_v1",
                    {"detection_id": detection.detection_id, "frame_index": detection.frame_index},
                    transform.source_to_crop.pixel_to_source,
                ),
                (transform,),
                k_tensor,
            )
            requests.append(
                _request(
                    "wilor.reconstruct",
                    case_id,
                    item_id,
                    timeline.metadata((detection.frame_index,)),
                    value,
                    self.config.model_revisions["wilor.reconstruct"],
                    native_shape=(1, 3, 256, 256),
                )
            )
        return tuple(requests)

    def _wilor_candidates(
        self,
        records: Sequence[AlgorithmResult[WiLoROutput]],
        detections: Sequence[HandDetectionRecord],
        unidepth_records: Sequence[AlgorithmResult[UniDepthOutput]],
        canonical: CanonicalKAggregation,
    ) -> tuple[_ManoCandidate, ...]:
        if len(records) != len(detections):
            raise StageResultError("WiLoR request/result detection count changed")
        candidates: list[_ManoCandidate] = []
        # WiLoR detections are sparse; support metric wrist depth over only a
        # short observed gap.  Long gaps remain unresolved and therefore cannot
        # silently acquire a rendered hand.
        depth_history: dict[HandSide, list[tuple[int, float]]] = {HandSide.LEFT: [], HandSide.RIGHT: []}
        for result, detection in zip(records, detections):
            output = result.output
            mano = output.mano
            if mano.global_orient.shape[0] != 1 or output.handedness != (detection.side,):
                raise StageResultError("WiLoR per-detection result changed batch size or handedness")
            slot = 0
            vertices_root = np.asarray(mano.vertices.array[slot], dtype=np.float32)
            joints_root = np.asarray(mano.joints.array[slot], dtype=np.float32)
            vertices_source_px = np.asarray(mano.vertices_source_px.array[slot], dtype=np.float32)
            joints_source_px = np.asarray(mano.joints_source_px.array[slot], dtype=np.float32)
            depth = _unidepth_hand_depth_m(
                _nearest_unidepth_record(unidepth_records, detection.frame_index),
                joints_source_px[0],
                detection,
            )
            depth_inferred = False
            if depth is not None:
                depth_history[detection.side].append((detection.frame_index, depth))
                depth_history[detection.side] = depth_history[detection.side][-15:]
            else:
                recent = [(frame, value) for frame, value in depth_history[detection.side] if detection.frame_index - frame <= 15]
                if recent:
                    depth = float(np.median([value for _, value in recent]))
                    depth_inferred = True
            metric_geometry = _metric_wilor_geometry(vertices_root, joints_root, joints_source_px, depth, canonical.k_canonical)
            if metric_geometry is None:
                vertices_camera = joints_camera = None
                camera_translation = np.zeros(3, dtype=np.float32)
            else:
                vertices_camera, joints_camera, camera_translation = metric_geometry
            candidates.append(
                _ManoCandidate(
                    detection.frame_index,
                    detection.side,
                    np.asarray(mano.global_orient.array[slot], dtype=np.float32),
                    np.asarray(mano.hand_pose.array[slot], dtype=np.float32),
                    np.asarray(mano.betas.array[slot], dtype=np.float32),
                    camera_translation,
                    vertices_camera,
                    joints_camera,
                    not depth_inferred,
                    depth_inferred,
                    float(mano.uncertainty.array[slot]),
                    "wilor.reconstruct",
                    output.ownership.scope,
                    vertices_source_px,
                    joints_source_px,
                )
            )
        return tuple(candidates)

    def _run_droid(
        self,
        source: FrameSource,
        case_id: str,
        item_id: str,
        canonical: CanonicalKAggregation,
        unidepth_records: Sequence[AlgorithmResult[UniDepthOutput]],
        detections: Sequence[HandDetectionRecord] = (),
        tracks: Sequence[HandTrack] = (),
        wilor_records: Sequence[AlgorithmResult[WiLoROutput]] = (),
    ) -> tuple[DroidExecutionRecords, tuple[RequestBatchTrace, ...]]:
        return self._run_droid_sessions(
            source, case_id, item_id, canonical, unidepth_records, detections, tracks, wilor_records,
        )

    def _droid_dynamic_masks(
        self,
        source: FrameSource,
        selected_indices: Sequence[int],
        detections: Sequence[HandDetectionRecord],
        tracks: Sequence[HandTrack],
        input_shape: tuple[int, int],
        wilor_records: Sequence[AlgorithmResult[WiLoROutput]] = (),
    ) -> dict[int, np.ndarray]:
        """Rasterize anatomical hand tracks on the exact DROID input grid.

        The same binary mask controls both RGB suppression and DROID BA weights:
        1 means dynamic/ignored and 0 means static/retained, matching the
        reference DROID ``video.masks > 0`` contract. Frame keys are source indices.
        """
        input_h, input_w = input_shape
        sx, sy = input_w / source.timeline.width_px, input_h / source.timeline.height_px
        boxes_by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
        tracked_ids = {record.detection_id for track in tracks for record in track.detections}
        for detection in detections:
            if tracks and detection.detection_id not in tracked_ids:
                continue
            if detection.occlusion_state != "out_of_frame":
                boxes_by_frame.setdefault(int(detection.frame_index), []).append(detection.box_xyxy_source)
        # Match the reference tracked-mask preparation: short internal detector
        # gaps retain a linearly propagated box; leading/trailing/long gaps do not.
        for track in tracks:
            ordered = sorted(track.detections, key=lambda row: row.frame_index)
            for left, right in zip(ordered[:-1], ordered[1:]):
                gap = right.frame_index - left.frame_index
                if gap <= 1 or gap - 1 > self.config.max_track_gap:
                    continue
                left_box = np.asarray(left.box_xyxy_source, dtype=np.float64)
                right_box = np.asarray(right.box_xyxy_source, dtype=np.float64)
                for offset in range(1, gap):
                    alpha = offset / float(gap)
                    box = tuple(((1.0 - alpha) * left_box + alpha * right_box).tolist())
                    boxes_by_frame.setdefault(left.frame_index + offset, []).append(box)
        # WiLoR requests/results are detection-ordered. Prefer same-frame
        # projected MANO vertices; boxes are a deterministic fallback when the
        # service omits projections or returns an invalid surface.
        projected_by_frame: dict[int, list[np.ndarray]] = {}
        for result, detection in zip(wilor_records, detections):
            if tracks and detection.detection_id not in tracked_ids:
                continue
            vertices_tensor = getattr(result.output.mano, "vertices_source_px", None)
            if vertices_tensor is None:
                continue
            vertices = np.asarray(vertices_tensor.array)
            if vertices.ndim == 3:
                vertices = vertices[0]
            if vertices.shape == (778, 2) and np.isfinite(vertices).all():
                projected_by_frame.setdefault(int(detection.frame_index), []).append(vertices)
        masks: dict[int, np.ndarray] = {}
        for frame_index in selected_indices:
            mask = np.zeros((input_h, input_w), dtype=np.uint8)
            projected = projected_by_frame.get(int(frame_index), ())
            if projected:
                import cv2
                points = np.concatenate(projected, axis=0).astype(np.float32)
                points[:, 0] *= np.float32(sx); points[:, 1] *= np.float32(sy)
                hull = cv2.convexHull(points.reshape(-1, 1, 2).astype(np.float32))
                cv2.fillConvexPoly(mask, hull.reshape(-1, 2).astype(np.int32), 1)
                # Publish this selected source key before bypassing box fallback.
                masks[int(frame_index)] = mask
                continue
            for box_row in boxes_by_frame.get(int(frame_index), ()):
                box = np.asarray(box_row, dtype=np.float64)
                if not np.isfinite(box).all():
                    continue
                x0 = max(0, min(input_w, int(math.floor(box[0] * sx))))
                y0 = max(0, min(input_h, int(math.floor(box[1] * sy))))
                x1 = max(0, min(input_w, int(math.ceil(box[2] * sx))))
                y1 = max(0, min(input_h, int(math.ceil(box[3] * sy))))
                if x0 < x1 and y0 < y1:
                    mask[y0:y1, x0:x1] = 1
            masks[int(frame_index)] = mask
        return masks

    def _run_droid_sessions(
        self,
        source: FrameSource,
        case_id: str,
        item_id: str,
        canonical: CanonicalKAggregation,
        unidepth_records: Sequence[AlgorithmResult[UniDepthOutput]],
        detections: Sequence[HandDetectionRecord],
        tracks: Sequence[HandTrack],
        wilor_records: Sequence[AlgorithmResult[WiLoROutput]],
    ) -> tuple[DroidExecutionRecords, tuple[RequestBatchTrace, ...]]:
        """Run independent capacity-bounded source-keyed sessions concurrently."""
        timeline = source.timeline
        selected = timeline.frame_indices if self.config.droid_fps is None else _droid_sample_source_indices(timeline, self.config.droid_fps)
        chunks = _droid_chunks_with_overlap(selected)
        sampled = np.zeros(timeline.frame_count, dtype=bool)
        sampled[list(selected)] = True
        coverage = DroidCoverage(
            timeline.frame_count, len(selected), (True,) * timeline.frame_count,
            tuple(bool(value) for value in sampled), chunks, self.config.droid_fps,
        )
        masks = self._droid_dynamic_masks(
            source, selected, detections, tracks, _droid_input_shape_yx(timeline, self.config), wilor_records,
        )
        if set(masks) != set(selected):
            raise StageResultError("DROID dynamic masks must cover every selected source key before session fanout")

        def run_chunk(chunk_index: int, source_indices: tuple[int, ...]):
            chunk_masks = {frame_index: masks[frame_index] for frame_index in source_indices}
            return self._run_droid_chunk(
                source, case_id, item_id, canonical, unidepth_records, chunk_index, source_indices,
                attempt=0, static_confidence_masks=chunk_masks,
            )

        workers = min(len(chunks), self.config.droid_session_workers)
        results: list[tuple[AlgorithmResult[DroidCreateOutput], tuple[AlgorithmResult[DroidPushOutput], ...], AlgorithmResult[DroidFinalizeOutput], tuple[RequestBatchTrace, ...]] | None] = [None] * len(chunks)
        # The bounded executor submits whole lifecycle chains independently; the
        # existing route dispatcher remains responsible for selecting replicas.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="droid-session") as executor:
            futures = {executor.submit(run_chunk, index, chunk): index for index, chunk in enumerate(chunks)}
            for future, index in list(futures.items()):
                results[index] = future.result()
        completed = [result for result in results if result is not None]
        if len(completed) != len(chunks):
            raise StageResultError("DROID session executor did not complete every planned source-keyed session")
        creates = tuple(result[0] for result in completed)
        pushes_by_attempt = tuple(result[1] for result in completed)
        finals = tuple(result[2] for result in completed)
        traces = tuple(trace for result in completed for trace in result[3])
        for chunk_index, (chunk, final) in enumerate(zip(chunks, finals)):
            if final.output.T_world_camera.shape != (len(chunk), 4, 4):
                raise StageResultError(f"DROID session {chunk_index} returned pose count inconsistent with its source-keyed schedule")
        session_outputs = tuple(final.output for final in finals)
        metric_scales, metric_reports = _estimate_metric3d_droid_scales(
            session_outputs, chunks, timeline, masks, source, canonical.k_canonical,
            hawor_root=self.config.droid_metric3d_hawor_root,
            metric_checkpoint=self.config.droid_metric3d_checkpoint,
            metric3d_python=self.config.droid_metric3d_python,
            cuda_visible_devices=self.config.droid_metric3d_cuda_visible_devices,
        )
        metric_video_scale = float(metric_scales[0])
        if not np.isfinite(metric_video_scale) or metric_video_scale <= 0:
            raise StageResultError("video-level Metric3D DROID scale must be finite and positive")
        metric_gauges = tuple(float(scale / metric_video_scale) for scale in metric_scales)
        if any(not np.isfinite(gauge) or gauge <= 0 for gauge in metric_gauges):
            raise StageResultError("DROID per-session Metric3D relative gauges must be finite and positive")
        normalized_outputs = tuple(
            _normalize_droid_session_metric_gauge(output, relative_gauge=gauge, session_index=index)
            for index, (output, gauge) in enumerate(zip(session_outputs, metric_gauges))
        )
        merged, residuals, mappings = _merge_source_keyed_droid_sessions(normalized_outputs, chunks)
        aggregate_disparities = _aggregate_sim3_normalized_scale_evidence(session_outputs, chunks, metric_gauges)
        aggregate_provenance = {
            "session_policy": "per_session_metric3d_then_anchor_stitch",
            "session_source_indices": [list(chunk) for chunk in chunks],
            "effective_unique_coverage_count": coverage.submitted_count,
            "actual_pushed_count": coverage.actual_pushed_count,
            "droid_session_workers": workers,
            "droid_anchor_stitch_reports": [dict(row) for row in residuals],
            "droid_anchor_source_frame_ids": [int(row["anchor_source_frame_id"]) for row in residuals],
            "metric3d_session_scales": list(metric_scales),
            "metric3d_video_scale": metric_video_scale,
            "metric3d_relative_gauges": list(metric_gauges),
            "metric3d_session_reports": [dict(row) for row in metric_reports],
            "metric3d_evidence": "all_sessions_keyframes_per_session_metric3d_gauge_normalized",
            "metric3d_scale_application": "video_level_scalar_applied_once_after_common_gauge_assembly",
        }
        aggregate_output = replace(
            normalized_outputs[-1],
            disparities=aggregate_disparities,
            scale_provenance={**dict(normalized_outputs[-1].scale_provenance), **aggregate_provenance},
        )
        final_output = _densify_merged_droid_output(aggregate_output, timeline, coverage, merged, mappings, residuals)
        final_output = _scale_droid_output_once(
            final_output, timeline, selected, masks, source, canonical.k_canonical,
            hawor_root=self.config.droid_metric3d_hawor_root,
            metric_checkpoint=self.config.droid_metric3d_checkpoint,
            metric3d_python=self.config.droid_metric3d_python,
            cuda_visible_devices=self.config.droid_metric3d_cuda_visible_devices,
            metric_scalar=metric_video_scale,
            metric_report={"scale": metric_video_scale, "session_scales": list(metric_scales), "session_reports": [dict(row) for row in metric_reports]},
        )
        final_result = replace(finals[-1], output=final_output, provenance=(*finals[-1].provenance, aggregate_provenance))
        finals = (*finals[:-1], final_result)
        outcomes = tuple(
            DroidChunkOutcome(
                chunk_index=index,
                source_indices=chunk,
                session_id=final.output.session_id,
                keyframe_count=final.output.keyframe_count,
                stitch_boundary_translation_error_m=None if index == 0 else float(residuals[index - 1]["translation_residual_m"]),
                stitch_boundary_rotation_error_rad=None if index == 0 else float(residuals[index - 1]["rotation_residual_rad"]),
                attempts=(DroidChunkAttemptOutcome(0, final.output.session_id, _scheduled_droid_options(
                    chunk_index=index, submitted_frame_count=len(chunk),
                    droid_fps=self.config.droid_fps or timeline.fps, attempt=0, filter_thresh=None,
                ), True),),
            )
            for index, (chunk, final) in enumerate(zip(chunks, finals))
        )
        accepted = all(outcome.keyframe_count > 1 for outcome in outcomes)
        blocker = None if accepted else "remote_droid_insufficient_keyframes"
        return DroidExecutionRecords(creates, pushes_by_attempt, finals, 0, accepted, blocker, coverage, outcomes), traces


    def _run_droid_chunk(
        self,
        source: FrameSource,
        case_id: str,
        item_id: str,
        canonical: CanonicalKAggregation,
        unidepth_records: Sequence[AlgorithmResult[UniDepthOutput]],
        chunk_index: int,
        source_indices: tuple[int, ...],
        *,
        attempt: int = 0,
        filter_thresh: float | None = None,
        frontend_thresh: float | None = None,
        backend_thresh: float | None = None,
        static_confidence_masks: Mapping[int, np.ndarray] | None = None,
    ) -> tuple[AlgorithmResult[DroidCreateOutput], tuple[AlgorithmResult[DroidPushOutput], ...], AlgorithmResult[DroidFinalizeOutput], tuple[RequestBatchTrace, ...]]:
        timeline = source.timeline
        input_h, input_w = _droid_input_shape_yx(timeline, self.config)
        model_shape = (input_h // 8, input_w // 8)
        source_to_input_matrix = np.array([[input_w / timeline.width_px, 0.0, 0.0], [0.0, input_h / timeline.height_px, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        geometry = DroidPixelGeometry(canonical.k_canonical, source_to_input_matrix, np.diag([1.0 / 8.0, 1.0 / 8.0, 1.0]), (input_h, input_w), model_shape)
        input_to_source = np.linalg.inv(source_to_input_matrix)
        source_to_input_spatial = SpatialTransform("droid_input", timeline.width_px, timeline.height_px, _matrix_tuple(input_to_source), "source_pixels")
        input_to_model_spatial = SpatialTransform("droid_model", input_w, input_h, _matrix_tuple(np.diag([8.0, 8.0, 1.0])), "droid_input_pixels")
        if len(source_indices) > DROID_SERVICE_PUSH_CAPACITY or tuple(sorted(set(source_indices))) != source_indices:
            raise TimelineDriverError("DROID session source keys must be strictly increasing and fit the 256-push capacity")
        options = _scheduled_droid_options(
            chunk_index=chunk_index,
            submitted_frame_count=len(source_indices),
            droid_fps=self.config.droid_fps or timeline.fps,
            attempt=attempt,
            filter_thresh=filter_thresh,
            frontend_thresh=frontend_thresh,
            backend_thresh=backend_thresh,
        )
        if static_confidence_masks is not None and set(static_confidence_masks) != set(source_indices):
            raise TimelineDriverError("DROID static-confidence masks must cover exactly the scheduled source frames")
        sampled_timeline = timeline.droid_sampled_metadata(source_indices)
        attempt_scope = f"chunk:{chunk_index:03d}:attempt:{attempt}"
        create_input = DroidCreateInput(_ownership(case_id, item_id, timeline.source_id, "droid.create_session", attempt_scope), sampled_timeline, geometry.k_droid_input_four, source_to_input_spatial, input_to_model_spatial, model_shape, self.config.require_rgbd_capability, self.config.allow_monocular_droid_smoke)
        create_request = _request("droid.create_session", case_id, item_id, sampled_timeline, create_input, self.config.model_revisions["droid.create_session"], native_shape=(1,), options=options)
        create_started = time.monotonic()
        create = _typed_result(self._execute_timed(create_request), DroidCreateOutput)
        create_completed = time.monotonic()
        self._validate_droid_capabilities(create.output.capabilities)
        push_started = time.monotonic()
        pushes: list[AlgorithmResult[DroidPushOutput]] = []
        for frame_index in source_indices:
            depth_output = _nearest_unidepth_record(unidepth_records, frame_index).output
            evidence = DepthEvidence(np.asarray(depth_output.depth_m.array[0], dtype=np.float32), np.asarray(depth_output.spatial.pixel_to_source, dtype=np.float64), f"{timeline.source_id}:unidepth:{frame_index:06d}", frame_index, depth_output.spatial.grid_id, np.asarray(depth_output.confidence.array[0], dtype=np.float32), min_confidence=0.0)
            payload = pack_native_sensor_depth(evidence, geometry)
            static_confidence_mask = None
            if static_confidence_masks is not None:
                mask_array = np.ascontiguousarray(static_confidence_masks[frame_index])
                if mask_array.shape != (input_h, input_w) or mask_array.dtype != np.uint8:
                    raise TimelineDriverError("DROID static-confidence mask must be uint8 on the exact DROID input grid")
                if not np.all((mask_array == 0) | (mask_array == 1)):
                    raise TimelineDriverError("DROID dynamic-ignore mask values must be binary 0=static,1=dynamic")
                static_confidence_mask = TypedTensor(
                    mask_array, "probability", "droid_input", "yx", "hand_dynamic_ignore_mask_v1",
                    {"frame_index": frame_index, "source_frame_index": frame_index, "scheduled_chunk": chunk_index, "attempt": attempt, "value_semantics": "1=dynamic_ignore,0=static_keep", "mask_provenance": "wilor_projected_surface_with_tracked_box_fallback"},
                    _matrix_tuple(input_to_source),
                )
            rgb = _resize_rgb(source.read_rgb(frame_index), (input_h, input_w))
            if static_confidence_mask is not None:
                rgb = np.where(static_confidence_mask.array[..., None] > 0, 0, rgb).astype(np.uint8)
            rgb_tensor = TypedTensor(rgb, "uint8_rgb", "droid_input", "hwc", "hand_dynamic_rgb_zeroed_v1", {"frame_index": frame_index, "source_frame_index": frame_index, "scheduled_chunk": chunk_index, "attempt": attempt, "mask_agreement": "rgb_zero_where_dynamic_ignore_positive"}, _matrix_tuple(input_to_source))
            push_input = DroidPushInput(_ownership(case_id, item_id, timeline.source_id, "droid.push_frame", f"{attempt_scope}:frame:{frame_index:06d}"), create.output.session_id, frame_index, timeline.frames[frame_index].timestamp_s, rgb_tensor, payload, geometry.k_droid_input_four, static_confidence_mask, self.config.require_rgbd_capability, self.config.allow_monocular_droid_smoke)
            push_request = _request("droid.push_frame", case_id, item_id, timeline.metadata((frame_index,)), push_input, self.config.model_revisions["droid.push_frame"], native_shape=(1,), options=options)
            push = _typed_result(self._execute_timed(push_request), DroidPushOutput)
            if push.output.session_id != create.output.session_id or push.output.frame_index != frame_index:
                raise StageResultError("DROID scheduled chunk push changed session/frame identity")
            self._validate_droid_capabilities(push.output.capabilities)
            pushes.append(push)
        push_completed = time.monotonic()
        finalize_input = DroidFinalizeInput(_ownership(case_id, item_id, timeline.source_id, "droid.finalize", attempt_scope), create.output.session_id, self.config.require_rgbd_capability, self.config.allow_monocular_droid_smoke)
        finalize_request = _request("droid.finalize", case_id, item_id, sampled_timeline, finalize_input, self.config.model_revisions["droid.finalize"], native_shape=(1,), options=options)
        finalize_started = time.monotonic()
        try:
            final = _typed_result(self._execute_timed(finalize_request), DroidFinalizeOutput)
        except Exception as exc:
            finalize_completed = time.monotonic()
            failure_traces = (
                RequestBatchTrace("droid.create_session", 1, 1, 1, (1,), create_started, create_completed),
                RequestBatchTrace("droid.push_frame", len(source_indices), 1, route_for("droid.push_frame").native_batch_cap, (1,), push_started, push_completed),
                RequestBatchTrace("droid.finalize", 1, 1, 1, (1,), finalize_started, finalize_completed),
            )
            raise DroidChunkFinalizeError(
                chunk_index=chunk_index,
                attempt=attempt,
                session_id=create.output.session_id,
                options=options,
                create_result=create,
                push_results=tuple(pushes),
                traces=failure_traces,
                cause=exc,
            ) from exc
        finalize_completed = time.monotonic()
        if final.output.session_id != create.output.session_id:
            raise StageResultError("DROID scheduled chunk finalization changed session identity")
        self._validate_droid_capabilities(final.output.capabilities)
        return create, tuple(pushes), final, (
            RequestBatchTrace("droid.create_session", 1, 1, 1, (1,), create_started, create_completed),
            RequestBatchTrace("droid.push_frame", len(source_indices), 1, route_for("droid.push_frame").native_batch_cap, (1,), push_started, push_completed),
            RequestBatchTrace("droid.finalize", 1, 1, 1, (1,), finalize_started, finalize_completed),
        )

    def _validate_droid_capabilities(self, capabilities: DroidCapabilities) -> None:
        if self.config.require_rgbd_capability:
            try:
                capabilities.require_rgbd()
            except TypedContractError as exc:
                raise StageResultError(f"remote_droid_capability_mismatch: {exc}") from exc
        else:
            capabilities.diagnostic_monocular()

    def _build_hawor_requests(
        self,
        source: FrameSource,
        case_id: str,
        item_id: str,
        tracks: Sequence[HandTrack],
        canonical: CanonicalKAggregation,
        unidepth_records: Sequence[AlgorithmResult[UniDepthOutput]],
        droid: DroidFinalizeOutput,
    ) -> tuple[tuple[AlgorithmRequest[HaworTrackInput], ...], tuple[HaworChunkTrace, ...]]:
        requests: list[AlgorithmRequest[HaworTrackInput]] = []
        traces: list[HaworChunkTrace] = []
        timeline = source.timeline
        droid_valid = _droid_validity_from_output(droid, timeline.frame_count)
        k_tensor = _typed_K(canonical.k_canonical, timeline.source_id)
        for track in tracks:
            detections = {record.frame_index: record for record in track.detections}
            track_length = track.end_frame - track.start_frame + 1
            for start in self.config.hawor_coverage.starts(track_length, offset=track.start_frame):
                crops: list[np.ndarray] = []
                transforms: list[CropTransform] = []
                frame_slots: list[int] = []
                observed_slots: list[bool] = []
                padded_slots: list[bool] = []
                observation_values: list[tuple[float, float, float, float]] = []
                observation_records: list[HaworObservation] = []
                depth_summary: list[float] = []
                for slot in range(16):
                    frame_index = start + slot
                    padded = frame_index > track.end_frame or frame_index >= timeline.frame_count
                    detection = None if padded else detections.get(frame_index)
                    observed = detection is not None
                    if detection is not None:
                        crop, transform = _normalized_crop(source.read_rgb(frame_index), detection, timeline, self.config.crop_scale)
                        confidence = detection.score
                        uncertainty = detection.uncertainty
                        visibility_code = _visibility_code(detection.occlusion_state)
                        occlusion = detection.occlusion_state
                    else:
                        crop = np.zeros((3, 256, 256), dtype=np.float32)
                        transform = _neutral_crop_transform(track.side, timeline)
                        confidence = 0.0
                        uncertainty = float("inf")
                        visibility_code = _visibility_code("unresolved")
                        occlusion = "unresolved"
                    crops.append(crop)
                    transforms.append(transform)
                    frame_slots.append(-1 if padded else frame_index)
                    observed_slots.append(observed)
                    padded_slots.append(padded)
                    observation_values.append((1.0 if observed else 0.0, confidence, uncertainty, visibility_code))
                    timestamp = -1.0 if padded else timeline.frames[frame_index].timestamp_s
                    wire_frame_index = max(0, min(frame_index, timeline.frame_count - 1))
                    observation_records.append(HaworObservation(wire_frame_index, timestamp, track.side, occlusion, confidence))
                    if padded:
                        depth_summary.append(0.0)
                    else:
                        depth = _nearest_unidepth_record(unidepth_records, frame_index).output.depth_m.array[0]
                        valid = depth[np.isfinite(depth) & (depth > 0)]
                        depth_summary.append(float(np.median(valid)) if valid.size else 0.0)
                crop_array = np.stack(crops, axis=0)[None]
                observations = np.asarray(observation_values, dtype=np.float32)[None]
                real_slots = tuple(index for index in frame_slots if index >= 0)
                if any(not bool(droid_valid[index]) for index in real_slots):
                    # The frozen HaWoR route consumes DROID poses; no tail pose is
                    # fabricated merely to keep an otherwise invalid chunk alive.
                    continue
                pose_indices = [index if index >= 0 else real_slots[-1] for index in frame_slots]
                droid_chunk = TypedTensor(
                    np.asarray(droid.T_world_camera.array[pose_indices], dtype=np.float32),
                    "metres", "world_from_camera", "tij", "droid_observed_chunk_poses_v1",
                    {"frame_slots": frame_slots, "droid_pose_valid": [True] * len(frame_slots)},
                )
                droid_ts = TypedTensor(
                    np.asarray([timeline.frames[index].timestamp_s if index >= 0 else -1.0 for index in frame_slots], dtype=np.float64),
                    "seconds", "source_timeline", "t", "source_timestamps_v1", {"frame_slots": frame_slots},
                )
                ownership = _ownership(case_id, item_id, timeline.source_id, "hawor.infer_tracks", f"track:{track.track_id}:start:{start}")
                value = HaworTrackInput(
                    ownership,
                    TypedTensor(crop_array, "imagenet_normalized", "hawor_crop", "btcyx", "hawor_real_normalized_16_frame_crops_v1", {"track_id": track.track_id, "start_frame": start}),
                    (tuple(frame_slots),),
                    tuple(transforms),
                    TypedTensor(observations, "observation_features", "source_timeline", "btf", "hawor_observation_mask_confidence_uncertainty_visibility_v1", {"padded_slots": padded_slots}),
                    TypedTensor(np.asarray(depth_summary, dtype=np.float32)[None, :, None, None], "metres", "source_depth_summary", "btyx", "unidepth_metric_depth_summary_v1", {"aggregation": "valid_median", "frame_slots": frame_slots}),
                    k_tensor,
                    droid_chunk,
                    droid_ts,
                    tuple(observation_records),
                )
                request = _request("hawor.infer_tracks", case_id, item_id, timeline.metadata(real_slots or (track.start_frame,)), value, self.config.model_revisions["hawor.infer_tracks"], native_shape=(1, 16, 3, 256, 256), chunk_length=16)
                requests.append(request)
                traces.append(HaworChunkTrace(track.track_id, track.side, start, tuple(frame_slots), tuple(observed_slots), tuple(padded_slots), ownership.scope))
        return tuple(requests), tuple(traces)

    def _hawor_candidates(
        self,
        records: Sequence[AlgorithmResult[HaworTrackOutput]],
        chunks: Sequence[HaworChunkTrace],
    ) -> tuple[_ManoCandidate, ...]:
        if len(records) != len(chunks):
            raise StageResultError("HaWoR request/result trace count changed")
        candidates: list[_ManoCandidate] = []
        for result, chunk in zip(records, chunks):
            output = result.output
            root = _time_array(output.root_orient.array, 16)
            pose = _time_array(output.hand_pose.array, 16)
            trans = _time_array(output.trans_camera_m.array, 16)
            vertices = _time_array(output.vertices_camera_m.array, 16)
            joints = _time_array(output.joints_camera_m.array, 16)
            observed = _time_array(output.observed.array, 16).astype(bool)
            uncertainty = _time_array(output.uncertainty_m.array, 16)
            betas_tensor = getattr(output, "betas", None)
            if betas_tensor is None:
                raise StageResultError("HaWoR output lacks reproducible MANO betas")
            betas = _time_array(betas_tensor.array, 16)
            for slot, frame_index in enumerate(chunk.source_frame_slots):
                if frame_index < 0:
                    if bool(observed[slot]):
                        raise StageResultError("HaWoR tail padding was promoted to an observed pose")
                    continue
                is_observed = bool(observed[slot]) and chunk.observed_slots[slot]
                candidates.append(
                    _ManoCandidate(
                        frame_index,
                        chunk.side,
                        root[slot],
                        pose[slot],
                        betas[slot],
                        trans[slot],
                        vertices[slot],
                        joints[slot],
                        is_observed,
                        not is_observed,
                        float(uncertainty[slot]),
                        "hawor.infer_tracks",
                        chunk.request_scope,
                    )
                )
        return tuple(candidates)

    def _build_infiller_requests(
        self,
        case_id: str,
        item_id: str,
        timeline: SourceTimeline,
        canonical: CanonicalKAggregation,
        droid: DroidFinalizeOutput,
        candidates: Sequence[_ManoCandidate],
    ) -> tuple[tuple[AlgorithmRequest[InfillerInput], ...], tuple[InfillerWindowTrace, ...]]:
        anchors = _best_candidates(candidates)
        droid_valid = _droid_validity_from_output(droid, timeline.frame_count)
        requests: list[AlgorithmRequest[InfillerInput]] = []
        traces: list[InfillerWindowTrace] = []
        k_tensor = _typed_K(canonical.k_canonical, timeline.source_id)
        starts = self.config.infiller_coverage.starts(timeline.frame_count)
        for start in starts:
            real_slots = tuple(range(start, min(start + 120, timeline.frame_count)))
            if any(not bool(droid_valid[index]) for index in real_slots):
                frame_slots = tuple(index if index < timeline.frame_count else -1 for index in range(start, start + 120))
                padded_slots = tuple(index < 0 for index in frame_slots)
                traces.append(InfillerWindowTrace(
                    f"window-{start:06d}", start, frame_slots, (False,) * 120, (False,) * 120,
                    padded_slots, False, f"window-{start:06d}:service_capacity_{DROID_SERVICE_PUSH_CAPACITY}_exceeded",
                ))
                continue
            state = np.zeros((120, 218), dtype=np.float32)
            mask = np.zeros((120, 2), dtype=np.uint8)
            timestamps = np.full((120,), -1.0, dtype=np.float64)
            poses = np.tile(np.eye(4, dtype=np.float32), (120, 1, 1))
            frame_slots: list[int] = []
            padded_slots: list[bool] = []
            frame_records: list[InfillerFrame] = []
            left_observed: list[bool] = []
            right_observed: list[bool] = []
            last_valid_left: _ManoCandidate | None = None
            last_valid_right: _ManoCandidate | None = None
            for slot in range(120):
                frame_index = start + slot
                padded = frame_index >= timeline.frame_count
                frame_slots.append(-1 if padded else frame_index)
                padded_slots.append(padded)
                if not padded:
                    timestamps[slot] = timeline.frames[frame_index].timestamp_s
                    poses[slot] = droid.T_world_camera.array[frame_index]
                else:
                    # The frozen Infiller groups frames by timestamp. A shared -1
                    # padding timestamp collapses the tail into one step, so keep
                    # each padded slot distinct while marking it non-source data.
                    timestamps[slot] = _infiller_window_timestamp(timeline, frame_index)
                    poses[slot] = droid.T_world_camera.array[-1]
                for side_index, side in enumerate((HandSide.LEFT, HandSide.RIGHT)):
                    candidate = None if padded else anchors.get((side, frame_index))
                    is_valid_candidate = (
                        candidate is not None
                        and candidate.observed
                        and candidate.is_eligible
                    )
                    last_valid = last_valid_left if side is HandSide.LEFT else last_valid_right
                    if is_valid_candidate:
                        root = candidate.root_orient
                        pose = candidate.hand_pose
                        betas = candidate.betas
                        trans = candidate.trans
                        uncertainty = candidate.uncertainty
                        state[slot, side_index * 109:(side_index + 1) * 109] = _mano_109(candidate)
                        mask[slot, side_index] = 1
                        if side is HandSide.LEFT:
                            last_valid_left = candidate
                        else:
                            last_valid_right = candidate
                    else:
                        if last_valid is not None:
                            root = last_valid.root_orient
                            pose = last_valid.hand_pose
                            betas = last_valid.betas
                            trans = last_valid.trans
                            uncertainty = last_valid.uncertainty
                            state[slot, side_index * 109:(side_index + 1) * 109] = _mano_109(last_valid)
                        else:
                            root = np.eye(3, dtype=np.float32)
                            pose = np.tile(np.eye(3, dtype=np.float32), (15, 1, 1))
                            betas = np.zeros(10, dtype=np.float32)
                            trans = np.zeros(3, dtype=np.float32)
                            uncertainty = 0.08
                        mask[slot, side_index] = 0
                    (left_observed if side is HandSide.LEFT else right_observed).append(is_valid_candidate)
                    frame_records.append(
                        InfillerFrame(
                            -1 if padded else frame_index,
                            timestamps[slot],
                            side,
                            _matrix_tuple(root),
                            tuple(tuple(float(value) for value in row) for row in _rotmat_to_axis_angle(pose)),
                            tuple(float(value) for value in trans),
                            tuple(float(value) for value in betas),
                            is_valid_candidate,
                            uncertainty,
                        )
                    )
            window_id = f"window-{start:06d}"
            blocker: str | None = None
            if not any(left_observed):
                blocker = f"{window_id}:no_left_hawor_anchor"
            elif not any(right_observed):
                blocker = f"{window_id}:no_right_hawor_anchor"
            submitted = blocker is None
            traces.append(InfillerWindowTrace(window_id, start, tuple(frame_slots), tuple(left_observed), tuple(right_observed), tuple(padded_slots), submitted, blocker))
            if not submitted:
                continue
            ownership = _ownership(case_id, item_id, timeline.source_id, "hawor_infiller.fill", window_id)
            value = InfillerInput(
                ownership,
                TypedTensor(state, "coupled_mano_state", "camera", "td", "infiller_two_hand_120x218_v1", {"frame_slots": frame_slots, "tail_policy": "pad_unobserved"}),
                TypedTensor(mask, "boolean", "source_timeline", "th", "infiller_observation_mask_v1", {"side_order": ["left", "right"]}),
                TypedTensor(timestamps, "seconds", "source_timeline", "t", "infiller_window_timestamps_v1", {"frame_slots": frame_slots, "padded_timestamps": "extrapolated_from_last_source_frame"}),
                TypedTensor(poses, "metres", "world_from_camera", "tij", "droid_window_poses_v1", {"frame_slots": frame_slots, "padded_pose": "repeat_last_source_pose"}),
                k_tensor,
                tuple(frame_records),
            )
            real_indices = tuple(index for index in frame_slots if index >= 0)
            requests.append(_request("hawor_infiller.fill", case_id, item_id, timeline.metadata(real_indices), value, self.config.model_revisions["hawor_infiller.fill"], native_shape=(1, 120, 218), temporal_window=120))
        return tuple(requests), tuple(traces)

    def _infiller_candidates(
        self,
        records: Sequence[AlgorithmResult[InfillerOutput]],
        windows: Sequence[InfillerWindowTrace],
    ) -> tuple[_ManoCandidate, ...]:
        if len(records) != len(windows):
            raise StageResultError("Infiller request/result trace count changed")
        candidates: list[_ManoCandidate] = []
        for result, window in zip(records, windows):
            output = result.output
            observed = np.asarray(output.observed.array, dtype=bool)
            inferred = np.asarray(output.inferred.array, dtype=bool)
            uncertainty = np.asarray(output.uncertainty_m.array, dtype=np.float32)
            root_tensor = getattr(output, "root_orient", None)
            pose_tensor = getattr(output, "hand_pose", None)
            betas_tensor = getattr(output, "betas", None)
            trans_tensor = getattr(output, "trans_camera_m", None)
            if any(value is None for value in (root_tensor, pose_tensor, betas_tensor, trans_tensor)):
                root, pose, betas, trans = _decode_infiller_state(output.state_2x120x109.array)
            else:
                root = np.asarray(root_tensor.array)
                pose = np.asarray(pose_tensor.array)
                betas = np.asarray(betas_tensor.array)
                trans = np.asarray(trans_tensor.array)
            vertices_tensor = getattr(output, "vertices_camera_m", None)
            joints_tensor = getattr(output, "joints_camera_m", None)
            vertices = None if vertices_tensor is None else np.asarray(vertices_tensor.array)
            joints = None if joints_tensor is None else np.asarray(joints_tensor.array)
            for slot, frame_index in enumerate(window.source_frame_slots):
                if frame_index < 0:
                    if np.any(observed[:, slot]):
                        raise StageResultError("Infiller tail padding was promoted to observed state")
                    continue
                for side_index, side in enumerate((HandSide.LEFT, HandSide.RIGHT)):
                    candidates.append(
                        _ManoCandidate(
                            frame_index,
                            side,
                            root[side_index, slot],
                            pose[side_index, slot],
                            betas[side_index, slot],
                            trans[side_index, slot],
                            None if vertices is None else vertices[side_index, slot],
                            None if joints is None else joints[side_index, slot],
                            bool(observed[side_index, slot]),
                            bool(inferred[side_index, slot]),
                            float(uncertainty[side_index, slot]),
                            "hawor_infiller.fill",
                            result.output.ownership.scope,
                        )
                    )
        return tuple(candidates)

    def _run_many_traced(
        self,
        stage_id: str,
        requests: Sequence[AlgorithmRequest[Any]],
    ) -> tuple[tuple[AlgorithmResult[Any], ...], RequestBatchTrace]:
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=_stage_worker_count(requests), thread_name_prefix=f"stage-{stage_id}") as pool:
            futures = [pool.submit(self._execute_timed, request) for request in requests]
            results = tuple(future.result() for future in futures)
        completed = time.monotonic()
        route = route_for(stage_id)
        shape = requests[0].work.native_shape if requests else _empty_native_shape(stage_id)
        return results, RequestBatchTrace(stage_id, len(requests), len(requests), route.native_batch_cap, shape, started, completed)


def inspect_video(path: str | os.PathLike[str]) -> SourceTimeline:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise PreflightError(f"video does not exist or is not a file: {source}")
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise PreflightError(f"OpenCV cannot open video: {source}")
    try:
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if frame_count <= 0 or not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        raise PreflightError("video metadata has invalid frame/fps/size values")
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    duration = frame_count / fps
    source_id = f"sha256:{digest.hexdigest()}"
    spatial = _identity_spatial("source_rgb", width, height)
    frames = tuple(SourceFrame(index, index / fps, f"{source_id}:frame:{index:06d}", spatial) for index in range(frame_count))
    return SourceTimeline(source_id, str(source), digest.hexdigest(), size, frame_count, fps, duration, width, height, "RGB", frames)


def preflight_single_video(
    video_path: str | os.PathLike[str],
    *,
    case_id: str,
    fresh_root: str | os.PathLike[str],
    profile: ValidatedVideoProfile | None = None,
) -> SingleVideoPreflight:
    if not case_id:
        raise PreflightError("case_id is required")
    root = Path(fresh_root).expanduser().resolve()
    if root.exists():
        raise PreflightError(f"fresh run root already exists: {root}")
    timeline = inspect_video(video_path)
    checks = ["path", "fresh_root_absent", "sha256", "size_bytes", "frame_count", "fps", "duration_s", "dimensions"]
    if profile is not None:
        if profile.case_id != case_id:
            raise PreflightError(f"validated profile case mismatch: expected {profile.case_id}, got {case_id}")
        actual: Mapping[str, object] = {
            "source_sha256": timeline.source_sha256,
            "source_size_bytes": timeline.source_size_bytes,
            "frame_count": timeline.frame_count,
            "width_px": timeline.width_px,
            "height_px": timeline.height_px,
        }
        expected: Mapping[str, object] = {
            "source_sha256": profile.source_sha256,
            "source_size_bytes": profile.source_size_bytes,
            "frame_count": profile.frame_count,
            "width_px": profile.width_px,
            "height_px": profile.height_px,
        }
        mismatches = [key for key in expected if actual[key] != expected[key]]
        if not math.isclose(timeline.fps, profile.fps, rel_tol=0.0, abs_tol=1e-6):
            mismatches.append("fps")
        if not math.isclose(timeline.duration_s, profile.duration_s, rel_tol=0.0, abs_tol=max(1e-6, 0.5 / profile.fps)):
            mismatches.append("duration_s")
        if mismatches:
            raise PreflightError(f"validated profile mismatch: {sorted(set(mismatches))}")
        checks.append(f"validated_profile:{profile.profile_id}")
    return SingleVideoPreflight(case_id, str(root), timeline, None if profile is None else profile.profile_id, tuple(checks))


def load_validated_profile(path: str | os.PathLike[str], profile_id: str) -> ValidatedVideoProfile:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot load validated profile config: {exc}") from exc
    profiles = payload.get("profiles") if isinstance(payload, Mapping) else None
    if not isinstance(profiles, list):
        raise PreflightError("validated profile config must contain a profiles list")
    for row in profiles:
        if isinstance(row, Mapping) and row.get("profile_id") == profile_id:
            return ValidatedVideoProfile(
                profile_id=str(row["profile_id"]),
                case_id=str(row["case_id"]),
                source_sha256=str(row["source_sha256"]),
                source_size_bytes=int(row["source_size_bytes"]),
                frame_count=int(row["frame_count"]),
                fps=float(row["fps"]),
                duration_s=float(row["duration_s"]),
                width_px=int(row["width_px"]),
                height_px=int(row["height_px"]),
            )
    raise PreflightError(f"validated profile {profile_id!r} is not configured")


def plan_single_video(preflight: SingleVideoPreflight, config: FullVideoDriverConfig) -> dict[str, object]:
    timeline = preflight.timeline
    return {
        "case_id": preflight.case_id,
        "video": timeline.source_path,
        "fresh_root": preflight.fresh_root,
        "profile_id": preflight.profile_id,
        "source": {
            "sha256": timeline.source_sha256,
            "size_bytes": timeline.source_size_bytes,
            "frame_count": timeline.frame_count,
            "fps": timeline.fps,
            "duration_s": timeline.duration_s,
            "width_px": timeline.width_px,
            "height_px": timeline.height_px,
        },
        "backend_mode": "strict_rgbd" if config.require_rgbd_capability else "diagnostic_monocular_unaccepted",
        "cosmos": "enabled" if config.cosmos_enabled else "disabled",
        "item_batch_size": config.item_batch_size,
        "coverage": {
            "droid": {
                "submission": "source_keyed_sessions_at_most_256_pushes_then_fixed_64_frame_exact_overlap_sim3_merge",
                "target_fps": config.droid_fps,
                "none_policy": "every_source_frame",
                "session_count": "one when <=256; otherwise one plus ceil((unique_selected_frames-256)/192)",
                "terminate_count": "one_finalize_per_session",
                "keyframe_buffer": config.droid_keyframe_buffer,
                "session_workers": config.droid_session_workers,
                "effective_unique_coverage": "all selected source frames",
                "actual_push_count": "unique selected frames plus 64 exact canonical overlap pushes per session boundary",
                "dense_pose_validity": "all_source_frames_after_endpoint_inclusive_interpolation",
            },
            "hawor": {"length": 16, "stride": config.hawor_coverage.stride, "tail": config.hawor_coverage.tail},
            "infiller": {"length": 120, "features": 218, "stride": config.infiller_coverage.stride, "tail": config.infiller_coverage.tail},
        },
        "creates_run_root": False,
        "preflight_checks": list(preflight.checks),
    }


def _validate_result_envelope(request: AlgorithmRequest[Any], result: object) -> None:
    if not isinstance(result, AlgorithmResult):
        raise StageResultError(f"{request.algorithm_id} backend returned no typed AlgorithmResult")
    for field_name in ("algorithm_id", "model_revision", "case_id", "item_id", "source_id"):
        if getattr(result, field_name) != getattr(request, field_name):
            raise StageResultError(f"{request.algorithm_id} result changed {field_name}")
    if result.timeline.to_mapping() != request.timeline.to_mapping():
        raise StageResultError(f"{request.algorithm_id} result changed request ownership/timestamps")
    output_ownership = getattr(result.output, "ownership", None)
    input_ownership = getattr(request.input, "ownership", None)
    if output_ownership != input_ownership:
        raise StageResultError(f"{request.algorithm_id} output ownership mismatch")
    output_indices = getattr(result.output, "frame_indices", None)
    output_timestamps = getattr(result.output, "timestamps_s", None)
    if output_indices is not None and tuple(output_indices) != request.timeline.frame_indices:
        raise StageResultError(f"{request.algorithm_id} output frame ownership mismatch")
    if output_timestamps is not None and not np.allclose(output_timestamps, request.timeline.timestamps_s, atol=1e-9, rtol=0.0):
        raise StageResultError(f"{request.algorithm_id} output timestamp mismatch")


def _typed_result(result: AlgorithmResult[Any], output_type: type[Any]) -> AlgorithmResult[Any]:
    if not isinstance(result.output, output_type):
        raise StageResultError(f"{result.algorithm_id} output is not {output_type.__name__}")
    return result


def _typed_results(results: Sequence[AlgorithmResult[Any]], output_type: type[Any]) -> tuple[AlgorithmResult[Any], ...]:
    return tuple(_typed_result(result, output_type) for result in results)


def _ownership(case_id: str, item_id: str, source_id: str, stage_id: str, scope: str) -> Ownership:
    return Ownership(case_id, item_id, source_id, route_for(stage_id).owner, f"{stage_id}:{scope}")


def _request(
    stage_id: str,
    case_id: str,
    item_id: str,
    timeline: FrameTimelineMetadata,
    value: Any,
    revision: str,
    *,
    native_shape: tuple[int, ...],
    chunk_length: int | None = None,
    temporal_window: int | None = None,
    options: Mapping[str, str | int | float | bool] | None = None,
) -> AlgorithmRequest[Any]:
    route = route_for(stage_id)
    native_batch_axis = route.native_batch_axis
    native_batch_size = native_shape[native_batch_axis] if native_batch_axis is not None else 1
    work = NativeWorkDescription(
        work_unit_type=stage_id,
        compatibility_key=f"{stage_id}:{revision}:{native_shape}",
        native_batch_axis=native_batch_axis,
        native_batch_size=native_batch_size,
        native_batch_cap=route.native_batch_cap,
        native_shape=native_shape,
        chunk_length=chunk_length,
        temporal_window=temporal_window,
        outer_item_batch_size=1,
    )
    return AlgorithmRequest(
        algorithm_id=stage_id,
        model_revision=revision,
        case_id=case_id,
        item_id=item_id,
        source_id=timeline.source_id,
        timeline=timeline,
        stage=StageMetadata(stage_id, route.owner, getattr(value, "ownership").scope, revision),
        work=work,
        input=value,
        options=dict(options or {}),
    )


def _identity_spatial(grid_id: str, width: int, height: int) -> SpatialTransform:
    return SpatialTransform(grid_id, width, height, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), "source_pixels")


def _matrix_tuple(matrix: np.ndarray | Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3):
        raise TimelineDriverError("pixel/rotation matrix must be 3x3")
    return tuple(tuple(float(item) for item in row) for row in value)


def _resize_rgb(rgb: np.ndarray, shape_yx: tuple[int, int]) -> np.ndarray:
    value = np.asarray(rgb)
    if value.shape[:2] == shape_yx:
        return np.ascontiguousarray(value)
    import cv2

    return np.ascontiguousarray(cv2.resize(value, (shape_yx[1], shape_yx[0]), interpolation=cv2.INTER_LINEAR))


def _prepare_inference_rgb(rgb: np.ndarray, timeline: SourceTimeline, shape_yx: tuple[int, int] | None) -> tuple[np.ndarray, SpatialTransform]:
    if shape_yx is None:
        return np.ascontiguousarray(rgb), timeline.frames[0].spatial
    output = _resize_rgb(rgb, shape_yx)
    height, width = shape_yx
    pixel_to_source = np.array([[timeline.width_px / width, 0.0, 0.0], [0.0, timeline.height_px / height, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return output, SpatialTransform("inference_rgb", timeline.width_px, timeline.height_px, _matrix_tuple(pixel_to_source), "source_pixels")


def _transform_box(box: np.ndarray, pixel_to_source: np.ndarray) -> np.ndarray:
    corners = np.array([[box[0], box[1], 1.0], [box[2], box[3], 1.0]], dtype=np.float64).T
    transformed = pixel_to_source @ corners
    transformed /= transformed[2:3]
    return np.array([transformed[0, 0], transformed[1, 0], transformed[0, 1], transformed[1, 1]], dtype=np.float64)


def _box_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float64)


def _box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    union = _box_area(a) + _box_area(b) - intersection
    return 0.0 if union <= 0 else intersection / union


def _crop_transform(
    detection: HandDetectionRecord,
    timeline: SourceTimeline,
    crop_scale: float,
) -> tuple[np.ndarray, CropTransform]:
    box = np.asarray(detection.box_xyxy_source, dtype=np.float64)
    center = _box_center(box)
    size = max(float(box[2] - box[0]), float(box[3] - box[1])) * crop_scale
    size = max(size, 2.0)
    source_scale = 256.0 / size
    x_scale = -source_scale if detection.side is HandSide.LEFT else source_scale
    source_to_crop_h = np.array(
        [
            [x_scale, 0.0, 128.0 - x_scale * center[0]],
            [0.0, source_scale, 128.0 - source_scale * center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    crop_to_source = np.linalg.inv(source_to_crop_h)
    spatial = SpatialTransform(f"wilor_crop:{detection.detection_id}", timeline.width_px, timeline.height_px, _matrix_tuple(crop_to_source), "source_pixels")
    return source_to_crop_h[:2].astype(np.float32), CropTransform((float(center[0]), float(center[1])), size, spatial, detection.side)


def _normalize_crop_batch(crops_hwc: np.ndarray) -> np.ndarray:
    crops = np.asarray(crops_hwc).astype(np.float32)
    crops *= np.float32(1.0 / 255.0)
    crops -= np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    crops /= np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return np.ascontiguousarray(crops.transpose(0, 3, 1, 2))


def _normalized_crop_batch(
    source: FrameSource,
    detections: Sequence[HandDetectionRecord],
    timeline: SourceTimeline,
    crop_scale: float,
) -> tuple[np.ndarray, tuple[CropTransform, ...]]:
    import cv2

    by_frame: dict[int, list[int]] = {}
    for detection_index, detection in enumerate(detections):
        by_frame.setdefault(detection.frame_index, []).append(detection_index)
    crops_hwc = np.empty((len(detections), 256, 256, 3), dtype=np.uint8)
    transforms: list[CropTransform | None] = [None] * len(detections)
    covered = 0
    for frame_index, rgb in source.iter_rgb(tuple(sorted(by_frame))):
        frame = np.asarray(rgb)
        for detection_index in by_frame[frame_index]:
            source_to_crop, transform = _crop_transform(detections[detection_index], timeline, crop_scale)
            cv2.warpAffine(
                frame,
                source_to_crop,
                (256, 256),
                dst=crops_hwc[detection_index],
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            transforms[detection_index] = transform
            covered += 1
    if covered != len(detections) or any(transform is None for transform in transforms):
        raise TimelineDriverError("sequential WiLoR frame decode did not cover every detection")
    return _normalize_crop_batch(crops_hwc), tuple(transform for transform in transforms if transform is not None)


def _normalized_crop(
    rgb: np.ndarray,
    detection: HandDetectionRecord,
    timeline: SourceTimeline,
    crop_scale: float,
) -> tuple[np.ndarray, CropTransform]:
    import cv2

    source_to_crop, transform = _crop_transform(detection, timeline, crop_scale)
    crop_hwc = cv2.warpAffine(np.asarray(rgb), source_to_crop, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _normalize_crop_batch(crop_hwc[None])[0], transform


def _neutral_crop_transform(side: HandSide, timeline: SourceTimeline) -> CropTransform:
    center = (timeline.width_px / 2.0, timeline.height_px / 2.0)
    size = float(max(timeline.width_px, timeline.height_px))
    crop_to_source = np.array([[size / 256.0, 0.0, center[0] - size / 2.0], [0.0, size / 256.0, center[1] - size / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    spatial = SpatialTransform("unobserved_crop", timeline.width_px, timeline.height_px, _matrix_tuple(crop_to_source), "source_pixels")
    return CropTransform(center, size, spatial, side)


def _typed_K(k: np.ndarray, source_id: str) -> TypedTensor:
    return TypedTensor(np.asarray(k, dtype=np.float32), "pixels", "source_pixels", "ij", "canonical_full_K_v1", {"source_id": source_id, "parameters": ["fx", "fy", "cx", "cy"]})


def _visibility_code(value: str) -> float:
    return {"visible": 0.0, "partially_visible": 1.0, "occluded": 2.0, "out_of_frame": 3.0, "unresolved": 4.0}.get(value, 4.0)


def _infiller_window_timestamp(timeline: SourceTimeline, frame_index: int) -> float:
    if frame_index < timeline.frame_count:
        return float(timeline.frames[frame_index].timestamp_s)
    return float(timeline.frames[-1].timestamp_s + (frame_index - timeline.frame_count + 1) / timeline.fps)


def _time_array(array: np.ndarray, length: int) -> np.ndarray:
    value = np.asarray(array)
    if value.shape[0] == 1 and value.ndim >= 2 and value.shape[1] == length:
        value = value[0]
    if value.shape[0] != length:
        raise StageResultError(f"temporal model output must contain {length} frames")
    return value


def _rotmat_to_rot6d(rotmat: np.ndarray) -> np.ndarray:
    value = np.asarray(rotmat, dtype=np.float32)
    if value.shape[-2:] != (3, 3):
        raise StageResultError("rotation matrix must end in 3x3")
    return np.ascontiguousarray(value[..., :2].reshape(*value.shape[:-2], 6))


def _rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    value = np.asarray(rot6d, dtype=np.float32)
    raw = value.reshape(-1, 3, 2)
    a1, a2 = raw[:, :, 0], raw[:, :, 1]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    a2 = a2 - np.sum(a2 * b1, axis=-1, keepdims=True) * b1
    b2 = a2 / np.maximum(np.linalg.norm(a2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1).reshape(*value.shape[:-1], 3, 3)


def _rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    matrices = np.asarray(rotmat, dtype=np.float64).reshape(-1, 3, 3)
    result = np.zeros((len(matrices), 3), dtype=np.float64)
    for index, matrix in enumerate(matrices):
        cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
        angle = math.acos(cosine)
        if angle <= 1e-8:
            continue
        axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]], dtype=np.float64)
        denominator = 2.0 * math.sin(angle)
        if abs(denominator) <= 1e-8:
            continue
        result[index] = axis / denominator * angle
    return result.reshape(*np.asarray(rotmat).shape[:-2], 3).astype(np.float32)


def _mano_109(candidate: _ManoCandidate) -> np.ndarray:
    return np.concatenate((
        np.asarray(candidate.trans, dtype=np.float32).reshape(3),
        np.asarray(candidate.betas, dtype=np.float32).reshape(10),
        _rotmat_to_rot6d(candidate.root_orient).reshape(6),
        _rotmat_to_rot6d(candidate.hand_pose).reshape(90),
    ))


def _decode_infiller_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(state, dtype=np.float32)
    if value.shape != (2, 120, 109):
        raise StageResultError("Infiller state must be [2,120,109]")
    trans = value[..., :3]
    betas = value[..., 3:13]
    root = _rot6d_to_rotmat(value[..., 13:19])
    pose = _rot6d_to_rotmat(value[..., 19:109].reshape(2, 120, 15, 6))
    return root, pose, betas, trans


def _hawor_geometry_diagnostics(
    records: Sequence[AlgorithmResult[HaworTrackOutput]],
    chunks: Sequence[HaworChunkTrace],
) -> Mapping[str, object]:
    if len(records) != len(chunks):
        raise StageResultError("HaWoR request/result trace count changed while reporting geometry anomalies")
    anomaly_slots: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []
    for result, chunk in zip(records, chunks):
        vertices = _time_array(result.output.vertices_camera_m.array, 16)
        joints = _time_array(result.output.joints_camera_m.array, 16)
        chunk_slots: list[dict[str, object]] = []
        for slot, frame_index in enumerate(chunk.source_frame_slots):
            codes = _geometry_anomaly_codes(vertices[slot], joints[slot])
            if not codes:
                continue
            row = {
                "request_scope": chunk.request_scope,
                "track_id": chunk.track_id,
                "chunk_start_frame": chunk.start_frame,
                "slot": slot,
                "frame_index": None if frame_index < 0 else frame_index,
                "side": chunk.side.value,
                "observed_input": bool(chunk.observed_slots[slot]),
                "padded_input": bool(chunk.padded_slots[slot]),
                "anomaly_codes": list(codes),
            }
            anomaly_slots.append(row)
            chunk_slots.append(row)
        if chunk_slots:
            chunk_rows.append({
                "request_scope": chunk.request_scope,
                "track_id": chunk.track_id,
                "chunk_start_frame": chunk.start_frame,
                "side": chunk.side.value,
                "degenerate_slot_count": len(chunk_slots),
                "slots": chunk_slots,
                "message": f"HaWoR output was degenerate for {len(chunk_slots)} slots in chunk {chunk.request_scope}",
            })
    return {
        "status": "anomalies_detected" if anomaly_slots else "ok",
        "hawor_degenerate_slot_count": len(anomaly_slots),
        "hawor_degenerate_slots": anomaly_slots,
        "hawor_chunks_with_degenerate_geometry": chunk_rows,
    }


def _best_candidates(candidates: Sequence[_ManoCandidate]) -> dict[tuple[HandSide, int], _ManoCandidate]:
    best: dict[tuple[HandSide, int], _ManoCandidate] = {}
    for candidate in candidates:
        key = (candidate.side, candidate.frame_index)
        previous = best.get(key)
        if previous is None or candidate.rank < previous.rank:
            best[key] = candidate
    return best


def _merge_timeline_candidates(frame_count: int, candidates: Sequence[_ManoCandidate]) -> TimelineInferenceState:
    best = _best_candidates(candidates)
    root = np.full((2, frame_count, 3, 3), np.nan, dtype=np.float32)
    pose = np.full((2, frame_count, 15, 3, 3), np.nan, dtype=np.float32)
    betas = np.full((2, frame_count, 10), np.nan, dtype=np.float32)
    trans = np.full((2, frame_count, 3), np.nan, dtype=np.float32)
    vertices = np.full((2, frame_count, 778, 3), np.nan, dtype=np.float32)
    joint_count = max((candidate.joints.shape[-2] for candidate in best.values() if candidate.joints is not None), default=21)
    joints = np.full((2, frame_count, joint_count, 3), np.nan, dtype=np.float32)
    vertices_source_px = np.full((2, frame_count, 778, 2), np.nan, dtype=np.float32)
    joints_source_px = np.full((2, frame_count, joint_count, 2), np.nan, dtype=np.float32)
    valid = np.zeros((2, frame_count), dtype=np.uint8)
    observed = np.zeros((2, frame_count), dtype=np.uint8)
    inferred = np.zeros((2, frame_count), dtype=np.uint8)
    uncertainty = np.full((2, frame_count), np.inf, dtype=np.float32)
    visibility = [["unresolved"] * frame_count for _ in range(2)]
    provenance: list[TimelineFrameProvenance] = []
    for (side, frame_index), candidate in sorted(best.items(), key=lambda item: (item[0][1], item[0][0].value)):
        side_index = 0 if side is HandSide.LEFT else 1
        root[side_index, frame_index] = candidate.root_orient
        pose[side_index, frame_index] = candidate.hand_pose
        betas[side_index, frame_index] = candidate.betas
        trans[side_index, frame_index] = candidate.trans
        if candidate.vertices is not None:
            vertices[side_index, frame_index] = candidate.vertices
        if candidate.joints is not None:
            count = min(joint_count, candidate.joints.shape[-2])
            joints[side_index, frame_index, :count] = candidate.joints[:count]
        if candidate.vertices_source_px is not None:
            vertices_source_px[side_index, frame_index] = candidate.vertices_source_px
        if candidate.joints_source_px is not None:
            count = min(joint_count, candidate.joints_source_px.shape[-2])
            joints_source_px[side_index, frame_index, :count] = candidate.joints_source_px[:count]
        valid[side_index, frame_index] = 1 if candidate.is_eligible else 0
        observed[side_index, frame_index] = 1 if candidate.observed else 0
        inferred[side_index, frame_index] = 1 if candidate.inferred else 0
        uncertainty[side_index, frame_index] = candidate.uncertainty
        visibility[side_index][frame_index] = "visible" if candidate.observed else "occluded_inferred"
        provenance.append(TimelineFrameProvenance(frame_index, side, candidate.source_stage, candidate.source_scope, candidate.observed, candidate.inferred, candidate.uncertainty))
    tensor_args = {"provenance": {"merge_policy": "observed_then_renderable_geometry_then_low_uncertainty_preserve_source"}}
    return TimelineInferenceState(
        frame_count,
        (HandSide.LEFT, HandSide.RIGHT),
        TypedTensor(root, "rotation", "camera", "htij", "timeline_mano_root_orient_v1", **tensor_args),
        TypedTensor(pose, "rotation", "camera", "htjij", "timeline_mano_hand_pose_v1", **tensor_args),
        TypedTensor(betas, "shape", "mano", "htd", "timeline_mano_betas_v1", **tensor_args),
        TypedTensor(trans, "metres", "camera", "htx", "timeline_mano_translation_v1", **tensor_args),
        TypedTensor(vertices, "metres", "camera", "htvx", "timeline_mano_vertices_v1", **tensor_args),
        TypedTensor(joints, "metres", "camera", "htjx", "timeline_mano_joints_v1", **tensor_args),
        TypedTensor(vertices_source_px, "pixels", "source_pixels", "htvu", "timeline_mano_vertices_source_px_v1", **tensor_args),
        TypedTensor(joints_source_px, "pixels", "source_pixels", "htju", "timeline_mano_joints_source_px_v1", **tensor_args),
        TypedTensor(valid, "boolean", "source_timeline", "ht", "timeline_mano_valid_v1", **tensor_args),
        TypedTensor(observed, "boolean", "source_timeline", "ht", "timeline_mano_observed_v1", **tensor_args),
        TypedTensor(inferred, "boolean", "source_timeline", "ht", "timeline_mano_inferred_v1", **tensor_args),
        TypedTensor(uncertainty, "metres", "camera", "ht", "timeline_mano_uncertainty_v1", **tensor_args),
        (tuple(visibility[0]), tuple(visibility[1])),
        tuple(provenance),
        "observed_then_renderable_geometry_then_low_uncertainty_preserve_source",
    )


def _empty_timeline_state(frame_count: int) -> TimelineInferenceState:
    return _merge_timeline_candidates(frame_count, ())


def _empty_native_shape(stage_id: str) -> tuple[int, ...]:
    if stage_id == "wilor.reconstruct":
        return (1, 3, 256, 256)
    if stage_id == "hawor.infer_tracks":
        return (1, 16, 3, 256, 256)
    if stage_id == "hawor_infiller.fill":
        return (1, 120, 218)
    return (1,)


__all__ = [
    "AlgorithmAcceptance",
    "AlgorithmBackend",
    "AlgorithmStageClient",
    "CoveragePolicy",
    "DroidExecutionRecords",
    "FrameSource",
    "FullVideoAlgorithmState",
    "FullVideoDriverConfig",
    "FullVideoTimelineDriver",
    "HandDetectionRecord",
    "HandTrack",
    "HaworChunkTrace",
    "InMemoryFrameSource",
    "InfillerWindowTrace",
    "LiveFrozenApiStageClient",
    "OpenCvFrameSource",
    "PreflightError",
    "RequestBatchTrace",
    "ScriptAlgorithmStageClient",
    "SingleVideoPreflight",
    "SourceFrame",
    "SourceTimeline",
    "StageConfigurationError",
    "StageResultError",
    "TimelineDriverError",
    "TimelineFrameProvenance",
    "TimelineInferenceState",
    "ValidatedVideoProfile",
    "inspect_video",
    "load_validated_profile",
    "plan_single_video",
    "preflight_single_video",
]
