"""Ray-free resident DROID adapter: one DroidNet, isolated per-session state.

This module is importable without Ray installed (ordinary unit tests import
``DroidAdapter`` and a fake backend directly). The Ray Serve deployment wrapper
lives in ``ego_annotation.serving.droid_deployment`` and imports this adapter.

Design invariants (validated against the real ``droid_slam`` source):

* **One resident DroidNet / CUDA backend.** The adapter loads ``DroidNet`` once at
  replica startup (``model_load_count == 1``) and shares its ``fnet``/``cnet``/
  ``update`` modules across every session. A request whose ``model_revision`` does
  not match the resident revision is rejected before any work.

* **Isolated mutable state per explicit session.** Each session owns its own
  ``DepthVideo`` (keyframe images/poses/disps/intrinsics/masks/fmaps), its own
  motion-filter state (previous keyframe ``fmap``/``net``/``inp``), its own
  ``DroidFrontend`` factor graph, ``DroidBackend``, and ``PoseTrajectoryFiller``.
  Cross-session state never aliases. Unknown / closed / finalized sessions are
  rejected.

* **True cross-session feature-network batching.** Only ``fnet`` is batchable
  across sessions: compatible next-ready frames from distinct sessions stack into
  one ``[B,C,H,W]`` forward. At most one ready frame per session enters a batch.
  Correlation, recurrent update, context-network for newly added keyframes,
  factor-graph mutation, and bundle adjustment are session-local and are traced as
  session-local forwards — never reported as a fused batch.

* **Canonical pose convention.** DROID stores keyframe poses internally as
  camera-from-world. ``Droid.terminate()`` returns
  ``camera_trajectory.inv()`` = world-from-camera. Both dense (trajectory-filler)
  and keyframe exports invert the same canonical ``DepthVideo.poses`` to
  ``T_world_camera``; ``T_camera_world`` is the matrix inverse of that, so the two
  directions can never disagree. The synthetic nonidentity SE(3) test in the test
  suite forces this: a sign/quaternion-inversion error makes dense and keyframe
  disagree under any nonidentity transform.

* **Monocular scale honesty.** DROID translation is up to scale. Every finalize
  result declares ``scale_status="up_to_scale"``; metric depth requires the shared
  UniDepth camera-scale step, which is outside this model contract.

* **Source-timestamp preservation.** The caller-declared ``source_timestamp_s`` is
  carried through ``push_frame`` into the keyframe/dense timestamp mapping so
  downstream timeline joins use source time, never assumed frame-index agreement.

The real DROID frontend/backend in this repo's ``droid.py`` run synchronously
inside ``track``/``terminate`` (the visualizer is the only ``Process``; it is
disabled with ``disable_vis=True``). A single Serve replica therefore owns one
DroidNet plus N isolated session objects without multiprocessing spawn.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from ego_annotation.serving.batch_snapshot import BatchSnapshotTracker, ExecutionSpec
from ego_annotation.serving.batching import BatchPolicy
logger = logging.getLogger(__name__)

from ego_annotation.serving.contracts import (
    ContractValidationError,
    DeploymentStatus,
    DroidBatchTrace,
    DroidCamera,
    DroidCreateSessionRequest,
    DroidCreateSessionResponse,
    DroidFinalizeRequest,
    DroidFinalizeResponse,
    DroidFrameRequest,
    DroidFrameResponse,
    DroidImageShape,
    DroidPhaseTiming,
    DroidSessionOptions,
    DroidUncertainty,
    ErrorCode,
    FrameValidity,
    Ownership,
    ServiceError,
    ServerIdentity,
    SCHEMA_VERSION,
    StepStatus,
    TensorPayload,
    CameraState,
    KeyframeSourceMapping,
    DenseSourceMapping,
)


# --------------------------------------------------------------------------- #
# Backend protocol and fake backend (tests inject a fake; the replica injects
# the real DroidNet wrapper below). All methods operate on torch tensors on GPU.
# --------------------------------------------------------------------------- #


class DroidBackend(Protocol):
    """The shared model boundary. One instance backs every session."""

    def fnet(self, images: Any) -> Any:
        """Feature network. ``images`` is ``[B,C,H,W]`` normalized float; returns ``[B,128,gh,gw]``."""

    def cnet(self, image: Any) -> tuple[Any, Any]:
        """Context network for a single keyframe image ``[1,C,H,W]``; returns (net, inp)."""

    @property
    def update_op(self) -> Any:
        """The recurrent update operator (shared across sessions)."""


TensorResolver = Callable[[Any, tuple[int, ...], str], Any]
BackendFactory = Callable[["DroidModelConfig"], DroidBackend]
SessionFactory = Callable[[DroidBackend, "DroidModelConfig", DroidCamera, DroidImageShape, DroidSessionOptions], Mapping[str, Any]]


# --------------------------------------------------------------------------- #
# Pure-numpy pose convention helpers. These are the single canonical conversion
# from DROID's internal camera-from-world poses to world-from-camera 4x4
# matrices. Both the dense and keyframe exports use these so the two directions
# can never disagree. The synthetic nonidentity SE(3) test exercises them
# directly without torch/lietorch.
# --------------------------------------------------------------------------- #


def _quat_xyzw_to_matrix(qxyzw: "Any") -> "Any":
    """Convert a single [qx,qy,qz,qw] quaternion to a 3x3 rotation (numpy)."""
    import numpy as np

    qx, qy, qz, qw = float(qxyzw[0]), float(qxyzw[1]), float(qxyzw[2]), float(qxyzw[3])
    nrm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if nrm == 0.0:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / nrm, qy / nrm, qz / nrm, qw / nrm
    return np.asarray([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def camera_from_world_xyzw_to_world_camera_matrix(poses_xyzw: "Any") -> "Any":
    """Canonical conversion: DROID camera-from-world [N,7] -> world-from-camera [N,4,4].

    DROID stores keyframe poses as ``[tx,ty,tz,qx,qy,qz,qw]`` camera-from-world.
    ``Droid.terminate()`` returns ``camera_trajectory.inv()`` = world-from-camera.
    This function materializes that inverse as 4x4 matrices using one code path,
    so dense and keyframe exports are guaranteed consistent. A translation-sign or
    quaternion-inversion error here fails the nonidentity SE(3) test.
    """
    import numpy as np

    arr = np.asarray(poses_xyzw, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    n = arr.shape[0]
    out = np.zeros((n, 4, 4), dtype=np.float64)
    out[:, 3, 3] = 1.0
    for i in range(n):
        R_cw = _quat_xyzw_to_matrix(arr[i, 3:7])
        t_cw = arr[i, :3]
        # World-from-camera = inverse of camera-from-world: R_wc = R_cw^T, t_wc = -R_cw^T t_cw
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        out[i, :3, :3] = R_wc
        out[i, :3, 3] = t_wc
    return out


@dataclass(frozen=True)
class DroidModelConfig:
    """Server-owned model settings; no request field can select a server path."""

    weights: str
    model_revision: str
    device: str = "cuda:0"
    replica_id: str = "droid-gpu2"
    assigned_gpu: int = 2
    # Experiment-only identity fields are all-or-none. Production leaves them null.
    experiment_id: str | None = None
    application_release_path: str | None = None
    gcs_address: str | None = None
    http_port: int | None = None
    temp_dir: str | None = None
    # Recovered-source experiment fields are all-or-none. Production deliberately
    # leaves them absent and continues to use its already-resident historical worker.
    droid_source_release_path: str | None = None
    droid_source_digest: str | None = None
    droid_source_amendment_id: str | None = None
    # Bounds that protect GPU2 memory and enforce bounded rejection. Each session
    # allocates a DepthVideo buffer; bounding sessions bounds resident CUDA memory.
    max_sessions: int = 8
    max_queued_frames_per_session: int = 4
    # One fnet compatibility bucket is (image_shape, revision). The cross-session
    # fnet batch admits at most one ready frame per session, so max_fnet_batch_size
    # also caps concurrent sessions in one forward.
    max_fnet_batch_size: int = 8
    fnet_batch_wait_timeout_s: float = 0.5
    # An OPEN session is abandoned after its last successful create/push response
    # has not been followed by another lifecycle request for this interval.
    idle_session_ttl_s: float = 60.0
    idle_reaper_interval_s: float = 10.0
    # Every accepted frame result remains addressable for the session lifetime.
    # Session buffer <= journal bound makes duplicate-frame mutation impossible
    # without an unbounded cache. Terminal sessions retain their finalize result
    # in a separate count-bounded LRU tombstone set.
    max_result_journal_entries_per_session: int = 8193
    # The caller's ``options.buffer`` is only the initial allocation.  The
    # serving layer may grow a session up to this hard bound while quiescent.
    max_buffer_slots: int = 8192
    max_terminal_tombstones: int = 128
    # Experiment-only placement policy. The default is the established
    # GPU-resident DepthVideo path; offload is enabled solely by an explicit
    # experiment environment flag in the deployment wrapper.
    cpu_offload: bool = False
    # Concurrent BA is scoped to the GPU-resident path ONLY. The CPU-offload
    # H2D transfer + tensor lifetime interaction under concurrency caused the
    # off8/off9 illegal-memory-access regression; offload stays serialized.
    # When cpu_offload=False, up to this many independent GPU-resident sessions
    # may overlap BA on separate CUDA streams. 1 = serialized (production).
    max_concurrent_ba: int = 1
    # Explicit experiment/runtime attestation. The deployment dual-accepts both
    # HTTP framings; this records which framing a caller deliberately selected.
    performance_instrumentation: bool = False
    wire_format: str = "multipart"
    batch_policy: BatchPolicy = BatchPolicy(
        max_batch_size=8,
        batch_wait_timeout_s=2.0,
        max_queued_requests=64,
    )

    def __post_init__(self) -> None:
        if self.assigned_gpu < 0:
            raise ContractValidationError("assigned_gpu must be non-negative")
        if not self.weights or not self.model_revision:
            raise ContractValidationError("DROID weights and model_revision are required server configuration")
        identity_fields = (self.experiment_id, self.application_release_path, self.gcs_address, self.http_port, self.temp_dir)
        if any(value is not None for value in identity_fields) and any(value is None or value == "" for value in identity_fields):
            raise ContractValidationError("experimental DROID identity requires experiment/release/GCS/HTTP/temp fields together")
        source_fields = (self.droid_source_release_path, self.droid_source_digest, self.droid_source_amendment_id)
        if any(value is not None for value in source_fields) and any(value is None or value == "" for value in source_fields):
            raise ContractValidationError("DROID source release/digest/amendment fields are all-or-none")
        if self.experiment_id is not None and any(value is None for value in source_fields):
            raise ContractValidationError("experimental DROID requires a verified immutable source release identity")
        if self.max_sessions <= 0 or self.max_fnet_batch_size <= 0:
            raise ContractValidationError("DROID session/batch bounds must be positive")
        if self.fnet_batch_wait_timeout_s <= 0:
            raise ContractValidationError("DROID fnet batch wait timeout must be positive")
        if self.idle_session_ttl_s <= 0 or self.idle_reaper_interval_s <= 0:
            raise ContractValidationError("DROID idle session TTL and reaper interval must be positive")
        if self.max_result_journal_entries_per_session <= 0 or self.max_buffer_slots <= 0 or self.max_terminal_tombstones <= 0:
            raise ContractValidationError("DROID journal, buffer, and terminal-tombstone bounds must be positive")
        if self.wire_format not in {"multipart", "envelope"}:
            raise ContractValidationError("wire_format must be multipart or envelope")

    def runtime_config_wire(self) -> dict[str, object]:
        """Server-owned policy evidence, including the selected HTTP framing."""
        return {
            "schema": "ego.droid-runtime-config.v1",
            "batch_policy": {
                "max_batch_size": self.batch_policy.max_batch_size,
                "batch_wait_timeout_ms": round(self.batch_policy.batch_wait_timeout_s * 1_000.0, 6),
                "max_queued_requests": self.batch_policy.max_queued_requests,
            },
            "max_sessions": self.max_sessions,
            "max_buffer_slots": self.max_buffer_slots,
            "max_queued_frames_per_session": self.max_queued_frames_per_session,
            "max_fnet_batch_size": self.max_fnet_batch_size,
            "fnet_batch_wait_timeout_ms": round(self.fnet_batch_wait_timeout_s * 1_000.0, 6),
            "idle_session_ttl_s": self.idle_session_ttl_s,
            "idle_reaper_interval_s": self.idle_reaper_interval_s,
            "cpu_offload": self.cpu_offload,
            # Concurrent BA applies only to the GPU-resident path; offload is serialized.
            "max_concurrent_ba": self.max_concurrent_ba,
            "effective_max_concurrent_ba": self.max_concurrent_ba if not self.cpu_offload else 1,
            "wire_format": self.wire_format,
        }

    def runtime_config_digest(self) -> str:
        raw = json.dumps(self.runtime_config_wire(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(raw).hexdigest()


def expected_droid_runtime_config(*, wire_format: str = "multipart") -> dict[str, object]:
    """Build the stable runtime attestation without loading DROID or CUDA."""
    config = DroidModelConfig(weights="runtime-config-only", model_revision="runtime-config-only", wire_format=wire_format)
    return {"runtime_config": config.runtime_config_wire(), "runtime_config_digest": config.runtime_config_digest()}


def build_droid_model_config(
    *,
    weights: str,
    model_revision: str,
    device: str = "cuda:0",
    replica_id: str = "droid-gpu2",
    assigned_gpu: int = 2,
    experiment_id: str | None = None,
    application_release_path: str | None = None,
    gcs_address: str | None = None,
    http_port: int | None = None,
    temp_dir: str | None = None,
    droid_source_release_path: str | None = None,
    droid_source_digest: str | None = None,
    droid_source_amendment_id: str | None = None,
    max_sessions: int = 8,
    max_queued_frames_per_session: int = 4,
    max_fnet_batch_size: int = 8,
    fnet_batch_wait_timeout_s: float = 0.5,
    idle_session_ttl_s: float = 60.0,
    idle_reaper_interval_s: float = 10.0,
    max_result_journal_entries_per_session: int = 8193,
    max_buffer_slots: int = 8192,
    max_terminal_tombstones: int = 128,
    cpu_offload: bool = False,
    max_concurrent_ba: int = 1,
    performance_instrumentation: bool = False,
    wire_format: str = "multipart",
    batch_policy: BatchPolicy | None = None,
) -> DroidModelConfig:
    return DroidModelConfig(
        weights=weights,
        model_revision=model_revision,
        device=device,
        replica_id=replica_id,
        assigned_gpu=assigned_gpu,
        experiment_id=experiment_id,
        application_release_path=application_release_path,
        gcs_address=gcs_address,
        http_port=http_port,
        temp_dir=temp_dir,
        droid_source_release_path=droid_source_release_path,
        droid_source_digest=droid_source_digest,
        droid_source_amendment_id=droid_source_amendment_id,
        max_sessions=max_sessions,
        max_queued_frames_per_session=max_queued_frames_per_session,
        max_fnet_batch_size=max_fnet_batch_size,
        fnet_batch_wait_timeout_s=fnet_batch_wait_timeout_s,
        idle_session_ttl_s=idle_session_ttl_s,
        idle_reaper_interval_s=idle_reaper_interval_s,
        max_result_journal_entries_per_session=max_result_journal_entries_per_session,
        max_buffer_slots=max_buffer_slots,
        max_terminal_tombstones=max_terminal_tombstones,
        cpu_offload=cpu_offload,
        max_concurrent_ba=max_concurrent_ba,
        performance_instrumentation=performance_instrumentation,
        wire_format=wire_format,
        batch_policy=batch_policy or BatchPolicy(max_batch_size=8, batch_wait_timeout_s=2.0, max_queued_requests=64),
    )


# --------------------------------------------------------------------------- #
# Tensor resolution (same shape as the UniDepth resolver; importable w/o torch).
# --------------------------------------------------------------------------- #


def _default_tensor_resolver(data: Any, shape: tuple[int, ...], dtype: str) -> Any:
    import numpy as np

    if isinstance(data, (bytes, bytearray, memoryview)):
        array = np.frombuffer(data, dtype=np.dtype(dtype))
        if array.size != int(np.prod(shape)):
            raise ContractValidationError("binary tensor byte length does not match shape and dtype")
        return array.reshape(shape)
    array = np.asarray(data)
    if tuple(array.shape) != shape:
        raise ContractValidationError("in-cluster tensor shape does not match contract metadata")
    return array


def _as_tensor_payload(value: Any) -> TensorPayload:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    return TensorPayload(data=array.tobytes(), shape=tuple(int(dim) for dim in array.shape), dtype=array.dtype.name)


# --------------------------------------------------------------------------- #
# Per-session state. This is the isolated mutable state the task requires. It is
# deliberately a plain class (not a Ray actor) so the single Serve replica owns
# the concurrency boundary; the adapter serializes session access.
# --------------------------------------------------------------------------- #


@dataclass
class _ContinuationPhaseTimings:
    """Mutable per-session host timings collected by the real continuation."""

    correlation_update_s: float | None = None
    cnet_s: float | None = None
    frontend_ba_s: float | None = None

    def add(self, name: str, elapsed_s: float) -> None:
        current = getattr(self, name)
        setattr(self, name, elapsed_s if current is None else current + elapsed_s)


class _SessionLifecycle(str, Enum):
    """Durable per-session ownership state; terminal tombstones remain queryable."""

    OPEN = "open"
    FINALIZING = "finalizing"
    QUARANTINED = "quarantined"
    UNRESOLVED = "unresolved"
    FINALIZED = "finalized"


@dataclass
class _ResultJournalEntry:
    """One stable operation owner and its eventually durable typed outcome."""

    signature: str
    response: DroidFrameResponse | DroidFinalizeResponse | None = None


_DEPTH_VIDEO_TENSOR_FIELDS = (
    "tstamp", "images", "dirty", "red", "poses", "disps", "disps_sens",
    "disps_up", "intrinsics", "masks", "fmaps", "nets", "inps",
)


@dataclass
class _CpuOffloadState:
    """CPU virtual buffer and transient active-window accounting for one video.

    ``cpu_capacity_bytes`` is the complete DepthVideo index-space.  It is virtual
    storage at create time, not a claim that its pages have been faulted or pinned.
    ``cpu_resident_bytes`` grows only as a prefix is initialized for BA work.
    """

    cpu_tensors: dict[str, Any]
    capacity: int
    cpu_resident_bytes: int = 0
    cpu_capacity_bytes: int = 0
    materialized_slots: int = 0
    gpu_resident_bytes: int = 0
    gpu_peak_resident_bytes: int = 0
    active_slots: int = 0
    last_transfer_ms: float = 0.0


class _DepthVideoCpuOffloader:
    """Keep DepthVideo tensors off GPU while materializing only touched prefixes.

    Upstream DROID indexes its video tensors directly.  A new ``DepthVideo`` has
    deterministic defaults (zero tensors, unit disparity, identity poses), so
    copying its fully initialized GPU buffer to CPU at session creation is wasted
    work.  This controller preserves the complete tensor index-space with empty
    CPU tensors, then initializes and pins/stages only the prefix about to enter a
    session-local continuation or BA call.  Existing prefix state is copied back
    after each active window; untouched suffixes receive their exact constructor
    defaults on their first use.
    """

    def __init__(self, device: str) -> None:
        self._device = device

    @staticmethod
    def _tensor_bytes(tensor: Any) -> int:
        numel = getattr(tensor, "numel", None)
        element_size = getattr(tensor, "element_size", None)
        if callable(numel) and callable(element_size):
            return int(numel()) * int(element_size())
        nbytes = getattr(tensor, "nbytes", None)
        return int(nbytes) if isinstance(nbytes, int) else 0

    @staticmethod
    def _is_torch_tensor(value: Any) -> bool:
        return hasattr(value, "detach") and hasattr(value, "to") and hasattr(value, "device")

    @staticmethod
    def _default_prefix(name: str, tensor: Any, start: int, stop: int) -> None:
        """Recreate the corresponding fresh DepthVideo constructor values."""
        prefix = tensor[start:stop]
        prefix.zero_()
        if name == "disps":
            prefix.fill_(1.0)
        elif name == "poses":
            prefix[..., 6].fill_(1.0)

    def initialize(self, video: Any) -> _CpuOffloadState:
        """Replace a fresh GPU DepthVideo with an unmaterialized CPU index-space.

        ``torch.empty_like`` reserves the shape/dtype without copying or touching
        the 15.5 GiB constructor buffer.  Prefix initialization is deliberately
        deferred to ``activate`` and is charged to that BA operation's transfer
        timing, not hidden in create-session latency.
        """
        import torch

        tensors: dict[str, Any] = {}
        for name in _DEPTH_VIDEO_TENSOR_FIELDS:
            value = getattr(video, name, None)
            if self._is_torch_tensor(value):
                cpu = torch.empty_like(value.detach(), device="cpu")
                tensors[name] = cpu
                setattr(video, name, cpu)
        # Capacity is the keyframe slot count from the multi-dimensional data
        # tensors (images, fmaps, disps, poses, ...), not a global minimum that
        # can be capped by an auxiliary small tensor (e.g. an index/mask with
        # first dim 8, which previously truncated every session at 8 slots).
        data_dims = [
            int(tensor.shape[0])
            for tensor in tensors.values()
            if getattr(tensor, "ndim", 0) >= 2 and int(tensor.shape[0]) > 0
        ]
        capacity = min(data_dims) if data_dims else 0
        return _CpuOffloadState(
            cpu_tensors=tensors,
            capacity=capacity,
            cpu_capacity_bytes=sum(self._tensor_bytes(tensor) for tensor in tensors.values()),
        )

    def _materialize_prefix(self, offload: _CpuOffloadState, slots: int) -> None:
        """Fault and initialize only never-used slots before native DROID reads them."""
        if slots <= offload.materialized_slots:
            return
        start = offload.materialized_slots
        for name, cpu in offload.cpu_tensors.items():
            self._default_prefix(name, cpu, start, slots)
        offload.materialized_slots = slots
        offload.cpu_resident_bytes = sum(
            self._tensor_bytes(tensor[:slots]) for tensor in offload.cpu_tensors.values()
        )

    @staticmethod
    def _pinned_prefix(cpu: Any) -> Any:
        """Return a short-lived pinned staging copy when the runtime supports it."""
        try:
            return cpu.pin_memory()
        except (RuntimeError, AttributeError):
            return cpu

    def activate(self, offload: _CpuOffloadState, video: Any, required_slots: int) -> None:
        """Materialize, pin/stage, and transfer the active prefix with honest timing."""
        if not offload.cpu_tensors:
            offload.last_transfer_ms = 0.0
            return
        import torch

        slots = min(offload.capacity, max(1, required_slots))
        started = time.monotonic()
        self._materialize_prefix(offload, slots)
        # pin_memory() copies just this prefix. Its allocation and copy belong to
        # the transfer span, ensuring the first BA reports deferred materialization.
        staging = {
            name: self._pinned_prefix(cpu[:slots])
            for name, cpu in offload.cpu_tensors.items()
        }
        for name, pinned in staging.items():
            setattr(video, name, pinned.to(self._device, non_blocking=bool(pinned.is_pinned())))
        # Completion is explicit because transfer timing is evidence, not a host
        # enqueue span. The following DROID kernels use the same stream either way.
        if str(self._device).startswith("cuda"):
            torch.cuda.synchronize(device=self._device)
        offload.active_slots = slots
        offload.gpu_resident_bytes = sum(self._tensor_bytes(getattr(video, name)) for name in offload.cpu_tensors)
        offload.gpu_peak_resident_bytes = max(offload.gpu_peak_resident_bytes, offload.gpu_resident_bytes)
        offload.last_transfer_ms = (time.monotonic() - started) * 1_000.0

    def release(self, offload: _CpuOffloadState, video: Any) -> None:
        """Synchronize D2H state copy, then restore CPU tensors and free GPU refs."""
        if not offload.cpu_tensors:
            return
        import torch

        started = time.monotonic()
        slots = offload.active_slots
        for name, cpu in offload.cpu_tensors.items():
            gpu = getattr(video, name)
            cpu[:slots].copy_(gpu, non_blocking=False)
        if str(self._device).startswith("cuda"):
            torch.cuda.synchronize(device=self._device)
        for name, cpu in offload.cpu_tensors.items():
            setattr(video, name, cpu)
        offload.gpu_resident_bytes = 0
        offload.active_slots = 0
        offload.last_transfer_ms += (time.monotonic() - started) * 1_000.0


CpuOffloadControllerFactory = Callable[[str], _DepthVideoCpuOffloader]


@dataclass
class _SessionState:
    """Isolated DROID sequence state for one explicit session."""

    session_id: str
    camera: DroidCamera
    image_shape: DroidImageShape
    options: DroidSessionOptions
    ownership: Ownership
    created_monotonic_s: float
    # Backend-owned per-session objects (DepthVideo, MotionFilter-state, Frontend,
    # Backend, TrajFiller). These are opaque to the contract layer; the adapter
    # populates them from the injected backend factory / DROID construction.
    video: Any = None
    motion_state: Any = None  # holds previous-keyframe fmap/net/inp + counter
    frontend: Any = None
    backend: Any = None
    filler: Any = None
    # CPU canonical DepthVideo storage; None preserves the established resident
    # GPU path. The controller only materializes this session's active prefix.
    cpu_offload_state: _CpuOffloadState | None = None
    # Source-timeline bookkeeping: maps DROID internal index -> (frame_id, ts).
    keyframe_source: list[tuple[str, float]] = field(default_factory=list)
    dense_source: list[tuple[str, float]] = field(default_factory=list)
    # Exact BGR CHW uint8 source frames retained on CPU for PoseTrajectoryFiller.
    # Non-keyframes cannot be replaced by a keyframe image: the filler estimates
    # each dense pose from that frame's own features.
    dense_images_bgr_chw_uint8: list[Any] = field(default_factory=list)
    # Exact CPU RGB/mask inputs for a fresh low-threshold keyframe-selection retry.
    # This stays independent of the GPU-resident mask grid and works with offload.
    replay_frames: list[tuple[Any, Any | None]] = field(default_factory=list)
    # Session lifecycle. A quarantined/finalized state retains only its lightweight
    # tombstone; mutable CUDA and retained-frame state is released explicitly.
    lifecycle: _SessionLifecycle = _SessionLifecycle.OPEN
    mutation_started: bool = False
    frames_pushed: int = 0
    # Service-owned capacity contracts. ``options.buffer`` is the initial
    # capacity; these values grow atomically with the underlying slot tensors.
    effective_capacity: int = 0
    video_capacity: int = 0
    journal_capacity: int = 0
    # Per-session ready-frame queue bound (bounded memory / backpressure).
    pending_frames: list["_PreparedFrame"] = field(default_factory=list)
    in_flight: bool = False
    continuation_phase_timings: _ContinuationPhaseTimings = field(default_factory=_ContinuationPhaseTimings)
    # Ordered only for deterministic bounded inspection; live entries are never
    # evicted because accepted frame count is itself bounded by options.buffer.
    result_journal: OrderedDict[str, _ResultJournalEntry] = field(default_factory=OrderedDict)
    frame_request_owners: dict[str, str] = field(default_factory=dict)
    finalize_request_id: str | None = None
    # Updated immediately before a successful create/push response is returned.
    # A missing value means no successful lifecycle response has been sent yet.
    last_response_sent_monotonic_s: float | None = None


@dataclass(frozen=True)
class _PreparedFrame:
    """A decoded, admission-validated frame ready for the batched fnet forward."""

    request: DroidFrameRequest
    # Exact caller RGB HWC uint8 array retained on CPU for dense trajectory fill.
    rgb_hwc_uint8: Any
    # ImageNet-normalized [1,1,C,H,W] RGB tensor on the backend device.
    normalized_input: Any
    # Resolved static-confidence mask at the 1/8 grid (or None).
    mask_grid: Any
    # Exact CPU mask payload before grid conversion, retained for retry replay.
    static_confidence_mask: Any | None
    preprocessing_h2d_started_monotonic_s: float
    preprocessing_h2d_completed_monotonic_s: float
    admitted_monotonic_s: float


@dataclass(frozen=True)
class _ContinuationResult:
    local_forward_count: int = 0
    keyframe_added: bool = False
    error: Exception | None = None
    mutation_started: bool = False


@dataclass(frozen=True)
class _GroupExecution:
    fnet_started_monotonic_s: float
    fnet_completed_monotonic_s: float
    results: tuple[_ContinuationResult, ...]
    fnet_error: Exception | None = None


# Session-local continuation after the fused fnet forward. Returns
# (local_forward_count, keyframe_added). Injectable so CPU tests verify fnet-batching
# honesty without lietorch/droid_slam; the replica uses the real DROID continuation.
ContinuationFn = Callable[["DroidAdapter", _SessionState, _PreparedFrame, Any], tuple[int, bool]]


# --------------------------------------------------------------------------- #
# Real backend loader. Imports droid_slam only inside the factory so the module
# imports without DROID/torch installed.
# --------------------------------------------------------------------------- #


def _verified_droid_source(config: DroidModelConfig):
    """Verify the release before any DROID import; environment paths are only inputs."""
    if config.droid_source_release_path is None:
        return None
    from ego_annotation.serving.droid_source import verify_droid_source_release

    return verify_droid_source_release(
        config.droid_source_release_path,
        expected_digest=config.droid_source_digest,
        expected_amendment_id=config.droid_source_amendment_id,
    )


def _require_verified_source_import_policy(release: object | None) -> None:
    """A release never permits import-cache bytecode to replace manifested source."""
    if release is not None and not sys.dont_write_bytecode:
        raise RuntimeError("verified DROID source imports require PYTHONDONTWRITEBYTECODE")


def _verified_droid_import_root(config: DroidModelConfig):
    release = _verified_droid_source(config)
    if release is not None:
        _require_verified_source_import_policy(release)
        return release.source_root
    # Backward-compatible production path: experiments cannot arrive here because
    # DroidModelConfig requires source identity whenever experiment_id is set.
    import os
    return os.environ.get(
        "EGO_DROID_REPO",
        "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/DROID-SLAM/droid_slam",
    )


def _verify_loaded_droid_net(config: DroidModelConfig):
    """Bind the imported module to the verified source bytes, not PYTHONPATH order."""
    import sys
    from ego_annotation.serving.droid_source import verify_imported_droid_module

    release = _verified_droid_source(config)
    module = sys.modules.get("droid_net")
    module_file = getattr(module, "__file__", None)
    if release is None or not isinstance(module_file, str):
        raise RuntimeError("DROID source release did not import a file-backed droid_net module")
    return release, verify_imported_droid_module(module_file, release, module=module)


class _TorchDroidBackend:
    """Wraps one resident DroidNet; its modules are shared across all sessions."""

    def __init__(self, net: Any, device: str) -> None:
        self._net = net
        self._device = device
        # ImageNet normalization constants (match MotionFilter / DroidNet).
        import torch

        self._torch = torch
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    def fnet(self, images: Any) -> Any:
        # images: [B,C,H,W] already normalized; fnet returns [B,128,gh,gw].
        with self._torch.inference_mode():
            return self._net.fnet(images)

    def cnet(self, image: Any) -> tuple[Any, Any]:
        with self._torch.inference_mode():
            net = self._net.cnet(image)
        net, inp = net.split([128, 128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @property
    def update_op(self) -> Any:
        return self._net.update


def _share_droid_weights_ipc(net: Any) -> dict[str, Any]:
    """Create CUDA IPC handles for all DroidNet parameters and buffers.

    Returns a picklable dict of metadata tuples that other processes can use to
    reconstruct the same GPU tensors without re-allocating VRAM. The caller
    process must stay alive for the shared memory to remain valid.
    """
    handles: dict[str, Any] = {}
    for name, tensor in list(net.named_parameters()) + list(net.named_buffers()):
        tensor.share_memory_()
        # PyTorch 1.13: tensor.storage() returns torch.Storage; PyTorch 2.0+:
        # tensor.untyped_storage(). Use whichever is available.
        storage = getattr(tensor, "untyped_storage", None)
        if storage is None:
            storage = tensor.storage()
        handles[name] = {
            "storage_handle": storage._share_cuda_(),
            "dtype": tensor.dtype,
            "device": tensor.device,
            "storage_offset": tensor.storage_offset(),
            "size": tuple(tensor.size()),
            "stride": tuple(tensor.stride()),
        }
    return handles


def _reconstruct_droid_weights_ipc(net: Any, handles: dict[str, Any]) -> None:
    """Point a fresh DroidNet's parameters/buffers at IPC-shared GPU memory.

    The net's parameters are replaced by views onto the host process's shared
    CUDA allocations, avoiding a second VRAM copy.
    """
    import torch

    for name, tensor in list(net.named_parameters()) + list(net.named_buffers()):
        info = handles.get(name)
        if info is None:
            continue
        # PyTorch 1.13: torch.UntypedStorage._new_shared_cuda; 2.0+: same API.
        # Fallback to torch.Storage._new_shared_cuda for older builds.
        new_shared = getattr(torch.UntypedStorage, "_new_shared_cuda", None)
        if new_shared is None:
            new_shared = torch.Storage._new_shared_cuda
        storage = new_shared(*info["storage_handle"])
        tensor.data = torch.empty(0, dtype=info["dtype"], device=info["device"]).set_(
            storage, info["storage_offset"], info["size"], info["stride"],
        )


def _load_droid_backend(config: DroidModelConfig) -> DroidBackend:
    """Load the real upstream DroidNet once inside the assigned Serve replica.

    The working DROID ABI is ``ray_serve_hawor`` (Python 3.10, Torch 1.13.0+cu117,
    Ray 2.55.1, lietorch + droid_backends). Weights are server-owned.
    """
    import os
    import sys

    import torch

    droid_repo = _verified_droid_import_root(config)
    if droid_repo and str(droid_repo) not in sys.path:
        sys.path.insert(0, str(droid_repo))
    from droid_net import DroidNet  # type: ignore[import-not-found]
    if config.droid_source_release_path is not None:
        _verify_loaded_droid_net(config)
    from collections import OrderedDict

    # CUDA IPC shared-weight path: if a handle file is provided, either create
    # handles (first replica) or reconstruct from them (subsequent replicas).
    ipc_handle_path = os.environ.get("EGO_DROID_IPC_HANDLE_FILE")
    if ipc_handle_path:
        net = DroidNet()
        if os.path.exists(ipc_handle_path):
            import pickle
            with open(ipc_handle_path, "rb") as fh:
                handles = pickle.load(fh)
            _reconstruct_droid_weights_ipc(net, handles)
            net.to(config.device).eval()
            return _TorchDroidBackend(net, config.device)
        # First replica: load from checkpoint, create IPC handles for others.
        weights = os.environ.get("EGO_DROID_WEIGHTS", config.weights)
        state_dict = OrderedDict(
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()
        )
        for key in ("update.weight.2.weight", "update.weight.2.bias",
                    "update.delta.2.weight", "update.delta.2.bias"):
            state_dict[key] = state_dict[key][:2]
        net.load_state_dict(state_dict)
        net.to(config.device).eval()
        handles = _share_droid_weights_ipc(net)
        os.makedirs(os.path.dirname(ipc_handle_path), exist_ok=True)
        import pickle
        with open(ipc_handle_path, "wb") as fh:
            pickle.dump(handles, fh)
        return _TorchDroidBackend(net, config.device)

    weights = os.environ.get("EGO_DROID_WEIGHTS", config.weights)
    net = DroidNet()
    state_dict = OrderedDict(
        (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()
    )
    # Match Droid.load_weights: the final GRU layers are sliced to 2 outputs.
    for key in ("update.weight.2.weight", "update.weight.2.bias",
                "update.delta.2.weight", "update.delta.2.bias"):
        state_dict[key] = state_dict[key][:2]
    net.load_state_dict(state_dict)
    net.to(config.device).eval()
    return _TorchDroidBackend(net, config.device)


def _build_droid_session_objects(
    backend: DroidBackend,
    config: DroidModelConfig,
    camera: DroidCamera,
    image_shape: DroidImageShape,
    options: DroidSessionOptions,
) -> dict[str, Any]:
    """Construct the isolated per-session DepthVideo + motion/frontend/backend/filler.

    This mirrors ``Droid.__init__`` but reuses the shared backend modules instead
    of constructing a new DroidNet, and runs single-process (no spawn, no
    visualizer Process).
    """
    import os
    import sys

    droid_repo = _verified_droid_import_root(config)
    if droid_repo and str(droid_repo) not in sys.path:
        sys.path.insert(0, str(droid_repo))
    from depth_video import DepthVideo  # type: ignore[import-not-found]
    from motion_filter import MotionFilter  # type: ignore[import-not-found]
    from droid_frontend import DroidFrontend  # type: ignore[import-not-found]
    from droid_backend import DroidBackend as _DroidBackend  # type: ignore[import-not-found]
    from trajectory_filler import PoseTrajectoryFiller  # type: ignore[import-not-found]
    from argparse import Namespace

    image_size = [image_shape.height, image_shape.width]
    # A shared namespace carries the validated session options into the frontend/
    # backend constructors exactly as Droid expects.
    args = Namespace(
        weights=config.weights,
        buffer=options.buffer,
        image_size=image_size,
        disable_vis=True,
        beta=options.beta,
        filter_thresh=options.filter_thresh,
        warmup=options.warmup,
        keyframe_thresh=options.keyframe_thresh,
        frontend_thresh=options.frontend_thresh,
        frontend_window=options.frontend_window,
        frontend_radius=options.frontend_radius,
        frontend_nms=options.frontend_nms,
        backend_thresh=options.backend_thresh,
        backend_radius=options.backend_radius,
        backend_nms=options.backend_nms,
        upsample=options.upsample,
        asynchronous=False,
        frontend_device=config.device,
        backend_device=config.device,
        stereo=options.stereo,
    )
    video = DepthVideo(image_size, options.buffer, stereo=options.stereo, device=config.device)

    # MotionFilter needs the net modules; build it but we drive its internals
    # through our batched fnet path. We still construct it so session-local
    # correlation/update use the identical normalization constants and logic.
    filterx = MotionFilter.__new__(MotionFilter)
    filterx.cnet = backend.cnet
    filterx.fnet = backend.fnet
    filterx.update = backend.update_op
    filterx.video = video
    filterx.thresh = options.filter_thresh
    filterx.device = config.device
    filterx.count = 0
    import torch

    filterx.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=config.device)[:, None, None]
    filterx.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=config.device)[:, None, None]
    # Per-session motion state carried across frames (previous keyframe context).
    filterx.net = None
    filterx.inp = None
    filterx.fmap = None

    frontend = DroidFrontend(_SharedNet(backend), video, args)
    droid_backend = _DroidBackend(_SharedNet(backend), video, args)
    filler = PoseTrajectoryFiller(_SharedNet(backend), video, device=config.device)
    return {
        "video": video,
        "filter": filterx,
        "frontend": frontend,
        "backend": droid_backend,
        "filler": filler,
    }


class _SharedNet:
    """Adapter exposing fnet/cnet/update as attributes, matching DroidFrontend/Backend expectations.

    DroidFrontend/Backend/__init__ do ``self.update_op = net.update`` and store the
    net; some paths call ``net.update``. This wraps the shared backend so every
    session references the same resident modules.
    """

    def __init__(self, backend: DroidBackend) -> None:
        self._backend = backend

    @property
    def fnet(self) -> Any:
        return self._backend.fnet

    @property
    def cnet(self) -> Any:
        return self._backend.cnet

    @property
    def update(self) -> Any:
        return self._backend.update_op


# --------------------------------------------------------------------------- #
# The adapter.
# --------------------------------------------------------------------------- #


def _run_frontend_without_grad(frontend: Any) -> None:
    """Run persistent frontend state updates without retaining autograd graphs."""
    import torch

    with torch.no_grad():
        frontend()


def _record_exact_dense_source_frame(state: _SessionState, item: _PreparedFrame) -> None:
    """Append the source-owned BGR frame consumed by dense trajectory filling."""
    state.dense_source.append((item.request.frame_id, item.request.source_timestamp_s))
    state.dense_images_bgr_chw_uint8.append(
        item.rgb_hwc_uint8[..., ::-1].transpose(2, 0, 1).copy()
    )


def _record_replay_frame(state: _SessionState, item: _PreparedFrame) -> None:
    """Retain the exact CPU inputs needed to retry keyframe selection internally."""
    import numpy as np

    mask = item.static_confidence_mask
    state.replay_frames.append((
        np.ascontiguousarray(item.rgb_hwc_uint8).copy(),
        None if mask is None else np.ascontiguousarray(mask).copy(),
    ))


def _droid_session_local_track(adapter: "DroidAdapter", state: _SessionState, item: _PreparedFrame, gmap_any: Any) -> tuple[int, bool]:
    """Session-local continuation after the fused fnet: correlation/update/append/frontend.

    Returns (count of session-local forwards performed, keyframe_added). Mirrors
    ``MotionFilter.track`` and ``Droid.track`` (frontend) faithfully. This is the
    real DROID continuation injected by default; CPU tests inject a fake.
    """
    import torch
    import lietorch
    import geom.projective_ops as pops  # type: ignore[import-not-found]
    from modules.corr import CorrBlock  # type: ignore[import-not-found]

    phases = _ContinuationPhaseTimings()
    state.continuation_phase_timings = phases
    filt = state.motion_state
    video = state.video
    gmap = gmap_any.squeeze(0)  # [128,gh,gw]
    mask = item.mask_grid
    local_fwds = 0
    keyframe_added = False
    ht = gmap.shape[-2]
    wd = gmap.shape[-1]
    if mask is None:
        mask = torch.zeros([ht, wd], device=gmap.device)

    Id = lietorch.SE3.Identity(1,).data.squeeze()
    intrinsics = torch.as_tensor(
        list(state.camera.intrinsics), device=video.poses.device, dtype=torch.float32
    ) / 8.0

    with torch.no_grad():
        if video.counter.value == 0:
            # First frame: always add. cnet is a session-local forward.
            cnet_started = time.monotonic()
            net, inp = adapter._backend.cnet(item.normalized_input[:, [0]])
            phases.add("cnet_s", time.monotonic() - cnet_started)
            local_fwds += 1
            filt.net, filt.inp, filt.fmap = net, inp, gmap
            video.append(
                item.request.source_timestamp_s,
                adapter._rgb_hwc_to_bgr_chw_uint8(item)[0],
                Id, 1.0, None, intrinsics, gmap, net[0, 0], inp[0, 0], mask,
            )
            state.keyframe_source.append((item.request.frame_id, item.request.source_timestamp_s))
            keyframe_added = True
        else:
            correlation_update_started = time.monotonic()
            coords0 = pops.coords_grid(ht, wd, device=gmap.device)[None, None]
            corr = CorrBlock(filt.fmap[None, [0]], gmap[None, [0]])(coords0)
            _, delta, weight = adapter._backend.update_op(filt.net[None], filt.inp[None], corr)
            phases.add("correlation_update_s", time.monotonic() - correlation_update_started)
            local_fwds += 1
            if delta.norm(dim=-1).mean().item() > filt.thresh:
                filt.count = 0
                cnet_started = time.monotonic()
                net, inp = adapter._backend.cnet(item.normalized_input[:, [0]])
                phases.add("cnet_s", time.monotonic() - cnet_started)
                local_fwds += 1
                filt.net, filt.inp, filt.fmap = net, inp, gmap
                video.append(
                    item.request.source_timestamp_s,
                    adapter._rgb_hwc_to_bgr_chw_uint8(item)[0],
                    None, None, None, intrinsics, gmap, net[0], inp[0], mask,
                )
                state.keyframe_source.append((item.request.frame_id, item.request.source_timestamp_s))
                keyframe_added = True
            else:
                filt.count += 1
    # Frontend runs session-local BA over the factor graph (synchronous, as in
    # this repo's droid.py). A CUDA exception is a hard request failure: hiding
    # it leaves the replica's CUDA context potentially poisoned and makes the
    # next unrelated request report a misleading cuDNN mapping error.
    frontend_before = state.frontend.is_initialized
    frontend_started = time.monotonic()
    _run_frontend_without_grad(state.frontend)
    phases.add("frontend_ba_s", time.monotonic() - frontend_started)
    if state.frontend.is_initialized and not frontend_before:
        local_fwds += 1
    elif state.frontend.is_initialized:
        local_fwds += 1
    # Preserve every admitted source frame independently of keyframe selection.
    # PoseTrajectoryFiller requires each non-keyframe's own image features.
    _record_exact_dense_source_frame(state, item)
    state.frames_pushed += 1
    return local_fwds, keyframe_added


class DroidAdapter:
    """Resident DroidNet owner used by a single Ray Serve GPU2 replica."""

    def __init__(
        self,
        config: DroidModelConfig,
        *,
        backend_factory: BackendFactory = _load_droid_backend,
        session_factory: SessionFactory = _build_droid_session_objects,
        tensor_resolver: TensorResolver = _default_tensor_resolver,
        continuation_fn: ContinuationFn | None = None,
        cpu_offload_controller_factory: CpuOffloadControllerFactory = _DepthVideoCpuOffloader,
        runtime_evidence_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config
        # Verify dependency bytes before the factory has any opportunity to import
        # droid_net or construct CUDA state. The loaded module is bound below.
        self._verified_source_release = _verified_droid_source(config)
        _require_verified_source_import_policy(self._verified_source_release)
        self._tensor_resolver = tensor_resolver
        self._backend = backend_factory(config)
        self._loaded_droid_net_path = None
        if self._verified_source_release is not None:
            _release, self._loaded_droid_net_path = _verify_loaded_droid_net(config)
        self._session_factory = session_factory
        self._continuation_fn = continuation_fn or _droid_session_local_track
        self._cpu_offloader = cpu_offload_controller_factory(config.device) if config.cpu_offload else None
        self._model_load_count = 1
        self._running_batches = 0
        self._batch_snapshot = BatchSnapshotTracker(
            service="droid", replica_id=config.replica_id,
            capacity={"kind": "resident_sessions", "max_sessions": config.max_sessions, "max_batch_size": config.max_fnet_batch_size, "max_ongoing_requests": 32},
        )
        self._sessions: dict[str, _SessionState] = {}
        self._terminal_lru: OrderedDict[str, None] = OrderedDict()
        self._admitted_pending = 0
        # State transitions may occur on the Serve actor and the serialized worker
        # thread. CUDA-facing work (including H2D admission and session allocation)
        # shares one lock, so no session-local or shared-backend operation overlaps.
        self._state_lock = threading.RLock()
        self._gpu_execution_lock = threading.Lock()
        # Concurrent BA is scoped to the GPU-resident path ONLY (cpu_offload=False).
        # The CPU-offload H2D transfer + tensor lifetime interaction under
        # concurrency caused the off8/off9 illegal-memory-access regression.
        # GPU-resident sessions hold fully-resident tensors, so up to
        # max_concurrent_ba independent sessions may overlap BA on separate CUDA
        # streams. 1 = serialized (production). Offload ignores the permit.
        self._ba_permit = threading.Semaphore(config.max_concurrent_ba)
        # This counts BA continuations after their permit is acquired.  It is
        # server-owned evidence of actual overlap, distinct from the configured
        # limit and from fnet's cross-session batch size.
        self._ba_overlap_lock = threading.Lock()
        self._running_ba_forwards = 0
        self._peak_simultaneous_forwards = 0
        self._server_runtime_identity: ServerIdentity | None = None
        if config.experiment_id is not None:
            from ego_annotation.serving.benchmark.release import derive_worker_runtime_evidence

            derive = runtime_evidence_factory or derive_worker_runtime_evidence
            evidence = derive(
                release_root=config.application_release_path,
                checkpoint_path=config.weights,
                imported_module_file=__file__,
            )
            if evidence.physical_gpu != config.assigned_gpu:
                raise RuntimeError(
                    f"DROID worker physical GPU {evidence.physical_gpu} differs from planned GPU {config.assigned_gpu}"
                )
            self._server_runtime_identity = ServerIdentity(
                experiment_id=config.experiment_id,
                replica_id=config.replica_id,
                assigned_gpu=evidence.physical_gpu,
                worker_pid=evidence.worker_pid,
                gcs_address=config.gcs_address,
                http_port=config.http_port,
                temp_dir=config.temp_dir,
                model_revision=config.model_revision,
                checkpoint_digest=evidence.checkpoint_digest,
                schema_version=SCHEMA_VERSION,
                release_sha=evidence.source_sha,
                release_digest=evidence.release_digest,
                cuda_uuid=evidence.cuda_uuid,
                module_root=str(evidence.module_root),
                dependency_digest=self._verified_source_release.source_digest if self._verified_source_release else None,
                dependency_root=str(self._verified_source_release.source_root) if self._verified_source_release else None,
                source_amendment_id=self._verified_source_release.amendment_id if self._verified_source_release else None,
            )

    @property
    def server_identity(self) -> ServerIdentity | None:
        return self._server_runtime_identity

    @property
    def config(self) -> DroidModelConfig:
        return self._config

    @property
    def resident_session_count(self) -> int:
        with self._state_lock:
            return self._active_session_count()

    def _active_session_count(self) -> int:
        return sum(state.lifecycle in {_SessionLifecycle.OPEN, _SessionLifecycle.FINALIZING} for state in self._sessions.values())

    # ---- create_session -------------------------------------------------- #

    def create_session(self, request: DroidCreateSessionRequest) -> DroidCreateSessionResponse:
        if request.model_revision != self._config.model_revision:
            return self._validation_error(request.ownership, "model_revision mismatch with resident DROID backend")
        if request.options.buffer > self._config.max_buffer_slots:
            return self._validation_error(
                request.ownership,
                "session initial buffer exceeds configured max_buffer_slots",
            )
        if request.options.buffer + 1 > self._config.max_result_journal_entries_per_session:
            return self._validation_error(
                request.ownership,
                "session initial buffer exceeds bounded idempotency journal capacity",
            )
        # Session factories allocate DepthVideo tensors and therefore share the
        # backend's CUDA context with pushes/finalizes. Keep capacity reservation and
        # allocation atomic with respect to all lifecycle transitions.
        with self._state_lock:
            if self._active_session_count() >= self._config.max_sessions:
                return DroidCreateSessionResponse(
                    ownership=request.ownership,
                    error=ServiceError(
                        ErrorCode.BACKPRESSURE,
                        f"DROID replica at max_sessions={self._config.max_sessions}; retry later",
                        retryable=True,
                        ownership=request.ownership,
                    ),
                )
            session_id = uuid4().hex
            try:
                with self._gpu_execution_lock:
                    objects = dict(self._session_factory(
                        self._backend, self._config, request.camera, request.image_shape, request.options,
                    ))
            except Exception as exc:
                return self._validation_error(request.ownership, f"failed to construct DROID session: {exc}")
            state = _SessionState(
                session_id=session_id,
                camera=request.camera,
                image_shape=request.image_shape,
                options=request.options,
                ownership=request.ownership,
                created_monotonic_s=time.monotonic(),
                video=objects.get("video"),
                motion_state=objects.get("filter"),
                frontend=objects.get("frontend"),
                backend=objects.get("backend"),
                filler=objects.get("filler"),
                effective_capacity=request.options.buffer,
                video_capacity=self._infer_video_capacity(objects.get("video")) or request.options.buffer,
                journal_capacity=request.options.buffer + 1,
            )
            if self._cpu_offloader is not None and state.video is not None:
                state.cpu_offload_state = self._cpu_offloader.initialize(state.video)
            self._sessions[session_id] = state
        return DroidCreateSessionResponse(ownership=request.ownership, session_id=session_id)

    # ---- push_frame admission ------------------------------------------- #

    def admit_frame(self, request: DroidFrameRequest) -> _PreparedFrame | DroidFrameResponse:
        """Admit new work or return the durable outcome for an idempotent retry.

        The stable request id and frame id jointly own one mutation. Reusing either
        identity with different request content is an explicit conflict. A retry
        while the shielded worker still owns CUDA receives retryable backpressure;
        after completion it receives the exact journaled typed response.
        """
        if request.model_revision and request.model_revision != self._config.model_revision:
            raise ContractValidationError("model_revision mismatch with resident DROID backend")
        key = self._push_journal_key(request)
        signature = self._push_signature(request)
        with self._state_lock:
            state = self._sessions.get(request.session_id)
            if state is None:
                raise ContractValidationError(f"unknown DROID session {request.session_id!r}")
            existing = state.result_journal.get(key)
            if existing is not None:
                self._touch_terminal(request.session_id)
                if existing.signature != signature:
                    return self._push_conflict(request, "request_id was reused with different push-frame content")
                if existing.response is not None:
                    assert isinstance(existing.response, DroidFrameResponse)
                    return existing.response
                return self._push_in_progress(request)
            owner = state.frame_request_owners.get(request.frame_id)
            if owner is not None and owner != request.ownership.request_id:
                return self._push_conflict(request, "frame_id is already owned by a different request_id")
            state = self._require_open_session(request.session_id)
            if request.rgb.shape != state.image_shape.shape_hwc:
                raise ContractValidationError(
                    f"frame RGB shape {request.rgb.shape} does not match session image grid "
                    f"{state.image_shape.shape_hwc}"
                )
            if state.in_flight:
                raise ContractValidationError(
                    f"session {request.session_id} already has a ready frame in flight; "
                    f"at most one ready frame per session may enter a cross-session batch"
                )
            # Reserve one predictive slot beyond the current keyframe counter:
            # frontend.append() seeds counter+1 and the next frame may itself add
            # a keyframe. This runs before the frame is accepted and while no
            # continuation/backend/filler is active.
            required_slots = int(state.video.counter.value) + 2
            self._ensure_video_capacity(state, required_slots)
            # The result journal records one entry per PUSHED frame, but the buffer
            # growth above is keyed on keyframe count (counter). The motion filter
            # rejects ~60% of frames as non-keyframes, so journal entries grow
            # ~2.5x faster than counter. Ensure journal_capacity covers pushed frames.
            needed_journal = state.frames_pushed + len(state.pending_frames) + 2
            if needed_journal > state.journal_capacity:
                with self._gpu_execution_lock:
                    target = max(state.journal_capacity, 1)
                    while target < needed_journal:
                        target = min(self._config.max_result_journal_entries_per_session, max(target * 2, needed_journal))
                    if target <= self._config.max_result_journal_entries_per_session:
                        state.journal_capacity = target
            if len(state.result_journal) >= state.journal_capacity:
                logger.error(
                    "DROID journal full: session=%s len(journal)=%d journal_capacity=%d "
                    "frames_pushed=%d counter=%d pending=%d effective_capacity=%d",
                    request.session_id, len(state.result_journal), state.journal_capacity,
                    state.frames_pushed, int(state.video.counter.value),
                    len(state.pending_frames), state.effective_capacity,
                )
                raise ContractValidationError("bounded DROID result journal is full")
            state.result_journal[key] = _ResultJournalEntry(signature=signature)
            state.frame_request_owners[request.frame_id] = request.ownership.request_id
            # Reserve session ownership before H2D so finalize cannot release GPU
            # state while preprocessing dereferences it.
            state.in_flight = True
        try:
            # Decode/normalization can issue H2D work. It shares the actor-level GPU
            # lock with fnet, continuation, finalize, and session construction.
            preprocessing_h2d_started = time.monotonic()
            with self._gpu_execution_lock:
                rgb = self._decode_rgb(request)
                static_confidence_mask = self._decode_static_confidence_mask(request, state)
                mask_grid = self._static_confidence_mask_to_grid(static_confidence_mask, state)
                normalized = self._normalize_input(rgb, state)
            preprocessing_h2d_completed = time.monotonic()
            prepared = _PreparedFrame(
                request=request,
                rgb_hwc_uint8=rgb.copy(),
                normalized_input=normalized,
                mask_grid=mask_grid,
                static_confidence_mask=static_confidence_mask,
                preprocessing_h2d_started_monotonic_s=preprocessing_h2d_started,
                preprocessing_h2d_completed_monotonic_s=preprocessing_h2d_completed,
                admitted_monotonic_s=time.monotonic(),
            )
            with self._state_lock:
                state = self._require_open_session(request.session_id)
                if not state.in_flight:
                    raise RuntimeError("DROID push lost its admission reservation during preprocessing")
                state.pending_frames.append(prepared)
                self._admitted_pending += 1
            return prepared
        except BaseException:
            with self._state_lock:
                self._forget_push_reservation(state, request)
            raise

    def request_dispatched(self, prepared: _PreparedFrame) -> None:
        """Record Serve-batch dispatch exactly once; retains ownership until worker completion."""
        with self._state_lock:
            state = self._sessions.get(prepared.request.session_id)
            if state is not None:
                for index, pending in enumerate(state.pending_frames):
                    if pending is prepared:
                        state.pending_frames.pop(index)
                        self._admitted_pending -= 1
                        break

    def request_abandoned(self, prepared: _PreparedFrame) -> None:
        """Undo admission and its reservation only before batch dispatch."""
        with self._state_lock:
            state = self._sessions.get(prepared.request.session_id)
            if state is not None:
                for index, pending in enumerate(state.pending_frames):
                    if pending is prepared:
                        state.pending_frames.pop(index)
                        state.in_flight = False
                        self._admitted_pending -= 1
                        self._forget_push_reservation(state, prepared.request)
                        break

    def request_completed(self, session_id: str) -> None:
        """Release a session only after its serialized local continuation has stopped."""
        with self._state_lock:
            state = self._sessions.get(session_id)
            if state is not None:
                state.in_flight = False

    def response_sent(self, session_id: str, *, now_monotonic: float | None = None) -> bool:
        """Record a successful create/push response for idle-session ownership.

        The deployment invokes this immediately before returning the typed response.
        Only an OPEN session is eligible: terminal outcomes retain their journals but
        must never be revived by a late response callback.
        """
        sent = time.monotonic() if now_monotonic is None else now_monotonic
        with self._state_lock:
            state = self._sessions.get(session_id)
            if state is None or state.lifecycle is not _SessionLifecycle.OPEN:
                return False
            state.last_response_sent_monotonic_s = sent
            return True

    def reap_idle_sessions(self, *, now_monotonic: float | None = None) -> int:
        """Quarantine OPEN sessions abandoned after a successful response.

        GPU work is serialized by ``_gpu_execution_lock``, while several admission
        paths legitimately acquire ``_state_lock`` before that lock. The reaper
        therefore never waits for the state lock while holding the GPU lock: if a
        state transition is active it yields and retries on its next scan. Once it
        owns both locks it rechecks the lifecycle, in-flight reservation, and idle
        deadline before releasing DepthVideo and all per-session references.
        """
        now = time.monotonic() if now_monotonic is None else now_monotonic
        with self._state_lock:
            candidates = tuple(
                state.session_id
                for state in self._sessions.values()
                if state.lifecycle is _SessionLifecycle.OPEN
                and not state.in_flight
                and state.last_response_sent_monotonic_s is not None
                and now - state.last_response_sent_monotonic_s > self._config.idle_session_ttl_s
            )
        reaped = 0
        for session_id in candidates:
            self._gpu_execution_lock.acquire()
            state_lock_acquired = self._state_lock.acquire(blocking=False)
            if not state_lock_acquired:
                self._gpu_execution_lock.release()
                continue
            try:
                state = self._sessions.get(session_id)
                if (
                    state is not None
                    and state.lifecycle is _SessionLifecycle.OPEN
                    and not state.in_flight
                    and state.last_response_sent_monotonic_s is not None
                    and now - state.last_response_sent_monotonic_s > self._config.idle_session_ttl_s
                ):
                    self._mark_terminal(state, _SessionLifecycle.QUARANTINED)
                    reaped += 1
            finally:
                self._state_lock.release()
                self._gpu_execution_lock.release()
        return reaped

    # ---- single-push request-local fnet + ordered continuation ------------ #

    async def infer_frames(self, frames: Sequence[DroidFrameRequest]) -> list[DroidFrameResponse]:
        """Process a complete, single-session infer request without a batch window.

        Admission deliberately reuses ``admit_frame`` for every frame so RGB,
        optional mask/depth payload handling, normalization, per-frame ownership,
        and all session options are identical to legacy ``push_frame``.  Fnet is
        the only request-wide operation.  Its rows are copied into owning feature
        tensors before any later fnet forward or stateful continuation executes.
        """
        if not frames:
            return []
        if len(frames) > 256:
            raise ContractValidationError("droid.infer accepts at most 256 frames")
        prepared = self._admit_infer_frames(frames)
        group_size = self._config.max_fnet_batch_size
        group_count = (len(prepared) + group_size - 1) // group_size
        self._running_batches += group_count
        worker = asyncio.create_task(asyncio.to_thread(self._infer_fnet_then_continue, prepared, group_size))
        try:
            groups = await asyncio.shield(worker)
        except asyncio.CancelledError:
            worker.add_done_callback(lambda task: self._complete_cancelled_infer_groups(task, prepared, group_count))
            raise
        except Exception as exc:
            groups = [(uuid4().hex, tuple(prepared), _GroupExecution(
                time.monotonic(), time.monotonic(), (), exc,
            ))]
        return self._complete_infer_groups(groups, prepared, group_count)

    def _admit_infer_frames(self, frames: Sequence[DroidFrameRequest]) -> list[_PreparedFrame]:
        """Use legacy per-frame admission while holding one session reservation."""
        session_ids = {frame.session_id for frame in frames}
        if len(session_ids) != 1:
            raise ContractValidationError("droid.infer frames must belong to exactly one session")
        frame_ids = [frame.frame_id for frame in frames]
        if len(set(frame_ids)) != len(frame_ids):
            raise ContractValidationError("droid.infer frame_id values must be unique")
        prepared: list[_PreparedFrame] = []
        try:
            for frame in frames:
                item = self.admit_frame(frame)
                if isinstance(item, DroidFrameResponse):
                    raise ContractValidationError(item.error.message if item.error else "DROID infer admission failed")
                prepared.append(item)
                self.request_dispatched(item)
                # ``admit_frame`` enforces legacy one-ready-frame ownership.  The
                # request has all frames locally, so release that temporary guard
                # only between admissions, then reserve the session as one unit.
                with self._state_lock:
                    self._require_open_session(frame.session_id).in_flight = False
            with self._state_lock:
                self._require_open_session(frames[0].session_id).in_flight = True
            return prepared
        except BaseException:
            with self._state_lock:
                state = self._sessions.get(frames[0].session_id)
                if state is not None:
                    for item in prepared:
                        self._forget_push_reservation(state, item.request)
                    state.in_flight = False
            raise

    @staticmethod
    def _own_fnet_row(fmaps: Any, index: int) -> Any:
        """Return an owning, non-autograd feature tensor for one frame.

        ``fmaps[index:index+1]`` is a view.  A request-local cache must survive
        future forwards and must not expose another frame's storage to a mutable
        continuation, so detach before clone for torch and copy for CPU fakes.
        """
        row = fmaps[index:index + 1]
        detach = getattr(row, "detach", None)
        if callable(detach):
            row = detach()
        clone = getattr(row, "clone", None)
        if callable(clone):
            return clone()
        copy = getattr(row, "copy", None)
        if callable(copy):
            return copy()
        raise TypeError("DROID fnet result rows must support clone() or copy()")

    def _infer_fnet_then_continue(
        self, items: Sequence[_PreparedFrame], group_size: int,
    ) -> list[tuple[str, tuple[_PreparedFrame, ...], _GroupExecution]]:
        """Cache every fnet row, then perform the unchanged continuation in order."""
        import torch

        groups: list[tuple[str, tuple[_PreparedFrame, ...], float, float]] = []
        cached_features: list[Any] = []
        with self._gpu_execution_lock:
            try:
                for start in range(0, len(items), group_size):
                    group_items = tuple(items[start:start + group_size])
                    batch_id = uuid4().hex
                    fnet_started = time.monotonic()
                    try:
                        # Each legacy input is [1,1,3,H,W]; remove only its outer
                        # singleton then stack into BasicEncoder's valid B axis.
                        stacked = torch.stack([item.normalized_input[0] for item in group_items], dim=0)
                        fmaps = self._backend.fnet(stacked)
                        if str(self._config.device).startswith("cuda"):
                            torch.cuda.synchronize(device=self._config.device)
                        cached_features.extend(
                            self._own_fnet_row(fmaps, index) for index in range(len(group_items))
                        )
                    except BaseException as exc:
                        return [(batch_id, group_items, _GroupExecution(
                            fnet_started, time.monotonic(), (), exc,
                        ))]
                    groups.append((batch_id, group_items, fnet_started, time.monotonic()))

                results: list[_ContinuationResult] = []
                for index, item in enumerate(items):
                    state = self._sessions[item.request.session_id]
                    state.mutation_started = True
                    try:
                        self._ensure_video_capacity_locked(state, int(state.video.counter.value) + 2)
                        with self._active_video_window(state, int(state.video.counter.value) + 2):
                            local_forwards, keyframe_added = self._continuation_fn(
                                self, state, item, cached_features[index],
                            )
                            if str(self._config.device).startswith("cuda"):
                                torch.cuda.synchronize(device=self._config.device)
                        _record_replay_frame(state, item)
                    except Exception as exc:
                        results.append(_ContinuationResult(error=exc, mutation_started=True))
                        results.extend(
                            _ContinuationResult(
                                error=RuntimeError("prior infer frame continuation failed"), mutation_started=False,
                            )
                            for _ in items[index + 1:]
                        )
                        break
                    else:
                        results.append(_ContinuationResult(
                            local_forwards, keyframe_added, mutation_started=True,
                        ))
            except Exception as exc:
                return [(uuid4().hex, tuple(items), _GroupExecution(
                    time.monotonic(), time.monotonic(), (), exc,
                ))]

        completed: list[tuple[str, tuple[_PreparedFrame, ...], _GroupExecution]] = []
        offset = 0
        for batch_id, group_items, fnet_started, fnet_completed in groups:
            count = len(group_items)
            completed.append((batch_id, group_items, _GroupExecution(
                fnet_started, fnet_completed, tuple(results[offset:offset + count]),
            )))
            offset += count
        return completed

    def _complete_infer_groups(
        self,
        groups: Sequence[tuple[str, tuple[_PreparedFrame, ...], _GroupExecution]],
        prepared: Sequence[_PreparedFrame],
        expected_group_count: int,
    ) -> list[DroidFrameResponse]:
        responses: list[DroidFrameResponse] = []
        completed_items = 0
        for batch_id, group_items, execution in groups:
            completed_items += len(group_items)
            dispatched = min((item.admitted_monotonic_s for item in group_items), default=time.monotonic())
            responses.extend(self._complete_group(batch_id, group_items, dispatched, dispatched, execution))
        self._running_batches -= expected_group_count - len(groups)
        if completed_items < len(prepared):
            with self._state_lock:
                state = self._sessions.get(prepared[0].request.session_id)
                if state is not None:
                    for item in prepared[completed_items:]:
                        self._forget_push_reservation(state, item.request)
        return responses

    def _complete_cancelled_infer_groups(
        self,
        task: "asyncio.Task[list[tuple[str, tuple[_PreparedFrame, ...], _GroupExecution]]]",
        prepared: Sequence[_PreparedFrame],
        expected_group_count: int,
    ) -> None:
        try:
            groups = task.result()
        except Exception:
            self._running_batches -= expected_group_count
            return
        self._complete_infer_groups(groups, prepared, expected_group_count)

    # ---- the batched fnet forward + session-local continuation ---------- #

    async def push_frame_batch(self, prepared: Sequence[_PreparedFrame]) -> list[DroidFrameResponse]:
        """One fused ``fnet`` forward across compatible ready frames, then session-local work.

        The batch contains at most one ready frame per session (enforced at
        admission). Compatible items share one ``fnet`` forward; incompatible
        shapes are split into separate fnet forwards, each honestly traced. After
        the fused fnet, each item runs its session-local correlation/update/
        cnet-if-keyframe/frontend continuation. ``fnet_forward_count`` records the
        number of fused forwards (>=1); ``session_local_forward_count`` records the
        per-item local work.
        """
        if not prepared:
            return []
        # Group by compatibility key (image_shape). Every physical fnet forward
        # receives its own batch_id; a callback that splits by shape must not claim
        # that multiple forwards were one batch.
        responses: list[DroidFrameResponse | None] = [None] * len(prepared)
        groups: dict[tuple[int, int], list[int]] = {}
        with self._state_lock:
            for idx, item in enumerate(prepared):
                shape = _session_image_shape(self._sessions, item.request.session_id)
                groups.setdefault(shape, []).append(idx)
        for indices in groups.values():
            group_items = [prepared[i] for i in indices]
            group_responses = await self._run_fnet_group(uuid4().hex, group_items)
            for i, response in zip(indices, group_responses):
                responses[i] = response
        return [r if r is not None else self._unexpected_error(prepared[i].request) for i, r in enumerate(responses)]

    async def _run_fnet_group(self, batch_id: str, items: Sequence[_PreparedFrame]) -> list[DroidFrameResponse]:
        dispatched = time.monotonic()
        admitted = min((it.admitted_monotonic_s for it in items), default=dispatched)
        for item in items:
            self.request_dispatched(item)
        self._running_batches += 1
        worker = asyncio.create_task(asyncio.to_thread(self._fnet_and_continue, items))
        try:
            execution = await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Shield prevents cancellation from interrupting CUDA work. The callback
            # joins the worker's result and performs idempotent cleanup only after
            # its continuation has stopped mutating session state.
            worker.add_done_callback(
                lambda task: self._complete_cancelled_group(task, batch_id, items, admitted, dispatched)
            )
            raise
        except Exception as exc:
            execution = _GroupExecution(time.monotonic(), time.monotonic(), (), exc)
        return self._complete_group(batch_id, items, admitted, dispatched, execution)

    def _complete_cancelled_group(
        self,
        task: "asyncio.Task[_GroupExecution]",
        batch_id: str,
        items: Sequence[_PreparedFrame],
        admitted: float,
        dispatched: float,
    ) -> None:
        try:
            execution = task.result()
        except Exception as exc:
            execution = _GroupExecution(time.monotonic(), time.monotonic(), (), exc)
        self._complete_group(batch_id, items, admitted, dispatched, execution)

    def _complete_group(
        self,
        batch_id: str,
        items: Sequence[_PreparedFrame],
        admitted: float,
        dispatched: float,
        execution: _GroupExecution,
    ) -> list[DroidFrameResponse]:
        self._running_batches -= 1
        completed = time.monotonic()
        if execution.fnet_error is not None:
            failed: list[DroidFrameResponse] = []
            for item in items:
                response = DroidFrameResponse(
                    ownership=item.request.ownership,
                    error=ServiceError(
                        ErrorCode.MODEL_FAILURE, str(execution.fnet_error), retryable=False,
                        ownership=item.request.ownership, batch_id=batch_id,
                    ),
                )
                with self._state_lock:
                    state = self._sessions.get(item.request.session_id)
                    if state is not None:
                        self._record_push_result(state, item.request, response)
                self.request_completed(item.request.session_id)
                failed.append(response)
            return failed

        responses: list[DroidFrameResponse] = []
        for index, item in enumerate(items):
            result = execution.results[index]
            with self._state_lock:
                state = self._sessions.get(item.request.session_id)
                if state is None:
                    responses.append(self._unexpected_error(item.request))
                    continue
                if state.lifecycle in {
                    _SessionLifecycle.FINALIZED,
                    _SessionLifecycle.QUARANTINED,
                    _SessionLifecycle.UNRESOLVED,
                }:
                    # An infer caller may disconnect after this shielded worker
                    # completes but before its event-loop callback runs. The
                    # endpoint has already released the terminal session; never
                    # dereference its cleared DepthVideo while publishing a result
                    # nobody can consume.
                    self.request_completed(item.request.session_id)
                    responses.append(self._unexpected_error(item.request))
                    continue
                if result.error is not None:
                    # Continuation and its CUDA sync execute after session mutation
                    # may have begun. The state has no rollback protocol, so poison
                    # only this session and preserve every other session's owner.
                    if result.mutation_started:
                        self._quarantine_session(state, str(result.error))
                    response = DroidFrameResponse(
                        ownership=item.request.ownership,
                        error=ServiceError(
                            ErrorCode.MODEL_FAILURE, str(result.error), retryable=False,
                            ownership=item.request.ownership, batch_id=batch_id,
                        ),
                    )
                    self._record_push_result(state, item.request, response)
                    self.request_completed(item.request.session_id)
                    responses.append(response)
                    continue
                state.mutation_started = False
                keyframe_count = int(state.video.counter.value) if state.video is not None else len(state.keyframe_source)
                validity = FrameValidity(
                    frame_id=item.request.frame_id,
                    source_timestamp_s=item.request.source_timestamp_s,
                    admitted=True,
                    keyframe_added=result.keyframe_added,
                    skip_reason=None if result.keyframe_added else "insufficient_motion_or_first_frame",
                )
                trace = DroidBatchTrace(
                    batch_id=batch_id,
                    replica_id=self._config.replica_id,
                    admitted_monotonic_s=admitted,
                    dispatched_monotonic_s=dispatched,
                    fnet_forward_started_monotonic_s=execution.fnet_started_monotonic_s,
                    fnet_completed_monotonic_s=execution.fnet_completed_monotonic_s,
                    completed_monotonic_s=completed,
                    fnet_forward_count=1,
                    session_local_forward_count=result.local_forward_count,
                    request_count=len(items),
                    effective_work_units=len(items),
                    model_load_count=self._model_load_count,
                    session_ids=tuple(other.request.session_id for other in items),
                    request_ids=tuple(other.request.ownership.request_id for other in items),
                    served_at_wall_unix_s=time.time(),
                    phase_timing=self._push_phase_timing(
                        item, state, execution.fnet_completed_monotonic_s - execution.fnet_started_monotonic_s,
                    ),
                )
                response = DroidFrameResponse(
                    ownership=item.request.ownership,
                    status=StepStatus(
                        ownership=item.request.ownership,
                        session_id=item.request.session_id,
                        frame_id=item.request.frame_id,
                        source_timestamp_s=item.request.source_timestamp_s,
                        validity=validity,
                        keyframe_count=keyframe_count,
                        trace=trace,
                    ),
                )
                self._record_push_result(state, item.request, response)
                self.request_completed(item.request.session_id)
            responses.append(response)
        return responses

    # ---- the actual DROID mechanism, run in a thread --------------------- #

    def _fnet_and_continue(self, items: Sequence[_PreparedFrame]) -> _GroupExecution:
        """Run a serialized fused fnet then independently report each continuation.

        A shared fnet failure affects every item because no continuation ran. Once
        fnet succeeds, each continuation is isolated: one session's exception does
        not rewrite successful sibling responses. CUDA synchronization follows fnet
        and each continuation so an asynchronous device error is attributed before a
        later session can mutate.
        """
        import torch

        with self._gpu_execution_lock:
            fnet_started = time.monotonic()
            try:
                # fnet consumes only the newly admitted normalized input; it does
                # not dereference DepthVideo. The CPU-offload window therefore
                # begins immediately afterwards around the session-local cnet/BA
                # continuation that actually reads and mutates video storage.
                stacked = torch.stack([item.normalized_input[0] for item in items], dim=0)
                execution = self._batch_snapshot.begin(ExecutionSpec(
                    operation="push_frame.fnet", execution_kind="fused_fnet_group", request_count=len(items),
                    item_count=len(items), item_kind="rgb_frame", image_item_count=len(items), forward_count=1,
                    max_batch_size=self._config.max_fnet_batch_size,
                ))
                try:
                    fmaps = self._backend.fnet(stacked)
                    if str(self._config.device).startswith("cuda"):
                        torch.cuda.synchronize(device=self._config.device)
                except BaseException as exc:
                    execution.finish(success=False, error_type=type(exc).__name__)
                    raise
                else:
                    execution.finish(success=True)
                fnet_completed = time.monotonic()
            except Exception as exc:
                return _GroupExecution(fnet_started, time.monotonic(), (), exc)
            fmaps_list = [fmaps[i:i + 1] for i in range(len(items))]
            # GPU-resident concurrent BA: when offload is disabled and
            # max_concurrent_ba > 1, independent sessions overlap continuations on
            # separate CUDA streams (bounded by the permit). GPU-resident tensors
            # are fully resident, so there is no H2D transfer lifetime hazard.
            # Offload (or serialized) runs the sequential loop below.
            gpu_resident = all(
                self._sessions[item.request.session_id].cpu_offload_state is None
                for item in items
            )
            if gpu_resident and self._config.max_concurrent_ba > 1:
                return self._run_continuations_concurrently(items, fmaps_list, fnet_started, fnet_completed)
            results: list[_ContinuationResult] = []
            for index, item in enumerate(items):
                state = self._sessions[item.request.session_id]
                state.mutation_started = True
                try:
                    # DROID's frontend may write the next pose/depth slot after a
                    # keyframe append, hence the current count plus one slot.
                    with self._active_video_window(state, int(state.video.counter.value) + 2):
                        local_forwards, keyframe_added = self._continuation_fn(self, state, item, fmaps_list[index])
                        if str(self._config.device).startswith("cuda"):
                            torch.cuda.synchronize(device=self._config.device)
                    _record_replay_frame(state, item)
                except Exception as exc:
                    import traceback as _tb
                    logger.error(
                        "DROID continuation failed for session %s counter=%d:\n%s",
                        item.request.session_id,
                        int(state.video.counter.value),
                        _tb.format_exc(),
                    )
                    results.append(_ContinuationResult(error=exc, mutation_started=True))
                else:
                    results.append(_ContinuationResult(local_forwards, keyframe_added, mutation_started=True))
        return _GroupExecution(fnet_started, fnet_completed, tuple(results))

    def _run_continuations_concurrently(self, items, fmaps_list, fnet_started, fnet_completed):
        """Overlap independent GPU-resident session continuations on separate CUDA
        streams, bounded by the BA permit. GPU-resident only; the shared fused
        fnet already completed before this runs.
        """
        import torch

        use_cuda = str(self._config.device).startswith("cuda")
        main_stream = torch.cuda.current_stream(self._config.device) if use_cuda else None
        states = [self._sessions[item.request.session_id] for item in items]
        for state in states:
            state.mutation_started = True

        def run_one(index: int):
            state = states[index]
            self._ba_permit.acquire()
            with self._ba_overlap_lock:
                self._running_ba_forwards += 1
                self._peak_simultaneous_forwards = max(
                    self._peak_simultaneous_forwards, self._running_ba_forwards,
                )
            try:
                if use_cuda:
                    stream = torch.cuda.Stream(self._config.device)
                    stream.wait_stream(main_stream)
                else:
                    stream = None
                with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                    try:
                        local_forwards, keyframe_added = self._continuation_fn(self, state, items[index], fmaps_list[index])
                        _record_replay_frame(state, items[index])
                    except Exception as exc:
                        return _ContinuationResult(error=exc, mutation_started=True)
                    if stream is not None:
                        main_stream.wait_stream(stream)
                        torch.cuda.synchronize(device=self._config.device)
                    return _ContinuationResult(local_forwards, keyframe_added, mutation_started=True)
            finally:
                with self._ba_overlap_lock:
                    self._running_ba_forwards -= 1
                self._ba_permit.release()

        max_workers = min(len(items), self._config.max_concurrent_ba)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="droid-gpu-resident-ba") as executor:
            results = list(executor.map(run_one, range(len(items))))
        return _GroupExecution(fnet_started, fnet_completed, tuple(results))

    @staticmethod
    def _infer_video_capacity(video: Any) -> int:
        """Read the slot dimension without relying on auxiliary index tensors."""
        if video is None:
            return 0
        for name in ("images", "poses", "disps", "tstamp"):
            value = getattr(video, name, None)
            shape = getattr(value, "shape", None)
            if shape and len(shape) >= 1:
                return int(shape[0])
        return 0

    @staticmethod
    def _initialize_depth_video_suffix(name: str, tensor: Any, start: int, stop: int) -> None:
        """Apply the recovered DepthVideo constructor defaults to a new suffix."""
        suffix = tensor[start:stop]
        if hasattr(suffix, "zero_"):
            suffix.zero_()
            if name == "disps":
                suffix.fill_(1.0)
            elif name == "poses" and getattr(suffix, "ndim", 0) >= 1 and suffix.shape[-1] >= 7:
                suffix[..., 6].fill_(1.0)
            return
        import numpy as np

        suffix[...] = 0
        if name == "disps":
            suffix[...] = 1.0
        elif name == "poses" and getattr(suffix, "ndim", 0) >= 1 and suffix.shape[-1] >= 7:
            suffix[..., 6] = 1.0

    @classmethod
    def _grow_slot_tensor(cls, name: str, tensor: Any, capacity: int, preserved_slots: int) -> Any:
        """Reallocate one slot-indexed tensor, preserving the predictive prefix."""
        shape = getattr(tensor, "shape", None)
        if not shape or len(shape) == 0 or int(shape[0]) >= capacity:
            return tensor
        new_shape = (capacity, *tuple(int(dim) for dim in shape[1:]))
        if hasattr(tensor, "new_empty"):
            grown = tensor.new_empty(new_shape)
        else:
            import numpy as np

            grown = np.empty(new_shape, dtype=getattr(tensor, "dtype", None))
        cls._initialize_depth_video_suffix(name, grown, 0, capacity)
        copied = min(max(0, int(preserved_slots)), int(shape[0]), capacity)
        if copied:
            if hasattr(grown, "copy_"):
                grown[:copied].copy_(tensor[:copied])
            else:
                grown[:copied] = tensor[:copied]
        return grown

    def _ensure_video_capacity(self, state: _SessionState, required_slots: int) -> None:
        """Grow a session at a quiescent point under the actor GPU lock."""
        with self._gpu_execution_lock:
            self._ensure_video_capacity_locked(state, required_slots)

    def _ensure_video_capacity_locked(self, state: _SessionState, required_slots: int) -> None:
        """Locked implementation shared by push admission and finalization.

        The frontend's ``graph.damping`` is always kept at least as large as the
        required slot count, independently of whether the DepthVideo buffer itself
        grew. ``damping`` is created from a transient GPU prefix during frontend
        initialization and can be undersized even when ``video.disps`` is already
        large (e.g. the B1024 path where the buffer never triggers growth). Without
        this, BA indexes ``damping[counter]`` past its transient-init size.
        """
        required = int(required_slots)
        damping_grown = self._ensure_damping_capacity(state, required)
        if required <= state.effective_capacity and required <= state.video_capacity and not damping_grown:
            return
        hard_max = min(
            self._config.max_buffer_slots,
            self._config.max_result_journal_entries_per_session - 1,
        )
        if required > hard_max:
            raise ContractValidationError(
                f"DROID session {state.session_id} requires {required} slots, exceeding "
                f"effective max_buffer_slots={hard_max}"
            )
        target = max(state.effective_capacity, state.video_capacity, 1)
        while target < required:
            target = min(hard_max, max(target * 2, required))
        if target + 1 > self._config.max_result_journal_entries_per_session:
            raise ContractValidationError(
                "DROID result journal bound cannot cover the grown session capacity"
            )
        video = state.video
        if video is None:
            raise ContractValidationError("DROID session has no DepthVideo storage")
        # The frontend's graph damping is the only persistent factor-graph tensor
        # indexed by keyframe slots. Edge tensors are factor-sized and stay put.
        tensors: list[tuple[Any, str]] = []
        for name in _DEPTH_VIDEO_TENSOR_FIELDS:
            value = getattr(video, name, None)
            if getattr(value, "shape", None) and len(value.shape) >= 1:
                tensors.append((video, name))
        graph = getattr(getattr(state.frontend, "graph", None), "damping", None)
        if getattr(graph, "shape", None) and len(graph.shape) >= 1:
            tensors.append((state.frontend.graph, "damping"))
        preserved = int(getattr(getattr(video, "counter", None), "value", 0)) + 1
        for owner, name in tensors:
            old = getattr(owner, name)
            setattr(owner, name, self._grow_slot_tensor(name, old, target, preserved))
        if state.cpu_offload_state is not None:
            # Offload owns the canonical CPU tensors. Rebind its map to the same
            # grown objects and keep its materialized prefix/accounting monotone.
            state.cpu_offload_state.cpu_tensors = {
                name: getattr(video, name)
                for name in _DEPTH_VIDEO_TENSOR_FIELDS
                if getattr(video, name, None) is not None
            }
            state.cpu_offload_state.capacity = target
            state.cpu_offload_state.cpu_capacity_bytes = sum(
                _DepthVideoCpuOffloader._tensor_bytes(value)
                for value in state.cpu_offload_state.cpu_tensors.values()
            )
        state.video_capacity = target
        state.effective_capacity = target
        state.journal_capacity = target + 1
        # The damping tensor is slot-indexed and may have been created from a
        # transient GPU prefix; keep it in lockstep with the grown video capacity.
        self._ensure_damping_capacity(state, target)

    def _ensure_damping_capacity(self, state: _SessionState, required_slots: int) -> bool:
        """Grow ``frontend.graph.damping`` to cover ``required_slots`` if needed.

        Returns True when a grow happened. ``damping`` is the one persistent
        FactorGraph tensor indexed by keyframe slots; edge tensors are factor-sized
        and stay untouched. This runs independently of video-buffer growth because
        damping can be undersized from a transient GPU prefix even when the DepthVideo
        buffer is already large.
        """
        graph = getattr(state.frontend, "graph", None)
        damping = getattr(graph, "damping", None)
        shape = getattr(damping, "shape", None)
        if not shape or len(shape) == 0 or int(shape[0]) >= required_slots:
            return False
        preserved = int(getattr(getattr(state.video, "counter", None), "value", 0)) + 1
        grown = self._grow_slot_tensor("disps", damping, required_slots, preserved)
        if graph is not None:
            graph.damping = grown
        return True

    @contextlib.contextmanager
    def _active_video_window(self, state: _SessionState, required_slots: int):
        """Materialize and release a session-local DepthVideo window around CUDA work."""
        offload = state.cpu_offload_state
        if offload is None:
            yield
            return
        assert self._cpu_offloader is not None
        self._cpu_offloader.activate(offload, state.video, required_slots)
        try:
            yield
        finally:
            self._cpu_offloader.release(offload, state.video)

    def _push_phase_timing(
        self, item: _PreparedFrame, state: _SessionState, fnet_s: float,
    ) -> DroidPhaseTiming:
        """Emit only measured host spans; stages not reached remain explicit nulls."""
        continuation = state.continuation_phase_timings
        values = {
            "preprocessing_h2d_s": (
                item.preprocessing_h2d_completed_monotonic_s
                - item.preprocessing_h2d_started_monotonic_s
            ),
            "fnet_s": fnet_s,
            "correlation_update_s": continuation.correlation_update_s,
            "cnet_s": continuation.cnet_s,
            "frontend_ba_s": continuation.frontend_ba_s,
            "backend_7_s": None,
            "backend_12_s": None,
            "filler_s": None,
            "encoding_s": None,
        }
        unavailable = tuple(stage.removesuffix("_s") for stage, value in values.items() if value is None)
        transfer_ms = state.cpu_offload_state.last_transfer_ms if state.cpu_offload_state is not None else None
        return DroidPhaseTiming(
            **values,
            cpu_offload_transfer_ms=transfer_ms,
            unavailable_stages=unavailable,
        )

    def _session_local_track(self, state: _SessionState, item: _PreparedFrame, gmap_any: Any) -> int:
        """Session-local continuation after the fused fnet: correlation/update/append/frontend.

        Returns the count of session-local forwards performed (cnet + update + BA).
        Mirrors ``MotionFilter.track`` and ``Droid.track`` (frontend) faithfully.
        """
        import torch
        import lietorch
        import geom.projective_ops as pops  # type: ignore[import-not-found]
        from modules.corr import CorrBlock  # type: ignore[import-not-found]

        filt = state.motion_state
        video = state.video
        gmap = gmap_any.squeeze(0)  # [128,gh,gw]
        mask = item.mask_grid
        local_fwds = 0
        ht = gmap.shape[-2]
        wd = gmap.shape[-1]
        if mask is None:
            mask = torch.zeros([ht, wd], device=gmap.device)

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        intrinsics = torch.as_tensor(
            list(state.camera.intrinsics), device=video.poses.device, dtype=torch.float32
        ) / 8.0

        with torch.no_grad():
            if video.counter.value == 0:
                # First frame: always add. cnet is a session-local forward.
                net, inp = self._backend.cnet(item.normalized_input[:, [0]])
                local_fwds += 1
                filt.net, filt.inp, filt.fmap = net, inp, gmap
                video.append(
                    item.request.source_timestamp_s,
                    self._rgb_hwc_to_bgr_chw_uint8(item)[0],
                    Id, 1.0, None, intrinsics, gmap, net[0, 0], inp[0, 0], mask,
                )
                state.keyframe_source.append((item.request.frame_id, item.request.source_timestamp_s))
            else:
                coords0 = pops.coords_grid(ht, wd, device=gmap.device)[None, None]
                corr = CorrBlock(filt.fmap[None, [0]], gmap[None, [0]])(coords0)
                _, delta, weight = self._backend.update_op(filt.net[None], filt.inp[None], corr)
                local_fwds += 1
                if delta.norm(dim=-1).mean().item() > filt.thresh:
                    filt.count = 0
                    net, inp = self._backend.cnet(item.normalized_input[:, [0]])
                    local_fwds += 1
                    filt.net, filt.inp, filt.fmap = net, inp, gmap
                    video.append(
                        item.request.source_timestamp_s,
                        self._rgb_hwc_to_bgr_chw_uint8(item)[0],
                        None, None, None, intrinsics, gmap, net[0], inp[0], mask,
                    )
                    state.keyframe_source.append((item.request.frame_id, item.request.source_timestamp_s))
                else:
                    filt.count += 1
        # Keep the unused compatibility implementation semantically identical to
        # the injected real continuation above.
        frontend_before = state.frontend.is_initialized
        _run_frontend_without_grad(state.frontend)
        if state.frontend.is_initialized and not frontend_before:
            local_fwds += 1
        elif state.frontend.is_initialized:
            local_fwds += 1
        _record_exact_dense_source_frame(state, item)
        state.frames_pushed += 1
        return local_fwds

    # ---- finalize -------------------------------------------------------- #

    async def finalize(self, request: DroidFinalizeRequest) -> DroidFinalizeResponse:
        if request.model_revision and request.model_revision != self._config.model_revision:
            return self._finalize_error(request.ownership, ErrorCode.VALIDATION, "model_revision mismatch with resident DROID backend")
        key = self._finalize_journal_key(request)
        signature = self._finalize_signature(request)
        with self._state_lock:
            state = self._sessions.get(request.session_id)
            if state is None:
                return self._finalize_error(request.ownership, ErrorCode.VALIDATION, f"unknown DROID session {request.session_id!r}")
            existing = state.result_journal.get(key)
            if existing is not None:
                self._touch_terminal(request.session_id)
                if existing.signature != signature:
                    return self._finalize_error(request.ownership, ErrorCode.CONFLICT, "request_id was reused with different finalize content")
                if existing.response is not None:
                    assert isinstance(existing.response, DroidFinalizeResponse)
                    return existing.response
                return self._finalize_error(
                    request.ownership, ErrorCode.BACKPRESSURE,
                    "finalize outcome is still in progress; retry with the same ownership", retryable=True,
                )
            if state.finalize_request_id is not None and state.finalize_request_id != request.ownership.request_id:
                return self._finalize_error(
                    request.ownership, ErrorCode.CONFLICT,
                    f"session finalization is owned by request_id={state.finalize_request_id!r}",
                )
            if state.lifecycle is not _SessionLifecycle.OPEN:
                return self._finalize_error(
                    request.ownership, ErrorCode.CONFLICT,
                    f"DROID session {request.session_id!r} is terminal ({state.lifecycle.value})",
                    terminal=state.lifecycle in {
                        _SessionLifecycle.FINALIZED, _SessionLifecycle.QUARANTINED, _SessionLifecycle.UNRESOLVED,
                    },
                )
            if state.in_flight:
                return self._finalize_error(
                    request.ownership, ErrorCode.VALIDATION,
                    "cannot finalize a DROID session while a push_frame is admitted or in flight",
                )
            if len(state.result_journal) >= state.journal_capacity:
                return self._finalize_error(request.ownership, ErrorCode.BACKPRESSURE, "bounded DROID result journal is full", retryable=True)
            state.result_journal[key] = _ResultJournalEntry(signature=signature)
            state.finalize_request_id = request.ownership.request_id
            # Fewer than two keyframes cannot constrain motion. This finalize is a
            # terminal unresolved result: retain the typed outcome, free every GPU
            # and dense-frame reference, and leave only a bounded tombstone.
            if state.video is None or int(state.video.counter.value) < 2:
                response = self._finalize_error(
                    request.ownership,
                    ErrorCode.UNRESOLVED,
                    "DROID finalize is unresolved with fewer than two keyframes; no static trajectory was emitted",
                    terminal=True,
                )
                self._record_finalize_result(state, request, response)
                self._mark_terminal(state, _SessionLifecycle.UNRESOLVED)
                return response
            state.lifecycle = _SessionLifecycle.FINALIZING
            state.mutation_started = False
        batch_id = uuid4().hex
        started = time.monotonic()
        self._running_batches += 1
        worker = asyncio.create_task(asyncio.to_thread(
            self._finalize_session, state, batch_id, started, request.ownership,
        ))
        try:
            result = await asyncio.shield(worker)
        except asyncio.CancelledError:
            # The worker owns the tracker inside the GPU lock. Cancellation only
            # detaches this HTTP caller; it never declares the GPU work terminal.
            worker.add_done_callback(lambda task: self._complete_cancelled_finalize(task, state, request, batch_id))
            raise
        except Exception as exc:
            self._running_batches -= 1
            response = self._finalize_error(
                request.ownership, ErrorCode.MODEL_FAILURE, str(exc), batch_id=batch_id,
                terminal=state.mutation_started,
            )
            self._complete_finalize_failure(state, request, response, exc)
            return response
        self._running_batches -= 1
        response = DroidFinalizeResponse(ownership=request.ownership, camera_state=result, terminal=True)
        with self._state_lock:
            self._record_finalize_result(state, request, response)
            self._mark_terminal(state, _SessionLifecycle.FINALIZED)
        return response

    def _complete_cancelled_finalize(
        self,
        task: "asyncio.Task[CameraState]",
        state: _SessionState,
        request: DroidFinalizeRequest,
        batch_id: str,
    ) -> None:
        self._running_batches -= 1
        try:
            result = task.result()
        except Exception as exc:
            response = self._finalize_error(
                request.ownership, ErrorCode.MODEL_FAILURE, str(exc), batch_id=batch_id,
                terminal=state.mutation_started,
            )
            self._complete_finalize_failure(state, request, response, exc)
        else:
            response = DroidFinalizeResponse(ownership=request.ownership, camera_state=result, terminal=True)
            with self._state_lock:
                # release_session may have already quarantined this lifecycle
                # after the caller disconnected. Do not overwrite that terminal
                # state or resurrect cleared resources from a late callback.
                if state.lifecycle is _SessionLifecycle.FINALIZING:
                    self._record_finalize_result(state, request, response)
                    self._mark_terminal(state, _SessionLifecycle.FINALIZED)

    def _complete_finalize_failure(
        self,
        state: _SessionState,
        request: DroidFinalizeRequest,
        response: DroidFinalizeResponse,
        exc: Exception,
    ) -> None:
        logger.error("DROID finalize failed for session %s: %s", state.session_id, exc)
        with self._state_lock:
            self._record_finalize_result(state, request, response)
            if state.mutation_started:
                self._mark_terminal(state, _SessionLifecycle.QUARANTINED)
            else:
                state.lifecycle = _SessionLifecycle.OPEN
                state.finalize_request_id = None

    def _release_finalized_session(self, state: _SessionState) -> None:
        """Compatibility helper: retire GPU state into the bounded LRU."""
        with self._state_lock:
            self._mark_terminal(state, _SessionLifecycle.FINALIZED)

    async def release_session(self, session_id: str) -> bool:
        """Quarantine an abandoned lifecycle after every CUDA owner is quiescent."""
        return await asyncio.to_thread(self._release_session_after_gpu_quiescent, session_id)

    def _release_session_after_gpu_quiescent(self, session_id: str) -> bool:
        """Wait for shielded push/finalize CUDA work before clearing its state."""
        with self._gpu_execution_lock:
            with self._state_lock:
                state = self._sessions.get(session_id)
                if state is None or state.lifecycle in {
                    _SessionLifecycle.FINALIZED,
                    _SessionLifecycle.QUARANTINED,
                    _SessionLifecycle.UNRESOLVED,
                }:
                    return False
                self._mark_terminal(state, _SessionLifecycle.QUARANTINED)
                return True

    def _retry_with_lower_filter_thresholds_locked(self, state: _SessionState) -> bool:
        """Replay accepted frames into fresh DROID objects with lower motion thresholds.

        The caller holds ``_gpu_execution_lock``. Retry state remains isolated until
        it has at least two keyframes, so an unsuccessful threshold attempt cannot
        corrupt the original sequence. The runtime's CPU-offload and capacity state
        is rebuilt with the fresh DROID objects before it is installed atomically.
        """
        source_frames = tuple(state.dense_source)
        if len(source_frames) != len(state.replay_frames):
            raise RuntimeError("DROID internal retry cannot align replay frames with the dense source timeline")

        for retry_thresh in (1.2, 0.6):
            retry_options = replace(state.options, filter_thresh=retry_thresh)
            retry_objects = dict(self._session_factory(
                self._backend, self._config, state.camera, state.image_shape, retry_options,
            ))
            retry_state = _SessionState(
                session_id=state.session_id,
                camera=state.camera,
                image_shape=state.image_shape,
                options=retry_options,
                ownership=state.ownership,
                created_monotonic_s=state.created_monotonic_s,
                video=retry_objects.get("video"),
                motion_state=retry_objects.get("filter"),
                frontend=retry_objects.get("frontend"),
                backend=retry_objects.get("backend"),
                filler=retry_objects.get("filler"),
                effective_capacity=retry_options.buffer,
                video_capacity=self._infer_video_capacity(retry_objects.get("video")) or retry_options.buffer,
                journal_capacity=retry_options.buffer + 1,
            )
            if self._cpu_offloader is not None and retry_state.video is not None:
                retry_state.cpu_offload_state = self._cpu_offloader.initialize(retry_state.video)

            for (frame_id, timestamp_s), (rgb_hwc, static_confidence_mask) in zip(
                source_frames, state.replay_frames,
            ):
                item = self._prepare_replay_frame(
                    retry_state, frame_id, timestamp_s, rgb_hwc, static_confidence_mask,
                )
                required_slots = int(retry_state.video.counter.value) + 2
                self._ensure_video_capacity(retry_state, required_slots)
                with self._active_video_window(retry_state, required_slots):
                    # Identical fnet input, mask conversion, RGB normalization, BGR
                    # DepthVideo storage, and continuation as the original push path.
                    gmap = self._backend.fnet(item.normalized_input)
                    self._continuation_fn(self, retry_state, item, gmap)

            retry_n_key = int(retry_state.video.counter.value)
            if retry_n_key < 2:
                continue
            if len(retry_state.dense_source) != len(source_frames):
                raise RuntimeError("DROID internal retry did not rebuild the dense source timeline")
            if len(retry_state.keyframe_source) != retry_n_key:
                raise RuntimeError("DROID internal retry did not rebuild the keyframe source timeline")

            state.options = retry_options
            state.video = retry_state.video
            state.motion_state = retry_state.motion_state
            state.frontend = retry_state.frontend
            state.backend = retry_state.backend
            state.filler = retry_state.filler
            state.cpu_offload_state = retry_state.cpu_offload_state
            state.keyframe_source = retry_state.keyframe_source
            state.dense_source = retry_state.dense_source
            state.dense_images_bgr_chw_uint8 = retry_state.dense_images_bgr_chw_uint8
            # Retain the original accepted inputs; retry state is only a fresh
            # execution image and does not own a second replay journal.
            state.continuation_phase_timings = retry_state.continuation_phase_timings
            state.frames_pushed = retry_state.frames_pushed
            state.effective_capacity = retry_state.effective_capacity
            state.video_capacity = retry_state.video_capacity
            state.journal_capacity = retry_state.journal_capacity
            return True
        return False

    def _prepare_replay_frame(
        self,
        state: _SessionState,
        frame_id: str,
        timestamp_s: float,
        rgb_hwc: Any,
        static_confidence_mask: Any | None,
    ) -> _PreparedFrame:
        """Recreate exactly the push-frame preprocessing inputs from CPU retention."""
        import numpy as np

        rgb = np.ascontiguousarray(np.asarray(rgb_hwc, dtype=np.uint8)).copy()
        mask = (
            None
            if static_confidence_mask is None
            else np.ascontiguousarray(np.asarray(static_confidence_mask, dtype=np.float32)).copy()
        )
        request = DroidFrameRequest(
            ownership=state.ownership,
            session_id=state.session_id,
            frame_id=frame_id,
            source_timestamp_s=float(timestamp_s),
            rgb=TensorPayload(data=rgb.tobytes(), shape=tuple(rgb.shape), dtype="uint8"),
            static_confidence_mask=(
                None if mask is None
                else TensorPayload(data=mask.tobytes(), shape=tuple(mask.shape), dtype="float32")
            ),
            model_revision=self._config.model_revision,
        )
        return _PreparedFrame(
            request=request,
            rgb_hwc_uint8=rgb,
            normalized_input=self._normalize_input(rgb, state),
            mask_grid=self._static_confidence_mask_to_grid(mask, state),
            static_confidence_mask=mask,
            preprocessing_h2d_started_monotonic_s=0.0,
            preprocessing_h2d_completed_monotonic_s=0.0,
            admitted_monotonic_s=0.0,
        )

    def _finalize_session(
        self, state: _SessionState, batch_id: str, started: float, ownership: Ownership,
    ) -> CameraState:
        """Serialize finalization and measure only the GPU-lock-held execution."""
        with self._gpu_execution_lock:
            execution = self._batch_snapshot.begin(ExecutionSpec(
                operation="finalize", execution_kind="session_finalize", request_count=1,
                item_count=None, item_kind="session", image_item_count=None, forward_count=1,
                max_batch_size=None,
            ))
            try:
                result = self._finalize_session_locked(state, batch_id, started, ownership)
            except BaseException as exc:
                execution.finish(success=False, error_type=type(exc).__name__)
                raise
            else:
                execution.finish(success=True)
                return result

    def _finalize_session_locked(
        self, state: _SessionState, batch_id: str, started: float, ownership: Ownership,
    ) -> CameraState:
        """Run finalization with only its addressable video prefix on GPU."""
        n_key = int(state.video.counter.value)
        if n_key < 2 and state.replay_frames:
            self._retry_with_lower_filter_thresholds_locked(state)
            n_key = int(state.video.counter.value)
        if n_key < 2:
            raise RuntimeError("DROID finalize unresolved: fewer than two keyframes after internal retry")
        # PoseTrajectoryFiller appends dense frames in chunks of at most 16 before
        # restoring the keyframe counter. Grow the complete slot manifest first;
        # this is still quiescent because the caller holds _gpu_execution_lock.
        self._ensure_video_capacity_locked(state, n_key + 16)
        with self._active_video_window(state, n_key + 16):
            result = self._finalize_session_active_locked(state, batch_id, started, ownership)
        if state.cpu_offload_state is None:
            return result
        phase = replace(
            result.trace.phase_timing,
            cpu_offload_transfer_ms=state.cpu_offload_state.last_transfer_ms,
        )
        return replace(result, trace=replace(result.trace, phase_timing=phase))

    def _finalize_session_active_locked(
        self, state: _SessionState, batch_id: str, started: float, ownership: Ownership,
    ) -> CameraState:
        """Execute native BA/fill on an already-materialized DepthVideo prefix."""
        import numpy as np
        import torch
        import lietorch

        video = state.video
        n_key = int(video.counter.value)
        # This is the first irreversible finalization mutation. Any later failure
        # quarantines this session instead of presenting a rolled-back fiction.
        state.mutation_started = True
        state.frontend = None
        torch.cuda.empty_cache()
        backend_7_started = time.monotonic()
        state.backend(7)
        backend_7_completed = time.monotonic()
        torch.cuda.empty_cache()
        backend_12_started = time.monotonic()
        state.backend(12)
        backend_12_completed = time.monotonic()

        # Dense trajectory fill over every exact pushed source frame. Keep the
        # retained images on CPU and move one frame at a time to the backend device;
        # materializing the complete stream on GPU would make session memory scale
        # with video length.
        source_frames = state.dense_source or state.keyframe_source
        if state.dense_source and len(state.dense_images_bgr_chw_uint8) != len(state.dense_source):
            raise RuntimeError("DROID dense source/image timeline is incomplete")
        intrinsics_full = torch.as_tensor(
            list(state.camera.intrinsics), device=video.poses.device, dtype=torch.float32
        )

        # PoseTrajectoryFiller estimates poses only for frames rejected by the
        # motion filter. Feeding it frames that are already keyframes duplicates
        # their slots in DepthVideo and creates self-edges at matching timestamps;
        # a fully keyframed stream then can produce non-finite poses during the
        # filler graph update. Preserve the source timeline by scattering native
        # keyframe poses and filler poses back into their original frame order.
        key_xyzw = video.poses[:n_key].clone().cpu().numpy()
        keyframe_index = {
            frame_id: index
            for index, (frame_id, _timestamp_s) in enumerate(state.keyframe_source[:n_key])
        }
        non_key_indices = [
            dense_index
            for dense_index, (frame_id, _timestamp_s) in enumerate(source_frames)
            if frame_id not in keyframe_index
        ]

        def non_key_source_stream() -> Any:
            for dense_index in non_key_indices:
                frame_id, ts = source_frames[dense_index]
                yield (
                    float(ts),
                    self._source_frame_image(state, frame_id, video, dense_index=dense_index),
                    intrinsics_full,
                )

        filler_started = time.monotonic()
        if non_key_indices:
            camera_trajectory = state.filler(non_key_source_stream())  # camera-from-world, [non-key N,7]
            filled_xyzw = camera_trajectory.data.cpu().numpy()
            if len(filled_xyzw) != len(non_key_indices):
                raise RuntimeError("DROID trajectory filler returned the wrong non-keyframe count")
        else:
            filled_xyzw = np.empty((0, 7), dtype=key_xyzw.dtype)
        filler_completed = time.monotonic()

        dense_xyzw = np.empty((len(source_frames), 7), dtype=key_xyzw.dtype)
        filled_index = 0
        for dense_index, (frame_id, _timestamp_s) in enumerate(source_frames):
            key_index = keyframe_index.get(frame_id)
            if key_index is not None:
                dense_xyzw[dense_index] = key_xyzw[key_index]
            else:
                dense_xyzw[dense_index] = filled_xyzw[filled_index]
                filled_index += 1
        if filled_index != len(filled_xyzw):
            raise RuntimeError("DROID trajectory filler/source-frame alignment drifted")

        # Canonical source: internal poses are camera-from-world. Invert to
        # world-from-camera for BOTH dense and keyframe exports using one code path
        # (camera_from_world_xyzw_to_world_camera_matrix), so dense and keyframe
        # can never disagree. DROID's terminate() does camera_trajectory.inv();
        # this materializes that inverse as 4x4 matrices.
        T_world_camera_dense = camera_from_world_xyzw_to_world_camera_matrix(dense_xyzw)
        T_world_camera_key = camera_from_world_xyzw_to_world_camera_matrix(key_xyzw)

        # T_camera_world is the matrix inverse of T_world_camera (one source).
        T_camera_world_dense = np.linalg.inv(T_world_camera_dense)

        # Intrinsics in the model pixel grid (DROID stores /8 internally; restore).
        intr_model = video.intrinsics[:max(n_key, 1)].clone()
        intr_model[:, :2] *= 8.0
        intr_model[:, 2:] *= 8.0
        # Expand to 3x3 for downstream reuse.
        intr_3x3 = self._intrinsics_to_3x3(intr_model, n_key)

        disps = video.disps[:n_key].detach().cpu().numpy()

        reprojection_error = float(state.backend.errors[-1]) if getattr(state.backend, "errors", None) else None
        finite_dense = bool(np.isfinite(T_world_camera_dense).all())
        finite_key = bool(np.isfinite(T_world_camera_key).all())
        valid_ratio = 1.0 if n_key > 0 else 0.0
        finite_ratio = float(finite_dense and finite_key)
        uncertainty = DroidUncertainty(
            scale_status="up_to_scale",
            reprojection_error=reprojection_error,
            valid_keyframe_ratio=valid_ratio,
            finite_pose_ratio=finite_ratio,
            note="DROID monocular translation is up to scale; pair with UniDepth for metric scale.",
        )

        keyframe_mapping = tuple(
            KeyframeSourceMapping(
                keyframe_index=i,
                source_frame_id=state.keyframe_source[i][0],
                source_timestamp_s=state.keyframe_source[i][1],
            )
            for i in range(min(n_key, len(state.keyframe_source)))
        )
        dense_mapping = tuple(
            DenseSourceMapping(
                dense_index=i,
                source_frame_id=source_frames[i][0],
                source_timestamp_s=source_frames[i][1],
            )
            for i in range(len(source_frames))
        )

        encoding_started = time.monotonic()
        T_world_camera_payload = _as_tensor_payload(T_world_camera_dense)
        T_camera_world_payload = _as_tensor_payload(T_camera_world_dense)
        intrinsics_payload = _as_tensor_payload(
            intr_3x3 if intr_3x3.ndim == 3 else intr_3x3.reshape(1, 3, 3)
        )
        disparities_payload = _as_tensor_payload(disps)
        encoding_completed = time.monotonic()
        phase_values = {
            "preprocessing_h2d_s": None,
            "fnet_s": None,
            "correlation_update_s": None,
            "cnet_s": None,
            "frontend_ba_s": None,
            "backend_7_s": backend_7_completed - backend_7_started,
            "backend_12_s": backend_12_completed - backend_12_started,
            "filler_s": filler_completed - filler_started,
            "encoding_s": encoding_completed - encoding_started,
        }
        finalize_phase_timing = DroidPhaseTiming(
            **phase_values,
            # The wrapper replaces this with the completed H2D+D2H span after
            # it releases the active prefix. Keeping it separate prevents it
            # from being attributed to backend BA or trajectory filling.
            cpu_offload_transfer_ms=(
                state.cpu_offload_state.last_transfer_ms
                if state.cpu_offload_state is not None else None
            ),
            unavailable_stages=tuple(stage.removesuffix("_s") for stage, value in phase_values.items() if value is None),
        )
        completed = encoding_completed
        trace = DroidBatchTrace(
            batch_id=batch_id,
            replica_id=self._config.replica_id,
            admitted_monotonic_s=started,
            dispatched_monotonic_s=started,
            fnet_forward_started_monotonic_s=started,
            fnet_completed_monotonic_s=started,
            completed_monotonic_s=completed,
            fnet_forward_count=0,  # finalize runs BA/filler, not fnet
            session_local_forward_count=1,  # BA + filler are session-local
            request_count=1,
            effective_work_units=1,
            model_load_count=self._model_load_count,
            session_ids=(state.session_id,),
            request_ids=(ownership.request_id,),
            served_at_wall_unix_s=time.time(),
            phase_timing=finalize_phase_timing,
        )
        return CameraState(
            ownership=ownership,
            session_id=state.session_id,
            T_world_camera=T_world_camera_payload,
            T_camera_world=T_camera_world_payload,
            intrinsics_px=intrinsics_payload,
            disparities=disparities_payload,
            keyframe_mapping=keyframe_mapping,
            dense_mapping=dense_mapping,
            uncertainty=uncertainty,
            model_revision=self._config.model_revision,
            trace=trace,
        )

    # ---- helpers --------------------------------------------------------- #

    def _require_open_session(self, session_id: str) -> _SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            raise ContractValidationError(f"unknown DROID session {session_id!r}")
        if state.lifecycle is _SessionLifecycle.FINALIZING:
            raise ContractValidationError(f"DROID session {session_id!r} is finalizing")
        if state.lifecycle is _SessionLifecycle.QUARANTINED:
            raise ContractValidationError(f"DROID session {session_id!r} is quarantined after a partial mutation failure")
        if state.lifecycle is _SessionLifecycle.UNRESOLVED:
            raise ContractValidationError(f"DROID session {session_id!r} is unresolved and retired")
        if state.lifecycle is _SessionLifecycle.FINALIZED:
            raise ContractValidationError(f"DROID session {session_id!r} is finalized")
        return state

    def _release_session_resources(self, state: _SessionState) -> None:
        """Release all per-session GPU references and bounded exact-frame retention."""
        state.pending_frames.clear()
        state.dense_images_bgr_chw_uint8.clear()
        state.replay_frames.clear()
        state.dense_source.clear()
        state.keyframe_source.clear()
        state.video = None
        state.cpu_offload_state = None
        state.motion_state = None
        state.frontend = None
        state.backend = None
        state.filler = None
        state.in_flight = False

    def _quarantine_session(self, state: _SessionState, reason: str) -> None:
        """Poison an irreversibly mutated session while preserving its identity."""
        self._mark_terminal(state, _SessionLifecycle.QUARANTINED)
        logger.error("DROID session %s quarantined: %s", state.session_id, reason, exc_info=True)

    def _mark_terminal(self, state: _SessionState, lifecycle: _SessionLifecycle) -> None:
        if lifecycle not in {
            _SessionLifecycle.FINALIZED, _SessionLifecycle.QUARANTINED, _SessionLifecycle.UNRESOLVED,
        }:
            raise ValueError(f"non-terminal DROID lifecycle {lifecycle}")
        state.lifecycle = lifecycle
        self._release_session_resources(state)
        self._terminal_lru[state.session_id] = None
        self._terminal_lru.move_to_end(state.session_id)
        while len(self._terminal_lru) > self._config.max_terminal_tombstones:
            evicted, _ = self._terminal_lru.popitem(last=False)
            candidate = self._sessions.get(evicted)
            if candidate is not None and candidate.lifecycle in {
                _SessionLifecycle.FINALIZED, _SessionLifecycle.QUARANTINED, _SessionLifecycle.UNRESOLVED,
            }:
                del self._sessions[evicted]

    def _touch_terminal(self, session_id: str) -> None:
        if session_id in self._terminal_lru:
            self._terminal_lru.move_to_end(session_id)

    @staticmethod
    def _push_journal_key(request: DroidFrameRequest) -> str:
        return f"push:{request.ownership.request_id}"

    @staticmethod
    def _finalize_journal_key(request: DroidFinalizeRequest) -> str:
        return f"finalize:{request.ownership.request_id}"

    @staticmethod
    def _tensor_identity(tensor: TensorPayload | None) -> dict[str, Any] | None:
        if tensor is None:
            return None
        data = tensor.data
        if isinstance(data, (bytes, bytearray, memoryview)):
            digest = hashlib.sha256(bytes(data)).hexdigest()
        elif hasattr(data, "binary") and callable(data.binary):
            digest = hashlib.sha256(bytes(data.binary())).hexdigest()
        else:
            digest = f"{type(data).__module__}.{type(data).__qualname__}:{repr(data)}"
        return {"shape": list(tensor.shape), "dtype": tensor.dtype, "sha256": digest}

    def _push_signature(self, request: DroidFrameRequest) -> str:
        payload = {
            "ownership": request.ownership.to_wire(),
            "session_id": request.session_id,
            "frame_id": request.frame_id,
            "source_timestamp_s": request.source_timestamp_s,
            "model_revision": request.model_revision,
            "rgb": self._tensor_identity(request.rgb),
            "static_confidence_mask": self._tensor_identity(request.static_confidence_mask),
            "depth_m": self._tensor_identity(request.depth_m),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _finalize_signature(request: DroidFinalizeRequest) -> str:
        payload = {
            "ownership": request.ownership.to_wire(),
            "session_id": request.session_id,
            "model_revision": request.model_revision,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _forget_push_reservation(self, state: _SessionState, request: DroidFrameRequest) -> None:
        key = self._push_journal_key(request)
        entry = state.result_journal.get(key)
        if entry is not None and entry.response is None:
            del state.result_journal[key]
            if state.frame_request_owners.get(request.frame_id) == request.ownership.request_id:
                del state.frame_request_owners[request.frame_id]
            if not state.pending_frames:
                state.in_flight = False

    def _record_push_result(
        self, state: _SessionState, request: DroidFrameRequest, response: DroidFrameResponse,
    ) -> None:
        entry = state.result_journal.get(self._push_journal_key(request))
        if entry is None:
            raise RuntimeError("DROID push completed without an idempotency reservation")
        entry.response = response

    def _record_finalize_result(
        self, state: _SessionState, request: DroidFinalizeRequest, response: DroidFinalizeResponse,
    ) -> None:
        entry = state.result_journal.get(self._finalize_journal_key(request))
        if entry is None:
            raise RuntimeError("DROID finalize completed without an idempotency reservation")
        entry.response = response

    @staticmethod
    def _push_conflict(request: DroidFrameRequest, message: str) -> DroidFrameResponse:
        return DroidFrameResponse(
            ownership=request.ownership,
            error=ServiceError(ErrorCode.CONFLICT, message, retryable=False, ownership=request.ownership),
        )

    @staticmethod
    def _push_in_progress(request: DroidFrameRequest) -> DroidFrameResponse:
        return DroidFrameResponse(
            ownership=request.ownership,
            error=ServiceError(
                ErrorCode.BACKPRESSURE,
                "push-frame outcome is still in progress; retry with the same ownership",
                retryable=True,
                ownership=request.ownership,
            ),
        )

    @staticmethod
    def _finalize_error(
        ownership: Ownership,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        batch_id: str | None = None,
        terminal: bool = False,
    ) -> DroidFinalizeResponse:
        return DroidFinalizeResponse(
            ownership=ownership,
            error=ServiceError(code, message, retryable=retryable, ownership=ownership, batch_id=batch_id),
            terminal=terminal,
        )

    def _validation_error(self, ownership: Ownership, message: str) -> DroidCreateSessionResponse:
        return DroidCreateSessionResponse(
            ownership=ownership,
            error=ServiceError(ErrorCode.VALIDATION, message, retryable=False, ownership=ownership),
        )

    def _unexpected_error(self, request: DroidFrameRequest) -> DroidFrameResponse:
        return DroidFrameResponse(
            ownership=request.ownership,
            error=ServiceError(
                ErrorCode.MODEL_FAILURE, "session vanished during batch", retryable=False,
                ownership=request.ownership,
            ),
        )

    def _decode_rgb(self, request: DroidFrameRequest) -> Any:
        import numpy as np

        rgb = self._tensor_resolver(request.rgb.data, request.rgb.shape, request.rgb.dtype)
        if tuple(rgb.shape) != request.rgb.shape:
            raise ContractValidationError("decoded DROID RGB shape does not match contract")
        if rgb.dtype.name != request.rgb.dtype:
            rgb = rgb.astype(request.rgb.dtype, copy=False)
        if rgb.dtype != np.dtype("uint8"):
            raise ContractValidationError("DROID RGB must be uint8 after resolution")
        if not np.isfinite(rgb).all() or float(rgb.min()) < 0.0 or float(rgb.max()) > 255.0:
            raise ContractValidationError("DROID RGB values must be finite uint8 in [0, 255]")
        return np.ascontiguousarray(rgb)

    def _decode_static_confidence_mask(self, request: DroidFrameRequest, state: _SessionState) -> Any | None:
        """Resolve and validate the CPU mask before retaining it for a retry replay."""
        if request.static_confidence_mask is None:
            return None
        import numpy as np

        mask = request.static_confidence_mask
        arr = np.asarray(self._tensor_resolver(mask.data, mask.shape, mask.dtype), dtype=np.float32)
        gh, gw = state.image_shape.height // 8, state.image_shape.width // 8
        if arr.shape not in {(gh, gw), (state.image_shape.height, state.image_shape.width)}:
            raise ContractValidationError(
                f"static_confidence_mask shape {arr.shape} is neither the 1/8 grid ({gh},{gw}) "
                f"nor the full image grid"
            )
        return np.ascontiguousarray(arr).copy()

    def _static_confidence_mask_to_grid(self, mask: Any | None, state: _SessionState) -> Any:
        if mask is None:
            return None
        import torch

        gh, gw = state.image_shape.height // 8, state.image_shape.width // 8
        if mask.shape == (gh, gw):
            return torch.as_tensor(mask, device=self._config.device)
        # Downsample the validated full image grid by mean pooling (matches DROID).
        import torch.nn.functional as F

        t = torch.as_tensor(mask, device=self._config.device)[None, None]
        return F.avg_pool2d(t, 8)[0, 0]

    def _decode_mask(self, request: DroidFrameRequest, state: _SessionState) -> Any:
        """Compatibility wrapper for callers/tests that only need the model grid."""
        return self._static_confidence_mask_to_grid(
            self._decode_static_confidence_mask(request, state), state,
        )

    def _normalize_input(self, rgb_hwc: Any, state: _SessionState) -> Any:
        """Normalize an RGB HWC uint8 array to the exact tensor DROID's fnet consumes.

        The real ``MotionFilter.track`` receives ``image`` as ``[1,3,H,W]`` (cv2 BGR)
        and computes ``inputs = image[None, :, [2,1,0]] / 255`` then ImageNet-normalize.
        ``image[None, :, [2,1,0]]`` on ``[1,3,H,W]`` yields ``[1,1,3,H,W]`` (5D), and
        ``BasicEncoder`` in eval mode accepts that 5D input, returning ``[1,1,128,gh,gw]``
        which ``squeeze(0)`` reduces to ``[1,128,gh,gw]``.

        Callers send RGB ``[H,W,3]``. DROID's internal cv2 frames are BGR and the
        ``[2,1,0]`` swap converts BGR->RGB before ImageNet normalization. Since our
        caller already supplies RGB, we build the identical ``[1,1,3,H,W]`` RGB tensor
        (no BGR->RGB swap needed) and normalize with the same ImageNet constants.
        """
        import numpy as np
        import torch

        t = torch.as_tensor(np.asarray(rgb_hwc).copy(), device=self._config.device)  # [H,W,3] uint8 RGB
        t = t.permute(2, 0, 1)[None, None]  # [1,1,3,H,W] RGB
        t = t.to(self._config.device).float() / 255.0
        t = t.sub_(state.motion_state.MEAN).div_(state.motion_state.STDV)
        return t

    def _rgb_hwc_to_bgr_chw_uint8(self, item: _PreparedFrame) -> Any:
        """The uint8 BGR [1,3,H,W] image DROID stores in DepthVideo.images.

        DROID stores images in BGR (cv2 convention); callers send RGB, so swap.
        """
        import numpy as np
        import torch

        t = torch.as_tensor(np.array(item.rgb_hwc_uint8, copy=True), device=self._config.device)  # [H,W,3] RGB uint8
        return t.permute(2, 0, 1)[None][:, [2, 1, 0]]  # [1,3,H,W] BGR uint8

    def _intrinsics_to_3x3(self, intr_4: Any, n: int) -> Any:
        import numpy as np

        arr = intr_4[:max(n, 1)].detach().cpu().numpy()  # [N,4] = fx,fy,cx,cy
        out = np.zeros((arr.shape[0], 3, 3), dtype=np.float64)
        out[:, 0, 0] = arr[:, 0]
        out[:, 1, 1] = arr[:, 1]
        out[:, 0, 2] = arr[:, 2]
        out[:, 1, 2] = arr[:, 3]
        out[:, 2, 2] = 1.0
        return out

    def _source_frame_image(
        self,
        state: _SessionState,
        frame_id: str,
        video: Any,
        *,
        dense_index: int | None = None,
    ) -> Any:
        """Return the exact BGR ``[1,3,H,W]`` image for trajectory filling."""
        import torch

        if dense_index is not None and dense_index < len(state.dense_images_bgr_chw_uint8):
            image = state.dense_images_bgr_chw_uint8[dense_index]
            return torch.as_tensor(image, device=self._config.device)[None]
        # Compatibility for keyframe-only unit fixtures and legacy short sessions.
        for i, (fid, _ts) in enumerate(state.keyframe_source):
            if fid == frame_id:
                image = video.images[i]
                return image[None] if getattr(image, "ndim", 0) == 3 else image
        raise RuntimeError(f"DROID exact dense source image is unavailable for {frame_id}")

    @staticmethod
    def _depth_video_residency(video: Any) -> tuple[int, int]:
        """Return CPU/GPU bytes attributable to a single DepthVideo object."""
        cpu_bytes = 0
        gpu_bytes = 0
        for name in _DEPTH_VIDEO_TENSOR_FIELDS:
            tensor = getattr(video, name, None)
            bytes_ = _DepthVideoCpuOffloader._tensor_bytes(tensor)
            if bool(getattr(tensor, "is_cuda", False)):
                gpu_bytes += bytes_
            else:
                cpu_bytes += bytes_
        return cpu_bytes, gpu_bytes

    def session_allocator_memory(self, session_id: str) -> dict[str, int | bool]:
        """Per-session DepthVideo residency, never inferred from process-wide VRAM.

        The resident path reports the tensor placement it owns. The lazy offload
        path separates virtual index-space capacity from pages initialized by BA:
        a create response therefore reports zero materialized CPU bytes rather
        than implying a complete 15.5 GiB CPU buffer already exists.
        """
        with self._state_lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise ContractValidationError(f"unknown DROID session {session_id!r}")
            offload = state.cpu_offload_state
            if offload is not None:
                return {
                    "cpu_offload": True,
                    "cpu_resident_bytes": offload.cpu_resident_bytes,
                    "cpu_capacity_bytes": offload.cpu_capacity_bytes,
                    "cpu_materialized_slots": offload.materialized_slots,
                    "gpu_resident_bytes": offload.gpu_resident_bytes,
                    "gpu_peak_resident_bytes": offload.gpu_peak_resident_bytes,
                }
            cpu_bytes, gpu_bytes = self._depth_video_residency(state.video)
            return {
                "cpu_offload": False,
                "cpu_resident_bytes": cpu_bytes,
                "gpu_resident_bytes": gpu_bytes,
                "gpu_peak_resident_bytes": gpu_bytes,
            }

    def concurrency_diagnostics(self) -> dict[str, int]:
        """Return measured continuation overlap for the instrumented worker."""
        with self._ba_overlap_lock:
            return {"peak_simultaneous_forwards": self._peak_simultaneous_forwards}

    def batch_snapshot(self) -> dict[str, Any]:
        return self._batch_snapshot.snapshot(adapter_status=self.status().to_wire())

    def status(self) -> DeploymentStatus:
        return DeploymentStatus(
            deployment_name="droid",
            replica_id=self._config.replica_id,
            assigned_gpu=self._config.assigned_gpu,
            loaded_models=(self._config.model_revision,),
            admitted_pending=self._admitted_pending,
            running_batches=self._running_batches,
            model_load_count=self._model_load_count,
            active_sessions=self.resident_session_count,
        )


# --------------------------------------------------------------------------- #
# Helpers bridging the session image grid to frame validation. These are plain
# module functions so the frozen contract dataclasses stay pure.
# --------------------------------------------------------------------------- #


def _session_image_shape(sessions: dict[str, _SessionState], session_id: str) -> tuple[int, int]:
    state = sessions.get(session_id)
    if state is None:
        return (0, 0)
    return (state.image_shape.height, state.image_shape.width)
