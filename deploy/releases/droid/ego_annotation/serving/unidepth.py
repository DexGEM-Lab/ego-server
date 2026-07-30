"""Ray-free resident UniDepth adapter: one model forward per Serve batch callback.

This module is importable without Ray installed (ordinary unit tests import
``UniDepthAdapter`` directly). The Ray Serve deployment wrapper lives in
``ego_annotation.serving.deployment`` and imports this adapter.

Contract invariants enforced here:

* Request RGB is uint8 HWC only (``UNIDEPTH_RGB_DTYPE``). Float RGB is range
  ambiguous and rejected at the contract boundary.
* The real backend receives contiguous ``[B,C,H,W]`` uint8: the adapter stacks the
  per-image HWC arrays to ``[B,H,W,C]`` and transposes to ``[B,C,H,W]`` without
  dividing by 255. Upstream UniDepth ``model.infer`` performs the /255 itself.
* The resident config owns the model revision: a request whose ``model_revision``
  does not match the configured revision is rejected before batching, and every
  result carries the configured revision only.
* The deployment advertises one canonical HxW compatibility bucket. The adapter
  rejects incompatible (shape/dtype/revision/option) or overweight requests at
  admission, before the Serve batch callback, so one callback is exactly one
  upstream forward. ``infer_batch`` never splits a callback into several forwards.
* Real UniDepth outputs are squeezed from per-batch ``[B,1,H,W]`` depth/confidence
  and ``[B,3,3]`` intrinsics into documented per-image shapes.
* Batch timings are truthful monotonic ``time.monotonic()`` readings: admission,
  dispatch, model-forward start, and completion.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from ego_annotation.serving.batch_snapshot import BatchSnapshotTracker, ExecutionSpec, await_tracked_thread
from ego_annotation.serving.batching import BatchPolicy, assert_one_forward
from ego_annotation.serving.contracts import (
    SCHEMA_VERSION,
    BatchTrace,
    ContractValidationError,
    DeploymentStatus,
    ErrorCode,
    ServerIdentity,
    ServiceError,
    TensorPayload,
    UniDepthRequest,
    UniDepthResponse,
    UniDepthResult,
)


class UniDepthBackend(Protocol):
    """The model boundary. ``rgb`` is contiguous ``[B,C,H,W]`` uint8 in [0, 255]."""

    def infer(self, rgb: Any) -> Mapping[str, Any] | "UniDepthBackendResult": ...


TensorResolver = Callable[[Any, tuple[int, ...], str], Any]
BackendFactory = Callable[["UniDepthModelConfig"], UniDepthBackend]


@dataclass(frozen=True)
class UniDepthModelConfig:
    """Server-owned model settings; no request field can select a server path."""

    checkpoint: str
    model_revision: str
    device: str = "cuda"
    replica_id: str = "unidepth-gpu0"
    assigned_gpu: int = 0
    # Experiment-only attribution is deliberately opt-in.  Production defaults do
    # not synchronize CUDA or reset allocator counters on the request path.
    experiment_id: str | None = None
    application_release_sha: str | None = None
    checkpoint_digest: str | None = None
    application_release_path: str | None = None
    gcs_address: str | None = None
    http_port: int | None = None
    temp_dir: str | None = None
    performance_instrumentation: bool = False
    # One canonical HxW compatibility bucket. Default 540x960 preserves 16:9 from
    # representative EgoScale 1080x1920 frames at an explicit 0.5 px/K scale;
    # upstream UniDepth ``infer`` handles internal model resize. A square default
    # would silently distort geometry unless a letterbox/crop transform were
    # declared, so the default is non-square. Configurable per deployment.
    canonical_height: int = 540
    canonical_width: int = 960
    batch_policy: BatchPolicy = BatchPolicy(
        max_batch_size=8,
        batch_wait_timeout_s=2.0,
        max_queued_requests=64,
    )
    # ``None`` preserves the production adapter's historical unconstrained callback
    # behavior.  An explicit positive value is an experiment treatment which bounds
    # physical model forwards without disabling Serve request batching.
    max_concurrent_forwards: int | None = None
    # Experimental transport attribution. The endpoint still dual-accepts both
    # formats; this records the planned benchmark treatment in worker diagnostics.
    wire_format: str = "multipart"

    def __post_init__(self) -> None:
        if self.assigned_gpu < 0:
            raise ContractValidationError("assigned_gpu must be non-negative")
        if not self.checkpoint or not self.model_revision:
            raise ContractValidationError("UniDepth checkpoint and model_revision are required server configuration")
        if self.canonical_height <= 0 or self.canonical_width <= 0:
            raise ContractValidationError("canonical HxW must be positive")
        if self.max_concurrent_forwards is not None and self.max_concurrent_forwards < 1:
            raise ContractValidationError("max_concurrent_forwards must be positive or None")
        if self.wire_format not in {"multipart", "envelope"}:
            raise ContractValidationError("wire_format must be multipart or envelope")
        identity_fields = (self.experiment_id, self.application_release_sha, self.checkpoint_digest, self.gcs_address, self.http_port, self.temp_dir)
        if any(value is not None for value in identity_fields) and any(value is None or value == "" for value in identity_fields):
            raise ContractValidationError("experimental UniDepth identity requires experiment/release/checkpoint/GCS/HTTP/temp fields together")

    @property
    def canonical_shape(self) -> tuple[int, int, int]:
        return (self.canonical_height, self.canonical_width, 3)

    def runtime_config_wire(self) -> dict[str, object]:
        """Server-constructed batch/forward policy evidence for experiment responses.

        This is constructed from the resident config after deployment initialization;
        it is never copied from a request or from benchmark-client metadata.
        """
        return {
            "schema": "ego.unidepth-runtime-config.v1",
            "batch_policy": {
                "max_batch_size": self.batch_policy.max_batch_size,
                "batch_wait_timeout_ms": round(self.batch_policy.batch_wait_timeout_s * 1_000.0, 6),
                "max_queued_requests": self.batch_policy.max_queued_requests,
            },
            "max_concurrent_forwards": self.max_concurrent_forwards,
            "canonical_shape": list(self.canonical_shape),
            "wire_format": self.wire_format,
        }

    def runtime_config_digest(self) -> str:
        encoded = json.dumps(self.runtime_config_wire(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


def expected_unidepth_runtime_config(
    *,
    batch_cap: int,
    batch_wait_ms: float,
    max_concurrent_forwards: int | None,
    canonical_height: int = 540,
    canonical_width: int = 960,
    max_queued_requests: int = 64,
    wire_format: str = "multipart",
) -> dict[str, object]:
    """Build the exact runtime-policy attestation expected from an experiment worker."""
    config = UniDepthModelConfig(
        checkpoint="runtime-config-only",
        model_revision="runtime-config-only",
        canonical_height=canonical_height,
        canonical_width=canonical_width,
        batch_policy=BatchPolicy(
            max_batch_size=batch_cap,
            batch_wait_timeout_s=batch_wait_ms / 1_000.0,
            max_queued_requests=max_queued_requests,
        ),
        max_concurrent_forwards=max_concurrent_forwards,
        wire_format=wire_format,
    )
    return {"runtime_config": config.runtime_config_wire(), "runtime_config_digest": config.runtime_config_digest()}


def build_unidepth_model_config(
    *,
    checkpoint: str,
    model_revision: str,
    device: str = "cuda",
    replica_id: str = "unidepth-gpu0",
    assigned_gpu: int = 0,
    experiment_id: str | None = None,
    application_release_sha: str | None = None,
    checkpoint_digest: str | None = None,
    application_release_path: str | None = None,
    gcs_address: str | None = None,
    http_port: int | None = None,
    temp_dir: str | None = None,
    performance_instrumentation: bool = False,
    canonical_height: int = 540,
    canonical_width: int = 960,
    batch_policy: BatchPolicy | None = None,
    max_concurrent_forwards: int | None = None,
    wire_format: str = "multipart",
) -> UniDepthModelConfig:
    """Build a server-owned UniDepth config from explicit kwargs (no request input).

    Used by the deployment module to read environment-provided server settings.
    Defaults reflect the working ``ego_foundation`` ABI and the corrected UniDepth
    checkpoint at ``/home/zjh/ego-annation-checkpoints/unidepth``.
    """
    return UniDepthModelConfig(
        checkpoint=checkpoint,
        model_revision=model_revision,
        device=device,
        replica_id=replica_id,
        assigned_gpu=assigned_gpu,
        experiment_id=experiment_id,
        application_release_sha=application_release_sha,
        checkpoint_digest=checkpoint_digest,
        application_release_path=application_release_path,
        gcs_address=gcs_address,
        http_port=http_port,
        temp_dir=temp_dir,
        performance_instrumentation=performance_instrumentation,
        canonical_height=canonical_height,
        canonical_width=canonical_width,
        batch_policy=batch_policy or BatchPolicy(max_batch_size=8, batch_wait_timeout_s=2.0, max_queued_requests=64),
        max_concurrent_forwards=max_concurrent_forwards,
        wire_format=wire_format,
    )


@dataclass(frozen=True)
class _PreparedRequest:
    request: UniDepthRequest
    rgb: Any


@dataclass(frozen=True)
class UniDepthBackendResult:
    """Backend outputs plus optional experiment-only CUDA/allocator evidence."""

    outputs: Mapping[str, Any]
    diagnostics: Mapping[str, Any] | None = None


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


def decode_rgb(request: UniDepthRequest, resolver: TensorResolver) -> Any:
    """Decode a request to a contiguous uint8 HWC array in [0, 255]."""
    import numpy as np

    rgb = resolver(request.rgb.data, request.rgb.shape, request.rgb.dtype)
    if tuple(rgb.shape) != request.rgb.shape:
        raise ContractValidationError("decoded RGB shape does not match contract")
    if rgb.dtype.name != request.rgb.dtype:
        rgb = rgb.astype(request.rgb.dtype, copy=False)
    if rgb.dtype != np.dtype("uint8"):
        raise ContractValidationError("RGB must be uint8 after resolution")
    if not np.isfinite(rgb).all() or float(rgb.min()) < 0.0 or float(rgb.max()) > 255.0:
        raise ContractValidationError("RGB values must be finite uint8 in [0, 255]")
    return np.ascontiguousarray(rgb)


def stack_to_bchw(rgb_list: Sequence[Any]) -> Any:
    """Stack per-image HWC uint8 into contiguous [B,C,H,W] for upstream ``infer``.

    No division by 255: upstream UniDepth ``model.infer`` owns normalization.
    """
    import numpy as np

    bhwc = np.stack([np.ascontiguousarray(item) for item in rgb_list], axis=0)
    # [B,H,W,C] -> [B,C,H,W], made contiguous for the torch.from_numpy boundary.
    bchw = np.ascontiguousarray(np.transpose(bhwc, (0, 3, 1, 2)))
    return bchw


def squeeze_unidepth_outputs(outputs: Mapping[str, Any], count: int, height: int, width: int) -> tuple[Any, Any, Any]:
    """Squeeze real UniDepth per-batch outputs into documented per-image arrays.

    UniDepth returns ``depth`` as ``[B,1,H,W]``, ``intrinsics`` as ``[B,3,3]``, and
    ``confidence`` as ``[B,1,H,W]``. This validates those batch shapes and squeezes
    the leading channel-1 for depth/confidence so per-image depth/confidence are
    ``[H,W]`` and per-image intrinsics are ``[3,3]``.
    """
    import numpy as np

    if "depth" not in outputs:
        raise ContractValidationError("UniDepth backend did not return required depth")
    if "intrinsics" not in outputs:
        raise ContractValidationError("UniDepth backend did not return required intrinsics")
    if "confidence" not in outputs:
        raise ContractValidationError("UniDepth backend did not return required confidence")

    depth = np.asarray(outputs["depth"])
    intrinsics = np.asarray(outputs["intrinsics"])
    confidence = np.asarray(outputs["confidence"])

    if depth.shape != (count, 1, height, width):
        raise ContractValidationError(
            f"UniDepth backend depth must be [B,1,H,W]=[{count},1,{height},{width}], got {tuple(depth.shape)}"
        )
    if intrinsics.shape != (count, 3, 3):
        raise ContractValidationError(
            f"UniDepth backend intrinsics must be [B,3,3]=[{count},3,3], got {tuple(intrinsics.shape)}"
        )
    if confidence.shape != (count, 1, height, width):
        raise ContractValidationError(
            f"UniDepth backend confidence must be [B,1,H,W]=[{count},1,{height},{width}], got {tuple(confidence.shape)}"
        )
    # Squeeze channel-1: [B,1,H,W] -> [B,H,W]; intrinsics already [B,3,3].
    depth_per_image = depth[:, 0]
    intrinsics_per_image = intrinsics
    confidence_per_image = confidence[:, 0]
    _assert_finite_positive_outputs(depth_per_image, intrinsics_per_image, confidence_per_image, count)
    return depth_per_image, intrinsics_per_image, confidence_per_image


def _assert_finite_positive_outputs(
    depth: Any, intrinsics: Any, confidence: Any, count: int
) -> None:
    """Enforce finite-positive depth/K/validity semantics on backend outputs.

    Depth must be finite and positive (metric depth is strictly positive); intrinsics
    must be finite with positive focal entries (diagonal ``fx``/``fy``); confidence
    must be finite. A backend that returns NaN/Inf or non-positive geometry is a model
    or ABI failure, not usable downstream evidence. We do not require confidence to be
    positive (it can be a signed logit) but it must be finite.
    """
    import numpy as np

    if not np.isfinite(depth).all():
        raise ContractValidationError("UniDepth backend depth must be finite")
    if float(depth.min()) <= 0.0:
        raise ContractValidationError("UniDepth backend depth must be strictly positive (metric)")
    if not np.isfinite(intrinsics).all():
        raise ContractValidationError("UniDepth backend intrinsics must be finite")
    # The focal diagonal (indices [0,0] and [1,1]) must be positive for a valid pinhole.
    for index in range(count):
        fx = float(intrinsics[index, 0, 0])
        fy = float(intrinsics[index, 1, 1])
        if fx <= 0.0 or fy <= 0.0:
            raise ContractValidationError(
                f"UniDepth backend intrinsics must have positive fx/fy; got fx={fx}, fy={fy}"
            )
    if not np.isfinite(confidence).all():
        raise ContractValidationError("UniDepth backend confidence must be finite")


def _as_tensor_payload(value: Any) -> TensorPayload:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    return TensorPayload(data=array.tobytes(), shape=tuple(int(dim) for dim in array.shape), dtype=array.dtype.name)


def _load_unidepth_backend(config: UniDepthModelConfig) -> UniDepthBackend:
    """Load the real upstream UniDepth model once inside the assigned Serve replica.

    Uses the known-working local checkpoint loader rather than an unverified
    ``from_pretrained`` download/path fallback:

    1. Read ``config.json`` from the checkpoint directory.
    2. Construct ``UniDepthV2(config)`` directly.
    3. Load ``model.safetensors`` with ``safetensors.load_file``.
    4. ``load_state_dict(strict=False)`` (the corrected checkpoint is byte-aligned but
       the mixin's extra buffers are not required for inference).
    5. Move once to GPU and reuse the resident instance across all requests.

    The model repo path is added to ``sys.path`` so ``from unidepth.models import
    UniDepthV2`` resolves the server-owned source mirror. This factory does NOT divide
    RGB by 255; upstream ``model.infer`` does it itself (``normalize=True``).
    """
    import json
    import os
    import sys

    import numpy as np
    import torch
    from safetensors import safe_open

    repo_path = os.environ.get("EGO_UNIDEPTH_REPO", "/home/zjh/ego-annation-checkpoints/unidepth_repo")
    checkpoint = os.environ.get("EGO_UNIDEPTH_CHECKPOINT", config.checkpoint)
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    # UniDepth's ``unidepth/utils/visualization.py`` imports ``wandb`` at module top,
    # but wandb's compiled protos require protobuf >=5 while this serving env pins
    # protobuf 3.20.3 for Ray Serve's ``_proto_to_dict`` (``FieldDescriptor.label``).
    # The inference path never uses wandb (only training logging does), so stub it
    # before importing UniDepth to decouple the model boundary from the logging ABI.
    import types
    if "wandb" not in sys.modules:
        _wandb_stub = types.ModuleType("wandb")
        _wandb_stub.log = lambda *a, **k: None
        _wandb_stub.Image = lambda *a, **k: None
        _wandb_stub.init = lambda *a, **k: _wandb_stub
        sys.modules["wandb"] = _wandb_stub
    from unidepth.models import UniDepthV2  # type: ignore[import-not-found]

    config_path = os.path.join(checkpoint, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"UniDepth checkpoint missing config.json at {config_path}")
    with open(config_path) as handle:
        model_config = json.load(handle)

    model = UniDepthV2(model_config)

    weights_path = os.path.join(checkpoint, "model.safetensors")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"UniDepth checkpoint missing model.safetensors at {weights_path}")
    state_dict: dict[str, Any] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    info = model.load_state_dict(state_dict, strict=False)
    del state_dict

    model = model.to(config.device).eval()

    class TorchUniDepthBackend:
        def infer(self, rgb: Any) -> Mapping[str, Any] | UniDepthBackendResult:
            # rgb is contiguous [B,C,H,W] uint8; no /255 here.  CUDA events and
            # synchronization are opt-in because their synchronization boundary
            # intentionally perturbs production latency.
            if not config.performance_instrumentation:
                batch = torch.from_numpy(np.ascontiguousarray(rgb)).to(config.device)
                with torch.inference_mode():
                    output = model.infer(batch)
                return {
                    "depth": output["depth"].detach().cpu().numpy(),
                    "intrinsics": output["intrinsics"].detach().cpu().numpy(),
                    "confidence": output["confidence"].detach().cpu().numpy(),
                }

            diagnostics: dict[str, Any] = {
                "schema": "ego.unidepth-batch-diagnostics.v1",
                "availability": "unavailable_cpu",
                "cpu_collate_ms": None,
                "h2d_ms": None,
                "cuda_model_ms": None,
                "d2h_ms": None,
                "validation_ms": None,
                "adapter_result_construction_ms": None,
                "http_serialization_ms": None,
                "encoding_ms": None,
                "allocator_memory": {
                    "allocated_bytes": None,
                    "reserved_bytes": None,
                    "max_allocated_bytes": None,
                    "max_reserved_bytes": None,
                },
                "batch_work_units": int(rgb.shape[0]),
            }
            if not config.device.startswith("cuda") or not torch.cuda.is_available():
                batch = torch.from_numpy(np.ascontiguousarray(rgb)).to(config.device)
                with torch.inference_mode():
                    output = model.infer(batch)
                return UniDepthBackendResult({
                    "depth": output["depth"].detach().cpu().numpy(),
                    "intrinsics": output["intrinsics"].detach().cpu().numpy(),
                    "confidence": output["confidence"].detach().cpu().numpy(),
                }, diagnostics)

            h2d_start, h2d_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            model_start, model_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            h2d_start.record()
            batch = torch.from_numpy(np.ascontiguousarray(rgb)).to(config.device)
            h2d_end.record()
            model_start.record()
            with torch.inference_mode():
                output = model.infer(batch)
            model_end.record()
            # D2H values must wait for the copies they describe.  This explicit
            # synchronization is confined to experiment telemetry.
            d2h_started = time.perf_counter()
            depth = output["depth"].detach().cpu().numpy()
            intrinsics = output["intrinsics"].detach().cpu().numpy()
            confidence = output["confidence"].detach().cpu().numpy()
            torch.cuda.synchronize()
            diagnostics.update({
                "availability": "cuda",
                "h2d_ms": h2d_start.elapsed_time(h2d_end),
                "cuda_model_ms": model_start.elapsed_time(model_end),
                "d2h_ms": (time.perf_counter() - d2h_started) * 1000.0,
                "allocator_memory": {
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                    "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "max_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
            })
            return UniDepthBackendResult(
                {"depth": depth, "intrinsics": intrinsics, "confidence": confidence}, diagnostics
            )

    return TorchUniDepthBackend()


class UniDepthAdapter:
    """Resident model owner used by a single Ray Serve GPU replica."""

    def __init__(
        self,
        config: UniDepthModelConfig,
        *,
        backend_factory: BackendFactory = _load_unidepth_backend,
        tensor_resolver: TensorResolver = _default_tensor_resolver,
    ) -> None:
        self._config = config
        self._tensor_resolver = tensor_resolver
        self._backend = backend_factory(config)
        self._model_load_count = 1
        self._running_batches = 0
        self._batch_snapshot = BatchSnapshotTracker(
            service="unidepth.infer", replica_id=config.replica_id,
            capacity={"kind": "serve_callback_batch", "max_batch_size": config.batch_policy.max_batch_size, "max_ongoing_requests": 16},
        )
        self._running_forwards = 0
        self._peak_simultaneous_forwards = 0
        self._forward_semaphore = (
            asyncio.Semaphore(config.max_concurrent_forwards)
            if config.max_concurrent_forwards is not None
            else None
        )
        self._admitted_pending = 0
        # Per-request admission monotonic time, keyed by request_id. The earliest
        # admission across a callback becomes the batch's admitted_monotonic_s.
        self._admitted_at: dict[str, float] = {}
        self._worker_pid = os.getpid()
        self._release_digest = None
        self._actual_checkpoint_digest = None
        self._cuda_uuid = None
        if self._config.experiment_id is not None:
            # Experiment identity is derived from bytes and the loaded process, not
            # from EGO_* values echoed back by the caller.
            from ego_annotation.serving.benchmark.release import artifact_digest, verify_release
            if self._config.application_release_path:
                self._release_digest = verify_release(self._config.application_release_path).release_digest
            self._actual_checkpoint_digest = artifact_digest(self._config.checkpoint)
            try:
                import torch
                if torch.cuda.is_available():
                    self._cuda_uuid = str(torch.cuda.get_device_properties(0).uuid)
            except Exception:
                self._cuda_uuid = None

    @property
    def config(self) -> UniDepthModelConfig:
        return self._config

    def admit(self, request: UniDepthRequest) -> _PreparedRequest:
        """Validate and decode a request against the resident config before admission.

        Returns a prepared request carrying the decoded contiguous uint8 HWC array
        and the admission monotonic timestamp. Raises ``ContractValidationError``
        for revision mismatch, shape/dtype incompatibility with the canonical
        bucket, an overweight item, or a binary that fails to decode to the declared
        shape. The caller (Serve ``infer``) must call this before the request enters
        the batch callback, so a callback never admits an incompatible or undecodable
        item that would force splitting into several forwards.
        """
        if request.model_revision != self._config.model_revision:
            raise ContractValidationError(
                f"request model_revision {request.model_revision!r} does not match resident "
                f"revision {self._config.model_revision!r}"
            )
        if request.rgb.shape != self._config.canonical_shape:
            raise ContractValidationError(
                f"request RGB shape {request.rgb.shape} is incompatible with the canonical "
                f"bucket {self._config.canonical_shape}"
            )
        if request.rgb.dtype != "uint8":
            raise ContractValidationError("request RGB dtype must be uint8")
        if request.work_units != 1:
            raise ContractValidationError("each UniDepth request must be exactly one canonical work unit")
        rgb = decode_rgb(request, self._tensor_resolver)
        admitted_at = time.monotonic()
        self._admitted_at[request.ownership.request_id] = admitted_at
        self._admitted_pending += 1
        return _PreparedRequest(request=request, rgb=rgb)

    def request_dispatched(self, request_id: str) -> None:
        self._admitted_pending = max(0, self._admitted_pending - 1)
        self._admitted_at.pop(request_id, None)

    def _server_identity(self) -> ServerIdentity | None:
        config = self._config
        if config.experiment_id is None:
            return None
        assert config.application_release_sha is not None
        assert config.checkpoint_digest is not None
        assert config.gcs_address is not None
        assert config.http_port is not None
        assert config.temp_dir is not None
        return ServerIdentity(
            experiment_id=config.experiment_id,
            replica_id=config.replica_id,
            assigned_gpu=config.assigned_gpu,
            worker_pid=self._worker_pid,
            gcs_address=config.gcs_address,
            http_port=config.http_port,
            temp_dir=config.temp_dir,
            model_revision=config.model_revision,
            checkpoint_digest=self._actual_checkpoint_digest or config.checkpoint_digest,
            schema_version=SCHEMA_VERSION,
            release_sha=self._release_digest or config.application_release_sha,
            release_digest=self._release_digest,
            cuda_uuid=self._cuda_uuid,
        )

    async def infer(self, request: UniDepthRequest) -> UniDepthResponse:
        prepared = self.admit(request)
        return (await self.infer_batch([prepared]))[0]

    async def infer_batch(self, requests: Sequence[_PreparedRequest]) -> list[UniDepthResponse]:
        """Run exactly one upstream forward for one Serve batch callback.

        All items were already admitted (revision/shape/dtype/weight validated and
        binary decoded) before the callback. They are mutually compatible by
        construction, so this method performs one ``backend.infer`` and never splits.
        """
        import numpy as np  # noqa: F401  (used transitively by stack/squeeze)

        if not requests:
            return []
        assert_one_forward(requests, policy=self._config.batch_policy)

        batch_id = uuid4().hex
        dispatched_monotonic_s = time.monotonic()
        admitted_monotonic_s = min(
            (self._admitted_at.get(item.request.ownership.request_id, dispatched_monotonic_s) for item in requests),
            default=dispatched_monotonic_s,
        )
        for item in requests:
            self.request_dispatched(item.request.ownership.request_id)

        self._running_batches += 1
        forward_started_monotonic_s = time.monotonic()
        diagnostics: dict[str, Any] | None = None
        try:
            collate_started = time.perf_counter()
            bchw = stack_to_bchw([item.rgb for item in requests])
            collate_ms = (time.perf_counter() - collate_started) * 1000.0
            # Batch callbacks may overlap in Serve.  The treatment semaphore bounds
            # only the physical backend forward, leaving request admission and Serve
            # batch formation fully concurrent.
            forward_wait_started = time.perf_counter()
            if self._forward_semaphore is not None:
                await self._forward_semaphore.acquire()
            forward_started_monotonic_s = time.monotonic()
            forward_wait_ms = (time.perf_counter() - forward_wait_started) * 1000.0
            self._running_forwards += 1
            self._peak_simultaneous_forwards = max(self._peak_simultaneous_forwards, self._running_forwards)

            def release_forward_slot() -> None:
                self._running_forwards -= 1
                if self._forward_semaphore is not None:
                    self._forward_semaphore.release()

            backend_result = await await_tracked_thread(
                self._batch_snapshot,
                ExecutionSpec(
                    operation="infer", execution_kind="fused_callback", request_count=len(requests),
                    item_count=len(requests), item_kind="rgb_image", image_item_count=len(requests),
                    forward_count=1, max_batch_size=self._config.batch_policy.max_batch_size,
                ),
                self._backend.infer, bchw,
                on_worker_terminal=release_forward_slot,
            )
            if isinstance(backend_result, UniDepthBackendResult):
                outputs = backend_result.outputs
                diagnostics = dict(backend_result.diagnostics or {})
            else:
                outputs = backend_result
            if self._config.performance_instrumentation:
                diagnostics = diagnostics or {
                    "schema": "ego.unidepth-batch-diagnostics.v1",
                    "availability": "unavailable_cpu" if not self._config.device.startswith("cuda") else "unavailable_backend",
                    "h2d_ms": None,
                    "cuda_model_ms": None,
                    "d2h_ms": None,
                    "allocator_memory": {
                        "allocated_bytes": None, "reserved_bytes": None,
                        "max_allocated_bytes": None, "max_reserved_bytes": None,
                    },
                }
                diagnostics["cpu_collate_ms"] = collate_ms
                diagnostics["batch_work_units"] = len(requests)
                diagnostics["adapter_forward_wait_ms"] = forward_wait_ms
                diagnostics["peak_simultaneous_forwards"] = self._peak_simultaneous_forwards
                diagnostics["runtime_config"] = self._config.runtime_config_wire()
                diagnostics["runtime_config_digest"] = self._config.runtime_config_digest()
            validation_started = time.perf_counter()
            height = self._config.canonical_height
            width = self._config.canonical_width
            depth, intrinsics, confidence = squeeze_unidepth_outputs(outputs, len(requests), height, width)
            if diagnostics is not None:
                diagnostics["validation_ms"] = (time.perf_counter() - validation_started) * 1000.0
        except Exception as exc:
            return [
                UniDepthResponse(
                    ownership=item.request.ownership,
                    error=ServiceError(
                        ErrorCode.MODEL_FAILURE, str(exc), retryable=False, ownership=item.request.ownership, batch_id=batch_id
                    ),
                )
                for item in requests
            ]
        finally:
            self._running_batches -= 1
        completed_monotonic_s = time.monotonic()

        trace = BatchTrace(
            batch_id=batch_id,
            replica_id=self._config.replica_id,
            admitted_monotonic_s=admitted_monotonic_s,
            dispatched_monotonic_s=dispatched_monotonic_s,
            forward_started_monotonic_s=forward_started_monotonic_s,
            completed_monotonic_s=completed_monotonic_s,
            effective_work_units=len(requests),
            request_count=len(requests),
            request_ids=tuple(item.request.ownership.request_id for item in requests),
            forward_count=1,
            model_load_count=self._model_load_count,
            served_at_wall_unix_s=time.time(),
        )
        responses: list[UniDepthResponse] = []
        encoding_started = time.perf_counter()
        server_identity = self._server_identity()
        for index, item in enumerate(requests):
            try:
                responses.append(
                    UniDepthResponse(
                        ownership=item.request.ownership,
                        result=UniDepthResult(
                            ownership=item.request.ownership,
                            depth_m=_as_tensor_payload(depth[index]),
                            K_px=_as_tensor_payload(intrinsics[index]),
                            confidence=_as_tensor_payload(confidence[index]),
                            spatial=item.request.spatial,
                            # Results carry the resident configured revision only.
                            model_revision=self._config.model_revision,
                            trace=trace,
                            batch_diagnostics=diagnostics,
                            server_identity=server_identity,
                        ),
                    )
                )
            except Exception as exc:
                responses.append(
                    UniDepthResponse(
                        ownership=item.request.ownership,
                        error=ServiceError(
                            ErrorCode.RESULT_SPLIT_FAILURE,
                            str(exc),
                            retryable=False,
                            ownership=item.request.ownership,
                            batch_id=batch_id,
                        ),
                    )
                )
        if diagnostics is not None:
            diagnostics["adapter_result_construction_ms"] = (time.perf_counter() - encoding_started) * 1000.0
            diagnostics["http_serialization_ms"] = None
            # Compatibility field only; benchmark extraction deliberately ignores
            # it as HTTP serialization and uses the labeled boundary below.
            diagnostics["encoding_ms"] = diagnostics["adapter_result_construction_ms"]
            diagnostics["timing_boundaries"] = {
                "adapter_result_construction_ms": "tensor payload and typed result construction inside adapter",
                "http_serialization_ms": "unavailable; ASGI multipart construction occurs after adapter return",
                "encoding_ms": "legacy alias for adapter_result_construction_ms; never HTTP serialization",
            }
            # Responses already reference the same mapping; copy it into each result
            # after all encoding work has completed so the value is truthful.
            for response in responses:
                if response.result is not None:
                    object.__setattr__(response.result, "batch_diagnostics", dict(diagnostics))
        return responses

    def batch_snapshot(self) -> dict[str, Any]:
        return self._batch_snapshot.snapshot(adapter_status=self.status().to_wire())

    def status(self) -> DeploymentStatus:
        return DeploymentStatus(
            deployment_name="unidepth.infer",
            replica_id=self._config.replica_id,
            assigned_gpu=self._config.assigned_gpu,
            loaded_models=(self._config.model_revision,),
            admitted_pending=self._admitted_pending,
            running_batches=self._running_batches,
            model_load_count=self._model_load_count,
        )
