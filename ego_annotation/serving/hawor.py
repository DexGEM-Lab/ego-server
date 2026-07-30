"""Ray-free resident HaWoR track-chunk adapter: one model forward per Serve batch.

This module is importable without Ray installed (CPU-only unit tests import
``HaWoRAdapter`` directly with an injected fake backend). The Ray Serve deployment
wrapper lives in ``hawor_deployment.py``.

Contract invariants enforced here:

* The resident model is the real upstream ``HAWOR`` checkpoint loaded once via
  ``HAWOR.load_from_checkpoint``. The API never reruns DROID or Metric3D/UniDepth:
  the canonical masked-DROID camera trajectory and the UniDepth metric scale/K
  arrive as typed inputs (``DroidCameraEvidence`` / ``UniDepthScaleK``).
* Each request is one 16-frame hand-crop chunk. The crop batch is the normalized
  float32 ``[16,3,256,256]`` BCHW tensor that ``TrackDatasetEval`` would produce
  (ToTensor + ImageNet Normalize); the per-frame crop/source transforms (center,
  scale, img_focal, img_center, do_flip) travel alongside because HaWoR's
  ``bbox_est`` (CLIFF feature) and ``get_trans`` (metric camera translation) require
  them. The API does NOT re-crop or re-read images.
* One Serve batch callback is exactly one ``HAWOR.forward`` over the stacked
  ``[B,16,3,256,256]`` batch. ``infer_batch`` never splits a callback into several
  forwards; incompatible items are rejected at admission.
* Output is observed camera-space metric MANO: ``root_orient`` [16,3,3],
  ``hand_pose`` [16,15,3,3], ``trans`` [16,3] (metres), ``betas`` [16,10],
  ``vertices`` [16,778,3], ``joints`` [16,16,3], per-frame ``observed`` /
  ``occlusion_state`` / ``uncertainty``. A CPU world-lift fusion produces the
  world-space per-frame SE(3) when DROID evidence is provided; otherwise the world
  lift is explicitly ``unavailable`` (never silently invented).
* Batch timings are truthful monotonic ``time.monotonic()`` readings.
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

from ego_annotation.serving.batching import BatchPolicy, assert_one_forward
from ego_annotation.serving.contracts import (
    BatchTrace,
    ContractValidationError,
    DeploymentStatus,
    ErrorCode,
    ImageSize,
    PixelTransform,
    ServiceError,
    SpatialMetadata,
    TensorPayload,
    SCHEMA_VERSION,
    ServerIdentity,
)
from ego_annotation.serving.hawor_contracts import (
    HAWOR_CHUNK_LEN,
    HAWOR_CROP_CHANNELS,
    HAWOR_CROP_H,
    HAWOR_CROP_W,
    HAWOR_CROP_DTYPE,
    HAWOR_HAND_JOINTS,
    HAWOR_MANO_VERTS,
    CameraSpaceManoResult,
    CropSourceTransform,
    DroidCameraEvidence,
    FrameObservation,
    HandSide,
    OcclusionState,
    TrackChunkRequest,
    UniDepthScaleK,
)


class HaWoRBackend(Protocol):
    """The model boundary.

    ``infer_tracks`` receives a stacked ``[B,16,3,256,256]`` normalized float32 BCHW
    crop batch and per-frame ``[B,16,4]`` crop/source geometry (center_x, center_y,
    scale, img_focal) plus ``[B,16,2]`` img_center and ``[B]`` do_flip flags. It runs
    exactly one ``HAWOR.forward`` per call and returns per-frame camera-space MANO.
    """

    def infer_tracks(
        self,
        crop_batch: Any,
        crop_geometry: Any,
        img_center: Any,
        do_flip: Any,
    ) -> Mapping[str, Any]: ...


TensorResolver = Callable[[Any, tuple[int, ...], str], Any]
BackendFactory = Callable[["HaWoRModelConfig"], HaWoRBackend]


@dataclass(frozen=True)
class HaWoRModelConfig:
    checkpoint: str
    model_revision: str
    device: str = "cuda"
    replica_id: str = "hawor-gpu3"
    assigned_gpu: int = 3
    batch_policy: BatchPolicy = BatchPolicy(
        max_batch_size=8,
        batch_wait_timeout_s=0.1,
        max_queued_requests=64,
    )
    performance_instrumentation: bool = False
    wire_format: str = "multipart"
    experiment_id: str | None = None
    application_release_sha: str | None = None
    checkpoint_digest: str | None = None
    application_release_path: str | None = None
    gcs_address: str | None = None
    http_port: int | None = None
    temp_dir: str | None = None

    def __post_init__(self) -> None:
        if self.assigned_gpu < 0:
            raise ContractValidationError("assigned_gpu must be non-negative")
        if not self.checkpoint or not self.model_revision:
            raise ContractValidationError("HaWoR checkpoint and model_revision are required")
        if self.wire_format not in {"multipart", "envelope"}:
            raise ContractValidationError("wire_format must be multipart or envelope")

    def runtime_config_wire(self) -> dict[str, object]:
        return {
            "schema": "ego.hawor-runtime-config.v1",
            "batch_policy": {
                "max_batch_size": self.batch_policy.max_batch_size,
                "batch_wait_timeout_ms": round(self.batch_policy.batch_wait_timeout_s * 1_000.0, 6),
                "max_queued_requests": self.batch_policy.max_queued_requests,
            },
            "chunk_shape": [HAWOR_CHUNK_LEN, HAWOR_CROP_CHANNELS, HAWOR_CROP_H, HAWOR_CROP_W],
            "wire_format": self.wire_format,
        }

    def runtime_config_digest(self) -> str:
        raw = json.dumps(self.runtime_config_wire(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(raw).hexdigest()


def expected_hawor_runtime_config(
    *, batch_cap: int = 8, batch_wait_ms: float = 100.0, max_queued_requests: int = 64, wire_format: str = "multipart",
) -> dict[str, object]:
    config = HaWoRModelConfig(
        checkpoint="runtime-config-only", model_revision="runtime-config-only",
        batch_policy=BatchPolicy(max_batch_size=batch_cap, batch_wait_timeout_s=batch_wait_ms / 1_000.0, max_queued_requests=max_queued_requests),
        wire_format=wire_format,
    )
    return {"runtime_config": config.runtime_config_wire(), "runtime_config_digest": config.runtime_config_digest()}


def build_hawor_model_config(
    *,
    checkpoint: str,
    model_revision: str,
    device: str = "cuda",
    replica_id: str = "hawor-gpu3",
    assigned_gpu: int = 3,
    batch_policy: BatchPolicy | None = None,
    performance_instrumentation: bool = False,
    wire_format: str = "multipart",
    experiment_id: str | None = None,
    application_release_sha: str | None = None,
    checkpoint_digest: str | None = None,
    application_release_path: str | None = None,
    gcs_address: str | None = None,
    http_port: int | None = None,
    temp_dir: str | None = None,
) -> HaWoRModelConfig:
    return HaWoRModelConfig(
        checkpoint=checkpoint,
        model_revision=model_revision,
        device=device,
        replica_id=replica_id,
        assigned_gpu=assigned_gpu,
        batch_policy=batch_policy or BatchPolicy(max_batch_size=8, batch_wait_timeout_s=0.1, max_queued_requests=64),
        performance_instrumentation=performance_instrumentation,
        wire_format=wire_format,
        experiment_id=experiment_id,
        application_release_sha=application_release_sha,
        checkpoint_digest=checkpoint_digest,
        application_release_path=application_release_path,
        gcs_address=gcs_address,
        http_port=http_port,
        temp_dir=temp_dir,
    )


@dataclass(frozen=True)
class _PreparedTrackChunk:
    request: TrackChunkRequest
    crop_batch: Any            # [16,3,256,256] float32 BCHW
    crop_geometry: Any         # [16,4] float32 (cx, cy, scale, img_focal)
    img_center: Any            # [16,2] float32
    do_flip: bool


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


# ImageNet normalization constants (must match TrackDatasetEval.normalize_img /
# lib.core.constants.IMG_NORM_MEAN/STD) so callers can verify crop normalization.
IMG_NORM_MEAN = (0.485, 0.456, 0.406)
IMG_NORM_STD = (0.229, 0.224, 0.225)


def decode_crop_batch(request: TrackChunkRequest, resolver: TensorResolver) -> Any:
    """Decode a request crop batch into a contiguous ``[16,3,256,256]`` float32 array."""
    import numpy as np

    crops = resolver(request.crop_batch.data, request.crop_batch.shape, request.crop_batch.dtype)
    crops = np.asarray(crops)
    if tuple(crops.shape) != request.crop_batch.shape:
        raise ContractValidationError("decoded crop batch shape does not match contract")
    if crops.dtype != np.dtype(HAWOR_CROP_DTYPE):
        crops = crops.astype(np.dtype(HAWOR_CROP_DTYPE), copy=False)
    if not np.isfinite(crops).all():
        raise ContractValidationError("crop batch must be finite float32")
    return np.ascontiguousarray(crops)


def build_crop_geometry(request: TrackChunkRequest) -> tuple[Any, Any, bool]:
    """Stack per-frame crop/source transforms into the geometry HaWoR requires.

    Returns ``(crop_geometry [16,4], img_center [16,2], do_flip)``. ``do_flip`` is
    taken from the first frame's transform (HaWoR resolves one side per chunk).
    """
    import numpy as np

    geom = np.zeros((HAWOR_CHUNK_LEN, 4), dtype=np.float32)
    ctr = np.zeros((HAWOR_CHUNK_LEN, 2), dtype=np.float32)
    for i, t in enumerate(request.crop_transforms):
        geom[i, 0] = float(t.center[0])
        geom[i, 1] = float(t.center[1])
        geom[i, 2] = float(t.scale)
        geom[i, 3] = float(t.img_focal)
        ctr[i, 0] = float(t.img_center[0])
        ctr[i, 1] = float(t.img_center[1])
    do_flip = bool(request.crop_transforms[0].do_flip)
    return geom, ctr, do_flip


def _as_tensor_payload(value: Any) -> TensorPayload:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    return TensorPayload(data=array.tobytes(), shape=tuple(int(d) for d in array.shape), dtype=array.dtype.name)


def _load_hawor_backend(config: HaWoRModelConfig) -> HaWoRBackend:
    """Load the real upstream HaWoR model once inside the assigned Serve replica.

    The working HaWoR ABI is the ``ray_serve_hawor`` environment (Python 3.10,
    Torch 1.13.0+cu117, Ray 2.55.1). The model repo is server-owned. This factory
    constructs the real ``HAWOR`` checkpoint and runs the real ``forward``; it does
    NOT run DROID, Metric3D, UniDepth, detection, or tracking.
    """
    import sys

    import numpy as np
    import torch

    repo_path = os.environ.get("EGO_HAWOR_REPO", "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/HaWoR")
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    # The MANO data dir lives inside the repo.
    os.chdir(repo_path)

    from scripts.scripts_test_video.hawor_video import load_hawor  # type: ignore[import-not-found]

    model, model_cfg = load_hawor(config.checkpoint)
    model = model.to(config.device)
    model.eval()

    class TorchHaWoRBackend:
        def infer_tracks(self, crop_batch: Any, crop_geometry: Any, img_center: Any, do_flip: Any) -> Mapping[str, Any]:
            # crop_batch: [B,16,3,256,256] float32 BCHW normalized
            B = crop_batch.shape[0]
            crops_t = torch.from_numpy(np.ascontiguousarray(crop_batch)).to(config.device)
            geom_t = torch.from_numpy(np.ascontiguousarray(crop_geometry)).to(config.device)  # [B,16,4]
            ctr_t = torch.from_numpy(np.ascontiguousarray(img_center)).to(config.device)  # [B,16,2]
            flip_flags = np.asarray(do_flip).reshape(-1)
            batch = _hawor_forward_batch(crops_t, geom_t, ctr_t, do_flip)
            if bool(flip_flags[0]):
                batch["do_flip"] = torch.tensor([1.0] * B, device=config.device)
            with torch.inference_mode():
                output = model.forward(batch)
                out = output["out"]
            # out['pred_rotmat']: [B*16, 16, 3, 3]; out['trans_full']: [B*16, 1, 3]
            # out['pred_shape']: [B*16, 10]; pred_keypoints_3d: [B*16, 16, 3]
            pred_rotmat = out["pred_rotmat"].reshape(B, HAWOR_CHUNK_LEN, 16, 3, 3).cpu().numpy()
            trans = out["trans_full"].reshape(B, HAWOR_CHUNK_LEN, 3).cpu().numpy()
            betas = out["pred_shape"].reshape(B, HAWOR_CHUNK_LEN, 10).cpu().numpy()
            j3d = output["pred_keypoints_3d"].reshape(B, HAWOR_CHUNK_LEN, -1, 3).cpu().numpy()
            vertices = output["pred_vertices"].reshape(B, HAWOR_CHUNK_LEN, HAWOR_MANO_VERTS, 3).cpu().numpy()
            return {
                "pred_rotmat": pred_rotmat,
                "trans": trans,
                "betas": betas,
                "vertices": vertices,
                "joints": j3d,
            }

    return TorchHaWoRBackend()


def _hawor_forward_batch(crops_t: Any, geom_t: Any, ctr_t: Any, do_flip: Any) -> dict[str, Any]:
    """Pack the HAWOR.forward batch dict with upstream-required shapes.

    Upstream ``bbox_est`` (lib/models/hawor.py) requires center [N,2], scale [N],
    img_focal [N]: ``cx, cy, b = center[:, 0], center[:, 1], scale * 200`` followed
    by ``torch.stack([cx - img_cx, cy - img_cy, b], dim=-1)``. A trailing [N,1]
    breaks that stack, so the scalar columns are sliced, never kept as [..., 1].
    """
    return {
        "img": crops_t,
        "center": geom_t[..., :2],
        "scale": geom_t[..., 2],
        "img_focal": geom_t[..., 3],
        "img_center": ctr_t,
    }


def _occlusion_to_observed(state: OcclusionState) -> bool:
    return state in (OcclusionState.VISIBLE, OcclusionState.PARTIALLY_VISIBLE)


def _uncertainty_for(state: OcclusionState, confidence: float) -> float:
    """Per-frame wrist-radius-equivalent uncertainty in metres.

    Visible high-confidence observations carry a base uncertainty (~5mm target
    floor). Partially-visible / low-confidence raise it; occluded / out-of-frame /
    unresolved frames are carried as padding and marked fully uncertain.
    """
    if state == OcclusionState.VISIBLE:
        return 0.005 + 0.010 * (1.0 - max(0.0, min(1.0, confidence)))
    if state == OcclusionState.PARTIALLY_VISIBLE:
        return 0.020 + 0.020 * (1.0 - max(0.0, min(1.0, confidence)))
    return 0.080  # occluded / out-of-frame / unresolved: padding, fully uncertain


def _world_lift(
    droid: DroidCameraEvidence,
    trans_cam: Any,
    root_orient_cam: Any,
    timestamps_s: Any,
) -> Any:
    """CPU world-lift fusion: per-frame world-from-camera SE(3) from DROID evidence.

    This is the explicit CPU fusion that consumes the typed DROID trajectory. It
    resamples the DROID world-from-camera poses to the HaWoR chunk source timestamps
    by nearest-neighbour on source time (DROID keyframe timestamps join by source
    time, not frame index). It returns ``[16,4,4]`` world-from-camera SE(3); the
    caller does not silently invent a world frame. The MANO camera->world transform
    itself is left to the downstream CPU fusion module (section 4 step 7) per the
    task spec; here we expose the verified metric camera trajectory resampled to
    HaWoR timestamps so the world lift is reproducible and auditable.
    """
    import numpy as np

    poses = droid.poses_world_from_camera  # binary
    # poses.data may be bytes or an in-cluster array; the adapter resolver decoded
    # the request tensors already, but droid_evidence arrives as a typed input —
    # decode here with numpy.
    if isinstance(poses.data, (bytes, bytearray, memoryview)):
        poses_arr = np.frombuffer(poses.data, dtype=np.dtype(poses.dtype)).reshape(poses.shape)
    else:
        poses_arr = np.asarray(poses.data, dtype=poses.dtype).reshape(poses.shape)
    if isinstance(droid.timestamps_s.data, (bytes, bytearray, memoryview)):
        ts_arr = np.frombuffer(droid.timestamps_s.data, dtype=np.dtype(droid.timestamps_s.dtype)).reshape(
            droid.timestamps_s.shape
        )
    else:
        ts_arr = np.asarray(droid.timestamps_s.data, dtype=droid.timestamps_s.dtype).reshape(
            droid.timestamps_s.shape
        )
    ts_arr = ts_arr.astype(np.float64)
    chunk_ts = np.asarray(timestamps_s, dtype=np.float64)
    # nearest-neighbour resample on source time
    idx = np.array([int(np.argmin(np.abs(ts_arr - t))) for t in chunk_ts], dtype=np.int64)
    lifted = poses_arr[idx].astype(np.float32)  # [16,4,4]
    return lifted


class HaWoRAdapter:
    """Resident model owner used by the single Ray Serve GPU3 HaWoR replica."""

    def __init__(
        self,
        config: HaWoRModelConfig,
        *,
        backend_factory: BackendFactory = _load_hawor_backend,
        tensor_resolver: TensorResolver = _default_tensor_resolver,
        runtime_evidence_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config
        self._tensor_resolver = tensor_resolver
        self._backend = backend_factory(config)
        self._model_load_count = 1
        self._running_batches = 0
        self._admitted_pending = 0
        self._admitted_at: dict[str, float] = {}
        self._server_runtime_identity: ServerIdentity | None = None
        if config.experiment_id is not None:
            release_root, gcs_address, temp_dir, http_port = (
                config.application_release_path, config.gcs_address, config.temp_dir, config.http_port,
            )
            if release_root is None or gcs_address is None or temp_dir is None or http_port is None:
                raise RuntimeError("HaWoR experiment identity requires release root, GCS address, HTTP port, and temp dir")
            from ego_annotation.serving.benchmark.release import derive_worker_runtime_evidence

            derive = runtime_evidence_factory or derive_worker_runtime_evidence
            evidence = derive(
                release_root=release_root,
                checkpoint_path=config.checkpoint,
                imported_module_file=__file__,
            )
            if evidence.physical_gpu != config.assigned_gpu:
                raise RuntimeError(
                    f"HaWoR worker physical GPU {evidence.physical_gpu} differs from planned GPU {config.assigned_gpu}"
                )
            self._server_runtime_identity = ServerIdentity(
                experiment_id=config.experiment_id,
                replica_id=config.replica_id,
                assigned_gpu=evidence.physical_gpu,
                worker_pid=evidence.worker_pid,
                gcs_address=gcs_address,
                http_port=http_port,
                temp_dir=temp_dir,
                model_revision=config.model_revision,
                checkpoint_digest=evidence.checkpoint_digest,
                schema_version=SCHEMA_VERSION,
                release_sha=evidence.source_sha,
                release_digest=evidence.release_digest,
                cuda_uuid=evidence.cuda_uuid,
                module_root=str(evidence.module_root),
            )

    @property
    def server_identity(self) -> ServerIdentity | None:
        return self._server_runtime_identity

    @property
    def config(self) -> HaWoRModelConfig:
        return self._config

    def admit(self, request: TrackChunkRequest) -> _PreparedTrackChunk:
        if request.model_revision != self._config.model_revision:
            raise ContractValidationError(
                f"request model_revision {request.model_revision!r} does not match resident "
                f"revision {self._config.model_revision!r}"
            )
        if request.crop_batch.shape != (HAWOR_CHUNK_LEN, HAWOR_CROP_CHANNELS, HAWOR_CROP_H, HAWOR_CROP_W):
            raise ContractValidationError("request crop_batch shape is incompatible with the 16-frame bucket")
        if request.crop_batch.dtype != HAWOR_CROP_DTYPE:
            raise ContractValidationError(f"request crop_batch dtype must be {HAWOR_CROP_DTYPE}")
        if request.work_units != 1:
            raise ContractValidationError("each HaWoR track chunk must be exactly one work unit")
        crop_batch = decode_crop_batch(request, self._tensor_resolver)
        geom, ctr, do_flip = build_crop_geometry(request)
        admitted_at = time.monotonic()
        self._admitted_at[request.ownership.request_id] = admitted_at
        self._admitted_pending += 1
        return _PreparedTrackChunk(
            request=request, crop_batch=crop_batch, crop_geometry=geom, img_center=ctr, do_flip=do_flip
        )

    def request_dispatched(self, request_id: str) -> None:
        self._admitted_pending = max(0, self._admitted_pending - 1)
        self._admitted_at.pop(request_id, None)

    async def infer_tracks(self, request: TrackChunkRequest) -> tuple[CameraSpaceManoResult | None, ServiceError | None]:
        prepared = self.admit(request)
        return (await self.infer_batch([prepared]))[0]

    async def infer_batch(
        self, requests: Sequence[_PreparedTrackChunk]
    ) -> list[tuple[CameraSpaceManoResult | None, ServiceError | None]]:
        """Run exactly one ``HAWOR.forward`` for one Serve batch callback."""
        import numpy as np  # noqa: F401

        if not requests:
            return []
        assert_one_forward(requests, policy=self._config.batch_policy)

        batch_id = uuid4().hex
        dispatched_monotonic_s = time.monotonic()
        admitted_monotonic_s = min(
            (self._admitted_at.get(r.request.ownership.request_id, dispatched_monotonic_s) for r in requests),
            default=dispatched_monotonic_s,
        )
        for r in requests:
            self.request_dispatched(r.request.ownership.request_id)

        self._running_batches += 1
        forward_started_monotonic_s = time.monotonic()
        try:
            import numpy as np

            bchw = np.stack([r.crop_batch for r in requests], axis=0)  # [B,16,3,256,256]
            geom = np.stack([r.crop_geometry for r in requests], axis=0)  # [B,16,4]
            ctr = np.stack([r.img_center for r in requests], axis=0)  # [B,16,2]
            flips = np.array([r.do_flip for r in requests], dtype=bool)
            outputs = await asyncio.to_thread(self._backend.infer_tracks, bchw, geom, ctr, flips)
        except Exception as exc:
            return [
                (
                    None,
                    ServiceError(
                        ErrorCode.MODEL_FAILURE, str(exc), retryable=False,
                        ownership=r.request.ownership, batch_id=batch_id,
                    ),
                )
                for r in requests
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
            forward_count=1,
            model_load_count=self._model_load_count,
        )

        results: list[tuple[CameraSpaceManoResult | None, ServiceError | None]] = []
        pred_rotmat = outputs["pred_rotmat"]  # [B,16,16,3,3]
        trans = outputs["trans"]  # [B,16,3]
        betas = outputs["betas"]  # [B,16,10]
        joints = outputs["joints"]  # [B,16,J,3]
        try:
            vertices = outputs["vertices"]
            vertices = np.asarray(vertices, dtype=np.float32)
            expected_vertices_shape = (len(requests), HAWOR_CHUNK_LEN, HAWOR_MANO_VERTS, 3)
            if vertices.shape != expected_vertices_shape or not np.isfinite(vertices).all():
                raise ValueError(f"invalid backend vertices: expected {expected_vertices_shape}, got {vertices.shape}")
        except Exception as exc:
            return [
                (None, ServiceError(ErrorCode.MODEL_FAILURE, str(exc), retryable=False,
                                    ownership=r.request.ownership, batch_id=batch_id))
                for r in requests
            ]
        for i, r in enumerate(requests):
            try:
                root_orient = pred_rotmat[i, :, 0, :, :]  # [16,3,3]
                hand_pose = pred_rotmat[i, :, 1:, :, :]  # [16,15,3,3]
                trans_i = trans[i]  # [16,3]
                betas_i = betas[i]  # [16,10]
                joints_i = joints[i]  # [16,J,3]
                observed = np.array(
                    [_occlusion_to_observed(o.occlusion_state) for o in r.request.observations], dtype=bool
                )
                occ_states = tuple(o.occlusion_state for o in r.request.observations)
                uncertainty = np.array(
                    [_uncertainty_for(o.occlusion_state, o.detection_confidence) for o in r.request.observations],
                    dtype=np.float32,
                )
                chunk_ts = np.array(
                    [o.source_timestamp_s for o in r.request.observations], dtype=np.float64
                )
                verts_i = np.asarray(vertices[i], dtype=np.float32)
                if verts_i.shape != (HAWOR_CHUNK_LEN, HAWOR_MANO_VERTS, 3) or not np.isfinite(verts_i).all():
                    raise ValueError("invalid backend vertices for request")

                world_lift = None
                world_lift_status = "unavailable"
                if r.request.droid_evidence is not None:
                    world_lift = _as_tensor_payload(
                        _world_lift(r.request.droid_evidence, trans_i, root_orient, chunk_ts)
                    )
                    world_lift_status = (
                        f"resampled_droid_world_from_camera;"
                        f"scale={r.request.droid_evidence.metric_scale:.6f};"
                        f"residual={r.request.droid_evidence.scale_residual:.6f};"
                        f"confidence={r.request.droid_evidence.scale_confidence:.3f}"
                    )

                spatial = SpatialMetadata(
                    source_size=r.request.crop_transforms[0].source_size,
                    model_size=ImageSize(width=HAWOR_CROP_W, height=HAWOR_CROP_H),
                    color_space="RGB",
                    pixel_transform=r.request.crop_transforms[0].pixel_transform,
                    K_px=tuple(tuple(row) for row in r.request.unidepth.K_px),
                )
                result = CameraSpaceManoResult(
                    ownership=r.request.ownership,
                    track_id=r.request.track_id,
                    side=r.request.side,
                    root_orient=_as_tensor_payload(root_orient),
                    hand_pose=_as_tensor_payload(hand_pose),
                    trans=_as_tensor_payload(trans_i),
                    betas=_as_tensor_payload(betas_i),
                    vertices=_as_tensor_payload(verts_i),
                    joints=_as_tensor_payload(joints_i),
                    observed=_as_tensor_payload(observed),
                    occlusion_state=occ_states,
                    uncertainty=_as_tensor_payload(uncertainty),
                    world_lift=world_lift,
                    world_lift_status=world_lift_status,
                    spatial=spatial,
                    model_revision=self._config.model_revision,
                    trace=trace,
                    server_identity=self._server_runtime_identity,
                    batch_diagnostics=(                        {"runtime_config": self._config.runtime_config_wire(), "runtime_config_digest": self._config.runtime_config_digest()}
                        if self._config.performance_instrumentation else None
                    ),
                )
                results.append((result, None))
            except Exception as exc:
                results.append(
                    (
                        None,
                        ServiceError(
                            ErrorCode.RESULT_SPLIT_FAILURE, str(exc), retryable=False,
                            ownership=r.request.ownership, batch_id=batch_id,
                        ),
                    )
                )
        return results

    def status(self) -> DeploymentStatus:
        return DeploymentStatus(
            deployment_name="hawor.infer_tracks",
            replica_id=self._config.replica_id,
            assigned_gpu=self._config.assigned_gpu,
            loaded_models=(self._config.model_revision,),
            admitted_pending=self._admitted_pending,
            running_batches=self._running_batches,
            model_load_count=self._model_load_count,
        )


__all__ = [
    "HaWoRAdapter",
    "HaWoRBackend",
    "HaWoRModelConfig",
    "IMG_NORM_MEAN",
    "IMG_NORM_STD",
    "build_crop_geometry",
    "build_hawor_model_config",
    "expected_hawor_runtime_config",
    "decode_crop_batch",
]
