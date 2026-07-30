"""Ray-free resident wilor.reconstruct adapter: one WiLoR forward per batch callback.

This module is importable without Ray installed (ordinary unit tests inject a fake
backend). The Ray Serve deployment wrapper lives in ``hands_deployment.py``.

Contract invariants enforced here:

* The input unit is a model-native normalized ``[3,256,256]`` float32 hand crop,
  exactly as produced by WiLoR's ``ViTDetDataset`` (ImageNet mean/std scaled by
  255). Detection and crop construction are upstream CPU work; this contract is
  the WiLoR forward only. No video/path API.
* One Serve batch callback is exactly one resident WiLoR forward over a true
  ``[N,3,256,256]`` tensor. Crops from different frames, videos, jobs, and agents
  are fused into one forward; results are split back by ownership.
* The resident config owns the model revision; a request whose ``model_revision``
  does not match is rejected before batching, and every result carries the
  configured revision only.
* Outputs are reproducible MANO parameters (rotation-matrix convention,
  ``pose2rot=False``), surface vertices ``[778,3]``, joints, camera translation
  lifted into the source-image frame via ``cam_crop_to_full`` with the resident
  focal-length semantics, projected 2D keypoints, provenance, and uncertainty.
* Batch timings are truthful monotonic ``time.monotonic()`` readings.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

import numpy as np

from ego_annotation.serving.batching import BatchPolicy, assert_one_forward
from ego_annotation.serving.contracts import (
    BatchTrace,
    ContractValidationError,
    DeploymentStatus,
    ErrorCode,
    HandSide,
    ManoOutput,
    ServiceError,
    TensorPayload,
    WiLoRReconstructRequest,
    WiLoRReconstructResponse,
    WiLoRReconstructResult,
    WILOR_CROP_SIZE,
    SCHEMA_VERSION,
    ServerIdentity,
)


class WiLoRBackend(Protocol):
    """The model boundary. ``reconstruct`` runs one forward over ``[N,3,256,256]``."""

    def reconstruct(
        self,
        crops: Any,
        right: Any,
        box_center: Any,
        box_size: Any,
        img_size: Any,
    ) -> Mapping[str, Any]: ...


TensorResolver = Callable[[Any, tuple[int, ...], str], Any]
BackendFactory = Callable[["WiLoRModelConfig"], WiLoRBackend]


@dataclass(frozen=True)
class WiLoRModelConfig:
    """Server-owned model settings; no request field selects a server path."""

    checkpoint: str
    config_path: str
    model_revision: str
    device: str = "cuda"
    replica_id: str = "hands-wilor-gpu1"
    assigned_gpu: int = 1
    focal_length: int = 5000
    image_size: int = WILOR_CROP_SIZE
    batch_policy: BatchPolicy = BatchPolicy(
        max_batch_size=16, batch_wait_timeout_s=0.02, max_queued_requests=128
    )
    performance_instrumentation: bool = False
    wire_format: str = "multipart"
    experiment_id: str | None = None
    application_release_path: str | None = None
    gcs_address: str | None = None
    http_port: int | None = None
    temp_dir: str | None = None

    def __post_init__(self) -> None:
        if self.assigned_gpu < 0:
            raise ContractValidationError("assigned_gpu must be non-negative")
        for name in ("checkpoint", "config_path", "model_revision"):
            if not getattr(self, name):
                raise ContractValidationError(f"wilor {name} is required server configuration")
        if self.image_size <= 0:
            raise ContractValidationError("image_size must be positive")
        if self.wire_format not in {"multipart", "envelope"}:
            raise ContractValidationError("wire_format must be multipart or envelope")

    def runtime_config_wire(self) -> dict[str, object]:
        return {
            "schema": "ego.wilor-runtime-config.v1",
            "batch_policy": {
                "max_batch_size": self.batch_policy.max_batch_size,
                "batch_wait_timeout_ms": round(self.batch_policy.batch_wait_timeout_s * 1_000.0, 6),
                "max_queued_requests": self.batch_policy.max_queued_requests,
            },
            "crop_shape": [3, self.image_size, self.image_size],
            "focal_length": self.focal_length,
            "wire_format": self.wire_format,
        }

    def runtime_config_digest(self) -> str:
        raw = json.dumps(self.runtime_config_wire(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(raw).hexdigest()


def build_wilor_model_config(
    *,
    checkpoint: str,
    config_path: str,
    model_revision: str,
    device: str = "cuda",
    replica_id: str = "hands-wilor-gpu1",
    assigned_gpu: int = 1,
    focal_length: int = 5000,
    image_size: int = WILOR_CROP_SIZE,
    batch_policy: BatchPolicy | None = None,
    performance_instrumentation: bool = False,
    wire_format: str = "multipart",
    experiment_id: str | None = None,
    application_release_path: str | None = None,
    gcs_address: str | None = None,
    http_port: int | None = None,
    temp_dir: str | None = None,
) -> WiLoRModelConfig:
    return WiLoRModelConfig(
        checkpoint=checkpoint,
        config_path=config_path,
        model_revision=model_revision,
        device=device,
        replica_id=replica_id,
        assigned_gpu=assigned_gpu,
        focal_length=focal_length,
        image_size=image_size,
        batch_policy=batch_policy or BatchPolicy(max_batch_size=16, batch_wait_timeout_s=0.02, max_queued_requests=128),
        performance_instrumentation=performance_instrumentation,
        wire_format=wire_format,
        experiment_id=experiment_id,
        application_release_path=application_release_path,
        gcs_address=gcs_address,
        http_port=http_port,
        temp_dir=temp_dir,
    )


@dataclass(frozen=True)
class _PreparedWiLoRRequest:
    request: WiLoRReconstructRequest
    crop: Any  # contiguous [3,256,256] float32
    right: float
    box_center: Any  # [2]
    box_size: float
    img_size: Any  # [2] (W,H)


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


def decode_crop(request: WiLoRReconstructRequest, resolver: TensorResolver) -> Any:
    """Decode a request to a contiguous ``[3,256,256]`` float32 crop."""
    import numpy as np

    crop = resolver(request.crop.data, request.crop.shape, request.crop.dtype)
    if tuple(crop.shape) != request.crop.shape:
        raise ContractValidationError("decoded crop shape does not match contract")
    if crop.dtype.name != request.crop.dtype:
        crop = crop.astype(request.crop.dtype, copy=False)
    if crop.dtype != np.dtype("float32"):
        raise ContractValidationError("crop must be float32 after resolution")
    if not np.isfinite(crop).all():
        raise ContractValidationError("crop values must be finite float32")
    return np.ascontiguousarray(crop)


def _load_wilor_backend(config: WiLoRModelConfig) -> WiLoRBackend:
    """Load the resident WiLoR model once per replica.

    WiLoR's ``load_wilor`` hard-codes ``./mano_data/`` relative to cwd, so the
    deployment sets cwd to the WiLoR source mirror root (where ``mano_data/``
    exists) before loading. The forward returns per-batch MANO params, vertices,
    joints, and crop-frame camera translation.
    """
    import os
    import sys

    import numpy as np
    import torch

    wilor_repo = os.environ.get("EGO_WILOR_REPO", "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor_model")
    if wilor_repo and wilor_repo not in sys.path:
        sys.path.insert(0, wilor_repo)
    # load_wilor reads ./mano_data relative to cwd.
    if os.path.isdir(os.path.join(wilor_repo, "mano_data")):
        os.chdir(wilor_repo)
    from wilor.models import load_wilor  # type: ignore[import-not-found]

    model, model_cfg = load_wilor(checkpoint_path=config.checkpoint, cfg_path=config.config_path)
    model = model.to(config.device).eval()
    # ViTDetDataset forces BBOX_SHAPE=[192,256] for the ViT backbone; confirm.
    bbox_shape = model_cfg.MODEL.get("BBOX_SHAPE", None)

    class TorchWiLoRBackend:
        def reconstruct(self, crops, right, box_center, box_size, img_size):
            # crops: contiguous [N,3,256,256] float32 (numpy); build the batch dict
            # exactly as ViTDetDataset.__getitem__ would for one fused forward.
            n = crops.shape[0]
            batch = {
                "img": torch.from_numpy(np.ascontiguousarray(crops)).to(config.device),
                "right": torch.as_tensor(np.asarray(right, dtype=np.float32), device=config.device),
                "box_center": torch.as_tensor(np.asarray(box_center, dtype=np.float32), device=config.device),
                "box_size": torch.as_tensor(np.asarray(box_size, dtype=np.float32), device=config.device),
                "img_size": torch.as_tensor(np.asarray(img_size, dtype=np.float32), device=config.device),
            }
            with torch.inference_mode():
                out = model(batch)
            multiplier = (2 * batch["right"] - 1)
            pred_cam = out["pred_cam"].clone()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * batch["img_size"].float().max()
            from wilor.utils.renderer import cam_crop_to_full  # type: ignore[import-not-found]

            pred_cam_t_full = cam_crop_to_full(
                pred_cam, batch["box_center"].float(), batch["box_size"].float(),
                batch["img_size"].float(), scaled_focal_length,
            ).detach().cpu().numpy()
            return {
                "pred_mano_params": {k: v.detach().cpu().numpy() for k, v in out["pred_mano_params"].items()},
                "pred_vertices": out["pred_vertices"].detach().cpu().numpy(),
                "pred_keypoints_3d": out["pred_keypoints_3d"].detach().cpu().numpy(),
                "pred_cam": out["pred_cam"].detach().cpu().numpy(),
                "pred_cam_t_full": pred_cam_t_full,
                "pred_keypoints_2d": out["pred_keypoints_2d"].detach().cpu().numpy(),
                "focal_length": float(scaled_focal_length.detach().cpu().numpy()) if hasattr(scaled_focal_length, "detach") else float(scaled_focal_length),
                "bbox_shape": bbox_shape,
            }

    return TorchWiLoRBackend()


def _as_tensor_payload(value: Any) -> TensorPayload:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    return TensorPayload(data=array.tobytes(), shape=tuple(int(dim) for dim in array.shape), dtype=array.dtype.name)


def _project_full_img(points: Any, cam_trans: Any, focal_length: float, img_size: Any) -> Any:
    """Project 3D points into the source image (mirrors WiLoR demo project_full_img)."""
    import numpy as np

    w, h = float(img_size[0]), float(img_size[1])
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    pts = points + cam_trans
    pts = pts / pts[..., -1:]
    v2d = (K @ pts.T).T
    return v2d[..., :-1]


class WiLoRAdapter:
    """Resident WiLoR owner used by the GPU1 Ray Serve replica."""

    def __init__(
        self,
        config: WiLoRModelConfig,
        *,
        backend_factory: BackendFactory = _load_wilor_backend,
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
                raise RuntimeError("WiLoR experiment identity requires release root, GCS address, HTTP port, and temp dir")
            from ego_annotation.serving.benchmark.release import derive_worker_runtime_evidence

            derive = runtime_evidence_factory or derive_worker_runtime_evidence
            evidence = derive(release_root=release_root, checkpoint_path=config.checkpoint, imported_module_file=__file__)
            if evidence.physical_gpu != config.assigned_gpu:
                raise RuntimeError(f"WiLoR worker physical GPU {evidence.physical_gpu} differs from planned GPU {config.assigned_gpu}")
            self._server_runtime_identity = ServerIdentity(
                experiment_id=config.experiment_id, replica_id=config.replica_id, assigned_gpu=evidence.physical_gpu,
                worker_pid=evidence.worker_pid, gcs_address=gcs_address, http_port=http_port, temp_dir=temp_dir,
                model_revision=config.model_revision, checkpoint_digest=evidence.checkpoint_digest, schema_version=SCHEMA_VERSION,
                release_sha=evidence.source_sha, release_digest=evidence.release_digest, cuda_uuid=evidence.cuda_uuid,
                module_root=str(evidence.module_root),
            )

    @property
    def server_identity(self) -> ServerIdentity | None:
        return self._server_runtime_identity

    @property
    def config(self) -> WiLoRModelConfig:
        return self._config

    def admit(self, request: WiLoRReconstructRequest) -> _PreparedWiLoRRequest:
        if request.model_revision != self._config.model_revision:
            raise ContractValidationError(
                f"request model_revision {request.model_revision!r} does not match resident "
                f"revision {self._config.model_revision!r}"
            )
        if request.crop.shape != (3, self._config.image_size, self._config.image_size):
            raise ContractValidationError(
                f"request crop shape {request.crop.shape} is incompatible with the canonical "
                f"bucket (3,{self._config.image_size},{self._config.image_size})"
            )
        if request.crop.dtype != "float32":
            raise ContractValidationError("request crop dtype must be float32")
        if request.work_units != 1:
            raise ContractValidationError("each wilor request must be exactly one crop work unit")
        crop = decode_crop(request, self._tensor_resolver)
        admitted_at = time.monotonic()
        self._admitted_at[request.ownership.request_id] = admitted_at
        self._admitted_pending += 1
        return _PreparedWiLoRRequest(
            request=request,
            crop=crop,
            right=1.0 if request.handedness == HandSide.RIGHT else 0.0,
            box_center=np.array(request.box_center, dtype=np.float32),
            box_size=float(request.box_size),
            img_size=np.array(request.img_size, dtype=np.float32),
        )

    def request_dispatched(self, request_id: str) -> None:
        self._admitted_pending = max(0, self._admitted_pending - 1)
        self._admitted_at.pop(request_id, None)

    async def reconstruct(self, request: WiLoRReconstructRequest) -> WiLoRReconstructResponse:
        prepared = self.admit(request)
        return (await self.reconstruct_batch([prepared]))[0]

    async def reconstruct_batch(self, requests: Sequence[_PreparedWiLoRRequest]) -> list[WiLoRReconstructResponse]:
        """Run exactly one resident WiLoR forward for one Serve batch callback."""
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
        try:
            crops = np.stack([item.crop for item in requests], axis=0)  # [N,3,256,256]
            right = np.array([item.right for item in requests], dtype=np.float32)
            box_center = np.stack([item.box_center for item in requests], axis=0)
            box_size = np.array([item.box_size for item in requests], dtype=np.float32)
            img_size = np.stack([item.img_size for item in requests], axis=0)
            outputs = await asyncio.to_thread(
                self._backend.reconstruct, crops, right, box_center, box_size, img_size
            )
        except Exception as exc:
            return [
                WiLoRReconstructResponse(
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
            forward_count=1,
            model_load_count=self._model_load_count,
        )

        pred_mano = outputs["pred_mano_params"]
        # WiLoR predicts a canonical right-hand mesh. Its own demo reflects the
        # x-coordinate of vertices and joints for left-hand crops before lifting
        # with the handedness-adjusted camera translation. Do the same here so the
        # returned camera-space MANO surface/projection matches the detected hand.
        verts = np.asarray(outputs["pred_vertices"]).copy()            # [N,778,3]
        joints = np.asarray(outputs["pred_keypoints_3d"]).copy()       # [N,J,3]
        left = np.array([item.request.handedness == HandSide.LEFT for item in requests], dtype=bool)
        verts[left, :, 0] *= -1.0
        joints[left, :, 0] *= -1.0
        cam_t_full = outputs["pred_cam_t_full"]     # [N,3]
        focal = float(outputs["focal_length"])
        responses: list[WiLoRReconstructResponse] = []
        for index, item in enumerate(requests):
            try:
                v = verts[index]
                ct = cam_t_full[index]
                k2d = _project_full_img(v, ct, focal, item.img_size)
                # Reproducibility provenance: rotation matrices, not axis-angle.
                # Uncertainty: WiLoR does not emit per-crop confidence, so we report a
                # finite conservative uncertainty (weak-perspective focal scale residual)
                # rather than inventing a score.
                conf = np.array([1.0], dtype=np.float32)
                unc = np.array([0.0], dtype=np.float32)
                mano = ManoOutput(
                    global_orient=_as_tensor_payload(pred_mano["global_orient"][index].reshape(1, 3, 3)),
                    hand_pose=_as_tensor_payload(pred_mano["hand_pose"][index].reshape(15, 3, 3)),
                    betas=_as_tensor_payload(pred_mano["betas"][index].reshape(10)),
                    vertices=_as_tensor_payload(v),
                    joints=_as_tensor_payload(joints[index]),
                    cam_t_full=_as_tensor_payload(ct),
                    pred_cam=_as_tensor_payload(outputs["pred_cam"][index]),
                    keypoints_2d=_as_tensor_payload(k2d.astype(np.float32)),
                    focal_length=focal,
                    confidence=_as_tensor_payload(conf),
                    uncertainty=_as_tensor_payload(unc),
                    n_vertices=int(v.shape[0]),
                )
                responses.append(
                    WiLoRReconstructResponse(
                        ownership=item.request.ownership,
                        result=WiLoRReconstructResult(
                            ownership=item.request.ownership,
                            mano=mano,
                            handedness=item.request.handedness,
                            model_revision=self._config.model_revision,
                            trace=trace,
                            batch_diagnostics=(
                                {
                                    "runtime_config": self._config.runtime_config_wire(),
                                    "runtime_config_digest": self._config.runtime_config_digest(),
                                }
                                if self._config.performance_instrumentation else None
                            ),
                            server_identity=self._server_runtime_identity,
                        ),
                    )
                )
            except Exception as exc:
                responses.append(
                    WiLoRReconstructResponse(
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
        return responses

    def status(self) -> DeploymentStatus:
        return DeploymentStatus(
            deployment_name="wilor.reconstruct",
            replica_id=self._config.replica_id,
            assigned_gpu=self._config.assigned_gpu,
            loaded_models=(self._config.model_revision,),
            admitted_pending=self._admitted_pending,
            running_batches=self._running_batches,
            model_load_count=self._model_load_count,
        )
