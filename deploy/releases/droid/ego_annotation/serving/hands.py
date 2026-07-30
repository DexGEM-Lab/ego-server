"""Ray-free resident hands.detect adapter: detector boxes prompting SAM2 for real masks.

This module is importable without Ray installed (ordinary unit tests inject a fake
backend). The Ray Serve deployment wrapper lives in ``hands_deployment.py``.

Contract invariants enforced here:

* Request RGB is uint8 HWC only (``HANDS_RGB_DTYPE``). Float RGB is range
  ambiguous and rejected at the contract boundary.
* The detector (Ultralytics YOLO ``detector.pt``) produces boxes/scores/sides but
  **not masks**. Masks are produced by a real mask mechanism: SAM2 prompted with
  each detector box. A box/ellipse/patch is never converted into a claimed mask.
* The resident config owns the model revision; a request whose ``model_revision``
  does not match is rejected before batching, and every result carries the
  configured revision only.
* One canonical HxW compatibility bucket. Incompatible/overweight items are
  rejected at admission so one Serve callback is exactly one detector forward
  (``batch=`` passed to Ultralytics for a single fused forward). SAM2 mask calls
  run per image per box as postprocessing and are counted in ``sam2_mask_calls``.
* Visibility and uncertainty are derived from real mechanism outputs (mask fill
  within the box, detector score, box-edge proximity), never invented.
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

from ego_annotation.serving.batching import BatchPolicy, assert_one_forward
from ego_annotation.serving.contracts import (
    BatchTrace,
    ContractValidationError,
    DeploymentStatus,
    ErrorCode,
    HandDetection,
    HandsDetectRequest,
    HandsDetectResponse,
    HandsDetectResult,
    ServiceError,
    TensorPayload,
    SCHEMA_VERSION,
    ServerIdentity,
)


class HandsBackend(Protocol):
    """The model boundary for detection + real mask generation.

    ``detect`` receives a list of contiguous uint8 HWC images and returns a list
    (one per image) of dicts with ``boxes`` [K,4] xyxy, ``scores`` [K],
    ``sides`` [K] (0=left,1=right). ``mask`` receives one RGB HWC uint8 image and
    its boxes and returns ``[K,H,W]`` bool masks produced by a real mask model.
    """

    def detect(self, images: Sequence[Any]) -> list[Mapping[str, Any]]: ...

    def mask(self, image_rgb: Any, boxes: Any) -> Any: ...


TensorResolver = Callable[[Any, tuple[int, ...], str], Any]
BackendFactory = Callable[["HandsModelConfig"], HandsBackend]


@dataclass(frozen=True)
class HandsModelConfig:
    """Server-owned model settings; no request field selects a server path."""

    detector_checkpoint: str
    sam2_checkpoint: str
    sam2_config: str
    model_revision: str
    device: str = "cuda"
    replica_id: str = "hands-wilor-gpu1"
    assigned_gpu: int = 1
    # One canonical HxW compatibility bucket. Default 540x960 preserves the
    # EgoScale 16:9 aspect ratio at an explicit 0.5 source-pixel scale (source
    # 1080x1920). The detector resizes internally; masks are produced at model HxW.
    canonical_height: int = 540
    canonical_width: int = 960
    det_conf: float = 0.3
    rescale_factor: float = 2.0
    batch_policy: BatchPolicy = BatchPolicy(
        max_batch_size=8, batch_wait_timeout_s=0.02, max_queued_requests=64
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
        for name in ("detector_checkpoint", "sam2_checkpoint", "sam2_config", "model_revision"):
            if not getattr(self, name):
                raise ContractValidationError(f"hands {name} is required server configuration")
        if self.canonical_height <= 0 or self.canonical_width <= 0:
            raise ContractValidationError("canonical HxW must be positive")
        if self.wire_format not in {"multipart", "envelope"}:
            raise ContractValidationError("wire_format must be multipart or envelope")

    def runtime_config_wire(self) -> dict[str, object]:
        return {
            "schema": "ego.hands-runtime-config.v1",
            "batch_policy": {
                "max_batch_size": self.batch_policy.max_batch_size,
                "batch_wait_timeout_ms": round(self.batch_policy.batch_wait_timeout_s * 1_000.0, 6),
                "max_queued_requests": self.batch_policy.max_queued_requests,
            },
            "canonical_shape": list(self.canonical_shape),
            "det_conf": self.det_conf,
            "rescale_factor": self.rescale_factor,
            "wire_format": self.wire_format,
        }

    def runtime_config_digest(self) -> str:
        raw = json.dumps(self.runtime_config_wire(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    @property
    def canonical_shape(self) -> tuple[int, int, int]:
        return (self.canonical_height, self.canonical_width, 3)


def build_hands_model_config(
    *,
    detector_checkpoint: str,
    sam2_checkpoint: str,
    sam2_config: str,
    model_revision: str,
    device: str = "cuda",
    replica_id: str = "hands-wilor-gpu1",
    assigned_gpu: int = 1,
    canonical_height: int = 540,
    canonical_width: int = 960,
    det_conf: float = 0.3,
    rescale_factor: float = 2.0,
    batch_policy: BatchPolicy | None = None,
    performance_instrumentation: bool = False,
    wire_format: str = "multipart",
    experiment_id: str | None = None,
    application_release_path: str | None = None,
    gcs_address: str | None = None,
    http_port: int | None = None,
    temp_dir: str | None = None,
) -> HandsModelConfig:
    return HandsModelConfig(
        detector_checkpoint=detector_checkpoint,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        model_revision=model_revision,
        device=device,
        replica_id=replica_id,
        assigned_gpu=assigned_gpu,
        canonical_height=canonical_height,
        canonical_width=canonical_width,
        det_conf=det_conf,
        rescale_factor=rescale_factor,
        batch_policy=batch_policy or BatchPolicy(max_batch_size=8, batch_wait_timeout_s=0.02, max_queued_requests=64),
        performance_instrumentation=performance_instrumentation,
        wire_format=wire_format,
        experiment_id=experiment_id,
        application_release_path=application_release_path,
        gcs_address=gcs_address,
        http_port=http_port,
        temp_dir=temp_dir,
    )


@dataclass(frozen=True)
class _PreparedHandsRequest:
    request: HandsDetectRequest
    rgb: Any  # contiguous uint8 HWC


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


def decode_rgb(request: HandsDetectRequest, resolver: TensorResolver) -> Any:
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


def _load_hands_backend(config: HandsModelConfig) -> HandsBackend:
    """Load the Ultralytics detector + SAM2 image predictor once per replica.

    Masks are produced by SAM2 prompted with detector boxes (real masks, not box
    fills). The detector does not divide by 255; Ultralytics owns normalization.
    """
    import os
    import sys

    import numpy as np

    sam2_repo = os.environ.get("EGO_SAM2_REPO", "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/sam2")
    if sam2_repo and sam2_repo not in sys.path:
        sys.path.insert(0, sam2_repo)
    from sam2.build_sam import build_sam2  # type: ignore[import-not-found]
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import-not-found]
    from ultralytics import YOLO  # type: ignore[import-not-found]

    detector = YOLO(config.detector_checkpoint).to(config.device)
    sam2_model = build_sam2(config.sam2_config, config.sam2_checkpoint, device=config.device, mode="eval")
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    class TorchHandsBackend:
        def detect(self, images: Sequence[Any]) -> list[Mapping[str, Any]]:
            # One fused detector forward: pass batch= so Ultralytics stacks the
            # images into a single forward instead of looping.
            results = detector(list(images), conf=config.det_conf, batch=len(images), verbose=False)
            out: list[Mapping[str, Any]] = []
            for res in results:
                if len(res) == 0:
                    out.append({"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32), "sides": np.zeros((0,), np.int64)})
                    continue
                data = res.boxes.data.cpu().numpy()  # [K,6] x1,y1,x2,y2,score,cls
                out.append({
                    "boxes": data[:, :4].astype(np.float32),
                    "scores": data[:, 4].astype(np.float32),
                    "sides": data[:, 5].astype(np.int64),  # 0=left,1=right
                })
            return out

        def mask(self, image_rgb: Any, boxes: Any) -> Any:
            # Real SAM2 masks: set the image embedding once, predict per box.
            sam2_predictor.set_image(np.ascontiguousarray(image_rgb))
            masks = []
            for box in np.asarray(boxes):
                m, _ious, _low = sam2_predictor.predict(box=box.astype(np.float32)[None, :], multimask_output=False)
                masks.append(np.asarray(m[0], dtype=bool))
            if not masks:
                return np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=bool)
            return np.stack(masks, axis=0)

    return TorchHandsBackend()


def _as_tensor_payload(value: Any) -> TensorPayload:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    return TensorPayload(data=array.tobytes(), shape=tuple(int(dim) for dim in array.shape), dtype=array.dtype.name)


def _compute_visibility_uncertainty(masks: Any, boxes: Any, scores: Any, height: int, width: int) -> tuple[Any, Any]:
    """Real, mechanism-grounded per-hand visibility and uncertainty.

    visibility = mask fill fraction within the detector box (how much of the box
    the real SAM2 silhouette occupies; lower => more occlusion/truncation).
    uncertainty = (1 - score) + edge-truncation penalty (box touching the image
    border => higher uncertainty because the hand may extend out of frame).
    """
    import numpy as np

    k = len(scores)
    visibility = np.zeros((k,), np.float32)
    uncertainty = np.zeros((k,), np.float32)
    for i in range(k):
        box = boxes[i]
        x1, y1, x2, y2 = box
        x1c, y1c, x2c, y2c = max(0, int(round(x1))), max(0, int(round(y1))), min(width, int(round(x2))), min(height, int(round(y2)))
        box_area = max(1.0, (x2c - x1c) * (y2c - y1c))
        m = masks[i]
        mask_area = float(m[y1c:y2c, x1c:x2c].sum()) if (y2c > y1c and x2c > x1c) else 0.0
        visibility[i] = float(np.clip(mask_area / box_area, 0.0, 1.0))
        edge = 0.0
        if x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
            edge = 0.1
        uncertainty[i] = float(np.clip((1.0 - float(scores[i])) + edge, 0.0, 1.0))
    return visibility, uncertainty


class HandsAdapter:
    """Resident detector+SAM2 owner used by the GPU1 Ray Serve replica."""

    def __init__(
        self,
        config: HandsModelConfig,
        *,
        backend_factory: BackendFactory = _load_hands_backend,
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
                raise RuntimeError("Hands experiment identity requires release root, GCS address, HTTP port, and temp dir")
            from ego_annotation.serving.benchmark.release import derive_worker_runtime_evidence

            derive = runtime_evidence_factory or derive_worker_runtime_evidence
            evidence = derive(release_root=release_root, checkpoint_path=config.detector_checkpoint, imported_module_file=__file__)
            if evidence.physical_gpu != config.assigned_gpu:
                raise RuntimeError(f"Hands worker physical GPU {evidence.physical_gpu} differs from planned GPU {config.assigned_gpu}")
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
    def config(self) -> HandsModelConfig:
        return self._config

    def admit(self, request: HandsDetectRequest) -> _PreparedHandsRequest:
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
            raise ContractValidationError("each hands request must be exactly one canonical work unit")
        rgb = decode_rgb(request, self._tensor_resolver)
        admitted_at = time.monotonic()
        self._admitted_at[request.ownership.request_id] = admitted_at
        self._admitted_pending += 1
        return _PreparedHandsRequest(request=request, rgb=rgb)

    def request_dispatched(self, request_id: str) -> None:
        self._admitted_pending = max(0, self._admitted_pending - 1)
        self._admitted_at.pop(request_id, None)

    async def infer(self, request: HandsDetectRequest) -> HandsDetectResponse:
        prepared = self.admit(request)
        return (await self.infer_batch([prepared]))[0]

    async def infer_batch(self, requests: Sequence[_PreparedHandsRequest]) -> list[HandsDetectResponse]:
        """Run exactly one detector forward for one Serve batch callback.

        SAM2 mask generation runs per image per box as postprocessing; it produces
        real masks (box-prompted) and is counted in ``sam2_mask_calls``.
        """
        import numpy as np  # noqa: F401

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
            images = [item.rgb for item in requests]
            detections = await asyncio.to_thread(self._backend.detect, images)
            height = self._config.canonical_height
            width = self._config.canonical_width
            per_image: list[HandDetection] = []
            sam2_calls = 0
            for det in detections:
                boxes = det["boxes"]
                scores = det["scores"]
                sides = det["sides"]
                k = len(scores)
                if k == 0:
                    per_image.append(HandDetection.empty(height, width))
                    continue
                rgb_for_mask = images[len(per_image)]
                masks = await asyncio.to_thread(self._backend.mask, rgb_for_mask, boxes)
                sam2_calls += 1
                visibility, uncertainty = _compute_visibility_uncertainty(masks, boxes, scores, height, width)
                per_image.append(
                    HandDetection(
                        boxes=_as_tensor_payload(boxes.astype(np.float32)),
                        scores=_as_tensor_payload(scores.astype(np.float32)),
                        sides=_as_tensor_payload(sides.astype(np.uint8)),
                        masks=_as_tensor_payload(masks.astype(np.uint8)),
                        visibility=_as_tensor_payload(visibility.astype(np.float32)),
                        uncertainty=_as_tensor_payload(uncertainty.astype(np.float32)),
                        n_hands=k,
                    )
                )
        except Exception as exc:
            return [
                HandsDetectResponse(
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
        responses: list[HandsDetectResponse] = []
        for index, item in enumerate(requests):
            try:
                responses.append(
                    HandsDetectResponse(
                        ownership=item.request.ownership,
                        result=HandsDetectResult(
                            ownership=item.request.ownership,
                            detection=per_image[index],
                            spatial=item.request.spatial,
                            model_revision=self._config.model_revision,
                            trace=trace,
                            sam2_mask_calls=sam2_calls,
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
                    HandsDetectResponse(
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
            deployment_name="hands.detect",
            replica_id=self._config.replica_id,
            assigned_gpu=self._config.assigned_gpu,
            loaded_models=(self._config.model_revision,),
            admitted_pending=self._admitted_pending,
            running_batches=self._running_batches,
            model_load_count=self._model_load_count,
        )
