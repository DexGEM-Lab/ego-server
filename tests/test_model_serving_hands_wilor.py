"""CPU-only tests for the GPU1 hands.detect + wilor.reconstruct model API foundation.

These never import Ray and never load a model. They cover:
- uint8-only RGB for hands + float32 [3,256,256] crop for wilor at the contract boundary
- real-mask invariant: the fake SAM2 backend returns a silhouette, not a box fill; the
  adapter never converts boxes into masks
- one detector forward / one WiLoR forward per Serve batch callback (forward_count==1)
- resident-revision ownership (request revision must match; results carry resident revision)
- ownership-safe batch splitting (mixed jobs/agents return their own results)
- truthful monotonic batch tracing
- multipart binary transport for hands and wilor (request + response)
- corrected native-GPU lifecycle for GPU1 (ray_serve_hands interpreter, ports, num_gpus=1)
- no-Ray adapter imports
"""
from __future__ import annotations

import asyncio
import importlib
import sys
import types
from typing import Any, cast

import numpy as np
import pytest

from ego_annotation.serving.batching import BatchPolicy, assert_one_forward, canonical_batch_size_fn
from ego_annotation.serving.contracts import (
    ContractValidationError,
    ErrorCode,
    HandSide,
    ImageSize,
    Ownership,
    PixelTransform,
    SpatialMetadata,
    TensorPayload,
)
from ego_annotation.serving.hands import (
    HANDS_YOLO_V2_REVISION,
    HandsAdapter,
    _load_hands_backend,
    HandsModelConfig,
    build_hands_model_config,
)
from ego_annotation.serving.lifecycle import (
    COMMITTED_GPU_GROUPS,
    RAY_VERSION,
    hands_gpu_group,
    hands_wilor_serve_config,
)
from ego_annotation.serving.transport import (
    build_multipart_request_fields,
    build_multipart_response,
    parse_multipart_request_fields,
    parse_multipart_response,
)
from ego_annotation.serving.wilor import (
    WiLoRAdapter,
    WiLoRModelConfig,
    build_wilor_model_config,
)
from ego_annotation.serving.contracts import (
    HandsDetectRequest,
    WiLoRReconstructRequest,
)


HANDS_REV = "hands-yolo-sam2.1-hiera-l"
HANDS_YOLO_REV = HANDS_YOLO_V2_REVISION
WILOR_REV = "wilor-final-v1"
H, W = 8, 12  # canonical hands test size


# --------------------------------------------------------------------------- helpers


def make_hands_request(
    request_id: str,
    *,
    job_id: str = "job-a",
    pixels: int = 200,
    shape: tuple[int, int] = (H, W),
    dtype: str = "uint8",
    model_revision: str = HANDS_REV,
) -> HandsDetectRequest:
    height, width = shape
    if dtype == "uint8":
        rgb = np.full((height, width, 3), pixels, dtype=np.uint8)
    else:
        rgb = np.full((height, width, 3), pixels, dtype=np.float32)
    return HandsDetectRequest(
        ownership=Ownership(
            request_id=request_id, job_id=job_id, item_id=f"frame-{request_id}",
            stage_id="hands.detect", source_id=f"source-{request_id}", source_timestamp_s=1.25,
        ),
        rgb=TensorPayload(data=rgb.tobytes(), shape=rgb.shape, dtype=dtype),
        spatial=SpatialMetadata(
            source_size=ImageSize(width=width, height=height),
            model_size=ImageSize(width=width, height=height),
            color_space="RGB",
            pixel_transform=PixelTransform.identity(),
        ),
        model_revision=model_revision,
    )


def make_wilor_request(
    request_id: str,
    *,
    job_id: str = "job-a",
    handedness: HandSide = HandSide.RIGHT,
    model_revision: str = WILOR_REV,
) -> WiLoRReconstructRequest:
    crop = np.random.RandomState(int(request_id, 16) % 2**31).randn(3, 256, 256).astype(np.float32)
    return WiLoRReconstructRequest(
        ownership=Ownership(
            request_id=request_id, job_id=job_id, item_id=f"hand-{request_id}",
            stage_id="wilor.reconstruct", source_id=f"source-{request_id}", source_timestamp_s=1.25,
        ),
        crop=TensorPayload(data=crop.tobytes(), shape=crop.shape, dtype="float32"),
        handedness=handedness,
        box_center=(100.0, 200.0),
        box_size=300.0,
        img_size=(1920.0, 1080.0),
        model_revision=model_revision,
    )


def make_hands_config(**overrides: Any) -> HandsModelConfig:
    config = {
        "detector_checkpoint": "server-owned-detector.pt",
        "sam2_checkpoint": "server-owned-sam2.pt",
        "sam2_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "model_revision": HANDS_REV,
        "canonical_height": H,
        "canonical_width": W,
        "batch_policy": BatchPolicy(max_batch_size=8, batch_wait_timeout_s=0.01, max_queued_requests=4),
    }
    config.update(overrides)
    return build_hands_model_config(**config)


def make_wilor_config(**overrides: Any) -> WiLoRModelConfig:
    return build_wilor_model_config(
        checkpoint="server-owned-wilor.ckpt",
        config_path="server-owned-config.yaml",
        model_revision=WILOR_REV,
        batch_policy=BatchPolicy(max_batch_size=16, batch_wait_timeout_s=0.01, max_queued_requests=4),
        **overrides,
    )


# --------------------------------------------------------------------------- hands


class FakeHandsBackend:
    """Fake detector + SAM2. The mask is a real silhouette (circle inside box), NOT a box fill."""

    def __init__(self) -> None:
        self.detect_calls = 0
        self.mask_calls = 0
        self.detect_batches: list[int] = []
        # Configurable detection: per-image number of hands and box placement.
        self.n_hands_per_image: list[int] = []

    def detect(self, images: list[Any]) -> list[dict[str, Any]]:
        self.detect_calls += 1
        self.detect_batches.append(len(images))
        out: list[dict[str, Any]] = []
        for i, img in enumerate(images):
            n = self.n_hands_per_image[i] if i < len(self.n_hands_per_image) else 1
            h, w = img.shape[:2]
            boxes, scores, sides = [], [], []
            for k in range(n):
                x1, y1 = 1 + k, 1 + k
                x2, y2 = w - 1 - k, h - 1 - k
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                scores.append(0.9 - 0.05 * k)
                sides.append(k % 2)  # alternate left/right
            out.append({
                "boxes": np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), np.float32),
                "scores": np.array(scores, dtype=np.float32) if scores else np.zeros((0,), np.float32),
                "sides": np.array(sides, dtype=np.int64) if sides else np.zeros((0,), np.int64),
            })
        return out

    def mask(self, image_rgb: Any, boxes: Any) -> Any:
        self.mask_calls += 1
        h, w = image_rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        masks = []
        for box in np.asarray(boxes):
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            rx = (box[2] - box[0]) / 4.0  # circle radius = 1/4 box width => fill ~ pi/16 ~ 0.2
            ry = (box[3] - box[1]) / 4.0
            m = ((xx - cx) ** 2 / (rx * rx + 1e-9) + (yy - cy) ** 2 / (ry * ry + 1e-9)) <= 1.0
            masks.append(m)
        return np.stack(masks, axis=0) if masks else np.zeros((0, h, w), dtype=bool)


def make_hands_adapter(backend: FakeHandsBackend | None = None, *, config: HandsModelConfig | None = None) -> tuple[HandsAdapter, FakeHandsBackend]:
    backend = backend or FakeHandsBackend()
    return HandsAdapter(config or make_hands_config(), backend_factory=lambda _: backend), backend


def test_hands_contract_rejects_float_rgb() -> None:
    with pytest.raises(ContractValidationError, match="uint8"):
        make_hands_request("x", dtype="float32")


def test_hands_adapter_one_detector_forward_and_real_masks() -> None:
    backend = FakeHandsBackend()
    backend.n_hands_per_image = [2, 1]
    adapter, backend = make_hands_adapter(backend)
    responses = asyncio.run(adapter.infer_batch([
        adapter.admit(make_hands_request("a", job_id="job-a")),
        adapter.admit(make_hands_request("b", job_id="job-b")),
    ]))
    # One detector forward for the whole batch.
    assert backend.detect_calls == 1
    assert backend.detect_batches == [2]
    assert all(r.result is not None for r in responses)
    trace = responses[0].result.trace
    assert trace.forward_count == 1
    assert trace.request_count == 2
    # Ownership split: each response carries its own ownership.
    assert responses[0].ownership.request_id == "a"
    assert responses[1].ownership.request_id == "b"
    # Real masks: not a box fill. Box fill would be ~1.0; the circle silhouette
    # fill within the box is ~0.2.
    det0 = responses[0].result.detection
    assert det0.n_hands == 2
    masks = np.frombuffer(det0.masks.data, dtype="uint8").reshape(det0.masks.shape)
    box = np.frombuffer(det0.boxes.data, dtype="float32").reshape(-1, 4)
    for i in range(det0.n_hands):
        x1, y1, x2, y2 = [int(round(v)) for v in box[i]]
        region = masks[i, y1:y2, x1:x2]
        fill = float(region.mean())
        assert fill < 0.9, f"mask {i} box-fill={fill} looks like a box fill, not a real silhouette"
        assert fill > 0.0, f"mask {i} is empty"
    # SAM2 mask calls: one per image with >=1 box.
    assert backend.mask_calls == 2
    assert responses[0].result.sam2_mask_calls == 2


def test_hands_visibility_and_uncertainty_are_finite_and_bounded() -> None:
    backend = FakeHandsBackend()
    backend.n_hands_per_image = [1]
    adapter, _ = make_hands_adapter(backend)
    responses = asyncio.run(adapter.infer_batch([adapter.admit(make_hands_request("a"))]))
    det = responses[0].result.detection
    vis = np.frombuffer(det.visibility.data, dtype="float32")
    unc = np.frombuffer(det.uncertainty.data, dtype="float32")
    assert np.all(vis >= 0) and np.all(vis <= 1)
    assert np.all(unc >= 0) and np.all(unc <= 1)
    # uncertainty reflects 1 - score (score ~0.9 => uncertainty ~0.1, plus edge term).
    assert float(unc[0]) >= 1.0 - 0.9 - 1e-6


def test_hands_empty_detection_is_valid() -> None:
    backend = FakeHandsBackend()
    backend.n_hands_per_image = [0]
    adapter, _ = make_hands_adapter(backend)
    responses = asyncio.run(adapter.infer_batch([adapter.admit(make_hands_request("empty"))]))
    det = responses[0].result.detection
    assert det.n_hands == 0
    assert det.boxes.shape == (0, 4)
    # No SAM2 call when there are no boxes.
    assert responses[0].result.sam2_mask_calls == 0


def test_hands_yolo_v2_backend_loads_no_sam2_and_fuses_yolo(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[Any], dict[str, Any]]] = []

    class FakeTensor:
        def __init__(self, data: Any) -> None:
            self.data = data

        def cpu(self) -> "FakeTensor":
            return self

        def numpy(self) -> Any:
            return self.data

    class FakeResult:
        def __init__(self, data: Any) -> None:
            self.boxes = types.SimpleNamespace(data=FakeTensor(data))

        def __len__(self) -> int:
            return len(self.boxes.data.data)

    class FakeYOLO:
        def __init__(self, checkpoint: str) -> None:
            assert checkpoint == "server-owned-detector.pt"

        def to(self, device: str) -> "FakeYOLO":
            assert device == "cuda"
            return self

        def __call__(self, images: list[Any], **kwargs: Any) -> list[FakeResult]:
            calls.append((images, kwargs))
            return [FakeResult(np.array([[1, 2, 3, 4, 0.9, 1]], dtype=np.float32)) for _ in images]

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.delitem(sys.modules, "sam2", raising=False)
    backend = _load_hands_backend(make_hands_config(model_revision=HANDS_YOLO_REV))
    detections = backend.detect([np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W, 3), dtype=np.uint8)])

    assert len(calls) == 1
    assert calls[0][1] == {"conf": 0.3, "batch": 2, "verbose": False}
    assert len(detections) == 2
    assert detections[0]["boxes"].dtype == np.float32
    assert detections[0]["scores"].dtype == np.float32
    assert detections[0]["sides"].dtype == np.int64


def test_hands_yolo_v2_skips_masks_and_preserves_detector_contract() -> None:
    backend = FakeHandsBackend()
    backend.n_hands_per_image = [1, 0]
    adapter, backend = make_hands_adapter(backend, config=make_hands_config(model_revision=HANDS_YOLO_REV))
    responses = asyncio.run(adapter.infer_batch([
        adapter.admit(make_hands_request("detected", model_revision=HANDS_YOLO_REV)),
        adapter.admit(make_hands_request("empty", model_revision=HANDS_YOLO_REV)),
    ]))

    detected, empty = (response.result for response in responses)
    assert detected is not None and empty is not None
    assert backend.detect_calls == 1
    assert backend.detect_batches == [2]
    assert backend.mask_calls == 0
    assert [response.sam2_mask_calls for response in (detected, empty)] == [0, 0]
    assert detected.model_revision == HANDS_YOLO_REV

    det = detected.detection
    assert det.masks is None
    assert det.boxes.shape == (1, 4) and det.boxes.dtype == "float32"
    assert det.scores.shape == (1,) and det.scores.dtype == "float32"
    assert det.sides.shape == (1,) and det.sides.dtype == "uint8"
    assert det.visibility.shape == (1,) and det.visibility.dtype == "float32"
    assert det.uncertainty.shape == (1,) and det.uncertainty.dtype == "float32"
    assert np.array_equal(np.frombuffer(det.visibility.data, dtype=np.float32), np.ones((1,), dtype=np.float32))
    # Fake score is 0.9 and its box touches the edge, so uncertainty is 0.1 + 0.1.
    assert np.allclose(np.frombuffer(det.uncertainty.data, dtype=np.float32), [0.2])
    assert "masks" not in det.to_wire()

    empty_det = empty.detection
    assert empty_det.n_hands == 0
    assert empty_det.masks is None
    assert empty_det.boxes.shape == (0, 4)
    assert empty_det.scores.shape == empty_det.sides.shape == empty_det.visibility.shape == empty_det.uncertainty.shape == (0,)

    with pytest.raises(ContractValidationError, match="model_revision"):
        adapter.admit(make_hands_request("bad-revision", model_revision=HANDS_REV))


def test_hands_admission_rejects_revision_mismatch() -> None:
    adapter, _ = make_hands_adapter()
    with pytest.raises(ContractValidationError, match="model_revision"):
        adapter.admit(make_hands_request("bad", model_revision="other-rev"))


def test_hands_results_carry_only_configured_revision() -> None:
    adapter, _ = make_hands_adapter()
    responses = asyncio.run(adapter.infer_batch([adapter.admit(make_hands_request("a"))]))
    assert responses[0].result.model_revision == HANDS_REV


def test_hands_invalid_binary_is_isolated_at_admission() -> None:
    adapter, _ = make_hands_adapter()
    with pytest.raises(ContractValidationError):
        adapter.admit(HandsDetectRequest(
            ownership=Ownership("r", "j", "i", "s", "src"),
            rgb=TensorPayload(data=b"short", shape=(H, W, 3), dtype="uint8"),
            spatial=SpatialMetadata(ImageSize(W, H), ImageSize(W, H), "RGB", PixelTransform.identity()),
            model_revision=HANDS_REV,
        ))


def test_hands_assert_one_forward_rejects_oversized_callback() -> None:
    policy = BatchPolicy(max_batch_size=4, batch_wait_timeout_s=0.01, max_queued_requests=2)
    with pytest.raises(ContractValidationError, match="multiple forwards"):
        assert_one_forward([1, 2, 3, 4, 5], policy=policy)


# --------------------------------------------------------------------------- wilor


class FakeWiLoRBackend:
    """Fake WiLoR returning real-shaped MANO outputs for [N,3,256,256] crops."""

    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    def reconstruct(self, crops, right, box_center, box_size, img_size):
        import torch  # noqa: F401  (not needed; pure numpy)
        self.calls += 1
        n = crops.shape[0]
        self.batch_sizes.append(n)
        return {
            "pred_mano_params": {
                "global_orient": np.broadcast_to(np.eye(3, dtype=np.float32), (n, 1, 3, 3)).copy(),
                "hand_pose": np.zeros((n, 15, 3, 3), dtype=np.float32),
                "betas": np.zeros((n, 10), dtype=np.float32),
            },
            "pred_vertices": np.random.RandomState(0).randn(n, 778, 3).astype(np.float32) * 0.05,
            "pred_keypoints_3d": np.random.RandomState(1).randn(n, 21, 3).astype(np.float32) * 0.05,
            "pred_cam": np.array([[1.0, 0.0, 0.0]] * n, dtype=np.float32),
            "pred_cam_t_full": np.array([[0.1, 0.2, 0.5]] * n, dtype=np.float32),
            "pred_keypoints_2d": np.zeros((n, 21, 2), dtype=np.float32),
            "focal_length": 5000.0,
            "bbox_shape": [192, 256],
        }


def make_wilor_adapter(backend: FakeWiLoRBackend | None = None, *, config: WiLoRModelConfig | None = None) -> tuple[WiLoRAdapter, FakeWiLoRBackend]:
    backend = backend or FakeWiLoRBackend()
    return WiLoRAdapter(config or make_wilor_config(), backend_factory=lambda _: backend), backend


def test_wilor_contract_rejects_wrong_crop_shape() -> None:
    with pytest.raises(ContractValidationError, match="crop"):
        WiLoRReconstructRequest(
            ownership=Ownership("r", "j", "i", "s", "src"),
            crop=TensorPayload(data=b"x" * 100, shape=(3, 128, 128), dtype="float32"),
            handedness=HandSide.RIGHT, box_center=(1.0, 2.0), box_size=10.0, img_size=(100.0, 100.0),
            model_revision=WILOR_REV,
        )


def test_wilor_contract_rejects_non_float_crop() -> None:
    with pytest.raises(ContractValidationError, match="float32"):
        WiLoRReconstructRequest(
            ownership=Ownership("r", "j", "i", "s", "src"),
            crop=TensorPayload(data=b"x" * (3 * 256 * 256), shape=(3, 256, 256), dtype="uint8"),
            handedness=HandSide.RIGHT, box_center=(1.0, 2.0), box_size=10.0, img_size=(100.0, 100.0),
            model_revision=WILOR_REV,
        )


def test_wilor_one_forward_over_true_batch_and_mano_shapes() -> None:
    backend = FakeWiLoRBackend()
    adapter, backend = make_wilor_adapter(backend)
    responses = asyncio.run(adapter.reconstruct_batch([
        adapter.admit(make_wilor_request("a", job_id="job-a")),
        adapter.admit(make_wilor_request("b", job_id="job-b", handedness=HandSide.LEFT)),
        adapter.admit(make_wilor_request("c", job_id="job-c")),
    ]))
    # Exactly one forward over [N=3,3,256,256].
    assert backend.calls == 1
    assert backend.batch_sizes == [3]
    assert all(r.result is not None for r in responses)
    trace = responses[0].result.trace
    assert trace.forward_count == 1
    assert trace.request_count == 3
    # Ownership split.
    assert [r.ownership.request_id for r in responses] == ["a", "b", "c"]
    assert [r.ownership.job_id for r in responses] == ["job-a", "job-b", "job-c"]
    # MANO shapes.
    mano = responses[0].result.mano
    assert mano.global_orient.shape == (1, 3, 3)
    assert mano.hand_pose.shape == (15, 3, 3)
    assert mano.betas.shape == (10,)
    assert mano.vertices.shape == (778, 3)
    assert mano.joints.shape[0] == 21 and mano.joints.shape[1] == 3
    assert mano.cam_t_full.shape == (3,)
    assert mano.pred_cam.shape == (3,)
    assert mano.n_vertices == 778
    # Finite MANO state.
    verts = np.frombuffer(mano.vertices.data, dtype="float32").reshape(778, 3)
    assert np.isfinite(verts).all()
    # Provenance: handedness preserved per owner.
    assert responses[0].result.handedness == HandSide.RIGHT
    assert responses[1].result.handedness == HandSide.LEFT


def test_wilor_left_surface_is_reflected_like_upstream_demo() -> None:
    """WiLoR outputs canonical right-hand geometry; left input must mirror x."""
    adapter, _ = make_wilor_adapter()
    right = asyncio.run(adapter.reconstruct_batch([adapter.admit(make_wilor_request("a", handedness=HandSide.RIGHT))]))[0]
    left = asyncio.run(adapter.reconstruct_batch([adapter.admit(make_wilor_request("b", handedness=HandSide.LEFT))]))[0]
    right_vertices = np.frombuffer(right.result.mano.vertices.data, dtype=np.float32).reshape(778, 3)
    left_vertices = np.frombuffer(left.result.mano.vertices.data, dtype=np.float32).reshape(778, 3)
    assert np.allclose(left_vertices[:, 0], -right_vertices[:, 0])
    assert np.allclose(left_vertices[:, 1:], right_vertices[:, 1:])


def test_wilor_admission_rejects_revision_mismatch() -> None:
    adapter, _ = make_wilor_adapter()
    with pytest.raises(ContractValidationError, match="model_revision"):
        adapter.admit(make_wilor_request("bad", model_revision="other-rev"))


def test_wilor_results_carry_only_configured_revision() -> None:
    adapter, _ = make_wilor_adapter()
    responses = asyncio.run(adapter.reconstruct_batch([adapter.admit(make_wilor_request("a"))]))
    assert responses[0].result.model_revision == WILOR_REV


def test_wilor_trace_is_monotonic() -> None:
    adapter, _ = make_wilor_adapter()
    responses = asyncio.run(adapter.reconstruct_batch([adapter.admit(make_wilor_request("a"))]))
    t = responses[0].result.trace
    assert t.admitted_monotonic_s <= t.dispatched_monotonic_s <= t.forward_started_monotonic_s <= t.completed_monotonic_s


# --------------------------------------------------------------------------- transport


def test_hands_multipart_round_trip() -> None:
    rgb = np.full((H, W, 3), 200, dtype=np.uint8)
    metadata = {"ownership": {"request_id": "r1"}, "model_revision": HANDS_REV, "spatial": {"source_size": {"width": W, "height": H}, "model_size": {"width": W, "height": H}, "color_space": "RGB", "pixel_transform": PixelTransform.identity().to_wire()}}
    body, content_type = build_multipart_request_fields(metadata, {"rgb": (rgb.tobytes(), rgb.shape, "uint8")})
    meta, fields = parse_multipart_request_fields(body, content_type)
    assert meta["model_revision"] == HANDS_REV
    assert "rgb" in fields
    assert fields["rgb"][1] == (H, W, 3)
    assert fields["rgb"][2] == "uint8"
    assert np.frombuffer(fields["rgb"][0], dtype=np.uint8).tolist() == rgb.flatten().tolist()


def test_wilor_multipart_round_trip_with_crop_field() -> None:
    crop = np.zeros((3, 256, 256), dtype=np.float32)
    metadata = {"ownership": {"request_id": "r1"}, "model_revision": WILOR_REV, "handedness": 1, "box_center": [100.0, 200.0], "box_size": 300.0, "img_size": [1920.0, 1080.0]}
    body, content_type = build_multipart_request_fields(metadata, {"crop": (crop.tobytes(), crop.shape, "float32")})
    meta, fields = parse_multipart_request_fields(body, content_type)
    assert meta["handedness"] == 1
    assert "crop" in fields
    assert fields["crop"][1] == (3, 256, 256)
    assert fields["crop"][2] == "float32"


def test_hands_multipart_response_round_trip_preserves_arrays() -> None:
    boxes = np.array([[1, 2, 3, 4]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    sides = np.array([1], dtype=np.uint8)
    masks = np.zeros((1, H, W), dtype=np.uint8)
    vis = np.array([0.5], dtype=np.float32)
    unc = np.array([0.1], dtype=np.float32)
    arrays = {
        "boxes": (boxes.tobytes(), boxes.shape, "float32"),
        "scores": (scores.tobytes(), scores.shape, "float32"),
        "sides": (sides.tobytes(), sides.shape, "uint8"),
        "masks": (masks.tobytes(), masks.shape, "uint8"),
        "visibility": (vis.tobytes(), vis.shape, "float32"),
        "uncertainty": (unc.tobytes(), unc.shape, "float32"),
    }
    body, content_type = build_multipart_response({"ok": True}, arrays)
    meta, parsed = parse_multipart_response(body, content_type)
    assert meta["ok"] is True
    assert set(parsed.keys()) == set(arrays.keys())
    assert parsed["masks"][1] == (1, H, W)
    assert parsed["sides"][2] == "uint8"


# --------------------------------------------------------------------------- lifecycle


def test_gpu1_hands_uses_ray_serve_hands_interpreter() -> None:
    g1 = hands_gpu_group()
    assert g1.gpu_id == 1
    assert g1.adapter_implemented is True
    assert g1.interpreter.endswith("ray_serve_hands/bin/python"), g1.interpreter
    assert g1.logical_apis == ("hands.detect",)


def test_gpu1_lifecycle_records_native_gpu_ports_and_cpu_cap() -> None:
    g1 = hands_gpu_group()
    lc = g1.lifecycle
    assert lc.gpu_id == 1
    assert lc.num_gpus == 1
    assert lc.num_cpus == 4
    assert lc.ray_version == RAY_VERSION
    cmd = lc.startup_command("ego-hands-wilor")
    assert "CUDA_VISIBLE_DEVICES=1" in cmd
    assert "--num-gpus=1" in cmd
    assert "--num-cpus=4" in cmd
    # Ray 2.55's `ray start --head` rejects this old, unsupported flag.
    assert "--cluster-name=" not in cmd
    # Components 27000-27006, workers 27100-27131, HTTP 28001.
    assert lc.ports.gcs_port == 27000
    assert lc.ports.dashboard_port == 27004
    assert lc.ports.serve_http_port == 28001
    workers = [int(p) for p in lc.ports.worker_port_list.split(",")]
    assert workers[0] == 27100
    assert workers[-1] == 27131
    assert len(workers) == 32


def test_gpu1_serve_config_points_at_hands_deployment_module() -> None:
    config = hands_wilor_serve_config()
    app_spec = config["applications"][0]
    assert app_spec["import_path"] == "ego_annotation.serving.hands_deployment:hands_app"
    deployment = app_spec["deployments"][0]
    assert deployment["name"] == "hands_wilor"
    assert deployment["ray_actor_options"] == {"num_gpus": 1}
    assert deployment["num_replicas"] == 1


def test_all_gpu_groups_remain_native_num_gpus() -> None:
    for group in COMMITTED_GPU_GROUPS:
        assert group.ray_actor_options == {"num_gpus": 1}


def test_gpu1_ports_disjoint_from_all_clusters() -> None:
    all_ports: list[int] = []
    for group in COMMITTED_GPU_GROUPS:
        all_ports.extend(group.lifecycle.ports.all_ports())
    assert len(set(all_ports)) == len(all_ports)


# --------------------------------------------------------------------------- deployment import-path


def test_hands_deployment_module_exposes_bound_application() -> None:
    import importlib.util
    import os

    spec = importlib.util.find_spec("ego_annotation.serving.hands_deployment")
    assert spec is not None
    source_path = os.path.join(os.path.dirname(spec.origin), "hands_deployment.py")
    with open(source_path) as handle:
        source = handle.read()
    assert "from ray import serve" in source
    assert "@serve.deployment(" in source
    # FastAPI ingress, rather than a direct deployment __call__, is the ASGI path
    # that preserves Starlette Response bytes through Ray Serve.
    assert "@serve.ingress(build_hands_api)" in source
    assert "serve.get_replica_context().servable_object" in source
    assert "@serve.batch(" in source
    assert "num_gpus=1" in source
    assert "hands_app: Any = HandsWiLoRDeployment.bind()" in source
    # Both logical APIs are separate batched methods on one replica.
    assert "_batched_detect" in source
    assert "_batched_reconstruct" in source


def test_deployment_errors_are_binary_starlette_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deployment must return an ASGI Response, not a dict containing bytes.

    A dict makes Ray Serve use its JSON encoder, which corrupts multipart byte
    payloads. Stub only the decorators to exercise the deployment's transport
    function without importing the heavyweight Ray runtime.
    """
    class FakeServe:
        @staticmethod
        def deployment(**_kwargs: Any):
            def decorate(cls: type[Any]) -> type[Any]:
                cls.bind = classmethod(lambda bound_cls: bound_cls)  # type: ignore[attr-defined]
                return cls
            return decorate

        @staticmethod
        def batch(**_kwargs: Any):
            return lambda func: func

        @staticmethod
        def ingress(_app: Any):
            return lambda cls: cls

    fake_ray = types.ModuleType("ray")
    fake_ray.serve = FakeServe
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    sys.modules.pop("ego_annotation.serving.hands_deployment", None)
    try:
        deployment = importlib.import_module("ego_annotation.serving.hands_deployment")
        response = deployment._error_response("bad request")
        assert response.__class__.__module__.startswith("starlette.")
        assert response.headers["content-type"].startswith("multipart/form-data;")
        metadata, arrays = parse_multipart_response(response.body, response.headers["content-type"])
        assert metadata["error"] == {"code": "validation", "message": "bad request", "retryable": False}
        assert arrays == {}

        # The deployment's default revision is the YOLO-only contract, and its
        # multipart response omits masks instead of emitting a fabricated empty tensor.
        monkeypatch.delenv("EGO_HANDS_REVISION", raising=False)
        assert deployment._hands_config_from_env().model_revision == HANDS_YOLO_REV
        adapter, _ = make_hands_adapter(config=make_hands_config(model_revision=HANDS_YOLO_REV))
        result = asyncio.run(adapter.infer_batch([
            adapter.admit(make_hands_request("wire", model_revision=HANDS_YOLO_REV)),
        ]))[0]
        wire = deployment._hands_response_to_multipart_wire(result)
        wire_metadata, wire_arrays = parse_multipart_response(wire.body, wire.headers["content-type"])
        assert set(wire_arrays) == {"boxes", "scores", "sides", "visibility", "uncertainty"}
        assert "masks" not in wire_metadata["result"]["detection"]
        assert wire_arrays["boxes"][1:] == ((1, 4), "float32")
        assert wire_arrays["scores"][1:] == ((1,), "float32")
        assert wire_arrays["sides"][1:] == ((1,), "uint8")
        assert wire_arrays["visibility"][1:] == ((1,), "float32")
        assert wire_arrays["uncertainty"][1:] == ((1,), "float32")
    finally:
        sys.modules.pop("ego_annotation.serving.hands_deployment", None)


def test_ordinary_hands_wilor_imports_do_not_require_ray() -> None:
    from ego_annotation.serving.hands import HandsAdapter  # noqa: F401
    from ego_annotation.serving.wilor import WiLoRAdapter  # noqa: F401
    from ego_annotation.serving.transport import build_multipart_request_fields  # noqa: F401
