from __future__ import annotations

import io
import json
from dataclasses import replace
import urllib.error
from email.parser import BytesParser
from email.policy import default as default_email_policy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

import ego_annotation.api_backend as api_backend_module
from ego_annotation.api_backend import ApiBackend, ApiBackendConfig, ApiProtocolError, ApiTransportError, CapabilityMismatchError, DroidRecoveryPolicy
from ego_annotation.api_routes import route_for
from ego_annotation.api_run_config import DEFAULT_FROZEN_SINGLE_VIDEO, RunPreflightError, SourceProbe, validate_preflight
from ego_annotation.live_wire import LiveRouteResponse, LiveWireError, decode_live_response, encode_live_request
from ego_annotation.multipart import decode_future_multipart, encode_future_multipart
from ego_annotation.scripted import AlgorithmRequest, DepthEvidence, DroidPixelGeometry, FrameTimelineMetadata, NativeWorkDescription, StageMetadata, pack_native_sensor_depth
from ego_annotation.typed_contracts import (
    BinaryAsset,
    CosmosGeneration,
    CosmosInput,
    CosmosMessage,
    CropTransform,
    DroidCapabilities,
    DroidCreateInput,
    DroidFinalizeInput,
    DroidPushInput,
    HandSide,
    HandsInput,
    HaworObservation,
    HaworTrackInput,
    InfillerFrame,
    InfillerInput,
    Ownership,
    SpatialTransform,
    TypedTensor,
    UniDepthInput,
    WiLoRInput,
)
from ego_annotation.typed_dag import AdapterConfigurationError, DagLane, DroidRetryStrategy, RunnerConfig, TypedDag


FROZEN_GENERIC_FORBIDDEN = {"protocol", "status", "request_id", "algorithm_id", "case_id", "item_id", "source_id", "input"}
FROZEN_SCHEMA = "ego.model-service.v1"


class FrozenResponse:
    status = 200

    def __init__(self, body: bytes, content_type: str) -> None:
        self._body = body
        self.headers = {"Content-Type": content_type}

    def read(self) -> bytes:
        return self._body


def wire_owner(stage: str) -> dict[str, object]:
    return {
        "request_id": f"case:item:{stage}",
        "job_id": "case",
        "item_id": "item",
        "stage_id": stage,
        "source_id": "video.mp4",
        "schema_version": FROZEN_SCHEMA,
        "source_timestamp_s": None,
        "submitted_at": "1970-01-01T00:00:00Z",
    }


def typed_owner(stage: str) -> Ownership:
    return Ownership("case", "item", "video.mp4", "fixture-owner", stage)


def spatial(width: int = 16, height: int = 16) -> SpatialTransform:
    return SpatialTransform("source", width, height, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), "camera")


def tensor(array: np.ndarray, tag: str, *, order: str = "thwc", units: str = "unitless") -> TypedTensor:
    return TypedTensor(array, units, "source", order, tag, {"source_artifact_id": "fixture"})


def timeline(n: int = 2) -> FrameTimelineMetadata:
    return FrameTimelineMetadata("video.mp4", tuple(range(n)), tuple(index / 25.0 for index in range(n)), width_px=16, height_px=16, fps=25.0)


def request(stage: str, value: object, cap: int, shape: tuple[int, ...], *, revision: str = "fixture-revision", axis: int | None = 0) -> AlgorithmRequest[Any]:
    return AlgorithmRequest(
        stage,
        revision,
        "case",
        "item",
        "video.mp4",
        timeline(),
        StageMetadata(stage, "fixture-owner", stage, revision),
        NativeWorkDescription(stage, stage, axis, shape[axis] if axis is not None else 1, cap, shape),
        value,
    )


def parse_like_frozen_server(body: bytes, content_type: str) -> tuple[dict[str, object], dict[str, tuple[bytes, tuple[int, ...] | None, str | None, Mapping[str, str]]]]:
    """Independent fixture parser mirroring frozen email/Content-Disposition behavior."""
    message = BytesParser(policy=default_email_policy).parsebytes(f"Content-Type: {content_type}\r\n\r\n".encode() + body)
    assert message.is_multipart()
    metadata: dict[str, object] | None = None
    parts: dict[str, tuple[bytes, tuple[int, ...] | None, str | None, Mapping[str, str]]] = {}
    for part in message.iter_parts():
        name = str(part.get_param("name", header="content-disposition"))
        payload = bytes(part.get_payload(decode=True))
        if name == "metadata":
            metadata = json.loads(payload.decode("utf-8"))
            continue
        shape_text = part.get_param("shape", header="content-disposition")
        dtype = part.get_param("dtype", header="content-disposition")
        shape = tuple(int(item) for item in str(shape_text).split(",")) if shape_text is not None else None
        extras = {key: str(part.get_param(key, header="content-disposition")) for key in ("kind", "media_type", "source_index") if part.get_param(key, header="content-disposition") is not None}
        parts[name] = (payload, shape, str(dtype) if dtype is not None else None, extras)
    assert metadata is not None
    return metadata, parts


def frozen_multipart(metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray]) -> FrozenResponse:
    boundary = "frozen-fixture-boundary"
    out = bytearray()
    delim = b"--" + boundary.encode()
    out += delim + b"\r\nContent-Disposition: form-data; name=\"metadata\"\r\nContent-Type: application/json\r\n\r\n"
    out += json.dumps(metadata, separators=(",", ":")).encode() + b"\r\n"
    for name, array in arrays.items():
        arr = np.ascontiguousarray(array)
        shape = ",".join(str(dim) for dim in arr.shape)
        out += delim + f"\r\nContent-Disposition: form-data; name=\"{name}\"; shape=\"{shape}\"; dtype=\"{arr.dtype.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode()
        out += arr.tobytes() + b"\r\n"
    out += delim + b"--\r\n"
    return FrozenResponse(bytes(out), f"multipart/form-data; boundary={boundary}")


def assert_frozen_request(req: AlgorithmRequest[Any], expected_keys: set[str], expected_parts: set[str]) -> tuple[dict[str, object], dict[str, Any]]:
    body, content_type, metadata_from_adapter, names = encode_live_request(req)
    metadata, parts = parse_like_frozen_server(body, content_type)
    assert metadata == metadata_from_adapter
    assert set(metadata) == expected_keys
    assert not FROZEN_GENERIC_FORBIDDEN.intersection(metadata)
    assert set(parts) == expected_parts == set(names)
    for _name, (payload, shape, dtype, _extras) in parts.items():
        assert payload
        if shape is not None:
            assert dtype is not None
            assert len(payload) == int(np.prod(shape)) * np.dtype(dtype).itemsize
    return metadata, parts


def make_unidepth() -> AlgorithmRequest[Any]:
    value = UniDepthInput(typed_owner("unidepth.infer"), tensor(np.zeros((1, 16, 16, 3), np.uint8), "rgb"), (0,), (0.0,), spatial())
    return request("unidepth.infer", value, 8, (1, 16, 16, 3))


def make_hands() -> AlgorithmRequest[Any]:
    value = HandsInput(typed_owner("hands.detect"), tensor(np.zeros((1, 16, 16, 3), np.uint8), "rgb"), (0,), (0.0,), spatial())
    return request("hands.detect", value, 8, (1, 16, 16, 3))


def make_wilor() -> AlgorithmRequest[Any]:
    crop = CropTransform((8.0, 8.0), 12.0, spatial(), HandSide.LEFT)
    value = WiLoRInput(typed_owner("wilor.reconstruct"), tensor(np.zeros((1, 3, 256, 256), np.float32), "wilor_crop", order="bcyx"), (crop,), tensor(np.eye(3, dtype=np.float32), "K", order="yx", units="pixels"))
    return request("wilor.reconstruct", value, 16, (1, 3, 256, 256))


def test_split_stage_direct_urls_keep_hands_on_28001_and_send_wilor_to_28004() -> None:
    observed_urls: list[str] = []

    def opener(http_request: Any, _timeout: float) -> FrozenResponse:
        observed_urls.append(http_request.full_url)
        return FrozenResponse(b"accepted", "application/json")

    for stage_id, stage_request in (("hands.detect", make_hands()), ("wilor.reconstruct", make_wilor())):
        backend = ApiBackend(ApiBackendConfig.for_stage(stage_id), opener=opener)
        backend._post(route_for(stage_id), b"request", "application/octet-stream", request=stage_request)

    assert observed_urls == [
        "http://127.0.0.1:28001/hands.detect",
        "http://127.0.0.1:28004/wilor.reconstruct",
    ]


def make_hawor() -> AlgorithmRequest[Any]:
    transforms = tuple(CropTransform((8.0, 8.0), 12.0, spatial(), HandSide.LEFT) for _ in range(16))
    observations = tuple(HaworObservation(index, index / 25.0, HandSide.LEFT, "visible", 0.9) for index in range(16))
    value = HaworTrackInput(
        typed_owner("hawor.infer_tracks"),
        tensor(np.zeros((1, 16, 3, 256, 256), np.float32), "hawor_crop", order="btcyx"),
        (tuple(range(16)),), transforms,
        tensor(np.ones((1, 16, 1), np.float32), "observations", order="btf"),
        tensor(np.ones((16, 16, 16), np.float32), "depth", order="tyx", units="metres"),
        tensor(np.eye(3, dtype=np.float32), "K", order="yx", units="pixels"),
        tensor(np.repeat(np.eye(4, dtype=np.float32)[None], 16, axis=0), "poses", order="tyx", units="metres"),
        tensor(np.arange(16, dtype=np.float64) / 25.0, "timestamps", order="t", units="seconds"), observations,
    )
    return request("hawor.infer_tracks", value, 4, (1, 16, 3, 256, 256))


def make_infiller() -> AlgorithmRequest[Any]:
    root = tuple(tuple(float(v) for v in row) for row in np.eye(3))
    pose = tuple((0.0, 0.0, 0.0) for _ in range(15))
    frames = tuple(InfillerFrame(index, index / 25.0, side, root, pose, (0.0, 0.0, 1.0), (0.0,) * 10, True, 0.01) for index in range(120) for side in (HandSide.LEFT, HandSide.RIGHT))
    value = InfillerInput(
        typed_owner("hawor_infiller.fill"),
        tensor(np.zeros((120, 218), np.float32), "infiller_window", order="td"),
        tensor(np.ones((120, 2), np.uint8), "observation_mask", order="th"),
        tensor(np.arange(120, dtype=np.float64) / 25.0, "timestamps", order="t", units="seconds"),
        tensor(np.repeat(np.eye(4, dtype=np.float32)[None], 120, axis=0), "poses", order="tyx", units="metres"),
        tensor(np.eye(3, dtype=np.float32), "K", order="yx", units="pixels"), frames,
    )
    return request("hawor_infiller.fill", value, 2, (1, 120, 218))


def make_droid_create(*, strict: bool) -> AlgorithmRequest[Any]:
    s = spatial()
    value = DroidCreateInput(typed_owner("droid.create_session"), timeline(), (900.0, 1100.0, 8.0, 8.0), s, s, (2, 2), require_rgbd_capability=strict, allow_monocular_droid_smoke=not strict)
    return request("droid.create_session", value, 1, (1,), revision="droid-v1", axis=None)


def native_depth(frame: int):
    geometry = DroidPixelGeometry(np.array([[900.0, 0.0, 8.0], [0.0, 1100.0, 8.0], [0.0, 0.0, 1.0]]), np.eye(3), np.diag([1 / 8, 1 / 8, 1]), (16, 16), (2, 2))
    return pack_native_sensor_depth(DepthEvidence(np.full((16, 16), 2.0 + frame, np.float32), np.eye(3), f"depth-{frame}", frame, "source"), geometry)


def make_droid_push(session: str, frame: int) -> AlgorithmRequest[Any]:
    value = DroidPushInput(typed_owner("droid.push_frame"), session, frame, frame / 25.0, tensor(np.full((16, 16, 3), frame, np.uint8), "rgb", order="hwc"), native_depth(frame), (900.0, 1100.0, 8.0, 8.0), tensor(np.ones((16, 16), np.uint8), "static_mask", order="yx"), require_rgbd_capability=False, allow_monocular_droid_smoke=True)
    return request("droid.push_frame", value, 8, (1,), revision="droid-v1", axis=None)


def make_droid_finalize(session: str) -> AlgorithmRequest[Any]:
    value = DroidFinalizeInput(typed_owner("droid.finalize"), session, require_rgbd_capability=False, allow_monocular_droid_smoke=True)
    return request("droid.finalize", value, 1, (1,), revision="droid-v1", axis=None)


def test_internal_generic_codec_binary_roundtrip_and_corruption_rejection() -> None:
    value = tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4), "metric", order="tyx", units="metres")
    body, content_type = encode_future_multipart({"purpose": "future-custom-codec"}, {"depth": value})
    decoded = decode_future_multipart(body, content_type)
    assert decoded.parts["depth"].data == value.canonical_bytes
    assert decoded.parts["depth"].descriptor["canonical_tensor_digest"] == value.canonical_tensor_digest
    corrupted = bytearray(value.canonical_bytes)
    corrupted[0] ^= 1
    bad_body = body.replace(value.canonical_bytes, bytes(corrupted), 1)
    with pytest.raises(Exception, match="digest"):
        decode_future_multipart(bad_body, content_type)


def wilor_full_camera_response(*, keypoint_offset_px: float = 0.0, focal_length_px: float | None = 100.0, keypoint_count: int = 778) -> tuple[FrozenResponse, dict[str, np.ndarray]]:
    vertices = np.zeros((778, 3), dtype=np.float32)
    vertices[:, 0] = np.linspace(-0.04, 0.06, 778)
    vertices[:, 1] = np.linspace(-0.03, 0.05, 778)
    vertices[:, 2] = np.linspace(-0.02, 0.02, 778)
    joints = vertices[:21].copy()
    cam_t_full = np.asarray([0.05, -0.02, 1.2], dtype=np.float32)
    focal = 100.0 if focal_length_px is None else focal_length_px
    translated_surface = vertices[:keypoint_count] + cam_t_full[None]
    keypoints = focal * translated_surface[:, :2] / translated_surface[:, 2:3] + np.asarray([8.0, 8.0], dtype=np.float32)
    keypoints = keypoints.astype(np.float32)
    keypoints[:, 0] += np.float32(keypoint_offset_px)
    arrays = {
        "global_orient": np.eye(3, dtype=np.float32)[None],
        "hand_pose": np.repeat(np.eye(3, dtype=np.float32)[None], 15, axis=0),
        "betas": np.zeros(10, dtype=np.float32),
        "vertices": vertices,
        "joints": joints,
        "cam_t_full": cam_t_full,
        "pred_cam": np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
        "keypoints_2d": keypoints,
        "confidence": np.asarray([0.9], dtype=np.float32),
        "uncertainty": np.asarray([0.02], dtype=np.float32),
    }
    mano = {name: wire_tensor(array) for name, array in arrays.items()}
    if focal_length_px is not None:
        mano["focal_length"] = focal_length_px
    mano["n_vertices"] = 778
    return frozen_multipart({"ownership": wire_owner("wilor.reconstruct"), "result": {"mano": mano}}, arrays), arrays


def test_wilor_full_camera_projection_consumes_returned_keypoints_and_cam_contract() -> None:
    response, arrays = wilor_full_camera_response(keypoint_offset_px=5.0e-4)
    backend = ApiBackend(ApiBackendConfig.for_stage("wilor.reconstruct"), opener=lambda _req, _timeout: response)

    output = backend.execute(make_wilor()).output
    mano = output.mano
    translated_vertices = arrays["vertices"] + arrays["cam_t_full"][None]
    expected_vertices = 100.0 * translated_vertices[:, :2] / translated_vertices[:, 2:3] + np.asarray([8.0, 8.0])

    translated_joints = arrays["joints"] + arrays["cam_t_full"][None]
    expected_joints = 100.0 * translated_joints[:, :2] / translated_joints[:, 2:3] + np.asarray([8.0, 8.0])
    np.testing.assert_array_equal(mano.vertices_source_px.array[0], arrays["keypoints_2d"])
    np.testing.assert_allclose(mano.vertices_source_px.array[0], expected_vertices, atol=1.0e-3)
    np.testing.assert_allclose(mano.joints_source_px.array[0], expected_joints, atol=1.0e-5)
    np.testing.assert_array_equal(mano.cam_t_full.array[0], arrays["cam_t_full"])
    assert mano.vertices_source_px.provenance["projection"] == "wilor_returned_full_camera_focal_cam_t_source_image_center"
    assert mano.vertices_source_px.provenance["returned_surface_keypoints_validated"] is True
    assert mano.vertices_source_px.provenance["metric_world_lift"] == "unchanged_unidepth_wrist_ray_lift"


def test_wilor_full_camera_projection_rejects_inconsistent_returned_keypoints() -> None:
    response, _ = wilor_full_camera_response(keypoint_offset_px=2.0)
    backend = ApiBackend(ApiBackendConfig.for_stage("wilor.reconstruct"), opener=lambda _req, _timeout: response)

    with pytest.raises(ApiProtocolError, match="keypoints_2d disagree"):
        backend.execute(make_wilor())


def test_wilor_full_camera_projection_rejects_joint_cardinality_keypoints() -> None:
    response, _ = wilor_full_camera_response(keypoint_count=21)
    backend = ApiBackend(ApiBackendConfig.for_stage("wilor.reconstruct"), opener=lambda _req, _timeout: response)

    with pytest.raises(ApiProtocolError, match="778-vertex MANO surface axis"):
        backend.execute(make_wilor())


def test_wilor_full_camera_projection_rejects_missing_focal_metadata() -> None:
    response, _ = wilor_full_camera_response(focal_length_px=None)
    backend = ApiBackend(ApiBackendConfig.for_stage("wilor.reconstruct"), opener=lambda _req, _timeout: response)

    with pytest.raises(ApiProtocolError, match="focal_length"):
        backend.execute(make_wilor())


def test_golden_request_keys_and_parts_for_every_frozen_route() -> None:
    assert_frozen_request(make_unidepth(), {"ownership", "spatial", "model_revision", "options", "rgb_shape", "rgb_dtype"}, {"rgb"})
    assert_frozen_request(make_hands(), {"ownership", "spatial", "model_revision", "options"}, {"rgb"})
    assert_frozen_request(make_wilor(), {"ownership", "handedness", "box_center", "box_size", "img_size", "source_K_px", "model_revision", "options"}, {"crop"})
    assert_frozen_request(make_droid_create(strict=True), {"ownership", "camera", "image_shape", "model_revision", "options"}, set())
    assert_frozen_request(make_droid_push("session", 0), {"ownership", "session_id", "frame_id", "source_timestamp_s", "rgb", "static_confidence_mask", "depth_m", "model_revision"}, {"rgb", "static_confidence_mask", "depth_m"})
    assert_frozen_request(make_droid_finalize("session"), {"ownership", "session_id", "model_revision"}, set())
    assert_frozen_request(make_hawor(), {"ownership", "track_id", "side", "crop_batch", "crop_transforms", "observations", "unidepth", "droid_evidence", "model_revision", "options"}, {"crop_batch", "droid_poses", "droid_timestamps"})
    assert_frozen_request(make_infiller(), {"ownership", "window_id", "frames", "droid_evidence", "unidepth", "model_revision", "options"}, {"droid_poses", "droid_timestamps"})
    cosmos = CosmosInput(typed_owner("cosmos3.reason"), "describe", (), CosmosGeneration(512), (BinaryAsset(b"jpeg", "image/jpeg", "frame-0", (0,)),), (0,))
    metadata, parts = assert_frozen_request(request("cosmos3.reason", cosmos, 1, (1,), axis=None), {"ownership", "prompt", "messages", "generation"}, {"media_0"})
    assert metadata["prompt"] == "describe" and metadata["messages"] == []
    assert metadata["generation"] == {"max_tokens": 512, "temperature": 0.0, "top_p": 1.0}
    assert parts["media_0"][3] == {"kind": "image", "media_type": "image/jpeg", "source_index": "0"}


def test_hawor_wire_uses_actual_track_side_source_geometry_and_calibration() -> None:
    req = make_hawor()
    source_spatial = SpatialTransform(
        "hawor-source-crop",
        1920,
        1080,
        ((1.5, 0.0, 600.0), (0.0, 1.25, 270.0), (0.0, 0.0, 1.0)),
        "source_pixels",
    )
    transforms = tuple(CropTransform((792.0, 430.0), 320.0, source_spatial, HandSide.RIGHT) for _ in range(16))
    observations = tuple(HaworObservation(index, index / 25.0, HandSide.RIGHT, "visible", 0.9) for index in range(16))
    K = tensor(np.asarray([[900.0, 0.0, 960.0], [0.0, 625.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float32), "K", order="yx", units="pixels")
    req = replace(req, input=replace(req.input, crop_transforms=transforms, observation_records=observations, unidepth_K_px=K))

    metadata, _ = assert_frozen_request(req, {"ownership", "track_id", "side", "crop_batch", "crop_transforms", "observations", "unidepth", "droid_evidence", "model_revision", "options"}, {"crop_batch", "droid_poses", "droid_timestamps"})

    assert metadata["side"] == "right"
    assert metadata["unidepth"] == {
        "K_px": [[900.0, 0.0, 960.0], [0.0, 625.0, 540.0], [0.0, 0.0, 1.0]],
        "img_focal": 750.0,
        "img_center": [960.0, 540.0],
        "source_size": {"width": 1920, "height": 1080},
        "metric_scale": 1.0,
        "source": "unidepth",
    }
    for crop in metadata["crop_transforms"]:
        assert crop["do_flip"] is False
        assert crop["img_focal"] == 750.0
        assert crop["img_center"] == [960.0, 540.0]
        assert crop["source_size"] == {"width": 1920, "height": 1080}
        source_to_model = np.asarray(crop["pixel_transform"]["source_to_model"])
        model_to_source = np.asarray(crop["pixel_transform"]["model_to_source"])
        np.testing.assert_allclose(source_to_model @ model_to_source, np.eye(3), atol=1.0e-9)
        assert not np.array_equal(source_to_model, model_to_source)


def test_droid_create_forwards_only_frozen_session_options() -> None:
    request_1440 = replace(
        make_droid_create(strict=True),
        options={"buffer": 1440, "filter_thresh": 8.0, "attempt": 0, "bounded_lower_filter_retry": True, "unrelated": "omit"},
    )
    metadata, _ = assert_frozen_request(request_1440, {"ownership", "camera", "image_shape", "model_revision", "options"}, set())
    assert metadata["options"] == {"buffer": 1440, "filter_thresh": 8.0}

    request_360 = replace(make_droid_create(strict=True), options={"buffer": 1024})
    metadata, _ = assert_frozen_request(request_360, {"ownership", "camera", "image_shape", "model_revision", "options"}, set())
    assert metadata["options"] == {"buffer": 1024}


def test_droid_create_rejects_non_frozen_option_types() -> None:
    with pytest.raises(LiveWireError, match="buffer.*frozen type int"):
        encode_live_request(replace(make_droid_create(strict=True), options={"buffer": True}))
    with pytest.raises(LiveWireError, match="filter_thresh.*frozen type float"):
        encode_live_request(replace(make_droid_create(strict=True), options={"filter_thresh": 8}))


def wire_tensor(array: np.ndarray) -> dict[str, object]:
    import base64
    return {"data_b64": base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii"), "shape": list(array.shape), "dtype": array.dtype.name}


def test_golden_original_response_parts_for_all_tensor_routes() -> None:
    cases: list[tuple[str, dict[str, object], dict[str, np.ndarray]]] = []
    depth = np.ones((4, 4), np.float32)
    K = np.eye(3, dtype=np.float32)
    cases.append(("unidepth.infer", {"depth_m": wire_tensor(depth), "K_px": wire_tensor(K), "confidence": wire_tensor(depth), "model_revision": "fixture"}, {"depth_m": depth, "K_px": K, "confidence": depth}))
    boxes, scores, sides = np.zeros((1, 4), np.float32), np.ones((1,), np.float32), np.zeros((1,), np.uint8)
    cases.append(("hands.detect", {"model_revision": "hands-yolo-v2", "detection": {"boxes": wire_tensor(boxes), "scores": wire_tensor(scores), "sides": wire_tensor(sides), "visibility": wire_tensor(scores), "uncertainty": wire_tensor(scores)}}, {"boxes": boxes, "scores": scores, "sides": sides, "visibility": scores, "uncertainty": scores}))
    wilor_arrays = {"global_orient": np.eye(3, dtype=np.float32)[None], "hand_pose": np.repeat(np.eye(3, dtype=np.float32)[None], 15, axis=0), "betas": np.zeros(10, np.float32), "vertices": np.zeros((778, 3), np.float32), "joints": np.zeros((16, 3), np.float32), "cam_t_full": np.zeros(3, np.float32), "pred_cam": np.zeros(3, np.float32), "keypoints_2d": np.zeros((16, 2), np.float32), "confidence": np.ones(1, np.float32), "uncertainty": np.zeros(1, np.float32)}
    cases.append(("wilor.reconstruct", {"mano": {name: wire_tensor(array) for name, array in wilor_arrays.items()}}, wilor_arrays))
    hawor_arrays = {"root_orient": np.repeat(np.eye(3, dtype=np.float32)[None], 16, axis=0), "hand_pose": np.repeat(np.eye(3, dtype=np.float32)[None, None], 16 * 15, axis=0).reshape(16, 15, 3, 3), "trans": np.zeros((16, 3), np.float32), "betas": np.zeros((16, 10), np.float32), "vertices": np.zeros((16, 778, 3), np.float32), "joints": np.zeros((16, 16, 3), np.float32), "observed": np.ones(16, np.uint8), "uncertainty": np.zeros(16, np.float32), "world_lift": np.repeat(np.eye(4, dtype=np.float32)[None], 16, axis=0)}
    cases.append(("hawor.infer_tracks", {**{name: wire_tensor(array) for name, array in hawor_arrays.items()}, "occlusion_state": ["visible"] * 16}, hawor_arrays))
    infiller_arrays = {"root_orient": np.repeat(np.eye(3, dtype=np.float32)[None, None], 2 * 120, axis=0).reshape(2, 120, 3, 3), "hand_pose": np.repeat(np.eye(3, dtype=np.float32)[None, None, None], 2 * 120 * 15, axis=0).reshape(2, 120, 15, 3, 3), "trans": np.zeros((2, 120, 3), np.float32), "betas": np.zeros((2, 120, 10), np.float32), "observed": np.ones((2, 120), np.uint8), "inferred": np.zeros((2, 120), np.uint8), "uncertainty": np.zeros((2, 120), np.float32), "timestamps_s": np.arange(120, dtype=np.float64) / 25.0}
    cases.append(("hawor_infiller.fill", {name: wire_tensor(array) for name, array in infiller_arrays.items()}, infiller_arrays))
    for stage, result, arrays in cases:
        response = frozen_multipart({"ownership": wire_owner(stage), "result": result}, arrays)
        decoded = decode_live_response(stage, typed_owner(stage), response.read(), response.headers["Content-Type"])
        assert set(decoded.parts) == set(arrays)


def test_compact_service_tensor_descriptors_decode_from_authenticated_multipart_parts() -> None:
    depth = np.ones((4, 4), np.float32)
    K = np.eye(3, dtype=np.float32)
    response = frozen_multipart(
        {
            "ownership": wire_owner("unidepth.infer"),
            "result": {
                "depth_m": {"shape": [4, 4], "dtype": "float32"},
                "K_px": {"shape": [3, 3], "dtype": "float32"},
                "confidence": {"shape": [4, 4], "dtype": "float32"},
            },
        },
        {"depth_m": depth, "K_px": K, "confidence": depth},
    )

    decoded = decode_live_response("unidepth.infer", typed_owner("unidepth.infer"), response.read(), response.headers["Content-Type"])

    np.testing.assert_array_equal(np.frombuffer(decoded.parts["depth_m"].data, dtype=np.float32).reshape(4, 4), depth)


def test_yolo_only_hands_response_decodes_without_mask_payload() -> None:
    boxes = np.asarray([[1.0, 2.0, 10.0, 12.0]], dtype=np.float32)
    scores = np.asarray([0.8], dtype=np.float32)
    sides = np.asarray([1], dtype=np.uint8)
    visibility = np.asarray([1.0], dtype=np.float32)
    uncertainty = np.asarray([0.2], dtype=np.float32)
    arrays = {
        "boxes": boxes,
        "scores": scores,
        "sides": sides,
        "visibility": visibility,
        "uncertainty": uncertainty,
    }
    response = frozen_multipart(
        {
            "ownership": wire_owner("hands.detect"),
            "result": {
                "model_revision": "hands-yolo-v2",
                "detection": {name: wire_tensor(array) for name, array in arrays.items()},
            },
        },
        arrays,
    )
    backend = ApiBackend(ApiBackendConfig.for_stage("hands.detect"), opener=lambda _req, _timeout: response)

    result, timing = backend.execute_timed(make_hands())
    output = result.output

    assert timing.available is True
    assert timing.total_wall_s >= timing.client_prepare_s
    assert timing.transport_wait_s >= 0.0
    assert timing.client_decode_postprocess_s >= 0.0
    assert output.model_revision == "hands-yolo-v2"
    assert not hasattr(output.detections, "masks")
    np.testing.assert_array_equal(output.detections.boxes_xyxy.array[0], boxes)
    np.testing.assert_array_equal(output.detections.visibility.array[0], visibility)
    np.testing.assert_array_equal(output.detections.uncertainty.array[0], uncertainty)


def test_infiller_frame_rejects_nonfinite_uncertainty() -> None:
    with pytest.raises(Exception, match="finite"):
        InfillerFrame(0, 0.0, HandSide.LEFT, tuple(tuple(float(v) for v in row) for row in np.eye(3)), tuple((0.0, 0.0, 0.0) for _ in range(15)), (0.0, 0.0, 1.0), (0.0,) * 10, False, float("inf"))


def test_api_backend_preserves_hawor_betas() -> None:
    arrays = {
        "root_orient": np.repeat(np.eye(3, dtype=np.float32)[None], 16, axis=0),
        "hand_pose": np.repeat(np.eye(3, dtype=np.float32)[None, None], 16 * 15, axis=0).reshape(16, 15, 3, 3),
        "trans": np.zeros((16, 3), np.float32),
        "betas": np.arange(160, dtype=np.float32).reshape(16, 10),
        "vertices": np.zeros((16, 778, 3), np.float32),
        "joints": np.zeros((16, 16, 3), np.float32),
        "observed": np.ones(16, np.uint8),
        "uncertainty": np.zeros(16, np.float32),
    }
    response = frozen_multipart(
        {
            "ownership": wire_owner("hawor.infer_tracks"),
            "result": {
                **{name: wire_tensor(array) for name, array in arrays.items()},
                "occlusion_state": ["visible"] * 16,
            },
        },
        arrays,
    )
    backend = ApiBackend(ApiBackendConfig.for_stage("hawor.infer_tracks"), opener=lambda _req, _timeout: response)

    output = backend.execute(make_hawor()).output

    assert output.betas.shape == (16, 10)
    np.testing.assert_array_equal(output.betas.array, arrays["betas"])


def test_route_response_shape_dtype_digest_mismatch_fails() -> None:
    depth = np.ones((4, 4), np.float32)
    result = {"depth_m": wire_tensor(depth), "K_px": wire_tensor(np.eye(3, dtype=np.float32)), "confidence": wire_tensor(depth)}
    result["depth_m"] = dict(result["depth_m"], dtype="float64")
    response = frozen_multipart({"ownership": wire_owner("unidepth.infer"), "result": result}, {"depth_m": depth, "K_px": np.eye(3, dtype=np.float32), "confidence": depth})
    with pytest.raises(LiveWireError, match="shape or dtype"):
        decode_live_response("unidepth.infer", typed_owner("unidepth.infer"), response.read(), response.headers["Content-Type"])


def test_cosmos_response_top_level_schema_error_ownership_and_generic_negative() -> None:
    owner = typed_owner("cosmos3.reason")
    error_body = json.dumps({"ownership": wire_owner("cosmos3.reason"), "result": None, "error": {"code": "validation", "message": "bad"}}).encode()
    with pytest.raises(LiveWireError, match="explicit error"):
        decode_live_response("cosmos3.reason", owner, error_body, "application/json")
    wrong = frozen_multipart({"ownership": wire_owner("wilor.reconstruct"), "result": {}, "error": None}, {})
    with pytest.raises(LiveWireError, match="ownership mismatch"):
        decode_live_response("cosmos3.reason", owner, wrong.read(), wrong.headers["Content-Type"])
    missing_error = frozen_multipart({"ownership": wire_owner("cosmos3.reason"), "result": {}}, {})
    with pytest.raises(LiveWireError, match="top-level"):
        decode_live_response("cosmos3.reason", owner, missing_error.read(), missing_error.headers["Content-Type"])
    wrapped = json.dumps({"metadata": {"ownership": wire_owner("cosmos3.reason"), "result": {}, "error": None}}).encode()
    with pytest.raises(LiveWireError, match="top-level"):
        decode_live_response("cosmos3.reason", owner, wrapped, "application/json")
    generic_body, generic_type = encode_future_multipart({"protocol": "ego.annotation.api.v1", "status": "ok", "ownership": wire_owner("cosmos3.reason"), "result": {}, "error": None}, {})
    with pytest.raises(LiveWireError, match="top-level|generic|result"):
        decode_live_response("cosmos3.reason", owner, generic_body, generic_type)


def test_contract_derived_droid_profile_is_fail_closed_and_not_response_overridable() -> None:
    profile = DroidCapabilities.frozen_3572551()
    assert profile.full_K_consumed is True
    assert profile.native_sensor_depth_consumed is False
    assert profile.frontend_no_grad is False
    assert profile.capability_source == "frozen_contract"
    assert profile.service_release == "3572551/observed-config"
    with pytest.raises(Exception, match="native-depth|capability"):
        profile.require_rgbd()

    called = False
    def opener(req: Any, timeout: float) -> FrozenResponse:
        nonlocal called
        called = True
        metadata, _parts = parse_like_frozen_server(req.data, req.get_header("Content-type"))
        response = {"ownership": metadata["ownership"], "session_id": "strict-session", "capabilities": {"native_sensor_depth_consumed": True}}
        return FrozenResponse(json.dumps(response).encode(), "application/json")

    backend = ApiBackend(ApiBackendConfig.for_stage("droid.create_session"), opener=opener)
    with pytest.raises(CapabilityMismatchError, match="remote_droid_capability_mismatch"):
        backend.execute(make_droid_create(strict=True))
    assert called is False  # contract-derived frozen profile cannot be overridden by a response envelope


def test_droid_original_wire_ordered_sparse_sequence_is_diagnostic_only() -> None:
    state = {"session": None, "next": 0, "pushes": [], "expected_frames": [0, 3]}

    def opener(req: Any, timeout: float) -> FrozenResponse:
        assert req.get_header("X-ego-video-job-id") == "case"
        assert req.get_header("X-ego-video-item-id") == "item"
        stage = req.full_url.rsplit("/", 1)[-1]
        metadata, parts = parse_like_frozen_server(req.data, req.get_header("Content-type"))
        assert not FROZEN_GENERIC_FORBIDDEN.intersection(metadata)
        if stage == "droid.create_session":
            assert state["session"] is None
            state["session"] = "smoke-session"
            return FrozenResponse(json.dumps({"ownership": metadata["ownership"], "session_id": state["session"]}).encode(), "application/json")
        if stage == "droid.push_frame":
            assert metadata["session_id"] == state["session"]
            frame = state["expected_frames"][state["next"]]
            assert int(metadata["frame_id"]) == frame
            assert float(metadata["source_timestamp_s"]) == frame / 25.0
            assert set(parts) == {"rgb", "static_confidence_mask", "depth_m"}
            assert parts["depth_m"][1:] [0] == (16, 16)
            state["pushes"].append(frame)
            state["next"] += 1
            status = {"ownership": metadata["ownership"], "session_id": state["session"], "frame_id": str(frame), "source_timestamp_s": frame / 25.0, "validity": {"frame_id": str(frame), "source_timestamp_s": frame / 25.0, "admitted": True, "keyframe_added": True, "skip_reason": None}, "keyframe_count": state["next"], "trace": {}}
            return FrozenResponse(json.dumps({"ownership": metadata["ownership"], "status": status}).encode(), "application/json")
        assert stage == "droid.finalize"
        assert metadata["session_id"] == state["session"] and state["pushes"] == [0, 3]
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
        disparities = np.ones((2, 2, 2), np.float32)
        def tw(array: np.ndarray) -> dict[str, object]:
            import base64
            return {"data_b64": base64.b64encode(array.tobytes()).decode("ascii"), "shape": list(array.shape), "dtype": array.dtype.name}
        camera_state = {"ownership": metadata["ownership"], "session_id": state["session"], "T_world_camera": tw(poses), "T_camera_world": tw(poses), "intrinsics_px": tw(intrinsics), "disparities": tw(disparities), "keyframe_mapping": [{"keyframe_index": 0}, {"keyframe_index": 1}], "dense_mapping": [], "uncertainty": {"scale_status": "up_to_scale"}, "model_revision": "droid-v1", "trace": {}}
        return frozen_multipart({"ownership": metadata["ownership"], "camera_state": camera_state}, {"T_world_camera": poses, "T_camera_world": poses, "intrinsics_px": intrinsics, "disparities": disparities})

    backend = ApiBackend(ApiBackendConfig.for_stage("droid.create_session"), opener=opener)
    created = backend.execute(make_droid_create(strict=False))
    assert created.output.capabilities.native_sensor_depth_consumed is False
    backend.execute(make_droid_push(created.output.session_id, 0))
    backend.execute(make_droid_push(created.output.session_id, 3))
    finalized = backend.execute(make_droid_finalize(created.output.session_id))
    assert finalized.output.scale_mode == "up_to_scale_monocular"
    assert finalized.output.diagnostic_only is True
    assert finalized.output.acceptance is False
    assert finalized.output.capabilities.native_sensor_depth_consumed is False


def test_droid_recovery_direction_and_dag_lanes() -> None:
    policy = DroidRecoveryPolicy(oom_filter_thresh=8.0, exclusive_fixed_release="fe087cd0", keyframe_retry_filter_thresh=1.2)
    assert policy.on_oom() == {"classification": "remote_droid_oom", "filter_thresh": 8.0, "exclusive_fixed_release": "fe087cd0", "automatic_retry": False}
    assert policy.on_finalize_keyframes(1, 0)["action"] == "retry_lower_filter_thresh"
    exhausted = policy.on_finalize_keyframes(1, 1)
    assert exhausted["action"] == "insufficient_trajectory" and exhausted["skip_pairwise_ba_and_filler"] is True
    strategy = DroidRetryStrategy(keyframe_retry_filter_thresh=1.2)
    assert strategy.on_finalize_keyframes(1).action == "retry_lower_filter_thresh"
    assert strategy.on_finalize_keyframes(1, 1).action == "remote_droid_insufficient_keyframes"

    dag = TypedDag()
    order = dag.start_order(include_semantic=False)
    assert order.index("unidepth.infer") < order.index("droid.create_session")
    assert order.index("hands.detect") < order.index("wilor.reconstruct")
    assert "cosmos3.reason" not in order
    assert all(node.lane == DagLane.PHYSICAL for node in dag.nodes if node.stage_id in order)
    config = RunnerConfig(cosmos_enabled=False, fresh_run_root="/fresh/run")
    assert config.semantic_lane().status == "absent_disabled" and config.semantic_lane().rows == ()
    assert 28004 in config.allowed_service_ports
    with pytest.raises(Exception, match="missing typed request"):
        dag.run({}, object(), config)  # type: ignore[arg-type]
    with pytest.raises(Exception, match="shared production"):
        RunnerConfig(cosmos_enabled=False, fresh_run_root="/fresh/run", shared_production_actors=True)


def test_next_run_fixture_preflight_requires_exact_source_and_fresh_root() -> None:
    fixture = DEFAULT_FROZEN_SINGLE_VIDEO
    observed = SourceProbe(fixture.input_sha256, fixture.input_size_bytes, fixture.frame_count, fixture.duration_s, fixture.fps, fixture.width_px, fixture.height_px)
    validate_preflight(fixture, observed, run_root="/home/zjh/data/v22_api_backend_frozen_single_feishu_validation_task10_20260720T170000Z", run_root_exists=False, run_root_nonempty=False)
    with pytest.raises(RunPreflightError, match="source probe mismatch"):
        validate_preflight(fixture, SourceProbe("0" * 64, fixture.input_size_bytes, fixture.frame_count, fixture.duration_s, fixture.fps, fixture.width_px, fixture.height_px), run_root="/home/zjh/data/v22_api_backend_frozen_single_x", run_root_exists=False, run_root_nonempty=False)
    with pytest.raises(RunPreflightError, match="already exists"):
        validate_preflight(fixture, observed, run_root="/home/zjh/data/v22_api_backend_frozen_single_feishu_validation_task10_20260720T170000Z", run_root_exists=True, run_root_nonempty=True)


def test_http_429_backpressure_retries_with_bounded_exponential_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    waits: list[int] = []

    def opener(http_request: Any, _timeout: float) -> FrozenResponse:
        nonlocal attempts
        attempts += 1
        if attempts <= 5:
            raise urllib.error.HTTPError(
                http_request.full_url,
                429,
                "Too Many Requests",
                {},
                io.BytesIO(b'{"error":{"code":"backpressure"}}'),
            )
        return FrozenResponse(b"accepted", "application/json")

    monkeypatch.setattr(api_backend_module.time, "sleep", waits.append)
    backend = ApiBackend(ApiBackendConfig.for_stage("hands.detect"), opener=opener)

    body, content_type = backend._post(route_for("hands.detect"), b"request", "application/octet-stream", request=make_hands())

    assert (body, content_type) == (b"accepted", "application/json")
    assert attempts == 6
    assert waits == [1, 2, 4, 8, 16]


def test_proxy_exhausted_429_is_not_retried_a_second_time(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    waits: list[int] = []

    def opener(http_request: Any, _timeout: float) -> FrozenResponse:
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            http_request.full_url,
            429,
            "Too Many Requests",
            {"X-Ego-Admission-Retry-Complete": "1"},
            io.BytesIO(b'{"error":{"code":"backpressure"}}'),
        )

    monkeypatch.setattr(api_backend_module.time, "sleep", waits.append)
    backend = ApiBackend(ApiBackendConfig.for_stage("hands.detect"), opener=opener)

    with pytest.raises(ApiTransportError, match="hands.detect HTTP 429"):
        backend._post(route_for("hands.detect"), b"request", "application/octet-stream", request=make_hands())

    assert attempts == 1
    assert waits == []


def test_http_error_other_than_429_remains_single_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    waits: list[int] = []

    def opener(http_request: Any, _timeout: float) -> FrozenResponse:
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(http_request.full_url, 503, "Service Unavailable", {}, io.BytesIO(b"unavailable"))

    monkeypatch.setattr(api_backend_module.time, "sleep", waits.append)
    backend = ApiBackend(ApiBackendConfig.for_stage("hands.detect"), opener=opener)

    with pytest.raises(ApiTransportError, match="hands.detect HTTP 503: unavailable"):
        backend._post(route_for("hands.detect"), b"request", "application/octet-stream", request=make_hands())

    assert attempts == 1
    assert waits == []


def test_droid_finalize_http_error_preserves_nonfinite_trajectory_detail() -> None:
    detail = json.dumps({
        "ownership": {"padding": "x" * 700},
        "camera_state": None,
        "error": {"code": "model_failure", "message": "CameraState.T_world_camera must contain only finite values"},
    }).encode()

    def opener(http_request: Any, _timeout: float) -> FrozenResponse:
        raise urllib.error.HTTPError(http_request.full_url, 503, "Service Unavailable", {}, io.BytesIO(detail))

    backend = ApiBackend(ApiBackendConfig.for_stage("droid.finalize"), opener=opener)

    with pytest.raises(ApiTransportError, match="T_world_camera must contain only finite values"):
        backend._post(route_for("droid.finalize"), b"request", "application/octet-stream", request=make_droid_finalize("session"))


def test_transport_source_does_not_import_forbidden_callers() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("api_backend.py", "live_wire.py", "typed_contracts.py", "multipart.py"):
        source = (root / "ego_annotation" / name).read_text(encoding="utf-8")
        assert "call_feishu" not in source and "run_feishu" not in source and "ego_annotation.serving" not in source
    assert "encode_future_multipart(" not in (root / "ego_annotation" / "api_backend.py").read_text(encoding="utf-8")


def make_cosmos(media_count: int = 1, *, data: bytes = b"jpeg") -> AlgorithmRequest[Any]:
    assets = tuple(BinaryAsset(data, "image/jpeg", f"frame-{index}", (index,)) for index in range(media_count))
    value = CosmosInput(typed_owner("cosmos3.reason"), "describe exact frames", (), CosmosGeneration(512), assets, tuple(range(media_count)))
    return request("cosmos3.reason", value, 1, (1,), axis=None)


def cosmos_result(req: AlgorithmRequest[Any], *, text: object = "{\"items\":[]}", prompt_tokens: object = 3, provenance: object | None = None) -> dict[str, object]:
    value = req.input
    if provenance is None:
        provenance = [
            {"kind": "image", "media_type": media.media_type, "source_index": index, "bytes": len(media.data)}
            for media, index in zip(value.media, value.source_frame_indices)
        ]
    return {
        "ownership": wire_owner("cosmos3.reason"),
        "text": text,
        "finish_reason": "stop",
        "stop_reason": None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 2,
        "total_tokens": 5,
        "timings": {"queue_wait_s": 0.1, "prefill_s": 0.1, "time_to_first_token_s": 0.1, "decode_s": 0.1, "e2e_s": 0.4},
        "model_revision": "server-cosmos3",
        "trace": {"batch_id": "batch", "replica_id": "gpu6", "admitted_monotonic_s": 1.0, "dispatched_monotonic_s": 1.1, "forward_started_monotonic_s": 1.2, "completed_monotonic_s": 1.3, "effective_work_units": 1, "request_count": 1, "forward_count": 1, "model_load_count": 1},
        "media_provenance": provenance,
    }


def test_cosmos_typed_boundary_rejects_bool_float_and_string_numeric_fields() -> None:
    asset = BinaryAsset(b"jpeg", "image/jpeg", "frame", (0,))
    for bad_index in (False, 0.0, "0"):
        with pytest.raises(Exception, match="indices"):
            BinaryAsset(b"jpeg", "image/jpeg", "frame", (bad_index,))  # type: ignore[arg-type]
        with pytest.raises(Exception, match="indices"):
            CosmosInput(typed_owner("cosmos3.reason"), "prompt", (), CosmosGeneration(5), (asset,), (bad_index,))  # type: ignore[arg-type]
    with pytest.raises(Exception, match="temperature"):
        CosmosGeneration(5, temperature=False)
    with pytest.raises(Exception, match="top_p"):
        CosmosGeneration(5, top_p=True)
    for bad_max_tokens in (True, 5.0, "5"):
        with pytest.raises(Exception, match="max_tokens"):
            CosmosGeneration(bad_max_tokens)  # type: ignore[arg-type]


def test_cosmos_typed_limits_prompt_xor_and_order() -> None:
    assert len(make_cosmos(8).input.media) == 8
    with pytest.raises(Exception, match="at most 8"):
        make_cosmos(9)
    with pytest.raises(Exception, match="16 MiB"):
        make_cosmos(1, data=b"x" * (16 * 1024 * 1024 + 1))
    shared = b"x" * (13 * 1024 * 1024)
    with pytest.raises(Exception, match="64 MiB"):
        make_cosmos(5, data=shared)
    asset = BinaryAsset(b"jpeg", "image/jpeg", "frame", (0,))
    with pytest.raises(Exception, match="XOR"):
        CosmosInput(typed_owner("cosmos3.reason"), "prompt", (CosmosMessage("user", "also"),), CosmosGeneration(5), (asset,), (0,))
    with pytest.raises(Exception, match="exactly match"):
        CosmosInput(typed_owner("cosmos3.reason"), "prompt", (), CosmosGeneration(5), (asset,), (1,))
    descending_assets = (
        BinaryAsset(b"jpeg", "image/jpeg", "frame-1", (1,)),
        BinaryAsset(b"jpeg", "image/jpeg", "frame-0", (0,)),
    )
    with pytest.raises(Exception, match="nondecreasing"):
        CosmosInput(typed_owner("cosmos3.reason"), "prompt", (), CosmosGeneration(5), descending_assets, (1, 0))
    premature_padding_assets = (
        BinaryAsset(b"jpeg", "image/jpeg", "frame-0", (0,)),
        BinaryAsset(b"jpeg", "image/jpeg", "frame-1a", (1,)),
        BinaryAsset(b"jpeg", "image/jpeg", "frame-1b", (1,)),
    )
    with pytest.raises(Exception, match="only pad final"):
        CosmosInput(typed_owner("cosmos3.reason"), "prompt", (), CosmosGeneration(5), premature_padding_assets, (0, 1, 1))


def test_cosmos_padded_media_preserves_duplicate_source_provenance() -> None:
    source_indices = (240, 270, 300, 330, 330, 330, 330, 330)
    assets = tuple(BinaryAsset(b"jpeg", "image/jpeg", f"frame-{index}-{slot}", (index,)) for slot, index in enumerate(source_indices))
    cosmos = CosmosInput(typed_owner("cosmos3.reason"), "describe local slots", (), CosmosGeneration(512), assets, source_indices)

    _metadata, parts = assert_frozen_request(request("cosmos3.reason", cosmos, 1, (1,), axis=None), {"ownership", "prompt", "messages", "generation"}, {f"media_{slot}" for slot in range(8)})

    assert [parts[f"media_{slot}"][3]["source_index"] for slot in range(8)] == [str(index) for index in source_indices]


def test_cosmos_complete_frozen_result_decodes_losslessly_and_strictly() -> None:
    req = make_cosmos(2)
    result = cosmos_result(req)
    response = frozen_multipart({"ownership": wire_owner("cosmos3.reason"), "result": result, "error": None}, {})
    backend = ApiBackend(ApiBackendConfig.for_stage("cosmos3.reason", cosmos_enabled=True), opener=lambda _req, _timeout: response)

    output = backend.execute(req).output

    assert output.text == "{\"items\":[]}"
    assert output.total_tokens == 5
    assert output.stop_reason is None
    assert output.model_revision == "server-cosmos3"
    assert output.trace["replica_id"] == "gpu6"
    assert [item["source_index"] for item in output.media_provenance] == [0, 1]
    assert backend.service_batch_traces == (
        {
            "stage_id": "cosmos3.reason",
            "case_id": "case",
            "item_id": "item",
            "source_id": "video.mp4",
            "trace_location": "result.trace",
            "trace": result["trace"],
        },
    )

    for field, bad_value, message in (
        ("text", "", "text"),
        ("prompt_tokens", "3", "prompt_tokens"),
        ("total_tokens", 99, "total_tokens"),
    ):
        bad = dict(result, **{field: bad_value})
        bad_response = frozen_multipart({"ownership": wire_owner("cosmos3.reason"), "result": bad, "error": None}, {})
        bad_backend = ApiBackend(ApiBackendConfig.for_stage("cosmos3.reason", cosmos_enabled=True), opener=lambda _req, _timeout, r=bad_response: r)
        with pytest.raises(ApiProtocolError, match=message):
            bad_backend.execute(req)

    wrong_nested = dict(result, ownership={**wire_owner("cosmos3.reason"), "item_id": "other"})
    wrong_response = frozen_multipart({"ownership": wire_owner("cosmos3.reason"), "result": wrong_nested, "error": None}, {})
    with pytest.raises(ApiProtocolError, match="nested"):
        ApiBackend(ApiBackendConfig.for_stage("cosmos3.reason", cosmos_enabled=True), opener=lambda _req, _timeout: wrong_response).execute(req)

    wrong_provenance = [dict(result["media_provenance"][0]), dict(result["media_provenance"][1], source_index=7)]
    provenance_response = frozen_multipart({"ownership": wire_owner("cosmos3.reason"), "result": dict(result, media_provenance=wrong_provenance), "error": None}, {})
    with pytest.raises(ApiProtocolError, match="provenance"):
        ApiBackend(ApiBackendConfig.for_stage("cosmos3.reason", cosmos_enabled=True), opener=lambda _req, _timeout: provenance_response).execute(req)


def test_service_trace_capture_preserves_droid_specific_counts() -> None:
    backend = ApiBackend(ApiBackendConfig(base_url="http://127.0.0.1"))
    request_value = make_cosmos()
    trace = {
        "batch_id": "droid-batch",
        "request_count": 2,
        "fnet_forward_count": 1,
        "effective_work_units": 8,
        "replica_id": "gpu-1",
        "session_local_forward_count": 5,
        "session_count": 1,
    }
    decoded = LiveRouteResponse({}, {"trace": trace}, {}, {})

    backend._record_service_batch_trace(request_value, decoded)

    assert backend.service_batch_traces[0]["trace"] == trace


def test_cosmos_response_rejects_extra_fields_and_nonexact_provenance_primitives() -> None:
    req = make_cosmos()
    result = cosmos_result(req)

    def assert_rejected(metadata: dict[str, object], expected: str) -> None:
        response = frozen_multipart(metadata, {})
        backend = ApiBackend(ApiBackendConfig.for_stage("cosmos3.reason", cosmos_enabled=True), opener=lambda _req, _timeout: response)
        with pytest.raises(ApiProtocolError, match=expected):
            backend.execute(req)

    assert_rejected({"ownership": wire_owner("cosmos3.reason"), "result": result, "error": None, "extra": "nope"}, "top-level")
    assert_rejected({"ownership": wire_owner("cosmos3.reason"), "result": {**result, "extra": "nope"}, "error": None}, "result fields")
    for bad_value in (False, 0.0, "0"):
        bad_provenance = [dict(result["media_provenance"][0], source_index=bad_value)]
        assert_rejected({"ownership": wire_owner("cosmos3.reason"), "result": {**result, "media_provenance": bad_provenance}, "error": None}, "provenance")
    assert_rejected(
        {"ownership": wire_owner("cosmos3.reason"), "result": {**result, "media_provenance": [dict(result["media_provenance"][0], bytes=4.0)]}, "error": None},
        "provenance",
    )
    assert_rejected(
        {"ownership": wire_owner("cosmos3.reason"), "result": {**result, "timings": {**result["timings"], "e2e_s": float("nan")}}, "error": None},
        "timings",
    )
    assert_rejected(
        {"ownership": wire_owner("cosmos3.reason"), "result": {**result, "trace": {**result["trace"], "request_count": False}}, "error": None},
        "trace counts",
    )


def test_stage_capture_is_full_hash_atomic_bounded_and_indexed(tmp_path: Path) -> None:
    req = make_cosmos()
    response = frozen_multipart({"ownership": wire_owner("cosmos3.reason"), "result": cosmos_result(req), "error": None}, {})
    root = tmp_path / "captures"
    backend = ApiBackend(ApiBackendConfig.for_stage("cosmos3.reason", cosmos_enabled=True, stage_capture_root=str(root)), opener=lambda _req, _timeout: response)

    backend.execute(req)
    backend.execute(req)

    manifests = list(root.glob("cosmos3_reason/*/*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    request_bytes = (manifests[0].parent / "request.multipart").read_bytes()
    response_bytes = (manifests[0].parent / "response.multipart").read_bytes()
    import hashlib
    assert hashlib.sha256(request_bytes).hexdigest() == manifest["request"]["sha256"] == manifests[0].parents[1].name
    assert hashlib.sha256(response_bytes).hexdigest() == manifest["response"]["sha256"] == manifests[0].parent.name
    assert manifest["request"]["content_type"].startswith("multipart/form-data")
    assert "authorization" not in json.dumps(manifest).lower()
    index = json.loads((root / "fixture_index.json").read_text(encoding="utf-8"))
    assert len(index["entries"]) == 1 and index["entries"][0]["algorithm_id"] == "cosmos3.reason"
    assert not list(root.rglob(".*.tmp"))


def test_cosmos_capture_limit_can_be_bounded_without_expanding_hawor_captures(tmp_path: Path) -> None:
    req = make_cosmos()
    root = tmp_path / "captures"
    def opener(http_request, _timeout):
        metadata, _ = parse_like_frozen_server(http_request.data, http_request.get_header("Content-type"))
        ownership = metadata["ownership"]
        result = cosmos_result(req)
        result["ownership"] = ownership
        return frozen_multipart({"ownership": ownership, "result": result, "error": None}, {})

    backend = ApiBackend(
        ApiBackendConfig.for_stage(
            "cosmos3.reason",
            cosmos_enabled=True,
            stage_capture_root=str(root),
            stage_capture_limits={"cosmos3.reason": 2},
        ),
        opener=opener,
    )
    backend.execute(req)
    repair = replace(req, input=replace(req.input, ownership=replace(req.input.ownership, scope=req.input.ownership.scope + ":repair:1"), prompt="repair"))
    backend.execute(repair)
    assert len(list(root.glob("cosmos3_reason/*/*/manifest.json"))) == 2


def test_capture_allowlist_budgets_one_exchange_for_each_required_stage(tmp_path: Path) -> None:
    backend = ApiBackend(ApiBackendConfig(base_url="http://127.0.0.1", stage_capture_root=str(tmp_path)))
    for req in (make_hawor(), make_infiller(), make_cosmos()):
        decoded = LiveRouteResponse(wire_owner(req.algorithm_id), {"valid": True}, {"ownership": wire_owner(req.algorithm_id), "result": {"valid": True}}, {})
        body, content_type, metadata, names = encode_live_request(req)
        backend._capture_stage_exchange(req, body, content_type, metadata, names, b"response-" + req.algorithm_id.encode(), "multipart/form-data; boundary=resp", decoded)
        backend._capture_stage_exchange(req, body, content_type, metadata, names, b"second-response", "multipart/form-data; boundary=resp", decoded)
    index = json.loads((tmp_path / "fixture_index.json").read_text(encoding="utf-8"))
    assert {entry["algorithm_id"] for entry in index["entries"]} == {"hawor.infer_tracks", "hawor_infiller.fill", "cosmos3.reason"}
    assert len(index["entries"]) == 3


def test_capture_collision_and_atomic_failure_leave_no_valid_fixture(monkeypatch, tmp_path: Path) -> None:
    req = make_cosmos()
    body, content_type, metadata, names = encode_live_request(req)
    decoded = LiveRouteResponse(wire_owner(req.algorithm_id), {"valid": True}, {"ownership": wire_owner(req.algorithm_id), "result": {"valid": True}}, {})
    response = b"response"
    import hashlib
    address = tmp_path / "collision" / "cosmos3_reason" / hashlib.sha256(body).hexdigest() / hashlib.sha256(response).hexdigest()
    address.mkdir(parents=True)
    (address / "request.multipart").write_bytes(b"different")
    (address / "response.multipart").write_bytes(response)
    backend = ApiBackend(ApiBackendConfig(base_url="http://127.0.0.1", stage_capture_root=str(tmp_path / "collision")))
    with pytest.raises(ApiProtocolError, match="collision"):
        backend._capture_stage_exchange(req, body, content_type, metadata, names, response, "application/json", decoded)
    assert backend._capture_counts["cosmos3.reason"] == 0

    atomic_root = tmp_path / "atomic"
    atomic_backend = ApiBackend(ApiBackendConfig(base_url="http://127.0.0.1", stage_capture_root=str(atomic_root)))
    original = atomic_backend._write_capture_file
    calls = 0

    def fail_second(path: Path, data: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        original(path, data)

    monkeypatch.setattr(atomic_backend, "_write_capture_file", fail_second)
    with pytest.raises(OSError, match="injected"):
        atomic_backend._capture_stage_exchange(req, body, content_type, metadata, names, response, "application/json", decoded)
    assert not list(atomic_root.glob("cosmos3_reason/*/*/manifest.json"))
    assert not list(atomic_root.rglob(".*"))
    assert atomic_backend._capture_counts["cosmos3.reason"] == 0
