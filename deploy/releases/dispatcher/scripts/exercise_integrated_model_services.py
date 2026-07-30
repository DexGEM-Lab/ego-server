#!/usr/bin/env python3
"""Exercise all seven live model-native services from one curated client.

This driver is deliberately client-only: it performs HTTP calls against already-live
endpoints and never imports Ray, changes deployments, or invokes service lifecycle
commands.  It reuses the real payload corpora/manifests preserved by the lane
benchmarks, starts the seven logical service calls concurrently, and freezes one
reviewable artifact containing ownership, revision, latency, typed errors, endpoint
health, payload hashes, and physical GPU mapping.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
import uuid

# Make a curated checkout directly executable without requiring an editable install
# or a caller-provided PYTHONPATH.
_CLIENT_ROOT = Path(__file__).resolve().parents[1]
if str(_CLIENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLIENT_ROOT))

import cv2
import httpx
import numpy as np

from ego_annotation.serving.router import ModelApiName, ModelServiceRouter, service_for_api
from ego_annotation.serving.transport import (
    build_cosmos3_request,
    build_multipart_request_fields,
    parse_cosmos3_response,
    parse_multipart_response,
)

BENCHMARK_ROOT = Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_benchmarks")
DEFAULT_UNIDEPTH_MANIFEST = BENCHMARK_ROOT / "gpu0_unidepth_endpoint_openloop_20260716T200015Z_final/open_loop_sweep/payload_manifest.json"
DEFAULT_UNIDEPTH_FRAME = Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_lanes/gpu0_unidepth_a77a57e_20260716_190232/egoscale_frames/frame_A_idx000.npy")
DEFAULT_GPU1_ROOT = BENCHMARK_ROOT / "gpu1_hands_wilor_20260716T2001Z"
DEFAULT_GPU1_FRAMES = Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/data/gpuheavy_tensor_batch_bench/20260715T_tensor_batch_v1/hawor/hawor_frames")
DEFAULT_WILOR_ROOT = Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor_model")
DEFAULT_WILOR_CONFIG = Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor/model_config.yaml")
DEFAULT_DROID_MANIFEST = BENCHMARK_ROOT / "droid_openloop_final_20260716T204043Z/droid/payload_manifest.json"
DEFAULT_HAWOR_BENCHMARK = BENCHMARK_ROOT / "gpu3_hawor_infiller_20260716T2003Z/smoke_open_loop.json"
DEFAULT_HAWOR_REPO = Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/HaWoR")
DEFAULT_HAWOR_SEQUENCE = DEFAULT_HAWOR_REPO / "example/annotation_1efc7a381e58"
DEFAULT_COSMOS_MANIFEST = BENCHMARK_ROOT / "cosmos3_open_loop_3f980de_20260716T205845Z/cosmos3/benchmark_manifest.json"
SCHEMA_VERSION = "ego.model-service.v1"
GPU_ASSIGNMENTS = {
    "unidepth": 0,
    "hands": 1,
    "wilor": 1,
    "droid": 2,
    "hawor_tracks": 3,
    "hawor_infiller": 3,
    "cosmos3": 6,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ownership(run_id: str, api_name: str, item_id: str, source_id: str, timestamp_s: float | None = None) -> dict[str, Any]:
    return {
        "request_id": f"{run_id}-{api_name}-{uuid.uuid4().hex[:10]}",
        "job_id": f"integrated-{run_id}",
        "item_id": item_id,
        "stage_id": api_name,
        "source_id": source_id,
        "schema_version": SCHEMA_VERSION,
        "source_timestamp_s": timestamp_s,
        "submitted_at": utc_now(),
    }


def identity_spatial(width: int, height: int) -> dict[str, Any]:
    eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return {
        "source_size": {"width": width, "height": height},
        "model_size": {"width": width, "height": height},
        "color_space": "RGB",
        "pixel_transform": {
            "source_to_model": eye,
            "model_to_source": eye,
            "resize_mode": "canonical_input",
            "crop_xywh": None,
            "pad_ltrb": None,
        },
        "K_px": None,
    }


def array_summary(arrays: Mapping[str, tuple[bytes, tuple[int, ...], str]]) -> dict[str, Any]:
    return {
        name: {"shape": list(shape), "dtype": dtype, "bytes": len(data), "sha256": sha256(data)}
        for name, (data, shape, dtype) in sorted(arrays.items())
    }


def response_ownership(wire: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates: list[Any] = [wire.get("ownership")]
    for key in ("result", "status", "camera_state"):
        value = wire.get(key)
        if isinstance(value, Mapping):
            candidates.extend((value.get("ownership"), value.get("result", {}).get("ownership") if isinstance(value.get("result"), Mapping) else None))
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return candidate
    return None


def response_revision(wire: Mapping[str, Any]) -> str | None:
    candidates: list[Any] = [wire.get("model_revision")]
    for key in ("result", "status", "camera_state"):
        value = wire.get(key)
        if isinstance(value, Mapping):
            candidates.extend((value.get("model_revision"), value.get("revision")))
            loaded = value.get("loaded_models")
            if isinstance(loaded, list) and len(loaded) == 1:
                candidates.append(loaded[0])
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return None


def response_trace(wire: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("result", "status", "camera_state"):
        value = wire.get(key)
        if isinstance(value, Mapping) and isinstance(value.get("trace"), Mapping):
            return value["trace"]
    return None


def typed_error(wire: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = wire.get("error")
    return value if isinstance(value, Mapping) else None


async def post_fields(
    client: httpx.AsyncClient,
    *,
    api_name: str,
    url: str,
    metadata: Mapping[str, Any],
    fields: Mapping[str, tuple[bytes, Sequence[int], str]],
) -> dict[str, Any]:
    body, content_type = build_multipart_request_fields(metadata, fields)
    started = time.monotonic()
    try:
        response = await client.post(url, content=body, headers={"Content-Type": content_type})
        latency_ms = (time.monotonic() - started) * 1000.0
        response_type = response.headers.get("content-type", "")
        meta, arrays = parse_multipart_response(response.content, response_type)
        error = typed_error(meta)
        return {
            "api_name": api_name,
            "endpoint": url,
            "http_status": response.status_code,
            "latency_ms": latency_ms,
            "request_ownership": metadata.get("ownership"),
            "response_ownership": response_ownership(meta),
            "model_revision": response_revision(meta),
            "typed_error": error,
            "trace": response_trace(meta),
            "response_arrays": array_summary(arrays),
            "response_metadata": meta,
            "success": response.status_code == 200 and error is None,
        }
    except Exception as exc:
        return {
            "api_name": api_name,
            "endpoint": url,
            "http_status": None,
            "latency_ms": (time.monotonic() - started) * 1000.0,
            "request_ownership": metadata.get("ownership"),
            "response_ownership": None,
            "model_revision": None,
            "typed_error": {"code": "transport", "message": repr(exc), "retryable": False},
            "trace": None,
            "response_arrays": {},
            "success": False,
        }


async def post_json_or_multipart(
    client: httpx.AsyncClient,
    *,
    api_name: str,
    url: str,
    metadata: Mapping[str, Any],
    fields: Mapping[str, tuple[bytes, Sequence[int], str]],
) -> dict[str, Any]:
    body, content_type = build_multipart_request_fields(metadata, fields)
    started = time.monotonic()
    try:
        response = await client.post(url, content=body, headers={"Content-Type": content_type})
        latency_ms = (time.monotonic() - started) * 1000.0
        try:
            wire = response.json()
        except Exception:
            wire, arrays = parse_multipart_response(response.content, response.headers.get("content-type", ""))
        else:
            arrays = {}
        if not isinstance(wire, Mapping):
            raise TypeError(f"response is {type(wire).__name__}, expected object")
        error = typed_error(wire)
        return {
            "api_name": api_name,
            "endpoint": url,
            "http_status": response.status_code,
            "latency_ms": latency_ms,
            "request_ownership": metadata.get("ownership"),
            "response_ownership": response_ownership(wire),
            "model_revision": response_revision(wire),
            "typed_error": error,
            "trace": response_trace(wire),
            "response_arrays": array_summary(arrays),
            "response_metadata": wire,
            "success": response.status_code == 200 and error is None,
        }
    except Exception as exc:
        return {
            "api_name": api_name,
            "endpoint": url,
            "http_status": None,
            "latency_ms": (time.monotonic() - started) * 1000.0,
            "request_ownership": metadata.get("ownership"),
            "response_ownership": None,
            "model_revision": None,
            "typed_error": {"code": "transport", "message": repr(exc), "retryable": False},
            "trace": None,
            "response_arrays": {},
            "success": False,
        }


def verify_manifest_hash(data: bytes, expected: str, label: str) -> None:
    actual = sha256(data)
    if actual != expected:
        raise RuntimeError(f"{label} payload hash mismatch: manifest={expected}, actual={actual}")


async def call_unidepth(args: argparse.Namespace, client: httpx.AsyncClient, router: ModelServiceRouter) -> dict[str, Any]:
    manifest = read_json(args.unidepth_manifest)
    item = manifest[0]
    rgb = np.ascontiguousarray(np.load(args.unidepth_frame).astype(np.uint8))
    verify_manifest_hash(rgb.tobytes(), item["sha256"], "UniDepth")
    own = ownership(args.run_id, "unidepth.infer", item["source_id"], item["source_id"], 0.0)
    metadata = {
        "ownership": own,
        "spatial": identity_spatial(rgb.shape[1], rgb.shape[0]),
        "model_revision": router.endpoint_for(ModelApiName.UNIDEPTH_INFER).model_revision,
        "options": {},
    }
    result = await post_fields(client, api_name="unidepth.infer", url=router.url_for(ModelApiName.UNIDEPTH_INFER), metadata=metadata,
                               fields={"rgb": (rgb.tobytes(), rgb.shape, "uint8")})
    result["payload"] = {"manifest": str(args.unidepth_manifest), "source_id": item["source_id"], "sha256": item["sha256"]}
    return result


def load_gpu1_frame(args: argparse.Namespace) -> tuple[np.ndarray, Mapping[str, Any]]:
    manifest = read_json(args.gpu1_root / "payload_manifest.json")
    item = manifest[0]
    bgr = cv2.imread(str(args.gpu1_frames / item["source_file"]), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot decode GPU1 payload {item['source_file']}")
    rgb = np.ascontiguousarray(cv2.cvtColor(cv2.resize(bgr, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    verify_manifest_hash(rgb.tobytes(), item["rgb_sha256"], "GPU1 image")
    return rgb, item


async def call_hands(args: argparse.Namespace, client: httpx.AsyncClient, router: ModelServiceRouter, rgb: np.ndarray, item: Mapping[str, Any]) -> dict[str, Any]:
    own = ownership(args.run_id, "hands.detect", f"frame-{item['index']}", f"egoscale-{item['source_file']}", float(item["index"]) / 30.0)
    metadata = {
        "ownership": own,
        "spatial": identity_spatial(960, 540),
        "model_revision": router.endpoint_for(ModelApiName.HANDS_DETECT).model_revision,
        "options": {},
    }
    result = await post_fields(client, api_name="hands.detect", url=router.url_for(ModelApiName.HANDS_DETECT), metadata=metadata,
                               fields={"rgb": (rgb.tobytes(), rgb.shape, "uint8")})
    result["payload"] = {"manifest": str(args.gpu1_root / "payload_manifest.json"), "source_file": item["source_file"], "sha256": item["rgb_sha256"]}
    return result


def build_wilor_crop(args: argparse.Namespace, rgb: np.ndarray) -> tuple[np.ndarray, Mapping[str, Any]]:
    crop_manifest = read_json(args.gpu1_root / "wilor_crop_manifest.json")
    crop_item = crop_manifest[0]
    evidence = read_json(args.gpu1_root / "results.json")
    detection = next(row for row in evidence["detect_evidence"] if int(row["index"]) == int(crop_item["index"]))
    box = np.asarray(detection["boxes"][0], dtype=np.float32)
    old_cwd = Path.cwd()
    try:
        os.chdir(args.wilor_root)
        sys.path.insert(0, str(args.wilor_root))
        from wilor.configs import get_config
        from wilor.datasets.vitdet_dataset import ViTDetDataset

        config = get_config(str(args.wilor_config), update_cachedir=True)
        config.defrost()
        if "BBOX_SHAPE" not in config.MODEL:
            config.MODEL.BBOX_SHAPE = [192, 256]
        config.freeze()
        data = ViTDetDataset(
            config,
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            np.asarray([box], dtype=np.float32),
            np.asarray([crop_item["handedness"]], dtype=np.float32),
            rescale_factor=2.0,
        )[0]
        crop = data["img"]
        if hasattr(crop, "numpy"):
            crop = crop.numpy()
        crop = np.ascontiguousarray(np.asarray(crop, dtype=np.float32))
    finally:
        os.chdir(old_cwd)
    verify_manifest_hash(crop.tobytes(), crop_item["sha256"], "WiLoR crop")
    for key, observed in (("box_center", data["box_center"]), ("box_size", data["box_size"]), ("img_size", data["img_size"])):
        expected = np.asarray(crop_item[key], dtype=np.float64)
        actual = np.asarray(observed, dtype=np.float64)
        if not np.allclose(expected, actual, atol=1e-5):
            raise RuntimeError(f"WiLoR {key} differs from preserved manifest")
    return crop, crop_item


async def call_wilor(args: argparse.Namespace, client: httpx.AsyncClient, router: ModelServiceRouter, crop: np.ndarray, item: Mapping[str, Any]) -> dict[str, Any]:
    own = ownership(args.run_id, "wilor.reconstruct", f"crop-{item['index']}", f"egoscale-crop-{item['index']}", float(item["index"]) / 30.0)
    metadata = {
        "ownership": own,
        "handedness": int(item["handedness"]),
        "box_center": item["box_center"],
        "box_size": item["box_size"],
        "img_size": item["img_size"],
        "source_K_px": None,
        "model_revision": router.endpoint_for(ModelApiName.WILOR_RECONSTRUCT).model_revision,
        "options": {},
    }
    result = await post_fields(client, api_name="wilor.reconstruct", url=router.url_for(ModelApiName.WILOR_RECONSTRUCT), metadata=metadata,
                               fields={"crop": (crop.tobytes(), crop.shape, "float32")})
    result["payload"] = {"manifest": str(args.gpu1_root / "wilor_crop_manifest.json"), "index": item["index"], "sha256": item["sha256"]}
    return result


def camera_contract() -> dict[str, Any]:
    eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return {
        "intrinsics": [408.96, 408.96, 284.0, 160.0],
        "K_px": None,
        "source_size": {"width": 568, "height": 320},
        "pixel_transform": {"source_to_model": eye, "model_to_source": eye, "resize_mode": "identity"},
    }


async def call_droid(args: argparse.Namespace, client: httpx.AsyncClient, router: ModelServiceRouter) -> dict[str, Any]:
    manifest = read_json(args.droid_manifest)
    endpoint = router.base_url_for(ModelApiName.DROID_CREATE_SESSION)
    source_id = "integrated-droid-real-sequence"
    create_own = ownership(args.run_id, "droid.create_session", source_id, source_id)
    create_meta = {
        "ownership": create_own,
        "camera": camera_contract(),
        "image_shape": {"height": 320, "width": 568},
        "options": {"buffer": 128, "filter_thresh": 1.0, "keyframe_thresh": 2.0, "warmup": 2},
        "model_revision": router.endpoint_for(ModelApiName.DROID_CREATE_SESSION).model_revision,
    }
    create = await post_json_or_multipart(client, api_name="droid.create_session", url=router.url_for(ModelApiName.DROID_CREATE_SESSION), metadata=create_meta, fields={})
    session_id = (create.get("response_metadata") or {}).get("session_id")
    pushes: list[dict[str, Any]] = []
    if session_id:
        for payload in manifest["payloads"][: args.droid_frames]:
            rgb = Path(payload["rgb_path"]).read_bytes()
            mask = Path(payload["mask_path"]).read_bytes()
            verify_manifest_hash(rgb, payload["rgb_sha256"], f"DROID {payload['payload_id']} RGB")
            verify_manifest_hash(mask, payload["mask_sha256"], f"DROID {payload['payload_id']} mask")
            push_own = ownership(args.run_id, "droid.push_frame", payload["payload_id"], f"{source_id}:{payload['source_frame_index']}", payload["timestamp_s"])
            push_meta = {
                "ownership": push_own,
                "session_id": session_id,
                "frame_id": f"frame-{payload['source_frame_index']}",
                "source_timestamp_s": payload["timestamp_s"],
                "model_revision": router.endpoint_for(ModelApiName.DROID_PUSH_FRAME).model_revision,
            }
            pushes.append(await post_json_or_multipart(
                client,
                api_name="droid.push_frame",
                url=router.url_for(ModelApiName.DROID_PUSH_FRAME),
                metadata=push_meta,
                fields={
                    "rgb": (rgb, (320, 568, 3), "uint8"),
                    "static_confidence_mask": (mask, (320, 568), "float32"),
                },
            ))
    finalize_own = ownership(args.run_id, "droid.finalize", source_id, source_id)
    finalize_meta = {
        "ownership": finalize_own,
        "session_id": session_id or "create-failed-no-session",
        "model_revision": router.endpoint_for(ModelApiName.DROID_FINALIZE).model_revision,
    }
    finalize = await post_json_or_multipart(client, api_name="droid.finalize", url=router.url_for(ModelApiName.DROID_FINALIZE), metadata=finalize_meta, fields={})
    all_calls = [create, *pushes, finalize]
    return {
        "api_name": "droid",
        "endpoint": endpoint,
        "session_id": session_id,
        "create": create,
        "pushes": pushes,
        "finalize": finalize,
        "http_status": finalize.get("http_status"),
        "latency_ms": sum(float(row.get("latency_ms") or 0.0) for row in all_calls),
        "request_ownership": create_own,
        "response_ownership": finalize.get("response_ownership"),
        "model_revision": finalize.get("model_revision") or create.get("model_revision"),
        "typed_error": next((row["typed_error"] for row in all_calls if row.get("typed_error")), None),
        "success": bool(session_id) and len(pushes) == args.droid_frames and all(row.get("success") for row in all_calls),
        "payload": {
            "manifest": str(args.droid_manifest),
            "source_video": manifest["source_video"],
            "frames_requested": args.droid_frames,
            "rgb_hashes": [item["rgb_sha256"] for item in manifest["payloads"][: args.droid_frames]],
        },
    }


def quaternion_matrix(q_xyzw: Sequence[float]) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def droid_evidence(sequence: Path) -> tuple[np.ndarray, np.ndarray, float]:
    slam = np.load(sequence / "SLAM/hawor_slam_w_scale_0_30.npz", allow_pickle=True)
    trajectory = slam["traj"]
    scale = float(slam["scale"])
    poses = np.zeros((trajectory.shape[0], 4, 4), dtype=np.float32)
    for index, row in enumerate(trajectory):
        poses[index, :3, :3] = quaternion_matrix(row[3:])
        poses[index, :3, 3] = row[:3] * scale
        poses[index, 3, 3] = 1.0
    timestamps = np.arange(trajectory.shape[0], dtype=np.float64) / 30.0
    return poses, timestamps, scale


def build_hawor_payload(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, tuple[bytes, Sequence[int], str]], dict[str, Any]]:
    sys.path.insert(0, str(args.hawor_repo))
    from lib.utils.imutils import boxes_2_cs, crop

    sequence = args.hawor_sequence
    tracks = np.load(sequence / "tracks_0_30/model_tracks.npy", allow_pickle=True).item()
    focal = float(read_json(sequence / "v19_hawor_focal_cache_contract.json")["img_focal"])
    images = [sequence / "extracted_images" / f"{index:04d}.jpg" for index in range(30)]
    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    height, width = first.shape[:2]
    boxes = np.concatenate([row["det_box"] for row in tracks[1.0]])[:16]
    centers, scales = boxes_2_cs(boxes)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    crops: list[np.ndarray] = []
    transforms: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    eye = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for index in range(16):
        image = cv2.imread(str(images[index]), cv2.IMREAD_COLOR)[:, :, ::-1]
        image_crop = crop(image, centers[index], scales[index] * 1.2, [256, 256], rot=0).astype(np.uint8)
        tensor = image_crop.transpose(2, 0, 1).astype(np.float32) / 255.0
        crops.append((tensor - mean) / std)
        transforms.append({
            "center": [float(centers[index][0]), float(centers[index][1])],
            "scale": float(scales[index]),
            "img_focal": focal,
            "img_center": [width / 2.0, height / 2.0],
            "do_flip": False,
            "source_size": {"width": width, "height": height},
            "pixel_transform": {"source_to_model": eye, "model_to_source": eye, "resize_mode": "identity"},
        })
        observations.append({
            "frame_index": index,
            "source_timestamp_s": index / 30.0,
            "occlusion_state": "visible",
            "detection_confidence": float(boxes[index, 4]),
            "side": "right",
        })
    crop_batch = np.ascontiguousarray(np.stack(crops).astype(np.float32))
    poses, timestamps, scale = droid_evidence(sequence)
    metadata = {
        "track_id": "track-integrated-right",
        "side": "right",
        "crop_transforms": transforms,
        "observations": observations,
        "unidepth": {
            "K_px": [[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]],
            "img_focal": focal,
            "img_center": [width / 2.0, height / 2.0],
            "source_size": {"width": width, "height": height},
            "metric_scale": 1.0,
            "source": "unidepth_v2_vitl14",
        },
        "droid_evidence": {"metric_scale": scale, "scale_residual": 0.001, "scale_confidence": 0.95, "source": "droid+unidepth_scale"},
        "model_revision": "hawor-v1",
        "options": {},
    }
    fields = {
        "crop_batch": (crop_batch.tobytes(), crop_batch.shape, "float32"),
        "droid_poses": (poses.tobytes(), poses.shape, "float32"),
        "droid_timestamps": (timestamps.tobytes(), timestamps.shape, "float64"),
    }
    source = {
        "manifest": str(args.hawor_benchmark),
        "sequence": str(sequence),
        "crop_sha256": sha256(crop_batch.tobytes()),
        "droid_pose_sha256": sha256(poses.tobytes()),
    }
    return metadata, fields, source


async def call_hawor(args: argparse.Namespace, client: httpx.AsyncClient, router: ModelServiceRouter,
                     base_meta: Mapping[str, Any], fields: Mapping[str, tuple[bytes, Sequence[int], str]], source: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(base_meta)
    metadata["ownership"] = ownership(args.run_id, "hawor.infer_tracks", "track-integrated-right", args.hawor_sequence.name, 0.0)
    metadata["model_revision"] = router.endpoint_for(ModelApiName.HAWOR_INFER_TRACKS).model_revision
    result = await post_fields(client, api_name="hawor.infer_tracks", url=router.url_for(ModelApiName.HAWOR_INFER_TRACKS), metadata=metadata, fields=fields)
    result["payload"] = dict(source)
    return result


def build_infiller_payload(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, tuple[bytes, Sequence[int], str]], dict[str, Any]]:
    sequence = args.hawor_sequence
    left = read_json(sequence / "cam_space/0/0_29.json")
    right = read_json(sequence / "cam_space/1/0_29.json")
    frames: list[dict[str, Any]] = []
    for index in range(30):
        for side, camera in (("left", left), ("right", right)):
            root = np.asarray(camera["init_root_orient"])[0, index]
            hand_rotations = np.asarray(camera["init_hand_pose"])[0, index]
            hand_pose = np.stack([cv2.Rodrigues(matrix.astype(np.float64))[0].reshape(3) for matrix in hand_rotations])
            frames.append({
                "frame_index": index,
                "source_timestamp_s": index / 30.0,
                "side": side,
                "root_orient": root.tolist(),
                "hand_pose": hand_pose.tolist(),
                "trans": np.asarray(camera["init_trans"])[0, index].tolist(),
                "betas": np.asarray(camera["init_betas"])[0, index].tolist(),
                "observed": index % 7 != 0,
                "uncertainty": 0.005,
            })
    poses, timestamps, scale = droid_evidence(sequence)
    focal, width, height = 1680.10606401, 1920, 1080
    metadata = {
        "window_id": "window-integrated",
        "frames": frames,
        "unidepth": {
            "K_px": [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
            "img_focal": focal,
            "img_center": [width / 2, height / 2],
            "source_size": {"width": width, "height": height},
            "metric_scale": 1.0,
            "source": "unidepth_v2_vitl14",
        },
        "droid_evidence": {"metric_scale": scale, "scale_residual": 0.001, "scale_confidence": 0.95, "source": "droid+unidepth_scale"},
        "model_revision": "hawor-infiller-v1",
        "options": {},
    }
    fields = {
        "droid_poses": (poses.tobytes(), poses.shape, "float32"),
        "droid_timestamps": (timestamps.tobytes(), timestamps.shape, "float64"),
    }
    source = {
        "manifest": str(args.hawor_benchmark),
        "sequence": str(sequence),
        "camera_state_sha256": sha256(json.dumps(frames, sort_keys=True).encode("utf-8")),
        "droid_pose_sha256": sha256(poses.tobytes()),
    }
    return metadata, fields, source


async def call_infiller(args: argparse.Namespace, client: httpx.AsyncClient, router: ModelServiceRouter,
                        base_meta: Mapping[str, Any], fields: Mapping[str, tuple[bytes, Sequence[int], str]], source: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(base_meta)
    metadata["ownership"] = ownership(args.run_id, "hawor_infiller.fill", "window-integrated", args.hawor_sequence.name, 0.0)
    metadata["model_revision"] = router.endpoint_for(ModelApiName.HAWOR_INFILLER_FILL).model_revision
    result = await post_fields(client, api_name="hawor_infiller.fill", url=router.url_for(ModelApiName.HAWOR_INFILLER_FILL), metadata=metadata, fields=fields)
    result["payload"] = dict(source)
    return result


async def call_cosmos(args: argparse.Namespace, client: httpx.AsyncClient, router: ModelServiceRouter) -> dict[str, Any]:
    manifest = read_json(args.cosmos_manifest)
    item = manifest["media_items"][0]
    media = Path(item["path"]).read_bytes()
    verify_manifest_hash(media, item["sha256"], "Cosmos3 media")
    own = ownership(args.run_id, "cosmos3.reason", item["item_id"], item["source_id"], item["source_timestamp_s"])
    metadata = {
        "ownership": own,
        "prompt": item["prompt"],
        "messages": [],
        "generation": {"max_tokens": 96, "temperature": 0.0, "top_p": 1.0},
    }
    body, content_type = build_cosmos3_request(metadata, [(media, "image", item["media_type"], 0)])
    started = time.monotonic()
    try:
        response = await client.post(router.url_for(ModelApiName.COSMOS3_REASON), content=body, headers={"Content-Type": content_type})
        wire = parse_cosmos3_response(response.content, response.headers.get("content-type", ""))
        error = typed_error(wire)
        result = {
            "api_name": "cosmos3.reason",
            "endpoint": router.url_for(ModelApiName.COSMOS3_REASON),
            "http_status": response.status_code,
            "latency_ms": (time.monotonic() - started) * 1000.0,
            "request_ownership": own,
            "response_ownership": response_ownership(wire),
            "model_revision": response_revision(wire),
            "typed_error": error,
            "trace": response_trace(wire),
            "response_metadata": wire,
            "response_arrays": {},
            "success": response.status_code == 200 and error is None,
        }
    except Exception as exc:
        result = {
            "api_name": "cosmos3.reason",
            "endpoint": router.url_for(ModelApiName.COSMOS3_REASON),
            "http_status": None,
            "latency_ms": (time.monotonic() - started) * 1000.0,
            "request_ownership": own,
            "response_ownership": None,
            "model_revision": None,
            "typed_error": {"code": "transport", "message": repr(exc), "retryable": False},
            "trace": None,
            "response_arrays": {},
            "success": False,
        }
    result["payload"] = {"manifest": str(args.cosmos_manifest), "item_id": item["item_id"], "media_sha256": item["sha256"]}
    return result


async def endpoint_health(client: httpx.AsyncClient, router: ModelServiceRouter) -> list[dict[str, Any]]:
    async def one(gpu_id: int, base_url: str, api_names: list[str]) -> dict[str, Any]:
        path = "/droid/status" if gpu_id == 2 else "/-/routes"
        started = time.monotonic()
        try:
            response = await client.get(base_url + path)
            try:
                body: Any = response.json()
            except Exception:
                body = response.text[:2000]
            return {
                "gpu_id": gpu_id,
                "base_url": base_url,
                "probe_path": path,
                "api_names": api_names,
                "http_status": response.status_code,
                "latency_ms": (time.monotonic() - started) * 1000.0,
                "body": body,
                "healthy": response.status_code < 500,
            }
        except Exception as exc:
            return {
                "gpu_id": gpu_id,
                "base_url": base_url,
                "probe_path": path,
                "api_names": api_names,
                "http_status": None,
                "latency_ms": (time.monotonic() - started) * 1000.0,
                "body": None,
                "healthy": False,
                "error": repr(exc),
            }

    by_gpu: dict[int, tuple[str, list[str]]] = {}
    for api in router.all_apis():
        endpoint = router.endpoint_for(api)
        base, names = by_gpu.setdefault(endpoint.gpu_id, (router.base_url_for(api), []))
        names.append(api.value)
    return await asyncio.gather(*(one(gpu, base, names) for gpu, (base, names) in sorted(by_gpu.items())))


def gpu_snapshot() -> dict[str, Any]:
    query = "index,uuid,name,memory.used,memory.total"
    command = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    raw = subprocess.check_output(command, text=True)
    gpus = []
    for line in raw.splitlines():
        index, gpu_uuid, name, used, total = [item.strip() for item in line.split(",", 4)]
        gpus.append({"index": int(index), "uuid": gpu_uuid, "name": name, "memory_used_mib": int(used), "memory_total_mib": int(total)})
    process_command = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]
    processes = []
    for line in subprocess.check_output(process_command, text=True).splitlines():
        gpu_uuid, pid, memory = [item.strip() for item in line.split(",", 2)]
        processes.append({"gpu_uuid": gpu_uuid, "pid": int(pid), "memory_used_mib": int(memory)})
    return {"captured_at": utc_now(), "assignments": GPU_ASSIGNMENTS, "gpus": gpus, "compute_processes": processes}


def validate_ownership(record: Mapping[str, Any]) -> str | None:
    request = record.get("request_ownership")
    response = record.get("response_ownership")
    if not isinstance(request, Mapping) or not isinstance(response, Mapping):
        return "missing request or response ownership"
    for key in ("request_id", "job_id", "item_id", "stage_id", "source_id"):
        if request.get(key) != response.get(key):
            return f"ownership mismatch for {key}: request={request.get(key)!r}, response={response.get(key)!r}"
    return None


async def async_main(args: argparse.Namespace) -> int:
    artifact_root = args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=False)
    router = ModelServiceRouter.canonical()

    # Payload preparation is CPU-only and completed before the concurrent network
    # gate so all seven service tasks become runnable together.
    gpu1_rgb, gpu1_item = load_gpu1_frame(args)
    wilor_crop, wilor_item = build_wilor_crop(args, gpu1_rgb)
    hawor_meta, hawor_fields, hawor_source = build_hawor_payload(args)
    infiller_meta, infiller_fields, infiller_source = build_infiller_payload(args)

    gpu_before = gpu_snapshot()
    timeout = httpx.Timeout(args.timeout_s)
    async with httpx.AsyncClient(timeout=timeout) as client:
        health = await endpoint_health(client, router)
        tasks = [
            call_unidepth(args, client, router),
            call_hands(args, client, router, gpu1_rgb, gpu1_item),
            call_wilor(args, client, router, wilor_crop, wilor_item),
            call_droid(args, client, router),
            call_hawor(args, client, router, hawor_meta, hawor_fields, hawor_source),
            call_infiller(args, client, router, infiller_meta, infiller_fields, infiller_source),
            call_cosmos(args, client, router),
        ]
        launched = time.monotonic()
        responses = list(await asyncio.gather(*tasks))
        concurrent_wall_ms = (time.monotonic() - launched) * 1000.0
    gpu_after = gpu_snapshot()

    ownership_errors: dict[str, Any] = {}
    for record in responses:
        if record["api_name"] == "droid":
            checks = [record["create"], *record["pushes"], record["finalize"]]
            errors = [error for error in (validate_ownership(item) for item in checks) if error]
            if errors:
                ownership_errors["droid"] = errors
        else:
            error = validate_ownership(record)
            if error:
                ownership_errors[record["api_name"]] = error

    endpoint_outcomes = []
    for record in responses:
        service = next(service for service in GPU_ASSIGNMENTS if service == record["api_name"] or (record["api_name"] == "droid" and service == "droid")) if record["api_name"] in GPU_ASSIGNMENTS or record["api_name"] == "droid" else service_for_api(record["api_name"]).name
        endpoint_outcomes.append({
            "service": record["api_name"],
            "gpu_id": GPU_ASSIGNMENTS[service],
            "endpoint": record["endpoint"],
            "success": record["success"],
            "http_status": record.get("http_status"),
            "latency_ms": record.get("latency_ms"),
            "model_revision": record.get("model_revision"),
            "typed_error": record.get("typed_error"),
        })

    source_manifests = {
        "unidepth": str(args.unidepth_manifest),
        "hands": str(args.gpu1_root / "payload_manifest.json"),
        "wilor": str(args.gpu1_root / "wilor_crop_manifest.json"),
        "droid": str(args.droid_manifest),
        "hawor_tracks": str(args.hawor_benchmark),
        "hawor_infiller": str(args.hawor_benchmark),
        "cosmos3": str(args.cosmos_manifest),
    }
    run_manifest = {
        "schema": "ego.integrated-client-exercise.v1",
        "run_id": args.run_id,
        "created_at": utc_now(),
        "integration_revision": args.integration_revision,
        "client_workspace": str(Path.cwd()),
        "artifact_root": str(artifact_root),
        "concurrent_wall_ms": concurrent_wall_ms,
        "source_manifests": source_manifests,
        "gpu_assignments": GPU_ASSIGNMENTS,
        "constraints": {"services_redeployed": False, "services_restarted": False, "global_ray_commands": False},
    }
    summary = {
        "run_id": args.run_id,
        "integration_revision": args.integration_revision,
        "all_endpoint_health_probes_passed": all(row["healthy"] for row in health),
        "all_service_calls_passed": all(row["success"] for row in responses),
        "ownership_errors": ownership_errors,
        "concurrent_wall_ms": concurrent_wall_ms,
        "endpoint_outcomes": endpoint_outcomes,
    }
    (artifact_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (artifact_root / "endpoint_health.json").write_text(json.dumps(health, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (artifact_root / "gpu_mapping.json").write_text(json.dumps({"before": gpu_before, "after": gpu_after}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (artifact_root / "responses.json").write_text(json.dumps(responses, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (artifact_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_endpoint_health_probes_passed"] and summary["all_service_calls_passed"] and not ownership_errors else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--integration-revision", required=True)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--droid-frames", type=int, default=12)
    parser.add_argument("--unidepth-manifest", type=Path, default=DEFAULT_UNIDEPTH_MANIFEST)
    parser.add_argument("--unidepth-frame", type=Path, default=DEFAULT_UNIDEPTH_FRAME)
    parser.add_argument("--gpu1-root", type=Path, default=DEFAULT_GPU1_ROOT)
    parser.add_argument("--gpu1-frames", type=Path, default=DEFAULT_GPU1_FRAMES)
    parser.add_argument("--wilor-root", type=Path, default=DEFAULT_WILOR_ROOT)
    parser.add_argument("--wilor-config", type=Path, default=DEFAULT_WILOR_CONFIG)
    parser.add_argument("--droid-manifest", type=Path, default=DEFAULT_DROID_MANIFEST)
    parser.add_argument("--hawor-benchmark", type=Path, default=DEFAULT_HAWOR_BENCHMARK)
    parser.add_argument("--hawor-repo", type=Path, default=DEFAULT_HAWOR_REPO)
    parser.add_argument("--hawor-sequence", type=Path, default=DEFAULT_HAWOR_SEQUENCE)
    parser.add_argument("--cosmos-manifest", type=Path, default=DEFAULT_COSMOS_MANIFEST)
    args = parser.parse_args(argv)
    expected_parent = BENCHMARK_ROOT
    if expected_parent not in args.artifact_root.parents:
        parser.error(f"artifact root must be a fresh child of {expected_parent}")
    if args.droid_frames < 2:
        parser.error("--droid-frames must be at least 2")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main(parse_args())))
