#!/usr/bin/env python3
"""Client-only concurrent stress check for the seven resident model services.

The command prepares real model-native payloads before the network gate, then starts
UniDepth, Hands, WiLoR, DROID's full create→16 push→finalize lifecycle, HaWoR,
HaWoR Infiller, and Cosmos concurrently.  It never imports Ray or changes a service.

Example (on A800, using the environment that owns the GPU1 WiLoR preprocessing deps):

    /home/zjh/miniconda3/envs/ray_serve_hands/bin/python -m scripts.parallel_stress_test

Payload locations are CLI-overridable and can also be set through the documented
EGO_STRESS_* environment variables.  The JSON artifact records exact input paths and
hashes; ``summary.csv`` is a compact review table.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence
import uuid

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx

from ego_annotation.serving.router import ModelApiName, ModelServiceRouter
from ego_annotation.serving.transport import build_multipart_request_fields, parse_multipart_response


ARTIFACT_BASE = Path("/home/zjh/ray_serve_benchmarks")
DEFAULT_UNIDEPTH_FRAME = Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_lanes/gpu0_unidepth_a77a57e_20260716_190232/egoscale_frames/frame_A_idx000.npy")
DEFAULT_GPU1_ROOT = Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_benchmarks/gpu1_hands_wilor_20260716T2001Z")
DEFAULT_GPU1_FRAMES = Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/data/gpuheavy_tensor_batch_bench/20260715T_tensor_batch_v1/hawor/hawor_frames")
DEFAULT_WILOR_ROOT = Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor_model")
DEFAULT_WILOR_CONFIG = Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor/model_config.yaml")
DEFAULT_DROID_MANIFEST = Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_benchmarks/droid_openloop_final_20260716T204043Z/droid/payload_manifest.json")
DEFAULT_HAWOR_ROOTS = (
    Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/v2_benchmarks/hawor_20260723T100724Z"),
    Path("/vePFS-Mindverse/user/yiwen/user-home/zjh/v2_benchmarks/hawor_20260723T100544Z"),
)
DEFAULT_INFILLER_REQUEST = Path("/home/zjh/data/v22_api_jobs/annotation_83445831a32a/stage_captures/hawor_infiller_fill/69ac36dc774070a0579e698a318c48e9ca035b0ab723dd80d6ae141980d68b64/cc841db9282dcb7ae9d818c8d0862e21234f4dc5c4e006f3fb9b5ac2fc029d92/request.multipart")
DEFAULT_COSMOS_REQUEST = Path("/home/zjh/ego_model_services_runtime/current/assets/cosmos3/representative_request.multipart")
DEFAULT_COSMOS_HEADERS = Path("/home/zjh/ego_model_services_runtime/current/assets/cosmos3/representative_request_headers.json")
# The live A800 topology splits WiLoR from the GPU1 Hands route.  These remain
# explicit overrides rather than changing the repository-wide canonical router.
DEFAULT_WILOR_ENDPOINT = "http://127.0.0.1:28004/wilor.reconstruct"
DEFAULT_HANDS_REVISION = "hands-yolo-v2"

GPU_MAPPING = {
    "unidepth": 0,
    "hands": 1,
    "wilor": 1,
    "droid": 2,
    "hawor": 3,
    "infiller": 3,
    "cosmos": 6,
}
SERVICE_ALIASES = {
    "unidepth": "unidepth",
    "unidepth.infer": "unidepth",
    "hands": "hands",
    "hands.detect": "hands",
    "wilor": "wilor",
    "wilor.reconstruct": "wilor",
    "droid": "droid",
    "droid.create_session": "droid",
    "hawor": "hawor",
    "hawor.infer_tracks": "hawor",
    "infiller": "infiller",
    "hawor_infiller.fill": "infiller",
    "cosmos": "cosmos",
    "cosmos3.reason": "cosmos",
}
DEFAULT_SERVICES = tuple(GPU_MAPPING)


@dataclass(frozen=True)
class PreparedOperation:
    service: str
    endpoint: str
    invoke: Callable[[httpx.AsyncClient, str], Awaitable[dict[str, Any]]]
    payload: Mapping[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_existing(label: str, preferred: Path | None, fallbacks: Sequence[Path]) -> Path:
    """Select an explicit override or the first verified fallback; never invent paths."""
    candidates = ((preferred,) if preferred is not None else ()) + tuple(fallbacks)
    for path in candidates:
        if path.exists():
            return path
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{label} is unavailable; checked: {rendered}")


def env_path(name: str, cli_value: Path | None) -> Path | None:
    if cli_value is not None:
        return cli_value
    value = os.environ.get(name)
    return Path(value) if value else None


def ownership(run_id: str, api_name: str, item_id: str, source_id: str, timestamp_s: float | None = None) -> dict[str, Any]:
    return {
        "request_id": f"{run_id}-{api_name}-{uuid.uuid4().hex[:10]}",
        "job_id": f"parallel-stress-{run_id}",
        "item_id": item_id,
        "stage_id": api_name,
        "source_id": source_id,
        "schema_version": "ego.model-service.v1",
        "source_timestamp_s": timestamp_s,
        "submitted_at": utc_now(),
    }


def identity_spatial(width: int, height: int) -> dict[str, Any]:
    eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return {
        "source_size": {"width": width, "height": height},
        "model_size": {"width": width, "height": height},
        "color_space": "RGB",
        "pixel_transform": {"source_to_model": eye, "model_to_source": eye, "resize_mode": "canonical_input", "crop_xywh": None, "pad_ltrb": None},
        "K_px": None,
    }


def response_error(wire: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = wire.get("error")
    return value if isinstance(value, Mapping) else None


def response_trace(wire: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for candidate in (wire, wire.get("result"), wire.get("status"), wire.get("camera_state")):
        if isinstance(candidate, Mapping) and isinstance(candidate.get("trace"), Mapping):
            return candidate["trace"]
    return None


def batch_wait_ms(trace: Mapping[str, Any] | None) -> float | None:
    if not trace:
        return None
    admitted = trace.get("admitted_monotonic_s")
    dispatched = trace.get("dispatched_monotonic_s")
    if isinstance(admitted, (int, float)) and isinstance(dispatched, (int, float)):
        return (float(dispatched) - float(admitted)) * 1000.0
    for key in ("batch_wait_ms", "queue_wait_ms"):
        value = trace.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def compact_wire(wire: Mapping[str, Any]) -> dict[str, Any]:
    """Keep artifacts reviewable without duplicating multi-MB tensor response metadata."""
    return {
        key: value
        for key, value in wire.items()
        if key in {"error", "result", "status", "camera_state", "model_revision", "ownership", "session_id"}
    }


async def post_raw(client: httpx.AsyncClient, *, service: str, endpoint: str, body: bytes, content_type: str) -> dict[str, Any]:
    started = time.monotonic()
    try:
        response = await client.post(endpoint, content=body, headers={"Content-Type": content_type})
        latency_ms = (time.monotonic() - started) * 1000.0
        content_type_response = response.headers.get("content-type", "")
        try:
            if "multipart/" in content_type_response:
                wire, arrays = parse_multipart_response(response.content, content_type_response)
            else:
                decoded = response.json()
                wire = decoded if isinstance(decoded, Mapping) else {"body": decoded}
                arrays = {}
        except Exception as exc:
            wire, arrays = {"parse_error": repr(exc), "body_excerpt": response.text[:1000]}, {}
        error = response_error(wire)
        trace = response_trace(wire)
        return {
            "service": service,
            "endpoint": endpoint,
            "http_status": response.status_code,
            "latency_ms": latency_ms,
            "success": response.status_code == 200 and error is None and "parse_error" not in wire,
            "typed_error": error,
            "result_hash": sha256(response.content),
            "batch_wait_ms": batch_wait_ms(trace),
            "trace": trace,
            "response": compact_wire(wire),
            "response_arrays": {name: {"shape": list(shape), "dtype": dtype, "sha256": sha256(data)} for name, (data, shape, dtype) in arrays.items()},
        }
    except Exception as exc:
        return {
            "service": service,
            "endpoint": endpoint,
            "http_status": None,
            "latency_ms": (time.monotonic() - started) * 1000.0,
            "success": False,
            "typed_error": {"code": "transport", "message": repr(exc), "retryable": False},
            "result_hash": None,
            "batch_wait_ms": None,
            "trace": None,
            "response": {},
            "response_arrays": {},
        }


def multipart_operation(service: str, endpoint: str, metadata: Mapping[str, Any], fields: Mapping[str, tuple[bytes, Sequence[int], str]], payload: Mapping[str, Any]) -> PreparedOperation:
    body, content_type = build_multipart_request_fields(metadata, fields)

    async def invoke(client: httpx.AsyncClient, _run_id: str) -> dict[str, Any]:
        return await post_raw(client, service=service, endpoint=endpoint, body=body, content_type=content_type)

    return PreparedOperation(service, endpoint, invoke, {**payload, "request_sha256": sha256(body), "request_bytes": len(body)})


def replay_operation(service: str, endpoint: str, request_path: Path, payload: Mapping[str, Any]) -> PreparedOperation:
    body = request_path.read_bytes()
    if not body.startswith(b"--"):
        raise ValueError(f"{service} replay body is not multipart: {request_path}")
    boundary = body.split(b"\r\n", 1)[0][2:].decode("ascii")
    content_type = f"multipart/form-data; boundary={boundary}"

    async def invoke(client: httpx.AsyncClient, _run_id: str) -> dict[str, Any]:
        return await post_raw(client, service=service, endpoint=endpoint, body=body, content_type=content_type)

    return PreparedOperation(service, endpoint, invoke, {**payload, "request_path": str(request_path), "request_sha256": sha256(body), "request_bytes": len(body)})


def build_wilor_crop(gpu1_root: Path, gpu1_frames: Path, wilor_root: Path, wilor_config: Path) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    """Reconstruct the preserved real crop with WiLoR's upstream ViTDetDataset."""
    import cv2
    import numpy as np

    image_item = read_json(gpu1_root / "payload_manifest.json")[0]
    crop_item = read_json(gpu1_root / "wilor_crop_manifest.json")[0]
    evidence = read_json(gpu1_root / "results.json")
    detection = next(row for row in evidence["detect_evidence"] if int(row["index"]) == int(crop_item["index"]))
    bgr = cv2.imread(str(gpu1_frames / image_item["source_file"]), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot decode WiLoR source frame {image_item['source_file']}")
    rgb = np.ascontiguousarray(cv2.cvtColor(cv2.resize(bgr, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    if sha256(rgb.tobytes()) != image_item["rgb_sha256"]:
        raise RuntimeError("WiLoR source RGB hash disagrees with preserved payload manifest")
    prior_cwd = Path.cwd()
    try:
        os.chdir(wilor_root)
        if str(wilor_root) not in sys.path:
            sys.path.insert(0, str(wilor_root))
        from wilor.configs import get_config
        from wilor.datasets.vitdet_dataset import ViTDetDataset

        config = get_config(str(wilor_config), update_cachedir=True)
        config.defrost()
        if "BBOX_SHAPE" not in config.MODEL:
            config.MODEL.BBOX_SHAPE = [192, 256]
        config.freeze()
        item = ViTDetDataset(config, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), np.asarray([detection["boxes"][0]], dtype=np.float32), np.asarray([crop_item["handedness"]], dtype=np.float32), rescale_factor=2.0)[0]
        crop = item["img"]
        if hasattr(crop, "numpy"):
            crop = crop.numpy()
        crop = np.ascontiguousarray(np.asarray(crop, dtype=np.float32))
    finally:
        os.chdir(prior_cwd)
    if sha256(crop.tobytes()) != crop_item["sha256"]:
        raise RuntimeError("WiLoR crop hash disagrees with preserved crop manifest")
    for key in ("box_center", "box_size", "img_size"):
        if not np.allclose(np.asarray(item[key], dtype=np.float64), np.asarray(crop_item[key], dtype=np.float64), atol=1e-5):
            raise RuntimeError(f"WiLoR preserved {key} disagrees with upstream crop reconstruction")
    return crop, image_item, crop_item


def load_hands_rgb(gpu1_root: Path, gpu1_frames: Path) -> tuple[Any, Mapping[str, Any]]:
    import cv2
    import numpy as np

    item = read_json(gpu1_root / "payload_manifest.json")[0]
    bgr = cv2.imread(str(gpu1_frames / item["source_file"]), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot decode Hands payload {item['source_file']}")
    rgb = np.ascontiguousarray(cv2.cvtColor(cv2.resize(bgr, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    if sha256(rgb.tobytes()) != item["rgb_sha256"]:
        raise RuntimeError("Hands RGB hash disagrees with preserved payload manifest")
    return rgb, item


def droid_camera_contract() -> dict[str, Any]:
    eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return {"intrinsics": [408.96, 408.96, 284.0, 160.0], "K_px": None, "source_size": {"width": 568, "height": 320}, "pixel_transform": {"source_to_model": eye, "model_to_source": eye, "resize_mode": "identity"}}


def droid_operation(endpoint_base: str, manifest_path: Path, router: ModelServiceRouter) -> PreparedOperation:
    manifest = read_json(manifest_path)
    payloads = manifest["payloads"][:16]
    if len(payloads) != 16:
        raise ValueError(f"DROID manifest must contain at least 16 payloads: {manifest_path}")

    async def invoke(client: httpx.AsyncClient, run_id: str) -> dict[str, Any]:
        started = time.monotonic()
        source_id = "parallel-stress-droid-sequence"
        create_meta = {"ownership": ownership(run_id, "droid.create_session", source_id, source_id), "camera": droid_camera_contract(), "image_shape": {"height": 320, "width": 568}, "options": {"buffer": 256, "filter_thresh": 1.0, "keyframe_thresh": 2.0, "warmup": 8}, "model_revision": router.endpoint_for(ModelApiName.DROID_CREATE_SESSION).model_revision}
        create_body, create_ct = build_multipart_request_fields(create_meta, {})
        create = await post_raw(client, service="droid.create_session", endpoint=f"{endpoint_base}/droid.create_session", body=create_body, content_type=create_ct)
        session_id = (create.get("response") or {}).get("session_id")
        if not isinstance(session_id, str) or not session_id:
            # The dispatcher status contains the current lane/lease view.  Capture it
            # once before reporting create failure; never issue blind replacement creates.
            status = await post_droid_status(client, endpoint_base)
            return droid_result(endpoint_base, started, create, [], None, manifest_path, payloads, status)
        pushes: list[dict[str, Any]] = []
        for payload in payloads:
            rgb = Path(payload["rgb_path"]).read_bytes()
            mask = Path(payload["mask_path"]).read_bytes()
            if sha256(rgb) != payload["rgb_sha256"] or sha256(mask) != payload["mask_sha256"]:
                raise RuntimeError(f"DROID payload hash mismatch: {payload['payload_id']}")
            metadata = {"ownership": ownership(run_id, "droid.push_frame", payload["payload_id"], f"{source_id}:{payload['source_frame_index']}", payload["timestamp_s"]), "session_id": session_id, "frame_id": f"frame-{payload['source_frame_index']}", "source_timestamp_s": payload["timestamp_s"], "model_revision": router.endpoint_for(ModelApiName.DROID_PUSH_FRAME).model_revision}
            body, content_type = build_multipart_request_fields(metadata, {"rgb": (rgb, (320, 568, 3), "uint8"), "static_confidence_mask": (mask, (320, 568), "float32")})
            pushes.append(await post_raw(client, service="droid.push_frame", endpoint=f"{endpoint_base}/droid.push_frame", body=body, content_type=content_type))
            if not pushes[-1]["success"]:
                break
        finalize: dict[str, Any] | None = None
        if len(pushes) == 16 and all(row["success"] for row in pushes):
            metadata = {"ownership": ownership(run_id, "droid.finalize", source_id, source_id), "session_id": session_id, "model_revision": router.endpoint_for(ModelApiName.DROID_FINALIZE).model_revision}
            body, content_type = build_multipart_request_fields(metadata, {})
            finalize = await post_raw(client, service="droid.finalize", endpoint=f"{endpoint_base}/droid.finalize", body=body, content_type=content_type)
        return droid_result(endpoint_base, started, create, pushes, finalize, manifest_path, payloads, None)

    return PreparedOperation("droid", endpoint_base, invoke, {"manifest": str(manifest_path), "frames": 16, "source_video": manifest.get("source_video"), "payload_ids": [item["payload_id"] for item in payloads]})


async def post_droid_status(client: httpx.AsyncClient, endpoint_base: str) -> dict[str, Any]:
    try:
        response = await client.get(f"{endpoint_base}/droid/status")
        try:
            body: Any = response.json()
        except Exception:
            body = response.text[:2000]
        return {"endpoint": f"{endpoint_base}/droid/status", "http_status": response.status_code, "body": body}
    except Exception as exc:
        return {"endpoint": f"{endpoint_base}/droid/status", "http_status": None, "error": repr(exc)}


def droid_result(endpoint_base: str, started: float, create: Mapping[str, Any], pushes: list[dict[str, Any]], finalize: Mapping[str, Any] | None, manifest_path: Path, payloads: Sequence[Mapping[str, Any]], create_failure_status: Mapping[str, Any] | None) -> dict[str, Any]:
    steps = [create, *pushes, *( [finalize] if finalize is not None else [])]
    error = next((row.get("typed_error") for row in steps if not row.get("success")), None)
    final_hash = (finalize or create).get("result_hash")
    waits = [row["batch_wait_ms"] for row in steps if isinstance(row.get("batch_wait_ms"), (int, float))]
    return {
        "service": "droid", "endpoint": endpoint_base, "http_status": (finalize or create).get("http_status"), "latency_ms": (time.monotonic() - started) * 1000.0,
        "success": bool(finalize and create.get("success") and len(pushes) == 16 and all(row.get("success") for row in pushes) and finalize.get("success")),
        "typed_error": error, "result_hash": final_hash, "batch_wait_ms": (sum(waits) / len(waits) if waits else None),
        "gpu_mapping": 2, "lifecycle": {"create": create, "pushes": pushes, "finalize": finalize, "create_failure_dispatcher_status": create_failure_status},
        "payload": {"manifest": str(manifest_path), "frames_requested": 16, "payload_hashes": [item["rgb_sha256"] for item in payloads]},
    }


def normalize_apis(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_SERVICES
    names: list[str] = []
    for supplied in raw.split(","):
        key = supplied.strip()
        if not key:
            continue
        canonical = SERVICE_ALIASES.get(key)
        if canonical is None:
            raise ValueError(f"unknown --apis value {key!r}; choose from {', '.join(DEFAULT_SERVICES)}")
        if canonical not in names:
            names.append(canonical)
    if not names:
        raise ValueError("--apis selected no services")
    return tuple(names)


def prepare_operations(args: argparse.Namespace, router: ModelServiceRouter, selected: Sequence[str]) -> dict[str, PreparedOperation]:
    """Build each selected real request once, before the concurrent network gate."""
    operations: dict[str, PreparedOperation] = {}
    endpoint = lambda api: (args.wilor_endpoint or os.environ.get("EGO_STRESS_WILOR_ENDPOINT") or DEFAULT_WILOR_ENDPOINT) if api is ModelApiName.WILOR_RECONSTRUCT else router.url_for(api)
    if "unidepth" in selected:
        import numpy as np
        frame_path = resolve_existing("UniDepth frame", env_path("EGO_STRESS_UNIDEPTH_FRAME", args.unidepth_frame), (DEFAULT_UNIDEPTH_FRAME,))
        rgb = np.ascontiguousarray(np.load(frame_path).astype(np.uint8))
        metadata = {"ownership": ownership(args.run_id, "unidepth.infer", frame_path.stem, frame_path.stem, 0.0), "spatial": identity_spatial(int(rgb.shape[1]), int(rgb.shape[0])), "model_revision": router.endpoint_for(ModelApiName.UNIDEPTH_INFER).model_revision, "options": {}}
        operations["unidepth"] = multipart_operation("unidepth", endpoint(ModelApiName.UNIDEPTH_INFER), metadata, {"rgb": (rgb.tobytes(), rgb.shape, "uint8")}, {"frame": str(frame_path), "rgb_sha256": sha256(rgb.tobytes())})
    if "hands" in selected or "wilor" in selected:
        gpu1_root = resolve_existing("Hands/WiLoR corpus", env_path("EGO_STRESS_GPU1_ROOT", args.gpu1_root), (DEFAULT_GPU1_ROOT,))
        gpu1_frames = resolve_existing("Hands/WiLoR frames", env_path("EGO_STRESS_GPU1_FRAMES", args.gpu1_frames), (DEFAULT_GPU1_FRAMES,))
    if "hands" in selected:
        rgb, item = load_hands_rgb(gpu1_root, gpu1_frames)
        metadata = {"ownership": ownership(args.run_id, "hands.detect", f"frame-{item['index']}", item["source_file"], float(item["index"]) / 30.0), "spatial": identity_spatial(960, 540), "model_revision": args.hands_revision or os.environ.get("EGO_STRESS_HANDS_REVISION") or DEFAULT_HANDS_REVISION, "options": {}}
        operations["hands"] = multipart_operation("hands", endpoint(ModelApiName.HANDS_DETECT), metadata, {"rgb": (rgb.tobytes(), rgb.shape, "uint8")}, {"manifest": str(gpu1_root / "payload_manifest.json"), "source_file": item["source_file"], "rgb_sha256": item["rgb_sha256"]})
    if "wilor" in selected:
        wilor_root = resolve_existing("WiLoR source", env_path("EGO_STRESS_WILOR_ROOT", args.wilor_root), (DEFAULT_WILOR_ROOT,))
        wilor_config = resolve_existing("WiLoR config", env_path("EGO_STRESS_WILOR_CONFIG", args.wilor_config), (DEFAULT_WILOR_CONFIG,))
        crop, image_item, crop_item = build_wilor_crop(gpu1_root, gpu1_frames, wilor_root, wilor_config)
        metadata = {"ownership": ownership(args.run_id, "wilor.reconstruct", f"crop-{crop_item['index']}", image_item["source_file"], float(crop_item["index"]) / 30.0), "handedness": int(crop_item["handedness"]), "box_center": crop_item["box_center"], "box_size": crop_item["box_size"], "img_size": crop_item["img_size"], "source_K_px": None, "model_revision": args.wilor_revision or os.environ.get("EGO_STRESS_WILOR_REVISION") or router.endpoint_for(ModelApiName.WILOR_RECONSTRUCT).model_revision, "options": {}}
        operations["wilor"] = multipart_operation("wilor", endpoint(ModelApiName.WILOR_RECONSTRUCT), metadata, {"crop": (crop.tobytes(), crop.shape, "float32")}, {"manifest": str(gpu1_root / "wilor_crop_manifest.json"), "source_frame": image_item["source_file"], "crop_sha256": crop_item["sha256"]})
    if "droid" in selected:
        manifest = resolve_existing("DROID manifest", env_path("EGO_STRESS_DROID_MANIFEST", args.droid_manifest), (DEFAULT_DROID_MANIFEST,))
        base = router.base_url_for(ModelApiName.DROID_CREATE_SESSION)
        operations["droid"] = droid_operation(base, manifest, router)
    if "hawor" in selected:
        root = resolve_existing("HaWoR benchmark root", env_path("EGO_STRESS_HAWOR_ROOT", args.hawor_root), DEFAULT_HAWOR_ROOTS)
        summary = read_json(root / "summary.ego.v2.json")
        capture = Path(summary["fixture_provenance"]["request"]["path"])
        capture = resolve_existing("HaWoR captured request", env_path("EGO_STRESS_HAWOR_REQUEST", args.hawor_request), (capture,))
        operations["hawor"] = replay_operation("hawor", endpoint(ModelApiName.HAWOR_INFER_TRACKS), capture, {"benchmark_root": str(root), "source": "v2 summary fixture_provenance.request"})
    if "infiller" in selected:
        request = resolve_existing("HaWoR Infiller captured request", env_path("EGO_STRESS_INFILLER_REQUEST", args.infiller_request), (DEFAULT_INFILLER_REQUEST,))
        operations["infiller"] = replay_operation("infiller", endpoint(ModelApiName.HAWOR_INFILLER_FILL), request, {"source": "verified v22 captured multipart"})
    if "cosmos" in selected:
        request = resolve_existing("Cosmos representative request", env_path("EGO_STRESS_COSMOS_REQUEST", args.cosmos_request), (DEFAULT_COSMOS_REQUEST,))
        headers = resolve_existing("Cosmos request headers", env_path("EGO_STRESS_COSMOS_HEADERS", args.cosmos_headers), (DEFAULT_COSMOS_HEADERS,))
        expected_hash = read_json(headers).get("payload_sha256")
        observed_hash = sha256(request.read_bytes())
        # The installed runtime currently carries a stale declared hash beside a
        # usable request body.  Preserve both values in the artifact rather than
        # silently accepting the drift or refusing to exercise the real endpoint.
        operations["cosmos"] = replay_operation("cosmos", endpoint(ModelApiName.COSMOS3_REASON), request, {"headers": str(headers), "source": "runtime representative request", "declared_sha256": expected_hash, "observed_sha256": observed_hash, "header_hash_matches": expected_hash == observed_hash})
    return operations


async def run_once(operations: Mapping[str, PreparedOperation], client: httpx.AsyncClient, run_id: str) -> tuple[float, list[dict[str, Any]]]:
    """Start all selected service calls together and isolate every client-side failure."""
    async def guarded(operation: PreparedOperation) -> dict[str, Any]:
        started = time.monotonic()
        try:
            row = await operation.invoke(client, run_id)
        except Exception as exc:
            row = {"service": operation.service, "endpoint": operation.endpoint, "http_status": None, "latency_ms": (time.monotonic() - started) * 1000.0, "success": False, "typed_error": {"code": "client_preparation", "message": repr(exc), "retryable": False}, "result_hash": None, "batch_wait_ms": None}
        row.setdefault("gpu_mapping", GPU_MAPPING[operation.service])
        row.setdefault("payload", dict(operation.payload))
        return row

    started = time.monotonic()
    rows = await asyncio.gather(*(guarded(operation) for operation in operations.values()))
    return (time.monotonic() - started) * 1000.0, rows


def write_artifacts(artifact_dir: Path, args: argparse.Namespace, selected: Sequence[str], operations: Mapping[str, PreparedOperation], repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for repetition in repetitions for row in repetition["results"]]
    summary = {
        "schema": "ego.parallel-stress.v1", "run_id": args.run_id, "created_at": utc_now(), "artifact_dir": str(artifact_dir),
        "client_only": True, "services_redeployed": False, "services_restarted": False, "selected_services": list(selected), "gpu_mapping": GPU_MAPPING,
        "repetitions": repetitions, "all_success": all(row["success"] for row in rows), "service_success_counts": {service: sum(1 for row in rows if row["service"] == service and row["success"]) for service in selected},
        "payloads": {service: operation.payload for service, operation in operations.items()},
    }
    (artifact_dir / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["repetition", "service", "success", "latency_ms", "http_status", "result_hash", "endpoint", "batch_wait_ms", "gpu_mapping", "typed_error"]
    with (artifact_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for repetition in repetitions:
            for row in repetition["results"]:
                writer.writerow({"repetition": repetition["index"], **{key: json.dumps(row[key], sort_keys=True) if key == "typed_error" and row.get(key) is not None else row.get(key) for key in fields if key != "repetition"}})
    return summary


async def async_main(args: argparse.Namespace) -> int:
    selected = normalize_apis(args.apis)
    router = ModelServiceRouter.canonical()
    operations = prepare_operations(args, router, selected)
    artifact_dir = args.out / args.run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    timeout = httpx.Timeout(args.timeout_s)
    repetitions: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=64, max_keepalive_connections=32)) as client:
        for index in range(args.repetitions):
            wall_ms, rows = await run_once(operations, client, f"{args.run_id}-r{index + 1}")
            repetitions.append({"index": index + 1, "concurrent_wall_ms": wall_ms, "results": rows})
    summary = write_artifacts(artifact_dir, args, selected, operations, repetitions)
    print(f"artifact_dir={artifact_dir}")
    for row in repetitions[-1]["results"]:
        print(f"service={row['service']} success={row['success']} latency_ms={row['latency_ms']:.1f} endpoint={row['endpoint']} batch_wait_ms={row.get('batch_wait_ms')} gpu={row['gpu_mapping']} result_hash={row.get('result_hash')}")
    return 0 if summary["all_success"] else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--apis", help="comma-separated service/API filters; default: all seven")
    parser.add_argument("--out", type=Path, default=ARTIFACT_BASE, help="artifact base directory")
    parser.add_argument("--run-id", help="artifact directory name; defaults to parallel_stress_YYYYMMDDTHHMMSSZ")
    parser.add_argument("--unidepth-frame", type=Path)
    parser.add_argument("--gpu1-root", type=Path)
    parser.add_argument("--gpu1-frames", type=Path)
    parser.add_argument("--wilor-root", type=Path)
    parser.add_argument("--wilor-config", type=Path)
    parser.add_argument("--droid-manifest", type=Path)
    parser.add_argument("--hawor-root", type=Path)
    parser.add_argument("--hawor-request", type=Path)
    parser.add_argument("--infiller-request", type=Path)
    parser.add_argument("--cosmos-request", type=Path)
    parser.add_argument("--cosmos-headers", type=Path)
    parser.add_argument("--hands-revision", help="resident Hands revision; default matches the live A800 route")
    parser.add_argument("--wilor-revision", help="resident WiLoR revision override")
    parser.add_argument("--wilor-endpoint", help="WiLoR endpoint override; default is the live A800 port 28004")
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    args.run_id = args.run_id or datetime.now(timezone.utc).strftime("parallel_stress_%Y%m%dT%H%M%SZ")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main(parse_args())))
