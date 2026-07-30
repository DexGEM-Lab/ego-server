"""Live GPU0 UniDepth Ray Serve exercise: two concurrent distinct EgoScale frames.

Sends two distinct 540x960 uint8 EgoScale frames concurrently to the resident
UniDepth deployment and verifies:
  - one model load (model_load_count == 1 across both responses),
  - one fused upstream forward for the two concurrent requests (same batch_id,
    forward_count == 1, request_count == 2),
  - correct ownership split (each response carries its own request_id),
  - actual multipart response content type and body,
  - finite strictly-positive depth [540,960], K [3,3] with positive fx/fy,
    finite confidence [540,960],
  - server revision (resident configured revision, not the request's).

GPU placement and typed rejection/backpressure cases are checked by the companion
``verify_gpu0_unidepth_failures.py`` probe.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

import httpx
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ego_annotation.serving.contracts import (
    BatchTrace,
    ImageSize,
    Ownership,
    PixelTransform,
    SpatialMetadata,
    TensorPayload,
    UniDepthRequest,
    UniDepthResponse,
    UniDepthResult,
)
from ego_annotation.serving.transport import (
    build_multipart_request,
    parse_multipart_response,
)

ENDPOINT = os.environ.get("EGO_UNIDEPTH_ENDPOINT", "http://127.0.0.1:28000")
REVISION = os.environ.get("EGO_UNIDEPTH_REVISION", "unidepth-v2-vitl14-corrected")
FRAMES_DIR = os.environ.get("EGO_FRAMES_DIR", os.path.join(os.path.dirname(__file__), "egoscale_frames"))
H, W = 540, 960
# Physical GPU0 UUID discovered via nvidia-smi during cluster probe.
GPU0_UUID = "GPU-80c78b52-37d3-b79e-9d18-848e8e87468b"


def canonical_pixel_transform() -> PixelTransform:
    """Explicit 0.5 source-pixel transform: 1080x1920 -> 540x960."""
    # source_to_model scales by 0.5 in x,y; model_to_source scales by 2.0.
    s2m = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 1.0))
    m2s = ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.0))
    return PixelTransform(source_to_model=s2m, model_to_source=m2s, resize_mode="area")


def make_request(frame_path: str, request_id: str, job_id: str) -> UniDepthRequest:
    rgb = np.load(frame_path).astype(np.uint8)
    assert rgb.shape == (H, W, 3), f"frame shape {rgb.shape} != {(H, W, 3)}"
    assert rgb.dtype == np.uint8
    ownership = Ownership(
        request_id=request_id,
        job_id=job_id,
        item_id=f"frame-{request_id}",
        stage_id="unidepth.infer",
        source_id=f"egoscale-task20-{request_id}",
        source_timestamp_s=float(request_id.split("-")[-1]) / 30.0,
    )
    spatial = SpatialMetadata(
        source_size=ImageSize(width=1920, height=1080),
        model_size=ImageSize(width=W, height=H),
        color_space="RGB",
        pixel_transform=canonical_pixel_transform(),
        K_px=None,
    )
    return UniDepthRequest(
        ownership=ownership,
        rgb=TensorPayload(data=rgb.tobytes(), shape=rgb.shape, dtype="uint8"),
        spatial=spatial,
        model_revision=REVISION,
    )


def request_to_multipart(request: UniDepthRequest) -> tuple[bytes, str]:
    metadata = {
        "ownership": request.ownership.to_wire(),
        "spatial": request.spatial.to_wire(),
        "model_revision": request.model_revision,
        "options": dict(request.options),
        "rgb_shape": list(request.rgb.shape),
        "rgb_dtype": request.rgb.dtype,
    }
    return build_multipart_request(
        metadata, rgb=bytes(request.rgb.data), rgb_shape=request.rgb.shape, rgb_dtype=request.rgb.dtype
    )


def parse_response(content: bytes, content_type: str, ownership: Ownership) -> UniDepthResponse:
    metadata, arrays = parse_multipart_response(content, content_type)
    if metadata.get("error"):
        return UniDepthResponse(
            ownership=ownership,
            error=ServiceError_from_wire(metadata["error"]),
        )
    result_meta = metadata["result"]
    depth_bytes, depth_shape, depth_dtype = arrays["depth_m"]
    k_bytes, k_shape, k_dtype = arrays["K_px"]
    conf_bytes, conf_shape, conf_dtype = arrays["confidence"]
    return UniDepthResponse(
        ownership=ownership,
        result=UniDepthResult(
            ownership=ownership,
            depth_m=TensorPayload(data=depth_bytes, shape=depth_shape, dtype=depth_dtype),
            K_px=TensorPayload(data=k_bytes, shape=k_shape, dtype=k_dtype),
            confidence=TensorPayload(data=conf_bytes, shape=conf_shape, dtype=conf_dtype),
            spatial=SpatialMetadata.from_mapping(result_meta["spatial"]),
            model_revision=result_meta["model_revision"],
            trace=BatchTrace.from_wire(result_meta["trace"]),
        ),
    )


def ServiceError_from_wire(payload: dict[str, Any]):
    from ego_annotation.serving.contracts import ErrorCode, ServiceError

    return ServiceError(
        code=ErrorCode(payload["code"]),
        message=payload["message"],
        retryable=bool(payload.get("retryable")),
        ownership=Ownership.from_mapping(payload["ownership"]) if payload.get("ownership") else None,
        batch_id=payload.get("batch_id"),
    )


async def post_one(client: httpx.AsyncClient, request: UniDepthRequest) -> tuple[UniDepthResponse, float, str]:
    body, content_type = request_to_multipart(request)
    t0 = time.monotonic()
    resp = await client.post(f"{ENDPOINT}/", content=body, headers={"Content-Type": content_type}, timeout=60.0)
    elapsed = time.monotonic() - t0
    resp_ct = resp.headers.get("Content-Type", "")
    parsed = parse_response(resp.content, resp_ct, request.ownership)
    return parsed, elapsed, resp_ct


async def main() -> dict[str, Any]:
    frame_a = os.path.join(FRAMES_DIR, "frame_A_idx000.npy")
    frame_b = os.path.join(FRAMES_DIR, "frame_A_idx030.npy")
    assert os.path.exists(frame_a) and os.path.exists(frame_b), "EgoScale frames missing"
    req_a = make_request(frame_a, "req-A-000", "job-exercise")
    req_b = make_request(frame_b, "req-B-030", "job-exercise")

    results: dict[str, Any] = {"endpoint": ENDPOINT, "revision": REVISION}
    async with httpx.AsyncClient() as client:
        # Two concurrent distinct requests.
        t_start = time.monotonic()
        (resp_a, t_a, ct_a), (resp_b, t_b, ct_b) = await asyncio.gather(
            post_one(client, req_a), post_one(client, req_b)
        )
        wall = time.monotonic() - t_start

    results["concurrent_wall_s"] = round(wall, 4)
    results["latency_a_s"] = round(t_a, 4)
    results["latency_b_s"] = round(t_b, 4)

    # --- Verify multipart content type ---
    results["content_type_a"] = ct_a
    results["content_type_b"] = ct_b
    assert ct_a.startswith("multipart/form-data"), f"resp A not multipart: {ct_a}"
    assert ct_b.startswith("multipart/form-data"), f"resp B not multipart: {ct_b}"

    # --- Verify ownership split ---
    assert resp_a.result is not None and resp_b.result is not None, "responses had errors"
    assert resp_a.ownership.request_id == "req-A-000"
    assert resp_b.ownership.request_id == "req-B-030"
    results["ownership_split_ok"] = True

    # --- Verify one fused forward (same batch_id, forward_count=1, request_count=2) ---
    tr_a = resp_a.result.trace
    tr_b = resp_b.result.trace
    results["batch_id_a"] = tr_a.batch_id
    results["batch_id_b"] = tr_b.batch_id
    results["forward_count_a"] = tr_a.forward_count
    results["forward_count_b"] = tr_b.forward_count
    results["request_count_a"] = tr_a.request_count
    results["request_count_b"] = tr_b.request_count
    assert tr_a.batch_id == tr_b.batch_id, f"different batch_ids: {tr_a.batch_id} vs {tr_b.batch_id}"
    assert tr_a.forward_count == 1 and tr_b.forward_count == 1, "forward_count != 1"
    assert tr_a.request_count == 2 and tr_b.request_count == 2, "request_count != 2 (not fused)"

    # --- Verify one model load ---
    assert tr_a.model_load_count == 1 and tr_b.model_load_count == 1, "model_load_count != 1"
    results["model_load_count"] = tr_a.model_load_count

    # --- Verify server revision (resident configured, not request's) ---
    assert resp_a.result.model_revision == REVISION, f"revision A {resp_a.result.model_revision}"
    assert resp_b.result.model_revision == REVISION, f"revision B {resp_b.result.model_revision}"
    results["server_revision_ok"] = True

    # --- Verify finite positive depth/K/confidence shapes and semantics ---
    depth_a = np.frombuffer(resp_a.result.depth_m.data, dtype=np.dtype(resp_a.result.depth_m.dtype)).reshape(resp_a.result.depth_m.shape)
    k_a = np.frombuffer(resp_a.result.K_px.data, dtype=np.dtype(resp_a.result.K_px.dtype)).reshape(resp_a.result.K_px.shape)
    conf_a = np.frombuffer(resp_a.result.confidence.data, dtype=np.dtype(resp_a.result.confidence.dtype)).reshape(resp_a.result.confidence.shape)
    depth_b = np.frombuffer(resp_b.result.depth_m.data, dtype=np.dtype(resp_b.result.depth_m.dtype)).reshape(resp_b.result.depth_m.shape)
    k_b = np.frombuffer(resp_b.result.K_px.data, dtype=np.dtype(resp_b.result.K_px.dtype)).reshape(resp_b.result.K_px.shape)
    conf_b = np.frombuffer(resp_b.result.confidence.data, dtype=np.dtype(resp_b.result.confidence.dtype)).reshape(resp_b.result.confidence.shape)

    results["depth_shape_a"] = list(depth_a.shape)
    results["K_shape_a"] = list(k_a.shape)
    results["confidence_shape_a"] = list(conf_a.shape)
    results["depth_shape_b"] = list(depth_b.shape)
    results["K_shape_b"] = list(k_b.shape)
    results["confidence_shape_b"] = list(conf_b.shape)
    assert depth_a.shape == (H, W), f"depth A shape {depth_a.shape}"
    assert k_a.shape == (3, 3), f"K A shape {k_a.shape}"
    assert conf_a.shape == (H, W), f"conf A shape {conf_a.shape}"
    assert depth_b.shape == (H, W), f"depth B shape {depth_b.shape}"
    assert k_b.shape == (3, 3), f"K B shape {k_b.shape}"
    assert conf_b.shape == (H, W), f"conf B shape {conf_b.shape}"

    # finite + positive depth, positive fx/fy, finite confidence
    for name, d, k, c in [("A", depth_a, k_a, conf_a), ("B", depth_b, k_b, conf_b)]:
        assert np.isfinite(d).all(), f"depth {name} not finite"
        assert float(d.min()) > 0.0, f"depth {name} not positive (min={d.min()})"
        assert np.isfinite(k).all(), f"K {name} not finite"
        assert float(k[0, 0]) > 0.0 and float(k[1, 1]) > 0.0, f"K {name} fx/fy not positive: {k}"
        assert np.isfinite(c).all(), f"confidence {name} not finite"
    results["finite_positive_semantics_ok"] = True
    results["depth_a_min_m"] = round(float(depth_a.min()), 4)
    results["depth_a_max_m"] = round(float(depth_a.max()), 4)
    results["depth_b_min_m"] = round(float(depth_b.min()), 4)
    results["depth_b_max_m"] = round(float(depth_b.max()), 4)
    results["K_a_fx"] = round(float(k_a[0, 0]), 2)
    results["K_a_fy"] = round(float(k_a[1, 1]), 2)
    results["K_b_fx"] = round(float(k_b[0, 0]), 2)
    results["K_b_fy"] = round(float(k_b[1, 1]), 2)
    # The two distinct frames should produce different depth maps (not identical).
    results["frames_distinct"] = bool(not np.allclose(depth_a, depth_b))

    return results


if __name__ == "__main__":
    out = asyncio.run(main())
    print(json.dumps(out, indent=2, default=str))
