"""Verify physical GPU0 UUID and typed overweight/revision/backpressure failures
against the live GPU0 UniDepth Ray Serve deployment.

Does NOT request num_gpus inside the GPU0 cluster (the only GPU is held by the
resident replica, so a num_gpus=1 task would pend forever). It verifies placement
from the replica's CUDA binding, GPU0 UUID, and explicit-GCS native Ray resources.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ego_annotation.serving.contracts import (  # noqa: E402
    ErrorCode,
    ImageSize,
    Ownership,
    PixelTransform,
    SpatialMetadata,
    TensorPayload,
    UniDepthRequest,
)
from ego_annotation.serving.transport import (  # noqa: E402
    build_multipart_request,
    parse_multipart_response,
)

ENDPOINT = os.environ.get("EGO_UNIDEPTH_ENDPOINT", "http://127.0.0.1:28000")
REVISION = os.environ.get("EGO_UNIDEPTH_REVISION", "unidepth-v2-vitl14-corrected")
FRAMES_DIR = os.environ.get("EGO_FRAMES_DIR", os.path.join(os.path.dirname(__file__), "egoscale_frames"))
H, W = 540, 960
GPU0_UUID_EXPECTED = "GPU-80c78b52-37d3-b79e-9d18-848e8e87468b"


def _pixel_transform() -> PixelTransform:
    s2m = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 1.0))
    m2s = ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.0))
    return PixelTransform(source_to_model=s2m, model_to_source=m2s, resize_mode="area")


def _build_request(rgb: np.ndarray, *, request_id: str, job_id: str, model_h: int,
                   model_w: int, revision: str) -> UniDepthRequest:
    ownership = Ownership(
        request_id=request_id, job_id=job_id, item_id=f"item-{request_id}",
        stage_id="unidepth.infer", source_id=f"src-{request_id}",
    )
    spatial = SpatialMetadata(
        source_size=ImageSize(width=model_w * 2, height=model_h * 2),
        model_size=ImageSize(width=model_w, height=model_h),
        color_space="RGB", pixel_transform=_pixel_transform(), K_px=None,
    )
    return UniDepthRequest(
        ownership=ownership,
        rgb=TensorPayload(data=rgb.tobytes(), shape=rgb.shape, dtype="uint8"),
        spatial=spatial, model_revision=revision,
    )


def _to_multipart(request: UniDepthRequest) -> tuple[bytes, str]:
    metadata = {
        "ownership": request.ownership.to_wire(),
        "spatial": request.spatial.to_wire(),
        "model_revision": request.model_revision,
        "options": dict(request.options),
        "rgb_shape": list(request.rgb.shape),
        "rgb_dtype": request.rgb.dtype,
    }
    return build_multipart_request(metadata, rgb=bytes(request.rgb.data),
                                   rgb_shape=request.rgb.shape, rgb_dtype=request.rgb.dtype)


def _parse(content: bytes, content_type: str) -> dict:
    metadata, arrays = parse_multipart_response(content, content_type)
    return {"metadata": metadata, "arrays": arrays, "content_type": content_type}


def gpu0_identity() -> dict:
    """Verify the replica's GPU0 binding without joining namespace-mismatched PIDs.

    ``nvidia-smi --query-compute-apps`` reports host-PID-namespace PIDs here, while
    the Ray worker is observed from the container namespace.  Comparing those PIDs
    falsely reports that a real replica is absent.  The direct evidence is instead:
    the replica's inherited ``CUDA_VISIBLE_DEVICES=0`` and its open ``/dev/nvidia0``
    descriptors, plus the GPU0 UUID and Ray's one-held-native-GPU status.
    """
    ps = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True, check=True)
    replica_pids = []
    for line in ps.stdout.splitlines():
        if "ServeReplica" in line and "unidepth.infer" in line:
            try:
                replica_pids.append(int(line.split()[0]))
            except ValueError:
                pass

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    uuid_by_index = {
        int(line.split(",")[0].strip()): line.split(",")[1].strip()
        for line in smi.stdout.strip().splitlines()
    }
    replica_evidence = []
    for pid in replica_pids:
        try:
            environ = Path(f"/proc/{pid}/environ").read_bytes().decode("utf-8", errors="replace")
            cuda_visible = next(
                (entry.split("=", 1)[1] for entry in environ.split("\0") if entry.startswith("CUDA_VISIBLE_DEVICES=")),
                None,
            )
            device_targets = sorted({
                os.readlink(fd)
                for fd in Path(f"/proc/{pid}/fd").iterdir()
                if os.path.islink(fd) and os.readlink(fd).startswith("/dev/nvidia")
            })
            replica_evidence.append({
                "pid": pid,
                "cuda_visible_devices": cuda_visible,
                "nvidia_device_fds": device_targets,
                "bound_to_gpu0": cuda_visible == "0" and "/dev/nvidia0" in device_targets,
            })
        except (FileNotFoundError, ProcessLookupError, PermissionError) as exc:
            replica_evidence.append({"pid": pid, "inspection_error": str(exc), "bound_to_gpu0": False})

    import ray
    ray.init(address=os.environ.get("EGO_UNIDEPTH_GCS_ADDRESS", "192.168.42.193:26000"), namespace="serve", ignore_reinit_error=True, logging_level="ERROR")
    try:
        cluster_gpu = float(ray.cluster_resources().get("GPU", 0.0))
        available_gpu = float(ray.available_resources().get("GPU", 0.0))
    finally:
        ray.shutdown()
    return {
        "replica_pids": replica_pids,
        "gpu0_uuid": uuid_by_index.get(0),
        "gpu0_uuid_matches_expected": uuid_by_index.get(0) == GPU0_UUID_EXPECTED,
        "replica_binding_evidence": replica_evidence,
        "replica_on_gpu0": bool(replica_evidence) and all(item.get("bound_to_gpu0") for item in replica_evidence),
        "cluster_total_gpu": cluster_gpu,
        "cluster_available_gpu": available_gpu,
        "native_gpu_owned_by_replica": cluster_gpu == 1.0 and available_gpu == 0.0,
        "all_gpu_uuids": uuid_by_index,
    }


async def revision_mismatch_probe(client: httpx.AsyncClient) -> dict:
    """Wrong model_revision -> typed VALIDATION error, no forward."""
    rgb = np.load(os.path.join(FRAMES_DIR, "frame_A_idx000.npy")).astype(np.uint8)
    req = _build_request(rgb, request_id="req-bad-rev", job_id="job-rev",
                         model_h=H, model_w=W, revision="some-wrong-revision")
    body, ct = _to_multipart(req)
    resp = await client.post(f"{ENDPOINT}/", content=body, headers={"Content-Type": ct}, timeout=30.0)
    parsed = _parse(resp.content, resp.headers.get("Content-Type", ""))
    err = parsed["metadata"].get("error")
    return {
        "http_status": resp.status_code,
        "content_type": resp.headers.get("Content-Type", ""),
        "has_error": bool(err),
        "error_code": err.get("code") if err else None,
        "error_message": err.get("message") if err else None,
        "typed_validation": bool(err and err.get("code") == ErrorCode.VALIDATION.value),
    }


async def overweight_probe(client: httpx.AsyncClient) -> dict:
    """An overweight request: a 1080x1920 image (4x canonical pixels) declared with a
    matching model_size so contract construction passes, then rejected at admission as
    incompatible with the canonical 540x960 bucket -> typed VALIDATION error."""
    rgb = np.load(os.path.join(FRAMES_DIR, "frame_A_idx000.npy")).astype(np.uint8)
    # Upscale to 1080x1920 (4x pixels, overweight relative to canonical bucket).
    big = np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1)
    assert big.shape == (1080, 1920, 3)
    req = _build_request(big, request_id="req-overweight", job_id="job-ow",
                         model_h=1080, model_w=1920, revision=REVISION)
    body, ct = _to_multipart(req)
    resp = await client.post(f"{ENDPOINT}/", content=body, headers={"Content-Type": ct}, timeout=30.0)
    parsed = _parse(resp.content, resp.headers.get("Content-Type", ""))
    err = parsed["metadata"].get("error")
    return {
        "http_status": resp.status_code,
        "content_type": resp.headers.get("Content-Type", ""),
        "payload_shape": list(big.shape),
        "has_error": bool(err),
        "error_code": err.get("code") if err else None,
        "error_message": err.get("message") if err else None,
        "typed_rejection": bool(err and err.get("code") == ErrorCode.VALIDATION.value),
    }


async def backpressure_probe(client: httpx.AsyncClient) -> dict:
    """Bounded burst exceeding max_queued_requests (64) + max_ongoing_requests (16) = 80
    admission slots. Requests beyond 80 must be rejected with a typed backpressure
    signal: HTTP 429/503 at the proxy, or a BACKPRESSURE error in the multipart body.

    Uses short per-request timeouts and return_exceptions so the probe cannot hang.
    """
    rgb = np.load(os.path.join(FRAMES_DIR, "frame_A_idx000.npy")).astype(np.uint8)
    burst = 96  # > 80 admission slots
    tasks = []
    for i in range(burst):
        req = _build_request(rgb, request_id=f"req-bp-{i:03d}", job_id="job-bp",
                             model_h=H, model_w=W, revision=REVISION)
        body, ct = _to_multipart(req)
        tasks.append(client.post(f"{ENDPOINT}/", content=body,
                                 headers={"Content-Type": ct}, timeout=20.0))
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    ok = 0
    backpressure = 0
    other_errors = 0
    statuses = []
    bp_error_codes = []
    for r in responses:
        if isinstance(r, Exception):
            other_errors += 1
            continue
        statuses.append(r.status_code)
        if r.status_code in (429, 503):
            backpressure += 1
            try:
                parsed = _parse(r.content, r.headers.get("Content-Type", ""))
                err = parsed["metadata"].get("error")
                if err:
                    bp_error_codes.append(err.get("code"))
            except Exception:
                pass
        elif r.status_code == 200:
            # A 200 could still carry a BACKPRESSURE error body (item-scoped rejection).
            try:
                parsed = _parse(r.content, r.headers.get("Content-Type", ""))
                err = parsed["metadata"].get("error")
                if err and err.get("code") == ErrorCode.BACKPRESSURE.value:
                    backpressure += 1
                    bp_error_codes.append(err.get("code"))
                else:
                    ok += 1
            except Exception:
                ok += 1
        else:
            other_errors += 1
    return {
        "burst_size": burst,
        "ok_200": ok,
        "backpressure": backpressure,
        "other_errors": other_errors,
        "distinct_statuses": sorted(set(statuses)),
        "bp_error_codes": sorted(set(bp_error_codes)),
        "typed_backpressure": backpressure > 0,
    }


async def main() -> dict:
    results: dict = {"endpoint": ENDPOINT, "revision": REVISION}
    results["gpu0_identity"] = gpu0_identity()
    async with httpx.AsyncClient() as client:
        results["revision_mismatch"] = await revision_mismatch_probe(client)
        results["overweight"] = await overweight_probe(client)
        results["backpressure"] = await backpressure_probe(client)
    return results


if __name__ == "__main__":
    out = asyncio.run(main())
    print(json.dumps(out, indent=2, default=str))
