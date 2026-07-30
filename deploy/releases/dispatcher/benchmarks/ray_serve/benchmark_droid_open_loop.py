#!/usr/bin/env python3
"""Open-loop HTTP benchmark for the live stateful DROID endpoint.

The driver caches a corpus of *real RGB frames plus real model masks* before it
opens any DROID session.  Push waves are released by a precomputed monotonic
schedule, not by the completion of a prior wave.  A session can therefore reject
a scheduled frame while its previous frame remains in flight; that rejection is
measured as the intended state/backpressure boundary, never retried or hidden.

This program is deliberately an external client: it connects to the already-live
Serve HTTP endpoint and the existing Ray GCS, and never starts, stops, or changes
Ray/Serve.  Run it from the GPU2 serving Python environment on dex-a800.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from ego_annotation.serving.benchmark.droid_scaling import (
    AUTHORIZED_EXPERIMENT_GPU_IDS, PRODUCTION_PORTS, validate_droid_server_identity,
)
from ego_annotation.serving.benchmark.measurement import NvmlSampler, validate_gpu_samples
from ego_annotation.serving.contracts import (
    ContractValidationError,
    DroidCamera,
    DroidCreateSessionRequest,
    DroidCreateSessionResponse,
    DroidFinalizeRequest,
    DroidFinalizeResponse,
    DroidFrameRequest,
    DroidFrameResponse,
    DroidImageShape,
    DroidSessionOptions,
    ErrorCode,
    Ownership,
    ServerIdentity,
    TensorPayload,
)
from ego_annotation.serving.transport import build_multipart_request_fields, parse_droid_finalize_response

DEFAULT_VIDEO = (
    "/vePFS-Mindverse/user/yiwen/user-home/zjh/data/v22_api_uploads/"
    "annotation_7987b8625386/20251121_1019_Recdf53_P1_S237208_task_3.mp4"
)
# This tolerance is exclusively for scheduler/clock timestamp skew. It never
# relaxes exact equal work assignment across compared replicas.
OFFER_TIMESTAMP_JITTER_TOLERANCE_S = 0.050

DEFAULT_HAND_MASKS = (
    "/vePFS-Mindverse/user/yiwen/user-home/zjh/data/v22_wave_batch_runs/"
    "v22_wave32_preflight_fix_20260710T015402Z/entries/"
    "egoscale30h_stage_batch_egoscale_tasks_20251121_1019_Recdf53_P1_S237208_task_3_"
    "20251121_1019_Recdf53_P1_S237208_task_3/work/hawor_resident/"
    "egoscale30h_stage_batch_egoscale_tasks_20251121_1019_Recdf53_P1_S237208_task_3_"
    "20251121_1019_Recdf53_P1_S237208_task_3/tracks_0_720/model_masks.npy"
)


@dataclass(frozen=True)
class CachedPayload:
    payload_id: str
    source_frame_index: int
    timestamp_s: float
    rgb_sha256: str
    mask_sha256: str
    rgb_path: str
    mask_path: str


@dataclass
class RequestRecord:
    operation: str
    level: str
    request_id: str
    session_id: str | None
    scheduled_s: float | None
    sent_s: float
    completed_s: float
    latency_s: float
    http_status: int
    outcome: str
    error_code: str | None
    error_message: str | None
    trace: dict[str, Any] | None
    diagnostics: dict[str, Any] | None = None
    # Terminal CameraState QC emitted by the worker. Retain it per request so a
    # valid finalize cannot be reduced to a generic HTTP-200 count in summary.
    finite_pose_ratio: float | None = None
    endpoint: str | None = None
    replica_id: str | None = None
    semantic_valid: bool = False
    treatment_role: str = "two_replica_scaling"
    planned_session_assignment: dict[str, int] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def rate(count: int, duration_s: float) -> float:
    return count / duration_s if duration_s > 0 else 0.0


def multipart_body(metadata: dict[str, Any], arrays: dict[str, tuple[bytes, tuple[int, ...], str]]) -> tuple[bytes, str]:
    """Compatibility wrapper around the repository's canonical typed transport."""
    return build_multipart_request_fields(metadata, arrays)


def ownership(request_id: str, stage_id: str, source_id: str, timestamp_s: float | None = None) -> Ownership:
    return Ownership(
        request_id=request_id,
        job_id=f"droid-open-loop-{request_id.rsplit('-', 1)[0]}",
        item_id=source_id,
        stage_id=stage_id,
        source_id=source_id,
        source_timestamp_s=timestamp_s,
    )


# Pinhole prior for the default task_3 video, from the V19 calibration contract
# (focal_scale 0.8756290246 × max(1920,1080) = 1681.2078 at 1920×1080), the same
# calibration the V22 DROID trajectory stage used (measurements/camera_trajectory/
# droid_full_frame/droid_qc.json). The corpus frames are downscaled to 568×320 by
# cache_payloads, so the model-grid intrinsics are scaled by (568/1920, 320/1080).
_SOURCE_FOCAL_1920 = 1681.2077272759332
_SOURCE_SIZE_1920 = (1920.0, 1080.0)
_MODEL_SIZE_568 = (568.0, 320.0)


def camera_contract() -> dict[str, Any]:
    """Model-grid camera for the downscaled corpus, with the real transform recorded.

    The focal is the V19-contract focal rescaled to 568×320; it must never be a
    hardcoded constant unrelated to the video calibration (the prior 408.96 constant
    was 0.823× the correct focal and diverged BA on every multi-frame session).
    """
    sx = _MODEL_SIZE_568[0] / _SOURCE_SIZE_1920[0]
    sy = _MODEL_SIZE_568[1] / _SOURCE_SIZE_1920[1]
    fx = _SOURCE_FOCAL_1920 * sx
    fy = _SOURCE_FOCAL_1920 * sy
    cx = (_SOURCE_SIZE_1920[0] / 2.0) * sx
    cy = (_SOURCE_SIZE_1920[1] / 2.0) * sy
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return {
        "intrinsics": [fx, fy, cx, cy],
        "K_px": None,
        "source_size": {"width": int(_SOURCE_SIZE_1920[0]), "height": int(_SOURCE_SIZE_1920[1])},
        "model_size": {"width": int(_MODEL_SIZE_568[0]), "height": int(_MODEL_SIZE_568[1])},
        "pixel_transform": {
            "source_to_model": [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
            "model_to_source": [[1.0 / sx, 0.0, 0.0], [0.0, 1.0 / sy, 0.0], [0.0, 0.0, 1.0]],
            "resize_mode": "downscale_area",
        },
        "calibration_provenance": "v19_camera_calibration_contract task_3 focal_scale*max(W,H)",
    }


@dataclass(frozen=True)
class ReplicaEndpoint:
    base_url: str
    expected_replica_id: str
    expected_model_revision: str
    runtime_identity: ServerIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        if not self.base_url or not self.expected_replica_id or not self.expected_model_revision:
            raise ValueError("DROID endpoint URL, replica id, and model revision are required")
        if self.runtime_identity.replica_id != self.expected_replica_id:
            raise ValueError("DROID endpoint replica id differs from verified runtime identity")
        if self.runtime_identity.model_revision != self.expected_model_revision:
            raise ValueError("DROID endpoint revision differs from verified runtime identity")
        if self.runtime_identity.release_digest is None or self.runtime_identity.cuda_uuid is None or self.runtime_identity.module_root is None:
            raise ValueError("DROID benchmark endpoint requires content/GPU/module-root runtime identity")
        if not self.runtime_identity.dependency_digest or not self.runtime_identity.dependency_root or not self.runtime_identity.source_amendment_id:
            raise ValueError("DROID benchmark endpoint requires verified non-empty dependency source identity")
        port = urlsplit(self.base_url).port
        if port is None or port != self.runtime_identity.http_port or port in PRODUCTION_PORTS:
            raise ValueError("DROID endpoint port differs from verified experimental runtime identity")
        if self.runtime_identity.assigned_gpu not in AUTHORIZED_EXPERIMENT_GPU_IDS:
            raise ValueError("DROID benchmark runtime identity names an unauthorized physical GPU")


@dataclass
class TypedCall:
    operation: str
    endpoint: ReplicaEndpoint
    sent_s: float
    completed_s: float
    http_status: int
    response: DroidCreateSessionResponse | DroidFrameResponse | DroidFinalizeResponse | None = None
    parse_error: str | None = None

    @property
    def latency_s(self) -> float:
        return self.completed_s - self.sent_s


def _response_content_type(response: Any) -> str:
    headers = getattr(response, "headers", {})
    return str(headers.get("content-type", headers.get("Content-Type", "")))


def _validate_typed_response(
    operation: str,
    request: DroidCreateSessionRequest | DroidFrameRequest | DroidFinalizeRequest,
    response: DroidCreateSessionResponse | DroidFrameResponse | DroidFinalizeResponse,
    endpoint: ReplicaEndpoint,
) -> None:
    if response.ownership != request.ownership:
        raise ContractValidationError("response ownership does not match request ownership")
    try:
        validate_droid_server_identity(endpoint.runtime_identity, response)
    except ValueError as exc:
        raise ContractValidationError(str(exc)) from exc
    if operation == "create_session":
        if not isinstance(response, DroidCreateSessionResponse):
            raise ContractValidationError("create returned the wrong response type")
        return
    if operation == "push_frame":
        if not isinstance(request, DroidFrameRequest) or not isinstance(response, DroidFrameResponse):
            raise ContractValidationError("push returned the wrong response type")
        if response.status is not None:
            status = response.status
            if status.session_id != request.session_id or status.frame_id != request.frame_id:
                raise ContractValidationError("push response session/frame identity mismatch")
            if status.source_timestamp_s != request.source_timestamp_s:
                raise ContractValidationError("push response source timestamp mismatch")
            if status.trace.replica_id != endpoint.expected_replica_id:
                raise ContractValidationError("push response came from the wrong sticky replica")
        return
    if not isinstance(request, DroidFinalizeRequest) or not isinstance(response, DroidFinalizeResponse):
        raise ContractValidationError("finalize returned the wrong response type")
    if response.camera_state is not None:
        state = response.camera_state
        if state.session_id != request.session_id:
            raise ContractValidationError("finalize response session identity mismatch")
        if state.model_revision != endpoint.expected_model_revision:
            raise ContractValidationError("finalize response model revision mismatch")
        if state.trace.replica_id != endpoint.expected_replica_id:
            raise ContractValidationError("finalize response came from the wrong sticky replica")


async def _post_typed(
    client: Any,
    endpoint: ReplicaEndpoint,
    operation: str,
    request: DroidCreateSessionRequest | DroidFrameRequest | DroidFinalizeRequest,
) -> TypedCall:
    arrays: dict[str, tuple[bytes, tuple[int, ...], str]] = {}
    if isinstance(request, DroidFrameRequest):
        metadata = {
            "ownership": request.ownership.to_wire(),
            "session_id": request.session_id,
            "frame_id": request.frame_id,
            "source_timestamp_s": request.source_timestamp_s,
            "model_revision": request.model_revision,
        }
        for name, tensor in (
            ("rgb", request.rgb),
            ("static_confidence_mask", request.static_confidence_mask),
            ("depth_m", request.depth_m),
        ):
            if tensor is not None:
                if not isinstance(tensor.data, (bytes, bytearray, memoryview)):
                    raise ContractValidationError(f"benchmark {name} tensor must be materialized bytes")
                arrays[name] = (bytes(tensor.data), tensor.shape, tensor.dtype)
    else:
        metadata = request.to_wire()
    body, content_type = build_multipart_request_fields(metadata, arrays)
    path = {
        "create_session": "droid.create_session",
        "push_frame": "droid.push_frame",
        "finalize": "droid.finalize",
    }[operation]
    sent_s = time.monotonic()
    try:
        raw = await client.post(
            f"{endpoint.base_url}/{path}", content=body, headers={"Content-Type": content_type},
        )
        completed_s = time.monotonic()
    except Exception as exc:
        return TypedCall(operation, endpoint, sent_s, time.monotonic(), 0, parse_error=f"transport: {exc!r}")
    try:
        if operation == "finalize" and "multipart/form-data" in _response_content_type(raw).lower():
            parsed = parse_droid_finalize_response(bytes(raw.content), _response_content_type(raw))
        else:
            wire = raw.json()
            if not isinstance(wire, Mapping):
                raise ContractValidationError("response JSON must be an object")
            response_type = {
                "create_session": DroidCreateSessionResponse,
                "push_frame": DroidFrameResponse,
                "finalize": DroidFinalizeResponse,
            }[operation]
            parsed = response_type.from_wire(wire)
            if operation == "finalize" and isinstance(parsed, DroidFinalizeResponse) and parsed.camera_state is not None:
                raise ContractValidationError("successful finalize response must use multipart binary transport")
        _validate_typed_response(operation, request, parsed, endpoint)
        return TypedCall(operation, endpoint, sent_s, completed_s, int(raw.status_code), response=parsed)
    except Exception as exc:
        return TypedCall(
            operation, endpoint, sent_s, completed_s, int(raw.status_code),
            parse_error=f"semantic_parse_failure: {exc}",
        )


def _terminal_response(operation: str, response: DroidFrameResponse | DroidFinalizeResponse) -> bool:
    if isinstance(response, DroidFinalizeResponse):
        # Error codes explain a failure; only the actor can attest session retirement.
        return response.terminal
    # Push errors never prove the stateful session has been retired.
    return False


class StickyDroidRouter:
    """Round-robin creation, active affinity, and bounded terminal retry routes."""

    def __init__(self, endpoints: Iterable[ReplicaEndpoint], *, max_terminal_routes: int = 128) -> None:
        self.endpoints = tuple(endpoints)
        if not self.endpoints:
            raise ValueError("at least one DROID endpoint is required")
        if max_terminal_routes <= 0:
            raise ValueError("max_terminal_routes must be positive")
        self._next_endpoint = 0
        self._session_endpoints: dict[str, ReplicaEndpoint] = {}
        # The server retains a bounded terminal result journal. Preserve the matching
        # route for retries, then evict it LRU-style instead of leaking active affinity.
        self._terminal_endpoints: OrderedDict[str, ReplicaEndpoint] = OrderedDict()
        self._max_terminal_routes = max_terminal_routes
        self._lock = asyncio.Lock()

    async def create_session(self, client: Any, request: DroidCreateSessionRequest) -> TypedCall:
        async with self._lock:
            endpoint = self.endpoints[self._next_endpoint % len(self.endpoints)]
            self._next_endpoint += 1
        call = await _post_typed(client, endpoint, "create_session", request)
        if isinstance(call.response, DroidCreateSessionResponse) and call.response.session_id is not None:
            async with self._lock:
                session_id = call.response.session_id
                if session_id in self._session_endpoints or session_id in self._terminal_endpoints:
                    call.response = None
                    call.parse_error = "semantic_parse_failure: duplicate session_id returned by replicas"
                else:
                    self._session_endpoints[session_id] = endpoint
        return call

    async def _endpoint_for(self, session_id: str) -> ReplicaEndpoint | None:
        async with self._lock:
            endpoint = self._session_endpoints.get(session_id)
            if endpoint is not None:
                return endpoint
            endpoint = self._terminal_endpoints.get(session_id)
            if endpoint is not None:
                self._terminal_endpoints.move_to_end(session_id)
            return endpoint

    async def _retire_to_terminal_route(self, session_id: str, endpoint: ReplicaEndpoint) -> None:
        async with self._lock:
            self._session_endpoints.pop(session_id, None)
            self._terminal_endpoints[session_id] = endpoint
            self._terminal_endpoints.move_to_end(session_id)
            while len(self._terminal_endpoints) > self._max_terminal_routes:
                self._terminal_endpoints.popitem(last=False)

    async def push_frame(self, client: Any, request: DroidFrameRequest) -> TypedCall:
        endpoint = await self._endpoint_for(request.session_id)
        if endpoint is None:
            now = time.monotonic()
            return TypedCall(
                "push_frame", self.endpoints[0], now, now, 0,
                parse_error=f"routing_conflict: unknown sticky session {request.session_id!r}",
            )
        call = await _post_typed(client, endpoint, "push_frame", request)
        return call

    async def finalize(self, client: Any, request: DroidFinalizeRequest) -> TypedCall:
        endpoint = await self._endpoint_for(request.session_id)
        if endpoint is None:
            now = time.monotonic()
            return TypedCall(
                "finalize", self.endpoints[0], now, now, 0,
                parse_error=f"routing_conflict: unknown sticky session {request.session_id!r}",
            )
        call = await _post_typed(client, endpoint, "finalize", request)
        if isinstance(call.response, DroidFinalizeResponse) and _terminal_response("finalize", call.response):
            await self._retire_to_terminal_route(request.session_id, endpoint)
        # Transport/parse failures retain the active or terminal route. A retry never
        # migrates mutable DROID state or loses a server-side idempotency journal.
        return call

    async def endpoint_for_session(self, session_id: str) -> ReplicaEndpoint | None:
        return await self._endpoint_for(session_id)

    async def active_session_count(self) -> int:
        async with self._lock:
            return len(self._session_endpoints)

    async def terminal_route_count(self) -> int:
        async with self._lock:
            return len(self._terminal_endpoints)


def capture_gcs(gcs_address: str) -> dict[str, Any]:
    """Connect explicitly to the live GCS; shutdown disconnects only this client."""
    result: dict[str, Any] = {"address": gcs_address}
    try:
        import ray

        ray.init(address=gcs_address, ignore_reinit_error=True, logging_level="ERROR")
        result.update({
            "connected": True,
            "available_resources": ray.available_resources(),
            "cluster_resources": ray.cluster_resources(),
            "live_nodes": sum(1 for node in ray.nodes() if node.get("Alive")),
        })
        ray.shutdown()
    except Exception as exc:  # Preserve evidence if Ray status itself is unavailable.
        result.update({"connected": False, "error": repr(exc)})
    return result


def _write_payload_manifest(
    run_root: Path, payloads: list[CachedPayload], *, source: dict[str, Any],
) -> list[CachedPayload]:
    manifest = {
        "created_at": utc_now(),
        "source": source,
        "model_size_hwc": [320, 568, 3],
        "mask_size_hw": [320, 568],
        "payloads": [asdict(item) for item in payloads],
        "unique_rgb_hashes": len({item.rgb_sha256 for item in payloads}),
        "unique_mask_hashes": len({item.mask_sha256 for item in payloads}),
    }
    manifest["corpus_digest"] = _payload_corpus_digest(payloads)
    (run_root / "droid" / "payload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["unique_rgb_hashes"] != len(payloads):
        raise RuntimeError("payload corpus contains duplicate RGB frames; refusing benchmark")
    return payloads


def _load_preserved_payloads(args: argparse.Namespace, run_root: Path) -> list[CachedPayload]:
    raw = json.loads(args.preserved_payload_manifest.read_text(encoding="utf-8"))
    entries = raw.get("payloads") if isinstance(raw, Mapping) else None
    if not isinstance(entries, list) or len(entries) < args.payload_count:
        raise RuntimeError("preserved DROID payload manifest lacks requested payload count")
    payloads: list[CachedPayload] = []
    for item in entries[:args.payload_count]:
        if not isinstance(item, Mapping):
            raise RuntimeError("preserved DROID payload entry must be an object")
        payload = CachedPayload(**{name: item[name] for name in CachedPayload.__dataclass_fields__})
        rgb = Path(payload.rgb_path).read_bytes()
        mask = Path(payload.mask_path).read_bytes()
        if len(rgb) != 320 * 568 * 3 or len(mask) != 320 * 568 * 4:
            raise RuntimeError(f"preserved payload {payload.payload_id} has invalid tensor byte lengths")
        if hashlib.sha256(rgb).hexdigest() != payload.rgb_sha256 or hashlib.sha256(mask).hexdigest() != payload.mask_sha256:
            raise RuntimeError(f"preserved payload {payload.payload_id} hash differs from its manifest")
        payloads.append(payload)
    return _write_payload_manifest(
        run_root, payloads,
        source={"kind": "preserved-droid-binary-payload-manifest.v1", "manifest": str(args.preserved_payload_manifest),
                "manifest_sha256": hashlib.sha256(args.preserved_payload_manifest.read_bytes()).hexdigest()},
    )


def _cache_raw_manifest_payloads(args: argparse.Namespace, run_root: Path) -> list[CachedPayload]:
    import cv2
    import numpy as np

    raw = json.loads(args.raw_frame_manifest.read_text(encoding="utf-8"))
    frames = raw.get("frames") if isinstance(raw, Mapping) else None
    if not isinstance(frames, list) or len(frames) < args.frame_start + args.payload_count:
        raise RuntimeError("raw-frame manifest lacks requested production-length frame range")
    payload_root = run_root / "droid" / "payloads"
    payload_root.mkdir(parents=True, exist_ok=True)
    masks = np.load(args.hand_masks, mmap_mode="r")
    payloads: list[CachedPayload] = []
    for payload_index, entry in enumerate(frames[args.frame_start:args.frame_start + args.payload_count]):
        if not isinstance(entry, Mapping):
            raise RuntimeError("raw-frame manifest entry must be an object")
        source_frame_index = int(entry.get("source_frame_idx", entry.get("frame_idx", payload_index)))
        rgb_path = Path(str(entry.get("raw_frame_path", entry.get("rgb", ""))))
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot read preserved raw frame {rgb_path}")
        if source_frame_index < 0 or source_frame_index >= masks.shape[0]:
            raise RuntimeError(f"raw-frame index {source_frame_index} lacks a preserved static mask")
        rgb = cv2.cvtColor(cv2.resize(bgr, (568, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        static_mask = cv2.resize((~np.asarray(masks[source_frame_index], dtype=bool)).astype(np.float32), (568, 320), interpolation=cv2.INTER_NEAREST)
        rgb_bytes, mask_bytes = rgb.tobytes(order="C"), static_mask.tobytes(order="C")
        cached_rgb = payload_root / f"payload_{payload_index:04d}.rgb.uint8.bin"
        cached_mask = payload_root / f"payload_{payload_index:04d}.static_mask.float32.bin"
        cached_rgb.write_bytes(rgb_bytes)
        cached_mask.write_bytes(mask_bytes)
        payloads.append(CachedPayload(
            payload_id=f"raw-manifest-{payload_index:04d}", source_frame_index=source_frame_index,
            timestamp_s=float(entry.get("source_time_s", entry.get("time_s", source_frame_index / float(raw["fps"])))),
            rgb_sha256=hashlib.sha256(rgb_bytes).hexdigest(), mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
            rgb_path=str(cached_rgb), mask_path=str(cached_mask),
        ))
    return _write_payload_manifest(
        run_root, payloads,
        source={"kind": "preserved-v22-raw-frame-manifest.v1", "manifest": str(args.raw_frame_manifest),
                "manifest_sha256": hashlib.sha256(args.raw_frame_manifest.read_bytes()).hexdigest(),
                "extraction": "none; decoded preserved RGB frame files only"},
    )


def cache_payloads(args: argparse.Namespace, run_root: Path) -> list[CachedPayload]:
    if args.preserved_payload_manifest is not None:
        return _load_preserved_payloads(args, run_root)
    if args.raw_frame_manifest is not None:
        return _cache_raw_manifest_payloads(args, run_root)

    import cv2
    import numpy as np

    payload_root = run_root / "droid" / "payloads"
    payload_root.mkdir(parents=True, exist_ok=True)
    masks = np.load(args.hand_masks, mmap_mode="r")
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"invalid FPS for {args.video}: {fps}")
    if masks.shape[0] < args.payload_count:
        raise RuntimeError(f"mask corpus has {masks.shape[0]} frames, need {args.payload_count}")

    cached: list[CachedPayload] = []
    # Stride through the first tracked interval to avoid near-identical adjacent
    # frames while preserving source-order timestamps within each session stream.
    selected = [args.frame_start + i * args.frame_stride for i in range(args.payload_count)]
    if max(selected) >= masks.shape[0]:
        raise RuntimeError(f"largest selected source frame {max(selected)} exceeds mask corpus {masks.shape[0]}")
    for payload_index, frame_index in enumerate(selected):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"failed decoding source frame {frame_index} from {args.video}")
        rgb = cv2.cvtColor(cv2.resize(bgr, (568, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        # model_masks is the real HaWoR hand/foreground mask. DROID's input is
        # a static-confidence mask, hence its physical complement (1 static,
        # 0 hand/foreground), resized with nearest-neighbor semantics.
        static_mask = (~np.asarray(masks[frame_index], dtype=bool)).astype(np.float32)
        static_mask = cv2.resize(static_mask, (568, 320), interpolation=cv2.INTER_NEAREST)
        rgb_path = payload_root / f"payload_{payload_index:04d}.rgb.uint8.bin"
        mask_path = payload_root / f"payload_{payload_index:04d}.static_mask.float32.bin"
        rgb_path.write_bytes(rgb.tobytes(order="C"))
        mask_path.write_bytes(static_mask.tobytes(order="C"))
        cached.append(CachedPayload(
            payload_id=f"payload-{payload_index:04d}",
            source_frame_index=frame_index,
            timestamp_s=frame_index / fps,
            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            mask_sha256=hashlib.sha256(static_mask.tobytes()).hexdigest(),
            rgb_path=str(rgb_path),
            mask_path=str(mask_path),
        ))
    capture.release()
    return _write_payload_manifest(
        run_root, cached,
        source={"kind": "video-decode.v1", "video": args.video, "hand_masks": args.hand_masks,
                "mask_semantics": "static_confidence_mask = logical NOT of real HaWoR model_masks; 1=static, 0=masked foreground"},
    )


def _payload_corpus_digest(payloads: Iterable[CachedPayload]) -> str:
    """Hash semantic ownership/content evidence without cache-root paths or volatile manifest fields."""
    identity = [
        {
            "payload_id": payload.payload_id,
            "source_frame_index": payload.source_frame_index,
            "timestamp_s": payload.timestamp_s,
            "rgb_sha256": payload.rgb_sha256,
            "mask_sha256": payload.mask_sha256,
        }
        for payload in payloads
    ]
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_payload(payload: CachedPayload) -> tuple[bytes, bytes]:
    return Path(payload.rgb_path).read_bytes(), Path(payload.mask_path).read_bytes()


def record_from_call(
    call: TypedCall,
    *,
    level: str,
    request_id: str,
    session_id: str | None,
    scheduled_s: float | None,
) -> RequestRecord:
    response = call.response
    error = response.error if response is not None else None
    trace: dict[str, Any] | None = None
    actual_replica = call.endpoint.expected_replica_id
    if response is not None and response.server_identity is not None:
        actual_replica = response.server_identity.replica_id
    if isinstance(response, DroidFrameResponse) and response.status is not None:
        trace = response.status.trace.to_wire()
        actual_replica = response.status.trace.replica_id
    elif isinstance(response, DroidFinalizeResponse) and response.camera_state is not None:
        trace = response.camera_state.trace.to_wire()
        actual_replica = response.camera_state.trace.replica_id

    if call.parse_error is not None:
        code = call.parse_error.split(":", 1)[0]
        outcome, error_code, error_message = "failed", code, call.parse_error
    elif call.http_status != 200:
        error_code = error.code.value if error is not None else f"http_{call.http_status}"
        error_message = error.message if error is not None else "typed response returned non-200 HTTP status"
        outcome = (
            "rejected" if error is not None and error.code in {
                ErrorCode.VALIDATION, ErrorCode.CONFLICT, ErrorCode.BACKPRESSURE, ErrorCode.UNRESOLVED,
            } else "failed"
        )
    elif error is not None:
        error_code, error_message = error.code.value, error.message
        outcome = (
            "rejected"
            if error.code in {ErrorCode.VALIDATION, ErrorCode.CONFLICT, ErrorCode.BACKPRESSURE, ErrorCode.UNRESOLVED}
            else "failed"
        )
    else:
        outcome, error_code, error_message = "completed", None, None
    finite_pose_ratio = (
        float(response.camera_state.uncertainty.finite_pose_ratio)
        if isinstance(response, DroidFinalizeResponse) and response.camera_state is not None
        else None
    )
    return RequestRecord(
        operation=call.operation,
        level=level,
        request_id=request_id,
        session_id=session_id,
        scheduled_s=scheduled_s,
        sent_s=call.sent_s,
        completed_s=call.completed_s,
        latency_s=call.latency_s,
        http_status=call.http_status,
        outcome=outcome,
        error_code=error_code,
        error_message=error_message,
        trace=trace,
        diagnostics=(dict(response.batch_diagnostics) if response is not None and isinstance(response.batch_diagnostics, Mapping) else None),
        finite_pose_ratio=finite_pose_ratio,
        endpoint=call.endpoint.base_url,
        replica_id=actual_replica,
        semantic_valid=call.response is not None,
    )


async def await_scheduled(loop: asyncio.AbstractEventLoop, deadline: float) -> None:
    """Wait for an explicit offered-load deadline without polling or asyncio.sleep."""
    future: asyncio.Future[None] = loop.create_future()
    loop.call_at(deadline, future.set_result, None)
    await future


async def create_sessions(
    args: argparse.Namespace,
    client: Any,
    router: StickyDroidRouter,
    level: str,
    level_index: int,
    sessions: int,
) -> tuple[list[str], list[RequestRecord]]:
    async def one(session_index: int) -> tuple[str | None, RequestRecord]:
        request_id = f"{level}-create-{session_index}"
        source_id = f"egoscale-task3-bench-session-{level_index}-{session_index}"
        request = DroidCreateSessionRequest(
            ownership=ownership(request_id, "droid.create_session", source_id),
            camera=DroidCamera.from_mapping(camera_contract()),
            image_shape=DroidImageShape(height=320, width=568),
            # Validated working configuration: serving defaults, matching the V22
            # trajectory run that produced 720 finite frames. The previous weak
            # profile (buffer=128, filter_thresh=1.0, keyframe_thresh=2.0, warmup=2)
            # initialized frontend BA from 2 views and diverged every multi-frame
            # session — proven by the GPU7 options ablation (defaults finite 12/12
            # at every boundary; weak profile NaN at the frontend).
            options=DroidSessionOptions(buffer=getattr(args, "session_buffer", 1024), filter_thresh=2.4, keyframe_thresh=4.0, warmup=8),
            model_revision=args.model_revision,
        )
        call = await router.create_session(client, request)
        record = record_from_call(
            call, level=level, request_id=request_id, session_id=None, scheduled_s=None,
        )
        session_id = (
            call.response.session_id
            if record.outcome == "completed" and isinstance(call.response, DroidCreateSessionResponse)
            else None
        )
        return session_id, record

    results = await asyncio.gather(*(one(i) for i in range(sessions)))
    return [sid for sid, _ in results if sid], [record for _, record in results]


async def benchmark_level(
    args: argparse.Namespace,
    client: Any,
    router: StickyDroidRouter,
    payloads: list[CachedPayload],
    level_index: int,
    sessions: int,
    wave_rate: float,
) -> list[RequestRecord]:
    level = f"sessions-{sessions}_waves-per-s-{wave_rate:g}"
    one_replica_control = sessions == 1
    if not one_replica_control and sessions % len(router.endpoints) != 0:
        raise ValueError(
            f"{level} cannot compare {len(router.endpoints)} replicas with equal sticky session assignment"
        )
    session_ids, records = await create_sessions(args, client, router, level, level_index, sessions)
    if len(session_ids) != sessions:
        raise RuntimeError(
            f"{level} lacks expected replica completeness: created {len(session_ids)} of {sessions} sessions"
        )
    assigned = await asyncio.gather(*(router.endpoint_for_session(session_id) for session_id in session_ids))
    assignment_counts = Counter(endpoint.expected_replica_id for endpoint in assigned if endpoint is not None)
    planned_session_assignment = {
        endpoint.expected_replica_id: (
            assignment_counts[endpoint.expected_replica_id] if one_replica_control else sessions // len(router.endpoints)
        )
        for endpoint in router.endpoints
    }
    if dict(assignment_counts) != {replica_id: count for replica_id, count in planned_session_assignment.items() if count}:
        raise RuntimeError(
            f"{level} violates planned sticky session assignment; expected {planned_session_assignment}, "
            f"observed {dict(assignment_counts)}"
        )
    loop = asyncio.get_running_loop()
    start = loop.time() + args.start_delay_s

    async def push(session_index: int, wave_index: int) -> RequestRecord:
        scheduled_s = start + wave_index / wave_rate
        await await_scheduled(loop, scheduled_s)
        payload_index = (level_index * args.waves * sessions + wave_index * sessions + session_index) % len(payloads)
        payload = payloads[payload_index]
        rgb, mask = read_payload(payload)
        request_id = f"{level}-push-{wave_index:03d}-{session_index}"
        source_id = f"egoscale-task3:{payload.source_frame_index}"
        request = DroidFrameRequest(
            ownership=ownership(request_id, "droid.push_frame", source_id, payload.timestamp_s),
            session_id=session_ids[session_index],
            frame_id=f"{session_index}-{wave_index}-{payload.source_frame_index}",
            source_timestamp_s=payload.timestamp_s,
            rgb=TensorPayload(rgb, (320, 568, 3), "uint8"),
            static_confidence_mask=TensorPayload(mask, (320, 568), "float32"),
            model_revision=args.model_revision,
        )
        call = await router.push_frame(client, request)
        return record_from_call(
            call, level=level, request_id=request_id,
            session_id=session_ids[session_index], scheduled_s=scheduled_s,
        )

    pushed = await asyncio.gather(*(push(session_index, wave_index)
                                    for wave_index in range(args.waves) for session_index in range(sessions)))
    records.extend(pushed)

    async def finalize(session_index: int) -> RequestRecord:
        request_id = f"{level}-finalize-{session_index}"
        source_id = f"egoscale-task3-bench-session-{level_index}-{session_index}"
        request = DroidFinalizeRequest(
            ownership=ownership(request_id, "droid.finalize", source_id),
            session_id=session_ids[session_index],
            model_revision=args.model_revision,
        )
        call = await router.finalize(client, request)
        return record_from_call(
            call, level=level, request_id=request_id,
            session_id=session_ids[session_index], scheduled_s=None,
        )
    # Independent sessions remain sticky while their terminal work is gathered in
    # parallel. Client serialization must not become an unreported finalize limit.
    records.extend(await asyncio.gather(*(finalize(session_index) for session_index in range(sessions))))
    replica_ids = tuple(endpoint.expected_replica_id for endpoint in router.endpoints)
    if one_replica_control:
        validate_one_replica_control_offers(
            records, expected_replica_ids=replica_ids, planned_session_assignment=planned_session_assignment,
        )
    else:
        validate_equal_replica_offers(records, expected_replica_ids=replica_ids)
    treatment_role = "one_replica_control" if one_replica_control else "two_replica_scaling"
    for record in records:
        record.treatment_role = treatment_role
        record.planned_session_assignment = dict(planned_session_assignment)
    return records


def validate_one_replica_control_offers(
    records: Iterable[RequestRecord], *, expected_replica_ids: tuple[str, ...],
    planned_session_assignment: dict[str, int],
) -> None:
    """Accept S=1 only as an explicitly labeled one-replica control.

    With a sticky round-robin router, exactly one endpoint owns the sole
    session; the owner rotates across sequential levels. That 1/0 assignment is
    useful as the single-replica baseline but
    cannot be represented as balanced two-replica scaling evidence.

    A single-endpoint topology is the degenerate case: all work goes to the
    only replica, which is trivially a valid one-replica control.
    """
    if not expected_replica_ids:
        raise ValueError("one-replica control requires at least one configured replica")
    if set(planned_session_assignment) != set(expected_replica_ids):
        raise ValueError("one-replica control requires every configured replica in its assignment evidence")
    active = [replica_id for replica_id, count in planned_session_assignment.items() if count]
    if len(active) != 1 or any(count not in {0, 1} for count in planned_session_assignment.values()):
        raise ValueError("one-replica control must explicitly declare a 1/0 session assignment")
    offered = [record for record in records if record.operation == "push_frame"]
    if not offered:
        raise RuntimeError("one-replica control lacks push offers")
    unexpected = [record.replica_id for record in offered if record.replica_id != active[0]]
    if unexpected:
        raise RuntimeError(f"one-replica control moved work away from its sticky owner: {unexpected}")


def validate_equal_replica_offers(
    records: Iterable[RequestRecord], *, expected_replica_ids: tuple[str, ...],
    timestamp_jitter_tolerance_s: float = OFFER_TIMESTAMP_JITTER_TOLERANCE_S,
) -> None:
    """Reject an unequal/missing replica treatment before it becomes scaling evidence."""
    if not expected_replica_ids or len(set(expected_replica_ids)) != len(expected_replica_ids):
        raise ValueError("expected replica ids must be non-empty and unique")
    if timestamp_jitter_tolerance_s < 0:
        raise ValueError("timestamp jitter tolerance must be non-negative")
    per_replica: dict[str, list[RequestRecord]] = {replica_id: [] for replica_id in expected_replica_ids}
    for record in records:
        if record.operation != "push_frame":
            continue
        if record.replica_id not in per_replica:
            raise RuntimeError(f"push offer has missing or unexpected replica identity {record.replica_id!r}")
        per_replica[record.replica_id].append(record)
    counts = {replica_id: len(items) for replica_id, items in per_replica.items()}
    if not all(counts.values()):
        raise RuntimeError(f"scaling offer lacks expected replica work: {counts}")
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"scaling offer work assignment is unequal: {counts}")
    first_submits = {replica_id: min(item.sent_s for item in items) for replica_id, items in per_replica.items()}
    final_submits = {replica_id: max(item.sent_s for item in items) for replica_id, items in per_replica.items()}
    common_first = min(first_submits.values())
    common_final = max(final_submits.values())
    if any(abs(value - common_first) > timestamp_jitter_tolerance_s for value in first_submits.values()):
        raise RuntimeError(
            f"scaling offer does not share a common start within {timestamp_jitter_tolerance_s}s jitter: {first_submits}"
        )
    if any(abs(value - common_final) > timestamp_jitter_tolerance_s for value in final_submits.values()):
        raise RuntimeError(
            f"scaling offer does not share a common end within {timestamp_jitter_tolerance_s}s jitter: {final_submits}"
        )


def _offer_accounting(records: Iterable[RequestRecord]) -> dict[str, Any]:
    """Use actual client submissions as the offer denominator, not drain completion."""
    offered = sorted(records, key=lambda item: item.sent_s)
    if not offered:
        return {"run_start_s": None, "first_submit_s": None, "final_submit_s": None,
                "actual_offer_window_s": 0.0, "drain_end_s": None, "drain_duration_s": 0.0,
                "actual_submitted_rate_per_s": 0.0}
    run_start_s = min(item.scheduled_s for item in offered if item.scheduled_s is not None)
    first_submit_s, final_submit_s = offered[0].sent_s, offered[-1].sent_s
    drain_end_s = max(item.completed_s for item in offered)
    window_s = max(final_submit_s - first_submit_s, 0.0)
    return {
        "run_start_s": run_start_s,
        "first_submit_s": first_submit_s,
        "final_submit_s": final_submit_s,
        "actual_offer_window_s": window_s,
        "drain_end_s": drain_end_s,
        "drain_duration_s": max(drain_end_s - final_submit_s, 0.0),
        "actual_submitted_rate_per_s": rate(len(offered), window_s),
    }


def _allocator_summary(records: Iterable[RequestRecord]) -> dict[str, int | None]:
    """Summarize only allocator facts emitted by the resident worker."""
    snapshots = [
        diagnostics.get("allocator_memory")
        for item in records
        if isinstance((diagnostics := item.diagnostics), Mapping)
    ]
    snapshots = [snapshot for snapshot in snapshots if isinstance(snapshot, Mapping)]

    names = ("allocated_bytes", "reserved_bytes", "max_allocated_bytes", "max_reserved_bytes")
    result: dict[str, int | None] = {}
    for name in names:
        values = [int(snapshot[name]) for snapshot in snapshots if isinstance(snapshot.get(name), int) and snapshot[name] >= 0]
        result[name] = max(values) if values else None
    return result


def summarize(records: list[RequestRecord]) -> list[dict[str, Any]]:
    by_level: dict[str, list[RequestRecord]] = defaultdict(list)
    for record in records:
        by_level[record.level].append(record)
    rows: list[dict[str, Any]] = []
    for level, group in sorted(by_level.items()):
        push_items = [item for item in group if item.operation == "push_frame"]
        offer = _offer_accounting(push_items)
        per_replica_push: dict[str, list[RequestRecord]] = defaultdict(list)
        for item in push_items:
            per_replica_push[item.replica_id or "missing"].append(item)
        treatment_roles = {item.treatment_role for item in group}
        planned_assignments = {json.dumps(item.planned_session_assignment, sort_keys=True) for item in group}
        if len(treatment_roles) != 1 or len(planned_assignments) != 1:
            raise RuntimeError(f"{level} has inconsistent treatment labeling or sticky assignment evidence")
        treatment_role = next(iter(treatment_roles))
        planned_session_assignment = json.loads(next(iter(planned_assignments)))
        if planned_session_assignment is None:
            # Unit-level accounting callers can construct records directly; live
            # benchmark records always carry the explicit sticky assignment above.
            planned_session_assignment = {replica_id: 1 for replica_id in sorted(per_replica_push)}
        for operation in ("create_session", "push_frame", "finalize"):
            items = [item for item in group if item.operation == operation]
            if not items:
                continue
            completed = [item for item in items if item.outcome == "completed"]
            rejected = [item for item in items if item.outcome == "rejected"]
            traces = [item.trace for item in completed if item.trace]
            batches = {str(trace["batch_id"]): trace for trace in traces if "batch_id" in trace}
            batch_sizes = Counter(int(trace["effective_work_units"]) for trace in batches.values() if "effective_work_units" in trace)
            finite_pose_ratios = [
                item.finite_pose_ratio
                for item in completed
                if item.finite_pose_ratio is not None
            ]
            row: dict[str, Any] = {
                "level": level, "operation": operation,
                "treatment_role": treatment_role,
                "planned_session_assignment": planned_session_assignment,
                "offered_count": len(items), "admitted_count": len(completed),
                "completed_count": len(completed), "rejected_count": len(rejected),
                "failed_count": len(items) - len(completed) - len(rejected),
                "response_observation_s": max(item.completed_s for item in items) - min(item.sent_s for item in items),
                "response_p50_s": percentile((item.latency_s for item in items), 0.50),
                "response_p95_s": percentile((item.latency_s for item in items), 0.95),
                "response_p99_s": percentile((item.latency_s for item in items), 0.99),
                "fused_batch_size_distribution": dict(sorted(batch_sizes.items())),
                "fused_forward_count": sum(int(trace.get("fnet_forward_count", 0)) for trace in batches.values()),
                "session_local_forward_count": sum(int(trace.get("session_local_forward_count", 0)) for trace in traces),
                "model_load_counts": sorted({int(trace["model_load_count"]) for trace in traces if "model_load_count" in trace}),
                "semantic_valid_count": sum(1 for item in items if item.semantic_valid),
                "sticky_endpoints": dict(Counter(item.endpoint or "none" for item in items)),
                "sticky_replicas": dict(Counter(item.replica_id or "none" for item in items)),
                "failure_modes": dict(Counter((item.error_code or "none") for item in items if item.outcome != "completed")),
                "reject_reasons": dict(Counter((item.error_message or item.error_code or "none") for item in rejected)),
                "allocator_memory": _allocator_summary(items),
                "finite_pose_ratio": {
                    "count": len(finite_pose_ratios),
                    "min": min(finite_pose_ratios) if finite_pose_ratios else None,
                    "max": max(finite_pose_ratios) if finite_pose_ratios else None,
                    "values": finite_pose_ratios,
                },
            }
            if operation == "push_frame":
                window_s = offer["actual_offer_window_s"]
                row.update(offer)
                row.update({
                    "offered_rate_per_s": offer["actual_submitted_rate_per_s"],
                    "admitted_rate_per_s": rate(len(completed), window_s),
                    "completed_rate_per_s": rate(len(completed), window_s),
                    "rejected_rate_per_s": rate(len(rejected), window_s),
                    "per_replica_actual_offer": {
                        replica_id: {"submitted_count": len(per_replica_push[replica_id]), **_offer_accounting(per_replica_push[replica_id])}
                        for replica_id in sorted(planned_session_assignment)
                    },
                })
            else:
                row.update({"offered_rate_per_s": None, "admitted_rate_per_s": None,
                            "completed_rate_per_s": None, "rejected_rate_per_s": None})
            rows.append(row)
    return rows


def write_plots(rows: list[dict[str, Any]], plot_dir: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return repr(exc)
    plot_dir.mkdir(parents=True, exist_ok=True)
    push = [r for r in rows if r["operation"] == "push_frame"]
    if not push:
        return "no push rows"
    x = list(range(len(push)))
    labels = [r["level"].replace("_waves-per-s-", "\n") for r in push]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    axes[0].plot(x, [r["offered_rate_per_s"] for r in push], "o-", label="offered")
    axes[0].plot(x, [r["completed_rate_per_s"] for r in push], "o-", label="completed")
    axes[0].plot(x, [r["rejected_rate_per_s"] for r in push], "o-", label="rejected")
    axes[0].set_ylabel("frames / s")
    axes[0].legend()
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    for q, style in (("response_p50_s", "o-"), ("response_p95_s", "s-"), ("response_p99_s", "^-")):
        axes[1].plot(x, [r[q] or 0.0 for r in push], style, label=q.replace("response_", ""))
    axes[1].set_ylabel("response latency (s)")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].legend()
    fig.savefig(plot_dir / "push_throughput_latency.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    widths = []
    for r in push:
        widths.append(sum(count for size, count in r["fused_batch_size_distribution"].items() if int(size) > 1))
    ax.bar(x, widths)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("push responses in fused batches > 1")
    fig.savefig(plot_dir / "fused_batch_distribution.png", dpi=160)
    plt.close(fig)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint", required=True,
        help="comma-separated DROID HTTP base URLs; sessions are round-robin created and sticky thereafter",
    )
    parser.add_argument(
        "--replica-ids", default=None,
        help="comma-separated server replica ids aligned with --endpoint (default: ids from runtime identity file)",
    )
    parser.add_argument(
        "--runtime-identities", required=True, type=Path,
        help="DROID launch-plan JSON containing expected_server_identity for every explicit endpoint",
    )
    parser.add_argument("--gcs-address", required=True, help="comma-separated existing Ray GCS addresses for evidence only")
    parser.add_argument("--run-root", required=True, help="fresh /vePFS-Mindverse/.../ray_serve_benchmarks/<run_id> root")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--hand-masks", default=DEFAULT_HAND_MASKS)
    parser.add_argument("--payload-count", type=int, default=192)
    parser.add_argument("--preserved-payload-manifest", type=Path,
                        help="read-only validated binary payload manifest; no video decode or extraction")
    parser.add_argument("--raw-frame-manifest", type=Path,
                        help="read-only V22 raw-frame manifest; decode its preserved RGB files without video extraction")
    parser.add_argument("--frame-start", type=int, default=60)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--waves", type=int, default=12)
    parser.add_argument("--session-buffer", type=int, default=1024,
                        help="DepthVideo buffer slots; all other session options remain validated serving defaults")
    parser.add_argument("--sessions", default="1,2,4", help="D1/D2/D4 active-session levels; add 8 only when memory/queue evidence permits")
    parser.add_argument("--wave-rates", default="0.5,1,2,4,8")
    parser.add_argument("--start-delay-s", type=float, default=0.25)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--model-revision", default="droid-v1")
    parser.add_argument("--corpus-digest", required=True, help="expected SHA256 from a prior payload manifest")
    parser.add_argument("--measurement-interval-id", required=True,
                        help="immutable label shared only by directly comparable offered-load treatments")
    parser.add_argument("--nvml-interval-s", type=float, default=0.2)
    return parser.parse_args()


def _runtime_identities(path: Path) -> tuple[ServerIdentity, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    replicas = raw.get("replicas") if isinstance(raw, Mapping) else None
    if not isinstance(replicas, list) or not replicas:
        raise ValueError("runtime identity file must be a DROID launch plan with non-empty replicas")
    identities: list[ServerIdentity] = []
    for replica in replicas:
        if not isinstance(replica, Mapping) or not isinstance(replica.get("expected_server_identity"), Mapping):
            raise ValueError("every launch-plan replica requires expected_server_identity")
        identities.append(ServerIdentity.from_wire(replica["expected_server_identity"]))
    return tuple(identities)


def replica_endpoints(args: argparse.Namespace) -> tuple[ReplicaEndpoint, ...]:
    urls = tuple(value.strip() for value in args.endpoint.split(",") if value.strip())
    if not urls:
        raise ValueError("--endpoint must contain at least one URL")
    if len(set(urls)) != len(urls):
        raise ValueError("DROID benchmark endpoints must be disjoint")
    identities = _runtime_identities(args.runtime_identities)
    plan_contract = json.loads(args.runtime_identities.read_text(encoding="utf-8"))
    for field in ("corpus_digest", "measurement_interval_id"):
        expected = getattr(args, field, None)
        planned = plan_contract.get(field) if isinstance(plan_contract, Mapping) else None
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(f"DROID benchmark requires non-empty {field}")
        if not isinstance(planned, str) or not planned.strip():
            raise ValueError(f"DROID launch plan requires non-empty {field}")
        if planned != expected:
            raise ValueError(f"DROID benchmark {field} differs from immutable launch plan")
    if len(identities) != len(urls):
        raise ValueError("runtime identity count must equal explicit endpoint count")
    if args.replica_ids:
        replica_ids = tuple(value.strip() for value in args.replica_ids.split(",") if value.strip())
        if len(replica_ids) != len(urls):
            raise ValueError("--replica-ids count must equal --endpoint count")
    else:
        replica_ids = tuple(identity.replica_id for identity in identities)
    if len(set(replica_ids)) != len(replica_ids):
        raise ValueError("DROID benchmark replica ids must be disjoint")
    endpoints = tuple(
        ReplicaEndpoint(url, replica_id, args.model_revision, identity)
        for url, replica_id, identity in zip(urls, replica_ids, identities)
    )
    if len({endpoint.expected_replica_id for endpoint in endpoints}) != len(endpoints):
        raise ValueError("DROID benchmark replica ids must be disjoint")
    if len({endpoint.base_url for endpoint in endpoints}) != len(endpoints):
        raise ValueError("DROID benchmark endpoints must be disjoint")
    if len({endpoint.runtime_identity.assigned_gpu for endpoint in endpoints}) != len(endpoints):
        raise ValueError("DROID benchmark replicas must own disjoint physical GPUs")
    if len({endpoint.runtime_identity.cuda_uuid.strip().lower().removeprefix("gpu-") for endpoint in endpoints}) != len(endpoints):
        raise ValueError("DROID benchmark replicas must own disjoint physical CUDA identities")
    compatibility = {
        "experiment_id": {endpoint.runtime_identity.experiment_id for endpoint in endpoints},
        "checkpoint_digest": {endpoint.runtime_identity.checkpoint_digest for endpoint in endpoints},
        "release_digest": {endpoint.runtime_identity.release_digest for endpoint in endpoints},
        "release_source": {endpoint.runtime_identity.release_sha for endpoint in endpoints},
        "dependency_digest": {endpoint.runtime_identity.dependency_digest for endpoint in endpoints},
        "dependency_root": {endpoint.runtime_identity.dependency_root for endpoint in endpoints},
        "source_amendment_id": {endpoint.runtime_identity.source_amendment_id for endpoint in endpoints},
        "model_revision": {endpoint.runtime_identity.model_revision for endpoint in endpoints},
        "schema_version": {endpoint.runtime_identity.schema_version for endpoint in endpoints},
    }
    mismatched = [name for name, values in compatibility.items() if len(values) != 1 or None in values]
    if mismatched:
        raise ValueError("DROID benchmark endpoints have incompatible immutable identity: " + ", ".join(mismatched))
    return endpoints


async def async_main(args: argparse.Namespace) -> int:
    import httpx

    run_root = Path(args.run_root)
    droid_root = run_root / "droid"
    droid_root.mkdir(parents=True, exist_ok=False)
    (droid_root / "raw").mkdir()
    endpoints = replica_endpoints(args)
    gcs_addresses = tuple(value.strip() for value in args.gcs_address.split(",") if value.strip())
    expected_gcs = tuple(endpoint.runtime_identity.gcs_address for endpoint in endpoints)
    if gcs_addresses != expected_gcs:
        raise ValueError("--gcs-address values must align exactly with verified endpoint runtime identities")
    command = {
        "argv": sys.argv,
        "started_at": utc_now(),
        "cwd": os.getcwd(),
        "gcs": [capture_gcs(address) for address in gcs_addresses],
        "replicas": [{"base_url": endpoint.base_url, "expected_replica_id": endpoint.expected_replica_id,
                      "expected_model_revision": endpoint.expected_model_revision,
                      "runtime_identity": endpoint.runtime_identity.to_wire()} for endpoint in endpoints],
    }
    (droid_root / "raw" / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    payloads = cache_payloads(args, run_root)
    payload_manifest = json.loads((droid_root / "payload_manifest.json").read_text(encoding="utf-8"))
    if payload_manifest.get("corpus_digest") != args.corpus_digest:
        raise ValueError("cached DROID corpus digest differs from --corpus-digest")
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout_s)) as client:
        status_before: dict[str, Any] = {}
        for endpoint in endpoints:
            try:
                raw_status = await client.get(f"{endpoint.base_url}/droid/status")
                status_before[endpoint.expected_replica_id] = {
                    "endpoint": endpoint.base_url,
                    "http_status": int(raw_status.status_code),
                    "body": raw_status.text,
                }
            except Exception as exc:
                status_before[endpoint.expected_replica_id] = {
                    "endpoint": endpoint.base_url, "error": repr(exc),
                }
        (droid_root / "raw" / "status_before.json").write_text(json.dumps(status_before, indent=2) + "\n")
        router = StickyDroidRouter(endpoints)
        records: list[RequestRecord] = []
        levels = [(sessions, wave_rate) for sessions in (int(v) for v in args.sessions.split(","))
                  for wave_rate in (float(v) for v in args.wave_rates.split(","))]
        experiment_id = endpoints[0].runtime_identity.experiment_id
        release_digest = endpoints[0].runtime_identity.release_digest
        assert release_digest is not None
        gpu_uuids = {endpoint.runtime_identity.assigned_gpu: str(endpoint.runtime_identity.cuda_uuid) for endpoint in endpoints}
        sampler = NvmlSampler(
            gpu_ids=tuple(gpu_uuids), gpu_uuids=gpu_uuids,
            experiment_id=experiment_id, release_digest=release_digest,
            interval_s=args.nvml_interval_s,
        )
        run_start_s = sampler.start()
        try:
            for index, (sessions, wave_rate) in enumerate(levels):
                sampler.set_level(f"D{sessions}:waves-per-s-{wave_rate:g}")
                records.extend(await benchmark_level(args, client, router, payloads, index, sessions, wave_rate))
        finally:
            run_end_s = sampler.stop()
            gpu_samples_path = sampler.write(droid_root / "raw" / "gpu_samples.json")
        status_after: dict[str, Any] = {}
        for endpoint in endpoints:
            try:
                raw_status = await client.get(f"{endpoint.base_url}/droid/status")
                status_after[endpoint.expected_replica_id] = {
                    "endpoint": endpoint.base_url,
                    "http_status": int(raw_status.status_code),
                    "body": raw_status.text,
                }
            except Exception as exc:
                status_after[endpoint.expected_replica_id] = {
                    "endpoint": endpoint.base_url, "error": repr(exc),
                }
        (droid_root / "raw" / "status_after.json").write_text(json.dumps(status_after, indent=2) + "\n")
    gpu_evidence = validate_gpu_samples(
        gpu_samples_path,
        gpu_ids=tuple(gpu_uuids),
        gpu_uuids=gpu_uuids,
        experiment_id=experiment_id,
        release_digest=release_digest,
        run_start_s=run_start_s,
        run_end_s=run_end_s,
        min_samples_per_gpu=2,
    )
    with (droid_root / "requests.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")
    rows = summarize(records)
    (droid_root / "summary.json").write_text(json.dumps({
        "created_at": utc_now(), "experiment_id": experiment_id,
        "release_digest": release_digest, "release_source": endpoints[0].runtime_identity.release_sha,
        "checkpoint_digest": endpoints[0].runtime_identity.checkpoint_digest,
        "model_revision": endpoints[0].runtime_identity.model_revision,
        "schema_version": endpoints[0].runtime_identity.schema_version,
        "corpus_digest": args.corpus_digest, "measurement_interval_id": args.measurement_interval_id,
        "session_buffer": args.session_buffer,
        "gpu_ids": list(gpu_uuids), "rows": rows,
    }, indent=2) + "\n")
    (droid_root / "run_manifest.json").write_text(json.dumps({
        "schema": "ego.droid-scaling-run.v1",
        "experiment_id": experiment_id,
        "release_digest": release_digest,
        "explicit_endpoints": [endpoint.base_url for endpoint in endpoints],
        "runtime_identities": [endpoint.runtime_identity.to_wire() for endpoint in endpoints],
        "session_levels": sorted({sessions for sessions, _ in levels}),
        "wave_rates": sorted({wave_rate for _, wave_rate in levels}),
        "session_buffer": args.session_buffer,
        "payload_source": (
            "preserved-binary-manifest" if args.preserved_payload_manifest is not None
            else "preserved-raw-frame-manifest" if args.raw_frame_manifest is not None else "video-decode"
        ),
        "corpus_digest": args.corpus_digest,
        "measurement_interval_id": args.measurement_interval_id,
        "nvml": {
            "path": str(gpu_samples_path),
            "schema": gpu_evidence.get("schema"),
            "interval_s": gpu_evidence.get("sample_interval_s"),
            "run_start_s": run_start_s,
            "run_end_s": run_end_s,
            "gpu_uuids": gpu_uuids,
        },
        "canonical_router_mutated": False,
        "measurement_definition": "droid-open-loop-sticky-v3-actual-submission-offer-window",
    }, indent=2) + "\n")
    with (droid_root / "summary.csv").open("w", newline="") as handle:
        # Row shapes differ by operation (create rows lack push offer-accounting
        # fields); the header must be the union in first-appearance order.
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["level"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    plot_error = write_plots(rows, droid_root / "plots")
    if plot_error:
        (droid_root / "plot_error.txt").write_text(plot_error + "\n")
    print(json.dumps({"run_root": str(run_root), "summary": str(droid_root / "summary.json"), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main(parse_args())))
