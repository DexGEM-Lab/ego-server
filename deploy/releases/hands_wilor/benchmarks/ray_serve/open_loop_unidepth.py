"""Isolated open-loop HTTP sweep for the live GPU0 UniDepth Ray Serve endpoint.

This is intentionally scoped to the durable GPU0 lane: it never creates a Ray
cluster, never changes Serve state, and connects to the already-running GPU0 cluster
only through its explicit GCS address.  Arrivals are scheduled at fixed offered rates
rather than issued after a preceding request completes, so queueing latency remains
visible.

The two preserved 540x960 EgoScale frames are alternated and their SHA-256 hashes are
recorded.  The resulting raw request rows include ownership, latency, batch trace,
and any explicit endpoint error; summary percentiles are computed only from successful
requests.  This is a GPU0 endpoint capacity measurement, not a claim about a broader
EgoScale corpus.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import numpy as np
import ray

LANE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LANE_ROOT))
from ego_annotation.serving.contracts import (  # noqa: E402
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

GCS_ADDRESS = os.environ.get("EGO_UNIDEPTH_GCS_ADDRESS", "192.168.42.193:26000")
ENDPOINT = os.environ.get("EGO_UNIDEPTH_ENDPOINT", "http://127.0.0.1:28000")
REVISION = os.environ.get("EGO_UNIDEPTH_REVISION", "unidepth-v2-vitl14-corrected")
FRAME_DIR = Path(os.environ.get("EGO_FRAMES_DIR", str(LANE_ROOT / "egoscale_frames")))
H, W = 540, 960


def _pixel_transform() -> PixelTransform:
    return PixelTransform(
        source_to_model=((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 1.0)),
        model_to_source=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.0)),
        resize_mode="area",
    )


def _request(rgb: np.ndarray, *, request_id: str, source_id: str) -> UniDepthRequest:
    ownership = Ownership(
        request_id=request_id,
        job_id="gpu0-open-loop-sweep",
        item_id=source_id,
        stage_id="unidepth.infer",
        source_id=source_id,
    )
    spatial = SpatialMetadata(
        source_size=ImageSize(width=1920, height=1080),
        model_size=ImageSize(width=W, height=H),
        color_space="RGB",
        pixel_transform=_pixel_transform(),
        K_px=None,
    )
    return UniDepthRequest(
        ownership=ownership,
        rgb=TensorPayload(data=rgb.tobytes(), shape=rgb.shape, dtype="uint8"),
        spatial=spatial,
        model_revision=REVISION,
    )


def _multipart(request: UniDepthRequest) -> tuple[bytes, str]:
    metadata = {
        "ownership": request.ownership.to_wire(),
        "spatial": request.spatial.to_wire(),
        "model_revision": request.model_revision,
        "options": dict(request.options),
    }
    return build_multipart_request(
        metadata,
        rgb=bytes(request.rgb.data),
        rgb_shape=request.rgb.shape,
        rgb_dtype=request.rgb.dtype,
    )


def _cluster_snapshot() -> dict[str, Any]:
    """Collect best-effort health evidence without invalidating completed load rows."""
    try:
        ray.init(address=GCS_ADDRESS, namespace="serve", ignore_reinit_error=True, logging_level="ERROR")
        snapshot = {
            "gcs_address": GCS_ADDRESS,
            "cluster_resources": dict(ray.cluster_resources()),
            "available_resources": dict(ray.available_resources()),
        }
        try:
            from ray import serve

            snapshot["serve_status"] = str(serve.status())
        except Exception as exc:
            # A detached benchmark client can see a transient stale controller handle;
            # request rows are still valid endpoint evidence and must survive it.
            snapshot["serve_status_error"] = f"{type(exc).__name__}: {exc}"
        return snapshot
    except Exception as exc:
        return {"gcs_address": GCS_ADDRESS, "cluster_snapshot_error": f"{type(exc).__name__}: {exc}"}
    finally:
        if ray.is_initialized():
            ray.shutdown()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


async def _await_scheduled(loop: asyncio.AbstractEventLoop, when: float) -> None:
    """Yield until an absolute offered-load arrival without sleep/polling."""
    ready: asyncio.Future[None] = loop.create_future()
    loop.call_at(when, ready.set_result, None)
    await ready


async def _post_one(
    client: httpx.AsyncClient,
    *,
    row_id: int,
    level: float,
    scheduled_at: float,
    rgb: np.ndarray,
    source_id: str,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    await _await_scheduled(loop, scheduled_at)
    request_id = f"sweep-{level:g}-{row_id:04d}-{uuid4().hex[:8]}"
    request = _request(rgb, request_id=request_id, source_id=source_id)
    body, content_type = _multipart(request)
    started = time.monotonic()
    row: dict[str, Any] = {
        "request_id": request_id,
        "job_id": request.ownership.job_id,
        "source_id": source_id,
        "offered_images_per_s": level,
        "scheduled_offset_s": scheduled_at,
        "actual_start_offset_s": started,
        "schedule_lag_ms": round((started - scheduled_at) * 1_000, 3),
    }
    try:
        response = await client.post(
            f"{ENDPOINT}/", content=body, headers={"Content-Type": content_type}, timeout=90.0
        )
        completed = time.monotonic()
        row.update({
            "http_status": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "latency_ms": round((completed - started) * 1_000, 3),
        })
        try:
            metadata, arrays = parse_multipart_response(response.content, row["content_type"])
            error = metadata.get("error")
            if error:
                row.update({"outcome": "endpoint_error", "error_code": error.get("code"), "error_message": error.get("message")})
            else:
                result = metadata["result"]
                trace = result["trace"]
                row.update({
                    "outcome": "success",
                    "ownership_response_id": metadata["ownership"]["request_id"],
                    "batch_id": trace["batch_id"],
                    "batch_request_count": trace["request_count"],
                    "forward_count": trace["forward_count"],
                    "model_load_count": trace["model_load_count"],
                    # The live BatchTrace exposes monotonic callback boundaries, not
                    # fabricated per-stage timings.  These are replica-clock spans:
                    # admission-to-dispatch includes queue wait/batch formation;
                    # forward-to-complete includes the model forward plus output split.
                    "batch_formation_delay_ms": round(
                        (trace["dispatched_monotonic_s"] - trace["admitted_monotonic_s"]) * 1_000, 3
                    ),
                    "batch_forward_and_output_ms": round(
                        (trace["completed_monotonic_s"] - trace["forward_started_monotonic_s"]) * 1_000, 3
                    ),
                    "depth_shape": arrays["depth_m"][1],
                    "K_shape": arrays["K_px"][1],
                })
        except Exception as exc:  # A malformed response is a measured endpoint failure.
            row.update({"outcome": "transport_parse_error", "error_code": type(exc).__name__, "error_message": str(exc)})
    except Exception as exc:  # Timeout/connection errors remain explicit raw results.
        completed = time.monotonic()
        row.update({
            "outcome": "client_error",
            "error_code": type(exc).__name__,
            "error_message": str(exc),
            "latency_ms": round((completed - started) * 1_000, 3),
        })
    return row


async def _level(level: float, count: int, frames: list[tuple[str, np.ndarray]]) -> list[dict[str, Any]]:
    loop = asyncio.get_running_loop()
    level_start = loop.time() + 0.2
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=count, max_keepalive_connections=count)) as client:
        tasks = [
            _post_one(
                client,
                row_id=index,
                level=level,
                scheduled_at=level_start + index / level,
                rgb=frames[index % len(frames)][1],
                source_id=frames[index % len(frames)][0],
            )
            for index in range(count)
        ]
        rows = await asyncio.gather(*tasks)
    for row in rows:
        row["scheduled_offset_s"] = round(row["scheduled_offset_s"] - level_start, 6)
        row["actual_start_offset_s"] = round(row["actual_start_offset_s"] - level_start, 6)
    return rows


def _summarize(level: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = [row for row in rows if row.get("outcome") == "success"]
    latencies = [float(row["latency_ms"]) for row in success]
    elapsed = max((float(row["actual_start_offset_s"]) + float(row["latency_ms"]) / 1_000 for row in rows), default=0.0)
    batch_sizes = [int(row["batch_request_count"]) for row in success]
    ownership_ok = all(row.get("ownership_response_id") == row["request_id"] for row in success)
    return {
        "offered_images_per_s": level,
        "request_count": len(rows),
        "success_count": len(success),
        "endpoint_error_count": sum(row.get("outcome") == "endpoint_error" for row in rows),
        "client_error_count": sum(row.get("outcome") == "client_error" for row in rows),
        "transport_parse_error_count": sum(row.get("outcome") == "transport_parse_error" for row in rows),
        "achieved_images_per_s": round(len(success) / elapsed, 4) if elapsed else 0.0,
        "response_latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 3) if latencies else None,
            "p95": round(_percentile(latencies, 0.95), 3) if latencies else None,
            "p99": round(_percentile(latencies, 0.99), 3) if latencies else None,
        },
        "effective_batch_size": {
            "min": min(batch_sizes) if batch_sizes else None,
            "max": max(batch_sizes) if batch_sizes else None,
            "mean": round(statistics.mean(batch_sizes), 4) if batch_sizes else None,
            "distribution": {str(size): batch_sizes.count(size) for size in sorted(set(batch_sizes))},
        },
        "one_forward_per_batch": all(row.get("forward_count") == 1 for row in success),
        "ownership_split_ok": ownership_ok,
        "resident_model_load_counts": sorted(set(row.get("model_load_count") for row in success)),
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    frame_paths = [FRAME_DIR / "frame_A_idx000.npy", FRAME_DIR / "frame_A_idx030.npy"]
    frames: list[tuple[str, np.ndarray]] = []
    manifest = []
    for path in frame_paths:
        rgb = np.load(path).astype(np.uint8)
        if rgb.shape != (H, W, 3):
            raise ValueError(f"expected canonical {H}x{W}x3 frame, got {rgb.shape} from {path}")
        source_id = path.stem
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        frames.append((source_id, rgb))
        manifest.append({"source_id": source_id, "shape": list(rgb.shape), "dtype": rgb.dtype.name, "sha256": digest})
    (out_dir / "payload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    pre = _cluster_snapshot()
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    raw_path = out_dir / "requests.jsonl"
    with raw_path.open("w") as raw_handle:
        for level in args.rates:
            rows = await _level(level, args.count, frames)
            all_rows.extend(rows)
            raw_handle.writelines(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            raw_handle.flush()
            summaries.append(_summarize(level, rows))
            # A completed level is useful evidence even if a later status query or
            # offered-load level fails; never defer durability to finalization.
            (out_dir / "partial_summary.json").write_text(json.dumps({
                "endpoint": ENDPOINT,
                "gcs_address": GCS_ADDRESS,
                "model_revision": REVISION,
                "payload_manifest": "payload_manifest.json",
                "request_rows": "requests.jsonl",
                "count_per_level": args.count,
                "levels": summaries,
                "pre_cluster": pre,
            }, indent=2, sort_keys=True) + "\n")
    post = _cluster_snapshot()

    report = {
        "endpoint": ENDPOINT,
        "gcs_address": GCS_ADDRESS,
        "model_revision": REVISION,
        "payload_manifest": "payload_manifest.json",
        "request_rows": "requests.jsonl",
        "count_per_level": args.count,
        "levels": summaries,
        "pre_cluster": pre,
        "post_cluster": post,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--rates", nargs="+", type=float, default=[1.0, 2.0, 4.0, 6.0, 8.0])
    args = parser.parse_args()
    if args.count < 1 or any(rate <= 0 for rate in args.rates):
        raise SystemExit("count and offered rates must be positive")
    print(json.dumps(asyncio.run(_run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
