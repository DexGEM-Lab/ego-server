#!/usr/bin/env python3
"""Long-video, continuous-stream DROID saturation measurement.

Every session consumes the same ordered 720-frame production sequence.  S
independent continuous session tasks offer their next frame immediately after the
prior frame completes; S8/S16/S32 therefore issue aggregate demand far above the
single actor's known CPU-bound rate.  Backpressure is recorded, never hidden: a
rejected frame is re-offered with the same frame/timestamp until admission, so a
session cannot skip a production frame merely because the saturated server said
no.  This is intentionally unlike the flawed short-session wave benchmark.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.ray_serve.benchmark_droid_open_loop import (
    ReplicaEndpoint,
    StickyDroidRouter,
    _load_preserved_payloads,
    _post_typed,
    camera_contract,
    read_payload,
    record_from_call,
)
from ego_annotation.serving.benchmark.measurement import NvmlSampler
from ego_annotation.serving.contracts import (
    DroidCamera,
    DroidCreateSessionRequest,
    DroidCreateSessionResponse,
    DroidFinalizeRequest,
    DroidFinalizeResponse,
    DroidFrameRequest,
    DroidImageShape,
    DroidSessionOptions,
    Ownership,
    ServerIdentity,
    TensorPayload,
)


@dataclass(frozen=True)
class CpuSample:
    monotonic_s: float
    host_busy_pct: float
    replica_cpu_pct: float


class CpuSampler:
    """Sample host busy time and the server-attested replica PID from /proc."""

    def __init__(self, pid: int, interval_s: float = 0.2) -> None:
        self.pid, self.interval_s = pid, interval_s
        self.samples: list[CpuSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._previous: tuple[int, int, int] | None = None

    @staticmethod
    def _ticks() -> tuple[int, int, int]:
        fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        total = sum(int(value) for value in fields)
        idle = int(fields[3]) + (int(fields[4]) if len(fields) > 4 else 0)
        proc = Path(f"/proc/{os.getpid()}/stat")  # overwritten by instance sample
        return total, idle, int(proc.stat().st_mtime_ns)  # sentinel only

    def _sample(self) -> None:
        fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        total = sum(int(value) for value in fields)
        idle = int(fields[3]) + (int(fields[4]) if len(fields) > 4 else 0)
        process_fields = Path(f"/proc/{self.pid}/stat").read_text().split()
        process = int(process_fields[13]) + int(process_fields[14])
        now = time.monotonic()
        if self._previous is not None:
            old_total, old_idle, old_process = self._previous
            delta_total = total - old_total
            if delta_total > 0:
                host_busy = 100.0 * (1.0 - (idle - old_idle) / delta_total)
                # Process CPU is normalized to one logical CPU, as reported by top.
                replica = 100.0 * (process - old_process) / (delta_total / os.cpu_count())
                self.samples.append(CpuSample(now, host_busy, replica))
        self._previous = (total, idle, process)

    def start(self) -> None:
        self._sample()
        def loop() -> None:
            while not self._stop.wait(self.interval_s):
                self._sample()
        self._thread = threading.Thread(target=loop, name="droid-cpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {"count": len(values), "mean": sum(values) / len(values) if values else None,
            "p95": _percentile(values, 0.95), "max": max(values) if values else None}


def _ownership(request_id: str, stage: str, source_id: str, timestamp_s: float | None = None) -> Ownership:
    return Ownership(request_id=request_id, job_id="droid-continuous-long-video", item_id=source_id,
                     stage_id=stage, source_id=source_id, source_timestamp_s=timestamp_s)


def _endpoint(args: argparse.Namespace) -> ReplicaEndpoint:
    identity = ServerIdentity.from_wire(json.loads(args.runtime_identity.read_text(encoding="utf-8")))
    return ReplicaEndpoint(args.endpoint, identity.replica_id, identity.model_revision, identity)


async def _create_all(args: argparse.Namespace, client: Any, router: StickyDroidRouter) -> tuple[list[str], ServerIdentity]:
    async def create(index: int):
        request = DroidCreateSessionRequest(
            ownership=_ownership(f"create-{index}", "droid.create_session", f"b3s1-session-{index}"),
            camera=DroidCamera.from_mapping(camera_contract()), image_shape=DroidImageShape(320, 568),
            options=DroidSessionOptions(buffer=args.session_buffer, filter_thresh=2.4, keyframe_thresh=4.0, warmup=8),
            model_revision=args.model_revision,
        )
        return await router.create_session(client, request)
    calls = await asyncio.gather(*(create(index) for index in range(args.sessions)))
    sessions: list[str] = []
    identities: list[ServerIdentity] = []
    for call in calls:
        if not isinstance(call.response, DroidCreateSessionResponse) or call.http_status != 200 or call.response.session_id is None:
            raise RuntimeError(f"session creation failed: {call.parse_error or call.response}")
        if call.response.server_identity is None:
            raise RuntimeError("create response lacks server identity")
        sessions.append(call.response.session_id)
        identities.append(call.response.server_identity)
    if len({identity.worker_pid for identity in identities}) != 1:
        raise RuntimeError("creation responses did not originate from one stable replica worker")
    return sessions, identities[0]


async def _stream_session(args: argparse.Namespace, client: Any, router: StickyDroidRouter, session_id: str,
                          payloads: list[Any], session_index: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    # DROID permits one ready frame/session. S concurrent session tasks keep the
    # actor's global admission path overloaded; retaining the same index after a
    # rejection preserves the real video timeline while making backpressure visible.
    for frame_index, payload in enumerate(payloads):
        rgb, mask = read_payload(payload)
        reoffer = 0
        reoffer_started = time.monotonic()
        while True:
            request_id = f"s{session_index}-f{frame_index:04d}-a{reoffer:03d}"
            request = DroidFrameRequest(
                ownership=_ownership(request_id, "droid.push_frame", f"b3s1:{payload.source_frame_index}", payload.timestamp_s),
                session_id=session_id, frame_id=f"s{session_index}-frame-{frame_index:04d}",
                source_timestamp_s=payload.timestamp_s,
                rgb=TensorPayload(rgb, (320, 568, 3), "uint8"),
                static_confidence_mask=TensorPayload(mask, (320, 568), "float32"),
                model_revision=args.model_revision,
            )
            call = await router.push_frame(client, request)
            record = record_from_call(call, level=f"S{args.sessions}", request_id=request_id,
                                      session_id=session_id, scheduled_s=None).to_json()
            record["frame_index"] = frame_index
            record["reoffer_attempt"] = reoffer
            records.append(record)
            if record["outcome"] == "completed":
                break
            if record["outcome"] != "rejected":
                # Preserve a server/model/transport anomaly rather than concealing
                # it behind a retry. A backpressure rejection is the sole re-offer.
                return records
            reoffer += 1
            if reoffer > 200 and time.monotonic() - reoffer_started > 30.0:
                # Bound re-offers by TIME (30s) not count. The server may legitimately
                # process the prior frame for 1-2s; a count-based bound fails during
                # late-frame processing when latency grows. Surfaces structural issues
                # only if the server cannot admit for 30 continuous seconds.
                record["outcome"] = "reoffer_timeout_exceeded"
                records.append(record)
                return records
            # Brief yield so other session tasks can dispatch; prevents this
            # task from monopolizing the event loop during backpressure.
            await asyncio.sleep(0.005)
    return records


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.sessions <= 0:
        raise ValueError("sessions must be positive")
    if args.frames <= 0 or args.session_buffer <= 0:
        raise ValueError("frames and session_buffer must be positive")
    if args.frames < args.session_buffer and args.session_buffer < 720:
        # Buffer larger than the video but less than 720 is allowed for VRAM-saturation
        # sweeps where we deliberately under-fill to test scaling.
        pass
    args.run_root.mkdir(parents=True, exist_ok=False)
    (args.run_root / "records").mkdir()
    # The preserved-manifest loader writes a path-independent copy of the
    # semantic corpus manifest under droid/; create it before loading.
    (args.run_root / "droid").mkdir()
    payloads = _load_preserved_payloads(args, args.run_root)
    if len(payloads) != 720:
        raise RuntimeError(f"need exactly 720 preserved production payloads, found {len(payloads)}")
    source_indices = [payload.source_frame_index for payload in payloads]
    timestamps = [payload.timestamp_s for payload in payloads]
    if any(right <= left for left, right in zip(source_indices, source_indices[1:])) or any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise RuntimeError("payload manifest is not an increasing consecutive production timeline")
    endpoint = _endpoint(args)
    import httpx
    router = StickyDroidRouter((endpoint,))
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout_s), limits=httpx.Limits(max_connections=args.sessions * 2)) as client:
        sessions, worker_identity = await _create_all(args, client, router)
        gpu = NvmlSampler(gpu_ids=[worker_identity.assigned_gpu], gpu_uuids={worker_identity.assigned_gpu: worker_identity.cuda_uuid or ""},
                          experiment_id=worker_identity.experiment_id, release_digest=worker_identity.release_digest or "", interval_s=0.2)
        cpu = CpuSampler(worker_identity.worker_pid, interval_s=0.2)
        # The target quantity begins at the first push and ends at the last push
        # completion, excluding create and terminal BA/finalize by definition.
        started = time.monotonic()
        gpu.start(); cpu.start()
        streams = await asyncio.gather(*(_stream_session(args, client, router, session, payloads, index)
                                         for index, session in enumerate(sessions)))
        ended = time.monotonic()
        cpu.stop(); gpu.stop()
        finals = await asyncio.gather(*(
            router.finalize(client, DroidFinalizeRequest(
                ownership=_ownership(f"finalize-{index}", "droid.finalize", f"b3s1-session-{index}"),
                session_id=session, model_revision=args.model_revision,
            )) for index, session in enumerate(sessions)
        ))
    all_pushes = [record for stream in streams for record in stream]
    for index, stream in enumerate(streams):
        (args.run_root / "records" / f"session_{index:02d}.json").write_text(json.dumps(stream, indent=2) + "\n")
    terminal = []
    for call in finals:
        valid = isinstance(call.response, DroidFinalizeResponse) and call.http_status == 200 and call.response.camera_state is not None and math.isfinite(call.response.camera_state.uncertainty.finite_pose_ratio) and call.response.camera_state.uncertainty.finite_pose_ratio == 1.0
        terminal.append(valid)
    nvml_path = gpu.write(args.run_root / "gpu_samples.json")
    gpu_util = [sample.utilization_gpu_pct for sample in gpu.samples if started <= sample.timestamp_s <= ended]
    vram = [sample.memory_used_bytes for sample in gpu.samples if started <= sample.timestamp_s <= ended]
    complete = sum(1 for record in all_pushes if record["outcome"] == "completed")
    statuses = Counter(str(record["http_status"]) for record in all_pushes if record["outcome"] != "completed")
    error_codes = Counter(str(record["error_code"]) for record in all_pushes if record["outcome"] != "completed")
    report = {
        "schema": "ego.droid-real-workload-continuous.v1",
        "workload": {"sessions": args.sessions, "frames_per_session": args.frames, "buffer": args.session_buffer, "cpu_offload": args.cpu_offload,
                     "sequence": "each session frame 0..719, closed-loop next-frame-on-completion; no waves/retries", "payload_manifest": str(args.preserved_payload_manifest)},
        "identity": worker_identity.to_wire(),
        "measurement": {"first_push_to_last_completion_s": ended - started, "offered_push_attempts": len(all_pushes),
                        "offered_fps": len(all_pushes) / (ended - started), "completed_frames": complete,
                        "expected_frames": args.sessions * args.frames, "sustained_fps": complete / (ended - started),
                        "all_sessions_completed_720": all(sum(item["outcome"] == "completed" for item in stream) == args.frames and all(item["frame_index"] == index for index, item in enumerate([item for item in stream if item["outcome"] == "completed"])) for stream in streams),
                        "serve_noncomplete_by_status": dict(statuses), "serve_noncomplete_by_error_code": dict(error_codes)},
        "saturation_interpretation": "S independent continuous streams are intentionally offered without client pacing. A replica CPU mean near 100% and/or explicit noncompletion counts show actor-side admission is the limiter; offered_fps must exceed sustained_fps to establish rejected demand." ,
        "terminal_validity": {"all_sessions_finite": all(terminal), "finite_sessions": sum(terminal), "sessions": args.sessions},
        "gpu": {"utilization_pct": _summary(gpu_util), "vram_used_bytes": {"peak": max(vram) if vram else None}, "samples": str(nvml_path)},
        "cpu": {"host_busy_pct": _summary([sample.host_busy_pct for sample in cpu.samples]), "replica_cpu_pct": _summary([sample.replica_cpu_pct for sample in cpu.samples]), "samples": [asdict(sample) for sample in cpu.samples]},
        "artifacts": {"per_session_push_records": str(args.run_root / "records"), "gpu_samples": str(nvml_path)},
    }
    (args.run_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DROID 720-frame continuous real-workload benchmark")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--runtime-identity", required=True, type=Path, help="server-derived identity JSON from typed readiness")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--preserved-payload-manifest", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--sessions", type=int, required=True)
    parser.add_argument("--frames", type=int, default=720)
    parser.add_argument("--session-buffer", type=int, default=1024)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True,
                        help="attest whether the server uses CPU offload (default true)")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    # The preserved-payload loader is shared with the open-loop harness, where
    # the requested count is named payload_count. Here the real-workload
    # contract exposes that same count as frames; bind the two explicitly so
    # the client can load its 720-frame timeline before opening a session.
    args.payload_count = args.frames
    return args


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run(parse_args())), indent=2))
