#!/usr/bin/env python3
"""Run one independent DROID open-loop client process per live replica.

DROID sessions are stateful, so each child receives exactly one verified endpoint
and keeps all of its 32 sessions there.  The parent has no HTTP client: it only
writes one-replica identity views, launches the children once, and combines their
terminal/throughput/GPU evidence.  This prevents a shared httpx stack from
becoming the apparent horizontal-scaling ceiling.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_REPLICA_COUNT = 3
EXPECTED_SESSIONS = 32
EXPECTED_WAVES = 8
EXPECTED_BUFFER = 256
EXPECTED_WAVE_RATE = 4.0
EXPECTED_SINGLE_REPLICA_FPS = 46.08


class ShardRunInvalid(ValueError):
    """The aggregate cannot support the requested horizontal-scaling claim."""


@dataclass(frozen=True)
class ChildSpec:
    replica_id: str
    endpoint: str
    gpu_id: int
    gcs_address: str
    identity_path: Path
    run_root: Path
    stdout_path: Path
    stderr_path: Path
    cpu_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShardRunInvalid(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ShardRunInvalid(f"JSON root must be an object: {path}")
    return raw


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_child_specs(plan_path: Path, output_root: Path) -> tuple[dict[str, Any], tuple[ChildSpec, ...]]:
    """Validate the single hr3 launch plan and make one immutable view per child."""
    plan = _read_json(plan_path)
    replicas = plan.get("replicas")
    if plan.get("schema") != "ego.droid-scaling-launch-plan.v1" or not isinstance(replicas, list):
        raise ShardRunInvalid("runtime identities must be a DROID scaling launch plan")
    if len(replicas) != EXPECTED_REPLICA_COUNT:
        raise ShardRunInvalid(f"hr3 requires exactly {EXPECTED_REPLICA_COUNT} replicas")
    launch = plan.get("launch_configuration")
    if not isinstance(launch, Mapping):
        raise ShardRunInvalid("launch plan lacks launch_configuration")
    expected_launch = {"cpu_offload": True, "max_sessions": EXPECTED_SESSIONS, "max_concurrent_ba": 1}
    mismatches = {name: (launch.get(name), expected) for name, expected in expected_launch.items() if launch.get(name) != expected}
    if mismatches:
        raise ShardRunInvalid(f"hr3 launch configuration differs from S32 lazy serialized contract: {mismatches}")
    seen_replicas: set[str] = set()
    seen_gpus: set[int] = set()
    seen_endpoints: set[str] = set()
    child_specs: list[ChildSpec] = []
    for replica in replicas:
        if not isinstance(replica, Mapping):
            raise ShardRunInvalid("launch-plan replica must be an object")
        identity = replica.get("expected_server_identity")
        if not isinstance(identity, Mapping):
            raise ShardRunInvalid("launch-plan replica lacks expected_server_identity")
        replica_id, endpoint = replica.get("replica_id"), replica.get("endpoint")
        gpu_id = replica.get("gpu_id")
        if not isinstance(replica_id, str) or not isinstance(endpoint, str) or not isinstance(gpu_id, int):
            raise ShardRunInvalid("launch-plan replica identity is malformed")
        if identity.get("replica_id") != replica_id or identity.get("assigned_gpu") != gpu_id:
            raise ShardRunInvalid(f"launch-plan identity disagrees for {replica_id}")
        gcs_address = identity.get("gcs_address")
        if not isinstance(gcs_address, str) or not gcs_address:
            raise ShardRunInvalid(f"launch-plan replica lacks GCS identity: {replica_id}")
        if replica_id in seen_replicas or gpu_id in seen_gpus or endpoint in seen_endpoints:
            raise ShardRunInvalid("hr3 requires distinct replica IDs, physical GPUs, and endpoints")
        seen_replicas.add(replica_id)
        seen_gpus.add(gpu_id)
        seen_endpoints.add(endpoint)
        child_root = output_root / "children" / replica_id
        child_plan = {**plan, "replicas": [dict(replica)]}
        identity_path = child_root / "runtime_identity.json"
        _write_json(identity_path, child_plan)
        child_specs.append(ChildSpec(
            replica_id=replica_id, endpoint=endpoint, gpu_id=gpu_id, gcs_address=gcs_address,
            identity_path=identity_path, run_root=child_root / "run",
            stdout_path=child_root / "stdout.log", stderr_path=child_root / "stderr.log",
            cpu_path=child_root / "cpu_seconds.txt",
        ))
    return plan, tuple(child_specs)


def _push_row(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = summary.get("rows")
    if not isinstance(rows, list):
        raise ShardRunInvalid("child summary lacks rows")
    found = [row for row in rows if isinstance(row, Mapping) and row.get("operation") == "push_frame"]
    if len(found) != 1:
        raise ShardRunInvalid("each process-sharded child must have exactly one push level")
    return found[0]


def _finalize_row(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = summary.get("rows")
    found = [row for row in rows if isinstance(row, Mapping) and row.get("operation") == "finalize"] if isinstance(rows, list) else []
    if len(found) != 1:
        raise ShardRunInvalid("each process-sharded child must have exactly one finalize level")
    return found[0]


def _require_int(row: Mapping[str, Any], name: str) -> int:
    value = row.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ShardRunInvalid(f"summary {name} must be an integer")
    return value


def _require_float(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ShardRunInvalid(f"summary {name} must be finite")
    return float(value)


def _finite_terminal(finalize: Mapping[str, Any]) -> bool:
    if _require_int(finalize, "completed_count") != EXPECTED_SESSIONS:
        return False
    if _require_int(finalize, "semantic_valid_count") != EXPECTED_SESSIONS:
        return False
    ratio = finalize.get("finite_pose_ratio")
    return (
        isinstance(ratio, Mapping)
        and _require_int(ratio, "count") == EXPECTED_SESSIONS
        and _require_float(ratio, "min") == 1.0
        and _require_float(ratio, "max") == 1.0
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ShardRunInvalid("cannot calculate percentile from no samples")
    index = (len(ordered) - 1) * percentile
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _gpu_summary(run_manifest: Mapping[str, Any], expected_gpu: int) -> dict[str, Any]:
    nvml = run_manifest.get("nvml")
    if not isinstance(nvml, Mapping) or not isinstance(nvml.get("path"), str):
        raise ShardRunInvalid("child run manifest lacks GPU telemetry path")
    samples = _read_json(Path(nvml["path"])).get("samples")
    if not isinstance(samples, list):
        raise ShardRunInvalid("child GPU telemetry lacks samples")
    selected = [sample for sample in samples if isinstance(sample, Mapping) and sample.get("gpu_id") == expected_gpu]
    if len(selected) < 2:
        raise ShardRunInvalid(f"GPU{expected_gpu} has insufficient samples")
    utilization = [_require_float(sample, "utilization_gpu_pct") for sample in selected]
    memory = [_require_int(sample, "memory_used_bytes") for sample in selected]
    return {
        "gpu_id": expected_gpu,
        "sample_count": len(selected),
        "utilization_pct": {"mean": sum(utilization) / len(utilization), "p95": _percentile(utilization, 0.95), "max": max(utilization)},
        "vram_used_bytes": {"max": max(memory), "final": memory[-1]},
        "raw_samples": str(nvml["path"]),
    }


def aggregate(plan: Mapping[str, Any], children: Sequence[ChildSpec], *, cpu_seconds: Mapping[str, float], wall_seconds: Mapping[str, float]) -> dict[str, Any]:
    """Combine completed work over the shared actual offer interval, never rates alone."""
    per_replica: dict[str, Any] = {}
    first_submits: list[float] = []
    final_submits: list[float] = []
    completed_total = 0
    for child in children:
        summary_path = child.run_root / "droid" / "summary.json"
        manifest_path = child.run_root / "droid" / "run_manifest.json"
        summary, manifest = _read_json(summary_path), _read_json(manifest_path)
        expected_plan_digest = plan.get("application_release", {}).get("release_digest") if isinstance(plan.get("application_release"), Mapping) else None
        if summary.get("release_digest") != expected_plan_digest:
            raise ShardRunInvalid(f"child {child.replica_id} release identity differs from launch plan")
        push, finalize = _push_row(summary), _finalize_row(summary)
        sticky = push.get("sticky_replicas")
        expected_assignment = {child.replica_id: EXPECTED_SESSIONS}
        # ``sticky_replicas`` counts all submitted push requests (8 waves × 32
        # sessions); ownership is established by its singleton key, while the
        # planned assignment records the distinct session count.
        if (
            not isinstance(sticky, Mapping)
            or set(sticky) != {child.replica_id}
            or push.get("planned_session_assignment") != expected_assignment
        ):
            raise ShardRunInvalid(f"child {child.replica_id} violated one-replica sticky session ownership")
        offer = push.get("per_replica_actual_offer")
        if not isinstance(offer, Mapping) or set(offer) != {child.replica_id}:
            raise ShardRunInvalid(f"child {child.replica_id} lacks its independent actual-offer evidence")
        clock = offer[child.replica_id]
        if not isinstance(clock, Mapping):
            raise ShardRunInvalid("per-replica offer evidence must be an object")
        first, final = _require_float(clock, "first_submit_s"), _require_float(clock, "final_submit_s")
        if final <= first:
            raise ShardRunInvalid("child actual offer window is not positive")
        completed = _require_int(push, "completed_count")
        completed_total += completed
        first_submits.append(first)
        final_submits.append(final)
        cpu_s, wall_s = cpu_seconds[child.replica_id], wall_seconds[child.replica_id]
        per_replica[child.replica_id] = {
            "gpu_id": child.gpu_id,
            "endpoint": child.endpoint,
            "completed_pushes": completed,
            "completed_fps_own_offer_window": _require_float(push, "completed_rate_per_s"),
            "offer_window_s": _require_float(clock, "actual_offer_window_s"),
            "terminal_all_finite": _finite_terminal(finalize),
            "terminal_finite_pose_ratio": finalize.get("finite_pose_ratio"),
            "gpu": _gpu_summary(manifest, child.gpu_id),
            "client": {
                "pid_cpu_seconds": cpu_s,
                "wall_seconds": wall_s,
                "one_core_cpu_fraction": cpu_s / wall_s if wall_s > 0 else None,
            },
            "summary": str(child.run_root / "droid" / "summary.json"),
        }
    shared_window = max(final_submits) - min(first_submits)
    if shared_window <= 0:
        raise ShardRunInvalid("aggregate offer window is not positive")
    aggregate_fps = completed_total / shared_window
    expected = EXPECTED_REPLICA_COUNT * EXPECTED_SINGLE_REPLICA_FPS
    return {
        "schema": "ego.droid-client-sharded-aggregate.v1",
        "experiment_id": plan.get("experiment_id"),
        "process_sharding": {
            "child_processes": len(children),
            "httpx_stacks": len(children),
            "endpoint_to_child": {child.endpoint: child.replica_id for child in children},
            "shared_httpx_client": False,
        },
        "measurement": {
            "sessions_per_replica": EXPECTED_SESSIONS,
            "waves": EXPECTED_WAVES,
            "session_buffer": EXPECTED_BUFFER,
            "wave_rate_per_s": EXPECTED_WAVE_RATE,
            "completed_pushes": completed_total,
            "aggregate_actual_offer_window_s": shared_window,
            "aggregate_completed_fps": aggregate_fps,
            "expected_linear_fps": expected,
            "scaling_fraction_of_expected": aggregate_fps / expected,
            "all_terminals_finite": all(value["terminal_all_finite"] for value in per_replica.values()),
        },
        "per_replica": per_replica,
        "client_saturation_evidence": {
            "interpretation": "Each endpoint has a distinct OS process and httpx stack; one_core_cpu_fraction near 1.0 flags a remaining per-client CPU ceiling.",
            "one_core_cpu_fractions": {replica_id: value["client"]["one_core_cpu_fraction"] for replica_id, value in per_replica.items()},
        },
    }


def _child_command(args: argparse.Namespace, child: ChildSpec) -> list[str]:
    benchmark = Path(args.benchmark_script).resolve()
    return [
        args.python, str(benchmark),
        "--endpoint", child.endpoint,
        "--replica-ids", child.replica_id,
        "--runtime-identities", str(child.identity_path),
        "--gcs-address", child.gcs_address,
        "--run-root", str(child.run_root),
        "--preserved-payload-manifest", str(Path(args.preserved_payload_manifest).resolve()),
        "--payload-count", str(args.payload_count),
        "--waves", str(args.waves),
        "--session-buffer", str(args.session_buffer),
        "--sessions", str(args.sessions),
        "--wave-rates", str(args.wave_rate),
        "--start-delay-s", str(args.start_delay_s),
        "--timeout-s", str(args.timeout_s),
        "--corpus-digest", args.corpus_digest,
        "--measurement-interval-id", args.measurement_interval_id,
    ]


def _resource_wrapped_child_command(args: argparse.Namespace, child: ChildSpec, wrapper: str) -> list[str]:
    """Run the benchmark script in the accounting process, not a nested Python ELF."""
    command = _child_command(args, child)
    if not command or command[0] != args.python:
        raise ShardRunInvalid("child command must begin with the configured Python interpreter")
    return [args.python, "-c", wrapper, *command[1:]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DROID one-process-per-replica client sharding benchmark")
    parser.add_argument("--runtime-identities", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--preserved-payload-manifest", required=True, type=Path)
    parser.add_argument("--corpus-digest", required=True)
    parser.add_argument("--measurement-interval-id", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--benchmark-script", default=str(Path(__file__).parents[1] / "benchmarks/ray_serve/benchmark_droid_open_loop.py"))
    parser.add_argument("--payload-count", type=int, default=512)
    parser.add_argument("--waves", type=int, default=EXPECTED_WAVES)
    parser.add_argument("--session-buffer", type=int, default=EXPECTED_BUFFER)
    parser.add_argument("--sessions", type=int, default=EXPECTED_SESSIONS)
    parser.add_argument("--wave-rate", type=float, default=EXPECTED_WAVE_RATE)
    parser.add_argument("--start-delay-s", type=float, default=0.25)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (args.payload_count, args.waves, args.session_buffer, args.sessions, args.wave_rate) != (
        512, EXPECTED_WAVES, EXPECTED_BUFFER, EXPECTED_SESSIONS, EXPECTED_WAVE_RATE,
    ):
        raise SystemExit("hr3 is fixed to the proven 512-payload/S32/B256/waves=8/rate=4 treatment")
    if args.run_root.exists():
        raise SystemExit(f"refusing to mix evidence into existing run root: {args.run_root}")
    if not args.preserved_payload_manifest.is_file():
        raise SystemExit("preserved DROID payload manifest is absent")
    plan, children = build_child_specs(args.runtime_identities, args.run_root)
    if plan.get("corpus_digest") != args.corpus_digest or plan.get("measurement_interval_id") != args.measurement_interval_id:
        raise SystemExit("sharded client corpus/measurement identity differs from immutable launch plan")
    application = plan.get("application_release")
    if not isinstance(application, Mapping) or not isinstance(application.get("path"), str):
        raise SystemExit("launch plan lacks application release path")
    release_root = Path(application["path"])
    environment = {**os.environ, "PYTHONPATH": str(release_root), "PYTHONDONTWRITEBYTECODE": "1"}
    child_wrapper = """import os
import resource
import runpy
import sys
cpu_path = os.environ['EGO_DROID_SHARD_CPU_PATH']
sys.argv = sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
finally:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    with open(cpu_path, 'w', encoding='utf-8') as handle:
        handle.write(f'{usage.ru_utime} {usage.ru_stime}\\n')
"""
    started: dict[str, float] = {}
    processes: dict[str, subprocess.Popen[bytes]] = {}
    handles: list[Any] = []
    try:
        for child in children:
            child.stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout = child.stdout_path.open("wb")
            stderr = child.stderr_path.open("wb")
            handles.extend((stdout, stderr))
            started[child.replica_id] = time.monotonic()
            child_environment = {**environment, "EGO_DROID_SHARD_CPU_PATH": str(child.cpu_path)}
            processes[child.replica_id] = subprocess.Popen(
                _resource_wrapped_child_command(args, child, child_wrapper),
                cwd=release_root, env=child_environment, stdout=stdout, stderr=stderr,
            )
        exit_codes = {replica_id: process.wait() for replica_id, process in processes.items()}
    finally:
        for handle in handles:
            handle.close()
    finished = {replica_id: time.monotonic() for replica_id in processes}
    wall_seconds = {replica_id: finished[replica_id] - started[replica_id] for replica_id in processes}
    cpu_seconds: dict[str, float] = {}
    for child in children:
        try:
            user_s, system_s = child.cpu_path.read_text(encoding="utf-8").strip().split()
            cpu_seconds[child.replica_id] = float(user_s) + float(system_s)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"one-attempt shard CPU evidence unavailable for {child.replica_id}: {exc}") from exc
    _write_json(args.run_root / "processes.json", {
        "schema": "ego.droid-client-shard-processes.v1",
        "children": [{"replica_id": child.replica_id, "pid": processes[child.replica_id].pid,
                      "command": _child_command(args, child), "exit_code": exit_codes[child.replica_id],
                      "wall_seconds": wall_seconds[child.replica_id], "cpu_seconds": cpu_seconds[child.replica_id],
                      "cpu_evidence": str(child.cpu_path)} for child in children],
        "note": "One Python child process owns one httpx stack and one endpoint; resource.getrusage CPU seconds exclude the parent.",
    })
    if any(exit_codes.values()):
        raise SystemExit(f"one-attempt shard run failed; raw child logs retained: {exit_codes}")
    report = aggregate(plan, children, cpu_seconds=cpu_seconds, wall_seconds=wall_seconds)
    _write_json(args.run_root / "aggregate_summary.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
