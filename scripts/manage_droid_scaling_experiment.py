"""Plan, execute, or stop isolated DROID scaling replicas without touching production."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from ego_annotation.serving.benchmark.droid_scaling import (
    LocalBlockingCommandRunner,
    build_droid_scaling_plan,
    droid_preflight_check_command,
    execute_droid_scoped_plan,
    stop_droid_scoped_experiment,
    typed_readiness_sequence,
)
from ego_annotation.serving.benchmark.release import checkpoint_digest
from ego_annotation.serving.contracts import (
    DroidCamera,
    DroidCreateSessionRequest,
    DroidFrameRequest,
    DroidImageShape,
    DroidSessionOptions,
    Ownership,
    ImageSize,
    PixelTransform,
    TensorPayload,
)
from ego_annotation.serving.lifecycle import droid_gpu_group


def _add_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--gpus", required=True, help="authorized vacant GPU ids, initially 7; later 4,5 after cleanup")
    parser.add_argument("--gpu-uuids", required=True, help="fresh NVML UUIDs aligned one-for-one with --gpus")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--application-release", required=True, type=Path)
    parser.add_argument("--droid-source-release", required=True, type=Path,
                        help="verified digest-named recovered DROID source release")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path, help="exact resident DROID checkpoint; digest is derived from bytes")
    parser.add_argument("--corpus-digest", required=True, help="SHA256 of the immutable benchmark payload manifest")
    parser.add_argument("--measurement-interval-id", required=True,
                        help="label shared only by directly comparable offered-load treatments")
    parser.add_argument(
        "--cpu-offload", action="store_true",
        help="place DROID DepthVideo storage on CPU; default is GPU-resident",
    )
    parser.add_argument(
        "--max-sessions", type=int, default=16,
        help="explicit DROID actor admission limit, attested in the immutable launch plan",
    )
    parser.add_argument(
        "--max-concurrent-ba", type=int, default=1,
        help="GPU-resident-only concurrent BA bound; offload ignores it (serialized). Attested in the launch plan.",
    )
    parser.add_argument(
        "--ipc-handle-file", type=str, default=None,
        help="Path to shared CUDA IPC handle file. First replica creates it; subsequent replicas reconstruct from it (zero-copy weight sharing).",
    )
    parser.add_argument("--component-port-base", type=int, default=30000)
    parser.add_argument("--worker-port-base", type=int, default=30100)
    parser.add_argument("--serve-port-base", type=int, default=32000)
    parser.add_argument("--component-port-bases", help="comma-separated explicit component-port bases aligned with --gpus")
    parser.add_argument("--worker-port-bases", help="comma-separated explicit worker-port bases aligned with --gpus")
    parser.add_argument("--serve-port-bases", help="comma-separated explicit HTTP-port bases aligned with --gpus")
    parser.add_argument("--replica-labels", help="comma-separated unique replica/temp labels aligned with --gpus; required when a GPU repeats")


def _optional_port_bases(value: str | None) -> tuple[int, ...] | None:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip()) if value else None


def _optional_labels(value: str | None) -> tuple[str, ...] | None:
    return tuple(item.strip() for item in value.split(",") if item.strip()) if value else None


def _plan(args: argparse.Namespace):
    production_checkpoint = Path(dict(droid_gpu_group().lifecycle.env_vars)["EGO_DROID_WEIGHTS"])
    if args.checkpoint != production_checkpoint:
        raise ValueError(f"DROID experiment checkpoint must be exact production checkpoint {production_checkpoint}")
    gpu_ids = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    gpu_uuids = tuple(value.strip() for value in args.gpu_uuids.split(",") if value.strip())
    return build_droid_scaling_plan(
        experiment_id=args.experiment_id,
        gpu_ids=gpu_ids,
        gpu_uuids=gpu_uuids,
        run_root=args.run_root,
        application_release_path=args.application_release,
        droid_source_release_path=args.droid_source_release,
        source_sha=args.source_sha,
        checkpoint_digest=checkpoint_digest(args.checkpoint),
        corpus_digest=args.corpus_digest,
        measurement_interval_id=args.measurement_interval_id,
        cpu_offload=args.cpu_offload,
        max_sessions=args.max_sessions,
        max_concurrent_ba=args.max_concurrent_ba,
        ipc_handle_file=args.ipc_handle_file,
        component_port_base=args.component_port_base,
        worker_port_base=args.worker_port_base,
        serve_port_base=args.serve_port_base,
        component_port_bases=_optional_port_bases(args.component_port_bases),
        worker_port_bases=_optional_port_bases(args.worker_port_bases),
        serve_port_bases=_optional_port_bases(args.serve_port_bases),
        replica_labels=_optional_labels(args.replica_labels),
    )


def _plan_wire(plan) -> dict[str, object]:
    return {
        "schema": "ego.droid-scaling-launch-plan.v1",
        "experiment_id": plan.experiment_id,
        "application_release": {
            "path": str(plan.application_release.path),
            "release_digest": plan.application_release.release_digest,
            "source_sha": plan.application_release.source_sha,
        },
        "canonical_router_mutated": False,
        "droid_source_release": {
            "path": str(plan.droid_source_release.path),
            "droid_slam_root": str(plan.droid_source_release.source_root),
            "source_digest": plan.droid_source_release.source_digest,
            "amendment_id": plan.droid_source_release.amendment_id,
            "core_group_digest": plan.droid_source_release.core_group_digest,
        },
        "corpus_digest": plan.corpus_digest,
        "measurement_interval_id": plan.measurement_interval_id,
        "launch_configuration": plan.launch_configuration,
        "launch_configuration_digest": plan.launch_configuration_digest,
        "preflight_mode": "isolation-only-no-production-interaction",
        "replicas": [
            {
                "replica_id": replica.replica_id,
                "gpu_id": replica.gpu_id,
                "endpoint": replica.endpoint.base_url,
                "temp_dir": replica.lifecycle.temp_dir,
                "ports": list(replica.lifecycle.ports.all_ports()),
                "preflight": droid_preflight_check_command(replica),
                "launch_commands": list(replica.launch_commands()),
                "stop_commands": list(replica.stop_commands()),
                "expected_server_identity": replica.expected_server_identity.to_wire(),
            }
            for replica in plan.replicas
        ],
    }


def _readiness_requests(args: argparse.Namespace, replica):
    rgb = args.readiness_rgb.read_bytes()
    expected_rgb_bytes = args.readiness_height * args.readiness_width * 3
    if len(rgb) != expected_rgb_bytes:
        raise ValueError(f"readiness RGB bytes {len(rgb)} != expected {expected_rgb_bytes}")
    mask = args.readiness_mask.read_bytes() if args.readiness_mask else None
    if mask is not None and len(mask) != args.readiness_height * args.readiness_width * 4:
        raise ValueError("readiness float32 mask byte length does not match HxW")
    base = f"readiness-{replica.replica_id}"
    create = DroidCreateSessionRequest(
        ownership=Ownership(base + "-create", base, base, "droid.create_session", base),
        camera=DroidCamera(
            intrinsics=(args.fx, args.fy, args.cx, args.cy),
            K_px=None,
            source_size=ImageSize(args.readiness_width, args.readiness_height),
            pixel_transform=PixelTransform.identity(),
        ),
        image_shape=DroidImageShape(args.readiness_height, args.readiness_width),
        options=DroidSessionOptions(buffer=8, warmup=2),
        model_revision=replica.expected_server_identity.model_revision,
    )
    frame = DroidFrameRequest(
        ownership=Ownership(base + "-push", base, base, "droid.push_frame", base, source_timestamp_s=0.0),
        session_id="assigned-after-create",
        frame_id=base + "-frame",
        source_timestamp_s=0.0,
        rgb=TensorPayload(rgb, (args.readiness_height, args.readiness_width, 3), "uint8"),
        static_confidence_mask=(
            TensorPayload(mask, (args.readiness_height, args.readiness_width), "float32") if mask is not None else None
        ),
        model_revision=replica.expected_server_identity.model_revision,
    )
    return create, frame


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    _add_plan_args(plan_parser)
    execute_parser = sub.add_parser("execute")
    _add_plan_args(execute_parser)
    execute_parser.add_argument("--readiness-rgb", required=True, type=Path)
    execute_parser.add_argument("--readiness-mask", type=Path)
    execute_parser.add_argument("--readiness-height", type=int, default=320)
    execute_parser.add_argument("--readiness-width", type=int, default=568)
    execute_parser.add_argument("--fx", type=float, required=True)
    execute_parser.add_argument("--fy", type=float, required=True)
    execute_parser.add_argument("--cx", type=float, required=True)
    execute_parser.add_argument("--cy", type=float, required=True)
    execute_parser.add_argument("--readiness-timeout-s", type=float, default=120.0)
    execute_parser.add_argument(
        "--failure-evidence-dir", type=Path,
        help="copy named Ray failure logs here before exact-scope rollback",
    )
    stop_parser = sub.add_parser("stop")
    stop_parser.add_argument("--temp-dir", required=True, action="append")
    args = parser.parse_args(argv)

    if args.command == "stop":
        results = [{"temp_dir": value, "stopped_pids": stop_droid_scoped_experiment(value)} for value in args.temp_dir]
        print(json.dumps({"schema": "ego.droid-scaling-stop.v1", "results": results}, indent=2))
        return 0

    plan = _plan(args)
    if args.command == "plan":
        print(json.dumps(_plan_wire(plan), indent=2))
        return 0

    def readiness(replica) -> None:
        async def run() -> None:
            import httpx

            create, frame = _readiness_requests(args, replica)
            async with httpx.AsyncClient(timeout=httpx.Timeout(args.readiness_timeout_s)) as client:
                await typed_readiness_sequence(
                    client=client,
                    endpoint=replica.endpoint,
                    expected_identity=replica.expected_server_identity,
                    create_request=create,
                    frame_request=frame,
                )
        asyncio.run(run())

    result = execute_droid_scoped_plan(
        plan,
        command_runner=LocalBlockingCommandRunner(),
        typed_readiness_probe=readiness,
        failure_evidence_dir=args.failure_evidence_dir,
    )
    print(json.dumps({
        "schema": "ego.droid-scaling-execution.v1",
        "started": result.started_replica_ids,
        "ready": result.readiness_replica_ids,
        "rollback": result.rollback_replica_ids,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
