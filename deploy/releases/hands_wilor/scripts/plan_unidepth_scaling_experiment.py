"""Print an isolated vacant-GPU UniDepth scaling launch/stop contract.

This is a planning command only: it does not SSH, inspect GPUs, start Ray, deploy a
model, or mutate the canonical router.  An authorized dex-a800 operator must run the
printed vacancy check immediately before manually executing each printed launch
command in an isolated experiment window.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from ego_annotation.serving.benchmark.unidepth_scaling import (
    LocalBlockingCommandRunner,
    build_unidepth_scaling_plan,
    execute_scoped_plan,
    production_health_check_commands,
    validate_server_identity,
    vacancy_check_command,
)
from ego_annotation.serving.benchmark.manifest import load_payload_manifest
from ego_annotation.serving.gateway import ModelServiceGateway, RetryPolicy
from ego_annotation.serving.router import ModelApiName, ModelServiceRouter


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan isolated UniDepth one-vs-two replica scaling runs")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated authorized vacant physical GPU ids, e.g. 4 or 4,5")
    parser.add_argument("--run-root", required=True, type=Path, help="fresh server benchmark result root")
    parser.add_argument("--application-release", required=True, type=Path, help="immutable release directory containing RELEASE.json; never runtime/current")
    parser.add_argument("--source-sha", required=True, help="exact source SHA attested by RELEASE.json")
    parser.add_argument("--checkpoint-digest", help="checkpoint digest; derived from --checkpoint when supplied")
    parser.add_argument("--checkpoint", type=Path, help="actual loaded checkpoint file/directory; digest is recomputed from bytes")
    parser.add_argument("--gpu-uuids", help="comma-separated expected physical GPU UUIDs")
    parser.add_argument("--component-port-base", type=int, default=29000)
    parser.add_argument("--worker-port-base", type=int, default=29100)
    parser.add_argument("--serve-port-base", type=int, default=31000)
    parser.add_argument("--experiment-batch-cap", type=int, choices=(8, 16), default=8)
    parser.add_argument("--experiment-batch-wait-ms", type=int, choices=(20, 50), default=20)
    parser.add_argument("--wire-format", choices=("multipart", "envelope"), default="multipart", help="attested default wire format; both formats remain accepted by the endpoint")
    parser.add_argument("--max-concurrent-forwards", type=int, default=1, help="physical forward semaphore treatment; omit only with --current-forward-behavior")
    parser.add_argument("--current-forward-behavior", action="store_true", help="omit the experiment forward semaphore")
    parser.add_argument("--execute", action="store_true", help="blocking scoped start/deploy + typed readiness; never touches production")
    parser.add_argument("--readiness-payload-dir", type=Path, help="explicit payload corpus used for one typed readiness request per replica")
    parser.add_argument("--readiness-timeout-s", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.current_forward_behavior and args.max_concurrent_forwards != 1:
        parser.error("--current-forward-behavior cannot be combined with --max-concurrent-forwards")
    gpu_ids = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    if args.checkpoint is not None:
        from ego_annotation.serving.benchmark.release import checkpoint_digest
        computed_checkpoint_digest = checkpoint_digest(args.checkpoint)
        if args.checkpoint_digest and args.checkpoint_digest != computed_checkpoint_digest:
            parser.error("--checkpoint-digest does not match bytes in --checkpoint")
        args.checkpoint_digest = computed_checkpoint_digest
    if not args.checkpoint_digest:
        parser.error("provide --checkpoint so its digest is derived, or --checkpoint-digest for a precomputed immutable artifact")
    gpu_uuids = tuple(v.strip() for v in args.gpu_uuids.split(",") if v.strip()) if args.gpu_uuids else None
    plan = build_unidepth_scaling_plan(
        experiment_id=args.experiment_id,
        gpu_ids=gpu_ids,
        run_root=args.run_root,
        component_port_base=args.component_port_base,
        worker_port_base=args.worker_port_base,
        serve_port_base=args.serve_port_base,
        application_release_path=args.application_release,
        source_sha=args.source_sha,
        checkpoint_digest=args.checkpoint_digest,
        gpu_uuids=gpu_uuids,
        experiment_batch_cap=args.experiment_batch_cap,
        experiment_batch_wait_ms=args.experiment_batch_wait_ms,
        max_concurrent_forwards=None if args.current_forward_behavior else args.max_concurrent_forwards,
        wire_format=args.wire_format,
    )
    if args.execute:
        if args.readiness_payload_dir is None:
            parser.error("--execute requires --readiness-payload-dir for a typed readiness request")
        readiness_manifest_path = args.readiness_payload_dir / "unidepth.infer.json"
        if not readiness_manifest_path.is_file():
            parser.error(f"typed readiness payload manifest is absent: {readiness_manifest_path}")
        readiness_manifest = load_payload_manifest(readiness_manifest_path, expected_api=ModelApiName.UNIDEPTH_INFER, limit=1)
        if not readiness_manifest.items:
            parser.error("typed readiness payload manifest has no items")

        def typed_probe(replica) -> None:
            async def call() -> None:
                router = ModelServiceRouter.canonical().with_overrides({ModelApiName.UNIDEPTH_INFER: replica.endpoint.url})
                gateway = ModelServiceGateway.with_httpx(router, timeout_s=args.readiness_timeout_s, retry_policy=RetryPolicy(max_attempts=1))
                try:
                    response = await gateway.call(readiness_manifest.items[0].to_gateway_request())
                    validate_server_identity(replica.expected_server_identity, response)
                finally:
                    await gateway.aclose()
            asyncio.run(call())

        result = execute_scoped_plan(plan, command_runner=LocalBlockingCommandRunner(), typed_readiness_probe=typed_probe)
        print(json.dumps({"schema": "ego.unidepth-scaling-execution.v1", "started": result.started_replica_ids,
                          "ready": result.readiness_replica_ids, "rollback": result.rollback_replica_ids}, indent=2))
        return 0
    print(json.dumps({
        "schema": "ego.unidepth-scaling-launch-plan.v1",
        "experiment_id": plan.experiment_id,
        "run_root": str(plan.run_root),
        "application_release": {
            "path": str(plan.application_release.path),
            "release_sha": plan.application_release.release_sha,
            "source_sha": plan.application_release.source_sha,
        },
        "canonical_router_mutated": False,
        "production_health_before_and_after": list(production_health_check_commands()),
        "replicas": [
            {
                "replica_id": replica.replica_id,
                "gpu_id": replica.gpu_id,
                "endpoint": replica.endpoint.url,
                "temp_dir": replica.lifecycle.temp_dir,
                "ports": replica.lifecycle.ports.all_ports(),
                "prelaunch_vacancy_check": vacancy_check_command(replica.gpu_id),
                "launch_commands": list(replica.launch_commands()),
                "stop_commands": list(replica.stop_commands()),
                "result_dir": str(replica.result_dir),
                "expected_server_identity": replica.expected_server_identity.to_wire(),
                "expected_runtime_config": dict(replica.expected_runtime_config),
            }
            for replica in plan.replicas
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
