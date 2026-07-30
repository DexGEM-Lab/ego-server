"""Scoped single-GPU HaWoR + infiller experiment lifecycle.

The GPU3 production pair is copied only as an immutable code/checkpoint contract.
An experiment owns a fresh Ray head, short temporary root, disjoint sub-32768 ports,
and worker-attested identities for both logical APIs on one separately verified GPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from ego_annotation.serving.benchmark.unidepth_scaling import (
    EXPERIMENT_TEMP_ROOT,
    ExperimentConfigurationError,
    ImmutableApplicationRelease,
    PRODUCTION_GPU_IDS,
    PRODUCTION_PORTS,
    PRODUCTION_TEMP_DIRS,
    ReplicaEndpoint,
    _experiment_ports,
    _replace_env,
)
from ego_annotation.serving.contracts import SCHEMA_VERSION, ServerIdentity
from ego_annotation.serving.hawor import expected_hawor_runtime_config
from ego_annotation.serving.infiller import expected_infiller_runtime_config
from ego_annotation.serving.lifecycle import ClusterLifecycleConfig, hawor_gpu_group


HAWOR_EXPERIMENT_TEMP_ROOT = Path("/tmp/ehw")
_MAX_RAY_TEMP_DIR_BYTES = 32


@dataclass(frozen=True)
class HaworExperimentalReplica:
    experiment_id: str
    gpu_id: int
    lifecycle: ClusterLifecycleConfig
    result_dir: Path
    track_endpoint: ReplicaEndpoint
    infiller_endpoint: ReplicaEndpoint
    track_identity: ServerIdentity
    infiller_identity: ServerIdentity
    track_runtime_config: Mapping[str, object]
    infiller_runtime_config: Mapping[str, object]

    @property
    def replica_id(self) -> str:
        return self.track_endpoint.replica_id

    @property
    def endpoint(self) -> ReplicaEndpoint:
        """Compatibility with shared preflight/cleanup helpers."""
        return self.track_endpoint

    @property
    def expected_server_identity(self) -> ServerIdentity:
        return self.track_identity

    def launch_commands(self) -> tuple[str, str]:
        environment = " ".join(
            f"{name}={value}" for name, value in (("CUDA_VISIBLE_DEVICES", str(self.gpu_id)), *self.lifecycle.env_vars)
        )
        start = f"env -u RAY_ADDRESS {self.lifecycle.startup_command()}"
        driver = self.lifecycle.experimental_driver_command(
            release_root=dict(self.lifecycle.env_vars)["EGO_APPLICATION_RELEASE_ROOT"],
            app_choice="hawor", app_name=f"hawor-exp-{self.experiment_id}-gpu{self.gpu_id}",
        )
        release_cd, driver_body = driver.split(" && ", 1)
        deploy = f"{release_cd} && env -u RAY_ADDRESS {environment} {driver_body}"
        return start, deploy

    def stop_commands(self) -> tuple[str, str]:
        return (
            self.lifecycle.serve_shutdown_command(),
            f"{self.lifecycle.interpreter} -m ego_annotation.serving.benchmark.hawor_experiment "
            f"--scoped-stop --temp-dir {self.lifecycle.temp_dir}",
        )


@dataclass(frozen=True)
class HaworExperimentPlan:
    experiment_id: str
    run_root: Path
    application_release: ImmutableApplicationRelease
    replicas: tuple[HaworExperimentalReplica, ...]

    def assert_isolated(self) -> None:
        if len(self.replicas) != 1:
            raise ExperimentConfigurationError("HaWoR experiment requires exactly one isolated physical GPU")
        replica = self.replicas[0]
        if replica.gpu_id in PRODUCTION_GPU_IDS:
            raise ExperimentConfigurationError(f"GPU{replica.gpu_id} is reserved for production")
        if replica.lifecycle.temp_dir in PRODUCTION_TEMP_DIRS:
            raise ExperimentConfigurationError("HaWoR experiment temp dir overlaps production")
        temp_dir = Path(replica.lifecycle.temp_dir).resolve()
        if temp_dir.parent != (HAWOR_EXPERIMENT_TEMP_ROOT / self.experiment_id).resolve():
            raise ExperimentConfigurationError("HaWoR experiment temp dir must be one exact /tmp/ehw replica directory")
        if len(os.fsencode(str(temp_dir))) > _MAX_RAY_TEMP_DIR_BYTES:
            raise ExperimentConfigurationError("HaWoR experiment temp dir is too long for Ray AF_UNIX sockets")
        replica.lifecycle.assert_gpu_pinned()
        ports = set(replica.lifecycle.ports.all_ports())
        if ports & PRODUCTION_PORTS:
            raise ExperimentConfigurationError("HaWoR experiment ports overlap production")
        if any("runtime/current" in value for _, value in replica.lifecycle.env_vars):
            raise ExperimentConfigurationError("HaWoR experiment must not import mutable runtime/current")
        if {replica.track_identity.release_digest, replica.infiller_identity.release_digest} != {self.application_release.release_digest}:
            raise ExperimentConfigurationError("HaWoR worker identities do not bind the pinned release")


def build_hawor_experiment_plan(
    *, experiment_id: str, gpu_id: int, run_root: str | Path, application_release_path: str | Path,
    source_sha: str, hawor_checkpoint_digest: str, infiller_checkpoint_digest: str,
    gpu_uuid: str | None = None, component_port_base: int = 30000, worker_port_base: int = 30100,
    serve_port: int = 32000, wire_format: str = "envelope",
) -> HaworExperimentPlan:
    if not experiment_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in experiment_id):
        raise ExperimentConfigurationError("experiment_id must use only letters, digits, hyphen, and underscore")
    if gpu_id in PRODUCTION_GPU_IDS or gpu_id < 0:
        raise ExperimentConfigurationError(f"GPU{gpu_id} cannot be a HaWoR experiment target")
    if wire_format not in {"multipart", "envelope"}:
        raise ExperimentConfigurationError("wire_format must be multipart or envelope")
    if not hawor_checkpoint_digest or not infiller_checkpoint_digest:
        raise ExperimentConfigurationError("both HaWoR and infiller checkpoint digests are required")
    release = ImmutableApplicationRelease.pin(application_release_path, expected_source_sha=source_sha)
    production = hawor_gpu_group().lifecycle
    temp_dir = HAWOR_EXPERIMENT_TEMP_ROOT / experiment_id / f"gpu{gpu_id}"
    ports = _experiment_ports(component_port_base, worker_port_base, serve_port)
    immutable_pythonpath = ":".join(
        [str(release.path)] + [part for part in dict(production.env_vars).get("PYTHONPATH", "").split(":") if part and "runtime/current" not in part]
    )
    track_id = f"hawor-exp-{experiment_id}-gpu{gpu_id}"
    infiller_id = f"hawor-infiller-exp-{experiment_id}-gpu{gpu_id}"
    lifecycle = replace(
        production,
        gpu_id=gpu_id,
        temp_dir=str(temp_dir),
        ports=ports,
        env_vars=_replace_env(production.env_vars, {
            "PYTHONPATH": immutable_pythonpath,
            "EGO_HAWOR_GPU": str(gpu_id),
            "EGO_HAWOR_REPLICA_ID": track_id,
            "EGO_HAWOR_INFILLER_REPLICA_ID": infiller_id,
            "EGO_EXPERIMENT_ID": experiment_id,
            "EGO_APPLICATION_RELEASE_SHA": release.release_sha,
            "EGO_APPLICATION_RELEASE_ROOT": str(release.path),
            "EGO_HAWOR_CHECKPOINT_DIGEST": hawor_checkpoint_digest,
            "EGO_HAWOR_INFILLER_CHECKPOINT_DIGEST": infiller_checkpoint_digest,
            "EGO_EXPERIMENT_GCS_ADDRESS": f"127.0.0.1:{ports.gcs_port}",
            "EGO_EXPERIMENT_HTTP_PORT": str(serve_port),
            "EGO_EXPERIMENT_TEMP_DIR": str(temp_dir),
            "EGO_HAWOR_EXPERIMENT_TELEMETRY": "1",
            "EGO_HAWOR_INFILLER_EXPERIMENT_TELEMETRY": "1",
            "EGO_HAWOR_EXPERIMENT_WIRE_FORMAT": wire_format,
            "EGO_HAWOR_INFILLER_EXPERIMENT_WIRE_FORMAT": wire_format,
            "PYTHONDONTWRITEBYTECODE": "1",
        }),
    )
    replica = HaworExperimentalReplica(
        experiment_id=experiment_id, gpu_id=gpu_id, lifecycle=lifecycle,
        result_dir=Path(run_root) / experiment_id / track_id,
        track_endpoint=ReplicaEndpoint(track_id, f"http://127.0.0.1:{serve_port}/hawor.infer_tracks", gpu_id),
        infiller_endpoint=ReplicaEndpoint(infiller_id, f"http://127.0.0.1:{serve_port}/hawor_infiller.fill", gpu_id),
        track_identity=ServerIdentity(
            experiment_id=experiment_id, replica_id=track_id, assigned_gpu=gpu_id, worker_pid=1,
            gcs_address=lifecycle.gcs_address, http_port=serve_port, temp_dir=str(temp_dir),
            model_revision=str(dict(lifecycle.env_vars).get("EGO_HAWOR_REVISION", "hawor-v1")),
            checkpoint_digest=hawor_checkpoint_digest, schema_version=SCHEMA_VERSION,
            release_sha=release.release_sha, release_digest=release.release_digest,
            cuda_uuid=gpu_uuid, module_root=str(release.path),
        ),
        infiller_identity=ServerIdentity(
            experiment_id=experiment_id, replica_id=infiller_id, assigned_gpu=gpu_id, worker_pid=1,
            gcs_address=lifecycle.gcs_address, http_port=serve_port, temp_dir=str(temp_dir),
            model_revision=str(dict(lifecycle.env_vars).get("EGO_HAWOR_INFILLER_REVISION", "hawor-infiller-v1")),
            checkpoint_digest=infiller_checkpoint_digest, schema_version=SCHEMA_VERSION,
            release_sha=release.release_sha, release_digest=release.release_digest,
            cuda_uuid=gpu_uuid, module_root=str(release.path),
        ),
        track_runtime_config=expected_hawor_runtime_config(wire_format=wire_format),
        infiller_runtime_config=expected_infiller_runtime_config(wire_format=wire_format),
    )
    plan = HaworExperimentPlan(experiment_id, Path(run_root), release, (replica,))
    plan.assert_isolated()
    return plan


def _main(argv: Sequence[str] | None = None) -> int:
    """Environment-owned exact-process cleanup for the HaWoR temp-root only."""
    import argparse
    import json
    from ego_annotation.serving.benchmark.unidepth_scaling import _candidate_process_pids, stop_scoped_experiment

    parser = argparse.ArgumentParser(description="Inspect or scoped-stop a HaWoR experiment")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--scoped-stop", action="store_true")
    action.add_argument("--scoped-status", action="store_true")
    parser.add_argument("--temp-dir", required=True)
    args = parser.parse_args(argv)
    resolved = str(Path(args.temp_dir).resolve())
    relative = Path(resolved).relative_to(HAWOR_EXPERIMENT_TEMP_ROOT.resolve())
    if len(relative.parts) != 2 or not relative.parts[1].startswith("gpu") or not relative.parts[1][3:].isdigit():
        raise ExperimentConfigurationError("refusing non-exact HaWoR experiment temp dir")
    if args.scoped_status:
        pids = _candidate_process_pids(resolved)
        print(json.dumps({"temp_dir": resolved, "candidate_pids": pids}))
        return 1 if pids else 0
    pids = stop_scoped_experiment(resolved, experiment_temp_root=HAWOR_EXPERIMENT_TEMP_ROOT)
    print(json.dumps({"temp_dir": resolved, "stopped_pids": pids}))
    return 0


if __name__ == "__main__":  # pragma: no cover - server-only scoped cleanup
    raise SystemExit(_main())
