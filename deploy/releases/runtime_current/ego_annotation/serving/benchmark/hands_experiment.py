"""Scoped single-GPU Hands/SAM2 + WiLoR experiment lifecycle.

The production GPU1 pair is an immutable model/runtime contract only.  An experiment
owns a fresh pinned-release Ray head, one vacant physical GPU, a short private
``/tmp/ehn`` tree, and two worker-attested logical identities on one Serve proxy.
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
from ego_annotation.serving.benchmark.unidepth_scaling import _canonical_gpu_uuid
from ego_annotation.serving.hands import build_hands_model_config
from ego_annotation.serving.lifecycle import ClusterLifecycleConfig, hands_wilor_gpu_group
from ego_annotation.serving.wilor import build_wilor_model_config


HANDS_EXPERIMENT_TEMP_ROOT = Path("/tmp/ehn")
_MAX_RAY_TEMP_DIR_BYTES = 32


def expected_hands_runtime_config(*, wire_format: str) -> Mapping[str, object]:
    """The exact runtime contract asserted by Hands typed readiness."""
    return build_hands_model_config(
        detector_checkpoint="attested-at-worker", sam2_checkpoint="attested-at-worker",
        sam2_config="attested-at-worker", model_revision="hands-yolo-sam2.1-hiera-l",
        wire_format=wire_format,
    ).runtime_config_wire()


def expected_wilor_runtime_config(*, wire_format: str) -> Mapping[str, object]:
    """The exact runtime contract asserted by WiLoR typed readiness."""
    return build_wilor_model_config(
        checkpoint="attested-at-worker", config_path="attested-at-worker", model_revision="wilor-final-v1",
        wire_format=wire_format,
    ).runtime_config_wire()


def validate_hands_typed_readiness(expected: ServerIdentity, expected_runtime_config: Mapping[str, object], result: object) -> ServerIdentity:
    """Reject a successful Hands/WiLoR result unless its worker attests this plan.

    The caller supplies the parsed typed result, never a client-owned endpoint label.
    Both logical APIs must expose the same invariant family: trace identity, exact
    release and checkpoint facts derived by the worker, and the selected wire mode.
    """
    actual = getattr(result, "server_identity", None)
    trace = getattr(result, "trace", None)
    diagnostics = getattr(result, "batch_diagnostics", None)
    if not isinstance(actual, ServerIdentity):
        raise ExperimentConfigurationError("typed Hands readiness response omitted server_identity")
    if trace is None or getattr(trace, "replica_id", None) != actual.replica_id:
        raise ExperimentConfigurationError("typed Hands readiness trace does not agree with worker identity")
    fields = ("experiment_id", "replica_id", "assigned_gpu", "gcs_address", "http_port", "temp_dir", "model_revision", "checkpoint_digest", "schema_version", "release_sha", "release_digest")
    mismatches = [field for field in fields if getattr(actual, field) != getattr(expected, field)]
    if expected.cuda_uuid is not None and _canonical_gpu_uuid(actual.cuda_uuid) != _canonical_gpu_uuid(expected.cuda_uuid):
        mismatches.append("cuda_uuid")
    if mismatches or actual.worker_pid <= 0:
        raise ExperimentConfigurationError("Hands worker identity rejected: " + ", ".join(mismatches or ["worker_pid"]))
    if not isinstance(diagnostics, Mapping) or diagnostics.get("runtime_config") != dict(expected_runtime_config):
        raise ExperimentConfigurationError("typed Hands readiness runtime configuration differs from planned wire contract")
    return actual


@dataclass(frozen=True)
class HandsExperimentalReplica:
    experiment_id: str
    gpu_id: int
    lifecycle: ClusterLifecycleConfig
    result_dir: Path
    hands_endpoint: ReplicaEndpoint
    wilor_endpoint: ReplicaEndpoint
    hands_identity: ServerIdentity
    wilor_identity: ServerIdentity
    hands_runtime_config: Mapping[str, object]
    wilor_runtime_config: Mapping[str, object]

    @property
    def replica_id(self) -> str:
        return self.hands_endpoint.replica_id

    @property
    def endpoint(self) -> ReplicaEndpoint:
        """Compatibility with shared preflight and scoped cleanup helpers."""
        return self.hands_endpoint

    @property
    def expected_server_identity(self) -> ServerIdentity:
        return self.hands_identity

    def launch_commands(self) -> tuple[str, str]:
        environment = " ".join(
            f"{name}={value}" for name, value in (("CUDA_VISIBLE_DEVICES", str(self.gpu_id)), *self.lifecycle.env_vars)
        )
        start = f"env -u RAY_ADDRESS {self.lifecycle.startup_command()}"
        driver = self.lifecycle.experimental_driver_command(
            release_root=dict(self.lifecycle.env_vars)["EGO_APPLICATION_RELEASE_ROOT"],
            app_choice="hands", app_name=f"hands-exp-{self.experiment_id}-gpu{self.gpu_id}",
        )
        release_cd, driver_body = driver.split(" && ", 1)
        return start, f"{release_cd} && env -u RAY_ADDRESS {environment} {driver_body}"

    def stop_commands(self) -> tuple[str, str]:
        return (
            self.lifecycle.serve_shutdown_command(),
            f"PYTHONPATH={dict(self.lifecycle.env_vars)['EGO_APPLICATION_RELEASE_ROOT']} {self.lifecycle.interpreter} "
            f"-m ego_annotation.serving.benchmark.hands_experiment --scoped-stop --temp-dir {self.lifecycle.temp_dir}",
        )


@dataclass(frozen=True)
class HandsExperimentPlan:
    experiment_id: str
    run_root: Path
    application_release: ImmutableApplicationRelease
    replicas: tuple[HandsExperimentalReplica, ...]

    def assert_isolated(self) -> None:
        if len(self.replicas) != 1:
            raise ExperimentConfigurationError("Hands experiment requires exactly one isolated physical GPU")
        replica = self.replicas[0]
        if replica.gpu_id in PRODUCTION_GPU_IDS:
            raise ExperimentConfigurationError(f"GPU{replica.gpu_id} is reserved for production")
        if replica.lifecycle.temp_dir in PRODUCTION_TEMP_DIRS:
            raise ExperimentConfigurationError("Hands experiment temp dir overlaps production")
        temp_dir = Path(replica.lifecycle.temp_dir).resolve()
        if temp_dir.parent != (HANDS_EXPERIMENT_TEMP_ROOT / self.experiment_id).resolve():
            raise ExperimentConfigurationError("Hands experiment temp dir must be one exact /tmp/ehn replica directory")
        if len(os.fsencode(str(temp_dir))) > _MAX_RAY_TEMP_DIR_BYTES:
            raise ExperimentConfigurationError("Hands experiment temp dir is too long for Ray AF_UNIX sockets")
        replica.lifecycle.assert_gpu_pinned()
        ports = set(replica.lifecycle.ports.all_ports())
        if ports & PRODUCTION_PORTS:
            raise ExperimentConfigurationError("Hands experiment ports overlap production")
        if any(port >= 32768 for port in ports):
            raise ExperimentConfigurationError("Hands experiment ports must remain below 32768")
        if any("runtime/current" in value for _, value in replica.lifecycle.env_vars):
            raise ExperimentConfigurationError("Hands experiment must not import mutable runtime/current")
        identities = (replica.hands_identity, replica.wilor_identity)
        if {identity.release_digest for identity in identities} != {self.application_release.release_digest}:
            raise ExperimentConfigurationError("Hands worker identities do not bind the pinned release")
        if len({identity.replica_id for identity in identities}) != 2:
            raise ExperimentConfigurationError("Hands and WiLoR require distinct worker-derived identities")


def build_hands_experiment_plan(
    *, experiment_id: str, gpu_id: int, run_root: str | Path, application_release_path: str | Path,
    sam2_source_release_path: str | Path, source_sha: str, detector_checkpoint_digest: str,
    sam2_checkpoint_digest: str, wilor_checkpoint_digest: str,
    gpu_uuid: str | None = None, component_port_base: int = 30400, worker_port_base: int = 30500,
    serve_port: int = 32200, wire_format: str = "envelope",
) -> HandsExperimentPlan:
    if not experiment_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in experiment_id):
        raise ExperimentConfigurationError("experiment_id must use only letters, digits, hyphen, and underscore")
    if gpu_id in PRODUCTION_GPU_IDS or gpu_id < 0:
        raise ExperimentConfigurationError(f"GPU{gpu_id} cannot be a Hands experiment target")
    if wire_format not in {"multipart", "envelope"}:
        raise ExperimentConfigurationError("wire_format must be multipart or envelope")
    if not all((detector_checkpoint_digest, sam2_checkpoint_digest, wilor_checkpoint_digest)):
        raise ExperimentConfigurationError("detector, SAM2, and WiLoR checkpoint digests are required")
    release = ImmutableApplicationRelease.pin(application_release_path, expected_source_sha=source_sha)
    # The mutable recovered checkout is evidence, never a worker import root.  Verify
    # the closed bundle before including it in either driver or worker PYTHONPATH.
    from ego_annotation.serving.sam2_source import verify_sam2_source_release
    sam2_source = verify_sam2_source_release(sam2_source_release_path, expected_core_group_digest=None)
    production = hands_wilor_gpu_group().lifecycle
    temp_dir = HANDS_EXPERIMENT_TEMP_ROOT / experiment_id / f"gpu{gpu_id}"
    ports = _experiment_ports(component_port_base, worker_port_base, serve_port)
    if any(port >= 32768 for port in ports.all_ports()):
        raise ExperimentConfigurationError("Hands experiment ports must be below 32768")
    immutable_pythonpath = ":".join(
        [str(release.path), str(sam2_source.path)] + [
            part for part in dict(production.env_vars).get("PYTHONPATH", "").split(":")
            if part and "runtime/current" not in part and part != dict(production.env_vars).get("EGO_SAM2_REPO")
        ]
    )
    hands_id = f"hands-exp-{experiment_id}-gpu{gpu_id}"
    wilor_id = f"wilor-exp-{experiment_id}-gpu{gpu_id}"
    lifecycle = replace(
        production, gpu_id=gpu_id, temp_dir=str(temp_dir), ports=ports,
        env_vars=_replace_env(production.env_vars, {
            "PYTHONPATH": immutable_pythonpath,
            "EGO_SAM2_REPO": str(sam2_source.path),
            "EGO_HANDS_GPU": str(gpu_id), "EGO_HANDS_REPLICA_ID": hands_id,
            "EGO_WILOR_REPLICA_ID": wilor_id, "EGO_EXPERIMENT_ID": experiment_id,
            "EGO_APPLICATION_RELEASE_SHA": release.source_sha, "EGO_APPLICATION_RELEASE_ROOT": str(release.path),
            "EGO_HANDS_DETECTOR_CHECKPOINT_DIGEST": detector_checkpoint_digest,
            "EGO_SAM2_CHECKPOINT_DIGEST": sam2_checkpoint_digest,
            "EGO_WILOR_CHECKPOINT_DIGEST": wilor_checkpoint_digest,
            "EGO_EXPERIMENT_GCS_ADDRESS": f"127.0.0.1:{ports.gcs_port}",
            "EGO_EXPERIMENT_HTTP_PORT": str(serve_port), "EGO_EXPERIMENT_TEMP_DIR": str(temp_dir),
            "EGO_HANDS_EXPERIMENT_TELEMETRY": "1", "EGO_WILOR_EXPERIMENT_TELEMETRY": "1",
            "EGO_HANDS_EXPERIMENT_WIRE_FORMAT": wire_format, "EGO_WILOR_EXPERIMENT_WIRE_FORMAT": wire_format,
            "PYTHONDONTWRITEBYTECODE": "1",
        }),
    )
    replica = HandsExperimentalReplica(
        experiment_id=experiment_id, gpu_id=gpu_id, lifecycle=lifecycle,
        result_dir=Path(run_root) / experiment_id / hands_id,
        hands_endpoint=ReplicaEndpoint(hands_id, f"http://127.0.0.1:{serve_port}/hands.detect", gpu_id),
        wilor_endpoint=ReplicaEndpoint(wilor_id, f"http://127.0.0.1:{serve_port}/wilor.reconstruct", gpu_id),
        hands_identity=ServerIdentity(
            experiment_id=experiment_id, replica_id=hands_id, assigned_gpu=gpu_id, worker_pid=1,
            gcs_address=lifecycle.gcs_address, http_port=serve_port, temp_dir=str(temp_dir),
            model_revision=str(dict(lifecycle.env_vars).get("EGO_HANDS_REVISION", "hands-yolo-sam2.1-hiera-l")),
            checkpoint_digest=detector_checkpoint_digest, schema_version=SCHEMA_VERSION, release_sha=release.source_sha,
            release_digest=release.release_digest, cuda_uuid=gpu_uuid, module_root=str(release.path),
        ),
        wilor_identity=ServerIdentity(
            experiment_id=experiment_id, replica_id=wilor_id, assigned_gpu=gpu_id, worker_pid=1,
            gcs_address=lifecycle.gcs_address, http_port=serve_port, temp_dir=str(temp_dir),
            model_revision=str(dict(lifecycle.env_vars).get("EGO_WILOR_REVISION", "wilor-final-v1")),
            checkpoint_digest=wilor_checkpoint_digest, schema_version=SCHEMA_VERSION, release_sha=release.source_sha,
            release_digest=release.release_digest, cuda_uuid=gpu_uuid, module_root=str(release.path),
        ),
        hands_runtime_config=expected_hands_runtime_config(wire_format=wire_format),
        wilor_runtime_config=expected_wilor_runtime_config(wire_format=wire_format),
    )
    plan = HandsExperimentPlan(experiment_id, Path(run_root), release, (replica,))
    plan.assert_isolated()
    return plan


def _main(argv: Sequence[str] | None = None) -> int:
    """Environment-owned exact-process cleanup for the Hands temp-root only."""
    import argparse
    import json
    from ego_annotation.serving.benchmark.unidepth_scaling import _candidate_process_pids, stop_scoped_experiment

    parser = argparse.ArgumentParser(description="Inspect or scoped-stop a Hands experiment")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--scoped-stop", action="store_true")
    action.add_argument("--scoped-status", action="store_true")
    parser.add_argument("--temp-dir", required=True)
    args = parser.parse_args(argv)
    resolved = str(Path(args.temp_dir).resolve())
    relative = Path(resolved).relative_to(HANDS_EXPERIMENT_TEMP_ROOT.resolve())
    if len(relative.parts) != 2 or not relative.parts[1].startswith("gpu") or not relative.parts[1][3:].isdigit():
        raise ExperimentConfigurationError("refusing non-exact Hands experiment temp dir")
    if args.scoped_status:
        pids = _candidate_process_pids(resolved)
        print(json.dumps({"temp_dir": resolved, "candidate_pids": pids}))
        return 1 if pids else 0
    pids = stop_scoped_experiment(resolved, experiment_temp_root=HANDS_EXPERIMENT_TEMP_ROOT)
    print(json.dumps({"temp_dir": resolved, "stopped_pids": pids}))
    return 0


if __name__ == "__main__":  # pragma: no cover - server-only scoped cleanup
    raise SystemExit(_main())
