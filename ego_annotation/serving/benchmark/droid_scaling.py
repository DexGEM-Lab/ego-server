"""Isolated vacant-GPU DROID experiment lifecycle and typed identity contracts.

The planner reuses the content-addressed release, exact preflight transaction,
rollback, and port allocation mechanisms already exercised by UniDepth.  It never
changes the canonical router or committed GPU topology.  GPU vacancy remains an
immediate executable preflight observation, not a durable assignment.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from ego_annotation.serving.benchmark.unidepth_scaling import (
    PRODUCTION_GPU_IDS,
    PRODUCTION_PORTS,
    PRODUCTION_TEMP_DIRS,
    CommandOutcome,
    ExperimentConfigurationError,
    ImmutableApplicationRelease,
    LocalBlockingCommandRunner,
    ScopedExecutionResult,
    _candidate_process_pids,
    _experiment_ports,
    _replace_env,
    execute_scoped_plan,
    preflight_check_command,
    stop_scoped_experiment,
)
from ego_annotation.serving.contracts import (
    SCHEMA_VERSION,
    ContractValidationError,
    DroidCreateSessionRequest,
    DroidCreateSessionResponse,
    DroidFinalizeRequest,
    DroidFinalizeResponse,
    DroidFrameRequest,
    DroidFrameResponse,
    ErrorCode,
    Ownership,
    ServerIdentity,
)
from ego_annotation.serving.lifecycle import ClusterLifecycleConfig, droid_gpu_group
from ego_annotation.serving.droid_source import VerifiedDroidSourceRelease, verify_droid_source_release
from ego_annotation.serving.transport import build_multipart_request_fields, parse_droid_finalize_response

AUTHORIZED_EXPERIMENT_GPU_IDS = frozenset({4, 5, 7})
# The isolation-only contract is intentionally independent of production process
# or HTTP state.  These are immutable resource allocations, checked from the
# experiment plan before launch; no production endpoint is contacted.
ISOLATION_PRODUCTION_GPU_IDS = frozenset({0, 1, 2, 3, 6})
ISOLATION_PRODUCTION_HTTP_PORTS = frozenset(range(28000, 28007))
ISOLATION_PRODUCTION_TEMP_PREFIX = "/tmp/ray-ego-serve-gpu"
# Keep the deepest Ray socket path below AF_UNIX's 107-byte limit. This short,
# experiment-specific `/tmp` parent is created by the zjh preflight itself; `/home/zjh`
# is a long GPFS symlink on dex-a800 and cannot satisfy the socket-path budget.
DROID_EXPERIMENT_TEMP_ROOT = Path("/tmp/zjheds")
_MAX_RAY_TEMP_DIR_BYTES = 36
# dex-a800's observed Linux ephemeral source-port range. Ray opens local control
# connections while child agents bind their explicit ports; placing dashboard/worker
# ports inside this range creates a preflight-to-bind race with ephemeral allocation.
_EPHEMERAL_PORT_MIN = 32768
_EPHEMERAL_PORT_MAX = 60999


@dataclass(frozen=True)
class DroidReplicaEndpoint:
    replica_id: str
    base_url: str
    gpu_id: int

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.port is None:
            raise ExperimentConfigurationError(f"invalid DROID endpoint {self.base_url!r}")
        if parsed.port in PRODUCTION_PORTS:
            raise ExperimentConfigurationError("DROID experiment endpoint collides with production")
        if self.gpu_id not in AUTHORIZED_EXPERIMENT_GPU_IDS:
            raise ExperimentConfigurationError(f"GPU{self.gpu_id} is not authorized experiment capacity")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))


@dataclass(frozen=True)
class ExperimentalDroidReplica:
    replica_id: str
    gpu_id: int
    lifecycle: ClusterLifecycleConfig
    endpoint: DroidReplicaEndpoint
    result_dir: Path
    expected_server_identity: ServerIdentity

    def launch_commands(self) -> tuple[str, str]:
        environment = " ".join(
            f"{name}={value}" for name, value in (("CUDA_VISIBLE_DEVICES", str(self.gpu_id)), *self.lifecycle.env_vars)
        )
        start = f"env -u RAY_ADDRESS {self.lifecycle.startup_command()}"
        driver = self.lifecycle.experimental_driver_command(
            release_root=dict(self.lifecycle.env_vars)["EGO_APPLICATION_RELEASE_ROOT"],
            app_choice="droid",
            app_name=self.replica_id,
        )
        release_cd, driver_body = driver.split(" && ", 1)
        deploy = f"{release_cd} && env -u RAY_ADDRESS {environment} {driver_body}"
        return start, deploy

    def stop_commands(self) -> tuple[str, str]:
        return (
            self.lifecycle.serve_shutdown_command(),
            f"{self.lifecycle.interpreter} -m ego_annotation.serving.benchmark.droid_scaling "
            f"--scoped-stop --temp-dir {self.lifecycle.temp_dir}",
        )


@dataclass(frozen=True)
class DroidScalingExperimentPlan:
    experiment_id: str
    run_root: Path
    application_release: ImmutableApplicationRelease
    droid_source_release: VerifiedDroidSourceRelease
    corpus_digest: str
    measurement_interval_id: str
    cpu_offload: bool
    max_sessions: int
    max_concurrent_ba: int = 1
    replicas: tuple[ExperimentalDroidReplica, ...] = ()

    @property
    def launch_configuration(self) -> dict[str, Any]:
        """Worker-affecting treatment state attested by the launch plan.

        The deployment emits the same ``cpu_offload`` value in its typed runtime
        diagnostics.  Keeping it in the immutable plan makes resident/offload
        comparisons distinguishable before any model work starts.
        """
        return {
            "application_release_digest": self.application_release.release_digest,
            "corpus_digest": self.corpus_digest,
            "cpu_offload": self.cpu_offload,
            "max_sessions": self.max_sessions,
            "max_concurrent_ba": self.max_concurrent_ba,
            "droid_source_digest": self.droid_source_release.source_digest,
            "measurement_interval_id": self.measurement_interval_id,
        }

    @property
    def launch_configuration_digest(self) -> str:
        encoded = json.dumps(self.launch_configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def endpoints(self) -> tuple[DroidReplicaEndpoint, ...]:
        return tuple(replica.endpoint for replica in self.replicas)

    @property
    def expected_server_identities(self) -> dict[str, ServerIdentity]:
        return {replica.replica_id: replica.expected_server_identity for replica in self.replicas}

    def assert_isolated(self) -> None:
        ports: list[int] = []
        if len({replica.replica_id for replica in self.replicas}) != len(self.replicas):
            raise ExperimentConfigurationError("each DROID replica must have a distinct replica id")
        if len({replica.lifecycle.temp_dir for replica in self.replicas}) != len(self.replicas):
            raise ExperimentConfigurationError("each DROID replica must have a distinct exact temp directory")
        if len({replica.endpoint.base_url for replica in self.replicas}) != len(self.replicas):
            raise ExperimentConfigurationError("each DROID replica must have a distinct HTTP endpoint")
        for replica in self.replicas:
            if replica.gpu_id not in AUTHORIZED_EXPERIMENT_GPU_IDS or replica.gpu_id in PRODUCTION_GPU_IDS:
                raise ExperimentConfigurationError(f"GPU{replica.gpu_id} is not authorized vacant experiment capacity")
            if replica.lifecycle.interpreter != droid_gpu_group().lifecycle.interpreter:
                raise ExperimentConfigurationError("DROID experiment must use the exact ray_serve_hawor interpreter")
            if replica.lifecycle.temp_dir in PRODUCTION_TEMP_DIRS or not _is_exact_droid_temp_dir(replica.lifecycle.temp_dir):
                raise ExperimentConfigurationError("DROID experiment temp directory is outside its exact scoped root")
            replica.lifecycle.assert_gpu_pinned()
            env = dict(replica.lifecycle.env_vars)
            production_env = dict(droid_gpu_group().lifecycle.env_vars)
            for name in ("EGO_DROID_WEIGHTS", "EGO_DROID_REVISION", "EGO_DROID_DEVICE"):
                if env.get(name) != production_env.get(name):
                    raise ExperimentConfigurationError(f"DROID experiment changed exact production ABI field {name}")
            if env.get("EGO_DROID_REPO") != str(self.droid_source_release.source_root):
                raise ExperimentConfigurationError("DROID experiment must import only the verified bundle droid_slam root")
            if (
                env.get("EGO_DROID_SOURCE_ROOT") != str(self.droid_source_release.path)
                or env.get("EGO_DROID_SOURCE_DIGEST") != self.droid_source_release.source_digest
                or env.get("EGO_DROID_SOURCE_AMENDMENT") != self.droid_source_release.amendment_id
            ):
                raise ExperimentConfigurationError("DROID experiment source environment differs from verified source release")
            if env.get("PYTHONDONTWRITEBYTECODE") != "1":
                raise ExperimentConfigurationError("DROID experiment must disable import bytecode writes")
            if env.get("EGO_DROID_CPU_OFFLOAD") != ("1" if self.cpu_offload else "0"):
                raise ExperimentConfigurationError("DROID experiment CPU-offload environment differs from launch plan")
            if env.get("EGO_DROID_MAX_SESSIONS") != str(self.max_sessions):
                raise ExperimentConfigurationError("DROID experiment max-session environment differs from launch plan")
            if env.get("EGO_DROID_MAX_CONCURRENT_BA") != str(self.max_concurrent_ba):
                raise ExperimentConfigurationError("DROID experiment concurrent-BA environment differs from launch plan")
            if "runtime/current" in env.get("PYTHONPATH", ""):
                raise ExperimentConfigurationError("DROID experiment cannot import mutable runtime/current")
            if (
                replica.expected_server_identity.release_digest != self.application_release.release_digest
                or replica.expected_server_identity.release_sha != self.application_release.source_sha
            ):
                raise ExperimentConfigurationError("DROID planned runtime release digest/source differs from pinned release")
            if (
                replica.expected_server_identity.dependency_digest != self.droid_source_release.source_digest
                or replica.expected_server_identity.dependency_root != str(self.droid_source_release.source_root)
                or replica.expected_server_identity.source_amendment_id != self.droid_source_release.amendment_id
            ):
                raise ExperimentConfigurationError("DROID planned dependency source differs from verified source release")
            candidate_ports = replica.lifecycle.ports.all_ports()
            if any(port <= 0 or port > 65535 for port in candidate_ports):
                raise ExperimentConfigurationError("DROID experiment ports must be in 1..65535")
            if any(_EPHEMERAL_PORT_MIN <= port <= _EPHEMERAL_PORT_MAX for port in candidate_ports):
                raise ExperimentConfigurationError(
                    "DROID Ray/HTTP ports must stay outside dex-a800's ephemeral source-port range"
                )
            ports.extend(candidate_ports)
        if len(ports) != len(set(ports)):
            raise ExperimentConfigurationError("DROID replica ports overlap")
        overlap = set(ports) & (PRODUCTION_PORTS | ISOLATION_PRODUCTION_HTTP_PORTS)
        if overlap:
            raise ExperimentConfigurationError(f"DROID experiment ports overlap production: {sorted(overlap)}")
        for replica in self.replicas:
            if replica.gpu_id in ISOLATION_PRODUCTION_GPU_IDS:
                raise ExperimentConfigurationError(f"DROID experiment GPU{replica.gpu_id} is a production resource")
            if str(replica.lifecycle.temp_dir).startswith(ISOLATION_PRODUCTION_TEMP_PREFIX):
                raise ExperimentConfigurationError("DROID experiment temp directory overlaps production temp namespace")


def build_droid_scaling_plan(
    *,
    experiment_id: str,
    gpu_ids: Sequence[int],
    gpu_uuids: Sequence[str],
    run_root: str | Path,
    application_release_path: str | Path,
    droid_source_release_path: str | Path,
    source_sha: str,
    checkpoint_digest: str,
    corpus_digest: str,
    measurement_interval_id: str,
    cpu_offload: bool = False,
    max_sessions: int = 16,
    max_concurrent_ba: int = 1,
    ipc_handle_file: str | None = None,
    component_port_base: int = 30000,
    worker_port_base: int = 30100,
    serve_port_base: int = 32000,
    component_port_bases: Sequence[int] | None = None,
    worker_port_bases: Sequence[int] | None = None,
    serve_port_bases: Sequence[int] | None = None,
    replica_labels: Sequence[str] | None = None,
    temp_root: str | Path = DROID_EXPERIMENT_TEMP_ROOT,
) -> DroidScalingExperimentPlan:
    if not experiment_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in experiment_id):
        raise ExperimentConfigurationError("experiment_id must use letters, digits, hyphen, and underscore")
    gpu_ids = tuple(gpu_ids)
    gpu_uuids = tuple(gpu_uuids)
    if not gpu_ids:
        raise ExperimentConfigurationError("DROID experiment GPU ids must be non-empty")
    if any(gpu not in AUTHORIZED_EXPERIMENT_GPU_IDS for gpu in gpu_ids):
        raise ExperimentConfigurationError("DROID experiments are authorized only on GPU4, GPU5, and GPU7")
    if len(gpu_uuids) != len(gpu_ids) or any(not uuid.strip() for uuid in gpu_uuids):
        raise ExperimentConfigurationError("one preflighted physical GPU UUID is required per DROID replica")
    if replica_labels is None:
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ExperimentConfigurationError("same-GPU DROID replicas require explicit unique replica labels")
        labels = tuple(f"gpu{gpu_id}" for gpu_id in gpu_ids)
    else:
        labels = tuple(replica_labels)
        if len(labels) != len(gpu_ids) or len(set(labels)) != len(labels):
            raise ExperimentConfigurationError("DROID replica labels must be unique and align one-for-one with GPUs")
        if any(not label or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in label) for label in labels):
            raise ExperimentConfigurationError("DROID replica labels must use only letters, digits, hyphen, and underscore")
    if not checkpoint_digest:
        raise ExperimentConfigurationError("actual DROID checkpoint digest is required")
    if not corpus_digest or not measurement_interval_id:
        raise ExperimentConfigurationError("DROID plan requires one immutable corpus digest and measurement interval id")
    if not isinstance(cpu_offload, bool):
        raise ExperimentConfigurationError("DROID CPU-offload treatment must be an explicit boolean")
    if not isinstance(max_sessions, int) or isinstance(max_sessions, bool) or max_sessions <= 0:
        raise ExperimentConfigurationError("DROID max_sessions must be a positive integer")
    explicit_port_bases = (component_port_bases, worker_port_bases, serve_port_bases)
    if any(value is not None for value in explicit_port_bases):
        if any(value is None for value in explicit_port_bases):
            raise ExperimentConfigurationError("DROID explicit component, worker, and HTTP port bases must be supplied together")
        component_bases = tuple(component_port_bases or ())
        worker_bases = tuple(worker_port_bases or ())
        serve_bases = tuple(serve_port_bases or ())
        if any(len(values) != len(gpu_ids) for values in (component_bases, worker_bases, serve_bases)):
            raise ExperimentConfigurationError("DROID explicit port-base lists must align one-for-one with GPUs")
    else:
        component_bases = tuple(component_port_base + index * 200 for index in range(len(gpu_ids)))
        worker_bases = tuple(worker_port_base + index * 200 for index in range(len(gpu_ids)))
        serve_bases = tuple(serve_port_base + index for index in range(len(gpu_ids)))
    canonical_temp = Path(temp_root).resolve()
    if canonical_temp != DROID_EXPERIMENT_TEMP_ROOT.resolve():
        raise ExperimentConfigurationError(f"DROID temp root must be exactly {DROID_EXPERIMENT_TEMP_ROOT}")
    release = ImmutableApplicationRelease.pin(application_release_path, expected_source_sha=source_sha)
    droid_source_release = verify_droid_source_release(droid_source_release_path)
    if not (release.path / "ego_annotation" / "serving" / "droid_deployment.py").is_file():
        raise ExperimentConfigurationError("release is missing the DROID deployment application")
    production = droid_gpu_group().lifecycle
    production_env = dict(production.env_vars)
    replicas: list[ExperimentalDroidReplica] = []
    seen_gpu_ids: set[int] = set()
    for index, (gpu_id, gpu_uuid, label) in enumerate(zip(gpu_ids, gpu_uuids, labels)):
        ports = _experiment_ports(component_bases[index], worker_bases[index], serve_bases[index])
        if any(port <= 0 or port > 65535 for port in ports.all_ports()):
            raise ExperimentConfigurationError("DROID experiment ports must be in 1..65535")
        temp_dir = canonical_temp / experiment_id / label
        if len(os.fsencode(str(temp_dir))) > _MAX_RAY_TEMP_DIR_BYTES:
            raise ExperimentConfigurationError(
                f"DROID experiment temp dir is too long for Ray AF_UNIX sockets: {temp_dir} "
                f"({len(os.fsencode(str(temp_dir)))} > {_MAX_RAY_TEMP_DIR_BYTES} bytes)"
            )
        replica_id = f"droid-exp-{experiment_id}-{label}"
        pythonpath = ":".join(
            [str(release.path)] + [part for part in production_env.get("PYTHONPATH", "").split(":") if part and "runtime/current" not in part]
        )
        lifecycle = replace(
            production,
            gpu_id=gpu_id,
            temp_dir=str(temp_dir),
            ports=ports,
            env_vars=_replace_env(production.env_vars, {
                "PYTHONPATH": pythonpath,
                "EGO_DROID_GPU": str(gpu_id),
                "EGO_DROID_REPLICA_ID": replica_id,
                "EGO_DROID_REPO": str(droid_source_release.source_root),
                "EGO_DROID_SOURCE_ROOT": str(droid_source_release.path),
                "EGO_DROID_SOURCE_DIGEST": droid_source_release.source_digest,
                "EGO_DROID_SOURCE_AMENDMENT": droid_source_release.amendment_id,
                "EGO_EXPERIMENT_ID": experiment_id,
                "EGO_APPLICATION_RELEASE_SHA": release.source_sha,
                "EGO_APPLICATION_RELEASE_ROOT": str(release.path),
                "EGO_EXPERIMENT_GCS_ADDRESS": f"127.0.0.1:{ports.gcs_port}",
                "EGO_EXPERIMENT_HTTP_PORT": str(ports.serve_http_port),
                "EGO_EXPERIMENT_TEMP_DIR": str(temp_dir),
                # The explicit plan limit gates actor admission and is carried in
                # typed worker diagnostics through the immutable launch config.
                "EGO_DROID_MAX_SESSIONS": str(max_sessions),
                "EGO_DROID_CPU_OFFLOAD": "1" if cpu_offload else "0",
                "EGO_DROID_MAX_CONCURRENT_BA": str(max_concurrent_ba),
                **({"EGO_DROID_IPC_HANDLE_FILE": ipc_handle_file} if ipc_handle_file else {}),
                "EGO_DROID_EXPERIMENT_TELEMETRY": "1",
                # A same-GPU plan establishes vacancy before the first cluster only.
                # Later replicas share that intentionally occupied GPU but retain
                # independent port/temp ownership and typed identity checks.
                "EGO_EXPERIMENT_REQUIRE_GPU_VACANCY": "0" if gpu_id in seen_gpu_ids else "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }),
        )
        identity = ServerIdentity(
            experiment_id=experiment_id,
            replica_id=replica_id,
            assigned_gpu=gpu_id,
            worker_pid=1,
            gcs_address=lifecycle.gcs_address,
            http_port=ports.serve_http_port,
            temp_dir=str(temp_dir),
            model_revision=production_env["EGO_DROID_REVISION"],
            checkpoint_digest=checkpoint_digest,
            schema_version=SCHEMA_VERSION,
            release_sha=release.source_sha,
            release_digest=release.release_digest,
            cuda_uuid=gpu_uuid,
            module_root=str(release.path),
            dependency_digest=droid_source_release.source_digest,
            dependency_root=str(droid_source_release.source_root),
            source_amendment_id=droid_source_release.amendment_id,
        )
        replicas.append(ExperimentalDroidReplica(
            replica_id=replica_id,
            gpu_id=gpu_id,
            lifecycle=lifecycle,
            endpoint=DroidReplicaEndpoint(replica_id, f"http://127.0.0.1:{ports.serve_http_port}", gpu_id),
            result_dir=Path(run_root) / experiment_id / replica_id,
            expected_server_identity=identity,
        ))
        seen_gpu_ids.add(gpu_id)
    plan = DroidScalingExperimentPlan(
        experiment_id, Path(run_root), release, droid_source_release, corpus_digest, measurement_interval_id,
        cpu_offload, max_sessions, max_concurrent_ba, tuple(replicas)
    )
    plan.assert_isolated()
    return plan


def validate_droid_server_identity(
    expected: ServerIdentity,
    response: DroidCreateSessionResponse | DroidFrameResponse | DroidFinalizeResponse,
) -> ServerIdentity:
    actual = response.server_identity
    if actual is None:
        raise ExperimentConfigurationError("typed DROID response omitted worker runtime identity")
    for identity_name, identity in (("expected", expected), ("actual", actual)):
        missing = [
            field for field in ("dependency_digest", "dependency_root", "source_amendment_id")
            if not isinstance(getattr(identity, field), str) or not getattr(identity, field).strip()
        ]
        if missing:
            raise ExperimentConfigurationError(
                f"DROID {identity_name} runtime identity lacks dependency source identity: {', '.join(missing)}"
            )
    fields = (
        "experiment_id", "replica_id", "assigned_gpu", "gcs_address", "http_port", "temp_dir",
        "model_revision", "checkpoint_digest", "schema_version", "release_sha", "release_digest", "module_root",
        "dependency_digest", "dependency_root", "source_amendment_id",
    )
    mismatches = [name for name in fields if getattr(actual, name) != getattr(expected, name)]
    if _canonical_cuda_uuid(actual.cuda_uuid) != _canonical_cuda_uuid(expected.cuda_uuid):
        mismatches.append("cuda_uuid")
    if mismatches:
        raise ExperimentConfigurationError("DROID worker runtime identity mismatch: " + ", ".join(mismatches))
    if actual.worker_pid <= 0:
        raise ExperimentConfigurationError("DROID worker PID evidence must be positive")
    trace = None
    if isinstance(response, DroidFrameResponse) and response.status is not None:
        trace = response.status.trace
    elif isinstance(response, DroidFinalizeResponse) and response.camera_state is not None:
        trace = response.camera_state.trace
    if trace is not None and trace.replica_id != actual.replica_id:
        raise ExperimentConfigurationError("DROID trace replica disagrees with worker runtime identity")
    return actual


@dataclass(frozen=True)
class DroidTypedCall:
    operation: str
    sent_s: float
    completed_s: float
    http_status: int
    response: DroidCreateSessionResponse | DroidFrameResponse | DroidFinalizeResponse | None = None
    error: str | None = None


async def post_droid_typed(
    client: Any,
    endpoint: DroidReplicaEndpoint,
    operation: str,
    request: DroidCreateSessionRequest | DroidFrameRequest | DroidFinalizeRequest,
) -> DroidTypedCall:
    arrays: dict[str, tuple[bytes, tuple[int, ...], str]] = {}
    if isinstance(request, DroidFrameRequest):
        metadata = {
            "ownership": request.ownership.to_wire(), "session_id": request.session_id,
            "frame_id": request.frame_id, "source_timestamp_s": request.source_timestamp_s,
            "model_revision": request.model_revision,
        }
        for name, tensor in (("rgb", request.rgb), ("static_confidence_mask", request.static_confidence_mask), ("depth_m", request.depth_m)):
            if tensor is not None:
                if not isinstance(tensor.data, (bytes, bytearray, memoryview)):
                    raise ContractValidationError(f"DROID HTTP {name} must be binary")
                arrays[name] = (bytes(tensor.data), tensor.shape, tensor.dtype)
    else:
        metadata = request.to_wire()
    body, content_type = build_multipart_request_fields(metadata, arrays)
    path = {"create_session": "droid.create_session", "push_frame": "droid.push_frame", "finalize": "droid.finalize"}[operation]
    sent = time.monotonic()
    try:
        raw = await client.post(f"{endpoint.base_url}/{path}", content=body, headers={"Content-Type": content_type})
        completed = time.monotonic()
        response_content_type = str(raw.headers.get("content-type", raw.headers.get("Content-Type", "")))
        if operation == "finalize" and "multipart/form-data" in response_content_type.lower():
            parsed = parse_droid_finalize_response(bytes(raw.content), response_content_type)
        else:
            wire = raw.json()
            if not isinstance(wire, Mapping):
                raise ContractValidationError("typed DROID response JSON must be an object")
            cls = {"create_session": DroidCreateSessionResponse, "push_frame": DroidFrameResponse, "finalize": DroidFinalizeResponse}[operation]
            parsed = cls.from_wire(wire)
        if parsed.ownership != request.ownership:
            raise ContractValidationError("typed DROID response ownership mismatch")
        return DroidTypedCall(operation, sent, completed, int(raw.status_code), parsed)
    except Exception as exc:
        return DroidTypedCall(operation, sent, time.monotonic(), 0, error=str(exc))


async def typed_readiness_sequence(
    *,
    client: Any,
    endpoint: DroidReplicaEndpoint,
    expected_identity: ServerIdentity,
    create_request: DroidCreateSessionRequest,
    frame_request: DroidFrameRequest,
) -> dict[str, Any]:
    """Create, push one real frame, and terminally finalize without leaking a session."""
    create = await post_droid_typed(client, endpoint, "create_session", create_request)
    if not isinstance(create.response, DroidCreateSessionResponse) or create.response.session_id is None:
        raise ExperimentConfigurationError(f"DROID typed readiness create failed: {create.error or create.response}")
    session_id = create.response.session_id
    primary_error: BaseException | None = None
    push: DroidTypedCall | None = None
    terminal: DroidTypedCall | None = None
    try:
        validate_droid_server_identity(expected_identity, create.response)
        push_request = replace(frame_request, session_id=session_id)
        push = await post_droid_typed(client, endpoint, "push_frame", push_request)
        if not isinstance(push.response, DroidFrameResponse) or push.response.status is None:
            raise ExperimentConfigurationError(f"DROID typed readiness push failed: {push.error or push.response}")
        validate_droid_server_identity(expected_identity, push.response)
    except BaseException as exc:
        primary_error = exc
    finally:
        finalize = DroidFinalizeRequest(
            ownership=Ownership(
                request_id=f"{create_request.ownership.request_id}-terminal-finalize",
                job_id=create_request.ownership.job_id,
                item_id=create_request.ownership.item_id,
                stage_id="droid.finalize",
                source_id=create_request.ownership.source_id,
            ),
            session_id=session_id,
            model_revision=expected_identity.model_revision,
        )
        terminal = await post_droid_typed(client, endpoint, "finalize", finalize)
    terminal_response = terminal.response
    if not isinstance(terminal_response, DroidFinalizeResponse):
        raise ExperimentConfigurationError(f"DROID readiness could not terminally finalize: {terminal.error}")
    validate_droid_server_identity(expected_identity, terminal_response)
    if not terminal_response.terminal:
        raise ExperimentConfigurationError("DROID readiness finalize omitted terminal lifecycle evidence")
    if primary_error is not None:
        raise primary_error
    return {
        "endpoint": endpoint.base_url,
        "replica_id": expected_identity.replica_id,
        "server_identity": terminal_response.server_identity.to_wire() if terminal_response.server_identity else None,
        "create_http_status": create.http_status,
        "push_http_status": push.http_status if push else None,
        "terminal_http_status": terminal.http_status,
        "terminal_outcome": (
            "camera_state" if terminal_response.camera_state is not None
            else terminal_response.error.code.value if terminal_response.error is not None else "invalid"
        ),
    }


def droid_preflight_check_command(replica: ExperimentalDroidReplica) -> str:
    """Reserve one vacant, statically disjoint DROID experiment scope.

    Static disjointness (GPU assignment, the complete 28000--28006 HTTP range,
    and the production temp namespace) is enforced by ``plan.assert_isolated``.
    This executable half observes only the candidate GPU, candidate ports, and
    candidate temp root.  It must never curl, probe, or otherwise interact with a
    production service.
    """
    root = shlex.quote(str(DROID_EXPERIMENT_TEMP_ROOT))
    temp_dir = shlex.quote(replica.lifecycle.temp_dir)
    env = dict(replica.lifecycle.env_vars)
    if env.get("EGO_EXPERIMENT_REQUIRE_GPU_VACANCY") == "1":
        resource_check = preflight_check_command(replica)
    else:
        ports = "|".join(str(port) for port in replica.lifecycle.ports.all_ports())
        expected = replica.expected_server_identity.cuda_uuid or ""
        resource_check = (
            f"( test -z \"{expected}\" || test \"$(nvidia-smi -i {replica.gpu_id} --query-gpu=uuid "
            f"--format=csv,noheader | tr -d ' ')\" = \"{expected}\" ) && "
            f"! ss -ltn | awk '{{print $4}}' | grep -Eq '(:({ports}))$'"
        )
    return (
        f"mkdir -p {root} && test -d {root} && test -w {root} && test -x {root} && "
        f"test ! -e {temp_dir} && {resource_check}"
    )


_FAILURE_LOG_NAMES = (
    "dashboard.err",
    "dashboard.log",
    "dashboard_MetricsHead.err",
    "dashboard_MetricsHead.log",
    "dashboard_agent.err",
    "dashboard_agent.log",
    "monitor.err",
    "monitor.log",
    "raylet.err",
    "raylet.out",
    "gcs_server.err",
    "gcs_server.out",
)


def retain_droid_failure_logs(temp_dir: str | Path, evidence_dir: str | Path) -> dict[str, Any]:
    """Copy bounded named Ray logs before cleanup of one exact DROID temp scope.

    A failed Ray start can remove the only child-process traceback during scoped
    rollback. The source is constrained to the exact DROID experiment temp dir;
    missing sessions or named files are recorded rather than treated as a second
    failure.
    """
    resolved = Path(temp_dir).resolve()
    if not _is_exact_droid_temp_dir(resolved):
        raise ExperimentConfigurationError("refusing failure-log retention outside one exact DROID temp directory")
    destination = Path(evidence_dir)
    if destination.is_symlink():
        raise ExperimentConfigurationError("refusing symlinked DROID failure-evidence directory")
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()

    session_root: Path | None = None
    latest = resolved / "session_latest"
    if latest.exists():
        candidate = latest.resolve()
        try:
            candidate.relative_to(resolved)
        except ValueError:
            candidate = None
        if candidate is not None and candidate.is_dir():
            session_root = candidate
    if session_root is None and resolved.is_dir():
        sessions = sorted(
            (path for path in resolved.glob("session_*") if path.is_dir() and not path.is_symlink()),
            key=lambda path: path.name,
        )
        if sessions:
            session_root = sessions[-1]

    log_dir = session_root / "logs" if session_root is not None else None
    copied: list[str] = []
    missing: list[str] = []
    for name in _FAILURE_LOG_NAMES:
        source = log_dir / name if log_dir is not None else None
        if source is None or not source.is_file() or source.is_symlink():
            missing.append(name)
            continue
        shutil.copyfile(source, destination / name)
        copied.append(name)
    report = {
        "schema": "ego.droid-failure-log-retention.v1",
        "temp_dir": str(resolved),
        "session_dir": str(session_root) if session_root is not None else None,
        "copied": copied,
        "missing": missing,
    }
    (destination / "failure_log_retention.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def execute_droid_scoped_plan(
    plan: DroidScalingExperimentPlan,
    *,
    command_runner: Any,
    typed_readiness_probe: Any,
    preflight_runner: Any | None = None,
    cleanup_verifier: Any | None = None,
    failure_evidence_dir: str | Path | None = None,
) -> ScopedExecutionResult:
    def retain(replica: ExperimentalDroidReplica) -> None:
        if failure_evidence_dir is not None:
            retain_droid_failure_logs(replica.lifecycle.temp_dir, Path(failure_evidence_dir) / replica.replica_id)

    return execute_scoped_plan(
        plan,
        command_runner=command_runner,
        typed_readiness_probe=typed_readiness_probe,
        preflight_runner=preflight_runner,
        cleanup_verifier=cleanup_verifier,
        failure_artifact_hook=retain if failure_evidence_dir is not None else None,
        preflight_command_factory=droid_preflight_check_command,
        cleanup_command_factory=cleanup_verification_command,
    )


def _canonical_cuda_uuid(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized[4:] if normalized.startswith("gpu-") else normalized


def _is_exact_droid_temp_dir(temp_dir: str | Path) -> bool:
    try:
        relative = Path(temp_dir).resolve().relative_to(DROID_EXPERIMENT_TEMP_ROOT.resolve())
    except ValueError:
        return False
    return (
        len(relative.parts) == 2
        and bool(relative.parts[0])
        and bool(relative.parts[1])
        and all(char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in relative.parts[0])
        and all(char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in relative.parts[1])
    )


def cleanup_verification_command(replica: ExperimentalDroidReplica) -> str:
    ports = "|".join(str(port) for port in replica.lifecycle.ports.all_ports())
    return (
        f"{replica.lifecycle.interpreter} -m ego_annotation.serving.benchmark.droid_scaling "
        f"--scoped-status --temp-dir {shlex.quote(replica.lifecycle.temp_dir)} && "
        f"! ss -ltn | awk '{{print $4}}' | grep -Eq '(:({ports}))$'"
    )


def stop_droid_scoped_experiment(temp_dir: str | Path, **kwargs: Any) -> tuple[int, ...]:
    """Stop and retire exactly one DROID scope so its next preflight is genuinely fresh."""
    resolved = str(Path(temp_dir).resolve())
    if not _is_exact_droid_temp_dir(resolved):
        raise ExperimentConfigurationError("refusing non-exact DROID experiment stop target")
    stopped = stop_scoped_experiment(resolved, experiment_temp_root=DROID_EXPERIMENT_TEMP_ROOT, **kwargs)
    pid_lookup = kwargs.get("pid_lookup", _candidate_process_pids)
    if pid_lookup(resolved):
        raise ExperimentConfigurationError("refusing to remove DROID temp directory while scoped processes remain")
    target = Path(resolved)
    if target.is_symlink():
        raise ExperimentConfigurationError("refusing to remove symlinked DROID temp directory")
    if target.exists():
        shutil.rmtree(target)
    return stopped


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect or stop one exact DROID experimental process scope")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--scoped-stop", action="store_true")
    action.add_argument("--scoped-status", action="store_true")
    parser.add_argument("--temp-dir", required=True)
    args = parser.parse_args(argv)
    resolved = str(Path(args.temp_dir).resolve())
    if not _is_exact_droid_temp_dir(resolved):
        raise ExperimentConfigurationError("refusing non-exact DROID experiment status target")
    if args.scoped_status:
        from ego_annotation.serving.benchmark.unidepth_scaling import _candidate_process_pids

        pids = _candidate_process_pids(resolved)
        print(json.dumps({"temp_dir": resolved, "candidate_pids": pids}))
        return 1 if pids else 0
    print(json.dumps({"temp_dir": resolved, "stopped_pids": stop_droid_scoped_experiment(resolved)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
