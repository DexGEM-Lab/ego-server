"""Committed GPU topology and per-cluster Ray lifecycle configuration.

Runtime architecture: one Ray cluster per committed physical GPU group. Each cluster
starts with only its assigned physical GPU visible (``CUDA_VISIBLE_DEVICES`` at the
process level before ``ray start``), advertises exactly one native Ray GPU
(``--num-gpus=1``), and each model replica requests ``num_gpus=1`` so Ray owns and
excludes that physical GPU. This replaces the previous custom-resource-only pinning,
which did not give Ray native GPU ownership and could not exclude competing actors.

Each cluster's lifecycle config records: the physical GPU id, a per-cluster
Python interpreter/environment path (different model groups run on different Python
minor versions — GPU0 UniDepth uses its Python 3.11 serving env, while GPU1
Hands, GPU4 WiLoR, GPU2 DROID, and GPU3 HaWoR use model-specific Python 3.10 envs),
disjoint component ports (dashboard/GCS/object)
reserved outside the worker port range, an explicit worker port list, CPU cap, temp
dir, Ray 2.55.1, and the startup command/environment.

A single-node Ray head on Ray 2.55.1 is identified by its GCS address (``--port``),
NOT a cluster name: ``ray start`` has no ``--cluster-name`` option (that flag exists
only on the ``ray up/down/exec/...`` autoscaler launcher). All Serve commands
(``serve deploy/run/status/shutdown``) therefore carry an explicit ``-a`` address
(``dashboard_address`` for deploy/status/shutdown, ``gcs_address`` for run) so a
cutover never silently binds to another cluster via a stray ``RAY_ADDRESS`` env.

Component ports and worker ports are disjoint within and across clusters by
construction: components occupy a contiguous high block (e.g. 26000-26006) while
workers occupy a separate explicit list (e.g. 26100-26131), avoiding Ray's default
worker range 10002-19999 which collided with a GCS port at 16379.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Callable, Sequence


RAY_VERSION = "2.55.1"


@dataclass(frozen=True)
class ClusterPorts:
    """Disjoint component and worker ports for one Ray cluster.

    Component ports are reserved outside the worker port list so Ray's worker
    allocation never collides with GCS/dashboard/object-manager ports. The layout
    matches the verified Ray 2.55.1 ``ray start`` CLI: ``--port`` (GCS),
    ``--object-manager-port``, ``--node-manager-port``, ``--ray-client-server-port``,
    ``--dashboard-port``, ``--dashboard-agent-listen-port``, ``--dashboard-agent-grpc-port``,
    and ``--worker-port-list``. ``serve_http_port`` is consumed at ``serve run`` time,
    not by ``ray start``.
    """

    gcs_port: int                       # --port (head node main / GCS port)
    object_manager_port: int            # --object-manager-port
    node_manager_port: int              # --node-manager-port
    ray_client_server_port: int         # --ray-client-server-port
    dashboard_port: int                 # --dashboard-port
    dashboard_agent_listen_port: int    # --dashboard-agent-listen-port
    dashboard_agent_grpc_port: int      # --dashboard-agent-grpc-port
    metrics_export_port: int             # --metrics-export-port
    autoscaler_metric_port: int          # AUTOSCALER_METRIC_PORT
    dashboard_metric_port: int           # DASHBOARD_METRIC_PORT
    worker_port_list: str               # --worker-port-list (comma-separated)
    serve_http_port: int                # Serve HTTP port used by `serve run --port`

    def all_ports(self) -> tuple[int, ...]:
        components = (
            self.gcs_port,
            self.object_manager_port,
            self.node_manager_port,
            self.ray_client_server_port,
            self.dashboard_port,
            self.dashboard_agent_listen_port,
            self.dashboard_agent_grpc_port,
            self.metrics_export_port,
            self.autoscaler_metric_port,
            self.dashboard_metric_port,
            self.serve_http_port,
        )
        tokens = [token.strip() for token in self.worker_port_list.split(",") if token.strip()]
        if not tokens or any("-" in token for token in tokens):
            raise ValueError("worker_port_list must be a non-empty explicit comma-separated port list")
        try:
            workers = tuple(int(token) for token in tokens)
        except ValueError as exc:
            raise ValueError("worker_port_list contains a non-integer port") from exc
        return components + workers

    def assert_disjoint(self) -> None:
        ports = list(self.all_ports())
        if len(set(ports)) != len(ports):
            raise ValueError(f"cluster ports are not disjoint: {ports}")


@dataclass(frozen=True)
class ClusterLifecycleConfig:
    """Everything needed to start one per-GPU Ray Serve cluster."""

    gpu_id: int
    interpreter: str
    ray_version: str = RAY_VERSION
    num_gpus: int = 1
    num_cpus: int = 4
    temp_dir: str = "/tmp/ray-ego-serve"
    ports: ClusterPorts = field(
        default_factory=lambda: ClusterPorts(
            6379, 6380, 6381, 10001, 8265, 52365, 52366, 52367, 52368, 52369,
            ",".join(str(port) for port in range(26100, 26132)), 8000,
        )
    )
    env_vars: tuple[tuple[str, str], ...] = ()

    @property
    def gcs_address(self) -> str:
        """Explicit head-node GCS address used instead of ambiguous ``auto``."""
        return f"127.0.0.1:{self.ports.gcs_port}"

    @property
    def dashboard_address(self) -> str:
        """Explicit dashboard URL for deploy, status, and shutdown operations."""
        return f"http://127.0.0.1:{self.ports.dashboard_port}"

    def launch_environment_prefix(self) -> str:
        """Shell prefix shared by the Ray head and its detached Serve driver."""
        committed_env = (
            ("AUTOSCALER_METRIC_PORT", str(self.ports.autoscaler_metric_port)),
            ("DASHBOARD_METRIC_PORT", str(self.ports.dashboard_metric_port)),
        ) + self.env_vars
        return " ".join(f"{key}={value}" for key, value in (("CUDA_VISIBLE_DEVICES", str(self.gpu_id)),) + committed_env)

    def startup_command(self, cluster_name: str = "") -> str:
        """The exact ``ray start --head`` command for this cluster.

        ``CUDA_VISIBLE_DEVICES`` is set in the environment before launch so only the
        assigned physical GPU is visible; ``--num-gpus=1`` advertises it as one Ray
        GPU so replicas requesting ``num_gpus=1`` get native ownership.

        ``cluster_name`` is accepted for caller compatibility but not emitted because
        Ray 2.55.1 ``ray start`` has no such flag. The GCS port identifies the head.
        """
        prefix = self.launch_environment_prefix()
        return (
            f"{prefix} {self.interpreter} -m ray.scripts.scripts start --head "
            f"--port={self.ports.gcs_port} "
            f"--object-manager-port={self.ports.object_manager_port} "
            f"--node-manager-port={self.ports.node_manager_port} "
            f"--ray-client-server-port={self.ports.ray_client_server_port} "
            f"--dashboard-port={self.ports.dashboard_port} "
            f"--dashboard-agent-listen-port={self.ports.dashboard_agent_listen_port} "
            f"--dashboard-agent-grpc-port={self.ports.dashboard_agent_grpc_port} "
            f"--metrics-export-port={self.ports.metrics_export_port} "
            f"--worker-port-list={self.ports.worker_port_list} "
            f"--num-gpus={self.num_gpus} --num-cpus={self.num_cpus} "
            f"--temp-dir={self.temp_dir} "
            f"--include-dashboard=true"
        )

    def serve_deploy_command(self, config_path: str) -> str:
        """``serve deploy`` pinned to this cluster's interpreter and explicit dashboard address."""
        return (
            f"{self.interpreter} -m ray.serve.scripts deploy {config_path} "
            f"-a {self.dashboard_address}"
        )

    def serve_run_command(self, import_path: str) -> str:
        """Return the legacy declarative form for production documentation only.

        Ray 2.55 does not accept ``--port`` on ``serve run`` and the command blocks.
        Experimental launches must use ``experimental_driver_command`` below.  The
        production method remains for older runbooks that only print it.
        """
        return f"{self.interpreter} -m ray.serve.scripts run {import_path} -a {self.gcs_address}"

    def experimental_driver_command(
        self, *, release_root: str, app_choice: str, app_name: str = "ego-experiment", route_prefix: str = "/"
    ) -> str:
        """Launch Serve through the detached, non-blocking Ray Python driver."""
        return (
            f"cd {release_root} && PYTHONPATH={release_root} "
            f"{self.interpreter} -m ego_annotation.serving.benchmark.experiment_driver "
            f"--release-root {release_root} --gcs-address {self.gcs_address} "
            f"--http-port {self.ports.serve_http_port} --app-choice {app_choice} "
            f"--app-name {app_name} --route-prefix {route_prefix}"
        )

    def serve_status_command(self) -> str:
        return f"{self.interpreter} -m ray.serve.scripts status -a {self.dashboard_address}"

    def serve_shutdown_command(self) -> str:
        return f"{self.interpreter} -m ray.serve.scripts shutdown -a {self.dashboard_address} -y"

    def assert_gpu_pinned(self) -> None:
        """Mechanically enforce native GPU pinning for this cluster.

        Raises if the GPU id is unset, ``num_gpus != 1``, or the startup command does
        not carry both ``CUDA_VISIBLE_DEVICES=<gpu_id>`` and ``--num-gpus=1``. Called by
        the cutover preflight so a misconfigured cluster cannot reach cutover.
        """
        if self.gpu_id < 0:
            raise ValueError("cluster gpu_id must be a non-negative physical GPU index")
        if self.num_gpus != 1:
            raise ValueError(f"cluster must advertise exactly one native GPU (num_gpus=1), got {self.num_gpus}")
        cmd = self.startup_command()
        if f"CUDA_VISIBLE_DEVICES={self.gpu_id}" not in cmd:
            raise ValueError(f"startup command must pin CUDA_VISIBLE_DEVICES={self.gpu_id}: {cmd}")
        if "--num-gpus=1" not in cmd:
            raise ValueError(f"startup command must advertise --num-gpus=1: {cmd}")


@dataclass(frozen=True)
class GpuServiceGroup:
    """A committed physical GPU group and its resident model API(s).

    ``ray_actor_options`` uses native ``num_gpus=1`` (Ray owns the physical GPU and
    excludes competing replicas) plus the per-cluster interpreter. There is no
    custom-resource-only scheduling label: the GPU is pinned by process-level
    ``CUDA_VISIBLE_DEVICES`` at cluster start and native Ray GPU accounting.
    """

    gpu_id: int
    physical_group: str
    logical_apis: tuple[str, ...]
    adapter_implemented: bool
    interpreter: str
    lifecycle: ClusterLifecycleConfig
    # Empty means this plain Serve application has no FastAPI OpenAPI contract.
    # Non-empty entries name accepted ingress states and their exact route paths.
    openapi_route_states: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def ray_actor_options(self) -> dict[str, Any]:
        # Native GPU ownership: the replica requests the one Ray GPU the cluster
        # advertised. No custom resource label, no CUDA_VISIBLE_DEVICES here (that is
        # set at cluster start, not per-actor).
        return {"num_gpus": 1}


# Disjoint port allocations per physical GPU group. The layout follows the
# verified Ray 2.55.1 ``ray start`` CLI: GCS (--port), object manager, node manager,
# ray client, dashboard, dashboard-agent listen (http), dashboard-agent grpc, then a
# separate explicit comma-separated worker port list, and a Serve HTTP port consumed
# by ``serve run``. Each cluster's ports are disjoint from every other cluster's by
# distinct component bases and distinct worker/serve ranges.
def _ports(
    base_gcs: int,
    base_worker: int,
    *,
    serve_http_port: int,
) -> ClusterPorts:
    worker_ports = ",".join(str(base_worker + i) for i in range(32))
    ports = ClusterPorts(
        gcs_port=base_gcs,
        object_manager_port=base_gcs + 1,
        node_manager_port=base_gcs + 2,
        ray_client_server_port=base_gcs + 3,
        dashboard_port=base_gcs + 4,
        dashboard_agent_listen_port=base_gcs + 5,
        dashboard_agent_grpc_port=base_gcs + 6,
        metrics_export_port=base_gcs + 10,
        autoscaler_metric_port=base_gcs + 11,
        dashboard_metric_port=base_gcs + 12,
        worker_port_list=worker_ports,
        serve_http_port=serve_http_port,
    )
    ports.assert_disjoint()
    return ports


# Each physical group uses its validated model ABI rather than the caller's Python.
_GPU0_INTERPRETER = "/home/zjh/miniconda3/envs/ray_serve_unidepth/bin/python"
_GPU1_INTERPRETER = "/home/zjh/miniconda3/envs/ray_serve_hands/bin/python"
# DROID's compiled lietorch/droid_backends ABI is available only in this env.
_GPU2_INTERPRETER = "/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python"
# GPU3 HaWoR Serve runs in the same Ray-bearing HaWoR/DROID ABI clone. The bare
# `envs/hawor` model env (Torch 1.13.0+cu117) has no Ray installed, so it cannot
# run `ray start`/`serve run`; proven GPU3 Serve evidence (lane 28003, commit
# 271de1a) is in `ray_serve_hawor`.
_GPU3_INTERPRETER = "/home/zjh/miniconda3/envs/ray_serve_hawor/bin/python"
_GPU6_INTERPRETER = "/home/zjh/cosmos3_ray_serve/standalone/.venv/bin/python"

# Immutable server release selected by the atomic ``current`` symlink. Every Ray
# worker imports the same integrated serving package rather than a lane workspace.
RUNTIME_CURRENT_DIR = "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_model_services_runtime/current"
# Standalone cutover consumes these staged artifacts without provisioning or touching
# the bare Cosmos3 vLLM process on GPU6/8001.
STANDALONE_ARTIFACTS_DIR = "/home/zjh/cosmos3_ray_serve/standalone"
COSMOS3_WORKSPACE_DIR = RUNTIME_CURRENT_DIR
COSMOS3_HF_HOME = "/home/ylang/.cache/huggingface"


COMMITTED_GPU_GROUPS = (
    GpuServiceGroup(
        gpu_id=0, physical_group="unidepth", logical_apis=("unidepth.infer",), adapter_implemented=True,
        interpreter=_GPU0_INTERPRETER,
        lifecycle=ClusterLifecycleConfig(
            gpu_id=0, interpreter=_GPU0_INTERPRETER,
            temp_dir="/tmp/ray-ego-serve-gpu0",
            ports=_ports(26000, 26100, serve_http_port=28000),
            env_vars=(
                ("PYTHONPATH", f"{RUNTIME_CURRENT_DIR}:/home/zjh/ego-annation-checkpoints/unidepth_repo"),
                ("EGO_UNIDEPTH_REPO", "/home/zjh/ego-annation-checkpoints/unidepth_repo"),
                ("EGO_UNIDEPTH_CHECKPOINT", "/home/zjh/ego-annation-checkpoints/unidepth/unidepth_v2_vitl14_corrected"),
                ("EGO_UNIDEPTH_REVISION", "unidepth-v2-vitl14-corrected"),
                ("EGO_UNIDEPTH_CANONICAL_H", "540"),
                ("EGO_UNIDEPTH_CANONICAL_W", "960"),
                ("EGO_UNIDEPTH_GPU", "0"),
            ),
        ),
    ),
    GpuServiceGroup(
        gpu_id=1, physical_group="hands", logical_apis=("hands.detect",), adapter_implemented=True,
        interpreter=_GPU1_INTERPRETER,
        openapi_route_states=(
            ("hands-only", ("/hands.detect",)),
            # The combined rollback app remains a valid, stable GPU1 state.
            ("hands-wilor-rollback", ("/hands.detect", "/wilor.reconstruct")),
        ),
        lifecycle=ClusterLifecycleConfig(
            gpu_id=1, interpreter=_GPU1_INTERPRETER,
            temp_dir="/tmp/ray-ego-serve-gpu1",
            ports=_ports(27000, 27100, serve_http_port=28001),
            env_vars=(
                ("PYTHONPATH", f"{RUNTIME_CURRENT_DIR}:/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/sam2:/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor_model"),
                ("EGO_SAM2_REPO", "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/sam2"),
                ("EGO_WILOR_REPO", "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor_model"),
                ("EGO_HANDS_GPU", "1"),
                ("EGO_HANDS_REVISION", "hands-yolo-v2"),
                ("EGO_MODEL_CLUSTER_ID", "ego-hands-gpu1-27000"),
                ("EGO_MODEL_GCS_ADDRESS", "127.0.0.1:27000"),
                ("EGO_MODEL_TEMP_DIR", "/tmp/ray-ego-serve-gpu1"),
                ("EGO_HANDS_EXPERIMENT_WIRE_FORMAT", "multipart"),
                ("EGO_WILOR_EXPERIMENT_WIRE_FORMAT", "multipart"),
                ("EGO_HANDS_EXPERIMENT_TELEMETRY", "0"),
                ("EGO_WILOR_EXPERIMENT_TELEMETRY", "0"),
            ),
        ),
    ),
    GpuServiceGroup(
        gpu_id=4, physical_group="wilor", logical_apis=("wilor.reconstruct",), adapter_implemented=True,
        interpreter=_GPU1_INTERPRETER,
        openapi_route_states=(("wilor-only", ("/wilor.reconstruct",)),),
        lifecycle=ClusterLifecycleConfig(
            gpu_id=4, interpreter=_GPU1_INTERPRETER,
            temp_dir="/tmp/ray-ego-serve-gpu4-wilor",
            ports=_ports(27200, 27300, serve_http_port=28004),
            env_vars=(
                ("PYTHONPATH", f"{RUNTIME_CURRENT_DIR}:/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/sam2:/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor_model"),
                ("EGO_SAM2_REPO", "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/sam2"),
                ("EGO_WILOR_REPO", "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor_model"),
                # WiLoR's existing provenance field retains this historical name.
                ("EGO_HANDS_GPU", "4"),
                ("EGO_WILOR_REVISION", "wilor-final-v1"),
                ("EGO_MODEL_CLUSTER_ID", "ego-wilor-gpu4-27200"),
                ("EGO_MODEL_GCS_ADDRESS", "127.0.0.1:27200"),
                ("EGO_MODEL_TEMP_DIR", "/tmp/ray-ego-serve-gpu4-wilor"),
                ("EGO_HANDS_EXPERIMENT_WIRE_FORMAT", "multipart"),
                ("EGO_WILOR_EXPERIMENT_WIRE_FORMAT", "multipart"),
                ("EGO_HANDS_EXPERIMENT_TELEMETRY", "0"),
                ("EGO_WILOR_EXPERIMENT_TELEMETRY", "0"),
            ),
        ),
    ),
    GpuServiceGroup(
        gpu_id=2, physical_group="droid", logical_apis=("droid.create_session", "droid.push_frame", "droid.finalize"), adapter_implemented=True,
        interpreter=_GPU2_INTERPRETER,
        lifecycle=ClusterLifecycleConfig(
            gpu_id=2, interpreter=_GPU2_INTERPRETER,
            temp_dir="/tmp/ray-ego-serve-gpu2",
            ports=_ports(26400, 26500, serve_http_port=28002),
            env_vars=(
                ("PYTHONPATH", RUNTIME_CURRENT_DIR),
                ("EGO_DROID_REPO", "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/DROID-SLAM/droid_slam"),
                ("EGO_DROID_WEIGHTS", "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/droid/droid.pth"),
                ("EGO_DROID_REVISION", "droid-v1"),
                ("EGO_DROID_GPU", "2"),
                ("EGO_DROID_DEVICE", "cuda:0"),
            ),
        ),
    ),
    GpuServiceGroup(
        gpu_id=3, physical_group="hawor", logical_apis=("hawor.infer_tracks", "hawor_infiller.fill"), adapter_implemented=True,
        interpreter=_GPU3_INTERPRETER,
        lifecycle=ClusterLifecycleConfig(
            gpu_id=3, interpreter=_GPU3_INTERPRETER,
            temp_dir="/tmp/ray-ego-serve-gpu3",
            ports=_ports(26600, 26700, serve_http_port=28003),
            env_vars=(
                ("PYTHONPATH", RUNTIME_CURRENT_DIR),
                ("EGO_HAWOR_REPO", "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/HaWoR"),
                ("EGO_HAWOR_CHECKPOINT", "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/hawor.ckpt"),
                ("EGO_HAWOR_INFILLER_CHECKPOINT", "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/hawor/infiller.pt"),
                ("EGO_HAWOR_GPU", "3"),
                # Default wire remains multipart. A benchmark treatment must set the
                # matching endpoint env before launch so the result digest attributes it.
                ("EGO_HAWOR_EXPERIMENT_WIRE_FORMAT", "multipart"),
                ("EGO_HAWOR_INFILLER_EXPERIMENT_WIRE_FORMAT", "multipart"),
                ("EGO_HAWOR_EXPERIMENT_TELEMETRY", "0"),
                ("EGO_HAWOR_INFILLER_EXPERIMENT_TELEMETRY", "0"),
            ),
        ),
    ),
    GpuServiceGroup(
        gpu_id=6, physical_group="cosmos3", logical_apis=("cosmos3.reason",), adapter_implemented=True,
        interpreter=_GPU6_INTERPRETER,
        lifecycle=ClusterLifecycleConfig(
            gpu_id=6,
            interpreter=_GPU6_INTERPRETER,
            num_cpus=8,
            temp_dir="/tmp/ray-ego-serve-cosmos3",
            ports=ClusterPorts(
                gcs_port=26801,
                object_manager_port=26802,
                node_manager_port=26803,
                ray_client_server_port=26804,
                dashboard_port=26800,
                dashboard_agent_listen_port=26805,
                dashboard_agent_grpc_port=26806,
                metrics_export_port=26810,
                autoscaler_metric_port=26811,
                dashboard_metric_port=26812,
                worker_port_list=",".join(str(port) for port in range(26900, 26932)),
                serve_http_port=28006,
            ),
            env_vars=(("PYTHONPATH", COSMOS3_WORKSPACE_DIR), ("HF_HOME", COSMOS3_HF_HOME)),
        ),
    ),
)


def committed_serve_http_ports() -> tuple[int, ...]:
    """Canonical public lanes derived from the committed topology."""
    return tuple(sorted(group.lifecycle.ports.serve_http_port for group in COMMITTED_GPU_GROUPS))


def unidepth_gpu_group() -> GpuServiceGroup:
    return next(group for group in COMMITTED_GPU_GROUPS if group.gpu_id == 0)


def droid_gpu_group() -> GpuServiceGroup:
    """The GPU2 stateful DROID group (one resident DroidNet, isolated sessions)."""
    return next(group for group in COMMITTED_GPU_GROUPS if group.gpu_id == 2)


def hawor_gpu_group() -> GpuServiceGroup:
    """The GPU3 HaWoR + infiller group (one shared physical resident lane)."""
    return next(group for group in COMMITTED_GPU_GROUPS if group.gpu_id == 3)


def cosmos3_gpu_group() -> GpuServiceGroup:
    """The GPU6 Cosmos3 group (the second implemented persistent deployment)."""
    return next(group for group in COMMITTED_GPU_GROUPS if group.gpu_id == 6)


# Serve HTTP port for the GPU6 Cosmos3 cluster. Disjoint from the component/worker
# port ranges (26800-26806 / 26900-26931) per the committed topology.
COSMOS3_SERVE_HTTP_PORT = 28006


def cosmos3_serve_config() -> dict[str, Any]:
    """Serve config for the persistent GPU6 Cosmos3 deployment.

    vLLM owns continuous batching inside the resident engine, so the Serve layer does
    NOT use ``@serve.batch``; ``max_ongoing_requests``/``max_queued_requests`` bound the
    admission queue and the vLLM engine capacity bounds the running batch. The HTTP
    endpoint is on port 28006 (disjoint from the cluster component/worker ports).
    The local GPU6 Ray head propagates the curated workspace through PYTHONPATH;
    declarative Serve configs reject it as a local runtime_env.working_dir.
    """
    group = cosmos3_gpu_group()
    return {
        "http_options": {"host": "0.0.0.0", "port": COSMOS3_SERVE_HTTP_PORT},
        "applications": [
            {
                "name": "cosmos3",
                "route_prefix": "/",
                "import_path": "ego_annotation.serving.cosmos3_deployment:app",
                "deployment_config": {
                    "ray_actor_options": group.ray_actor_options,
                },
                "deployments": [
                    {
                        "name": "cosmos3.reason",
                        "num_replicas": 1,
                        "ray_actor_options": group.ray_actor_options,
                        # vLLM owns the running batch; these bound admission only.
                        "max_ongoing_requests": 16,
                        "max_queued_requests": 32,
                    }
                ],
            }
        ],
    }


def unidepth_serve_config() -> dict[str, Any]:
    """Serve config for the one implemented persistent GPU0 deployment.

    The import path points at the deployment-only module that exports a bound Ray
    Serve Application; ``serve run``/``serve deploy`` resolve it against the GPU0
    cluster's ``ray_serve_unidepth`` interpreter. The Serve HTTP port
    (``serve_http_port``) is consumed at ``serve run`` time.
    """
    group = unidepth_gpu_group()
    return {
        "applications": [
            {
                "name": "ego-model-services",
                "route_prefix": "/",
                "import_path": "ego_annotation.serving.deployment:app",
                "deployment_config": {
                    "ray_actor_options": group.ray_actor_options,
                },
                "deployments": [
                    {
                        "name": "unidepth.infer",
                        "num_replicas": 1,
                        "ray_actor_options": group.ray_actor_options,
                        "max_ongoing_requests": 16,
                        "max_queued_requests": 64,
                    }
                ],
            }
        ]
    }


def droid_serve_config() -> dict[str, Any]:
    """Serve config for the persistent GPU2 stateful DROID deployment.

    The deployment runs in the GPU2 ``ray_serve_hawor`` interpreter, whose native
    DROID extensions are ABI-compatible. One replica owns one resident DroidNet
    while retaining isolated typed sessions and the shared native-GPU lifecycle.
    """
    group = droid_gpu_group()
    return {
        "applications": [
            {
                "name": "ego-droid-service",
                "route_prefix": "/",
                "import_path": "ego_annotation.serving.droid_deployment:app",
                "deployment_config": {
                    "ray_actor_options": group.ray_actor_options,
                },
                "deployments": [
                    {
                        "name": "droid",
                        "num_replicas": 1,
                        "ray_actor_options": group.ray_actor_options,
                        "max_ongoing_requests": 32,
                        "max_queued_requests": 64,
                    }
                ],
            }
        ]
    }


def hands_gpu_group() -> GpuServiceGroup:
    return next(group for group in COMMITTED_GPU_GROUPS if group.gpu_id == 1)


def wilor_gpu_group() -> GpuServiceGroup:
    return next(group for group in COMMITTED_GPU_GROUPS if group.gpu_id == 4)


def hands_wilor_gpu_group() -> GpuServiceGroup:
    """Compatibility accessor for the GPU1 rollback topology owner."""
    return hands_gpu_group()


def hands_wilor_serve_config() -> dict[str, Any]:
    """Serve config for the GPU1 hands.detect + wilor.reconstruct deployment.

    One replica (``num_gpus=1``) hosts both logical APIs with separate batch queues.
    The HTTP proxy listens on port 28001 (``serve run --port 28001``). The import
    path points at the deployment-only module that exports a bound Ray Serve
    Application; resolved against the GPU1 cluster's ``ray_serve_hands`` interpreter.
    """
    group = hands_gpu_group()
    return {
        "applications": [
            {
                "name": "ego-hands-wilor",
                "route_prefix": "/",
                "import_path": "ego_annotation.serving.hands_deployment:hands_app",
                "deployment_config": {
                    "ray_actor_options": group.ray_actor_options,
                },
                "deployments": [
                    {
                        "name": "hands_wilor",
                        "num_replicas": 1,
                        "ray_actor_options": group.ray_actor_options,
                        "max_ongoing_requests": 32,
                        "max_queued_requests": 128,
                    }
                ],
            }
        ],
    }


def hands_only_serve_config() -> dict[str, Any]:
    """Serve config for GPU1's exclusive Hands detector application."""
    group = hands_gpu_group()
    return {
        "applications": [{
            "name": "ego-hands",
            "route_prefix": "/",
            "import_path": "ego_annotation.serving.hands_deployment:hands_only_app",
            "deployment_config": {"ray_actor_options": group.ray_actor_options},
            "deployments": [{
                "name": "hands", "num_replicas": 1,
                "ray_actor_options": group.ray_actor_options,
                "max_ongoing_requests": 32, "max_queued_requests": 128,
            }],
        }],
    }


def wilor_only_serve_config() -> dict[str, Any]:
    """Serve config for GPU4's exclusive WiLoR reconstruction application."""
    group = wilor_gpu_group()
    return {
        "applications": [{
            "name": "ego-wilor",
            "route_prefix": "/",
            "import_path": "ego_annotation.serving.hands_deployment:wilor_only_app",
            "deployment_config": {"ray_actor_options": group.ray_actor_options},
            "deployments": [{
                "name": "wilor", "num_replicas": 1,
                "ray_actor_options": group.ray_actor_options,
                "max_ongoing_requests": 32, "max_queued_requests": 128,
            }],
        }],
    }


# --- Cosmos3 cutover target wiring ------------------------------------------
#
# Explicit addresses for the GPU6 cutover. The bare Cosmos3 endpoint on GPU6/port
# 8001 is the production baseline. This code never stops it. Because the bare vLLM
# process occupies GPU6 memory, an authorized operator must stop it before a guarded
# candidate can load on the disjoint Serve HTTP port 28006. The guard records
# readiness only after candidate health/equivalence evidence; it never moves callers.
COSMOS3_GCS_ADDRESS = cosmos3_gpu_group().lifecycle.gcs_address
COSMOS3_DASHBOARD_ADDRESS = cosmos3_gpu_group().lifecycle.dashboard_address
# The bare endpoint is labeled BASELINE only: any measurement against it is a
# pre-cutover baseline, never Ray-managed evidence.
COSMOS3_BARE_BASELINE_URL = "http://127.0.0.1:8001"
COSMOS3_RAY_MANAGED_URL = f"http://127.0.0.1:{COSMOS3_SERVE_HTTP_PORT}"


def cosmos3_lifecycle() -> ClusterLifecycleConfig:
    """The GPU6 cluster lifecycle config (explicit-address, GPU6-pinned)."""
    return cosmos3_gpu_group().lifecycle


@dataclass(frozen=True)
class ClusterOwnership:
    """Exact process identity for a committed independent Ray head."""

    cluster_id: str
    gcs_address: str
    dashboard_address: str
    temp_dir: str
    ray_client_server_port: int
    interpreter: str


def cluster_ownership(cluster_id: str) -> ClusterOwnership:
    """Resolve one allowlisted split cluster; unknown IDs cannot be stopped."""
    for group in COMMITTED_GPU_GROUPS:
        env = dict(group.lifecycle.env_vars)
        if env.get("EGO_MODEL_CLUSTER_ID") == cluster_id:
            return ClusterOwnership(
                cluster_id=cluster_id,
                gcs_address=group.lifecycle.gcs_address,
                dashboard_address=group.lifecycle.dashboard_address,
                temp_dir=group.lifecycle.temp_dir,
                ray_client_server_port=group.lifecycle.ports.ray_client_server_port,
                interpreter=group.lifecycle.interpreter,
            )
    raise ValueError(f"unknown model cluster ID {cluster_id!r}")


@dataclass(frozen=True, order=True)
class ProcessIdentity:
    pid: int
    starttime_ticks: int


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    ppid: int
    pgid: int
    sid: int
    starttime_ticks: int
    command: bytes
    environment: frozenset[bytes]

    @property
    def identity(self) -> ProcessIdentity:
        return ProcessIdentity(self.pid, self.starttime_ticks)

    @property
    def signature(self) -> tuple[int, int]:
        return self.pgid, self.sid


@dataclass(frozen=True)
class ClusterMembershipSnapshot:
    signature: tuple[int, int]
    session_root: str
    root_identities: tuple[ProcessIdentity, ...]
    member_identities: tuple[ProcessIdentity, ...]
    outsider_identities: tuple[ProcessIdentity, ...]


def _process_records(proc_root: str | Path = "/proc") -> dict[int, ProcessRecord]:
    """Capture identity-bearing `/proc` evidence for one scoped generation."""
    records: dict[int, ProcessRecord] = {}
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
            stat_tail = (entry / "stat").read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
            ppid, pgid, sid = (int(stat_tail[index]) for index in (1, 2, 3))
            # Starttime is field 22, index 19 after stripping the comm field. Old
            # synthetic records omit it; zero remains an identity value in tests.
            starttime = int(stat_tail[19]) if len(stat_tail) > 19 else 0
            environment = frozenset(
                item for item in (entry / "environ").read_bytes().split(b"\0")
                if item.startswith(b"EGO_MODEL_")
            )
        except (OSError, IndexError, ValueError):
            continue
        pid = int(entry.name)
        records[pid] = ProcessRecord(pid, ppid, pgid, sid, starttime, command, environment)
    return records


def _ancestor_chain(pid: int, records: dict[int, ProcessRecord]) -> tuple[int, ...]:
    chain: list[int] = []
    while pid in records and pid not in chain:
        chain.append(pid)
        pid = records[pid].ppid
    return tuple(chain)


def _has_exact_temp_dir(command: bytes, temp_dir: bytes) -> bool:
    """Match a temp root as a command argument/value, never as a string prefix."""
    for token in command.replace(b"\0", b" ").split():
        value = token.split(b"=", 1)[-1]
        if value == temp_dir or value.startswith(temp_dir + b"/"):
            return True
    return False


def _exact_argument(command: bytes, names: Sequence[bytes], value: bytes) -> bool:
    tokens = command.replace(b"\0", b" ").split()
    for index, token in enumerate(tokens):
        if token in names and index + 1 < len(tokens) and tokens[index + 1] == value:
            return True
        if any(token.startswith(name + b"=") and token.split(b"=", 1)[1] == value for name in names):
            return True
    return False


def _extract_ray_session_dir(command: bytes, temp_dir: bytes) -> str | None:
    for token in command.replace(b"\0", b" ").split():
        value = token.split(b"=", 1)[-1].replace(b"\\", b"/")
        prefix = temp_dir + b"/"
        if value.startswith(prefix):
            component = value[len(prefix):].split(b"/", 1)[0]
            if component.startswith(b"session"):
                return (prefix + component).decode("utf-8", "surrogateescape")
    return None


def _has_exact_gcs_server_port(command: bytes, gcs_port: bytes) -> bool:
    """Recognize only exact Ray GCS-server port arguments, never digit substrings."""
    return _exact_argument(command, (b"--gcs_server_port",), gcs_port)


_NATIVE_RAY_EXECUTABLES = frozenset({b"gcs-server", b"raylet"})
_APPROVED_RAY_MODULES = frozenset({
    b"ray._private.monitor", b"ray._private.log_monitor", b"ray.dashboard.dashboard",
    b"ray.dashboard.agent", b"ray._private.runtime_env.agent.main",
    b"ray.autoscaler._private.monitor", b"ray.util.client.server",
})
_APPROVED_RAY_SCRIPT_SUFFIXES = (
    b"/ray/_private/monitor.py", b"/ray/_private/log_monitor.py",
    b"/ray/dashboard/dashboard.py", b"/ray/dashboard/agent.py",
    b"/ray/_private/runtime_env/agent/main.py", b"/ray/autoscaler/_private/monitor.py",
)
_PYTHON_NO_VALUE_FLAGS = frozenset({b"-u"})


def _command_tokens(command: bytes) -> list[bytes]:
    return command.lower().replace(b"\0", b" ").split()


def _is_environment_assignment(token: bytes) -> bool:
    return not token.startswith(b"-") and b"=" in token


def _invocation_index(tokens: Sequence[bytes]) -> int | None:
    """Locate argv0 after deliberate direct/env assignment handling."""
    index = 0
    while index < len(tokens) and _is_environment_assignment(tokens[index]):
        index += 1
    if index == len(tokens):
        return None
    executable = tokens[index].replace(b"\\", b"/").rsplit(b"/", 1)[-1]
    if executable != b"env":
        return index
    index += 1
    while index < len(tokens):
        token = tokens[index]
        if token == b"--":
            return index + 1 if index + 1 < len(tokens) else None
        if token == b"-u":
            index += 2
        elif token.startswith(b"-u") and len(token) > 2:
            index += 1
        elif _is_environment_assignment(token):
            index += 1
        elif token.startswith(b"-"):
            return None
        else:
            return index
    return None


def _is_python_executable(token: bytes) -> bool:
    name = token.replace(b"\\", b"/").rsplit(b"/", 1)[-1]
    return name == b"python" or name == b"python3" or name.startswith(b"python3.")


def _python_operand(command: bytes) -> tuple[str, bytes] | None:
    tokens = _command_tokens(command)
    invocation = _invocation_index(tokens)
    if invocation is None or not _is_python_executable(tokens[invocation]):
        return None
    index = invocation + 1
    while index < len(tokens) and tokens[index] in _PYTHON_NO_VALUE_FLAGS:
        index += 1
    if index >= len(tokens):
        return None
    if tokens[index] == b"-m":
        return ("module", tokens[index + 1]) if index + 1 < len(tokens) else None
    return None if tokens[index].startswith(b"-") else ("script", tokens[index].replace(b"\\", b"/"))


def _is_approved_ray_script(path: bytes) -> bool:
    return any(path.endswith(suffix) or path == suffix.lstrip(b"/") for suffix in _APPROVED_RAY_SCRIPT_SUFFIXES)


def _native_role(command: bytes) -> tuple[bytes, bytes] | None:
    tokens = _command_tokens(command)
    invocation = _invocation_index(tokens)
    if invocation is None:
        return None
    executable = tokens[invocation].replace(b"\\", b"/").rsplit(b"/", 1)[-1]
    return executable, executable.removesuffix(b".py").replace(b"_", b"-")


def _is_ray_process(command: bytes) -> bool:
    """Classify only executable/module/script invocation, never later argv values."""
    native = _native_role(command)
    if native is None:
        return False
    executable, native_role = native
    if executable.startswith(b"ray::") or native_role in _NATIVE_RAY_EXECUTABLES:
        return True
    operand = _python_operand(command)
    return operand is not None and (
        operand[0] == "module" and operand[1] in _APPROVED_RAY_MODULES
        or operand[0] == "script" and _is_approved_ray_script(operand[1])
    )


def _required_ownership_markers(ownership: ClusterOwnership) -> frozenset[bytes]:
    return frozenset({
        f"EGO_MODEL_CLUSTER_ID={ownership.cluster_id}".encode(),
        f"EGO_MODEL_GCS_ADDRESS={ownership.gcs_address}".encode(),
        f"EGO_MODEL_TEMP_DIR={ownership.temp_dir}".encode(),
    })


def _is_exact_module(command: bytes, module: bytes) -> bool:
    return _python_operand(command) == ("module", module)


def _is_exact_script(command: bytes, suffix: bytes) -> bool:
    operand = _python_operand(command)
    return operand is not None and operand[0] == "script" and (operand[1].endswith(suffix) or operand[1] == suffix.lstrip(b"/"))


def _is_reparented_daemon_role(command: bytes) -> bool:
    return (
        _is_exact_module(command, b"ray.autoscaler._private.monitor")
        or _is_exact_script(command, b"/ray/autoscaler/_private/monitor.py")
        or _is_exact_module(command, b"ray.util.client.server")
    )


def _strict_cluster_seeds(
    ownership: ClusterOwnership, records: dict[int, ProcessRecord],
) -> tuple[tuple[ProcessRecord, ...], tuple[int, int], str]:
    required, temp = _required_ownership_markers(ownership), ownership.temp_dir.encode()
    invoker, invoker_chain = os.getpid(), set(_ancestor_chain(os.getpid(), records))
    seeds = tuple(record for record in records.values() if (
        record.pid != 1 and record.pid not in invoker_chain and invoker not in _ancestor_chain(record.pid, records)
        and required.issubset(record.environment) and _is_ray_process(record.command)
        and not _is_reparented_daemon_role(record.command)
        and _has_exact_temp_dir(record.command, temp) and _extract_ray_session_dir(record.command, temp) is not None
    ))
    if not seeds:
        raise ValueError("marker-era membership found no fully owned strict seeds")
    signatures, sessions = {record.signature for record in seeds}, {
        _extract_ray_session_dir(record.command, temp) for record in seeds
    }
    if len(signatures) != 1 or len(sessions) != 1 or None in sessions:
        raise ValueError("marker-era strict seeds are generation-ambiguous")
    signature, session_root = next(iter(signatures)), next(iter(sessions))
    if signature[0] in (0, 1) or signature[1] in (0, 1):
        raise ValueError("marker-era strict seed signature is trivial")
    invoker_record = records.get(invoker)
    if invoker_record is not None and invoker_record.signature == signature:
        raise ValueError("marker-era signature overlaps lifecycle invoker")
    gcs_port = ownership.gcs_address.rsplit(":", 1)[1].encode()
    gcs = [record for record in seeds if _native_role(record.command) == (b"gcs_server", b"gcs-server") and _has_exact_gcs_server_port(record.command, gcs_port)]
    raylets = [record for record in seeds if _native_role(record.command) == (b"raylet", b"raylet")]
    if len(gcs) != 1 or len(raylets) != 1:
        raise ValueError("marker-era seeds require exactly one GCS and one raylet")
    for record in records.values():
        if record in seeds or not _is_ray_process(record.command):
            continue
        cluster_markers = [item for item in record.environment if item.startswith(b"EGO_MODEL_CLUSTER_ID=")]
        if cluster_markers and cluster_markers[0] != f"EGO_MODEL_CLUSTER_ID={ownership.cluster_id}".encode():
            other_session = _extract_ray_session_dir(record.command, temp)
            if record.signature == signature or other_session == session_root:
                raise ValueError("marker-era sibling overlaps target generation")
    return seeds, signature, session_root


def _corroborated_daemon_roots(
    ownership: ClusterOwnership, records: dict[int, ProcessRecord], signature: tuple[int, int], session_root: str,
) -> tuple[ProcessRecord, ...]:
    required, temp = _required_ownership_markers(ownership), ownership.temp_dir.encode()
    address, client_port = ownership.gcs_address.encode(), str(ownership.ray_client_server_port).encode()
    roots: list[ProcessRecord] = []
    for record in records.values():
        if not (required.issubset(record.environment) and record.signature == signature):
            continue
        if _extract_ray_session_dir(record.command, temp) != session_root:
            continue
        has_address = _exact_argument(record.command, (b"--gcs-address", b"--address"), address)
        monitor = _is_exact_module(record.command, b"ray.autoscaler._private.monitor") or _is_exact_script(record.command, b"/ray/autoscaler/_private/monitor.py")
        client = _is_exact_module(record.command, b"ray.util.client.server")
        if monitor and has_address:
            roots.append(record)
        if client and has_address and _exact_argument(record.command, (b"--port", b"--ray-client-server-port"), client_port):
            roots.append(record)
    return tuple(roots)


def _descendant_closure(
    records: dict[int, ProcessRecord], roots: Sequence[ProcessRecord], signature: tuple[int, int],
) -> tuple[ProcessRecord, ...]:
    members = {record.pid: record for record in roots}
    changed = True
    while changed:
        changed = False
        for record in records.values():
            if record.pid not in members and record.signature == signature and record.ppid in members:
                members[record.pid] = record
                changed = True
    return tuple(members.values())


def _owner_sleep_outsider(record: ProcessRecord, roots: Sequence[ProcessRecord]) -> bool:
    return (
        record.pid == record.pgid == record.sid and record.command.strip() == b"sleep infinity"
        and not record.environment and record.pid not in {root.pid for root in roots}
    )


def _cluster_membership_snapshot(
    ownership: ClusterOwnership, *, proc_root: str | Path = "/proc",
) -> ClusterMembershipSnapshot:
    records = _process_records(proc_root)
    seeds, signature, session_root = _strict_cluster_seeds(ownership, records)
    roots = tuple({record.pid: record for record in (*seeds, *_corroborated_daemon_roots(ownership, records, signature, session_root))}.values())
    members = _descendant_closure(records, roots, signature)
    member_pids = {record.pid for record in members}
    outsiders = tuple(record for record in records.values() if record.signature == signature and record.pid not in member_pids)
    if len(outsiders) != 1 or not _owner_sleep_outsider(outsiders[0], roots):
        raise ValueError("marker-era generation has an unowned same-signature outsider")
    return ClusterMembershipSnapshot(
        signature, session_root,
        tuple(sorted((record.identity for record in roots))),
        tuple(sorted((record.identity for record in members))),
        tuple(sorted(record.identity for record in outsiders)),
    )


def _candidate_cluster_pids(ownership: ClusterOwnership, *, proc_root: str | Path = "/proc") -> tuple[int, ...]:
    """Compatibility projection of complete authenticated marker-era membership."""
    return tuple(identity.pid for identity in _cluster_membership_snapshot(ownership, proc_root=proc_root).member_identities)


def _legacy_gpu1_process_pids(ownership: ClusterOwnership, *, proc_root: str | Path = "/proc") -> tuple[int, ...]:
    """Find only the pre-marker GPU1 Ray tree from exact temp/GCS evidence."""
    if ownership.cluster_id != "ego-hands-gpu1-27000":
        raise ValueError("legacy teardown is allowlisted only for the GPU1 Hands/WiLoR cluster")
    records, temp = _process_records(proc_root), ownership.temp_dir.encode()
    gcs_port, invoking_pid = ownership.gcs_address.rsplit(":", 1)[1].encode(), os.getpid()
    ray_pids = {pid for pid, record in records.items() if pid != invoking_pid and _is_ray_process(record.command)}
    temp_pids = {pid for pid in ray_pids if _has_exact_temp_dir(records[pid].command, temp)}
    gcs_pids = {pid for pid in ray_pids if _has_exact_gcs_server_port(records[pid].command, gcs_port)}
    roots: set[int] = set()
    for temp_pid in temp_pids:
        for gcs_pid in gcs_pids:
            common = {pid for pid in set(_ancestor_chain(temp_pid, records)) & set(_ancestor_chain(gcs_pid, records)) if pid != 1 and pid in ray_pids}
            if common:
                roots.add(min(common, key=lambda pid: _ancestor_chain(temp_pid, records).index(pid) + _ancestor_chain(gcs_pid, records).index(pid)))
    return tuple(sorted(pid for pid in ray_pids if any(root in _ancestor_chain(pid, records) for root in roots)))


@dataclass(frozen=True)
class ScopedStopResult:
    """Auditable outcome of a bounded shutdown followed by scoped PID teardown."""

    stopped_pids: tuple[int, ...]
    serve_shutdown_returncode: int | None
    serve_shutdown_stdout: str
    serve_shutdown_stderr: str
    serve_shutdown_timed_out: bool
    legacy_gpu1: bool
    term_pids: tuple[int, ...] = ()
    kill_pids: tuple[int, ...] = ()
    final_survivors: tuple[int, ...] = ()
    generation_signature: tuple[int, int] | None = None


def _identity_is_live(identity: ProcessIdentity, records: dict[int, ProcessRecord], signature: tuple[int, int]) -> bool:
    record = records.get(identity.pid)
    return record is not None and record.identity == identity and record.signature == signature


def _identity_is_present(identity: ProcessIdentity, records: dict[int, ProcessRecord]) -> bool:
    record = records.get(identity.pid)
    return record is not None and record.identity == identity


def _child_first(identities: Sequence[ProcessIdentity], records: dict[int, ProcessRecord]) -> tuple[ProcessIdentity, ...]:
    member_pids = {identity.pid for identity in identities}

    def depth(identity: ProcessIdentity) -> int:
        value, pid, seen = 0, identity.pid, set()
        while pid in records and pid not in seen:
            seen.add(pid)
            parent = records[pid].ppid
            if parent not in member_pids:
                break
            value += 1
            pid = parent
        return value

    return tuple(sorted(identities, key=lambda identity: (-depth(identity), identity.pid)))


def _post_shutdown_members(
    snapshot: ClusterMembershipSnapshot,
    frozen: Sequence[ProcessIdentity],
    records: dict[int, ProcessRecord],
) -> tuple[ProcessIdentity, ...]:
    """Preserve frozen identities through reparenting and add only causal descendants."""
    expected_by_pid = {identity.pid: identity for identity in frozen}
    known = {
        identity.pid: records[identity.pid]
        for identity in frozen
        if _identity_is_live(identity, records, snapshot.signature)
    }
    changed = True
    while changed:
        changed = False
        for record in records.values():
            # A PID that changed starttime is a new unrelated process, never a
            # descendant continuation of the frozen generation.
            if record.pid in expected_by_pid and record.identity != expected_by_pid[record.pid]:
                continue
            if record.pid not in known and record.signature == snapshot.signature and record.ppid in known:
                known[record.pid] = record
                changed = True
    root_pids = {identity.pid for identity in snapshot.root_identities}
    outsiders = [
        record for record in records.values()
        if record.signature == snapshot.signature and record.pid not in known
        and not (record.pid == record.pgid == record.sid and record.command.strip() == b"sleep infinity"
                 and not record.environment and record.pid not in root_pids)
    ]
    if outsiders:
        raise RuntimeError("post-shutdown generation gained an unlinked same-signature process")
    return tuple(sorted(record.identity for record in known.values()))


def _listener_pids(port: int, proc_root: str | Path) -> tuple[int, ...]:
    """Best-effort listener census; unavailable proc data is conservatively empty."""
    root, inodes = Path(proc_root), set()
    for table in (root / "net" / "tcp", root / "net" / "tcp6"):
        try:
            lines = table.read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) > 9 and fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(fields[9])
    pids: set[int] = set()
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            for fd in (entry / "fd").iterdir():
                if fd.resolve().name.removeprefix("socket:[").removesuffix("]") in inodes:
                    pids.add(int(entry.name))
        except OSError:
            continue
    return tuple(sorted(pids))


def _serve_shutdown(
    ownership: ClusterOwnership, *, run: Callable[..., Any], timeout_s: float,
) -> tuple[int | None, str, str, bool]:
    command = [ownership.interpreter, "-m", "ray.serve.scripts", "shutdown", "-a", ownership.dashboard_address, "-y"]
    try:
        completed = run(command, check=False, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return None, stdout, stderr, True
    return completed.returncode, completed.stdout or "", completed.stderr or "", False


def scoped_stop_cluster(
    *,
    cluster_id: str,
    temp_dir: str,
    gcs_address: str,
    dashboard_address: str,
    legacy_gpu1: bool = False,
    timeout_s: float = 20.0,
    term_grace_s: float = 1.0,
    run: Callable[..., Any] = subprocess.run,
    pid_lookup: Callable[[ClusterOwnership], Sequence[int]] = _candidate_cluster_pids,
    legacy_pid_lookup: Callable[[ClusterOwnership], Sequence[int]] = _legacy_gpu1_process_pids,
    kill: Callable[[int, int], None] = os.kill,
    proc_root: str | Path = "/proc",
    wait: Callable[[float], None] = time.sleep,
) -> ScopedStopResult:
    """Shutdown one allowlisted Ray generation; never signal a process group.

    The default marker-era path authenticates and freezes a complete generation before
    Serve shutdown. Legacy/custom lookup injection remains a narrow compatibility path
    for existing pre-marker operations and tests.
    """
    ownership = cluster_ownership(cluster_id)
    supplied = (temp_dir, gcs_address, dashboard_address)
    expected = (ownership.temp_dir, ownership.gcs_address, ownership.dashboard_address)
    if supplied != expected:
        raise ValueError(f"refusing mismatched scoped-stop target for {cluster_id}: {supplied!r}")
    if legacy_gpu1 and ownership.cluster_id != "ego-hands-gpu1-27000":
        raise ValueError("--legacy-gpu1 is allowlisted only for ego-hands-gpu1-27000")

    # Legacy discovery remains narrow, but production legacy teardown freezes its
    # ProcessRecord identities too; custom lookup injection is retained for tests.
    if legacy_gpu1 and legacy_pid_lookup is _legacy_gpu1_process_pids:
        initial = _process_records(proc_root)
        pids = tuple(pid for pid in legacy_pid_lookup(ownership) if pid != os.getpid())
        identities = tuple(initial[pid].identity for pid in pids if pid in initial)
        if not identities:
            raise ValueError("legacy GPU1 teardown found no corroborated live process tree")
        returncode, stdout, stderr, timed_out = _serve_shutdown(ownership, run=run, timeout_s=timeout_s)
        term_pids: list[int] = []
        for identity in _child_first(identities, initial):
            if _identity_is_present(identity, _process_records(proc_root)):
                try:
                    kill(identity.pid, signal.SIGTERM)
                    term_pids.append(identity.pid)
                except ProcessLookupError:
                    pass
        wait(term_grace_s)
        after_term = _process_records(proc_root)
        survivors = tuple(identity for identity in identities if _identity_is_present(identity, after_term))
        kill_pids: list[int] = []
        for identity in _child_first(survivors, after_term):
            if _identity_is_present(identity, _process_records(proc_root)):
                try:
                    kill(identity.pid, signal.SIGKILL)
                    kill_pids.append(identity.pid)
                except ProcessLookupError:
                    pass
        wait(term_grace_s)
        final = tuple(identity for identity in survivors if _identity_is_present(identity, _process_records(proc_root)))
        if final:
            raise RuntimeError(f"legacy scoped stop incomplete: members={[identity.pid for identity in final]}")
        return ScopedStopResult(pids, returncode, stdout, stderr, timed_out, True,
                                tuple(term_pids), tuple(kill_pids), (), None)

    if pid_lookup is not _candidate_cluster_pids or legacy_gpu1:
        lookup = legacy_pid_lookup if legacy_gpu1 else pid_lookup
        invoking_pid = os.getpid()
        pids = tuple(pid for pid in lookup(ownership) if pid != invoking_pid)
        if legacy_gpu1 and not pids:
            raise ValueError("legacy GPU1 teardown found no corroborated Ray process tree")
        returncode, stdout, stderr, timed_out = _serve_shutdown(ownership, run=run, timeout_s=timeout_s)
        term_pids: list[int] = []
        for pid in pids:
            try:
                kill(pid, signal.SIGTERM)
                term_pids.append(pid)
            except ProcessLookupError:
                pass
        survivors = tuple(pid for pid in lookup(ownership) if pid != invoking_pid and pid in pids)
        kill_pids: list[int] = []
        for pid in survivors:
            try:
                kill(pid, signal.SIGKILL)
                kill_pids.append(pid)
            except ProcessLookupError:
                pass
        return ScopedStopResult(pids, returncode, stdout, stderr, timed_out, legacy_gpu1,
                                tuple(term_pids), tuple(kill_pids), survivors, None)

    snapshot = _cluster_membership_snapshot(ownership, proc_root=proc_root)
    frozen = snapshot.member_identities
    # Membership ambiguity is raised above before this targeted Serve action.
    returncode, stdout, stderr, timed_out = _serve_shutdown(ownership, run=run, timeout_s=timeout_s)
    after_shutdown = _process_records(proc_root)
    members = _post_shutdown_members(snapshot, frozen, after_shutdown)
    term_pids: list[int] = []
    for identity in _child_first(members, after_shutdown):
        current = _process_records(proc_root)
        if _identity_is_live(identity, current, snapshot.signature):
            try:
                kill(identity.pid, signal.SIGTERM)
                term_pids.append(identity.pid)
            except ProcessLookupError:
                pass
    wait(term_grace_s)
    after_term = _process_records(proc_root)
    survivors = _post_shutdown_members(snapshot, members, after_term)
    kill_pids: list[int] = []
    for identity in _child_first(survivors, after_term):
        current = _process_records(proc_root)
        # PID reuse changes starttime and is explicitly not a SIGKILL survivor.
        if _identity_is_live(identity, current, snapshot.signature):
            try:
                kill(identity.pid, signal.SIGKILL)
                kill_pids.append(identity.pid)
            except ProcessLookupError:
                pass
    wait(term_grace_s)
    final_records = _process_records(proc_root)
    final_members = _post_shutdown_members(snapshot, survivors, final_records)
    listeners = _listener_pids(int(ownership.gcs_address.rsplit(":", 1)[1]), proc_root)
    if final_members or listeners:
        raise RuntimeError(
            f"scoped stop incomplete: members={[identity.pid for identity in final_members]} listeners={listeners}"
        )
    return ScopedStopResult(
        tuple(identity.pid for identity in frozen), returncode, stdout, stderr, timed_out, False,
        tuple(term_pids), tuple(kill_pids), (), snapshot.signature,
    )


def lifecycle_main(argv: Sequence[str] | None = None) -> int:
    """CLI used only for allowlisted, cluster-scoped cutover cleanup."""
    parser = argparse.ArgumentParser(description="Scoped shutdown for one committed Ego Ray cluster")
    parser.add_argument("--scoped-stop", action="store_true")
    parser.add_argument("--legacy-gpu1", action="store_true", help="one-time pre-marker teardown for the exact GPU1 tuple")
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--temp-dir", required=True)
    parser.add_argument("--gcs-address", required=True)
    parser.add_argument("--dashboard-address", required=True)
    args = parser.parse_args(argv)
    if not args.scoped_stop:
        parser.error("--scoped-stop is required; lifecycle never performs a broad stop")
    result = scoped_stop_cluster(
        cluster_id=args.cluster_id, temp_dir=args.temp_dir, gcs_address=args.gcs_address,
        dashboard_address=args.dashboard_address, legacy_gpu1=args.legacy_gpu1,
    )
    print(json.dumps({"cluster_id": args.cluster_id, **result.__dict__}, sort_keys=True))
    return 0


def cosmos3_cutover_targets(*, bare_url: str = COSMOS3_BARE_BASELINE_URL,
                            ray_url: str = COSMOS3_RAY_MANAGED_URL,
                            serve_config_path: str = "",
                            standalone_artifacts_dir: str = STANDALONE_ARTIFACTS_DIR) -> dict[str, Any]:
    """Explicit-address cutover target bundle consumed by ``cosmos3_cutover``.

    All addresses are explicit (no ``auto``); the gate refuses to cut over if any is
    empty or ``auto``. ``bare_url`` is the production baseline on 8001, which this
    code never stops; ``ray_url`` is the Ray-managed candidate on 28006.
    """
    lc = cosmos3_lifecycle()
    return {
        "bare_url": bare_url,
        "ray_url": ray_url,
        "gcs_address": lc.gcs_address,
        "dashboard_address": lc.dashboard_address,
        "interpreter": lc.interpreter,
        "gpu_id": lc.gpu_id,
        "serve_http_port": COSMOS3_SERVE_HTTP_PORT,
        "serve_config_path": serve_config_path,
        "standalone_artifacts_dir": standalone_artifacts_dir,
        "equivalence_prompt": "Reply with exactly: EGO_COSMOS3_EQUIVALENCE_PROBE",
    }


if __name__ == "__main__":  # pragma: no cover - invoked by an authorized operator
    raise SystemExit(lifecycle_main())
