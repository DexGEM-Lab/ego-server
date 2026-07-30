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
hands/WiLoR, GPU2 DROID, and GPU3 HaWoR use model-specific Python 3.10 envs),
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
from typing import Any


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

    def startup_command(self, cluster_name: str = "") -> str:
        """The exact ``ray start --head`` command for this cluster.

        ``CUDA_VISIBLE_DEVICES`` is set in the environment before launch so only the
        assigned physical GPU is visible; ``--num-gpus=1`` advertises it as one Ray
        GPU so replicas requesting ``num_gpus=1`` get native ownership.

        ``cluster_name`` is accepted for caller compatibility but not emitted because
        Ray 2.55.1 ``ray start`` has no such flag. The GCS port identifies the head.
        """
        committed_env = (
            ("AUTOSCALER_METRIC_PORT", str(self.ports.autoscaler_metric_port)),
            ("DASHBOARD_METRIC_PORT", str(self.ports.dashboard_metric_port)),
        ) + self.env_vars
        env_prefix = " ".join(f"{k}={v}" for k, v in committed_env)
        cuda_prefix = f"CUDA_VISIBLE_DEVICES={self.gpu_id}"
        prefix = f"{cuda_prefix} {env_prefix}".strip()
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
        gpu_id=1, physical_group="hands-wilor", logical_apis=("hands.detect", "wilor.reconstruct"), adapter_implemented=True,
        interpreter=_GPU1_INTERPRETER,
        lifecycle=ClusterLifecycleConfig(
            gpu_id=1, interpreter=_GPU1_INTERPRETER,
            temp_dir="/tmp/ray-ego-serve-gpu1",
            ports=_ports(27000, 27100, serve_http_port=28001),
            env_vars=(
                ("PYTHONPATH", f"{RUNTIME_CURRENT_DIR}:/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/sam2:/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor_model"),
                ("EGO_SAM2_REPO", "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego_annotation-master/third_party/sam2"),
                ("EGO_WILOR_REPO", "/vePFS-Mindverse/user/yiwen/user-home/zjh/ego-annation-checkpoints/wilor_model"),
                ("EGO_HANDS_GPU", "1"),
                # Multipart remains production default. Explicit envelope treatments
                # turn on matching result diagnostics for both logical APIs.
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


def unidepth_gpu_group() -> GpuServiceGroup:
    return COMMITTED_GPU_GROUPS[0]


def droid_gpu_group() -> GpuServiceGroup:
    """The GPU2 stateful DROID group (one resident DroidNet, isolated sessions)."""
    return COMMITTED_GPU_GROUPS[2]


def hawor_gpu_group() -> GpuServiceGroup:
    """The GPU3 HaWoR + infiller group (one shared physical resident lane)."""
    return COMMITTED_GPU_GROUPS[3]


def cosmos3_gpu_group() -> GpuServiceGroup:
    """The GPU6 Cosmos3 group (the second implemented persistent deployment)."""
    return COMMITTED_GPU_GROUPS[4]


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


def hands_wilor_gpu_group() -> GpuServiceGroup:
    return COMMITTED_GPU_GROUPS[1]


def hands_wilor_serve_config() -> dict[str, Any]:
    """Serve config for the GPU1 hands.detect + wilor.reconstruct deployment.

    One replica (``num_gpus=1``) hosts both logical APIs with separate batch queues.
    The HTTP proxy listens on port 28001 (``serve run --port 28001``). The import
    path points at the deployment-only module that exports a bound Ray Serve
    Application; resolved against the GPU1 cluster's ``ray_serve_hands`` interpreter.
    """
    group = hands_wilor_gpu_group()
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
