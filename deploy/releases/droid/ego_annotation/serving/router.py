"""Typed multi-cluster router: stable public API names -> GPU Serve HTTP endpoints.

Ray remains internal. Callers see stable model-native API names (``unidepth.infer``,
``hands.detect``, ...) and the router resolves each to the Serve HTTP endpoint of the
physical GPU group that owns it. This module never imports Ray.

Endpoint configuration lives here (not in ``lifecycle.py``) so the public routing
surface is owned by the gateway layer and the lifecycle/Ray-internal configuration is
not disturbed. Each committed GPU group has one Serve HTTP "lane" port equal to
``28000 + gpu_id`` (GPU0=28000 ... GPU3=28003, GPU6=28006). This is the canonical
public lane-port block; it is overridable per GPU via
``EGO_SERVE_HTTP_PORT_GPU<id>`` and globally via ``EGO_SERVE_HOST``.

The current bare Cosmos3 process listens on ``8001`` (and ``7861``). It is **not** a
canonical lane port: it is a transient baseline that must be reached only through an
explicit, labeled override (``COSMOS3_BASELINE_URL`` / ``cosmos3_baseline_url``), never
as a router default. Once the Ray-managed GPU6 deployment is live it serves on the
canonical lane port 28006 and the baseline override is no longer used.

The router is constructible from the canonical ``CLUSTER_ENDPOINTS`` and may be
overridden per API name (for tests and for pointing at a live endpoint discovered at
run time). Endpoint runners check liveness *once* (see ``benchmark.endpoints``); the
router itself never polls and never blocks on model lanes becoming available.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

from ego_annotation.serving.lifecycle import COMMITTED_GPU_GROUPS


# Canonical public Serve HTTP lane-port block. Each committed GPU group's lane port is
# 28000 + gpu_id (GPU0=28000, GPU1=28001, GPU2=28002, GPU3=28003, GPU6=28006). These
# are the ports the gateway and benchmark target by default; they are distinct from
# the Ray lifecycle component/worker ports owned by ``lifecycle.py``.
_LANE_PORT_BASE = 28000


# Native work-unit label per API. The benchmark offered-load generator expresses
# intensity in these native units (images/s, crops/s, track chunks/s, ...) rather
# than a count-only request rate, so a request carrying N work units contributes N
# to offered load. These labels are descriptive metadata; the weighted work value
# travels on each payload manifest item.
class WorkUnit(str, Enum):
    IMAGES = "images"
    CROPS = "crops"
    TRACK_CHUNKS = "track_chunks"
    TEMPORAL_WINDOWS = "temporal_windows"
    READY_FRAMES = "ready_frames"
    SESSIONS = "sessions"
    MEDIA_REQUESTS = "media_requests"


class ModelApiName(str, Enum):
    """Stable public model-native API names. Ray deployment names map 1:1 to these."""

    UNIDEPTH_INFER = "unidepth.infer"
    HANDS_DETECT = "hands.detect"
    WILOR_RECONSTRUCT = "wilor.reconstruct"
    DROID_CREATE_SESSION = "droid.create_session"
    DROID_PUSH_FRAME = "droid.push_frame"
    DROID_FINALIZE = "droid.finalize"
    HAWOR_INFER_TRACKS = "hawor.infer_tracks"
    HAWOR_INFILLER_FILL = "hawor_infiller.fill"
    COSMOS3_REASON = "cosmos3.reason"


# Default model revisions. The serving replica owns the resident revision and
# rejects mismatches at admission; these are the revisions the gateway advertises to
# callers and that the benchmark uses to build requests. They are overridable via
# environment so a deployment cutover does not require a code change.
DEFAULT_MODEL_REVISIONS: dict[ModelApiName, str] = {
    ModelApiName.UNIDEPTH_INFER: "unidepth-v2-vitl14-corrected",
    ModelApiName.HANDS_DETECT: "hands-yolo-sam2.1-hiera-l",
    ModelApiName.WILOR_RECONSTRUCT: "wilor-final-v1",
    ModelApiName.DROID_CREATE_SESSION: "droid-v1",
    ModelApiName.DROID_PUSH_FRAME: "droid-v1",
    ModelApiName.DROID_FINALIZE: "droid-v1",
    ModelApiName.HAWOR_INFER_TRACKS: "hawor-v1",
    ModelApiName.HAWOR_INFILLER_FILL: "hawor-infiller-v1",
    ModelApiName.COSMOS3_REASON: "cosmos3-nano-v1",
}


# Native work unit per API (drives benchmark offered-load units).
API_WORK_UNITS: dict[ModelApiName, WorkUnit] = {
    ModelApiName.UNIDEPTH_INFER: WorkUnit.IMAGES,
    ModelApiName.HANDS_DETECT: WorkUnit.IMAGES,
    ModelApiName.WILOR_RECONSTRUCT: WorkUnit.CROPS,
    ModelApiName.DROID_CREATE_SESSION: WorkUnit.SESSIONS,
    ModelApiName.DROID_PUSH_FRAME: WorkUnit.READY_FRAMES,
    ModelApiName.DROID_FINALIZE: WorkUnit.SESSIONS,
    ModelApiName.HAWOR_INFER_TRACKS: WorkUnit.TRACK_CHUNKS,
    ModelApiName.HAWOR_INFILLER_FILL: WorkUnit.TEMPORAL_WINDOWS,
    ModelApiName.COSMOS3_REASON: WorkUnit.MEDIA_REQUESTS,
}


# The seven model-native services. The public API surface is nine API *method* names;
# DROID is one service exposed through three session-method API names. This grouping
# reconciles the method-level API list with the seven resident services so the
# benchmark can sweep services (and report per-service results) while still supporting
# DROID's three session methods as distinct, individually-addressable endpoints.
@dataclass(frozen=True)
class ModelService:
    """One resident model-native service and its public API method name(s)."""

    name: str
    gpu_id: int
    api_names: tuple[ModelApiName, ...]

    @property
    def is_stateful_session(self) -> bool:
        """DROID exposes a create/push/finalize session lifecycle (three methods)."""
        return len(self.api_names) > 1


MODEL_SERVICES: tuple[ModelService, ...] = (
    ModelService(name="unidepth", gpu_id=0, api_names=(ModelApiName.UNIDEPTH_INFER,)),
    ModelService(name="hands", gpu_id=1, api_names=(ModelApiName.HANDS_DETECT,)),
    ModelService(name="wilor", gpu_id=1, api_names=(ModelApiName.WILOR_RECONSTRUCT,)),
    ModelService(name="droid", gpu_id=2, api_names=(
        ModelApiName.DROID_CREATE_SESSION, ModelApiName.DROID_PUSH_FRAME, ModelApiName.DROID_FINALIZE,
    )),
    ModelService(name="hawor_tracks", gpu_id=3, api_names=(ModelApiName.HAWOR_INFER_TRACKS,)),
    ModelService(name="hawor_infiller", gpu_id=3, api_names=(ModelApiName.HAWOR_INFILLER_FILL,)),
    ModelService(name="cosmos3", gpu_id=6, api_names=(ModelApiName.COSMOS3_REASON,)),
)


def service_for_api(api_name: ModelApiName | str) -> ModelService:
    """The resident service that owns a public API method name."""
    key = ModelApiName(api_name) if isinstance(api_name, str) else api_name
    for svc in MODEL_SERVICES:
        if key in svc.api_names:
            return svc
    raise RouterError(f"no service configured for {api_name!r}")


def _default_serve_http_port(gpu_id: int) -> int:
    """Canonical Serve HTTP lane port for a GPU group: 28000 + gpu_id.

    This is the public lane-port block the gateway and benchmark target: GPU0=28000,
    GPU1=28001, GPU2=28002, GPU3=28003, GPU6=28006. Distinct from Ray lifecycle
    component/worker ports (owned by ``lifecycle.py``), the bare Cosmos3 process
    (8001/7861, a transient baseline reached only via an explicit override), and
    Cosmos2 (stopped, 8000/7860). Overridable via ``EGO_SERVE_HTTP_PORT_GPU<id>``.
    """
    env_name = f"EGO_SERVE_HTTP_PORT_GPU{gpu_id}"
    return int(os.environ.get(env_name, str(_LANE_PORT_BASE + gpu_id)))


# Bare Cosmos3 vLLM process currently live on GPU6. This is a *transient baseline* —
# the Ray-managed GPU6 deployment (lane port 28006) is the canonical target. The bare
# port must be reachable only through an explicit, labeled override so no code path
# silently treats it as the canonical Cosmos3 endpoint. Once the Ray-managed GPU6
# deployment passes equivalent health/inference, this baseline is retired.
COSMOS3_BASELINE_HOST = "127.0.0.1"
COSMOS3_BASELINE_PORT = 8001
COSMOS3_BASELINE_URL = f"http://{COSMOS3_BASELINE_HOST}:{COSMOS3_BASELINE_PORT}/cosmos3.reason"


def cosmos3_baseline_override(url: str | None = None) -> dict[ModelApiName, str]:
    """Explicit, labeled override that points ``cosmos3.reason`` at the bare baseline.

    Returns a per-API URL override map to pass to ``ModelServiceRouter.with_overrides``.
    The default ``url`` is the current bare Cosmos3 vLLM process (port 8001). This is
    the **only** supported way to route ``cosmos3.reason`` at the baseline; the
    canonical default remains the lane port 28006. Callers that use this must record
    that they are benchmarking the baseline, not the Ray-managed deployment.
    """
    return {ModelApiName.COSMOS3_REASON: url or COSMOS3_BASELINE_URL}


def _default_serve_host() -> str:
    return os.environ.get("EGO_SERVE_HOST", "127.0.0.1")


@dataclass(frozen=True)
class ServeEndpoint:
    """One public API's resolved Serve HTTP endpoint.

    ``route_path`` is the HTTP path the Serve deployment serves the API at; it equals
    the API name so callers and the router agree on the public surface. ``base_url``
    is the cluster's Serve HTTP origin (host:port).
    """

    api_name: ModelApiName
    gpu_id: int
    physical_group: str
    host: str
    serve_http_port: int
    route_path: str
    model_revision: str
    work_unit: WorkUnit

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.serve_http_port}"

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.route_path}"

    def to_manifest(self) -> dict[str, str]:
        return {
            "api_name": self.api_name.value,
            "gpu_id": str(self.gpu_id),
            "physical_group": self.physical_group,
            "host": self.host,
            "serve_http_port": str(self.serve_http_port),
            "route_path": self.route_path,
            "model_revision": self.model_revision,
            "work_unit": self.work_unit.value,
            "url": self.url,
        }


@dataclass(frozen=True)
class ClusterEndpointConfig:
    """Serve HTTP endpoint configuration for one committed GPU group."""

    gpu_id: int
    physical_group: str
    serve_http_host: str
    serve_http_port: int
    api_names: tuple[ModelApiName, ...]


def _cluster_endpoint_for_group(gpu_id: int) -> ClusterEndpointConfig:
    group = next(g for g in COMMITTED_GPU_GROUPS if g.gpu_id == gpu_id)
    return ClusterEndpointConfig(
        gpu_id=gpu_id,
        physical_group=group.physical_group,
        serve_http_host=_default_serve_host(),
        serve_http_port=_default_serve_http_port(gpu_id),
        api_names=tuple(ModelApiName(name) for name in group.logical_apis),
    )


def canonical_cluster_endpoints() -> tuple[ClusterEndpointConfig, ...]:
    """The five committed GPU groups (0/1/2/3/6) and their Serve HTTP ports."""
    return tuple(_cluster_endpoint_for_group(gpu) for gpu in (0, 1, 2, 3, 6))


CLUSTER_ENDPOINTS: tuple[ClusterEndpointConfig, ...] = canonical_cluster_endpoints()


class RouterError(KeyError):
    """Raised when an API name has no configured endpoint."""


@dataclass(frozen=True)
class ModelServiceRouter:
    """Maps stable public API names to resolved Serve HTTP endpoints.

    Constructed from the canonical cluster endpoints by default. Per-API URL
    overrides (``api_url_overrides``) let tests and runtime discovery point an API at
    a specific URL (e.g. a fake server, or a lane that became live after the run
    manifest was written) without changing the canonical configuration.

    The router performs no network I/O and never polls. Liveness checks are the
    endpoint runner's responsibility (``benchmark.endpoints``), performed once.
    """

    endpoints: tuple[ServeEndpoint, ...]
    api_url_overrides: Mapping[ModelApiName, str] = field(default_factory=dict)

    @classmethod
    def canonical(cls, *, api_url_overrides: Mapping[ModelApiName, str] | None = None) -> "ModelServiceRouter":
        endpoints: list[ServeEndpoint] = []
        for cluster in CLUSTER_ENDPOINTS:
            for api_name in cluster.api_names:
                endpoints.append(
                    ServeEndpoint(
                        api_name=api_name,
                        gpu_id=cluster.gpu_id,
                        physical_group=cluster.physical_group,
                        host=cluster.serve_http_host,
                        serve_http_port=cluster.serve_http_port,
                        route_path=f"/{api_name.value}",
                        model_revision=DEFAULT_MODEL_REVISIONS[api_name],
                        work_unit=API_WORK_UNITS[api_name],
                    )
                )
        return cls(tuple(endpoints), api_url_overrides or {})

    def __post_init__(self) -> None:
        seen: set[ModelApiName] = set()
        for ep in self.endpoints:
            if ep.api_name in seen:
                raise RouterError(f"duplicate endpoint for {ep.api_name}")
            seen.add(ep.api_name)

    def endpoint_for(self, api_name: ModelApiName | str) -> ServeEndpoint:
        try:
            key = ModelApiName(api_name) if isinstance(api_name, str) else api_name
        except ValueError as exc:
            raise RouterError(f"unknown api name {api_name!r}") from exc
        for ep in self.endpoints:
            if ep.api_name == key:
                return ep
        raise RouterError(f"no endpoint configured for {api_name!r}")

    def url_for(self, api_name: ModelApiName | str) -> str:
        ep = self.endpoint_for(api_name)
        override = self.api_url_overrides.get(ep.api_name)
        return override or ep.url

    def base_url_for(self, api_name: ModelApiName | str) -> str:
        """Origin (scheme://host:port) of the resolved endpoint, honoring overrides.

        Used by the endpoint runner to probe a health path on the same origin the
        API is actually routed to (which may be a fake server or a lane discovered
        at run time via an override).
        """
        from urllib.parse import urlsplit

        resolved = self.url_for(api_name)
        parts = urlsplit(resolved)
        return f"{parts.scheme}://{parts.netloc}"

    def apis_for_gpu(self, gpu_id: int) -> tuple[ServeEndpoint, ...]:
        return tuple(ep for ep in self.endpoints if ep.gpu_id == gpu_id)

    def all_apis(self) -> tuple[ModelApiName, ...]:
        return tuple(ep.api_name for ep in self.endpoints)

    def with_overrides(
        self, api_url_overrides: Mapping[ModelApiName, str] | Mapping[str, str]
    ) -> "ModelServiceRouter":
        normalized: dict[ModelApiName, str] = dict(self.api_url_overrides)
        for key, value in api_url_overrides.items():
            normalized[ModelApiName(key) if isinstance(key, str) else key] = value
        return ModelServiceRouter(self.endpoints, normalized)


def route_path_for(api_name: ModelApiName | str) -> str:
    """The canonical HTTP path an API is served at. Equals the public API name."""
    return f"/{ModelApiName(api_name).value if isinstance(api_name, str) else api_name.value}"
