"""Typed algorithm DAG interface and execution policy.

The graph owns dependency and lane semantics only. Stage adapters are injected;
missing adapters fail explicitly instead of fabricating an algorithm result.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from ego_annotation.api_routes import route_for
from ego_annotation.scripted.contracts import AlgorithmRequest, AlgorithmResult


class AdapterConfigurationError(RuntimeError):
    """A required stage adapter/request was not configured."""


class DagContractError(ValueError):
    """The graph violates wave or lane invariants."""


class DagLane(str, Enum):
    PHYSICAL = "physical"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class StageNode:
    stage_id: str
    lane: DagLane
    wave: int
    dependencies: tuple[str, ...] = ()
    remote_route: bool = True

    def __post_init__(self) -> None:
        if not self.stage_id or self.wave < 1:
            raise DagContractError("stage node id and positive wave are required")
        if self.remote_route:
            route_for(self.stage_id)


# The graph contains algorithm/consumer boundaries only. Source timeline and
# package adapters are outside the graph and are injected by the caller.
DAG_NODES: tuple[StageNode, ...] = (
    StageNode("unidepth.infer", DagLane.PHYSICAL, 1),
    StageNode("hands.detect", DagLane.PHYSICAL, 1),
    StageNode("cosmos3.reason", DagLane.SEMANTIC, 1),
    StageNode("droid.create_session", DagLane.PHYSICAL, 2, ("unidepth.infer",)),
    StageNode("droid.push_frame", DagLane.PHYSICAL, 2, ("droid.create_session", "unidepth.infer")),
    StageNode("droid.finalize", DagLane.PHYSICAL, 2, ("droid.push_frame",)),
    StageNode("wilor.reconstruct", DagLane.PHYSICAL, 2, ("hands.detect",)),
    StageNode("hawor.infer_tracks", DagLane.PHYSICAL, 3, ("unidepth.infer", "hands.detect", "droid.finalize")),
    StageNode("hawor_infiller.fill", DagLane.PHYSICAL, 4, ("unidepth.infer", "droid.finalize", "hawor.infer_tracks")),
    StageNode("physical.render", DagLane.PHYSICAL, 5, ("hawor_infiller.fill", "wilor.reconstruct", "droid.finalize"), False),
    StageNode("physical.qc", DagLane.PHYSICAL, 6, ("physical.render", "hawor_infiller.fill", "droid.finalize"), False),
    StageNode("physical.evaluator", DagLane.PHYSICAL, 7, ("physical.qc", "hawor_infiller.fill", "droid.finalize"), False),
    StageNode("semantic.alignment", DagLane.SEMANTIC, 2, ("cosmos3.reason",), False),
    StageNode("semantic.subtitle_render", DagLane.SEMANTIC, 3, ("semantic.alignment",), False),
)


@dataclass(frozen=True)
class SemanticLaneState:
    status: str
    rows: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        if self.status not in {"enabled", "absent_disabled"}:
            raise DagContractError("semantic lane status must be enabled or absent_disabled")
        if self.status == "absent_disabled" and self.rows:
            raise DagContractError("disabled semantic lane cannot contain rows")


@dataclass(frozen=True)
class RunnerConfig:
    cosmos_enabled: bool = False
    fresh_run_root: str = ""
    remote_origin: str = "http://127.0.0.1"
    allowed_service_ports: tuple[int, ...] = (28000, 28001, 28002, 28003, 28004, 28006)
    shared_production_actors: bool = False
    performance_attribution: str = "valid"
    allow_shared_smoke: bool = False

    def __post_init__(self) -> None:
        if not self.fresh_run_root:
            raise DagContractError("remote run requires a fresh run root")
        if not self.remote_origin.startswith("http://127.0.0.1") and not self.remote_origin.startswith("http://localhost"):
            raise DagContractError("remote API run must originate from A800 localhost")
        if self.cosmos_enabled and 28006 not in self.allowed_service_ports:
            raise DagContractError("Cosmos cannot be enabled without its fixed port")
        if self.performance_attribution not in {"valid", "invalid"}:
            raise DagContractError("performance_attribution must be valid or invalid")
        if self.shared_production_actors and (not self.allow_shared_smoke or self.performance_attribution != "invalid"):
            raise DagContractError("shared production smoke requires explicit opt-in and invalid performance attribution")
        if not self.shared_production_actors and self.performance_attribution == "invalid":
            raise DagContractError("invalid attribution requires shared production actor marker")

    def semantic_lane(self) -> SemanticLaneState:
        return SemanticLaneState("enabled", ()) if self.cosmos_enabled else SemanticLaneState("absent_disabled", ())


@dataclass(frozen=True)
class DroidRetryDecision:
    action: str
    reason: str
    filter_thresh: float | None = None


@dataclass(frozen=True)
class DroidRetryStrategy:
    """Explicit bounded recovery policy with direction tied to evidence."""

    oom_filter_thresh: float | None = None
    exclusive_fixed_release: str | None = None
    keyframe_retry_filter_thresh: float | None = None
    max_keyframe_retries: int = 1

    def __post_init__(self) -> None:
        if self.max_keyframe_retries != 1:
            raise DagContractError("keyframe recovery is exactly one bounded retry")
        if self.oom_filter_thresh is not None and self.oom_filter_thresh <= 0:
            raise DagContractError("OOM filter_thresh must be positive")
        if self.keyframe_retry_filter_thresh is not None and self.keyframe_retry_filter_thresh <= 0:
            raise DagContractError("keyframe retry filter_thresh must be positive")

    def on_oom(self) -> DroidRetryDecision:
        return DroidRetryDecision(
            action="remote_droid_oom",
            reason="session-local frontend memory failure; repair requires exclusive fixed release/no_grad",
            filter_thresh=self.oom_filter_thresh,
        )

    def on_finalize_keyframes(self, keyframe_count: int, retries_used: int = 0) -> DroidRetryDecision:
        if keyframe_count > 1:
            return DroidRetryDecision("accept", "measured trajectory has more than one keyframe")
        if retries_used < 0 or retries_used > 1:
            raise DagContractError("retries_used must be 0 or 1")
        if retries_used == 0 and self.keyframe_retry_filter_thresh is not None:
            return DroidRetryDecision(
                "retry_lower_filter_thresh",
                "user strategy attempts to increase keyframes; direction is not directly validated by OOM attempts",
                self.keyframe_retry_filter_thresh,
            )
        return DroidRetryDecision(
            "remote_droid_insufficient_keyframes",
            "preserve the sole measured pose; skip or fail pairwise BA/filler explicitly",
        )


class StageAdapter(Protocol):
    def execute(self, request: AlgorithmRequest[Any]) -> AlgorithmResult[Any]:
        ...


@dataclass
class AdapterRegistry:
    adapters: Mapping[str, StageAdapter]

    def adapter_for(self, stage_id: str) -> StageAdapter:
        adapter = self.adapters.get(stage_id)
        if adapter is None:
            raise AdapterConfigurationError(f"no adapter configured for stage {stage_id!r}")
        return adapter


class TypedDag:
    def __init__(self, nodes: Sequence[StageNode] = DAG_NODES) -> None:
        self.nodes = tuple(nodes)
        self._by_id = {node.stage_id: node for node in self.nodes}
        if len(self._by_id) != len(self.nodes):
            raise DagContractError("DAG contains duplicate stage ids")
        self._validate()

    def _validate(self) -> None:
        for node in self.nodes:
            for dependency in node.dependencies:
                if dependency not in self._by_id:
                    raise DagContractError(f"{node.stage_id} depends on missing stage {dependency}")
                dependency_node = self._by_id[dependency]
                if dependency_node.wave > node.wave:
                    raise DagContractError(f"{node.stage_id} starts before dependency {dependency}")
                if node.lane == DagLane.PHYSICAL and dependency_node.lane == DagLane.SEMANTIC:
                    raise DagContractError(f"physical stage {node.stage_id} cannot depend on semantic stage {dependency}")
        droid = self._by_id["droid.create_session"]
        if "unidepth.infer" not in droid.dependencies:
            raise DagContractError("DROID must wait for UniDepth")
        if "hands.detect" not in self._by_id["wilor.reconstruct"].dependencies:
            raise DagContractError("WiLoR must wait for Hands")
        if any(node.lane == DagLane.PHYSICAL and "cosmos3.reason" in node.dependencies for node in self.nodes):
            raise DagContractError("Cosmos cannot block physical lane")

    def start_order(self, *, include_semantic: bool = True) -> tuple[str, ...]:
        active = {node.stage_id for node in self.nodes if include_semantic or node.lane != DagLane.SEMANTIC}
        remaining = set(active)
        order: list[str] = []
        while remaining:
            ready = sorted(
                (self._by_id[stage] for stage in remaining if all(dep not in remaining for dep in self._by_id[stage].dependencies)),
                key=lambda node: (node.wave, node.stage_id),
            )
            if not ready:
                raise DagContractError("DAG contains a cycle")
            for node in ready:
                order.append(node.stage_id)
                remaining.remove(node.stage_id)
        return tuple(order)

    def run(
        self,
        requests: Mapping[str, AlgorithmRequest[Any]],
        registry: AdapterRegistry,
        config: RunnerConfig,
    ) -> tuple[dict[str, AlgorithmResult[Any]], SemanticLaneState, tuple[str, ...]]:
        outputs: dict[str, AlgorithmResult[Any]] = {}
        started: list[str] = []
        for stage_id in self.start_order(include_semantic=config.cosmos_enabled):
            node = self._by_id[stage_id]
            if node.lane == DagLane.SEMANTIC and not config.cosmos_enabled:
                continue
            request = requests.get(stage_id)
            if request is None:
                raise AdapterConfigurationError(f"missing typed request for stage {stage_id!r}")
            for dependency in node.dependencies:
                if dependency not in outputs and self._by_id[dependency].lane == DagLane.PHYSICAL:
                    raise AdapterConfigurationError(f"stage {stage_id!r} started without physical dependency {dependency!r}")
            adapter = registry.adapter_for(stage_id)
            outputs[stage_id] = adapter.execute(request)
            started.append(stage_id)
        return outputs, config.semantic_lane(), tuple(started)


__all__ = [
    "AdapterConfigurationError",
    "AdapterRegistry",
    "DAG_NODES",
    "DagLane",
    "DagContractError",
    "DroidRetryDecision",
    "DroidRetryStrategy",
    "RunnerConfig",
    "SemanticLaneState",
    "StageAdapter",
    "StageNode",
    "TypedDag",
]
