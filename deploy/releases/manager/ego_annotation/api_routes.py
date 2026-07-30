"""Fixed logical stage routes and native batch declarations.

The route table is a source-level service contract. A request can select an
algorithm, but it cannot supply a URL, host, port, or private server path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RouteSpec:
    stage_id: str
    path: str
    owner: str
    native_batch_axis: int | None
    native_batch_cap: int
    native_shape: tuple[int, ...]
    stateful: bool = False

    def __post_init__(self) -> None:
        if not self.stage_id or not self.path.startswith("/") or self.path.count("/") < 1:
            raise ValueError("route stage/path are required")
        if self.native_batch_cap <= 0 or not self.native_shape or any(int(dim) <= 0 for dim in self.native_shape):
            raise ValueError("route native batch declaration is invalid")
        if self.native_batch_axis is not None and self.native_shape[self.native_batch_axis] <= 0:
            raise ValueError("route native batch axis is invalid")


ROUTES: Mapping[str, RouteSpec] = {
    "unidepth.infer": RouteSpec("unidepth.infer", "/unidepth.infer", "gpu0-unidepth", 0, 8, (8, 540, 960, 3)),
    "hands.detect": RouteSpec("hands.detect", "/hands.detect", "gpu1-hands", 0, 8, (8, 540, 960, 3)),
    "wilor.reconstruct": RouteSpec("wilor.reconstruct", "/wilor.reconstruct", "gpu1-wilor", 0, 16, (16, 3, 256, 256)),
    "droid.create_session": RouteSpec("droid.create_session", "/droid.create_session", "gpu2-droid", None, 1, (1,), True),
    "droid.push_frame": RouteSpec("droid.push_frame", "/droid.push_frame", "gpu2-droid", None, 8, (8,), True),
    "droid.finalize": RouteSpec("droid.finalize", "/droid.finalize", "gpu2-droid", None, 1, (1,), True),
    "hawor.infer_tracks": RouteSpec("hawor.infer_tracks", "/hawor.infer_tracks", "gpu3-hawor", 0, 4, (4, 16, 3, 256, 256)),
    "hawor_infiller.fill": RouteSpec("hawor_infiller.fill", "/hawor_infiller.fill", "gpu3-infiller", 0, 2, (2, 120, 218)),
    "cosmos3.reason": RouteSpec("cosmos3.reason", "/cosmos3.reason", "gpu6-cosmos3", None, 1, (1,)),
}


def route_for(stage_id: str) -> RouteSpec:
    try:
        return ROUTES[stage_id]
    except KeyError as exc:
        raise KeyError(f"no fixed route for stage {stage_id!r}") from exc


def all_routes() -> tuple[RouteSpec, ...]:
    return tuple(ROUTES.values())


__all__ = ["ROUTES", "RouteSpec", "all_routes", "route_for"]
