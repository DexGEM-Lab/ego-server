from __future__ import annotations

import sqlite3

from ego_annotation.serving.lane_dispatcher import LaneDispatcher, LeaseRegistry


def test_reconcile_returns_idle_reaper_capacity_without_dropping_affinity(tmp_path) -> None:
    replica = "http://127.0.0.1:29002"
    registry = LeaseRegistry(str(tmp_path / "leases.sqlite3"), max_sessions_per_lane=8)
    assert registry.bind_session("orphaned-session", replica)
    dispatcher = LaneDispatcher(registry=registry, droid_replicas=(replica,), unidepth_replicas=(replica,))
    dispatcher._droid_active_sessions = lambda _replica: 0  # type: ignore[method-assign]

    dispatcher.reconcile_droid_sessions()

    # The backend's status is authoritative after its idle reaper frees the
    # DepthVideo. The sticky row is retained so a late client request remains
    # attributable to its original owner rather than silently being rerouted.
    assert registry.lookup_session("orphaned-session") == replica
    with sqlite3.connect(registry.db_path) as connection:
        assert connection.execute(
            "SELECT active_sessions FROM replica_inflight WHERE replica_url = ?", (replica,),
        ).fetchone() == (0,)
