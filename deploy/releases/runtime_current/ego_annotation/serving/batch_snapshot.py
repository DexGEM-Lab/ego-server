"""Read-only, process-local model-execution snapshots.

The tracker is deliberately independent of Ray/ASGI.  It observes only the span
that actually calls a model backend, leaving ingress admission and HTTP parsing
outside its accounting.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from uuid import uuid4


@dataclass(frozen=True)
class ExecutionSpec:
    operation: str
    execution_kind: str
    request_count: int
    item_count: int | None
    item_kind: str
    image_item_count: int | None
    forward_count: int
    max_batch_size: int | None

    def __post_init__(self) -> None:
        if not self.operation or not self.execution_kind or not self.item_kind:
            raise ValueError("execution operation, kind, and item_kind are required")
        for name in ("request_count", "forward_count"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("item_count", "image_item_count", "max_batch_size"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None")


class ExecutionHandle:
    """One idempotently terminal execution owned by :class:`BatchSnapshotTracker`."""

    def __init__(self, tracker: "BatchSnapshotTracker", execution_id: str) -> None:
        self._tracker = tracker
        self._execution_id = execution_id

    def finish(self, *, success: bool, error_type: str | None = None) -> bool:
        return self._tracker._finish(self._execution_id, success=success, error_type=error_type)


class BatchSnapshotTracker:
    """Thread-safe process-local state for actual model execution spans.

    Monotonic time defines duration and ordering. Wall time is retained only for
    cross-process event ages, where NTP adjustments are an unavoidable source of
    uncertainty. A delayed completion can never overwrite a newer completion.
    """

    def __init__(self, *, service: str, replica_id: str, capacity: Mapping[str, Any]) -> None:
        self._service = service
        self._replica_id = replica_id
        self._capacity = dict(capacity)
        self._instance_id = uuid4().hex
        self._lock = threading.Lock()
        self._active: dict[str, tuple[ExecutionSpec, int, int]] = {}
        self._last_success: dict[str, Any] | None = None
        self._last_success_by_operation: dict[str, dict[str, Any]] = {}
        self._last_terminal: dict[str, Any] | None = None
        self._last_terminal_mono_ns = -1
        self._last_success_mono_ns = -1
        self._last_success_by_operation_mono_ns: dict[str, int] = {}
        self._counters = {"started": 0, "completed": 0, "failed": 0, "duplicate_finish": 0}

    @property
    def instance_id(self) -> str:
        return self._instance_id

    def begin(self, spec: ExecutionSpec) -> ExecutionHandle:
        started_wall_ns = time.time_ns()
        started_mono_ns = time.monotonic_ns()
        execution_id = uuid4().hex
        with self._lock:
            self._active[execution_id] = (spec, started_wall_ns, started_mono_ns)
            self._counters["started"] += 1
        return ExecutionHandle(self, execution_id)

    def _finish(self, execution_id: str, *, success: bool, error_type: str | None) -> bool:
        completed_wall_ns = time.time_ns()
        completed_mono_ns = time.monotonic_ns()
        with self._lock:
            active = self._active.pop(execution_id, None)
            if active is None:
                self._counters["duplicate_finish"] += 1
                return False
            spec, started_wall_ns, started_mono_ns = active
            duration_s = max(0.0, (completed_mono_ns - started_mono_ns) / 1_000_000_000.0)
            terminal = {
                "operation": spec.operation,
                "execution_kind": spec.execution_kind,
                "success": bool(success),
                "error_type": None if success else (error_type or "UnknownError"),
                "started_at_unix_ns": started_wall_ns,
                "started_monotonic_ns": started_mono_ns,
                "completed_at_unix_ns": completed_wall_ns,
                "completed_monotonic_ns": completed_mono_ns,
                "duration_s": duration_s,
            }
            if completed_mono_ns >= self._last_terminal_mono_ns:
                self._last_terminal = terminal
                self._last_terminal_mono_ns = completed_mono_ns
            if success:
                record = {
                    "operation": spec.operation,
                    "execution_kind": spec.execution_kind,
                    "started_at_unix_ns": started_wall_ns,
                    "started_monotonic_ns": started_mono_ns,
                    "completed_at_unix_ns": completed_wall_ns,
                    "completed_monotonic_ns": completed_mono_ns,
                    "duration_s": duration_s,
                    "request_count": spec.request_count,
                    "item_count": spec.item_count,
                    "item_kind": spec.item_kind,
                    "image_item_count": spec.image_item_count,
                    "forward_count": spec.forward_count,
                }
                if completed_mono_ns >= self._last_success_mono_ns:
                    self._last_success = record
                    self._last_success_mono_ns = completed_mono_ns
                if completed_mono_ns >= self._last_success_by_operation_mono_ns.get(spec.operation, -1):
                    self._last_success_by_operation[spec.operation] = record
                    self._last_success_by_operation_mono_ns[spec.operation] = completed_mono_ns
                self._counters["completed"] += 1
            else:
                self._counters["failed"] += 1
            return True

    def snapshot(self, *, adapter_status: Mapping[str, Any]) -> dict[str, Any]:
        """Return a JSON-ready point-in-time snapshot without mutating workload state."""
        with self._lock:
            # Capture the sample only after completing all state reads. A wall-clock
            # rollback cannot make it predate any returned terminal or per-operation
            # success record, even when monotonic ordering selected an older wall time.
            event_walls = [
                self._last_terminal.get("completed_at_unix_ns", 0) if self._last_terminal else 0,
                self._last_success.get("completed_at_unix_ns", 0) if self._last_success else 0,
                *(record.get("completed_at_unix_ns", 0) for record in self._last_success_by_operation.values()),
            ]
            sampled_wall_ns = max(time.time_ns(), *event_walls)
            sampled_mono_ns = time.monotonic_ns()
            active_items = list(self._active.values())
            operations: dict[str, dict[str, int]] = {}
            request_count = 0
            item_count = 0
            has_unknown_items = False
            for spec, _started_wall, _started_mono in active_items:
                bucket = operations.setdefault(spec.operation, {"execution_count": 0, "request_count": 0, "item_count": 0})
                bucket["execution_count"] += 1
                bucket["request_count"] += spec.request_count
                request_count += spec.request_count
                if spec.item_count is None:
                    has_unknown_items = True
                else:
                    bucket["item_count"] += spec.item_count
                    item_count += spec.item_count
            oldest_wall = min((started_wall for _spec, started_wall, _started_mono in active_items), default=None)
            oldest_mono = min((started_mono for _spec, _started_wall, started_mono in active_items), default=None)
            return {
                "schema": "ego.service-batch-snapshot.v1",
                "sampled_at_unix_ns": sampled_wall_ns,
                "sampled_at_monotonic_ns": sampled_mono_ns,
                "service": self._service,
                "replica_id": self._replica_id,
                "tracker_instance_id": self._instance_id,
                "adapter_status": dict(adapter_status),
                "capacity": dict(self._capacity),
                "active": {
                    "execution_count": len(active_items),
                    "request_count": request_count,
                    "item_count": None if has_unknown_items else item_count,
                    "oldest_started_unix_ns": oldest_wall,
                    "oldest_started_monotonic_ns": oldest_mono,
                    "operations": operations,
                },
                "last_success": dict(self._last_success) if self._last_success else None,
                "last_success_by_operation": {key: dict(value) for key, value in self._last_success_by_operation.items()},
                "last_terminal": dict(self._last_terminal) if self._last_terminal else None,
                "counters": dict(self._counters),
            }


async def await_thread_for_execution(handle: ExecutionHandle, function: Callable[..., Any], *args: Any) -> Any:
    """Await one worker belonging to an already-open execution handle.

    Cancellation leaves the handle active until that physical worker returns. The
    owner may keep using the handle for subsequent model calls in the same logical
    execution; ordinary exceptions are deliberately re-raised to that owner.
    """
    worker = asyncio.create_task(asyncio.to_thread(function, *args))

    def finish_cancelled_worker(task: "asyncio.Task[Any]") -> None:
        # Consume any worker exception but preserve caller cancellation as the
        # terminal reason. The worker's completion, not cancellation delivery,
        # remains the boundary that removes it from active state.
        try:
            task.result()
        except BaseException:
            pass
        handle.finish(success=False, error_type="CancelledError")

    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        worker.add_done_callback(finish_cancelled_worker)
        raise


async def await_tracked_thread(
    tracker: BatchSnapshotTracker,
    spec: ExecutionSpec,
    function: Callable[..., Any],
    *args: Any,
    on_worker_terminal: Callable[[], None] | None = None,
) -> Any:
    """Await a thread worker without hiding it when the caller is cancelled.

    ``asyncio.to_thread`` cancellation only cancels the awaiter. The running model
    thread remains the physical execution owner, so its done callback is the only
    cancellation terminal boundary for the tracker.
    """
    handle = tracker.begin(spec)
    worker = asyncio.create_task(asyncio.to_thread(function, *args))

    def terminal(*, success: bool, error_type: str | None = None) -> None:
        try:
            handle.finish(success=success, error_type=error_type)
        finally:
            if on_worker_terminal is not None:
                on_worker_terminal()

    def finish_cancelled_worker(task: "asyncio.Task[Any]") -> None:
        try:
            task.result()
        except BaseException:
            pass
        terminal(success=False, error_type="CancelledError")

    try:
        result = await asyncio.shield(worker)
    except asyncio.CancelledError:
        worker.add_done_callback(finish_cancelled_worker)
        raise
    except BaseException as exc:
        terminal(success=False, error_type=type(exc).__name__)
        raise
    else:
        terminal(success=True)
        return result


def snapshot_collection(*snapshots: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "ego.service-batch-snapshot-collection.v1", "snapshots": [dict(snapshot) for snapshot in snapshots]}
