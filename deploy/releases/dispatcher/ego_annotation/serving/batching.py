"""Admission helpers used before a Ray Serve batch callback admits a request.

These helpers never queue requests. Ray Serve owns routing, batching windows, and
queue bounds; this module only validates that a request is compatible with the
resident deployment's single canonical compatibility bucket before it reaches the
batch callback, and computes normalized canonical-image work for ``batch_size_fn``.

The initial UniDepth deployment advertises exactly one canonical HxW compatibility
bucket. Every admitted request is one normalized work unit, so a single Serve batch
callback executes exactly one upstream forward. Incompatible or overweight items are
rejected at admission (before the callback) so a callback never has to split into
several forwards to honor compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, TypeVar

from ego_annotation.serving.contracts import ContractValidationError


T = TypeVar("T")


@dataclass(frozen=True)
class BatchPolicy:
    max_batch_size: int
    batch_wait_timeout_s: float
    max_queued_requests: int

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0 or self.max_queued_requests < 0:
            raise ContractValidationError("batch limits must be positive (queue limit may be zero)")
        if self.batch_wait_timeout_s < 0:
            raise ContractValidationError("batch_wait_timeout_s must be non-negative")


def canonical_batch_size_fn(items: Sequence[T]) -> int:
    """``serve.batch(batch_size_fn=...)`` work function.

    Each admitted request is one normalized canonical-image work unit (validated at
    admission), so the effective batch size is the number of items. Ray fills a batch
    when this reaches ``max_batch_size`` or ``batch_wait_timeout_s`` elapses.
    """
    return len(items)


def total_work(items: Iterable[T], work_units: Callable[[T], int]) -> int:
    return sum(work_units(item) for item in items)


def assert_one_forward(items: Sequence[T], *, policy: BatchPolicy) -> None:
    """A Serve callback must execute exactly one upstream forward.

    With one canonical compatibility bucket and per-request weight 1, a callback
    receives at most ``max_batch_size`` mutually compatible items, which is one
    upstream forward. This asserts the invariant the trace records as
    ``forward_count == 1``.
    """
    if len(items) > policy.max_batch_size:
        raise ContractValidationError(
            f"a Serve callback received {len(items)} items but max_batch_size is "
            f"{policy.max_batch_size}; this would require splitting into multiple forwards"
        )
