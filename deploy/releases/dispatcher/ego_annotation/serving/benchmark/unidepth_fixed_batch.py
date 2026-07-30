"""Direct physical UniDepth batch benchmark, deliberately outside HTTP/Ray batching.

One resident backend is loaded once.  Each measured forward receives a contiguous
BCHW tensor built from B distinct 540x960 corpus tensors, so B=1/2/4/8/16 measures
model scaling rather than client serialization or Serve queue formation.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from ego_annotation.serving.unidepth import UniDepthBackendResult, stack_to_bchw


REQUIRED_FIXED_BATCHES = (1, 2, 4, 8, 16)


class FixedBatchBackend(Protocol):
    def infer(self, rgb: Any) -> Mapping[str, Any] | UniDepthBackendResult: ...


@dataclass(frozen=True)
class FixedBatchRow:
    batch_size: int
    repeat: int
    payload_hashes: tuple[str, ...]
    wall_ms: float
    cuda_model_ms: float | None
    h2d_ms: float | None
    d2h_ms: float | None
    allocated_bytes: int | None
    reserved_bytes: int | None
    max_allocated_bytes: int | None
    max_reserved_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None


def _integer(mapping: Mapping[str, object], name: str) -> int | None:
    value = mapping.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Linearly interpolate a percentile without a NumPy dependency at call sites."""
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run_fixed_batch_sweep(
    backend: FixedBatchBackend,
    *,
    tensors: Sequence[np.ndarray],
    payload_hashes: Sequence[str],
    batch_sizes: Sequence[int] = REQUIRED_FIXED_BATCHES,
    warmup_forwards: int = 5,
    repeats: int = 20,
) -> dict[str, object]:
    """Run a direct fixed-B sweep and return CUDA/allocator records per forward.

    The caller owns model construction and NVML sampling.  This function neither
    creates a server nor sends an HTTP request; its only model operation is the
    backend's exact resident ``infer`` call.
    """
    if warmup_forwards < 1 or repeats < 1:
        raise ValueError("warmup_forwards and repeats must be positive")
    if len(tensors) != len(payload_hashes) or len(set(payload_hashes)) != len(payload_hashes):
        raise ValueError("fixed-batch sweep requires one distinct hash per tensor")
    if not batch_sizes or any(batch not in (*REQUIRED_FIXED_BATCHES, 32) for batch in batch_sizes):
        raise ValueError("batch sizes must be selected from 1,2,4,8,16,32")
    if any(batch > len(tensors) for batch in batch_sizes):
        raise ValueError("corpus lacks distinct tensors for a requested physical batch")
    rows: list[FixedBatchRow] = []
    for batch_size in batch_sizes:
        for forward_index in range(warmup_forwards + repeats):
            offset = (forward_index * batch_size) % len(tensors)
            indices = tuple((offset + item) % len(tensors) for item in range(batch_size))
            if len(set(indices)) != batch_size:
                raise ValueError("one physical batch reused a payload")
            # The instrumented resident CUDA backend synchronizes after D2H before
            # returning. Its preceding invocation has the same boundary, so this
            # host span begins and ends with completed GPU work rather than merely
            # measuring asynchronous CUDA launch overhead.
            batch = stack_to_bchw([tensors[index] for index in indices])
            wall_started = time.perf_counter()
            result = backend.infer(batch)
            wall_ms = (time.perf_counter() - wall_started) * 1_000.0
            if forward_index < warmup_forwards:
                continue
            diagnostics = result.diagnostics if isinstance(result, UniDepthBackendResult) and result.diagnostics else {}
            allocator = diagnostics.get("allocator_memory") if isinstance(diagnostics, Mapping) else {}
            allocator = allocator if isinstance(allocator, Mapping) else {}
            rows.append(FixedBatchRow(
                batch_size=batch_size, repeat=forward_index - warmup_forwards,
                payload_hashes=tuple(payload_hashes[index] for index in indices), wall_ms=wall_ms,
                cuda_model_ms=_number(diagnostics.get("cuda_model_ms")) if isinstance(diagnostics, Mapping) else None,
                h2d_ms=_number(diagnostics.get("h2d_ms")) if isinstance(diagnostics, Mapping) else None,
                d2h_ms=_number(diagnostics.get("d2h_ms")) if isinstance(diagnostics, Mapping) else None,
                allocated_bytes=_integer(allocator, "allocated_bytes"), reserved_bytes=_integer(allocator, "reserved_bytes"),
                max_allocated_bytes=_integer(allocator, "max_allocated_bytes"), max_reserved_bytes=_integer(allocator, "max_reserved_bytes"),
            ))
    summaries = []
    for batch_size in batch_sizes:
        sample = [row for row in rows if row.batch_size == batch_size]
        cuda = [row.cuda_model_ms for row in sample if row.cuda_model_ms is not None]
        wall = [row.wall_ms for row in sample]
        wall_mean = statistics.mean(wall)
        summaries.append({
            "batch_size": batch_size, "measured_forwards": len(sample),
            "wall_ms_mean": wall_mean,
            "wall_ms_p50": statistics.median(wall),
            "wall_ms_p95": _percentile(wall, .95),
            "wall_ms_per_image": wall_mean / batch_size,
            "cuda_model_ms_mean": statistics.mean(cuda) if cuda else None,
            "images_per_s": (batch_size * 1_000.0 / statistics.mean(cuda)) if cuda else None,
            "cuda_ms_per_image": (statistics.mean(cuda) / batch_size) if cuda else None,
            "peak_allocator_reserved_bytes": max((row.max_reserved_bytes or 0 for row in sample), default=None),
            "peak_allocator_allocated_bytes": max((row.max_allocated_bytes or 0 for row in sample), default=None),
        })
    return {"schema": "ego.unidepth-fixed-physical-batch.v1", "warmup_forwards": warmup_forwards,
            "repeats": repeats, "rows": [row.to_dict() for row in rows], "summaries": summaries}
