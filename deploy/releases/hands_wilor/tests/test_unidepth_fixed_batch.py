from __future__ import annotations

from typing import Any, cast

import numpy as np

from ego_annotation.serving.benchmark.unidepth_fixed_batch import run_fixed_batch_sweep
from ego_annotation.serving.unidepth import UniDepthBackendResult


def test_fixed_batch_sweep_uses_distinct_inputs_and_reports_cuda_allocator_scaling():
    calls = []
    class Backend:
        def infer(self, rgb):
            calls.append(rgb.shape)
            b, _, h, w = rgb.shape
            return UniDepthBackendResult(
                {"depth": np.ones((b, 1, h, w), np.float32)},
                {"cuda_model_ms": float(b), "h2d_ms": .1, "d2h_ms": .2,
                 "allocator_memory": {"allocated_bytes": b, "reserved_bytes": b + 1,
                                      "max_allocated_bytes": b + 2, "max_reserved_bytes": b + 3}},
            )
    tensors = [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(16)]
    result = run_fixed_batch_sweep(Backend(), tensors=tensors, payload_hashes=[f"hash-{i}" for i in range(16)],
                                   batch_sizes=(1, 2, 4, 8, 16), warmup_forwards=1, repeats=2)
    rows = cast(list[dict[str, Any]], result["rows"])
    summaries = cast(list[dict[str, Any]], result["summaries"])
    assert calls[0] == (1, 3, 2, 3)
    assert {row["batch_size"] for row in rows} == {1, 2, 4, 8, 16}
    assert all(len(row["payload_hashes"]) == len(set(row["payload_hashes"])) for row in rows)
    summary16 = next(row for row in summaries if row["batch_size"] == 16)
    assert summary16["images_per_s"] == 1000.0 and summary16["peak_allocator_reserved_bytes"] == 19
    assert all(row["wall_ms"] >= 0.0 for row in rows)
    assert summary16["wall_ms_mean"] >= 0.0
    assert summary16["wall_ms_p50"] >= 0.0
    assert summary16["wall_ms_p95"] >= 0.0
    assert summary16["wall_ms_per_image"] >= 0.0
