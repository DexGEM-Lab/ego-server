"""Run-aligned GPU and profiler evidence contracts for experiments.

The sampler records a real interval around the offered-load run.  Empty, stale,
or unrelated samples cannot be mistaken for a bandwidth measurement.
"""
from __future__ import annotations

import json
import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class GpuSample:
    timestamp_s: float
    gpu_id: int
    gpu_uuid: str
    utilization_gpu_pct: float
    memory_used_bytes: int
    experiment_id: str
    release_digest: str
    level: str | None = None

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


class NvmlSampler:
    """Best-effort NVML sampler with explicit start/end boundaries.

    NVML is context evidence, not active-window bandwidth attribution; NCU/CUPTI
    is validated separately.  ``sample_fn`` is injectable for GPU-free tests.
    """

    def __init__(self, *, gpu_ids: Sequence[int], gpu_uuids: Mapping[int, str], experiment_id: str, release_digest: str,
                 sample_fn: Callable[[int], Mapping[str, object]] | None = None, clock: Callable[[], float] = time.monotonic,
                 interval_s: float = 0.2):
        self.gpu_ids = tuple(gpu_ids)
        self.gpu_uuids = dict(gpu_uuids)
        self.experiment_id = experiment_id
        self.release_digest = release_digest
        self.sample_fn = sample_fn or self._nvml_sample
        self.clock = clock
        if interval_s <= 0:
            raise ValueError("sampling interval must be positive")
        self.interval_s = interval_s
        self.started_at_s: float | None = None
        self.ended_at_s: float | None = None
        self.samples: list[GpuSample] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._level: str | None = None

    @staticmethod
    def _nvml_sample(gpu_id: int) -> Mapping[str, object]:
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            return {"gpu_uuid": uuid.decode() if isinstance(uuid, bytes) else str(uuid), "utilization_gpu_pct": util.gpu, "memory_used_bytes": mem.used}
        except ImportError:
            # The exact ray_serve_hawor ABI has no pynvml; nvidia-smi requires no
            # Python package and is indexed by the same physical GPU id.
            import subprocess

            query = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu_id),
                    "--query-gpu=uuid,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if query.returncode != 0 or not query.stdout.strip():
                raise RuntimeError(
                    f"nvidia-smi fallback failed for GPU{gpu_id}: rc={query.returncode} "
                    f"stderr={query.stderr.strip()!r}"
                )
            uuid_s, util_s, mem_s = (part.strip() for part in query.stdout.strip().splitlines()[0].split(","))
            return {"gpu_uuid": uuid_s, "utilization_gpu_pct": float(util_s), "memory_used_bytes": int(mem_s) * 1024 * 1024}

    def start(self) -> float:
        self.started_at_s = self.clock()
        self.sample(level=self._level)
        def loop() -> None:
            while not self._stop_event.wait(self.interval_s):
                self.sample(level=self._level)
        self._thread = threading.Thread(target=loop, name="ego-nvml-sampler", daemon=True)
        self._thread.start()
        return self.started_at_s

    def set_level(self, level: str | None) -> None:
        self._level = level

    def sample(self, *, level: str | None = None, timestamp_s: float | None = None) -> None:
        timestamp = self.clock() if timestamp_s is None else timestamp_s
        for gpu_id in self.gpu_ids:
            raw = self.sample_fn(gpu_id)
            self.samples.append(GpuSample(timestamp, gpu_id, str(raw["gpu_uuid"]), float(raw["utilization_gpu_pct"]), int(raw["memory_used_bytes"]), self.experiment_id, self.release_digest, level))

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self.interval_s))
        self.sample(level=self._level)
        self.ended_at_s = self.clock()
        return self.ended_at_s

    def write(self, path: str | Path) -> Path:
        if self.started_at_s is None or self.ended_at_s is None:
            raise ValueError("sampler must start and stop before writing")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": "ego.gpu-samples.v2", "experiment_id": self.experiment_id, "release_digest": self.release_digest,
                   "started_at_s": self.started_at_s, "ended_at_s": self.ended_at_s, "sample_interval_s": self.interval_s,
                   "sample_count": len(self.samples), "samples": [sample.to_dict() for sample in self.samples]}
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target


def validate_gpu_samples(path: str | Path, *, gpu_ids: Sequence[int], experiment_id: str, release_digest: str,
                         run_start_s: float, run_end_s: float, gpu_uuids: Mapping[int, str] | None = None,
                         min_samples_per_gpu: int = 2) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") not in {"ego.gpu-samples.v2", "ego.gpu-samples.v1"}:
        raise ValueError("unexpected GPU telemetry schema")
    if raw.get("experiment_id") != experiment_id or raw.get("release_digest") != release_digest:
        raise ValueError("GPU telemetry experiment/release identity mismatch")
    start, end = float(raw.get("started_at_s", -1)), float(raw.get("ended_at_s", -1))
    if start > run_start_s or end < run_end_s:
        raise ValueError("GPU telemetry interval does not cover the complete experiment run")
    samples = raw.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("GPU telemetry samples are empty")
    wanted = set(gpu_ids)
    by_gpu: dict[int, list[Mapping[str, object]]] = {gpu: [] for gpu in wanted}
    for sample in samples:
        if not isinstance(sample, Mapping) or sample.get("gpu_id") not in wanted:
            continue
        if sample.get("experiment_id") != experiment_id or sample.get("release_digest") != release_digest:
            raise ValueError("GPU telemetry sample identity mismatch")
        gpu_id = int(sample["gpu_id"])
        if gpu_uuids and gpu_id in gpu_uuids and sample.get("gpu_uuid") != gpu_uuids[gpu_id]:
            raise ValueError(f"GPU telemetry UUID mismatch for GPU{gpu_id}")
        ts = float(sample.get("timestamp_s", -1))
        if run_start_s <= ts <= run_end_s:
            by_gpu[int(sample["gpu_id"])].append(sample)
    missing = [gpu for gpu, values in by_gpu.items() if len(values) < min_samples_per_gpu]
    if missing:
        raise ValueError(f"GPU telemetry has insufficient run-overlap samples for GPUs {missing}")
    return dict(raw)


def validate_profiler_artifact(path: str | Path, *, experiment_id: str, release_digest: str,
                               run_start_s: float, run_end_s: float) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("tool") not in {"ncu", "cupti"}:
        raise ValueError("profiler tool must be ncu or cupti")
    if raw.get("experiment_id") != experiment_id or raw.get("release_digest") != release_digest:
        raise ValueError("profiler identity mismatch")
    if float(raw.get("end_s", -1)) < run_start_s or float(raw.get("start_s", -1)) > run_end_s:
        raise ValueError("profiler interval does not overlap experiment")
    kernels, counters = raw.get("kernels"), raw.get("counters")
    if not isinstance(kernels, list) or not kernels or not isinstance(counters, Mapping) or not counters:
        raise ValueError("profiler artifact has no kernels/counters; bandwidth attribution unavailable")
    return dict(raw)
