"""Open-loop benchmark runner: probe, sweep offered levels, write raw artifacts.

A ``BenchmarkRunner`` run:

1. Probes configured live endpoints once (no polling, no waiting for lanes).
2. For each live API with a payload manifest, runs a sweep of offered-load levels
   (underload -> knee -> saturation -> overload) through the open-loop generator.
3. Writes raw ``items.jsonl`` / ``levels.csv`` / ``batches.csv`` / ``manifest.json``
   / ``run_manifest.json`` artifacts under a fresh run directory.
4. Returns the run manifest so the harness can be re-invoked later as more lanes
   become available.

The runner is transport-agnostic: the gateway, probe transport, and clock are all
injected. Tests inject a deterministic fake HTTP server and a fake clock; live
benchmarks inject an httpx gateway and ``time.monotonic``.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ego_annotation.serving.benchmark.artifacts import (
    write_batches_csv,
    write_items_jsonl,
    write_levels_csv,
    write_manifest_json,
    write_run_manifest,
)
from ego_annotation.serving.benchmark.endpoints import (
    EndpointObservation,
    EndpointProbeConfig,
    RunManifest,
    build_run_manifest,
    probe_endpoints_once,
)
from ego_annotation.serving.benchmark.generator import LevelRunResult, OpenLoopGenerator
from ego_annotation.serving.benchmark.manifest import PayloadManifest
from ego_annotation.serving.benchmark.metrics import LevelSummary, ItemRecord, summarize
from ego_annotation.serving.gateway import ModelServiceGateway
from ego_annotation.serving.router import ModelApiName, ModelServiceRouter


@dataclass(frozen=True)
class ApiBenchmarkPlan:
    """One API's benchmark plan: manifest + offered-level sweep."""

    api_name: ModelApiName
    manifest: PayloadManifest
    offered_levels: tuple   # tuple[OfferedLevel]


@dataclass
class BenchmarkRunResult:
    run_id: str
    run_dir: Path
    run_manifest: RunManifest
    summaries: list[LevelSummary] = field(default_factory=list)
    all_records: list[ItemRecord] = field(default_factory=list)
    level_results: list[LevelRunResult] = field(default_factory=list)


class BenchmarkRunner:
    """Orchestrates an open-loop benchmark run over live endpoints.

    ``gateway`` and ``probe_transport`` are the only network boundaries. ``clock``
    defaults to ``time.monotonic``. The runner writes artifacts under ``base_dir``.
    """

    def __init__(
        self,
        *,
        router: ModelServiceRouter,
        gateway: ModelServiceGateway,
        probe_transport,
        base_dir: Path,
        probe_config: EndpointProbeConfig | None = None,
        clock: Callable[[], float] | None = None,
        generator: OpenLoopGenerator | None = None,
    ) -> None:
        self._router = router
        self._gateway = gateway
        self._probe_transport = probe_transport
        self._base_dir = Path(base_dir)
        self._probe_config = probe_config or EndpointProbeConfig()
        self._clock = clock or time.monotonic
        self._generator = generator or OpenLoopGenerator(gateway, clock=self._clock)

    async def run(
        self,
        plans: Sequence[ApiBenchmarkPlan],
        *,
        run_id: str | None = None,
        apis: Sequence[ModelApiName | str] | None = None,
    ) -> BenchmarkRunResult:
        run_id = run_id or f"bench-{uuid.uuid4().hex[:12]}"
        run_dir = self._base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Probe live endpoints once.
        observations = await probe_endpoints_once(
            self._router, self._probe_transport, apis=apis, config=self._probe_config, clock=self._clock
        )
        live_apis = {o.api_name for o in observations if o.live}

        summaries: list[LevelSummary] = []
        all_records: list[ItemRecord] = []
        level_results: list[LevelRunResult] = []
        artifact_paths: dict[str, str] = {}

        for plan in plans:
            if plan.api_name not in live_apis:
                # Endpoint not live at probe time: skip, record in run manifest notes.
                continue
            manifest_path = run_dir / f"manifest_{plan.api_name.value}.json"
            write_manifest_json(manifest_path, plan.manifest)
            api_records: list[ItemRecord] = []
            for level in plan.offered_levels:
                result = await self._generator.run_level(plan.manifest, level)
                level_results.append(result)
                api_records.extend(result.records)
                summary = summarize(
                    result.records,
                    api_name=plan.api_name.value,
                    offered_intensity_per_s=level.offered_intensity_per_s,
                    duration_s=result.duration_s,
                )
                summaries.append(summary)
            all_records.extend(api_records)
            items_path = run_dir / f"items_{plan.api_name.value}.jsonl"
            batches_path = run_dir / f"batches_{plan.api_name.value}.csv"
            write_items_jsonl(items_path, api_records)
            write_batches_csv(batches_path, api_records)
            artifact_paths[f"{plan.api_name.value}.items"] = str(items_path)
            artifact_paths[f"{plan.api_name.value}.batches"] = str(batches_path)
            artifact_paths[f"{plan.api_name.value}.manifest"] = str(manifest_path)

        levels_csv = run_dir / "levels.csv"
        write_levels_csv(levels_csv, summaries)
        artifact_paths["levels_csv"] = str(levels_csv)

        run_manifest = build_run_manifest(
            run_id=run_id,
            observations=observations,
            probe_config=self._probe_config,
            artifact_paths=artifact_paths,
            notes=tuple(
                f"skipped {p.api_name.value}: endpoint not live at probe time"
                for p in plans if p.api_name not in live_apis
            ),
            clock=self._clock,
        )
        write_run_manifest(run_dir / "run_manifest.json", run_manifest)
        return BenchmarkRunResult(
            run_id=run_id,
            run_dir=run_dir,
            run_manifest=run_manifest,
            summaries=summaries,
            all_records=all_records,
            level_results=level_results,
        )
