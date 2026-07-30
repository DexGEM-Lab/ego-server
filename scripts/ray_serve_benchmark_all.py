"""Unified open-loop benchmark: probe all lane endpoints once, sweep the live ones.

This is the single command that drives the durable gateway/benchmark lane:

* it probes every configured Serve lane endpoint **exactly once** (never polls, never
  waits for a model lane to come up, never modifies a deployment);
* for each endpoint that was live at probe time it runs an isolated open-loop
  offered-load sweep (one API at a time, native work units);
* it writes a unified ``run_manifest.json`` (endpoint observations + live/down APIs +
  artifact paths), a combined ``levels.csv``, per-API ``items_<api>.jsonl`` /
  ``batches_<api>.csv`` / ``manifest_<api>.json``, and the throughput-latency +
  batch-distribution plots;
* endpoints that were down at probe time are recorded as ``down`` and skipped for this
  run — re-running the command re-probes once, so sweeps start automatically as lanes
  become ready.

The command never touches model deployments: it only reads the public Serve HTTP
endpoints. Heavy live sweeps run on the server (``EGO_SERVE_HOST``); the local
workstation only runs the ``--fake-server`` smoke path.

Usage (deterministic fake-server smoke, no Ray/GPU):

    python -m scripts.ray_serve_benchmark_all \\
        --out /tmp/ego_bench_all_smoke --fake-server

Usage (live, on the server; requires preserved real per-model payloads):

    EGO_SERVE_HOST=dex-a800 python -m scripts.ray_serve_benchmark_all \\
        --payload-dir /vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_payloads \\
        --out /vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_benchmarks

To benchmark the current bare Cosmos3 vLLM process (transient baseline, port 8001)
instead of the future Ray-managed GPU6 lane port 28006, pass ``--cosmos3-baseline``;
this applies an explicit, labeled override so the run manifest records that cosmos3
was measured against the baseline, not the canonical lane.

Artifact schema (under ``<out>/<run_id>/``):

    run_manifest.json              run id, probe config, per-endpoint observations,
                                  live_apis, down_apis, artifact_paths, notes
    levels.csv                     one row per (api, offered-intensity) level summary
    items_<api>.jsonl              one ItemRecord per settled item (live+swept APIs)
    batches_<api>.csv              one row per distinct server batch
    manifest_<api>.json            payload manifest (distinct hashes + provenance)
    plots/throughput_latency_<api>.png
    plots/batch_distribution_<api>.png
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from ego_annotation.serving.benchmark.endpoints import EndpointProbeConfig
from ego_annotation.serving.benchmark.generator import OfferedLevel
from ego_annotation.serving.benchmark.plotting import plot_batch_distribution, plot_throughput_latency
from ego_annotation.serving.benchmark.runner import ApiBenchmarkPlan, BenchmarkRunner
from ego_annotation.serving.gateway import ModelServiceGateway, RetryPolicy
from ego_annotation.serving.router import (
    COSMOS3_BASELINE_URL,
    ModelApiName,
    ModelServiceRouter,
    cosmos3_baseline_override,
)

# Reuse the single-API script's per-API manifest builder so both commands share one
# source of truth for model-native payloads.
from scripts.ray_serve_benchmark import build_manifest_for_api


# Default offered-load sweep per API, in native work units/s. Levels span
# underload -> knee -> saturation -> overload. They are conservative defaults; pass
# --levels to override every API uniformly, or keep these per-API native defaults.
DEFAULT_LEVELS_PER_API: dict[ModelApiName, tuple[float, ...]] = {
    ModelApiName.UNIDEPTH_INFER: (1.0, 4.0, 16.0, 64.0, 256.0),        # images/s
    ModelApiName.HANDS_DETECT: (1.0, 4.0, 16.0, 64.0, 256.0),          # images/s
    ModelApiName.WILOR_RECONSTRUCT: (4.0, 16.0, 64.0, 256.0, 1024.0),  # crops/s
    ModelApiName.DROID_CREATE_SESSION: (0.5, 1.0, 2.0, 4.0, 8.0),      # sessions/s
    ModelApiName.DROID_PUSH_FRAME: (4.0, 16.0, 64.0, 256.0),           # ready_frames/s
    ModelApiName.DROID_FINALIZE: (0.5, 1.0, 2.0, 4.0, 8.0),            # sessions/s
    ModelApiName.HAWOR_INFER_TRACKS: (1.0, 4.0, 16.0, 64.0),           # track_chunks/s
    ModelApiName.HAWOR_INFILLER_FILL: (0.5, 1.0, 2.0, 4.0),            # temporal_windows/s
    ModelApiName.COSMOS3_REASON: (1.0, 4.0, 16.0, 64.0),               # media_requests/s
}

# APIs swept by default. DROID push_frame is stateful: a live sweep needs session_ids
# from real create_session calls. Until DROID is live, push_frame is included with a
# seeded session_id pool (the manifest builder); when DROID comes up, run create first
# and feed the returned session_ids (documented residual dependency).
DEFAULT_SWEEP_APIS: tuple[ModelApiName, ...] = tuple(DEFAULT_LEVELS_PER_API.keys())


def _build_plans(
    apis: tuple[ModelApiName, ...],
    *,
    manifest_count: int,
    target_completed: int,
    max_offered: int,
    levels_override: tuple[float, ...] | None,
    payload_dir: Path | None,
    allow_synthetic: bool,
) -> list[ApiBenchmarkPlan]:
    plans: list[ApiBenchmarkPlan] = []
    for api in apis:
        manifest = build_manifest_for_api(
            api, count=manifest_count, payload_dir=payload_dir, allow_synthetic=allow_synthetic,
        )
        rates = levels_override if levels_override else DEFAULT_LEVELS_PER_API[api]
        offered_levels = tuple(
            OfferedLevel(
                api_name=api,
                offered_intensity_per_s=rate,
                target_completed=target_completed,
                max_offered=max_offered,
            )
            for rate in rates
        )
        plans.append(ApiBenchmarkPlan(api_name=api, manifest=manifest, offered_levels=offered_levels))
    return plans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified open-loop benchmark: probe once, sweep live lanes")
    parser.add_argument("--out", required=True, help="output base directory")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--apis", default=None,
        help="comma-separated API names to sweep (default: all 9 / 7 services)",
    )
    parser.add_argument(
        "--levels", default=None,
        help="comma-separated offered intensities applied uniformly to every swept API "
             "(default: per-API native levels)",
    )
    parser.add_argument("--target-completed", type=int, default=100)
    parser.add_argument("--max-offered", type=int, default=400)
    parser.add_argument("--manifest-count", type=int, default=400, help="distinct payload items per API")
    parser.add_argument(
        "--payload-dir", type=Path,
        help="directory containing real <api>.json payload-source descriptors and binary parts; required unless --fake-server",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--deadline-s", type=float, default=5.0)
    parser.add_argument("--wire-format", choices=("multipart", "envelope"), default="multipart", help="explicit transport treatment; multipart remains the default")
    parser.add_argument("--probe-timeout-s", type=float, default=2.0)
    parser.add_argument("--health-path", default="/health")
    parser.add_argument(
        "--fake-server", action="store_true",
        help="deterministic in-process fake HTTP server (smoke; no Ray, no GPU)",
    )
    parser.add_argument("--serve-host", default="127.0.0.1", help="host for the fake-server smoke path")
    parser.add_argument(
        "--cosmos3-baseline", action="store_true",
        help="route cosmos3.reason at the bare baseline (port 8001) via an explicit, "
             "labeled override instead of the canonical lane port 28006",
    )
    args = parser.parse_args(argv)

    apis: tuple[ModelApiName, ...]
    if args.apis:
        apis = tuple(ModelApiName(a.strip()) for a in args.apis.split(",") if a.strip())
    else:
        apis = DEFAULT_SWEEP_APIS
    if not args.fake_server and args.payload_dir is None:
        parser.error("--payload-dir is required for live benchmarks; synthetic payloads are limited to --fake-server")
    levels_override = (
        tuple(float(x) for x in args.levels.split(",") if x.strip()) if args.levels else None
    )
    plans = _build_plans(
        apis, manifest_count=args.manifest_count, target_completed=args.target_completed,
        max_offered=args.max_offered, levels_override=levels_override,
        payload_dir=args.payload_dir, allow_synthetic=args.fake_server,
    )

    base_overrides: dict[ModelApiName, str] = {}
    cosmos3_baseline_note = ""
    if args.cosmos3_baseline:
        base_overrides.update(cosmos3_baseline_override())
        cosmos3_baseline_note = (
            "cosmos3.reason routed at the bare baseline "
            f"({COSMOS3_BASELINE_URL}) via explicit --cosmos3-baseline override; "
            "the canonical lane port 28006 was NOT used."
        )

    out_dir = Path(args.out)
    probe_config = EndpointProbeConfig(health_path=args.health_path, timeout_s=args.probe_timeout_s)

    if args.fake_server:
        from ego_annotation.serving.benchmark.fakeserver import (
            FakeHttpGatewayTransport,
            FakeHttpProbeTransport,
            start_fake_server,
        )

        async def _run() -> int:
            server = await start_fake_server(host=args.serve_host, port=0)
            gateway_transport = FakeHttpGatewayTransport(server.runner)
            probe_transport = FakeHttpProbeTransport(server.runner)
            try:
                # Point every swept API at the fake server so the smoke path
                # exercises real multipart bytes end-to-end.
                fake_base = f"http://{args.serve_host}:{server.port}"
                overrides = dict(base_overrides)
                overrides.update({api: f"{fake_base}/{api.value}" for api in apis})
                router = ModelServiceRouter.canonical().with_overrides(
                    {key.value: value for key, value in overrides.items()}
                )
                gateway = ModelServiceGateway(
                    router, gateway_transport,
                    retry_policy=RetryPolicy(max_attempts=args.max_attempts, deadline_s=args.deadline_s),
                    wire_format=args.wire_format,
                )
                runner = BenchmarkRunner(
                    router=router, gateway=gateway, probe_transport=probe_transport,
                    base_dir=out_dir, probe_config=probe_config,
                )
                notes = [cosmos3_baseline_note] if cosmos3_baseline_note else []
                result = await runner.run(plans, run_id=args.run_id)
                _emit(result, notes=notes)
                await _write_plots(result.run_dir)
                return 0
            finally:
                await gateway_transport.aclose()
                await probe_transport.aclose()
                await server.stop()

        return asyncio.run(_run())

    # Live: httpx transport. Probes each lane endpoint once; never polls, never
    # modifies a deployment.
    router = ModelServiceRouter.canonical().with_overrides(
        {key.value: value for key, value in base_overrides.items()}
    )
    gateway = ModelServiceGateway.with_httpx(
        router, retry_policy=RetryPolicy(max_attempts=args.max_attempts, deadline_s=args.deadline_s),
        wire_format=args.wire_format,
    )

    async def _run_live() -> int:
        from scripts._benchmark_live_transport import HttpxProbeTransport

        runner = BenchmarkRunner(
            router=router, gateway=gateway, probe_transport=HttpxProbeTransport(),
            base_dir=out_dir, probe_config=probe_config,
        )
        notes = [cosmos3_baseline_note] if cosmos3_baseline_note else []
        result = await runner.run(plans, run_id=args.run_id)
        _emit(result, notes=notes)
        await _write_plots(result.run_dir)
        await gateway.aclose()
        return 0

    return asyncio.run(_run_live())


def _emit(result, *, notes: list[str]) -> None:
    print(f"run_id={result.run_id}")
    print(f"run_dir={result.run_dir}")
    print(f"live_apis={list(result.run_manifest.live_apis)}")
    print(f"down_apis={list(result.run_manifest.down_apis)}")
    print(f"summaries={len(result.summaries)}")
    for s in result.summaries:
        print(
            f"  {s.api_name} offered={s.offered_intensity_per_s}/s "
            f"completed={s.completed_count} rejected={s.rejected_count} "
            f"throughput={s.throughput_work_units_per_s:.2f}wu/s "
            f"p50={s.response_latency_p50_ms}ms p95={s.response_latency_p95_ms}ms "
            f"batch_mean={s.batch_size_mean} loads={s.model_load_count_min}-{s.model_load_count_max}"
        )
    for note in notes:
        if note:
            print(f"note={note}")
    # Surface the unified manifest path + artifact schema for reproducibility.
    run_manifest_path = result.run_dir / "run_manifest.json"
    print(f"run_manifest={run_manifest_path}")
    rm = json.loads(run_manifest_path.read_text())
    print(f"artifact_paths={list(rm.get('artifact_paths', {}).keys())}")


async def _write_plots(run_dir: Path) -> None:
    out_dir = run_dir / "plots"
    levels_csv = run_dir / "levels.csv"
    written_tl: list[Path] = []
    if levels_csv.exists():
        written_tl = plot_throughput_latency(levels_csv, out_dir)
    written_bd: list[Path] = []
    for items_path in sorted(run_dir.glob("items_*.jsonl")):
        written_bd.extend(plot_batch_distribution(items_path, out_dir))
    print(f"plots={len(written_tl)} throughput-latency + {len(written_bd)} batch-distribution -> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
