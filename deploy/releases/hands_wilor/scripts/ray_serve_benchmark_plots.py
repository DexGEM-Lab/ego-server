"""Generate throughput-latency and batch-distribution plots from raw benchmark artifacts.

Usage:
    python -m scripts.ray_serve_benchmark_plots --run-dir /tmp/ego_bench_run/<run_id>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ego_annotation.serving.benchmark.plotting import plot_batch_distribution, plot_throughput_latency


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot benchmark artifacts")
    parser.add_argument("--run-dir", required=True, help="path to a benchmark run directory")
    parser.add_argument("--out", default=None, help="output plots directory (default <run-dir>/plots)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "plots"
    levels_csv = run_dir / "levels.csv"

    # Throughput-latency: concatenate per-API levels CSVs if a single levels.csv
    # exists (the runner writes one combined levels.csv).
    written_tl: list[Path] = []
    if levels_csv.exists():
        written_tl = plot_throughput_latency(levels_csv, out_dir)

    # Batch distribution: one per API items_<api>.jsonl.
    written_bd: list[Path] = []
    for items_path in sorted(run_dir.glob("items_*.jsonl")):
        written_bd.extend(plot_batch_distribution(items_path, out_dir))

    print(f"wrote {len(written_tl)} throughput-latency plots and {len(written_bd)} batch plots to {out_dir}")
    for p in written_tl + written_bd:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
