"""Plotting commands for throughput-latency and batch distributions.

Produces two plot families from raw artifacts:

* **throughput-latency**: achieved throughput (work units/s) vs offered intensity,
  with p50/p95/p99 response latency on a second axis. One figure per API.
* **batch distribution**: histogram of effective batch sizes (request_count) and
  batch work units per API.

Plots are generated from the raw ``items.jsonl`` / ``levels.csv`` artifacts so every
plotted point traces to a raw result file. The matplotlib backend is forced to
``Agg`` so plotting runs headless on the server.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def _load_levels_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_items_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_throughput_latency(levels_csv: Path, out_dir: Path) -> list[Path]:
    """One throughput-latency figure per API.

    X: offered intensity (work units/s). Left Y: achieved throughput (work units/s).
    Right Y: response latency p50/p95/p99 (ms). Response latency is plotted
    separately from amortized cost (not shown here) to avoid conflation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_levels_csv(levels_csv)
    by_api: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_api[row["api_name"]].append(row)
    written: list[Path] = []
    for api_name, api_rows in by_api.items():
        api_rows.sort(key=lambda r: float(r["offered_intensity_per_s"]))
        offered = [float(r["offered_intensity_per_s"]) for r in api_rows]
        throughput = [_fnum(r["throughput_work_units_per_s"]) for r in api_rows]
        p50 = [_fnum(r["response_latency_p50_ms"]) for r in api_rows]
        p95 = [_fnum(r["response_latency_p95_ms"]) for r in api_rows]
        p99 = [_fnum(r["response_latency_p99_ms"]) for r in api_rows]
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.set_xlabel("offered intensity (work units/s)")
        ax1.set_ylabel("throughput (work units/s)", color="tab:blue")
        ax1.plot(offered, throughput, "o-", color="tab:blue", label="throughput")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx()
        ax2.set_ylabel("response latency (ms)", color="tab:red")
        ax2.plot(offered, p50, "s--", color="tab:orange", label="p50")
        ax2.plot(offered, p95, "^--", color="tab:red", label="p95")
        ax2.plot(offered, p99, "v--", color="tab:purple", label="p99")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax1.set_title(f"throughput vs offered load — {api_name}")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        fig.tight_layout()
        out = out_dir / f"throughput_latency_{_safe(api_name)}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        written.append(out)
    return written


def plot_batch_distribution(items_jsonl: Path, out_dir: Path) -> list[Path]:
    """Batch-size and batch-work-unit histograms per API, from raw item records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_items_jsonl(items_jsonl)
    by_api: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("batch_size") is not None:
            by_api[str(row["api_name"])].append(row)
    written: list[Path] = []
    for api_name, api_rows in by_api.items():
        sizes = [int(r["batch_size"]) for r in api_rows if r.get("batch_size") is not None]
        works = [float(r["batch_work_units"]) for r in api_rows if r.get("batch_work_units") is not None]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        ax1.hist(sizes, bins=range(1, max(sizes) + 2), color="tab:blue", edgecolor="black")
        ax1.set_xlabel("effective batch size (request_count)")
        ax1.set_ylabel("item count")
        ax1.set_title(f"batch size — {api_name}")
        if works:
            ax2.hist(works, bins=20, color="tab:green", edgecolor="black")
            ax2.set_xlabel("batch work units")
            ax2.set_ylabel("item count")
            ax2.set_title(f"batch weight — {api_name}")
        fig.tight_layout()
        out = out_dir / f"batch_distribution_{_safe(api_name)}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        written.append(out)
    return written


def _fnum(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _safe(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")
