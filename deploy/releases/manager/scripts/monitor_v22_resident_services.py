#!/usr/bin/env python3
"""Record resident service metrics and per-process GPU telemetry."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as response:
        payload = json.loads(response.read().decode())
    return payload if isinstance(payload, dict) else {"status": "invalid_json"}


def gpu_snapshot() -> list[dict[str, Any]]:
    command = ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"]
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    rows = []
    for line in proc.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 7:
            rows.append({"index": fields[0], "name": fields[1], "memory_used_mib": fields[2], "memory_total_mib": fields[3], "utilization_gpu_pct": fields[4], "temperature_c": fields[5], "power_w": fields[6]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        row = {"observed_unix": time.time(), "services": {}, "gpus": gpu_snapshot()}
        for model, spec in config["services"].items():
            url = f"http://{config['host']}:{spec['port']}/metrics"
            try:
                row["services"][model] = get_json(url)
            except Exception as exc:
                row["services"][model] = {"status": "unreachable", "error": str(exc), "port": spec["port"]}
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.once:
            return
        time.sleep(max(0.1, args.interval_s))


if __name__ == "__main__":
    main()
