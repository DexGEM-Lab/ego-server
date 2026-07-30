"""CLI wrapper for verified sticky D1/D2/D4 DROID scaling benchmarks."""
from __future__ import annotations

import asyncio

from benchmarks.ray_serve.benchmark_droid_open_loop import async_main, parse_args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main(parse_args())))
