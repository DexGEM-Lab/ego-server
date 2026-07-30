"""Durable foreground owner for a live GPU6 Cosmos3 Ray Serve application.

The Ray head and Serve controller are daemon processes, but a successful cutover must
not be mistaken for a completed, disposable client command.  This driver connects to
the explicitly addressed GPU6 cluster, proves the Serve application is healthy, then
waits in the tmux-owned foreground until an operator intentionally ends that window.
It never tears down or restores services: rollback is only armed before acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time
from typing import Sequence


def _write_state(path: Path, *, event: str, address: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"event": event, "address": address, "pid": __import__("os").getpid(), "time_s": time.time()}) + "\n",
        encoding="utf-8",
    )


def verify_live_application(address: str, application: str = "cosmos3") -> None:
    """Connect only to the explicit GPU6 head and require an active application."""
    import ray
    from ray import serve

    ray.init(address=address, namespace="cosmos3-resident-driver", ignore_reinit_error=True)
    status = serve.status()
    app = status.applications.get(application)
    if app is None:
        raise RuntimeError(f"Serve application {application!r} is absent")
    if app.status.value not in {"RUNNING", "DEPLOYING"}:
        raise RuntimeError(f"Serve application {application!r} status is {app.status.value}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hold the accepted Cosmos3 Serve driver in a durable tmux foreground")
    parser.add_argument("--address", default="127.0.0.1:26801")
    parser.add_argument("--application", default="cosmos3")
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    state_file = Path(args.state_file)
    verify_live_application(args.address, args.application)
    _write_state(state_file, event="driver_verified", address=args.address)
    if args.check_only:
        return 0

    stopping = False

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    _write_state(state_file, event="driver_holding", address=args.address)
    while not stopping:
        signal.pause()
    _write_state(state_file, event="driver_stopped", address=args.address)
    return 0


if __name__ == "__main__":  # pragma: no cover - invoked in durable tmux only
    raise SystemExit(main())
