#!/usr/bin/env python3
"""Bound one Ray Serve log in place without disconnecting inherited O_APPEND writers.

The guard holds an exclusive flock on the *log inode*, snapshots only the trailing
``max_retained_bytes``, keeps the last ``max_lines`` from that snapshot, then uses
ftruncate/write on the already-open descriptor.  Rename rotation is intentionally
not used: inherited Ray file descriptors must continue writing to this inode.

Writers do not cooperate with the flock. A write racing between the trailing snapshot
and ftruncate can be lost; writes after truncate land in the retained inode. This is
an explicit diagnostic-log tradeoff that prevents an unbounded filesystem failure.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import sys
import time
from dataclasses import dataclass
from typing import Sequence

DEFAULT_THRESHOLD_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_RETAINED_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_LINES = 5000
DEFAULT_INTERVAL_S = 1.0


@dataclass(frozen=True)
class CompactionResult:
    compacted: bool
    inode: int
    before_bytes: int
    after_bytes: int
    retained_lines: int


def _retained_tail(fd: int, *, size: int, max_retained_bytes: int, max_lines: int) -> tuple[bytes, int]:
    start = max(0, size - max_retained_bytes)
    data = os.pread(fd, min(size, max_retained_bytes), start)
    lines = data.splitlines(keepends=True)
    retained = b"".join(lines[-max_lines:])
    # A malformed huge line has no newline; the pread cap still bounds the result.
    return retained[-max_retained_bytes:], min(len(lines), max_lines)


def compact_if_needed(
    fd: int,
    *,
    threshold_bytes: int = DEFAULT_THRESHOLD_BYTES,
    max_retained_bytes: int = DEFAULT_MAX_RETAINED_BYTES,
    max_lines: int = DEFAULT_MAX_LINES,
) -> CompactionResult:
    """Compact one already-locked log descriptor while preserving its inode."""
    if threshold_bytes <= 0 or max_retained_bytes <= 0 or max_lines <= 0:
        raise ValueError("threshold, retained-byte cap, and line target must be positive")
    before = os.fstat(fd)
    if before.st_size <= threshold_bytes:
        return CompactionResult(False, before.st_ino, before.st_size, before.st_size, 0)
    retained, retained_lines = _retained_tail(
        fd, size=before.st_size, max_retained_bytes=max_retained_bytes, max_lines=max_lines,
    )
    os.ftruncate(fd, 0)
    # O_APPEND on this descriptor makes the retained diagnostic tail and every
    # pre-existing writer's subsequent bytes land on the same inode.
    if retained:
        os.write(fd, retained)
    after = os.fstat(fd)
    return CompactionResult(True, after.st_ino, before.st_size, after.st_size, retained_lines)


def _lock_log(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o640)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError(f"log guard already owns {path}") from None
    return fd


def run_guard(
    path: Path,
    *,
    threshold_bytes: int,
    max_retained_bytes: int,
    max_lines: int,
    interval_s: float,
    once: bool,
) -> int:
    if interval_s <= 0:
        raise ValueError("interval must be positive")
    fd = _lock_log(path)
    print(
        f"log-guard started path={path} inode={os.fstat(fd).st_ino} threshold_bytes={threshold_bytes} "
        f"max_retained_bytes={max_retained_bytes} max_lines={max_lines} interval_s={interval_s}",
        flush=True,
    )
    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    previous_term = signal.signal(signal.SIGTERM, stop)
    previous_int = signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            result = compact_if_needed(
                fd, threshold_bytes=threshold_bytes, max_retained_bytes=max_retained_bytes, max_lines=max_lines,
            )
            if result.compacted:
                print(
                    f"log-guard compacted path={path} inode={result.inode} before_bytes={result.before_bytes} "
                    f"after_bytes={result.after_bytes} retained_lines={result.retained_lines}",
                    flush=True,
                )
            if once:
                return 0
            time.sleep(interval_s)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    print(f"log-guard stopped path={path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bound one Ray Serve log in place while preserving its inode")
    parser.add_argument("--log-path", required=True, type=Path)
    parser.add_argument("--threshold-bytes", type=int, default=DEFAULT_THRESHOLD_BYTES)
    parser.add_argument("--max-retained-bytes", type=int, default=DEFAULT_MAX_RETAINED_BYTES)
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    parser.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--once", action="store_true", help="perform one bounded compaction check and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_guard(
            args.log_path, threshold_bytes=args.threshold_bytes, max_retained_bytes=args.max_retained_bytes,
            max_lines=args.max_lines, interval_s=args.interval_s, once=args.once,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"log-guard error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI process
    raise SystemExit(main())
