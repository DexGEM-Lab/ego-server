"""Live, disposable tests for same-inode bounded Ray Serve log compaction."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from scripts import guard_ego_serve_log as guard


def _wait_for(predicate, *, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    tick = threading.Event()
    while time.monotonic() < deadline:
        if predicate():
            return True
        tick.wait(0.01)
    return predicate()


def test_live_o_append_writer_guard_preserves_inode_and_bounds_retention(tmp_path: Path) -> None:
    log = tmp_path / "gpu0.log"
    writer_fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
    try:
        os.write(writer_fd, b"old-line-" + b"x" * 48 + b"\n")
        for _ in range(80):
            os.write(writer_fd, b"old-line-" + b"x" * 48 + b"\n")
        inode_before = os.fstat(writer_fd).st_ino
        command = [
            sys.executable, str(Path(guard.__file__)), "--log-path", str(log),
            "--threshold-bytes", "512", "--max-retained-bytes", "256",
            "--max-lines", "5", "--interval-s", "0.02",
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert _wait_for(lambda: log.stat().st_size <= 256)
            retained = log.read_bytes()
            assert len(retained) <= 256
            assert len(retained.splitlines()) <= 5
            assert log.stat().st_ino == inode_before

            # The original O_APPEND fd stays valid after in-place ftruncate/write.
            os.write(writer_fd, b"AFTER_COMPACTION\n")
            assert _wait_for(lambda: b"AFTER_COMPACTION\n" in log.read_bytes())

            second = subprocess.run(command + ["--once"], capture_output=True, text=True, check=False, timeout=3)
            assert second.returncode == 2
            assert "already owns" in second.stderr
        finally:
            process.terminate()
            stdout, stderr = process.communicate(timeout=3)
        assert "log-guard started" in stdout
        assert "log-guard compacted" in stdout
        assert stderr == ""
    finally:
        os.close(writer_fd)


def test_huge_single_line_never_exceeds_retained_byte_cap(tmp_path: Path) -> None:
    log = tmp_path / "gpu1.log"
    with log.open("wb") as handle:
        handle.write(b"z" * 8192)  # no newline: line count alone cannot bound this case
    inode_before = log.stat().st_ino
    result = subprocess.run(
        [
            sys.executable, str(Path(guard.__file__)), "--log-path", str(log),
            "--threshold-bytes", "512", "--max-retained-bytes", "256", "--max-lines", "5000", "--once",
        ],
        capture_output=True, text=True, check=False, timeout=3,
    )
    assert result.returncode == 0
    assert log.stat().st_ino == inode_before
    assert log.stat().st_size <= 256
    assert "retained_lines=1" in result.stdout
