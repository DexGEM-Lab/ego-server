#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zjh/ego-annotation-feat-parallel}"
ROOT="${SERVICE_ROOT:-/home/zjh/data/v22_resident_services}"
CONFIG="${CONFIG:-$REPO/configs/v22_resident_services.json}"
STOP_ENABLED="${V22_ARCHIVED_DEPLOYMENT_STOP:-}"

python3 - "$CONFIG" "$ROOT" "$STOP_ENABLED" "$REPO" <<'PY'
import json
import os
import re
import signal
import shutil
import subprocess
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
stop_enabled = sys.argv[3] == "1"
expected_script = str(Path(sys.argv[4]) / "scripts" / "serve_v22_resident_model.py")
services = config["services"]


def fail(message):
    print(f"v22 archive stop refused: {message}", file=sys.stderr)
    raise SystemExit(2)


ss = shutil.which("ss")
if not ss:
    fail("cannot verify listener ownership: ss is unavailable")
ss_result = subprocess.run([ss, "-H", "-ltnp"], capture_output=True, text=True, check=False)
if ss_result.returncode != 0:
    fail(f"cannot verify listener ownership: {ss_result.stderr.strip() or ss_result.returncode}")

listeners = {}
for line in ss_result.stdout.splitlines():
    fields = line.split()
    if len(fields) < 4:
        continue
    try:
        port = int(fields[3].rsplit(":", 1)[1].rstrip("]"))
    except (IndexError, ValueError):
        continue
    listeners.setdefault(port, set()).update(int(pid) for pid in re.findall(r"pid=(\d+)", line))


def read_cmdline(pid):
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError):
        return []
    return [token for token in raw.decode(errors="replace").split("\0") if token]


def flag_value(tokens, flag):
    try:
        return tokens[tokens.index(flag) + 1]
    except (ValueError, IndexError):
        return None


pid_dir = root / "pids"
for model, spec in services.items():
    port = int(spec["port"])
    pid_path = pid_dir / f"{model}.pid"
    if not pid_path.exists():
        print(f"{model}: no pid file ({pid_path}); nothing to inspect")
        continue
    try:
        pid = int(pid_path.read_text().strip())
    except ValueError:
        print(f"{model}: SKIP malformed pid file {pid_path}", file=sys.stderr)
        continue

    tokens = read_cmdline(pid)
    command_ok = (
        bool(tokens)
        and expected_script in tokens
        and flag_value(tokens, "--model") == model
        and flag_value(tokens, "--port") == str(port)
    )
    port_ok = pid in listeners.get(port, set())
    if not command_ok or not port_ok:
        print(
            f"{model}: SKIP strict match failed pid={pid} "
            f"cmdline={'ok' if command_ok else 'mismatch'} "
            f"port={port} owner={'ok' if port_ok else 'mismatch'}",
            file=sys.stderr,
        )
        continue

    if not stop_enabled:
        print(f"{model}: DRY-RUN would SIGTERM pid={pid} on port={port}; set V22_ARCHIVED_DEPLOYMENT_STOP=1 to stop")
        continue

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"{model}: pid={pid} exited after verification; no signal sent")
    except PermissionError as exc:
        print(f"{model}: SIGTERM denied for pid={pid}: {exc}", file=sys.stderr)
        continue
    else:
        print(f"{model}: SIGTERM sent to verified pid={pid} port={port}")
PY
