#!/usr/bin/env bash
set -euo pipefail

if [[ "${V22_ARCHIVED_DEPLOYMENT_OPT_IN:-}" != "1" ]]; then
  cat >&2 <<'EOF'
Refusing to start withdrawn V22 resident services.
This deployment is archive-only and disabled. The authoritative runtime is Feishu Ray Serve.
To explicitly acknowledge the withdrawn custom runtime, set V22_ARCHIVED_DEPLOYMENT_OPT_IN=1.
EOF
  exit 78
fi

REPO="${REPO:-/home/zjh/ego-annotation-feat-parallel}"
ROOT="${SERVICE_ROOT:-/home/zjh/data/v22_resident_services}"
CONFIG="${CONFIG:-$REPO/configs/v22_resident_services.json}"

python3 - "$CONFIG" "$REPO" "$ROOT" <<'PY'
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
repo = Path(sys.argv[2])
root = Path(sys.argv[3])
services = config["services"]

if (
    config.get("archive_only") is not True
    or config.get("deployment_state") != "withdrawn"
    or config.get("enabled") is not False
):
    print(
        "v22 archive start refused: config must remain archive_only=true, "
        "deployment_state=withdrawn, enabled=false",
        file=sys.stderr,
    )
    raise SystemExit(2)


def refuse(message):
    print(f"v22 archive start refused: {message}", file=sys.stderr)
    raise SystemExit(2)


def check_ports():
    ss = shutil.which("ss")
    if not ss:
        refuse("cannot inspect listener ports: ss is unavailable")
    result = subprocess.run(
        [ss, "-H", "-ltnp"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        refuse(f"cannot inspect listener ports: {result.stderr.strip() or result.returncode}")
    occupied = []
    ports = {int(spec["port"]): model for model, spec in services.items()}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        local_address = fields[3]
        try:
            port = int(local_address.rsplit(":", 1)[1].rstrip("]"))
        except (IndexError, ValueError):
            continue
        if port in ports:
            occupied.append(f"{ports[port]} port {port}: {line}")
    if occupied:
        refuse("configured port(s) already have listeners; no process will be stopped:\n" + "\n".join(occupied))


def check_visible_gpu_occupancy():
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        refuse("cannot inspect visible GPU occupancy: nvidia-smi is unavailable")
    gpu_result = subprocess.run(
        [nvidia_smi, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if gpu_result.returncode != 0:
        refuse(f"cannot inspect GPU inventory: {gpu_result.stderr.strip() or gpu_result.returncode}")
    uuid_by_index = {}
    for line in gpu_result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2 and parts[0].isdigit():
            uuid_by_index[int(parts[0])] = parts[1]
    requested = {int(spec["gpu"]): model for model, spec in services.items()}
    missing = sorted(set(requested) - set(uuid_by_index))
    if missing:
        refuse(f"configured GPU index(es) are not visible: {missing}")
    process_result = subprocess.run(
        [
            nvidia_smi,
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process_result.returncode != 0:
        refuse(f"cannot inspect GPU processes: {process_result.stderr.strip() or process_result.returncode}")
    selected_uuids = {uuid_by_index[index]: (index, model) for index, model in requested.items()}
    occupied = []
    for line in process_result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if parts and parts[0] in selected_uuids:
            index, model = selected_uuids[parts[0]]
            occupied.append(f"{model} GPU {index}: {line}")
    if occupied:
        refuse("configured GPU(s) have visible compute processes; no process will be stopped:\n" + "\n".join(occupied))


# Archive opt-in never overrides physical conflict checks.
check_ports()
check_visible_gpu_occupancy()

(root / "logs").mkdir(parents=True, exist_ok=True)
(root / "pids").mkdir(parents=True, exist_ok=True)
(root / "artifacts").mkdir(parents=True, exist_ok=True)
wilor_mano = repo / "third_party" / "algorithms" / "wilor" / "mano_data"
wilor_mano.mkdir(parents=True, exist_ok=True)
for name in ("MANO_RIGHT.pkl", "MANO_LEFT.pkl"):
    target = Path("/home/zjh/ego-annation-checkpoints/mano") / name
    link = wilor_mano / name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)

for model, spec in services.items():
    pid_path = root / "pids" / f"{model}.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            pid_path.unlink(missing_ok=True)
        else:
            refuse(f"pid file {pid_path} points to a live process {pid}; inspect it instead of reusing it")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
    env["PYTHONUNBUFFERED"] = "1"
    env["V22_WILOR_CHECKPOINT_ROOT"] = "/home/zjh/ego-annation-checkpoints/wilor"
    env["V22_MANO_RIGHT"] = "/home/zjh/ego-annation-checkpoints/mano/MANO_RIGHT.pkl"
    cmd = [
        spec["python"],
        str(repo / "scripts/serve_v22_resident_model.py"),
        "--model", model,
        "--host", config["host"],
        "--port", str(spec["port"]),
        "--artifact-root", str(root / "artifacts" / "shared"),
        "--wait-s", str(config["wait_window_s"]),
        "--pending-limit", str(config["pending_request_limit"]),
        "--native-batch-cap", str(spec["native_batch_cap"]),
    ]
    log = (root / "logs" / f"{model}.log").open("ab")
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_path.write_text(str(proc.pid))
    print(f"{model}: started pid={proc.pid} gpu={spec['gpu']} port={spec['port']}")
PY
