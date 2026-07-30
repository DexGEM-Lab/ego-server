"""One idempotent server-side command to start all five Ego Ray Serve groups.

Runs on dex-a800 from the curated repo. It brings up the committed per-GPU Serve
clusters (GPU0 UniDepth, GPU1 hands/WiLoR, GPU2 DROID, GPU3 HaWoR+infiller, GPU6
Cosmos3), one Ray head + Serve app per physical GPU group, each in its own window
of the single ``ego_annotation`` tmux session with a durable log.

Design constraints (mirrors ``AGENTS.md`` and ``lifecycle.py``):

* The exact ``ray start``/``serve run`` commands come from ``lifecycle.py`` so the
  physical CUDA pinning, native ``--num-gpus=1`` GPU resource, disjoint
  component/worker ports, explicit GCS/dashboard/HTTP addresses, and the absence of
  any ambient ``RAY_ADDRESS`` are the single source of truth. This module never
  restates a port from memory.
* Idempotent: it probes each group's canonical health lane (``28000 + gpu_id`` at
  ``/health``) exactly once (never polls, never sleeps) and skips healthy groups. A
  down group with a stale/dead tmux window has that window replaced; a down group
  with no window gets a fresh one.
* GPU6 Cosmos3 is never re-implemented here: when it is down the guarded launcher
  ``scripts/cosmos3_guarded_cutover.sh`` (which owns the scoped bare-8001 stop, the
  GPU6 head + Serve deploy, acceptance, and the resident driver) is invoked.
* It never installs, provisions, runs ``ray stop`` globally, or touches GPU4/5/7.
  Failures are surfaced (non-zero exit); there is no silent fallback.

Invoke as::

    python -m scripts.start_ego_model_services            # start missing groups
    python -m scripts.start_ego_model_services --status    # report health only
    python -m scripts.start_ego_model_services --dry-run   # print commands only
    python -m scripts.start_ego_model_services --groups gpu0,gpu1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import shlex
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from ego_annotation.serving.lifecycle import (
    COMMITTED_GPU_GROUPS,
    GpuServiceGroup,
)

# Single project tmux session (AGENTS.md: one session, one window per group).
TMUX_SESSION = "ego_annotation"
# Durable per-window logs live outside any Ray temp dir so they survive teardown.
WINDOW_LOG_DIR = "/tmp/ego-serve-windows"
HEALTH_PATH = "/health"
# GPU6 Cosmos3 is brought up only through this guarded launcher.
COSMOS3_GPU_ID = 6
COSMOS3_LAUNCHER = Path(__file__).resolve().parent / "cosmos3_guarded_cutover.sh"
# The Cosmos3 launcher requires a fresh run root strictly under this benchmark root.
COSMOS3_BENCHMARK_ROOT = "/vePFS-Mindverse/user/yiwen/user-home/zjh/ray_serve_benchmarks"

# Non-Cosmos groups deploy through one short detached driver. It explicitly starts
# both GPU3 applications; a single ``serve run`` import cannot do that on Ray 2.55.1.
GROUP_DRIVER_MODULE = "scripts.serve_group_driver"


def serve_host() -> str:
    """Canonical host for the Serve HTTP lanes (matches the router default)."""
    return os.environ.get("EGO_SERVE_HOST", "127.0.0.1")


def window_name(gpu_id: int) -> str:
    """Stable per-group window name so reuse/replacement is deterministic."""
    return f"ego-serve-gpu{gpu_id}"


def log_path(gpu_id: int) -> str:
    return f"{WINDOW_LOG_DIR}/gpu{gpu_id}.log"


def lane_port(group: GpuServiceGroup) -> int:
    """Canonical Serve HTTP lane port for a group, resolved from lifecycle."""
    return group.lifecycle.ports.serve_http_port


def serve_launch_script(group: GpuServiceGroup) -> str:
    """Bash script run inside the group's tmux window.

    Chains the exact lifecycle ``ray start`` (returns once the head is up) and the
    detached Serve deployment driver. Both carry ``env -u RAY_ADDRESS`` so no
    ambient address can bind another cluster. Output is appended to a durable log.
    The final ``tail`` keeps an inspectable tmux window while Ray/Serve persist
    independently of that window.
    """
    lc = group.lifecycle
    start_cmd = f"env -u RAY_ADDRESS {lc.startup_command()}"
    serve_cmd = (
        f"env -u RAY_ADDRESS {lc.interpreter} -m {GROUP_DRIVER_MODULE} "
        f"--gpu-id {group.gpu_id} --address {lc.gcs_address} "
        f"--port {lc.ports.serve_http_port}"
    )
    header = (
        "printf '=== %s start gpu"
        f"{group.gpu_id} ({group.physical_group}) ===\\n' "
        '"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'
    )
    return "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {WINDOW_LOG_DIR}",
            f"exec >> {log_path(group.gpu_id)} 2>&1",
            header,
            start_cmd,
            serve_cmd,
            f"exec tail -n +1 -f {log_path(group.gpu_id)}",
        ]
    )


def serve_window_argv(session: str, group: GpuServiceGroup) -> list[str]:
    """tmux argv that creates the detached group window running the launch script."""
    return [
        "tmux",
        "new-window",
        "-d",
        "-t",
        session,
        "-n",
        window_name(group.gpu_id),
        "bash",
        "-lc",
        serve_launch_script(group),
    ]


def remain_on_exit_argv(session: str, gpu_id: int) -> list[str]:
    """Keep a failed window open so the operator can read its final state."""
    return [
        "tmux",
        "set-option",
        "-t",
        f"{session}:{window_name(gpu_id)}",
        "-w",
        "remain-on-exit",
        "on",
    ]


def kill_window_argv(session: str, gpu_id: int) -> list[str]:
    return ["tmux", "kill-window", "-t", f"{session}:{window_name(gpu_id)}"]


def default_cosmos_run_root() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{COSMOS3_BENCHMARK_ROOT}/ego_start_{stamp}"


def cosmos_launcher_argv(run_root: str, launcher: Path = COSMOS3_LAUNCHER) -> list[str]:
    """argv for the guarded Cosmos3 service start (acceptance checks, no load sweep)."""
    return ["bash", str(launcher), "--run-root", run_root, "--skip-benchmark"]


# --- Planning (pure, GPU/tmux-free) -----------------------------------------

# Action outcomes for one group, decided before any side effect runs.
SKIP_HEALTHY = "skip-healthy"
START = "start"
REPLACE = "replace"  # down group whose stale/dead window is replaced
STATUS_ONLY = "status"


@dataclass(frozen=True)
class GroupPlan:
    gpu_id: int
    physical_group: str
    lane_port: int
    healthy: bool
    action: str
    # argv lists (and, for serve, the window script) that WOULD run / DID run.
    commands: tuple[list[str], ...] = field(default_factory=tuple)
    launch_script: str | None = None


def plan_group(
    group: GpuServiceGroup,
    *,
    healthy: bool,
    window_exists: bool,
    status_only: bool,
    session: str,
    cosmos_run_root: str,
) -> GroupPlan:
    """Decide the action and the exact commands for one group. No side effects."""
    lp = lane_port(group)
    base = dict(gpu_id=group.gpu_id, physical_group=group.physical_group, lane_port=lp, healthy=healthy)

    if status_only:
        return GroupPlan(**base, action=STATUS_ONLY)
    if healthy:
        return GroupPlan(**base, action=SKIP_HEALTHY)

    if group.gpu_id == COSMOS3_GPU_ID:
        # The guarded launcher owns its own (timestamped) window and lifecycle.
        return GroupPlan(**base, action=START, commands=(cosmos_launcher_argv(cosmos_run_root),))

    commands: list[list[str]] = []
    action = START
    if window_exists:
        action = REPLACE
        commands.append(kill_window_argv(session, group.gpu_id))
    commands.append(serve_window_argv(session, group))
    commands.append(remain_on_exit_argv(session, group.gpu_id))
    return GroupPlan(**base, action=action, commands=tuple(commands), launch_script=serve_launch_script(group))


# --- Environment probes (injectable for tests) ------------------------------


def probe_health(host: str, port: int, *, timeout_s: float = 2.0) -> bool:
    """Single ``GET /health`` probe. Live iff reachable with status < 500.

    Never polls, never sleeps. Any connection error / timeout means "not live",
    which is the signal to (re)start the group, not a failure to surface.
    """
    url = f"http://{host}:{port}{HEALTH_PATH}"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 (loopback only)
            return resp.status < 500
    except urllib.error.HTTPError as exc:
        return exc.code < 500
    except Exception:
        return False


def tmux_window_names(session: str, *, runner: Callable[[list[str]], subprocess.CompletedProcess]) -> set[str]:
    """Current window names in the session (empty set if the session is absent)."""
    proc = runner(["tmux", "list-windows", "-t", session, "-F", "#{window_name}"])
    if proc.returncode != 0:
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _default_runner(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True)


# --- Orchestration ----------------------------------------------------------


def select_groups(tokens: Sequence[str] | None) -> tuple[GpuServiceGroup, ...]:
    """Resolve ``--groups gpu0,gpu2`` (or ``0,2``) to committed groups, in order."""
    if not tokens:
        return tuple(COMMITTED_GPU_GROUPS)
    by_id = {g.gpu_id: g for g in COMMITTED_GPU_GROUPS}
    wanted: list[int] = []
    for raw in tokens:
        tok = raw.strip().lower()
        if tok.startswith("gpu"):
            tok = tok[3:]
        if not tok.isdigit() or int(tok) not in by_id:
            valid = ",".join(f"gpu{g.gpu_id}" for g in COMMITTED_GPU_GROUPS)
            raise SystemExit(f"error: unknown group {raw!r}; valid groups: {valid}")
        gid = int(tok)
        if gid not in wanted:
            wanted.append(gid)
    return tuple(by_id[g] for g in wanted)


def with_gpu1_wire_format(group: GpuServiceGroup, wire_format: str) -> GpuServiceGroup:
    """Return an explicit GPU1 dual-API transport treatment with attestation."""
    if group.gpu_id != 1:
        return group
    if wire_format not in {"multipart", "envelope"}:
        raise ValueError("GPU1 wire format must be multipart or envelope")
    env = dict(group.lifecycle.env_vars)
    env.update({
        "EGO_HANDS_EXPERIMENT_WIRE_FORMAT": wire_format,
        "EGO_WILOR_EXPERIMENT_WIRE_FORMAT": wire_format,
        "EGO_HANDS_EXPERIMENT_TELEMETRY": "1",
        "EGO_WILOR_EXPERIMENT_TELEMETRY": "1",
    })
    lifecycle = replace(group.lifecycle, env_vars=tuple(env.items()))
    return replace(group, lifecycle=lifecycle)


def with_gpu3_wire_format(group: GpuServiceGroup, wire_format: str) -> GpuServiceGroup:
    """Return a GPU3 launch treatment with endpoint-attested wire configuration.

    The normal production group remains multipart.  This pure plan helper is used
    only when an operator explicitly requests a GPU3 restart in the envelope
    treatment; both colocated logical APIs get the same wire value and telemetry.
    """
    if group.gpu_id != 3:
        return group
    if wire_format not in {"multipart", "envelope"}:
        raise ValueError("GPU3 wire format must be multipart or envelope")
    env = dict(group.lifecycle.env_vars)
    env.update({
        "EGO_HAWOR_EXPERIMENT_WIRE_FORMAT": wire_format,
        "EGO_HAWOR_INFILLER_EXPERIMENT_WIRE_FORMAT": wire_format,
        "EGO_HAWOR_EXPERIMENT_TELEMETRY": "1",
        "EGO_HAWOR_INFILLER_EXPERIMENT_TELEMETRY": "1",
    })
    lifecycle = replace(group.lifecycle, env_vars=tuple(env.items()))
    return replace(group, lifecycle=lifecycle)


def _format_table(plans: Sequence[GroupPlan]) -> str:
    rows = [("GPU", "GROUP", "LANE", "HEALTH", "ACTION")]
    action_label = {
        SKIP_HEALTHY: "skipped (healthy)",
        START: "started",
        REPLACE: "replaced (stale window)",
        STATUS_ONLY: "-",
    }
    for p in plans:
        rows.append(
            (
                str(p.gpu_id),
                p.physical_group,
                str(p.lane_port),
                "healthy" if p.healthy else "down",
                action_label[p.action],
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)


def run(
    groups: Sequence[GpuServiceGroup],
    *,
    status_only: bool,
    dry_run: bool,
    session: str = TMUX_SESSION,
    cosmos_run_root: str | None = None,
    health_probe: Callable[[str, int], bool] | None = None,
    tmux_runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
    host: str | None = None,
    out=sys.stdout,
) -> list[GroupPlan]:
    """Probe, plan, and (unless status/dry-run) execute. Returns the per-group plans."""
    probe = health_probe or (lambda h, p: probe_health(h, p))
    runner = tmux_runner or _default_runner
    resolved_host = host or serve_host()
    run_root = cosmos_run_root or default_cosmos_run_root()

    existing = set() if status_only else tmux_window_names(session, runner=runner)

    plans: list[GroupPlan] = []
    for group in groups:
        healthy = probe(resolved_host, lane_port(group))
        plan = plan_group(
            group,
            healthy=healthy,
            window_exists=window_name(group.gpu_id) in existing,
            status_only=status_only,
            session=session,
            cosmos_run_root=run_root,
        )
        plans.append(plan)

    if not (status_only or dry_run):
        if any(plan.commands for plan in plans):
            has_session = runner(["tmux", "has-session", "-t", session]).returncode == 0
            if not has_session:
                create = runner([
                    "setsid", "tmux", "new-session", "-d", "-s", session,
                    "-n", "control", "bash", "-lc", "exec tail -f /dev/null",
                ])
                if create.returncode != 0:
                    raise SystemExit(
                        f"error: could not create persistent tmux session {session}: "
                        f"{(create.stderr or create.stdout or '').strip()}"
                    )
        for plan in plans:
            for argv in plan.commands:
                proc = runner(argv)
                if proc.returncode != 0:
                    # Surface the failure explicitly; do not fall back silently.
                    raise SystemExit(
                        f"error: gpu{plan.gpu_id} command failed ({' '.join(argv)}): "
                        f"{(proc.stderr or proc.stdout or '').strip()}"
                    )

    if dry_run:
        for plan in plans:
            if plan.action in (SKIP_HEALTHY, STATUS_ONLY):
                continue
            print(f"# gpu{plan.gpu_id} ({plan.physical_group}) -> {plan.action}", file=out)
            if plan.launch_script is not None:
                print(f"#   window script for {window_name(plan.gpu_id)}:", file=out)
                for line in plan.launch_script.splitlines():
                    print(f"#     {line}", file=out)
            for argv in plan.commands:
                print("  " + " ".join(shlex.quote(a) for a in argv), file=out)

    print(_format_table(plans), file=out)
    return plans


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.start_ego_model_services",
        description="Idempotently start the five committed Ego Ray Serve GPU groups on dex-a800.",
    )
    parser.add_argument("--status", action="store_true", help="Report per-group health only; make no changes.")
    parser.add_argument("--dry-run", action="store_true", help="Print the exact commands that would run; make no changes.")
    parser.add_argument(
        "--groups",
        default=None,
        help="Comma-separated subset, e.g. gpu0,gpu1 (or 0,1). Default: all five groups.",
    )
    parser.add_argument(
        "--gpu1-wire-format", choices=("multipart", "envelope"), default="multipart",
        help="GPU1 Hands/WiLoR restart treatment recorded in runtime-config diagnostics; multipart is the production default.",
    )
    parser.add_argument(
        "--gpu3-wire-format", choices=("multipart", "envelope"), default="multipart",
        help="GPU3 HaWoR/infiller restart treatment recorded in runtime-config diagnostics; multipart is the production default.",
    )
    parser.add_argument(
        "--cosmos-run-root",
        default=os.environ.get("EGO_RAY_SERVE_BENCHMARK_ROOT"),
        help="Fresh run root under the benchmark root for the guarded Cosmos3 launcher.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokens = args.groups.split(",") if args.groups else None
    groups = select_groups(tokens)
    if args.gpu1_wire_format == "envelope":
        groups = tuple(with_gpu1_wire_format(group, args.gpu1_wire_format) for group in groups)
    if args.gpu3_wire_format == "envelope":
        groups = tuple(with_gpu3_wire_format(group, args.gpu3_wire_format) for group in groups)
    run(
        groups,
        status_only=args.status,
        dry_run=args.dry_run,
        cosmos_run_root=args.cosmos_run_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
