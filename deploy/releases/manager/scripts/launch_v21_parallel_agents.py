#!/usr/bin/env python3
"""Launch V21 parallel runner agents in one tmux session."""
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-manifest", type=Path, required=True)
    p.add_argument("--parallelism", type=int, default=64)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--tmux-session", default="ego_annotation")
    p.add_argument("--window-prefix", default="v21p")
    p.add_argument("--provider", default="occ")
    p.add_argument("--model", default="gpt-5.5:xhigh")
    p.add_argument("--tools", default="read,bash,edit,write")
    p.add_argument("--system-prompt", default="configs/v21_agent_system_prompt.md")
    p.add_argument("--runner-prompt", default=".pi/prompts/v21_parallel_runner.md")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def tmux_has_session(session: str) -> bool:
    proc = subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return proc.returncode == 0


def main() -> int:
    args = parse_args()
    if args.parallelism <= 0:
        raise SystemExit("--parallelism must be positive")
    repo_root = args.repo_root.expanduser().resolve()
    manifest = args.batch_manifest.expanduser()
    if not manifest.is_absolute():
        manifest = (repo_root / manifest).resolve()
    commands: list[tuple[str, str]] = []
    for idx in range(args.parallelism):
        runner_id = f"runner_{idx:03d}"
        prompt = f"/v21_parallel_runner {manifest} {runner_id}"
        pi_cmd = [
            "pi",
            "--provider",
            shlex.quote(args.provider),
            "--model",
            shlex.quote(args.model),
            "--system-prompt",
            f"\"$(cat {shlex.quote(args.system_prompt)})\"",
            "--tools",
            shlex.quote(args.tools),
            "--prompt-template",
            shlex.quote(args.runner_prompt),
            shlex.quote(prompt),
        ]
        command = f"cd {shlex.quote(str(repo_root))} && " + " ".join(pi_cmd)
        commands.append((runner_id, command))

    if args.dry_run:
        for runner_id, command in commands:
            print(f"[{runner_id}] {command}")
        return 0

    if not tmux_has_session(args.tmux_session):
        subprocess.run(["tmux", "new-session", "-d", "-s", args.tmux_session, "bash"], check=True)
    for runner_id, command in commands:
        window_name = f"{args.window_prefix}-{runner_id.split('_')[-1]}"
        subprocess.run(["tmux", "new-window", "-t", args.tmux_session, "-n", window_name, command], check=True)
    print(f"launched {len(commands)} V21 parallel runner agents in tmux session {args.tmux_session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
