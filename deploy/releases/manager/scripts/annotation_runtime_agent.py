#!/usr/bin/env python3
"""Create and launch a single-video Pi annotation runtime agent."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RuntimeAgentError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeAgentError(f"expected JSON object: {path}")
    return payload


def runtime_system_prompt() -> str:
    return """You are the single-video Ego annotation runtime executor.

You receive exactly one runtime request JSON. Your job is to execute that request and produce the result JSON. Do not inspect task memory, project planning docs, AGENTS.md, GT files, evaluator targets, or unrelated repository history. Do not create commits, push code, install packages, repair environments, or launch subagents. If the requested command fails, preserve the failure in the result/log and stop.

The run is prediction-side only. You may read the request JSON, run the executor command named in the prompt, and read the result JSON. Heavy model work must run only through the configured remote/A800 or GPU wrapper path already encoded in the request. D9b semantic review, if requested, is performed only by the executor's fixed semantic-agent stage; do not add free-form judgments for D2, D4, D7, D8, D10, or D11.
"""


def runtime_prompt(request_path: Path, result_path: Path, repo_root: Path) -> str:
    return f"""Execute this one annotation runtime request.

Request JSON:
{request_path}

Required command:
{sys.executable} {repo_root / 'scripts' / 'execute_annotation_runtime_request.py'} --request {request_path} --result {result_path}

Use the bash tool once for that command and set the bash tool timeout to 7200 seconds. After the command exits, read {result_path} and answer with only the JSON status summary. If the command fails, report the command failure and the log/result path. Do not manually perform annotation, setup, git, package installation, or repository edits outside the executor command.
"""


def remote_payload(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "host": str(config.host),
        "repo_root": str(config.repo_root),
        "output_root": str(config.output_root),
        "upload_root": str(config.upload_root),
        "package_root": str(config.package_root),
        "python": str(config.python),
    }


def build_runtime_request(
    *,
    repo_root: Path,
    job_id: str,
    video_uri: str,
    run_root: Path,
    package_root: Path,
    local_video: bool,
    remote_config: Any,
    timeout_s: int,
    metadata: dict[str, Any] | None,
    pipeline_flags: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "ego.annotation.runtime_request.v1",
        "created_at": utc_now(),
        "job_id": job_id,
        "case_id": job_id,
        "video_uri": video_uri,
        "video_metadata": metadata or {},
        "repo_root": str(repo_root),
        "run_root": str(run_root),
        "package_root": str(package_root),
        "execution_backend": "remote_ssh_script" if remote_config is not None else "local_script",
        "remote": remote_payload(remote_config),
        "local_video": bool(local_video),
        "timeout_s": int(timeout_s),
        "pipeline_flags": pipeline_flags,
        "model_requests": {
            "directory": str(run_root / "requests"),
            "expected_files": ["unidepth.json", "wilor.json", "droid.json", "hawor.json"],
        },
        "result_path": str(run_root / "logs" / "runtime_agent_result.json"),
        "claim_scope": "Single uploaded video prediction request. No batch, no parallel annotation, no GT sidecars in runtime agent context.",
    }


def build_runtime_bundle(
    *,
    repo_root: Path,
    run_root: Path,
    request_payload: dict[str, Any],
) -> dict[str, Path]:
    bundle_dir = run_root / "runtime_agent_bundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    system_prompt_path = bundle_dir / "runtime_system_prompt.md"
    prompt_path = bundle_dir / "runtime_prompt.md"
    request_path = bundle_dir / "runtime_request.json"
    result_path = Path(str(request_payload["result_path"]))
    system_prompt_path.write_text(runtime_system_prompt(), encoding="utf-8")
    prompt_path.write_text(runtime_prompt(request_path, result_path, repo_root), encoding="utf-8")
    write_json(request_path, request_payload)
    return {
        "bundle_dir": bundle_dir,
        "system_prompt": system_prompt_path,
        "prompt": prompt_path,
        "request": request_path,
        "result": result_path,
    }


def build_pi_command(*, paths: dict[str, Path], pi_binary: str, model: str | None, provider: str | None) -> list[str]:
    cmd = [
        pi_binary,
        "-p",
        "--no-context-files",
        "--no-skills",
        "--no-extensions",
        "--session-dir",
        str(paths["bundle_dir"] / "pi_session"),
        "--tools",
        "bash,read",
        "--system-prompt",
        paths["system_prompt"].read_text(encoding="utf-8"),
    ]
    if provider:
        cmd.extend(["--provider", provider])
    if model:
        cmd.extend(["--model", model])
    cmd.append(paths["prompt"].read_text(encoding="utf-8"))
    return cmd


def launch_runtime_agent(
    *,
    repo_root: Path,
    run_root: Path,
    request_payload: dict[str, Any],
    pi_binary: str | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    paths = build_runtime_bundle(repo_root=repo_root, run_root=run_root, request_payload=request_payload)
    resolved_pi = pi_binary or os.environ.get("ANNOTATION_PI_BINARY", "pi")
    cmd = build_pi_command(paths=paths, pi_binary=resolved_pi, model=model or os.environ.get("ANNOTATION_PI_MODEL"), provider=provider or os.environ.get("ANNOTATION_PI_PROVIDER"))
    log_path = run_root / "logs" / "pi_annotation_agent.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(paths["bundle_dir"]), text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed_s = (datetime.now(timezone.utc) - started).total_seconds()
    status = {
        "schema": "ego.annotation.runtime_agent_status.v1",
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": int(proc.returncode),
        "elapsed_s": elapsed_s,
        "bundle_dir": str(paths["bundle_dir"]),
        "request_path": str(paths["request"]),
        "result_path": str(paths["result"]),
        "log": str(log_path),
        "command": cmd[:1] + ["<pi args redacted; see runtime bundle>"],
    }
    write_json(run_root / "logs" / "pi_annotation_agent_status.json", status)
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeAgentError(f"pi_runtime_agent_failed rc={proc.returncode}; log={log_path}\n{tail}")
    if not paths["result"].exists():
        raise RuntimeAgentError(f"pi_runtime_agent_missing_result: {paths['result']}; log={log_path}")
    result = load_json(paths["result"])
    return {"agent_status": status, "result": result}
