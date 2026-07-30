"""Guarded Cosmos3 GPU6 candidate cutover.

This module has no Ray/vLLM import so its preflight can run with the verified
standalone interpreter before GPU6 is touched.  It consumes the standalone reports
in place; reports are deliberately never copied into the repository.

``--execute`` is intentionally the only operation which can start the candidate.
It requires both an explicit acknowledgement and evidence that the bare 8001
listener is gone.  It never stops the bare service, changes port 8001, or invokes
unscoped ``ray stop``. If candidate startup or deployment fails, the runbook gives
an operator-only rollback sequence scoped to its unique Ray temp directory.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
import signal
import socket
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from ego_annotation.serving.lifecycle import (
    COSMOS3_BARE_BASELINE_URL,
    RAY_VERSION,
    STANDALONE_ARTIFACTS_DIR,
    cosmos3_lifecycle,
)

VERIFY_REPORT_NAME = "verify_report.json"
FINALIZE_REPORT_NAME = "finalize_20260717T000000Z.json"
BARE_COSMOS3_RESTORE_COMMAND = (
    "su - ylang -c 'bash /home/zjh/cosmos3_ray_serve/RESTORE_BARE_COSMOS3.sh'"
)


class CutoverGateError(RuntimeError):
    """The standalone evidence or an operational precondition is not sufficient."""


@dataclass(frozen=True)
class StandaloneEvidence:
    """Validated facts from the two in-place standalone reports."""

    artifacts_dir: Path
    interpreter: str
    verify_report: Path
    finalize_report: Path


@dataclass(frozen=True)
class CutoverCommands:
    """Exact candidate lifecycle commands, all pinned to explicit addresses."""

    start_cluster: str
    deploy: str
    status: str
    scoped_rollback: tuple[str, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CutoverGateError(message)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def _read_json_object(path: Path, *, allow_preamble: bool) -> Mapping[str, Any]:
    """Read a JSON object, accepting only a textual warning preamble when requested.

    The verified report currently starts with a transformers deprecation warning
    before its JSON object.  Parsing from the first ``{`` preserves the structured
    result without pretending that the whole file is strict JSON.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CutoverGateError(f"cannot read {path}: {exc}") from exc
    start = text.find("{") if allow_preamble else 0
    _require(start >= 0, f"{path} contains no JSON object")
    if not allow_preamble:
        _require(not text[:start].strip(), f"{path} has an unexpected preamble")
    try:
        parsed = json.loads(text[start:])
    except json.JSONDecodeError as exc:
        raise CutoverGateError(f"invalid JSON in {path}: {exc}") from exc
    return _mapping(parsed, str(path))


def _require_ok_check(checks: Mapping[str, Any], name: str, **expected: Any) -> None:
    check = _mapping(checks.get(name), f"checks.{name}")
    _require(check.get("ok") is True, f"checks.{name}.ok must be true")
    for key, value in expected.items():
        _require(check.get(key) == value, f"checks.{name}.{key} must be {value!r}")


def validate_standalone_artifacts(artifacts_dir: str | Path = STANDALONE_ARTIFACTS_DIR) -> StandaloneEvidence:
    """Validate the verified standalone environment reports in place.

    This is intentionally a content gate rather than a sentinel/file-existence gate:
    it binds the interpreter, no-overlay property, Cosmos3 registration, and Ray/vLLM
    ABI versions that the candidate will use.
    """
    root = Path(artifacts_dir)
    verify_path = root / "logs" / VERIFY_REPORT_NAME
    finalize_path = root / "logs" / FINALIZE_REPORT_NAME
    verify = _read_json_object(verify_path, allow_preamble=True)
    finalize = _read_json_object(finalize_path, allow_preamble=False)
    expected_venv = str(root / ".venv")
    expected_interpreter = str(root / ".venv" / "bin" / "python")

    _require(verify.get("ok") is True, "verify report is not ok")
    _require(verify.get("venv") == expected_venv, "verify report venv does not match standalone artifacts")
    _require(verify.get("errors") == [], "verify report contains errors")
    verify_versions = _mapping(verify.get("versions"), "verify.versions")
    _require(verify_versions.get("ray") == RAY_VERSION, "verify report Ray version does not match lifecycle")
    _require(verify_versions.get("vllm") == "0.19.1", "verify report vLLM version must be 0.19.1")
    _require(verify_versions.get("torch") == "2.10.0", "verify report Torch version must be 2.10.0")
    checks = _mapping(verify.get("checks"), "verify.checks")
    _require_ok_check(checks, "import:ray")
    _require_ok_check(checks, "import:ray.serve")
    _require_ok_check(checks, "import:vllm")
    _require_ok_check(checks, "import:torch")
    _require_ok_check(checks, "import:transformers_cosmos3")
    _require_ok_check(checks, "import:vllm_cosmos3")
    _require_ok_check(checks, "no_pth_overlay_into_ylang", offending=[])
    _require_ok_check(checks, "ModelRegistry.Cosmos3ReasonerForConditionalGeneration")
    _require_ok_check(checks, "AsyncEngineArgs.construct_no_weights", model="nvidia/Cosmos3-Nano")
    cuda = _mapping(checks.get("torch.cuda"), "checks.torch.cuda")
    _require(cuda.get("available") is True, "standalone report does not show CUDA available")

    _require(finalize.get("report_type") == "standalone_finalize", "finalize report type is invalid")
    _require(finalize.get("host") == "dex-a800", "finalize report is not for dex-a800")
    interpreter = _mapping(finalize.get("interpreter"), "finalize.interpreter")
    _require(interpreter.get("path") == expected_interpreter, "finalize interpreter is not the standalone interpreter")
    _require(interpreter.get("venv") == expected_venv, "finalize venv does not match standalone artifacts")
    _require(interpreter.get("version") == verify.get("python"), "verify/finalize Python versions disagree")
    final_versions = _mapping(finalize.get("versions"), "finalize.versions")
    for package in ("ray", "vllm", "torch"):
        _require(final_versions.get(package) == verify_versions.get(package), f"verify/finalize {package} versions disagree")
    production = _mapping(finalize.get("production_port_8001"), "finalize.production_port_8001")
    _require(production.get("status") == "unchanged", "finalize report does not preserve bare port 8001")
    _require(production.get("model_endpoint") == f"{COSMOS3_BARE_BASELINE_URL}/v1/models", "unexpected bare model endpoint")
    _require("nvidia/Cosmos3-Nano" in production.get("models", []), "bare model identity is missing")
    overlay = _mapping(finalize.get("pth_overlay_guard"), "finalize.pth_overlay_guard")
    _require(overlay.get("status") == "absent" and overlay.get("offending_files") == [], "standalone environment has a ylang .pth overlay")
    plugins = _mapping(finalize.get("cosmos_plugins"), "finalize.cosmos_plugins")
    _require(_mapping(plugins.get("transformers_cosmos3"), "plugins.transformers_cosmos3").get("import_ok") is True,
             "transformers Cosmos3 plugin did not import")
    _require(_mapping(plugins.get("vllm_cosmos3"), "plugins.vllm_cosmos3").get("class_accessible") is True,
             "vLLM Cosmos3 model class is unavailable")
    _require(_mapping(plugins.get("ModelRegistry"), "plugins.ModelRegistry").get("registration_confirmed") is True,
             "Cosmos3 architecture is not registered")
    _require(_mapping(plugins.get("AsyncEngineArgs"), "plugins.AsyncEngineArgs").get("construct_ok") is True,
             "Cosmos3 engine arguments are not valid")
    verification = _mapping(finalize.get("verification"), "finalize.verification")
    _require(verification.get("imports_all_pass") is True and verification.get("errors") == [],
             "finalize import verification did not pass")
    _require(verification.get("report") == str(verify_path), "finalize report does not reference this verify report")
    diagnostic = _mapping(finalize.get("cpu_diagnostic_cluster"), "finalize.cpu_diagnostic_cluster")
    _require(diagnostic.get("ran") is True and diagnostic.get("gpu_advertised") is False,
             "CPU diagnostic did not prove a CPU-only Ray cluster")
    _require(_mapping(diagnostic.get("port_disjoint_from_gpu6"), "diagnostic.port_disjoint_from_gpu6").get("no_overlap") is True,
             "diagnostic ports overlap GPU6 ports")
    _require(diagnostic.get("residual_processes") == [], "CPU diagnostic left residual Ray processes")

    lifecycle = cosmos3_lifecycle()
    _require(lifecycle.interpreter == expected_interpreter, "lifecycle interpreter diverges from verified standalone interpreter")
    return StandaloneEvidence(root, expected_interpreter, verify_path, finalize_path)


def _port_is_listening(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _candidate_process_pids(temp_dir: str, *, proc_root: str | Path = "/proc") -> list[int]:
    """Return only processes whose command line names the candidate temp directory."""
    pids: list[int] = []
    root = Path(proc_root)
    caller_pid = os.getpid()
    for entry in root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == caller_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if temp_dir in command:
            pids.append(int(entry.name))
    return sorted(pids)


def stop_scoped_candidate(temp_dir: str, *, pid_lookup: Callable[[str], Sequence[int]] = _candidate_process_pids,
                          kill: Callable[[int, int], None] = os.kill) -> tuple[int, ...]:
    """Stop only PIDs that explicitly name the fixed Cosmos3 candidate directory."""
    _require(temp_dir == cosmos3_lifecycle().temp_dir,
             f"scoped rollback only permits {cosmos3_lifecycle().temp_dir}")
    pids = tuple(pid_lookup(temp_dir))
    if not pids:
        return ()
    for pid in pids:
        kill(pid, signal.SIGTERM)
    survivors = set(pid_lookup(temp_dir)) & set(pids)
    for pid in sorted(survivors):
        kill(pid, signal.SIGKILL)
    return pids


def scoped_rollback_commands() -> tuple[str, ...]:
    """Return a rollback plan that cannot use unscoped ``ray stop``."""
    lifecycle = cosmos3_lifecycle()
    return (
        lifecycle.serve_shutdown_command(),
        f"{lifecycle.interpreter} -m ego_annotation.serving.cosmos3_cutover --scoped-stop --temp-dir {lifecycle.temp_dir}",
        BARE_COSMOS3_RESTORE_COMMAND,
    )


def guarded_cutover_commands(serve_config_path: str) -> CutoverCommands:
    """Build the exact explicit-address commands after evidence validation."""
    _require(bool(serve_config_path), "serve config path is required")
    lifecycle = cosmos3_lifecycle()
    lifecycle.assert_gpu_pinned()
    worker_ports = lifecycle.ports.worker_port_list.split(",")
    _require(len(worker_ports) == 32 and worker_ports[0] == "26900" and worker_ports[-1] == "26931",
             "GPU6 worker ports must be the explicit 26900..26931 list")
    _require("-" not in lifecycle.ports.worker_port_list, "GPU6 worker ports must not use range syntax")
    return CutoverCommands(
        start_cluster=lifecycle.startup_command(),
        deploy=lifecycle.serve_deploy_command(serve_config_path),
        status=lifecycle.serve_status_command(),
        scoped_rollback=scoped_rollback_commands(),
    )


def _command_args(command: str, env: dict[str, str]) -> list[str]:
    """Split a lifecycle command, applying its leading ``NAME=value`` entries to env."""
    parts = shlex.split(command)
    while parts and "=" in parts[0] and not parts[0].startswith("-"):
        name, value = parts[0].split("=", 1)
        if not name.isidentifier():
            break
        env[name] = value
        parts.pop(0)
    _require(bool(parts), "lifecycle command has no executable")
    return parts


def execute_guarded_cutover(serve_config_path: str, *, artifacts_dir: str | Path = STANDALONE_ARTIFACTS_DIR,
                            bare_listener: Callable[[str, int], bool] = _port_is_listening,
                            run: Callable[..., Any] = subprocess.run) -> CutoverCommands:
    """Start/deploy only after standalone proof and an explicit bare-service stop.

    This function deliberately cannot stop the bare service.  If deployment fails it
    leaves the candidate rollback decision explicit rather than restarting a process
    behind the operator's back.
    """
    validate_standalone_artifacts(artifacts_dir)
    _require(not bare_listener("127.0.0.1", 8001),
             "bare Cosmos3 is still listening on 127.0.0.1:8001; stop it under an authorized cutover window first")
    commands = guarded_cutover_commands(serve_config_path)
    env = dict(os.environ)
    env.pop("RAY_ADDRESS", None)
    for command in (commands.start_cluster, commands.deploy):
        run(_command_args(command, env), check=True, env=env)
    return commands


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate or execute the guarded Cosmos3 GPU6 candidate cutover")
    parser.add_argument("--standalone-artifacts-dir", default=STANDALONE_ARTIFACTS_DIR)
    parser.add_argument("--serve-config")
    parser.add_argument("--execute", action="store_true", help="start and deploy the GPU6 candidate after all guards pass")
    parser.add_argument("--bare-cosmos3-stopped", action="store_true",
                        help="acknowledge that an authorized operator stopped the production 8001 service")
    parser.add_argument("--scoped-stop", action="store_true", help="stop only candidate Ray PIDs matching --temp-dir")
    parser.add_argument("--temp-dir", default=cosmos3_lifecycle().temp_dir)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.scoped_stop:
        stopped = stop_scoped_candidate(args.temp_dir)
        print(json.dumps({"temp_dir": args.temp_dir, "stopped_pids": stopped}))
        return 0
    try:
        evidence = validate_standalone_artifacts(args.standalone_artifacts_dir)
        if not args.execute:
            print(json.dumps({"status": "preflight_passed", "interpreter": evidence.interpreter,
                              "verify_report": str(evidence.verify_report), "finalize_report": str(evidence.finalize_report)}))
            return 0
        _require(args.bare_cosmos3_stopped, "--execute requires --bare-cosmos3-stopped")
        commands = execute_guarded_cutover(args.serve_config, artifacts_dir=args.standalone_artifacts_dir)
        print(json.dumps({"status": "candidate_deployed", "deploy": commands.deploy, "status_command": commands.status}))
        return 0
    except CutoverGateError as exc:
        print(f"cutover gate refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through main() tests
    raise SystemExit(main())
