#!/usr/bin/env python3
"""Check whether a VGGT/Omega camera backend runtime is ready for a smoke run.

This script is intentionally preflight-only: it checks repository files, exact
Python import symbols, checkpoint/download policy, and CPU checkpoint
serialization. It does not instantiate models, download weights, use GPU, or run
inference.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def third_party_hint(repo_root: Path, backend: str) -> Path:
    return repo_root / "third_party" / ("vggt" if backend == "vggt" else "vggt-omega")


def expected_symbol_labels(backend: str) -> list[str]:
    if backend == "vggt":
        return ["torch", "vggt.models.vggt.VGGT", "vggt.utils.pose_enc.pose_encoding_to_extri_intri"]
    if backend == "vggt_omega":
        return ["torch", "vggt_omega.models.VGGTOmega", "vggt_omega.utils.pose_enc.encoding_to_camera"]
    raise RuntimeError(f"unsupported backend: {backend}")


def resolve_checkpoint(raw: Path | None) -> Path | None:
    if raw is None:
        return None
    return raw.expanduser().resolve()


def probe_python_environment(
    python_path: Path,
    *,
    backend: str,
    repo_root: Path,
    checkpoint: Path | None,
    timeout_s: float,
) -> dict[str, Any]:
    hint = third_party_hint(repo_root, backend)
    checkpoint_arg = str(checkpoint) if checkpoint is not None else None
    code = """
import json
import sys
from pathlib import Path

backend = {backend!r}
hint = Path({hint!r})
checkpoint_arg = {checkpoint!r}
if hint.exists():
    sys.path.insert(0, str(hint))

imports = {{}}
torch_obj = None


def record_import(label, fn):
    try:
        value = fn()
        imports[label] = {{
            "imported": True,
            "module": getattr(value, "__module__", None),
            "qualname": getattr(value, "__qualname__", getattr(value, "__name__", None)),
            "type": type(value).__name__,
        }}
        return value
    except Exception as exc:
        imports[label] = {{"imported": False, "error": repr(exc)}}
        return None


torch_obj = record_import("torch", lambda: __import__("torch"))
if backend == "vggt":
    record_import("vggt.models.vggt.VGGT", lambda: getattr(__import__("vggt.models.vggt", fromlist=["VGGT"]), "VGGT"))
    record_import(
        "vggt.utils.pose_enc.pose_encoding_to_extri_intri",
        lambda: getattr(__import__("vggt.utils.pose_enc", fromlist=["pose_encoding_to_extri_intri"]), "pose_encoding_to_extri_intri"),
    )
elif backend == "vggt_omega":
    record_import("vggt_omega.models.VGGTOmega", lambda: getattr(__import__("vggt_omega.models", fromlist=["VGGTOmega"]), "VGGTOmega"))
    record_import(
        "vggt_omega.utils.pose_enc.encoding_to_camera",
        lambda: getattr(__import__("vggt_omega.utils.pose_enc", fromlist=["encoding_to_camera"]), "encoding_to_camera"),
    )
else:
    imports["backend"] = {{"imported": False, "error": f"unsupported backend: {{backend}}"}}

checkpoint_load = {{"status": "not_requested"}}
if checkpoint_arg is not None:
    if torch_obj is None:
        checkpoint_load = {{"status": "failed", "error": "torch_import_failed"}}
    else:
        try:
            state = torch_obj.load(checkpoint_arg, map_location="cpu")
            summary = {{"status": "ok", "type": type(state).__name__}}
            if isinstance(state, dict):
                keys = [str(key) for key in list(state.keys())[:12]]
                summary.update({{"top_level_key_count": len(state), "top_level_key_sample": keys}})
            checkpoint_load = summary
        except Exception as exc:
            checkpoint_load = {{"status": "failed", "error": repr(exc)}}

status = "ok"
if any(not row.get("imported") for row in imports.values()):
    status = "failed"
if checkpoint_load.get("status") == "failed":
    status = "failed"
print(json.dumps({{
    "status": status,
    "python": sys.executable,
    "third_party_hint": str(hint),
    "imports": imports,
    "checkpoint_load": checkpoint_load,
}}, sort_keys=True))
""".format(backend=backend, hint=str(hint), checkpoint=checkpoint_arg)
    try:
        proc = subprocess.run(
            [str(python_path), "-c", code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=float(timeout_s),
        )
    except FileNotFoundError as exc:
        return {"status": "failed", "error": f"python_not_found: {exc}", "python": str(python_path), "imports": {}, "checkpoint_load": {"status": "not_requested"}}
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "error": f"python_probe_timeout_after_{timeout_s}s",
            "python": str(python_path),
            "stdout": exc.stdout,
            "stderr": exc.stderr,
            "imports": {},
            "checkpoint_load": {"status": "not_requested"},
        }
    if proc.returncode != 0:
        return {
            "status": "failed",
            "returncode": int(proc.returncode),
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "python": str(python_path),
            "imports": {},
            "checkpoint_load": {"status": "not_requested"},
        }
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {
            "status": "failed",
            "error": f"invalid_probe_json: {exc}",
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "python": str(python_path),
            "imports": {},
            "checkpoint_load": {"status": "not_requested"},
        }
    if not isinstance(payload, dict):
        return {
            "status": "failed",
            "error": "probe_output_not_object",
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "python": str(python_path),
            "imports": {},
            "checkpoint_load": {"status": "not_requested"},
        }
    return payload


def repo_file_checks(repo_root: Path) -> list[dict[str, Any]]:
    rels = [
        "scripts/run_v22_resident_vggt_camera_batch.py",
        "scripts/run_v22_camera_trajectory_stage.py",
        "scripts/v22_model_request_helpers.py",
    ]
    return [{"path": str(repo_root / rel), "relative_path": rel, "exists": (repo_root / rel).exists()} for rel in rels]


def checkpoint_status(backend: str, checkpoint: Path | None, allow_remote_model_download: bool) -> dict[str, Any]:
    if checkpoint is not None:
        exists = checkpoint.exists()
        payload: dict[str, Any] = {"status": "checkpoint_present" if exists else "checkpoint_missing", "path": str(checkpoint), "exists": exists}
        if exists:
            payload["size_bytes"] = checkpoint.stat().st_size
        return payload
    if backend == "vggt" and allow_remote_model_download:
        return {"status": "remote_download_explicitly_allowed", "path": None, "exists": None, "network_or_cache_required": True}
    if backend == "vggt":
        return {"status": "blocked_missing_checkpoint_or_download_permission", "path": None, "exists": None, "required_action": "provide --checkpoint or pass --allow-remote-model-download explicitly"}
    return {"status": "blocked_missing_checkpoint", "path": None, "exists": None, "required_action": "provide --checkpoint for vggt_omega"}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    checkpoint = resolve_checkpoint(args.checkpoint)
    repo_checks = repo_file_checks(repo_root)
    weight_status = checkpoint_status(args.backend, checkpoint, bool(args.allow_remote_model_download))
    checkpoint_to_probe = checkpoint if checkpoint is not None and checkpoint.exists() else None
    python_probe = probe_python_environment(
        args.python_path,
        backend=args.backend,
        repo_root=repo_root,
        checkpoint=checkpoint_to_probe,
        timeout_s=float(args.timeout_s),
    )
    blocked_reasons: list[str] = []
    for row in repo_checks:
        if not row["exists"]:
            blocked_reasons.append(f"missing_repo_file:{row['relative_path']}")
    if python_probe.get("status") != "ok":
        blocked_reasons.append(f"python_probe_failed:{python_probe.get('error') or python_probe.get('returncode') or python_probe.get('status')}")
    imports = python_probe.get("imports") or {}
    for label in expected_symbol_labels(args.backend):
        row = imports.get(label)
        if not isinstance(row, dict) or not row.get("imported"):
            blocked_reasons.append(f"missing_or_unimportable_symbol:{label}")
    checkpoint_load = python_probe.get("checkpoint_load") or {}
    if checkpoint_to_probe is not None and checkpoint_load.get("status") != "ok":
        blocked_reasons.append("checkpoint_not_torch_loadable")
    if str(weight_status.get("status", "")).startswith("blocked") or weight_status.get("status") == "checkpoint_missing":
        blocked_reasons.append(str(weight_status.get("status")))
    return {
        "schema": "v22_vggt_camera_backend_readiness.v1",
        "created_at": utc_now(),
        "status": "blocked" if blocked_reasons else "ok",
        "backend": args.backend,
        "repo_root": str(repo_root),
        "python": str(args.python_path),
        "third_party_hint": str(third_party_hint(repo_root, args.backend)),
        "repo_files": repo_checks,
        "python_probe": python_probe,
        "checkpoint": weight_status,
        "blocked_reasons": blocked_reasons,
        "claim_scope": "Preflight readiness only: exact imports and CPU checkpoint deserialization are checked; no model instantiation, load_state_dict validation, weight download, GPU inference, trajectory output, or quality claim.",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("vggt", "vggt_omega"), required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--python", type=Path, dest="python_path", default=Path(sys.executable))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--allow-remote-model-download", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate(args)
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
