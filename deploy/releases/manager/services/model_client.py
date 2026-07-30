#!/usr/bin/env python3
"""Caller-side request JSON uploader and resident-service client."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PATH_KEYS = {
    "input_video", "raw_frame_manifest", "source_frame_manifest", "calibration_contract",
    "camera_artifact", "track_manifest", "mask_manifest", "dynamic_mask", "frame_chunks",
    "source", "rgb", "raw_frame_path", "image_path", "manifest_path", "artifact_path",
}
SKIP_KEYS = {"output_dir", "run_root", "repo_root", "checkpoint", "hawor_root", "model_root", "work_root", "output_root"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_request(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"request JSON must be an object: {path}")
    return payload


def _resolve(raw: str, base: Path) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _walk_input_paths(value: Any, *, key: str | None, base: Path, found: dict[str, tuple[str, Path]]) -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            _walk_input_paths(child, key=str(child_key), base=base, found=found)
        return
    if isinstance(value, list):
        for child in value:
            _walk_input_paths(child, key=key, base=base, found=found)
        return
    if not isinstance(value, str) or key in SKIP_KEYS:
        return
    if key not in PATH_KEYS:
        return
    path = _resolve(value, base)
    if not path.is_file():
        return
    found[str(path)] = (key or "input", path)
    if path.suffix.lower() == ".json":
        try:
            nested = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        _walk_input_paths(nested, key="manifest_reference", base=path.parent, found=found)


def collect_input_artifacts(request: dict[str, Any], request_path: Path) -> list[dict[str, Any]]:
    base = request_path.parent
    found: dict[str, tuple[str, Path]] = {}
    _walk_input_paths(request, key=None, base=base, found=found)
    declared = request.get("input_artifacts")
    if isinstance(declared, list):
        for row in declared:
            if isinstance(row, dict) and row.get("path"):
                path = _resolve(str(row["path"]), base)
                if path.is_file():
                    found[str(path)] = (str(row.get("role") or "input"), path)
    ordered = sorted(found.items())

    def describe(item: tuple[str, tuple[str, Path]]) -> dict[str, Any]:
        source, (role, path) = item
        artifact_id = sha256_file(path)
        return {"role": role, "source_path": source, "artifact_id": artifact_id, "sha256": artifact_id, "bytes": path.stat().st_size}

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(ordered))), thread_name_prefix="artifact_hash") as pool:
        result = list(pool.map(describe, ordered))
    if not result:
        raise ValueError(f"request declares no materializable input files: {request_path}")
    return result


def _request(url: str, method: str, body: bytes | None = None, headers: dict[str, str] | None = None, timeout: float = 3600.0) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"service returned non-object JSON: {url}")
    return payload


def _artifact_exists(endpoint: str, artifact_id: str, timeout: float) -> bool:
    request = urllib.request.Request(f"{endpoint.rstrip('/')}/v1/artifacts/{artifact_id}", method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def upload_and_infer(request_json: Path, *, endpoint: str, timeout_s: float = 86400.0, request_id: str | None = None) -> dict[str, Any]:
    request_json = request_json.resolve()
    payload = load_request(request_json)
    artifacts = collect_input_artifacts(payload, request_json)
    endpoint = endpoint.rstrip("/")
    upload_started = time.time()

    def upload_one(artifact: dict[str, Any]) -> dict[str, Any]:
        artifact_id = str(artifact["artifact_id"])
        if _artifact_exists(endpoint, artifact_id, min(timeout_s, 30.0)):
            return {"status": "ok", "artifact_id": artifact_id, "deduplicated": True}
        path = Path(str(artifact["source_path"]))
        raw = path.read_bytes()
        return _request(
            f"{endpoint}/v1/artifacts/{artifact_id}",
            "PUT",
            raw,
            {"Content-Type": mimetypes.guess_type(str(path))[0] or "application/octet-stream", "Content-Length": str(len(raw))},
            timeout=timeout_s,
        )

    # Frame manifests contain many independent files. Uploading them serially
    # makes request arrival depend on frame count and prevents a logical wave
    # from entering the service coalescer together.
    workers = min(8, max(1, len(artifacts)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="artifact_upload") as pool:
        upload_results = list(pool.map(upload_one, artifacts))
    upload_finished = time.time()
    request_id = request_id or str(payload.get("request_id") or f"{payload.get('job_id') or 'job'}_{time.time_ns()}")
    envelope = {
        "schema": "ego.annotation.resident_transport.v1",
        "request_id": request_id,
        "request_emitted_at_unix": upload_finished,
        "request": payload,
        "artifacts": artifacts,
    }
    model = str(payload.get("model") or "model").split("_")[0]
    response = _request(
        f"{endpoint}/v1/{model}/infer",
        "POST",
        json.dumps(envelope, ensure_ascii=False).encode("utf-8"),
        {"Content-Type": "application/json"},
        timeout=timeout_s,
    )
    response["client_transport"] = {
        "request_json": str(request_json),
        "request_id": request_id,
        "artifact_count": len(artifacts),
        "upload_elapsed_s": upload_finished - upload_started,
        "request_emitted_at_unix": envelope["request_emitted_at_unix"],
        "upload_results": upload_results,
    }
    return response


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--timeout-s", type=float, default=86400.0)
    parser.add_argument("--request-id", default=None)
    args = parser.parse_args()
    response = upload_and_infer(args.request_json, endpoint=args.endpoint, timeout_s=args.timeout_s, request_id=args.request_id)
    print(json.dumps(response, indent=2, ensure_ascii=False))
    if response.get("status") not in {"ok", "completed_with_errors", "completed_with_partial_camera_coverage"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
