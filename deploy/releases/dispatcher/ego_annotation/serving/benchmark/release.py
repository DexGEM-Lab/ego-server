"""Content-addressed experiment releases.

A release is a directory whose name is its SHA256 digest.  The digest covers a
canonical sorted manifest of relative file names and bytes; ``RELEASE.json`` is
excluded from that manifest so the attestation can describe itself without
changing the digest.  The helpers are deliberately Ray-free and are used by both
plan-time tooling and the detached worker before importing application code.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_RELEASE_FILE = "RELEASE.json"
_HEX64 = set("0123456789abcdef")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_digest(path: str | Path) -> str:
    """Hash a checkpoint file or directory using sorted relative paths."""
    root = Path(path).resolve(strict=True)
    if root.is_file():
        return _sha256_file(root)
    entries: list[dict[str, str]] = []
    for file_path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        entries.append({"path": file_path.relative_to(root).as_posix(), "sha256": _sha256_file(file_path)})
    return hashlib.sha256(_canonical_json(entries)).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def file_manifest(root: str | Path) -> list[dict[str, str]]:
    base = Path(root).resolve(strict=True)
    if not base.is_dir():
        raise ValueError(f"release source is not a directory: {base}")
    entries: list[dict[str, str]] = []
    for path in sorted(p for p in base.rglob("*") if p.is_file() and not p.is_symlink()):
        relative = path.relative_to(base)
        rel = relative.as_posix()
        # Python imports may create derived bytecode inside an otherwise immutable,
        # writable release. Bytecode is not source identity and is excluded at build
        # time, so verification must ignore it symmetrically.
        if rel == _RELEASE_FILE or "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        entries.append({"path": rel, "sha256": _sha256_file(path)})
    return entries


def release_digest_from_manifest(manifest: Sequence[Mapping[str, str]]) -> str:
    canonical = [{"path": str(item["path"]), "sha256": str(item["sha256"])} for item in manifest]
    canonical.sort(key=lambda item: item["path"])
    return hashlib.sha256(_canonical_json(canonical)).hexdigest()


def _git_source_sha(root: Path) -> str:
    try:
        value = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return value


def build_release(
    source_root: str | Path,
    output_root: str | Path,
    *,
    source_sha: str | None = None,
    include: Sequence[str] | None = None,
) -> Path:
    """Copy source into a digest-named release and write its attestation.

    ``include`` is optional; when omitted every regular file is copied except
    VCS/cache material.  Symlinks are rejected rather than silently resolving a
    mutable ``current`` tree.
    """
    source = Path(source_root).resolve(strict=True)
    destination_root = Path(output_root).resolve()
    selected: list[Path] = []
    prefixes = tuple(Path(item) for item in (include or ()))
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        if path.is_symlink():
            raise ValueError(f"release source contains symlink: {rel}")
        if rel.as_posix() == _RELEASE_FILE or any(part in {".git", "__pycache__", ".pytest_cache"} for part in rel.parts):
            continue
        if prefixes and not any(rel == prefix or prefix in rel.parents for prefix in prefixes):
            continue
        selected.append(path)
    if not selected:
        raise ValueError("release source has no regular files to publish")
    manifest = [{"path": p.relative_to(source).as_posix(), "sha256": _sha256_file(p)} for p in selected]
    manifest.sort(key=lambda item: item["path"])
    digest = release_digest_from_manifest(manifest)
    destination = destination_root / digest
    if destination.exists():
        verify_release(destination, expected_digest=digest, expected_source_sha=source_sha)
        return destination
    staging = destination_root / f".{digest}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for source_file in selected:
            target = staging / source_file.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target)
        attestation = {
            "schema": "ego.experiment-release.v1",
            "release_digest": digest,
            "source_sha": source_sha or _git_source_sha(source),
            "manifest": manifest,
        }
        (staging / _RELEASE_FILE).write_bytes(_canonical_json(attestation) + b"\n")
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verify_release(destination, expected_digest=digest, expected_source_sha=source_sha)
    return destination


def verify_release(
    path: str | Path,
    *,
    expected_digest: str | None = None,
    expected_source_sha: str | None = None,
) -> "VerifiedRelease":
    requested = Path(path)
    if requested.is_symlink():
        raise ValueError("release root must be a real directory, not a symlink")
    root = requested.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("release root must be a real directory")
    raw = json.loads((root / _RELEASE_FILE).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema") != "ego.experiment-release.v1":
        raise ValueError("invalid release attestation schema")
    manifest = raw.get("manifest")
    if not isinstance(manifest, list):
        raise ValueError("release attestation manifest is missing")
    actual_manifest = file_manifest(root)
    canonical = [{"path": str(item.get("path")), "sha256": str(item.get("sha256"))} for item in manifest if isinstance(item, Mapping)]
    canonical.sort(key=lambda item: item["path"])
    if canonical != actual_manifest:
        raise ValueError("release file manifest does not match bytes on disk")
    actual_digest = release_digest_from_manifest(actual_manifest)
    if raw.get("release_digest") != actual_digest or root.name != actual_digest:
        raise ValueError("release digest does not match directory/content")
    if expected_digest is not None and expected_digest != actual_digest:
        raise ValueError("release digest differs from requested digest")
    source_sha = raw.get("source_sha")
    if not isinstance(source_sha, str) or not source_sha:
        raise ValueError("release source_sha is missing")
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise ValueError("release source_sha differs from requested provenance")
    return VerifiedRelease(root, actual_digest, source_sha, tuple((item["path"], item["sha256"]) for item in actual_manifest))


@dataclass(frozen=True)
class VerifiedRelease:
    path: Path
    release_digest: str
    source_sha: str
    manifest: tuple[tuple[str, str], ...]

    @property
    def module_root(self) -> Path:
        return self.path


def checkpoint_digest(path: str | Path) -> str:
    """Alias with explicit checkpoint semantics for worker provenance."""
    return artifact_digest(path)


@dataclass(frozen=True)
class WorkerRuntimeEvidence:
    """Facts derived inside a loaded experimental worker, never caller labels."""

    release_digest: str
    source_sha: str
    module_root: Path
    checkpoint_digest: str
    worker_pid: int
    cuda_uuid: str
    physical_gpu: int


def derive_worker_runtime_evidence(
    *,
    release_root: str | Path,
    checkpoint_path: str | Path,
    imported_module_file: str | Path,
) -> WorkerRuntimeEvidence:
    """Bind imported code, checkpoint bytes, process, and CUDA device to one worker.

    A CUDA-masked worker sees its assigned physical device as logical device zero.
    The single numeric ``CUDA_VISIBLE_DEVICES`` token provides the physical index;
    the CUDA runtime UUID is later compared with the immediate NVML preflight UUID.
    Together those independently bind the worker to the authorized physical GPU.
    """
    verified = verify_release(release_root)
    module_file = Path(imported_module_file).resolve(strict=True)
    try:
        module_file.relative_to(verified.module_root)
    except ValueError as exc:
        raise RuntimeError(
            f"worker imported application module outside verified release root: {module_file}"
        ) from exc
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if len(tokens) != 1 or not tokens[0].isdigit():
        raise RuntimeError("experimental worker requires one numeric CUDA_VISIBLE_DEVICES physical GPU")
    physical_gpu = int(tokens[0])
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            raise RuntimeError("experimental worker has no CUDA runtime")
        device_count = getattr(torch.cuda, "device_count", lambda: 1)()
        if device_count != 1:
            raise RuntimeError(f"experimental worker must see exactly one CUDA device, got {device_count}")
        properties = torch.cuda.get_device_properties(0)
        torch_uuid = getattr(properties, "uuid", None)
        if torch_uuid:
            cuda_uuid = str(torch_uuid)
        else:
            # Torch 1.13 in ray_serve_hawor does not consistently expose UUID on
            # _CudaDeviceProperties. Prefer NVML (Ray's physical-device evidence);
            # fall back to nvidia-smi, which is indexed by the same preflighted CVD
            # physical id and requires no Python package in the exact runtime ABI.
            nvml_error: Exception | None = None
            try:
                import pynvml  # type: ignore

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu)
                nvml_uuid = pynvml.nvmlDeviceGetUUID(handle)
                cuda_uuid = nvml_uuid.decode() if isinstance(nvml_uuid, bytes) else str(nvml_uuid)
            except Exception as exc:  # ImportError or NVML init failure
                nvml_error = exc
                import subprocess

                query = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i",
                        str(physical_gpu),
                        "--query-gpu=uuid",
                        "--format=csv,noheader",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                smi_uuid = query.stdout.strip().splitlines()[0].strip() if query.returncode == 0 and query.stdout.strip() else ""
                if smi_uuid:
                    cuda_uuid = smi_uuid
                else:
                    raise RuntimeError(
                        f"cannot derive experimental CUDA UUID: pynvml failed ({nvml_error}); "
                        f"nvidia-smi rc={query.returncode} stdout={query.stdout.strip()!r} stderr={query.stderr.strip()!r}"
                    ) from exc
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"cannot derive experimental CUDA UUID: {exc}") from exc
    if not cuda_uuid:
        raise RuntimeError("experimental CUDA UUID is empty")
    return WorkerRuntimeEvidence(
        release_digest=verified.release_digest,
        source_sha=verified.source_sha,
        module_root=verified.module_root,
        checkpoint_digest=artifact_digest(checkpoint_path),
        worker_pid=os.getpid(),
        cuda_uuid=cuda_uuid,
        physical_gpu=physical_gpu,
    )
