"""Immutable, content-addressed DROID-SLAM source releases for experiments.

The production GPU2 process intentionally remains on its historical loaded source.
Cold experimental workers use only a verified copy of the recovered HaWoR-vendored
DROID source.  This module is deliberately dependency-free so source identity is
established before Torch, DROID extensions, or Ray are imported.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SOURCE_MANIFEST_NAME = "DROID_SOURCE_MANIFEST.json"
SOURCE_SCHEMA = "ego.droid-source-release.v1"
RECOVERED_AMENDMENT_ID = "recovered-hawor-droid-core-v1"
# Canonical core-group digest over the six coupled modules of the verified candidate-B
# tree (droid_slam prefix, trajectory_filler included), computed 2026-07-22 by this
# module's own construction. This supersedes the earlier recorded 41ab291b… value,
# which came from a provenance one-off over a different six-module set (droid.py in
# place of trajectory_filler.py) with an undocumented construction; every overlapping
# per-file hash was re-verified unchanged against that report.
EXPECTED_CORE_GROUP_DIGEST = "523bc9b92a8f11f3dfd061efd86d25c5b687d7983592cf5f5bc3f45b6ebaa9c6"
# These are the six coupled Python modules whose historical agreement supports the
# amended baseline.  The source release still manifests every regular file.
CORE_MODULES = (
    "droid_slam/droid_net.py",
    "droid_slam/depth_video.py",
    "droid_slam/motion_filter.py",
    "droid_slam/droid_frontend.py",
    "droid_slam/droid_backend.py",
    "droid_slam/trajectory_filler.py",
)
_EXCLUDED_PARTS = frozenset({".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".cache"})
_EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})


class DroidSourceVerificationError(ValueError):
    """The recovered source no longer proves the planned dependency identity."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_excluded(relative: Path) -> bool:
    return any(part in _EXCLUDED_PARTS for part in relative.parts) or relative.suffix in _EXCLUDED_SUFFIXES


def _manifest_path(value: str) -> Path:
    relative = Path(value)
    if (
        not value
        or relative.is_absolute()
        or ".." in relative.parts
        or value != relative.as_posix()
        or value == SOURCE_MANIFEST_NAME
    ):
        raise DroidSourceVerificationError("DROID source manifest contains invalid path")
    return relative


def _assert_real_tree(root: Path, *, allow_manifest: bool) -> None:
    if root.is_symlink() or not root.is_dir():
        raise DroidSourceVerificationError("DROID source root must be a real directory, not a symlink")
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        # An excluded cache is not identity-bearing, but symlink indirection is
        # never acceptable in a bundle transaction (including under a cache name).
        if path.is_symlink():
            raise DroidSourceVerificationError(f"DROID source contains symlink: {relative.as_posix()}")
        if _is_excluded(relative):
            continue
        if path.is_dir() or path.is_file():
            continue
        raise DroidSourceVerificationError(f"DROID source contains non-regular entry: {relative.as_posix()}")
    if not allow_manifest and (root / SOURCE_MANIFEST_NAME).exists():
        raise DroidSourceVerificationError(f"candidate source must not contain {SOURCE_MANIFEST_NAME}")


def source_file_manifest(root: str | Path, *, include_excluded: bool = False) -> list[dict[str, str]]:
    """Return regular source files in deterministic order.

    Candidates deliberately omit VCS and interpreter-cache material.  A published
    release, however, is a closed tree: callers use ``include_excluded=True`` so
    injected files cannot disappear from its byte-level inventory.
    """
    requested = Path(root)
    if requested.is_symlink():
        raise DroidSourceVerificationError("DROID source root must not be a symlink")
    base = requested.resolve(strict=True)
    _assert_real_tree(base, allow_manifest=True)
    manifest: list[dict[str, str]] = []
    paths = sorted(base.rglob("*"), key=lambda path: path.relative_to(base).as_posix())
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        if relative.as_posix() == SOURCE_MANIFEST_NAME:
            continue
        if not include_excluded and _is_excluded(relative):
            continue
        manifest.append({"path": relative.as_posix(), "sha256": _sha256_file(path)})
    if not manifest:
        raise DroidSourceVerificationError("DROID source contains no regular files")
    return manifest


def _assert_manifest_files_are_private(
    root: Path, manifest: Sequence[Mapping[str, str] | tuple[str, str]],
) -> None:
    """Reject hard links: a source byte must have one release-owned inode."""
    for row in manifest:
        relative = _manifest_path(str(row["path"] if isinstance(row, Mapping) else row[0]))
        path = root / relative
        if path.stat().st_nlink > 1:
            raise DroidSourceVerificationError(f"DROID source manifest file has hard links: {relative.as_posix()}")


def _assert_release_contains_no_bytecode(root: Path) -> None:
    """Bytecode is executable import state, never source-release content."""
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(root).suffix in _EXCLUDED_SUFFIXES:
            raise DroidSourceVerificationError(
                f"DROID source release contains executable bytecode: {path.relative_to(root).as_posix()}"
            )


def source_digest_from_manifest(manifest: Sequence[Mapping[str, str]]) -> str:
    normalized = [{"path": str(row["path"]), "sha256": str(row["sha256"])} for row in manifest]
    normalized.sort(key=lambda row: row["path"])
    if len({row["path"] for row in normalized}) != len(normalized):
        raise DroidSourceVerificationError("DROID source manifest contains duplicate paths")
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def core_hashes_from_manifest(manifest: Sequence[Mapping[str, str]]) -> dict[str, str]:
    by_path = {str(row["path"]): str(row["sha256"]) for row in manifest}
    missing = [name for name in CORE_MODULES if name not in by_path]
    if missing:
        raise DroidSourceVerificationError("DROID source is missing required core modules: " + ", ".join(missing))
    return {name: by_path[name] for name in CORE_MODULES}


def core_group_digest(core_hashes: Mapping[str, str]) -> str:
    missing = [name for name in CORE_MODULES if name not in core_hashes]
    extras = sorted(set(core_hashes) - set(CORE_MODULES))
    if missing or extras:
        raise DroidSourceVerificationError("DROID core hash set must contain exactly the six required modules")
    rows = [{"path": name, "sha256": str(core_hashes[name])} for name in CORE_MODULES]
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


@dataclass(frozen=True)
class VerifiedDroidSourceRelease:
    path: Path
    source_root: Path
    source_digest: str
    amendment_id: str
    core_group_digest: str
    manifest: tuple[tuple[str, str], ...]


def _read_manifest(root: Path) -> Mapping[str, Any]:
    manifest_path = root / SOURCE_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DroidSourceVerificationError("DROID source release manifest is missing or a symlink")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DroidSourceVerificationError("DROID source release manifest is unreadable") from exc
    if not isinstance(raw, Mapping) or raw.get("schema") != SOURCE_SCHEMA:
        raise DroidSourceVerificationError("invalid DROID source release manifest schema")
    return raw


def verify_droid_source_release(
    release_path: str | Path,
    *,
    expected_digest: str | None = None,
    expected_amendment_id: str | None = RECOVERED_AMENDMENT_ID,
    expected_core_group_digest: str | None = EXPECTED_CORE_GROUP_DIGEST,
) -> VerifiedDroidSourceRelease:
    """Verify a release from bytes, never from an environment echo or path label."""
    requested = Path(release_path)
    if requested.is_symlink():
        raise DroidSourceVerificationError("DROID source release root must not be a symlink")
    root = requested.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise DroidSourceVerificationError("DROID source release root must be a real directory")
    raw = _read_manifest(root)
    source_root = root / "droid_slam"
    if source_root.is_symlink() or not source_root.is_dir():
        raise DroidSourceVerificationError("DROID source release must contain real droid_slam root")
    _assert_real_tree(root, allow_manifest=True)
    raw_manifest = raw.get("manifest")
    if not isinstance(raw_manifest, list):
        raise DroidSourceVerificationError("DROID source manifest is missing complete file manifest")
    declared = []
    for row in raw_manifest:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
            raise DroidSourceVerificationError("DROID source manifest contains invalid entry")
        relative = _manifest_path(row["path"])
        if _is_excluded(relative):
            raise DroidSourceVerificationError("DROID source release manifest cannot declare cache or bytecode files")
        declared.append({"path": row["path"], "sha256": row["sha256"]})
    declared.sort(key=lambda row: row["path"])
    _assert_release_contains_no_bytecode(root)
    actual = source_file_manifest(root, include_excluded=True)
    if declared != actual:
        raise DroidSourceVerificationError("DROID source manifest does not match complete bytes on disk")
    _assert_manifest_files_are_private(root, actual)
    digest = source_digest_from_manifest(actual)
    if raw.get("source_digest") != digest or root.name != digest:
        raise DroidSourceVerificationError("DROID source digest does not match directory/content")
    if expected_digest is not None and digest != expected_digest:
        raise DroidSourceVerificationError("DROID source digest differs from planned dependency digest")
    amendment_id = raw.get("amendment_id")
    if not isinstance(amendment_id, str) or not amendment_id:
        raise DroidSourceVerificationError("DROID source amendment id is missing")
    if expected_amendment_id is not None and amendment_id != expected_amendment_id:
        raise DroidSourceVerificationError("DROID source amendment id differs from planned amendment")
    core_hashes = core_hashes_from_manifest(actual)
    if raw.get("core_hashes") != core_hashes:
        raise DroidSourceVerificationError("DROID source core hashes do not match manifest bytes")
    actual_group_digest = core_group_digest(core_hashes)
    if raw.get("core_group_digest") != actual_group_digest:
        raise DroidSourceVerificationError("DROID source core group digest does not match core hashes")
    if expected_core_group_digest is not None and actual_group_digest != expected_core_group_digest:
        raise DroidSourceVerificationError("DROID source core group digest differs from accepted recovered baseline")
    origin = raw.get("origin_evidence")
    if not isinstance(origin, Mapping) or not origin:
        raise DroidSourceVerificationError("DROID source origin evidence is missing")
    return VerifiedDroidSourceRelease(
        root, source_root, digest, amendment_id, actual_group_digest,
        tuple((row["path"], row["sha256"]) for row in actual),
    )


def build_droid_source_release(
    candidate_root: str | Path,
    output_root: str | Path,
    *,
    origin_evidence: Mapping[str, Any],
    amendment_id: str = RECOVERED_AMENDMENT_ID,
    expected_core_group_digest: str = EXPECTED_CORE_GROUP_DIGEST,
) -> VerifiedDroidSourceRelease:
    """Snapshot a candidate root atomically, rejecting races and symlink indirection."""
    requested = Path(candidate_root)
    if requested.is_symlink():
        raise DroidSourceVerificationError("DROID candidate root must not be a symlink")
    candidate = requested.resolve(strict=True)
    _assert_real_tree(candidate, allow_manifest=False)
    initial_manifest = source_file_manifest(candidate)
    _assert_manifest_files_are_private(candidate, initial_manifest)
    initial_core = core_hashes_from_manifest(initial_manifest)
    actual_group_digest = core_group_digest(initial_core)
    if actual_group_digest != expected_core_group_digest:
        raise DroidSourceVerificationError("DROID candidate core group digest differs from accepted recovered baseline")
    if not isinstance(origin_evidence, Mapping) or not origin_evidence:
        raise DroidSourceVerificationError("DROID source release requires non-empty origin evidence")
    # The supplied evidence records recovery/provenance facts; retain the actual
    # candidate location too so the release can be traced without making that
    # mutable location a runtime import path.
    origin = {**dict(origin_evidence), "candidate_root": str(candidate)}
    digest = source_digest_from_manifest(initial_manifest)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise DroidSourceVerificationError("DROID source release output root must not be a symlink")
    destination = output / digest
    if destination.exists() or destination.is_symlink():
        return verify_droid_source_release(
            destination, expected_digest=digest, expected_amendment_id=amendment_id,
            expected_core_group_digest=expected_core_group_digest,
        )
    staging = output / f".{digest}.staging-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise DroidSourceVerificationError(f"DROID source staging path already exists: {staging}")
    try:
        for row in initial_manifest:
            source_file = candidate / row["path"]
            target = staging / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target)
        # Catch a mutation during copy rather than producing an ambiguous release.
        if source_file_manifest(candidate) != initial_manifest or source_file_manifest(staging) != initial_manifest:
            raise DroidSourceVerificationError("DROID candidate mutated while source release was copied")
        _assert_manifest_files_are_private(staging, initial_manifest)
        attestation = {
            "schema": SOURCE_SCHEMA,
            "source_digest": digest,
            "amendment_id": amendment_id,
            "core_hashes": initial_core,
            "core_group_digest": actual_group_digest,
            "origin_evidence": origin,
            "manifest": initial_manifest,
        }
        (staging / SOURCE_MANIFEST_NAME).write_bytes(_canonical_json(attestation) + b"\n")
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file():
                path.chmod(0o444)
            elif path.is_dir():
                path.chmod(0o555)
        staging.chmod(0o555)
        os.replace(staging, destination)
    except Exception:
        if staging.exists() and not staging.is_symlink():
            for path in sorted(staging.rglob("*"), reverse=True):
                try:
                    path.chmod(0o700 if path.is_dir() else 0o600)
                except OSError:
                    pass
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_droid_source_release(
        destination, expected_digest=digest, expected_amendment_id=amendment_id,
        expected_core_group_digest=expected_core_group_digest,
    )


def _code_behavior_signature(code: types.CodeType) -> tuple[object, ...]:
    """Compare executable semantics without trusting a module's ``__file__``."""
    constants = tuple(
        _code_behavior_signature(value) if isinstance(value, types.CodeType) else value
        for value in code.co_consts
    )
    return (
        code.co_name, code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
        code.co_flags, code.co_code, constants, code.co_names, code.co_varnames,
    )


def _expected_droid_net_methods(source_path: Path) -> dict[str, types.CodeType]:
    compiled = compile(source_path.read_bytes(), str(source_path), "exec", dont_inherit=True)
    droid_net = next(
        (value for value in compiled.co_consts if isinstance(value, types.CodeType) and value.co_name == "DroidNet"),
        None,
    )
    if droid_net is None:
        raise DroidSourceVerificationError("manifested droid_net source does not define DroidNet")
    return {
        value.co_name: value
        for value in droid_net.co_consts
        if isinstance(value, types.CodeType)
    }


def _actual_method_code(value: object) -> types.CodeType | None:
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    return getattr(value, "__code__", None) if isinstance(getattr(value, "__code__", None), types.CodeType) else None


def _assert_loaded_droid_net_behavior(module: object, source_path: Path) -> None:
    expected = _expected_droid_net_methods(source_path)
    loaded_class = getattr(module, "DroidNet", None)
    if not isinstance(loaded_class, type):
        raise DroidSourceVerificationError("imported droid_net does not expose manifested DroidNet class")
    for name, expected_code in expected.items():
        actual_code = _actual_method_code(vars(loaded_class).get(name))
        if actual_code is None or _code_behavior_signature(actual_code) != _code_behavior_signature(expected_code):
            raise DroidSourceVerificationError(f"imported droid_net behavior differs from manifested source: DroidNet.{name}")


def verify_imported_droid_module(
    module_file: str | Path,
    release: VerifiedDroidSourceRelease,
    *,
    module: object | None = None,
) -> Path:
    """Bind imported ``droid_net`` execution to the exact manifested source file."""
    path = Path(module_file)
    if path.is_symlink():
        raise DroidSourceVerificationError("imported droid_net module is a symlink")
    resolved = path.resolve(strict=True)
    expected = (release.source_root / "droid_net.py").resolve(strict=True)
    if resolved != expected:
        raise DroidSourceVerificationError("imported droid_net module is not the manifested core path")
    if not resolved.is_file():
        raise DroidSourceVerificationError("imported droid_net module is not a regular file")
    if dict(release.manifest).get("droid_slam/droid_net.py") != _sha256_file(resolved):
        raise DroidSourceVerificationError("imported droid_net source bytes differ from verified manifest")
    _assert_manifest_files_are_private(release.path, release.manifest)
    _assert_release_contains_no_bytecode(release.path)
    if module is not None:
        if not sys.dont_write_bytecode:
            raise DroidSourceVerificationError("DROID source imports require PYTHONDONTWRITEBYTECODE")
        _assert_loaded_droid_net_behavior(module, resolved)
    return resolved
