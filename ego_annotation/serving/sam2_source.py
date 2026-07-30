"""Immutable content-addressed SAM2 source bundles for isolated serving launches.

The bundle is created before importing Torch, Ray, checkpoints, or model code.  It
copies the complete exact-justified SAM2 tree, excluding VCS/cache/bytecode state,
and then verifies every published byte against a manifest.  The serving launch must
set ``EGO_SAM2_REPO`` to the digest-named release root, never its mutable origin.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SOURCE_MANIFEST_NAME = "SAM2_SOURCE_MANIFEST.json"
SOURCE_SCHEMA = "ego.sam2-source-release.v1"
RECOVERED_AMENDMENT_ID = "recovered-sam2-core-v1"
EXACT_GIT_HEAD = "e0a637648f587bc487b5184799a8201a07e3c536"

# The complete import closure observed for build_sam2 + SAM2ImagePredictor under
# the exact ray_serve_hands ABI.  Config-driven construction reaches these model
# modules after the two serving entrypoints import; the bundle still manifests every
# source byte, so this smaller set is an independently pinned execution core.
CORE_MODULE_HASHES = {
    "sam2/__init__.py": "b87ca1e95cd54b81766e8c74acf0e937952639529e40d2b4088693286f49419e",
    "sam2/build_sam.py": "856d64d71e44407401297b551884fb4cd1aba8082de0ff5fc60fac3dd42f094d",
    "sam2/modeling/__init__.py": "34bd8069c54764e7b8d73a78905dbe6467140a2f73170875128f6ca4d8cdd0aa",
    "sam2/modeling/backbones/__init__.py": "34bd8069c54764e7b8d73a78905dbe6467140a2f73170875128f6ca4d8cdd0aa",
    "sam2/modeling/backbones/hieradet.py": "03785581ca304d0451ae0df7a08ee0bf1e1dbe66fad285066e6b9ffc0d88d64f",
    "sam2/modeling/backbones/image_encoder.py": "16eaad232220386f510ca8dfab8655e63699977149a4daf9ce9c6f374e6777a9",
    "sam2/modeling/backbones/utils.py": "c4a0657db2a92bda2a7ff116cd4017f6b1c59af6408093d8874eb08a0df476ad",
    "sam2/modeling/memory_attention.py": "07358cb7f58ec3788e88ff4e6415f8a9d628a3963529d6e2c915494a5347c8e2",
    "sam2/modeling/memory_encoder.py": "73f7089ae5fdacdfcaaf3deca1ca6d9f84d89e7deb9cf97589622a9ef17ba42f",
    "sam2/modeling/position_encoding.py": "b51404718c0d38f381293c8e5e00a15d129651b7f09b1158002d8974a30967b5",
    "sam2/modeling/sam/__init__.py": "34bd8069c54764e7b8d73a78905dbe6467140a2f73170875128f6ca4d8cdd0aa",
    "sam2/modeling/sam/mask_decoder.py": "ca3523c58365574faddf1bfb54f374e4b4beba05127185ea2b17d22b916e099a",
    "sam2/modeling/sam/prompt_encoder.py": "4965ccb4a4504aa4d7246b2ad66ffeacd2626415c86a261dc1c65ffd8ae1d40d",
    "sam2/modeling/sam/transformer.py": "cda19052331e775190ce8b1159efcb828a4632d931a6d8bd4d47109f121782f1",
    "sam2/modeling/sam2_base.py": "69a46b44e8625f509791352bd09beaafe589cd7d872721384e63d37f9fdc6e41",
    "sam2/modeling/sam2_utils.py": "e35bbf13bc2e544a0272cd3f9539af38a752676ac3cd744be31cd8220afee804",
    "sam2/sam2_image_predictor.py": "f13e5f9d94e5c8d9d2c3622dab20c8f334c089ef2ee5ea8e199da7d332b029ba",
    "sam2/utils/__init__.py": "34bd8069c54764e7b8d73a78905dbe6467140a2f73170875128f6ca4d8cdd0aa",
    "sam2/utils/misc.py": "01600c01c161cd079d7106fb1d4da845cf91aa31ab2bcecaf8cb151b6d6d20a2",
    "sam2/utils/transforms.py": "ba3a64f4600c62f209206a6df3b40e3fcf133edae32fad658831bb0c2a6d1146",
}
CORE_MODULES = tuple(CORE_MODULE_HASHES)
MATERIALIZED_SYMLINKS = frozenset()
_EXCLUDED_PARTS = frozenset({".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".cache"})
_EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})


class SAM2SourceVerificationError(ValueError):
    """The source release does not prove the exact recovery identity."""


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
    if not value or relative.is_absolute() or ".." in relative.parts or value != relative.as_posix() or value == SOURCE_MANIFEST_NAME:
        raise SAM2SourceVerificationError("SAM2 source manifest contains invalid path")
    return relative


def _assert_real_tree(root: Path, *, allow_manifest: bool, allow_materialized_symlinks: bool = False) -> None:
    if root.is_symlink() or not root.is_dir():
        raise SAM2SourceVerificationError("SAM2 source root must be a real directory, not a symlink")
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            if (
                allow_materialized_symlinks
                and relative.as_posix() in MATERIALIZED_SYMLINKS
                and path.resolve(strict=True).is_file()
            ):
                continue
            raise SAM2SourceVerificationError(f"SAM2 source contains symlink: {relative.as_posix()}")
        if _is_excluded(relative):
            continue
        if path.is_dir() or path.is_file():
            continue
        raise SAM2SourceVerificationError(f"SAM2 source contains non-regular entry: {relative.as_posix()}")
    if not allow_manifest and (root / SOURCE_MANIFEST_NAME).exists():
        raise SAM2SourceVerificationError(f"candidate source must not contain {SOURCE_MANIFEST_NAME}")


def source_file_manifest(
    root: str | Path, *, include_excluded: bool = False, allow_materialized_symlinks: bool = False,
) -> list[dict[str, str]]:
    """Return the deterministic regular-file inventory for a candidate or release."""
    requested = Path(root)
    if requested.is_symlink():
        raise SAM2SourceVerificationError("SAM2 source root must not be a symlink")
    base = requested.resolve(strict=True)
    _assert_real_tree(base, allow_manifest=True, allow_materialized_symlinks=allow_materialized_symlinks)
    manifest: list[dict[str, str]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        if relative.as_posix() == SOURCE_MANIFEST_NAME:
            continue
        if not include_excluded and _is_excluded(relative):
            continue
        manifest.append({"path": relative.as_posix(), "sha256": _sha256_file(path)})
    # pathlib ordering varies around dotted names (e.g. sam2 vs sam2.1); the
    # published JSON manifest is canonical relative-path order on every host.
    manifest.sort(key=lambda row: row["path"])
    if not manifest:
        raise SAM2SourceVerificationError("SAM2 source contains no regular files")
    return manifest


def _assert_manifest_files_are_private(root: Path, manifest: Sequence[Mapping[str, str]]) -> None:
    for row in manifest:
        relative = _manifest_path(str(row["path"]))
        if (root / relative).stat().st_nlink > 1:
            raise SAM2SourceVerificationError(f"SAM2 source manifest file has hard links: {relative.as_posix()}")


def _assert_release_contains_no_bytecode(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(root).suffix in _EXCLUDED_SUFFIXES:
            raise SAM2SourceVerificationError(
                f"SAM2 source release contains executable bytecode: {path.relative_to(root).as_posix()}"
            )


def source_digest_from_manifest(manifest: Sequence[Mapping[str, str]]) -> str:
    rows = [{"path": str(row["path"]), "sha256": str(row["sha256"])} for row in manifest]
    rows.sort(key=lambda row: row["path"])
    if len({row["path"] for row in rows}) != len(rows):
        raise SAM2SourceVerificationError("SAM2 source manifest contains duplicate paths")
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


def core_hashes_from_manifest(manifest: Sequence[Mapping[str, str]]) -> dict[str, str]:
    by_path = {str(row["path"]): str(row["sha256"]) for row in manifest}
    missing = [name for name in CORE_MODULES if name not in by_path]
    if missing:
        raise SAM2SourceVerificationError("SAM2 source is missing required adapter modules: " + ", ".join(missing))
    return {name: by_path[name] for name in CORE_MODULES}


def core_group_digest(core_hashes: Mapping[str, str]) -> str:
    missing = [name for name in CORE_MODULES if name not in core_hashes]
    extras = sorted(set(core_hashes) - set(CORE_MODULES))
    if missing or extras:
        raise SAM2SourceVerificationError("SAM2 core hash set must contain exactly the adapter module closure")
    rows = [{"path": name, "sha256": str(core_hashes[name])} for name in CORE_MODULES]
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


EXPECTED_CORE_GROUP_DIGEST = core_group_digest(CORE_MODULE_HASHES)


@dataclass(frozen=True)
class VerifiedSAM2SourceRelease:
    path: Path
    source_digest: str
    amendment_id: str
    git_head: str
    core_group_digest: str
    manifest: tuple[tuple[str, str], ...]


def _read_manifest(root: Path) -> Mapping[str, Any]:
    path = root / SOURCE_MANIFEST_NAME
    if path.is_symlink() or not path.is_file():
        raise SAM2SourceVerificationError("SAM2 source release manifest is missing or a symlink")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SAM2SourceVerificationError("SAM2 source release manifest is unreadable") from exc
    if not isinstance(raw, Mapping) or raw.get("schema") != SOURCE_SCHEMA:
        raise SAM2SourceVerificationError("invalid SAM2 source release manifest schema")
    return raw


def verify_sam2_source_release(
    release_path: str | Path,
    *,
    expected_digest: str | None = None,
    expected_amendment_id: str | None = RECOVERED_AMENDMENT_ID,
    expected_git_head: str | None = EXACT_GIT_HEAD,
    expected_core_group_digest: str | None = EXPECTED_CORE_GROUP_DIGEST,
) -> VerifiedSAM2SourceRelease:
    """Verify a closed SAM2 release from bytes and asserted provenance values."""
    requested = Path(release_path)
    if requested.is_symlink():
        raise SAM2SourceVerificationError("SAM2 source release root must not be a symlink")
    root = requested.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise SAM2SourceVerificationError("SAM2 source release root must be a real directory")
    raw = _read_manifest(root)
    _assert_real_tree(root, allow_manifest=True)
    declared_raw = raw.get("manifest")
    if not isinstance(declared_raw, list):
        raise SAM2SourceVerificationError("SAM2 source manifest is missing complete file manifest")
    declared: list[dict[str, str]] = []
    for row in declared_raw:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
            raise SAM2SourceVerificationError("SAM2 source manifest contains invalid entry")
        relative = _manifest_path(row["path"])
        if _is_excluded(relative):
            raise SAM2SourceVerificationError("SAM2 source release manifest cannot declare cache or bytecode files")
        declared.append({"path": row["path"], "sha256": row["sha256"]})
    declared.sort(key=lambda row: row["path"])
    _assert_release_contains_no_bytecode(root)
    actual = source_file_manifest(root, include_excluded=True)
    if declared != actual:
        raise SAM2SourceVerificationError("SAM2 source manifest does not match complete bytes on disk")
    _assert_manifest_files_are_private(root, actual)
    digest = source_digest_from_manifest(actual)
    if raw.get("source_digest") != digest or root.name != digest:
        raise SAM2SourceVerificationError("SAM2 source digest does not match directory/content")
    if expected_digest is not None and digest != expected_digest:
        raise SAM2SourceVerificationError("SAM2 source digest differs from planned dependency digest")
    amendment_id = raw.get("amendment_id")
    if not isinstance(amendment_id, str) or not amendment_id:
        raise SAM2SourceVerificationError("SAM2 source amendment id is missing")
    if expected_amendment_id is not None and amendment_id != expected_amendment_id:
        raise SAM2SourceVerificationError("SAM2 source amendment id differs from planned amendment")
    git_head = raw.get("git_head")
    if not isinstance(git_head, str) or not git_head:
        raise SAM2SourceVerificationError("SAM2 source git head is missing")
    if expected_git_head is not None and git_head != expected_git_head:
        raise SAM2SourceVerificationError("SAM2 source git head differs from exact justified source")
    core_hashes = core_hashes_from_manifest(actual)
    if raw.get("core_hashes") != core_hashes:
        raise SAM2SourceVerificationError("SAM2 source core hashes do not match manifest bytes")
    if core_hashes != CORE_MODULE_HASHES:
        raise SAM2SourceVerificationError("SAM2 source adapter module hashes differ from exact justified source")
    group_digest = core_group_digest(core_hashes)
    if raw.get("core_group_digest") != group_digest:
        raise SAM2SourceVerificationError("SAM2 source core group digest does not match core hashes")
    if expected_core_group_digest is not None and group_digest != expected_core_group_digest:
        raise SAM2SourceVerificationError("SAM2 source core group digest differs from exact justified source")
    origin = raw.get("origin_evidence")
    if not isinstance(origin, Mapping) or not origin:
        raise SAM2SourceVerificationError("SAM2 source origin evidence is missing")
    return VerifiedSAM2SourceRelease(
        root, digest, amendment_id, git_head, group_digest,
        tuple((row["path"], row["sha256"]) for row in actual),
    )


def build_sam2_source_release(
    candidate_root: str | Path,
    output_root: str | Path,
    *,
    origin_evidence: Mapping[str, Any],
    git_head: str = EXACT_GIT_HEAD,
    amendment_id: str = RECOVERED_AMENDMENT_ID,
    expected_core_group_digest: str = EXPECTED_CORE_GROUP_DIGEST,
) -> VerifiedSAM2SourceRelease:
    """Copy the exact source transactionally, then publish a read-only digest root."""
    requested = Path(candidate_root)
    if requested.is_symlink():
        raise SAM2SourceVerificationError("SAM2 candidate root must not be a symlink")
    candidate = requested.resolve(strict=True)
    _assert_real_tree(candidate, allow_manifest=False, allow_materialized_symlinks=True)
    initial_manifest = source_file_manifest(candidate, allow_materialized_symlinks=True)
    _assert_manifest_files_are_private(candidate, initial_manifest)
    core_hashes = core_hashes_from_manifest(initial_manifest)
    if core_hashes != CORE_MODULE_HASHES:
        raise SAM2SourceVerificationError("SAM2 candidate adapter module hashes differ from exact justified source")
    group_digest = core_group_digest(core_hashes)
    if group_digest != expected_core_group_digest:
        raise SAM2SourceVerificationError("SAM2 candidate core group digest differs from exact justified source")
    if git_head != EXACT_GIT_HEAD:
        raise SAM2SourceVerificationError("SAM2 candidate git head differs from exact justified source")
    if not isinstance(origin_evidence, Mapping) or not origin_evidence:
        raise SAM2SourceVerificationError("SAM2 source release requires non-empty origin evidence")
    digest = source_digest_from_manifest(initial_manifest)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise SAM2SourceVerificationError("SAM2 source release output root must not be a symlink")
    destination = output / digest
    if destination.exists() or destination.is_symlink():
        return verify_sam2_source_release(
            destination, expected_digest=digest, expected_amendment_id=amendment_id,
            expected_git_head=git_head, expected_core_group_digest=expected_core_group_digest,
        )
    staging = output / f".{digest}.staging-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise SAM2SourceVerificationError(f"SAM2 source staging path already exists: {staging}")
    try:
        for row in initial_manifest:
            source = candidate / row["path"]
            target = staging / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if (
            source_file_manifest(candidate, allow_materialized_symlinks=True) != initial_manifest
            or source_file_manifest(staging) != initial_manifest
        ):
            raise SAM2SourceVerificationError("SAM2 candidate mutated while source release was copied")
        _assert_manifest_files_are_private(staging, initial_manifest)
        attestation = {
            "schema": SOURCE_SCHEMA,
            "source_digest": digest,
            "amendment_id": amendment_id,
            "git_head": git_head,
            "core_hashes": core_hashes,
            "core_group_digest": group_digest,
            "origin_evidence": {**dict(origin_evidence), "candidate_root": str(candidate)},
            "manifest": initial_manifest,
        }
        (staging / SOURCE_MANIFEST_NAME).write_bytes(_canonical_json(attestation) + b"\n")
        for path in sorted(staging.rglob("*"), reverse=True):
            path.chmod(0o444 if path.is_file() else 0o555)
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
    return verify_sam2_source_release(
        destination, expected_digest=digest, expected_amendment_id=amendment_id,
        expected_git_head=git_head, expected_core_group_digest=expected_core_group_digest,
    )
