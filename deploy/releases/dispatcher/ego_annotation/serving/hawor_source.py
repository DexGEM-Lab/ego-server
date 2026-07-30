"""Immutable content-addressed HaWoR source bundles for isolated serving launches.

The bundle is created before importing Torch, Ray, checkpoints, or model code.  It
copies the complete exact-justified HaWoR tree, excluding VCS/cache/bytecode state,
and then verifies every published byte against a manifest.  The serving launch must
set ``EGO_HAWOR_REPO`` to the digest-named release root, never its mutable origin.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SOURCE_MANIFEST_NAME = "HAWOR_SOURCE_MANIFEST.json"
SOURCE_SCHEMA = "ego.hawor-source-release.v1"
RECOVERED_AMENDMENT_ID = "recovered-hawor-core-exact-v1"
EXACT_GIT_HEAD = "7b80b2311fa4ac5621be08f1b63a7089966ccd11"

# These source files are the adapter import closure recorded by the provenance
# report.  ``hawor.py`` loads ``load_hawor`` and ``infiller.py`` loads
# ``TransformerModel``; the remaining entries are the source modules imported by
# those entrypoints/configuration path.  The full tree is still manifested.
CORE_MODULE_HASHES = {
    "scripts/scripts_test_video/hawor_video.py": "d325ab7cb64c3da0ecd931dd90f82c3ba8b158525120b3babeaddf2b8eeb1cb6",
    "hawor/configs/__init__.py": "8af4f8aa408bb2d67a91b0c3e6fddffb39072c22b01762fe72ddb3d275fefd0b",
    "hawor/utils/geometry.py": "4a4df217db764d0d16bd82532f880d8f56cd4182ce3add2d6c3985218c355344",
    "hawor/utils/process.py": "54fbffd70058997490b9f3535181127c2fb3d1ad676e738a23498143a72bf27e",
    "hawor/utils/pylogger.py": "c8a816419a6b61389321f971427e1b88f7fda0345ae49f579318abdcb8349296",
    "hawor/utils/render_openpose.py": "6387d2c82ffe6bfc17adb291e8ed9e2802aa6cda0e584ec9c221d87c38bc19bf",
    "hawor/utils/rotation.py": "b2fde35678f2338ee68b768459240224b44143eb5d1bfad3e2fb868214a60691",
    "infiller/lib/model/network.py": "0bdc234dbc7d0baf0456e30a0901b621aa48ec7171ee40d40d95ad99e48947b5",
    "lib/core/constants.py": "3d25a45df302f5d576b983b9dce1e2b13fb461a41e701d2b9afb39fed989c948",
    "lib/datasets/track_dataset.py": "79ef470ce35177f71d2a5145b39ad148e53771de971a53ea88b872d0aa1b89c6",
    "lib/eval_utils/custom_utils.py": "5c9547303ee64537f0a335f10e6c7ef6f00a072d90496f17e737599b2e7cf7ff",
    "lib/eval_utils/filling_utils.py": "0d5480973163841394dc5ff5e714428fa77c7f9fa41e8f0ebcabc12a7ea8e699",
    "lib/models/backbones/__init__.py": "86810a8ada8017d1b1c7e850f1f772289e56439d796f717d27785d6aa43c9b45",
    "lib/models/backbones/vit.py": "557011238c4759f4017c00182f8235b68d865b2f89cb0760e22321068703b1b5",
    "lib/models/components/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "lib/models/components/pose_transformer.py": "ee8a11a8b19361d6e069fd7103d00c19cf9403268f4f3e4caf6f6a1e8a0feb93",
    "lib/models/components/t_cond_mlp.py": "d8d0c820bb576012ea7ee90de6abe40eebe13357ace19338188a0617b2b21a8e",
    "lib/models/hawor.py": "f375d339399ac93b20ae0dcdbcf52f9921c51eefd19a8657b8d932dbd8877e94",
    "lib/models/mano_wrapper.py": "89c3c0ade660110e2757da7c746b3593de18eed4c58e249d4581cd3314915a44",
    "lib/models/modules.py": "b66c0fd5df0f946760de9422405afe06b58029702b46cb57b69f7e2544c7e32c",
    "lib/pipeline/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "lib/pipeline/tools.py": "62ebd8454ce7b017a0220814a85bb3d0eabf495b3f9eaed33b5087688e458e83",
    "lib/utils/geometry.py": "02ca44763d1ee738306f44f19d467ba197e026be7f2e373ca281948e2a972536",
    "lib/utils/imutils.py": "a3998d65a97a203572e65cf2fb03acc543824b956f58966e8563e93b87c12286",
    "lib/vis/renderer.py": "08c7e281be05084b793d7691b959319c8e32c46eede7973ea5e7fb2664d53016",
    "lib/vis/tools.py": "47b455736ba5aac3d61911379ebfa662550b8cac5d1f6cccda665aaba13ca628",
}
CORE_MODULES = tuple(CORE_MODULE_HASHES)
# The justified checkout carries these two MANO assets as links to checkpoint-root
# files. The release materializes their regular-file bytes so the launch no longer
# follows a mutable external link. No other candidate symlink is accepted.
MATERIALIZED_SYMLINKS = frozenset({
    "_DATA/data_left/mano_left/MANO_LEFT.pkl",
    "_DATA/data/mano/MANO_RIGHT.pkl",
})
_EXCLUDED_PARTS = frozenset({".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".cache"})
_EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})


class HaWoRSourceVerificationError(ValueError):
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
        raise HaWoRSourceVerificationError("HaWoR source manifest contains invalid path")
    return relative


def _assert_real_tree(root: Path, *, allow_manifest: bool, allow_materialized_symlinks: bool = False) -> None:
    if root.is_symlink() or not root.is_dir():
        raise HaWoRSourceVerificationError("HaWoR source root must be a real directory, not a symlink")
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            if (
                allow_materialized_symlinks
                and relative.as_posix() in MATERIALIZED_SYMLINKS
                and path.resolve(strict=True).is_file()
            ):
                continue
            raise HaWoRSourceVerificationError(f"HaWoR source contains symlink: {relative.as_posix()}")
        if _is_excluded(relative):
            continue
        if path.is_dir() or path.is_file():
            continue
        raise HaWoRSourceVerificationError(f"HaWoR source contains non-regular entry: {relative.as_posix()}")
    if not allow_manifest and (root / SOURCE_MANIFEST_NAME).exists():
        raise HaWoRSourceVerificationError(f"candidate source must not contain {SOURCE_MANIFEST_NAME}")


def source_file_manifest(
    root: str | Path, *, include_excluded: bool = False, allow_materialized_symlinks: bool = False,
) -> list[dict[str, str]]:
    """Return the deterministic regular-file inventory for a candidate or release."""
    requested = Path(root)
    if requested.is_symlink():
        raise HaWoRSourceVerificationError("HaWoR source root must not be a symlink")
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
    if not manifest:
        raise HaWoRSourceVerificationError("HaWoR source contains no regular files")
    return manifest


def _assert_manifest_files_are_private(root: Path, manifest: Sequence[Mapping[str, str]]) -> None:
    for row in manifest:
        relative = _manifest_path(str(row["path"]))
        if (root / relative).stat().st_nlink > 1:
            raise HaWoRSourceVerificationError(f"HaWoR source manifest file has hard links: {relative.as_posix()}")


def _assert_release_contains_no_bytecode(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(root).suffix in _EXCLUDED_SUFFIXES:
            raise HaWoRSourceVerificationError(
                f"HaWoR source release contains executable bytecode: {path.relative_to(root).as_posix()}"
            )


def source_digest_from_manifest(manifest: Sequence[Mapping[str, str]]) -> str:
    rows = [{"path": str(row["path"]), "sha256": str(row["sha256"])} for row in manifest]
    rows.sort(key=lambda row: row["path"])
    if len({row["path"] for row in rows}) != len(rows):
        raise HaWoRSourceVerificationError("HaWoR source manifest contains duplicate paths")
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


def core_hashes_from_manifest(manifest: Sequence[Mapping[str, str]]) -> dict[str, str]:
    by_path = {str(row["path"]): str(row["sha256"]) for row in manifest}
    missing = [name for name in CORE_MODULES if name not in by_path]
    if missing:
        raise HaWoRSourceVerificationError("HaWoR source is missing required adapter modules: " + ", ".join(missing))
    return {name: by_path[name] for name in CORE_MODULES}


def core_group_digest(core_hashes: Mapping[str, str]) -> str:
    missing = [name for name in CORE_MODULES if name not in core_hashes]
    extras = sorted(set(core_hashes) - set(CORE_MODULES))
    if missing or extras:
        raise HaWoRSourceVerificationError("HaWoR core hash set must contain exactly the adapter module closure")
    rows = [{"path": name, "sha256": str(core_hashes[name])} for name in CORE_MODULES]
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


EXPECTED_CORE_GROUP_DIGEST = core_group_digest(CORE_MODULE_HASHES)


@dataclass(frozen=True)
class VerifiedHaWoRSourceRelease:
    path: Path
    source_digest: str
    amendment_id: str
    git_head: str
    core_group_digest: str
    manifest: tuple[tuple[str, str], ...]


def _read_manifest(root: Path) -> Mapping[str, Any]:
    path = root / SOURCE_MANIFEST_NAME
    if path.is_symlink() or not path.is_file():
        raise HaWoRSourceVerificationError("HaWoR source release manifest is missing or a symlink")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HaWoRSourceVerificationError("HaWoR source release manifest is unreadable") from exc
    if not isinstance(raw, Mapping) or raw.get("schema") != SOURCE_SCHEMA:
        raise HaWoRSourceVerificationError("invalid HaWoR source release manifest schema")
    return raw


def verify_hawor_source_release(
    release_path: str | Path,
    *,
    expected_digest: str | None = None,
    expected_amendment_id: str | None = RECOVERED_AMENDMENT_ID,
    expected_git_head: str | None = EXACT_GIT_HEAD,
    expected_core_group_digest: str | None = EXPECTED_CORE_GROUP_DIGEST,
) -> VerifiedHaWoRSourceRelease:
    """Verify a closed HaWoR release from bytes and asserted provenance values."""
    requested = Path(release_path)
    if requested.is_symlink():
        raise HaWoRSourceVerificationError("HaWoR source release root must not be a symlink")
    root = requested.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise HaWoRSourceVerificationError("HaWoR source release root must be a real directory")
    raw = _read_manifest(root)
    _assert_real_tree(root, allow_manifest=True)
    declared_raw = raw.get("manifest")
    if not isinstance(declared_raw, list):
        raise HaWoRSourceVerificationError("HaWoR source manifest is missing complete file manifest")
    declared: list[dict[str, str]] = []
    for row in declared_raw:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
            raise HaWoRSourceVerificationError("HaWoR source manifest contains invalid entry")
        relative = _manifest_path(row["path"])
        if _is_excluded(relative):
            raise HaWoRSourceVerificationError("HaWoR source release manifest cannot declare cache or bytecode files")
        declared.append({"path": row["path"], "sha256": row["sha256"]})
    declared.sort(key=lambda row: row["path"])
    _assert_release_contains_no_bytecode(root)
    actual = source_file_manifest(root, include_excluded=True)
    actual.sort(key=lambda row: row["path"])
    if declared != actual:
        raise HaWoRSourceVerificationError("HaWoR source manifest does not match complete bytes on disk")
    _assert_manifest_files_are_private(root, actual)
    digest = source_digest_from_manifest(actual)
    if raw.get("source_digest") != digest or root.name != digest:
        raise HaWoRSourceVerificationError("HaWoR source digest does not match directory/content")
    if expected_digest is not None and digest != expected_digest:
        raise HaWoRSourceVerificationError("HaWoR source digest differs from planned dependency digest")
    amendment_id = raw.get("amendment_id")
    if not isinstance(amendment_id, str) or not amendment_id:
        raise HaWoRSourceVerificationError("HaWoR source amendment id is missing")
    if expected_amendment_id is not None and amendment_id != expected_amendment_id:
        raise HaWoRSourceVerificationError("HaWoR source amendment id differs from planned amendment")
    git_head = raw.get("git_head")
    if not isinstance(git_head, str) or not git_head:
        raise HaWoRSourceVerificationError("HaWoR source git head is missing")
    if expected_git_head is not None and git_head != expected_git_head:
        raise HaWoRSourceVerificationError("HaWoR source git head differs from exact justified source")
    core_hashes = core_hashes_from_manifest(actual)
    if raw.get("core_hashes") != core_hashes:
        raise HaWoRSourceVerificationError("HaWoR source core hashes do not match manifest bytes")
    if core_hashes != CORE_MODULE_HASHES:
        raise HaWoRSourceVerificationError("HaWoR source adapter module hashes differ from exact justified source")
    group_digest = core_group_digest(core_hashes)
    if raw.get("core_group_digest") != group_digest:
        raise HaWoRSourceVerificationError("HaWoR source core group digest does not match core hashes")
    if expected_core_group_digest is not None and group_digest != expected_core_group_digest:
        raise HaWoRSourceVerificationError("HaWoR source core group digest differs from exact justified source")
    origin = raw.get("origin_evidence")
    if not isinstance(origin, Mapping) or not origin:
        raise HaWoRSourceVerificationError("HaWoR source origin evidence is missing")
    return VerifiedHaWoRSourceRelease(
        root, digest, amendment_id, git_head, group_digest,
        tuple((row["path"], row["sha256"]) for row in actual),
    )


def build_hawor_source_release(
    candidate_root: str | Path,
    output_root: str | Path,
    *,
    origin_evidence: Mapping[str, Any],
    git_head: str = EXACT_GIT_HEAD,
    amendment_id: str = RECOVERED_AMENDMENT_ID,
    expected_core_group_digest: str = EXPECTED_CORE_GROUP_DIGEST,
) -> VerifiedHaWoRSourceRelease:
    """Copy the exact source transactionally, then publish a read-only digest root."""
    requested = Path(candidate_root)
    if requested.is_symlink():
        raise HaWoRSourceVerificationError("HaWoR candidate root must not be a symlink")
    candidate = requested.resolve(strict=True)
    _assert_real_tree(candidate, allow_manifest=False, allow_materialized_symlinks=True)
    initial_manifest = source_file_manifest(candidate, allow_materialized_symlinks=True)
    _assert_manifest_files_are_private(candidate, initial_manifest)
    core_hashes = core_hashes_from_manifest(initial_manifest)
    if core_hashes != CORE_MODULE_HASHES:
        raise HaWoRSourceVerificationError("HaWoR candidate adapter module hashes differ from exact justified source")
    group_digest = core_group_digest(core_hashes)
    if group_digest != expected_core_group_digest:
        raise HaWoRSourceVerificationError("HaWoR candidate core group digest differs from exact justified source")
    if git_head != EXACT_GIT_HEAD:
        raise HaWoRSourceVerificationError("HaWoR candidate git head differs from exact justified source")
    if not isinstance(origin_evidence, Mapping) or not origin_evidence:
        raise HaWoRSourceVerificationError("HaWoR source release requires non-empty origin evidence")
    digest = source_digest_from_manifest(initial_manifest)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise HaWoRSourceVerificationError("HaWoR source release output root must not be a symlink")
    destination = output / digest
    if destination.exists() or destination.is_symlink():
        return verify_hawor_source_release(
            destination, expected_digest=digest, expected_amendment_id=amendment_id,
            expected_git_head=git_head, expected_core_group_digest=expected_core_group_digest,
        )
    staging = output / f".{digest}.staging-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise HaWoRSourceVerificationError(f"HaWoR source staging path already exists: {staging}")
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
            raise HaWoRSourceVerificationError("HaWoR candidate mutated while source release was copied")
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
    return verify_hawor_source_release(
        destination, expected_digest=digest, expected_amendment_id=amendment_id,
        expected_git_head=git_head, expected_core_group_digest=expected_core_group_digest,
    )
