"""CPU-only adversarial tests for immutable HaWoR source identity."""
from __future__ import annotations

from pathlib import Path

import pytest

import ego_annotation.serving.hawor_source as source


def _candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    root = tmp_path / "candidate"
    hashes: dict[str, str] = {}
    for index, relative in enumerate(source.CORE_MODULES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\nVALUE = {index}\n")
        hashes[relative] = source._sha256_file(path)
    (root / "README.md").write_text("exact candidate fixture")
    assets = tmp_path / "checkpoint-assets"
    assets.mkdir()
    for index, relative in enumerate(source.MATERIALIZED_SYMLINKS):
        target = assets / f"mano-{index}.pkl"
        target.write_bytes(f"mano-{index}".encode())
        link = root / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("excluded")
    cache = root / "hawor" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "module.pyc").write_bytes(b"excluded")
    monkeypatch.setattr(source, "CORE_MODULE_HASHES", hashes)
    return root, source.core_group_digest(hashes)


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    candidate, group = _candidate(tmp_path, monkeypatch)
    return source.build_hawor_source_release(
        candidate, tmp_path / "releases", origin_evidence={"healthy_worker_pythonpath_report": "fixture"},
        expected_core_group_digest=group,
    )


def _writable(root: Path) -> None:
    for path in (root, *sorted(root.rglob("*"))):
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)


def test_builder_publishes_closed_digest_named_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release = _build(tmp_path, monkeypatch)
    assert release.path.name == release.source_digest
    assert not release.path.is_symlink()
    assert not (release.path / ".git").exists()
    assert not (release.path / "hawor" / "__pycache__").exists()
    assert all(not (release.path / relative).is_symlink() for relative in source.MATERIALIZED_SYMLINKS)
    verified = source.verify_hawor_source_release(
        release.path, expected_digest=release.source_digest, expected_core_group_digest=release.core_group_digest,
    )
    assert verified.manifest == release.manifest
    assert set(dict(verified.manifest)) >= set(source.CORE_MODULES)


def test_verifier_rejects_source_mutation_symlink_and_bytecode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release = _build(tmp_path, monkeypatch)
    _writable(release.path)
    changed = release.path / source.CORE_MODULES[0]
    changed.write_text("mutated")
    with pytest.raises(source.HaWoRSourceVerificationError, match="manifest does not match"):
        source.verify_hawor_source_release(release.path, expected_core_group_digest=release.core_group_digest)

    clean = _build(tmp_path / "clean", monkeypatch)
    _writable(clean.path)
    linked = clean.path / source.CORE_MODULES[0]
    linked.unlink()
    linked.symlink_to(Path("README.md"))
    with pytest.raises(source.HaWoRSourceVerificationError, match="symlink"):
        source.verify_hawor_source_release(clean.path, expected_core_group_digest=clean.core_group_digest)

    fresh = _build(tmp_path / "bytecode", monkeypatch)
    _writable(fresh.path)
    cache = fresh.path / "infiller" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "network.pyc").write_bytes(b"injected bytecode")
    with pytest.raises(source.HaWoRSourceVerificationError, match="executable bytecode"):
        source.verify_hawor_source_release(fresh.path, expected_core_group_digest=fresh.core_group_digest)


def test_builder_rejects_candidate_symlink_and_wrong_core_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate, group = _candidate(tmp_path, monkeypatch)
    outside = tmp_path / "outside.py"
    outside.write_text("outside")
    target = candidate / source.CORE_MODULES[0]
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(source.HaWoRSourceVerificationError, match="symlink"):
        source.build_hawor_source_release(
            candidate, tmp_path / "releases", origin_evidence={"healthy_worker_pythonpath_report": "fixture"},
            expected_core_group_digest=group,
        )

    candidate, group = _candidate(tmp_path / "wrong", monkeypatch)
    (candidate / source.CORE_MODULES[1]).write_text("different exact core")
    with pytest.raises(source.HaWoRSourceVerificationError, match="adapter module hashes differ"):
        source.build_hawor_source_release(
            candidate, tmp_path / "wrong-releases", origin_evidence={"healthy_worker_pythonpath_report": "fixture"},
            expected_core_group_digest=group,
        )
