"""CPU-only adversarial tests for recovered immutable DROID source identity."""
from __future__ import annotations

import importlib.util
import py_compile
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ego_annotation.serving.droid_source import (
    CORE_MODULES,
    DroidSourceVerificationError,
    build_droid_source_release,
    core_group_digest,
    core_hashes_from_manifest,
    source_file_manifest,
    verify_droid_source_release,
    verify_imported_droid_module,
)


def _candidate(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    for relative in CORE_MODULES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\nVALUE = {relative!r}\n")
    (root / "droid_slam" / "other.py").write_text("OTHER = 1\n")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("ignored")
    (root / "droid_slam" / "__pycache__").mkdir()
    (root / "droid_slam" / "__pycache__" / "droid_net.pyc").write_bytes(b"ignored")
    return root


def _group(root: Path) -> str:
    manifest = source_file_manifest(root)
    return core_group_digest(core_hashes_from_manifest(manifest))


def _release(tmp_path: Path):
    candidate = _candidate(tmp_path)
    return build_droid_source_release(
        candidate, tmp_path / "releases", origin_evidence={"candidate": "test-candidate-B"},
        expected_core_group_digest=_group(candidate),
    )


def test_builder_publishes_digest_named_full_manifest_non_symlink_bundle(tmp_path: Path) -> None:
    release = _release(tmp_path)
    assert release.path.name == release.source_digest
    assert release.source_root == release.path / "droid_slam"
    assert not release.path.is_symlink()
    assert not (release.path / ".git").exists()
    assert not (release.path / "droid_slam" / "__pycache__").exists()
    verified = verify_droid_source_release(
        release.path, expected_digest=release.source_digest,
        expected_core_group_digest=release.core_group_digest,
    )
    assert verified.manifest == release.manifest
    assert set(dict(verified.manifest)) >= set(CORE_MODULES)


def test_manifest_uses_posix_lexical_order_for_dotfile_directory_prefixes(tmp_path: Path) -> None:
    """Keep builder ordering identical to the verifier's canonical string ordering."""
    candidate = _candidate(tmp_path)
    (candidate / "thirdparty" / "eigen" / ".gitlab-ci.yml").parent.mkdir(parents=True)
    (candidate / "thirdparty" / "eigen" / ".gitlab-ci.yml").write_text("ci")
    (candidate / "thirdparty" / "eigen" / ".gitlab" / "issue.md").parent.mkdir()
    (candidate / "thirdparty" / "eigen" / ".gitlab" / "issue.md").write_text("issue")

    manifest = source_file_manifest(candidate)
    assert [row["path"] for row in manifest] == sorted(row["path"] for row in manifest)
    group = core_group_digest(core_hashes_from_manifest(manifest))
    release = build_droid_source_release(
        candidate, tmp_path / "releases", origin_evidence={"candidate": "ordering-regression"},
        expected_core_group_digest=group,
    )
    assert verify_droid_source_release(release.path, expected_core_group_digest=group).source_digest == release.source_digest


def test_verifier_rejects_mutated_bytes_symlink_missing_core_and_wrong_group(tmp_path: Path) -> None:
    release = _release(tmp_path)
    mutated = release.path / "droid_slam" / "droid_net.py"
    mutated.chmod(0o644)
    mutated.write_text("mutated")
    with pytest.raises(DroidSourceVerificationError, match="manifest does not match"):
        verify_droid_source_release(release.path, expected_core_group_digest=release.core_group_digest)

    clean = _release(tmp_path / "clean")
    module = clean.path / "droid_slam" / "droid_net.py"
    module.parent.chmod(0o755)
    module.chmod(0o644)
    module.unlink()
    module.symlink_to(Path("other.py"))
    with pytest.raises(DroidSourceVerificationError, match="symlink"):
        verify_droid_source_release(clean.path, expected_core_group_digest=clean.core_group_digest)

    candidate = _candidate(tmp_path / "missing")
    (candidate / CORE_MODULES[0]).unlink()
    with pytest.raises(DroidSourceVerificationError, match="missing required core"):
        build_droid_source_release(
            candidate, tmp_path / "missing-out", origin_evidence={"candidate": "missing"},
            expected_core_group_digest="not-the-real-group",
        )

    candidate = _candidate(tmp_path / "wrong")
    with pytest.raises(DroidSourceVerificationError, match="core group digest differs"):
        build_droid_source_release(
            candidate, tmp_path / "wrong-out", origin_evidence={"candidate": "wrong"},
            expected_core_group_digest="0" * 64,
        )


def test_builder_rejects_candidate_mutation_during_transaction(tmp_path: Path, monkeypatch) -> None:
    candidate = _candidate(tmp_path)
    expected_group = _group(candidate)
    import ego_annotation.serving.droid_source as source_module
    original_copy = source_module.shutil.copyfile
    mutated = False

    def copy_then_mutate(source, target):
        nonlocal mutated
        result = original_copy(source, target)
        if not mutated:
            mutated = True
            (candidate / "droid_slam" / "other.py").write_text("changed while copying")
        return result

    monkeypatch.setattr(source_module.shutil, "copyfile", copy_then_mutate)
    with pytest.raises(DroidSourceVerificationError, match="mutated while"):
        build_droid_source_release(
            candidate, tmp_path / "releases", origin_evidence={"candidate": "raced"},
            expected_core_group_digest=expected_group,
        )
    assert not list((tmp_path / "releases").glob("[0-9a-f]" * 64))


def _make_writable(root: Path) -> None:
    for path in (root, *sorted(root.rglob("*"))):
        if path.is_dir():
            path.chmod(0o755)


def test_release_verifier_rejects_injected_unmanifested_bytecode(tmp_path: Path) -> None:
    release = _release(tmp_path)
    _make_writable(release.path)
    cache = release.source_root / "__pycache__"
    cache.mkdir()
    (cache / "droid_net.cpython-test.pyc").write_bytes(b"injected bytecode")
    with pytest.raises(DroidSourceVerificationError, match="executable bytecode"):
        verify_droid_source_release(release.path, expected_core_group_digest=release.core_group_digest)


def test_unchecked_hash_bytecode_cannot_execute_as_droid_net(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    source = candidate / "droid_slam" / "droid_net.py"
    source.write_text("class DroidNet:\n    def forward(self):\n        return 'manifested'\n")
    release = build_droid_source_release(
        candidate, tmp_path / "releases", origin_evidence={"candidate": "unchecked-pyc"},
        expected_core_group_digest=_group(candidate),
    )
    _make_writable(release.path)
    marker = tmp_path / "malicious-bytecode-executed"
    malicious_source = tmp_path / "malicious_droid_net.py"
    malicious_source.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        "class DroidNet:\n    def forward(self):\n        return 'malicious'\n"
    )
    cache_file = Path(importlib.util.cache_from_source(str(release.source_root / "droid_net.py")))
    cache_file.parent.mkdir()
    py_compile.compile(
        str(malicious_source), cfile=str(cache_file), doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    # Adapter startup verifies this closed bundle before it places droid_slam on
    # sys.path, so the unchecked cache is rejected before Python can execute it.
    with pytest.raises(DroidSourceVerificationError, match="executable bytecode"):
        verify_droid_source_release(release.path, expected_core_group_digest=release.core_group_digest)
    assert not marker.exists()


def test_release_verifier_rejects_hardlinked_manifest_source(tmp_path: Path) -> None:
    release = _release(tmp_path)
    _make_writable(release.path)
    source = release.source_root / "droid_net.py"
    outside = tmp_path / "same-bytes.py"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    source.hardlink_to(outside)
    with pytest.raises(DroidSourceVerificationError, match="hard links"):
        verify_droid_source_release(release.path, expected_core_group_digest=release.core_group_digest)


def test_imported_module_requires_exact_core_path_and_manifest_behavior(tmp_path: Path, monkeypatch) -> None:
    candidate = _candidate(tmp_path)
    source = candidate / "droid_slam" / "droid_net.py"
    source.write_text("class DroidNet:\n    def forward(self):\n        return 'manifested'\n")
    release = build_droid_source_release(
        candidate, tmp_path / "releases", origin_evidence={"candidate": "module-binding"},
        expected_core_group_digest=_group(candidate),
    )
    assert verify_imported_droid_module(release.source_root / "droid_net.py", release).name == "droid_net.py"
    with pytest.raises(DroidSourceVerificationError, match="manifested core path"):
        verify_imported_droid_module(release.source_root / "other.py", release)

    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location("honest_droid_net", release.source_root / "droid_net.py")
    assert spec is not None and spec.loader is not None
    honest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(honest)
    assert verify_imported_droid_module(honest.__file__, release, module=honest).name == "droid_net.py"

    malicious = ModuleType("droid_net")
    exec("class DroidNet:\n    def forward(self):\n        return 'malicious'\n", malicious.__dict__)
    malicious.__file__ = str(release.source_root / "droid_net.py")
    with pytest.raises(DroidSourceVerificationError, match="behavior differs"):
        verify_imported_droid_module(malicious.__file__, release, module=malicious)


def test_behavior_verifier_does_not_inherit_its_own_future_flags(tmp_path: Path, monkeypatch) -> None:
    """Compile manifested code in its own flag domain, not droid_source's future imports."""
    candidate = _candidate(tmp_path)
    source = candidate / "droid_slam" / "droid_net.py"
    source.write_text(
        "class DroidNet:\n"
        "    def __init__(self):\n"
        "        self.value = 1\n"
        "    def forward(self, value):\n"
        "        return value + self.value\n"
    )
    release = build_droid_source_release(
        candidate, tmp_path / "releases", origin_evidence={"candidate": "future-flags-regression"},
        expected_core_group_digest=_group(candidate),
    )
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location("candidate_droid_net", release.source_root / "droid_net.py")
    assert spec is not None and spec.loader is not None
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)
    assert verify_imported_droid_module(release.source_root / "droid_net.py", release, module=imported).name == "droid_net.py"

    changed = ModuleType("candidate_droid_net")
    exec("class DroidNet:\n    def __init__(self):\n        self.value = 2\n    def forward(self, value):\n        return value + self.value\n", changed.__dict__)
    changed.__file__ = str(release.source_root / "droid_net.py")
    with pytest.raises(DroidSourceVerificationError, match="behavior differs"):
        verify_imported_droid_module(changed.__file__, release, module=changed)
