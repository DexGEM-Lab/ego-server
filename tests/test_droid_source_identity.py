"""CPU-only worker dependency identity tests for recovered DROID cold starts."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ego_annotation.serving.benchmark.release import WorkerRuntimeEvidence
from ego_annotation.serving.contracts import ContractValidationError
from ego_annotation.serving.droid import DroidAdapter, DroidModelConfig, build_droid_model_config
from ego_annotation.serving.droid_source import (
    CORE_MODULES, build_droid_source_release, core_group_digest, core_hashes_from_manifest, source_file_manifest,
)


def _candidate(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    for relative in CORE_MODULES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n")
    return root


def _release(tmp_path: Path):
    candidate = _candidate(tmp_path)
    group = core_group_digest(core_hashes_from_manifest(source_file_manifest(candidate)))
    return build_droid_source_release(
        candidate, tmp_path / "releases", origin_evidence={"candidate": "test-candidate-B"},
        expected_core_group_digest=group,
    )

def test_config_requires_all_source_fields_and_experiment_source_identity() -> None:
    with pytest.raises(ContractValidationError, match="all-or-none"):
        DroidModelConfig(weights="weights", model_revision="droid-v1", droid_source_digest="digest")
    with pytest.raises(ContractValidationError, match="requires a verified immutable source"):
        DroidModelConfig(
            weights="weights", model_revision="droid-v1", experiment_id="exp",
            application_release_path="/release", gcs_address="127.0.0.1:1", http_port=1, temp_dir="/tmp/x",
        )
    # Existing production callers need no recovered-source fields.
    assert build_droid_model_config(weights="weights", model_revision="droid-v1").droid_source_digest is None


def test_fake_amended_cold_start_identity_uses_verified_manifest_not_env_echo(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    release = _release(tmp_path)
    import ego_annotation.serving.droid as droid_module

    module = SimpleNamespace(__file__=str(release.source_root / "droid_net.py"))
    monkeypatch.setitem(sys.modules, "droid_net", module)
    monkeypatch.setattr(droid_module, "_verified_droid_source", lambda _config: release)
    monkeypatch.setattr(droid_module, "_verify_loaded_droid_net", lambda _config: (release, Path(module.__file__)))
    evidence = WorkerRuntimeEvidence(
        release_digest="application-digest", source_sha="app-source", module_root=tmp_path,
        checkpoint_digest="checkpoint", worker_pid=123, cuda_uuid="GPU-7", physical_gpu=7,
    )
    config = DroidModelConfig(
        weights="weights", model_revision="droid-v1", replica_id="droid-exp", assigned_gpu=7,
        experiment_id="exp", application_release_path="/ignored-env-release", gcs_address="127.0.0.1:30000",
        http_port=32000, temp_dir="/tmp/zjheds/exp/gpu7", droid_source_release_path=str(release.path),
        droid_source_digest=release.source_digest, droid_source_amendment_id=release.amendment_id,
    )
    adapter = DroidAdapter(config, backend_factory=lambda _config: object(), runtime_evidence_factory=lambda **_kwargs: evidence)
    identity = adapter.server_identity
    assert identity is not None
    assert identity.dependency_digest == release.source_digest
    assert identity.dependency_root == str(release.source_root)
    assert identity.source_amendment_id == release.amendment_id
