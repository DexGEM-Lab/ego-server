"""CPU-only lifecycle/identity tests for an isolated Hands + WiLoR soak."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ego_annotation.serving.benchmark.experiment_driver import EXPERIMENT_APPLICATIONS
from ego_annotation.serving.benchmark.hands_experiment import HANDS_EXPERIMENT_TEMP_ROOT, build_hands_experiment_plan, validate_hands_typed_readiness
from ego_annotation.serving.benchmark.release import WorkerRuntimeEvidence, build_release
from ego_annotation.serving import sam2_source
from ego_annotation.serving.hands import HandsAdapter, build_hands_model_config
from ego_annotation.serving.wilor import WiLoRAdapter, build_wilor_model_config

SOURCE_SHA = "a" * 40


def _release(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "ego_annotation" / "serving").mkdir(parents=True)
    (source / "ego_annotation" / "__init__.py").write_text("")
    (source / "ego_annotation" / "serving" / "__init__.py").write_text("")
    (source / "ego_annotation" / "serving" / "deployment.py").write_text("app = object()\n")
    return build_release(source, tmp_path / "releases", source_sha=SOURCE_SHA)


def _evidence(release: Path, checkpoint_digest: str) -> WorkerRuntimeEvidence:
    return WorkerRuntimeEvidence("release-bytes", "source-bytes", release, checkpoint_digest, 4321, "GPU-derived", 5)


def _sam2_release(tmp_path: Path, monkeypatch) -> Path:
    candidate = tmp_path / "sam2-candidate"
    hashes = {}
    for index, relative in enumerate(sam2_source.CORE_MODULES):
        path = candidate / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\\nVALUE = {index}\\n")
        hashes[relative] = sam2_source._sha256_file(path)
    (candidate / "README.md").write_text("fixture")
    monkeypatch.setattr(sam2_source, "CORE_MODULE_HASHES", hashes)
    monkeypatch.setattr(sam2_source, "CORE_MODULES", tuple(hashes))
    release = sam2_source.build_sam2_source_release(
        candidate, tmp_path / "sam2-releases", origin_evidence={"provenance_report_sha256": "fixture"},
        expected_core_group_digest=sam2_source.core_group_digest(hashes),
    )
    return release.path


def test_hands_plan_owns_one_gpu5_head_short_temp_disjoint_safe_ports_and_pair_identity(tmp_path: Path, monkeypatch):
    release = _release(tmp_path)
    sam2_release = _sam2_release(tmp_path, monkeypatch)
    plan = build_hands_experiment_plan(
        experiment_id="h5e1", gpu_id=5, run_root=tmp_path / "runs", application_release_path=release,
        sam2_source_release_path=sam2_release, source_sha=SOURCE_SHA,
        detector_checkpoint_digest="detector-bytes", sam2_checkpoint_digest="sam2-bytes",
        wilor_checkpoint_digest="wilor-bytes", gpu_uuid="GPU-5", wire_format="envelope",
    )
    plan.assert_isolated()
    replica = plan.replicas[0]
    assert Path(replica.lifecycle.temp_dir).is_relative_to(HANDS_EXPERIMENT_TEMP_ROOT)
    assert all(port < 32768 for port in replica.lifecycle.ports.all_ports())
    assert str(replica.lifecycle.ports.metrics_export_port) not in replica.lifecycle.ports.worker_port_list.split(",")
    assert replica.hands_endpoint.url.endswith("/hands.detect")
    assert replica.wilor_endpoint.url.endswith("/wilor.reconstruct")
    assert replica.hands_identity.replica_id != replica.wilor_identity.replica_id
    assert replica.hands_identity.checkpoint_digest == "detector-bytes"
    assert replica.wilor_identity.checkpoint_digest == "wilor-bytes"
    assert dict(replica.lifecycle.env_vars)["EGO_SAM2_REPO"] == str(sam2_release)
    assert str(sam2_release) in dict(replica.lifecycle.env_vars)["PYTHONPATH"].split(":")
    assert replica.hands_runtime_config["wire_format"] == "envelope"
    assert replica.wilor_runtime_config["wire_format"] == "envelope"
    assert "--app-choice hands" in replica.launch_commands()[1]
    assert "hands_experiment --scoped-stop" in replica.stop_commands()[1]


def test_hands_and_wilor_worker_identity_comes_from_loaded_worker_evidence(tmp_path: Path):
    release = _release(tmp_path)
    detector = tmp_path / "detector.pt"; detector.write_bytes(b"detector")
    wilor = tmp_path / "wilor.ckpt"; wilor.write_bytes(b"wilor")
    common = dict(experiment_id="h5e1", application_release_path=str(release), gcs_address="127.0.0.1:30400", http_port=32200, temp_dir="/tmp/ehn/h5e1/gpu5", assigned_gpu=5)

    class HandsBackend:
        def detect(self, _images): return []
        def mask(self, _image, _boxes): return None

    hands = HandsAdapter(build_hands_model_config(detector_checkpoint=str(detector), sam2_checkpoint="sam2", sam2_config="cfg", model_revision="hands-v1", replica_id="hands-worker", **common), backend_factory=lambda _: HandsBackend(), runtime_evidence_factory=lambda **_: _evidence(release, "derived-detector"))
    wilor_adapter = WiLoRAdapter(build_wilor_model_config(checkpoint=str(wilor), config_path="cfg", model_revision="wilor-v1", replica_id="wilor-worker", **common), backend_factory=lambda _: object(), runtime_evidence_factory=lambda **_: _evidence(release, "derived-wilor"))
    assert hands.server_identity is not None and wilor_adapter.server_identity is not None
    assert hands.server_identity.worker_pid == 4321 and hands.server_identity.cuda_uuid == "GPU-derived"
    assert hands.server_identity.checkpoint_digest == "derived-detector"
    assert wilor_adapter.server_identity.checkpoint_digest == "derived-wilor"
    assert hands.server_identity.release_sha == wilor_adapter.server_identity.release_sha == "source-bytes"


def test_typed_readiness_requires_worker_identity_trace_and_runtime_contract(tmp_path: Path, monkeypatch):
    release = _release(tmp_path)
    sam2_release = _sam2_release(tmp_path, monkeypatch)
    plan = build_hands_experiment_plan(
        experiment_id="h5e1", gpu_id=5, run_root=tmp_path / "runs", application_release_path=release,
        sam2_source_release_path=sam2_release, source_sha=SOURCE_SHA,
        detector_checkpoint_digest="detector-bytes", sam2_checkpoint_digest="sam2-bytes", wilor_checkpoint_digest="wilor-bytes",
    )
    replica = plan.replicas[0]
    actual = replica.hands_identity.__class__(**{**replica.hands_identity.__dict__, "worker_pid": 4321})
    result = SimpleNamespace(server_identity=actual, trace=SimpleNamespace(replica_id=actual.replica_id), batch_diagnostics={"runtime_config": replica.hands_runtime_config})
    assert validate_hands_typed_readiness(replica.hands_identity, replica.hands_runtime_config, result) == actual
    result.batch_diagnostics = {"runtime_config": {"wire_format": "multipart"}}
    try:
        validate_hands_typed_readiness(replica.hands_identity, replica.hands_runtime_config, result)
    except ValueError as exc:
        assert "runtime configuration" in str(exc)
    else:
        raise AssertionError("readiness must reject a mismatched wire runtime")


def test_detached_driver_allowlists_hands_deployment():
    assert EXPERIMENT_APPLICATIONS["hands"] == "ego_annotation.serving.hands_deployment:hands_app"
