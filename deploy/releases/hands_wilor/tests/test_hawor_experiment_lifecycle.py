"""CPU-only contract tests for the isolated GPU4 HaWoR envelope experiment."""
from __future__ import annotations

from pathlib import Path

from ego_annotation.serving.benchmark.hawor_experiment import HAWOR_EXPERIMENT_TEMP_ROOT, build_hawor_experiment_plan
from ego_annotation.serving.benchmark.release import WorkerRuntimeEvidence, build_release
from ego_annotation.serving.hawor import HaWoRAdapter, build_hawor_model_config


SOURCE_SHA = "a" * 40


def _release(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "ego_annotation" / "serving").mkdir(parents=True)
    (source / "ego_annotation" / "__init__.py").write_text("")
    (source / "ego_annotation" / "serving" / "__init__.py").write_text("")
    # The generic release gate requires this normal application module as well.
    (source / "ego_annotation" / "serving" / "deployment.py").write_text("app = object()\n")
    return build_release(source, tmp_path / "releases", source_sha=SOURCE_SHA)


def test_hawor_plan_has_short_disjoint_port_temp_and_dual_identity_contract(tmp_path):
    release = _release(tmp_path)
    plan = build_hawor_experiment_plan(
        experiment_id="h3e1", gpu_id=5, run_root=tmp_path / "runs", application_release_path=release,
        source_sha=SOURCE_SHA, hawor_checkpoint_digest="hawor-bytes", infiller_checkpoint_digest="infiller-bytes",
        gpu_uuid="GPU-5", wire_format="envelope",
    )
    plan.assert_isolated()
    replica = plan.replicas[0]
    assert Path(replica.lifecycle.temp_dir).is_relative_to(HAWOR_EXPERIMENT_TEMP_ROOT)
    assert all(port < 32768 for port in replica.lifecycle.ports.all_ports())
    assert replica.track_endpoint.url.endswith("/hawor.infer_tracks")
    assert replica.infiller_endpoint.url.endswith("/hawor_infiller.fill")
    assert replica.track_identity.replica_id != replica.infiller_identity.replica_id
    assert replica.track_identity.checkpoint_digest == "hawor-bytes"
    assert replica.infiller_identity.checkpoint_digest == "infiller-bytes"
    assert replica.track_runtime_config["runtime_config"]["wire_format"] == "envelope"
    assert "--app-choice hawor" in replica.launch_commands()[1]
    assert "hawor_experiment --scoped-stop" in replica.stop_commands()[1]


def test_hawor_worker_identity_is_derived_from_loaded_evidence_not_caller_fields(tmp_path):
    release = _release(tmp_path)
    checkpoint = tmp_path / "hawor.ckpt"
    checkpoint.write_bytes(b"real-checkpoint-bytes")
    config = build_hawor_model_config(
        checkpoint=str(checkpoint), model_revision="hawor-v1", device="cpu", replica_id="worker-replica",
        assigned_gpu=4, experiment_id="h3e1", application_release_sha="caller-label", checkpoint_digest="caller-label",
        application_release_path=str(release), gcs_address="127.0.0.1:30000", http_port=32000,
        temp_dir="/tmp/ehw/h3e1/gpu4",
    )
    evidence = WorkerRuntimeEvidence(
        release_digest="release-bytes", source_sha="source-bytes", module_root=release,
        checkpoint_digest="checkpoint-bytes", worker_pid=4321, cuda_uuid="GPU-derived", physical_gpu=4,
    )

    class Backend:
        def infer_tracks(self, crop_batch, crop_geometry, img_center, do_flip):
            return {}

    adapter = HaWoRAdapter(config, backend_factory=lambda _: Backend(), runtime_evidence_factory=lambda **_: evidence)
    identity = adapter.server_identity
    assert identity is not None
    assert identity.release_sha == "source-bytes"
    assert identity.release_digest == "release-bytes"
    assert identity.checkpoint_digest == "checkpoint-bytes"
    assert identity.worker_pid == 4321 and identity.cuda_uuid == "GPU-derived"


def test_experiment_driver_allowlists_hawor_pair():
    from ego_annotation.serving.benchmark.experiment_driver import EXPERIMENT_APPLICATIONS

    assert EXPERIMENT_APPLICATIONS["hawor"] == "ego_annotation.serving.hawor_deployment:app"
