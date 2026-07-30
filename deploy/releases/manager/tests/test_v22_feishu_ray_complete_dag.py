from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import run_v22_minimal_annotation_pipeline as pipeline


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_complete_non_cosmos_feishu_ray_dag_uses_only_service_model_commands(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"video")
    steps: list[tuple[str, list[str]]] = []

    def fake_run_step(name: str, cmd: list[str], **_: object) -> dict:
        steps.append((name, cmd))
        if name == "prepare_single_video":
            clip = run_root / "input" / "clips" / "case.mp4"
            clip.parent.mkdir(parents=True, exist_ok=True)
            clip.write_bytes(input_video.read_bytes())
            _write_json(run_root / "input" / "input_manifest.json", {"case_id": "case", "primary_video": str(clip)})
            _write_json(run_root / "input" / "raw_frame_manifest" / "manifest.json", {"case_id": "case", "frame_count": 2, "fps": 30.0, "video": {"width": 8, "height": 6, "fps": 30.0, "frame_count": 2}, "frames": [{"frame_idx": 0, "source_width": 8, "source_height": 6}, {"frame_idx": 1, "source_width": 8, "source_height": 6}]})
        elif name == "calibration_contract":
            _write_json(run_root / "state" / "calibration" / "v19_camera_calibration_contract.json", {"intrinsics_fx_fy_cx_cy": [4.0, 4.0, 4.0, 3.0], "intrinsics_source": "test"})
        elif name == "camera_trajectory_droid":
            _write_json(run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_shared_geometry.json", {"schema": "v22_shared_droid_geometry.v1", "status": "ok", "backend": "droid", "droid_invocation": {"instance_count": 1, "terminate_call_count": 1}})
        return {"step": name, "status": "ok", "returncode": 0, "elapsed_s": 0.0, "log": "fake"}

    def fake_publish(root: Path) -> dict[str, object]:
        overlay = root / "renders" / "v22_overlay.mp4"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay.write_bytes(b"overlay")
        return {"v22_overlay": str(overlay), "overlay_source": "test", "primary_overlay_report": None, "hand_overlay": None, "hybrid_hand_overlay": None, "world_head_hand_3d": None, "semantic_subtitle": None}

    monkeypatch.setattr(pipeline, "run_step", fake_run_step)
    monkeypatch.setattr(pipeline, "publish_overlay", fake_publish)
    monkeypatch.setattr(pipeline, "ffprobe", lambda _: {"status": "ok"})
    monkeypatch.setattr(sys, "argv", ["run_v22_minimal_annotation_pipeline.py", "--case-id", "case", "--input-video", str(input_video), "--run-root", str(run_root), "--repo-root", str(tmp_path), "--model-execution", "feishu_ray", "--camera-backend", "droid", "--run-camera-trajectory", "--run-hawor-metric-hands", "--run-hybrid-hands", "--run-gt-free-drift-self-calibration", "--run-self-consistency-qc", "--run-evaluator", "--write-product-bundle", "--skip-cosmos"])
    summary = pipeline.run(pipeline.parse_args())

    names = [name for name, _ in steps]
    assert names.index("calibration_contract") < names.index("camera_trajectory_droid")
    assert names.index("wilor_hands") < names.index("camera_trajectory_droid")
    assert names.index("camera_trajectory_droid") < names.index("hawor_metric_hands")
    assert names.index("hawor_metric_hands") < names.index("hybrid_hand_fusion")
    assert "gt_free_drift_self_calibration" in names
    assert "self_consistency_qc" in names
    assert "evaluator" in names
    assert "product_annotation_bundle" in names
    commands = {name: cmd for name, cmd in steps}
    assert commands["unidepth"][1] == "scripts/run_feishu_ray_annotation_stage.py"
    assert commands["unidepth"][2] == "unidepth"
    assert commands["wilor_hands"][1] == "scripts/run_feishu_ray_annotation_stage.py"
    assert commands["wilor_hands"][2] == "wilor"
    assert commands["camera_trajectory_droid"][1] == "scripts/run_feishu_ray_annotation_stage.py"
    assert commands["camera_trajectory_droid"][2] == "droid"
    assert commands["hawor_metric_hands"][1] == "scripts/run_feishu_ray_hawor_stage.py"
    flattened = "\n".join(" ".join(cmd) for _, cmd in steps)
    assert "run_droid_full_frame.py" not in flattened
    assert "run_v22_hawor_metric_hand_stage.py" not in flattened
    assert "run_v22_cosmos_captioning_source.py" not in flattened
    assert "run_v22_captioning_stage.py" not in flattened
    assert "render_v22_semantic_subtitle_video.py" not in flattened
    assert summary["enabled_stages"]["captioning"] is False
    assert summary["execution_topology"].startswith("complete_non_cosmos_feishu_ray")
