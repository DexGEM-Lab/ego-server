from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import run_v22_minimal_annotation_pipeline as pipeline


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_hawor_starts_only_after_shared_droid_manifest(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"video")
    step_order: list[str] = []

    def fake_run_step(name: str, cmd: list[str], **_: object) -> dict:
        step_order.append(name)
        if name == "prepare_single_video":
            clip = run_root / "input" / "clips" / "case.mp4"
            clip.parent.mkdir(parents=True, exist_ok=True)
            clip.write_bytes(input_video.read_bytes())
            write_json(
                run_root / "input" / "input_manifest.json",
                {"case_id": "case", "primary_video": str(clip), "source_fingerprint": {"sha256": "abc"}},
            )
            write_json(
                run_root / "input" / "raw_frame_manifest" / "manifest.json",
                {
                    "frame_count": 2,
                    "video": {"width": 640, "height": 480, "fps": 30.0},
                    "frames": [
                        {"frame_idx": 0, "source_width": 640, "source_height": 480},
                        {"frame_idx": 1, "source_width": 640, "source_height": 480},
                    ],
                },
            )
        elif name == "calibration_contract":
            write_json(
                run_root / "state" / "calibration" / "v19_camera_calibration_contract.json",
                {"intrinsics_fx_fy_cx_cy": [800.0, 800.0, 320.0, 240.0], "intrinsics_source": "test"},
            )
        elif name == "hawor_motion_preparation":
            mask_path = run_root / "prepared_masks.npy"
            mask_path.write_bytes(b"mask")
            write_json(
                run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_motion_preparation.json",
                {
                    "schema": "v22_hawor_motion_preparation.v0",
                    "status": "ok",
                    "timeline": {"frame_count": 2},
                    "artifacts": {"dynamic_mask": {"path": str(mask_path), "sha256": "abc"}},
                },
            )
        elif name == "camera_trajectory_droid":
            assert "--hawor-preparation-report" in cmd
            write_json(
                run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_shared_geometry.json",
                {
                    "schema": "v22_shared_droid_geometry.v1",
                    "status": "ok",
                    "backend": "droid",
                    "droid_invocation": {"instance_count": 1, "terminate_call_count": 1},
                },
            )
        elif name == "hawor_metric_hands":
            manifest = run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_shared_geometry.json"
            assert manifest.is_file()
            assert "--droid-shared-manifest" in cmd
            assert str(manifest) in cmd
        return {"step": name, "status": "ok", "returncode": 0, "elapsed_s": 0.0, "log": "fake"}

    def fake_publish(root: Path) -> dict[str, object]:
        overlay = root / "renders" / "v22_overlay.mp4"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay.write_bytes(b"overlay")
        return {
            "v22_overlay": str(overlay),
            "overlay_source": "test",
            "primary_overlay_report": None,
            "hand_overlay": None,
            "hybrid_hand_overlay": None,
            "world_head_hand_3d": None,
            "semantic_subtitle": None,
        }

    monkeypatch.setattr(pipeline, "run_step", fake_run_step)
    monkeypatch.setattr(pipeline, "publish_overlay", fake_publish)
    monkeypatch.setattr(pipeline, "ffprobe", lambda _: {"status": "ok"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_v22_minimal_annotation_pipeline.py",
            "--case-id",
            "case",
            "--input-video",
            str(input_video),
            "--run-root",
            str(run_root),
            "--repo-root",
            str(tmp_path),
            "--run-camera-trajectory",
            "--camera-backend",
            "droid",
            "--run-hawor-metric-hands",
        ],
    )
    summary = pipeline.run(pipeline.parse_args())
    assert summary["status"] == "ok"
    assert step_order.count("hawor_motion_preparation") == 1
    assert step_order.count("camera_trajectory_droid") == 1
    assert step_order.count("hawor_metric_hands") == 1
    assert step_order.index("hawor_motion_preparation") < step_order.index("camera_trajectory_droid") < step_order.index("hawor_metric_hands")
    assert "mask_preparation_then_one_mask_aware_shared_droid_then_hawor_adapter" in summary["execution_topology"]
