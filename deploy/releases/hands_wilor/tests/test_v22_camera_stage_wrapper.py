from __future__ import annotations

import json
from pathlib import Path

from scripts.run_v22_camera_trajectory_stage import parse_args, run


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    clip = run_root / "input" / "clips" / "case.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"fake video")
    write_json(run_root / "input" / "input_manifest.json", {"case_id": "case", "primary_video": str(clip)})
    write_json(
        run_root / "input" / "raw_frame_manifest" / "manifest.json",
        {
            "frame_count": 2,
            "video": {"width": 1920, "height": 1080, "fps": 30.0},
            "frames": [{"frame_idx": 0, "source_width": 1920, "source_height": 1080}, {"frame_idx": 1, "source_width": 1920, "source_height": 1080}],
        },
    )
    write_json(
        run_root / "state" / "calibration" / "v19_camera_calibration_contract.json",
        {"intrinsics_fx_fy_cx_cy": [960.0, 960.0, 960.0, 540.0], "intrinsics_source": "canonical_test_k"},
    )
    return run_root


def test_camera_stage_dry_run_uses_canonical_k_for_droid_focal_scale(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    args = parse_args([
        "--run-root",
        str(run_root),
        "--repo-root",
        str(tmp_path),
        "--droid-root",
        str(tmp_path / "DROID-SLAM"),
        "--droid-weights",
        str(tmp_path / "DROID-SLAM" / "droid.pth"),
        "--runner-python",
        "/opt/droid/bin/python",
        "--dry-run",
    ])
    stage = run(args)
    assert stage["status"] == "dry_run"
    assert stage["focal_scale_from_canonical_k"] == 0.5
    cmd = stage["command"]
    assert cmd[0] == "/opt/droid/bin/python"
    assert "--focal-scale" in cmd
    assert "0.5000000000" in cmd
    assert stage["gauge_declaration"]["scale_status"] == "video_derived_uncertain_without_external_metric_anchor"
    assert stage["claim_scope"].startswith("D4 video-derived camera trajectory")
