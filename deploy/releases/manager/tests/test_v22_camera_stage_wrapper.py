from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

from scripts.run_v22_camera_trajectory_stage import parse_args, run


def test_camera_stage_vggt_defaults_match_model_preprocessing() -> None:
    args = parse_args(["--run-root", "/tmp/nonexistent", "--camera-backend", "vggt"])

    assert args.vggt_target_size == 518
    assert args.vggt_patch_multiple == 14


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    clip = run_root / "input" / "clips" / "case.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"fake video")
    rgb_dir = run_root / "input" / "raw_frame_manifest" / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    for idx in range(2):
        rgb_path = rgb_dir / f"{idx:06d}.png"
        Image.new("RGB", (64, 48), (20 + idx, 30, 40)).save(rgb_path)
        frame_rows.append({"frame_idx": idx, "source_width": 64, "source_height": 48, "rgb": str(rgb_path)})
    write_json(run_root / "input" / "input_manifest.json", {"case_id": "case", "primary_video": str(clip)})
    write_json(
        run_root / "input" / "raw_frame_manifest" / "manifest.json",
        {
            "frame_count": 2,
            "video": {"width": 64, "height": 48, "fps": 30.0},
            "frames": frame_rows,
        },
    )
    write_json(
        run_root / "state" / "calibration" / "v19_camera_calibration_contract.json",
        {"intrinsics_fx_fy_cx_cy": [32.0, 32.0, 32.0, 24.0], "intrinsics_source": "canonical_test_k"},
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
    assert stage["outputs"]["shared_geometry_manifest"].endswith("droid_shared_geometry.json")
    assert stage["claim_scope"].startswith("D4 video-derived camera trajectory")


def test_camera_stage_droid_consumes_hash_bound_hawor_preparation_report(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    prep_dir = run_root / "measurements" / "hand_candidates" / "hawor_world"
    prep_dir.mkdir(parents=True, exist_ok=True)
    mask_path = prep_dir / "model_masks.npy"
    mask_path.write_bytes(b"mask")
    prep_path = prep_dir / "hawor_motion_preparation.json"
    write_json(
        prep_path,
        {
            "schema": "v22_hawor_motion_preparation.v0",
            "status": "ok",
            "video": {"path": str(run_root / "input" / "clips" / "case.mp4")},
            "timeline": {"frame_count": 2},
            "artifacts": {"dynamic_mask": {"path": str(mask_path), "sha256": "mask-hash"}},
        },
    )
    args = parse_args([
        "--run-root", str(run_root), "--repo-root", str(tmp_path),
        "--droid-root", str(tmp_path / "DROID-SLAM"), "--droid-weights", str(tmp_path / "DROID-SLAM" / "droid.pth"),
        "--runner-python", "/opt/droid/bin/python", "--hawor-preparation-report", str(prep_path), "--dry-run",
    ])
    stage = run(args)
    assert stage["dynamic_mask"]["status"] == "applied_from_hawor_preparation"
    assert "--dynamic-mask-npy" in stage["command"]
    assert str(mask_path.resolve()) in stage["command"]
    assert "--dynamic-mask-sha256" in stage["command"]
    request = json.loads((run_root / "requests" / "droid.json").read_text(encoding="utf-8"))
    assert request["dynamic_mask"]["preparation_report"] == str(prep_path.resolve())


def test_camera_stage_vggt_omega_dry_run_writes_worker_and_compat_requests(tmp_path: Path, monkeypatch) -> None:
    run_root = make_run_root(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")
    monkeypatch.chdir(tmp_path)
    args = parse_args([
        "--run-root",
        str(run_root),
        "--repo-root",
        str(repo_root),
        "--camera-backend",
        "vggt_omega",
        "--vggt-python",
        sys.executable,
        "--vggt-device",
        "cuda",
        "--vggt-target-size",
        "64",
        "--vggt-checkpoint",
        "weights.pt",
        "--dry-run",
    ])
    stage = run(args)

    assert stage["status"] == "dry_run"
    assert stage["camera_backend"] == "vggt_omega"
    assert stage["replacement_for"] == "D4_droid_head_camera_trajectory"
    assert stage["outputs"]["output_dir"].endswith("droid_full_frame")
    assert "run_v22_resident_vggt_camera_batch.py" in " ".join(stage["command"])
    worker_request = json.loads((run_root / "requests" / "vggt_camera_batch.json").read_text(encoding="utf-8"))
    compat_request = json.loads((run_root / "requests" / "droid.json").read_text(encoding="utf-8"))
    assert worker_request["backend"] == "vggt_omega"
    assert worker_request["sequence_length"] == 2
    assert worker_request["items"][0]["output_dir"].endswith("droid_full_frame")
    assert worker_request["checkpoint"] == str(checkpoint.resolve())
    assert str(checkpoint.resolve()) in stage["command"]
    assert "allow_remote_model_download" not in worker_request
    assert "--allow-remote-model-download" not in stage["command"]
    assert compat_request["model"] == "vggt_omega_camera_geometry"
    assert compat_request["parameters"]["batch_contract"]["tensor_shape"] == "[B,S,3,H,W]"


def test_camera_stage_contract_backend_runs_worker_to_droid_compat_outputs(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    args = parse_args([
        "--run-root",
        str(run_root),
        "--repo-root",
        str(Path.cwd()),
        "--camera-backend",
        "contract",
        "--vggt-python",
        sys.executable,
        "--vggt-device",
        "cpu",
        "--vggt-target-size",
        "64",
        "--vggt-patch-multiple",
        "16",
    ])
    stage = run(args)

    assert stage["status"] == "ok"
    assert stage["camera_backend"] == "vggt_camera_contract_backend"
    assert stage["backend"] == "vggt_camera_contract_backend"
    assert stage["batch_tensor_shape"] == [1, 2, 3, 64, 64]
    cam_dir = run_root / "measurements" / "camera_trajectory" / "droid_full_frame"
    assert (cam_dir / "droid_dense_trajectory.npz").exists()
    compat_request = json.loads((run_root / "requests" / "droid.json").read_text(encoding="utf-8"))
    assert compat_request["camera"]["source"] == str(run_root / "state" / "calibration" / "v19_camera_calibration_contract.json")
    assert compat_request["parameters"]["batch_contract"]["item_isolation"] is True
