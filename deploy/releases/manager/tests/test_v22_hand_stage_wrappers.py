from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_v22_hawor_metric_hand_stage import parse_args as parse_hawor_args
from scripts.run_v22_hawor_metric_hand_stage import run as run_hawor
from scripts.run_v22_hybrid_hand_fusion_stage import parse_args as parse_hybrid_args
from scripts.run_v22_hybrid_hand_fusion_stage import run as run_hybrid


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    clip = run_root / "input" / "clips" / "case.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"fake-video-for-hash")
    write_json(
        run_root / "input" / "input_manifest.json",
        {"case_id": "case", "primary_video": str(clip), "source_fingerprint": {"sha256": "sourcehash"}},
    )
    write_json(
        run_root / "input" / "raw_frame_manifest" / "manifest.json",
        {"frame_count": 2, "frames": [{"frame_idx": 0}, {"frame_idx": 1}]},
    )
    write_json(
        run_root / "state" / "calibration" / "v19_camera_calibration_contract.json",
        {"intrinsics_fx_fy_cx_cy": [800.0, 950.0, 480.0, 270.0], "intrinsics_source": "test_canonical_k"},
    )
    write_json(
        run_root / "state" / "annotations_v22_renderable.json",
        {
            "measurements": {
                "wilor_raw_hands": str(run_root / "measurements" / "hand_candidates" / "wilor_v21" / "wilor_raw_hands.json"),
                "calibration_contract": str(run_root / "state" / "calibration" / "v19_camera_calibration_contract.json"),
            }
        },
    )
    write_json(run_root / "measurements" / "hand_candidates" / "wilor_v21" / "wilor_raw_hands.json", {"frames": []})
    hawor_dir = run_root / "measurements" / "hand_candidates" / "hawor_world"
    hawor_dir.mkdir(parents=True, exist_ok=True)
    (hawor_dir / "hawor_world_hands.npz").write_bytes(b"placeholder")
    write_json(
        run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_shared_geometry.json",
        {"schema": "v22_shared_droid_geometry.v1", "status": "ok", "backend": "droid"},
    )
    return run_root


def test_hawor_stage_dry_run_uses_canonical_calibration_focal_and_clip_hash(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    args = parse_hawor_args([
        "--run-root",
        str(run_root),
        "--repo-root",
        str(tmp_path),
        "--hawor-work-root",
        str(tmp_path / "hawor_work"),
        "--direct-export",
        "--runner-python",
        "/opt/hawor/bin/python",
        "--hawor-root",
        str(tmp_path),
        "--checkpoint",
        str(run_root / "input" / "clips" / "case.mp4"),
        "--infiller-weight",
        str(run_root / "input" / "clips" / "case.mp4"),
        "--model-config",
        str(run_root / "input" / "clips" / "case.mp4"),
        "--dry-run",
    ])
    stage = run_hawor(args)
    assert stage["status"] == "dry_run"
    assert abs(stage["canonical_focal_px"] - (800.0 * 950.0) ** 0.5) < 1e-6
    assert stage["environment"]["EGO_HAWOR_IMG_FOCAL"].startswith("871.779")
    assert stage["environment"]["EGO_HAWOR_CLIP_SHA256"]
    assert stage["environment"]["EGO_HAWOR_CLIP"] == str(run_root / "input" / "clips" / "case.mp4")
    assert stage["command"][0] == "/opt/hawor/bin/python"
    assert "export_hawor_world.py" in stage["command"][1]
    assert "--img_focal" in stage["command"]
    assert "--droid-shared-manifest" in stage["command"]
    assert stage["droid_shared_manifest"].endswith("droid_shared_geometry.json")
    assert stage["model_request_payload"]["droid_shared_manifest"] == stage["droid_shared_manifest"]
    assert "shared_geometry" not in stage["model_request_payload"]
    assert stage["claim_scope"].startswith("D5 HaWoR metric MANO")


def test_hawor_request_manifest_drives_adapter_command(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    manifest = tmp_path / "shared" / "droid_shared_geometry.json"
    write_json(manifest, {"schema": "v22_shared_droid_geometry.v1", "status": "ok", "backend": "droid"})
    args = parse_hawor_args([
        "--run-root", str(run_root), "--repo-root", str(tmp_path), "--direct-export",
        "--runner-python", "/opt/hawor/bin/python", "--hawor-root", str(tmp_path),
        "--checkpoint", str(run_root / "input" / "clips" / "case.mp4"),
        "--infiller-weight", str(run_root / "input" / "clips" / "case.mp4"),
        "--model-config", str(run_root / "input" / "clips" / "case.mp4"),
        "--droid-shared-manifest", str(manifest), "--dry-run",
    ])
    stage = run_hawor(args)
    assert stage["model_request_payload"]["droid_shared_manifest"] == str(manifest.resolve())
    assert str(manifest.resolve()) in stage["command"]


def test_hawor_motion_preparation_dry_run_does_not_require_shared_manifest(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    (run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_shared_geometry.json").unlink()
    args = parse_hawor_args([
        "--run-root", str(run_root), "--repo-root", str(tmp_path), "--direct-export",
        "--runner-python", "/opt/hawor/bin/python", "--hawor-root", str(tmp_path),
        "--checkpoint", str(run_root / "input" / "clips" / "case.mp4"),
        "--infiller-weight", str(run_root / "input" / "clips" / "case.mp4"),
        "--model-config", str(run_root / "input" / "clips" / "case.mp4"),
        "--prepare-motion-only", "--dry-run",
    ])
    stage = run_hawor(args)
    assert stage["status"] == "dry_run"
    assert stage["prepare_only"] is True
    assert "--prepare-only" in stage["command"]
    assert "--droid-shared-manifest" not in stage["command"]
    assert stage["droid_shared_manifest"] is None
    assert stage["model_request"].endswith("hawor_prepare.json")


def test_hawor_stage_fails_closed_without_shared_droid_manifest(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    (run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_shared_geometry.json").unlink()
    args = parse_hawor_args(["--run-root", str(run_root), "--repo-root", str(tmp_path), "--dry-run"])
    with pytest.raises(FileNotFoundError, match="shared DROID manifest"):
        run_hawor(args)


def test_hybrid_stage_dry_run_requires_hawor_wilor_and_canonical_calibration(tmp_path: Path) -> None:
    run_root = make_run_root(tmp_path)
    args = parse_hybrid_args([
        "--run-root",
        str(run_root),
        "--repo-root",
        str(tmp_path),
        "--dry-run",
    ])
    stage = run_hybrid(args)
    assert stage["status"] == "dry_run"
    cmd = stage["command"]
    assert "--calibration-contract" in cmd
    assert str(run_root / "state" / "calibration" / "v19_camera_calibration_contract.json") in cmd
    assert "--allow-heuristic-intrinsics" not in cmd
    assert "--hawor-npz" in cmd
    assert "--wilor-raw" in cmd
    assert stage["translation_policy"] == "hawor_wrist_aligned"
    assert stage["claim_scope"].startswith("D7 candidate hand fusion")
