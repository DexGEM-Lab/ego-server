from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.adapt_droid_to_hawor import adapt_droid_to_hawor, load_shared_geometry


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_shared_geometry(tmp_path: Path, *, frames: int = 4) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    root = tmp_path / "droid"
    root.mkdir(parents=True)
    frame_idx = np.arange(frames, dtype=np.int32)
    traj = np.zeros((frames, 7), dtype=np.float32)
    traj[:, 0] = np.linspace(0.0, 0.3, frames)
    traj[:, 6] = 1.0
    matrices = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
    matrices[:, 0, 3] = -traj[:, 0]
    dense_path = root / "droid_dense_trajectory.npz"
    np.savez_compressed(
        dense_path,
        frame_idx=frame_idx,
        pose_world_camera_xyzw=traj,
        T_world_camera=matrices,
        intrinsics_source=np.asarray([800.0, 800.0, 320.0, 240.0], dtype=np.float32),
        fps=np.asarray([30.0], dtype=np.float32),
    )
    tstamp = np.asarray([0, frames - 1], dtype=np.float32)
    disps = np.full((2, 8, 12), 0.5, dtype=np.float32)
    reconstruction_path = root / "droid_keyframe_reconstruction.npz"
    np.savez(
        reconstruction_path,
        tstamps=tstamp,
        disps=disps,
        depth_level=np.asarray("upsampled"),
    )
    keyframes_path = root / "droid_keyframes.json"
    keyframes_path.write_text(json.dumps({"keyframes": []}), encoding="utf-8")
    qc_path = root / "droid_qc.json"
    qc_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    manifest = {
        "schema": "v22_shared_droid_geometry.v1",
        "status": "ok",
        "backend": "droid",
        "processed_frames": frames,
        "full_source_timeline": True,
        "droid_invocation": {"instance_count": 1, "track_call_count": frames, "terminate_call_count": 1},
        "artifacts": {
            "dense_trajectory": {"path": str(dense_path), "sha256": sha256(dense_path)},
            "keyframe_reconstruction": {"path": str(reconstruction_path), "sha256": sha256(reconstruction_path)},
            "keyframes": {"path": str(keyframes_path), "sha256": sha256(keyframes_path)},
            "droid_qc": {"path": str(qc_path), "sha256": sha256(qc_path)},
        },
    }
    manifest_path = root / "droid_shared_geometry.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, traj, tstamp.astype(np.int32), disps


def test_hawor_export_does_not_import_legacy_droid_slam_path() -> None:
    tree = ast.parse((REPO_ROOT / "scripts" / "export_hawor_world.py").read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "demo" not in imported_modules
    assert not any(module.endswith("hawor_slam") for module in imported_modules)


def test_droid_module_call_contract_remains_single_stream_instance() -> None:
    tree = ast.parse((REPO_ROOT / "scripts" / "run_droid_full_frame.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    constructors = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "Droid"]
    track_calls = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "track"]
    terminate_calls = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "terminate"]
    assert len(constructors) == 1
    assert len(track_calls) == 1
    assert len(terminate_calls) == 1
    assert [keyword.arg for keyword in track_calls[0].keywords] == ["intrinsics", "mask"]
    assert len(terminate_calls[0].args) == 1


def test_dynamic_mask_matches_hawor_masked_droid_contract(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from scripts.run_droid_full_frame import VideoInfo, apply_dynamic_mask, load_dynamic_masks

    masks_path = tmp_path / "model_masks.npy"
    masks = np.zeros((2, 8, 16), dtype=np.uint8)
    masks[:, :, 8:] = 1
    np.save(masks_path, masks)
    resolved, loaded = load_dynamic_masks(
        masks_path,
        VideoInfo(fps=30.0, width=16, height=8, frame_count=2),
    )
    assert resolved == masks_path.resolve()
    image = torch.full((1, 3, 8, 16), 7, dtype=torch.uint8)
    masked, confidence, coverage = apply_dynamic_mask(image, loaded[0])
    assert torch.all(masked[..., :8] == 7)
    assert torch.all(masked[..., 8:] == 0)
    assert confidence.shape == (1, 2)
    assert coverage == pytest.approx(0.5)

    with pytest.raises(RuntimeError, match="full video timeline"):
        load_dynamic_masks(masks_path, VideoInfo(fps=30.0, width=16, height=8, frame_count=3))
    invalid = masks.astype(np.float32)
    invalid[0, 0, 0] = 2.0
    with pytest.raises(RuntimeError, match=r"values must lie in \[0, 1\]"):
        apply_dynamic_mask(image, invalid[0])


def test_shared_geometry_preserves_raw_droid_trajectory_and_keyframe_depth(tmp_path: Path) -> None:
    manifest_path, traj, tstamp, disps = make_shared_geometry(tmp_path)
    geometry = load_shared_geometry(manifest_path, expected_frames=4)
    np.testing.assert_array_equal(geometry["frame_idx"], np.arange(4, dtype=np.int32))
    np.testing.assert_allclose(geometry["traj"], traj)
    np.testing.assert_array_equal(geometry["tstamp"], tstamp)
    np.testing.assert_allclose(geometry["disps"], disps)


def test_adapter_writes_legacy_hawor_slam_contract_without_droid(tmp_path: Path) -> None:
    manifest_path, traj, tstamp, disps = make_shared_geometry(tmp_path)
    masks_path = tmp_path / "model_masks.npy"
    np.save(masks_path, np.zeros((4, 8, 12), dtype=np.uint8))
    output_path = tmp_path / "SLAM" / "hawor_slam_w_scale_0_4.npz"
    report = adapt_droid_to_hawor(
        manifest_path,
        output_path=output_path,
        imgfiles=[],
        masks_path=masks_path,
        img_focal=900.0,
        img_center=(320.0, 240.0),
        hawor_root=tmp_path,
        expected_frames=4,
        scale=1.75,
    )
    blob = np.load(output_path, allow_pickle=True)
    assert {"tstamp", "disps", "traj", "img_focal", "img_center", "scale"}.issubset(blob.files)
    np.testing.assert_array_equal(blob["tstamp"], tstamp)
    np.testing.assert_allclose(blob["disps"], disps)
    np.testing.assert_allclose(blob["traj"], traj)
    assert float(blob["scale"]) == pytest.approx(1.75)
    assert int(blob["droid_invocation_count"][0]) == 1
    assert not bool(blob["legacy_hawor_droid_executed"][0])
    assert report["legacy_hawor_droid_executed"] is False
    assert report["droid_invocation_count"] == 1
    assert Path(report["report_path"]).is_file()


def test_adapter_rejects_mask_different_from_shared_droid_mask(tmp_path: Path) -> None:
    manifest_path, _, _, _ = make_shared_geometry(tmp_path)
    shared_mask = tmp_path / "shared_masks.npy"
    other_mask = tmp_path / "other_masks.npy"
    np.save(shared_mask, np.zeros((4, 8, 12), dtype=np.uint8))
    np.save(other_mask, np.ones((4, 8, 12), dtype=np.uint8))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"]["dynamic_mask"] = {"path": str(shared_mask), "sha256": sha256(shared_mask)}
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match the mask-bound shared DROID manifest"):
        adapt_droid_to_hawor(
            manifest_path,
            output_path=tmp_path / "SLAM" / "hawor_slam_w_scale_0_4.npz",
            imgfiles=[],
            masks_path=other_mask,
            img_focal=900.0,
            img_center=(320.0, 240.0),
            hawor_root=tmp_path,
            expected_frames=4,
            scale=1.0,
        )


def test_shared_geometry_fails_closed_on_hash_or_timeline_mismatch(tmp_path: Path) -> None:
    manifest_path, _, _, _ = make_shared_geometry(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"]["dense_trajectory"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_shared_geometry(manifest_path, expected_frames=4)

    manifest_path, _, _, _ = make_shared_geometry(tmp_path / "second")
    with pytest.raises(RuntimeError, match="frame count"):
        load_shared_geometry(manifest_path, expected_frames=5)
