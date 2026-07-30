from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_feishu_ray_hawor_stage import (
    FeishuRayAdapterError,
    _crop_transform,
    estimate_metric_scale_from_depth_disparity,
    hawor_root_cwd,
    replay_mano_parameters,
    run_hawor,
)


def _array_row(name: str, value: np.ndarray) -> dict:
    value = np.ascontiguousarray(value)
    return {"name": name, "data": value.tobytes(), "shape": tuple(value.shape), "dtype": value.dtype.name}


def _report(owner: dict, result: dict, arrays: dict[str, np.ndarray]) -> dict:
    return {"status": "ok", "http_status": 200, "metadata": {"ownership": dict(owner), "result": result}, "arrays": [_array_row(name, value) for name, value in arrays.items()]}


def test_hawor_root_cwd_restores_after_failure(tmp_path: Path) -> None:
    original = Path.cwd()
    with pytest.raises(RuntimeError, match="stop"):
        with hawor_root_cwd(tmp_path):
            assert Path.cwd() == tmp_path
            raise RuntimeError("stop")
    assert Path.cwd() == original


def test_left_crop_transform_maps_source_center_and_mirrors_x() -> None:
    transform = _crop_transform(np.asarray([40.0, 30.0]), 80.0, width=100, height=60, do_flip=True)
    matrix = np.asarray(transform["source_to_model"], dtype=np.float64)
    center = matrix @ np.asarray([40.0, 30.0, 1.0])

    assert center[:2] == pytest.approx([128.0, 128.0])
    assert matrix[0, 0] < 0.0
    assert transform["resize_mode"] == "hawor_crop_flip"
    assert np.asarray(transform["model_to_source"]) @ center == pytest.approx([40.0, 30.0, 1.0])


def test_metric_scale_direction_uses_depth_times_disparity() -> None:
    depth = np.full((2, 4, 6), 4.0, dtype=np.float32)
    frame_idx = np.arange(2, dtype=np.int32)
    keyframes = np.asarray([0, 1], dtype=np.int32)
    disparities = np.full((2, 2, 3), 0.5, dtype=np.float32)
    masks = np.zeros_like(depth)
    scale, report = estimate_metric_scale_from_depth_disparity(depth, frame_idx, keyframes, disparities, masks)
    assert scale == pytest.approx(2.0)
    assert report["direction"] == "metric_scale = median(depth_m * disparity)"


def test_mano_replay_rejects_zero_vertices() -> None:
    def zero_runner(**_: object) -> dict[str, np.ndarray]:
        return {"vertices": np.zeros((1, 778, 3), dtype=np.float32), "joints": np.zeros((1, 21, 3), dtype=np.float32), "faces": np.asarray([[0, 1, 2]], dtype=np.int32)}

    with pytest.raises(FeishuRayAdapterError, match="zero or degenerate"):
        replay_mano_parameters(
            "right",
            np.eye(3, dtype=np.float32)[None],
            np.eye(3, dtype=np.float32)[None, None].repeat(15, axis=1),
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 10), dtype=np.float32),
            Path("/unused"),
            runner=zero_runner,
        )


def _make_run_root(tmp_path: Path, *, right: bool = True) -> Path:
    from PIL import Image

    run_root = tmp_path / "run"
    frame_dir = run_root / "input" / "raw_frame_manifest"
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir(parents=True)
    frames = []
    for index in range(16):
        path = rgb_dir / f"{index:04d}.png"
        Image.fromarray(np.full((6, 8, 3), 64 + index, dtype=np.uint8)).save(path)
        frames.append({"frame_idx": index, "time_s": index / 30.0, "rgb": str(path), "source_width": 8, "source_height": 6})
    frame_dir.mkdir(parents=True)
    (frame_dir / "manifest.json").write_text(json.dumps({"case_id": "case", "fps": 30.0, "frame_count": 16, "video": {"width": 8, "height": 6, "fps": 30.0}, "frames": frames}), encoding="utf-8")
    clip = run_root / "input" / "clips" / "case.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"clip")
    (run_root / "input" / "input_manifest.json").write_text(json.dumps({"case_id": "case", "primary_video": str(clip)}), encoding="utf-8")
    depth_dir = run_root / "measurements" / "depth_candidates" / "unidepth_v2"
    depth_dir.mkdir(parents=True)
    np.savez_compressed(depth_dir / "unidepth_v2_depth.npz", depth=np.full((16, 6, 8), 4.0, dtype=np.float32), confidence=np.ones((16, 6, 8), dtype=np.float32), frame_idx=np.arange(16, dtype=np.int32), intrinsics_fx_fy_cx_cy=np.tile(np.asarray([4.0, 4.0, 4.0, 3.0], dtype=np.float32), (16, 1)), source_size=np.asarray([8, 6], dtype=np.int32))
    droid_dir = run_root / "measurements" / "camera_trajectory" / "droid_full_frame"
    droid_dir.mkdir(parents=True)
    dense = droid_dir / "droid_dense_trajectory.npz"
    T = np.tile(np.eye(4, dtype=np.float32)[None], (16, 1, 1))
    T[:, 0, 3] = np.linspace(0.0, 1.0, 16)
    traj = np.zeros((16, 7), dtype=np.float32)
    traj[:, 0] = T[:, 0, 3]
    traj[:, 6] = 1.0
    np.savez_compressed(dense, frame_idx=np.arange(16, dtype=np.int32), pose_world_camera_xyzw=traj, T_world_camera=T, intrinsics_source=np.asarray([4.0, 4.0, 4.0, 3.0], dtype=np.float32), fps=np.asarray([30.0], dtype=np.float32))
    reconstruction = droid_dir / "droid_keyframe_reconstruction.npz"
    np.savez_compressed(reconstruction, tstamps=np.asarray([0, 15], dtype=np.int32), disps=np.full((2, 2, 3), 0.5, dtype=np.float32))
    for filename, payload in (("droid_keyframes.json", {"keyframes": []}), ("droid_qc.json", {"status": "ok"})):
        (droid_dir / filename).write_text(json.dumps(payload), encoding="utf-8")
    import hashlib

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (droid_dir / "droid_shared_geometry.json").write_text(json.dumps({"schema": "v22_shared_droid_geometry.v1", "status": "ok", "backend": "droid", "processed_frames": 16, "full_source_timeline": True, "droid_invocation": {"instance_count": 1, "track_call_count": 16, "terminate_call_count": 1}, "artifacts": {"dense_trajectory": {"path": str(dense), "sha256": sha(dense)}, "keyframe_reconstruction": {"path": str(reconstruction), "sha256": sha(reconstruction)}, "keyframes": {"path": str(droid_dir / "droid_keyframes.json"), "sha256": sha(droid_dir / "droid_keyframes.json")}, "droid_qc": {"path": str(droid_dir / "droid_qc.json"), "sha256": sha(droid_dir / "droid_qc.json")}}}), encoding="utf-8")
    detector_dir = run_root / "measurements" / "hand_detections" / "feishu_ray_hands"
    detector_dir.mkdir(parents=True)
    mask_path = detector_dir / "masks.npz"
    mask_count = 16 * (2 if right else 1)
    np.savez_compressed(mask_path, masks_packbits=np.zeros((mask_count, 6), dtype=np.uint8), frame_idx=np.repeat(np.arange(16, dtype=np.int32), 2 if right else 1), detection_idx=np.tile(np.arange(2 if right else 1, dtype=np.int32), 16), side=np.tile(np.asarray([0, 1] if right else [0], dtype=np.int32), 16), source_size=np.asarray([8, 6], dtype=np.int32), mask_bit_count=np.asarray([48], dtype=np.int64), packbits_bitorder=np.asarray("little"))
    detector_frames = []
    for index in range(16):
        observations = [{"detection_index": 0, "side": "left", "side_index": 0, "score": 0.9, "visibility": 1.0, "uncertainty": 0.1, "bbox_xyxy_source": [1.0, 1.0, 4.0, 4.0], "mask_archive_index": index, "mask_source_pixel_count": 0}]
        if right:
            observations.append({"detection_index": 1, "side": "right", "side_index": 1, "score": 0.8, "visibility": 1.0, "uncertainty": 0.2, "bbox_xyxy_source": [3.0, 1.0, 7.0, 5.0], "mask_archive_index": index * 2 + 1, "mask_source_pixel_count": 0})
        detector_frames.append({"frame_idx": index, "time_s": index / 30.0, "observations": observations})
    detector_payload = {"schema": "ego.annotation.hands_detector_timeline.v1", "frames": detector_frames, "mask_archive": {"path": str(mask_path), "sha256": sha(mask_path)}}
    (detector_dir / "hands_detector_timeline.json").write_text(json.dumps(detector_payload), encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"schema": "ego.annotation.feishu_ray_services_profile.v1", "profile": "test", "services": {"hawor": {"base_url": "http://127.0.0.1:28003", "routes": ["/hawor.infer_tracks", "/hawor_infiller.fill"]}}}), encoding="utf-8")
    return run_root, profile


def test_fake_feishu_hawor_requests_replay_and_legacy_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root, profile = _make_run_root(tmp_path)

    original_cwd = Path.cwd()

    class FakeDataset:
        def __init__(self, image_paths, boxes, **kwargs: object):
            assert Path.cwd() == tmp_path
            assert kwargs["dilate"] == pytest.approx(1.2)
            self.items = [{"img": np.ones((3, 256, 256), dtype=np.float32), "center": np.asarray([2.0, 2.0], dtype=np.float32), "scale": np.asarray(4.0, dtype=np.float32)} for _ in image_paths]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            assert Path.cwd() == tmp_path
            if index >= len(self.items):
                raise IndexError(index)
            return self.items[index]

    monkeypatch.setattr("scripts.run_feishu_ray_hawor_stage._load_track_dataset", lambda _: FakeDataset)
    calls: list[dict] = []
    infiller_inputs: list[dict] = []

    def fake_caller(**kwargs):
        calls.append(kwargs)
        owner = kwargs["metadata"]["ownership"]
        route = kwargs["route"]
        assert owner["job_id"] == "case"
        if route == "/hawor.infer_tracks":
            assert kwargs["metadata"]["side"] in {"left", "right"}
            assert kwargs["arrays"]["droid_poses"][1] == (16, 4, 4)
            assert kwargs["arrays"]["droid_timestamps"][1] == (16,)
            assert kwargs["arrays"]["crop_batch"][1] == (16, 3, 256, 256)
            pixel_transform = kwargs["metadata"]["crop_transforms"][0]["pixel_transform"]
            assert set(("source_to_model", "model_to_source", "resize_mode")) <= set(pixel_transform)
            assert pixel_transform["resize_mode"].startswith("hawor_crop")
            result = {"ownership": owner, "model_revision": "hawor-v1"}
            return _report(owner, result, {"root_orient": np.tile(np.eye(3, dtype=np.float32)[None], (16, 1, 1)), "hand_pose": np.tile(np.eye(3, dtype=np.float32)[None, None], (16, 15, 1, 1)), "trans": np.zeros((16, 3), dtype=np.float32), "betas": np.zeros((16, 10), dtype=np.float32), "joints": np.zeros((16, 16, 3), dtype=np.float32), "observed": np.ones(16, dtype=np.bool_), "uncertainty": np.full(16, 0.1, dtype=np.float32), "vertices": np.zeros((16, 778, 3), dtype=np.float32)})
        assert route == "/hawor_infiller.fill"
        infiller_inputs.append(kwargs)
        assert len(kwargs["metadata"]["frames"]) == 32
        assert all(float(row["trans"][0]) == pytest.approx(0.0) for row in kwargs["metadata"]["frames"])
        result = {"ownership": owner, "model_revision": "hawor-infiller-v1"}
        return _report(owner, result, {"root_orient": np.tile(np.eye(3, dtype=np.float32)[None, None], (2, 16, 1, 1)), "hand_pose": np.tile(np.eye(3, dtype=np.float32)[None, None, None], (2, 16, 15, 1, 1)), "trans": np.zeros((2, 16, 3), dtype=np.float32), "betas": np.zeros((2, 16, 10), dtype=np.float32), "observed": np.ones((2, 16), dtype=np.bool_), "inferred": np.zeros((2, 16), dtype=np.bool_), "uncertainty": np.full((2, 16), 0.2, dtype=np.float32), "timestamps_s": np.arange(16, dtype=np.float64) / 30.0})

    def mano_runner(**kwargs):
        n = len(kwargs["trans"])
        vertices = np.zeros((n, 778, 3), dtype=np.float32)
        vertices[:, :, 0] = np.linspace(0.0, 0.1, 778, dtype=np.float32)
        joints = np.zeros((n, 21, 3), dtype=np.float32)
        joints[:, :, 0] = np.linspace(0.0, 0.1, 21, dtype=np.float32)
        return {"vertices": vertices, "joints": joints, "faces": np.asarray([[0, 1, 2]], dtype=np.int32)}

    stage = run_hawor(SimpleNamespace(run_root=run_root, repo_root=tmp_path, profile=profile, base_url=None, hawor_root=tmp_path, timeout_s=5.0, job_id="case"), caller=fake_caller, mano_runner=mano_runner)
    assert len(calls) == 3
    archive = np.load(run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_world_hands.npz")
    assert archive["left_vertices_world_m"].shape == (16, 778, 3)
    assert archive["right_joints_world_m"].shape == (16, 21, 3)
    assert archive["left_valid"].sum() == 16
    assert archive["right_valid"].sum() == 16
    assert archive["t_c2w"][15, 0] == pytest.approx(2.0)
    assert archive["left_vertices_world_m"][15, 0, 0] == pytest.approx(2.0)
    assert infiller_inputs
    assert stage["legacy_hawor_droid_executed"] is False
    assert Path.cwd() == original_cwd
    assert json.loads((run_root / "measurements" / "hand_candidates" / "hawor_world" / "qc_hawor_world_hands.json").read_text())["service_vertices_used"] is False


def test_missing_side_skips_infiller_without_fabrication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root, profile = _make_run_root(tmp_path, right=False)

    class FakeDataset:
        def __init__(self, image_paths, boxes, **kwargs: object):
            assert kwargs["dilate"] == pytest.approx(1.2)
            self.items = [{"img": np.ones((3, 256, 256), dtype=np.float32), "center": np.asarray([2.0, 2.0]), "scale": np.asarray(4.0)} for _ in image_paths]
        def __len__(self): return len(self.items)
        def __getitem__(self, index):
            if index >= len(self.items): raise IndexError(index)
            return self.items[index]

    monkeypatch.setattr("scripts.run_feishu_ray_hawor_stage._load_track_dataset", lambda _: FakeDataset)
    routes: list[str] = []

    def fake_caller(**kwargs):
        routes.append(kwargs["route"])
        owner = kwargs["metadata"]["ownership"]
        return _report(owner, {"ownership": owner, "model_revision": "hawor-v1"}, {"root_orient": np.tile(np.eye(3, dtype=np.float32)[None], (16, 1, 1)), "hand_pose": np.tile(np.eye(3, dtype=np.float32)[None, None], (16, 15, 1, 1)), "trans": np.zeros((16, 3), dtype=np.float32), "betas": np.zeros((16, 10), dtype=np.float32), "joints": np.zeros((16, 16, 3), dtype=np.float32), "observed": np.ones(16, dtype=np.bool_), "uncertainty": np.ones(16, dtype=np.float32)})

    def mano_runner(**kwargs):
        n = len(kwargs["trans"])
        vertices = np.zeros((n, 778, 3), dtype=np.float32)
        vertices[:, :, 0] = np.linspace(0.0, 0.1, 778, dtype=np.float32)
        joints = np.zeros((n, 21, 3), dtype=np.float32)
        joints[:, :, 0] = np.linspace(0.0, 0.1, 21, dtype=np.float32)
        return {"vertices": vertices, "joints": joints, "faces": np.asarray([[0, 1, 2]], dtype=np.int32)}

    stage = run_hawor(SimpleNamespace(run_root=run_root, repo_root=tmp_path, profile=profile, base_url=None, hawor_root=tmp_path, timeout_s=5.0, job_id="case"), caller=fake_caller, mano_runner=mano_runner)
    assert routes == ["/hawor.infer_tracks"]
    archive = np.load(run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_world_hands.npz")
    assert archive["left_valid"].sum() == 16
    assert archive["right_valid"].sum() == 0
    assert json.loads((run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_slam_adapter_report.json").read_text())["skipped_infiller_windows"][0]["reason"] == "both_side_anchors_required"


def test_nonfinite_infiller_window_is_preserved_and_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root, profile = _make_run_root(tmp_path)

    class FakeDataset:
        def __init__(self, image_paths, boxes, **kwargs: object):
            assert kwargs["dilate"] == pytest.approx(1.2)
            self.items = [{"img": np.ones((3, 256, 256), dtype=np.float32), "center": np.asarray([2.0, 2.0]), "scale": np.asarray(4.0)} for _ in image_paths]
        def __len__(self): return len(self.items)
        def __getitem__(self, index):
            if index >= len(self.items): raise IndexError(index)
            return self.items[index]

    monkeypatch.setattr("scripts.run_feishu_ray_hawor_stage._load_track_dataset", lambda _: FakeDataset)

    def fake_caller(**kwargs):
        owner = kwargs["metadata"]["ownership"]
        if kwargs["route"] == "/hawor.infer_tracks":
            return _report(owner, {"ownership": owner, "model_revision": "hawor-v1"}, {
                "root_orient": np.tile(np.eye(3, dtype=np.float32)[None], (16, 1, 1)),
                "hand_pose": np.tile(np.eye(3, dtype=np.float32)[None, None], (16, 15, 1, 1)),
                "trans": np.zeros((16, 3), dtype=np.float32),
                "betas": np.zeros((16, 10), dtype=np.float32),
                "joints": np.zeros((16, 16, 3), dtype=np.float32),
                "observed": np.ones(16, dtype=np.bool_),
                "uncertainty": np.full(16, 0.1, dtype=np.float32),
            })
        root_orient = np.tile(np.eye(3, dtype=np.float32)[None, None], (2, 16, 1, 1))
        root_orient[0, 4, 0, 0] = np.nan
        return _report(owner, {"ownership": owner, "model_revision": "hawor-infiller-v1"}, {
            "root_orient": root_orient,
            "hand_pose": np.tile(np.eye(3, dtype=np.float32)[None, None, None], (2, 16, 15, 1, 1)),
            "trans": np.zeros((2, 16, 3), dtype=np.float32),
            "betas": np.zeros((2, 16, 10), dtype=np.float32),
            "observed": np.ones((2, 16), dtype=np.bool_),
            "inferred": np.zeros((2, 16), dtype=np.bool_),
            "uncertainty": np.full((2, 16), 0.2, dtype=np.float32),
            "timestamps_s": np.arange(16, dtype=np.float64) / 30.0,
        })

    def mano_runner(**kwargs):
        n = len(kwargs["trans"])
        vertices = np.zeros((n, 778, 3), dtype=np.float32)
        vertices[:, :, 0] = np.linspace(0.0, 0.1, 778, dtype=np.float32)
        joints = np.zeros((n, 21, 3), dtype=np.float32)
        joints[:, :, 0] = np.linspace(0.0, 0.1, 21, dtype=np.float32)
        return {"vertices": vertices, "joints": joints, "faces": np.asarray([[0, 1, 2]], dtype=np.int32)}

    run_hawor(
        SimpleNamespace(run_root=run_root, repo_root=tmp_path, profile=profile, base_url=None, hawor_root=tmp_path, timeout_s=5.0, job_id="case"),
        caller=fake_caller,
        mano_runner=mano_runner,
    )

    archive = np.load(run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_world_hands.npz")
    assert archive["left_valid"].sum() == 16
    assert archive["right_valid"].sum() == 16
    report = json.loads((run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_slam_adapter_report.json").read_text())
    skipped = report["skipped_infiller_windows"]
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "invalid_service_response"
    assert skipped[0]["error_code"] == "hawor_rotation_invalid"
    failure_path = Path(skipped[0]["failure_path"])
    assert failure_path.is_file()
    failure = json.loads(failure_path.read_text())
    assert failure["status"] == "failed_response_validation"
    assert failure["successful_hawor_artifacts_published"] is False
    assert len(failure["typed_arrays"]) == 8
