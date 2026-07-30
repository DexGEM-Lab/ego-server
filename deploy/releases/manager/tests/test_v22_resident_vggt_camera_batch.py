from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.adapt_v22_minimal_run_to_annotation_bundle import camera_trajectory_rows
from scripts.run_v22_resident_vggt_camera_batch import VggtCameraBackend, backend_from_request, frame_rgb_path, iter_sequence_rows, parse_args, run


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_item(tmp_path: Path, item_id: str, frame_count: int, intensity: int, calibration_intrinsics: list[float] | None = None) -> Path:
    run_root = tmp_path / item_id
    clip = run_root / "input" / "clips" / f"{item_id}.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"video")
    frames = []
    for idx in range(frame_count):
        rgb = run_root / "input" / "raw_frame_manifest" / "rgb" / f"{idx:06d}.png"
        rgb.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 48), (intensity + idx, intensity, intensity)).save(rgb)
        frames.append(
            {
                "frame_idx": idx,
                "time_s": idx / 30.0,
                "rgb": str(rgb),
                "source_width": 80,
                "source_height": 48,
            }
        )
    write_json(run_root / "input" / "input_manifest.json", {"case_id": item_id, "primary_video": str(clip)})
    write_json(
        run_root / "input" / "raw_frame_manifest" / "manifest.json",
        {
            "schema": "v22_raw_frame_manifest.v0",
            "frame_count": frame_count,
            "fps": 30.0,
            "video": {"width": 80, "height": 48, "fps": 30.0, "frame_count": frame_count},
            "frames": frames,
        },
    )
    write_json(
        run_root / "state" / "calibration" / "v19_camera_calibration_contract.json",
        {"intrinsics_fx_fy_cx_cy": calibration_intrinsics or [64.0, 64.0, 40.0, 24.0], "intrinsics_source": f"unit_test_{item_id}"},
    )
    return run_root


def test_worker_defaults_match_vggt_preprocessing_contract(tmp_path: Path) -> None:
    args = parse_args(["--request", str(tmp_path / "request.json")])

    assert args.target_size == 518
    assert args.patch_multiple == 14


def test_iter_sequence_rows_rejects_mixed_lengths(tmp_path: Path) -> None:
    run_a = make_item(tmp_path, "item_a", 3, 10)
    run_b = make_item(tmp_path, "item_b", 3, 40)
    request = {
        "job_id": "job_vggt",
        "sequence_length": 3,
        "items": [
            {"item_id": "item_a", "run_root": str(run_a)},
            {"item_id": "item_b", "run_root": str(run_b), "sequence_length": 2},
        ],
    }
    try:
        iter_sequence_rows(request)
    except RuntimeError as exc:
        assert "mixed sequence lengths" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mixed sequence lengths must be rejected")


def test_vggt_backend_import_failure_names_checkout_or_runtime_install(tmp_path: Path) -> None:
    backend = VggtCameraBackend(
        variant="vggt",
        repo_root=tmp_path,
        checkpoint=None,
        model_id="facebook/VGGT-1B",
        model_file="model.pt",
        device="cpu",
    )

    real_import = __import__

    def fake_import(name: str, *args, **kwargs):
        if name.startswith("vggt"):
            raise ModuleNotFoundError("no module named vggt")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        try:
            backend._load(torch=object())
        except RuntimeError as exc:
            assert "provide third_party/vggt or install vggt" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("missing VGGT runtime should fail explicitly")


def test_relative_manifests_and_frames_resolve_under_each_item_run_root(tmp_path: Path, monkeypatch) -> None:
    run_a = make_item(tmp_path, "item_a", 3, 10)
    run_b = make_item(tmp_path, "item_b", 3, 40)
    for run_root in (run_a, run_b):
        manifest_path = run_root / "input" / "raw_frame_manifest" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for frame in manifest["frames"]:
            frame["rgb"] = f"input/raw_frame_manifest/rgb/{int(frame['frame_idx']):06d}.png"
        write_json(manifest_path, manifest)
    monkeypatch.chdir(run_a)
    request = {
        "job_id": "job_vggt",
        "sequence_length": 3,
        "items": [
            {"item_id": "item_a", "run_root": str(run_a), "raw_frame_manifest": "input/raw_frame_manifest/manifest.json"},
            {"item_id": "item_b", "run_root": str(run_b), "raw_frame_manifest": "input/raw_frame_manifest/manifest.json"},
        ],
    }

    rows = iter_sequence_rows(request)

    assert rows[0].raw_frame_manifest == (run_a / "input" / "raw_frame_manifest" / "manifest.json").resolve()
    assert rows[1].raw_frame_manifest == (run_b / "input" / "raw_frame_manifest" / "manifest.json").resolve()
    assert frame_rgb_path(rows[0].frames[0], rows[0].run_root) == (run_a / "input" / "raw_frame_manifest" / "rgb" / "000000.png").resolve()
    assert frame_rgb_path(rows[1].frames[0], rows[1].run_root) == (run_b / "input" / "raw_frame_manifest" / "rgb" / "000000.png").resolve()


def test_vggt_backend_requires_checkpoint_without_explicit_download(tmp_path: Path) -> None:
    backend = VggtCameraBackend(
        variant="vggt",
        repo_root=tmp_path,
        checkpoint=None,
        model_id="facebook/VGGT-1B",
        model_file="model.pt",
        device="cpu",
    )
    model_mod = types.ModuleType("vggt.models.vggt")
    model_mod.VGGT = object
    pose_mod = types.ModuleType("vggt.utils.pose_enc")
    pose_mod.pose_encoding_to_extri_intri = object()
    fake_modules = {
        "vggt": types.ModuleType("vggt"),
        "vggt.models": types.ModuleType("vggt.models"),
        "vggt.models.vggt": model_mod,
        "vggt.utils": types.ModuleType("vggt.utils"),
        "vggt.utils.pose_enc": pose_mod,
    }
    with patch.dict(sys.modules, fake_modules):
        try:
            backend._load(torch=object())
        except RuntimeError as exc:
            assert "requires --checkpoint unless --allow-remote-model-download" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("vggt backend must not download weights implicitly")


def test_vggt_backend_passes_target_size_to_constructor_when_supported(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")

    class FakeVGGT:
        last_img_size = None

        def __init__(self, img_size: int) -> None:
            type(self).last_img_size = img_size

        def load_state_dict(self, state) -> None:
            self.state = state

        def to(self, device: str):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    class FakeTorch:
        @staticmethod
        def load(path: str, map_location: str = "cpu"):
            return {"weight": path, "map_location": map_location}

    model_mod = types.ModuleType("vggt.models.vggt")
    model_mod.VGGT = FakeVGGT
    pose_mod = types.ModuleType("vggt.utils.pose_enc")
    pose_mod.pose_encoding_to_extri_intri = object()
    fake_modules = {
        "vggt": types.ModuleType("vggt"),
        "vggt.models": types.ModuleType("vggt.models"),
        "vggt.models.vggt": model_mod,
        "vggt.utils": types.ModuleType("vggt.utils"),
        "vggt.utils.pose_enc": pose_mod,
    }
    backend = VggtCameraBackend(
        variant="vggt",
        repo_root=tmp_path,
        checkpoint=checkpoint,
        model_id="facebook/VGGT-1B",
        model_file="model.pt",
        device="cpu",
        target_size=518,
    )

    with patch.dict(sys.modules, fake_modules):
        backend._load(torch=FakeTorch)

    assert FakeVGGT.last_img_size == 518


def test_worker_request_download_permission_requires_json_boolean(tmp_path: Path) -> None:
    args = parse_args(["--request", str(tmp_path / "request.json"), "--backend", "vggt"])
    request = {"backend": "vggt", "allow_remote_model_download": "false"}
    try:
        backend_from_request(args, request)
    except RuntimeError as exc:
        assert "allow_remote_model_download must be a JSON boolean" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("string download permission must not be truthy")


def test_vggt_checkpoint_model_version_identifies_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pt"
    payload = b"unit-test-weights"
    checkpoint.write_bytes(payload)
    backend = VggtCameraBackend(
        variant="vggt",
        repo_root=tmp_path,
        checkpoint=checkpoint,
        model_id="facebook/VGGT-1B",
        model_file="model.pt",
        device="cpu",
    )

    assert str(checkpoint.resolve()) in backend.model_version
    assert f"size_bytes={len(payload)}" in backend.model_version
    assert f"sha256={hashlib.sha256(payload).hexdigest()}" in backend.model_version
    assert "facebook/VGGT-1B" not in backend.model_version


def test_contract_backend_writes_droid_compatible_per_item_outputs(tmp_path: Path) -> None:
    run_a = make_item(tmp_path, "item_a", 3, 10, [64.0, 64.0, 40.0, 24.0])
    run_b = make_item(tmp_path, "item_b", 3, 40, [72.0, 73.0, 41.0, 25.0])
    request_path = tmp_path / "request.json"
    output_root = tmp_path / "reports"
    request = {
        "job_id": "job_vggt",
        "backend": "contract",
        "worker_id": "worker_vggt_contract",
        "stage_id": "vggt_omega_camera_geometry_resident",
        "batch_size": 2,
        "sequence_length": 3,
        "target_size": 64,
        "patch_multiple": 16,
        "device": "cpu",
        "output_root": str(output_root),
        "compat_request_name": "droid",
        "items": [
            {"item_id": "item_a", "run_root": str(run_a), "output_dir": str(run_a / "measurements" / "camera_trajectory" / "droid_full_frame")},
            {"item_id": "item_b", "run_root": str(run_b), "output_dir": str(run_b / "measurements" / "camera_trajectory" / "droid_full_frame")},
        ],
    }
    write_json(request_path, request)

    report = run(parse_args(["--request", str(request_path), "--device", "cpu"]))

    assert report["model_load_count"] == 1
    assert report["batch_inference_count"] == 1
    assert report["batch_tensor_shapes"] == [[2, 3, 3, 64, 64]]
    assert len(report["items"]) == 2

    expected_intrinsics = {"item_a": [64.0, 64.0, 40.0, 24.0], "item_b": [72.0, 73.0, 41.0, 25.0]}
    for run_root, item_id in ((run_a, "item_a"), (run_b, "item_b")):
        cam_dir = run_root / "measurements" / "camera_trajectory" / "droid_full_frame"
        stage = json.loads((cam_dir / "v22_camera_trajectory_stage.json").read_text(encoding="utf-8"))
        dense = json.loads((cam_dir / "droid_dense_trajectory.json").read_text(encoding="utf-8"))
        qc = json.loads((cam_dir / "droid_qc.json").read_text(encoding="utf-8"))
        request_payload = json.loads((run_root / "requests" / "droid.json").read_text(encoding="utf-8"))
        blob = np.load(cam_dir / "droid_dense_trajectory.npz")

        assert stage["status"] == "ok"
        assert stage["replacement_for"] == "D4_droid_head_camera_trajectory"
        assert stage["batch_tensor_shape"] == [2, 3, 3, 64, 64]
        assert stage["item_id"] == item_id
        assert stage["camera_backend"] == "vggt_camera_contract_backend"
        assert stage["calibration_contract"] == str(run_root / "state" / "calibration" / "v19_camera_calibration_contract.json")
        assert qc["processed_frames"] == 3
        assert qc["batch_tensor_shape"] == [2, 3, 3, 64, 64]
        assert len(dense["frames"]) == 3
        assert blob["T_world_camera"].shape == (3, 4, 4)
        assert blob["intrinsics_source"].shape == (4,)
        assert request_payload["stage"] == "D4_camera_trajectory"
        assert request_payload["camera"]["source"] == str(run_root / "state" / "calibration" / "v19_camera_calibration_contract.json")
        assert request_payload["camera"]["intrinsics_px"] == expected_intrinsics[item_id]
        assert request_payload["parameters"]["batch_contract"]["tensor_shape"] == "[B,S,3,H,W]"
        assert request_payload["parameters"]["batch_contract"]["padding_status"] == "not_supported_without_explicit_model_attention_mask"
        assert request_payload["parameters"]["output_contract"]["writes_droid_compatible_camera_artifacts"] is True

        adapter_rows, adapter_stage = camera_trajectory_rows(run_root)
        assert adapter_stage is not None
        assert adapter_stage["backend"] == "vggt_camera_contract_backend"
        assert len(adapter_rows) == 3
        assert adapter_rows[0]["T_world_camera"] == dense["frames"][0]["T_world_camera"]
        assert adapter_rows[0]["scale_status"] == "video_derived_uncertain_without_external_metric_anchor"
