from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.render_v22_hybrid_hand_overlay import parse_args, render


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_hybrid_overlay_renderer_writes_frame_count(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe are required for video renderer smoke")
    run_root = tmp_path / "run"
    rgb_dir = run_root / "input" / "raw_frame_manifest" / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for idx in range(2):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        image[:, :, 1] = 32
        path = rgb_dir / f"{idx:06d}.jpg"
        cv2.imwrite(str(path), image)
        frames.append({"frame_idx": idx, "rgb": str(path), "source_width": 160, "source_height": 120, "manifest_width": 160, "manifest_height": 120, "time_s": idx / 30})
    write_json(run_root / "input" / "raw_frame_manifest" / "manifest.json", {"fps": 30.0, "frame_count": 2, "frames": frames})
    write_json(run_root / "state" / "calibration" / "v19_camera_calibration_contract.json", {"intrinsics_fx_fy_cx_cy": [120.0, 120.0, 80.0, 60.0]})
    hand = np.zeros((2, 21, 3), dtype=np.float32)
    for i in range(21):
        hand[:, i, 0] = (i % 5 - 2) * 0.01
        hand[:, i, 1] = (i // 5 - 2) * 0.01
        hand[:, i, 2] = 0.5
    hands_dir = run_root / "state" / "hands_metric"
    hands_dir.mkdir(parents=True, exist_ok=True)
    payload = {"frame_idx": np.asarray([0, 1], dtype=np.int32), "R_c2w": np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0), "t_c2w": np.zeros((2, 3), dtype=np.float32)}
    for side in ("left", "right"):
        offset = -0.03 if side == "left" else 0.03
        payload[f"{side}_joints_world_m"] = hand + np.asarray([offset, 0.0, 0.0], dtype=np.float32)[None, None, :]
        payload[f"{side}_valid"] = np.ones(2, dtype=np.uint8)
        payload[f"{side}_wilor_fit_reprojection_median_px"] = np.asarray([10.0, 80.0], dtype=np.float32)
    np.savez_compressed(run_root / "state" / "hands_metric" / "v22_hybrid_hands_metric.npz", **payload)
    args = parse_args(["--run-root", str(run_root)])
    report = render(args)
    assert report["status"] == "ok"
    assert report["frame_count"] == 2
    assert report["video_frame_count"] == 2
    assert report["draw_counts"]["hands_projected"] == 4
    assert Path(report["output_video"]).exists()
