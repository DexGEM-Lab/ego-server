from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np


BRIDGE_PATH = Path(__file__).parents[1] / "scripts" / "estimate_droid_metric3d_scale_bridge.py"
spec = importlib.util.spec_from_file_location("estimate_droid_metric3d_scale_bridge", BRIDGE_PATH)
assert spec and spec.loader
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)


def test_multi_bridge_loads_metric_once_and_reuses_one_depth_pass(tmp_path, monkeypatch) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for frame_index in range(4):
        assert cv2.imwrite(str(image_dir / f"{frame_index:08d}.jpg"), np.full((8, 8, 3), frame_index, dtype=np.uint8))
    masks = tmp_path / "masks.npy"
    np.save(masks, np.zeros((4, 4, 4), dtype=np.uint8), allow_pickle=False)
    calib = tmp_path / "calib.npy"
    np.save(calib, np.asarray([10.0, 10.0, 4.0, 4.0], dtype=np.float32), allow_pickle=False)
    geometry = tmp_path / "geometry.npz"
    np.savez_compressed(
        geometry,
        frame_idx=np.arange(4, dtype=np.int32),
        session_count=np.asarray(2, dtype=np.int32),
        session_0_tstamp=np.asarray([0, 1], dtype=np.int32),
        session_0_disps=np.ones((2, 2, 2), dtype=np.float32),
        session_1_tstamp=np.asarray([2, 3], dtype=np.int32),
        session_1_disps=np.full((2, 2, 2), 2.0, dtype=np.float32),
    )
    output = tmp_path / "scales.json"
    calls: list[str] = []
    models: list[str] = []

    class FakeMetric3D:
        def __init__(self, checkpoint: str) -> None:
            models.append(checkpoint)

        def __call__(self, image_path: str, _calib: np.ndarray) -> np.ndarray:
            calls.append(Path(image_path).name)
            return np.full((4, 4), 2.0, dtype=np.float32)

    def fake_est(slam_depth, _metric_depth, **_kwargs) -> float:
        return float(np.mean(slam_depth))

    monkeypatch.setattr(bridge, "import_scale_helpers", lambda _root: (FakeMetric3D, fake_est))
    monkeypatch.setattr(sys, "argv", [
        "estimate_droid_metric3d_scale_bridge.py",
        "--multi-geometry", str(geometry), "--image-dir", str(image_dir),
        "--masks", str(masks), "--calib", str(calib), "--hawor-root", str(tmp_path),
        "--metric-checkpoint", "/bin/true", "--output", str(output),
    ])

    assert bridge.main() == 0
    payload = json.loads(output.read_text())
    assert models == ["/bin/true"]
    assert calls == [f"{index:08d}.jpg" for index in range(4)]
    assert payload["shared_metric3d"] == {
        "metric_model_load_count": 1,
        "metric_depth_pass_count": 4,
        "source_frame_count": 4,
    }
    assert [row["report"]["exact_keyframe_source_ids"] for row in payload["sessions"]] == [[0, 1], [2, 3]]
    assert [row["report"]["metric_depth_pass_count"] for row in payload["sessions"]] == [4, 4]
