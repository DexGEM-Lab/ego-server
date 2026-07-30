from __future__ import annotations

import numpy as np
import pytest

from scripts.run_v22_hawor_no_droid_stage import _load_camera, _world_validity


@pytest.mark.parametrize("frame_count", [1025, 1440])
def test_masked_droid_camera_never_endpoint_fills_uncovered_tail(tmp_path, frame_count: int) -> None:
    supported = 1024
    path = tmp_path / f"droid_{frame_count}.npz"
    poses = np.tile(np.eye(4, dtype=np.float32)[None], (frame_count, 1, 1))
    poses[:, 0, 3] = np.arange(frame_count, dtype=np.float32)
    # Deliberately endpoint-filled input tail: the mask must make it unusable.
    poses[supported:] = poses[supported - 1]
    mask = np.zeros(frame_count, dtype=np.uint8)
    mask[:supported] = 1
    np.savez_compressed(path, frame_idx=np.arange(frame_count, dtype=np.int32), T_world_camera=poses, droid_pose_valid=mask)

    dense, camera_valid, status = _load_camera(path, frame_count)

    assert status == "upstream_camera_artifact_masked:droid_pose_valid"
    assert np.array_equal(camera_valid, mask.astype(bool))
    assert np.isfinite(dense[:supported]).all()
    assert np.isnan(dense[supported:]).all()
    # Model-local rows cannot be emitted as world rows after camera coverage ends.
    model_valid = np.ones(frame_count, dtype=np.uint8)
    world_valid = _world_validity(model_valid, camera_valid)
    assert world_valid[:supported].all()
    assert not world_valid[supported:].any()


def test_maskless_droid_camera_artifact_is_rejected(tmp_path) -> None:
    path = tmp_path / "droid_dense_trajectory.npz"
    poses = np.tile(np.eye(4, dtype=np.float32)[None], (1025, 1, 1))
    poses[1024] = poses[1023]
    np.savez_compressed(path, frame_idx=np.arange(1025, dtype=np.int32), T_world_camera=poses)

    with pytest.raises(RuntimeError, match="requires authoritative"):
        _load_camera(path, 1025)


def test_missing_camera_never_uses_identity_as_world_camera() -> None:
    poses, valid, status = _load_camera(None, 4)

    assert status == "camera_absent_world_unresolved"
    assert not valid.any()
    assert np.isnan(poses).all()
