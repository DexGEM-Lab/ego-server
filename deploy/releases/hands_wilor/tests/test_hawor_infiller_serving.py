"""CPU-only tests for the HaWoR + motion-infiller GPU3 Ray Serve vertical slice.

These tests never import Ray and never load a model. They verify:

* The HaWoR/Infiller contract shapes and the typed DROID/UniDepth input enforcement.
* The infiller's real coupled 120-step / 218-D two-hand/world checkpoint contract is
  represented honestly (single-hand rejected, two-hand required).
* The camera<->world adapter is physically reversible (round-trip identity on
  rotation/translation up to the documented MANO-wrist-offset constant).
* The rot6d<->rotmat conversions are exact inverses.
* One Serve batch callback is one model forward (HaWoR) and one-per-window (infiller).
* Resident revision ownership, admission rejection of incompatible shapes, and
  per-frame occlusion/uncertainty propagation.
"""
from __future__ import annotations

import asyncio
from typing import Any, Mapping

import numpy as np
import pytest

from ego_annotation.serving.batching import BatchPolicy
from ego_annotation.serving.contracts import (
    ContractValidationError,
    ImageSize,
    Ownership,
    PixelTransform,
    SpatialMetadata,
    TensorPayload,
)
from ego_annotation.serving.hawor import (
    HaWoRAdapter,
    HaWoRModelConfig,
    IMG_NORM_MEAN,
    IMG_NORM_STD,
    build_crop_geometry,
    build_hawor_model_config,
    decode_crop_batch,
)
from ego_annotation.serving.hawor_contracts import (
    HAWOR_CHUNK_LEN,
    HAWOR_CROP_H,
    HAWOR_CROP_W,
    INFILLER_HORIZON,
    INFILLER_PER_HAND_DIM,
    INFILLER_REPR_DIM,
    CropSourceTransform,
    DroidCameraEvidence,
    FrameObservation,
    HandSide,
    HandStateFrame,
    OcclusionState,
    TrackChunkRequest,
    UniDepthScaleK,
)
from ego_annotation.serving.infiller import (
    InfillerAdapter,
    InfillerModelConfig,
    _aa_to_rotmat,
    _build_218_sequence,
    _camera_to_world,
    _canonicalize_hand,
    _decanonicalize_hand,
    _rot6d_to_rotmat,
    _rotmat_to_rot6d,
    _world_to_camera,
    build_infiller_model_config,
)


HAWOR_REV = "hawor-v1"
INFILLER_REV = "hawor-infiller-v1"


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _ownership(rid: str) -> Ownership:
    return Ownership(
        request_id=rid, job_id="job-a", item_id=f"track-{rid}", stage_id="hawor.infer_tracks",
        source_id=f"src-{rid}", source_timestamp_s=0.0,
    )


def _crop_transform(i: int, *, do_flip: bool = False) -> CropSourceTransform:
    return CropSourceTransform(
        center=(960.0 + i, 540.0),
        scale=2.5,
        img_focal=1680.0,
        img_center=(960.0, 540.0),
        do_flip=do_flip,
        source_size=ImageSize(width=1920, height=1080),
        pixel_transform=PixelTransform.identity(),
    )


def _observations(i0: int = 0) -> tuple[FrameObservation, ...]:
    return tuple(
        FrameObservation(
            frame_index=i0 + i, source_timestamp_s=float(i0 + i) / 30.0,
            occlusion_state=OcclusionState.VISIBLE if i % 4 != 0 else OcclusionState.OCCLUDED,
            detection_confidence=0.85, side=HandSide.RIGHT,
        )
        for i in range(HAWOR_CHUNK_LEN)
    )


def _unidepth() -> UniDepthScaleK:
    return UniDepthScaleK(
        K_px=((1680.0, 0.0, 960.0), (0.0, 1680.0, 540.0), (0.0, 0.0, 1.0)),
        img_focal=1680.0, img_center=(960.0, 540.0),
        source_size=ImageSize(width=1920, height=1080),
        metric_scale=1.0, source="unidepth_v2_vitl14",
    )


def _droid(T: int = HAWOR_CHUNK_LEN) -> DroidCameraEvidence:
    # A non-identity SE(3) trajectory: translate along z and rotate about y.
    poses = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    for i in range(T):
        theta = 0.02 * i  # ~1deg/frame
        c, s = np.cos(theta), np.sin(theta)
        poses[i, 0, 0] = c
        poses[i, 0, 2] = s
        poses[i, 2, 0] = -s
        poses[i, 2, 2] = c
        poses[i, 1, 3] = 0.01 * i  # 1cm/frame translation
    ts = (np.arange(T, dtype=np.float64) / 30.0)
    return DroidCameraEvidence(
        poses_world_from_camera=TensorPayload(data=poses.tobytes(), shape=poses.shape, dtype="float32"),
        timestamps_s=TensorPayload(data=ts.tobytes(), shape=ts.shape, dtype="float64"),
        metric_scale=1.0, scale_residual=0.001, scale_confidence=0.95, source="droid+unidepth_scale",
    )


def _crop_batch_bytes() -> bytes:
    rng = np.random.default_rng(0)
    crops = rng.normal(0.0, 1.0, size=(HAWOR_CHUNK_LEN, 3, HAWOR_CROP_H, HAWOR_CROP_W)).astype(np.float32)
    return crops.tobytes()


def make_track_request(rid: str, *, model_revision: str = HAWOR_REV, do_flip: bool = False, with_droid: bool = True) -> TrackChunkRequest:
    return TrackChunkRequest(
        ownership=_ownership(rid),
        track_id=f"track-{rid}",
        side=HandSide.LEFT if do_flip else HandSide.RIGHT,
        crop_batch=TensorPayload(
            data=_crop_batch_bytes(),
            shape=(HAWOR_CHUNK_LEN, 3, HAWOR_CROP_H, HAWOR_CROP_W),
            dtype="float32",
        ),
        crop_transforms=tuple(_crop_transform(i, do_flip=do_flip) for i in range(HAWOR_CHUNK_LEN)),
        observations=_observations(),
        unidepth=_unidepth(),
        droid_evidence=_droid() if with_droid else None,
        model_revision=model_revision,
    )


def _hawor_config() -> HaWoRModelConfig:
    return build_hawor_model_config(
        checkpoint="server-owned-hawor.ckpt",
        model_revision=HAWOR_REV,
        device="cpu",
        batch_policy=BatchPolicy(max_batch_size=4, batch_wait_timeout_s=0.01, max_queued_requests=4),
    )


def _infiller_config() -> InfillerModelConfig:
    return build_infiller_model_config(
        checkpoint="server-owned-infiller.pt",
        model_revision=INFILLER_REV,
        device="cpu",
        batch_policy=BatchPolicy(max_batch_size=2, batch_wait_timeout_s=0.01, max_queued_requests=4),
    )


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestHaWoRContract:
    def test_crop_batch_shape_enforced(self) -> None:
        with pytest.raises(ContractValidationError):
            TrackChunkRequest(
                ownership=_ownership("r1"), track_id="t", side=HandSide.RIGHT,
                crop_batch=TensorPayload(data=b"\x00" * 3, shape=(8, 3, 256, 256), dtype="float32"),
                crop_transforms=tuple(_crop_transform(i) for i in range(HAWOR_CHUNK_LEN)),
                observations=_observations(), unidepth=_unidepth(), droid_evidence=None,
                model_revision=HAWOR_REV,
            )

    def test_uint8_rejected_float32_required(self) -> None:
        req = make_track_request("r1")
        bad = TensorPayload(data=req.crop_batch.data, shape=req.crop_batch.shape, dtype="uint8")
        with pytest.raises(ContractValidationError):
            TrackChunkRequest(
                ownership=req.ownership, track_id=req.track_id, side=req.side, crop_batch=bad,
                crop_transforms=req.crop_transforms, observations=req.observations,
                unidepth=req.unidepth, droid_evidence=None, model_revision=HAWOR_REV,
            )

    def test_filesystem_fields_rejected(self) -> None:
        payload = make_track_request("r1").to_wire()
        payload["input_video"] = "/tmp/x.mp4"
        with pytest.raises(ContractValidationError):
            TrackChunkRequest.from_wire(payload)

    def test_droid_evidence_optional_world_lift_unavailable(self) -> None:
        req = make_track_request("r1", with_droid=False)
        # Adapter with fake backend that returns minimal shapes.
        class FakeBackend:
            def infer_tracks(self, crop_batch, crop_geometry, img_center, do_flip) -> Mapping[str, Any]:
                B = crop_batch.shape[0]
                return {
                    "pred_rotmat": np.zeros((B, HAWOR_CHUNK_LEN, 16, 3, 3), dtype=np.float32),
                    "trans": np.zeros((B, HAWOR_CHUNK_LEN, 3), dtype=np.float32),
                    "betas": np.zeros((B, HAWOR_CHUNK_LEN, 10), dtype=np.float32),
                    "joints": np.zeros((B, HAWOR_CHUNK_LEN, 16, 3), dtype=np.float32),
                }
        adapter = HaWoRAdapter(_hawor_config(), backend_factory=lambda cfg: FakeBackend())
        result, err = asyncio.run(adapter.infer_tracks(req))
        assert err is None
        assert result is not None
        assert result.world_lift is None
        assert result.world_lift_status == "unavailable"
        assert result.trans.shape == (HAWOR_CHUNK_LEN, 3)
        assert result.root_orient.shape == (HAWOR_CHUNK_LEN, 3, 3)
        assert result.hand_pose.shape == (HAWOR_CHUNK_LEN, 15, 3, 3)
        assert result.joints.shape[0] == HAWOR_CHUNK_LEN
        assert len(result.occlusion_state) == HAWOR_CHUNK_LEN

    def test_droid_evidence_produces_world_lift(self) -> None:
        req = make_track_request("r1", with_droid=True)
        class FakeBackend:
            def infer_tracks(self, crop_batch, crop_geometry, img_center, do_flip) -> Mapping[str, Any]:
                B = crop_batch.shape[0]
                return {
                    "pred_rotmat": np.tile(np.eye(3, dtype=np.float32), (B, HAWOR_CHUNK_LEN, 16, 1, 1)),
                    "trans": np.full((B, HAWOR_CHUNK_LEN, 3), 0.3, dtype=np.float32),
                    "betas": np.zeros((B, HAWOR_CHUNK_LEN, 10), dtype=np.float32),
                    "joints": np.zeros((B, HAWOR_CHUNK_LEN, 16, 3), dtype=np.float32),
                }
        adapter = HaWoRAdapter(_hawor_config(), backend_factory=lambda cfg: FakeBackend())
        result, err = asyncio.run(adapter.infer_tracks(req))
        assert err is None
        assert result.world_lift is not None
        assert result.world_lift.shape == (HAWOR_CHUNK_LEN, 4, 4)
        assert "resampled_droid_world_from_camera" in result.world_lift_status
        assert result.model_revision == HAWOR_REV


# ---------------------------------------------------------------------------
# Infiller contract + adapter reversibility tests
# ---------------------------------------------------------------------------


def test_load_hawor_backend_returns_backend_with_1d_scalar_geometry(tmp_path, monkeypatch):
    """Regression for the mis-nested class defect: the loader must return a backend,
    and the forward batch must carry 1-D scale/img_focal per upstream bbox_est."""
    import os
    import sys
    import types
    from ego_annotation.serving import hawor as hawor_module

    captured: dict[str, Any] = {}

    class FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array)
        @property
        def shape(self):
            return self.array.shape
        def to(self, _device):
            return self
        def __getitem__(self, key):
            return FakeTensor(self.array[key])
        def reshape(self, *shape):
            return FakeTensor(self.array.reshape(*shape))
        def cpu(self):
            return self
        def numpy(self):
            return self.array

    class FakeModel:
        def to(self, _device):
            return self
        def eval(self):
            return self
        def forward(self, batch):
            captured["batch"] = batch
            b = batch["img"].shape[0]
            rotmat = np.tile(np.eye(3, dtype=np.float32), (b * 16, 16, 1, 1))
            return {
                "out": {
                    "pred_rotmat": FakeTensor(rotmat),
                    "trans_full": FakeTensor(np.zeros((b * 16, 1, 3), np.float32)),
                    "pred_shape": FakeTensor(np.zeros((b * 16, 10), np.float32)),
                },
                "pred_keypoints_3d": FakeTensor(np.zeros((b * 16, 16, 3), np.float32)),
            }

    fake_torch = types.SimpleNamespace(
        from_numpy=lambda a: FakeTensor(a),
        tensor=lambda v, device=None: FakeTensor(v),
        inference_mode=lambda: __import__("contextlib").nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    hawor_video = types.ModuleType("scripts.scripts_test_video.hawor_video")
    hawor_video.load_hawor = lambda _ckpt: (FakeModel(), object())
    pkg_scripts = types.ModuleType("scripts")
    pkg_test_video = types.ModuleType("scripts.scripts_test_video")
    monkeypatch.setitem(sys.modules, "scripts", pkg_scripts)
    monkeypatch.setitem(sys.modules, "scripts.scripts_test_video", pkg_test_video)
    monkeypatch.setitem(sys.modules, "scripts.scripts_test_video.hawor_video", hawor_video)
    monkeypatch.setenv("EGO_HAWOR_REPO", str(tmp_path))

    cwd = os.getcwd()
    try:
        backend = hawor_module._load_hawor_backend(_hawor_config())
    finally:
        os.chdir(cwd)
    assert backend is not None, "backend loader returned None (class nesting defect)"
    assert hasattr(backend, "infer_tracks")
    crop_batch = np.zeros((1, 16, 3, 256, 256), dtype=np.float32)
    crop_geometry = np.zeros((1, 16, 4), dtype=np.float32)
    img_center = np.zeros((1, 16, 2), dtype=np.float32)
    result = backend.infer_tracks(crop_batch, crop_geometry, img_center, [False])
    assert result["trans"].shape == (1, 16, 3)
    batch = captured["batch"]
    assert batch["scale"].shape == (1, 16), batch["scale"].shape
    assert batch["img_focal"].shape == (1, 16), batch["img_focal"].shape
    assert batch["center"].shape == (1, 16, 2)
    assert batch["img_center"].shape == (1, 16, 2)


class TestInfillerContract:
    def test_repr_dim_is_218_two_hand(self) -> None:
        assert INFILLER_REPR_DIM == 218
        assert INFILLER_PER_HAND_DIM == 109
        assert INFILLER_HORIZON == 120

    def test_single_hand_window_rejected(self) -> None:
        from ego_annotation.serving.hawor_contracts import HandSequenceRequest
        frames = tuple(
            HandStateFrame(
                frame_index=i, source_timestamp_s=float(i) / 30.0, side=HandSide.RIGHT,
                root_orient=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                hand_pose=tuple((0.0, 0.0, 0.0) for _ in range(15)),
                trans=(0.0, 0.0, 0.3), betas=tuple(0.0 for _ in range(10)),
                observed=True, uncertainty=0.005,
            )
            for i in range(20)
        )
        with pytest.raises(ContractValidationError):
            HandSequenceRequest(
                ownership=_ownership("r1"), window_id="w1", frames=frames,
                droid_evidence=_droid(20), unidepth=_unidepth(), model_revision=INFILLER_REV,
            )

    def test_rot6d_round_trip(self) -> None:
        from scipy.spatial.transform import Rotation as R
        rng = np.random.default_rng(1)
        for _ in range(10):
            mat = R.random(random_state=rng).as_matrix().astype(np.float32)
            r6 = _rotmat_to_rot6d(mat)
            mat2 = _rot6d_to_rotmat(r6)
            assert np.allclose(mat, mat2, atol=1e-5)

    def test_camera_world_round_trip_identity(self) -> None:
        # Camera -> world -> camera must recover the original (rigid point transform).
        T = 16
        rng = np.random.default_rng(2)
        from scipy.spatial.transform import Rotation as R
        trans_cam = rng.normal(0, 0.3, (T, 3)).astype(np.float32)
        root_cam = np.stack([R.random(random_state=rng).as_matrix().astype(np.float32) for _ in range(T)])
        hp_cam = np.tile(np.eye(3, dtype=np.float32), (T, 15, 1, 1))
        betas = np.zeros((T, 10), dtype=np.float32)
        R_c2w = np.stack([R.random(random_state=rng).as_matrix().astype(np.float32) for _ in range(T)])
        t_c2w = rng.normal(0, 0.5, (T, 3)).astype(np.float32)
        R_w2c = np.transpose(R_c2w, (0, 2, 1))
        t_w2c = -np.einsum("tij,tj->ti", R_w2c, t_c2w)
        root_w, trans_w, hp_w, betas_w = _camera_to_world(R_c2w, t_c2w, trans_cam, root_cam, hp_cam, betas, HandSide.RIGHT)
        root_c2, trans_c2, _, _ = _world_to_camera(R_w2c, t_w2c, trans_w, root_w, hp_w, betas_w)
        assert np.allclose(trans_cam, trans_c2, atol=1e-5)
        assert np.allclose(root_cam, root_c2, atol=1e-5)

    def test_canonical_round_trip_identity(self) -> None:
        T = 16
        rng = np.random.default_rng(3)
        from scipy.spatial.transform import Rotation as R
        trans_w = rng.normal(0, 0.3, (T, 3)).astype(np.float32)
        root_w = np.stack([R.random(random_state=rng).as_matrix().astype(np.float32) for _ in range(T)])
        hp_w = np.tile(np.eye(3, dtype=np.float32), (T, 15, 1, 1))
        betas = rng.normal(0, 0.1, (T, 10)).astype(np.float32)
        valid = np.array([i % 4 != 0 for i in range(T)], dtype=bool)
        valid[0] = True  # ensure first frame observed
        tc, rc, hpc, bc, transform = _canonicalize_hand(trans_w, root_w, hp_w, betas, valid)
        tw2, rw2, hpw2, bw2 = _decanonicalize_hand(tc, rc, hpc, bc, transform)
        assert np.allclose(trans_w, tw2, atol=1e-5)
        assert np.allclose(root_w, rw2, atol=1e-5)

    def test_build_218_shape_and_padding(self) -> None:
        T = 30
        left = {
            "trans": np.zeros((T, 3), dtype=np.float32),
            "root": np.tile(np.eye(3, dtype=np.float32), (T, 1, 1)),
            "hand_pose": np.tile(np.eye(3, dtype=np.float32), (T, 15, 1, 1)),
            "betas": np.zeros((T, 10), dtype=np.float32),
        }
        right = {k: v.copy() for k, v in left.items()}
        valid = np.ones((2, T), dtype=bool)
        seq, vp, T_orig = _build_218_sequence(left, right, valid, INFILLER_HORIZON)
        assert seq.shape == (INFILLER_HORIZON, INFILLER_REPR_DIM)
        assert vp.shape == (2, INFILLER_HORIZON)
        assert T_orig == T

    def test_infiller_fill_round_trip_through_fake_backend(self) -> None:
        # Fake backend echoes the input 218 sequence (identity fill). The adapter
        # must still round-trip camera->world->canonical->world->camera.
        from ego_annotation.serving.hawor_contracts import HandSequenceRequest
        T = 30
        from scipy.spatial.transform import Rotation as R
        rng = np.random.default_rng(4)
        frames = []
        for i in range(T):
            for side in (HandSide.LEFT, HandSide.RIGHT):
                mat = R.random(random_state=rng).as_matrix().astype(np.float32)
                frames.append(HandStateFrame(
                    frame_index=i, source_timestamp_s=float(i) / 30.0, side=side,
                    root_orient=tuple(tuple(row) for row in mat),
                    hand_pose=tuple((0.0, 0.0, 0.0) for _ in range(15)),
                    trans=(0.1, 0.0, 0.3 + 0.001 * i), betas=tuple(0.0 for _ in range(10)),
                    observed=(i % 5 != 0), uncertainty=0.005,
                ))
        req = HandSequenceRequest(
            ownership=_ownership("r1"), window_id="w1", frames=tuple(frames),
            droid_evidence=_droid(T), unidepth=_unidepth(), model_revision=INFILLER_REV,
        )

        class IdentityBackend:
            def fill(self, sequence_218, valid_mask) -> Mapping[str, Any]:
                T_orig = sequence_218.shape[0]
                out = sequence_218.copy().reshape(T_orig, 2, INFILLER_PER_HAND_DIM)
                return {"output": out, "T_original": T_orig}

        adapter = InfillerAdapter(_infiller_config(), backend_factory=lambda cfg: IdentityBackend())
        result, err = asyncio.run(adapter.fill(req))
        assert err is None, err
        assert result is not None
        assert result.trans.shape == (2, T, 3)
        assert result.root_orient.shape == (2, T, 3, 3)
        assert result.observed.shape == (2, T)
        assert result.inferred.shape == (2, T)
        # Inferred frames must have raised uncertainty (>= 0.08).
        unc = np.frombuffer(result.uncertainty.data, dtype=np.float32).reshape(2, T)
        inferred = np.frombuffer(result.inferred.data, dtype=bool).reshape(2, T)
        assert (unc[inferred] >= 0.08).all()
        assert "two-hand coupled 218-D" in result.adapter_notes[0]
        assert result.model_revision == INFILLER_REV
        trace = result.to_wire()["trace"]
        assert trace["request_count"] == 1
        assert trace["forward_count"] == 1
        assert trace["model_load_count"] == 1


# ---------------------------------------------------------------------------
# Batching / one-forward / ownership tests
# ---------------------------------------------------------------------------


class TestBatchingAndOwnership:
    def test_hawor_batch_two_distinct_ownerships_one_forward(self) -> None:
        reqs = [make_track_request(f"r{i}", with_droid=False) for i in range(2)]
        forward_call_count = {"n": 0}

        class FakeBackend:
            def infer_tracks(self, crop_batch, crop_geometry, img_center, do_flip) -> Mapping[str, Any]:
                forward_call_count["n"] += 1
                B = crop_batch.shape[0]
                return {
                    "pred_rotmat": np.zeros((B, HAWOR_CHUNK_LEN, 16, 3, 3), dtype=np.float32),
                    "trans": np.zeros((B, HAWOR_CHUNK_LEN, 3), dtype=np.float32),
                    "betas": np.zeros((B, HAWOR_CHUNK_LEN, 10), dtype=np.float32),
                    "joints": np.zeros((B, HAWOR_CHUNK_LEN, 16, 3), dtype=np.float32),
                }
        adapter = HaWoRAdapter(_hawor_config(), backend_factory=lambda cfg: FakeBackend())
        prepared = [adapter.admit(r) for r in reqs]
        results = asyncio.run(adapter.infer_batch(prepared))
        assert forward_call_count["n"] == 1, "two distinct chunks must fuse into one forward"
        assert len(results) == 2
        owners = {r.request.ownership.request_id for r in prepared}
        out_owners = {res[0].ownership.request_id for res in results}
        assert owners == out_owners, "ownership must split back to callers"
        traces = [res[0].to_wire()["trace"] for res in results]
        assert len({trace["batch_id"] for trace in traces}) == 1
        assert all(trace["request_count"] == 2 and trace["forward_count"] == 1 for trace in traces)
        assert all(trace["model_load_count"] == 1 for trace in traces)

    def test_hawor_revision_mismatch_rejected_at_admission(self) -> None:
        req = make_track_request("r1", model_revision="wrong-revision")
        class FakeBackend:
            def infer_tracks(self, *a, **k): raise AssertionError("should not forward")
        adapter = HaWoRAdapter(_hawor_config(), backend_factory=lambda cfg: FakeBackend())
        with pytest.raises(ContractValidationError):
            adapter.admit(req)

    def test_hawor_status_model_load_count_constant(self) -> None:
        class FakeBackend:
            def infer_tracks(self, crop_batch, crop_geometry, img_center, do_flip) -> Mapping[str, Any]:
                B = crop_batch.shape[0]
                return {
                    "pred_rotmat": np.zeros((B, HAWOR_CHUNK_LEN, 16, 3, 3), dtype=np.float32),
                    "trans": np.zeros((B, HAWOR_CHUNK_LEN, 3), dtype=np.float32),
                    "betas": np.zeros((B, HAWOR_CHUNK_LEN, 10), dtype=np.float32),
                    "joints": np.zeros((B, HAWOR_CHUNK_LEN, 16, 3), dtype=np.float32),
                }
        adapter = HaWoRAdapter(_hawor_config(), backend_factory=lambda cfg: FakeBackend())
        assert adapter.status().model_load_count == 1
        asyncio.run(adapter.infer_tracks(make_track_request("r1", with_droid=False)))
        asyncio.run(adapter.infer_tracks(make_track_request("r2", with_droid=False)))
        assert adapter.status().model_load_count == 1, "model must load once and stay resident"
