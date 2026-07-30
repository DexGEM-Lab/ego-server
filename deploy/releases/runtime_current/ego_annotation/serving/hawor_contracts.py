"""HaWoR + motion-infiller model-native contracts for the GPU3 Ray Serve slice.

This module is Python 3.10-compatible (the HaWoR serving environment runs Python
3.10) and is importable without Ray installed. It reuses the shared ownership /
spatial / batch-trace / transport primitives from ``contracts.py`` and adds only
the model-specific request/response types for the two colocated GPU3 APIs:

* ``hawor.infer_tracks``: consumes true ``[B,16,3,256,256]`` timestamped hand-crop
  chunks plus the crop/source transforms (center, scale, img_focal, img_center,
  do_flip) and the observation/visibility evidence that the HaWoR backbone requires
  for its CLIFF bbox feature and metric camera translation. It also consumes the
  canonical masked-DROID camera trajectory and the UniDepth metric scale/K as typed
  inputs; it never reruns DROID or Metric3D. Output is observed camera-space metric
  MANO state/surfaces with provenance, uncertainty, and per-frame occlusion states.

* ``hawor_infiller.fill``: exposes the real checkpoint's semantics honestly. The
  checkpoint is a coupled 120-step, 218-D two-hand/world-sequence transformer
  (``2 * (3 + 10 + 96)`` = 218: per hand ``trans(3) + betas(10) + rot6d(16*6=96)``).
  The requested independent camera-space contract needs an adapter; we implement a
  physically explicit reversible camera<->world adapter driven by the typed DROID
  camera trajectory, and represent the residual semantic mismatch (two-hand
  coupling, world/canonical operating frame) as a documented constraint rather than
  relabeling a different tensor as the infiller.

Numerical invariants enforced here:

* The HaWoR crop batch is normalized float32 BCHW ``[B,16,3,256,256]`` matching the
  ImageNet-normalized output of ``TrackDatasetEval.__getitem__`` (the model's own
  ``crop``+``Normalize`` path). The API does NOT re-crop; it takes pre-cropped
  normalized tensors plus the transforms that produced them, so a batch callback is
  exactly one ``HAWOR.forward`` and the crop/source transforms travel alongside.
* Each track chunk is exactly 16 frames (the model pads internally to multiples of
  16, but the public contract admits one canonical 16-frame bucket so one Serve
  callback is one forward). Shorter chunks are rejected at admission; the caller
  pads or splits upstream.
* The infiller window is exactly 120 frames (the checkpoint horizon). The two-hand
  coupling is structural: a window must carry both hands with a per-frame
  observation/occlusion mask per hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence, cast

from ego_annotation.serving.contracts import (
    BatchTrace,
    ContractValidationError,
    ImageSize,
    Ownership,
    PixelTransform,
    SpatialMetadata,
    TensorPayload,
    _matrix3,
    _positive_int,
    _required_text,
    _timestamp,
    reject_filesystem_fields,
)


HAWOR_CHUNK_LEN = 16
HAWOR_CROP_H = 256
HAWOR_CROP_W = 256
HAWOR_CROP_CHANNELS = 3
# Normalized float32 crops, matching TrackDatasetEval (ToTensor + ImageNet Normalize).
HAWOR_CROP_DTYPE = "float32"
HAWOR_HAND_JOINTS = 15
HAWOR_MANO_VERTS = 778
INFILLER_HORIZON = 120
INFILLER_REPR_DIM = 218  # 2 * (3 + 10 + 96)
INFILLER_PER_HAND_DIM = 109  # 3 + 10 + 6 + 90


class HandSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class OcclusionState(str, Enum):
    """Per-frame hand visibility state for HaWoR track chunks.

    ``visible``: detected and used as a model observation.
    ``partially_visible``: detected with low support; carried with raised uncertainty.
    ``occluded``: not detected this frame; carried as padding, output marked uncertain.
    ``out_of_frame``: crop transform is out of source frame; carried as padding.
    ``unresolved``: observation support unknown.
    """

    VISIBLE = "visible"
    PARTIALLY_VISIBLE = "partially_visible"
    OCCLUDED = "occluded"
    OUT_OF_FRAME = "out_of_frame"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class CropSourceTransform:
    """The crop->source geometry HaWoR's bbox_est / get_trans require.

    These are exactly the fields ``TrackDatasetEval`` produces per item and that
    ``HAWOR.forward_step`` consumes: ``center`` (crop center in source pixels),
    ``scale`` (boxes_2_cs scale = max(w,h)/200, pre-dilate), ``img_focal`` (scalar
    focal length), ``img_center`` (source image center [cx, cy]), and ``do_flip``
    (left-hand mirroring). The source image size and a full pixel transform are
    carried for downstream reprojection joins.
    """

    center: tuple[float, float]
    scale: float
    img_focal: float
    img_center: tuple[float, float]
    do_flip: bool
    source_size: ImageSize
    pixel_transform: PixelTransform

    def __post_init__(self) -> None:
        if len(self.center) != 2 or len(self.img_center) != 2:
            raise ContractValidationError("center and img_center must be 2-vectors")
        for name in ("center", "img_center"):
            for v in getattr(self, name):
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise ContractValidationError(f"{name} must be numeric")
        if not isinstance(self.scale, (int, float)) or isinstance(self.scale, bool) or self.scale <= 0:
            raise ContractValidationError("scale must be a positive number")
        if not isinstance(self.img_focal, (int, float)) or isinstance(self.img_focal, bool) or self.img_focal <= 0:
            raise ContractValidationError("img_focal must be a positive number")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CropSourceTransform":
        reject_filesystem_fields(payload, context="crop_transform")
        return cls(
            center=tuple(float(x) for x in payload["center"]),
            scale=float(payload["scale"]),
            img_focal=float(payload["img_focal"]),
            img_center=tuple(float(x) for x in payload["img_center"]),
            do_flip=bool(payload["do_flip"]),
            source_size=ImageSize.from_mapping(payload["source_size"]),
            pixel_transform=PixelTransform.from_mapping(payload.get("pixel_transform", {})),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "center": list(self.center),
            "scale": self.scale,
            "img_focal": self.img_focal,
            "img_center": list(self.img_center),
            "do_flip": self.do_flip,
            "source_size": self.source_size.to_wire(),
            "pixel_transform": self.pixel_transform.to_wire(),
        }


@dataclass(frozen=True)
class FrameObservation:
    """Per-frame observation/visibility evidence for one track chunk frame."""

    frame_index: int
    source_timestamp_s: float
    occlusion_state: OcclusionState
    detection_confidence: float
    # Per-frame side (may switch within a track in principle; HaWoR resolves one
    # side per chunk via do_flip, but the evidence is carried for provenance).
    side: HandSide

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ContractValidationError("frame_index must be non-negative")
        if not isinstance(self.source_timestamp_s, (int, float)) or isinstance(self.source_timestamp_s, bool):
            raise ContractValidationError("source_timestamp_s must be numeric")
        if not (0.0 <= self.detection_confidence <= 1.0):
            raise ContractValidationError("detection_confidence must be in [0, 1]")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FrameObservation":
        reject_filesystem_fields(payload, context="observation")
        return cls(
            frame_index=int(payload["frame_index"]),
            source_timestamp_s=float(payload["source_timestamp_s"]),
            occlusion_state=OcclusionState(payload["occlusion_state"]),
            detection_confidence=float(payload["detection_confidence"]),
            side=HandSide(payload["side"]),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "source_timestamp_s": self.source_timestamp_s,
            "occlusion_state": self.occlusion_state.value,
            "detection_confidence": self.detection_confidence,
            "side": self.side.value,
        }


@dataclass(frozen=True)
class DroidCameraEvidence:
    """Canonical masked-DROID camera trajectory consumed as a typed input.

    HaWoR must NOT rerun DROID. The trajectory is the world-from-camera pose stream
    (``T_world_camera``) at DROID keyframe/source timestamps, plus the metric scale
    and the dense disparity gauge. ``poses`` is ``[T, 4, 4]`` SE(3) world-from-camera
    (rotation+translation), matching the verified DROID convention
    (``Droid.terminate()`` returns ``camera_trajectory.inv()``). Timestamps join to
    the HaWoR chunk frames by source time, not by frame index.
    """

    poses_world_from_camera: TensorPayload  # [T, 4, 4] float32
    timestamps_s: TensorPayload  # [T] float64
    metric_scale: float
    scale_residual: float
    scale_confidence: float
    source: str  # e.g. "droid+unidepth_scale"

    def __post_init__(self) -> None:
        if self.poses_world_from_camera.shape[-2:] != (4, 4) or len(self.poses_world_from_camera.shape) != 3:
            raise ContractValidationError(
                "poses_world_from_camera must be [T,4,4] float32 world-from-camera SE(3)"
            )
        if len(self.timestamps_s.shape) != 1 or self.timestamps_s.shape[0] != self.poses_world_from_camera.shape[0]:
            raise ContractValidationError("timestamps_s must be [T] matching poses")
        if self.metric_scale <= 0:
            raise ContractValidationError("metric_scale must be positive")
        _required_text(self.source, "droid evidence source")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DroidCameraEvidence":
        reject_filesystem_fields(payload, context="droid_evidence")
        return cls(
            poses_world_from_camera=TensorPayload.from_wire(payload["poses_world_from_camera"]),
            timestamps_s=TensorPayload.from_wire(payload["timestamps_s"]),
            metric_scale=float(payload["metric_scale"]),
            scale_residual=float(payload["scale_residual"]),
            scale_confidence=float(payload["scale_confidence"]),
            source=_required_text(payload.get("source"), "droid evidence source"),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "poses_world_from_camera": self.poses_world_from_camera.to_wire(),
            "timestamps_s": self.timestamps_s.to_wire(),
            "metric_scale": self.metric_scale,
            "scale_residual": self.scale_residual,
            "scale_confidence": self.scale_confidence,
            "source": self.source,
        }


@dataclass(frozen=True)
class UniDepthScaleK:
    """UniDepth metric scale / intrinsics consumed as a typed input.

    HaWoR must NOT rerun UniDepth/Metric3D. The intrinsic matrix ``K_px`` (source
    pixel grid) and the metric scale anchor the camera-space metric translation.
    ``img_focal``/``img_center`` are derived from K for HaWoR's scalar-focal path
    and are carried explicitly so the adapter does not silently invent intrinsics.
    """

    K_px: tuple[tuple[float, float, float], ...]
    img_focal: float
    img_center: tuple[float, float]
    source_size: ImageSize
    metric_scale: float
    source: str  # e.g. "unidepth_v2_vitl14"

    def __post_init__(self) -> None:
        object.__setattr__(self, "K_px", _matrix3(self.K_px, "K_px"))
        if len(self.img_center) != 2:
            raise ContractValidationError("img_center must be a 2-vector")
        if self.img_focal <= 0 or self.metric_scale <= 0:
            raise ContractValidationError("img_focal and metric_scale must be positive")
        _required_text(self.source, "unidepth source")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "UniDepthScaleK":
        reject_filesystem_fields(payload, context="unidepth_scale_k")
        return cls(
            K_px=cast(tuple[tuple[float, float, float], ...], payload["K_px"]),
            img_focal=float(payload["img_focal"]),
            img_center=tuple(float(x) for x in payload["img_center"]),
            source_size=ImageSize.from_mapping(payload["source_size"]),
            metric_scale=float(payload["metric_scale"]),
            source=_required_text(payload.get("source"), "unidepth source"),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "K_px": [list(row) for row in self.K_px],
            "img_focal": self.img_focal,
            "img_center": list(self.img_center),
            "source_size": self.source_size.to_wire(),
            "metric_scale": self.metric_scale,
            "source": self.source,
        }


@dataclass(frozen=True)
class TrackChunkRequest:
    """One ``hawor.infer_tracks`` request: one 16-frame hand-crop chunk.

    The crop batch is the normalized float32 ``[16,3,256,256]`` tensor (BCHW) that
    ``TrackDatasetEval`` would produce; the per-frame crop/source transforms and
    observation evidence travel alongside so the model's bbox_est / get_trans run on
    the real geometry. The DROID camera evidence and UniDepth scale/K are typed
    inputs (never recomputed); they drive the world-lift fusion output. If
    ``droid_evidence`` is None the response carries only camera-space state with
    ``world_lift_status="unavailable"`` rather than silently inventing a world frame.
    """

    ownership: Ownership
    track_id: str
    side: HandSide
    crop_batch: TensorPayload  # [16,3,256,256] float32 normalized
    crop_transforms: tuple[CropSourceTransform, ...]  # len 16
    observations: tuple[FrameObservation, ...]  # len 16
    unidepth: UniDepthScaleK
    droid_evidence: DroidCameraEvidence | None
    model_revision: str
    options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.track_id, "track_id")
        _required_text(self.model_revision, "model_revision")
        expected_crop = (HAWOR_CHUNK_LEN, HAWOR_CROP_CHANNELS, HAWOR_CROP_H, HAWOR_CROP_W)
        if self.crop_batch.shape != expected_crop:
            raise ContractValidationError(
                f"crop_batch must be {expected_crop} float32 normalized BCHW, got {self.crop_batch.shape}"
            )
        if self.crop_batch.dtype != HAWOR_CROP_DTYPE:
            raise ContractValidationError(f"crop_batch dtype must be {HAWOR_CROP_DTYPE}")
        if len(self.crop_transforms) != HAWOR_CHUNK_LEN:
            raise ContractValidationError(f"crop_transforms must have {HAWOR_CHUNK_LEN} entries")
        if len(self.observations) != HAWOR_CHUNK_LEN:
            raise ContractValidationError(f"observations must have {HAWOR_CHUNK_LEN} entries")

    @property
    def work_units(self) -> int:
        # One track chunk == one model forward work unit (one 16-frame forward).
        return 1

    @property
    def compatibility_key(self) -> tuple[Any, ...]:
        return (
            "hawor.infer_tracks",
            self.crop_batch.shape,
            self.crop_batch.dtype,
            self.model_revision,
            self.side,
            self.options,
        )

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "TrackChunkRequest":
        reject_filesystem_fields(payload)
        options = payload.get("options", {})
        if not isinstance(options, Mapping):
            raise ContractValidationError("options must be an object")
        return cls(
            ownership=Ownership.from_mapping(payload["ownership"]),
            track_id=_required_text(payload.get("track_id"), "track_id"),
            side=HandSide(payload["side"]),
            crop_batch=TensorPayload.from_wire(payload["crop_batch"]),
            crop_transforms=tuple(CropSourceTransform.from_mapping(c) for c in payload["crop_transforms"]),
            observations=tuple(FrameObservation.from_mapping(o) for o in payload["observations"]),
            unidepth=UniDepthScaleK.from_mapping(payload["unidepth"]),
            droid_evidence=DroidCameraEvidence.from_mapping(payload["droid_evidence"]) if payload.get("droid_evidence") else None,
            model_revision=_required_text(payload.get("model_revision"), "model_revision"),
            options=tuple(sorted((str(k), str(v)) for k, v in options.items())),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "track_id": self.track_id,
            "side": self.side.value,
            "crop_batch": self.crop_batch.to_wire(),
            "crop_transforms": [c.to_wire() for c in self.crop_transforms],
            "observations": [o.to_wire() for o in self.observations],
            "unidepth": self.unidepth.to_wire(),
            "droid_evidence": self.droid_evidence.to_wire() if self.droid_evidence else None,
            "model_revision": self.model_revision,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class CameraSpaceManoResult:
    """Observed camera-space metric MANO output for one track chunk.

    All arrays are per-frame over the 16-frame chunk. ``root_orient``/``hand_pose``
    are rotation matrices (the model's native output representation); ``trans`` is
    metric camera-space translation (metres). ``vertices``/``joints`` are the MANO
    surface/joints in the same camera frame. ``occlusion_state`` and ``uncertainty``
    are per-frame; ``observed`` flags which frames were real observations vs padding.
    ``world_lift`` carries the CPU-fused world-space lift when DROID evidence was
    provided, else ``None`` with an explicit status.
    """

    ownership: Ownership
    track_id: str
    side: HandSide
    root_orient: TensorPayload  # [16,3,3] float32
    hand_pose: TensorPayload  # [16,15,3,3] float32
    trans: TensorPayload  # [16,3] float32 metric camera-space
    betas: TensorPayload  # [16,10] float32
    vertices: TensorPayload  # [16,778,3] float32 camera-space
    joints: TensorPayload  # [16,16,3] float32 camera-space (wrist + 15 joints)
    observed: TensorPayload  # [16] bool
    occlusion_state: tuple[OcclusionState, ...]  # len 16
    uncertainty: TensorPayload  # [16] float32 in metres (wrist-radius equivalent)
    world_lift: TensorPayload | None  # [16,4,4] float32 world-from-camera per frame, if DROID provided
    world_lift_status: str
    spatial: SpatialMetadata
    model_revision: str
    trace: BatchTrace
    batch_diagnostics: Mapping[str, Any] | None = None
    server_identity: Any | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "track_id": self.track_id,
            "side": self.side.value,
            "root_orient": self.root_orient.to_wire(),
            "hand_pose": self.hand_pose.to_wire(),
            "trans": self.trans.to_wire(),
            "betas": self.betas.to_wire(),
            "vertices": self.vertices.to_wire(),
            "joints": self.joints.to_wire(),
            "observed": self.observed.to_wire(),
            "occlusion_state": [s.value for s in self.occlusion_state],
            "uncertainty": self.uncertainty.to_wire(),
            "world_lift": self.world_lift.to_wire() if self.world_lift else None,
            "world_lift_status": self.world_lift_status,
            "spatial": self.spatial.to_wire(),
            "model_revision": self.model_revision,
            "trace": self.trace.to_wire(),
            **({"server_identity": self.server_identity.to_wire()} if self.server_identity is not None else {}),
            **({"batch_diagnostics": dict(self.batch_diagnostics)} if self.batch_diagnostics is not None else {}),
        }


# ---------------------------------------------------------------------------
# Infiller contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HandStateFrame:
    """One frame of observed camera-space hand state for the infiller.

    ``root_orient``/``hand_pose`` are rotation matrices; ``trans`` is metric
    camera-space translation. ``observed`` is False for occluded/missing frames the
    infiller must fill. ``uncertainty`` is the observed-state uncertainty in metres.
    """

    frame_index: int
    source_timestamp_s: float
    side: HandSide
    root_orient: tuple[tuple[float, float, float], ...]  # [3,3]
    hand_pose: tuple[tuple[float, float, float], ...]  # [15,3] flattened? No: [45] -> store as 15x3
    trans: tuple[float, float, float]
    betas: tuple[float, ...]  # [10]
    observed: bool
    uncertainty: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_orient", _matrix3(self.root_orient, "root_orient"))
        if len(self.hand_pose) != HAWOR_HAND_JOINTS or any(len(r) != 3 for r in self.hand_pose):
            raise ContractValidationError("hand_pose must be [15,3]")
        if len(self.betas) != 10:
            raise ContractValidationError("betas must be [10]")
        if len(self.trans) != 3:
            raise ContractValidationError("trans must be [3]")
        if self.uncertainty < 0:
            raise ContractValidationError("uncertainty must be non-negative")


@dataclass(frozen=True)
class HandSequenceRequest:
    """One ``hawor_infiller.fill`` request: one ~120-frame two-hand camera-space window.

    The requested independent camera-space contract is adapted to the checkpoint's
    real coupled 120-step, 218-D two-hand/world semantics by a physically explicit
    reversible adapter (see ``infiller.py``). The adapter requires the DROID camera
    trajectory to lift camera->world before the infiller and invert world->camera
    after; without it the request is rejected (the mismatch is surfaced as a typed
    blocker, not silently dropped). Both hands must be present in the window because
    the 218-D vector is structurally two-hand coupled: a single-hand window cannot be
    served by this checkpoint and is rejected with an explicit error.
    """

    ownership: Ownership
    window_id: str
    frames: tuple[HandStateFrame, ...]  # <= 120 frames, two hands interleaved per timestamp
    # Per-timestamp observation mask: which hand(s) are observed at each source time.
    # frames carry their own side; the adapter groups by timestamp into the two-hand vector.
    droid_evidence: DroidCameraEvidence
    unidepth: UniDepthScaleK
    model_revision: str
    options: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.window_id, "window_id")
        _required_text(self.model_revision, "model_revision")
        if not self.frames:
            raise ContractValidationError("infiller window must contain at least one frame")
        if len(self.frames) > INFILLER_HORIZON * 2:
            raise ContractValidationError(
                f"infiller window exceeds {INFILLER_HORIZON}*2 frames (two hands)"
            )
        sides = {f.side for f in self.frames}
        if len(sides) < 2:
            raise ContractValidationError(
                "infiller checkpoint is structurally two-hand coupled (218-D); "
                "a single-hand window cannot be served. Provide both hands."
            )

    @property
    def work_units(self) -> int:
        # One infiller window == one 120-step forward work unit.
        return 1

    @property
    def compatibility_key(self) -> tuple[Any, ...]:
        return ("hawor_infiller.fill", self.model_revision, self.options)

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "HandSequenceRequest":
        reject_filesystem_fields(payload)
        options = payload.get("options", {})
        if not isinstance(options, Mapping):
            raise ContractValidationError("options must be an object")
        return cls(
            ownership=Ownership.from_mapping(payload["ownership"]),
            window_id=_required_text(payload.get("window_id"), "window_id"),
            frames=tuple(_hand_state_frame_from_wire(f) for f in payload["frames"]),
            droid_evidence=DroidCameraEvidence.from_mapping(payload["droid_evidence"]),
            unidepth=UniDepthScaleK.from_mapping(payload["unidepth"]),
            model_revision=_required_text(payload.get("model_revision"), "model_revision"),
            options=tuple(sorted((str(k), str(v)) for k, v in options.items())),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "window_id": self.window_id,
            "frames": [_hand_state_frame_to_wire(f) for f in self.frames],
            "droid_evidence": self.droid_evidence.to_wire(),
            "unidepth": self.unidepth.to_wire(),
            "model_revision": self.model_revision,
            "options": dict(self.options),
        }


def _hand_state_frame_from_wire(payload: Mapping[str, Any]) -> HandStateFrame:
    reject_filesystem_fields(payload, context="hand_state_frame")
    hp = payload["hand_pose"]
    if isinstance(hp, list) and hp and isinstance(hp[0], (int, float)):
        # flattened [45] -> [15,3]
        if len(hp) != 45:
            raise ContractValidationError("flattened hand_pose must be [45]")
        hp = [hp[i * 3:(i + 1) * 3] for i in range(15)]
    return HandStateFrame(
        frame_index=int(payload["frame_index"]),
        source_timestamp_s=float(payload["source_timestamp_s"]),
        side=HandSide(payload["side"]),
        root_orient=cast(tuple[tuple[float, float, float], ...], payload["root_orient"]),
        hand_pose=cast(tuple[tuple[float, float, float], ...], hp),
        trans=tuple(float(x) for x in payload["trans"]),
        betas=tuple(float(x) for x in payload["betas"]),
        observed=bool(payload["observed"]),
        uncertainty=float(payload["uncertainty"]),
    )


def _hand_state_frame_to_wire(f: HandStateFrame) -> dict[str, Any]:
    return {
        "frame_index": f.frame_index,
        "source_timestamp_s": f.source_timestamp_s,
        "side": f.side.value,
        "root_orient": [list(row) for row in f.root_orient],
        "hand_pose": [list(row) for row in f.hand_pose],
        "trans": list(f.trans),
        "betas": list(f.betas),
        "observed": f.observed,
        "uncertainty": f.uncertainty,
    }


@dataclass(frozen=True)
class CompletedHandSequenceResult:
    """Completed two-hand camera-space state with observed-vs-inferred flags.

    The infiller fills occluded/missing frames in the coupled two-hand world
    canonical sequence, then the adapter inverts to camera-space. ``inferred`` flags
    which frames were produced by the infiller (vs carried from observation).
    ``uncertainty`` is raised for inferred frames.
    """

    ownership: Ownership
    window_id: str
    # Per-hand, per-frame camera-space state. Order: [left_frames, right_frames].
    root_orient: TensorPayload  # [2, T, 3, 3] float32
    hand_pose: TensorPayload  # [2, T, 15, 3, 3] float32
    trans: TensorPayload  # [2, T, 3] float32 metric camera-space
    betas: TensorPayload  # [2, T, 10] float32
    observed: TensorPayload  # [2, T] bool
    inferred: TensorPayload  # [2, T] bool
    uncertainty: TensorPayload  # [2, T] float32 metres
    timestamps_s: TensorPayload  # [T] float64
    spatial: SpatialMetadata
    model_revision: str
    trace: BatchTrace
    # adapter provenance: documents the camera<->world lift and the two-hand coupling.
    adapter_notes: tuple[str, ...]
    batch_diagnostics: Mapping[str, Any] | None = None
    server_identity: Any | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "ownership": self.ownership.to_wire(),
            "window_id": self.window_id,
            "root_orient": self.root_orient.to_wire(),
            "hand_pose": self.hand_pose.to_wire(),
            "trans": self.trans.to_wire(),
            "betas": self.betas.to_wire(),
            "observed": self.observed.to_wire(),
            "inferred": self.inferred.to_wire(),
            "uncertainty": self.uncertainty.to_wire(),
            "timestamps_s": self.timestamps_s.to_wire(),
            "spatial": self.spatial.to_wire(),
            "model_revision": self.model_revision,
            "trace": self.trace.to_wire(),
            "adapter_notes": list(self.adapter_notes),
            **({"server_identity": self.server_identity.to_wire()} if self.server_identity is not None else {}),
            **({"batch_diagnostics": dict(self.batch_diagnostics)} if self.batch_diagnostics is not None else {}),
        }
