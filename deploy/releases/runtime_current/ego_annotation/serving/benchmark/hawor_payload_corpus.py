"""Build a real-frame, typed HaWoR track-chunk benchmark corpus.

The HaWoR API does not accept a generic video frame.  A work unit is a sixteen-frame
normalized hand-crop sequence coupled to source-crop geometry, visibility evidence,
UniDepth intrinsics/scale, and a DROID trajectory.  This builder combines only two
existing measurements:

* V22 raw-frame manifests supply decoded source RGB frames; and
* preserved ``hawor.infer_tracks`` payload descriptors supply the hand-track geometry
  and metric-camera evidence already produced upstream.

It refuses to invent either source.  The source JPEGs are sampled through the
historically-attested crop transforms, then ImageNet-normalized into the exact public
``[16,3,256,256] float32`` tensor.  Every other typed input is retained from the
historic request's binary/metadata contract.  The published descriptor is path-free
at the request boundary and is atomically moved into a fresh destination only after
it re-loads and reconstructs every ``TrackChunkRequest`` successfully.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from ego_annotation.serving.benchmark.manifest import PAYLOAD_SOURCE_SCHEMA, _hash_parts, load_payload_manifest
from ego_annotation.serving.contracts import ContractValidationError, ImageSize, Ownership, PixelTransform, TensorPayload, reject_filesystem_fields
from ego_annotation.serving.hawor import IMG_NORM_MEAN, IMG_NORM_STD
from ego_annotation.serving.hawor_contracts import (
    HAWOR_CHUNK_LEN,
    HAWOR_CROP_CHANNELS,
    HAWOR_CROP_DTYPE,
    HAWOR_CROP_H,
    HAWOR_CROP_W,
    CropSourceTransform,
    DroidCameraEvidence,
    FrameObservation,
    HandSide,
    OcclusionState,
    TrackChunkRequest,
    UniDepthScaleK,
)
from ego_annotation.serving.hawor_transport import track_chunk_gateway_request
from ego_annotation.serving.router import ModelApiName

SOURCE_WIDTH = 1920
SOURCE_HEIGHT = 1080
MODEL_REVISION_DEFAULT = "hawor-v1"


class CorpusBuildError(ValueError):
    """The requested corpus cannot truthfully be constructed."""


@dataclass(frozen=True)
class SourceSpec:
    """One public V22 source identity, frames, and optional native evidence root."""

    manifest_path: Path
    source_id: str
    evidence_root: Path | None = None


@dataclass(frozen=True)
class SourceFrame:
    source_id: str
    frame_index: int
    timestamp_s: float
    jpeg_path: Path

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.jpeg_path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class CorpusBuildResult:
    output_root: Path
    descriptor_path: Path
    item_count: int
    descriptor_sha256: str
    payload_hashes: tuple[str, ...]


def _path_free_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(token in value for token in ("/", "\\", "~")):
        raise CorpusBuildError(f"{label} must be a non-empty path-free identifier")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CorpusBuildError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise CorpusBuildError(f"{label} must be finite")
    return result


def _frame_index(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorpusBuildError(f"{label} must be a non-negative integer")
    return value


def _load_source_frames(specs: Sequence[SourceSpec]) -> dict[str, dict[int, SourceFrame]]:
    if not specs:
        raise CorpusBuildError("at least one V22 raw-frame manifest is required")
    result: dict[str, dict[int, SourceFrame]] = {}
    for spec in specs:
        source_id = _path_free_identifier(spec.source_id, "source_id")
        if source_id in result:
            raise CorpusBuildError(f"duplicate source_id {source_id!r}")
        path = Path(spec.manifest_path).resolve()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusBuildError(f"cannot read V22 raw-frame manifest {path}") from exc
        if not isinstance(raw, Mapping) or raw.get("schema") not in (None, "v22_raw_frame_manifest.v0"):
            raise CorpusBuildError(f"{path} is not a supported V22 raw-frame manifest")
        if raw.get("fps") != 30 and raw.get("fps") != 30.0:
            raise CorpusBuildError(f"{path} must declare the V22 30fps source cadence")
        frames = raw.get("frames")
        if not isinstance(frames, list) or not frames:
            raise CorpusBuildError(f"{path} must contain non-empty frames")
        if raw.get("frame_count") is not None and raw["frame_count"] != len(frames):
            raise CorpusBuildError(f"{path} frame_count does not match frames")
        indexed: dict[int, SourceFrame] = {}
        for position, row in enumerate(frames):
            if not isinstance(row, Mapping):
                raise CorpusBuildError(f"{path.name}.frames[{position}] must be an object")
            index = _frame_index(row.get("source_frame_idx", row.get("frame_idx")), f"{path.name}.frames[{position}].frame_index")
            if index in indexed:
                raise CorpusBuildError(f"{path.name} duplicates source frame index {index}")
            width = row.get("source_width", row.get("manifest_width", raw.get("source_width", raw.get("manifest_width"))))
            height = row.get("source_height", row.get("manifest_height", raw.get("source_height", raw.get("manifest_height"))))
            if (width, height) != (SOURCE_WIDTH, SOURCE_HEIGHT):
                raise CorpusBuildError(f"{path.name}.frames[{position}] must declare {SOURCE_WIDTH}x{SOURCE_HEIGHT}")
            timestamp = _finite_number(row.get("source_time_s", row.get("time_s")), f"{path.name}.frames[{position}].source_timestamp_s")
            rgb = row.get("rgb")
            if not isinstance(rgb, str) or not rgb:
                raise CorpusBuildError(f"{path.name}.frames[{position}].rgb must be a JPEG path")
            jpeg = Path(rgb)
            if not jpeg.is_absolute():
                jpeg = path.parent / jpeg
            try:
                jpeg = jpeg.resolve(strict=True)
            except OSError as exc:
                raise CorpusBuildError(f"source JPEG is unavailable for {source_id}:{index}") from exc
            indexed[index] = SourceFrame(source_id, index, timestamp, jpeg)
        result[source_id] = indexed
    return result


def _template_request(item: Any) -> tuple[TrackChunkRequest, dict[str, TensorPayload]]:
    """Decode one preserved generic payload descriptor into the typed HaWoR request.

    The historical crop tensor is intentionally *not* reused.  It only proves the
    upstream transform/evidence contract.  Real V22 frames replace that one part.
    """
    if item.api_name is not ModelApiName.HAWOR_INFER_TRACKS:
        raise CorpusBuildError(f"historic payload {item.item_id!r} is not hawor.infer_tracks")
    parts = {part.name: part for part in item.parts}
    required = {"crop_batch", "droid_poses", "droid_timestamps"}
    if set(parts) != required:
        raise CorpusBuildError(f"historic payload {item.item_id!r} parts must be exactly {sorted(required)}")
    crop = parts["crop_batch"]
    if crop.shape != (HAWOR_CHUNK_LEN, HAWOR_CROP_CHANNELS, HAWOR_CROP_H, HAWOR_CROP_W) or crop.dtype != HAWOR_CROP_DTYPE:
        raise CorpusBuildError(f"historic payload {item.item_id!r} has an incompatible crop_batch contract")
    metadata = dict(item.metadata)
    try:
        droid_wire = metadata["droid_evidence"]
        if not isinstance(droid_wire, Mapping):
            raise ContractValidationError("droid_evidence must be an object")
        droid = DroidCameraEvidence(
            poses_world_from_camera=TensorPayload(parts["droid_poses"].data, parts["droid_poses"].shape, parts["droid_poses"].dtype),
            timestamps_s=TensorPayload(parts["droid_timestamps"].data, parts["droid_timestamps"].shape, parts["droid_timestamps"].dtype),
            metric_scale=float(droid_wire["metric_scale"]), scale_residual=float(droid_wire["scale_residual"]),
            scale_confidence=float(droid_wire["scale_confidence"]), source=str(droid_wire["source"]),
        )
        request = TrackChunkRequest(
            ownership=item.ownership,
            track_id=str(metadata["track_id"]), side=HandSide(metadata["side"]),
            crop_batch=TensorPayload(crop.data, crop.shape, crop.dtype),
            crop_transforms=tuple(CropSourceTransform.from_mapping(value) for value in metadata["crop_transforms"]),
            observations=tuple(FrameObservation.from_mapping(value) for value in metadata["observations"]),
            unidepth=UniDepthScaleK.from_mapping(metadata["unidepth"]), droid_evidence=droid,
            model_revision=item.model_revision,
            options=tuple(sorted((str(key), str(value)) for key, value in dict(metadata.get("options", {})).items())),
        )
    except (KeyError, TypeError, ValueError, ContractValidationError) as exc:
        raise CorpusBuildError(f"historic payload {item.item_id!r} does not satisfy the typed HaWoR request contract: {exc}") from exc
    return request, {
        "droid_poses": droid.poses_world_from_camera,
        "droid_timestamps": droid.timestamps_s,
    }


def _crop_from_real_frame(frame: SourceFrame, transform: CropSourceTransform) -> np.ndarray:
    """Apply a historical HaWoR crop transform to one actual V22 RGB frame."""
    if (transform.source_size.width, transform.source_size.height) != (SOURCE_WIDTH, SOURCE_HEIGHT):
        raise CorpusBuildError(f"crop transform for {frame.source_id}:{frame.frame_index} is not on the V22 source grid")
    side = float(transform.scale) * 200.0
    left = float(transform.center[0]) - side / 2.0
    top = float(transform.center[1]) - side / 2.0
    try:
        with Image.open(frame.jpeg_path) as image:
            if image.format != "JPEG" or image.size != (SOURCE_WIDTH, SOURCE_HEIGHT):
                raise CorpusBuildError(f"source {frame.source_id}:{frame.frame_index} is not a {SOURCE_WIDTH}x{SOURCE_HEIGHT} JPEG")
            # PIL's AFFINE coefficients map each output pixel to source coordinates.
            crop = image.convert("RGB").transform(
                (HAWOR_CROP_W, HAWOR_CROP_H), Image.Transform.AFFINE,
                (side / HAWOR_CROP_W, 0.0, left, 0.0, side / HAWOR_CROP_H, top),
                resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0),
            )
    except CorpusBuildError:
        raise
    except (OSError, ValueError) as exc:
        raise CorpusBuildError(f"cannot decode source JPEG for {frame.source_id}:{frame.frame_index}") from exc
    if transform.do_flip:
        crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    rgb = np.asarray(crop, dtype=np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    normalized = (chw - np.asarray(IMG_NORM_MEAN, dtype=np.float32)[:, None, None]) / np.asarray(IMG_NORM_STD, dtype=np.float32)[:, None, None]
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _rebuild_request(template: TrackChunkRequest, droid_arrays: Mapping[str, TensorPayload], source_frames: Mapping[str, Mapping[int, SourceFrame]], index: int, manifest_id: str, job_id: str) -> TrackChunkRequest:
    frames = source_frames.get(template.ownership.source_id)
    if frames is None:
        raise CorpusBuildError(f"no V22 source manifest for historic source_id {template.ownership.source_id!r}")
    crops: list[np.ndarray] = []
    for observation, transform in zip(template.observations, template.crop_transforms):
        source = frames.get(observation.frame_index)
        if source is None:
            raise CorpusBuildError(f"historic chunk {template.track_id!r} requests absent source frame {observation.frame_index}")
        if abs(source.timestamp_s - observation.source_timestamp_s) > 1e-6:
            raise CorpusBuildError(f"historic chunk {template.track_id!r} timestamp disagrees with V22 frame {observation.frame_index}")
        crops.append(_crop_from_real_frame(source, transform))
    crop_batch = np.stack(crops, axis=0)
    item_id = f"{manifest_id}-item-{index:05d}"
    ownership = Ownership(
        request_id=f"{manifest_id}-request-{index:05d}", job_id=job_id, item_id=item_id,
        stage_id="hawor.infer_tracks", source_id=template.ownership.source_id,
        source_timestamp_s=template.observations[0].source_timestamp_s,
    )
    droid = template.droid_evidence
    assert droid is not None
    return TrackChunkRequest(
        ownership=ownership, track_id=template.track_id, side=template.side,
        crop_batch=TensorPayload(crop_batch.tobytes(), tuple(crop_batch.shape), HAWOR_CROP_DTYPE),
        crop_transforms=template.crop_transforms, observations=template.observations,
        unidepth=template.unidepth,
        droid_evidence=DroidCameraEvidence(
            droid_arrays["droid_poses"], droid_arrays["droid_timestamps"], droid.metric_scale,
            droid.scale_residual, droid.scale_confidence, droid.source,
        ),
        model_revision=template.model_revision, options=template.options,
    )


def _request_descriptor_item(request: TrackChunkRequest, index: int, manifest_id: str, raw_dir: Path, source_frames: Mapping[str, Mapping[int, SourceFrame]]) -> dict[str, object]:
    gateway = track_chunk_gateway_request(request)
    by_name = {part.name: part for part in gateway.parts}
    expected_parts = ("crop_batch", "droid_poses", "droid_timestamps")
    if tuple(part.name for part in gateway.parts) != expected_parts:
        raise CorpusBuildError("HaWoR gateway part contract changed unexpectedly")
    part_rows: list[dict[str, object]] = []
    hash_inputs: list[tuple[str, bytes, tuple[int, ...], str]] = []
    for name in expected_parts:
        part = by_name[name]
        data = bytes(part.data)
        relative = f"raw/{index:05d}.{name}.bin"
        (raw_dir.parent / relative).write_bytes(data)
        part_rows.append({"name": name, "file": relative, "shape": list(part.shape), "dtype": part.dtype})
        hash_inputs.append((name, data, part.shape, part.dtype))
    metadata = dict(gateway.metadata)
    # Tensor bytes are carried in the named arrays only.  Keep only scalar/semantic
    # evidence in JSON so the request remains path-free and transport-invariant.
    droid = metadata.get("droid_evidence")
    if not isinstance(droid, Mapping):
        raise CorpusBuildError("reconstructed HaWoR request omitted DROID evidence metadata")
    metadata["droid_evidence"] = {
        key: droid[key] for key in ("metric_scale", "scale_residual", "scale_confidence", "source")
    }
    metadata.update({
        "source_frame_indices": [observation.frame_index for observation in request.observations],
        "source_timestamps_s": [observation.source_timestamp_s for observation in request.observations],
        "source_frame_sha256": [source_frames[request.ownership.source_id][observation.frame_index].sha256 for observation in request.observations],
        "crop_sha256": hashlib.sha256(bytes(by_name["crop_batch"].data)).hexdigest(),
    })
    reject_filesystem_fields(metadata, context="generated HaWoR request metadata")
    item_id = f"{manifest_id}-item-{index:05d}"
    return {
        "item_id": item_id,
        "ownership": request.ownership.to_wire(),
        "parts": part_rows,
        "spatial": None,
        "model_revision": request.model_revision,
        "work_units": request.work_units,
        "source_timestamp_s": request.ownership.source_timestamp_s,
        "payload_hash": _hash_parts(hash_inputs),
        "metadata": metadata,
    }


def _validate_descriptor(path: Path, expected_count: int) -> tuple[str, ...]:
    try:
        manifest = load_payload_manifest(path, expected_api=ModelApiName.HAWOR_INFER_TRACKS, limit=expected_count)
    except ValueError as exc:
        raise CorpusBuildError(f"published corpus fails generic payload validation: {exc}") from exc
    if len(manifest.items) != expected_count:
        raise CorpusBuildError("published corpus item count changed during validation")
    hashes: list[str] = []
    for item in manifest.items:
        template, _ = _template_request(item)
        gateway = track_chunk_gateway_request(template)
        if tuple(part.name for part in gateway.parts) != ("crop_batch", "droid_poses", "droid_timestamps"):
            raise CorpusBuildError("published item does not reconstruct the exact HaWoR part contract")
        reject_filesystem_fields(gateway.metadata, context="validated HaWoR request metadata")
        hashes.append(item.payload_hash)
    if len(set(hashes)) != len(hashes):
        raise CorpusBuildError("duplicate typed HaWoR payload hashes are forbidden")
    return tuple(hashes)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v22_job_id(root: Path) -> str:
    try:
        request = json.loads((root / "requests/hawor.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusBuildError(f"V22 evidence root lacks preserved requests/hawor.json provenance: {root}") from exc
    return _path_free_identifier(request.get("job_id"), "V22 job_id")


def _native_v22_templates(spec: SourceSpec, frames: Mapping[int, SourceFrame]) -> tuple[tuple[TrackChunkRequest, dict[str, TensorPayload]], ...]:
    """Derive requests from the real V22 inputs consumed by the production hand path.

    WiLoR detections provide crop centres/scales and visibility; dense DROID provides
    the camera stream; the persisted UniDepth calibration provides K.  Missing input
    excludes a candidate chunk rather than filling it.  The DROID scale is explicitly
    marked as the persisted unit gauge with zero confidence: this is a load input,
    not a claim that the historic monocular trajectory is metric ground truth.
    """
    if spec.evidence_root is None:
        raise CorpusBuildError("a historic payload descriptor or --evidence-root is required for every source")
    root = Path(spec.evidence_root).resolve()
    paths = {
        "wilor": root / "measurements/hand_candidates/wilor_v21/wilor_raw_hands.json",
        "droid": root / "measurements/camera_trajectory/droid_full_frame/droid_dense_trajectory.json",
        "calibration": root / "state/calibration/v19_camera_calibration_contract.json",
    }
    try:
        payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusBuildError(f"V22 evidence root lacks readable WiLoR/DROID/calibration inputs: {root}") from exc
    wilor_frames = payloads["wilor"].get("frames")
    droid_frames = payloads["droid"].get("frames")
    intrinsics = payloads["calibration"].get("intrinsics_fx_fy_cx_cy")
    if not isinstance(wilor_frames, list) or not isinstance(droid_frames, list) or not isinstance(intrinsics, list) or len(intrinsics) != 4:
        raise CorpusBuildError(f"V22 evidence root has incompatible WiLoR/DROID/calibration schema: {root}")
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    if min(fx, fy) <= 0:
        raise CorpusBuildError("V22 calibration has non-positive focal length")
    droid_by_frame: dict[int, tuple[tuple[float, ...], ...]] = {}
    for row in droid_frames:
        if not isinstance(row, Mapping) or "frame_idx" not in row or "T_world_camera" not in row:
            continue
        try:
            index = _frame_index(row["frame_idx"], "droid.frame_idx")
            matrix = tuple(tuple(float(value) for value in matrix_row) for matrix_row in row["T_world_camera"])
            if len(matrix) != 4 or any(len(matrix_row) != 4 for matrix_row in matrix):
                continue
            droid_by_frame[index] = matrix
        except (TypeError, ValueError):
            continue
    detections: dict[HandSide, dict[int, tuple[float, tuple[float, float, float, float]]]] = {HandSide.LEFT: {}, HandSide.RIGHT: {}}
    for row in wilor_frames:
        if not isinstance(row, Mapping):
            continue
        try:
            index = _frame_index(row["frame_idx"], "wilor.frame_idx")
            hands = row.get("raw_hands", [])
            if not isinstance(hands, list):
                continue
            for hand in hands:
                if not isinstance(hand, Mapping):
                    continue
                side = HandSide(hand["side"])
                confidence = float(hand["detector_score"])
                box = tuple(float(value) for value in hand["bbox_xyxy"])
                if len(box) != 4 or not (0.0 <= confidence <= 1.0) or box[2] <= box[0] or box[3] <= box[1]:
                    continue
                prior = detections[side].get(index)
                if prior is None or confidence > prior[0]:
                    detections[side][index] = (confidence, box)
        except (KeyError, TypeError, ValueError):
            continue
    source_id = _path_free_identifier(spec.source_id, "source_id")
    templates: list[tuple[TrackChunkRequest, dict[str, TensorPayload]]] = []
    # Non-overlapping windows avoid reusing an actual source frame as benchmark load.
    for side, by_frame in detections.items():
        for start in range(0, max(frames) + 1, HAWOR_CHUNK_LEN):
            indices = tuple(range(start, start + HAWOR_CHUNK_LEN))
            if any(index not in frames or index not in by_frame or index not in droid_by_frame for index in indices):
                continue
            observations = tuple(
                FrameObservation(index, frames[index].timestamp_s, OcclusionState.VISIBLE, by_frame[index][0], side)
                for index in indices
            )
            transforms = []
            for index in indices:
                _, (x0, y0, x1, y1) = by_frame[index]
                transforms.append(CropSourceTransform(
                    center=((x0 + x1) / 2.0, (y0 + y1) / 2.0), scale=max(x1 - x0, y1 - y0) / 200.0,
                    img_focal=(fx * fy) ** 0.5, img_center=(cx, cy), do_flip=side is HandSide.LEFT,
                    source_size=ImageSize(SOURCE_WIDTH, SOURCE_HEIGHT),
                    pixel_transform=PixelTransform.identity(),
                ))
            poses = np.asarray([droid_by_frame[index] for index in indices], dtype=np.float32)
            timestamps = np.asarray([frames[index].timestamp_s for index in indices], dtype=np.float64)
            droid = DroidCameraEvidence(
                TensorPayload(poses.tobytes(), poses.shape, "float32"), TensorPayload(timestamps.tobytes(), timestamps.shape, "float64"),
                1.0, 0.0, 0.0, "derived-v22-droid-unit-gauge",
            )
            unidepth = UniDepthScaleK(
                ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), (fx * fy) ** 0.5, (cx, cy),
                ImageSize(SOURCE_WIDTH, SOURCE_HEIGHT), 1.0, "derived-v22-unidepth-calibration",
            )
            # This placeholder is never published or inferred: it lets the exact
            # public dataclass validate the evidence before real crops replace it.
            placeholder = TensorPayload(bytes(HAWOR_CHUNK_LEN * HAWOR_CROP_CHANNELS * HAWOR_CROP_H * HAWOR_CROP_W * 4), (HAWOR_CHUNK_LEN, HAWOR_CROP_CHANNELS, HAWOR_CROP_H, HAWOR_CROP_W), HAWOR_CROP_DTYPE)
            request = TrackChunkRequest(
                Ownership(f"v22-template-{source_id}-{side.value}-{start}", "v22-evidence", f"v22-template-{start}", "hawor.infer_tracks", source_id, source_timestamp_s=frames[start].timestamp_s),
                f"wilor-{side.value}-{start:06d}", side, placeholder, tuple(transforms), observations, unidepth, droid, MODEL_REVISION_DEFAULT,
            )
            templates.append((request, {"droid_poses": droid.poses_world_from_camera, "droid_timestamps": droid.timestamps_s}))
    if not templates:
        raise CorpusBuildError(f"V22 evidence root yielded no complete 16-frame HaWoR chunks: {root}")
    return tuple(templates)


def build_hawor_payload_corpus(*, sources: Sequence[SourceSpec], historic_payload_manifest: str | Path | None = None, output_root: str | Path, count: int | None = None, manifest_id: str = "hawor-v22-track-corpus", job_id: str = "hawor-envelope-soak") -> CorpusBuildResult:
    """Publish a fresh descriptor of distinct real-frame HaWoR temporal track chunks."""
    manifest_id = _path_free_identifier(manifest_id, "manifest_id")
    job_id = _path_free_identifier(job_id, "job_id")
    source_frames = _load_source_frames(sources)
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count <= 0):
        raise CorpusBuildError("count must be a positive integer when supplied")
    templates: tuple[tuple[TrackChunkRequest, dict[str, TensorPayload]], ...]
    evidence_source: dict[str, object]
    if historic_payload_manifest is not None:
        try:
            historic = load_payload_manifest(historic_payload_manifest, expected_api=ModelApiName.HAWOR_INFER_TRACKS)
        except ValueError as exc:
            raise CorpusBuildError(f"cannot load preserved historic HaWoR requests: {exc}") from exc
        templates = tuple(_template_request(item) for item in historic.items)
        evidence_source = {"kind": "preserved-hawor-request-descriptor", "descriptor_sha256": _file_sha256(Path(historic_payload_manifest))}
    else:
        if any(spec.evidence_root is None for spec in sources):
            raise CorpusBuildError("--evidence-root is required for every source when no historic payload descriptor is supplied")
        templates = tuple(template for spec in sources for template in _native_v22_templates(spec, source_frames[spec.source_id]))
        evidence_source = {
            "kind": "derived-from-v22-pipeline-inputs",
            "derivation_chain": ["V22 raw RGB frames", "WiLoR hand detections", "dense DROID T_world_camera", "UniDepth calibration"],
            "sources": [{"source_id": spec.source_id, "job_id": _v22_job_id(Path(spec.evidence_root or ".")), "manifest_sha256": _file_sha256(Path(spec.manifest_path)),
                         "wilor_droid_calibration_sha256": {name: _file_sha256(Path(spec.evidence_root or ".") / relative) for name, relative in {
                             "wilor": "measurements/hand_candidates/wilor_v21/wilor_raw_hands.json",
                             "droid": "measurements/camera_trajectory/droid_full_frame/droid_dense_trajectory.json",
                             "calibration": "state/calibration/v19_camera_calibration_contract.json"}.items()}} for spec in sources],
        }
    selected = templates if count is None else templates[:count]
    if count is not None and len(selected) != count:
        raise CorpusBuildError(f"HaWoR evidence source has {len(templates)} complete distinct chunks but {count} were requested")
    target = Path(output_root)
    if target.exists() or target.is_symlink():
        raise CorpusBuildError(f"output root already exists; refusing reuse or mutation: {target}")
    parent = target.parent.resolve()
    if not parent.is_dir():
        raise CorpusBuildError(f"output parent does not exist: {parent}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=parent))
    try:
        raw_dir = temporary / "raw"
        raw_dir.mkdir()
        items: list[dict[str, object]] = []
        for index, (template, droid_arrays) in enumerate(selected):
            rebuilt = _rebuild_request(template, droid_arrays, source_frames, index, manifest_id, job_id)
            # Round-trip through the exact public typed parser before publication.
            typed_wire = rebuilt.to_wire()
            if TrackChunkRequest.from_wire(typed_wire) != rebuilt:
                raise CorpusBuildError("typed HaWoR request parser changed reconstructed request")
            items.append(_request_descriptor_item(rebuilt, index, manifest_id, raw_dir, source_frames))
        descriptor = {
            "schema": PAYLOAD_SOURCE_SCHEMA,
            "manifest_id": manifest_id,
            "api_name": ModelApiName.HAWOR_INFER_TRACKS.value,
            "model_revision": MODEL_REVISION_DEFAULT if not items else items[0]["model_revision"],
            "item_count": len(items),
            "source_contract": "v22-30fps-real-rgb+typed-hawor-track-evidence.v2",
            "evidence_source": evidence_source,
            "exclusions": {"policy": "missing frame, WiLoR detection, DROID pose, or calibration excludes a chunk; no filling", "complete_chunk_count": len(templates), "published_chunk_count": len(items)},
            "items": items,
        }
        descriptor_path = temporary / "hawor.infer_tracks.json"
        descriptor_path.write_text(json.dumps(descriptor, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        payload_hashes = _validate_descriptor(descriptor_path, len(items))
        digest = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return CorpusBuildResult(target, target / "hawor.infer_tracks.json", len(items), digest, payload_hashes)


def parse_source_specs(manifest_paths: Iterable[str | Path], source_ids: Iterable[str], evidence_roots: Iterable[str | Path] | None = None) -> tuple[SourceSpec, ...]:
    paths, ids, roots = tuple(manifest_paths), tuple(source_ids), tuple(evidence_roots or ())
    if not paths or len(paths) != len(ids) or (roots and len(roots) != len(paths)):
        raise CorpusBuildError("--manifest and --source-id must be supplied equally; --evidence-root is omitted or paired equally")
    return tuple(SourceSpec(Path(path), source_id, Path(roots[index]) if roots else None) for index, (path, source_id) in enumerate(zip(paths, ids)))


__all__ = [
    "CorpusBuildError", "CorpusBuildResult", "MODEL_REVISION_DEFAULT", "SourceSpec",
    "build_hawor_payload_corpus", "parse_source_specs",
]
