"""Publish path-free Hands and WiLoR benchmark requests from preserved V22 evidence.

Hands payload pixels are decoded from the V22 RGB frame named by preserved request
ownership and resampled through an explicit 1920x1080 -> 960x540 transform.  WiLoR
crops are retained only from preserved typed WiLoR requests, whose crop geometry and
ownership are parsed again through the public request contract.  Neither branch
creates a detector box, crop, RGB frame, or ground truth from a heuristic.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from ego_annotation.serving.benchmark.manifest import PAYLOAD_SOURCE_SCHEMA, _hash_parts, load_payload_manifest
from ego_annotation.serving.contracts import (
    HandsDetectRequest, ImageSize, Ownership, PixelTransform, SpatialMetadata,
    TensorPayload, WiLoRReconstructRequest, reject_filesystem_fields,
)
from ego_annotation.serving.hands_transport import hands_detect_gateway_request, wilor_reconstruct_gateway_request
from ego_annotation.serving.router import ModelApiName

SOURCE_SIZE = (1920, 1080)
CANONICAL_SIZE = (960, 540)


class CorpusBuildError(ValueError):
    """The requested corpus cannot be reconstructed from real preserved inputs."""


@dataclass(frozen=True)
class SourceSpec:
    manifest_path: Path
    source_id: str


@dataclass(frozen=True)
class SourceFrame:
    source_id: str
    timestamp_s: float
    jpeg_path: Path

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.jpeg_path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class CorpusBuildResult:
    output_root: Path
    hands_descriptor_path: Path
    wilor_descriptor_path: Path
    hands_count: int
    wilor_count: int
    descriptor_sha256: Mapping[str, str]
    exclusions: Mapping[str, int]


def _path_free(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or any(token in value for token in ("/", "\\", "~")):
        raise CorpusBuildError(f"{label} must be a non-empty path-free identifier")
    return value


def _load_frames(sources: Sequence[SourceSpec]) -> dict[tuple[str, float], SourceFrame]:
    if not sources:
        raise CorpusBuildError("at least one V22 raw-frame manifest is required")
    frames: dict[tuple[str, float], SourceFrame] = {}
    ids: set[str] = set()
    for spec in sources:
        source_id = _path_free(spec.source_id, "source_id")
        if source_id in ids:
            raise CorpusBuildError(f"duplicate source_id {source_id!r}")
        ids.add(source_id)
        manifest = Path(spec.manifest_path).resolve()
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusBuildError(f"cannot read V22 frame manifest {manifest}") from exc
        if not isinstance(raw, Mapping) or raw.get("schema") not in (None, "v22_raw_frame_manifest.v0") or raw.get("fps") not in (30, 30.0):
            raise CorpusBuildError(f"{manifest} is not a supported 30fps V22 raw-frame manifest")
        rows = raw.get("frames")
        if not isinstance(rows, list) or not rows:
            raise CorpusBuildError(f"{manifest} has no frame rows")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise CorpusBuildError(f"{manifest.name}.frames[{index}] must be an object")
            timestamp = row.get("source_time_s", row.get("time_s"))
            if isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)):
                raise CorpusBuildError(f"{manifest.name}.frames[{index}] has no numeric source timestamp")
            path_value = row.get("rgb")
            if not isinstance(path_value, str) or not path_value:
                raise CorpusBuildError(f"{manifest.name}.frames[{index}] has no RGB JPEG")
            jpeg = Path(path_value)
            if not jpeg.is_absolute():
                jpeg = manifest.parent / jpeg
            try:
                jpeg = jpeg.resolve(strict=True)
            except OSError as exc:
                raise CorpusBuildError(f"V22 RGB JPEG is unavailable: {source_id}:{timestamp}") from exc
            width = row.get("source_width", raw.get("source_width", SOURCE_SIZE[0]))
            height = row.get("source_height", raw.get("source_height", SOURCE_SIZE[1]))
            if (width, height) != SOURCE_SIZE:
                raise CorpusBuildError(f"V22 RGB must be {SOURCE_SIZE[0]}x{SOURCE_SIZE[1]}")
            key = (source_id, float(timestamp))
            if key in frames:
                raise CorpusBuildError(f"duplicate V22 frame timestamp for {source_id}:{timestamp}")
            frames[key] = SourceFrame(source_id, float(timestamp), jpeg)
    return frames


def _frame_for(frames: Mapping[tuple[str, float], SourceFrame], ownership: Ownership) -> SourceFrame | None:
    if ownership.source_timestamp_s is None:
        return None
    exact = frames.get((ownership.source_id, float(ownership.source_timestamp_s)))
    if exact is not None:
        return exact
    # JSON decimal serialization can move a 30fps timestamp by a few ulps, never
    # enough to select an adjacent frame.
    candidates = [frame for (source_id, timestamp), frame in frames.items()
                  if source_id == ownership.source_id and abs(timestamp - ownership.source_timestamp_s) <= 1e-6]
    return candidates[0] if len(candidates) == 1 else None


def _hands_template(item: Any) -> HandsDetectRequest:
    if item.api_name is not ModelApiName.HANDS_DETECT or len(item.parts) != 1 or item.parts[0].name != "rgb":
        raise CorpusBuildError(f"preserved item {item.item_id!r} is not a typed hands.detect request")
    try:
        return HandsDetectRequest(
            ownership=item.ownership,
            rgb=TensorPayload(item.parts[0].data, item.parts[0].shape, item.parts[0].dtype),
            spatial=(item.spatial if isinstance(item.spatial, SpatialMetadata) else SpatialMetadata.from_mapping(item.spatial or {})), model_revision=item.model_revision,
            options=tuple(sorted((str(k), str(v)) for k, v in dict(item.metadata.get("options", {})).items())),
        )
    except Exception as exc:
        raise CorpusBuildError(f"preserved hands request {item.item_id!r} fails typed parser: {exc}") from exc


def _wilor_template(item: Any) -> WiLoRReconstructRequest:
    if item.api_name is not ModelApiName.WILOR_RECONSTRUCT or len(item.parts) != 1 or item.parts[0].name != "crop":
        raise CorpusBuildError(f"preserved item {item.item_id!r} is not a typed wilor.reconstruct request")
    wire = dict(item.metadata)
    wire.update({"ownership": item.ownership.to_wire(), "model_revision": item.model_revision,
                 "crop": TensorPayload(item.parts[0].data, item.parts[0].shape, item.parts[0].dtype).to_wire()})
    try:
        return WiLoRReconstructRequest.from_wire(wire)
    except Exception as exc:
        raise CorpusBuildError(f"preserved WiLoR request {item.item_id!r} fails typed parser: {exc}") from exc


def _resize_v22_rgb(frame: SourceFrame) -> bytes:
    try:
        with Image.open(frame.jpeg_path) as image:
            if image.format != "JPEG" or image.size != SOURCE_SIZE:
                raise CorpusBuildError(f"V22 source {frame.source_id}:{frame.timestamp_s} is not a {SOURCE_SIZE} JPEG")
            rgb = image.convert("RGB").resize(CANONICAL_SIZE, Image.Resampling.BILINEAR)
    except CorpusBuildError:
        raise
    except (OSError, ValueError) as exc:
        raise CorpusBuildError(f"cannot decode V22 RGB source {frame.source_id}:{frame.timestamp_s}") from exc
    return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)).tobytes()


def _hands_rebuilt(template: HandsDetectRequest, frame: SourceFrame, index: int, manifest_id: str, job_id: str) -> HandsDetectRequest:
    width, height = CANONICAL_SIZE
    source_width, source_height = SOURCE_SIZE
    transform = PixelTransform(
        ((width / source_width, 0.0, 0.0), (0.0, height / source_height, 0.0), (0.0, 0.0, 1.0)),
        ((source_width / width, 0.0, 0.0), (0.0, source_height / height, 0.0), (0.0, 0.0, 1.0)), "resize", None, None,
    )
    spatial = SpatialMetadata(ImageSize(source_width, source_height), ImageSize(width, height), "RGB", transform, template.spatial.K_px)
    ownership = Ownership(f"{manifest_id}-hands-{index:05d}", job_id, f"{manifest_id}-hands-{index:05d}", "hands.detect", frame.source_id, source_timestamp_s=frame.timestamp_s)
    return HandsDetectRequest(ownership, TensorPayload(_resize_v22_rgb(frame), (height, width, 3), "uint8"), spatial, template.model_revision, template.options)


def _wilor_reowned(template: WiLoRReconstructRequest, frame: SourceFrame, index: int, manifest_id: str, job_id: str) -> WiLoRReconstructRequest:
    ownership = Ownership(f"{manifest_id}-wilor-{index:05d}", job_id, f"{manifest_id}-wilor-{index:05d}", "wilor.reconstruct", frame.source_id, source_timestamp_s=frame.timestamp_s)
    return WiLoRReconstructRequest(ownership, template.crop, template.handedness, template.box_center, template.box_size, template.img_size, template.model_revision, template.source_K_px, template.options)


def _write_item(request: HandsDetectRequest | WiLoRReconstructRequest, *, index: int, raw_dir: Path, api_name: ModelApiName, frame: SourceFrame) -> dict[str, object]:
    if api_name is ModelApiName.HANDS_DETECT:
        if not isinstance(request, HandsDetectRequest):
            raise CorpusBuildError("Hands descriptor received a non-Hands request")
        gateway = hands_detect_gateway_request(request)
    else:
        if not isinstance(request, WiLoRReconstructRequest):
            raise CorpusBuildError("WiLoR descriptor received a non-WiLoR request")
        gateway = wilor_reconstruct_gateway_request(request)
    parts: list[dict[str, object]] = []
    hash_parts: list[tuple[str, bytes, tuple[int, ...], str]] = []
    for part in gateway.parts:
        payload = bytes(part.data)
        relative = f"raw/{api_name.value.split('.')[0]}-{index:05d}.{part.name}.bin"
        (raw_dir.parent / relative).write_bytes(payload)
        parts.append({"name": part.name, "file": relative, "shape": list(part.shape), "dtype": part.dtype})
        hash_parts.append((part.name, payload, part.shape, part.dtype))
    metadata = dict(gateway.metadata)
    metadata.update({"v22_source_rgb_sha256": frame.sha256, "v22_source_timestamp_s": frame.timestamp_s})
    reject_filesystem_fields(metadata, context="generated Hands/WiLoR request metadata")
    return {"item_id": request.ownership.item_id, "ownership": request.ownership.to_wire(), "parts": parts,
            "spatial": gateway.spatial.to_wire() if gateway.spatial else None, "model_revision": request.model_revision,
            "work_units": request.work_units, "source_timestamp_s": request.ownership.source_timestamp_s,
            "payload_hash": _hash_parts(hash_parts), "metadata": metadata}


def _validate_descriptor(path: Path, api_name: ModelApiName) -> tuple[str, ...]:
    manifest = load_payload_manifest(path, expected_api=api_name)
    hashes: list[str] = []
    for item in manifest.items:
        if api_name is ModelApiName.HANDS_DETECT:
            request = _hands_template(item)
            gateway = hands_detect_gateway_request(request)
        else:
            request = _wilor_template(item)
            gateway = wilor_reconstruct_gateway_request(request)
        reject_filesystem_fields(gateway.metadata, context="validated Hands/WiLoR request metadata")
        if len(gateway.parts) != 1:
            raise CorpusBuildError("published request no longer has exactly one binary input")
        hashes.append(item.payload_hash)
    if len(set(hashes)) != len(hashes):
        raise CorpusBuildError("duplicate typed payload hashes are forbidden")
    return tuple(hashes)


def build_hands_wilor_payload_corpus(
    *, sources: Sequence[SourceSpec], preserved_hands_manifest: str | Path, preserved_wilor_manifest: str | Path,
    output_root: str | Path, count: int | None = None, manifest_id: str = "hands-wilor-v22-corpus", job_id: str = "hands-envelope-soak",
) -> CorpusBuildResult:
    """Build both real Hands frames and preserved real WiLoR crops atomically."""
    manifest_id, job_id = _path_free(manifest_id, "manifest_id"), _path_free(job_id, "job_id")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count <= 0):
        raise CorpusBuildError("count must be a positive integer when supplied")
    frames = _load_frames(sources)
    try:
        historic_hands = load_payload_manifest(preserved_hands_manifest, expected_api=ModelApiName.HANDS_DETECT)
        historic_wilor = load_payload_manifest(preserved_wilor_manifest, expected_api=ModelApiName.WILOR_RECONSTRUCT)
    except ValueError as exc:
        raise CorpusBuildError(f"cannot load preserved Hands/WiLoR descriptor: {exc}") from exc
    target = Path(output_root)
    if target.exists() or target.is_symlink():
        raise CorpusBuildError(f"output root already exists; refusing reuse or mutation: {target}")
    parent = target.parent.resolve()
    if not parent.is_dir():
        raise CorpusBuildError(f"output parent does not exist: {parent}")
    exclusions = {"hands_missing_v22_source": 0, "wilor_missing_v22_source": 0}
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=parent))
    try:
        raw_dir = temporary / "raw"; raw_dir.mkdir()
        items_by_api: dict[ModelApiName, list[dict[str, object]]] = {ModelApiName.HANDS_DETECT: [], ModelApiName.WILOR_RECONSTRUCT: []}
        for api_name, historic, exclusion in (
            (ModelApiName.HANDS_DETECT, historic_hands, "hands_missing_v22_source"),
            (ModelApiName.WILOR_RECONSTRUCT, historic_wilor, "wilor_missing_v22_source"),
        ):
            for original in historic.items:
                template = _hands_template(original) if api_name is ModelApiName.HANDS_DETECT else _wilor_template(original)
                frame = _frame_for(frames, template.ownership)
                if frame is None:
                    exclusions[exclusion] += 1
                    continue
                if count is not None and len(items_by_api[api_name]) >= count:
                    continue
                index = len(items_by_api[api_name])
                if api_name is ModelApiName.HANDS_DETECT:
                    if not isinstance(template, HandsDetectRequest):
                        raise CorpusBuildError("Hands template type changed")
                    rebuilt: HandsDetectRequest | WiLoRReconstructRequest = _hands_rebuilt(template, frame, index, manifest_id, job_id)
                else:
                    if not isinstance(template, WiLoRReconstructRequest):
                        raise CorpusBuildError("WiLoR template type changed")
                    rebuilt = _wilor_reowned(template, frame, index, manifest_id, job_id)
                # Public typed parser round-trip is a publication gate, not a claim
                # about model correctness.
                parsed = type(rebuilt).from_wire(rebuilt.to_wire())
                if parsed != rebuilt:
                    raise CorpusBuildError("typed request parser changed reconstructed payload")
                items_by_api[api_name].append(_write_item(rebuilt, index=index, raw_dir=raw_dir, api_name=api_name, frame=frame))
        if not items_by_api[ModelApiName.HANDS_DETECT] or not items_by_api[ModelApiName.WILOR_RECONSTRUCT]:
            raise CorpusBuildError("preserved V22 evidence yielded no complete request for one logical API")
        descriptors: dict[ModelApiName, Path] = {}
        for api_name, items in items_by_api.items():
            descriptor = {"schema": PAYLOAD_SOURCE_SCHEMA, "manifest_id": f"{manifest_id}-{api_name.value.split('.')[0]}",
                          "api_name": api_name.value, "item_count": len(items),
                          "source_contract": "v22-real-rgb+preserved-typed-wilor-crop.v1",
                          "evidence_source": {"preserved_hands_descriptor_sha256": hashlib.sha256(Path(preserved_hands_manifest).read_bytes()).hexdigest(),
                                              "preserved_wilor_descriptor_sha256": hashlib.sha256(Path(preserved_wilor_manifest).read_bytes()).hexdigest()},
                          "exclusions": dict(exclusions), "items": items}
            path = temporary / f"{api_name.value}.json"
            path.write_text(json.dumps(descriptor, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            _validate_descriptor(path, api_name)
            descriptors[api_name] = path
        digests = {api.value: hashlib.sha256(path.read_bytes()).hexdigest() for api, path in descriptors.items()}
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return CorpusBuildResult(target, target / "hands.detect.json", target / "wilor.reconstruct.json",
                             len(items_by_api[ModelApiName.HANDS_DETECT]), len(items_by_api[ModelApiName.WILOR_RECONSTRUCT]), digests, exclusions)


__all__ = ["CorpusBuildError", "CorpusBuildResult", "SourceSpec", "build_hands_wilor_payload_corpus"]
