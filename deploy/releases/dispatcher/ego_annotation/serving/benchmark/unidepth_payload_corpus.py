"""Build a deterministic, real-frame UniDepth payload corpus.

The builder is deliberately a preparation step: it reads already materialized V22
raw-frame JPEGs, decodes them to RGB, and writes raw HWC uint8 tensor bytes.  The
result is a path-free ``ego.benchmark-payload-source.v1`` descriptor that the
open-loop scaling harness can load without decoding images during measurement.
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

from PIL import Image

from ego_annotation.serving.benchmark.manifest import (
    PAYLOAD_SOURCE_SCHEMA,
    _hash_parts,
    load_payload_manifest,
)
from ego_annotation.serving.contracts import ContractValidationError, SpatialMetadata, reject_filesystem_fields
from ego_annotation.serving.router import ModelApiName

SOURCE_WIDTH = 1920
SOURCE_HEIGHT = 1080
MODEL_WIDTH = 960
MODEL_HEIGHT = 540
MODEL_REVISION_DEFAULT = "unidepth-v2-vitl14-corrected"
_SELECTION_POLICIES = frozenset(("source-order", "uniform"))


@dataclass(frozen=True)
class SourceSpec:
    """One V22 manifest, public source identity, and optional local policy."""

    manifest_path: Path
    source_id: str
    take_count: int
    selection_policy: str | None = None


@dataclass(frozen=True)
class SelectedFrame:
    """Validated source frame selected before any output is materialized."""

    source_id: str
    source_frame_index: int
    source_timestamp_s: float
    jpeg_path: Path
    k_px: tuple[tuple[float, float, float], ...] | None


@dataclass(frozen=True)
class CorpusBuildResult:
    output_root: Path
    descriptor_path: Path
    item_count: int
    descriptor_sha256: str
    payload_hashes: tuple[str, ...]


class CorpusBuildError(ValueError):
    """The requested corpus cannot truthfully be represented."""


def _json_load(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CorpusBuildError(f"cannot read V22 raw-frame manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CorpusBuildError(f"V22 raw-frame manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise CorpusBuildError("V22 raw-frame manifest must be a JSON object")
    if payload.get("schema") not in (None, "v22_raw_frame_manifest.v0"):
        raise CorpusBuildError(f"unsupported V22 raw-frame manifest schema {payload.get('schema')!r}")
    return payload


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorpusBuildError(f"{label} must be a non-negative integer")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CorpusBuildError(f"{label} must be numeric")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise CorpusBuildError(f"{label} must be finite")
    return number


def _path_free_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusBuildError(f"{label} must be a non-empty string")
    if "/" in value or "\\" in value or value.startswith("~"):
        raise CorpusBuildError(f"{label} must be a path-free identifier")
    return value


def _source_dimensions(manifest: Mapping[str, Any], frame: Mapping[str, Any], label: str) -> tuple[int, int]:
    """Require dimensions declared by the source and ensure they are canonical."""
    candidates = (
        (frame.get("source_width"), frame.get("source_height")),
        (frame.get("manifest_width"), frame.get("manifest_height")),
        (manifest.get("source_width"), manifest.get("source_height")),
        (manifest.get("manifest_width"), manifest.get("manifest_height")),
    )
    video = manifest.get("video")
    if isinstance(video, Mapping):
        candidates += ((video.get("width"), video.get("height")),)
    declared = False
    for width, height in candidates:
        if width is None and height is None:
            continue
        declared = True
        if width is None or height is None:
            raise CorpusBuildError(f"{label} has incomplete source dimensions")
        actual = (_positive_int(width, f"{label}.width"), _positive_int(height, f"{label}.height"))
        if actual != (SOURCE_WIDTH, SOURCE_HEIGHT):
            raise CorpusBuildError(
                f"{label} declares {actual[0]}x{actual[1]}, expected {SOURCE_WIDTH}x{SOURCE_HEIGHT}"
            )
    if not declared:
        raise CorpusBuildError(f"{label} has no source dimensions")
    return (SOURCE_WIDTH, SOURCE_HEIGHT)


def _source_k_px(manifest: Mapping[str, Any], frame: Mapping[str, Any], label: str) -> tuple[tuple[float, float, float], ...] | None:
    value: object | None = frame.get("K_px", frame.get("intrinsics_px"))
    if value is None:
        value = manifest.get("K_px", manifest.get("intrinsics_px"))
    if value is None and isinstance(manifest.get("camera"), Mapping):
        camera = manifest["camera"]
        value = camera.get("K_px", camera.get("intrinsics_px"))
    if value is None:
        return None
    try:
        # SpatialMetadata owns the exact 3x3 numeric validation used by the service.
        spatial = SpatialMetadata.from_mapping({
            "source_size": {"width": SOURCE_WIDTH, "height": SOURCE_HEIGHT},
            "model_size": {"width": MODEL_WIDTH, "height": MODEL_HEIGHT},
            "color_space": "RGB",
            "pixel_transform": _spatial_transform(),
            "K_px": value,
        })
    except ContractValidationError as exc:
        raise CorpusBuildError(f"{label}.K_px is invalid: {exc}") from exc
    return spatial.K_px


def _spatial_transform() -> dict[str, object]:
    return {
        "source_to_model": [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
        "model_to_source": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
        "resize_mode": "lanczos",
        "crop_xywh": None,
        "pad_ltrb": None,
    }


def _frame_timestamp(frame: Mapping[str, Any], label: str) -> float:
    value = frame.get("source_time_s", frame.get("time_s"))
    if value is None:
        raise CorpusBuildError(f"{label} has no source timestamp")
    return _finite_number(value, f"{label}.source_timestamp_s")


def _frame_path(manifest_path: Path, frame: Mapping[str, Any], label: str) -> Path:
    rgb = frame.get("rgb")
    if not isinstance(rgb, str) or not rgb:
        raise CorpusBuildError(f"{label}.rgb must be a non-empty JPEG path")
    candidate = Path(rgb)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise CorpusBuildError(f"{label}.rgb cannot be read") from exc


def _select_indices(available: int, count: int, policy: str) -> tuple[int, ...]:
    if count <= 0:
        raise CorpusBuildError("take_count must be positive")
    if count > available:
        raise CorpusBuildError(f"source has {available} frames but {count} were requested")
    if policy == "source-order":
        return tuple(range(count))
    if policy == "uniform":
        if count == 1:
            return (0,)
        return tuple(index * (available - 1) // (count - 1) for index in range(count))
    raise CorpusBuildError(f"selection policy must be one of {sorted(_SELECTION_POLICIES)}")


def _load_selected_source(spec: SourceSpec, default_policy: str) -> tuple[SelectedFrame, ...]:
    source_id = _path_free_identifier(spec.source_id, "source_id")
    policy = spec.selection_policy or default_policy
    if policy not in _SELECTION_POLICIES:
        raise CorpusBuildError(f"selection policy must be one of {sorted(_SELECTION_POLICIES)}")
    if isinstance(spec.take_count, bool) or not isinstance(spec.take_count, int) or spec.take_count <= 0:
        raise CorpusBuildError("take_count must be a positive integer")
    manifest_path = Path(spec.manifest_path).resolve()
    raw = _json_load(manifest_path)
    frames = raw.get("frames")
    if not isinstance(frames, list) or not frames:
        raise CorpusBuildError(f"{manifest_path} must have a non-empty frames list")
    declared_count = raw.get("frame_count")
    if declared_count is not None and _positive_int(declared_count, f"{manifest_path.name}.frame_count") != len(frames):
        raise CorpusBuildError(f"{manifest_path.name}.frame_count does not match frames")

    validated: list[SelectedFrame] = []
    seen_source_indices: set[int] = set()
    for position, row in enumerate(frames):
        label = f"{manifest_path.name}.frames[{position}]"
        if not isinstance(row, Mapping):
            raise CorpusBuildError(f"{label} must be an object")
        source_frame_index = _positive_int(row.get("source_frame_idx", row.get("frame_idx")), f"{label}.frame_idx")
        if source_frame_index in seen_source_indices:
            raise CorpusBuildError(f"{label} duplicates source frame index {source_frame_index}")
        seen_source_indices.add(source_frame_index)
        _source_dimensions(raw, row, label)
        validated.append(SelectedFrame(
            source_id=source_id,
            source_frame_index=source_frame_index,
            source_timestamp_s=_frame_timestamp(row, label),
            jpeg_path=_frame_path(manifest_path, row, label),
            k_px=_source_k_px(raw, row, label),
        ))

    return tuple(validated[index] for index in _select_indices(len(validated), spec.take_count, policy))


def _decode_and_resize(frame: SelectedFrame) -> bytes:
    """Decode a real JPEG and return only contiguous HWC RGB tensor bytes."""
    try:
        with Image.open(frame.jpeg_path) as image:
            if image.format != "JPEG":
                raise CorpusBuildError(f"source frame {frame.source_id}:{frame.source_frame_index} is not a JPEG")
            if image.size != (SOURCE_WIDTH, SOURCE_HEIGHT):
                raise CorpusBuildError(
                    f"source frame {frame.source_id}:{frame.source_frame_index} decodes to "
                    f"{image.size[0]}x{image.size[1]}, expected {SOURCE_WIDTH}x{SOURCE_HEIGHT}"
                )
            rgb = image.convert("RGB").resize((MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.LANCZOS)
            raw = rgb.tobytes("raw", "RGB")
    except CorpusBuildError:
        raise
    except (OSError, ValueError) as exc:
        raise CorpusBuildError(f"cannot decode JPEG for {frame.source_id}:{frame.source_frame_index}") from exc
    expected_size = MODEL_HEIGHT * MODEL_WIDTH * 3
    if len(raw) != expected_size or raw.startswith(b"\xff\xd8"):
        raise CorpusBuildError("decoded RGB tensor is not contiguous 960x540 HWC uint8 bytes")
    return raw


def _spatial(k_px: tuple[tuple[float, float, float], ...] | None) -> dict[str, object]:
    return {
        "source_size": {"width": SOURCE_WIDTH, "height": SOURCE_HEIGHT},
        "model_size": {"width": MODEL_WIDTH, "height": MODEL_HEIGHT},
        "color_space": "RGB",
        "pixel_transform": _spatial_transform(),
        "K_px": [list(row) for row in k_px] if k_px is not None else None,
    }


def _validate_descriptor(descriptor_path: Path, expected_count: int) -> tuple[str, ...]:
    """Validate exact bytes/hashes through the same loader used by the harness."""
    loaded = load_payload_manifest(descriptor_path, expected_api=ModelApiName.UNIDEPTH_INFER, limit=expected_count)
    if len(loaded.items) != expected_count:
        raise CorpusBuildError("harness loader returned an unexpected item count")
    hashes = tuple(item.payload_hash for item in loaded.items)
    if len(set(hashes)) != len(hashes):
        raise CorpusBuildError("duplicate harness payload hashes are forbidden")
    for item in loaded.items:
        if len(item.parts) != 1 or item.parts[0].name != "rgb":
            raise CorpusBuildError("UniDepth corpus item must contain exactly one rgb part")
        part = item.parts[0]
        if part.shape != (MODEL_HEIGHT, MODEL_WIDTH, 3) or part.dtype != "uint8":
            raise CorpusBuildError("UniDepth rgb part must be 540x960x3 uint8")
        if len(part.data) != MODEL_HEIGHT * MODEL_WIDTH * 3 or part.data.startswith(b"\xff\xd8"):
            raise CorpusBuildError("JPEG bytes must not be represented as an RGB tensor")
        reject_filesystem_fields(item.metadata, context="validated request metadata")
    return hashes


def _descriptor_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_unidepth_payload_corpus(
    *,
    sources: Sequence[SourceSpec],
    output_root: str | Path,
    selection_policy: str = "source-order",
    manifest_id: str = "unidepth-v22-multivideo-corpus",
    job_id: str = "unidepth-scaling-corpus",
    model_revision: str = MODEL_REVISION_DEFAULT,
) -> CorpusBuildResult:
    """Materialize a fresh payload root atomically from real V22 JPEG manifests.

    The destination must not exist.  All source validation and every output hash
    validation occur in a sibling temporary directory; only a complete, validated
    corpus is moved into the requested root.
    """
    if not sources:
        raise CorpusBuildError("at least one source manifest is required")
    if selection_policy not in _SELECTION_POLICIES:
        raise CorpusBuildError(f"selection policy must be one of {sorted(_SELECTION_POLICIES)}")
    manifest_id = _path_free_identifier(manifest_id, "manifest_id")
    job_id = _path_free_identifier(job_id, "job_id")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise CorpusBuildError("model_revision must be a non-empty string")

    requested_root = Path(output_root)
    if requested_root.exists() or requested_root.is_symlink():
        raise CorpusBuildError(f"output root already exists; refusing reuse or mutation: {requested_root}")
    output_parent = requested_root.parent.resolve()
    if not output_parent.is_dir():
        raise CorpusBuildError(f"output parent does not exist: {output_parent}")

    selected_with_policy: list[tuple[SelectedFrame, str]] = []
    for spec in sources:
        source_policy = spec.selection_policy or selection_policy
        selected_with_policy.extend((frame, source_policy) for frame in _load_selected_source(spec, selection_policy))
    selected = [frame for frame, _ in selected_with_policy]
    selected_frame_keys = {(frame.source_id, frame.source_frame_index) for frame in selected}
    if len(selected_frame_keys) != len(selected):
        raise CorpusBuildError("duplicate selected source frame ownership is forbidden")

    temporary_root = Path(tempfile.mkdtemp(prefix=f".{requested_root.name}.building-", dir=output_parent))
    try:
        raw_dir = temporary_root / "raw"
        raw_dir.mkdir()
        items: list[dict[str, object]] = []
        raw_hashes: set[str] = set()
        harness_hashes: set[str] = set()
        for item_index, (frame, source_policy) in enumerate(selected_with_policy):
            raw = _decode_and_resize(frame)
            raw_sha256 = hashlib.sha256(raw).hexdigest()
            if raw_sha256 in raw_hashes:
                raise CorpusBuildError("duplicate decoded raw RGB payload hash is forbidden")
            raw_hashes.add(raw_sha256)
            file_name = f"raw/{item_index:05d}.rgb.uint8.bin"
            (temporary_root / file_name).write_bytes(raw)
            part_shape = (MODEL_HEIGHT, MODEL_WIDTH, 3)
            payload_hash = _hash_parts((("rgb", raw, part_shape, "uint8"),))
            if payload_hash in harness_hashes:
                raise CorpusBuildError("duplicate harness payload hash is forbidden")
            harness_hashes.add(payload_hash)
            source_timestamp_s = frame.source_timestamp_s
            metadata: dict[str, object] = {
                "source_frame_index": frame.source_frame_index,
                "source_timestamp_s": source_timestamp_s,
                "selection_policy": source_policy,
                "raw_sha256": raw_sha256,
            }
            reject_filesystem_fields(metadata, context="generated request metadata")
            item_id = f"{manifest_id}-item-{item_index:05d}"
            items.append({
                "item_id": item_id,
                "ownership": {
                    "request_id": f"{manifest_id}-request-{item_index:05d}",
                    "job_id": job_id,
                    "item_id": item_id,
                    "stage_id": "unidepth.infer",
                    "source_id": frame.source_id,
                    "source_timestamp_s": source_timestamp_s,
                },
                "parts": [{"name": "rgb", "file": file_name, "shape": list(part_shape), "dtype": "uint8"}],
                "spatial": _spatial(frame.k_px),
                "model_revision": model_revision,
                "work_units": 1,
                "source_timestamp_s": source_timestamp_s,
                "source_frame_index": frame.source_frame_index,
                "raw_sha256": raw_sha256,
                "payload_hash": payload_hash,
                "metadata": metadata,
            })

        descriptor_path = temporary_root / "unidepth.infer.json"
        descriptor = {
            "schema": PAYLOAD_SOURCE_SCHEMA,
            "manifest_id": manifest_id,
            "api_name": "unidepth.infer",
            "model_revision": model_revision,
            "selection_policy": selection_policy if all(
                policy == selection_policy for _, policy in selected_with_policy
            ) else "per-source",
            "source_size": {"width": SOURCE_WIDTH, "height": SOURCE_HEIGHT},
            "model_size": {"width": MODEL_WIDTH, "height": MODEL_HEIGHT},
            "item_count": len(items),
            "items": items,
        }
        descriptor_path.write_text(json.dumps(descriptor, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        payload_hashes = _validate_descriptor(descriptor_path, len(items))
        descriptor_sha256 = _descriptor_sha256(descriptor_path)
        # The descriptor is the final artifact written, after its raw bytes and
        # harness hashes were validated.  os.replace makes the destination appear
        # only as a complete corpus.
        os.replace(temporary_root, requested_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    return CorpusBuildResult(
        output_root=requested_root,
        descriptor_path=requested_root / "unidepth.infer.json",
        item_count=len(selected),
        descriptor_sha256=descriptor_sha256,
        payload_hashes=payload_hashes,
    )


def parse_source_specs(
    manifest_paths: Iterable[str | Path],
    source_ids: Iterable[str],
    take_counts: Iterable[int],
    selection_policies: Iterable[str] | None = None,
) -> tuple[SourceSpec, ...]:
    """Pair repeated CLI options while refusing ambiguous partial source metadata."""
    paths, ids, counts = tuple(manifest_paths), tuple(source_ids), tuple(take_counts)
    policies = tuple(selection_policies or ())
    if not paths or len(paths) != len(ids) or len(paths) != len(counts):
        raise CorpusBuildError("--manifest, --source-id, and --take-count must be supplied equally, once per source")
    if policies and len(policies) != len(paths):
        raise CorpusBuildError("--selection-policy must be omitted or supplied once per source")
    return tuple(
        SourceSpec(manifest_path=Path(path), source_id=source_id, take_count=count,
                   selection_policy=policies[index] if policies else None)
        for index, (path, source_id, count) in enumerate(zip(paths, ids, counts))
    )
