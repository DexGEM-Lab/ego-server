"""Distinct real payload manifests for open-loop offered-load benchmarks.

A manifest is a reproducible, ordered collection of *distinct* payload items. Each
item is model-native (named binary tensor/media parts + spatial metadata + source
timestamps + pixel/K transforms + per-item ownership) and carries a content hash so
the benchmark can prove it did not create load by repeating one identical payload.

Manifests are built from real content, not from repeating one video. The default
synthetic builder produces distinct deterministic arrays (different per-item pixel
fills, shapes, and timestamps) so tests are reproducible without EgoScale assets; a
real-payload loader (``load_payload_manifest``) reads preserved model-native
measurements from a directory of distinct sources for live benchmarks.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ego_annotation.serving.contracts import (
    ContractValidationError,
    ImageSize,
    Ownership,
    PixelTransform,
    SpatialMetadata,
    reject_filesystem_fields,
)
from ego_annotation.serving.gateway import GatewayBinaryPart, GatewayRequest
from ego_annotation.serving.router import ModelApiName


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class PayloadPartSpec:
    """A declared binary part of a manifest item (bytes + shape + dtype)."""

    name: str
    data: bytes
    shape: tuple[int, ...]
    dtype: str

    def to_manifest(self) -> dict[str, object]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype, "bytes": len(self.data)}


@dataclass(frozen=True)
class PayloadItem:
    """One distinct reproducible payload for one API call."""

    item_id: str
    api_name: ModelApiName
    ownership: Ownership
    parts: tuple[PayloadPartSpec, ...]
    spatial: SpatialMetadata | None
    model_revision: str
    work_units: int
    source_timestamp_s: float
    payload_hash: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_gateway_request(self) -> GatewayRequest:
        gw_parts = tuple(
            GatewayBinaryPart(name=p.name, data=p.data, shape=p.shape, dtype=p.dtype) for p in self.parts
        )
        meta: dict[str, object] = {"work_units": self.work_units}
        meta.update(self.metadata)
        return GatewayRequest(
            api_name=self.api_name,
            ownership=self.ownership,
            parts=gw_parts,
            spatial=self.spatial,
            metadata=meta,
            model_revision=self.model_revision,
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "api_name": self.api_name.value,
            "ownership": self.ownership.to_wire(),
            "parts": [p.to_manifest() for p in self.parts],
            "spatial": self.spatial.to_wire() if self.spatial else None,
            "model_revision": self.model_revision,
            "work_units": self.work_units,
            "source_timestamp_s": self.source_timestamp_s,
            "payload_hash": self.payload_hash,
            "metadata": dict(self.metadata),
        }


def _hash_parts(parts: Sequence[tuple[str, bytes, tuple[int, ...], str]]) -> str:
    h = hashlib.sha256()
    for name, data, shape, dtype in parts:
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(",".join(str(d) for d in shape).encode("ascii"))
        h.update(b"\x00")
        h.update(dtype.encode("ascii"))
        h.update(b"\x00")
        h.update(data)
        h.update(b"\x1e")
    return h.hexdigest()


@dataclass(frozen=True)
class PayloadManifest:
    """A reproducible collection of distinct payload items for one API."""

    manifest_id: str
    api_name: ModelApiName
    items: tuple[PayloadItem, ...]
    created_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("a payload manifest must contain at least one item")
        hashes = {item.payload_hash for item in self.items}
        if len(hashes) != len(self.items):
            raise ValueError(
                "payload manifest items must be distinct; found duplicate payload hashes — "
                "a benchmark must not repeat one identical payload"
            )
        for item in self.items:
            if item.api_name != self.api_name:
                raise ValueError(f"manifest {self.manifest_id} for {self.api_name} contains item for {item.api_name}")

    @property
    def manifest_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.manifest_id.encode("utf-8"))
        h.update(self.api_name.value.encode("utf-8"))
        for item in self.items:
            h.update(item.payload_hash.encode("ascii"))
        return h.hexdigest()

    def to_manifest(self) -> dict[str, object]:
        return {
            "manifest_id": self.manifest_id,
            "api_name": self.api_name.value,
            "manifest_hash": self.manifest_hash,
            "created_at": self.created_at,
            "item_count": len(self.items),
            "items": [item.to_manifest() for item in self.items],
        }


PAYLOAD_SOURCE_SCHEMA = "ego.benchmark-payload-source.v1"


def load_payload_manifest(
    source_path: str | Path,
    *,
    expected_api: ModelApiName | str | None = None,
    limit: int | None = None,
    item_indices: Sequence[int] | None = None,
) -> PayloadManifest:
    """Load a preserved, model-native benchmark corpus without leaking file paths.

    ``source_path`` is a local benchmark-input descriptor, not a service request. It
    uses ``ego.benchmark-payload-source.v1`` and stores binary parts beside the JSON
    descriptor. Each item has ``ownership``, ``parts`` (``name``, relative ``file``,
    ``shape``, ``dtype``), optional ``spatial``, ``model_revision``, ``work_units``,
    ``source_timestamp_s``, and optional path-free ``metadata``. The returned
    ``PayloadManifest`` contains bytes and provenance only: no local file name can
    reach the gateway or the saved benchmark artifact.

    Hashes are recomputed from the binary bytes. If a source supplies
    ``payload_hash``, it must match the recomputed value. ``limit`` selects the first
    distinct source items and refuses to repeat a payload when the corpus is too
    small.
    """
    descriptor_path = Path(source_path).resolve()
    try:
        raw = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read payload source manifest {descriptor_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"payload source manifest is not valid JSON: {descriptor_path}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("payload source manifest must be a JSON object")
    if raw.get("schema") != PAYLOAD_SOURCE_SCHEMA:
        raise ValueError(f"payload source schema must be {PAYLOAD_SOURCE_SCHEMA!r}")
    try:
        api_name = ModelApiName(raw.get("api_name"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload source has unknown api_name {raw.get('api_name')!r}") from exc
    if expected_api is not None:
        wanted = ModelApiName(expected_api) if isinstance(expected_api, str) else expected_api
        if api_name is not wanted:
            raise ValueError(f"payload source is for {api_name.value}, expected {wanted.value}")
    raw_items = raw.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("payload source manifest must contain non-empty items")
    if limit is not None and item_indices is not None:
        raise ValueError("payload source selection accepts either limit or item_indices, not both")
    if limit is not None:
        if limit <= 0:
            raise ValueError("payload manifest limit must be positive")
        if len(raw_items) < limit:
            raise ValueError(
                f"payload source has {len(raw_items)} distinct items but {limit} were requested; "
                "refusing to repeat benchmark payloads"
            )
        raw_items = raw_items[:limit]
    elif item_indices is not None:
        indices = tuple(item_indices)
        if not indices or any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise ValueError("payload item_indices must be a non-empty sequence of integers")
        if len(set(indices)) != len(indices):
            raise ValueError("payload item_indices must be unique; refusing payload reuse")
        if any(index < 0 or index >= len(raw_items) for index in indices):
            raise ValueError(f"payload item_indices must be inside [0, {len(raw_items)})")
        raw_items = [raw_items[index] for index in indices]

    source_root = descriptor_path.parent
    items: list[PayloadItem] = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, Mapping):
            raise ValueError(f"payload source item {index} must be an object")
        try:
            ownership_raw = raw_item.get("ownership", {})
            if not isinstance(ownership_raw, Mapping):
                raise ValueError("ownership must be an object")
            ownership = Ownership.from_mapping(ownership_raw)
            spatial_raw = raw_item.get("spatial")
            if spatial_raw is not None and not isinstance(spatial_raw, Mapping):
                raise ValueError("spatial must be an object or null")
            spatial = SpatialMetadata.from_mapping(spatial_raw) if spatial_raw is not None else None
            metadata = raw_item.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ContractValidationError("metadata must be an object")
            reject_filesystem_fields(metadata, context=f"source.items[{index}].metadata")
            raw_parts = raw_item.get("parts")
            if not isinstance(raw_parts, list) or not raw_parts:
                raise ValueError("parts must be a non-empty list")
            parts: list[PayloadPartSpec] = []
            hash_inputs: list[tuple[str, bytes, tuple[int, ...], str]] = []
            for raw_part in raw_parts:
                if not isinstance(raw_part, Mapping):
                    raise ValueError("each part must be an object")
                name, file_name, dtype, shape = (
                    raw_part.get("name"), raw_part.get("file"), raw_part.get("dtype"), raw_part.get("shape"),
                )
                if not isinstance(name, str) or not name:
                    raise ValueError("part.name must be a non-empty string")
                if not isinstance(file_name, str) or not file_name:
                    raise ValueError("part.file must be a non-empty relative filename")
                if not isinstance(dtype, str) or not dtype:
                    raise ValueError("part.dtype must be a non-empty string")
                if not isinstance(shape, list) or not shape or any(not isinstance(d, int) or d <= 0 for d in shape):
                    raise ValueError("part.shape must be a non-empty list of positive integers")
                part_path = (source_root / file_name).resolve()
                try:
                    part_path.relative_to(source_root)
                except ValueError as exc:
                    raise ValueError("part.file must stay inside the payload source directory") from exc
                data = part_path.read_bytes()
                shape_tuple = tuple(shape)
                parts.append(PayloadPartSpec(name=name, data=data, shape=shape_tuple, dtype=dtype))
                hash_inputs.append((name, data, shape_tuple, dtype))
            payload_hash = _hash_parts(hash_inputs)
            expected_hash = raw_item.get("payload_hash")
            if expected_hash is not None and expected_hash != payload_hash:
                raise ValueError(f"payload hash mismatch for source item {index}")
            item_id = raw_item.get("item_id")
            model_revision = raw_item.get("model_revision")
            work_units = raw_item.get("work_units")
            source_timestamp_s = raw_item.get("source_timestamp_s")
            if not isinstance(item_id, str) or not item_id:
                raise ValueError("item_id must be a non-empty string")
            if not isinstance(model_revision, str) or not model_revision:
                raise ValueError("model_revision must be a non-empty string")
            if isinstance(work_units, bool) or not isinstance(work_units, int) or work_units <= 0:
                raise ValueError("work_units must be a positive integer")
            if isinstance(source_timestamp_s, bool) or not isinstance(source_timestamp_s, (int, float)):
                raise ValueError("source_timestamp_s must be numeric")
            items.append(PayloadItem(
                item_id=item_id, api_name=api_name, ownership=ownership, parts=tuple(parts),
                spatial=spatial, model_revision=model_revision, work_units=work_units,
                source_timestamp_s=float(source_timestamp_s), payload_hash=payload_hash,
                metadata=dict(metadata),
            ))
        except (AttributeError, ContractValidationError, OSError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("payload "):
                raise
            raise ValueError(f"invalid payload source item {index}: {exc}") from exc
    return PayloadManifest(
        manifest_id=str(raw.get("manifest_id") or descriptor_path.stem), api_name=api_name,
        items=tuple(items), created_at=str(raw.get("created_at") or _utc_now()),
    )


def _identity_spatial(width: int, height: int, k_px: tuple[tuple[float, float, float], ...] | None) -> SpatialMetadata:
    return SpatialMetadata(
        source_size=ImageSize(width=width, height=height),
        model_size=ImageSize(width=width, height=height),
        color_space="RGB",
        pixel_transform=PixelTransform.identity(),
        K_px=k_px,
    )


def _synthetic_k_px(fx: float, fy: float, cx: float, cy: float) -> tuple[tuple[float, float, float], ...]:
    """A distinct 3x3 intrinsics matrix for payloads that carry K_px.

    Used by hand-detection, WiLoR, DROID session-create, and HaWoR manifests so the
    spatial contract is real (not a placeholder) while staying synthetic and
    reproducible. Distinct fx/fy per call keeps payload hashes distinct.
    """
    return (
        (fx, 0.0, cx),
        (0.0, fy, cy),
        (0.0, 0.0, 1.0),
    )


def build_synthetic_unidepth_manifest(
    *,
    manifest_id: str,
    count: int,
    height: int = 8,
    width: int = 12,
    base_source_timestamp_s: float = 0.0,
    dt_s: float = 1.0 / 30.0,
    job_ids: Sequence[str] = ("job-a", "job-b"),
    model_revision: str = "unidepth-v2-vitl14-corrected",
) -> PayloadManifest:
    """Build a manifest of ``count`` distinct uint8 RGB images for ``unidepth.infer``.

    Each item has a different per-pixel fill and timestamp so payload hashes differ.
    Items are split across multiple job/agent IDs so the benchmark exercises mixed
    ownership inside one batch.
    """
    import numpy as np

    items: list[PayloadItem] = []
    for index in range(count):
        # Distinct content: per-item fill pattern + a unique corner pixel.
        rgb = np.full((height, width, 3), (index * 7) % 256, dtype=np.uint8)
        rgb[0, 0] = (index * 13) % 256
        rgb[0, 1] = (index * 17) % 256
        data = rgb.tobytes()
        shape = (height, width, 3)
        dtype = "uint8"
        payload_hash = _hash_parts([("rgb", data, shape, dtype)])
        source_ts = base_source_timestamp_s + index * dt_s
        job_id = job_ids[index % len(job_ids)]
        ownership = Ownership(
            request_id=f"req-{manifest_id}-{index:04d}",
            job_id=job_id,
            item_id=f"frame-{index:04d}",
            stage_id="unidepth.infer",
            source_id=f"video-{index % 3}",
            source_timestamp_s=source_ts,
        )
        items.append(
            PayloadItem(
                item_id=f"{manifest_id}-item-{index:04d}",
                api_name=ModelApiName.UNIDEPTH_INFER,
                ownership=ownership,
                parts=(PayloadPartSpec("rgb", data, shape, dtype),),
                spatial=_identity_spatial(width, height, None),
                model_revision=model_revision,
                work_units=1,
                source_timestamp_s=source_ts,
                payload_hash=payload_hash,
            )
        )
    return PayloadManifest(manifest_id=manifest_id, api_name=ModelApiName.UNIDEPTH_INFER, items=tuple(items))


def build_synthetic_crops_manifest(
    *,
    manifest_id: str,
    api_name: ModelApiName,
    count: int,
    crop_size: int = 256,
    parts: Sequence[tuple[str, str]] = (("crops", "uint8"),),
    work_units: int = 1,
    model_revision: str = "synthetic-v1",
    dt_s: float = 1.0 / 30.0,
    job_ids: Sequence[str] = ("job-a", "job-b"),
) -> PayloadManifest:
    """Build a manifest of distinct crops/track-chunks/temporal-windows for any API.

    ``parts`` declares (name, dtype) for each binary part; each item gets distinct
    content. Used for WiLoR crops, HaWoR track chunks, infiller windows, and the
    hand detector (which uses the same image-batch contract as unidepth but may
    carry a mask part).
    """
    import numpy as np

    items: list[PayloadItem] = []
    for index in range(count):
        part_specs: list[PayloadPartSpec] = []
        hash_inputs: list[tuple[str, bytes, tuple[int, ...], str]] = []
        for name, dtype in parts:
            shape = (crop_size, crop_size, 3) if dtype == "uint8" else (crop_size, crop_size)
            arr = np.full(shape, (index * 7) % 256, dtype=np.dtype(dtype))
            arr[0, 0] = (index * 13) % 256
            data = arr.tobytes()
            part_specs.append(PayloadPartSpec(name, data, shape, dtype))
            hash_inputs.append((name, data, shape, dtype))
        payload_hash = _hash_parts(hash_inputs)
        source_ts = index * dt_s
        job_id = job_ids[index % len(job_ids)]
        ownership = Ownership(
            request_id=f"req-{manifest_id}-{index:04d}",
            job_id=job_id,
            item_id=f"{api_name.value}-{index:04d}",
            stage_id=api_name.value,
            source_id=f"source-{index % 4}",
            source_timestamp_s=source_ts,
        )
        items.append(
            PayloadItem(
                item_id=f"{manifest_id}-item-{index:04d}",
                api_name=api_name,
                ownership=ownership,
                parts=tuple(part_specs),
                spatial=_identity_spatial(crop_size, crop_size, None),
                model_revision=model_revision,
                work_units=work_units,
                source_timestamp_s=source_ts,
                payload_hash=payload_hash,
            )
        )
    return PayloadManifest(manifest_id=manifest_id, api_name=api_name, items=tuple(items))


# --- hand detection / WiLoR reconstruction -----------------------------------------


def build_synthetic_hand_images_manifest(
    *,
    manifest_id: str,
    count: int,
    height: int = 8,
    width: int = 12,
    base_source_timestamp_s: float = 0.0,
    dt_s: float = 1.0 / 30.0,
    job_ids: Sequence[str] = ("job-a", "job-b"),
    model_revision: str = "hands-yolo-sam2.1-hiera-l",
) -> PayloadManifest:
    """Build a manifest of distinct uint8 RGB images for ``hands.detect``.

    Each item is a full source image with a distinct content fill, a distinct
    timestamp, source ``K_px`` intrinsics, and mixed job ownership — matching the
    hand-detection contract (full images weighted by pixels, boxes/handedness/scores
    + masks returned). No caller server paths.
    """
    import numpy as np

    items: list[PayloadItem] = []
    for index in range(count):
        rgb = np.full((height, width, 3), (index * 7) % 256, dtype=np.uint8)
        rgb[0, 0] = (index * 13) % 256
        rgb[0, 2] = (index * 19) % 256
        data = rgb.tobytes()
        shape = (height, width, 3)
        dtype = "uint8"
        payload_hash = _hash_parts([("rgb", data, shape, dtype)])
        source_ts = base_source_timestamp_s + index * dt_s
        job_id = job_ids[index % len(job_ids)]
        k_px = _synthetic_k_px(fx=525.0 + index, fy=525.0 + index, cx=width / 2.0, cy=height / 2.0)
        ownership = Ownership(
            request_id=f"req-{manifest_id}-{index:04d}",
            job_id=job_id,
            item_id=f"hand-frame-{index:04d}",
            stage_id="hands.detect",
            source_id=f"video-{index % 3}",
            source_timestamp_s=source_ts,
        )
        items.append(
            PayloadItem(
                item_id=f"{manifest_id}-item-{index:04d}",
                api_name=ModelApiName.HANDS_DETECT,
                ownership=ownership,
                parts=(PayloadPartSpec("rgb", data, shape, dtype),),
                spatial=_identity_spatial(width, height, k_px),
                model_revision=model_revision,
                work_units=1,
                source_timestamp_s=source_ts,
                payload_hash=payload_hash,
            )
        )
    return PayloadManifest(manifest_id=manifest_id, api_name=ModelApiName.HANDS_DETECT, items=tuple(items))


def build_synthetic_hand_crops_manifest(
    *,
    manifest_id: str,
    count: int,
    crop_size: int = 256,
    base_source_timestamp_s: float = 0.0,
    dt_s: float = 1.0 / 30.0,
    job_ids: Sequence[str] = ("job-a", "job-b"),
    model_revision: str = "wilor-final-v1",
) -> PayloadManifest:
    """Build a manifest of distinct WiLoR hand crops for ``wilor.reconstruct``.

    Each item is one ``[3,256,256]`` normalized hand crop carrying handedness, a
    crop-to-source pixel transform, source ``K_px``, frame/hand IDs, and a source
    timestamp — the model-native WiLoR contract (crops bucketed by shape, drawn across
    frames/videos/jobs/agents). Distinct per-crop content keeps hashes distinct. No
    caller server paths.
    """
    import numpy as np

    items: list[PayloadItem] = []
    for index in range(count):
        # [C,H,W] float32 crop, distinct fill per crop.
        crop = np.full((3, crop_size, crop_size), ((index * 7) % 100) / 100.0, dtype=np.float32)
        crop[0, 0, 0] = ((index * 13) % 100) / 100.0
        data = crop.tobytes()
        shape = (3, crop_size, crop_size)
        dtype = "float32"
        payload_hash = _hash_parts([("crop", data, shape, dtype)])
        source_ts = base_source_timestamp_s + index * dt_s
        job_id = job_ids[index % len(job_ids)]
        handedness = "left" if index % 2 == 0 else "right"
        # Distinct crop-to-source transform per crop (a scaled+translated box).
        sx = crop_size / (320.0 + index % 5)
        sy = crop_size / (320.0 + index % 5)
        s2m = ((sx, 0.0, float(index % 4)), (0.0, sy, float(index % 3)), (0.0, 0.0, 1.0))
        m2s = ((1.0 / sx, 0.0, -float(index % 4) / sx), (0.0, 1.0 / sy, -float(index % 3) / sy), (0.0, 0.0, 1.0))
        pixel_transform = PixelTransform(
            source_to_model=s2m, model_to_source=m2s, resize_mode="letterbox",
            crop_xywh=(float(index % 4), float(index % 3), 320.0, 320.0), pad_ltrb=None,
        )
        k_px = _synthetic_k_px(fx=525.0 + index, fy=525.0 + index, cx=crop_size / 2.0, cy=crop_size / 2.0)
        spatial = SpatialMetadata(
            source_size=ImageSize(width=320, height=320),
            model_size=ImageSize(width=crop_size, height=crop_size),
            color_space="RGB",
            pixel_transform=pixel_transform,
            K_px=k_px,
        )
        ownership = Ownership(
            request_id=f"req-{manifest_id}-{index:04d}",
            job_id=job_id,
            item_id=f"hand-{index % 9:02d}-frame-{index:04d}",
            stage_id="wilor.reconstruct",
            source_id=f"video-{index % 4}",
            source_timestamp_s=source_ts,
        )
        items.append(
            PayloadItem(
                item_id=f"{manifest_id}-item-{index:04d}",
                api_name=ModelApiName.WILOR_RECONSTRUCT,
                ownership=ownership,
                parts=(PayloadPartSpec("crop", data, shape, dtype),),
                spatial=spatial,
                model_revision=model_revision,
                work_units=1,
                source_timestamp_s=source_ts,
                payload_hash=payload_hash,
                metadata={"handedness": handedness, "hand_id": f"hand-{index % 9:02d}", "frame_id": index},
            )
        )
    return PayloadManifest(manifest_id=manifest_id, api_name=ModelApiName.WILOR_RECONSTRUCT, items=tuple(items))


def build_synthetic_media_manifest(
    *,
    manifest_id: str,
    api_name: ModelApiName,
    count: int,
    model_revision: str = "cosmos3-nano-v1",
    job_ids: Sequence[str] = ("job-a", "job-b", "job-c"),
) -> PayloadManifest:
    """Build a manifest of distinct media/prompt payloads for ``cosmos3.reason``.

    Each item carries a distinct prompt string and distinct pseudo-image bytes so
    multimodal/prefix cache does not define the result.
    """
    items: list[PayloadItem] = []
    for index in range(count):
        prompt = f"describe scene {index}: action {index % 5} object {index % 7}"
        prompt_bytes = prompt.encode("utf-8")
        # Distinct pseudo-image bytes.
        img = bytes((index * 11 + i) % 256 for i in range(64))
        payload_hash = _hash_parts([("prompt", prompt_bytes, (len(prompt_bytes),), "bytes"), ("media", img, (64,), "bytes")])
        job_id = job_ids[index % len(job_ids)]
        ownership = Ownership(
            request_id=f"req-{manifest_id}-{index:04d}",
            job_id=job_id,
            item_id=f"media-{index:04d}",
            stage_id=api_name.value,
            source_id=f"media-source-{index % 6}",
            source_timestamp_s=float(index),
        )
        items.append(
            PayloadItem(
                item_id=f"{manifest_id}-item-{index:04d}",
                api_name=api_name,
                ownership=ownership,
                parts=(
                    PayloadPartSpec("prompt", prompt_bytes, (len(prompt_bytes),), "bytes"),
                    PayloadPartSpec("media", img, (64,), "bytes"),
                ),
                spatial=None,
                model_revision=model_revision,
                work_units=1,
                source_timestamp_s=float(index),
                payload_hash=payload_hash,
                metadata={"prompt_text": prompt},
            )
        )
    return PayloadManifest(manifest_id=manifest_id, api_name=api_name, items=tuple(items))


# --- stateful DROID session lifecycle (three session methods) -----------------------
#
# DROID is one service exposed through three session-method API names:
# ``droid.create_session`` -> ``droid.push_frame`` (N times) -> ``droid.finalize``.
# Because a ``PayloadManifest`` spans a single api_name, the session lifecycle is
# modeled as three per-method manifests plus a coherent ``DroidSessionPlan`` that
# ties them together with one shared ``session_id``. The benchmark sweeps each method
# in isolation; a live DROID push_frame sweep requires session_ids produced by real
# create_session calls, so the builders seed a session_id pool (documented residual
# dependency when DROID is not yet live).


def _droid_k_px(index: int, width: int, height: int) -> tuple[tuple[float, float, float], ...]:
    return _synthetic_k_px(fx=420.0 + index % 7, fy=420.0 + index % 7, cx=width / 2.0, cy=height / 2.0)


def build_synthetic_droid_create_session_manifest(
    *,
    manifest_id: str,
    count: int,
    image_shape: tuple[int, int] = (8, 12),  # (height, width)
    base_source_timestamp_s: float = 0.0,
    dt_s: float = 1.0,
    job_ids: Sequence[str] = ("job-a", "job-b"),
    model_revision: str = "droid-v1",
) -> PayloadManifest:
    """Build a manifest of distinct ``droid.create_session`` payloads.

    Each item initializes an isolated sequence state: a camera intrinsics ``K_px``,
    an image shape, and session options. ``metadata.session_id`` carries the new
    session identifier the caller proposes (the replica assigns/records it). One item
    is one new session (work unit = sessions). No caller server paths.
    """
    import numpy as np

    height, width = image_shape
    items: list[PayloadItem] = []
    for index in range(count):
        k_bytes = np.array(_droid_k_px(index, width, height), dtype=np.float32).tobytes()
        opts = {"max_frames": 64, "mode": "dense"}
        opts_bytes = json.dumps(opts, sort_keys=True).encode("utf-8")
        payload_hash = _hash_parts([
            ("K_px", k_bytes, (3, 3), "float32"),
            ("options", opts_bytes, (len(opts_bytes),), "bytes"),
        ])
        source_ts = base_source_timestamp_s + index * dt_s
        job_id = job_ids[index % len(job_ids)]
        session_id = f"session-{manifest_id}-{index:04d}"
        ownership = Ownership(
            request_id=f"req-{manifest_id}-{index:04d}",
            job_id=job_id,
            item_id=session_id,
            stage_id="droid.create_session",
            source_id=f"sequence-{index % 5}",
            source_timestamp_s=source_ts,
        )
        items.append(
            PayloadItem(
                item_id=f"{manifest_id}-item-{index:04d}",
                api_name=ModelApiName.DROID_CREATE_SESSION,
                ownership=ownership,
                parts=(
                    PayloadPartSpec("K_px", k_bytes, (3, 3), "float32"),
                    PayloadPartSpec("options", opts_bytes, (len(opts_bytes),), "bytes"),
                ),
                spatial=None,
                model_revision=model_revision,
                work_units=1,
                source_timestamp_s=source_ts,
                payload_hash=payload_hash,
                metadata={
                    "session_id": session_id,
                    "image_shape": [height, width],
                    "camera_model": "pinhole",
                },
            )
        )
    return PayloadManifest(manifest_id=manifest_id, api_name=ModelApiName.DROID_CREATE_SESSION, items=tuple(items))


def build_synthetic_droid_push_frame_manifest(
    *,
    manifest_id: str,
    count: int,
    session_ids: Sequence[str],
    image_shape: tuple[int, int] = (8, 12),
    base_source_timestamp_s: float = 0.0,
    dt_s: float = 1.0 / 30.0,
    job_ids: Sequence[str] = ("job-a", "job-b"),
    model_revision: str = "droid-v1",
) -> PayloadManifest:
    """Build a manifest of distinct ``droid.push_frame`` ready-frame payloads.

    Each item is the next ordered model-size frame for one session: an RGB frame
    ``[H,W,3]`` uint8 and a per-frame static-confidence mask ``[H,W]`` uint8. ``work``
    is measured in ready frames; at most one frame per session is ready in a
    cross-session batch (the replica enforces this). ``metadata.session_id`` and
    ``metadata.frame_id`` carry the per-session ordering. No caller server paths.
    """
    import numpy as np

    height, width = image_shape
    if not session_ids:
        raise ValueError("push_frame manifest requires at least one session_id")
    items: list[PayloadItem] = []
    for index in range(count):
        session_id = session_ids[index % len(session_ids)]
        frame_id = index // len(session_ids)  # per-session frame index
        rgb = np.full((height, width, 3), (index * 7) % 256, dtype=np.uint8)
        rgb[0, 0] = (index * 13) % 256
        mask = np.full((height, width), (index * 5) % 2, dtype=np.uint8)
        rgb_data = rgb.tobytes()
        mask_data = mask.tobytes()
        payload_hash = _hash_parts([
            ("rgb", rgb_data, (height, width, 3), "uint8"),
            ("static_mask", mask_data, (height, width), "uint8"),
        ])
        source_ts = base_source_timestamp_s + index * dt_s
        job_id = job_ids[index % len(job_ids)]
        ownership = Ownership(
            request_id=f"req-{manifest_id}-{index:04d}",
            job_id=job_id,
            item_id=f"{session_id}-frame-{frame_id:04d}",
            stage_id="droid.push_frame",
            source_id=f"sequence-{index % 5}",
            source_timestamp_s=source_ts,
        )
        items.append(
            PayloadItem(
                item_id=f"{manifest_id}-item-{index:04d}",
                api_name=ModelApiName.DROID_PUSH_FRAME,
                ownership=ownership,
                parts=(
                    PayloadPartSpec("rgb", rgb_data, (height, width, 3), "uint8"),
                    PayloadPartSpec("static_mask", mask_data, (height, width), "uint8"),
                ),
                spatial=None,
                model_revision=model_revision,
                work_units=1,
                source_timestamp_s=source_ts,
                payload_hash=payload_hash,
                metadata={"session_id": session_id, "frame_id": frame_id, "timestamp_s": source_ts},
            )
        )
    return PayloadManifest(manifest_id=manifest_id, api_name=ModelApiName.DROID_PUSH_FRAME, items=tuple(items))


def build_synthetic_droid_finalize_manifest(
    *,
    manifest_id: str,
    session_ids: Sequence[str],
    base_source_timestamp_s: float = 0.0,
    dt_s: float = 1.0,
    job_ids: Sequence[str] = ("job-a", "job-b"),
    model_revision: str = "droid-v1",
) -> PayloadManifest:
    """Build a manifest of distinct ``droid.finalize`` payloads (one per session).

    Each item closes one session and requests dense ``T_world_camera``, derived
    ``T_camera_world``, keyframe timestamps, disparity/depth gauge, and
    ``scale_status="up_to_scale"``. One item is one completed session. No paths.
    """
    items: list[PayloadItem] = []
    for index, session_id in enumerate(session_ids):
        marker = f"finalize-{session_id}".encode("utf-8")
        payload_hash = _hash_parts([("finalize_marker", marker, (len(marker),), "bytes")])
        source_ts = base_source_timestamp_s + index * dt_s
        job_id = job_ids[index % len(job_ids)]
        ownership = Ownership(
            request_id=f"req-{manifest_id}-{index:04d}",
            job_id=job_id,
            item_id=f"{session_id}-finalize",
            stage_id="droid.finalize",
            source_id=f"sequence-{index % 5}",
            source_timestamp_s=source_ts,
        )
        items.append(
            PayloadItem(
                item_id=f"{manifest_id}-item-{index:04d}",
                api_name=ModelApiName.DROID_FINALIZE,
                ownership=ownership,
                parts=(PayloadPartSpec("finalize_marker", marker, (len(marker),), "bytes"),),
                spatial=None,
                model_revision=model_revision,
                work_units=1,
                source_timestamp_s=source_ts,
                payload_hash=payload_hash,
                metadata={
                    "session_id": session_id,
                    "request_dense": True,
                    "request_keyframes": True,
                    "scale_status": "up_to_scale",
                },
            )
        )
    return PayloadManifest(manifest_id=manifest_id, api_name=ModelApiName.DROID_FINALIZE, items=tuple(items))


@dataclass(frozen=True)
class DroidSessionPlan:
    """A coherent DROID session: one create + N push_frame + one finalize.

    All three method payloads share one ``session_id`` so the benchmark can run the
    full create -> push(N) -> finalize lifecycle for one or more sessions. This is
    not a single ``PayloadManifest`` (it spans three api_names); it bundles the three
    per-method manifests that share the session identifier.
    """

    session_id: str
    create_manifest: PayloadManifest
    push_frame_manifest: PayloadManifest
    finalize_manifest: PayloadManifest


def build_synthetic_droid_session_plan(
    *,
    plan_id: str,
    session_count: int = 1,
    frames_per_session: int = 16,
    image_shape: tuple[int, int] = (8, 12),
    job_ids: Sequence[str] = ("job-a", "job-b"),
    model_revision: str = "droid-v1",
) -> tuple[DroidSessionPlan, ...]:
    """Build one coherent DROID session plan per session (create+push+finalize).

    Returns one ``DroidSessionPlan`` per session, each sharing its ``session_id``
    across its create/push/finalize manifests. Lets the benchmark sweep the full
    stateful DROID session lifecycle without caller server paths. The push_frame
    manifest seeds ``session_id``s; when DROID is live a push_frame sweep must use
    session_ids returned by real create_session calls (documented residual dependency).
    """
    plans: list[DroidSessionPlan] = []
    for s in range(session_count):
        session_id = f"session-{plan_id}-{s:04d}"
        create = build_synthetic_droid_create_session_manifest(
            manifest_id=f"{plan_id}-s{s:04d}-create", count=1, image_shape=image_shape,
            base_source_timestamp_s=float(s), job_ids=job_ids, model_revision=model_revision,
        )
        # Rewrite the single create item's seeded session_id to this plan's id.
        create_item = create.items[0]
        create = PayloadManifest(
            manifest_id=create.manifest_id, api_name=create.api_name,
            items=(
                PayloadItem(
                    item_id=create_item.item_id, api_name=create_item.api_name,
                    ownership=Ownership(
                        request_id=create_item.ownership.request_id, job_id=create_item.ownership.job_id,
                        item_id=session_id, stage_id=create_item.ownership.stage_id,
                        source_id=create_item.ownership.source_id,
                        source_timestamp_s=create_item.ownership.source_timestamp_s,
                    ),
                    parts=create_item.parts, spatial=create_item.spatial,
                    model_revision=create_item.model_revision, work_units=create_item.work_units,
                    source_timestamp_s=create_item.source_timestamp_s, payload_hash=create_item.payload_hash,
                    metadata={**create_item.metadata, "session_id": session_id},
                ),
            ),
        )
        push = build_synthetic_droid_push_frame_manifest(
            manifest_id=f"{plan_id}-s{s:04d}-push", count=frames_per_session,
            session_ids=(session_id,), image_shape=image_shape,
            base_source_timestamp_s=float(s), job_ids=job_ids, model_revision=model_revision,
        )
        finalize = build_synthetic_droid_finalize_manifest(
            manifest_id=f"{plan_id}-s{s:04d}-finalize", session_ids=(session_id,),
            base_source_timestamp_s=float(s), job_ids=job_ids, model_revision=model_revision,
        )
        plans.append(DroidSessionPlan(
            session_id=session_id, create_manifest=create,
            push_frame_manifest=push, finalize_manifest=finalize,
        ))
    return tuple(plans)


# --- HaWoR track chunks + temporal infiller -----------------------------------------


def build_synthetic_hawor_tracks_manifest(
    *,
    manifest_id: str,
    count: int,
    chunk_len: int = 16,
    crop_size: int = 256,
    base_source_timestamp_s: float = 0.0,
    dt_s: float = 1.0 / 30.0,
    job_ids: Sequence[str] = ("job-a", "job-b"),
    model_revision: str = "hawor-v1",
) -> PayloadManifest:
    """Build a manifest of distinct HaWoR track chunks for ``hawor.infer_tracks``.

    Each item is one temporal hand-crop chunk ``[chunk_len,3,crop_size,crop_size]``
    (default ``[16,3,256,256]``) carrying track/hand identity, handedness, a
    crop-to-source transform, source ``K_px``, ``chunk_len`` per-frame source
    timestamps, and a ``[chunk_len]`` observation-support mask. One item is one track
    chunk (work unit = track_chunks). No caller server paths.
    """
    import numpy as np

    items: list[PayloadItem] = []
    for index in range(count):
        chunk = np.full((chunk_len, 3, crop_size, crop_size), ((index * 7) % 100) / 100.0, dtype=np.float32)
        chunk[0, 0, 0, 0] = ((index * 13) % 100) / 100.0
        chunk_data = chunk.tobytes()
        chunk_shape = (chunk_len, 3, crop_size, crop_size)
        # Per-frame source timestamps and an observation-support mask.
        ts = np.arange(chunk_len, dtype=np.float32) * dt_s + base_source_timestamp_s + index * dt_s
        obs = np.ones(chunk_len, dtype=np.uint8)
        obs[index % chunk_len] = 0  # one occluded frame per chunk for realism
        ts_data = ts.tobytes()
        obs_data = obs.tobytes()
        payload_hash = _hash_parts([
            ("track_chunk", chunk_data, chunk_shape, "float32"),
            ("source_timestamps", ts_data, (chunk_len,), "float32"),
            ("observation_mask", obs_data, (chunk_len,), "uint8"),
        ])
        source_ts = base_source_timestamp_s + index * dt_s
        job_id = job_ids[index % len(job_ids)]
        handedness = "left" if index % 2 == 0 else "right"
        k_px = _synthetic_k_px(fx=525.0 + index, fy=525.0 + index, cx=crop_size / 2.0, cy=crop_size / 2.0)
        ownership = Ownership(
            request_id=f"req-{manifest_id}-{index:04d}",
            job_id=job_id,
            item_id=f"track-{index % 9:02d}-chunk-{index:04d}",
            stage_id="hawor.infer_tracks",
            source_id=f"video-{index % 4}",
            source_timestamp_s=source_ts,
        )
        items.append(
            PayloadItem(
                item_id=f"{manifest_id}-item-{index:04d}",
                api_name=ModelApiName.HAWOR_INFER_TRACKS,
                ownership=ownership,
                parts=(
                    PayloadPartSpec("track_chunk", chunk_data, chunk_shape, "float32"),
                    PayloadPartSpec("source_timestamps", ts_data, (chunk_len,), "float32"),
                    PayloadPartSpec("observation_mask", obs_data, (chunk_len,), "uint8"),
                ),
                spatial=_identity_spatial(crop_size, crop_size, k_px),
                model_revision=model_revision,
                work_units=1,
                source_timestamp_s=source_ts,
                payload_hash=payload_hash,
                metadata={
                    "track_id": f"track-{index % 9:02d}", "handedness": handedness,
                    "chunk_len": chunk_len, "crop_size": crop_size,
                },
            )
        )
    return PayloadManifest(manifest_id=manifest_id, api_name=ModelApiName.HAWOR_INFER_TRACKS, items=tuple(items))


def build_synthetic_hawor_infiller_manifest(
    *,
    manifest_id: str,
    count: int,
    window_len: int = 120,
    mano_dim: int = 61,  # pose(48) + shape(10) + trans(3)
    base_source_timestamp_s: float = 0.0,
    dt_s: float = 1.0 / 30.0,
    job_ids: Sequence[str] = ("job-a", "job-b"),
    model_revision: str = "hawor-infiller-v1",
) -> PayloadManifest:
    """Build a manifest of distinct HaWoR infiller temporal windows for ``hawor_infiller.fill``.

    Each item is one temporal window (default 120 frames) of camera-space MANO state
    ``[window_len, mano_dim]`` float32, with ``window_len`` source timestamps, a
    ``[window_len]`` observation/occlusion mask (uint8), and a ``[window_len]``
    observed-state uncertainty (float32). One item is one temporal window (work unit =
    temporal_windows). No caller server paths.
    """
    import numpy as np

    items: list[PayloadItem] = []
    for index in range(count):
        mano = np.full((window_len, mano_dim), ((index * 7) % 100) / 100.0, dtype=np.float32)
        mano[0, 0] = ((index * 13) % 100) / 100.0
        mano_data = mano.tobytes()
        mano_shape = (window_len, mano_dim)
        ts = np.arange(window_len, dtype=np.float32) * dt_s + base_source_timestamp_s + index * dt_s
        obs = np.ones(window_len, dtype=np.uint8)
        # Occluded interval to be filled: roughly 20% of the window.
        gap_start = (index * 3) % max(1, window_len - 24)
        gap_end = min(gap_start + 24, window_len)
        obs[gap_start:gap_end] = 0
        unc = np.full(window_len, 0.02, dtype=np.float32)
        unc[gap_start:gap_end] = 0.0
        ts_data = ts.tobytes()
        obs_data = obs.tobytes()
        unc_data = unc.tobytes()
        payload_hash = _hash_parts([
            ("mano_state", mano_data, mano_shape, "float32"),
            ("source_timestamps", ts_data, (window_len,), "float32"),
            ("observation_mask", obs_data, (window_len,), "uint8"),
            ("uncertainty", unc_data, (window_len,), "float32"),
        ])
        source_ts = base_source_timestamp_s + index * dt_s
        job_id = job_ids[index % len(job_ids)]
        handedness = "left" if index % 2 == 0 else "right"
        ownership = Ownership(
            request_id=f"req-{manifest_id}-{index:04d}",
            job_id=job_id,
            item_id=f"infill-track-{index % 9:02d}-win-{index:04d}",
            stage_id="hawor_infiller.fill",
            source_id=f"video-{index % 4}",
            source_timestamp_s=source_ts,
        )
        items.append(
            PayloadItem(
                item_id=f"{manifest_id}-item-{index:04d}",
                api_name=ModelApiName.HAWOR_INFILLER_FILL,
                ownership=ownership,
                parts=(
                    PayloadPartSpec("mano_state", mano_data, mano_shape, "float32"),
                    PayloadPartSpec("source_timestamps", ts_data, (window_len,), "float32"),
                    PayloadPartSpec("observation_mask", obs_data, (window_len,), "uint8"),
                    PayloadPartSpec("uncertainty", unc_data, (window_len,), "float32"),
                ),
                spatial=None,
                model_revision=model_revision,
                work_units=1,
                source_timestamp_s=source_ts,
                payload_hash=payload_hash,
                metadata={
                    "track_id": f"track-{index % 9:02d}", "handedness": handedness,
                    "window_len": window_len, "gap_frames": [gap_start, gap_end],
                },
            )
        )
    return PayloadManifest(manifest_id=manifest_id, api_name=ModelApiName.HAWOR_INFILLER_FILL, items=tuple(items))
