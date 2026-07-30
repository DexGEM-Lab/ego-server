"""CPU-only tests for deterministic V22 -> UniDepth raw tensor corpus preparation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from ego_annotation.serving.benchmark.manifest import PAYLOAD_SOURCE_SCHEMA, load_payload_manifest
from ego_annotation.serving.benchmark.unidepth_payload_corpus import (
    CorpusBuildError,
    SourceSpec,
    build_unidepth_payload_corpus,
)
from ego_annotation.serving.gateway import ModelServiceGateway
from ego_annotation.serving.router import ModelApiName, ModelServiceRouter
from ego_annotation.serving.transport import parse_multipart_request


def _jpeg(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (1920, 1080), color)
    # A non-uniform mark makes RGB conversion/resizing exercise real image samples.
    image.putpixel((101, 203), tuple((channel + 17) % 256 for channel in color))
    image.save(path, format="JPEG", quality=95, subsampling=0)


def _source_manifest(
    root: Path,
    *,
    source_name: str,
    colors: list[tuple[int, int, int]],
    source_indices: list[int] | None = None,
    width: int = 1920,
    height: int = 1080,
    k_px: list[list[float]] | None = None,
) -> Path:
    root.mkdir(parents=True)
    source_indices = source_indices or list(range(len(colors)))
    assert len(source_indices) == len(colors)
    frames = []
    for position, (color, source_frame_idx) in enumerate(zip(colors, source_indices)):
        jpeg_name = f"{source_name}-{position:03d}.jpg"
        _jpeg(root / jpeg_name, color)
        frame = {
            "frame_idx": position,
            "source_frame_idx": source_frame_idx,
            "time_s": position / 30.0,
            "source_time_s": 10.0 + position / 30.0,
            "rgb": jpeg_name,
            "source_width": width,
            "source_height": height,
            "manifest_width": width,
            "manifest_height": height,
        }
        if k_px is not None:
            frame["K_px"] = k_px
        frames.append(frame)
    path = root / "manifest.json"
    path.write_text(json.dumps({
        "schema": "v22_raw_frame_manifest.v0",
        "frame_count": len(frames),
        "fps": 30.0,
        "frames": frames,
    }), encoding="utf-8")
    return path


def _build(tmp_path: Path, sources: list[SourceSpec], *, name: str = "corpus", policy: str = "source-order"):
    return build_unidepth_payload_corpus(
        sources=sources,
        output_root=tmp_path / name,
        selection_policy=policy,
        manifest_id="deterministic-v22",
        job_id="scaling-inputs",
        model_revision="unidepth-v2-test",
    )


def test_uniform_selection_is_deterministic_and_exact(tmp_path: Path) -> None:
    manifest = _source_manifest(
        tmp_path / "source",
        source_name="first",
        colors=[(10 + index * 10, 20, 30) for index in range(5)],
        source_indices=[10, 20, 30, 40, 50],
    )
    source = SourceSpec(manifest, "Reca87c-task_3", 3, selection_policy="uniform")
    # Source-local policies allow the real corpus to take the two shorter videos
    # in source order while uniformly subsampling the long third recording.
    first = _build(tmp_path, [source], name="first", policy="source-order")
    second = _build(tmp_path, [source], name="second", policy="source-order")
    first_descriptor = json.loads(first.descriptor_path.read_text())
    second_descriptor = json.loads(second.descriptor_path.read_text())

    assert first_descriptor["selection_policy"] == "per-source"
    assert [item["source_frame_index"] for item in first_descriptor["items"]] == [10, 30, 50]
    assert {item["metadata"]["selection_policy"] for item in first_descriptor["items"]} == {"uniform"}
    assert first_descriptor == second_descriptor
    assert first.descriptor_sha256 == second.descriptor_sha256
    assert first.item_count == 3
    assert len(set(first.payload_hashes)) == 3


def test_source_order_exact_resize_bytes_hash_and_harness_schema(tmp_path: Path) -> None:
    manifest = _source_manifest(tmp_path / "source", source_name="first", colors=[(36, 64, 128)])
    result = _build(tmp_path, [SourceSpec(manifest, "Reca87c-task_3", 1)])
    descriptor = json.loads(result.descriptor_path.read_text())
    item = descriptor["items"][0]
    raw_path = result.output_root / item["parts"][0]["file"]
    raw = raw_path.read_bytes()
    with Image.open(tmp_path / "source" / "first-000.jpg") as image:
        expected = image.convert("RGB").resize((960, 540), Image.Resampling.LANCZOS).tobytes("raw", "RGB")

    assert descriptor["schema"] == PAYLOAD_SOURCE_SCHEMA
    assert raw == expected
    assert len(raw) == 540 * 960 * 3
    assert not raw.startswith(b"\xff\xd8"), "the part is raw RGB tensor bytes, never JPEG bytes"
    assert item["raw_sha256"] == hashlib.sha256(expected).hexdigest()
    assert item["metadata"]["raw_sha256"] == item["raw_sha256"]
    assert item["parts"][0] == {
        "name": "rgb", "file": "raw/00000.rgb.uint8.bin", "shape": [540, 960, 3], "dtype": "uint8"
    }
    assert item["spatial"] == {
        "source_size": {"width": 1920, "height": 1080},
        "model_size": {"width": 960, "height": 540},
        "color_space": "RGB",
        "pixel_transform": {
            "source_to_model": [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            "model_to_source": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
            "resize_mode": "lanczos", "crop_xywh": None, "pad_ltrb": None,
        },
        "K_px": None,
    }
    loaded = load_payload_manifest(result.descriptor_path, expected_api=ModelApiName.UNIDEPTH_INFER, limit=1)
    assert loaded.items[0].parts[0].data == expected
    assert loaded.items[0].payload_hash == item["payload_hash"]
    assert loaded.items[0].ownership.source_timestamp_s == 10.0


def test_real_corpus_item_has_equivalent_multipart_and_envelope_request_content(tmp_path: Path) -> None:
    source = _source_manifest(tmp_path / "source", source_name="equivalent", colors=[(36, 64, 128)])
    corpus = _build(tmp_path, [SourceSpec(source, "Reca87c-task_3", 1)], name="equivalent")
    item = load_payload_manifest(corpus.descriptor_path, expected_api=ModelApiName.UNIDEPTH_INFER, limit=1).items[0]
    request = item.to_gateway_request()

    class NoNetwork:
        async def post(self, *_args, **_kwargs):
            raise AssertionError("wire construction must not issue HTTP")

    router = ModelServiceRouter.canonical()
    multipart = ModelServiceGateway(router, NoNetwork(), wire_format="multipart")
    envelope = ModelServiceGateway(router, NoNetwork(), wire_format="envelope")
    multipart_body, multipart_type = multipart._build_body(request)
    envelope_body, envelope_type = envelope._build_body(request)
    assert isinstance(multipart_body, bytes)
    multipart_meta, multipart_rgb, multipart_shape, multipart_dtype = parse_multipart_request(multipart_body, multipart_type)

    from ego_annotation.serving.gateway import _parse_generic_envelope

    envelope_meta, envelope_arrays = _parse_generic_envelope(b"".join(envelope_body.iovecs))
    assert envelope_type == "application/vnd.ego.binary-envelope"
    assert envelope_meta == multipart_meta
    assert envelope_arrays["rgb"][1:] == (multipart_shape, multipart_dtype)
    assert envelope_arrays["rgb"][0].tobytes() == multipart_rgb


def test_manifest_loads_explicit_disjoint_indices_without_materializing_other_parts(tmp_path: Path) -> None:
    manifest = _source_manifest(
        tmp_path / "source", source_name="select", colors=[(20, 30, 40), (40, 50, 60), (60, 70, 80)],
    )
    result = _build(tmp_path, [SourceSpec(manifest, "source-a", 3)])
    selected = load_payload_manifest(
        result.descriptor_path, expected_api=ModelApiName.UNIDEPTH_INFER, item_indices=(2, 0),
    )
    assert [item.item_id for item in selected.items] == ["deterministic-v22-item-00002", "deterministic-v22-item-00000"]
    assert len({item.payload_hash for item in selected.items}) == 2
    with pytest.raises(ValueError, match="unique"):
        load_payload_manifest(result.descriptor_path, item_indices=(0, 0))


def test_three_sources_preserve_ownership_timestamps_indices_and_optional_intrinsics(tmp_path: Path) -> None:
    k_px = [[800.0, 0.0, 960.0], [0.0, 801.0, 540.0], [0.0, 0.0, 1.0]]
    manifests = [
        _source_manifest(tmp_path / "source-a", source_name="a", colors=[(20, 40, 60)], source_indices=[101]),
        _source_manifest(tmp_path / "source-b", source_name="b", colors=[(30, 70, 90)], source_indices=[202], k_px=k_px),
        _source_manifest(tmp_path / "source-c", source_name="c", colors=[(40, 80, 120)], source_indices=[303]),
    ]
    result = _build(tmp_path, [
        SourceSpec(manifests[0], "Reca87c-task_3", 1),
        SourceSpec(manifests[1], "Rec6487-task_13", 1),
        SourceSpec(manifests[2], "Rec4c71-task_1", 1),
    ])
    descriptor = json.loads(result.descriptor_path.read_text())
    items = descriptor["items"]

    assert [item["ownership"]["source_id"] for item in items] == [
        "Reca87c-task_3", "Rec6487-task_13", "Rec4c71-task_1",
    ]
    assert [item["source_frame_index"] for item in items] == [101, 202, 303]
    assert [item["ownership"]["source_timestamp_s"] for item in items] == [10.0, 10.0, 10.0]
    assert items[0]["spatial"]["K_px"] is None
    assert items[1]["spatial"]["K_px"] == k_px
    assert len({item["payload_hash"] for item in items}) == 3
    for item in items:
        rendered_metadata = json.dumps(item["metadata"])
        assert "path" not in rendered_metadata.lower()
        assert "/" not in rendered_metadata and "\\" not in rendered_metadata


def test_rejects_duplicate_selected_frames_payloads_and_existing_output(tmp_path: Path) -> None:
    duplicate_frame_manifest = _source_manifest(
        tmp_path / "duplicate-frame", source_name="dupe", colors=[(10, 20, 30), (40, 50, 60)], source_indices=[7, 7]
    )
    with pytest.raises(CorpusBuildError, match="duplicates source frame index"):
        _build(tmp_path, [SourceSpec(duplicate_frame_manifest, "source-a", 1)], name="duplicate-frame-out")

    same_selected_manifest = _source_manifest(
        tmp_path / "same-selected", source_name="same", colors=[(2, 4, 6)], source_indices=[9]
    )
    with pytest.raises(CorpusBuildError, match="duplicate selected source frame ownership"):
        _build(tmp_path, [
            SourceSpec(same_selected_manifest, "source-a", 1),
            SourceSpec(same_selected_manifest, "source-a", 1),
        ], name="same-selected-out")

    duplicate_pixel_manifest = _source_manifest(
        tmp_path / "duplicate-pixels", source_name="pixels", colors=[(10, 20, 30), (10, 20, 30)]
    )
    with pytest.raises(CorpusBuildError, match="duplicate decoded raw RGB payload hash"):
        _build(tmp_path, [SourceSpec(duplicate_pixel_manifest, "source-a", 2)], name="duplicate-pixel-out")

    valid_manifest = _source_manifest(tmp_path / "valid", source_name="valid", colors=[(30, 60, 90)])
    _build(tmp_path, [SourceSpec(valid_manifest, "source-a", 1)], name="existing")
    with pytest.raises(CorpusBuildError, match="output root already exists"):
        _build(tmp_path, [SourceSpec(valid_manifest, "source-a", 1)], name="existing")


def test_rejects_shortfall_metadata_mismatch_and_non_jpeg_input(tmp_path: Path) -> None:
    short_manifest = _source_manifest(tmp_path / "short", source_name="short", colors=[(1, 2, 3)])
    with pytest.raises(CorpusBuildError, match="were requested"):
        _build(tmp_path, [SourceSpec(short_manifest, "source-a", 2)], name="short-out")

    wrong_metadata = _source_manifest(
        tmp_path / "wrong-meta", source_name="wrong", colors=[(4, 5, 6)], width=1280, height=720
    )
    with pytest.raises(CorpusBuildError, match="declares 1280x720"):
        _build(tmp_path, [SourceSpec(wrong_metadata, "source-a", 1)], name="wrong-meta-out")

    non_jpeg_root = tmp_path / "non-jpeg"
    non_jpeg_root.mkdir()
    (non_jpeg_root / "not-a-jpeg.jpg").write_bytes(b"not a jpeg")
    non_jpeg = non_jpeg_root / "manifest.json"
    non_jpeg.write_text(json.dumps({
        "schema": "v22_raw_frame_manifest.v0",
        "frames": [{
            "frame_idx": 0, "time_s": 0.0, "rgb": "not-a-jpeg.jpg",
            "source_width": 1920, "source_height": 1080,
        }],
    }))
    with pytest.raises(CorpusBuildError, match="cannot decode JPEG"):
        _build(tmp_path, [SourceSpec(non_jpeg, "source-a", 1)], name="non-jpeg-out")
    assert not (tmp_path / "non-jpeg-out").exists(), "failed builds never expose a partial output root"
