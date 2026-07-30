"""CPU-only tests for V22-real-frame HaWoR benchmark corpus preparation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ego_annotation.serving.benchmark.hawor_payload_corpus import (
    CorpusBuildError,
    SourceSpec,
    _template_request,
    build_hawor_payload_corpus,
)
from ego_annotation.serving.benchmark.manifest import PAYLOAD_SOURCE_SCHEMA, _hash_parts, load_payload_manifest
from ego_annotation.serving.gateway import ModelServiceGateway, _parse_generic_envelope
from ego_annotation.serving.hawor import decode_crop_batch
from ego_annotation.serving.hawor_contracts import HAWOR_CHUNK_LEN
from ego_annotation.serving.hawor_transport import track_chunk_gateway_request
from ego_annotation.serving.router import ModelApiName, ModelServiceRouter
from ego_annotation.serving.transport import _iter_multipart


def _jpeg(path: Path, index: int) -> None:
    image = Image.new("RGB", (1920, 1080), ((index * 13) % 256, (index * 29) % 256, (index * 47) % 256))
    image.putpixel((index % 1900, index % 1000), (255, 0, 255))
    image.save(path, format="JPEG", quality=95, subsampling=0)


def _v22_manifest(root: Path, *, count: int = 32, timestamp_offset: float = 10.0) -> Path:
    root.mkdir(parents=True)
    frames = []
    for index in range(count):
        filename = f"{index:04d}.jpg"
        _jpeg(root / filename, index)
        frames.append({
            "frame_idx": index, "source_frame_idx": index, "time_s": timestamp_offset + index / 30.0,
            "source_time_s": timestamp_offset + index / 30.0, "rgb": filename,
            "source_width": 1920, "source_height": 1080,
        })
    path = root / "manifest.json"
    path.write_text(json.dumps({"schema": "v22_raw_frame_manifest.v0", "frame_count": count, "fps": 30.0, "frames": frames}))
    return path


def _historic_payload(root: Path, *, chunks: int = 2, source_id: str = "Rec4c71-task_1", timestamp_offset: float = 10.0) -> Path:
    root.mkdir(parents=True)
    items = []
    poses = np.tile(np.eye(4, dtype=np.float32), (HAWOR_CHUNK_LEN, 1, 1))
    timestamps = np.arange(HAWOR_CHUNK_LEN, dtype=np.float64) / 30.0 + timestamp_offset
    poses_data, timestamps_data = poses.tobytes(), timestamps.tobytes()
    for chunk in range(chunks):
        base = chunk * HAWOR_CHUNK_LEN
        # Historic crop content is deliberately a non-source marker. The output must
        # replace it with V22-derived pixels while retaining its typed geometry/evidence.
        crop = np.full((16, 3, 256, 256), 99 + chunk, dtype=np.float32).tobytes()
        rows = [
            ("crop_batch", crop, (16, 3, 256, 256), "float32"),
            ("droid_poses", poses_data, (16, 4, 4), "float32"),
            ("droid_timestamps", timestamps_data, (16,), "float64"),
        ]
        part_rows = []
        for name, data, shape, dtype in rows:
            file_name = f"{chunk:02d}.{name}.bin"
            (root / file_name).write_bytes(data)
            part_rows.append({"name": name, "file": file_name, "shape": list(shape), "dtype": dtype})
        observations = [
            {"frame_index": base + index, "source_timestamp_s": timestamp_offset + (base + index) / 30.0,
             "occlusion_state": "visible", "detection_confidence": 0.9, "side": "right"}
            for index in range(HAWOR_CHUNK_LEN)
        ]
        transforms = [
            {"center": [960.0, 540.0], "scale": 5.4, "img_focal": 1200.0, "img_center": [960.0, 540.0],
             "do_flip": False, "source_size": {"width": 1920, "height": 1080}, "pixel_transform": {"source_to_model": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "model_to_source": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "resize_mode": "identity", "crop_xywh": None, "pad_ltrb": None}}
            for _ in range(HAWOR_CHUNK_LEN)
        ]
        metadata = {
            "track_id": f"right-track-{chunk}", "side": "right", "crop_transforms": transforms,
            "observations": observations,
            "unidepth": {"K_px": [[1200.0, 0.0, 960.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]],
                         "img_focal": 1200.0, "img_center": [960.0, 540.0],
                         "source_size": {"width": 1920, "height": 1080}, "metric_scale": 1.0, "source": "unidepth"},
            "droid_evidence": {"metric_scale": 1.0, "scale_residual": 0.01, "scale_confidence": 0.9,
                               "source": "droid+unidepth_scale"},
            "options": {},
        }
        items.append({
            "item_id": f"historic-{chunk}",
            "ownership": {"request_id": f"historic-request-{chunk}", "job_id": "historic-job", "item_id": f"historic-{chunk}",
                          "stage_id": "hawor.infer_tracks", "source_id": source_id,
                          "source_timestamp_s": timestamp_offset + base / 30.0},
            "parts": part_rows, "spatial": None, "model_revision": "hawor-v1", "work_units": 1,
            "source_timestamp_s": timestamp_offset + base / 30.0, "payload_hash": _hash_parts(rows), "metadata": metadata,
        })
    path = root / "historic-hawor.json"
    path.write_text(json.dumps({"schema": PAYLOAD_SOURCE_SCHEMA, "manifest_id": "historic", "api_name": "hawor.infer_tracks", "items": items}))
    return path


def _build(tmp_path: Path, *, name: str = "corpus", count: int | None = 2):
    raw = _v22_manifest(tmp_path / "v22")
    history = _historic_payload(tmp_path / "historic")
    return build_hawor_payload_corpus(
        sources=[SourceSpec(raw, "Rec4c71-task_1")], historic_payload_manifest=history,
        output_root=tmp_path / name, count=count, manifest_id="hawor-v22-test", job_id="h3e1-test",
    )


def test_builds_real_16_frame_typed_chunks_with_path_free_request_metadata(tmp_path: Path) -> None:
    result = _build(tmp_path)
    descriptor = json.loads(result.descriptor_path.read_text())
    assert descriptor["schema"] == PAYLOAD_SOURCE_SCHEMA
    assert descriptor["item_count"] == 2
    assert len(set(result.payload_hashes)) == 2
    first = descriptor["items"][0]
    assert [part["name"] for part in first["parts"]] == ["crop_batch", "droid_poses", "droid_timestamps"]
    assert first["parts"][0]["shape"] == [16, 3, 256, 256]
    assert first["parts"][0]["dtype"] == "float32"
    assert first["metadata"]["source_frame_indices"] == list(range(16))
    assert first["metadata"]["source_timestamps_s"] == [10.0 + index / 30.0 for index in range(16)]
    assert "data_b64" not in json.dumps(first["metadata"])
    assert "/" not in json.dumps(first["metadata"]) and "\\" not in json.dumps(first["metadata"])

    loaded = load_payload_manifest(result.descriptor_path, expected_api=ModelApiName.HAWOR_INFER_TRACKS)
    typed, _ = _template_request(loaded.items[0])
    crop = decode_crop_batch(typed, lambda data, shape, dtype: np.frombuffer(data, dtype=dtype).reshape(shape))
    historic_crop = (tmp_path / "historic" / "00.crop_batch.bin").read_bytes()
    assert crop.shape == (16, 3, 256, 256)
    assert crop.dtype == np.float32
    assert typed.crop_batch.data != historic_crop
    assert hashlib.sha256(typed.crop_batch.data).hexdigest() == first["metadata"]["crop_sha256"]
    assert typed.droid_evidence is not None
    assert typed.droid_evidence.poses_world_from_camera.shape == (16, 4, 4)


def test_rebuilt_item_has_equivalent_generic_multipart_and_envelope_bytes(tmp_path: Path) -> None:
    result = _build(tmp_path, count=1)
    item = load_payload_manifest(result.descriptor_path, expected_api=ModelApiName.HAWOR_INFER_TRACKS).items[0]
    typed, _ = _template_request(item)
    request = track_chunk_gateway_request(typed)

    class NoNetwork:
        async def post(self, *_args, **_kwargs):
            raise AssertionError("wire construction must not issue HTTP")

    multipart = ModelServiceGateway(ModelServiceRouter.canonical(), NoNetwork(), wire_format="multipart")
    envelope = ModelServiceGateway(ModelServiceRouter.canonical(), NoNetwork(), wire_format="envelope")
    multipart_body, multipart_type = multipart._build_body(request)
    envelope_body, _ = envelope._build_body(request)
    assert isinstance(multipart_body, bytes)
    assert not isinstance(envelope_body, bytes)
    multipart_wire = {name: (data, params) for name, data, params in _iter_multipart(multipart_body, multipart_type)}
    multipart_meta = json.loads(multipart_wire.pop("metadata")[0])
    envelope_meta, envelope_parts = _parse_generic_envelope(b"".join(envelope_body.iovecs))
    assert envelope_meta == multipart_meta
    assert set(envelope_parts) == {"crop_batch", "droid_poses", "droid_timestamps"}
    for name, (data, shape, dtype) in envelope_parts.items():
        m_data, m_params = multipart_wire[name]
        m_shape = tuple(int(dimension) for dimension in m_params["shape"].split(","))
        assert (shape, dtype, data.tobytes()) == (m_shape, m_params["dtype"], m_data)


def test_refuses_misaligned_historic_evidence_and_does_not_publish_partial_root(tmp_path: Path) -> None:
    raw = _v22_manifest(tmp_path / "v22")
    historic = _historic_payload(tmp_path / "historic", timestamp_offset=11.0)
    target = tmp_path / "must-not-exist"
    with pytest.raises(CorpusBuildError, match="timestamp disagrees"):
        build_hawor_payload_corpus(
            sources=[SourceSpec(raw, "Rec4c71-task_1")], historic_payload_manifest=historic,
            output_root=target, count=1,
        )
    assert not target.exists()


def test_refuses_count_shortfall_and_existing_destination(tmp_path: Path) -> None:
    _build(tmp_path, name="existing", count=1)
    raw = _v22_manifest(tmp_path / "v22-other")
    historic = _historic_payload(tmp_path / "historic-other", chunks=1)
    with pytest.raises(CorpusBuildError, match="already exists"):
        build_hawor_payload_corpus(sources=[SourceSpec(raw, "Rec4c71-task_1")], historic_payload_manifest=historic, output_root=tmp_path / "existing")
    with pytest.raises(CorpusBuildError, match="but 2 were requested"):
        build_hawor_payload_corpus(sources=[SourceSpec(raw, "Rec4c71-task_1")], historic_payload_manifest=historic, output_root=tmp_path / "shortfall", count=2)


def test_derives_typed_chunks_from_real_v22_wilor_droid_and_calibration_inputs(tmp_path: Path) -> None:
    raw = _v22_manifest(tmp_path / "case" / "input" / "raw_frame_manifest", count=16)
    evidence = tmp_path / "case"
    wilor = evidence / "measurements/hand_candidates/wilor_v21/wilor_raw_hands.json"
    droid = evidence / "measurements/camera_trajectory/droid_full_frame/droid_dense_trajectory.json"
    calibration = evidence / "state/calibration/v19_camera_calibration_contract.json"
    wilor.parent.mkdir(parents=True)
    droid.parent.mkdir(parents=True)
    calibration.parent.mkdir(parents=True)
    (evidence / "requests").mkdir()
    (evidence / "requests/hawor.json").write_text(json.dumps({"job_id": "v22-real-job"}))
    wilor.write_text(json.dumps({"frames": [
        {"frame_idx": index, "raw_hands": [{"side": "right", "detector_score": 0.8,
                                                  "bbox_xyxy": [600.0, 300.0, 900.0, 700.0]}]}
        for index in range(16)
    ]}))
    droid.write_text(json.dumps({"frames": [
        {"frame_idx": index, "T_world_camera": np.eye(4).tolist()} for index in range(16)
    ]}))
    calibration.write_text(json.dumps({"intrinsics_fx_fy_cx_cy": [1200.0, 1200.0, 960.0, 540.0]}))
    result = build_hawor_payload_corpus(
        sources=[SourceSpec(raw, "Rec4c71-task_1", evidence)], output_root=tmp_path / "derived", count=1,
        manifest_id="derived-test", job_id="h3e1-test",
    )
    descriptor = json.loads(result.descriptor_path.read_text())
    assert descriptor["evidence_source"]["kind"] == "derived-from-v22-pipeline-inputs"
    assert descriptor["evidence_source"]["sources"][0]["job_id"] == "v22-real-job"
    assert descriptor["exclusions"] == {
        "policy": "missing frame, WiLoR detection, DROID pose, or calibration excludes a chunk; no filling",
        "complete_chunk_count": 1, "published_chunk_count": 1,
    }
    item = descriptor["items"][0]
    assert item["metadata"]["droid_evidence"]["source"] == "derived-v22-droid-unit-gauge"
    assert len(item["metadata"]["source_frame_sha256"]) == 16
    assert all(len(value) == 64 for value in item["metadata"]["source_frame_sha256"])
    typed, _ = _template_request(load_payload_manifest(result.descriptor_path).items[0])
    assert typed.side.value == "right"
    assert typed.crop_transforms[0].center == (750.0, 500.0)
