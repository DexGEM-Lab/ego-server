"""CPU-only contract tests for V22-derived Hands/WiLoR envelope payloads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ego_annotation.serving.benchmark.hands_payload_corpus import SourceSpec, build_hands_wilor_payload_corpus
from ego_annotation.serving.benchmark.manifest import PAYLOAD_SOURCE_SCHEMA, _hash_parts, load_payload_manifest
from ego_annotation.serving.benchmark.hands_payload_corpus import _hands_template, _wilor_template
from ego_annotation.serving.hands_transport import hands_detect_gateway_request, wilor_reconstruct_gateway_request
from ego_annotation.serving.router import ModelApiName


def _manifest(root: Path) -> Path:
    root.mkdir(parents=True)
    image = Image.new("RGB", (1920, 1080), (12, 34, 56)); image.putpixel((1, 1), (255, 0, 255))
    image.save(root / "0000.jpg", "JPEG", quality=95, subsampling=0)
    path = root / "manifest.json"
    path.write_text(json.dumps({"schema": "v22_raw_frame_manifest.v0", "fps": 30.0, "frames": [{"frame_idx": 0, "source_time_s": 10.0, "rgb": "0000.jpg", "source_width": 1920, "source_height": 1080}]}))
    return path


def _write_historic(root: Path, api: ModelApiName) -> Path:
    root.mkdir(parents=True)
    if api is ModelApiName.HANDS_DETECT:
        data, shape, dtype, metadata, spatial, revision = bytes(540 * 960 * 3), (540, 960, 3), "uint8", {"options": {}}, {
            "source_size": {"width": 1920, "height": 1080}, "model_size": {"width": 960, "height": 540}, "color_space": "RGB",
            "pixel_transform": {"source_to_model": [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], "model_to_source": [[2, 0, 0], [0, 2, 0], [0, 0, 1]], "resize_mode": "resize", "crop_xywh": None, "pad_ltrb": None},
            "K_px": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
        }, "hands-yolo-sam2.1-hiera-l"
        name = "rgb"
    else:
        data, shape, dtype, metadata, spatial, revision = (np.arange(3 * 256 * 256, dtype=np.float32) / 255).tobytes(), (3, 256, 256), "float32", {
            "handedness": 1, "box_center": [900.0, 500.0], "box_size": 400.0, "img_size": [1920.0, 1080.0], "source_K_px": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], "options": {},
        }, None, "wilor-final-v1"
        name = "crop"
    binary = root / f"{name}.bin"; binary.write_bytes(data)
    item = {"item_id": f"historic-{api.value}", "ownership": {"request_id": "historic-r", "job_id": "historic-j", "item_id": "historic-i", "stage_id": api.value, "source_id": "V22case", "source_timestamp_s": 10.0}, "parts": [{"name": name, "file": binary.name, "shape": list(shape), "dtype": dtype}], "spatial": spatial, "model_revision": revision, "work_units": 1, "source_timestamp_s": 10.0, "payload_hash": _hash_parts([(name, data, shape, dtype)]), "metadata": metadata}
    path = root / f"{api.value}.json"
    path.write_text(json.dumps({"schema": PAYLOAD_SOURCE_SCHEMA, "manifest_id": f"historic-{api.value}", "api_name": api.value, "items": [item]}))
    return path


def test_builds_real_v22_hands_pixels_and_preserved_typed_wilor_crop_with_roundtrip_validation(tmp_path: Path):
    source = _manifest(tmp_path / "v22")
    hands = _write_historic(tmp_path / "hands", ModelApiName.HANDS_DETECT)
    wilor = _write_historic(tmp_path / "wilor", ModelApiName.WILOR_RECONSTRUCT)
    result = build_hands_wilor_payload_corpus(sources=[SourceSpec(source, "V22case")], preserved_hands_manifest=hands, preserved_wilor_manifest=wilor, output_root=tmp_path / "out", manifest_id="hands-v22-test", job_id="h5e1")
    assert result.hands_count == result.wilor_count == 1
    assert result.exclusions == {"hands_missing_v22_source": 0, "wilor_missing_v22_source": 0}
    hands_item = load_payload_manifest(result.hands_descriptor_path, expected_api=ModelApiName.HANDS_DETECT).items[0]
    wilor_item = load_payload_manifest(result.wilor_descriptor_path, expected_api=ModelApiName.WILOR_RECONSTRUCT).items[0]
    hands_request, wilor_request = _hands_template(hands_item), _wilor_template(wilor_item)
    assert hands_request.rgb.shape == (540, 960, 3) and any(hands_request.rgb.data)
    assert hands_item.metadata["v22_source_rgb_sha256"] == hashlib.sha256((tmp_path / "v22" / "0000.jpg").read_bytes()).hexdigest()
    assert wilor_request.crop.data == (tmp_path / "wilor" / "crop.bin").read_bytes()
    assert "/" not in json.dumps(hands_item.metadata) and "\\" not in json.dumps(wilor_item.metadata)
    assert hands_detect_gateway_request(hands_request).parts[0].shape == (540, 960, 3)
    assert wilor_reconstruct_gateway_request(wilor_request).parts[0].shape == (3, 256, 256)


def test_counts_unmatched_evidence_and_refuses_empty_logical_api(tmp_path: Path):
    source = _manifest(tmp_path / "v22")
    hands = _write_historic(tmp_path / "hands", ModelApiName.HANDS_DETECT)
    wilor = _write_historic(tmp_path / "wilor", ModelApiName.WILOR_RECONSTRUCT)
    raw = json.loads(hands.read_text()); raw["items"][0]["ownership"]["source_timestamp_s"] = 11.0; hands.write_text(json.dumps(raw))
    try:
        build_hands_wilor_payload_corpus(sources=[SourceSpec(source, "V22case")], preserved_hands_manifest=hands, preserved_wilor_manifest=wilor, output_root=tmp_path / "out")
    except ValueError as exc:
        assert "no complete request" in str(exc)
    else:
        raise AssertionError("missing V22 match must not publish a partial corpus")
    assert not (tmp_path / "out").exists()
