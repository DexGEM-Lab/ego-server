from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ego_annotation.cosmos_semantics import CosmosSemanticError
from ego_annotation.full_video_timeline import InMemoryFrameSource
from ego_annotation.semantic_adapters import SemanticArtifactAdapter


def rows() -> tuple[dict[str, object], ...]:
    common = {"grounding_status": "cosmos_gallery_boundary_video_understanding", "source": "cosmos3.reason", "claim_scope": "semantic_only_not_physical_evidence", "per_hand": {}}
    return (
        {**common, "clip_id": "a", "start_frame": 0, "end_frame": 2, "start_s": 0.0, "end_s": 0.5, "duration_s": 0.5, "caption": "first action", "evidence_frames": [0]},
        {**common, "clip_id": "b", "start_frame": 2, "end_frame": 4, "start_s": 0.5, "end_s": 1.0, "duration_s": 0.5, "caption": "second action", "evidence_frames": [2]},
    )


def test_semantic_adapter_writes_full_duration_visible_subtitle(tmp_path: Path) -> None:
    source = InMemoryFrameSource([np.zeros((80, 120, 3), np.uint8) for _ in range(4)], fps=4.0)

    result = SemanticArtifactAdapter().render(rows(), source, tmp_path)

    capture = cv2.VideoCapture(result.subtitle_video)
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(4.0)
    ok, frame = capture.read()
    capture.release()
    assert ok and frame is not None
    assert np.count_nonzero(frame) > 0
    report = json.loads(Path(result.report_json).read_text(encoding="utf-8"))
    assert report["coverage"] == {"start_frame": 0, "end_frame": 4, "fraction": 1.0}
    assert report["claim_scope"] == "semantic_only_not_physical_evidence"


def test_semantic_adapter_reports_and_visibly_marks_enum_anomaly(tmp_path: Path) -> None:
    source = InMemoryFrameSource([np.zeros((80, 120, 3), np.uint8) for _ in range(4)], fps=4.0)
    anomalous = list(rows())
    anomalous[0] = {**anomalous[0], "semantic_anomalies": [{"raw_field": "LA", "raw_value": "black", "normalized_value": "unknown"}]}

    result = SemanticArtifactAdapter().render(anomalous, source, tmp_path)

    report = json.loads(Path(result.report_json).read_text(encoding="utf-8"))
    assert report["status"] == "completed_with_anomalies"
    assert report["anomaly_count"] == 1
    assert report["anomaly_annotated_frame_count"] == 2


def test_semantic_adapter_rejects_gap_and_physical_claim_scope(tmp_path: Path) -> None:
    source = InMemoryFrameSource([np.zeros((20, 20, 3), np.uint8) for _ in range(4)], fps=4.0)
    bad = list(rows())
    bad[1] = {**bad[1], "start_frame": 3}
    with pytest.raises(CosmosSemanticError, match="contiguous"):
        SemanticArtifactAdapter().render(bad, source, tmp_path)
    promoted = list(rows())
    promoted[0] = {**promoted[0], "claim_scope": "physical_contact_proof"}
    with pytest.raises(CosmosSemanticError, match="claim scope"):
        SemanticArtifactAdapter().render(promoted, source, tmp_path)
