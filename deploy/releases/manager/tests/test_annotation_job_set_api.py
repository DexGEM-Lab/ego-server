from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from scripts import serve_v22_annotation_api as api


def test_annotation_job_set_endpoint_is_disabled_for_single_video_mvp() -> None:
    with pytest.raises(api.HTTPException) as exc_info:
        api.run_annotation_job_set(api.AnnotationJobSetRequest(job_id="set_job", video_uris=["/remote/data/a.mp4"]))
    assert exc_info.value.status_code == 410
    assert exc_info.value.detail["code"] == "annotation_job_sets_disabled"
    assert "exactly one uploaded video" in exc_info.value.detail["reason"]
