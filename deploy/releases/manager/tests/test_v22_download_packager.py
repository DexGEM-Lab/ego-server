from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.package_v22_annotation_result import PackageError, create_result_package, resolve_download_package


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_result_package_contains_final_overlay_at_root(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    overlay = run_root / "renders" / "v22_overlay.mp4"
    hybrid = run_root / "renders" / "v22_hybrid_hand_overlay.mp4"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_bytes(b"final-overlay")
    hybrid.write_bytes(b"hybrid-overlay")
    product = run_root / "product_bundle" / "job_product" / "manifest.json"
    write_json(product, {"schema": "ego.annotation.output", "status": "completed_with_errors"})
    write_json(product.parent / "tables" / "frames.ndjson", {"frame_idx": 0})
    write_json(run_root / "requests" / "unidepth.json", {"input_video": "/input.mp4", "output_dir": "/unidepth"})
    write_json(run_root / "requests" / "wilor.json", {"input_video": "/input.mp4", "output_dir": "/wilor"})
    write_json(run_root / "requests" / "droid.json", {"input_video": "/input.mp4", "camera": {}, "output_dir": "/droid"})
    write_json(run_root / "requests" / "hawor.json", {"input_video": "/input.mp4", "camera": {}, "output_dir": "/hawor"})
    write_json(run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_shared_geometry.json", {"backend": "droid", "droid_invocation": {"instance_count": 1}})
    write_json(run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_motion_preparation.json", {"schema": "v22_hawor_motion_preparation.v0", "status": "ok"})
    write_json(run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_slam_adapter_report.json", {"legacy_hawor_droid_executed": False})
    gpu_log = run_root / "logs" / "gpu_usage_snapshots.jsonl"
    gpu_log.parent.mkdir(parents=True, exist_ok=True)
    gpu_log.write_text('{"status":"unavailable","stage":"unidepth"}\n', encoding="utf-8")
    (run_root / "logs" / "gpu_wrapper_events.jsonl").write_text('{"event":"start","module_id":"M02"}\n', encoding="utf-8")
    write_json(
        run_root / "input" / "raw_frame_manifest" / "manifest.json",
        {"frame_count": 2, "fps": 30.0, "duration_s": 0.0667, "frames": [{"frame_idx": 0}, {"frame_idx": 1}]},
    )
    write_json(
        run_root / "annotation_pipeline_manifest.json",
        {
            "case_id": "job_a",
            "steps": [
                {"step": "prepare_single_video", "status": "ok", "returncode": 0, "elapsed_s": 0.01, "log": "prepare.log"},
                {"step": "product_annotation_bundle", "status": "ok", "returncode": 0, "elapsed_s": 0.02, "log": "bundle.log"},
            ],
            "renders": {
                "v22_overlay": str(overlay),
                "hybrid_hand_overlay": str(hybrid),
                "overlay_source": "hybrid_hand_state",
            },
            "product_manifest_path": str(product),
            "execution_topology": "single_item_intra_video_parallel_model_lanes",
            "parallel_groups": [{"group": "post_ingest_model_lanes", "status": "ok", "lanes": ["D3_unidepth", "D6_wilor"], "elapsed_s": 1.0}],
        },
    )
    result = create_result_package(run_root, tmp_path / "downloads")
    package_path = Path(result["package_path"])
    assert package_path.exists()
    with zipfile.ZipFile(package_path) as zf:
        names = set(zf.namelist())
        assert "v22_overlay.mp4" in names
        assert "annotation_pipeline_manifest.json" in names
        assert "package_manifest.json" in names
        assert "product_bundle/manifest.json" in names
        assert "product_bundle/tables/frames.ndjson" in names
        assert "requests/unidepth.json" in names
        assert "requests/wilor.json" in names
        assert "requests/droid.json" in names
        assert "requests/hawor.json" in names
        assert "measurements/camera_trajectory/droid_full_frame/droid_shared_geometry.json" in names
        assert "measurements/hand_candidates/hawor_world/hawor_motion_preparation.json" in names
        assert "measurements/hand_candidates/hawor_world/hawor_slam_adapter_report.json" in names
        assert "logs/gpu_usage_snapshots.jsonl" in names
        assert "renders/v22_depth_overlay.mp4" not in names
        assert "state/timing_ledger.json" in names
        timing_ledger = json.loads(zf.read("state/timing_ledger.json").decode("utf-8"))
        assert timing_ledger["schema"] == "v22.single_item_timing_ledger.v0"
        assert timing_ledger["status"] == "ok"
        assert timing_ledger["execution_topology"] == "single_item_intra_video_parallel_model_lanes"
        assert timing_ledger["parallel_groups"][0]["group"] == "post_ingest_model_lanes"
        assert zf.read("v22_overlay.mp4") == b"final-overlay"
        package_manifest = json.loads(zf.read("package_manifest.json").decode("utf-8"))
        assert package_manifest["final_overlay"] == "v22_overlay.mp4"
        assert package_manifest["render_source"] == "hybrid_hand_state"


def test_result_package_requires_request_and_gpu_evidence(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    overlay = run_root / "renders" / "v22_overlay.mp4"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_bytes(b"final-overlay")
    write_json(
        run_root / "annotation_pipeline_manifest.json",
        {
            "case_id": "job_missing_evidence",
            "steps": [],
            "renders": {"v22_overlay": str(overlay)},
        },
    )
    write_json(
        run_root / "input" / "raw_frame_manifest" / "manifest.json",
        {"frame_count": 1, "fps": 30.0, "duration_s": 0.033, "frames": [{"frame_idx": 0}]},
    )

    with pytest.raises(PackageError, match="missing required package evidence"):
        create_result_package(run_root, tmp_path / "downloads")


def test_safe_download_path_rejects_traversal(tmp_path: Path) -> None:
    package_root = tmp_path / "downloads"
    package_root.mkdir()
    package = package_root / "job.zip"
    package.write_bytes(b"zip")
    assert resolve_download_package("job.zip", package_root=package_root) == package.resolve()
    for bad in ("../job.zip", "nested/job.zip", "job.txt"):
        assert resolve_download_package(bad, package_root=package_root) is None
