from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.package_v22_annotation_result import create_result_package, resolve_download_package


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_result_package_contains_final_overlay_at_root(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    overlay = run_root / "renders" / "v22_overlay.mp4"
    hybrid = run_root / "renders" / "v22_hybrid_hand_overlay.mp4"
    depth = run_root / "renders" / "v22_depth_overlay.mp4"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_bytes(b"final-overlay")
    hybrid.write_bytes(b"hybrid-overlay")
    depth.write_bytes(b"depth-overlay")
    product = run_root / "product_bundle" / "job_product" / "manifest.json"
    write_json(product, {"schema": "ego.annotation.output", "status": "completed_with_errors"})
    write_json(product.parent / "tables" / "frames.ndjson", {"frame_idx": 0})
    write_json(
        run_root / "annotation_pipeline_manifest.json",
        {
            "case_id": "job_a",
            "renders": {
                "v22_overlay": str(overlay),
                "hybrid_hand_overlay": str(hybrid),
                "depth_overlay": str(depth),
                "overlay_source": "hybrid_hand_state",
            },
            "product_manifest_path": str(product),
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
        assert zf.read("v22_overlay.mp4") == b"final-overlay"
        package_manifest = json.loads(zf.read("package_manifest.json").decode("utf-8"))
        assert package_manifest["final_overlay"] == "v22_overlay.mp4"
        assert package_manifest["render_source"] == "hybrid_hand_state"


def test_safe_download_path_rejects_traversal(tmp_path: Path) -> None:
    package_root = tmp_path / "downloads"
    package_root.mkdir()
    package = package_root / "job.zip"
    package.write_bytes(b"zip")
    assert resolve_download_package("job.zip", package_root=package_root) == package.resolve()
    for bad in ("../job.zip", "nested/job.zip", "job.txt"):
        assert resolve_download_package(bad, package_root=package_root) is None
