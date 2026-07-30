#!/usr/bin/env python3
"""Create a downloadable V22 annotation result package.

The package is intentionally product-facing: it includes the final overlay at
zip root plus manifests, render videos, and the product bundle. It does not copy
raw decoded frames or bulky intermediate model archives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PackageError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PackageError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_package_name(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw).strip("._-")
    return cleaned[:220] or "annotation_result"


def resolve_download_package(filename: str, package_root: Path) -> Path | None:
    if filename != Path(filename).name or not filename.endswith(".zip"):
        return None
    root = package_root.expanduser().resolve()
    path = (root / filename).resolve()
    if path.parent != root or not path.exists() or not path.is_file():
        return None
    return path


def resolve_existing_path(run_root: Path, value: Any, default: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if isinstance(value, str) and value:
        path = Path(value)
        candidates.append(path if path.is_absolute() else run_root / path)
    if default is not None:
        candidates.append(default)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def add_file(zf: zipfile.ZipFile, source: Path, arcname: str, added: set[str], included: list[dict[str, Any]]) -> None:
    if not source.exists() or not source.is_file():
        return
    normalized = str(Path(arcname).as_posix()).lstrip("/")
    if normalized in added:
        return
    if normalized.startswith("../") or "/../" in normalized:
        raise PackageError(f"unsafe archive path: {arcname}")
    zf.write(source, normalized)
    added.add(normalized)
    included.append({"path": normalized, "source": str(source), "size_bytes": int(source.stat().st_size), "sha256": sha256_file(source)})


def add_required_file(zf: zipfile.ZipFile, source: Path, arcname: str, added: set[str], included: list[dict[str, Any]]) -> None:
    if not source.exists() or not source.is_file():
        raise PackageError(f"missing required package evidence: {source}")
    add_file(zf, source, arcname, added, included)


def add_tree(zf: zipfile.ZipFile, source_dir: Path, arc_prefix: str, added: set[str], included: list[dict[str, Any]]) -> None:
    if not source_dir.exists() or not source_dir.is_dir():
        return
    for source in sorted(source_dir.rglob("*")):
        if source.is_file():
            rel = source.relative_to(source_dir).as_posix()
            add_file(zf, source, f"{arc_prefix.rstrip('/')}/{rel}", added, included)


def ensure_timing_ledger(run_root: Path) -> Path:
    ledger = run_root / "state" / "timing_ledger.json"
    if ledger.exists() and ledger.is_file():
        return ledger
    builder = Path(__file__).resolve().with_name("build_v22_timing_ledger.py")
    cmd = [sys.executable, str(builder), "--run-root", str(run_root), "--output", str(ledger)]
    proc = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise PackageError(
            f"timing ledger build failed rc={proc.returncode}\n"
            f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
        )
    if not ledger.exists() or not ledger.is_file():
        raise PackageError(f"timing ledger build did not create expected file: {ledger}")
    return ledger


def create_result_package(run_root: Path, package_root: Path, *, package_name: str | None = None) -> dict[str, Any]:
    run_root = run_root.resolve()
    package_root = package_root.resolve()
    manifest_path = run_root / "annotation_pipeline_manifest.json"
    if not manifest_path.exists():
        raise PackageError(f"missing pipeline manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    job_id = str(manifest.get("case_id") or run_root.name)
    safe_name = clean_package_name(package_name or f"{job_id}_annotation_result")
    package_root.mkdir(parents=True, exist_ok=True)
    output_zip = package_root / f"{safe_name}.zip"
    temp_zip = package_root / f".{safe_name}.{os.getpid()}.tmp"

    renders = manifest.get("renders") if isinstance(manifest.get("renders"), dict) else {}
    api_ify_manifest = manifest.get("pipeline") == "single_video_api_ify"
    render_key = "v22_combined" if api_ify_manifest else "v22_overlay"
    fallback = run_root / "renders" / ("v22_combined.mp4" if api_ify_manifest else "v22_overlay.mp4")
    final_render = resolve_existing_path(run_root, renders.get(render_key), fallback)
    if final_render is None:
        raise PackageError(f"missing final render: expected renders.{render_key} or {fallback.name}")
    final_render_name = final_render.name
    if not api_ify_manifest:
        ensure_timing_ledger(run_root)

    product_manifest = resolve_existing_path(run_root, manifest.get("product_manifest_path"))
    product_root = product_manifest.parent if product_manifest is not None else None
    included: list[dict[str, Any]] = []
    added: set[str] = set()
    package_manifest = {
        "schema": "ego.annotation.download_package.v0",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "job_id": job_id,
        "final_overlay": final_render_name,
        "pipeline_manifest": "annotation_pipeline_manifest.json",
        "product_manifest": "product_bundle/manifest.json" if product_manifest is not None else None,
        "render_source": renders.get("render_source") if api_ify_manifest else renders.get("overlay_source"),
        "claim_scope": "Download package for annotation outputs; final overlay is included at zip root for user inspection.",
    }

    try:
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            add_file(zf, final_render, final_render_name, added, included)
            add_file(zf, manifest_path, "annotation_pipeline_manifest.json", added, included)
            if api_ify_manifest:
                for rel in (
                    "state/v22_physical_state.npz",
                    "run_result.json",
                    "integrated_run_report.json",
                    "integrated_run_report.md",
                    "renders/physical_adapter_report.json",
                    "state/semantic_clips/v22_cosmos_semantic_review.json",
                    "stage_captures/fixture_index.json",
                ):
                    add_required_file(zf, run_root / rel, rel, added, included)
                add_tree(zf, run_root / "stage_captures", "stage_captures", added, included)
            else:
                for key in ("v22_overlay", "hand_overlay", "hybrid_hand_overlay", "world_head_hand_3d", "semantic_subtitle", "depth_overlay"):
                    path = resolve_existing_path(run_root, renders.get(key))
                    if path is not None:
                        add_file(zf, path, f"renders/{path.name}", added, included)
                for rel in (
                    "state/timing_ledger.json",
                    "logs/gpu_usage_snapshots.jsonl",
                    "logs/gpu_wrapper_events.jsonl",
                    "requests/unidepth.json",
                    "requests/wilor.json",
                    "requests/droid.json",
                    "requests/hawor.json",
                ):
                    add_required_file(zf, run_root / rel, rel, added, included)
                for rel in (
                    "measurements/camera_trajectory/droid_full_frame/droid_shared_geometry.json",
                    "measurements/camera_trajectory/droid_full_frame/droid_qc.json",
                    "measurements/camera_trajectory/droid_full_frame/v22_camera_trajectory_stage.json",
                    "measurements/hand_candidates/hawor_world/hawor_motion_preparation.json",
                    "measurements/hand_candidates/hawor_world/v22_hawor_motion_preparation_stage.json",
                    "measurements/hand_candidates/hawor_world/hawor_slam_adapter_report.json",
                    "measurements/hand_candidates/hawor_world/qc_hawor_world_hands.json",
                    "measurements/hand_candidates/hawor_world/v22_hawor_metric_hand_stage.json",
                    "requests/hawor_prepare.json",
                    "renders/v22_hybrid_hand_overlay_report.json",
                    "renders/v22_world_head_hand_3d_report.json",
                    "renders/v22_primary_overlay_report.json",
                    "renders/v22_semantic_subtitle_report.json",
                    "state/annotations_v22_renderable.json",
                    "state/v22_physical_state.json",
                    "state/v22_uncertainty_state.json",
                    "state/agent_evidence_review.json",
                    "state/gt_free_self_calibration/v22_gt_free_drift_self_calibration.json",
                    "state/semantic_clips/v22_captioning_stage.json",
                    "state/semantic_clips/v22_cosmos_semantic_review.json",
                    "state/semantic_clips/v22_semantic_review.json",
                    "state/self_consistency/v22_full_self_consistency_qc.json",
                    "evaluation/v22_evaluator_stage.json",
                ):
                    add_file(zf, run_root / rel, rel, added, included)
                if product_manifest is not None:
                    add_file(zf, product_manifest, "product_bundle/manifest.json", added, included)
                if product_root is not None:
                    add_tree(zf, product_root, "product_bundle", added, included)
            package_manifest["included_files"] = included
            zf.writestr("package_manifest.json", json.dumps(package_manifest, indent=2, ensure_ascii=False))
        shutil.move(str(temp_zip), str(output_zip))
    finally:
        if temp_zip.exists():
            temp_zip.unlink()

    integrated_report = resolve_existing_path(
        run_root,
        manifest.get("integrated_run_report_path"),
        run_root / "integrated_run_report.json",
    )
    result = {
        "status": "ok",
        "package_path": str(output_zip),
        "package_name": output_zip.name,
        "size_bytes": int(output_zip.stat().st_size),
        "sha256": sha256_file(output_zip),
        "run_root": str(run_root),
        "final_overlay_arcname": final_render_name,
        "final_video_path": str(final_render),
        "final_report_path": str(integrated_report) if integrated_report is not None else None,
        "included_file_count": len(included) + 1,
        "created_s": time.time(),
    }
    write_json(package_root / f"{safe_name}.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--package-name", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = create_result_package(args.run_root, args.package_root, package_name=args.package_name)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
