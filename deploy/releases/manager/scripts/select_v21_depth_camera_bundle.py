#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected_json_object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def resolve_existing_path(run_root: Path, raw: Any) -> str | None:
    if raw is None:
        return None
    path = Path(str(raw))
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([run_root / path, Path.cwd() / path])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(path)


def candidate_from_depthpro(run_root: Path) -> dict[str, Any] | None:
    report_path = run_root / "measurements" / "depth_candidates" / "depthpro_full_frame" / "qc_depthpro_full_frame_v21.json"
    depth_archive = run_root / "measurements" / "depth_candidates" / "depthpro_full_frame" / "depthpro_full_frame_depth_v21.npz"
    report = read_if_exists(report_path)
    if report is None and not depth_archive.exists():
        return None
    return {
        "candidate_id": "depthpro",
        "kind": "monocular_metric_depth_focal",
        "backend": "Depth Pro",
        "status": report.get("status", "ok") if report else "ok_cached_archive",
        "report": str(report_path) if report_path.exists() else None,
        "depth_archive": str(depth_archive),
        "frame_count": report.get("frame_count") if report else None,
        "coverage": "full_or_strided_timeline_from_report",
        "focal_px": report.get("depthpro_focal_px") if report else None,
        "depth_median_m": report.get("depth_median_m") if report else None,
        "valid_fraction": report.get("valid_fraction") if report else None,
        "metric_depth_available": True,
        "selection_eligibility": "eligible_monocular_baseline",
    }


def candidate_from_unidepth(run_root: Path) -> dict[str, Any] | None:
    current_dir = run_root / "measurements" / "depth_candidates" / "unidepth_v2"
    current_npz = current_dir / "unidepth_v2_depth.npz"
    legacy_report = run_root / "measurements" / "depth_candidates" / "unidepth_full_frame" / "qc_unidepth_full_frame_v3.json"
    legacy = read_if_exists(legacy_report)
    if not current_npz.exists() and legacy is None:
        return None
    return {
        "candidate_id": "unidepth_v2",
        "kind": "monocular_metric_depth_intrinsics",
        "backend": "UniDepth v2",
        "status": legacy.get("status", "ok") if legacy else "ok_cached_archive",
        "report": str(legacy_report) if legacy_report.exists() else None,
        "depth_archive": str(current_npz if current_npz.exists() else legacy.get("depth_archive")),
        "frame_count": legacy.get("frames") if legacy else None,
        "focal_px": legacy.get("unidepth_focal_px") if legacy else None,
        "depth_median_m": legacy.get("depth_median_m") if legacy else None,
        "metric_depth_available": True,
        "selection_eligibility": "eligible_monocular_baseline",
    }


def candidate_from_depth_anything(run_root: Path) -> dict[str, Any] | None:
    depth_archive = run_root / "measurements" / "depth_candidates" / "depth_anything_v2" / "depth_anything_v2_depth.npz"
    if not depth_archive.exists():
        return None
    return {
        "candidate_id": "depth_anything_v2",
        "kind": "relative_monocular_depth",
        "backend": "Depth Anything v2",
        "status": "ok_cached_archive",
        "report": None,
        "depth_archive": str(depth_archive),
        "metric_depth_available": False,
        "selection_eligibility": "diagnostic_scale_aligned_only",
    }


def candidate_from_stereo_sgbm(run_root: Path) -> dict[str, Any] | None:
    report_path = run_root / "measurements" / "depth_candidates" / "stereo_sgbm" / "report.json"
    report = read_if_exists(report_path)
    if report is None:
        return None
    return {
        "candidate_id": "stereo_sgbm",
        "kind": "uncalibrated_stereo_relative_inverse_depth",
        "backend": "OpenCV SGBM",
        "status": report.get("status"),
        "report": str(report_path),
        "depth_archive": report.get("output_npz"),
        "frame_count": report.get("frame_count"),
        "valid_fraction": report.get("valid_fraction"),
        "median_disparity_px": report.get("median_disparity_px"),
        "metric_depth_available": False,
        "selection_eligibility": "diagnostic_assisted_depth_only_not_metric_primary",
    }


def registry_candidates(run_root: Path) -> list[dict[str, Any]]:
    registry_path = run_root / "measurements" / "camera_depth" / "v20_depth_registry" / "depth_candidate_registry.json"
    registry = read_if_exists(registry_path)
    rows = registry.get("candidates") if isinstance(registry, dict) else None
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def v20_selected_primary(run_root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    report_path = run_root / "measurements" / "camera_depth" / "v20_depth_selection_report.json"
    report = read_if_exists(report_path)
    if report is None:
        return None, []
    selected = report.get("selected") if isinstance(report.get("selected"), dict) else {}
    primary_id = selected.get("primary_for_visible_surface") or selected.get("primary_for_hand_depth")
    if not primary_id:
        return None, ["V20 depth selection report exists but did not select a primary candidate."]
    candidates = registry_candidates(run_root)
    candidate = next((row for row in candidates if str(row.get("candidate_id")) == str(primary_id)), None)
    if not candidate:
        return None, [f"V20 selected candidate {primary_id} was not found in depth registry."]
    archive = resolve_existing_path(run_root, candidate.get("depth_npz") or candidate.get("depth_candidate_npz") or candidate.get("npz_path"))
    if not archive:
        return None, [f"V20 selected candidate {primary_id} has no depth archive path."]
    return {
        "candidate_id": str(primary_id),
        "kind": candidate.get("method_family"),
        "backend": candidate.get("candidate_id"),
        "status": "selected_by_v20_residual_bundle",
        "report": str(report_path),
        "depth_archive": archive,
        "metric_depth_available": candidate.get("method_family") != "relative_monocular_depth",
        "selection_eligibility": "selected_by_v20_depth_observation_bundle",
        "candidate": candidate,
    }, [f"Primary depth selected by V20 residual bundle: {primary_id}."]


def choose_primary(run_root: Path, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[str], str]:
    selected, notes = v20_selected_primary(run_root)
    if selected is not None:
        return selected, notes, "v20_depth_selection_bundle"
    depthpro = next((c for c in candidates if c["candidate_id"] == "depthpro" and str(c.get("status", "")).startswith("ok")), None)
    unidepth = next((c for c in candidates if c["candidate_id"] == "unidepth_v2" and str(c.get("status", "")).startswith("ok")), None)
    if depthpro and unidepth:
        return depthpro, ["Depth Pro selected as provisional primary; UniDepth v2 retained as independent metric comparator because no V20 selection was available."], "v21_provisional_metric_priority"
    if depthpro:
        return depthpro, ["Depth Pro selected as provisional monocular primary. UniDepth comparator missing, so assisted-depth selection remains provisional."], "v21_provisional_metric_priority"
    if unidepth:
        return unidepth, ["UniDepth v2 selected as provisional monocular primary. Depth Pro comparator missing, so assisted-depth selection remains provisional."], "v21_provisional_metric_priority"
    return None, ["No monocular metric baseline is available; stereo/relative candidates cannot be selected as metric primary under V21 policy."], "no_metric_primary"


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root
    input_manifest = load_json(args.input_manifest)
    modality_report = load_json(args.modality_report)
    candidates = []
    for candidate in [
        candidate_from_depthpro(run_root),
        candidate_from_unidepth(run_root),
        candidate_from_depth_anything(run_root),
        candidate_from_stereo_sgbm(run_root),
    ]:
        if candidate is not None:
            candidates.append(candidate)
    primary, notes, selection_source = choose_primary(run_root, candidates)
    report = {
        "schema": "v21_depth_camera_selection_report.v1",
        "status": "ok" if primary is not None else "blocked_no_monocular_metric_baseline",
        "method": "select_v21_depth_camera_bundle",
        "case_id": input_manifest.get("case_id"),
        "run_root": str(run_root),
        "input_manifest": str(args.input_manifest),
        "modality_report": str(args.modality_report),
        "v20_depth_selection_report": str(run_root / "measurements" / "camera_depth" / "v20_depth_selection_report.json"),
        "v20_depth_registry": str(run_root / "measurements" / "camera_depth" / "v20_depth_registry" / "depth_candidate_registry.json"),
        "candidates": candidates,
        "selected_primary_depth_camera": primary,
        "selection_source": selection_source,
        "selection_notes": notes,
        "policy": modality_report.get("monocular_baseline_policy"),
        "assisted_not_worse_policy": modality_report.get("assisted_not_worse_policy"),
        "claim_scope": "V21 depth/camera bundle selection. Prefer the V20 residual-driven multi-source depth bundle when available; otherwise keep a provisional monocular metric primary and retain other candidates as uncertainty evidence.",
    }
    write_json(args.output_report, report)
    state_path = run_root / "state" / "v21_physical_state.json"
    if state_path.exists():
        state = load_json(state_path)
    else:
        state = {"schema": "v21_physical_state.v0", "case_id": input_manifest.get("case_id"), "run_root": str(run_root)}
    if primary is not None:
        state["camera_depth"] = {
            "state": "selected_multi_source_depth_candidate" if selection_source == "v20_depth_selection_bundle" else "selected_provisional_monocular_metric_depth_candidate",
            "selection_report": str(args.output_report),
            "selection_source": selection_source,
            "primary_candidate_id": primary["candidate_id"],
            "primary_depth_archive": primary.get("depth_archive"),
            "metric_depth_available": bool(primary.get("metric_depth_available", True)),
            "provisional": selection_source != "v20_depth_selection_bundle",
        }
    else:
        state["camera_depth"] = {
            "state": "blocked_no_monocular_metric_baseline",
            "selection_report": str(args.output_report),
            "metric_depth_available": False,
            "next_required_action": "run_depthpro_or_unidepth_monocular_baseline",
        }
    write_json(state_path, state)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select V21 depth/camera bundle after monocular baseline and assisted candidates run.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--modality-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
