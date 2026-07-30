#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Assemble the current V18 interval-MANO frontier artifact.

This is not a new physical factor.  It is the artifact-consumption step for the
current workbench state: expose the full-video interval MANO renders and backing
solver states that actually drive the present hand annotation, instead of the
older sparse H-prime final roots that have been ruled out for the MANO objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_OUTPUT_ROOT = Path("/data2/ego_annotation_outputs/v18_current_frontier_interval_mano_artifact_v5")
DEFAULT_CASE_RENDER_ROOTS = {
    "task5_tomato_960": Path("/data2/ego_annotation_outputs/v18_task5_joint_mano_surface_support_uncertain_sanitized_base_full_video_v1/task5_tomato_960"),
    "trash_1050": Path("/data2/ego_annotation_outputs/v18_trash_handobs_extend_1004_1008_posterior_full_video_v1/trash_1050"),
}
SANITIZED_ANNOTATION_ROOT = "/data2/ego_annotation_outputs/v18_full_pipeline_sanitized_base_for_hprime"
REJECTED_HPRIME_ROOT = "/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard"

DEFAULT_REVIEW_FRAMES = {
    "task5_tomato_960": [481, 499, 525, 648, 690, 720, 780, 873, 902],
    "trash_1050": [720, 735, 779, 824, 830, 869, 893, 958, 970, 971, 972, 988, 1000, 1002, 1004, 1006, 1008, 1009, 1020, 1027],
}
CASE_CLAIMS = {
    "task5_tomato_960": {
        "frontier_mechanism": "support_bounded_interval_mano",
        "closure_role": "scoped_v18_component_with_explicit_support_limit",
        "claim": "Tomato exact-surface/contact residuals are mostly below independent object-support uncertainty; the scoped V18 MANO deliverable therefore renders support-bounded hand uncertainty rather than forcing a confident millimetre contact/nonpenetration correction.",
        "not_claimed": [
            "solved contact state",
            "accepted hidden tomato volume nonpenetration",
            "millimetre-accurate object pose",
        ],
    },
    "trash_1050": {
        "frontier_mechanism": "latent_occlusion_transition_plus_occluded_translation_posterior_interval_mano",
        "closure_role": "scoped_v18_component_with_explicit_occlusion_information_limit",
        "claim": "Late trash MANO observations become invalid as the hand transitions under the lid; the scoped V18 MANO deliverable keeps an optimized latent trajectory only where supported and visibly exposes a one-dimensional additional camera-z hard-bound feasible/energy stress-test for zero-observation rows. The explicit observation-invalid interval now covers the original hard occlusion and the visually continuous left-hand 1004-1008 boundary rows; because the solve is interval-coupled, those five directly zeroed rows also create small temporal propagation through later left-hand rows rather than an isolated-frame patch. The selected first-surface evidence mostly exceeds the translation bound, and the optimizer falls back to feasible representative points, so the hidden hand is represented as broad/conflicted uncertainty rather than a known reconstructed pose or optimized MAP trajectory.",
        "not_claimed": [
            "known hidden-hand pose",
            "hidden articulation reconstruction",
            "optimized MAP trajectory through failed optimizer rows",
            "calibrated posterior probability distribution",
            "solved hand-lid contact",
            "accepted compact-lid hidden-volume nonpenetration",
        ],
    },
}
EXPECTED_VIDEO_NAMES = {
    "overlay": "v18_overlay_joint_mano_full_video_correction.mp4",
    "world": "v18_world_joint_mano_full_video_correction.mp4",
    "side_by_side": "v18_side_by_side_joint_mano_full_video_correction.mp4",
}
STANDARD_VIDEO_NAMES = {
    "overlay": "v18_overlay.mp4",
    "world": "v18_world.mp4",
    "side_by_side": "v18_side_by_side.mp4",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def parse_case_render_roots(items: list[str]) -> dict[str, Path]:
    if not items:
        return dict(DEFAULT_CASE_RENDER_ROOTS)
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--case-render-root must be CASE=PATH, got {item!r}")
        case, raw = item.split("=", 1)
        case = case.strip()
        if not case:
            raise ValueError(f"empty case in {item!r}")
        out[case] = Path(raw).expanduser()
    return out


def parse_review_frames(items: list[str]) -> dict[str, list[int]]:
    out = {k: list(v) for k, v in DEFAULT_REVIEW_FRAMES.items()}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--review-frames must be CASE=f0,f1,..., got {item!r}")
        case, raw = item.split("=", 1)
        frames = [int(x) for x in raw.replace(";", ",").split(",") if x.strip()]
        out[case.strip()] = frames
    return out


def copy_or_hardlink(src: Path, dst: Path, *, prefer_hardlink: bool = False) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    method = "copy"
    if prefer_hardlink:
        try:
            os.link(src, dst)
            method = "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            method = "copy"
    else:
        shutil.copy2(src, dst)
    return {"source": str(src), "path": str(dst), "method": method, "bytes": int(dst.stat().st_size)}


def ffprobe_video(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        raw = json.loads(proc.stdout)
        streams = raw.get("streams") if isinstance(raw, dict) else None
        stream_raw = streams[0] if isinstance(streams, list) and streams and isinstance(streams[0], dict) else {}
        width_raw = stream_raw.get("width")
        height_raw = stream_raw.get("height")
        nb_frames_raw = stream_raw.get("nb_frames")
        duration_raw = stream_raw.get("duration")
        width = int(width_raw) if width_raw is not None else None
        height = int(height_raw) if height_raw is not None else None
        if isinstance(nb_frames_raw, str) and nb_frames_raw.isdigit():
            nb_frames: int | str | None = int(nb_frames_raw)
        elif isinstance(nb_frames_raw, int):
            nb_frames = nb_frames_raw
        elif nb_frames_raw is None:
            nb_frames = None
        else:
            nb_frames = str(nb_frames_raw)
        duration_s = float(duration_raw) if duration_raw is not None else None
        return {
            "width": width,
            "height": height,
            "nb_frames": nb_frames,
            "r_frame_rate": stream_raw.get("r_frame_rate"),
            "duration_s": duration_s,
        }
    except Exception as exc:  # ffprobe is evidence, not the artifact source.
        return {"ffprobe_error": repr(exc)}


def summarize_numeric(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
    }


def norm3(raw: Any) -> float | None:
    try:
        arr = np.asarray(raw, dtype=float)
    except Exception:
        return None
    if arr.shape != (3,) or not np.isfinite(arr).all():
        return None
    return float(np.linalg.norm(arr))


def optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def require_sanitized_annotation_input(state: dict[str, Any], path: Path) -> str:
    inputs_raw = state.get("inputs")
    parameters_raw = state.get("parameters")
    inputs: dict[str, Any] = inputs_raw if isinstance(inputs_raw, dict) else {}
    parameters: dict[str, Any] = parameters_raw if isinstance(parameters_raw, dict) else {}
    annotation_input = inputs.get("annotations") or parameters.get("annotations")
    if annotation_input is None:
        raise ValueError(f"{path} does not declare the annotation input used to solve H_t")
    annotation_input_str = str(annotation_input)
    if REJECTED_HPRIME_ROOT in annotation_input_str:
        raise ValueError(f"{path} uses rejected final-v7/H-prime annotation input: {annotation_input_str}")
    if SANITIZED_ANNOTATION_ROOT not in annotation_input_str:
        raise ValueError(f"{path} is not solved from sanitized non-H-prime annotations: {annotation_input_str}")
    return annotation_input_str


def summarize_states(case: str, state_paths: list[Path], artifact_state_copies: list[Path] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    for state_i, path in enumerate(state_paths):
        state = load_json(path)
        rows = state.get("per_frame_states")
        if not isinstance(rows, list):
            raise ValueError(f"{path} has no per_frame_states list")
        summary = state.get("summary") if isinstance(state.get("summary"), dict) else {}
        annotation_input = require_sanitized_annotation_input(state, path)
        interval = {
            "state_path": str(path),
            "annotation_input": annotation_input,
            "artifact_state_copy": str(artifact_state_copies[state_i]) if artifact_state_copies is not None else None,
            "method": state.get("method"),
            "case": state.get("case"),
            "object_id": state.get("object_id"),
            "row_count": len(rows),
            "summary": summary,
            "claim_scope": state.get("claim_scope"),
            "scientific_test": state.get("scientific_test"),
            "inputs": state.get("inputs") if isinstance(state.get("inputs"), dict) else None,
            "parameters": state.get("parameters") if isinstance(state.get("parameters"), dict) else None,
        }
        intervals.append(interval)
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{path} contains non-dict per_frame_state")
            out = dict(row)
            out["source_interval_state_path"] = str(path)
            merged.append(out)
    frame_ids = sorted({int(r["frame_idx"]) for r in merged if "frame_idx" in r})
    sides = sorted({str(r.get("hand_side")) for r in merged if r.get("hand_side") is not None})
    translation_norms = [v for r in merged if (v := norm3(r.get("optimized_translation_world_m"))) is not None]
    ray_shift_norms = [v for r in merged if (v := norm3(r.get("hand_ray_shift_prior_translation_world_m"))) is not None]
    visible_surface_initial = [float(r.get("visible_surface_depth_order_selected_initial_in_front_count") or 0.0) for r in merged]
    visible_surface_final = [float(r.get("visible_surface_depth_order_selected_final_in_front_count") or 0.0) for r in merged]
    support_uncertainty = [float(r.get("observed_surface_support_uncertainty_m") or 0.0) for r in merged]
    contact_rows = [r for r in merged if r.get("contact_patch_factor_state") == "active_contact_patch"]
    hand_obs_zeroed = []
    for r in merged:
        multiplier = optional_float(r.get("hand_observation_visibility_weight_multiplier"))
        if multiplier is not None and multiplier <= 1e-9:
            hand_obs_zeroed.append(r)
    summary = {
        "case": case,
        "optimized_state_count": int(len(merged)),
        "unique_optimized_frame_count": int(len(frame_ids)),
        "first_optimized_frame": int(frame_ids[0]) if frame_ids else None,
        "last_optimized_frame": int(frame_ids[-1]) if frame_ids else None,
        "hand_sides": sides,
        "interval_count": int(len(intervals)),
        "intervals": intervals,
        "optimized_translation_norm_m": summarize_numeric(translation_norms),
        "hand_ray_shift_prior_norm_m": summarize_numeric(ray_shift_norms),
        "visible_surface_depth_order_initial_in_front_sum": int(sum(visible_surface_initial)),
        "visible_surface_depth_order_final_in_front_sum": int(sum(visible_surface_final)),
        "observed_surface_support_uncertainty_m": summarize_numeric(support_uncertainty),
        "active_contact_patch_state_count": int(len(contact_rows)),
        "zero_weight_hand_observation_state_count": int(len(hand_obs_zeroed)),
    }
    return merged, summary


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read frame {frame_idx} from {video_path}")
    return frame


def cell_with_label(img: np.ndarray, label: str, cell_w: int, cell_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(cell_w / max(1, w), (cell_h - 24) / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((cell_h, cell_w, 3), 255, dtype=np.uint8)
    x0 = (cell_w - nw) // 2
    y0 = 22 + (cell_h - 24 - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    cv2.putText(canvas, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def make_review_sheet(case: str, case_dir: Path, frames: list[int]) -> dict[str, Any]:
    views = ["overlay", "world", "side_by_side"]
    video_paths = {view: case_dir / STANDARD_VIDEO_NAMES[view] for view in views}
    cell_w, cell_h = 300, 210
    rows: list[np.ndarray] = []
    failures: list[dict[str, Any]] = []
    for frame_idx in frames:
        cells: list[np.ndarray] = []
        for view in views:
            try:
                img = read_frame(video_paths[view], frame_idx)
                cell = cell_with_label(img, f"{case} f{frame_idx} {view}", cell_w, cell_h)
            except Exception as exc:
                failures.append({"frame_idx": int(frame_idx), "view": view, "error": repr(exc)})
                cell = np.full((cell_h, cell_w, 3), 220, dtype=np.uint8)
                cv2.putText(cell, f"missing f{frame_idx} {view}", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cells.append(cell)
        rows.append(np.concatenate(cells, axis=1))
    sheet = np.concatenate(rows, axis=0) if rows else np.zeros((1, 1, 3), dtype=np.uint8)
    out_path = case_dir / "current_frontier_interval_mano_review.jpg"
    ok = cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"failed to write {out_path}")
    return {"path": str(out_path), "frames": [int(x) for x in frames], "views": views, "failures": failures}


def safe_state_copy_name(index: int, path: Path) -> str:
    parts = [p for p in path.parts[-5:] if p not in {"", "/"}]
    stem = "__".join(parts).replace(os.sep, "__").replace(":", "_")
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    max_stem_len = 150
    if len(stem) > max_stem_len:
        stem = stem[:max_stem_len].rstrip("_")
    return f"{index:02d}__{digest}__{stem}"


def summarize_occluded_translation_posterior(report: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    optimizer_fallback_sides: list[str] = []
    optimizer_failed_sides: list[str] = []
    for side_report in report.get("side_reports") or []:
        if not isinstance(side_report, dict):
            continue
        side = str(side_report.get("hand_side"))
        opt_raw = side_report.get("optimization")
        opt: dict[str, Any] = opt_raw if isinstance(opt_raw, dict) else {}
        if bool(opt.get("used_feasible_start_fallback")):
            optimizer_fallback_sides.append(side)
        if opt.get("success") is False:
            optimizer_failed_sides.append(side)
        rows.extend([r for r in side_report.get("rows") or [] if isinstance(r, dict)])
    zero_rows = [r for r in rows if str(r.get("posterior_state")) != "visible_or_nonzero_observation_row_fixed"]
    missing_delta_rows = [r for r in zero_rows if not isinstance(r.get("selected_depth_order_final_delta_values_m"), list)]
    missing_fingerprint_rows = [r for r in zero_rows if not isinstance(r.get("base_state_fingerprint_sha256"), str)]
    out_of_bound_map_rows = []
    for r in zero_rows:
        s = optional_float(r.get("additional_camera_z_shift_map_m"))
        lo = optional_float(r.get("additional_camera_z_shift_lower_bound_m"))
        hi = optional_float(r.get("additional_camera_z_shift_upper_bound_m"))
        if s is None or lo is None or hi is None or s < lo - 1.0e-9 or s > hi + 1.0e-9:
            out_of_bound_map_rows.append({"frame_idx": r.get("frame_idx"), "hand_side": r.get("hand_side"), "map": s, "lower": lo, "upper": hi})
    if missing_delta_rows:
        raise ValueError(f"occluded translation posterior missing per-vertex deltas on {len(missing_delta_rows)} zero-observation rows")
    if missing_fingerprint_rows:
        raise ValueError(f"occluded translation posterior missing state fingerprints on {len(missing_fingerprint_rows)} zero-observation rows")
    if out_of_bound_map_rows:
        raise ValueError(f"occluded translation posterior has out-of-bound MAP rows: {out_of_bound_map_rows[:5]}")
    return {
        "method": report.get("method"),
        "claim_scope": report.get("claim_scope"),
        "inputs": report.get("inputs"),
        "parameters": report.get("parameters"),
        "summary": report.get("summary") if isinstance(report.get("summary"), dict) else {},
        "row_count": int(len(rows)),
        "zero_observation_row_count": int(len(zero_rows)),
        "optimizer_failed_sides": sorted(set(optimizer_failed_sides)),
        "optimizer_feasible_start_fallback_sides": sorted(set(optimizer_fallback_sides)),
        "map_in_translation_bounds": True,
        "all_zero_rows_preserve_selected_depth_deltas": True,
        "all_zero_rows_preserve_base_state_fingerprints": True,
        "artifact_interpretation": (
            "The rendered magenta skeletons are lower/upper additional camera-z translation interval endpoints for zero-observation rows. "
            "They do not certify the MAP as a hidden-hand reconstruction; optimizer fallback rows are explicit conflict/feasible-set evidence."
        ),
    }


def build_case(case: str, render_root: Path, output_root: Path, review_frames: list[int], *, prefer_hardlink: bool) -> dict[str, Any]:
    if not render_root.exists():
        raise FileNotFoundError(render_root)
    render_manifest_path = render_root / "v18_joint_mano_interval_correction_render_manifest.json"
    render_manifest = load_json(render_manifest_path)
    state_paths = [Path(p) for p in render_manifest.get("state_paths") or []]
    if not state_paths:
        raise ValueError(f"{render_manifest_path} has no state_paths")
    for state_path in state_paths:
        if not state_path.exists():
            raise FileNotFoundError(state_path)

    case_dir = output_root / case
    case_dir.mkdir(parents=True, exist_ok=True)
    linked_videos: dict[str, Any] = {}
    video_probe: dict[str, Any] = {}
    for view, src_name in EXPECTED_VIDEO_NAMES.items():
        src = render_root / src_name
        dst = case_dir / STANDARD_VIDEO_NAMES[view]
        linked_videos[view] = copy_or_hardlink(src, dst, prefer_hardlink=prefer_hardlink)
        video_probe[view] = ffprobe_video(dst)

    linked_manifest = copy_or_hardlink(render_manifest_path, case_dir / "source_render_manifest.json", prefer_hardlink=prefer_hardlink)
    posterior_artifact: dict[str, Any] | None = None
    posterior_report_raw = render_manifest.get("occluded_translation_posterior_report")
    if posterior_report_raw is not None:
        posterior_report_path = Path(str(posterior_report_raw))
        if not posterior_report_path.exists():
            raise FileNotFoundError(posterior_report_path)
        posterior_report = load_json(posterior_report_path)
        posterior_summary = summarize_occluded_translation_posterior(posterior_report)
        posterior_copy = case_dir / "source_occluded_translation_posterior_report.json"
        linked_posterior = copy_or_hardlink(posterior_report_path, posterior_copy, prefer_hardlink=prefer_hardlink)
        posterior_artifact = {
            "source_report": str(posterior_report_path),
            "artifact_report_copy": linked_posterior,
            "summary": posterior_summary,
        }
    state_copy_dir = case_dir / "source_interval_states"
    if state_copy_dir.exists():
        shutil.rmtree(state_copy_dir)
    artifact_state_copies: list[Path] = []
    for i, state_path in enumerate(state_paths):
        artifact_copy = state_copy_dir / safe_state_copy_name(i, state_path)
        copy_or_hardlink(state_path, artifact_copy, prefer_hardlink=prefer_hardlink)
        artifact_state_copies.append(artifact_copy)
    merged_rows, state_summary = summarize_states(case, state_paths, artifact_state_copies)
    frame_count = render_manifest.get("frame_count")
    try:
        frame_count_int = int(frame_count)
    except (TypeError, ValueError):
        frame_count_int = None
    optimized_unique_count = int(state_summary.get("unique_optimized_frame_count", 0) or 0)
    state_summary["full_video_frame_count"] = frame_count_int
    state_summary["context_passthrough_frame_count"] = (
        max(0, frame_count_int - optimized_unique_count) if frame_count_int is not None else None
    )
    state_summary["frame_policy"] = (
        "Frames with interval solver states are rendered from optimized MANO variables; "
        "frames without interval solver states are full-video context/passthrough frames and do not claim a new MANO correction."
    )
    if posterior_artifact is not None:
        state_summary["occluded_translation_posterior"] = posterior_artifact["summary"]
    backing_path = case_dir / "frontier_interval_mano_states.json"
    write_json(backing_path, {
        "method": "build_v18_current_frontier_interval_artifact.merge_interval_mano_states",
        "case": case,
        "source_render_manifest": str(render_manifest_path),
        "frontier_claim_scope": CASE_CLAIMS.get(case, {}),
        "state_summary": state_summary,
        "occluded_translation_posterior_report": posterior_artifact,
        "per_frame_states": merged_rows,
    })
    review = make_review_sheet(case, case_dir, review_frames)
    return {
        "case": case,
        "render_root_source": str(render_root),
        "case_output_root": str(case_dir),
        "frontier_claim_scope": CASE_CLAIMS.get(case, {}),
        "videos": linked_videos,
        "video_probe": video_probe,
        "source_render_manifest": linked_manifest,
        "backing_interval_mano_states": str(backing_path),
        "state_summary": state_summary,
        "occluded_translation_posterior_report": posterior_artifact,
        "review_sheet": review,
        "frame_count_from_source_render_manifest": frame_count,
        "full_video_from_source_render_manifest": render_manifest.get("full_video"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--case-render-root", action="append", default=[], help="CASE=render_root. Defaults to current task5/trash frontier roots.")
    ap.add_argument("--review-frames", action="append", default=[], help="CASE=f0,f1,... for review sheet frames.")
    ap.add_argument("--hardlink-existing-files", action="store_true", help="Use hardlinks for existing videos/manifests. Default copies to freeze the artifact against later in-place source overwrites.")
    ap.add_argument("--skip-uncertainty-classification", action="store_true", help="Do not attach the final physical-cause uncertainty classification. Default builds it so the frontier artifact remains self-explaining.")
    args = ap.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    case_roots = parse_case_render_roots(args.case_render_root)
    review_frames = parse_review_frames(args.review_frames)

    cases: dict[str, Any] = {}
    for case, root in case_roots.items():
        cases[case] = build_case(case, root, output_root, review_frames.get(case, []), prefer_hardlink=bool(args.hardlink_existing_files))

    artifact_manifest = {
        "method": "build_v18_current_frontier_interval_artifact",
        "purpose": "Expose the scoped V18 full-video interval-MANO deliverable and backing optimized MANO states from sanitized non-H-prime annotation inputs; avoid presenting older sparse H-prime roots as the V18 MANO answer.",
        "output_root": str(output_root),
        "claim_scope": {
            "closure_status": "scoped_v18_bounded_mano_closure_under_current_evidence",
            "primary_deliverable": "full-video rendered metric MANO trajectory artifact with explicit support-bounded and occlusion-bounded physical uncertainty",
            "closure_basis": [
                "full-duration overlay/world/side-by-side videos and backing interval MANO states use sanitized non-H-prime inputs",
                "Task5 local-contact/support evidence is consumed as a support limit rather than an overconfident correction",
                "Trash observation-invalid occlusion spans are rendered with broad hard-bound camera-z posterior uncertainty",
                "remaining important uncertainty is classified by physical cause and context-only frames are separated from optimized interval states",
            ],
            "not_claimed": [
                "solved contact",
                "solved object pose",
                "solved nonpenetration",
                "known hidden-hand pose through occlusion",
                "hidden-hand articulation reconstruction",
                "calibrated posterior probability distribution",
            ],
            "ruled_out_as_driving_mano_annotation_source": REJECTED_HPRIME_ROOT,
            "driving_annotation_source": SANITIZED_ANNOTATION_ROOT,
        },
        "cases": cases,
    }
    manifest_path = output_root / "v18_current_frontier_interval_mano_artifact_manifest.json"
    write_json(manifest_path, artifact_manifest)
    if not bool(args.skip_uncertainty_classification):
        classifier = Path(__file__).resolve().parent / "build_v18_frontier_uncertainty_classification.py"
        if not classifier.exists():
            raise FileNotFoundError(classifier)
        subprocess.run([sys.executable, str(classifier), "--frontier-root", str(output_root)], check=True)
        artifact_manifest = load_json(manifest_path)
    print(json.dumps({
        "status": "ok",
        "output_root": str(output_root),
        "cases": {case: {
            "optimized_state_count": data["state_summary"]["optimized_state_count"],
            "unique_optimized_frame_count": data["state_summary"]["unique_optimized_frame_count"],
            "review_sheet": data["review_sheet"]["path"],
        } for case, data in cases.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
