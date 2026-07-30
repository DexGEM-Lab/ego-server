#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def require_path(raw: object, key: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(f"{key} must be a non-empty path string")
    path = Path(raw)
    if not path.exists():
        raise RuntimeError(f"{key} does not exist: {path}")
    return path


def run_command(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def summary_number(report: dict, path: tuple[str, ...]) -> float:
    cur: object = report
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise RuntimeError(f"report lacks summary path: {'.'.join(path)}")
        cur = cur[key]
    return float(cur)


def hand_rows(path: Path, frame_start: int, frame_end: int) -> int:
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(f"annotations lacks frames list: {path}")
    total = 0
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        if int(frame_start) <= frame_idx <= int(frame_end):
            hands = frame.get("hands")
            if isinstance(hands, list):
                total += len(hands)
    return total


def write_physics_rejection(
    args: argparse.Namespace,
    replay: dict,
    mesh_archive: Path,
    contact_annotations: Path,
    normalization_report: str | None,
    reason: str,
    metrics: dict,
) -> dict:
    report = {
        "status": "rejected",
        "annotation_ready": False,
        "method": "run_v7_candidate_physics_qc",
        "claim_tested": "an image-replay-accepted object mesh must have MANO hand evidence before physical consistency can be judged",
        "replay_report": str(args.replay_report),
        "mesh_archive": str(mesh_archive),
        "annotations": str(contact_annotations),
        "annotation_normalization": normalization_report,
        "contact_report": None,
        "selected_contact_sdf_report": None,
        "full_window_hand_object_sdf_report": None,
        "frame_start": int(replay["frame_start"]),
        "frame_end": int(replay["frame_end"]),
        "contact_claim_active": False,
        "metrics": metrics,
        "pass": {"hand_evidence_available": False},
        "reason": reason,
    }
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def run(args: argparse.Namespace) -> dict:
    replay = load_json(args.replay_report)
    if replay.get("status") != "accepted":
        raise RuntimeError(f"physics QC requires accepted replay report, got {replay.get('status')}: {args.replay_report}")
    replay_controls = replay.get("replay_controls", {})
    if not bool(replay_controls.get("full_fidelity_zbuffer", False)):
        raise RuntimeError(f"physics QC requires full-fidelity z-buffer replay, got diagnostic replay controls: {args.replay_report}")
    mesh_archive = require_path(replay.get("aligned_mesh_archive"), "replay.aligned_mesh_archive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contact_annotations = args.annotations
    normalization_report = None
    if args.manifest is not None:
        normalized_annotations = args.output_dir / "annotations_with_manifest_object_masks.json"
        run_command(
            [
                sys.executable,
                str(args.scripts_dir / "normalize_v7_annotations_from_manifest_masks.py"),
                "--annotations",
                str(args.annotations),
                "--manifest",
                str(args.manifest),
                "--frame-start",
                str(replay["frame_start"]),
                "--frame-end",
                str(replay["frame_end"]),
                "--output-json",
                str(normalized_annotations),
            ]
        )
        contact_annotations = normalized_annotations
        normalization_report = str(normalized_annotations)
    total_hand_rows = hand_rows(contact_annotations, int(replay["frame_start"]), int(replay["frame_end"]))
    if total_hand_rows == 0:
        return write_physics_rejection(
            args,
            replay,
            mesh_archive,
            contact_annotations,
            normalization_report,
            "physics QC requires MANO hand rows; this target has none in the requested frame window",
            {
                "hand_rows": 0,
                "reliable_temporal_contact_rows": None,
                "geometry_backed_temporal_contact_rows": None,
                "selected_contact_abs_sdf_p95_m": None,
                "selected_contact_near_surface_fraction": None,
                "selected_contact_penetration_fraction": None,
                "full_window_hand_penetration_fraction": None,
            },
        )
    contact_json = args.output_dir / "mesh_surface_contact_qc.json"
    selected_sdf_json = args.output_dir / "selected_contact_sdf_qc.json"
    full_window_sdf_json = args.output_dir / "full_window_hand_object_sdf_qc.json"
    run_command(
        [
            sys.executable,
            str(args.scripts_dir / "diagnose_mesh_surface_contact_v3.py"),
            "--annotations",
            str(contact_annotations),
            "--metric-depth-npz",
            str(args.metric_depth_npz),
            "--object-mesh-npz",
            str(mesh_archive),
            "--frame-start",
            str(replay["frame_start"]),
            "--frame-end",
            str(replay["frame_end"]),
            "--intrinsics-source",
            args.intrinsics_source,
            "--output-json",
            str(contact_json),
            "--min-detector-score",
            str(args.contact_min_detector_score),
            "--max-good-median-reprojection-px",
            str(args.contact_max_good_median_reprojection_px),
            "--max-good-depth-bias-m",
            str(args.contact_max_good_depth_bias_m),
            "--min-good-depth-joints",
            str(args.contact_min_good_depth_joints),
            "--min-stable-depth-fraction",
            str(args.contact_min_stable_depth_fraction),
            "--min-bone-scale-m",
            str(args.contact_min_bone_scale_m),
            "--max-bone-scale-m",
            str(args.contact_max_bone_scale_m),
            "--mask-contact-distance-px",
            str(args.contact_mask_contact_distance_px),
            "--accept-patch-distance-p95-m",
            str(args.contact_accept_patch_distance_p95_m),
            "--accept-patch-signed-gap-m",
            str(args.contact_accept_patch_signed_gap_m),
            "--accept-patch-signed-gap-p95-m",
            str(args.contact_accept_patch_signed_gap_p95_m),
            "--accept-patch-spread-m",
            str(args.contact_accept_patch_spread_m),
            "--accept-patch-local-spread-m",
            str(args.contact_accept_patch_local_spread_m),
            "--accept-anatomical-patch-local-spread-m",
            str(args.contact_accept_anatomical_patch_local_spread_m),
            "--accept-patch-penetration-fraction",
            str(args.contact_accept_patch_penetration_fraction),
            "--min-temporal-patch-frames",
            str(args.contact_min_temporal_patch_frames),
            "--max-temporal-patch-gap-frames",
            str(args.contact_max_temporal_patch_gap_frames),
            "--accept-temporal-patch-local-drift-m",
            str(args.contact_accept_temporal_patch_local_drift_m),
            "--accept-temporal-anchor-relative-drift-m",
            str(args.contact_accept_temporal_anchor_relative_drift_m),
            "--keep-detail",
        ]
    )
    contact = load_json(contact_json)
    reliable_rows = int(contact.get("reliable_temporal_contact_rows", 0))
    geometry_rows = int(contact.get("geometry_backed_temporal_contact_rows", 0))
    measured_rows = int(contact.get("measured_rows", 0))
    if measured_rows == 0:
        return write_physics_rejection(
            args,
            replay,
            mesh_archive,
            contact_annotations,
            normalization_report,
            "physics QC requires measured MANO hand rows; this target has hand geometry but no measured hand evidence in the requested frame window",
            {
                "hand_rows": total_hand_rows,
                "measured_rows": 0,
                "reliable_temporal_contact_rows": reliable_rows,
                "geometry_backed_temporal_contact_rows": geometry_rows,
                "selected_contact_abs_sdf_p95_m": None,
                "selected_contact_near_surface_fraction": None,
                "selected_contact_penetration_fraction": None,
                "full_window_hand_penetration_fraction": None,
            },
        )
    run_command(
        [
            sys.executable,
            str(args.scripts_dir / "diagnose_full_window_hand_object_sdf_v7.py"),
            "--annotations",
            str(contact_annotations),
            "--mesh-archive",
            str(mesh_archive),
            "--frame-start",
            str(replay["frame_start"]),
            "--frame-end",
            str(replay["frame_end"]),
            "--output-json",
            str(full_window_sdf_json),
            "--pitch-m",
            str(args.full_window_sdf_pitch_m),
        ]
    )
    full_window = load_json(full_window_sdf_json)
    full_pen = summary_number(full_window, ("summary", "penetration_fraction"))
    has_contact_evidence = reliable_rows > 0 or geometry_rows > 0
    pass_rows = {
        "full_window_hand_penetration_fraction": bool(full_pen <= float(args.max_full_hand_penetration_fraction)),
    }
    selected_abs_p95 = None
    selected_near = None
    selected_pen = None
    if has_contact_evidence:
        run_command(
            [
                sys.executable,
                str(args.scripts_dir / "diagnose_volume_sdf_contact_v3.py"),
                "--annotations",
                str(contact_annotations),
                "--mesh-archive",
                str(mesh_archive),
                "--contact-report",
                str(contact_json),
                "--frame-start",
                str(replay["frame_start"]),
                "--frame-end",
                str(replay["frame_end"]),
                "--output-json",
                str(selected_sdf_json),
                "--pitch-m",
                str(args.selected_contact_sdf_pitch_m),
                "--local-sdf-crop-margin-m",
                str(args.selected_contact_sdf_crop_margin_m),
                "--local-sdf-min-faces",
                str(args.selected_contact_sdf_min_faces),
            ]
        )
        selected = load_json(selected_sdf_json)
        selected_abs_p95 = summary_number(selected, ("summary", "abs_sdf_m", "p95"))
        selected_near = summary_number(selected, ("summary", "near_surface_fraction"))
        selected_pen = summary_number(selected, ("summary", "penetration_fraction"))
        pass_rows.update(
            {
                "selected_contact_abs_sdf_p95": bool(selected_abs_p95 <= float(args.max_selected_contact_abs_sdf_p95_m)),
                "selected_contact_near_surface_fraction": bool(selected_near >= float(args.min_selected_contact_near_surface_fraction)),
                "selected_contact_penetration_fraction": bool(selected_pen <= float(args.max_selected_contact_penetration_fraction)),
            }
        )
    accepted = all(pass_rows.values())
    report = {
        "status": "accepted" if accepted else "rejected",
        "annotation_ready": bool(accepted),
        "method": "run_v7_candidate_physics_qc",
        "claim_tested": "an image-replay-accepted object mesh satisfies full-window hand/object nonpenetration, and satisfies selected-contact SDF only when contact evidence is present",
        "replay_report": str(args.replay_report),
        "mesh_archive": str(mesh_archive),
        "annotations": str(contact_annotations),
        "annotation_normalization": normalization_report,
        "contact_report": str(contact_json),
        "selected_contact_sdf_report": str(selected_sdf_json) if has_contact_evidence else None,
        "full_window_hand_object_sdf_report": str(full_window_sdf_json),
        "frame_start": int(replay["frame_start"]),
        "frame_end": int(replay["frame_end"]),
        "contact_claim_active": bool(has_contact_evidence),
        "metrics": {
            "reliable_temporal_contact_rows": reliable_rows,
            "geometry_backed_temporal_contact_rows": geometry_rows,
            "selected_contact_abs_sdf_p95_m": selected_abs_p95,
            "selected_contact_near_surface_fraction": selected_near,
            "selected_contact_penetration_fraction": selected_pen,
            "full_window_hand_penetration_fraction": full_pen,
        },
        "thresholds": {
            "max_selected_contact_abs_sdf_p95_m": float(args.max_selected_contact_abs_sdf_p95_m),
            "min_selected_contact_near_surface_fraction": float(args.min_selected_contact_near_surface_fraction),
            "max_selected_contact_penetration_fraction": float(args.max_selected_contact_penetration_fraction),
            "max_full_hand_penetration_fraction": float(args.max_full_hand_penetration_fraction),
            "selected_contact_sdf_pitch_m": float(args.selected_contact_sdf_pitch_m),
            "selected_contact_sdf_crop_margin_m": float(args.selected_contact_sdf_crop_margin_m),
            "selected_contact_sdf_min_faces": int(args.selected_contact_sdf_min_faces),
            "full_window_sdf_pitch_m": float(args.full_window_sdf_pitch_m),
            "contact": {
                "min_detector_score": float(args.contact_min_detector_score),
                "max_good_median_reprojection_px": float(args.contact_max_good_median_reprojection_px),
                "max_good_depth_bias_m": float(args.contact_max_good_depth_bias_m),
                "min_good_depth_joints": int(args.contact_min_good_depth_joints),
                "min_stable_depth_fraction": float(args.contact_min_stable_depth_fraction),
                "min_bone_scale_m": float(args.contact_min_bone_scale_m),
                "max_bone_scale_m": float(args.contact_max_bone_scale_m),
                "mask_contact_distance_px": float(args.contact_mask_contact_distance_px),
                "accept_patch_distance_p95_m": float(args.contact_accept_patch_distance_p95_m),
                "accept_patch_signed_gap_m": float(args.contact_accept_patch_signed_gap_m),
                "accept_patch_signed_gap_p95_m": float(args.contact_accept_patch_signed_gap_p95_m),
                "accept_patch_spread_m": float(args.contact_accept_patch_spread_m),
                "accept_patch_local_spread_m": float(args.contact_accept_patch_local_spread_m),
                "accept_anatomical_patch_local_spread_m": float(args.contact_accept_anatomical_patch_local_spread_m),
                "accept_patch_penetration_fraction": float(args.contact_accept_patch_penetration_fraction),
                "min_temporal_patch_frames": int(args.contact_min_temporal_patch_frames),
                "max_temporal_patch_gap_frames": int(args.contact_max_temporal_patch_gap_frames),
                "accept_temporal_patch_local_drift_m": float(args.contact_accept_temporal_patch_local_drift_m),
                "accept_temporal_anchor_relative_drift_m": float(args.contact_accept_temporal_anchor_relative_drift_m),
            },
        },
        "pass": pass_rows,
    }
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--intrinsics-source", choices=("annotation-vggt", "hand", "cli"), default="annotation-vggt")
    parser.add_argument("--sdf-pitch-m", type=float, default=None)
    parser.add_argument("--selected-contact-sdf-pitch-m", type=float, default=0.001)
    parser.add_argument("--selected-contact-sdf-crop-margin-m", type=float, default=0.040)
    parser.add_argument("--selected-contact-sdf-min-faces", type=int, default=256)
    parser.add_argument("--full-window-sdf-pitch-m", type=float, default=0.003)
    parser.add_argument("--max-selected-contact-abs-sdf-p95-m", type=float, default=0.006)
    parser.add_argument("--min-selected-contact-near-surface-fraction", type=float, default=0.75)
    parser.add_argument("--max-selected-contact-penetration-fraction", type=float, default=0.10)
    parser.add_argument("--max-full-hand-penetration-fraction", type=float, default=0.02)
    parser.add_argument("--contact-min-detector-score", type=float, default=0.15)
    parser.add_argument("--contact-max-good-median-reprojection-px", type=float, default=18.0)
    parser.add_argument("--contact-max-good-depth-bias-m", type=float, default=0.020)
    parser.add_argument("--contact-min-good-depth-joints", type=int, default=12)
    parser.add_argument("--contact-min-stable-depth-fraction", type=float, default=0.75)
    parser.add_argument("--contact-min-bone-scale-m", type=float, default=0.120)
    parser.add_argument("--contact-max-bone-scale-m", type=float, default=0.240)
    parser.add_argument("--contact-mask-contact-distance-px", type=float, default=8.0)
    parser.add_argument("--contact-accept-patch-distance-p95-m", type=float, default=0.025)
    parser.add_argument("--contact-accept-patch-signed-gap-m", type=float, default=0.015)
    parser.add_argument("--contact-accept-patch-signed-gap-p95-m", type=float, default=0.025)
    parser.add_argument("--contact-accept-patch-spread-m", type=float, default=0.040)
    parser.add_argument("--contact-accept-patch-local-spread-m", type=float, default=0.040)
    parser.add_argument("--contact-accept-anatomical-patch-local-spread-m", type=float, default=0.025)
    parser.add_argument("--contact-accept-patch-penetration-fraction", type=float, default=0.25)
    parser.add_argument("--contact-min-temporal-patch-frames", type=int, default=2)
    parser.add_argument("--contact-max-temporal-patch-gap-frames", type=int, default=6)
    parser.add_argument("--contact-accept-temporal-patch-local-drift-m", type=float, default=0.030)
    parser.add_argument("--contact-accept-temporal-anchor-relative-drift-m", type=float, default=0.025)
    args = parser.parse_args()
    if args.sdf_pitch_m is not None:
        args.selected_contact_sdf_pitch_m = float(args.sdf_pitch_m)
        args.full_window_sdf_pitch_m = float(args.sdf_pitch_m)
    return args


if __name__ == "__main__":
    run(parse_args())
