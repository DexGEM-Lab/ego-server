#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def finite(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def refit_metric(hand: dict, key: str) -> float | None:
    refit = hand.get("v3_target_similarity_refit")
    if isinstance(refit, dict):
        value = finite(refit.get(key))
        if value is not None:
            return value
    articulation = hand.get("v3_mano_articulation_mask_depth_refit")
    if not isinstance(articulation, dict):
        return None
    mapped = {
        "median_reprojection_after_px": "silhouette_distance_median_px",
        "mano_minus_unidepth_after_m": "mano_minus_mask_depth_median_m",
        "hand_bone_after_m": "hand_bone_m",
    }
    return finite(articulation.get(mapped.get(key, key)))


def hypothesis_score(hand: dict, args: argparse.Namespace) -> tuple[float, dict]:
    articulation = hand.get("v3_mano_articulation_mask_depth_refit")
    if isinstance(articulation, dict):
        rtmlib = finite(articulation.get("rtmlib_joint_reprojection_median_px"))
        silhouette = finite(articulation.get("silhouette_distance_p95_px"))
        mask_depth = finite(articulation.get("mano_minus_mask_depth_median_m"))
        depth = finite(articulation.get("depth_acceptance_value_m"))
        bone = finite(articulation.get("hand_bone_m"))
        pose = finite(articulation.get("pose_delta_abs_max_rad"))
        scale = finite(articulation.get("scale"))
        if silhouette is None or mask_depth is None or depth is None or bone is None or pose is None or scale is None:
            return float("inf"), {"status": "missing_required_articulation_metric"}
        reproj_term = 0.0 if rtmlib is None else (rtmlib / float(args.sigma_articulation_rtmlib_px)) ** 2
        score = (
            reproj_term
            + (silhouette / float(args.sigma_articulation_silhouette_px)) ** 2
            + (mask_depth / float(args.sigma_depth_m)) ** 2
            + (depth / float(args.sigma_articulation_vertex_depth_m)) ** 2
            + ((bone - float(args.hand_bone_prior_m)) / float(args.sigma_bone_m)) ** 2
            + (pose / float(args.sigma_articulation_pose_rad)) ** 2
            + ((scale - 1.0) / float(args.sigma_articulation_scale)) ** 2
        )
        metrics = {
            "status": "scored_articulation_refit",
            "rtmlib_joint_reprojection_median_px": rtmlib,
            "silhouette_distance_p95_px": silhouette,
            "mano_minus_mask_depth_median_m": mask_depth,
            "depth_acceptance_value_m": depth,
            "hand_bone_m": bone,
            "pose_delta_abs_max_rad": pose,
            "scale": scale,
            "score": float(score),
            "sigma_articulation_rtmlib_px": float(args.sigma_articulation_rtmlib_px),
            "sigma_articulation_silhouette_px": float(args.sigma_articulation_silhouette_px),
            "sigma_depth_m": float(args.sigma_depth_m),
            "sigma_articulation_vertex_depth_m": float(args.sigma_articulation_vertex_depth_m),
            "hand_bone_prior_m": float(args.hand_bone_prior_m),
            "sigma_bone_m": float(args.sigma_bone_m),
            "sigma_articulation_pose_rad": float(args.sigma_articulation_pose_rad),
            "sigma_articulation_scale": float(args.sigma_articulation_scale),
        }
        return float(score), metrics

    reproj = refit_metric(hand, "median_reprojection_after_px")
    depth = refit_metric(hand, "mano_minus_unidepth_after_m")
    bone = refit_metric(hand, "hand_bone_after_m")
    if reproj is None or depth is None or bone is None:
        return float("inf"), {"status": "missing_required_metric"}
    score = (
        (reproj / float(args.sigma_reprojection_px)) ** 2
        + (depth / float(args.sigma_depth_m)) ** 2
        + ((bone - float(args.hand_bone_prior_m)) / float(args.sigma_bone_m)) ** 2
    )
    metrics = {
        "status": "scored",
        "median_reprojection_after_px": reproj,
        "mano_minus_unidepth_after_m": depth,
        "hand_bone_after_m": bone,
        "score": float(score),
        "sigma_reprojection_px": float(args.sigma_reprojection_px),
        "sigma_depth_m": float(args.sigma_depth_m),
        "hand_bone_prior_m": float(args.hand_bone_prior_m),
        "sigma_bone_m": float(args.sigma_bone_m),
    }
    return float(score), metrics


def side_allowed(hand: dict, args: argparse.Namespace) -> bool:
    if args.required_side is None:
        return True
    return str(hand.get("side")) == str(args.required_side)


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frames = annotations.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(f"{args.annotations} has no frames list")
    output_frames = []
    rows = []
    for frame in frames:
        frame_idx = int(frame.get("frame_idx", -1))
        if frame_idx < int(args.frame_start) or frame_idx > int(args.frame_end):
            continue
        out_frame = copy.deepcopy(frame)
        candidates = []
        for hand_i, hand in enumerate(frame.get("hands", [])):
            if not bool(hand.get("measurement_available")):
                continue
            if not side_allowed(hand, args):
                continue
            score, metrics = hypothesis_score(hand, args)
            if not math.isfinite(score):
                continue
            candidates.append((score, hand_i, hand, metrics))
        if not candidates:
            out_frame["hands"] = []
            rows.append({"frame_idx": frame_idx, "status": "no_measured_candidates"})
        else:
            candidates.sort(key=lambda item: (item[0], item[1]))
            score, hand_i, selected, metrics = candidates[0]
            selected = copy.deepcopy(selected)
            selected["v7_hypothesis_selection"] = {
                **metrics,
                "status": "selected_by_reprojection_depth_bone_residual",
                "selected_hand_idx": int(hand_i),
                "candidate_count": int(len(candidates)),
            }
            out_frame["hands"] = [selected]
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "status": "selected",
                    "selected_hand_idx": int(hand_i),
                    "side": selected.get("side"),
                    "score": float(score),
                    "candidate_count": int(len(candidates)),
                    "selected_metrics": metrics,
                }
            )
        output_frames.append(out_frame)
    if not output_frames:
        raise RuntimeError("no frames selected")
    selected_rows = [row for row in rows if row["status"] == "selected"]
    report = {
        "status": "ok" if len(selected_rows) == len(output_frames) else "incomplete",
        "method": "select_hand_hypotheses_by_residual_v7",
        "annotations": str(args.annotations),
        "output_annotations": str(args.output_annotations),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames": int(len(output_frames)),
        "selected_frames": int(len(selected_rows)),
        "rows": rows,
    }
    save_json(args.output_annotations, {"frames": output_frames})
    save_json(args.output_qc, report)
    print(json.dumps({k: report[k] for k in ("status", "frames", "selected_frames")}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--sigma-reprojection-px", type=float, default=6.0)
    parser.add_argument("--sigma-depth-m", type=float, default=0.020)
    parser.add_argument("--hand-bone-prior-m", type=float, default=0.150)
    parser.add_argument("--sigma-bone-m", type=float, default=0.030)
    parser.add_argument("--sigma-articulation-rtmlib-px", type=float, default=18.0)
    parser.add_argument("--sigma-articulation-silhouette-px", type=float, default=20.0)
    parser.add_argument("--sigma-articulation-vertex-depth-m", type=float, default=0.060)
    parser.add_argument("--sigma-articulation-pose-rad", type=float, default=0.75)
    parser.add_argument("--sigma-articulation-scale", type=float, default=0.12)
    parser.add_argument("--required-side", choices=["left", "right"])
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
