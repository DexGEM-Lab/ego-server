#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Recover late trash-lid part evidence without the stale object-mask gate.

The accepted V18 OWLv2/SAM2 lid track has valid late prompt boxes (864/956),
but its materializer only saves frames whose SAM2 mask overlaps the old object
mask. That old object mask/compact-rigid association is precisely the current
trash blocker. This script reruns the same SAM2 prompt evidence, saves the raw
model masks independently of object-mask containment, lifts selected late masks
through metric depth, and measures current/candidate MANO surfaces against the
observed mask-depth surface.

This is not a coordinate correction. It is a discriminating mechanism test:
late model-produced part evidence either gives a physically plausible observed
surface that can constrain/falsify MANO, or it visibly/model-geometrically fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v18_owlv2_sam2_part_tracks import (  # noqa: E402
    bbox_from_mask,
    box_iou,
    center_from_mask,
    import_sam2,
    load_mask,
    mask_fraction_in_box,
    raw_video_dir,
)
from build_v18_part_visible_surfaces import (  # noqa: E402
    build_faces,
    load_metric_depth,
    remove_unreferenced,
    resize_bool_mask,
)
from build_v18_temporal_mano_articulated_interval_state import (  # noqa: E402
    bridge_vertices_and_joints,
    load_wilor_mano_class,
    patch_legacy_mano_loader,
)
from build_v18_observed_surface_mano_constraint_state import make_candidate_vertices  # noqa: E402
from build_v18_temporal_mano_translation_interval_state import (  # noqa: E402
    as_list,
    frame_camera_pose,
    load_json,
    numeric_summary,
    write_json,
)

DEFAULT_PROMPT_REPORT = Path(
    "/data2/ego_annotation_outputs/v18_owlv2_sam2_part_tracks/trash_1050/accepted_tracks/"
    "owlv2_sam2_pink_lid_trash_can_second_lid/v18_owlv2_sam2_part_track_report.json"
)
DEFAULT_ANNOTATIONS = Path(
    "/data2/ego_annotation_outputs/v18_full_pipeline_verified_hprime_final_v7_full_signed_temporal_guard/"
    "trash_1050/annotations_v18_full.json"
)
DEFAULT_TEMPORAL = Path(
    "/data2/ego_annotation_outputs/v18_trash_lid_temporal_mano_articulated_leftreplay_v1/"
    "trash_1050/v18_temporal_mano_articulated_interval_state.json"
)
DEFAULT_DEPTH = Path("/data2/ego_annotation_outputs/v16_full_pipeline/trash_1050/unidepth_metric/unidepth_metric_depth_v3.npz")
DEFAULT_OUTPUT = Path("/data2/ego_annotation_outputs/v18_trash_late_lid_part_evidence_v1")
DEFAULT_LEFT = Path("/data/dex_home/yiwen/mano_assets/mano/models/MANO_LEFT.pkl")
DEFAULT_RIGHT = Path("third_party/WiLoR/mano_data/MANO_RIGHT.pkl")
TARGET_FRAMES = [864, 901, 956, 958, 1003, 1049]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-report", type=Path, default=DEFAULT_PROMPT_REPORT)
    ap.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    ap.add_argument("--temporal-mano-state", type=Path, default=DEFAULT_TEMPORAL)
    ap.add_argument("--depth-npz", type=Path, default=DEFAULT_DEPTH)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--sam2-repo", type=Path, default=Path("third_party/sam2"))
    ap.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    ap.add_argument("--sam2-checkpoint", type=Path, default=Path("/data2/ego_annotation_outputs/checkpoints/sam2.1_hiera_small.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    ap.add_argument("--wilor-mano-right", type=Path, default=DEFAULT_RIGHT)
    ap.add_argument("--wilor-mano-left", type=Path, default=DEFAULT_LEFT)
    ap.add_argument("--target-frames", type=int, nargs="+", default=TARGET_FRAMES)
    ap.add_argument("--reuse-mask-dir", type=Path, default=None, help="Reuse previously recovered raw SAM2 masks instead of rerunning SAM2.")
    ap.add_argument("--save-frame-start", type=int, default=820)
    ap.add_argument("--save-frame-end", type=int, default=1049)
    ap.add_argument("--min-mask-area-px", type=int, default=50)
    ap.add_argument("--mask-stride", type=int, default=6)
    ap.add_argument("--min-depth-pixels", type=int, default=50)
    ap.add_argument("--min-vertices", type=int, default=8)
    ap.add_argument("--min-faces", type=int, default=6)
    ap.add_argument("--target-surface-vertices", type=int, default=200)
    ap.add_argument("--target-surface-faces", type=int, default=180)
    ap.add_argument("--depth-low-quantile", type=float, default=0.02)
    ap.add_argument("--depth-high-quantile", type=float, default=0.98)
    ap.add_argument("--min-depth-m", type=float, default=0.05)
    ap.add_argument("--max-depth-m", type=float, default=5.0)
    ap.add_argument("--max-triangle-edge-m", type=float, default=0.06)
    ap.add_argument("--depth-order-margin-m", type=float, default=0.01)
    ap.add_argument("--nearest-surface-threshold-m", type=float, default=0.02)
    ap.add_argument("--max-review-hand-points", type=int, default=180)
    ap.add_argument("--offload-video-to-cpu", action="store_true", default=True)
    ap.add_argument("--offload-state-to-cpu", action="store_true")
    ap.add_argument("--vos-optimized", action="store_true")
    return ap.parse_args()


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def load_gray_mask(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L")) > 0
    if shape_hw is not None:
        arr = resize_bool_mask(arr, shape_hw)
    return arr


def temporal_state_map(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    payload = load_json(path)
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in payload.get("per_frame_states", []) if isinstance(payload, dict) else []:
        if isinstance(row, dict) and row.get("frame_idx") is not None and row.get("hand_side") is not None:
            out[(int(row["frame_idx"]), str(row["hand_side"]))] = row
    return out


def frame_by_idx(annotations: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out = {}
    for frame in as_list(annotations.get("frames")):
        if isinstance(frame, dict) and frame.get("frame_idx") is not None:
            out[int(frame["frame_idx"])] = frame
    return out


def object_mask_index(annotations: dict[str, Any], object_id: str) -> dict[int, str]:
    out = {}
    for frame in as_list(annotations.get("frames")):
        if not isinstance(frame, dict) or frame.get("frame_idx") is None:
            continue
        frame_idx = int(frame["frame_idx"])
        for obj in as_list(frame.get("objects")):
            if not isinstance(obj, dict) or obj.get("object_id") != object_id:
                continue
            if obj.get("renderable_mask") is True and isinstance(obj.get("mask_path"), str) and Path(str(obj["mask_path"])).exists():
                out[frame_idx] = str(obj["mask_path"])
    return out


def load_reused_masks(mask_dir: Path) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for path in sorted(mask_dir.glob("*.png")):
        try:
            frame_idx = int(path.stem)
        except ValueError:
            continue
        masks[frame_idx] = np.asarray(Image.open(path).convert("L")) > 0
    if not masks:
        raise RuntimeError(f"reuse mask dir contains no frame PNG masks: {mask_dir}")
    return masks


def run_sam2_masks(args: argparse.Namespace, annotations: dict[str, Any], prompt_report: dict[str, Any]) -> dict[int, np.ndarray]:
    predictor = import_sam2(args)
    video_dir = raw_video_dir(annotations)
    state = predictor.init_state(
        str(video_dir),
        offload_video_to_cpu=bool(args.offload_video_to_cpu),
        offload_state_to_cpu=bool(args.offload_state_to_cpu),
    )
    for prompt in prompt_report.get("prompt_detections", []) if isinstance(prompt_report.get("prompt_detections"), list) else []:
        box = np.asarray(prompt["bbox_xyxy"], dtype=np.float32)
        predictor.add_new_points_or_box(state, frame_idx=int(prompt["frame_idx"]), obj_id=1, box=box)
    masks: dict[int, np.ndarray] = {}
    with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
        for reverse in (False, True):
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state, start_frame_idx=None, reverse=reverse):
                if mask_logits is None:
                    continue
                for i, obj_id in enumerate(obj_ids):
                    if int(obj_id) != 1:
                        continue
                    mask = (mask_logits[i].detach().cpu().numpy() > 0.0)
                    if mask.ndim == 3:
                        mask = mask[0]
                    masks[int(frame_idx)] = mask.astype(bool)
    return masks


def surface_from_mask(
    *,
    frame_idx: int,
    mask: np.ndarray,
    frame: dict[str, Any],
    depth: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    depth_i = depth["frame_to_i"].get(frame_idx)
    if depth_i is None:
        raise RuntimeError("metric_depth_missing_for_frame")
    depth_m = np.asarray(depth["depth"][int(depth_i)], dtype=np.float64)
    mask_d = resize_bool_mask(mask.astype(bool), depth_m.shape)
    valid = mask_d & np.isfinite(depth_m) & (depth_m >= float(args.min_depth_m)) & (depth_m <= float(args.max_depth_m))
    values = depth_m[valid]
    if values.size < int(args.min_depth_pixels):
        raise RuntimeError("too_few_valid_masked_depth_pixels")
    lo = float(np.quantile(values, float(args.depth_low_quantile)))
    hi = float(np.quantile(values, float(args.depth_high_quantile)))
    keep = valid & (depth_m >= lo) & (depth_m <= hi)
    ys_keep, xs_keep = np.nonzero(keep)
    if xs_keep.size < int(args.min_depth_pixels):
        raise RuntimeError("too_few_depth_pixels_after_quantile_filter")
    y0, y1 = int(ys_keep.min()), int(ys_keep.max())
    x0, x1 = int(xs_keep.min()), int(xs_keep.max())
    fx, fy, cx, cy = np.asarray(depth["intrinsics"][int(depth_i)], dtype=np.float64).tolist()
    stride_candidates = []
    stride = max(1, int(args.mask_stride))
    while stride > 1:
        stride_candidates.append(stride)
        stride = max(1, stride // 2)
    stride_candidates.append(1)
    best_vertices_c = np.zeros((0, 3), dtype=np.float64)
    best_faces = np.zeros((0, 3), dtype=np.int32)
    sampled_vertex_count = 0
    stride_used = stride_candidates[-1]
    last_rejection = "too_few_sampled_vertices"
    for stride in stride_candidates:
        ys = np.arange(y0, y1 + 1, stride, dtype=np.int32)
        xs = np.arange(x0, x1 + 1, stride, dtype=np.int32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        sampled = keep[np.ix_(ys, xs)]
        flat_x = grid_x[sampled].astype(np.float64)
        flat_y = grid_y[sampled].astype(np.float64)
        flat_z = depth_m[np.ix_(ys, xs)][sampled].astype(np.float64)
        sampled_vertex_count = int(len(flat_z))
        stride_used = int(stride)
        if len(flat_z) < int(args.min_vertices):
            last_rejection = "too_few_sampled_vertices"
            continue
        candidate_vertices_c = np.column_stack(((flat_x - cx) * flat_z / fx, (flat_y - cy) * flat_z / fy, flat_z))
        index_grid = np.full(sampled.shape, -1, dtype=np.int32)
        index_grid[sampled] = np.arange(len(candidate_vertices_c), dtype=np.int32)
        candidate_faces = build_faces(index_grid, candidate_vertices_c, float(args.max_triangle_edge_m))
        candidate_vertices_c, candidate_faces = remove_unreferenced(candidate_vertices_c, candidate_faces)
        if len(candidate_vertices_c) > len(best_vertices_c) or len(candidate_faces) > len(best_faces):
            best_vertices_c = candidate_vertices_c
            best_faces = candidate_faces
        if len(candidate_vertices_c) >= int(args.target_surface_vertices) and len(candidate_faces) >= int(args.target_surface_faces):
            best_vertices_c = candidate_vertices_c
            best_faces = candidate_faces
            break
        last_rejection = "surface_sampling_target_not_met_before_stride_exhausted"
    if len(best_vertices_c) < int(args.min_vertices) or len(best_faces) < int(args.min_faces):
        raise RuntimeError(last_rejection)
    r_c2w, t_c2w = frame_camera_pose(frame)
    vertices_w = best_vertices_c @ r_c2w.T + t_c2w[None, :]
    row = {
        "frame_idx": int(frame_idx),
        "coordinate_frame": "world_from_metric_depth_camera_T_world_camera_metric",
        "mask_area_px_raw": int(mask.sum()),
        "mask_area_px_depth_grid": int(mask_d.sum()),
        "valid_depth_pixels": int(values.size),
        "depth_low_m": lo,
        "depth_high_m": hi,
        "depth_median_m": float(np.median(values)),
        "vertices": int(len(vertices_w)),
        "faces": int(len(best_faces)),
        "mask_stride_used": int(stride_used),
        "sampled_vertex_count_before_connectivity": int(sampled_vertex_count),
        "bbox_world_min_m": vertices_w.min(axis=0).astype(float).tolist(),
        "bbox_world_max_m": vertices_w.max(axis=0).astype(float).tolist(),
        "extent_world_m": (vertices_w.max(axis=0) - vertices_w.min(axis=0)).astype(float).tolist(),
        "geometry_claim": "model_produced_late_lid_visible_surface_only_not_hidden_geometry",
    }
    return vertices_w.astype(np.float32), best_faces.astype(np.int32), row


def save_surface_archive(path: Path, observations: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_offsets = [0]
    face_offsets = [0]
    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    for obs in observations:
        vertices = obs.pop("_vertices_world")
        faces = obs.pop("_faces")
        offset = vertex_offsets[-1]
        vertices_parts.append(vertices.astype(np.float32))
        faces_parts.append((faces + offset).astype(np.int32))
        vertex_offsets.append(offset + len(vertices))
        face_offsets.append(face_offsets[-1] + len(faces))
    vertices_all = np.vstack(vertices_parts).astype(np.float32) if vertices_parts else np.zeros((0, 3), dtype=np.float32)
    faces_all = np.vstack(faces_parts).astype(np.int32) if faces_parts else np.zeros((0, 3), dtype=np.int32)
    np.savez_compressed(
        path,
        frame_idx=np.asarray([obs["frame_idx"] for obs in observations], dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int64),
        face_offsets=np.asarray(face_offsets, dtype=np.int64),
        vertices_world=vertices_all,
        faces=faces_all,
        metadata_json=json.dumps({"claim": "late SAM2 lid visible surface; no hidden completion or accepted object pose"}),
    )


def project_world(points_world: np.ndarray, frame: dict[str, Any], intrinsics: list[float]) -> tuple[np.ndarray, np.ndarray]:
    r_c2w, t_c2w = frame_camera_pose(frame)
    pts_c = (points_world - t_c2w[None, :]) @ r_c2w
    z = pts_c[:, 2]
    fx, fy, cx, cy = [float(v) for v in intrinsics]
    uv = np.full((len(points_world), 2), np.nan, dtype=np.float64)
    valid = z > 1.0e-6
    uv[valid, 0] = fx * pts_c[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * pts_c[valid, 1] / z[valid] + cy
    return uv, z


def hand_intrinsics(frame: dict[str, Any], side: str, depth: dict[str, Any], frame_idx: int) -> list[float]:
    for hand in as_list(frame.get("hands")):
        if isinstance(hand, dict) and hand.get("hand_side") == side:
            metric = hand.get("metric_mano_state") if isinstance(hand.get("metric_mano_state"), dict) else {}
            intr = hand.get("current_v18_camera_intrinsics_fx_fy_cx_cy") or metric.get("current_v18_camera_intrinsics_fx_fy_cx_cy")
            if isinstance(intr, list) and len(intr) == 4:
                return [float(v) for v in intr]
    depth_i = depth["frame_to_i"][frame_idx]
    return [float(v) for v in np.asarray(depth["intrinsics"][int(depth_i)], dtype=np.float64).tolist()]


def measure_hand_to_surface(
    *,
    frame_idx: int,
    frame: dict[str, Any],
    side: str,
    label: str,
    vertices_world: np.ndarray,
    surface_vertices_world: np.ndarray,
    mask: np.ndarray,
    depth: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    tree = cKDTree(surface_vertices_world.astype(np.float64))
    distances, _ = tree.query(vertices_world.astype(np.float64), k=1, workers=-1)
    depth_i = depth["frame_to_i"].get(frame_idx)
    if depth_i is None:
        raise RuntimeError("metric_depth_missing_for_frame")
    depth_m = np.asarray(depth["depth"][int(depth_i)], dtype=np.float64)
    mask_d = resize_bool_mask(mask.astype(bool), depth_m.shape)
    intr = hand_intrinsics(frame, side, depth, frame_idx)
    uv, z = project_world(vertices_world.astype(np.float64), frame, intr)
    xs = np.rint(uv[:, 0]).astype(np.int64)
    ys = np.rint(uv[:, 1]).astype(np.int64)
    in_bounds = (xs >= 0) & (xs < depth_m.shape[1]) & (ys >= 0) & (ys < depth_m.shape[0]) & np.isfinite(z)
    inside = np.zeros(len(vertices_world), dtype=bool)
    inside[in_bounds] = mask_d[ys[in_bounds], xs[in_bounds]]
    sampled_depth = np.full(len(vertices_world), np.nan, dtype=np.float64)
    sampled_depth[in_bounds] = depth_m[ys[in_bounds], xs[in_bounds]]
    finite_inside = inside & np.isfinite(sampled_depth)
    depth_delta = z - sampled_depth
    behind = finite_inside & (depth_delta > float(args.depth_order_margin_m))
    in_front = finite_inside & (depth_delta < -float(args.depth_order_margin_m))
    near_depth = finite_inside & (np.abs(depth_delta) <= float(args.depth_order_margin_m))
    near_surface = distances <= float(args.nearest_surface_threshold_m)
    return {
        "frame_idx": int(frame_idx),
        "hand_side": side,
        "mano_surface": label,
        "hand_vertex_count": int(len(vertices_world)),
        "nearest_visible_lid_surface_distance_m": numeric_summary(np.asarray(distances, dtype=float)),
        "vertices_within_nearest_surface_threshold": int(np.count_nonzero(near_surface)),
        "projected_inside_late_lid_mask_count": int(np.count_nonzero(inside)),
        "projected_inside_late_lid_mask_fraction": float(np.count_nonzero(inside) / max(1, len(vertices_world))),
        "depth_order_inside_mask": {
            "finite_inside_count": int(np.count_nonzero(finite_inside)),
            "hand_behind_observed_lid_count": int(np.count_nonzero(behind)),
            "hand_in_front_of_observed_lid_count": int(np.count_nonzero(in_front)),
            "hand_near_observed_lid_depth_count": int(np.count_nonzero(near_depth)),
            "depth_delta_hand_minus_lid_m": numeric_summary(depth_delta[finite_inside]),
        },
        "constraint_scope": "observed_visible_lid_mask_depth_surface_only_no_hidden_volume_no_signed_interior",
    }


def load_mano_models(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    patch_legacy_mano_loader()
    cls = load_wilor_mano_class(args.wilor_root)
    out: dict[str, Any] = {}
    if args.wilor_mano_right.exists():
        out["right"] = cls(model_path=str(args.wilor_mano_right), is_rhand=True, use_pca=False, flat_hand_mean=False, batch_size=1).to(device)
    if args.wilor_mano_left.exists():
        left = cls(model_path=str(args.wilor_mano_left), is_rhand=False, use_pca=False, flat_hand_mean=False, batch_size=1).to(device)
        with torch.no_grad():
            left.shapedirs[:, 0, :] *= -1
        out["left"] = left
    for model in out.values():
        model.eval()
    return out


def draw_review_sheet(
    *,
    output_path: Path,
    frames: list[int],
    annotations_by_frame: dict[int, dict[str, Any]],
    mask_paths: dict[int, Path],
    prompt_report: dict[str, Any],
    hand_measurements: list[dict[str, Any]],
    projected_hand_points: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    prompt_by_frame = {int(row["frame_idx"]): row for row in prompt_report.get("prompt_detections", []) if isinstance(row, dict)}
    tiles: list[Image.Image] = []
    font = ImageFont.load_default()
    measurement_index = {(int(m["frame_idx"]), str(m["hand_side"]), str(m["mano_surface"])): m for m in hand_measurements}
    points_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in projected_hand_points:
        points_by_frame.setdefault(int(row["frame_idx"]), []).append(row)
    for frame_idx in frames:
        frame = annotations_by_frame.get(frame_idx)
        if frame is None:
            continue
        image = Image.open(str(frame["raw_frame_path"])).convert("RGB")
        overlay = image.convert("RGBA")
        draw = ImageDraw.Draw(overlay)
        mask_path = mask_paths.get(frame_idx)
        if mask_path is not None and mask_path.exists():
            mask = Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.NEAREST)
            color = Image.new("RGBA", image.size, (255, 0, 255, 88))
            overlay.alpha_composite(Image.composite(color, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask))
            # Draw a contour-like bounding rectangle from mask bbox.
            mask_np = np.asarray(mask) > 0
            bbox = bbox_from_mask(mask_np)
            if bbox is not None:
                draw.rectangle([int(v) for v in bbox], outline=(255, 0, 255, 255), width=3)
        prompt = prompt_by_frame.get(frame_idx)
        if prompt is not None:
            draw.rectangle([int(round(float(v))) for v in prompt["bbox_xyxy"]], outline=(0, 255, 255, 255), width=3)
        for pts_row in points_by_frame.get(frame_idx, []):
            side = str(pts_row.get("hand_side"))
            surface = str(pts_row.get("mano_surface"))
            if side == "left" and surface.startswith("current"):
                color = (0, 160, 255, 230)
            elif side == "right" and surface.startswith("current"):
                color = (0, 255, 80, 230)
            elif side == "left":
                color = (0, 255, 255, 230)
            else:
                color = (255, 230, 0, 230)
            for x, y in pts_row.get("points_xy", []):
                x_i = int(round(float(x)))
                y_i = int(round(float(y)))
                if 0 <= x_i < image.width and 0 <= y_i < image.height:
                    draw.ellipse((x_i - 1, y_i - 1, x_i + 1, y_i + 1), fill=color)
        lines = [f"frame {frame_idx}: mask magenta; current L/R blue/green; cand L/R cyan/yellow"]
        for side in ("left", "right"):
            cur = measurement_index.get((frame_idx, side, "current_bridge_mano"))
            cand = measurement_index.get((frame_idx, side, "articulated_candidate_mano"))
            for name, m in (("cur", cur), ("cand", cand)):
                if not m:
                    continue
                inside = m.get("projected_inside_late_lid_mask_count")
                behind = m.get("depth_order_inside_mask", {}).get("hand_behind_observed_lid_count")
                near = m.get("vertices_within_nearest_surface_threshold")
                lines.append(f"{side} {name}: inside={inside} behind={behind} nearSurf={near}")
        y = 4
        for line in lines[:7]:
            draw.rectangle((4, y - 1, min(image.width - 4, 4 + len(line) * 6), y + 11), fill=(0, 0, 0, 160))
            draw.text((6, y), line, fill=(255, 255, 255, 255), font=font)
            y += 13
        tiles.append(overlay.convert("RGB"))
    if not tiles:
        return
    w = max(tile.width for tile in tiles)
    h = max(tile.height for tile in tiles)
    cols = min(3, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * w, (i // cols) * h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    prompt_report = load_json(args.prompt_report)
    annotations = load_json(args.annotations)
    frames = frame_by_idx(annotations)
    object_id = str(prompt_report.get("object_id"))
    object_masks = object_mask_index(annotations, object_id)
    depth = load_metric_depth(args.depth_npz)
    target_frames = sorted({int(f) for f in args.target_frames})
    masks = load_reused_masks(args.reuse_mask_dir) if args.reuse_mask_dir is not None else run_sam2_masks(args, annotations, prompt_report)

    out_case = args.output_root / "trash_1050"
    raw_mask_dir = out_case / "raw_sam2_late_lid_masks"
    selected_mask_paths: dict[int, Path] = {}
    mask_rows: list[dict[str, Any]] = []
    prompt_by_frame = {int(row["frame_idx"]): row for row in prompt_report.get("prompt_detections", []) if isinstance(row, dict)}
    for frame_idx in sorted(masks):
        mask = masks[frame_idx]
        area = int(mask.sum())
        save_frame = frame_idx in target_frames or (int(args.save_frame_start) <= frame_idx <= int(args.save_frame_end) and area >= int(args.min_mask_area_px))
        mask_path = raw_mask_dir / f"{frame_idx:06d}.png"
        if save_frame:
            save_mask(mask_path, mask)
            selected_mask_paths[frame_idx] = mask_path
        object_mask_path = object_masks.get(frame_idx)
        containment = None
        coverage = None
        if object_mask_path is not None and Path(object_mask_path).exists():
            object_mask = load_mask(Path(object_mask_path), mask.shape)
            inter = int(np.logical_and(mask, object_mask).sum())
            object_area = int(object_mask.sum())
            containment = inter / float(area) if area > 0 else 0.0
            coverage = inter / float(object_area) if object_area > 0 else 0.0
        prompt = prompt_by_frame.get(frame_idx)
        bbox = bbox_from_mask(mask)
        mask_rows.append(
            {
                "frame_idx": int(frame_idx),
                "saved_mask_path": str(mask_path) if save_frame else None,
                "area_px": area,
                "bbox_xyxy": bbox,
                "center_xy": center_from_mask(mask),
                "old_object_mask_path": object_mask_path,
                "containment_in_old_object_mask": containment,
                "old_object_coverage_by_late_mask": coverage,
                "is_prompt_frame": prompt is not None,
                "prompt_box_iou_with_mask_bbox": box_iou(bbox, prompt["bbox_xyxy"]) if prompt is not None else None,
                "prompt_mask_fraction_inside_box": mask_fraction_in_box(mask, prompt["bbox_xyxy"]) if prompt is not None else None,
                "accepted_by_old_object_mask_gate": bool(containment is not None and coverage is not None and area >= 50 and containment >= 0.5 and coverage >= 0.005 and coverage <= 0.90),
            }
        )

    surface_rows: list[dict[str, Any]] = []
    surface_rows_for_archive: list[dict[str, Any]] = []
    surface_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rejected_surfaces: list[dict[str, Any]] = []
    for frame_idx in target_frames:
        mask = masks.get(frame_idx)
        frame = frames.get(frame_idx)
        if mask is None or frame is None:
            rejected_surfaces.append({"frame_idx": int(frame_idx), "reason": "missing_mask_or_frame"})
            continue
        try:
            vertices_w, faces, row = surface_from_mask(frame_idx=frame_idx, mask=mask, frame=frame, depth=depth, args=args)
        except RuntimeError as exc:
            rejected_surfaces.append({"frame_idx": int(frame_idx), "reason": str(exc)})
            continue
        row["mask_path"] = str(selected_mask_paths.get(frame_idx, raw_mask_dir / f"{frame_idx:06d}.png"))
        surface_by_frame[frame_idx] = (vertices_w.astype(np.float64), faces)
        archive_row = dict(row)
        archive_row["_vertices_world"] = vertices_w
        archive_row["_faces"] = faces
        surface_rows_for_archive.append(archive_row)
        surface_rows.append(row)
    archive_rows_copy = [dict(row) for row in surface_rows_for_archive]
    archive_path = out_case / "late_lid_visible_surface_world.npz"
    save_surface_archive(archive_path, archive_rows_copy)

    device = torch.device("cpu")
    models_by_side = load_mano_models(args, device)
    temporal = temporal_state_map(args.temporal_mano_state)
    bridge_cache: dict[Path, Any] = {}
    source_cache: dict[Path, Any] = {}
    measurements: list[dict[str, Any]] = []
    projected_hand_points: list[dict[str, Any]] = []
    for frame_idx in target_frames:
        frame = frames.get(frame_idx)
        mask = masks.get(frame_idx)
        surface = surface_by_frame.get(frame_idx)
        if frame is None or mask is None or surface is None:
            continue
        surface_vertices_w, _faces = surface
        for hand in as_list(frame.get("hands")):
            if not isinstance(hand, dict):
                continue
            side = str(hand.get("hand_side"))
            bridge = bridge_vertices_and_joints(hand, bridge_cache)
            if bridge is None:
                continue
            current_vertices, _current_joints = bridge
            current_vertices_f = np.asarray(current_vertices, dtype=np.float64)
            measurements.append(
                measure_hand_to_surface(
                    frame_idx=frame_idx,
                    frame=frame,
                    side=side,
                    label="current_bridge_mano",
                    vertices_world=current_vertices_f,
                    surface_vertices_world=surface_vertices_w,
                    mask=mask,
                    depth=depth,
                    args=args,
                )
            )
            intr = hand_intrinsics(frame, side, depth, frame_idx)
            uv_current, _z_current = project_world(current_vertices_f, frame, intr)
            valid_current = uv_current[np.isfinite(uv_current).all(axis=1)]
            if len(valid_current):
                step = max(1, int(np.ceil(len(valid_current) / max(1, int(args.max_review_hand_points)))))
                projected_hand_points.append({"frame_idx": int(frame_idx), "hand_side": side, "mano_surface": "current_bridge_mano", "points_xy": valid_current[::step].astype(float).tolist()})
            temporal_row = temporal.get((frame_idx, side))
            if temporal_row is None:
                continue
            candidate_vertices, candidate_info = make_candidate_vertices(
                models_by_side=models_by_side,
                temporal_row=temporal_row,
                hand=hand,
                current_vertices=np.asarray(current_vertices, dtype=np.float64),
                bridge_cache=bridge_cache,
                source_cache=source_cache,
                device=device,
            )
            if candidate_vertices is None:
                measurements.append(
                    {
                        "frame_idx": int(frame_idx),
                        "hand_side": side,
                        "mano_surface": "articulated_candidate_mano",
                        "candidate_reconstruction": candidate_info,
                        "constraint_scope": "candidate_unavailable",
                    }
                )
                continue
            candidate_vertices_f = np.asarray(candidate_vertices, dtype=np.float64)
            m = measure_hand_to_surface(
                frame_idx=frame_idx,
                frame=frame,
                side=side,
                label="articulated_candidate_mano",
                vertices_world=candidate_vertices_f,
                surface_vertices_world=surface_vertices_w,
                mask=mask,
                depth=depth,
                args=args,
            )
            uv_candidate, _z_candidate = project_world(candidate_vertices_f, frame, hand_intrinsics(frame, side, depth, frame_idx))
            valid_candidate = uv_candidate[np.isfinite(uv_candidate).all(axis=1)]
            if len(valid_candidate):
                step = max(1, int(np.ceil(len(valid_candidate) / max(1, int(args.max_review_hand_points)))))
                projected_hand_points.append({"frame_idx": int(frame_idx), "hand_side": side, "mano_surface": "articulated_candidate_mano", "points_xy": valid_candidate[::step].astype(float).tolist()})
            m["candidate_reconstruction"] = candidate_info
            m["temporal_mano_state_input"] = temporal_row.get("temporal_mano_state")
            m["coordinate_correction_accepted"] = False
            measurements.append(m)

    review_path = out_case / "late_lid_mask_mano_review_sheet.jpg"
    draw_review_sheet(
        output_path=review_path,
        frames=target_frames,
        annotations_by_frame=frames,
        mask_paths=selected_mask_paths,
        prompt_report=prompt_report,
        hand_measurements=measurements,
        projected_hand_points=projected_hand_points,
        args=args,
    )

    saved_rows = [row for row in mask_rows if row.get("saved_mask_path")]
    target_mask_rows = [row for row in mask_rows if int(row["frame_idx"]) in target_frames]
    report = {
        "method": "build_v18_trash_late_lid_part_evidence",
        "status": "ok",
        "claim_scope": (
            "Recovers raw SAM2 late lid masks from OWLv2 prompt evidence without using the old object mask as a hard gate; "
            "lifts selected masks through metric depth and remeasures MANO against observed visible surface only. "
            "No hidden geometry, part pose, signed nonpenetration, or coordinate MANO correction is accepted."
        ),
        "inputs": {
            "prompt_report": str(args.prompt_report),
            "annotations": str(args.annotations),
            "temporal_mano_state": str(args.temporal_mano_state),
            "depth_npz": str(args.depth_npz),
            "sam2_checkpoint": str(args.sam2_checkpoint),
            "reuse_mask_dir": str(args.reuse_mask_dir) if args.reuse_mask_dir is not None else None,
        },
        "parameters": {
            "target_frames": target_frames,
            "depth_order_margin_m": float(args.depth_order_margin_m),
            "nearest_surface_threshold_m": float(args.nearest_surface_threshold_m),
            "mask_stride": int(args.mask_stride),
        },
        "summary": {
            "sam2_frame_count": int(len(masks)),
            "saved_mask_count": int(len(saved_rows)),
            "target_surface_count": int(len(surface_rows)),
            "target_surface_rejection_count": int(len(rejected_surfaces)),
            "target_old_object_gate_accept_count": int(sum(1 for row in target_mask_rows if row.get("accepted_by_old_object_mask_gate") is True)),
            "target_prompt_frame_count": int(sum(1 for row in target_mask_rows if row.get("is_prompt_frame") is True)),
            "measurement_count": int(len(measurements)),
            "coordinate_correction_accepted": False,
        },
        "prompt_detections": prompt_report.get("prompt_detections"),
        "target_mask_rows": target_mask_rows,
        "saved_mask_rows_after_start": saved_rows,
        "surface_archive_npz": str(archive_path),
        "surface_rows": surface_rows,
        "rejected_surfaces": rejected_surfaces,
        "mano_measurements": measurements,
        "projected_hand_points_for_review": projected_hand_points,
        "review_sheet": str(review_path),
        "physical_conclusion_template": (
            "Interpret only after visual inspection: if the magenta mask follows the late lid/part, its observed depth surface can replace the broad hidden lid volume as a MANO constraint; "
            "if it follows background/hand or is disconnected, late part recovery is falsified and a new model-produced segmentation/articulation branch is required."
        ),
    }
    write_json(out_case / "v18_trash_late_lid_part_evidence_report.json", report)
    print(json.dumps({"report": str(out_case / "v18_trash_late_lid_part_evidence_report.json"), "review_sheet": str(review_path), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
