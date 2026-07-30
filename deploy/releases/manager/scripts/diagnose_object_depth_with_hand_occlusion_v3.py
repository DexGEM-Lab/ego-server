#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fit_mano_to_hand_mask_depth_v3 import load_mano_faces
from optimize_contact_patch_object_pose_graph_v3 import hand_vertices_camera
from render_bundlesdf_mesh_qc_v3 import camera_points, intrinsics_for_frame, load_depth_archive, load_json, load_mesh_archive
from render_mesh_zbuffer_qc_v3 import summarize_by_mask_distance, triangle_zbuffer


def summarize(values: np.ndarray | list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def project_camera(vertices_camera: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = vertices_camera[:, 2]
    uv = np.full((len(vertices_camera), 2), np.nan, dtype=np.float64)
    positive = z > 0.0
    uv[positive, 0] = K[0, 0] * vertices_camera[positive, 0] / z[positive] + K[0, 2]
    uv[positive, 1] = K[1, 1] * vertices_camera[positive, 1] / z[positive] + K[1, 2]
    return uv, z


def visible_object_error_row(
    frame_idx: int,
    object_zbuf: np.ndarray,
    hand_zbuf: np.ndarray,
    object_mask: np.ndarray,
    depth_m: np.ndarray,
    occlusion_margin_m: float,
) -> dict:
    valid_object = np.isfinite(object_zbuf) & object_mask & np.isfinite(depth_m) & (depth_m > 0.0)
    hand_in_front = np.isfinite(hand_zbuf) & (hand_zbuf < object_zbuf - float(occlusion_margin_m))
    unoccluded = valid_object & ~hand_in_front
    occluded = valid_object & hand_in_front
    distance_to_mask_edge = cv2.distanceTransform(object_mask.astype(np.uint8), cv2.DIST_L2, 3)

    def block(mask: np.ndarray) -> dict:
        err = object_zbuf[mask].astype(np.float64) - depth_m[mask]
        dist = distance_to_mask_edge[mask].astype(np.float64)
        return {
            "samples": int(len(err)),
            "signed_m": summarize(err),
            "abs_m": summarize(np.abs(err)),
            "closer_than_depth_fraction_5mm": float(np.mean(err < -0.005)) if len(err) else None,
            "farther_than_depth_fraction_5mm": float(np.mean(err > 0.005)) if len(err) else None,
            "by_mask_distance": summarize_by_mask_distance(err, dist) if len(err) else [],
        }

    return {
        "frame_idx": int(frame_idx),
        "valid_object_samples": int(np.count_nonzero(valid_object)),
        "hand_occluded_samples": int(np.count_nonzero(occluded)),
        "hand_occluded_fraction": float(np.count_nonzero(occluded) / max(np.count_nonzero(valid_object), 1)),
        "all": block(valid_object),
        "hand_unoccluded": block(unoccluded),
        "hand_occluded": block(occluded),
    }


def run(args: argparse.Namespace) -> dict:
    manifest = load_json(args.manifest)
    annotations = load_json(args.annotations)
    entries = manifest.get("frames")
    frames = annotations.get("frames")
    if not isinstance(entries, list) or not isinstance(frames, list):
        raise RuntimeError("manifest and annotations must contain frames lists")
    annotations_by_idx = {int(frame["frame_idx"]): frame for frame in frames}
    meshes = load_mesh_archive(args.mesh_archive)
    depth_archive = load_depth_archive(args.metric_depth_npz)
    hand_faces = load_mano_faces(args.mano_model)
    rows = []
    for entry in entries:
        frame_idx = int(entry["frame_idx"])
        if args.frame_start is not None and frame_idx < int(args.frame_start):
            continue
        if args.frame_end is not None and frame_idx > int(args.frame_end):
            continue
        if frame_idx not in annotations_by_idx:
            raise RuntimeError(f"annotations lack frame {frame_idx}")
        if frame_idx not in meshes:
            raise RuntimeError(f"mesh archive lacks frame {frame_idx}")
        if frame_idx not in depth_archive:
            raise RuntimeError(f"metric depth archive lacks frame {frame_idx}")
        mask = cv2.imread(str(Path(entry["mask"])), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask for frame {frame_idx}")
        object_mask = mask > 0
        depth_m = np.asarray(depth_archive[frame_idx], dtype=np.float64)
        if depth_m.shape != object_mask.shape:
            raise RuntimeError(f"depth shape {depth_m.shape} does not match mask shape {object_mask.shape}")
        annotation = annotations_by_idx[frame_idx]
        K = intrinsics_for_frame(args, entry, annotation)
        T_world_camera = np.asarray(annotation["camera"]["T_world_camera_metric"], dtype=np.float64)
        vertices_world, object_faces = meshes[frame_idx]
        object_camera = camera_points(vertices_world, T_world_camera)
        object_uv, object_z = project_camera(object_camera, K)
        object_zbuf = triangle_zbuffer(object_mask.shape, object_uv, object_z, object_faces, args.max_faces)
        hand_vertices = []
        hand_faces_all = []
        offset = 0
        for hand in annotation.get("hands", []):
            vertices = hand_vertices_camera(hand)
            hand_vertices.append(vertices)
            hand_faces_all.append(hand_faces + offset)
            offset += int(len(vertices))
        if not hand_vertices:
            raise RuntimeError(f"frame {frame_idx} has no MANO hand vertices")
        hand_camera = np.vstack(hand_vertices)
        hand_uv, hand_z = project_camera(hand_camera, K)
        hand_zbuf = triangle_zbuffer(object_mask.shape, hand_uv, hand_z, np.vstack(hand_faces_all), args.max_hand_faces)
        rows.append(
            visible_object_error_row(
                frame_idx,
                object_zbuf,
                hand_zbuf,
                object_mask,
                depth_m,
                float(args.occlusion_margin_m),
            )
        )
    if not rows:
        raise RuntimeError("no rows generated")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ok",
        "method": "object_depth_with_hand_occlusion_v3",
        "mesh_archive": str(args.mesh_archive),
        "manifest": str(args.manifest),
        "annotations": str(args.annotations),
        "metric_depth_npz": str(args.metric_depth_npz),
        "intrinsics_source": str(args.intrinsics_source),
        "mano_model": str(args.mano_model),
        "rows": rows,
        "summary": {
            "hand_occluded_fraction": summarize([row["hand_occluded_fraction"] for row in rows]),
            "all_abs_median_m": summarize([row["all"]["abs_m"].get("median", np.nan) for row in rows]),
            "all_abs_p95_m": summarize([row["all"]["abs_m"].get("p95", np.nan) for row in rows]),
            "unoccluded_abs_median_m": summarize([row["hand_unoccluded"]["abs_m"].get("median", np.nan) for row in rows]),
            "unoccluded_abs_p95_m": summarize([row["hand_unoccluded"]["abs_m"].get("p95", np.nan) for row in rows]),
            "occluded_abs_median_m": summarize([row["hand_occluded"]["abs_m"].get("median", np.nan) for row in rows]),
            "occluded_abs_p95_m": summarize([row["hand_occluded"]["abs_m"].get("p95", np.nan) for row in rows]),
        },
        "parameters": {
            "occlusion_margin_m": float(args.occlusion_margin_m),
            "max_faces": None if args.max_faces is None else int(args.max_faces),
            "max_hand_faces": None if args.max_hand_faces is None else int(args.max_hand_faces),
        },
    }
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metric-depth-npz", type=Path, required=True)
    parser.add_argument("--mano-model", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--intrinsics-source", choices=["manifest", "annotation-vggt"], default="manifest")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--max-hand-faces", type=int, default=0)
    parser.add_argument("--occlusion-margin-m", type=float, default=0.003)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_faces is not None and int(args.max_faces) <= 0:
        args.max_faces = None
    if args.max_hand_faces is not None and int(args.max_hand_faces) <= 0:
        args.max_hand_faces = None
    run(args)


if __name__ == "__main__":
    main()
