#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree  # type: ignore[reportMissingTypeStubs]

STATUS = "v18_mesh_contact_evidence"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def finite_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def bbox_tuple(value: Any) -> tuple[float, float, float, float] | None:
    if not (isinstance(value, list) and len(value) == 4):
        return None
    vals = [finite_float(v, float("nan")) for v in value]
    if not all(math.isfinite(v) for v in vals):
        return None
    x0, y0, x1, y1 = vals
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def bbox_iou(a: Any, b: Any) -> float:
    aa = bbox_tuple(a)
    bb = bbox_tuple(b)
    if aa is None or bb is None:
        return 0.0
    ax0, ay0, ax1, ay1 = aa
    bx0, by0, bx1, by1 = bb
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def load_v16(case: str, args: argparse.Namespace) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    ann_path = args.v16_root / case / "annotations_v16_full.json"
    ann = load_json(ann_path)
    frames = {frame["frame_idx"]: frame for frame in ann.get("frames", []) if isinstance(frame, dict) and isinstance(frame.get("frame_idx"), int)}
    mesh_archive = None
    for frame in frames.values():
        obj = frame.get("object")
        if isinstance(obj, dict) and obj.get("mesh_archive"):
            mesh_archive = Path(str(obj.get("mesh_archive")))
            break
    if mesh_archive is None:
        return frames, {"ann_path": str(ann_path), "mesh_archive": None, "frame_rows": {}}
    data = np.load(mesh_archive, allow_pickle=True)
    row_by_frame = {int(f): i for i, f in enumerate(data["frame_idx"])}
    return frames, {"ann_path": str(ann_path), "mesh_archive": str(mesh_archive), "data": data, "frame_rows": row_by_frame}


def v16_object_vertices(mesh_index: dict[str, Any], frame_idx: int, max_points: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    data = mesh_index.get("data")
    rows = mesh_index.get("frame_rows")
    if data is None or not isinstance(rows, dict):
        return None, {"blocker": "missing_v16_mesh_archive"}
    row_idx = rows.get(frame_idx)
    if row_idx is None:
        return None, {"blocker": "missing_v16_mesh_frame"}
    start = int(data["vertex_offsets"][row_idx])
    end = int(data["vertex_offsets"][row_idx + 1])
    pts = np.asarray(data["vertices"][start:end], dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return None, {"blocker": "invalid_v16_mesh_vertices"}
    source_count = int(pts.shape[0])
    if pts.shape[0] > max_points:
        step = max(1, int(math.ceil(pts.shape[0] / max_points)))
        pts = pts[::step]
    return pts, {"v16_mesh_row_index": int(row_idx), "v16_mesh_source_vertices": source_count, "v16_mesh_sampled_vertices": int(pts.shape[0])}


def v16_hand_vertices(v16_frame: dict[str, Any], hand_side: str, max_points: int) -> tuple[np.ndarray | None, list[str], dict[str, Any]]:
    hands = v16_frame.get("hands")
    if not isinstance(hands, list):
        return None, ["missing_v16_hands"], {}
    hand = next((h for h in hands if isinstance(h, dict) and str(h.get("side")) == hand_side), None)
    if hand is None:
        return None, ["missing_v16_hand_side"], {}
    verts = hand.get("vertices_world_m")
    joints = hand.get("joints3d_world_m")
    source = "vertices_world_m"
    pts_raw = verts if isinstance(verts, list) and verts else joints
    if pts_raw is joints:
        source = "joints3d_world_m"
    if not isinstance(pts_raw, list) or not pts_raw:
        return None, ["missing_v16_hand_world_geometry"], {}
    pts = np.asarray(pts_raw, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        return None, ["invalid_v16_hand_world_geometry"], {}
    source_count = int(pts.shape[0])
    if pts.shape[0] > max_points:
        step = max(1, int(math.ceil(pts.shape[0] / max_points)))
        pts = pts[::step]
    return pts, [], {"v16_hand_geometry_source": source, "v16_hand_source_points": source_count, "v16_hand_sampled_points": int(pts.shape[0])}


def best_v18_object_for_v16_mesh(frame: dict[str, Any], v16_frame: dict[str, Any]) -> dict[str, Any]:
    v16_obj = v16_frame.get("object") if isinstance(v16_frame.get("object"), dict) else {}
    v16_bbox = v16_obj.get("bbox_xyxy") if isinstance(v16_obj, dict) else None
    best: dict[str, Any] = {"object_id": None, "bbox_iou": 0.0, "v16_object_label": v16_obj.get("label") if isinstance(v16_obj, dict) else None}
    for obj in frame.get("objects", []):
        if not isinstance(obj, dict):
            continue
        score = bbox_iou(v16_bbox, obj.get("bbox_xyxy"))
        if best["object_id"] is None or score > finite_float(best.get("bbox_iou"), 0.0):
            best = {"object_id": obj.get("object_id"), "name": obj.get("name"), "bbox_iou": score, "v16_object_label": v16_obj.get("label") if isinstance(v16_obj, dict) else None}
    return best


def support_from_distance(distance_m: float, sigma_m: float) -> float:
    if not math.isfinite(distance_m):
        return 0.0
    sigma = max(1e-6, sigma_m)
    return float(math.exp(-0.5 * (distance_m / sigma) ** 2))


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    annotation_path = args.full_pipeline_root / case / "annotations_v18_full.json"
    annotation_bytes = annotation_path.read_bytes()
    annotation_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
    snapshot_path = args.output_root / case / "source_annotations_for_mesh_contact.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(annotation_bytes)
    ann = json.loads(annotation_bytes.decode("utf-8"))
    v16_frames, mesh_index = load_v16(case, args)
    rows: list[dict[str, Any]] = []
    blocker_counts: dict[str, int] = {}
    finite_rows = 0
    for frame in ann.get("frames", []):
        if not isinstance(frame, dict):
            continue
        frame_idx_raw = frame.get("frame_idx")
        if not isinstance(frame_idx_raw, int):
            continue
        frame_idx = frame_idx_raw
        v16_frame = v16_frames.get(frame_idx, {})
        v16_mesh_match = best_v18_object_for_v16_mesh(frame, v16_frame)
        object_points, object_meta = v16_object_vertices(mesh_index, frame_idx, args.max_object_points)
        hand_cache: dict[str, tuple[np.ndarray | None, list[str], dict[str, Any]]] = {}
        for hyp in frame.get("contact_hypotheses", []):
            if not isinstance(hyp, dict):
                continue
            hand_side = str(hyp.get("hand_side"))
            object_id = str(hyp.get("object_id"))
            blockers: list[str] = []
            if object_points is None:
                blockers.append(str(object_meta.get("blocker", "missing_v16_object_mesh")))
            if object_id != str(v16_mesh_match.get("object_id")):
                blockers.append("v18_object_not_best_v16_mesh_bbox_match")
            if hand_side not in hand_cache:
                hand_cache[hand_side] = v16_hand_vertices(v16_frame, hand_side, args.max_hand_points)
            hand_points, hand_blockers, hand_meta = hand_cache[hand_side]
            blockers.extend(hand_blockers)
            result: dict[str, Any] = {
                "frame_idx": frame_idx,
                "hand_side": hand_side,
                "object_id": object_id,
                "source_contact_state": hyp.get("state"),
                "source_contact_confidence": hyp.get("confidence"),
                "source_contact_evidence": hyp.get("evidence"),
                "v16_mesh_match": v16_mesh_match,
                "contact_owner_claim": "not_accepted_contact_owner_v16_mesh_distance_evidence_only",
                "blockers": blockers,
            }
            if object_points is not None and hand_points is not None and not blockers:
                tree = cKDTree(object_points)
                dists, indices = tree.query(hand_points, k=1)
                min_idx = int(np.argmin(dists))
                min_dist = float(dists[min_idx])
                nearest_object_point = object_points[int(indices[min_idx])]
                support = support_from_distance(min_dist, args.contact_sigma_m)
                result.update(
                    {
                        "min_hand_surface_to_v16_object_mesh_m": min_dist,
                        "nearest_hand_surface_point_index": min_idx,
                        "nearest_hand_surface_point_world_m": [float(v) for v in hand_points[min_idx].tolist()],
                        "nearest_object_mesh_point_world_m": [float(v) for v in nearest_object_point.tolist()],
                        "mesh_contact_support_score": support,
                        "mesh_contact_energy": float((min_dist / max(args.contact_sigma_m, 1e-6)) ** 2),
                        **object_meta,
                        **hand_meta,
                    }
                )
                finite_rows += 1
            else:
                for blocker in blockers:
                    blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
            rows.append(result)
    out = {
        "method": "build_v18_mesh_contact_evidence",
        "status": STATUS,
        "claim": "Computes metric V16 MANO-hand-surface to V16 measured-object-mesh distances for the V18 object best matching the V16 object by bbox overlap. This is contact evidence, not accepted contact ownership or full nonpenetration.",
        "case": case,
        "sources": {
            "v18_full_annotations": str(annotation_path),
            "v18_full_annotations_sha256": annotation_sha256,
            "v18_full_annotations_snapshot": str(snapshot_path),
            "v16_annotations": str(mesh_index.get("ann_path")),
            "v16_object_mesh_archive": mesh_index.get("mesh_archive"),
        },
        "contact_evidence_rows": len(rows),
        "finite_mesh_distance_rows": finite_rows,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "parameters": {
            "max_object_points": args.max_object_points,
            "max_hand_points": args.max_hand_points,
            "contact_sigma_m": args.contact_sigma_m,
            "v18_object_association": "argmax_bbox_iou_to_v16_single_object_mesh_per_frame_no_acceptance_threshold",
        },
        "rows": rows,
        "contact_ownership_accepted_rows": 0,
        "contact_ownership_complete": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        "annotation_ready": True,
        "deliverable_ready": True,
    }
    write_json(args.output_root / case / "v18_mesh_contact_evidence_report.json", out)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_mesh_contact_evidence",
        "status": STATUS,
        "case_count": len(reports),
        "cases": [
            {
                "case": r["case"],
                "contact_evidence_rows": r["contact_evidence_rows"],
                "finite_mesh_distance_rows": r["finite_mesh_distance_rows"],
                "contact_ownership_accepted_rows": r["contact_ownership_accepted_rows"],
            }
            for r in reports
        ],
        "claim_scope": "metric_v16_mesh_distance_contact_evidence_not_accepted_ownership",
    }
    write_json(args.output_root / "v18_mesh_contact_evidence_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-pipeline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--v16-root", type=Path, default=Path("/data2/ego_annotation_outputs/v16_full_pipeline"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_mesh_contact_evidence"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--max-object-points", type=int, default=50000)
    parser.add_argument("--max-hand-points", type=int, default=778)
    parser.add_argument("--contact-sigma-m", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
