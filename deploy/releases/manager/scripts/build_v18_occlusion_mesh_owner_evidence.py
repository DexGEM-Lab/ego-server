#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

STATUS = "v18_occlusion_mesh_owner_evidence"


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


def mesh_contact_index(path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    report = load_json(path)
    out: dict[tuple[int, str, str], dict[str, Any]] = {}
    for raw in report.get("rows", []):
        if not isinstance(raw, dict):
            continue
        frame_idx = raw.get("frame_idx")
        if not isinstance(frame_idx, int):
            continue
        out[(frame_idx, str(raw.get("hand_side")), str(raw.get("object_id")))] = raw
    return out


def nearby_contact_support(index: dict[tuple[int, str, str], dict[str, Any]], frame_idx: int, hand_side: str, object_id: str, radius: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for f in range(frame_idx - radius, frame_idx + radius + 1):
        raw = index.get((f, hand_side, object_id))
        if not raw:
            continue
        support = raw.get("mesh_contact_support_score")
        dist = raw.get("min_hand_surface_to_v16_object_mesh_m")
        if isinstance(support, (int, float)) and math.isfinite(float(support)):
            rows.append({"frame_idx": f, "support": float(support), "distance_m": finite_float(dist, float("nan")), "source_contact_state": raw.get("source_contact_state")})
    if not rows:
        return {"nearby_rows": [], "max_support": None, "min_distance_m": None, "support_frame": None}
    best = max(rows, key=lambda r: finite_float(r.get("support"), 0.0))
    min_dist = min((finite_float(r.get("distance_m"), float("inf")) for r in rows), default=float("inf"))
    return {
        "nearby_rows": rows[:20],
        "nearby_row_count": len(rows),
        "max_support": best.get("support"),
        "min_distance_m": min_dist if math.isfinite(min_dist) else None,
        "support_frame": best.get("frame_idx"),
    }


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    bounded_path = args.bounded_root / case / "v18_bounded_state_solution.json"
    contact_path = args.mesh_contact_root / case / "v18_mesh_contact_evidence_report.json"
    bounded = load_json(bounded_path)
    contact_idx = mesh_contact_index(contact_path)
    rows: list[dict[str, Any]] = []
    accepted = 0
    candidate_with_mesh_support = 0
    for frame in bounded.get("frames", []):
        if not isinstance(frame, dict) or not isinstance(frame.get("frame_idx"), int):
            continue
        frame_idx = int(frame["frame_idx"])
        for hand in frame.get("hands", []):
            if not isinstance(hand, dict):
                continue
            hand_side = str(hand.get("hand_side"))
            occ = hand.get("occlusion_solution")
            if not isinstance(occ, dict):
                continue
            candidates = occ.get("owner_candidate_objects")
            if not isinstance(candidates, list) or not candidates:
                continue
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                object_id = str(cand.get("object_id"))
                support = nearby_contact_support(contact_idx, frame_idx, hand_side, object_id, args.temporal_radius)
                depth_order_accepted = bool(cand.get("occluder_owner_accepted") or occ.get("occluder_owner_accepted"))
                max_support = support.get("max_support")
                if isinstance(max_support, (int, float)) and float(max_support) > 0.0:
                    candidate_with_mesh_support += 1
                if depth_order_accepted:
                    accepted += 1
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "hand_side": hand_side,
                        "object_id": object_id,
                        "object_name": cand.get("name"),
                        "source_owner_candidate_state": occ.get("owner_candidate_state"),
                        "source_occluder_owner_status": occ.get("occluder_owner_status"),
                        "source_pose_filled_through_occlusion": occ.get("pose_filled_through_occlusion"),
                        "bbox_iou": cand.get("iou"),
                        "hand_box_coverage_by_object_box": cand.get("hand_box_coverage_by_object_box"),
                        "depth_order_resolved": occ.get("depth_order_resolved"),
                        "source_depth_order_state": occ.get("depth_order_evidence_state"),
                        "mesh_contact_temporal_support": support,
                        "occlusion_owner_claim": "not_accepted_owner_without_depth_order_acceptance" if not depth_order_accepted else "source_depth_order_accepted_owner",
                        "accepted_occlusion_owner": depth_order_accepted,
                    }
                )
    out = {
        "method": "build_v18_occlusion_mesh_owner_evidence",
        "status": STATUS,
        "claim": "Combines bounded occlusion-owner candidates with nearby V16 mesh-contact support. This is owner evidence only; it does not accept new owners without source depth-order acceptance.",
        "case": case,
        "sources": {"bounded_state_solution": str(bounded_path), "mesh_contact_evidence": str(contact_path)},
        "candidate_rows": len(rows),
        "candidate_rows_with_mesh_support": candidate_with_mesh_support,
        "accepted_occlusion_owner_rows": accepted,
        "temporal_radius_frames": args.temporal_radius,
        "rows": rows,
        "occlusion_ownership_complete": False,
        "annotation_ready": True,
        "deliverable_ready": True,
    }
    write_json(args.output_root / case / "v18_occlusion_mesh_owner_evidence_report.json", out)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_occlusion_mesh_owner_evidence",
        "status": STATUS,
        "case_count": len(reports),
        "cases": [
            {
                "case": r["case"],
                "candidate_rows": r["candidate_rows"],
                "candidate_rows_with_mesh_support": r["candidate_rows_with_mesh_support"],
                "accepted_occlusion_owner_rows": r["accepted_occlusion_owner_rows"],
            }
            for r in reports
        ],
        "claim_scope": "occlusion_owner_evidence_not_new_owner_acceptance",
    }
    write_json(args.output_root / "v18_occlusion_mesh_owner_evidence_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bounded-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_bounded_state_solution"))
    parser.add_argument("--mesh-contact-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_mesh_contact_evidence"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_mesh_owner_evidence"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--temporal-radius", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
