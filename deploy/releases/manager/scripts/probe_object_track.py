from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def surface_gap(frame: dict) -> float | None:
    obj = frame.get("object", {})
    center = obj.get("center_world_m")
    radius = obj.get("radius_m")
    if center is None or radius is None:
        return None
    c = np.asarray(center, dtype=float)
    gaps = []
    for hand in frame.get("hands", []):
        joints = np.asarray(hand.get("joints3d_world_m", []), dtype=float)
        if joints.size:
            gaps.extend((np.linalg.norm(joints - c[None, :], axis=1) - float(radius)).tolist())
        verts = np.asarray(hand.get("vertices_world_m", hand.get("vertices_world_m_sample", [])), dtype=float)
        if verts.size:
            step = max(1, len(verts) // 128)
            gaps.extend((np.linalg.norm(verts[::step] - c[None, :], axis=1) - float(radius)).tolist())
    if not gaps:
        return None
    arr = np.asarray(gaps, dtype=float)
    return float(arr[np.argmin(np.abs(arr))])


def frame_report(frame: dict) -> dict:
    obj = frame.get("object", {})
    report = {
        "frame_idx": frame.get("frame_idx"),
        "object_label": obj.get("label"),
        "status": obj.get("status"),
        "pose_status": obj.get("pose_status"),
        "bbox_xyxy": obj.get("bbox_xyxy"),
        "area_px": obj.get("area_px"),
        "proposal_source": obj.get("proposal_source"),
        "contact_ratio": obj.get("contact_ratio"),
        "min_tip_dist_px": obj.get("min_tip_dist_px"),
        "association_dist_px": obj.get("association_dist_px"),
        "prev_center_dist_px": obj.get("prev_center_dist_px"),
        "depth_m": obj.get("depth_m"),
        "radius_m": obj.get("radius_m"),
        "world_evidence": obj.get("world_evidence"),
        "surface_gap_m": surface_gap(frame),
    }
    if report["bbox_xyxy"] is not None:
        x1, y1, x2, y2 = map(float, report["bbox_xyxy"])
        report["bbox_center"] = [0.5 * (x1 + x2), 0.5 * (y1 + y2)]
        report["bbox_size"] = [x2 - x1 + 1.0, y2 - y1 + 1.0]
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = load(args.annotations)["frames"]
    by_idx = {int(frame["frame_idx"]): frame for frame in frames}
    missing = [idx for idx in args.frames if idx not in by_idx]
    if missing:
        raise RuntimeError(f"requested frames not present: {missing}")
    reports = [frame_report(by_idx[idx]) for idx in args.frames]
    gaps = [r["surface_gap_m"] for r in reports if r["surface_gap_m"] is not None and math.isfinite(r["surface_gap_m"])]
    print(json.dumps({"frames": reports, "surface_gap_abs_median_m": float(np.median(np.abs(gaps))) if gaps else None}, indent=2))


if __name__ == "__main__":
    main()
