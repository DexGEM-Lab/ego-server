#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def validate_case(path: Path) -> dict[str, Any]:
    ann = load_json(path)
    case = str(ann.get("case"))
    frames = ann.get("frames")
    require(isinstance(frames, list) and len(frames) > 0, f"{case}: missing frames")
    part_rows = 0
    part_rot_rows = 0
    part_graph_vars = 0
    part_graph_6d = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        for obj in frame.get("objects", []):
            if not isinstance(obj, dict):
                continue
            for part in obj.get("parts", []):
                if not isinstance(part, dict):
                    continue
                part_rows += 1
                pose_raw = part.get("pose_candidate")
                pose: dict[str, Any] = pose_raw if isinstance(pose_raw, dict) else {}
                rotvec = pose.get("rotation_camera_from_part_rotvec")
                if isinstance(rotvec, list) and len(rotvec) == 3:
                    part_rot_rows += 1
                    require(pose.get("type") == "approximate_part_visible_surface_pca_se3_candidate", f"{case}: part rotvec without PCA SE3 type")
        fg_raw = frame.get("factor_graph_solution")
        fg: dict[str, Any] = fg_raw if isinstance(fg_raw, dict) else {}
        variables_raw = fg.get("variables")
        variables: dict[str, Any] = variables_raw if isinstance(variables_raw, dict) else {}
        part_vars = variables.get("part_se3")
        if isinstance(part_vars, list):
            for var in part_vars:
                if not isinstance(var, dict):
                    continue
                part_graph_vars += 1
                if int(var.get("dimension", 0)) == 6 and var.get("estimate_semantics") == "translation_xyz_m_and_rotation_vector_xyz_rad":
                    part_graph_6d += 1
    require(part_rows > 0, f"{case}: no part rows")
    require(part_rot_rows > 0, f"{case}: no part PCA rotation rows")
    require(part_graph_vars > 0, f"{case}: no part graph variables")
    require(part_graph_6d > 0, f"{case}: no 6D part SE3 graph variables")
    summary_raw = ann.get("factor_graph_summary")
    summary: dict[str, Any] = summary_raw if isinstance(summary_raw, dict) else {}
    status_raw = summary.get("implemented_variable_status")
    status: dict[str, Any] = status_raw if isinstance(status_raw, dict) else {}
    require("pca_rotvec" in str(status.get("part_se3")), f"{case}: summary does not report part PCA rotvec")
    return {"case": case, "part_rows": part_rows, "part_pca_rotation_rows": part_rot_rows, "part_graph_variables": part_graph_vars, "part_graph_6d_variables": part_graph_6d}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    cases = [validate_case(args.root / case / "annotations_v18_full.json") for case in args.cases]
    print(json.dumps({"status": "ok", "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
