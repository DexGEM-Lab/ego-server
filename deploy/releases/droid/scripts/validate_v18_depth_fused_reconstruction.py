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


def validate_case(path: Path, full_annotation_path: Path | None = None) -> dict[str, Any]:
    report = load_json(path)
    case = report.get("case")
    rows = report.get("object_rows")
    require(isinstance(rows, list) and len(rows) > 0, f"{case}: object rows missing")
    require(report.get("default_path_uses_bundlesdf_or_nerf") is False, f"{case}: forbidden reconstruction backend flag")
    require(report.get("object_geometry_complete") is False, f"{case}: should not claim complete geometry")
    poisson = 0
    hull = 0
    point_clouds = 0
    total_points = 0
    for row in rows:
        require(isinstance(row, dict), f"{case}: malformed row")
        require(row.get("object_geometry_complete") is False, f"{case}: object row overclaims complete geometry")
        require("not_accepted_complete_geometry" in str(row.get("hidden_geometry_status")), f"{case}: hidden geometry limitation missing")
        mesh = row.get("mesh_reconstruction")
        require(isinstance(mesh, dict), f"{case}: mesh reconstruction missing")
        total_points += int(row.get("sampled_point_count", 0))
        pcd = mesh.get("fused_point_cloud_path")
        if pcd:
            require(Path(str(pcd)).exists(), f"{case}: missing point cloud {pcd}")
            point_clouds += 1
        path = mesh.get("poisson_mesh_path")
        if path:
            require(Path(str(path)).exists(), f"{case}: missing poisson mesh {path}")
            require(int(mesh.get("poisson_vertices", 0)) > 0 and int(mesh.get("poisson_faces", 0)) > 0, f"{case}: invalid poisson mesh counts")
            poisson += 1
        hpath = mesh.get("convex_hull_mesh_path")
        if hpath:
            require(Path(str(hpath)).exists(), f"{case}: missing hull mesh {hpath}")
            require(int(mesh.get("convex_hull_vertices", 0)) > 0 and int(mesh.get("convex_hull_faces", 0)) > 0, f"{case}: invalid hull mesh counts")
            hull += 1
    require(point_clouds > 0, f"{case}: no fused point clouds written")
    require(poisson > 0 or hull > 0, f"{case}: no reconstructed mesh artifacts written")
    full_annotation_depth_fused_states = 0
    if full_annotation_path is not None and full_annotation_path.exists():
        ann = load_json(full_annotation_path)
        for frame in ann.get("frames", []):
            if not isinstance(frame, dict):
                continue
            for obj in frame.get("objects", []):
                if not isinstance(obj, dict):
                    continue
                hidden = obj.get("hidden_geometry_candidate")
                if isinstance(hidden, dict) and hidden.get("method") == "depth_fused_visible_surface_poisson_and_hull_candidate" and hidden.get("poisson_mesh_path"):
                    full_annotation_depth_fused_states += 1
        require(full_annotation_depth_fused_states > 0, f"{case}: full pipeline annotations do not consume depth-fused geometry")
    return {"case": case, "object_rows": len(rows), "point_clouds": point_clouds, "poisson_meshes": poisson, "hull_meshes": hull, "sampled_points": total_points, "full_annotation_depth_fused_states": full_annotation_depth_fused_states}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_depth_fused_reconstruction"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--full-pipeline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_full_pipeline"))
    args = parser.parse_args()
    rows = []
    for case in args.cases:
        rows.append(validate_case(args.root / case / "v18_depth_fused_reconstruction_report.json", args.full_pipeline_root / case / "annotations_v18_full.json"))
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
