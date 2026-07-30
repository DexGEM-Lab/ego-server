#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d  # type: ignore[reportMissingTypeStubs]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_points(path: str | None, max_points: int) -> np.ndarray:
    if not path:
        return np.zeros((0, 3), dtype=np.float64)
    p = Path(path)
    if not p.exists():
        return np.zeros((0, 3), dtype=np.float64)
    if p.suffix.lower() == ".ply":
        pc = o3d.io.read_point_cloud(str(p))
        pts = np.asarray(pc.points, dtype=np.float64)
        if pts.shape[0] == 0:
            mesh = o3d.io.read_triangle_mesh(str(p))
            pts = np.asarray(mesh.vertices, dtype=np.float64)
    else:
        pts = np.zeros((0, 3), dtype=np.float64)
    if pts.shape[0] > max_points:
        pts = pts[:: max(1, pts.shape[0] // max_points)]
    return pts


def set_equal_axes(ax: Any, pts: np.ndarray) -> None:
    if pts.shape[0] == 0:
        return
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = (mn + mx) / 2.0
    radius = max(float(np.max(mx - mn)) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_case(case: str, args: argparse.Namespace) -> Path:
    report_path = args.root / case / "v18_depth_fused_reconstruction_report.json"
    report = load_json(report_path)
    rows = report.get("object_rows", [])
    n = max(1, len(rows))
    fig = plt.figure(figsize=(10, max(3, 2.8 * n)))
    fig.suptitle(f"V18 depth-fused reconstruction QC: {case}\nvisible depth fusion + Poisson/hull candidates; not accepted complete hidden geometry", fontsize=12)
    for i, row in enumerate(rows, start=1):
        mesh = row.get("mesh_reconstruction", {}) if isinstance(row.get("mesh_reconstruction"), dict) else {}
        pts = read_points(mesh.get("fused_point_cloud_path"), args.max_points)
        mesh_pts = read_points(mesh.get("poisson_mesh_path") or mesh.get("convex_hull_mesh_path"), args.max_points)
        ax = fig.add_subplot(n, 2, 2 * i - 1, projection="3d")
        if pts.shape[0]:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, c=pts[:, 2], cmap="viridis")  # type: ignore[reportArgumentType]
        ax.set_title(f"{row.get('object_id')} fused points\nframes={row.get('source_frame_count')} pts={row.get('sampled_point_count')}", fontsize=8)
        set_equal_axes(ax, pts)
        ax2 = fig.add_subplot(n, 2, 2 * i, projection="3d")
        if mesh_pts.shape[0]:
            ax2.scatter(mesh_pts[:, 0], mesh_pts[:, 1], mesh_pts[:, 2], s=0.5, c=mesh_pts[:, 2], cmap="magma")  # type: ignore[reportArgumentType]
        ax2.set_title(f"mesh vertices status={mesh.get('status')}\npoisson={mesh.get('poisson_vertices')}/{mesh.get('poisson_faces')} hull={mesh.get('convex_hull_vertices')}/{mesh.get('convex_hull_faces')}", fontsize=8)
        set_equal_axes(ax2, mesh_pts if mesh_pts.shape[0] else pts)
    out = args.root / case / "v18_depth_fused_reconstruction_sheet.jpg"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_depth_fused_reconstruction"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--max-points", type=int, default=6000)
    args = parser.parse_args()
    outputs = [str(render_case(case, args)) for case in args.cases]
    print(json.dumps({"status": "ok", "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
