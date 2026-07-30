#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def run(args: argparse.Namespace) -> dict:
    mesh = trimesh.load(args.input, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"invalid mesh: {args.input}")
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    if not np.isfinite(extent).all() or float(extent.max()) <= 0.0:
        raise RuntimeError("mesh extent is invalid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    report = {
        "status": "ok",
        "input": str(args.input),
        "output": str(args.output),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "extent": [float(x) for x in extent],
        "center": [float(x) for x in vertices.mean(axis=0)],
    }
    qc = args.qc if args.qc is not None else args.output.with_name(f"qc_{args.output.stem}_convert_mesh_format_v3.json")
    qc.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qc", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
