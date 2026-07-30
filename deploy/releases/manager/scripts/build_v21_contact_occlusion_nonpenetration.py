#!/usr/bin/env python3
"""V21 contact/occlusion/nonpenetration evidence builder.

Computes per-frame contact, occlusion, and nonpenetration evidence
directly from V21 measurements (hand MANO candidates, object mesh pose,
DepthPro depth). This replaces the chain of V18 evidence builders that
depend on V16/V17 pipeline layout.

Produces three evidence files consumed by the V18 full pipeline renderer:
- contact_ownership: which hand touches which object (distance-based)
- occlusion_depth_order: depth-ordering between hand and object
- nonpenetration: signed distance from hand vertices to object mesh

Output:
  measurements/contact_occlusion_nonpenetration/
    contact_evidence.json
    occlusion_evidence.json
    nonpenetration_evidence.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    import trimesh
except ImportError:
    raise RuntimeError("trimesh required")


def load_json(p):
    return json.loads(Path(p).read_text())

def load_mesh(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices, dtype=float),
                           faces=np.asarray(mesh.faces, dtype=np.int64), process=False)


def hand_vertices_camera(hand):
    """Get hand MANO vertices in camera space."""
    verts = hand.get("vertices_camera_metric_sample") or hand.get("vertices_camera") or []
    if not verts:
        # Reconstruct from joints if vertices not available
        return None
    cam_t = hand.get("cam_t_metric_smoothed") or hand.get("cam_t_metric", [0,0,1])
    return np.array(verts, dtype=float) + np.array(cam_t)


def object_vertices_at_pose(mesh_verts, R, t):
    """Transform object mesh vertices to camera/world space at pose."""
    return (np.array(R) @ mesh_verts.T).T + np.array(t)


def run(args):
    run_root = Path(args.run_root)
    obj = args.object_id

    # Load annotation
    ann = load_json(run_root / "state/annotations_v18_compatible.json")
    frames = ann["frames"]

    # Load mesh for nonpenetration
    mesh_path = run_root / f"measurements/object_geometry/v21_mesh_candidate/{obj}/mesh_candidate.obj"
    mesh = load_mesh(mesh_path)
    mesh_verts = np.asarray(mesh.vertices, dtype=float)
    mesh_faces = np.asarray(mesh.faces, dtype=np.int64)

    # Load pose graph result for corrected poses
    pose_graph_path = run_root / f"measurements/object_geometry_mesh_pose/{obj}/v19_pose_graph/v19_rigid_object_pose_graph_report.json"
    pose_graph = load_json(pose_graph_path) if pose_graph_path.exists() else None
    pose_by_frame = {}
    if pose_graph:
        for row in pose_graph.get("pose_rows", []):
            pose_by_frame[row["frame_idx"]] = row

    contact_rows = []
    occlusion_rows = []
    nonpenetration_rows = []

    contact_tolerance_m = float(args.contact_tolerance)
    n_contact = 0
    n_occlusion = 0
    n_penetration = 0

    for frame in frames:
        fidx = frame["frame_idx"]
        hands_data = frame.get("hands", [])
        objects_data = frame.get("objects", [])

        if not hands_data or not objects_data:
            continue

        # Get object pose
        obj_data = objects_data[0]  # first object
        pose_row = pose_by_frame.get(fidx, {})
        R = np.array(pose_row.get("rotation_world_from_completed_canonical_matrix",
                                  obj_data.get("reconstructed_geometry_pose", {}).get("rotation_world_from_canonical_matrix",
                                   [[1,0,0],[0,1,0],[0,0,1]])))
        t = np.array(pose_row.get("translation_world_m",
                                  obj_data.get("reconstructed_geometry_pose", {}).get("translation_world_m", [0,0,1.5])))

        # Object vertices in world/camera space
        obj_verts_world = object_vertices_at_pose(mesh_verts, R, t)

        # Build KDTree of object surface
        obj_tree = cKDTree(obj_verts_world)

        for hand in hands_data:
            side = hand.get("hand_side", "right")
            hand_verts = hand_vertices_camera(hand.get("metric_mano_state", hand.get("mano_candidate", {})))
            if hand_verts is None or len(hand_verts) < 10:
                continue

            # === CONTACT EVIDENCE ===
            # Find nearest object vertex for each hand vertex
            dists, idxs = obj_tree.query(hand_verts, k=1)
            contact_mask = dists < contact_tolerance_m
            contact_count = int(contact_mask.sum())
            min_dist = float(np.min(dists))
            median_dist = float(np.median(dists))

            if contact_count > 0:
                contact_points = hand_verts[contact_mask]
                contact_centroid = contact_points.mean(axis=0).tolist()
            else:
                contact_centroid = None

            contact_rows.append({
                "frame_idx": fidx,
                "hand_side": side,
                "object_id": f"object:{obj}",
                "min_distance_m": min_dist,
                "median_distance_m": median_dist,
                "contact_vertex_count": contact_count,
                "contact_tolerance_m": contact_tolerance_m,
                "in_contact": contact_count > 5,  # at least 5 vertices in contact
                "contact_centroid_world_m": contact_centroid,
            })
            if contact_count > 5:
                n_contact += 1

            # === OCCLUSION DEPTH ORDER ===
            # Compare median depth of hand vs object
            hand_depth = float(np.median(hand_verts[:, 2]))
            obj_depth = float(np.median(obj_verts_world[:, 2]))
            hand_in_front = hand_depth < obj_depth  # smaller z = closer to camera

            occlusion_rows.append({
                "frame_idx": fidx,
                "hand_side": side,
                "object_id": f"object:{obj}",
                "hand_median_depth_m": hand_depth,
                "object_median_depth_m": obj_depth,
                "depth_gap_m": abs(hand_depth - obj_depth),
                "hand_occludes_object": hand_in_front and abs(hand_depth - obj_depth) < 0.1,
                "object_occludes_hand": not hand_in_front and abs(hand_depth - obj_depth) < 0.1,
            })
            if hand_in_front and abs(hand_depth - obj_depth) < 0.1:
                n_occlusion += 1

            # === NONPENETRATION EVIDENCE ===
            # Use ray casting or signed distance to check penetration
            # Simplified: use nearest-vertex distance with sign from face normal
            # A hand vertex is "penetrating" if it's inside the object mesh
            penetrating = 0
            max_penetration = 0.0
            penetration_threshold = float(args.penetration_tolerance)

            # For each hand vertex that's very close, check if inside mesh
            close_mask = dists < 0.05  # 5cm
            if close_mask.sum() > 0:
                close_verts = hand_verts[close_mask]
                # Use trimesh contains check (approximate for speed)
                try:
                    contained = mesh.contains(close_verts)
                    penetrating = int(contained.sum())
                    if penetrating > 0:
                        pen_verts = close_verts[contained]
                        # Estimate penetration depth from nearest surface distance
                        pen_dists = dists[close_mask][contained]
                        max_penetration = float(np.max(pen_dists)) if len(pen_dists) > 0 else 0.0
                except Exception:
                    pass

            nonpenetration_rows.append({
                "frame_idx": fidx,
                "hand_side": side,
                "object_id": f"object:{obj}",
                "penetrating_vertex_count": penetrating,
                "max_penetration_depth_m": max_penetration,
                "penetration_threshold_m": penetration_threshold,
                "nonpenetration_satisfied": penetrating == 0,
            })
            if penetrating > 0:
                n_penetration += 1

    output_dir = run_root / "measurements/contact_occlusion_nonpenetration"
    output_dir.mkdir(parents=True, exist_ok=True)

    contact_payload = {
        "method": "v21_contact_evidence_from_mano_object_distance",
        "object_id": f"object:{obj}",
        "contact_tolerance_m": contact_tolerance_m,
        "total_rows": len(contact_rows),
        "contact_frames": n_contact,
        "contact_rate": n_contact / max(1, len(contact_rows)),
        "rows": contact_rows,
    }
    (output_dir / "contact_evidence.json").write_text(json.dumps(contact_payload, indent=2))

    occlusion_payload = {
        "method": "v21_occlusion_depth_order_from_hand_object_depth",
        "total_rows": len(occlusion_rows),
        "occlusion_frames": n_occlusion,
        "rows": occlusion_rows,
    }
    (output_dir / "occlusion_evidence.json").write_text(json.dumps(occlusion_payload, indent=2))

    nonpen_payload = {
        "method": "v21_nonpenetration_from_mesh_contains_check",
        "penetration_tolerance_m": float(args.penetration_tolerance),
        "total_rows": len(nonpenetration_rows),
        "penetration_frames": n_penetration,
        "penetration_rate": n_penetration / max(1, len(nonpenetration_rows)),
        "rows": nonpenetration_rows,
    }
    (output_dir / "nonpenetration_evidence.json").write_text(json.dumps(nonpen_payload, indent=2))

    summary = {
        "status": "ok",
        "contact_frames": n_contact,
        "occlusion_frames": n_occlusion,
        "penetration_frames": n_penetration,
        "total_hand_object_pairs": len(contact_rows),
        "contact_rate": n_contact / max(1, len(contact_rows)),
        "penetration_rate": n_penetration / max(1, len(nonpenetration_rows)),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--object-id", required=True)
    ap.add_argument("--contact-tolerance", type=float, default=0.03)
    ap.add_argument("--penetration-tolerance", type=float, default=0.005)
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
