#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from v20_common import ContractError, write_json


def load_hawor_process(hawor_root: Path):
    root = str(hawor_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from hawor.utils.rotation import rotation_matrix_to_angle_axis  # type: ignore
    from hawor.utils.process import run_mano, run_mano_left  # type: ignore
    return rotation_matrix_to_angle_axis, run_mano, run_mano_left


def infer_side(track_index: int, side_hint: str | None) -> str:
    if side_hint in {"left", "right"}:
        return side_hint
    return "left" if int(track_index) == 0 else "right"


def adapt(args: argparse.Namespace) -> dict:
    rotation_matrix_to_angle_axis, run_mano, run_mano_left = load_hawor_process(args.hawor_root)
    rows = []
    side_rows = []
    chunk_paths = sorted(args.cam_space.glob("*/*.json"))
    if not chunk_paths:
        raise ContractError(f"missing_hawor_cam_space_chunks: {args.cam_space}")
    old_cwd = Path.cwd()
    os.chdir(args.hawor_root)
    try:
        chunk_iter = list(chunk_paths)
    finally:
        os.chdir(old_cwd)
    for path in chunk_iter:
        track_index = int(path.parent.name)
        side = infer_side(track_index, args.side_hint)
        data = json.loads(path.read_text(encoding="utf-8"))
        root_rotmat = torch.tensor(data["init_root_orient"], dtype=torch.float32)
        hand_rotmat = torch.tensor(data["init_hand_pose"], dtype=torch.float32)
        trans = torch.tensor(data["init_trans"], dtype=torch.float32)
        betas = torch.tensor(data["init_betas"], dtype=torch.float32)
        if root_rotmat.ndim != 4 or hand_rotmat.ndim != 5 or trans.ndim != 3:
            raise ContractError(f"invalid_hawor_chunk_shapes: {path}")
        root_aa = rotation_matrix_to_angle_axis(root_rotmat)
        hand_aa = rotation_matrix_to_angle_axis(hand_rotmat)
        if args.cpu:
            root_aa = root_aa.cpu()
            hand_aa = hand_aa.cpu()
            trans = trans.cpu()
            betas = betas.cpu()
        old_cwd = Path.cwd()
        os.chdir(args.hawor_root)
        try:
            if side == "left":
                out = run_mano_left(trans, root_aa, hand_aa, betas=betas, use_cuda=not args.cpu)
            else:
                out = run_mano(trans, root_aa, hand_aa, betas=betas, use_cuda=not args.cpu)
        finally:
            os.chdir(old_cwd)
        joints = out["joints"][0].detach().cpu().numpy().astype(np.float32)
        verts = out["vertices"][0].detach().cpu().numpy().astype(np.float32)
        T = joints.shape[0]
        if trans.shape[1] != T:
            raise ContractError(f"hawor_mano_frame_count_mismatch: {path}")
        frame_start = int(path.stem.split("_")[0])
        for local_i in range(T):
            frame_idx = frame_start + local_i + int(args.source_frame_offset)
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "side": side,
                    "track_index": track_index,
                    "joints": joints[local_i],
                    "vertices": verts[local_i],
                    "betas": betas[0, local_i].detach().cpu().numpy().astype(np.float32),
                    "trans": trans[0, local_i].detach().cpu().numpy().astype(np.float32),
                    "source_chunk": str(path),
                }
            )
        side_rows.append({"path": str(path), "track_index": track_index, "side": side, "frames": T})
    if not rows:
        raise ContractError("hawor_camspace_adapter_produced_no_rows")
    frame_idx = np.asarray([r["frame_idx"] for r in rows], dtype=np.int32)
    side = np.asarray([r["side"] for r in rows], dtype="<U8")
    joints = np.stack([r["joints"] for r in rows], axis=0).astype(np.float32)
    vertices = np.stack([r["vertices"] for r in rows], axis=0).astype(np.float32)
    betas = np.stack([r["betas"] for r in rows], axis=0).astype(np.float32)
    trans = np.stack([r["trans"] for r in rows], axis=0).astype(np.float32)
    source_chunk = np.asarray([r["source_chunk"] for r in rows], dtype=object)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        frame_idx=frame_idx,
        side=side,
        joints_camera_m=joints,
        vertices_camera_m=vertices,
        betas=betas,
        trans_camera_m=trans,
        source_chunk=source_chunk,
        coordinate_status=np.asarray(["hawor_camera_space_prediction_side"], dtype=object),
    )
    report = {
        "status": "ok",
        "method": "adapt_hawor_camspace_to_v20_mano_npz",
        "hawor_root": str(args.hawor_root),
        "cam_space": str(args.cam_space),
        "output_npz": str(args.output_npz),
        "row_count": int(len(rows)),
        "unique_frames": int(len(set(int(x) for x in frame_idx.tolist()))),
        "sides": sorted(set(str(x) for x in side.tolist())),
        "chunks": side_rows,
        "eval_refs_loaded": False,
        "claim_scope": "Converts HaWoR prediction-side camera-space MANO chunks into V20 metric MANO npz; no benchmark eval refs are read.",
    }
    write_json(args.output_report, report)
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt HaWoR camera-space chunks to V20 MANO npz.")
    parser.add_argument("--hawor-root", type=Path, required=True)
    parser.add_argument("--cam-space", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--side-hint", choices=("left", "right"), default=None)
    parser.add_argument("--source-frame-offset", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    adapt(parse_args())
