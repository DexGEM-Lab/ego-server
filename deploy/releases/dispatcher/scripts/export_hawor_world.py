#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve_path(path: str | Path, *, base: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def file_info(path: Path, *, hash_file: bool = False) -> dict:
    info = {"path": str(path), "exists": path.exists(), "is_file": path.is_file() if path.exists() else False, "bytes": path.stat().st_size if path.exists() and path.is_file() else None}
    if hash_file and path.exists() and path.is_file():
        info["sha256"] = sha256(path)
    return info


def prepare_focal_cache_contract(seq_folder: Path, start_idx: int, end_idx: int, img_focal: float | None, *, force_refresh: bool) -> dict:
    """Prevent silent reuse of focal-dependent HaWoR cache products.

    HaWoR stores motion chunks, rendered hand masks, and SLAM outputs under a
    sequence folder keyed by the video pathname.  Those artifacts depend on
    img_focal, but the upstream code does not include focal in the cache key.  A
    V19 rerun that changes --img_focal must therefore either use a new sequence
    folder or explicitly invalidate focal-dependent artifacts.
    """
    if img_focal is None:
        return {"enabled": False, "reason": "img_focal_not_explicit"}
    focal = float(img_focal)
    tracks_dir = seq_folder / f"tracks_{int(start_idx)}_{int(end_idx)}"
    contract_path = seq_folder / "v19_hawor_focal_cache_contract.json"
    focal_artifacts = [
        tracks_dir / "frame_chunks_all.npy",
        tracks_dir / "model_masks.npy",
        seq_folder / "SLAM" / f"hawor_slam_w_scale_{int(start_idx)}_{int(end_idx)}.npz",
    ]
    focal_dirs = [seq_folder / "cam_space"]
    existing = [str(p) for p in focal_artifacts if p.exists()] + [str(p) for p in focal_dirs if p.exists()]
    previous: dict | None = None
    if contract_path.exists():
        try:
            previous = json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"invalid HaWoR focal cache contract {contract_path}: {exc}") from exc
    previous_focal = None
    if isinstance(previous, dict) and previous.get("img_focal") is not None:
        previous_focal = float(previous["img_focal"])
    compatible = previous_focal is not None and abs(previous_focal - focal) <= max(1.0e-3, abs(focal) * 1.0e-6)
    removed: list[str] = []
    if existing and not compatible:
        if not force_refresh:
            raise RuntimeError(
                "HaWoR focal-dependent cache exists under the sequence folder but does not match the requested --img_focal. "
                f"seq_folder={seq_folder} requested={focal} previous={previous_focal} existing={existing[:6]}. "
                "Use a focal-specific video path/sequence folder or pass --force-focal-cache-refresh to delete motion/SLAM cache artifacts."
            )
        for p in focal_artifacts:
            if p.exists():
                p.unlink()
                removed.append(str(p))
        for p in focal_dirs:
            if p.exists():
                shutil.rmtree(p)
                removed.append(str(p))
    payload = {
        "status": "ok",
        "img_focal": focal,
        "seq_folder": str(seq_folder),
        "tracks_range": [int(start_idx), int(end_idx)],
        "focal_dependent_artifacts_seen_before_refresh": existing,
        "removed_for_force_refresh": removed,
        "compatible_previous_contract": bool(compatible),
        "force_refresh": bool(force_refresh),
        "claim_scope": "guards HaWoR motion/mask/SLAM cache reuse for focal-dependent V19 metric hand export",
    }
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {**payload, "path": str(contract_path)}


def load_track_support(seq_folder: Path, start_idx: int, end_idx: int, frame_count: int) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Return same-frame HaWoR detection support by side.

    HaWoR's infiller can produce MANO rows where there is no same-frame detector
    box. The exported metric MANO arrays are still useful evidence, but V18 must
    not conflate infilled rows with detector-supported rows. This function keeps
    that provenance in the NPZ/QC contract.
    """
    tracks_path = seq_folder / f"tracks_{start_idx}_{end_idx}" / "model_tracks.npy"
    support: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        support[side] = {
            "detected_same_frame": np.zeros(frame_count, dtype=np.uint8),
            "det_box_xyxyscore": np.full((frame_count, 5), np.nan, dtype=np.float32),
            "track_id": np.full(frame_count, "", dtype="<U64"),
        }
    report = {
        "tracks_path": str(tracks_path),
        "tracks_file_exists": tracks_path.exists(),
        "side_handedness_mapping": {"left": 0, "right": 1},
        "records_read": 0,
        "records_used": 0,
    }
    if not tracks_path.exists():
        report["status"] = "tracks_file_missing_detection_support_unavailable"
        return support, report
    try:
        tracks_obj = np.load(tracks_path, allow_pickle=True)
        tracks = tracks_obj.item() if getattr(tracks_obj, "shape", None) == () else tracks_obj
    except Exception as exc:  # pragma: no cover - defensive runtime provenance
        report["status"] = "tracks_file_load_failed_detection_support_unavailable"
        report["error"] = repr(exc)
        return support, report
    if not isinstance(tracks, dict):
        report["status"] = "tracks_file_not_dict_detection_support_unavailable"
        report["tracks_object_type"] = str(type(tracks))
        return support, report
    for track_id, records in tracks.items():
        for rec in records:
            report["records_read"] += 1
            try:
                f = int(rec.get("frame", -1))
                if f < 0 or f >= frame_count:
                    continue
                handed_arr = np.asarray(rec.get("det_handedness"), dtype=np.float32).reshape(-1)
                if handed_arr.size == 0:
                    continue
                handed = int(round(float(handed_arr[0])))
                side = "left" if handed == 0 else "right" if handed == 1 else None
                if side is None:
                    continue
                box_arr = np.asarray(rec.get("det_box"), dtype=np.float32).reshape(-1)
                if box_arr.size < 5 or not np.isfinite(box_arr[:5]).all():
                    continue
                current = support[side]["det_box_xyxyscore"][f]
                if support[side]["detected_same_frame"][f] == 0 or float(box_arr[4]) > float(current[4]):
                    support[side]["detected_same_frame"][f] = 1
                    support[side]["det_box_xyxyscore"][f] = box_arr[:5]
                    support[side]["track_id"][f] = str(track_id)
                    report["records_used"] += 1
            except Exception:
                continue
    report["status"] = "ok"
    report["detected_same_frame_counts"] = {side: int(np.count_nonzero(support[side]["detected_same_frame"])) for side in support}
    return support, report


def run(args: argparse.Namespace) -> dict:
    hawor_root = args.hawor_root.resolve()
    video_path_obj = Path(args.video_path).expanduser()
    if not video_path_obj.is_absolute():
        video_path_obj = (Path.cwd() / video_path_obj).resolve()
    checkpoint_path = resolve_path(args.checkpoint, base=hawor_root)
    infiller_path = resolve_path(args.infiller_weight, base=hawor_root)
    model_config_path = resolve_path(args.model_config, base=hawor_root)
    args.video_path = str(video_path_obj)
    args.checkpoint = str(checkpoint_path)
    args.infiller_weight = str(infiller_path)
    args.model_config = str(model_config_path)
    sys.path.insert(0, str(hawor_root))
    os.chdir(hawor_root)

    from demo import hawor_infiller  # type: ignore
    from hawor.utils.process import get_mano_faces, run_mano, run_mano_left  # type: ignore
    from lib.eval_utils.custom_utils import load_slam_cam  # type: ignore
    from scripts.scripts_test_video.detect_track_video import detect_track_video  # type: ignore
    from scripts.scripts_test_video.hawor_slam import hawor_slam  # type: ignore
    from scripts.scripts_test_video.hawor_video import hawor_motion_estimation  # type: ignore

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_sha256 = sha256(video_path_obj) if video_path_obj.exists() and video_path_obj.is_file() else None
    export_provenance = {
        "hawor_root": str(hawor_root),
        "video": file_info(video_path_obj, hash_file=True),
        "checkpoint": file_info(checkpoint_path, hash_file=True),
        "infiller_weight": file_info(infiller_path, hash_file=True),
        "model_config": file_info(model_config_path, hash_file=True),
    }
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    focal_cache_contract = prepare_focal_cache_contract(
        Path(seq_folder),
        int(start_idx),
        int(end_idx),
        args.img_focal,
        force_refresh=bool(args.force_focal_cache_refresh),
    )
    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
    slam_path = Path(seq_folder) / "SLAM" / f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
    if not slam_path.exists():
        hawor_slam(args, start_idx, end_idx)
    if not slam_path.exists():
        raise RuntimeError(f"HaWoR SLAM output missing: {slam_path}")

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(args, start_idx, end_idx, frame_chunks_all)
    _, _, R_c2w, t_c2w = load_slam_cam(str(slam_path))

    faces = np.asarray(get_mano_faces(), dtype=np.int32)
    hands = {}
    hand_to_idx = {"left": 0, "right": 1}
    for side, idx in hand_to_idx.items():
        trans = pred_trans[idx : idx + 1]
        rot = pred_rot[idx : idx + 1]
        pose = pred_hand_pose[idx : idx + 1]
        betas = pred_betas[idx : idx + 1]
        if side == "left":
            mano = run_mano_left(trans, rot, pose, betas=betas)
            hand_faces = faces[:, [0, 2, 1]]
        else:
            mano = run_mano(trans, rot, pose, betas=betas)
            hand_faces = faces
        hands[side] = {
            "vertices_world_m": as_numpy(mano["vertices"][0]).astype(np.float32),
            "joints_world_m": as_numpy(mano["joints"][0]).astype(np.float32),
            "trans_world_m": as_numpy(trans[0]).astype(np.float32),
            "root_orient_axis_angle": as_numpy(rot[0]).astype(np.float32),
            "hand_pose_axis_angle": as_numpy(pose[0]).astype(np.float32),
            "betas": as_numpy(betas[0]).astype(np.float32),
            "valid": as_numpy(pred_valid[idx]).astype(np.uint8),
            "faces": hand_faces.astype(np.int32),
        }

    frame_idx = np.arange(len(imgfiles), dtype=np.int32)
    track_support, track_support_report = load_track_support(Path(seq_folder), int(start_idx), int(end_idx), len(frame_idx))
    out_npz = args.output_dir / "hawor_world_hands.npz"
    np.savez_compressed(
        out_npz,
        frame_idx=frame_idx,
        R_c2w=as_numpy(R_c2w).astype(np.float32),
        t_c2w=as_numpy(t_c2w).astype(np.float32),
        left_vertices_world_m=hands["left"]["vertices_world_m"],
        left_joints_world_m=hands["left"]["joints_world_m"],
        left_trans_world_m=hands["left"]["trans_world_m"],
        left_root_orient_axis_angle=hands["left"]["root_orient_axis_angle"],
        left_hand_pose_axis_angle=hands["left"]["hand_pose_axis_angle"],
        left_betas=hands["left"]["betas"],
        left_valid=hands["left"]["valid"],
        left_detected_same_frame=track_support["left"]["detected_same_frame"],
        left_det_box_xyxyscore=track_support["left"]["det_box_xyxyscore"],
        left_track_id=track_support["left"]["track_id"],
        left_faces=hands["left"]["faces"],
        right_vertices_world_m=hands["right"]["vertices_world_m"],
        right_joints_world_m=hands["right"]["joints_world_m"],
        right_trans_world_m=hands["right"]["trans_world_m"],
        right_root_orient_axis_angle=hands["right"]["root_orient_axis_angle"],
        right_hand_pose_axis_angle=hands["right"]["hand_pose_axis_angle"],
        right_betas=hands["right"]["betas"],
        right_valid=hands["right"]["valid"],
        right_detected_same_frame=track_support["right"]["detected_same_frame"],
        right_det_box_xyxyscore=track_support["right"]["det_box_xyxyscore"],
        right_track_id=track_support["right"]["track_id"],
        right_faces=hands["right"]["faces"],
        img_focal=np.asarray([float(img_focal)], dtype=np.float32),
        video_path=np.asarray([str(args.video_path)]),
        video_sha256=np.asarray([video_sha256 or ""]),
        checkpoint_sha256=np.asarray([export_provenance["checkpoint"].get("sha256") or ""]),
        infiller_weight_sha256=np.asarray([export_provenance["infiller_weight"].get("sha256") or ""]),
        model_config_sha256=np.asarray([export_provenance["model_config"].get("sha256") or ""]),
        seq_folder=np.asarray([str(seq_folder)]),
        track_support_status=np.asarray([track_support_report.get("status", "unknown")]),
        track_support_path=np.asarray([track_support_report.get("tracks_path", "")]),
        focal_cache_contract_path=np.asarray([focal_cache_contract.get("path", "")]),
        focal_cache_contract_status=np.asarray([focal_cache_contract.get("status", "disabled")]),
    )
    valid_counts = {side: int(np.count_nonzero(hands[side]["valid"])) for side in hands}
    detected_counts = {side: int(np.count_nonzero(track_support[side]["detected_same_frame"])) for side in hands}
    qc = {
        "status": "ok",
        "video_path": str(args.video_path),
        "video_sha256": video_sha256,
        "export_provenance": export_provenance,
        "seq_folder": str(seq_folder),
        "output_npz": str(out_npz),
        "frames": int(len(frame_idx)),
        "img_focal": float(img_focal),
        "valid_hand_frames": valid_counts,
        "detected_same_frame_hand_frames": detected_counts,
        "track_support": track_support_report,
        "focal_cache_contract": focal_cache_contract,
        "slam_path": str(slam_path),
    }
    (args.output_dir / "qc_hawor_world_hands.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hawor-root", type=Path, default=Path("/mnt/user-home/yiwen/ego_annotation_remote/hawor_work/third_party/HaWoR"))
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--input_type", type=str, default="file")
    parser.add_argument("--checkpoint", type=str, default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", type=str, default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--model_config", type=str, default="./weights/hawor/model_config.yaml")
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--force-focal-cache-refresh", action="store_true", help="Delete focal-dependent HaWoR motion/mask/SLAM cache artifacts in the sequence folder when their recorded focal differs from --img_focal. Prefer a fresh focal-specific video path when possible.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
