#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEXYCB_OBJECTS = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}

REQUIRED_DEXYCB_LABEL_KEYS = ("seg", "pose_y", "pose_m", "joint_3d", "joint_2d")
REQUIRED_HO3D_KEYS = (
    "camMat",
    "handPose",
    "handTrans",
    "handBeta",
    "handJoints3D",
    "objRot",
    "objTrans",
    "objCorners3D",
    "objCorners3DRest",
    "objName",
    "objLabel",
)

HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

OBJECT_COLORS = (
    (0, 190, 255),
    (0, 255, 80),
    (255, 160, 0),
    (210, 80, 255),
    (255, 255, 0),
    (80, 220, 220),
)

HAND_COLOR = (255, 80, 80)
CONTACT_COLOR = (30, 30, 255)


class ContractError(RuntimeError):
    pass


def normalize_dataset_name(name: str) -> str:
    lowered = name.strip().lower()
    if lowered in {"ycb", "dexycb", "dex-ycb"}:
        return "dexycb"
    if lowered == "ho3d":
        return "ho3d"
    raise ContractError(f"v20_benchmark_dataset_contract_failed: unsupported_dataset {name}")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_scalar(text: str) -> Any:
    value = text.strip()
    if value == "":
        return ""
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_basic_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_path {path}")
    result: dict[str, Any] = {}
    current_key: str | None = None
    current_container: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent == 0:
            if ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            key = key.strip()
            value = raw_value.strip()
            current_key = key
            current_container = key if value == "" else None
            result[key] = [] if value == "" else parse_scalar(value)
            continue
        stripped = line.strip()
        if stripped.startswith("-") and current_key is not None:
            value = stripped[1:].strip()
            if not isinstance(result.get(current_key), list):
                result[current_key] = []
            result[current_key].append(parse_scalar(value))
            continue
        if current_container and ":" in stripped:
            key, raw_value = stripped.split(":", 1)
            if not isinstance(result.get(current_container), dict):
                result[current_container] = {}
            section = result[current_container]
            section[key.strip()] = parse_scalar(raw_value.strip())
    return result


def ensure_dir_absent_or_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise ContractError(f"run_root_exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def count_glob(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern)))


def selected_frames(total_count: int, frame_start: int, frame_count: int | None) -> list[int]:
    if frame_start < 0:
        raise ContractError(f"frame_start_must_be_nonnegative: {frame_start}")
    if frame_start >= total_count:
        raise ContractError(f"frame_start_out_of_range: start={frame_start} total={total_count}")
    end = total_count if frame_count is None else min(total_count, frame_start + frame_count)
    if end <= frame_start:
        raise ContractError(f"frame_count_must_select_at_least_one_frame: start={frame_start} count={frame_count}")
    return list(range(frame_start, end))


def selected_from_ordered_frame_ids(frame_ids: list[int], frame_start: int, frame_count: int | None) -> list[int]:
    if not frame_ids:
        raise ContractError("no_annotated_frames_available")
    if frame_start < 0:
        raise ContractError(f"frame_start_must_be_nonnegative: {frame_start}")
    start_pos = 0
    while start_pos < len(frame_ids) and frame_ids[start_pos] < frame_start:
        start_pos += 1
    if start_pos >= len(frame_ids):
        raise ContractError(f"frame_start_out_of_annotated_range: start={frame_start} last={frame_ids[-1]}")
    end_pos = len(frame_ids) if frame_count is None else min(len(frame_ids), start_pos + frame_count)
    if end_pos <= start_pos:
        raise ContractError(f"frame_count_must_select_at_least_one_frame: start={frame_start} count={frame_count}")
    return frame_ids[start_pos:end_pos]


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ContractError(f"could_not_read_image: {path}")
    return int(image.shape[1]), int(image.shape[0])


def decode_ho3d_depth_m(path: Path) -> np.ndarray:
    depth_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise ContractError(f"could_not_read_depth: {path}")
    if depth_img.ndim == 2:
        raw = depth_img.astype(np.float32)
    elif depth_img.shape[2] >= 3:
        raw = depth_img[:, :, 2].astype(np.float32) + depth_img[:, :, 1].astype(np.float32) * 256.0
    else:
        raise ContractError(f"ho3d_depth_decode_failed_unexpected_channels: {path} shape={depth_img.shape}")
    return raw * 0.00012498664727900177


def read_depth_stats(path: Path, dataset: str) -> dict[str, Any]:
    if dataset == "ho3d":
        depth = decode_ho3d_depth_m(path)
        valid = depth > 0
        return {
            "path": str(path),
            "decode": "ho3d_lsb_red_msb_green_scale_0.00012498664727900177_m",
            "valid_fraction": float(valid.mean()),
            "valid_min_m": float(depth[valid].min()) if valid.any() else None,
            "valid_max_m": float(depth[valid].max()) if valid.any() else None,
        }
    depth_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise ContractError(f"could_not_read_depth: {path}")
    valid = depth_img > 0
    return {
        "path": str(path),
        "decode": "dexycb_aligned_depth_to_color_uint16_dataset_units_scale_not_revalidated_here",
        "valid_fraction": float(valid.mean()),
        "valid_min_raw": int(depth_img[valid].min()) if valid.any() else None,
        "valid_max_raw": int(depth_img[valid].max()) if valid.any() else None,
    }


def ffprobe_frame_count(path: Path) -> int | None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count if count >= 0 else None


def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.Rodrigues(axis_angle.astype(np.float64).reshape(3, 1))
    return matrix.astype(np.float32)


def project_points(points: np.ndarray, intrinsics: np.ndarray, convention: str) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if convention == "opengl_negative_z":
        depth = -points[:, 2]
        u = fx * points[:, 0] / depth + cx
        v = cy - fy * points[:, 1] / depth
    else:
        depth = points[:, 2]
        u = fx * points[:, 0] / depth + cx
        v = fy * points[:, 1] / depth + cy
    return np.stack([u, v], axis=1).astype(np.float32)


def draw_text_block(image: np.ndarray, lines: list[str], origin: tuple[int, int]) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y += 17


def draw_projected_points(image: np.ndarray, points_2d: np.ndarray, color: tuple[int, int, int], radius: int = 2) -> None:
    height, width = image.shape[:2]
    for point in points_2d:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)


def draw_hand(image: np.ndarray, joints_2d: np.ndarray) -> None:
    height, width = image.shape[:2]
    for a, b in HAND_EDGES:
        if a >= len(joints_2d) or b >= len(joints_2d):
            continue
        pa = joints_2d[a]
        pb = joints_2d[b]
        xa, ya = int(round(float(pa[0]))), int(round(float(pa[1])))
        xb, yb = int(round(float(pb[0]))), int(round(float(pb[1])))
        if 0 <= xa < width and 0 <= ya < height and 0 <= xb < width and 0 <= yb < height:
            cv2.line(image, (xa, ya), (xb, yb), HAND_COLOR, 2, cv2.LINE_AA)
    draw_projected_points(image, joints_2d, (255, 255, 255), 3)
    draw_projected_points(image, joints_2d, HAND_COLOR, 2)


def load_points_xyz(path: Path, max_points: int = 512) -> np.ndarray:
    if not path.exists():
        raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_object_geometry_points {path}")
    points = np.loadtxt(path, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ContractError(f"invalid_object_geometry_points: {path}")
    points = points[:, :3]
    if len(points) > max_points:
        step = max(1, len(points) // max_points)
        points = points[::step][:max_points]
    return points.astype(np.float32)


def default_model_root(dataset_root: Path) -> Path | None:
    candidates = [
        dataset_root / "models",
        dataset_root.parent / "models",
        Path("/mnt/nas/dex-ycb/models"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def model_paths(model_root: Path, object_name: str) -> dict[str, str]:
    root = model_root / object_name
    mesh = root / "textured.obj"
    points = root / "points.xyz"
    if not mesh.exists():
        raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_object_mesh {mesh}")
    if not points.exists():
        raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_object_points {points}")
    return {"model_root": str(root), "mesh_path": str(mesh), "points_path": str(points)}


def intrinsics_from_dex(path: Path) -> np.ndarray:
    parsed = parse_basic_yaml(path)
    color = parsed.get("color")
    if not isinstance(color, dict):
        raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_intrinsics_color_section {path}")
    required = ["fx", "fy", "ppx", "ppy"]
    missing = [key for key in required if key not in color]
    if missing:
        raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_intrinsics_keys {path} {missing}")
    return np.array(
        [
            [float(color["fx"]), 0.0, float(color["ppx"])],
            [0.0, float(color["fy"]), float(color["ppy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def parse_mano_betas(path: Path) -> list[float] | None:
    if not path.exists():
        return None
    parsed = parse_basic_yaml(path)
    betas = parsed.get("betas")
    if isinstance(betas, list):
        return [float(v) for v in betas]
    return None


def validate_dexycb(args: argparse.Namespace, dataset_root: Path, model_root: Path) -> dict[str, Any]:
    sample_parts = Path(args.sample_id).parts
    if len(sample_parts) != 3:
        raise ContractError("v20_benchmark_dataset_contract_failed: dexycb_sample_id_must_be_subject_sequence_camera")
    sequence_root = dataset_root / sample_parts[0] / sample_parts[1]
    camera_root = sequence_root / sample_parts[2]
    meta_path = sequence_root / "meta.yml"
    pose_path = sequence_root / "pose.npz"
    meta = parse_basic_yaml(meta_path)
    for key in ("serials", "num_frames", "ycb_ids", "mano_sides"):
        if key not in meta:
            raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_meta_key {meta_path}:{key}")
    if not pose_path.exists():
        raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_path {pose_path}")
    pose = np.load(pose_path, allow_pickle=False)
    for key in ("pose_m", "pose_y"):
        if key not in pose.files:
            raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_pose_npz_key {pose_path}:{key}")
    if not camera_root.exists():
        raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_path {camera_root}")
    num_frames = int(meta["num_frames"])
    counts = {
        "color": count_glob(camera_root, "color_*.jpg"),
        "depth": count_glob(camera_root, "aligned_depth_to_color_*.png"),
        "labels": count_glob(camera_root, "labels_*.npz"),
    }
    for label, count in counts.items():
        if count != num_frames:
            raise ContractError(f"v20_benchmark_dataset_contract_failed: {label}_count_mismatch count={count} meta_num_frames={num_frames}")
    frames = selected_frames(num_frames, args.frame_start, args.frame_count)
    serial = sample_parts[2]
    intrinsics_path = dataset_root / "calibration" / "intrinsics" / f"{serial}_640x480.yml"
    intrinsics = intrinsics_from_dex(intrinsics_path)
    extrinsics_id = str(meta.get("extrinsics", "")).strip("'")
    extrinsics_path = dataset_root / "calibration" / f"extrinsics_{extrinsics_id}" / "extrinsics.yml" if extrinsics_id else None
    if extrinsics_path is not None and not extrinsics_path.exists():
        raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_extrinsics {extrinsics_path}")
    ycb_ids = [int(v) for v in meta["ycb_ids"]]
    object_records = []
    for index, ycb_id in enumerate(ycb_ids):
        object_name = DEXYCB_OBJECTS.get(ycb_id)
        if object_name is None:
            raise ContractError(f"v20_benchmark_dataset_contract_failed: unknown_dexycb_ycb_id {ycb_id}")
        paths = model_paths(model_root, object_name)
        object_records.append({
            "object_index": index,
            "dataset_label_id": ycb_id,
            "object_name": object_name,
            "object_id": f"object:dexycb_{ycb_id:03d}_{object_name}",
            **paths,
        })
    frame_rows = []
    first_size: tuple[int, int] | None = None
    for frame_idx in frames:
        rgb = camera_root / f"color_{frame_idx:06d}.jpg"
        depth = camera_root / f"aligned_depth_to_color_{frame_idx:06d}.png"
        label = camera_root / f"labels_{frame_idx:06d}.npz"
        for path in (rgb, depth, label):
            if not path.exists():
                raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_path {path}")
        labels = np.load(label, allow_pickle=False)
        for key in REQUIRED_DEXYCB_LABEL_KEYS:
            if key not in labels.files:
                raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_label_npz_key {label}:{key}")
        size = image_size(rgb)
        if first_size is None:
            first_size = size
        elif size != first_size:
            raise ContractError(f"v20_benchmark_dataset_contract_failed: rgb_resolution_mismatch {rgb} {size} expected={first_size}")
        frame_rows.append({
            "frame_index": frame_idx,
            "rgb_path": str(rgb),
            "depth_path": str(depth),
            "gt_label_path": str(label),
            "depth_stats": read_depth_stats(depth, "dexycb") if frame_idx == frames[0] else None,
        })
    mano_calib = meta.get("mano_calib")
    beta_records = []
    if isinstance(mano_calib, list):
        for item in mano_calib:
            mano_path = dataset_root / "calibration" / f"mano_{item}" / "mano.yml"
            beta_records.append({"source": str(mano_path), "betas": parse_mano_betas(mano_path)})
    return {
        "dataset": "dexycb",
        "dataset_aliases": ["ycb", "dexycb", "dex-ycb"],
        "dataset_root": str(dataset_root),
        "sample_id": args.sample_id,
        "sequence_root": str(sequence_root),
        "camera_root": str(camera_root),
        "meta_path": str(meta_path),
        "pose_path": str(pose_path),
        "intrinsics_path": str(intrinsics_path),
        "extrinsics_path": str(extrinsics_path) if extrinsics_path else None,
        "frame_count_total": num_frames,
        "selected_frames": frames,
        "resolution": {"width": first_size[0], "height": first_size[1]} if first_size else None,
        "fps_assumed": float(args.output_fps),
        "camera_intrinsics": intrinsics,
        "coordinate_convention": "dexycb_color_camera_opencv_positive_z_for_projection",
        "depth_semantics": "aligned_depth_to_color_uint16_dataset_units_scale_recorded_not_revalidated",
        "gt_fields_supported": ["joint_3d", "joint_2d", "pose_m", "pose_y", "seg", "ycb_model_geometry"],
        "unsupported_physical_variables": ["contact_gt", "occlusion_owner_gt", "nonpenetration_gt"],
        "mano_sides": meta["mano_sides"],
        "mano_beta_sources": beta_records,
        "objects": object_records,
        "frames": frame_rows,
        "raw_meta": meta,
    }


def validate_ho3d(args: argparse.Namespace, dataset_root: Path, model_root: Path) -> dict[str, Any]:
    sample = args.sample_id.strip("/")
    parts = sample.split("/")
    if len(parts) == 1:
        split = "train"
        sequence = parts[0]
    elif len(parts) == 2:
        split, sequence = parts
    else:
        raise ContractError("v20_benchmark_dataset_contract_failed: ho3d_sample_id_must_be_sequence_or_split_sequence")
    if split != "train":
        raise ContractError(f"v20_benchmark_dataset_contract_failed: unsupported_ho3d_split {split}")
    sequence_root = dataset_root / split / sequence
    rgb_dir = sequence_root / "rgb"
    depth_dir = sequence_root / "depth"
    meta_dir = sequence_root / "meta"
    for path in (rgb_dir, depth_dir, meta_dir):
        if not path.exists():
            raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_path {path}")
    counts = {
        "rgb": count_glob(rgb_dir, "*.jpg"),
        "depth": count_glob(depth_dir, "*.png"),
        "meta": count_glob(meta_dir, "*.pkl"),
    }
    if len(set(counts.values())) != 1:
        raise ContractError(f"v20_benchmark_dataset_contract_failed: ho3d_count_mismatch {counts}")
    total_count = counts["rgb"]
    train_list_path = dataset_root / "train.txt"
    if not train_list_path.exists():
        raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_train_txt {train_list_path}")
    train_entries = set(train_list_path.read_text(encoding="utf-8").splitlines())
    annotated_frame_ids = sorted(
        int(entry.split("/", 1)[1])
        for entry in train_entries
        if entry.startswith(f"{sequence}/") and entry.split("/", 1)[1].isdigit()
    )
    frames = selected_from_ordered_frame_ids(annotated_frame_ids, args.frame_start, args.frame_count)
    frame_rows = []
    first_size: tuple[int, int] | None = None
    first_meta: dict[str, Any] | None = None
    object_names = set()
    for frame_idx in frames:
        frame_id = f"{frame_idx:04d}"
        rgb = rgb_dir / f"{frame_id}.jpg"
        depth = depth_dir / f"{frame_id}.png"
        meta_path = meta_dir / f"{frame_id}.pkl"
        for path in (rgb, depth, meta_path):
            if not path.exists():
                raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_path {path}")
        entry = f"{sequence}/{frame_id}"
        if entry not in train_entries:
            raise ContractError(f"v20_benchmark_dataset_contract_failed: ho3d_frame_not_in_train_txt {entry}")
        with meta_path.open("rb") as handle:
            meta = pickle.load(handle)
        for key in REQUIRED_HO3D_KEYS:
            if key not in meta:
                raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_ho3d_meta_key {meta_path}:{key}")
            if meta[key] is None:
                raise ContractError(f"v20_benchmark_dataset_contract_failed: none_ho3d_meta_value {meta_path}:{key}")
        size = image_size(rgb)
        if first_size is None:
            first_size = size
        elif size != first_size:
            raise ContractError(f"v20_benchmark_dataset_contract_failed: rgb_resolution_mismatch {rgb} {size} expected={first_size}")
        if first_meta is None:
            first_meta = meta
        object_names.add(str(meta["objName"]))
        frame_rows.append({
            "frame_index": frame_idx,
            "frame_id": frame_id,
            "rgb_path": str(rgb),
            "depth_path": str(depth),
            "gt_meta_path": str(meta_path),
            "depth_stats": read_depth_stats(depth, "ho3d") if frame_idx == frames[0] else None,
        })
    object_records = []
    for object_name in sorted(object_names):
        paths = model_paths(model_root, object_name)
        object_records.append({
            "object_name": object_name,
            "object_id": f"object:ho3d_{object_name}",
            **paths,
        })
    intrinsics = np.asarray(first_meta["camMat"], dtype=np.float32)
    return {
        "dataset": "ho3d",
        "dataset_aliases": ["ho3d"],
        "dataset_root": str(dataset_root),
        "sample_id": args.sample_id,
        "split": split,
        "sequence": sequence,
        "sequence_root": str(sequence_root),
        "train_list_path": str(train_list_path),
        "frame_count_total": total_count,
        "annotated_frame_count": len(annotated_frame_ids),
        "selected_frames": frames,
        "resolution": {"width": first_size[0], "height": first_size[1]} if first_size else None,
        "fps_assumed": float(args.output_fps),
        "camera_intrinsics": intrinsics,
        "coordinate_convention": "ho3d_opengl_negative_z_camera",
        "depth_semantics": "rgb_encoded_lsb_red_msb_green_scale_0.00012498664727900177_m",
        "gt_fields_supported": ["handPose", "handTrans", "handBeta", "handJoints3D", "objRot", "objTrans", "objCorners3D", "objCorners3DRest", "objName", "objLabel", "handVertContact", "handVertDist", "handVertIntersec", "handVertObjSurfProj"],
        "unsupported_physical_variables": ["dataset_visibility_labels"],
        "objects": object_records,
        "frames": frame_rows,
    }


def load_dex_frame_state(manifest: dict[str, Any], frame_row: dict[str, Any], model_cache: dict[str, np.ndarray]) -> dict[str, Any]:
    labels = np.load(frame_row["gt_label_path"], allow_pickle=False)
    intrinsics = np.asarray(manifest["camera_intrinsics"], dtype=np.float32)
    hands = []
    joint_3d = labels["joint_3d"]
    joint_2d = labels["joint_2d"]
    pose_m = labels["pose_m"]
    sides = manifest.get("mano_sides") or ["unknown"]
    for idx in range(joint_3d.shape[0]):
        side = str(sides[idx]) if idx < len(sides) else "unknown"
        hands.append({
            "hand_id": f"hand:{side}:{idx}",
            "side": side,
            "joint_3d_camera_m": joint_3d[idx],
            "joint_2d_px": joint_2d[idx],
            "mano_pose_m_dataset": pose_m[idx],
            "visibility": "visible_or_partially_visible_from_dataset_label",
        })
    objects = []
    poses = labels["pose_y"]
    for obj in manifest["objects"]:
        index = int(obj["object_index"])
        transform = np.asarray(poses[index], dtype=np.float32)
        points = model_cache[obj["object_id"]]
        points_camera = points @ transform[:, :3].T + transform[:, 3]
        points_2d = project_points(points_camera, intrinsics, "opencv_positive_z")
        objects.append({
            "object_id": obj["object_id"],
            "object_name": obj["object_name"],
            "dataset_label_id": obj["dataset_label_id"],
            "T_camera_object_3x4": transform,
            "sample_points_camera_m": points_camera,
            "sample_points_2d_px": points_2d,
            "mesh_path": obj["mesh_path"],
            "geometry_points_path": obj["points_path"],
            "visibility": "visible_or_partially_visible_from_dataset_segmentation",
        })
    return {"hands": hands, "objects": objects, "contacts": []}


def load_ho3d_frame_state(manifest: dict[str, Any], frame_row: dict[str, Any], model_cache: dict[str, np.ndarray]) -> dict[str, Any]:
    with Path(frame_row["gt_meta_path"]).open("rb") as handle:
        meta = pickle.load(handle)
    intrinsics = np.asarray(meta["camMat"], dtype=np.float32)
    joints_3d = np.asarray(meta["handJoints3D"], dtype=np.float32)
    joints_2d = project_points(joints_3d, intrinsics, "opengl_negative_z")
    hands = [{
        "hand_id": "hand:right:0",
        "side": "right_or_dataset_unspecified",
        "joint_3d_camera_m": joints_3d,
        "joint_2d_px": joints_2d,
        "mano_pose": np.asarray(meta["handPose"], dtype=np.float32),
        "mano_trans_camera_m": np.asarray(meta["handTrans"], dtype=np.float32),
        "mano_betas": np.asarray(meta["handBeta"], dtype=np.float32),
        "visibility": "visible_or_partially_visible_from_annotated_train_frame",
    }]
    object_name = str(meta["objName"])
    object_id = f"object:ho3d_{object_name}"
    rotation = rodrigues(np.asarray(meta["objRot"], dtype=np.float32))
    translation = np.asarray(meta["objTrans"], dtype=np.float32).reshape(3)
    points = model_cache[object_id]
    points_camera = points @ rotation.T + translation
    points_2d = project_points(points_camera, intrinsics, "opengl_negative_z")
    corners_camera = np.asarray(meta["objCorners3D"], dtype=np.float32)
    corners_2d = project_points(corners_camera, intrinsics, "opengl_negative_z")
    object_records = [{
        "object_id": object_id,
        "object_name": object_name,
        "dataset_label_id": int(meta["objLabel"]),
        "R_camera_object": rotation,
        "t_camera_object_m": translation,
        "sample_points_camera_m": points_camera,
        "sample_points_2d_px": points_2d,
        "corners_camera_m": corners_camera,
        "corners_2d_px": corners_2d,
        "mesh_path": next(obj["mesh_path"] for obj in manifest["objects"] if obj["object_id"] == object_id),
        "geometry_points_path": next(obj["points_path"] for obj in manifest["objects"] if obj["object_id"] == object_id),
        "visibility": "visible_or_partially_visible_from_annotated_train_frame",
    }]
    contacts = []
    contact_mask = meta.get("handVertContact")
    contact_proj = meta.get("handVertObjSurfProj")
    if contact_mask is not None and contact_proj is not None:
        mask = np.asarray(contact_mask).astype(bool)
        proj = np.asarray(contact_proj, dtype=np.float32)
        if mask.any() and proj.ndim == 2 and proj.shape[1] == 3:
            mean_point = proj[mask].mean(axis=0)
            point_2d = project_points(mean_point.reshape(1, 3), intrinsics, "opengl_negative_z")[0]
            contacts.append({
                "frame_idx": frame_row["frame_index"],
                "hand_side": "right_or_dataset_unspecified",
                "object_id": object_id,
                "render_point_camera_m": mean_point,
                "render_point_2d_px": point_2d,
                "render_point_source": "mean_gt_handVertObjSurfProj_for_render_only_oracle_bootstrap",
                "input_contact_mode": "active_physical_contact" if mask.any() else "unresolved",
                "source_surfaces": ["ho3d_handVertContact", "ho3d_handVertObjSurfProj"],
                "uncertainty_radius_m": 0.004,
                "evidence_created": False,
            })
    return {"hands": hands, "objects": object_records, "contacts": contacts}


def draw_world(frame_state: dict[str, Any], width: int, height: int, dataset: str) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)
    center = np.array([width * 0.5, height * 0.72], dtype=np.float32)
    scale = min(width, height) * 0.45
    cv2.line(canvas, (0, int(center[1])), (width, int(center[1])), (80, 80, 80), 1)
    cv2.line(canvas, (int(center[0]), 0), (int(center[0]), height), (80, 80, 80), 1)
    draw_text_block(canvas, ["World/camera metric view", "x horizontal, depth vertical"], (10, 22))
    for hand in frame_state["hands"]:
        points = np.asarray(hand["joint_3d_camera_m"], dtype=np.float32)
        depth = -points[:, 2] if dataset == "ho3d" else points[:, 2]
        xy = np.stack([points[:, 0], depth], axis=1)
        pixels = center + xy * np.array([scale, -scale], dtype=np.float32)
        draw_projected_points(canvas, pixels, HAND_COLOR, 3)
    for obj_index, obj in enumerate(frame_state["objects"]):
        points = np.asarray(obj["sample_points_camera_m"], dtype=np.float32)
        depth = -points[:, 2] if dataset == "ho3d" else points[:, 2]
        xy = np.stack([points[:, 0], depth], axis=1)
        pixels = center + xy * np.array([scale, -scale], dtype=np.float32)
        draw_projected_points(canvas, pixels, OBJECT_COLORS[obj_index % len(OBJECT_COLORS)], 1)
    for contact in frame_state["contacts"]:
        point = np.asarray(contact["render_point_camera_m"], dtype=np.float32).reshape(3)
        depth = -point[2] if dataset == "ho3d" else point[2]
        pixel = center + np.array([point[0], depth], dtype=np.float32) * np.array([scale, -scale], dtype=np.float32)
        draw_projected_points(canvas, pixel.reshape(1, 2), CONTACT_COLOR, 5)
    return canvas


def render_outputs(run_root: Path, manifest: dict[str, Any], frame_states: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    renders_dir = run_root / "renders"
    review_dir = renders_dir / "review_frames"
    renders_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    width = int(manifest["resolution"]["width"])
    height = int(manifest["resolution"]["height"])
    fps = float(args.output_fps)
    overlay_path = renders_dir / "v20_overlay.mp4"
    world_path = renders_dir / "v20_world.mp4"
    side_path = renders_dir / "v20_side_by_side.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))
    world_writer = cv2.VideoWriter(str(world_path), fourcc, fps, (width, height))
    side_writer = cv2.VideoWriter(str(side_path), fourcc, fps, (width * 2, height))
    if not overlay_writer.isOpened() or not world_writer.isOpened() or not side_writer.isOpened():
        raise ContractError("render_writer_open_failed")
    review_indices = {0, max(0, len(frame_states) // 2), max(0, len(frame_states) - 1)}
    for local_index, frame_state in enumerate(frame_states):
        rgb_path = frame_state["rgb_path"]
        overlay = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if overlay is None:
            raise ContractError(f"could_not_read_rgb_for_render: {rgb_path}")
        for obj_index, obj in enumerate(frame_state["objects"]):
            color = OBJECT_COLORS[obj_index % len(OBJECT_COLORS)]
            draw_projected_points(overlay, np.asarray(obj["sample_points_2d_px"], dtype=np.float32), color, 1)
            if "corners_2d_px" in obj:
                draw_projected_points(overlay, np.asarray(obj["corners_2d_px"], dtype=np.float32), color, 4)
        for hand in frame_state["hands"]:
            draw_hand(overlay, np.asarray(hand["joint_2d_px"], dtype=np.float32))
        for contact in frame_state["contacts"]:
            draw_projected_points(overlay, np.asarray(contact["render_point_2d_px"], dtype=np.float32).reshape(1, 2), CONTACT_COLOR, 6)
        draw_text_block(
            overlay,
            [
                "V20 benchmark oracle/bootstrap",
                f"dataset={manifest['dataset']} frame={frame_state['frame_index']}",
                "GT used only to validate harness/evaluation path",
            ],
            (10, 20),
        )
        world = draw_world(frame_state, width, height, manifest["dataset"])
        side = np.concatenate([overlay, world], axis=1)
        overlay_writer.write(overlay)
        world_writer.write(world)
        side_writer.write(side)
        if local_index in review_indices:
            cv2.imwrite(str(review_dir / f"frame_{frame_state['frame_index']:06d}_overlay.jpg"), overlay)
            cv2.imwrite(str(review_dir / f"frame_{frame_state['frame_index']:06d}_world.jpg"), world)
    overlay_writer.release()
    world_writer.release()
    side_writer.release()
    summary = {
        "overlay": str(overlay_path),
        "world": str(world_path),
        "side_by_side": str(side_path),
        "expected_frame_count": len(frame_states),
        "overlay_frame_count": ffprobe_frame_count(overlay_path),
        "world_frame_count": ffprobe_frame_count(world_path),
        "side_by_side_frame_count": ffprobe_frame_count(side_path),
        "render_claim_scope": "GT-driven oracle/bootstrap render for V20 benchmark harness validation; not a non-oracle annotation result.",
    }
    summary["frame_count_match"] = (
        summary["overlay_frame_count"] == len(frame_states)
        and summary["world_frame_count"] == len(frame_states)
        and summary["side_by_side_frame_count"] == len(frame_states)
    )
    if not summary["frame_count_match"]:
        raise ContractError(f"render_frame_count_mismatch: {summary}")
    write_json(renders_dir / "render_summary.json", summary)
    return summary




def sparse_depth_from_frame_states(run_root: Path, manifest: dict[str, Any], frame_states: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    frames: list[int] = []
    us: list[float] = []
    vs: list[float] = []
    depths: list[float] = []
    source_types: list[str] = []
    dataset = manifest["dataset"]
    for frame_state in frame_states:
        frame_index = int(frame_state["frame_index"])
        for hand in frame_state["hands"]:
            joints_2d = np.asarray(hand["joint_2d_px"], dtype=np.float32)
            joints_3d = np.asarray(hand["joint_3d_camera_m"], dtype=np.float32)
            if joints_2d.ndim != 2 or joints_3d.ndim != 2:
                continue
            z = -joints_3d[:, 2] if dataset == "ho3d" else joints_3d[:, 2]
            for point_2d, depth in zip(joints_2d, z):
                if np.isfinite(point_2d).all() and np.isfinite(depth) and depth > 0:
                    frames.append(frame_index)
                    us.append(float(point_2d[0]))
                    vs.append(float(point_2d[1]))
                    depths.append(float(depth))
                    source_types.append("gt_hand_joint")
        for obj in frame_state["objects"]:
            points_2d = np.asarray(obj["sample_points_2d_px"], dtype=np.float32)
            points_3d = np.asarray(obj["sample_points_camera_m"], dtype=np.float32)
            if points_2d.ndim != 2 or points_3d.ndim != 2:
                continue
            z = -points_3d[:, 2] if dataset == "ho3d" else points_3d[:, 2]
            for point_2d, depth in zip(points_2d, z):
                if np.isfinite(point_2d).all() and np.isfinite(depth) and depth > 0:
                    frames.append(frame_index)
                    us.append(float(point_2d[0]))
                    vs.append(float(point_2d[1]))
                    depths.append(float(depth))
                    source_types.append("gt_object_geometry_sample")
    output_dir = run_root / "measurements" / "depth_candidates" / "sparse_gt_metric_anchor"
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "depth_candidate.npz"
    np.savez(
        npz_path,
        frame_indices=np.asarray(frames, dtype=np.int32),
        u_px=np.asarray(us, dtype=np.float32),
        v_px=np.asarray(vs, dtype=np.float32),
        depth_m=np.asarray(depths, dtype=np.float32),
        source_type=np.asarray(source_types),
    )
    report = {
        "candidate_id": candidate_id,
        "method_family": "benchmark_gt_sparse_metric_depth_anchor_oracle",
        "depth_candidate_npz": str(npz_path),
        "point_count": len(frames),
        "frame_count": len(set(frames)),
        "promotion_status": "secondary_scale_anchor_in_oracle_bootstrap",
        "accepted_depth": False,
        "oracle_bootstrap_depth": True,
        "uncertainty": {"non_oracle_uncertainty": "not_estimated"},
    }
    write_json(output_dir / "depth_candidate_report.json", report)
    return report


def build_measurements(run_root: Path, manifest: dict[str, Any], frame_states: list[dict[str, Any]]) -> dict[str, Any]:
    depth_dir = run_root / "measurements" / "depth_candidates"
    geometry_dir = run_root / "measurements" / "geometry_completion"
    hand_shape_dir = run_root / "measurements" / "hand_shape"
    contact_dir = run_root / "measurements" / "contact_visualization"
    depth_modality = {
        "has_native_depth": True,
        "has_rgbd_stream": manifest["dataset"] == "ho3d",
        "has_stereo_pair": False,
        "has_calibration": True,
        "has_camera_trajectory": False,
        "rgb_only": False,
        "source_notes": [manifest["depth_semantics"]],
    }
    depth_candidate_id = f"depth:{manifest['dataset']}:native_dataset_depth"
    sparse_depth_candidate_id = f"depth:{manifest['dataset']}:gt_sparse_metric_anchor"
    native_report_dir = depth_dir / "native_dataset_depth"
    native_report_dir.mkdir(parents=True, exist_ok=True)
    native_report = {
        "candidate_id": depth_candidate_id,
        "method_family": "native_or_rgbd_dataset_depth_measurement",
        "source_dataset": manifest["dataset"],
        "frame_indices": manifest["selected_frames"],
        "depth_paths": [row["depth_path"] for row in manifest["frames"]],
        "calibration": manifest["camera_intrinsics"],
        "depth_semantics": manifest["depth_semantics"],
        "promotion_status": "primary_for_visible_surface_in_benchmark_bootstrap",
        "accepted_depth": False,
        "oracle_bootstrap_depth": True,
        "uncertainty": {"scale_status": "dataset_semantics_recorded_not_reestimated"},
    }
    write_json(native_report_dir / "depth_candidate_report.json", native_report)
    sparse_report = sparse_depth_from_frame_states(run_root, manifest, frame_states, sparse_depth_candidate_id)
    depth_registry = {
        "schema": "v20_depth_candidate_registry.v0",
        "candidates": [native_report, sparse_report],
    }
    depth_selection = {
        "schema": "v20_depth_selection_report.v0",
        "selected": {
            "primary_for_visible_surface": depth_candidate_id,
            "primary_for_hand_depth": depth_candidate_id,
            "secondary_scale_anchor": sparse_depth_candidate_id,
        },
        "residuals_evaluated": ["dataset_contract", "frame_count", "calibration_presence", "depth_decode_presence", "sparse_gt_metric_anchor_self_consistency"],
        "non_oracle_claim_scope": "No learned depth model was run; these are benchmark-provided depth candidates for harness validation.",
    }
    write_json(depth_dir / "depth_modality_report.json", depth_modality)
    write_json(depth_dir / "depth_candidate_registry.json", depth_registry)
    write_json(depth_dir / "depth_selection_report.json", depth_selection)
    geometry_candidates = []
    promoted_geometry_candidate_ids = []
    for obj in manifest["objects"]:
        object_id = obj["object_id"]
        packet = {
            "object_id": object_id,
            "semantic_name": obj["object_name"],
            "physical_branch": "rigid",
            "agent_shape_description": "dataset YCB object mesh used as oracle/bootstrap geometry for benchmark harness validation",
            "visible_evidence": {
                "keyframes": manifest["selected_frames"][: min(5, len(manifest["selected_frames"]))],
                "masks": [],
                "visible_surfaces": [],
                "depth_candidate_ids": [depth_candidate_id],
            },
            "size_scale_hints": {"metric_extent_range_m": None, "supporting_depth_sources": [depth_candidate_id]},
            "occlusion_notes": ["dataset bootstrap does not solve occlusion ownership"],
            "contact_notes": ["contact visualization rows are render-only"],
            "negative_constraints": [
                "must_not_replace_visible_depth_surface",
                "must_not_claim_non_oracle_inference",
                "must_not_enter_contact_evidence_graph",
            ],
        }
        packet_path = geometry_dir / "agent_conditioning_packets" / f"{object_id.replace(':', '_').replace('/', '_')}.json"
        write_json(packet_path, packet)
        candidate_dir = geometry_dir / object_id.replace(":", "_").replace("/", "_") / "dataset_oracle_candidate"
        candidate = {
            "candidate_id": f"geometry:{object_id}:dataset_oracle_mesh",
            "object_id": object_id,
            "method_family": "dataset_gt_ycb_mesh_or_point_sample_for_benchmark_bootstrap",
            "mesh_path": obj["mesh_path"],
            "points_path": obj["points_path"],
            "agent_conditioning_packet": str(packet_path),
            "visible_surface_alignment_transform": "dataset_gt_pose_per_frame",
            "scale_estimate": 1.0,
            "visible_inferred_face_labels": "not_rasterized_in_bootstrap_renderer",
            "residual_summary": {"gt_pose_self_error": 0.0},
            "uncertainty": {"non_oracle_uncertainty": "not_estimated_in_oracle_bootstrap"},
            "accepted_geometry": False,
            "oracle_bootstrap_geometry": True,
            "promotion_status": "oracle_promoted_for_benchmark_bootstrap_only",
        }
        write_json(candidate_dir / "geometry_candidate.json", candidate)
        geometry_candidates.append(candidate)
        promoted_geometry_candidate_ids.append(candidate["candidate_id"])
        conditioned_dir = geometry_dir / object_id.replace(":", "_").replace("/", "_") / "conditioned_generation_candidate"
        conditioned_candidate = {
            "candidate_id": f"geometry:{object_id}:conditioned_generation_missing",
            "object_id": object_id,
            "method_family": "agent_evidence_conditioned_geometry_generation",
            "agent_conditioning_packet": str(packet_path),
            "accepted_geometry": False,
            "oracle_bootstrap_geometry": False,
            "promotion_status": "missing_implementation",
            "blocked_variable": "conditioned_geometry_generation_model_or_adapter",
            "claim_scope": "The V20 harness records the required branch without pretending a conditioned generator ran.",
        }
        write_json(conditioned_dir / "geometry_candidate.json", conditioned_candidate)
        geometry_candidates.append(conditioned_candidate)
    geometry_registry = {
        "schema": "v20_geometry_candidate_registry.v0",
        "candidates": geometry_candidates,
        "candidate_count": len(geometry_candidates),
        "promoted_count": len(promoted_geometry_candidate_ids),
        "promoted_candidate_ids": promoted_geometry_candidate_ids,
        "claim_scope": "Dataset mesh candidates are promoted only for oracle/bootstrap benchmark harness validation; conditioned-generation candidates remain explicit missing implementations.",
    }
    write_json(geometry_dir / "geometry_candidate_registry.json", geometry_registry)
    hand_reports = []
    if manifest["dataset"] == "ho3d":
        first_hand = frame_states[0]["hands"][0]
        hand_reports.append({
            "hand_track_id": first_hand["hand_id"],
            "betas_estimate": first_hand.get("mano_betas"),
            "scale_estimate": 1.0,
            "uncertainty": {"source": "ho3d_gt_handBeta", "non_oracle_uncertainty": "not_estimated"},
            "support_frames": manifest["selected_frames"],
            "occluded_frames_not_constraining_shape": [],
            "residual_summary": {"gt_beta_self_error": 0.0},
            "promotion_status": "oracle_shape_prior_for_benchmark_bootstrap_only",
        })
    else:
        for index, beta_source in enumerate(manifest.get("mano_beta_sources") or []):
            if beta_source.get("betas") is None:
                continue
            hand_reports.append({
                "hand_track_id": f"hand:{manifest.get('mano_sides', ['unknown'])[index] if index < len(manifest.get('mano_sides', [])) else 'unknown'}:{index}",
                "betas_estimate": beta_source["betas"],
                "scale_estimate": 1.0,
                "uncertainty": {"source": beta_source["source"], "non_oracle_uncertainty": "not_estimated"},
                "support_frames": manifest["selected_frames"],
                "occluded_frames_not_constraining_shape": [],
                "residual_summary": {"dataset_calibration_beta_loaded": True},
                "promotion_status": "oracle_shape_prior_for_benchmark_bootstrap_only",
            })
    hand_report = {"schema": "v20_hand_shape_solve_report.v0", "tracks": hand_reports}
    write_json(hand_shape_dir / "hand_shape_solve_report.json", hand_report)
    if hand_reports:
        np.savez(hand_shape_dir / "mano_betas_posterior.npz", **{f"track_{i}": np.asarray(row["betas_estimate"], dtype=np.float32) for i, row in enumerate(hand_reports) if row.get("betas_estimate") is not None})
    contact_rows = []
    for frame_state in frame_states:
        contact_rows.extend(frame_state["contacts"])
    contact_payload = {
        "schema": "v20_contact_point_render_rows.v0",
        "evidence_created": False,
        "rows": contact_rows,
        "claim_scope": "Render-only contact points; they do not enter contact evidence or ownership solving.",
    }
    write_json(contact_dir / "contact_point_render_rows.json", contact_payload)
    observation_bundle = {
        "schema": "v20_observation_bundle.v0",
        "depth": {
            "candidate_registry": str(depth_dir / "depth_candidate_registry.json"),
            "selection_report": str(depth_dir / "depth_selection_report.json"),
            "primary_depth_by_scope": depth_selection["selected"],
        },
        "geometry_candidates": {
            "registry": str(geometry_dir / "geometry_candidate_registry.json"),
            "selected_candidate_ids": promoted_geometry_candidate_ids,
            "selection_scope": "oracle_bootstrap_only",
        },
        "hand_shape": {
            "track_solve_reports": [str(hand_shape_dir / "hand_shape_solve_report.json")],
            "accepted_shape_prior_tracks": [row["hand_track_id"] for row in hand_reports],
            "selection_scope": "oracle_bootstrap_only",
        },
        "contact_visualization": {
            "render_rows": str(contact_dir / "contact_point_render_rows.json"),
            "evidence_created": False,
        },
    }
    write_json(run_root / "state" / "v20_observation_bundle.json", observation_bundle)
    return {
        "depth_modality": depth_modality,
        "depth_registry": depth_registry,
        "depth_selection": depth_selection,
        "geometry_registry": geometry_registry,
        "hand_shape_report": hand_report,
        "contact_rows": contact_payload,
        "observation_bundle": observation_bundle,
    }


def compact_frame_state(frame_state: dict[str, Any]) -> dict[str, Any]:
    hands = []
    for hand in frame_state["hands"]:
        hands.append({
            "hand_id": hand["hand_id"],
            "side": hand["side"],
            "joint_3d_camera_m": hand["joint_3d_camera_m"],
            "joint_2d_px": hand["joint_2d_px"],
            "visibility": hand["visibility"],
        })
    objects = []
    for obj in frame_state["objects"]:
        record = {
            "object_id": obj["object_id"],
            "object_name": obj["object_name"],
            "mesh_path": obj["mesh_path"],
            "geometry_points_path": obj["geometry_points_path"],
            "visibility": obj["visibility"],
        }
        if "T_camera_object_3x4" in obj:
            record["T_camera_object_3x4"] = obj["T_camera_object_3x4"]
        if "R_camera_object" in obj:
            record["R_camera_object"] = obj["R_camera_object"]
            record["t_camera_object_m"] = obj["t_camera_object_m"]
        objects.append(record)
    return {
        "frame_index": frame_state["frame_index"],
        "rgb_path": frame_state["rgb_path"],
        "depth_path": frame_state["depth_path"],
        "hands": hands,
        "objects": objects,
        "contacts": frame_state["contacts"],
    }


def build_state(run_root: Path, manifest: dict[str, Any], frame_states: list[dict[str, Any]], measurements: dict[str, Any], render_summary: dict[str, Any], args: argparse.Namespace) -> None:
    state_dir = run_root / "state"
    frame_count = len(frame_states)
    physical_state = {
        "schema": "v20_physical_state.v0",
        "mode": "v20_benchmark",
        "benchmark_mode_detail": "oracle_bootstrap",
        "run_root": str(run_root),
        "dataset": {
            "name": manifest["dataset"],
            "root": manifest["dataset_root"],
            "sample_id": manifest["sample_id"],
            "gt_loaded": True,
            "gt_use_policy": "GT drives oracle/bootstrap state only; do not report as non-oracle prediction.",
        },
        "timeline": {
            "frame_count": frame_count,
            "source_frame_count": manifest["frame_count_total"],
            "selected_frames": manifest["selected_frames"],
            "fps": manifest["fps_assumed"],
            "duration_s": frame_count / float(manifest["fps_assumed"]),
            "resolution": manifest["resolution"],
        },
        "camera": {
            "state": "dataset_calibrated",
            "intrinsics": manifest["camera_intrinsics"],
            "coordinate_convention": manifest["coordinate_convention"],
            "required_for_metric_claims": True,
        },
        "depth": {
            "candidate_registry": measurements["observation_bundle"]["depth"]["candidate_registry"],
            "selected_observation_bundle": str(state_dir / "v20_observation_bundle.json"),
            "primary_depth_by_scope": measurements["depth_selection"]["selected"],
        },
        "geometry_candidates": {
            "registry": measurements["observation_bundle"]["geometry_candidates"]["registry"],
            "candidate_count": measurements["geometry_registry"]["candidate_count"],
            "promoted_count": measurements["geometry_registry"]["promoted_count"],
            "promotion_scope": "oracle_bootstrap_only",
        },
        "hand_shape": {
            "track_solve_reports": measurements["observation_bundle"]["hand_shape"]["track_solve_reports"],
            "accepted_shape_prior_tracks": measurements["observation_bundle"]["hand_shape"]["accepted_shape_prior_tracks"],
            "promotion_scope": "oracle_bootstrap_only",
        },
        "hands": [compact_frame_state(row)["hands"] for row in frame_states],
        "objects": [compact_frame_state(row)["objects"] for row in frame_states],
        "contacts": [compact_frame_state(row)["contacts"] for row in frame_states],
        "occlusions": [],
        "nonpenetration": [],
        "contact_visualization": {
            "render_rows": measurements["observation_bundle"]["contact_visualization"]["render_rows"],
            "evidence_created": False,
        },
        "renderer_boundary": "renders consume this state directory only",
        "renders": render_summary,
    }
    uncertainty_state = {
        "schema": "v20_uncertainty_state.v0",
        "mode": "v20_benchmark",
        "benchmark_mode_detail": "oracle_bootstrap",
        "non_oracle_inference_uncertainty": "not_measured_by_this_bootstrap_run",
        "unsupported_physical_variables": manifest["unsupported_physical_variables"],
        "occlusion": "not solved unless dataset labels provide compatible evidence",
        "contact": "render-only rows do not create contact evidence",
        "geometry": "dataset mesh candidates are oracle/bootstrap candidates, not generated inference outputs",
    }
    annotations = {
        "schema": "annotations_v20_renderable.v0",
        "mode": "v20_benchmark",
        "benchmark_mode_detail": "oracle_bootstrap",
        "frame_count": frame_count,
        "frames": [compact_frame_state(row) for row in frame_states],
        "render_inputs": {
            "physical_state": str(state_dir / "v20_physical_state.json"),
            "uncertainty_state": str(state_dir / "v20_uncertainty_state.json"),
            "observation_bundle": str(state_dir / "v20_observation_bundle.json"),
        },
    }
    write_json(state_dir / "v20_physical_state.json", physical_state)
    write_json(state_dir / "v20_uncertainty_state.json", uncertainty_state)
    write_json(state_dir / "annotations_v20_renderable.json", annotations)
    evidence = [
        "# V20 Benchmark Oracle/Bootstrap Evidence",
        "",
        f"Dataset: `{manifest['dataset']}`",
        f"Sample: `{manifest['sample_id']}`",
        f"Selected frames: `{manifest['selected_frames'][0]}` to `{manifest['selected_frames'][-1]}` ({frame_count} frames)",
        "",
        "Observation: dataset RGB/depth/calibration/GT contracts passed before render/evaluation.",
        "Interpretation: the V20 benchmark harness can now produce state, render, and GT-evaluation artifacts for this sample.",
        "Commitment: this is an oracle/bootstrap run; GT is not a non-oracle prediction path and must not be compared as inferred annotation quality.",
        "",
        f"Overlay render: `{render_summary['overlay']}`",
        f"World render: `{render_summary['world']}`",
        f"Side-by-side render: `{render_summary['side_by_side']}`",
    ]
    write_text(state_dir / "v20_agent_evidence.md", "\n".join(evidence) + "\n")


def evaluate(run_root: Path, manifest: dict[str, Any], frame_states: list[dict[str, Any]], render_summary: dict[str, Any], max_iterations: int) -> dict[str, Any]:
    evaluation_dir = run_root / "evaluation"
    iteration_dir = evaluation_dir / "iteration_0"
    hand_joint_errors = []
    object_translation_errors = []
    object_rotation_errors = []
    contact_rows = 0
    for frame_state in frame_states:
        for hand in frame_state["hands"]:
            hand_joint_errors.append(0.0)
        for obj in frame_state["objects"]:
            object_translation_errors.append(0.0)
            object_rotation_errors.append(0.0)
        contact_rows += len(frame_state["contacts"])
    metrics = {
        "schema": "v20_gt_metrics.v0",
        "mode": "v20_benchmark",
        "benchmark_mode_detail": "oracle_bootstrap",
        "dataset": manifest["dataset"],
        "sample_id": manifest["sample_id"],
        "frame_count": len(frame_states),
        "hand_joint_camera_m": {
            "supported": bool(hand_joint_errors),
            "mean_error_m": 0.0 if hand_joint_errors else None,
            "max_error_m": 0.0 if hand_joint_errors else None,
            "source": "state copied from GT for oracle/bootstrap harness validation",
        },
        "object_pose_camera": {
            "supported": bool(object_translation_errors),
            "translation_mean_error_m": 0.0 if object_translation_errors else None,
            "rotation_mean_error_rad": 0.0 if object_rotation_errors else None,
            "source": "state copied from GT for oracle/bootstrap harness validation",
        },
        "contact": {
            "supported": manifest["dataset"] == "ho3d" and contact_rows > 0,
            "render_only_rows": contact_rows,
            "metric_scope": "render-only oracle contact visualization when HO3D handVertContact is present; not contact solver evidence",
        },
        "render_frame_count_match": render_summary["frame_count_match"],
        "non_oracle_prediction_metrics_supported": False,
    }
    alignment = {
        "schema": "v20_gt_alignment.v0",
        "dataset": manifest["dataset"],
        "camera_intrinsics": manifest["camera_intrinsics"],
        "coordinate_convention": manifest["coordinate_convention"],
        "hand_joint_alignment": "camera-frame GT copied to oracle/bootstrap state",
        "object_pose_alignment": "dataset camera-frame object transforms copied to oracle/bootstrap state",
        "world_alignment": "world view is a camera-relative visualization; no absolute world-origin metric claimed",
        "depth_alignment": manifest["depth_semantics"],
    }
    failures = {
        "schema": "v20_failure_clusters.v0",
        "clusters": [],
        "unresolved_non_oracle_blocks": [
            "learned MANO candidate generation not run in oracle/bootstrap",
            "object segmentation/tracking not run in oracle/bootstrap",
            "geometry completion not inferred in oracle/bootstrap",
            "contact/occlusion/nonpenetration solver not run in oracle/bootstrap",
        ],
    }
    report = [
        "# V20 Benchmark Evaluation Agent Report",
        "",
        "Observation: dataset GT was loaded, V20 state/renders were produced, and frame-count QC passed.",
        "Interpretation: benchmark harness mechanics are runnable for this sample under oracle/bootstrap mode.",
        "Scoped claim: zero pose/joint error only verifies GT alignment and evaluator semantics; it does not prove non-oracle annotation inference quality.",
        f"Max iterations requested: `{max_iterations}`; actual iterations executed: `1` because no non-oracle intervention is applied in oracle/bootstrap mode.",
    ]
    write_json(iteration_dir / "gt_metrics.json", metrics)
    write_json(iteration_dir / "gt_alignment.json", alignment)
    write_json(iteration_dir / "failure_clusters.json", failures)
    write_text(iteration_dir / "evaluation_agent_report.md", "\n".join(report) + "\n")
    write_json(evaluation_dir / "gt_metrics.json", metrics)
    write_json(evaluation_dir / "gt_alignment.json", alignment)
    append_jsonl(evaluation_dir / "benchmark_iterations.jsonl", {
        "iteration": 0,
        "mode": "oracle_bootstrap",
        "metrics": str(iteration_dir / "gt_metrics.json"),
        "alignment": str(iteration_dir / "gt_alignment.json"),
        "failure_clusters": str(iteration_dir / "failure_clusters.json"),
        "renders": render_summary,
        "controller_decision": "stop_after_oracle_bootstrap_no_non_oracle_intervention",
    })
    append_jsonl(evaluation_dir / "algorithm_parameter_changes.jsonl", {
        "iteration": 0,
        "change": None,
        "reason": "oracle/bootstrap harness validation does not apply algorithm changes to prediction modules",
    })
    final_report = [
        "# V20 Benchmark Final Selection Report",
        "",
        "Final selected result: iteration 0 oracle/bootstrap state and renders.",
        "Selection reason: this run validates dataset loading, V20 state boundary, render production, and GT evaluator semantics without invoking heavy local models.",
        "Non-oracle physical annotation remains a separate measurement/optimization run and must not reuse these GT states as predictions.",
    ]
    write_text(evaluation_dir / "final_selection_report.md", "\n".join(final_report) + "\n")
    return {"metrics": metrics, "alignment": alignment, "failure_clusters": failures}


def build_frame_states(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    model_cache = {}
    for obj in manifest["objects"]:
        model_cache[obj["object_id"]] = load_points_xyz(Path(obj["points_path"]))
    frame_states = []
    for frame_row in manifest["frames"]:
        if manifest["dataset"] == "dexycb":
            state = load_dex_frame_state(manifest, frame_row, model_cache)
        else:
            state = load_ho3d_frame_state(manifest, frame_row, model_cache)
        state["frame_index"] = frame_row["frame_index"]
        state["rgb_path"] = frame_row["rgb_path"]
        state["depth_path"] = frame_row["depth_path"]
        frame_states.append(state)
    return frame_states


def write_manifests(run_root: Path, manifest: dict[str, Any], args: argparse.Namespace) -> None:
    input_manifest = {
        "schema": "v20_input_manifest.v0",
        "mode": "v20_benchmark",
        "benchmark_mode_detail": "oracle_bootstrap",
        "dataset": manifest["dataset"],
        "sample_id": manifest["sample_id"],
        "run_root": str(run_root),
        "frame_count": len(manifest["selected_frames"]),
        "source_frame_count": manifest["frame_count_total"],
        "selected_frames": manifest["selected_frames"],
        "resolution": manifest["resolution"],
        "fps": manifest["fps_assumed"],
        "created_by": "scripts/run_v20_benchmark_oracle_bootstrap.py",
    }
    dataset_manifest = dict(manifest)
    dataset_manifest["schema"] = "v20_dataset_manifest.v0"
    dataset_manifest["mode"] = "v20_benchmark"
    dataset_manifest["benchmark_mode_detail"] = "oracle_bootstrap"
    dataset_manifest["max_benchmark_iterations_requested"] = int(args.max_iterations)
    write_json(run_root / "input" / "input_manifest.json", input_manifest)
    write_json(run_root / "input" / "dataset_manifest.json", dataset_manifest)


def run(args: argparse.Namespace) -> dict[str, Any]:
    raise ContractError(
        "v20_oracle_bootstrap_disabled: GT/reference labels may be used only by evaluation after "
        "prediction-side state/render exist. Use scripts/prepare_v20_benchmark_dataset.py for benchmark setup "
        "and scripts/evaluate_v20_benchmark_gt.py after prediction."
    )
    start = time.perf_counter()
    dataset = normalize_dataset_name(args.dataset)
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_dataset_root {dataset_root}")
    if Path(args.sample_id).exists():
        raise ContractError("v20_benchmark_dataset_contract_failed: sample_list_files_not_supported_by_oracle_bootstrap_tool")
    model_root = args.model_root.resolve() if args.model_root else default_model_root(dataset_root)
    if model_root is None or not model_root.exists():
        raise ContractError("v20_benchmark_dataset_contract_failed: missing_ycb_model_root")
    run_root = args.run_root.resolve()
    ensure_dir_absent_or_overwrite(run_root, args.overwrite)
    try:
        if dataset == "dexycb":
            manifest = validate_dexycb(args, dataset_root, model_root)
        else:
            manifest = validate_ho3d(args, dataset_root, model_root)
        manifest["model_root"] = str(model_root)
        write_manifests(run_root, manifest, args)
        append_jsonl(run_root / "logs" / "harness_events.jsonl", {"event": "dataset_contract_validated", "dataset": dataset, "sample_id": args.sample_id})
        frame_states = build_frame_states(manifest)
        measurements = build_measurements(run_root, manifest, frame_states)
        render_summary = render_outputs(run_root, manifest, frame_states, args)
        build_state(run_root, manifest, frame_states, measurements, render_summary, args)
        evaluation = evaluate(run_root, manifest, frame_states, render_summary, args.max_iterations)
        elapsed = time.perf_counter() - start
        summary = {
            "status": "ok",
            "mode": "v20_benchmark",
            "benchmark_mode_detail": "oracle_bootstrap",
            "dataset": dataset,
            "sample_id": args.sample_id,
            "run_root": str(run_root),
            "frame_count": len(frame_states),
            "renders": render_summary,
            "evaluation": {
                "gt_metrics": str(run_root / "evaluation" / "iteration_0" / "gt_metrics.json"),
                "gt_alignment": str(run_root / "evaluation" / "iteration_0" / "gt_alignment.json"),
                "failure_clusters": str(run_root / "evaluation" / "iteration_0" / "failure_clusters.json"),
            },
            "elapsed_s": elapsed,
            "claim_scope": "Runnable V20 benchmark harness bootstrap; GT-driven oracle state is not non-oracle physical annotation inference.",
        }
        write_json(run_root / "run_summary.json", summary)
        append_jsonl(run_root / "logs" / "harness_events.jsonl", {"event": "run_complete", "summary": summary})
        return summary
    except Exception:
        if args.keep_failed_run_root:
            append_jsonl(run_root / "logs" / "harness_events.jsonl", {"event": "run_failed"})
        else:
            shutil.rmtree(run_root, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deprecated explicit oracle ablation for V20 benchmark harness debugging. Default V20 benchmark must use GT-isolated prepare/evaluate tools.")
    parser.add_argument("--dataset", required=True, help="ycb, dexycb, dex-ycb, or ho3d")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--output-fps", type=float, default=10.0)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-failed-run-root", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        summary = run(parse_args())
    except ContractError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
