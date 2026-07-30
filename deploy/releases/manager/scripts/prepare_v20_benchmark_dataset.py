#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from v20_common import ContractError, write_json

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
REQUIRED_HO3D_REFERENCE_KEYS = (
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
            result[current_container][key.strip()] = parse_scalar(raw_value.strip())
    return result


def normalize_dataset_name(name: str) -> str:
    lowered = name.strip().lower()
    if lowered in {"ycb", "dexycb", "dex-ycb"}:
        return "dexycb"
    if lowered == "ho3d":
        return "ho3d"
    raise ContractError(f"v20_benchmark_dataset_contract_failed: unsupported_dataset {name}")


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ContractError(f"could_not_read_image: {path}")
    return int(image.shape[1]), int(image.shape[0])


def selected_frames(total_count: int, frame_start: int, frame_count: int | None) -> list[int]:
    if frame_start < 0 or frame_start >= total_count:
        raise ContractError(f"frame_start_out_of_range: start={frame_start} total={total_count}")
    end = total_count if frame_count is None else min(total_count, frame_start + frame_count)
    if end <= frame_start:
        raise ContractError(f"frame_count_must_select_at_least_one_frame: start={frame_start} count={frame_count}")
    return list(range(frame_start, end))


def selected_from_ordered_frame_ids(frame_ids: list[int], frame_start: int, frame_count: int | None) -> list[int]:
    if not frame_ids:
        raise ContractError("no_annotated_frames_available")
    start_pos = 0
    while start_pos < len(frame_ids) and frame_ids[start_pos] < frame_start:
        start_pos += 1
    if start_pos >= len(frame_ids):
        raise ContractError(f"frame_start_out_of_annotated_range: start={frame_start} last={frame_ids[-1]}")
    end_pos = len(frame_ids) if frame_count is None else min(len(frame_ids), start_pos + frame_count)
    return frame_ids[start_pos:end_pos]


def count_glob(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern)))


def intrinsics_from_dex(path: Path) -> np.ndarray:
    parsed = parse_basic_yaml(path)
    color = parsed.get("color")
    if not isinstance(color, dict):
        raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_intrinsics_color_section {path}")
    required = ["fx", "fy", "ppx", "ppy"]
    missing = [key for key in required if key not in color]
    if missing:
        raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_intrinsics_keys {path} {missing}")
    return np.array([[float(color["fx"]), 0.0, float(color["ppx"])], [0.0, float(color["fy"]), float(color["ppy"])], [0.0, 0.0, 1.0]], dtype=np.float32)


def default_model_root(dataset_root: Path) -> Path | None:
    for candidate in (dataset_root / "models", dataset_root.parent / "models", Path("/mnt/nas/dex-ycb/models")):
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


def evaluation_ref(path: Path, keys: list[str]) -> dict[str, Any]:
    return {"path": str(path), "keys": keys, "access_policy": "evaluation_only_forbidden_for_prediction_state_render_or_candidate_generation"}


def validate_dexycb(args: argparse.Namespace, dataset_root: Path, model_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    parts = Path(args.sample_id).parts
    if len(parts) != 3:
        raise ContractError("v20_benchmark_dataset_contract_failed: dexycb_sample_id_must_be_subject_sequence_camera")
    sequence_root = dataset_root / parts[0] / parts[1]
    camera_root = sequence_root / parts[2]
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
    num_frames = int(meta["num_frames"])
    for label, count in {"color": count_glob(camera_root, "color_*.jpg"), "depth": count_glob(camera_root, "aligned_depth_to_color_*.png"), "labels": count_glob(camera_root, "labels_*.npz")}.items():
        if count != num_frames:
            raise ContractError(f"v20_benchmark_dataset_contract_failed: {label}_count_mismatch count={count} meta_num_frames={num_frames}")
    frames = selected_frames(num_frames, args.frame_start, args.frame_count)
    intrinsics_path = dataset_root / "calibration" / "intrinsics" / f"{parts[2]}_640x480.yml"
    intrinsics = intrinsics_from_dex(intrinsics_path)
    objects = []
    for index, ycb_id in enumerate(int(v) for v in meta["ycb_ids"]):
        object_name = DEXYCB_OBJECTS.get(ycb_id)
        if object_name is None:
            raise ContractError(f"v20_benchmark_dataset_contract_failed: unknown_dexycb_ycb_id {ycb_id}")
        objects.append({"object_index": index, "dataset_label_id": ycb_id, "object_name": object_name, "object_id": f"object:dexycb_{ycb_id:03d}_{object_name}", **model_paths(model_root, object_name)})
    frame_rows = []
    reference_rows = []
    first_size = None
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
        frame_rows.append({"frame_index": frame_idx, "rgb_path": str(rgb), "depth_path": str(depth)})
        reference_rows.append({"frame_index": frame_idx, "label_npz": evaluation_ref(label, list(REQUIRED_DEXYCB_LABEL_KEYS))})
    public_manifest = {
        "schema": "v20_benchmark_dataset_manifest.v0",
        "mode": "v20_benchmark",
        "benchmark_mode_detail": "prediction_eval_refs_sealed",
        "dataset": "dexycb",
        "dataset_aliases": ["ycb", "dexycb", "dex-ycb"],
        "dataset_root": str(dataset_root),
        "sample_id": args.sample_id,
        "sequence_root": str(sequence_root),
        "camera_root": str(camera_root),
        "meta_path": str(meta_path),
        "intrinsics_path": str(intrinsics_path),
        "eval_ref_public_semantics": "Eval ref paths are withheld from prediction manifest; evaluator manifest lists them separately.",
        "frame_count_total": num_frames,
        "selected_frames": frames,
        "resolution": {"width": first_size[0], "height": first_size[1]} if first_size else None,
        "fps_assumed": float(args.output_fps),
        "camera_intrinsics": intrinsics,
        "coordinate_convention": "dexycb_color_camera_opencv_positive_z_for_projection",
        "depth_semantics": "aligned_depth_to_color_uint16_dataset_units_scale_recorded_not_revalidated",
        "public_object_model_roster": objects,
        "objects": objects,
        "objects_field_semantics": "public_object_model_roster_only_not_prediction_target_list",
        "frames": frame_rows,
        "evaluation_reference_policy": "Eval ref files are excluded from this prediction manifest and may be read only by evaluation.",
    }
    eval_manifest = {
        "schema": "v20_benchmark_evaluation_reference_manifest.v0",
        "dataset": "dexycb",
        "sample_id": args.sample_id,
        "evaluation_reference_policy": "evaluation_only_after_prediction_state_and_renders_exist",
        "pose_npz": evaluation_ref(pose_path, ["pose_m", "pose_y"]),
        "frames": reference_rows,
        "reference_fields_supported": ["joint_3d", "joint_2d", "pose_m", "pose_y", "seg", "ycb_model_geometry"],
    }
    return public_manifest, eval_manifest


def validate_ho3d(args: argparse.Namespace, dataset_root: Path, model_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sample = args.sample_id.strip("/")
    parts = sample.split("/")
    split, sequence = ("train", parts[0]) if len(parts) == 1 else (parts[0], parts[1])
    if split != "train":
        raise ContractError(f"v20_benchmark_dataset_contract_failed: unsupported_ho3d_split {split}")
    sequence_root = dataset_root / split / sequence
    rgb_dir = sequence_root / "rgb"
    depth_dir = sequence_root / "depth"
    meta_dir = sequence_root / "meta"
    for path in (rgb_dir, depth_dir, meta_dir):
        if not path.exists():
            raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_path {path}")
    counts = {"rgb": count_glob(rgb_dir, "*.jpg"), "depth": count_glob(depth_dir, "*.png"), "meta": count_glob(meta_dir, "*.pkl")}
    if len(set(counts.values())) != 1:
        raise ContractError(f"v20_benchmark_dataset_contract_failed: ho3d_count_mismatch {counts}")
    train_list_path = dataset_root / "train.txt"
    if not train_list_path.exists():
        raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_train_txt {train_list_path}")
    train_entries = set(train_list_path.read_text(encoding="utf-8").splitlines())
    annotated = sorted(int(entry.split("/", 1)[1]) for entry in train_entries if entry.startswith(f"{sequence}/") and entry.split("/", 1)[1].isdigit())
    frames = selected_from_ordered_frame_ids(annotated, args.frame_start, args.frame_count)
    frame_rows = []
    reference_rows = []
    first_size = None
    first_meta = None
    object_names = set()
    for frame_idx in frames:
        frame_id = f"{frame_idx:04d}"
        rgb = rgb_dir / f"{frame_id}.jpg"
        depth = depth_dir / f"{frame_id}.png"
        meta_path = meta_dir / f"{frame_id}.pkl"
        for path in (rgb, depth, meta_path):
            if not path.exists():
                raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_path {path}")
        if f"{sequence}/{frame_id}" not in train_entries:
            raise ContractError(f"v20_benchmark_dataset_contract_failed: ho3d_frame_not_in_train_txt {sequence}/{frame_id}")
        with meta_path.open("rb") as handle:
            meta = pickle.load(handle)
        for key in REQUIRED_HO3D_REFERENCE_KEYS:
            if key not in meta or meta[key] is None:
                raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_or_none_ho3d_meta_key {meta_path}:{key}")
        size = image_size(rgb)
        if first_size is None:
            first_size = size
            first_meta = meta
        elif size != first_size:
            raise ContractError(f"v20_benchmark_dataset_contract_failed: rgb_resolution_mismatch {rgb} {size} expected={first_size}")
        object_names.add(str(meta["objName"]))
        frame_rows.append({"frame_index": frame_idx, "frame_id": frame_id, "rgb_path": str(rgb), "depth_path": str(depth)})
        reference_rows.append({"frame_index": frame_idx, "frame_id": frame_id, "meta_pkl": evaluation_ref(meta_path, list(REQUIRED_HO3D_REFERENCE_KEYS))})
    objects = [{"object_name": name, "object_id": f"object:ho3d_{name}", **model_paths(model_root, name)} for name in sorted(object_names)]
    intrinsics = np.asarray(first_meta["camMat"], dtype=np.float32)
    public_manifest = {
        "schema": "v20_benchmark_dataset_manifest.v0",
        "mode": "v20_benchmark",
        "benchmark_mode_detail": "prediction_eval_refs_sealed",
        "dataset": "ho3d",
        "dataset_root": str(dataset_root),
        "sample_id": args.sample_id,
        "split": split,
        "sequence": sequence,
        "sequence_root": str(sequence_root),
        "train_list_path": str(train_list_path),
        "frame_count_total": counts["rgb"],
        "annotated_frame_count": len(annotated),
        "selected_frames": frames,
        "resolution": {"width": first_size[0], "height": first_size[1]} if first_size else None,
        "fps_assumed": float(args.output_fps),
        "camera_intrinsics": intrinsics,
        "coordinate_convention": "ho3d_opengl_negative_z_camera",
        "depth_semantics": "ho3d_rgb_encoded_lsb_red_msb_green_scale_0.00012498664727900177_m",
        "public_object_model_roster": objects,
        "objects": objects,
        "objects_field_semantics": "public_object_model_roster_only_not_prediction_target_list",
        "frames": frame_rows,
        "evaluation_reference_policy": "Eval ref files are excluded from this prediction manifest and may be read only by evaluation.",
    }
    eval_manifest = {
        "schema": "v20_benchmark_evaluation_reference_manifest.v0",
        "dataset": "ho3d",
        "sample_id": args.sample_id,
        "evaluation_reference_policy": "evaluation_only_after_prediction_state_and_renders_exist",
        "frames": reference_rows,
        "reference_fields_supported": ["handPose", "handTrans", "handBeta", "handJoints3D", "objRot", "objTrans", "objCorners3D", "objCorners3DRest", "objName", "objLabel", "handVertContact", "handVertDist", "handVertIntersec", "handVertObjSurfProj"],
    }
    return public_manifest, eval_manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = normalize_dataset_name(args.dataset)
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise ContractError(f"v20_benchmark_dataset_contract_failed: missing_dataset_root {dataset_root}")
    model_root = args.model_root.resolve() if args.model_root else default_model_root(dataset_root)
    if model_root is None or not model_root.exists():
        raise ContractError("v20_benchmark_dataset_contract_failed: missing_ycb_model_root")
    if args.run_root.exists():
        if not args.overwrite:
            raise ContractError(f"run_root_exists: {args.run_root}")
        shutil.rmtree(args.run_root)
    args.run_root.mkdir(parents=True, exist_ok=False)
    if dataset == "dexycb":
        public_manifest, eval_manifest = validate_dexycb(args, dataset_root, model_root)
    else:
        public_manifest, eval_manifest = validate_ho3d(args, dataset_root, model_root)
    input_manifest = {
        "schema": "v20_input_manifest.v0",
        "mode": "v20_benchmark",
        "benchmark_mode_detail": "prediction_eval_refs_sealed",
        "dataset": public_manifest["dataset"],
        "sample_id": args.sample_id,
        "run_root": str(args.run_root.resolve()),
        "frame_count": len(public_manifest["selected_frames"]),
        "source_frame_count": public_manifest["frame_count_total"],
        "selected_frames": public_manifest["selected_frames"],
        "resolution": public_manifest["resolution"],
        "fps": public_manifest["fps_assumed"],
        "created_by": "scripts/prepare_v20_benchmark_dataset.py",
        "evaluation_reference_policy": "Eval refs are absent from prediction input_manifest/dataset_manifest and present only in evaluation/reference_manifest.json.",
    }
    write_json(args.run_root / "input" / "input_manifest.json", input_manifest)
    write_json(args.run_root / "input" / "dataset_manifest.json", public_manifest)
    write_json(args.run_root / "evaluation" / "reference_manifest.json", eval_manifest)
    summary = {
        "status": "ok",
        "mode": "v20_benchmark",
        "benchmark_mode_detail": "prediction_eval_refs_sealed",
        "dataset": public_manifest["dataset"],
        "sample_id": args.sample_id,
        "run_root": str(args.run_root.resolve()),
        "frame_count": len(public_manifest["selected_frames"]),
        "prediction_manifest": str(args.run_root / "input" / "dataset_manifest.json"),
        "evaluation_reference_manifest": str(args.run_root / "evaluation" / "reference_manifest.json"),
        "next_required_step": "run_v20_prediction_measurement_optimization_render_before_reference_evaluation",
    }
    write_json(args.run_root / "run_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare V20 benchmark manifests with strict eval-ref isolation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--output-fps", type=float, default=10.0)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
    except ContractError as exc:
        raise SystemExit(str(exc)) from exc
