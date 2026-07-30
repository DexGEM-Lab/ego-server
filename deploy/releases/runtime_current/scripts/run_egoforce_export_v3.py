#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics.astype(float)
    z = np.clip(points[:, 2], 1e-6, None)
    return np.c_[fx * points[:, 0] / z + cx, fy * points[:, 1] / z + cy]


def source_to_world(points: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points), dtype=float)]
    return (T_world_camera @ homog.T).T[:, :3]


def annotation_frame_map(annotations: dict) -> dict[int, dict]:
    frames = annotations.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError("annotations must contain frames list")
    return {int(frame["frame_idx"]): frame for frame in frames}


def observed_hand(frame: dict, side: str) -> dict | None:
    hands = [
        hand
        for hand in frame.get("hands", [])
        if str(hand.get("side", "")).lower() == side and bool(hand.get("measurement_available", False))
    ]
    if not hands:
        return None
    return max(hands, key=lambda hand: float(hand.get("detector_score", 0.0)))


def fallback_intrinsics(frame: dict, explicit_intrinsics: np.ndarray | None) -> np.ndarray:
    if explicit_intrinsics is not None:
        return explicit_intrinsics.astype(float)
    for hand in frame.get("hands", []):
        intr = np.asarray(hand.get("source_intrinsics", []), dtype=float)
        if intr.shape == (4,):
            return intr
    raise RuntimeError("no intrinsics found; pass --intrinsics fx fy cx cy")


def bbox_from_points(points: np.ndarray, width: int, height: int, padding: float) -> list[float]:
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise RuntimeError("invalid 2D points")
    finite = np.isfinite(points).all(axis=1)
    if int(np.count_nonzero(finite)) < 3:
        raise RuntimeError("too few finite 2D points")
    pts = points[finite]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    side = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 1.0)
    side = side * (1.0 + 2.0 * float(padding))
    box = np.r_[center - 0.5 * side, center + 0.5 * side]
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(width - 1))
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(height - 1))
    return [float(v) for v in box]


def hand_data_from_bbox(bbox: list[float], width: int, height: int) -> dict:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    keypoints = np.asarray(
        [
            [x0, y0],
            [x1, y1],
            [0.5 * (x0 + x1), 0.5 * (y0 + y1)],
        ],
        dtype=float,
    )
    valid = x1 > x0 and y1 > y0 and x0 < width and y0 < height and x1 >= 0 and y1 >= 0
    if not valid:
        raise RuntimeError(f"invalid detector bbox {bbox}")
    return {"bbox": np.asarray([x0, y0, x1, y1], dtype=float), "keypoint": keypoints}


def detector_box_is_valid(box: dict, width: int, height: int) -> bool:
    bbox = np.asarray(box.get("bbox", []), dtype=float).reshape(-1)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        return False
    keypoints = np.asarray(box.get("keypoint", []), dtype=float)
    if keypoints.ndim != 2 or keypoints.shape[0] < 3 or keypoints.shape[1] < 2:
        return False
    if not np.isfinite(keypoints[:, :2]).all():
        return False
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return x1 > x0 and y1 > y0 and x0 < width and y0 < height and x1 >= 0.0 and y1 >= 0.0


def sanitize_detector_boxes(boxes: dict, width: int, height: int) -> dict:
    sanitized = {"left": dict(boxes.get("left", {})), "right": dict(boxes.get("right", {}))}
    for side in ("left", "right"):
        side_boxes = sanitized[side]
        for key in ("hand", "arm"):
            if key in side_boxes and not detector_box_is_valid(side_boxes[key], width, height):
                del side_boxes[key]
    return sanitized


def build_box_payload(
    side: str,
    frame: dict,
    width: int,
    height: int,
    use_pseudo_arm_boxes: bool,
) -> dict:
    observed = observed_hand(frame, side)
    if observed is None:
        return {}
    hand_payload = hand_data_from_bbox([float(v) for v in observed["bbox_xyxy"]], width, height)
    payload = {"hand": hand_payload}
    if use_pseudo_arm_boxes:
        joints = np.asarray(observed.get("joints2d_raw", []), dtype=float)
        if joints.shape == (21, 2):
            wrist = joints[0]
            mid = np.median(joints[[5, 9, 13, 17]], axis=0)
            ext = wrist + 1.35 * (wrist - mid)
            arm_points = np.vstack([wrist, ext, mid])
            try:
                payload["arm"] = hand_data_from_bbox(
                    bbox_from_points(arm_points, width, height, padding=0.65),
                    width,
                    height,
                )
            except RuntimeError:
                pass
    return payload


def make_camera_model(egoforce_root: Path, intrinsics: np.ndarray, width: int, height: int):
    module_path = egoforce_root / "camera_models" / "pinhole.py"
    spec = importlib.util.spec_from_file_location("egoforce_pinhole_camera_model", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load EgoForce pinhole camera module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PinholeCameraModel = module.PinholeCameraModel

    return PinholeCameraModel(intrinsics[:2].astype(np.float32), intrinsics[2:].astype(np.float32), width, height)


def egoforce_crop_points_to_full_image(crop_points: torch.Tensor, bbox: torch.Tensor, crop_size: torch.Tensor) -> torch.Tensor:
    inp_w = 224.0
    inp_h = 224.0
    scale = crop_size[:, None, :] / crop_points.new_tensor([inp_w, inp_h]).view(1, 1, 2)
    return crop_points * scale + bbox[:, None, :2]


def pinhole_unit_rays(uv: torch.Tensor, focal: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    xy = (uv - center[:, None, :]) / focal[:, None, :].clamp_min(1e-8)
    rays = torch.cat([xy, torch.ones((*xy.shape[:2], 1), dtype=xy.dtype, device=xy.device)], dim=-1)
    return rays / torch.linalg.norm(rays, dim=-1, keepdim=True).clamp_min(1e-12)


def ray_translation_solve(points: torch.Tensor, rays: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if weights.ndim == 3:
        weights = weights.squeeze(-1)
    weights = weights.clamp_min(0.0)
    eye = torch.eye(3, dtype=points.dtype, device=points.device).view(1, 1, 3, 3)
    d = rays.unsqueeze(-1)
    projectors = eye - d @ d.transpose(-2, -1)
    weighted_projectors = weights[:, :, None, None] * projectors
    lhs = weighted_projectors.sum(dim=1)
    rhs = -(weights[:, :, None] * (projectors @ points.unsqueeze(-1)).squeeze(-1)).sum(dim=1)
    damped = lhs + torch.eye(3, dtype=points.dtype, device=points.device).view(1, 3, 3) * 1e-8
    return torch.linalg.solve(damped, rhs).unsqueeze(1)


def solve_camera_space_pinhole(meta: dict, limb_output, hand_2d: torch.Tensor, arm_2d: torch.Tensor) -> object:
    from types import SimpleNamespace

    device = hand_2d.device
    focal = meta["focal_length"].to(device=device, dtype=hand_2d.dtype)
    center = meta["principal_point"].to(device=device, dtype=hand_2d.dtype)
    hand_bbox = meta["hand_bbox"].to(device=device, dtype=hand_2d.dtype)
    arm_bbox = meta["arm_bbox"].to(device=device, dtype=hand_2d.dtype)
    hand_crop_size = meta["hand_crop_size"].to(device=device, dtype=hand_2d.dtype)
    arm_crop_size = meta["arm_crop_size"].to(device=device, dtype=hand_2d.dtype)

    hand_uv = egoforce_crop_points_to_full_image(hand_2d, hand_bbox, hand_crop_size)
    arm_uv = egoforce_crop_points_to_full_image(arm_2d, arm_bbox, arm_crop_size)
    rays = pinhole_unit_rays(torch.cat([hand_uv, arm_uv], dim=1), focal, center)

    hand_points = limb_output.hand.joints
    arm_points = limb_output.arm.joints
    points = torch.cat([hand_points, arm_points], dim=1)
    weights = torch.cat([limb_output.hand.confidence, limb_output.arm.confidence], dim=1)
    translation = ray_translation_solve(points, rays, weights)
    return SimpleNamespace(
        hand=SimpleNamespace(vertices=limb_output.hand.vertices + translation, joints=hand_points + translation),
        arm=SimpleNamespace(vertices=limb_output.arm.vertices + translation, joints=arm_points + translation),
        transl=translation,
    )


def make_inference(egoforce_root: Path, camera_model, disable_kalman: bool, pose_head_only: bool):
    sys.path.insert(0, str(egoforce_root / "demo"))
    sys.path.insert(0, str(egoforce_root))
    if not pose_head_only:
        from inference import Inference

        inference = Inference(camera_model=camera_model, undistort_inp=False)
        if disable_kalman:
            inference.enable_kalman_filter = False
            inference.left_kalman_filter = None
            inference.right_kalman_filter = None
        return inference

    from demo_hand_arm_loader import DemoHandArmLoader
    from settings import config as cfg
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)
    models_pkg = types.ModuleType("models")
    models_pkg.__path__ = [str(egoforce_root / "models")]
    models_pkg.__package__ = "models"
    sys.modules["models"] = models_pkg
    from models.halo import HALO
    from models.limb_model import LimbModel
    from types import SimpleNamespace

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HALO(cfg)
    model.load_state_dict(torch.load(cfg.POSE_3D.CHECKPOINT_PATH, map_location=device), strict=True)
    model.to(device).eval()
    inference = SimpleNamespace()
    inference.device = device
    inference.model = model
    inference.limb_model = LimbModel(cfg, device=device, use_pose_pca=False, n_components=5)
    inference.camera_model = camera_model
    inference.left_dataset = DemoHandArmLoader(cfg, camera_model, undistort_inp=False, return_complete_image=False, hand_type="left")
    inference.right_dataset = DemoHandArmLoader(cfg, camera_model, undistort_inp=False, return_complete_image=False, hand_type="right")
    inference.undistort_inp = False
    inference.enable_kalman_filter = False
    inference.left_kalman_filter = None
    inference.right_kalman_filter = None
    inference.pose_head_only = True
    return inference


def pose_head_infer(inference, left_data, right_data) -> dict:
    from settings import config as cfg

    device = inference.device
    data_items = [left_data[0], right_data[0]]
    meta_items = [left_data[1], right_data[1]]
    model_data = {
        key: torch.stack([item[key] for item in data_items], dim=0).unsqueeze(1).to(device)
        for key in ("hand_crop", "hand_sparse_kpe", "arm_crop", "arm_sparse_kpe")
    }
    meta = {key: torch.stack([item[key] for item in meta_items], dim=0).to(device) for key in meta_items[0]}
    pred_hand_type = torch.tensor([[0], [1]], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = inference.model(
            model_data["hand_crop"],
            model_data["hand_sparse_kpe"],
            model_data["arm_crop"],
            model_data["arm_sparse_kpe"],
        )

    pred_betas = outputs["betas"].float()
    pred_global_orient = outputs["global_orient"].float()
    pred_hand_pose = outputs["hand_pose"].float()
    pred_kpts_2d = outputs["hand_kpts_2d"].squeeze(1).float()
    pred_arm_kpts_2d = outputs["arm_kpts_2d"].squeeze(1).float()
    pred_hand_kpt_w = outputs["hand_kpt_w"].squeeze(1).float()
    pred_arm_kpt_w = outputs["arm_kpt_w"].squeeze(1).float()
    pred_arm_shape = outputs["arm_shape"].float()
    pred_arm_R = outputs["arm_R"].float()
    zT = torch.zeros(pred_global_orient.shape[0], pred_global_orient.shape[1], 3, device=device)
    batch, time = pred_global_orient.shape[:2]
    limb_output = inference.limb_model(
        pred_betas.reshape(batch * time, *pred_betas.shape[2:]),
        pred_global_orient.reshape(batch * time, *pred_global_orient.shape[2:]),
        pred_hand_pose.reshape(batch * time, *pred_hand_pose.shape[2:]),
        zT.reshape(batch * time, *zT.shape[2:]),
        pred_hand_type.reshape(batch * time),
        pred_arm_shape.reshape(batch * time, *pred_arm_shape.shape[2:]),
        pred_arm_R.reshape(batch * time, *pred_arm_R.shape[2:]),
    )
    limb_output.hand.crop_j2d = pred_kpts_2d
    limb_output.arm.crop_j2d = pred_arm_kpts_2d
    limb_output.hand.confidence = pred_hand_kpt_w
    limb_output.arm.confidence = pred_arm_kpt_w
    cs = solve_camera_space_pinhole(meta, limb_output, pred_kpts_2d, pred_arm_kpts_2d)
    pred_j3d = cs.hand.joints.detach().cpu().numpy()
    pred_vertices = cs.hand.vertices.detach().cpu().numpy()
    pred_arm_j3d = cs.arm.joints.detach().cpu().numpy()
    pred_arm_vertices = cs.arm.vertices.detach().cpu().numpy()
    pred_j2d = inference.camera_model.camera_to_uv(pred_j3d)
    return {
        "pred_j3d": pred_j3d,
        "pred_j2d": pred_j2d,
        "pred_vertices": pred_vertices,
        "pred_arm_j3d": pred_arm_j3d,
        "pred_arm_vertices": pred_arm_vertices,
    }


def infer_arrays(inference, rgb: np.ndarray, frame: dict, width: int, height: int, use_pseudo_arm_boxes: bool) -> dict:
    left_payload = build_box_payload("left", frame, width, height, use_pseudo_arm_boxes)
    right_payload = build_box_payload("right", frame, width, height, use_pseudo_arm_boxes)
    if not left_payload and not right_payload:
        raise RuntimeError("no observed annotation hand boxes for EgoForce crops")
    boxes = {"left": left_payload, "right": right_payload}
    left = inference.left_dataset.transform(rgb, boxes["left"])
    right = inference.right_dataset.transform(rgb, boxes["right"])
    if getattr(inference, "pose_head_only", False):
        out = pose_head_infer(inference, left, right)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        if not hasattr(inference, "undistort_inp"):
            inference.undistort_inp = False
        if not hasattr(inference, "enable_kalman_filter"):
            inference.enable_kalman_filter = False
        from inference import infer
        from settings import config as cfg

        with torch.no_grad():
            out = infer(inference, cfg, inference.model, inference.limb_model, left, right, inference.device)
    return {"out": out, "boxes": boxes}


def infer_arrays_from_detector(inference, rgb: np.ndarray) -> dict:
    height, width = rgb.shape[:2]
    boxes = sanitize_detector_boxes(inference.detect_bounding_boxes(rgb), width, height)
    if "hand" not in boxes.get("left", {}) and "hand" not in boxes.get("right", {}):
        raise RuntimeError("EgoForce detector found no hand boxes")
    left = inference.left_dataset.transform(rgb, boxes["left"])
    right = inference.right_dataset.transform(rgb, boxes["right"])
    from inference import infer
    from settings import config as cfg

    with torch.no_grad():
        out = infer(inference, cfg, inference.model, inference.limb_model, left, right, inference.device)
    return {"out": out, "boxes": boxes}


def make_hand(
    *,
    side: str,
    side_i: int,
    out: dict,
    frame: dict,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    T_world_camera: np.ndarray,
    backend: str,
    detector_box: dict | None,
    crop_source: str,
) -> tuple[dict, dict]:
    joints = np.asarray(out["pred_j3d"][side_i], dtype=float)
    vertices = np.asarray(out["pred_vertices"][side_i], dtype=float)
    arm_joints = np.asarray(out["pred_arm_j3d"][side_i], dtype=float)
    arm_vertices = np.asarray(out["pred_arm_vertices"][side_i], dtype=float)
    if joints.shape != (21, 3) or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError(f"{side} EgoForce output has invalid hand geometry")
    if np.any(joints[:, 2] <= 0.0) or np.any(vertices[:, 2] <= 0.0):
        raise RuntimeError(f"{side} EgoForce output has non-positive camera depth")
    projected = project(joints, intrinsics)
    observed = observed_hand(frame, side)
    if observed is not None and np.asarray(observed.get("joints2d_raw", [])).shape == (21, 2):
        raw2d = np.asarray(observed["joints2d_raw"], dtype=float)
        measurement_available = True
        detector_score = float(observed.get("detector_score", 0.0))
        bbox = observed.get("bbox_xyxy")
    else:
        raw2d = projected
        measurement_available = False
        detector_score = 0.0
        bbox = None
    reproj = np.linalg.norm(projected - raw2d, axis=1)
    joints_world = source_to_world(joints, T_world_camera)
    vertices_world = source_to_world(vertices, T_world_camera)
    cam_t = joints[0].astype(float)
    hand = {
        "backend": backend,
        "side": side,
        "measurement_available": bool(measurement_available),
        "detector_score": detector_score,
        "crop_source": crop_source,
        "filter_status": "egoforce_camera_space",
        "source_intrinsics": intrinsics.astype(float).tolist(),
        "cam_t": cam_t.astype(float).tolist(),
        "joints3d_camera": (joints - cam_t[None, :]).astype(float).tolist(),
        "vertices_camera": (vertices - cam_t[None, :]).astype(float).tolist(),
        "joints3d_source_camera_m": joints.astype(float).tolist(),
        "vertices_source_camera_m": vertices.astype(float).tolist(),
        "joints3d_world_m": joints_world.astype(float).tolist(),
        "vertices_world_m": vertices_world.astype(float).tolist(),
        "joints2d": projected.astype(float).tolist(),
        "joints2d_raw": raw2d.astype(float).tolist(),
        "projection_residual_to_measurement_px": {
            "median": float(np.median(reproj)),
            "p95": float(np.percentile(reproj, 95.0)),
        },
        "egoforce_arm_joints3d_source_camera_m": arm_joints.astype(float).tolist(),
        "egoforce_arm_vertices_source_camera_m": arm_vertices.astype(float).tolist(),
        "world_coordinate_status": "egoforce_camera_space_transformed_by_existing_annotation_camera_pose",
        "mano_surface_status": "full_vertices",
        "mano_vertex_count": int(len(vertices)),
    }
    if bbox is not None:
        hand["bbox_xyxy"] = [float(v) for v in bbox]
    if detector_box is not None and "hand" in detector_box:
        det_hand = detector_box["hand"]
        detector_bbox = np.asarray(det_hand.get("bbox", []), dtype=float).reshape(-1)
        if not detector_box_is_valid(det_hand, image_width, image_height):
            raise RuntimeError(f"{side} EgoForce detector hand box is invalid: {detector_bbox.tolist()}")
        hand["egoforce_detector_bbox_xyxy"] = [float(v) for v in detector_bbox]
        hand["egoforce_detector_score"] = float(det_hand.get("score", 0.0))
        keypoint = np.asarray(det_hand.get("keypoint", []), dtype=float)
        if keypoint.ndim == 2 and keypoint.shape[1] == 2:
            hand["egoforce_detector_keypoints2d"] = keypoint.astype(float).tolist()
    row = {
        "frame_idx": int(frame["frame_idx"]),
        "side": side,
        "measurement_available": bool(measurement_available),
        "detector_score": detector_score,
        "joint_reprojection_px_median": float(np.median(reproj)),
        "joint_reprojection_px_p95": float(np.percentile(reproj, 95.0)),
        "median_camera_depth_m": float(np.median(joints[:, 2])),
        "vertex_count": int(len(vertices)),
        "arm_vertex_count": int(len(arm_vertices)),
        "crop_source": crop_source,
    }
    if detector_box is not None and "hand" in detector_box:
        row["egoforce_detector_score"] = float(detector_box["hand"].get("score", 0.0))
    return hand, row


def summarize(rows: list[dict], key: str) -> dict:
    vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    if not vals:
        return {"count": 0}
    arr = np.asarray(vals, dtype=float)
    return {
        "count": int(len(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run(args: argparse.Namespace) -> dict:
    annotations = load_json(args.annotations)
    frames_by_idx = annotation_frame_map(annotations)
    explicit_intrinsics = None if args.intrinsics is None else np.asarray(args.intrinsics, dtype=float)
    if explicit_intrinsics is not None and explicit_intrinsics.shape != (4,):
        raise RuntimeError("--intrinsics must contain fx fy cx cy")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("video dimensions are invalid")

    first_frame = frames_by_idx.get(args.frame_start)
    if first_frame is None:
        raise RuntimeError(f"annotation frame {args.frame_start} missing")
    intrinsics = fallback_intrinsics(first_frame, explicit_intrinsics)
    camera_model = make_camera_model(args.egoforce_root, intrinsics, width, height)
    pose_head_only = args.crop_source == "annotation_boxes"
    inference = make_inference(args.egoforce_root, camera_model, disable_kalman=args.disable_kalman, pose_head_only=pose_head_only)
    if hasattr(inference, "set_kalman_filter_frequency"):
        inference.set_kalman_filter_frequency(fps)

    output = copy.deepcopy(annotations)
    output_by_idx = annotation_frame_map(output)
    rows: list[dict] = []
    skipped: list[dict] = []
    raw_by_frame: dict[str, dict] = {}
    overlays: list[np.ndarray] = []

    for frame_idx in range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)):
        frame = frames_by_idx.get(frame_idx)
        out_frame = output_by_idx.get(frame_idx)
        if frame is None or out_frame is None:
            skipped.append({"frame_idx": frame_idx, "reason": "missing_annotation"})
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok:
            skipped.append({"frame_idx": frame_idx, "reason": "video_decode_failed"})
            continue
        out_frame["hands"] = []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            if args.crop_source == "egoforce_detector":
                inference_result = infer_arrays_from_detector(inference, rgb.copy())
            else:
                inference_result = infer_arrays(inference, rgb.copy(), frame, width, height, args.use_pseudo_arm_boxes)
            out = inference_result["out"]
            boxes = inference_result["boxes"]
            T = np.asarray(frame["camera"]["T_world_camera_metric"], dtype=float)
            if T.shape != (4, 4):
                raise RuntimeError("invalid T_world_camera_metric")
            new_hands = []
            raw_entry = {}
            for side_i, side in enumerate(("left", "right")):
                detector_side_box = boxes.get(side, {})
                if args.crop_source == "annotation_boxes" and observed_hand(frame, side) is None:
                    skipped.append({"frame_idx": frame_idx, "side": side, "reason": "missing_observed_hand_box"})
                    continue
                if args.crop_source == "egoforce_detector" and "hand" not in boxes.get(side, {}):
                    skipped.append({"frame_idx": frame_idx, "side": side, "reason": "egoforce_detector_missing_hand_box"})
                    continue
                try:
                    hand, row = make_hand(
                        side=side,
                        side_i=side_i,
                        out=out,
                        frame=frame,
                        intrinsics=intrinsics,
                        image_width=width,
                        image_height=height,
                        T_world_camera=T,
                        backend="EgoForce",
                        detector_box=detector_side_box,
                        crop_source=args.crop_source,
                    )
                except Exception as exc:
                    skipped.append({"frame_idx": frame_idx, "side": side, "reason": str(exc)})
                    continue
                new_hands.append(hand)
                rows.append(row)
                raw_entry[side] = {
                    "joints3d_source_camera_m": hand["joints3d_source_camera_m"],
                    "vertices_source_camera_m": hand["vertices_source_camera_m"],
                    "joints2d": hand["joints2d"],
                    "arm_joints3d_source_camera_m": hand["egoforce_arm_joints3d_source_camera_m"],
                    "arm_vertices_source_camera_m": hand["egoforce_arm_vertices_source_camera_m"],
                }
            if not new_hands:
                skipped.append({"frame_idx": frame_idx, "reason": "no_valid_side_output"})
                continue
            out_frame["hands"] = new_hands
            raw_by_frame[str(frame_idx)] = raw_entry
            if args.overlay_video:
                draw = bgr.copy()
                colors = {"left": (0, 180, 255), "right": (255, 120, 40)}
                for hand in new_hands:
                    pts = np.asarray(hand["joints2d"], dtype=float)
                    color = colors[str(hand["side"])]
                    for x, y in pts:
                        cv2.circle(draw, (int(round(x)), int(round(y))), 3, color, -1, cv2.LINE_AA)
                    label_y = 36 if hand["side"] == "left" else 66
                    cv2.putText(draw, f"{hand['side']} EgoForce", (20, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                overlays.append(draw)
        except Exception as exc:
            skipped.append({"frame_idx": frame_idx, "reason": str(exc)})

    cap.release()
    if not rows:
        raise RuntimeError(f"EgoForce produced no usable hand rows; skipped={skipped[:20]}")

    args.output_annotations.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output_annotations, output)
    np.savez_compressed(
        args.output_npz,
        frame_idx=np.asarray([int(k) for k in raw_by_frame.keys()], dtype=int),
        raw_by_frame_json=np.asarray(json.dumps(raw_by_frame), dtype=object),
        intrinsics=intrinsics.astype(float),
        fps=np.asarray(fps, dtype=float),
        image_size=np.asarray([width, height], dtype=int),
    )
    if args.overlay_video and overlays:
        args.overlay_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.overlay_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps / max(1, args.frame_stride),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not create overlay video: {args.overlay_video}")
        for frame in overlays:
            writer.write(frame)
        writer.release()

    report = {
        "status": "ok",
        "annotation_ready": False,
        "diagnostic_only": True,
        "video": str(args.video),
        "annotations": str(args.annotations),
        "egoforce_root": str(args.egoforce_root),
        "output_annotations": str(args.output_annotations),
        "output_npz": str(args.output_npz),
        "overlay_video": None if args.overlay_video is None else str(args.overlay_video),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frame_stride": int(args.frame_stride),
        "crop_source": args.crop_source,
        "frames_requested": int(len(range(args.frame_start, args.frame_end + 1, max(1, args.frame_stride)))),
        "hand_rows": int(len(rows)),
        "skipped_frames": int(len(skipped)),
        "intrinsics": intrinsics.astype(float).tolist(),
        "video_info": {"fps": fps, "width": width, "height": height},
        "summary": {
            "joint_reprojection_px": summarize(rows, "joint_reprojection_px_median"),
            "median_camera_depth_m": summarize(rows, "median_camera_depth_m"),
        },
        "rows_preview": rows[:180],
        "skipped_preview": skipped[:180],
    }
    save_json(args.output_qc, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows_preview", "skipped_preview"}}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--egoforce-root", type=Path, required=True)
    parser.add_argument("--output-annotations", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-qc", type=Path, required=True)
    parser.add_argument("--overlay-video", type=Path)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--intrinsics", nargs=4, type=float)
    parser.add_argument("--disable-kalman", action="store_true")
    parser.add_argument("--crop-source", choices=["egoforce_detector", "annotation_boxes"], default="egoforce_detector")
    parser.add_argument("--use-pseudo-arm-boxes", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
