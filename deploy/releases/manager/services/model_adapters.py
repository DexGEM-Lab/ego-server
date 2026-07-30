#!/usr/bin/env python3
"""Real resident model adapters for the four V22 algorithm services."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from services.resident_model_service import PendingRequest


def _manifest_path(payload: dict[str, Any]) -> Path:
    meta = payload.get("video_meta") if isinstance(payload.get("video_meta"), dict) else {}
    raw = meta.get("raw_frame_manifest") or payload.get("raw_frame_manifest")
    if not raw:
        raise RuntimeError("request does not contain a materialized raw_frame_manifest")
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"materialized raw frame manifest missing: {path}")
    return path


def _frames(payload: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = json.loads(_manifest_path(payload).read_text(encoding="utf-8"))
    rows = manifest.get("frames")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("raw frame manifest has no frames")
    return [row for row in rows if isinstance(row, dict)]


def _output_dir(payload: dict[str, Any], default_name: str) -> Path:
    raw = payload.get("output_dir")
    if not raw:
        raise RuntimeError("request output_dir is required")
    path = Path(str(raw)).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _frame_path(row: dict[str, Any]) -> Path:
    raw = row.get("rgb") or row.get("raw_frame_path") or row.get("image_path")
    if not raw:
        raise RuntimeError(f"frame {row.get('frame_idx')} has no materialized RGB path")
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"materialized RGB frame missing: {path}")
    return path


class BaseAdapter:
    model_name = "model"
    native_batch_cap = 1

    def __init__(self) -> None:
        self.last_native_forward_count = 0
        self.last_native_batch_shapes: list[list[int]] = []
        self.last_rows_processed = 0
        self.model_load_count = 0

    def _reset_metrics(self) -> None:
        self.last_native_forward_count = 0
        self.last_native_batch_shapes = []
        self.last_rows_processed = 0


class UniDepthAdapter(BaseAdapter):
    model_name = "unidepth"
    native_batch_cap = 32

    def __init__(self, *, repo: Path, model_dir: Path, device: str = "cuda", native_batch_cap: int = 32) -> None:
        super().__init__()
        self.repo = repo
        self.model_dir = model_dir
        self.device = device
        self.native_batch_cap = int(native_batch_cap)

    def load(self) -> None:
        from scripts.run_v22_resident_unidepth_batch import load_model

        self.model, self.config, self.torch = load_model(self.repo, self.model_dir, None, self.device)
        self.model_load_count = 1

    def compatibility_key(self, payload: dict[str, Any]) -> str:
        meta = payload.get("video_meta") if isinstance(payload.get("video_meta"), dict) else {}
        params = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
        return f"unidepth|{meta.get('width')}x{meta.get('height')}|{params.get('preprocessing_version', 'v1')}|{self.device}"

    def process_batch(self, entries: list[PendingRequest]) -> list[dict[str, Any]]:
        from scripts.run_v22_resident_unidepth_batch import load_image_tensor

        self._reset_metrics()
        per_request: dict[str, dict[str, list[Any]]] = {}
        rows: list[dict[str, Any]] = []
        for entry in entries:
            request_id = entry.request_id
            bucket = per_request.setdefault(request_id, {"entry": entry, "depth": [], "frame_idx": [], "intrinsics": [], "metas": []})
            for frame in _frames(entry.payload):
                rows.append({"request_id": request_id, "frame": frame, "rgb_path": str(_frame_path(frame)), "bucket": bucket})
        self.last_rows_processed = len(rows)
        for start in range(0, len(rows), self.native_batch_cap):
            group = rows[start : start + self.native_batch_cap]
            tensors = []
            metas = []
            for row in group:
                tensor, meta = load_image_tensor({"rgb_path": row["rgb_path"]}, None, self.torch)
                tensors.append(tensor)
                metas.append(meta)
            shapes = {tuple(int(x) for x in tensor.shape) for tensor in tensors}
            if len(shapes) != 1:
                raise RuntimeError(f"UniDepth compatibility bucket received mixed tensor shapes: {sorted(shapes)}")
            batch = self.torch.stack(tensors, dim=0).to(self.device)
            with self.torch.inference_mode():
                predictions = self.model.infer(batch)
            self.last_native_forward_count += 1
            self.last_native_batch_shapes.append([int(x) for x in batch.shape])
            depth = predictions["depth"][:, 0].detach().cpu().numpy()
            intrinsics = predictions.get("intrinsics")
            intr_np = intrinsics.detach().cpu().numpy() if intrinsics is not None else None
            for pos, row in enumerate(group):
                bucket = row["bucket"]
                bucket["depth"].append(depth[pos].astype("float16"))
                bucket["frame_idx"].append(int(row["frame"].get("frame_idx", len(bucket["frame_idx"]))))
                bucket["metas"].append(metas[pos])
                if intr_np is not None:
                    intr = intr_np[pos]
                    bucket["intrinsics"].append([float(intr[0, 0]), float(intr[1, 1]), float(intr[0, 2]), float(intr[1, 2])])
        results = []
        for request_id, bucket in per_request.items():
            entry = bucket["entry"]
            output_dir = _output_dir(entry.payload, "unidepth")
            archive = output_dir / "unidepth_v2_depth.npz"
            payload: dict[str, Any] = {"depth": np.stack(bucket["depth"]).astype("float16"), "frame_idx": np.asarray(bucket["frame_idx"], dtype=np.int32)}
            if bucket["intrinsics"]:
                payload["intrinsics_fx_fy_cx_cy"] = np.asarray(bucket["intrinsics"], dtype=np.float32)
            np.savez_compressed(archive, **payload)
            qc = {"schema": "v22_unidepth_service_qc.v1", "status": "ok", "request_id": request_id, "frame_count": len(bucket["frame_idx"]), "depth_archive": str(archive), "model_load_count": 1, "native_batch_shapes": self.last_native_batch_shapes, "claim_scope": "resident UniDepth frame-batch depth/intrinsics candidate"}
            qc_path = output_dir / "qc_unidepth_v2.json"
            qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
            results.append({"status": "ok", "output_artifacts": {"depth_archive": str(archive), "qc": str(qc_path)}, "frame_count": len(bucket["frame_idx"]), "native_forward_count": self.last_native_forward_count, "native_batch_shapes": self.last_native_batch_shapes})
        return results


class VGGTAdapter(BaseAdapter):
    model_name = "vggt"
    native_batch_cap = 2

    def __init__(self, *, repo_root: Path, checkpoint: Path, device: str = "cuda", native_batch_cap: int = 2, sequence_length: int = 32, target_size: int = 518, patch_multiple: int = 14) -> None:
        super().__init__()
        self.repo_root = repo_root
        self.checkpoint = checkpoint
        self.device = device
        self.native_batch_cap = int(native_batch_cap)
        self.sequence_length = int(sequence_length)
        self.target_size = int(target_size)
        self.patch_multiple = int(patch_multiple)

    def load(self) -> None:
        import torch
        from scripts.run_v22_resident_vggt_camera_batch import VggtCameraBackend

        self.torch = torch
        self.backend = VggtCameraBackend(variant="vggt", repo_root=self.repo_root, checkpoint=self.checkpoint, model_id="facebook/VGGT-1B", model_file="model.pt", device=self.device, target_size=self.target_size)
        self.backend._load(torch)
        self.model_load_count = 1

    def compatibility_key(self, payload: dict[str, Any]) -> str:
        params = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
        contract = params.get("batch_contract") if isinstance(params.get("batch_contract"), dict) else {}
        seq = int(params.get("sequence_length") or contract.get("sequence_length") or self.sequence_length)
        return f"vggt|S={seq}|H={self.target_size}|P={self.patch_multiple}|{self.device}"

    def _sequence_indices(self, frames: list[dict[str, Any]], sequence_length: int) -> list[int]:
        if sequence_length >= len(frames):
            return [int(row.get("frame_idx", i)) for i, row in enumerate(frames)]
        positions = [int(round(i * float(len(frames) - 1) / float(sequence_length - 1))) for i in range(sequence_length)]
        return [int(frames[pos].get("frame_idx", pos)) for pos in positions]

    def process_batch(self, entries: list[PendingRequest]) -> list[dict[str, Any]]:
        from scripts.run_v22_resident_vggt_camera_batch import SequenceRow, load_sequence_tensor, write_item_camera_outputs, video_meta_from_manifest

        self._reset_metrics()
        rows = []
        for entry in entries:
            frames = _frames(entry.payload)
            params = entry.payload.get("parameters") if isinstance(entry.payload.get("parameters"), dict) else {}
            contract = params.get("batch_contract") if isinstance(params.get("batch_contract"), dict) else {}
            seq_len = int(params.get("sequence_length") or contract.get("sequence_length") or self.sequence_length)
            indices = self._sequence_indices(frames, seq_len)
            by_idx = {int(row.get("frame_idx", pos)): row for pos, row in enumerate(frames)}
            picked = [by_idx[idx] for idx in indices]
            manifest = json.loads(_manifest_path(entry.payload).read_text(encoding="utf-8"))
            output_dir = _output_dir(entry.payload, "vggt")
            rows.append(SequenceRow(row_id=f"{entry.request_id}_seq0", job_id=str(entry.payload.get("job_id") or entry.request_id), item_id=str(entry.payload.get("case_id") or entry.request_id), batch_id="service_pending", stage_id=str(entry.payload.get("stage") or "D4_camera_trajectory"), agent_id="api_client", attempt_id="attempt_0001", run_root=Path(str(entry.payload.get("run_root") or entry.request_root)), raw_frame_manifest=_manifest_path(entry.payload), output_dir=output_dir, calibration_contract=None, frames=picked, video_meta=video_meta_from_manifest(manifest, frames), full_source_timeline=len(picked) == len(frames), model_request_path=None))
        results = []
        for start in range(0, len(rows), self.native_batch_cap):
            group = rows[start : start + self.native_batch_cap]
            tensors, metas = [], []
            for row in group:
                sequence, image_metas = load_sequence_tensor(row, self.target_size, self.patch_multiple, self.torch)
                tensors.append(sequence)
                metas.append(image_metas)
            shapes = {tuple(int(x) for x in tensor.shape) for tensor in tensors}
            if len(shapes) != 1:
                raise RuntimeError(f"VGGT compatibility bucket received mixed sequence shapes: {sorted(shapes)}")
            batch = self.torch.stack(tensors, dim=0).to(self.device)
            prediction = self.backend.infer(batch, group, self.torch)
            self.last_native_forward_count += 1
            self.last_native_batch_shapes.append([int(x) for x in batch.shape])
            for pos, row in enumerate(group):
                report = write_item_camera_outputs(row=row, prediction={key: value[pos] for key, value in prediction.items()}, image_meta=metas[pos], backend_name="vggt_resident_batch", backend_version=self.backend.model_version, worker_id=f"vggt_pid_{os.getpid()}", batch_id=f"service_batch_{self.last_native_forward_count:08d}", batch_tensor_shape=[int(x) for x in batch.shape], target_size=self.target_size, patch_multiple=self.patch_multiple, translation_scale=1.0, scale_status="video_derived_uncertain_without_external_metric_anchor")
                results.append({"request_id": next(entry.request_id for entry in entries if entry.payload.get("case_id", entry.request_id) == row.item_id), "status": "ok", "output_artifacts": report})
        self.last_rows_processed = len(rows)
        by_id = {entry.request_id: result for entry, result in zip(entries, results)}
        return [by_id.get(entry.request_id, {"status": "failed", "error": {"code": "missing_vggt_result"}}) for entry in entries]


class WiLoRAdapter(BaseAdapter):
    model_name = "wilor"
    native_batch_cap = 128

    def __init__(self, *, wilor_root: Path, device: str = "cuda", detector_frame_cap: int = 8, crop_batch_cap: int = 128) -> None:
        super().__init__()
        self.wilor_root = wilor_root
        self.device = device
        self.detector_frame_cap = int(detector_frame_cap)
        self.native_batch_cap = int(crop_batch_cap)

    def load(self) -> None:
        from scripts.run_v21_wilor_hand_candidates import load_wilor_backend

        self.model, self.cfg, self.detector, self.device_obj = load_wilor_backend(self.wilor_root)
        self.model_load_count = 1

    def compatibility_key(self, payload: dict[str, Any]) -> str:
        return f"wilor|crop={int(self.cfg.MODEL.IMAGE_SIZE)}|detector_frame_cap={self.detector_frame_cap}|{self.device}"

    def process_batch(self, entries: list[PendingRequest]) -> list[dict[str, Any]]:
        import cv2
        import torch
        from torch.utils.data import ConcatDataset, DataLoader
        from scripts.run_v21_wilor_hand_candidates import project_full_image
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.utils import recursive_to
        from wilor.utils.renderer import cam_crop_to_full

        self._reset_metrics()
        per_request = {}
        frame_rows: dict[tuple[str, int], dict[str, Any]] = {}
        for entry in entries:
            rows = []
            for frame_pos, frame in enumerate(_frames(entry.payload)):
                frame_idx = int(frame.get("frame_idx", frame_pos))
                row = {"frame_idx": frame_idx, "time_s": float(frame.get("time_s", frame_idx)), "raw_hands": []}
                rows.append(row)
                frame_rows[(entry.request_id, frame_idx)] = row
            per_request[entry.request_id] = {"entry": entry, "frames": rows}
        tasks = []
        max_frames = max(len(_frames(entry.payload)) for entry in entries)
        for frame_pos in range(max_frames):
            for entry in entries:
                frames = _frames(entry.payload)
                if frame_pos < len(frames):
                    tasks.append((entry, frames[frame_pos], frame_pos))
        for start in range(0, len(tasks), self.detector_frame_cap):
            frame_group = tasks[start : start + self.detector_frame_cap]
            images = []
            valid_tasks = []
            for entry, frame, frame_pos in frame_group:
                image = cv2.imread(str(_frame_path(frame)))
                if image is not None:
                    images.append(image)
                    valid_tasks.append((entry, frame, frame_pos, image))
            if not images:
                continue
            detector_outputs = self.detector(images, conf=0.3, verbose=False)
            datasets = []
            crop_rows = []
            for det_result, (entry, frame, frame_pos, image) in zip(detector_outputs, valid_tasks):
                boxes, scores, is_right = [], [], []
                for det in det_result:
                    arr = det.boxes.data.cpu().detach().squeeze().numpy()
                    if arr.ndim == 0 or arr.size < 6:
                        continue
                    boxes.append(arr[:4].astype(float).tolist())
                    scores.append(float(arr[4]))
                    is_right.append(float(det.boxes.cls.cpu().detach().squeeze().item()))
                if not boxes:
                    continue
                boxes_np = np.asarray(boxes, dtype=np.float32)
                right_np = np.asarray(is_right, dtype=np.float32)
                dataset = ViTDetDataset(self.cfg, image, boxes_np, right_np, rescale_factor=2.0, fp16=False)
                datasets.append(dataset)
                for crop_id, (score, box, side) in enumerate(zip(scores, boxes, is_right)):
                    frame_idx = int(frame.get("frame_idx", frame_pos))
                    crop_rows.append({"entry": entry, "frame": frame, "frame_pos": frame_pos, "frame_row": frame_rows[(entry.request_id, frame_idx)], "crop_id": crop_id, "score": score, "bbox": box, "side": "right" if side >= 0.5 else "left"})
            if not datasets:
                continue
            loader = DataLoader(ConcatDataset(datasets), batch_size=self.native_batch_cap, shuffle=False, num_workers=0)
            cursor = 0
            for batch in loader:
                batch = recursive_to(batch, self.device_obj)
                with torch.inference_mode():
                    output = self.model(batch)
                self.last_native_forward_count += 1
                self.last_native_batch_shapes.append([int(x) for x in batch["img"].shape])
                pred_cam = output["pred_cam"].clone()
                pred_cam[:, 1] = (2 * batch["right"] - 1) * pred_cam[:, 1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
                cam_t = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal).detach().cpu().numpy()
                for n in range(batch["img"].shape[0]):
                    crop = crop_rows[cursor + n]
                    side = crop["side"]
                    verts = output["pred_vertices"][n].detach().cpu().numpy().astype(float)
                    joints = output["pred_keypoints_3d"][n].detach().cpu().numpy().astype(float)
                    side_sign = 1.0 if side == "right" else -1.0
                    verts[:, 0] = side_sign * verts[:, 0]
                    joints[:, 0] = side_sign * joints[:, 0]
                    focal = float(scaled_focal.detach().cpu().numpy())
                    joints2d = project_full_image(joints, cam_t[n], focal, img_size[n].detach().cpu().numpy())
                    mano_params = {key: value[n].detach().cpu().numpy().astype(float).tolist() for key, value in output["pred_mano_params"].items()}
                    crop["frame_row"]["raw_hands"].append({"backend": "WiLoR", "side": side, "detector_score": crop["score"], "bbox_xyxy": crop["bbox"], "cam_t": cam_t[n].astype(float).tolist(), "focal_length": focal, "joints3d_camera": joints.tolist(), "joints2d": joints2d.astype(float).tolist(), "mano_params": mano_params, "vertices_camera": verts.tolist(), "vertices_camera_sample": verts[::10].tolist(), "filter_status": "measured_raw", "crop_id": crop["crop_id"]})
                cursor += int(batch["img"].shape[0])
        results = []
        for request_id, bucket in per_request.items():
            entry = bucket["entry"]
            frames = sorted(bucket["frames"], key=lambda row: int(row["frame_idx"]))
            output_dir = _output_dir(entry.payload, "wilor")
            raw_path = output_dir / "wilor_raw_hands.json"
            raw_path.write_text(json.dumps({"schema": "v22_wilor_service_candidates.v1", "backend": "WiLoR", "frame_count": len(_frames(entry.payload)), "frames": frames}, indent=2), encoding="utf-8")
            qc_path = output_dir / "wilor_qc.json"
            qc_path.write_text(json.dumps({"schema": "v22_wilor_service_qc.v1", "status": "ok", "frame_count": len(frames), "raw_path": str(raw_path), "model_load_count": 1, "native_batch_shapes": self.last_native_batch_shapes, "detector_batch_frame_cap": self.detector_frame_cap, "claim_scope": "resident WiLoR detector-frame and global crop tensor batches"}, indent=2), encoding="utf-8")
            results.append({"status": "ok", "output_artifacts": {"raw_hands": str(raw_path), "qc": str(qc_path)}, "frame_count": len(frames), "native_forward_count": self.last_native_forward_count, "native_batch_shapes": self.last_native_batch_shapes})
        self.last_rows_processed = sum(len(bucket["frames"]) for bucket in per_request.values())
        return results


class HaWoRAdapter(BaseAdapter):
    """Resident HaWoR temporal model; DROID-SLAM is intentionally absent."""

    model_name = "hawor"
    native_batch_cap = 8

    def __init__(self, *, hawor_root: Path, checkpoint: Path, model_config: Path, mano_root: Path, detector_checkpoint: Path, device: str = "cuda", native_batch_cap: int = 8) -> None:
        super().__init__()
        self.hawor_root = hawor_root
        self.checkpoint = checkpoint
        self.model_config = model_config
        self.mano_root = mano_root
        self.detector_checkpoint = detector_checkpoint
        self.device = device
        self.native_batch_cap = int(native_batch_cap)

    def load(self) -> None:
        sys.path.insert(0, str(self.hawor_root.resolve()))
        os.chdir(self.hawor_root)
        # HaWoR's postprocessing helpers use fixed relative MANO paths for
        # right/left hands. Materialize those runtime assets once at load time.
        mano_right_dir = self.hawor_root / "_DATA" / "data" / "mano"
        mano_left_dir = self.hawor_root / "_DATA" / "data_left" / "mano_left"
        mano_right_dir.mkdir(parents=True, exist_ok=True)
        mano_left_dir.mkdir(parents=True, exist_ok=True)
        for target, source in ((mano_right_dir / "MANO_RIGHT.pkl", self.mano_root / "MANO_RIGHT.pkl"), (mano_left_dir / "MANO_LEFT.pkl", self.mano_root / "MANO_LEFT.pkl")):
            if target.is_symlink() or target.exists():
                target.unlink()
            target.symlink_to(source)
        from hawor.configs import get_config
        from lib.models.hawor import HAWOR
        self.cfg = get_config(str(self.model_config), update_cachedir=False)
        self.cfg.defrost()
        self.cfg.MANO.MODEL_PATH = str(self.mano_root)
        self.cfg.freeze()
        self.model = HAWOR.load_from_checkpoint(str(self.checkpoint), strict=False, cfg=self.cfg)
        self.model = self.model.to(self.device).eval()
        self.torch = __import__("torch")
        self.model_load_count = 1
        # Detector is loaded once as a separate resident stateful stage. It is not batched across videos.
        from ultralytics import YOLO
        self.detector = YOLO(str(self.detector_checkpoint))
        self.detector.to(self.device)

    def compatibility_key(self, payload: dict[str, Any]) -> str:
        params = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
        contract = params.get("batch_contract") if isinstance(params.get("batch_contract"), dict) else {}
        return f"hawor|T={int(contract.get('sequence_length') or 16)}|{self.device}|droid_slam=excluded"

    def process_batch(self, entries: list[PendingRequest]) -> list[dict[str, Any]]:
        # The full stateful HaWoR adapter is deliberately kept explicit: detector/tracking
        # state is item-affine, while the model receives a real [B,16,...] chunk tensor.
        from torch.utils.data import default_collate
        from scripts.run_v22_hawor_no_droid_stage import run_hawor_service_batch

        self._reset_metrics()
        results = run_hawor_service_batch(self, entries, default_collate)
        self.last_native_forward_count = int(results.pop("native_forward_count", 0))
        self.last_native_batch_shapes = list(results.pop("native_batch_shapes", []))
        self.last_rows_processed = int(results.pop("rows_processed", 0))
        return results["results"]
