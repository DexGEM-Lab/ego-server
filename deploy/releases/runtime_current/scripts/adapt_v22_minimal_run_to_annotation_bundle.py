#!/usr/bin/env python3
"""Adapt a completed minimal V22 run root into an ego.annotation.output bundle.

This adapter does not run perception. It promotes the measurements that already
exist in a V22 minimal run root into the product output contract and records the
missing required modules as explicit errors. Raw WiLoR rows remain candidate
visible-hand evidence; they are not accepted metric MANO state.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

try:
    from ego_annotation.artifacts import ArtifactBundle, status_from_errors, write_json
    from ego_annotation.metrics import build_metric_rows, throughput_forecast
    from ego_annotation.schema import SCHEMA_NAME, SCHEMA_VERSION
except ModuleNotFoundError:
    SCHEMA_NAME = "ego.annotation.output"
    SCHEMA_VERSION = "1.0.0-alpha"

    METRIC_SPECS = {
        "head_camera_ate_translation_m": ("head_camera", "m", ("p50", "p95", "rmse"), "0.005 m"),
        "head_camera_rpe_translation_m": ("head_camera", "m", ("p50", "p95", "rmse"), "0.005 m"),
        "head_camera_rotation_deg": ("head_camera", "deg", ("p50", "p95", "rmse"), "reported toward 5mm equivalent projection effect"),
        "head_camera_scale_error_ratio": ("head_camera", "ratio", ("p50", "p95", "rmse"), "1.0"),
        "hand_wrist_root_error_m": ("hand", "m", ("p50", "p95", "rmse"), "0.005 m"),
        "hand_all_joint_mpjpe_m": ("hand", "m", ("p50", "p95", "rmse"), "0.005 m"),
        "hand_root_relative_mpjpe_m": ("hand", "m", ("p50", "p95", "rmse"), "0.005 m"),
        "hand_mpvpe_surface_m": ("hand", "m", ("p50", "p95", "rmse"), "0.005 m"),
        "hand_reprojection_error_px": ("projection", "px", ("p50", "p95", "rmse"), "minimize with calibrated K"),
        "visibility_state_accuracy": ("visibility", "ratio", ("mean",), "1.0"),
        "temporal_wrist_jitter_m_per_frame": ("temporal", "m/frame", ("p50", "p95", "rmse"), "minimize without dragging off detector evidence"),
        "temporal_root_rotation_jitter_deg_per_frame": ("temporal", "deg/frame", ("p50", "p95", "rmse"), "minimize without hiding source switches"),
        "semantic_segment_duration_s": ("semantic", "s", ("p50", "p95", "coverage"), "mostly 2-3 s segments with full timeline coverage"),
        "semantic_grounding_score": ("semantic", "ratio", ("mean", "p50"), "1.0"),
        "throughput_module_speed_x": ("throughput", "realtime_x", ("p50", "p95", "mean"), "59.5 aggregate realtime per active module for 10k video-hours/week"),
        "throughput_gpu_hours_per_video_hour": ("throughput", "gpu_h/video_h", ("mean", "p95"), "capacity model input"),
        "throughput_queue_wait_s": ("throughput", "s", ("p50", "p95"), "bounded by service SLO"),
        "throughput_batch_fill_ratio": ("throughput", "ratio", ("mean", "p50"), "near saturated under batch load"),
        "throughput_worker_residency_ratio": ("throughput", "ratio", ("mean", "p50"), "resident models amortize load cost"),
        "explicit_failure_rate": ("throughput", "ratio", ("mean",), "0.0 silent failures; explicit failures only"),
    }

    def write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_ndjson(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        flat = []
        fields: list[str] = []
        for row in rows:
            item = {k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            flat.append(item)
            for key in item:
                if key not in fields:
                    fields.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)

    def status_from_errors(errors: list[dict[str, Any]]) -> str:
        severities = {str(row.get("severity")) for row in errors}
        if "error" in severities:
            return "completed_with_errors"
        if severities:
            return "completed_with_degraded_outputs"
        return "ok"

    def _finite_values(values: Any) -> list[float]:
        out = []
        for value in values if isinstance(values, list) else []:
            try:
                f = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                out.append(f)
        return out

    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return float("nan")
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * q / 100.0
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return ordered[int(lo)]
        return ordered[int(lo)] * (hi - pos) + ordered[int(hi)] * (pos - lo)

    def _summarize(values: list[float], summaries: tuple[str, ...]) -> dict[str, Any]:
        result: dict[str, Any] = {"count": len(values)}
        if not values:
            return result
        for summary in summaries:
            if summary == "p50":
                result[summary] = float(median(values))
            elif summary == "p95":
                result[summary] = float(_percentile(values, 95.0))
            elif summary == "mean":
                result[summary] = float(mean(values))
            elif summary == "rmse":
                result[summary] = float(math.sqrt(sum(v * v for v in values) / len(values)))
            elif summary == "coverage":
                result[summary] = float(sum(1 for v in values if v > 0) / max(1, len(values)))
        return result

    def build_metric_rows(observations: dict[str, Any], throughput_rows: list[dict[str, Any]], *, calibration_status: str) -> list[dict[str, Any]]:
        rows = []
        for metric_id, (axis, unit, summaries, ideal_target) in METRIC_SPECS.items():
            direct = observations.get(metric_id, [])
            metadata = {key: value for key, value in direct.items() if key != "values"} if isinstance(direct, dict) else {}
            values = _finite_values(direct.get("values", []) if isinstance(direct, dict) else direct)
            if metric_id == "throughput_module_speed_x":
                values = _finite_values([row.get("module_speed_x") for row in throughput_rows])
            elif metric_id == "explicit_failure_rate":
                values = _finite_values([1.0 if row.get("failed") else 0.0 for row in throughput_rows])
            status = str(metadata.get("status") or ("measured" if values else "unmeasured"))
            row = {"metric_id": metric_id, "axis": axis, "unit": unit, "ideal_target": ideal_target, "status": status, "summary": _summarize(values, summaries)}
            for key in ("measurement_role", "source", "claim_scope", "evaluator_status"):
                if key in metadata:
                    row[key] = metadata[key]
            if axis in {"head_camera", "projection"}:
                row["calibration_status"] = calibration_status
            rows.append(row)
        return rows

    def throughput_forecast(throughput_rows: list[dict[str, Any]], target_video_hours_per_week: float = 10000.0) -> dict[str, Any]:
        speeds = _finite_values([row.get("module_speed_x") for row in throughput_rows])
        target = target_video_hours_per_week / (7.0 * 24.0)
        measured = sum(speeds)
        return {"target_video_hours_per_week": target_video_hours_per_week, "target_realtime_aggregate_x": target, "measured_module_speed_x_sum": measured, "measured_module_speed_x_count": len(speeds), "estimated_active_worker_equivalent_for_target": target / measured if measured > 0 else None, "mean_gpu_hours_per_video_hour": None, "status": "measured" if speeds else "unmeasured"}

    class ArtifactBundle:
        def __init__(self, root: Path, job_id: str) -> None:
            self.root = root.resolve() / job_id
            self.job_id = job_id
            self.tables_dir = self.root / "tables"
            self.events_dir = self.root / "events"
            self.state_dir = self.root / "state"
            self.errors: list[dict[str, Any]] = []
            self.provenance: list[dict[str, Any]] = []

        def add_error(self, code: str, severity: str, message: str, mechanism: str, **extra: Any) -> None:
            self.errors.append({"code": code, "severity": severity, "message": message, "mechanism": mechanism, **extra})

        def add_provenance(self, stage: str, event: str, **extra: Any) -> None:
            self.provenance.append({"stage": stage, "event": event, "time_utc": _utc_now(), **extra})

        def write(self, *, request: dict[str, Any], calibration_contract: dict[str, Any], tables: dict[str, list[dict[str, Any]]], events: dict[str, list[dict[str, Any]]], throughput_forecast: dict[str, Any], status: str, render_artifacts: dict[str, Any] | None = None) -> Path:
            self.tables_dir.mkdir(parents=True, exist_ok=True)
            self.events_dir.mkdir(parents=True, exist_ok=True)
            self.state_dir.mkdir(parents=True, exist_ok=True)
            write_json(self.state_dir / "calibration_contract.json", calibration_contract)
            table_artifacts = {}
            for name in ["frames", "head_camera", "hand_states", "semantic_clips", "validation_metrics"]:
                rows = tables.get(name, [])
                ndjson = self.tables_dir / f"{name}.ndjson"
                csv_path = self.tables_dir / f"{name}.csv"
                _write_ndjson(ndjson, rows)
                _write_csv(csv_path, rows)
                table_artifacts[name] = {"rows": len(rows), "ndjson": str(ndjson), "csv": str(csv_path), "parquet": None, "format_status": "jsonl_csv_written"}
            merged = dict(events)
            merged["errors"] = self.errors
            merged["provenance"] = self.provenance
            stream_artifacts = {}
            for name in ["overlay_events", "caption_events", "provenance", "errors"]:
                path = self.events_dir / f"{name}.ndjson"
                rows = merged.get(name, [])
                _write_ndjson(path, rows)
                stream_artifacts[name] = {"rows": len(rows), "ndjson": str(path)}
            renders = {"optional_qc_demo": [], "note": "renders are projections of numeric state and cannot change numeric results"}
            if render_artifacts:
                renders.update(render_artifacts)
            manifest = {"schema": SCHEMA_NAME, "schema_version": SCHEMA_VERSION, "job_id": self.job_id, "status": status_from_errors(self.errors), "created_utc": _utc_now(), "request": request, "coordinate_frames": ["image_px", "camera_t", "world_w0", "head_t", "mano_left", "mano_right"], "calibration_contract": str(self.state_dir / "calibration_contract.json"), "tables": table_artifacts, "events": stream_artifacts, "renders": renders, "throughput_forecast": throughput_forecast, "errors_count": len(self.errors), "provenance_count": len(self.provenance)}
            manifest_path = self.root / "manifest.json"
            write_json(manifest_path, manifest)
            return manifest_path


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return payload


def resolve_path(run_root: Path, value: Any, fallback: Path) -> Path:
    if value:
        candidate = Path(str(value)).expanduser()
        if candidate.is_absolute():
            return candidate
        return (run_root / candidate).resolve()
    return fallback.resolve()


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def finite_positive(value: Any) -> float | None:
    out = finite_float(value)
    return out if out is not None and out > 0.0 else None


def normalize_calibration(contract: dict[str, Any], errors: list[dict[str, Any]]) -> dict[str, Any]:
    values = contract.get("intrinsics_fx_fy_cx_cy")
    k: list[float] | None = None
    if isinstance(values, list) and len(values) == 4:
        parsed = [finite_positive(v) for v in values]
        if all(v is not None for v in parsed):
            k = [float(v) for v in parsed if v is not None]
    if k is None:
        errors.append(
            {
                "code": "calibration_unresolved",
                "severity": "error",
                "message": "The V22 run did not contain a finite canonical intrinsics_fx_fy_cx_cy contract.",
                "mechanism": "Product consumers require one cited K per clip; silent image-center fallback is forbidden.",
            }
        )
    if not isinstance(contract.get("distortion"), dict):
        errors.append(
            {
                "code": "calibration_distortion_unresolved",
                "severity": "degraded",
                "message": "The V22 UniDepth-derived calibration contract contains K but no measured distortion model.",
                "mechanism": "The product bundle preserves distortion as unresolved instead of assuming zero distortion.",
            }
        )
    diagnostics = contract.get("diagnostics") if isinstance(contract.get("diagnostics"), dict) else {}
    selected_stats = diagnostics.get("selected_stats") if isinstance(diagnostics.get("selected_stats"), dict) else {}
    focal_stats = selected_stats.get("focal_geom") if isinstance(selected_stats.get("focal_geom"), dict) else {}
    uncertainty = {
        "intrinsics_source_spread": {
            "focal_relative_mad_fraction": focal_stats.get("relative_mad_fraction"),
            "focal_relative_p05_p95_fraction": focal_stats.get("relative_p05_p95_fraction"),
            "selected_frame_count": diagnostics.get("selected_frame_count"),
        },
        "scale_gauge": "monocular_depth_intrinsics_support_only_no_metric_head_camera_pose",
        "use_for_metric_error": False,
    }
    return {
        "status": "resolved" if k is not None else "unresolved",
        "intrinsics_fx_fy_cx_cy": k,
        "distortion": contract.get("distortion") if isinstance(contract.get("distortion"), dict) else {"model": "unresolved", "coefficients": [], "status": "not_measured_in_v22_minimal_run"},
        "rectification": contract.get("rectification") if isinstance(contract.get("rectification"), dict) else {"model": "identity", "validity": "unverified"},
        "axis_convention": {
            "image_px": "+u right, +v down, origin top-left",
            "camera_frame": "not_established_by_minimal_calibration_contract",
        },
        "source": str(contract.get("intrinsics_source") or contract.get("method") or "v22_minimal_calibration_contract"),
        "uncertainty": uncertainty,
        "source_contract": contract,
        "contract": "one canonical camera model per clip/session; all consumers cite this product contract",
    }


def frame_rows(raw_manifest: dict[str, Any], input_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    frames = raw_manifest.get("frames") if isinstance(raw_manifest.get("frames"), list) else []
    fingerprint = input_manifest.get("source_fingerprint") if isinstance(input_manifest.get("source_fingerprint"), dict) else {}
    video_uri = input_manifest.get("original_video") or input_manifest.get("primary_video") or raw_manifest.get("clip")
    rows: list[dict[str, Any]] = []
    for pos, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        rows.append(
            {
                "frame_idx": int(frame.get("frame_idx", frame.get("index", pos))),
                "source_frame_idx": int(frame.get("source_frame_idx", frame.get("frame_idx", pos))),
                "time_s": finite_float(frame.get("time_s")),
                "source_time_s": finite_float(frame.get("source_time_s")),
                "rgb_path": frame.get("rgb") or frame.get("raw_frame_path"),
                "video_uri": video_uri,
                "input_sha256": fingerprint.get("sha256"),
                "width": int(frame.get("manifest_width") or frame.get("width") or 0) or None,
                "height": int(frame.get("manifest_height") or frame.get("height") or 0) or None,
                "source_width": int(frame.get("source_width") or 0) or None,
                "source_height": int(frame.get("source_height") or 0) or None,
                "status": "decoded_frame_manifest",
                "coordinate_frame": "image_px",
            }
        )
    return rows


def camera_trajectory_rows(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    stage_path = run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "v22_camera_trajectory_stage.json"
    if not stage_path.exists():
        return [], None
    stage = load_json(stage_path)
    outputs = stage.get("outputs") if isinstance(stage.get("outputs"), dict) else {}
    dense_json = Path(str(outputs.get("dense_json") or run_root / "measurements" / "camera_trajectory" / "droid_full_frame" / "droid_dense_trajectory.json"))
    if stage.get("status") != "ok" or not dense_json.exists():
        return [], stage
    payload = load_json(dense_json)
    frames = payload.get("frames") if isinstance(payload.get("frames"), list) else []
    rows: list[dict[str, Any]] = []
    for row in frames:
        if not isinstance(row, dict):
            continue
        pose = row.get("pose_world_camera_xyzw")
        translation = pose[:3] if isinstance(pose, list) and len(pose) >= 7 else None
        quat = pose[3:7] if isinstance(pose, list) and len(pose) >= 7 else None
        rows.append(
            {
                "frame_idx": int(row.get("frame_idx", len(rows))),
                "valid": True,
                "source": "droid_full_frame_video_tracking",
                "gauge": (stage.get("gauge_declaration") or {}).get("trajectory_frame", "DROID arbitrary world gauge"),
                "scale_status": (stage.get("gauge_declaration") or {}).get("scale_status", "video_derived_uncertain_without_external_metric_anchor"),
                "t_world_camera_m": translation,
                "q_world_camera_xyzw": quat,
                "T_world_camera": row.get("T_world_camera"),
                "calibration_contract": stage.get("calibration_contract"),
                "stage_manifest": str(stage_path),
            }
        )
    return rows, stage


def _as_py(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def hybrid_hand_rows(run_root: Path, residual_warning_px: float) -> tuple[list[dict[str, Any]], dict[str, Any] | None, bool]:
    stage_path = run_root / "state" / "hands_metric" / "v22_hybrid_hand_fusion_stage.json"
    if not stage_path.exists():
        return [], None, False
    stage = load_json(stage_path)
    outputs = stage.get("outputs") if isinstance(stage.get("outputs"), dict) else {}
    npz_path = Path(str(outputs.get("hybrid_npz") or run_root / "state" / "hands_metric" / "v22_hybrid_hands_metric.npz"))
    if stage.get("status") != "ok" or not npz_path.exists():
        return [], stage, False
    blob = np.load(npz_path, allow_pickle=True)
    frame_idx = np.asarray(blob["frame_idx"], dtype=int)
    rows: list[dict[str, Any]] = []
    degraded = False
    for pos, idx in enumerate(frame_idx.tolist()):
        for side in ("left", "right"):
            valid = bool(int(np.asarray(blob[f"{side}_valid"])[pos])) if f"{side}_valid" in blob.files else True
            detected = bool(int(np.asarray(blob[f"{side}_detected_same_frame"])[pos])) if f"{side}_detected_same_frame" in blob.files else False
            source_arr = np.asarray(blob[f"{side}_hybrid_source"]) if f"{side}_hybrid_source" in blob.files else None
            source = str(_as_py(source_arr[pos])) if source_arr is not None else "hawor_metric_mano"
            med = finite_float(np.asarray(blob[f"{side}_wilor_fit_reprojection_median_px"])[pos]) if f"{side}_wilor_fit_reprojection_median_px" in blob.files else None
            p90 = finite_float(np.asarray(blob[f"{side}_wilor_fit_reprojection_p90_px"])[pos]) if f"{side}_wilor_fit_reprojection_p90_px" in blob.files else None
            quality = "candidate"
            if med is not None and med > residual_warning_px:
                quality = "degraded_large_wilor_reprojection_residual"
                degraded = True
            joints = np.asarray(blob[f"{side}_joints_world_m"])[pos].astype(float).tolist() if f"{side}_joints_world_m" in blob.files else None
            vertices = np.asarray(blob[f"{side}_vertices_world_m"])[pos].astype(float)
            rows.append(
                {
                    "frame_idx": int(idx),
                    "side": side,
                    "valid": valid,
                    "visibility": "visible" if detected else "inferred_or_fallback",
                    "source": source,
                    "state_role": "hybrid_metric_hand_candidate",
                    "accepted_metric_mano": quality == "candidate",
                    "quality_status": quality,
                    "wrist_t_world_m": joints[0] if isinstance(joints, list) and joints else None,
                    "joints_world_m": joints,
                    "mano_vertex_count": int(vertices.shape[0]) if vertices.ndim == 2 else None,
                    "root_orient_axis_angle": np.asarray(blob[f"{side}_root_orient_axis_angle"])[pos].astype(float).tolist() if f"{side}_root_orient_axis_angle" in blob.files else None,
                    "hand_pose_axis_angle": np.asarray(blob[f"{side}_hand_pose_axis_angle"])[pos].astype(float).tolist() if f"{side}_hand_pose_axis_angle" in blob.files else None,
                    "betas": np.asarray(blob[f"{side}_betas"])[pos].astype(float).tolist() if f"{side}_betas" in blob.files else None,
                    "wilor_fit_reprojection_median_px": med,
                    "wilor_fit_reprojection_p90_px": p90,
                    "stage_manifest": str(stage_path),
                    "hybrid_npz": str(npz_path),
                }
            )
    return rows, stage, degraded


def hand_candidate_rows(raw_manifest: dict[str, Any], wilor_raw: dict[str, Any], wilor_path: Path) -> list[dict[str, Any]]:
    raw_frames = raw_manifest.get("frames") if isinstance(raw_manifest.get("frames"), list) else []
    wilor_frames = wilor_raw.get("frames") if isinstance(wilor_raw.get("frames"), list) else []
    by_frame: dict[int, dict[str, Any]] = {}
    for row in wilor_frames:
        if isinstance(row, dict) and row.get("frame_idx") is not None:
            by_frame[int(row["frame_idx"])] = row
    rows: list[dict[str, Any]] = []
    for pos, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, dict):
            continue
        frame_idx = int(raw_frame.get("frame_idx", raw_frame.get("index", pos)))
        wframe = by_frame.get(frame_idx, {})
        hands = wframe.get("raw_hands") if isinstance(wframe.get("raw_hands"), list) else []
        if not hands:
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "time_s": finite_float(raw_frame.get("time_s")),
                    "side": "unknown",
                    "visibility": "not_detected",
                    "source": "wilor_v21_raw_candidates",
                    "state_role": "presence_absence_evidence_not_metric_mano",
                    "accepted_metric_mano": False,
                    "raw_source_path": str(wilor_path),
                    "candidate_count_in_frame": 0,
                }
            )
            continue
        for hand_idx, hand in enumerate(hands):
            if not isinstance(hand, dict):
                continue
            joints2d = hand.get("joints2d") or hand.get("joints2d_raw")
            joints3d = hand.get("joints3d_camera")
            wrist_candidate = None
            if isinstance(joints3d, list) and joints3d and isinstance(joints3d[0], list):
                wrist_candidate = joints3d[0]
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "time_s": finite_float(raw_frame.get("time_s")),
                    "side": str(hand.get("side") or "unknown"),
                    "visibility": "visible",
                    "source": "wilor_v21_raw_candidates",
                    "state_role": "visible_geometry_candidate_not_metric_mano",
                    "accepted_metric_mano": False,
                    "detector_score": finite_float(hand.get("detector_score")),
                    "bbox_xyxy": hand.get("bbox_xyxy") or hand.get("bbox") or hand.get("box"),
                    "crop_metadata": hand.get("crop") or hand.get("crop_metadata") or hand.get("crop_box"),
                    "joints2d_px": joints2d,
                    "wrist_t_camera_m_candidate": wrist_candidate,
                    "cam_t_candidate": hand.get("cam_t"),
                    "raw_source_path": str(wilor_path),
                    "raw_candidate_index": hand_idx,
                    "candidate_id": f"wilor:{frame_idx}:{hand_idx}",
                    "candidate_count_in_frame": len(hands),
                }
            )
    return rows


def semantic_rows_from_actions(actions_json: Path | None, raw_manifest: dict[str, Any], max_clip_s: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if actions_json is None:
        return [], []
    payload = load_json(actions_json)
    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []
    actions = tasks[0].get("actions") if tasks and isinstance(tasks[0], dict) and isinstance(tasks[0].get("actions"), list) else payload.get("actions")
    if not isinstance(actions, list):
        raise RuntimeError(f"{actions_json} lacks tasks[0].actions or actions")
    fps = finite_positive(raw_manifest.get("fps"))
    if fps is None:
        video = raw_manifest.get("video") if isinstance(raw_manifest.get("video"), dict) else {}
        fps = finite_positive(video.get("fps"))
    if fps is None:
        raise RuntimeError("raw frame manifest lacks fps; cannot align action captions")
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for action_idx, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        start_frame = int(action.get("start_frame", action.get("start", -1)))
        end_frame = int(action.get("end_frame", action.get("end", -1)))
        caption = str(action.get("description") or action.get("action") or "").strip()
        if start_frame < 0 or end_frame <= start_frame or not caption:
            continue
        segment_frames = max(1, int(round(max_clip_s * fps)))
        clip_start = start_frame
        part = 0
        while clip_start < end_frame:
            clip_end = min(end_frame, clip_start + segment_frames)
            row = {
                "clip_id": f"action_{action_idx:04d}_{part:03d}",
                "start_s": float(clip_start / fps),
                "end_s": float(clip_end / fps),
                "duration_s": float((clip_end - clip_start) / fps),
                "caption": caption,
                "confidence": finite_float(action.get("confidence")) if action.get("confidence") is not None else None,
                "source": f"action_json:{actions_json}",
                "grounding_status": "aligned_from_existing_task_action_frames",
                "evidence_frames": [clip_start, max(clip_start, clip_end - 1)],
            }
            rows.append(row)
            events.append({"event": "semantic_clip_from_action_json", **row})
            clip_start = clip_end
            part += 1
    return rows, events


def semantic_rows_from_caption_stage(stage_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    if not stage_path.exists():
        return [], [], None
    stage = load_json(stage_path)
    rows = stage.get("semantic_rows") if isinstance(stage.get("semantic_rows"), list) else []
    events = stage.get("caption_events") if isinstance(stage.get("caption_events"), list) else []
    return [row for row in rows if isinstance(row, dict)], [row for row in events if isinstance(row, dict)], stage


def copy_stage_artifact(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def throughput_rows_from_steps(pipeline_manifest: dict[str, Any], raw_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    video = raw_manifest.get("video") if isinstance(raw_manifest.get("video"), dict) else {}
    duration = finite_positive(video.get("duration_s"))
    if duration is None:
        fps = finite_positive(raw_manifest.get("fps")) or finite_positive(video.get("fps"))
        frame_count = finite_positive(raw_manifest.get("frame_count")) or finite_positive(video.get("frame_count"))
        if fps and frame_count:
            duration = frame_count / fps
    rows: list[dict[str, Any]] = []
    steps = pipeline_manifest.get("steps") if isinstance(pipeline_manifest.get("steps"), list) else []
    for step in steps:
        if not isinstance(step, dict):
            continue
        elapsed = finite_positive(step.get("elapsed_s"))
        row = {
            "module": str(step.get("step") or "unknown"),
            "status": str(step.get("status") or "unknown"),
            "elapsed_s": elapsed,
            "input_duration_s": duration,
            "failed": str(step.get("status")) != "ok",
            "log": step.get("log"),
        }
        if elapsed and duration:
            row["module_speed_x"] = duration / elapsed
        rows.append(row)
    return rows


def build_self_consistency_qc(
    raw_manifest: dict[str, Any],
    wilor_raw: dict[str, Any],
    hand_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    pipeline_manifest: dict[str, Any],
) -> dict[str, Any]:
    frame_count = int(raw_manifest.get("frame_count") or len(raw_manifest.get("frames") or []))
    visible_rows = [row for row in hand_rows if row.get("visibility") == "visible"]
    frames_with_candidates = len({int(row["frame_idx"]) for row in visible_rows if row.get("frame_idx") is not None})
    wilor_frames = wilor_raw.get("frames") if isinstance(wilor_raw.get("frames"), list) else []
    streams = ((pipeline_manifest.get("ffprobe_overlay") or {}).get("ffprobe") or {}).get("streams")
    render_frames = None
    render_duration = None
    render_width = None
    render_height = None
    if isinstance(streams, list) and streams:
        stream0 = streams[0]
        if isinstance(stream0, dict):
            render_frames = finite_float(stream0.get("nb_read_frames"))
            render_duration = finite_float(stream0.get("duration"))
            render_width = finite_float(stream0.get("width"))
            render_height = finite_float(stream0.get("height"))
    return {
        "schema": "ego.annotation.self_consistency_qc.v0",
        "status": "partial_available_for_minimal_v22_run",
        "frame_count_raw_manifest": frame_count,
        "frame_count_wilor_stream": len(wilor_frames),
        "frames_with_wilor_candidates": frames_with_candidates,
        "wilor_candidate_frame_coverage": float(frames_with_candidates / frame_count) if frame_count > 0 else None,
        "wilor_visible_candidate_rows": len(visible_rows),
        "render_overlay": {
            "nb_read_frames": int(render_frames) if render_frames is not None else None,
            "duration_s": render_duration,
            "width": int(render_width) if render_width is not None else None,
            "height": int(render_height) if render_height is not None else None,
            "frame_count_matches_raw_manifest": bool(render_frames == frame_count) if render_frames is not None else None,
        },
        "semantic_clip_count": len(semantic_rows),
        "limitations": [
            "Head/camera residual metrics require fixed-gauge GT or external metric anchors; DROID rows alone do not establish ATE/RPE accuracy.",
            "Hybrid hand rows may be quality-degraded when WiLoR-vs-HaWoR residuals are large; D8 drift correction is not applied.",
            "No caption grounding metrics unless action-json captions are supplied.",
        ],
    }


def add_missing_module_errors(
    bundle: ArtifactBundle,
    *,
    has_semantics: bool,
    has_camera: bool,
    has_hawor: bool,
    has_hybrid: bool,
    hybrid_quality_degraded: bool,
    d8_stage: dict[str, Any] | None,
    caption_stage: dict[str, Any] | None,
    d10_stage: dict[str, Any] | None,
    d11_stage: dict[str, Any] | None,
) -> None:
    if not has_camera:
        bundle.add_error(
            "head_camera_unavailable",
            "error",
            "D4 head/camera trajectory is not produced by this V22 run root.",
            "The bundle must not infer camera pose from UniDepth intrinsics or overlay renders; a DROID/VIO/SLAM/head-pose stage is required.",
        )
    else:
        bundle.add_error(
            "head_camera_video_derived_uncertain_gauge",
            "degraded",
            "D4 camera trajectory is present but video-derived and lacks an external fixed-gauge metric anchor.",
            "DROID provides a first-class trajectory estimate; ATE/RPE metric claims still require device VIO/SLAM/IMU, fiducial/mocap, or benchmark GT.",
        )
    if not has_hawor:
        bundle.add_error(
            "hawor_metric_mano_unavailable",
            "error",
            "D5 HaWoR metric MANO wrist/root source is not present in this V22 run root.",
            "Raw WiLoR visible geometry cannot substitute for HaWoR metric wrist/root translation.",
        )
    if not has_hybrid:
        bundle.add_error(
            "hybrid_temporal_hand_fusion_unavailable",
            "error",
            "D7 HaWoR+WiLoR detector-bounded temporal fusion has not run.",
            "Visible WiLoR candidate rows are preserved, but they are not the final fused metric hand state.",
        )
    elif hybrid_quality_degraded:
        bundle.add_error(
            "hybrid_temporal_hand_fusion_quality_degraded",
            "degraded",
            "D7 hybrid hand fusion ran, but WiLoR-vs-HaWoR projection residuals are large on at least one row.",
            "The output is a metric hand candidate with explicit residuals; downstream accuracy claims require correction or evaluator support.",
        )
    if d8_stage is None:
        bundle.add_error(
            "gt_free_hand_self_calibration_unavailable",
            "error",
            "D8 GT-free smooth-drift hand self-calibration stage artifact is absent.",
            "The pipeline must run the cross-source residual/bias estimation stage before product adaptation.",
        )
    elif d8_stage.get("status") not in {"ok"}:
        bundle.add_error(
            "gt_free_hand_self_calibration_degraded",
            "degraded",
            f"D8 stage ran with status {d8_stage.get('status')}.",
            "The output is preserved as a GT-free correction hypothesis; absent support or large residuals do not certify fixed-gauge 3D accuracy.",
        )
    elif (d8_stage.get("summary") or {}).get("accepted_correction_rows", 0) == 0:
        bundle.add_error(
            "gt_free_hand_self_calibration_no_correction_accepted",
            "degraded",
            "D8 stage ran but accepted no image-plane bias correction rows.",
            "The stage preserves the residual field and correction family even when current evidence does not justify applying corrections.",
        )
    if not has_semantics:
        status = caption_stage.get("status") if isinstance(caption_stage, dict) else "missing"
        bundle.add_error(
            "caption_source_unavailable",
            "degraded" if caption_stage is not None else "error",
            f"D9b captioning stage produced no semantic clip rows; status={status}.",
            "Caption rows require source-backed task/action captions or explicit external review. The runtime does not hallucinate captions from filenames or visual guesses.",
        )
    if d10_stage is None:
        bundle.add_error(
            "self_consistency_qc_unavailable",
            "error",
            "D10 self-consistency QC stage artifact is absent.",
            "The product bundle needs frozen-artifact consistency checks rather than relying on render existence alone.",
        )
    elif d10_stage.get("status") != "ok":
        bundle.add_error(
            "self_consistency_qc_degraded",
            "degraded" if d10_stage.get("status") != "failed" else "error",
            f"D10 self-consistency QC ran with status {d10_stage.get('status')}.",
            "The QC artifact records which frozen streams disagreed; it is an internal consistency check, not fixed-gauge metric accuracy.",
        )
    if d11_stage is None:
        bundle.add_error(
            "offline_evaluator_unavailable",
            "error",
            "D11 evaluator stage artifact is absent.",
            "Evaluation must run over frozen prediction artifacts and record GT availability instead of being silently skipped.",
        )
    elif d11_stage.get("status") != "ok":
        bundle.add_error(
            "offline_evaluator_gt_unavailable",
            "degraded",
            f"D11 evaluator stage ran with status {d11_stage.get('status')}.",
            "Evaluator readiness and prediction row counts are recorded; fixed-gauge correctness metrics remain unmeasured without GT sidecars.",
        )


def adapt(args: argparse.Namespace) -> Path:
    run_root = args.run_root.resolve()
    pipeline_manifest_path = (args.pipeline_manifest or (run_root / "annotation_pipeline_manifest.json")).resolve()
    pipeline_manifest = load_json(pipeline_manifest_path)
    state_path = run_root / "state" / "annotations_v22_renderable.json"
    state = load_json(state_path) if state_path.exists() else {}
    measurements = state.get("measurements") if isinstance(state.get("measurements"), dict) else {}
    input_manifest_path = run_root / "input" / "input_manifest.json"
    input_manifest = load_json(input_manifest_path)
    raw_manifest_path = resolve_path(run_root, measurements.get("raw_frame_manifest"), run_root / "input" / "raw_frame_manifest" / "manifest.json")
    calibration_path = resolve_path(run_root, measurements.get("calibration_contract"), run_root / "state" / "calibration" / "v19_camera_calibration_contract.json")
    wilor_path = resolve_path(run_root, measurements.get("wilor_raw_hands"), run_root / "measurements" / "hand_candidates" / "wilor_v21" / "wilor_raw_hands.json")
    raw_manifest = load_json(raw_manifest_path)
    calibration_source = load_json(calibration_path)
    wilor_raw = load_json(wilor_path)
    stage_artifacts = pipeline_manifest.get("stage_artifacts") if isinstance(pipeline_manifest.get("stage_artifacts"), dict) else {}
    d8_path = resolve_path(run_root, stage_artifacts.get("gt_free_drift_self_calibration"), run_root / "state" / "gt_free_self_calibration" / "v22_gt_free_drift_self_calibration.json")
    caption_stage_path = resolve_path(run_root, stage_artifacts.get("captioning"), run_root / "state" / "semantic_clips" / "v22_captioning_stage.json")
    d10_path = resolve_path(run_root, stage_artifacts.get("self_consistency_qc"), run_root / "state" / "self_consistency" / "v22_full_self_consistency_qc.json")
    d11_path = resolve_path(run_root, stage_artifacts.get("evaluator"), run_root / "evaluation" / "v22_evaluator_stage.json")
    d8_stage = load_json(d8_path) if d8_path.exists() else None
    caption_stage_rows, caption_stage_events, caption_stage = semantic_rows_from_caption_stage(caption_stage_path)
    d10_stage = load_json(d10_path) if d10_path.exists() else None
    d11_stage = load_json(d11_path) if d11_path.exists() else None

    job_id = args.job_id or str(pipeline_manifest.get("case_id") or input_manifest.get("case_id") or run_root.name)
    bundle = ArtifactBundle(args.output_root, job_id)
    bundle.add_provenance("v22_adapter", "adaptation_started", schema=SCHEMA_NAME, schema_version=SCHEMA_VERSION, run_root=str(run_root), pipeline_manifest=str(pipeline_manifest_path))
    bundle.add_provenance("ingestion", "raw_frame_manifest_promoted", raw_frame_manifest=str(raw_manifest_path), rows=int(raw_manifest.get("frame_count") or len(raw_manifest.get("frames") or [])))
    fingerprint = input_manifest.get("source_fingerprint") if isinstance(input_manifest.get("source_fingerprint"), dict) else {}
    if fingerprint.get("sha256"):
        bundle.add_provenance("ingestion", "input_hash_resolved", sha256=fingerprint["sha256"])
    else:
        bundle.add_error("input_hash_unavailable", "degraded", "V22 input manifest did not contain a source SHA256.", "D1 requires input hash provenance for reproducible annotation outputs.")

    calibration_contract = normalize_calibration(calibration_source, bundle.errors)
    bundle.add_provenance("calibration", "v22_calibration_contract_promoted", source=str(calibration_path), status=calibration_contract["status"])

    frames = frame_rows(raw_manifest, input_manifest)
    camera_rows, camera_stage = camera_trajectory_rows(run_root)
    hybrid_rows, hybrid_stage, hybrid_quality_degraded = hybrid_hand_rows(run_root, float(args.hybrid_residual_warning_px))
    hand_rows = hybrid_rows if hybrid_rows else hand_candidate_rows(raw_manifest, wilor_raw, wilor_path)
    semantic_rows, caption_events = (caption_stage_rows, caption_stage_events) if caption_stage is not None else semantic_rows_from_actions(args.actions_json, raw_manifest, float(args.max_semantic_clip_s))
    throughput_rows = throughput_rows_from_steps(pipeline_manifest, raw_manifest)
    qc = d10_stage if d10_stage is not None else build_self_consistency_qc(raw_manifest, wilor_raw, hand_rows, semantic_rows, pipeline_manifest)
    qc["head_camera_rows"] = len(camera_rows)
    qc["hand_state_source"] = "hybrid_metric_hand_candidate" if hybrid_rows else "wilor_raw_candidates"
    qc["hybrid_quality_degraded"] = bool(hybrid_quality_degraded)
    bundle.state_dir.mkdir(parents=True, exist_ok=True)
    qc_path = bundle.state_dir / "self_consistency_qc.json"
    write_json(qc_path, qc)
    d8_product_path = copy_stage_artifact(d8_path, bundle.state_dir / "gt_free_drift_self_calibration.json")
    caption_product_path = copy_stage_artifact(caption_stage_path, bundle.state_dir / "captioning_stage.json")
    d10_product_path = copy_stage_artifact(d10_path, bundle.state_dir / "full_self_consistency_qc.json")
    d11_product_path = copy_stage_artifact(d11_path, bundle.state_dir / "evaluator_stage.json")

    has_hawor = (run_root / "measurements" / "hand_candidates" / "hawor_world" / "hawor_world_hands.npz").exists()
    add_missing_module_errors(
        bundle,
        has_semantics=bool(semantic_rows),
        has_camera=bool(camera_rows),
        has_hawor=has_hawor,
        has_hybrid=bool(hybrid_rows),
        hybrid_quality_degraded=bool(hybrid_quality_degraded),
        d8_stage=d8_stage,
        caption_stage=caption_stage,
        d10_stage=d10_stage,
        d11_stage=d11_stage,
    )
    metric_observations: dict[str, Any] = {}

    def merge_metric_observation(metric_id: str, values: Any, metadata: dict[str, Any] | None = None) -> None:
        metadata = metadata or {}
        raw_values = values.get("values", []) if isinstance(values, dict) else values
        if not isinstance(raw_values, list):
            return
        existing = metric_observations.get(metric_id)
        if metadata or isinstance(existing, dict):
            if not isinstance(existing, dict):
                prior_values = existing if isinstance(existing, list) else []
                existing = {"values": list(prior_values)}
                metric_observations[metric_id] = existing
            existing.setdefault("values", [])
            existing["values"].extend(raw_values)
            existing.update(metadata)
        else:
            metric_observations.setdefault(metric_id, [])
            if isinstance(metric_observations[metric_id], list):
                metric_observations[metric_id].extend(raw_values)

    if semantic_rows:
        metric_observations["semantic_segment_duration_s"] = [row["duration_s"] for row in semantic_rows]
    if isinstance(d11_stage, dict):
        d11_status = str(d11_stage.get("status") or "unknown")
        diagnostic_metadata = {
            "status": "prediction_diagnostic",
            "measurement_role": "prediction_only_diagnostic_not_gt_accuracy",
            "evaluator_status": d11_status,
        }
        if isinstance(d11_stage.get("metric_observations"), dict):
            for key, values in d11_stage["metric_observations"].items():
                merge_metric_observation(key, values, diagnostic_metadata if d11_status == "no_gt_unmeasured" else None)
        if isinstance(d11_stage.get("diagnostic_observations"), dict):
            for key, values in d11_stage["diagnostic_observations"].items():
                merge_metric_observation(key, values, diagnostic_metadata)
    metric_rows = build_metric_rows(metric_observations, throughput_rows, calibration_status=str(calibration_contract["status"]))
    forecast = throughput_forecast(throughput_rows)
    initial_status = status_from_errors(bundle.errors)
    renders = pipeline_manifest.get("renders") if isinstance(pipeline_manifest.get("renders"), dict) else {}
    render_path_keys = ("v22_overlay", "hand_overlay", "hybrid_hand_overlay", "depth_overlay")
    render_artifacts = {
        "optional_qc_demo": [renders[key] for key in render_path_keys if isinstance(renders.get(key), str) and renders.get(key)],
        "v22_minimal_renders": renders,
        "render_metadata": {"overlay_source": renders.get("overlay_source")},
        "state_artifacts": {
            "self_consistency_qc": str(qc_path),
            "gt_free_drift_self_calibration": d8_product_path,
            "captioning_stage": caption_product_path,
            "full_self_consistency_qc": d10_product_path,
            "evaluator_stage": d11_product_path,
        },
        "note": "renders are projections of promoted candidate/numeric state and cannot change numeric results",
    }
    manifest_path = bundle.write(
        request={
            "job_id": job_id,
            "video_uri": input_manifest.get("original_video") or input_manifest.get("primary_video"),
            "public_endpoint": "/v1/annotation-jobs",
            "source_run_root": str(run_root),
            "source_pipeline_manifest": str(pipeline_manifest_path),
        },
        calibration_contract=calibration_contract,
        tables={
            "frames": frames,
            "head_camera": camera_rows,
            "hand_states": hand_rows,
            "semantic_clips": semantic_rows,
            "validation_metrics": metric_rows,
        },
        events={
            "overlay_events": [
                {"event": "v22_minimal_overlay_available", "renders": renders, "state_role": "qc_demo_projection_of_current_hand_rows"},
                {"event": "self_consistency_qc_written", "path": str(qc_path), "status": qc["status"]},
                {"event": "camera_trajectory_stage", "status": camera_stage.get("status") if isinstance(camera_stage, dict) else "missing", "rows": len(camera_rows)},
                {"event": "hybrid_hand_stage", "status": hybrid_stage.get("status") if isinstance(hybrid_stage, dict) else "missing", "rows": len(hybrid_rows)},
                {"event": "gt_free_drift_self_calibration_stage", "status": d8_stage.get("status") if isinstance(d8_stage, dict) else "missing", "path": d8_product_path},
                {"event": "captioning_stage", "status": caption_stage.get("status") if isinstance(caption_stage, dict) else "missing", "rows": len(semantic_rows), "path": caption_product_path},
                {"event": "full_self_consistency_qc_stage", "status": d10_stage.get("status") if isinstance(d10_stage, dict) else qc.get("status"), "path": d10_product_path or str(qc_path)},
                {"event": "evaluator_stage", "status": d11_stage.get("status") if isinstance(d11_stage, dict) else "missing", "path": d11_product_path},
            ],
            "caption_events": caption_events,
        },
        throughput_forecast=forecast,
        status=initial_status,
        render_artifacts=render_artifacts,
    )
    print(json.dumps({"status": status_from_errors(bundle.errors), "manifest_path": str(manifest_path), "errors": len(bundle.errors)}, indent=2))
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Completed V22 minimal run root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory where the ego.annotation.output bundle will be written.")
    parser.add_argument("--job-id", default=None, help="Output bundle job id. Defaults to V22 case_id/run-root name.")
    parser.add_argument("--pipeline-manifest", type=Path, default=None, help="Defaults to <run-root>/annotation_pipeline_manifest.json.")
    parser.add_argument("--actions-json", type=Path, default=None, help="Optional existing task/action caption JSON to adapt into semantic clips.")
    parser.add_argument("--max-semantic-clip-s", type=float, default=3.0)
    parser.add_argument("--hybrid-residual-warning-px", type=float, default=50.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    adapt(parse_args())
