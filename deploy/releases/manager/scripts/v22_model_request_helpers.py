#!/usr/bin/env python3
"""Build future-API request JSONs for V22 GPU-heavy model calls."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelRequestError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ModelRequestError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _resolve_path(raw: Any, *, base: Path) -> Path:
    if raw is None:
        raise ModelRequestError("missing path value")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def input_manifest(run_root: Path) -> dict[str, Any]:
    return load_json(run_root / "input" / "input_manifest.json")


def raw_frame_manifest(run_root: Path) -> dict[str, Any]:
    return load_json(run_root / "input" / "raw_frame_manifest" / "manifest.json")


def primary_video(run_root: Path) -> Path:
    manifest = input_manifest(run_root)
    return _resolve_path(manifest.get("primary_video") or manifest.get("clip_video"), base=run_root)


def case_id(run_root: Path) -> str:
    manifest = input_manifest(run_root)
    return str(manifest.get("case_id") or run_root.name)


def video_metadata(run_root: Path) -> dict[str, Any]:
    manifest = raw_frame_manifest(run_root)
    video = manifest.get("video") if isinstance(manifest.get("video"), dict) else {}
    frames = manifest.get("frames") if isinstance(manifest.get("frames"), list) else []
    first = frames[0] if frames and isinstance(frames[0], dict) else {}
    width = video.get("width") or first.get("source_width") or first.get("manifest_width")
    height = video.get("height") or first.get("source_height") or first.get("manifest_height")
    frame_count = manifest.get("frame_count") or video.get("frame_count") or len(frames)
    fps = manifest.get("fps") or video.get("fps")
    duration_s = video.get("duration_s") or manifest.get("duration_s")
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_s": duration_s,
        "format": "single_rgb_video",
        "raw_frame_manifest": str(run_root / "input" / "raw_frame_manifest" / "manifest.json"),
        "source_frame_manifest": str(run_root / "input" / "source_frame_manifest" / "manifest.json"),
    }


def declared_input_artifacts(run_root: Path, *, calibration_contract: Path | None = None) -> list[dict[str, str]]:
    """Declare source-side files the API client must materialize before inference."""
    candidates = [
        ("input_video", primary_video(run_root), True),
        ("raw_frame_manifest", run_root / "input" / "raw_frame_manifest" / "manifest.json", True),
        ("source_frame_manifest", run_root / "input" / "source_frame_manifest" / "manifest.json", False),
    ]
    if calibration_contract is not None:
        candidates.append(("calibration_contract", calibration_contract, True))
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for role, raw_path, required in candidates:
        path = Path(raw_path).expanduser().resolve()
        if str(path) in seen:
            continue
        if not path.is_file():
            if required:
                raise ModelRequestError(f"declared input artifact is missing: {path}")
            continue
        result.append({"role": role, "path": str(path)})
        seen.add(str(path))
    return result


def camera_contract(run_root: Path, calibration_contract: Path | None = None) -> dict[str, Any]:
    path = calibration_contract or (run_root / "state" / "calibration" / "v19_camera_calibration_contract.json")
    contract = load_json(path)
    values = contract.get("intrinsics_fx_fy_cx_cy")
    if not isinstance(values, list) or len(values) != 4:
        raise ModelRequestError(f"calibration contract lacks intrinsics_fx_fy_cx_cy: {path}")
    meta = video_metadata(run_root)
    image_size = [int(meta["width"]), int(meta["height"])] if meta.get("width") and meta.get("height") else None
    return {
        "model": "pinhole",
        "intrinsics_px": [float(values[0]), float(values[1]), float(values[2]), float(values[3])],
        "image_size": image_size,
        "distortion": contract.get("distortion") if "distortion" in contract else None,
        "source": str(path),
        "source_method": contract.get("intrinsics_source") or contract.get("method"),
    }


def model_request(
    *,
    run_root: Path,
    model: str,
    stage: str,
    output_dir: Path,
    request_path: Path,
    camera: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    video = primary_video(run_root)
    payload: dict[str, Any] = {
        "schema": "ego.annotation.model_request.v1",
        "created_at": utc_now(),
        "job_id": case_id(run_root),
        "case_id": case_id(run_root),
        "model": model,
        "stage": stage,
        "input_video": str(video),
        "video_meta": video_metadata(run_root),
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "input_artifacts": declared_input_artifacts(run_root),
        "execution": execution or {"mode": "script", "api_ready_contract": True},
        "parameters": parameters or {},
        "provenance": {
            "source": "scripts/v22_model_request_helpers.py",
            "contract_basis": "Feishu Ego parallel pipeline per-algorithm input contract; current execution may still be local/remote script-backed.",
        },
    }
    if camera is not None:
        payload["camera"] = camera
    return write_json(request_path, payload)


def write_unidepth_request(run_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    run_root = run_root.resolve()
    return model_request(
        run_root=run_root,
        model="unidepth",
        stage="D3_depth_intrinsics_support",
        output_dir=(output_dir or (run_root / "measurements" / "depth_candidates" / "unidepth_v2")).resolve(),
        request_path=run_root / "requests" / "unidepth.json",
        parameters={"input_contract": {"required_fields": ["input_video", "output_dir"]}},
    )


def write_wilor_request(run_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    run_root = run_root.resolve()
    return model_request(
        run_root=run_root,
        model="wilor",
        stage="D6_visible_hand_candidates",
        output_dir=(output_dir or (run_root / "measurements" / "hand_candidates" / "wilor_v21")).resolve(),
        request_path=run_root / "requests" / "wilor.json",
        parameters={"input_contract": {"required_fields": ["input_video", "output_dir"]}},
    )


def write_droid_request(run_root: Path, output_dir: Path | None = None, calibration_contract: Path | None = None) -> dict[str, Any]:
    run_root = run_root.resolve()
    payload = model_request(
        run_root=run_root,
        model="droid",
        stage="D4_camera_trajectory",
        output_dir=(output_dir or (run_root / "measurements" / "camera_trajectory" / "droid_full_frame")).resolve(),
        request_path=run_root / "requests" / "droid.json",
        camera=camera_contract(run_root, calibration_contract),
        parameters={
            "input_contract": {"required_fields": ["input_video", "camera", "output_dir"]},
            "output_contract": {
                "preserve_existing_d4_files": True,
                "shared_geometry_manifest": "droid_shared_geometry.json",
                "shared_consumers": ["D4_camera_trajectory"],
            },
            "execution_contract": {
                "droid_instance_count_per_video": 1,
                "hawor_service_must_not_instantiate_droid": True,
            },
        },
    )
    if calibration_contract is not None:
        payload["input_artifacts"].append({"role": "calibration_contract", "path": str(calibration_contract.resolve())})
    return write_json(run_root / "requests" / "droid.json", payload)


def write_vggt_camera_request(
    run_root: Path,
    output_dir: Path | None = None,
    calibration_contract: Path | None = None,
    request_path: Path | None = None,
    backend: str = "vggt_omega",
    stage: str = "D4_camera_trajectory",
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    merged_parameters: dict[str, Any] = {
        "input_contract": {"required_fields": ["input_video", "camera", "output_dir"]},
        "batch_contract": {
            "tensor_shape": "[B,S,3,H,W]",
            "batch_axis_semantics": "independent videos/items; no cross-item state or attention mixing",
            "requires_equal_sequence_length_or_bucketed_windows": True,
            "padding_status": "not_supported_without_explicit_model_attention_mask",
        },
        "output_contract": {
            "writes_droid_compatible_camera_artifacts": True,
            "dense_trajectory_files": [
                "droid_dense_trajectory.npz",
                "droid_dense_trajectory.json",
                "droid_qc.json",
                "v22_camera_trajectory_stage.json",
            ],
        },
    }
    if parameters:
        for key, value in parameters.items():
            if isinstance(value, dict) and isinstance(merged_parameters.get(key), dict):
                merged = dict(merged_parameters[key])
                merged.update(value)
                merged_parameters[key] = merged
            else:
                merged_parameters[key] = value
    payload = model_request(
        run_root=run_root,
        model=f"{backend}_camera_geometry",
        stage=stage,
        output_dir=(output_dir or (run_root / "measurements" / "camera_trajectory" / "vggt_omega_full_frame")).resolve(),
        request_path=(request_path or (run_root / "requests" / "vggt_camera.json")).resolve(),
        camera=camera_contract(run_root, calibration_contract),
        parameters=merged_parameters,
        execution={"mode": "resident_tensor_batch", "api_ready_contract": True, "droid_replacement_candidate": True},
    )
    if calibration_contract is not None:
        payload["input_artifacts"].append({"role": "calibration_contract", "path": str(calibration_contract.resolve())})
    return write_json((request_path or (run_root / "requests" / "vggt_camera.json")).resolve(), payload)


def write_hawor_request(
    run_root: Path,
    output_dir: Path | None = None,
    calibration_contract: Path | None = None,
    track_manifest: Path | None = None,
    mask_manifest: Path | None = None,
    camera_artifact: Path | None = None,
    droid_shared_manifest: Path | None = None,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    default_calibration = run_root / "state" / "calibration" / "v19_camera_calibration_contract.json"
    calibration = calibration_contract.resolve() if calibration_contract is not None else (default_calibration.resolve() if default_calibration.is_file() else None)
    payload = model_request(
        run_root=run_root,
        model="hawor",
        stage="D5_metric_mano_source",
        output_dir=(output_dir or (run_root / "measurements" / "hand_candidates" / "hawor_world")).resolve(),
        request_path=run_root / "requests" / "hawor.json",
        camera=camera_contract(run_root, calibration) if calibration is not None else None,
        parameters={
            "input_contract": {
                "required_fields": ["input_video", "output_dir", "raw_frame_manifest"],
                "optional_fields": ["camera", "track_manifest", "mask_manifest", "camera_artifact"],
                "droid_slam": "excluded_from_hawor_service",
            },
            "batch_contract": {
                "work_unit": "equal_length_temporal_track_chunk",
                "sequence_length": 16,
                "state_scope": "item_affine",
            },
        },
    )
    if calibration is not None:
        payload["input_artifacts"].append({"role": "calibration_contract", "path": str(calibration)})
    for role, path in (("track_manifest", track_manifest), ("mask_manifest", mask_manifest), ("camera_artifact", camera_artifact)):
        if path is not None:
            resolved = path.resolve()
            if not resolved.is_file():
                raise ModelRequestError(f"HaWoR input artifact is missing: {resolved}")
            payload["input_artifacts"].append({"role": role, "path": str(resolved)})
    payload["parameters"]["hawor_contract"] = {
        "droid_slam_included": False,
        "droid_shared_manifest_allowed": False,
        "upstream_camera_artifact_role": "camera_artifact" if camera_artifact is not None else None,
    }
    # Backward compatibility is limited to the legacy script-mode adapter. The
    # resident HaWoR service and API launcher never set this argument.
    if droid_shared_manifest is not None:
        legacy_manifest = droid_shared_manifest.resolve()
        payload["droid_shared_manifest"] = str(legacy_manifest)
        payload["parameters"]["input_contract"]["required_fields"].append("droid_shared_manifest")
        payload["parameters"]["hawor_contract"] = {
            "droid_slam_included": False,
            "legacy_script_adapter_only": True,
            "resident_service_contract": "hawor_without_droid",
        }
    return write_json(run_root / "requests" / "hawor.json", payload)


def write_all_available_model_requests(run_root: Path) -> dict[str, str]:
    requests = {
        "unidepth": write_unidepth_request(run_root),
        "wilor": write_wilor_request(run_root),
    }
    calibration = run_root / "state" / "calibration" / "v19_camera_calibration_contract.json"
    if calibration.exists():
        requests["droid"] = write_droid_request(run_root, calibration_contract=calibration)
        requests["hawor"] = write_hawor_request(run_root, calibration_contract=calibration)
    return {name: str(run_root / "requests" / f"{name}.json") for name in requests}
