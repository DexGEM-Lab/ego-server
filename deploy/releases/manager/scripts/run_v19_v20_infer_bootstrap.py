#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2


class ContractError(RuntimeError):
    pass


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_run_root(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise ContractError(f"run_root_exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def open_video(path: Path) -> tuple[cv2.VideoCapture, dict[str, Any]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ContractError(f"failed_to_open_video: {path}")
    metadata = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    if metadata["fps"] <= 0.0 or metadata["width"] <= 0 or metadata["height"] <= 0 or metadata["frame_count"] <= 0:
        cap.release()
        raise ContractError(f"invalid_video_metadata: {path} {metadata}")
    metadata["duration_s"] = float(metadata["frame_count"] / metadata["fps"])
    return cap, metadata


def video_metadata(path: Path) -> dict[str, Any]:
    cap, metadata = open_video(path)
    cap.release()
    return metadata


def collect_video_sources(root: Path) -> list[dict[str, Any]]:
    if not root.exists() or not root.is_dir():
        raise ContractError(f"missing_multiview_root: {root}")
    sources: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.mp4")):
        try:
            metadata = video_metadata(path)
        except ContractError as exc:
            metadata = {"status": "unreadable", "error": str(exc)}
        sources.append({"path": path, "metadata": metadata})
    if not sources:
        raise ContractError(f"multiview_root_contains_no_mp4: {root}")
    return sources


def build_raw_frame_manifest(clip: Path, output_dir: Path, render_width: int) -> dict[str, Any]:
    cap, metadata = open_video(clip)
    render_height = int(round(render_width * metadata["height"] / metadata["width"]))
    if render_height % 2:
        render_height += 1
    rgb_dir = output_dir / "input" / "raw_frame_manifest" / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    try:
        for frame_idx in range(int(metadata["frame_count"])):
            ok, frame = cap.read()
            if not ok:
                raise ContractError(f"video_ended_early: frame={frame_idx} expected={metadata['frame_count']} path={clip}")
            resized = cv2.resize(frame, (render_width, render_height), interpolation=cv2.INTER_AREA)
            frame_path = rgb_dir / f"{frame_idx:06d}.jpg"
            if not cv2.imwrite(str(frame_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                raise ContractError(f"failed_to_write_frame: {frame_path}")
            frames.append(
                {
                    "index": int(frame_idx),
                    "frame_idx": int(frame_idx),
                    "time_s": float(frame_idx / metadata["fps"]),
                    "rgb": frame_path,
                    "source_width": int(metadata["width"]),
                    "source_height": int(metadata["height"]),
                    "manifest_width": int(render_width),
                    "manifest_height": int(render_height),
                }
            )
    finally:
        cap.release()
    manifest = {
        "status": "ok",
        "method": "standalone_raw_frame_manifest_from_v16_component",
        "clip": clip,
        "video": metadata,
        "render_width": int(render_width),
        "render_height": int(render_height),
        "frames": frames,
    }
    manifest_path = output_dir / "input" / "raw_frame_manifest" / "manifest.json"
    write_json(manifest_path, manifest)
    return {"path": manifest_path, "frame_count": len(frames), "video": metadata}


def initial_physical_state(mode: str, input_video: Path, raw_manifest: dict[str, Any], case_id: str, modality: str) -> dict[str, Any]:
    prefix = "v20" if mode == "v20" else "v19"
    pending = [
        "target_object_plan_not_created_by_timeline_bootstrap",
        "segmentation_tracking_not_run_by_timeline_bootstrap",
        "depth_or_camera_measurements_not_run_by_timeline_bootstrap",
        "hand_model_measurements_not_run_by_timeline_bootstrap",
        "branch_optimization_not_run_by_timeline_bootstrap",
        "rendering_not_run_by_timeline_bootstrap",
    ]
    if mode == "v20" and modality == "stereo_pair":
        pending.append("metric_stereo_depth_requires_calibration_or_weak_nonmetric_disparity_report")
    if mode == "v20" and modality == "multiview_or_fisheye_directory":
        pending.append("multiview_metric_depth_requires_camera_model_or_native_depth_adapter")
    return {
        "schema": f"{prefix}_physical_state.v0",
        "mode": f"{prefix}_infer",
        "case_id": case_id,
        "status": "timeline_bootstrap_complete_measurements_pending",
        "input_video": input_video,
        "timeline": {
            "frame_count": int(raw_manifest["video"]["frame_count"]),
            "fps": float(raw_manifest["video"]["fps"]),
            "duration_s": float(raw_manifest["video"]["duration_s"]),
            "resolution": [int(raw_manifest["video"]["width"]), int(raw_manifest["video"]["height"])],
            "raw_frame_manifest": raw_manifest["path"],
        },
        "camera": {"state": "unmeasured", "required_for_metric_claims": True},
        "hands": [],
        "objects": [],
        "contacts": [],
        "occlusions": [],
        "nonpenetration": [],
        "pending_measurement_and_optimization_stages": pending,
        "renderer_boundary": f"renders are not produced by this {prefix} timeline bootstrap; V20 final renders must consume optimized annotations and state assembly outputs",
    }


def uncertainty_state(mode: str, modality: str) -> dict[str, Any]:
    prefix = "v20" if mode == "v20" else "v19"
    return {
        "schema": f"{prefix}_uncertainty_state.v0",
        "status": "timeline_observed_physical_measurements_pending_after_input_bootstrap",
        "modality": modality,
        "timeline_uncertainty": "low: decoded frame count/fps/resolution recorded from source video",
        "camera_depth_uncertainty": "unmeasured by bootstrap: run V20 depth registry/selector or weak stereo/native depth adapters next",
        "hand_state_uncertainty": "unmeasured by bootstrap: run prediction-side MANO candidate stages when a human hand is applicable",
        "object_geometry_pose_uncertainty": "unmeasured by bootstrap: create target object plan, masks, visible surfaces, and branch optimization next",
        "contact_occlusion_nonpenetration_uncertainty": "unsupported until hand/object/camera/depth states exist",
    }


def write_depth_candidate_artifacts(run_root: Path, mode: str, input_video: Path, stereo_right: Path | None, multiview_sources: list[dict[str, Any]] | None) -> None:
    if mode == "v20":
        candidates = [
            {
                "candidate_id": "monocular_video_depth_candidate",
                "source": input_video,
                "status": "available_not_run_in_local_bootstrap",
                "blocked_by": "heavy_depth_or_slam_inference_requires_declared_server_or_a800_target",
            }
        ]
        if stereo_right is not None:
            candidates.append(
                {
                    "candidate_id": "stereo_depth_candidate",
                    "left_video": input_video,
                    "right_video": stereo_right,
                    "right_video_metadata": video_metadata(stereo_right),
                    "status": "registered_not_executed_by_bootstrap",
                    "next_stage": "run calibrated stereo depth if calibration exists, otherwise run build_v20_uncalibrated_stereo_disparity.py as weak non-metric evidence",
                }
            )
        if multiview_sources is not None:
            candidates.append(
                {
                    "candidate_id": "multiview_or_fisheye_depth_candidate",
                    "source_count": len(multiview_sources),
                    "sources": multiview_sources,
                    "status": "registered_not_executed_by_bootstrap",
                    "next_stage": "run native depth/camera adapter when available; metric multiview claims require camera model/calibration",
                }
            )
        registry = {"schema": "v20_depth_candidate_registry.v0", "status": "registered_candidates_not_selected", "candidates": candidates}
        selection = {
            "schema": "v20_depth_selection_report.v0",
            "selected_candidate_id": None,
            "status": "no_candidate_executed_in_bootstrap",
            "reason": "bootstrap only verifies input timeline and records missing implemented depth/camera stages",
        }
        write_json(run_root / "measurements" / "depth_candidates" / "depth_candidate_registry.json", registry)
        write_json(run_root / "measurements" / "depth_candidates" / "depth_selection_report.json", selection)
        write_json(
            run_root / "state" / "v20_observation_bundle.json",
            {
                "schema": "v20_observation_bundle.v0",
                "status": "input_timeline_observed_physical_measurements_unresolved",
                "depth_candidate_registry": run_root / "measurements" / "depth_candidates" / "depth_candidate_registry.json",
            },
        )
    else:
        write_json(
            run_root / "measurements" / "depth_slam" / "depth_candidate_status.json",
            {
                "schema": "v19_depth_candidate_status.v0",
                "status": "not_run_in_bootstrap",
                "reason": "v19 prompt has no implemented stereo_or_multiview_specific adapter in current repository state",
            },
        )


def write_missing_input_record(args: argparse.Namespace, started: float) -> dict[str, Any]:
    run_root = args.run_root
    ensure_run_root(run_root, args.overwrite)
    mode = args.mode
    prefix = "v20" if mode == "v20" else "v19"
    status = {
        "status": "input_contract_failed",
        "mode": f"{prefix}_infer",
        "input_video": args.input,
        "input_exists": False,
        "case_id": args.case_id,
        "elapsed_s": float(time.time() - started),
    }
    write_json(run_root / "input" / "input_manifest.json", {"schema": f"{prefix}_input_manifest.v0", **status})
    write_text(
        run_root / "INPUT_CONTRACT_FAILED.md",
        f"# Input Contract Failed\n\nRequested `{prefix}_infer` input does not exist on this machine:\n\n`{args.input}`\n",
    )
    write_json(run_root / "run_summary.json", status)
    return status


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    input_video = args.input
    if not input_video.exists():
        if args.record_missing_input:
            return write_missing_input_record(args, started)
        raise ContractError(f"missing_input_video: {input_video}")
    if not input_video.is_file():
        raise ContractError(f"input_must_be_video_file_for_current_prompt_contract: {input_video}")
    if args.stereo_right is not None and not args.stereo_right.exists():
        raise ContractError(f"missing_stereo_right_video: {args.stereo_right}")
    multiview_sources = collect_video_sources(args.multiview_root) if args.multiview_root is not None else None
    modality = "single_video"
    if args.stereo_right is not None:
        modality = "stereo_pair"
    if multiview_sources is not None:
        modality = "multiview_or_fisheye_directory"
    ensure_run_root(args.run_root, args.overwrite)
    raw_manifest = build_raw_frame_manifest(input_video, args.run_root, int(args.render_width))
    prefix = "v20" if args.mode == "v20" else "v19"
    input_manifest = {
        "schema": f"{prefix}_input_manifest.v0",
        "mode": f"{prefix}_infer",
        "case_id": args.case_id,
        "input_video": input_video,
        "input_exists": True,
        "input_modality": modality,
        "raw_frame_manifest": raw_manifest["path"],
        "video": raw_manifest["video"],
        "stereo_right_video": args.stereo_right,
        "stereo_right_metadata": video_metadata(args.stereo_right) if args.stereo_right is not None else None,
        "multiview_root": args.multiview_root,
        "multiview_sources": multiview_sources,
        "claim_scope": "input_timeline_bootstrap_only_not_full_physical_annotation_inference",
    }
    write_json(args.run_root / "input" / "input_manifest.json", input_manifest)
    append_jsonl(
        args.run_root / "logs" / "harness_events.jsonl",
        {
            "event": "input_timeline_bootstrap_complete",
            "mode": f"{prefix}_infer",
            "case_id": args.case_id,
            "frame_count": raw_manifest["frame_count"],
            "input_modality": modality,
        },
    )
    physical_state = initial_physical_state(args.mode, input_video, raw_manifest, args.case_id, modality)
    uncertainty = uncertainty_state(args.mode, modality)
    write_json(args.run_root / "state" / f"{prefix}_physical_state.json", physical_state)
    write_json(args.run_root / "state" / f"{prefix}_uncertainty_state.json", uncertainty)
    write_depth_candidate_artifacts(args.run_root, args.mode, input_video, args.stereo_right, multiview_sources)
    evidence = f"""# {prefix.upper()} Infer Bootstrap Evidence\n\n## Observation\n\n- Input video exists and was decoded into a full raw-frame manifest.\n- Frame count: `{raw_manifest['video']['frame_count']}`.\n- FPS: `{raw_manifest['video']['fps']}`.\n- Resolution: `{raw_manifest['video']['width']}x{raw_manifest['video']['height']}`.\n- Input modality recorded as `{modality}`.\n\n## Interpretation\n\nThis script only establishes the input timeline and records pending measurement/optimization stages. V20 now has a minimal renderable infer closure, but it still requires target object planning, real segmentation tracks, depth/weak-depth evidence, optional hand candidates, temporal branch optimization, and rendering after bootstrap.\n\n## Commitment\n\nNo MANO hand state, object geometry/pose, contact, occlusion, nonpenetration, or annotation render is claimed from this bootstrap. The output is useful as input/timeline evidence for the next artifact-producing stages.\n"""
    write_text(args.run_root / "state" / f"{prefix}_agent_evidence.md", evidence)
    summary = {
        "status": "timeline_bootstrap_complete_measurements_pending",
        "mode": f"{prefix}_infer",
        "case_id": args.case_id,
        "run_root": args.run_root,
        "input_video": input_video,
        "input_modality": modality,
        "frame_count": raw_manifest["video"]["frame_count"],
        "fps": raw_manifest["video"]["fps"],
        "duration_s": raw_manifest["video"]["duration_s"],
        "raw_frame_manifest": raw_manifest["path"],
        "renders_created": False,
        "pending_measurement_and_optimization_stages": physical_state["pending_measurement_and_optimization_stages"],
        "elapsed_s": float(time.time() - started),
    }
    write_json(args.run_root / "run_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("v19", "v20"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--case-id", default="infer_bootstrap")
    parser.add_argument("--stereo-right", type=Path)
    parser.add_argument("--multiview-root", type=Path)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--record-missing-input", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = run(parse_args())
    except ContractError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(to_jsonable(result), indent=2, ensure_ascii=False))
