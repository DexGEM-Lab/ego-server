#!/usr/bin/env python3
"""Run bounded 256/256 DROID client probes with reproducible Hands-box masking.

``cache`` makes one exact 256-frame Hands YOLO cache and derives a separate
static-confidence raster: uint8 1 means static/keep and 0 means a dynamic hand
region. ``probe`` submits exactly one DROID session using the fixed 30fps,
328x584 geometry and a named factorial condition. The cache is input evidence,
not a segmentation claim: every ignored pixel originates from a recorded,
conservatively expanded Hands YOLO box.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from ego_annotation.api_backend import ApiBackend, ApiBackendConfig
from ego_annotation.full_video_timeline import (
    DROID_SERVICE_PUSH_CAPACITY,
    DroidChunkFinalizeError,
    FullVideoDriverConfig,
    FullVideoTimelineDriver,
    LiveFrozenApiStageClient,
    OpenCvFrameSource,
    _droid_input_shape_yx,
    _transform_box,
)
from ego_annotation.typed_contracts import TypedTensor


FACTORIAL_CONDITIONS: dict[str, dict[str, object]] = {
    "P1": {
        "filter_thresh": 8.0,
        "mask": False,
        "prediction": "If historical motion filtering is the decisive missing coupling, P1 restores a finite finalize without dynamic masking.",
    },
    "P2": {
        "filter_thresh": 2.4,
        "mask": True,
        "prediction": "If hand motion corrupts the graph independently of keyframe filtering, P2 restores a finite finalize with the unchanged default filter.",
    },
    "P3": {
        "filter_thresh": 8.0,
        "mask": True,
        "prediction": "If both retained finite-request mechanisms are necessary, only P3 restores a finite finalize.",
    },
}
DROID_DEFAULT_OPTIONS: dict[str, object] = {
    "buffer": 256,
    "filter_thresh": 2.4,
    "warmup": 8,
    "keyframe_thresh": 4.0,
    "frontend_thresh": 16.0,
    "frontend_window": 25,
    "frontend_radius": 2,
    "frontend_nms": 1,
    "backend_thresh": 22.0,
    "backend_radius": 2,
    "backend_nms": 3,
    "upsample": True,
    "beta": 0.3,
    "stereo": False,
}
CACHE_SCHEMA = "ego.annotation.droid_first_chunk_hands_cache.v1"
PROBE_SCHEMA = "ego.annotation.droid_first_chunk_factorial_probe.v1"
# YOLO boxes are localizers rather than hand silhouettes. Expanding each side by
# 10% of the larger box dimension (at least two source pixels) prevents a hand
# boundary from being treated as static after DROID's 8x pooling.
BOX_EXPANSION_FRACTION = 0.10
BOX_EXPANSION_MIN_SOURCE_PX = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cache", "probe"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--canonical-k-npz", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(FACTORIAL_CONDITIONS))
    parser.add_argument("--hands-cache-root", type=Path)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args()
    if args.mode == "cache" and (args.condition is not None or args.hands_cache_root is not None):
        parser.error("cache mode accepts neither --condition nor --hands-cache-root")
    if args.mode == "probe" and args.condition is None:
        parser.error("probe mode requires --condition")
    if args.mode == "probe" and bool(FACTORIAL_CONDITIONS[args.condition]["mask"]) != (args.hands_cache_root is not None):
        parser.error("P2/P3 require --hands-cache-root; P1 must not receive it")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_indices() -> tuple[int, ...]:
    return tuple(range(DROID_SERVICE_PUSH_CAPACITY))


def synthetic_inert_depth_record(source: OpenCvFrameSource, k_canonical: np.ndarray) -> SimpleNamespace:
    """Keep the typed production depth ABI valid for diagnostic monocular DROID."""
    height, width = source.timeline.height_px, source.timeline.width_px
    depth = np.full((1, height, width), 2.0, dtype=np.float32)
    confidence = np.ones_like(depth)
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    output = SimpleNamespace(
        frame_indices=(0,),
        depth_m=TypedTensor(depth, "metres", "source", "tyx", "diagnostic_inert_depth", {"consumed_by_service": False}),
        confidence=TypedTensor(confidence, "probability", "source", "tyx", "diagnostic_inert_confidence", {}),
        spatial=SimpleNamespace(pixel_to_source=identity, grid_id="source"),
        K_px=TypedTensor(k_canonical[None].astype(np.float32), "pixels", "source", "tij", "K", {}),
    )
    return SimpleNamespace(output=output)


def finite_summary(array: np.ndarray) -> dict[str, object]:
    values = np.asarray(array)
    finite = np.isfinite(values)
    return {
        "shape": list(values.shape),
        "finite_count": int(finite.sum()),
        "value_count": int(values.size),
        "all_finite": bool(finite.all()),
    }


def _validated_probe_source(args: argparse.Namespace) -> tuple[OpenCvFrameSource, np.ndarray]:
    source = OpenCvFrameSource.from_video(args.input)
    if source.timeline.frame_count != 2000 or source.timeline.fps != 30.0:
        raise ValueError("probe input must be the exact 2000-frame, 30fps task_4 clip")
    if source.timeline.width_px != 1920 or source.timeline.height_px != 1080:
        raise ValueError("probe input must retain the exact 1920x1080 source grid")
    with np.load(args.canonical_k_npz, allow_pickle=False) as state:
        k_canonical = np.asarray(state["K_canonical"], dtype=np.float64)
    if k_canonical.shape != (3, 3) or not np.isfinite(k_canonical).all():
        raise ValueError("canonical K must be finite 3x3")
    return source, k_canonical


def _driver(timeout_s: float) -> tuple[FullVideoTimelineDriver, FullVideoDriverConfig]:
    config = FullVideoDriverConfig(
        fps_condition="unidepth_full__droid_full",
        droid_fps=30.0,
        require_rgbd_capability=False,
        allow_monocular_droid_smoke=True,
        lower_filter_retry_thresh=None,
    )
    backend = ApiBackend(ApiBackendConfig(base_url="http://127.0.0.1", timeout_s=timeout_s, cosmos_enabled=False))
    return FullVideoTimelineDriver(LiveFrozenApiStageClient(backend), config), config


def _geometry_report(source: OpenCvFrameSource, config: FullVideoDriverConfig, k_canonical: np.ndarray) -> dict[str, object]:
    input_h, input_w = _droid_input_shape_yx(source.timeline, config)
    if (input_h, input_w) != (328, 584):
        raise RuntimeError(f"factorial probe must use effective 328x584 geometry, got {(input_h, input_w)}")
    source_to_input = np.asarray(
        [[input_w / source.timeline.width_px, 0.0, 0.0], [0.0, input_h / source.timeline.height_px, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return {
        "selection": "target_area",
        "shape_yx": [input_h, input_w],
        "source_to_droid_input": source_to_input.tolist(),
        "K_droid_input": (source_to_input @ k_canonical).tolist(),
    }


def _hands_trace_summary(records: tuple[Any, ...], trace: Any) -> dict[str, object]:
    """Serialize only fields carried by RequestBatchTrace and retained responses."""
    return {
        "elapsed_s": float(trace.completed_monotonic_s - trace.started_monotonic_s),
        "native_request_count": int(trace.request_count),
        "response_count": len(records),
        "response_cardinality_matches_requests": len(records) == int(trace.request_count),
    }


def _rasterize_static_mask(
    source_boxes: list[np.ndarray],
    *,
    source_width: int,
    source_height: int,
    input_width: int,
    input_height: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    mask = np.ones((input_height, input_width), dtype=np.uint8)
    raster_boxes: list[dict[str, object]] = []
    for box in source_boxes:
        x0, y0, x1, y1 = (float(value) for value in box)
        margin = max(BOX_EXPANSION_MIN_SOURCE_PX, BOX_EXPANSION_FRACTION * max(x1 - x0, y1 - y0))
        expanded = np.asarray(
            [max(0.0, x0 - margin), max(0.0, y0 - margin), min(float(source_width), x1 + margin), min(float(source_height), y1 + margin)],
            dtype=np.float64,
        )
        ix0 = max(0, min(input_width, int(math.floor(expanded[0] * input_width / source_width))))
        iy0 = max(0, min(input_height, int(math.floor(expanded[1] * input_height / source_height))))
        ix1 = max(0, min(input_width, int(math.ceil(expanded[2] * input_width / source_width))))
        iy1 = max(0, min(input_height, int(math.ceil(expanded[3] * input_height / source_height))))
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        mask[iy0:iy1, ix0:ix1] = 0
        raster_boxes.append(
            {
                "source_box_xyxy": [x0, y0, x1, y1],
                "expanded_source_box_xyxy": expanded.tolist(),
                "expansion_margin_source_px": margin,
                "droid_input_box_xyxy": [ix0, iy0, ix1, iy1],
            }
        )
    return mask, raster_boxes


def build_hands_cache(args: argparse.Namespace) -> int:
    if args.run_root.exists():
        raise FileExistsError(f"fresh --run-root required: {args.run_root}")
    args.run_root.mkdir(parents=True)
    source, k_canonical = _validated_probe_source(args)
    selected = source_indices()
    source.build_frame_store(selected, spill_dir=args.run_root / "frame_store")
    driver, config = _driver(args.timeout_s)
    geometry = _geometry_report(source, config, k_canonical)
    requests = tuple(driver._frame_request(source, "droid_factorial_hands_cache", "task4_first2000", index, "hands.detect") for index in selected)
    records, trace = driver._run_many_traced("hands.detect", requests)
    input_h, input_w = geometry["shape_yx"]
    masks = np.empty((len(selected), int(input_h), int(input_w)), dtype=np.uint8)
    frames: list[dict[str, object]] = []
    for position, (frame_index, result) in enumerate(zip(selected, records)):
        output = result.output
        if output.frame_indices != (frame_index,) or output.timestamps_s != (source.timeline.frames[frame_index].timestamp_s,):
            raise RuntimeError(f"Hands response changed exact source identity at frame {frame_index}")
        boxes = np.asarray(output.detections.boxes_xyxy.array[0], dtype=np.float32)
        scores = np.asarray(output.detections.scores.array[0], dtype=np.float32)
        sides = np.asarray(output.detections.sides.array[0])
        visibility = np.asarray(output.detections.visibility.array[0], dtype=np.float32)
        uncertainty = np.asarray(output.detections.uncertainty.array[0], dtype=np.float32)
        if not (boxes.shape == (scores.shape[0], 4) and sides.shape == scores.shape == visibility.shape == uncertainty.shape):
            raise RuntimeError(f"Hands output tensor shape mismatch at frame {frame_index}")
        slots: list[dict[str, object]] = []
        valid_source_boxes: list[np.ndarray] = []
        transform = np.asarray(output.spatial.pixel_to_source, dtype=np.float64)
        for slot, (box, score, side, vis, unc) in enumerate(zip(boxes, scores, sides, visibility, uncertainty)):
            source_box = _transform_box(np.asarray(box, dtype=np.float64), transform)
            valid = bool(
                np.isfinite(score)
                and float(score) >= config.min_hand_score
                and np.isfinite(source_box).all()
                and source_box[2] > source_box[0]
                and source_box[3] > source_box[1]
            )
            if valid:
                valid_source_boxes.append(source_box)
            slots.append(
                {
                    "slot": slot,
                    "box_xyxy_detector_grid": [float(value) for value in box],
                    "box_xyxy_source": [float(value) for value in source_box],
                    "score": float(score),
                    "side_code": int(side),
                    "visibility": float(vis),
                    "uncertainty": float(unc),
                    "valid_for_mask": valid,
                }
            )
        mask, raster_boxes = _rasterize_static_mask(
            valid_source_boxes,
            source_width=source.timeline.width_px,
            source_height=source.timeline.height_px,
            input_width=int(input_w),
            input_height=int(input_h),
        )
        masks[position] = mask
        frames.append(
            {
                "frame_index": frame_index,
                "timestamp_s": source.timeline.frames[frame_index].timestamp_s,
                "request_identity": {
                    "algorithm_id": result.algorithm_id,
                    "model_revision": result.model_revision,
                    "case_id": result.case_id,
                    "item_id": result.item_id,
                    "source_id": result.source_id,
                    "timeline": result.timeline.to_mapping(),
                    "ownership": output.ownership.to_wire(),
                },
                "slots": slots,
                "rasterized_dynamic_ignore_boxes": raster_boxes,
                "valid_box_count": len(valid_source_boxes),
                "dynamic_ignore_fraction": float(np.mean(mask == 0)),
                "static_keep_fraction": float(np.mean(mask == 1)),
            }
        )
    mask_path = args.run_root / "static_confidence_masks.npz"
    np.savez_compressed(mask_path, masks=masks)
    report = {
        "schema": CACHE_SCHEMA,
        "source": {
            "path": str(args.input.resolve()),
            "sha256": sha256_file(args.input),
            "frame_count": source.timeline.frame_count,
            "fps": source.timeline.fps,
            "width_px": source.timeline.width_px,
            "height_px": source.timeline.height_px,
        },
        "canonical_k_source": {"path": str(args.canonical_k_npz.resolve()), "sha256": sha256_file(args.canonical_k_npz)},
        "schedule": {
            "source_indices": list(selected),
            "source_timestamps_s": [source.timeline.frames[index].timestamp_s for index in selected],
            "push_count_requested": len(selected),
        },
        "droid_input_geometry": geometry,
        "hands": {
            "model_revision": config.model_revisions["hands.detect"],
            "request_count": len(records),
            "trace": _hands_trace_summary(records, trace),
            "output_contract": "Hands YOLO boxes/scores/sides/visibility/uncertainty only; no segmentation output was requested or fabricated",
        },
        "mask": {
            "path": str(mask_path.resolve()),
            "sha256": sha256_file(mask_path),
            "shape": list(masks.shape),
            "dtype": str(masks.dtype),
            "provenance": "box_rasterized_dynamic_ignore",
            "value_semantics": "1=static_keep,0=dynamic_ignore",
            "box_expansion": {
                "fraction_of_larger_box_side": BOX_EXPANSION_FRACTION,
                "minimum_source_px": BOX_EXPANSION_MIN_SOURCE_PX,
                "rationale": "retain a dynamic margin through DROID 8x mean pooling",
            },
            "coverage": {
                "mean_dynamic_ignore_fraction": float(np.mean(masks == 0)),
                "max_dynamic_ignore_fraction": float(np.max(np.mean(masks == 0, axis=(1, 2)))),
                "frames_with_dynamic_ignore": int(np.count_nonzero(np.any(masks == 0, axis=(1, 2)))),
                "frames_all_static": int(np.count_nonzero(np.all(masks == 1, axis=(1, 2)))),
            },
        },
        "frames": frames,
        "frame_store": source.frame_store_report(),
    }
    (args.run_root / "hands_mask_cache.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


def load_hands_masks(cache_root: Path, source: OpenCvFrameSource, geometry: Mapping[str, object]) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    report_path = cache_root / "hands_mask_cache.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Hands cache report missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != CACHE_SCHEMA:
        raise ValueError("unsupported Hands cache schema")
    source_report = report.get("source")
    schedule = report.get("schedule")
    mask_report = report.get("mask")
    if not isinstance(source_report, Mapping) or not isinstance(schedule, Mapping) or not isinstance(mask_report, Mapping):
        raise ValueError("Hands cache lacks source/schedule/mask provenance")
    if source_report.get("sha256") != source.timeline.source_sha256 or schedule.get("source_indices") != list(source_indices()):
        raise ValueError("Hands cache source identity or exact 256-frame schedule does not match probe")
    if report.get("droid_input_geometry") != geometry:
        raise ValueError("Hands cache DROID geometry differs from probe")
    mask_path = Path(str(mask_report.get("path", "")))
    if not mask_path.is_file() or sha256_file(mask_path) != mask_report.get("sha256"):
        raise ValueError("Hands cache mask path/hash is invalid")
    with np.load(mask_path, allow_pickle=False) as archive:
        masks = np.asarray(archive["masks"])
    expected_shape = (DROID_SERVICE_PUSH_CAPACITY, *geometry["shape_yx"])
    if masks.shape != expected_shape or masks.dtype != np.uint8 or not np.all((masks == 0) | (masks == 1)):
        raise ValueError("Hands cache mask tensor is not exact binary uint8 DROID-grid evidence")
    return {index: masks[position] for position, index in enumerate(source_indices())}, dict(mask_report)


def run_probe(args: argparse.Namespace) -> int:
    if args.run_root.exists():
        raise FileExistsError(f"fresh --run-root required: {args.run_root}")
    args.run_root.mkdir(parents=True)
    source, k_canonical = _validated_probe_source(args)
    selected = source_indices()
    source.build_frame_store(selected, spill_dir=args.run_root / "frame_store")
    driver, config = _driver(args.timeout_s)
    geometry = _geometry_report(source, config, k_canonical)
    condition = FACTORIAL_CONDITIONS[args.condition]
    effective_options = dict(DROID_DEFAULT_OPTIONS)
    effective_options["filter_thresh"] = float(condition["filter_thresh"])
    masks: dict[int, np.ndarray] | None = None
    mask_report: dict[str, object] = {"status": "not_used", "value_semantics": "no mask submitted"}
    if bool(condition["mask"]):
        assert args.hands_cache_root is not None
        masks, mask_report = load_hands_masks(args.hands_cache_root, source, geometry)
        mask_report = {"status": "submitted", **mask_report}
    report: dict[str, object] = {
        "schema": PROBE_SCHEMA,
        "condition": args.condition,
        "prediction": condition["prediction"],
        "source": {
            "path": str(args.input.resolve()), "sha256": sha256_file(args.input), "frame_count": source.timeline.frame_count,
            "fps": source.timeline.fps, "width_px": source.timeline.width_px, "height_px": source.timeline.height_px,
        },
        "canonical_k_source": {"path": str(args.canonical_k_npz.resolve()), "sha256": sha256_file(args.canonical_k_npz)},
        "K_canonical": k_canonical.tolist(),
        "droid_input_geometry": geometry,
        "schedule": {
            "source_indices": list(selected), "source_timestamps_s": [source.timeline.frames[index].timestamp_s for index in selected],
            "source_index_stride": 1, "push_count_requested": len(selected),
            "frame_id_semantics": "unchanged decimal provenance string of source index; numerically inert in resident service",
        },
        "forwarded_options": {"buffer": 256, "filter_thresh": float(condition["filter_thresh"])},
        "effective_session_options": effective_options,
        "mask": mask_report,
        "mode": {"scale_mode": "up_to_scale_monocular", "diagnostic_only": True, "acceptance": False},
        "contract": {"buffer": 256, "static_confidence_mask": "uint8 full DROID grid; 1=static_keep,0=dynamic_ignore", "rgb": "unchanged source-backed RGB"},
    }
    overall_started = time.monotonic()
    world: np.ndarray | None = None
    camera: np.ndarray | None = None
    try:
        create, pushes, final, traces = driver._run_droid_chunk(
            source, f"droid_factorial_{args.condition.lower()}", "task4_first2000", SimpleNamespace(k_canonical=k_canonical),
            (synthetic_inert_depth_record(source, k_canonical),), 0, selected,
            filter_thresh=float(condition["filter_thresh"]), static_confidence_masks=masks,
        )
        world, camera = np.asarray(final.output.T_world_camera.array), np.asarray(final.output.T_camera_world.array)
        report.update({
            "status": "succeeded_finite" if np.isfinite(world).all() and np.isfinite(camera).all() else "succeeded_nonfinite",
            "session_id": create.output.session_id, "push_count_accepted": len(pushes), "keyframe_count": final.output.keyframe_count,
            "finalize": {"result": "success", "T_world_camera": finite_summary(world), "T_camera_world": finite_summary(camera)},
            "timings_s": {"overall": float(time.monotonic() - overall_started), **{trace.stage_id: float(trace.completed_monotonic_s - trace.started_monotonic_s) for trace in traces}},
            "error_type": None, "error": None,
        })
    except DroidChunkFinalizeError as exc:
        report.update({
            "status": "failed_finalize", "session_id": exc.session_id, "push_count_accepted": len(exc.push_results),
            "keyframe_count": exc.push_results[-1].output.keyframe_count if exc.push_results else None,
            "finalize": {"result": "error", "T_world_camera": None, "T_camera_world": None},
            "timings_s": {"overall": float(time.monotonic() - overall_started), **{trace.stage_id: float(trace.completed_monotonic_s - trace.started_monotonic_s) for trace in exc.traces}},
            "error_type": exc.cause_type, "error": exc.cause_message,
        })
    except Exception as exc:
        report.update({
            "status": "failed_before_finalize_result", "session_id": None, "push_count_accepted": None, "keyframe_count": None,
            "finalize": {"result": "not_observed", "T_world_camera": None, "T_camera_world": None},
            "timings_s": {"overall": float(time.monotonic() - overall_started)}, "error_type": type(exc).__name__, "error": str(exc),
        })
    report["frame_store"] = source.frame_store_report()
    if world is not None and camera is not None:
        np.savez_compressed(args.run_root / "trajectory.npz", T_world_camera=world, T_camera_world=camera)
    (args.run_root / "probe_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["status"] == "succeeded_finite" else 1


def main() -> int:
    args = parse_args()
    return build_hands_cache(args) if args.mode == "cache" else run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
