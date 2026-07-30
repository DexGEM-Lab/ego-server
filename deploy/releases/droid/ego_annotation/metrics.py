"""Metric-vector aggregation for ego.annotation.output v1."""
from __future__ import annotations

import math
from statistics import mean, median
from typing import Any, Iterable

from ego_annotation.schema import METRIC_VECTOR, MetricSpec


def _finite_values(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def rmse(values: list[float]) -> float:
    if not values:
        return float("nan")
    return math.sqrt(sum(v * v for v in values) / len(values))


def summarize(values: list[float], summaries: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(values)}
    if not values:
        return result
    for summary in summaries:
        if summary == "p50":
            result[summary] = float(median(values))
        elif summary == "p95":
            result[summary] = float(percentile(values, 95.0))
        elif summary == "p90":
            result[summary] = float(percentile(values, 90.0))
        elif summary == "mean":
            result[summary] = float(mean(values))
        elif summary == "rmse":
            result[summary] = float(rmse(values))
        elif summary == "coverage":
            result[summary] = float(sum(1 for v in values if v > 0.0) / max(1, len(values)))
    return result


def _values_for_metric(metric_id: str, observations: dict[str, Any], throughput_rows: list[dict[str, Any]]) -> list[float]:
    direct = observations.get(metric_id)
    if isinstance(direct, dict) and "values" in direct:
        return _finite_values(direct.get("values") or [])
    if isinstance(direct, (list, tuple)):
        return _finite_values(direct)
    if isinstance(direct, (int, float)):
        return _finite_values([direct])

    if metric_id.startswith("throughput_") or metric_id == "explicit_failure_rate":
        key = metric_id.replace("throughput_", "")
        if metric_id == "throughput_module_speed_x":
            values = []
            for row in throughput_rows:
                if "module_speed_x" in row:
                    values.append(row["module_speed_x"])
                else:
                    duration = row.get("input_duration_s")
                    elapsed = row.get("elapsed_s")
                    try:
                        duration_f = float(duration)
                        elapsed_f = float(elapsed)
                    except (TypeError, ValueError):
                        continue
                    if duration_f > 0 and elapsed_f > 0:
                        values.append(duration_f / elapsed_f)
            return _finite_values(values)
        if metric_id == "explicit_failure_rate":
            values = []
            for row in throughput_rows:
                if "failed" in row:
                    values.append(1.0 if row.get("failed") else 0.0)
                elif "status" in row:
                    values.append(0.0 if str(row.get("status")) == "ok" else 1.0)
            return _finite_values(values)
        return _finite_values(row.get(key) for row in throughput_rows)

    return []


def _metric_metadata(metric_id: str, observations: dict[str, Any]) -> dict[str, Any]:
    direct = observations.get(metric_id)
    if not isinstance(direct, dict):
        return {}
    return {key: value for key, value in direct.items() if key != "values"}


def build_metric_rows(
    observations: dict[str, Any],
    throughput_rows: list[dict[str, Any]],
    *,
    calibration_status: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in METRIC_VECTOR:
        values = _values_for_metric(spec.metric_id, observations, throughput_rows)
        row = metric_row(spec, values)
        metadata = _metric_metadata(spec.metric_id, observations)
        status_override = metadata.get("status")
        if isinstance(status_override, str) and (values or status_override != "measured"):
            row["status"] = status_override
        for key in ("measurement_role", "source", "claim_scope", "evaluator_status"):
            if key in metadata:
                row[key] = metadata[key]
        if spec.axis in {"head_camera", "projection"}:
            row["calibration_status"] = calibration_status
        if not values:
            row["next_required_measurement"] = str(metadata.get("next_required_measurement") or next_required_measurement(spec))
        rows.append(row)
    return rows


def metric_row(spec: MetricSpec, values: list[float]) -> dict[str, Any]:
    return {
        "metric_id": spec.metric_id,
        "axis": spec.axis,
        "unit": spec.unit,
        "ideal_target": spec.ideal_target,
        "description": spec.description,
        "status": "measured" if values else "unmeasured",
        "summary": summarize(values, spec.summaries),
    }


def next_required_measurement(spec: MetricSpec) -> str:
    if spec.axis == "head_camera":
        return "fixed-gauge camera/head GT or metric VIO/SLAM/head-pose metadata with known extrinsics"
    if spec.axis == "hand":
        return "HOT3D or equivalent MANO GT/evaluator rows for wrist/root, all-joint MPJPE, and surface"
    if spec.axis == "projection":
        return "final fused hand state plus independent 2D/crop evidence under canonical K"
    if spec.axis == "visibility":
        return "per-frame visible/partial/occluded/out-of-frame labels or validated detector evidence"
    if spec.axis == "semantic":
        return "full-timeline semantic clips with grounding evidence frames and confidence"
    if spec.axis == "throughput":
        return "module benchmark rows with elapsed time, input duration, queue wait, batch fill, residency, and failures"
    return "axis-specific evaluator input"


def throughput_forecast(throughput_rows: list[dict[str, Any]], target_video_hours_per_week: float = 10000.0) -> dict[str, Any]:
    speeds = _values_for_metric("throughput_module_speed_x", {}, throughput_rows)
    gpu_hours = _values_for_metric("throughput_gpu_hours_per_video_hour", {}, throughput_rows)
    target_realtime_aggregate = float(target_video_hours_per_week) / (7.0 * 24.0)
    measured_realtime = sum(speeds)
    if measured_realtime > 0:
        active_worker_equivalent = target_realtime_aggregate / measured_realtime
    else:
        active_worker_equivalent = None
    return {
        "target_video_hours_per_week": float(target_video_hours_per_week),
        "target_realtime_aggregate_x": target_realtime_aggregate,
        "measured_module_speed_x_sum": measured_realtime,
        "measured_module_speed_x_count": len(speeds),
        "estimated_active_worker_equivalent_for_target": active_worker_equivalent,
        "mean_gpu_hours_per_video_hour": float(mean(gpu_hours)) if gpu_hours else None,
        "status": "measured" if speeds else "unmeasured",
    }
