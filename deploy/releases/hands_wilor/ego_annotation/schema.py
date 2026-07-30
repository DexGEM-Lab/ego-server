"""Stable product schema constants for annotation output bundles."""
from __future__ import annotations

from dataclasses import dataclass

SCHEMA_NAME = "ego.annotation.output"
SCHEMA_VERSION = "1.0.0-alpha"
PUBLIC_ANNOTATION_ENDPOINT = "/v1/annotation-jobs"

COORDINATE_FRAMES = ["image_px", "camera_t", "world_w0", "head_t", "mano_left", "mano_right"]

PARQUET_TABLES = [
    "frames",
    "head_camera",
    "hand_states",
    "semantic_clips",
    "validation_metrics",
]

NDJSON_STREAMS = ["overlay_events", "caption_events", "provenance", "errors"]


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    axis: str
    unit: str
    summaries: tuple[str, ...]
    ideal_target: str
    description: str


METRIC_VECTOR: tuple[MetricSpec, ...] = (
    MetricSpec(
        "head_camera_ate_translation_m",
        "head_camera",
        "m",
        ("p50", "p95", "rmse"),
        "0.005 m",
        "Fixed-gauge camera/head absolute trajectory translation error; no per-clip Sim(3) fitting.",
    ),
    MetricSpec(
        "head_camera_rpe_translation_m",
        "head_camera",
        "m",
        ("p50", "p95", "rmse"),
        "0.005 m",
        "Fixed-gauge relative pose translation error.",
    ),
    MetricSpec(
        "head_camera_rotation_deg",
        "head_camera",
        "deg",
        ("p50", "p95", "rmse"),
        "reported toward 5mm equivalent projection effect",
        "Camera/head rotation error under known extrinsics or fixed-gauge GT.",
    ),
    MetricSpec(
        "head_camera_scale_error_ratio",
        "head_camera",
        "ratio",
        ("p50", "p95", "rmse"),
        "1.0",
        "Metric scale error without per-clip scale fitting.",
    ),
    MetricSpec(
        "hand_wrist_root_error_m",
        "hand",
        "m",
        ("p50", "p95", "rmse"),
        "0.005 m",
        "Wrist/root camera-frame metric error.",
    ),
    MetricSpec(
        "hand_all_joint_mpjpe_m",
        "hand",
        "m",
        ("p50", "p95", "rmse"),
        "0.005 m",
        "All-joint MANO MPJPE; first protected priority after wrist/root.",
    ),
    MetricSpec(
        "hand_root_relative_mpjpe_m",
        "hand",
        "m",
        ("p50", "p95", "rmse"),
        "0.005 m",
        "Root-relative articulation error for shape/pose separation.",
    ),
    MetricSpec(
        "hand_mpvpe_surface_m",
        "hand",
        "m",
        ("p50", "p95", "rmse"),
        "0.005 m",
        "Hand surface/MPVPE where GT or surface correspondence exists.",
    ),
    MetricSpec(
        "hand_reprojection_error_px",
        "projection",
        "px",
        ("p50", "p95", "rmse"),
        "minimize with calibrated K",
        "Final fused hand layer projection residual against independent visible evidence.",
    ),
    MetricSpec(
        "visibility_state_accuracy",
        "visibility",
        "ratio",
        ("mean",),
        "1.0",
        "Visible/partial/occluded/out-of-frame/unresolved state accuracy where labels exist.",
    ),
    MetricSpec(
        "temporal_wrist_jitter_m_per_frame",
        "temporal",
        "m/frame",
        ("p50", "p95", "rmse"),
        "minimize without dragging off detector evidence",
        "Frame-to-frame wrist displacement after final fusion.",
    ),
    MetricSpec(
        "temporal_root_rotation_jitter_deg_per_frame",
        "temporal",
        "deg/frame",
        ("p50", "p95", "rmse"),
        "minimize without hiding source switches",
        "Frame-to-frame root orientation change after final fusion.",
    ),
    MetricSpec(
        "semantic_segment_duration_s",
        "semantic",
        "s",
        ("p50", "p95", "coverage"),
        "mostly 2-3 s segments with full timeline coverage",
        "Semantic clip duration and coverage compliance.",
    ),
    MetricSpec(
        "semantic_grounding_score",
        "semantic",
        "ratio",
        ("mean", "p50"),
        "1.0",
        "Caption entity/action grounding against visible evidence frames or reviewer consensus.",
    ),
    MetricSpec(
        "throughput_module_speed_x",
        "throughput",
        "realtime_x",
        ("p50", "p95", "mean"),
        "59.5 aggregate realtime per active module for 10k video-hours/week",
        "Module processing speed relative to input duration.",
    ),
    MetricSpec(
        "throughput_gpu_hours_per_video_hour",
        "throughput",
        "gpu_h/video_h",
        ("mean", "p95"),
        "capacity model input",
        "GPU-hours consumed per input video-hour by active lane.",
    ),
    MetricSpec(
        "throughput_queue_wait_s",
        "throughput",
        "s",
        ("p50", "p95"),
        "bounded by service SLO",
        "Queue wait before module execution.",
    ),
    MetricSpec(
        "throughput_batch_fill_ratio",
        "throughput",
        "ratio",
        ("mean", "p50"),
        "near saturated under batch load",
        "Batch fill efficiency for resident GPU workers.",
    ),
    MetricSpec(
        "throughput_worker_residency_ratio",
        "throughput",
        "ratio",
        ("mean", "p50"),
        "resident models amortize load cost",
        "Fraction of runtime with model actors warm and resident.",
    ),
    MetricSpec(
        "explicit_failure_rate",
        "throughput",
        "ratio",
        ("mean",),
        "0.0 silent failures; explicit failures only",
        "Fraction of jobs or modules ending in explicit error states.",
    ),
)
