#!/usr/bin/env python3
"""Run one full source video through the frozen typed API backend."""
from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path

from ego_annotation.api_backend import ApiBackend, ApiBackendConfig
from ego_annotation.fps_config import DEFAULT_FPS_CONDITION, FPS_CONDITION_BY_NAME, get_fps_condition
from ego_annotation.full_video_timeline import (
    FullVideoDriverConfig,
    FullVideoTimelineDriver,
    OpenCvFrameSource,
    preflight_single_video,
)
from ego_annotation.integrated_run_report import admission_events_path_from_environment, write_integrated_run_report
from ego_annotation.physical_adapters import PhysicalArtifactAdapter

# Capturing every Cosmos response for one item keeps later strict parse failures
# inspectable without expanding the much larger HaWoR representative captures.
COSMOS_STAGE_CAPTURE_LIMIT = 128

STAGES = (
    "unidepth.infer",
    "hands.detect",
    "wilor.reconstruct",
    "droid.create_session",
    "droid.push_frame",
    "droid.finalize",
    "hawor.infer_tracks",
    "hawor_infiller.fill",
    "cosmos3.reason",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-monocular",
        action="store_true",
        default=True,
        help="run DROID as diagnostic monocular (the annotation client has no native sensor depth)",
    )
    parser.add_argument("--timeout-s", type=float, default=86400.0)
    parser.add_argument("--service-origins-json", default="{}", help=argparse.SUPPRESS)
    parser.add_argument("--fps-condition", choices=sorted(FPS_CONDITION_BY_NAME), default=DEFAULT_FPS_CONDITION, help=argparse.SUPPRESS)
    # Human-friendly FPS downsampling multipliers: 0.5 means sample at half the
    # source FPS. These are converted into the internal fps_condition + absolute
    # FPS after the source timeline is known (see run()).
    parser.add_argument("--unidepth-fps-drop", type=float, default=None, help="UniDepth sampling as a fraction of source FPS (e.g. 0.5 = half)")
    parser.add_argument("--droid-fps-drop", type=float, default=None, help="DROID sampling as a fraction of source FPS (e.g. 0.5 = half)")
    parser.add_argument("--no-report", action="store_true", default=False, help="skip the integrated timing/batch report and package")
    return parser.parse_args(argv)


def cosmos_capture_index(capture_root: Path) -> dict[str, dict[str, str]]:
    """Index immutable raw Cosmos request/response captures by ownership scope."""
    captures: dict[str, dict[str, str]] = {}
    for manifest_path in sorted(capture_root.glob("cosmos3_reason/*/*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            scope = manifest["ownership"]["scope"]
            request_hash = manifest["request"]["sha256"]
            response_hash = manifest["response"]["sha256"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if all(isinstance(value, str) and value for value in (scope, request_hash, response_hash)):
            captures[scope] = {
                "request_sha256": request_hash,
                "response_sha256": response_hash,
                "manifest": str(manifest_path.relative_to(capture_root)),
            }
    return captures


def attach_cosmos_capture_hashes(entries: object, capture_root: Path, *, scope_key: str = "scope") -> list[dict[str, object]]:
    """Bind semantic attempts/anomaly rows to their immutable raw capture."""
    if not isinstance(entries, (list, tuple)):
        return []
    captures = cosmos_capture_index(capture_root)
    linked: list[dict[str, object]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        scope = entry.get(scope_key)
        if isinstance(scope, str) and scope in captures:
            entry["raw_response_capture"] = captures[scope]
            # Retain the earlier trace key for existing consumers while making
            # the anomaly ledger's raw-response relationship explicit.
            entry["capture"] = captures[scope]
        linked.append(entry)
    return linked


def parse_service_origins(raw: str) -> dict[str, str]:
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid --service-origins-json: {exc}") from exc
    if not isinstance(payload, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in payload.items()):
        raise ValueError("--service-origins-json must be a JSON object with string keys and values")
    unknown = sorted(set(payload) - set(STAGES))
    if unknown:
        raise ValueError(f"unknown service origin stage IDs: {unknown}")
    return {str(key): str(value) for key, value in payload.items()}


def write_frame_store_report(source: OpenCvFrameSource, run_root: Path) -> dict[str, object]:
    report = source.frame_store_report()
    # Linux reports ru_maxrss in KiB. Production runs are Linux-only.
    report["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    path = run_root / "frame_store_decode_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def annotation_client_driver_config(*, fps_condition: str, frame_store_spill_dir: Path) -> FullVideoDriverConfig:
    """Select the truthful DROID capability mode for RGB + predicted depth input."""

    return FullVideoDriverConfig(
        require_rgbd_capability=False,
        allow_monocular_droid_smoke=True,
        cosmos_enabled=True,
        fps_condition=fps_condition,
        frame_store_spill_dir=str(frame_store_spill_dir),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    preflight = preflight_single_video(args.input, case_id=args.case_id, fresh_root=args.run_root)
    source = OpenCvFrameSource.from_video(args.input)
    fps_condition = get_fps_condition(args.fps_condition)
    # Human-friendly FPS-drop overrides: convert fractional drops (0.5 = half)
    # into absolute target FPS using the inspected source FPS, then override
    # the config so the pipeline honors them exactly.
    unidepth_fps_override: float | None = fps_condition.unidepth_fps
    droid_fps_override: float | None = fps_condition.droid_fps
    if getattr(args, "unidepth_fps_drop", None) is not None:
        drop = float(args.unidepth_fps_drop)
        if drop <= 0 or drop > 1:
            raise ValueError("--unidepth-fps-drop must be in (0, 1]; e.g. 0.5 = half the source FPS")
        unidepth_fps_override = round(source.timeline.fps * drop, 6)
    if getattr(args, "droid_fps_drop", None) is not None:
        drop = float(args.droid_fps_drop)
        if drop <= 0 or drop > 1:
            raise ValueError("--droid-fps-drop must be in (0, 1]; e.g. 0.5 = half the source FPS")
        droid_fps_override = round(source.timeline.fps * drop, 6)
    config = annotation_client_driver_config(
        fps_condition=fps_condition.name,
        frame_store_spill_dir=args.run_root / "frame_store",
    )
    if unidepth_fps_override is not None:
        object.__setattr__(config, "unidepth_fps", float(unidepth_fps_override))
    if droid_fps_override is not None:
        object.__setattr__(config, "droid_fps", float(droid_fps_override))
    service_origins = parse_service_origins(args.service_origins_json)
    stage_backend = ApiBackend(
        ApiBackendConfig(
            base_url="http://127.0.0.1",
            timeout_s=args.timeout_s,
            cosmos_enabled=True,
            service_origins=service_origins,
            stage_capture_root=str(args.run_root / "stage_captures"),
            stage_capture_limits={"cosmos3.reason": COSMOS_STAGE_CAPTURE_LIMIT},
        )
    )
    from ego_annotation.full_video_timeline import LiveFrozenApiStageClient

    args.run_root.mkdir(parents=True, exist_ok=False)
    driver = FullVideoTimelineDriver(LiveFrozenApiStageClient(stage_backend), config)
    # The manager may point every logical stage at its shared local admission
    # proxy. The typed DAG and native stage scheduling remain unchanged.
    state = driver.run(source, case_id=args.case_id)
    write_frame_store_report(source, args.run_root)
    render_started = time.monotonic()
    result = PhysicalArtifactAdapter().render(state, source, args.run_root)
    module_timings_s = {str(key): float(value) for key, value in getattr(state, "module_timings_s", {}).items()}
    module_timings_s["render"] = float(time.monotonic() - render_started)
    module_timing_breakdown_s = {
        str(key): dict(value) for key, value in getattr(state, "module_timing_breakdown_s", {}).items()
    }
    module_timing_breakdown_s["render"] = {
        "client_prepare_s": 0.0,
        "transport_wait_s": 0.0,
        "client_decode_postprocess_s": 0.0,
        "local_assembly_write_s": float(getattr(result, "local_draw_assembly_s", 0.0)),
        "local_write_encode_s": float(getattr(result, "local_write_encode_s", 0.0)),
        "total_wall_s": float(getattr(result, "total_wall_s", module_timings_s["render"])),
        "request_count": 0,
    }
    timing_notes = dict(getattr(state, "module_timing_breakdown_notes", {}))
    timing_notes.update({
        "transport_wait_s": "client-observed HTTP send/receive duration; includes queue, network, server compute, and response read; not pure server compute",
        "total_wall_s": "stage wall span; concurrent per-request sums may exceed this",
        "unavailable_fields": "zero means no boundary was available and the module_timing_breakdown_notes entry explains why",
        "manager_queue_wait_s": "not separately instrumentable from this client; included in transport_wait_s when manager HTTP is used",
    })
    frame_store_write_started = time.monotonic()
    frame_store_report = write_frame_store_report(source, args.run_root)
    frame_store_write_s = time.monotonic() - frame_store_write_started
    if "frame_store" in module_timing_breakdown_s:
        module_timing_breakdown_s["frame_store"]["local_assembly_write_s"] = float(module_timing_breakdown_s["frame_store"].get("local_assembly_write_s", 0.0)) + frame_store_write_s
    semantic_review_path = args.run_root / "state" / "semantic_clips" / "v22_cosmos_semantic_review.json"
    semantic_review_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_review = dict(state.semantic_review)
    semantic_review["attempts"] = attach_cosmos_capture_hashes(
        semantic_review.get("attempts"), args.run_root / "stage_captures"
    )
    semantic_review["anomaly_ledger"] = attach_cosmos_capture_hashes(
        semantic_review.get("anomaly_ledger"), args.run_root / "stage_captures", scope_key="request_scope"
    )
    semantic_review_path.write_text(json.dumps(semantic_review, indent=2, ensure_ascii=True), encoding="utf-8")
    hawor_geometry_diagnostics = state.hawor_geometry_diagnostics
    performance = {
        "request_traces": [asdict(trace) for trace in state.batch_request_traces],
        "unidepth_request_count": len(state.unidepth_records),
        "hands_request_count": len(state.hands_records),
        "wilor_request_count": len(state.wilor_records),
        "droid_create_count": len(state.droid_records.create_results),
        "droid_push_count_by_attempt": [len(rows) for rows in state.droid_records.push_results_by_attempt],
        "droid_finalize_count": len(state.droid_records.finalize_results),
        "hawor_track_request_count": len(state.hawor_records),
        "hawor_infiller_request_count": sum(1 for window in state.infiller_windows if window.submitted),
        "cosmos_request_count": state.semantic_request_count,
        "frame_store_decode": frame_store_report,
        "module_timings_s": module_timings_s,
        "module_timing_breakdown_s": module_timing_breakdown_s,
        "module_timing_breakdown_notes": timing_notes,
        "timing_schema": "client_timing_breakdown.v1; transport_wait_s is end-to-end HTTP wait, not server-only compute",
        "render_timing": {"local_draw_assembly_s": float(getattr(result, "local_draw_assembly_s", 0.0)), "local_write_encode_s": float(getattr(result, "local_write_encode_s", 0.0)), "total_wall_s": float(getattr(result, "total_wall_s", module_timings_s["render"]))},
        "sampling": {"condition": fps_condition.name, "unidepth_fps": fps_condition.unidepth_fps, "droid_fps": fps_condition.droid_fps, "droid_submission": "uniform_endpoint_inclusive_chunked_stitch_interpolate" if fps_condition.droid_fps is not None else "exact_source_prefix_capped_at_256"},
        "droid_chunks": [
            {
                "chunk_index": outcome.chunk_index,
                "source_indices": list(outcome.source_indices),
                "session_id": outcome.session_id,
                "keyframe_count": outcome.keyframe_count,
                "stitch_boundary_translation_error_m": outcome.stitch_boundary_translation_error_m,
                "stitch_boundary_rotation_error_rad": outcome.stitch_boundary_rotation_error_rad,
                "attempts": [attempt.to_wire() for attempt in outcome.attempts],
            }
            for outcome in getattr(state.droid_records, "chunk_outcomes", ())
        ],
    }
    service_batch_traces = list(getattr(stage_backend, "service_batch_traces", ()))
    performance["service_batch_traces"] = service_batch_traces
    performance["service_batch_trace_status"] = "available" if service_batch_traces else "unavailable_without_complete_service_batch_trace"
    # Partial DROID coverage is a distinct physical-state condition, not a
    # generic warning: downstream observers use this exact status to keep the
    # unsubmitted pose tail explicit.
    pipeline_status = (
        "completed_with_partial_camera_coverage"
        if state.droid_records.coverage.partial
        else "completed_with_warnings"
        if state.semantic_status == "completed_with_anomalies"
        else "ok"
        if state.acceptance.accepted
        else "diagnostic_or_uncertain"
    )
    payload = {
        "status": pipeline_status,
        "case_id": args.case_id,
        "run_root": str(args.run_root.resolve()),
        "frame_count": result.frame_count,
        "duration_s": result.duration_s,
        "acceptance": {
            "accepted": state.acceptance.accepted,
            "diagnostic_only": state.acceptance.diagnostic_only,
            "scale_mode": state.acceptance.scale_mode,
            "reasons": list(state.acceptance.reasons),
        },
        "renders": {"combined": result.combined_video},
        "physical_state": result.state_npz,
        "report": result.report_json,
        "preflight_checks": list(preflight.checks),
        "cosmos": {
            "status": state.semantic_status,
            "request_count": state.semantic_request_count,
            "semantic_row_count": len(state.semantic_rows),
            "anomaly_count": int(semantic_review.get("anomaly_count", 0)),
            "anomaly_ledger": list(semantic_review.get("anomaly_ledger", ())),
            "review_json": str(semantic_review_path),
            "captioned_combined_video": result.combined_video,
        },
        "hand_geometry_diagnostics": hawor_geometry_diagnostics,
        "droid": {**state.droid_records.coverage.to_wire(), "chunks": performance["droid_chunks"]},
        "item_batch_size": 1,
        "service_origins": service_origins,
        "fps_sampling": {"condition": fps_condition.name, "unidepth_fps": fps_condition.unidepth_fps, "droid_fps": fps_condition.droid_fps, "droid_submission": "uniform_endpoint_inclusive_chunked_stitch_interpolate" if fps_condition.droid_fps is not None else "exact_source_prefix_capped_at_256", "source_timeline_preserved": True},
        "module_timings_s": module_timings_s,
        "module_timing_breakdown_s": module_timing_breakdown_s,
        "module_timing_breakdown_notes": timing_notes,
        "performance": performance,
    }
    run_result_path = args.run_root / "run_result.json"
    manifest_path = args.run_root / "annotation_pipeline_manifest.json"
    report_paths = write_integrated_run_report(
        args.run_root,
        case_id=args.case_id,
        performance=performance,
        service_batch_traces=service_batch_traces,
        artifacts={
            "run_result": str(run_result_path),
            "pipeline_manifest": str(manifest_path),
            "combined_video": result.combined_video,
            "physical_state": result.state_npz,
            "physical_report": result.report_json,
        },
        admission_events_path=admission_events_path_from_environment(),
    )
    performance["integrated_run_report"] = report_paths
    payload["integrated_run_report"] = report_paths
    run_result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest = {
        "schema": "ego.annotation.pipeline_manifest.v1",
        "pipeline": "single_video_api_ify",
        "status": payload["status"],
        "case_id": args.case_id,
        "frame_count": result.frame_count,
        "duration_s": result.duration_s,
        "acceptance": payload["acceptance"],
        "renders": {
            "v22_combined": result.combined_video,
            "render_source": "PhysicalArtifactAdapter",
        },
        "physical_state": result.state_npz,
        "physical_report": result.report_json,
        "run_result": str(run_result_path),
        "integrated_run_report": report_paths,
        "integrated_run_report_path": report_paths["json"],
        "stage_captures": str(args.run_root / "stage_captures"),
        "stage_capture_index": str(args.run_root / "stage_captures" / "fixture_index.json"),
        "semantic_review": str(semantic_review_path),
        "cosmos": payload["cosmos"],
        "hand_geometry_diagnostics": payload["hand_geometry_diagnostics"],
        "droid": payload["droid"],
        "item_batch_size": 1,
        "service_origins": service_origins,
        "fps_sampling": payload["fps_sampling"],
        "module_timings_s": module_timings_s,
        "module_timing_breakdown_s": module_timing_breakdown_s,
        "module_timing_breakdown_notes": timing_notes,
        "timing_schema": "client_timing_breakdown.v1",
        "performance": performance,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run(args)
    except Exception as exc:
        args.run_root.mkdir(parents=True, exist_ok=True)
        failure_trace = attach_cosmos_capture_hashes(
            getattr(exc, "attempts", ()), args.run_root / "stage_captures"
        )
        trace_path: Path | None = None
        if failure_trace:
            trace_path = args.run_root / "state" / "semantic_clips" / "v22_cosmos_semantic_failure_trace.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps({
                "schema": "v22_cosmos_semantic_failure_trace.v1",
                "attempts": list(failure_trace),
                "repair_count": sum(1 for attempt in failure_trace if attempt.get("repair_count") == 1),
            }, indent=2, ensure_ascii=True), encoding="utf-8")
        error_payload: dict[str, object] = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        if trace_path is not None:
            error_payload["cosmos_failure_trace"] = str(trace_path)
            error_payload["repair_count"] = sum(1 for attempt in failure_trace if attempt.get("repair_count") == 1)
        (args.run_root / "run_error.json").write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
        raise
    print(json.dumps(payload, indent=2))
    report = payload.get("integrated_run_report")
    if isinstance(report, dict):
        print(f"FINAL_REPORT_PATH={report.get('json')}")
    renders = payload.get("renders")
    if isinstance(renders, dict):
        print(f"FINAL_VIDEO_PATH={renders.get('combined')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
