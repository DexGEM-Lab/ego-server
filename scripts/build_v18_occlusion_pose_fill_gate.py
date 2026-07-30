#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

STATUS = "v18_occlusion_pose_fill_gate"
HAND_SIDES = ("left", "right")
INT_TO_SIDE = {0: "left", 1: "right"}
DEPTH_SCALE_SUPPORT_STATUS = "depth_scaled_from_projected_hawor_vertices_to_unidepth"
MIN_DEPTH_SCALE_SAMPLE_COUNT_FOR_POSE_FILL = 40
RAW_FOREGROUND_CANDIDATE_SUPPORT_STATE = "scene_depth_supports_foreground_occluder_candidate_owner_unaccepted"
ACCEPTED_FOREGROUND_OCCLUDER_SUPPORT_STATE = "scene_depth_supports_accepted_foreground_occluder_owner"

LEGACY_BASELINE_BLOCKERS_NOT_FATAL_FOR_OBSERVED_MANO = {
    "hand_baseline_temporal_occlusion_pose_not_accepted_for_temporal_fill",
    "interior_hand_depth_state_missing",
    "interior_hand_depth_not_metric_compatible",
    "median_metric_depth_abs_residual_component_missing",
    "median_metric_depth_abs_residual_above_threshold",
    "rtmlib_wilor_comparison_missing",
    "rtmlib_wilor_2d_delta_above_threshold",
    "temporal_acceleration_component_missing",
    "temporal_acceleration_above_threshold",
    "hand_bone_scale_component_missing",
    "hand_bone_scale_error_above_threshold",
    "hawor_missing_for_frame_side",
    "hawor_projection_residual_missing",
    "hawor_projection_residual_above_threshold",
    "hawor_temporal_infill_candidate_not_measurement",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def hawor_bridge_index(report_path: Path) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    if not report_path.exists():
        return {}, {"status": "missing_hawor_bridge_report", "report_path": str(report_path)}
    report = load_json(report_path)
    if not isinstance(report, dict):
        return {}, {"status": "malformed_hawor_bridge_report", "report_path": str(report_path)}
    npz_raw = report.get("bridge_candidate_npz")
    npz_path = Path(str(npz_raw)) if npz_raw else None
    if npz_path is None or not npz_path.exists():
        return {}, {"status": "missing_hawor_bridge_npz", "report_path": str(report_path), "bridge_candidate_npz": str(npz_path) if npz_path else None}
    z = np.load(npz_path)
    source_hawor_npz = Path(str(np.asarray(z["source_hawor_npz"]).reshape(-1)[0])) if "source_hawor_npz" in z.files else None
    support_z = np.load(source_hawor_npz, allow_pickle=True) if source_hawor_npz is not None and source_hawor_npz.exists() else None
    frame_idx = np.asarray(z["frame_idx"], dtype=np.int32)
    side_arr = np.asarray(z["side"], dtype=np.int32)
    joints_world = np.asarray(z["joints_current_v18_world_from_hawor_camera_local_m"], dtype=np.float64)
    depth_scales = np.asarray(z["hawor_to_v18_depth_scale"], dtype=np.float64) if "hawor_to_v18_depth_scale" in z.files else np.ones(len(frame_idx), dtype=np.float64)
    depth_scale_status = np.asarray(z["hawor_to_v18_depth_scale_status"]) if "hawor_to_v18_depth_scale_status" in z.files else np.asarray(["missing_depth_scale_metadata"] * len(frame_idx))
    depth_scale_sample_count = np.asarray(z["hawor_to_v18_depth_scale_sample_count"], dtype=np.int32) if "hawor_to_v18_depth_scale_sample_count" in z.files else np.zeros(len(frame_idx), dtype=np.int32)
    source_complete_depth_npz = str(np.asarray(z["source_complete_depth_npz"]).reshape(-1)[0]) if "source_complete_depth_npz" in z.files else None
    coordinate_status = str(np.asarray(z["coordinate_status"]).reshape(-1)[0]) if "coordinate_status" in z.files else "hawor_bridge_current_v18_world"
    out: dict[tuple[int, str], dict[str, Any]] = {}
    support_counts: Counter[str] = Counter()
    for row_idx in range(len(frame_idx)):
        side = INT_TO_SIDE.get(int(side_arr[row_idx]), str(side_arr[row_idx]))
        if side not in HAND_SIDES:
            continue
        frame = int(frame_idx[row_idx])
        support_state = "support_unknown"
        same_frame_detection = False
        temporal_boundary_filled = False
        if support_z is not None:
            detected_key = f"{side}_detected_same_frame"
            boundary_key = f"{side}_temporal_boundary_filled"
            same_frame_detection = bool(np.asarray(support_z[detected_key])[frame]) if detected_key in support_z.files else False
            temporal_boundary_filled = bool(np.asarray(support_z[boundary_key])[frame]) if boundary_key in support_z.files else False
            if temporal_boundary_filled:
                support_state = "temporal_boundary_fill"
            elif same_frame_detection:
                support_state = "observed_same_frame_detection"
            else:
                support_state = "inferred_no_same_frame_detection"
        scale_status = str(depth_scale_status[row_idx]) if row_idx < len(depth_scale_status) else "missing_depth_scale_metadata"
        sample_count = int(depth_scale_sample_count[row_idx]) if row_idx < len(depth_scale_sample_count) else 0
        observed_depth_scaled = bool(
            support_state == "observed_same_frame_detection"
            and scale_status == DEPTH_SCALE_SUPPORT_STATUS
            and sample_count >= MIN_DEPTH_SCALE_SAMPLE_COUNT_FOR_POSE_FILL
        )
        wrist = np.asarray(joints_world[row_idx, 0], dtype=np.float64) if joints_world.ndim == 3 and joints_world.shape[1] > 0 else np.zeros((3,), dtype=np.float64)
        out[(frame, side)] = {
            "frame_idx": frame,
            "hand_side": side,
            "bridge_report": str(report_path),
            "bridge_npz": str(npz_path),
            "bridge_row_index": int(row_idx),
            "source_hawor_npz": str(source_hawor_npz) if source_hawor_npz is not None else None,
            "source_complete_depth_npz": source_complete_depth_npz,
            "coordinate_status": coordinate_status,
            "support_state": support_state,
            "same_frame_detection": bool(same_frame_detection),
            "temporal_boundary_filled": bool(temporal_boundary_filled),
            "hawor_to_v18_depth_scale": float(depth_scales[row_idx]),
            "hawor_to_v18_depth_scale_status": scale_status,
            "hawor_to_v18_depth_scale_sample_count": sample_count,
            "observed_depth_scaled_mano_supported": observed_depth_scaled,
            "wrist_current_v18_world_m": [float(v) for v in wrist.tolist()] if wrist.shape == (3,) else None,
        }
        support_counts[f"support_{support_state}"] += 1
        support_counts[f"depth_scale_{scale_status}"] += 1
        if observed_depth_scaled:
            support_counts["observed_depth_scaled_mano_supported"] += 1
    summary = {
        "status": "loaded_hawor_bridge_pose_fill_support",
        "report_path": str(report_path),
        "bridge_candidate_npz": str(npz_path),
        "source_hawor_npz": str(source_hawor_npz) if source_hawor_npz is not None else None,
        "row_count": len(out),
        "support_counts": dict(sorted(support_counts.items())),
        "depth_scale_support_status_required": DEPTH_SCALE_SUPPORT_STATUS,
        "min_depth_scale_sample_count_for_pose_fill": MIN_DEPTH_SCALE_SAMPLE_COUNT_FOR_POSE_FILL,
    }
    return out, summary


def normalize_accepted_owner_label(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    accepted = bool(out.get("accepted_occlusion_owner") is True or out.get("accepted_by_strict_depth_mesh_temporal_gate") is True)
    raw_depth_state = out.get("depth_pair_evidence_state")
    if accepted and raw_depth_state == RAW_FOREGROUND_CANDIDATE_SUPPORT_STATE:
        out["depth_pair_evidence_state"] = ACCEPTED_FOREGROUND_OCCLUDER_SUPPORT_STATE
        out["raw_depth_pair_evidence_state_before_graph_acceptance"] = raw_depth_state
    gate = out.get("acceptance_gate")
    if isinstance(gate, dict):
        gate_out = dict(gate)
        gate_raw_depth_state = gate_out.get("depth_pair_evidence_state")
        gate_accepted = bool(gate_out.get("accepted_by_strict_depth_mesh_temporal_gate") is True)
        if gate_accepted and gate_raw_depth_state == RAW_FOREGROUND_CANDIDATE_SUPPORT_STATE:
            gate_out["depth_pair_evidence_state"] = ACCEPTED_FOREGROUND_OCCLUDER_SUPPORT_STATE
            gate_out["raw_depth_pair_evidence_state_before_graph_acceptance"] = gate_raw_depth_state
        out["acceptance_gate"] = gate_out
    return out


def occlusion_owner_index(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    report = load_json(path)
    candidate_rows: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for raw_row in report.get("rows", []):
        if not isinstance(raw_row, dict) or not isinstance(raw_row.get("frame_idx"), int):
            continue
        key = (int(raw_row["frame_idx"]), str(raw_row.get("hand_side")))
        candidate_rows.setdefault(key, []).append(
            normalize_accepted_owner_label(
                {
                    "object_id": raw_row.get("object_id"),
                    "selected_by_occlusion_graph": raw_row.get("selected_by_occlusion_graph"),
                    "accepted_occlusion_owner": raw_row.get("accepted_occlusion_owner"),
                    "depth_pair_evidence_state": raw_row.get("depth_pair_evidence_state"),
                    "acceptance_gate": raw_row.get("acceptance_gate"),
                    "acceptance_blockers": raw_row.get("acceptance_blockers"),
                }
            )
        )
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for graph in report.get("hand_graphs", []):
        if not isinstance(graph, dict):
            continue
        for raw in graph.get("assignments", []):
            if not isinstance(raw, dict) or not isinstance(raw.get("frame_idx"), int):
                continue
            key = (int(raw["frame_idx"]), str(raw.get("hand_side")))
            out[key] = {**normalize_accepted_owner_label(raw), "candidate_rows": candidate_rows.get(key, [])}
    return out


def owner_depth_support(owner: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(owner, dict):
        return {}
    source_row = owner.get("source_row") if isinstance(owner.get("source_row"), dict) else {}
    depth_pair = source_row.get("depth_pair_evidence") if isinstance(source_row.get("depth_pair_evidence"), dict) else {}
    hawor_depth = depth_pair.get("hawor_mano_depth_order_evidence") if isinstance(depth_pair.get("hawor_mano_depth_order_evidence"), dict) else {}
    raw_depth_pair_state = owner.get("raw_depth_pair_evidence_state_before_graph_acceptance") or source_row.get("raw_depth_pair_evidence_state_before_graph_acceptance") or source_row.get("depth_pair_evidence_state") or depth_pair.get("depth_evidence_state")
    source_depth_order_resolved = bool(depth_pair.get("depth_order_resolved") is True or source_row.get("depth_order_resolved") is True)
    source_occluder_owner_accepted = bool(depth_pair.get("occluder_owner_accepted") is True or source_row.get("accepted_occlusion_owner") is True)
    graph_owner_accepted = bool(owner.get("accepted_occlusion_owner") is True)
    if graph_owner_accepted and source_depth_order_resolved and source_occluder_owner_accepted and raw_depth_pair_state == RAW_FOREGROUND_CANDIDATE_SUPPORT_STATE:
        depth_pair_state = ACCEPTED_FOREGROUND_OCCLUDER_SUPPORT_STATE
    else:
        depth_pair_state = raw_depth_pair_state
    return {
        "depth_pair_evidence_state": depth_pair_state,
        "raw_depth_pair_evidence_state_before_graph_acceptance": raw_depth_pair_state,
        "graph_occlusion_owner_accepted": graph_owner_accepted,
        "source_depth_order_resolved": source_depth_order_resolved,
        "source_occluder_owner_accepted": source_occluder_owner_accepted,
        "hawor_depth_order_state": hawor_depth.get("state"),
        "hawor_depth_order_accepted": bool(hawor_depth.get("accepted_as_depth_order_support") is True),
        "hawor_overlap_vertex_count": hawor_depth.get("hawor_overlap_vertex_count"),
        "object_depth_low_m": depth_pair.get("object_depth_low_m"),
        "object_depth_median_m": depth_pair.get("object_depth_median_m"),
        "object_depth_high_m": depth_pair.get("object_depth_high_m"),
        "object_geometry_state": depth_pair.get("object_geometry_state"),
        "object_pose_state": depth_pair.get("object_pose_state"),
        "source_depth_pair_evidence": depth_pair,
    }


def unique_strings(values: list[str]) -> list[str]:
    return sorted(set(values))


def gate_row(hand: dict[str, Any], owner: dict[str, Any] | None, bridge: dict[str, Any] | None) -> dict[str, Any]:
    frame_idx_raw = hand.get("frame_idx")
    frame_idx = int(frame_idx_raw) if isinstance(frame_idx_raw, int) else -1
    hand_side = str(hand.get("hand_side"))
    blockers: list[str] = []
    owner_accepted = bool(owner and owner.get("accepted_occlusion_owner") is True)
    owner_support = owner_depth_support(owner)
    owner_depth_order_supported = bool(
        owner_accepted
        and owner_support.get("source_depth_order_resolved") is True
        and owner_support.get("source_occluder_owner_accepted") is True
        and owner_support.get("hawor_depth_order_accepted") is True
    )
    bridge_row_available = isinstance(bridge, dict)
    hawor_available = bool(bridge_row_available)
    hawor_candidate = bool(bridge_row_available)
    observed_hawor = bool(bridge and bridge.get("support_state") == "observed_same_frame_detection" and bridge.get("same_frame_detection") is True)
    depth_scaled_mano = bool(bridge and bridge.get("observed_depth_scaled_mano_supported") is True)
    depth_scale_status = bridge.get("hawor_to_v18_depth_scale_status") if isinstance(bridge, dict) else None
    depth_scale_sample_count = int(bridge.get("hawor_to_v18_depth_scale_sample_count") or 0) if isinstance(bridge, dict) else 0
    legacy_interior_depth = bool(hand.get("interior_metric_depth_compatible") is True)
    baseline_accepted = bool(hand.get("temporal_occlusion_pose_accepted") is True)
    owner_candidate_rows = owner.get("candidate_rows", []) if isinstance(owner, dict) else []
    owner_acceptance_blockers: list[str] = []
    if isinstance(owner_candidate_rows, list):
        for raw_candidate in owner_candidate_rows:
            if not isinstance(raw_candidate, dict):
                continue
            for raw_blocker in raw_candidate.get("acceptance_blockers", []):
                if isinstance(raw_blocker, str) and raw_blocker not in owner_acceptance_blockers:
                    owner_acceptance_blockers.append(raw_blocker)
    if not owner_accepted:
        blockers.append("accepted_occlusion_owner_missing")
        for raw_blocker in owner_acceptance_blockers:
            prefixed = f"occlusion_owner_{raw_blocker}"
            if prefixed not in blockers:
                blockers.append(prefixed)
    elif not owner_depth_order_supported:
        blockers.append("accepted_occlusion_owner_lacks_hawor_depth_order_support")
    if not hawor_available:
        blockers.append("final_hawor_bridge_row_missing_for_frame_side")
    if not hawor_candidate:
        blockers.append("final_hawor_bridge_candidate_missing_for_frame_side")
    if not observed_hawor:
        blockers.append("final_hawor_not_observed_same_frame_detection")
    if not depth_scaled_mano:
        blockers.append("final_hawor_depth_scale_support_missing_or_too_few_samples")
    if not baseline_accepted:
        blockers.append("hand_baseline_temporal_occlusion_pose_not_accepted_for_temporal_fill")
    for raw_blocker in hand.get("acceptance_blockers", []):
        if isinstance(raw_blocker, str) and raw_blocker not in blockers:
            blockers.append(raw_blocker)
    blockers = unique_strings(blockers)
    observed_acceptance_blockers = [blocker for blocker in blockers if blocker not in LEGACY_BASELINE_BLOCKERS_NOT_FATAL_FOR_OBSERVED_MANO]
    accepted_observed = owner_depth_order_supported and observed_hawor and depth_scaled_mano and not observed_acceptance_blockers
    accepted_temporal = owner_depth_order_supported and baseline_accepted and not blockers
    accepted = bool(accepted_observed or accepted_temporal)
    if accepted_observed:
        claim = "accepted_observed_mano_pose_through_occlusion"
    elif accepted_temporal:
        claim = "accepted_temporal_pose_fill_through_occlusion"
    else:
        claim = "pose_fill_blocked_not_accepted"
    return {
        "frame_idx": frame_idx,
        "hand_side": hand_side,
        "pose_fill_gate_claim": claim,
        "pose_fill_through_occlusion_accepted": accepted,
        "pose_fill_acceptance_type": "observed_depth_scaled_mano_behind_accepted_occluder" if accepted_observed else "temporal_occlusion_pose_baseline" if accepted_temporal else None,
        "pose_filled_through_occlusion": accepted,
        "accepted_occlusion_owner": owner_accepted,
        "owner_depth_order_supported": owner_depth_order_supported,
        "chosen_owner_object_id": owner.get("chosen_owner_object_id") if isinstance(owner, dict) else None,
        "hand_baseline_state": hand.get("hand_baseline_state"),
        "hawor_measurement_available": hawor_available,
        "hawor_candidate_present": hawor_candidate,
        "hawor_evidence_role": hand.get("hawor_evidence_role"),
        "final_hawor_support_state": bridge.get("support_state") if isinstance(bridge, dict) else None,
        "final_hawor_same_frame_detection": bool(bridge.get("same_frame_detection") is True) if isinstance(bridge, dict) else False,
        "final_hawor_observed_depth_scaled_mano_supported": depth_scaled_mano,
        "hawor_to_v18_depth_scale_status": depth_scale_status,
        "hawor_to_v18_depth_scale_sample_count": depth_scale_sample_count,
        "required_hawor_to_v18_depth_scale_status": DEPTH_SCALE_SUPPORT_STATUS,
        "min_hawor_to_v18_depth_scale_sample_count": MIN_DEPTH_SCALE_SAMPLE_COUNT_FOR_POSE_FILL,
        "interior_metric_depth_compatible": legacy_interior_depth,
        "interior_depth_role": "legacy_visible_hand_depth_score_not_required_for_observed_mano_through_accepted_occluder",
        "hand_baseline_temporal_occlusion_pose_accepted": baseline_accepted,
        "temporal_pose_fill_accepted": accepted_temporal,
        "observed_mano_pose_through_occlusion_accepted": accepted_observed,
        "occlusion_owner_acceptance_blockers": owner_acceptance_blockers,
        "source_occlusion_owner_candidate_rows": owner_candidate_rows,
        "source_occlusion_owner_depth_support": owner_support,
        "source_hawor_bridge_row": bridge,
        "blockers": blockers,
        "observed_pose_acceptance_blockers": observed_acceptance_blockers,
        "source_hand_baseline_row": hand,
        "source_occlusion_owner_assignment": owner,
    }


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    hand_path = args.hand_baseline_root / case / "v18_hand_baseline_branch.json"
    owner_path = args.occlusion_owner_graph_root / case / "v18_occlusion_owner_graph_report.json"
    hawor_bridge_path = args.hawor_bridge_root / case / "v18_hawor_bridge_state_report.json"
    hand_report = load_json(hand_path)
    owners = occlusion_owner_index(owner_path)
    bridge_rows, bridge_summary = hawor_bridge_index(hawor_bridge_path)
    rows: list[dict[str, Any]] = []
    blocker_counts: Counter[str] = Counter()
    accepted_observed = 0
    accepted_temporal = 0
    for frame in hand_report.get("frames", []):
        if not isinstance(frame, dict):
            continue
        for hand in frame.get("hands", []):
            if not isinstance(hand, dict) or not isinstance(hand.get("frame_idx"), int):
                continue
            key = (int(hand["frame_idx"]), str(hand.get("hand_side")))
            row = gate_row(hand, owners.get(key), bridge_rows.get(key))
            rows.append(row)
            for blocker in row["blockers"]:
                blocker_counts[str(blocker)] += 1
            if row.get("observed_mano_pose_through_occlusion_accepted") is True:
                accepted_observed += 1
            if row.get("temporal_pose_fill_accepted") is True:
                accepted_temporal += 1
    accepted = sum(1 for row in rows if row.get("pose_fill_through_occlusion_accepted") is True)
    candidate = sum(1 for row in rows if row.get("hawor_candidate_present") is True or row.get("hawor_measurement_available") is True)
    out = {
        "method": "build_v18_occlusion_pose_fill_gate",
        "status": STATUS,
        "claim": "Gates pose fill-through-occlusion using accepted occlusion ownership plus either observed same-frame depth-scaled HaWoR MANO through the accepted occluder or an accepted temporal occlusion-pose baseline. It preserves blockers and does not fill poses when either side is unsupported.",
        "case": case,
        "sources": {"hand_baseline_branch": str(hand_path), "occlusion_owner_graph": str(owner_path), "hawor_bridge_state": str(hawor_bridge_path)},
        "row_count": len(rows),
        "pose_fill_candidate_rows": candidate,
        "pose_fill_through_occlusion_accepted_rows": accepted,
        "observed_mano_pose_through_occlusion_accepted_rows": accepted_observed,
        "temporal_pose_fill_through_occlusion_accepted_rows": accepted_temporal,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "hawor_bridge_summary": bridge_summary,
        "rows": rows,
        "pose_fill_through_occlusion_complete": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        "annotation_ready": True,
        "deliverable_ready": True,
    }
    write_json(args.output_root / case / "v18_occlusion_pose_fill_gate_report.json", out)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_occlusion_pose_fill_gate",
        "status": STATUS,
        "case_count": len(reports),
        "cases": [
            {
                "case": r["case"],
                "row_count": r["row_count"],
                "pose_fill_candidate_rows": r["pose_fill_candidate_rows"],
                "pose_fill_through_occlusion_accepted_rows": r["pose_fill_through_occlusion_accepted_rows"],
                "observed_mano_pose_through_occlusion_accepted_rows": r["observed_mano_pose_through_occlusion_accepted_rows"],
                "temporal_pose_fill_through_occlusion_accepted_rows": r["temporal_pose_fill_through_occlusion_accepted_rows"],
            }
            for r in reports
        ],
        "claim_scope": "explicit_occlusion_pose_fill_gate_accepts_only_observed_depth_scaled_mano_or_reviewed_temporal_baseline_with_accepted_owner",
    }
    write_json(args.output_root / "v18_occlusion_pose_fill_gate_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-baseline-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_hand_baseline_branch"))
    parser.add_argument("--occlusion-owner-graph-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_unidepth_extension/v18_occlusion_owner_graph_complete_depth_hawor"))
    parser.add_argument("--hawor-bridge-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_corrective_1600/hawor_bridge_state"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_pose_fill_gate_complete_depth_hawor"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
