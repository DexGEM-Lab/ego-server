#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

STATUS = "v18_occlusion_owner_graph"
NONE_OWNER = "__none__"
SUPPORT_STATE = "scene_depth_supports_foreground_occluder_candidate_owner_unaccepted"
CONTRADICTION_STATE = "scene_depth_contradicts_foreground_occluder_candidate"
METRIC_COMPATIBLE_STATE = "hand_scene_depth_metric_compatible_no_foreground_occluder_signal"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def finite_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def mesh_support(row: dict[str, Any]) -> float:
    support = row.get("mesh_contact_temporal_support")
    if isinstance(support, dict):
        return max(0.0, min(1.0, finite_float(support.get("max_support"), 0.0)))
    return 0.0


def load_depth_pair_evidence(path: Path) -> tuple[dict[tuple[int, str, str], dict[str, Any]], dict[tuple[int, str], Counter[str]]]:
    if not path.exists():
        return {}, {}
    report = load_json(path)
    pair_index: dict[tuple[int, str, str], dict[str, Any]] = {}
    row_counts: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    for raw_row in report.get("row_records", []):
        if not isinstance(raw_row, dict) or not isinstance(raw_row.get("frame_idx"), int):
            continue
        frame_idx = int(raw_row["frame_idx"])
        hand_side = str(raw_row.get("hand_side"))
        for raw_pair in raw_row.get("candidate_pair_depth_evidence", []):
            if not isinstance(raw_pair, dict):
                continue
            object_id = str(raw_pair.get("object_id"))
            state = str(raw_pair.get("depth_evidence_state"))
            pair_index[(frame_idx, hand_side, object_id)] = raw_pair
            row_counts[(frame_idx, hand_side)][state] += 1
    return pair_index, row_counts


def enrich_depth_evidence(row: dict[str, Any], pair_index: dict[tuple[int, str, str], dict[str, Any]], row_counts: dict[tuple[int, str], Counter[str]]) -> dict[str, Any]:
    frame_idx_raw = row.get("frame_idx")
    frame_idx = int(frame_idx_raw) if isinstance(frame_idx_raw, int) else -1
    hand_side = str(row.get("hand_side"))
    object_id = str(row.get("object_id"))
    pair = pair_index.get((frame_idx, hand_side, object_id), {})
    counts = row_counts.get((frame_idx, hand_side), Counter())
    state = str(pair.get("depth_evidence_state") or row.get("source_depth_order_state") or "missing_pair_depth_evidence")
    return {
        **row,
        "depth_pair_evidence_state": state,
        "depth_pair_evidence": pair if pair else None,
        "same_frame_depth_pair_state_counts": dict(sorted(counts.items())),
        "same_frame_foreground_contradiction_count": int(counts.get(CONTRADICTION_STATE, 0)),
        "same_frame_foreground_support_count": int(counts.get(SUPPORT_STATE, 0)),
    }


def acceptance_gate(row: dict[str, Any], assignment: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    state = str(row.get("depth_pair_evidence_state") or "missing_pair_depth_evidence")
    pair_raw = row.get("depth_pair_evidence")
    pair: dict[str, Any] = pair_raw if isinstance(pair_raw, dict) else {}
    selected = assignment is not None
    margin = finite_float(assignment.get("unary_energy_margin"), -999.0) if assignment is not None else None
    support = mesh_support(row)
    same_frame_contradictions = int(row.get("same_frame_foreground_contradiction_count") or 0)
    source_depth_order_resolved = bool(pair.get("depth_order_resolved") is True or row.get("depth_order_resolved") is True)
    source_occluder_owner_accepted = bool(pair.get("occluder_owner_accepted") is True or row.get("accepted_occlusion_owner") is True)
    exact_foreground_support = state == SUPPORT_STATE
    no_same_frame_contradiction = same_frame_contradictions == 0
    mesh_support_ok = support >= args.accept_mesh_support_min
    margin_ok = margin is not None and margin >= args.accept_energy_margin
    blockers: list[str] = []
    if not selected:
        blockers.append("temporal_graph_not_selected")
    if not exact_foreground_support:
        if state == CONTRADICTION_STATE:
            blockers.append("foreground_depth_contradicts_candidate")
        elif state == METRIC_COMPATIBLE_STATE:
            blockers.append("metric_depth_compatible_no_foreground_occluder_signal")
        else:
            blockers.append("foreground_depth_support_missing_or_untrusted")
    if not no_same_frame_contradiction:
        blockers.append("same_frame_foreground_depth_contradiction_present")
    if not mesh_support_ok:
        blockers.append("mesh_temporal_support_below_acceptance_threshold")
    if not margin_ok:
        blockers.append("temporal_graph_margin_below_acceptance_threshold")
    if not source_depth_order_resolved:
        blockers.append("source_depth_order_not_resolved")
    if not source_occluder_owner_accepted:
        blockers.append("source_occluder_owner_not_accepted")
    accepted = len(blockers) == 0
    return {
        "accepted_by_strict_depth_mesh_temporal_gate": accepted,
        "acceptance_blockers": sorted(set(blockers)),
        "selected_by_temporal_graph": selected,
        "exact_foreground_depth_support": exact_foreground_support,
        "depth_pair_evidence_state": state,
        "same_frame_foreground_contradiction_count": same_frame_contradictions,
        "mesh_temporal_support": support,
        "mesh_support_threshold": args.accept_mesh_support_min,
        "temporal_graph_margin": margin,
        "temporal_graph_margin_threshold": args.accept_energy_margin,
        "source_depth_order_resolved": source_depth_order_resolved,
        "source_occluder_owner_accepted": source_occluder_owner_accepted,
        "evidence_scope": "strict_gate_evaluates_depth_mesh_temporal_conditions_without_accepting_unresolved_source_depth_order",
    }


def candidate_energy(row: dict[str, Any]) -> float:
    iou = max(0.0, min(1.0, finite_float(row.get("bbox_iou"), 0.0)))
    coverage = max(0.0, min(1.0, finite_float(row.get("hand_box_coverage_by_object_box"), 0.0)))
    support = mesh_support(row)
    pair_raw = row.get("depth_pair_evidence")
    pair: dict[str, Any] = pair_raw if isinstance(pair_raw, dict) else {}
    depth_accepted = bool(row.get("accepted_occlusion_owner") is True or pair.get("occluder_owner_accepted") is True)
    depth_state = str(row.get("depth_pair_evidence_state") or pair.get("depth_evidence_state") or row.get("source_depth_order_state") or "unknown_depth_state")
    support_score = max(0.50 * coverage + 0.30 * iou + 0.20 * support, support * 0.75)
    energy = (1.0 - support_score) ** 2
    if depth_state == SUPPORT_STATE:
        energy *= 0.55
    elif depth_state == CONTRADICTION_STATE:
        energy += 0.75
    elif depth_state == METRIC_COMPATIBLE_STATE:
        energy += 0.15
    elif depth_state == "missing_pair_depth_evidence":
        energy += 0.20
    if not depth_accepted:
        energy += 0.35
    else:
        energy *= 0.35
    return float(energy)


def none_energy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    best = max(max(0.0, min(1.0, 0.5 * finite_float(r.get("hand_box_coverage_by_object_box"), 0.0) + 0.3 * finite_float(r.get("bbox_iou"), 0.0) + 0.2 * mesh_support(r))) for r in rows)
    return float(0.20 + 0.85 * best)


def transition_energy(prev: str, cur: str, switch_penalty: float, onoff_penalty: float) -> float:
    if prev == cur:
        return 0.0
    if prev == NONE_OWNER or cur == NONE_OWNER:
        return onoff_penalty
    return switch_penalty


def solve_sequence(hand_side: str, rows_by_frame: dict[int, list[dict[str, Any]]], args: argparse.Namespace) -> dict[str, Any]:
    frame_indices = sorted(rows_by_frame)
    if not frame_indices:
        return {"hand_side": hand_side, "assignments": [], "accepted_occlusion_owner_rows": 0, "selected_owner_rows": 0}
    row_by_state: dict[int, dict[str, dict[str, Any]]] = {}
    unary_by_frame: dict[int, dict[str, float]] = {}
    for frame_idx in frame_indices:
        best_rows: dict[str, dict[str, Any]] = {}
        for row in rows_by_frame[frame_idx]:
            object_id = str(row.get("object_id"))
            prev = best_rows.get(object_id)
            if prev is None or candidate_energy(row) < candidate_energy(prev):
                best_rows[object_id] = row
        row_by_state[frame_idx] = best_rows
        unary = {obj: candidate_energy(row) for obj, row in best_rows.items()}
        unary[NONE_OWNER] = none_energy(rows_by_frame[frame_idx])
        unary_by_frame[frame_idx] = unary
    costs: list[dict[str, float]] = [dict(unary_by_frame[frame_indices[0]])]
    back: list[dict[str, str | None]] = [{state: None for state in unary_by_frame[frame_indices[0]]}]
    for i in range(1, len(frame_indices)):
        cur_costs: dict[str, float] = {}
        cur_back: dict[str, str | None] = {}
        frame_gap = frame_indices[i] - frame_indices[i - 1]
        use_temporal_transition = frame_gap <= args.max_temporal_gap_frames
        for cur, unary in unary_by_frame[frame_indices[i]].items():
            best_prev = None
            best_cost = float("inf")
            for prev, prev_cost in costs[i - 1].items():
                temporal = transition_energy(prev, cur, args.object_switch_penalty, args.owner_onoff_penalty) if use_temporal_transition else 0.0
                total = prev_cost + temporal + unary
                if total < best_cost:
                    best_cost = total
                    best_prev = prev
            cur_costs[cur] = best_cost
            cur_back[cur] = best_prev
        costs.append(cur_costs)
        back.append(cur_back)
    final = min(costs[-1], key=lambda state: costs[-1][state])
    path = [final]
    for i in range(len(frame_indices) - 1, 0, -1):
        prev = back[i][path[-1]]
        path.append(prev if prev is not None else NONE_OWNER)
    path.reverse()
    assignments: list[dict[str, Any]] = []
    accepted = 0
    selected = 0
    for i, frame_idx in enumerate(frame_indices):
        chosen = path[i]
        energies = unary_by_frame[frame_idx]
        ranked = sorted(energies.items(), key=lambda kv: kv[1])
        next_best = next((energy for state, energy in ranked if state != chosen), None)
        margin = (float(next_best) - float(energies[chosen])) if next_best is not None else None
        row = row_by_state[frame_idx].get(chosen) if chosen != NONE_OWNER else None
        assignment_stub = {
            "unary_energy_margin": margin,
            "chosen_unary_energy": float(energies[chosen]),
            "next_best_unary_energy": float(next_best) if next_best is not None else None,
        }
        gate = acceptance_gate(row, assignment_stub, args) if row is not None else None
        accepted_row = bool(gate and gate.get("accepted_by_strict_depth_mesh_temporal_gate") is True)
        if chosen != NONE_OWNER:
            selected += 1
        if accepted_row:
            accepted += 1
        assignments.append(
            {
                "frame_idx": frame_idx,
                "hand_side": hand_side,
                "chosen_owner_object_id": None if chosen == NONE_OWNER else chosen,
                "chosen_unary_energy": float(energies[chosen]),
                "next_best_unary_energy": float(next_best) if next_best is not None else None,
                "unary_energy_margin": margin,
                "previous_candidate_frame_gap": (frame_idx - frame_indices[i - 1]) if i > 0 else None,
                "temporal_transition_applied": bool(i > 0 and (frame_idx - frame_indices[i - 1]) <= args.max_temporal_gap_frames),
                "accepted_occlusion_owner": accepted_row,
                "occlusion_owner_claim": "accepted_occlusion_owner_by_strict_depth_mesh_temporal_gate" if accepted_row else ("temporal_graph_selected_not_accepted" if chosen != NONE_OWNER else "no_occlusion_owner_selected"),
                "acceptance_gate": gate,
                "acceptance_blockers": gate.get("acceptance_blockers") if isinstance(gate, dict) else [],
                "depth_pair_evidence_state": gate.get("depth_pair_evidence_state") if isinstance(gate, dict) else None,
                "source_row": row,
                "candidate_unary_energies": {None if state == NONE_OWNER else state: float(energy) for state, energy in ranked},
            }
        )
    return {"hand_side": hand_side, "assignment_count": len(assignments), "selected_owner_rows": selected, "accepted_occlusion_owner_rows": accepted, "assignments": assignments}


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.occlusion_mesh_root / case / "v18_occlusion_mesh_owner_evidence_report.json"
    depth_path = args.occlusion_depth_root / case / "v18_occlusion_depth_order_evidence_report.json"
    source = load_json(source_path)
    depth_pair_index, depth_row_counts = load_depth_pair_evidence(depth_path)
    source_rows: list[dict[str, Any]] = []
    by_hand_frame: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for raw in source.get("rows", []):
        if not isinstance(raw, dict) or not isinstance(raw.get("frame_idx"), int):
            continue
        row = enrich_depth_evidence(raw, depth_pair_index, depth_row_counts)
        source_rows.append(row)
        by_hand_frame[str(row.get("hand_side"))][int(row["frame_idx"])].append(row)
    hand_graphs = [solve_sequence(hand_side, frames, args) for hand_side, frames in sorted(by_hand_frame.items())]
    assignment_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for graph in hand_graphs:
        for assignment in graph.get("assignments", []):
            if not isinstance(assignment, dict):
                continue
            chosen = assignment.get("chosen_owner_object_id")
            if chosen is not None:
                assignment_by_key[(int(assignment["frame_idx"]), str(assignment["hand_side"]), str(chosen))] = assignment
    annotated_rows: list[dict[str, Any]] = []
    selected = 0
    accepted = 0
    blocker_counts: Counter[str] = Counter()
    strict_candidate_rows = 0
    foreground_support_not_selected = 0
    for raw in source_rows:
        key = (int(raw["frame_idx"]), str(raw.get("hand_side")), str(raw.get("object_id")))
        assignment = assignment_by_key.get(key)
        selected_row = assignment is not None
        gate = acceptance_gate(raw, assignment, args)
        accepted_row = bool(gate.get("accepted_by_strict_depth_mesh_temporal_gate") is True)
        selected += int(selected_row)
        accepted += int(accepted_row)
        blocker_counts.update([str(v) for v in gate.get("acceptance_blockers", []) if isinstance(v, str)])
        if raw.get("depth_pair_evidence_state") == SUPPORT_STATE and mesh_support(raw) >= args.accept_mesh_support_min:
            strict_candidate_rows += 1
            if not selected_row:
                foreground_support_not_selected += 1
        annotated_rows.append({**raw, "temporal_graph_assignment": assignment, "selected_by_occlusion_graph": selected_row, "accepted_occlusion_owner": accepted_row, "occlusion_owner_claim": "accepted_occlusion_owner_by_strict_depth_mesh_temporal_gate" if accepted_row else ("temporal_graph_selected_not_accepted" if selected_row else "not_selected_by_occlusion_graph"), "acceptance_gate": gate, "acceptance_blockers": gate.get("acceptance_blockers")})
    out = {
        "method": "build_v18_occlusion_owner_graph",
        "status": STATUS,
        "claim": "Solves a temporal object-or-none occlusion-owner graph over bounded candidates with mesh-contact and exact pair-level depth-order evidence. It emits strict acceptance blockers and accepts ownership only if depth, mesh, temporal margin, and source depth-order acceptance all pass.",
        "case": case,
        "sources": {"occlusion_mesh_owner_evidence": str(source_path), "occlusion_depth_order_evidence": str(depth_path)},
        "parameters": {"object_switch_penalty": args.object_switch_penalty, "owner_onoff_penalty": args.owner_onoff_penalty, "accept_energy_margin": args.accept_energy_margin, "max_temporal_gap_frames": args.max_temporal_gap_frames, "accept_mesh_support_min": args.accept_mesh_support_min},
        "strict_acceptance_candidate_rows": strict_candidate_rows,
        "foreground_support_mesh_supported_not_selected_rows": foreground_support_not_selected,
        "acceptance_blocker_counts": dict(sorted(blocker_counts.items())),
        "hand_graphs": hand_graphs,
        "rows": annotated_rows,
        "selected_occlusion_owner_rows": selected,
        "accepted_occlusion_owner_rows": accepted,
        "occlusion_ownership_complete": False,
        "default_path_uses_bundlesdf_or_nerf": False,
        "annotation_ready": True,
        "deliverable_ready": True,
    }
    write_json(args.output_root / case / "v18_occlusion_owner_graph_report.json", out)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [build_case(case, args) for case in args.cases]
    summary = {"method": "build_v18_occlusion_owner_graph", "status": STATUS, "case_count": len(reports), "cases": [{"case": r["case"], "selected_occlusion_owner_rows": r["selected_occlusion_owner_rows"], "accepted_occlusion_owner_rows": r["accepted_occlusion_owner_rows"], "strict_acceptance_candidate_rows": r.get("strict_acceptance_candidate_rows"), "foreground_support_mesh_supported_not_selected_rows": r.get("foreground_support_mesh_supported_not_selected_rows"), "occlusion_ownership_complete": r["occlusion_ownership_complete"]} for r in reports], "claim_scope": "temporal_occlusion_owner_selection_with_strict_depth_mesh_gate_not_unsupported_acceptance"}
    write_json(args.output_root / "v18_occlusion_owner_graph_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--occlusion-mesh-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_mesh_owner_evidence"))
    parser.add_argument("--occlusion-depth-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_depth_order_evidence"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_occlusion_owner_graph"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--object-switch-penalty", type=float, default=0.40)
    parser.add_argument("--owner-onoff-penalty", type=float, default=0.25)
    parser.add_argument("--accept-energy-margin", type=float, default=0.25)
    parser.add_argument("--accept-mesh-support-min", type=float, default=0.50)
    parser.add_argument("--max-temporal-gap-frames", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
