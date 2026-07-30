#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

STATUS = "v18_contact_ownership_graph"
NONE_OWNER = "__none__"


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


def finite_distance(row: dict[str, Any]) -> float | None:
    value = row.get("min_hand_surface_to_v16_object_mesh_m")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def blockers(row: dict[str, Any]) -> list[str]:
    raw = row.get("blockers")
    return [str(x) for x in raw] if isinstance(raw, list) else []


def row_source_penalty(row: dict[str, Any]) -> float:
    state = str(row.get("source_contact_state"))
    evidence_raw = row.get("source_contact_evidence")
    evidence: dict[str, Any] = evidence_raw if isinstance(evidence_raw, dict) else {}
    penalty = 0.0
    if evidence.get("metric_depth_compatible_candidate") is True:
        penalty -= 0.15
    if evidence.get("pair_contact_image_candidate") is True:
        penalty -= 0.10
    elif evidence.get("image_overlap_candidate") is True:
        penalty -= 0.05
    if state == "no_contact_image_evidence":
        penalty += 0.25
    elif state == "unobserved_pair":
        penalty += 0.40
    elif "rejected" in state or "contradiction" in state:
        # Existing depth predicates are useful but were shown to contradict sub-millimeter mesh evidence.
        penalty += 0.15
    return penalty


def association_iou(row: dict[str, Any]) -> float:
    v16_match_raw = row.get("v16_mesh_match")
    v16_match: dict[str, Any] = v16_match_raw if isinstance(v16_match_raw, dict) else {}
    return max(0.0, min(1.0, finite_float(v16_match.get("bbox_iou"), 0.0)))


def candidate_energy(row: dict[str, Any], contact_tolerance_m: float) -> float:
    dist = finite_distance(row)
    row_blockers = blockers(row)
    if dist is None or row_blockers:
        return 8.0 + float(len(row_blockers))
    tol = max(1e-6, contact_tolerance_m)
    support = max(0.0, min(1.0, finite_float(row.get("mesh_contact_support_score"), 0.0)))
    bbox_iou = association_iou(row)
    mesh_term = (dist / tol) ** 2
    support_term = (1.0 - support) ** 2
    low_association_penalty = max(0.0, 0.20 - bbox_iou) * 0.5
    return float(max(0.0, mesh_term + support_term + row_source_penalty(row) + low_association_penalty))


def none_energy(rows: list[dict[str, Any]]) -> float:
    finite_rows = [r for r in rows if finite_distance(r) is not None and not blockers(r)]
    if not finite_rows:
        return 0.15
    best_support = max(max(0.0, min(1.0, finite_float(r.get("mesh_contact_support_score"), 0.0))) for r in finite_rows)
    finite_dists = [d for r in finite_rows for d in [finite_distance(r)] if d is not None]
    best_dist = min(finite_dists)
    close_bonus = max(0.0, 1.0 - float(best_dist) / 0.02)
    return float(0.25 + 1.75 * best_support + 0.75 * close_bonus)


def transition_energy(prev: str, cur: str, switch_penalty: float, onoff_penalty: float) -> float:
    if prev == cur:
        return 0.0
    if prev == NONE_OWNER or cur == NONE_OWNER:
        return onoff_penalty
    return switch_penalty


def solve_hand_sequence(hand_side: str, frame_rows: dict[int, list[dict[str, Any]]], args: argparse.Namespace) -> dict[str, Any]:
    frame_indices = sorted(frame_rows)
    state_rows_by_frame: dict[int, dict[str, dict[str, Any]]] = {}
    unary_by_frame: dict[int, dict[str, float]] = {}
    for frame_idx in frame_indices:
        rows = frame_rows[frame_idx]
        best_by_object: dict[str, dict[str, Any]] = {}
        for row in rows:
            object_id = str(row.get("object_id"))
            prev = best_by_object.get(object_id)
            if prev is None or candidate_energy(row, args.contact_tolerance_m) < candidate_energy(prev, args.contact_tolerance_m):
                best_by_object[object_id] = row
        state_rows_by_frame[frame_idx] = best_by_object
        unary = {obj: candidate_energy(row, args.contact_tolerance_m) for obj, row in best_by_object.items()}
        unary[NONE_OWNER] = none_energy(rows)
        unary_by_frame[frame_idx] = unary

    if not frame_indices:
        return {"hand_side": hand_side, "assignments": [], "objective": {}, "factor_counts": {}}

    costs: list[dict[str, float]] = []
    back: list[dict[str, str | None]] = []
    first = frame_indices[0]
    costs.append(dict(unary_by_frame[first]))
    back.append({state: None for state in unary_by_frame[first]})
    for i in range(1, len(frame_indices)):
        frame_idx = frame_indices[i]
        prev_states = costs[i - 1]
        cur_costs: dict[str, float] = {}
        cur_back: dict[str, str | None] = {}
        for cur, unary in unary_by_frame[frame_idx].items():
            best_prev: str | None = None
            best_cost = float("inf")
            for prev, prev_cost in prev_states.items():
                total = prev_cost + transition_energy(prev, cur, args.object_switch_penalty, args.contact_onoff_penalty) + unary
                if total < best_cost:
                    best_cost = total
                    best_prev = prev
            cur_costs[cur] = best_cost
            cur_back[cur] = best_prev
        costs.append(cur_costs)
        back.append(cur_back)

    final_state = min(costs[-1], key=lambda state: costs[-1][state])
    path: list[str] = [final_state]
    for i in range(len(frame_indices) - 1, 0, -1):
        prev = back[i][path[-1]]
        path.append(prev if prev is not None else NONE_OWNER)
    path.reverse()

    assignments: list[dict[str, Any]] = []
    accepted_count = 0
    selected_object_count = 0
    for i, frame_idx in enumerate(frame_indices):
        chosen = path[i]
        energies = unary_by_frame[frame_idx]
        ranked = sorted(energies.items(), key=lambda kv: kv[1])
        chosen_unary = energies[chosen]
        next_best = next((energy for state, energy in ranked if state != chosen), None)
        margin = (float(next_best) - float(chosen_unary)) if next_best is not None else None
        row = state_rows_by_frame[frame_idx].get(chosen) if chosen != NONE_OWNER else None
        dist = finite_distance(row) if row else None
        row_blockers = blockers(row) if row else []
        assoc_iou = association_iou(row) if row else 0.0
        accepted = bool(
            chosen != NONE_OWNER
            and row is not None
            and dist is not None
            and dist <= args.contact_tolerance_m
            and assoc_iou >= args.association_accept_iou
            and not row_blockers
            and margin is not None
            and margin >= args.accept_energy_margin
        )
        if chosen != NONE_OWNER:
            selected_object_count += 1
        if accepted:
            accepted_count += 1
        assignments.append(
            {
                "frame_idx": frame_idx,
                "hand_side": hand_side,
                "chosen_owner_object_id": None if chosen == NONE_OWNER else chosen,
                "chosen_unary_energy": float(chosen_unary),
                "next_best_unary_energy": float(next_best) if next_best is not None else None,
                "unary_energy_margin": margin,
                "accepted_contact_owner": accepted,
                "contact_owner_claim": "accepted_contact_owner_by_temporal_mesh_distance_graph" if accepted else ("temporal_graph_selected_not_accepted" if chosen != NONE_OWNER else "no_contact_owner_selected"),
                "min_hand_surface_to_object_mesh_m": dist,
                "v16_to_v18_mesh_association_iou": assoc_iou,
                "association_accept_iou": args.association_accept_iou,
                "source_row_blockers": row_blockers,
                "candidate_unary_energies": {None if state == NONE_OWNER else state: float(energy) for state, energy in ranked},
                "nonpenetration_status": "unsigned_surface_distance_only_signed_nonpenetration_unresolved",
            }
        )

    none_baseline = sum(unary_by_frame[f][NONE_OWNER] for f in frame_indices)
    greedy_unary = sum(min(unary_by_frame[f].values()) for f in frame_indices)
    path_unary = sum(unary_by_frame[f][path[i]] for i, f in enumerate(frame_indices))
    path_transition = sum(transition_energy(path[i - 1], path[i], args.object_switch_penalty, args.contact_onoff_penalty) for i in range(1, len(path)))
    return {
        "hand_side": hand_side,
        "frame_count": len(frame_indices),
        "assignment_count": len(assignments),
        "selected_object_frame_count": selected_object_count,
        "accepted_contact_owner_rows": accepted_count,
        "assignments": assignments,
        "objective": {
            "none_baseline_energy": float(none_baseline),
            "greedy_unary_energy": float(greedy_unary),
            "inferred_path_energy": float(path_unary + path_transition),
            "path_unary_energy": float(path_unary),
            "path_temporal_energy": float(path_transition),
            "energy_units": "mixed_metric_distance_tolerance_unary_plus_temporal_switch_penalty",
        },
        "factor_counts": {
            "contact_owner_unary_factors": int(sum(len(unary_by_frame[f]) for f in frame_indices)),
            "contact_owner_temporal_factors": max(0, len(frame_indices) - 1),
        },
    }


def build_case(case: str, args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.mesh_contact_root / case / "v18_mesh_contact_evidence_report.json"
    source = load_json(source_path)
    rows = [row for row in source.get("rows", []) if isinstance(row, dict)]
    by_hand_frame: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        frame_idx = row.get("frame_idx")
        if not isinstance(frame_idx, int):
            continue
        by_hand_frame[str(row.get("hand_side"))][frame_idx].append(row)

    hand_graphs = [solve_hand_sequence(hand_side, frame_rows, args) for hand_side, frame_rows in sorted(by_hand_frame.items())]
    assignment_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for graph in hand_graphs:
        for assignment in graph.get("assignments", []):
            if not isinstance(assignment, dict):
                continue
            chosen = assignment.get("chosen_owner_object_id")
            if chosen is not None:
                assignment_by_key[(int(assignment["frame_idx"]), str(assignment["hand_side"]), str(chosen))] = assignment

    annotated_rows: list[dict[str, Any]] = []
    accepted_rows = 0
    selected_rows = 0
    for row in rows:
        frame_idx = row.get("frame_idx")
        hand_side = str(row.get("hand_side"))
        object_id = str(row.get("object_id"))
        assignment = assignment_by_key.get((int(frame_idx), hand_side, object_id)) if isinstance(frame_idx, int) else None
        is_selected = assignment is not None
        accepted = bool(assignment and assignment.get("accepted_contact_owner") is True)
        selected_rows += int(is_selected)
        accepted_rows += int(accepted)
        annotated_rows.append(
            {
                "frame_idx": frame_idx,
                "hand_side": hand_side,
                "object_id": object_id,
                "source_contact_state": row.get("source_contact_state"),
                "min_hand_surface_to_v16_object_mesh_m": row.get("min_hand_surface_to_v16_object_mesh_m"),
                "mesh_contact_support_score": row.get("mesh_contact_support_score"),
                "v16_mesh_match": row.get("v16_mesh_match"),
                "blockers": row.get("blockers"),
                "graph_assignment": assignment,
                "selected_by_contact_graph": is_selected,
                "accepted_contact_owner": accepted,
                "contact_owner_claim": "accepted_contact_owner_by_temporal_mesh_distance_graph" if accepted else ("temporal_graph_selected_not_accepted" if is_selected else "not_selected_by_contact_graph"),
                "nonpenetration_status": "unsigned_surface_distance_only_signed_nonpenetration_unresolved",
            }
        )

    factor_counts: dict[str, int] = defaultdict(int)
    for graph in hand_graphs:
        for key, value in graph.get("factor_counts", {}).items():
            factor_counts[str(key)] += int(value)
    out = {
        "method": "build_v18_contact_ownership_graph",
        "status": STATUS,
        "claim": "Solves a hand-level discrete contact-owner graph over object-or-none states using V16 MANO-to-object mesh distances, image/depth evidence penalties, and temporal continuity. It accepts only rows with close metric distance, no blockers, and an energy margin; signed nonpenetration remains unresolved.",
        "case": case,
        "sources": {"mesh_contact_evidence": str(source_path)},
        "parameters": {
            "contact_tolerance_m": args.contact_tolerance_m,
            "accept_energy_margin": args.accept_energy_margin,
            "association_accept_iou": args.association_accept_iou,
            "object_switch_penalty": args.object_switch_penalty,
            "contact_onoff_penalty": args.contact_onoff_penalty,
        },
        "variable_semantics": "contact_owner[hand,frame] in {none} union candidate V18 object ids",
        "factor_semantics": ["metric_surface_distance_unary", "image_depth_evidence_unary", "object_or_none_temporal_continuity"],
        "hand_graphs": hand_graphs,
        "rows": annotated_rows,
        "contact_graph_selected_rows": selected_rows,
        "contact_ownership_accepted_rows": accepted_rows,
        "contact_ownership_complete": False,
        "signed_nonpenetration_solved": False,
        "factor_counts": dict(sorted(factor_counts.items())),
        "default_path_uses_bundlesdf_or_nerf": False,
        "annotation_ready": True,
        "deliverable_ready": True,
    }
    write_json(args.output_root / case / "v18_contact_ownership_graph_report.json", out)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [build_case(case, args) for case in args.cases]
    summary = {
        "method": "build_v18_contact_ownership_graph",
        "status": STATUS,
        "case_count": len(reports),
        "cases": [
            {
                "case": report["case"],
                "contact_graph_selected_rows": report["contact_graph_selected_rows"],
                "contact_ownership_accepted_rows": report["contact_ownership_accepted_rows"],
                "signed_nonpenetration_solved": report["signed_nonpenetration_solved"],
            }
            for report in reports
        ],
        "claim_scope": "partial_temporal_mesh_distance_contact_ownership_not_full_signed_nonpenetration",
    }
    write_json(args.output_root / "v18_contact_ownership_graph_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-contact-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_mesh_contact_evidence"))
    parser.add_argument("--output-root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_contact_ownership_graph"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    parser.add_argument("--contact-tolerance-m", type=float, default=0.01)
    parser.add_argument("--accept-energy-margin", type=float, default=0.25)
    parser.add_argument("--association-accept-iou", type=float, default=0.15)
    parser.add_argument("--object-switch-penalty", type=float, default=0.55)
    parser.add_argument("--contact-onoff-penalty", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
