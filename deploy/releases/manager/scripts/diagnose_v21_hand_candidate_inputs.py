#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"expected_json_object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def module_probe(name: str) -> dict[str, Any]:
    return {
        "module": name,
        "available_in_current_python": importlib.util.find_spec(name) is not None,
        "vantage": "current_orchestration_python_only",
        "global_availability_inference": "not_inferred",
    }


def path_probe(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "state": "not_configured", "exists": False, "error": None}
    try:
        exists = bool(path.exists())
    except PermissionError as exc:
        return {
            "path": str(path),
            "state": "undetermined_permission_denied_from_current_user",
            "exists": None,
            "error": f"permission_denied: {exc}",
            "global_availability_inference": "not_inferred_reprobe_from_execution_user_or_host",
        }
    except OSError as exc:
        return {
            "path": str(path),
            "state": "undetermined_os_error_from_current_user",
            "exists": None,
            "error": f"os_error: {exc}",
            "global_availability_inference": "not_inferred_reprobe_from_execution_user_or_host",
        }
    return {"path": str(path), "state": "present" if exists else "missing_from_current_vantage", "exists": exists, "error": None}


def state_is_present(probe: dict[str, Any]) -> bool:
    return probe.get("state") == "present"


def state_is_missing(probe: dict[str, Any]) -> bool:
    return probe.get("state") == "missing_from_current_vantage"


def state_is_undetermined(probe: dict[str, Any]) -> bool:
    return str(probe.get("state", "")).startswith("undetermined")


def summarize_path_group(group: dict[str, dict[str, Any]]) -> str:
    states = [probe.get("state") for probe in group.values()]
    if states and all(state == "present" for state in states):
        return "present_current_vantage"
    if any(str(state).startswith("undetermined") for state in states):
        return "undetermined_current_vantage"
    if any(state == "missing_from_current_vantage" for state in states):
        return "missing_current_vantage"
    return "not_configured_or_empty"


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_manifest = load_json(args.input_manifest)
    wilor_assets = {
        "wilor_final.ckpt": path_probe(args.wilor_root / "pretrained_models" / "wilor_final.ckpt") if args.wilor_root else path_probe(None),
        "detector.pt": path_probe(args.wilor_root / "pretrained_models" / "detector.pt") if args.wilor_root else path_probe(None),
        "model_config.yaml": path_probe(args.wilor_root / "pretrained_models" / "model_config.yaml") if args.wilor_root else path_probe(None),
        "mano_mean_params.npz": path_probe(args.wilor_root / "mano_data" / "mano_mean_params.npz") if args.wilor_root else path_probe(None),
    }
    hamer_checkpoint = path_probe(args.hamer_checkpoint)
    probes = {
        "rtmlib": {
            **module_probe("rtmlib"),
            "required_for": "2D hand keypoints / HaMeR boxes",
            "execution_availability_inference": "current Python result is only a local-vantage probe; actual runner env must be probed separately",
        },
        "wilor": {
            "module": module_probe("wilor"),
            "repo": path_probe(args.wilor_root),
            "assets": wilor_assets,
            "asset_group_state": summarize_path_group(wilor_assets),
            "required_for": "3D MANO hand candidate stream",
            "execution_availability_inference": "permission-denied or local module absence does not prove assets/model are missing on the actual execution host/user",
        },
        "hamer": {
            "module": module_probe("hamer"),
            "repo": path_probe(args.hamer_root),
            "checkpoint": hamer_checkpoint,
            "required_for": "HaMeR MANO candidate from 2D boxes",
        },
    }

    hard_local_blockers: list[str] = []
    undetermined_execution_requirements: list[str] = []
    local_env_gaps: list[str] = []

    if not probes["rtmlib"]["available_in_current_python"]:
        local_env_gaps.append("rtmlib_unavailable_in_current_orchestration_python")
    if not probes["wilor"]["module"]["available_in_current_python"]:
        local_env_gaps.append("wilor_unavailable_in_current_orchestration_python")
    if probes["wilor"]["asset_group_state"] == "undetermined_current_vantage":
        undetermined_execution_requirements.append("wilor_assets_unreadable_from_current_user_reprobe_from_execution_user_or_host")
    elif probes["wilor"]["asset_group_state"] == "missing_current_vantage":
        hard_local_blockers.append("wilor_assets_missing_from_current_vantage")
    if not probes["hamer"]["module"]["available_in_current_python"]:
        local_env_gaps.append("hamer_unavailable_in_current_orchestration_python")
    if state_is_missing(hamer_checkpoint):
        hard_local_blockers.append("hamer_checkpoint_missing_from_readable_local_path")
    elif state_is_undetermined(hamer_checkpoint):
        undetermined_execution_requirements.append("hamer_checkpoint_unreadable_from_current_user_reprobe_from_execution_user_or_host")

    if hard_local_blockers or local_env_gaps or undetermined_execution_requirements:
        status = "hand_candidate_execution_not_run_environment_unresolved"
    else:
        status = "ready_for_hand_candidate_generation_in_current_environment"

    report = {
        "schema": "v21_hand_candidate_input_diagnosis.v1",
        "status": status,
        "method": "diagnose_v21_hand_candidate_inputs",
        "case_id": input_manifest.get("case_id"),
        "input_manifest": str(args.input_manifest),
        "probes": probes,
        "hard_local_blockers": hard_local_blockers,
        "undetermined_execution_requirements": undetermined_execution_requirements,
        "local_env_gaps_not_global_blockers": local_env_gaps,
        "next_required_action": (
            "probe_or_run_hand_generators_from_the_actual_execution_environment; do_not_reinstall_WiLoR_based_only_on_current_user_permission_denied"
            if undetermined_execution_requirements
            else "install_or_point_current_environment_to_missing_local_hand_assets_then_run_candidate_generators"
            if hard_local_blockers or local_env_gaps
            else "run_rtmlib_wilor_hamer_candidates"
        ),
        "claim_scope": "Hand candidate dependency diagnosis only. Local permission-denied/current-python module absence is not treated as proof of global model absence. No hand state, MANO state, contact, or nonpenetration is produced by this report.",
    }
    write_json(args.output_report, report)
    state_path = args.run_root / "state" / "v21_physical_state.json"
    if state_path.exists():
        state = load_json(state_path)
        state["hands"] = {
            "state": "metric_mano_candidate_not_run_environment_unresolved",
            "diagnosis_report": str(args.output_report),
            "metric_mano_available": False,
            "contact_nonpenetration_enabled": False,
            "hard_local_blockers": hard_local_blockers,
            "undetermined_execution_requirements": undetermined_execution_requirements,
            "local_env_gaps_not_global_blockers": local_env_gaps,
        }
        write_json(state_path, state)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose V21 hand candidate generator availability without creating hand state.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--wilor-root", type=Path, default=Path("/mnt/user-home/yiwen/ego_annotation_remote/repo/third_party/WiLoR"))
    parser.add_argument("--hamer-root", type=Path, default=Path("/mnt/user-home/zjh/ego-pipeline/hamer"))
    parser.add_argument("--hamer-checkpoint", type=Path, default=Path("/mnt/user-home/zjh/ego-pipeline/hamer/checkpoints/hamer.ckpt"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
