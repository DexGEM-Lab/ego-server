#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def validate_case(path: Path) -> dict[str, Any]:
    report = load_json(path)
    case = str(report.get("case"))
    frame_count = int(report.get("frame_count", -1))
    observation_rows = int(report.get("observation_rows", -1))
    full_rows = report.get("rows")
    direct_rows = report.get("direct_observation_rows")
    require(frame_count > 0, f"{case}: invalid frame count")
    require(report.get("camera_depth_correction_variable_ready") is True, f"{case}: correction not ready")
    require(report.get("camera_depth_correction_complete") is False, f"{case}: overclaims complete camera/depth correction")
    require(isinstance(full_rows, list) and len(full_rows) == frame_count, f"{case}: full timeline rows mismatch")
    require(isinstance(direct_rows, list) and len(direct_rows) == observation_rows and observation_rows > 0, f"{case}: direct observations missing")
    objective = report.get("objective") if isinstance(report.get("objective"), dict) else {}
    require(float(objective.get("temporal_smoothed_observation_energy", 0.0)) <= float(objective.get("identity_prior_energy", 0.0)) + 1e-9, f"{case}: smoothed correction did not improve over identity prior")
    observed = 0
    interpolated = 0
    for row in full_rows:
        require(isinstance(row, dict), f"{case}: malformed row")
        scale = row.get("depth_scale_estimate")
        require(isinstance(scale, (int, float)) and float(scale) > 0, f"{case}: invalid scale estimate")
        if row.get("has_direct_observation") is True:
            observed += 1
            require(row.get("state") == "observed_depth_scale_correction", f"{case}: wrong observed state")
        else:
            interpolated += 1
            require(row.get("state") == "interpolated_or_nearest_depth_scale_no_direct_observation", f"{case}: wrong interpolated state")
    require(observed == observation_rows, f"{case}: observed row count mismatch")
    return {"case": case, "frame_count": frame_count, "observation_rows": observed, "interpolated_rows": interpolated, "scale_stats": report.get("depth_scale_estimate_stats"), "identity_prior_energy": objective.get("identity_prior_energy"), "smoothed_energy": objective.get("temporal_smoothed_observation_energy")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data2/ego_annotation_outputs/v18_camera_depth_correction"))
    parser.add_argument("--cases", nargs="+", default=["trash_1050", "task5_tomato_960"])
    args = parser.parse_args()
    rows = [validate_case(args.root / case / "v18_camera_depth_correction_report.json") for case in args.cases]
    print(json.dumps({"status": "ok", "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
