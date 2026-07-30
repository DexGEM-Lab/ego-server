#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNNER_AGENT_ID = "runner_agent"
ACTIVE_ONLY_POLICY = "active_algorithm_ids_only; deprecated historical algorithms and alias directories are excluded; execution is scheduled by runner_agent"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected_json_object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def link_or_copy_file(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def materialize_path(src_raw: str | None, dst_root: Path) -> dict[str, Any] | None:
    if not src_raw:
        return None
    src = Path(src_raw)
    if not src.exists():
        return {"source": str(src), "exists": False}
    if src.is_file():
        dst = dst_root / src.name
        method = link_or_copy_file(src, dst)
        return {"source": str(src), "materialized": str(dst), "exists": True, "kind": "file", "method": method}
    if not src.is_dir():
        return {"source": str(src), "exists": False, "kind": "unsupported"}
    dst_dir = dst_root / src.name
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    file_count = 0
    method_counts: dict[str, int] = {}
    for child in src.rglob("*"):
        if child.is_dir():
            continue
        rel = child.relative_to(src)
        method = link_or_copy_file(child, dst_dir / rel)
        method_counts[method] = method_counts.get(method, 0) + 1
        file_count += 1
    return {
        "source": str(src),
        "materialized": str(dst_dir),
        "exists": True,
        "kind": "directory",
        "file_count": file_count,
        "methods": method_counts,
    }


def materialized_file_count(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file())


def case_algorithm_ids(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        case = str(row.get("case"))
        algo = str(row.get("algorithm_id"))
        out.setdefault(case, [])
        if algo not in out[case]:
            out[case].append(algo)
    return {case: sorted(vals) for case, vals in sorted(out.items())}


def materialize(audit_path: Path, output_root: Path, clear: bool = False) -> dict[str, Any]:
    audit = load_json(audit_path)
    rows = audit.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"audit_missing_rows: {audit_path}")
    active_rows = [r for r in rows if not r.get("deprecated")]
    if clear and output_root.exists():
        for child in output_root.iterdir():
            if child.name == "logs":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_root.mkdir(parents=True, exist_ok=True)

    missing_not_materialized: list[dict[str, Any]] = []
    for row in active_rows:
        case = str(row.get("case"))
        algo = str(row.get("algorithm_id"))
        atom_dir = output_root / case / algo
        if atom_dir.exists():
            shutil.rmtree(atom_dir)
        atom_dir.mkdir(parents=True, exist_ok=True)

        data_result = materialize_path(row.get("data_path"), atom_dir / "data")
        overlay_result = materialize_path(row.get("overlay_path") or row.get("native_overlay_path"), atom_dir)
        qc_result = materialize_path(row.get("qc_path"), atom_dir)
        alt_data_result = materialize_path(row.get("alt_data_path"), atom_dir / "alt_data")

        tuning_results = []
        tuning_records = row.get("tuning_records") or []
        if isinstance(tuning_records, list):
            for item in tuning_records:
                tuning_results.append(materialize_path(str(item), atom_dir / "tuning"))

        execution_path = output_root / "logs" / "execution" / case / algo / "execution.json"
        overlay_execution_path = output_root / "logs" / "execution" / case / f"overlay_{algo}" / "execution.json"
        execution_record = load_json(execution_path) if execution_path.exists() else None
        overlay_execution_record = load_json(overlay_execution_path) if overlay_execution_path.exists() else None

        record = {
            "schema": "materialized_atomic_algorithm_record.v1",
            "case": case,
            "algorithm_id": algo,
            "family": row.get("family"),
            "required": row.get("required"),
            "optional": row.get("optional"),
            "status": row.get("status"),
            "source_audit": str(audit_path),
            "source_row": row,
            "runner_agent": row.get("runner_agent") or (execution_record or {}).get("runner_agent") or (overlay_execution_record or {}).get("runner_agent") or RUNNER_AGENT_ID,
            "materialized": {
                "data": data_result,
                "alt_data": alt_data_result,
                "overlay": overlay_result,
                "qc": qc_result,
                "tuning_records": tuning_results,
            },
            "execution": execution_record,
            "overlay_execution": overlay_execution_record,
        }
        write_json(atom_dir / "record.json", record)
        execution_state = str(execution_record.get("execution_state")) if isinstance(execution_record, dict) else ""
        blocked_status = row.get("status") in {"missing_data", "invalid_data", "overlay_without_current_data"}
        blocked_execution = execution_state.startswith("blocked_") or execution_state in {"not_rerun_heavy_disabled", "imported_unrerunnable_legacy_input"}
        if blocked_status or blocked_execution:
            blocked = {
                "schema": "blocked_atomic_algorithm.v1",
                "case": case,
                "algorithm_id": algo,
                "family": row.get("family"),
                "reason": (execution_record or {}).get("reason") if isinstance(execution_record, dict) else "data_path_missing_in_source_run",
                "audit_status": row.get("status"),
                "execution_state": execution_state or None,
                "runner_agent": row.get("runner_agent") or RUNNER_AGENT_ID,
                "overlay_execution_state": overlay_execution_record.get("execution_state") if isinstance(overlay_execution_record, dict) else None,
                "source_row": row,
            }
            if not blocked["reason"]:
                blocked["reason"] = "data_path_missing_in_source_run"
            write_json(atom_dir / "BLOCKED.json", blocked)
            if blocked_status:
                missing_not_materialized.append({"case": case, "algorithm_id": algo, "family": row.get("family"), "status": row.get("status"), "execution_state": execution_state or None})

    source_copy = output_root / "final_source_atomic_algorithm_overlay_audit.json"
    shutil.copy2(audit_path, source_copy)

    summary = audit.get("summary", {}) if isinstance(audit.get("summary"), dict) else {}
    manifest = {
        "schema": "active_atomic_algorithm_runs_manifest.v3",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(output_root),
        "policy": ACTIVE_ONLY_POLICY,
        "runner_agent": RUNNER_AGENT_ID,
        "source_audit": str(audit_path),
        "source_audit_summary": summary,
        "materialized_result_directory_count": len(active_rows),
        "materialized_file_count": materialized_file_count(output_root),
        "active_algorithm_ids_by_case": case_algorithm_ids(active_rows),
        "missing_active_result_not_materialized": missing_not_materialized,
    }
    write_json(output_root / "MANIFEST.json", manifest)

    readme = [
        "# Pico and Living-Room Atomic Algorithm Runs",
        "",
        "This directory materializes the active V21 atomic algorithm inventory for the Pico and living-room runs.",
        "",
        f"Policy: {ACTIVE_ONLY_POLICY}.",
        "",
        f"Source audit: `{audit_path}`.",
        "",
        f"Active rows materialized: {len(active_rows)}.",
        f"Overlay-ready rows in source audit: {summary.get('overlay_ready_count')} / {summary.get('active_required_count')}.",
        "",
        "The only expected active missing-data row is living_room/contact_patch_pose_graph unless the source audit says otherwise.",
    ]
    (output_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="outputs/v21_per_algorithm_results/atomic_algorithm_overlay_audit.json")
    ap.add_argument("--output-root", default="outputs/atomic_algorithm_runs_pico_living_room_20260630")
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args()
    manifest = materialize(Path(args.audit), Path(args.output_root), clear=args.clear)
    print(json.dumps({"status": "ok", "manifest": str(Path(args.output_root) / "MANIFEST.json"), "summary": manifest["source_audit_summary"]}, indent=2))


if __name__ == "__main__":
    main()
