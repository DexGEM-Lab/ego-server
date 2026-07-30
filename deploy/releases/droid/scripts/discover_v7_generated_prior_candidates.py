#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def case_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) > 1:
        parent = rel.parts[0]
        if parent:
            return parent
    return path.stem


def target_for_case(case: str, mapping: dict) -> str | None:
    if case in mapping:
        return str(mapping[case])
    matches = [target for prefix, target in mapping.items() if case.startswith(str(prefix))]
    if len(matches) == 1:
        return str(matches[0])
    return None


def candidate_name(source_name: str, case: str, mesh_path: Path) -> str:
    suffix = mesh_path.suffix.lower().lstrip(".") or "mesh"
    path_key = "_".join(mesh_path.with_suffix("").parts[-3:])
    safe_key = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in path_key)
    return f"{source_name}_{case}_{safe_key}_{suffix}"


def discover_source(source_name: str, raw: object) -> list[dict]:
    if not isinstance(raw, dict):
        raise RuntimeError(f"source {source_name} must be a JSON object")
    root = Path(str(raw.get("root", "")))
    if not root.exists():
        return []
    globs = raw.get("mesh_globs")
    if not isinstance(globs, list) or not globs:
        raise RuntimeError(f"source {source_name} lacks mesh_globs")
    mapping = raw.get("case_targets")
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(f"source {source_name} lacks case_targets")
    rows = []
    seen = set()
    for pattern in globs:
        for mesh_path in sorted(root.glob(str(pattern))):
            if not mesh_path.is_file() or mesh_path in seen:
                continue
            seen.add(mesh_path)
            case = case_name(mesh_path, root)
            target = target_for_case(case, mapping)
            if target is None:
                raise RuntimeError(
                    f"source {source_name} produced unmapped mesh {mesh_path} with inferred case {case}"
                )
            rows.append(
                {
                    "target_id": target,
                    "candidate_name": candidate_name(source_name, case, mesh_path),
                    "mesh_path": str(mesh_path),
                    "note": f"{source_name} generated complete-mesh prior for {case}",
                    "source": source_name,
                    "case": case,
                }
            )
    return rows


def candidate_arg(row: dict) -> str:
    return "|".join([row["target_id"], row["candidate_name"], row["mesh_path"], row["note"]])


def run(args: argparse.Namespace) -> dict:
    sources = load_json(args.sources_json)
    if args.source:
        missing_sources = sorted(set(args.source).difference(sources))
        if missing_sources:
            raise RuntimeError(f"requested sources are not configured: {', '.join(missing_sources)}")
        sources = {name: sources[name] for name in args.source}
    rows = []
    for source_name, raw in sources.items():
        rows.extend(discover_source(str(source_name), raw))
    rows.sort(key=lambda row: (row["target_id"], row["candidate_name"], row["mesh_path"]))
    if args.require_candidates and not rows:
        raise RuntimeError(f"no generated prior candidates found from {args.sources_json}")
    report = {
        "status": "ok",
        "method": "discover_v7_generated_prior_candidates",
        "sources_json": str(args.sources_json),
        "candidate_count": int(len(rows)),
        "candidates": rows,
        "batch_candidate_args": [candidate_arg(row) for row in rows],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_args is not None:
        args.output_args.parent.mkdir(parents=True, exist_ok=True)
        args.output_args.write_text("\n".join(report["batch_candidate_args"]) + ("\n" if rows else ""), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-json", type=Path, default=REPO_DIR / "configs" / "v7_generated_candidate_sources.json")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-args", type=Path)
    parser.add_argument("--require-candidates", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
