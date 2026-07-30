#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_v21_atomic_algorithm_overlays import ALGORITHMS, RUNNER_AGENT_ID, RUNNER_POLICY, RUNS, materialize as resolve_pattern  # noqa: E402

ALGORITHM_BY_ID = {str(spec["id"]): spec for spec in ALGORITHMS}

DEFAULT_PYTHON = Path("/home/zjh/miniconda3/envs/ego_foundation/bin/python")
DEFAULT_SAM2_REPO = Path("/mnt/user-home/zjh/ego-pipeline/ego_annotation_v19/third_party/sam2")
DEFAULT_SAM2_CKPT = Path("/mnt/user-home/zjh/ego-pipeline/v19_assets/checkpoints/sam2.1/sam2.1_hiera_large.pt")
DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_DEPTHPRO_REPO = Path("/mnt/user-home/zjh/ego-pipeline/v21_model_work/depthpro_work/ml-depth-pro")
DEFAULT_RTMLIB_PYTHON = Path("/home/zjh/miniconda3/envs/hamer/bin/python")
DEFAULT_WILOR_PYTHON = Path("/home/zjh/miniconda3/envs/ego_v19/bin/python")
DEFAULT_OWLV2_MODEL = Path("/mnt/user-home/zjh/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble/snapshots/cfd3195ba4ea9592eec887ded089f4c08eff231d")

FULL_RERUN_ATOMS = {
    "raw_frame_manifest",
    "source_frame_manifest",
    "depth_modality_report",
    "depth_candidate_registry",
    "depth_selection_bundle",
    "depth_camera_selection",
    "segmentation_stable_keyframes",
    "object_plan_agent",
    "object_plan_current",
    "owlv2_bbox",
    "owlv2_bbox_approved_prompts",
    "sam2_proper",
    "v21_renderable_annotations",
    "visible_surface",
    "v21_mesh_candidate",
    "v18_full_mano_annotations",
    "v18_compact_rigid_pose_fit",
    "v19_rigid_pose_graph",
    "adopted_object_pose",
    "v21_physical_state",
    "v21_uncertainty_state",
    "v21_final_overlay",
    "v21_final_world",
    "v21_final_side_by_side",
}

HEAVY_RERUN_ATOMS = {
    "depthpro",
    "unidepth_v2",
    "rtmlib_2d",
    "wilor_mano",
    "wilor_metric_refit",
    "active_mano",
}

BLOCKED_NO_ENTRYPOINT = {
    "depth_anything_v2": "no run_v21_depth_anything_v2 entrypoint exists in scripts/",
    "droid_or_camera_trajectory": "no current script found to regenerate depthpro_as_droid.npz",
    "hamer": "HaMeR runner exists only as legacy stream integration and is not wired to the V21 atom output contract",
    "hawor": "HaWoR atom is inherited from the v18 hand-baseline branch and has no current V21 rerun entrypoint",
    "heightfield_observed": "v18 heightfield atom is inherited; current V21 mesh candidate is rerun instead",
    "object_factor_graph": "v18 object factor graph atom is inherited; no current V21 command is wired here",
    "mesh_prior_graph": "v18 mesh-prior graph atom is inherited; no current V21 command is wired here",
    "contact_patch_pose_graph": "v18 contact-patch graph atom is inherited; living-room data is missing in the current source run",
    "contact_ownership_graph": "v18 contact-ownership graph atom is inherited; current V21 contact/occlusion/nonpenetration report is rerun instead",
    "occlusion_owner_graph": "v18 occlusion-owner graph atom is inherited; current V21 contact/occlusion/nonpenetration report is rerun instead",
    "signed_nonpenetration": "v18 signed-nonpenetration atom is inherited; current V21 contact/occlusion/nonpenetration report is rerun instead",
    "triangle_nonpenetration": "v18 triangle-nonpenetration atom is inherited; current V21 contact/occlusion/nonpenetration report is rerun instead",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected_json_object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_current_output_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    path = Path(str(raw))
    candidates = [path]
    text = str(raw)
    if text.startswith("outputs/"):
        candidates.append(Path("output") / Path(text).relative_to("outputs"))
    historical_prefix = "/mnt/user-home/zjh/ego-pipeline/ego_annotation-master/outputs/"
    if text.startswith(historical_prefix):
        candidates.append(REPO_ROOT / "output" / Path(text[len(historical_prefix) :]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


class AtomicSuiteRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.python = Path(args.python_bin)
        self.rtmlib_python = Path(args.rtmlib_python_bin)
        self.wilor_python = Path(args.wilor_python_bin)
        self.atomic_root = Path(args.atomic_root)
        self.overlay_root = Path(args.overlay_root)
        self.execution_root = self.atomic_root / "logs" / "execution"
        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.summary: dict[str, Any] = {
            "schema": "v21_atomic_algorithm_suite_execution_summary.v2",
            "started_at": utc_now(),
            "runner_agent": RUNNER_AGENT_ID,
            "runner_policy": RUNNER_POLICY,
            "schedule_source": "audit_v21_atomic_algorithm_overlays.ALGORITHMS",
            "refresh_overlays_only": bool(args.refresh_overlays_only),
            "atomic_root": str(self.atomic_root),
            "overlay_root": str(self.overlay_root),
            "compute_target": args.compute_target,
            "include_heavy": bool(args.include_heavy),
            "cases": [],
            "records": [],
        }

    def exec_path(self, case: str, atom: str) -> Path:
        return self.execution_root / case / atom / "execution.json"

    def log_path(self, case: str, atom: str) -> Path:
        return self.execution_root / case / atom / "run.log"

    def write_execution(self, case: str, atom: str, state: str, **fields: Any) -> None:
        base_atom = atom[len("overlay_") :] if atom.startswith("overlay_") else atom
        spec = ALGORITHM_BY_ID.get(base_atom)
        payload = {
            "schema": "v21_atomic_algorithm_execution_record.v2",
            "case": case,
            "algorithm_id": atom,
            "base_algorithm_id": base_atom,
            "algorithm_family": spec.get("family") if spec else None,
            "runner_agent": RUNNER_AGENT_ID,
            "runner_policy": RUNNER_POLICY,
            "runner_schedule_kind": "overlay" if atom.startswith("overlay_") else "algorithm",
            "execution_state": state,
            "updated_at": utc_now(),
            "atomic_rerun_id": self.timestamp,
            **fields,
        }
        write_json(self.exec_path(case, atom), payload)
        self.summary["records"].append({"case": case, "algorithm_id": atom, "base_algorithm_id": base_atom, "execution_state": state})

    def archive_path(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        suffix = f"_before_atomic_rerun_{self.timestamp}"
        target = path.with_name(path.name + suffix)
        i = 1
        while target.exists():
            target = path.with_name(path.name + suffix + f"_{i:02d}")
            i += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        return {"source": str(path), "archived_to": str(target)}

    def archive_targets(self, targets: list[Path]) -> list[dict[str, Any]]:
        archives: list[dict[str, Any]] = []
        if not self.args.archive_existing:
            return archives
        for path in targets:
            archived = self.archive_path(path)
            if archived is not None:
                archives.append(archived)
        return archives

    def previous_success_satisfies_targets(self, case: str, atom: str, targets: list[Path]) -> bool:
        if not self.args.resume_success:
            return False
        record_path = self.exec_path(case, atom)
        if not record_path.exists():
            return False
        try:
            record = load_json(record_path)
        except Exception:
            return False
        if record.get("execution_state") != "reran_now":
            return False
        return all(path.exists() for path in targets)

    def heavy_env(self) -> dict[str, str]:
        env: dict[str, str] = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        if self.args.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = str(self.args.cuda_visible_devices)
        return env

    def run_cmd(
        self,
        case: str,
        atom: str,
        cmd: list[str | Path],
        *,
        targets: list[Path] | None = None,
        env: dict[str, str] | None = None,
        cwd: Path = REPO_ROOT,
        allow_resume: bool = True,
    ) -> bool:
        target_list = targets or []
        if allow_resume and self.previous_success_satisfies_targets(case, atom, target_list):
            self.summary["records"].append({"case": case, "algorithm_id": atom, "execution_state": "resumed_previous_success"})
            print(f"{case}/{atom}: resumed_previous_success", flush=True)
            return True
        cmd_s = [str(x) for x in cmd]
        archives = self.archive_targets(target_list)
        log_path = self.log_path(case, atom)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        started_at = utc_now()
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"RUNNER_AGENT {RUNNER_AGENT_ID}\n")
            handle.write("COMMAND " + json.dumps(cmd_s, ensure_ascii=False) + "\n")
            handle.write(f"CWD {cwd}\n")
            handle.write(f"STARTED_AT {started_at}\n")
            handle.flush()
            merged_env = os.environ.copy()
            if env:
                merged_env.update(env)
            proc = subprocess.run(
                cmd_s,
                cwd=str(cwd),
                env=merged_env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            handle.write(f"FINISHED_AT {utc_now()}\n")
            handle.write(f"RETURN_CODE {proc.returncode}\n")
        state = "reran_now" if proc.returncode == 0 else "failed_rerun"
        self.write_execution(
            case,
            atom,
            state,
            command=cmd_s,
            cwd=str(cwd),
            returncode=int(proc.returncode),
            log=str(log_path),
            archived_targets=archives,
            elapsed_s=float(time.time() - started),
            compute_target=self.args.compute_target,
        )
        print(f"{case}/{atom}: {state}", flush=True)
        return proc.returncode == 0

    def mark_non_rerun_inventory(self, case: str, *, rerun_skipped_reason: str | None = None) -> None:
        for spec in ALGORITHMS:
            atom = str(spec["id"])
            if self.exec_path(case, atom).exists():
                continue
            if atom in BLOCKED_NO_ENTRYPOINT:
                self.write_execution(
                    case,
                    atom,
                    "blocked_no_rerun_entrypoint",
                    reason=BLOCKED_NO_ENTRYPOINT[atom],
                    source=str(spec.get("source")),
                )
            elif atom in HEAVY_RERUN_ATOMS and not self.args.include_heavy:
                self.write_execution(
                    case,
                    atom,
                    "not_rerun_heavy_disabled",
                    reason="heavy rerun disabled by --include-heavy=false",
                    required_compute_target=self.args.compute_target,
                )
            elif rerun_skipped_reason is not None:
                self.write_execution(
                    case,
                    atom,
                    "not_rerun_overlay_refresh_only",
                    reason=rerun_skipped_reason,
                    command_known_to_runner=atom in FULL_RERUN_ATOMS,
                    source=str(spec.get("source")),
                )
            elif atom not in FULL_RERUN_ATOMS:
                self.write_execution(
                    case,
                    atom,
                    "imported_unrerunnable_legacy_input",
                    reason="active audit atom has no current rerun command in run_v21_atomic_algorithm_suite.py",
                    source=str(spec.get("source")),
                )

    def case_info(self, case: str) -> tuple[Path, str, str]:
        info = RUNS[case]
        run_root = Path(str(info["run_root"]))
        object_id = str(info["object_id"])
        run_case = run_root.name
        return run_root, object_id, run_case

    def manifest_paths(self, run_root: Path) -> tuple[Path, Path, Path]:
        return (
            run_root / "input" / "input_manifest.json",
            run_root / "input" / "raw_frame_manifest" / "manifest.json",
            run_root / "input" / "source_frame_manifest" / "manifest.json",
        )

    def build_keyframe_review_sheet(self, case: str, run_root: Path, report_path: Path, output_path: Path) -> None:
        report = load_json(report_path)
        rows = report.get("selected_keyframes")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"stable_keyframe_report_has_no_selected_keyframes: {report_path}")
        tiles = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            frame_idx = int(row.get("frame_idx"))
            img_path = run_root / "input" / "source_frame_manifest" / "rgb" / f"{frame_idx:06d}.jpg"
            if not img_path.exists():
                img_path = run_root / "input" / "raw_frame_manifest" / "rgb" / f"{frame_idx:06d}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                raise RuntimeError(f"could_not_read_keyframe_image: {img_path}")
            scale = 360.0 / max(1, img.shape[1])
            thumb = cv2.resize(img, (360, max(1, int(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            label = f"frame {frame_idx} | {row.get('segment_class') or row.get('interaction_class') or 'segment'}"
            cv2.rectangle(thumb, (0, 0), (thumb.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(thumb, label[:56], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(thumb)
        if not tiles:
            raise RuntimeError("no_keyframe_tiles_written")
        height = max(tile.shape[0] for tile in tiles)
        padded = []
        for tile in tiles:
            if tile.shape[0] < height:
                pad = cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
            else:
                pad = tile
            padded.append(pad)
        sheet = cv2.hconcat(padded) if len(padded) > 1 else padded[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), sheet):
            raise RuntimeError(f"could_not_write_keyframe_review: {output_path}")
        self.write_execution(
            case,
            "segmentation_stable_keyframes_review_sheet",
            "reran_now_support_artifact",
            output=str(output_path),
            source_report=str(report_path),
        )

    def sync_plan_from_keyframes(self, case: str, run_root: Path) -> None:
        plan = run_root / "measurements" / "object_candidates" / "object_plan_segmentation_stable_keyframes.json"
        targets = [
            run_root / "measurements" / "object_candidates" / "object_plan_current.json",
            run_root / "measurements" / "object_candidates" / "object_plan_agent.json",
        ]
        if not plan.exists():
            raise RuntimeError(f"missing_stable_object_plan: {plan}")
        archives = self.archive_targets(targets)
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(plan, target)
        for atom, target in [("object_plan_current", targets[0]), ("object_plan_agent", targets[1])]:
            self.write_execution(
                case,
                atom,
                "reran_now",
                mechanism="copied object_plan_segmentation_stable_keyframes.json produced by segmentation_stable_keyframes atom",
                output=str(target),
                source_plan=str(plan),
                archived_targets=archives,
            )

    def input_contract(self, case: str, run_root: Path) -> None:
        input_manifest = run_root / "input" / "input_manifest.json"
        self.write_execution(
            case,
            "input_manifest",
            "retained_raw_input_contract",
            reason="input_manifest is the raw-video contract for this rerun; preparing it again would rewrite the same source clip contract rather than execute a perception atom",
            output=str(input_manifest),
        )

    def run_input_adapters(self, case: str, run_root: Path) -> None:
        input_manifest, raw_manifest, source_manifest = self.manifest_paths(run_root)
        meta = load_json(input_manifest).get("primary_video_metadata", {})
        render_width = int(load_json(input_manifest).get("raw_frame_manifest_summary", {}).get("render_width") or 960)
        self.run_cmd(
            case,
            "raw_frame_manifest",
            [
                self.python,
                SCRIPTS / "rebuild_v21_raw_frame_manifest_from_input.py",
                "--input-manifest",
                input_manifest,
                "--output-dir",
                raw_manifest.parent,
                "--output-manifest",
                raw_manifest,
                "--render-width",
                str(render_width),
            ],
            targets=[raw_manifest.parent],
        )
        self.run_cmd(
            case,
            "source_frame_manifest",
            [
                self.python,
                SCRIPTS / "build_v21_source_frame_manifest.py",
                "--input-manifest",
                input_manifest,
                "--output-dir",
                source_manifest.parent,
                "--output-manifest",
                source_manifest,
            ],
            targets=[source_manifest.parent],
        )
        if meta:
            self.summary.setdefault("input_frames", {})[case] = meta.get("frame_count")

    def run_depth_atoms(self, case: str, run_root: Path) -> None:
        input_manifest, raw_manifest, _ = self.manifest_paths(run_root)
        input_payload = load_json(input_manifest)
        depth_dir = run_root / "measurements" / "camera_depth"
        self.run_cmd(
            case,
            "depth_modality_report",
            [self.python, SCRIPTS / "build_v21_depth_modality_report.py", "--input-manifest", input_manifest, "--output-report", depth_dir / "depth_modality_report.json"],
            targets=[depth_dir / "depth_modality_report.json"],
        )
        frame_count = int(input_payload.get("primary_video_metadata", {}).get("frame_count", 0))
        width = int(input_payload.get("primary_video_metadata", {}).get("width", 0))
        height = int(input_payload.get("primary_video_metadata", {}).get("height", 0))
        if self.args.include_heavy:
            self.run_cmd(
                case,
                "depthpro",
                [
                    self.python,
                    SCRIPTS / "run_v21_depthpro_full_frame_candidate.py",
                    "--raw-frame-manifest",
                    raw_manifest,
                    "--output-dir",
                    run_root / "measurements" / "depth_candidates" / "depthpro_full_frame",
                    "--frame-start",
                    "0",
                    "--frame-end",
                    str(frame_count - 1),
                    "--frame-stride",
                    "1",
                    "--depthpro-repo",
                    Path(self.args.depthpro_repo),
                    "--source-width",
                    str(width),
                    "--source-height",
                    str(height),
                ],
                targets=[run_root / "measurements" / "depth_candidates" / "depthpro_full_frame"],
                env=self.heavy_env(),
            )
            self.run_cmd(
                case,
                "unidepth_v2",
                [self.python, SCRIPTS / "run_v21_unidepth.py", "--run-root", run_root],
                targets=[run_root / "measurements" / "depth_candidates" / "unidepth_v2"],
                env=self.heavy_env(),
            )
        stereo_right = input_payload.get("stereo_right_video")
        if stereo_right:
            self.run_cmd(
                case,
                "stereo_sgbm",
                [
                    self.python,
                    SCRIPTS / "run_v21_stereo_sgbm_candidate.py",
                    "--raw-frame-manifest",
                    raw_manifest,
                    "--left-video",
                    input_payload["primary_video"],
                    "--right-video",
                    stereo_right,
                    "--output-npz",
                    run_root / "measurements" / "depth_candidates" / "stereo_sgbm" / "relative_inverse_depth.npz",
                    "--output-report",
                    run_root / "measurements" / "depth_candidates" / "stereo_sgbm" / "report.json",
                    "--preview-dir",
                    run_root / "measurements" / "depth_candidates" / "stereo_sgbm" / "preview",
                ],
                targets=[run_root / "measurements" / "depth_candidates" / "stereo_sgbm"],
            )
        self.run_depth_registry_selection(case, run_root)
        self.run_cmd(
            case,
            "depth_camera_selection",
            [
                self.python,
                SCRIPTS / "select_v21_depth_camera_bundle.py",
                "--run-root",
                run_root,
                "--input-manifest",
                input_manifest,
                "--modality-report",
                depth_dir / "depth_modality_report.json",
                "--output-report",
                depth_dir / "depth_camera_selection_report.json",
            ],
            targets=[depth_dir / "depth_camera_selection_report.json"],
        )

    def run_hand_atoms(self, case: str, run_root: Path) -> None:
        if not self.args.include_heavy:
            return
        input_manifest = load_json(run_root / "input" / "input_manifest.json")
        frame_count = int(input_manifest.get("primary_video_metadata", {}).get("frame_count", 0))
        clip = Path(str(input_manifest["primary_video"]))
        wilor_raw = run_root / "measurements" / "hand_candidates" / "wilor_v21" / "wilor_raw_hands.json"
        self.run_cmd(
            case,
            "rtmlib_2d",
            [
                self.rtmlib_python,
                SCRIPTS / "run_rtmlib_hand2d_v3.py",
                "--clip",
                clip,
                "--output-dir",
                run_root / "measurements" / "hand_candidates" / "rtmlib_2d",
                "--frame-start",
                "0",
                "--frame-end",
                str(frame_count - 1),
                "--source-frame-offset",
                "0",
                "--device",
                "cuda",
            ],
            targets=[run_root / "measurements" / "hand_candidates" / "rtmlib_2d"],
            env=self.heavy_env(),
        )
        self.run_cmd(
            case,
            "wilor_mano",
            [
                self.wilor_python,
                SCRIPTS / "run_v21_wilor_hand_candidates.py",
                "--run-root",
                run_root,
                "--repo-root",
                REPO_ROOT,
                "--compute-target",
                self.args.compute_target,
            ],
            targets=[run_root / "measurements" / "hand_candidates" / "wilor_v21"],
            env=self.heavy_env(),
        )
        self.run_cmd(
            case,
            "wilor_metric_refit",
            [self.python, SCRIPTS / "run_v21_mano_metric_refit.py", "--run-root", run_root],
            targets=[run_root / "measurements" / "hand_candidates" / "wilor_v21_metric"],
        )
        self.run_cmd(
            case,
            "active_mano",
            [
                self.wilor_python,
                SCRIPTS / "solve_v21_active_mano.py",
                "--run-root",
                run_root,
                "--max-frames",
                str(self.args.active_mano_max_frames),
                "--iters",
                str(self.args.active_mano_iters),
            ],
            targets=[run_root / "measurements" / "hand_candidates" / "v21_active_mano"],
            env=self.heavy_env(),
        )

    def run_keyframe_and_segmentation_atoms(self, case: str, run_root: Path, object_id: str) -> None:
        _, raw_manifest, source_manifest = self.manifest_paths(run_root)
        object_dir = run_root / "measurements" / "object_candidates"
        plan_in = object_dir / "object_plan_current.json"
        if not plan_in.exists():
            plan_in = object_dir / "object_plan_agent.json"
        if not plan_in.exists():
            plan_in = object_dir / "object_plan_seed_from_v20_visual_qc.json"
        keyframe_output = object_dir / "segmentation_stable_keyframes.json"
        plan_output = object_dir / "object_plan_segmentation_stable_keyframes.json"
        cmd: list[str | Path] = [
            self.python,
            SCRIPTS / "select_v21_agent_keyframes_from_plan.py",
            "--run-root",
            run_root,
            "--track-id",
            object_id,
            "--object-plan",
            plan_in,
            "--output",
            keyframe_output,
            "--object-plan-output",
            plan_output,
            "--raw-frame-manifest",
            raw_manifest,
            "--source-frame-manifest",
            source_manifest,
        ]
        ok = self.run_cmd(case, "segmentation_stable_keyframes", cmd, targets=[keyframe_output, plan_output])
        if ok:
            self.build_keyframe_review_sheet(case, run_root, keyframe_output, object_dir / "segmentation_stable_keyframes_review.jpg")
            self.sync_plan_from_keyframes(case, run_root)
        approved_prompts = object_dir / "owlv2_bbox_approved_prompts.json"
        sam2_summary = run_root / "measurements" / "object_tracks" / "sam2_proper_summary.json"
        if self.args.include_heavy:
            self.run_cmd(
                case,
                "owlv2_bbox",
                [
                    self.python,
                    SCRIPTS / "run_v21_owlv2_bbox_proposals.py",
                    "--run-root",
                    run_root,
                    "--raw-frame-manifest",
                    source_manifest,
                    "--object-plan",
                    object_dir / "object_plan_current.json",
                    "--output",
                    object_dir / "owlv2_bbox_proposals.json",
                    "--owlv2-model",
                    Path(self.args.owlv2_model),
                    "--keyframe-selection-report",
                    keyframe_output,
                    "--compute-target",
                    self.args.compute_target,
                ],
                targets=[object_dir / "owlv2_bbox_proposals.json"],
                env=self.heavy_env(),
            )
            self.run_cmd(
                case,
                "owlv2_bbox_approved_prompts",
                [
                    self.python,
                    SCRIPTS / "approve_v21_owlv2_bbox_prompts.py",
                    "--run-root",
                    run_root,
                    "--object-id",
                    object_id,
                    "--owlv2-proposals",
                    object_dir / "owlv2_bbox_proposals.json",
                    "--keyframe-selection-report",
                    keyframe_output,
                    "--output",
                    approved_prompts,
                ],
                targets=[approved_prompts],
            )
            sam2_env = self.heavy_env()
            sam2_env["PYTHONPATH"] = f"{SCRIPTS}:{self.args.sam2_repo}" + (f":{os.environ['PYTHONPATH']}" if os.environ.get("PYTHONPATH") else "")
            self.run_cmd(
                case,
                "sam2_proper",
                [
                    self.python,
                    SCRIPTS / "run_v21_sam2_proper_segmentation.py",
                    "--run-root",
                    run_root,
                    "--object-id",
                    object_id,
                    "--approved-bbox-prompts",
                    approved_prompts,
                    "--output-summary",
                    sam2_summary,
                    "--sam2-checkpoint",
                    Path(self.args.sam2_checkpoint),
                    "--sam2-model-cfg",
                    self.args.sam2_model_cfg,
                    "--compute-target",
                    self.args.compute_target,
                ],
                targets=[run_root / "measurements" / "object_tracks" / "sam2_proper" / object_id / "sam2_track.json", sam2_summary],
                env=sam2_env,
            )
        elif not sam2_summary.exists():
            raise RuntimeError(f"active_v21_sam2_proper_summary_missing_and_heavy_disabled: {sam2_summary}")
        review_report = run_root / "review" / "segmentation_sam2_proper" / "segmentation_contamination_review.json"
        self.run_cmd(
            case,
            "sam2_proper_review",
            [
                self.python,
                SCRIPTS / "review_v21_segmentation_contamination.py",
                "--input-manifest",
                run_root / "input" / "input_manifest.json",
                "--sam2-summary",
                sam2_summary,
                "--output-report",
                review_report,
                "--output-dir",
                review_report.parent,
                "--min-visible-fraction",
                "0.01",
            ],
            targets=[review_report.parent],
        )
        self.run_cmd(
            case,
            "v21_renderable_annotations",
            [
                self.python,
                SCRIPTS / "assemble_v21_segmentation_state.py",
                "--input-manifest",
                run_root / "input" / "input_manifest.json",
                "--segmentation-review",
                review_report,
                "--output-annotations",
                run_root / "state" / "annotations_v21_renderable.json",
                "--output-state",
                run_root / "state" / "v21_physical_state.json",
                "--output-summary",
                run_root / "state" / "v21_segmentation_state_summary.json",
            ],
            targets=[run_root / "state" / "annotations_v21_renderable.json", run_root / "state" / "v21_segmentation_state_summary.json"],
        )

    def run_geometry_pose_render_atoms(self, case: str, run_root: Path, object_id: str) -> None:
        _, _, source_manifest = self.manifest_paths(run_root)
        visible_root = run_root / "measurements" / "object_visible_surfaces" / "depthpro_local_grabcut"
        self.run_cmd(
            case,
            "visible_surface",
            [
                self.python,
                SCRIPTS / "run_v21_visible_surface_from_depth.py",
                "--v21-state",
                run_root / "state" / "v21_physical_state.json",
                "--v21-annotations",
                run_root / "state" / "annotations_v21_renderable.json",
                "--depth-selection-report",
                run_root / "measurements" / "camera_depth" / "depth_camera_selection_report.json",
                "--source-frame-manifest",
                source_manifest,
                "--object-plan",
                run_root / "measurements" / "object_candidates" / "object_plan_current.json",
                "--output-root",
                visible_root,
                "--output-summary",
                visible_root / "visible_surface_summary.json",
            ],
            targets=[visible_root],
        )
        self.run_cmd(
            case,
            "v21_mesh_candidate",
            [
                self.python,
                SCRIPTS / "build_v21_mesh_candidate_from_observed.py",
                "--run-root",
                run_root,
                "--object-id",
                object_id,
                "--depth-selection-report",
                run_root / "measurements" / "camera_depth" / "depth_camera_selection_report.json",
                "--depth-registry",
                run_root / "measurements" / "camera_depth" / "v20_depth_registry" / "depth_candidate_registry.json",
            ],
            targets=[run_root / "measurements" / "object_geometry" / "v21_mesh_candidate" / object_id],
        )
        self.run_cmd(case, "v21_rigid_pose_estimate", [self.python, SCRIPTS / "solve_v21_rigid_pose_estimate.py", "--run-root", run_root, "--object-id", object_id], targets=[run_root / "measurements" / "object_pose" / object_id / "rigid_pose_estimate"])
        self.run_cmd(case, "v21_rigid_pose_fit", [self.python, SCRIPTS / "solve_v21_rigid_pose_fit.py", "--run-root", run_root, "--object-id", object_id], targets=[run_root / "measurements" / "object_geometry_mesh_pose" / object_id / "v21_pose_fit.json", run_root / "measurements" / "object_geometry_mesh_pose" / object_id / "v21_pose_fit_qc.json"])
        v18_compatible = run_root / "state" / "annotations_v18_compatible.json"
        v18_full = run_root / "state" / "annotations_v18_full_mano.json"
        if self.run_cmd(
            case,
            "v18_full_mano_annotations",
            [self.python, SCRIPTS / "assemble_v21_to_v18_annotations.py", "--run-root", run_root, "--object-id", object_id, "--repo-root", REPO_ROOT],
            targets=[v18_compatible, v18_full],
        ):
            if not v18_compatible.exists():
                raise RuntimeError(f"missing_v18_compatible_annotations: {v18_compatible}")
            shutil.copy2(v18_compatible, v18_full)
        v18_fit_dir = run_root / "measurements" / "object_geometry_mesh_pose" / object_id / "v18_icp_fit"
        completion_report = run_root / "measurements" / "object_geometry" / "v21_mesh_candidate" / object_id / "mesh_completion_report.json"
        self.run_cmd(
            case,
            "v18_compact_rigid_pose_fit",
            [
                self.python,
                SCRIPTS / "fit_v18_compact_rigid_object_pose.py",
                "--annotations",
                run_root / "state" / "annotations_v18_full_mano.json",
                "--completion-report",
                completion_report,
                "--object-id",
                object_id,
                "--output-dir",
                v18_fit_dir,
            ],
            targets=[v18_fit_dir],
        )
        v19_dir = run_root / "measurements" / "object_geometry_mesh_pose" / object_id / "v19_pose_graph"
        v18_pose_report = v18_fit_dir / "v18_compact_rigid_object_pose_fit_report.json"
        v18_pose_payload = load_json(v18_pose_report) if v18_pose_report.exists() else {}
        pose_rows = v18_pose_payload.get("pose_rows")
        if isinstance(pose_rows, list) and len(pose_rows) == 0:
            self.archive_targets([v19_dir])
            self.write_execution(
                case,
                "v19_rigid_pose_graph",
                "blocked_no_pose_observations",
                reason="v18_compact_rigid_pose_fit produced zero pose_rows for the strict stable-keyframe SAM2 track",
                pose_report=str(v18_pose_report),
            )
            self.write_execution(
                case,
                "adopted_object_pose",
                "blocked_no_v19_pose_graph",
                reason="no V19 pose graph is available because V18 compact fit produced zero pose observations",
                pose_report=str(v18_pose_report),
            )
        else:
            self.run_cmd(
                case,
                "v19_rigid_pose_graph",
                [
                    self.python,
                    SCRIPTS / "solve_v19_rigid_object_pose_graph.py",
                    "--annotations",
                    run_root / "state" / "annotations_v18_full_mano.json",
                    "--pose-report",
                    v18_pose_report,
                    "--completion-report",
                    completion_report,
                    "--object-id",
                    object_id,
                    "--output-dir",
                    v19_dir,
                ],
                targets=[v19_dir],
            )
            self.write_execution(
                case,
                "adopted_object_pose",
                "reran_now",
                mechanism="adopted pose is the just-reran v19_rigid_pose_graph report",
                output=str(v19_dir / "v19_rigid_object_pose_graph_report.json"),
            )
        contact_root = run_root / "measurements" / "contact_occlusion_nonpenetration"
        self.run_cmd(
            case,
            "v21_contact_occlusion_nonpenetration",
            [self.python, SCRIPTS / "build_v21_contact_occlusion_nonpenetration.py", "--run-root", run_root, "--object-id", object_id],
            targets=[
                contact_root / "contact_evidence.json",
                contact_root / "occlusion_evidence.json",
                contact_root / "nonpenetration_evidence.json",
            ],
        )
        self.run_cmd(case, "v21_physical_state", [self.python, SCRIPTS / "assemble_v21_state.py", "--run-root", run_root, "--object-id", object_id], targets=[run_root / "state" / "v21_uncertainty_state.json", run_root / "state" / "v21_agent_evidence.md"])
        self.write_execution(case, "v21_uncertainty_state", "reran_now", mechanism="produced by assemble_v21_state.py", output=str(run_root / "state" / "v21_uncertainty_state.json"))
        self.run_depth_registry_after_state(case, run_root)
        self.run_render_atoms(case, run_root, object_id)

    def run_depth_registry_selection(
        self,
        case: str,
        run_root: Path,
        *,
        annotations: Path | None = None,
        contact_report: Path | None = None,
    ) -> None:
        input_payload = load_json(run_root / "input" / "input_manifest.json")
        registry_dir = run_root / "measurements" / "camera_depth" / "v20_depth_registry"
        candidates: list[str | Path] = []
        depthpro = run_root / "measurements" / "depth_candidates" / "depthpro_full_frame" / "depthpro_full_frame_depth_v21.npz"
        if depthpro.exists():
            candidates.extend(["--candidate", f"depthpro|npz|{depthpro}|monocular_metric_depth|rgb|primary_candidate|1.0"])
        unidepth = run_root / "measurements" / "depth_candidates" / "unidepth_v2" / "unidepth_v2_depth.npz"
        if unidepth.exists():
            candidates.extend(["--candidate", f"unidepth_v2|npz|{unidepth}|monocular_metric_depth|rgb|comparator|0.8"])
        depth_anything = run_root / "measurements" / "depth_candidates" / "depth_anything_v2" / "depth_anything_v2_depth.npz"
        if depth_anything.exists():
            candidates.extend(["--candidate", f"depth_anything_v2|npz|{depth_anything}|relative_monocular_depth|rgb|diagnostic_scale_aligned|0.2"])
        cmd: list[str | Path] = [self.python, SCRIPTS / "build_v20_depth_candidate_registry.py", "--input-video", input_payload["primary_video"], "--output-dir", registry_dir]
        if input_payload.get("stereo_right_video"):
            cmd.extend(["--stereo-right-video", input_payload["stereo_right_video"]])
        cmd.extend(candidates)
        self.run_cmd(case, "depth_candidate_registry", cmd, targets=[registry_dir], allow_resume=False)
        selection_report = run_root / "measurements" / "camera_depth" / "v20_depth_selection_report.json"
        selection_bundle = run_root / "measurements" / "camera_depth" / "v20_depth_selection_bundle.json"
        selection_cmd: list[str | Path] = [
            self.python,
            SCRIPTS / "select_v20_depth_observation_bundle.py",
            "--registry",
            registry_dir / "depth_candidate_registry.json",
        ]
        if annotations is not None and annotations.exists():
            selection_cmd.extend(["--annotations", annotations])
        if contact_report is not None and contact_report.exists():
            selection_cmd.extend(["--contact-report", contact_report])
        selection_cmd.extend(["--output-report", selection_report, "--output-bundle", selection_bundle])
        self.run_cmd(
            case,
            "depth_selection_bundle",
            selection_cmd,
            targets=[selection_report, selection_bundle],
            allow_resume=False,
        )

    def run_depth_registry_after_state(self, case: str, run_root: Path) -> None:
        contact_report = run_root / "measurements" / "contact_occlusion_nonpenetration" / "v21_current" / "contact_occlusion_nonpenetration_report.json"
        self.run_depth_registry_selection(
            case,
            run_root,
            annotations=run_root / "state" / "annotations_v18_full_mano.json",
            contact_report=contact_report,
        )

    def run_render_atoms(self, case: str, run_root: Path, object_id: str) -> None:
        render_dir = run_root / "renders"
        self.run_cmd(
            case,
            "v21_segmentation_overlay",
            [
                self.python,
                SCRIPTS / "render_v21_segmentation_overlay.py",
                "--annotations",
                run_root / "state" / "annotations_v21_renderable.json",
                "--output-video",
                render_dir / "v21_segmentation_overlay.mp4",
                "--output-summary",
                render_dir / "v21_segmentation_overlay_summary.json",
            ],
            targets=[render_dir / "v21_segmentation_overlay.mp4", render_dir / "v21_segmentation_overlay_summary.json"],
        )
        self.run_cmd(
            case,
            "v21_visible_surface_overlay",
            [
                self.python,
                SCRIPTS / "render_v21_visible_surface_overlay.py",
                "--visible-geometry-annotations",
                run_root / "measurements" / "object_visible_surfaces" / "depthpro_local_grabcut" / object_id / "annotations_v19_visible_geometry.json",
                "--output-video",
                render_dir / "v21_visible_surface_overlay.mp4",
                "--output-summary",
                render_dir / "v21_visible_surface_overlay_summary.json",
            ],
            targets=[render_dir / "v21_visible_surface_overlay.mp4", render_dir / "v21_visible_surface_overlay_summary.json"],
        )
        self.run_cmd(case, "v21_hand_overlay", [self.python, SCRIPTS / "render_v21_hand_overlay.py", "--run-root", run_root, "--repo-root", REPO_ROOT], targets=[render_dir / "v21_hand_overlay.mp4"])
        self.run_cmd(case, "v21_integrated_overlay", [self.python, SCRIPTS / "render_v21_integrated_overlay.py", "--run-root", run_root, "--object-id", object_id, "--repo-root", REPO_ROOT], targets=[render_dir / "v21_integrated_overlay.mp4"])
        self.run_cmd(case, "v21_final_overlay", [self.python, SCRIPTS / "render_v21_full_annotation.py", "--run-root", run_root, "--object-id", object_id], targets=[render_dir / "v21_overlay.mp4", render_dir / "v21_world.mp4", render_dir / "v21_side_by_side.mp4"])
        self.write_execution(case, "v21_final_world", "reran_now", mechanism="produced by render_v21_full_annotation.py", output=str(render_dir / "v21_world.mp4"))
        self.write_execution(case, "v21_final_side_by_side", "reran_now", mechanism="produced by render_v21_full_annotation.py", output=str(render_dir / "v21_side_by_side.mp4"))
        final_map = {
            "v21_final_overlay": render_dir / "v21_overlay.mp4",
            "v21_final_world": render_dir / "v21_world.mp4",
            "v21_final_side_by_side": render_dir / "v21_side_by_side.mp4",
        }
        for atom, src in final_map.items():
            if src.exists():
                dst = self.overlay_root / case / atom / "overlay.mp4"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    def refresh_algorithm_overlays(self, case: str, run_root: Path, object_id: str, run_case: str) -> None:
        for spec in ALGORITHMS:
            atom = str(spec["id"])
            overlay_type = spec.get("overlay_type")
            data_rel = resolve_pattern(spec.get("data"), case, run_case, object_id)
            alt_rel = resolve_pattern(spec.get("alt_data"), case, run_case, object_id)
            native_rel = resolve_pattern(spec.get("native_overlay"), case, run_case, object_id)
            data_path = run_root / data_rel if data_rel else None
            alt_path = run_root / alt_rel if alt_rel else None
            native_overlay = run_root / native_rel if native_rel else None
            if (not data_path or not data_path.exists()) and alt_path and alt_path.exists():
                data_path = alt_path
            out_dir = self.overlay_root / case / atom
            overlay_path = out_dir / "overlay.mp4"
            if native_overlay and native_overlay.exists():
                out_dir.mkdir(parents=True, exist_ok=True)
                archives = self.archive_targets([overlay_path])
                shutil.copy2(native_overlay, overlay_path)
                self.write_execution(
                    case,
                    f"overlay_{atom}",
                    "native_overlay_copied",
                    source_native_overlay=str(native_overlay),
                    output=str(overlay_path),
                    archived_targets=archives,
                    overlay_type="native_overlay_copy",
                )
                continue
            if data_path and data_path.exists() and overlay_type:
                self.run_cmd(
                    case,
                    f"overlay_{atom}",
                    [self.python, SCRIPTS / "generate_algorithm_overlay.py", "--type", str(overlay_type), "--run-root", run_root, "--data-path", data_path, "--output-dir", out_dir],
                    targets=[overlay_path],
                )
                continue
            if data_path and data_path.exists():
                self.write_execution(
                    case,
                    f"overlay_{atom}",
                    "blocked_no_overlay_renderer",
                    reason="atomic algorithm has data but no overlay_type or native_overlay renderer in the runner inventory",
                    data_path=str(data_path),
                )
            else:
                self.write_execution(
                    case,
                    f"overlay_{atom}",
                    "blocked_missing_overlay_input",
                    reason="no current data, alternate data, or native overlay exists for this atom in the run root",
                    expected_data_path=str(run_root / data_rel) if data_rel else None,
                    expected_alt_data_path=str(run_root / alt_rel) if alt_rel else None,
                    expected_native_overlay_path=str(native_overlay) if native_overlay else None,
                    overlay_type=str(overlay_type) if overlay_type else None,
                )

    def materialize_atomic_outputs(self) -> None:
        audit_path = self.overlay_root / "atomic_algorithm_overlay_audit.json"
        self.run_cmd(
            "all_cases",
            "atomic_overlay_audit",
            [self.python, SCRIPTS / "audit_v21_atomic_algorithm_overlays.py", "--overlay-root", self.overlay_root, "--output", audit_path],
            targets=[audit_path],
            allow_resume=False,
        )
        self.run_cmd(
            "all_cases",
            "atomic_overlay_qc",
            [self.python, SCRIPTS / "write_v21_atomic_overlay_qc.py", "--audit", audit_path, "--output", self.overlay_root / "atomic_overlay_qc_materialization.json"],
            targets=[self.overlay_root / "atomic_overlay_qc_materialization.json"],
            allow_resume=False,
        )
        self.run_cmd(
            "all_cases",
            "materialize_atomic_results",
            [self.python, SCRIPTS / "materialize_v21_atomic_algorithm_results.py", "--audit", audit_path, "--output-root", self.atomic_root, "--clear"],
            targets=[],
            allow_resume=False,
        )

    def run_case(self, case: str) -> None:
        run_root, object_id, run_case = self.case_info(case)
        self.summary["cases"].append({"case": case, "run_root": str(run_root), "object_id": object_id})
        if self.args.refresh_overlays_only:
            self.refresh_algorithm_overlays(case, run_root, object_id, run_case)
            self.mark_non_rerun_inventory(case, rerun_skipped_reason="--refresh-overlays-only requested; pipeline atoms were not rerun")
            return
        self.input_contract(case, run_root)
        self.run_input_adapters(case, run_root)
        self.run_depth_atoms(case, run_root)
        self.run_hand_atoms(case, run_root)
        self.run_keyframe_and_segmentation_atoms(case, run_root, object_id)
        self.run_geometry_pose_render_atoms(case, run_root, object_id)
        self.refresh_algorithm_overlays(case, run_root, object_id, run_case)
        self.mark_non_rerun_inventory(case)

    def run(self) -> None:
        for case in self.args.cases:
            self.run_case(case)
        self.materialize_atomic_outputs()
        self.summary["finished_at"] = utc_now()
        write_json(self.atomic_root / "logs" / "atomic_suite_execution_summary.json", self.summary)
        print(json.dumps({"status": "ok", "summary": str(self.atomic_root / "logs" / "atomic_suite_execution_summary.json")}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun V21 Pico/living-room atomic algorithm suite with per-atom execution records.")
    parser.add_argument("--atomic-root", default="output/atomic_algorithm_runs_pico_living_room_20260630")
    parser.add_argument("--overlay-root", default="output/v21_per_algorithm_results")
    parser.add_argument("--cases", nargs="+", default=["pico", "living_room"], choices=sorted(RUNS.keys()))
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    parser.add_argument("--rtmlib-python-bin", default=str(DEFAULT_RTMLIB_PYTHON))
    parser.add_argument("--wilor-python-bin", default=str(DEFAULT_WILOR_PYTHON))
    parser.add_argument("--include-heavy", action="store_true", help="Run GPU/heavy model atoms instead of marking them not_rerun_heavy_disabled.")
    parser.add_argument("--refresh-overlays-only", action="store_true", help="Do not rerun pipeline atoms; generate/copy overlays for existing atomic data and materialize audit/QC records.")
    parser.add_argument("--active-mano-max-frames", type=int, default=5)
    parser.add_argument("--active-mano-iters", type=int, default=2)
    parser.add_argument("--compute-target", default=os.environ.get("V21_COMPUTE_TARGET", "ssh -p 57938 zjh@115.190.235.210"))
    parser.add_argument("--cuda-visible-devices", default=None, help="CUDA_VISIBLE_DEVICES value for heavy model atoms, for example 2 or 3.")
    parser.add_argument("--sam2-repo", default=str(DEFAULT_SAM2_REPO))
    parser.add_argument("--sam2-checkpoint", default=str(DEFAULT_SAM2_CKPT))
    parser.add_argument("--sam2-model-cfg", default=DEFAULT_SAM2_MODEL_CFG)
    parser.add_argument("--owlv2-model", default=str(DEFAULT_OWLV2_MODEL))
    parser.add_argument("--depthpro-repo", default=str(DEFAULT_DEPTHPRO_REPO))
    parser.add_argument("--archive-existing", action="store_true", default=True)
    parser.add_argument("--resume-success", action="store_true", help="Skip atoms that already have reran_now execution records and existing target outputs.")
    return parser.parse_args()


if __name__ == "__main__":
    AtomicSuiteRunner(parse_args()).run()
