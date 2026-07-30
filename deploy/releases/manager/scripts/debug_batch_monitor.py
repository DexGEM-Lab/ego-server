#!/usr/bin/env python3
"""Read-only TUI for supervising batch annotation tmux windows and run roots.

This script is intentionally diagnostic-only. It never attaches to tmux and never
sends input to a pane; it only lists windows, captures pane text, and summarizes
batch run directories.
"""

from __future__ import annotations

import argparse
import curses
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Iterable

DEFAULT_ROOTS = (
    "/home/zjh/data/v22_wave_batch_runs",
    "/home/zjh/data/v22_stage_batch_runs",
    "/home/zjh/data/v22_parallel_runs",
)
DEFAULT_SESSION = "ego_annotation"
RUN_ROOT_RE = re.compile(
    r"(/[^\s;'\"`]+/v22_(?:wave_batch_runs|stage_batch_runs|parallel_runs)/[^\s;'\"`]+)"
)


@dataclass
class WindowInfo:
    index: str
    name: str
    active: bool
    command: str
    pane_pid: str
    detected_root: str | None = None


@dataclass
class RunSummary:
    root: str
    name: str
    mtime: float | None = None
    exists: bool = True
    item_count: int = 0
    wave_count: int = 0
    waves: dict[str, int] | None = None
    topology: str = ""
    manifest_status_counts: dict[str, int] | None = None
    packages: int = 0
    overlays: int = 0
    resident_reports: dict[str, int] | None = None
    reports_present: dict[str, bool] | None = None
    last_event: str = ""
    current_stage: str = ""
    current_wave: str = ""
    report_status: str = ""
    role: str = "RUN"
    validity: str = "unknown"
    note: str = ""
    error: str = ""


@dataclass
class AgentInfo:
    run_root: str
    wave_id: str
    wave_index: int
    agent_id: str
    runner_id: str
    item_id: str
    case_id: str
    run_root_item: str
    manifest_status: str
    current_stage: str
    status: str
    gpu_id: str = ""
    returncode: str = ""
    log_path: str = ""
    artifacts: dict[str, bool] | None = None
    last_event_at: str = ""


@dataclass
class Row:
    kind: str
    title: str
    detail: str
    window: WindowInfo | None = None
    run: RunSummary | None = None
    agent: AgentInfo | None = None


class CommandRunner:
    def __init__(self, ssh_host: str | None = None) -> None:
        self.ssh_host = ssh_host

    def run_shell(self, script: str, timeout: int = 10) -> subprocess.CompletedProcess[str]:
        if self.ssh_host:
            cmd = ["ssh", self.ssh_host, "bash", "-lc", shlex.quote(script)]
        else:
            cmd = ["bash", "-lc", script]
        return subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )

    def run_argv(self, argv: list[str], timeout: int = 10) -> subprocess.CompletedProcess[str]:
        return self.run_shell(shlex.join(argv), timeout=timeout)


def json_loads_or(value: str, default: Any) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return default


def utc_from_mtime(mtime: float | None) -> str:
    if not mtime:
        return ""
    return dt.datetime.fromtimestamp(mtime, tz=dt.timezone.utc).strftime("%m-%d %H:%MZ")


def shorten(text: str, width: int) -> str:
    text = text.replace("\n", " ")
    if width <= 1:
        return text[:width]
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def detect_run_root(text: str) -> str | None:
    matches = RUN_ROOT_RE.findall(text)
    if not matches:
        return None
    cleaned = [m.rstrip(".,);]") for m in matches]
    return cleaned[-1]


def collect_windows(runner: CommandRunner, session: str, capture_lines: int) -> list[WindowInfo]:
    proc = runner.run_argv(
        [
            "tmux",
            "list-windows",
            "-t",
            session,
            "-F",
            "#{window_index}\t#{window_name}\t#{window_active}\t#{pane_current_command}\t#{pane_pid}",
        ],
        timeout=5,
    )
    if proc.returncode != 0:
        return []

    windows: list[WindowInfo] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        index, name, active, command, pane_pid = parts[:5]
        info = WindowInfo(
            index=index,
            name=name,
            active=active == "1",
            command=command,
            pane_pid=pane_pid,
        )
        target = f"{session}:{index}"
        cap = runner.run_argv(
            ["tmux", "capture-pane", "-t", target, "-p", "-S", f"-{capture_lines}"],
            timeout=5,
        )
        if cap.returncode == 0:
            info.detected_root = detect_run_root(cap.stdout)
        windows.append(info)
    return windows


def collect_run_summaries(runner: CommandRunner, roots: Iterable[str], max_runs: int) -> list[RunSummary]:
    py = r'''
import glob
import json
import os
import sys
from collections import Counter

roots = json.loads(sys.argv[1])
max_runs = int(sys.argv[2])
report_names = [
    "batch_summary.json",
    "wave_summary.json",
    "resident_model_summary.json",
    "gpu_summary.json",
    "token_summary.json",
    "worker_summary.json",
    "visual_report.html",
]
stage_patterns = {
    "unidepth": "reports/resident_workers/unidepth_v2_depth_resident/**/resident_unidepth_worker_report.json",
    "wilor": "reports/resident_workers/wilor_v21_hand_candidates_resident/**/resident_wilor_worker_report.json",
    "droid": "reports/resident_workers/droid_camera_trajectory_resident/**/resident_droid_worker_report.json",
    "hawor": "reports/resident_workers/hawor_metric_hands_resident/**/resident_hawor_worker_report.json",
}

def safe_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def tail_event(root):
    candidates = [
        os.path.join(root, "logs", "stage_batch_events.jsonl"),
        os.path.join(root, "logs", "stage_batch_events.ndjson"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 262144), os.SEEK_SET)
                chunk = f.read().decode("utf-8", errors="replace")
            lines = [ln for ln in chunk.splitlines() if ln.strip()]
            for line in reversed(lines):
                obj = safe_line_json(line)
                if obj:
                    return obj
                return {"event": line[-240:]}
        except Exception as exc:
            return {"event": "event_read_error", "error": str(exc)}
    return {}

def safe_line_json(line):
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None

def count(pattern):
    return len(glob.glob(pattern, recursive=True))

def as_int(value, default=0):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default

def classify_run(name, root, topology, item_count, wave_count, packages, reports_present, resident_reports, last_event, batch_summary):
    marker_invalid = os.path.exists(os.path.join(root, "STAGE_MAJOR_INVALID_STOPPED.txt"))
    has_final_report = bool(reports_present.get("batch_summary.json"))
    if name == "v22_wave32_preflight_fix_20260710T015402Z":
        return "CURRENT_PREFLIGHT_RUNNING", "live", "active corrected 32-item wave-major preflight; not final 256 production"
    if "wave32_preflight_gpu1_5_20260709T171332Z" in name:
        return "FAILED_PREFLIGHT", "failed", "older 32-item preflight; failed in HaWoR stdout protocol before fix"
    if marker_invalid or "parallelsetup_gpu0to7_20260709T090324Z" in name:
        return "INVALID_STAGE_MAJOR_STOPPED", "invalid", "stopped negative evidence; stage-major ordering violates wave-major requirement"
    if name.startswith("v22_stage_batch_resident256_"):
        return "SUPERSEDED_256_ATTEMPT", "superseded", "early 256 stage-batch launch/setup attempt; not a delivery result"
    if "smoke" in name.lower() or item_count <= 2:
        return "SMOKE_TEST", "diagnostic", "small mechanism test only"
    if name.startswith("v22_parallel_"):
        complete = isinstance(batch_summary, dict) and batch_summary.get("manifest_counts", {}).get("completed") == item_count and packages >= item_count > 0
        if complete:
            return "OLD_ITEM_PARALLEL_COMPLETED", "old_complete_not_wave_major", "old MVP item-level 256 result; packages exist but no wave-major/resident-worker proof"
        return "OLD_ITEM_PARALLEL_RUN", "old", "old MVP item-level parallel run; not wave-major"
    if topology == "wave_major_end_to_end" and packages >= item_count > 0 and has_final_report:
        return "WAVE_MAJOR_RESULT", "completed", "completed wave-major run"
    if topology == "wave_major_end_to_end":
        return "WAVE_MAJOR_RUN", "in_progress_or_incomplete", "wave-major run without final package/report closure"
    return "RUN", "unknown", "unclassified historical run"

run_dirs = []
for root in roots:
    if not os.path.isdir(root):
        continue
    for child in glob.glob(os.path.join(root, "*")):
        if os.path.isdir(child):
            try:
                mtime = os.path.getmtime(child)
            except OSError:
                mtime = 0.0
            run_dirs.append((mtime, child))
run_dirs.sort(reverse=True)
run_dirs = run_dirs[:max_runs]

out = []
for mtime, root in run_dirs:
    item_count = 0
    wave_counts = Counter()
    status_counts = Counter()
    topology = ""
    wave_count = 0
    manifest = safe_json(os.path.join(root, "batch_manifest.json"))
    if isinstance(manifest, dict):
        entries = manifest.get("entries") or manifest.get("items") or []
        if isinstance(entries, list):
            item_count = len(entries)
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                wave = entry.get("wave_id") or entry.get("batch_id") or "none"
                wave_counts[str(wave)] += 1
                status_counts[str(entry.get("status") or "missing")] += 1
        waves = manifest.get("waves")
        if isinstance(waves, list):
            wave_count = len(waves)
        elif isinstance(waves, dict):
            wave_count = len(waves)
        else:
            wave_count = len(wave_counts)
        topology = str(manifest.get("execution_topology") or "")
    reports_present = {name: os.path.exists(os.path.join(root, "reports", name)) for name in report_names}
    batch_summary = safe_json(os.path.join(root, "reports", "batch_summary.json"))
    report_status = ""
    packages = count(os.path.join(root, "entries", "*", "packages", "*.zip"))
    overlays = count(os.path.join(root, "entries", "*", "renders", "*.mp4")) + count(os.path.join(root, "entries", "*", "outputs", "*.mp4"))
    if isinstance(batch_summary, dict):
        report_status = str(batch_summary.get("status") or batch_summary.get("job_status") or "")
        artifact_counts = batch_summary.get("artifact_counts")
        if isinstance(artifact_counts, dict):
            report_packages = max(as_int(artifact_counts.get("packages_for_manifest_completed")), as_int(artifact_counts.get("packages_total_on_disk")))
            report_overlays = as_int(artifact_counts.get("overlays"))
            if report_packages:
                packages = report_packages
            if report_overlays:
                overlays = report_overlays
    ev = tail_event(root)
    last_event = str(ev.get("event") or "") if isinstance(ev, dict) else ""
    current_stage = str(ev.get("stage") or "") if isinstance(ev, dict) else ""
    current_wave = str(ev.get("wave_id") or "") if isinstance(ev, dict) else ""
    resident_reports = {key: count(os.path.join(root, pattern)) for key, pattern in stage_patterns.items()}
    role, validity, note = classify_run(os.path.basename(root), root, topology, item_count, wave_count, packages, reports_present, resident_reports, last_event, batch_summary)
    out.append({
        "root": root,
        "name": os.path.basename(root),
        "mtime": mtime,
        "item_count": item_count,
        "wave_count": wave_count,
        "waves": dict(wave_counts),
        "topology": topology,
        "manifest_status_counts": dict(status_counts),
        "packages": packages,
        "overlays": overlays,
        "resident_reports": resident_reports,
        "reports_present": reports_present,
        "last_event": last_event,
        "current_stage": current_stage,
        "current_wave": current_wave,
        "report_status": report_status,
        "role": role,
        "validity": validity,
        "note": note,
    })
print(json.dumps(out, ensure_ascii=True))
'''
    proc = runner.run_shell(
        "python3 - " + shlex.quote(json.dumps(list(roots))) + " " + shlex.quote(str(max_runs)) + " <<'PY'\n" + py + "\nPY",
        timeout=20,
    )
    if proc.returncode != 0:
        return [RunSummary(root="", name="run-scan-error", exists=False, error=proc.stderr.strip() or proc.stdout.strip())]
    data = json_loads_or(proc.stdout, [])
    summaries: list[RunSummary] = []
    if not isinstance(data, list):
        return summaries
    for item in data:
        if not isinstance(item, dict):
            continue
        summaries.append(
            RunSummary(
                root=str(item.get("root") or ""),
                name=str(item.get("name") or ""),
                mtime=item.get("mtime") if isinstance(item.get("mtime"), (int, float)) else None,
                item_count=int(item.get("item_count") or 0),
                wave_count=int(item.get("wave_count") or 0),
                waves=item.get("waves") if isinstance(item.get("waves"), dict) else {},
                topology=str(item.get("topology") or ""),
                manifest_status_counts=item.get("manifest_status_counts") if isinstance(item.get("manifest_status_counts"), dict) else {},
                packages=int(item.get("packages") or 0),
                overlays=int(item.get("overlays") or 0),
                resident_reports=item.get("resident_reports") if isinstance(item.get("resident_reports"), dict) else {},
                reports_present=item.get("reports_present") if isinstance(item.get("reports_present"), dict) else {},
                last_event=str(item.get("last_event") or ""),
                current_stage=str(item.get("current_stage") or ""),
                current_wave=str(item.get("current_wave") or ""),
                report_status=str(item.get("report_status") or ""),
                role=str(item.get("role") or "RUN"),
                validity=str(item.get("validity") or "unknown"),
                note=str(item.get("note") or ""),
            )
        )
    return summaries


def collect_agents(runner: CommandRunner, run: RunSummary) -> list[AgentInfo]:
    if not run.root:
        return []
    py = r'''
import glob
import json
import os
import sys

root = sys.argv[1]

def safe_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_events(root):
    path = os.path.join(root, "logs", "stage_batch_events.jsonl")
    if not os.path.exists(path):
        path = os.path.join(root, "logs", "stage_batch_events.ndjson")
    events = []
    if os.path.exists(path):
        for line in open(path, "r", encoding="utf-8", errors="replace"):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                events.append(obj)
    return events

def artifact_flags(item_root):
    checks = {
        "depth": ["measurements/depth_candidates/unidepth_v2/unidepth_v2_depth.npz"],
        "wilor": ["measurements/hand_candidates/wilor_v21/wilor_qc.json", "measurements/hand_candidates/wilor_v21/wilor_raw_hands.json"],
        "droid": ["measurements/camera_trajectory/droid_full_frame/droid_qc.json"],
        "hawor": ["measurements/metric_hands/hawor_v22/hawor_qc.json", "measurements/metric_hands/hawor_v22/v22_metric_hands_stage.json"],
        "package": ["packages/*.zip"],
    }
    out = {}
    for key, rels in checks.items():
        found = False
        for rel in rels:
            if glob.glob(os.path.join(item_root, rel)):
                found = True
                break
        out[key] = found
    return out

events = load_events(root)
current_wave = ""
current_stage = ""
for ev in events:
    if ev.get("wave_id"):
        current_wave = str(ev.get("wave_id"))
    if ev.get("stage"):
        current_stage = str(ev.get("stage"))

manifests = sorted(glob.glob(os.path.join(root, "wave_manifests", "*.json")))
if not current_wave and manifests:
    current_wave = os.path.splitext(os.path.basename(manifests[-1]))[0].replace("_manifest", "")
manifest_path = os.path.join(root, "wave_manifests", f"{current_wave}_manifest.json") if current_wave else ""
if not os.path.exists(manifest_path) and manifests:
    manifest_path = manifests[-1]
manifest = safe_json(manifest_path) or {}
wave_id = str(manifest.get("wave_id") or current_wave or os.path.splitext(os.path.basename(manifest_path))[0].replace("_manifest", ""))
wave_index = int(manifest.get("wave_index") or 0)
entries = manifest.get("entries") or manifest.get("items") or []

agent_events = {}
for ev in events:
    rid = ev.get("runner_id")
    if rid:
        agent_events[str(rid)] = ev

out = []
for entry in entries:
    if not isinstance(entry, dict):
        continue
    agent_id = str(entry.get("agent_id") or "")
    runner_id = f"{wave_id}_{agent_id}" if agent_id else ""
    log_path = os.path.join(root, "logs", "stage_batch_item_agent_processes", f"{runner_id}.log") if runner_id else ""
    ev = agent_events.get(runner_id, {})
    event_name = ev.get("event")
    if event_name == "item_agent_finish":
        rc = ev.get("returncode")
        status = "completed" if rc == 0 else f"failed_rc_{rc}"
    elif event_name == "item_agent_start":
        status = "running"
    elif current_stage == "item_agents" and wave_id == current_wave:
        status = "pending_item_agent"
    else:
        status = f"waiting_for_{current_stage or 'pipeline'}"
    item_root = str(entry.get("run_root") or "")
    out.append({
        "run_root": root,
        "wave_id": wave_id,
        "wave_index": wave_index,
        "agent_id": agent_id,
        "runner_id": runner_id,
        "item_id": str(entry.get("item_id") or ""),
        "case_id": str(entry.get("case_id") or os.path.basename(item_root)),
        "run_root_item": item_root,
        "manifest_status": str(entry.get("status") or ""),
        "current_stage": current_stage,
        "status": status,
        "gpu_id": str(ev.get("gpu_id") or ""),
        "returncode": "" if ev.get("returncode") is None else str(ev.get("returncode")),
        "log_path": log_path if os.path.exists(log_path) else "",
        "artifacts": artifact_flags(item_root) if item_root else {},
        "last_event_at": str(ev.get("at") or ""),
    })
print(json.dumps(out, ensure_ascii=True))
'''
    proc = runner.run_shell("python3 - " + shlex.quote(run.root) + " <<'PY'\n" + py + "\nPY", timeout=15)
    if proc.returncode != 0:
        return [AgentInfo(run_root=run.root, wave_id="", wave_index=0, agent_id="agent-scan-error", runner_id="", item_id="", case_id=proc.stderr.strip() or proc.stdout.strip(), run_root_item="", manifest_status="", current_stage="", status="error")]
    data = json_loads_or(proc.stdout, [])
    agents: list[AgentInfo] = []
    if not isinstance(data, list):
        return agents
    for item in data:
        if not isinstance(item, dict):
            continue
        agents.append(
            AgentInfo(
                run_root=str(item.get("run_root") or run.root),
                wave_id=str(item.get("wave_id") or ""),
                wave_index=int(item.get("wave_index") or 0),
                agent_id=str(item.get("agent_id") or ""),
                runner_id=str(item.get("runner_id") or ""),
                item_id=str(item.get("item_id") or ""),
                case_id=str(item.get("case_id") or ""),
                run_root_item=str(item.get("run_root_item") or ""),
                manifest_status=str(item.get("manifest_status") or ""),
                current_stage=str(item.get("current_stage") or ""),
                status=str(item.get("status") or ""),
                gpu_id=str(item.get("gpu_id") or ""),
                returncode=str(item.get("returncode") or ""),
                log_path=str(item.get("log_path") or ""),
                artifacts=item.get("artifacts") if isinstance(item.get("artifacts"), dict) else {},
                last_event_at=str(item.get("last_event_at") or ""),
            )
        )
    agents.sort(key=lambda a: a.agent_id)
    return agents


def is_primary_row(run: RunSummary) -> bool:
    return run.role in {"CURRENT_PREFLIGHT_RUNNING", "WAVE_MAJOR_RUN", "WAVE_MAJOR_RESULT"}


def make_rows(windows: list[WindowInfo], runs: list[RunSummary], include_all_windows: bool, show_history: bool) -> list[Row]:
    visible_runs = runs if show_history else [run for run in runs if is_primary_row(run)]
    if not visible_runs and runs:
        visible_runs = runs[:1]
    visible_roots = {run.root for run in visible_runs if run.root}
    run_by_root = {run.root: run for run in visible_runs if run.root}
    seen_roots: set[str] = set()
    rows: list[Row] = []

    for window in windows:
        run = run_by_root.get(window.detected_root or "")
        if run:
            seen_roots.add(run.root)
            rows.append(row_for_window(window, run))
        elif include_all_windows and (show_history or not visible_roots):
            title = f"tmux {window.index}:{window.name}"
            detail = f"cmd={window.command} pid={window.pane_pid}"
            rows.append(Row(kind="window", title=title, detail=detail, window=window))

    for run in visible_runs:
        if run.root and run.root not in seen_roots:
            rows.append(row_for_run(run))

    def sort_key(row: Row) -> tuple[int, int, float, str]:
        role_priority = {
            "CURRENT_PREFLIGHT_RUNNING": 0,
            "WAVE_MAJOR_RUN": 1,
            "WAVE_MAJOR_RESULT": 2,
            "FAILED_PREFLIGHT": 3,
            "INVALID_STAGE_MAJOR_STOPPED": 4,
            "OLD_ITEM_PARALLEL_COMPLETED": 5,
            "SUPERSEDED_256_ATTEMPT": 6,
            "SMOKE_TEST": 7,
        }
        running = 0 if row.window and row.window.command not in {"bash", "zsh", "sh"} else 1
        role_rank = role_priority.get(row.run.role if row.run else "", 8)
        mtime = -(row.run.mtime or 0.0) if row.run else 0.0
        return (role_rank, running, mtime, row.title)

    rows.sort(key=sort_key)
    return rows


def row_for_window(window: WindowInfo, run: RunSummary) -> Row:
    title = f"{run.role} tmux {window.index}:{window.name}  {run.name}"
    detail = format_run_detail(run) + f"  cmd={window.command} pid={window.pane_pid}"
    return Row(kind="window-run", title=title, detail=detail, window=window, run=run)


def row_for_run(run: RunSummary) -> Row:
    title = f"{run.role} run {run.name}"
    return Row(kind="run", title=title, detail=format_run_detail(run), run=run)


def row_for_agent(agent: AgentInfo) -> Row:
    title = f"{agent.wave_id} {agent.agent_id} {agent.item_id}"
    return Row(kind="agent", title=title, detail=format_agent_detail(agent), agent=agent)


def agent_progress(agent: AgentInfo) -> tuple[int, int, str, str]:
    flags = agent.artifacts or {}
    order = [("depth", "D"), ("wilor", "W"), ("droid", "Dr"), ("hawor", "H"), ("package", "P")]
    done = sum(1 for key, _ in order if flags.get(key))
    total = len(order)
    badges = " ".join(label if flags.get(key) else "--" for key, label in order)
    next_stage = next((key for key, _ in order if not flags.get(key)), "done")
    return done, total, badges, next_stage


def format_agent_detail(agent: AgentInfo) -> str:
    done, total, badges, next_stage = agent_progress(agent)
    context = "log" if agent.log_path else "waiting-context"
    return (
        f"progress={done}/{total} [{badges}] next={next_stage} status={agent.status} "
        f"stage={agent.current_stage or '-'} gpu={agent.gpu_id or '-'} rc={agent.returncode or '-'} "
        f"ctx={context} case={shorten(agent.case_id, 70)}"
    )


def format_run_detail(run: RunSummary) -> str:
    report_bits = run.resident_reports or {}
    reports = " ".join(f"{k}:{v}" for k, v in report_bits.items())
    status = run.report_status or run.last_event or "no-report"
    wave = run.current_wave or "-"
    stage = run.current_stage or "-"
    updated = utc_from_mtime(run.mtime)
    return (
        f"validity={run.validity} items={run.item_count} waves={run.wave_count} "
        f"pkgs={run.packages} overlays={run.overlays} topo={run.topology or '-'} "
        f"wave={wave} stage={stage} reports[{reports}] last={status} updated={updated} "
        f"note={run.note}"
    )


def draw_text_lines(stdscr: Any, lines: list[str], top: int, header: list[str], selected: int | None = None) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    y = 0
    for line in header[: max(0, height - 1)]:
        stdscr.addnstr(y, 0, shorten(line, width - 1), width - 1, curses.A_BOLD if y == 0 else 0)
        y += 1
    body_height = max(0, height - y - 1)
    visible = lines[top : top + body_height]
    for offset, line in enumerate(visible):
        attr = curses.A_REVERSE if selected is not None and top + offset == selected else 0
        stdscr.addnstr(y + offset, 0, shorten(line, width - 1), width - 1, attr)
    footer = "q quit  r refresh  Enter open read-only  Up/Down select"
    stdscr.addnstr(height - 1, 0, shorten(footer, width - 1), width - 1, curses.A_DIM)
    stdscr.refresh()


def fetch_capture(runner: CommandRunner, session: str, window: WindowInfo, scrollback: int) -> tuple[list[str], str]:
    target = f"{session}:{window.index}"
    proc = runner.run_argv(
        ["tmux", "capture-pane", "-t", target, "-p", "-S", f"-{scrollback}"],
        timeout=8,
    )
    if proc.returncode != 0:
        return [proc.stderr.strip() or proc.stdout.strip() or "capture failed"], target
    return proc.stdout.splitlines(), target


def fetch_agent_detail(runner: CommandRunner, agent: AgentInfo, event_lines: int) -> list[str]:
    payload = {
        "run_root": agent.run_root,
        "wave_id": agent.wave_id,
        "agent_id": agent.agent_id,
        "runner_id": agent.runner_id,
        "item_id": agent.item_id,
        "case_id": agent.case_id,
        "item_root": agent.run_root_item,
        "status": agent.status,
        "current_stage": agent.current_stage,
        "log_path": agent.log_path,
        "artifacts": agent.artifacts or {},
        "event_lines": event_lines,
    }
    py = r'''
import json
import os
import sys

payload = json.loads(sys.argv[1])
root = payload["run_root"]
wave = payload["wave_id"]
runner = payload["runner_id"]
log_path = payload.get("log_path") or ""
flags = payload.get("artifacts") or {}
order = [("depth", "UniDepth"), ("wilor", "WiLoR"), ("droid", "DROID"), ("hawor", "HaWoR"), ("package", "Package")]

def yn(key):
    return "done" if flags.get(key) else "pending"

def compact_event(ev):
    at = ev.get("at", "?")
    name = ev.get("event", "event")
    parts = [f"[EVENT] {at} {name}"]
    if ev.get("wave_id"):
        parts.append(f"wave={ev.get('wave_id')}")
    if ev.get("stage"):
        parts.append(f"stage={ev.get('stage')}")
    if ev.get("runner_id"):
        parts.append(f"runner={ev.get('runner_id')}")
    if ev.get("gpu_id"):
        parts.append(f"gpu={ev.get('gpu_id')}")
    if ev.get("returncode") is not None:
        parts.append(f"rc={ev.get('returncode')}")
    response = ev.get("response")
    if isinstance(response, dict):
        if response.get("model_load_count") is not None:
            parts.append(f"model_load={response.get('model_load_count')}")
        if response.get("items_failed") is not None:
            parts.append(f"items_failed={response.get('items_failed')}")
        if response.get("frame_rows_inferred") is not None:
            parts.append(f"rows={response.get('frame_rows_inferred')}")
        if response.get("rows_inferred") is not None:
            parts.append(f"rows={response.get('rows_inferred')}")
        if response.get("sequence_batch_count") is not None:
            parts.append(f"seq={response.get('sequence_batch_count')}")
    return "  ".join(parts)

print(f"[SYSTEM] Read-only monitor for {payload['runner_id']}. No input is sent to the process.")
print(f"[ASSIGNMENT] wave={wave}  agent={payload['agent_id']}  item={payload['item_id']}")
print(f"[ASSIGNMENT] case={payload['case_id']}")
print(f"[STATE] status={payload['status']}  current_stage={payload['current_stage']}")
progress = "  ".join(f"{label}={yn(key)}" for key, label in order)
print(f"[PROGRESS] {progress}")
print("")
print("[CONTEXT] Agent dialogue/context")
if log_path and os.path.exists(log_path):
    for raw in open(log_path, "r", encoding="utf-8", errors="replace"):
        line = raw.rstrip()
        if line.startswith("START "):
            print(f"[AGENT] started {line[6:]}")
        elif line.startswith("END "):
            print(f"[AGENT] finished {line[4:]}")
        elif line.startswith("CMD "):
            cmd = line[4:]
            cmd = cmd.replace(payload["run_root"], "$RUN_ROOT")
            cmd = cmd.replace(payload["item_root"], "$ITEM_ROOT")
            print(f"[AGENT] command {cmd}")
        else:
            print(f"[AGENT] {line}")
else:
    print(f"[AGENT] not started yet; waiting for pipeline stage `{payload['current_stage']}` to finish before this item-agent worker starts")
print("")
print("[CONTEXT] Recent relevant events")
paths = [os.path.join(root, "logs", "stage_batch_events.jsonl"), os.path.join(root, "logs", "stage_batch_events.ndjson")]
events = []
for path in paths:
    if not os.path.exists(path):
        continue
    for raw in open(path, "r", encoding="utf-8", errors="replace"):
        try:
            ev = json.loads(raw)
        except Exception:
            continue
        if ev.get("runner_id") == runner or ev.get("wave_id") == wave:
            events.append(ev)
    break
for ev in events[-int(payload.get("event_lines") or 80):]:
    print(compact_event(ev))
'''
    proc = runner.run_shell("python3 - " + shlex.quote(json.dumps(payload)) + " <<'PY'\n" + py + "\nPY", timeout=10)
    text = proc.stdout if proc.returncode == 0 else (proc.stderr or proc.stdout)
    return text.splitlines()


def fetch_run_detail(runner: CommandRunner, run: RunSummary, event_lines: int) -> list[str]:
    root = run.root
    script = f'''
ROOT={shlex.quote(root)}
echo "root: $ROOT"
echo ""
if test -f "$ROOT/reports/batch_summary.json"; then
  echo "reports/batch_summary.json"
  python3 -m json.tool "$ROOT/reports/batch_summary.json" 2>/dev/null | head -120
else
  echo "reports/batch_summary.json: missing"
fi
echo ""
for f in "$ROOT/reports/wave_summary.json" "$ROOT/reports/resident_model_summary.json" "$ROOT/reports/gpu_summary.json" "$ROOT/reports/token_summary.json" "$ROOT/reports/worker_summary.json"; do
  if test -f "$f"; then echo "present: $f"; else echo "missing: $f"; fi
done
echo ""
echo "recent events:"
if test -f "$ROOT/logs/stage_batch_events.jsonl"; then tail -{event_lines} "$ROOT/logs/stage_batch_events.jsonl"; elif test -f "$ROOT/logs/stage_batch_events.ndjson"; then tail -{event_lines} "$ROOT/logs/stage_batch_events.ndjson"; else echo "no event log"; fi
'''
    proc = runner.run_shell(script, timeout=10)
    text = proc.stdout if proc.returncode == 0 else (proc.stderr or proc.stdout)
    return text.splitlines()


def view_row(stdscr: Any, runner: CommandRunner, session: str, row: Row, scrollback: int, event_lines: int) -> str:
    top = 0
    lines: list[str] = []
    title = row.title

    def refresh() -> None:
        nonlocal lines, title
        if row.window:
            lines, target = fetch_capture(runner, session, row.window, scrollback)
            title = f"READ ONLY tmux capture: {target}  {row.window.name}"
        elif row.agent:
            lines = fetch_agent_detail(runner, row.agent, event_lines)
            title = f"READ ONLY agent: {row.agent.runner_id}"
        elif row.run:
            lines = fetch_run_detail(runner, row.run, event_lines)
            title = f"READ ONLY run detail: {row.run.name}"
        else:
            lines = ["No target"]

    refresh()
    stdscr.timeout(1000)
    while True:
        height, width = stdscr.getmaxyx()
        header = [title, "b/Esc back  q quit  r refresh  Up/Down/PgUp/PgDn scroll  (no tmux input is sent)"]
        body_top = len(header)
        body_height = max(1, height - body_top - 1)
        max_top = max(0, len(lines) - body_height)
        top = max(0, min(top, max_top))
        stdscr.erase()
        for y, line in enumerate(header[:height]):
            attr = curses.A_BOLD if y == 0 else curses.A_DIM
            stdscr.addnstr(y, 0, shorten(line, width - 1), width - 1, attr)
        for i, line in enumerate(lines[top : top + body_height]):
            stdscr.addnstr(body_top + i, 0, shorten(line, width - 1), width - 1)
        footer = f"lines {top + 1}-{min(len(lines), top + body_height)} / {len(lines)}"
        stdscr.addnstr(height - 1, 0, shorten(footer, width - 1), width - 1, curses.A_DIM)
        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            return "quit"
        if key in (ord("b"), ord("B"), 27):
            return "back"
        if key in (ord("r"), ord("R")):
            refresh()
            top = max(0, len(lines) - body_height)
            continue
        if key == curses.KEY_UP:
            top -= 1
        elif key == curses.KEY_DOWN:
            top += 1
        elif key == curses.KEY_PPAGE:
            top -= body_height
        elif key == curses.KEY_NPAGE:
            top += body_height
        elif key == curses.KEY_HOME:
            top = 0
        elif key == curses.KEY_END:
            top = max_top


def choose_current_run(runs: list[RunSummary]) -> RunSummary | None:
    for run in runs:
        if is_primary_row(run):
            return run
    return runs[0] if runs else None


def agent_rows_for_current_run(runner: CommandRunner, runs: list[RunSummary]) -> tuple[list[Row], str]:
    run = choose_current_run(runs)
    if not run:
        return [], "no current run found"
    agents = collect_agents(runner, run)
    rows = [row_for_agent(agent) for agent in agents]
    return rows, f"run={run.name} {format_run_detail(run)}"


def list_once(runner: CommandRunner, args: argparse.Namespace) -> int:
    windows = collect_windows(runner, args.session, args.capture_lines)
    runs = collect_run_summaries(runner, args.roots, args.max_runs)
    if args.runs:
        rows = make_rows(windows, runs, include_all_windows=args.all_windows, show_history=args.history)
        heading = "runs"
    else:
        rows, heading = agent_rows_for_current_run(runner, runs)
    print(f"# {heading}")
    for i, row in enumerate(rows):
        print(f"{i:02d} {row.title}")
        print(f"   {row.detail}")
    return 0


def tui(stdscr: Any, runner: CommandRunner, args: argparse.Namespace) -> None:
    curses.curs_set(0)
    stdscr.keypad(True)
    stdscr.timeout(1000)
    selected = 0
    top = 0
    rows: list[Row] = []
    status = ""

    def refresh() -> None:
        nonlocal rows, selected, top, status
        try:
            windows = collect_windows(runner, args.session, args.capture_lines)
            runs = collect_run_summaries(runner, args.roots, args.max_runs)
            if args.runs:
                rows = make_rows(windows, runs, include_all_windows=args.all_windows, show_history=args.history)
                scope = "run-history" if args.history else "current-run"
            else:
                rows, run_heading = agent_rows_for_current_run(runner, runs)
                scope = f"agents {run_heading}"
            selected = min(selected, max(0, len(rows) - 1))
            top = min(top, selected)
            where = f"ssh:{args.ssh_host}" if args.ssh_host else "local"
            status = f"{where} session={args.session} scope={scope} rows={len(rows)} refreshed={dt.datetime.now().strftime('%H:%M:%S')}"
        except Exception as exc:
            rows = [Row(kind="error", title="refresh error", detail=str(exc))]
            selected = 0
            top = 0
            status = "refresh failed"

    refresh()
    while True:
        height, width = stdscr.getmaxyx()
        body_height = max(1, height - 3)
        if selected < top:
            top = selected
        if selected >= top + body_height:
            top = selected - body_height + 1
        lines = [f"{row.title}  {row.detail}" for row in rows]
        title = "Batch Agent Read-Only Monitor" if not args.runs else "Batch Run Read-Only Monitor"
        header = [title, status]
        draw_text_lines(stdscr, lines, top, header, selected=selected)
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            return
        if key in (ord("r"), ord("R")):
            refresh()
            continue
        if key == curses.KEY_UP or key == ord("k"):
            selected = max(0, selected - 1)
        elif key == curses.KEY_DOWN or key == ord("j"):
            selected = min(max(0, len(rows) - 1), selected + 1)
        elif key == curses.KEY_PPAGE:
            selected = max(0, selected - body_height)
        elif key == curses.KEY_NPAGE:
            selected = min(max(0, len(rows) - 1), selected + body_height)
        elif key in (10, 13, curses.KEY_ENTER):
            if rows:
                result = view_row(stdscr, runner, args.session, rows[selected], args.scrollback, args.event_lines)
                if result == "quit":
                    return
                refresh()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only TUI for batch annotation tmux windows and run roots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              scripts/debug_batch_monitor.py
              scripts/debug_batch_monitor.py --ssh-host 115.190.235.210
              scripts/debug_batch_monitor.py --once --ssh-host 115.190.235.210
              scripts/debug_batch_monitor.py --runs --history --ssh-host 115.190.235.210

            Keys:
              default view: current run's 16 item agents
              list view: Up/Down choose, Enter open, r refresh, q quit
              pane view: Up/Down/PgUp/PgDn scroll, r refresh, b back, q quit
            """
        ),
    )
    parser.add_argument("--ssh-host", default=os.environ.get("ANNOTATION_REMOTE_HOST", ""), help="optional remote host to query over ssh")
    parser.add_argument("--session", default=os.environ.get("BATCH_MONITOR_TMUX_SESSION", DEFAULT_SESSION), help="tmux session name")
    parser.add_argument("--roots", nargs="*", default=list(DEFAULT_ROOTS), help="run-root parent directories to summarize")
    parser.add_argument("--max-runs", type=int, default=30, help="maximum recent run roots to list")
    parser.add_argument("--capture-lines", type=int, default=120, help="pane lines used for root detection")
    parser.add_argument("--scrollback", type=int, default=2000, help="pane lines captured in read-only view")
    parser.add_argument("--event-lines", type=int, default=120, help="event lines shown for run-root detail view")
    parser.add_argument("--runs", action="store_true", help="show run roots instead of the current run's item-agent list")
    parser.add_argument("--history", action="store_true", help="with --runs, include historical, failed, smoke, and superseded run roots")
    parser.add_argument("--all-windows", action=argparse.BooleanOptionalAction, default=False, help="with --runs, include tmux windows even when no current batch root is detected")
    parser.add_argument("--once", action="store_true", help="print one noninteractive snapshot and exit")
    args = parser.parse_args(argv)
    args.ssh_host = args.ssh_host.strip() or None
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    runner = CommandRunner(args.ssh_host)
    if args.once:
        return list_once(runner, args)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("debug_batch_monitor.py needs a TTY for the interactive view; use --once for a text snapshot.", file=sys.stderr)
        return 2
    curses.wrapper(tui, runner, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
