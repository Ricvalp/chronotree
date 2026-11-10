"""Repeatable, bounded local teacher inspection; no training or HPC submission.

The usual inspection retains its original short execution horizon. The history
study uses execute-ten and fixed validation scenes, exactly as the reference.
Checkpoint-specific findings belong in generated reports, not in this launcher.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import quote

import yaml

from stmem.config import source_revision
from stmem.inspection.common import REPO, discover_checkpoints, read_json
from stmem.inspection.pipeline import InspectionInterrupted, inspection_lock, termination_signals, terminate_stage
from stmem.training.checkpoints import atomic_json, file_sha256


REFERENCE = REPO / "configs/inspection_teacher_reference.yaml"
ASSETS = REPO / "configs/inspection.yaml"
PATH_NAMES = ("usual", "history", "recent")


def _resolve(value):
    return (REPO / Path(value).expanduser()).resolve()


def load_reference(path):
    """Accept a saved suite or an explicit recipe for earlier standalone runs."""
    path = Path(path).resolve()
    if path.is_dir():
        suite = read_json(path / "teacher_inspection.json")
        value = {**suite["paths"], "model": suite["model"],
                 "checkpoint_step": read_json(path / "usual/checkpoint_audit.json")["checkpoints"][suite["model"]]["step"],
                 "checkpoint_sha256": suite["checkpoint"]["sha256"]}
    else:
        value = yaml.safe_load(path.read_text())
    required = {*PATH_NAMES, "model", "checkpoint_step", "checkpoint_sha256"}
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError(f"Reference must define {sorted(required)}")
    result = {key: str(_resolve(value[key])) for key in PATH_NAMES}
    result.update({key: value[key] for key in required - set(PATH_NAMES)})
    usual = read_json(Path(result["usual"]) / "inspection_manifest.json")
    expected = (result["checkpoint_sha256"], result["checkpoint_step"])
    entry = usual["checkpoints"][result["model"]]
    if (entry["sha256"], entry["step"]) != expected:
        raise ValueError("Reference usual checkpoint disagrees with the declared anchor")
    if read_json(Path(result["usual"]) / "summary.json")["status"] != "complete":
        raise ValueError("Reference usual inspection is incomplete")
    from stmem.inspection.history_use_report import load_report
    from stmem.inspection.history_use import TASKS, ARMS

    for name in ("history", "recent"):
        report = load_report(result[name])
        checkpoint = report["manifest"]["checkpoint"]
        if (checkpoint["sha256"], checkpoint["step"]) != expected:
            raise ValueError(f"Reference {name} checkpoint disagrees with the declared anchor")
        if not report["summary"]["complete"]:
            raise ValueError(f"Reference {name} inspection is incomplete")
        manifest = report["manifest"]
        tasks = TASKS if name == "history" else ("SwingXtimes",)
        if manifest["planned_scenes"] != [[task, i] for task in tasks for i in range(4)]:
            raise ValueError("Reference differs from the fixed four-scene history protocol")
        if manifest["execution_override"]["execute_actions"] != 10:
            raise ValueError("Reference history protocol must execute ten predicted targets")
        if name == "history" and manifest["arms"] != ARMS:
            raise ValueError("Reference must contain the four fixed FULL/erasure/CURRENT arms")
        if name == "recent" and manifest["arms"] != {
            "full": ARMS["full"], "recent_execution": {"method": "FULL", "intervention": "none", "completed_blocks": 2},
        }:
            raise ValueError("Reference recent-window protocol must retain two completed blocks")
    return result


def reference_identities(reference):
    """Reference records cannot change while a new paired evaluation runs."""
    files = []
    for name in PATH_NAMES:
        directory = Path(reference[name])
        files.extend(path for path in directory.glob("*.json") if path.name in {
            "inspection_manifest.json", "checkpoint_audit.json", "summary.json", "manifest.json", "config.json",
        })
        if name != "usual":
            files.extend(directory.glob("results/*/episodes/*.json"))
    return {str(path): file_sha256(path) for path in sorted(set(files))}


def create_plan(bundle, output, reference_path, assets_config):
    entries = discover_checkpoints(bundle)
    reference = load_reference(reference_path)
    model = reference["model"]
    if set(entries) != {model}:
        raise ValueError(f"This teacher suite needs a single-checkpoint bundle labeled {model!r}")
    assets_config = Path(assets_config).resolve()
    assets = yaml.safe_load(assets_config.read_text())
    clip = _resolve(assets["clip_checkpoint"])
    if not clip.is_file():
        raise FileNotFoundError(clip)
    return {
        "schema_version": 1, "scope": "Fixed-scene teacher diagnostics, not all-task benchmark success",
        "bundle": str(Path(bundle).resolve()), "output": str(output), "model": model,
        "checkpoint": entries[model], "reference": reference,
        "reference_identities": reference_identities(reference),
        "assets_config": str(assets_config), "assets_config_sha256": file_sha256(assets_config),
        "clip_checkpoint": str(clip), "clip_sha256": file_sha256(clip),
        "source_sha256": source_revision()["source_sha256"],
        "paths": {name: str(output / name) for name in PATH_NAMES},
        "protocol": {"episodes_per_task": 4, "execute_actions": 10,
                     "recent_completed_blocks": 2, "history_rollouts": 48, "fresh_recent_rollouts": 4,
                     "usual": "Original usual-check protocol, including four native commands per replan"},
    }


def verify_plan(plan):
    files = {**plan["reference_identities"],
             plan["checkpoint"]["path"]: plan["checkpoint"]["sha256"],
             plan["assets_config"]: plan["assets_config_sha256"],
             plan["clip_checkpoint"]: plan["clip_sha256"]}
    for path, expected in files.items():
        if file_sha256(path) != expected:
            raise ValueError(f"Inspection input changed: {path}")
    if source_revision()["source_sha256"] != plan["source_sha256"]:
        raise ValueError("Inspection source changed; use a fresh output directory")


def verify_output_plan(plan, *, resume=False, report_only=False):
    """Reject an invalid detached request before claiming it was launched."""
    output = Path(plan["output"])
    manifest = output / "teacher_inspection.json"
    if manifest.exists():
        if not (resume or report_only):
            raise FileExistsError("Teacher inspection exists; repeat with --resume")
        if read_json(manifest) != plan:
            raise ValueError("Teacher inspection inputs/source/reference changed; use a new output directory")
    elif report_only:
        raise FileNotFoundError("--report-only requires an existing teacher_inspection.json")
    elif output.exists() and any(p.name not in {".inspection.lock", "logs"} for p in output.iterdir()):
        raise FileExistsError("Nonempty output lacks a teacher-inspection manifest")


def stage_commands(plan, workers, *, resume=False, audit_only=False, report_only=False):
    paths, reference = plan["paths"], plan["reference"]
    usual = [sys.executable, "-u", "-m", "stmem.inspection.pipeline", "--bundle", plan["bundle"],
             "--output", paths["usual"], "--assets-config", plan["assets_config"],
             "--compare-to", reference["usual"]]
    if resume and (Path(paths["usual"]) / "inspection_manifest.json").is_file():
        usual.append("--resume")
    if audit_only:
        usual.append("--audit-only")
    elif report_only or (resume and (Path(paths["usual"]) / "summary.json").is_file()
                         and read_json(Path(paths["usual"]) / "summary.json").get("status") == "complete"):
        usual.append("--report-only")
    commands = {"usual": usual}
    if audit_only or report_only:
        return commands
    for stage in ("history", "recent"):
        command = [sys.executable, "-u", "-m", "stmem.inspection.history_use", "--checkpoint",
                   plan["checkpoint"]["path"], "--output", paths[stage], "--episodes", "4",
                   "--workers", str(workers), "--execute-actions", "10",
                   "--clip-checkpoint", plan["clip_checkpoint"]]
        if stage == "recent":
            command += ["--recent-swing", "--reference-output", paths["history"]]
        if resume and (Path(paths[stage]) / "manifest.json").is_file():
            command.append("--resume")
        commands[stage] = command
    return commands


def _status(output, status, **fields):
    path = output / "status.json"
    record = read_json(path) if path.exists() else {}
    record.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
    if status not in {"failed", "interrupted"}:
        record.pop("error", None)
    atomic_json(record, path)


def run_command(stage, command, output, deadline):
    """One bounded child stage; verbose logs stay on disk."""
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    log = logs / f"{stage}.log"
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith("STMEM_") or name in {"RANK", "LOCAL_RANK", "WORLD_SIZE"}:
            env.pop(name)
    env.update(PYTHONPATH=str(REPO / "src"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    _status(output, "running", stage=stage, log=str(log))
    print(json.dumps({"event": "teacher_inspection_stage", "stage": stage, "log": str(log)}), flush=True)
    process = None
    try:
        if time.monotonic() >= deadline:
            raise TimeoutError("Teacher inspection reached --max-hours; resume unfinished work later")
        with log.open("a") as stream:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            notice = time.monotonic()
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Teacher inspection reached --max-hours; resume unfinished work later")
                if time.monotonic() - notice >= 30:
                    _status(output, "running", stage=stage, log=str(log), child_pid=process.pid)
                    print(json.dumps({"event": "stage_running", "stage": stage, "log": str(log)}), flush=True)
                    notice = time.monotonic()
                time.sleep(1)
        if process.returncode:
            raise RuntimeError(f"{stage} exited {process.returncode}; inspect {log}")
    except BaseException:
        terminate_stage(process)
        raise


def usual_key_metrics(usual):
    """Small production-precision table; do not mix in the FP32 command probe."""
    metrics = []
    for row in usual["offline_summary_pairs"]:
        if row["kind"] != "visual" or row["state_subset"] != "all_states":
            continue
        if row["variant"] not in {"baseline", "erase_conditioning", "erase_execution"}:
            continue
        for window in (4, 16) if row["variant"] == "baseline" else (16,):
            value = row["metrics"][f"native{window}_tcp_error_mm"]
            metrics.append({"metric": f"Expert-state TCP error, {window} native commands, {row['variant']} (mm)",
                            "reference": value["reference"], "current": value["current"]})
    for key, label in (("first_close_observed_xy_mm", "Fresh PickXtimes first-close XY miss (mm)"),
                       ("max_cube_lift_mm", "Fresh PickXtimes maximum cube rise (mm)")):
        metrics.append({"metric": label, **{side: usual["grounding"][side][key] for side in ("reference", "current")}})
    handoffs = usual["handoff"]["paired_runs"]
    for task in sorted({row["key"][0] for row in handoffs}):
        rows = [row for row in handoffs if row["key"][0] == task]
        metrics.append({"metric": f"{task} expert-prefix successes within diagnostic cap",
                        **{side: f"{sum(row[side]['status'] == 'success' for row in rows)}/{len(rows)}"
                           for side in ("reference", "current")}})
    return metrics


def build_reports(plan, *, validate_media=True):
    """CPU-only aggregation, also usable with earlier standalone result paths."""
    from stmem.inspection.history_comparison import compare_history_runs
    from stmem.inspection.teacher_comparison import compare_behavior, compare_usual

    output = Path(plan["output"])
    comparison = output / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    paths, reference = plan["paths"], plan["reference"]
    history = compare_history_runs(reference["history"], paths["history"], output=comparison,
                                   reference_recent=reference["recent"], recent=paths["recent"],
                                   require_complete=True, validate_media=validate_media)
    if not history["complete"] or not history["valid"]:
        raise ValueError("Paired-history audit failed; see comparison/history_comparison.json")
    usual = compare_usual(reference["usual"], paths["usual"], model=plan["model"],
                          output=comparison / "usual_comparison.json")
    behavior = compare_behavior(reference["history"], paths["history"],
                                output=comparison / "behavior_comparison.json", require_complete=True)
    summary = {"status": "complete", "scope": plan["scope"], "history": history,
               "usual": usual, "behavior": behavior,
               "usual_key_metrics": usual_key_metrics(usual),
               "limitations": ["History erasure is an OOD control, not proof of semantic recall.",
                               "Later closed-loop states differ; same-state comparisons use reset or fixed expert states.",
                               "Video interpretation still requires review; this report does not invent visual conclusions."]}
    compatibility = usual.get("compatibility", {})
    if history.get("requires_source_review") or compatibility.get("source_review_required"):
        summary["limitations"].append("Source identities differ between reference and current results; code compatibility requires review.")
    if compatibility.get("runtime_review_required"):
        summary["limitations"].append("Runtime identities differ; review the environment comparison before interpreting differences.")
    atomic_json(summary, output / "summary.json")
    old = {(r["task"], r["arm"]): r for r in history["old_counts"]}
    rows = []
    for row in history["new_counts"]:
        previous = old[(row["task"], row["arm"])]
        rows.append(f"| {row['task']} | {row['arm']} | {previous['success']}/{previous['planned']} | {row['success']}/{row['planned']} |")
    changed = [row for row in history["comparison"]["cases"]
               if row.get("old_status") != row.get("new_status")]
    atomic_json({"scope": "Prioritize changed outcomes for manual video review; not automatic causal interpretation",
                 "cases": changed, "behavior_candidates": behavior["suggested_video_review"],
                 "handoff_candidates": [row for row in usual["handoff"]["paired_runs"]
                                        if row["reference"]["status"] != row["current"]["status"]]},
                output / "review_queue.json")
    links = [("Paired history comparison and videos", comparison / "history_comparison.html"),
             ("Usual checks and point clouds", Path(paths["usual"]) / "index.html"),
             ("History gallery", Path(paths["history"]) / "index.html"),
             ("Recent-history gallery", Path(paths["recent"]) / "index.html"),
             ("Behavior measurements", comparison / "behavior_comparison.json"),
             ("Changed outcomes to review", output / "review_queue.json")]
    def relative(path):
        return quote(Path(os.path.relpath(path, output)).as_posix())
    title = f"Teacher inspection: {history['new_checkpoint']['step']:,} versus {history['old_checkpoint']['step']:,} updates"
    def format_value(value):
        return f"{value:.3f}" if isinstance(value, (float, int)) else str(value)
    usual_rows = [f"| {row['metric']} | {format_value(row['reference'])} | {format_value(row['current'])} |"
                  for row in summary["usual_key_metrics"]]
    md = [f"# {title}", "", plan["scope"], "",
          "| Task | History read | Reference successes | Current successes |", "|---|---|---:|---:|", *rows, "",
          "Usual checks retain their original shorter execution interval; offline fit uses fixed training states.", "",
          "| Measurement | Reference | Current |", "|---|---:|---:|", *usual_rows, "",
          *[f"- [{label}]({relative(path)})" for label, path in links], "", *summary["limitations"]]
    (output / "summary.md").write_text("\n".join(md) + "\n")
    table = "".join("<tr>" + "".join(f"<td>{escape(cell.strip())}</td>" for cell in row.strip("|").split("|")) + "</tr>" for row in rows)
    usual_table = "".join("<tr>" + "".join(f"<td>{escape(cell.strip())}</td>" for cell in row.strip("|").split("|")) + "</tr>" for row in usual_rows)
    html = ("<!doctype html><meta charset='utf-8'><title>Teacher inspection</title>"
            "<style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}"
            "td,th{padding:8px 15px;text-align:left;border-bottom:1px solid #ccc}a{color:#1763a4}</style>"
            f"<h1>{escape(title)}</h1><p>{escape(plan['scope'])}</p>"
            "<table><tr><th>Task</th><th>History read</th><th>Reference</th><th>Current</th></tr>" + table + "</table>"
            "<p>Usual checks retain their shorter execution interval. Offline fit uses fixed training states.</p>"
            "<table><tr><th>Measurement</th><th>Reference</th><th>Current</th></tr>" + usual_table + "</table><ul>"
            + "".join(f'<li><a href="{relative(path)}">{escape(label)}</a></li>' for label, path in links) + "</ul>"
            + "".join(f"<p>{escape(note)}</p>" for note in summary["limitations"]))
    (output / "index.html").write_text(html)
    return summary


def detach(argv, output):
    """Leave the bounded coordinator independent of the calling terminal/tool."""
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    log = logs / f"launcher_{stamp}.log"
    command = [sys.executable, "-u", "-m", "stmem.inspection.teacher_suite", *[arg for arg in argv if arg != "--detach"]]
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    record = {"pid": process.pid, "log": str(log), "status_path": str(output / "status.json"),
              "command": command, "note": "Coordinator launched; inspect status.json for progress or failure"}
    atomic_json(record, logs / f"launch_{stamp}.json")
    print(json.dumps(record, indent=2))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to <bundle>_teacher_inspection")
    parser.add_argument("--reference", type=Path, default=REFERENCE, help="Reference YAML or completed teacher-suite directory")
    parser.add_argument("--assets-config", type=Path, default=ASSETS)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--max-hours", type=float, default=4, help="Bounded time per invocation; --resume continues missing work")
    parser.add_argument("--resume", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Validate transfer/reference and print commands; no new results")
    modes.add_argument("--audit-only", action="store_true", help="CPU checkpoint/assets checks only")
    modes.add_argument("--report-only", action="store_true", help="Validate and summarize existing completed results; no simulations")
    parser.add_argument("--detach", action="store_true", help="Run with durable logs independently of this terminal")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        parser.error("--max-hours must be finite and positive")
    if args.detach and (args.dry_run or args.report_only or args.audit_only):
        parser.error("--detach is for simulation runs; omit it for read-only/CPU modes")
    output = (args.output or Path(str(args.bundle.resolve()) + "_teacher_inspection")).resolve()
    plan = create_plan(args.bundle, output, args.reference, args.assets_config)
    if args.dry_run:
        print(json.dumps({"plan": plan, "commands": stage_commands(plan, args.workers, resume=args.resume)}, indent=2))
        return
    verify_output_plan(plan, resume=args.resume, report_only=args.report_only)
    if args.detach:
        # Resolve user-supplied relative paths before changing the child cwd.
        child = ["--bundle", plan["bundle"], "--output", str(output), "--reference", str(args.reference.resolve()),
                 "--assets-config", plan["assets_config"], "--workers", str(args.workers), "--max-hours", str(args.max_hours)]
        if args.resume:
            child.append("--resume")
        detach(child, output)
        return
    deadline = time.monotonic() + args.max_hours * 3600
    with inspection_lock(output), termination_signals():
        manifest = output / "teacher_inspection.json"
        verify_output_plan(plan, resume=args.resume, report_only=args.report_only)
        if not manifest.exists():
            atomic_json(plan, manifest)
        try:
            _status(output, "running", pid=os.getpid(), stage="preflight")
            commands = stage_commands(plan, args.workers, resume=args.resume or args.report_only,
                                      audit_only=args.audit_only, report_only=args.report_only)
            for name, command in commands.items():
                verify_plan(plan)
                run_command(name, command, output, deadline)
            verify_plan(plan)
            if args.audit_only:
                _status(output, "audit_complete", stage="usual")
                return
            _status(output, "running", stage="comparison")
            build_reports(plan)
            verify_plan(plan)
            _status(output, "complete", stage="complete", report=str(output / "index.html"))
            print(json.dumps({"status": "complete", "report": str(output / "index.html")}), flush=True)
        except BaseException as error:
            _status(output, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    error=f"{type(error).__name__}: {error}")
            if isinstance(error, InspectionInterrupted):
                raise SystemExit(128 + error.signum) from None
            raise


if __name__ == "__main__":
    main()
