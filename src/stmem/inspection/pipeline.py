"""One-command local inspection of immutable HPC checkpoint bundles."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from stmem.inspection.common import REPO, policy_source_hash, prepare_manifest, read_json, resolve_assets
from stmem.training.checkpoints import atomic_json, file_sha256

STAGES = ("offline", "grounding", "handoff")


class InspectionInterrupted(KeyboardInterrupt):
    """Translate termination signals into orderly subprocess cleanup."""

    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"Inspection interrupted by {signal.Signals(signum).name}")


@contextmanager
def termination_signals():
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous = {signum: signal.getsignal(signum) for signum in signals}

    def interrupt(signum, _frame):
        # A second signal must not interrupt the bounded child cleanup and leave
        # an independent process group writing after the output lock is released.
        for item in signals:
            signal.signal(item, signal.SIG_IGN)
        raise InspectionInterrupted(signum)

    try:
        for signum in signals:
            signal.signal(signum, interrupt)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def pipeline_status(output, status, **details):
    path = output / "pipeline_status.json"
    result = read_json(path) if path.exists() else {"stages": {}}
    result.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat(), **details)
    if status not in {"failed", "interrupted"}:
        result.pop("error", None)
    atomic_json(result, path)


def inspect_runtime():
    """Observe installed versions without initializing CUDA for CPU audits."""
    import torch

    packages = {}
    for name in ("torch", "robomme", "mani-skill", "sapien", "numpy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    device = None
    if torch.cuda.is_initialized():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        device = {
            "index": index, "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
        }
    return {
        "python": sys.version, "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable, "platform": platform.platform(),
        "packages": packages, "torch_build": torch.__version__, "torch_wheel_cuda": torch.version.cuda,
        "initialized_cuda_device": device,
    }


def verify_runtime(manifest_path):
    """Keep the first observed environment separate from the immutable protocol."""
    path = manifest_path.parent / "runtime_identity.json"
    identity = inspect_runtime()
    manifest_hash = file_sha256(manifest_path)
    if path.exists():
        previous = read_json(path)
        if previous.get("inspection_manifest_sha256") != manifest_hash:
            raise ValueError("Runtime identity belongs to a different inspection manifest")
        old = previous["identity"]
        # CUDA may already be initialized by an embedding caller. Compare the
        # hardware only when both invocations observed it; never initialize it
        # just to make an audit or a report.
        names = set(old) | set(identity)
        names.discard("initialized_cuda_device")
        changed = [name for name in sorted(names) if old.get(name) != identity.get(name)]
        if (old.get("initialized_cuda_device") is not None
                and identity["initialized_cuda_device"] is not None
                and old["initialized_cuda_device"] != identity["initialized_cuda_device"]):
            changed.append("initialized_cuda_device")
        if changed:
            raise ValueError(f"Inspection runtime environment changed ({', '.join(changed)}); use a new output directory")
        return previous
    record = {
        "schema_version": 1, "inspection_manifest_sha256": manifest_hash,
        "first_observed_utc": datetime.now(timezone.utc).isoformat(), "identity": identity,
        "scope": "Environment observed at this launcher invocation. Existing manually generated stage outputs may predate this record; it is not retrospective version proof.",
    }
    atomic_json(record, path)
    return record


def verify_final_inputs(manifest, assets_config):
    """Reject data/source changes while subprocesses were producing results."""
    if policy_source_hash() != manifest["policy_source_sha256"]:
        raise ValueError("Policy source changed during inspection; use a new output directory")
    assets, hashes = resolve_assets(assets_config)
    if assets != manifest["assets"] or hashes != manifest["asset_hashes"]:
        raise ValueError("Inspection assets changed during inspection; use a new output directory")


def terminate_stage(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    finally:
        # The session leader may exit promptly while a loader/renderer ignores
        # SIGTERM. Ensure no surviving descendant can write after lock release.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@contextmanager
def inspection_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".inspection.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another inspection is running in {output}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def stage_code_hash(stage):
    prefixes = (stage, "reset") if stage == "grounding" else (stage,)
    paths = {path for prefix in prefixes for path in Path(__file__).parent.glob(prefix + "*.py")}
    if stage == "grounding":
        paths.add(REPO / "scripts/visualize_reset_cube_chunks.py")
    result = hashlib.sha256()
    for path in sorted(paths):
        result.update(str(path.relative_to(REPO)).encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def run_stage(stage, manifest_path, *, resume):
    """Keep full subprocess output on disk and emit lightweight progress."""
    output = manifest_path.parent
    status_path = output / "pipeline_status.json"
    status = read_json(status_path) if status_path.exists() else {"stages": {}}
    code_hash = stage_code_hash(stage)
    previous = status["stages"].get(stage)
    if resume and previous and previous.get("code_sha256") != code_hash:
        raise ValueError(f"{stage} diagnostic code changed; choose a new output directory")
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    logfile = logs / f"{stage}.log"
    command = [sys.executable, "-u", "-m", f"stmem.inspection.{stage}", "--manifest", str(manifest_path)]
    if resume and stage != "handoff":
        command.append("--resume")
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    # Diagnostic modules receive explicit local assets/configuration. Do not
    # allow an old HPC environment or legacy worktree to redirect those paths.
    for key in tuple(env):
        if key.startswith("STMEM_") or key in {"RANK", "LOCAL_RANK", "WORLD_SIZE"}:
            env.pop(key)
    env["PYTHONPATH"] = str(REPO / "src")
    env["WANDB_MODE"] = "disabled"
    record = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": code_hash, "command": command, "log": str(logfile),
    }
    status["stages"][stage] = record
    status["status"] = "running"
    atomic_json(status, status_path)
    started = time.monotonic()
    print(json.dumps({"event": "stage_started", "stage": stage, "log": str(logfile)}), flush=True)
    process = None
    try:
        with logfile.open("a" if resume else "w") as stream:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            last_notice = started
            while process.poll() is None:
                now = time.monotonic()
                if now - last_notice >= 30:
                    print(json.dumps({"event": "stage_running", "stage": stage,
                                      "elapsed_seconds": round(now-started), "log": str(logfile)}), flush=True)
                    last_notice = now
                time.sleep(1)
    except BaseException as error:
        terminate_stage(process)
        outcome = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        record.update(status=outcome, elapsed_seconds=time.monotonic()-started,
                      error=f"{type(error).__name__}: {error}")
        status["status"] = outcome
        atomic_json(status, status_path)
        raise
    record.update(status="complete" if process.returncode == 0 else "failed",
                  exit_code=process.returncode, elapsed_seconds=time.monotonic()-started)
    if stage_code_hash(stage) != code_hash:
        record["status"] = "code_changed_during_execution"
    status["status"] = record["status"] if record["status"] != "complete" else "running"
    atomic_json(status, status_path)
    if record["status"] != "complete":
        raise RuntimeError(f"Inspection stage {stage} failed; see {logfile}; fix the cause then use --resume")
    print(json.dumps({"event": "stage_complete", "stage": stage,
                      "elapsed_seconds": record["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path, help="Transferred directory with SHA256SUMS")
    parser.add_argument("--output", type=Path, help="Defaults to <bundle>_analysis")
    parser.add_argument("--assets-config", type=Path, default=REPO / "configs/inspection.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true", help="Reuse only matching, validated diagnostic outputs")
    parser.add_argument("--audit-only", action="store_true", help="Verify checkpoints/assets without GPU inference")
    parser.add_argument("--report-only", action="store_true", help="Rebuild the report from completed results")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--compare-to", type=Path, help="Optional earlier inspection output for matched comparisons")
    args = parser.parse_args()
    if args.audit_only and args.report_only:
        parser.error("Choose only one of --audit-only and --report-only")
    output = (args.output or Path(str(args.bundle.resolve()) + "_analysis")).resolve()
    import torch
    torch.set_num_threads(2)
    with inspection_lock(output), termination_signals():
        try:
            manifest, path = prepare_manifest(args.bundle, output, args.assets_config,
                                              device=args.device, resume=args.resume)
            verify_runtime(path)
            if args.audit_only:
                pipeline_status(output, "audit_complete")
                print(f"Integrity checks passed: {path}")
                return
            if not args.report_only:
                if not args.device.startswith("cuda") or not torch.cuda.is_available():
                    raise RuntimeError("The full simulator diagnostic requires a working CUDA device")
                for stage in args.stages:
                    run_stage(stage, path, resume=args.resume)
            from stmem.inspection.report import build_report
            verify_final_inputs(manifest, args.assets_config)
            verify_runtime(path)
            report = build_report(manifest, compare_to=args.compare_to,
                                  allow_partial=set(args.stages) != set(STAGES) and not args.report_only)
            result = read_json(output / "summary.json")
            if result.get("status") not in {"complete", "partial"}:
                raise ValueError("Report did not establish a complete or partial inspection status")
            pipeline_status(output, result["status"], report=str(report))
            print(f"Inspection report: {report}")
        except BaseException as error:
            # Before a manifest exists, leave the directory available for a clean
            # retry of input verification rather than creating a misleading run.
            if (output / "inspection_manifest.json").exists():
                status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
                pipeline_status(output, status, error=f"{type(error).__name__}: {error}")
            if isinstance(error, InspectionInterrupted):
                raise SystemExit(128 + error.signum) from None
            raise


if __name__ == "__main__":
    main()
