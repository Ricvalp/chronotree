import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from stmem.training.checkpoints import atomic_save, atomic_copy, capture_rng, restore_rng, mark_complete
from stmem.evaluation.metrics import summarize_episodes, paired_comparison

REPO = Path(__file__).resolve().parents[1]


def test_checkpoint_rng_and_previous_survive_failed_write(tmp_path):
    path = tmp_path / "last.pt"
    atomic_save({"step": 3, "rng": capture_rng()}, path)
    expected = (random.random(), np.random.random(), torch.rand(3))
    saved = torch.load(path, weights_only=False)
    restore_rng(saved["rng"])
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    atomic_save({"step": 4}, path)
    assert torch.load(path.with_name("last.pt.previous"), weights_only=False)["step"] == 3
    with pytest.raises(Exception):
        atomic_save({"unserializable": lambda: None}, path)
    assert torch.load(path, weights_only=False)["step"] == 4
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError):
        mark_complete(tmp_path / "complete.json", completed_steps=4, intended_steps=10)
    assert not (tmp_path / "complete.json").exists()
    immutable = tmp_path / "selected.pt"
    atomic_copy(path, immutable)
    atomic_copy(path, immutable)
    atomic_save({"step": 5}, path)
    with pytest.raises(FileExistsError):
        atomic_copy(path, immutable)
    assert torch.load(immutable, weights_only=False)["step"] == 4


def test_signal_wrapper_retains_child_status_and_resumes(tmp_path):
    marker, checkpoint, ready = (tmp_path / name for name in ("stop", "checkpoint", "ready"))
    child = tmp_path / "child.py"
    child.write_text("""import pathlib, sys, time
marker, checkpoint, ready = map(pathlib.Path, sys.argv[1:])
step = int(checkpoint.read_text()) if checkpoint.exists() else 0
ready.write_text('ready')
while step < 50:
    time.sleep(.01)
    step += 1
    if marker.exists():
        checkpoint.write_text(str(step))
        sys.exit(75)
checkpoint.write_text(str(step))
""")
    command = [
        "bash",
        str(REPO / "hpc/run_job.sh"),
        str(marker),
        sys.executable,
        str(child),
        str(marker),
        str(checkpoint),
        str(ready),
    ]
    first = subprocess.Popen(command)
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()
    os.kill(first.pid, signal.SIGUSR1)
    assert first.wait(timeout=5) == 75
    assert 0 < int(checkpoint.read_text()) < 50
    assert marker.exists()
    # The new allocation clears its stale local marker and actually finishes.
    assert subprocess.run(command, timeout=5).returncode == 0
    assert int(checkpoint.read_text()) == 50
    assert not marker.exists()


def test_plan_is_bounded_and_selection_precedes_cache(tmp_path):
    target = tmp_path / "generated"
    subprocess.run(
        [sys.executable, str(REPO / "hpc/plan.py"), "--dry-run", "--output-dir", str(target)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    plan = json.loads((target / "plan.json").read_text())
    assert plan["maximum_concurrent_allocations"] == 2
    assert plan["seeds"] == [0] and len(plan["tasks"]) == 16
    jobs = {job["name"]: job for job in plan["jobs"]}
    assert jobs["seed0_cache"]["deps"] == ["seed0_select_teacher"]
    assert len(jobs["seed0_select_teacher"]["deps"]) == 2
    assert len(jobs["seed0_select_compact"]["deps"]) == 4
    selection_command = jobs["seed0_select_compact"]["commands"][0]
    assert selection_command[selection_command.index("--evaluations") + 1].endswith("15000/tree")
    assert selection_command[selection_command.index("--evaluations") + 4].endswith("30000/flat")
    assert "seed0_select_compact" in jobs["seed0_labels_tree"]["deps"]
    assert jobs["seed0_router_tree_final"]["deps"] == ["seed0_labels_tree_refresh"]
    assert all("--split test" not in " ".join(job["commands"][0]) for job in jobs.values())
    for script in list((target / "jobs").glob("*.sbatch")) + list((REPO / "hpc/jobs").glob("*")):
        subprocess.run(["bash", "-n", str(script)], check=True)
    teacher_script = Path(jobs["seed0_teacher"]["script"]).read_text()
    assert "#SBATCH --nodes=1" in teacher_script
    assert "--nproc-per-node=1" in teacher_script
    assert "B:USR1@180" in teacher_script
    assert not (target / "submitted.json").exists()


def test_report_missing_entire_shard_keeps_manifest_denominator(tmp_path):
    from stmem.report import build_report

    output = tmp_path / "seed0/evaluation/val/TREE/shard0"
    (output / "episodes").mkdir(parents=True)
    (output / "manifest.json").write_text(
        json.dumps({"planned": [["BinFill", 0]], "all_planned": [["BinFill", 0], ["BinFill", 1]]})
    )
    (output / "episodes/BinFill_000.json").write_text(
        json.dumps(
            {
                "task": "BinFill",
                "episode_idx": 0,
                "status": "success",
                "decisions": [{"decision_seconds": 0.1, "complete_online_seconds": 0.3}],
            }
        )
    )
    report = build_report(tmp_path)
    tree = report["seeds"]["seed0"]["methods"]["TREE"]
    assert tree["overall"]["planned"] == 2
    assert tree["overall"]["missing"] == 1
    assert tree["macro_success"] == 0.5
    assert tree["latency"]["all"]["p50_seconds"] == 0.3
    assert report["seeds"]["seed0"]["methods"]["FLAT"]["macro_success"] is None


def test_metrics_keep_policy_errors_and_missing_in_denominator():
    rows = [
        {"task": "BinFill", "episode_idx": 0, "status": "success"},
        {"task": "BinFill", "episode_idx": 1, "status": "policy_error"},
    ]
    stats = summarize_episodes(rows, [("BinFill", i) for i in range(4)])
    assert stats["macro_success"] == 0.25
    assert stats["overall"]["missing"] == 2
    assert stats["overall"]["errors"] == 1
    assert stats["status"] == "partial"
    paired = paired_comparison(rows, rows, bootstrap_samples=30)
    assert paired["macro_success_difference"] == 0
    assert paired["ci95"] == [0, 0]
    with pytest.raises(ValueError, match="exactly the same"):
        paired_comparison(rows, rows[:1])
    not_run = summarize_episodes([], [("BinFill", 0)])
    assert not_run["status"] == "not yet run"
    assert not_run["macro_success"] is None


def test_submitter_dependencies_with_fake_sbatch_only(tmp_path):
    """Exercise submission arguments through a fake executable, never Slurm."""
    executable = tmp_path / "bin" / "sbatch"
    executable.parent.mkdir()
    calls = tmp_path / "calls.jsonl"
    executable.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, pathlib, sys
path = pathlib.Path(os.environ['FAKE_SBATCH_CALLS'])
index = len(path.read_text().splitlines()) if path.exists() else 0
with path.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
print(1000 + index)
"""
    )
    executable.chmod(0o755)
    site = tmp_path / "site.env"
    site.write_text(f"STMEM_REPO_ROOT={REPO}\nSTMEM_RUN_ROOT={tmp_path / 'runs'}\n")
    env = dict(
        os.environ,
        PATH=str(executable.parent) + os.pathsep + os.environ.get("PATH", ""),
        FAKE_SBATCH_CALLS=str(calls),
    )
    target = tmp_path / "plan"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "hpc/plan.py"),
            "--submit",
            "--site",
            str(site),
            "--output-dir",
            str(target),
        ],
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    plan = json.loads((target / "plan.json").read_text())
    submissions = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(submissions) == plan["job_count"]
    assert all(command[0] == "--parsable" for command in submissions)
    assert any("afterok:" in part for command in submissions for part in command)
    assert any("afterany:" in part for command in submissions for part in command)
    assert len(json.loads((target / "submitted.json").read_text())) == plan["job_count"]


def test_frozen_protocol_blocks_direct_cli_changes(tmp_path):
    from copy import deepcopy
    from stmem.config import load_config, source_revision
    from stmem.evaluation.protocol import inference_contract, validate_frozen
    from stmem.training.checkpoints import file_sha256

    cfg = load_config(REPO / "configs/compact.yaml")
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"immutable selected weights")
    configuration = tmp_path / "frozen.yaml"
    configuration.write_text("budget: 128\n")
    hashes = {str(checkpoint): file_sha256(checkpoint)}
    frozen = {
        "protocol": {
            "methods": ["UNIFORM"],
            "seeds": [cfg["seed"]],
            "tasks": cfg["data"]["tasks"],
            "budget": 128,
            "evaluation": {"final_test_episodes_per_task": 50},
        },
        "source_sha256": source_revision()["source_sha256"],
        "checkpoints": hashes,
        "configuration_hashes": {str(configuration): file_sha256(configuration)},
        "checkpoint_roles": {str(cfg["seed"]): {"compact": str(checkpoint)}},
        "inference_contracts": {"compact": inference_contract(cfg)},
    }
    validate_frozen(cfg, frozen, "UNIFORM", 128, 50, hashes)
    for budget, count in ((64, 50), (128, 2)):
        with pytest.raises(ValueError):
            validate_frozen(cfg, frozen, "UNIFORM", budget, count, hashes)
    changed = deepcopy(cfg)
    changed["model"]["flow_steps"] += 1
    with pytest.raises(ValueError, match="Inference configuration"):
        validate_frozen(changed, frozen, "UNIFORM", 128, 50, hashes)
    with pytest.raises(ValueError, match="interventions"):
        validate_frozen(cfg, frozen, "UNIFORM", 128, 50, hashes, intervention="erase_execution")
    configuration.write_text("budget: 64\n")
    with pytest.raises(ValueError, match="configuration changed"):
        validate_frozen(cfg, frozen, "UNIFORM", 128, 50, hashes)


def test_two_rank_cpu_training_with_empty_history(tmp_path):
    import yaml

    cfg = {
        "extends": str(REPO / "configs/debug/teacher.yaml"),
        "paths": {"run_root": str(tmp_path / "runs")},
        "data": {"episodes_per_task": 2, "synthetic_execution_frames": 4, "synthetic_conditioning_frames": 0},
        "model": {
            "points": 16,
            "visual_tokens": 4,
            "current_tokens": 4,
            "width": 32,
            "layers": 1,
            "text_layers": 1,
            "horizon": 4,
        },
        "training": {
            "steps": 2,
            "effective_batch": 2,
            "checkpoint_steps": [2],
            "checkpoint_every": 1,
            "warmup_steps": 1,
            "precision": "fp32",
        },
    }
    path = tmp_path / "ddp.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            "-m",
            "stmem.train",
            "--stage",
            "teacher",
            "--config",
            str(path),
        ],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    saved = torch.load(tmp_path / "runs/seed0/teacher/last.pt", weights_only=False)
    assert saved["step"] == 2
    assert len(saved["rng_by_rank"]) == 2
    assert (tmp_path / "runs/seed0/teacher/complete.json").exists()
