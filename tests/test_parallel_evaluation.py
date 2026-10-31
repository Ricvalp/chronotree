"""CPU-only synthetic subprocess checks of parallel evaluation orchestration."""

from copy import deepcopy
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import yaml

from stmem.evaluation.parallel import (
    EvaluationInterrupted, Worker, completed_shard, cpu_assignments, output_lock,
    registered_spawn, save_manifest, supervise, termination_signals, worker_plan,
)


def manifest():
    return {"config": {"tracking": {"mode": "offline"}, "model": {"execute": 4}},
            "source": {"source_sha256": "fixed-scientific-source", "dirty": False},
            "environment": {"gpu": "old-node"}, "num_shards": 2,
            "all_planned": [["PickXtimes", 0], ["PickXtimes", 1]]}


def worker(tmp_path, index, code):
    return Worker(index, [sys.executable, "-u", "-c", code], dict(os.environ),
                  tmp_path / "shards" / f"shard{index:03d}", [min(os.sched_getaffinity(0))],
                  {"uuid": "synthetic-cpu-subprocess"})


def complete_worker(tmp_path, index):
    directory = tmp_path / "shards" / f"shard{index:03d}"
    code = f"""
import hashlib,json
from pathlib import Path
p=Path({str(directory)!r});p.mkdir(parents=True,exist_ok=True)
m=p/'manifest.json';m.write_text(json.dumps({{'planned':[['PickXtimes',{index}]]}}))
(p/'complete.json').write_text(json.dumps({{'status':'complete','planned':1,'recorded':1,
 'manifest_sha256':hashlib.sha256(m.read_bytes()).hexdigest()}}))
"""
    return worker(tmp_path, index, code)


@pytest.mark.parametrize("workers,threads", [(16, 2), (32, 2)])
def test_cpu_affinity_respects_sparse_slurm_allocation(workers, threads):
    available = [2 * i + 3 for i in range(64)]
    assignments = cpu_assignments(available, workers, threads)
    assert len(assignments) == workers
    assert sorted(cpu for group in assignments for cpu in group) == available
    assert all(len(group) >= threads for group in assignments)
    with pytest.raises(ValueError, match="allocated CPU"):
        cpu_assignments(available[:12], workers, threads)


def test_bootstrap_sets_affinity_before_importing_evaluator(tmp_path):
    from stmem.evaluation import parallel_worker
    package = tmp_path / "stmem"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "evaluate.py").write_text(
        "import json,os,sys;print(json.dumps({'cpus':sorted(os.sched_getaffinity(0)),'args':sys.argv[1:]}))"
    )
    allowed = sorted(os.sched_getaffinity(0))
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    command = [sys.executable, parallel_worker.__file__, "--cpu-affinity", str(allowed[0]), "--", "--fixture"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"cpus": [allowed[0]], "args": ["--fixture"]}
    command[3] = str(max(allowed) + 100)
    rejected = subprocess.run(command, env=env, capture_output=True, text=True)
    assert rejected.returncode != 0 and "outside the inherited allocation" in rejected.stderr


def test_worker_environment_selects_allocated_gpu_and_cleans_training_state(tmp_path):
    args = SimpleNamespace(threads_per_worker=2, checkpoint=tmp_path / "weights.pt", method="FULL",
                           split="val", episode_count=4, video_cap=10, execute_actions=8,
                           budget=None, router_checkpoint=None, targeting_diagnostics=False)
    devices = [{"uuid": "GPU-" + ("a" if i == 0 else "b") * 8 + "-1234-1234-1234-123456789012",
                "pci_bus_id": f"0000:0{i}:00.0"}
               for i in range(2)]
    env = {"STMEM_RUN_ROOT": "wrong-training-root", "RANK": "3", "WORLD_SIZE": "4",
           "CUDA_VISIBLE_DEVICES": "7,3", "WANDB_MODE": "online", "WANDB_PROJECT": "chronotree"}
    plan = worker_plan(args, tmp_path, {"num_shards": 4}, devices, environment=env,
                       available_cpus=list(range(16)))
    assert [w.device["uuid"] for w in plan] == [d["uuid"] for d in devices] * 2
    for w in plan:
        assert w.env["CUDA_VISIBLE_DEVICES"] == w.device["uuid"]
        assert w.env["STMEM_EXPECTED_GPU_UUID"] == w.device["uuid"]
        assert w.env["WANDB_MODE"] == "disabled" and "--no-tracking" in w.command
        assert not {"STMEM_RUN_ROOT", "RANK", "WORLD_SIZE"} & w.env.keys()
        assert w.command[w.command.index("--shard-index") + 1] == str(w.index)
        assert w.command[w.command.index("--num-shards") + 1] == "4"
        assert w.command[w.command.index("--execute-actions") + 1] == "8"
        assert w.command[w.command.index("--video-cap") + 1] == "10"
        assert w.command[w.command.index("--render-device") + 1] == "cuda:0"


def test_manifest_resume_preserves_identity_but_allows_new_allocation(tmp_path):
    original = manifest()
    with output_lock(tmp_path):
        save_manifest(tmp_path, original, resume=False)
    before = (tmp_path / "manifest.json").read_bytes()
    replacement = deepcopy(original)
    replacement["environment"]["gpu"] = "new-node"
    replacement["config"]["tracking"]["mode"] = "online"
    replacement["source"]["dirty"] = True
    assert save_manifest(tmp_path, replacement, resume=True) == original
    assert (tmp_path / "manifest.json").read_bytes() == before
    with pytest.raises(FileExistsError, match="--resume"):
        save_manifest(tmp_path, original, resume=False)
    replacement["config"]["model"]["execute"] = 8
    with pytest.raises(ValueError, match="inputs/layout changed"):
        save_manifest(tmp_path, replacement, resume=True)


def test_resume_restores_only_missing_derived_config(tmp_path):
    original = manifest()
    save_manifest(tmp_path, original, resume=False)
    config = tmp_path / "evaluation.yaml"
    expected = yaml.safe_load(config.read_text())
    config.unlink()
    save_manifest(tmp_path, original, resume=True)
    assert yaml.safe_load(config.read_text()) == expected
    config.write_text("modified: true\n")
    save_manifest(tmp_path, original, resume=True)
    assert config.read_text() == "modified: true\n"  # main must reject, never silently overwrite


def test_lock_prevents_two_parent_writers(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another parallel evaluation"):
            with output_lock(tmp_path):
                pytest.fail("Second parent acquired the lock")
    with output_lock(tmp_path):
        pass


def test_supervisor_completes_and_reuses_only_validated_shards(tmp_path):
    plan = [complete_worker(tmp_path, i) for i in range(2)]
    progress = []
    state = supervise(plan, tmp_path, lambda: progress.append(time.monotonic()), poll_seconds=0.01)
    assert state["status"] == "complete" and len(progress) >= 2
    assert all(completed_shard(w.directory) for w in plan)
    for w in plan:
        w.command = ["never-launch-this-completed-worker"]
    reused = supervise(plan, tmp_path, lambda: None, resume=True, poll_seconds=0.01)
    assert {w["status"] for w in reused["workers"].values()} == {"reused"}
    (plan[0].directory / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Invalid completed shard identity"):
        supervise(plan, tmp_path, lambda: None, resume=True, poll_seconds=0.01)


def test_worker_exit_zero_without_completion_is_not_success(tmp_path):
    with pytest.raises(RuntimeError, match="without valid completion"):
        supervise([worker(tmp_path, 0, "pass")], tmp_path, lambda: None, poll_seconds=0.01)
    assert json.loads((tmp_path / "parallel_status.json").read_text())["status"] == "failed"


def test_failed_worker_stops_sibling_process_and_preserves_failure(tmp_path):
    pid_file = tmp_path / "sibling.pid"
    sibling = worker(tmp_path, 0, f"""
import os,time
from pathlib import Path
Path({str(pid_file)!r}).write_text(str(os.getpid()))
time.sleep(90)
""")
    failing = worker(tmp_path, 1, f"""
import time
from pathlib import Path
for _ in range(200):
 if Path({str(pid_file)!r}).exists():break
 time.sleep(.01)
raise SystemExit(7)
""")
    with pytest.raises(RuntimeError, match="worker 1 exited 7"):
        supervise([sibling, failing], tmp_path, lambda: None, poll_seconds=0.01)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
    state = json.loads((tmp_path / "parallel_status.json").read_text())
    assert state["status"] == "failed" and state["workers"]["1"]["exit_code"] == 7
    assert state["workers"]["0"]["status"] == "stopped"


def test_stop_file_interrupts_before_any_worker_launch(tmp_path):
    stop = tmp_path / "stop"
    stop.touch()
    plan = [worker(tmp_path, 0, "raise AssertionError('must not launch')")]
    with pytest.raises(EvaluationInterrupted):
        supervise(plan, tmp_path, lambda: None, stop_file=stop, poll_seconds=0.01)
    assert json.loads((tmp_path / "parallel_status.json").read_text())["status"] == "interrupted"


def test_signal_during_spawn_is_delayed_until_worker_registration():
    registered = False
    with termination_signals():
        with pytest.raises(EvaluationInterrupted) as error:
            with registered_spawn():
                os.kill(os.getpid(), signal.SIGTERM)
                registered = True
        assert registered and error.value.signum == signal.SIGTERM


def test_successful_worker_does_not_leave_renderer_descendants_alive(tmp_path):
    pid_file = tmp_path / "renderer.pid"
    w = complete_worker(tmp_path, 0)
    w.command[-1] += f"""
import subprocess,sys
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(90)'])
Path({str(pid_file)!r}).write_text(str(child.pid))
"""
    state = supervise([w], tmp_path, lambda: None, poll_seconds=0.01)
    assert state["status"] == "complete"
    # An orphan may remain a zombie until init reaps it, but it must not run.
    from pathlib import Path
    status = Path(f"/proc/{int(pid_file.read_text())}/stat")
    for _ in range(100):
        if not status.exists() or status.read_text().split()[2] == "Z":
            break
        time.sleep(0.01)
    assert not status.exists() or status.read_text().split()[2] == "Z"


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_signals_cleanup_children_before_unlocking_output(tmp_path, signum):
    pid_file = tmp_path / "worker.pid"
    child_code = f"import os,time;from pathlib import Path;Path({str(pid_file)!r}).write_text(str(os.getpid()));time.sleep(90)"
    driver = tmp_path / "driver.py"
    driver.write_text(f"""
import os,sys
from pathlib import Path
from stmem.evaluation.parallel import Worker,supervise,output_lock,termination_signals,EvaluationInterrupted
p=Path({str(tmp_path)!r})
w=Worker(0,[sys.executable,'-u','-c',{child_code!r}],dict(os.environ),p/'shards/shard000',[0],{{}})
try:
 with output_lock(p),termination_signals():
  supervise([w],p,lambda:None,poll_seconds=.01)
except EvaluationInterrupted as error:
 raise SystemExit(128+error.signum)
""")
    process = subprocess.Popen([sys.executable, str(driver)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for _ in range(300):
            if pid_file.exists():
                break
            if process.poll() is not None:
                pytest.fail(process.communicate()[1].decode())
            time.sleep(0.01)
        assert pid_file.exists()
        process.send_signal(signum)
        _, err = process.communicate(timeout=15)
        assert process.returncode == 128+signum, err.decode()
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
        with output_lock(tmp_path):
            pass
        assert json.loads((tmp_path / "parallel_status.json").read_text())["status"] == "interrupted"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize("source_changed", [False, True])
def test_parent_main_aggregates_full_plan_with_one_tracker_and_checks_identity(tmp_path, monkeypatch, source_changed):
    """Real subprocess/result aggregation; CPU fixtures explicitly are not policy evaluations."""
    from stmem.evaluation import devices, parallel
    from stmem.evaluation.metrics import TASKS

    cfg = {"seed": 0, "data": {"tasks": list(TASKS)}, "model": {"budget": 128},
           "evaluation": {"seed": 1701, "episode_count": 4, "video_cap": 0},
           "tracking": {"enabled": True}}
    checkpoint = tmp_path / "synthetic.pt"
    checkpoint.write_bytes(b"synthetic orchestration fixture, not a checkpoint")
    output = tmp_path / "result"
    trackers = []

    class Tracker:
        def __init__(self, _cfg, _output, **kwargs):
            self.kwargs, self.logged, self.summaries = kwargs, [], {}
            trackers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def log(self, values, *, step):
            self.logged.append((step, values))

        def summary(self, values):
            self.summaries.update(values)

    def fake_plan(_args, out, parent, _devices):
        plan = []
        for index in range(parent["num_shards"]):
            child = deepcopy(parent)
            child["planned"] = parent["all_planned"][index::parent["num_shards"]]
            child["shard_index"] = index
            child["config"]["tracking"]["enabled"] = False
            directory = out / "shards" / f"shard{index:03d}"
            code = f"""
import hashlib,json
from pathlib import Path
p=Path({str(directory)!r})
child=json.loads({json.dumps(child)!r})
def write(path, value):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)
write(p/'manifest.json',child)
identity=hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()
for task,episode in child['planned']:
 row={{'task':task,'episode_idx':episode,'status':'timeout','method':'FULL','split':'val',
       'training_seed':0,'noise_seed':1701+child['config']['data']['tasks'].index(task)*1000+episode,
       'checkpoint_sha256':child['policy_checkpoint_sha256'],'evaluation_manifest_sha256':identity}}
 write(p/'episodes'/f'{{task}}_{{episode:03d}}.json',row)
write(p/'complete.json',{{'status':'complete','manifest_sha256':identity,
 'planned':len(child['planned']),'recorded':len(child['planned'])}})
"""
            plan.append(worker(out, index, code))
        return plan

    source_calls = []

    def source():
        source_calls.append(True)
        return {"source_sha256": "changed" if source_changed and len(source_calls) > 1 else "fixed"}

    monkeypatch.setattr(parallel, "RunTracker", Tracker)
    monkeypatch.setattr(parallel, "load_config", lambda _: deepcopy(cfg))
    monkeypatch.setattr(parallel, "load_checkpoint", lambda _: {"config": cfg})
    monkeypatch.setattr(parallel, "evaluation_execution_config", lambda _checkpoint, config, **_: (config, None))
    monkeypatch.setattr(parallel, "environment_manifest", lambda: {"synthetic": True})
    monkeypatch.setattr(parallel, "source_revision", source)
    monkeypatch.setattr(parallel, "worker_plan", fake_plan)
    monkeypatch.setattr(devices, "discover_devices", lambda count: [{"synthetic": True}] * count)
    argv = ["parallel", "--config", "synthetic.yaml", "--checkpoint", str(checkpoint),
            "--output", str(output), "--num-gpus", "2", "--workers-per-gpu", "2",
            "--video-cap", "0", "--poll-seconds", ".01"]
    monkeypatch.setattr(sys, "argv", argv)
    if source_changed:
        with pytest.raises(ValueError, match="Source changed"):
            parallel.main()
        assert json.loads((output / "parallel_status.json").read_text())["status"] == "failed"
        assert not (output / "complete.json").exists()
        assert trackers[0].summaries["evaluation/status"] == "failed"
        return
    parallel.main()
    assert len(trackers) == 1
    assert trackers[0].kwargs["job_type"] == "evaluation"
    assert {key for _, values in trackers[0].logged for key in values} == {
        "evaluation/success_rate_so_far_percent", "evaluation/progress_percent"}
    summary = json.loads((output / "summary.json").read_text())
    assert summary["overall"]["planned"] == summary["overall"]["recorded"] == 64
    assert summary["worker_completion"]["complete"] == 4
    assert json.loads((output / "complete.json").read_text())["recorded"] == 64
    # Exact-output resume verifies existing artifacts and skips every subprocess.
    argv.append("--resume")
    monkeypatch.setattr(subprocess, "Popen", lambda *_, **__: pytest.fail("Completed workers must be reused"))
    parallel.main()
    assert len(trackers) == 2 and trackers[1].kwargs["resume"] is True
    assert trackers[1].summaries["evaluation/progress"]["reused_episodes"] == 64
    assert trackers[1].summaries["evaluation/progress"]["new_episodes"] == 0

