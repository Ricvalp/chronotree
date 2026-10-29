"""Synthetic CPU diagnostics for the isolated teacher memory profiler.

These checks compare real optimizer updates rather than accepting a report
whose instrumentation silently changes the teacher computation.
"""

from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from stmem.config import load_config
from stmem.data.dataset import synthetic_episode
from stmem.training.data import TrainingData


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def profiler():
    name = "test_profile_teacher_memory_script"
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/profile_teacher_memory.py")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses need their defining module registered during class creation.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def tiny_config(tmp_path):
    cfg = load_config(REPO / "configs/debug/teacher.yaml")
    cfg["paths"]["run_root"] = str(tmp_path / "training")
    cfg["model"].update(
        width=16,
        heads=4,
        layers=1,
        text_layers=1,
        max_text_bytes=16,
        points=8,
        current_tokens=4,
        visual_tokens=4,
        horizon=2,
        execute=2,
        checkpoint_blocks=False,
    )
    cfg["training"].update(
        steps=6,
        effective_batch=2,
        microbatch=1,  # Exercise accumulation as well as batched forward.
        workers=0,
        precision="fp32",
        encoder_block_batch_size=2,
        checkpoint_every=6,
        checkpoint_steps=[],
        log_every=1,
    )
    cfg["data"].update(
        episodes_per_task=2,
        synthetic_execution_frames=5,
        synthetic_conditioning_frames=2,
        decision_stride=1,
    )
    cfg["tracking"].update(enabled=True, mode="offline", scenes={"enabled": False})
    return cfg


class FakeTracker:
    instances = []

    def __init__(self, cfg, directory, **kwargs):
        self.directory = Path(directory)
        self.cfg, self.options = deepcopy(cfg), kwargs
        self.identity = {"id": f"diagnostic-{len(self.instances)}", "attempt": 1}
        self.enabled = False
        self.rows, self.summaries = [], []
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def log(self, values, *, step):
        self.rows.append((step, deepcopy(values)))

    def summary(self, values):
        self.summaries.append(deepcopy(values))

    def checkpoint(self, *_args):
        pass

    def finish(self, **_kwargs):
        pass


def assert_identical(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_identical(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("trace_steps", [0, 1])
@pytest.mark.parametrize("history_mode", ["full", "none"])
def test_profile_updates_match_uninstrumented_teacher_and_leave_training_artifacts(
    profiler,
    tmp_path,
    monkeypatch,
    trace_steps,
    history_mode,
):
    from stmem import train

    cfg = tiny_config(tmp_path)
    cfg["model"]["history_mode"] = history_mode
    if history_mode == "none":
        cfg["model"].update(proprioception_mode="none", condition_on_previous_commands=False)
    cfg["training"]["steps"] += trace_steps
    cfg["training"]["checkpoint_every"] = cfg["training"]["steps"]
    original_cfg = deepcopy(cfg)
    FakeTracker.instances = []
    monkeypatch.setattr(train, "RunTracker", FakeTracker)
    monkeypatch.setattr(profiler, "RunTracker", FakeTracker)
    monkeypatch.setattr(train, "distributed_setup", lambda: (torch.device("cpu"), 0, 1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(1)
    monkeypatch.setattr(torch, "set_num_threads", lambda _n: None)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")

    assert train.train_policy(cfg, "teacher") == 0
    directory = Path(cfg["paths"]["run_root"]) / "seed0/teacher"
    checkpoint = torch.load(directory / "last.pt", weights_only=False, map_location="cpu")
    existing_files = {
        str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()
    }
    captured = {}
    original_policy = profiler.MemoryPolicy
    original_optimizer = torch.optim.AdamW

    def policy_factory(*args, **kwargs):
        captured["policy"] = original_policy(*args, **kwargs)
        return captured["policy"]

    def optimizer_factory(*args, **kwargs):
        captured["optimizer"] = original_optimizer(*args, **kwargs)
        return captured["optimizer"]

    monkeypatch.setattr(profiler, "MemoryPolicy", policy_factory)
    monkeypatch.setattr(torch.optim, "AdamW", optimizer_factory)
    output = tmp_path / "profile"
    result = profiler.run_profile(
        cfg,
        profiler.ProfileOptions(output=output, warmup_steps=5, steps=1, trace_steps=trace_steps, cpu=True),
    )
    assert_identical(cfg, original_cfg)
    assert_identical(captured["policy"].state_dict(), checkpoint["model"])
    assert_identical(captured["optimizer"].state_dict(), checkpoint["optimizer"])
    assert {
        str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()
    } == existing_files
    assert not list(output.rglob("*.pt"))
    assert not list(output.rglob("complete.json"))
    assert not list(output.rglob(".stages"))
    assert result["warmup_steps"] == 5
    assert result["measured_steps"] == 1
    assert result["trace_steps"] == trace_steps
    assert result["ranks"][0]["warmup"]["updates"] == 5
    assert result["ranks"][0]["measure"]["updates"] == 1
    assert result["max_measured_peak_allocated_bytes_across_ranks"] is None
    assert (output / "summary.json").exists()
    trace_path = output / "rank0/trace.json"
    if trace_steps:
        assert result["ranks"][0]["trace"]["updates"] == 1
        trace = json.loads(trace_path.read_text())
        labels = {item["name"] for item in trace["traceEvents"] if "name" in item}
        assert {"teacher/current_encoder", "teacher/decoder", "teacher/backward"} <= labels
        assert ("teacher/history_encoder" in labels) == (history_mode == "full")
    else:
        assert not trace_path.exists()
        assert "trace" not in result["ranks"][0]
    tracking = FakeTracker.instances[-1]
    assert tracking.directory == output
    assert tracking.options["job_type"] != "teacher"
    assert tracking.identity["id"] != FakeTracker.instances[0].identity["id"]


def test_existing_profile_directory_is_never_reused(profiler, tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.json"
    sentinel.write_text('{"must_remain": true}\n')

    def forbidden(*_args, **_kwargs):
        pytest.fail("An existing output directory reached model construction or tracking")

    monkeypatch.setattr(profiler, "MemoryPolicy", forbidden)
    monkeypatch.setattr(profiler, "RunTracker", forbidden)
    monkeypatch.setenv("WANDB_MODE", "offline")
    with pytest.raises(FileExistsError):
        profiler.run_profile(
            cfg, profiler.ProfileOptions(output=output, warmup_steps=5, steps=1, trace_steps=0, cpu=True)
        )
    assert json.loads(sentinel.read_text()) == {"must_remain": True}
    assert list(output.iterdir()) == [sentinel]


def test_representative_selection_preserves_training_sampler(profiler, tmp_path):
    cfg = tiny_config(tmp_path)
    original = TrainingData(cfg)
    profiled = profiler.ProfileTrainingData(cfg, scenario="representative")
    positions = list(range(113))
    assert [profiled.sample_reference(i) for i in positions] == [
        original.sample_reference(i) for i in positions
    ]
    assert profiled.candidates == original.candidates


def test_long_history_stress_keeps_tasks_valid_decisions_and_entire_causal_prefix(
    profiler,
    tmp_path,
    monkeypatch,
):
    cfg = tiny_config(tmp_path)
    cfg["data"].update(episodes_per_task=3, synthetic_execution_frames=9)

    def different_history_lengths(self, index):
        task, number, seed = self.refs[index]
        episode = synthetic_episode(
            length=(7, 10, 8)[number],
            conditioning_length=(9, 1, 4)[number],
            points=self.cfg["model"]["points"],
            seed=seed,
        )
        episode.task, episode.episode_id = task, f"episode_{number}"
        return episode

    monkeypatch.setattr(TrainingData, "episode", different_history_lengths)
    original = TrainingData(cfg)
    profiled = profiler.ProfileTrainingData(cfg, scenario="long-history")
    references = [profiled.sample_reference(position) for position in range(24)]
    task_by_episode = {index: task for task, indices in original.by_task.items() for index in indices}
    assert Counter(task_by_episode[index] for index, _ in references) == dict.fromkeys(original.tasks, 12)
    for task in original.tasks:
        expected_longest = max(
            original.candidates[task],
            key=lambda ref: (len(original.episode(ref[0]).conditioning) + ref[1], -ref[0]),
        )
        selected = [ref for ref in references if task_by_episode[ref[0]] == task]
        assert set(selected) == {expected_longest}
    for episode_index, decision_index in references:
        task = task_by_episode[episode_index]
        assert (episode_index, decision_index) in original.candidates[task]
        expected = original.decision(episode_index, decision_index)
        actual = profiled.decision(episode_index, decision_index)
        assert len(actual.conditioning) == len(expected.conditioning)
        assert len(actual.execution_prefix) == len(expected.execution_prefix) == decision_index
        assert all(frame.source_index < actual.current.source_index for frame in actual.execution_prefix)
        assert all(not frame.action_available for frame in actual.conditioning)
        torch.testing.assert_close(actual.target_chunk, expected.target_chunk, rtol=0, atol=0)
        torch.testing.assert_close(actual.target_valid, expected.target_valid)
    # The diagnostic sampler must not replace the normal training sampler.
    subsequent = TrainingData(cfg)
    assert [subsequent.sample_reference(i) for i in range(12)] == [
        original.sample_reference(i) for i in range(12)
    ]


def test_cpu_probe_reports_absent_gpu_memory_and_instrumentation_restores_functions(profiler, tmp_path):
    torch.set_num_threads(1)
    cfg = tiny_config(tmp_path)
    policy = profiler.MemoryPolicy(cfg)
    probe = profiler.MemoryProbe(torch.device("cpu"))
    original_history = profiler.batched_policy.encode_history_bank
    original_decoder = policy.decoder.forward
    probe.begin_update(enabled=True)
    with profiler.instrument(policy, probe):
        assert profiler.batched_policy.encode_history_bank is not original_history
        with probe.phase("cpu_diagnostic"):
            assert torch.ones(2).sum() == 2
    assert profiler.batched_policy.encode_history_bank is original_history
    assert policy.decoder.forward == original_decoder
    result = probe.finish_update()
    assert result["seconds"] >= 0
    assert all(value is None for key, value in result.items() if key.startswith("gpu_"))
    assert len(probe.phases) == 1
    phase = probe.phases[0]
    assert phase["phase"] == "cpu_diagnostic"
    assert phase["cuda_elapsed_ms"] is None
    assert all(value is None for key, value in phase.items() if key.endswith("_bytes"))
    probe.begin_update()
    with pytest.raises(RuntimeError, match="diagnostic failure"):
        with profiler.instrument(policy, probe), probe.phase("failing_phase"):
            raise RuntimeError("diagnostic failure")
    assert profiler.batched_policy.encode_history_bank is original_history
    assert policy.decoder.forward == original_decoder


def test_update_peak_includes_allocations_between_instrumented_phases(profiler, monkeypatch):
    """Resetting a phase counter must not erase an earlier update-wide peak."""
    counters = {"allocated": 12, "reserved": 20, "peak_allocated": 12, "peak_reserved": 20}

    def allocate(allocated, reserved):
        counters["allocated"], counters["reserved"] = allocated, reserved
        counters["peak_allocated"] = max(counters["peak_allocated"], allocated)
        counters["peak_reserved"] = max(counters["peak_reserved"], reserved)

    def reset(_device):
        counters["peak_allocated"] = counters["allocated"]
        counters["peak_reserved"] = counters["reserved"]

    class Event:
        def __init__(self, **_kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, _other):
            return 1.0

    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: counters["allocated"])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: counters["reserved"])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: counters["peak_allocated"])
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: counters["peak_reserved"])
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (900, 1000))
    monkeypatch.setattr(torch.cuda, "Event", Event)
    probe = profiler.MemoryProbe(torch.device("cuda", 0))
    probe.begin_update()
    allocate(101, 120)  # An allocation in a gap outside named model phases.
    allocate(6, 20)
    with probe.phase("first"):
        allocate(47, 70)
        allocate(30, 70)
    with probe.phase("second"):
        allocate(35, 70)
        allocate(15, 70)
    result = probe.finish_update()
    assert [row["peak_allocated_bytes"] for row in probe.phases] == [47, 35]
    assert result["gpu_peak_allocated_bytes"] == 101
    assert result["gpu_peak_reserved_bytes"] == 120
    assert result["gpu_allocated_bytes"] == 15
    assert result["gpu_free_bytes_at_step_end"] == 900


@pytest.mark.parametrize("mode,failed_step", [("warmup", 1), ("measure", 6), ("trace", 7)])
def test_model_failure_keeps_phase_and_completed_measurements(
    profiler,
    tmp_path,
    monkeypatch,
    mode,
    failed_step,
):
    cfg = tiny_config(tmp_path)
    output = tmp_path / "failed-profile"
    original_policy = profiler.MemoryPolicy
    original_error = RuntimeError("injected decoder failure")
    calls = 0

    def policy_factory(*args, **kwargs):
        policy = original_policy(*args, **kwargs)
        original_forward = policy.decoder.forward

        def fail_at_update(*forward_args, **forward_kwargs):
            nonlocal calls
            calls += 1
            if calls == (failed_step - 1) * 2 + 1:
                raise original_error
            return original_forward(*forward_args, **forward_kwargs)

        policy.decoder.forward = fail_at_update
        return policy

    monkeypatch.setattr(profiler, "MemoryPolicy", policy_factory)
    monkeypatch.setattr(profiler, "RunTracker", FakeTracker)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    for variable in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(variable, raising=False)
    options = profiler.ProfileOptions(
        output=output, warmup_steps=5, steps=1, trace_steps=int(mode == "trace"), cpu=True
    )
    with pytest.raises(RuntimeError, match="injected decoder failure") as raised:
        profiler.run_profile(cfg, options)
    assert raised.value is original_error
    failure = json.loads((output / "rank0/failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["failed_phase"] == "decoder"
    assert failure["mode"] == mode
    assert failure["optimizer_step"] == failure["attempt"] == failed_step
    assert failure["completed_attempts"] == failed_step - 1
    assert failure["error_type"] == "RuntimeError"
    assert failure["error"] == "injected decoder failure"
    assert all(value is None for key, value in failure.items() if key.endswith("_bytes"))
    assert not (output / "summary.json").exists()
    assert not Path(cfg["paths"]["run_root"]).exists()
    if mode == "trace":
        measurements = json.loads((output / "rank0/measurements.json").read_text())
        assert measurements["warmup"]["updates"] == 5
        assert measurements["measure"]["updates"] == 1
        assert "trace" not in measurements


@pytest.mark.parametrize("training_fails", [False, True])
def test_trace_export_failure_preserves_training_error_and_always_disables_history(
    profiler,
    tmp_path,
    monkeypatch,
    training_fails,
):
    """A broken export must neither conceal the original OOM nor leave recording active."""
    original_error = torch.cuda.OutOfMemoryError("original training OOM")
    trace_error = OSError("trace export failed")
    history_calls, snapshots = [], []

    class BrokenExporter:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def export_chrome_trace(self, path):
            assert Path(path) == tmp_path / "trace.json"
            raise trace_error

    def fail_snapshot(path):
        snapshots.append(Path(path))
        raise OSError("snapshot export failed")

    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: BrokenExporter())
    monkeypatch.setattr(torch.cuda.memory, "_record_memory_history", lambda **kw: history_calls.append(kw))
    monkeypatch.setattr(torch.cuda.memory, "_dump_snapshot", fail_snapshot)
    with pytest.raises((torch.cuda.OutOfMemoryError, OSError)) as raised:
        with profiler.detailed_trace(tmp_path, torch.device("cuda", 0)):
            if training_fails:
                raise original_error
    assert raised.value is (original_error if training_fails else trace_error)
    assert snapshots == [tmp_path / "memory_snapshot.pickle"]
    assert history_calls[0]["enabled"] == "all"
    assert history_calls[-1] == {"enabled": None}
    report = json.loads((tmp_path / "trace_export_errors.json").read_text())
    assert report["errors"] == ["OSError: trace export failed", "OSError: snapshot export failed"]
    if training_fails:
        assert any("Trace export errors" in note for note in raised.value.__notes__)


@pytest.mark.parametrize("warmup", [0, 4, 11])
def test_warmup_outside_requested_five_to_ten_steps_is_rejected(profiler, warmup):
    with pytest.raises(ValueError, match="5"):
        profiler.ProfileOptions(warmup_steps=warmup).validate()
