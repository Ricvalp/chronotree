"""Synthetic CPU diagnostics for observational, resume-safe attention probes."""

from copy import deepcopy
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from stmem.config import load_config
from stmem.data.dataset import synthetic_episode
from stmem.models.attention_diagnostics import GROUP_NAMES
from stmem.models.policy import MemoryPolicy
from stmem.training.attention import PolicyAttention, context_groups
from stmem.training.data import TrainingData
from stmem.visualization.diagnostics import PolicyScenes


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(281017)


def tiny_config():
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["model"].update(
        width=16, heads=4, layers=2, text_layers=1, max_text_bytes=24,
        points=12, block_size=2, current_tokens=4, visual_tokens=8, horizon=4,
        execute=2, flow_steps=2, checkpoint_blocks=False,
        budget=32, budgets=[32],
        condition_on_previous_joint_commands=False,
    )
    cfg["training"].update(workers=0, effective_batch=2, microbatch=2, encoder_block_batch_size=2)
    cfg["data"].update(
        tasks=["VideoUnmask", "PickXtimes"], episodes_per_task=2,
        synthetic_execution_frames=12, synthetic_conditioning_frames=2,
    )
    cfg["tracking"].update(enabled=True, system_stats=False, mode="offline")
    cfg["tracking"]["scenes"]["count"] = 2
    return cfg


class Tracker:
    def __init__(self, directory, *, enabled=True):
        self.enabled, self.directory = enabled, directory
        self.identity = {"attempt": 2}
        self.logs, self.summaries = [], []

    def log(self, values, *, step):
        self.logs.append((step, values))

    def summary(self, values):
        self.summaries.append(values)


def setup_probe(tmp_path, *, every=10, enabled=True, stage="teacher"):
    cfg = tiny_config()
    data = TrainingData(cfg)
    policy = MemoryPolicy(cfg)
    tracker = Tracker(tmp_path, enabled=enabled)
    scenes = PolicyScenes(cfg, data, policy, tracker, stage=stage)
    probe = PolicyAttention(cfg, policy, tracker, scenes, stage=stage, every=every)
    return cfg, data, policy, tracker, scenes, probe


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(before):
    after = rng_state()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    np.testing.assert_array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    torch.testing.assert_close(before[2], after[2], atol=0, rtol=0)


def hook_counts(policy):
    return [(len(m._forward_pre_hooks), len(m._forward_hooks)) for m in policy.modules()]


def subprocess_environment():
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("STMEM_", "WANDB_", "SLURM_"))
        and key not in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
    }
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", WANDB_MODE="offline")
    return env


def test_fixed_probes_preserve_state_sampler_and_rng_and_recompute_learned_content(tmp_path, monkeypatch):
    cfg, data, policy, tracker, scenes, probe = setup_probe(tmp_path)
    policy.train()
    policy.encoder.eval()  # Deliberately mixed modes, not just one model.training flag.
    policy.decoder.layers[0].ff.eval()
    for index, parameter in enumerate(policy.parameters()):
        parameter.grad = torch.full_like(parameter, (index + 1) * 0.001)
    weights = deepcopy(policy.state_dict())
    gradients = [p.grad.clone() for p in policy.parameters()]
    modes = [m.training for m in policy.modules()]
    original_hooks = hook_counts(policy)
    references = [data.sample_reference(i) for i in range(12)]
    cache_info = data.episode.cache_info(), data._permutation.cache_info()
    candidates = deepcopy(data.candidates)
    encoded, noises, current_contents = [], [], []
    original_encode, original_predict = policy.encode, policy.predict

    def encode(inputs):
        assert not inputs.target_chunk.any() and not inputs.target_valid.any()
        assert not torch.is_grad_enabled() and torch.is_inference_mode_enabled()
        assert not any(m.training for m in policy.modules())
        assert not torch.is_autocast_enabled("cpu")
        # A dependency may consume randomness; the enclosing probe must restore it.
        random.random()
        np.random.random()
        torch.rand(3)
        encoded.append(inputs)
        evidence = original_encode(inputs)
        current_contents.append(torch.stack([r.content for r in evidence.current_records]).clone())
        return evidence

    def predict(evidence, records, *, noise):
        noises.append(noise.clone())
        return original_predict(evidence, records, noise=noise)

    monkeypatch.setattr(policy, "encode", encode)
    monkeypatch.setattr(policy, "predict", predict)
    before = rng_state()
    probe.log(10)
    assert_rng_equal(before)
    assert len(scenes.decisions) == 2
    first_decisions = list(scenes.decisions)
    assert [(d.task, d.episode_id, d.index) for d in first_decisions] == [
        ("VideoUnmask", "episode_0", 8), ("PickXtimes", "episode_1", 8),
    ]
    # Poison supervision after caching. Inference metrics must remain identical.
    with torch.inference_mode():
        for decision in scenes.decisions:
            decision.target_chunk.fill_(float("nan"))
            decision.target_valid.fill_(True)
    random.random()
    before = rng_state()
    probe.log(20)
    assert_rng_equal(before)
    assert data.episode.cache_info() == cache_info[0]
    assert data._permutation.cache_info() == cache_info[1]
    assert data.candidates == candidates
    assert [data.sample_reference(i) for i in range(12)] == references
    assert [m.training for m in policy.modules()] == modes
    assert hook_counts(policy) == original_hooks
    assert torch.is_grad_enabled()
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
    for parameter, gradient in zip(policy.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient, atol=0, rtol=0)
    for index in range(2):
        assert encoded[index].current is encoded[index + 2].current is first_decisions[index].current
        torch.testing.assert_close(noises[index], noises[index + 2], atol=0, rtol=0)
        expected = torch.randn((4, 8), generator=torch.Generator().manual_seed(cfg["seed"] * 1009 + index))
        torch.testing.assert_close(noises[index], expected, atol=0, rtol=0)
    rows = [json.loads(line) for line in (tmp_path / "attention.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [10, 20]
    assert rows[0]["metrics"] == rows[1]["metrics"]
    assert rows[0]["flow_time"] == 0.5 and len(rows[0]["examples"]) == 2
    assert rows[0]["attempt"] == 2
    for row in rows:
        assert sum(row["metrics"][f"mass/{group}"] for group in GROUP_NAMES) == pytest.approx(1)
    # Only raw inputs persist: changing visual weights changes the next encoded evidence.
    with torch.no_grad():
        next(policy.encoder.visual.parameters()).add_(0.03)
    probe.log(30)
    assert len(encoded) == 6
    assert not torch.equal(current_contents[0], current_contents[4])
    assert [step for step, _ in tracker.logs] == [10, 20, 30]


@pytest.mark.parametrize("every,enabled,stage", [(0, True, "teacher"), (10, False, "teacher"), (10, True, "perceiver")])
def test_disabled_probe_has_no_loading_encoding_or_hooks(tmp_path, monkeypatch, every, enabled, stage):
    _, _, policy, tracker, scenes, probe = setup_probe(tmp_path, every=every, enabled=enabled, stage=stage)

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled attention probes must not load examples or run the policy")

    monkeypatch.setattr(scenes, "_examples", forbidden)
    monkeypatch.setattr(policy, "encode", forbidden)
    monkeypatch.setattr(policy, "predict", forbidden)
    hooks = hook_counts(policy)
    probe.log(10)
    assert not tracker.logs and scenes.decisions is None
    assert hook_counts(policy) == hooks
    assert not (tmp_path / "attention.jsonl").exists()


@pytest.mark.parametrize("failure", ["encoding", "decoder"])
def test_failure_restores_modes_rng_and_removes_hooks(tmp_path, monkeypatch, failure):
    _, _, policy, tracker, _, probe = setup_probe(tmp_path)
    policy.train()
    policy.encoder.eval()
    modes, hooks = [m.training for m in policy.modules()], hook_counts(policy)

    def fail(*args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(2)
        raise RuntimeError("injected diagnostic failure")

    if failure == "encoding":
        monkeypatch.setattr(policy, "encode", fail)
    else:
        # Failure after cross-attention hooks have run, inside an actual forward.
        monkeypatch.setattr(policy.decoder.out, "forward", fail)
    before = rng_state()
    with pytest.raises(RuntimeError, match="injected diagnostic failure"):
        probe.log(10)
    assert_rng_equal(before)
    assert [m.training for m in policy.modules()] == modes
    assert hook_counts(policy) == hooks
    assert torch.is_grad_enabled() and not tracker.logs
    assert probe.events == 0 and not (tmp_path / "attention.jsonl").exists()


@pytest.mark.parametrize("empty_visual,history", [(False, True), (True, True), (True, False)])
def test_context_groups_follow_actual_serial_context_including_missing_visuals(empty_visual, history):
    policy = MemoryPolicy(tiny_config())
    episode = synthetic_episode(6, 2 if history else 0, 12, seed=281019)
    decision = episode.decision(3 if history else 0, 4)
    if empty_visual:
        for frame in [decision.current, *decision.conditioning, *decision.execution_prefix]:
            frame.valid.zero_()
    with torch.no_grad():
        evidence = policy.encode(decision)
        records = evidence.archive.leaves
        context, bias = policy.context(evidence, records)
    groups = context_groups(evidence, records, policy.device)
    names = [GROUP_NAMES[index] for index in groups.tolist()]
    assert context.shape[0] == len(names) == bias.shape[0]
    current_count = len(evidence.current_records)
    assert names[:current_count] == ["current_visual"] * current_count
    assert names[current_count:current_count + 3] == ["current_state", "goal", "clock"]
    assert names[current_count + 3:] == [
        "history_visual" if record.modality == "visual" else "history_event" for record in records
    ]
    if empty_visual:
        assert "current_visual" not in names and "history_visual" not in names
    if history:
        assert names.count("history_event") == 5  # Two conditioning + three past execution states.
    else:
        assert names == ["current_state", "goal", "clock"]


def test_default_ten_step_probe_logs_independently_of_scalar_interval(tmp_path):
    cfg = tiny_config()
    cfg["paths"]["run_root"] = str(tmp_path / "run")
    cfg["training"].update(
        steps=11, checkpoint_every=11, checkpoint_steps=[11], log_every=50, precision="fp32",
    )
    cfg["tracking"]["scenes"]["enabled"] = False
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = subprocess.run(
        [sys.executable, "-m", "stmem.train", "--config", str(path)],
        env=subprocess_environment(), text=True, capture_output=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    directory = tmp_path / "run/seed0/teacher"
    attention = [json.loads(line) for line in (directory / "attention.jsonl").read_text().splitlines()]
    assert [row["step"] for row in attention] == [10]
    metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in metrics] == [1, 11]
    journal = [json.loads(line)["metrics"] for line in (directory / "tracking_history.jsonl").read_text().splitlines()]
    measured = [row for row in journal if "attention/mass_current" in row]
    assert [row["progress/step"] for row in measured] == [10]
    assert measured[0]["attention/scenes"] == 2


def assert_nested_equal(before, after):
    if isinstance(before, torch.Tensor):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    elif isinstance(before, np.ndarray):
        np.testing.assert_array_equal(before, after)
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            assert_nested_equal(before[key], after[key])
    elif isinstance(before, (list, tuple)):
        assert type(before) is type(after) and len(before) == len(after)
        for left, right in zip(before, after):
            assert_nested_equal(left, right)
    else:
        assert before == after


def test_two_rank_cpu_attention_barrier_preserves_exact_training_and_writes_only_rank_zero(tmp_path):
    checkpoints = []
    for interval in (0, 1):
        cfg = tiny_config()
        run = tmp_path / f"interval{interval}"
        cfg["paths"]["run_root"] = str(run)
        cfg["training"].update(
            steps=2, checkpoint_every=2, checkpoint_steps=[2], log_every=1, precision="fp32",
        )
        cfg["tracking"]["scenes"]["enabled"] = False
        path = tmp_path / f"config{interval}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        result = subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                "--nproc-per-node=2", "-m", "stmem.train", "--config", str(path),
                "--attention-every", str(interval),
            ],
            env=subprocess_environment(), text=True, capture_output=True, timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        directory = run / "seed0/teacher"
        saved = torch.load(directory / "last.pt", weights_only=False, map_location="cpu")
        assert len(saved["rng_by_rank"]) == saved["provenance"]["world_size"] == 2
        checkpoints.append(saved)
        if interval:
            rows = [json.loads(line) for line in (directory / "attention.jsonl").read_text().splitlines()]
            assert [row["step"] for row in rows] == [1, 2]
            assert all(row["attempt"] == 1 and len(row["examples"]) == 2 for row in rows)
            journal = [json.loads(line)["metrics"] for line in (directory / "tracking_history.jsonl").read_text().splitlines()]
            assert len([row for row in journal if "attention/mass_current" in row]) == 2
            assert len(list((directory / "wandb").glob("offline-run-*"))) == 1
        else:
            assert not (directory / "attention.jsonl").exists()
    for key in ("model", "optimizer", "scheduler", "scaler", "sampler", "rng_by_rank", "provenance", "step"):
        assert_nested_equal(checkpoints[0][key], checkpoints[1][key])
