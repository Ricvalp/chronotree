"""Fixed scene probes cannot change training state or expose targets to inference."""

from copy import deepcopy
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch
from torch import nn

from stmem.config import load_config, validate_config
from stmem.data.canonical import Decision, Frame
from stmem.memory.hierarchy import Archive
from stmem.memory.records import leaf_record
from stmem.tracking import RunTracker
from stmem.visualization.diagnostics import PolicyScenes


REST = torch.tensor([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4], dtype=torch.float32)


def decision(episode, index, horizon, task):
    current = Frame(
        points=torch.tensor([[0.1, 0.2, 0.3, 0.2, 0.5, 0.7], [0.2, 0.3, 0.1, 0.8, 0.1, 0.6]]),
        valid=torch.ones(2, dtype=torch.bool),
        proprio=torch.cat((REST, torch.tensor([-0.0605, 0, 0.5211, 0, 1, 0, 0, 0.04]))),
        previous_action=torch.zeros(8),
        action_available=False,
        time=float(index),
        source="execution",
        event_id=f"{episode}:{index}",
    )
    target = torch.cat((REST, torch.tensor([1.0]))).repeat(horizon, 1)
    target[:, 0] += 0.01 * torch.arange(1, horizon + 1)
    return Decision(
        current,
        [object()],
        [object()],
        target,
        torch.ones_like(target, dtype=torch.bool),
        "diagnostic fixture",
        task,
        episode,
        index,
    )


class SceneData:
    synthetic = False
    horizon = 4
    tasks = ["VideoUnmask", "ButtonUnmask"]
    refs = ["ref0", "ref1"]
    candidates = {"VideoUnmask": [(0, 0), (0, 4), (0, 9)], "ButtonUnmask": [(1, 1), (1, 4)]}

    def __init__(self, consume):
        self.consume = consume
        self.loads = []
        self.decisions = []
        self.dataset = SimpleNamespace(load_episode=self.load_episode)

    def load_episode(self, reference):
        self.consume()
        self.loads.append(reference)
        episode = self.refs.index(reference)

        def get_decision(index, horizon):
            item = decision(f"episode_{episode}", index, horizon, self.tasks[episode])
            self.decisions.append(item)
            return item

        return SimpleNamespace(decision=get_decision)

    def episode(self, *_):
        raise AssertionError("Diagnostic loading must not perturb the training LRU")

    def sample_position(self, *_):
        raise AssertionError("Diagnostic loading must not consume the training sampler")


def assert_target_free(inputs):
    assert not inputs.target_chunk.any()
    assert not inputs.target_valid.any()
    assert not torch.is_grad_enabled()
    assert torch.is_inference_mode_enabled()


class ScenePolicy(nn.Module):
    device = torch.device("cpu")

    def __init__(self, consume):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
        self.consume = consume
        self.encodings = []
        self.predictions = []
        self.archive = Archive()
        self.archive.add_block(
            [
                leaf_record(
                    str(i), torch.ones(8) * i, torch.tensor([float(i), 0, 0]), float(i), "execution", "visual"
                )
                for i in range(4)
            ]
        )
        # Keep intentionally mixed modes, as with a frozen/eval encoder in a trainer.
        self.train()
        self.layers[0].eval()

    def evidence(self, inputs):
        assert_target_free(inputs)
        assert not any(module.training for module in self.modules())
        self.consume()
        return SimpleNamespace(archive=self.archive, decision=inputs)

    def encode(self, inputs):
        self.encodings.append(inputs)
        return self.evidence(inputs)

    def predict(self, evidence, records, *, noise, perceiver):
        assert_target_free(evidence.decision)
        assert not any(module.training for module in self.modules())
        self.consume()
        self.predictions.append((evidence, list(records), noise.clone(), perceiver))
        # Deliberately ignore expert targets; fixed independent noise affects prediction.
        return torch.cat((REST, torch.ones(1))).repeat(len(noise), 1) + 1e-4 * noise


class SceneCache:
    def __init__(self, policy):
        self.policy = policy
        self.calls = []

    def evidence(self, inputs):
        self.calls.append(inputs)
        return self.policy.evidence(inputs)


class SceneTracker:
    enabled = True

    def __init__(self, consume):
        self.consume = consume
        self.media = []
        self.logs = []
        self.summaries = []

    def log_scenes(self, scenes, *, step):
        self.consume()
        self.media.append((step, scenes))
        return {"seconds": 0.001, "bytes": 123}

    def log(self, values, *, step):
        self.logs.append((step, values))

    def summary(self, values):
        self.summaries.append(values)


@pytest.fixture
def rng_probe(monkeypatch):
    # Exercise CUDA RNG preservation on CPU CI without initializing any GPU.
    cuda_state = [torch.tensor([11, 29], dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [value.clone() for value in cuda_state])
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all", lambda states: cuda_state.__setitem__(slice(None), states)
    )

    def consume():
        random.random()
        np.random.random()
        torch.rand(3)
        cuda_state[0].add_(1)

    def snapshot():
        return random.getstate(), np.random.get_state(), torch.get_rng_state(), cuda_state[0].clone()

    def assert_unchanged(before):
        after = snapshot()
        assert after[0] == before[0]
        assert after[1][0] == before[1][0]
        np.testing.assert_array_equal(after[1][1], before[1][1])
        assert after[1][2:] == before[1][2:]
        assert torch.equal(after[2], before[2])
        assert torch.equal(after[3], before[3])

    return SimpleNamespace(consume=consume, snapshot=snapshot, assert_unchanged=assert_unchanged)


def setup_probe(rng_probe, *, stage="teacher", cache=False, count=2):
    cfg = {
        "seed": 7,
        "model": {"horizon": 4, "budget": 2},
        "tracking": {"scenes": {"enabled": True, "count": count, "max_points": 1}},
    }
    data, policy, tracker = (
        SceneData(rng_probe.consume),
        ScenePolicy(rng_probe.consume),
        SceneTracker(rng_probe.consume),
    )
    cached = SceneCache(policy) if cache else None
    probe = PolicyScenes(cfg, data, policy, tracker, stage=stage, cache=cached)
    return probe, data, policy, tracker, cached


@pytest.mark.parametrize(
    "stage,method,records,cached",
    [
        ("teacher", "FULL", 4, False),
        ("compact", "UNIFORM", 2, True),
        ("perceiver", "PERCEIVER", 4, True),
    ],
)
def test_fixed_scenes_noise_and_context_preserve_training_state(rng_probe, stage, method, records, cached):
    probe, data, policy, tracker, cache = setup_probe(rng_probe, stage=stage, cache=cached)
    modes = [module.training for module in policy.modules()]
    weights = deepcopy(policy.state_dict())
    before = rng_probe.snapshot()
    probe.log(4)
    rng_probe.assert_unchanged(before)
    assert [module.training for module in policy.modules()] == modes
    assert data.loads == ["ref0", "ref1"]
    # First available index >= two chunks; shorter episodes use their first candidate.
    assert [item.index for item in data.decisions] == [9, 1]
    # Training can consume RNG between diagnostics without changing fixed probes.
    rng_probe.consume()
    before = rng_probe.snapshot()
    probe.log(8)
    rng_probe.assert_unchanged(before)
    assert [module.training for module in policy.modules()] == modes
    assert data.loads == ["ref0", "ref1"]
    assert len(policy.predictions) == 4
    assert len(cache.calls) == 4 if cached else len(policy.encodings) == 4
    if cached:
        assert not policy.encodings
    for i in range(2):
        first, second = policy.predictions[i], policy.predictions[i + 2]
        assert first[0].decision.current is second[0].decision.current
        assert first[0].decision.current is data.decisions[i].current
        assert first[0].decision.execution_prefix is data.decisions[i].execution_prefix
        assert first[0].decision.conditioning is data.decisions[i].conditioning
        assert len(first[1]) == records
        assert first[3] == (stage == "perceiver")
        assert torch.equal(first[2], second[2])
        expected_noise = torch.randn((4, 8), generator=torch.Generator().manual_seed(7 * 1009 + i))
        assert torch.equal(first[2], expected_noise)
        assert data.decisions[i].target_chunk.any() and data.decisions[i].target_valid.all()
        earlier, later = tracker.media[0][1][i], tracker.media[1][1][i]
        np.testing.assert_array_equal(earlier.payload["points"], later.payload["points"])
        assert earlier.metadata["recorded_actions_raw"] == data.decisions[i].target_chunk.tolist()
        assert later.metadata["context_method"] == method
        assert later.metadata["context_records"] == records
        assert later.metadata["context_budget"] == 2
        assert later.metadata["optimizer_step"] == 8
        assert later.metadata["diagnostic"] == "training_examples_only"
    assert not torch.equal(policy.predictions[0][2], policy.predictions[1][2])
    for name, value in policy.state_dict().items():
        assert torch.equal(value, weights[name])
    assert probe.events == 2
    assert tracker.logs[-1][1]["diagnostics/scenes_logged"] == 2


@pytest.mark.parametrize(
    "count,single_episode,candidates,expected",
    [
        (1, False, [0, 4, 9], [9]),
        (1, True, [0, 4, 9], [9]),
        (2, True, [0, 4, 9], [9, 0]),
        (2, True, [0], [0]),
    ],
)
def test_one_scene_and_single_episode_fallback_selection(
    rng_probe, count, single_episode, candidates, expected
):
    probe, data, _, tracker, _ = setup_probe(rng_probe, count=count)
    if single_episode:
        data.tasks = ["VideoUnmask"]
        data.candidates = {"VideoUnmask": [(0, index) for index in candidates]}
    probe.log(4)
    assert data.loads == ["ref0"]
    assert [item.index for item in probe.decisions] == expected
    assert len({item.current.event_id for item in probe.decisions}) == len(expected)
    assert len(tracker.media[-1][1]) == len(expected)
    probe.log(8)
    assert data.loads == ["ref0"]


@pytest.mark.parametrize("cached", [False, True])
def test_evidence_stays_fp32_while_decoder_uses_requested_autocast(rng_probe, monkeypatch, cached):
    probe, _, policy, _, _ = setup_probe(rng_probe, stage="compact" if cached else "teacher", cache=cached)
    probe.precision = "bf16"
    original_evidence, original_predict = policy.evidence, policy.predict
    observed = []

    def evidence(inputs):
        observed.append(("evidence", torch.is_autocast_enabled("cpu")))
        return original_evidence(inputs)

    def predict(*args, **kwargs):
        observed.append(("decoder", torch.is_autocast_enabled("cpu")))
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(policy, "evidence", evidence)
    monkeypatch.setattr(policy, "predict", predict)
    probe.log(4)
    assert observed == [("evidence", False), ("decoder", True)] * 2


@pytest.mark.parametrize("failure", ["load", "encode", "predict", "media"])
def test_exceptions_restore_all_rng_and_exact_mixed_module_modes(rng_probe, monkeypatch, failure):
    probe, data, policy, tracker, _ = setup_probe(rng_probe)

    def fail(*_, **__):
        rng_probe.consume()
        raise RuntimeError(f"fixture {failure} failure")

    owner, attribute = {
        "load": (data.dataset, "load_episode"),
        "encode": (policy, "encode"),
        "predict": (policy, "predict"),
        "media": (tracker, "log_scenes"),
    }[failure]
    monkeypatch.setattr(owner, attribute, fail)
    modes = [module.training for module in policy.modules()]
    before = rng_probe.snapshot()
    with pytest.raises(RuntimeError, match=f"fixture {failure} failure"):
        probe.log(4)
    rng_probe.assert_unchanged(before)
    assert [module.training for module in policy.modules()] == modes
    assert probe.events == 0


@pytest.mark.parametrize("disabled", ["scene_option", "tracker", "rank", "synthetic"])
def test_disabled_nonzero_rank_and_synthetic_never_load_scenes(rng_probe, tmp_path, monkeypatch, disabled):
    probe, data, policy, tracker, _ = setup_probe(rng_probe)
    if disabled == "scene_option":
        probe.cfg["tracking"]["scenes"]["enabled"] = False
    elif disabled == "tracker":
        tracker.enabled = False
    elif disabled == "rank":
        monkeypatch.delenv("WANDB_MODE", raising=False)
        tracker = RunTracker(probe.cfg, tmp_path / "rank1", job_type="teacher", rank=1)
        assert not tracker.enabled
    else:
        data.synthetic = True
    probe = PolicyScenes(probe.cfg, data, policy, tracker, stage="teacher")
    before = rng_probe.snapshot()
    probe.log(4)
    rng_probe.assert_unchanged(before)
    assert not probe.enabled and probe.decisions is None
    assert not data.loads and not policy.encodings and not policy.predictions
    if disabled == "synthetic":
        assert "Synthetic" in tracker.summaries[-1]["diagnostics/scenes_skipped"]
    assert not (tmp_path / "rank1").exists()


def test_calibration_error_is_explicit_and_not_retried(rng_probe):
    probe, data, policy, tracker, _ = setup_probe(rng_probe)
    probe.decisions = probe._examples()
    probe.decisions[0].current.proprio[7] += 0.01
    before = rng_probe.snapshot()
    with pytest.warns(UserWarning, match="inconsistent.*position="):
        probe.log(4)
    rng_probe.assert_unchanged(before)
    assert probe.failed == {0}
    assert set(tracker.media[-1][1]) == {1}
    skipped = tracker.logs[-1][1]["diagnostics/scenes_skipped"]
    assert "inconsistent" in skipped["0"] and "position=" in skipped["0"]
    assert tracker.summaries[-1]["diagnostics/scenes_failed"] == [0]
    count = len(policy.predictions)
    probe.log(8)
    assert len(policy.predictions) == count + 1
    assert set(tracker.media[-1][1]) == {1}
    assert tracker.summaries[-1]["diagnostics/scenes_failed"] == [0]


def test_nonfinite_prediction_is_explicit_but_retried_at_next_checkpoint(rng_probe, monkeypatch):
    probe, data, policy, tracker, _ = setup_probe(rng_probe, count=1)
    original_predict = policy.predict

    def transient_prediction(*args, **kwargs):
        prediction = original_predict(*args, **kwargs)
        if len(policy.predictions) == 1:
            prediction[0, 0] = float("nan")
        return prediction

    monkeypatch.setattr(policy, "predict", transient_prediction)
    before = rng_probe.snapshot()
    with pytest.warns(UserWarning, match="nonfinite predicted"):
        probe.log(4)
    rng_probe.assert_unchanged(before)
    assert not probe.failed
    assert tracker.media[-1][1] == {}
    assert "nonfinite predicted" in tracker.logs[-1][1]["diagnostics/scenes_skipped"]["0"]
    probe.log(8)
    rng_probe.assert_unchanged(before)
    assert not probe.failed
    assert set(tracker.media[-1][1]) == {0}
    assert len(policy.predictions) == 2
    assert torch.equal(policy.predictions[0][2], policy.predictions[1][2])
    assert data.loads == ["ref0"]


@pytest.mark.parametrize(
    "options",
    [
        {"count": 0},
        {"count": 3},
        {"count": True},
        {"count": 1.5},
        {"max_points": 0},
        {"max_points": 4097},
        {"max_points": False},
        {"enabled": "yes"},
        {"every_step": True},
        [],
    ],
)
def test_scene_configuration_rejects_unbounded_or_ambiguous_options(options):
    cfg = load_config("configs/base.yaml")
    cfg["tracking"]["scenes"] = options
    with pytest.raises(ValueError, match="tracking.scenes"):
        validate_config(cfg)


@pytest.mark.parametrize("count,max_points", [(1, 1), (2, 4096)])
def test_scene_configuration_accepts_documented_bounds(count, max_points):
    cfg = load_config("configs/base.yaml")
    cfg["tracking"]["scenes"] = {"count": count, "max_points": max_points}
    validate_config(cfg)
