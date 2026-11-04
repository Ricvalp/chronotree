"""Synthetic CPU checks for causal button diagnostics and execution accounting."""

from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stmem.evaluation.robomme_adapter import ResetResult, RoboMMEAdapter, StepResult
from stmem.evaluation.runner import run_episode
from stmem.inspection import button_probe
from stmem.inspection.button_branch import branch_commands, state_error


def test_metrics_detect_motion_confined_to_discarded_half_without_claiming_contact():
    current, button = np.array([0., 0., .15]), np.array([0., 0., .02])
    targets = np.repeat(current[None], 16, axis=0)
    targets[8:, 2] = np.linspace(.14, .04, 8)
    metrics = button_probe.chunk_metrics(targets, current, button)
    assert metrics["progress_at_target8_mm"] == 0
    assert metrics["first8_displacement_mm"] == 0
    assert metrics["progress_at_target16_mm"] == pytest.approx(110.)
    assert metrics["extra_tail_progress_mm"] == pytest.approx(110.)
    assert metrics["descent_at_target16_mm"] == pytest.approx(110.)
    assert "success" not in metrics and "press" not in metrics


@pytest.mark.parametrize("targets,execute", [
    (np.zeros((16, 2)), 8),
    (np.zeros((16, 3)), 16),
    (np.full((16, 3), np.nan), 8),
])
def test_malformed_chunks_cannot_produce_plausible_geometric_metrics(targets, execute):
    with pytest.raises(ValueError):
        button_probe.chunk_metrics(targets, [0, 0, .15], [0, 0, 0], execute)


def test_phase_annotations_only_read_state_and_use_active_index(monkeypatch):
    def forbidden():
        raise AssertionError("Diagnostic observer must not invoke task predicates")

    env = SimpleNamespace(
        timestep=1, current_task_index=0,
        task_list=[{"name": "put down", "func": forbidden}, {"name": "press button", "func": forbidden}],
        cap_link=[SimpleNamespace(pose=SimpleNamespace(p=np.array([[.1, 0., .02]])))],
        button=SimpleNamespace(get_qpos=lambda: np.array([[-.003]])), evaluate=forbidden,
    )
    adapter = SimpleNamespace(env=SimpleNamespace(unwrapped=env), _step=80)
    proprio = np.zeros(15)
    proprio[7:10] = [.1, 0., .15]
    monkeypatch.setattr(button_probe, "native_proprio", lambda _: proprio)
    before = (env.timestep, env.current_task_index, deepcopy(proprio))
    row = button_probe.diagnostic_state(adapter)
    assert row["task_index"] == 1 and row["final_button_phase"]
    assert row["task_name"] == "press button"
    assert row["height_above_button_mm"] == pytest.approx(130.)
    assert row["button_depth_m"] == pytest.approx(.003)
    assert (env.timestep, env.current_task_index) == before[:2]
    np.testing.assert_array_equal(proprio, before[2])


def test_extra_probes_preserve_production_rng_and_use_same_observed_evidence(monkeypatch):
    online = button_probe.TracedOnline.__new__(button_probe.TracedOnline)
    online.generator = torch.Generator().manual_seed(391)
    online.selection_rng = random.Random(519)
    online.method, online.budget = "TREE", 128
    online.autocast_dtype, online.precision = torch.bfloat16, "fp32"
    online.cfg = {"model": {"flow_steps": 8}}
    online.trace_adapter = SimpleNamespace(task="VideoRepick")
    online.memory = SimpleNamespace(archive=object(), current=object())
    online.text = object()
    online.clocks = {"conditioning": 20, "execution": 80}
    seen = []

    def choose(evidence, method, budget, *, rng):
        assert rng is not online.selection_rng
        rng.random()
        seen.append((evidence, method))
        return SimpleNamespace(records=[])

    def integrate(context, bias, *, steps, generator):
        assert generator is not online.generator
        return torch.randn((16, 8), generator=generator) * .01

    fake = SimpleNamespace(
        device=torch.device("cpu"), encoder=SimpleNamespace(current=lambda frame: frame),
        context=lambda evidence, records: (torch.zeros(1), None),
        decoder=SimpleNamespace(integrate=integrate),
        normalizer=SimpleNamespace(denormalize=lambda x: x),
    )
    online.policy = fake
    online.comparison = fake
    monkeypatch.setattr(button_probe, "choose_records", choose)
    torch_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    flow_rng = online.generator.get_state().clone()
    selection_rng = online.selection_rng.getstate()
    rows = online.additional_predictions({"tcp_xyz": [0., 0., .15], "button_xyz": [0., 0., .02]})
    assert len(rows) == 8
    assert [r["noise_seed"] for r in rows] == [2701, 2702, 2703, 2704] * 2
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert python_rng == random.getstate()
    assert torch.equal(flow_rng, online.generator.get_state())
    assert selection_rng == online.selection_rng.getstate()
    assert [method for _, method in seen] == ["TREE", "FULL"]
    assert seen[0][0] is seen[1][0]
    assert seen[0][0].archive is online.memory.archive
    assert seen[0][0].current is online.memory.current


def test_production_executor_discards_tail_holds_prefix_and_stops_on_terminal():
    class Adapter:
        def __init__(self):
            self.commands = []

        def reset(self):
            return ResetResult([], None, "synthetic", {"control_hz": 20, "effective_max_steps": 40})

        def step(self, action):
            self.commands.append(action.cpu().numpy())
            done = len(self.commands) == 17
            return StepResult([], "success" if done else "ongoing", done, False,
                              command=self.commands[-1], observation_skipped=True)

    class Online:
        method, intervention, budget = "UNIFORM", "none", 128
        policy = SimpleNamespace(device="cpu")
        cfg = {"seed": 0, "model": {"horizon": 16, "execute": 8, "action_stride": 2},
               "data": {"observation_stride": 2, "nominal_control_hz": 20},
               "evaluation": {"max_steps": 40}}

        def reset(self, reset, seed):
            self.reset_encoding_seconds, self.update_seconds, self.calls = 0., 0., 0

        def decide(self):
            self.calls += 1
            chunk = torch.zeros(16, 8)
            chunk[:8, 0] = torch.arange(8) + self.calls * 10
            chunk[8:, 0] = 999  # unmistakable tail, never to be issued
            return chunk, {"complete_online_seconds": 0.01}

    adapter, online = Adapter(), Online()
    result = run_episode(adapter, online, task="VideoRepick", episode_idx=0, split="val", seed=0)
    assert result["status"] == "success"
    assert result["primitive_commands"] == 17 and online.calls == 2
    values = np.asarray(adapter.commands)[:, 0]
    np.testing.assert_array_equal(values[:16], np.repeat(np.arange(8) + 10, 2))
    assert values[16] == 20 and 999 not in values


def test_branch_comparison_shares_prefix_and_changes_only_saved_tail():
    first = np.arange(16 * 8).reshape(16, 8).astype(float)
    second = first + 1000.
    baseline = np.vstack([np.repeat(first[:8], 2, axis=0), np.repeat(second[:8], 2, axis=0)])
    trial = {"action_stride": 2, "execute_actions": 8, "actions": baseline.tolist()}
    row = {"native_step": 0, "physical_actions": first.tolist()}
    control = branch_commands(trial, row, "replan8")
    extended = branch_commands(trial, row, "execute16")
    np.testing.assert_array_equal(control[:16], extended[:16])
    np.testing.assert_array_equal(extended[16:], np.repeat(first[8:], 2, axis=0))
    np.testing.assert_array_equal(control[16:], np.repeat(second[:8], 2, axis=0))
    bad = deepcopy(row)
    bad["physical_actions"][0][0] += 1
    with pytest.raises(ValueError, match="identical initial commands"):
        branch_commands(trial, bad, "execute16")


def test_replay_state_check_rejects_structural_nonfinite_and_numeric_mismatches():
    saved = {"robot": [torch.tensor([[1., 2.]]), np.array([3.])], "objects": np.zeros((2, 7))}
    assert state_error(saved, deepcopy(saved)) == 0.
    changed = deepcopy(saved)
    changed["objects"][1, 3] += .012
    assert state_error(changed, saved) == pytest.approx(.012)
    with pytest.raises(ValueError, match="keys"):
        state_error({"robot": saved["robot"]}, saved)
    changed["objects"][1, 3] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        state_error(changed, saved)


def test_stable_diagnostic_id_remapping_preserves_sensors_and_default_transport_ids():
    from test_evaluation_protocol import FakeBenchmark

    ordinary = button_probe.TracedAdapter("VideoRepick", episode_idx=9, env=FakeBenchmark(), points=8, crop=None)
    stable = button_probe.TracedAdapter("SwingXtimes", episode_idx=7, env=FakeBenchmark(), points=8, crop=None,
                                        stable_observation_ids=True)
    # Exercise production reset/ingestion without task-specific privileged labels.
    raw = RoboMMEAdapter.reset(ordinary)
    matched = RoboMMEAdapter.reset(stable)
    raw_frames, matched_frames = raw.conditioning + [raw.current], matched.conditioning + [matched.current]
    assert [f.event_id for f in raw_frames] == [f"{ordinary._reset_id}/reset/{i}" for i in range(3)]
    assert [f.event_id for f in matched_frames] == [f"button-diagnostic/reset/{i}" for i in range(3)]
    for a, b in zip(raw_frames, matched_frames):
        for field in ("points", "valid", "proprio", "previous_action"):
            assert torch.equal(getattr(a, field), getattr(b, field))
        assert (a.source, a.source_index, a.time) == (b.source, b.source_index, b.time)
    action = torch.tensor([0., 0., 0., -1.5, 0., 1.5, .7, 1.])
    step = RoboMMEAdapter.step(stable, action)
    assert step.frames[0].event_id == "button-diagnostic/step/1/0"
    assert all("SwingXtimes" not in f.event_id and "episode" not in f.event_id for f in matched_frames)
    # An ordinary duplicate delivery retains the underlying ingestion deduplication.
    duplicate = stable.ingest(stable.last_raw_observation, {}, event_id="a-new-transport-uuid", command=action)
    assert duplicate == []
