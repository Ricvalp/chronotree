"""CPU invariants for compact-specific local simulator jobs; no simulator runs."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stmem.config import load_config
from stmem.inspection.student_simulation import (
    prepare_simulation_job, simulation_jobs, simulation_report_links, validate_simulation_job,
)
from stmem.training.checkpoints import file_sha256


def plan(tmp_path):
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(b"synthetic checkpoint identity fixture")
    return {
        "output": str(tmp_path / "output"), "device": "cuda:0",
        "assets": {"clip_checkpoint": str(tmp_path / "clip.pt")},
        "protocol": {"execute_targets": 8, "budget": 128},
        "checkpoints": {"s_packet": {"path": str(source), "sha256": file_sha256(source), "step": 50000}},
    }


def test_explicit_student_readers_match_cost_and_episode_protocol(tmp_path):
    setup = plan(tmp_path)
    jobs = [job for job in simulation_jobs(setup, python="python") if job["stage"] != "handoff"]
    assert [job["stage"] for job in jobs] == ["reset", "rollouts", "rollouts"]
    assert [job["method"] for job in jobs] == ["CURRENT", "TREE", "FLAT"]
    for job in jobs:
        command = job["command"]
        assert command[command.index("--execute-actions") + 1] == "8"
        assert command[command.index("--method") + 1] == job["method"]
        assert "--submit" not in command and "FULL" not in command
        if job["stage"] == "rollouts":
            assert job["episodes"] == [0] and job["task"] == "VideoRepick"
            assert "--stable-observation-ids" in command
    setup["protocol"]["budget"] = 256
    with pytest.raises(ValueError, match="B128"):
        simulation_jobs(setup)


def _reset_generator():
    path = Path(__file__).resolve().parents[1] / "scripts/visualize_reset_cube_chunks.py"
    spec = importlib.util.spec_from_file_location("reset_for_student_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method,selection", [("CURRENT", None), ("TREE", "tree"), ("FLAT", "flat")])
def test_reset_reader_and_prefix_override_preserve_checkpoint_config(monkeypatch, method, selection):
    module = _reset_generator()
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["model"].update(horizon=16, execute=2, action_stride=2)
    cfg["data"].update(observation_stride=2, process_every_observation=False)
    original = deepcopy(cfg)
    calls = []

    def fake_online(policy, config, reader, budget, **kwargs):
        calls.append((reader, budget, kwargs))
        return SimpleNamespace(method=reader, cfg=config)

    monkeypatch.setattr(module, "OnlinePolicy", fake_online)
    args = SimpleNamespace(visual_checkpoint=None, execute_actions=8, split="val", method=method)
    online, effective, override = module.online_policy_for_reset(object(), {"config": cfg, "stage": "compact"}, args)
    assert cfg == original
    assert effective["model"]["execute"] == 8
    assert override["checkpoint_execute_actions"] == 2
    assert override["replan_primitive_steps"] == 16
    assert online.method == method
    assert calls == [(method, cfg["model"]["budget"], {"selection_family": selection})]


def test_legacy_reset_defaults_keep_saved_execution_length(monkeypatch):
    module = _reset_generator()
    cfg = load_config("configs/debug/teacher.yaml")
    monkeypatch.setattr(module, "OnlinePolicy", lambda _policy, config, method, budget, **_kw:
                        SimpleNamespace(method=method, cfg=config))
    args = SimpleNamespace(visual_checkpoint=None, split="val")
    online, effective, override = module.online_policy_for_reset(object(), {"config": cfg, "stage": "teacher"}, args)
    assert online.method == "FULL" and override is None
    assert effective["model"]["execute"] == cfg["model"]["execute"]


def saved_rollout(tmp_path):
    setup = plan(tmp_path)
    job = simulation_jobs(setup)[1]
    output = Path(job["output"])
    output.mkdir(parents=True)
    video = output / "rollout.mp4"
    video.write_bytes(b"synthetic nonempty video artifact; not a media decode test")
    entry = setup["checkpoints"]["s_packet"]
    trial = {"episode_idx": 0, "task": "VideoRepick", "split": "val", "method": "TREE",
             "checkpoint_sha256": entry["sha256"], "execute_actions": 8, "action_stride": 2,
             "budget": 128, "status": "timeout", "video": str(video)}
    payload = {"checkpoint": {**entry, "stage": "compact"},
               "protocol": {"task": "VideoRepick", "episodes": [0], "execute_actions": 8,
                            "action_stride": 2, "horizon": 16, "stable_observation_ids": True},
               "trials": [trial]}
    (output / "results.json").write_text(json.dumps(payload))
    (output / "config.json").write_text(json.dumps({"model": {"flow_steps": 8}}))
    (output / "index.html").write_text("synthetic report")
    return setup, job, payload


def test_rollout_completion_keeps_timeouts_and_rejects_missing_episodes(tmp_path):
    setup, job, payload = saved_rollout(tmp_path)
    result = validate_simulation_job(setup, job)
    assert result["statuses"] == {"timeout": 1}
    assert result["success_rate"] == 0 and result["planned"] == result["recorded"] == 1
    payload["trials"] = []
    (Path(job["output"]) / "results.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing or duplicated"):
        validate_simulation_job(setup, job)


@pytest.mark.parametrize("change,error", [("method", "reader"), ("budget", "budget"), ("status", "unresolved")])
def test_rollout_wrong_reader_budget_and_infrastructure_error_cannot_pass(tmp_path, change, error):
    setup, job, payload = saved_rollout(tmp_path)
    payload["trials"][0][change] = {"method": "FULL", "budget": 256, "status": "infrastructure_error"}[change]
    (Path(job["output"]) / "results.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=error):
        validate_simulation_job(setup, job)


def test_mutated_checkpoint_invalidates_existing_simulations(tmp_path):
    setup, job, _payload = saved_rollout(tmp_path)
    Path(setup["checkpoints"]["s_packet"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        validate_simulation_job(setup, job)


def test_resume_reuses_published_timeout_and_seals_all_artifacts(tmp_path):
    setup, job, _payload = saved_rollout(tmp_path)
    result = prepare_simulation_job(setup, job)
    assert result["reuse"] is True
    assert result["result"]["statuses"] == {"timeout": 1}
    assert str(Path(job["output"]) / "rollout.mp4") in result["result"]["files"]
    assert simulation_report_links(setup) == [{
        "label": "s_packet: VideoRepick TREE episode 0",
        "path": "rollouts/s_packet/VideoRepick/tree/episode_00/index.html",
    }]


def test_unpublished_interruption_is_preserved_before_same_seed_retry(tmp_path):
    setup = plan(tmp_path)
    job = simulation_jobs(setup)[1]
    directory = Path(job["output"])
    directory.mkdir(parents=True)
    (directory / "interrupted.mp4").write_bytes(b"partial")
    result = prepare_simulation_job(setup, job)
    assert not result["reuse"]
    assert not directory.exists()
    assert (Path(result["preserved_interruption"]) / "interrupted.mp4").read_bytes() == b"partial"


def test_resume_never_retries_recorded_infrastructure_error(tmp_path):
    setup, job, payload = saved_rollout(tmp_path)
    payload["trials"][0]["status"] = "infrastructure_error"
    path = Path(job["output"]) / "results.json"
    path.write_text(json.dumps(payload))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="unresolved"):
        prepare_simulation_job(setup, job)
    assert path.read_bytes() == original
    assert not (Path(setup["output"]) / "interrupted_simulations").exists()


def test_resume_recovers_published_episode_when_aggregate_write_was_interrupted(tmp_path):
    setup, job, payload = saved_rollout(tmp_path)
    directory = Path(job["output"])
    episode = directory / "VideoRepick_00"
    episode.mkdir()
    (episode / "result.json").write_text(json.dumps(payload["trials"][0]))
    payload["trials"] = []
    (directory / "results.json").write_text(json.dumps(payload))
    recovered = prepare_simulation_job(setup, job)
    assert recovered["reuse"] and recovered["result"]["statuses"] == {"timeout": 1}
    assert len(json.loads((directory / "results.json").read_text())["trials"]) == 1


def test_each_rollout_job_is_one_episode_to_preserve_published_outcomes(tmp_path):
    setup = plan(tmp_path)
    setup["protocol"]["simulation_episodes"] = [0, 1]
    jobs = [job for job in simulation_jobs(setup) if job["stage"] == "rollouts"]
    assert len(jobs) == 4
    assert all(len(job["episodes"]) == 1 for job in jobs)
    assert len({job["output"] for job in jobs}) == 4


def test_handoff_jobs_are_bounded_explicit_and_have_one_shared_replay_control(tmp_path):
    setup = plan(tmp_path)
    setup["checkpoints"].update({label: dict(setup["checkpoints"]["s_packet"])
                                 for label in ("s_std", "s_stage")})
    jobs = [job for job in simulation_jobs(setup) if job["stage"] == "handoff"]
    assert len(jobs) == 37 and sum(job["expert_control"] for job in jobs) == 1
    control = next(job for job in jobs if job["expert_control"])
    assert control["task"] == "VideoUnmask" and control["episodes"] == [0]
    for label in setup["checkpoints"]:
        cases = [job for job in jobs if job["label"] == label and not job["expert_control"]]
        assert len(cases) == 12
        assert {(job["task"], job["episodes"][0]) for job in cases} == {("VideoUnmask", 0), ("PickXtimes", 0)}
        assert {job["prefix_actions"] for job in cases} == {0, 16, 32}
        assert {job["method"] for job in cases} == {"TREE", "FLAT"}
        assert all("--plan" in job["command"] and "--submit" not in job["command"] for job in cases)
    assert len({job["output"] for job in jobs}) == len(jobs)


def test_handoff_resume_reuses_verified_cap_and_preserves_recorded_errors(tmp_path, monkeypatch):
    from stmem.inspection import student_handoff
    setup = plan(tmp_path)
    job = next(job for job in simulation_jobs(setup) if job["stage"] == "handoff" and not job["expert_control"])
    directory = Path(job["output"])
    directory.mkdir(parents=True)
    (directory / "result.json").write_text('{"status":"stopped_at_declared_diagnostic_cap"}')
    (directory / "config.json").write_text('{}')
    (directory / "rollout.mp4").write_bytes(b"synthetic video artifact")
    def validate(_plan, _job):
        data = json.loads((directory / "result.json").read_text())
        if data["status"] == "diagnostic_error":
            raise ValueError("Handoff unresolved error")
        return {"outcome": data["status"], "status": "complete"}
    monkeypatch.setattr(student_handoff, "validate_result", validate)
    monkeypatch.setattr(student_handoff, "write_report", lambda output: (output / "index.html").write_text("fixture"))
    result = prepare_simulation_job(setup, job)
    assert result["reuse"] and result["result"]["outcome"] == "stopped_at_declared_diagnostic_cap"
    assert str(directory / "rollout.mp4") in result["result"]["files"]
    (directory / "result.json").write_text('{"status":"diagnostic_error"}')
    with pytest.raises(ValueError, match="unresolved"):
        prepare_simulation_job(setup, job)
    assert json.loads((directory / "result.json").read_text())["status"] == "diagnostic_error"
