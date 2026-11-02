"""Synthetic serialized adoption fixtures; no real training or benchmark score."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
import yaml

from stmem.config import load_config
from stmem.evaluation.aggregation import aggregate_shards
from stmem.evaluation.execution import evaluation_execution_config
from stmem.evaluation.metrics import TASKS, summarize_episodes
from stmem.evaluation.workload import planned_episodes, video_episodes
from stmem.models.policy import MemoryPolicy
from stmem.training.adopt_teacher import adopt_teacher
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.runtime import validate_selection

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def evidence(tmp_path):
    torch.set_num_threads(2)
    cfg = load_config(REPO / "configs/debug/teacher.yaml")
    # Small models and generated results exercise validation only. They are
    # deliberately not presented as actual real-data training/evaluation.
    cfg["debug"] = False
    cfg["data"].update(synthetic=False, tasks=list(TASKS), observation_stride=2,
                       process_every_observation=False)
    cfg["model"].update(horizon=16, execute=2, action_stride=2, proprioception_mode="none",
                        condition_on_previous_commands=False, condition_on_previous_joint_commands=False)
    cfg["evaluation"].update(episode_count=4, video_cap=0)
    policy = MemoryPolicy(cfg)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"stage": "teacher", "step": 104110, "synthetic": False,
                "config": cfg, "model": policy.state_dict()}, checkpoint)
    sha = file_sha256(checkpoint)
    effective, override = evaluation_execution_config(cfg, cfg, execute_actions=8)
    plan = planned_episodes(effective)
    manifest = {
        "schema_version": 1, "kind": "parallel_validation_evaluation", "config": effective,
        "source": {"source_sha256": "synthetic-test-source"}, "method": "FULL", "split": "val",
        "checkpoints": {str(checkpoint): sha}, "policy_checkpoint_sha256": sha,
        "planned": plan, "all_planned": plan, "num_shards": 2,
        "selection_family": None, "intervention": "none", "budget": 128,
        "targeting_diagnostics": False, "video_episodes": video_episodes(effective),
        "execution_override": override,
    }
    root = tmp_path / "evaluation"
    atomic_json(manifest, root / "manifest.json")
    (root / "evaluation.yaml").write_text(yaml.safe_dump(effective))
    for index in range(2):
        child = deepcopy(manifest)
        child["planned"] = plan[index::2]
        child["shard_index"] = index
        child["config"]["tracking"]["enabled"] = False
        shard = root / "shards" / f"shard{index:03d}"
        atomic_json(child, shard / "manifest.json")
        manifest_hash = file_sha256(shard / "manifest.json")
        for task, episode in child["planned"]:
            row = {
                "task": task, "episode_idx": episode, "method": "FULL", "split": "val",
                "training_seed": cfg["seed"], "checkpoint_sha256": sha,
                "noise_seed": cfg["evaluation"]["seed"] + list(TASKS).index(task)*1000 + episode,
                "evaluation_manifest_sha256": manifest_hash, "execution_override": override,
                "status": "success" if plan.index((task, episode)) < 20 else "fail",
            }
            atomic_json(row, shard / "episodes" / f"{task}_{episode:03d}.json")
        atomic_json({"status": "complete", "planned": len(child["planned"]),
                     "recorded": len(child["planned"]), "manifest_sha256": manifest_hash},
                    shard / "complete.json")
    aggregate_shards(root, read(root / "manifest.json"))
    atomic_json({"status": "complete", "planned": len(plan), "recorded": len(plan),
                 "manifest_sha256": file_sha256(root / "manifest.json"), "num_shards": 2},
                root / "complete.json")
    output = tmp_path / "phase2/seed0/teacher/selected.pt"
    return checkpoint, root, output


def test_acceptance_freezes_bytes_and_preserves_separate_rollout_override(evidence):
    checkpoint, evaluation, output = evidence
    original = {str(p.relative_to(evaluation)): file_sha256(p)
                for p in evaluation.rglob("*") if p.is_file()}
    assert adopt_teacher(checkpoint, evaluation, output) == output
    assert output.read_bytes() == checkpoint.read_bytes()
    validate_selection(output)
    record = read(output.with_suffix(".selection.json"))
    assert record["macro_success"] == 0.3125 and record["overall"]["successes"] == 20
    assert record["schema"] == "accepted_teacher_v1" and record["not_candidate_selection"]
    assert record["execution_override"]["execute_actions"] == 8
    assert record["evaluation_protocol"]["replan_primitive_steps"] == 16
    cfg = torch.load(output, weights_only=False)["config"]
    assert cfg["model"]["execute"] == 2 and cfg["model"]["proprioception_mode"] == "none"
    assert not cfg["model"]["condition_on_previous_commands"]
    assert cfg["data"]["observation_stride"] == 2
    snapshot = output.with_suffix(".evaluation")
    assert len(list(snapshot.glob("shards/*/episodes/*.json"))) == 64
    assert not list(snapshot.rglob("*.mp4"))
    assert original == {str(p.relative_to(evaluation)): file_sha256(p)
                        for p in evaluation.rglob("*") if p.is_file()}
    before = output.with_suffix(".selection.json").read_bytes()
    assert adopt_teacher(checkpoint, evaluation, output) == output
    assert output.with_suffix(".selection.json").read_bytes() == before


@pytest.mark.parametrize("file,field,value", [
    ("summary.json", "macro_success", 1.0),
    ("manifest.json", "method", "CURRENT"),
    ("manifest.json", "split", "test"),
    ("manifest.json", "intervention", "erase_execution"),
    ("manifest.json", "policy_checkpoint_sha256", "wrong-policy"),
    ("complete.json", "manifest_sha256", "wrong-manifest"),
    ("complete.json", "recorded", 63),
])
def test_rejects_mismatched_acceptance_evidence(evidence, file, field, value):
    checkpoint, evaluation, output = evidence
    data = read(evaluation / file)
    data[field] = value
    atomic_json(data, evaluation / file)
    with pytest.raises(ValueError):
        adopt_teacher(checkpoint, evaluation, output)
    assert not output.exists() and not output.with_suffix(".selection.json").exists()


def test_rejects_missing_episodes_and_mixed_shard_protocols(evidence):
    checkpoint, evaluation, output = evidence
    path = next(evaluation.glob("shards/*/episodes/*.json"))
    data = read(path)
    data["checkpoint_sha256"] = "different-model"
    atomic_json(data, path)
    with pytest.raises(ValueError, match="episode policy checkpoint"):
        adopt_teacher(checkpoint, evaluation, output)
    path.unlink()
    with pytest.raises(ValueError, match="completion marker"):
        adopt_teacher(checkpoint, evaluation, output)


def test_rejects_unexpected_empty_worker_directory(evidence):
    checkpoint, evaluation, output = evidence
    (evaluation / "shards/shard999").mkdir()
    with pytest.raises(ValueError, match="unexpected shard"):
        adopt_teacher(checkpoint, evaluation, output)


def test_rejects_errors_even_if_summary_and_completion_claim_success(evidence):
    checkpoint, evaluation, output = evidence
    path = next(evaluation.glob("shards/*/episodes/*.json"))
    data = read(path)
    data["status"] = "infrastructure_error"
    atomic_json(data, path)
    aggregate_shards(evaluation, read(evaluation / "manifest.json"))
    with pytest.raises(ValueError, match="without missing/error"):
        adopt_teacher(checkpoint, evaluation, output)


def test_rejects_missing_or_tampered_adopted_artifacts(evidence):
    checkpoint, evaluation, output = evidence
    adopt_teacher(checkpoint, evaluation, output)
    snapshot = output.with_suffix(".evaluation") / "summary.json"
    snapshot.write_text("{}")
    with pytest.raises(ValueError, match="missing or changed"):
        adopt_teacher(checkpoint, evaluation, output)


def test_interrupted_publication_recovers_only_identical_choice(evidence):
    checkpoint, evaluation, output = evidence
    adopt_teacher(checkpoint, evaluation, output)
    output.with_suffix(".selection.json").unlink()
    assert adopt_teacher(checkpoint, evaluation, output) == output
    output.with_suffix(".selection.json").unlink()
    output.write_bytes(b"unrelated checkpoint")
    with pytest.raises(FileExistsError, match="different checkpoint"):
        adopt_teacher(checkpoint, evaluation, output)


def test_input_mutation_during_copy_is_rejected(evidence, monkeypatch):
    from stmem.training import adopt_teacher as module

    checkpoint, evaluation, output = evidence
    original = module.shutil.copyfile

    def changed(source, target):
        result = original(source, target)
        if Path(source) == checkpoint:
            checkpoint.write_bytes(b"advanced last.pt")
        return result

    monkeypatch.setattr(module.shutil, "copyfile", changed)
    with pytest.raises(ValueError, match="checkpoint changed"):
        module.adopt_teacher(checkpoint, evaluation, output)
    assert not output.exists()


def test_serial_complete_evaluation_supported(evidence):
    checkpoint, evaluation, output = evidence
    manifest = read(evaluation / "manifest.json")
    del manifest["kind"]
    del manifest["policy_checkpoint_sha256"]
    manifest.update(num_shards=1, shard_index=0)
    root = evaluation.parent / "serial_evaluation"
    atomic_json(manifest, root / "manifest.json")
    records = []
    for path in evaluation.glob("shards/*/episodes/*.json"):
        row = read(path)
        row["evaluation_manifest_sha256"] = file_sha256(root / "manifest.json")
        records.append(row)
        atomic_json(row, root / "episodes" / path.name)
    atomic_json(summarize_episodes(records, planned_episodes(manifest["config"])), root / "summary.json")
    atomic_json({"status": "complete", "planned": 64, "recorded": 64,
                 "manifest_sha256": file_sha256(root / "manifest.json")}, root / "complete.json")
    assert adopt_teacher(checkpoint, root, output) == output


def test_fake_model_and_synthetic_teacher_rejected(evidence):
    checkpoint, evaluation, output = evidence
    # Keep the evaluation hash synchronized so model validation is the failing
    # condition; a failed checkpoint hash alone would not test model integrity.
    state = torch.load(checkpoint, weights_only=False)
    state["synthetic"] = True
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="real-data full-history"):
        adopt_teacher(checkpoint, evaluation, output)
