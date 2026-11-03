"""Synthetic protocol/result fixtures; no simulator or benchmark scores."""

from copy import deepcopy
import json

import pytest

from stmem.evaluation.metrics import summarize_episodes
from stmem.evaluation.selection import validate_selection_protocol
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.evaluation.select_compact_checkpoint import select


@pytest.mark.parametrize("method,family,router,stage,split", [
    ("FULL", "tree", None, "compact", "val"),
    ("TREE", "flat", None, "compact", "val"),
    ("TREE", "tree", "router.pt", "compact", "val"),
    ("TREE", "tree", None, "teacher", "val"),
    ("FLAT", "flat", None, "compact", "test"),
    ("TREE", None, None, "compact", "val"),
])
def test_selection_protocol_rejects_ambiguous_or_unmatched_methods(method, family, router, stage, split):
    with pytest.raises(ValueError):
        validate_selection_protocol(method, family, router, checkpoint_stage=stage, split=split)


@pytest.mark.parametrize("family", ["tree", "flat"])
def test_random_and_learned_routing_remain_distinct_valid_protocols(family):
    validate_selection_protocol(family.upper(), family, None, checkpoint_stage="compact", split="val")
    validate_selection_protocol(family.upper(), None, "router.pt", checkpoint_stage="compact", split="val")


def complete(directory, manifest, summary):
    atomic_json(manifest, directory / "manifest.json")
    atomic_json(summary, directory / "summary.json")
    atomic_json({"status": "complete", "manifest_sha256": file_sha256(directory / "manifest.json"),
                 "planned": len(manifest["planned"]), "recorded": len(manifest["planned"])},
                directory / "complete.json")


@pytest.fixture
def evidence(tmp_path):
    candidates, directories = [], []
    plan = [("PickXtimes", i) for i in range(4)]
    cfg = {"seed": 0, "data": {"tasks": ["PickXtimes"], "observation_stride": 2},
           "model": {"horizon": 16, "execute": 8, "action_stride": 2, "budget": 128},
           "geometry": {"fixture": "synthetic"}, "training": {"precision": "bf16"},
           "evaluation": {"seed": 1701, "episode_count": 4, "max_steps": 1300, "video_cap": 10}}
    for index, successes in enumerate(((2, 0), (1, 2))):
        checkpoint = tmp_path / f"candidate{index}.pt"
        checkpoint.write_bytes(f"synthetic serialized identity fixture {index}".encode())
        candidates.append(checkpoint)
        for family, count in zip(("tree", "flat"), successes):
            directory = tmp_path / f"candidate{index}_{family}"
            directory.mkdir()
            manifest = {"config": deepcopy(cfg), "split": "val", "selection_family": family,
                        "method": family.upper(), "checkpoints": {str(checkpoint): file_sha256(checkpoint)},
                        "source": {"source_sha256": "fixed"}, "budget": 128, "planned": plan,
                        "all_planned": plan, "intervention": "none",
                        "execution_override": {"execute_actions": 8}}
            summary = summarize_episodes([
                {"task": task, "episode_idx": i, "status": "success" if i < count else "timeout"}
                for task, i in plan
            ], plan)
            summary["worker_completion"] = {"planned": 4, "complete": 4}
            complete(directory, manifest, summary)
            directories.append(directory)
    return candidates, directories, tmp_path / "selected.pt"


def test_selection_averages_both_families_and_records_immutable_choice(evidence):
    candidates, directories, output = evidence
    assert select(candidates, directories, output) == output
    assert output.read_bytes() == candidates[1].read_bytes()
    record = json.loads(output.with_suffix(".selection.json").read_text())
    assert record["scores"] == [0.25, 0.375]
    assert select(candidates, directories, output) == output


@pytest.mark.parametrize("change", [
    "budget", "seed", "execute", "source", "family", "method", "router", "intervention",
    "workers", "partial", "checkpoint", "noise_seed", "stale_completion",
])
def test_selection_cannot_silently_compare_unmatched_or_incomplete_results(evidence, change):
    candidates, directories, output = evidence
    directory = directories[-1]
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if change == "budget":
        manifest["budget"] = 64
    elif change == "seed":
        manifest["planned"][0][1] = 20
        manifest["all_planned"] = manifest["planned"]
    elif change == "execute":
        manifest["execution_override"]["execute_actions"] = 4
    elif change == "source":
        manifest["source"]["source_sha256"] = "changed"
    elif change == "family":
        manifest.update(selection_family="tree", method="TREE")
    elif change == "method":
        manifest["method"] = "UNIFORM"
    elif change == "router":
        manifest["checkpoints"]["router.pt"] = "learned-router"
    elif change == "intervention":
        manifest["intervention"] = "erase_execution"
    elif change == "workers":
        summary["worker_completion"]["complete"] = 3
    elif change == "partial":
        summary["overall"]["recorded"] = 3
    elif change == "checkpoint":
        manifest["checkpoints"] = {"wrong.pt": "wrong"}
    elif change == "noise_seed":
        manifest["config"]["evaluation"]["seed"] = 9999
    complete(directory, manifest, summary)
    if change == "stale_completion":
        manifest["budget"] = 64
        atomic_json(manifest, directory / "manifest.json")
    with pytest.raises(ValueError):
        select(candidates, directories, output)
    assert not output.exists()


def test_tie_uses_earlier_predeclared_candidate(evidence):
    candidates, directories, output = evidence
    for index, directory in enumerate(directories[2:]):
        manifest = json.loads((directory / "manifest.json").read_text())
        summary = json.loads((directories[index] / "summary.json").read_text())
        complete(directory, manifest, summary)
    select(candidates, directories, output)
    assert output.read_bytes() == candidates[0].read_bytes()
