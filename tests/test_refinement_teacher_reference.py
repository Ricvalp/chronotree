"""CPU-only identity checks for a separately adopted teacher utility study."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from stmem.config import digest
from stmem.inspection import refinement_data as data
from stmem.training.checkpoints import file_sha256
from stmem.training.utility_study import read_study_manifest


def teacher_manifest():
    dependencies = {
        "teacher_sha256": "a" * 64, "compact_sha256": "a" * 64,
        "teacher_step": 140000, "compact_step": 140000,
        "encoded_cache_id": "b" * 24, "hierarchy_cache_id": "c" * 24,
        "accepted_cache_plan": "/accepted/plan.json", "accepted_cache_plan_sha256": "d" * 64,
        "teacher_selection": "/accepted/selected.selection.json", "teacher_selection_sha256": "e" * 64,
        "reference": deepcopy(data.TEACHER_REFERENCE),
    }
    return {
        **{key: dependencies[key] for key in ("teacher_sha256", "compact_sha256", "encoded_cache_id", "hierarchy_cache_id")},
        "dependencies": dependencies, "reference": deepcopy(data.TEACHER_REFERENCE),
        "primary_endpoints": list(data.NEW_STUDENTS),
        "protocol": {"primary_endpoints": list(data.NEW_STUDENTS),
                     "accepted_teacher": {"step": 140000, "sha256": "a" * 64}},
        "training_pool": [{"task": "A", "episode_id": "episode_0"}],
        "diagnostic_episodes": [{"task": "A", "episode_id": "episode_1"}],
        "reserved_validation_episodes": [{"task": "A", "episode_id": "episode_2"}],
        "episode_partitions": [
            {"task": "A", "episode_id": "episode_0", "partition": "train"},
            {"task": "A", "episode_id": "episode_1", "partition": "train"},
            {"task": "A", "episode_id": "episode_2", "partition": "validation"},
        ],
    }


def test_new_study_is_separately_pinned_and_excludes_old_s0(tmp_path):
    manifest = teacher_manifest()
    assert data.primary_students(manifest) == data.NEW_STUDENTS
    assert data.reference_name(manifest) == "S_INIT"
    assert data.teacher_identity(manifest) == ("a" * 64, 140000)
    assert data.reference_identity(manifest) == ("a" * 64, 140000)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data.seal_manifest(manifest)))
    assert read_study_manifest(path)["dependencies"]["teacher_step"] == 140000
    changed = deepcopy(manifest)
    changed["compact_sha256"] = changed["dependencies"]["compact_sha256"] = data.COMPACT_SHA
    changed["dependencies"]["compact_step"] = 90000
    with pytest.raises(ValueError, match="older compact"):
        data.study_dependencies(changed)
    changed = deepcopy(manifest)
    changed["primary_endpoints"].insert(0, "S0")
    with pytest.raises(ValueError, match="Primary students"):
        data.study_dependencies(changed)


def test_new_pins_require_explicit_contract_and_complete_adoption_receipts():
    manifest = teacher_manifest()
    del manifest["dependencies"]["accepted_cache_plan_sha256"]
    with pytest.raises(ValueError, match="missing dependency"):
        data.study_dependencies(manifest)
    manifest = teacher_manifest()
    del manifest["reference"]
    del manifest["dependencies"]["reference"]
    with pytest.raises(ValueError, match="Legacy student study"):
        data.study_dependencies(manifest)
    assert data.teacher_identity({}) == (data.TEACHER_SHA, 104110)
    assert data.reference_identity({}) == (data.COMPACT_SHA, 90000)
    assert data.primary_students({}) == ("S0", *data.NEW_STUDENTS)


def test_accepted_teacher_protocol_rejects_wrong_saved_step_and_hash(tmp_path):
    manifest = teacher_manifest()
    data.validate_protocol_teacher(manifest["protocol"], manifest["dependencies"])
    for key, value in (("step", 160000), ("sha256", "f" * 64)):
        protocol = deepcopy(manifest["protocol"])
        protocol["accepted_teacher"][key] = value
        with pytest.raises(ValueError, match="accepted_teacher"):
            data.validate_protocol_teacher(protocol, manifest["dependencies"])
        changed = deepcopy(manifest)
        changed["protocol"] = protocol
        path = tmp_path / "wrong_protocol.json"
        path.write_text(json.dumps(data.seal_manifest(changed)))
        with pytest.raises(ValueError, match="accepted_teacher"):
            read_study_manifest(path)


@pytest.fixture
def accepted_cache(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    from hpc import train_compact

    teacher = tmp_path / "selected.pt"
    teacher.write_bytes(b"synthetic checkpoint identity fixture")
    teacher_sha = file_sha256(teacher)
    identity = {"format": "synthetic_identity_fixture", "teacher_sha256": teacher_sha,
                "geometry": {}, "model_encoder": {}, "normalization_version": "test",
                "frozen_encoder_precision": "fp32", "prefix_convention": "test", "code": "test",
                "data_identity": "test"}
    encoded = tmp_path / "cache" / digest(identity)[:24]
    hierarchy = encoded / "hierarchy" / "test"
    hierarchy.mkdir(parents=True)
    leaf = {"identity": identity, "shards": {}}
    parents = {"identity": {"leaf": digest(identity)}, "shards": {}}
    (encoded / "manifest.json").write_text(json.dumps(leaf))
    (hierarchy / "manifest.json").write_text(json.dumps(parents))
    cfg = {"model": {"horizon": 16, "action_stride": 2, "proprioception_mode": "none",
                     "condition_on_previous_commands": False},
           "data": {"observation_stride": 2}, "paths": {"cache_root": str(tmp_path / "cache")}}
    selection = tmp_path / "selected.selection.json"
    selection.write_text(json.dumps({"schema": "accepted_teacher_v1", "sha256": teacher_sha, "step": 140000}))
    plan = {"checkpoint": str(teacher), "checkpoint_sha256": teacher_sha, "checkpoint_step": 140000,
            "selection": str(selection), "selection_sha256": file_sha256(selection)}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(train_compact, "inspect_cache", lambda _: (cfg, plan, leaf))
    monkeypatch.setattr(data, "load_checkpoint", lambda _: {"step": 140000, "stage": "teacher", "config": cfg})
    monkeypatch.setattr(data, "validate_policy_config", lambda *_: None)
    monkeypatch.setattr(data, "cache_identity", lambda *_: deepcopy(identity))
    monkeypatch.setattr(data, "validate_hierarchy_cache", lambda *_: parents)
    return cfg, teacher, encoded, hierarchy, plan_path


def test_new_teacher_dependencies_require_trusted_cache_and_exact_reference(accepted_cache, tmp_path):
    cfg, teacher, encoded, hierarchy, plan_path = accepted_cache
    result = data.verify_dependencies(cfg, teacher, teacher, encoded, hierarchy,
                                      teacher_reference=True, cache_plan=plan_path)
    assert result["teacher_sha256"] == result["compact_sha256"] == file_sha256(teacher)
    assert result["teacher_step"] == result["compact_step"] == 140000
    assert result["reference"] == data.TEACHER_REFERENCE
    with pytest.raises(ValueError, match="legacy"):
        data.verify_dependencies(cfg, teacher, teacher, encoded, hierarchy)
    with pytest.raises(ValueError, match="adopted-cache"):
        data.verify_dependencies(cfg, teacher, teacher, encoded, hierarchy, teacher_reference=True)
    old_compact = tmp_path / "old_compact.pt"
    old_compact.write_bytes(b"old student weights")
    with pytest.raises(ValueError, match="inputs differ"):
        data.verify_dependencies(cfg, teacher, old_compact, encoded, hierarchy,
                                 teacher_reference=True, cache_plan=plan_path)
    with pytest.raises(ValueError, match="not the one verified"):
        data.verify_dependencies(cfg, teacher, teacher, tmp_path / "wrong_cache", hierarchy,
                                 teacher_reference=True, cache_plan=plan_path)


def test_new_teacher_dependencies_reject_incompatible_representation(accepted_cache, monkeypatch):
    cfg, teacher, encoded, hierarchy, plan_path = accepted_cache
    expected = data.cache_identity(cfg, teacher, "test")
    expected["teacher_sha256"] = "f" * 64
    monkeypatch.setattr(data, "cache_identity", lambda *_: expected)
    with pytest.raises(ValueError, match="Frozen cache representation"):
        data.verify_dependencies(cfg, teacher, teacher, encoded, hierarchy,
                                 teacher_reference=True, cache_plan=plan_path)


def test_reference_receipts_are_rechecked_without_reencoding(tmp_path):
    manifest = teacher_manifest()
    for key in ("accepted_cache_plan", "teacher_selection"):
        path = tmp_path / (key + ".json")
        path.write_text("{}")
        manifest["dependencies"][key] = str(path)
        manifest["dependencies"][key + "_sha256"] = file_sha256(path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data.seal_manifest(manifest)))
    assert data.verify_manifest(path, verify_source=False, verify_dependencies=False)["reference"]["name"] == "S_INIT"
    Path(manifest["dependencies"]["teacher_selection"]).write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Accepted teacher dependency changed"):
        data.verify_manifest(path, verify_source=False)


def test_new_teacher_launcher_pins_prepare_and_limits_reference_to_smoke(accepted_cache, tmp_path, monkeypatch):
    cfg, teacher, encoded, hierarchy, plan_path = accepted_cache
    spec = importlib.util.spec_from_file_location(
        "new_teacher_utility_launcher", Path(__file__).parents[1] / "hpc/refinement_utility.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    cache_plan_path = repo / "hpc/generated/cache140k/plan.json"
    cache_plan_path.parent.mkdir(parents=True)
    cache_plan_path.write_bytes(plan_path.read_bytes())
    plan = json.loads(plan_path.read_text())
    leaf = json.loads((encoded / "manifest.json").read_text())
    parents = json.loads((hierarchy / "manifest.json").read_text())
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(yaml.safe_dump({
        "primary_endpoints": list(data.NEW_STUDENTS),
        "accepted_teacher": {"step": 140000, "sha256": file_sha256(teacher)},
        "student_training": {"steps": 100000},
    }))
    monkeypatch.setattr(launcher, "read_site", lambda _: {"STMEM_REPO_ROOT": str(repo), "STMEM_PYTHON": "python"})
    monkeypatch.setattr(launcher, "inspect_cache", lambda _: (cfg, plan, leaf))
    monkeypatch.setattr(launcher, "inspect_hierarchy", lambda *_: parents)
    monkeypatch.setattr(launcher, "default_hierarchy_root", lambda *_: hierarchy)
    calls = []

    def shell_check(command, **kwargs):
        calls.append(command)
        assert command[:2] == ["bash", "-n"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", shell_check)
    launcher.main(["--name", "teacher140k", "--cache-name", "cache140k", "--teacher-reference",
                   "--protocol", str(protocol), "--stage", "prepare", "--dry-run"])
    prepared = json.loads((repo / "hpc/generated/teacher140k/prepare_S_INIT_train.json").read_text())
    command = prepared["commands"][0]
    assert command[-3:] == ["--teacher-reference", "--cache-plan", str(cache_plan_path)]
    assert command[command.index("--compact") + 1] == str(teacher)
    assert prepared["inputs"][str(cache_plan_path)] == file_sha256(cache_plan_path)
    assert "S0" not in prepared["dependency_graph"]
    manifest_path = repo / "runs/teacher140k/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = teacher_manifest()
    manifest["dependencies"]["compact"] = str(teacher)
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(launcher, "verify_manifest", lambda *_args, **_kwargs: manifest)
    launcher.main(["--name", "teacher140k", "--stage", "smoke", "--dry-run"])
    smoke_plan = json.loads(next((repo / "hpc/generated/teacher140k").glob("smoke_*.json")).read_text())
    assert smoke_plan["student"] == "S_INIT"
    assert "--gres=gpu:1" in Path(smoke_plan["script"]).read_text()
    for extra in (["--stage", "measure"], ["--stage", "smoke", "--student", "S0"],
                  ["--stage", "smoke", "--panel-partition", "validation"]):
        with pytest.raises(SystemExit):
            launcher.main(["--name", "teacher140k", *extra, "--dry-run"])
    assert len(calls) == 2
