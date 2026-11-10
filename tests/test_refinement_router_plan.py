"""The conditional router blueprint cannot relax episode boundaries or scope."""
from copy import deepcopy
import json

import pytest
import yaml

from stmem.inspection.refinement_router_plan import BLOCKERS, validate_training_pool
from stmem.inspection.refinement_data import COMPACT_SHA, ENCODED_ID, HIERARCHY_ID, TEACHER_SHA
from stmem.training.checkpoints import file_sha256


def manifest():
    return {"training_pool": [{"task": "VideoUnmask", "episode_id": "episode_4"}],
            "diagnostic_episodes": [{"task": "VideoUnmask", "episode_id": "episode_0"}],
            "reserved_validation_episodes": [{"task": "VideoUnmask", "episode_id": "episode_8"}],
            "episode_partitions": [{"task": "VideoUnmask", "episode_id": "episode_0", "partition": "train"},
                                    {"task": "VideoUnmask", "episode_id": "episode_4", "partition": "train"},
                                    {"task": "VideoUnmask", "episode_id": "episode_8", "partition": "validation"}],
            "configuration": {"data": {"tasks": ["VideoUnmask"]}}}


def test_fitting_pool_cannot_reintroduce_diagnostic_or_router_validation_episodes():
    valid = manifest()
    result = validate_training_pool(valid)
    assert result["episodes"] == 1 and result["excluded_episodes"] == 2
    for episode in ("episode_0", "episode_8", "episode_unknown"):
        invalid = deepcopy(valid)
        invalid["training_pool"].append({"task": "VideoUnmask", "episode_id": episode})
        with pytest.raises(ValueError, match="diagnostic, reserved validation or unknown"):
            validate_training_pool(invalid)


def test_conditional_spec_is_bounded_disabled_and_separate_from_old_label_objectives():
    with open("configs/inspection/refinement_router_conditional.yaml") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["enabled"] is False
    assert cfg["maximum_students"] == 2
    assert cfg["fitting_states_per_student"] == 4096
    assert cfg["credit_targets"] == ["expert_first_move", "expert_completed_packet"]
    assert cfg["teacher_agreement"] == "diagnostic_only"
    assert cfg["fit"]["tracking_project"] == "chronotree-routers"
    assert cfg["closed_loop"]["split"] == "val"
    assert cfg["closed_loop"]["executed_targets"] == 10
    assert len(BLOCKERS) >= 4


def test_conditional_plan_uses_declared_100k_endpoints_and_rejects_unmatched_students(tmp_path, monkeypatch):
    from stmem.inspection import refinement_router_plan

    fixed = manifest()
    fixed.update(experiment_id='fixed-study', encoded_cache_id=ENCODED_ID, hierarchy_cache_id=HIERARCHY_ID,
                 dependencies={'teacher_sha256': TEACHER_SHA}, protocol={'student_training': {'steps': 100000}})
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(fixed))
    monkeypatch.setattr(refinement_router_plan, 'verify_manifest', lambda path: fixed)
    monkeypatch.setattr(refinement_router_plan, 'source_revision', lambda: {'source_sha256': 'synthetic'})
    states, checkpoints = {}, {}
    for variant in ('S_STD', 'S_STAGE'):
        path = tmp_path / f'{variant}.pt'
        path.write_bytes(b'synthetic-checkpoint-identity-only')
        checkpoints[variant] = path
        study = {'variant': variant, 'experiment_id': fixed['experiment_id'],
                 'manifest_sha256': file_sha256(manifest_path), 'teacher_sha256': TEACHER_SHA,
                 'encoded_cache_id': ENCODED_ID, 'hierarchy_cache_id': HIERARCHY_ID,
                 'initialization': 'accepted_teacher_weights', 'initialization_sha256': TEACHER_SHA,
                 'optimizer_initialization': 'fresh', 'primary_updates': 100000,
                 'development_steps': list(range(5000, 100000, 5000))}
        states[path] = {'stage': 'compact', 'step': 100000,
                        'provenance': {'dependencies': {'teacher': TEACHER_SHA}},
                        'config': {'training': {'utility_study': study, 'steps': 100000, 'schedule_steps': 100000}}}
    monkeypatch.setattr(refinement_router_plan, 'load_checkpoint', lambda path: states[path])
    result = refinement_router_plan.prepare(manifest_path, tmp_path / 'plan.json', checkpoints=checkpoints)
    assert all(row['endpoint'] == row['step'] == 100000 for row in result['students'].values())
    assert result['enabled'] is False
    states[checkpoints['S_STAGE']]['step'] = 30000
    with pytest.raises(ValueError, match='predeclared primary endpoint'):
        refinement_router_plan.prepare(manifest_path, tmp_path / 'bad.json', checkpoints=checkpoints)
    other = states[checkpoints['S_STAGE']]
    other['step'] = 10000
    other['config']['training']['utility_study'].update(
        initialization='S0_weights', initialization_sha256=COMPACT_SHA, continuation_updates=10000)
    with pytest.raises(ValueError, match='share initialization and declared training duration'):
        refinement_router_plan.prepare(manifest_path, tmp_path / 'bad.json', checkpoints=checkpoints)
