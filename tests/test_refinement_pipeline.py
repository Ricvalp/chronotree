"""Synthetic CPU orchestration checks; no real training or Slurm submission."""
# ruff: noqa: F811
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_cached_context_loader import fixture  # noqa: F401
from test_packed_context import indexed  # noqa: F401
from stmem.config import digest
from stmem.inspection.refinement_data import (
    COMPACT_SHA, ENCODED_ID, HIERARCHY_ID, TEACHER_SHA,
    UtilityDataReader, seal_manifest, select_cached_diagnostic_states, select_states, verify_manifest,
)
from stmem.inspection.refinement_utility import (
    choose_smoke, load_rows, measurement_directory, normalized_panel, read_panel,
    save_rows, validate_student_endpoint,
)
from stmem.training.checkpoints import file_sha256


def test_fixed_state_panel_preserves_episode_partition_and_full_available_horizon(indexed):
    _, data, _, _, _ = indexed
    first = select_states(data, seed=86, episodes_per_partition=1)
    assert first == select_states(data, seed=86, episodes_per_partition=1)
    held = {(row['task'], row['episode_id']) for row in first['diagnostic_episodes']}
    assert not held & {(row['task'], row['episode_id']) for row in first['training_pool']}
    assert all(row['valid_coordinates'] == 16 for row in first['states'])
    pairs = {}
    for row in first['states']:
        pairs.setdefault((row['task'], row['episode_id']), []).append(row['decision_index'])
    assert all(len(indices) == 2 and abs(indices[0] - indices[1]) >= data.stride for indices in pairs.values())
    for state in first['states']:
        assert state['partition'] == data.label_partition(state['episode_index'])


def test_asset_limited_panel_is_separate_and_has_no_optimization_pool(indexed):
    cfg, data, _, cache, hierarchy = indexed
    reader = UtilityDataReader(data, cfg, cache.root, hierarchy)
    try:
        panel = select_cached_diagnostic_states(data, reader, seed=1)
        assert panel['training_pool'] == []
        assert len(panel['states']) == 8
        assert not panel['required_multitask_panel_satisfied']  # only two synthetic tasks
        assert all(state['diagnostic_asset_selection'] for state in panel['states'])
        for state in panel['states']:
            decision, _, archive = reader.archive(state)
            contexts = {name: archive.coarse(name) for name in ('TREE', 'FLAT')}
            item, hydrated = reader.prepare(state, contexts)
            assert item['valid'].all() and decision.info.index == state['decision_index']
            for name, bank in hydrated.items():
                assert bank.content.shape[1] == archive.records_count(contexts[name])
    finally:
        reader.close()


def endpoint_fixture(tmp_path):
    manifest = {'experiment_id': 'fixed-experiment', 'encoded_cache_id': ENCODED_ID,
                'hierarchy_cache_id': HIERARCHY_ID}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    study = {'variant': 'S_STAGE', 'experiment_id': manifest['experiment_id'],
             'manifest_sha256': file_sha256(path), 'teacher_sha256': TEACHER_SHA,
             'encoded_cache_id': ENCODED_ID, 'hierarchy_cache_id': HIERARCHY_ID,
             'initialization': 'accepted_teacher_weights', 'initialization_sha256': TEACHER_SHA,
             'optimizer_initialization': 'fresh'}
    state = {'step': 30000, 'config': {'training': {'utility_study': study}},
             'provenance': {'dependencies': {'teacher': TEACHER_SHA}}}
    return manifest, path, state


def test_endpoint_binding_rejects_wrong_student_panel_and_confirmation_reuse(tmp_path):
    manifest, path, state = endpoint_fixture(tmp_path)
    assert validate_student_endpoint(manifest, path, 'S_STAGE', state, 'validation')['endpoint'] == 30000
    with pytest.raises(ValueError, match='variant'):
        validate_student_endpoint(manifest, path, 'S_PACKET', state, 'validation')
    for key in ('experiment_id', 'manifest_sha256', 'teacher_sha256', 'hierarchy_cache_id'):
        changed = deepcopy(state)
        changed['config']['training']['utility_study'][key] = 'changed'
        with pytest.raises(ValueError, match=key):
            validate_student_endpoint(manifest, path, 'S_STAGE', changed, 'train')
    state['step'] = 15000
    assert validate_student_endpoint(manifest, path, 'S_STAGE', state, 'train')['endpoint'] == 30000
    with pytest.raises(ValueError, match='development-only'):
        validate_student_endpoint(manifest, path, 'S_STAGE', state, 'both')
    assert measurement_directory(tmp_path, 'S_STAGE', 15000, 'main', 'train') != measurement_directory(
        tmp_path, 'S_STAGE', 30000, 'main', 'validation')


def test_extended_student_schedule_reserves_confirmation_for_declared_100k_endpoint(tmp_path):
    manifest, path, state = endpoint_fixture(tmp_path)
    manifest['protocol'] = {'student_training': {'steps': 100000, 'checkpoint_every': 1000,
                                                'snapshot_every': 5000}}
    path.write_text(json.dumps(manifest))
    training = state['config']['training']
    training.update(steps=100000, schedule_steps=100000)
    study = training['utility_study']
    study.update(primary_updates=100000, development_steps=list(range(5000, 100000, 5000)),
                 manifest_sha256=file_sha256(path))
    state['step'] = 100000
    assert validate_student_endpoint(manifest, path, 'S_STAGE', state, 'validation')['endpoint'] == 100000
    for step in (5000, 15000, 30000, 95000):
        state['step'] = step
        assert validate_student_endpoint(manifest, path, 'S_STAGE', state, 'train')['endpoint'] == 100000
        with pytest.raises(ValueError, match='development-only'):
            validate_student_endpoint(manifest, path, 'S_STAGE', state, 'validation')
    for step in (1000, 99999, 105000):
        state['step'] = step
        with pytest.raises(ValueError, match='predeclared'):
            validate_student_endpoint(manifest, path, 'S_STAGE', state, 'train')
    state['step'] = 100000
    training['schedule_steps'] = 30000
    with pytest.raises(ValueError, match='cosine schedule'):
        validate_student_endpoint(manifest, path, 'S_STAGE', state, 'validation')
    training['schedule_steps'] = 100000
    study.pop('development_steps')
    with pytest.raises(ValueError, match='both'):
        validate_student_endpoint(manifest, path, 'S_STAGE', state, 'validation')
    study.pop('primary_updates')
    with pytest.raises(ValueError, match='declared training duration'):
        validate_student_endpoint(manifest, path, 'S_STAGE', state, 'validation')


@pytest.mark.parametrize(('steps', 'expected'), [(100000, 300000), (None, 90000)])
def test_context_planning_counts_use_declared_training_duration(tmp_path, monkeypatch, steps, expected):
    from stmem.inspection import refinement_utility
    protocol = {'discovery_draws': 4, 'confirmation_draws': 4, 'max_generated_cases': 16}
    if steps is not None:
        protocol['student_training'] = {'steps': steps}
    manifest = {'experiment_id': 'synthetic', 'dependencies': {'encoded': 'unused', 'hierarchy': 'unused'},
                'configuration': {}, 'protocol': protocol, 'states': [], 'capability_states': []}
    monkeypatch.setattr(refinement_utility, 'verify_manifest', lambda *args: manifest)
    monkeypatch.setattr(refinement_utility, 'TrainingData', lambda *args: None)
    monkeypatch.setattr(refinement_utility, 'UtilityDataReader', lambda *args: SimpleNamespace(close=lambda: None))
    result = refinement_utility.build_contexts(tmp_path / 'manifest.json')
    assert result['primary_training_updates'] == expected
    assert result['primary_training_example_exposures'] == expected * 128


def test_continuation_is_not_silently_relabelled_as_teacher_initialized(tmp_path):
    manifest, path, state = endpoint_fixture(tmp_path)
    study = state['config']['training']['utility_study']
    study.update(initialization='S0_weights', initialization_sha256=COMPACT_SHA, continuation_updates=10000)
    state['step'] = 10000
    assert validate_student_endpoint(manifest, path, 'S_STAGE', state, 'train')['initialization'] == 'S0_weights'
    state['step'] = 30000
    with pytest.raises(ValueError, match='predeclared'):
        validate_student_endpoint(manifest, path, 'S_STAGE', state, 'train')


def test_rows_and_panel_identity_survive_reload_and_reject_tampering(tmp_path):
    path = tmp_path / 'state.npz'
    rows = [{'context_id': 'ctx', 'bank': 'discovery', 'draw_index': 0, 'expert_full': .2}]
    save_rows(path, rows, {'identity': {'experiment': 'fixed'}})
    actual, metadata = load_rows(path)
    assert actual == rows and metadata['rows'] == 1
    path.write_bytes(path.read_bytes()[:-8] + b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        load_rows(path)
    (tmp_path / 'contexts').mkdir()
    panel = {'experiment_id': 'fixed', 'state_id': 'state', 'contexts': {}}
    panel['panel_sha256'] = digest(panel)
    panel_path = tmp_path / 'contexts/state.json'
    panel_path.write_text(json.dumps(panel))
    assert read_panel(tmp_path, 'state', 'fixed') == panel
    panel['state_id'] = 'other'
    panel_path.write_text(json.dumps(panel))
    with pytest.raises(ValueError, match='changed'):
        read_panel(tmp_path, 'state', 'fixed')


def test_verified_shard_modifications_reject_resume_even_with_unchanged_manifest(tmp_path):
    dependencies = {}
    for key in ('encoded', 'hierarchy'):
        root = tmp_path / key
        root.mkdir()
        (root / 'manifest.json').write_text('{}')
        dependencies[key] = str(root)
        dependencies[key + '_manifest_sha256'] = file_sha256(root / 'manifest.json')
    shard = tmp_path / 'encoded/data.h5'
    shard.write_bytes(b'12345')
    dependencies['verified_shards'] = {str(shard): {'sha256': file_sha256(shard),
                                                   'bytes': 5, 'mtime_ns': shard.stat().st_mtime_ns}}
    manifest = seal_manifest({'dependencies': dependencies})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    verify_manifest(path, verify_source=False)
    shard.write_bytes(b'54321')
    with pytest.raises(ValueError, match='content changed'):
        verify_manifest(path, verify_source=False)


def test_smoke_selection_retains_sources_and_partial_block_before_generic_filling():
    states = [{'task': 'A', 'episode_id': str(i), 'execution_history_frames': 16, 'conditioning_frames': 0}
              for i in range(12)]
    states += [{'task': 'B', 'episode_id': '1', 'execution_history_frames': 17, 'conditioning_frames': 3},
               {'task': 'C', 'episode_id': '2', 'execution_history_frames': 32, 'conditioning_frames': 0,
                'structural_action_mask': [True] * 7 + [False]}]
    chosen = choose_smoke(states, 8)
    assert len(chosen) == 8 and {state['task'] for state in chosen} == {'A', 'B', 'C'}
    assert any(state['execution_history_frames'] % 8 and state['conditioning_frames'] for state in chosen)
    assert any(not all(state.get('structural_action_mask', [True])) for state in chosen)


def test_strong_paired_flag_survives_report_normalization():
    panel = {'families': [{'reader': 'TREE', 'panel': 'packet', 'background': 56, 'final_budget': 64,
                           'paired': True, 'base': 'base', 'controls': {},
                           'candidates': [{'candidate_id': 'one', 'prefixes': {'1': 'a', '8': 'b'},
                                           'source': 'conditioning', 'depth': 3, 'support_count': 16}]}]}
    group = normalized_panel(panel)['groups'][0]
    assert group['paired'] and group['candidates'][0]['source'] == 'conditioning'


def test_hpc_dry_run_requests_untyped_gpu_and_pins_endpoint_without_slurm(tmp_path, monkeypatch):
    module_path = Path(__file__).parents[1] / 'hpc/refinement_utility.py'
    spec = importlib.util.spec_from_file_location('utility_hpc_test_launcher', module_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    repo = tmp_path / 'repo'
    (repo / 'src').mkdir(parents=True)
    source = repo / 'checkpoint.pt'
    source.write_bytes(b'explicit synthetic fixture')
    manifest_path = repo / 'manifest.json'
    manifest_path.write_text('{}')
    monkeypatch.setattr(launcher, 'read_site', lambda _: {'STMEM_REPO_ROOT': str(repo), 'STMEM_PYTHON': 'python'})
    monkeypatch.setattr(launcher, 'verify_manifest', lambda _: {
        'dependencies': {'compact': str(source)}, 'protocol': {'student_training': {'steps': 100000}}})
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[:2] == ['bash', '-n']
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launcher.subprocess, 'run', run)
    launcher.main(['--stage', 'measure', '--manifest', str(manifest_path), '--name', 'fixture', '--dry-run'])
    plans = list((repo / 'hpc/generated/fixture').glob('*.json'))
    assert len(plans) == 1
    plan = json.loads(plans[0].read_text())
    script = Path(plan['script']).read_text()
    assert '#SBATCH --gres=gpu:1' in script and '#SBATCH --array=0-3%4' in script
    assert '--exclusive' not in script and 'gpu:h200' not in script
    assert 'unset STMEM_DATA_ROOT STMEM_CACHE_ROOT STMEM_RUN_ROOT' in script
    assert file_sha256(source)[:12] in Path(plan['script']).name
    assert plan['commands'][0][-4:-2] == ['--workers', '4']
    for student in ('S_STD', 'S_STAGE', 'S_PACKET'):
        assert plan['dependency_graph'][student][-1] == 'training_100000'
    assert len(calls) == 1


def test_separate_reference_pass_counts_compute_once_and_preserves_peak_memory():
    from stmem.inspection.refinement_utility import combine_reference_timings, empty_measurement_timings
    compact = empty_measurement_timings()
    compact.update(compact_model_examples=384, wall_seconds=2., identity_seconds=.1,
                   gpu_peak_allocated_bytes=100)
    reference = {'teacher_model_examples': 8, 'teacher_context_seconds': .2,
                 'teacher_forward_seconds': .3, 'wall_seconds': .7, 'identity_seconds': .05,
                 'gpu_peak_allocated_bytes': 150, 'compact_model_examples': 0}
    result = combine_reference_timings(compact, reference)
    assert result['teacher_model_examples'] == 8 and result['compact_model_examples'] == 384
    assert result['wall_seconds'] == 2.7 and result['gpu_peak_allocated_bytes'] == 150
    assert result['identity_seconds'] == pytest.approx(.15)
    reused = combine_reference_timings(compact, {})
    assert reused['teacher_model_examples'] == 0 and reused['wall_seconds'] == 2.
    assert all(value == 0 for value in empty_measurement_timings().values())


def test_saved_results_report_verifies_manifest_seal_without_remote_cache_access(tmp_path):
    manifest = seal_manifest({'dependencies': {'encoded': '/unavailable/hpc/cache',
                                               'hierarchy': '/unavailable/hpc/hierarchy'}})
    path = tmp_path / 'transferred_manifest.json'
    path.write_text(json.dumps(manifest))
    assert verify_manifest(path, verify_source=False, verify_dependencies=False) == manifest
    with pytest.raises(FileNotFoundError):
        verify_manifest(path, verify_source=False)
    manifest['dependencies']['encoded'] = '/tampered'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='content hash'):
        verify_manifest(path, verify_source=False, verify_dependencies=False)


@pytest.mark.parametrize(('initialization', 'endpoint', 'steps'), [
    ('accepted_teacher_weights', 30000, (15000, 30000)),
    ('accepted_teacher_weights', 100000, (15000, 30000, 95000, 100000)),
    ('S0_weights', 10000, (10000,)),
])
def test_report_uses_declared_endpoint_and_rejects_mixed_initializations(
    tmp_path, monkeypatch, initialization, endpoint, steps,
):
    from stmem.inspection import refinement_report, refinement_utility

    state = {'state_id': 'state', 'task': 'T', 'episode_id': 'episode_0', 'partition': 'train'}
    manifest = seal_manifest({'kind': 'synthetic_pipeline_test', 'states': [state],
                              'protocol': {'gain_epsilon': 1e-6}})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    panel = {'state_id': 'state', 'experiment_id': manifest['experiment_id'], 'families': [],
             'contexts': {}, 'capability': {}}
    panel['panel_sha256'] = digest(panel)
    (tmp_path / 'contexts').mkdir()
    (tmp_path / 'contexts/state.json').write_text(json.dumps(panel))
    monkeypatch.setattr(refinement_report, 'write_report', lambda *args: None)

    def add_measurement(step, origin, declared_endpoint):
        identity = {'experiment_id': manifest['experiment_id'], 'student': 'S_STD', 'step': step,
                    'initialization': origin, 'initialization_sha256': origin,
                    'training_manifest_sha256': file_sha256(path), 'endpoint': declared_endpoint,
                    'mode': 'main', 'precision': 'fp32'}
        target = measurement_directory(tmp_path, 'S_STD', step, 'main', 'train') / 'state.npz'
        save_rows(target, [], {'identity': identity, 'state': state, 'panel_sha256': panel['panel_sha256'],
                              'timings': {}, 'numerical_controls': {}, 'mask': {'terminal_padding': False}})

    for step in steps:
        add_measurement(step, initialization, endpoint)
    result = refinement_utility.report(path)
    inventory = {row['student']: row for row in result['inventory']}
    assert inventory['S_STD']['step'] == inventory['S_STD']['declared_endpoint'] == endpoint
    assert inventory['S_STD']['initialization'] == initialization
    assert inventory['S_STD']['primary_endpoint'] and inventory['S_STD']['status'] == 'complete'
    assert not inventory['S_STAGE']['primary_endpoint']
    analyses = json.loads((tmp_path / 'analysis.json').read_text())['analyses']
    assert len(analyses) == 1 and analyses[0]['identity']['step'] == endpoint
    assert analyses[0]['mask'] == {'terminal_padding': False}
    add_measurement(20000, initialization, 20000)
    with pytest.raises(ValueError, match='Mixed initializations'):
        refinement_utility.report(path)
