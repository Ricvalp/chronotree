"""CPU-only repair proof and immutable-result recovery invariants."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import pytest

from stmem.config import digest
from stmem.training.checkpoints import atomic_json

spec = importlib.util.spec_from_file_location('recover_student_handoffs', Path('scripts/recover_student_handoffs.py'))
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def source_fixture(tmp_path, monkeypatch):
    folder = tmp_path / 'src/stmem/inspection'
    folder.mkdir(parents=True)
    validator = folder / 'handoff_validation.py'
    validator.write_text('fixed prefix\n' + recovery.NEW_BLOCK + 'fixed suffix\n')
    extra = folder / 'unrelated.py'
    extra.write_text('unchanged inference code\n')
    (tmp_path / 'scripts').mkdir()
    for name in ('inspect_students.py', 'visualize_reset_cube_chunks.py'):
        (tmp_path / 'scripts' / name).write_text('fixed script\n')
    monkeypatch.setattr(recovery.suite, 'REPO', tmp_path)
    monkeypatch.setattr(recovery, 'VALIDATOR', validator)
    h = hashlib.sha256()
    for path in sorted([*folder.glob('*.py'), *tmp_path.glob('scripts/*.py')]):
        raw = path.read_text().replace(recovery.NEW_BLOCK, recovery.OLD_BLOCK)
        h.update(str(path.relative_to(tmp_path)).encode())
        h.update(raw.encode())
    plan = {'schema': recovery.suite.SCHEMA, 'diagnostic_source_sha256': h.hexdigest(), 'runtime': {'python': 'fixed'}}
    plan['identity'] = digest(plan)
    return plan, validator, extra


def test_repair_proof_allows_only_known_validator_diff(tmp_path, monkeypatch):
    plan, validator, extra = source_fixture(tmp_path, monkeypatch)
    before = deepcopy(plan)
    proof = recovery.prove_source_change(plan)
    assert proof['old_diagnostic_sha256'] == plan['diagnostic_source_sha256']
    assert proof['new_diagnostic_sha256'] != proof['old_diagnostic_sha256']
    assert proof['allowed_diff'] and plan == before
    extra.write_text('changed inference code\n')
    with pytest.raises(ValueError, match='extend beyond'):
        recovery.prove_source_change(plan)
    extra.write_text('unchanged inference code\n')
    validator.write_text(validator.read_text().replace('assert sum(counts) == leaves', 'assert sum(counts) <= leaves'))
    with pytest.raises(ValueError, match='one allowed'):
        recovery.prove_source_change(plan)


def test_repair_rechecks_plan_seal_assets_and_runtime_without_mutation(tmp_path, monkeypatch):
    plan, _, _ = source_fixture(tmp_path, monkeypatch)
    inspected = []
    monkeypatch.setattr(recovery, 'ORIGINAL_VERIFY', lambda checking: inspected.append(deepcopy(checking)))
    monkeypatch.setattr(recovery.suite, 'inspect_runtime', lambda: {'python': 'fixed', 'initialized_cuda_device': None})
    before = deepcopy(plan)
    recovery.verify_compatible(plan)
    assert plan == before and inspected[0]['identity'] != plan['identity']
    assert inspected[0]['runtime'] == plan['runtime']
    monkeypatch.setattr(recovery.suite, 'inspect_runtime', lambda: {'python': 'changed'})
    with pytest.raises(ValueError, match='runtime'):
        recovery.verify_compatible(plan)
    plan['runtime'] = {'python': 'tampered'}
    with pytest.raises(ValueError, match='seal'):
        recovery.prove_source_change(plan)


def test_repair_receipt_rejects_changes_in_script_or_source(tmp_path):
    proof = {'repair': recovery.REPAIR, 'repair_script_sha256': 'original'}
    with pytest.raises(ValueError, match='requires'):
        recovery.repair_receipt(tmp_path, proof, require=True)
    recovery.repair_receipt(tmp_path, proof)
    recovery.repair_receipt(tmp_path, proof, require=True)
    with pytest.raises(ValueError, match='receipt changed'):
        recovery.repair_receipt(tmp_path, {**proof, 'repair_script_sha256': 'changed'})


def test_parent_reuses_published_and_unsealed_valid_handoffs(tmp_path, monkeypatch):
    plan, proof = {'output': str(tmp_path)}, {'repair': 'fixed'}
    jobs = [{'name': name, 'stage': stage} for name, stage in
            [('measure', 'utility'), ('published', 'handoff'), ('unsealed', 'handoff'), ('pending', 'handoff')]]
    done, launched, prepared = {'measure', 'published'}, [], []
    monkeypatch.setattr(recovery, 'preflight', lambda _: (plan, proof, jobs))
    monkeypatch.setattr(recovery.suite, 'job_complete', lambda _, job: job['name'] in done)
    def prepare(_, job):
        prepared.append(job['name'])
        return {'reuse': job['name'] == 'unsealed'}
    monkeypatch.setattr(recovery, 'prepare_simulation_job', prepare)
    monkeypatch.setattr(recovery.suite, 'run_command', lambda name, *_: launched.append(name))
    monkeypatch.setattr(recovery.suite, 'finish_job', lambda _, job: done.add(job['name']))
    monkeypatch.setattr(recovery.suite, 'build_report', lambda _: {'status': 'complete'})
    recovery.recover(tmp_path, .1)
    assert launched == ['pending'] and prepared == ['unsealed', 'pending']
    assert done == {job['name'] for job in jobs}


def test_preflight_refuses_unfinished_non_handoff_stage(tmp_path, monkeypatch):
    plan = {'output': str(tmp_path)}
    atomic_json(plan, tmp_path / 'student_inspection.json')
    monkeypatch.setattr(recovery, 'verify_compatible', lambda _: {})
    monkeypatch.setattr(recovery.suite, 'stage_commands', lambda *_: [{'name': 'measure', 'stage': 'utility'}])
    monkeypatch.setattr(recovery.suite, 'job_complete', lambda *_: False)
    with pytest.raises(ValueError, match='non-handoff'):
        recovery.preflight(tmp_path)
