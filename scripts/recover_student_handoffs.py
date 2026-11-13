#!/usr/bin/env python3
"""Resume only handoffs after the reviewed missing compact-method validator fix.

The original plan and outcomes are immutable. Reversing one exact validator
block must reproduce the original complete diagnostic source hash. Every other
source, runtime, asset and checkpoint guard remains enforced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from stmem.config import digest
from stmem.inspection import student_suite as suite
from stmem.inspection.student_simulation import prepare_simulation_job
from stmem.training.checkpoints import atomic_json, file_sha256

SCRIPT = Path(__file__).resolve()
VALIDATOR = suite.REPO / "src/stmem/inspection/handoff_validation.py"
ORIGINAL_VERIFY = suite.verify_plan
REPAIR = "compact_handoff_numeric_reader_validation_v1"
OLD_BLOCK = '                assert decision["timing"]["method"] == data["method"]\n                assert decision["timing"]["history_tokens"] <= data["budget"]\n                assert sum(decision["timing"]["represented_leaf_counts"]) == decision["timing"]["archive_fine_leaves"]\n'
NEW_BLOCK = '                timing = decision["timing"]\n                # Production compact selectors report numeric reader stats,\n                # while method/selection_family are recorded at trial level.\n                if "method" in timing:\n                    assert timing["method"] == data["method"]\n                if "selection_family" in data:\n                    assert data["selection_family"] == data["method"].lower()\n                counts = timing["represented_leaf_counts"]\n                tokens, leaves = timing["history_tokens"], timing["archive_fine_leaves"]\n                assert tokens == len(counts) == timing["active_tokens"] == timing["records_fetched"]\n                assert tokens == min(data["budget"], leaves)\n                assert all(type(count) is int and count > 0 for count in counts)\n                assert sum(counts) == leaves\n                if data["method"] == "TREE":\n                    assert timing["leaves_scored"] == 0\n                    assert timing["roots"] + timing["splits"] == tokens\n                else:\n                    assert timing["nodes_scored"] == 0\n                    assert timing["selected_leaves"] == sum(count == 1 for count in counts)\n                    assert timing["residuals"] == sum(count > 1 for count in counts)\n                    assert timing["selected_leaves"] + timing["residuals"] == tokens\n                    assert timing["leaves_scored"] == (leaves if leaves > data["budget"] else 0)\n'


def prove_source_change(plan):
    body = {key: value for key, value in plan.items() if key != 'identity'}
    if plan.get('schema') != suite.SCHEMA or digest(body) != plan.get('identity'):
        raise ValueError('Original inspection plan seal differs')
    current = VALIDATOR.read_text()
    if current.count(NEW_BLOCK) != 1:
        raise ValueError('Validator differs from the one allowed repair')
    old = current.replace(NEW_BLOCK, OLD_BLOCK, 1)
    paths = list((suite.REPO / 'src/stmem/inspection').glob('*.py'))
    paths.extend([suite.REPO / 'scripts/inspect_students.py',
                  suite.REPO / 'scripts/visualize_reset_cube_chunks.py'])
    previous, present = hashlib.sha256(), hashlib.sha256()
    for path in sorted(paths):
        key, raw = str(path.relative_to(suite.REPO)).encode(), path.read_bytes()
        previous.update(key)
        previous.update(old.encode() if path == VALIDATOR else raw)
        present.update(key)
        present.update(raw)
    if previous.hexdigest() != plan['diagnostic_source_sha256']:
        raise ValueError('Changes extend beyond the allowed validator repair')
    if present.hexdigest() != suite.diagnostic_source_hash():
        raise ValueError('Diagnostic source hashing contract changed')
    return {'repair': REPAIR, 'identity': plan['identity'],
            'old_diagnostic_sha256': previous.hexdigest(),
            'new_diagnostic_sha256': present.hexdigest(),
            'repair_script_sha256': file_sha256(SCRIPT),
            'allowed_diff': ''.join(difflib.unified_diff(
                old.splitlines(keepends=True), current.splitlines(keepends=True),
                fromfile='original/handoff_validation.py', tofile='repaired/handoff_validation.py'))}


def verify_compatible(plan):
    proof = prove_source_change(plan)
    checking = dict(plan)
    checking['diagnostic_source_sha256'] = proof['new_diagnostic_sha256']
    checking['identity'] = digest({k: v for k, v in checking.items() if k != 'identity'})
    ORIGINAL_VERIFY(checking)  # Never write this checking-only copy.
    runtime = {k: v for k, v in suite.inspect_runtime().items() if k != 'initialized_cuda_device'}
    if runtime != plan['runtime']:
        raise ValueError('Inspection runtime changed')
    return proof


def preflight(output):
    path = output / 'student_inspection.json'
    plan = suite.read_json(path)
    if Path(plan['output']).resolve() != output.resolve():
        raise ValueError('Recovery directory differs from original plan')
    proof = verify_compatible(plan)
    proof['original_plan_sha256'] = file_sha256(path)
    jobs = suite.stage_commands(plan, path)
    for job in jobs:
        if not suite.job_complete(plan, job) and job['stage'] != 'handoff':
            raise ValueError('Recovery requires all non-handoff stages already complete')
    return plan, proof, jobs


def repair_receipt(output, proof, *, require=False):
    path = output / 'handoff_validator_repair.json'
    if path.exists():
        if suite.read_json(path) != proof:
            raise ValueError('Repair receipt changed or belongs to another implementation')
    elif require:
        raise ValueError('Recovery worker requires its parent repair receipt')
    else:
        atomic_json(proof, path)


def run_worker(output, name):
    plan, proof, jobs = preflight(output)
    repair_receipt(output, proof, require=True)
    matches = [job for job in jobs if job['name'] == name and job['stage'] == 'handoff']
    if len(matches) != 1 or suite.job_complete(plan, matches[0]):
        raise ValueError('Worker must identify one pending original handoff')
    job = matches[0]
    # Scoped to this dedicated subprocess; other entrypoints retain strict seals.
    suite.verify_plan = verify_compatible
    from stmem.inspection.student_handoff import main as handoff_main
    marker = job['command'].index('stmem.inspection.student_handoff')
    handoff_main(job['command'][marker + 1:])
    if verify_compatible(plan) != {k: v for k, v in proof.items() if k != 'original_plan_sha256'}:
        raise ValueError('Repair implementation changed during simulation')


def recover(output, hours):
    with suite.inspection_lock(output), suite.termination_signals():
        plan, proof, jobs = preflight(output)
        repair_receipt(output, proof)
        deadline = time.monotonic() + hours * 3600
        try:
            for job in jobs:
                if suite.job_complete(plan, job):
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError('Handoff recovery reached --max-hours; repeat recovery command')
                if not prepare_simulation_job(plan, job)['reuse']:
                    command = [sys.executable, '-u', str(SCRIPT), '--output', str(output), '--job', job['name']]
                    suite.run_command(job['name'], command, output, deadline)
                suite.finish_job(plan, job)
            if preflight(output)[1] != proof:
                raise ValueError('Recovery inputs changed')
            summary = suite.build_report(plan)
            suite.status(output, summary['status'], report=str(output / 'index.html'), repair=REPAIR)
        except BaseException as error:
            report_error = None
            try:
                suite.build_report(plan)
            except Exception as failure:
                report_error = f'{type(failure).__name__}: {failure}'
            suite.status(output, 'interrupted' if isinstance(error, (KeyboardInterrupt, TimeoutError)) else 'failed',
                         error=f'{type(error).__name__}: {error}', report_error=report_error, repair=REPAIR)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-hours', type=float, default=1)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--job', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        parser.error('--max-hours must be finite and positive')
    if args.job and (args.detach or args.dry_run):
        parser.error('Worker flags conflict')
    output = args.output.resolve()
    if args.job:
        run_worker(output, args.job)
        return
    plan, proof, jobs = preflight(output)
    pending = [job['name'] for job in jobs if not suite.job_complete(plan, job)]
    if args.dry_run:
        print(json.dumps({'proof': proof, 'pending_handoffs': pending,
                          'action': 'read-only; no outputs modified or simulations launched'}, indent=2))
    elif args.detach:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        log = output / 'logs' / f'handoff_recovery_{stamp}.log'
        command = [sys.executable, '-u', str(SCRIPT), '--output', str(output), '--max-hours', str(args.max_hours)]
        with log.open('x') as stream:
            process = subprocess.Popen(command, cwd=suite.REPO, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': process.pid, 'log': str(log), 'status_path': str(output / 'status.json'),
                          'pending_handoffs': len(pending)}, indent=2))
    else:
        recover(output, args.max_hours)


if __name__ == '__main__':
    main()
