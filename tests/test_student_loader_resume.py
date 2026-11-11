"""Synthetic submission preflight; no Slurm calls or scientific jobs."""

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from stmem.training.loader import runtime_loader_settings
from test_utility_students import rebased_manifest

spec = importlib.util.spec_from_file_location(
    "student_loader_resume_launcher", Path(__file__).resolve().parents[1] / "hpc/student_utility.py")
student_utility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(student_utility)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "1.5"])
def test_runtime_prefetch_rejects_nonpositive_or_noninteger_depth(bad):
    with pytest.raises(argparse.ArgumentTypeError, match="prefetch factor"):
        runtime_loader_settings({"training": {"workers": 16, "prefetch_factor": 1}}, bad)


def test_runtime_prefetch_preserves_config():
    cfg = {"training": {"workers": 16, "prefetch_factor": 1}}
    original = deepcopy(cfg)
    assert runtime_loader_settings(cfg, 2) == {
        "workers": 16, "prefetch_factor": 2, "prefetch_capacity_rank": 32,
    }
    assert cfg == original


@pytest.mark.parametrize("active", [True, False])
def test_three_student_resume_preflights_all_jobs_before_rendering(tmp_path, monkeypatch, active):
    manifest = rebased_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(student_utility, "read_site", lambda _: {"STMEM_REPO_ROOT": str(tmp_path)})
    monkeypatch.setattr(student_utility, "read_study_manifest", lambda _: manifest)
    monkeypatch.setattr(student_utility, "active_user_job_ids", lambda: {"73562"} if active else set())
    # The last arm is active: the first two plans must not be touched or submitted.
    for variant, job_id in zip(("s_std", "s_stage", "s_packet"), ("73560", "73561", "73562")):
        directory = tmp_path / "hpc/generated" / ("study_" + variant)
        directory.mkdir(parents=True)
        (directory / "submitted.json").write_text(json.dumps({"job_id": job_id}))
    calls = []

    def prepare(args, *_args):
        calls.append((args.utility_study["variant"], args.submit))
        assert args.resume and args.upgrade_loader and args.prefetch_factor == 2
        assert args.continue_to_step is None and args.microbatch == 64

    monkeypatch.setattr(student_utility, "prepare", prepare)
    command = ["--manifest", str(path), "--name", "study", "--microbatch", "64",
               "--resume", "--upgrade-loader", "--prefetch-factor", "2", "--submit"]
    if active:
        with pytest.raises(SystemExit):
            student_utility.main(command)
        assert not calls
    else:
        student_utility.main(command)
        assert calls == [(variant, submit) for submit in (False, True)
                         for variant in ("S_STD", "S_STAGE", "S_PACKET")]


def test_student_loader_upgrade_requires_resume(tmp_path):
    with pytest.raises(SystemExit):
        student_utility.main(["--manifest", str(tmp_path / "missing.json"), "--upgrade-loader"])
