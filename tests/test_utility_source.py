"""CPU-only provenance tests for the source-bound HPC utility measurements."""

import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc.plan import source_hash  # noqa: E402
from hpc.utility_source import (  # noqa: E402
    RECEIPT, _digest, _export_archive, create_study_source, verify_study_source,
)


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def committed_study(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src/stmem").mkdir(parents=True)
    (repo / "hpc").mkdir()
    (repo / "src/stmem/measurement.py").write_text("VALUE = 'original measurement'\n")
    (repo / "hpc/launch.py").write_text("print('original launch')\n")
    (repo / "hpc/run.sh").write_text("#!/bin/sh\nexit 0\n")
    (repo / "hpc/run.sh").chmod(0o755)
    (repo / ".gitignore").write_text("hpc/generated/\nstudy.json\n")
    git(repo, "init", "--quiet")
    git(repo, "config", "user.name", "Synthetic provenance fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "Original study source")
    revision = git(repo, "rev-parse", "HEAD")
    manifest = {"source": {"git": revision, "dirty": False, "source_sha256": source_hash(repo)},
                "states": ["synthetic fixed state"], "training_pool": ["synthetic fixed episode"]}
    manifest["experiment_id"] = _digest(manifest)
    path = repo / "study.json"
    path.write_text(json.dumps(manifest))
    return repo, path, manifest


def reseal(path, manifest):
    manifest.pop("experiment_id", None)
    manifest["experiment_id"] = _digest(manifest)
    path.write_text(json.dumps(manifest))


def test_exports_original_source_after_checkout_advances_without_changing_git(committed_study):
    repo, manifest_path, manifest = committed_study
    (repo / "src/stmem/measurement.py").write_text("VALUE = 'new unrelated runtime'\n")
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "New source after panel was sealed")
    before = {"head": git(repo, "rev-parse", "HEAD"), "status": git(repo, "status", "--porcelain"),
              "worktrees": git(repo, "worktree", "list", "--porcelain"),
              "refs": git(repo, "show-ref"), "manifest": manifest_path.read_bytes()}
    root = create_study_source(repo, manifest_path)
    assert source_hash(root) == manifest["source"]["source_sha256"] != source_hash(repo)
    assert root == repo / "hpc/generated/utility_sources" / manifest["source"]["source_sha256"] / manifest["source"]["git"]
    assert (root / "src/stmem/measurement.py").read_text() == "VALUE = 'original measurement'\n"
    assert (root / "hpc/run.sh").stat().st_mode & 0o111
    assert not (root / ".git").exists()
    assert git(repo, "rev-parse", "HEAD") == before["head"]
    assert git(repo, "status", "--porcelain") == before["status"]
    assert git(repo, "worktree", "list", "--porcelain") == before["worktrees"]
    assert git(repo, "show-ref") == before["refs"]
    assert manifest_path.read_bytes() == before["manifest"]
    assert create_study_source(repo, manifest_path) == root


@pytest.mark.parametrize("relative", ["src/stmem/measurement.py", "hpc/launch.py", "hpc/extra.py"])
def test_reuse_rejects_changed_or_added_runtime_files(committed_study, relative):
    repo, manifest_path, _ = committed_study
    root = create_study_source(repo, manifest_path)
    (root / relative).write_text("print('altered')\n")
    with pytest.raises(ValueError, match="source differs|source files changed"):
        create_study_source(repo, manifest_path)


def test_reuse_allows_only_import_bytecode(committed_study):
    repo, manifest_path, _ = committed_study
    root = create_study_source(repo, manifest_path)
    directory = root / "src/stmem/__pycache__"
    directory.mkdir()
    (directory / "measurement.cpython-311.pyc").write_bytes(b"synthetic bytecode")
    assert create_study_source(repo, manifest_path) == root
    (directory / "unexpected.py").write_text("VALUE = 'new source'\n")
    with pytest.raises(ValueError, match="source differs"):
        verify_study_source(root, manifest_path)


def test_receipt_is_sealed_and_cannot_be_silently_replaced(committed_study):
    repo, manifest_path, _ = committed_study
    root = create_study_source(repo, manifest_path)
    receipt = json.loads((root / RECEIPT).read_text())
    receipt["files"].pop("hpc/launch.py")
    (root / RECEIPT).write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt changed"):
        verify_study_source(root, manifest_path)


def test_uncommitted_source_cannot_be_reconstructed_or_published(committed_study):
    repo, manifest_path, manifest = committed_study
    (repo / "src/stmem/measurement.py").write_text("VALUE = 'uncommitted experiment'\n")
    manifest["source"].update(dirty=True, source_sha256=source_hash(repo))
    reseal(manifest_path, manifest)
    with pytest.raises(ValueError, match="does not reconstruct.*source fingerprint"):
        create_study_source(repo, manifest_path)
    parent = repo / "hpc/generated/utility_sources" / manifest["source"]["source_sha256"]
    assert not (parent / manifest["source"]["git"]).exists()
    assert not [path for path in parent.iterdir() if path.is_dir()]


def test_dirty_manifest_with_identical_committed_source_is_valid(committed_study):
    repo, manifest_path, manifest = committed_study
    manifest["source"]["dirty"] = True
    reseal(manifest_path, manifest)
    assert create_study_source(repo, manifest_path).is_dir()


def test_commits_with_same_python_source_keep_distinct_launch_files(committed_study):
    repo, manifest_path, manifest = committed_study
    first = create_study_source(repo, manifest_path)
    (repo / "hpc/launch.py").write_text("print('new launcher with unchanged source')\n")
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "Only launcher changes")
    manifest["source"]["git"] = git(repo, "rev-parse", "HEAD")
    reseal(manifest_path, manifest)
    second = create_study_source(repo, manifest_path)
    assert first != second and first.parent == second.parent
    assert source_hash(first) == source_hash(second)
    assert (first / "hpc/launch.py").read_bytes() != (second / "hpc/launch.py").read_bytes()


def test_missing_commit_does_not_fall_back_to_current_checkout(committed_study):
    repo, manifest_path, manifest = committed_study
    manifest["source"]["git"] = "f" * 40
    reseal(manifest_path, manifest)
    with pytest.raises(ValueError, match="unavailable locally"):
        create_study_source(repo, manifest_path)


def test_modified_manifest_is_rejected_before_export(committed_study):
    repo, manifest_path, manifest = committed_study
    manifest["states"].append("different selected state")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest content hash differs"):
        create_study_source(repo, manifest_path)
    assert not (repo / "hpc/generated/utility_sources").exists()


@pytest.mark.parametrize("kind,name", [
    (tarfile.REGTYPE, "../outside.py"),
    (tarfile.REGTYPE, "/absolute.py"),
    (tarfile.REGTYPE, ".git/config"),
    (tarfile.SYMTYPE, "symlink"),
    (tarfile.LNKTYPE, "hardlink"),
])
def test_archive_rejects_unsafe_paths_and_links(tmp_path, kind, name):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = "../outside"
        if kind == tarfile.REGTYPE:
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
        else:
            archive.addfile(member)
    destination = tmp_path / "archive"
    destination.mkdir()
    with pytest.raises(ValueError, match="Unsafe|unsupported"):
        _export_archive(raw.getvalue(), destination)
    assert not (tmp_path / "outside.py").exists()


def test_reuse_rejects_a_symlink_even_when_it_points_to_identical_bytes(committed_study):
    repo, manifest_path, _ = committed_study
    root = create_study_source(repo, manifest_path)
    target = root / "hpc/launch.py"
    target.unlink()
    target.symlink_to(repo / "hpc/launch.py")
    with pytest.raises(ValueError, match="symbolic link"):
        verify_study_source(root, manifest_path)
