"""Reconstruct the exact source of a sealed utility study without changing Git.

The study and its trained students retain their original manifest identity.
Only committed source whose complete ``src`` fingerprint matches that manifest
can be exported. The isolated copy is reusable, content checked, and contains no
Git worktree or branch metadata. This module performs no GPU work or submission.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile

from hpc.plan import sha256, source_hash


RECEIPT = ".utility-source.json"
SCHEMA = "utility_committed_source_v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _manifest_source(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    body = {key: value for key, value in manifest.items() if key != "experiment_id"}
    if _digest(body) != manifest.get("experiment_id"):
        raise ValueError("Study manifest content hash differs")
    source = manifest.get("source", {})
    revision, fingerprint = source.get("git"), source.get("source_sha256")
    if not isinstance(revision, str) or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision):
        raise ValueError("Study source requires its recorded full Git commit ID")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("Study source requires its recorded source SHA256")
    return revision, fingerprint


def _inventory(root):
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"Isolated study source contains a symbolic link: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Isolated study source contains a nonregular file: {relative}")
        if relative.as_posix() == RECEIPT:
            continue
        # Imports may create bytecode, which is never a canonical source input.
        if "__pycache__" in relative.parts and path.suffix == ".pyc":
            continue
        files[relative.as_posix()] = {
            "sha256": sha256(path), "executable": bool(path.stat().st_mode & 0o111),
        }
    return files


def verify_study_source(code_root, manifest_path):
    """Check a published source copy and return its immutable export receipt."""
    root = Path(code_root)
    revision, fingerprint = _manifest_source(manifest_path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Isolated study source must be a real directory")
    if (root / RECEIPT).is_symlink():
        raise ValueError("Isolated study source receipt must not be a symbolic link")
    receipt = json.loads((root / RECEIPT).read_text())
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (receipt.get("schema") != SCHEMA or receipt.get("receipt_sha256") != _digest(body)
            or receipt.get("git") != revision or receipt.get("source_sha256") != fingerprint):
        raise ValueError("Isolated study source receipt changed or belongs to another revision")
    if source_hash(root) != fingerprint:
        raise ValueError("Isolated study source differs from the original study fingerprint")
    if _inventory(root) != receipt.get("files"):
        raise ValueError("Isolated study source files changed, including runtime launch files")
    return receipt


def _export_archive(raw, destination):
    """Extract regular Git archive files only; reject links and unsafe paths."""
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        seen = set()
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                    or relative.parts[0] == ".git" or relative.as_posix() == RECEIPT
                    or relative.as_posix() in seen):
                raise ValueError(f"Unsafe or duplicate source archive entry: {member.name}")
            seen.add(relative.as_posix())
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Unreadable source archive entry: {member.name}")
                with stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
            else:
                raise ValueError(f"Source archive links and special entries are unsupported: {member.name}")


def create_study_source(repo, manifest_path):
    """Atomically export and verify the study's recorded commit; return its root.

    Separate commits may share the same Python source hash but differ in shell
    launch files. Both identities therefore namespace the export directory.
    A dirty manifest is accepted only if the recorded commit reconstructs its
    exact source bytes (for example, if only documentation was dirty).
    """
    repo, manifest_path = Path(repo).resolve(), Path(manifest_path).resolve()
    revision, fingerprint = _manifest_source(manifest_path)
    parent = repo / "hpc/generated/utility_sources" / fingerprint
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / revision
    with (parent / f"{revision}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            verify_study_source(destination, manifest_path)
            return destination
        try:
            resolved = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "--verify", revision + "^{commit}"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            if resolved != revision:
                raise ValueError("Recorded study revision is not the exact commit ID")
            archive = subprocess.run(
                ["git", "-C", str(repo), "archive", "--format=tar", revision],
                check=True, capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"Recorded study commit {revision} is unavailable locally; fetch that commit first"
            ) from error
        temporary = Path(tempfile.mkdtemp(prefix=f".{revision}.", dir=parent))
        try:
            _export_archive(archive, temporary)
            if source_hash(temporary) != fingerprint:
                raise ValueError(
                    "The recorded Git commit does not reconstruct the study source fingerprint. "
                    "The study may include uncommitted Python changes; restore those exact bytes "
                    "instead of bypassing or rewriting its source guard."
                )
            receipt = {
                "schema": SCHEMA, "git": revision, "source_sha256": fingerprint,
                "archive_sha256": hashlib.sha256(archive).hexdigest(), "files": _inventory(temporary),
            }
            receipt["receipt_sha256"] = _digest(receipt)
            (temporary / RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            verify_study_source(temporary, manifest_path)
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return destination
