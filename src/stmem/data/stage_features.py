"""Optionally stage immutable frozen descriptors on explicitly chosen local storage.

Run this once in the allocated job, before torchrun. No scientific configuration
is rewritten: STMEM_FEATURE_CACHE_ROOT changes only the consumer's physical path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
import uuid

import h5py

from stmem.config import digest, load_config
from stmem.data.feature_cache import feature_cache_manifest
from stmem.vision.frozen import resolve_visual_features, visual_feature_signature


STAGING_VERSION = "immutable_feature_staging_v1"
RESERVE_BYTES = 64 * 1024 * 1024
COPY_BLOCK_BYTES = 16 * 1024 * 1024


def consumer_feature_cache_root(config) -> Path:
    """Resolve a runtime read-only location without changing feature identity."""
    override = os.environ.get("STMEM_FEATURE_CACHE_ROOT", "")
    path = Path(override or resolve_visual_features(config)["cache_dir"]).expanduser()
    if override and not path.is_absolute():
        raise ValueError("STMEM_FEATURE_CACHE_ROOT must be an absolute directory")
    return path


def _fingerprint(path: Path) -> dict:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Feature staging requires an ordinary file, not a symlink: {path}")
    return {
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def _directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Unsafe feature staging directory: {path}") from None


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(COPY_BLOCK_BYTES), b""):
            value.update(block)
    return value.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".partial.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(path: Path):
    # Do not follow a preexisting symlink or truncate another job's lock inode.
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _copy_verified(source: Path, destination: Path, expected: dict) -> dict:
    """Publish only a complete, read-back-verified copy of an unchanged source."""
    temporary = destination.with_name(destination.name + f".partial.{os.getpid()}.{uuid.uuid4().hex}")
    copied = hashlib.sha256()
    try:
        if _fingerprint(source) != expected:
            raise ValueError(f"Source changed before feature staging: {source}")
        with source.open("rb") as reader, temporary.open("xb") as writer:
            for block in iter(lambda: reader.read(COPY_BLOCK_BYTES), b""):
                writer.write(block)
                copied.update(block)
            writer.flush()
            os.fsync(writer.fileno())
        if _fingerprint(source) != expected or temporary.stat().st_size != expected["size"]:
            raise ValueError(f"Source changed while staging immutable features: {source}")
        if _sha256(temporary) != copied.hexdigest():
            raise OSError(f"Staged feature copy checksum mismatch: {source}")
        if _fingerprint(source) != expected:
            raise ValueError(f"Source changed while verifying staged features: {source}")
        temporary.chmod(0o444)
        # The namespace lock prevents another legitimate writer racing this check.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to replace existing staged features: {destination}")
        os.replace(temporary, destination)
        return {"sha256": copied.hexdigest(), "destination": _fingerprint(destination)}
    finally:
        temporary.unlink(missing_ok=True)


def _recover_verified(source: Path, destination: Path, expected: dict) -> dict:
    """Recover a publication interrupted before its checksum record was saved.

    This uncommon path reads both entire files. An untracked file is never
    trusted from its filename/size, and is never overwritten on a mismatch.
    """
    before = _fingerprint(destination)
    if before["size"] != expected["size"] or _fingerprint(source) != expected:
        raise ValueError(f"Untracked staged feature file does not match its source: {destination}")
    source_sha = _sha256(source)
    destination_sha = _sha256(destination)
    if (
        source_sha != destination_sha
        or _fingerprint(source) != expected
        or _fingerprint(destination) != before
    ):
        raise ValueError(f"Untracked staged feature file failed checksum verification: {destination}")
    return {"sha256": destination_sha, "destination": before}


def stage_feature_cache(config: dict, scratch_root: str | Path, *, progress=None) -> dict:
    """Stage selected task sidecars once, with bounded-copy memory and safe reuse.

    Reuse trusts the recorded, verified SHA256 only while both source and staged
    file stat fingerprints remain unchanged. A modified staged file is rejected,
    never overwritten while another job might still be reading it.
    """
    started = time.perf_counter()
    emit = progress or (lambda _event: None)
    spec = resolve_visual_features(config)
    if spec["mode"] == "xyzrgb":
        return {"status": "not_needed", "path": None, "copied_bytes": 0, "reused_bytes": 0}
    scratch = Path(scratch_root).expanduser()
    if not str(scratch_root).strip() or not scratch.is_absolute():
        raise ValueError("Feature staging requires an explicit absolute STMEM_SCRATCH_ROOT")
    if not scratch.is_dir():
        raise FileNotFoundError(f"Scratch directory does not exist: {scratch}; choose a site-provided location")
    scratch = scratch.resolve()
    source_root = Path(spec["cache_dir"]).expanduser().resolve()
    prepared_root = Path(config["paths"]["data_root"]) / config["data"].get(
        "prepared_subdir", "prepared_balanced_v2"
    )
    tasks = list(dict.fromkeys(config["data"]["tasks"]))
    if not tasks:
        raise ValueError("Feature staging requires at least one configured task")
    sources = {}
    for task in tasks:
        if not isinstance(task, str) or not task or Path(task).name != task or task in (".", ".."):
            raise ValueError(f"Unsafe feature staging task name: {task!r}")
        source = source_root / f"{task}.h5"
        fingerprint = _fingerprint(source)
        with h5py.File(prepared_root / f"{task}.h5", "r") as handle:
            prepared = json.loads(handle.attrs["manifest"])
        manifest = feature_cache_manifest(source, prepared, spec)
        sources[task] = {
            "path": str(source),
            "fingerprint": fingerprint,
            "manifest_sha256": digest(manifest),
        }
    identity = {
        "format": STAGING_VERSION,
        "feature_signature": visual_feature_signature(spec),
        "sources": sources,
    }
    namespace = scratch / "chronotree_features"
    if namespace == source_root or source_root in namespace.parents:
        raise ValueError("Feature staging destination must be outside the source feature directory")
    _directory(namespace)
    namespace = namespace / f"uid_{os.getuid()}"
    _directory(namespace)
    destination_root = namespace / digest(identity)
    if destination_root == source_root or source_root in destination_root.parents:
        raise ValueError("Feature staging destination must be outside the source feature directory")
    _directory(destination_root)
    record_path = destination_root / "staging.json"
    copied_bytes = reused_bytes = 0
    with _lock(destination_root / ".lock"):
        if record_path.is_symlink():
            raise ValueError(f"Unsafe feature staging record: {record_path}")
        record = json.loads(record_path.read_text()) if record_path.exists() else {**identity, "files": {}}
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Staging identity differs from immutable source: {destination_root}")
        if not isinstance(record.get("files"), dict):
            raise ValueError(f"Invalid feature staging file record: {record_path}")
        pending = []
        for task, source in sources.items():
            if _fingerprint(Path(source["path"])) != source["fingerprint"]:
                raise ValueError(f"Source changed while waiting for feature staging: {source['path']}")
            destination = destination_root / f"{task}.h5"
            saved = record.get("files", {}).get(task)
            if destination.exists() or destination.is_symlink():
                if saved is None:
                    emit({"event": "feature_staging", "phase": "verify_existing", "task": task,
                          "bytes": source["fingerprint"]["size"]})
                    saved = _recover_verified(Path(source["path"]), destination, source["fingerprint"])
                    record["files"][task] = saved
                    _atomic_json(record_path, record)
                if not isinstance(saved, dict) or saved.get("destination") != _fingerprint(destination):
                    raise ValueError(f"Staged feature file changed or is untracked: {destination}")
                if not isinstance(saved.get("sha256"), str) or len(saved["sha256"]) != 64:
                    raise ValueError(f"Staged feature checksum record is missing: {destination}")
                reused_bytes += source["fingerprint"]["size"]
            else:
                pending.append((task, source, destination))
        required = sum(source["fingerprint"]["size"] for _, source, _ in pending)
        free = shutil.disk_usage(destination_root).free
        emit({"event": "feature_staging", "phase": "space_check", "path": str(destination_root),
              "required_bytes": required, "free_bytes": free, "reused_bytes": reused_bytes})
        if required and free < required + RESERVE_BYTES:
            raise OSError(
                f"Insufficient scratch space for frozen features: need {required:,} bytes plus "
                f"{RESERVE_BYTES:,} reserve, have {free:,} at {scratch}. "
                "Choose larger local storage or clear STMEM_SCRATCH_ROOT to use shared features."
            )
        for task, source, destination in pending:
            emit({"event": "feature_staging", "phase": "copy_started", "task": task,
                  "bytes": source["fingerprint"]["size"]})
            result = _copy_verified(Path(source["path"]), destination, source["fingerprint"])
            record["files"][task] = result
            _atomic_json(record_path, record)
            copied_bytes += source["fingerprint"]["size"]
            emit({"event": "feature_staging", "phase": "copy_complete", "task": task,
                  "bytes": source["fingerprint"]["size"]})
    report = {"event": "feature_staging", "status": "ready", "path": str(destination_root),
              "copied_bytes": copied_bytes, "reused_bytes": reused_bytes,
              "elapsed_seconds": time.perf_counter() - started}
    emit(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scratch-root", default=os.environ.get("STMEM_SCRATCH_ROOT", ""))
    parser.add_argument("--print-path", action="store_true", help="Print only the runtime cache path on stdout")
    args = parser.parse_args()
    report = stage_feature_cache(
        load_config(args.config), args.scratch_root,
        progress=lambda row: print(json.dumps(row), file=sys.stderr, flush=True),
    )
    print((report["path"] or "") if args.print_path else json.dumps(report))


if __name__ == "__main__":
    main()
