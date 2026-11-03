"""Persist deterministic hierarchy parents once over immutable encoded leaves.

Payload leaves remain in the original encoder shards. HDF5 virtual datasets
present leaves and newly stored parents as one indexed node table. The derived
cache is bound to that absolute leaf-cache location; missing sources are rejected
before reading VDS data, whose HDF5 default otherwise silently supplies fill data.
No neural network, raw point cloud or trainable parameter is used here.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import time

import h5py
import numpy as np
import torch

from stmem.config import digest, source_revision
from stmem.data.cache import read_records
from stmem.data.cache_builder import locked, read_json, require_file
from stmem.memory.hierarchy import local_tree, stable_id
from stmem.memory.records import aggregate
from stmem.training.checkpoints import atomic_json, file_sha256


FORMAT = 1
PAYLOAD_FIELDS = ("content_sum", "anchor", "bounds_min", "bounds_max", "max_content")
STRING_FIELDS = ("node_id", "source", "modality")
SCALAR_FIELDS = ("count", "time_min", "time_max", "depth", "scale", "completed",
                 "disconnected", "time_anchor", "occupancy")


class HierarchyInterrupted(RuntimeError):
    pass


def hierarchy_code_fingerprint():
    root = Path(__file__).resolve().parents[3]
    result = hashlib.sha256(b"persisted_hierarchy_builder_v1\0")
    for relative in ("src/stmem/data/hierarchy_cache.py", "src/stmem/data/cache.py",
                     "src/stmem/memory/hierarchy.py", "src/stmem/memory/records.py"):
        content = (root / relative).read_bytes()
        result.update(relative.encode() + b"\0" + len(content).to_bytes(8, "big") + content)
    return result.hexdigest()


def _leaf_manifest(leaf_root):
    root = Path(leaf_root).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("identity", {}).get("format") != 2:
        raise ValueError("Persisted hierarchy requires a complete format-2 encoder cache")
    tasks = manifest["tasks"]
    if (not tasks or len(tasks) != len(set(tasks))
            or any(not re.fullmatch(r"[A-Za-z0-9_]+", name) for name in tasks)
            or set(manifest["shards"]) != {task + ".h5" for task in tasks}):
        raise ValueError("Invalid encoder-cache task coverage")
    return manifest


def hierarchy_identity(leaf_root, manifest=None):
    root = Path(leaf_root).resolve()
    manifest = _leaf_manifest(root) if manifest is None else manifest
    return {"format": FORMAT, "leaf_root": str(root),
            "leaf_manifest_sha256": file_sha256(root / "manifest.json"),
            "leaf_identity_sha256": digest(manifest["identity"]),
            "builder_code_sha256": hierarchy_code_fingerprint(),
            "representation": "indexed_parent_payload_and_virtual_leaves_v1"}


def default_hierarchy_root(leaf_root, manifest=None):
    return Path(leaf_root).resolve() / "hierarchy" / digest(hierarchy_identity(leaf_root, manifest))[:24]


def _location(leaf_root, output_root, leaf_manifest):
    return Path(output_root).resolve() if output_root else default_hierarchy_root(leaf_root, leaf_manifest)


def _check_leaf_file(root, task, manifest, *, full_hash=False):
    path = root / f"{task}.h5"
    expected = manifest["shards"][path.name]
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Missing or changed virtual-dataset leaf source: {path}")
    receipt = read_json(root / f"{task}.complete.json")
    if (receipt.get("kind") != "encoder_cache_task" or receipt.get("status") != "complete"
            or receipt.get("identity_sha256") != digest(manifest["identity"])
            or receipt.get("task") != task
            or any(receipt.get(key) != expected[key] for key in ("sha256", "bytes", "episodes"))):
        raise ValueError(f"Leaf source receipt differs from manifest: {path}")
    if full_hash and file_sha256(path) != expected["sha256"]:
        raise ValueError(f"Corrupted immutable leaf source: {path}")


def hierarchy_status(leaf_root, output_root=None):
    leaf_root = Path(leaf_root).resolve()
    leaf_manifest = _leaf_manifest(leaf_root)
    root = _location(leaf_root, output_root, leaf_manifest)
    done = [task for task in leaf_manifest["tasks"]
            if (root / f"{task}.h5").is_file() and (root / f"{task}.complete.json").is_file()]
    complete = (len(done) == len(leaf_manifest["tasks"])
                and (root / "manifest.json").is_file() and (root / "complete.json").is_file())
    return {"status": "complete" if complete else (
                "incomplete" if root.exists() else "not_started"),
            "hierarchy_cache_root": str(root), "completed_tasks": done,
            "missing_tasks": [task for task in leaf_manifest["tasks"] if task not in done],
            "note": "Filesystem progress only; validate_hierarchy_cache checks compatible metadata and sizes."}


def validate_hierarchy_cache(leaf_root, output_root=None):
    """Check all source/derived identities, receipts and sizes, without full scans."""
    leaf_root = Path(leaf_root).resolve()
    leaf_manifest = _leaf_manifest(leaf_root)
    identity = hierarchy_identity(leaf_root, leaf_manifest)
    root = _location(leaf_root, output_root, leaf_manifest)
    manifest = read_json(root / "manifest.json")
    marker = read_json(root / "complete.json")
    if (manifest.get("status") != "complete" or manifest.get("identity") != identity
            or manifest.get("tasks") != leaf_manifest["tasks"]
            or set(manifest.get("shards", {})) != set(leaf_manifest["shards"])
            or marker.get("identity_sha256") != digest(identity)
            or marker.get("manifest_sha256") != file_sha256(root / "manifest.json")):
        raise ValueError("Stale or incomplete persisted hierarchy cache")
    for task in manifest["tasks"]:
        _check_leaf_file(leaf_root, task, leaf_manifest)
        path = root / f"{task}.h5"
        receipt = read_json(root / f"{task}.complete.json")
        entry = manifest["shards"][path.name]
        if (not path.is_file() or path.stat().st_size != entry["bytes"]
                or receipt.get("kind") != "hierarchy_cache_task" or receipt.get("status") != "complete"
                or receipt.get("identity_sha256") != digest(identity)
                or receipt.get("task") != task or receipt.get("leaf_sha256") != leaf_manifest["shards"][path.name]["sha256"]
                or any(receipt.get(key) != entry[key] for key in ("sha256", "bytes", "episodes"))):
            raise ValueError(f"Missing or inconsistent hierarchy shard: {path}")
        # Check location binding/header too: a missing VDS source must never be
        # interpreted as valid zero-valued learned content.
        with h5py.File(path, "r") as file:
            if (file.attrs.get("identity_sha256") != digest(identity)
                    or file.attrs.get("leaf_source") != str(leaf_root / path.name)):
                raise ValueError(f"Hierarchy virtual-source binding differs: {path}")
    return manifest


@dataclass
class _Ref:
    record: object
    index: int


class _EpisodeWriter:
    def __init__(self, out, source, width):
        self.out, self.source, self.width = out, source, width
        self.rows = {field: [] for field in (*STRING_FIELDS, *SCALAR_FIELDS, "children")}
        self.parent = out.create_group("parent_payload")
        for field in PAYLOAD_FIELDS:
            dim = width if field in ("content_sum", "max_content") else 3
            self.parent.create_dataset(field, shape=(0, dim), maxshape=(None, dim), dtype="f4", chunks=True)
        self.parent_count = 0
        self.segments = []

    def _metadata(self, record, children):
        index = len(self.rows["node_id"])
        for field in (*STRING_FIELDS, *SCALAR_FIELDS):
            self.rows[field].append(getattr(record, field))
        self.rows["children"].append(children)
        return index

    def parents(self, records, child_indices):
        if not records:
            return []
        start = len(self.rows["node_id"])
        indices = [self._metadata(record, children) for record, children in zip(records, child_indices)]
        offset, size = self.parent_count, len(records)
        for field in PAYLOAD_FIELDS:
            values = torch.stack([getattr(record, field) for record in records]).numpy()
            dataset = self.parent[field]
            dataset.resize(offset + size, axis=0)
            dataset[offset:offset + size] = values
        self.parent_count += size
        self.segments.append((start, size, None, offset))
        return indices

    def block(self, name, key):
        original = self.source[name][key]
        leaves = read_records(original)
        start = len(self.rows["node_id"])
        lookup = {id(record): self._metadata(record, (-1, -1)) for record in leaves}
        if leaves:
            self.segments.append((start, len(leaves), original.name, 0))
        groups = defaultdict(list)
        for leaf in leaves:
            if not leaf.is_leaf or leaf.count != 1:
                raise ValueError("Hierarchy inputs must be unit-mass original leaves")
            expected_source = "execution" if name == "prefix" else name
            if leaf.source != expected_source or leaf.completed != (name != "prefix"):
                raise ValueError("Cached block crosses a source/completion boundary")
            groups[leaf.group].append(leaf)
        roots = [(group, local_tree(members)) for group, members in sorted(groups.items())]
        parents, children = [], []

        def visit(record):
            if not record.children:
                return lookup[id(record)]
            pair = tuple(visit(child) for child in record.children)
            index = start + len(leaves) + len(parents)
            parents.append(record)
            children.append(pair)
            lookup[id(record)] = index
            return index

        for _, root in roots:
            visit(root)
        self.parents(parents, children)
        target = self.out.require_group("blocks/" + name).create_group(key)
        target.attrs["leaf_start"], target.attrs["leaf_count"] = start, len(leaves)
        target.create_dataset("roots", data=np.asarray([lookup[id(root)] for _, root in roots], dtype="i8"))
        return [(group, _Ref(replace(root, children=()), lookup[id(root)])) for group, root in roots]

    def append(self, forest, roots):
        for kind, root in roots:
            rows = forest.setdefault(kind, [])
            if rows and root.record.time_min < rows[-1].record.time_min:
                raise ValueError("Encoded blocks are not in chronological order")
            while rows and rows[-1].record.scale == root.record.scale:
                previous = rows.pop()
                parent = aggregate(
                    (previous.record, root.record), stable_id("temporal", (previous.record.node_id, root.record.node_id)),
                    children=True, scale=root.record.scale + 1,
                )
                index = self.parents([parent], [(previous.index, root.index)])[0]
                root = _Ref(replace(parent, children=()), index)
            rows.append(root)

    def finish(self):
        nodes = self.out.create_group("nodes")
        for field, rows in self.rows.items():
            if field in STRING_FIELDS:
                nodes.create_dataset(field, data=rows, dtype=h5py.string_dtype("utf-8"))
            elif field == "children":
                nodes.create_dataset(field, data=np.asarray(rows, dtype="i8").reshape(-1, 2))
            else:
                dtype = "?" if field in ("completed", "disconnected") else (
                    "i8" if field in ("count", "depth", "scale") else "f8"
                )
                nodes.create_dataset(field, data=np.asarray(rows, dtype=dtype))
        for field in PAYLOAD_FIELDS:
            dim = self.width if field in ("content_sum", "max_content") else 3
            shape = len(self.rows["node_id"]), dim
            layout = h5py.VirtualLayout(shape=shape, dtype="f4")
            for start, count, source_group, parent_offset in self.segments:
                if source_group is None:
                    source = h5py.VirtualSource(".", self.parent[field].name, shape=(self.parent_count, dim))
                    layout[start:start + count] = source[parent_offset:parent_offset + count]
                else:
                    source = h5py.VirtualSource(str(Path(self.source.file.filename).resolve()),
                                               source_group + "/" + field, shape=(count, dim))
                    layout[start:start + count] = source
            if shape[0]:
                nodes.create_virtual_dataset(field, layout)
            else:
                nodes.create_dataset(field, shape=shape, dtype="f4")
        self.out.attrs["node_count"] = len(self.rows["node_id"])
        self.out.attrs["parent_count"] = self.parent_count
        self.out.attrs["width"] = self.width


def write_episode_hierarchy(source, target, identity, leaf_manifest, check_stop=lambda: None):
    """One episode, at most one block's tree payload resident at a time."""
    model = leaf_manifest["identity"]["model_encoder"]
    block_size = model["block_size"]
    observation_stride = leaf_manifest["identity"]["data"].get("observation_stride", 1)
    target.attrs.update(task=source.attrs["task"], episode_id=source.attrs["episode_id"])
    writer = _EpisodeWriter(target, source, model["width"])
    forests = {}
    for name in ("conditioning", "execution", "prefix"):
        target.require_group("blocks/" + name)
    for key in sorted(source["conditioning"], key=int):
        check_stop()
        writer.append(forests, writer.block("conditioning", key))

    def roots():
        return [ref for kind in sorted(forests) for ref in forests[kind]]

    snapshots = [roots()]
    execution = sorted(source["execution"], key=int)
    if [int(key) for key in execution] != list(range(0, len(execution) * block_size, block_size)):
        raise ValueError("Execution cache blocks are not contiguous retained-frame blocks")
    for key in execution:
        check_stop()
        writer.append(forests, writer.block("execution", key))
        snapshots.append(roots())
    prefixes = {}
    for key in sorted(source["prefix"], key=lambda key: tuple(map(int, key.split("_")))):
        check_stop()
        prefixes[key] = [ref for _, ref in writer.block("prefix", key)]
    decisions = target.create_group("decisions")
    used_prefixes = set()
    for key in sorted(source["current"], key=int):
        check_stop()
        index = int(key)
        if index % observation_stride:
            raise ValueError("Cached decision is not on the retained observation cadence")
        current = source["current"][key]
        if int(current.attrs["length"]) and not np.all(np.asarray(current["time_anchor"]) == index):
            raise ValueError("Hierarchy requires the prepared contiguous source-local frame clock")
        nfull, pending = divmod(index // observation_stride, block_size)
        if nfull >= len(snapshots):
            raise ValueError("Cached current decision exceeds completed execution evidence")
        prefix_key = f"{nfull * block_size}_{pending}" if pending else ""
        records = list(snapshots[nfull])
        if pending:
            records += prefixes[prefix_key]
            used_prefixes.add(prefix_key)
        if any(record.record.source == "execution" and record.record.time_max >= index for record in records):
            raise ValueError("Persisted hierarchy would include current/future execution evidence")
        group = decisions.create_group(key)
        group.attrs.update(nfull=nfull, prefix_key=prefix_key,
                           history_leaf_count=sum(record.record.count for record in records))
        group.create_dataset("roots", data=np.asarray([record.index for record in records], dtype="i8"))
    if used_prefixes != set(prefixes):
        raise ValueError("Causal prefix variants do not match the cached decision schedule")
    writer.finish()
    target.attrs["identity_sha256"] = digest(identity)


def _recover(path, identity, kind):
    receipt_path = path.with_suffix(".complete.json")
    if not receipt_path.exists():
        if path.exists():
            raise ValueError(f"Immutable hierarchy file has no receipt: {path}")
        return None
    receipt = read_json(receipt_path)
    temporary = path.with_suffix(".partial.h5")
    candidate = path if path.exists() else temporary
    require_file(candidate, receipt, identity=identity, kind=kind)
    if candidate != path:
        os.replace(candidate, path)
    return receipt


def _build_task(leaf_root, root, task, identity, leaf_manifest, check_stop):
    with locked(root / f"{task}.lock"):
        _check_leaf_file(leaf_root, task, leaf_manifest, full_hash=True)
        destination = root / f"{task}.h5"
        if _recover(destination, identity, "hierarchy_cache_task") is not None:
            return
        work = root / f".{task}.episodes"
        work.mkdir(exist_ok=True)
        episodes = []
        with h5py.File(leaf_root / f"{task}.h5", "r") as source:
            for key in sorted(source):
                check_stop()
                path = work / f"{key}.h5"
                receipt = _recover(path, identity, "hierarchy_cache_episode")
                if receipt is None:
                    temporary = path.with_suffix(".partial.h5")
                    with h5py.File(temporary, "w", libver="latest") as out:
                        write_episode_hierarchy(source[key], out.create_group(key), identity, leaf_manifest, check_stop)
                        out.flush()
                    check_stop()
                    receipt = {"kind": "hierarchy_cache_episode", "status": "complete",
                               "identity_sha256": digest(identity), "task": task, "episode_key": key,
                               "sha256": file_sha256(temporary), "bytes": temporary.stat().st_size}
                    atomic_json(receipt, path.with_suffix(".complete.json"))
                    os.replace(temporary, path)
                if receipt.get("task") != task or receipt.get("episode_key") != key:
                    raise ValueError("Hierarchy episode receipt differs from the source episode")
                episodes.append((key, path))
                print(json.dumps({"event": "hierarchy_episode_complete", "task": task, "episode_key": key,
                                  "episodes_done": len(episodes), "episodes_total": len(source)}), flush=True)
        temporary = destination.with_suffix(".partial.h5")
        with h5py.File(temporary, "w", libver="latest") as out:
            out.attrs.update(identity_sha256=digest(identity), leaf_source=str(leaf_root / f"{task}.h5"))
            for key, path in episodes:
                check_stop()
                with h5py.File(path, "r") as episode:
                    episode.copy(key, out)
            out.flush()
        _check_leaf_file(leaf_root, task, leaf_manifest, full_hash=True)
        if hierarchy_identity(leaf_root) != identity:
            raise ValueError("Hierarchy source/implementation changed during construction")
        check_stop()
        receipt = {"kind": "hierarchy_cache_task", "status": "complete", "task": task,
                   "identity_sha256": digest(identity), "sha256": file_sha256(temporary),
                   "bytes": temporary.stat().st_size, "episodes": len(episodes),
                   "leaf_sha256": leaf_manifest["shards"][f"{task}.h5"]["sha256"]}
        atomic_json(receipt, destination.with_suffix(".complete.json"))
        os.replace(temporary, destination)
        for _, path in episodes:
            path.unlink()
            path.with_suffix(".complete.json").unlink()
        try:
            work.rmdir()
        except OSError:
            pass


def build_hierarchy_cache(leaf_root, *, output_root=None, task=None, task_index=None, threads=1, stop_file=None):
    """Build one/all independent task shards; matching completed episodes resume."""
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    leaf_root = Path(leaf_root).resolve()
    leaf_manifest = _leaf_manifest(leaf_root)
    identity = hierarchy_identity(leaf_root, leaf_manifest)
    root = _location(leaf_root, output_root, leaf_manifest)
    if task is not None and task_index is not None:
        raise ValueError("Choose task or task_index")
    tasks = leaf_manifest["tasks"]
    if task_index is not None:
        if not 0 <= task_index < len(tasks):
            raise ValueError("task_index is outside the configured task list")
        task = tasks[task_index]
    if task is not None and task not in tasks:
        raise ValueError("Task is absent from the encoder cache")
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / ".identity.lock", wait=True):
        record = root / "identity.json"
        if record.exists() and read_json(record) != identity:
            raise ValueError("Output directory contains a different hierarchy identity")
        if not record.exists():
            atomic_json(identity, record)

    def check_stop():
        if stop_file and Path(stop_file).exists():
            raise HierarchyInterrupted("Hierarchy preparation stopped; completed episode commits are reusable")

    started = time.perf_counter()
    for name in [task] if task else tasks:
        check_stop()
        _build_task(leaf_root, root, name, identity, leaf_manifest, check_stop)
    with locked(root / ".finalize.lock", wait=True):
        missing = [name for name in tasks if not (root / f"{name}.h5").is_file()
                   or not (root / f"{name}.complete.json").is_file()]
        if not missing:
            shards = {}
            for name in tasks:
                path = root / f"{name}.h5"
                receipt = read_json(root / f"{name}.complete.json")
                require_file(path, receipt, identity=identity, kind="hierarchy_cache_task")
                _check_leaf_file(leaf_root, name, leaf_manifest)
                if (receipt.get("task") != name
                        or receipt.get("episodes") != leaf_manifest["shards"][path.name]["episodes"]
                        or receipt.get("leaf_sha256") != leaf_manifest["shards"][path.name]["sha256"]):
                    raise ValueError("Hierarchy shard was built over different leaf bytes")
                shards[path.name] = {key: receipt[key] for key in ("sha256", "bytes", "episodes")}
            if hierarchy_identity(leaf_root) != identity:
                raise ValueError("Hierarchy source changed before publication")
            manifest = root / "manifest.json"
            if manifest.exists():
                previous = read_json(manifest)
                if previous["identity"] != identity or previous["shards"] != shards:
                    raise ValueError("Completed hierarchy manifest changed")
            else:
                atomic_json({"format": FORMAT, "identity": identity, "status": "complete", "tasks": tasks,
                             "shards": shards, "source": source_revision(),
                             "finalizing_job_seconds": time.perf_counter() - started}, manifest)
            atomic_json({"status": "complete", "identity_sha256": digest(identity),
                         "manifest_sha256": file_sha256(manifest)}, root / "complete.json")
        print(json.dumps({"event": "hierarchy_complete" if not missing else "hierarchy_tasks_pending",
                          "hierarchy_cache_root": str(root), "missing_tasks": missing}), flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--task")
    choice.add_argument("--task-index", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--stop-file", type=Path, default=os.environ.get("STMEM_STOP_FILE"))
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(hierarchy_status(args.cache_root, args.output_root), indent=2))
        return
    try:
        build_hierarchy_cache(args.cache_root, output_root=args.output_root, task=args.task,
                              task_index=args.task_index, threads=args.threads, stop_file=args.stop_file)
    except HierarchyInterrupted as error:
        print(json.dumps({"event": "hierarchy_stopped", "reason": str(error)}), flush=True)
        raise SystemExit(75) from error


if __name__ == "__main__":
    main()
