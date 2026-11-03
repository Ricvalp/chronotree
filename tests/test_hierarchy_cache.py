"""Synthetic invariants for parent-only persisted hierarchies and causal roots."""

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from stmem.config import digest
from stmem.data.cache import FrozenCache, episode_key, write_records
from stmem.data import hierarchy_cache as hc
from stmem.memory.records import leaf_record
from stmem.training.checkpoints import file_sha256


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def leafs(task, episode, source, frames, *, completed=True):
    result = []
    for frame in frames:
        for token, modality in enumerate(("visual", "visual", "event")):
            content = torch.arange(4).float() + frame + token * 0.25 + (0 if completed else 1000)
            anchor = torch.tensor([frame * 0.1, token * 0.01, 0.05])
            result.append(leaf_record(
                f"{task}/{episode}/{source}/{frame}/{token}/{completed}", content, anchor,
                float(frame), source, modality, completed=completed,
            ))
    return result


@pytest.fixture
def leaf_cache(tmp_path):
    torch.set_num_threads(1)
    root = tmp_path / "leaves"
    root.mkdir()
    identity = {"format": 2, "data": {"tasks": ["TaskA", "TaskB"], "observation_stride": 2},
                "model_encoder": {"width": 4, "block_size": 2}, "synthetic_diagnostic": True}
    shards = {}
    for task in identity["data"]["tasks"]:
        path = root / f"{task}.h5"
        with h5py.File(path, "w") as file:
            file.attrs["identity_sha256"] = digest(identity)
            for episode in ("episode_0", "episode_1"):
                group = file.create_group(episode_key(task, episode))
                group.attrs.update(task=task, episode_id=episode)
                group.create_dataset("text", data=np.arange(4, dtype="f4"))
                for name in ("conditioning", "execution", "prefix", "current"):
                    group.create_group(name)
                for key, frames in (("0", [0, 2]), ("2", [4])):
                    write_records(group["conditioning"].create_group(key), leafs(task, episode, "conditioning", frames))
                for key, frames in (("0", [0, 2]), ("2", [4, 6])):
                    write_records(group["execution"].create_group(key), leafs(task, episode, "execution", frames))
                for key, frames in (("0_1", [0]), ("2_1", [4])):
                    write_records(group["prefix"].create_group(key),
                                  leafs(task, episode, "execution", frames, completed=False))
                for index in (0, 2, 4, 6, 8):
                    write_records(group["current"].create_group(str(index)), leafs(task, episode, "execution", [index]))
        entry = {"sha256": file_sha256(path), "bytes": path.stat().st_size, "episodes": 2}
        shards[path.name] = entry
        write_json(path.with_suffix(".complete.json"), {
            "kind": "encoder_cache_task", "status": "complete", "task": task,
            "identity_sha256": digest(identity), **entry,
        })
    write_json(root / "manifest.json", {"identity": identity, "status": "complete",
                                       "tasks": identity["data"]["tasks"], "shards": shards})
    return root


def reference(root, task, episode, index):
    cache = FrozenCache.__new__(FrozenCache)
    cache.cfg, cache.root = {"model": {"block_size": 2}}, root
    return cache.evidence(SimpleNamespace(
        task=task, episode_id=episode, index=index, execution_prefix=[None] * (index // 2),
        conditioning=[SimpleNamespace(time=4.0)], current=SimpleNamespace(time=float(index)),
    ))


def assert_tree(nodes, index, record):
    def scalar(name):
        value = nodes[name][index]
        return value.decode() if isinstance(value, bytes) else value

    for field in (*hc.STRING_FIELDS, *hc.SCALAR_FIELDS):
        assert scalar(field) == getattr(record, field)
    for field in hc.PAYLOAD_FIELDS:
        np.testing.assert_array_equal(nodes[field][index], getattr(record, field).numpy())
    children = nodes["children"][index]
    if record.children:
        assert all(child < index for child in children)
        result = []
        for child, expected in zip(children, record.children):
            result += assert_tree(nodes, int(child), expected)
        return result
    assert children.tolist() == [-1, -1]
    return [record.node_id]


def refresh_leaf_manifest(root, task):
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / f"{task}.h5"
    manifest["shards"][path.name].update(sha256=file_sha256(path), bytes=path.stat().st_size)
    receipt = json.loads(path.with_suffix(".complete.json").read_text())
    receipt.update(manifest["shards"][path.name])
    write_json(path.with_suffix(".complete.json"), receipt)
    write_json(root / "manifest.json", manifest)


def test_parent_only_vds_survives_episode_assembly_and_atomic_rename(leaf_cache):
    original = {path.name: file_sha256(path) for path in leaf_cache.iterdir() if path.is_file()}
    root = hc.build_hierarchy_cache(leaf_cache)
    manifest = hc.validate_hierarchy_cache(leaf_cache, root)
    assert manifest["tasks"] == ["TaskA", "TaskB"]
    assert hc.hierarchy_status(leaf_cache, root)["status"] == "complete"
    for task in manifest["tasks"]:
        with h5py.File(root / f"{task}.h5") as file:
            for episode in ("episode_0", "episode_1"):
                group = file[episode_key(task, episode)]
                nodes = group["nodes"]
                assert nodes["content_sum"].is_virtual
                assert nodes["max_content"].is_virtual
                assert len(group["parent_payload/content_sum"]) == group.attrs["parent_count"]
                assert group.attrs["parent_count"] < group.attrs["node_count"]
                for index in (0, 2, 4, 6, 8):
                    evidence = reference(leaf_cache, task, episode, index)
                    query = group[f"decisions/{index}"]
                    roots = query["roots"][:]
                    assert len(roots) == len(evidence.archive.roots)
                    found = []
                    for node, record in zip(roots, evidence.archive.roots):
                        found += assert_tree(nodes, int(node), record)
                    assert Counter(found) == Counter(record.node_id for record in evidence.archive.leaves)
                    assert all(count == 1 for count in Counter(found).values())
                    assert query.attrs["history_leaf_count"] == len(evidence.archive.leaves)
                    assert query.attrs["nfull"] == index // 4
    assert {name: file_sha256(leaf_cache / name) for name in original} == original
    assert not list(root.glob(".*.episodes"))


def test_task_arrays_publish_only_after_all_tasks_and_resume_without_rewrite(leaf_cache):
    root = hc.build_hierarchy_cache(leaf_cache, task_index=0)
    assert hc.hierarchy_status(leaf_cache, root)["missing_tasks"] == ["TaskB"]
    assert not (root / "manifest.json").exists()
    before = (file_sha256(root / "TaskA.h5"), (root / "TaskA.h5").stat().st_mtime_ns)
    hc.build_hierarchy_cache(leaf_cache, output_root=root, task="TaskA")
    assert before == (file_sha256(root / "TaskA.h5"), (root / "TaskA.h5").stat().st_mtime_ns)
    hc.build_hierarchy_cache(leaf_cache, output_root=root, task_index=1)
    assert hc.validate_hierarchy_cache(leaf_cache, root)["status"] == "complete"
    (root / "complete.json").unlink()
    assert hc.hierarchy_status(leaf_cache, root)["status"] == "incomplete"
    with pytest.raises(FileNotFoundError):
        hc.validate_hierarchy_cache(leaf_cache, root)
    hc.build_hierarchy_cache(leaf_cache, output_root=root, task="TaskB")
    assert hc.hierarchy_status(leaf_cache, root)["status"] == "complete"
    (root / "TaskB.h5").rename(root / "TaskB.moved.h5")
    status = hc.hierarchy_status(leaf_cache, root)
    assert status["status"] == "incomplete" and status["missing_tasks"] == ["TaskB"]


def test_missing_virtual_source_is_rejected_instead_of_returning_fill_data(leaf_cache):
    root = hc.build_hierarchy_cache(leaf_cache)
    (leaf_cache / "TaskA.h5").rename(leaf_cache / "moved_TaskA.h5")
    with pytest.raises(ValueError, match="virtual-dataset leaf source"):
        hc.validate_hierarchy_cache(leaf_cache, root)


def test_corrupted_leaf_bytes_rejected_even_if_size_matches(leaf_cache):
    path = leaf_cache / "TaskA.h5"
    with h5py.File(path, "r+") as file:
        values = file[episode_key("TaskA", "episode_0") + "/execution/0/content_sum"]
        values[0, 0] += 1
    with pytest.raises(ValueError, match="Corrupted|changed"):
        hc.build_hierarchy_cache(leaf_cache, task="TaskA")


def test_corrupted_derived_shard_rejected_on_resume(leaf_cache):
    root = hc.build_hierarchy_cache(leaf_cache, task="TaskA")
    path = root / "TaskA.h5"
    with h5py.File(path, "r+") as file:
        values = file[episode_key("TaskA", "episode_0") + "/parent_payload/content_sum"]
        values[0, 0] += 1
    with pytest.raises(ValueError, match="corrupted"):
        hc.build_hierarchy_cache(leaf_cache, output_root=root, task="TaskA")


def test_builder_code_change_invalidates_derived_cache_only(leaf_cache, monkeypatch):
    root = hc.build_hierarchy_cache(leaf_cache)
    monkeypatch.setattr(hc, "hierarchy_code_fingerprint", lambda: "changed builder")
    assert hc.default_hierarchy_root(leaf_cache) != root
    with pytest.raises(ValueError, match="Stale"):
        hc.validate_hierarchy_cache(leaf_cache, root)
    with pytest.raises(ValueError, match="different hierarchy identity"):
        hc.build_hierarchy_cache(leaf_cache, output_root=root)


def test_interruption_reuses_committed_episode_and_never_publishes_partial(leaf_cache, tmp_path, monkeypatch):
    stop = tmp_path / "stop"
    original = hc.atomic_json
    committed = []

    def stop_after_commit(value, path):
        original(value, path)
        if value.get("kind") == "hierarchy_cache_episode" and not committed:
            committed.append(value["episode_key"])
            stop.touch()

    monkeypatch.setattr(hc, "atomic_json", stop_after_commit)
    with pytest.raises(hc.HierarchyInterrupted):
        hc.build_hierarchy_cache(leaf_cache, task="TaskA", stop_file=stop)
    root = hc.default_hierarchy_root(leaf_cache)
    assert not (root / "TaskA.h5").exists() and not (root / "manifest.json").exists()
    assert list((root / ".TaskA.episodes").glob("*.complete.json"))
    stop.unlink()
    original_writer = hc.write_episode_hierarchy

    def no_rewrite(source, *args, **kwargs):
        assert source.name.strip("/") != committed[0]
        return original_writer(source, *args, **kwargs)

    monkeypatch.setattr(hc, "write_episode_hierarchy", no_rewrite)
    hc.build_hierarchy_cache(leaf_cache, task="TaskA", stop_file=stop)
    assert (root / "TaskA.h5").exists()


def test_invalid_source_or_completion_flags_are_not_accepted(leaf_cache):
    with h5py.File(leaf_cache / "TaskA.h5", "r+") as file:
        file[episode_key("TaskA", "episode_0") + "/prefix/0_1/completed"][0] = True
    refresh_leaf_manifest(leaf_cache, "TaskA")
    with pytest.raises(ValueError, match="source/completion"):
        hc.build_hierarchy_cache(leaf_cache, task="TaskA")


def test_receipt_backed_episode_partial_recovers_after_interrupted_rename(leaf_cache, monkeypatch):
    original = hc.os.replace
    interrupted = []

    def interrupt_episode_publish(source, destination):
        if str(source).endswith(".partial.h5") and ".TaskA.episodes" in str(source) and not interrupted:
            interrupted.append(str(source))
            raise OSError("interrupted episode publication")
        return original(source, destination)

    monkeypatch.setattr(hc.os, "replace", interrupt_episode_publish)
    with pytest.raises(OSError, match="interrupted episode publication"):
        hc.build_hierarchy_cache(leaf_cache, task="TaskA")
    root = hc.default_hierarchy_root(leaf_cache)
    assert list((root / ".TaskA.episodes").glob("*.partial.h5"))
    assert not (root / "TaskA.h5").exists()
    hc.build_hierarchy_cache(leaf_cache, task="TaskA")
    assert (root / "TaskA.h5").exists()
    assert not list(root.glob(".*.episodes"))


def test_empty_conditioning_history_and_current_records_are_supported(leaf_cache):
    with h5py.File(leaf_cache / "TaskA.h5", "r+") as file:
        for group in file.values():
            for name in ("conditioning", "execution", "prefix", "current"):
                for key in list(group[name]):
                    del group[name][key]
                    write_records(group[name].create_group(key), [])
    refresh_leaf_manifest(leaf_cache, "TaskA")
    root = hc.build_hierarchy_cache(leaf_cache, task="TaskA")
    with h5py.File(root / "TaskA.h5") as file:
        for group in file.values():
            assert group.attrs["node_count"] == 0
            for query in group["decisions"].values():
                assert len(query["roots"]) == 0 and query.attrs["history_leaf_count"] == 0
