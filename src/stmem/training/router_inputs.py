"""Replay fixed initial-label decisions to add causal router input features.

This CPU-only preprocessing never evaluates a teacher or changes a gain label.
Every replayed query and candidate must exactly match the original label row.
Current tokens are stored once per context; TREE child descriptors stay paired
with their own geometry. Node IDs and selection IDs are bookkeeping only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from stmem.config import dependency, digest, load_config
from stmem.data.cache import FrozenCache
from stmem.data.hierarchy_cache import validate_hierarchy_cache
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.label_data import LabelDataReader, LabelPlan


SCHEMA = 1
ROW_FIELDS = ("source_row", "row_to_context", "candidate_id")
CONTEXT_FIELDS = ("context_id", "current", "current_valid", "decision_index", "budget", "partition",
                  "episode", "selection_ids")
LABEL_FIELDS = ("query", "candidate", "gain", "context_id", "partition", "episode", "budget")


def _strings(values):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)


def _feature_source():
    # Changes to unrelated training/launch code do not invalidate replayed input
    # bytes; changes to input construction do require a new sidecar location.
    base = Path(__file__).resolve().parent.parent
    return digest({name: file_sha256(base / name) for name in
                   ("training/router_inputs.py", "training/label_data.py", "training/packed_context.py",
                    "training/cached_loader.py", "models/router.py", "models/metadata_adapter.py")})


def _label_data(path):
    with h5py.File(path, "r") as source:
        identity = json.loads(source.attrs["identity"])
        if not source.attrs.get("complete") or identity.get("schema_version") != 2:
            raise ValueError("Router input replay requires a complete schema-2 label shard")
        if identity.get("refresh") is not False or identity.get("router") is not None:
            raise ValueError("Router input replay currently supports fixed INITIAL labels only")
        arrays = {name: np.asarray(source[name]) for name in LABEL_FIELDS}
        attempted = np.asarray(source["attempted_context_ids"])
    rows = len(arrays["gain"])
    if any(len(value) != rows for value in arrays.values()):
        raise ValueError("Original label row lengths disagree")
    expected = np.arange(identity["shard_index"], identity["contexts"], identity["shard_count"])
    if not np.array_equal(np.sort(attempted), expected):
        raise ValueError("Original label shard does not cover its attempted contexts")
    if not np.isin(arrays["context_id"], attempted).all():
        raise ValueError("Original label rows contain unplanned contexts")
    return identity, arrays, attempted


def _identity(cfg, source, original):
    return {"schema_version": SCHEMA, "kind": "router_architecture_inputs",
            "source_label_sha256": file_sha256(source), "label_identity": original,
            "feature_source": _feature_source(), "width": cfg["model"]["width"],
            "current_tokens": cfg["model"]["current_tokens"],
            "current_schema": "raw_content_plus_read_time_metadata_v1",
            "children_schema": "individual_mean_max_static_metadata_v1",
            "synthetic_diagnostic": bool(cfg["data"].get("synthetic"))}


def _receipt(path):
    path = Path(path)
    with h5py.File(path, "r") as file:
        identity = json.loads(file.attrs["identity"])
        if not file.attrs.get("complete"):
            raise ValueError("Published router input sidecar is incomplete")
        record = {"status": "complete", "kind": "router_architecture_inputs",
                  "sha256": file_sha256(path), "source_label_sha256": identity["source_label_sha256"],
                  "identity": identity, "row_count": len(file["source_row"]),
                  "context_count": len(file["context_id"])}
    atomic_json(record, path.with_suffix(".json"))
    return record


def validate_input_shard(path, source_label_path, width, current_tokens):
    """Validate bytes and row/context bookkeeping without opening encoder caches."""
    path, source_label_path = Path(path), Path(source_label_path)
    original, labels, attempted = _label_data(source_label_path)
    record = json.loads(path.with_suffix(".json").read_text())
    if (record.get("status") != "complete" or record.get("kind") != "router_architecture_inputs"
            or record.get("sha256") != file_sha256(path)):
        raise ValueError(f"Router input sidecar receipt/hash mismatch: {path}")
    with h5py.File(path, "r") as file:
        identity = json.loads(file.attrs["identity"])
        if (not file.attrs.get("complete") or identity != record.get("identity")
                or identity.get("schema_version") != SCHEMA
                or identity.get("source_label_sha256") != file_sha256(source_label_path)
                or identity.get("label_identity") != original
                or identity.get("feature_source") != _feature_source()
                or identity.get("width") != width or identity.get("current_tokens") != current_tokens):
            raise ValueError("Router input sidecar identity differs from the original labels/configuration")
        rows, contexts = len(labels["gain"]), len(attempted)
        shapes = {"source_row": (rows,), "row_to_context": (rows,), "candidate_id": (rows,),
                  "context_id": (contexts,), "current": (contexts, current_tokens, width + 23),
                  "current_valid": (contexts, current_tokens)}
        shapes.update({name: (contexts,) for name in CONTEXT_FIELDS if name not in shapes})
        if original["reader"] == "tree":
            shapes["children"] = (rows, 2, 2 * width + 23)
        elif "children" in file:
            raise ValueError("FLAT sidecars must not contain invented child descriptors")
        if any(name not in file or file[name].shape != shape for name, shape in shapes.items()):
            raise ValueError("Router input sidecar feature shapes disagree")
        if (record.get("row_count") != rows or record.get("context_count") != contexts
                or json.loads(file.attrs["commit"]) != [rows, contexts]):
            raise ValueError("Router input sidecar committed lengths disagree")
        source_rows, indices = np.asarray(file["source_row"]), np.asarray(file["row_to_context"])
        if (not np.issubdtype(source_rows.dtype, np.integer)
                or not np.issubdtype(indices.dtype, np.integer)
                or not np.array_equal(np.sort(source_rows), np.arange(rows))
                or np.any(indices < 0) or np.any(indices >= contexts)):
            raise ValueError("Router input row mapping is invalid")
        context_ids = np.asarray(file["context_id"])
        if (not np.array_equal(np.sort(context_ids), np.sort(attempted))
                or not np.array_equal(context_ids[indices], labels["context_id"][source_rows])):
            raise ValueError("Router input context mapping differs from labels")
        for name in ("budget", "partition", "episode"):
            actual, expected = np.asarray(file[name])[indices], labels[name][source_rows]
            if name == "episode":
                actual, expected = _strings(actual), _strings(expected)
            if not np.array_equal(actual, expected):
                raise ValueError(f"Router input {name} differs from labels")
        pairs = list(zip(context_ids[indices].tolist(), _strings(np.asarray(file["candidate_id"])).tolist()))
        if len(set(pairs)) != rows or any(not name for _, name in pairs):
            raise ValueError("Router input candidate IDs repeat or are empty")
        if file["current_valid"].dtype != np.dtype(bool):
            raise ValueError("Current token validity must be boolean")
        for name in ("current", "children"):
            if name in file:
                for start in range(0, len(file[name]), 256):
                    if not np.isfinite(file[name][start:start + 256]).all():
                        raise ValueError("Nonfinite router input features")
    return record


@contextmanager
def _writer(path, identity, width, current_tokens):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Router input sidecar already has a writer: {path}") from None
        partial = path.with_suffix(".partial.h5")
        if path.exists():
            with h5py.File(path, "r") as file:
                if json.loads(file.attrs["identity"]) != identity or not file.attrs.get("complete"):
                    raise ValueError("Existing router input sidecar has different inputs")
            yield None, partial
            return
        with h5py.File(partial, "a") as file:
            if "identity" in file.attrs:
                if json.loads(file.attrs["identity"]) != identity:
                    raise ValueError("Partial router input sidecar has different inputs")
            else:
                file.attrs["identity"] = json.dumps(identity, sort_keys=True)
                file.attrs["commit"] = json.dumps([0, 0])
                specs = [("source_row", (), "i8"), ("row_to_context", (), "i8"),
                         ("candidate_id", (), h5py.string_dtype()), ("context_id", (), "i8"),
                         ("current", (current_tokens, width + 23), "f4"),
                         ("current_valid", (current_tokens,), "?"), ("decision_index", (), "i8"),
                         ("budget", (), "i8"), ("partition", (), "i1"),
                         ("episode", (), h5py.string_dtype()), ("selection_ids", (), h5py.string_dtype())]
                if identity["label_identity"]["reader"] == "tree":
                    specs.append(("children", (2, 2 * width + 23), "f4"))
                for name, shape, dtype in specs:
                    # Modest chunks also bound write amplification when committing one context.
                    file.create_dataset(name, shape=(0, *shape), maxshape=(None, *shape),
                                        chunks=(32, *shape), dtype=dtype, compression="lzf")
            rows, contexts = json.loads(file.attrs["commit"])
            for name in file:
                size = contexts if name in CONTEXT_FIELDS else rows
                if len(file[name]) < size:
                    raise ValueError("Damaged committed router input sidecar")
                file[name].resize(size, axis=0)
            yield file, partial
            publish = bool(file.attrs.get("complete"))
            if publish:
                file.flush()
                handle = file.id.get_vfd_handle()
                if isinstance(handle, int):
                    os.fsync(handle)
        if publish:
            # Publication and receipt remain under the same single-writer lock.
            os.replace(partial, path)
            _receipt(path)


def _append(file, name, values):
    start = len(file[name])
    file[name].resize(start + len(values), axis=0)
    if len(values):
        file[name][start:] = values


def write_input_shard(cfg, data, cache_root, hierarchy_root, source, output, *, stop_file=None):
    """Replay one already validated cache/shard; useful for bounded CPU diagnostics."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Router inputs must never overwrite original label shards")
    original, labels, attempted = _label_data(source)
    family, width, count = cfg["reader"], cfg["model"]["width"], cfg["model"]["current_tokens"]
    if (original["reader"] != family or original["objective"] != cfg["labels"].get("objective", "velocity")
            or original["integration_steps"] != cfg["labels"].get("integration_steps", 8)):
        raise ValueError("Original labels differ from the requested reader/objective")
    plan = LabelPlan(data, cfg)
    all_items = plan.items()
    if digest(all_items) != original["sample_plan"] or len(all_items) != original["contexts"]:
        raise ValueError("Router input replay sample plan differs from original labels")
    sampling = {key: cfg["labels"][key] for key in
                ("seed", "budgets", "candidates", "exploration_fraction", "split_fraction")}
    if original.get("label_sampling", sampling) != sampling:
        raise ValueError("Router input replay sampler settings differ from labels")
    identity = _identity(cfg, source, original)
    started = time.perf_counter()
    with _writer(output, identity, width, count) as (file, partial):
        if file is None:
            if not output.with_suffix(".json").exists():
                _receipt(output)
            validate_input_shard(output, source, width, count)
            return 0
        done = set(map(int, np.asarray(file["context_id"])))
        if len(done) != len(file["context_id"]) or not done <= set(map(int, attempted)):
            raise ValueError("Duplicate or foreign committed router input contexts")
        with LabelDataReader(data, cfg, cache_root, hierarchy_root) as reader:
            for context in sorted(map(int, attempted)):
                if context in done:
                    continue
                if stop_file and Path(stop_file).exists():
                    return 75
                item = reader.prepare(all_items[context], router_inputs=True, inputs_only=True,
                                      reader_family=family)
                entry = item["readers"][family]
                source_rows = np.flatnonzero(labels["context_id"] == context)
                if len(source_rows) != len(entry["candidate"]):
                    raise ValueError(f"Replayed candidate count differs at context {context}")
                expected_query = np.repeat(item["query"].numpy()[None], len(source_rows), axis=0)
                if (not np.array_equal(expected_query, labels["query"][source_rows])
                        or not np.array_equal(entry["candidate"].numpy(), labels["candidate"][source_rows])):
                    raise ValueError(f"Replayed query/candidate features differ at context {context}")
                for key in ("budget", "partition", "episode"):
                    actual = labels[key][source_rows]
                    if key == "episode":
                        actual = _strings(actual)
                    if not np.all(actual == item[key]):
                        raise ValueError(f"Replayed {key} differs at context {context}")
                expected_partition = "validation" if item["partition"] else "train"
                if original["partition"].get(item["episode"]) != expected_partition:
                    raise ValueError("Replayed episode partition differs from label manifest")
                current = torch.cat((item["current"].content[0], item["current"].metadata[0]), -1).numpy()
                current_valid = item["current"].valid[0].numpy()
                if len(current) > count:
                    raise ValueError("Current token count exceeds the frozen encoder configuration")
                padded = np.zeros((count, width + 23), dtype=np.float32)
                valid = np.zeros(count, dtype=bool)
                padded[:len(current)], valid[:len(current)] = current, current_valid
                position = len(file["context_id"])
                context_values = {
                    "context_id": [context], "current": padded[None], "current_valid": valid[None],
                    "decision_index": [item["decision_index"]], "budget": [item["budget"]],
                    "partition": [item["partition"]], "episode": [item["episode"]],
                    "selection_ids": [json.dumps(entry["frontier_ids"])],
                }
                row_values = {"source_row": source_rows, "row_to_context": np.full(len(source_rows), position),
                              "candidate_id": entry["candidate_ids"]}
                if family == "tree":
                    row_values["children"] = entry["children"].numpy()
                for name, values in {**context_values, **row_values}.items():
                    _append(file, name, values)
                file.flush()
                file.attrs["commit"] = json.dumps([len(file["source_row"]), len(file["context_id"])])
                file.flush()
                done.add(context)
                if len(done) % 100 == 0 or len(done) == len(attempted):
                    print(json.dumps({"event": "router_inputs_progress", "reader": family,
                                      "shard": original["shard_index"], "contexts_completed": len(done),
                                      "contexts_planned": len(attempted), "rows": len(file["source_row"]),
                                      "elapsed_seconds": time.perf_counter() - started}), flush=True)
        if len(file["source_row"]) != len(labels["gain"]):
            raise ValueError("Replayed router input rows do not cover the original labels")
        if file_sha256(source) != identity["source_label_sha256"]:
            raise ValueError("Original label shard changed during input replay")
        file.attrs["complete"] = True
        file.flush()
    validate_input_shard(output, source, width, count)
    return 0


def load_router_inputs(cfg, label_paths, merged):
    """Load verified features in exactly _load_labels' sorted initial-row order."""
    settings = cfg.get("labels", {})
    supplied, sources = settings.get("router_inputs"), settings.get("initial_shards")
    if settings.get("use_refresh", False):
        raise ValueError("Richer router inputs currently require initial-only labels")
    if (not isinstance(supplied, (list, tuple)) or not isinstance(sources, (list, tuple))
            or len(supplied) != len(sources) or len(supplied) != len(label_paths)):
        raise ValueError("labels.router_inputs must match the ordered initial_shards list")
    mapping = {Path(source).resolve(): Path(sidecar) for source, sidecar in zip(sources, supplied)}
    if len(mapping) != len(sources) or len({str(Path(p).resolve()) for p in supplied}) != len(supplied):
        raise ValueError("Duplicate router input/source shard paths")
    rows, currents, masks, children, records, context_offset = [], [], [], [], [], 0
    for source in label_paths:
        path = mapping.get(Path(source).resolve())
        if path is None:
            raise ValueError("Router input shard does not match a loaded label shard")
        record = validate_input_shard(path, source, cfg["model"]["width"], cfg["model"]["current_tokens"])
        _, original, _ = _label_data(source)
        with h5py.File(path, "r") as file:
            source_rows = np.asarray(file["source_row"])
            values = {key: original[key][source_rows] for key in LABEL_FIELDS}
            values["row_to_context"] = np.asarray(file["row_to_context"]) + context_offset
            rows.append(values)
            currents.append(np.asarray(file["current"]))
            masks.append(np.asarray(file["current_valid"]))
            if "children" in file:
                children.append(np.asarray(file["children"]))
            context_offset += len(file["context_id"])
        records.append({"sha256": record["sha256"], "source_label_sha256": record["source_label_sha256"],
                        "identity": record["identity"]})
    joined = {key: np.concatenate([values[key] for values in rows]) for key in rows[0]}
    order = np.argsort(joined["context_id"], kind="stable")
    for key in ("query", "candidate", "gain", "context_id", "partition", "episode"):
        actual, expected = joined[key][order], merged[key]
        if key == "episode":
            actual, expected = _strings(actual), _strings(expected)
        if not np.array_equal(actual, expected):
            raise ValueError(f"Router inputs and merged label {key} rows disagree")
    if children and len(children) != len(rows):
        raise ValueError("Router input shards mix child schemas")
    arrays = {"current": np.concatenate(currents), "current_valid": np.concatenate(masks),
              "row_to_context": joined["row_to_context"][order],
              "children": np.concatenate(children)[order] if children else None}
    return arrays, {"schema_version": SCHEMA, "shards": records}


def generate(cfg, *, shard_index=0, stop_file=None):
    """Validate real-data provenance before CPU replay, without loading policy weights."""
    from stmem.training.data import TrainingData
    from stmem.training.runtime import validate_selection
    from stmem.training.label_experiment import _episode_name

    if cfg["labels"].get("use_refresh", False):
        raise ValueError("Router input replay supports initial-only label datasets")
    sources, outputs = cfg["labels"]["initial_shards"], cfg["labels"]["router_inputs"]
    if len(sources) != len(outputs) or not 0 <= shard_index < len(sources):
        raise ValueError("Invalid router input shard assignment")
    original, _, _ = _label_data(sources[shard_index])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    data = TrainingData(cfg, progress=lambda row: print(json.dumps({"event": "router_inputs_startup", **row}), flush=True))
    teacher = dependency(cfg, "teacher")
    compact = dependency(cfg, "compact")
    for path in (teacher, compact):
        validate_selection(path, debug=bool(cfg.get("debug") and data.synthetic))
    cache = FrozenCache(cfg, teacher, data)
    hierarchy = cfg["training"]["hierarchy_cache_root"]
    validate_hierarchy_cache(cache.root, hierarchy)
    partition = {_episode_name(data, index): data.label_partition(index) for index in range(len(data.refs))}
    expected = {"teacher": file_sha256(teacher), "compact": file_sha256(compact),
                "cache": digest(cache.manifest), "data": data.identity, "partition": partition}
    if any(original.get(key) != value for key, value in expected.items()):
        raise ValueError("Router input replay cache/data/policy/partition differs from original labels")
    return write_input_shard(cfg, data, cache.root, hierarchy, sources[shard_index], outputs[shard_index],
                             stop_file=stop_file)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--stop-file", default=os.environ.get("STMEM_STOP_FILE"))
    args = parser.parse_args(argv)
    raise SystemExit(generate(load_config(args.config), shard_index=args.shard_index, stop_file=args.stop_file))


if __name__ == "__main__":
    main()
