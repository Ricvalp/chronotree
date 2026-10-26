"""Inspect one causal episode prefix with the real encoder and memory hierarchy.

Example:
    python -m stmem.visualization.inspect_encoder --config configs/teacher.yaml \
        --task VideoUnmask --episode 0 --decision 16 --detail-blocks 2 \
        --output runs/visualizations/VideoUnmask_ep0_t16.html

This CPU diagnostic never trains a model, invokes the simulator, or changes
prepared data. Optional checkpoints must be trusted local project artifacts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import torch

from stmem.config import digest, load_config, source_revision
from stmem.data.dataset import EpisodeDataset
from stmem.data.download import TASKS
from stmem.data.paths import alignment_path
from stmem.models.spacetime_encoder import BlockEncoder
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.runtime import load_checkpoint, validate_policy_config


def _episode_key(value):
    if not re.fullmatch(r"(?:episode_)?\d+", str(value)):
        raise argparse.ArgumentTypeError("Episode must be a nonnegative raw ID, e.g. 0 or episode_0")
    return value if str(value).startswith("episode_") else f"episode_{int(value)}"


def _file_identity(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _output_paths(output, inputs):
    """Resolve both artifacts independently and never overwrite a data input."""
    output = Path(output).expanduser()
    destinations = (output.resolve(), output.with_suffix(".json").resolve())
    protected = [Path(path).expanduser().resolve() for path in inputs if path is not None]
    for destination in destinations:
        for source in protected:
            if destination == source or (
                destination.exists() and source.exists() and destination.samefile(source)
            ):
                raise ValueError(f"Inspector output would overwrite an input file: {destination}")
    first, second = destinations
    if first == second or (first.exists() and second.exists() and first.samefile(second)):
        raise ValueError("HTML and JSON outputs must be distinct files")
    return destinations


@contextmanager
def _diagnostic_runtime(seed, threads):
    """Initialize CPU weights without changing the caller's random generators."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        with torch.random.fork_rng(devices=[]):
            # torch.manual_seed also schedules CUDA seeds; this is CPU-only.
            torch.random.default_generator.manual_seed(int(seed))
            yield
    finally:
        torch.set_num_threads(previous_threads)
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _encoder(config):
    model = config["model"]
    return (
        BlockEncoder(
            width=model["width"],
            visual_tokens=model["visual_tokens"],
            block_size=model["block_size"],
            current_tokens=model["current_tokens"],
            variant=model.get("encoder", "joint"),
            checkpoint_blocks=False,
            condition_on_previous_joint_commands=model.get("condition_on_previous_joint_commands", True),
            **model.get("encoder_options", {}),
        )
        .cpu()
        .eval()
        .requires_grad_(False)
    )


def _load_encoder(config, checkpoint):
    encoder = _encoder(config)
    if checkpoint is None:
        return encoder, {
            "encoder_parameters": "configuration_defaults",
            "encoder_label": "Configuration-default geometry; randomly initialized encoder, not a trained model",
            "checkpoint_sha256": None,
            "initialization_seed": int(config.get("seed", 0)),
        }
    checkpoint = Path(checkpoint).expanduser().resolve()
    before = _file_identity(checkpoint)
    state = load_checkpoint(checkpoint)
    if not isinstance(state, dict) or not isinstance(state.get("config"), dict):
        raise ValueError(
            "Inspector checkpoint must include its config and a policy or BlockEncoder state dict"
        )
    validate_policy_config(state["config"], config)
    weights = state.get("encoder", state.get("model"))
    if not isinstance(weights, dict):
        raise ValueError("Checkpoint must provide encoder or model weights")
    if any(key.startswith("encoder.") for key in weights):
        weights = {
            key.removeprefix("encoder."): value
            for key, value in weights.items()
            if key.startswith("encoder.")
        }
    encoder.load_state_dict(weights, strict=True)
    checkpoint_hash = file_sha256(checkpoint)
    if _file_identity(checkpoint) != before:
        raise RuntimeError("Checkpoint changed while loading; retry with a completed immutable checkpoint")
    return encoder, {
        "encoder_parameters": "checkpoint",
        "encoder_label": "Checkpoint encoder weights; diagnostic visualization, not a policy evaluation",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_stage": state.get("stage"),
        "checkpoint_step": state.get("step"),
        "checkpoint_config_sha256": digest(state["config"]),
        "checkpoint_debug": bool(state["config"].get("debug", False)),
        "checkpoint_synthetic_debug_only": bool(state["config"].get("data", {}).get("synthetic", False)),
    }


def inspect_encoder(
    config,
    *,
    task,
    episode_key,
    decision_index,
    detail_blocks=2,
    prepared=None,
    alignment=None,
    checkpoint=None,
    threads=1,
):
    """Return a JSON-safe diagnostic without opening other task files.

    EpisodeDataset indexes small source masks within the selected task file;
    only the requested episode's point clouds and robot state are loaded.
    """
    config = copy.deepcopy(config)
    if task not in TASKS:
        raise ValueError(f"Unknown benchmark task: {task}")
    episode_key = _episode_key(str(episode_key))
    if isinstance(decision_index, bool) or not isinstance(decision_index, int) or decision_index < 0:
        raise ValueError("decision_index must be a nonnegative execution-frame index")
    if isinstance(detail_blocks, bool) or not isinstance(detail_blocks, int) or detail_blocks < 0:
        raise ValueError("detail_blocks must be nonnegative")
    if isinstance(threads, bool) or threads not in range(1, 5):
        raise ValueError("threads must be between one and four")
    path = (
        Path(prepared)
        if prepared is not None
        else (
            Path(config["paths"]["data_root"])
            / config["data"].get("prepared_subdir", "prepared_balanced_v2")
            / f"{task}.h5"
        )
    )
    if path.name.endswith(".partial.h5") or path.resolve().name.endswith(".partial.h5"):
        raise ValueError("An in-progress .partial.h5 file cannot be inspected; wait for the final task file")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Completed prepared task file not found: {path}; preparation may still be running"
        )
    alignment = (
        Path(alignment).expanduser().resolve() if alignment is not None else alignment_path(config).resolve()
    )
    if not alignment.is_file():
        raise FileNotFoundError(
            f"Verified alignment registry not found: {alignment}; specify --alignment if relocated"
        )
    prepared_identity, alignment_identity = _file_identity(path), _file_identity(alignment)
    started = time.perf_counter()
    dataset = EpisodeDataset(path, tasks=[task], horizon=config["model"]["horizon"], alignment_path=alignment)
    dataset.validate_preprocessing(config)
    matches = [
        index for index, ref in enumerate(dataset.episodes) if ref.task == task and ref.key == episode_key
    ]
    if not matches:
        available = ", ".join(ref.key for ref in dataset.episodes)
        raise ValueError(f"Raw episode {episode_key} is absent from {path.name}; available: {available}")
    episode = dataset.load_episode(matches[0])
    if decision_index >= len(episode.execution):
        raise ValueError(
            f"Decision {decision_index} is outside execution frames 0..{len(episode.execution) - 1}"
        )
    # Use the existing fail-closed alignment contract before exposing previous commands.
    # Its target arrays are discarded and never included in the visualization payload.
    episode.decision(decision_index, config["model"]["horizon"])
    prepared_hash, alignment_hash = file_sha256(path), file_sha256(alignment)
    if _file_identity(path) != prepared_identity or _file_identity(alignment) != alignment_identity:
        raise RuntimeError(
            "Prepared data or alignment changed while loading; retry after preparation finishes"
        )
    from stmem.visualization.encoder_structure import build_structure

    with _diagnostic_runtime(config.get("seed", 0), threads), torch.no_grad():
        encoder, encoder_provenance = _load_encoder(config, checkpoint)
        payload = build_structure(episode, encoder, decision_index, detail_blocks=detail_blocks)
    payload.setdefault("provenance", {}).update(
        {
            **encoder_provenance,
            "diagnostic_only": True,
            "device": "cpu",
            "threads": threads,
            "task": task,
            "episode_id": episode_key,
            "decision_index": decision_index,
            "conditioning_frames": len(episode.conditioning),
            "execution_prefix_frames": decision_index,
            "execution_frames_in_episode": len(episode.execution),
            "detail_blocks_per_source": detail_blocks,
            "prepared_path": str(path),
            "prepared_sha256": prepared_hash,
            "prepared_bytes": prepared_identity[2],
            "prepared_manifest": dataset.episodes[matches[0]].metadata,
            "alignment_path": str(alignment),
            "alignment_sha256": alignment_hash,
            "alignment_convention": episode.alignment,
            "alignment_record": dataset.alignment,
            "config": config,
            "config_sha256": digest(config),
            "source_revision": source_revision(),
            "elapsed_seconds": time.perf_counter() - started,
            "causal_scope": "All conditioning frames and execution strictly before the current frame; current encoded separately",
        }
    )
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument(
        "--episode",
        default="episode_0",
        type=_episode_key,
        help="Raw episode ID or episode_N; not ordinal position",
    )
    parser.add_argument("--decision", type=int, default=16, help="Current execution-frame index")
    parser.add_argument(
        "--detail-blocks",
        type=int,
        default=2,
        help="Last N historical blocks per source with full geometric details",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Trusted local policy or BlockEncoder checkpoint, including its config",
    )
    parser.add_argument("--prepared", type=Path, help="Explicit completed task HDF5 file")
    parser.add_argument("--alignment", type=Path, help="Explicit verified alignment registry")
    parser.add_argument("--data-root", type=Path, help="Override configuration data root")
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args(argv)
    if args.decision < 0 or args.detail_blocks < 0:
        parser.error("--decision and --detail-blocks must be nonnegative")
    if args.output.suffix.lower() not in (".html", ".htm"):
        parser.error("--output must end in .html or .htm; a neighboring .json is written automatically")
    config = load_config(args.config)
    if args.data_root is not None:
        config.setdefault("paths", {})["data_root"] = str(args.data_root)
    payload = inspect_encoder(
        config,
        task=args.task,
        episode_key=args.episode,
        decision_index=args.decision,
        detail_blocks=args.detail_blocks,
        prepared=args.prepared,
        alignment=args.alignment,
        checkpoint=args.checkpoint,
        threads=args.threads,
    )
    provenance = payload["provenance"]
    output, json_output = _output_paths(
        args.output,
        [
            args.config,
            provenance["prepared_path"],
            provenance["alignment_path"],
            provenance.get("checkpoint_path"),
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    from stmem.visualization.structure_html import write_structure_html

    write_structure_html(payload, output)
    atomic_json(payload, json_output)
    print(
        json.dumps(
            {
                "html": str(output),
                "json": str(json_output),
                "encoder_parameters": payload["provenance"]["encoder_parameters"],
                "task": args.task,
                "episode_id": args.episode,
                "decision_index": args.decision,
            },
            indent=2,
        )
    )
    return payload


if __name__ == "__main__":
    main()
