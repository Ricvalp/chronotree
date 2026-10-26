"""Explicit matched teacher continuations; ordinary resume remains strict."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from stmem.config import digest, validate_config

COMMAND_FLAG = "condition_on_previous_joint_commands"


def load_fork_checkpoint(path, expected_sha256: str) -> dict:
    """Hash and load one open file so an atomic last.pt replacement cannot race."""
    from stmem.training.runtime import load_checkpoint

    with open(path, "rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
            raise ValueError("Teacher fork checkpoint SHA256 mismatch at loading")
        stream.seek(0)
        return load_checkpoint(stream)


def schedule_steps(training: dict) -> int:
    """A bounded continuation keeps the parent run's cosine-decay horizon."""
    return training.get("schedule_steps", training["steps"])


def teacher_fork_config(
    saved: dict,
    *,
    run_root,
    additional_steps: int = 10000,
    condition_on_previous_joint_commands: bool,
) -> dict:
    if saved["stage"] != "teacher":
        raise ValueError("Only teacher checkpoints support matched continuations")
    if not isinstance(additional_steps, int) or isinstance(additional_steps, bool) or additional_steps < 1:
        raise ValueError("additional_steps must be a positive integer")
    cfg = deepcopy(saved["config"])
    start = saved["step"]
    horizon = schedule_steps(cfg["training"])
    end = start + additional_steps
    if end > horizon:
        raise ValueError("Continuation would exceed the original learning-rate schedule")
    cfg["paths"]["run_root"] = str(run_root)
    cfg["model"][COMMAND_FLAG] = condition_on_previous_joint_commands
    cfg["training"].update(
        steps=end,
        schedule_steps=horizon,
        checkpoint_steps=sorted({start + max(1, additional_steps // 2), end}),
    )
    validate_config(cfg)
    validate_teacher_fork(
        saved,
        cfg,
        {**saved["provenance"], "model": digest(cfg["model"]), "training": digest(cfg["training"])},
    )
    return cfg


def validate_teacher_fork(saved: dict, cfg: dict, provenance: dict) -> None:
    """Permit only the declared command ablation and bounded save/end settings.

    All Adam, RNG, sampler, normalization and scheduler state is restored by the
    trainer. Data identity and rank assignment must match the actual new job.
    """
    if saved["stage"] != "teacher" or cfg.get("stage") != "teacher":
        raise ValueError("Only teacher checkpoints support matched continuations")
    parent = saved["config"]
    recorded = saved["provenance"]
    if recorded.get("model") != digest(parent["model"]) or recorded.get("training") != digest(
        parent["training"]
    ):
        raise ValueError("Parent checkpoint configuration disagrees with its provenance")
    old_model, new_model = deepcopy(parent["model"]), deepcopy(cfg["model"])
    for model in (old_model, new_model):
        flag = model.pop(COMMAND_FLAG, True)
        if not isinstance(flag, bool):
            raise ValueError(f"model.{COMMAND_FLAG} must be boolean")
    if old_model != new_model:
        raise ValueError("Teacher fork may change only previous-joint-command conditioning in the model")
    for key in ("data", "geometry", "seed", "debug"):
        if parent.get(key) != cfg.get(key):
            raise ValueError(f"Teacher fork {key} must match the parent checkpoint")
    old_training, new_training = deepcopy(parent["training"]), deepcopy(cfg["training"])
    for training in (old_training, new_training):
        training["schedule_steps"] = schedule_steps(training)
        training.pop("steps")
        training.pop("checkpoint_steps", None)
    if old_training != new_training:
        raise ValueError("Teacher fork training settings and learning-rate schedule must match the parent")
    start, end = saved["step"], cfg["training"]["steps"]
    if not 0 <= start < end <= schedule_steps(cfg["training"]):
        raise ValueError("Teacher fork endpoint must follow its source step within the original schedule")
    expected = {
        **recorded,
        "model": digest(cfg["model"]),
        "training": digest(cfg["training"]),
    }
    if expected != provenance:
        raise ValueError("Teacher fork data/dependencies/world size/sampler provenance mismatch")
    world = provenance["world_size"]
    if len(saved["rng_by_rank"]) != world:
        raise ValueError("Teacher fork requires one parent RNG state per rank")
    expected_sampler = {
        "next_global_position": start * provenance["effective_batch"],
        "boundary": "optimizer_step",
        "version": provenance["sampler"],
        "rank_assignment": provenance["rank_assignment"],
    }
    if saved.get("sampler") != expected_sampler:
        raise ValueError("Teacher fork requires the parent's exact optimizer-step sampler boundary")
