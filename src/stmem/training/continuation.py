"""Explicit open-ended teacher continuation without changing saved provenance."""

from stmem.training.forks import schedule_steps


def continuation_record(cfg, saved, *, requested):
    """Retain the original decay and record the removal of its stopping budget.

    This changes how long we train, not the optimizer or its schedule. Requiring
    the flag on every resume prevents an old launch command from silently
    reinstating the original endpoint, even before the schedule has finished.
    """
    previous = saved.get("continuation")
    if not requested:
        if previous is not None:
            raise ValueError(
                "This checkpoint uses open-ended training; resume with --train-until-stopped"
            )
        return None
    training = cfg["training"]
    expected = {
        "mode": "constant_lr_tail_v1",
        "original_steps": training["steps"],
        "schedule_steps": schedule_steps(training),
        "final_lr": 0.1 * training["lr"],
    }
    if previous is None:
        return {**expected, "enabled_at_step": saved["step"]}
    if not isinstance(previous, dict) or set(previous) != {*expected, "enabled_at_step"}:
        raise ValueError("Invalid open-ended training continuation record")
    if any(previous[key] != value for key, value in expected.items()):
        raise ValueError("Continuation schedule disagrees with the saved training configuration")
    enabled_at = previous["enabled_at_step"]
    if isinstance(enabled_at, bool) or not isinstance(enabled_at, int) or not 0 <= enabled_at <= saved["step"]:
        raise ValueError("Invalid continuation enabled_at_step")
    return dict(previous)
