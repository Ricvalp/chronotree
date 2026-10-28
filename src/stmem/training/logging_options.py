"""Runtime logging controls that do not change scientific configuration hashes."""

from __future__ import annotations

import argparse
import math
from numbers import Integral


def positive_log_every(value):
    """Parse a positive optimizer-update interval, including direct API calls."""
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise argparse.ArgumentTypeError("log interval must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError("log interval must be a positive integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("log interval must be a positive integer")
    return result


def positive_progress_seconds(value):
    """Parse a finite positive elapsed-time interval for progress reporting."""
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError("progress interval must be a finite positive number of seconds")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError(
            "progress interval must be a finite positive number of seconds"
        ) from error
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("progress interval must be a finite positive number of seconds")
    return result


def nonnegative_attention_every(value):
    """Parse an optimizer-update interval; zero disables attention diagnostics."""
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise argparse.ArgumentTypeError("attention interval must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError("attention interval must be a nonnegative integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("attention interval must be a nonnegative integer")
    return result


def resolve_logging(cfg, log_every=None, progress_every_seconds=None, attention_every=None):
    """Resolve execution controls without mutating the checkpointed configuration."""
    return {
        "log_every": positive_log_every(cfg["training"]["log_every"] if log_every is None else log_every),
        "progress_every_seconds": None
        if progress_every_seconds is None
        else positive_progress_seconds(progress_every_seconds),
        "attention_every": nonnegative_attention_every(10 if attention_every is None else attention_every),
    }
