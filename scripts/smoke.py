#!/usr/bin/env python3
"""Run a bounded synthetic T/C/P/L/R/refresh/Perceiver/inference diagnostic.

Debug checkpoint copies are deliberately NOT official validation selections.
Every synthetic artifact is isolated under the provided debug run root.
"""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = (
        args.root or repo / "runs" / "smoke" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ).resolve()
    if root.exists():
        raise FileExistsError(f"Use a new smoke root: {root}")
    root.mkdir(parents=True)
    configs = root / "configs"
    configs.mkdir()

    def config(stage, reader=None, final=False):
        base = f"router_{reader}" if reader else stage
        value = {
            "extends": str(repo / "configs" / "debug" / f"{base}.yaml"),
            "paths": {"run_root": str(root), "cache_root": str(root / "cache")},
            "labels": {"contexts": 30, "refresh_contexts": 15, "use_refresh": final},
            "training": {
                "steps": 12 if stage != "router" or final else 6,
                "checkpoint_steps": [6, 12] if stage != "router" or final else [6],
            },
        }
        path = configs / f"{base}{'_final' if final else ''}.yaml"
        path.write_text(yaml.safe_dump(value))
        return str(path)

    def call(module, *arguments):
        command = [sys.executable, "-m", module, *map(str, arguments)]
        print("Running", " ".join(command), flush=True)
        subprocess.run(command, cwd=repo, check=True)

    def debug_selection(stage):
        directory = root / "seed0" / stage
        shutil.copyfile(directory / "last.pt", directory / "selected.pt")
        (directory / "selected.selection.json").write_text(
            json.dumps(
                {
                    "synthetic_debug_only": True,
                    "rule": "Final tiny debug checkpoint; NOT a scientific selection or official validation result",
                },
                indent=2,
            )
        )

    teacher = config("teacher")
    compact = config("compact")
    call("stmem.train", "--stage", "teacher", "--config", teacher)
    debug_selection("teacher")
    call("stmem.data.cache", "--config", teacher, "--checkpoint", root / "seed0/teacher/selected.pt")
    call("stmem.train", "--stage", "compact", "--config", compact)
    debug_selection("compact")
    for reader in ("tree", "flat"):
        initial = config("router", reader)
        final = config("router", reader, final=True)
        call("stmem.training.gain_labels", "--reader", reader, "--config", initial)
        call("stmem.train", "--stage", "router", "--config", initial)
        checkpoint = root / "seed0" / f"router_{reader}" / "step_00000006.pt"
        call(
            "stmem.training.gain_labels",
            "--reader",
            reader,
            "--config",
            final,
            "--refresh",
            "--router-checkpoint",
            checkpoint,
        )
        call("stmem.train", "--stage", "router", "--config", final, "--resume", checkpoint)
    call("stmem.train", "--stage", "perceiver", "--config", config("perceiver"))
    # Exercise every reader at inference, including selected CPU records -> GPU transfer.
    import torch
    from stmem.config import load_config
    from stmem.training.data import TrainingData
    from stmem.training.runtime import load_policy, load_router
    from stmem.evaluation.runner import OnlinePolicy
    from stmem.evaluation.robomme_adapter import ResetResult
    from stmem.training.checkpoints import atomic_json, file_sha256

    cfg = load_config(compact)
    data = TrainingData(cfg)
    episode = data.episode(0)
    output = {}
    for method in ("FULL", "TREE", "FLAT", "UNIFORM", "PERCEIVER", "CURRENT"):
        stage = "teacher" if method == "FULL" else "perceiver" if method == "PERCEIVER" else "compact"
        path = root / "seed0" / stage / "last.pt"
        policy, _ = load_policy(path, "cuda" if torch.cuda.is_available() else "cpu")
        router_path = root / "seed0" / f"router_{method.lower()}" / "last.pt"
        router = load_router(router_path, policy.device)[0] if method in ("TREE", "FLAT") else None
        online = OnlinePolicy(
            policy,
            cfg,
            method,
            cfg["model"]["budget"],
            router=router,
            router_hash=file_sha256(router_path) if router else "",
        )
        online.reset(ResetResult(episode.conditioning, episode.execution[0], episode.goal), 7)
        for frame in episode.execution[1:7]:
            online.observe(frame)
        action, info = online.decide()
        assert action.shape == (cfg["model"]["horizon"], 8) and torch.isfinite(action).all()
        output[method] = info
    atomic_json(
        {"status": "complete", "synthetic_debug_only": True, "readers": output}, root / "smoke_complete.json"
    )
    print(f"Synthetic smoke complete: {root}")


if __name__ == "__main__":
    main()
