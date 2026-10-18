"""W&B batch configuration survives rendering without persisting credentials."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def planner(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("STMEM_", "WANDB_")):
            monkeypatch.delenv(key)
    path = Path(__file__).resolve().parents[1] / "hpc/plan.py"
    spec = importlib.util.spec_from_file_location("wandb_hpc_test_planner", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_site_wandb_allowlist_precedence_and_credential_exclusion(planner, monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_PROJECT", "ambient-project")
    monkeypatch.setenv("WANDB_ENTITY", "ambient-team")
    monkeypatch.setenv("WANDB_API_KEY", "a-secret-value-that-must-not-be-rendered")
    monkeypatch.setenv("WANDB_RUN_ID", "must-not-override-stage-identity")
    site = tmp_path / "site.env"
    site.write_text('WANDB_MODE=offline\nWANDB_PROJECT=chronotree\nWANDB_DIR="/shared/with spaces"\n')
    values = planner.read_site(site)
    assert values == {
        "WANDB_MODE": "offline",
        "WANDB_PROJECT": "chronotree",
        "WANDB_ENTITY": "ambient-team",
        "WANDB_DIR": "/shared/with spaces",
    }
    assert "WANDB_API_KEY" not in values and "WANDB_RUN_ID" not in values


@pytest.mark.parametrize("key", ["WANDB_API_KEY", "WANDB_RUN_ID", "WANDB_BASE_URL"])
def test_site_rejects_unapproved_wandb_settings_without_value_in_error(planner, tmp_path, key):
    path = tmp_path / "site.env"
    secret = "never-include-this-value-in-output"
    path.write_text(f"{key}={secret}\n")
    with pytest.raises(ValueError) as error:
        planner.read_site(path)
    assert key in str(error.value)
    assert secret not in str(error.value)


def test_wandb_values_keep_literal_parser_semantics(planner, tmp_path):
    path = tmp_path / "site.env"
    path.write_text('WANDB_DIR="$HOME/wandb"\n')
    with pytest.raises(ValueError, match="literal values"):
        planner.read_site(path)
    path.write_text("WANDB_MODE=onlien\n")
    with pytest.raises(ValueError, match="WANDB_MODE must"):
        planner.read_site(path)


def test_render_forwards_nonsecret_settings_and_keeps_scientific_identity(planner, monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_API_KEY", "do-not-persist-me")
    monkeypatch.setenv("WANDB_ENTITY", "old-team")
    site_path = tmp_path / "site.env"
    directory = str(tmp_path / "log directory;literal")
    site_path.write_text(
        f"WANDB_MODE=offline\nWANDB_PROJECT=chronotree\nWANDB_ENTITY=\nWANDB_DIR={shlex.quote(directory)}\n"
    )
    site = planner.read_site(site_path)
    job = planner.Job("seed0_teacher", "teacher", [[sys.executable, "-m", "stmem.train"]])
    options = dict(
        repo=tmp_path,
        generated=tmp_path / "generated",
        logs=tmp_path / "logs",
        python=sys.executable,
        profile="h200_1gpu",
    )
    planner.render_job(job, site=site, **options)
    signature = job.signature
    script = Path(job.script).read_text()
    assert "export WANDB_MODE=offline" in script
    assert "export WANDB_PROJECT=chronotree" in script
    assert "unset WANDB_ENTITY" in script
    assert f"export WANDB_DIR={shlex.quote(directory)}" in script
    assert "WANDB_API_KEY" not in script and "do-not-persist-me" not in script
    subprocess.run(["bash", "-n", job.script], check=True)
    # Simulate the remote batch shell without the submit shell's W&B values.
    # Stop before the actual launcher: this executes only the exported settings.
    prefix = script.split("\nexec ", 1)[0]
    inspect = shlex.join(
        [
            sys.executable,
            "-c",
            "import json, os; print(json.dumps({k: os.environ.get(k) for k in "
            "('WANDB_MODE', 'WANDB_PROJECT', 'WANDB_ENTITY', 'WANDB_DIR')}))",
        ]
    )
    remote = subprocess.run(
        ["bash", "-c", prefix + "\n" + inspect],
        env={"PATH": os.environ.get("PATH", ""), "WANDB_ENTITY": "remote-default-team"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(remote.stdout) == {
        "WANDB_MODE": "offline",
        "WANDB_PROJECT": "chronotree",
        "WANDB_ENTITY": None,
        "WANDB_DIR": directory,
    }
    site["WANDB_MODE"] = "online"
    planner.render_job(job, site=site, **options)
    assert job.signature == signature


@pytest.mark.parametrize("render_fails", [False, True])
def test_node_smoke_tracks_diagnostics_outside_timing(monkeypatch, tmp_path, render_fails):
    import torch

    path = Path(__file__).resolve().parents[1] / "hpc/smoke.py"
    spec = importlib.util.spec_from_file_location("wandb_hpc_test_smoke", path)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    trackers = []
    events = []

    class FakeTracker:
        def __init__(self, cfg, directory, *, job_type, metadata):
            assert cfg["data"]["synthetic"] is True
            assert metadata["diagnostic_only"] is True
            self.directory, self.job_type = directory, job_type
            self.identity, self.logs = {}, []
            trackers.append(self)

        def __enter__(self):
            self.identity = {"id": self.job_type, "status": "running"}
            return self

        def log(self, values, step):
            if self.job_type == "hpc-smoke-training":
                assert events[-2:] == ["synchronize", "synchronize"]
            events.append("log")
            self.logs.append((values, step))

        def summary(self, values):
            pass

        def __exit__(self, exc_type, *args):
            self.identity["status"] = "failed" if exc_type else "completed"

    class TinyPolicy(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.normalizer = None

        def forward(self, decision, x, t):
            return x * self.weight

    class RenderAdapter:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self):
            if render_fails:
                raise RuntimeError("diagnostic renderer failed")
            return SimpleNamespace(
                current=SimpleNamespace(proprio=torch.zeros(15), valid=torch.ones(5, dtype=torch.bool)),
                conditioning=[object()],
                metadata={"backend": "test-double"},
            )

        def step(self, command):
            return SimpleNamespace(frames=[object()], status="continue")

        def close(self):
            events.append("renderer_closed")

    decision = SimpleNamespace(target=torch.ones(2, 8), target_valid=torch.ones(2, 8, dtype=torch.bool))
    monkeypatch.setattr(smoke, "RunTracker", FakeTracker)
    monkeypatch.setattr(
        smoke, "load_config", lambda path: {"debug": True, "model": {"points": 16, "horizon": 2}}
    )
    monkeypatch.setattr(smoke, "diagnose", lambda: {"device": "cpu", "ok": True})
    monkeypatch.setattr(smoke, "MemoryPolicy", TinyPolicy)
    monkeypatch.setattr(
        smoke, "synthetic_episode", lambda *args: SimpleNamespace(decision=lambda *args: decision)
    )
    monkeypatch.setattr(smoke, "flow_sample", lambda target, *args: (target, torch.zeros(1), target, None))
    monkeypatch.setattr(smoke, "choose_precision", lambda device: "fp32")
    monkeypatch.setattr(smoke, "synchronize", lambda device: events.append("synchronize"))
    monkeypatch.setattr(smoke, "RoboMMEAdapter", RenderAdapter)
    output = tmp_path / "node_smoke.json"
    monkeypatch.setattr(sys, "argv", [str(path), "--render", "--updates", "1", "--output", str(output)])
    with pytest.raises(SystemExit) as exit_status:
        smoke.main()
    assert exit_status.value.code == int(render_fails)
    report = json.loads(output.read_text())
    assert report["bounded_training"]["tracking"]["status"] == "completed"
    assert len(trackers[0].logs) == 1 and trackers[0].logs[0][1] == 1
    assert trackers[0].directory == tmp_path / "node_smoke/training"
    assert trackers[1].directory == tmp_path / "node_smoke/rendering"
    assert report["rendering"]["tracking"]["status"] == ("failed" if render_fails else "completed")
    assert "renderer_closed" in events
