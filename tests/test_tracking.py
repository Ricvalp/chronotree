"""Tracking must not change training RNG, lose restarts or silently omit runs."""

from copy import deepcopy
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stmem.tracking import RunTracker


@pytest.fixture
def fake_wandb(monkeypatch, tmp_path):
    for key in ("WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_DIR"):
        monkeypatch.delenv(key, raising=False)
    runs = []

    def init(**kwargs):
        random.random()
        np.random.random()
        torch.rand(3)
        run = SimpleNamespace(
            kwargs=kwargs,
            dir=str(tmp_path / f"sdk{len(runs)}" / "files"),
            step=0,
            url="https://example.invalid/run",
            summary={},
            history=[],
            finished=[],
            metrics=[],
        )
        run.define_metric = lambda *a, **kw: run.metrics.append((a, kw))

        def log(row, *, step, commit):
            run.history.append((deepcopy(row), step, commit))
            run.step = step + 1

        def finish(*, exit_code):
            random.random()
            np.random.random()
            torch.rand(3)
            run.finished.append(exit_code)

        run.log, run.finish = log, finish
        runs.append(run)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **kw: kw))
    return runs


def config(tmp_path):
    return {"seed": 7, "data": {"synthetic": True}, "paths": {"run_root": str(tmp_path)}}


def test_rank_zero_only_and_explicit_disable(fake_wandb, tmp_path):
    for rank, mode in [(1, "offline"), (0, "disabled")]:
        cfg = config(tmp_path)
        cfg["tracking"] = {"mode": mode}
        with RunTracker(cfg, tmp_path / str(rank), job_type="teacher", rank=rank) as tracker:
            tracker.log({"loss": 2}, step=1)
            tracker.summary({"done": True})
    assert not fake_wandb
    assert not (tmp_path / "0").exists() and not (tmp_path / "1").exists()


def test_sdk_init_finish_preserve_training_rng(fake_wandb, tmp_path):
    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)
    with RunTracker(config(tmp_path), tmp_path / "run", job_type="teacher") as tracker:
        tracker.log({"train/loss": 0.5}, step=1)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
    assert fake_wandb[0].kwargs["settings"]["x_disable_stats"] is False
    assert fake_wandb[0].kwargs["settings"]["x_disable_meta"] is False
    assert fake_wandb[0].kwargs["settings"]["console"] == "off"


@pytest.mark.parametrize("job_type", ["teacher", "compact", "perceiver", "router", "evaluate"])
@pytest.mark.parametrize("system_stats", [True, False])
def test_system_stats_setting_across_job_types(fake_wandb, tmp_path, job_type, system_stats):
    cfg = config(tmp_path)
    cfg["tracking"] = {"system_stats": system_stats}
    with RunTracker(cfg, tmp_path / "run", job_type=job_type):
        pass
    settings = fake_wandb[0].kwargs["settings"]
    assert settings["x_disable_stats"] is (not system_stats)
    assert settings["x_disable_meta"] is (not system_stats)


def test_resume_keeps_id_and_monotonic_history_with_rewound_optimizer(fake_wandb, tmp_path):
    cfg = config(tmp_path)
    directory = tmp_path / "run"
    with RunTracker(cfg, directory, job_type="teacher") as tracker:
        tracker.log({"train/loss": 1.0}, step=50)
        tracker.finish(status="checkpointed_partial", exit_code=75)
    # Simulate a process dying after publishing but before updating tracking.json.
    state = json.loads((directory / "tracking.json").read_text())
    state["next_history_step"] = 0
    (directory / "tracking.json").write_text(json.dumps(state))
    with RunTracker(cfg, directory, job_type="teacher", resume=True) as tracker:
        tracker.log({"train/loss": 0.5}, step=40)
    first, second = fake_wandb
    assert first.kwargs["id"] == second.kwargs["id"]
    assert first.history[0][1] == 0 and second.history[0][1] == 1
    assert second.history[0][0]["progress/step"] == 40
    assert second.history[0][0]["tracking/attempt"] == 2
    assert first.finished == [75] and second.finished == [0]
    assert json.loads((directory / "tracking.json").read_text())["status"] == "completed"


def test_online_environment_failure_finalization_and_identity_guard(fake_wandb, monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_PROJECT", "research")
    monkeypatch.setenv("WANDB_ENTITY", "my-team")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "scratch"))
    with pytest.raises(RuntimeError, match="training failed"):
        with RunTracker(config(tmp_path), tmp_path / "run", job_type="teacher"):
            raise RuntimeError("training failed")
    run = fake_wandb[0]
    assert run.kwargs["resume"] == "allow"
    assert run.kwargs["project"] == "research" and run.kwargs["entity"] == "my-team"
    assert run.finished == [1]
    assert run.summary["tracking/status"] == "failed"
    monkeypatch.setenv("WANDB_PROJECT", "different")
    with pytest.raises(ValueError, match="project changed"):
        with RunTracker(config(tmp_path), tmp_path / "run", job_type="teacher", resume=True):
            pass


@pytest.mark.parametrize("malformed", [True, False])
def test_unterminated_journal_survives_two_restarts(fake_wandb, tmp_path, malformed):
    directory = tmp_path / "run"
    cfg = config(tmp_path)
    with RunTracker(cfg, directory, job_type="teacher") as tracker:
        tracker.log({"loss": 1.0}, step=10)
    journal = directory / "tracking_history.jsonl"
    content = journal.read_text()
    journal.write_text(content + '{"history_step":' if malformed else content.rstrip("\n"))
    for step in (8, 9):
        state_path = directory / "tracking.json"
        state = json.loads(state_path.read_text())
        state["next_history_step"] = 0
        state_path.write_text(json.dumps(state))
        with RunTracker(cfg, directory, job_type="teacher", resume=True) as tracker:
            tracker.log({"loss": 0.5}, step=step)
    assert [run.history[0][1] for run in fake_wandb] == [0, 1, 2]


def test_scalar_only_logging_and_no_checkpoint_upload(fake_wandb, tmp_path):
    cfg = config(tmp_path)
    cfg["api_key"] = "do-not-record"
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    with RunTracker(cfg, tmp_path / "run", job_type="teacher") as tracker:
        with pytest.raises(TypeError, match="CPU scalars"):
            tracker.log({"loss": torch.tensor(1.0)}, step=1)
        tracker.log({"loss": np.float32(0.25), "undefined": float("nan")}, step=1)
        tracker.checkpoint(checkpoint, 1)
    run = fake_wandb[0]
    assert run.kwargs["config"]["api_key"] == "[redacted]"
    assert run.summary["checkpoint/bytes"] == 10
    assert run.history[0][0]["undefined"] is None


def test_initialization_failure_cleanup_preserves_rng_and_closes_journal(fake_wandb, monkeypatch, tmp_path):
    sdk = sys.modules["wandb"]
    original = sdk.init

    class BrokenSummary(dict):
        def update(self, *args, **kwargs):
            raise RuntimeError("summary failed")

    def failing_init(**kwargs):
        run = original(**kwargs)
        run.summary = BrokenSummary()
        return run

    monkeypatch.setattr(sdk, "init", failing_init)
    random.seed(55)
    expected = random.random()
    random.seed(55)
    tracker = RunTracker(config(tmp_path), tmp_path / "run", job_type="teacher")
    with pytest.raises(RuntimeError, match="summary failed"):
        with tracker:
            pass
    assert random.random() == expected
    assert tracker._journal.closed
    assert fake_wandb[0].finished == [1]
    assert json.loads((tmp_path / "run/tracking.json").read_text())["status"] == "initialization_failed"


def test_repeated_training_request_does_not_reopen_completed_run(fake_wandb, tmp_path):
    from stmem.config import load_config
    from stmem.train import train_policy

    cfg = load_config("configs/debug/teacher.yaml")
    cfg["paths"]["run_root"] = str(tmp_path)
    directory = tmp_path / "seed0/teacher"
    directory.mkdir(parents=True)
    (directory / "last.pt").touch()
    with pytest.raises(FileExistsError, match="use --resume"):
        train_policy(cfg, "teacher")
    assert not fake_wandb
    assert not (directory / "tracking.json").exists()


def read_offline(path):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_scan(str(path))
    records = []
    try:
        while (payload := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(payload)
            records.append(record)
    finally:
        store.close()
    return records


def test_real_sdk_offline_files_contain_metrics_and_resumed_progress(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    for key in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_DIR"):
        monkeypatch.delenv(key, raising=False)
    cfg = config(tmp_path)
    directory = tmp_path / "run"
    for attempt, step in enumerate([5, 3]):
        with RunTracker(cfg, directory, job_type="teacher", resume=bool(attempt)) as tracker:
            tracker.log({"train/loss": 1.0 / (attempt + 1)}, step=step)
            tracker.summary({"completed": True})
    files = sorted(directory.glob("wandb/offline-run-*/*.wandb"))
    assert len(files) == 2
    histories = []
    ids = []
    for path in files:
        records = read_offline(path)
        ids += [r.run.run_id for r in records if r.HasField("run")]
        for record in records:
            if record.HasField("history"):
                histories.append(
                    {
                        item.key or "/".join(item.nested_key): json.loads(item.value_json)
                        for item in record.history.item
                    }
                )
    assert len(set(ids)) == 1
    assert [h["progress/step"] for h in histories] == [5, 3]
    assert [h["_step"] for h in histories] == [0, 1]
    assert [h["train/loss"] for h in histories] == [1.0, 0.5]
