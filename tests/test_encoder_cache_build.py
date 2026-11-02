"""CPU synthetic diagnostics for parallel learned-encoder cache publication."""

from copy import deepcopy
import json
from pathlib import Path

import h5py
import pytest
import torch

from stmem.config import load_config
from stmem.data.cache import FrozenCache, build_cache, episode_key, read_records
from stmem.data import cache_builder
from stmem.data.feature_cache import FeatureRow
from stmem.memory.streaming import StreamingMemory
from stmem.models.policy import MemoryPolicy
from stmem.training.checkpoints import file_sha256
from stmem.training.data import TrainingData
from stmem.vision.frozen import visual_feature_signature


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(37)
    for name in ("STMEM_DATA_ROOT", "STMEM_CACHE_ROOT", "STMEM_RUN_ROOT", "STMEM_STOP_FILE"):
        monkeypatch.delenv(name, raising=False)
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["paths"].update(cache_root=str(tmp_path / "cache"), run_root=str(tmp_path / "run"))
    cfg["data"].update(
        tasks=["TaskA", "TaskB"],
        episodes_per_task=2,
        observation_stride=2,
        synthetic_execution_frames=12,
        synthetic_conditioning_frames=5,
        decision_stride=2,
    )
    cfg["model"].update(
        width=16,
        heads=4,
        layers=1,
        text_layers=1,
        max_text_bytes=32,
        points=8,
        block_size=2,
        visual_tokens=4,
        current_tokens=3,
        horizon=2,
        execute=2,
        action_stride=2,
        proprioception_mode="none",
        condition_on_previous_commands=False,
    )
    cfg["model"]["encoder_options"].update(neighbor_cap=4, center_chunk=2)
    policy = MemoryPolicy(cfg).eval()
    checkpoint = tmp_path / "selected.pt"
    torch.save({"stage": "teacher", "step": 1, "config": cfg, "model": policy.state_dict()}, checkpoint)
    checkpoint.with_suffix(".selection.json").write_text(json.dumps({"synthetic_debug_only": True}))
    source = {"source_sha256": "synthetic_cache_source"}
    monkeypatch.setattr("stmem.data.cache.cache_code_fingerprint", lambda: source["source_sha256"])
    monkeypatch.setattr(cache_builder, "cache_code_fingerprint", lambda: source["source_sha256"])
    return cfg, checkpoint, policy, source


def test_task_array_complete_only_after_all_shards_and_matches_streaming(fixture):
    cfg, checkpoint, policy, _ = fixture
    root = build_cache(cfg, checkpoint, task_index=0, device="cpu")
    assert (root / "TaskA.h5").exists() and not (root / "manifest.json").exists()
    assert not (Path(cfg["paths"]["run_root"]) / "seed0/cache/complete.json").exists()
    assert cache_builder.cache_status(cfg, checkpoint)["missing_tasks"] == ["TaskB"]
    before = file_sha256(root / "TaskA.h5")
    build_cache(cfg, checkpoint, task="TaskB", device="cpu")
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["identity"]["data"]["tasks"] == ["TaskA", "TaskB"]
    assert set(manifest["shards"]) == {"TaskA.h5", "TaskB.h5"}
    assert not list(root.glob(".*.episodes"))
    data = TrainingData(cfg)
    cache = FrozenCache(cfg, checkpoint, data)
    ep = data.episode(0)
    stream = StreamingMemory(policy.encoder)
    stream.reset(ep.decision(0, 2).conditioning, ep.execution[0])
    with torch.no_grad():
        for time in range(0, 10, 2):
            if time:
                stream.observe(ep.execution[time])
            decision = ep.decision(time, 2)
            cached = cache.evidence(decision)
            online = policy.encode(decision, archive=stream.archive)
            assert [r.node_id for r in cached.archive.records] == [r.node_id for r in online.archive.records]
            for x, y in zip(cached.archive.records, online.archive.records):
                torch.testing.assert_close(x.content, y.content)
                assert x.time_max < decision.current.time or x.source == "conditioning"
                assert x.completed == y.completed
            for x, y in zip(cached.current_records, online.current_records):
                torch.testing.assert_close(x.content, y.content)
    assert build_cache(cfg, checkpoint, task="TaskA", device="cpu") == root
    assert file_sha256(root / "TaskA.h5") == before
    assert cache_builder.cache_status(cfg, checkpoint)["status"] == "complete"


def test_interrupted_episode_commits_resume_without_reencoding_previous(fixture, monkeypatch, tmp_path):
    cfg, checkpoint, _, _ = fixture
    stop = tmp_path / "stop"
    original = cache_builder.write_episode
    calls = []

    def write(*args):
        calls.append(args[1].episode_id)
        result = original(*args)
        stop.touch()
        return result

    monkeypatch.setattr(cache_builder, "write_episode", write)
    with pytest.raises(InterruptedError):
        build_cache(cfg, checkpoint, task="TaskA", device="cpu", stop_file=stop)
    root = Path(cache_builder.cache_status(cfg, checkpoint)["cache_path"])
    assert not (root / "manifest.json").exists()
    assert len(list((root / ".TaskA.episodes").glob("*.json"))) == 1
    stop.unlink()
    monkeypatch.setattr(cache_builder, "write_episode", original)
    build_cache(cfg, checkpoint, task="TaskA", device="cpu", stop_file=stop)
    assert calls == ["episode_0"]
    assert (root / "TaskA.h5").exists()


def test_corrupt_completed_shard_is_not_silently_rebuilt(fixture):
    cfg, checkpoint, _, _ = fixture
    root = build_cache(cfg, checkpoint, device="cpu")
    with (root / "TaskA.h5").open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"X")
    with pytest.raises(ValueError, match="corrupted"):
        build_cache(cfg, checkpoint, task="TaskA", device="cpu")


def test_source_change_rejects_partial_publication(fixture, monkeypatch):
    cfg, checkpoint, _, source = fixture
    original = cache_builder.write_episode

    def write(*args):
        result = original(*args)
        source["source_sha256"] = "changed"
        return result

    monkeypatch.setattr(cache_builder, "write_episode", write)
    with pytest.raises(ValueError, match="Source changed"):
        build_cache(cfg, checkpoint, task="TaskA", device="cpu")
    assert not list(Path(cfg["paths"]["cache_root"]).rglob("manifest.json"))


def test_writer_lock_is_exclusive_and_invalid_task_does_not_create_outputs(fixture):
    cfg, checkpoint, _, _ = fixture
    for args in (
        {"task": "Unknown"},
        {"task_index": -1},
        {"task_index": 9},
        {"task": "TaskA", "task_index": 0},
    ):
        with pytest.raises(ValueError):
            build_cache(cfg, checkpoint, **args)
    assert not Path(cfg["paths"]["cache_root"]).exists()
    root = build_cache(cfg, checkpoint, task="TaskA", device="cpu")
    with cache_builder.locked(root / "TaskA.lock"):
        with pytest.raises(RuntimeError, match="Another encoder-cache writer"):
            build_cache(cfg, checkpoint, task="TaskA", device="cpu")


def test_receipt_backed_pending_rename_recovers(fixture):
    cfg, checkpoint, _, _ = fixture
    root = build_cache(cfg, checkpoint, task="TaskA", device="cpu")
    (root / "TaskA.h5").rename(root / "TaskA.assembling.h5")
    build_cache(cfg, checkpoint, task="TaskA", device="cpu")
    assert (root / "TaskA.h5").exists()


def test_lazy_frozen_descriptors_and_no_prop_10hz_match_direct_encoder(fixture, tmp_path):
    cfg, _, _, _ = fixture
    cfg = deepcopy(cfg)
    cfg["model"]["visual_features"] = {"mode": "clip_rn50_layer1", "projection_dim": 128}
    policy = MemoryPolicy(cfg).eval()
    data = TrainingData(cfg)
    ep = data.episode(0)
    signature = visual_feature_signature(cfg)
    rows = torch.randn(len(ep.conditioning) + len(ep.execution), 8, 256).half()
    path = tmp_path / "synthetic-descriptors.bin"
    path.write_bytes(rows.numpy().tobytes())
    for i, frame in enumerate(ep.conditioning + ep.execution):
        frame.visual_feature_signature = signature
        frame.visual_features = FeatureRow(str(path), ep.episode_id, i, 8, signature, i * 8 * 256 * 2)
    identity = {"code": "synthetic_cache_source"}
    committed = tmp_path / "episode.h5"
    with torch.no_grad():
        cache_builder.write_episode(committed, ep, policy, data, identity, lambda: None)
    assert all(isinstance(f.visual_features, FeatureRow) for f in ep.conditioning + ep.execution)
    with torch.no_grad(), h5py.File(committed, "r") as f:
        group = f[episode_key(ep.task, ep.episode_id)]
        for index in range(0, 10, 2):
            decision = ep.decision(index, 2)
            direct = policy.encode(decision)
            for x, y in zip(read_records(group["current"][str(index)]), direct.current_records):
                torch.testing.assert_close(x.content, y.content, atol=0, rtol=0)
            pending = len(decision.execution_prefix) % policy.encoder.block_size
            if pending:
                offset = (
                    len(decision.execution_prefix) // policy.encoder.block_size * policy.encoder.block_size
                )
                records = read_records(group["prefix"][f"{offset}_{pending}"])
                direct_records = policy.encoder.encode_frames(
                    decision.execution_prefix[offset:], source="execution", completed=False
                )
                assert all(not r.completed and r.modality == "visual" for r in records)
                for x, y in zip(records, direct_records):
                    torch.testing.assert_close(x.content, y.content, atol=0, rtol=0)


@pytest.mark.parametrize(
    "relative",
    [
        "src/stmem/evaluation/parallel.py",
        "src/stmem/evaluation/nested/new_worker.py",
        "src/stmem/inspection/pipeline.py",
        "src/stmem/report.py",
    ],
)
def test_nonencoding_tool_changes_preserve_cache_code_fingerprint(tmp_path, relative):
    from stmem.data.cache_source import cache_code_fingerprint

    encoder = tmp_path / "src/stmem/models/encoder.py"
    encoder.parent.mkdir(parents=True)
    encoder.write_text("def encode(x): return x\n")
    before = cache_code_fingerprint(tmp_path)
    tool = tmp_path / relative
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_text("# new evaluation/inspection/report source\n")
    assert cache_code_fingerprint(tmp_path) == before
    tool.write_text("# changed orchestration only\n")
    assert cache_code_fingerprint(tmp_path) == before
    tool.unlink()
    assert cache_code_fingerprint(tmp_path) == before


@pytest.mark.parametrize(
    "relative",
    [
        "src/stmem/models/spacetime_encoder.py",
        "src/stmem/data/canonical.py",
        "src/stmem/training/runtime.py",
        "src/stmem/config.py",
        "src/stmem/vision/frozen.py",
        "src/stmem/memory/streaming.py",
        "src/stmem/data/cache_source.py",
        "src/stmem/evaluation_utils.py",
        "src/stmem/reporting.py",
        "src/another_package/inspection/tool.py",
    ],
)
def test_encoding_and_all_other_source_changes_invalidate_cache_code_fingerprint(tmp_path, relative):
    from stmem.data.cache_source import cache_code_fingerprint

    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_text("# original source\n")
    original = cache_code_fingerprint(tmp_path)
    source.write_text("# changed source\n")
    assert cache_code_fingerprint(tmp_path) != original
    source.write_text("# original source\n")
    assert cache_code_fingerprint(tmp_path) == original
    source.rename(source.with_name("renamed_" + source.name))
    assert cache_code_fingerprint(tmp_path) != original


def test_final_cache_records_full_source_separately_from_compatibility(fixture, monkeypatch):
    cfg, checkpoint, _, _ = fixture
    full_source = {
        "source_sha256": "full_repo_source_including_evaluation",
        "git": "synthetic",
        "dirty": False,
    }
    monkeypatch.setattr(cache_builder, "source_revision", lambda: dict(full_source))
    root = build_cache(cfg, checkpoint, device="cpu")
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["source"] == full_source
    assert manifest["identity"]["code"] == "synthetic_cache_source"
