"""Synthetic CPU checks of the controlled student-training comparison."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from test_packed_context import indexed  # noqa: F401 -- shared synthetic indexed fixture
from test_cached_context_loader import fixture  # noqa: F401 -- dependency of indexed
from stmem.config import digest
from stmem.training.cached_loader import CachedPolicyDataset, LightweightDecisions
from stmem.training.cached_policy import CachedPolicyForward
from stmem.training.checkpoints import file_sha256
from stmem.training.packed_context import PackedContextReader
from stmem.training.utility_study import (
    COMPACT_SHA256, ENCODED_CACHE_ID, HIERARCHY_CACHE_ID, TEACHER_SHA256,
    consumed_packet_costs, paired_flow_randomness, primary_schedule, read_study_manifest, restrict_training_pool,
    summarize_training_rows,
)


def study_cfg(cfg, root, variant="S_STD"):
    result = deepcopy(cfg)
    result["stage"] = "compact"
    result["training"].update(hierarchy_cache_root=str(root), utility_study={
        "variant": variant, "manifest": "synthetic-unread-manifest", "manifest_sha256": "synthetic",
    })
    return result


def manifest_for(data):
    rows = [{"task": task, "episode_id": f"episode_{data.refs[index][1]}",
             "partition": data.label_partition(index)}
            for task in data.tasks for index in data.by_task[task]]
    return sealed({"schema_version": 1,
            "teacher_sha256": TEACHER_SHA256, "compact_sha256": COMPACT_SHA256,
            "encoded_cache_id": ENCODED_CACHE_ID, "hierarchy_cache_id": HIERARCHY_CACHE_ID,
            "episode_partitions": rows, "diagnostic_episodes": [],
            "training_pool": [row for row in rows if row["partition"] == "train"],
            "reserved_validation_episodes": [row for row in rows if row["partition"] == "validation"]})


def sealed(manifest):
    result = {key: value for key, value in manifest.items() if key != "experiment_id"}
    return {**result, "experiment_id": digest(result)}


def rebased_manifest():
    """Synthetic cache IDs; the new accepted teacher's declared checkpoint identity."""
    teacher_sha = "d17dee6cfe70caae8ff8d1013b0fe50aee01d1cd816feda4a25f0f4d388c0d34"
    dependencies = {
        "teacher_sha256": teacher_sha, "teacher_step": 140000,
        "compact_sha256": teacher_sha, "compact_step": 140000,
        "encoded_cache_id": "a" * 24, "hierarchy_cache_id": "b" * 24,
        "accepted_cache_plan": "/synthetic/adopted/plan.json", "accepted_cache_plan_sha256": "c" * 64,
        "teacher_selection": "/synthetic/adopted/selection.json", "teacher_selection_sha256": "d" * 64,
    }
    return sealed({
        **dependencies, "dependencies": dependencies,
        "reference": {"name": "S_INIT", "kind": "frozen_teacher_compact_smoke", "compact_updates": 0},
        "primary_endpoints": ["S_STD", "S_STAGE", "S_PACKET"],
        "protocol": {
            "accepted_teacher": {"step": 140000, "sha256": teacher_sha},
            "student_training": {"steps": 100000, "checkpoint_every": 1000, "snapshot_every": 5000},
        },
    })


def test_new_training_boundary_preserves_original_partitions_and_cache_identity(indexed, tmp_path):  # noqa: F811
    cfg, data, _, _, root = indexed
    original_identity, original_refs = data.identity, list(data.refs)
    original_partition = [data.label_partition(i) for i in range(len(data.refs))]
    cfg = study_cfg(cfg, root)
    manifest = manifest_for(data)
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(manifest))
    cfg["training"]["utility_study"].update(manifest=str(path), manifest_sha256=file_sha256(path))
    result = restrict_training_pool(data, cfg)
    assert result["training_episodes"] == 2
    assert data.identity == original_identity and data.refs == original_refs
    assert [data.label_partition(i) for i in range(len(data.refs))] == original_partition
    assert all(data.label_partition(data.sample_reference(i)[0]) == "train" for i in range(100))
    manifest["diagnostic_episodes"] = manifest["training_pool"][:1]
    path.write_text(json.dumps(sealed(manifest)))
    with pytest.raises(ValueError, match="overlaps"):
        read_study_manifest(path)
    with pytest.raises(ValueError, match="changed"):
        restrict_training_pool(data, cfg)


def test_first_student_launch_rejects_missing_or_stale_manifest_seal(indexed, tmp_path):  # noqa: F811
    _, data, _, _, _ = indexed
    manifest = manifest_for(data)
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(manifest))
    assert read_study_manifest(path) == manifest
    for malformed in ({key: value for key, value in manifest.items() if key != "experiment_id"},
                      {**manifest, "sampling_scope": "changed_after_sealing"}):
        path.write_text(json.dumps(malformed))
        # Even a newly pinned byte hash must not bless an invalid internal seal.
        for expected_hash in (None, file_sha256(path)):
            with pytest.raises(ValueError, match="seal"):
                read_study_manifest(path, expected_hash)


def test_consumed_packet_lengths_stop_at_the_sampled_prefix():
    route = {"initial_count": 4, "contexts": {count: {} for count in range(4, 24)},
             "packets": [{"start_count": 4, "cost": 8}, {"start_count": 12, "cost": 8},
                         {"start_count": 20, "cost": 3}]}
    assert consumed_packet_costs(route, 4) == []
    assert consumed_packet_costs(route, 7) == [3]
    assert consumed_packet_costs(route, 12) == [8]
    assert consumed_packet_costs(route, 17) == [8, 5]
    assert consumed_packet_costs(route, 23) == [8, 8, 3]
    with pytest.raises(ValueError, match="absent"):
        consumed_packet_costs(route, 24)


@pytest.mark.parametrize("family", ["tree", "flat", "uniform", "current"])
@pytest.mark.parametrize("variant", ["S_STD", "S_STAGE", "S_PACKET"])
def test_sampler_and_audit_preserve_empty_causal_history(family, variant):
    import numpy as np
    from test_refinement_candidates import forest
    from stmem.config import load_config
    from stmem.training.utility_sampler_audit import context_statistics
    from stmem.training.utility_study import study_context

    cfg = study_cfg(load_config("configs/debug/compact.yaml"), "synthetic-hierarchy", variant)
    cfg["training"]["context_mix"] = {family: 1.0}
    # Later rows may exist in the episode cache. At this decision none are
    # causal, so the empty archive must stay empty for every recipe and family.
    index, _, _ = forest((4, 4))
    roots = leaves = np.empty(0, dtype=np.int64)
    for position in range(20):
        context, metadata = study_context(index, roots, leaves, cfg, position)
        assert context["rows"] == [] and context["token_count"] == 0
        assert metadata["family"] == family.upper()
        statistics = context_statistics(index, roots, leaves, context)
        assert statistics == {"token_count": 0, "leaf_fraction": 0.0, "mean_depth": 0.0,
                              "max_depth": 0, "source_coverage": [], "root_record_concentration": 0.0}


def test_paired_flow_draws_independent_of_microbatch_order_and_global_rng():
    actions = torch.zeros(4, 16, 8)
    a, u = paired_flow_randomness(actions, [1, 9, 15, 40], 0)
    torch.randn(1000)
    b, v = paired_flow_randomness(actions, [40, 15, 1, 9], 0)
    assert torch.equal(a[[3, 2, 0, 1]], b) and torch.equal(u[[3, 2, 0, 1]], v)
    for row, position in enumerate([1, 9, 15, 40]):
        one, time = paired_flow_randomness(actions[:1], [position], 0)
        assert torch.equal(one[0], a[row]) and torch.equal(time[0], u[row])


def test_standard_context_matches_legacy_and_stage_packet_exposures_match(indexed):  # noqa: F811
    cfg, data, _, cache, root = indexed
    light = LightweightDecisions(data)
    readers = {variant: PackedContextReader(study_cfg(cfg, root, variant), cache.root, root)
               for variant in ("S_STD", "S_STAGE", "S_PACKET")}
    legacy = PackedContextReader(cfg, cache.root, root)
    templates = 0
    try:
        for position in range(40):
            decision = light.decision(position)
            old = legacy.prepare(decision, position)
            rows = {variant: reader.prepare(decision, position) for variant, reader in readers.items()}
            for key in ("content", "metadata", "valid", "log_count"):
                torch.testing.assert_close(getattr(rows["S_STD"].history, key), getattr(old.history, key),
                                           rtol=0, atol=0)
            a, b = rows["S_STAGE"].study_context, rows["S_PACKET"].study_context
            for key in ("family", "template", "requested_target", "initial_count", "target_count",
                        "prefix_count", "fallback_reason", "actual_count"):
                assert a.get(key) == b.get(key), key
            templates += a["template"]
            if a["template"]:
                for metadata in (a, b):
                    assert sum(metadata["consumed_packet_costs"]) == metadata["prefix_count"] - metadata["initial_count"]
                    assert sum(metadata["packet_costs"]) == metadata["target_count"] - metadata["initial_count"]
            for value in rows.values():
                torch.testing.assert_close(value.current.content, old.current.content, rtol=0, atol=0)
                assert value.history.valid.sum() == value.study_context["actual_count"]
        assert templates > 0
    finally:
        light.close()
        legacy.close()
        for reader in readers.values():
            reader.close()


def test_student_cached_context_gradients_reach_only_decoder_adapter_clock(indexed):  # noqa: F811
    cfg, data, policy, cache, root = indexed
    data.cfg = study_cfg(cfg, root, "S_PACKET")
    policy.requires_grad_(True)
    policy.freeze_encoders()
    policy.train()
    loader = CachedPolicyDataset(data, cache.root)
    try:
        batch = loader[tuple(range(4))]
        prediction, _ = CachedPolicyForward(policy)(batch, torch.randn_like(batch.target_chunk), torch.rand(4))
        prediction.square().mean().backward()
        assert not policy.encoder.training and not policy.text_encoder.training
        assert all(parameter.grad is None and not parameter.requires_grad
                   for module in (policy.encoder, policy.text_encoder) for parameter in module.parameters())
        for module in (policy.decoder, policy.adapter, policy.clock):
            assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad)
                       for parameter in module.parameters())
        assert len(batch.study_contexts) == 4
    finally:
        loader.close()


def test_stratified_metrics_keep_counts_and_separate_time_and_phase():
    rows = [{"task": "A", "phase": "late", "family": "TREE", "actual_count": 128,
             "history_frames": 100, "flow_time": 0.1, "loss": loss, "template": True}
            for loss in (1.0, 3.0)]
    result = summarize_training_rows(rows)
    assert all(row["count"] == 2 and row["mean_expert_fm_loss"] == 2 for row in result)
    assert {row["dimension"] for row in result} >= {"task", "phase", "history_length", "budget", "flow_time"}


def test_conditional_rehearsal_keeps_full_leaves_and_current_control_unchanged(indexed):  # noqa: F811
    cfg, data, _, cache, root = indexed
    cfg = study_cfg(cfg, root)
    cfg["training"]["utility_study"]["followup"] = "rehearsal_full"
    light = LightweightDecisions(data)
    reader = PackedContextReader(cfg, cache.root, root)
    replacements = current_controls = 0
    try:
        for position in range(100):
            prepared = reader.prepare(light.decision(position), position)
            context = prepared.study_context
            if context.get("rehearsal"):
                replacements += 1
                assert prepared.history.valid.sum().item() == prepared.leaf_count
                assert torch.count_nonzero(prepared.history.log_count) == 0
            if context["family"] == "CURRENT":
                current_controls += 1
                assert not context.get("rehearsal")
                assert prepared.history.valid.numel() == 0
        assert replacements > 0 and current_controls > 0
    finally:
        reader.close()
        light.close()


def test_compact_study_config_preserves_model_and_declares_fixed_endpoint():
    import importlib.util
    from pathlib import Path
    from stmem.config import load_config

    path = Path(__file__).resolve().parents[1] / "hpc/train_compact.py"
    spec = importlib.util.spec_from_file_location("utility_compact_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = load_config("configs/teacher_fv_no_prop_10hz.yaml")
    args = SimpleNamespace(partition="gpuq", gpus=2, effective_batch=128, microbatch=32, workers=16,
                           hierarchy_root="/immutable/hierarchy", legacy_cache=False,
                           utility_study={"variant": "S_STD", "experiment_id": "test"})
    original_model = deepcopy(cfg["model"])
    result = module.compact_config(cfg, args)
    assert result["model"] == original_model
    assert result["data"] == cfg["data"]
    assert result["training"]["steps"] == 30000 and result["training"]["checkpoint_steps"] == [15000, 30000]
    assert result["training"]["lr"] == 1e-4 and result["training"]["warmup_steps"] == 1000
    assert result["tracking"]["project"] == "chronotree-routers"

    # Extend only the opt-in study; the existing compact launcher keeps its
    # original 30k protocol. Every new arm gets the same 100k decay horizon.
    from stmem.training.checkpoints import resolve_checkpointing
    from hpc.student_utility import make_settings

    args.continuation_updates, args.continuation_lr = 10000, 1e-5
    manifest = {"experiment_id": "test", "teacher_sha256": TEACHER_SHA256,
                "compact_sha256": COMPACT_SHA256, "encoded_cache_id": ENCODED_CACHE_ID,
                "hierarchy_cache_id": HIERARCHY_CACHE_ID,
                "protocol": {"student_training": {
        "steps": 100000, "checkpoint_every": 1000, "snapshot_every": 5000}}}
    for variant in ("S_STD", "S_STAGE", "S_PACKET"):
        args.utility_study = make_settings(manifest, path, variant, args)
        result = module.compact_config(cfg, args)
        train = result["training"]
        assert result["model"] == original_model and result["data"] == cfg["data"]
        assert train["steps"] == train["schedule_steps"] == 100000
        assert train["checkpoint_steps"] == list(range(5000, 100001, 5000))
        assert train["checkpoint_every"] == 1000
        assert resolve_checkpointing(result) == {"periodic_every": 1000, "retain_periodic": False,
                                                "step_basis": "absolute_completed_optimizer_steps"}
        assert train["lr"] == 1e-4 and train["warmup_steps"] == 1000
        assert train["effective_batch"] == 128 and train["microbatch"] == 32
        assert args.utility_study["checkpoint_rule"] == "fixed_100000_endpoint"

    args.utility_study = make_settings(manifest, path, "S_STD", args, initialization="immutable-S0.pt")
    result = module.compact_config(cfg, args)
    assert result["training"]["steps"] == 10000
    assert result["training"]["lr"] == 1e-5 and result["training"]["warmup_steps"] == 0
    assert result["training"]["checkpoint_steps"] == [10000]


def test_student_schedule_preserves_sealed_protocol_and_always_retains_endpoint():
    from hpc.student_utility import manifest_schedule

    manifest = {"protocol": {"student_training": {
        "steps": 100000, "checkpoint_every": 1000, "snapshot_every": 5000}}}
    schedule = manifest_schedule(manifest, SimpleNamespace())
    assert schedule["primary_updates"] == 100000
    assert schedule["development_steps"] == list(range(5000, 100000, 5000))
    assert primary_schedule(12345)["checkpoint_steps"] == [5000, 10000, 12345]
    for field in ("steps", "checkpoint_every", "snapshot_every"):
        with pytest.raises(ValueError, match="sealed study protocol"):
            manifest_schedule(manifest, SimpleNamespace(**{field: 999}))
    assert manifest_schedule({}, SimpleNamespace())["checkpoint_steps"] == [15000, 30000]
    with pytest.raises(ValueError, match="sealed study protocol"):
        manifest_schedule({}, SimpleNamespace(steps=100000))
    for value in (True, 0, -1, 3.5, "100000"):
        for field in ("steps", "checkpoint_every", "snapshot_every"):
            values = {"steps": 100000, field: value}
            with pytest.raises(ValueError, match="positive integer"):
                primary_schedule(**values)


@pytest.mark.parametrize("rebased", [False, True])
def test_student_launcher_renders_three_matched_100k_jobs_without_submission(tmp_path, monkeypatch, rebased):
    from pathlib import Path
    from hpc import student_utility, train_compact
    from stmem.config import load_config

    cfg = load_config("configs/teacher_fv_no_prop_10hz.yaml")
    protocol = load_config("configs/inspection/refinement_utility_pilot.yaml")
    manifest = rebased_manifest() if rebased else {
        "experiment_id": "synthetic-launcher-only", "protocol": protocol,
        "teacher_sha256": TEACHER_SHA256, "compact_sha256": COMPACT_SHA256,
        "encoded_cache_id": ENCODED_CACHE_ID, "hierarchy_cache_id": HIERARCHY_CACHE_ID,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(student_utility, "read_site", lambda _: {"STMEM_REPO_ROOT": str(tmp_path)})
    monkeypatch.setattr(student_utility, "read_study_manifest", lambda _: manifest)
    rendered = {}

    def prepare(args, site, repo, cache_plan, generated):
        assert not args.submit
        assert site["WANDB_PROJECT"] == "chronotree-routers"
        assert args.gpus == 2 and args.cpus == 64 and args.mem == "128G"
        assert cache_plan == tmp_path / "hpc/generated/synthetic_selected_cache/plan.json"
        args.hierarchy_root = "/immutable/hierarchy"
        rendered[args.utility_study["variant"]] = train_compact.compact_config(cfg, args)

    monkeypatch.setattr(student_utility, "prepare", prepare)
    student_utility.main(["--manifest", str(manifest_path), "--cache-name", "synthetic_selected_cache",
                          "--name", "synthetic", "--dry-run"])
    matrix = json.loads((tmp_path / "hpc/generated/synthetic/matrix.json").read_text())
    assert matrix["total_updates"] == 300000 and matrix["total_example_exposures"] == 38400000
    assert set(rendered) == {"S_STD", "S_STAGE", "S_PACKET"}
    expected_teacher = manifest["teacher_sha256"]
    expected_step = 140000 if rebased else 104110
    assert matrix["teacher"] == {"sha256": expected_teacher, "step": expected_step}
    assert matrix["reference"]["variant"] == ("S_INIT" if rebased else "S0")
    if rebased:
        assert matrix["reference"]["compact_updates"] == 0
        assert matrix["primary_endpoints"] == ["S_STD", "S_STAGE", "S_PACKET"]
    for variant, config in rendered.items():
        assert config["training"]["steps"] == config["training"]["schedule_steps"] == 100000
        assert config["training"]["checkpoint_steps"] == list(range(5000, 100001, 5000))
        assert config["training"]["utility_study"]["development_steps"][-1] == 95000
        settings = config["training"]["utility_study"]
        assert settings["teacher_sha256"] == settings["initialization_sha256"] == expected_teacher
        assert settings["teacher_step"] == settings["parent_step"] == expected_step
        assert settings["encoded_cache_id"] == manifest["encoded_cache_id"]
        assert settings["hierarchy_cache_id"] == manifest["hierarchy_cache_id"]
        assert settings["initialization_checkpoint"] is None
        assert settings["optimizer_initialization"] == "fresh"
        assert Path(matrix["students"][variant]["primary_checkpoint"]).name == "step_00100000.pt"


def test_rebased_students_reject_historical_compact_continuation_before_pinning(tmp_path, monkeypatch):
    from hpc import student_utility

    manifest = rebased_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    args = SimpleNamespace(continuation_updates=10000, continuation_lr=1e-5)
    with pytest.raises(ValueError, match="historical S0"):
        student_utility.make_settings(manifest, path, "S_STD", args, initialization="old-S0.pt")
    monkeypatch.setattr(student_utility, "read_site", lambda _: {"STMEM_REPO_ROOT": str(tmp_path)})
    monkeypatch.setattr(student_utility, "read_study_manifest", lambda _: manifest)
    monkeypatch.setattr(student_utility, "prepare", lambda *args: pytest.fail("Must not prepare a continuation"))
    with pytest.raises(SystemExit):
        student_utility.main(["--manifest", str(path), "--name", "forbidden",
                              "--initialization", "continuation", "--initial-checkpoint", "old-S0.pt"])
    assert not (tmp_path / "hpc/generated/forbidden").exists()


def test_rehearsal_parent_uses_same_rebased_teacher_caches_and_experiment():
    from hpc.student_utility_rehearsal import validate_parent

    panel = rebased_manifest()
    dependencies = panel["dependencies"]
    settings = {**dependencies, "experiment_id": panel["experiment_id"], "variant": "S_PACKET"}
    parent = {"stage": "compact", "provenance": {"dependencies": {"teacher": panel["teacher_sha256"]}},
              "config": {"training": {"utility_study": settings}}}
    assert validate_parent(panel, parent)["teacher_step"] == 140000
    for field, wrong in (("experiment_id", "another-study"), ("variant", "S_INIT"),
                         ("teacher_step", 104110), ("teacher_sha256", TEACHER_SHA256),
                         ("encoded_cache_id", ENCODED_CACHE_ID), ("hierarchy_cache_id", HIERARCHY_CACHE_ID)):
        different = deepcopy(parent)
        different["config"]["training"]["utility_study"][field] = wrong
        with pytest.raises(ValueError, match="Rehearsal parent"):
            validate_parent(panel, different)
    for stage, teacher in (("teacher", panel["teacher_sha256"]), ("compact", TEACHER_SHA256)):
        different = deepcopy(parent)
        different.update(stage=stage, provenance={"dependencies": {"teacher": teacher}})
        with pytest.raises(ValueError, match="selected teacher"):
            validate_parent(panel, different)
