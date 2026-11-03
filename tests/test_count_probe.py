"""Synthetic CPU invariants for the fixed-scene count experiment, not performance."""

import numpy as np
import pytest
import torch

from stmem.data.canonical import Frame
from stmem.inspection.count_probe import build_cases, summarize, tensor_state_hash, trajectory_metrics, valid_targets


def source():
    frames = []
    for i in range(18):
        frames.append(Frame(torch.full((4, 6), float(i)), torch.ones(4, dtype=torch.bool),
                            torch.arange(15).float(), torch.zeros(8), False, float(2*i),
                            "execution", f"real-frame-{i}", 2*i))
    return {"execution": frames, "boundaries": [
        {"completed_cycles": k, "frame_index": i} for k, i in zip((1, 2, 3), (4, 8, 12))],
        "checkpoint_config": {"data": {"observation_stride": 2}}}


def test_cases_hold_current_and_clock_and_never_include_future_expert_frames():
    data = source()
    cases = build_cases(data)
    assert len(cases) == 9
    for case in cases:
        end = 4*case.k
        assert all(float(f.points[0, 0]) < end for f in case.prefix)
        assert all(f.time < case.current.time for f in case.prefix)
        assert len({f.event_id for f in case.prefix}) == len(case.prefix)
        assert all(b.time-a.time == 2 for a, b in zip(case.prefix, case.prefix[1:]))
        if case.mode == "natural":
            assert case.current is data["execution"][end]
            assert case.prefix == data["execution"][:end]
        else:
            assert case.current.points is data["execution"][4].points
            assert case.current.time == 24
            assert case.prefix[-1].time == 22
        if case.mode == "fixed_duration":
            assert len(case.prefix) == 12
            assert case.padding_frames == 12-end
            assert case.prefix[0].time == 0
        assert [f.event_id for f in case.prefix[case.padding_frames:]] == [
            f.event_id for f in data["execution"][:end]]
    assert [f.time for f in data["execution"]] == list(range(0, 36, 2))
    assert [f.event_id for f in data["execution"]] == [f"real-frame-{i}" for i in range(18)]
    fixed = [case for case in cases if case.mode != "natural"]
    assert all(case.current is fixed[0].current for case in fixed)
    zero_pad = [case for case in fixed if case.k == 3]
    assert [(f.event_id, f.time) for f in zero_pad[0].prefix] == [
        (f.event_id, f.time) for f in zero_pad[1].prefix]


def test_invalid_history_order_and_missing_cycle_are_rejected():
    data = source()
    data["boundaries"][1]["frame_index"] = 3
    with pytest.raises(ValueError, match="boundaries"):
        build_cases(data)
    data = source()
    data["execution"][2].time = 1
    with pytest.raises(ValueError, match="clocks"):
        build_cases(data)
    data = source()
    data["boundaries"].pop()
    with pytest.raises(ValueError, match="three"):
        build_cases(data)


def test_overshoot_is_excluded_from_valid_expert_continuations():
    assert [(k, x) for k in (1, 2, 3) for x in valid_targets(k)] == [
        (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    with pytest.raises(ValueError):
        valid_targets(0)


def test_geometric_preference_is_signed_progress_and_first8_is_actual_prefix():
    xyz = np.zeros((16, 3))
    xyz[:8, 0] = np.arange(1, 9)*0.01
    xyz[8:, 0] = -1
    row = trajectory_metrics(xyz, [0, 0, 0], [1, 0, 0], [-1, 0, 0])
    assert row["tcp_displacement_first8_mm"] == pytest.approx(80)
    assert row["button_closing_first8_mm"] == pytest.approx(80)
    assert row["cube_closing_first8_mm"] == pytest.approx(-80)
    assert row["button_preference_first8_mm"] == pytest.approx(160)
    assert row["button_distance_end_mm"] == pytest.approx(2000)


def test_frozen_fingerprint_includes_scalar_buffers_but_ignores_decoder():
    a = {"encoder.temperature": torch.tensor(0.5), "text_encoder.weight": torch.eye(2),
         "decoder.weight": torch.ones(4)}
    b = dict(a, **{"decoder.weight": torch.zeros(4)})
    prefix = ("encoder.", "text_encoder.")
    assert tensor_state_hash(a, prefix) == tensor_state_hash(b, prefix)
    b["encoder.temperature"] = torch.tensor(0.6)
    assert tensor_state_hash(a, prefix) != tensor_state_hash(b, prefix)


def test_paired_contrasts_match_noise_and_do_not_compare_different_current_scenes():
    rows = []
    for mode in ("natural", "fixed_current"):
        for k in (1, 2):
            for x in valid_targets(k):
                rows.append({"label": "teacher", "history_mode": mode, "k": k, "x": x,
                             "noise_seed": 2701, "expected_next": "press_button" if k == x else "pick_cube",
                             "tcp_positions_m": np.full((16, 3), (k-x)*0.01).tolist(),
                             "metrics": {"button_preference_first8_mm": float(k-x)}})
    output = summarize(rows)
    history = [r for r in output["paired_contrasts"] if r["kind"].startswith("history")]
    assert len(history) == 1
    assert history[0]["history_mode"] == "fixed_current"
    assert history[0]["press_case"] == {"k": 2, "x": 2}
    assert history[0]["pick_case"] == {"k": 1, "x": 2}
    assert history[0]["mean_first8_tcp_change_mm"] == pytest.approx(np.sqrt(3)*10)
