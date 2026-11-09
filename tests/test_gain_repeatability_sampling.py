"""A noise audit must never silently resample its memory candidates."""
import torch

from stmem.inspection.gain_repeatability import fixed_noise_items


def test_noise_resampling_keeps_all_evidence_and_moves_fixed():
    before, after, current, full = object(), object(), object(), object()
    reader = {"before": before, "after": [after], "costs": torch.tensor([1.]),
              "candidate_ids": ["parent"], "frontier_ids": ["parent", "other"]}
    prepared = {"seed": 7, "current": current, "full": full,
                "readers": {"tree": reader}, "target": torch.zeros(16, 8)}
    items = fixed_noise_items(prepared, [11, 12, 13])
    assert [item["seed"] for item in items] == [11, 12, 13]
    for item in items:
        assert item["current"] is current and item["full"] is full
        assert item["target"] is prepared["target"]
        assert item["readers"]["tree"]["before"] is before
        assert item["readers"]["tree"]["after"] == [after, before]
        assert item["readers"]["tree"]["candidate_ids"] is reader["candidate_ids"]
        assert item["readers"]["tree"]["frontier_ids"] is reader["frontier_ids"]
    assert prepared["seed"] == 7
    assert reader["after"] == [after]
    assert len(reader["costs"]) == 1


def test_noop_is_optional_and_empty_reader_stays_empty():
    empty = {"before": object(), "after": [], "costs": torch.empty(0)}
    prepared = {"seed": 7, "readers": {"flat": empty}}
    assert fixed_noise_items(prepared, [8])[0]["readers"]["flat"]["after"] == []
    before, after = object(), object()
    prepared["readers"]["flat"] = {"before": before, "after": [after], "costs": torch.ones(1)}
    item = fixed_noise_items(prepared, [8], noop=False)[0]
    assert item["readers"]["flat"]["after"] == [after]
