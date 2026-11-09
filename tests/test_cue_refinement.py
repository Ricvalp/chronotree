"""Units and observation-only annotations for the bounded cue diagnostic."""
import numpy as np
import pytest

from stmem.inspection.cue_refinement import observed_color_centers, paired_path_metrics


def test_path_changes_use_all_valid_targets_and_report_centimetres():
    a = np.zeros((2, 3, 3))
    b = a.copy()
    b[:, 0, 0] = .01
    b[:, 1, 0] = .03
    b[:, 2, 0] = 100  # Padding must not dominate the reported chunk difference.
    actions = np.zeros((2, 3, 8))
    changed = actions.copy()
    changed[:, :2] = 2
    changed[:, 2] = 100
    valid = np.ones((3, 8), dtype=bool)
    valid[2] = False
    metrics = paired_path_metrics(a, b, actions, changed, valid, np.array([.1, 0]))
    assert metrics["mean_tcp_delta_cm"] == pytest.approx([2, 2])
    assert metrics["max_tcp_delta_cm"] == pytest.approx([3, 3])
    assert metrics["endpoint_delta_cm"] == pytest.approx([3, 3])
    assert metrics["normalized_action_rmse"] == pytest.approx([2, 2])
    assert metrics["endpoint_target_distance_a_cm"] == pytest.approx([10, 10])
    assert metrics["endpoint_target_distance_b_cm"] == pytest.approx([7, 7])


def test_noop_is_zero_and_draw_mismatch_is_rejected():
    paths, actions = np.ones((2, 16, 3)), np.ones((2, 16, 8))
    valid = np.ones((16, 8), dtype=bool)
    values = paired_path_metrics(paths, paths, actions, actions, valid, [0, 0])
    for key in ("mean_tcp_delta_cm", "max_tcp_delta_cm", "endpoint_delta_cm", "normalized_action_rmse"):
        assert values[key] == [0, 0]
    with pytest.raises(ValueError, match="same"):
        paired_path_metrics(paths[:1], paths, actions, actions, valid, [0, 0])


def test_color_annotation_ignores_robot_height_and_handles_uint_rgb():
    blue = [[.1, -.1, .03, 0, 0, 255]] * 6
    red = [[-.05, .05, .03, 255, 0, 0]] * 6
    robot = [[-.05, .05, .5, 255, 0, 0]] * 100
    values = observed_color_centers({"points": blue + red + robot})
    assert values["blue"]["points"] == values["red"]["points"] == 6
    assert values["blue"]["center"] == pytest.approx([.1, -.1, .03])
    assert values["red"]["center"] == pytest.approx([-.05, .05, .03])
    with pytest.raises(ValueError, match="blue"):
        observed_color_centers({"points": red + robot})
