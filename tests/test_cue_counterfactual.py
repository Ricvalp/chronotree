"""Observed RGB-D masks must not edit unrelated or invalid pixels."""
import numpy as np

from stmem.inspection.cue_counterfactual import spatial_pixel_mask


def test_edit_region_uses_world_geometry_and_positive_depth():
    # Camera translated +10cm in world-to-camera coordinates: its central
    # pixel therefore backprojects at world x=-10cm, not x=+10cm.
    observation = {
        "front_rgb": np.full((2, 2, 3), 100, dtype=np.uint8),
        "front_depth": np.array([[30, 30], [0, 200]], dtype=np.int16),
        "front_camera_extrinsic": np.concatenate((np.eye(3), np.array([[.1], [0], [0]])), axis=1),
    }
    mask = spatial_pixel_mask(observation, "front", np.eye(3), [[-.1, 0, .03]], .005)
    assert mask.tolist() == [[True, False], [False, False]]
    opposite = spatial_pixel_mask(observation, "front", np.eye(3), [[.1, 0, .03]], .005)
    assert not opposite.any()
