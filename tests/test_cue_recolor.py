"""Appearance edits must preserve unaffected pixels and canonical point order."""
import numpy as np
import pytest
import torch

from stmem.inspection.cue_recolor import canonical_point_rgb, swap_red_blue


def test_swap_is_involutive_and_preserves_green_gray_and_dim_pixels():
    original = np.array([[[220, 20, 10], [10, 20, 220], [20, 230, 15],
                          [150, 150, 150], [40, 5, 5], [255, 200, 0]]], dtype=np.uint8)
    snapshot = original.copy()
    changed, mask = swap_red_blue(original)
    assert mask.tolist() == [[True, True, False, False, False, False]]
    assert changed[0, :2].tolist() == [[10, 20, 220], [220, 20, 10]]
    assert np.array_equal(changed[~mask], original[~mask])
    assert np.array_equal(original, snapshot)
    restored, second_mask = swap_red_blue(changed)
    assert np.array_equal(restored, original)
    assert np.array_equal(second_mask, mask)


def test_spatial_constraint_prevents_unrelated_color_edits():
    image = np.full((2, 2, 3), [250, 20, 10], dtype=np.uint8)
    region = np.array([[True, False], [False, False]])
    edited, mask = swap_red_blue(image, region)
    assert np.array_equal(mask, region)
    assert edited[0, 0].tolist() == [10, 20, 250]
    assert np.array_equal(edited[~region], image[~region])
    with pytest.raises(ValueError, match="boolean"):
        swap_red_blue(image, region.astype(int))
    with pytest.raises(ValueError, match="boolean"):
        swap_red_blue(image, np.ones((2, 3), dtype=bool))
    with pytest.raises(ValueError, match="uint8"):
        swap_red_blue(image.astype(float) / 255)


def test_point_rgb_uses_camera_ownership_and_xy_pixel_order_without_padding_reads():
    front, wrist = np.zeros((2, 3, 3), dtype=np.uint8), np.zeros((2, 3, 3), dtype=np.uint8)
    front[1, 2] = [255, 0, 128]
    wrist[0, 1] = [0, 64, 255]
    obs = {"front_rgb": front, "wrist_rgb": wrist}
    result = canonical_point_rgb(obs, torch.tensor([0, 1, -1]),
                                 torch.tensor([[2, 1], [1, 0], [-99, 900]]),
                                 torch.tensor([True, True, False]))
    torch.testing.assert_close(result, torch.tensor([[1., 0., 128/255], [0., 64/255, 1.], [0., 0., 0.]]))
    assert result.dtype == torch.float32 and result.device.type == "cpu"


@pytest.mark.parametrize("camera,uv,valid,message", [
    ([2], [[0, 0]], [True], "unknown"),
    ([0], [[3, 0]], [True], "outside"),
    ([0], [[0, -1]], [True], "outside"),
    ([0.], [[0, 0]], [True], "integer"),
    ([0], [[0, 0]], [1], "boolean"),
])
def test_valid_point_correspondence_errors_are_not_silently_clipped(camera, uv, valid, message):
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=message):
        canonical_point_rgb({"front_rgb": image, "wrist_rgb": image}, camera, uv, valid)
