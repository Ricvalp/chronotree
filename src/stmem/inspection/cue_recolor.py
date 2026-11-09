"""Observed RGB appearance interventions for an explicitly offline diagnostic.

These masks never enter policy inputs. A caller may constrain edits using
RGB-D-derived visible geometry; no simulator object labels are required.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


CAMERAS = ("front", "wrist")


def _rgb(value):
    value = np.asarray(value)
    if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise ValueError("RGB must be a native uint8 [H,W,3] image")
    return value


def swap_red_blue(rgb, spatial_mask=None):
    """Swap R/B channels at saturated red or blue pixels, preserving everything else.

    The symmetric mask requires a channel above 25% intensity and 1.5 times each
    other channel. Thresholds are part of this diagnostic intervention, not a
    model feature. The operation is exactly involutive, including its mask.
    Optional spatial constraints must be boolean and in the same image pixels.
    """
    rgb = _rgb(rgb)
    color = rgb.astype(np.float32)
    red, green, blue = color[..., 0], color[..., 1], color[..., 2]
    red_mask = (red > 255 * .25) & (red > 1.5 * green) & (red > 1.5 * blue)
    blue_mask = (blue > 255 * .25) & (blue > 1.5 * red) & (blue > 1.5 * green)
    changed = red_mask | blue_mask
    if spatial_mask is not None:
        spatial_mask = np.asarray(spatial_mask)
        if spatial_mask.dtype != np.bool_ or spatial_mask.shape != rgb.shape[:2]:
            raise ValueError("Spatial mask must be boolean [H,W] in the source RGB image")
        changed &= spatial_mask
    edited = rgb.copy()
    edited[changed] = rgb[changed][:, [2, 1, 0]]
    return edited, changed


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def canonical_point_rgb(observation: Mapping, camera_indices, pixel_uv, valid):
    """Sample edited RGB at immutable canonical camera/UV correspondences.

    Returns CPU float32 [N,3] in [0,1]. Invalid point slots remain zero. This is
    direct source-pixel lookup: no reprojection, image resizing, voxelization,
    point sampling or center selection occurs.
    """
    camera, uv, valid = map(_numpy, (camera_indices, pixel_uv, valid))
    count = len(camera)
    if camera.shape != (count,) or uv.shape != (count, 2) or valid.shape != (count,):
        raise ValueError("Require camera[N], original pixel_uv[N,2], valid[N]")
    if camera.dtype.kind not in "iu" or uv.dtype.kind not in "iu" or valid.dtype != np.bool_:
        raise ValueError("Camera/UV must be integer and validity boolean")
    if np.any((camera[valid] < 0) | (camera[valid] >= len(CAMERAS))):
        raise ValueError("Valid point references an unknown observing camera")
    sampled = np.zeros((count, 3), dtype=np.float32)
    for number, name in enumerate(CAMERAS):
        image = _rgb(observation[f"{name}_rgb"])
        selected = valid & (camera == number)
        pixels = uv[selected]
        height, width = image.shape[:2]
        if (np.any(pixels < 0) or np.any(pixels[:, 0] >= width)
                or np.any(pixels[:, 1] >= height)):
            raise ValueError("Valid point pixel lies outside its original RGB image")
        sampled[selected] = image[pixels[:, 1], pixels[:, 0]].astype(np.float32) / 255.0
    return torch.from_numpy(sampled)
