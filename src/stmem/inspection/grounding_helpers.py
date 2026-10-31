"""Small, diagnostic-only geometry helpers for recorded PickXtimes probes."""

from pathlib import Path

import numpy as np
import torch

from stmem.evaluation.robomme_adapter import numpy_value


def json_default(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return numpy_value(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def cube_state(env):
    """Oracle geometry used for labels only, never passed to the policy."""
    cube = env.target_cube
    return {
        "p": numpy_value(cube.pose.p).reshape(-1, 3)[0].copy(),
        "q": numpy_value(cube.pose.q).reshape(-1, 4)[0].copy(),
        "half_size": np.broadcast_to(numpy_value(env.cube_half_size), (3,)).copy(),
    }


def cube_mask(points, cube, tolerance=0.003):
    """Diagnostic oriented-box membership; this never selects encoder points."""
    q = np.asarray(cube["q"], dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    local = (np.asarray(points)[..., :3] - np.asarray(cube["p"])) @ rotation
    return (np.abs(local) <= np.asarray(cube["half_size"]) + tolerance).all(-1)


def verified_existing(path, identity, *, resume):
    """Do not silently reuse another checkpoint/protocol's completed output."""
    import json

    path = Path(path)
    if not path.exists():
        return None
    if not resume:
        raise FileExistsError(f"Existing diagnostic: {path}; use --resume")
    value = json.loads(path.read_text())
    for key, expected in identity.items():
        if value.get(key) != expected:
            raise ValueError(f"Diagnostic identity mismatch in {path}: {key}")
    return value if value.get("diagnostic_complete") else None
