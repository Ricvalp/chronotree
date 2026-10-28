"""Allowlisted continuous joint-action adapter for pinned RoboMME.

No simulator object state, oracle language, episode identifiers or outcome flags are
placed in Frame. Reset's conditioning history is retained and its actions masked.
"""

from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field
import uuid
import time
from functools import wraps
import numpy as np
import torch
from stmem.data.canonical import Frame
from stmem.data.pointcloud import (
    DEFAULT_CROP,
    rgbd_to_points,
    rgbd_to_points_with_provenance,
    proprioception,
    resolve_point_sampling,
)
from stmem.data.center_sampling import (
    resolve_center_sampling,
    center_sampling_signature,
    prepare_center_indices,
)

JOINT_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
JOINT_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float32)
OBSERVATION_FIELDS = (
    "front_rgb",
    "wrist_rgb",
    "front_depth",
    "wrist_depth",
    "front_camera_extrinsic",
    "wrist_camera_extrinsic",
    "joint_state",
    "eef_state",
    "gripper_state",
)
TERMINAL_STATUSES = {"success", "fail", "timeout", "error"}


def numpy_value(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def flag(value):
    a = numpy_value(value).reshape(-1)
    if len(a) != 1:
        raise ValueError("Expected scalar termination flag")
    return bool(a[0])


def physical_action(action, task, *, clip=True):
    a = numpy_value(action).astype(np.float32).reshape(-1)
    if a.shape != (8,) or not np.isfinite(a).all():
        raise ValueError("Policy command must be finite shape [8]")
    a = a.copy()
    if clip:
        a[:7] = np.clip(a[:7], JOINT_LOW, JOINT_HIGH)
    elif np.any(a[:7] < JOINT_LOW) or np.any(a[:7] > JOINT_HIGH):
        raise ValueError("Joint command outside Panda position limits")
    a[7] = 0 if task in ("PatternLock", "RouteStick") else (1 if a[7] >= 0 else -1)
    return a


@dataclass
class ResetResult:
    conditioning: list[Frame]
    current: Frame
    goal: str
    metadata: dict = dataclass_field(default_factory=dict)


@dataclass
class StepResult:
    frames: list[Frame]
    status: str
    terminated: bool
    truncated: bool
    error: str | None = None
    command: np.ndarray | None = None

    @property
    def done(self):
        return self.terminated or self.truncated or self.status in TERMINAL_STATUSES


def measure_preprocessing(method):
    @wraps(method)
    def measured(self, *args, **kwargs):
        self.last_preprocess_seconds = 0.0
        started = time.perf_counter()
        try:
            return method(self, *args, **kwargs)
        finally:
            self.last_preprocess_seconds = time.perf_counter() - started

    return measured


class RoboMMEAdapter:
    def __init__(
        self,
        task,
        episode_idx=0,
        split="val",
        points=2048,
        max_steps=1300,
        crop=DEFAULT_CROP,
        voxel_size=0.005,
        env=None,
        max_depth_m=4.0,
        point_sampling=None,
        center_sampling=None,
        center_counts=(32,),
        feature_extractor=None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown official split")
        self.task = task
        self.episode_idx = episode_idx
        self.split = split
        self.points = points
        self.crop = crop
        self.voxel_size = voxel_size
        self.max_depth_m = max_depth_m
        self.point_sampling = resolve_point_sampling(point_sampling)
        self.center_sampling = resolve_center_sampling(center_sampling)
        self.center_counts = tuple(center_counts)
        self.feature_extractor = feature_extractor
        self.center_sampling_signature = center_sampling_signature(self.center_sampling)
        self.requested_max_steps = max_steps
        self.env = env
        self.metadata = {}
        self.calibration = {}
        self.goal = ""
        self._seen = {}
        self._clock = {"conditioning": 0, "execution": 0}
        self._step = 0
        self._reset_id = ""
        self.last_rgb = []
        self.last_raw_observation = None
        self.last_preprocess_seconds = 0.0
        if self.env is None:
            try:
                import robomme.robomme_env  # noqa: F401 -- official task registration
                from robomme.env_record_wrapper import BenchmarkEnvBuilder
            except ImportError as exc:
                raise RuntimeError(
                    "Pinned RoboMME simulator is unavailable. Run bash scripts/setup_sim.sh, then execute with .venv/bin/python (or uv run --extra simulator python)."
                ) from exc
            builder = BenchmarkEnvBuilder(
                env_id=task, dataset=split, action_space="joint_angle", gui_render=False, max_steps=max_steps
            )
            if (task, episode_idx) not in builder.metadata_index:
                raise ValueError(f"No official {split} seed for {task}/{episode_idx}")
            self.env = builder.make_env_for_episode(
                episode_idx,
                include_front_depth=True,
                include_wrist_depth=True,
                include_front_camera_extrinsic=True,
                include_wrist_camera_extrinsic=True,
                include_front_camera_intrinsic=True,
                include_wrist_camera_intrinsic=True,
                include_maniskill_obs=False,
                include_available_multi_choices=False,
            )

    def _runtime_metadata(self):
        unwrapped = getattr(self.env, "unwrapped", self.env)
        result = {
            "requested_max_steps": self.requested_max_steps,
            "action_space": "joint_angle",
            "clock": "source_local_recorded_frame",
        }
        for output, attr in [("control_hz", "control_freq"), ("physics_hz", "sim_freq")]:
            value = getattr(unwrapped, attr, None)
            if value is not None:
                result[output] = float(value)
        node = self.env
        for _ in range(16):
            if "max_steps_without_demonstration" in vars(node):
                result["effective_max_steps"] = int(node.max_steps_without_demonstration)
            if not hasattr(node, "env"):
                break
            node = node.env
        return result

    @measure_preprocessing
    def ingest(self, obs, info, *, event_id, reset=False, command=None):
        """Event identity belongs to protocol transport, not image/pose deduplication.

        Equal consecutive images are still distinct acquired observations. Repeated
        delivery of the same event returns no new frames and never advances clocks.
        """
        if event_id in self._seen:
            return []
        if not obs:
            self._seen[event_id] = True
            return []
        if "front_rgb_list" not in obs:
            raise ValueError("Nonempty observation lacks front_rgb_list")
        n = len(obs["front_rgb_list"])
        if n == 0:
            self._seen[event_id] = True
            return []
        for field in OBSERVATION_FIELDS:
            if field + "_list" not in obs or len(obs[field + "_list"]) != n:
                raise ValueError(f"Missing/misaligned allowed observation field: {field}")
        frames = []
        self.last_rgb = []
        for i in range(n):
            source = "conditioning" if reset and i < n - 1 else "execution"
            allowed = {field: numpy_value(obs[field + "_list"][i]) for field in OBSERVATION_FIELDS}
            cloud = (
                rgbd_to_points_with_provenance if self.feature_extractor is not None else rgbd_to_points
            )(
                allowed,
                self.calibration,
                count=self.points,
                crop=self.crop,
                voxel_size=self.voxel_size,
                max_depth_m=self.max_depth_m,
                **self.point_sampling,
            )
            points, mask = cloud[:2]
            feature_kwargs = {}
            if self.feature_extractor is not None:
                camera, uv = cloud[2:]
                feature_kwargs = {
                    "visual_features": self.feature_extractor.sample(allowed, camera, uv, mask).cpu(),
                    "visual_feature_signature": self.feature_extractor.signature,
                    "point_camera": camera,
                    "point_uv": uv,
                }
            action_known = not reset and command is not None and n == 1
            previous = (
                torch.as_tensor(command, dtype=torch.float32).clone() if action_known else torch.zeros(8)
            )
            index = self._clock[source]
            frames.append(
                Frame(
                    points,
                    mask,
                    proprioception(allowed),
                    previous,
                    action_known,
                    float(index),
                    source,
                    f"{event_id}/{i}",
                    index,
                    center_indices=prepare_center_indices(
                        points, mask, self.center_counts, self.center_sampling
                    ),
                    center_sampling_signature=self.center_sampling_signature,
                    **feature_kwargs,
                )
            )
            self._clock[source] += 1
            self.last_rgb.append(
                np.concatenate([allowed["front_rgb"], allowed["wrist_rgb"]], axis=1).astype(np.uint8)
            )
        self._seen[event_id] = True
        return frames

    def reset(self):
        self._seen.clear()
        self._clock = {"conditioning": 0, "execution": 0}
        self._step = 0
        self._reset_id = uuid.uuid4().hex
        obs, info = self.env.reset()
        status = info.get("status", "ongoing")
        if not obs:
            raise RuntimeError(
                f"Simulator reset returned no observation: {status}: {info.get('error_message', '')}"
            )
        self.calibration = {
            f"{c}_camera_intrinsic": numpy_value(info[f"{c}_camera_intrinsic"]) for c in ("front", "wrist")
        }
        raw_goal = info.get("task_goal", "")
        self.goal = str(raw_goal[0] if isinstance(raw_goal, (list, tuple)) and raw_goal else raw_goal)
        frames = self.ingest(obs, info, event_id=f"{self._reset_id}/reset", reset=True)
        if not frames or frames[-1].source != "execution":
            raise ValueError("Reset requires one initial execution observation")
        self.metadata = self._runtime_metadata()
        self.metadata["conditioning_frames"] = len(frames) - 1
        self.last_raw_observation = obs
        return ResetResult(frames[:-1], frames[-1], self.goal, self.metadata.copy())

    def step(self, action):
        command = physical_action(action, self.task)
        actual = command[:7] if self.task in ("PatternLock", "RouteStick") else command
        obs, reward, terminated, truncated, info = self.env.step(actual)
        # Outcome is inspected before any attempt to index an empty terminal obs.
        status = str(info.get("status", "ongoing"))
        terminated = flag(terminated)
        truncated = flag(truncated)
        self._step += 1
        frames = self.ingest(obs, info, event_id=f"{self._reset_id}/step/{self._step}", command=command)
        self.last_raw_observation = obs
        return StepResult(frames, status, terminated, truncated, info.get("error_message"), command)

    def close(self):
        if self.env is not None:
            self.env.close()
