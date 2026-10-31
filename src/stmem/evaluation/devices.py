"""Explicit CUDA/Vulkan placement for independent evaluation worker processes.

CUDA indices are relative to the Slurm-visible allocation. Worker subprocesses
receive physical GPU UUIDs, then use logical CUDA device zero for both policy
inference and SAPIEN rendering. The physical PCI identity is checked afterward;
Vulkan enumeration order is never assumed to equal CUDA enumeration order.
"""

from __future__ import annotations

import os
import re
from types import FunctionType, MethodType


def gpu_uuid(value):
    value = str(value)
    if value.startswith("GPU-"):
        value = value[4:]
    if not re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
        raise ValueError(f"A complete physical GPU UUID is required, got {value!r}")
    return "GPU-" + value.lower()


def pci_address(value):
    """Normalize Nvidia's eight-digit domain and SAPIEN's four-digit domain."""
    match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", str(value))
    if not match:
        raise ValueError(f"Invalid GPU PCI address: {value!r}")
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function}"


def _properties_identity(index, properties):
    # These properties are supplied by the pinned Torch 2.9 CUDA runtime, whose
    # visible ordinals already obey CUDA_VISIBLE_DEVICES/Slurm isolation.
    return {
        "visible_index": index,
        "uuid": gpu_uuid(properties.uuid),
        "pci_bus_id": pci_address(
            f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:{properties.pci_device_id:02x}.0"
        ),
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
    }


def discover_devices(num_gpus):
    import torch

    if not isinstance(num_gpus, int) or isinstance(num_gpus, bool) or num_gpus < 1:
        raise ValueError("num_gpus must be a positive integer")
    available = torch.cuda.device_count()
    if num_gpus > available:
        raise ValueError(f"Requested {num_gpus} GPUs, but the allocation exposes only {available}")
    devices = [_properties_identity(i, torch.cuda.get_device_properties(i)) for i in range(num_gpus)]
    if len({device["uuid"] for device in devices}) != num_gpus:
        raise ValueError("Visible GPU UUIDs are not distinct physical devices")
    return devices


def worker_device_environment(device):
    """Use only a UUID discovered inside this allocation, never a host ordinal."""
    return {
        "CUDA_VISIBLE_DEVICES": gpu_uuid(device["uuid"]),
        "STMEM_EXPECTED_GPU_UUID": gpu_uuid(device["uuid"]),
        "STMEM_EXPECTED_GPU_PCI": pci_address(device["pci_bus_id"]),
    }


def verify_torch_device(render_device):
    import torch

    if not isinstance(render_device, str) or not re.fullmatch(r"cuda:[0-9]+", render_device):
        raise ValueError("Explicit evaluation render device must be a logical CUDA device such as cuda:0")
    index = int(render_device.split(":")[1])
    if not torch.cuda.is_available():
        raise RuntimeError("Explicit GPU rendering requires a working CUDA device")
    if index != torch.cuda.current_device():
        raise ValueError("Rendering must use the same logical CUDA device as policy inference")
    actual = _properties_identity(index, torch.cuda.get_device_properties(index))
    expected_uuid = os.environ.get("STMEM_EXPECTED_GPU_UUID")
    expected_pci = os.environ.get("STMEM_EXPECTED_GPU_PCI")
    if expected_uuid is not None or expected_pci is not None:
        if not expected_uuid or not expected_pci:
            raise ValueError("Worker GPU binding must include both UUID and PCI identity")
        if torch.cuda.device_count() != 1:
            raise ValueError("Parallel evaluation workers must see exactly one CUDA GPU")
        if actual["uuid"] != gpu_uuid(expected_uuid) or actual["pci_bus_id"] != pci_address(expected_pci):
            raise ValueError("Worker CUDA device differs from the allocation's assigned physical GPU")
    return actual


def make_env_with_render_device(builder, episode_idx, render_device, **observation_flags):
    """Run the pinned official builder unchanged except for its render backend.

    The pinned builder does not expose gym.make's render_backend argument. A
    private function namespace injects it at that single factory call, retaining
    the original seed/difficulty resolution, observation flags and wrappers.
    No third-party file or shared module global is modified, even on exceptions.
    """
    original = builder.make_env_for_episode
    function = original.__func__
    gym = function.__globals__["gym"]

    class RenderFactory:
        def __getattr__(self, name):
            return getattr(gym, name)

        def make(self, env_id, *args, **kwargs):
            if env_id != builder.env_id:
                raise ValueError("Official builder attempted to construct an unexpected task")
            if "render_backend" in kwargs and kwargs["render_backend"] != render_device:
                raise ValueError("Official builder already selected a different rendering backend")
            kwargs["render_backend"] = render_device
            return gym.make(env_id, *args, **kwargs)

    namespace = dict(function.__globals__, gym=RenderFactory())
    bound = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return MethodType(bound, builder)(episode_idx, **observation_flags)


def verify_render_device(env, render_device, torch_identity):
    unwrapped = getattr(env, "unwrapped", env)
    backend = getattr(unwrapped, "backend", None)
    selected = getattr(backend, "render_device", None)
    if selected is None or not selected.is_cuda() or not selected.can_render():
        raise RuntimeError("Explicit CUDA renderer did not resolve to a render-capable CUDA device")
    actual_pci = pci_address(selected.pci_string)
    if actual_pci != torch_identity["pci_bus_id"] or selected.cuda_id != torch_identity["visible_index"]:
        raise RuntimeError("SAPIEN rendering and Torch inference selected different physical GPUs")
    return {
        "requested_render_device": render_device,
        "torch": torch_identity,
        "renderer": {"name": selected.name, "cuda_id": selected.cuda_id, "pci_bus_id": actual_pci},
        "simulation_backend": backend.sim_backend,
        "verified_same_physical_gpu": True,
    }
