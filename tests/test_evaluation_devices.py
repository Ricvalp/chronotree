"""CPU checks of CUDA/Vulkan binding and unchanged official-builder semantics."""

from types import SimpleNamespace
import sys

import pytest
import torch

from stmem.evaluation import devices
from stmem.evaluation.robomme_adapter import RoboMMEAdapter

UUID_A = "GPU-11111111-2222-3333-4444-555555555555"
UUID_B = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def properties(uuid=UUID_A, bus=10):
    return SimpleNamespace(uuid=uuid, pci_domain_id=0, pci_bus_id=bus, pci_device_id=0,
                           name="Synthetic device identity", total_memory=1024)


def fake_cuda(monkeypatch, values):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(values))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: values[i])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(values))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


def test_uses_allocation_visible_uuid_order_not_host_ordinals(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,3")
    fake_cuda(monkeypatch, [properties(UUID_B, 42), properties(UUID_A, 17)])
    found = devices.discover_devices(2)
    assert [d["uuid"] for d in found] == [UUID_B, UUID_A]
    assert [d["pci_bus_id"] for d in found] == ["0000:2a:00.0", "0000:11:00.0"]
    child = devices.worker_device_environment(found[1])
    assert child == {"CUDA_VISIBLE_DEVICES": UUID_A, "STMEM_EXPECTED_GPU_UUID": UUID_A,
                     "STMEM_EXPECTED_GPU_PCI": "0000:11:00.0"}
    with pytest.raises(ValueError, match="exposes only 2"):
        devices.discover_devices(3)


def test_worker_rejects_gpu_identity_or_mask_mismatch(monkeypatch):
    fake_cuda(monkeypatch, [properties()])
    monkeypatch.setenv("STMEM_EXPECTED_GPU_UUID", UUID_A)
    monkeypatch.setenv("STMEM_EXPECTED_GPU_PCI", "00000000:0a:00.0")
    assert devices.verify_torch_device("cuda:0")["uuid"] == UUID_A
    monkeypatch.setenv("STMEM_EXPECTED_GPU_UUID", UUID_B)
    with pytest.raises(ValueError, match="assigned physical GPU"):
        devices.verify_torch_device("cuda:0")
    fake_cuda(monkeypatch, [properties(), properties(UUID_B, 11)])
    with pytest.raises(ValueError, match="exactly one"):
        devices.verify_torch_device("cuda:0")


def test_invalid_gpu_aliases_and_mig_are_not_silently_reinterpreted(monkeypatch):
    fake_cuda(monkeypatch, [properties()])
    for alias in ["cuda", "gpu", "pci:0000:0a:00.0", "cuda:0;echo hello"]:
        with pytest.raises(ValueError, match="logical CUDA device"):
            devices.verify_torch_device(alias)
    with pytest.raises(ValueError, match="same logical"):
        devices.verify_torch_device("cuda:1")
    with pytest.raises(ValueError, match="complete physical"):
        devices.gpu_uuid("MIG-" + UUID_A)


gym = None


class SyntheticOfficialBuilder:
    env_id = "VideoUnmask"

    def make_env_for_episode(self, episode_idx, *, include_front_depth=False):
        # Mimic the pinned builder's globals reference, metadata and wrapping.
        env = gym.make(self.env_id, seed=700 + episode_idx, difficulty="fixed",
                       obs_mode="rgb+depth+segmentation", control_mode="pd_joint_pos",
                       render_mode="rgb_array", reward_mode="dense")
        return {"env": env, "include_front_depth": include_front_depth, "wrapper": "official"}


def test_private_render_factory_preserves_seed_wrappers_and_shared_globals(monkeypatch):
    calls = []
    factory = SimpleNamespace(make=lambda task, **kwargs: calls.append((task, kwargs)) or "raw environment")
    monkeypatch.setattr(sys.modules[__name__], "gym", factory)
    builder = SyntheticOfficialBuilder()
    result = devices.make_env_with_render_device(builder, 3, "cuda:0", include_front_depth=True)
    assert result == {"env": "raw environment", "include_front_depth": True, "wrapper": "official"}
    assert calls == [("VideoUnmask", {"seed": 703, "difficulty": "fixed",
                     "obs_mode": "rgb+depth+segmentation", "control_mode": "pd_joint_pos",
                     "render_mode": "rgb_array", "reward_mode": "dense", "render_backend": "cuda:0"})]
    assert SyntheticOfficialBuilder.make_env_for_episode.__globals__["gym"] is factory
    calls.clear()
    builder.make_env_for_episode(3)
    assert "render_backend" not in calls[0][1]


def test_private_render_factory_does_not_leak_on_exception(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("renderer failed")

    factory = SimpleNamespace(make=fail)
    monkeypatch.setattr(sys.modules[__name__], "gym", factory)
    with pytest.raises(RuntimeError, match="renderer failed"):
        devices.make_env_with_render_device(SyntheticOfficialBuilder(), 0, "cuda:0")
    assert SyntheticOfficialBuilder.make_env_for_episode.__globals__["gym"] is factory


def rendered_environment(pci="0000:0a:00.0", cuda_id=0):
    device = SimpleNamespace(pci_string=pci, cuda_id=cuda_id, name="Synthetic renderer",
                             is_cuda=lambda: True, can_render=lambda: True)
    environment = SimpleNamespace(backend=SimpleNamespace(render_device=device, sim_backend="physx_cpu"))
    environment.closed = False
    environment.close = lambda: setattr(environment, "closed", True)
    return environment


def test_physical_renderer_is_verified_and_mismatched_environment_closed(monkeypatch):
    identity = devices._properties_identity(0, properties())
    monkeypatch.setattr(devices, "verify_torch_device", lambda _: identity)
    env = rendered_environment()
    adapter = RoboMMEAdapter("VideoUnmask", env=env, points=8, crop=None, render_device="cuda:0")
    record = adapter._runtime_metadata()["render_binding"]
    assert record["verified_same_physical_gpu"] is True
    assert record["simulation_backend"] == "physx_cpu"
    assert record["renderer"]["pci_bus_id"] == identity["pci_bus_id"]
    wrong = rendered_environment(pci="0000:0b:00.0")
    with pytest.raises(RuntimeError, match="different physical GPUs"):
        RoboMMEAdapter("VideoUnmask", env=wrong, points=8, crop=None, render_device="cuda:0")
    assert wrong.closed

