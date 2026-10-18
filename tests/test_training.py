from copy import deepcopy
import os
import subprocess
import sys
import torch
import yaml

from stmem.config import load_config
from stmem.data.dataset import synthetic_episode
from stmem.models.policy import MemoryPolicy
from stmem.models.action_flow import ActionNormalizer, masked_mse, flow_sample


def tiny_config():
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["model"].update(
        width=32,
        heads=4,
        layers=1,
        current_tokens=4,
        visual_tokens=4,
        horizon=4,
        points=16,
        checkpoint_blocks=False,
    )
    cfg["training"].update(
        steps=2, effective_batch=2, checkpoint_every=1, checkpoint_steps=[2], precision="fp32"
    )
    cfg["data"].update(synthetic_execution_frames=5, synthetic_conditioning_frames=2)
    return cfg


def test_normalization_masks_and_fixed_tiny_loss_reduction():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    actions = torch.randn(6, 8)
    valid = torch.ones_like(actions, dtype=torch.bool)
    valid[:, 7] = False
    normalizer = ActionNormalizer()
    normalizer.fit(actions, valid)
    torch.testing.assert_close(normalizer.denormalize(normalizer.normalize(actions)), actions)
    x, u, target, noise = flow_sample(actions, valid, normalizer, time=torch.tensor(0.3))
    assert not x[:, 7].any() and not target[:, 7].any()
    pred = torch.zeros_like(target, requires_grad=True)
    masked_mse(pred, target, valid).backward()
    assert not pred.grad[:, 7].any()
    cfg = tiny_config()
    model = MemoryPolicy(cfg)
    episode = synthetic_episode(6, 2, 16)
    decision = episode.decision(3, 4)
    model.normalizer.fit(episode.actions, episode.action_valid)
    noisy, t, target, _ = flow_sample(
        decision.target,
        decision.target_valid,
        model.normalizer,
        noise=torch.zeros_like(decision.target),
        time=torch.tensor(0.5),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        prediction = model(decision, noisy, t)
        loss = masked_mse(prediction, target, decision.target_valid)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.25


def test_training_safe_stop_resume_matches_uninterrupted(tmp_path):
    cfg = tiny_config()
    complete = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    configs = []
    for root in (complete, resumed):
        copy = deepcopy(cfg)
        copy["paths"]["run_root"] = str(root)
        path = tmp_path / f"{root.name}.yaml"
        path.write_text(yaml.safe_dump(copy))
        configs.append(path)

    def call(path, *args):
        return subprocess.run(
            [sys.executable, "-m", "stmem.train", "--config", str(path), *map(str, args)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    a = call(configs[0])
    assert a.returncode == 0, a.stderr
    marker = tmp_path / "stop"
    marker.touch()
    b = call(configs[1], "--stop-file", marker)
    assert b.returncode == 75, b.stderr
    assert not (resumed / "seed0/teacher/complete.json").exists()
    marker.unlink()
    checkpoint = resumed / "seed0/teacher/last.pt"
    c = call(configs[1], "--resume", checkpoint, "--stop-file", marker)
    assert c.returncode == 0, c.stderr
    before = torch.load(complete / "seed0/teacher/last.pt", weights_only=False, map_location="cpu")
    after = torch.load(checkpoint, weights_only=False, map_location="cpu")
    assert after["step"] == before["step"] == 2
    for key in before["model"]:
        torch.testing.assert_close(before["model"][key], after["model"][key], rtol=0, atol=0)
    assert checkpoint.with_name("last.pt.previous").exists()
