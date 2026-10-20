"""CPU diagnostic: accumulated DDP gradients retain earlier-only encoder use."""

import os
import subprocess
import sys
from textwrap import dedent


def test_ddp_accumulation_reduces_parameters_unused_in_final_microbatch(tmp_path):
    script = tmp_path / "ddp_unused.py"
    script.write_text(
        dedent(
            """
            from contextlib import nullcontext
            from copy import deepcopy
            import torch
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel
            from stmem.config import load_config
            from stmem.data.dataset import synthetic_episode
            from stmem.models.policy import MemoryPolicy
            from stmem.models.action_flow import masked_mse

            torch.set_num_threads(1)
            dist.init_process_group("gloo")
            rank = dist.get_rank()
            torch.manual_seed(123)
            cfg = load_config("configs/debug/teacher.yaml")
            cfg["model"].update(
                width=16, heads=4, layers=1, text_layers=1, max_text_bytes=12,
                block_size=2, current_tokens=2, visual_tokens=4, horizon=2,
                points=4, checkpoint_blocks=True,
                encoder_options={"neighbor_cap": 4, "center_chunk": 2},
            )
            cfg["training"]["encoder_block_batch_size"] = 2
            policy = MemoryPolicy(cfg)
            reference = deepcopy(policy)
            model = DistributedDataParallel(policy, find_unused_parameters=True)

            def example(worker, index):
                if index == 0:
                    decision = synthetic_episode(6, 2, 4, seed=100 + worker).decision(3, 2)
                else:
                    # The final synchronized forward uses neither historical
                    # events nor occupied visual tokens. Earlier accumulated
                    # encoder gradients must still be reduced across ranks.
                    decision = synthetic_episode(4, 0, 0, seed=200 + worker).decision(0, 2)
                generator = torch.Generator().manual_seed(300 + worker * 2 + index)
                noisy = torch.randn(1, 2, 8, generator=generator)
                target = torch.randn(1, 2, 8, generator=generator)
                return decision, noisy, target

            for index in range(2):
                decision, noisy, target = example(rank, index)
                with model.no_sync() if index == 0 else nullcontext():
                    prediction = model([decision], noisy, torch.tensor([0.4]))
                    loss = masked_mse(prediction, target, torch.ones_like(target, dtype=torch.bool))
                    (loss / 2).backward()

            for worker in range(2):
                for index in range(2):
                    decision, noisy, target = example(worker, index)
                    prediction = reference([decision], noisy, torch.tensor([0.4]))
                    loss = masked_mse(prediction, target, torch.ones_like(target, dtype=torch.bool))
                    (loss / 4).backward()

            for (name, actual), (_, expected) in zip(
                policy.named_parameters(), reference.named_parameters(), strict=True
            ):
                assert (actual.grad is None) == (expected.grad is None), name
                if actual.grad is not None:
                    torch.testing.assert_close(
                        actual.grad, expected.grad, rtol=1e-4, atol=1e-6,
                        msg=f"rank {rank}: {name}",
                    )
            print(f"rank {rank}: accumulated gradient parity passed", flush=True)
            dist.destroy_process_group()
            """
        )
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            str(script),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2):
        assert f"rank {rank}: accumulated gradient parity passed" in result.stdout
