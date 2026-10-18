"""Run a small real-device Torch forward/backward and precision diagnostic."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import platform
import subprocess
import sys
import time
from typing import Any

import torch
from torch.nn import functional as F


def _driver() -> dict[str, Any]:
    try:
        run = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        return {"returncode": run.returncode, "gpu_query": run.stdout.strip(), "stderr": run.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def precision_test(device: torch.device | str, dtype: torch.dtype = torch.float32) -> dict[str, Any]:
    """Actually execute SDPA, matrix multiplication, and finite gradients."""
    device = torch.device(device)
    start = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(0)
            q = torch.randn(2, 4, 16, 32, device=device, requires_grad=True)
            k = torch.randn(2, 4, 24, 32, device=device, requires_grad=True)
            v = torch.randn(2, 4, 24, 32, device=device, requires_grad=True)
            w = torch.randn(32, 32, device=device, requires_grad=True)
            context = torch.autocast(device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()
            with context:
                output = F.scaled_dot_product_attention(q, k, v) @ w
                loss = output.float().square().mean()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            finite = bool(
                torch.isfinite(loss).all()
                and all(item.grad is not None and torch.isfinite(item.grad).all() for item in (q, k, v, w))
            )
            if not finite:
                raise RuntimeError("Nonfinite output or gradient")
            return {
                "ok": True,
                "dtype": str(dtype),
                "output_dtype": str(output.dtype),
                "loss": float(loss.detach()),
                "seconds": time.perf_counter() - start,
            }
    except Exception as error:
        return {"ok": False, "dtype": str(dtype), "error": f"{type(error).__name__}: {error}"}


def choose_precision(device: torch.device | str) -> str:
    device = torch.device(device)
    if device.type != "cuda":
        return "fp32"
    if precision_test(device, torch.bfloat16)["ok"]:
        return "bf16"
    if precision_test(device, torch.float16)["ok"]:
        return "fp16"
    if precision_test(device, torch.float32)["ok"]:
        return "fp32"
    raise RuntimeError("No usable precision on this CUDA device; run python -m stmem.doctor")


def diagnose(device: str | None = None) -> dict[str, Any]:
    selected = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "wheel_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "driver": _driver(),
        "device": str(selected),
        "supported_architectures": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "rendering": {
            "status": "not tested",
            "reason": "CUDA compute does not establish Vulkan/RGB-D support",
        },
    }
    if selected.type == "cuda":
        properties = torch.cuda.get_device_properties(selected)
        capability = torch.cuda.get_device_capability(selected)
        report.update(
            gpu_name=properties.name,
            capability=list(capability),
            total_memory_bytes=properties.total_memory,
            native_architecture_present=f"sm_{capability[0]}{capability[1]}"
            in report["supported_architectures"],
        )
    report["tests"] = {
        name: precision_test(selected, dtype)
        for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16))
    }
    report["recommended_precision"] = next(
        (
            name
            for name in (("bf16", "fp16", "fp32") if selected.type == "cuda" else ("fp32",))
            if report["tests"][name]["ok"]
        ),
        None,
    )
    report["fp16_requires_grad_scaler"] = True
    report["ok"] = report["tests"]["fp32"]["ok"] and report["recommended_precision"] is not None
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = diagnose(args.device)
    if args.output:
        from stmem.training.checkpoints import atomic_json

        atomic_json(report, args.output)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
