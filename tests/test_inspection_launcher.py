"""CPU-only launcher lifecycle checks; no scientific inference is performed."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch

from stmem.inspection import pipeline, report
from stmem.training.checkpoints import atomic_json


def configured_launcher(tmp_path, monkeypatch, *, stages=None):
    output = tmp_path / "analysis"

    def prepare(_bundle, actual_output, _assets, **_kwargs):
        assert actual_output == output
        manifest = {"output": str(output)}
        path = output / "inspection_manifest.json"
        atomic_json(manifest, path)
        return manifest, path

    monkeypatch.setattr(pipeline, "prepare_manifest", prepare)
    monkeypatch.setattr(pipeline, "verify_final_inputs", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pipeline, "run_stage", lambda *_args, **_kwargs: None)
    argv = ["inspect_checkpoints", "--bundle", str(tmp_path / "bundle"), "--output", str(output)]
    if stages:
        argv += ["--stages", *stages]
    monkeypatch.setattr(sys, "argv", argv)
    return output


@pytest.mark.parametrize("status,stages", [("complete", None), ("partial", ["offline"])])
def test_successful_report_finishes_pipeline_status(tmp_path, monkeypatch, status, stages):
    output = configured_launcher(tmp_path, monkeypatch, stages=stages)

    def build(manifest, **_kwargs):
        assert manifest["output"] == str(output)
        atomic_json({"status": status}, output / "summary.json")
        return output / "index.html"

    monkeypatch.setattr(report, "build_report", build)
    pipeline.main()
    result = json.loads((output / "pipeline_status.json").read_text())
    assert result["status"] == status
    assert result["finished_utc"] and result["report"] == str(output / "index.html")


def test_failed_report_marks_pipeline_failed_and_restores_handlers(tmp_path, monkeypatch):
    output = configured_launcher(tmp_path, monkeypatch)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}

    def fail(*_args, **_kwargs):
        raise ValueError("mismatched validation identity")

    monkeypatch.setattr(report, "build_report", fail)
    with pytest.raises(ValueError, match="mismatched validation identity"):
        pipeline.main()
    result = json.loads((output / "pipeline_status.json").read_text())
    assert result["status"] == "failed" and "mismatched validation identity" in result["error"]
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers
    with pipeline.inspection_lock(output):
        pass


def test_report_only_revalidates_inputs_without_requesting_cuda(tmp_path, monkeypatch):
    output = configured_launcher(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--report-only"])

    def fail(*_args):
        raise ValueError("Inspection assets changed during inspection")

    monkeypatch.setattr(pipeline, "verify_final_inputs", fail)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("report-only requested CUDA"))
    monkeypatch.setattr(report, "build_report", lambda *_args, **_kwargs: pytest.fail("used changed inputs"))
    with pytest.raises(ValueError, match="assets changed"):
        pipeline.main()
    assert json.loads((output / "pipeline_status.json").read_text())["status"] == "failed"


def test_subprocess_start_failure_does_not_leave_stage_running(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("could not start child")

    monkeypatch.setattr(pipeline.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="could not start child"):
        pipeline.run_stage("offline", tmp_path / "inspection_manifest.json", resume=False)
    result = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert result["status"] == result["stages"]["offline"]["status"] == "failed"


def test_runtime_sidecar_is_immutable_and_rejects_changed_environment(tmp_path, monkeypatch):
    manifest = tmp_path / "inspection_manifest.json"
    atomic_json({"immutable": "diagnostic protocol"}, manifest)
    identity = {"python": "3.11", "packages": {"torch": "2.9"}, "initialized_cuda_device": None}
    monkeypatch.setattr(pipeline, "inspect_runtime", lambda: identity)
    record = pipeline.verify_runtime(manifest)
    path = tmp_path / "runtime_identity.json"
    original = path.read_bytes()
    assert pipeline.verify_runtime(manifest) == record and path.read_bytes() == original
    monkeypatch.setattr(pipeline, "inspect_runtime", lambda: {**identity, "packages": {"torch": "3.0"}})
    with pytest.raises(ValueError, match="runtime environment changed"):
        pipeline.verify_runtime(manifest)
    assert path.read_bytes() == original
    monkeypatch.setattr(pipeline, "inspect_runtime", lambda: identity)
    atomic_json({"immutable": "different protocol"}, manifest)
    with pytest.raises(ValueError, match="different inspection manifest"):
        pipeline.verify_runtime(manifest)


def test_observing_runtime_does_not_initialize_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    def forbidden(*_args, **_kwargs):
        pytest.fail("CPU runtime observation attempted to initialize CUDA")

    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    identity = pipeline.inspect_runtime()
    assert identity["initialized_cuda_device"] is None
    assert set(identity["packages"]) == {"torch", "robomme", "mani-skill", "sapien", "numpy"}
    assert identity["python"] and identity["torch_build"]


@pytest.mark.parametrize("change", ["asset", "path", "source"])
def test_final_identity_check_rejects_midrun_changes(tmp_path, monkeypatch, change):
    manifest = {
        "policy_source_sha256": "policy", "assets": {"prepared": "/original/data"},
        "asset_hashes": {"prepared": "original bytes"},
    }
    monkeypatch.setattr(pipeline, "policy_source_hash", lambda: "policy")
    monkeypatch.setattr(pipeline, "resolve_assets", lambda _config: (manifest["assets"], manifest["asset_hashes"]))
    pipeline.verify_final_inputs(manifest, tmp_path / "assets.yaml")
    if change == "source":
        monkeypatch.setattr(pipeline, "policy_source_hash", lambda: "changed policy")
    else:
        paths = {"prepared": "/other/data"} if change == "path" else manifest["assets"]
        hashes = {"prepared": "changed bytes"} if change == "asset" else manifest["asset_hashes"]
        monkeypatch.setattr(pipeline, "resolve_assets", lambda _config: (paths, hashes))
    with pytest.raises(ValueError, match="changed during inspection"):
        pipeline.verify_final_inputs(manifest, tmp_path / "assets.yaml")


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_termination_cleans_child_group_before_releasing_output(tmp_path, signum):
    """Kill a real CPU launcher; its independent child session must also exit."""
    child = tmp_path / "cpu_child.py"
    child.write_text(
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "grandchild = subprocess.Popen([sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)'])\n"
        f"Path({str(tmp_path / 'pids.json')!r}).write_text(json.dumps([os.getpid(), grandchild.pid]))\n"
        "time.sleep(120)\n"
    )
    driver = tmp_path / "cpu_launcher.py"
    driver.write_text(
        "import sys, torch\n"
        "from pathlib import Path\n"
        "from stmem.inspection import pipeline\n"
        "from stmem.training.checkpoints import atomic_json\n"
        "def prepare(bundle, output, assets, **kwargs):\n"
        "    manifest = {'output': str(output)}\n"
        "    path = output / 'inspection_manifest.json'\n"
        "    atomic_json(manifest, path)\n"
        "    return manifest, path\n"
        "pipeline.prepare_manifest = prepare\n"
        "pipeline.inspect_runtime = lambda: {'fixture': 'CPU subprocess lifecycle', 'initialized_cuda_device': None}\n"
        "torch.cuda.is_available = lambda: True\n"
        "real_popen = pipeline.subprocess.Popen\n"
        f"pipeline.subprocess.Popen = lambda command, **kw: real_popen([sys.executable, {str(child)!r}], **kw)\n"
        f"sys.argv = ['inspect', '--bundle', {str(tmp_path / 'bundle')!r}, '--output', {str(tmp_path / 'output')!r}, '--stages', 'offline']\n"
        "pipeline.main()\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    process = subprocess.Popen([sys.executable, str(driver)], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=env, text=True)
    pids = []
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "pids.json").exists():
            if process.poll() is not None:
                pytest.fail(f"CPU fixture exited early: {process.communicate()[0]}")
            if time.monotonic() >= deadline:
                pytest.fail("CPU fixture did not start in 30 seconds")
            time.sleep(0.05)
        pids = json.loads((tmp_path / "pids.json").read_text())
        process.send_signal(signum)
        logs, _ = process.communicate(timeout=35)
        assert process.returncode == 128 + signum, logs
        state = json.loads((tmp_path / "output/pipeline_status.json").read_text())
        assert state["status"] == state["stages"]["offline"]["status"] == "interrupted"
        assert signal.Signals(signum).name in state["error"]
        for pid in pids:
            # An orphan grandchild may briefly be a zombie until init reaps it;
            # it has exited and cannot write or consume GPU resources.
            stat = Path(f"/proc/{pid}/stat")
            assert not stat.exists() or stat.read_text().split(") ", 1)[1].startswith("Z ")
        with pipeline.inspection_lock(tmp_path / "output"):
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if pids:
            try:
                os.killpg(pids[0], signal.SIGKILL)
            except ProcessLookupError:
                pass
