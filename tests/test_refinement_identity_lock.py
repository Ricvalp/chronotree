"""Real CPU process contention at shared utility initialization boundaries."""

import json
import multiprocessing

from stmem.inspection.refinement_utility import lock, publish_measurement_identity


def _contender(output, identity, operation, ready, results):
    ready.set()
    try:
        if operation == "identity":
            publish_measurement_identity(output, identity)
        else:
            with lock(output / "worker_0.lock"):
                pass
        results.put({"status": "ok"})
    except Exception as error:
        results.put({"status": "error", "type": type(error).__name__, "message": str(error)})


def _stop(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
        process.join(timeout=10)


def test_distinct_shards_wait_and_publish_the_same_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    context = multiprocessing.get_context("spawn")
    identity = {"student": "S_STD", "step": 95000, "workers": 4, "partition": "train"}
    results = context.Queue()
    processes = []
    ready = [context.Event(), context.Event()]
    try:
        # Hold the actual shared lock until both independent processes enter
        # initialization; neither may fail or publish while another owns it.
        with lock(tmp_path / "identity.lock"):
            for event in ready:
                process = context.Process(target=_contender,
                                          args=(tmp_path, identity, "identity", event, results))
                process.start()
                processes.append(process)
            assert all(event.wait(timeout=30) for event in ready)
            for process in processes:
                process.join(timeout=.2)
                assert process.is_alive()
            assert not (tmp_path / "identity.json").exists()
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
        assert [results.get(timeout=5) for _ in processes] == [{"status": "ok"}] * 2
        assert json.loads((tmp_path / "identity.json").read_text()) == identity
    finally:
        _stop(processes)
        results.close()
        results.join_thread()


def test_contending_changed_identity_is_rejected_after_waiting(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    context = multiprocessing.get_context("spawn")
    original = {"student": "S_STD", "step": 95000, "workers": 4, "partition": "train"}
    publish_measurement_identity(tmp_path, original)
    before = (tmp_path / "identity.json").read_bytes()
    ready, results = context.Event(), context.Queue()
    process = context.Process(target=_contender,
                              args=(tmp_path, {**original, "workers": 1}, "identity", ready, results))
    try:
        with lock(tmp_path / "identity.lock"):
            process.start()
            assert ready.wait(timeout=30)
            process.join(timeout=.2)
            assert process.is_alive()
        process.join(timeout=20)
        assert process.exitcode == 0
        outcome = results.get(timeout=5)
        assert outcome["status"] == "error" and outcome["type"] == "ValueError"
        assert "Measurement identity changed" in outcome["message"]
        assert (tmp_path / "identity.json").read_bytes() == before
    finally:
        _stop([process])
        results.close()
        results.join_thread()


def test_duplicate_worker_still_fails_without_waiting(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    context = multiprocessing.get_context("spawn")
    ready, results = context.Event(), context.Queue()
    process = context.Process(target=_contender, args=(tmp_path, {}, "worker", ready, results))
    try:
        with lock(tmp_path / "worker_0.lock"):
            process.start()
            assert ready.wait(timeout=30)
            process.join(timeout=20)
            assert process.exitcode == 0
            outcome = results.get(timeout=5)
            assert outcome["status"] == "error" and outcome["type"] == "BlockingIOError"
    finally:
        _stop([process])
        results.close()
        results.join_thread()
