"""Compute status semantics without network access or GPUs."""

import json
import threading
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app import compute_status
from app.main import create_app


class Clock:
    def __init__(self):
        self.now_value = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
        self.seconds = 100.0

    def now(self):
        return self.now_value

    def monotonic(self):
        return self.seconds

    def advance(self, seconds):
        self.now_value += timedelta(seconds=seconds)
        self.seconds += seconds


class Health:
    def __init__(self, running=0, initializing=0, in_progress=0, in_queue=0):
        self.running = running
        self.initializing = initializing
        self.in_progress = in_progress
        self.in_queue = in_queue
        self.calls = 0
        self.error = None

    def health(self):
        self.calls += 1
        if self.error:
            raise self.error
        return {"workers": {"running": self.running, "initializing": self.initializing,
                            "ready": 1, "idle": 1},
                "jobs": {"inProgress": self.in_progress, "inQueue": self.in_queue}}

    def close(self):
        pass


def monitor(tmp_path, clock, health, endpoint="endpoint-1"):
    return compute_status.ComputeStatus(
        tmp_path, endpoint, client_factory=lambda: health,
        now=clock.now, monotonic=clock.monotonic)


def test_deadline_tracks_terminal_completion_and_provider_running_is_authoritative(tmp_path):
    clock, health = Clock(), Health(running=1)
    status = monitor(tmp_path, clock, health)
    assert status.snapshot() == {
        "enabled": True, "state": "ready", "server_time": "2026-10-23T12:00:00Z",
        "idle_deadline": None, "idle_timeout_seconds": 600}
    status.start()
    assert status.snapshot()["state"] == "busy"
    status.finish(terminal=True)
    expected = "2026-10-23T12:10:00Z"
    assert status.snapshot()["idle_deadline"] == expected
    clock.advance(600)
    expired = status.snapshot()
    assert expired["state"] == "ready"
    assert expired["idle_deadline"] == expected
    assert expired["server_time"] == "2026-10-23T12:10:00Z"
    health.running = 0
    clock.advance(10)
    sleeping = status.snapshot()
    assert sleeping["state"] == "sleeping"
    assert sleeping["idle_deadline"] is None


def test_reads_are_nonwarming_cached_across_threads_and_do_not_reset_deadline(tmp_path):
    clock, health = Clock(), Health(running=1)
    status = monitor(tmp_path, clock, health)
    status.start()
    status.finish(terminal=True)
    before = (tmp_path / "runpod-activity.json").read_bytes()
    results = []
    threads = [threading.Thread(target=lambda: results.append(status.snapshot())) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert health.calls == 1
    assert all(result["idle_deadline"] == "2026-10-23T12:10:00Z" for result in results)
    clock.advance(9)
    assert status.snapshot()["server_time"] == "2026-10-23T12:00:09Z"
    assert health.calls == 1
    assert (tmp_path / "runpod-activity.json").read_bytes() == before
    clock.advance(1)
    status.snapshot()
    assert health.calls == 2


def test_restart_recovers_only_matching_endpoint_completion(tmp_path):
    clock, health = Clock(), Health(running=1)
    first = monitor(tmp_path, clock, health)
    first.start()
    first.finish(terminal=True)
    assert json.loads((tmp_path / "runpod-activity.json").read_text()) == {
        "endpoint_id": "endpoint-1", "completed_at": "2026-10-23T12:00:00Z"}
    clock.advance(90)
    assert monitor(tmp_path, clock, Health(running=1)).snapshot()["idle_deadline"] == (
        "2026-10-23T12:10:00Z")
    assert monitor(tmp_path, clock, Health(running=1), "endpoint-2").snapshot()["idle_deadline"] is None


def test_states_ignore_idle_ready_counts_and_respect_active_or_provider_queue(tmp_path):
    clock, health = Clock(), Health()
    status = monitor(tmp_path, clock, health)
    assert status.snapshot()["state"] == "sleeping"
    status.start()
    assert status.snapshot()["state"] == "starting"
    status.finish()
    clock.advance(10)
    health.initializing = 1
    assert status.snapshot()["state"] == "starting"
    clock.advance(10)
    health.initializing, health.in_queue = 0, 1
    assert status.snapshot()["state"] == "starting"
    clock.advance(10)
    health.in_queue, health.in_progress = 0, 1
    assert status.snapshot()["state"] == "busy"
    clock.advance(10)
    health.in_progress = 0
    assert status.snapshot()["state"] == "sleeping"


def test_health_failure_is_unavailable_and_does_not_disclose_error(tmp_path):
    clock, health = Clock(), Health()
    health.error = RuntimeError("secret-provider-message")
    status = monitor(tmp_path, clock, health)
    payload = status.snapshot()
    assert payload["state"] == "unavailable"
    assert "secret-provider-message" not in repr(payload)


def test_local_route_disabled_and_remote_route_contains_no_provider_details(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "local")
    with TestClient(create_app(tmp_path / "local")) as client:
        response = client.get("/api/compute-status")
        assert response.json() == {"enabled": False}
        assert response.headers["Cache-Control"] == "no-store"
    monkeypatch.setenv("INFERENCE_BACKEND", "runpod")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "endpoint-1")
    monkeypatch.setenv("RUNPOD_API_KEY", "secret-key")
    health = Health()
    monkeypatch.setattr(compute_status, "RunPodClient", lambda: health)
    with TestClient(create_app(tmp_path / "remote")) as client:
        response = client.get("/api/compute-status")
        payload = response.json()
        assert response.headers["Cache-Control"] == "no-store"
    assert set(payload) == {"enabled", "state", "server_time", "idle_deadline",
                            "idle_timeout_seconds"}
    assert payload["state"] == "sleeping"
    assert "secret-key" not in repr(payload)
    assert "endpoint-1" not in repr(payload)


def test_status_poll_does_not_wait_for_inference_client_lock(tmp_path):
    from app.runpod_client import RunPodClient

    clock, health = Clock(), Health(running=1)
    status = monitor(tmp_path, clock, health)
    submitted = threading.Event()
    release = threading.Event()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, limit):
            return b'{"id":"job-1","status":"COMPLETED","output":{}}'

    class BlockingOpener:
        def open(self, request, timeout):
            assert request.get_method() == "POST"
            submitted.set()
            assert release.wait(3)
            return Response()

    inference = RunPodClient("key", "endpoint-1", opener=BlockingOpener(), activity=status)
    worker = threading.Thread(target=lambda: inference.run({"action": "count"}))
    worker.start()
    assert submitted.wait(3)
    polled = threading.Event()
    result = {}

    def poll():
        result.update(status.snapshot())
        polled.set()

    reader = threading.Thread(target=poll)
    reader.start()
    try:
        assert polled.wait(.5), "status read blocked behind the in-flight inference request"
        assert result["state"] == "busy"
    finally:
        release.set()
        worker.join(3)
        reader.join(3)



def test_new_job_invalidates_persisted_completion_even_if_outcome_is_unknown(tmp_path):
    clock, health = Clock(), Health(running=1)
    first = monitor(tmp_path, clock, health)
    first.start()
    first.finish(terminal=True)
    assert first.snapshot()["idle_deadline"] == "2026-10-23T12:10:00Z"
    first.start()
    assert json.loads((tmp_path / "runpod-activity.json").read_text())["completed_at"] is None
    first.finish()
    assert first.snapshot()["idle_deadline"] is None
    assert monitor(tmp_path, clock, Health(running=1)).snapshot()["idle_deadline"] is None


def test_restored_completion_is_discarded_when_untracked_provider_work_is_seen(tmp_path):
    clock = Clock()
    first = monitor(tmp_path, clock, Health(running=1))
    first.start()
    first.finish(terminal=True)
    provider = Health(in_progress=1)
    restarted = monitor(tmp_path, clock, provider)
    assert restarted.snapshot()["state"] == "busy"
    assert json.loads((tmp_path / "runpod-activity.json").read_text())["completed_at"] is None
    clock.advance(10)
    provider.in_progress, provider.running = 0, 1
    ready = restarted.snapshot()
    assert ready["state"] == "ready"
    assert ready["idle_deadline"] is None
