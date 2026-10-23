"""Offline checks for opt-in RunPod SDK result timing."""

import asyncio
import hashlib
import inspect
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from app import runpod_observability as timing
from app import runpod_worker as worker


def _records(capsys, forbidden=()):
    output = capsys.readouterr().out
    assert all(value not in output for value in forbidden)
    lines = output.splitlines()
    return [json.loads(line.removeprefix("COUNTWELL_WORKER_TIMING "))
            for line in lines if line.startswith("COUNTWELL_WORKER_TIMING ")]


def test_disabled_main_uses_unwrapped_handler(monkeypatch):
    handlers = []
    runpod = ModuleType("runpod")
    runpod.serverless = SimpleNamespace(start=lambda config: handlers.append(config["handler"]))
    monkeypatch.setitem(sys.modules, "runpod", runpod)
    monkeypatch.setenv("RUNPOD_PRELOAD", "none")
    monkeypatch.delenv("RUNPOD_TRACE_TIMING", raising=False)

    worker.main()

    assert handlers == [worker.handler]


def test_enabled_main_installs_sdk_wrappers(monkeypatch):
    handlers = []
    runpod = ModuleType("runpod")
    serverless = ModuleType("runpod.serverless")
    modules = ModuleType("runpod.serverless.modules")
    serverless.start = lambda config: handlers.append(config["handler"])
    runpod.serverless = serverless

    async def transmit(*args):
        return None

    async def send_result(*args):
        return None

    modules.rp_http = SimpleNamespace(_transmit=transmit)
    modules.rp_job = SimpleNamespace(send_result=send_result)
    monkeypatch.setitem(sys.modules, "runpod", runpod)
    monkeypatch.setitem(sys.modules, "runpod.serverless", serverless)
    monkeypatch.setitem(sys.modules, "runpod.serverless.modules", modules)
    monkeypatch.setenv("RUNPOD_PRELOAD", "none")
    monkeypatch.setenv("RUNPOD_TRACE_TIMING", "1")

    worker.main()

    assert handlers[0] is not worker.handler
    assert modules.rp_job.send_result._countwell_timing_wrapped
    assert modules.rp_http._transmit._countwell_timing_wrapped


def test_timing_covers_handler_sdk_setup_and_post_without_payloads(capsys):
    secret = "private-image-or-token-in-result"
    attempts = []
    rp_http = SimpleNamespace()

    async def transmit(session, url, job_data):
        attempts.append((url, job_data))
        await asyncio.sleep(0)
        await asyncio.sleep(0)  # Simulates SDK retry/wait inside one _transmit call.
        return "posted"

    rp_http._transmit = transmit

    async def send_result(session, job_data, job):
        serialized = json.dumps(job_data)
        await asyncio.sleep(0)
        return await rp_http._transmit(session, "https://private.example/result?token=" + secret,
                                       serialized)

    rp_job = SimpleNamespace(send_result=send_result)
    result_payload = {"result": {"annotation_data": secret,
                                 "detections": [{"box": [1, 2, 3, 4]}]}}
    observed_handler = timing.install(lambda event: result_payload, rp_job, rp_http)
    assert inspect.iscoroutinefunction(rp_job.send_result)
    assert inspect.iscoroutinefunction(rp_http._transmit)

    async def job():
        event = {"id": "private-job-id", "input": {"image_base64": secret}}
        result = observed_handler(event)
        return result, await rp_job.send_result(None, result, event)

    result, posted = asyncio.run(job())

    assert result == result_payload
    assert posted == "posted"
    assert len(attempts) == 1
    record, = _records(capsys, (secret, "annotation_data", "detections",
                                "private-job-id", "https://"))
    assert record["job_id_hash"] == hashlib.sha256(b"private-job-id").hexdigest()[:16]
    assert record["handler_status"] == "returned"
    assert record["sdk_result_status"] == "returned"
    assert record["post_status"] == "completed"
    assert record["handler_ms"] >= 0
    assert record["handler_to_result_ms"] >= 0
    assert record["sdk_result_pre_post_ms"] >= 0
    assert record["sdk_result_post_ms"] >= 0
    assert record["sdk_result_handoff_ms"] >= record["sdk_result_post_ms"]


def test_post_failure_is_sanitized_even_when_sdk_swallows_it(capsys):
    secret = "secret-error-message"
    rp_http = SimpleNamespace()

    async def transmit(*args):
        raise RuntimeError(secret)

    rp_http._transmit = transmit

    async def send_result(session, job_data, job):
        try:
            await rp_http._transmit(session, "private-url", job_data)
        except RuntimeError:
            return None  # RunPod 1.12.0 catches selected result errors internally.

    rp_job = SimpleNamespace(send_result=send_result)
    observed_handler = timing.install(lambda event: {"result": secret}, rp_job, rp_http)

    async def job():
        event = {"id": "job-1"}
        return await rp_job.send_result(None, observed_handler(event), event)

    assert asyncio.run(job()) is None
    record, = _records(capsys, (secret, "private-url"))
    assert record["sdk_result_status"] == "returned"
    assert record["post_status"] == "raised"
    assert record["post_error_type"] == "RuntimeError"


def test_wrappers_propagate_sdk_exception_and_do_not_double_wrap(capsys):
    error = ValueError("secret-data")
    rp_http = SimpleNamespace(_transmit=None)

    async def transmit(*args):
        return None

    rp_http._transmit = transmit

    async def send_result(*args):
        await asyncio.sleep(0)
        raise error

    rp_job = SimpleNamespace(send_result=send_result)
    observed_handler = timing.install(lambda event: "ok", rp_job, rp_http)
    wrapped_send = rp_job.send_result
    wrapped_post = rp_http._transmit
    timing.install(lambda event: "ok", rp_job, rp_http)
    assert rp_job.send_result is wrapped_send
    assert rp_http._transmit is wrapped_post

    async def job():
        event = {"id": "job-2"}
        observed_handler(event)
        await rp_job.send_result(None, {}, event)

    with pytest.raises(ValueError) as raised:
        asyncio.run(job())
    assert raised.value is error
    record, = _records(capsys, ("secret-data",))
    assert record["sdk_result_status"] == "raised"
    assert record["sdk_result_error_type"] == "ValueError"
    assert record["post_status"] == "not_started"
