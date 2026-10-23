"""Queue protocol and remote result validation without external requests."""

import base64
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError

from PIL import Image
import pytest

from app import runpod_client as remote


class _Response:
    def __init__(self, data):
        self.data = json.dumps(data).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return self.data[:limit]


class _Opener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def open(self, req, timeout):
        self.calls.append((req, timeout))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return _Response(response)


def _client(responses, **kwargs):
    opener = _Opener(responses)
    return remote.RunPodClient("top-secret-key", "endpoint-1", opener=opener, **kwargs), opener


def _image(path):
    Image.new("RGB", (40, 30), "white").save(path)
    return str(path)


def _count_output(*, box=None):
    return {"result": {"count": 1, "detections": [
        {"index": 1, "box": box or [1, 2, 11, 12], "confidence": .85}],
        "width": 40, "height": 30, "elapsed_ms": 12.5,
        "warning": "Experimental model; review overlapping objects."},
        "metrics": {"gpu_name": "NVIDIA A100", "worker_id": "pod-1", "cold_start": True,
                    "handler_ms": 17, "worker_ready_ms": 110, "preload_ms": 0,
                    "startup_model_load_ms": 92,
                    "peak_cuda_allocated_bytes": 1234, "model": "countgd-plusplus",
                    "model_sha256": "a" * 64, "build_revision": "b" * 40,
                    "preloaded_model": "count", "error": "top-secret-key"}}


def test_async_submit_poll_and_sanitized_timings(monkeypatch):
    client, opener = _client([
        {"id": "job-1", "status": "IN_QUEUE"},
        {"id": "job-1", "status": "IN_PROGRESS"},
        {"id": "job-1", "status": "COMPLETED", "output": _count_output(),
         "delayTime": 30, "executionTime": 120},
    ])
    monkeypatch.setattr(remote.time, "sleep", lambda seconds: None)
    status = client.run({"action": "count", "image_base64": "abc"})
    assert status["status"] == "COMPLETED"
    assert [req.get_method() for req, _ in opener.calls] == ["POST", "GET", "GET"]
    assert [req.full_url.split("/v2/endpoint-1/")[1] for req, _ in opener.calls] == [
        "run", "status/job-1", "status/job-1"]
    assert json.loads(opener.calls[0][0].data) == {"input": {"action": "count", "image_base64": "abc"}}
    assert opener.calls[0][0].get_header("Authorization") == "Bearer top-secret-key"
    assert client.last_job["job_id"] == "job-1"
    assert client.last_job["delayTime"] == 30
    assert client.last_job["executionTime"] == 120
    assert client.last_job["worker_metrics"]["cold_start"] is True
    assert client.last_job["worker_metrics"]["model"] == "countgd-plusplus"
    assert client.last_job["worker_metrics"]["model_sha256"] == "a" * 64
    assert client.last_job["worker_metrics"]["build_revision"] == "b" * 40
    assert client.last_job["worker_metrics"]["preloaded_model"] == "count"
    assert client.last_job["worker_metrics"]["startup_model_load_ms"] == 92
    assert "error" not in client.last_job["worker_metrics"]
    assert "top-secret-key" not in repr(client.last_job)


def test_timeout_requests_cancel_without_resubmission(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(remote.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(remote.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds + .01))
    client, opener = _client([
        {"id": "job-2", "status": "IN_QUEUE"},
        {"id": "job-2", "status": "IN_PROGRESS"},
        {"id": "job-2", "status": "CANCELLED"},
    ], timeout_seconds=.01)
    with pytest.raises(remote.RunPodError, match="timed out"):
        client.run({"action": "suggest", "image_base64": "abc"})
    assert [req.get_method() for req, _ in opener.calls] == ["POST", "GET", "POST"]
    assert opener.calls[-1][0].full_url.endswith("/cancel/job-2")


def test_transient_status_errors_retry_within_budget_without_resubmitting(monkeypatch):
    transient = HTTPError("https://api.runpod.ai/v2/endpoint-1/status/job-6", 503,
                          "top-secret-key provider detail", {}, BytesIO(b"secret image content"))
    client, opener = _client([
        {"id": "job-6", "status": "IN_QUEUE"}, transient,
        {"id": "job-6", "status": "COMPLETED", "output": _count_output()},
    ])
    monkeypatch.setattr(remote.time, "sleep", lambda seconds: None)
    assert client.run({"action": "count"})["status"] == "COMPLETED"
    assert [req.get_method() for req, _ in opener.calls] == ["POST", "GET", "GET"]


def test_transient_status_retries_are_bounded(monkeypatch):
    failures = [HTTPError("https://api.runpod.ai/v2/endpoint-1/status/job-7", 503,
                          "top-secret-key provider detail", {}, BytesIO(b"secret"))
                for _ in range(4)]
    client, opener = _client([{"id": "job-7", "status": "IN_QUEUE"}, *failures,
                              {"id": "job-7", "status": "CANCELLED"}])
    monkeypatch.setattr(remote.time, "sleep", lambda seconds: None)
    with pytest.raises(remote.RunPodError, match="503") as exc:
        client.run({"action": "count"})
    assert "secret" not in str(exc.value)
    assert [req.get_method() for req, _ in opener.calls] == ["POST", "GET", "GET", "GET", "GET", "POST"]
    assert opener.calls[-1][0].full_url.endswith("/cancel/job-7")


def test_nontransient_status_error_is_not_retried():
    failure = HTTPError("https://api.runpod.ai/v2/endpoint-1/status/job-8", 401,
                        "top-secret-key provider detail", {}, BytesIO(b"secret"))
    client, opener = _client([{"id": "job-8", "status": "IN_QUEUE"}, failure,
                              {"id": "job-8", "status": "CANCELLED"}])
    with pytest.raises(remote.RunPodError, match="401"):
        client.run({"action": "count"})
    assert [req.get_method() for req, _ in opener.calls] == ["POST", "GET", "POST"]


@pytest.mark.parametrize("bad_status", [
    {"id": "job-9", "status": "UNKNOWN"},
    {"id": "other-job", "status": "IN_PROGRESS"},
])
def test_invalid_status_cancels_job(bad_status):
    client, opener = _client([{"id": "job-9", "status": "IN_QUEUE"}, bad_status,
                              {"id": "job-9", "status": "CANCELLED"}])
    with pytest.raises(remote.RunPodError):
        client.run({"action": "count"})
    assert [req.get_method() for req, _ in opener.calls] == ["POST", "GET", "POST"]
    assert opener.calls[-1][0].full_url.endswith("/cancel/job-9")


def test_interrupt_cancels_job_and_preserves_interrupt():
    client, opener = _client([{"id": "job-10", "status": "IN_QUEUE"}, KeyboardInterrupt(),
                              {"id": "job-10", "status": "CANCELLED"}])
    with pytest.raises(KeyboardInterrupt):
        client.run({"action": "count"})
    assert [req.get_method() for req, _ in opener.calls] == ["POST", "GET", "POST"]


def test_failed_cancel_does_not_mask_polling_failure():
    status_error = HTTPError("https://api.runpod.ai/v2/endpoint-1/status/job-11", 401,
                             "provider secret", {}, BytesIO(b"secret"))
    cancel_error = HTTPError("https://api.runpod.ai/v2/endpoint-1/cancel/job-11", 503,
                             "different provider secret", {}, BytesIO(b"secret"))
    client, opener = _client([{"id": "job-11", "status": "IN_QUEUE"}, status_error,
                              cancel_error])
    with pytest.raises(remote.RunPodError, match="401") as exc:
        client.run({"action": "count"})
    assert "secret" not in str(exc.value)
    assert [req.get_method() for req, _ in opener.calls] == ["POST", "GET", "POST"]


def test_identity_metrics_reject_arbitrary_strings():
    metrics = _count_output()["metrics"]
    metrics.update(gpu_name="NVIDIA top-secret-key", worker_id="top-secret-key",
                   build_revision="top-secret-key", model="top-secret-key",
                   model_sha256="top-secret-key", preloaded_model="top-secret-key",
                   startup_model_load_ms=float("nan"))
    assert remote._safe_metrics(metrics, api_key="top-secret-key") == {
        "handler_ms": 17, "worker_ready_ms": 110, "preload_ms": 0,
        "peak_cuda_allocated_bytes": 1234, "cold_start": True,
    }


def test_provider_error_body_and_credential_never_surface():
    bad = HTTPError("https://api.runpod.ai/v2/endpoint-1/run", 403,
                    "top-secret-key malicious provider detail", {}, BytesIO(b"secret image content"))
    client, opener = _client([bad])
    with pytest.raises(remote.RunPodError) as exc:
        client.submit({"image_base64": "secret image content"})
    assert len(opener.calls) == 1
    assert "403" in str(exc.value)
    assert "secret" not in str(exc.value)
    assert "malicious" not in str(exc.value)


def test_endpoint_and_job_ids_cannot_inject_urls():
    with pytest.raises(ValueError, match="ENDPOINT_ID"):
        remote.RunPodClient("key", "https://evil.test/path")
    client, _ = _client([])
    with pytest.raises(ValueError, match="job ID"):
        client.wait("../../evil")
    assert remote._NoRedirect().redirect_request(None, None, 302, "", {},
                                                 "https://evil.test") is None


def test_payload_limits_preserve_bytes_and_select_sync_once(monkeypatch):
    monkeypatch.setattr(remote, "MAX_RUN_PAYLOAD_BYTES", 100)
    monkeypatch.setattr(remote, "MAX_SYNC_PAYLOAD_BYTES", 300)
    client, opener = _client([{"id": "sync-job", "status": "COMPLETED",
                               "output": _count_output()}])
    client.run({"image_base64": "A" * 100})
    assert len(opener.calls) == 1
    assert "/runsync?wait=" in opener.calls[0][0].full_url
    with pytest.raises(ValueError, match="object storage"):
        client.run({"image_base64": "A" * 400})
    assert len(opener.calls) == 1


def test_adapter_count_preserves_image_bytes_and_draws_locally(tmp_path):
    source = _image(tmp_path / "input.png")
    client, opener = _client([
        {"id": "job-3", "status": "IN_QUEUE"},
        {"id": "job-3", "status": "COMPLETED", "output": _count_output(),
         "delayTime": 1, "executionTime": 2},
    ])
    engine = remote.RunPodInferenceEngine(client=client)
    result = engine.count("countgd-plusplus", [{"path": source, "boxes": [[2, 3, 12, 15]]}],
                          [source], str(tmp_path / "out"), confidence=.8,
                          label="rings", reference_mode="per_image")[0]
    payload = json.loads(opener.calls[0][0].data)["input"]
    assert payload == {"action": "count", "image_base64": base64.b64encode(Path(source).read_bytes()).decode(),
                       "boxes": [[2, 3, 12, 15]], "confidence": .8, "label": "rings"}
    assert result["count"] == 1 and result["width"] == 40 and result["height"] == 30
    assert result["annotated_path"].startswith(str(tmp_path / "out"))
    with Image.open(result["annotated_path"]) as image:
        assert image.format == "JPEG" and image.size == (40, 30)
    assert engine.last_jobs[0]["worker_metrics"]["gpu_name"] == "NVIDIA A100"


@pytest.mark.parametrize("mutation", [
    lambda result: result.update(width=41),
    lambda result: result.update(count=2),
    lambda result: result["detections"][0].update(box=[0, 0, 41, 10]),
    lambda result: result["detections"][0].update(confidence=float("nan")),
    lambda result: result.update(annotated_path="/tmp/malicious.jpg"),
    lambda result: result.update(processing={"tiles": -1}),
])
def test_adapter_rejects_bad_count_result_without_annotation(tmp_path, mutation):
    source = _image(tmp_path / "input.png")
    output = _count_output()
    mutation(output["result"])
    client, _ = _client([{"id": "job-4", "status": "IN_QUEUE"},
                         {"id": "job-4", "status": "COMPLETED", "output": output}])
    engine = remote.RunPodInferenceEngine(client=client)
    with pytest.raises(remote.RunPodError):
        engine.count("countgd-plusplus", [{"path": source, "boxes": [[2, 3, 12, 15]]}],
                     [source], str(tmp_path / "out"), reference_mode="per_image")
    assert not list((tmp_path / "out").glob("*.jpg"))


def test_adapter_preserves_count_when_edge_boxes_clip_to_same_coordinates(tmp_path):
    source = _image(tmp_path / "input.png")
    output = _count_output()
    output["result"]["count"] = 2
    output["result"]["detections"] = [
        {"index": 1, "box": [0, 0, 2, 3], "confidence": .8},
        {"index": 2, "box": [0, 0, 2, 3], "confidence": .7},
    ]
    client, _ = _client([{"id": "job-12", "status": "IN_QUEUE"},
                         {"id": "job-12", "status": "COMPLETED", "output": output}])
    engine = remote.RunPodInferenceEngine(client=client)
    result = engine.count("countgd-plusplus", [{"path": source, "boxes": [[2, 3, 12, 15]]}],
                          [source], str(tmp_path / "out"), reference_mode="per_image")[0]
    assert result["count"] == 2
    assert [item["index"] for item in result["detections"]] == [1, 2]
    assert result["detections"][0]["box"] == result["detections"][1]["box"]


def test_adapter_rejects_non_countgd_and_shared_before_network(tmp_path):
    source = _image(tmp_path / "input.png")
    client, opener = _client([])
    engine = remote.RunPodInferenceEngine(client=client)
    for model_id, mode in (("yoloe-26n-seg", "per_image"), ("countgd-plusplus", "shared")):
        with pytest.raises(ValueError, match="only CountGD"):
            engine.count(model_id, [{"path": source, "boxes": [[1, 1, 10, 10]]}],
                         [source], str(tmp_path), reference_mode=mode)
    assert not opener.calls


def test_suggest_response_shape_and_validation(tmp_path):
    source = _image(tmp_path / "input.png")
    result = {"width": 40, "height": 30, "model": "fastsam-s", "boxes": [[1, 1, 10, 10]],
              "elapsed_ms": 12, "warning": None}
    client, opener = _client([{"id": "job-5", "status": "IN_QUEUE"},
                              {"id": "job-5", "status": "COMPLETED",
                               "output": {"result": result, "metrics": {}}}])
    engine = remote.RunPodInferenceEngine(client=client)
    assert engine.suggest_reference_boxes(source, "screws")["boxes"] == [[1, 1, 10, 10]]
    assert json.loads(opener.calls[0][0].data)["input"]["label"] == "screws"


def test_catalog_requires_endpoint_and_key(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "good-id")
    assert not remote.remote_model_catalog()["models"][0]["available"]
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    assert remote.remote_model_catalog()["models"][0]["available"]
    assert remote.remote_model_catalog()["device"] == "remote"
