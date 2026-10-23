"""Management helpers are tested without credentials or paid API calls."""

import io
import json
from urllib.error import HTTPError

import pytest

from scripts.runpod_deploy import (ApiError, OwnedEndpoints, RunPodApi, endpoint_body,
                                  normalize_gpu, select_gpu)


CATALOG = [
    {"id": "NVIDIA RTX A5000", "name": "RTX A5000", "pool": "AMPERE_24", "price": {"serverless": .576}},
    {"id": "NVIDIA GeForce RTX 3090", "name": "RTX 3090", "pool": "AMPERE_24", "price": {"serverless": .576}},
    {"id": "NVIDIA L4", "name": "L4", "pool": "ADA_24", "price": {"serverless": .684}},
    {"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "pool": "ADA_24", "price": {"serverless": 1.1}},
]


class FakeApi:
    api_key = "not-a-real-key"

    def __init__(self):
        self.calls, self.live = [], {}
        self.fail_create = False
        self.fail_delete = False

    def catalog(self):
        return CATALOG

    def request(self, method, path, data=None, **kwargs):
        self.calls.append((method, path, data))
        if method == "POST" and path == "/serverless":
            endpoint = {**data, "id": f"endpoint-{len(self.live) + 1}"}
            self.live[endpoint["id"]] = endpoint
            if self.fail_create:
                raise ApiError("Network error after remote create")
            return endpoint
        if method == "GET" and path.startswith("/serverless?"):
            return {"endpoints": list(self.live.values()), "pagination": {"hasNextPage": False}}
        if path.endswith("/workers"):
            return {"workers": []}
        if path.startswith("/billing/serverless?"):
            return {"records": []}
        endpoint_id = path.rsplit("/", 1)[-1]
        if endpoint_id not in self.live:
            raise ApiError("Not found", 404)
        if method == "GET":
            return self.live[endpoint_id]
        if method == "DELETE":
            if self.fail_delete:
                raise ApiError("Unavailable", 503)
            del self.live[endpoint_id]
            return {}
        raise AssertionError((method, path))


def test_exact_gpu_selection_excludes_every_other_card_in_pool():
    gpu = select_gpu(CATALOG, "RTX A5000")
    body = endpoint_body("benchmark", "ghcr.io/example/worker@sha256:abc", gpu)
    assert body["gpu"] == {"pools": ["AMPERE_24"], "excludedTypes": ["NVIDIA GeForce RTX 3090"],
                           "count": 1, "minCudaVersion": "12.8"}
    assert "allowedCudaVersions" not in body["gpu"]
    assert body["workers"] == {"min": 0, "max": 1, "idleTimeout": 30}
    assert body["timeout"] == 300000
    assert body["flashboot"] == "OFF"
    assert "templateId" not in body
    assert normalize_gpu("NVIDIA L4") == normalize_gpu("L4")


@pytest.mark.parametrize("catalog,name", [
    (CATALOG, "A5000"),
    (CATALOG + [CATALOG[0]], "RTX A5000"),
    ([{**CATALOG[0], "pool": None}], "RTX A5000"),
    ([{**CATALOG[0], "price": {"serverless": 0}}], "RTX A5000"),
])
def test_ambiguous_or_unquoted_hardware_is_rejected(catalog, name):
    with pytest.raises(ValueError):
        select_gpu(catalog, name)


def test_manifest_persists_intent_and_cleanup_recovers_uncertain_create(tmp_path):
    api = FakeApi()
    api.fail_create = True
    owned = OwnedEndpoints(api, tmp_path / "resources.json")
    with pytest.raises(ApiError):
        owned.create("worker:v1", select_gpu(CATALOG, "L4"), env={"SECRET": "do-not-save"})
    persisted = (tmp_path / "resources.json").read_text()
    assert "do-not-save" not in persisted
    assert json.loads(persisted)["resources"][0]["state"] == "creating"
    assert not OwnedEndpoints(api, tmp_path / "resources.json").cleanup()
    assert not api.live
    assert sum(method == "POST" for method, _, _ in api.calls) == 1


def test_cleanup_never_deletes_unowned_or_changed_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.runpod_deploy.time.sleep", lambda seconds: None)
    api = FakeApi()
    owned = OwnedEndpoints(api, tmp_path / "resources.json")
    record = owned.create("worker:v1", select_gpu(CATALOG, "L4"))
    api.live["unrelated"] = {"id": "unrelated", "name": "someone-else", "image": "worker:v1"}
    api.live[record["id"]]["name"] = "renamed"
    assert owned.cleanup()
    assert len(api.live) == 2
    assert not any(method == "DELETE" for method, _, _ in api.calls)


def test_cleanup_can_be_retried_after_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.runpod_deploy.time.sleep", lambda seconds: None)
    api = FakeApi()
    owned = OwnedEndpoints(api, tmp_path / "resources.json")
    record = owned.create("worker:v1", select_gpu(CATALOG, "L4"))
    api.fail_delete = True
    assert owned.cleanup()[0]["id"] == record["id"]
    api.fail_delete = False
    assert not OwnedEndpoints(api, tmp_path / "resources.json").cleanup()
    assert not api.live


def test_http_failure_does_not_expose_key_or_provider_response():
    class Opener:
        def open(self, req, **kwargs):
            assert req.get_header("Authorization") == "Bearer secret-value"
            raise HTTPError(req.full_url, 403, "secret-value", {}, io.BytesIO(b"secret-value"))

    with pytest.raises(ApiError) as exc:
        RunPodApi("secret-value", opener=Opener()).request("GET", "/catalog/gpus")
    assert str(exc.value) == "RunPod management request failed (HTTP 403)."


def test_management_rejects_full_or_protocol_relative_url():
    api = RunPodApi("not-real")
    for path in ("https://other-host/path", "//other-host/path"):
        with pytest.raises(ValueError):
            api.request("GET", path)


def test_created_endpoint_safeguards_are_read_back_before_any_job(tmp_path):
    class WidenedApi(FakeApi):
        def request(self, method, path, data=None, **kwargs):
            value = super().request(method, path, data, **kwargs)
            if method == "GET" and path.startswith("/serverless/"):
                value = {**value, "workers": {"min": 0, "max": 10, "idleTimeout": 30}}
            return value

    api = WidenedApi()
    owned = OwnedEndpoints(api, tmp_path / "resources.json")
    with pytest.raises(ApiError, match="spending safeguards"):
        owned.create("worker:v1", select_gpu(CATALOG, "L4"))
    assert not owned.cleanup()
    assert not api.live


def test_definite_rejected_create_needs_no_resource_deletion(tmp_path):
    class RejectedApi(FakeApi):
        def request(self, method, path, data=None, **kwargs):
            if method == "POST":
                raise ApiError("Forbidden", 403)
            raise AssertionError("Rejected creation must not trigger cleanup network calls")

    owned = OwnedEndpoints(RejectedApi(), tmp_path / "resources.json")
    with pytest.raises(ApiError):
        owned.create("worker:v1", select_gpu(CATALOG, "L4"))
    assert not owned.cleanup()
    assert owned.data["resources"][0]["state"] == "rejected"


@pytest.mark.parametrize("reported_version", [None, "12.6"])
def test_missing_or_changed_cuda_floor_rejects_created_endpoint(tmp_path, reported_version):
    class WrongCudaApi(FakeApi):
        def request(self, method, path, data=None, **kwargs):
            value = super().request(method, path, data, **kwargs)
            if method == "GET" and path.startswith("/serverless/"):
                value = {**value, "gpu": {**value["gpu"], "minCudaVersion": reported_version}}
            return value

    api = WrongCudaApi()
    owned = OwnedEndpoints(api, tmp_path / "resources.json")
    with pytest.raises(ApiError, match="spending safeguards"):
        owned.create("worker:v1", select_gpu(CATALOG, "L4"))
    assert not owned.cleanup()
    assert not api.live
