"""Detection-only persistence and completion notifications, without a model or cloud job."""

import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
import pytest

from app.main import create_app
from test_api import FakeEngine, png, settled, upload


MODELS = [{"id": "countgd-plusplus", "available": True, "reference_mode": "per_image"}]


def create_count(client):
    response = client.post("/api/sessions", data={"model": "countgd-plusplus"})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def submit(client, sid):
    response = client.post(f"/api/sessions/{sid}/batches",
                           data={"image_boxes": "[[[1,1,8,8]]]"},
                           files=[("images", upload("parts.png", png("blue")))])
    assert response.status_code == 202, response.text
    return response.json()["id"]


def test_count_and_review_save_data_without_rendering_images(tmp_path, monkeypatch):
    import app.inference as inference

    monkeypatch.setattr(inference, "_draw_detections", lambda *args: pytest.fail("JPEG rendered"))
    app = create_app(tmp_path, FakeEngine, lambda: MODELS)
    with TestClient(app) as client:
        sid = create_count(client)
        saved, batch = settled(client, sid, submit(client, sid))
        assert batch["status"] == "complete", batch
        item = batch["results"][0]
        assert item["count"] == 1 and len(item["detections"]) == 1
        assert item["annotated_url"] is None and item["model_annotated_url"] is None
        assert client.get(item["original_url"]).status_code == 200
        assert len(list((tmp_path / sid / "files").iterdir())) == 1
        changed = client.post(f"/api/sessions/{sid}/images/{item['id']}/review", json={
            "revision": 0, "boxes": [[1, 1, 8, 8], [9, 1, 15, 8]], "reviewed": True})
        assert changed.status_code == 200, changed.text
        reviewed = changed.json()["batches"][0]["results"][0]
        assert reviewed["count"] == 2 and reviewed["model_count"] == 1
        assert reviewed["annotated_url"] is None
        assert changed.json()["revision"] != saved["revision"]
        assert len(list((tmp_path / sid / "files").iterdir())) == 1
    with TestClient(create_app(tmp_path, FakeEngine, lambda: MODELS)) as client:
        restored = client.get(f"/api/sessions/{sid}").json()
        assert restored["total_count"] == 2
        assert restored["batches"][0]["results"][0]["detections"] == reviewed["detections"]


class BlockedEngine(FakeEngine):
    def __init__(self, fail=False):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail_next = fail

    def count(self, *args, **kwargs):
        self.entered.set()
        assert self.release.wait(5), "test did not release counting"
        assert kwargs["annotate"] is False
        return super().count(*args, **kwargs)


@pytest.mark.parametrize("fail", [False, True])
def test_long_poll_wakes_all_clients_on_completion_or_failure(tmp_path, monkeypatch, fail):
    engine = BlockedEngine(fail)
    app = create_app(tmp_path, lambda: engine, lambda: MODELS)
    with TestClient(app) as client:
        sid = create_count(client)
        submit(client, sid)
        assert engine.entered.wait(2)
        revision = client.get(f"/api/sessions/{sid}").json()["revision"]
        read = app.state.store.detail
        reads = threading.Semaphore(0)

        def observed_read(session_id):
            result = read(session_id)
            reads.release()
            return result

        monkeypatch.setattr(app.state.store, "detail", observed_read)
        try:
            with ThreadPoolExecutor(2) as pool:
                waiting = [pool.submit(client.get, f"/api/sessions/{sid}",
                                       params={"after": revision, "wait": 25}) for _ in range(2)]
                assert all(reads.acquire(timeout=2) for _ in range(2))
                assert all(not future.done() for future in waiting)
                engine.release.set()
                replies = [future.result(timeout=3) for future in waiting]
            for reply in replies:
                assert reply.status_code == 200, reply.text
                result = reply.json()
                assert result["revision"] != revision
                assert result["batches"][0]["status"] == ("failed" if fail else "complete")
                assert result["total_count"] == (0 if fail else 1)
        finally:
            engine.release.set()


def test_long_poll_timeout_and_already_changed_revision(tmp_path):
    engine = BlockedEngine()
    with TestClient(create_app(tmp_path, lambda: engine, lambda: MODELS)) as client:
        sid = create_count(client)
        before = client.get(f"/api/sessions/{sid}").json()
        batch_id = submit(client, sid)
        assert engine.entered.wait(2)
        try:
            changed = client.get(f"/api/sessions/{sid}", params={"after": before["revision"], "wait": 25})
            assert changed.json()["revision"] != before["revision"]
            revision = changed.json()["revision"]
            unchanged = client.get(f"/api/sessions/{sid}", params={"after": revision, "wait": .05})
            assert unchanged.json()["revision"] == revision
        finally:
            engine.release.set()
        done, _ = settled(client, sid, batch_id)
        assert client.get(f"/api/sessions/{sid}", params={"after": revision, "wait": 25}).json() == done
        assert client.get("/api/sessions/missing", params={"after": revision, "wait": 25}).status_code == 404
        for wait in (-1, 31):
            assert client.get(f"/api/sessions/{sid}", params={"wait": wait}).status_code == 422


@pytest.mark.parametrize("mutation", [
    {"width": 15},
    {"elapsed_ms": True},
    {"detections": [{"index": 1, "box": [1, 1, 30, 8], "confidence": .9}]},
    {"detections": [{"index": 1, "box": [1, 1, 8, 8], "confidence": None}]},
])
def test_metadata_only_results_are_validated_before_persisting(tmp_path, mutation):
    class BadEngine(FakeEngine):
        def count(self, *args, **kwargs):
            result = super().count(*args, **kwargs)
            result[0].update(mutation)
            return result

    with TestClient(create_app(tmp_path, BadEngine, lambda: MODELS)) as client:
        sid = create_count(client)
        saved, batch = settled(client, sid, submit(client, sid))
        assert batch["status"] == "failed"
        assert saved["total_count"] == 0 and batch["results"] == []
