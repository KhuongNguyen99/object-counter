import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models import CROSSCHECK_MODEL_ID
from app.verification import compare_detections
from test_api import FakeEngine, png, upload, new_batch, settled


class ReviewEngine(FakeEngine):
    def count(self, *args, **kwargs):
        output = super().count(*args, **kwargs)
        for result in output:
            result["verification"] = compare_detections(result["detections"], result["detections"],
                "countgd-plusplus", "yoloe-26x-seg", .23, .01)
        return output


def app(tmp_path, engine=None):
    return create_app(tmp_path, lambda: engine or ReviewEngine(), lambda: [
        {"id": CROSSCHECK_MODEL_ID, "available": True}, {"id": "yoloe-26s-seg", "available": True}])


def session(client, model=CROSSCHECK_MODEL_ID):
    response = client.post("/api/sessions", data={"model": model, "confidence": .23,
        "reference_boxes": json.dumps([[[1, 1, 9, 9]]])},
        files=[("references", upload("ref.png", png("red")))])
    assert response.status_code == 201, response.text
    return response.json()["id"]


def result(client, sid):
    batch = new_batch(client, sid, ("target.png", png("blue")))
    saved, batch = settled(client, sid, batch.json()["id"])
    assert batch["status"] == "complete", batch
    return saved, batch["results"][0]


def review_url(sid, item):
    return f"/api/sessions/{sid}/images/{item['id']}/review"


def save(client, sid, item, boxes=None, revision=0, reviewed=True):
    return client.post(review_url(sid, item), json={"revision": revision, "reviewed": reviewed,
        "boxes": boxes if boxes is not None else [d["box"] for d in item["detections"]]})


def test_review_roundtrip_preserves_model_and_updates_total_annotation_csv_and_finish(tmp_path):
    with TestClient(app(tmp_path)) as client:
        sid = session(client)
        before, item = result(client, sid)
        assert before["requires_review"] and before["unreviewed_image_count"] == 1
        assert item["verification"]["matched"] == 1 and not item["reviewed"]
        assert client.post(f"/api/sessions/{sid}/finish").status_code == 409
        original_annotation = client.get(item["model_annotated_url"]).content
        draft = save(client, sid, item, [[1, 1, 8, 8], [10, 1, 15, 8]], reviewed=False)
        assert draft.status_code == 200, draft.text
        changed = draft.json()["batches"][0]["results"][0]
        assert changed["count"] == draft.json()["total_count"] == 2
        assert changed["model_count"] == 1 and changed["model_detections"] == item["detections"]
        assert changed["review_revision"] == 1 and not changed["reviewed"]
        assert [d["index"] for d in changed["detections"]] == [1, 2]
        assert all(d["confidence"] is None for d in changed["detections"])
        assert client.get(changed["annotated_url"]).status_code == 200
        assert client.get(changed["model_annotated_url"]).content == original_annotation
        assert client.post(f"/api/sessions/{sid}/finish").status_code == 409
        assert save(client, sid, changed).status_code == 409
        approved = save(client, sid, changed, revision=1).json()
        assert approved["reviewed_image_count"] == 1 and approved["unreviewed_image_count"] == 0
        reviewed = approved["batches"][0]["results"][0]
        assert reviewed["reviewed"] and reviewed["reviewed_at"] and reviewed["review_revision"] == 2
        assert client.get(changed["annotated_url"]).status_code == 404
        rows = list(csv.reader(io.StringIO(client.get(f"/api/sessions/{sid}/export.csv").text)))
        assert rows[1][3:5] == ["2", "2"] and "Human reviewed; model count 1" in rows[1][5]
        assert client.post(f"/api/sessions/{sid}/finish").status_code == 200
        assert save(client, sid, reviewed, revision=2).status_code == 409
    with TestClient(app(tmp_path)) as client:
        restored = client.get(f"/api/sessions/{sid}").json()
        assert restored["total_count"] == 2 and restored["reviewed_image_count"] == 1
        assert restored["batches"][0]["results"][0]["verification"] == item["verification"]


@pytest.mark.parametrize("boxes", [[], [[1, 1, 8, 8]]])
def test_confirmation_can_keep_boxes_or_clear_all(tmp_path, boxes):
    with TestClient(app(tmp_path)) as client:
        sid = session(client)
        _, item = result(client, sid)
        saved = save(client, sid, item, boxes).json()
        assert saved["total_count"] == len(boxes)
        assert saved["batches"][0]["results"][0]["reviewed"]


@pytest.mark.parametrize("payload", [
    {"revision": True, "boxes": [], "reviewed": True},
    {"revision": 0, "boxes": [], "reviewed": "true"},
    {"revision": 0, "boxes": [], "reviewed": True, "count": 999},
    {"revision": 0, "boxes": [[0, 0, 17, 2]], "reviewed": True},
    {"revision": 0, "boxes": [[0, 0, 0, 2]], "reviewed": True},
    {"revision": 0, "boxes": [[True, 0, 1, 2]], "reviewed": True},
    {"revision": 0, "boxes": [[0, 0, 1]], "reviewed": True},
])
def test_invalid_review_is_rejected_without_changing_model_result(tmp_path, payload):
    with TestClient(app(tmp_path)) as client:
        sid = session(client)
        _, item = result(client, sid)
        assert client.post(review_url(sid, item), json=payload).status_code == 422
        saved = client.get(f"/api/sessions/{sid}").json()["batches"][0]["results"][0]
        assert saved["review_revision"] == 0 and saved["count"] == 1
        assert not list((tmp_path / sid / "files").glob("reviewed-*"))


def test_image_ownership_and_legacy_finish_behavior(tmp_path):
    with TestClient(app(tmp_path)) as client:
        sid = session(client, "yoloe-26s-seg")
        _, item = result(client, sid)
        other = session(client)
        assert save(client, other, item).status_code == 404
        assert client.get(item["model_annotated_url"].replace(sid, other)).status_code == 404
        assert client.post(f"/api/sessions/{sid}/finish").status_code == 200


def test_concurrent_reviews_allow_only_one_revision(tmp_path):
    with TestClient(app(tmp_path)) as client:
        sid = session(client)
        _, item = result(client, sid)
        with ThreadPoolExecutor(2) as pool:
            responses = list(pool.map(lambda _: save(client, sid, item), range(2)))
        assert sorted(response.status_code for response in responses) == [200, 409]
        assert len(list((tmp_path / sid / "files").glob("reviewed-*"))) == 1


def test_review_render_failure_does_not_change_result_or_leak_slot(tmp_path, monkeypatch):
    import app.inference as inference

    with TestClient(app(tmp_path), raise_server_exceptions=False) as client:
        sid = session(client)
        _, item = result(client, sid)
        def fail(*args):
            raise OSError("disk error")
        with monkeypatch.context() as context:
            context.setattr(inference, "_draw_detections", fail)
            assert save(client, sid, item).status_code == 500
        assert save(client, sid, item).status_code == 200


def test_review_busy_and_finish_race_are_checked_at_commit(tmp_path, monkeypatch):
    import app.inference as inference

    instance = app(tmp_path)
    with TestClient(instance) as client:
        sid = session(client, "yoloe-26s-seg")
        _, item = result(client, sid)
        original = inference._draw_detections
        def finish_during_draw(*args):
            original(*args)
            assert instance.state.store.finish(sid) is None
        monkeypatch.setattr(inference, "_draw_detections", finish_during_draw)
        assert save(client, sid, item).status_code == 409
        assert not list((tmp_path / sid / "files").glob("reviewed-*"))


def test_pending_batch_rejects_review_without_overwrite(tmp_path):
    instance = app(tmp_path)
    with TestClient(instance) as client:
        sid = session(client)
        _, item = result(client, sid)
        with instance.state.store.connection() as db:
            db.execute("INSERT INTO batches (id, session_id, status, created_at, error) VALUES (?,?,?,?,?)",
                       ("pending", sid, "queued", "now", None))
        assert save(client, sid, item).status_code == 409
        assert not list((tmp_path / sid / "files").glob("reviewed-*"))
