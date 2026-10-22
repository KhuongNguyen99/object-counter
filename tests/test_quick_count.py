import io
import json
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.quick_count import _filter_sizes
from app.storage import Store


MODELS = [{"id": "countgd-plusplus", "available": True, "reference_mode": "per_image",
           "required_reference_boxes": 10}]


def photo(width=40, height=20):
    image = Image.new("RGB", (width, height), "red")
    for x in range(width // 2, width):
        for y in range(height):
            image.putpixel((x, y), (0, 0, 255))
    data = io.BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


def upload(name, payload):
    return (name, payload, "image/png")


class QuickEngine:
    def __init__(self, groups=None, fail_at=None):
        self.groups = groups or []
        self.fail_at = fail_at
        self.crops = []
        self.crop_paths = []
        self.count_calls = 0

    def count(self, **kwargs):
        self.count_calls += 1
        raise AssertionError("Quick count must not run CountGD")

    def suggest_reference_boxes(self, image_path, label):
        self.crop_paths.append(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
            self.crops.append((image.size, image.getpixel((0, 0)), label))
        index = len(self.crops) - 1
        if index == self.fail_at:
            raise RuntimeError("fake helper failure")
        boxes = self.groups[index] if index < len(self.groups) else [[1, 1, 5, 5]]
        return {"model": "fastsam-s", "width": width, "height": height,
                "elapsed_ms": 1, "warning": None, "boxes": boxes}


def session(client):
    response = client.post("/api/sessions", data={"model": "countgd-plusplus", "label": "buttons"})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def batch(client, session_id, areas, image=None, **fields):
    return client.post(f"/api/sessions/{session_id}/batches",
                       data={"count_mode": "quick", "search_areas": json.dumps(areas), **fields},
                       files=[("images", upload("objects.png", image or photo()))])


def settled(client, session_id, batch_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.get(f"/api/sessions/{session_id}").json()
        selected = next(row for row in result["batches"] if row["id"] == batch_id)
        if selected["status"] in {"complete", "failed"}:
            return selected
        time.sleep(.01)
    raise AssertionError("batch did not finish")


def test_quick_counts_only_native_pixel_crops_and_offsets_boxes_without_references(tmp_path):
    engine = QuickEngine()
    with TestClient(create_app(tmp_path, lambda: engine, lambda: MODELS)) as client:
        sid = session(client)
        areas = [[2, 2, 18, 18], [20, 2, 36, 18]]
        started = batch(client, sid, [areas])
        assert started.status_code == 202, started.text
        result = settled(client, sid, started.json()["id"])
        assert result["status"] == "complete", result
        assert result["count_mode"] == "quick"
        item = result["results"][0]
        assert item["search_areas"] == areas and item["reference_boxes"] == []
        assert item["count"] == 2
        assert [d["box"] for d in item["detections"]] == [[3, 3, 7, 7], [21, 3, 25, 7]]
        assert [d["index"] for d in item["detections"]] == [1, 2]
        assert all(d["confidence"] is None for d in item["detections"])
        assert item["elapsed_ms"] >= 0
        assert client.get(item["annotated_url"]).status_code == 200
    assert engine.count_calls == 0
    assert [entry[0] for entry in engine.crops] == [(16, 16), (16, 16)]
    assert [entry[2] for entry in engine.crops] == ["buttons", "buttons"]
    assert engine.crops[0][1][0] > 240 and engine.crops[0][1][2] < 10
    assert engine.crops[1][1][2] > 240 and engine.crops[1][1][0] < 10
    assert all(not Path(path).exists() for path in engine.crop_paths)


def test_quick_merges_overlapping_areas_and_filters_size_outlier(tmp_path):
    engine = QuickEngine(groups=[[[4, 4, 12, 12]], [[3, 4, 11, 12]]])
    with TestClient(create_app(tmp_path, lambda: engine, lambda: MODELS)) as client:
        sid = session(client)
        started = batch(client, sid, [[[0, 0, 20, 20], [1, 0, 21, 20]]])
        assert started.status_code == 202
        result = settled(client, sid, started.json()["id"])
        assert result["results"][0]["count"] == 1
        assert result["results"][0]["detections"][0]["box"] == [4, 4, 12, 12]

    engine = QuickEngine(groups=[[[1, 1, 5, 5], [7, 1, 11, 5], [13, 1, 17, 5],
                                  [19, 1, 23, 5], [30, 30, 65, 65]]])
    with TestClient(create_app(tmp_path / "size", lambda: engine, lambda: MODELS)) as client:
        sid = session(client)
        started = batch(client, sid, [[[0, 0, 80, 80]]], photo(80, 80))
        assert started.status_code == 202
        result = settled(client, sid, started.json()["id"])
        assert result["results"][0]["count"] == 4


def test_quick_size_filter_matches_recorded_photo_without_ten_box_cap():
    fixture = Path(__file__).parent / "fixtures" / "photo_mad_proposals.json"
    boxes = json.loads(fixture.read_text(encoding="utf-8"))["boxes"]
    kept = _filter_sizes(boxes)
    kept_ids = [index for index, box in enumerate(boxes, 1) if box in kept]
    assert kept_ids == [index for index in range(1, 33) if index not in {1, 12, 20, 21}]
    assert len(kept) == 28 and boxes[30] in kept


@pytest.mark.parametrize("raw", [
    None, "not json", "[]", "[[]]", "[[[1,2,3]]]", "[[[1,2,3,4.5]]]",
    "[[[true,2,8,8]]]", "[[[0,0,4,3]]]", "[[[0,0,41,10]]]",
    json.dumps([[[0, 0, 4, 4]] * 33]),
])
def test_quick_rejects_missing_malformed_or_oversized_areas_before_queue(tmp_path, raw):
    engine = QuickEngine()
    with TestClient(create_app(tmp_path, lambda: engine, lambda: MODELS)) as client:
        sid = session(client)
        fields = {"count_mode": "quick"}
        if raw is not None:
            fields["search_areas"] = raw
        response = client.post(f"/api/sessions/{sid}/batches", data=fields,
                               files=[("images", upload("objects.png", photo()))])
        assert response.status_code == 422, response.text
        assert client.get(f"/api/sessions/{sid}").json()["batches"] == []
        assert list((tmp_path / sid / "files").iterdir()) == []
    assert engine.crops == []


def test_quick_rejects_reference_boxes_and_standard_rejects_areas(tmp_path):
    with TestClient(create_app(tmp_path, QuickEngine, lambda: MODELS)) as client:
        sid = session(client)
        assert batch(client, sid, [[[0, 0, 10, 10]]],
                     image_boxes="[[[0,0,10,10]]]").status_code == 422
        response = client.post(f"/api/sessions/{sid}/batches",
                               data={"search_areas": "[[[0,0,10,10]]]"},
                               files=[("images", upload("objects.png", photo()))])
        assert response.status_code == 422


def test_quick_failure_removes_crops_and_uploaded_images_and_releases_queue(tmp_path):
    engine = QuickEngine(fail_at=1)
    with TestClient(create_app(tmp_path, lambda: engine, lambda: MODELS)) as client:
        sid = session(client)
        started = batch(client, sid, [[[0, 0, 10, 10], [20, 0, 30, 10]]])
        assert started.status_code == 202
        result = settled(client, sid, started.json()["id"])
        assert result["status"] == "failed"
        assert list((tmp_path / sid / "files").iterdir()) == []
        assert not (tmp_path / sid / "work" / started.json()["id"]).exists()
        engine.fail_at = None
        second = batch(client, sid, [[[0, 0, 10, 10]]])
        assert second.status_code == 202
        assert settled(client, sid, second.json()["id"])["status"] == "complete"
    assert all(not Path(path).exists() for path in engine.crop_paths)


def test_old_database_migrates_to_standard_mode_and_empty_areas(tmp_path):
    Store(tmp_path)
    db_path = tmp_path / "sessions.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute("ALTER TABLE batches DROP COLUMN count_mode")
        db.execute("ALTER TABLE images DROP COLUMN search_areas")
        db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)",
                   ("old", "older", "countgd-plusplus", .8, "active", "yesterday"))
        db.execute("INSERT INTO batches VALUES (?,?,?,?,?)", ("old-batch", "old", "complete", "yesterday", None))
        db.execute("""INSERT INTO images (id,batch_id,ordinal,filename,original_filename,sha256,
                    count,detections,annotated_filename,width,height)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                   ("old-image", "old-batch", 0, "old.png", "old.jpg", "sha", 0, "[]",
                    "annotated.jpg", 40, 20))
    restored = Store(tmp_path).detail("old")
    assert restored["batches"][0]["count_mode"] == "standard"
    assert restored["batches"][0]["results"][0]["search_areas"] == []


@pytest.mark.parametrize("number", [32, 40])
def test_quick_counts_all_suggestions_without_a_32_box_limit_or_limit_warning(tmp_path, number):
    boxes = [[2 + (i % 10) * 8, 2 + (i // 10) * 8,
              6 + (i % 10) * 8, 6 + (i // 10) * 8] for i in range(number)]
    engine = QuickEngine(groups=[boxes])
    with TestClient(create_app(tmp_path, lambda: engine, lambda: MODELS)) as client:
        sid = session(client)
        started = batch(client, sid, [[[0, 0, 100, 80]]], photo(100, 80))
        assert started.status_code == 202, started.text
        result = settled(client, sid, started.json()["id"])
        assert result["status"] == "complete", result
        item = result["results"][0]
        assert item["count"] == number
        assert [detection["box"] for detection in item["detections"]] == boxes
        assert item["warning"] is None
