import asyncio
import csv
import io
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import httpx
from PIL import Image
import pytest

from app import main as main_module
from app.main import create_app


def test_dictionary_model_catalog_detects_runtime_device(monkeypatch):
    monkeypatch.setattr(main_module, "_device", lambda _configured: "cuda:0")
    catalog = main_module._model_catalog(lambda: {"models": [], "default_model": "countgd-plusplus"}, True)
    assert catalog["device"] == "cuda:0"
    assert catalog["default_model"] == "countgd-plusplus"


@pytest.mark.parametrize("countgd_available", [True, False])
def test_production_catalog_only_exposes_countgd_even_with_other_models(tmp_path, monkeypatch,
                                                                         countgd_available):
    from app import models as model_module

    monkeypatch.setattr(model_module, "list_models", lambda: [
        {"id": "sam3", "available": True},
        {"id": "yoloe-26s-seg", "available": True},
        {"id": "countgd-plusplus", "available": countgd_available,
         "reference_mode": "per_image", "required_reference_boxes": 10,
         "default_confidence": .80},
    ])
    monkeypatch.setattr(main_module, "_device", lambda _: "cpu")
    with TestClient(create_app(tmp_path)) as client:
        catalog = client.get("/api/models").json()
        assert catalog["default_model"] == "countgd-plusplus"
        assert [model["id"] for model in catalog["models"]] == ["countgd-plusplus"]
        assert catalog["models"][0]["available"] == countgd_available
        rejected = client.post("/api/sessions", data={"model": "sam3"})
        assert rejected.status_code == 422
        assert rejected.json()["detail"] == "Counting is unavailable right now. Please try again."
        if countgd_available:
            created = client.post("/api/sessions", data={"model": "countgd-plusplus"})
            assert created.status_code == 201, created.text
            assert created.json()["confidence"] == .80
            monkeypatch.setattr(model_module, "list_models", lambda: [
                {"id": "countgd-plusplus", "available": False,
                 "reference_mode": "per_image", "required_reference_boxes": 10},
            ])
            blocked = new_batch(client, created.json()["id"], ("target.png", png("blue")))
            assert blocked.status_code == 422
        else:
            assert client.post("/api/sessions", data={"model": "countgd-plusplus"}).status_code == 422


MODELS = [{"id": "yoloe-26s-seg", "name": "YOLOE-26 Small", "family": "YOLOE-26",
           "size": "s", "available": True, "bytes": 100}]


def png(color):
    output = io.BytesIO()
    Image.new("RGB", (16, 12), color).save(output, format="PNG")
    return output.getvalue()


def upload(name, data):
    return (name, data, "application/octet-stream")


def proposal_request(client, name, data, label="rings"):
    return client.post("/api/reference-boxes", data={} if label is None else {"label": label},
                       files={"image": upload(name, data)})


class FakeEngine:
    def __init__(self):
        self.fail_next = False
        self.calls = []
        self.reference_modes = []
        self.labels = []
        self.result_count = 1
        self.warning = None
        self.processing = None
        self.suggest_calls = []
        self.suggest_labels = []
        self.suggest_error = None
        self.proposal = None

    def suggest_reference_boxes(self, image_path, label):
        self.suggest_calls.append(image_path)
        self.suggest_labels.append(label)
        if self.suggest_error is not None:
            raise self.suggest_error
        with Image.open(image_path) as image:
            width, height = image.size
        return self.proposal if self.proposal is not None else {
            "width": width, "height": height, "boxes": [[1, 1, width - 1, height - 1]],
            "model": "fastsam-s", "elapsed_ms": 3.5, "warning": None,
        }

    def count(self, model_id, references, image_paths, output_dir, confidence, label="",
              reference_mode="shared"):
        self.calls.append((model_id, references, image_paths, confidence))
        self.reference_modes.append(reference_mode)
        self.labels.append(label)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("fake inference failed")
        output = []
        for i, path in enumerate(image_paths):
            annotated = f"{output_dir}/annotation-{i}.png"
            with Image.open(path) as image:
                image.save(annotated)
            detections = ([{"index": 1, "box": [1, 1, 8, 8], "confidence": .9}]
                          if self.result_count else [])
            result = {"count": self.result_count, "detections": detections,
                      "annotated_path": annotated, "width": 16, "height": 12,
                      "elapsed_ms": 2.5, "warning": self.warning}
            if self.processing is not None:
                result["processing"] = self.processing
            output.append(result)
        return output


def new_session(client, name="Count test", boxes=None, reference=None):
    return client.post("/api/sessions", data={
        "label": name, "model": "yoloe-26s-seg", "confidence": ".25",
        "reference_boxes": json.dumps(boxes if boxes is not None else [[[1, 1, 9, 9]]]),
    }, files=[("references", upload("reference.png", reference or png("red")))])


def new_batch(client, session_id, *images):
    return client.post(f"/api/sessions/{session_id}/batches",
                       files=[("images", upload(name, data)) for name, data in images])


def settled(client, session_id, batch_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        session = client.get(f"/api/sessions/{session_id}").json()
        batch = next(batch for batch in session["batches"] if batch["id"] == batch_id)
        if batch["status"] in {"complete", "failed"}:
            return session, batch
        time.sleep(.01)
    raise AssertionError("batch did not finish")


def test_historical_legacy_count_is_readable_but_cannot_start_another_batch(tmp_path, monkeypatch):
    from app import models as model_module

    with TestClient(create_app(tmp_path, FakeEngine, lambda: MODELS)) as client:
        created = new_session(client)
        assert created.status_code == 201, created.text
        session_id = created.json()["id"]
        batch = new_batch(client, session_id, ("target.png", png("blue")))
        assert batch.status_code == 202, batch.text
        assert settled(client, session_id, batch.json()["id"])[1]["status"] == "complete"

    monkeypatch.setattr(model_module, "list_models", lambda: [
        {"id": "countgd-plusplus", "available": True,
         "reference_mode": "per_image", "required_reference_boxes": 10},
        *MODELS,
    ])
    with TestClient(create_app(tmp_path, FakeEngine)) as client:
        saved = client.get(f"/api/sessions/{session_id}")
        assert saved.status_code == 200
        assert saved.json()["model"] == "yoloe-26s-seg"
        assert saved.json()["total_count"] == 1
        assert saved.json()["batches"][0]["results"][0]["count"] == 1
        assert new_batch(client, session_id, ("new.png", png("green"))).status_code == 422


@pytest.mark.parametrize("model", ["yoloe-26s-seg", "pet-dino-swin-t"])
@pytest.mark.parametrize("label", [None, "", "   "])
def test_reference_models_count_with_optional_description(tmp_path, model, label):
    engine = FakeEngine()
    models = [{"id": model, "available": True}]
    data = {"model": model, "reference_boxes": "[[[1,1,9,9]]]"}
    if label is not None:
        data["label"] = label
    with TestClient(create_app(tmp_path, lambda: engine, lambda: models)) as client:
        created = client.post("/api/sessions", data=data,
                              files=[("references", upload("ref.png", png("red")))])
        assert created.status_code == 201
        session_id = created.json()["id"]
        assert created.json()["label"] == ""
        batch = new_batch(client, session_id, ("target.png", png("blue"))).json()
        restored, result = settled(client, session_id, batch["id"])
        assert result["status"] == "complete" and restored["total_count"] == 1
        assert restored["label"] == "" and engine.labels == [""]


@pytest.mark.parametrize("label", [None, "", "   "])
def test_text_only_model_requires_description(tmp_path, label):
    models = [{"id": "sam3count", "available": True, "reference_mode": "none"}]
    data = {"model": "sam3count"}
    if label is not None:
        data["label"] = label
    with TestClient(create_app(tmp_path, FakeEngine, lambda: models)) as client:
        assert client.post("/api/sessions", data=data).status_code == 422


@pytest.mark.parametrize("label", ["bad\nlabel", "x" * 81])
def test_optional_description_still_validates_supplied_text(tmp_path, label):
    with TestClient(create_app(tmp_path, FakeEngine, lambda: MODELS)) as client:
        assert new_session(client, name=label).status_code == 422


def test_text_only_model_counts_without_references_and_rejects_ignored_uploads(tmp_path):
    engine = FakeEngine()
    models = MODELS + [{"id": "sam3count", "available": True, "reference_mode": "none"}]
    with TestClient(create_app(tmp_path, lambda: engine, lambda: models)) as client:
        data = {"label": "white rings", "model": "sam3count"}
        created = client.post("/api/sessions", data=data)
        assert created.status_code == 201
        session_id = created.json()["id"]
        assert created.json()["references"] == []
        batch = new_batch(client, session_id, ("rings.png", png("white")))
        saved, result = settled(client, session_id, batch.json()["id"])
        assert result["status"] == "complete" and saved["total_count"] == 1
        assert engine.calls[0][1] == [] and engine.labels == ["white rings"]
        assert client.post("/api/sessions", data=data,
                           files=[("references", upload("ref.png", png("red")))]).status_code == 422
        assert client.post("/api/sessions", data={**data, "reference_boxes": "[[[0,0,1,1]]]"}).status_code == 422
        assert client.post("/api/sessions", data={**data, "model": "yoloe-26s-seg"}).status_code == 422


def test_sam3_uses_each_target_as_its_own_reference_and_persists_boxes(tmp_path):
    engine = FakeEngine()
    models = MODELS + [{"id": "sam3", "available": True, "reference_mode": "per_image"}]
    boxes = [[[1, 1, 8, 8]] * 20, [[2, 2, 10, 9]] * 20]
    with TestClient(create_app(tmp_path, lambda: engine, lambda: models)) as client:
        created = client.post("/api/sessions", data={"model": "sam3"})
        assert created.status_code == 201, created.text
        session_id = created.json()["id"]
        assert created.json()["label"] == "" and created.json()["references"] == []
        assert client.post("/api/sessions", data={"model": "sam3"},
                           files=[("references", upload("ref.png", png("red")))]).status_code == 422
        assert client.post("/api/sessions", data={"model": "sam3",
                           "reference_boxes": "[[[1,1,8,8]]]"}).status_code == 422

        first = client.post(f"/api/sessions/{session_id}/batches",
                            data={"image_boxes": json.dumps(boxes)}, files=[
                                ("images", upload("blue.png", png("blue"))),
                                ("images", upload("green.png", png("green"))),
                            ])
        assert first.status_code == 202, first.text
        _, batch = settled(client, session_id, first.json()["id"])
        assert batch["status"] == "complete", batch
        model_id, references, paths, _ = engine.calls[0]
        assert model_id == "sam3"
        assert [ref["path"] for ref in references] == paths
        assert all(Path(path).is_absolute() for path in paths)
        assert [ref["boxes"] for ref in references] == boxes
        assert [result["reference_boxes"] for result in batch["results"]] == boxes

        later_boxes = [[[3, 3, 11, 10]]]
        later = client.post(f"/api/sessions/{session_id}/batches",
                            data={"image_boxes": json.dumps(later_boxes)},
                            files=[("images", upload("yellow.png", png("yellow")))])
        assert later.status_code == 202, later.text
        _, later_batch = settled(client, session_id, later.json()["id"])
        assert later_batch["status"] == "complete"
        assert len(engine.calls[1][1]) == 1
        assert engine.calls[1][1][0]["path"] == engine.calls[1][2][0]
        assert engine.calls[1][1][0]["boxes"] == later_boxes[0]

    with TestClient(create_app(tmp_path, FakeEngine, lambda: models)) as client:
        restored = client.get(f"/api/sessions/{session_id}").json()
        assert [[result["reference_boxes"] for result in batch["results"]]
                for batch in restored["batches"]] == [boxes, later_boxes]


def test_countgd_uses_ten_boxes_from_each_photo_across_batches(tmp_path):
    engine = FakeEngine()
    models = [{"id": "countgd-plusplus", "available": True, "reference_mode": "per_image",
               "required_reference_boxes": 10}]
    boxes = [[[i, 1, i + 4, 5] for i in range(1, 11)],
             [[i, 2, i + 4, 6] for i in range(1, 11)]]
    with TestClient(create_app(tmp_path, lambda: engine, lambda: models)) as client:
        created = client.post("/api/sessions", data={"model": "countgd-plusplus", "label": ""})
        assert created.status_code == 201, created.text
        session_id = created.json()["id"]
        assert created.json()["references"] == []
        assert created.json()["reference_mode"] == "per_image"
        assert created.json()["required_reference_boxes"] == 10
        assert client.post("/api/sessions", data={"model": "countgd-plusplus"},
                           files=[("references", upload("ref.png", png("red")))]).status_code == 422
        first = client.post(f"/api/sessions/{session_id}/batches",
                            data={"image_boxes": json.dumps(boxes)}, files=[
                                ("images", upload("blue.png", png("blue"))),
                                ("images", upload("green.png", png("green"))),
                            ])
        assert first.status_code == 202, first.text
        _, batch = settled(client, session_id, first.json()["id"])
        assert batch["status"] == "complete"
        assert engine.reference_modes == ["per_image"]
        assert [ref["path"] for ref in engine.calls[0][1]] == engine.calls[0][2]
        assert [ref["boxes"] for ref in engine.calls[0][1]] == boxes
        assert [result["reference_boxes"] for result in batch["results"]] == boxes

        later_boxes = [[[i, 3, i + 4, 7] for i in range(1, 11)]]
        later = client.post(f"/api/sessions/{session_id}/batches",
                            data={"image_boxes": json.dumps(later_boxes)},
                            files=[("images", upload("yellow.png", png("yellow")))])
        assert later.status_code == 202, later.text
        assert settled(client, session_id, later.json()["id"])[1]["status"] == "complete"
        assert len(engine.calls[1][1]) == 1
        assert engine.calls[1][1][0]["path"] == engine.calls[1][2][0]
        assert engine.calls[1][1][0]["boxes"] == later_boxes[0]

    with TestClient(create_app(tmp_path, FakeEngine, lambda: models)) as client:
        restored = client.get(f"/api/sessions/{session_id}").json()
        assert restored["reference_mode"] == "per_image"
        assert restored["required_reference_boxes"] == 10
        assert [[result["reference_boxes"] for result in batch["results"]]
                for batch in restored["batches"]] == [boxes, later_boxes]


@pytest.mark.parametrize("boxes", [
    [[i, 1, i + 4, 5] for i in range(1, 10)],
    [[i, 1, i + 4, 5] for i in range(1, 12)],
    [[1, 1, 8, 8]] * 10,
])
def test_countgd_rejects_wrong_or_duplicate_boxes_before_queue(tmp_path, boxes):
    engine = FakeEngine()
    models = [{"id": "countgd-plusplus", "available": True, "reference_mode": "per_image",
               "required_reference_boxes": 10}]
    with TestClient(create_app(tmp_path, lambda: engine, lambda: models)) as client:
        session_id = client.post("/api/sessions", data={"model": "countgd-plusplus"}).json()["id"]
        rejected = client.post(f"/api/sessions/{session_id}/batches",
                               data={"image_boxes": json.dumps([boxes])},
                               files=[("images", upload("target.png", png("blue")))])
        assert rejected.status_code == 422
        assert "exactly 10 distinct" in rejected.json()["detail"]
        assert engine.calls == []
        assert client.get(f"/api/sessions/{session_id}").json()["batches"] == []
        assert list((tmp_path / session_id / "files").iterdir()) == []


def test_legacy_countgd_session_keeps_shared_reference_batches(tmp_path):
    engine = FakeEngine()
    models = [{"id": "countgd-plusplus", "available": True, "reference_mode": "per_image",
               "required_reference_boxes": 10}]
    app = create_app(tmp_path, lambda: engine, lambda: models)
    session_id = "legacy-countgd-session"
    files_dir = tmp_path / session_id / "files"
    files_dir.mkdir(parents=True)
    (files_dir / "reference-old.jpg").write_bytes(png("red"))
    app.state.store.create_session(session_id, "rings", "countgd-plusplus", .25, [{
        "id": "legacy-ref", "filename": "reference-old.jpg", "width": 16, "height": 12,
        "boxes": [[1, 1, 8, 8]],
    }])
    with TestClient(app) as client:
        saved = client.get(f"/api/sessions/{session_id}").json()
        assert saved["reference_mode"] == "shared"
        assert saved["required_reference_boxes"] is None
        first = new_batch(client, session_id, ("blue.png", png("blue")))
        assert first.status_code == 202, first.text
        assert settled(client, session_id, first.json()["id"])[1]["status"] == "complete"
        second = new_batch(client, session_id, ("green.png", png("green")))
        assert second.status_code == 202, second.text
        assert settled(client, session_id, second.json()["id"])[1]["status"] == "complete"
        assert engine.reference_modes == ["shared", "shared"]
        assert all(call[1][0]["boxes"] == [[1, 1, 8, 8]] for call in engine.calls)
        assert all(call[1][0]["path"] == str(files_dir / "reference-old.jpg") for call in engine.calls)


@pytest.mark.parametrize("image_boxes", [
    None, "", "invalid", "null", "{}", "[]", "[[]]", "[[1,2,3,4]]",
    "[[[true,1,8,8]]]", "[[[NaN,1,8,8]]]", "[[[Infinity,1,8,8]]]",
    "[[[-1,1,8,8]]]", "[[[1,1,17,8]]]", "[[[8,1,1,8]]]",
    json.dumps([[[1, 1, 10 ** 400, 8]]]),
    json.dumps([[[1, 1, 8, 8]] * 33]),
])
def test_sam3_rejects_bad_target_boxes_without_files_or_queue(tmp_path, image_boxes):
    models = [{"id": "sam3", "available": True, "reference_mode": "per_image"}]
    with TestClient(create_app(tmp_path, FakeEngine, lambda: models)) as client:
        session_id = client.post("/api/sessions", data={"model": "sam3"}).json()["id"]
        data = {} if image_boxes is None else {"image_boxes": image_boxes}
        response = client.post(f"/api/sessions/{session_id}/batches", data=data,
                               files=[("images", upload("target.png", png("blue")))])
        assert response.status_code == 422, response.text
        assert client.get(f"/api/sessions/{session_id}").json()["batches"] == []
        assert list((tmp_path / session_id / "files").iterdir()) == []
        valid = client.post(f"/api/sessions/{session_id}/batches",
                            data={"image_boxes": "[[[1,1,8,8]]]"},
                            files=[("images", upload("target.png", png("blue")))])
        assert valid.status_code == 202, valid.text
        assert settled(client, session_id, valid.json()["id"])[1]["status"] == "complete"


def test_sam3_boxes_use_exif_normalized_target_dimensions(tmp_path):
    models = [{"id": "sam3", "available": True, "reference_mode": "per_image"}]
    engine = FakeEngine()
    exif = Image.Exif()
    exif[274] = 6
    source = io.BytesIO()
    Image.new("RGB", (10, 6), "red").save(source, format="JPEG", exif=exif)
    with TestClient(create_app(tmp_path, lambda: engine, lambda: models)) as client:
        session_id = client.post("/api/sessions", data={"model": "sam3"}).json()["id"]
        endpoint = f"/api/sessions/{session_id}/batches"
        rejected = client.post(endpoint, data={"image_boxes": "[[[1,1,9,5]]]"},
                               files=[("images", upload("rotated.jpg", source.getvalue()))])
        assert rejected.status_code == 422
        accepted = client.post(endpoint, data={"image_boxes": "[[[1,1,5,9]]]"},
                               files=[("images", upload("rotated.jpg", source.getvalue()))])
        assert accepted.status_code == 202, accepted.text
        assert settled(client, session_id, accepted.json()["id"])[1]["status"] == "complete"
        with Image.open(engine.calls[0][2][0]) as normalized:
            assert normalized.size == (6, 10)


@pytest.mark.parametrize("model,reference_mode", [
    ("yoloe-26s-seg", None), ("sam3count", "none"),
])
def test_other_models_reject_per_image_boxes(tmp_path, model, reference_mode):
    models = MODELS + [{"id": "sam3count", "available": True, "reference_mode": "none"}]
    with TestClient(create_app(tmp_path, FakeEngine, lambda: models)) as client:
        if reference_mode == "none":
            session_id = client.post("/api/sessions", data={"model": model,
                "label": "rings"}).json()["id"]
        else:
            session_id = new_session(client).json()["id"]
        rejected = client.post(f"/api/sessions/{session_id}/batches",
                               data={"image_boxes": "[[[1,1,8,8]]]"},
                               files=[("images", upload("target.png", png("blue")))])
        assert rejected.status_code == 422
        assert client.get(f"/api/sessions/{session_id}").json()["batches"] == []
        assert len(list((tmp_path / session_id / "files").iterdir())) == (0 if reference_mode else 1)


def test_reference_boxes_normalizes_exif_and_cleans_temp_file(tmp_path):
    engine = FakeEngine()
    app = create_app(tmp_path, engine_factory=lambda: engine, model_lister=lambda: MODELS)
    exif = Image.Exif()
    exif[274] = 6
    source = io.BytesIO()
    Image.new("RGB", (10, 6), "red").save(source, format="JPEG", exif=exif)
    with TestClient(app) as client:
        response = proposal_request(client, "photo.jpg", source.getvalue(), "  red rings  ")
    assert response.status_code == 200, response.text
    assert response.json() == {"width": 6, "height": 10, "boxes": [[1, 1, 5, 9]],
                               "model": "fastsam-s", "elapsed_ms": 3.5, "warning": None}
    assert len(engine.suggest_calls) == 1
    assert engine.suggest_labels == ["red rings"]
    assert not Path(engine.suggest_calls[0]).exists()
    assert list(tmp_path.iterdir()) == [tmp_path / "sessions.sqlite3"]


@pytest.mark.parametrize("proposal", [
    {"width": 16, "height": 12, "boxes": [[0, 0, 17, 2]],
     "model": "fastsam-s", "elapsed_ms": 1, "warning": None},
    {"width": 16, "height": 12, "boxes": [[0, 0, float("nan"), 2]],
     "model": "fastsam-s", "elapsed_ms": 1, "warning": None},
    {"width": 12, "height": 16, "boxes": [],
     "model": "fastsam-s", "elapsed_ms": 1, "warning": None},
    {"width": 16, "height": 12, "boxes": [[1, 1, 2, 2]],
     "model": "countgd-plusplus", "elapsed_ms": 1, "warning": None},
])
def test_reference_boxes_rejects_invalid_detector_output(tmp_path, proposal):
    engine = FakeEngine()
    engine.proposal = proposal
    app = create_app(tmp_path, engine_factory=lambda: engine, model_lister=lambda: MODELS)
    with TestClient(app) as client:
        response = proposal_request(client, "photo.png", png("red"))
    assert response.status_code == 500
    assert response.json() == {"detail": "Automatic box suggestions failed"}
    assert not Path(engine.suggest_calls[0]).exists()


def test_reference_boxes_bad_image_limits_and_model_failure(tmp_path, monkeypatch):
    from app.reference import ReferenceModelUnavailable

    engine = FakeEngine()
    app = create_app(tmp_path, engine_factory=lambda: engine, model_lister=lambda: MODELS)
    with TestClient(app) as client:
        assert proposal_request(client, "bad.png", b"bad").status_code == 422
        engine.suggest_error = ReferenceModelUnavailable("secret checkpoint path")
        response = proposal_request(client, "photo.png", png("red"))
        assert response.status_code == 503
        assert "secret" not in response.text
        assert not Path(engine.suggest_calls[-1]).exists()
        engine.suggest_error = RuntimeError("private stack detail")
        response = proposal_request(client, "photo.png", png("red"))
        assert response.status_code == 500
        assert "private" not in response.text
        assert not Path(engine.suggest_calls[-1]).exists()
        engine.suggest_error = None
        assert proposal_request(client, "photo.png", png("red")).status_code == 200
        monkeypatch.setattr(main_module, "MAX_FILE_BYTES", 10)
        assert proposal_request(client, "large.png", png("red")).status_code == 413
        monkeypatch.setattr(main_module, "MAX_FILE_BYTES", 20 * 1024 * 1024)
        monkeypatch.setattr(main_module, "MAX_PIXELS", 100)
        assert proposal_request(client, "wide.png", png("red")).status_code == 422


@pytest.mark.parametrize("label", [None, "", "  ", "  rings  "])
def test_reference_boxes_accepts_optional_description(tmp_path, label):
    engine = FakeEngine()
    app = create_app(tmp_path, engine_factory=lambda: engine, model_lister=lambda: MODELS)
    with TestClient(app) as client:
        response = proposal_request(client, "photo.png", png("red"), label)
    assert response.status_code == 200
    assert response.json()["model"] == "fastsam-s"
    assert engine.suggest_labels == [label.strip() if label is not None else ""]


@pytest.mark.parametrize("label", ["bad\nlabel", "x" * 81])
def test_reference_boxes_rejects_invalid_label(tmp_path, label):
    engine = FakeEngine()
    app = create_app(tmp_path, engine_factory=lambda: engine, model_lister=lambda: MODELS)
    with TestClient(app) as client:
        assert proposal_request(client, "photo.png", png("red"), label).status_code == 422
    assert engine.suggest_calls == []


def test_reference_boxes_share_counting_worker_and_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "MAX_QUEUE", 2)
    started = threading.Event()
    release = threading.Event()

    class BlockingEngine(FakeEngine):
        def suggest_reference_boxes(self, image_path, label):
            started.set()
            assert release.wait(5)
            return super().suggest_reference_boxes(image_path, label)

    engine = BlockingEngine()
    app = create_app(tmp_path, engine_factory=lambda: engine, model_lister=lambda: MODELS)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as requests:
        pending = requests.submit(proposal_request, client, "photo.png", png("red"))
        assert started.wait(5)
        session_id = new_session(client).json()["id"]
        batch = new_batch(client, session_id, ("target.png", png("blue")))
        assert batch.status_code == 202
        assert proposal_request(client, "other.png", png("blue")).status_code == 429
        assert not engine.calls
        release.set()
        assert pending.result(timeout=5).status_code == 200
        _, done = settled(client, session_id, batch.json()["id"])
        assert done["status"] == "complete"


def test_cancelled_reference_request_keeps_temp_and_slot_until_worker_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "MAX_QUEUE", 1)
    started = threading.Event()
    release = threading.Event()

    class BlockingEngine(FakeEngine):
        def suggest_reference_boxes(self, image_path, label):
            self.suggest_calls.append(image_path)
            self.suggest_labels.append(label)
            started.set()
            assert release.wait(5)
            assert Path(image_path).exists()
            return {"width": 16, "height": 12, "boxes": [], "model": "fastsam-s",
                    "elapsed_ms": 1, "warning": "No objects found"}

    engine = BlockingEngine()
    app = create_app(tmp_path, engine_factory=lambda: engine, model_lister=lambda: MODELS)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            pending = asyncio.create_task(proposal_request(client, "photo.png", png("red")))
            assert await asyncio.to_thread(started.wait, 5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert Path(engine.suggest_calls[0]).exists()
            queued = await proposal_request(client, "other.png", png("blue"))
            assert queued.status_code == 429
            release.set()
            deadline = time.monotonic() + 5
            while Path(engine.suggest_calls[0]).exists() and time.monotonic() < deadline:
                await asyncio.sleep(.01)
            assert not Path(engine.suggest_calls[0]).exists()
            retry = await proposal_request(client, "again.png", png("blue"))
            assert retry.status_code == 200
            assert retry.json()["boxes"] == []
            assert retry.json()["warning"] == "No objects found"

    asyncio.run(exercise())


def test_cancel_during_reference_normalization_keeps_worker_owned_temp(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "MAX_QUEUE", 1)
    original = main_module._normalize_image
    started = threading.Event()
    release = threading.Event()
    paths = []

    def blocked_normalize(data, destination):
        paths.append(destination)
        started.set()
        assert release.wait(5)
        assert destination.parent.exists()
        return original(data, destination)

    monkeypatch.setattr(main_module, "_normalize_image", blocked_normalize)
    engine = FakeEngine()
    app = create_app(tmp_path, engine_factory=lambda: engine, model_lister=lambda: MODELS)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            pending = asyncio.create_task(proposal_request(client, "photo.png", png("red")))
            assert await asyncio.to_thread(started.wait, 5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert paths[0].parent.exists()
            assert (await proposal_request(client, "next.png", png("blue"))).status_code == 429
            release.set()
            deadline = time.monotonic() + 5
            while paths[0].parent.exists() and time.monotonic() < deadline:
                await asyncio.sleep(.01)
            assert not paths[0].parent.exists()
            while time.monotonic() < deadline:
                retry = await proposal_request(client, "next.png", png("blue"))
                if retry.status_code != 429:
                    break
                await asyncio.sleep(.01)
            assert retry.status_code == 200
            assert len(engine.suggest_calls) == 2

    asyncio.run(exercise())


def test_count_accumulates_exports_and_finishes(tmp_path):
    engine = FakeEngine()
    engine.warning = "Detection cap reached; count may be low."
    app = create_app(tmp_path, engine_factory=lambda: engine, model_lister=lambda: MODELS)
    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        catalog = client.get("/api/models").json()
        assert catalog["default_model"] == "yoloe-26s-seg"
        assert catalog["models"] == MODELS

        response = new_session(client, name="=danger")
        assert response.status_code == 201, response.text
        session = response.json()
        session_id = session["id"]
        assert session["status"] == "active"
        assert session["total_count"] == session["image_count"] == 0
        assert client.get(session["references"][0]["url"]).status_code == 200

        first = new_batch(client, session_id, ("=formula.png", png("blue")))
        assert first.status_code == 202, first.text
        session, batch = settled(client, session_id, first.json()["id"])
        assert batch["status"] == "complete" and batch["total_count"] == 1
        assert session["total_count"] == session["image_count"] == 1
        result = batch["results"][0]
        assert result["filename"] == "=formula.png" and result["count"] == 1
        assert result["warning"] == engine.warning
        assert result["processing"] is None
        assert client.get(result["annotated_url"]).status_code == 200
        assert client.get(result["original_url"]).status_code == 200

        second = new_batch(client, session_id, ("other.png", png("green")))
        assert second.status_code == 202, second.text
        session, batch = settled(client, session_id, second.json()["id"])
        assert session["total_count"] == session["image_count"] == 2
        assert len(engine.calls) == 2
        assert engine.labels == ["=danger", "=danger"]
        assert engine.calls[0][1][0]["boxes"] == [[1, 1, 9, 9]]

        rows = list(csv.reader(io.StringIO(client.get(f"/api/sessions/{session_id}/export.csv").text)))
        assert rows[0] == ["session", "batch", "filename", "count", "running_total", "warning"]
        assert rows[1][0] == "'=danger" and rows[1][2] == "'=formula.png"
        assert [row[4] for row in rows[1:]] == ["1", "2"]
        assert [row[5] for row in rows[1:]] == [engine.warning, engine.warning]
        assert client.post(f"/api/sessions/{session_id}/finish").json()["status"] == "finished"
        assert new_batch(client, session_id, ("third.png", png("yellow"))).status_code == 409
        assert client.post(f"/api/sessions/{session_id}/finish").json()["status"] == "finished"


def test_processing_metadata_migrates_and_survives_restart(tmp_path):
    # An existing database may have neither optional result column.
    with sqlite3.connect(tmp_path / "sessions.sqlite3") as db:
        db.execute("""CREATE TABLE images (
            id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
            filename TEXT NOT NULL, original_filename TEXT NOT NULL, sha256 TEXT NOT NULL,
            count INTEGER, detections TEXT, annotated_filename TEXT,
            width INTEGER NOT NULL, height INTEGER NOT NULL, elapsed_ms REAL
        )""")
    engine = FakeEngine()
    app = create_app(tmp_path, engine_factory=lambda: engine, model_lister=lambda: MODELS)
    with TestClient(app) as client:
        session_id = new_session(client).json()["id"]
        old = new_batch(client, session_id, ("old.png", png("blue"))).json()
        session, _ = settled(client, session_id, old["id"])
        assert session["batches"][0]["results"][0]["processing"] is None

        engine.processing = {"tiles": 12, "tile_size": 1024, "duplicates_removed": 3}
        tiled = new_batch(client, session_id, ("tiled.png", png("green"))).json()
        session, batch = settled(client, session_id, tiled["id"])
        assert batch["results"][0]["processing"] == engine.processing
        assert session["total_count"] == 2
        assert session["image_count"] == 2

    restarted = create_app(tmp_path, engine_factory=FakeEngine, model_lister=lambda: MODELS)
    with TestClient(restarted) as client:
        session = client.get(f"/api/sessions/{session_id}").json()
        assert [batch["results"][0]["processing"] for batch in session["batches"]] == [
            None, {"tiles": 12, "tile_size": 1024, "duplicates_removed": 3}]
        assert session["total_count"] == session["image_count"] == 2


@pytest.mark.parametrize("processing", [
    None,
    [],
    {"tiles": True, "tile_size": 1024, "duplicates_removed": 0},
    {"tiles": 1.0, "tile_size": 1024, "duplicates_removed": 0},
    {"tiles": 0, "tile_size": 1024, "duplicates_removed": 0},
    {"tiles": 4097, "tile_size": 1024, "duplicates_removed": 0},
    {"tiles": 1, "tile_size": float("nan"), "duplicates_removed": 0},
    {"tiles": 1, "tile_size": 16385, "duplicates_removed": 0},
    {"tiles": 1, "tile_size": 1024, "duplicates_removed": -1},
    {"tiles": 1, "tile_size": 1024, "duplicates_removed": 1_000_001},
    {"tiles": 1, "tile_size": 1024},
    {"tiles": 1, "tile_size": 1024, "duplicates_removed": 0, "extra": 1},
])
def test_processing_metadata_validation(processing):
    with pytest.raises(ValueError, match="invalid image processing metadata"):
        main_module._validate_processing(processing)


def test_invalid_processing_fails_batch_without_changing_count(tmp_path):
    engine = FakeEngine()
    engine.processing = {"tiles": False, "tile_size": 1024, "duplicates_removed": 0}
    app = create_app(tmp_path, engine_factory=lambda: engine, model_lister=lambda: MODELS)
    with TestClient(app) as client:
        session_id = new_session(client).json()["id"]
        image = png("blue")
        bad = new_batch(client, session_id, ("target.png", image)).json()
        session, batch = settled(client, session_id, bad["id"])
        assert batch["status"] == "failed"
        assert "invalid image processing metadata" in batch["error"]
        assert session["total_count"] == session["image_count"] == 0

        engine.processing = {"tiles": 4, "tile_size": 512, "duplicates_removed": 0}
        retry = new_batch(client, session_id, ("target.png", image)).json()
        session, batch = settled(client, session_id, retry["id"])
        assert batch["status"] == "complete"
        assert batch["results"][0]["processing"] == engine.processing
        assert session["total_count"] == session["image_count"] == 1


def test_export_warns_for_historical_zero_without_saved_warning(tmp_path):
    engine = FakeEngine()
    engine.result_count = 0
    engine.warning = "=Review this result"
    app = create_app(tmp_path, engine_factory=lambda: engine, model_lister=lambda: MODELS)
    with TestClient(app) as client:
        session_id = new_session(client).json()["id"]
        first = new_batch(client, session_id, ("first.png", png("blue"))).json()
        _, first_batch = settled(client, session_id, first["id"])
        assert first_batch["results"][0]["warning"] == engine.warning

        engine.warning = None
        second = new_batch(client, session_id, ("second.png", png("green"))).json()
        _, second_batch = settled(client, session_id, second["id"])
        assert second_batch["results"][0]["warning"] is None

        rows = list(csv.reader(io.StringIO(client.get(f"/api/sessions/{session_id}/export.csv").text)))
        assert [row[3:5] for row in rows[1:]] == [["0", "0"], ["0", "0"]]
        assert rows[1][5] == "'=Review this result"
        assert rows[2][5] == (
            "No matches detected. Check the reference or use a model trained for this object."
        )


def test_validation_duplicates_and_file_scope(tmp_path):
    app = create_app(tmp_path, engine_factory=FakeEngine, model_lister=lambda: MODELS)
    with TestClient(app) as client:
        assert new_session(client, boxes=[[[1, 1, 100, 9]]]).status_code == 422
        assert new_session(client, reference=b"not an image").status_code == 422
        assert client.post("/api/sessions", data={"label": "x", "model": "missing",
            "reference_boxes": "[[[1,1,9,9]]]"},
            files=[("references", upload("ref.png", png("red")))]).status_code == 422
        session_id = new_session(client).json()["id"]
        other_id = new_session(client).json()["id"]
        same = png("blue")
        assert new_batch(client, session_id, ("one.png", same), ("two.png", same)).status_code == 409
        assert client.get(f"/api/sessions/{session_id}").json()["batches"] == []
        assert new_batch(client, session_id, ("fake.png", b"bad")).status_code == 422
        batch = new_batch(client, session_id, ("one.png", same))
        assert batch.status_code == 202
        session, result = settled(client, session_id, batch.json()["id"])
        assert result["status"] == "complete"
        assert new_batch(client, session_id, ("renamed.png", same)).status_code == 409
        filename = result["results"][0]["original_url"].rsplit("/", 1)[-1]
        assert client.get(f"/api/sessions/{other_id}/files/{filename}").status_code == 404
        assert client.get(f"/api/sessions/{session_id}/files/../sessions.sqlite3").status_code == 404


def test_failed_batch_can_retry_and_restart_marks_interrupted_failed(tmp_path):
    engine = FakeEngine()
    engine.fail_next = True
    app = create_app(tmp_path, engine_factory=lambda: engine, model_lister=lambda: MODELS)
    with TestClient(app) as client:
        session_id = new_session(client).json()["id"]
        original = png("blue")
        batch = new_batch(client, session_id, ("image.png", original)).json()
        session, failed = settled(client, session_id, batch["id"])
        assert failed["status"] == "failed" and "fake inference failed" in failed["error"]
        assert session["total_count"] == session["image_count"] == 0
        retry = new_batch(client, session_id, ("image.png", original))
        assert retry.status_code == 202
        session, completed = settled(client, session_id, retry.json()["id"])
        assert completed["status"] == "complete" and session["total_count"] == 1
        app.state.store.set_batch_status(retry.json()["id"], "running")

    restarted = create_app(tmp_path, engine_factory=FakeEngine, model_lister=lambda: MODELS)
    with TestClient(restarted) as client:
        session = client.get(f"/api/sessions/{session_id}").json()
        recovered = next(b for b in session["batches"] if b["id"] == retry.json()["id"])
        assert recovered["status"] == "failed"
        assert "Server restarted" in recovered["error"]
        assert session["total_count"] == 0


def test_body_limit_checks_content_length_and_stream(monkeypatch):
    monkeypatch.setattr(main_module, "MAX_BODY_BYTES", 128)
    app = FastAPI()
    app.add_middleware(main_module.BodyLimitMiddleware)

    @app.post("/consume")
    async def consume(request: Request):
        await request.body()
        return {"ok": True}

    with TestClient(app) as client:
        assert client.post("/consume", content=b"x" * 129).status_code == 413
        streamed = client.post("/consume", content=iter([b"x" * 80, b"x" * 80]))
        assert streamed.status_code == 413


def test_reference_proposal_api_accepts_more_than_32_and_validates_every_box(tmp_path):
    engine = FakeEngine()
    boxes = [[1 + (i % 10) * 8, 1 + (i // 10) * 8,
              5 + (i % 10) * 8, 5 + (i // 10) * 8] for i in range(40)]
    engine.proposal = {"width": 100, "height": 80, "boxes": boxes,
                       "model": "fastsam-s", "elapsed_ms": 1, "warning": None}
    with TestClient(create_app(tmp_path, engine_factory=lambda: engine, model_lister=lambda: MODELS)) as client:
        source = io.BytesIO()
        Image.new("RGB", (100, 80), "gray").save(source, format="PNG")
        response = proposal_request(client, "photo.png", source.getvalue())
        assert response.status_code == 200, response.text
        assert response.json()["boxes"] == boxes
        engine.proposal["boxes"] = boxes[:32] + [[1, 1, 101, 10]]
        rejected = proposal_request(client, "photo.png", source.getvalue())
        assert rejected.status_code == 500
