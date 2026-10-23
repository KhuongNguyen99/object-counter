"""Local API routing checks without RunPod, model assets, or CUDA."""

import io
from pathlib import Path
import time

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app import main as main_module
from app import runpod_client
from app.main import create_app
from app.models import COUNTGD_MODEL_ID


def photo():
    output = io.BytesIO()
    Image.new("RGB", (16, 12), "blue").save(output, format="PNG")
    return output.getvalue()


def test_runpod_backend_routes_catalog_suggestions_and_count(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "runpod")
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "endpoint-123")
    monkeypatch.setattr(main_module, "_device", lambda _: (_ for _ in ()).throw(
        AssertionError("Remote catalog must not inspect local CUDA")))
    engines = []

    class FakeRemoteEngine:
        def __init__(self):
            self.count_calls = []
            engines.append(self)

        def close(self):
            pass

        def suggest_reference_boxes(self, image_path, label):
            assert label == "nuts"
            assert Image.open(image_path).size == (16, 12)
            return {"width": 16, "height": 12, "boxes": [[1, 1, 8, 8]],
                    "model": "fastsam-s", "elapsed_ms": 1.5, "warning": None}

        def count(self, model_id, references, image_paths, output_dir, confidence, label,
                  reference_mode):
            self.count_calls.append((model_id, references, image_paths, reference_mode))
            assert references[0]["path"] == image_paths[0]
            annotation = Path(output_dir) / "annotation.jpg"
            Image.open(image_paths[0]).save(annotation)
            return [{"count": 1, "detections": [{"index": 1, "box": [1, 1, 8, 8],
                                                   "confidence": .9}],
                     "annotated_path": str(annotation), "width": 16, "height": 12,
                     "elapsed_ms": 2.5, "warning": None}]

    monkeypatch.setattr(runpod_client, "RunPodInferenceEngine", FakeRemoteEngine)
    with TestClient(create_app(tmp_path)) as client:
        catalog = client.get("/api/models").json()
        assert catalog["device"] == "remote"
        assert catalog["models"][0]["id"] == COUNTGD_MODEL_ID
        assert catalog["models"][0]["available"] is True
        proposal = client.post("/api/reference-boxes", data={"label": "nuts"},
                               files={"image": ("image.png", photo(), "image/png")})
        assert proposal.status_code == 200, proposal.text
        assert proposal.json()["boxes"] == [[1, 1, 8, 8]]
        session = client.post("/api/sessions", data={"model": COUNTGD_MODEL_ID})
        assert session.status_code == 201, session.text
        session_id = session.json()["id"]
        batch = client.post(f"/api/sessions/{session_id}/batches",
                            data={"image_boxes": "[[[1,1,8,8]]]"},
                            files={"images": ("image.png", photo(), "image/png")})
        assert batch.status_code == 202, batch.text
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            saved = client.get(f"/api/sessions/{session_id}").json()
            status = saved["batches"][0]["status"]
            if status in {"complete", "failed"}:
                break
            time.sleep(.01)
        assert status == "complete", saved
        assert saved["total_count"] == 1
    assert len(engines) == 1
    assert engines[0].count_calls[0][0] == COUNTGD_MODEL_ID
    assert engines[0].count_calls[0][3] == "per_image"


def test_remote_catalog_marks_missing_credentials_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "runpod")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "endpoint-123")
    with TestClient(create_app(tmp_path)) as client:
        catalog = client.get("/api/models").json()
        assert catalog["device"] == "remote"
        assert catalog["models"][0]["available"] is False
        assert client.post("/api/sessions", data={"model": COUNTGD_MODEL_ID}).status_code == 422


def test_explicit_catalog_and_factory_still_override_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "runpod")
    monkeypatch.setattr(runpod_client, "remote_model_catalog", lambda: 1 / 0)
    monkeypatch.setattr(runpod_client, "RunPodInferenceEngine", lambda: 1 / 0)
    custom_catalog = lambda: {"models": [], "default_model": COUNTGD_MODEL_ID,
                              "device": "injected"}
    with TestClient(create_app(tmp_path, engine_factory=lambda: object(),
                               model_lister=custom_catalog)) as client:
        assert client.get("/api/models").json()["device"] == "injected"


def test_unknown_backend_is_rejected_even_with_injected_dependencies(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "other")
    with pytest.raises(ValueError, match="INFERENCE_BACKEND must be local or runpod"):
        create_app(tmp_path, engine_factory=lambda: object(), model_lister=lambda: [])
