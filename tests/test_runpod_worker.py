"""RunPod boundary checks; no SDK, checkpoint, or CUDA is needed."""

import base64
import io
import math
from pathlib import Path

from PIL import Image
import pytest

from app import runpod_worker as worker
from app.models import COUNTGD_MODEL_ID


@pytest.fixture(autouse=True)
def reset_worker(monkeypatch):
    monkeypatch.setattr(worker, "_engine", None)
    monkeypatch.setattr(worker, "_first_handler", True)
    monkeypatch.setattr(worker, "_startup_metrics", {})
    monkeypatch.setattr(worker, "_start_gpu_metrics", lambda: (None, None))


def image_base64(fmt="PNG", size=(8, 6)):
    image = io.BytesIO()
    Image.new("RGB", size, "white").save(image, format=fmt)
    return base64.b64encode(image.getvalue()).decode("ascii")


def event(**overrides):
    return {"input": {"action": "count", "image_base64": image_base64(),
                      "boxes": [[1, 1, 4, 4]], **overrides}}


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_count_passes_exact_uploaded_bytes_and_omits_temporary_annotation(monkeypatch, fmt):
    calls = []
    encoded = image_base64(fmt)
    payload = event(image_base64=encoded, label="  bolts  ", confidence=.9)

    class FakeEngine:
        def count(self, model_id, references, image_paths, output_dir, confidence, label, reference_mode, annotate):
            calls.append((model_id, references, image_paths, output_dir, confidence, label, reference_mode, annotate))
            assert references[0]["path"] == image_paths[0]
            assert Path(image_paths[0]).read_bytes() == base64.b64decode(encoded)
            assert Image.open(image_paths[0]).format == fmt
            return [{"count": 2, "detections": [], "annotated_path": f"{output_dir}/annotation.jpg"}]

    monkeypatch.setattr(worker, "_engine", FakeEngine())
    monkeypatch.setenv("RUNPOD_WORKER_ID", "worker-a")
    first = worker.handler(payload)
    second = worker.handler(payload)

    assert first["result"] == {"count": 2, "detections": []}
    assert first["metrics"]["worker_id"] == "worker-a"
    assert first["metrics"]["cold_start"] is True
    assert second["metrics"]["cold_start"] is False
    assert first["metrics"]["handler_ms"] >= 0
    assert calls[0][0] == COUNTGD_MODEL_ID
    assert calls[0][1][0]["boxes"] == [[1, 1, 4, 4]]
    assert calls[0][4:] == (.9, "bolts", "per_image", False)


def test_count_clips_edge_detections_without_changing_count(monkeypatch):
    class FakeEngine:
        def count(self, *args, **kwargs):
            return [{"count": 3, "detections": [
                {"box": [-2.0, 1.0, 4.0, 7.0], "confidence": .9, "index": 1},
                {"box": [6.0, 4.0, 9.0, 8.0], "confidence": .8, "index": 2},
                {"box": [9.0, -5.0, 20.0, -1.0], "confidence": .7, "index": 3},
            ], "annotated_path": "temporary.jpg"}]

    monkeypatch.setattr(worker, "_engine", FakeEngine())
    result = worker.handler(event())["result"]
    assert result["count"] == 3
    assert [detection["box"] for detection in result["detections"]] == [
        [0.0, 1.0, 4.0, 6.0], [6.0, 4.0, 8.0, 6.0], [7.0, 0.0, 8.0, 1.0],
    ]
    assert [detection["index"] for detection in result["detections"]] == [1, 2, 3]


def test_suggest_returns_existing_proposal_shape(monkeypatch):
    class FakeEngine:
        def suggest_reference_boxes(self, image_path, label):
            assert Image.open(image_path).size == (8, 6)
            assert label == "nuts"
            return {"model": "fastsam-s", "boxes": [[1, 1, 3, 3]], "width": 8, "height": 6}

    monkeypatch.setattr(worker, "_engine", FakeEngine())
    result = worker.handler({"input": {"action": "suggest", "image_base64": image_base64(), "label": "nuts"}})
    assert result["result"] == {"model": "fastsam-s", "boxes": [[1, 1, 3, 3]], "width": 8, "height": 6}


@pytest.mark.parametrize("overrides", [
    {"boxes": []}, {"boxes": [[1, 1, 1, 4]]}, {"boxes": [[1, 1, 9, 4]]},
    {"boxes": [[True, 1, 4, 4]]}, {"boxes": [[1, 1, math.nan, 4]]},
    {"boxes": [[1, 1, 4, 4], [1, 1, 4, 4]]},
    {"boxes": [[1, 1, 4, 4]] * 33},
    {"confidence": math.inf}, {"confidence": True},
    {"image_base64": "http://example.com/image.jpg"},
    {"image_base64": image_base64("GIF")},
    {"image_base64": "%%%"},
    {"image_url": "https://example.com/image.jpg"},
    {"label": "bad\nlabel"},
    {"action": []},
])
def test_invalid_jobs_fail_before_engine_or_cuda(monkeypatch, overrides):
    monkeypatch.setattr(worker, "_get_engine", lambda: pytest.fail("engine allocated for invalid input"))
    monkeypatch.setattr(worker, "_start_gpu_metrics", lambda: pytest.fail("CUDA initialized for invalid input"))
    with pytest.raises(ValueError):
        worker.handler(event(**overrides))


def test_rejects_decoded_size_and_pixel_limits_before_engine(monkeypatch):
    monkeypatch.setattr(worker, "_get_engine", lambda: pytest.fail("engine allocated for invalid input"))
    monkeypatch.setattr(worker, "MAX_IMAGE_BYTES", 2)
    with pytest.raises(ValueError, match="20 MiB|20 MiB or smaller"):
        worker.handler(event())
    monkeypatch.setattr(worker, "MAX_IMAGE_BYTES", 20 * 1024 * 1024)
    monkeypatch.setattr(worker, "MAX_PIXELS", 47)
    with pytest.raises(ValueError, match="30 megapixels"):
        worker.handler(event())


def test_invalid_job_does_not_consume_cold_start(monkeypatch):
    class FakeEngine:
        def count(self, *args, **kwargs):
            return [{"count": 0, "detections": []}]

    monkeypatch.setattr(worker, "_engine", FakeEngine())
    with pytest.raises(ValueError):
        worker.handler(event(boxes=[[0, 0, 99, 99]]))
    assert worker.handler(event())["metrics"]["cold_start"] is True
    assert worker.handler(event())["metrics"]["cold_start"] is False
