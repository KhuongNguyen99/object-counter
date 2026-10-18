"""Unit checks plus opt-in real-checkpoint smoke tests: RUN_MODEL_TESTS=1 pytest -q."""

import os
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.inference import InferenceEngine, _reference_boxes
from app.models import MODEL_FILENAMES, checkpoint_path, list_models, validate_checkpoint


def test_countgd_assets_reject_missing_or_modified_checkpoint(tmp_path, monkeypatch):
    from app import models
    contents = b"trusted official checkpoint"
    monkeypatch.setattr(models, "COUNTGD_ASSETS", {
        "countgd_plusplus.pth": (len(contents), hashlib.sha256(contents).hexdigest()),
    })
    with pytest.raises(FileNotFoundError, match="missing"):
        models.validate_countgd_assets(tmp_path)
    checkpoint = tmp_path / "countgd_plusplus.pth"
    checkpoint.write_bytes(contents)
    models.validate_countgd_assets(tmp_path)
    checkpoint.write_bytes(b"x" * len(contents))
    with pytest.raises(ValueError, match="SHA-256"):
        models.validate_countgd_assets(tmp_path)


def test_reference_boxes_reject_invalid_coordinates():
    for boxes in ([], [[1, 2, 1, 5]], [[-1, 0, 4, 4]], [[0, 0, 101, 3]], [[0, 0, float("nan"), 4]]):
        with pytest.raises(ValueError):
            _reference_boxes(boxes, 100, 50)
    np.testing.assert_equal(_reference_boxes([[0, 0, 100, 50]], 100, 50), [[0, 0, 100, 50]])


def test_model_catalog_is_latest_family_and_requires_local_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    models = list_models()
    yoloe_models = [model for model in models if model["family"] == "YOLOE-26"]
    assert len(yoloe_models) == 5
    assert {model["size"] for model in yoloe_models} == {"n", "s", "m", "l", "x"}
    assert [model["id"] for model in models if model["family"] == "CountGD++"] == ["countgd-plusplus"]
    assert all(not model["available"] for model in models)
    with pytest.raises(ValueError):
        checkpoint_path("../../untrusted")
    with pytest.raises(FileNotFoundError, match="download_models"):
        InferenceEngine().count("yoloe-26n-seg", [{"path": "unused", "boxes": []}], ["unused"], str(tmp_path))
    fake = tmp_path / MODEL_FILENAMES[0]
    fake.write_text("<html>Not a checkpoint</html>")
    with pytest.raises(ValueError, match="incomplete or invalid"):
        validate_checkpoint(fake)


@pytest.fixture
def fake_yoloe(monkeypatch):
    """Exercise the real count/load loop with a cheap fake YOLOE and visual encoder."""
    import torch
    from app import inference

    state = SimpleNamespace(models=[], encoder_calls=0, prompts=[], predict_fn=lambda source: [])

    class FakeEncoder:
        def __init__(self, overrides):
            self.overrides = overrides

        def setup_model(self, model, verbose):
            assert model is state.models[-1].model

        def set_prompts(self, prompts):
            self.prompts = prompts
            state.prompts.append(prompts)

        def get_vpe(self, image):
            state.encoder_calls += 1
            left = torch.as_tensor(self.prompts["bboxes"][:, 0], dtype=torch.float32)
            return torch.stack((torch.ones_like(left), left + 1), dim=-1).unsqueeze(0)

    class FakeYOLOE:
        def __init__(self, path, task):
            assert task == "segment"
            self.model = SimpleNamespace()
            self.predictor = None
            self.calls = []
            self.embeddings = []
            state.models.append(self)

        def set_classes(self, classes, embeddings):
            assert classes == ["object"]
            self.model.pe = embeddings
            self.embeddings.append(embeddings.clone())

        def predict(self, source, **options):
            assert self.model.pe.shape[1] == 1
            assert options["imgsz"] >= max(source.size)
            assert options["max_det"] >= 1
            self.calls.append((source.copy(), options.copy()))
            records = state.predict_fn(source)
            boxes = torch.tensor([box for box, _ in records], dtype=torch.float32).reshape(-1, 4)
            scores = torch.tensor([score for _, score in records], dtype=torch.float32)
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=boxes, conf=scores))]

    ultralytics = ModuleType("ultralytics")
    models = ModuleType("ultralytics.models")
    yolo = ModuleType("ultralytics.models.yolo")
    yoloe = ModuleType("ultralytics.models.yolo.yoloe")
    ultralytics.YOLOE = FakeYOLOE
    yoloe.YOLOEVPSegPredictor = FakeEncoder
    for name, module in (("ultralytics", ultralytics), ("ultralytics.models", models),
                         ("ultralytics.models.yolo", yolo), ("ultralytics.models.yolo.yoloe", yoloe)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(inference, "validate_checkpoint", lambda path: None)
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("IMGSZ", "64")
    return state


def _fixture_image(path, size):
    image = Image.new("RGB", size)
    for x in range(size[0]):
        for y in range(size[1]):
            image.putpixel((x, y), (x % 256, y % 256, 0))
    image.save(path)
    return str(path)


def test_yoloe_native_crops_merge_translated_duplicates_and_draw_full_source(tmp_path, fake_yoloe):
    source = _fixture_image(tmp_path / "wide.png", (100, 64))
    fake_yoloe.predict_fn = lambda crop: (
        [([47, 10, 57, 20], .8), ([float("nan"), 0, 10, 10], .9)]
        if crop.getpixel((0, 0))[0] == 0 else
        [([11, 10, 21, 20], .9), ([40, 30, 50, 40], .7), ([5, 5, 5, 10], .8)]
    )
    result = InferenceEngine().count("yoloe-26n-seg",
                                     [{"path": source, "boxes": [[2, 2, 12, 12]]}],
                                     [source], str(tmp_path / "output"))[0]
    calls = fake_yoloe.models[0].calls
    assert [(image.size, image.getpixel((0, 0))) for image, _ in calls] == [
        ((64, 64), (0, 0, 0)), ((64, 64), (36, 0, 0))]
    assert result["processing"] == {"tiles": 2, "tile_size": 64, "duplicates_removed": 1}
    assert result["count"] == 2
    assert [item["index"] for item in result["detections"]] == [1, 2]
    np.testing.assert_allclose([item["box"] for item in result["detections"]],
                               [[47, 10, 57, 20], [76, 30, 86, 40]])
    assert "warning" not in result
    assert (result["width"], result["height"]) == (100, 64)
    with Image.open(result["annotated_path"]) as annotation:
        assert annotation.size == (100, 64)


def test_yoloe_prompt_encoded_once_per_batch_and_reset_between_runs(tmp_path, fake_yoloe):
    source = _fixture_image(tmp_path / "wide.png", (100, 64))
    engine = InferenceEngine()
    fake_yoloe.predict_fn = lambda crop: [([10, 10, 20, 20], .9)]
    first = engine.count("yoloe-26s-seg", [{"path": source, "boxes": [[2, 2, 12, 12]]}],
                         [source, source], str(tmp_path / "output"))
    model = fake_yoloe.models[0]
    assert len(first) == 2 and all(output["count"] == 2 for output in first)
    assert fake_yoloe.encoder_calls == 1
    assert len(model.calls) == 4
    assert len(model.embeddings) == 1
    first_prompt = model.embeddings[0].clone()
    model.predictor = object()
    second = engine.count("yoloe-26s-seg", [{"path": source, "boxes": [[30, 2, 40, 12]]}],
                          [source], str(tmp_path / "output"))
    assert second[0]["count"] == 2
    assert len(fake_yoloe.models) == 1
    assert model.predictor is None
    assert fake_yoloe.encoder_calls == 2
    assert len(model.embeddings) == 2
    assert not np.allclose(first_prompt.numpy(), model.embeddings[1].numpy())


def test_reference_proposals_reuse_countgd_core_and_reload_after_yoloe(tmp_path, fake_yoloe, monkeypatch):
    from app import countgd, reference

    source = _fixture_image(tmp_path / "small.png", (32, 32))
    state = SimpleNamespace(proposals=[], countgd=[])

    def fake_propose(core, image_path, label):
        assert core is state.countgd[-1] and image_path == source
        state.proposals.append(label)
        return {"width": 32, "height": 32, "boxes": [[1, 1, 10, 10]],
                "model": "countgd-plusplus", "elapsed_ms": 1, "warning": None}

    class FakeCountGD:
        def __init__(self, directory, device):
            self.device = device
            state.countgd.append(self)

        def count(self, *args):
            return []

    monkeypatch.setattr(reference, "propose_boxes", fake_propose, raising=False)
    monkeypatch.setattr(countgd, "CountGDEngine", FakeCountGD)
    engine = InferenceEngine()
    refs = [{"path": source, "boxes": [[2, 2, 12, 12]]}]
    fake_yoloe.predict_fn = lambda crop: []
    engine.count("yoloe-26n-seg", refs, [source], str(tmp_path / "out"))
    assert engine._model_id == "yoloe-26n-seg" and len(fake_yoloe.models) == 1
    assert engine.suggest_reference_boxes(source, "rings")["boxes"] == [[1, 1, 10, 10]]
    assert engine._model_id == "countgd-plusplus" and len(state.countgd) == 1
    assert state.countgd[0].device == "cpu"
    engine.suggest_reference_boxes(source, "bolts")
    assert len(state.countgd) == 1
    engine.count("countgd-plusplus", refs, [source], str(tmp_path / "out"))
    assert engine._model_id == "countgd-plusplus" and len(state.countgd) == 1
    engine.suggest_reference_boxes(source, "nuts")
    assert len(state.countgd) == 1 and state.proposals == ["rings", "bolts", "nuts"]
    engine.count("yoloe-26n-seg", refs, [source], str(tmp_path / "out"))
    assert engine._model_id == "yoloe-26n-seg" and len(fake_yoloe.models) == 2
    engine.suggest_reference_boxes(source, "washers")
    assert engine._model_id == "countgd-plusplus" and len(state.countgd) == 2


@pytest.mark.parametrize("failure", [FileNotFoundError("missing"), ValueError("corrupt"), ImportError("package")])
def test_reference_model_load_failures_are_unavailable(tmp_path, monkeypatch, failure):
    from app import countgd, reference

    def failing_countgd(*args):
        raise failure

    monkeypatch.setattr(countgd, "CountGDEngine", failing_countgd)
    with pytest.raises(reference.ReferenceModelUnavailable):
        InferenceEngine().suggest_reference_boxes(str(tmp_path / "image.jpg"), "rings")


def test_yoloe_detection_limit_is_per_tile_and_zero_warning_survives(tmp_path, fake_yoloe, monkeypatch):
    source = _fixture_image(tmp_path / "wide.png", (100, 64))
    reference = [{"path": source, "boxes": [[2, 2, 12, 12]]}]
    engine = InferenceEngine()
    fake_yoloe.predict_fn = lambda crop: [([10, 10, 20, 20], .9)]
    monkeypatch.setenv("MAX_DETECTIONS", "2")
    under_cap = engine.count("yoloe-26m-seg", reference, [source], str(tmp_path / "out"))[0]
    assert under_cap["count"] == 2  # Global count reaches cap; neither tile does.
    assert "warning" not in under_cap
    monkeypatch.setenv("MAX_DETECTIONS", "1")
    at_cap = engine.count("yoloe-26m-seg", reference, [source], str(tmp_path / "out"))[0]
    assert at_cap["count"] == 2
    assert "Detection limit reached" in at_cap["warning"]
    fake_yoloe.predict_fn = lambda crop: []
    empty = engine.count("yoloe-26m-seg", reference, [source], str(tmp_path / "out"))[0]
    assert empty["count"] == 0
    assert "No matches detected" in empty["warning"]


def test_yoloe_single_tile_keeps_original_result_shape(tmp_path, fake_yoloe):
    source = _fixture_image(tmp_path / "small.png", (32, 32))
    fake_yoloe.predict_fn = lambda crop: [([20, 20, 25, 25], .8), ([1, 1, 5, 5], .9)]
    result = InferenceEngine().count("yoloe-26l-seg",
                                     [{"path": source, "boxes": [[2, 2, 12, 12]]}],
                                     [source], str(tmp_path / "output"))[0]
    assert "processing" not in result
    assert "warning" not in result
    assert result["count"] == 2
    assert [d["box"] for d in result["detections"]] == [[1, 1, 5, 5], [20, 20, 25, 25]]
    assert [d["index"] for d in result["detections"]] == [1, 2]


@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="Set RUN_MODEL_TESTS=1 to run real local YOLOE models")
def test_real_models_multiple_references_prompt_reset_and_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "ultralytics-settings"))
    test_device = os.environ.get("MODEL_TEST_DEVICE", "cpu")
    monkeypatch.setenv("DEVICE", test_device)
    import ultralytics
    import requests
    import torch
    import urllib.request
    from ultralytics.nn.backends.pytorch import PyTorchBackend

    expected_device = "cpu" if test_device == "cpu" else "cuda"
    forward_phases = set()
    original_forward = PyTorchBackend.forward

    def checked_forward(backend, im, *args, **kwargs):
        # Predictors copy the CPU checkpoint model; inspect the model actually executing.
        parameter = next(backend.model.parameters())
        assert im.device.type == expected_device
        assert parameter.device == im.device
        assert im.dtype == parameter.dtype == torch.float32
        reference = kwargs.get("return_vpe", False)
        prompt = kwargs["vpe"] if reference else backend.model.pe
        assert prompt.device == im.device and prompt.dtype == torch.float32
        forward_phases.add("reference" if reference else "target")
        return original_forward(backend, im, *args, **kwargs)

    monkeypatch.setattr(PyTorchBackend, "forward", checked_forward)

    def no_network(*args, **kwargs):
        raise AssertionError("Inference attempted network access")

    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    image = str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg")
    person = {"path": image, "boxes": [[48, 398, 244, 902]]}
    engine = InferenceEngine()
    for filename in MODEL_FILENAMES:
        forward_phases.clear()
        result = engine.count(Path(filename).stem, [person], [image], str(tmp_path), confidence=0.2)[0]
        assert forward_phases == {"reference", "target"}
        assert result["count"] == len(result["detections"]) >= 1
        assert Path(result["annotated_path"]).is_file()
        assert (result["width"], result["height"]) == (810, 1080)
        assert result["processing"]["tiles"] > 1
        assert result["processing"]["tile_size"] == 640
        assert result["processing"]["duplicates_removed"] >= 0
        assert [item["index"] for item in result["detections"]] == list(range(1, result["count"] + 1))
        assert all(0.2 <= item["confidence"] <= 1 for item in result["detections"])

    single = engine.count("yoloe-26n-seg", [person], [image], str(tmp_path), confidence=0.2)[0]
    duplicated = engine.count("yoloe-26n-seg", [person, person], [image], str(tmp_path), confidence=0.2)[0]
    assert single["count"] == duplicated["count"]
    np.testing.assert_allclose(
        [item["box"] for item in single["detections"]],
        [item["box"] for item in duplicated["detections"]], atol=0.05,
    )
    person_embedding = engine._model.model.pe.detach().clone()
    bus = {"path": image, "boxes": [[22, 231, 805, 757]]}
    engine.count("yoloe-26n-seg", [bus], [image], str(tmp_path), confidence=0.2)
    assert not np.allclose(person_embedding.cpu().numpy(), engine._model.model.pe.cpu().numpy())
    multiple = {"path": image, "boxes": [[48, 398, 244, 902], [221, 405, 345, 858]]}
    result = engine.count("yoloe-26n-seg", [multiple, person], [image, image], str(tmp_path), confidence=0.2)
    assert len(result) == 2 and result[0]["count"] == result[1]["count"]
    assert engine._model.model.pe.shape[1] == 1
    monkeypatch.setenv("MAX_DETECTIONS", "1")
    limited = engine.count("yoloe-26n-seg", [person], [image], str(tmp_path), confidence=0.2)[0]
    assert 1 <= limited["count"] <= limited["processing"]["tiles"]
    assert "Detection limit reached" in limited["warning"]
    empty = engine.count("yoloe-26n-seg", [person], [image], str(tmp_path), confidence=1.0)[0]
    assert empty["count"] == 0
    assert "No matches detected" in empty["warning"]
