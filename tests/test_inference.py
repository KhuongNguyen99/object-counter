"""Unit checks plus opt-in real-checkpoint smoke tests: RUN_MODEL_TESTS=1 pytest -q."""

import os
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import zipfile

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


def test_fastsam_checkpoint_rejects_missing_or_same_length_modified_asset(tmp_path, monkeypatch):
    from app import models

    checkpoint = tmp_path / models.FASTSAM_FILENAME
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("checkpoint", b"trusted official checkpoint")
    contents = checkpoint.read_bytes()
    monkeypatch.setattr(models, "FASTSAM_ASSETS", {
        models.FASTSAM_FILENAME: (len(contents), hashlib.sha256(contents).hexdigest()),
    })
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="download_models"):
        models.validate_checkpoint(checkpoint)
    checkpoint.write_bytes(contents)
    models.validate_checkpoint(checkpoint)
    checkpoint.write_bytes(b"x" * len(contents))
    with pytest.raises(ValueError, match="SHA-256"):
        models.validate_checkpoint(checkpoint)


def test_reference_boxes_reject_invalid_coordinates():
    for boxes in ([], [[1, 2, 1, 5]], [[-1, 0, 4, 4]], [[0, 0, 101, 3]], [[0, 0, float("nan"), 4]]):
        with pytest.raises(ValueError):
            _reference_boxes(boxes, 100, 50)
    np.testing.assert_equal(_reference_boxes([[0, 0, 100, 50]], 100, 50), [[0, 0, 100, 50]])


def test_model_catalog_only_offers_countgd_and_requires_local_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    models = list_models()
    assert len(models) == 1
    countgd = models[0]
    assert countgd["id"] == "countgd-plusplus"
    assert countgd["default_confidence"] == .80
    assert countgd["reference_mode"] == "per_image"
    assert countgd["required_reference_boxes"] == 10
    assert not countgd["available"]
    with pytest.raises(ValueError):
        checkpoint_path("../../untrusted")
    with pytest.raises(FileNotFoundError, match="download_models"):
        InferenceEngine().count("yoloe-26n-seg", [{"path": "unused", "boxes": []}], ["unused"], str(tmp_path))
    fake = tmp_path / MODEL_FILENAMES[0]
    fake.write_text("<html>Not a checkpoint</html>")
    with pytest.raises(ValueError, match="incomplete or invalid"):
        validate_checkpoint(fake)


def test_countgd_per_image_runs_one_conditioning_call_per_target(tmp_path, monkeypatch):
    first = _fixture_image(tmp_path / "first.png", (16, 12))
    second = _fixture_image(tmp_path / "second.png", (16, 12))
    third = _fixture_image(tmp_path / "third.png", (16, 12))
    references = [{"path": first, "boxes": [[i, 1, i + 4, 5] for i in range(1, 11)]},
                  {"path": second, "boxes": [[i, 2, i + 4, 6] for i in range(1, 11)]}]

    class FakeCountGD:
        def __init__(self):
            self.calls = []

        def count(self, refs, paths, output_dir, confidence, label):
            self.calls.append((refs, paths, output_dir, confidence, label))
            return [{"source": paths[0], "boxes": refs[0]["boxes"]}]

    model = FakeCountGD()
    engine = InferenceEngine()
    monkeypatch.setattr(engine, "_device", lambda: "cpu")
    monkeypatch.setattr(engine, "_load_countgd_model", lambda device: model)
    result = engine.count("countgd-plusplus", references, [first, second], str(tmp_path / "out"),
                          label="", reference_mode="per_image")
    assert [item["source"] for item in result] == [first, second]
    assert len(model.calls) == 2
    assert [(refs[0]["path"], paths) for refs, paths, *_ in model.calls] == [
        (first, [first]), (second, [second])]
    assert [refs[0]["boxes"] for refs, *_ in model.calls] == [ref["boxes"] for ref in references]
    assert all(len(refs) == 1 for refs, *_ in model.calls)

    later = {"path": third, "boxes": [[i, 3, i + 4, 7] for i in range(1, 11)]}
    engine.count("countgd-plusplus", [later], [third], str(tmp_path / "out"),
                 reference_mode="per_image")
    assert model.calls[-1][0] == [later] and model.calls[-1][1] == [third]


@pytest.mark.parametrize("bad_references", [
    lambda first, second: [{"path": first, "boxes": [[1, 1, 8, 8]] * 9}],
    lambda first, second: [{"path": first, "boxes": [[1, 1, 8, 8]] * 11}],
    lambda first, second: [{"path": first, "boxes": [[1, 1, 8, 8]] * 10}],
    lambda first, second: [{"path": first, "boxes": [[True, 1, 8, 8]] +
                           [[i, 1, i + 4, 5] for i in range(2, 11)]}],
    lambda first, second: [{"path": first, "boxes": [["1", 1, 8, 8]] +
                           [[i, 1, i + 4, 5] for i in range(2, 11)]}],
    lambda first, second: [{"path": second, "boxes": [[1, 1, 8, 8]] * 10}],
    lambda first, second: [{"path": first, "boxes": [[1, 1, 8, 8]] * 10},
                           {"path": second, "boxes": [[1, 1, 8, 8]] * 10}],
])
def test_countgd_per_image_rejects_invalid_references_before_model_load(tmp_path, monkeypatch,
                                                                         bad_references):
    first = _fixture_image(tmp_path / "first.png", (16, 12))
    second = _fixture_image(tmp_path / "second.png", (16, 12))
    engine = InferenceEngine()
    monkeypatch.setattr(engine, "_device", lambda: pytest.fail("device selected before validation"))
    monkeypatch.setattr(engine, "_load_countgd_model", lambda _: pytest.fail("model loaded before validation"))
    with pytest.raises(ValueError, match="CountGD"):
        engine.count("countgd-plusplus", bad_references(first, second), [first], str(tmp_path / "out"),
                     reference_mode="per_image")


@pytest.fixture
def fake_yoloe(monkeypatch):
    """Exercise the real count/load loop with a cheap fake YOLOE and visual encoder."""
    import torch
    from app import inference

    state = SimpleNamespace(models=[], encoder_calls=0, prompts=[], predict_fn=lambda source: [], embedding_fn=None)

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
            if state.embedding_fn is not None:
                return state.embedding_fn(self.prompts["bboxes"])
            left = torch.as_tensor(self.prompts["bboxes"][:, 0], dtype=torch.float32)
            return torch.stack((torch.ones_like(left), left + 1), dim=-1).unsqueeze(0)

    class FakeYOLOE:
        def __init__(self, path, task):
            assert task == "segment"
            self.model = SimpleNamespace()
            self.predictor = None
            self.calls = []
            self.embeddings = []
            self.classes = []
            state.models.append(self)

        def set_classes(self, classes, embeddings):
            assert classes == ["object"] or classes == [f"object-{i}" for i in range(embeddings.shape[1])]
            self.model.pe = embeddings
            self.embeddings.append(embeddings.clone())
            self.classes.append(list(classes))

        def predict(self, source, **options):
            assert self.model.pe.shape[1] == len(self.classes[-1])
            assert options["imgsz"] >= max(source.size)
            assert options["max_det"] >= 1
            assert options["nms"] and options["agnostic_nms"] and options["iou"] == .5
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


def test_per_object_matches_frozen_tensor_recipe_and_next_normal_run_uses_mean(tmp_path, fake_yoloe):
    import torch

    source = _fixture_image(tmp_path / "wide.png", (100, 64))
    refs = [{"path": source, "boxes": [[2, 2, 12, 12], [10, 2, 20, 12]]},
            {"path": source, "boxes": [[20, 2, 30, 12]]}]
    groups = [torch.tensor([[[1., 3.], [1., 11.]]]), torch.tensor([[[1., 21.]]])]
    expected = torch.nn.functional.normalize(torch.cat(groups, dim=1), p=2, dim=-1)
    engine = InferenceEngine()
    per_object = engine.count("yoloe-26x-seg", refs, [source, source], str(tmp_path / "out"),
                              reference_strategy="per-object")
    model = fake_yoloe.models[0]
    torch.testing.assert_close(model.embeddings[-1], expected, rtol=0, atol=0)
    assert model.classes[-1] == ["object-0", "object-1", "object-2"]
    assert fake_yoloe.encoder_calls == 2 and len(model.calls) == 4
    assert all(result["processing"]["tiles"] == 2 for result in per_object)
    engine.count("yoloe-26x-seg", refs, [source], str(tmp_path / "out"))
    expected_mean = torch.nn.functional.normalize(torch.cat(groups, dim=1).mean(1, keepdim=True), p=2, dim=-1)
    assert len(fake_yoloe.models) == 1 and fake_yoloe.encoder_calls == 4
    assert model.classes[-1] == ["object"]
    torch.testing.assert_close(model.embeddings[-1], expected_mean, rtol=0, atol=0)


@pytest.mark.parametrize("bad_vector", [[0., 0.], [float("nan"), 1.]])
def test_per_object_rejects_unusable_individual_prototypes(tmp_path, fake_yoloe, bad_vector):
    import torch

    source = _fixture_image(tmp_path / "small.png", (32, 32))
    fake_yoloe.embedding_fn = lambda boxes: torch.tensor([[[1., 2.], bad_vector]])
    with pytest.raises((ValueError, RuntimeError), match="embedding"):
        InferenceEngine().count("yoloe-26x-seg",
                                [{"path": source, "boxes": [[2, 2, 12, 12], [20, 2, 30, 12]]}],
                                [source], str(tmp_path / "out"), reference_strategy="per-object")
    assert not fake_yoloe.models[0].calls and not fake_yoloe.models[0].embeddings


@pytest.mark.parametrize("model,strategy", [("yoloe-26x-seg", "unknown"), ("countgd-plusplus", "per-object")])
def test_reference_strategy_validation_precedes_model_loading(tmp_path, model, strategy):
    with pytest.raises(ValueError, match="[Ss]trategy"):
        InferenceEngine().count(model, [{"path": "unused", "boxes": []}], ["unused"], str(tmp_path),
                                reference_strategy=strategy)


def test_reference_proposals_reuse_fastsam_and_reload_after_yoloe(tmp_path, fake_yoloe, monkeypatch):
    from app import inference, reference

    source = _fixture_image(tmp_path / "small.png", (32, 32))
    state = SimpleNamespace(proposals=[], fastsam=[], validated=[], closed=[])

    def fake_propose(model, image_path, device):
        assert model is state.fastsam[-1] and image_path == source and device == "cpu"
        state.proposals.append(model)
        return {"width": 32, "height": 32, "boxes": [[1, 1, 10, 10]],
                "model": "fastsam-s", "elapsed_ms": 1, "warning": None}

    class FakeFastSAM:
        def __init__(self, path):
            assert path == str(tmp_path / "FastSAM-s.pt")
            assert state.validated[-1] == tmp_path / "FastSAM-s.pt"
            state.fastsam.append(self)

        def close(self):
            state.closed.append(self)

    def validate(path):
        if path.name == "FastSAM-s.pt":
            state.validated.append(path)

    monkeypatch.setattr(inference, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(inference, "validate_checkpoint", validate)
    monkeypatch.setattr(sys.modules["ultralytics"], "FastSAM", FakeFastSAM, raising=False)
    monkeypatch.setattr(reference, "propose_boxes", fake_propose, raising=False)
    engine = InferenceEngine()
    refs = [{"path": source, "boxes": [[2, 2, 12, 12]]}]
    fake_yoloe.predict_fn = lambda crop: []
    engine.count("yoloe-26n-seg", refs, [source], str(tmp_path / "out"))
    assert engine._model_id == "yoloe-26n-seg" and len(fake_yoloe.models) == 1
    assert engine.suggest_reference_boxes(source, "rings")["boxes"] == [[1, 1, 10, 10]]
    assert engine._model_id == "fastsam-s" and len(state.fastsam) == 1
    engine.suggest_reference_boxes(source)
    assert len(state.fastsam) == 1 and state.proposals == [state.fastsam[0]] * 2
    assert state.validated == [tmp_path / "FastSAM-s.pt"]
    engine.count("yoloe-26n-seg", refs, [source], str(tmp_path / "out"))
    assert engine._model_id == "yoloe-26n-seg" and len(fake_yoloe.models) == 2
    assert state.closed == [state.fastsam[0]]
    engine.suggest_reference_boxes(source, "washers")
    assert engine._model_id == "fastsam-s" and len(state.fastsam) == 2
    assert state.proposals[-1] is state.fastsam[1]
    assert state.validated == [tmp_path / "FastSAM-s.pt"] * 2


@pytest.mark.parametrize("failure", [FileNotFoundError("missing"), ValueError("corrupt"), ImportError("package")])
def test_reference_model_load_failures_are_unavailable(tmp_path, fake_yoloe, monkeypatch, failure):
    from app import inference, reference

    loaded = []

    def validate(path):
        assert path == tmp_path / "FastSAM-s.pt"
        if not isinstance(failure, ImportError):
            raise failure

    def failing_fastsam(path):
        loaded.append(path)
        raise failure

    monkeypatch.setattr(inference, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(inference, "validate_checkpoint", validate)
    monkeypatch.setattr(sys.modules["ultralytics"], "FastSAM", failing_fastsam, raising=False)
    with pytest.raises(reference.ReferenceModelUnavailable, match="FastSAM"):
        InferenceEngine().suggest_reference_boxes(str(tmp_path / "image.jpg"), "rings")
    assert loaded == ([str(tmp_path / "FastSAM-s.pt")] if isinstance(failure, ImportError) else [])


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
