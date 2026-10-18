"""PET-DINO visual conditioning, native-pixel results, and worker protocol."""

import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from app.petdino_runtime import PetDINORuntime, _image_input, _serve


def _reference(tmp_path, name="reference.png", color="red", size=(100, 100)):
    path = tmp_path / name
    Image.new("RGB", size, color).save(path)
    return {"path": str(path), "boxes": [[10, 10, 40, 40]]}


def _fake_engine():
    engine = PetDINORuntime.__new__(PetDINORuntime)
    engine._torch = torch
    engine.model = SimpleNamespace()
    return engine


def test_constructor_loads_local_fp32_weights_without_training_assets(tmp_path, monkeypatch):
    from app import models

    source = tmp_path / "source"
    config_path = source / "configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py"
    config_path.parent.mkdir(parents=True)
    config_path.touch()
    monkeypatch.setenv("PETDINO_SOURCE_DIR", str(source))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setattr(sys, "path", list(sys.path))
    validated, loaded = [], []
    monkeypatch.setattr(models, "validate_petdino_assets", lambda directory: validated.append(directory))
    config = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(name="remote"),
                              backbone=SimpleNamespace(init_cfg={"checkpoint": "remote"}),
                              memory_bank={"training": "dataset.json"}, use_autocast=True, test_cfg={}),
        visualizer=SimpleNamespace(vis_backends=["tensorboard"]), load_from="remote",
    )
    model = torch.nn.Linear(1, 1).double()

    def inferencer(**kwargs):
        loaded.append(kwargs)
        return SimpleNamespace(model=model)

    monkeypatch.setitem(sys.modules, "mmengine.config", SimpleNamespace(
        Config=SimpleNamespace(fromfile=lambda path: config)))
    monkeypatch.setitem(sys.modules, "mmdet.apis", SimpleNamespace(DetInferencer=inferencer))
    directory = tmp_path / "models/petdino"
    engine = PetDINORuntime(directory, "cpu")
    assert validated == [directory.resolve()]
    assert loaded[0]["weights"] == str(directory / "pet_dino_swin-t_8xb4_12e_obj365.pth")
    assert loaded[0]["device"] == "cpu"
    assert config.model.language_model.name == str(tmp_path / "models/countgd/bert-base-uncased")
    assert config.model.backbone.init_cfg is None and config.load_from is None
    assert config.model.memory_bank == {} and config.model.use_autocast is False
    assert config.model.test_cfg == {"max_per_img": 900, "chunked_size": -1, "prompt_type": "Visual"}
    assert config.visualizer.vis_backends == []
    assert engine.model.weight.dtype == torch.float32 and not engine.model.training
    assert os.environ["HF_HUB_OFFLINE"] == os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_references_keep_all_boxes_and_official_per_image_pooling(tmp_path):
    first = _reference(tmp_path)
    second = _reference(tmp_path, "blue.png", "blue")
    more_first = {"path": first["path"], "boxes": [[50, 20, 80, 60]]}
    received = []

    def pipeline(data):
        received.append(data)
        return data

    def extract(data):
        value = 2. if data[0]["img"][0, 0, 2] else 10.
        return torch.full((1, 1, 256), value)

    engine = _fake_engine()
    engine.inferencer = SimpleNamespace(prompt_image_test_pipeline=pipeline,
                                        collate_fn=lambda data: data, extract_visual_embedding=extract)
    pooled = engine._prepare_references([first, more_first, second])
    assert len(received) == 2
    assert received[0]["prompt_bboxes"] == first["boxes"] + more_first["boxes"]
    assert received[0]["prompt_bboxes_labels"] == [0, 0]
    assert received[1]["prompt_bboxes_labels"] == [0]
    assert pooled.shape == (1, 256)
    assert torch.equal(pooled, torch.full((1, 256), 6.))
    assert received[0]["img"][0, 0].tolist() == [0, 0, 255]


def test_optional_official_enhancer_runs_before_image_mean(tmp_path):
    engine = _fake_engine()
    engine.model.visual_prompt_enhancer = object()
    enhanced = []

    def enhance(features):
        enhanced.append(features)
        return features + 5

    engine.inferencer = SimpleNamespace(
        prompt_image_test_pipeline=lambda data: data, collate_fn=lambda data: data,
        extract_visual_embedding=lambda data: torch.ones((1, 1, 256)),
        forward_visual_prompt_enhancer=enhance,
    )
    assert torch.equal(engine._prepare_references([_reference(tmp_path)]), torch.full((1, 256), 6.))
    assert len(enhanced) == 1


@pytest.mark.parametrize("embedding", [torch.zeros((1, 2, 256)), torch.full((1, 1, 256), float("nan"))])
def test_invalid_reference_embedding_fails(tmp_path, embedding):
    engine = _fake_engine()
    engine.inferencer = SimpleNamespace(prompt_image_test_pipeline=lambda data: data,
                                        collate_fn=lambda data: data, extract_visual_embedding=lambda data: embedding)
    with pytest.raises(RuntimeError, match="invalid visual-reference embeddings"):
        engine._prepare_references([_reference(tmp_path)])


def test_prediction_uses_official_memory_branch_and_confidence_threshold():
    engine = _fake_engine()
    metadata, inputs = {}, []

    def pipeline(data):
        inputs.append(data)
        return {"data_samples": SimpleNamespace(set_metainfo=metadata.update), "inputs": "pixels"}

    def forward(data):
        assert len(data) == 1 and data[0]["inputs"] == "pixels"
        return [SimpleNamespace(pred_instances=SimpleNamespace(
            bboxes=torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]]),
            scores=torch.tensor([.2, .3, .8]),
        ))]

    engine.inferencer = SimpleNamespace(pipeline=pipeline, collate_fn=lambda data: data, forward=forward)
    embedding = torch.ones((1, 256))
    boxes, scores = engine._predict(Image.new("RGB", (20, 20), "red"), embedding, .3, "  white.ring  ")
    assert boxes.tolist() == [[9, 10, 11, 12]]
    np.testing.assert_allclose(scores, [.8])
    assert metadata["memory_visual"] is embedding
    assert metadata["memory_label"].tolist() == [0]
    assert metadata["prompt_type"] == "Visual"
    assert inputs[0]["text"] == ["white.ring"]
    assert inputs[0]["custom_entities"] is True


def test_conditioning_reused_across_tiles_and_images_then_refreshed(tmp_path):
    reference = _reference(tmp_path)
    target = _reference(tmp_path, "target.png", "gray", (1400, 800))["path"]
    engine = _fake_engine()
    prepared, received = [], []

    def prepare(references):
        condition = object()
        prepared.append((references, condition))
        return condition

    def predict(image, embedding, confidence, label):
        received.append((image.size, embedding, confidence, label))
        return np.array([[360., 360., 440., 440.]]), np.array([.9])

    engine._prepare_references, engine._predict = prepare, predict
    results = engine.count([reference], [target, target], str(tmp_path / "output"), .23, "ring")
    assert len(prepared) == 1 and len(received) == 4
    assert all(size == (800, 800) and emb is prepared[0][1] and conf == .23 and label == "ring"
               for size, emb, conf, label in received)
    assert all(result["count"] == 2 for result in results)
    second = _reference(tmp_path, "second.png", "blue")
    engine.count([second], [target], str(tmp_path / "next"))
    assert len(prepared) == 2 and prepared[1][0] == [second]
    assert received[-1][1] is prepared[1][1] and prepared[1][1] is not prepared[0][1]


def test_tiles_merge_duplicates_and_keep_original_size_numbered_boxes(tmp_path):
    reference = _reference(tmp_path)
    target = _reference(tmp_path, "target.png", "gray", (1400, 800))["path"]
    engine = _fake_engine()
    engine._prepare_references = lambda refs: None
    predictions = iter([
        (np.array([[660, 360, 740, 440]]), np.array([.8])),
        (np.array([[60, 360, 140, 440], [260, 360, 340, 440]]), np.array([.9, .85])),
    ])
    engine._predict = lambda *args: next(predictions)
    result = engine.count([reference], [target], str(tmp_path / "output"))[0]
    assert result["processing"] == {"tiles": 2, "tile_size": 800, "duplicates_removed": 1}
    assert result["count"] == len(result["detections"]) == 2
    assert [record["index"] for record in result["detections"]] == [1, 2]
    assert [record["box"] for record in result["detections"]] == [[660, 360, 740, 440], [860, 360, 940, 440]]
    with Image.open(result["annotated_path"]) as image:
        assert image.size == (1400, 800)


def test_query_limit_warns_per_tile_not_total(tmp_path, monkeypatch):
    from app import petdino_runtime
    monkeypatch.setattr(petdino_runtime, "MAX_QUERIES", 2)
    ref = _reference(tmp_path)
    target = _reference(tmp_path, "target.png", "gray", (1400, 800))["path"]
    engine = _fake_engine()
    engine._prepare_references = lambda refs: None
    engine._predict = lambda *args: (np.array([[100, 100, 200, 200]]), np.array([.8]))
    result = engine.count([ref], [target], str(tmp_path / "one"))[0]
    assert result["count"] == 2 and "Detection limit" not in result["warning"]
    engine._predict = lambda *args: (np.array([[100, 100, 200, 200], [300, 300, 400, 400]]), np.array([.8, .9]))
    assert "Detection limit reached" in engine.count([ref], [target], str(tmp_path / "two"))[0]["warning"]


@pytest.mark.parametrize("box, score", [([float("nan"), 0, 1, 1], .8), ([10, 10, 1, 1], .8),
                                       ([0, 0, 10, 10], float("nan")), ([0, 0, 10, 10], 1.5)])
def test_invalid_model_detection_is_rejected(tmp_path, box, score):
    reference = _reference(tmp_path)
    engine = _fake_engine()
    engine._prepare_references = lambda refs: None
    engine._predict = lambda *args: (np.array([box]), np.array([score]))
    with pytest.raises(ValueError, match="invalid detection"):
        engine.count([reference], [reference["path"]], str(tmp_path / "output"))


def test_thin_reference_rejected_before_resize():
    with pytest.raises(ValueError, match="aspect ratio"):
        _image_input(Image.new("RGB", (1, 5000)))


def test_worker_loads_once_handles_errors_and_keeps_stdout_protocol_clean(capsys):
    created, calls = [], []

    class Engine:
        def __init__(self, directory, device):
            created.append((directory, device))
            print("framework startup log")

        def count(self, *args):
            calls.append(args)
            print("framework prediction log")
            if args[-1] == "fail":
                raise ValueError("bad reference")
            return [{"count": 2}]

    message = {"action": "count", "references": [], "image_paths": ["image.png"], "output_dir": "out"}
    requests = io.StringIO("\n".join([
        json.dumps(message), json.dumps({**message, "label": "fail"}), "not-json", "[]",
        json.dumps(message), '{"action":"close"}', json.dumps(message),
    ]))
    responses = io.StringIO()
    assert _serve(Path("models"), "cpu", requests, responses, Engine) == 0
    records = [json.loads(line) for line in responses.getvalue().splitlines()]
    assert created == [(Path("models"), "cpu")]
    assert len(calls) == 3
    assert records[0] == {"ready": True}
    assert records[1] == records[-1] == {"ok": True, "results": [{"count": 2}]}
    assert records[2] == {"ok": False, "error": "bad reference"}
    assert records[3]["ok"] is False and records[4]["ok"] is False
    captured = capsys.readouterr()
    assert captured.out == "" and "framework startup log" in captured.err


def test_worker_reports_initialization_error_and_exits():
    def failed(*args):
        raise RuntimeError("missing assets")

    responses = io.StringIO()
    assert _serve(Path("models"), "cpu", io.StringIO(""), responses, failed) == 1
    assert json.loads(responses.getvalue()) == {"ready": False, "error": "missing assets"}


@pytest.mark.skipif(os.environ.get("RUN_PETDINO_TESTS") != "1", reason="Set RUN_PETDINO_TESTS=1 in the PET-DINO runtime")
def test_real_cached_external_reference_agrees_with_official_inferencer(tmp_path):
    from app.inference import _read_image

    root = Path(__file__).resolve().parents[1]
    reference = root / "test-results/ring-debug-input/reference-original.jpg"
    target = root / "test-results/ring-debug-input/target-original.jpg"
    if not reference.is_file() or not target.is_file():
        pytest.skip("Local external-reference benchmark images are unavailable")
    engine = PetDINORuntime(root / "models/petdino", os.environ.get("DEVICE", "auto"))
    boxes, label, confidence = [[808, 1640, 1512, 2340]], "white plastic ring", 0.
    image = _read_image(str(target)).crop((600, 900, 1400, 1700))
    exemplar = _read_image(str(reference))
    memory = engine._prepare_references([{"path": str(reference), "boxes": boxes}])
    actual_boxes, actual_scores = engine._predict(image, memory, confidence, label)
    official = engine.inferencer(
        _image_input(image), prompt_type="Visual", prompt_image=_image_input(exemplar),
        prompt_bboxes=boxes, prompt_bboxes_labels=[0], texts=label, custom_entities=True,
        no_save_vis=True, no_save_pred=True, return_datasamples=True,
    )["predictions"][0].pred_instances
    selected = official.scores > confidence
    assert len(actual_boxes) > 0
    np.testing.assert_allclose(actual_boxes, official.bboxes[selected].cpu().numpy(), rtol=0, atol=.001)
    np.testing.assert_allclose(actual_scores, official.scores[selected].cpu().numpy(), rtol=0, atol=.00001)
