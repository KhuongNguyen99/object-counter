"""Base SAM3 visual-exemplar boundary without loading a GPU checkpoint."""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from app import inference, models, sam3visual


def _image(path, size=(40, 30)):
    Image.new("RGB", size, "white").save(path)
    return str(path)


def test_base_integrity_does_not_require_overlay_or_count_source(tmp_path, monkeypatch):
    data = b"approved base"
    monkeypatch.setattr(models, "SAM3COUNT_ASSETS", {
        "sam3.pt": (len(data), hashlib.sha256(data).hexdigest()),
        "sam3_ft.pt": (20, "0" * 64),
    })
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("SAM3_SOURCE_DIR", str(tmp_path / "source"))
    monkeypatch.setenv("SAM3COUNT_SOURCE_DIR", str(tmp_path / "absent-counter"))
    monkeypatch.setenv("DEVICE", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    source = tmp_path / "source"
    for name in ("sam3/model_builder.py", "sam3/model/sam3_image_processor.py",
                 "sam3/assets/bpe_simple_vocab_16e6.txt.gz"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pinned")
    checkpoint = models.sam3count_dir() / "sam3.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(data)
    models.validate_sam3_base_asset()
    assert [entry["id"] for entry in models.list_models()] == [models.COUNTGD_MODEL_ID]
    checkpoint.write_bytes(b"x" * len(data))
    with pytest.raises(ValueError, match="SHA-256"):
        models.validate_sam3_base_asset()


def test_dispatch_validates_matching_bounded_boxes_before_model_load(tmp_path, monkeypatch):
    first = _image(tmp_path / "first.png")
    second = _image(tmp_path / "second.png")
    loaded = []

    class FakeEngine:
        def __init__(self, directory, device):
            loaded.append((directory, device))

        def count(self, references, image_paths, output_dir, confidence):
            assert [ref["path"] for ref in references] == image_paths
            assert confidence == .35
            return [{"count": 0} for _ in image_paths]

        def close(self):
            pass

    monkeypatch.setattr(sam3visual, "Sam3VisualEngine", FakeEngine)
    monkeypatch.setenv("DEVICE", "cpu")
    engine = inference.InferenceEngine()
    valid = [{"path": first, "boxes": [[1, 2, 8, 9]]},
             {"path": second, "boxes": [[10, 3, 20, 15]]}]
    for invalid in (valid[:1], [valid[1], valid[0]],
                    [{"path": first, "boxes": [[-1, 0, 8, 9]]}, valid[1]],
                    [{"path": first, "boxes": [[0, 0, float("nan"), 9]]}, valid[1]],
                    [{"path": first, "boxes": [[True, 0, 8, 9]]}, valid[1]],
                    [{"path": first, "boxes": [["1", 0, 8, 9]]}, valid[1]],
                    [{"path": first, "boxes": [[1, 0, 8, 9]] * 33}, valid[1]],
                    [{"path": first, "boxes": []}, valid[1]]):
        with pytest.raises(ValueError):
            engine.count("sam3", invalid, [first, second], str(tmp_path / "out"), .35, "ignored")
    assert not loaded
    assert engine.count("sam3", valid, [first, second], str(tmp_path / "out"), .35, "ignored") == [
        {"count": 0}, {"count": 0}]
    assert len(loaded) == 1


def test_each_image_uses_only_its_own_boxes_and_returns_standard_shape(tmp_path):
    first = _image(tmp_path / "first.png")
    second = _image(tmp_path / "second.png", (50, 30))
    engine = sam3visual.Sam3VisualEngine.__new__(sam3visual.Sam3VisualEngine)
    calls = []

    def predict(image, boxes, confidence):
        calls.append(boxes.tolist())
        return [{"box": boxes[0].tolist(), "confidence": .9}]

    engine._predict = predict
    result = engine.count([
        {"path": first, "boxes": [[1, 2, 8, 9]]},
        {"path": second, "boxes": [[12, 3, 22, 13]]},
    ], [first, second], str(tmp_path / "out"), .35)
    assert calls == [[[1., 2., 8., 9.]], [[12., 3., 22., 13.]]]
    assert [item["count"] for item in result] == [1, 1]
    assert [item["detections"][0]["index"] for item in result] == [1, 1]
    assert [(item["width"], item["height"]) for item in result] == [(40, 30), (50, 30)]
    assert all(Path(item["annotated_path"]).is_file() for item in result)
    json.dumps(result, allow_nan=False)


def test_standalone_rejects_non_numeric_or_too_many_boxes_and_reports_query_limit(tmp_path):
    source = _image(tmp_path / "target.png")
    engine = sam3visual.Sam3VisualEngine.__new__(sam3visual.Sam3VisualEngine)
    engine._predict = lambda image, boxes, confidence: [
        {"box": [1., 2., 8., 9.], "confidence": .8} for _ in range(190)]
    for boxes in ([[True, 0, 8, 9]], [["1", 0, 8, 9]], [[1, 0, 8, 9]] * 33):
        with pytest.raises(ValueError, match="1–32 boxes"):
            engine.count([{"path": source, "boxes": boxes}], [source], str(tmp_path / "out"))
    result = engine.count([{"path": source, "boxes": [[1, 0, 8, 9]]}],
                          [source], str(tmp_path / "out"))[0]
    assert result["count"] == 190 and "query limit" in result["warning"]


def test_visual_prompt_batches_normalized_boxes_once_and_never_uses_label(monkeypatch):
    calls = []

    class Prompt:
        def append_boxes(self, boxes, labels):
            calls.append((boxes.detach().cpu(), labels.detach().cpu()))

    class Processor:
        def set_image(self, image):
            calls.append("fresh image")
            return {"backbone_out": {}}

        def _forward_grounding(self, state):
            calls.append("grounding")
            assert state["backbone_out"]["mode"] == "visual"
            return {"boxes": torch.tensor([[1., 2., 8., 9.]]),
                    "scores": torch.tensor([.8])}

    model = SimpleNamespace(
        backbone=SimpleNamespace(forward_text=lambda tokens, device: (
            calls.append(tokens) or {"mode": tokens[0]})),
        _get_dummy_prompt=lambda: Prompt(),
    )
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    engine = sam3visual.Sam3VisualEngine.__new__(sam3visual.Sam3VisualEngine)
    engine._torch, engine.device, engine.model, engine.processor = torch, torch.device("cpu"), model, Processor()
    boxes = np.array([[1, 2, 8, 9], [10, 3, 20, 15]], dtype=np.float32)
    detections = engine._predict(Image.new("RGB", (40, 30)), boxes, .35)
    assert calls[0:2] == ["fresh image", ["visual"]]
    torch.testing.assert_close(calls[2][0], torch.tensor([
        [[.1125, .18333334, .175, .23333333]],
        [[.375, .3, .25, .4]],
    ]))
    assert calls[2][1].shape == (2, 1) and calls[2][1].all()
    assert calls[3:] == ["grounding"]
    assert detections == [{"box": [1., 2., 8., 9.], "confidence": pytest.approx(.8)}]
    engine._predict(Image.new("RGB", (40, 30)), boxes[1:], .35)
    assert calls[4:6] == ["fresh image", ["visual"]]
    assert calls[6][0].shape == (1, 1, 4)
    assert calls[7:] == ["grounding"]


def test_base_builder_never_loads_finetuned_overlay(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "sam3").mkdir(parents=True)
    (source / "sam3/model_builder.py").touch()
    monkeypatch.setenv("SAM3_SOURCE_DIR", str(source))
    monkeypatch.setattr(models, "validate_sam3_base_asset", lambda directory: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    packages = {}
    for name in ("sam3", "sam3.model", "sam3.model.vitdet", "sam3.model_builder",
                 "sam3.model.sam3_image_processor", "sam3.perflib", "sam3.perflib.fused"):
        packages[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, packages[name])
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(packages[parent], child, packages[name])
    kwargs_seen = []
    packages["sam3.model_builder"].build_sam3_image_model = lambda **kwargs: (
        kwargs_seen.append(kwargs) or torch.nn.Linear(1, 1))
    class FakeProcessor:
        def __init__(self, model, **kwargs):
            self.model = model

    packages["sam3.model.sam3_image_processor"].Sam3Processor = FakeProcessor
    packages["sam3.perflib.fused"].addmm_act = lambda *args: args[-1]
    engine = sam3visual.Sam3VisualEngine(tmp_path, "cuda:0")
    assert kwargs_seen[0]["checkpoint_path"] == str(tmp_path.resolve() / "sam3.pt")
    assert kwargs_seen[0]["load_from_HF"] is False
    assert kwargs_seen[0]["enable_inst_interactivity"] is False
    assert kwargs_seen[0]["enable_segmentation"] is False
    assert "sam3_ft.pt" not in str(kwargs_seen)
    engine.close()
