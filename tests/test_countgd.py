"""CountGD++ reference handling and result contract."""

import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from app.countgd import CountGDEngine, _exemplar_image


def test_single_reference_keeps_original_image_and_all_boxes(tmp_path):
    source = tmp_path / "reference.png"
    Image.new("RGB", (120, 80), "red").save(source)
    reference = {"path": str(source), "boxes": [[5, 10, 25, 30], [50, 35, 85, 65]]}
    image, boxes = _exemplar_image([reference])
    assert image.size == (120, 80)
    assert boxes == reference["boxes"]
    _, merged = _exemplar_image([reference, reference])
    assert merged == reference["boxes"] * 2


def test_separate_reference_images_make_valid_contact_sheet(tmp_path):
    sources = []
    for index, color in enumerate(("red", "blue")):
        source = tmp_path / f"{index}.png"
        Image.new("RGB", (100, 90), color).save(source)
        sources.append({"path": str(source), "boxes": [[20, 15, 70, 65]]})
    sheet, boxes = _exemplar_image(sources)
    assert sheet.size == (512, 256)
    assert len(boxes) == 2
    assert all(0 <= x1 < x2 <= sheet.width and 0 <= y1 < y2 <= sheet.height
               for x1, y1, x2, y2 in boxes)
    assert boxes[0][2] < 256 < boxes[1][0]
    assert sheet.getpixel((128, 128)) == (255, 0, 0)
    assert sheet.getpixel((384, 128)) == (0, 0, 255)


def test_contact_sheet_accepts_extremely_thin_valid_reference(tmp_path):
    thin = tmp_path / "thin.png"
    regular = tmp_path / "regular.png"
    Image.new("RGB", (10000, 1), "white").save(thin)
    Image.new("RGB", (20, 20), "black").save(regular)
    sheet, boxes = _exemplar_image([
        {"path": str(thin), "boxes": [[0, 0, 10000, 1]]},
        {"path": str(regular), "boxes": [[1, 1, 19, 19]]},
    ])
    assert sheet.size == (512, 256)
    assert all(x2 > x1 and y2 > y1 for x1, y1, x2, y2 in boxes)


def test_result_keeps_one_detection_per_box_and_forwards_label(tmp_path):
    source = tmp_path / "image.png"
    Image.new("RGB", (100, 100), "gray").save(source)
    engine = CountGDEngine.__new__(CountGDEngine)
    calls = []

    def predict(image, exemplar, boxes, confidence, label):
        calls.append((confidence, label, boxes))
        return np.array([[.7, .6, .2, .2], [.3, .2, .2, .2]]), np.array([.8, .9])

    engine._predict = predict
    output = engine.count([{"path": str(source), "boxes": [[10, 10, 40, 40]]}],
                          [str(source)], str(tmp_path / "output"), .23, "white plastic ring")[0]
    assert calls == [(.23, "white plastic ring", [[10, 10, 40, 40]])]
    assert output["count"] == len(output["detections"]) == 2
    assert [item["index"] for item in output["detections"]] == [1, 2]
    np.testing.assert_allclose(output["detections"][0]["box"], [20, 10, 40, 30], atol=.001)
    assert Path(output["annotated_path"]).is_file()
    assert (output["width"], output["height"]) == (100, 100)
    assert "Experimental model" in output["warning"]


@pytest.mark.parametrize("thin_is_target", [True, False])
def test_extreme_aspect_ratio_is_rejected_before_official_resize(thin_is_target):
    engine = CountGDEngine.__new__(CountGDEngine)
    thin = Image.new("RGB", (1, 5000))
    square = Image.new("RGB", (10, 10))
    target, exemplar = (thin, square) if thin_is_target else (square, thin)
    with pytest.raises(ValueError, match="aspect ratio"):
        engine._predict(target, exemplar, [[0, 0, 1, 1]], .23, "ring")


@pytest.mark.skipif(os.environ.get("RUN_COUNTGD_TESTS") != "1", reason="Set RUN_COUNTGD_TESTS=1 with local assets and source")
def test_real_external_reference_agrees_with_official_folder_inference(tmp_path):
    root = Path(__file__).resolve().parents[1]
    reference = root / "test-results/ring-debug-input/reference-original.jpg"
    target = root / "test-results/ring-debug-input/target-original.jpg"
    if not (reference.is_file() and target.is_file()):
        pytest.skip("Local benchmark images are unavailable")
    engine = CountGDEngine(root / "models/countgd", os.environ.get("DEVICE", "auto"))
    import ast
    import json
    # The upstream folder script executes CLI parsing at import time. Load its
    # unchanged imports/functions without launching the command-line program.
    source = Path(os.environ.get("COUNTGD_SOURCE_DIR", "/opt/countgd")) / "process_folder.py"
    parsed = ast.parse(source.read_text())
    definitions = ast.Module(body=[node for node in parsed.body if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.FunctionDef))], type_ignores=[])
    official = {}
    exec(compile(definitions, str(source), "exec"), official)
    predict = official["predict"]
    # Compare on the same device/runtime: GPU and CPU can select different
    # candidates, so a frozen GPU total does not establish CPU API agreement.
    image = Image.open(target).convert("RGB")
    exemplar = Image.open(reference).convert("RGB")
    official_boxes, _ = predict(engine.model, engine.transform, image, "white plastic ring",
                               {"image": exemplar, "points": [[808, 1640, 2., 1512, 2340, 3.]]},
                               [], [], engine.device, .23)
    # Compare the raw adapter, since count() now also tiles large source images.
    actual_boxes, actual_scores = engine._predict(image, exemplar, [[808, 1640, 1512, 2340]],
                                                 .23, "white plastic ring")
    (tmp_path / "countgd-result.json").write_text(json.dumps(actual_boxes.tolist(), indent=2))
    assert len(actual_boxes) == len(official_boxes) >= 50
    np.testing.assert_allclose(sorted(actual_boxes.tolist()), sorted(official_boxes.tolist()), atol=.0001)
    conditioning = engine._prepare_conditioning(exemplar, [[808, 1640, 1512, 2340]], "white plastic ring")
    cached_boxes, cached_scores = engine._predict(image, exemplar, [[808, 1640, 1512, 2340]],
                                                 .23, "white plastic ring", conditioning=conditioning)
    np.testing.assert_allclose(cached_boxes, actual_boxes, rtol=0, atol=.000001)
    np.testing.assert_allclose(cached_scores, actual_scores, rtol=0, atol=.000001)


def test_tiled_count_translates_and_merges_without_adding_tile_totals(tmp_path):
    source = tmp_path / "large.png"
    Image.new("RGB", (1400, 800), "gray").save(source)
    engine = CountGDEngine.__new__(CountGDEngine)
    predictions = iter([
        (np.array([[.875, .5, .1, .1]]), np.array([.8])),
        (np.array([[.125, .5, .1, .1], [.375, .5, .1, .1]]), np.array([.9, .85])),
    ])
    calls = []

    def predict(image, exemplar, boxes, confidence, label):
        calls.append((image.size, confidence, label))
        return next(predictions)

    engine._predict = predict
    result = engine.count([{"path": str(source), "boxes": [[10, 10, 40, 40]]}],
                          [str(source)], str(tmp_path / "output"), .23, "fitting")[0]
    assert calls == [((800, 800), .23, "fitting")] * 2
    assert result["processing"] == {"tiles": 2, "tile_size": 800, "duplicates_removed": 1}
    assert result["count"] == len(result["detections"]) == 2
    np.testing.assert_allclose(result["detections"][0]["box"], [660, 360, 740, 440], atol=.001)
    np.testing.assert_allclose(result["detections"][1]["box"], [860, 360, 940, 440], atol=.001)
    assert [d["index"] for d in result["detections"]] == [1, 2]
    with Image.open(result["annotated_path"]) as annotation:
        assert annotation.size == (1400, 800)


def test_query_limit_applies_to_each_tile_not_final_count(tmp_path, monkeypatch):
    from app import countgd
    monkeypatch.setattr(countgd, "MAX_QUERIES", 2)
    source = tmp_path / "large.png"
    Image.new("RGB", (1400, 800), "gray").save(source)
    engine = CountGDEngine.__new__(CountGDEngine)
    engine._predict = lambda *args: (np.array([[.5, .5, .1, .1]]), np.array([.8]))
    refs = [{"path": str(source), "boxes": [[10, 10, 40, 40]]}]
    result = engine.count(refs, [str(source)], str(tmp_path / "output"))[0]
    assert result["count"] == 2
    assert "Detection limit reached" not in result["warning"]
    engine._predict = lambda *args: (np.array([[.5, .5, .1, .1], [.8, .5, .1, .1]]), np.array([.8, .9]))
    limited = engine.count(refs, [str(source)], str(tmp_path / "limited"))[0]
    assert "Detection limit reached" in limited["warning"]


def test_invalid_model_boxes_fail_instead_of_being_reported_as_merged_duplicates(tmp_path):
    source = tmp_path / "image.png"
    Image.new("RGB", (1400, 800), "gray").save(source)
    engine = CountGDEngine.__new__(CountGDEngine)
    engine._predict = lambda *args: (np.array([[float("nan"), .5, .1, .1]]), np.array([.8]))
    with pytest.raises(ValueError, match="invalid detection"):
        engine.count([{"path": str(source), "boxes": [[10, 10, 40, 40]]}],
                     [str(source)], str(tmp_path / "output"))


def test_reference_conditioning_reused_across_tiles_and_images_but_not_requests(tmp_path):
    import torch

    class Model:
        def __init__(self):
            self.prepared = []
            self.received = []

        def prepare_conditioning(self, exemplar, boxes, negative_images, negative_boxes, captions):
            conditioning = {"caption": captions[0], "reference": exemplar[0].clone(), "boxes": boxes[0].clone()}
            self.prepared.append(conditioning)
            return conditioning

        def forward_conditioned(self, target, conditioning):
            self.received.append(conditioning)
            return {"input_ids": torch.tensor([[101, 1012, 102]]),
                    "pred_logits": torch.tensor([[[4., 4., -4.]]]),
                    "pred_boxes": torch.tensor([[[.5, .5, .2, .2]]])}

        def __call__(self, target, exemplar, boxes, negative_images, negative_boxes, captions):
            conditioning = self.prepare_conditioning(exemplar, boxes, negative_images, negative_boxes, captions)
            return self.forward_conditioned(target, conditioning)

    first, second, target = (tmp_path / name for name in ("first.png", "second.png", "target.png"))
    Image.new("RGB", (20, 20), "red").save(first)
    Image.new("RGB", (20, 20), "blue").save(second)
    Image.new("RGB", (1400, 800), "gray").save(target)
    transforms = []

    def transform(image, metadata):
        transforms.append("target" if metadata is None else "reference")
        return torch.tensor(image.getpixel((0, 0)), dtype=torch.float32), metadata

    engine = CountGDEngine.__new__(CountGDEngine)
    engine._torch, engine.device = torch, torch.device("cpu")
    engine._nested_tensor = lambda tensors: tensors
    engine.transform, engine.model = transform, Model()
    references = [{"path": str(first), "boxes": [[1, 2, 10, 12]]}]
    result = engine.count(references, [str(target), str(target)], str(tmp_path / "output"), label="white.ring")
    assert [item["processing"]["tiles"] for item in result] == [2, 2]
    assert transforms.count("reference") == 1 and transforms.count("target") == 4
    assert len(engine.model.prepared) == 1
    assert all(value is engine.model.prepared[0] for value in engine.model.received)
    assert engine.model.prepared[0]["caption"] == "white ring . "

    engine.count([{"path": str(second), "boxes": [[2, 3, 11, 13]]}], [str(target)],
                 str(tmp_path / "next"), label="cap")
    assert len(engine.model.prepared) == 2
    assert engine.model.received[-1] is engine.model.prepared[1]
    assert engine.model.prepared[1]["caption"] == "cap . "
    assert not torch.equal(engine.model.prepared[0]["reference"], engine.model.prepared[1]["reference"])
    assert not torch.equal(engine.model.prepared[0]["boxes"], engine.model.prepared[1]["boxes"])

    uncached = engine.count(references, [str(target)], str(tmp_path / "uncached"),
                            label="white.ring", cache_reference=False)
    assert len(engine.model.prepared) == 4
    assert transforms.count("reference") == 4
    assert uncached[0]["detections"] == result[0]["detections"]
    with Image.open(target) as image, Image.open(first) as exemplar:
        engine._predict(image, exemplar, references[0]["boxes"], .25, "ring")
    assert len(engine.model.prepared) == 5
