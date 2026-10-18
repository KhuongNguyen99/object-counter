"""SAM3Count prompt, native-image, mask suppression and local-loading contracts."""

import ast
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import List, Tuple

import numpy as np
from PIL import Image
import pytest
import torch

from app.sam3count import (Sam3CountEngine, _compact_mask, _counting_processor,
                           _load_algorithm, _load_finetuned_weights, _mask_iom, _mask_nms)


def _record(box, score=.8):
    x1, y1, x2, y2 = box
    mask = np.ones((y2 - y1, x2 - x1), dtype=bool)
    return {"box": list(box), "score": score, "mask": mask,
            "origin": (x1, y1), "area": int(mask.sum())}


@pytest.fixture(scope="module")
def algorithm():
    candidates = [Path(os.environ.get("SAM3COUNT_SOURCE_DIR", "/opt/sam3count")),
                  Path(__file__).resolve().parents[1] / ".vendor/SAM3Count"]
    source = next((path for path in candidates if (path / "sam3count_images.py").is_file()), None)
    if source is None:
        pytest.skip("Pinned SAM3Count source is needed for upstream algorithm agreement tests")
    return _load_algorithm(source), source


def test_helper_loader_ignores_demo_imports_and_side_effects(tmp_path, monkeypatch):
    from app import sam3count

    monkeypatch.setattr(sam3count, "_HELPERS", {"helper"})
    (tmp_path / "sam3count_images.py").write_text(
        "import unavailable_demo_dependency\nraise RuntimeError('demo ran')\n"
        "def helper(value):\n    return value + 1\n", encoding="utf-8")
    assert _load_algorithm(tmp_path).helper(4) == 5
    (tmp_path / "sam3count_images.py").write_text("def wrong(): pass", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pinned image algorithm"):
        _load_algorithm(tmp_path)


def test_constructor_loads_only_local_checkpoint_and_releases_model(tmp_path, monkeypatch):
    from app import models, sam3count

    calls, validated = [], []
    source = tmp_path / "sam3"
    monkeypatch.setenv("SAM3_SOURCE_DIR", str(source))
    monkeypatch.setenv("SAM3COUNT_SOURCE_DIR", str(tmp_path / "sam3count"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(models, "validate_sam3count_assets", lambda path: validated.append(path), raising=False)
    monkeypatch.setattr(sam3count, "_load_algorithm", lambda path: SimpleNamespace())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    packages = {}
    for name in ["sam3", "sam3.model", "sam3.model.vitdet", "sam3.model_builder",
                 "sam3.model.sam3_image_processor", "sam3.perflib", "sam3.perflib.fused"]:
        packages[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, packages[name])
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(packages[parent], child, packages[name])

    def build(**kwargs):
        calls.append(kwargs)
        return torch.nn.Linear(1, 1)

    class Processor:
        def __init__(self, model, **kwargs):
            self.model = model
            self.options = kwargs

    packages["sam3.model_builder"].build_sam3_image_model = build
    packages["sam3.model.sam3_image_processor"].Sam3Processor = Processor
    packages["sam3.perflib.fused"].addmm_act = lambda a, b, tensor: tensor.bfloat16()
    directory = tmp_path / "models"
    directory.mkdir()
    torch.save({"model": {"weight": torch.full((1, 1), 2.), "bias": torch.ones(1)}}, directory / "sam3_ft.pt")
    engine = Sam3CountEngine(directory, "cuda:0")
    assert validated == [directory.resolve()]
    assert calls[0]["checkpoint_path"] == str(directory.resolve() / "sam3.pt")
    assert calls[0]["load_from_HF"] is False and calls[0]["device"] == "cuda"
    assert calls[0]["enable_segmentation"] and not calls[0]["enable_inst_interactivity"]
    assert calls[0]["bpe_path"] == str(source.resolve() / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    assert not engine.model.training and engine.processor.options["resolution"] == 1008
    assert engine.model.weight.item() == 2.
    assert packages["sam3.model.vitdet"].addmm_act(None, None, torch.ones(1)).dtype == torch.float32
    engine.close()
    assert engine.model is None and engine.processor is None


def test_cpu_constructor_fails_explicitly_before_importing_cuda_sam(tmp_path, monkeypatch):
    from app import models, sam3count

    monkeypatch.setattr(models, "validate_sam3count_assets", lambda path: None, raising=False)
    monkeypatch.setattr(sam3count, "_load_algorithm", lambda path: SimpleNamespace())
    with pytest.raises(RuntimeError, match="CPU inference is unavailable"):
        Sam3CountEngine(tmp_path, "cpu")


def test_finetuned_weights_keep_base_segmentation_and_reject_other_missing_keys(tmp_path):
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(1, 1)
    model.segmentation_head = torch.nn.Linear(1, 1)
    original_head = model.segmentation_head.weight.detach().clone()
    checkpoint = tmp_path / "fine.pt"
    state = {"backbone.weight": torch.full((1, 1), 3.), "backbone.bias": torch.ones(1)}
    torch.save({"model": state}, checkpoint)
    _load_finetuned_weights(model, checkpoint, torch)
    assert model.backbone.weight.item() == 3.
    assert torch.equal(model.segmentation_head.weight, original_head)
    for broken in ({"backbone.weight": state["backbone.weight"]}, {**state, "unknown": torch.ones(1)}):
        torch.save({"model": broken}, checkpoint)
        with pytest.raises(ValueError, match="do not match the pinned model"):
            _load_finetuned_weights(model, checkpoint, torch)


def test_compact_masks_have_exact_global_intersection():
    first = np.zeros((100, 120), dtype=bool)
    first[20:40, 40:60] = True
    second = np.zeros_like(first)
    second[30:50, 45:65] = True
    a_mask, a_origin = _compact_mask(first)
    b_mask, b_origin = _compact_mask(second)
    a = {"mask": a_mask, "origin": a_origin, "area": int(first.sum())}
    b = {"mask": b_mask, "origin": b_origin, "area": int(second.sum())}
    assert a_mask.shape == b_mask.shape == (20, 20)
    assert _mask_iom(a, b) == np.logical_and(first, second).sum() / first.sum() == .375
    assert _compact_mask(np.zeros_like(first))[0].size == 0


def test_mask_nms_suppresses_contained_fragments_and_preserves_adjacent_objects():
    large = _record([20, 10, 60, 30], .9)
    fragment = _record([40, 10, 65, 30], .8)
    neighbor = _record([60, 10, 80, 30], .7)
    assert _mask_nms([fragment, neighbor, large]) == [large, neighbor]
    assert _mask_nms([]) == []


def test_compact_nms_agrees_with_unmodified_upstream(algorithm):
    _, source = algorithm
    tree = ast.parse((source / "sam3count_images.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"compute_iom", "nms_iom_simple"}]
    official = {"np": np, "List": List, "Tuple": Tuple}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "upstream", "exec"), official)
    rng = np.random.default_rng(42)
    masks, scores, records = [], [], []
    for index in range(50):
        x, y = rng.integers(0, 40, size=2)
        width, height = rng.integers(5, 30, size=2)
        mask = np.zeros((80, 80), dtype=bool)
        mask[y:y + height, x:x + width] = True
        mask[y + 2:y + 4, x + 2:x + 4] = False
        score = float(rng.choice([.4, .6, .8]))
        compact, origin = _compact_mask(mask)
        masks.append(mask)
        scores.append(score)
        records.append({"mask": compact, "origin": origin, "score": score,
                        "area": int(mask.sum()), "box": [index]})
    _, _, expected_ids = official["nms_iom_simple"](masks, scores, [[i] for i in range(50)], .5)
    assert [det["box"] for det in _mask_nms(records)] == expected_ids


def test_counting_processor_disables_presence_and_leaves_masks_small(monkeypatch):
    monkeypatch.setitem(sys.modules, "sam3.model.box_ops", SimpleNamespace(
        box_cxcywh_to_xyxy=lambda boxes: torch.stack(
            (boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
             boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2), dim=-1)))
    processor = _counting_processor(object)()
    processor.device = "cpu"
    processor.find_stage = object()
    processor.model = SimpleNamespace(forward_grounding=lambda **kwargs: {
        "pred_boxes": torch.tensor([[[.5, .5, .2, .4]]]),
        "pred_logits": torch.tensor([[[2.]]]), "presence_logit_dec": torch.tensor([[-100.]]),
        "pred_masks": torch.ones((1, 1, 8, 8)),
    })
    state = processor._forward_grounding({"backbone_out": {}, "geometric_prompt": None,
                                         "original_width": 4000, "original_height": 3000})
    assert state["scores"].item() > .8
    assert state["counting_mask_logits"].shape == (1, 8, 8)
    torch.testing.assert_close(state["boxes"], torch.tensor([[1600., 900., 2400., 2100.]]))
    assert "masks" not in state and "masks_logits" not in state


@pytest.mark.parametrize("box", [[float("nan"), 0, 4, 4], [3, 0, 1, 4], [21, 0, 30, 4]])
def test_invalid_model_coordinates_fail(box):
    engine = Sam3CountEngine.__new__(Sam3CountEngine)
    engine._torch = torch
    engine.device = torch.device("cpu")
    engine.processor = SimpleNamespace(set_image=lambda image: {}, set_text_prompt=lambda **kwargs: {
        "boxes": torch.tensor([box]), "scores": torch.tensor([.8]),
        "counting_mask_logits": torch.ones((1, 8, 8)),
    })
    with pytest.raises(ValueError, match="invalid detection"):
        engine._predict(Image.new("RGB", (20, 20)), "ring")


def test_prediction_preserves_thresholded_mask_pixels_and_clips_box():
    engine = Sam3CountEngine.__new__(Sam3CountEngine)
    engine._torch = torch
    engine.device = torch.device("cpu")
    logits = torch.tensor([[[-1., 1.], [-1., 1.]]])
    engine.processor = SimpleNamespace(set_image=lambda image: {}, set_text_prompt=lambda **kwargs: {
        "boxes": torch.tensor([[-1., 0, 5, 4]]), "scores": torch.tensor([.8]),
        "counting_mask_logits": logits,
    })
    det = engine._predict(Image.new("RGB", (4, 4)), "ring")[0]
    assert det["box"] == [0., 0., 4., 4.]
    assert det["origin"] == (2, 0) and det["area"] == 8
    assert det["mask"].shape == (4, 2) and det["mask"].all()


def test_tile_mask_floor_skips_low_score_allocation_without_reordering():
    engine = Sam3CountEngine.__new__(Sam3CountEngine)
    engine._torch = torch
    engine.device = torch.device("cpu")
    engine.processor = SimpleNamespace(set_image=lambda image: {}, set_text_prompt=lambda **kwargs: {
        "boxes": torch.tensor([[0., 0, 4, 4], [.1, 0, 4, 4], [.2, 0, 4, 4], [.3, 0, 4, 4]]),
        "scores": torch.tensor([.8, .1, .8, .4]), "counting_mask_logits": torch.ones((4, 2, 2)),
    })
    image = Image.new("RGB", (4, 4))
    original = engine._predict(image, "ring")
    reduced = engine._predict(image, "ring", mask_floor=.5)
    assert [det["score"] for det in reduced] == [det["score"] for det in original]
    assert [det["mask"].size for det in reduced] == [16, 0, 16, 0]
    assert [det["box"] for det in _mask_nms(reduced) if det["score"] >= .5] == [
        det["box"] for det in _mask_nms(original) if det["score"] >= .5]


def test_empty_low_score_masks_preserve_tied_high_score_nms_chains():
    rng = np.random.default_rng(123)
    records = []
    for index in range(80):
        x, y = map(int, rng.integers(0, 20, size=2))
        record = _record([x, y, x + 12, y + 12], float(rng.choice([.1, .2, .8, .8])))
        record["id"] = index
        records.append(record)
    reduced = [{**det, "mask": np.zeros((0, 0), dtype=bool), "area": 0}
               if det["score"] < .5 else det for det in records]
    assert [det["id"] for det in _mask_nms(reduced) if det["score"] >= .5] == [
        det["id"] for det in _mask_nms(records) if det["score"] >= .5]


@pytest.mark.parametrize("references,label,confidence,images,error", [
    ([{}], "ring", .35, ["image"], "does not accept reference"),
    ([], "   ", .35, ["image"], "requires an object description"),
    ([], "ring", float("nan"), ["image"], "Confidence"),
    ([], "ring", .35, [], "at least one image"),
])
def test_prompt_validation_precedes_inference(tmp_path, references, label, confidence, images, error):
    engine = Sam3CountEngine.__new__(Sam3CountEngine)
    with pytest.raises(ValueError, match=error):
        engine.count(references, images, str(tmp_path), confidence, label)


def test_sparse_result_uses_one_pass_with_numbered_safe_output(tmp_path, algorithm):
    source = tmp_path / "target.png"
    Image.new("RGB", (100, 80), "gray").save(source)
    engine = Sam3CountEngine.__new__(Sam3CountEngine)
    engine._algorithm = algorithm[0]
    calls = []

    def predict(image, label, **kwargs):
        calls.append((image.size, label))
        return [_record([60, 40, 70, 50], .8), _record([20, 10, 30, 20], .9)]

    engine._predict = predict
    result = engine.count([], [str(source)], str(tmp_path / "output"), .35, " white ring ")[0]
    assert calls == [((100, 80), "white ring")]
    assert result["count"] == len(result["detections"]) == 2
    assert [det["index"] for det in result["detections"]] == [1, 2]
    assert result["detections"][0]["box"] == [20, 10, 30, 20]
    assert result["processing"] == {"tiles": 1, "tile_size": 100, "duplicates_removed": 0}
    assert Path(result["annotated_path"]).is_file()
    json.dumps(result, allow_nan=False)


def test_zero_full_pass_still_covers_native_4k_and_offsets_duplicate_masks(tmp_path, algorithm):
    source = tmp_path / "target.png"
    image = Image.new("RGB", (3840, 2160))
    # Coordinates in pixel colors let a stub observe the actual crop origin.
    pixels = np.zeros((2160, 3840, 3), dtype=np.uint8)
    pixels[10:30, 1000:1020] = [255, 255, 255]
    Image.fromarray(pixels).save(source)
    engine = Sam3CountEngine.__new__(Sam3CountEngine)
    engine._algorithm = algorithm[0]
    calls = []

    def predict(crop, label, **kwargs):
        calls.append(crop.size)
        if crop.size == image.size:
            return []
        mask = np.asarray(crop)[:, :, 0] > 0
        rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        if not len(rows):
            return []
        return [_record([int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1])]

    engine._predict = predict
    result = engine.count([], [str(source)], str(tmp_path / "output"), .35, "ring")[0]
    assert calls[0] == (3840, 2160) and len(calls) > 2
    assert all(max(size) <= 1008 for size in calls[1:])
    assert result["count"] == 1
    assert result["detections"][0]["box"] == [1000, 10, 1020, 30]
    assert result["processing"]["duplicates_removed"] >= 1
    assert "Native-resolution tiling applied." in result["warning"]
    with Image.open(result["annotated_path"]) as annotated:
        assert annotated.size == (3840, 2160)


def test_native_tiles_replace_coarse_full_image_proposals(algorithm):
    engine = Sam3CountEngine.__new__(Sam3CountEngine)
    engine._algorithm = algorithm[0]
    engine._predict = lambda image, label, **kwargs: ([_record([0, 0, 1100, 100])] if image.width == 1100 else [])
    records, processing, _, native = engine._count_image(Image.new("RGB", (1100, 100)), .35, "ring")
    assert native and processing["tiles"] > 1 and records == []


@pytest.mark.parametrize("size", [(1008, 1008), (1100, 1100), (3840, 2160)])
def test_one_large_object_stays_one_across_native_crop_boundaries(size, algorithm):
    engine = Sam3CountEngine.__new__(Sam3CountEngine)
    engine._algorithm = algorithm[0]
    engine._predict = lambda image, label, **kwargs: [_record([0, 0, image.width, image.height], .9)]
    records, _, _, _ = engine._count_image(Image.new("RGB", size), .35, "large object")
    assert len(records) == 1 and records[0]["box"] == [0, 0, *size]


def test_native_fragment_association_preserves_two_neighboring_large_objects(algorithm):
    engine = Sam3CountEngine.__new__(Sam3CountEngine)
    engine._algorithm = algorithm[0]
    pixels = np.zeros((2160, 3840, 3), dtype=np.uint8)
    pixels[:, :1900, :] = 100
    pixels[:, 1940:, :] = 200
    image = Image.fromarray(pixels)

    def predict(crop, label, **kwargs):
        if crop.size == image.size:
            return [_record([0, 0, 3840, 2160], .95)]
        values = np.asarray(crop)[:, :, 0]
        records = []
        for color in (100, 200):
            rows, cols = np.flatnonzero((values == color).any(axis=1)), np.flatnonzero((values == color).any(axis=0))
            if len(rows):
                records.append(_record([int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1], .9))
        return records

    engine._predict = predict
    records, _, _, _ = engine._count_image(image, .35, "object")
    assert sorted(det["box"] for det in records) == [[0, 0, 1900, 2160], [1940, 0, 3840, 2160]]


def test_dense_ordinary_image_preserves_upstream_roi_guided_tiling(algorithm):
    engine = Sam3CountEngine.__new__(Sam3CountEngine)
    engine._algorithm = algorithm[0]
    calls = []

    def predict(image, label, **kwargs):
        calls.append(image.size)
        if len(calls) == 1:
            return [_record([20 + (i % 7) * 8, 20 + (i // 7) * 8,
                             23 + (i % 7) * 8, 23 + (i // 7) * 8]) for i in range(91)]
        return [_record([20, 20, 25, 25])]

    engine._predict = predict
    records, processing, _, native = engine._count_image(Image.new("RGB", (1008, 1008)), .35, "ring")
    assert not native and calls == [(1008, 1008), (504, 504)]
    assert len(records) == 1 and processing["tiles"] == 1


def test_upstream_low_confidence_fallback_retains_reduced_threshold(algorithm):
    engine = Sam3CountEngine.__new__(Sam3CountEngine)
    engine._algorithm = algorithm[0]
    calls = []

    def predict(image, label, **kwargs):
        calls.append(image.size)
        if len(calls) == 1:
            return [_record([i * 8, 5, i * 8 + 3, 8], .3) for i in range(11)]
        return [_record([20, 20, 30, 30], .31)]

    engine._predict = predict
    records, processing, _, native = engine._count_image(Image.new("RGB", (100, 100)), .35, "ring")
    assert not native and processing["tiles"] > 1 and len(calls) > 1
    assert records and all(det["score"] == .31 for det in records)


def test_low_score_queries_do_not_report_detection_cap(algorithm):
    engine = Sam3CountEngine.__new__(Sam3CountEngine)
    engine._algorithm = algorithm[0]
    engine._predict = lambda image, label, **kwargs: [_record([5, 5, 8, 8], .01) for _ in range(200)]
    _, _, limit_reached, _ = engine._count_image(Image.new("RGB", (100, 100)), .35, "ring")
    assert not limit_reached
    engine._predict = lambda image, label, **kwargs: [_record([5, 5, 8, 8], .9) for _ in range(200)]
    _, _, limit_reached, _ = engine._count_image(Image.new("RGB", (100, 100)), .35, "ring")
    assert limit_reached


def test_tile_limit_rejects_before_crop_inference(algorithm):
    engine = Sam3CountEngine.__new__(Sam3CountEngine)
    engine._algorithm = algorithm[0]
    calls = []
    engine._predict = lambda image, label, **kwargs: calls.append(image.size) or []
    with pytest.raises(ValueError, match="more than 256 tiles"):
        engine._count_image(Image.new("RGB", (40000, 1)), .35, "ring")
    assert calls == [(40000, 1)]
