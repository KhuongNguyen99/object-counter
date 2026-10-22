import math
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from app import reference


def test_complete_objects_survive_but_background_parts_and_duplicates_do_not():
    boxes = [[0, 0, 999, 999], [100, 200, 200, 340], [120, 202, 180, 230],
             [102, 202, 199, 339], [400, 200, 500, 340], [900, 950, 999, 999],
             [math.nan, 1, 4, 5], [12, 20, 13, 21], [30, 20, 10, 40]]
    selected = reference._select_boxes(boxes, [.99, .85, .98, .95, .9, .95, .9, .9, .9], 1000, 1000)
    assert [item[0] for item in selected] == [[400, 200, 500, 340], [100, 200, 200, 340]]


def test_overlapping_distinct_examples_remain_separate():
    boxes = [[100, 100, 200, 200], [150, 110, 250, 210], [400, 100, 402, 200]]
    assert len(reference._select_boxes(boxes, [.9, .9, .9], 1000, 1000)) == 3


def test_near_full_frame_mask_cannot_suppress_a_real_example():
    selected = reference._select_boxes([[3, 3, 997, 797], [400, 300, 500, 400]], [.81, .99], 1000, 800)
    assert selected == [([400, 300, 500, 400], .99)]


class FakeFastSAM:
    def __init__(self, boxes, scores):
        self.boxes, self.scores = boxes, scores

    def predict(self, image, **kwargs):
        self.call = (image, kwargs)
        result = SimpleNamespace(boxes=SimpleNamespace(
            xyxy=np.asarray(self.boxes, dtype=float), conf=np.asarray(self.scores, dtype=float)))
        return [result]


def test_prompt_free_proposals_use_original_pixels_and_suppress_nested_parts(tmp_path):
    detector = FakeFastSAM([[1000, 1700, 1300, 2150], [1055, 1720, 1245, 1800]], [.54, .98])
    path = tmp_path / "reference.jpg"
    Image.new("RGB", (2252, 4000), "brown").save(path)

    result = reference.propose_boxes(detector, str(path), "cpu")

    assert result["model"] == "fastsam-s"
    assert (result["width"], result["height"]) == (2252, 4000)
    assert result["boxes"] == [[991, 1691, 1309, 2159]]
    assert result["warning"] is None
    assert result["elapsed_ms"] >= 0
    image, kwargs = detector.call
    assert image.mode == "RGB" and image.size == (2252, 4000)
    assert kwargs == {"device": "cpu", "imgsz": 1024, "conf": .25, "iou": .9,
                      "retina_masks": False, "verbose": False, "save": False}


def test_no_candidates_invites_manual_box(tmp_path):
    detector = FakeFastSAM([], [])
    path = tmp_path / "blank.jpg"
    Image.new("RGB", (100, 100), "gray").save(path)
    result = reference.propose_boxes(detector, str(path), "0")
    assert result["boxes"] == []
    assert "draw a box manually" in result["warning"].lower()
    assert detector.call[1]["device"] == "0"


@pytest.mark.parametrize("box,score", [([20, 20, math.nan, 40], .9),
                                      ([20, 20, 40, 40], math.inf),
                                      ([20, 20, 101, 40], .9),
                                      ([20, 20, 20, 40], .9),
                                      ([20, 20, 40], .9),
                                      ([20, 20, 40, 40], 1.1)])
def test_invalid_model_outputs_fail_instead_of_creating_boxes(tmp_path, box, score):
    path = tmp_path / "reference.png"
    Image.new("RGB", (100, 100), "gray").save(path)
    with pytest.raises(ValueError, match="invalid box"):
        reference.propose_boxes(FakeFastSAM([box], [score]), str(path), "cpu")


def test_mismatched_box_and_score_counts_fail(tmp_path):
    path = tmp_path / "reference.png"
    Image.new("RGB", (100, 100), "gray").save(path)
    with pytest.raises(ValueError, match="invalid box"):
        reference.propose_boxes(FakeFastSAM([[20, 20, 40, 40]], []), str(path), "cpu")


def test_reference_proposals_keep_all_eligible_boxes_beyond_32(tmp_path):
    boxes = [[50 + (i % 10) * 80, 50 + (i // 10) * 80,
              70 + (i % 10) * 80, 74 + (i // 10) * 80] for i in range(40)]
    path = tmp_path / "reference.png"
    Image.new("RGB", (1000, 800), "gray").save(path)
    result = reference.propose_boxes(FakeFastSAM(boxes, [.5 + i / 1000 for i in range(40)]), str(path), "cpu")
    assert len(result["boxes"]) == 40
    assert all(all(isinstance(value, int) for value in box) for box in result["boxes"])
    assert result["warning"] is None
