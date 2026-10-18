import math

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


class FakeCore:
    def __init__(self, boxes, scores):
        self.boxes, self.scores = boxes, scores

    def _predict(self, image, exemplar, boxes, confidence, label):
        self.call = (image.size, image is exemplar, boxes, confidence, label)
        return self.boxes, self.scores


def test_text_only_proposals_map_to_original_pixels_and_suppress_nested_parts(tmp_path):
    detector = FakeCore([[1150 / 2252, 1925 / 4000, 300 / 2252, 450 / 4000],
                         [1150 / 2252, 1760 / 4000, 190 / 2252, 80 / 4000]], [.54, .33])
    path = tmp_path / "reference.jpg"
    Image.new("RGB", (2252, 4000), "brown").save(path)
    result = reference.propose_boxes(detector, str(path), "bolt with red cap")
    assert (result["width"], result["height"]) == (2252, 4000)
    assert len(result["boxes"]) == 1
    assert result["boxes"][0] == pytest.approx([991, 1691, 1309, 2159], abs=1)
    assert detector.call == ((2252, 4000), True, [], .23, "bolt with red cap")
    assert Image.open(path).size == (2252, 4000)


def test_no_confident_object_keeps_manual_fallback(tmp_path):
    detector = FakeCore([], [])
    path = tmp_path / "blank.jpg"
    Image.new("RGB", (100, 100), "gray").save(path)
    result = reference.propose_boxes(detector, str(path), "ring")
    assert result["boxes"] == []
    assert "draw a box manually" in result["warning"]


@pytest.mark.parametrize("box,score", [([.5, .5, math.nan, .1], .9), ([.5, .5, .1, .1], math.inf),
                                      ([1.1, .5, .1, .1], .9), ([.5, .5, 0, .1], .9)])
def test_invalid_model_outputs_fail_instead_of_creating_boxes(tmp_path, box, score):
    path = tmp_path / "reference.png"
    Image.new("RGB", (100, 100), "gray").save(path)
    with pytest.raises(ValueError, match="invalid box"):
        reference.propose_boxes(FakeCore([box], [score]), str(path), "part")


def test_reference_box_limit_reports_truncation(tmp_path):
    boxes = [[(50 + (i % 10) * 80) / 1000, (50 + (i // 10) * 80) / 800, .02, .03] for i in range(40)]
    path = tmp_path / "reference.png"
    Image.new("RGB", (1000, 800), "gray").save(path)
    result = reference.propose_boxes(FakeCore(boxes, [.5 + i / 1000 for i in range(40)]), str(path), "part")
    assert len(result["boxes"]) == 32
    assert "32 strongest" in result["warning"]
