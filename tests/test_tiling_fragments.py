import json
from pathlib import Path
import pytest
from app.tiling import Tile, make_tiles, merge_tile_detections

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "yoloe_crop_fragments.json").read_text())


def rotate_case(records, width, height, boxes, turns):
    """Rotate observations and expected boxes to exercise all four crop edges."""
    for _ in range(turns):
        def rotate(box):
            return (height - box[3], box[0], height - box[1], box[2])
        rotated = []
        for tile, items in records:
            next_tile = Tile(tile.index, rotate(tile.box), rotate(tile.core))
            next_items = []
            for record in items:
                world = [v + tile.box[i % 2] for i, v in enumerate(record["box"])]
                next_items.append({**record, "box": [v - next_tile.box[i % 2]
                                                    for i, v in enumerate(rotate(world))]})
            rotated.append((next_tile, next_items))
        records, boxes = rotated, [list(rotate(box)) for box in boxes]
        width, height = height, width
    return records, width, height, boxes


@pytest.mark.parametrize("turns", range(4))
@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["name"])
def test_complete_view_absorbs_real_jittered_fragments(fixture, turns):
    records = [(Tile(row["tile"]["index"], tuple(row["tile"]["box"]), tuple(row["tile"]["core"])), row["records"])
               for row in fixture["predictions"]]
    records, width, height, expected = rotate_case(records, fixture["width"], fixture["height"],
                                                  [fixture["expected_box"]], turns)
    merged = merge_tile_detections(records, width, height, include_members=True)
    assert len(merged) == 1
    assert merged[0]["box"] == pytest.approx(expected[0], abs=1e-8)
    assert len(merged[0]["members"]) == sum(len(items) for _, items in records)
    assert merged == merge_tile_detections(reversed(records), width, height, include_members=True)


@pytest.mark.parametrize("turns", range(4))
def test_thin_separate_object_seen_with_complete_object_is_kept(turns):
    tiles = make_tiles(1440, 800, 800)
    # The narrow object is distinct and overlaps a larger object at the cut.
    # An independent observation in the complete tile must preserve it.
    records = [
        (tiles[0], [{"box": [580, 300, 650, 370], "confidence": .9},
                    {"box": [638, 302, 658, 368], "confidence": .7}]),
        (tiles[1], [{"box": [0, 302, 18, 368], "confidence": .6}]),
    ]
    records, width, height, expected = rotate_case(records, 1440, 800,
                                                  [[580, 300, 650, 370], [638, 302, 658, 368]], turns)
    merged = merge_tile_detections(records, width, height)
    assert len(merged) == 2
    assert {tuple(r["box"]) for r in merged} == {tuple(box) for box in expected}


@pytest.mark.parametrize("turns", range(4))
def test_short_contained_fragment_does_not_inherit_large_object(turns):
    tiles = make_tiles(1440, 800, 800)
    records = [
        (tiles[0], [{"box": [580, 300, 650, 450], "confidence": .9}]),
        (tiles[1], [{"box": [0, 340, 18, 360], "confidence": .6}]),
    ]
    records, width, height, _ = rotate_case(records, 1440, 800, [], turns)
    assert len(merge_tile_detections(records, width, height)) == 2


@pytest.mark.parametrize("turns", range(4))
def test_thin_fragment_without_complete_support_is_kept(turns):
    tiles = make_tiles(1440, 800, 800)
    records = [(tiles[0], []), (tiles[1], [{"box": [0, 302, 18, 368], "confidence": .6}])]
    records, width, height, expected = rotate_case(records, 1440, 800, [[640, 302, 658, 368]], turns)
    merged = merge_tile_detections(records, width, height)
    assert len(merged) == 1
    assert merged[0]["box"] == expected[0]
