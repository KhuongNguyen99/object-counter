"""Native coverage and adversarial cross-crop counting geometry."""

import math

import pytest

from app.tiling import Tile, make_tiles, merge_tile_detections


def predictions(tiles, objects):
    """Ideal model: return the visible fragment of every intersecting object."""
    result = []
    for tile in tiles:
        records = []
        for box in objects:
            clipped = [max(box[0], tile.box[0]), max(box[1], tile.box[1]),
                       min(box[2], tile.box[2]), min(box[3], tile.box[3])]
            if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                records.append({"box": [clipped[0] - tile.box[0], clipped[1] - tile.box[1],
                                        clipped[2] - tile.box[0], clipped[3] - tile.box[1]],
                                "confidence": .8})
        result.append((tile, records))
    return result


@pytest.mark.parametrize("width,height,size,count", [
    (3840, 2160, 800, 24), (3840, 2160, 640, 32),
    (6000, 5000, 800, 80), (6000, 5000, 640, 120),
    (801, 33, 800, 2), (5, 800, 800, 1),
])
def test_native_pixel_coverage_and_cost(width, height, size, count):
    tiles = make_tiles(width, height, size)
    assert len(tiles) == count
    assert all(0 < tile.width <= size and 0 < tile.height <= size for tile in tiles)
    assert sum((tile.core[2] - tile.core[0]) * (tile.core[3] - tile.core[1]) for tile in tiles) == width * height
    assert min(tile.box[0] for tile in tiles) == min(tile.box[1] for tile in tiles) == 0
    assert max(tile.box[2] for tile in tiles) == width
    assert max(tile.box[3] for tile in tiles) == height
    for tile in tiles:
        assert tile.box[0] <= tile.core[0] < tile.core[2] <= tile.box[2]
        assert tile.box[1] <= tile.core[1] < tile.core[3] <= tile.box[3]
    for horizontal in (True, False):
        coordinate = 0 if horizontal else 1
        intervals = sorted({(tile.box[coordinate], tile.box[coordinate + 2]) for tile in tiles})
        assert all(a[1] >= b[0] for a, b in zip(intervals, intervals[1:]))


def test_extreme_panorama_fails_before_allocating_unbounded_tiles():
    with pytest.raises(ValueError, match="limit is 256"):
        make_tiles(30_000_000, 1, 800)


@pytest.mark.parametrize("width,height,box", [
    (1440, 800, [700, 310, 740, 360]),  # vertical seam, exact core boundary
    (800, 1440, [310, 700, 360, 740]),  # horizontal seam
    (1440, 1440, [700, 700, 740, 740]),  # four-way intersection
    (1700, 1230, [1200, 700, 1240, 740]),  # distributed final crop
])
def test_one_boundary_object_is_counted_once(width, height, box):
    tiles = make_tiles(width, height, 800)
    merged = merge_tile_detections(predictions(tiles, [box]), width, height)
    assert len(merged) == 1
    assert merged[0]["box"] == box


def test_jittered_duplicate_and_partial_edge_box_choose_complete_view():
    tiles = make_tiles(1440, 800, 800)
    records = predictions(tiles, [[785, 300, 825, 330]])
    records[0][1][0]["box"] = [786, 301, 799, 331]
    records[0][1][0]["confidence"] = .99
    merged = merge_tile_detections(records, 1440, 800)
    assert len(merged) == 1
    assert merged[0]["box"] == [785, 300, 825, 330]
    assert merged[0]["confidence"] == .8


@pytest.mark.parametrize("complete_tile,partial_tile,complete,partial", [
    (2, 1, [1390.508, 211.862, 1472.233, 300.721], [1397.434, 221.752, 1440, 300.442]),
    (1, 2, [1244.910, 316.421, 1321.915, 396.159], [1280, 323.783, 1314.059, 396.612]),
    (3, 9, [2438.731, 434.351, 2528.703, 518.405], [2443.061, 484, 2513.658, 514.492]),
])
def test_real_countgd_crop_jitter_retains_one_complete_box(complete_tile, partial_tile, complete, partial):
    tiles = make_tiles(4000, 2252, 800)
    records = []
    for tile_index, box in ((complete_tile, complete), (partial_tile, partial)):
        tile = tiles[tile_index]
        records.append((tile, [{"box": [box[0] - tile.box[0], box[1] - tile.box[1],
                                        box[2] - tile.box[0], box[3] - tile.box[1]],
                                "confidence": .8}]))
    merged = merge_tile_detections(records, 4000, 2252)
    assert len(merged) == 1
    assert merged[0]["box"] == complete


@pytest.mark.parametrize("views,expected", [
    ([(12, [617.547, 1431.866, 721.607, 1510.781]),
      (13, [640, 1429.006, 714.522, 1509.870]),
      (18, [616.608, 1452, 701.092, 1507.193]),
      (19, [640, 1452, 690.534, 1507.670])], [617.547, 1431.866, 721.607, 1510.781]),
    ([(7, [1255.546, 1185.352, 1340.947, 1269.232]),
      (8, [1280, 1184.399, 1329.588, 1254.272]),
      (13, [1257.011, 1185.575, 1336.645, 1262.497]),
      (14, [1280, 1179.907, 1321.399, 1255.678])], [1257.011, 1185.575, 1336.645, 1262.497]),
    ([(15, [2430.593, 1420.952, 2532.753, 1504.732]),
      (21, [2435.022, 1452, 2512.075, 1490.353])], [2430.593, 1420.952, 2532.753, 1504.732]),
    ([(10, [3155.533, 951.247, 3235.161, 1046.130]),
      (11, [3200, 951.034, 3226.144, 1038.146]),
      (16, [3155.067, 968, 3232.436, 1024.829])], [3155.533, 951.247, 3235.161, 1046.130]),
    ([(13, [1275.359, 1437.853, 1353.116, 1518.787]),
      (14, [1280, 1438.803, 1347.533, 1517.235]),
      (19, [1276.112, 1452, 1350.235, 1515.172]),
      (20, [1280, 1452, 1345.271, 1515.192])], [1275.359, 1437.853, 1353.116, 1518.787]),
])
def test_real_multiview_fragments_cannot_veto_the_complete_object(views, expected):
    tiles = make_tiles(4000, 2252, 800)
    records = []
    for tile_index, box in views:
        tile = tiles[tile_index]
        records.append((tile, [{"box": [box[0] - tile.box[0], box[1] - tile.box[1],
                                        box[2] - tile.box[0], box[3] - tile.box[1]],
                                "confidence": .8}]))
    merged = merge_tile_detections(records, 4000, 2252)
    assert len(merged) == 1
    assert merged[0]["box"] == expected


def test_complete_small_object_jitter_is_associated():
    tiles = make_tiles(1440, 800, 800)
    records = predictions(tiles, [[700, 300, 710, 310]])
    records[1][1][0]["box"] = [62, 301, 72, 311]
    assert len(merge_tile_detections(records, 1440, 800)) == 1


@pytest.mark.parametrize("objects", [
    [[700, 300, 710, 310], [712, 300, 722, 310]],
    [[700, 300, 720, 320], [704, 300, 724, 320]],  # true, strongly overlapping neighbors
    [[700, 300, 740, 340], [708, 308, 718, 318]],  # a nested distinct object
])
def test_nearby_objects_seen_together_are_never_collapsed(objects):
    tiles = make_tiles(1440, 800, 800)
    merged = merge_tile_detections(predictions(tiles, objects), 1440, 800)
    assert len(merged) == len(objects)
    assert {tuple(record["box"]) for record in merged} == {tuple(box) for box in objects}


def test_nonowner_only_detection_is_kept():
    tiles = make_tiles(1440, 800, 800)
    records = predictions(tiles, [[730, 300, 750, 320]])
    records[1][1].clear()  # its owning tile missed the object
    assert merge_tile_detections(records, 1440, 800)[0]["box"] == [730, 300, 750, 320]


def test_original_frame_edges_and_corner_are_kept():
    tiles = make_tiles(1440, 1440, 800)
    objects = [[0, 0, 25, 25], [1410, 10, 1440, 40], [5, 1410, 30, 1440],
               [1410, 1410, 1440, 1440], [0, 700, 20, 740]]
    merged = merge_tile_detections(predictions(tiles, objects), 1440, 1440)
    assert len(merged) == len(objects)
    assert {tuple(record["box"]) for record in merged} == {tuple(box) for box in objects}


@pytest.mark.parametrize("box", [
    [100, 300, 2100, 330], [700, 700, 750, 750], [500, 300, 1000, 340],
])
def test_long_objects_and_crossing_fragments_merge(box):
    tiles = make_tiles(2240, 1440, 800)
    merged = merge_tile_detections(predictions(tiles, [box]), 2240, 1440)
    assert len(merged) == 1
    assert merged[0]["box"] == box


def test_overlapping_partial_neighbors_do_not_chain_into_one_object():
    tiles = make_tiles(1440, 1440, 800)
    objects = [[780, 700, 840, 750], [790, 710, 850, 760]]
    merged = merge_tile_detections(predictions(tiles, objects), 1440, 1440)
    assert len(merged) == 2
    assert {tuple(record["box"]) for record in merged} == {tuple(box) for box in objects}


def test_two_large_overlapping_objects_do_not_produce_mixed_fragments():
    tiles = make_tiles(2240, 1440, 800)
    objects = [[1197, 438, 1740, 1207], [1312, 348, 1863, 1140]]
    merged = merge_tile_detections(predictions(tiles, objects), 2240, 1440)
    assert len(merged) == 2
    assert {tuple(record["box"]) for record in merged} == {tuple(box) for box in objects}


def test_near_identical_large_objects_with_independent_multiview_support_survive():
    tiles = make_tiles(2240, 1440, 800)
    objects = [[1154, 593, 2109, 1370], [1156, 658, 2080, 1360]]
    merged = merge_tile_detections(predictions(tiles, objects), 2240, 1440)
    assert len(merged) == 2
    assert {tuple(record["box"]) for record in merged} == {tuple(box) for box in objects}


def test_incompatible_endpoints_prevent_transitive_box_chains():
    tiles = [Tile(i, (offset, 0, offset + 800, 800), (offset, 0, offset + 800, 800))
             for i, offset in enumerate((0, 100, 200))]
    # Each neighboring pair has high IoU, but the first and last predictions
    # disagree in a region seen by all three crops. Do not collapse the chain.
    records = [(tile, [{"box": [700 + 4 * i - tile.box[0], 300,
                                720 + 4 * i - tile.box[0], 320], "confidence": .8}])
               for i, tile in enumerate(tiles)]
    assert len(merge_tile_detections(records, 1000, 800)) == 2


def test_tile_iteration_order_does_not_change_representative():
    tiles = make_tiles(1440, 1440, 800)
    records = predictions(tiles, [[700, 700, 740, 740], [750, 700, 780, 740]])
    assert merge_tile_detections(records, 1440, 1440) == merge_tile_detections(reversed(records), 1440, 1440)


def test_single_tile_preserves_same_tile_duplicates_without_mutating_inputs():
    tile = make_tiles(600, 400, 800)[0]
    records = [{"box": [10, 20, 40, 50], "confidence": .9, "extra": "retained"},
               {"box": [11, 20, 41, 50], "confidence": .8}]
    merged = merge_tile_detections([(tile, records)], 600, 400)
    assert merged == [{**record, "index": i} for i, record in enumerate(records, 1)]
    assert all("index" not in record for record in records)


def test_source_members_are_returned_only_when_requested():
    tiles = make_tiles(1000, 800, 800)
    records = predictions(tiles, [[700, 200, 900, 400]])
    for tile, items in records:
        for item in items:
            item["source"] = tile.index
    ordinary = merge_tile_detections(records, 1000, 800)
    detailed = merge_tile_detections(records, 1000, 800, include_members=True)
    assert len(ordinary) == len(detailed) == 1
    assert "members" not in ordinary[0]
    assert {item["source"] for item in detailed[0]["members"]} == {0, 1}
    assert {key: value for key, value in detailed[0].items() if key != "members"} == ordinary[0]
    assert all("members" not in item for _, items in records for item in items)


def test_bad_coordinates_do_not_create_spurious_detections():
    tile = Tile(0, (0, 0, 100, 50), (0, 0, 100, 50))
    records = [{"box": box, "confidence": .8} for box in
               ([0, 0, math.nan, 10], [30, 10, 20, 20], [150, 5, 160, 10], [-5, -5, 10, 10])]
    records.append({"box": [1, 1, 4, 4], "confidence": math.inf})
    assert merge_tile_detections([(tile, records)], 100, 50) == [
        {"box": [0., 0., 10., 10.], "confidence": .8, "index": 1}]
    with pytest.raises(ValueError, match="only once"):
        merge_tile_detections([(tile, []), (tile, [])], 100, 50)


@pytest.mark.parametrize("tile_predictions,expected_count", [
    ([
        [([58.665367, 400.249146, 200.108185, 640.0], 0.691398), ([231.161621, 407.097168, 343.314636, 640.0], 0.527154)],
        [([60.557053, 406.532959, 172.495285, 640.0], 0.525277), ([554.395203, 393.104248, 640.0, 640.0], 0.405277)],
        [([53.070747, 0.0, 241.753662, 462.366272], 0.661728), ([226.334869, 0.26001, 342.045746, 415.94278], 0.55241)],
        [([55.693192, 0.244598, 172.702423, 417.785004], 0.719507), ([502.628479, 0.0, 639.954529, 436.659424], 0.528075), ([0.0, 14.794495, 73.682594, 461.495544], 0.336803)],
    ], 3),
    ([
        [([60.18502, 399.332153, 200.678009, 640.0], 0.793758), ([234.181976, 407.944489, 341.682159, 639.862183], 0.756136)],
        [([60.635181, 408.211731, 172.16951, 639.911682], 0.730844), ([552.271973, 388.352844, 640.0, 640.0], 0.404353)],
        [([54.15921, 0.175049, 243.583923, 460.288208], 0.803211), ([226.729523, 0.0, 341.7742, 417.802887], 0.767962), ([0.0, 9.883484, 76.513519, 437.023621], 0.381946)],
        [([500.997467, 0.161133, 640.0, 437.350739], 0.802315), ([56.299927, 1.416321, 172.080414, 418.624939], 0.709753), ([0.0, 10.570557, 73.307495, 458.808167], 0.599181)],
    ], 4),
    ([
        [([59.31395, 400.492065, 200.522537, 639.900269], 0.915428), ([228.948441, 407.13324, 342.361389, 639.861877], 0.85198)],
        [([59.599171, 406.743103, 172.427322, 639.916809], 0.853866), ([552.375854, 385.129883, 640.0, 639.703003], 0.562828)],
        [([52.263557, 0.0, 244.863342, 460.085754], 0.831628), ([226.490067, 0.0, 342.376038, 417.162109], 0.82517), ([0.014488, 30.300476, 76.088371, 449.266357], 0.264797)],
        [([55.98674, 0.0, 172.588837, 416.984985], 0.812349), ([500.610809, 1.794403, 639.887939, 438.363098], 0.767607), ([0.061707, 8.700195, 74.194366, 459.073425], 0.476564)],
    ], 4),
    ([
        [([59.071831, 399.155609, 201.001587, 639.90686], 0.692779), ([226.88562, 406.920624, 341.34906, 639.995605], 0.479346)],
        [([57.299591, 407.066956, 171.494171, 640.0], 0.50362), ([551.951965, 392.114868, 639.886169, 640.0], 0.386713)],
        [([52.974808, 1.409515, 243.238113, 462.706512], 0.719569), ([225.545349, 1.082642, 341.671265, 418.855957], 0.653539), ([0.018784, 6.355164, 77.454712, 638.369507], 0.403736)],
        [([500.402954, 1.755859, 639.96106, 438.428833], 0.766714), ([55.623062, 0.0, 171.038361, 419.442444], 0.657853), ([0.033257, 0.0, 74.624527, 463.759338], 0.402391)],
    ], 4),
    ([
        [([59.435112, 399.345398, 201.171631, 640.0], 0.775986), ([230.921387, 406.806763, 342.853516, 639.999878], 0.675397)],
        [([551.253357, 392.498718, 640.0, 639.876404], 0.443784), ([60.987, 406.713715, 172.359024, 640.0], 0.24378)],
        [([51.797333, 0.0, 243.371948, 461.265076], 0.769527), ([225.393494, 0.0, 342.295227, 419.336792], 0.737291), ([0.002861, 19.346375, 76.809105, 630.824768], 0.205577)],
        [([500.442596, 0.303711, 639.943604, 437.779541], 0.855883), ([55.548569, 0.205872, 172.078064, 418.114563], 0.795662), ([0.088997, 1.878418, 73.758583, 463.002655], 0.661545)],
    ], 4),
], ids=["nano", "small", "medium", "large", "extra-large"])
def test_real_yoloe_person_fragments_reconstruct_one_box_per_detected_person(tile_predictions, expected_count):
    # Offline, fixed-image predictions from all five YOLOE-26 checkpoints.
    # A narrow white-coat corner fragment must not veto the broader views;
    # the right person's legs extend farther left than their upper body.
    tiles = make_tiles(810, 1080, 640)
    records = [(tile, [{"box": box, "confidence": confidence} for box, confidence in predictions])
               for tile, predictions in zip(tiles, tile_predictions)]
    merged = merge_tile_detections(records, 810, 1080)
    assert len(merged) == expected_count
    boxes = [record["box"] for record in merged]
    coat = [box for box in boxes if 45 < box[0] < 65 and 235 < box[2] < 250]
    center = [box for box in boxes if 220 < box[0] < 235 and 335 < box[2] < 350]
    right = [box for box in boxes if 660 < box[0] < 685 and box[2] > 805]
    assert len(coat) == len(center) == len(right) == 1
    assert coat[0][1] < 410 and coat[0][3] > 895
    assert center[0][1] < 420 and center[0][3] > 850
    assert right[0][1] < 400 and right[0][3] > 870
    if expected_count == 4:
        assert len([box for box in boxes if box[0] <= 1 and box[2] < 85]) == 1


def test_broader_clipped_support_does_not_override_a_complete_view_conflict():
    tiles = make_tiles(1440, 1440, 800)
    records = [
        (tiles[0], [{"box": [780, 695, 800, 725], "confidence": .8}]),
        (tiles[1], [{"box": [140, 700, 160, 720], "confidence": .8}]),
        (tiles[3], [{"box": [140, 50, 160, 90], "confidence": .8}]),
    ]
    assert len(merge_tile_detections(records, 1440, 1440)) == 2
