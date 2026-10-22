import copy

import pytest

from app.inference import InferenceEngine
from app.models import COUNTGD_MODEL_ID, CROSSCHECK_MODEL_ID
from app.verification import compare_detections


def detections(*boxes):
    return [{"index": i, "box": list(box), "confidence": .8} for i, box in enumerate(boxes, 1)]


def compare(a, b):
    return compare_detections(a, b, COUNTGD_MODEL_ID, "yoloe-26x-seg", .23, .01)


def test_equal_counts_can_have_different_objects_and_preserve_inputs():
    a = detections((0, 0, 10, 10), (100, 0, 110, 10))
    b = detections((0, 0, 10, 10), (200, 0, 210, 10))
    original = copy.deepcopy((a, b))
    result = compare(a, b)
    assert result["primary_count"] == result["secondary_count"] == 2
    assert result["matched"] == 1
    assert result["primary_only_indices"] == result["secondary_only_indices"] == [2]
    assert result["secondary_only"] == b[1:]
    assert result["secondary_detections"] == b
    assert len(result["secondary_detections"]) == 2, "Keep the matched box available for visual review too"
    assert (a, b) == original
    assert "reviewed" not in result and "accuracy" not in result


def test_matching_finds_augmenting_path_and_flags_ambiguous_neighbors():
    a = detections((0, 0, 10, 10), (4, 0, 14, 10))
    b = detections((2, 0, 12, 10), (-3, 0, 7, 10))
    result = compare(a, b)
    assert result["matched"] == 2
    assert result["pairs"] == [{"primary_index": 1, "secondary_index": 2},
                               {"primary_index": 2, "secondary_index": 1}]
    assert result["ambiguous_primary_indices"] == result["ambiguous_secondary_indices"] == [1, 2]


def test_center_match_tolerates_box_extent_but_not_unrelated_contained_objects():
    result = compare(detections((0, 0, 20, 20)), detections((5, 5, 15, 15)))
    assert result["matched"] == 1
    result = compare(detections((0, 0, 100, 100)), detections((45, 45, 55, 55)))
    assert result["matched"] == 0


def test_duplicates_are_one_to_one_and_empty_results_are_valid():
    result = compare(detections((0, 0, 10, 10)), detections((0, 0, 10, 10), (0, 0, 10, 10)))
    assert result["matched"] == 1 and len(result["secondary_only"]) == 1
    assert result["ambiguous_primary_indices"] == [1]
    assert compare([], [])["matched"] == 0
    assert compare([], detections((0, 0, 10, 10)))["secondary_only_indices"] == [1]


@pytest.mark.parametrize("box", [(0, 0, 0, 1), (0, 0, float("nan"), 1), (False, 0, 1, 1)])
def test_invalid_comparison_boxes_fail(box):
    with pytest.raises(ValueError, match="Invalid box"):
        compare(detections(box), [])


@pytest.mark.parametrize("fail", [False, True])
def test_composite_runs_same_inputs_serially_and_retains_primary_on_check_failure(tmp_path, monkeypatch, fail):
    engine = InferenceEngine()
    outer_count = engine.count
    calls, unloaded = [], []
    primary = {"count": 1, "detections": detections((0, 0, 10, 10)), "annotated_path": "original.jpg"}
    primary_times, secondary_times = [12.5, 23.75], [37.25, 46.5]
    refs, paths = [{"path": "reference", "boxes": [[0, 0, 10, 10]]}], ["target1", "target2"]
    clock = iter([1., 3.])
    monkeypatch.setattr("app.inference.time.perf_counter", lambda: next(clock))

    def count(model_id, references, image_paths, output_dir, confidence, label, *, reference_strategy="mean"):
        with engine._lock:
            assert references is refs and image_paths is paths
            calls.append((model_id, confidence, label, reference_strategy))
            if model_id != COUNTGD_MODEL_ID and fail:
                raise RuntimeError("secondary unavailable")
            elapsed = primary_times if model_id == COUNTGD_MODEL_ID else secondary_times
            return [{**copy.deepcopy(primary), "elapsed_ms": value} for value in elapsed]

    monkeypatch.setattr(engine, "count", count)
    monkeypatch.setattr(engine, "_unload_model", lambda: unloaded.append(True))
    results = outer_count(CROSSCHECK_MODEL_ID, refs, paths, str(tmp_path), .5, "nut")
    assert calls == [(COUNTGD_MODEL_ID, .5, "nut", "mean"), ("yoloe-26x-seg", .01, "nut", "per-object")]
    for i, result in enumerate(results):
        assert {key: result[key] for key in primary} == primary
        verification = result["verification"]
        assert verification["status"] == ("failed" if fail else "complete")
        assert verification["secondary_reference_strategy"] == "per-object"
        assert verification["primary_elapsed_ms"] == primary_times[i]
        assert verification["secondary_elapsed_ms"] == (None if fail else secondary_times[i])
        assert result["elapsed_ms"] == primary_times[i] + (0 if fail else secondary_times[i])
        assert verification["batch_wall_ms"] == 2000
        assert "reviewed" not in result
    assert bool(unloaded) == fail


def test_legacy_component_weights_do_not_expand_production_catalog(tmp_path, monkeypatch):
    from app import models

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("COUNTGD_SOURCE_DIR", str(tmp_path / "source"))
    monkeypatch.setattr(models, "COUNTGD_ASSETS", {"countgd_plusplus.pth": (1, "unused")})
    monkeypatch.setattr(models, "MODEL_ASSETS", {"yoloe-26x-seg.pt": (1, "unused")})
    (tmp_path / "source").mkdir()
    (tmp_path / "source/cfg_app.py").touch()
    (tmp_path / "countgd").mkdir()
    entry = lambda: models.list_models()[0]
    assert [m["id"] for m in models.list_models()] == [COUNTGD_MODEL_ID]
    assert not entry()["available"]
    (tmp_path / "countgd/countgd_plusplus.pth").write_bytes(b"x")
    assert entry()["available"]
    (tmp_path / "yoloe-26x-seg.pt").write_bytes(b"x")
    assert [m["id"] for m in models.list_models()] == [COUNTGD_MODEL_ID]
    assert entry()["available"] and entry()["default_confidence"] == .80
