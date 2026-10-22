"""CPU-only checks for benchmark provenance, matching, and resume behavior."""

import hashlib
import json

from PIL import Image
import pytest

from scripts.benchmark_reliability import evaluate, prepare_manifest, run


def fixture_manifest(tmp_path, *, boxes=False):
    target = tmp_path / "target.png"
    reference = tmp_path / "reference.png"
    Image.new("RGB", (100, 60), "white").save(target)
    Image.new("RGB", (100, 60), "blue").save(reference)
    item = {"id": "nuts-1", "category": "nuts", "split": "test", "path": "target.png",
            "source_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "points": [[20, 20], [60, 20]]}
    if boxes:
        item["boxes"] = [[10, 10, 30, 30], [50, 10, 70, 30]]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"targets": [item],
                                         "references": {"nuts": [{"path": "reference.png", "boxes": [[5, 5, 25, 25]]}]},
                                         "description": {"nuts": "hex nuts"}}), encoding="utf-8")
    return manifest_path


def test_maximum_one_to_one_matching_exposes_equal_count_miss_and_extra():
    detections = [{"box": [10, 10, 30, 30]}, {"box": [10, 10, 30, 30]}]
    score = evaluate([[20, 20], [60, 20]], detections)
    assert (score["matched"], score["missed"], score["extra"]) == (1, 1, 1)
    assert score["exact_image"] is False
    assert score["precision"] == score["recall"] == 0.5
    assert score["missed_point_indices"] == [1]
    # The first object has two possible boxes, the second only one. The augmenting
    # path must reassign the first object to achieve a cardinality-two match.
    score = evaluate([[20, 20], [60, 20]],
                     [{"box": [10, 10, 70, 30]}, {"box": [10, 10, 30, 30]}])
    assert score["matched"] == 2
    assert score["matched_pairs"] == [[0, 1], [1, 0]]


def test_optional_box_iou_is_reported_separately():
    score = evaluate([[20, 20]], [{"box": [0, 0, 40, 40]}], [[10, 10, 30, 30]])
    assert score["matched"] == 1
    assert score["box_iou_0.5"]["matched"] == 0
    assert "box_iou_0.5" not in evaluate([[20, 20]], [{"box": [0, 0, 40, 40]}])


def test_manifest_checks_byte_binding_and_reference_leakage(tmp_path):
    path = fixture_manifest(tmp_path)
    prepared = prepare_manifest(path)
    assert prepared["targets"][0]["image"]["sha256"]
    Image.new("RGB", (100, 60), "red").save(tmp_path / "target.png")
    with pytest.raises(ValueError, match="current image bytes"):
        prepare_manifest(path)
    path = fixture_manifest(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["references"]["nuts"][0]["path"] = "target.png"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps"):
        prepare_manifest(path)


def test_runner_records_metrics_and_resumes_only_successes(tmp_path):
    manifest = prepare_manifest(fixture_manifest(tmp_path, boxes=True))
    calls = []
    engines = []

    class FakeEngine:
        def __init__(self):
            self.closed = False
            engines.append(self)

        def count(self, model_id, refs, paths, output_dir, confidence, label):
            calls.append((model_id, refs, paths, confidence, label))
            if confidence == 0.2 and len([call for call in calls if call[3] == 0.2]) == 1:
                raise RuntimeError("transient")
            boxes = ([[10, 10, 30, 30], [50, 10, 70, 30]] if confidence == 0.1
                     else [[10, 10, 30, 30], [10, 10, 30, 30]])
            return [{"count": 2, "detections": [{"box": box, "confidence": 0.8} for box in boxes],
                     "width": 100, "height": 60, "elapsed_ms": 3.5,
                     "annotated_path": "fake.jpg"}]

        def close(self):
            self.closed = True

    output = tmp_path / "results"
    first = run(manifest, output, models=["yoloe-26x-seg"], thresholds=[0.1, 0.2], engine_factory=FakeEngine)
    assert first["failed_jobs"] == 1
    assert sorted(record["status"] for record in first["records"].values()) == ["error", "ok"]
    assert all(engine.closed for engine in engines)
    assert len(calls) == 2
    assert calls[0][1][0]["path"] == str(tmp_path / "reference.png")
    assert calls[0][2] == [str(tmp_path / "target.png")]
    assert calls[0][4] == "hex nuts"
    successful = next(record for record in first["records"].values() if record["status"] == "ok")
    assert successful["metrics"]["matched"] == 2
    assert successful["metrics"]["box_iou_0.5"]["matched"] == 2
    assert successful["target"]["sha256"] and successful["reference_images"][0]["sha256"]
    second = run(manifest, output, models=["yoloe-26x-seg"], thresholds=[0.1, 0.2],
                 resume=True, engine_factory=FakeEngine)
    assert second["failed_jobs"] == 0
    assert len(calls) == 3
    assert second["summary"][0]["images"] == 1
    assert sorted(record["status"] for record in second["records"].values()) == ["ok", "ok"]
    assert json.loads((output / "results.json").read_text(encoding="utf-8"))["failed_jobs"] == 0


def test_reference_variants_and_filters_select_only_requested_jobs(tmp_path):
    path = fixture_manifest(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["reference_variants"] = {"alternate": {"nuts": [{"path": "reference.png", "boxes": [[30, 10, 50, 30]]}]}}
    path.write_text(json.dumps(data), encoding="utf-8")
    manifest = prepare_manifest(path)
    calls = []

    class FakeEngine:
        def count(self, model_id, refs, paths, output_dir, confidence, label):
            calls.append(refs[0]["boxes"])
            return [{"count": 0, "detections": [], "elapsed_ms": 1}]

        def close(self):
            pass

    report = run(manifest, tmp_path / "out", models=["countgd-plusplus"], thresholds=[0.23],
                 categories=["nuts"], splits=["test"], targets=["nuts-1"],
                 variants=["alternate"], engine_factory=FakeEngine)
    assert calls == [[[30.0, 10.0, 50.0, 30.0]]]
    assert report["selected_jobs"] == 1
    assert report["summary"][0]["recall"] == 0
