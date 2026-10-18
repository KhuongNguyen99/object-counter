"""The comparison must preserve prompt modality and ground-truth provenance."""

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.compare_counting_models import COUNTGD_ID, SAM3_ID, prepare_cases, run


def fixture_case(tmp_path, with_truth=True):
    target = tmp_path / "target.jpg"
    reference = tmp_path / "reference.jpg"
    Image.new("RGB", (100, 60), "white").save(target)
    Image.new("RGB", (100, 60), "gray").save(reference)
    truth_path = None
    if with_truth:
        truth_path = tmp_path / "truth.json"
        truth_path.write_text(json.dumps({
            "source_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "image_width": 100, "image_height": 60, "count": 2,
            "rings": [{"id": 1, "x": 20, "y": 20, "radius": 10},
                      {"id": 2, "x": 60, "y": 20, "radius": 10}],
        }), encoding="utf-8")
    return {"name": "test-rings" if with_truth else "unverified-fitting",
            "target": target, "reference": reference, "reference_box": [10, 10, 30, 30],
            "label": "white ring", "countgd_confidence": 0.23, "truth": truth_path}


def test_ground_truth_requires_exact_target_bytes_and_dimensions(tmp_path):
    case = fixture_case(tmp_path)
    assert prepare_cases([case])[0]["truth_info"]["data"]["count"] == 2
    truth_path = case["truth"]
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth["source_sha256"] = "0" * 64
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    with pytest.raises(ValueError, match="exact target bytes"):
        prepare_cases([case])


def test_same_image_comparison_uses_model_native_prompts_and_instance_matching(tmp_path):
    cases = prepare_cases([fixture_case(tmp_path)])
    calls = []

    class FakeEngine:
        def count(self, model_id, refs, targets, output_dir, confidence, label):
            calls.append((model_id, refs, targets, confidence, label))
            annotation = Path(output_dir) / f"fake-{model_id}.jpg"
            Image.new("RGB", (100, 60), "blue").save(annotation)
            boxes = ([[10, 10, 30, 30], [50, 10, 70, 30]] if model_id == COUNTGD_ID
                     else [[10, 10, 30, 30], [10, 10, 30, 30]])
            return [{"count": 2, "detections": [{"index": i, "box": box, "confidence": 0.9}
                                                 for i, box in enumerate(boxes, 1)],
                     "width": 100, "height": 60, "elapsed_ms": 4,
                     "annotated_path": str(annotation)}]

        def close(self):
            pass

    output = tmp_path / "comparison"
    report = run(cases, output, sam3_confidence=0.5, engine_factory=FakeEngine)
    assert [call[0] for call in calls] == [COUNTGD_ID, SAM3_ID]
    assert calls[0][1] and calls[1][1] == []
    assert calls[0][2] == calls[1][2]
    assert calls[0][4] == calls[1][4] == "white ring"
    assert [call[3] for call in calls] == [0.23, 0.5]
    models = report["cases"]["test-rings"]["models"]
    assert models[COUNTGD_ID]["evaluation"]["box_iou_0.5"]["matched"] == 2
    assert models[SAM3_ID]["evaluation"]["box_iou_0.5"]["extra"] == 1
    assert models[SAM3_ID]["evaluation"]["box_iou_0.5"]["missed"] == 1
    assert models[SAM3_ID]["evaluation"]["box_iou_0.5"]["extra_indices"] == [2]
    assert models[SAM3_ID]["evaluation"]["absolute_count_error"] == 0
    assert "data:image/jpeg;base64," in (output / "comparison.html").read_text(encoding="utf-8")
    assert json.loads((output / "comparison.json").read_text(encoding="utf-8"))["cases"]["test-rings"]["target"]["sha256"]
    assert report["cases"]["test-rings"]["countgd_reference"]["sha256"]
    assert not (output / "comparison.partial.json").exists()


def test_completed_model_result_survives_later_failure(tmp_path):
    case = prepare_cases([fixture_case(tmp_path, with_truth=False)])

    class FailingEngine:
        def count(self, model_id, refs, targets, output_dir, confidence, label):
            if model_id == SAM3_ID:
                raise RuntimeError("SAM3Count unavailable")
            annotation = Path(output_dir) / "first.jpg"
            Image.new("RGB", (100, 60), "blue").save(annotation)
            return [{"count": 0, "detections": [], "width": 100, "height": 60,
                     "elapsed_ms": 2, "annotated_path": str(annotation)}]

        def close(self):
            pass

    output = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="SAM3Count unavailable"):
        run(case, output, sam3_confidence=0.35, engine_factory=FailingEngine)
    partial = json.loads((output / "comparison.partial.json").read_text(encoding="utf-8"))
    assert partial["cases"]["unverified-fitting"]["models"][COUNTGD_ID]["count"] == 0
    assert partial["cases"]["unverified-fitting"]["ground_truth"]["status"] == "unavailable"
