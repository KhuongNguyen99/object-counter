"""Compare CountGD++ and SAM3Count on identical image bytes.

Run inside the app image with the model assets installed:
    python scripts/compare_counting_models.py --check-inputs
    python scripts/compare_counting_models.py --output-dir test-results/sam3count-comparison

The two models receive the same target and text. CountGD++ also receives its
visual exemplar; SAM3Count is text only. Thresholds are recorded per model.
"""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
COUNTGD_ID = "countgd-plusplus"
SAM3_ID = "sam3count"
ATTACHMENTS = Path(".codex-remote-attachments/01a14b6c-1e76-7570-b308-4ac2aa56b41d/e1ec69e0-b412-4636-a935-0b80fa699bcc")
CASES = (
    {
        "name": "original-4k-fitting", "target": Path("test-results/fitting-debug/target-original.jpg"),
        "label": "bolt with red cap", "countgd_confidence": 0.5,
        "reference": Path("test-results/fitting-debug/reference-original.jpg"),
        "reference_box": [1000, 1702, 1379, 2256], "truth": None,
    },
    {
        "name": "original-4k-rings", "target": Path("test-results/ring-debug-input/target-original.jpg"),
        "label": "white plastic ring", "countgd_confidence": 0.23,
        "reference": Path("test-results/ring-debug-input/reference-original.jpg"),
        "reference_box": [675, 1531, 1598, 2484],
        "truth": Path("test-results/model-comparison/original-ground-truth/ground-truth.json"),
    },
    {
        "name": "attachment-rings", "target": ATTACHMENTS / "2-6006.jpg",
        "label": "white plastic ring", "countgd_confidence": 0.23,
        "reference": ATTACHMENTS / "1-6005.jpg",
        # The separate attachment reference crop and local box in ring-debug-agent/attached-results/report.json.
        "reference_box": [294, 586, 436, 727],
        "truth": Path("test-results/ring-ground-truth/ground-truth.json"),
    },
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_info(path):
    path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing fixture: {path}")
    with Image.open(path) as image:
        width, height = ImageOps.exif_transpose(image).size
    return {"path": str(path), "sha256": sha256(path), "width": width, "height": height}


def prepare_cases(cases=CASES):
    prepared = []
    for case in cases:
        target = image_info(case["target"])
        reference = image_info(case["reference"])
        x1, y1, x2, y2 = case["reference_box"]
        if not (0 <= x1 < x2 <= reference["width"] and 0 <= y1 < y2 <= reference["height"]):
            raise ValueError(f"Reference box is outside {case['name']} reference image")
        truth = None
        if case["truth"] is not None:
            truth_path = ROOT / case["truth"]
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            if (truth.get("source_sha256") != target["sha256"]
                    or truth.get("image_width") != target["width"]
                    or truth.get("image_height") != target["height"]
                    or truth.get("count") != len(truth.get("rings", []))):
                raise ValueError(f"Ground truth does not match exact target bytes: {case['name']}")
            truth = {"path": str(truth_path), "sha256": sha256(truth_path), "data": truth}
        prepared.append({**case, "target_info": target, "reference_info": reference, "truth_info": truth})
    return prepared


def evaluate(detections, truth):
    """Match predicted boxes to approximate audited ring circles one to one."""
    rings = truth["rings"]
    boxes = [item["box"] for item in detections]
    if any(len(box) != 4 or not all(math.isfinite(value) for value in box)
           or box[2] <= box[0] or box[3] <= box[1] for box in boxes):
        raise ValueError("Detections must contain finite, nonempty XYXY boxes")

    def iou(left, right):
        intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
            0, min(left[3], right[3]) - max(left[1], right[1]))
        union = ((left[2] - left[0]) * (left[3] - left[1])
                 + (right[2] - right[0]) * (right[3] - right[1]) - intersection)
        return intersection / union if union else 0

    def match(edges):
        owner = {}

        def assign(prediction, seen):
            for target in edges[prediction]:
                if target in seen:
                    continue
                seen.add(target)
                if target not in owner or assign(owner[target], seen):
                    owner[target] = prediction
                    return True
            return False

        for prediction in range(len(edges)):
            assign(prediction, set())
        count = len(owner)
        paired_predictions = set(owner.values())
        return {"matched": count, "extra": len(boxes) - count, "missed": len(rings) - count,
                "precision": count / len(boxes) if boxes else None,
                "recall": count / len(rings) if rings else None,
                "missed_ids": [ring["id"] for index, ring in enumerate(rings) if index not in owner],
                "extra_indices": [detections[index].get("index", index + 1)
                                  for index in range(len(boxes)) if index not in paired_predictions],
                "pairs": [{"prediction": detections[prediction].get("index", prediction + 1),
                           "ring_id": rings[target]["id"]}
                          for target, prediction in sorted(owner.items())]}

    circles = [[ring["x"] - ring["radius"], ring["y"] - ring["radius"],
                ring["x"] + ring["radius"], ring["y"] + ring["radius"]] for ring in rings]
    overlap = [sorted((index for index, circle in enumerate(circles) if iou(box, circle) >= 0.5),
                      key=lambda index: -iou(box, circles[index])) for box in boxes]
    centers = []
    for box in boxes:
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        distances = [math.hypot(x - ring["x"], y - ring["y"]) / ring["radius"] for ring in rings]
        centers.append(sorted((index for index, value in enumerate(distances) if value <= 0.65),
                              key=lambda index: distances[index]))
    return {"expected_count": len(rings), "predicted_count": len(boxes),
            "absolute_count_error": abs(len(boxes) - len(rings)),
            "box_iou_0.5": match(overlap), "center_within_0.65_radius": match(centers),
            "limitation": "Ground-truth circles are approximate; visually review crowded overlaps."}


def run(prepared, output_dir, sam3_confidence, engine_factory=None):
    if not 0 <= sam3_confidence <= 1:
        raise ValueError("SAM3Count confidence must be between 0 and 1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty to avoid mixing comparison runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir()
    if engine_factory is None:
        sys.path.insert(0, str(ROOT))
        from app.inference import InferenceEngine
        engine_factory = InferenceEngine
    records = {case["name"]: {"target": case["target_info"], "label": case["label"],
                              "countgd_reference": {**case["reference_info"],
                                                    "box": case["reference_box"]},
                              "ground_truth": ({"status": "verified_for_exact_image",
                                                "path": case["truth_info"]["path"],
                                                "sha256": case["truth_info"]["sha256"],
                                                "count": case["truth_info"]["data"]["count"]}
                                               if case["truth_info"] else {"status": "unavailable"}),
                              "models": {}} for case in prepared}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "runtime": {"python": platform.python_version(), "platform": platform.platform()},
              "model_metadata": {},
              "notes": ["Each row uses identical target bytes and text across models.",
                        "CountGD++ uses a separate visual reference; SAM3Count uses text only.",
                        "The two ring photographs show the same physical arrangement and are not independent validation scenes.",
                        "This comparison uses exact raw image bytes; app JPEG normalization can change detections.",
                        "Wall time includes model loading on each model's first call. The adapter's elapsed_ms has a narrower model-specific scope.",
                        "Ground-truth localization uses approximate audited ring circles; inspect overlays for crowded overlaps."],
              "cases": records}
    partial_path = output_dir / "comparison.partial.json"
    # Establish a small verified result before either model processes the 4K scenes.
    run_order = sorted(prepared, key=lambda case: (case["truth_info"] is None,
                                                   case["target_info"]["width"] * case["target_info"]["height"]))
    for model_id in (COUNTGD_ID, SAM3_ID):
        engine = engine_factory()
        try:
            for index, case in enumerate(run_order):
                target = case["target_info"]
                if sha256(Path(target["path"])) != target["sha256"]:
                    raise ValueError(f"Target bytes changed during comparison: {case['name']}")
                confidence = case["countgd_confidence"] if model_id == COUNTGD_ID else sam3_confidence
                refs = ([{"path": case["reference_info"]["path"],
                          "boxes": [case["reference_box"]]}] if model_id == COUNTGD_ID else [])
                if refs and sha256(Path(refs[0]["path"])) != case["reference_info"]["sha256"]:
                    raise ValueError(f"Reference bytes changed during comparison: {case['name']}")
                started = time.perf_counter()
                result = engine.count(model_id, refs, [target["path"]], str(images_dir), confidence,
                                      case["label"])[0]
                wall_ms = round((time.perf_counter() - started) * 1000, 2)
                if (result["count"] != len(result["detections"])
                        or (result["width"], result["height"]) != (target["width"], target["height"])):
                    raise ValueError(f"Invalid inference result for {case['name']} / {model_id}")
                annotation = Path(result["annotated_path"])
                if not annotation.is_file():
                    raise FileNotFoundError(f"Missing annotation: {annotation}")
                destination = images_dir / f"{case['name']}-{model_id}.jpg"
                shutil.copyfile(annotation, destination)
                if annotation != destination:
                    annotation.unlink()
                record = {key: value for key, value in result.items() if key != "annotated_path"}
                record.update({"annotation": destination.relative_to(output_dir).as_posix(),
                               "annotation_sha256": sha256(destination),
                               "wall_ms_including_model_load_and_prompt_setup": wall_ms,
                               "model_cached_before_call": index > 0,
                               "confidence_threshold": confidence,
                               "prompt": {"text": case["label"],
                                          "visual_references": refs,
                                          "modality": "text+image" if refs else "text-only"}})
                if case["truth_info"]:
                    record["evaluation"] = evaluate(result["detections"], case["truth_info"]["data"])
                records[case["name"]]["models"][model_id] = record
                if model_id not in report["model_metadata"]:
                    from app import models
                    catalog = next((item for item in models.list_models() if item["id"] == model_id), None)
                    assets = (models.COUNTGD_ASSETS if model_id == COUNTGD_ID
                              else models.SAM3COUNT_ASSETS)
                    loaded_model = getattr(engine, "_model", None)
                    torch = sys.modules.get("torch")
                    report["model_metadata"][model_id] = {
                        "catalog": catalog,
                        "checkpoint_variant": ("SAM3Count-FT detection weights + SAM 3 base"
                                               if model_id == SAM3_ID else "CountGD++"),
                        "pinned_asset_sha256": {name: digest for name, (_, digest) in assets.items()},
                        "loaded_class": (f"{type(loaded_model).__module__}.{type(loaded_model).__qualname__}"
                                         if loaded_model is not None else None),
                        "device": engine._device() if hasattr(engine, "_device") else None,
                        "torch": getattr(torch, "__version__", None),
                        "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None),
                        "gpu": (torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else None),
                    }
                partial_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"{case['name']} {model_id}: {result['count']} in {wall_ms:.0f} ms", flush=True)
        finally:
            engine.close()
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_html(report, output_dir)
    partial_path.unlink()
    return report


def write_html(report, output_dir):
    cards = []
    rows = []
    display_cases = sorted(report["cases"].items(), key=lambda item: (
        item[1]["ground_truth"]["status"] == "unavailable",
        item[1]["target"]["width"] * item[1]["target"]["height"]))
    for name, case in display_cases:
        truth = case["ground_truth"]
        columns = []
        summary = []
        for model_id, title in ((COUNTGD_ID, "CountGD++"), (SAM3_ID, "SAM3Count")):
            result = case["models"][model_id]
            score = result.get("evaluation")
            metrics = (f"Matched {score['box_iou_0.5']['matched']} · Extra {score['box_iou_0.5']['extra']} · "
                       f"Missed {score['box_iou_0.5']['missed']} · Count error {score['absolute_count_error']}"
                       if score else "No verified ground truth for this image")
            annotation = output_dir / result["annotation"]
            photo = base64.b64encode(annotation.read_bytes()).decode("ascii")
            modality = result["prompt"]["modality"]
            summary.append(f"<td><strong>{result['count']}</strong><br>"
                           + (f"{score['box_iou_0.5']['matched']} matched · "
                              f"{score['box_iou_0.5']['extra']} extra · "
                              f"{score['box_iou_0.5']['missed']} missed<br>" if score else "unverified<br>")
                           + f"{result.get('elapsed_ms', 0) / 1000:.2f} s adapter elapsed</td>")
            timing = (f"wall {result['wall_ms_including_model_load_and_prompt_setup']:.0f} ms "
                      f"({'cached' if result['model_cached_before_call'] else 'cold'}) · "
                      f"adapter elapsed {result.get('elapsed_ms', 'unavailable')} ms")
            columns.append(f"<article><h3>{title}: {result['count']}</h3>"
                           f"<p>{escape(metrics)}</p><p>{escape(modality)} · threshold {result['confidence_threshold']} · "
                           f"{escape(timing)}</p>"
                           f"<img src='data:image/jpeg;base64,{photo}' alt='{escape(name)} annotated by {title}'></article>")
        count_delta = case["models"][SAM3_ID]["count"] - case["models"][COUNTGD_ID]["count"]
        rows.append(f"<tr><th scope='row'>{escape(name)}</th><td>{truth.get('count', 'unavailable')}</td>"
                    + "".join(summary) + "</tr>")
        cards.append(f"<section><h2>{escape(name)}</h2><p>Target SHA-256 {escape(case['target']['sha256'])} · "
                     f"{case['target']['width']}×{case['target']['height']} · "
                     f"Ground truth: {escape(truth['status'])} · SAM3Count minus CountGD++ count: {count_delta:+d}</p>"
                     f"<div class='pair'>{''.join(columns)}</div></section>")
    html = ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            "<title>Counting model comparison</title><style>body{font:16px system-ui;margin:2rem auto;max-width:1500px;padding:0 1rem;background:#10161b;color:#eef4f7}"
            "section{border-top:1px solid #52616b;padding:1rem 0 2rem}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}"
            "article{background:#1d2931;padding:1rem;border-radius:9px}img{display:block;width:100%;height:auto}p{overflow-wrap:anywhere;line-height:1.45}"
            "table{border-collapse:collapse;width:100%;min-width:700px}th,td{border-bottom:1px solid #52616b;text-align:left;padding:.75rem;vertical-align:top}"
            ".table-scroll{overflow-x:auto}small{color:#b6c4cb}"
            "@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>"
            "<h1>SAM3Count vs CountGD++</h1><p>Same target bytes and text per case. CountGD++ also uses a visual exemplar; "
            "SAM3Count uses text only. The two ring photos show the same physical arrangement and are not independent validation scenes. "
            "These runs use exact raw image bytes; app JPEG normalization can change detections. First call per model includes loading.</p>"
            "<div class='table-scroll'><table><thead><tr><th>Image</th><th>Verified rings</th><th>CountGD++</th><th>SAM3Count-FT</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div><p><small>Matched, extra, and missed use one-to-one box IoU ≥ 0.5 against approximate audited ring locations. "
            "Adapter elapsed excludes model startup; cold wall time appears under each image. Fitting has no verified total.</small></p>"
            + "".join(cards) + "</html>")
    (output_dir / "comparison.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "test-results/sam3count-comparison")
    parser.add_argument("--sam3-confidence", type=float, default=0.35,
                        help="SAM3Count object score threshold; use the adapter's documented default")
    parser.add_argument("--check-inputs", action="store_true", help="Validate image and ground-truth provenance without inference")
    args = parser.parse_args()
    prepared = prepare_cases()
    if args.check_inputs:
        for case in prepared:
            print(f"{case['name']}: {case['target_info']['width']}x{case['target_info']['height']} "
                  f"sha256={case['target_info']['sha256']} "
                  f"truth={case['truth_info']['data']['count'] if case['truth_info'] else 'unavailable'}")
        return
    run(prepared, args.output_dir, args.sam3_confidence)


if __name__ == "__main__":
    main()
