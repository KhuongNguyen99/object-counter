"""Run reproducible, center-annotated hardware counting benchmarks.

Manifest paths are relative to the manifest file. Ground truth is used only
after inference. A center inside a predicted box is a possible match; maximum
one-to-one matching prevents a prediction from covering multiple objects.
Center annotations cannot establish box quality or object extent.

Example:
    python scripts/benchmark_reliability.py --manifest test-results/reliability-benchmark/data/manifest.json --output-dir test-results/reliability-benchmark/runs --model yoloe-26x-seg --model countgd-plusplus --threshold 0.23 --split test --resume
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
MODELS = tuple(f"yoloe-26{size}-seg" for size in "nsmlx") + ("countgd-plusplus",)
DEFAULT_THRESHOLDS = (0.10, 0.15, 0.23, 0.30, 0.40, 0.50)
MATCHING = "maximum one-to-one matching of ground-truth centers inside predicted boxes"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _image(path: Path, expected_hash: str | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256(path)
    if expected_hash is not None and expected_hash != digest:
        raise ValueError(f"Ground truth is not bound to the current image bytes: {path}")
    with Image.open(path) as image:
        width, height = ImageOps.exif_transpose(image).size
    return {"path": str(path), "sha256": digest, "width": width, "height": height}


def _point(point, width: int, height: int) -> list[float]:
    if (not isinstance(point, (list, tuple)) or len(point) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in point)):
        raise ValueError(f"Invalid ground-truth center: {point!r}")
    x, y = map(float, point)
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Ground-truth center is outside {width}x{height}: {point!r}")
    return [x, y]


def _box(box, width: int | None = None, height: int | None = None) -> list[float]:
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in box)):
        raise ValueError(f"Invalid box: {box!r}")
    x1, y1, x2, y2 = map(float, box)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Nonpositive-area box: {box!r}")
    if width is not None and not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Reference box is outside {width}x{height}: {box!r}")
    return [x1, y1, x2, y2]


def _resolved(base: Path, name: str) -> Path:
    return (base / name).resolve()


def prepare_manifest(path: Path) -> dict:
    path = path.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("targets"), list) or not isinstance(data.get("references"), dict):
        raise ValueError("Manifest needs targets[] and references{}")
    base = path.parent
    targets = []
    ids = set()
    for raw in data["targets"]:
        target_id, category = raw["id"], raw["category"]
        if not isinstance(target_id, str) or not target_id or target_id in ids:
            raise ValueError(f"Duplicate or empty target id: {target_id!r}")
        ids.add(target_id)
        image = _image(_resolved(base, raw["path"]), raw.get("source_sha256"))
        boxes = ([_box(box, image["width"], image["height"]) for box in raw["boxes"]]
                 if "boxes" in raw else None)
        points = ([_point(point, image["width"], image["height"]) for point in raw["points"]]
                  if "points" in raw else [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                                           for box in boxes or []])
        if boxes is not None and len(boxes) != len(points):
            raise ValueError(f"Target {target_id} has different point and box counts")
        targets.append({"id": target_id, "category": category, "split": raw["split"],
                        "image": image, "points": points, "boxes": boxes})

    def references(groups: dict) -> dict:
        prepared = {}
        for category, items in groups.items():
            if not isinstance(items, list) or not items:
                raise ValueError(f"No references for {category}")
            prepared[category] = []
            for item in items:
                image = _image(_resolved(base, item["path"]), item.get("source_sha256"))
                boxes = [_box(box, image["width"], image["height"]) for box in item["boxes"]]
                if not boxes:
                    raise ValueError(f"No reference boxes in {image['path']}")
                prepared[category].append({**image, "boxes": boxes})
        return prepared

    variants = {"default": references(data["references"])}
    for name, groups in data.get("reference_variants", {}).items():
        if name == "default":
            raise ValueError("Reference variant 'default' is reserved")
        variants[name] = references(groups)
    target_hashes = {target["image"]["sha256"] for target in targets}
    for variant, groups in variants.items():
        for category, items in groups.items():
            for item in items:
                if item["sha256"] in target_hashes:
                    raise ValueError(f"Reference image overlaps a benchmark target: {variant}/{category}/{item['path']}")
    return {"path": str(path), "sha256": sha256(path), "targets": targets,
            "references": variants, "description": data.get("description", {})}


def _match(neighbors: list[list[int]], detection_count: int) -> dict:
    """Find a maximum-cardinality bipartite match, with deterministic indices."""
    match_point = [None] * len(neighbors)
    match_box = [None] * detection_count
    for start in range(len(neighbors)):
        queue = deque([start])
        seen_points, seen_boxes, parent = {start}, set(), {}
        free_box = None
        while queue and free_box is None:
            point_index = queue.popleft()
            for box_index in neighbors[point_index]:
                if box_index in seen_boxes:
                    continue
                seen_boxes.add(box_index)
                parent[box_index] = point_index
                previous_point = match_box[box_index]
                if previous_point is None:
                    free_box = box_index
                    break
                if previous_point not in seen_points:
                    seen_points.add(previous_point)
                    queue.append(previous_point)
        while free_box is not None:
            point_index = parent[free_box]
            old_box = match_point[point_index]
            match_point[point_index] = free_box
            match_box[free_box] = point_index
            free_box = old_box
    matched = sum(index is not None for index in match_point)
    missed = len(neighbors) - matched
    extra = detection_count - matched
    return {"matched": matched, "missed": missed, "extra": extra,
            "precision": matched / detection_count if detection_count else float(not neighbors),
            "recall": matched / len(neighbors) if neighbors else 1.0,
            "exact_image": missed == 0 and extra == 0,
            "matched_pairs": [[i, j] for i, j in enumerate(match_point) if j is not None],
            "missed_point_indices": [i for i, j in enumerate(match_point) if j is None],
            "extra_detection_indices": [i for i, j in enumerate(match_box) if j is None]}


def evaluate(points: list[list[float]], detections: list[dict], gt_boxes=None) -> dict:
    boxes = [_box(detection["box"]) for detection in detections]
    neighbors = [[j for j, (x1, y1, x2, y2) in enumerate(boxes)
                  if x1 <= x <= x2 and y1 <= y <= y2] for x, y in points]
    result = _match(neighbors, len(boxes))
    if gt_boxes is not None:
        if len(gt_boxes) != len(points):
            raise ValueError("Ground-truth points and boxes have different lengths")
        gt_boxes = [_box(box) for box in gt_boxes]

        def iou(left, right):
            x1, y1 = max(left[0], right[0]), max(left[1], right[1])
            x2, y2 = min(left[2], right[2]), min(left[3], right[3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            left_area = (left[2] - left[0]) * (left[3] - left[1])
            right_area = (right[2] - right[0]) * (right[3] - right[1])
            return intersection / (left_area + right_area - intersection)

        result["box_iou_0.5"] = _match([[j for j, box in enumerate(boxes) if iou(gt, box) >= 0.5]
                                         for gt in gt_boxes], len(boxes))
    return result


def _settings() -> dict:
    import app

    adapter_dir = Path(app.__file__).resolve().parent
    return {"DEVICE": os.environ.get("DEVICE", "auto"),
            "IMGSZ": os.environ.get("IMGSZ", "640"),
            "MAX_DETECTIONS": os.environ.get("MAX_DETECTIONS", "3000"),
            "COUNTGD_SOURCE_DIR": os.environ.get("COUNTGD_SOURCE_DIR", "/opt/countgd"),
            "build": os.environ.get("BENCHMARK_BUILD", "unspecified"),
            "adapter_hashes": {name: sha256(adapter_dir / name)
                               for name in ("inference.py", "countgd.py", "tiling.py")
                               if (adapter_dir / name).is_file()}}


def _job_key(job: dict) -> str:
    identity = {key: job[key] for key in ("model", "threshold", "variant", "target_id", "category", "split",
                                          "target_sha256", "points", "gt_boxes", "references", "label", "settings")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _summary(records: dict) -> list[dict]:
    groups = {}
    for record in records.values():
        if record["status"] != "ok":
            continue
        group_key = (record["model"], record["threshold"], record["variant"], record["category"], record["split"])
        group = groups.setdefault(group_key, {"images": 0, "exact_images": 0, "matched": 0, "missed": 0,
                                              "extra": 0, "elapsed_ms": 0.0})
        metrics = record["metrics"]
        group["images"] += 1
        group["exact_images"] += metrics["exact_image"]
        for key in ("matched", "missed", "extra"):
            group[key] += metrics[key]
        group["elapsed_ms"] += record["prediction"]["elapsed_ms"]
    result = []
    for (model, threshold, variant, category, split), group in sorted(groups.items()):
        matched = group["matched"]
        predictions = matched + group["extra"]
        truth = matched + group["missed"]
        result.append({"model": model, "threshold": threshold, "variant": variant,
                       "category": category, "split": split, **group,
                       "precision": matched / predictions if predictions else float(not truth),
                       "recall": matched / truth if truth else 1.0,
                       "exact_image_rate": group["exact_images"] / group["images"],
                       "mean_elapsed_ms": group["elapsed_ms"] / group["images"]})
    return result


def run(manifest: dict, output_dir: Path, *, models=MODELS, thresholds=DEFAULT_THRESHOLDS,
        categories=None, splits=None, targets=None, variants=("default",), resume=False,
        engine_factory=None) -> dict:
    """Run selected jobs serially; successful matching records are safe to resume."""
    if not models or not thresholds or not variants:
        raise ValueError("Select at least one model, threshold, and reference variant")
    if any(model not in MODELS for model in models):
        raise ValueError("Unknown model requested")
    if any(not math.isfinite(threshold) or not 0 <= threshold <= 1 for threshold in thresholds):
        raise ValueError("Thresholds must be finite values between 0 and 1")
    if any(variant not in manifest["references"] for variant in variants):
        raise ValueError("Unknown reference variant requested")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "results.json"
    if resume and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("manifest_sha256") != manifest["sha256"]:
            raise ValueError("Manifest changed since these results; use a new output directory")
    else:
        report = {"schema_version": 1, "manifest_path": manifest["path"],
                  "manifest_sha256": manifest["sha256"], "matching": MATCHING,
                  "limitation": "Center-only ground truth cannot assess full object extent or box quality.",
                  "python": platform.python_version(), "records": {}, "summary": []}
        _write_report(report_path, report)
    settings = _settings()
    jobs = []
    for target in manifest["targets"]:
        if ((categories and target["category"] not in categories)
                or (splits and target["split"] not in splits)
                or (targets and target["id"] not in targets)):
            continue
        for variant in variants:
            refs = manifest["references"][variant].get(target["category"])
            if not refs:
                raise ValueError(f"No {variant} references for {target['category']}")
            for model in models:
                for threshold in thresholds:
                    job = {"model": model, "threshold": float(threshold), "variant": variant,
                           "target_id": target["id"], "category": target["category"], "split": target["split"],
                           "target_sha256": target["image"]["sha256"], "points": target["points"],
                           "gt_boxes": target["boxes"],
                           "references": [{"sha256": ref["sha256"], "boxes": ref["boxes"]} for ref in refs],
                           "label": manifest["description"].get(target["category"], target["category"]),
                           "settings": settings}
                    jobs.append((job, target, refs))
    if not jobs:
        raise ValueError("Filters selected no benchmark jobs")
    # Keep checkpoints resident while comparing images; wall time records the first cold load.
    jobs.sort(key=lambda item: (item[0]["model"], item[0]["variant"], item[0]["threshold"], item[0]["target_id"]))
    if engine_factory is None:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from app.inference import InferenceEngine
        engine_factory = InferenceEngine
    engine = None
    failed = 0
    try:
        for index, (job, target, refs) in enumerate(jobs, 1):
            key = _job_key(job)
            if resume and report["records"].get(key, {}).get("status") == "ok":
                print(f"[{index}/{len(jobs)}] skip {job['target_id']} {job['model']} {job['threshold']:.2f} {job['variant']}", flush=True)
                continue
            print(f"[{index}/{len(jobs)}] run {job['target_id']} {job['model']} {job['threshold']:.2f} {job['variant']}", flush=True)
            started = datetime.now(timezone.utc).isoformat()
            wall_start = time.perf_counter()
            try:
                if engine is None:
                    engine = engine_factory()
                response = engine.count(job["model"],
                                        [{"path": ref["path"], "boxes": ref["boxes"]} for ref in refs],
                                        [target["image"]["path"]], str(output_dir / "annotated"),
                                        confidence=job["threshold"], label=job["label"])
                if not isinstance(response, list) or len(response) != 1:
                    raise ValueError("Engine must return exactly one result for one target")
                prediction = response[0]
                detections = prediction["detections"]
                if prediction["count"] != len(detections):
                    raise ValueError("Engine count disagrees with detection boxes")
                metrics = evaluate(target["points"], detections, target["boxes"])
                record = {**job, "status": "ok", "target": target["image"],
                          "reference_images": refs, "prediction": prediction, "metrics": metrics,
                          "wall_ms": round((time.perf_counter() - wall_start) * 1000, 2),
                          "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat()}
                print(f"  matched={metrics['matched']} missed={metrics['missed']} extra={metrics['extra']}", flush=True)
            except Exception as exc:
                failed += 1
                record = {**job, "status": "error", "target": target["image"],
                          "reference_images": refs, "error": {"type": type(exc).__name__, "message": str(exc)},
                          "wall_ms": round((time.perf_counter() - wall_start) * 1000, 2),
                          "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat()}
                print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
                if engine is not None:
                    try:
                        engine.close()
                    except Exception as close_error:
                        record["close_error"] = {"type": type(close_error).__name__, "message": str(close_error)}
                    finally:
                        engine = None
            report["records"][key] = record
            report["summary"] = _summary(report["records"])
            _write_report(report_path, report)
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                failed += 1
                report["engine_close_error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["selected_jobs"] = len(jobs)
    report["failed_jobs"] = failed
    _write_report(report_path, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "test-results/reliability-benchmark/data/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "test-results/reliability-benchmark/runs")
    parser.add_argument("--model", action="append", choices=MODELS, help="Repeat to select models; default is all")
    parser.add_argument("--threshold", action="append", type=float, help="Repeat to select thresholds; default is sweep")
    parser.add_argument("--target", action="append", help="Repeat to select target IDs")
    parser.add_argument("--category", action="append", help="Repeat to select categories")
    parser.add_argument("--split", action="append", help="Repeat to select splits")
    parser.add_argument("--reference-variant", action="append", help="Named manifest reference variant; default is default")
    parser.add_argument("--resume", action="store_true", help="Skip only successful records with matching inputs and settings")
    args = parser.parse_args(argv)
    manifest = prepare_manifest(args.manifest)
    report = run(manifest, args.output_dir, models=args.model or MODELS,
                 thresholds=args.threshold or DEFAULT_THRESHOLDS,
                 categories=args.category, splits=args.split, targets=args.target,
                 variants=args.reference_variant or ("default",), resume=args.resume)
    print(f"Saved {report['selected_jobs']} selected jobs; {report['failed_jobs']} failed: {args.output_dir / 'results.json'}", flush=True)
    return int(report["failed_jobs"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
