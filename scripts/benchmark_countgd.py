"""Benchmark CountGD++ reference caching and check every prediction for drift.

Run from /app in the GPU image after copying the fitting-debug images there.
Each timed trial includes reference preparation, image loading, merging, and
annotation. Model loading and one uncached tile warmup are reported separately.
"""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TOLERANCE = 1e-5


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    fixture = Path("test-results/fitting-debug")
    parser.add_argument("--reference", type=Path, default=fixture / "reference-original.jpg")
    parser.add_argument("--target", type=Path, default=fixture / "target-original.jpg")
    parser.add_argument("--reference-box", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        default=[1039, 1775, 1344, 2170])
    parser.add_argument("--label", default="Bolt with red cap")
    parser.add_argument("--confidence", type=float, default=.5)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, or a CUDA device number")
    parser.add_argument("--model-dir", type=Path, default=Path("models/countgd"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmark-countgd"))
    parser.add_argument("--repeats", type=int, default=3, help="uncached/cached pairs (1 is useful on CPU)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not math.isfinite(args.confidence) or not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    x1, y1, x2, y2 = args.reference_box
    if not 0 <= x1 < x2 or not 0 <= y1 < y2:
        parser.error("--reference-box must have positive area and nonnegative coordinates")
    for name in ("reference", "target"):
        if not getattr(args, name).is_file():
            parser.error(f"--{name} image does not exist: {getattr(args, name)}")
    return args


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def metadata(args, engine, torch):
    source = Path(os.environ.get("COUNTGD_SOURCE_DIR", "/opt/countgd"))
    def stamp(name):
        path = source / name
        return path.read_text(encoding="utf-8").strip() if path.is_file() else None

    hardware = {"machine": platform.machine(), "processor": platform.processor(),
                "cpu_count": os.cpu_count()}
    if engine.device.type == "cuda":
        gpu = torch.cuda.get_device_properties(engine.device)
        hardware["gpu"] = {"name": gpu.name, "total_memory_bytes": gpu.total_memory,
                           "compute_capability": f"{gpu.major}.{gpu.minor}"}
    return {
        "reference": str(args.reference.resolve()), "target": str(args.target.resolve()),
        "reference_box": args.reference_box, "label": args.label,
        "confidence": args.confidence, "device": str(engine.device),
        "model_dtype": str(next(engine.model.parameters()).dtype),
        "torch_threads": torch.get_num_threads(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "model_dir": str(args.model_dir.resolve()), "repeats": args.repeats,
        "tolerance_normalized": TOLERANCE,
        "source_dir": str(source.resolve()), "source_commit": stamp(".counter-source-commit"),
        "source_patch": stamp(".counter-source-patches"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                     "torchvision": version("torchvision"), "transformers": version("transformers"),
                     "pillow": version("pillow"), "numpy": version("numpy")},
        "hardware": hardware,
    }


def normalized_tiles(captured):
    """Sort each tile's raw normalized cxcywh boxes and scores stably."""
    return [sorted(((tuple(map(float, box)), float(score))
                    for box, score in zip(boxes, scores, strict=True)),
                   key=lambda item: (item[0], item[1]))
            for boxes, scores in captured]


def compare(baseline, candidate, width, height):
    original_tiles, original_result, original_raw = baseline
    tiles, result, raw = candidate
    max_raw_box = max_raw_score = max_merged_box = max_merged_score = 0.0
    raw_counts_equal = len(tiles) == len(original_tiles)
    for before, after in zip(original_tiles, tiles):
        raw_counts_equal &= len(before) == len(after)
        for (box_a, score_a), (box_b, score_b) in zip(before, after):
            max_raw_box = max(max_raw_box, *(abs(a - b) for a, b in zip(box_a, box_b)))
            max_raw_score = max(max_raw_score, abs(score_a - score_b))
    merged_a, merged_b = original_result["detections"], result["detections"]
    merged_counts_equal = (original_result["count"] == result["count"]
                           and len(merged_a) == len(merged_b))
    indices_equal = [d["index"] for d in merged_a] == [d["index"] for d in merged_b]
    for before, after in zip(merged_a, merged_b):
        max_merged_box = max(max_merged_box, *(abs(a - b) / scale for a, b, scale in
                             zip(before["box"], after["box"], (width, height, width, height))))
        max_merged_score = max(max_merged_score,
                               abs(before["confidence"] - after["confidence"]))
    byte_exact = (len(original_raw) == len(raw) and all(
        a[0].dtype == b[0].dtype and a[0].shape == b[0].shape
        and a[1].dtype == b[1].dtype and a[1].shape == b[1].shape
        and a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()
        for a, b in zip(original_raw, raw)))
    passed = (raw_counts_equal and merged_counts_equal and indices_equal
              and max(max_raw_box, max_raw_score, max_merged_box, max_merged_score) <= TOLERANCE)
    return {"passed": passed, "raw_tile_counts_equal": bool(raw_counts_equal),
            "merged_counts_equal": merged_counts_equal, "merged_indices_equal": indices_equal,
            "raw_byte_exact": byte_exact, "max_raw_box_error_normalized": max_raw_box,
            "max_raw_score_error": max_raw_score,
            "max_merged_box_error_normalized": max_merged_box,
            "max_merged_score_error": max_merged_score}


def main():
    args = arguments()
    from app.countgd import CountGDEngine, _exemplar_image
    from app.inference import _read_image
    from app.tiling import make_tiles
    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sync = lambda: torch.cuda.synchronize(engine.device) if engine.device.type == "cuda" else None
    started = time.perf_counter()
    engine = CountGDEngine(args.model_dir, args.device)
    sync()
    load_ms = (time.perf_counter() - started) * 1000
    info = metadata(args, engine, torch)

    references = [{"path": str(args.reference), "boxes": [args.reference_box]}]
    exemplar, boxes = _exemplar_image(references)
    image = _read_image(str(args.target))
    scale = min(800 / min(image.size), 1333 / max(image.size))
    tiles = make_tiles(image.width, image.height, 800 if scale < 1 else max(image.size))
    crop = image if len(tiles) == 1 else image.crop(tiles[0].box)
    sync()
    started = time.perf_counter()
    engine._predict(crop, exemplar, boxes, args.confidence, args.label)
    sync()
    warmup_ms = (time.perf_counter() - started) * 1000

    calls = {"bert": 0, "backbone": 0}
    hooks = [engine.model.bert.register_forward_hook(
        lambda *_: calls.__setitem__("bert", calls["bert"] + 1)),
        engine.model.backbone.register_forward_hook(
            lambda *_: calls.__setitem__("backbone", calls["backbone"] + 1))]
    original_predict = engine._predict
    captured = []

    def capture(*predict_args, **predict_kwargs):
        prediction = original_predict(*predict_args, **predict_kwargs)
        captured.append(prediction)
        return prediction

    engine._predict = capture
    runs = []
    baseline = None
    expected_calls = {"uncached": {"bert": len(tiles), "backbone": 2 * len(tiles)},
                      "cached": {"bert": 1, "backbone": len(tiles) + 1}}
    try:
        for repeat in range(args.repeats):
            for cached in (False, True):
                mode = "cached" if cached else "uncached"
                name = f"run-{repeat + 1:02d}-{mode}"
                captured.clear()
                calls.update(bert=0, backbone=0)
                if engine.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(engine.device)
                sync()
                started = time.perf_counter()
                results = engine.count(references, [str(args.target)],
                                       str(args.output_dir / "annotations"),
                                       args.confidence, args.label, cache_reference=cached)
                sync()
                wall_ms = (time.perf_counter() - started) * 1000
                assert len(results) == 1 and len(captured) == len(tiles)
                raw = list(captured)
                current = (normalized_tiles(raw), results[0], raw)
                if baseline is None:
                    baseline = current
                comparison = compare(baseline, current, image.width, image.height)
                forward_calls_match_expected = calls == expected_calls[mode]
                peak = (torch.cuda.max_memory_allocated(engine.device)
                        if engine.device.type == "cuda" else None)
                run = {"name": name, "mode": mode, "repeat": repeat + 1,
                       "wall_ms": wall_ms, "peak_cuda_allocated_bytes": peak,
                       "forward_calls": dict(calls), "forward_calls_match_expected":
                       forward_calls_match_expected, "passed": comparison["passed"]
                       and forward_calls_match_expected, "tile_count": len(tiles),
                       "raw_tiles": [{"tile_index": i, "boxes": box_values.tolist(),
                                      "scores": score_values.tolist()}
                                     for i, (box_values, score_values) in enumerate(raw)],
                       "results": results, "comparison_to_first_uncached": comparison,
                       "metadata": info}
                (args.output_dir / f"{name}.json").write_text(
                    json.dumps(run, indent=2), encoding="utf-8")
                runs.append({key: run[key] for key in ("name", "mode", "wall_ms",
                             "peak_cuda_allocated_bytes", "forward_calls",
                             "forward_calls_match_expected", "passed", "tile_count")}
                            | {"count": results[0]["count"], "comparison": comparison})
                print(f"{name}: {wall_ms:.1f} ms, count={results[0]['count']}, "
                      f"BERT/backbone={calls['bert']}/{calls['backbone']}, "
                      f"comparison={'PASS' if comparison['passed'] else 'FAIL'}, "
                      f"forward calls={'PASS' if forward_calls_match_expected else 'FAIL'}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        engine._predict = original_predict

    medians = {mode: statistics.median(run["wall_ms"] for run in runs if run["mode"] == mode)
               for mode in ("uncached", "cached")}
    report = {"metadata": info, "load_ms": load_ms, "one_tile_uncached_warmup_ms": warmup_ms,
              "tile_count": len(tiles), "expected_forward_calls": expected_calls,
              "median_wall_ms": medians,
              "median_speedup": medians["uncached"] / medians["cached"],
              "runs": runs, "passed": all(run["passed"] for run in runs)}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("load_ms", "one_tile_uncached_warmup_ms", "median_wall_ms",
                       "median_speedup", "passed")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
