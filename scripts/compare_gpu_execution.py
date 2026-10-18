"""Run and compare exact CountGD++ GPU execution on the fitting fixture.

Run each image in its own isolated GPU container with a shared output mount,
then use ``compare`` on the two saved directories (CPU is sufficient).
"""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_countgd import metadata  # noqa: E402


ARRAYS = ("pred_boxes", "pred_logits", "sigmoid_scores", "input_ids",
          "selected_mask", "selected_boxes", "selected_scores")
INPUT_FIELDS = ("reference_sha256", "target_sha256", "reference_box", "label",
                "confidence", "device", "model_dtype", "torch_threads",
                "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "checkpoint_sha256",
                "source_commit", "versions", "gpu", "repeats")


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _merged(result):
    return {key: result[key] for key in ("count", "detections", "width", "height",
                                          "warning", "processing") if key in result}


def _capture(engine, references, args, output_dir):
    """Run an untimed pass; save every query and the adapter's chosen outputs."""
    captures = []
    original_forward = engine.model.forward_conditioned
    original_predict = engine._predict

    def forward(*forward_args, **forward_kwargs):
        prediction = original_forward(*forward_args, **forward_kwargs)
        torch = engine._torch
        with torch.inference_mode():
            tokens = prediction["input_ids"][0]
            positive_end = next((i for i, token in enumerate(tokens.tolist()) if token == 1012), None)
            if positive_end is None:
                raise RuntimeError("CountGD++ did not return a positive prompt separator.")
            logits = prediction["pred_logits"].sigmoid()[0]
            positive = logits[:, :positive_end + 1].amax(dim=-1)
            remaining = logits[:, positive_end + 1:]
            negative = (remaining.amax(dim=-1) if remaining.shape[1]
                        else torch.full_like(positive, -float("inf")))
            chosen = (positive > args.confidence) & (positive > negative)
            captures.append({
                "pred_boxes": prediction["pred_boxes"][0].detach().cpu().numpy().copy(),
                "pred_logits": prediction["pred_logits"][0].detach().cpu().numpy().copy(),
                "sigmoid_scores": logits.detach().cpu().numpy().copy(),
                "input_ids": tokens.detach().cpu().numpy().copy(),
                "selected_mask": chosen.detach().cpu().numpy().copy(),
            })
        return prediction

    def predict(*predict_args, **predict_kwargs):
        boxes, scores = original_predict(*predict_args, **predict_kwargs)
        tile = captures[-1]
        tile["selected_boxes"] = boxes.copy()
        tile["selected_scores"] = scores.copy()
        if (len(tile["pred_boxes"]) != 900
                or not np.array_equal(boxes, tile["pred_boxes"][tile["selected_mask"]])
                or not np.array_equal(scores, tile["sigmoid_scores"][tile["selected_mask"]].max(axis=-1))):
            raise AssertionError("Captured queries disagree with the adapter's selected detections")
        return boxes, scores

    engine.model.forward_conditioned = forward
    engine._predict = predict
    try:
        result = engine.count(references, [str(args.target)], str(output_dir / "annotations"),
                              args.confidence, args.label, cache_reference=True)[0]
    finally:
        engine.model.forward_conditioned = original_forward
        engine._predict = original_predict
    if len(captures) != result.get("processing", {}).get("tiles", 1):
        raise AssertionError("A target tile was not captured")
    arrays = {f"tile{i:03d}_{key}": tile[key] for i, tile in enumerate(captures)
              for key in ARRAYS}
    np.savez_compressed(output_dir / "queries.npz", **arrays)
    return result, len(captures), [int(tile["selected_mask"].sum()) for tile in captures]


def _enable_sdpa(model, torch):
    """Opt only decoder attention whose callers consume output[0] into SDPA."""
    layers = model.transformer.decoder.layers
    for layer in layers:
        for name in ("self_attn", "ca_text"):
            attention = getattr(layer, name)
            if not isinstance(attention, torch.nn.MultiheadAttention):
                raise TypeError(f"decoder {name} is not torch.nn.MultiheadAttention")
            original = attention.forward
            signature = inspect.signature(original)

            def forward(*args, _original=original, _signature=signature, **kwargs):
                bound = _signature.bind(*args, **kwargs)
                bound.arguments["need_weights"] = False
                return _original(*bound.args, **bound.kwargs)

            attention.forward = forward
    return len(layers)


def _serve_timings(engine, references, args, info):
    """Keep each model loaded while the caller alternates baseline/candidate trials."""
    torch = engine._torch

    def sync():
        if engine.device.type == "cuda":
            torch.cuda.synchronize(engine.device)

    engine.count(references, [str(args.target)], str(args.output_dir / "warmup"),
                 args.confidence, args.label, cache_reference=True)
    sync()
    (args.output_dir / "metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"READY {args.name}", flush=True)
    for line in sys.stdin:
        if line.strip() == "quit":
            break
        index = int(line.strip())
        destination = args.output_dir / f"trial-{index:02d}.json"
        if index < 0 or destination.exists():
            raise ValueError("Trial index must be new and nonnegative")
        sync()
        started = time.perf_counter()
        result = engine.count(references, [str(args.target)], str(args.output_dir / "annotations"),
                              args.confidence, args.label, cache_reference=True)[0]
        sync()
        record = {"index": index, "name": args.name,
                  "wall_ms": (time.perf_counter() - started) * 1000,
                  "merged": _merged(result), "annotation": result["annotated_path"]}
        destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"DONE {index} {record['wall_ms']:.1f} ms count={result['count']}", flush=True)
    return 0


def run(args):
    from app.countgd import CountGDEngine, _exemplar_image
    from app.inference import _read_image
    from app.models import COUNTGD_ASSETS
    from app.tiling import make_tiles
    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError("--output-dir must be empty to keep one run's files together")
    started = time.perf_counter()
    engine = CountGDEngine(args.model_dir, args.device)
    sdpa_layers = _enable_sdpa(engine.model, torch) if args.sdpa else 0

    def sync():
        if engine.device.type == "cuda":
            torch.cuda.synchronize(engine.device)

    sync()
    load_ms = (time.perf_counter() - started) * 1000
    info = metadata(args, engine, torch)
    info.update(name=args.name, reference_sha256=_sha256(args.reference),
                target_sha256=_sha256(args.target),
                checkpoint_sha256=COUNTGD_ASSETS["countgd_plusplus.pth"][1],
                gpu=info["hardware"].get("gpu"), sdpa=args.sdpa,
                sdpa_decoder_layers=sdpa_layers)
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

    if args.worker:
        return _serve_timings(engine, references, args, info)

    runs = []
    for repeat in range(args.repeats):
        if engine.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(engine.device)
        sync()
        started = time.perf_counter()
        result = engine.count(references, [str(args.target)], str(args.output_dir / "annotations"),
                              args.confidence, args.label, cache_reference=True)[0]
        sync()
        wall_ms = (time.perf_counter() - started) * 1000
        runs.append({"repeat": repeat + 1, "wall_ms": wall_ms, "count": result["count"],
                     "merged": _merged(result),
                     "annotation": f"annotations/{Path(result['annotated_path']).name}",
                     "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated(engine.device)
                                                   if engine.device.type == "cuda" else None)})
        print(f"{args.name} {repeat + 1}/{args.repeats}: {wall_ms:.1f} ms, count={result['count']}", flush=True)

    verification, tile_count, selected_counts = _capture(engine, references, args, args.output_dir)
    merged = _merged(verification)
    consistent = all(item["merged"] == merged for item in runs)
    report = {"metadata": info, "load_ms": load_ms,
              "one_tile_uncached_warmup_ms": warmup_ms,
              "median_wall_ms": statistics.median(item["wall_ms"] for item in runs),
              "tile_count": tile_count, "selected_tile_counts": selected_counts,
              "peak_cuda_allocated_bytes": max((item["peak_cuda_allocated_bytes"] or 0) for item in runs),
              "runs": runs, "verification": {"merged": merged,
              "annotation": f"annotations/{Path(verification['annotated_path']).name}"},
              "timed_runs_match_verification": consistent}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{args.name}: median {report['median_wall_ms']:.1f} ms; "
          f"tiles={tile_count}; verified={consistent}", flush=True)
    return 0 if consistent else 1


def _array_difference(before, after):
    result = {"exact": before.dtype == after.dtype and before.shape == after.shape
              and before.tobytes() == after.tobytes(), "baseline_shape": list(before.shape),
              "candidate_shape": list(after.shape), "baseline_dtype": str(before.dtype),
              "candidate_dtype": str(after.dtype), "different_values": None,
              "nonfinite_pattern_differences": None, "max_finite_error": None}
    if before.shape != after.shape:
        return result
    if np.issubdtype(before.dtype, np.number) and np.issubdtype(after.dtype, np.number):
        finite = np.isfinite(before) & np.isfinite(after)
        result["max_finite_error"] = (float(np.max(np.abs(before[finite].astype(np.float64)
                                                       - after[finite].astype(np.float64))))
                                      if finite.any() else 0.0)
        before_kind = np.select([np.isnan(before), np.isposinf(before), np.isneginf(before)],
                                [1, 2, 3], default=0)
        after_kind = np.select([np.isnan(after), np.isposinf(after), np.isneginf(after)],
                               [1, 2, 3], default=0)
        result["nonfinite_pattern_differences"] = int(np.count_nonzero(before_kind != after_kind))
        result["different_values"] = int(np.count_nonzero(before[finite] != after[finite]))
    else:
        result["different_values"] = int(np.count_nonzero(before != after))
        result["nonfinite_pattern_differences"] = 0
    return result


def compare(baseline_dir, candidate_dir, output_dir):
    baseline = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    candidate = json.loads((candidate_dir / "summary.json").read_text(encoding="utf-8"))
    if baseline["metadata"]["name"] != "baseline" or candidate["metadata"]["name"] != "candidate":
        raise ValueError("Expected baseline and candidate run directories")
    incompatible = [field for field in INPUT_FIELDS
                    if (field not in baseline["metadata"] or field not in candidate["metadata"]
                        or baseline["metadata"][field] != candidate["metadata"][field])]
    differences = {}
    tile_count_equal = baseline["tile_count"] == candidate["tile_count"]
    if not incompatible:
        with np.load(baseline_dir / "queries.npz", allow_pickle=False) as before, \
                np.load(candidate_dir / "queries.npz", allow_pickle=False) as after:
            required = {f"tile{i:03d}_{key}" for i in range(max(baseline["tile_count"],
                                                                 candidate["tile_count"]))
                        for key in ARRAYS}
            keys = set(before.files) | set(after.files) | required
            for key in sorted(keys):
                differences[key] = (_array_difference(before[key], after[key])
                                    if key in before and key in after
                                    else {"exact": False, "missing_from":
                                          "baseline" if key not in before else "candidate"})
    merged_before = baseline["verification"]["merged"]
    merged_after = candidate["verification"]["merged"]
    merged_equal = merged_before == merged_after
    counts_equal = (merged_before["count"] == merged_after["count"]
                    and baseline["selected_tile_counts"] == candidate["selected_tile_counts"]
                    and [run["count"] for run in baseline["runs"]]
                    == [run["count"] for run in candidate["runs"]])
    indices_equal = ([item["index"] for item in merged_before["detections"]]
                     == [item["index"] for item in merged_after["detections"]])
    exact_arrays = bool(differences) and all(item["exact"] for item in differences.values())
    passed = (not incompatible and tile_count_equal and exact_arrays and merged_equal
              and counts_equal and indices_equal
              and baseline["timed_runs_match_verification"]
              and candidate["timed_runs_match_verification"])
    base_ms, candidate_ms = baseline["median_wall_ms"], candidate["median_wall_ms"]
    report = {"passed": passed, "incompatible_metadata": incompatible,
              "tile_count_equal": tile_count_equal, "all_query_arrays_byte_exact": exact_arrays,
              "array_differences": differences, "merged_equal": merged_equal,
              "counts_equal": counts_equal, "indices_equal": indices_equal,
              "max_finite_array_error": max((item.get("max_finite_error") or 0
                                              for item in differences.values()), default=0),
              "different_finite_array_values": sum((item.get("different_values") or 0
                                                      for item in differences.values())),
              "nonfinite_pattern_differences": sum((item.get("nonfinite_pattern_differences") or 0
                                                     for item in differences.values())),
              "baseline_merged": merged_before, "candidate_merged": merged_after,
              "baseline_median_wall_ms": base_ms, "candidate_median_wall_ms": candidate_ms,
              "candidate_speedup": base_ms / candidate_ms if candidate_ms > 0 else None,
              "baseline_source_patch": baseline["metadata"].get("source_patch"),
              "candidate_source_patch": candidate["metadata"].get("source_patch")}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def annotation_link(directory, summary):
        path = directory / summary["verification"]["annotation"]
        return Path(os.path.relpath(path, output_dir)).as_posix()

    lines = ["# CountGD++ GPU execution comparison", "",
             f"**Exact output:** {'PASS' if passed else 'FAIL'}", "",
             "| Image | Source patch | SDPA | Median cached full image | Peak CUDA allocated | Tiles | Selected queries | Merged count | Annotation |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for title, directory, summary in (("Baseline", baseline_dir, baseline),
                                      ("Candidate", candidate_dir, candidate)):
        lines.append(f"| {title} | {summary['metadata'].get('source_patch')} | "
                     f"{summary['metadata'].get('sdpa', False)} | "
                     f"{summary['median_wall_ms']:.1f} ms | "
                     f"{summary['peak_cuda_allocated_bytes'] / 2**20:.1f} MiB | "
                     f"{summary['tile_count']} | {sum(summary['selected_tile_counts'])} | "
                     f"{summary['verification']['merged']['count']} | "
                     f"[view]({annotation_link(directory, summary)}) |")
    lines += ["", f"Candidate speedup: **{report['candidate_speedup']:.2f}×**"
              if report["candidate_speedup"] is not None else "Candidate speedup: unavailable", "",
              f"Metadata mismatches: {', '.join(incompatible) if incompatible else 'none'}.",
              f"All query arrays byte exact: {exact_arrays}; merged detections exact: {merged_equal}; "
              f"counts exact: {counts_equal}; indices exact: {indices_equal}."]
    changed = [key for key, value in differences.items() if not value["exact"]]
    if changed:
        lines += ["", "Changed arrays:", ""] + [
            f"- `{key}`: {differences[key]}" for key in changed]
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Exact comparison: {'PASS' if passed else 'FAIL'}; "
          f"candidate speedup {report['candidate_speedup']:.2f}x" if report["candidate_speedup"]
          else f"Exact comparison: {'PASS' if passed else 'FAIL'}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run one isolated GPU image")
    fixture = Path("test-results/fitting-debug")
    run_parser.add_argument("--name", choices=("baseline", "candidate"), required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--repeats", type=int, default=5)
    run_parser.add_argument("--reference", type=Path, default=fixture / "reference-original.jpg")
    run_parser.add_argument("--target", type=Path, default=fixture / "target-original.jpg")
    run_parser.add_argument("--reference-box", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                            default=[1039, 1775, 1344, 2170])
    run_parser.add_argument("--label", default="Bolt with red cap")
    run_parser.add_argument("--confidence", type=float, default=.5)
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument("--model-dir", type=Path, default=Path("models/countgd"))
    run_parser.add_argument("--sdpa", action="store_true",
                            help="experiment: request SDPA on decoder self/text attention")
    run_parser.add_argument("--worker", action="store_true",
                            help="keep the warm model loaded; accept trial numbers or quit on stdin")
    compare_parser = commands.add_parser("compare", help="compare saved runs exactly")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "compare":
        return 0 if compare(args.baseline, args.candidate, args.output_dir)["passed"] else 1
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
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
