"""Tune-only experimental reference and tiling strategies for the frozen benchmark.

Run inside the benchmark Docker image; this file never changes app inference.
Ground truth enters only through benchmark evaluation after prediction.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import benchmark_reliability as benchmark


STRATEGIES = {
    "yoloe-by-image": {"family": "yoloe", "prototypes": "mean per reference image, then L2 normalize"},
    "yoloe-per-object": {"family": "yoloe", "prototypes": "one per reference box, then L2 normalize"},
    "countgd-overlap50": {"family": "countgd", "tile_size": 800, "overlap": 0.5},
    "countgd-tile640": {"family": "countgd", "tile_size": 640, "overlap": 0.2},
    "countgd-tile512": {"family": "countgd", "tile_size": 512, "overlap": 0.2},
}


@contextmanager
def yoloe_prototypes(mode, yoloe_cls=None, predictor_cls=None, engine_cls=None):
    """Replace pooled class prototypes for one experiment, then restore methods."""
    if yoloe_cls is None or predictor_cls is None:
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        yoloe_cls, predictor_cls = YOLOE, YOLOEVPSegPredictor
    if engine_cls is None:
        from app.inference import InferenceEngine
        engine_cls = InferenceEngine
    import torch

    original_get_vpe = predictor_cls.get_vpe
    original_set_classes = yoloe_cls.set_classes
    original_count = engine_cls.count
    groups = []

    def count(self, *args, **kwargs):
        groups.clear()
        try:
            return original_count(self, *args, **kwargs)
        finally:
            groups.clear()

    def get_vpe(self, *args, **kwargs):
        result = original_get_vpe(self, *args, **kwargs)
        groups.append(result.detach())
        return result

    def set_classes(self, classes, embeddings=None):
        if not groups or embeddings is None:
            raise RuntimeError("Expected visual-reference embeddings before YOLOE.set_classes")
        try:
            if mode == "yoloe-by-image":
                vectors = torch.cat([group.mean(dim=1, keepdim=True) for group in groups], dim=1)
            elif mode == "yoloe-per-object":
                vectors = torch.cat(groups, dim=1)
            else:
                raise ValueError(f"Unknown prototype strategy: {mode}")
            if vectors.ndim != 3 or vectors.shape[0] != 1 or not torch.isfinite(vectors).all():
                raise RuntimeError("Invalid visual-reference prototypes")
            if (vectors.norm(dim=-1) < 1e-8).any():
                raise RuntimeError("A visual-reference prototype has zero magnitude")
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)
            return original_set_classes(self, [f"object-{i}" for i in range(vectors.shape[1])],
                                        embeddings=vectors)
        finally:
            groups.clear()

    predictor_cls.get_vpe = get_vpe
    yoloe_cls.set_classes = set_classes
    engine_cls.count = count
    try:
        yield
    finally:
        predictor_cls.get_vpe = original_get_vpe
        yoloe_cls.set_classes = original_set_classes
        engine_cls.count = original_count


@contextmanager
def countgd_tiles(tile_size, overlap, countgd_module=None):
    """Change native crop geometry without changing CountGD++ model resize."""
    if countgd_module is None:
        import app.countgd as countgd_module
    original_make_tiles = countgd_module.make_tiles
    engine_cls = countgd_module.CountGDEngine
    original_count = engine_cls.count

    def make_tiles(width, height, _baseline_size, **kwargs):
        return original_make_tiles(width, height, tile_size, overlap=overlap, **kwargs)

    def count(self, *args, **kwargs):
        outputs = original_count(self, *args, **kwargs)
        for output in outputs:
            if "processing" in output:
                # The production response hardcodes its baseline crop size.
                output["processing"]["tile_size"] = tile_size
                output["processing"]["overlap"] = overlap
        return outputs

    countgd_module.make_tiles = make_tiles
    engine_cls.count = count
    try:
        yield
    finally:
        countgd_module.make_tiles = original_make_tiles
        engine_cls.count = original_count


@contextmanager
def patched_strategy(name):
    spec = STRATEGIES[name]
    original_settings = benchmark._settings
    runner_sha256 = benchmark.sha256(Path(__file__))

    def settings():
        return {**original_settings(), "experiment": {"name": name, **spec,
                "runner_sha256": runner_sha256}}

    with ExitStack() as stack:
        benchmark._settings = settings
        stack.callback(setattr, benchmark, "_settings", original_settings)
        if spec["family"] == "yoloe":
            stack.enter_context(yoloe_prototypes(name))
        else:
            stack.enter_context(countgd_tiles(spec["tile_size"], spec["overlap"]))
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--manifest", type=Path, default=ROOT / "test-results/reliability-benchmark/data/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "test-results/reliability-benchmark/strategies/runs")
    parser.add_argument("--model", action="append", choices=benchmark.MODELS)
    parser.add_argument("--threshold", action="append", type=float)
    parser.add_argument("--target", action="append", help="Tune target ID; may be repeated")
    parser.add_argument("--category", action="append", help="Category; may be repeated")
    parser.add_argument("--reference-variant", action="append", help="Named manifest reference variant")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    family = STRATEGIES[args.strategy]["family"]
    models = args.model or (["yoloe-26x-seg"] if family == "yoloe" else ["countgd-plusplus"])
    if any((model == "countgd-plusplus") != (family == "countgd") for model in models):
        parser.error("Selected models must match the strategy family")
    thresholds = args.threshold or ([0.01, 0.05] if family == "yoloe" else [0.23, 0.40, 0.50, 0.60])
    manifest = benchmark.prepare_manifest(args.manifest)
    tune_ids = {target["id"] for target in manifest["targets"] if target["split"] == "tune"}
    if args.target and any(target not in tune_ids for target in args.target):
        parser.error("Every selected target must be in the tune split")
    output = args.output_dir / args.strategy
    with patched_strategy(args.strategy):
        previous = output / "results.json"
        if args.resume and previous.is_file():
            old = json.loads(previous.read_text(encoding="utf-8"))
            settings = benchmark._settings()
            if any(record.get("settings") != settings for record in old.get("records", {}).values()):
                parser.error("Existing strategy results use different code or settings; choose a new output directory")
        report = benchmark.run(manifest, output, models=models, thresholds=thresholds,
                               categories=args.category, splits=["tune"], targets=args.target,
                               variants=args.reference_variant or ["default"], resume=args.resume)
    print(f"Saved {report['selected_jobs']} tune jobs; {report['failed_jobs']} failed: {output / 'results.json'}", flush=True)
    return int(report["failed_jobs"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
