"""Tune-only CountGD++ negative-text trial; source targets and references stay untouched.

The first period separates positive and negative tokens in CountGD++'s
postprocessor. Results use the standard benchmark runner and its point matching.
"""

import argparse
import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve()
ROOT = next((parent for parent in HERE.parents if (parent / "scripts/benchmark_reliability.py").is_file()), Path("/app"))
PROMPTS = {"wood": "wood", "table": "wooden table background"}
TARGETS = {"screwcount-13", "screwcount-17", "screwcount-540", "screwcount-567"}
BENCHMARK_SCRIPT = next((path for path in (ROOT / "scripts/benchmark_reliability.py",
                                           HERE.parent / "benchmark_reliability.py",
                                           HERE.parent.parent / "benchmark_reliability.py") if path.is_file()),
                        ROOT / "scripts/benchmark_reliability.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "test-results/reliability-benchmark/data/manifest.json")
    parser.add_argument("--outdir", type=Path, default=HERE.parent / "runs")
    parser.add_argument("--benchmark-script", type=Path, default=BENCHMARK_SCRIPT)
    parser.add_argument("--prompt", action="append", choices=PROMPTS, help="Repeat; default: both variants")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("benchmark_reliability", args.benchmark_script)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    app_root = Path("/app") if Path("/app/app/countgd.py").is_file() else args.benchmark_script.resolve().parent.parent
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    from app.countgd import CountGDEngine

    manifest = benchmark.prepare_manifest(args.manifest)
    selected = {item["id"] for item in manifest["targets"]
                if item["split"] == "tune" and item["category"] in {"nuts", "screws"}}
    if selected != TARGETS or set(manifest["references"]) != {"default"}:
        raise ValueError(f"Expected four original nuts/screws tune targets, found {sorted(selected)}")

    original_prepare, original_settings = CountGDEngine._prepare_conditioning, benchmark._settings
    failed = False
    try:
        for name in dict.fromkeys(args.prompt or PROMPTS):
            negative = PROMPTS[name]

            def prepare(self, exemplar_image, exemplar_boxes, label):
                with self._torch.inference_mode():
                    exemplar, boxes = self._reference_tensors(exemplar_image, exemplar_boxes)
                    caption = label.replace(".", " ") + " . " + negative + " . "
                    return self.model.prepare_conditioning(exemplar, boxes, [], [], captions=[caption])

            def settings():
                return {**original_settings(), "countgd_negative_prompt": negative,
                        "negative_prompt_runner_sha256": benchmark.sha256(HERE)}

            CountGDEngine._prepare_conditioning, benchmark._settings = prepare, settings
            print(f"CountGD++ negative text: {negative!r}", flush=True)
            report = benchmark.run(manifest, args.outdir / name, models=("countgd-plusplus",),
                                   thresholds=(0.5,), categories=("nuts", "screws"), splits=("tune",),
                                   targets=TARGETS, variants=("default",), resume=True)
            failed |= bool(report["failed_jobs"])
    finally:
        CountGDEngine._prepare_conditioning, benchmark._settings = original_prepare, original_settings
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
