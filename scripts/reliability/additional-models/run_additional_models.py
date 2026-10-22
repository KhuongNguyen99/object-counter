"""Exploratory PET-DINO and SAM3Count screen on frozen tune images.

Run inside the prepared image, for example:
  python test-results/reliability-benchmark/additional-models/run_additional_models.py --dry-run
  python test-results/reliability-benchmark/additional-models/run_additional_models.py --model pet-dino-swin-t --resume
  python test-results/reliability-benchmark/additional-models/run_additional_models.py --include-mixed --resume
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import benchmark_reliability as benchmark  # noqa: E402
from app import models as asset_catalog  # noqa: E402


DATA = ROOT / "test-results/reliability-benchmark/data"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "runs/screen-001"
MODELS = (asset_catalog.PETDINO_MODEL_ID, asset_catalog.SAM3COUNT_MODEL_ID)
THRESHOLDS = {asset_catalog.PETDINO_MODEL_ID: 0.30, asset_catalog.SAM3COUNT_MODEL_ID: 0.35}
CASES = (("manifest.json", "screwcount-13"),
         ("manifest.json", "screwcount-567"),
         ("mixed-hardware-manifest.json", "arg-8-bolts"),
         ("mixed-hardware-manifest.json", "arg-8-washers"))


def _hash_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _metadata() -> dict:
    app_dir = Path(asset_catalog.__file__).resolve().parent
    code = {name: app_dir / name for name in
            ("inference.py", "petdino.py", "petdino_runtime.py", "sam3count.py", "models.py")}
    code = {**{f"app/{name}": path for name, path in code.items()},
            "scripts/benchmark_reliability.py": Path(benchmark.__file__).resolve(),
            "additional-models/run_additional_models.py": Path(__file__).resolve()}
    assets = {
        MODELS[0]: {name: {"bytes": size, "sha256": digest}
                    for name, (size, digest) in {**asset_catalog.PETDINO_ASSETS,
                                                 **{name: value for name, value in asset_catalog.COUNTGD_ASSETS.items()
                                                    if name.startswith("bert-base-uncased/")}}.items()},
        MODELS[1]: {name: {"bytes": size, "sha256": digest}
                    for name, (size, digest) in asset_catalog.SAM3COUNT_ASSETS.items()},
    }
    return {"thresholds": THRESHOLDS, "pinned_assets": assets,
            "code_sha256": {name: benchmark.sha256(path) for name, path in code.items()},
            "code_paths": {name: str(path) for name, path in code.items()},
            "runtime": {"python": platform.python_version(),
                        "DEVICE": os.environ.get("DEVICE", "auto"),
                        "MODEL_DIR": str(asset_catalog.model_dir()),
                        "PETDINO_PYTHON": os.environ.get("PETDINO_PYTHON", "/opt/petdino-venv/bin/python"),
                        "PETDINO_SOURCE_DIR": os.environ.get("PETDINO_SOURCE_DIR", "/opt/petdino"),
                        "SAM3_SOURCE_DIR": os.environ.get("SAM3_SOURCE_DIR", "/opt/sam3"),
                        "SAM3COUNT_SOURCE_DIR": os.environ.get("SAM3COUNT_SOURCE_DIR", "/opt/sam3count")}}


def _prepare(include_mixed: bool, wanted: list[str] | None, max_cases: int | None,
             data_dir: Path, initial_manifest: Path | None, mixed_manifest: Path | None):
    manifests = {}
    cases = []
    manifest_paths = {"manifest.json": initial_manifest or data_dir / "manifest.json",
                      "mixed-hardware-manifest.json": mixed_manifest or data_dir / "mixed-hardware-manifest.json"}
    chosen = CASES if include_mixed or wanted else CASES[:2]
    known = {target_id for _, target_id in CASES}
    unknown = set(wanted or []) - known
    if unknown:
        raise ValueError(f"Unknown or non-tune target IDs: {sorted(unknown)}")
    for filename, target_id in chosen:
        if wanted and target_id not in wanted:
            continue
        if filename not in manifests:
            manifests[filename] = benchmark.prepare_manifest(manifest_paths[filename])
        manifest = manifests[filename]
        target = next(target for target in manifest["targets"] if target["id"] == target_id)
        if target["split"] != "tune":
            raise ValueError(f"Exploratory screen may use tune images only: {target_id}")
        cases.append((manifest, target))
    if max_cases is not None:
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("No cases selected")
    return manifests, cases


def _jobs(cases, selected_models, metadata):
    jobs = []
    for model in selected_models:  # Keep the same checkpoint resident across target images.
        for manifest, target in cases:
            references = (manifest["references"]["default"][target["category"]]
                          if model == asset_catalog.PETDINO_MODEL_ID else [])
            label = manifest["description"].get(target["category"], target["category"])
            job = {"model": model, "threshold": THRESHOLDS[model],
                   "modality": "exemplar boxes plus object label" if references else "text label only",
                   "variant": "default" if references else "text-only", "target_id": target["id"],
                   "category": target["category"], "split": target["split"],
                   "manifest_path": manifest["path"], "manifest_sha256": manifest["sha256"],
                   "target_sha256": target["image"]["sha256"],
                   "points": target["points"], "gt_boxes": target["boxes"],
                   "references": [{"sha256": ref["sha256"], "boxes": ref["boxes"]} for ref in references],
                   "label": label, "settings": metadata}
            jobs.append((_hash_json(job), job, target, references))
    return jobs


def run(args, engine_factory=None) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    manifests, cases = _prepare(args.include_mixed, args.target, args.max_cases,
                                getattr(args, "data_dir", DATA),
                                getattr(args, "initial_manifest", None),
                                getattr(args, "mixed_manifest", None))
    selected_models = tuple(dict.fromkeys(args.model or MODELS))
    metadata = _metadata()
    jobs = _jobs(cases, selected_models, metadata)
    for _, job, _, refs in jobs:
        print(f"{job['model']} {job['target_id']} threshold={job['threshold']:.2f} "
              f"modality={job['modality']} references={len(refs)} label={job['label']!r}", flush=True)
    if args.dry_run:
        print(f"Dry run: {len(cases)} tune cases, {len(jobs)} model jobs; no inference or output written.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "results.json"
    inputs = {name: manifest["sha256"] for name, manifest in manifests.items()}
    if args.resume and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if any(name in report.get("manifest_sha256", {})
               and report["manifest_sha256"][name] != digest for name, digest in inputs.items()):
            raise ValueError("A source manifest changed; choose a new output directory")
        if report.get("model_metadata") != metadata:
            raise ValueError("Model/code/runtime metadata changed; choose a new output directory")
        report["manifest_sha256"].update(inputs)
    else:
        report = {"schema_version": 1, "kind": "exploratory additional-model tune screen",
                  "manifest_sha256": inputs, "model_metadata": metadata,
                  "matching": benchmark.MATCHING, "records": {}, "summary": [],
                  "interpretation": "PET-DINO uses exemplar boxes plus text; SAM3Count uses text alone. Scores compare end-to-end modalities, not equal prompts. Source labels and center-only ground truth limit conclusions."}
        benchmark._write_report(report_path, report)

    engine = None
    failed = 0
    try:
        for index, (key, job, target, refs) in enumerate(jobs, 1):
            if args.resume and report["records"].get(key, {}).get("status") == "ok":
                print(f"[{index}/{len(jobs)}] resume-skip {job['model']} {job['target_id']}", flush=True)
                continue
            print(f"[{index}/{len(jobs)}] run {job['model']} {job['target_id']}", flush=True)
            started = datetime.now(timezone.utc).isoformat()
            start = time.perf_counter()
            try:
                if engine is None:
                    if engine_factory is None:
                        from app.inference import InferenceEngine
                        engine = InferenceEngine()
                    else:
                        engine = engine_factory()
                response = engine.count(job["model"],
                                        [{"path": ref["path"], "boxes": ref["boxes"]} for ref in refs],
                                        [target["image"]["path"]], str(args.output_dir / "annotated"),
                                        confidence=job["threshold"], label=job["label"])
                if not isinstance(response, list) or len(response) != 1:
                    raise ValueError("Engine must return one prediction for one target")
                prediction = response[0]
                if prediction["count"] != len(prediction["detections"]):
                    raise ValueError("Engine count disagrees with detection list")
                if (prediction["width"], prediction["height"]) != (target["image"]["width"], target["image"]["height"]):
                    raise ValueError("Engine image dimensions differ from benchmark image")
                metrics = benchmark.evaluate(target["points"], prediction["detections"], target["boxes"])
                record = {**job, "status": "ok", "target": target["image"],
                          "reference_images": refs, "prediction": prediction, "metrics": metrics,
                          "wall_ms": round((time.perf_counter() - start) * 1000, 2),
                          "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat()}
                print(f"  matched={metrics['matched']} missed={metrics['missed']} extra={metrics['extra']}", flush=True)
            except Exception as exc:
                failed += 1
                record = {**job, "status": "error", "target": target["image"],
                          "reference_images": refs,
                          "error": {"type": type(exc).__name__, "message": str(exc)},
                          "wall_ms": round((time.perf_counter() - start) * 1000, 2),
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
            report["summary"] = benchmark._summary(report["records"])
            benchmark._write_report(report_path, report)
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                failed += 1
                report["engine_close_error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["selected_jobs"] = len(jobs)
        report["failed_jobs"] = failed
        benchmark._write_report(report_path, report)
    print(f"Saved {len(jobs)} selected jobs; {failed} failed: {report_path}", flush=True)
    return int(failed > 0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=MODELS,
                        help="Repeat to select models; default PET-DINO and SAM3Count")
    parser.add_argument("--target", action="append", help="Repeat to select one of the four frozen tune target IDs")
    parser.add_argument("--include-mixed", action="store_true", help="Add Photo 8 bolt and washer queries")
    parser.add_argument("--data-dir", type=Path, default=DATA,
                        help="Directory containing both benchmark manifests (default: repository data directory)")
    parser.add_argument("--initial-manifest", type=Path,
                        help="Override path to the initial 16-case manifest")
    parser.add_argument("--mixed-manifest", type=Path,
                        help="Override path to the 12-case mixed manifest")
    parser.add_argument("--max-cases", "--maxcases", type=int, metavar="N",
                        help="Use only the first N selected target queries, before multiplying by model count")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true", help="Skip only successful jobs with identical inputs")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list jobs without loading models")
    args = parser.parse_args(argv)
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
