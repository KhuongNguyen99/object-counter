"""Benchmark one or two exact RunPod GPUs with owned endpoints and auditable billing.

Omit --gpu to compare RTX A5000 and NVIDIA L4, or pass --gpu once for an
independent single-GPU trial. Each invocation has its own budget and manifest.

Fixture JSON: {"cases": [{"id": "fitting-count", "action": "count",
"image": "../fitting-debug/target-original.jpg", "boxes": [[1, 2, 30, 40]],
"confidence": 0.80, "label": "Bolt"}, {"id": "fitting-suggest",
"action": "suggest", "image": "../fitting-debug/target-original.jpg"}]}.
Images are sent unchanged. Paths resolve relative to the fixture manifest.

The wall-time cost envelope deliberately includes startup, transfer, queueing,
execution and idle, some of which may not be billed. Billing receipts include
the provider's GPU, CPU and disk charges; they can lag and are never described
as a finalized invoice. No endpoint is retained unless --keep-winner is set.
"""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runpod_deploy import (ApiError, OwnedEndpoints, RunPodApi, load_api_key,
                                  normalize_gpu, save_json, save_text, select_gpu, utc_now)

SOURCES = {
    "api_schema": "https://api.runpod.io/v2/openapi.json",
    "billing": "https://docs.runpod.io/api-reference-v2/billing/get-serverless-billing-history",
    "pricing": "https://docs.runpod.io/serverless/pricing",
    "job_api": "https://docs.runpod.io/serverless/endpoints/send-requests",
}


def finite(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def prepare_cases(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list) or not raw["cases"]:
        raise ValueError("Fixture must contain a nonempty cases array.")
    cases, ids = [], set()
    for case in raw["cases"]:
        name, action = case.get("id"), case.get("action", "count")
        if not isinstance(name, str) or not name or name in ids or action not in ("count", "suggest"):
            raise ValueError("Each fixture needs a unique id and count/suggest action.")
        ids.add(name)
        image_path = (path.parent / case["image"]).resolve()
        image = image_path.read_bytes()
        payload = {"action": action, "image_base64": base64.b64encode(image).decode("ascii"),
                   "label": case.get("label", "")}
        if not isinstance(payload["label"], str):
            raise ValueError("Fixture label must be text.")
        if action == "count":
            boxes, confidence = case.get("boxes"), case.get("confidence", .80)
            if not finite(confidence) or confidence > 1:
                raise ValueError("Confidence must be between zero and one.")
            if not isinstance(boxes, list) or not 1 <= len(boxes) <= 32:
                raise ValueError("Count fixture requires 1–32 boxes on its target image.")
            for box in boxes:
                if (not isinstance(box, list) or len(box) != 4 or
                        not all(finite(value) for value in box) or box[0] >= box[2] or box[1] >= box[3]):
                    raise ValueError("Fixture boxes must have nonnegative coordinates and positive area.")
            payload.update(boxes=boxes, confidence=confidence)
        serialized = json.dumps({"input": payload}, allow_nan=False, separators=(",", ":")).encode()
        if len(serialized) > 10_000_000:
            raise ValueError(f"Fixture {name!r} exceeds the RunPod 10 MB request limit.")
        metadata = {key: value for key, value in payload.items() if key != "image_base64"}
        metadata.update(id=name, image=str(image_path), image_sha256=hashlib.sha256(image).hexdigest(),
                        image_bytes=len(image), request_bytes=len(serialized))
        cases.append({"metadata": metadata, "payload": payload})
    return cases


def cost_envelope(hourly_usd, seconds, disk_gb):
    """List-rate envelope, not a receipt; rounds container disk to 5-min blocks."""
    compute = math.ceil(max(0, seconds)) * hourly_usd / 3600
    disk = math.ceil(max(0, seconds) / 300) * 300 * disk_gb * .10 / (30 * 86400)
    return {"compute_usd": compute, "disk_usd": disk, "total_usd": compute + disk}


def budget_preflight(gpus, *, budget_usd, trial_seconds, disk_gb, cleanup_seconds=360):
    if not finite(budget_usd, .01) or not 30 <= trial_seconds <= 1800:
        raise ValueError("Budget must be positive and per-GPU time limit must be 30–1800 seconds.")
    envelope = sum(cost_envelope(gpu["hourly_usd"], trial_seconds + cleanup_seconds, disk_gb)["total_usd"]
                   for gpu in gpus)
    reserved = envelope * 1.25
    if reserved > budget_usd:
        raise ValueError(f"Conservative budget reserve ${reserved:.4f} exceeds ${budget_usd:.4f}; reduce trial time.")
    return {"budget_usd": budget_usd, "reserved_usd": reserved,
            "margin_multiplier": 1.25, "per_gpu_seconds": trial_seconds,
            "cleanup_reserve_seconds": cleanup_seconds,
            "note": "Local time/worker safeguards, not a provider-enforced spending cap; prices are list quotes."}


def wait_stopped(api, endpoint_id, deadline):
    scaled_down = {"IDLE", "INITIALIZING", "THROTTLED"}
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        workers = api.request("GET", f"/serverless/{endpoint_id}/workers", timeout=min(20, remaining))
        if not isinstance(workers.get("workers"), list):
            raise ApiError("RunPod worker listing is missing its workers array.")
        if all(isinstance(worker, dict) and
               isinstance(worker.get("status", worker.get("state")), str) and
               worker.get("status", worker.get("state")).upper() in scaled_down
               for worker in workers["workers"]):
            return True
        time.sleep(min(2, max(0, deadline - time.monotonic())))


def read_billing(api, record):
    created = datetime.fromisoformat(record["created_at"].replace("Z", "+00:00"))
    start = created.replace(minute=0, second=0, microsecond=0)
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    from urllib.parse import urlencode
    query = urlencode({"serverlessId": record["id"], "bucketSize": "hour",
                       "startTime": start.isoformat(), "endTime": end.isoformat()})
    response = api.request("GET", "/billing/serverless?" + query)
    records = response.get("records")
    if not isinstance(records, list):
        raise ApiError("Billing response has no records array.")
    if any(row.get("serverlessId") != record["id"] for row in records):
        raise ApiError("Billing returned records for another endpoint; receipt rejected.")
    totals = {}
    for key in ("totalAmount", "gpuAmount", "cpuAmount", "diskAmount", "feeAmount"):
        if any(not finite(row.get(key)) for row in records):
            raise ApiError("Billing contains invalid or missing cost components.")
        totals[key] = sum(row[key] for row in records) if records else None
    return {"queried_at": utc_now(), "source": SOURCES["billing"], "records": records,
            "totals": totals, "observed_billed_usd": totals["totalAmount"], "finalized": False,
            "status": "observed_provider_receipt" if records else "pending_or_no_records",
            "note": "Endpoint-only hourly buckets; may be incomplete until provider billing catches up."
                    " The API exposes no finalization flag. Empty records do not mean zero cost."}


def sample_response(response, wall_ms, case, index, gpu):
    output = response.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("result"), dict):
        raise ValueError("Worker output is missing result.")
    metrics = output.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("cold_start"), bool):
        raise ValueError("Worker output is missing cold-start metrics.")
    actual_gpu = metrics.get("gpu_name")
    if normalize_gpu(actual_gpu) not in (normalize_gpu(gpu["id"]), normalize_gpu(gpu["name"])):
        raise ValueError("Worker GPU differs from the requested exact GPU; comparison stopped.")
    worker_id = metrics.get("worker_id") or response.get("workerId")
    if not isinstance(worker_id, str) or not worker_id:
        raise ValueError("Worker identity is missing; warm repeats cannot be verified.")
    for field in ("handler_ms",):
        if not finite(metrics.get(field)):
            raise ValueError("Worker output is missing valid timing metrics.")
    result = output["result"]
    stable = {key: value for key, value in result.items() if key not in ("elapsed_ms", "annotated_path")}
    sample = {"case_id": case["metadata"]["id"], "repeat": index, "job_id": response.get("id"),
              "worker_id": worker_id, "actual_gpu": actual_gpu, "cold_start": metrics["cold_start"],
              "wall_ms": wall_ms, "handler_ms": metrics["handler_ms"], "metrics": metrics,
              "result": result, "result_sha256": hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()}
    for field in ("executionTime", "delayTime"):
        if not finite(response.get(field)):
            raise ValueError("RunPod status is missing valid execution/delay timing.")
        sample[field] = response[field]
    return sample


def verify_identity(report, sample, action, expected_revision=None):
    """Different hardware may be compared only with identical code and model assets."""
    metrics = sample["metrics"]
    revision, digest, model = metrics.get("build_revision"), metrics.get("model_sha256"), metrics.get("model")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ValueError("Worker lacks a concrete build revision; rebuild with WORKER_BUILD_REVISION.")
    if expected_revision and revision != expected_revision:
        raise ValueError("Worker build revision differs from the requested revision.")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Worker lacks the model asset SHA-256.")
    expected_model = "countgd-plusplus" if action == "count" else "fastsam-s"
    if model != expected_model:
        raise ValueError("Worker used a model other than the requested counting/proposal model.")
    identity = report.setdefault("identity", {"build_revision": revision, "models": {}})
    if identity["build_revision"] != revision or identity["models"].get(model, digest) != digest:
        raise ValueError("Build or model assets changed across trials; comparison stopped.")
    identity["models"][model] = digest


def summarize_trial(trial):
    cases = {}
    for sample in trial.get("samples", []):
        cases.setdefault(sample["case_id"], []).append(sample)
    summaries = {}
    for name, samples in cases.items():
        warm = [sample for sample in samples if sample["repeat"] > 0 and not sample["cold_start"]
                and sample["worker_id"] == samples[0]["worker_id"]]
        summaries[name] = {"first_call_wall_ms": samples[0]["wall_ms"],
                           "first_call_cold_start": samples[0]["cold_start"], "warm_samples": len(warm),
                           "warm_wall_median_ms": statistics.median(s["wall_ms"] for s in warm) if warm else None,
                           "warm_handler_median_ms": statistics.median(s["handler_ms"] for s in warm) if warm else None,
                           "warm_execution_median_ms": statistics.median(s["executionTime"] for s in warm) if warm else None,
                           "counts": [s["result"].get("count", len(s["result"].get("boxes", []))) for s in samples],
                           "results_byte_stable": len({s["result_sha256"] for s in samples}) == 1}
    trial["summary"] = summaries
    trial["completed_jobs"] = len(trial.get("samples", []))
    return trial


def save_report(output, report):
    output = Path(output)
    rows = []
    lines = ["# RunPod GPU comparison", "", f"Status: {report.get('status', 'running')}", "",
             "Latency uses identical original image bytes and request settings. The first request includes cold startup; "
             "warm repeats require the same worker and cold_start=false.", "",
             "Observed billed amounts are provider records, including disk, and may lag. They are not finalized invoices. "
             "The cost envelope charges the entire endpoint lifetime at list rates, including time that may be unbilled. "
             "Per-job amounts allocate the batch's startup and idle overhead across completed jobs.", "",
             "| GPU | Case | First call ms | Warm median ms | Warm samples |",
             "| --- | --- | ---: | ---: | ---: |"]
    for trial in report.get("trials", []):
        summarize_trial(trial)
        for name, summary in trial["summary"].items():
            warm = summary["warm_wall_median_ms"]
            lines.append(f"| {trial['gpu']['name']} | {name} | {summary['first_call_wall_ms']:.1f} | "
                         f"{f'{warm:.1f}' if warm is not None else 'unavailable'} | {summary['warm_samples']} |")
        receipt = trial.get("billing", {})
        observed = receipt.get("observed_billed_usd")
        envelope = trial.get("cost_envelope", {}).get("total_usd")
        completed = trial.get("completed_jobs", 0)
        row = {"gpu": trial["gpu"]["name"], "endpoint_id": trial.get("endpoint_id"),
               "completed_jobs": completed, "observed_billed_usd": observed,
               "observed_billed_per_completed_job_usd": observed / completed if observed is not None and completed else None,
               "lifetime_cost_envelope_usd": envelope,
               "envelope_per_completed_job_usd": envelope / completed if envelope is not None and completed else None,
               "billing_finalized": False}
        rows.append(row)
    lines += ["", "| GPU | Completed jobs | Observed billed USD (may lag) | Lifecycle envelope USD |",
              "| --- | ---: | ---: | ---: |"]
    for row in rows:
        amount = lambda value: f"{value:.6f}" if value is not None else "pending"
        lines.append(f"| {row['gpu']} | {row['completed_jobs']} | {amount(row['observed_billed_usd'])} | "
                     f"{amount(row['lifetime_cost_envelope_usd'])} |")
    if report.get("cleanup_errors"):
        lines += ["", "Cleanup needs attention. Rerun runpod_deploy.py cleanup with resources.json."]
    lines += ["", "Sources: " + ", ".join(f"[{name}]({url})" for name, url in SOURCES.items()), ""]
    report["cost_summary"] = rows
    save_json(output / "report.json", report)
    save_text(output / "report.md", "\n".join(lines))
    stream = io.StringIO(newline="")
    fields = list(rows[0]) if rows else ["gpu", "completed_jobs", "observed_billed_usd"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    save_text(output / "costs.csv", stream.getvalue())
    samples = [{"gpu": trial["gpu"]["name"], **{key: sample[key] for key in
                ("case_id", "repeat", "job_id", "worker_id", "cold_start", "wall_ms", "handler_ms", "executionTime", "delayTime")}}
               for trial in report.get("trials", []) for sample in trial.get("samples", [])]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(samples[0]) if samples else ["gpu", "case_id"])
    writer.writeheader()
    writer.writerows(samples)
    save_text(output / "samples.csv", stream.getvalue())


def benchmark(args, api, client_factory):
    if not re.search(r"@sha256:[0-9a-f]{64}\Z", args.image):
        raise ValueError("Benchmark image must be pinned by an immutable @sha256 digest.")
    cases = prepare_cases(args.fixture)
    catalog = api.catalog()
    gpus = [select_gpu(catalog, name) for name in args.gpu]
    if not 1 <= len(gpus) <= 2 or len({gpu["id"] for gpu in gpus}) != len(gpus):
        raise ValueError("Select one or two distinct GPUs.")
    budget = budget_preflight(gpus, budget_usd=args.budget_usd, trial_seconds=args.trial_seconds, disk_gb=args.disk_gb)
    output = args.output_dir.resolve()
    if (output / "resources.json").exists():
        raise ValueError("Output already has a resource manifest; clean up or use a new output directory.")
    owned = OwnedEndpoints(api, output / "resources.json")
    report = {"schema": 1, "created_at": utc_now(), "image": args.image, "budget": budget,
              "fixture_manifest_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
              "fixture": [case["metadata"] for case in cases], "warm_repeats": args.repeats,
              "trials": [], "sources": SOURCES, "status": "running"}
    keep_id = None
    success = False
    try:
        for gpu in gpus:
            started = time.monotonic()
            deadline = started + args.trial_seconds
            record = owned.create(args.image, gpu, timeout_seconds=min(args.job_seconds, args.trial_seconds),
                                  idle_seconds=args.idle_seconds, disk_gb=args.disk_gb)
            trial = {"gpu": gpu, "endpoint_id": record["id"], "samples": [], "started_at": utc_now()}
            report["trials"].append(trial)
            client = client_factory(api.api_key, record["id"], timeout_seconds=args.job_seconds)
            for case in cases:
                for repeat in range(args.repeats + 1):
                    remaining = deadline - time.monotonic()
                    if remaining < 2:
                        raise TimeoutError("Per-GPU benchmark time limit reached.")
                    client.timeout_seconds = min(args.job_seconds, remaining)
                    before = time.monotonic()
                    response = client.run(case["payload"])
                    sample = sample_response(response, (time.monotonic() - before) * 1000, case, repeat, gpu)
                    trial["samples"].append(sample)
                    verify_identity(report, sample, case["metadata"]["action"], getattr(args, "expected_build_revision", None))
                    if len(trial["samples"]) == 1 and not sample["cold_start"]:
                        raise ValueError("First request was not a verified cold start; comparison stopped.")
                    save_report(output, report)
                    if repeat > 0:
                        prior = [item for item in trial["samples"] if item["case_id"] == case["metadata"]["id"]][0]
                        if sample["cold_start"] or sample["worker_id"] != prior["worker_id"]:
                            raise ValueError("A warm repeat changed workers or cold-started; comparison stopped.")
            # Stop and verify the first GPU before starting the second. Idle workers are billable.
            stopped = wait_stopped(api, record["id"], min(deadline + 60, time.monotonic() + args.idle_seconds + 30))
            if not stopped:
                owned.delete(record)
            trial["workers_stopped_verified"] = stopped
            trial["lifetime_seconds"] = time.monotonic() - started
            trial["cost_envelope"] = cost_envelope(gpu["hourly_usd"], trial["lifetime_seconds"], args.disk_gb)
            save_report(output, report)
        success = True
        if args.keep_winner:
            eligible = [trial for trial in report["trials"] if trial.get("workers_stopped_verified")]
            if not eligible:
                raise ValueError("No endpoint has verified zero active workers; none can be retained.")
            winner = min(eligible, key=lambda trial: sum(case["warm_wall_median_ms"] for case in trial["summary"].values()))
            keep_id = winner["endpoint_id"]
            report["retained_endpoint_id"] = keep_id
            report["selection_basis"] = "Lowest sum of per-case warm end-to-end median latency; cost receipts may be incomplete."
        report["status"] = "completed"
    except BaseException as exc:
        success = False
        keep_id = None
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        # Do not persist exception text from arbitrary libraries or a provider payload.
        report["failure_type"] = type(exc).__name__
        raise
    finally:
        report["cleanup_errors"] = owned.cleanup(keep_id=keep_id if success else None)
        if report["cleanup_errors"]:
            report["status"] = "cleanup_required"
        for trial in report["trials"]:
            record = next(item for item in owned.data["resources"] if item.get("id") == trial["endpoint_id"])
            if "cost_envelope" not in trial:
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(record["created_at"].replace("Z", "+00:00"))).total_seconds()
                trial["lifetime_seconds"] = elapsed
                trial["cost_envelope"] = cost_envelope(trial["gpu"]["hourly_usd"], elapsed, args.disk_gb)
            try:
                trial["billing"] = read_billing(api, record)
            except (ApiError, OSError) as exc:
                trial["billing"] = {"status": "unavailable", "error_type": type(exc).__name__, "finalized": False}
        report["finished_at"] = utc_now()
        save_report(output, report)
    return report


def refresh_billing(api, output):
    output = Path(output)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    owned = json.loads((output / "resources.json").read_text(encoding="utf-8"))
    for trial in report["trials"]:
        record = next(item for item in owned["resources"] if item.get("id") == trial["endpoint_id"])
        trial["billing"] = read_billing(api, record)
    save_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--refresh-billing", action="store_true", help="Read receipts again for an existing report; creates no resources")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "test-results/runpod-benchmark" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--image", help="Published worker container pinned by an immutable @sha256 digest")
    parser.add_argument("--expected-build-revision", help="Require this exact worker Git revision in every response")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--gpu", action="append", help="Select one GPU, or repeat twice for comparison; defaults RTX A5000 and NVIDIA L4")
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--trial-seconds", type=int, default=900, help="Maximum active trial time per GPU")
    parser.add_argument("--job-seconds", type=int, default=300)
    parser.add_argument("--idle-seconds", type=int, default=30)
    parser.add_argument("--disk-gb", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3, help="Warm repetitions per fixture in addition to one first call")
    parser.add_argument("--keep-winner", action="store_true", help="Retain at most one verified idle, min=0 endpoint, selected by warm latency")
    args = parser.parse_args(argv)
    try:
        key = load_api_key(args.env_file)
        api = RunPodApi(key)
        if args.refresh_billing:
            report = refresh_billing(api, args.output_dir)
        else:
            if not args.image or args.fixture is None:
                parser.error("--image and --fixture are required for a benchmark")
            if not 1 <= args.repeats <= 10 or not 1 <= args.job_seconds <= 1800:
                parser.error("repeats must be 1–10 and job-seconds must be 1–1800")
            args.gpu = args.gpu or ["RTX A5000", "NVIDIA L4"]
            args.budget_usd = args.budget_usd if args.budget_usd is not None else float(os.environ.get("RUNPOD_BENCHMARK_BUDGET_USD", "5.00"))
            from app.runpod_client import RunPodClient
            report = benchmark(args, api, RunPodClient)
        print(json.dumps({"status": report["status"], "report": str(args.output_dir / "report.md"),
                          "retained_endpoint_id": report.get("retained_endpoint_id")}))
        return int(bool(report.get("cleanup_errors")))
    except KeyboardInterrupt:
        print("Interrupted; cleanup attempted. Check resources.json and report.json.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Benchmark stopped ({type(exc).__name__}); check saved report and resource manifest.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
