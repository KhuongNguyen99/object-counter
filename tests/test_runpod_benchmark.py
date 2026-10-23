"""Benchmark latency, billing, budget and cleanup semantics, without GPU jobs."""

import base64
import json
from types import SimpleNamespace

import pytest

from scripts.runpod_benchmark import (benchmark, budget_preflight, cost_envelope,
                                     prepare_cases, read_billing, sample_response, summarize_trial, verify_identity)
from scripts.runpod_deploy import ApiError, select_gpu
from test_runpod_deploy import CATALOG, FakeApi


def fixture(tmp_path):
    image = tmp_path / "original.png"
    image.write_bytes(b"original-image-bytes")
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps({"cases": [{"id": "fitting", "image": "original.png", "action": "count",
                                          "boxes": [[1, 2, 3, 4]], "confidence": .80}]}))
    return path


def response(gpu="NVIDIA RTX A5000", *, cold=False, worker="worker-1", count=7):
    return {"id": "job-1", "status": "COMPLETED", "executionTime": 20, "delayTime": 5,
            "workerId": worker,
            "output": {"result": {"count": count, "detections": [], "elapsed_ms": 15},
                       "metrics": {"gpu_name": gpu, "worker_id": worker, "cold_start": cold, "handler_ms": 15,
                                   "build_revision": "a" * 40, "model_sha256": "b" * 64, "model": "countgd-plusplus"}}}


def test_prepared_requests_keep_original_bytes_and_confidence(tmp_path):
    cases = prepare_cases(fixture(tmp_path))
    case = cases[0]
    assert base64.b64decode(case["payload"]["image_base64"]) == b"original-image-bytes"
    assert case["payload"]["confidence"] == .80
    assert case["metadata"]["boxes"] == [[1, 2, 3, 4]]
    assert "image_base64" not in case["metadata"]


def test_budget_reserves_full_trial_plus_cleanup_and_storage():
    gpus = [select_gpu(CATALOG, name) for name in ("L4", "RTX A5000")]
    with pytest.raises(ValueError, match="budget reserve"):
        budget_preflight(gpus, budget_usd=.01, trial_seconds=900, disk_gb=30)
    result = budget_preflight(gpus, budget_usd=5, trial_seconds=900, disk_gb=30)
    assert result["reserved_usd"] > sum(gpu["hourly_usd"] * 900 / 3600 for gpu in gpus)
    envelope = cost_envelope(1, 1.1, 30)
    assert envelope["compute_usd"] == 2 / 3600
    assert envelope["disk_usd"] == 300 * 30 * .10 / (30 * 86400)


def test_wrong_actual_gpu_cannot_be_mixed_into_comparison(tmp_path):
    case = prepare_cases(fixture(tmp_path))[0]
    with pytest.raises(ValueError, match="Worker GPU differs"):
        sample_response(response(gpu="NVIDIA L4"), 50, case, 0, select_gpu(CATALOG, "RTX A5000"))


def test_warm_summary_excludes_cold_requests_and_other_workers(tmp_path):
    case = prepare_cases(fixture(tmp_path))[0]
    gpu = select_gpu(CATALOG, "RTX A5000")
    samples = [sample_response(response(cold=cold, worker=worker), wall, case, index, gpu)
               for index, (cold, worker, wall) in enumerate(
                   [(True, "w1", 1000), (False, "w1", 40), (True, "w1", 800), (False, "w2", 2), (False, "w1", 60)])]
    summary = summarize_trial({"samples": samples})["summary"]["fitting"]
    assert summary["first_call_wall_ms"] == 1000
    assert summary["warm_samples"] == 2
    assert summary["warm_wall_median_ms"] == 50


def test_empty_billing_is_pending_not_zero_and_receipts_include_disk():
    record = {"id": "owned-1", "created_at": "2026-10-22T12:10:00Z"}
    api = FakeApi()
    assert read_billing(api, record)["observed_billed_usd"] is None

    class BillingApi:
        def request(self, method, path):
            assert "serverlessId=owned-1" in path
            return {"records": [{"serverlessId": "owned-1", "totalAmount": .1234567,
                                  "gpuAmount": .12, "cpuAmount": 0, "diskAmount": .0034567, "feeAmount": 0}]}

    receipt = read_billing(BillingApi(), record)
    assert receipt["observed_billed_usd"] == .1234567
    assert receipt["finalized"] is False
    assert receipt["totals"]["diskAmount"] == .0034567


def test_billing_for_another_endpoint_is_rejected():
    class WrongBilling:
        def request(self, *args):
            return {"records": [{"serverlessId": "unrelated"}]}

    with pytest.raises(ApiError, match="another endpoint"):
        read_billing(WrongBilling(), {"id": "owned", "created_at": "2026-10-22T12:10:00Z"})


def arguments(tmp_path):
    return SimpleNamespace(fixture=fixture(tmp_path), gpu=["RTX A5000", "NVIDIA L4"], budget_usd=5,
                           trial_seconds=60, disk_gb=30, output_dir=tmp_path / "report", image="worker@sha256:" + "d" * 64,
                           job_seconds=30, idle_seconds=5, repeats=2, keep_winner=False)


def test_benchmark_runs_identical_requests_and_deletes_all_endpoints(tmp_path):
    args, api, calls = arguments(tmp_path), FakeApi(), []

    class Client:
        def __init__(self, key, endpoint_id, **kwargs):
            self.gpu = "NVIDIA RTX A5000" if len(api.live) == 1 else "NVIDIA L4"
            self.index = 0

        def run(self, payload):
            calls.append(payload)
            self.index += 1
            return response(self.gpu, cold=self.index == 1)

    report = benchmark(args, api, Client)
    assert report["status"] == "completed"
    assert len(calls) == 6 and all(call == calls[0] for call in calls)
    assert not api.live
    assert all(trial["summary"]["fitting"]["warm_samples"] == 2 for trial in report["trials"])
    assert all(row["observed_billed_usd"] is None for row in report["cost_summary"])
    assert (args.output_dir / "samples.csv").is_file()
    assert "not finalized invoices" in (args.output_dir / "report.md").read_text()


@pytest.mark.parametrize("failure", [TimeoutError, KeyboardInterrupt])
def test_failure_or_interrupt_deletes_owned_endpoint_and_preserves_report(tmp_path, failure):
    args, api = arguments(tmp_path), FakeApi()

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, payload):
            raise failure()

    with pytest.raises(failure):
        benchmark(args, api, Client)
    assert not api.live
    saved = json.loads((args.output_dir / "report.json").read_text())
    assert saved["status"] in ("failed", "interrupted")
    assert not saved["cleanup_errors"]
    manifest = json.loads((args.output_dir / "resources.json").read_text())
    assert manifest["resources"][0]["state"] == "deleted"


def test_keep_winner_leaves_only_one_verified_idle_endpoint(tmp_path):
    args, api = arguments(tmp_path), FakeApi()
    args.keep_winner = True

    class Client:
        def __init__(self, key, endpoint_id, **kwargs):
            self.gpu = "NVIDIA RTX A5000" if len(api.live) == 1 else "NVIDIA L4"
            self.index = 0

        def run(self, payload):
            self.index += 1
            return response(self.gpu, cold=self.index == 1)

    report = benchmark(args, api, Client)
    assert len(api.live) == 1
    retained = api.live[report["retained_endpoint_id"]]
    assert retained["workers"]["min"] == 0
    assert retained["workers"]["max"] == 1


def test_model_or_build_drift_is_rejected():
    report = {}
    sample = {"metrics": response()["output"]["metrics"]}
    verify_identity(report, sample, "count", "a" * 40)
    sample["metrics"]["model_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="changed across trials"):
        verify_identity(report, sample, "count")
    sample["metrics"]["build_revision"] = "unknown"
    with pytest.raises(ValueError, match="concrete build revision"):
        verify_identity(report, sample, "count")


def test_mutable_image_is_rejected_before_any_resource_call(tmp_path):
    args, api = arguments(tmp_path), FakeApi()
    args.image = "worker:latest"
    with pytest.raises(ValueError, match="immutable"):
        benchmark(args, api, None)
    assert not api.calls
