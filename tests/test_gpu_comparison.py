"""Saved GPU benchmark artifacts can be compared without a GPU or model."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.compare_gpu_execution import _enable_sdpa, compare, main


def _run(directory, name, *, arrays=None, metadata=None, merged=None, counts=None):
    directory.mkdir()
    arrays = arrays or {
        "pred_boxes": np.array([[.5, .5, .2, .2], [.2, .3, .1, .1]], dtype=np.float32),
        "pred_logits": np.array([[4., -np.inf, -4.], [1., -np.inf, -1.]], dtype=np.float32),
        "sigmoid_scores": np.array([[.98, 0., .02], [.73, 0., .27]], dtype=np.float32),
        "input_ids": np.array([101, 1012, 102], dtype=np.int64),
        "selected_mask": np.array([True, False]),
        "selected_boxes": np.array([[.5, .5, .2, .2]], dtype=np.float32),
        "selected_scores": np.array([.98], dtype=np.float32),
    }
    np.savez(directory / "queries.npz", **{f"tile000_{key}": value for key, value in arrays.items()})
    metadata = metadata or {
        "name": name, "reference_sha256": "refhash", "target_sha256": "targethash",
        "reference_box": [1039, 1775, 1344, 2170], "label": "Bolt with red cap",
        "confidence": .5, "device": "cuda:0", "model_dtype": "torch.float32",
        "torch_threads": 4, "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": True,
        "checkpoint_sha256": "modelhash", "source_commit": "officialcommit",
        "versions": {"torch": "x"}, "gpu": {"name": "GPU"}, "repeats": 2,
        "source_patch": f"patch-{name}", "sdpa": name == "candidate",
    }
    merged = merged or {"count": 1, "detections": [
        {"index": 1, "box": [10., 20., 30., 40.], "confidence": .98}],
        "width": 100, "height": 100, "warning": "Experimental model"}
    counts = counts or [1, 1]
    summary = {"metadata": metadata, "tile_count": 1, "selected_tile_counts": [1],
               "median_wall_ms": 10. if name == "baseline" else 8.,
               "peak_cuda_allocated_bytes": 2**30,
               "runs": [{"count": count} for count in counts],
               "verification": {"merged": merged, "annotation": "annotations/result.jpg"},
               "timed_runs_match_verification": all(count == merged["count"] for count in counts)}
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return arrays, metadata, merged


def test_exact_saved_queries_and_results_pass_with_different_patch_and_sdpa(tmp_path):
    baseline, candidate, output = (tmp_path / name for name in ("baseline", "candidate", "report"))
    _run(baseline, "baseline")
    _run(candidate, "candidate")
    result = compare(baseline, candidate, output)
    assert result["passed"]
    assert result["all_query_arrays_byte_exact"]
    assert result["candidate_speedup"] == 1.25
    assert "[view](../baseline/annotations/result.jpg)" in (output / "comparison.md").read_text()
    assert main(["compare", "--baseline", str(baseline), "--candidate", str(candidate),
                 "--output-dir", str(output)]) == 0


@pytest.mark.parametrize("change", ("finite", "nonfinite", "mask", "tokens", "counts", "merged"))
def test_any_query_or_result_drift_fails_but_still_writes_report(tmp_path, change):
    baseline, candidate, output = (tmp_path / name for name in ("baseline", "candidate", "report"))
    arrays, _, merged = _run(baseline, "baseline")
    arrays = {key: value.copy() for key, value in arrays.items()}
    counts = None
    if change == "finite":
        arrays["pred_boxes"][1, 0] += .001
    elif change == "nonfinite":
        arrays["pred_logits"][0, 1] = np.inf
    elif change == "mask":
        arrays["selected_mask"][1] = True
    elif change == "tokens":
        arrays["input_ids"][0] = 102
    elif change == "counts":
        counts = [1, 2]
    else:
        merged = json.loads(json.dumps(merged))
        merged["detections"][0]["box"][0] += .001
    _run(candidate, "candidate", arrays=arrays, merged=merged, counts=counts)
    result = compare(baseline, candidate, output)
    assert not result["passed"]
    assert (output / "comparison.json").is_file()
    assert main(["compare", "--baseline", str(baseline), "--candidate", str(candidate),
                 "--output-dir", str(output)]) == 1
    if change == "finite":
        assert result["max_finite_array_error"] > 0
    if change == "nonfinite":
        assert result["nonfinite_pattern_differences"] == 1
    if change == "counts":
        assert not result["counts_equal"]


def test_different_input_bytes_rejected_before_query_comparison(tmp_path):
    baseline, candidate, output = (tmp_path / name for name in ("baseline", "candidate", "report"))
    _run(baseline, "baseline")
    _, metadata, _ = _run(candidate, "candidate")
    metadata["target_sha256"] = "different image"
    summary_path = candidate / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["metadata"] = metadata
    summary_path.write_text(json.dumps(summary))
    result = compare(baseline, candidate, output)
    assert not result["passed"]
    assert result["incompatible_metadata"] == ["target_sha256"]
    assert result["array_differences"] == {}


def test_sdpa_switch_requests_no_attention_weights_even_when_passed_positionally():
    import torch

    layer = SimpleNamespace(self_attn=torch.nn.MultiheadAttention(8, 2),
                            ca_text=torch.nn.MultiheadAttention(8, 2))
    model = SimpleNamespace(transformer=SimpleNamespace(
        decoder=SimpleNamespace(layers=[layer])))
    assert _enable_sdpa(model, torch) == 1
    query = torch.zeros(2, 1, 8)
    for attention in (layer.self_attn, layer.ca_text):
        _, weights = attention(query, query, query, None, True)
        assert weights is None
