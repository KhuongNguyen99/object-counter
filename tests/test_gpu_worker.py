"""Paired timing workers keep inference warm and preserve every trial's results."""
import io
import json
from types import SimpleNamespace

from scripts.compare_gpu_execution import _serve_timings


def test_worker_keeps_one_engine_and_records_requested_trials(tmp_path, monkeypatch):
    calls = []

    def count(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"count": 0, "detections": [], "width": 10, "height": 10,
                 "annotated_path": "annotation.jpg"}]

    engine = SimpleNamespace(_torch=None, device=SimpleNamespace(type="cpu"), count=count)
    args = SimpleNamespace(output_dir=tmp_path, name="candidate", target="target.jpg",
                           confidence=.5, label="part")
    monkeypatch.setattr("sys.stdin", io.StringIO("1\n2\nquit\n3\n"))
    assert _serve_timings(engine, ["reference"], args, {"device": "cpu"}) == 0
    assert len(calls) == 3  # one full warmup, then two requested trials
    for index in (1, 2):
        record = json.loads((tmp_path / f"trial-{index:02d}.json").read_text())
        assert record["index"] == index and record["merged"]["count"] == 0
        assert record["wall_ms"] >= 0
    assert not (tmp_path / "trial-03.json").exists()
