"""Parent/worker failure recovery and PET-DINO catalog/model-cache integration."""

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from app import models, petdino
from app.inference import InferenceEngine


@pytest.fixture
def worker(monkeypatch):
    processes = []
    real_popen = subprocess.Popen
    monkeypatch.setattr(petdino, "validate_petdino_assets", lambda path: None)
    monkeypatch.setenv("PETDINO_PYTHON", sys.executable)

    def start(code):
        def factory(command, **kwargs):
            assert command[1:4] == ["-u", "-m", "app.petdino_runtime"]
            process = real_popen([sys.executable, "-u", "-c", code], **kwargs)
            processes.append(process)
            return process
        monkeypatch.setattr(petdino.subprocess, "Popen", factory)
        return petdino.PetDINOEngine(Path("models/petdino"), "cpu")

    start.processes = processes
    yield start
    for process in processes:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_worker_reuses_process_and_sends_fresh_batch_references(worker):
    engine = worker('''
import json,sys
print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin:
    request=json.loads(line)
    if request['action']=='close': break
    print(json.dumps({'ok':True,'results':[{'reference':request['references'][0],
          'label':request['label']} for _ in request['image_paths']]}),flush=True)
''')
    first = [{"path": "first.jpg", "boxes": [[1, 2, 3, 4]]}]
    second = [{"path": "second.jpg", "boxes": [[4, 5, 6, 7]]}]
    process = engine._process
    try:
        assert engine.count(first, ["target.jpg"], "output", .3, "ring")[0] == {
            "reference": first[0], "label": "ring"}
        assert engine.count(second, ["a.jpg", "b.jpg"], "output", .4, "fitting") == [
            {"reference": second[0], "label": "fitting"}] * 2
        assert engine._process is process
    finally:
        engine.close()
    assert process.poll() == 0
    engine.close()


@pytest.mark.parametrize("code,message", [
    ("print('{bad json',flush=True)", "invalid worker response"),
    ("print('{\"ready\":false,\"error\":\"missing weights\"}',flush=True)", "missing weights"),
    ("pass", "worker exited"),
])
def test_worker_startup_failures_are_reported_and_process_reaped(worker, code, message):
    with pytest.raises(RuntimeError, match=message):
        worker(code)
    assert worker.processes[-1].poll() is not None


def test_worker_reports_count_failure_and_wrong_result_count(worker):
    engine = worker('''
import json,sys
print('{"ready":true}',flush=True)
for line in sys.stdin:
    request=json.loads(line)
    if request['action']=='close': break
    print(json.dumps({'ok':True,'results':[]} if request['label']=='bad length'
          else {'ok':False,'error':'CUDA out of memory'}),flush=True)
''')
    try:
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            engine.count([], ["a.jpg"], "output")
        with pytest.raises(RuntimeError, match="wrong number"):
            engine.count([], ["a.jpg"], "output", label="bad length")
    finally:
        engine.close()


def test_worker_timeout_terminates_hung_process(worker):
    engine = worker('''
import time
print('{"ready":true}',flush=True)
time.sleep(60)
''')
    with pytest.raises(RuntimeError, match="stopped responding"):
        engine._receive(.01)
    assert engine._process.poll() is not None


def test_petdino_integrity_and_catalog_need_checkpoint_bert_source_and_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    source = tmp_path / "source/configs/pet_dino"
    source.mkdir(parents=True)
    (source / "pet_dino_swin-t_8xb4_12e_obj365.py").write_text("config")
    monkeypatch.setenv("PETDINO_SOURCE_DIR", str(tmp_path / "source"))
    monkeypatch.setenv("PETDINO_PYTHON", sys.executable)
    checkpoint = b"trusted weights"
    bert = b"trusted bert"
    monkeypatch.setattr(models, "PETDINO_ASSETS", {
        "checkpoint.pth": (len(checkpoint), hashlib.sha256(checkpoint).hexdigest())})
    monkeypatch.setattr(models, "COUNTGD_ASSETS", {
        "bert-base-uncased/config.json": (len(bert), hashlib.sha256(bert).hexdigest())})

    def entry():
        return next(m for m in models.list_models() if m["id"] == models.PETDINO_MODEL_ID)

    assert not entry()["available"]
    with pytest.raises(FileNotFoundError, match="missing"):
        models.validate_petdino_assets()
    models.petdino_dir().mkdir()
    (models.petdino_dir() / "checkpoint.pth").write_bytes(checkpoint)
    path = models.countgd_dir() / "bert-base-uncased/config.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(bert)
    models.validate_petdino_assets()
    assert entry()["available"] and entry()["default_confidence"] == .3
    path.write_bytes(b"x" * len(bert))
    with pytest.raises(ValueError, match="SHA-256"):
        models.validate_petdino_assets()
    monkeypatch.setenv("PETDINO_PYTHON", str(tmp_path / "missing-python"))
    assert not entry()["available"]


def test_petdino_cache_closes_worker_on_model_switch_shutdown_and_failure(tmp_path, monkeypatch):
    state = []

    class FakePET:
        def __init__(self, directory, device):
            self.closed = False
            self.device = device
            state.append(self)

        def close(self):
            self.closed = True

        def count(self, refs, paths, out, confidence, label):
            if label == "fail":
                raise RuntimeError("worker failed")
            return [label]

    monkeypatch.setattr(petdino, "PetDINOEngine", FakePET)
    monkeypatch.setenv("DEVICE", "0")
    engine = InferenceEngine()
    args = (models.PETDINO_MODEL_ID, [{"path": "reference.jpg", "boxes": [[1, 2, 3, 4]]}],
            ["target.jpg"], str(tmp_path))
    assert engine.count(*args, label="first") == ["first"]
    assert engine.count(*args, label="next") == ["next"]
    assert len(state) == 1 and state[0].device == "cuda:0"
    from app import countgd

    class FakeCountGD:
        def __init__(self, *args):
            pass

        def count(self, *args):
            return []

    monkeypatch.setattr(countgd, "CountGDEngine", FakeCountGD)
    engine.count(models.COUNTGD_MODEL_ID, *args[1:])
    assert state[0].closed and engine._model_id == models.COUNTGD_MODEL_ID
    engine.count(*args, label="back to PET")
    engine.close()
    assert state[1].closed and engine._model is None
    with pytest.raises(RuntimeError, match="worker failed"):
        engine.count(*args, label="fail")
    assert state[2].closed and engine._model_id is None
    engine.count(*args, label="retry")
    assert len(state) == 4
    engine.close()
