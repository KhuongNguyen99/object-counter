"""SAM3Count shares the existing single-model cache and explicit prompt contract."""

import pytest

from app import models, sam3count
from app.inference import InferenceEngine


def test_text_only_contract_and_cache_cleanup(tmp_path, monkeypatch):
    loaded = []

    class FakeSAM:
        def __init__(self, directory, device):
            self.closed = False
            loaded.append(self)

        def count(self, refs, paths, output, confidence, label):
            if label == "fail":
                raise RuntimeError("inference failed")
            return [label]

        def close(self):
            self.closed = True

    monkeypatch.setattr(sam3count, "Sam3CountEngine", FakeSAM)
    monkeypatch.setenv("DEVICE", "cpu")
    engine = InferenceEngine()
    args = (models.SAM3COUNT_MODEL_ID, [], ["target.jpg"], str(tmp_path))
    for refs, label in (([], ""), ([{"path": "ref.jpg"}], "rings")):
        with pytest.raises(ValueError, match="Object description"):
            engine.count(args[0], refs, *args[2:], label=label)
    assert loaded == []
    assert engine.count(*args, label="rings") == ["rings"]
    assert engine.count(*args, label="parts") == ["parts"]
    assert len(loaded) == 1
    with pytest.raises(RuntimeError, match="inference failed"):
        engine.count(*args, label="fail")
    assert loaded[0].closed and engine._model_id is None
    assert engine.count(*args, label="retry") == ["retry"]
    engine.close()
    assert loaded[1].closed
