import hashlib

import pytest
import torch

from app import models


def test_sam3count_integrity_and_catalog_require_both_weights_sources_and_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("SAM3_SOURCE_DIR", str(tmp_path / "source"))
    monkeypatch.setenv("SAM3COUNT_SOURCE_DIR", str(tmp_path / "counter-source"))
    monkeypatch.setenv("DEVICE", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    contents = {"sam3.pt": b"trusted base", "sam3_ft.pt": b"trusted overlay"}
    monkeypatch.setattr(models, "SAM3COUNT_ASSETS", {
        name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in contents.items()})
    entry = lambda: next(item for item in models.list_models() if item["id"] == "sam3count")
    assert not entry()["available"]
    with pytest.raises(FileNotFoundError, match="missing"):
        models.validate_sam3count_assets()
    for name, data in contents.items():
        path = models.sam3count_dir() / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for name in ("source/sam3/model_builder.py", "source/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                 "counter-source/sam3count_images.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source")
    models.validate_sam3count_assets()
    assert entry()["available"] and entry()["reference_mode"] == "none"
    monkeypatch.setenv("DEVICE", "cpu")
    assert not entry()["available"]
    path = models.sam3count_dir() / "sam3_ft.pt"
    path.write_bytes(b"x" * len(contents[path.name]))
    with pytest.raises(ValueError, match="SHA-256"):
        models.validate_sam3count_assets()
