import hashlib

import pytest

from app import models


def test_sam3count_integrity_remains_available_for_offline_research(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("SAM3_SOURCE_DIR", str(tmp_path / "source"))
    monkeypatch.setenv("SAM3COUNT_SOURCE_DIR", str(tmp_path / "counter-source"))
    contents = {"sam3.pt": b"trusted base", "sam3_ft.pt": b"trusted overlay"}
    monkeypatch.setattr(models, "SAM3COUNT_ASSETS", {
        name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in contents.items()})
    assert [entry["id"] for entry in models.list_models()] == [models.COUNTGD_MODEL_ID]
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
    assert [entry["id"] for entry in models.list_models()] == [models.COUNTGD_MODEL_ID]
    path = models.sam3count_dir() / "sam3_ft.pt"
    path.write_bytes(b"x" * len(contents[path.name]))
    with pytest.raises(ValueError, match="SHA-256"):
        models.validate_sam3count_assets()
