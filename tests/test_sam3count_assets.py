"""Integrity checks for the approved SAM 3 checkpoint preparation path."""

import hashlib
import zipfile

import pytest

from scripts import prepare_sam3count as prep


def _checkpoint(path):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"checkpoint metadata")
    return path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def test_user_supplied_base_is_copied_only_after_pinned_hash_matches(tmp_path, monkeypatch):
    source = tmp_path / "approved.pt"
    size, digest = _checkpoint(source)
    monkeypatch.setattr(prep, "SAM3_MODEL_SIZE", size)
    monkeypatch.setattr(prep, "SAM3_MODEL_SHA256", digest)

    destination = tmp_path / "models"
    prep.prepare_base(destination, source)
    assert prep.valid_base(destination)
    assert (destination / "sam3.pt").read_bytes() == source.read_bytes()

    (destination / "sam3.pt").write_bytes(b"tampered")
    assert not prep.valid_base(destination)


def test_user_supplied_base_rejects_changed_bytes(tmp_path, monkeypatch):
    source = tmp_path / "approved.pt"
    size, _ = _checkpoint(source)
    monkeypatch.setattr(prep, "SAM3_MODEL_SIZE", size)
    monkeypatch.setattr(prep, "SAM3_MODEL_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="pinned SHA-256"):
        prep.prepare_base(tmp_path / "models", source)
    assert not (tmp_path / "models" / "sam3.pt").exists()


def test_missing_base_without_auth_fails_before_download(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    with pytest.raises(RuntimeError, match="approved facebook/sam3 access"):
        prep.prepare_base(tmp_path / "models")
