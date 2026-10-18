"""Asset preparation must preserve verified files and reject damaged downloads."""

import hashlib
from io import BytesIO
import zipfile

import pytest

from scripts import prepare_petdino as prepare


class Response(BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


def test_verified_checkpoint_needs_no_network(tmp_path, monkeypatch):
    content = b"verified model bytes"
    path = tmp_path / "model.pth"
    path.write_bytes(content)
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Unexpected download"))
    prepare.download_checkpoint(tmp_path, path.name, len(content), hashlib.sha256(content).hexdigest())
    assert path.read_bytes() == content


@pytest.mark.parametrize("honors_range", [True, False])
def test_checkpoint_resumes_or_restarts_safely(tmp_path, monkeypatch, honors_range):
    content = b"complete checkpoint"
    (tmp_path / "model.pth.part").write_bytes(content[:5])

    def download(request, **kwargs):
        assert request.get_header("Range") == "bytes=5-"
        if honors_range:
            return Response(content[5:], 206, {"Content-Range": f"bytes 5-{len(content)-1}/{len(content)}"})
        return Response(content)

    monkeypatch.setattr(prepare.urllib.request, "urlopen", download)
    prepare.download_checkpoint(tmp_path, "model.pth", len(content), hashlib.sha256(content).hexdigest())
    assert (tmp_path / "model.pth").read_bytes() == content
    assert not (tmp_path / "model.pth.part").exists()


def test_corrupt_download_never_replaces_existing_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "model.pth"
    path.write_bytes(b"original")
    monkeypatch.setattr(prepare.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **kw: Response(b"bad"))
    with pytest.raises(RuntimeError, match="SHA-256"):
        prepare.download_checkpoint(tmp_path, path.name, 3, hashlib.sha256(b"new").hexdigest())
    assert path.read_bytes() == b"original"


def test_source_rejects_archive_traversal(tmp_path, monkeypatch):
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside", "untrusted")
    data = archive.getvalue()
    monkeypatch.setattr(prepare, "SOURCE_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **kw: Response(data))
    with pytest.raises(ValueError, match="Invalid source archive path"):
        prepare.prepare_source(tmp_path / "source")
    assert not (tmp_path / "outside").exists()
    assert not (tmp_path / "source").exists()
