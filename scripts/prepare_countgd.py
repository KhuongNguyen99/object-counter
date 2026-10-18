"""Prepare pinned CountGD++ source, official checkpoint, and offline BERT assets."""

import argparse
import hashlib
import http.client
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models import COUNTGD_ASSETS, countgd_dir, validate_countgd_assets
from scripts.patch_countgd import patch_source

SOURCE_COMMIT = "8489f836d75eb407ae778c56cff1f72cc6f8baae"
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
CHECKPOINT_ID = "1j6N22TtKu2NVcKpgfrf-sJHGeLDqs9hs"


def valid_asset(path: Path, filename: str) -> bool:
    size, digest = COUNTGD_ASSETS[filename]
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == digest


def prepare_source(destination: Path) -> None:
    stamp = destination / ".counter-source-commit"
    if stamp.is_file() and stamp.read_text().strip() == SOURCE_COMMIT:
        patch_source(destination)
        return
    if destination.exists():
        raise ValueError(f"Source destination already exists without the expected version: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        work = Path(temporary)
        archive = work / "source.zip"
        url = f"https://codeload.github.com/niki-amini-naieni/CountGDPlusPlus/zip/{SOURCE_COMMIT}"
        print(f"Downloading official CountGD++ source {SOURCE_COMMIT}", flush=True)
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "reference-object-counter/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                if not zipfile.is_zipfile(archive):
                    raise ValueError("Source download is not a ZIP archive")
                break
            except (OSError, http.client.HTTPException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                path = PurePosixPath(member)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Invalid source archive path")
            bundle.extractall(work)
        source = work / f"CountGDPlusPlus-{SOURCE_COMMIT}"
        patch_source(source)
        (source / ".counter-source-commit").write_text(SOURCE_COMMIT + "\n")
        shutil.move(str(source), str(destination))


def prepare_assets(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    filename = "countgd_plusplus.pth"
    if not valid_asset(directory / filename, filename):
        import gdown
        partial = directory / (filename + ".part")
        print("Downloading official CountGD++ checkpoint (1.25 GB)", flush=True)
        if not gdown.download(id=CHECKPOINT_ID, output=str(partial), quiet=False, resume=True):
            raise RuntimeError("The official CountGD++ download failed")
        if not valid_asset(partial, filename):
            raise ValueError("Downloaded CountGD++ checkpoint failed SHA-256 validation")
        partial.replace(directory / filename)
    bert_files = [filename for filename in COUNTGD_ASSETS if filename.startswith("bert-base-uncased/")]
    if not all(valid_asset(directory / filename, filename) for filename in bert_files):
        from transformers import AutoTokenizer, BertConfig, BertModel
        print("Preparing BERT using the official CountGD++ recipe and pinned upstream revision", flush=True)
        upstream = "bert-base-uncased"
        config = BertConfig.from_pretrained(upstream, revision=BERT_REVISION)
        model = BertModel.from_pretrained(upstream, revision=BERT_REVISION, add_pooling_layer=False, config=config)
        tokenizer = AutoTokenizer.from_pretrained(upstream, revision=BERT_REVISION)
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            target = Path(temporary)
            config.save_pretrained(target)
            model.save_pretrained(target)
            tokenizer.save_pretrained(target)
            for filename in bert_files:
                if not valid_asset(target / Path(filename).name, filename):
                    raise ValueError(f"Prepared BERT artifact failed SHA-256 validation: {filename}")
            destination = directory / "bert-base-uncased"
            destination.mkdir(exist_ok=True)
            for filename in bert_files:
                (target / Path(filename).name).replace(directory / filename)
    validate_countgd_assets(directory)
    print("All CountGD++ and BERT assets verified", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/opt/countgd"))
    parser.add_argument("--model-dir", type=Path, default=countgd_dir())
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        validate_countgd_assets(args.model_dir)
        print("All CountGD++ and BERT assets verified", flush=True)
    else:
        prepare_source(args.source_dir)
        prepare_assets(args.model_dir)


if __name__ == "__main__":
    main()
