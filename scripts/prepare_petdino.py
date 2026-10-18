"""Prepare pinned PET-DINO source and checkpoint for offline inference."""

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

SOURCE_COMMIT = "7830a462f9a320f34293cce0aabdda1256d9dc15"
SOURCE_SHA256 = "00f5cea550d87adad94ba23fddf2738f4144f48751117e9a7a8003c4265ef0d5"
CHECKPOINT_REVISION = "a8d8cddbe752bbcc3df299b2fe56d0aa67a26776"
CONFIG_PATH = "configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py"


def valid_file(path: Path, expected_size: int, expected_hash: str) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == expected_hash


def prepare_source(destination: Path) -> None:
    stamp = destination / ".counter-source-commit"
    if stamp.is_file() and stamp.read_text().strip() == SOURCE_COMMIT and (destination / CONFIG_PATH).is_file():
        return
    if destination.exists():
        raise ValueError(f"Source destination already exists without the expected version: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        work = Path(temporary)
        archive = work / "source.zip"
        url = f"https://codeload.github.com/fuweifuvtoo/PET_DINO/zip/{SOURCE_COMMIT}"
        print(f"Downloading official PET-DINO source {SOURCE_COMMIT}", flush=True)
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "reference-object-counter/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                with archive.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != SOURCE_SHA256:
                        raise ValueError("PET-DINO source failed SHA-256 validation")
                break
            except (OSError, http.client.HTTPException, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                path = PurePosixPath(member)
                if path.is_absolute() or ".." in path.parts or "\\" in member:
                    raise ValueError("Invalid source archive path")
            bundle.extractall(work)
        source = work / f"PET_DINO-{SOURCE_COMMIT}"
        if not (source / CONFIG_PATH).is_file():
            raise ValueError("PET-DINO source is missing its inference configuration")
        (source / ".counter-source-commit").write_text(SOURCE_COMMIT + "\n")
        shutil.move(str(source), str(destination))


def download_checkpoint(directory: Path, filename: str, expected_size: int, expected_hash: str) -> None:
    destination = directory / filename
    if valid_file(destination, expected_size, expected_hash):
        print(f"Verified {filename}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    url = f"https://huggingface.co/fuweifu/PET-DINO/resolve/{CHECKPOINT_REVISION}/{filename}"
    for attempt in range(1, 5):
        try:
            offset = partial.stat().st_size if partial.is_file() else 0
            if offset == expected_size and valid_file(partial, expected_size, expected_hash):
                partial.replace(destination)
                return
            if offset >= expected_size:
                partial.unlink()
                offset = 0
            headers = {"User-Agent": "reference-object-counter/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            print(f"Downloading PET-DINO ({expected_size / 1e9:.2f} GB, resume {offset / 1e6:.1f} MB)", flush=True)
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned an invalid resume offset")
                    mode = "ab"
                else:
                    offset = 0
                    mode = "wb"
                content_type = response.headers.get("Content-Type", "").lower()
                if "html" in content_type or "json" in content_type:
                    raise ValueError("Server returned a web page instead of model weights")
                with partial.open(mode) as output:
                    while chunk := response.read(1024 * 1024):
                        offset += len(chunk)
                        if offset > expected_size:
                            raise ValueError("Download exceeded the expected checkpoint size")
                        output.write(chunk)
            if not valid_file(partial, expected_size, expected_hash):
                raise ValueError("PET-DINO checkpoint failed SHA-256 validation")
            partial.replace(destination)
            print(f"Verified {filename}", flush=True)
            return
        except (OSError, http.client.HTTPException, ValueError) as error:
            if partial.is_file() and partial.stat().st_size >= expected_size:
                partial.unlink()
            if attempt == 4:
                raise RuntimeError(f"Could not download {filename}: {error}") from error
            print(f"Retry {attempt}/3: {error}", flush=True)
            time.sleep(attempt * 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/opt/petdino"))
    parser.add_argument("--model-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--source-only", action="store_true")
    mode.add_argument("--assets-only", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only and not args.assets_only:
        prepare_source(args.source_dir)
    if args.source_only:
        return
    from app.models import PETDINO_ASSETS, petdino_dir, validate_petdino_assets
    directory = args.model_dir or petdino_dir()
    if not args.verify_only:
        for filename, (size, digest) in PETDINO_ASSETS.items():
            download_checkpoint(directory, filename, size, digest)
    validate_petdino_assets(directory)
    print("All PET-DINO and shared BERT assets verified", flush=True)


if __name__ == "__main__":
    main()
