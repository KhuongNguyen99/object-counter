"""Prepare pinned official SAM 3/SAM3Count code and checkpoints for offline use."""

import argparse
import hashlib
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

SAM3_COMMIT = "7f473cfd983d5eb43040ff608f66bff0d7bdcb53"
SAM3_ARCHIVE_SHA256 = "11eb8f032e9ead04118ef54234f23cc852f76475f794e608e29d2350988e97ee"
SAM3COUNT_COMMIT = "1840710099a8b2b8bdf5674c3be8a4367aa76830"
SAM3COUNT_ARCHIVE_SHA256 = "1f138bc0f36bda3c1feaebf701b36bba4057ef6a4e7ecf44a1c6df2cfc704ff7"
SAM3_MODEL_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
SAM3_MODEL_SIZE = 3450062241
SAM3_MODEL_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
SAM3COUNT_FT_ID = "1uSpEeM9l9SN_Q_W-CJqYy75mQ-O75HpL"
SAM3COUNT_FT_SIZE = 3606641522
SAM3COUNT_FT_SHA256 = "9c2383bab0584e1f0600b45da294434f573644566c7c615b2998338286b98615"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid_file(path: Path, size: int, digest: str) -> bool:
    return path.is_file() and path.stat().st_size == size and sha256(path) == digest


def prepare_source(destination: Path, owner: str, repo: str, commit: str, digest: str, marker: str) -> None:
    stamp = destination / ".counter-source-commit"
    if stamp.is_file() and stamp.read_text().strip() == commit and (destination / marker).is_file():
        return
    if destination.exists():
        raise ValueError(f"Source destination has an unexpected version: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/{owner}/{repo}/zip/{commit}"
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        work = Path(temporary)
        archive = work / "source.zip"
        for attempt in range(3):
            try:
                print(f"Downloading official {repo} source {commit}", flush=True)
                request = urllib.request.Request(url, headers={"User-Agent": "reference-object-counter/1.0"})
                with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                if sha256(archive) != digest:
                    raise ValueError(f"{repo} source failed SHA-256 validation")
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                path = PurePosixPath(member)
                if path.is_absolute() or ".." in path.parts or "\\" in member:
                    raise ValueError(f"Invalid {repo} archive path")
            bundle.extractall(work)
        source = work / f"{repo}-{commit}"
        if not (source / marker).is_file() or not (source / "LICENSE").is_file():
            raise ValueError(f"{repo} source is missing expected files")
        (source / ".counter-source-commit").write_text(commit + "\n")
        shutil.move(str(source), str(destination))


def prepare_finetuned(directory: Path) -> None:
    destination = directory / "sam3_ft.pt"
    if valid_file(destination, SAM3COUNT_FT_SIZE, SAM3COUNT_FT_SHA256):
        print("Verified sam3_ft.pt", flush=True)
        return
    import gdown

    directory.mkdir(parents=True, exist_ok=True)
    partial = directory / "sam3_ft.pt.part"
    print("Downloading official SAM3Count fine-tuned checkpoint", flush=True)
    if not gdown.download(id=SAM3COUNT_FT_ID, output=str(partial), quiet=False, resume=True):
        raise RuntimeError("Official SAM3Count fine-tuned checkpoint download failed")
    if not valid_file(partial, SAM3COUNT_FT_SIZE, SAM3COUNT_FT_SHA256):
        raise ValueError("SAM3Count fine-tuned checkpoint failed size/SHA-256 validation")
    partial.replace(destination)
    print("Verified sam3_ft.pt", flush=True)


def valid_base(directory: Path) -> bool:
    return valid_file(directory / "sam3.pt", SAM3_MODEL_SIZE, SAM3_MODEL_SHA256)


def prepare_base(directory: Path, supplied: Path | None = None) -> None:
    from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_download, hf_hub_url
    from huggingface_hub.errors import GatedRepoError

    destination = directory / "sam3.pt"
    if valid_base(directory):
        print("Verified sam3.pt", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    if supplied is not None:
        if not supplied.is_file() or supplied.stat().st_size != SAM3_MODEL_SIZE:
            raise ValueError("User-supplied SAM 3 checkpoint has the wrong size")
        with zipfile.ZipFile(supplied) as bundle:
            if not any(name.endswith("/data.pkl") for name in bundle.namelist()):
                raise ValueError("User-supplied SAM 3 checkpoint is not a PyTorch state archive")
        if sha256(supplied) != SAM3_MODEL_SHA256:
            raise ValueError("User-supplied SAM 3 checkpoint failed pinned SHA-256 validation")
        if not destination.exists() or not supplied.samefile(destination):
            partial = destination.with_suffix(".pt.part")
            shutil.copyfile(supplied, partial)
            if not valid_file(partial, SAM3_MODEL_SIZE, SAM3_MODEL_SHA256):
                raise ValueError("Copied SAM 3 checkpoint failed size/SHA-256 validation")
            partial.replace(destination)
        print("Verified pinned user-supplied sam3.pt", flush=True)
        return
    token = get_token()
    if not token:
        raise RuntimeError(
            "Official SAM 3 base weights require approved facebook/sam3 access and an HF token, "
            "or --base-checkpoint pointing to an approved local sam3.pt"
        )
    try:
        url = hf_hub_url("facebook/sam3", "sam3.pt", revision=SAM3_MODEL_REVISION)
        metadata = get_hf_file_metadata(url, token=token)
        if metadata.size != SAM3_MODEL_SIZE or metadata.etag.lower() != SAM3_MODEL_SHA256:
            raise ValueError("Official SAM 3 checkpoint metadata differs from the pinned version")
        if not valid_base(directory):
            hf_hub_download(
                repo_id="facebook/sam3",
                filename="sam3.pt",
                revision=SAM3_MODEL_REVISION,
                local_dir=directory,
            )
        if not valid_base(directory):
            raise ValueError("SAM 3 checkpoint failed official LFS SHA-256 validation")
        print("Verified sam3.pt", flush=True)
    except GatedRepoError as error:
        raise RuntimeError(
            "Official SAM 3 weights require approved access to facebook/sam3 and an authenticated HF token"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-source-dir", type=Path, default=Path("/opt/sam3"))
    parser.add_argument("--sam3count-source-dir", type=Path, default=Path("/opt/sam3count"))
    parser.add_argument("--model-dir", type=Path, default=Path("/app/models/sam3count"))
    parser.add_argument("--base-checkpoint", type=Path, help="Approved official sam3.pt supplied by the user")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--source-only", action="store_true")
    mode.add_argument("--assets-only", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.assets_only and not args.verify_only:
        prepare_source(
            args.sam3_source_dir, "facebookresearch", "sam3", SAM3_COMMIT,
            SAM3_ARCHIVE_SHA256, "sam3/model_builder.py",
        )
        prepare_source(
            args.sam3count_source_dir, "Joan947", "SAM3Count", SAM3COUNT_COMMIT,
            SAM3COUNT_ARCHIVE_SHA256, "sam3count_images.py",
        )
    if args.source_only:
        return
    if not args.verify_only:
        prepare_finetuned(args.model_dir)
        prepare_base(args.model_dir, args.base_checkpoint)
    else:
        if not valid_file(args.model_dir / "sam3_ft.pt", SAM3COUNT_FT_SIZE, SAM3COUNT_FT_SHA256):
            raise ValueError("Invalid SAM3Count fine-tuned checkpoint")
        if not valid_base(args.model_dir):
            raise ValueError("Invalid SAM 3 base checkpoint")


if __name__ == "__main__":
    main()
