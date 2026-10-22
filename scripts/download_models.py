"""Download or verify bundled YOLOE and FastSAM checkpoints without importing torch."""

import argparse
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models import ASSET_RELEASE, FASTSAM_ASSETS, MODEL_ASSETS, model_dir, validate_checkpoint

DOWNLOAD_ASSETS = MODEL_ASSETS | FASTSAM_ASSETS


def download(filename: str, directory: Path) -> None:
    destination = directory / filename
    try:
        validate_checkpoint(destination)
        print(f"Verified {filename}", flush=True)
        return
    except (FileNotFoundError, ValueError):
        pass

    expected_size = DOWNLOAD_ASSETS[filename][0]
    partial = destination.with_suffix(".pt.part")
    url = f"https://github.com/ultralytics/assets/releases/download/{ASSET_RELEASE}/{filename}"
    for attempt in range(1, 5):
        try:
            offset = partial.stat().st_size if partial.is_file() else 0
            if offset == expected_size:
                try:
                    validate_checkpoint(partial, filename)
                    partial.replace(destination)
                    print(f"Verified {filename}", flush=True)
                    return
                except ValueError:
                    partial.unlink()
                    offset = 0
            elif offset > expected_size:
                partial.unlink()
                offset = 0
            headers = {"User-Agent": "visual-object-counter/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            print(f"Downloading {filename} ({expected_size / 1e6:.1f} MB, resume {offset / 1e6:.1f} MB)", flush=True)
            with urllib.request.urlopen(request, timeout=60) as response:
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
            validate_checkpoint(partial, filename)
            partial.replace(destination)
            print(f"Verified {filename}", flush=True)
            return
        except (OSError, ValueError) as error:
            if partial.is_file() and partial.stat().st_size >= expected_size:
                partial.unlink()
            if attempt == 4:
                raise RuntimeError(f"Could not download {filename}: {error}") from error
            print(f"Retry {attempt}/3: {error}", flush=True)
            time.sleep(attempt * 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=model_dir())
    parser.add_argument("--models", nargs="+", help="Optional checkpoint stems or filenames; default: all five YOLOE-26 models and FastSAM-s")
    parser.add_argument("--verify-only", action="store_true", help="Check all files without network access")
    args = parser.parse_args()
    filenames = [name if name.endswith(".pt") else f"{name}.pt" for name in args.models] if args.models else tuple(DOWNLOAD_ASSETS)
    unknown = set(filenames) - set(DOWNLOAD_ASSETS)
    if unknown:
        parser.error(f"Unknown models: {', '.join(sorted(unknown))}")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        if args.verify_only:
            validate_checkpoint(args.model_dir / filename)
            print(f"Verified {filename}", flush=True)
        else:
            download(filename, args.model_dir)
    print(f"All {len(filenames)} requested checkpoints verified in {args.model_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
