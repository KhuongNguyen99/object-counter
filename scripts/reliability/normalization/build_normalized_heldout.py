"""Re-encode frozen heldout benchmark images through app.main._normalize_image."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

from PIL import Image, __version__ as pillow_version


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
DEFAULT_DATA = ROOT / "test-results/reliability-benchmark/data"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalized_item(item: dict, manifest_path: Path, role: str, category: str,
                    normalize, output_dir: Path, mapping: dict) -> dict:
    original_path = (manifest_path.parent / item["path"]).resolve()
    original = original_path.read_bytes()
    original_hash = sha256(original)
    expected = item["source_sha256"] if role == "target" else item["sha256"]
    if original_hash != expected:
        raise ValueError(f"Source image hash changed: {original_path}")
    with Image.open(io.BytesIO(original)) as image:
        orientation = image.getexif().get(274, 1)
        original_size = image.size
    if orientation not in (None, 1):
        raise ValueError(f"EXIF orientation would move ground-truth coordinates: {original_path}")
    if original_size != (item["width"], item["height"]):
        raise ValueError(f"Source dimensions changed: {original_path}")

    if original_hash not in mapping:
        destination = output_dir / "images" / f"{original_hash}.jpg"
        temporary = destination.with_suffix(".jpg.tmp")
        try:
            width, height, normalized_hash = normalize(original, temporary)
            if (width, height) != original_size:
                raise ValueError(f"Server normalization changed dimensions: {original_path}")
            if sha256(temporary.read_bytes()) != normalized_hash:
                raise ValueError(f"Server returned an incorrect normalized hash: {original_path}")
            with Image.open(temporary) as image:
                if image.format != "JPEG" or image.size != original_size:
                    raise ValueError(f"Normalized JPEG is invalid: {original_path}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        mapping[original_hash] = {"original_sha256": original_hash,
                                  "normalized_sha256": normalized_hash,
                                  "normalized_path": f"images/{original_hash}.jpg",
                                  "width": width, "height": height, "origins": []}
    entry = mapping[original_hash]
    if (entry["width"], entry["height"]) != original_size:
        raise ValueError(f"Identical source bytes have inconsistent dimensions: {original_path}")
    entry["origins"].append({"path": str(original_path), "manifest": str(manifest_path),
                             "role": role, "category": category, "id": item.get("id")})
    converted = deepcopy(item)
    converted["path"] = entry["normalized_path"]
    converted["width"], converted["height"] = entry["width"], entry["height"]
    converted["source_sha256"] = entry["normalized_sha256"]
    if "sha256" in converted:
        converted["sha256"] = entry["normalized_sha256"]
    converted["normalization_origin"] = {"path": str(original_path), "sha256": original_hash}
    for name in ("points", "boxes"):
        if name in item and converted[name] != item[name]:
            raise AssertionError(f"Ground-truth {name} changed: {original_path}")
    return converted


def build(source_path: Path, output_dir: Path, normalize, mapping: dict,
          app_main_path: Path, app_main_hash: str, expected_queries: int) -> dict:
    source_path = source_path.resolve()
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    selected = [target for target in source["targets"] if target["split"] == "heldout"]
    if len(selected) != expected_queries:
        raise ValueError(f"Expected {expected_queries} heldout queries in {source_path}; got {len(selected)}")
    categories = {target["category"] for target in selected}
    result = deepcopy(source)
    result["targets"] = [normalized_item(target, source_path, "target", target["category"],
                                         normalize, output_dir, mapping) for target in selected]
    result["references"] = {
        category: [normalized_item(reference, source_path, "reference", category,
                                   normalize, output_dir, mapping) for reference in items]
        for category, items in source["references"].items() if category in categories
    }
    result.pop("reference_variants", None)
    result["normalization"] = {
        "method": "app.main._normalize_image", "app_main_path": str(app_main_path),
        "app_main_sha256": app_main_hash, "pillow_version": pillow_version,
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256(source_bytes), "mapping": "mapping.json",
        "server_output": "RGB JPEG, quality=95, subsampling=0",
        "coordinate_policy": "Original orientation was 1 or absent and dimensions stayed identical; points and boxes were copied unchanged.",
        "browser_reference_canvas": "The browser's reference-canvas JPEG quality 0.94 step is not reproduced."
    }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-manifest", type=Path, default=DEFAULT_DATA / "manifest.json")
    parser.add_argument("--mixed-manifest", type=Path, default=DEFAULT_DATA / "mixed-hardware-manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    # Importing app.main constructs the web app, so isolate its SQLite DATA_DIR.
    previous_data_dir = os.environ.get("DATA_DIR")
    with tempfile.TemporaryDirectory(prefix="normalization-app-", dir=output_dir) as isolated_data:
        os.environ["DATA_DIR"] = isolated_data
        try:
            from app.main import _normalize_image
            import app.main as app_main
            app_main_path = Path(app_main.__file__).resolve()
            app_main_hash = sha256(app_main_path.read_bytes())
            mapping = {}
            initial = build(args.initial_manifest, output_dir, _normalize_image, mapping,
                            app_main_path, app_main_hash, 8)
            mixed = build(args.mixed_manifest, output_dir, _normalize_image, mapping,
                          app_main_path, app_main_hash, 6)
        finally:
            if previous_data_dir is None:
                os.environ.pop("DATA_DIR", None)
            else:
                os.environ["DATA_DIR"] = previous_data_dir

    target_hashes = {item["normalization_origin"]["sha256"]
                     for item in initial["targets"] + mixed["targets"]}
    if len(target_hashes) != 11:
        raise ValueError(f"Expected 11 distinct heldout photos; got {len(target_hashes)}")
    save_json(output_dir / "initial-heldout-normalized.json", initial)
    save_json(output_dir / "mixed-heldout-normalized.json", mixed)
    save_json(output_dir / "mapping.json", {
        "app_main_path": str(app_main_path), "app_main_sha256": app_main_hash,
        "pillow_version": pillow_version,
        "entries": list(mapping.values()), "target_query_count": 14,
        "distinct_target_photo_count": 11,
        "note": "Server JPEG normalization only; browser reference-canvas JPEG quality 0.94 not reproduced."
    })
    from scripts.benchmark_reliability import prepare_manifest
    prepare_manifest(output_dir / "initial-heldout-normalized.json")
    prepare_manifest(output_dir / "mixed-heldout-normalized.json")
    print(f"Saved 14 heldout queries over 11 photos, {len(mapping)} normalized target/reference images: {output_dir}")
    print(f"Server source: {app_main_path} sha256={app_main_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
