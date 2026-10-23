"""RunPod Serverless entry point for one CountGD++ image per job."""

import base64
import binascii
import io
import math
import os
from pathlib import Path
import tempfile
import threading
import time

from PIL import Image, ImageOps, UnidentifiedImageError

from app.models import (COUNTGD_ASSETS, COUNTGD_MODEL_ID, FASTSAM_ASSETS,
                        FASTSAM_FILENAME, FASTSAM_MODEL_ID)


MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 30_000_000
_engine = None
_handler_lock = threading.Lock()
_first_handler = True
_startup_metrics = {}


def _image_from_input(encoded: object, directory: Path) -> tuple[Path, int, int]:
    if not isinstance(encoded, str) or not encoded or len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError("image_base64 must contain a JPEG, PNG, or WebP image of at most 20 MiB")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("image_base64 must be valid base64 without a URL or data prefix") from exc
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image must be 20 MiB or smaller")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Images must be JPEG, PNG, or WebP")
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("Images must be 30 megapixels or smaller")
            image = ImageOps.exif_transpose(source)
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Images must be 30 megapixels or smaller")
            image.load()
            suffix = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}[source.format]
            destination = directory / f"image.{suffix}"
            destination.write_bytes(data)
            return destination, *image.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Unreadable image; use a valid JPEG, PNG, or WebP") from exc


def _count_boxes(boxes: object, width: int, height: int) -> list[list[int | float]]:
    if not isinstance(boxes, list) or not 1 <= len(boxes) <= 32:
        raise ValueError("Each image needs 1–32 distinct boxes")
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4 or
                any(isinstance(n, bool) or not isinstance(n, (int, float)) or
                    (isinstance(n, float) and not math.isfinite(n))
                    for n in box)):
            raise ValueError("Each box needs four finite pixel coordinates")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Reference boxes must fit inside their image")
    if len({tuple(box) for box in boxes}) != len(boxes):
        raise ValueError("Each image needs distinct boxes")
    return boxes


def _job_input(event: object) -> dict:
    if not isinstance(event, dict) or not isinstance(event.get("input"), dict):
        raise ValueError("RunPod event needs an input object")
    job = event["input"]
    if set(job) - {"action", "image_base64", "boxes", "confidence", "label"}:
        raise ValueError("Unsupported input field")
    if not isinstance(job.get("action"), str) or job["action"] not in {"count", "suggest"}:
        raise ValueError("action must be count or suggest")
    label = job.get("label", "")
    if not isinstance(label, str) or len(label.strip()) > 80 or not all(c.isprintable() for c in label):
        raise ValueError("label must be at most 80 printable characters")
    confidence = job.get("confidence", .80)
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or
            (isinstance(confidence, float) and not math.isfinite(confidence)) or
            not 0 <= confidence <= 1):
        raise ValueError("confidence must be between 0 and 1")
    if job["action"] == "suggest" and "boxes" in job:
        raise ValueError("suggest does not accept boxes")
    return job


def _get_engine():
    global _engine
    if _engine is None:
        from app.inference import InferenceEngine
        _engine = InferenceEngine()
    return _engine


def _start_gpu_metrics():
    import torch

    if not torch.cuda.is_available():
        return None, None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda, torch.cuda.get_device_name()


def _clip_detections(result: dict, width: int, height: int) -> None:
    def clip_pair(start: float, end: float, limit: int) -> tuple[float, float]:
        start = max(0.0, min(start, limit - 1.0))
        end = max(0.0, min(end, float(limit)))
        return start, end if end > start else min(float(limit), start + 1.0)

    for detection in result["detections"]:
        x1, y1, x2, y2 = detection["box"]
        left, right = clip_pair(x1, x2, width)
        top, bottom = clip_pair(y1, y2, height)
        detection["box"] = [left, top, right, bottom]


def handler(event: dict) -> dict:
    """Validate a single image, invoke the cached engine, and return JSON-safe results."""
    global _first_handler
    started = time.perf_counter()
    with _handler_lock:
        job = _job_input(event)
        with tempfile.TemporaryDirectory(prefix="countwell-runpod-") as directory:
            image_path, width, height = _image_from_input(job.get("image_base64"), Path(directory))
            action = job["action"]
            boxes = _count_boxes(job.get("boxes"), width, height) if action == "count" else None
            cold_start = _first_handler
            _first_handler = False
            cuda, gpu_name = _start_gpu_metrics()
            engine = _get_engine()
            if action == "count":
                results = engine.count(
                    COUNTGD_MODEL_ID,
                    references=[{"path": str(image_path), "boxes": boxes}],
                    image_paths=[str(image_path)],
                    output_dir=directory,
                    confidence=job.get("confidence", .80),
                    label=job.get("label", "").strip(),
                    reference_mode="per_image",
                )
                if len(results) != 1:
                    raise RuntimeError("CountGD++ returned the wrong number of image results")
                result = {key: value for key, value in results[0].items() if key != "annotated_path"}
                _clip_detections(result, width, height)
            else:
                result = engine.suggest_reference_boxes(str(image_path), job.get("label", "").strip())
            if cuda is not None:
                cuda.synchronize()
            metrics = {
                "gpu_name": gpu_name,
                "worker_id": os.environ.get("RUNPOD_POD_ID") or os.environ.get("RUNPOD_WORKER_ID"),
                "cold_start": cold_start,
                "model": COUNTGD_MODEL_ID if action == "count" else FASTSAM_MODEL_ID,
                "model_sha256": (COUNTGD_ASSETS["countgd_plusplus.pth"][1] if action == "count"
                                 else FASTSAM_ASSETS[FASTSAM_FILENAME][1]),
                "build_revision": os.environ.get("WORKER_BUILD_REVISION", "unknown"),
                "handler_ms": round((time.perf_counter() - started) * 1000, 2),
                "peak_cuda_allocated_bytes": cuda.max_memory_allocated() if cuda is not None else None,
                **_startup_metrics,
            }
            return {"result": result, "metrics": metrics}


def main() -> None:
    """Preload one requested model before RunPod begins accepting jobs."""
    started = time.perf_counter()
    preload = os.environ.get("RUNPOD_PRELOAD", "count").lower()
    if preload not in {"count", "suggest", "none"}:
        raise ValueError("RUNPOD_PRELOAD must be count, suggest, or none")
    if preload != "none":
        preload_started = time.perf_counter()
        engine = _get_engine()
        cuda, _ = _start_gpu_metrics()
        if preload == "count":
            model_started = time.perf_counter()
            with engine._lock:
                engine._load_countgd_model(engine._device())
            if cuda is not None:
                cuda.synchronize()
            _startup_metrics["startup_model_load_ms"] = round((time.perf_counter() - model_started) * 1000, 2)
        else:
            with tempfile.TemporaryDirectory(prefix="countwell-preload-") as directory:
                image_path = Path(directory) / "image.jpg"
                Image.new("RGB", (64, 64), "white").save(image_path)
                engine.suggest_reference_boxes(str(image_path))
        if cuda is not None:
            cuda.synchronize()
        _startup_metrics["preload_ms"] = round((time.perf_counter() - preload_started) * 1000, 2)
        _startup_metrics["preloaded_model"] = preload
    _startup_metrics["worker_ready_ms"] = round((time.perf_counter() - started) * 1000, 2)
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
