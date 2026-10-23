"""Local web API for reference-guided object counting."""

import asyncio
import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.storage import Store
from app.models import COUNTGD_MODEL_ID, SAM3COUNT_MODEL_ID, SAM3_MODEL_ID
from app.quick_count import count_marked_areas, parse_search_areas


logger = logging.getLogger(__name__)
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_BODY_BYTES = 450 * 1024 * 1024
MAX_PIXELS = 30_000_000
MAX_REFERENCES = 8
MAX_TARGETS = 20
MAX_BOXES = 32
MAX_QUEUE = 8
DEFAULT_MODEL = COUNTGD_MODEL_ID
LEGACY_REFERENCE_MODES = {SAM3_MODEL_ID: "per_image", SAM3COUNT_MODEL_ID: "none"}


class _CORSFastAPI(FastAPI):
    def __init__(self, *args, allowed_origins, **kwargs):
        self.allowed_origins = allowed_origins
        super().__init__(*args, **kwargs)

    def build_middleware_stack(self):
        # Wrap FastAPI's error handler so even unexpected 500 responses carry CORS.
        return CORSMiddleware(super().build_middleware_stack(),
                              allow_origins=self.allowed_origins,
                              allow_methods=["GET", "POST"],
                              allow_headers=["Content-Type"],
                              expose_headers=["Content-Disposition"],
                              allow_credentials=False)


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = 0
        if length > MAX_BODY_BYTES:
            return await self._reject(send)
        received = 0
        started = False

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_BODY_BYTES:
                    raise ValueError("Request body exceeds 450 MB")
            return message

        async def tracked_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, tracked_send)
        except ValueError as exc:
            if str(exc) != "Request body exceeds 450 MB" or started:
                raise
            await self._reject(send)

    @staticmethod
    async def _reject(send):
        body = b'{"detail":"Request body exceeds 450 MB"}'
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def _id():
    return str(uuid.uuid4())


def _safe_filename(value):
    name = (value or "image").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(char if char.isprintable() else "_" for char in name).strip()
    return name[:180] or "image"


def _validate_label(value, *, required=True):
    label = value.strip()
    if required and not label:
        raise HTTPException(422, "Describe the object before continuing")
    if len(label) > 80 or not all(char.isprintable() for char in label):
        raise HTTPException(422, "Object description must be at most 80 printable characters")
    return label


async def _save_image(upload: UploadFile, destination: Path):
    data = await upload.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, "Each image must be 20 MB or smaller")
    return await run_in_threadpool(_normalize_image, data, destination)


def _normalize_image(data: bytes, destination: Path):
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(422, "Images must be JPEG, PNG, or WebP")
            if source.width * source.height > MAX_PIXELS:
                raise HTTPException(422, "Images must be 30 megapixels or smaller")
            image = ImageOps.exif_transpose(source).convert("RGB")
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(422, "Images must be 30 megapixels or smaller")
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=95, subsampling=0)
            normalized = output.getvalue()
            destination.write_bytes(normalized)
            return image.width, image.height, hashlib.sha256(normalized).hexdigest()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Unreadable image; use a valid JPEG, PNG, or WebP") from exc


def _parse_boxes(raw, dimensions, *, per_image=False):
    field = "image_boxes" if per_image else "reference_boxes"
    try:
        groups = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, f"{field} must be valid JSON") from exc
    if not isinstance(groups, list) or len(groups) != len(dimensions):
        kind = "target" if per_image else "reference"
        raise HTTPException(422, f"Provide one box group for each {kind} image")
    if not per_image and sum(len(group) for group in groups if isinstance(group, list)) > MAX_BOXES:
        raise HTTPException(422, "Use at most 32 reference boxes")
    for group, (width, height) in zip(groups, dimensions):
        if not isinstance(group, list) or not group or (per_image and len(group) > MAX_BOXES):
            if per_image:
                raise HTTPException(422, "Each target image needs 1–32 boxes")
            raise HTTPException(422, "Each reference image needs at least one box")
        for box in group:
            if (not isinstance(box, list) or len(box) != 4 or
                    any(isinstance(n, bool) or not isinstance(n, (int, float)) or
                        (isinstance(n, float) and not math.isfinite(n))
                        for n in box)):
                raise HTTPException(422, "Each box needs four finite pixel coordinates")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                kind = "Target" if per_image else "Reference"
                raise HTTPException(422, f"{kind} boxes must fit inside their image")
    return groups


@lru_cache(maxsize=8)
def _device(setting):
    if setting == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    if setting.isdigit() or setting.startswith("cuda"):
        return "cuda"
    return setting


def _model_catalog(model_lister, detect_device=False):
    catalog = model_lister()
    if isinstance(catalog, dict):
        models = catalog.get("models", [])
        default = catalog.get("default_model", DEFAULT_MODEL)
        device = catalog.get("device") or (_device(os.environ.get("DEVICE", "auto")) if detect_device else "cpu")
    else:
        models, default = catalog, (catalog[0]["id"] if catalog else DEFAULT_MODEL)
        device = _device(os.environ.get("DEVICE", "auto")) if detect_device else "cpu"
    return {"models": models, "default_model": default, "device": device}


def _csv_safe(value):
    text = str(value)
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def _validate_processing(value):
    limits = {"tiles": (1, 4096), "tile_size": (1, 16384),
              "duplicates_removed": (0, 1_000_000)}
    if (not isinstance(value, dict) or set(value) != set(limits) or
            any(type(value[key]) is not int or not lower <= value[key] <= upper
                for key, (lower, upper) in limits.items())):
        raise ValueError("The model returned invalid image processing metadata")
    return {key: value[key] for key in limits}


def _validate_reference_proposal(value, width, height):
    if not isinstance(value, dict):
        raise ValueError("The detector returned invalid reference boxes")
    boxes = value.get("boxes")
    elapsed = value.get("elapsed_ms")
    warning = value.get("warning")
    if (type(value.get("width")) is not int or value["width"] != width or
            type(value.get("height")) is not int or value["height"] != height or
            value.get("model") != "fastsam-s" or
            isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or
            not math.isfinite(elapsed) or elapsed < 0 or
            (warning is not None and not isinstance(warning, str)) or
            not isinstance(boxes, list)):
        raise ValueError("The detector returned invalid reference boxes")
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4 or
                any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n)
                    for n in box)):
            raise ValueError("The detector returned invalid reference boxes")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("The detector returned invalid reference boxes")
    return {"width": width, "height": height, "boxes": boxes,
            "model": "fastsam-s", "elapsed_ms": elapsed,
            "warning": warning[:500] if warning else None}


_ORIGIN_PATTERN = re.compile(
    r"https?://(?:[A-Za-z0-9.-]+|\[[0-9A-Fa-f:.]+\])(?::[0-9]{1,5})?/?\Z",
    re.IGNORECASE,
)


def _parse_frontend_origins(value):
    origins = []
    entries = value.split(",") if isinstance(value, str) else value
    for entry in entries:
        if not isinstance(entry, str):
            raise ValueError("FRONTEND_ORIGINS must contain only HTTP(S) origins")
        origin = entry.strip()
        if not origin:
            continue
        if not _ORIGIN_PATTERN.fullmatch(origin):
            raise ValueError(f"FRONTEND_ORIGINS must contain only HTTP(S) origins: {origin!r}")
        try:
            port = urlsplit(origin).port
        except ValueError as exc:
            raise ValueError(f"FRONTEND_ORIGINS contains an invalid port: {origin!r}") from exc
        if port == 0:
            raise ValueError(f"FRONTEND_ORIGINS contains an invalid port: {origin!r}")
        origin = origin.rstrip("/").lower()
        if origin not in origins:
            origins.append(origin)
    return origins


def create_app(data_dir=None, engine_factory=None, model_lister=None, frontend_origins=None):
    backend = os.environ.get("INFERENCE_BACKEND", "local").strip().lower()
    if backend not in {"local", "runpod"}:
        raise ValueError("INFERENCE_BACKEND must be local or runpod")
    allowed_origins = _parse_frontend_origins(
        os.environ.get("FRONTEND_ORIGINS", "") if frontend_origins is None else frontend_origins)
    data_dir = Path(data_dir or os.environ.get("DATA_DIR", "data")).resolve()
    store = Store(data_dir)
    slots = threading.BoundedSemaphore(MAX_QUEUE)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="counting")
    engine = None
    waiters = {}
    waiters_lock = threading.Lock()

    def wake_waiter(future):
        if not future.done():
            future.set_result(None)

    def notify_session(session_id):
        with waiters_lock:
            pending = list(waiters.get(session_id, ()))
        for future in pending:
            try:
                future.get_loop().call_soon_threadsafe(wake_waiter, future)
            except RuntimeError:
                pass  # A disconnected client's event loop may already be closed.

    detect_device = model_lister is None
    remote = backend == "runpod"
    if remote:
        from app.compute_status import ComputeStatus
        compute_status = ComputeStatus(data_dir, os.environ.get("RUNPOD_ENDPOINT_ID", ""))
    else:
        compute_status = None
    if model_lister is None:
        if remote:
            from app.runpod_client import remote_model_catalog
            model_lister = remote_model_catalog
        else:
            def model_lister():
                from app.models import list_models
                catalog = [entry for entry in list_models() if entry["id"] == COUNTGD_MODEL_ID]
                return {"models": catalog, "default_model": COUNTGD_MODEL_ID}

    if engine_factory is None:
        if remote:
            from app.runpod_client import RunPodInferenceEngine
            def engine_factory():
                return RunPodInferenceEngine(activity=compute_status)
        else:
            def engine_factory():
                from app.inference import InferenceEngine
                return InferenceEngine()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        executor.shutdown(wait=True)
        if engine is not None and hasattr(engine, "close"):
            engine.close()
        if compute_status is not None:
            compute_status.close()

    app = _CORSFastAPI(title="Object Counting", lifespan=lifespan,
                       allowed_origins=allowed_origins)
    app.add_middleware(BodyLimitMiddleware)
    app.state.store = store

    def detail_or_404(session_id):
        result = store.detail(session_id)
        if result is None:
            raise HTTPException(404, "Session not found")
        if result["model"] == COUNTGD_MODEL_ID:
            result["reference_mode"] = "shared" if result["references"] else "per_image"
            result["required_reference_boxes"] = None
            if not result["references"]:
                result["recommended_reference_boxes"] = 10
        else:
            entry = next((entry for entry in _model_catalog(model_lister, detect_device)["models"]
                          if entry["id"] == result["model"]), {})
            result["reference_mode"] = entry.get(
                "reference_mode", LEGACY_REFERENCE_MODES.get(result["model"], "shared"))
            result["required_reference_boxes"] = entry.get("required_reference_boxes")
        result["revision"] = hashlib.sha256(json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return result

    def run_batch(session_id, batch_id, items, per_image, count_mode):
        nonlocal engine
        session_dir = data_dir / session_id
        files_dir = session_dir / "files"
        output_dir = session_dir / "work" / batch_id
        created = []
        try:
            store.set_batch_status(batch_id, "running")
            notify_session(session_id)
            output_dir.mkdir(parents=True, exist_ok=True)
            if engine is None:
                engine = engine_factory()
            session = store.session_row(session_id)
            if count_mode == "quick":
                output = [count_marked_areas(
                    engine, str(files_dir / item["original_filename"]), item["search_areas"],
                    output_dir, session["label"], _validate_reference_proposal,
                    annotate=False) for item in items]
            else:
                if per_image:
                    references = [{"path": str(files_dir / item["original_filename"]),
                                   "boxes": item["reference_boxes"]} for item in items]
                else:
                    refs = detail_or_404(session_id)["references"]
                    references = [{"path": str(files_dir / ref["filename"]), "boxes": ref["boxes"]}
                                  for ref in refs]
                options = {"reference_mode": "per_image"} if per_image and session["model"] == COUNTGD_MODEL_ID else {}
                if options:
                    options["annotate"] = False
                output = engine.count(
                    model_id=session["model"], references=references,
                    image_paths=[str(files_dir / item["original_filename"]) for item in items],
                    output_dir=str(output_dir), confidence=session["confidence"], label=session["label"],
                    **options,
                )
            if not isinstance(output, list) or len(output) != len(items):
                raise ValueError("The model returned the wrong number of image results")
            results = []
            for item, result in zip(items, output):
                count = result["count"]
                detections = result["detections"]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError("The model returned an invalid count")
                if not isinstance(detections, list) or len(detections) != count:
                    raise ValueError("The model returned inconsistent detections")
                processing = (_validate_processing(result["processing"])
                              if "processing" in result else None)
                width, height = result["width"], result["height"]
                elapsed_ms = result["elapsed_ms"]
                if (type(width) is not int or type(height) is not int or
                        (width, height) != (item["width"], item["height"]) or
                        isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, (int, float)) or
                        not math.isfinite(elapsed_ms) or elapsed_ms < 0):
                    raise ValueError("The model returned invalid image dimensions or timing")
                if count > 30000:
                    raise ValueError("The model returned too many detections")
                for index, detection in enumerate(detections, 1):
                    if (not isinstance(detection, dict) or type(detection.get("index")) is not int or
                            detection["index"] != index or not isinstance(detection.get("box"), list) or
                            len(detection["box"]) != 4 or any(
                                isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n)
                                for n in detection["box"])):
                        raise ValueError("The model returned invalid detections")
                    x1, y1, x2, y2 = detection["box"]
                    confidence = detection.get("confidence")
                    valid_confidence = (confidence is None and count_mode == "quick") or (
                        not isinstance(confidence, bool) and isinstance(confidence, (int, float)) and
                        math.isfinite(confidence) and 0 <= confidence <= 1)
                    if (not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height) or
                            not valid_confidence):
                        raise ValueError("The model returned invalid detections")
                annotated_filename = None
                if count_mode != "quick" and not (per_image and session["model"] == COUNTGD_MODEL_ID):
                    annotation = Path(result["annotated_path"]).resolve()
                    if not annotation.is_file() or not annotation.is_relative_to(output_dir.resolve()):
                        raise ValueError("The model did not save an annotated image")
                    annotated_filename = f"annotated-{item['id']}.jpg"
                    with Image.open(annotation) as annotated:
                        if annotated.width * annotated.height > MAX_PIXELS:
                            raise ValueError("The model returned an oversized annotated image")
                        ImageOps.exif_transpose(annotated).convert("RGB").save(
                            files_dir / annotated_filename, format="JPEG", quality=95)
                    created.append(files_dir / annotated_filename)
                results.append({
                    "id": item["id"], "count": count, "detections": detections,
                    "annotated_filename": annotated_filename, "width": width,
                    "height": height, "elapsed_ms": elapsed_ms,
                    "warning": str(result["warning"])[:500] if result.get("warning") else None,
                    "processing": processing,
                    "verification": result.get("verification"),
                })
            store.complete_batch(batch_id, results)
            notify_session(session_id)
        except Exception as exc:
            logger.exception("Batch %s failed", batch_id)
            store.set_batch_status(batch_id, "failed", f"Counting failed: {exc}"[:300])
            notify_session(session_id)
            for path in created:
                path.unlink(missing_ok=True)
            for item in items:
                (files_dir / item["original_filename"]).unlink(missing_ok=True)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            slots.release()
            if compute_status is not None:
                compute_status.finish()

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/compute-status")
    def compute(response: Response):
        response.headers["Cache-Control"] = "no-store"
        return compute_status.snapshot() if compute_status is not None else {"enabled": False}

    @app.get("/reports/reliability.html", include_in_schema=False)
    def reliability_report():
        report = data_dir / "reports" / "reliability.html"
        if not report.is_file():
            raise HTTPException(404, "No benchmark report has been generated")
        return FileResponse(report, media_type="text/html")

    @app.get("/api/models")
    def models():
        return _model_catalog(model_lister, detect_device)

    @app.post("/api/reference-boxes")
    async def reference_boxes(image: UploadFile = File(...), label: str = Form("")):
        from app.reference import ReferenceModelUnavailable

        label = _validate_label(label, required=False)
        data = await image.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(413, "Each image must be 20 MB or smaller")
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "The inference queue is full; try again shortly")

        def run_reference():
            nonlocal engine
            try:
                with tempfile.TemporaryDirectory(prefix="reference-boxes-") as directory:
                    image_path = Path(directory) / "image.jpg"
                    width, height, _ = _normalize_image(data, image_path)
                    if engine is None:
                        engine = engine_factory()
                    return _validate_reference_proposal(engine.suggest_reference_boxes(str(image_path), label), width, height)
            finally:
                slots.release()
                if compute_status is not None:
                    compute_status.finish()

        if compute_status is not None:
            compute_status.start()
        try:
            future = executor.submit(run_reference)
        except Exception:
            slots.release()
            if compute_status is not None:
                compute_status.finish()
            logger.exception("Could not schedule reference box detection")
            raise HTTPException(500, "Could not schedule automatic box suggestions")
        try:
            return await asyncio.shield(asyncio.wrap_future(future))
        except HTTPException:
            raise
        except ReferenceModelUnavailable as exc:
            logger.warning("Reference box model unavailable: %s", exc)
            raise HTTPException(503, "Automatic box suggestions are unavailable") from exc
        except Exception as exc:
            logger.exception("Reference box detection failed")
            raise HTTPException(500, "Automatic box suggestions failed") from exc

    @app.post("/api/sessions", status_code=201)
    async def create_session(
        label: str = Form(""), references: list[UploadFile] = File(default=[]),
        model: str = Form(...), confidence: float = Form(.80),
        reference_boxes: str = Form("[]"),
    ):
        if not math.isfinite(confidence) or not .01 <= confidence <= .95:
            raise HTTPException(422, "Confidence must be between 0.01 and 0.95")
        available = {entry["id"]: entry for entry in _model_catalog(model_lister, detect_device)["models"]
                     if entry.get("available")}
        if model not in available:
            raise HTTPException(422, "Counting is unavailable right now. Please try again.")
        reference_mode = "per_image" if model == COUNTGD_MODEL_ID else available[model].get("reference_mode")
        label = _validate_label(label, required=reference_mode == "none")
        if reference_mode in {"none", "per_image"}:
            if references:
                detail = ("This model uses the Object description; do not upload reference images"
                          if reference_mode == "none" else "This model does not use separate reference images")
                raise HTTPException(422, detail)
        elif not 1 <= len(references) <= MAX_REFERENCES:
            raise HTTPException(422, "Upload 1–8 reference images")
        session_id = _id()
        files_dir = data_dir / session_id / "files"
        files_dir.mkdir(parents=True)
        refs = []
        try:
            for upload in references:
                ref_id = _id()
                filename = f"reference-{ref_id}.jpg"
                width, height, _ = await _save_image(upload, files_dir / filename)
                refs.append({"id": ref_id, "filename": filename, "width": width, "height": height})
            groups = _parse_boxes(reference_boxes, [(r["width"], r["height"]) for r in refs])
            for ref, boxes in zip(refs, groups):
                ref["boxes"] = boxes
            store.create_session(session_id, label, model, confidence, refs)
        except Exception:
            shutil.rmtree(data_dir / session_id, ignore_errors=True)
            raise
        return detail_or_404(session_id)

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str, after: str | None = Query(None, max_length=64),
                          wait: float = Query(0, ge=0, le=30)):
        if not after or not wait:
            return await run_in_threadpool(detail_or_404, session_id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait
        while True:
            future = loop.create_future()
            # Register before reading so a completion between read and wait is not lost.
            with waiters_lock:
                waiters.setdefault(session_id, set()).add(future)
            try:
                result = await run_in_threadpool(detail_or_404, session_id)
                if (result["revision"] != after or not any(
                        batch["status"] in {"queued", "running"} for batch in result["batches"])):
                    return result
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return result
                try:
                    await asyncio.wait_for(future, remaining)
                except TimeoutError:
                    return await run_in_threadpool(detail_or_404, session_id)
            finally:
                future.cancel()
                with waiters_lock:
                    pending = waiters.get(session_id)
                    if pending is not None:
                        pending.discard(future)
                        if not pending:
                            del waiters[session_id]

    @app.post("/api/sessions/{session_id}/batches", status_code=202)
    async def create_batch(session_id: str, images: list[UploadFile] = File(...),
                           image_boxes: str = Form("[]"), count_mode: str = Form("standard"),
                           search_areas: str = Form("[]")):
        session = store.session_row(session_id)
        if session is None:
            raise HTTPException(404, "Session not found")
        if not 1 <= len(images) <= MAX_TARGETS:
            raise HTTPException(422, "Upload 1–20 target images")
        saved = detail_or_404(session_id)
        available_models = {entry["id"] for entry in _model_catalog(model_lister, detect_device)["models"]
                            if entry.get("available")}
        if detect_device and (session["model"] not in available_models or
                              saved["reference_mode"] != "per_image"):
            raise HTTPException(422, "This session can no longer accept new counts")
        per_image = saved["reference_mode"] == "per_image"
        if count_mode not in {"standard", "quick"}:
            raise HTTPException(422, "Choose standard or Quick count")
        if count_mode == "quick" and session["model"] != COUNTGD_MODEL_ID:
            raise HTTPException(422, "Quick count is available for CountGD++ sessions")
        if count_mode == "quick" or not per_image:
            try:
                unused_boxes = json.loads(image_boxes)
            except (TypeError, ValueError) as exc:
                raise HTTPException(422, "image_boxes must be valid JSON") from exc
            if unused_boxes != []:
                raise HTTPException(422, "Quick count uses marked areas, not reference boxes" if count_mode == "quick"
                                    else "This model does not use per-image boxes")
        if count_mode == "standard":
            try:
                unused_areas = json.loads(search_areas)
            except (TypeError, ValueError) as exc:
                raise HTTPException(422, "search_areas must be valid JSON") from exc
            if unused_areas != []:
                raise HTTPException(422, "Standard count does not use marked areas")
        batch_id = _id()
        files_dir = data_dir / session_id / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        items = []
        created_files = []
        try:
            hashes = set()
            for upload in images:
                image_id = _id()
                filename = f"original-{image_id}.jpg"
                width, height, sha256 = await _save_image(upload, files_dir / filename)
                created_files.append(files_dir / filename)
                if sha256 in hashes:
                    raise HTTPException(409, "This request contains duplicate images")
                hashes.add(sha256)
                items.append({
                    "id": image_id, "filename": _safe_filename(upload.filename),
                    "original_filename": filename, "sha256": sha256,
                    "width": width, "height": height,
                })
            if count_mode == "quick":
                try:
                    groups = parse_search_areas(
                        search_areas, [(item["width"], item["height"]) for item in items])
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from exc
                for item, areas in zip(items, groups):
                    item["search_areas"] = areas
            elif per_image:
                groups = _parse_boxes(image_boxes, [(item["width"], item["height"]) for item in items],
                                      per_image=True)
                if session["model"] == COUNTGD_MODEL_ID and any(
                        len({tuple(box) for box in group}) != len(group) for group in groups):
                    raise HTTPException(422, "Each CountGD++ target image needs distinct boxes")
                for item, boxes in zip(items, groups):
                    item["reference_boxes"] = boxes
            if not slots.acquire(blocking=False):
                raise HTTPException(429, "The counting queue is full; try again shortly")
            try:
                error = store.create_batch(session_id, batch_id, items, count_mode=count_mode)
                if error:
                    status = 404 if error == "missing" else 409
                    detail = {"missing": "Session not found", "finished": "Session is finished",
                              "busy": "This session already has a queued or running batch",
                              "duplicate": "Image already exists in this session"}[error]
                    raise HTTPException(status, detail)
                notify_session(session_id)
                if compute_status is not None:
                    compute_status.start()
                try:
                    executor.submit(run_batch, session_id, batch_id, items, per_image, count_mode)
                except Exception:
                    if compute_status is not None:
                        compute_status.finish()
                    store.set_batch_status(batch_id, "failed", "Could not schedule counting. Please retry.")
                    notify_session(session_id)
                    raise
            except Exception:
                slots.release()
                raise
        except Exception:
            for path in created_files:
                path.unlink(missing_ok=True)
            raise
        return {"id": batch_id, "status": "queued"}

    @app.post("/api/sessions/{session_id}/finish")
    def finish_session(session_id: str):
        error = store.finish(session_id)
        if error == "missing":
            raise HTTPException(404, "Session not found")
        if error == "busy":
            raise HTTPException(409, "Wait for the current batch to finish")
        if error == "unreviewed":
            raise HTTPException(409, "Review and confirm every image before finishing this cross-check count")
        notify_session(session_id)
        return detail_or_404(session_id)

    def save_image_review(session_id, image_id, payload):
        destination = None
        committed = False
        try:
            session = store.session_row(session_id)
            item = store.review_image_row(session_id, image_id)
            if session is None or item is None:
                raise HTTPException(404, "Image not found in this session")
            if session["status"] != "active":
                raise HTTPException(409, "Session is finished")
            if item["batch_status"] != "complete":
                raise HTTPException(409, "Wait for this image to finish counting")
            if item["review_revision"] != payload["revision"]:
                raise HTTPException(409, "This image was reviewed elsewhere. Reload the latest result before saving")
            boxes = payload["boxes"]
            detections = []
            for index, box in enumerate(boxes, 1):
                if (not isinstance(box, list) or len(box) != 4 or
                        any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n)
                            for n in box)):
                    raise HTTPException(422, "Each review box needs four finite pixel coordinates")
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= item["width"] and 0 <= y1 < y2 <= item["height"]):
                    raise HTTPException(422, "Review boxes must have positive area and fit inside the image")
                detections.append({"index": index, "box": box, "confidence": None, "source": "review"})
            files_dir = data_dir / session_id / "files"
            filename = None
            if item["annotated_filename"]:
                from .inference import _draw_detections, _read_image
                filename = f"reviewed-{image_id}-{uuid.uuid4().hex}.jpg"
                destination = files_dir / filename
                image = _read_image(str(files_dir / item["original_filename"]))
                try:
                    _draw_detections(image, detections, destination)
                finally:
                    image.close()
            error = store.save_review(session_id, image_id, payload["revision"], detections,
                                      filename, payload["reviewed"])
            if error:
                raise HTTPException(404 if error == "missing" else 409, {
                    "missing": "Image not found in this session", "finished": "Session is finished",
                    "busy": "Wait for all batches in this session to finish before reviewing",
                    "stale": "This image was reviewed elsewhere. Reload the latest result before saving",
                }[error])
            committed = True
            notify_session(session_id)
            if item["review_annotated_filename"]:
                try:
                    (files_dir / item["review_annotated_filename"]).unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove superseded review annotation for %s", image_id)
            return detail_or_404(session_id)
        finally:
            if destination is not None and not committed:
                destination.unlink(missing_ok=True)
            slots.release()

    @app.post("/api/sessions/{session_id}/images/{image_id}/review")
    def review_image(session_id: str, image_id: str, payload: dict = Body(...)):
        if (set(payload) != {"revision", "boxes", "reviewed"} or
                type(payload["revision"]) is not int or payload["revision"] < 0 or
                type(payload["reviewed"]) is not bool or not isinstance(payload["boxes"], list) or
                len(payload["boxes"]) > 30000):
            raise HTTPException(422, "Provide revision, up to 30000 boxes, and a reviewed boolean")
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "The counting queue is full; try again shortly")
        try:
            future = executor.submit(save_image_review, session_id, image_id, payload)
        except Exception:
            slots.release()
            raise
        return future.result()

    @app.get("/api/sessions/{session_id}/export.csv")
    def export_csv(session_id: str):
        session = detail_or_404(session_id)
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["session", "batch", "filename", "count", "running_total", "warning"])
        running = 0
        for batch in session["batches"]:
            for result in batch["results"]:
                running += result["count"]
                warning = result.get("warning") or (
                    "No matches detected. Check the reference or use a model trained for this object."
                    if result["count"] == 0 else ""
                )
                if result["reviewed"]:
                    warning = f"Human reviewed; model count {result['model_count']}. " + warning
                elif session["requires_review"]:
                    warning = f"Unreviewed count; model count {result['model_count']}. " + warning
                writer.writerow([_csv_safe(session["label"]), batch["id"],
                                 _csv_safe(result["filename"]), result["count"], running,
                                 _csv_safe(warning)])
        return Response(output.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="session-{session_id}.csv"'})

    @app.get("/api/sessions/{session_id}/files/{filename}")
    def get_file(session_id: str, filename: str):
        if Path(filename).name != filename or not store.owns_file(session_id, filename):
            raise HTTPException(404, "File not found")
        path = data_dir / session_id / "files" / filename
        if not path.is_file():
            raise HTTPException(404, "File not found")
        return FileResponse(path, media_type="image/jpeg")

    return app


app = create_app()
