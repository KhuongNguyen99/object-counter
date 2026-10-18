"""Local web API for reference-guided object counting."""

import asyncio
import csv
import hashlib
import io
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.storage import Store


logger = logging.getLogger(__name__)
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_BODY_BYTES = 450 * 1024 * 1024
MAX_PIXELS = 30_000_000
MAX_REFERENCES = 8
MAX_TARGETS = 20
MAX_BOXES = 32
MAX_QUEUE = 8
DEFAULT_MODEL = "yoloe-26s-seg"


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


def _validate_label(value):
    label = value.strip()
    if not label or len(label) > 80 or not all(char.isprintable() for char in label):
        raise HTTPException(422, "Label must be 1–80 printable characters")
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


def _parse_boxes(raw, dimensions):
    try:
        groups = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "reference_boxes must be valid JSON") from exc
    if not isinstance(groups, list) or len(groups) != len(dimensions):
        raise HTTPException(422, "Provide one box group for each reference image")
    if sum(len(group) for group in groups if isinstance(group, list)) > MAX_BOXES:
        raise HTTPException(422, "Use at most 32 reference boxes")
    for group, (width, height) in zip(groups, dimensions):
        if not isinstance(group, list) or not group:
            raise HTTPException(422, "Each reference image needs at least one box")
        for box in group:
            if (not isinstance(box, list) or len(box) != 4 or
                    any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n)
                        for n in box)):
                raise HTTPException(422, "Each box needs four finite pixel coordinates")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise HTTPException(422, "Reference boxes must fit inside their image")
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
        models, default = catalog, DEFAULT_MODEL
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
            value.get("model") != "countgd-plusplus" or
            isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or
            not math.isfinite(elapsed) or elapsed < 0 or
            (warning is not None and not isinstance(warning, str)) or
            not isinstance(boxes, list) or len(boxes) > MAX_BOXES):
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
            "model": "countgd-plusplus", "elapsed_ms": elapsed,
            "warning": warning[:500] if warning else None}


def create_app(data_dir=None, engine_factory=None, model_lister=None):
    data_dir = Path(data_dir or os.environ.get("DATA_DIR", "data")).resolve()
    store = Store(data_dir)
    slots = threading.BoundedSemaphore(MAX_QUEUE)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="counting")
    engine = None

    detect_device = model_lister is None
    if model_lister is None:
        def model_lister():
            from app.models import COUNTGD_MODEL_ID, list_models
            catalog = list_models()
            preferred = COUNTGD_MODEL_ID if any(entry["id"] == COUNTGD_MODEL_ID and entry["available"]
                                               for entry in catalog) else DEFAULT_MODEL
            return {"models": catalog, "default_model": preferred}

    if engine_factory is None:
        def engine_factory():
            from app.inference import InferenceEngine
            return InferenceEngine()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        executor.shutdown(wait=True)
        if engine is not None and hasattr(engine, "close"):
            engine.close()

    app = FastAPI(title="Object Counting", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware)
    app.state.store = store

    def detail_or_404(session_id):
        result = store.detail(session_id)
        if result is None:
            raise HTTPException(404, "Session not found")
        return result

    def run_batch(session_id, batch_id, items):
        nonlocal engine
        session_dir = data_dir / session_id
        files_dir = session_dir / "files"
        output_dir = session_dir / "work" / batch_id
        created = []
        try:
            store.set_batch_status(batch_id, "running")
            refs = detail_or_404(session_id)["references"]
            references = [{"path": str(files_dir / ref["filename"]), "boxes": ref["boxes"]}
                          for ref in refs]
            output_dir.mkdir(parents=True, exist_ok=True)
            if engine is None:
                engine = engine_factory()
            session = store.session_row(session_id)
            output = engine.count(
                model_id=session["model"], references=references,
                image_paths=[str(files_dir / item["original_filename"]) for item in items],
                output_dir=str(output_dir), confidence=session["confidence"], label=session["label"],
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
                width, height = int(result["width"]), int(result["height"])
                elapsed_ms = float(result["elapsed_ms"])
                if width < 1 or height < 1 or not math.isfinite(elapsed_ms) or elapsed_ms < 0:
                    raise ValueError("The model returned invalid image dimensions or timing")
                results.append({
                    "id": item["id"], "count": count, "detections": detections,
                    "annotated_filename": annotated_filename, "width": width,
                    "height": height, "elapsed_ms": elapsed_ms,
                    "warning": str(result["warning"])[:500] if result.get("warning") else None,
                    "processing": processing,
                })
            store.complete_batch(batch_id, results)
        except Exception as exc:
            logger.exception("Batch %s failed", batch_id)
            store.set_batch_status(batch_id, "failed", f"Counting failed: {exc}"[:300])
            for path in created:
                path.unlink(missing_ok=True)
            for item in items:
                (files_dir / item["original_filename"]).unlink(missing_ok=True)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)
            slots.release()

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/models")
    def models():
        return _model_catalog(model_lister, detect_device)

    @app.post("/api/reference-boxes")
    async def reference_boxes(image: UploadFile = File(...), label: str = Form(...)):
        from app.reference import ReferenceModelUnavailable

        label = _validate_label(label)
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

        try:
            future = executor.submit(run_reference)
        except Exception:
            slots.release()
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
        label: str = Form(...), references: list[UploadFile] = File(default=[]),
        model: str = Form(...), confidence: float = Form(.25),
        reference_boxes: str = Form("[]"),
    ):
        label = _validate_label(label)
        if not math.isfinite(confidence) or not .01 <= confidence <= .95:
            raise HTTPException(422, "Confidence must be between 0.01 and 0.95")
        available = {entry["id"]: entry for entry in _model_catalog(model_lister, detect_device)["models"]
                     if entry.get("available")}
        if model not in available:
            raise HTTPException(422, "Choose an available model")
        if available[model].get("reference_mode") == "none":
            if references:
                raise HTTPException(422, "This model uses the Object description; do not upload reference images")
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
    def get_session(session_id: str):
        return detail_or_404(session_id)

    @app.post("/api/sessions/{session_id}/batches", status_code=202)
    async def create_batch(session_id: str, images: list[UploadFile] = File(...)):
        if store.session_row(session_id) is None:
            raise HTTPException(404, "Session not found")
        if not 1 <= len(images) <= MAX_TARGETS:
            raise HTTPException(422, "Upload 1–20 target images")
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
            if not slots.acquire(blocking=False):
                raise HTTPException(429, "The counting queue is full; try again shortly")
            try:
                error = store.create_batch(session_id, batch_id, items)
                if error:
                    status = 404 if error == "missing" else 409
                    detail = {"missing": "Session not found", "finished": "Session is finished",
                              "busy": "This session already has a queued or running batch",
                              "duplicate": "Image already exists in this session"}[error]
                    raise HTTPException(status, detail)
                try:
                    executor.submit(run_batch, session_id, batch_id, items)
                except Exception:
                    store.set_batch_status(batch_id, "failed", "Could not schedule counting. Please retry.")
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
        return detail_or_404(session_id)

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

    static_dir = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=static_dir, html=True, check_dir=False), name="static")
    return app


app = create_app()
