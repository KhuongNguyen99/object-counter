"""Serialized local inference with one cached model at a time."""

import gc
import math
import os
from pathlib import Path
import threading
import time
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .models import (COUNTGD_MODEL_ID, CROSSCHECK_MODEL_ID, FASTSAM_FILENAME, FASTSAM_MODEL_ID, PETDINO_MODEL_ID,
                     SAM3COUNT_MODEL_ID, SAM3_MODEL_ID, checkpoint_path, model_dir, countgd_dir,
                     petdino_dir, sam3count_dir, validate_checkpoint)
from .tiling import make_tiles, merge_tile_detections


def _read_image(path: str) -> Image.Image:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Image does not exist: {source.name}")
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _reference_boxes(boxes: list, width: int, height: int) -> np.ndarray:
    array = np.asarray(boxes, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 4 or not len(array):
        raise ValueError("Each reference image needs at least one box with four pixel coordinates.")
    if not np.isfinite(array).all():
        raise ValueError("Reference box coordinates must be finite numbers.")
    if (
        np.any(array[:, :2] < 0)
        or np.any(array[:, 2] > width)
        or np.any(array[:, 3] > height)
        or np.any(array[:, 2:] <= array[:, :2])
    ):
        raise ValueError("Reference boxes must have positive area and stay within their image.")
    return array


def _draw_detections(image: Image.Image, detections: list[dict], destination: Path) -> None:
    draw = ImageDraw.Draw(image)
    font_size = max(12, min(28, round(min(image.size) / 55)))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)
    line_width = max(2, round(min(image.size) / 400))
    color = "#34e3aa"
    for detection in detections:
        draw.rectangle(tuple(detection["box"]), outline=color, width=line_width)
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = str(detection["index"])
        bounds = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        left = max(0, min((x1 + x2 - text_width) / 2 - 3, image.width - text_width - 6))
        top = max(0, min((y1 + y2 - text_height) / 2 - 3, image.height - text_height - 6))
        draw.rounded_rectangle((left, top, left + text_width + 6, top + text_height + 6),
                               radius=3, fill="#073d32")
        draw.text((left + 3, top + 3 - bounds[1]), label, font=font, fill="white")
    label = f"COUNT: {len(detections)}"
    bounds = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, bounds[2] - bounds[0] + 24, bounds[3] - bounds[1] + 20), fill="#102522")
    draw.text((12, 10 - bounds[1]), label, font=font, fill="white")
    image.save(destination, format="JPEG", quality=94)


class InferenceEngine:
    """Cache one model; rebuild counting prompts on every run."""

    def __init__(self) -> None:
        self._model = None
        self._model_id = None
        self._lock = threading.RLock()

    @staticmethod
    def _device():
        import torch
        device = os.environ.get("DEVICE", "auto")
        if device == "auto":
            device = "0" if torch.cuda.is_available() else "cpu"
        return device.split(":", 1)[1] if device.startswith("cuda:") else device

    def _unload_model(self):
        if self._model is not None and hasattr(self._model, "close"):
            self._model.close()
        self._model = None
        self._model_id = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self):
        with self._lock:
            self._unload_model()

    def _load_countgd_model(self, device):
        if self._model_id != COUNTGD_MODEL_ID or self._model is None:
            from .countgd import CountGDEngine
            self._unload_model()
            target_device = f"cuda:{device}" if device.isdigit() else device
            self._model = CountGDEngine(countgd_dir(), target_device)
            self._model_id = COUNTGD_MODEL_ID
        return self._model

    def suggest_reference_boxes(self, image_path: str, label: str = "") -> dict:
        """Propose editable boxes using the same lock and one-model cache as counting."""
        with self._lock:
            from .reference import ReferenceModelUnavailable, propose_boxes
            try:
                if self._model_id != FASTSAM_MODEL_ID or self._model is None:
                    path = model_dir() / FASTSAM_FILENAME
                    validate_checkpoint(path)
                    from ultralytics import FastSAM

                    self._unload_model()
                    self._model = FastSAM(str(path))
                    self._model_id = FASTSAM_MODEL_ID
            except (FileNotFoundError, ValueError, ImportError) as exc:
                raise ReferenceModelUnavailable("FastSAM reference model is unavailable") from exc
            return propose_boxes(self._model, image_path, self._device())

    def _load_model(self, model_id: str):
        path = checkpoint_path(model_id)
        if self._model_id != model_id or self._model is None:
            # Check the exact local release bytes before Ultralytics can attempt a download.
            validate_checkpoint(path)
            os.environ["YOLO_OFFLINE"] = "true"
            os.environ["YOLO_AUTOINSTALL"] = "false"
            import torch
            from ultralytics import YOLOE

            self._unload_model()
            self._model = YOLOE(str(path), task="segment")
            self._model_id = model_id
        self._model.predictor = None
        if hasattr(self._model.model, "pe"):
            del self._model.model.pe
        return self._model

    def count(
        self,
        model_id: str,
        references: list[dict],
        image_paths: list[str],
        output_dir: str,
        confidence: float = 0.25,
        label: str = "",
        *,
        reference_strategy: str = "mean",
        reference_mode: str = "shared",
    ) -> list[dict]:
        """Combine exemplars once, then infer each target at native pixel resolution.

        DEVICE defaults to auto (CUDA when available, otherwise CPU). IMGSZ defaults to 640;
        MAX_DETECTIONS defaults to 3000 per tile. elapsed_ms includes target inference and annotation,
        excluding checkpoint loading and reference encoding shared by the whole batch.
        """
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        if reference_strategy not in ("mean", "per-object"):
            raise ValueError("Reference strategy must be mean or per-object.")
        if reference_strategy == "per-object" and not model_id.startswith("yoloe-"):
            raise ValueError("Per-object reference strategy is supported only for YOLOE.")
        if reference_mode not in ("shared", "per_image") or (reference_mode == "per_image" and model_id != COUNTGD_MODEL_ID):
            raise ValueError("Per-image reference mode is supported only for CountGD++.")
        if not image_paths or (not references and model_id != SAM3COUNT_MODEL_ID):
            raise ValueError("Provide reference boxes and at least one image to count.")
        if model_id == COUNTGD_MODEL_ID and reference_mode == "per_image":
            if len(references) != len(image_paths):
                raise ValueError("CountGD++ needs one set of reference boxes for each image to count.")
            for reference, image_path in zip(references, image_paths, strict=True):
                if (not isinstance(reference, dict) or not isinstance(reference.get("path"), str) or
                        not Path(reference["path"]).is_absolute() or
                        Path(reference["path"]).resolve() != Path(image_path).resolve()):
                    raise ValueError("Each CountGD++ reference path must match its target image.")
                if not isinstance(reference.get("boxes"), list) or len(reference["boxes"]) != 10:
                    raise ValueError("Each CountGD++ target image needs exactly 10 distinct boxes.")
                boxes = reference["boxes"]
                if any(not isinstance(box, list) or len(box) != 4 or
                       any(isinstance(n, bool) or not isinstance(n, (int, float)) or
                           (isinstance(n, float) and not math.isfinite(n)) for n in box)
                       for box in boxes):
                    raise ValueError("CountGD++ boxes need four finite numeric pixel coordinates.")
                if len({tuple(box) for box in boxes}) != 10:
                    raise ValueError("Each CountGD++ target image needs exactly 10 distinct boxes.")
                image = _read_image(image_path)
                _reference_boxes(boxes, *image.size)
        if model_id == SAM3COUNT_MODEL_ID and (references or not label.strip()):
            raise ValueError("SAM3Count requires an Object description and no reference photos.")
        if model_id == SAM3_MODEL_ID:
            from .sam3visual import _sam3_reference_boxes

            if len(references) != len(image_paths):
                raise ValueError("SAM3 needs one set of reference boxes for each image to count.")
            for reference, image_path in zip(references, image_paths, strict=True):
                if (not isinstance(reference, dict) or
                        not isinstance(reference.get("path"), str) or
                        not Path(reference["path"]).is_absolute() or
                        Path(reference["path"]).resolve() != Path(image_path).resolve()):
                    raise ValueError("Each SAM3 reference path must match its target image.")
                image = _read_image(image_path)
                _sam3_reference_boxes(reference.get("boxes"), *image.size)
        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if model_id == CROSSCHECK_MODEL_ID:
                return self._count_crosscheck(references, image_paths, str(directory), confidence, label)
            import torch
            device = self._device()
            if model_id == SAM3_MODEL_ID:
                if self._model_id != SAM3_MODEL_ID or self._model is None:
                    from .sam3visual import Sam3VisualEngine

                    self._unload_model()
                    self._model = Sam3VisualEngine(sam3count_dir(),
                                                   f"cuda:{device}" if device.isdigit() else device)
                    self._model_id = SAM3_MODEL_ID
                try:
                    return self._model.count(references, image_paths, str(directory), confidence)
                except Exception:
                    self._unload_model()
                    raise
            if model_id == SAM3COUNT_MODEL_ID:
                if self._model_id != SAM3COUNT_MODEL_ID or self._model is None:
                    from .sam3count import Sam3CountEngine
                    self._unload_model()
                    self._model = Sam3CountEngine(sam3count_dir(), f"cuda:{device}" if device.isdigit() else device)
                    self._model_id = SAM3COUNT_MODEL_ID
                try:
                    return self._model.count(references, image_paths, str(directory), confidence, label)
                except Exception:
                    self._unload_model()
                    raise
            if model_id == COUNTGD_MODEL_ID:
                model = self._load_countgd_model(device)
                if reference_mode == "per_image":
                    return [result for reference, image_path in zip(references, image_paths, strict=True)
                            for result in model.count([reference], [image_path], str(directory), confidence, label)]
                return model.count(references, image_paths, str(directory), confidence, label)
            if model_id == PETDINO_MODEL_ID:
                if self._model_id != PETDINO_MODEL_ID or self._model is None:
                    from .petdino import PetDINOEngine
                    self._unload_model()
                    self._model = PetDINOEngine(petdino_dir(), f"cuda:{device}" if device.isdigit() else device)
                    self._model_id = PETDINO_MODEL_ID
                try:
                    return self._model.count(references, image_paths, str(directory), confidence, label)
                except Exception:
                    self._unload_model()
                    raise
            model = self._load_model(model_id)
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
            image_size = int(os.environ.get("IMGSZ", "640"))
            max_detections = int(os.environ.get("MAX_DETECTIONS", "3000"))
            if not 32 <= image_size <= 4096 or not 1 <= max_detections <= 30000:
                raise ValueError("IMGSZ must be 32–4096 and MAX_DETECTIONS must be 1–30000.")
            options = {
                "device": device, "imgsz": image_size, "verbose": False,
                "save": False, "batch": 1, "nms": True, "agnostic_nms": True,
                "conf": confidence, "iou": 0.5, "max_det": max_detections,
            }
            with torch.inference_mode():
                encoder = YOLOEVPSegPredictor(overrides={**options, "task": "segment", "mode": "predict"})
                encoder.setup_model(model=model.model, verbose=False)
                embeddings = []
                for reference in references:
                    image = _read_image(reference["path"])
                    boxes = _reference_boxes(reference["boxes"], *image.size)
                    # Unique temporary IDs preserve every box before the selected pooling strategy.
                    encoder.set_prompts({"bboxes": boxes, "cls": np.arange(len(boxes), dtype=np.int32)})
                    embedding = encoder.get_vpe(image)
                    if (embedding.ndim != 3 or embedding.shape[:2] != (1, len(boxes)) or
                            embedding.shape[-1] == 0 or not torch.isfinite(embedding).all()):
                        raise RuntimeError("The model returned invalid visual-reference embeddings.")
                    embeddings.append(embedding)
                prototypes = torch.cat(embeddings, dim=1)
                if reference_strategy == "mean":
                    # Standard counting follows Ultralytics' multi-image visual validation.
                    prototypes = prototypes.mean(dim=1, keepdim=True)
                if (prototypes.norm(dim=-1) < 1e-8).any():
                    raise ValueError("These reference boxes did not produce a usable object embedding.")
                prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=-1)
                del encoder, embeddings
                classes = (["object"] if reference_strategy == "mean" else
                           [f"object-{i}" for i in range(prototypes.shape[1])])
                model.set_classes(classes, embeddings=prototypes)

                outputs = []
                for image_path in image_paths:
                    started = time.perf_counter()
                    image = _read_image(image_path)
                    tiles = make_tiles(image.width, image.height, image_size)
                    predictions = []
                    detection_limit_reached = False
                    for tile in tiles:
                        crop = image if len(tiles) == 1 else image.crop(tile.box)
                        prediction = model.predict(source=crop, **options)[0]
                        local_detections = []
                        if prediction.boxes is not None:
                            boxes = prediction.boxes.xyxy.detach().cpu().tolist()
                            scores = prediction.boxes.conf.detach().cpu().tolist()
                            detection_limit_reached |= len(boxes) >= max_detections
                            for box, score in zip(boxes, scores, strict=True):
                                if len(box) != 4:
                                    continue
                                coordinates = tuple(map(float, box))
                                score = float(score)
                                if (not all(math.isfinite(value) for value in (*coordinates, score))
                                        or not 0 <= score <= 1
                                        or min(coordinates[2], crop.width) <= max(coordinates[0], 0)
                                        or min(coordinates[3], crop.height) <= max(coordinates[1], 0)):
                                    continue
                                local_detections.append({"box": list(coordinates), "confidence": score})
                        predictions.append((tile, local_detections))
                    raw_count = sum(len(records) for _, records in predictions)
                    detections = (merge_tile_detections(predictions, image.width, image.height)
                                  if len(tiles) > 1 else predictions[0][1])
                    detections.sort(key=lambda item: (item["box"][1], item["box"][0]))
                    for index, detection in enumerate(detections, 1):
                        detection["index"] = index
                    destination = directory / f"{uuid4().hex}.jpg"
                    _draw_detections(image, detections, destination)
                    outputs.append({
                        "count": len(detections), "detections": detections,
                        "annotated_path": str(destination), "width": image.width, "height": image.height,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    })
                    if len(tiles) > 1:
                        outputs[-1]["processing"] = {
                            "tiles": len(tiles), "tile_size": image_size,
                            "duplicates_removed": raw_count - len(detections),
                        }
                    if detection_limit_reached:
                        outputs[-1]["warning"] = "Detection limit reached; this image may contain additional objects."
                    elif not detections:
                        outputs[-1]["warning"] = "No matches detected. Check the reference or use a model trained for this object."
                return outputs

    def _count_crosscheck(self, references, image_paths, output_dir, confidence, label):
        from .verification import compare_detections

        started = time.perf_counter()
        primary = self.count(COUNTGD_MODEL_ID, references, image_paths, output_dir, confidence, label)
        try:
            secondary = self.count("yoloe-26x-seg", references, image_paths, output_dir, .01, label,
                                   reference_strategy="per-object")
            if len(primary) != len(secondary):
                raise ValueError("Cross-check returned the wrong number of images")
            comparisons = [{**compare_detections(a["detections"], b["detections"], COUNTGD_MODEL_ID,
                                                "yoloe-26x-seg", confidence, .01),
                            "primary_elapsed_ms": a["elapsed_ms"],
                            "secondary_elapsed_ms": b["elapsed_ms"],
                            "secondary_warning": b.get("warning")}
                           for a, b in zip(primary, secondary, strict=True)]
        except Exception as exc:
            # A failed check must not erase a usable primary result or imply agreement.
            comparisons = [{"status": "failed", "primary_model": COUNTGD_MODEL_ID,
                            "secondary_model": "yoloe-26x-seg", "primary_confidence": confidence,
                            "secondary_confidence": .01, "primary_count": result["count"],
                            "primary_elapsed_ms": result["elapsed_ms"], "secondary_elapsed_ms": None,
                            "error": str(exc)[:300]} for result in primary]
            self._unload_model()
        batch_wall_ms = round((time.perf_counter() - started) * 1000, 2)
        for result, comparison in zip(primary, comparisons, strict=True):
            comparison["secondary_reference_strategy"] = "per-object"
            # Sum per-image inference/annotation times; batch wall time also includes model setup.
            result["elapsed_ms"] = round(comparison["primary_elapsed_ms"] +
                                         (comparison["secondary_elapsed_ms"] or 0), 2)
            result["verification"] = comparison
            result["warning"] = ((result.get("warning") or "") +
                                 " Cross-check requires image review; both models can miss the same object.").strip()
            comparison["batch_wall_ms"] = batch_wall_ms
        return primary
