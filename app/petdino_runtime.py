"""Persistent PET-DINO worker, run only with its isolated MMDetection runtime."""

import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import numpy as np
from PIL import Image

from .tiling import make_tiles, merge_tile_detections

MAX_QUERIES = 900
CHECKPOINT = "pet_dino_swin-t_8xb4_12e_obj365.pth"
CONFIG = "configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py"


def _image_input(image: Image.Image) -> np.ndarray:
    # MMDetection accepts BGR arrays; the configured preprocessor converts to RGB.
    if round(1333 * min(image.size) / max(image.size)) < 1:
        raise ValueError("PET-DINO cannot process this image aspect ratio.")
    return np.asarray(image)[:, :, ::-1].copy()


class PetDINORuntime:
    """Keep weights resident; prepare fresh visual prompts once per batch."""

    def __init__(self, model_dir: Path, device: str = "auto") -> None:
        from .models import countgd_dir, validate_petdino_assets

        model_dir = Path(model_dir).resolve()
        validate_petdino_assets(model_dir)
        source = Path(os.environ.get("PETDINO_SOURCE_DIR", "/opt/petdino")).resolve()
        if not (source / CONFIG).is_file():
            raise FileNotFoundError("PET-DINO source is missing from PETDINO_SOURCE_DIR.")
        sys.path.insert(0, str(source))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import torch
        from mmengine.config import Config
        from mmdet.apis import DetInferencer

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device.isdigit():
            device = f"cuda:{device}"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PET-DINO requested CUDA, but no CUDA device is available.")
        self._torch = torch
        config = Config.fromfile(str(source / CONFIG))
        config.model.language_model.name = str(countgd_dir() / "bert-base-uncased")
        config.model.backbone.init_cfg = None
        config.load_from = None
        # Training memory banks require dataset files and do not participate in prediction.
        config.model.memory_bank = {}
        config.model.use_autocast = False
        config.model.test_cfg.update(max_per_img=MAX_QUERIES, chunked_size=-1, prompt_type="Visual")
        config.visualizer.vis_backends = []
        self.inferencer = DetInferencer(
            model=config, weights=str(model_dir / CHECKPOINT), device=str(self.device),
            palette="random", show_progress=False,
        )
        self.model = self.inferencer.model.eval().float()

    def _prepare_references(self, references: list[dict]):
        from .inference import _read_image, _reference_boxes

        if not isinstance(references, list) or not references:
            raise ValueError("Provide at least one reference image.")
        grouped = {}
        for reference in references:
            path = str(Path(reference["path"]).resolve())
            grouped.setdefault(path, []).extend(reference["boxes"])
        embeddings = []
        torch = self._torch
        with torch.inference_mode():
            for path, raw_boxes in grouped.items():
                image = _read_image(path)
                boxes = _reference_boxes(raw_boxes, *image.size)
                data = self.inferencer.prompt_image_test_pipeline({
                    "img": _image_input(image), "prompt_type": "Visual",
                    "prompt_bboxes": boxes.tolist(), "prompt_bboxes_labels": [0] * len(boxes),
                })
                embedding = self.inferencer.extract_visual_embedding(self.inferencer.collate_fn([data]))
                if embedding.shape != (1, 1, 256) or not torch.isfinite(embedding).all():
                    raise RuntimeError("PET-DINO returned invalid visual-reference embeddings.")
                embeddings.append(embedding[0, 0])
            # Official extract-visual-embedding: learned aggregation within each image,
            # optional enhancement, then mean pooling across images of the same class.
            stacked = torch.stack(embeddings)
            if hasattr(self.model, "visual_prompt_enhancer"):
                stacked = self.inferencer.forward_visual_prompt_enhancer(stacked)
            pooled = stacked.mean(dim=0, keepdim=True)
            if not torch.isfinite(pooled).all():
                raise RuntimeError("PET-DINO returned invalid visual-reference embeddings.")
            return pooled

    def _predict(self, image: Image.Image, embedding, confidence: float, label: str):
        torch = self._torch
        with torch.inference_mode():
            data = self.inferencer.pipeline({
                "img": _image_input(image), "text": [label.strip() or "object"],
                "custom_entities": True, "tokens_positive": None,
            })
            # The official precomputed-memory branch needs no prompt image or encoder.
            data["data_samples"].set_metainfo({
                "prompt_type": "Visual", "memory_visual": embedding,
                "memory_label": torch.tensor([0]), "memory_class_name": [label.strip() or "object"],
            })
            prediction = self.inferencer.forward(self.inferencer.collate_fn([data]))[0].pred_instances
            selected = prediction.scores > confidence
            packed = torch.cat((prediction.bboxes[selected], prediction.scores[selected, None]), dim=-1)
            rows = packed.detach().cpu().numpy()
        return rows[:, :4], rows[:, 4]

    def count(self, references: list[dict], image_paths: list[str], output_dir: str,
              confidence: float = .3, label: str = "") -> list[dict]:
        from .inference import _draw_detections, _read_image

        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        if not isinstance(image_paths, list) or not image_paths:
            raise ValueError("Provide at least one image to count.")
        if not isinstance(label, str):
            raise ValueError("The object label must be text.")
        destination_dir = Path(output_dir).resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
        embedding = self._prepare_references(references)
        outputs = []
        for image_path in image_paths:
            started = time.perf_counter()
            image = _read_image(image_path)
            resize_scale = min(800 / min(image.size), 1333 / max(image.size))
            tiles = make_tiles(image.width, image.height, 800 if resize_scale < 1 else max(image.size))
            predictions = []
            detection_limit_reached = False
            for tile in tiles:
                crop = image if len(tiles) == 1 else image.crop(tile.box)
                boxes, scores = self._predict(crop, embedding, confidence, label)
                detection_limit_reached |= len(boxes) >= MAX_QUERIES
                detections = []
                for box, score in zip(boxes, scores, strict=True):
                    if len(box) != 4:
                        raise ValueError("PET-DINO returned an invalid detection.")
                    x1, y1, x2, y2 = map(float, box)
                    score = float(score)
                    if (not all(math.isfinite(value) for value in (x1, y1, x2, y2, score))
                            or not 0 <= score <= 1
                            or min(x2, crop.width) <= max(x1, 0)
                            or min(y2, crop.height) <= max(y1, 0)):
                        raise ValueError("PET-DINO returned an invalid detection.")
                    detections.append({"box": [max(0., x1), max(0., y1), min(float(crop.width), x2),
                                               min(float(crop.height), y2)], "confidence": score})
                predictions.append((tile, detections))
            raw_count = sum(len(records) for _, records in predictions)
            detections = (merge_tile_detections(predictions, image.width, image.height)
                          if len(tiles) > 1 else predictions[0][1])
            detections.sort(key=lambda item: (item["box"][1], item["box"][0]))
            for index, detection in enumerate(detections, 1):
                detection["index"] = index
            destination = destination_dir / f"{uuid4().hex}.jpg"
            _draw_detections(image, detections, destination)
            warning = "Experimental model; review the numbered detections."
            if detection_limit_reached:
                warning += " Detection limit reached in an image or tile; the count may be low."
            elif not detections:
                warning += " No matches detected. Check the reference or lower confidence."
            outputs.append({
                "count": len(detections), "detections": detections, "annotated_path": str(destination),
                "width": image.width, "height": image.height,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2), "warning": warning,
            })
            if len(tiles) > 1:
                outputs[-1]["processing"] = {
                    "tiles": len(tiles), "tile_size": 800, "duplicates_removed": raw_count - len(detections),
                }
        return outputs


def _serve(model_dir, device, requests, responses, factory=PetDINORuntime) -> int:
    def send(message):
        responses.write(json.dumps(message, allow_nan=False) + "\n")
        responses.flush()

    with redirect_stdout(sys.stderr):
        try:
            engine = factory(model_dir, device)
        except Exception as exc:
            send({"ready": False, "error": str(exc)[:2000] or type(exc).__name__})
            return 1
        send({"ready": True})
        for line in requests:
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Worker requests must be JSON objects.")
                if request.get("action") == "close":
                    return 0
                if request.get("action") != "count":
                    raise ValueError("Unknown PET-DINO worker action.")
                results = engine.count(
                    request["references"], request["image_paths"], request["output_dir"],
                    request.get("confidence", .3), request.get("label", ""),
                )
                send({"ok": True, "results": results})
            except Exception as exc:
                send({"ok": False, "error": str(exc)[:2000] or type(exc).__name__})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    args = parser.parse_args()
    # Keep even C/CUDA stdout and framework handlers away from the JSON pipe.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with protocol:
        return _serve(args.model_dir, args.device, sys.stdin, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
