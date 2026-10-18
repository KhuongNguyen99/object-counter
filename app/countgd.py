"""Optional CountGD++ inference with boxed examples from separate images."""

import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from PIL import Image

from .tiling import make_tiles, merge_tile_detections

MAX_QUERIES = 900

def _exemplar_image(references: list[dict]) -> tuple[Image.Image, list[list[float]]]:
    """Keep a single source exact; arrange examples from several sources in tiles."""
    from .inference import _read_image, _reference_boxes

    if not references:
        raise ValueError("Provide at least one reference image.")
    sources = [Path(reference["path"]).resolve() for reference in references]
    if len(set(sources)) == 1:
        image = _read_image(str(sources[0]))
        boxes = [box.tolist() for reference in references
                 for box in _reference_boxes(reference["boxes"], *image.size)]
        return image, boxes

    examples = []
    for reference in references:
        image = _read_image(reference["path"])
        for box in _reference_boxes(reference["boxes"], *image.size):
            examples.append((image, box))

    tile_size, inset = 256, 8
    columns = math.ceil(math.sqrt(len(examples)))
    rows = math.ceil(len(examples) / columns)
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), (128, 128, 128))
    sheet_boxes = []
    for index, (image, box) in enumerate(examples):
        x1, y1, x2, y2 = map(float, box)
        width, height = x2 - x1, y2 - y1
        margin_x, margin_y = max(4, width * .3), max(4, height * .3)
        crop_left, crop_top = max(0, math.floor(x1 - margin_x)), max(0, math.floor(y1 - margin_y))
        crop_right = min(image.width, math.ceil(x2 + margin_x))
        crop_bottom = min(image.height, math.ceil(y2 + margin_y))
        crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        scale = min((tile_size - 2 * inset) / crop.width, (tile_size - 2 * inset) / crop.height)
        new_width = max(1, round(crop.width * scale))
        new_height = max(1, round(crop.height * scale))
        crop = crop.resize((new_width, new_height), Image.Resampling.BILINEAR)
        scale_x, scale_y = new_width / (crop_right - crop_left), new_height / (crop_bottom - crop_top)
        origin_x = (index % columns) * tile_size + (tile_size - new_width) // 2
        origin_y = (index // columns) * tile_size + (tile_size - new_height) // 2
        sheet.paste(crop, (origin_x, origin_y))
        sheet_boxes.append([
            origin_x + (x1 - crop_left) * scale_x,
            origin_y + (y1 - crop_top) * scale_y,
            origin_x + (x2 - crop_left) * scale_x,
            origin_y + (y2 - crop_top) * scale_y,
        ])
    return sheet, sheet_boxes


class CountGDEngine:
    """Load one trusted CountGD++ checkpoint and reuse it for a batch of images."""

    def __init__(self, model_dir: Path, device: str = "auto") -> None:
        from .models import validate_countgd_assets

        model_dir = Path(model_dir).resolve()
        validate_countgd_assets(model_dir)
        source = Path(os.environ.get("COUNTGD_SOURCE_DIR", "/opt/countgd")).resolve()
        if not (source / "cfg_app.py").is_file():
            raise FileNotFoundError("CountGD++ source is missing from COUNTGD_SOURCE_DIR.")
        source_name = str(source)
        if source_name not in sys.path:
            sys.path.insert(0, source_name)

        import torch
        import datasets.transforms_app as transforms
        from models.GroundingDINO import groundingdino_app
        from util.misc import nested_tensor_from_tensor_list
        from util.slconfig import SLConfig

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device.isdigit():
            device = f"cuda:{device}"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CountGD++ requested CUDA, but no CUDA device is available.")
        self._torch = torch
        self._nested_tensor = nested_tensor_from_tensor_list

        config = SLConfig.fromfile(str(source / "cfg_app.py"))
        config.merge_from_dict({"text_encoder_type": str(model_dir / "bert-base-uncased")})
        args = SimpleNamespace(**config._cfg_dict.to_dict())
        args.device = str(self.device)
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        model, _, _ = groundingdino_app.build_groundingdino(args)
        # The pinned official checkpoint contains an argparse.Namespace; its hash is
        # checked above before allowing pickle deserialization on modern PyTorch.
        checkpoint = torch.load(model_dir / "countgd_plusplus.pth", map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(checkpoint, strict=False)
        self.model = model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.RandomResize([800], max_size=1333),
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            ]),
        ])

    def _predict(self, image: Image.Image, exemplar_image: Image.Image,
                 exemplar_boxes: list[list[float]], confidence: float, label: str,
                 *, conditioning=None):
        # The official 800/1333 resize rounds its shorter axis to zero for
        # extreme aspect ratios, which PIL cannot resize. Reject before it runs.
        for source_image in (image, exemplar_image):
            if round(1333 * min(source_image.size) / max(source_image.size)) < 1:
                raise ValueError("CountGD++ cannot process this image aspect ratio.")
        torch = self._torch
        input_image, _ = self.transform(image, None)
        # The first period divides positive from negative tokens in the official
        # postprocessor, so user punctuation must not precede its separator.
        caption = label.replace(".", " ") + " . "
        with torch.inference_mode():
            target = self._nested_tensor([input_image.to(self.device)])
            if conditioning is None:
                exemplar, boxes = self._reference_tensors(exemplar_image, exemplar_boxes)
                prediction = self.model(target, exemplar, boxes, [], [], captions=[caption])
            else:
                prediction = self.model.forward_conditioned(target, conditioning)
            tokens = prediction["input_ids"][0]
            positive_end = next((i for i, token in enumerate(tokens) if int(token) == 1012), None)
            if positive_end is None:
                raise RuntimeError("CountGD++ did not return a positive prompt separator.")
            logits = prediction["pred_logits"].sigmoid()[0]
            positive = logits[:, :positive_end + 1].amax(dim=-1)
            remaining = logits[:, positive_end + 1:]
            negative = remaining.amax(dim=-1) if remaining.shape[1] else torch.full_like(positive, -float("inf"))
            selected = (positive > confidence) & (positive > negative)
            boxes = prediction["pred_boxes"][0][selected].cpu().numpy()
            scores = logits[selected].amax(dim=-1).cpu().numpy()
        return boxes, scores

    def _reference_tensors(self, image: Image.Image, boxes: list[list[float]]):
        if round(1333 * min(image.size) / max(image.size)) < 1:
            raise ValueError("CountGD++ cannot process this image aspect ratio.")
        image, transformed = self.transform(
            image, {"exemplars": self._torch.tensor(boxes, dtype=self._torch.float32)},
        )
        return (self._nested_tensor([image.to(self.device)]),
                [transformed["exemplars"].to(self.device)])

    def _prepare_conditioning(self, exemplar_image, exemplar_boxes, label):
        with self._torch.inference_mode():
            exemplar, boxes = self._reference_tensors(exemplar_image, exemplar_boxes)
            caption = label.replace(".", " ") + " . "
            return self.model.prepare_conditioning(exemplar, boxes, [], [], captions=[caption])

    def count(self, references: list[dict], image_paths: list[str], output_dir: str,
              confidence: float = .25, label: str = "", *, cache_reference: bool = True) -> list[dict]:
        from .inference import _draw_detections, _read_image

        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        if not image_paths:
            raise ValueError("Provide at least one image to count.")
        exemplar_image, exemplar_boxes = _exemplar_image(references)
        destination_dir = Path(output_dir).resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        conditioning = None
        for image_path in image_paths:
            started = time.perf_counter()
            image = _read_image(image_path)
            # Tile only when the official resize would discard source detail.
            resize_scale = min(800 / min(image.size), 1333 / max(image.size))
            tiles = make_tiles(image.width, image.height, 800 if resize_scale < 1 else max(image.size))
            predictions = []
            detection_limit_reached = False
            if cache_reference and conditioning is None and hasattr(self, "model"):
                conditioning = self._prepare_conditioning(exemplar_image, exemplar_boxes, label)
            for tile in tiles:
                crop = image if len(tiles) == 1 else image.crop(tile.box)
                args = (crop, exemplar_image, exemplar_boxes, confidence, label)
                boxes, scores = (self._predict(*args, conditioning=conditioning) if conditioning is not None
                                 else self._predict(*args))
                detection_limit_reached |= len(boxes) >= MAX_QUERIES
                detections = []
                for box, score in zip(boxes, scores, strict=True):
                    center_x, center_y, width, height = map(float, box)
                    score = float(score)
                    if (not all(math.isfinite(value) for value in (center_x, center_y, width, height, score))
                            or not 0 <= center_x <= 1 or not 0 <= center_y <= 1
                            or width <= 0 or height <= 0 or not 0 <= score <= 1):
                        raise ValueError("CountGD++ returned an invalid detection.")
                    xyxy = [
                        (center_x - width / 2) * crop.width,
                        (center_y - height / 2) * crop.height,
                        (center_x + width / 2) * crop.width,
                        (center_y + height / 2) * crop.height,
                    ]
                    detections.append({"box": xyxy, "confidence": score})
                predictions.append((tile, detections))
            raw_count = sum(len(detections) for _, detections in predictions)
            detections = (merge_tile_detections(predictions, image.width, image.height)
                          if len(tiles) > 1 else predictions[0][1])
            detections.sort(key=lambda detection: (detection["box"][1], detection["box"][0]))
            for index, detection in enumerate(detections, start=1):
                detection["index"] = index
            destination = destination_dir / f"{uuid4().hex}.jpg"
            _draw_detections(image, detections, destination)
            warning = "Experimental model; review overlapping objects."
            if detection_limit_reached:
                warning += " Detection limit reached in an image or tile; the count may be low."
            outputs.append({
                "count": len(detections), "detections": detections,
                "annotated_path": str(destination), "width": image.width, "height": image.height,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "warning": warning,
            })
            if len(tiles) > 1:
                outputs[-1]["processing"] = {
                    "tiles": len(tiles), "tile_size": 800,
                    "duplicates_removed": raw_count - len(detections),
                }
        return outputs
