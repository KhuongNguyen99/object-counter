"""Pinned SAM3Count image pipeline with native-resolution coverage for large inputs.

Density, reliability, ROI and tile-grid decisions come from the MIT-licensed
Joan947/SAM3Count source. Box-IoU then mask-IoM suppression follows its image
pipeline; storing only each mask's occupied region avoids full-frame tile masks.
Images larger than SAM3's 1008px input additionally receive full tile coverage
and crop-boundary fragment association before upstream suppression, so NMS
cannot discard the overlapping observations needed to join a large object.
"""

import ast
from contextlib import nullcontext
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import numpy as np
from PIL import Image

from .tiling import Tile, merge_tile_detections

RESOLUTION = 1008
MAX_TILES = 256
QUERY_WARNING = 190
UPSTREAM_COMMIT = "1840710099a8b2b8bdf5674c3be8a4367aa76830"
_HELPERS = {
    "calculate_adaptive_tile_params", "calculate_density_smart",
    "DensityAwareTiledDetector", "_clip_box_to_image", "_boxes_union_roi",
    "_box_intersection_area", "_filter_tiles_by_roi",
    "assess_stage1_detection_reliability",
}


def _load_algorithm(source: Path):
    """Load unchanged helpers without upstream demo imports or global patches."""
    filename = source / "sam3count_images.py"
    if not filename.is_file():
        raise FileNotFoundError("SAM3Count source is missing from SAM3COUNT_SOURCE_DIR.")
    parsed = ast.parse(filename.read_text(encoding="utf-8"), filename=str(filename))
    nodes = [node for node in parsed.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _HELPERS]
    if {node.name for node in nodes} != _HELPERS:
        raise RuntimeError("SAM3Count source does not contain the pinned image algorithm.")
    namespace = {"np": np, "List": List, "Dict": Dict, "Any": Any, "Tuple": Tuple,
                 "print": lambda *args, **kwargs: None}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(filename), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in _HELPERS})


def _compact_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    rows, columns = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not len(rows) or not len(columns):
        return np.zeros((0, 0), dtype=bool), (0, 0)
    x, y = int(columns[0]), int(rows[0])
    return mask[y:int(rows[-1]) + 1, x:int(columns[-1]) + 1].copy(), (x, y)


def _mask_iom(first: dict, second: dict) -> float:
    """Exact full-frame mask IoM, evaluated only in intersecting stored regions."""
    area = min(first["area"], second["area"])
    if not area:
        return 0.
    a, b = first["mask"], second["mask"]
    ax, ay = first["origin"]
    bx, by = second["origin"]
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + a.shape[1], bx + b.shape[1]), min(ay + a.shape[0], by + b.shape[0])
    if right <= left or bottom <= top:
        return 0.
    intersection = np.logical_and(a[top-ay:bottom-ay, left-ax:right-ax],
                                  b[top-by:bottom-by, left-bx:right-bx]).sum()
    return float(intersection / area)


def _mask_nms(detections: list[dict]) -> list[dict]:
    # Match upstream's descending NumPy ordering, including tied scores.
    indices = np.argsort([det["score"] for det in detections])[::-1]
    kept, suppressed = [], set()
    for i in indices:
        if i in suppressed:
            continue
        kept.append(detections[i])
        for j in indices:
            if j != i and j not in suppressed and _mask_iom(detections[i], detections[j]) >= .5:
                suppressed.add(j)
    return kept


def _merge_native_masks(detections, tiles, width, height):
    predictions = []
    for tile_index, tile in enumerate(tiles):
        bounds = (tile["x"], tile["y"], tile["x_end"], tile["y_end"])
        records = []
        for det in detections:
            if det["source_tile"] == tile_index:
                local_box = [value - bounds[axis % 2] for axis, value in enumerate(det["box"])]
                records.append({**det, "box": local_box, "confidence": det["score"]})
        predictions.append((Tile(tile_index, bounds, bounds), records))
    merged = merge_tile_detections(predictions, width, height, include_members=True)
    for det in merged:
        members = det.pop("members")
        det["score"] = max(member["score"] for member in members)
        masks = [member for member in members if member["area"]]
        if not masks:
            continue
        left = min(member["origin"][0] for member in masks)
        top = min(member["origin"][1] for member in masks)
        right = max(member["origin"][0] + member["mask"].shape[1] for member in masks)
        bottom = max(member["origin"][1] + member["mask"].shape[0] for member in masks)
        mask = np.zeros((bottom - top, right - left), dtype=bool)
        for member in masks:
            x, y = member["origin"][0] - left, member["origin"][1] - top
            height_, width_ = member["mask"].shape
            mask[y:y + height_, x:x + width_] |= member["mask"]
        det.update(mask=mask, origin=(left, top), area=int(mask.sum()))
    return merged


def _load_finetuned_weights(model, checkpoint_path, torch):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("SAM3Count fine-tuned checkpoint does not contain model weights.")
    model_keys = set(model.state_dict())
    missing = sorted(key for key in model_keys - state.keys() if not key.startswith("segmentation_head."))
    unexpected = sorted(state.keys() - model_keys)
    if missing or unexpected:
        raise ValueError(f"SAM3Count fine-tuned weights do not match the pinned model: "
                         f"missing={missing[:5]}, unexpected={unexpected[:5]}.")
    # The author's detection-only fine-tuning retains the base segmentation head.
    model.load_state_dict(state, strict=False)


def _counting_processor(base):
    class CountingProcessor(base):
        def _forward_grounding(self, state):
            import torch
            from sam3.model.box_ops import box_cxcywh_to_xyxy

            outputs = self.model.forward_grounding(
                backbone_out=state["backbone_out"], find_input=self.find_stage,
                geometric_prompt=state["geometric_prompt"], find_target=None,
            )
            # SAM3Count requires presence gating disabled for counting.
            probabilities = outputs["pred_logits"].sigmoid().squeeze(-1)
            keep = probabilities > 0.
            scale = torch.tensor([state["original_width"], state["original_height"]] * 2,
                                 device=self.device)
            state["boxes"] = box_cxcywh_to_xyxy(outputs["pred_boxes"][keep]) * scale
            state["scores"] = probabilities[keep]
            # Upsample one mask at a time in _predict, instead of N full 4K masks.
            state["counting_mask_logits"] = outputs["pred_masks"][keep]
            return state
    return CountingProcessor


class Sam3CountEngine:
    """One cached SAM3 model with the official text-prompted counting algorithm."""

    def __init__(self, model_dir: Path, device: str = "auto") -> None:
        from .models import validate_sam3count_assets

        directory = Path(model_dir).resolve()
        validate_sam3count_assets(directory)
        source = Path(os.environ.get("SAM3_SOURCE_DIR", "/opt/sam3")).resolve()
        algorithm = Path(os.environ.get("SAM3COUNT_SOURCE_DIR", "/opt/sam3count")).resolve()
        self._algorithm = _load_algorithm(algorithm)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        import torch

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device.isdigit():
            device = f"cuda:{device}"
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This SAM3Count runtime requires an NVIDIA CUDA GPU; CPU inference is unavailable.")
        self._torch = torch
        import sam3.model.vitdet as vitdet
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.perflib.fused import addmm_act

        # Preserve the official SAM3Count fix for the fused MLP's bf16 output.
        def safe_addmm_act(activation, linear, inputs):
            return addmm_act(activation, linear, inputs).to(inputs.dtype)
        vitdet.addmm_act = safe_addmm_act

        with torch.cuda.device(self.device):
            self.model = build_sam3_image_model(
                bpe_path=str(source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
                device="cuda", checkpoint_path=str(directory / "sam3.pt"), load_from_HF=False,
                enable_segmentation=True, enable_inst_interactivity=False, eval_mode=True,
            ).eval()
            _load_finetuned_weights(self.model, directory / "sam3_ft.pt", torch)
        self.processor = _counting_processor(Sam3Processor)(
            self.model, resolution=RESOLUTION, device=self.device, confidence_threshold=0.,
        )

    def _predict(self, image: Image.Image, label: str, *, mask_floor: float = 0.) -> list[dict]:
        torch = self._torch
        precision = (torch.autocast("cuda", dtype=torch.bfloat16)
                     if self.device.type == "cuda" else nullcontext())
        with torch.inference_mode(), precision:
            state = self.processor.set_image(image)
            state = self.processor.set_text_prompt(state=state, prompt=label)
            boxes = state["boxes"].float().cpu().numpy()
            scores = state["scores"].float().cpu().numpy()
            logits = state["counting_mask_logits"]
            if boxes.shape != (len(scores), 4) or len(logits) != len(scores):
                raise ValueError("SAM3Count returned inconsistent detections.")
            detections = []
            for box, score, mask_logits in zip(boxes, scores, logits, strict=True):
                x1, y1, x2, y2 = map(float, box)
                score = float(score)
                if (not all(math.isfinite(v) for v in (x1, y1, x2, y2, score))
                        or not 0 <= score <= 1 or x2 <= x1 or y2 <= y1
                        or min(image.width, x2) <= max(0., x1)
                        or min(image.height, y2) <= max(0., y1)):
                    raise ValueError("SAM3Count returned an invalid detection.")
                compact, origin, area = np.zeros((0, 0), dtype=bool), (0, 0), 0
                if score >= mask_floor:
                    mask = torch.nn.functional.interpolate(
                        mask_logits[None, None], (image.height, image.width),
                        mode="bilinear", align_corners=False,
                    ).sigmoid().gt(.5)[0, 0].cpu().numpy()
                    area = int(mask.sum())
                    compact, origin = _compact_mask(mask)
                detections.append({
                    "box": [max(0., x1), max(0., y1), min(float(image.width), x2),
                            min(float(image.height), y2)],
                    "score": score, "mask": compact, "origin": origin, "area": area,
                })
            return detections

    def _count_image(self, image, confidence, label):
        algorithm = self._algorithm
        width, height = image.size
        initial = self._predict(image, label)
        limit_reached = sum(det["score"] >= confidence for det in initial) >= QUERY_WARNING
        post_nms = _mask_nms(initial)
        cleaned = [det for det in post_nms if det["score"] >= confidence]
        reliability = algorithm.assess_stage1_detection_reliability(
            [det["mask"] for det in initial], [det["score"] for det in initial],
            [det["box"] for det in initial], cleaned, width, height,
            count_after_nms=len(post_nms),
        )
        _, use_tiling, metrics = algorithm.calculate_density_smart(
            [det["box"] for det in cleaned], width, height,
        )
        rule = metrics["tiling_rule"]
        forced = not use_tiling and len(post_nms) > 10 and not cleaned
        if forced:
            use_tiling, rule = True, "SMALL"
        native = max(image.size) > RESOLUTION
        if not use_tiling and not native:
            return cleaned, {"tiles": 1, "tile_size": max(image.size),
                             "duplicates_removed": len(initial) - len(post_nms)}, limit_reached, native

        tile_size, overlap = algorithm.calculate_adaptive_tile_params(
            width, height, tiling_rule=rule if rule != "NONE" else "MEDIUM",
            max_tile_size=RESOLUTION,
        )
        while tile_size > min(width, height) // 2 and tile_size // 2 >= 96:
            tile_size //= 2
            overlap = tile_size // 4
        stride = tile_size - overlap
        if math.ceil(width / stride) * math.ceil(height / stride) > MAX_TILES:
            raise ValueError(f"Native-resolution SAM3Count needs more than {MAX_TILES} tiles. "
                             "Crop the image into smaller regions and count them separately.")
        detector = algorithm.DensityAwareTiledDetector(tile_size, overlap)
        tiles = detector.generate_tiles(height, width, force_tiling=True)
        # A coarse ROI can omit small objects elsewhere in a high-resolution image.
        if not native and reliability["use_roi_tiling"]:
            tiles = algorithm._filter_tiles_by_roi(tiles, reliability["roi_box"], .03) or tiles
        del initial, post_nms, cleaned
        tiled = []
        threshold = confidence * .85 if forced else confidence
        for tile_index, tile in enumerate(tiles):
            left, top = tile["x"], tile["y"]
            records = self._predict(image.crop((left, top, tile["x_end"], tile["y_end"])),
                                    label, mask_floor=threshold)
            limit_reached |= sum(det["score"] >= threshold for det in records) >= QUERY_WARNING
            for det in records:
                det["source_tile"] = tile_index
                det["box"] = [value + (left if axis % 2 == 0 else top)
                              for axis, value in enumerate(det["box"])]
                det["origin"] = (det["origin"][0] + left, det["origin"][1] + top)
            tiled.extend(records)
        raw_count = sum(det["score"] >= threshold for det in tiled)
        if native:
            # Associate before NMS can remove bridge tiles. Same-source-tile
            # exclusivity keeps adjacent objects observed together separate.
            tiled = _merge_native_masks([det for det in tiled if det["score"] >= threshold],
                                        tiles, width, height)
        tiled = _mask_nms(detector.nms_across_tiles(tiled))
        accepted = [det for det in tiled if det["score"] >= threshold]
        return (accepted,
                {"tiles": len(tiles), "tile_size": tile_size,
                 "duplicates_removed": raw_count - len(accepted)}, limit_reached, native)

    def count(self, references: list[dict], image_paths: list[str], output_dir: str,
              confidence: float = .35, label: str = "") -> list[dict]:
        from .inference import _draw_detections, _read_image

        if references:
            raise ValueError("SAM3Count uses the object description and does not accept reference photos.")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("SAM3Count requires an object description.")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        if not image_paths:
            raise ValueError("Provide at least one image to count.")
        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        results = []
        for path in image_paths:
            started = time.perf_counter()
            image = _read_image(path)
            records, processing, limit_reached, native = self._count_image(image, confidence, label.strip())
            records.sort(key=lambda det: (det["box"][1], det["box"][0]))
            detections = [{"box": det["box"], "confidence": det["score"], "index": index}
                          for index, det in enumerate(records, 1)]
            destination = directory / f"{uuid4().hex}.jpg"
            _draw_detections(image, detections, destination)
            warning = "Experimental model; review the numbered detections."
            if native:
                warning += " Native-resolution tiling applied."
            if limit_reached:
                warning += " Detection limit reached in an image or tile; the count may be low."
            elif not detections:
                warning += " No matches detected. Check the description or lower confidence."
            results.append({
                "count": len(detections), "detections": detections, "annotated_path": str(destination),
                "width": image.width, "height": image.height,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "warning": warning, "processing": processing,
            })
        return results

    def close(self):
        self.processor = None
        self.model = None
