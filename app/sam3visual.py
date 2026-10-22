"""Base SAM3 counting from positive boxes on each target image."""

import math
from numbers import Real
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import numpy as np
from PIL import Image


def _sam3_reference_boxes(boxes, width: int, height: int) -> np.ndarray:
    from .inference import _reference_boxes

    if (not isinstance(boxes, (list, tuple)) or not 1 <= len(boxes) <= 32 or
            any(not isinstance(box, (list, tuple)) or len(box) != 4 or
                any(isinstance(value, bool) or not isinstance(value, Real) for value in box)
                for box in boxes)):
        raise ValueError("SAM3 needs 1–32 boxes with real numeric pixel coordinates per image.")
    return _reference_boxes(boxes, width, height)


def _visual_processor(base):
    class VisualProcessor(base):
        def _forward_grounding(self, state):
            import torch
            from sam3.model.box_ops import box_cxcywh_to_xyxy

            outputs = self.model.forward_grounding(
                backbone_out=state["backbone_out"], find_input=self.find_stage,
                geometric_prompt=state["geometric_prompt"], find_target=None,
            )
            probabilities = (outputs["pred_logits"].sigmoid() *
                             outputs["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
            keep = probabilities > self.confidence_threshold
            scale = torch.tensor([state["original_width"], state["original_height"]] * 2,
                                 device=self.device)
            state["boxes"] = box_cxcywh_to_xyxy(outputs["pred_boxes"][keep]) * scale
            state["scores"] = probabilities[keep]
            return state

    return VisualProcessor


class Sam3VisualEngine:
    """One approved base model; fresh visual prompts for each image."""

    def __init__(self, model_dir: Path, device: str = "auto") -> None:
        from .models import validate_sam3_base_asset

        directory = Path(model_dir).resolve()
        validate_sam3_base_asset(directory)
        source = Path(os.environ.get("SAM3_SOURCE_DIR", "/opt/sam3")).resolve()
        if not (source / "sam3/model_builder.py").is_file():
            raise FileNotFoundError("SAM3 source is missing from SAM3_SOURCE_DIR.")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        import torch

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device.isdigit():
            device = f"cuda:{device}"
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SAM3 visual counting requires an NVIDIA CUDA GPU.")
        self._torch = torch
        import sam3.model.vitdet as vitdet
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.perflib.fused import addmm_act

        def safe_addmm_act(activation, linear, inputs):
            return addmm_act(activation, linear, inputs).to(inputs.dtype)
        vitdet.addmm_act = safe_addmm_act

        with torch.cuda.device(self.device):
            self.model = build_sam3_image_model(
                bpe_path=str(source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
                device="cuda", checkpoint_path=str(directory / "sam3.pt"), load_from_HF=False,
                enable_segmentation=False, enable_inst_interactivity=False, eval_mode=True,
            ).eval()
        self.processor = _visual_processor(Sam3Processor)(
            self.model, resolution=1008, device=self.device, confidence_threshold=0.,
        )

    def _predict(self, image: Image.Image, boxes: np.ndarray, confidence: float) -> list[dict]:
        torch = self._torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(image)
            # Pinned SAM3 uses its reserved "visual" token for box-only grounding.
            state["backbone_out"].update(self.model.backbone.forward_text(["visual"], device=self.device))
            state["geometric_prompt"] = self.model._get_dummy_prompt()
            xyxy = torch.as_tensor(boxes, device=self.device)
            scale = torch.tensor([image.width, image.height, image.width, image.height],
                                 device=self.device)
            normalized = xyxy / scale
            cxcywh = torch.stack((
                (normalized[:, 0] + normalized[:, 2]) / 2,
                (normalized[:, 1] + normalized[:, 3]) / 2,
                normalized[:, 2] - normalized[:, 0],
                normalized[:, 3] - normalized[:, 1],
            ), dim=-1)
            state["geometric_prompt"].append_boxes(
                cxcywh[:, None, :], torch.ones((len(boxes), 1), device=self.device, dtype=torch.bool),
            )
            state = self.processor._forward_grounding(state)
            detected_boxes = state["boxes"].float().cpu().numpy()
            scores = state["scores"].float().cpu().numpy()
        if detected_boxes.shape != (len(scores), 4):
            raise ValueError("SAM3 returned inconsistent detections.")
        detections = []
        for box, score in zip(detected_boxes, scores, strict=True):
            x1, y1, x2, y2 = map(float, box)
            score = float(score)
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score)) or not 0 <= score <= 1:
                raise ValueError("SAM3 returned an invalid detection.")
            clipped = [max(0., x1), max(0., y1), min(float(image.width), x2),
                       min(float(image.height), y2)]
            if score >= confidence and clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                detections.append({"box": clipped, "confidence": score})
        return detections

    def count(self, references: list[dict], image_paths: list[str], output_dir: str,
              confidence: float = .35) -> list[dict]:
        from .inference import _draw_detections, _read_image

        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        if not image_paths or len(references) != len(image_paths):
            raise ValueError("SAM3 needs one set of reference boxes for each image to count.")
        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        results = []
        for reference, image_path in zip(references, image_paths, strict=True):
            started = time.perf_counter()
            if (not isinstance(reference, dict) or
                    not isinstance(reference.get("path"), str) or
                    not Path(reference["path"]).is_absolute() or
                    Path(reference["path"]).resolve() != Path(image_path).resolve()):
                raise ValueError("Each SAM3 reference path must match its target image.")
            image = _read_image(image_path)
            boxes = _sam3_reference_boxes(reference.get("boxes"), *image.size)
            detections = self._predict(image, boxes, confidence)
            detections.sort(key=lambda det: (det["box"][1], det["box"][0]))
            for index, detection in enumerate(detections, 1):
                detection["index"] = index
            destination = directory / f"{uuid4().hex}.jpg"
            _draw_detections(image, detections, destination)
            result = {
                "count": len(detections), "detections": detections,
                "annotated_path": str(destination), "width": image.width, "height": image.height,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            warnings = []
            if len(detections) >= 190:
                warnings.append("SAM3 reached its detection query limit; the count may be low.")
            if max(image.size) > 1008:
                warnings.append("SAM3 resized this image to 1008 pixels for inference; review small objects.")
            if not detections:
                warnings.append("No matches detected. Check the selected boxes or lower confidence.")
            if warnings:
                result["warning"] = " ".join(warnings)
            results.append(result)
        return results

    def close(self):
        self.processor = None
        self.model = None
