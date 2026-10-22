"""Stage a tune-only experiment with FastSAM-masked hardware references.

Run in the project GPU image after the baseline benchmark has finished:
    python scripts/prepare_reference_backgrounds.py

This produces background/manifest.pending.json, selection.json, and category
contact sheets. Inspect every mask before using the pending manifest. Source
target files and their ground truth are never opened by this script.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "test-results/reliability-benchmark/data/manifest.json"
OUTPUT = ROOT / "test-results/reliability-benchmark/background"
NEUTRAL = {"gray": (192, 192, 192), "white": (255, 255, 255)}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def box_iou(left, right) -> float:
    overlap = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1]))
    a = (left[2] - left[0]) * (left[3] - left[1])
    b = (right[2] - right[0]) * (right[3] - right[1])
    return overlap / (a + b - overlap)


def _binary_masks(result, size):
    if result.masks is None or result.boxes is None:
        raise ValueError("FastSAM returned no instance masks")
    masks = result.masks.data.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy().tolist()
    scores = result.boxes.conf.detach().cpu().numpy().tolist()
    if len(masks) != len(boxes) or len(boxes) != len(scores):
        raise ValueError("FastSAM masks and boxes have different lengths")
    prepared = []
    for mask in masks:
        binary = mask > 0.5
        if binary.shape != (size[1], size[0]):
            binary = np.asarray(Image.fromarray(binary.astype("uint8") * 255).resize(size, Image.Resampling.NEAREST)) > 0
        prepared.append(binary)
    return prepared, boxes, scores


def _proposal_score(mask, mask_area, mask_box, proposal_box, quality, annotated_box):
    height, width = mask.shape
    x1, y1, x2, y2 = annotated_box
    left, top, right, bottom = max(0, math.floor(x1)), max(0, math.floor(y1)), min(width, math.ceil(x2)), min(height, math.ceil(y2))
    annotation_area = (right - left) * (bottom - top)
    intersection = int(mask[top:bottom, left:right].sum())
    if not mask_area or annotation_area <= 0:
        return None
    mask_iou = intersection / (annotation_area + mask_area - intersection)
    inside = intersection / mask_area
    bbox_iou = box_iou(mask_box, annotated_box)
    return {"mask_iou": round(mask_iou, 4), "bbox_iou": round(bbox_iou, 4),
            "mask_inside_reference": round(inside, 4), "mask_area_pixels": mask_area,
            "reference_coverage": round(intersection / annotation_area, 4),
            "mask_box": mask_box, "proposal_box": [round(float(value), 2) for value in proposal_box],
            "proposal_score": round(float(quality), 4)}


def select_masks(result, annotated_boxes, size):
    """Choose distinct whole-object masks by maximum summed mask/box IoU."""
    masks, proposal_boxes, scores = _binary_masks(result, size)
    if not masks:
        raise ValueError("FastSAM found no object proposals")
    candidates = []
    best_seen = [[] for _ in annotated_boxes]
    for proposal_index, (mask, box, quality) in enumerate(zip(masks, proposal_boxes, scores, strict=True)):
        ys, xs = np.where(mask)
        mask_area = len(xs)
        mask_box = ([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                    if mask_area else None)
        by_reference = []
        for reference_index, annotated in enumerate(annotated_boxes):
            score = _proposal_score(mask, mask_area, mask_box, box, quality, annotated)
            if score is None:
                by_reference.append(None)
                continue
            score["proposal_index"] = proposal_index
            best_seen[reference_index].append(score)
            # A mask can trace a narrow screw or a washer with a hole; a box fill
            # ratio alone would reject those valid shapes.
            eligible = (score["mask_iou"] >= .15 and score["bbox_iou"] >= .35
                        and score["mask_inside_reference"] >= .75)
            by_reference.append(score if eligible else None)
        candidates.append(by_reference)
    # Each proposal can explain at most one annotated exemplar.
    states = {0: (0.0, [])}
    for proposal_index, choices in enumerate(candidates):
        next_states = dict(states)
        for used, (total, picks) in states.items():
            for reference_index, score in enumerate(choices):
                flag = 1 << reference_index
                if used & flag or score is None:
                    continue
                key = used | flag
                value = total + score["mask_iou"]
                if key not in next_states or value > next_states[key][0]:
                    next_states[key] = (value, picks + [(reference_index, proposal_index)])
        states = next_states
    full = (1 << len(annotated_boxes)) - 1
    if full not in states:
        top = [[{"proposal_index": entry["proposal_index"], "mask_iou": entry["mask_iou"],
                 "bbox_iou": entry["bbox_iou"], "mask_inside_reference": entry["mask_inside_reference"]}
                for entry in sorted(entries, key=lambda item: -item["mask_iou"])[:3]] for entries in best_seen]
        raise ValueError(f"No distinct complete masks passed quality gates; top proposals: {top}")
    selections = []
    for reference_index, proposal_index in sorted(states[full][1]):
        selections.append({"reference_index": reference_index,
                           "annotated_box": annotated_boxes[reference_index],
                           **candidates[proposal_index][reference_index]})
    selected_mask = np.zeros((size[1], size[0]), dtype=bool)
    for selection in selections:
        selected_mask |= masks[selection["proposal_index"]]
    return selected_mask, selections, len(masks)


def compose(image, mask, color):
    """Use opaque RGB; the app's RGB conversion does not preserve alpha."""
    background = Image.new("RGB", image.size, color)
    background.paste(image, (0, 0), Image.fromarray(mask.astype("uint8") * 255))
    return background


def _preview(image, boxes, mask, outputs):
    original = image.copy()
    draw = ImageDraw.Draw(original)
    for index, box in enumerate(boxes, 1):
        draw.rectangle(box, outline="#ffd43b", width=3)
        draw.text((box[0], box[1]), str(index), fill="#ffd43b")
    tint = Image.new("RGB", image.size, "#13c7ae")
    overlay = Image.blend(image, tint, .45)
    masked = Image.composite(overlay, image, Image.fromarray(mask.astype("uint8") * 255))
    return [original, masked, outputs["gray"], outputs["white"]]


def contact_sheet(rows, destination):
    cell_width, cell_height = 340, 300
    headings = ["Original + boxes", "Selected mask", "Gray", "White"]
    sheet = Image.new("RGB", (4 * cell_width, (len(rows) + 1) * cell_height), "#eef2f0")
    draw = ImageDraw.Draw(sheet)
    for column, heading in enumerate(headings):
        draw.text((column * cell_width + 10, 12), heading, fill="#17313b")
    for row, (name, images) in enumerate(rows, 1):
        draw.text((10, row * cell_height + 5), name, fill="#17313b")
        for column, image in enumerate(images):
            thumb = image.copy()
            thumb.thumbnail((cell_width - 20, cell_height - 42))
            sheet.paste(thumb, (column * cell_width + (cell_width - thumb.width) // 2,
                                row * cell_height + 30 + (cell_height - 42 - thumb.height) // 2))
    sheet.save(destination, format="PNG")


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def prepare(source_path: Path, output_dir: Path, *, device="0", model_path=None):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from app.models import FASTSAM_FILENAME, model_dir, validate_checkpoint
    from ultralytics import FastSAM

    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not source.get("references") or not source.get("targets"):
        raise ValueError("Source manifest needs references and targets")
    model_path = Path(model_path) if model_path else model_dir() / FASTSAM_FILENAME
    validate_checkpoint(model_path, FASTSAM_FILENAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "references").mkdir(exist_ok=True)
    pending_path = output_dir / "manifest.pending.json"
    pending_path.unlink(missing_ok=True)
    model = FastSAM(str(model_path))
    selections = {"source_manifest": str(source_path), "source_sha256": sha256(source_path),
                  "fastsam_checkpoint_sha256": sha256(model_path), "created_at": datetime.now(timezone.utc).isoformat(),
                  "status": "needs_visual_audit", "references": [], "failures": []}
    variants = {name: {} for name in NEUTRAL}
    original = {}
    sheets = {}
    for category, references in source["references"].items():
        original[category] = []
        sheets[category] = []
        for index, reference in enumerate(references, 1):
            image_path = (source_path.parent / reference["path"]).resolve()
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            digest = sha256(image_path)
            if reference.get("source_sha256") and reference["source_sha256"] != digest:
                raise ValueError(f"Reference bytes changed: {image_path}")
            boxes = reference["boxes"]
            original[category].append({**reference, "path": _relative(image_path, output_dir), "source_sha256": digest})
            entry = {"category": category, "reference_index": index, "original_path": str(image_path),
                     "original_sha256": digest, "size": list(image.size), "annotated_boxes": boxes}
            try:
                results = model.predict(image, device=device, imgsz=1024, conf=.25, iou=.9,
                                        retina_masks=True, verbose=False, save=False)
                if len(results) != 1:
                    raise ValueError("FastSAM returned the wrong result count")
                mask, chosen, proposals = select_masks(results[0], boxes, image.size)
                entry.update({"status": "selected_pending_visual_audit", "proposal_count": proposals,
                              "selections": chosen, "union_mask_pixels": int(mask.sum())})
                outputs = {}
                for name, color in NEUTRAL.items():
                    output = output_dir / "references" / f"{category}-{index}-{name}.png"
                    outputs[name] = compose(image, mask, color)
                    outputs[name].save(output, format="PNG")
                    variants[name].setdefault(category, []).append({
                        **reference, "path": _relative(output, output_dir), "source_sha256": sha256(output),
                    })
                previews = _preview(image, boxes, mask, outputs)
                preview_path = output_dir / "references" / f"{category}-{index}-mask-preview.png"
                previews[1].save(preview_path, format="PNG")
                entry["mask_preview_path"] = _relative(preview_path, output_dir)
                sheets[category].append((image_path.name, previews))
            except Exception as exc:
                entry.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                selections["failures"].append({"category": category, "reference_index": index,
                                               "error": entry["error"]})
            selections["references"].append(entry)
            (output_dir / "selection.json").write_text(json.dumps(selections, indent=2) + "\n", encoding="utf-8")
            print(f"{category} reference {index}: {entry['status']}", flush=True)
        if sheets[category]:
            contact_sheet(sheets[category], output_dir / f"contact-{category}.png")
    if selections["failures"]:
        print(f"No comparison manifest written: {len(selections['failures'])} reference masks failed quality gates.", flush=True)
        return 1
    manifest = {key: value for key, value in source.items() if key not in ("targets", "references", "reference_variants")}
    manifest["title"] = source.get("title", "Hardware counting benchmark") + " — reference background, tune only"
    manifest["targets"] = [{**target, "path": _relative((source_path.parent / target["path"]).resolve(), output_dir)}
                           for target in source["targets"] if target["split"] == "tune"]
    manifest["references"] = original
    manifest["reference_variants"] = variants
    manifest["preprocessing"] = {"status": "needs_visual_audit", "selection_path": "selection.json",
                                 "source_manifest_sha256": selections["source_sha256"],
                                 "method": "FastSAM instance masks, full-resolution RGB composites on gray or white"}
    pending_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {pending_path}; visually inspect all contact sheets before benchmark use.", flush=True)
    return 0


def self_check():
    image = Image.new("RGB", (10, 10), "#9a251c")
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    mask[4:6, 4:6] = False
    gray = compose(image, mask, NEUTRAL["gray"])
    assert gray.getpixel((3, 3)) == (154, 37, 28)
    assert gray.getpixel((5, 5)) == NEUTRAL["gray"]  # A washer hole remains background.
    assert gray.getpixel((0, 0)) == NEUTRAL["gray"]
    assert box_iou([0, 0, 10, 10], [5, 5, 15, 15]) == 25 / 175
    class Tensor:
        def __init__(self, value): self.value = np.asarray(value)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value

    class Prediction:
        pass

    result = Prediction()
    result.masks = Prediction()
    result.boxes = Prediction()
    left = np.zeros((10, 10), dtype="float32")
    right = left.copy()
    left[1:4, 1:4] = 1
    right[6:9, 6:9] = 1
    result.masks.data = Tensor([left, right])
    result.boxes.xyxy = Tensor([[1, 1, 4, 4], [6, 6, 9, 9]])
    result.boxes.conf = Tensor([.9, .8])
    selected, entries, count = select_masks(result, [[1, 1, 4, 4], [6, 6, 9, 9]], (10, 10))
    assert count == 2 and [entry["proposal_index"] for entry in entries] == [0, 1]
    assert int(selected.sum()) == 18
    print("Compositing and geometry self-check passed.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="0", help="CUDA device for FastSAM; run after benchmark GPU work")
    parser.add_argument("--model-path", type=Path, help="Optional pinned FastSAM-s checkpoint")
    parser.add_argument("--self-check", action="store_true", help="Check compositing without loading the model")
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    return prepare(args.manifest, args.output_dir, device=args.device, model_path=args.model_path)


if __name__ == "__main__":
    raise SystemExit(main())
