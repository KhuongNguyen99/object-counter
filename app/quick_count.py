"""Count FastSAM object proposals inside user-marked image areas."""

import json
import math
import tempfile
import time
from pathlib import Path

from PIL import Image

from .inference import _draw_detections, _read_image


MAX_SEARCH_AREAS = 32


def parse_search_areas(raw, dimensions):
    """Validate groups in upload order against normalized image dimensions."""
    try:
        groups = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Marked areas must be valid JSON") from exc
    if not isinstance(groups, list) or len(groups) != len(dimensions):
        raise ValueError("Mark at least one area on every uploaded photo")
    for group, (width, height) in zip(groups, dimensions):
        if not isinstance(group, list) or not 1 <= len(group) <= MAX_SEARCH_AREAS:
            raise ValueError("Mark 1–32 areas on each photo")
        for box in group:
            if not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box):
                raise ValueError("Each marked area needs four whole-number pixel coordinates")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("Marked areas must fit inside their photo")
            if x2 - x1 < 4 or y2 - y1 < 4:
                raise ValueError("Each marked area must be at least 4 pixels wide and tall")
    return groups


def _overlap(box, other):
    intersection = max(0, min(box[2], other[2]) - max(box[0], other[0])) * max(
        0, min(box[3], other[3]) - max(box[1], other[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    other_area = (other[2] - other[0]) * (other[3] - other[1])
    return intersection / (area + other_area - intersection)


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _filter_sizes(boxes):
    """Match the log-area median/MAD rule used by reference-selection.js."""
    if not boxes:
        return []
    log_areas = [math.log(box[2] - box[0]) + math.log(box[3] - box[1]) for box in boxes]
    typical = _median(log_areas)
    mad = _median([abs(value - typical) for value in log_areas])
    tolerance = (math.log(10) if len(boxes) == 4 else math.inf) if len(boxes) < 5 else max(
        math.log(1.5), 2.5 * mad)
    return [box for box, area in zip(boxes, log_areas)
            if abs(area - typical) <= tolerance + 1e-12]


def count_marked_areas(engine, image_path, search_areas, output_dir, label, validate_proposal):
    """Run the cached FastSAM helper on native-pixel crops and merge full-photo boxes."""
    started = time.perf_counter()
    image = _read_image(image_path)
    proposals = []
    warnings = []
    output_dir = Path(output_dir)
    with tempfile.TemporaryDirectory(prefix="quick-crops-", dir=output_dir) as temporary:
        for index, (x1, y1, x2, y2) in enumerate(search_areas):
            crop_path = Path(temporary) / f"area-{index}.png"
            image.crop((x1, y1, x2, y2)).save(crop_path, format="PNG")
            proposal = validate_proposal(
                engine.suggest_reference_boxes(str(crop_path), label), x2 - x1, y2 - y1)
            if not proposal["boxes"]:
                warnings.append("No objects found in one marked area.")
            elif proposal["warning"]:
                warnings.append("Automatic detection may have missed objects in one marked area.")
            for left, top, right, bottom in proposal["boxes"]:
                proposals.append([left + x1, top + y1, right + x1, bottom + y1])

    unique = []
    for box in proposals:
        if not any(_overlap(box, other) >= .85 for other in unique):
            unique.append(box)
    boxes = _filter_sizes(unique)
    if not boxes:
        warnings.append("No objects found in the marked areas.")
    detections = [{"index": index, "box": box, "confidence": None, "source": "quick"}
                  for index, box in enumerate(boxes, 1)]
    annotation = output_dir / f"annotation-{Path(image_path).stem}.jpg"
    _draw_detections(image, detections, annotation)
    return {"count": len(detections), "detections": detections,
            "annotated_path": str(annotation), "width": image.width, "height": image.height,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "warning": " ".join(dict.fromkeys(warnings)) or None}
