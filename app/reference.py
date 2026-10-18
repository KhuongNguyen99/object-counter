"""Text-guided, editable reference boxes from the bundled CountGD++ detector."""

import math
import time

class ReferenceModelUnavailable(RuntimeError):
    pass


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _select_boxes(boxes, qualities, width, height):
    candidates = []
    for box, quality in zip(boxes, qualities, strict=True):
        if (len(box) != 4 or not all(math.isfinite(value) for value in (*box, quality))
                or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height
                or _area(box) < 16 or _area(box) >= .9 * width * height):
            continue
        # Complete exemplars avoid table masks and objects cut by the photo edge.
        if box[0] <= 2 or box[1] <= 2 or box[2] >= width - 2 or box[3] >= height - 2:
            continue
        candidates.append((list(box), float(quality)))
    kept = []
    for box, quality in sorted(candidates, key=lambda item: (-_area(item[0]), -item[1], item[0])):
        area = _area(box)
        for outer, _ in kept:
            overlap = max(0, min(box[2], outer[2]) - max(box[0], outer[0])) * max(
                0, min(box[3], outer[3]) - max(box[1], outer[1]))
            # Prefer the whole object over its nested cap, hole, or other parts.
            if overlap / area >= .9 or overlap / (area + _area(outer) - overlap) >= .85:
                break
        else:
            kept.append((box, quality))
    return sorted(kept, key=lambda item: (-item[1], item[0]))


def propose_boxes(model, image_path, label):
    from .inference import _read_image

    started = time.perf_counter()
    image = _read_image(image_path)
    width, height = image.size
    predictions, qualities = model._predict(image, image, [], .23, label)
    xyxy = []
    for box, quality in zip(predictions, qualities, strict=True):
        if len(box) != 4:
            raise ValueError("The reference detector returned an invalid box")
        cx, cy, bw, bh = map(float, box)
        if (not all(math.isfinite(value) for value in (cx, cy, bw, bh, quality))
                or not 0 <= cx <= 1 or not 0 <= cy <= 1 or bw <= 0 or bh <= 0
                or not 0 <= quality <= 1):
            raise ValueError("The reference detector returned an invalid box")
        xyxy.append([max(0, (cx - bw / 2) * width), max(0, (cy - bh / 2) * height),
                     min(width, (cx + bw / 2) * width), min(height, (cy + bh / 2) * height)])
    candidates = _select_boxes(xyxy, qualities, width, height)
    boxes = []
    for box, _quality in candidates[:32]:
        margin = max(2, .02 * max(box[2] - box[0], box[3] - box[1]))
        boxes.append([max(0, math.floor(box[0] - margin)), max(0, math.floor(box[1] - margin)),
                      min(width, math.ceil(box[2] + margin)), min(height, math.ceil(box[3] + margin))])
    boxes.sort(key=lambda box: (box[1], box[0]))
    warning = None
    if not boxes:
        warning = "No matching object found. Refine the description or draw a box manually."
    elif len(candidates) > 32:
        warning = "Showing the 32 strongest boxes. Remove unwanted examples."
    return {"model": "countgd-plusplus", "width": width, "height": height,
            "boxes": boxes, "warning": warning,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}
