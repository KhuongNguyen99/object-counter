"""Editable reference-box candidates from prompt-free FastSAM segmentation."""

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


def propose_boxes(model, image_path, device):
    from .inference import _read_image

    started = time.perf_counter()
    image = _read_image(image_path)
    width, height = image.size
    results = model.predict(image, device=device, imgsz=1024, conf=.25, iou=.9,
                            retina_masks=False, verbose=False, save=False)
    try:
        if len(results) != 1:
            raise ValueError
        xyxy = results[0].boxes.xyxy.tolist()
        qualities = results[0].boxes.conf.tolist()
        if len(xyxy) != len(qualities):
            raise ValueError
        checked_boxes, checked_qualities = [], []
        for box, quality in zip(xyxy, qualities, strict=True):
            if len(box) != 4:
                raise ValueError
            box = list(map(float, box))
            quality = float(quality)
            if (not all(math.isfinite(value) for value in (*box, quality))
                    or not 0 <= box[0] < box[2] <= width
                    or not 0 <= box[1] < box[3] <= height
                    or not 0 <= quality <= 1):
                raise ValueError
            checked_boxes.append(box)
            checked_qualities.append(quality)
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("FastSAM returned an invalid box") from exc
    candidates = _select_boxes(checked_boxes, checked_qualities, width, height)
    boxes = []
    for box, _quality in candidates:
        margin = max(2, .02 * max(box[2] - box[0], box[3] - box[1]))
        boxes.append([max(0, math.floor(box[0] - margin)), max(0, math.floor(box[1] - margin)),
                      min(width, math.ceil(box[2] + margin)), min(height, math.ceil(box[3] + margin))])
    boxes.sort(key=lambda box: (box[1], box[0]))
    warning = None
    if not boxes:
        warning = "No object candidates found. Draw a box manually."
    return {"model": "fastsam-s", "width": width, "height": height,
            "boxes": boxes, "warning": warning,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}
