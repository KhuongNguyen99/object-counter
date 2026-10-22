"""Object-level model disagreement diagnostics, never a claim of accuracy."""

import math


def compare_detections(primary, secondary, primary_model, secondary_model,
                       primary_confidence, secondary_confidence):
    """Maximum one-to-one geometric association; crowded alternatives remain flagged."""
    def geometry(detection):
        box = detection["box"]
        if (not isinstance(box, (list, tuple)) or len(box) != 4 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in box) or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("Invalid box in model comparison")
        x1, y1, x2, y2 = box
        return box, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

    a, b = [geometry(d) for d in primary], [geometry(d) for d in secondary]
    # Spatial cells avoid comparing every pair in large sparse photos.
    cell = 128
    buckets = {}
    for j, (_, cx, cy, _, _) in enumerate(b):
        buckets.setdefault((math.floor(cx / cell), math.floor(cy / cell)), []).append(j)
    edges = []
    reverse = [[] for _ in b]
    for i, (box, cx, cy, width, height) in enumerate(a):
        candidates = []
        # IoU >= .5 implies that the other center lies inside this expanded box.
        for y in range(math.floor((box[1] - height / 2) / cell),
                       math.floor((box[3] + height / 2) / cell) + 1):
            for x in range(math.floor((box[0] - width / 2) / cell),
                           math.floor((box[2] + width / 2) / cell) + 1):
                for j in buckets.get((x, y), []):
                    other, ox, oy, ow, oh = b[j]
                    overlap = max(0., min(box[2], other[2]) - max(box[0], other[0])) * max(
                        0., min(box[3], other[3]) - max(box[1], other[1]))
                    iou = overlap / (width * height + ow * oh - overlap)
                    centered = (abs(cx - ox) <= .2 * min(width, ow) and
                                abs(cy - oy) <= .2 * min(height, oh) and
                                min(width, ow) / max(width, ow) >= .5 and
                                min(height, oh) / max(height, oh) >= .5)
                    if iou >= .5 or centered:
                        candidates.append((iou, j))
                        reverse[j].append(i)
        edges.append([j for _, j in sorted(candidates, key=lambda item: (-item[0], item[1]))])

    # Iterative augmenting paths avoid recursion depth failures on crowded scenes.
    owner, assigned = {}, {}
    for start in range(len(a)):
        queue, seen_left, seen_right, previous = [start], {start}, set(), {}
        endpoint = None
        for i in queue:
            for j in edges[i]:
                if j in seen_right:
                    continue
                seen_right.add(j)
                previous[j] = i
                if j not in owner:
                    endpoint = j
                    break
                if owner[j] not in seen_left:
                    seen_left.add(owner[j])
                    queue.append(owner[j])
            if endpoint is not None:
                break
        while endpoint is not None:
            i = previous[endpoint]
            old = assigned.get(i)
            owner[endpoint], assigned[i] = i, endpoint
            endpoint = old

    primary_only = [i for i in range(len(a)) if i not in assigned]
    secondary_only = [j for j in range(len(b)) if j not in owner]
    # All vertices touching a non-unique correspondence need review.
    ambiguous_a = {i for i, options in enumerate(edges) if len(options) > 1}
    ambiguous_b = {j for j, options in enumerate(reverse) if len(options) > 1}
    ambiguous_a.update(i for j in ambiguous_b for i in reverse[j])
    ambiguous_b.update(j for i in ambiguous_a for j in edges[i])
    return {
        "status": "complete", "primary_model": primary_model, "secondary_model": secondary_model,
        "primary_confidence": primary_confidence, "secondary_confidence": secondary_confidence,
        "primary_count": len(a), "secondary_count": len(b), "matched": len(assigned),
        "pairs": [{"primary_index": i + 1, "secondary_index": j + 1}
                  for i, j in sorted(assigned.items())],
        "primary_only_indices": [i + 1 for i in primary_only],
        "secondary_only_indices": [j + 1 for j in secondary_only],
        "primary_only": [primary[i] for i in primary_only],
        "secondary_only": [secondary[j] for j in secondary_only],
        "secondary_detections": secondary,
        "ambiguous_primary_indices": [i + 1 for i in sorted(ambiguous_a)],
        "ambiguous_secondary_indices": [j + 1 for j in sorted(ambiguous_b)],
    }
