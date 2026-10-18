"""Native-pixel crops and conservative association of detections across crops.

There is no resizing here. Overlap gives small objects context near crop edges.
Ownership cores break representative ties; they never discard a detection merely
because the model missed it in its owning crop. Association uses boxes, not image
appearance, so intrinsically ambiguous/incorrect model predictions can still
overcount or undercount. Same-crop detections are never merged with each other.
"""

from dataclasses import dataclass
import math
from collections.abc import Iterable


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Tile:
    index: int
    box: tuple[int, int, int, int]
    core: Box

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


def make_tiles(width: int, height: int, tile_size: int, overlap: float = .2,
               max_tiles: int = 256) -> list[Tile]:
    """Cover a frame with crops no larger than ``tile_size`` on either axis.

    The overlap is a fraction of the crop side (default 20%). Starts are spread
    evenly so the final crop reaches the frame edge without a thin sliver crop.
    The pass limit is checked before constructing either axis or the tile list;
    exceeding it raises rather than silently shrinking the source image.
    """
    for name, value in (("width", width), ("height", height),
                        ("tile_size", tile_size), ("max_tiles", max_tiles)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not math.isfinite(overlap) or not 0 <= overlap <= .5:
        raise ValueError("Overlap must be between 0 and 0.5.")
    stride = max(1, math.floor(tile_size * (1 - overlap)))
    columns = 1 + math.ceil(max(0, width - tile_size) / stride)
    rows = 1 + math.ceil(max(0, height - tile_size) / stride)
    if columns * rows > max_tiles:
        raise ValueError(
            f"Native-resolution counting needs {columns * rows} tiles; the limit is "
            f"{max_tiles}. Crop the image into smaller regions and count them separately."
        )

    def axis(length: int, count: int) -> list[tuple[int, int, float, float]]:
        side = min(length, tile_size)
        starts = [round(i * (length - side) / (count - 1)) for i in range(count)] if count > 1 else [0]
        return [
            (start, start + side,
             (starts[i - 1] + side + start) / 2 if i else 0.,
             (start + side + starts[i + 1]) / 2 if i + 1 < count else float(length))
            for i, start in enumerate(starts)
        ]

    horizontal, vertical = axis(width, columns), axis(height, rows)
    return [
        Tile(row * columns + column, (left, top, right, bottom), (cx1, cy1, cx2, cy2))
        for row, (top, bottom, cy1, cy2) in enumerate(vertical)
        for column, (left, right, cx1, cx2) in enumerate(horizontal)
    ]


def _intersection(a: Box, b: Box) -> Box:
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def _area(box: Box) -> float:
    return max(0., box[2] - box[0]) * max(0., box[3] - box[1])


def _iou(a: Box, b: Box) -> float:
    intersection = _area(_intersection(a, b))
    return intersection / (_area(a) + _area(b) - intersection) if intersection else 0.


@dataclass
class _Detection:
    tile: Tile
    box: Box
    clipped: tuple[bool, bool, bool, bool]
    owned: bool
    record: dict


def _match(a: _Detection, b: _Detection, common: Box) -> float:
    """Return association strength, or zero for incompatible predictions."""
    overlap = _iou(a.box, b.box)
    if not any(a.clipped) and not any(b.clipped):
        return 2 + overlap if overlap >= .55 else 0.
    # A clipped fragment may be mostly contained in a complete prediction, or
    # two long-object fragments may overlap only in a narrow strip. Compare the
    # part actually visible to BOTH models, not IoS over an arbitrary small box.
    visible_a, visible_b = _intersection(a.box, common), _intersection(b.box, common)
    agreement = _iou(visible_a, visible_b)
    # Good overlap is already strong evidence despite crop-induced changes in
    # the regressed sides. Keep its ranking below exact shared-view matches.
    if overlap >= .55:
        return 2 + agreement
    intersection = _area(_intersection(visible_a, visible_b))
    smaller = min(_area(visible_a), _area(visible_b))
    containment = intersection / smaller if smaller else 0.
    complementary = any(
        (a.tile.box[axis] < b.tile.box[axis] and a.clipped[axis + 2] and b.clipped[axis])
        or (b.tile.box[axis] < a.tile.box[axis] and b.clipped[axis + 2] and a.clipped[axis])
        for axis in (0, 1)
    )
    if complementary and agreement >= .6 and containment >= .9:
        # Upper/lower (or left/right) fragments can have different silhouettes:
        # a foot below the shared strip can widen the lower person's box. Strong
        # shared overlap still establishes continuity across opposite crop edges.
        return 2 + agreement
    if agreement < .5 or (agreement < .65 and containment < .85):
        return 0.
    tolerances = (
        max(2., .35 * max(a.box[2] - a.box[0], b.box[2] - b.box[0])),
        max(2., .35 * max(a.box[3] - a.box[1], b.box[3] - b.box[1])),
    )
    # Uncropped sides must agree. This prevents a large box from swallowing a
    # nearby small object merely because their visible fragments are nested.
    # Scale tolerance to the more complete object, not the thin edge fragment.
    for side in range(4):
        if not a.clipped[side] and not b.clipped[side]:
            if abs(a.box[side] - b.box[side]) > tolerances[side % 2]:
                return 0.
    # Exact partial views must outrank an ordinary high-IoU pair belonging to
    # different overlapping objects. Rank by agreement in the shared view.
    return 2 + agreement


def _stronger_view(strong: _Detection, weak: _Detection) -> bool:
    """A substantially broader observation can explain a clipped fragment."""
    weak_area = _area(weak.box)
    return (
        any(weak.clipped)
        and sum(strong.clipped) <= sum(weak.clipped)
        and _area(strong.box) >= 1.5 * weak_area
        and _area(_intersection(strong.box, weak.box)) >= .85 * weak_area
    )


def _cells(box: Box, side: int):
    """Spatial buckets, with right/bottom treated as exclusive boundaries."""
    if not _area(box):
        return
    for y in range(math.floor(box[1] / side), math.ceil(box[3] / side)):
        for x in range(math.floor(box[0] / side), math.ceil(box[2] / side)):
            yield x, y


def merge_tile_detections(predictions: Iterable[tuple[Tile, list[dict]]],
                          width: int, height: int, *, include_members: bool = False) -> list[dict]:
    """Map tile-local ``box``/``confidence`` records to sorted frame detections.

    Ordinary duplicate boxes need IoU >= .55. Partial crop-edge boxes must agree
    in the shared visible region and on their untruncated sides. Associations
    are strongest-first and one-to-one per source tile; conflicting jointly
    visible pairs prevent transitive chains swallowing neighboring objects.
    A clipped fragment may defer to a substantially broader, containing view
    in its own group when that broader view agrees across the two groups.
    Complete boxes are preferred over clipped boxes. If every view is clipped,
    matching fragments' extents are joined to represent a long/crossing object.

    Invalid/nonfinite or zero-area predictions are discarded, and boxes are
    clipped to their source crop before offsetting. Other record fields survive
    from the chosen representative; ``index`` is rebuilt in reading order.
    ``include_members`` additionally returns source records for mask union.
    """
    detections: list[_Detection] = []
    by_tile: dict[int, list[int]] = {}
    tiles: dict[int, Tile] = {}
    for tile, records in predictions:
        if tile.index in tiles:
            raise ValueError("Each tile index must occur only once.")
        left, top, right, bottom = tile.box
        if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
            raise ValueError("Tiles must have positive area and stay inside the frame.")
        tiles[tile.index] = tile
        by_tile[tile.index] = []
        for record in records:
            try:
                local = tuple(map(float, record["box"]))
                score = float(record["confidence"])
            except (KeyError, TypeError, ValueError):
                continue
            if len(local) != 4 or not all(math.isfinite(v) for v in (*local, score)):
                continue
            local = _intersection(local, (0, 0, tile.width, tile.height))
            if not _area(local):
                continue
            box = (local[0] + left, local[1] + top, local[2] + left, local[3] + top)
            clipped = (left > 0 and local[0] <= 2., top > 0 and local[1] <= 2.,
                       right < width and local[2] >= tile.width - 2.,
                       bottom < height and local[3] >= tile.height - 2.)
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            owned = tile.core[0] <= cx < tile.core[2] and tile.core[1] <= cy < tile.core[3]
            by_tile[tile.index].append(len(detections))
            detections.append(_Detection(tile, box, clipped, owned, {**record, "confidence": score}))

    if not detections:
        return []
    # Buckets bound work to overlapping tile pairs, then nearby detections in
    # their shared strip; avoid a quadratic all-frame NMS on dense 30 MP images.
    tile_buckets: dict[tuple[int, int], list[int]] = {}
    tile_pairs: set[tuple[int, int]] = set()
    tile_side = max(max(tile.width, tile.height) for tile in tiles.values())
    for tile_id, tile in sorted(tiles.items()):
        for cell in _cells(tile.box, tile_side):
            for other_id in tile_buckets.get(cell, []):
                tile_pairs.add((other_id, tile_id))
            tile_buckets.setdefault(cell, []).append(tile_id)

    edges: list[tuple[float, int, int]] = []
    for first, second in sorted(tile_pairs):
        common = _intersection(tiles[first].box, tiles[second].box)
        if not _area(common):
            continue
        buckets: dict[tuple[int, int], list[int]] = {}
        for other in by_tile[second]:
            for cell in _cells(_intersection(detections[other].box, common), 64):
                buckets.setdefault(cell, []).append(other)
        for index in by_tile[first]:
            candidates: set[int] = set()
            for cell in _cells(_intersection(detections[index].box, common), 64):
                candidates.update(buckets.get(cell, []))
            for other in sorted(candidates):
                strength = _match(detections[index], detections[other], common)
                if strength:
                    edges.append((-strength, index, other))

    parents = list(range(len(detections)))
    members = {i: [i] for i in parents}
    source_tiles = {i: {detection.tile.index} for i, detection in enumerate(detections)}

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def compatible(first: int, second: int) -> bool:
        if source_tiles[first] & source_tiles[second]:
            return False
        support = []
        conflicts = []
        for i in members[first]:
            for j in members[second]:
                a, b = detections[i], detections[j]
                common = _intersection(a.tile.box, b.tile.box)
                if _area(_intersection(a.box, common)) and _area(_intersection(b.box, common)):
                    (support if _match(a, b, common) else conflicts).append((i, j))
        for i, j in conflicts:
            # A narrow corner fragment may regress to a different silhouette.
            # It must not veto agreement from a broader view in its OWN group.
            # Evidence must cross these two groups; complete-view conflicts and
            # chains without such independent support still block association.
            if not any(
                (a == i or _stronger_view(detections[a], detections[i]))
                and (b == j or _stronger_view(detections[b], detections[j]))
                for a, b in support
            ):
                return False
        return True

    for _, index, other in sorted(edges):
        first, second = root(index), root(other)
        if first == second or not compatible(first, second):
            continue
        parents[second] = first
        members[first].extend(members.pop(second))
        source_tiles[first].update(source_tiles.pop(second))

    result = []
    for group in members.values():
        chosen = min(group, key=lambda i: (sum(detections[i].clipped), not detections[i].owned,
                                           -detections[i].record["confidence"],
                                           detections[i].tile.index, i))
        detection = detections[chosen]
        box = detection.box
        if len(group) > 1 and all(any(detections[i].clipped) for i in group):
            box = (min(detections[i].box[0] for i in group), min(detections[i].box[1] for i in group),
                   max(detections[i].box[2] for i in group), max(detections[i].box[3] for i in group))
        result.append({**detection.record, "box": list(box)})
        if include_members:
            result[-1]["members"] = [detections[i].record for i in group]
    result.sort(key=lambda record: (record["box"][1], record["box"][0]))
    for index, record in enumerate(result, 1):
        record["index"] = index
    return result
