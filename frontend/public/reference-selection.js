/* Filter implausible proposal sizes, then spread initial references across the image. */
globalThis.ReferenceSelection = (() => {
  const median = (values) => {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  };

  const unitRandom = (random) => {
    const value = random();
    return Number.isFinite(value) ? Math.max(0, Math.min(1 - Number.EPSILON, value)) : 0.5;
  };

  function prepare(boxes, width, height, random = Math.random) {
    if (!Array.isArray(boxes) || !Number.isFinite(width) || !Number.isFinite(height) ||
        width <= 0 || height <= 0 || typeof random !== "function") return { candidates: [], boxes: [] };

    const seen = new Set();
    const valid = [];
    for (const box of boxes) {
      if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite) ||
          !(0 <= box[0] && box[0] < box[2] && box[2] <= width &&
            0 <= box[1] && box[1] < box[3] && box[3] <= height)) continue;
      const key = box.join(",");
      if (seen.has(key)) continue;
      seen.add(key);
      valid.push([...box]);
    }
    if (!valid.length) return { candidates: [], boxes: [] };

    const logAreas = valid.map(([x1, y1, x2, y2]) => Math.log(x2 - x1) + Math.log(y2 - y1));
    const typical = median(logAreas);
    const mad = median(logAreas.map((value) => Math.abs(value - typical)));
    // A factor-1.5 floor preserves size variation when the raw log-area MAD is zero.
    // With only four proposals, require a factor-ten difference before discarding one.
    const tolerance = valid.length < 5 ? (valid.length === 4 ? Math.log(10) : Infinity) :
      Math.max(Math.log(1.5), 2.5 * mad);
    // Include exact cutoff sizes despite floating-point log/subtraction rounding.
    const filtered = valid.filter((_, index) => Math.abs(logAreas[index] - typical) <= tolerance + 1e-12);
    const desired = Math.min(10, filtered.length);
    const selected = new Set();
    if (desired) selected.add(Math.floor(unitRandom(random) * filtered.length));

    const centers = filtered.map(([x1, y1, x2, y2]) => [
      (x1 / width + x2 / width) / 2, (y1 / height + y2 / height) / 2,
    ]);
    while (selected.size < desired) {
      let best = -1;
      let bestDistance = -1;
      let ties = 0;
      for (let i = 0; i < centers.length; i++) {
        if (selected.has(i)) continue;
        const distance = Math.min(...[...selected].map((chosen) => {
          const dx = centers[i][0] - centers[chosen][0];
          const dy = centers[i][1] - centers[chosen][1];
          return dx * dx + dy * dy;
        }));
        if (distance > bestDistance + 1e-12) {
          best = i;
          bestDistance = distance;
          ties = 1;
        } else if (Math.abs(distance - bestDistance) <= 1e-12 && unitRandom(random) < 1 / ++ties) {
          best = i;
        }
      }
      selected.add(best);
    }

    return {
      candidates: filtered.filter((_, index) => !selected.has(index)),
      boxes: filtered.filter((_, index) => selected.has(index)),
    };
  }

  return { prepare };
})();
