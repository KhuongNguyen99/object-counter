import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../frontend/public/reference-selection.js", import.meta.url), "utf8");
const context = vm.createContext({});
vm.runInContext(source, context, { filename: "reference-selection.js" });
const { prepare } = context.ReferenceSelection;
const plain = (value) => JSON.parse(JSON.stringify(value));
const square = (x, y, side = 10) => [x, y, x + side, y + side];
const all = (result) => [...result.candidates, ...result.boxes];

test("removes obvious tiny and huge proposals when regular boxes give zero MAD", () => {
  const regular = Array.from({ length: 12 }, (_, i) => square(i * 30, 0));
  const input = [square(500, 10, 1), ...regular, square(600, 100, 100)];
  const result = prepare(input, 1000, 1000, () => 0.25);
  assert.equal(result.boxes.length, 10);
  assert.equal(result.candidates.length, 2);
  assert.deepEqual(plain(all(result).sort((a, b) => a[0] - b[0])), regular);
});

test("zero-MAD filtering keeps sizes within a factor of 1.5", () => {
  const regular = Array.from({ length: 6 }, (_, i) => square(i * 25, 0));
  const near = [[200, 0, 210, 15], [225, 0, 235, 7]]; // Areas 150 (the cutoff) and 70.
  const beyond = [[250, 0, 266, 10], [275, 0, 285, 6]]; // Areas 160 and 60.
  const result = prepare([...regular, ...near, ...beyond], 400, 100, () => 0);
  assert.deepEqual(plain(result.boxes), [...regular, ...near]);
  assert.deepEqual(plain(result.candidates), []);
});

test("an exact factor-1.5 size remains eligible despite log rounding", () => {
  const regular = Array.from({ length: 6 }, (_, i) => square(i * 20, 0, 8));
  const boundary = [140, 0, 148, 12]; // Area 96, exactly 1.5 times 64.
  const outside = [180, 0, 188, 13];
  const result = prepare([...regular, boundary, outside], 300, 100, () => 0);
  assert.deepEqual(plain(result.boxes), [...regular, boundary]);
  assert.deepEqual(plain(result.candidates), []);

  const taller = Array.from({ length: 6 }, (_, i) => [i * 20, 0, i * 20 + 8, 12]);
  const reciprocalBoundary = square(140, 0, 8); // Area 64, exactly 96 / 1.5.
  const smallerOutside = [180, 0, 188, 7];
  const reciprocal = prepare([...taller, reciprocalBoundary, smallerOutside], 300, 100, () => 0);
  assert.deepEqual(plain(reciprocal.boxes), [...taller, reciprocalBoundary]);
  assert.deepEqual(plain(reciprocal.candidates), []);
});

test("small samples keep the no-filter and factor-ten fallback behavior", () => {
  const regular = [square(0, 0), square(50, 0), square(100, 0)];
  const huge = square(200, 0, 90);
  assert.deepEqual(plain(prepare([regular[0], regular[1], huge], 400, 200, () => 0).boxes),
    [regular[0], regular[1], huge]);

  const withinTen = [200, 0, 231, 30]; // Area 930, versus 100 for regular boxes.
  const beyondTen = [200, 0, 235, 35]; // Area 1225.
  assert.deepEqual(plain(prepare([...regular, withinTen], 400, 200, () => 0).boxes),
    [...regular, withinTen]);
  assert.deepEqual(plain(prepare([...regular, beyondTen], 400, 200, () => 0).boxes),
    regular);
});

test("photo proposals discard four small outliers while retaining the largest", () => {
  // Captured FastSAM-s proposals; IDs are one-based positions in the saved coordinate list.
  const { width, height, boxes } = JSON.parse(readFileSync(
    new URL("./fixtures/photo_mad_proposals.json", import.meta.url), "utf8"));
  assert.equal(boxes.length, 32);
  const result = prepare(boxes, width, height, () => 0.5);
  const ids = new Map(boxes.map((box, index) => [box.join(","), index + 1]));
  const kept = all(result).map((box) => ids.get(box.join(","))).sort((a, b) => a - b);
  assert.deepEqual(kept, Array.from({ length: 32 }, (_, index) => index + 1)
    .filter((id) => ![1, 12, 20, 21].includes(id)));
  assert.equal(result.boxes.length, 10);
  assert.equal(result.candidates.length, 18);
  assert.ok(kept.includes(31)); // Area 12,321: the largest proposal is still selectable.
});

test("size filtering is scale invariant and preserves ordinary size variation", () => {
  const varied = [8, 10, 12, 14, 16, 18].map((side, i) => square(i * 40, 10, side));
  const input = [...varied, square(300, 10, 1), square(400, 10, 90)];
  const first = prepare(input, 600, 200, () => 0.3);
  const scale = 7;
  const scaled = prepare(input.map((box) => box.map((coordinate) => coordinate * scale)),
    600 * scale, 200 * scale, () => 0.3);
  assert.deepEqual(plain(all(first)), varied);
  assert.deepEqual(plain(scaled.boxes.map((box) => box.map((coordinate) => coordinate / scale))),
    plain(first.boxes));
  assert.deepEqual(plain(scaled.candidates.map((box) => box.map((coordinate) => coordinate / scale))),
    plain(first.candidates));
});

test("ten selections cover all occupied quadrants despite imbalanced proposal density", () => {
  const clustered = Array.from({ length: 24 }, (_, i) => square(20 + (i % 6) * 8, 20 + Math.floor(i / 6) * 8));
  const distant = [square(780, 20), square(850, 70), square(20, 780), square(70, 850),
    square(780, 780), square(850, 850), square(700, 700), square(700, 20)];
  const result = prepare([...clustered, ...distant], 1000, 1000, () => 0.1);
  assert.equal(result.boxes.length, 10);
  assert.equal(result.candidates.length, 22);
  const quadrants = new Set(result.boxes.map(([x1, y1, x2, y2]) =>
    `${(x1 + x2) / 2 >= 500},${(y1 + y2) / 2 >= 500}`));
  assert.equal(quadrants.size, 4);
});

test("selection never restores an outlier to reach ten; fewer valid proposals remain selectable", () => {
  const regular = Array.from({ length: 9 }, (_, i) => square(i * 20, 20));
  const result = prepare([...regular, square(300, 300, 200)], 600, 600, () => 0);
  assert.deepEqual(plain(result.boxes), regular);
  assert.deepEqual(plain(result.candidates), []);
  assert.equal(prepare(regular.slice(0, 3), 600, 600, () => 0).boxes.length, 3);
});

test("skips malformed and duplicate boxes, retains candidate order, and never mutates input", () => {
  const valid = Array.from({ length: 12 }, (_, i) => Object.freeze(square(i * 20, 10)));
  const input = Object.freeze([valid[0], null, [1, 2, 3], [0, 0, 0, 10],
    [0, 0, Infinity, 10], [-1, 0, 5, 5], [0, 0, 1001, 5], valid[0], ...valid.slice(1)]);
  const result = prepare(input, 1000, 100, () => 0.4);
  assert.equal(result.boxes.length, 10);
  assert.equal(result.candidates.length, 2);
  assert.equal(new Set(all(result).map((box) => box.join(","))).size, 12);
  assert.deepEqual(plain(all(result).sort((a, b) => a[0] - b[0])), plain(valid));
  const indices = result.candidates.map((box) => box[0]);
  assert.deepEqual(plain(indices), [...indices].sort((a, b) => a - b));
  assert.ok(all(result).every((box) => !input.includes(box)));
  assert.deepEqual(plain(input[0]), [0, 10, 10, 20]);
});

test("random injection is reproducible, while empty or invalid image data yields no boxes", () => {
  const input = Array.from({ length: 14 }, (_, i) => square((i % 7) * 60, Math.floor(i / 7) * 80));
  const seeded = () => {
    let state = 42;
    return () => ((state = (1664525 * state + 1013904223) >>> 0) / 2 ** 32);
  };
  assert.deepEqual(plain(prepare(input, 500, 200, seeded())),
    plain(prepare(input, 500, 200, seeded())));
  assert.deepEqual(plain(prepare([], 500, 200)), { candidates: [], boxes: [] });
  assert.deepEqual(plain(prepare([square(0, 0)], 0, 200)), { candidates: [], boxes: [] });
  assert.deepEqual(plain(prepare(null, 500, 200)), { candidates: [], boxes: [] });
});
