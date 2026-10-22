import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../app/static/app.js", import.meta.url), "utf8").replace(
  /\}\)\(\);\s*$/,
  "globalThis.testApp = { state, render, openReview, saveReview, finishCount, reloadReview };})();",
);
const referenceEditorSource = readFileSync(new URL("../app/static/reference-review.js", import.meta.url), "utf8");

class Element {
  constructor(tag = "div") {
    this.tag = tag;
    this.children = [];
    this.listeners = {};
    this.dataset = {};
    this.style = {};
    this.className = "";
    this.classList = { toggle() {}, add() {}, remove() {} };
    this.value = "";
    this.disabled = false;
    this.hidden = false;
    this.clientWidth = 800;
    this.clientHeight = 600;
    this.scrollLeft = 0;
    this.scrollTop = 0;
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  add(child) { this.append(child); }
  setAttribute(name, value) { this[name] = value; }
  getAttribute(name) { return this[name]; }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  showModal() { this.open = true; }
  close() { this.open = false; this.listeners.close?.(); }
  getBoundingClientRect() { return this.rect || { left: 0, top: 0, width: 100, height: 60 }; }
  setPointerCapture() {}
  querySelector(selector) {
    const matches = (element) => selector === "img" ? element.tag === "img" :
      selector === '[aria-current="true"]' && element["aria-current"] === "true";
    const find = (element) => {
      for (const child of element.children) {
        if (matches(child)) return child;
        const nested = find(child);
        if (nested) return nested;
      }
      return null;
    };
    return find(this);
  }
  focus() {}
  reset() {}
}

const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
const settle = () => new Promise((resolve) => setImmediate(resolve));
const session = () => ({
  id: "session-1", label: "nuts", model: "crosscheck-countgd-yoloe26x", status: "active",
  requires_review: true, total_count: 1, image_count: 1, created_at: "2026-10-21T00:00:00Z",
  batches: [{ id: "batch-1", status: "complete", total_count: 1, results: [{
    id: "image-1", filename: "nuts.jpg", original_url: "/original.jpg", annotated_url: "/annotated.jpg",
    model_count: 1, model_detections: [{ index: 1, box: [10, 10, 30, 30], confidence: .8 }],
    count: 1, detections: [{ index: 1, box: [10, 10, 30, 30], confidence: .8 }],
    width: 100, height: 60, review_revision: 0, reviewed: false,
    verification: { status: "complete", primary_count: 1, secondary_count: 2,
      primary_only_indices: [], secondary_only_indices: [2],
      secondary_detections: [{ index: 1, box: [10, 10, 30, 30], confidence: .7 },
        { index: 2, box: [50, 10, 70, 30], confidence: .4 }],
      secondary_only: [{ index: 2, box: [50, 10, 70, 30], confidence: .4 }] },
  }] }],
});

function browser(fetch, confirm = () => true) {
  const elements = new Map();
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    createElement: (tag) => new Element(tag),
    createElementNS: (_, tag) => new Element(tag),
    querySelectorAll: () => [],
  };
  document.getElementById("reference-crop").append(new Element("img"));
  const context = {
    document,
    fetch: (path, options) => path === "/api/models" ? Promise.resolve(response({
      models: [{ id: "crosscheck-countgd-yoloe26x", name: "CountGD++ + YOLOE", available: true }],
      default_model: "crosscheck-countgd-yoloe26x", device: "cpu",
    })) : fetch(path, options),
    localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    URL: { revokeObjectURL() {} },
    Option: class extends Element { constructor(label, value) { super("option"); this.textContent = label; this.value = value; } },
    setTimeout: () => 1, clearTimeout() {}, window: { scrollTo() {}, confirm },
  };
  vm.createContext(context);
  vm.runInContext(referenceEditorSource, context, { filename: "reference-review.js" });
  vm.runInContext(source, context, { filename: "app.js" });
  return { ...context.testApp, get: document.getElementById };
}

test("review requires explicit acceptance, supports undo and add/remove, then saves draft and confirmation", async () => {
  let saved = session();
  const writes = [];
  const app = browser(async (path, options) => {
    if (path.endsWith("/review")) {
      const body = JSON.parse(options.body);
      writes.push(body);
      saved = structuredClone(saved);
      const result = saved.batches[0].results[0];
      result.detections = body.boxes.map((box, index) => ({ index: index + 1, box, confidence: null }));
      result.count = body.boxes.length;
      result.review_revision++;
      result.reviewed = body.reviewed;
      saved.total_count = result.count;
      return response(saved);
    }
    return response(saved);
  });
  await settle();
  app.state.session = saved;
  app.render();
  assert.equal(app.get("finish-button").disabled, true);
  assert.equal(app.get("finish-review-hint").hidden, false);
  app.openReview("image-1");
  assert.equal(app.state.review.boxes.length, 1, "second-model boxes stay as suggestions");
  assert.equal(app.get("review-overlay").children[0].className, "");
  app.get("review-overlay").children[0].listeners.pointerdown({ stopPropagation() {} });
  assert.equal(app.state.review.boxes.length, 2, "click explicitly accepts suggestion");
  app.get("review-undo").listeners.click();
  assert.equal(app.state.review.boxes.length, 1);

  const overlay = app.get("review-overlay");
  overlay.rect = { left: 0, top: 0, width: 100, height: 60 };
  overlay.listeners.pointerdown({ button: 0, pointerId: 1, clientX: 50, clientY: 10, preventDefault() {} });
  overlay.listeners.pointermove({ pointerId: 1, clientX: 70, clientY: 30 });
  overlay.listeners.pointerup({ pointerId: 1, clientX: 70, clientY: 30 });
  assert.equal(app.state.review.boxes.length, 2, "drawn box is added");
  app.get("review-remove").listeners.click();
  const counted = app.get("review-overlay").children.filter((element) => element["aria-label"]?.startsWith("Counted object"));
  counted[0].listeners.pointerdown({ stopPropagation() {} });
  assert.equal(app.state.review.boxes.length, 1, "remove mode removes the chosen box");
  await app.get("review-save").listeners.click();
  assert.equal(writes[0].revision, 0);
  assert.equal(writes[0].reviewed, false);
  assert.deepEqual(writes[0].boxes, [[50, 10, 70, 30]]);
  assert.equal(app.state.review.revision, 1);
  assert.equal(app.get("finish-button").disabled, true);
  await app.get("review-confirm").listeners.click();
  assert.equal(writes[1].revision, 1);
  assert.equal(writes[1].reviewed, true);
  assert.equal(app.state.review, null);
  assert.equal(app.get("finish-button").disabled, false);
  assert.equal(app.get("finish-review-hint").hidden, true);
});

test("all second-model boxes are view-only and never alter saved count", async () => {
  const saved = session();
  const originalSecondary = structuredClone(saved.batches[0].results[0].verification.secondary_detections);
  const writes = [];
  const app = browser(async (path, options) => {
    if (path.endsWith("/review")) {
      writes.push(JSON.parse(options.body));
      return response(saved);
    }
    return response(saved);
  });
  await settle();
  app.state.session = saved;
  app.render();
  app.openReview("image-1");
  const overlay = app.get("review-overlay");
  const allBoxes = () => overlay.children.filter((item) => item["aria-label"]?.startsWith("Other suggestion"));
  assert.equal(allBoxes().length, 0);
  assert.equal(app.get("review-secondary-row").hidden, false);
  const toggle = app.get("review-show-secondary");
  toggle.checked = true;
  toggle.listeners.change({ target: toggle });
  assert.equal(allBoxes().length, 2, "matched and unmatched second-model boxes are visible");
  assert.equal(allBoxes()[0]["pointer-events"], "none");
  assert.equal(app.state.review.boxes.length, 1);
  await app.saveReview(false);
  assert.deepEqual(writes[0].boxes, [[10, 10, 30, 30]]);
  assert.deepEqual(saved.batches[0].results[0].verification.secondary_detections, originalSecondary);
  toggle.checked = false;
  toggle.listeners.change({ target: toggle });
  assert.equal(allBoxes().length, 0);
});

test("failed verification disables full second-model overlay", async () => {
  const saved = session();
  saved.batches[0].results[0].verification = { status: "failed", primary_count: 1, error: "Unavailable" };
  const app = browser(async () => response(saved));
  await settle();
  app.state.session = saved;
  app.render();
  app.openReview("image-1");
  assert.equal(app.get("review-secondary-row").hidden, false);
  assert.equal(app.get("review-show-secondary").disabled, true);
});

test("stale review refetches session but preserves local edits until explicit reload", async () => {
  let saved = session();
  let reads = 0;
  const app = browser(async (path, options) => {
    if (path.endsWith("/review")) return response({ detail: "Stale review" }, 409);
    reads++;
    return response(saved);
  });
  await settle();
  app.state.session = saved;
  app.render();
  app.openReview("image-1");
  app.get("review-overlay").children[0].listeners.pointerdown({ stopPropagation() {} });
  assert.equal(app.state.review.boxes.length, 2);
  saved = structuredClone(saved);
  saved.batches[0].results[0].review_revision = 3;
  saved.batches[0].results[0].detections = [{ index: 1, box: [5, 5, 25, 25], confidence: null }];
  await app.saveReview(true);
  assert.equal(reads, 1);
  assert.equal(app.state.review.stale, true);
  assert.equal(app.state.review.boxes.length, 2, "unsaved edits must remain visible");
  assert.equal(app.state.review.revision, 0, "stale revision must not be silently advanced");
  assert.equal(app.get("review-confirm").disabled, true);
  assert.equal(app.get("review-reload").hidden, false);
  await app.reloadReview();
  assert.equal(app.state.review.stale, false);
  assert.equal(app.state.review.revision, 3);
  assert.deepEqual(JSON.parse(JSON.stringify(app.state.review.boxes)), [[5, 5, 25, 25]]);
});

test("finished session opens box review read-only", async () => {
  const app = browser(async () => response(session()));
  await settle();
  const finished = session();
  finished.status = "finished";
  finished.batches[0].results[0].reviewed = true;
  app.state.session = finished;
  app.render();
  app.openReview("image-1");
  assert.equal(app.state.review.readOnly, true);
  assert.equal(app.get("review-save").disabled, true);
  assert.equal(app.get("review-confirm").disabled, true);
  assert.equal(app.get("review-zoom-in").disabled, false);
  assert.equal(app.get("review-pan").disabled, false);
  assert.equal(app.state.review.mode, "pan");
  const toggle = app.get("review-show-secondary");
  toggle.checked = true;
  toggle.listeners.change({ target: toggle });
  assert.equal(app.get("review-overlay").children.filter((item) => item["aria-label"]?.startsWith("Other suggestion")).length, 2);
});

test("successful stale reload followed by closing review unlocks the main controls", async () => {
  const latest = session();
  latest.batches[0].results[0].reviewed = true;
  latest.batches[0].results[0].review_revision = 1;
  const app = browser(async (path) => path.endsWith("/review")
    ? response({ detail: "Stale review" }, 409) : response(latest));
  await settle();
  app.state.session = session();
  app.render();
  app.openReview("image-1");
  await app.saveReview(false);
  assert.equal(app.state.review.stale, true);
  await app.reloadReview();
  assert.equal(app.state.review.stale, false);
  assert.equal(app.state.review.reviewed, true);
  app.get("close-review").listeners.click();
  assert.equal(app.state.review, null);
  assert.equal(app.state.busy, false);
  for (const id of ["finish-button", "target-gallery", "target-camera", "next-gallery", "next-camera"]) {
    assert.equal(app.get(id).disabled, false, `${id} should unlock after reloading`);
  }
  const descendants = (element) => [element, ...element.children.flatMap(descendants)];
  const reviewButton = descendants(app.get("batch-list")).find((element) => element.textContent === "Review photo");
  assert.ok(reviewButton);
  assert.equal(reviewButton.disabled, false, "the photo can be reviewed again");
});

test("review clips out-of-bounds model boxes and guards unsaved edits on close", async () => {
  let canDiscard = false;
  const app = browser(async () => response(session()), () => canDiscard);
  await settle();
  const saved = session();
  saved.batches[0].results[0].detections[0].box = [-2, 10, 30, 30];
  app.state.session = saved;
  app.render();
  app.openReview("image-1");
  assert.deepEqual(JSON.parse(JSON.stringify(app.state.review.boxes)), [[0, 10, 30, 30]]);
  assert.match(app.get("review-message").textContent, /clipped/);
  app.get("review-overlay").children[0].listeners.pointerdown({ stopPropagation() {} });
  app.get("close-review").listeners.click();
  assert.equal(app.get("review-dialog").open, true, "unsaved edits stay open when discard is declined");
  canDiscard = true;
  app.get("close-review").listeners.click();
  assert.equal(app.get("review-dialog").open, false);
  assert.equal(app.state.review, null);
});

test("reload failure preserves stale local boxes for a later retry", async () => {
  let failRead = false;
  const app = browser(async () => failRead ? response({ detail: "Offline" }, 503) : response(session()));
  await settle();
  app.state.session = session();
  app.render();
  app.openReview("image-1");
  app.get("review-overlay").children[0].listeners.pointerdown({ stopPropagation() {} });
  app.state.review.stale = true;
  failRead = true;
  await app.reloadReview();
  assert.equal(app.state.review.stale, true);
  assert.equal(app.state.review.boxes.length, 2);
  assert.match(app.get("review-message").textContent, /still here/);
});
