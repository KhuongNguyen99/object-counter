import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

// Exercise the actual browser handlers with deferred network responses, without a browser dependency.
const source = readFileSync(new URL("../app/static/app.js", import.meta.url), "utf8").replace(
  /\}\)\(\);\s*$/,
  "globalThis.testApp = { state, render, addReferences, uploadBatch, finishCount };})();",
);

class Element {
  constructor(tag = "div") {
    this.tag = tag;
    this.children = [];
    this.listeners = {};
    this.dataset = {};
    this.style = {};
    this.className = "";
    this.classList = { toggle() {}, add() {} };
    this.value = "";
    this.disabled = false;
    this.hidden = false;
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  add(child) { this.append(child); }
  setAttribute(name, value) { this[name] = value; }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  reset() {}
  focus() { this.focused = true; }
  getContext() { return { drawImage() {} }; }
  toBlob(done) { done({ size: 512 }); }
}

const catalog = {
  models: [{ id: "yoloe-26s-seg", name: "YOLOE-26 Small", available: true }],
  default_model: "yoloe-26s-seg", device: "cpu",
};
const session = (status = "active") => ({
  id: "saved-session", label: "Apples", model: "yoloe-26s-seg", status,
  created_at: "2026-10-17T12:00:00Z", total_count: 0, image_count: 0,
  references: [], batches: [],
});
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
const deferred = () => {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
};
const settle = () => new Promise((resolve) => setImmediate(resolve));

function browser(fetch, savedId = null, modelCatalog = catalog) {
  const elements = new Map();
  const timers = new Map();
  let nextTimer = 1;
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    createElement: (tag) => new Element(tag),
    createElementNS: (_, tag) => new Element(tag),
    querySelectorAll(selector) {
      if (selector !== ".queue-item button") return [];
      const found = new Set();
      const visit = (element) => {
        if (element.className?.split(" ").includes("queue-item")) {
          for (const child of element.children) if (child.tag === "button") found.add(child);
        }
        for (const child of element.children || []) visit(child);
      };
      for (const element of elements.values()) visit(element);
      return [...found];
    },
  };
  const context = {
    document,
    fetch: (path, options) => path === "/api/models" ? Promise.resolve(response(modelCatalog)) : fetch(path, options),
    localStorage: { getItem: () => savedId, setItem() {}, removeItem() {} },
    URL: { createObjectURL: (() => { let next = 0; return () => `blob:reference-${++next}`; })(), revokeObjectURL() {} },
    createImageBitmap: async (file) => ({ width: file.width, height: file.height, close() {} }),
    File: class {
      constructor(parts, name, options) { this.name = name; this.type = options.type; this.size = parts[0].size; }
    },
    FormData: class {
      constructor() { this.items = []; }
      append(...entry) { this.items.push(entry); }
    },
    Option: class extends Element {
      constructor(label, value) { super("option"); this.textContent = label; this.value = value; }
    },
    setTimeout: (callback) => { const id = nextTimer++; timers.set(id, callback); return id; },
    clearTimeout: (id) => timers.delete(id),
    window: { scrollTo() {} },
  };
  vm.createContext(context);
  vm.runInContext(source, context, { filename: "app.js" });
  return { ...context.testApp, get: document.getElementById, timers };
}

test("upload locks thumbnail removal until the submitted batch is accepted", async () => {
  const accepted = deferred();
  let submitted;
  const app = browser(async (path, options) => {
    if (path.endsWith("/batches")) {
      submitted = options.body.items;
      return accepted.promise;
    }
    return response(session());
  });
  await settle();
  app.state.session = session();
  app.state.targets.push({ file: { name: "apples.jpg" }, url: "blob:apples" });
  app.render();
  const remove = app.get("next-queue").children[0].children[1];
  const upload = app.uploadBatch();
  assert.equal(app.state.busy, true);
  assert.equal(remove.disabled, true, "the visible remove control must lock immediately");
  remove.listeners.click();
  assert.equal(app.state.targets.length, 1, "the handler must also reject removal during upload");
  assert.equal(submitted.length, 1);
  accepted.resolve(response({ id: "batch-1", status: "queued" }, 202));
  await upload;
  assert.equal(app.state.targets.length, 0);
  assert.equal(app.state.busy, false);
});

test("a lost finish response reconciles the committed session and exposes New count", async () => {
  let saved = session();
  let reads = 0;
  const app = browser(async (_path, options) => {
    if (options?.method === "POST") {
      saved = session("finished");
      throw new TypeError("Connection lost after server committed");
    }
    reads++;
    return response(saved);
  });
  await settle();
  app.state.session = session();
  app.render();
  await app.finishCount();
  assert.equal(reads, 1, "an uncertain finish response must trigger a session read");
  assert.equal(app.state.session.status, "finished");
  assert.equal(app.get("new-count-button").hidden, false);
  assert.equal(app.get("next-batch-panel").hidden, true);
  assert.equal(app.state.busy, false);
});

test("restoring a saved session keeps setup locked and resumes polling a pending batch", async () => {
  const restored = deferred();
  const app = browser(() => restored.promise, "saved-session");
  await settle();
  assert.equal(app.state.restoring, true);
  assert.equal(app.get("setup-view").hidden, true);
  assert.equal(app.get("session-view").hidden, true);
  assert.equal(app.get("start-button").disabled, true);
  const saved = session();
  saved.batches.push({ id: "batch-1", status: "running", created_at: saved.created_at, results: [] });
  restored.resolve(response(saved));
  await settle();
  assert.equal(app.state.restoring, false);
  assert.equal(app.state.session.id, "saved-session");
  assert.equal(app.get("setup-view").hidden, true);
  assert.equal(app.get("session-view").hidden, false);
  assert.equal(app.get("upload-button").disabled, true);
  assert.equal(app.get("finish-button").disabled, true);
  assert.equal(app.timers.size, 1, "pending results must resume automatic polling");
});

test("model selection applies each model's default confidence", async () => {
  const models = {
    models: [
      { id: "countgd-plusplus", name: "CountGD++ (experimental)", family: "CountGD++", available: true, default_confidence: 0.23 },
      { id: "yoloe-26s-seg", name: "YOLOE-26 Small", available: true, default_confidence: 0.25 },
    ],
    default_model: "countgd-plusplus", device: "cpu",
  };
  const app = browser(async () => response(session()), null, models);
  await settle();
  assert.equal(app.get("model-select").value, "countgd-plusplus");
  assert.equal(app.get("confidence").value, "0.23");
  assert.equal(app.get("confidence-value").textContent, "0.23");

  const select = app.get("model-select");
  select.value = "yoloe-26s-seg";
  select.listeners.change({ target: select });
  assert.equal(app.get("confidence").value, "0.25");
  assert.equal(app.get("confidence-value").textContent, "0.25");
});

test("zero detections are flagged for review, including results saved before warnings existed", async () => {
  const app = browser(async () => response(session()));
  await settle();
  const saved = session("finished");
  saved.image_count = 1;
  saved.batches.push({ id: "batch-1", status: "complete", total_count: 0,
    created_at: saved.created_at, results: [{ filename: "rings.jpg", count: 0,
      width: 721, height: 1280, detections: [], warning: null }] });
  app.state.session = saved;
  app.render();
  assert.equal(app.get("results-state").textContent, "Review needed");
  assert.equal(app.get("metric-status").textContent, "No matches · review needed");
  const content = (element) => [element.textContent || "", ...element.children.map(content)].join(" ");
  const results = content(app.get("batch-list"));
  assert.match(results, /0 detected/);
  assert.match(results, /use a model trained for this object/);
});

test("tiled results show a compact tile count and keep details collapsed", async () => {
  const app = browser(async () => response(session()));
  await settle();
  const saved = session("finished");
  saved.total_count = 2;
  saved.image_count = 2;
  saved.batches.push({ id: "batch-1", status: "complete", total_count: 2,
    created_at: saved.created_at, results: [
      { filename: "large.jpg", count: 1, width: 3000, height: 2000,
        elapsed_ms: 2000, detections: [], processing: {
          tiles: 6, tile_size: 1024, duplicates_removed: 2,
        } },
      { filename: "old.jpg", count: 1, width: 600, height: 400,
        elapsed_ms: 500, detections: [], processing: null },
    ] });
  app.state.session = saved;
  app.render();

  const [tiled, historical] = app.get("batch-list").children[0].children.slice(1);
  const tiledBody = tiled.children[1];
  assert.equal(tiledBody.children[1].textContent, "6 tiles");
  const details = tiledBody.children[2];
  assert.equal(details.tag, "details");
  assert.match(details.children[1].textContent, /tile size 1024 px/);
  assert.match(details.children[1].textContent, /2 duplicates removed/);

  const oldBody = historical.children[1];
  assert.equal(oldBody.children[1].tag, "details", "historical results have no tile label");
});

const referenceFile = (name = "sample.png") => ({ name, size: 100, type: "image/png", width: 100, height: 80 });

test("a reference added before its description stays editable and can be boxed manually or retried", async () => {
  let requests = 0;
  const app = browser(async (path, options) => {
    assert.equal(path, "/api/reference-boxes");
    assert.deepEqual(options.body.items.map(([field, value]) => [field, value.name || value]),
      [["image", "sample.jpg"], ["label", "red-capped bolt"]]);
    requests++;
    return response({ width: 100, height: 80, boxes: [[1, 2, 20, 30]], model: "countgd-plusplus", warning: null });
  });
  await settle();
  const adding = app.addReferences([referenceFile()]);
  assert.equal(app.get("object-label").disabled, false, "Object stays editable without a description");
  await adding;
  assert.equal(requests, 0, "empty descriptions do not start detection");
  assert.equal(app.state.refs.length, 1);
  assert.equal(app.state.refs[0].boxes.length, 0);
  assert.equal(app.get("notice").textContent, "Describe the object, then use Auto-detect.");
  app.get("box-x1").valueAsNumber = 1;
  app.get("box-y1").valueAsNumber = 2;
  app.get("box-x2").valueAsNumber = 20;
  app.get("box-y2").valueAsNumber = 30;
  app.get("coordinate-form").listeners.submit({ preventDefault() {} });
  assert.equal(app.state.refs[0].boxes.length, 1, "manual boxing still works without a description");
  app.get("undo-box").listeners.click();
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(requests, 0);
  assert.equal(app.get("object-label").focused, true, "missing Object is focused");
  assert.equal(app.get("notice").textContent, "Describe the object, then use Auto-detect.");
  app.get("object-label").value = "  red-capped bolt  ";
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(requests, 1);
  assert.equal(app.state.refs[0].boxes.length, 1);
});

test("new references use the normalized image for automatic boxes and allow individual rejection", async () => {
  let uploads = 0;
  const app = browser(async (path, options) => {
    assert.equal(path, "/api/reference-boxes");
    assert.equal(options.method, "POST");
    assert.deepEqual(options.body.items.map(([field, value]) => [field, value.name || value, value.type || null]),
      [["image", "sample.jpg", "image/jpeg"], ["label", "red-capped bolt", null]]);
    uploads++;
    return response({ width: 100, height: 80, boxes: [[1, 2, 20, 30], [40, 10, 70, 60]], model: "countgd-plusplus", elapsed_ms: 120, warning: null });
  });
  await settle();
  app.get("object-label").value = "  red-capped bolt  ";
  await app.addReferences([referenceFile()]);
  assert.equal(uploads, 1);
  assert.equal(app.state.refs[0].boxes.length, 2);
  assert.equal(app.get("box-count").textContent, "2 boxes");
  assert.equal(app.get("auto-detect-boxes").disabled, true, "existing boxes cannot be replaced implicitly");
  const remove = app.get("box-list").children[0];
  assert.equal(remove["aria-label"], "Remove box 1 from reference 1");
  remove.listeners.click();
  assert.equal(app.state.refs[0].boxes.length, 1);
  assert.equal(app.state.refs[0].boxes[0][0], 40);
  app.get("undo-box").listeners.click();
  assert.equal(app.state.refs[0].boxes.length, 0);
  assert.equal(app.get("auto-detect-boxes").disabled, false);
});

test("detection locks editing while pending and failure keeps the reference for manual boxing", async () => {
  const pending = deferred();
  const app = browser(async (path) => path === "/api/reference-boxes" ? pending.promise : response(session()));
  await settle();
  app.get("object-label").value = "red-capped bolt";
  const adding = app.addReferences([referenceFile()]);
  await settle();
  assert.equal(app.state.busy, true);
  assert.equal(app.get("start-button").disabled, true);
  assert.equal(app.get("remove-reference").disabled, true);
  assert.equal(app.get("auto-detect-boxes").disabled, true);
  assert.equal(app.get("auto-detect-boxes").textContent, "Detecting…");
  assert.match(app.get("notice").textContent, /Finding objects/);
  app.get("box-x1").valueAsNumber = 1;
  app.get("box-y1").valueAsNumber = 2;
  app.get("box-x2").valueAsNumber = 20;
  app.get("box-y2").valueAsNumber = 30;
  app.get("coordinate-form").listeners.submit({ preventDefault() {} });
  assert.equal(app.state.refs[0].boxes.length, 0);
  pending.resolve(response({ detail: "Detector unavailable" }, 503));
  await adding;
  assert.equal(app.state.busy, false);
  assert.equal(app.state.refs.length, 1);
  assert.equal(app.get("auto-detect-boxes").disabled, false);
  assert.match(app.get("notice").textContent, /Draw boxes manually or retry/);
  app.get("coordinate-form").listeners.submit({ preventDefault() {} });
  assert.equal(app.state.refs[0].boxes.length, 1);
  assert.equal(app.get("auto-detect-boxes").disabled, true);
});

test("invalid detection coordinates are rejected, and an empty reference can retry", async () => {
  let calls = 0;
  const app = browser(async () => {
    calls++;
    return calls === 1
      ? response({ width: 99, height: 80, boxes: [[4, 5, 40, 50]], warning: null })
      : calls === 2
        ? response({ width: 100, height: 80, boxes: [[0, 0, Infinity, 20]], warning: null })
        : response({ width: 100, height: 80, boxes: [[4, 5, 40, 50]], warning: null });
  });
  await settle();
  app.get("object-label").value = "red-capped bolt";
  await app.addReferences([referenceFile()]);
  assert.equal(app.state.refs[0].boxes.length, 0);
  assert.match(app.get("notice").textContent, /did not match/);
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(app.state.refs[0].boxes.length, 0);
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(calls, 3);
  assert.equal(app.state.refs[0].boxes.length, 1);
});

test("suggestions stop at the 32-box total without blocking later manual review", async () => {
  let calls = 0;
  const app = browser(async () => {
    calls++;
    return response({ width: 100, height: 80, boxes: Array.from({ length: 33 }, () => [1, 2, 20, 30]), warning: null });
  });
  await settle();
  app.get("object-label").value = "red-capped bolt";
  await app.addReferences([referenceFile("first.png"), referenceFile("second.png")]);
  assert.equal(calls, 1, "detection stops once 32 boxes have been accepted");
  assert.equal(app.state.refs[0].boxes.length, 32);
  assert.equal(app.state.refs[1].boxes.length, 0);
  assert.equal(app.get("auto-detect-boxes").disabled, true);
  app.get("reference-thumbs").children[0].listeners.click();
  app.get("box-list").children[0].listeners.click();
  app.get("reference-thumbs").children[1].listeners.click();
  assert.equal(app.get("auto-detect-boxes").disabled, false);
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(calls, 2);
  assert.equal(app.state.refs[1].boxes.length, 1);
  assert.equal(app.state.refs.reduce((sum, ref) => sum + ref.boxes.length, 0), 32);
});
