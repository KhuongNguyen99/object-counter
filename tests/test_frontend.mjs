import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

// Exercise the actual browser handlers with deferred network responses, without a browser dependency.
const source = readFileSync(new URL("../frontend/public/app.js", import.meta.url), "utf8").replace(
  /\}\)\(\);\s*$/,
  "globalThis.testApp = { state, render, addReferences, addTargets, uploadBatch, finishCount, referenceReview, apiUrl };})();",
);
const referenceEditorSource = readFileSync(new URL("../frontend/public/reference-review.js", import.meta.url), "utf8");
const selectionSource = readFileSync(new URL("../frontend/public/reference-selection.js", import.meta.url), "utf8");

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
    this.clientWidth = 100;
    this.clientHeight = 100;
    this.scrollLeft = 0;
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  add(child) { this.append(child); }
  setAttribute(name, value) { this[name] = value; }
  getAttribute(name) { return this[name]; }
  addEventListener(type, listener) { this.listeners[type] = listener; }
  showModal() { this.open = true; }
  close() { this.open = false; this.listeners.close?.(); }
  reset() {}
  focus() { this.focused = true; }
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
  getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; }
  setPointerCapture() {}
  getContext() { return { drawImage: (...args) => { this.drawCalls = [...(this.drawCalls || []), args]; } }; }
  toBlob(done, type, quality) { this.blobOptions = { type, quality }; done({ size: 512, width: this.width, height: this.height }); }
}

const catalog = {
  models: [{ id: "yoloe-26s-seg", name: "YOLOE-26 Small", available: true }],
  default_model: "yoloe-26s-seg", device: "cpu",
};
const confidenceCatalog = {
  models: [
    { id: "countgd-plusplus", name: "CountGD++ (experimental)", family: "CountGD++", available: true, default_confidence: 0.80 },
    { id: "yoloe-26s-seg", name: "YOLOE-26 Small", available: true, default_confidence: 0.25 },
  ],
  default_model: "countgd-plusplus", device: "cpu",
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

function browser(fetch, savedId = null, modelCatalog = catalog, storage = new Map(), apiBaseUrl = "") {
  const elements = new Map();
  const timers = new Map();
  const revoked = [];
  const decoded = [];
  const canvases = [];
  let nextTimer = 1;
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    createElement: (tag) => { const element = new Element(tag); if (tag === "canvas") canvases.push(element); return element; },
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
  document.getElementById("reference-crop").append(new Element("img"));
  const sessionKey = apiBaseUrl ? `countwell.sessionId.${encodeURIComponent(apiBaseUrl.replace(/\/+$/, ""))}` : "countwell.sessionId";
  class BrowserURL extends URL {}
  BrowserURL.createObjectURL = (() => { let next = 0; return () => `blob:reference-${++next}`; })();
  BrowserURL.revokeObjectURL = (url) => { revoked.push(url); };
  const context = {
    document,
    COUNTWELL_CONFIG: { apiBaseUrl },
    fetch: (path, options) => path.endsWith("/api/models") ? Promise.resolve(response(modelCatalog)) : fetch(path, options),
    localStorage: {
      getItem(key) { return key === sessionKey && savedId !== null ? savedId : storage.get(key) ?? null; },
      setItem(key, value) { storage.set(key, String(value)); },
      removeItem(key) { storage.delete(key); },
    },
    URL: BrowserURL,
    createImageBitmap: async (file) => { decoded.push(file); return { width: file.width, height: file.height, close() {} }; },
    File: class {
      constructor(parts, name, options) {
        this.name = name; this.type = options.type; this.size = parts[0].size;
        this.width = parts[0].width; this.height = parts[0].height;
      }
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
  vm.runInContext(selectionSource, context, { filename: "reference-selection.js" });
  vm.runInContext(referenceEditorSource, context, { filename: "reference-review.js" });
  vm.runInContext(source, context, { filename: "app.js" });
  return { ...context.testApp, get: document.getElementById, timers, revoked, decoded, canvases };
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

test("a configured backend prefixes requests, result photos, export, and resumed polling", async () => {
  const base = "https://count.example/service/";
  const prefix = "https://count.example/service";
  const storage = new Map([["countwell.sessionId", "other-backend-session"]]);
  const calls = [];
  const saved = session();
  saved.image_count = 1;
  saved.batches.push({ id: "batch-1", status: "complete", total_count: 1, results: [{
    id: "image-1", filename: "apple.jpg", count: 1, width: 100, height: 80,
    original_url: "/images/original.jpg", annotated_url: "/images/annotated.jpg", detections: [],
  }] });
  const app = browser(async (path, options) => {
    calls.push([path, options?.method || "GET"]);
    if (path.endsWith("/batches")) return response({ id: "batch-2" }, 202);
    if (path.endsWith("/finish")) return response({ ...saved, status: "finished" });
    return response(saved);
  }, null, catalog, storage, base);
  await settle();
  assert.equal(app.state.restoring, false, "a same-origin session belongs to a different backend");
  assert.equal(app.apiUrl("/api/models"), `${prefix}/api/models`);
  assert.equal(app.apiUrl("blob:upload"), "blob:upload");
  assert.equal(app.apiUrl("https://cdn.example/image.jpg"), "https://cdn.example/image.jpg");
  app.state.refs.push({ file: { name: "example.jpg" }, url: "blob:example", boxes: [[1, 2, 20, 30]], candidates: [] });
  app.state.targets.push({ file: { name: "apple.jpg" }, url: "blob:upload" });
  await app.get("start-button").listeners.click();
  assert.ok(calls.some(([url, method]) => url === `${prefix}/api/sessions` && method === "POST"));
  assert.equal(storage.get(`countwell.sessionId.${encodeURIComponent(prefix)}`), "saved-session");
  assert.equal(app.get("batch-list").querySelector("img").src, `${prefix}/images/annotated.jpg`);
  app.get("batch-list").children[0].children[1].children[0].listeners.click();
  assert.equal(app.get("dialog-image").src, `${prefix}/images/annotated.jpg`);
  assert.equal(app.get("export-link").href, `${prefix}/api/sessions/saved-session/export.csv`);
  app.state.targets.push({ file: { name: "another.jpg" }, url: "blob:upload" });
  app.render();
  assert.equal(app.get("next-queue").querySelector("img").src, "blob:upload");
  await app.uploadBatch();
  assert.ok(calls.some(([url, method]) => url === `${prefix}/api/sessions/saved-session/batches` && method === "POST"));
  assert.ok(calls.some(([url]) => url === `${prefix}/api/sessions/saved-session`), "polling reads the configured backend");
  await app.finishCount();
  assert.ok(calls.some(([url, method]) => url === `${prefix}/api/sessions/saved-session/finish` && method === "POST"));

  const restored = browser(async (path) => { calls.push([path, "RESTORE"]); return response(saved); },
    "saved-session", catalog, storage, base);
  await settle();
  assert.equal(restored.state.session.id, "saved-session");
  assert.ok(calls.some(([url, method]) => url === `${prefix}/api/sessions/saved-session` && method === "RESTORE"));
  assert.equal(restored.get("batch-list").querySelector("img").src, `${prefix}/images/annotated.jpg`);
});

test("an invalid backend address leaves counting unavailable with a plain-language message", async () => {
  let calls = 0;
  const app = browser(async () => { calls++; return response(session()); }, null, catalog, new Map(), "ftp://count.example");
  await settle();
  assert.equal(calls, 0);
  assert.equal(app.get("start-button").disabled, true);
  assert.match(app.get("notice").textContent, /app setup/);
});

test("model selection applies each model's default confidence", async () => {
  const app = browser(async () => response(session()), null, confidenceCatalog);
  await settle();
  assert.equal(app.get("model-select").value, "countgd-plusplus");
  assert.equal(app.get("confidence").value, "0.8");
  assert.equal(app.get("confidence-value").textContent, "0.80");

  const select = app.get("model-select");
  select.value = "yoloe-26s-seg";
  select.listeners.change({ target: select });
  assert.equal(app.get("confidence").value, "0.25");
  assert.equal(app.get("confidence-value").textContent, "0.25");
});

test("confidence input persists its raw value and restores its label in a fresh browser context", async () => {
  const storage = new Map();
  const first = browser(async () => response(session()), null, confidenceCatalog, storage);
  await settle();
  assert.equal(storage.size, 0, "loading a model default must not write a preference");

  const slider = first.get("confidence");
  slider.value = "0.42";
  slider.listeners.input({ target: slider });
  assert.equal(storage.get("countwell.confidence.countgd-plusplus"), "0.42");
  assert.equal(storage.has("countwell.sessionId"), false);

  const restored = browser(async () => response(session()), null, confidenceCatalog, storage);
  await settle();
  assert.equal(restored.get("confidence").value, "0.42");
  assert.equal(restored.get("confidence-value").textContent, "0.42");
});

test("model switching restores each model's own saved confidence", async () => {
  const storage = new Map();
  const app = browser(async () => response(session()), null, confidenceCatalog, storage);
  await settle();
  const select = app.get("model-select");
  const slider = app.get("confidence");

  slider.value = "0.37";
  slider.listeners.input({ target: slider });
  select.value = "yoloe-26s-seg";
  select.listeners.change({ target: select });
  assert.equal(slider.value, "0.25", "an unsaved model uses its own default");
  slider.value = "0.68";
  slider.listeners.input({ target: slider });
  select.value = "countgd-plusplus";
  await select.listeners.change({ target: select });
  assert.equal(slider.value, "0.37");
  assert.equal(app.get("confidence-value").textContent, "0.37");
  select.value = "yoloe-26s-seg";
  select.listeners.change({ target: select });
  assert.equal(slider.value, "0.68");
  assert.equal(app.get("confidence-value").textContent, "0.68");
  assert.deepEqual([...storage.entries()].sort(), [
    ["countwell.confidence.countgd-plusplus", "0.37"],
    ["countwell.confidence.yoloe-26s-seg", "0.68"],
  ]);
});

test("invalid saved confidence falls back to the selected model default", async () => {
  for (const stored of ["", "not-a-number", "NaN", "Infinity", "0", "0.009", "0.951"]) {
    const storage = new Map([["countwell.confidence.countgd-plusplus", stored]]);
    const app = browser(async () => response(session()), null, confidenceCatalog, storage);
    await settle();
    assert.equal(app.get("confidence").value, "0.8", `stored value ${JSON.stringify(stored)}`);
    assert.equal(app.get("confidence-value").textContent, "0.80", `stored value ${JSON.stringify(stored)}`);
    assert.equal(storage.get("countwell.confidence.countgd-plusplus"), stored, "fallback must not overwrite storage");
  }
  for (const stored of ["0.01", "0.95"]) {
    const storage = new Map([["countwell.confidence.countgd-plusplus", stored]]);
    const app = browser(async () => response(session()), null, confidenceCatalog, storage);
    await settle();
    assert.equal(app.get("confidence").value, stored);
    assert.equal(app.get("confidence-value").textContent, stored);
  }
  const storage = new Map([["countwell.confidence.yoloe-26s-seg", "bad"]]);
  const app = browser(async () => response(session()), null, catalog, storage);
  await settle();
  assert.equal(app.get("confidence").value, "0.25", "models without a declared default use 0.25");
});

test("blocked local storage leaves confidence controls usable", async () => {
  const storage = new Map();
  storage.get = () => { throw new Error("storage unavailable"); };
  storage.set = () => { throw new Error("storage unavailable"); };
  const app = browser(async () => response(session()), null, confidenceCatalog, storage);
  await settle();
  assert.equal(app.get("confidence").value, "0.8");
  const slider = app.get("confidence");
  slider.value = "0.51";
  assert.doesNotThrow(() => slider.listeners.input({ target: slider }));
  assert.equal(slider.value, "0.51");
  assert.equal(app.get("confidence-value").textContent, "0.51");
});

test("text-only counting hides references and omits existing reference uploads", async () => {
  const models = { models: [...catalog.models,
    { id: "sam3count", name: "SAM3Count", available: true, reference_mode: "none", default_confidence: .3 }],
    default_model: "sam3count" };
  let submitted;
  const app = browser(async (path, options) => {
    if (path === "/api/sessions") {
      submitted = options.body.items;
      return response({ ...session(), model: "sam3count" }, 201);
    }
    if (path.endsWith("/batches")) return response({ id: "batch-1" }, 202);
    return response({ ...session(), model: "sam3count" });
  }, null, models);
  await settle();
  assert.equal(app.get("reference-panel").hidden, true);
  assert.equal(app.get("text-prompt-note").hidden, false);
  assert.equal(app.get("object-label").required, true);
  assert.equal(app.get("object-description-optional").hidden, true);
  app.state.refs.push({ file: { name: "old-reference.jpg" }, url: "blob:ref", boxes: [], candidates: [] });
  app.state.targets.push({ file: { name: "rings.jpg" }, url: "blob:ring" });
  await app.get("start-button").listeners.click();
  assert.equal(submitted, undefined, "text-only model needs a description");
  assert.equal(app.get("notice").textContent, "Describe the object you want to count.");
  app.get("object-label").value = "white rings";
  await app.get("start-button").listeners.click();
  assert.ok(submitted, "text-only count must start without boxed references");
  assert.equal(submitted.some(([key]) => key === "references"), false);
  assert.equal(submitted.find(([key]) => key === "reference_boxes")[1], "[]");
  app.state.session = null;
  app.get("model-select").value = "yoloe-26s-seg";
  app.get("model-select").listeners.change();
  assert.equal(app.get("reference-panel").hidden, false);
  assert.equal(app.get("text-prompt-note").hidden, true);
  assert.equal(app.get("object-label").required, false);
  assert.equal(app.get("object-description-optional").hidden, false);
});

test("reference-guided models submit an empty description with manual boxes", async () => {
  for (const id of ["countgd-plusplus", "yoloe-26s-seg", "pet-dino-swin-t"]) {
    let submitted;
    const blankSession = { ...session(), label: "", model: id };
    const app = browser(async (path, options) => {
      if (path === "/api/sessions") {
        submitted = options.body.items;
        return response(blankSession, 201);
      }
      if (path.endsWith("/batches")) return response({ id: "batch-1" }, 202);
      return response(blankSession);
    }, null, { models: [{ id, name: id, available: true, reference_mode: "shared" }], default_model: id });
    await settle();
    assert.equal(app.get("object-label").required, false, id);
    assert.equal(app.get("object-description-optional").hidden, false, id);
    app.state.refs.push({ file: { name: "reference.jpg" }, url: "blob:ref", boxes: [[1, 2, 20, 30]], candidates: [] });
    app.state.targets.push({ file: { name: "target.jpg" }, url: "blob:target" });
    await app.get("start-button").listeners.click();
    assert.ok(submitted, `${id} should accept a blank description`);
    assert.equal(submitted.find(([key]) => key === "label")[1], "", id);
    assert.equal(submitted.find(([key]) => key === "reference_boxes")[1], "[[[1,2,20,30]]]", id);
    assert.equal(app.get("session-title").textContent, "Object count", id);
    blankSession.status = "finished";
    app.state.session = blankSession;
    app.render();
    assert.equal(app.get("session-title").textContent, "Object count · final count", id);
  }
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
  assert.match(results, /0 objects/);
  assert.match(results, /Check this photo and your selected examples/);
});

test("results omit processing diagnostics and model names for new and historical photos", async () => {
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
  const content = (element) => [element.textContent || "", ...element.children.map(content)].join(" ");
  for (const result of [tiled, historical]) {
    assert.match(content(result), /1 object/);
    assert.doesNotMatch(content(result), /tile|px|duplicates|score|YOLO|CountGD|model/i);
    assert.equal(result.children[1].children.some((child) => child.tag === "details"), false);
  }
  assert.equal(app.get("session-subtitle").textContent.includes(saved.model), false);
});

const referenceFile = (name = "sample.png") => ({ name, size: 100, type: "image/png", width: 100, height: 80 });
const editorItems = (app) => app.referenceReview.data(app.referenceReview.ref).items;
const activateBox = (app, index) => app.referenceReview.activate(index);
const toggleBox = (app) => {
  const data = app.referenceReview.data(app.referenceReview.ref);
  if (data.selectedActive != null) app.referenceReview.deselect(data.selectedActive);
  else app.get("reference-box-toggle").listeners.click();
};
const detectPhoto = async (app, photo, area = [0, 0, photo.width, photo.height]) => {
  app.referenceReview.addArea(photo, area);
  const index = (app.state.session?.reference_mode === "per_image" ||
    ["sam3", "countgd-plusplus"].includes(app.get("model-select").value))
    ? app.state.targets.indexOf(photo) : app.state.refs.indexOf(photo);
  app.get("reference-thumbs").children[index].listeners.click();
  await app.get("auto-detect-boxes").listeners.click();
};
const visualCatalog = { models: [
  { id: "yoloe-26s-seg", name: "YOLOE", available: true, reference_mode: "shared" },
  { id: "sam3", name: "SAM3", available: true, reference_mode: "per_image" },
], default_model: "sam3" };
const countgdCatalog = { models: [{ id: "countgd-plusplus", name: "CountGD++", available: true,
  reference_mode: "per_image", required_reference_boxes: null, recommended_reference_boxes: 10 }], default_model: "countgd-plusplus" };
const countgdProposals = (shift = 0) => Array.from({ length: 16 }, (_, index) => {
  const x = 5 + index % 4 * 20 + shift, y = 5 + Math.floor(index / 4) * 15;
  return [x, y, x + 8, y + 8];
});

test("CountGD filters proposals, preselects ten per photo, and submits nine initial and eleven later boxes", async () => {
  const first = countgdProposals(), second = countgdProposals(1);
  const visualSession = { ...session(), model: "countgd-plusplus", label: "", reference_mode: "per_image", required_reference_boxes: null, recommended_reference_boxes: 10 };
  let detections = 0, sessionPost;
  const batches = [];
  const app = browser(async (path, options) => {
    if (path === "/api/reference-boxes") return response({ width: 100, height: 80,
      boxes: [...(++detections === 1 ? first : second), [96, 76, 97, 77], [0, 0, 100, 80]] });
    if (path === "/api/sessions") { sessionPost = options.body.items; return response(visualSession, 201); }
    if (path.endsWith("/batches")) { batches.push(options.body.items); return response({ id: "batch-1" }, 202); }
    return response(visualSession);
  }, null, countgdCatalog);
  await settle();
  await app.addTargets([referenceFile("first.png"), referenceFile("second.png")]);
  assert.equal(detections, 0, "adding count photos must not call the detector");
  await detectPhoto(app, app.state.targets[0]);
  await detectPhoto(app, app.state.targets[1]);
  assert.equal(detections, 2);
  assert.equal(app.get("reference-upload-actions").hidden, true);
  assert.equal(app.state.refs.length, 0);
  assert.deepEqual(Array.from(app.state.targets, (item) => item.boxes.length), [10, 10]);
  for (const [photo, allowed] of [[app.state.targets[0], first], [app.state.targets[1], second]]) {
    assert.equal(photo.boxes.length + photo.candidates.length, 16, "tiny and oversized proposals are removed");
    assert.ok([...photo.boxes, ...photo.candidates].every((box) => allowed.some((valid) => String(valid) === String(box))));
  }
  app.get("reference-thumbs").children[0].listeners.click();
  const selected = editorItems(app).findIndex((item) => item.selected);
  app.referenceReview.deselect(selected);
  assert.equal(app.state.targets[0].boxes.length, 9);
  assert.equal(app.get("reference-done").disabled, false);
  assert.equal(app.get("reference-selected-count").textContent, "9 selected · 10 recommended");
  const expected = JSON.stringify(app.state.targets.map((item) => item.boxes));
  await app.get("start-button").listeners.click();
  assert.equal(sessionPost.find(([key]) => key === "reference_boxes")[1], "[]");
  assert.equal(sessionPost.some(([key]) => key === "references"), false);
  assert.equal(batches[0].find(([key]) => key === "image_boxes")[1], expected);
  await app.addTargets([referenceFile("later.png")]);
  await detectPhoto(app, app.state.targets[0]);
  const later = app.state.targets[0];
  assert.equal(later.boxes.length, 10);
  const candidate = editorItems(app).findIndex((item) => !item.selected);
  activateBox(app, candidate);
  toggleBox(app);
  assert.equal(later.boxes.length, 11);
  assert.equal(app.get("reference-done").disabled, false);
  const laterExpected = JSON.stringify([later.boxes]);
  await app.uploadBatch();
  assert.equal(batches[1].find(([key]) => key === "image_boxes")[1], laterExpected);
});

test("CountGD never restores size outliers to reach ten and accepts manually drawn replacements", async () => {
  const boxes = countgdProposals().slice(0, 8);
  const app = browser(async () => response({ width: 100, height: 80,
    boxes: [...boxes, [97, 77, 98, 78], [0, 0, 100, 80]] }), null, countgdCatalog);
  await settle();
  await app.addTargets([referenceFile()]);
  const photo = app.state.targets[0];
  await detectPhoto(app, photo);
  assert.equal(photo.boxes.length, 8);
  assert.equal(photo.candidates.length, 0);
  app.referenceReview.addBox(photo, [80, 60, 90, 70]);
  app.referenceReview.addBox(photo, [80, 71, 90, 79]);
  assert.equal(photo.boxes.length, 10);
  assert.equal(app.get("reference-done").disabled, false);
});

test("CountGD keeps manually selected boxes and fills only the remaining slots", async () => {
  let detections = 0;
  const app = browser(async (path) => {
    assert.equal(path, "/api/reference-boxes");
    detections++;
    return response({ width: 100, height: 80, boxes: countgdProposals() });
  }, null, countgdCatalog);
  await settle();
  await app.addTargets([referenceFile()]);
  const photo = app.state.targets[0];
  app.referenceReview.addBox(photo, [80, 60, 88, 68]);
  app.referenceReview.addBox(photo, [80, 70, 88, 78]);
  await detectPhoto(app, photo);
  assert.equal(detections, 1);
  assert.deepEqual(Array.from(photo.boxes.slice(0, 2), (box) => Array.from(box)), [
    [80, 60, 88, 68], [80, 70, 88, 78],
  ]);
  assert.equal(photo.boxes.length, 10);
  assert.equal(photo.candidates.length, 8);
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(detections, 2);
  assert.deepEqual(Array.from(photo.boxes.slice(0, 2), (box) => Array.from(box)), [
    [80, 60, 88, 68], [80, 70, 88, 78],
  ], "a later successful run preserves the selected manual boxes");
  assert.equal(photo.boxes.length, 10);
});

test("saved CountGD shared-reference sessions keep their original batch contract", async () => {
  const legacy = { ...session(), model: "countgd-plusplus", reference_mode: "shared", required_reference_boxes: null };
  let submitted;
  const app = browser(async (path, options) => {
    if (path.endsWith("/batches")) { submitted = options.body.items; return response({ id: "later" }, 202); }
    return response(legacy);
  }, "saved-session", countgdCatalog);
  await settle();
  await app.addTargets([referenceFile()]);
  assert.equal(app.state.targets[0].normalizedForPrompts, undefined);
  await app.uploadBatch();
  assert.ok(submitted.some(([key]) => key === "images"));
  assert.equal(submitted.some(([key]) => key === "image_boxes"), false);
});

test("SAM3 mirrors prepared count photos, preserves shared references across model changes, and revokes each removed URL once", async () => {
  let detections = 0;
  const app = browser(async (path) => {
    if (path === "/api/reference-boxes") { detections++; return response({ width: 100, height: 80, boxes: [[1, 2, 20, 30]] }); }
    return response(session());
  }, null, { ...visualCatalog, default_model: "yoloe-26s-seg" });
  await settle();
  await app.addReferences([referenceFile("shared.png")]);
  assert.equal(detections, 0, "shared reference upload waits for a marked detection area");
  const shared = app.state.refs[0];
  await app.addTargets([referenceFile("first.png"), referenceFile("second.png")]);
  const originalUrls = app.state.targets.map((item) => item.url);
  const model = app.get("model-select");
  model.value = "sam3";
  await model.listeners.change();
  assert.equal(app.get("object-description-field").hidden, true);
  assert.equal(app.get("reference-upload-actions").hidden, true);
  assert.equal(app.get("reference-count").textContent, "2 / 20");
  assert.equal(app.state.refs[0], shared, "shared reference input survives a model change");
  assert.deepEqual(Array.from(app.state.targets, (item) => item.file.type), ["image/jpeg", "image/jpeg"]);
  assert.ok(app.state.targets.every((item) => item.normalizedForPrompts && item.candidates.length === 0));
  assert.equal(detections, 0, "switching models must not call the detector");
  await detectPhoto(app, app.state.targets[0]);
  await detectPhoto(app, app.state.targets[1]);
  assert.ok(app.state.targets.every((item) => item.candidates.length === 1));
  for (const [index, item] of app.state.targets.entries()) {
    assert.equal(app.get("reference-thumbs").children[index].children[0].src, item.url);
    assert.equal(app.get("target-queue").children[index].children[0].src, item.url);
    assert.ok(app.revoked.includes(originalUrls[index]), "the old preview URL was replaced");
  }
  const preparedUrls = app.state.targets.map((item) => item.url);
  model.value = "yoloe-26s-seg";
  await model.listeners.change();
  assert.equal(app.get("reference-count").textContent, "1 / 8");
  assert.equal(app.state.refs[0], shared);
  assert.ok(preparedUrls.every((url) => !app.revoked.includes(url)));
  model.value = "sam3";
  await model.listeners.change();
  assert.deepEqual(Array.from(app.state.targets, (item) => item.url), Array.from(preparedUrls), "prepared photos are reused");
  assert.equal(detections, 2, "model switching never re-runs detection");
  app.get("remove-reference").listeners.click();
  assert.equal(app.state.targets.length, 1);
  assert.equal(app.revoked.filter((url) => url === preparedUrls[0]).length, 1);
  assert.equal(app.state.refs[0], shared);
  app.get("target-queue").children[0].children[1].listeners.click();
  assert.equal(app.state.targets.length, 0);
  assert.equal(app.revoked.filter((url) => url === preparedUrls[1]).length, 1);
  assert.ok(!app.revoked.includes(shared.url));
});

test("switching SAM3 to CountGD preserves selected boxes and candidates without adding examples", async () => {
  const models = { models: [
    { id: "sam3", name: "SAM3", available: true, reference_mode: "per_image" },
    { id: "countgd-plusplus", name: "CountGD++", available: true, reference_mode: "per_image", required_reference_boxes: null, recommended_reference_boxes: 10 },
  ], default_model: "sam3" };
  let detections = 0;
  const app = browser(async (path) => { if (path === "/api/reference-boxes") detections++; return response(session()); }, null, models);
  await settle();
  await app.addTargets([referenceFile()]);
  const photo = app.state.targets[0];
  app.referenceReview.addBox(photo, [0, 0, 100, 80]);
  app.referenceReview.addArea(photo, [0, 0, 20, 20]);
  photo.candidates = countgdProposals();
  app.referenceReview.reset(photo);
  const model = app.get("model-select");
  model.value = "countgd-plusplus";
  await model.listeners.change();
  assert.equal(detections, 0, "neither photo upload nor model switch calls the detector");
  assert.equal(app.state.targets[0], photo);
  assert.deepEqual(Array.from(photo.boxes[0]), [0, 0, 100, 80], "the manually selected size outlier remains selected");
  assert.equal(photo.boxes.length, 1);
  assert.equal(photo.candidates.length, 16);
  assert.equal(app.get("reference-done").disabled, false);
  assert.deepEqual(Array.from(photo.detectionAreas[0]), [0, 0, 20, 20]);
});

test("switching SAM3 to CountGD keeps eleven manual selections and allows Done", async () => {
  const models = { models: [
    { id: "sam3", name: "SAM3", available: true, reference_mode: "per_image" },
    { id: "countgd-plusplus", name: "CountGD++", available: true, reference_mode: "per_image", required_reference_boxes: null, recommended_reference_boxes: 10 },
  ], default_model: "sam3" };
  let detections = 0;
  const app = browser(async (path) => { if (path === "/api/reference-boxes") detections++; return response(session()); }, null, models);
  await settle();
  await app.addTargets([referenceFile()]);
  const photo = app.state.targets[0];
  const selected = countgdProposals().slice(0, 11);
  for (const box of selected) app.referenceReview.addBox(photo, box);
  assert.equal(photo.boxes.length, 11);
  const model = app.get("model-select");
  model.value = "countgd-plusplus";
  await model.listeners.change();
  assert.equal(detections, 0);
  assert.deepEqual(Array.from(photo.boxes, (box) => Array.from(box)), selected,
    "the model change cannot silently discard a selected reference");
  assert.equal(app.get("notice").textContent, "");
  assert.equal(app.get("reference-done").disabled, false);
});

test("SAM3 submits only selected boxes in image order and keeps the editor usable for later batches", async () => {
  let sessionPost;
  let detections = 0;
  const batches = [];
  const visualSession = { ...session(), model: "sam3", label: "" };
  const app = browser(async (path, options) => {
    if (path === "/api/reference-boxes") { detections++; return response({ width: 100, height: 80,
      boxes: [[1, 2, 20, 30], [40, 10, 70, 60]] }); }
    if (path === "/api/sessions") { sessionPost = options.body.items; return response(visualSession, 201); }
    if (path.endsWith("/batches")) { batches.push(options.body.items); return response({ id: `batch-${batches.length}` }, 202); }
    return response(visualSession);
  }, null, visualCatalog);
  await settle();
  await app.addTargets([referenceFile("one.png"), referenceFile("two.png")]);
  assert.equal(detections, 0);
  await detectPhoto(app, app.state.targets[0]);
  await detectPhoto(app, app.state.targets[1]);
  app.get("reference-thumbs").children[0].listeners.click();
  toggleBox(app);
  await app.get("start-button").listeners.click();
  assert.equal(sessionPost, undefined, "all photos need a selected box before creating the session");
  assert.equal(app.state.selected, 1, "the missing photo opens automatically");
  assert.match(app.get("notice").textContent, /photo 2/);
  activateBox(app, 1);
  toggleBox(app);
  const uploadedUrls = app.state.targets.map((item) => item.url);
  await app.get("start-button").listeners.click();
  assert.equal(sessionPost.find(([key]) => key === "label")[1], "");
  assert.equal(sessionPost.find(([key]) => key === "reference_boxes")[1], "[]");
  assert.equal(sessionPost.some(([key]) => key === "references"), false);
  assert.equal(batches.length, 1);
  assert.deepEqual(batches[0].filter(([key]) => key === "images").map(([, file]) => file.name), ["one.jpg", "two.jpg"]);
  assert.equal(batches[0].find(([key]) => key === "image_boxes")[1], "[[[1,2,20,30]],[[40,10,70,60]]]");
  assert.equal(app.state.targets.length, 0);
  assert.ok(uploadedUrls.every((url) => app.revoked.filter((item) => item === url).length === 1));
  assert.equal(app.get("reference-panel").hidden, true, "empty photo selection is hidden between batches");
  assert.ok(app.get("session-reference-slot").children.includes(app.get("reference-panel")));
  await app.addTargets([referenceFile("three.png")]);
  assert.equal(detections, 2, "preparing a later batch must not call the detector");
  assert.equal(app.get("reference-panel").hidden, false);
  assert.equal(app.get("reference-editor").hidden, false);
  await app.uploadBatch();
  assert.equal(batches.length, 1, "a missing choice blocks a later batch too");
  await detectPhoto(app, app.state.targets[0]);
  toggleBox(app);
  await app.uploadBatch();
  assert.equal(batches.length, 2);
  assert.equal(batches[1].find(([key]) => key === "image_boxes")[1], "[[[1,2,20,30]]]");
});

test("SAM3 applies the 32-box limit separately to each photo and restores session mode by model id", async () => {
  const manyBoxes = Array.from({ length: 33 }, (_, index) => [index, 1, index + 1, 20]);
  const app = browser(async (path) => path === "/api/reference-boxes"
    ? response({ width: 100, height: 80, boxes: manyBoxes }) : response(session()), null, visualCatalog);
  await settle();
  await app.addTargets([referenceFile("one.png"), referenceFile("two.png")]);
  await detectPhoto(app, app.state.targets[0]);
  await detectPhoto(app, app.state.targets[1]);
  app.get("reference-thumbs").children[0].listeners.click();
  for (let index = 0; index < 32; index++) { activateBox(app, index); toggleBox(app); }
  activateBox(app, 32);
  assert.equal(app.get("reference-box-toggle").disabled, true);
  toggleBox(app);
  assert.equal(app.state.targets[0].boxes.length, 32);
  app.get("reference-thumbs").children[1].listeners.click();
  assert.equal(app.get("reference-box-toggle").disabled, false);
  toggleBox(app);
  activateBox(app, 1);
  toggleBox(app);
  assert.equal(app.state.targets[1].boxes.length, 2);
  const restored = browser(async () => response({ ...session(), model: "sam3", label: "" }),
    "saved-session", catalog);
  await settle();
  assert.equal(restored.get("reference-panel").hidden, true, "restored SAM3 sessions keep empty photo selection hidden");
  assert.ok(restored.get("session-reference-slot").children.includes(restored.get("reference-panel")));
});

test("explicit region detection proposes boxes with a blank description; only a chosen box reaches counting", async () => {
  const boxes = [[1, 2, 20, 30], [40, 10, 70, 60]];
  let submitted;
  let requests = 0;
  const app = browser(async (path, options) => {
    if (path === "/api/reference-boxes") {
      requests++;
      assert.equal(options.method, "POST");
      assert.deepEqual(options.body.items.map(([field, value]) => [field, value.name, value.type, value.width, value.height]),
        [["image", "sample.jpg", "image/jpeg", 100, 80]]);
      return response({ width: 100, height: 80, boxes, model: "fastsam-s", elapsed_ms: 120, warning: null });
    }
    if (path === "/api/sessions") { submitted = options.body.items; return response(session(), 201); }
    if (path.endsWith("/batches")) return response({ id: "batch-1" }, 202);
    return response(session());
  });
  await settle();
  await app.addReferences([referenceFile()]);
  assert.equal(requests, 0, "uploading a reference never starts detection");
  assert.equal(app.get("auto-detect-boxes").disabled, true, "a marked region is required");
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(requests, 0, "the handler also rejects a call without regions");
  await detectPhoto(app, app.state.refs[0]);
  assert.equal(requests, 1);
  assert.equal(app.decoded.filter((file) => file.name === "sample.jpg").length, 1);
  assert.equal(app.canvases.at(-1).blobOptions.quality, 0.94);
  assert.equal(app.state.refs[0].boxes.length, 0);
  assert.deepEqual(Array.from(app.state.refs[0].candidates, (box) => Array.from(box)), boxes);
  assert.equal(editorItems(app).length, 2);
  assert.equal(editorItems(app)[0].selected, false);
  assert.equal(app.get("reference-box-position").textContent, "1 / 2");
  app.state.targets.push({ file: { name: "target.jpg" }, url: "blob:target" });
  await app.get("start-button").listeners.click();
  assert.equal(submitted, undefined, "proposals cannot start a count");
  assert.equal(app.get("notice").textContent, "Select the object(s) to use as examples.");
  activateBox(app, 1);
  toggleBox(app);
  assert.equal(app.state.refs[0].boxes.length, 1);
  assert.equal(app.state.refs[0].candidates.length, 1);
  assert.equal(editorItems(app).length, 2, "internal box identities stay stable");
  assert.equal(editorItems(app)[1].selected, true);
  assert.equal(app.get("reference-crop").dataset.index, "0", "selected boxes leave the inspector queue");
  await app.get("start-button").listeners.click();
  assert.equal(submitted.find(([key]) => key === "label")[1], "");
  assert.equal(submitted.find(([key]) => key === "reference_boxes")[1], "[[[40,10,70,60]]]", "unchosen proposal is omitted");
});

test("multiple regions send exact JPEG crops, offset proposals, and discard results from removed areas", async () => {
  const sent = [];
  const app = browser(async (path, options) => {
    assert.equal(path, "/api/reference-boxes");
    const file = options.body.items[0][1];
    sent.push([file.width, file.height, file.type]);
    if (file.height === 50) return response({ width: 60, height: 50,
      boxes: [[45, 25, 55, 35], [1, 1, 11, 11]] });
    return response({ width: 60, height: 60,
      boxes: [[5.2, 5, 15.2, 15], [40, 40, 50, 50]] });
  });
  await settle();
  await app.addReferences([referenceFile()]);
  const ref = app.state.refs[0];
  app.referenceReview.addArea(ref, [0, 0, 60, 50]);
  app.referenceReview.addArea(ref, [40, 20, 100, 80]);
  await app.get("auto-detect-boxes").listeners.click();
  assert.deepEqual(sent, [[60, 50, "image/jpeg"], [60, 60, "image/jpeg"]]);
  assert.equal(app.decoded.filter((file) => file.name === "sample.jpg").length, 1,
    "all regions of one run share one decode");
  assert.deepEqual(app.canvases.slice(1).map((canvas) => canvas.drawCalls[0].slice(1)), [
    [0, 0, 60, 50, 0, 0, 60, 50], [40, 20, 60, 60, 0, 0, 60, 60],
  ]);
  assert.ok(app.canvases.slice(1).every((canvas) => canvas.blobOptions.type === "image/jpeg" && canvas.blobOptions.quality === 0.94));
  assert.deepEqual(Array.from(ref.candidates, (box) => Array.from(box)), [
    [45, 25, 55, 35], [1, 1, 11, 11], [80, 60, 90, 70],
  ], "near-duplicate proposals from overlapping regions appear once in photo coordinates");
  app.referenceReview.removeArea(0);
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(sent.length, 3, "only the remaining area is requested");
  assert.deepEqual(Array.from(ref.candidates, (box) => Array.from(box)), [
    [45.2, 25, 55.2, 35], [80, 60, 90, 70],
  ], "proposals from a removed region are cleared on the next run");
});

test("a failed later region keeps manual selections and earlier proposals until a complete retry", async () => {
  let calls = 0;
  const app = browser(async (path) => {
    assert.equal(path, "/api/reference-boxes");
    calls++;
    if (calls === 3) return response({ detail: "Detector unavailable" }, 503);
    if (calls === 1) return response({ width: 50, height: 40, boxes: [[30, 10, 40, 20]] });
    return calls % 2 === 0
      ? response({ width: 50, height: 40, boxes: [[35, 10, 45, 20]] })
      : response({ width: 50, height: 60, boxes: [[5, 5, 15, 15]] });
  });
  await settle();
  await app.addReferences([referenceFile()]);
  const ref = app.state.refs[0];
  app.referenceReview.addBox(ref, [10, 10, 20, 20]);
  app.referenceReview.addArea(ref, [0, 0, 50, 40]);
  await app.get("auto-detect-boxes").listeners.click();
  assert.deepEqual(Array.from(ref.candidates, (box) => Array.from(box)), [[30, 10, 40, 20]]);
  app.referenceReview.addArea(ref, [50, 20, 100, 80]);
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(calls, 3);
  assert.deepEqual(Array.from(ref.boxes, (box) => Array.from(box)), [[10, 10, 20, 20]]);
  assert.deepEqual(Array.from(ref.candidates, (box) => Array.from(box)), [[30, 10, 40, 20]],
    "a successful first crop must not partly replace previous proposals");
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(calls, 5);
  assert.deepEqual(Array.from(ref.boxes, (box) => Array.from(box)), [[10, 10, 20, 20]]);
  assert.deepEqual(Array.from(ref.candidates, (box) => Array.from(box)), [
    [35, 10, 45, 20], [55, 25, 65, 35],
  ]);
  const undo = () => app.get("undo-box").listeners.click();
  const boxes = (items) => Array.from(items, (box) => Array.from(box));
  assert.deepEqual(boxes(ref.detectionAreas), [[0, 0, 50, 40], [50, 20, 100, 80]]);
  undo();
  assert.deepEqual(boxes(ref.boxes), [[10, 10, 20, 20]], "Undo keeps the manual selection");
  assert.deepEqual(boxes(ref.candidates), [[30, 10, 40, 20]], "Undo restores proposals from before the two-region run");
  assert.deepEqual(boxes(ref.detectionAreas), [[0, 0, 50, 40], [50, 20, 100, 80]]);
  undo();
  assert.deepEqual(boxes(ref.detectionAreas), [[0, 0, 50, 40]], "the earlier area edit is still undoable");
  undo();
  assert.deepEqual(boxes(ref.candidates), [], "the earlier one-region detection is still undoable");
  undo();
  assert.deepEqual(boxes(ref.detectionAreas), [], "the first area edit is still undoable");
  undo();
  assert.deepEqual(boxes(ref.boxes), [], "the original manual box edit is still undoable");
});

test("manual boxes and Auto-detect work without a description alongside unchosen proposals", async () => {
  let requests = 0;
  const app = browser(async (path, options) => {
    assert.equal(path, "/api/reference-boxes");
    requests++;
    assert.deepEqual(options.body.items.map(([field, value]) => [field, value.name]), [["image", "sample.jpg"]]);
    return response({ width: 100, height: 80, boxes: [[1, 2, 20, 30]], model: "fastsam-s", warning: null });
  });
  await settle();
  await app.addReferences([referenceFile()]);
  const ref = app.state.refs[0];
  app.referenceReview.addArea(ref, [0, 0, 100, 80]);
  assert.equal(requests, 0);
  app.get("box-x1").valueAsNumber = 40;
  app.get("box-y1").valueAsNumber = 10;
  app.get("box-x2").valueAsNumber = 70;
  app.get("box-y2").valueAsNumber = 60;
  app.get("coordinate-form").listeners.submit({ preventDefault() {} });
  assert.equal(app.state.refs[0].boxes.length, 1, "manual box is confirmed");
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(requests, 1, "manual Auto-detect accepts a blank description and selected boxes");
  assert.deepEqual(Array.from(ref.boxes, (box) => [...box]), [[40, 10, 70, 60]], "selected manual boxes survive detection");
  assert.equal(ref.candidates.length, 1, "unselected proposal remains separate");
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(requests, 2);
  assert.equal(ref.candidates.length, 1, "retry merges, rather than duplicates, proposals");
  const proposal = editorItems(app).findIndex((item) => !item.selected);
  activateBox(app, proposal);
  toggleBox(app);
  assert.equal(editorItems(app).length, 2);
  assert.equal(editorItems(app)[proposal].selected, true);
  assert.equal(app.get("auto-detect-boxes").disabled, false, "more regions can be inspected after selecting boxes");
});

test("detection locks editing while pending and failure keeps the reference for manual boxing", async () => {
  const pending = deferred();
  const app = browser(async (path) => path === "/api/reference-boxes" ? pending.promise : response(session()));
  await settle();
  await app.addReferences([referenceFile()]);
  app.referenceReview.addArea(app.state.refs[0], [0, 0, 100, 80]);
  const detection = app.get("auto-detect-boxes").listeners.click();
  await settle();
  assert.equal(app.state.busy, true);
  assert.equal(app.get("start-button").disabled, true);
  assert.equal(app.get("remove-reference").disabled, true);
  assert.equal(app.get("auto-detect-boxes").disabled, true);
  assert.equal(app.get("auto-detect-boxes")["aria-busy"], "true");
  assert.equal(app.get("box-markers").children.length, 0);
  assert.match(app.get("notice").textContent, /Finding objects/);
  app.get("box-x1").valueAsNumber = 1;
  app.get("box-y1").valueAsNumber = 2;
  app.get("box-x2").valueAsNumber = 20;
  app.get("box-y2").valueAsNumber = 30;
  app.get("coordinate-form").listeners.submit({ preventDefault() {} });
  assert.equal(app.state.refs[0].boxes.length, 0);
  pending.resolve(response({ detail: "Detector unavailable" }, 503));
  await detection;
  assert.equal(app.state.busy, false);
  assert.equal(app.state.refs.length, 1);
  assert.equal(app.get("auto-detect-boxes").disabled, false);
  assert.match(app.get("notice").textContent, /Draw boxes manually or retry/);
  app.get("coordinate-form").listeners.submit({ preventDefault() {} });
  assert.equal(app.state.refs[0].boxes.length, 1);
  assert.equal(app.get("auto-detect-boxes").disabled, false, "marked regions can be retried with a manual selection");
});

test("retry locks visible proposals and preserves them after a detector failure", async () => {
  const pending = deferred();
  let calls = 0;
  const app = browser(async (path) => path === "/api/reference-boxes" && ++calls === 1
    ? response({ width: 100, height: 80, boxes: [[1, 2, 20, 30]], model: "fastsam-s", warning: null })
    : pending.promise);
  await settle();
  await app.addReferences([referenceFile()]);
  await detectPhoto(app, app.state.refs[0]);
  const retry = app.get("auto-detect-boxes").listeners.click();
  await settle();
  assert.equal(app.get("reference-box-toggle").disabled, true);
  app.get("reference-box-toggle").listeners.click();
  assert.equal(app.state.refs[0].boxes.length, 0);
  assert.equal(app.get("remove-reference").disabled, true);
  pending.resolve(response({ detail: "Detector unavailable" }, 503));
  await retry;
  assert.equal(app.state.refs[0].candidates.length, 1, "failed retry keeps previous proposals");
  assert.equal(app.get("reference-box-toggle").disabled, false);
  activateBox(app, 0);
  toggleBox(app);
  assert.equal(app.state.refs[0].boxes.length, 1);
  app.get("remove-reference").listeners.click();
  assert.equal(app.state.refs.length, 0, "reference and its proposals can be removed after retry");
  assert.equal(app.get("reference-editor").hidden, true);
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
  await app.addReferences([referenceFile()]);
  app.referenceReview.addArea(app.state.refs[0], [0, 0, 100, 80]);
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(app.state.refs[0].boxes.length, 0);
  assert.match(app.get("notice").textContent, /did not fit/);
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(app.state.refs[0].boxes.length, 0);
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(calls, 3);
  assert.equal(app.state.refs[0].boxes.length, 0);
  assert.equal(app.state.refs[0].candidates.length, 1);
});

test("proposals do not consume the 32-box limit until selected", async () => {
  let calls = 0;
  const app = browser(async () => {
    calls++;
    return response({ width: 100, height: 80, boxes: Array.from({ length: 33 }, (_, index) => [index * 2, 2, index * 2 + 1, 30]), warning: null });
  });
  await settle();
  await app.addReferences([referenceFile("first.png"), referenceFile("second.png")]);
  assert.equal(calls, 0);
  await detectPhoto(app, app.state.refs[0]);
  await detectPhoto(app, app.state.refs[1]);
  assert.equal(calls, 2, "both marked references receive proposals");
  assert.equal(app.state.refs[0].candidates.length, 33);
  assert.equal(app.state.refs[0].boxes.length, 0);
  assert.equal(app.state.refs[1].boxes.length, 0);
  app.get("reference-thumbs").children[0].listeners.click();
  for (let index = 0; index < 32; index++) { activateBox(app, index); toggleBox(app); }
  assert.equal(app.state.refs[0].boxes.length, 32);
  assert.equal(app.state.refs[0].candidates.length, 1);
  assert.equal(editorItems(app).length, 33);
  activateBox(app, 32);
  assert.equal(app.get("reference-box-toggle").disabled, true);
  app.get("reference-thumbs").children[1].listeners.click();
  assert.equal(app.get("reference-box-toggle").disabled, true);
  toggleBox(app);
  assert.equal(app.state.refs[1].boxes.length, 0, "limit is enforced even if the handler is invoked directly");
  app.get("reference-thumbs").children[0].listeners.click();
  activateBox(app, 0);
  toggleBox(app);
  app.get("reference-thumbs").children[1].listeners.click();
  assert.equal(app.get("reference-box-toggle").disabled, false);
  toggleBox(app);
  assert.equal(app.state.refs[1].boxes.length, 1);
  assert.equal(app.state.refs.reduce((sum, ref) => sum + ref.boxes.length, 0), 32);
});


test("CountGD blocks duplicate and empty per-photo references before creating a session", async () => {
  let submitted = false;
  const app = browser(async () => { submitted = true; return response(session(), 201); }, null, countgdCatalog);
  await settle();
  const box = [1, 2, 20, 30];
  app.state.targets.push({ file: referenceFile(), url: "blob:duplicate", width: 100, height: 80,
    normalizedForPrompts: true, boxes: Array.from({ length: 10 }, () => [...box]), candidates: [] });
  await app.get("start-button").listeners.click();
  assert.equal(submitted, false);
  assert.equal(app.state.session, null);
  assert.match(app.get("notice").textContent, /1 to 32 distinct boxes/);
  app.state.targets[0].boxes = [];
  await app.get("start-button").listeners.click();
  assert.equal(submitted, false);
  assert.match(app.get("notice").textContent, /Draw a box on photo 1/);
});

test("an older counting engine remains viewable but offers a new count instead of more uploads", async () => {
  const saved = session();
  const app = browser(async () => response(saved), "saved-session", countgdCatalog);
  await settle();
  assert.equal(app.get("next-batch-panel").hidden, true);
  assert.equal(app.get("new-count-button").hidden, false);
  assert.equal(app.get("session-subtitle").textContent.includes(saved.model), false);
  app.get("new-count-button").listeners.click();
  assert.equal(app.state.session, null);
  assert.equal(app.get("model-select").value, "countgd-plusplus");
});

test("technical failures become everyday instructions without exposing engine details", async () => {
  const app = browser(async () => response({ detail: "CountGD++ CUDA checkpoint failed" }, 503), null, countgdCatalog);
  await settle();
  await app.addTargets([referenceFile()]);
  const ref = app.state.targets[0];
  ref.detectionAreas = [[0, 0, 100, 80]];
  app.referenceReview.render(ref);
  await app.get("auto-detect-boxes").listeners.click();
  assert.match(app.get("notice").textContent, /Please try again/);
  assert.doesNotMatch(app.get("notice").textContent, /CountGD|CUDA|checkpoint/);
});


test("Quick count requires a marked area on every photo before creating a count", async () => {
  const calls = [];
  const app = browser(async (path) => { calls.push(path); return response(session()); }, null, countgdCatalog);
  await settle();
  await app.get("quick-count-button").listeners.click();
  assert.deepEqual(calls, []);
  await app.addTargets([referenceFile("first.png"), referenceFile("second.png")]);
  app.state.targets[0].detectionAreas = [[1, 1, 80, 60]];
  await app.get("quick-count-button").listeners.click();
  assert.deepEqual(calls, []);
  assert.equal(app.state.selected, 1);
  assert.equal(app.referenceReview.areaMode, true);
  assert.match(app.get("notice").textContent, /Mark an area in photo 2/);
  assert.equal(app.state.targets[1].boxes.length, 0);
});

test("Quick count sends marked areas in photo order without running suggestions or requiring ten examples", async () => {
  const calls = [], forms = [];
  const saved = { ...session(), label: "", model: "countgd-plusplus", reference_mode: "per_image" };
  const app = browser(async (path, options) => {
    calls.push(path);
    if (options?.body) forms.push([path, options.body.items]);
    if (path.endsWith("/batches")) return response({ id: "quick-batch", status: "queued" }, 202);
    return response(saved, path === "/api/sessions" ? 201 : 200);
  }, null, countgdCatalog);
  await settle();
  await app.addTargets([referenceFile("first.png"), referenceFile("second.png")]);
  const photos = [...app.state.targets];
  photos[0].detectionAreas = [[1, 2, 60, 70], [30, 5, 95, 75]];
  photos[1].detectionAreas = [[5, 3, 90, 65]];
  await app.get("start-button").listeners.click();
  assert.deepEqual(calls, [], "standard count still requires at least one example");
  await app.get("quick-count-button").listeners.click();
  assert.equal(calls.includes("/api/reference-boxes"), false);
  const entries = forms.find(([path]) => path.endsWith("/batches"))[1];
  assert.equal(entries.find(([name]) => name === "count_mode")[1], "quick");
  assert.deepEqual(JSON.parse(entries.find(([name]) => name === "search_areas")[1]), [photos[0].detectionAreas, photos[1].detectionAreas]);
  assert.equal(entries.some(([name]) => name === "image_boxes"), false);
  assert.equal(photos[0].boxes.length, 0);
  assert.equal(photos[1].boxes.length, 0);
  assert.equal(app.state.targets.length, 0);
});

test("later Quick count locks controls while uploading and keeps photos and areas on failure", async () => {
  const pending = deferred();
  const saved = { ...session(), model: "countgd-plusplus", reference_mode: "per_image" };
  const app = browser(async (path) => path.endsWith("/batches") ? pending.promise : response(saved), null, countgdCatalog);
  await settle();
  app.state.session = saved;
  await app.addTargets([referenceFile()]);
  const photo = app.state.targets[0];
  photo.detectionAreas = [[1, 2, 80, 70]];
  const upload = app.get("quick-upload-button").listeners.click();
  assert.equal(app.state.busy, true);
  assert.equal(app.get("quick-upload-button").disabled, true);
  assert.equal(app.get("upload-button").disabled, true);
  pending.resolve(response({ detail: "Please try again." }, 503));
  await upload;
  assert.equal(app.state.targets[0], photo);
  assert.deepEqual(photo.detectionAreas, [[1, 2, 80, 70]]);
  assert.equal(app.state.busy, false);
  assert.equal(app.get("quick-upload-button").disabled, false);
});

test("Quick result warnings stay visible without requiring examples or showing engine names", async () => {
  const saved = { ...session(), model: "countgd-plusplus", reference_mode: "per_image", image_count: 1 };
  saved.batches = [{ id: "quick", count_mode: "quick", status: "complete", total_count: 0,
    results: [{ filename: "parts.jpg", count: 0, detections: [], warning: "Many objects were found. Try Start counting for a fuller count." }] }];
  const app = browser(async () => response(saved), null, countgdCatalog);
  await settle();
  app.state.session = saved;
  app.render();
  const content = (element) => [element.textContent || "", ...element.children.map(content)].join(" ");
  assert.match(content(app.get("batch-list")), /Quick count/);
  assert.match(content(app.get("batch-list")), /Try Start counting for a fuller count/);
  saved.batches[0].results[0].warning = null;
  app.render();
  assert.match(content(app.get("batch-list")), /No objects found in the marked areas/);
  assert.doesNotMatch(content(app.get("batch-list")), /selected examples|model|FastSAM/);
});


test("Find objects preselects ten from more than 32 suggestions, then allows up to 32", async () => {
  const boxes = Array.from({ length: 40 }, (_, i) => {
    const x = 2 + i % 10 * 8, y = 2 + Math.floor(i / 10) * 8;
    return [x, y, x + 4, y + 4];
  });
  const app = browser(async (path) => {
    assert.equal(path, "/api/reference-boxes");
    return response({ width: 100, height: 80, boxes, warning: null });
  }, null, countgdCatalog);
  await settle();
  await app.addTargets([referenceFile()]);
  const photo = app.state.targets[0];
  photo.detectionAreas = [[0, 0, 100, 80]];
  app.referenceReview.render(photo);
  await app.get("auto-detect-boxes").listeners.click();
  assert.equal(photo.boxes.length, 10);
  assert.equal(photo.candidates.length, 30);
  assert.equal(new Set([...photo.boxes, ...photo.candidates].map(JSON.stringify)).size, 40);
  assert.equal(app.get("reference-selected-count").textContent, "10 selected · 10 recommended");
  for (const index of editorItems(app).flatMap((item, index) => item.selected ? [] : [index]).slice(0, 22)) {
    activateBox(app, index);
    toggleBox(app);
  }
  assert.equal(photo.boxes.length, 32);
  assert.equal(app.get("reference-done").disabled, false);
  assert.equal(app.get("reference-box-toggle").disabled, true);
  assert.equal(photo.candidates.length, 8, "remaining suggestions stay available but cannot exceed 32 selected boxes");
});
