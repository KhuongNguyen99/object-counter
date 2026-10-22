import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";

const source = readFileSync(new URL("../app/static/reference-review.js", import.meta.url), "utf8");
const markup = readFileSync(new URL("../app/static/index.html", import.meta.url), "utf8");

function fixture(config = {}) {
  const nodes = new Map();
  const document = { activeElement: null, documentElement: { clientWidth: 1000, clientHeight: 800 } };
  const canvasOps = [], canvas = {
    path: [],
    setTransform() {}, fillRect() {},
    drawImage(...args) { canvasOps.push({ type: "drawImage", args }); },
    setLineDash(value) { canvasOps.push({ type: "dash", value: [...value] }); },
    beginPath() { this.path = []; },
    moveTo(...args) { this.path.push(["moveTo", ...args]); },
    lineTo(...args) { this.path.push(["lineTo", ...args]); },
    stroke() { canvasOps.push({ type: "stroke", path: [...this.path], color: this.strokeStyle, width: this.lineWidth, cap: this.lineCap }); },
  };
  class Element {
    constructor(tag = "div", id = "") {
      this.tag = tag; this.id = id; this.children = []; this.style = {}; this.dataset = {}; this.attributes = {};
      this.handlers = {}; this.className = ""; this.classList = { add() {}, remove() {}, toggle() {} };
      this.clientWidth = 500; this.clientHeight = 400; this.scrollLeft = 0; this.scrollTop = 0;
      this.hidden = false; this.disabled = false;
    }
    append(...children) { for (const child of children) { child.parentElement = this; this.children.push(child); } }
    replaceChildren(...children) { this.children = []; this.append(...children); }
    addEventListener(type, fn) { (this.handlers[type] ||= []).push(fn); }
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key]; }
    focus() { document.activeElement = this; for (const fn of this.handlers.focus || []) fn(); }
    click() { if (!this.disabled) for (const fn of this.handlers.click || []) fn({ stopPropagation() {} }); }
    closest(selector) {
      for (let element = this; element; element = element.parentElement)
        if (selector.split(",").some((part) => {
          part = part.trim();
          if (part === `#${element.id}` || part === element.tag) return true;
          return part === '[contenteditable]:not([contenteditable="false"])' &&
            element.attributes.contenteditable != null && element.attributes.contenteditable !== "false";
        })) return element;
      return null;
    }
    querySelector(selector) {
      for (const child of this.children) {
        if (selector === "img" && child.tag === "img" ||
            selector === '.reference-box-marker[aria-current="true"]' && child.className === "reference-box-marker" && child.attributes["aria-current"] === "true" ||
            selector === '.reference-selected-thumbnail[aria-current="true"]' && child.className === "reference-selected-thumbnail" && child.attributes["aria-current"] === "true") return child;
        const nested = child.querySelector(selector); if (nested) return nested;
      }
      return null;
    }
    getBoundingClientRect() { return this.rect || { left: 0, top: 0, width: this.clientWidth, height: this.clientHeight, right: this.clientWidth, bottom: this.clientHeight }; }
    getContext(type) { return this.tag === "canvas" && type === "2d" ? canvas : null; }
    setPointerCapture() {}
  }
  document.getElementById = (id) => nodes.get(id);
  document.createElement = (tag) => new Element(tag);
  document.createElementNS = (_, tag) => new Element(tag);
  for (const [, tag, id] of markup.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"/g)) nodes.set(id, new Element(tag, id));
  nodes.get("reference-crop").append(new Element("img"));
  const context = vm.createContext({ document });
  vm.runInContext(source, context);
  let busy = false, limit = config.limit ?? 32, required = config.required ?? null, extra = 0;
  const messages = [], done = [];
  const editor = new context.ReferenceBoxEditor({ isBusy: () => busy, getLimit: () => limit, getRequired: () => required,
    getTotal: (ref) => ref.boxes.length + extra, message: (value) => messages.push(value), onDone: () => done.push(true) });
  const ref = { width: 2000, height: 1600, url: "test.png", boxes: config.boxes || [],
    candidates: config.candidates || [[100,100,150,150], [300,100,350,150], [500,100,550,150]],
    detectionAreas: config.detectionAreas || [] };
  editor.render(ref);
  return { editor, ref, nodes, document, messages, done, canvasOps, setBusy(value) { busy = value; }, setLimit(value) { limit = value; }, setExtra(value) { extra = value; },
    key(key, options = {}) { let prevented = false; editor.keydown({ key, target: nodes.get("reference-image-viewport"), preventDefault() { prevented = true; }, ...options }); return prevented; },
    pointer(target, button = 0) { for (const fn of nodes.get("reference-editor").handlers.pointerdown || []) fn({ target, button }); } };
}

const plain = (value) => JSON.parse(JSON.stringify(value));

function readyMagnifier(f) {
  Object.assign(f.nodes.get("editor-image"), { complete: true, naturalWidth: 2000, naturalHeight: 1600 });
  return { clientX: 250, clientY: 200, pointerType: "mouse" };
}

function wheel(f, deltaY, options = {}) {
  let prevented = false;
  f.nodes.get("reference-image-viewport").handlers.wheel[0]({
    ...readyMagnifier(f), deltaY, deltaMode: 0, altKey: false,
    preventDefault() { prevented = true; }, ...options,
  });
  return prevented;
}

test("plain areas of the reference editor take keyboard focus on pointer down", () => {
  const f = fixture(), root = f.nodes.get("reference-editor");
  assert.match(markup, /id="reference-editor"[^>]*tabindex="0"[^>]*role="group"[^>]*aria-label="Example selection"/);
  for (const target of [root, f.nodes.get("editor-image"), f.nodes.get("reference-crop"), f.nodes.get("reference-selected-count")]) {
    f.document.activeElement = null;
    f.pointer(target);
    assert.equal(f.document.activeElement, root, `${target.id || target.tag} focuses the editor`);
  }
  f.document.activeElement = null;
  f.pointer(f.nodes.get("editor-image"), 1);
  assert.equal(f.document.activeElement, root, "middle-button pan keeps shortcuts in the editor");
  f.document.activeElement = null;
  f.pointer(f.nodes.get("editor-image"), 2);
  assert.equal(f.document.activeElement, null, "right-click does not move focus");
  f.setBusy(true);
  f.pointer(f.nodes.get("editor-image"));
  assert.equal(f.document.activeElement, null, "busy editor does not take focus");
});

test("pointer focus leaves controls and editable descendants alone", () => {
  const f = fixture(), root = f.nodes.get("reference-editor");
  for (const tag of ["button", "input", "textarea", "select", "summary", "a"]) {
    const control = f.document.createElement(tag);
    control.parentElement = root;
    control.focus();
    f.pointer(control);
    assert.equal(f.document.activeElement, control, `${tag} keeps native focus`);
  }
  const editable = f.document.createElement("div"), child = f.document.createElement("span");
  editable.setAttribute("contenteditable", "true");
  editable.append(child);
  editable.parentElement = root;
  child.focus();
  f.pointer(child);
  assert.equal(f.document.activeElement, child, "contenteditable descendant keeps focus");
});

test("Space selects and advances from zoom, navigation, and Done buttons", () => {
  for (const id of ["reference-zoom-in", "reference-box-next", "reference-done", "reference-box-toggle"]) {
    const f = fixture(), button = f.nodes.get(id), first = f.nodes.get("reference-crop").dataset.box;
    button.focus();
    assert.equal(f.key(" ", { target: button }), true, `${id} prevents native Space activation`);
    assert.deepEqual(plain(f.ref.boxes), [JSON.parse(first)], `${id} selects the displayed proposal`);
    assert.notEqual(f.nodes.get("reference-crop").dataset.box, first, `${id} advances the inspector`);
    assert.equal(f.done.length, 0, "Done was not activated by Space");
  }
});

test("keyboard shortcuts spare editable fields and respect hidden or busy editor state", () => {
  const f = fixture(), input = f.nodes.get("box-x1");
  for (const key of [" ", "ArrowRight", "x"]) {
    assert.equal(f.key(key, { target: input }), false, `${key} keeps input's native handling`);
    assert.equal(f.ref.boxes.length, 0);
    assert.equal(f.editor.data(f.ref).active, 0);
  }
  f.nodes.get("reference-editor").hidden = true;
  assert.equal(f.key(" "), false);
  f.nodes.get("reference-editor").hidden = false;
  f.nodes.get("reference-panel").hidden = true;
  assert.equal(f.key(" "), false);
  f.nodes.get("reference-panel").hidden = false;
  f.setBusy(true);
  assert.equal(f.key(" "), false);
  assert.equal(f.ref.boxes.length, 0);
});

test("all boxes render on the image, selected crops stay in the bottom strip, and inspector excludes selected duplicates", () => {
  const f = fixture({ boxes: [[100,100,150,150]] });
  assert.equal(f.nodes.has("candidate-list"), false);
  const groups = f.nodes.get("box-markers").children;
  assert.equal(groups.length, 3);
  assert.deepEqual(groups.map((group) => group.children[0].className),
    ["reference-box-marker", "reference-proposal-marker", "reference-proposal-marker"]);
  assert.ok(groups.every((group) => group.children.length === 1), "image boxes have no remove buttons");
  assert.ok(groups.every((group) => group.children[0].children.length === 0), "image boxes have no number labels");
  assert.deepEqual(groups.map((group) => group.children[0].attributes["aria-pressed"]), ["true", "false", "false"]);
  assert.deepEqual(groups.map((group) => group.children[0].dataset.index), ["0", "1", "2"]);
  const selected = f.nodes.get("reference-selected-list");
  assert.equal(selected.hidden, false);
  assert.equal(selected.children.length, 1);
  assert.equal(selected.children[0].children[0].className, "reference-selected-thumbnail");
  assert.equal(selected.children[0].children[1].className, "reference-box-remove");
  assert.equal(selected.children[0].children[0].children[0].src, "test.png");
  assert.equal(selected.children[0].children[0].dataset.box, "[100,100,150,150]");
  assert.equal(f.editor.data(f.ref).items.length, 3);
  assert.equal(f.nodes.get("reference-crop").dataset.box, "[300,100,350,150]");
  assert.equal(f.nodes.get("reference-box-position").textContent, "1 / 2");
});

test("selected thumbnails crop exactly to offset wide and tall boxes while the inspector retains context", () => {
  const boxes = [[120, 235, 440, 315], [860, 450, 920, 690]];
  const f = fixture({ boxes });
  const entries = f.nodes.get("reference-selected-list").children;
  for (const [index, [x1, y1, x2, y2]] of boxes.entries()) {
    const entry = entries[index], image = entry.children[0].children[0];
    const width = x2 - x1, height = y2 - y1;
    const scale = 72 / Math.max(width, height);
    assert.ok(Math.abs(parseFloat(entry.style.width) - width * scale) < 1e-9);
    assert.ok(Math.abs(parseFloat(entry.style.height) - height * scale) < 1e-9);
    assert.ok(Math.abs(parseFloat(image.style.width) - 2000 / width * 100) < 1e-9);
    assert.ok(Math.abs(parseFloat(image.style.left) + x1 / width * 100) < 1e-9);
    assert.ok(Math.abs(parseFloat(image.style.top) + y1 / height * 100) < 1e-9);
  }
  const inspector = f.nodes.get("reference-crop");
  const frame = f.nodes.get("reference-crop-box");
  assert.ok(parseFloat(inspector.querySelector("img").style.width) < 2000 / 50 * 100,
    "proposal preview keeps surrounding photo context");
  assert.ok(parseFloat(frame.style.left) > 0 && parseFloat(frame.style.top) > 0);
});

test("the selected strip is hidden with no selections and contains only selected boxes", () => {
  const f = fixture();
  const list = f.nodes.get("reference-selected-list");
  assert.equal(list.hidden, true);
  assert.equal(list.children.length, 0);
  assert.equal(f.nodes.get("box-markers").children.length, 3);
  f.key(" ");
  assert.equal(list.hidden, false);
  assert.equal(list.children.length, 1);
  assert.equal(list.children[0].children[0].dataset.box, "[100,100,150,150]");
  assert.equal(f.nodes.get("box-markers").children.length, 3);
});

test("clicking a selected thumbnail highlights its matching image box without changing the proposal queue", () => {
  const f = fixture({ boxes: [[100,100,150,150], [300,100,350,150]] });
  const list = f.nodes.get("reference-selected-list");
  const before = f.nodes.get("reference-crop").dataset.box;
  list.children[1].children[0].click();
  assert.equal(f.editor.data(f.ref).selectedActive, 1);
  assert.equal(f.nodes.get("box-markers").children[1].children[0].attributes["aria-current"], "true");
  assert.equal(list.children[1].children[0].attributes["aria-current"], "true");
  assert.equal(f.nodes.get("reference-crop").dataset.box, before);
  assert.equal(f.nodes.get("reference-box-position").textContent, "1 / 1");
});

test("thumbnail corner remove returns the box to proposals, and Undo restores its crop", () => {
  const f = fixture({ boxes: [[100,100,150,150]] });
  const list = f.nodes.get("reference-selected-list");
  list.children[0].children[1].click();
  assert.equal(f.ref.boxes.length, 0);
  assert.equal(list.hidden, true);
  assert.equal(list.children.length, 0);
  assert.equal(f.nodes.get("box-markers").children.length, 3);
  assert.equal(f.nodes.get("box-markers").children[0].children[0].className, "reference-proposal-marker");
  assert.equal(f.editor.data(f.ref).items.filter((item) => !item.selected).length, 3);
  f.editor.undo();
  assert.equal(f.ref.boxes.length, 1);
  assert.equal(list.hidden, false);
  assert.equal(list.children.length, 1);
  assert.equal(list.children[0].children[0].dataset.box, "[100,100,150,150]");
});

test("Space always selects the displayed proposal, X deselects the selected target without skipping", () => {
  const f = fixture();
  f.key("x");
  assert.equal(f.editor.data(f.ref).active, 0);
  f.key(" ");
  assert.deepEqual(plain(f.ref.boxes), [[100,100,150,150]]);
  assert.equal(f.editor.data(f.ref).active, 1);
  f.key(" ");
  assert.equal(f.ref.boxes.length, 2);
  assert.equal(f.nodes.get("reference-crop").dataset.box, "[500,100,550,150]");
  f.editor.activate(0);
  f.key("x");
  assert.deepEqual(plain(f.ref.boxes), [[300,100,350,150]]);
  assert.equal(f.editor.data(f.ref).active, 2, "deselect does not advance the inspector");
  f.key("ArrowLeft");
  assert.equal(f.editor.data(f.ref).active, 0, "arrows skip selected proposals");
  f.key(" ");
  assert.equal(f.editor.data(f.ref).active, 2);
});

test("selection limit and Done follow model policy; undo respects the current limit", () => {
  const candidates = Array.from({ length: 12 }, (_, index) => [index * 40, 100, index * 40 + 20, 120]);
  const f = fixture({ limit: 10, required: 10, candidates });
  for (let i = 0; i < 9; i++) f.key(" ");
  f.editor.action("done");
  assert.equal(f.done.length, 0);
  assert.equal(f.nodes.get("reference-done").disabled, true);
  f.key(" ");
  assert.equal(f.nodes.get("reference-done").disabled, false);
  f.key(" ");
  assert.equal(f.ref.boxes.length, 10);
  assert.equal(f.nodes.get("reference-box-toggle").disabled, true);
  f.editor.action("done"); assert.equal(f.done.length, 1);
  f.key("x"); assert.equal(f.ref.boxes.length, 9);
  assert.equal(f.nodes.get("reference-done").disabled, true);
  f.setExtra(1); f.editor.undo();
  assert.equal(f.ref.boxes.length, 9, "undo cannot restore a selection above the aggregate cap");
  f.setExtra(0); f.editor.undo(); assert.equal(f.ref.boxes.length, 10);
});

test("undo restores proposal position and selected target without stale focus", () => {
  const f = fixture();
  f.key(" "); f.key(" "); f.key(" ");
  assert.equal(f.editor.data(f.ref).active, -1);
  assert.equal(f.nodes.get("reference-inspector").hidden, true);
  f.key("x");
  assert.equal(f.editor.data(f.ref).active, 2);
  f.editor.undo();
  assert.equal(f.editor.data(f.ref).active, -1);
  assert.equal(f.editor.data(f.ref).selectedActive, 2);
  f.editor.undo();
  assert.equal(f.editor.data(f.ref).active, 2);
  assert.equal(f.editor.data(f.ref).selectedActive, 1);
});

test("selected-box adjustments are saved or cancelled while crop remains unselected", () => {
  const f = fixture();
  f.key(" "); f.key("e");
  const crop = f.nodes.get("reference-crop").dataset.box;
  f.key("ArrowRight");
  assert.equal(f.editor.data(f.ref).items[0].box[0], 101);
  assert.equal(f.nodes.get("reference-crop").dataset.box, crop);
  f.key("Escape");
  assert.equal(f.editor.data(f.ref).items[0].box[0], 100);
  f.key("e"); f.key("ArrowRight"); f.key("Enter");
  assert.equal(f.ref.boxes[0][0], 101);
  f.editor.undo(); assert.equal(f.ref.boxes[0][0], 100);
});

test("switching photos cancels edit and drawing, resets zoom, and preserves saved coordinates", () => {
  const f = fixture();
  f.key(" "); f.key("e"); f.key("ArrowRight");
  f.editor.zoom = 3;
  const other = { ...f.ref, boxes: [], candidates: [] };
  f.editor.render(other);
  assert.equal(f.editor.edit, null);
  assert.equal(f.editor.zoom, 1);
  assert.equal(f.editor.data(f.ref).items[0].box[0], 100);
  f.editor.render(f.ref);
  f.editor.startDrawing({ button: 0, pointerId: 2, clientX: 50, clientY: 50, preventDefault() {} });
  f.editor.render(other);
  assert.equal(f.editor.drawing, null);
});

test("drawing transforms current zoomed and scrolled overlay coordinates back to source pixels", () => {
  const f = fixture();
  f.nodes.get("box-overlay").rect = { left: -300, top: -200, width: 4000, height: 3200 };
  f.editor.startDrawing({ button: 0, pointerId: 2, clientX: 100, clientY: 100, preventDefault() {} });
  f.editor.finishDrawing({ pointerId: 2, clientX: 500, clientY: 500 });
  assert.deepEqual(plain(f.ref.boxes[0]), [200,150,400,350]);
});

test("drawing in Select areas mode records source coordinates without selecting a reference", () => {
  const f = fixture(), candidates = plain(f.ref.candidates);
  const button = f.nodes.get("reference-select-areas");
  button.click();
  assert.equal(f.editor.areaMode, true);
  f.nodes.get("box-overlay").rect = { left: -300, top: -200, width: 4000, height: 3200 };
  f.editor.startDrawing({ button: 0, pointerId: 2, clientX: 100, clientY: 100, preventDefault() {} });
  f.editor.finishDrawing({ pointerId: 2, clientX: 500, clientY: 500 });
  assert.deepEqual(plain(f.ref.detectionAreas), [[200,150,400,350]]);
  assert.deepEqual(plain(f.ref.boxes), []);
  assert.deepEqual(plain(f.ref.candidates), candidates);
  assert.equal(f.nodes.get("box-markers").children.length, candidates.length);
  const layer = f.nodes.get("detection-area-markers");
  assert.equal(layer.children.length, 1, "areas render in their own layer");
  assert.equal(layer.children[0].className, "reference-detection-area");
  assert.equal(layer.children[0].children[0].className, "reference-detection-area-remove");
  assert.equal(f.nodes.get("auto-detect-boxes").disabled, false);
});

test("area add and corner remove are undoable without altering selected boxes", () => {
  const selected = [[100,100,150,150]], f = fixture({ boxes: selected });
  assert.equal(f.nodes.get("auto-detect-boxes").disabled, true, "detection needs an area");
  f.editor.addArea(f.ref, [250,200,650,600]);
  assert.equal(f.nodes.get("auto-detect-boxes").disabled, false, "selected references do not block detection");
  assert.deepEqual(plain(f.ref.boxes), selected);
  f.nodes.get("detection-area-markers").children[0].children[0].click();
  assert.deepEqual(plain(f.ref.detectionAreas), []);
  f.editor.undo();
  assert.deepEqual(plain(f.ref.detectionAreas), [[250,200,650,600]]);
  f.editor.undo();
  assert.deepEqual(plain(f.ref.detectionAreas), []);
  assert.deepEqual(plain(f.ref.boxes), selected);
});

test("Escape leaves Select areas mode and a photo switch cancels it", () => {
  const f = fixture();
  f.nodes.get("reference-select-areas").click();
  assert.equal(f.editor.areaMode, true);
  assert.equal(f.key("Escape"), true);
  assert.equal(f.editor.areaMode, false);
  f.nodes.get("reference-select-areas").click();
  const other = { ...f.ref, boxes: [], candidates: [], detectionAreas: [] };
  f.editor.render(other);
  assert.equal(f.editor.areaMode, false);
  assert.deepEqual(plain(f.ref.detectionAreas), []);
});

test("zoom anchors the source point and remains unchanged during selection renders", () => {
  const f = fixture();
  f.editor.setZoom(2, 150, 100);
  assert.equal(f.editor.zoom, 2);
  assert.equal(f.nodes.get("reference-image-viewport").scrollLeft, 150);
  assert.equal(f.nodes.get("reference-image-viewport").scrollTop, 100);
  f.key(" "); assert.equal(f.editor.zoom, 2);
  f.editor.setZoom(100); assert.equal(f.editor.zoom, 16);
  f.editor.setZoom(0.2); assert.equal(f.editor.zoom, 1);
});

test("Alt+wheel changes the lens crop without moving the photo; wheel still changes photo zoom", () => {
  const f = fixture(), viewport = f.nodes.get("reference-image-viewport");
  assert.match(markup, /<kbd>Alt \+ Scroll<\/kbd> Magnifier/);
  const point = readyMagnifier(f);
  viewport.scrollLeft = 37; viewport.scrollTop = 19;
  f.editor.magnify(point);
  const cropWidth = () => f.canvasOps.filter((op) => op.type === "drawImage").at(-1).args[3];
  const before = cropWidth();
  assert.equal(wheel(f, -300, { altKey: true }), true);
  assert.ok(cropWidth() < before, "Alt+wheel in samples a smaller source area");
  assert.equal(f.editor.zoom, 1);
  assert.deepEqual([viewport.scrollLeft, viewport.scrollTop], [37, 19]);
  assert.equal(wheel(f, 600, { altKey: true }), true);
  assert.ok(cropWidth() > before, "Alt+wheel out samples a larger source area");
  wheel(f, -100000, { altKey: true }); assert.equal(f.editor.magnifierZoom, 8);
  wheel(f, 100000, { altKey: true }); assert.equal(f.editor.magnifierZoom, 0.5);
  f.editor.render(f.ref);
  f.editor.render({ ...f.ref, boxes: [], candidates: [] });
  assert.equal(f.editor.magnifierZoom, 0.5, "lens setting survives renders and photo switches");
  assert.equal(wheel(f, -260), true);
  assert.ok(f.editor.zoom > 1, "ordinary wheel zooms the photo");
  assert.equal(f.editor.magnifierZoom, 0.5);
});

test("Alt+wheel works during drawing while normal wheel and busy state remain blocked", () => {
  const f = fixture();
  readyMagnifier(f);
  f.editor.startDrawing({ button: 0, pointerId: 3, clientX: 50, clientY: 50, preventDefault() {} });
  const drawing = f.editor.drawing;
  assert.equal(wheel(f, -160), false);
  assert.equal(f.editor.zoom, 1);
  assert.equal(wheel(f, -160, { altKey: true }), true);
  assert.ok(f.editor.magnifierZoom > 1);
  assert.equal(f.editor.drawing, drawing, "changing the lens keeps the active box draft");
  f.setBusy(true);
  const lensZoom = f.editor.magnifierZoom;
  assert.equal(wheel(f, -160, { altKey: true }), false);
  assert.equal(f.editor.magnifierZoom, lensZoom);
});

test("magnifier draws full-span dotted crosslines with dark and light strokes", () => {
  const f = fixture();
  f.editor.magnify(readyMagnifier(f));
  assert.deepEqual(plain(f.canvasOps.find((op) => op.type === "dash")?.value), [1, 5]);
  const strokes = f.canvasOps.filter((op) => op.type === "stroke");
  assert.equal(strokes.length, 2);
  const cross = [["moveTo", 0, 72], ["lineTo", 144, 72], ["moveTo", 72, 0], ["lineTo", 72, 144]];
  assert.deepEqual(plain(strokes[0].path), cross);
  assert.deepEqual(plain(strokes[1].path), cross);
  assert.deepEqual(strokes.map(({ width, cap }) => [width, cap]), [[3, "round"], [1.5, "round"]]);
  assert.notEqual(strokes[0].color, strokes[1].color);
});

test("busy state cancels adjustments and blocks keyboard mutations", () => {
  const f = fixture();
  f.key(" "); f.key("e"); f.key("ArrowRight");
  f.setBusy(true); f.editor.render(f.ref);
  f.key("x"); f.key(" ");
  assert.equal(f.editor.edit, null);
  assert.equal(f.ref.boxes.length, 1);
  assert.equal(f.ref.boxes[0][0], 100);
});


test("drawing an existing proposal selects it once and duplicate selected boxes are ignored", () => {
  const f = fixture();
  f.editor.addBox(f.ref, [100,100,150,150]);
  assert.equal(f.editor.data(f.ref).items.length, 3);
  assert.equal(f.ref.boxes.length, 1);
  assert.equal(f.ref.candidates.length, 2);
  f.editor.addBox(f.ref, [100,100,150,150]);
  assert.equal(f.ref.boxes.length, 1);
  assert.match(f.messages.at(-1), /already selected/);
});

test("adjusting a selected box onto another selected box cannot save or finish", () => {
  const boxes = [[100,100,150,150], [101,100,151,150], ...Array.from({ length: 8 }, (_, i) => [300 + i * 60,100,350 + i * 60,150])];
  const f = fixture({ limit: 10, required: 10, boxes, candidates: [] });
  assert.equal(f.editor.valid(), true);
  f.editor.activate(0); f.key("e"); f.key("ArrowRight"); f.key("Enter");
  assert.notEqual(f.editor.edit, null, "duplicate save keeps adjustment open");
  assert.match(f.messages.at(-1), /matches an existing box/);
  assert.equal(f.editor.data(f.ref).history.length, 0, "failed saves do not create undo entries");
  assert.deepEqual(plain(f.ref.boxes), boxes, "committed boxes stay distinct");
  assert.equal(f.nodes.get("reference-done").disabled, true);
  f.editor.action("done"); assert.equal(f.done.length, 0);
  f.key("ArrowRight"); f.key("Enter");
  assert.equal(f.editor.edit, null, "corrected coordinates can be saved");
  assert.deepEqual(plain(f.ref.boxes[0]), [102,100,152,150]);
  assert.equal(f.messages.at(-1), "");
  assert.equal(f.editor.valid(), true);
  f.editor.undo(); assert.deepEqual(plain(f.ref.boxes), boxes);
});

test("adjusting an unselected proposal onto a selected box keeps edit open and can cancel", () => {
  const f = fixture({ boxes: [[101,100,151,150]], candidates: [[100,100,150,150]] });
  f.editor.action("adjust-candidate"); f.key("ArrowRight"); f.key("Enter");
  assert.notEqual(f.editor.edit, null);
  assert.match(f.messages.at(-1), /matches an existing box/);
  assert.deepEqual(plain(f.ref.candidates), [[100,100,150,150]]);
  assert.deepEqual(plain(f.ref.boxes), [[101,100,151,150]]);
  f.key("Escape");
  assert.equal(f.editor.edit, null);
  assert.deepEqual(plain(f.editor.data(f.ref).items[0].box), [100,100,150,150]);
  assert.equal(f.editor.data(f.ref).history.length, 0);
  assert.equal(f.messages.at(-1), "");
});

test("adjusting a selected box onto an unselected proposal also blocks saving", () => {
  const f = fixture({ boxes: [[100,100,150,150]], candidates: [[101,100,151,150]] });
  f.editor.activate(1); f.key("e"); f.key("ArrowRight"); f.key("Enter");
  assert.notEqual(f.editor.edit, null);
  assert.match(f.messages.at(-1), /matches an existing box/);
  assert.deepEqual(plain(f.ref.boxes), [[100,100,150,150]]);
});

test("exact-count validation rejects malformed selected arrays with duplicate coordinates", () => {
  const boxes = Array.from({ length: 10 }, (_, i) => [100 + i * 60,100,150 + i * 60,150]);
  boxes[9] = [...boxes[0]];
  const f = fixture({ limit: 10, required: 10, boxes, candidates: [] });
  assert.equal(f.ref.boxes.length, 10);
  assert.equal(f.editor.valid(), false);
  assert.equal(f.nodes.get("reference-done").disabled, true);
  f.editor.action("done"); assert.equal(f.done.length, 0);
});
