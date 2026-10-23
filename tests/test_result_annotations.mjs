import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../frontend/public/result-annotations.js", import.meta.url), "utf8");

test("result overlay uses original pixels and detection coordinates; JPEG is composed only on download", async () => {
  const created = [];
  const calls = [];
  const links = [];
  const revoked = [];
  const timers = [];
  class Element {
    constructor(tag) {
      this.tag = tag;
      this.children = [];
      this.classList = { add: (...names) => { this.classes = [...(this.classes || []), ...names]; } };
    }
    setAttribute(name, value) { this[name] = value; }
    append(...children) { this.children.push(...children); }
    click() { links.push({ href: this.href, download: this.download }); }
    getContext() { return {
      drawImage: (...args) => calls.push(["drawImage", ...args]),
      strokeRect: (...args) => calls.push(["strokeRect", ...args]),
      measureText: (text) => ({ width: text.length * 8 }),
      fillRect: (...args) => calls.push(["fillRect", ...args]),
      fillText: (...args) => calls.push(["fillText", ...args]),
    }; }
    toBlob(resolve, type, quality) { calls.push(["toBlob", type, quality, this.width, this.height]); resolve({ type }); }
  }
  const context = {
    document: {
      createElementNS: (_ns, tag) => { const item = new Element(tag); created.push(item); return item; },
      createElement: (tag) => { const item = new Element(tag); created.push(item); return item; },
    },
    Image: class { async decode() { calls.push(["decode", this.src]); } },
    URL: { createObjectURL: () => "blob:download", revokeObjectURL: (url) => revoked.push(url) },
    setTimeout: (callback, delay) => timers.push({ callback, delay }),
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  const result = { filename: "apples.jpg", count: 2, width: 1200, height: 800,
    detections: [{ box: [10, 20, 110, 220] }, { box: [600, 200, 700, 300] }] };
  const svg = context.ResultAnnotations.render(result, "/original.jpg");
  assert.equal(svg.viewBox, "0 0 1200 800");
  assert.equal(svg.children[0].tag, "image");
  assert.equal(svg.children[0].href, "/original.jpg");
  assert.equal(svg.children[1].x, 10);
  assert.equal(svg.children[1].y, 20);
  assert.equal(svg.children[1].width, 100);
  assert.equal(svg.children[1].height, 200);
  assert.equal(svg.children[2].x, 600);
  assert.equal(svg.children[3].textContent, "1");
  assert.equal(svg.children[4].textContent, "2");
  assert.equal(calls.length, 0, "viewing results must not render or download a JPEG");

  await context.ResultAnnotations.download(result, "/original.jpg");
  assert.deepEqual(calls.filter(([kind]) => kind === "strokeRect").map(([, ...box]) => box),
    [[10, 20, 100, 200], [600, 200, 100, 100]]);
  assert.deepEqual(calls.find(([kind]) => kind === "toBlob"), ["toBlob", "image/jpeg", .94, 1200, 800]);
  assert.deepEqual(links, [{ href: "blob:download", download: "apples-counted.jpg" }]);
  assert.equal(timers[0].delay, 30_000);
  timers[0].callback();
  assert.deepEqual(revoked, ["blob:download"]);
});
