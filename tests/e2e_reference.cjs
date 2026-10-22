/* Real browser coverage for automatic reference proposals. No detector responses are mocked. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright-core");

const root = path.resolve(__dirname, "..");
const output = process.env.E2E_OUTPUT_DIR ? path.resolve(process.env.E2E_OUTPUT_DIR) :
  path.join(root, "test-results/reference-auto/browser");
const attachments = path.join(root, ".codex-remote-attachments/01a14b6c-1e76-7570-b308-4ac2aa56b41d/e1ec69e0-b412-4636-a935-0b80fa699bcc");
const ringReference = path.join(attachments, "1-6005.jpg");
const secondRingReference = path.join(root, "test-results/ring-debug-input/reference-original.jpg");
const ringTarget = path.join(attachments, "2-6006.jpg");
const fittingReference = path.join(root, "test-results/fitting-debug/reference-original.jpg");
const fittingTarget = process.env.REFERENCE_FITTING_TARGET || path.join(root, "test-results/fitting-debug/target-at-640.jpg");
const base = process.env.APP_URL || "http://localhost:8080";
const edge = process.env.EDGE_EXE || "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const flow = process.env.E2E_REFERENCE_FLOW || "all";
const report = { base, flow, startedAt: new Date().toISOString(), ring: {}, fitting: {}, errors: [] };

fs.mkdirSync(output, { recursive: true });
const save = (name, data) => fs.writeFileSync(path.join(output, name), JSON.stringify(data, null, 2));
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function settled(context, id, timeout = 600000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const response = await context.request.get(`${base}/api/sessions/${encodeURIComponent(id)}`);
    assert.equal(response.status(), 200, `Read count ${id}`);
    const session = await response.json();
    const batch = session.batches[0];
    assert.notEqual(batch?.status, "failed", batch?.error);
    if (batch?.status === "complete") return session;
    await pause(1500);
  }
  throw new Error(`Counting timed out for ${id}`);
}

async function noOverflow(page, label) {
  const bounds = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    document: document.documentElement.scrollWidth,
    body: document.body.scrollWidth,
  }));
  assert.ok(bounds.document <= bounds.viewport + 1 && bounds.body <= bounds.viewport + 1,
    `${label} overflow: ${JSON.stringify(bounds)}`);
  return bounds;
}

function matchingBox(proposal, [width, height], exemplar, label) {
  assert.equal(proposal.width, width, `${label} normalized width`);
  assert.equal(proposal.height, height, `${label} normalized height`);
  assert.equal(proposal.model, "fastsam-s");
  assert.ok(Array.isArray(proposal.boxes) && proposal.boxes.length > 0, `${label} should have suggested objects`);
  for (const box of proposal.boxes) {
    assert.ok(Array.isArray(box) && box.length === 4 && box.every(Number.isFinite), `${label} finite box`);
    const [x1, y1, x2, y2] = box;
    assert.ok(0 <= x1 && x1 < x2 && x2 <= width && 0 <= y1 && y1 < y2 && y2 <= height,
      `${label} in-bounds box: ${box}`);
  }
  if (!exemplar) return null;
  const [ex1, ey1, ex2, ey2] = exemplar;
  const exemplarArea = (ex2 - ex1) * (ey2 - ey1);
  const matches = proposal.boxes.map(([x1, y1, x2, y2], index) => ({
    index,
    area: (x2 - x1) * (y2 - y1),
    overlap: Math.max(0, Math.min(x2, ex2) - Math.max(x1, ex1)) *
      Math.max(0, Math.min(y2, ey2) - Math.max(y1, ey1)),
  })).filter(({ area, overlap }) => area < width * height * .2 && overlap >= exemplarArea * .5)
    .sort((a, b) => b.overlap / (b.area + exemplarArea - b.overlap) - a.overlap / (a.area + exemplarArea - a.overlap));
  assert.ok(matches.length, `${label} should include the known object`);
  return matches[0].index;
}

async function uploadReference(page, file, name, dimensions, exemplar) {
  const detection = page.waitForResponse((response) => response.url().endsWith("/api/reference-boxes") &&
    response.request().method() === "POST", { timeout: 240000 });
  await page.locator("#reference-gallery").setInputFiles(file);
  await page.locator("#reference-editor").waitFor({ state: "visible" });
  const response = await detection;
  assert.equal(response.status(), 200, `${name} detector: ${await response.text()}`);
  const proposal = await response.json();
  save(`${name}-proposal.json`, proposal);
  const match = matchingBox(proposal, dimensions, exemplar, name);
  await page.waitForFunction(() => !document.querySelector("#start-button").disabled &&
    document.querySelector("#auto-detect-boxes").getAttribute("aria-busy") !== "true", null, { timeout: 240000 });
  assert.equal(await page.locator(".reference-box-marker").count(), 0, `${name} proposals start unselected`);
  const crop = page.locator("#reference-crop");
  for (const [index, box] of proposal.boxes.entries()) {
    assert.equal(await crop.getAttribute("data-index"), String(index), `${name} proposal order`);
    assert.deepEqual(JSON.parse(await crop.getAttribute("data-box")), box, `${name} inspector matches detector response`);
    assert.equal(await page.locator("#reference-box-prev").isDisabled(), index === 0, "previous proposal clamps at first");
    assert.equal(await page.locator("#reference-box-next").isDisabled(), index === proposal.boxes.length - 1, "next proposal clamps at last");
    if (index < proposal.boxes.length - 1) await page.locator("#reference-box-next").click();
  }
  assert.equal(await page.locator("#auto-detect-boxes").isDisabled(), false);
  await page.screenshot({ path: path.join(output, `${name}-suggestions.png`), fullPage: true });
  return { proposal, match };
}

async function selectCandidate(page, proposal, index, label) {
  const crop = page.locator("#reference-crop");
  for (let step = 0; step < proposal.boxes.length; step++) {
    const current = Number(await crop.getAttribute("data-index"));
    if (current === index) break;
    const arrow = page.locator(current < index ? "#reference-box-next" : "#reference-box-prev");
    assert.equal(await arrow.isDisabled(), false, `${label} is reachable among unselected proposals`);
    await arrow.click();
  }
  assert.equal(await crop.getAttribute("data-index"), String(index), `${label} proposal focused`);
  assert.deepEqual(JSON.parse(await crop.getAttribute("data-box")), proposal.boxes[index]);
  await page.locator("#reference-box-toggle").click();
  const marker = page.locator(`.reference-box-marker[data-index="${index}"]`);
  assert.deepEqual(JSON.parse(await marker.getAttribute("data-box")), proposal.boxes[index], `${label} selected`);
  assert.notEqual(await crop.getAttribute("data-box"), JSON.stringify(proposal.boxes[index]), "selected box leaves the inspector");
  return proposal.boxes[index];
}

async function ringFlow(browser) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  try {
    const page = await context.newPage();
    page.on("pageerror", (error) => report.errors.push(`ring: ${error.message}`));
    await page.goto(base);
    await page.locator('#model-select option[value="yoloe-26n-seg"]').waitFor({ state: "attached" });
    await page.locator("#model-select").selectOption("yoloe-26n-seg");
    const { proposal: first, match: firstIndex } = await uploadReference(page, ringReference, "ring", [721, 1280], [294, 586, 436, 727]);
    report.ring.firstBoxes = first.boxes.length;
    await selectCandidate(page, first, firstIndex, "first ring");
    await page.locator(`.reference-box-remove[data-index="${firstIndex}"]`).click();
    assert.equal(await page.locator(".reference-box-marker").count(), 0, "a selected suggestion returns to proposals");

    await page.locator(".coordinate-entry summary").click();
    for (const [id, value] of Object.entries({ "box-x1": 294, "box-y1": 586, "box-x2": 436, "box-y2": 727 }))
      await page.locator(`#${id}`).fill(String(value));
    await page.locator("#add-coordinate-box").click();
    assert.equal(await page.locator(".reference-box-marker").count(), 1, "manual fallback remains available");
    assert.deepEqual(JSON.parse(await page.locator(".reference-box-marker").getAttribute("data-box")),
      [294, 586, 436, 727], "manual box matches entered coordinates");
    assert.equal(await page.locator("#auto-detect-boxes").isDisabled(), true, "manual boxes are preserved");
    await page.locator("#undo-box").click();
    assert.equal(await page.locator(".reference-box-marker").count(), 0);

    const retryResponse = page.waitForResponse((response) => response.url().endsWith("/api/reference-boxes") &&
      response.request().method() === "POST", { timeout: 240000 });
    await page.locator("#auto-detect-boxes").click();
    const retry = await retryResponse;
    assert.equal(retry.status(), 200, `Ring retry: ${await retry.text()}`);
    const accepted = await retry.json();
    save("ring-retry-proposal.json", accepted);
    const ringIndex = matchingBox(accepted, [721, 1280], [294, 586, 436, 727], "ring retry");
    await page.waitForFunction(() => !document.querySelector("#start-button").disabled, null, { timeout: 240000 });
    assert.equal(await page.locator(".reference-box-marker").count(), 0, "retry leaves proposals unselected");
    assert.deepEqual(JSON.parse(await page.locator("#reference-crop").getAttribute("data-box")),
      accepted.boxes[0], "retry replaces inspector proposals");
    const chosenRing = await selectCandidate(page, accepted, ringIndex, "ring");
    await page.locator("#reference-done").click();
    assert.equal(await page.locator("#reference-editor").isHidden(), true, "Done collapses first review");
    await page.screenshot({ path: path.join(output, "ring-retry-selected.png"), fullPage: true });
    const { proposal: second, match: secondIndex } = await uploadReference(page, secondRingReference, "ring-second", [2252, 4000], [808, 1640, 1512, 2340]);
    assert.equal(await page.locator("#reference-thumbs button").count(), 2, "multiple references receive independent proposals");
    report.ring.references = 2;
    await page.setViewportSize({ width: 390, height: 844 });
    report.ring.mobileOverflow = await noOverflow(page, "ring editor at 390px");
    await page.screenshot({ path: path.join(output, "ring-mobile-390.png"), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });

    await page.locator("#target-gallery").setInputFiles(ringTarget);
    await page.waitForFunction(() => document.querySelectorAll("#target-queue .queue-item").length === 1);
    await page.locator("#start-button").click();
    assert.equal(await page.locator("#notice").textContent(), "Select the object(s) to use as references.");
    const chosenSecondRing = await selectCandidate(page, second, secondIndex, "second ring");
    await page.locator("#reference-done").click();
    await page.locator("#object-label").fill("white plastic ring");
    const created = page.waitForResponse((response) => response.url().endsWith("/api/sessions") &&
      response.request().method() === "POST", { timeout: 60000 });
    await page.locator("#start-button").click();
    const creation = await created;
    assert.equal(creation.status(), 201, await creation.text());
    const initial = await creation.json();
    save("ring-created.json", initial);
    assert.equal(initial.model, "yoloe-26n-seg");
    assert.deepEqual(initial.references[0].boxes, [chosenRing], "YOLOE receives only the selected first-ring box");
    assert.equal(initial.references.length, 2);
    assert.deepEqual(initial.references[1].boxes, [chosenSecondRing], "YOLOE receives only the selected second-ring box");
    report.ring.sessionId = initial.id;
    const session = await settled(context, initial.id);
    save("ring-session.json", session);
    const result = session.batches[0].results[0];
    assert.equal(result.count, result.detections.length);
    if (!result.count) assert.match(result.warning, /No matches detected/, "zero matches remain flagged for review");
    report.ring.count = result.count;
    await page.waitForFunction(() => document.querySelectorAll(".result-card").length === 1, null, { timeout: 60000 });
    await page.screenshot({ path: path.join(output, "ring-results.png"), fullPage: true });
    await page.locator("#finish-button").click();
    await page.locator("#new-count-button").waitFor({ state: "visible", timeout: 60000 });
    report.ring.passed = true;
  } finally { await context.close(); }
}

async function fittingFlow(browser) {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 1 });
  try {
    const page = await context.newPage();
    page.on("pageerror", (error) => report.errors.push(`fitting: ${error.message}`));
    await page.goto(base);
    await page.locator('#model-select option[value="yoloe-26n-seg"]').waitFor({ state: "attached" });
    await page.locator("#model-select").selectOption("yoloe-26n-seg");
    const { proposal, match } = await uploadReference(page, fittingReference, "fitting", [2252, 4000], [1062, 1825, 1295, 2125]);
    report.fitting.boxes = proposal.boxes.length;
    report.fitting.mobileOverflow = await noOverflow(page, "fitting editor at 390px");
    await page.screenshot({ path: path.join(output, "fitting-mobile-390.png"), fullPage: true });

    await page.locator("#target-gallery").setInputFiles(fittingTarget);
    await page.waitForFunction(() => document.querySelectorAll("#target-queue .queue-item").length === 1);
    await page.locator("#start-button").click();
    assert.equal(await page.locator("#notice").textContent(), "Select the object(s) to use as references.");
    const chosenFitting = await selectCandidate(page, proposal, match, "fitting");
    const created = page.waitForResponse((response) => response.url().endsWith("/api/sessions") &&
      response.request().method() === "POST", { timeout: 60000 });
    await page.locator("#start-button").click();
    const creation = await created;
    assert.equal(creation.status(), 201, await creation.text());
    const initial = await creation.json();
    save("fitting-created.json", initial);
    assert.equal(initial.model, "yoloe-26n-seg");
    assert.equal(initial.label, "", "reference counting accepts a blank description");
    assert.deepEqual(initial.references[0].boxes, [chosenFitting], "YOLOE receives only the selected fitting box");
    report.fitting.sessionId = initial.id;
    const session = await settled(context, initial.id);
    save("fitting-session.json", session);
    const result = session.batches[0].results[0];
    assert.equal(result.count, result.detections.length);
    if (result.count === 0) assert.match(result.warning, /No matches detected/i);
    report.fitting.count = result.count;
    await page.waitForFunction(() => document.querySelectorAll(".result-card").length === 1, null, { timeout: 60000 });
    report.fitting.resultOverflow = await noOverflow(page, "fitting results at 390px");
    await page.screenshot({ path: path.join(output, "fitting-mobile-results.png"), fullPage: true });
    await page.locator("#finish-button").click();
    await page.locator("#new-count-button").waitFor({ state: "visible", timeout: 60000 });
    report.fitting.passed = true;
  } finally { await context.close(); }
}

(async () => {
  assert.ok(["all", "ring", "fitting"].includes(flow), `Unknown E2E_REFERENCE_FLOW: ${flow}`);
  assert.ok(fs.existsSync(edge), `Edge executable not found: ${edge}`);
  const browser = await chromium.launch({ executablePath: edge, headless: true });
  try {
    if (flow !== "fitting") await ringFlow(browser);
    if (flow !== "ring") await fittingFlow(browser);
    assert.deepEqual(report.errors, [], "browser page errors");
    report.passed = true;
    console.log(`[reference-auto] PASS: ring ${report.ring.count ?? "skipped"}, fitting ${report.fitting.count ?? "skipped"}`);
  } finally { await browser.close(); }
})().catch((error) => { report.error = error.stack || String(error); console.error(report.error); process.exitCode = 1; })
  .finally(() => { report.finishedAt = new Date().toISOString(); save("browser-report.json", report); });
