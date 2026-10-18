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

function proposalQuality(proposal, [width, height], exemplar, label) {
  assert.equal(proposal.width, width, `${label} normalized width`);
  assert.equal(proposal.height, height, `${label} normalized height`);
  assert.equal(proposal.model, "countgd-plusplus");
  assert.ok(Array.isArray(proposal.boxes) && proposal.boxes.length > 0, `${label} should have suggested objects`);
  for (const box of proposal.boxes) {
    assert.ok(Array.isArray(box) && box.length === 4 && box.every(Number.isFinite), `${label} finite box`);
    const [x1, y1, x2, y2] = box;
    assert.ok(0 < x1 && x1 < x2 && x2 < width && 0 < y1 && y1 < y2 && y2 < height,
      `${label} box avoids the image edge: ${box}`);
    assert.ok((x2 - x1) * (y2 - y1) < width * height * .2,
      `${label} excludes a full-frame background box: ${box}`);
  }
  if (!exemplar) return;
  const [ex1, ey1, ex2, ey2] = exemplar;
  const exemplarArea = (ex2 - ex1) * (ey2 - ey1);
  assert.ok(proposal.boxes.some(([x1, y1, x2, y2]) =>
    Math.max(0, Math.min(x2, ex2) - Math.max(x1, ex1)) *
    Math.max(0, Math.min(y2, ey2) - Math.max(y1, ey1)) >= exemplarArea * .85),
  `${label} should include the known object`);
}

async function uploadReference(page, file, name, dimensions, exemplar) {
  const detection = page.waitForResponse((response) => response.url().endsWith("/api/reference-boxes") &&
    response.request().method() === "POST", { timeout: 240000 });
  await page.locator("#reference-gallery").setInputFiles(file);
  await page.locator("#reference-editor").waitFor({ state: "visible" });
  await page.waitForFunction(() => document.querySelector("#auto-detect-boxes").textContent === "Detecting…", null, { timeout: 60000 });
  assert.equal(await page.locator("#start-button").isDisabled(), true, "counting is locked during detection");
  assert.equal(await page.locator("#remove-reference").isDisabled(), true, "reference edits are locked during detection");
  assert.equal(await page.locator("#add-coordinate-box").isDisabled(), true, "manual edits are locked during detection");
  assert.equal(await page.locator("#target-gallery").isDisabled(), true, "uploads are locked during detection");
  const response = await detection;
  assert.equal(response.status(), 200, `${name} detector: ${await response.text()}`);
  const proposal = await response.json();
  save(`${name}-proposal.json`, proposal);
  proposalQuality(proposal, dimensions, exemplar, name);
  await page.waitForFunction(() => !document.querySelector("#start-button").disabled &&
    document.querySelector("#auto-detect-boxes").textContent !== "Detecting…", null, { timeout: 240000 });
  const overlay = await page.locator("#box-overlay rect:not(.draft)").evaluateAll((rects) => rects.map((rect) => {
    const x1 = Number(rect.getAttribute("x")), y1 = Number(rect.getAttribute("y"));
    return [x1, y1, x1 + Number(rect.getAttribute("width")), y1 + Number(rect.getAttribute("height"))];
  }));
  assert.deepEqual(overlay, proposal.boxes.slice(0, 32), `${name} visible boxes match detector response`);
  assert.equal(await page.locator("#box-markers .box-marker").count(), overlay.length, "numbered markers match boxes");
  assert.equal(await page.locator("#box-list button").count(), overlay.length, "each box can be removed");
  assert.equal(await page.locator("#auto-detect-boxes").isDisabled(), true, "accepted boxes cannot be replaced implicitly");
  await page.screenshot({ path: path.join(output, `${name}-suggestions.png`), fullPage: true });
  return proposal;
}

async function ringFlow(browser) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  try {
    const page = await context.newPage();
    page.on("pageerror", (error) => report.errors.push(`ring: ${error.message}`));
    await page.goto(base);
    await page.locator('#model-select option[value="countgd-plusplus"]').waitFor({ state: "attached" });
    await page.locator("#object-label").fill("white plastic ring");
    const first = await uploadReference(page, ringReference, "ring", [721, 1280], [294, 586, 436, 727]);
    report.ring.firstBoxes = first.boxes.length;

    await page.locator("#box-list button").first().click();
    assert.equal(await page.locator("#box-list button").count(), Math.min(first.boxes.length, 32) - 1);
    while (await page.locator("#box-list button").count()) await page.locator("#box-list button").first().click();
    assert.equal(await page.locator("#box-markers .box-marker").count(), 0);
    assert.equal(await page.locator("#auto-detect-boxes").isDisabled(), false);

    await page.locator(".coordinate-entry summary").click();
    for (const [id, value] of Object.entries({ "box-x1": 294, "box-y1": 586, "box-x2": 436, "box-y2": 727 }))
      await page.locator(`#${id}`).fill(String(value));
    await page.locator("#add-coordinate-box").click();
    assert.equal(await page.locator("#box-list button").count(), 1, "manual fallback remains available");
    assert.equal(await page.locator("#auto-detect-boxes").isDisabled(), true, "manual boxes are preserved");
    await page.locator("#undo-box").click();
    assert.equal(await page.locator("#box-list button").count(), 0);

    const retryResponse = page.waitForResponse((response) => response.url().endsWith("/api/reference-boxes") &&
      response.request().method() === "POST", { timeout: 240000 });
    await page.locator("#auto-detect-boxes").click();
    assert.equal(await page.locator("#start-button").isDisabled(), true, "retry locks counting");
    const retry = await retryResponse;
    assert.equal(retry.status(), 200, `Ring retry: ${await retry.text()}`);
    const accepted = await retry.json();
    save("ring-retry-proposal.json", accepted);
    proposalQuality(accepted, [721, 1280], [294, 586, 436, 727], "ring retry");
    await page.waitForFunction(() => !document.querySelector("#start-button").disabled, null, { timeout: 240000 });
    assert.equal(await page.locator("#box-list button").count(), Math.min(accepted.boxes.length, 32));
    await page.screenshot({ path: path.join(output, "ring-retry-accepted.png"), fullPage: true });
    const second = await uploadReference(page, secondRingReference, "ring-second", [2252, 4000], null);
    assert.equal(await page.locator("#reference-thumbs button").count(), 2, "multiple references receive independent proposals");
    report.ring.references = 2;
    await page.setViewportSize({ width: 390, height: 844 });
    report.ring.mobileOverflow = await noOverflow(page, "ring editor at 390px");
    await page.screenshot({ path: path.join(output, "ring-mobile-390.png"), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });

    await page.locator("#target-gallery").setInputFiles(ringTarget);
    const created = page.waitForResponse((response) => response.url().endsWith("/api/sessions") &&
      response.request().method() === "POST", { timeout: 60000 });
    await page.locator("#start-button").click();
    const creation = await created;
    assert.equal(creation.status(), 201, await creation.text());
    const initial = await creation.json();
    save("ring-created.json", initial);
    assert.equal(initial.model, "countgd-plusplus");
    assert.deepEqual(initial.references[0].boxes, accepted.boxes.slice(0, 32), "CountGD receives accepted automatic boxes");
    assert.equal(initial.references.length, 2);
    assert.deepEqual(initial.references[1].boxes, second.boxes.slice(0, 32), "second reference preserves its original coordinates");
    report.ring.sessionId = initial.id;
    const session = await settled(context, initial.id);
    save("ring-session.json", session);
    const result = session.batches[0].results[0];
    assert.equal(result.count, result.detections.length);
    assert.ok(result.count > 0, "CountGD counts rings with automatically accepted references");
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
    await page.locator("#object-label").fill("Bolt with red cap");
    const proposal = await uploadReference(page, fittingReference, "fitting", [2252, 4000], [1062, 1825, 1295, 2125]);
    report.fitting.boxes = proposal.boxes.length;
    report.fitting.mobileOverflow = await noOverflow(page, "fitting editor at 390px");
    await page.screenshot({ path: path.join(output, "fitting-mobile-390.png"), fullPage: true });

    await page.locator("#target-gallery").setInputFiles(fittingTarget);
    const created = page.waitForResponse((response) => response.url().endsWith("/api/sessions") &&
      response.request().method() === "POST", { timeout: 60000 });
    await page.locator("#start-button").click();
    const creation = await created;
    assert.equal(creation.status(), 201, await creation.text());
    const initial = await creation.json();
    save("fitting-created.json", initial);
    assert.equal(initial.model, "yoloe-26n-seg");
    assert.deepEqual(initial.references[0].boxes, proposal.boxes.slice(0, 32), "YOLOE receives accepted automatic boxes");
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
