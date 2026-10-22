/* Real browser smoke test. Run with PLAYWRIGHT_MODULE pointing to a temporary playwright-core install. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright-core");

const base = process.env.APP_URL || "http://localhost:8080";
const edge = process.env.EDGE_EXE || "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const results = path.resolve(__dirname, "../test-results");
const bus = path.join(results, "bus.jpg");
const mirror = path.join(results, "bus-mirror.jpg");
const resized = path.join(results, "bus-resized.jpg");
const longReference = path.join(results, `reference-${"x".repeat(105)}.jpg`);
const report = { base, startedAt: new Date().toISOString(), desktop: {}, mobile: {}, browserErrors: [], resourceWarnings: [], referenceWarnings: [], httpErrors: [] };
fs.mkdirSync(results, { recursive: true });

const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const log = (message) => console.log(`[e2e] ${message}`);

async function waitForSession(id, expectedBatches, timeout = 240000) {
  const until = Date.now() + timeout;
  while (Date.now() < until) {
    const response = await fetch(`${base}/api/sessions/${encodeURIComponent(id)}`);
    assert.equal(response.status, 200, `GET session ${id}`);
    const session = await response.json();
    if (session.batches.some((batch) => batch.status === "failed")) {
      throw new Error(`Batch failed: ${session.batches.find((batch) => batch.error)?.error}`);
    }
    if (session.batches.length >= expectedBatches && session.batches.slice(0, expectedBatches).every((batch) => batch.status === "complete")) return session;
    await pause(1500);
  }
  throw new Error(`Timed out waiting for ${expectedBatches} completed batches in ${id}`);
}

async function waitForCards(page, count) {
  await page.waitForFunction((expected) => document.querySelectorAll(".result-card").length === expected, count, { timeout: 30000 });
}

async function clearSuggestedBoxes(page, detection) {
  const response = await detection;
  assert.equal(response.status(), 200, `Automatic reference boxes: ${await response.text()}`);
  await page.waitForFunction(() => document.querySelector("#auto-detect-boxes").getAttribute("aria-busy") !== "true", null, { timeout: 240000 });
  while (await page.locator(".reference-box-remove").count()) {
    await page.locator(".reference-box-remove").first().click();
  }
  assert.equal(await page.locator(".reference-box-marker").count(), 0, "automatic selections cleared");
}

async function noHorizontalOverflow(page, label) {
  const bounds = await page.evaluate(() => ({ viewport: document.documentElement.clientWidth, document: document.documentElement.scrollWidth, body: document.body.scrollWidth }));
  if (bounds.document > bounds.viewport + 1 || bounds.body > bounds.viewport + 1) {
    const offenders = await page.evaluate(() => [...document.querySelectorAll("body *")].map((element) => ({
      tag: element.tagName.toLowerCase(), id: element.id, className: typeof element.className === "string" ? element.className : "",
      text: element.textContent.trim().slice(0, 60), right: Math.round(element.getBoundingClientRect().right), width: Math.round(element.getBoundingClientRect().width),
    })).filter((element) => element.right > document.documentElement.clientWidth + 2).sort((a, b) => b.right - a.right).slice(0, 15));
    await page.screenshot({ path: path.join(results, "mobile-overflow.png"), fullPage: true });
    throw new Error(`${label} overflow: ${JSON.stringify({ bounds, offenders })}`);
  }
  return bounds;
}

async function openPreview(page, result, screenshot) {
  await page.locator(".result-image").first().click();
  await page.locator("#image-dialog").waitFor({ state: "visible" });
  await page.locator("#dialog-image").evaluate((image) => image.decode());
  const dimensions = await page.locator("#dialog-image").evaluate((image) => ({ width: image.naturalWidth, height: image.naturalHeight, complete: image.complete }));
  assert.ok(dimensions.complete && dimensions.width > 0 && dimensions.height > 0, "annotated preview loaded");
  assert.equal(dimensions.width, result.width, "full annotated image width");
  assert.equal(dimensions.height, result.height, "full annotated image height");
  await page.screenshot({ path: path.join(results, screenshot) });
  await page.locator("#close-dialog").click();
  return dimensions;
}

async function desktopFlow(browser) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true });
  const page = await context.newPage();
  page.on("pageerror", (error) => report.browserErrors.push(`desktop pageerror: ${error.message}`));
  page.on("console", (entry) => { if (entry.type() === "error") {
    const item = `desktop console: ${entry.text()} at ${entry.location().url}`;
    (entry.location().url.endsWith("/api/reference-boxes") ? report.referenceWarnings :
      entry.location().url.endsWith("/favicon.ico") ? report.resourceWarnings : report.browserErrors).push(item);
  } });
  page.on("response", (response) => { if (response.status() >= 400) (response.url().endsWith("/api/reference-boxes") ? report.referenceWarnings : report.httpErrors).push(`desktop HTTP ${response.status()}: ${response.url()}`); });
  try {
    log("Desktop: loading setup");
    await page.goto(base, { waitUntil: "networkidle" });
    await page.locator('#model-select option[value="yoloe-26n-seg"]').waitFor({ state: "attached" });
    assert.equal(await page.locator("#model-select").inputValue(), "countgd-plusplus", "CountGD++ is the default trial model");
    assert.equal(await page.locator("#confidence").inputValue(), "0.23", "CountGD++ uses its official threshold");
    const catalogIds = await page.locator("#model-select option").evaluateAll((options) => options.map((option) => option.value));
    for (const id of ["countgd-plusplus", ...["n", "s", "m", "l", "x"].map((size) => `yoloe-26${size}-seg`)]) {
      assert.ok(catalogIds.includes(id), `${id} appears in the model catalog`);
    }
    await page.screenshot({ path: path.join(results, "desktop-initial.png"), fullPage: true });
    report.desktop.initialOverflow = await noHorizontalOverflow(page, "desktop setup");

    await page.locator("#model-select").selectOption("yoloe-26n-seg");
    await page.locator("#object-label").fill("People");
    const referenceDetection = page.waitForResponse((response) => response.url().endsWith("/api/reference-boxes") &&
      response.request().method() === "POST", { timeout: 240000 });
    await page.locator("#reference-gallery").setInputFiles(bus);
    await page.locator("#reference-editor").waitFor({ state: "visible" });
    await clearSuggestedBoxes(page, referenceDetection);
    await page.locator(".coordinate-entry summary").click();
    for (const [id, value] of Object.entries({ "box-x1": 48, "box-y1": 398, "box-x2": 244, "box-y2": 902 })) await page.locator(`#${id}`).fill(String(value));
    await page.locator("#add-coordinate-box").click();
    assert.equal(await page.locator(".reference-box-marker").count(), 1, "manual reference box is selected");
    assert.deepEqual(JSON.parse(await page.locator(".reference-box-marker").getAttribute("data-box")), [48, 398, 244, 902], "selected reference overlay matches manual coordinates");
    await page.locator("#target-gallery").setInputFiles([bus, mirror]);
    await page.waitForFunction(() => document.querySelectorAll("#target-queue .queue-item").length === 2);
    await page.screenshot({ path: path.join(results, "desktop-ready.png"), fullPage: true });

    log("Desktop: creating session and first batch");
    const createdResponse = page.waitForResponse((response) => response.url().endsWith("/api/sessions") && response.request().method() === "POST");
    await page.locator("#start-button").click();
    const created = await createdResponse;
    assert.equal(created.status(), 201, `session create: ${await created.text()}`);
    const initial = await created.json();
    report.desktop.sessionId = initial.id;
    assert.deepEqual(initial.references[0].boxes[0], [48, 398, 244, 902], "uploaded pixel box matches editor");
    assert.equal(initial.references[0].width, 810);
    assert.equal(initial.references[0].height, 1080);
    const first = await waitForSession(initial.id, 1);
    await waitForCards(page, 2);
    assert.equal(first.batches[0].results.length, 2);
    for (const result of first.batches[0].results) {
      assert.ok(result.count >= 1 && result.count <= 4, "boundary fragments must not inflate the four-person fixture");
      assert.equal(result.processing.tiles, 4, "native YOLOE crops are used");
      assert.equal(result.processing.tile_size, 640);
    }
    assert.equal(first.batches[0].results.reduce((sum, result) => sum + result.count, 0), first.batches[0].total_count);
    assert.equal(first.batches[0].total_count, first.total_count);
    assert.equal((await page.locator("#metric-total").textContent()).trim(), String(first.total_count));
    assert.equal((await page.locator("#metric-latest").textContent()).trim(), String(first.batches[0].total_count));
    report.desktop.firstBatch = { counts: first.batches[0].results.map((result) => result.count), total: first.total_count };
    await page.screenshot({ path: path.join(results, "desktop-first-results.png"), fullPage: true });
    report.desktop.preview = await openPreview(page, first.batches[0].results[0], "desktop-preview.png");

    log("Desktop: uploading next batch");
    await page.locator("#next-gallery").setInputFiles(resized);
    await page.waitForFunction(() => document.querySelectorAll("#next-queue .queue-item").length === 1);
    await page.locator("#upload-button").click();
    const second = await waitForSession(initial.id, 2);
    await waitForCards(page, 3);
    const sum = second.batches.reduce((total, batch) => total + batch.total_count, 0);
    assert.equal(sum, second.total_count, "running total sums batches");
    assert.equal((await page.locator("#metric-total").textContent()).trim(), String(sum));
    assert.equal((await page.locator("#metric-latest").textContent()).trim(), String(second.batches[1].total_count));
    report.desktop.secondBatch = { total: second.batches[1].total_count, runningTotal: second.total_count };
    await page.screenshot({ path: path.join(results, "desktop-two-batches.png"), fullPage: true });

    log("Desktop: reloading and finishing");
    await page.reload({ waitUntil: "networkidle" });
    await page.locator("#session-view").waitFor({ state: "visible" });
    await waitForCards(page, 3);
    assert.equal((await page.locator("#metric-total").textContent()).trim(), String(second.total_count), "reload restores running total");
    assert.equal(await page.evaluate(() => localStorage.getItem("countwell.sessionId")), initial.id, "reload restores session ID");
    await page.locator("#finish-button").click();
    await page.locator("#new-count-button").waitFor({ state: "visible" });
    const final = await (await fetch(`${base}/api/sessions/${initial.id}`)).json();
    assert.equal(final.status, "finished");
    assert.equal(final.total_count, second.total_count);
    await page.screenshot({ path: path.join(results, "desktop-finished.png"), fullPage: true });
    const downloadPromise = page.waitForEvent("download");
    await page.locator("#export-link").click();
    const download = await downloadPromise;
    assert.match(download.suggestedFilename(), /\.csv$/);
    report.desktop.export = download.suggestedFilename();
    await page.locator("#new-count-button").click();
    await page.locator("#setup-view").waitFor({ state: "visible" });
    assert.equal(await page.evaluate(() => localStorage.getItem("countwell.sessionId")), null, "New count clears restored ID");
    report.desktop.completed = true;
  } finally {
    await context.close();
  }
}

async function mobileFlow(browser) {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.on("pageerror", (error) => report.browserErrors.push(`mobile pageerror: ${error.message}`));
  page.on("console", (entry) => { if (entry.type() === "error") {
    const item = `mobile console: ${entry.text()} at ${entry.location().url}`;
    (entry.location().url.endsWith("/api/reference-boxes") ? report.referenceWarnings :
      entry.location().url.endsWith("/favicon.ico") ? report.resourceWarnings : report.browserErrors).push(item);
  } });
  page.on("response", (response) => { if (response.status() >= 400) (response.url().endsWith("/api/reference-boxes") ? report.referenceWarnings : report.httpErrors).push(`mobile HTTP ${response.status()}: ${response.url()}`); });
  try {
    log("Mobile: loading setup and drawing by touch");
    await page.goto(base, { waitUntil: "networkidle" });
    await page.locator('#model-select option[value="yoloe-26n-seg"]').waitFor({ state: "attached" });
    report.mobile.initialOverflow = await noHorizontalOverflow(page, "mobile setup");
    await page.screenshot({ path: path.join(results, "mobile-initial.png"), fullPage: true });
    await page.locator("#model-select").selectOption("yoloe-26n-seg");
    await page.locator("#object-label").fill("P".repeat(80));
    const referenceDetection = page.waitForResponse((response) => response.url().endsWith("/api/reference-boxes") &&
      response.request().method() === "POST", { timeout: 240000 });
    await page.locator("#reference-gallery").setInputFiles(longReference);
    await page.locator("#reference-editor").waitFor({ state: "visible" });
    await clearSuggestedBoxes(page, referenceDetection);
    const overlay = page.locator("#box-overlay");
    await overlay.scrollIntoViewIfNeeded();
    const rect = await overlay.boundingBox();
    assert.ok(rect?.width > 0 && rect?.height > 0, "mobile overlay has a box");
    const point = (x, y) => ({ x: rect.x + x * rect.width / 810, y: rect.y + y * rect.height / 1080 });
    const start = point(48, 398), end = point(244, 902);
    const touchGeometry = () => page.evaluate(() => {
      const bounds = (id) => {
        const rect = document.getElementById(id).getBoundingClientRect();
        return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
      };
      return { scrollY: window.scrollY, overlay: bounds("box-overlay"), image: bounds("editor-image"),
        noticeHidden: document.getElementById("notice").hidden, notice: bounds("notice"),
        boxCount: document.querySelectorAll(".reference-box-marker").length };
    });
    const beforeTouch = await touchGeometry();
    const scrollBefore = beforeTouch.scrollY;
    const cdp = await context.newCDPSession(page);
    await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [start] });
    for (let fraction = 0.25; fraction <= 1; fraction += 0.25) {
      await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: [{ x: start.x + (end.x - start.x) * fraction, y: start.y + (end.y - start.y) * fraction }] });
      await pause(35);
    }
    const afterLastMove = await touchGeometry();
    await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
    await page.waitForFunction(() => document.querySelectorAll(".reference-box-marker").length === 1);
    const afterCompletion = await touchGeometry();
    const touchBox = await page.locator(".reference-box-marker").evaluate((marker) => {
      const [x1, y1, x2, y2] = JSON.parse(marker.dataset.box);
      return [x1, y1, x2 - x1, y2 - y1];
    });
    const diagnostic = { start, end, beforeTouch, afterLastMove, afterCompletion, touchBox };
    const diagnosticDir = path.join(results, "reference-auto/manual-regression");
    fs.mkdirSync(diagnosticDir, { recursive: true });
    fs.writeFileSync(path.join(diagnosticDir, "touch-diagnostic.json"), JSON.stringify(diagnostic, null, 2));
    await page.screenshot({ path: path.join(diagnosticDir, "after-touch.png"), fullPage: false });
    assert.ok(Math.abs(afterLastMove.scrollY - scrollBefore) <= 2,
      `touch drag scrolled the page from ${scrollBefore} to ${afterLastMove.scrollY}`);
    for (const name of ["overlay", "image"]) {
      for (const axis of ["x", "y"]) {
        assert.ok(Math.abs(afterLastMove[name][axis] - beforeTouch[name][axis]) <= 2,
          `${name} moved during touch drag on ${axis}`);
        assert.ok(Math.abs(afterCompletion[name][axis] - beforeTouch[name][axis]) <= 2,
          `${name} jumped in the viewport after drawing on ${axis}`);
      }
    }
    for (const [index, expected] of [48, 398, 196, 504].entries()) {
      assert.ok(Math.abs(touchBox[index] - expected) <= 3,
        `touch box coordinate ${index} was ${touchBox[index]}, expected ${expected}`);
    }
    report.mobile.touchBox = touchBox;
    report.mobile.touchScroll = { before: scrollBefore, duringDrag: afterLastMove.scrollY, afterLayout: afterCompletion.scrollY };
    report.mobile.editorOverflow = await noHorizontalOverflow(page, "mobile reference editor");
    await page.screenshot({ path: path.join(results, "mobile-reference-box.png"), fullPage: true });
    await page.locator("#target-gallery").setInputFiles(bus);
    await page.waitForFunction(() => document.querySelectorAll("#target-queue .queue-item").length === 1);
    const createdResponse = page.waitForResponse((response) => response.url().endsWith("/api/sessions") && response.request().method() === "POST");
    await page.locator("#start-button").click();
    const created = await createdResponse;
    assert.equal(created.status(), 201, `mobile session create: ${await created.text()}`);
    const initial = await created.json();
    report.mobile.sessionId = initial.id;
    const session = await waitForSession(initial.id, 1);
    await waitForCards(page, 1);
    report.mobile.resultOverflow = await noHorizontalOverflow(page, "mobile results");
    await page.screenshot({ path: path.join(results, "mobile-results.png"), fullPage: true });
    report.mobile.preview = await openPreview(page, session.batches[0].results[0], "mobile-preview.png");
    await page.locator("#finish-button").click();
    await page.locator("#new-count-button").waitFor({ state: "visible" });
    report.mobile.total = session.total_count;
    report.mobile.completed = true;
  } finally {
    await context.close();
  }
}

(async () => {
  assert.ok(fs.existsSync(edge), `Edge executable not found: ${edge}`);
  const browser = await chromium.launch({ executablePath: edge, headless: true });
  try {
    if (process.env.E2E_FLOW !== "mobile") await desktopFlow(browser);
    if (process.env.E2E_FLOW !== "desktop") await mobileFlow(browser);
    assert.deepEqual(report.browserErrors, [], "browser console and page errors");
    assert.deepEqual(report.resourceWarnings, [], "browser resource errors");
    assert.deepEqual(report.httpErrors, [], "HTTP resource errors");
    report.finishedAt = new Date().toISOString();
    fs.writeFileSync(path.join(results, "e2e-report.json"), JSON.stringify(report, null, 2));
    log(`PASS. Desktop total ${report.desktop.secondBatch?.runningTotal ?? "skipped"}; mobile total ${report.mobile.total ?? "skipped"}.`);
  } catch (error) {
    report.error = error.stack || String(error);
    fs.writeFileSync(path.join(results, "e2e-report.json"), JSON.stringify(report, null, 2));
    console.error(report.error);
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
})();
