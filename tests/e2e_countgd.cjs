/* Real local CountGD++ browser flow using the user's supplied ring photographs. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright-core");
const root = path.resolve(__dirname, "..");
const attachments = path.join(root, ".codex-remote-attachments/01a14b6c-1e76-7570-b308-4ac2aa56b41d/e1ec69e0-b412-4636-a935-0b80fa699bcc");
const output = path.join(root, "test-results/countgd-app");
const base = process.env.APP_URL || "http://localhost:8080";
const report = { errors: [], startedAt: new Date().toISOString() };

async function settled(id, batches) {
  const deadline = Date.now() + 240000;
  while (Date.now() < deadline) {
    const response = await fetch(`${base}/api/sessions/${id}`);
    assert.equal(response.status, 200);
    const session = await response.json();
    const failed = session.batches.find((batch) => batch.status === "failed");
    assert.ok(!failed, failed?.error);
    if (session.batches.length === batches && session.batches.every((batch) => batch.status === "complete")) return session;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error("Counting timed out");
}

async function clearSuggestedBoxes(page) {
  await page.waitForFunction(() => !document.querySelector("#start-button").disabled &&
    document.querySelector("#auto-detect-boxes").textContent !== "Detecting…", null, { timeout: 240000 });
  while (await page.locator("#box-list button").count()) await page.locator("#box-list button").first().click();
}

(async () => {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ executablePath: process.env.EDGE_EXE || "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage();
    page.on("pageerror", (error) => report.errors.push(error.message));
    await page.goto(base);
    await page.locator('#model-select option[value="countgd-plusplus"]').waitFor({ state: "attached" });
    assert.equal(await page.locator("#model-select").inputValue(), "countgd-plusplus");
    assert.equal(await page.locator("#confidence").inputValue(), "0.23");
    await page.locator("#object-label").fill("white plastic ring");
    await page.locator("#reference-gallery").setInputFiles(path.join(attachments, "1-6005.jpg"));
    await page.locator("#reference-editor").waitFor({ state: "visible" });
    await clearSuggestedBoxes(page);
    await page.locator(".coordinate-entry summary").click();
    for (const [id, value] of Object.entries({ "box-x1": 294, "box-y1": 586, "box-x2": 436, "box-y2": 727 })) await page.locator(`#${id}`).fill(String(value));
    await page.locator("#add-coordinate-box").click();
    await page.locator("#target-gallery").setInputFiles(path.join(attachments, "2-6006.jpg"));
    const created = page.waitForResponse((response) => response.url().endsWith("/api/sessions") && response.request().method() === "POST");
    await page.locator("#start-button").click();
    const response = await created;
    assert.equal(response.status(), 201, await response.text());
    const initial = await response.json();
    assert.equal(initial.model, "countgd-plusplus");
    assert.equal(initial.confidence, .23);
    report.sessionId = initial.id;
    console.log("CountGD++: first ring image submitted");
    const first = await settled(initial.id, 1);
    const result = first.batches[0].results[0];
    assert.ok(result.count >= 50, "new model materially improves on the zero-detection YOLOE result");
    assert.equal(result.count, result.detections.length);
    assert.deepEqual(result.detections.map((item) => item.index), Array.from({ length: result.count }, (_, i) => i + 1));
    await page.waitForFunction(() => document.querySelectorAll(".result-card").length === 1);
    assert.equal(await page.locator("#results-state").textContent(), "Review needed");
    await page.screenshot({ path: path.join(output, "desktop-results.png"), fullPage: true });
    report.first = result;
    await page.locator("#next-gallery").setInputFiles(path.join(root, "test-results/ring-debug-input/target-original.jpg"));
    await page.locator("#upload-button").click();
    const accumulated = await settled(initial.id, 2);
    const expected = accumulated.batches.reduce((sum, batch) => sum + batch.total_count, 0);
    assert.equal(accumulated.total_count, expected);
    assert.equal(accumulated.image_count, 2);
    report.second = accumulated.batches[1].results[0];
    report.total = accumulated.total_count;
    console.log(`CountGD++: counts ${report.first.count}, ${report.second.count}; additive total ${report.total}`);
    await page.waitForFunction(() => document.querySelectorAll(".result-card").length === 2);
    await page.locator("#finish-button").click();
    await page.locator("#new-count-button").waitFor({ state: "visible" });
    await page.reload();
    await page.locator("#new-count-button").waitFor({ state: "visible" });
    assert.equal((await page.locator("#metric-total").textContent()).trim(), String(expected));
    await page.setViewportSize({ width: 390, height: 1000 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1), true);
    await page.screenshot({ path: path.join(output, "mobile-finished.png"), fullPage: true });
    const csv = await (await fetch(`${base}/api/sessions/${initial.id}/export.csv`)).text();
    assert.ok(csv.includes("warning") && csv.includes("Experimental model"));
    fs.writeFileSync(path.join(output, "export.csv"), csv);
    assert.deepEqual(report.errors, []);
    report.completed = true;
    report.finishedAt = new Date().toISOString();
  } finally {
    fs.writeFileSync(path.join(output, "browser-report.json"), JSON.stringify(report, null, 2));
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
