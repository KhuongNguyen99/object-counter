/* Real SAM3/FastSAM browser flow; every counting image supplies its own visual examples. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright-core");

const root = path.resolve(__dirname, "..");
const base = process.env.APP_URL || "http://localhost:8080";
const output = path.resolve(process.env.E2E_OUTPUT_DIR || path.join(root, "test-results/sam3-visual/browser"));
const inputs = [path.join(root, "test-results/bus.jpg"),
  path.join(root, ".codex-remote-attachments/01a14b6c-1e76-7570-b308-4ac2aa56b41d/e1ec69e0-b412-4636-a935-0b80fa699bcc/2-6006.jpg")];
const nextInput = path.join(root, "test-results/bus-mirror.jpg");
const report = { base, startedAt: new Date().toISOString(), errors: [], proposals: [] };
fs.mkdirSync(output, { recursive: true });

async function settled(context, id, count) {
  const deadline = Date.now() + 600000;
  while (Date.now() < deadline) {
    const response = await context.request.get(`${base}/api/sessions/${id}`);
    assert.equal(response.status(), 200);
    const session = await response.json();
    assert.ok(!session.batches.some((batch) => batch.status === "failed"), JSON.stringify(session.batches));
    if (session.batches.length === count && session.batches.every((batch) => batch.status === "complete")) return session;
    await new Promise((resolve) => setTimeout(resolve, 1200));
  }
  throw new Error("SAM3 counting timed out");
}

async function uploadAndSelect(page, input, files) {
  const firstProposal = report.proposals.length;
  await page.locator(input).setInputFiles(files);
  await page.waitForFunction((count) => document.querySelectorAll("#reference-thumbs button").length === count &&
    document.querySelector("#auto-detect-boxes").getAttribute("aria-busy") !== "true" &&
    !document.querySelector("#next-gallery").disabled, files.length, { timeout: 180000 });
  const deadline = Date.now() + 180000;
  while (report.proposals.length < firstProposal + files.length && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  assert.equal(report.proposals.length, firstProposal + files.length, "every target is auto-detected independently");
  const selected = [];
  for (let photo = 0; photo < files.length; photo++) {
    const proposal = report.proposals[firstProposal + photo];
    assert.ok(proposal.boxes.length, `FastSAM found examples in photo ${photo + 1}`);
    await page.locator("#reference-thumbs button").nth(photo).click();
    const index = photo === 0 && path.basename(files[photo]) === "bus.jpg" ? proposal.boxes.reduce((best, box, i) => {
      const target = [44, 398, 244, 905];
      const overlap = Math.max(0, Math.min(box[2], target[2]) - Math.max(box[0], target[0])) *
        Math.max(0, Math.min(box[3], target[3]) - Math.max(box[1], target[1]));
      const area = (box[2] - box[0]) * (box[3] - box[1]);
      const score = overlap / (area + (target[2] - target[0]) * (target[3] - target[1]) - overlap);
      return score > best.score ? { index: i, score } : best;
    }, { index: 0, score: 0 }).index : 0;
    const wanted = JSON.stringify(proposal.boxes[index]);
    for (let step = 0; step < proposal.boxes.length; step++) {
      if (await page.locator('#reference-crop').getAttribute('data-box') === wanted) break;
      await page.locator('#reference-box-next').click();
    }
    assert.equal(await page.locator('#reference-crop').getAttribute('data-box'), wanted);
    await page.locator('#reference-image-viewport').focus();
    await page.keyboard.press("Space");
    assert.equal(await page.locator('.reference-box-marker').count(), 1);
    assert.equal(await page.locator('.reference-box-marker').getAttribute('data-box'), wanted);
    selected.push([proposal.boxes[index]]);
  }
  return selected;
}

(async () => {
  for (const file of [...inputs, nextInput]) assert.ok(fs.existsSync(file), `Missing fixture ${file}`);
  const browser = await chromium.launch({ executablePath: process.env.EDGE_EXE ||
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", headless: true });
  try {
    const context = await browser.newContext({ ignoreHTTPSErrors: true, viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage();
    page.on("pageerror", (error) => report.errors.push(error.message));
    page.on("response", async (response) => {
      if (response.url().endsWith("/api/reference-boxes") && response.request().method() === "POST") {
        if (response.status() !== 200) report.errors.push(`FastSAM ${response.status()}`);
        else report.proposals.push(await response.json());
      }
    });
    await page.goto(base);
    await page.locator('#model-select option[value="sam3"]').waitFor({ state: "attached" });
    await page.locator("#model-select").selectOption("sam3");
    assert.equal(await page.locator("#object-label").isHidden(), true);
    const firstBoxes = await uploadAndSelect(page, "#target-gallery", inputs);
    await page.screenshot({ path: path.join(output, "mirrored-references-desktop.png"), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
    await page.screenshot({ path: path.join(output, "mirrored-references-mobile.png"), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });
    const creation = page.waitForResponse((response) => response.url().endsWith("/api/sessions") && response.request().method() === "POST");
    await page.locator("#start-button").click();
    const created = await creation;
    assert.equal(created.status(), 201, await created.text());
    const initial = await created.json();
    report.sessionId = initial.id;
    assert.equal(initial.model, "sam3");
    assert.equal(initial.label, "");
    assert.deepEqual(initial.references, []);
    const first = await settled(context, initial.id, 1);
    assert.deepEqual(first.batches[0].results.map((result) => result.reference_boxes), firstBoxes);
    for (const result of first.batches[0].results) {
      assert.equal(result.count, result.detections.length);
      assert.ok(result.count > 0, "A real visual example produces matches without text");
    }
    await page.reload();
    await page.locator("#next-gallery").waitFor({ state: "attached" });
    await page.waitForFunction(() => !document.querySelector("#session-view").hidden && !document.querySelector("#next-gallery").disabled);
    const laterBoxes = await uploadAndSelect(page, "#next-gallery", [nextInput]);
    assert.equal(await page.locator("#reference-panel").isVisible(), true);
    await page.locator("#upload-button").click();
    const final = await settled(context, initial.id, 2);
    assert.deepEqual(final.batches[1].results[0].reference_boxes, laterBoxes[0]);
    assert.deepEqual(final.batches[0].results.map((result) => result.reference_boxes), firstBoxes, "later batch preserves earlier photo examples");
    assert.equal(final.total_count, final.batches.reduce((sum, batch) => sum + batch.total_count, 0));
    report.session = final;
    await page.waitForFunction(() => document.querySelectorAll(".result-card").length === 3);
    await page.screenshot({ path: path.join(output, "counted-results.png"), fullPage: true });
    await page.locator("#finish-button").click();
    await page.locator("#new-count-button").waitFor({ state: "visible" });
    assert.deepEqual(report.errors, []);
    report.passed = true;
    console.log(JSON.stringify({ passed: true, sessionId: initial.id, counts: final.batches.flatMap((batch) => batch.results.map((result) => result.count)) }));
  } finally {
    report.finishedAt = new Date().toISOString();
    fs.writeFileSync(path.join(output, "browser-report.json"), JSON.stringify(report, null, 2));
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
