/* Real PET-DINO browser flow with the fitting photos. Detector responses are not mocked. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');

const root = path.resolve(__dirname, '..');
const base = process.env.APP_URL || 'http://localhost:8080';
const edge = process.env.EDGE_EXE || 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const output = process.env.E2E_OUTPUT_DIR ? path.resolve(process.env.E2E_OUTPUT_DIR) :
  path.join(root, 'test-results/petdino/browser');
const model = 'pet-dino-swin-t';
const referenceBox = [1039, 1775, 1344, 2170];
const reference = path.join(root, 'test-results/fitting-debug/reference-original.jpg');
const target = path.join(root, 'test-results/fitting-debug/target-original.jpg');
const nextTarget = path.join(root, 'test-results/fitting-debug/target-at-640.jpg');
const report = { base, model, startedAt: new Date().toISOString(), errors: [], reviewWarnings: [] };

fs.mkdirSync(output, { recursive: true });
const save = (name, data) => fs.writeFileSync(path.join(output, name), JSON.stringify(data, null, 2));
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function settled(context, id, batches) {
  const deadline = Date.now() + 900000;
  while (Date.now() < deadline) {
    const response = await context.request.get(`${base}/api/sessions/${encodeURIComponent(id)}`);
    assert.equal(response.status(), 200, `Read session ${id}`);
    const session = await response.json();
    const failed = session.batches.find((batch) => batch.status === 'failed');
    assert.ok(!failed, failed?.error || 'A counting batch failed');
    if (session.batches.length === batches && session.batches.every((batch) => batch.status === 'complete'))
      return session;
    await pause(1500);
  }
  throw new Error(`PET-DINO timed out waiting for ${batches} completed batches`);
}

async function cards(page, count) {
  await page.waitForFunction((expected) => document.querySelectorAll('.result-card').length === expected,
    count, { timeout: 60000 });
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

function checkResult(result, label) {
  assert.ok(Number.isInteger(result.width) && result.width > 0 &&
    Number.isInteger(result.height) && result.height > 0, `${label} dimensions`);
  assert.ok(Number.isInteger(result.count) && result.count >= 0, `${label} count`);
  assert.equal(result.count, result.detections.length, `${label} count matches detections`);
  assert.deepEqual(result.detections.map((detection) => detection.index),
    Array.from({ length: result.count }, (_, index) => index + 1), `${label} numbered detections`);
  for (const detection of result.detections) {
    assert.ok(Array.isArray(detection.box) && detection.box.length === 4 &&
      detection.box.every(Number.isFinite), `${label} finite detection box`);
    const [x1, y1, x2, y2] = detection.box;
    assert.ok(0 <= x1 && x1 < x2 && x2 <= result.width &&
      0 <= y1 && y1 < y2 && y2 <= result.height, `${label} in-bounds detection: ${detection.box}`);
  }
  assert.ok(result.annotated_url, `${label} has an individual annotation`);
  if (result.count === 0) report.reviewWarnings.push(`${label}: ${result.warning || 'No matches detected'}`);
}

async function preview(page, index, result) {
  const card = page.locator('.result-card').nth(index);
  const image = card.locator('.result-image img');
  await image.evaluate((element) => element.decode());
  assert.equal(await image.evaluate((element) => new URL(element.src).pathname), result.annotated_url);
  await card.locator('.result-image').click();
  await page.locator('#image-dialog').waitFor({ state: 'visible' });
  const full = page.locator('#dialog-image');
  await full.evaluate((element) => element.decode());
  assert.deepEqual(await full.evaluate((element) => [element.naturalWidth, element.naturalHeight]),
    [result.width, result.height], 'Full-image preview keeps native resolution');
  await page.locator('#close-dialog').click();
}

(async () => {
  for (const file of [reference, target, nextTarget]) assert.ok(fs.existsSync(file), `Missing fixture: ${file}`);
  assert.ok(fs.existsSync(edge), `Edge executable not found: ${edge}`);
  const browser = await chromium.launch({ executablePath: edge, headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage();
    page.on('pageerror', (error) => report.errors.push(`Page: ${error.message}`));
    page.on('console', (entry) => {
      if (entry.type() === 'error') report.errors.push(`Console: ${entry.text()}`);
    });
    page.on('response', (response) => {
      if (response.status() >= 400) report.errors.push(`HTTP ${response.status()}: ${response.url()}`);
    });

    await page.goto(base);
    const catalogResponse = await context.request.get(`${base}/api/models`);
    assert.equal(catalogResponse.status(), 200);
    const catalog = await catalogResponse.json();
    const pet = catalog.models.find((entry) => entry.id === model);
    assert.ok(pet?.available, 'PET-DINO checkpoint is available in the catalog');
    assert.equal(pet.name, 'PET-DINO Swin-T (experimental)');
    assert.equal(pet.family, 'PET-DINO');
    assert.equal(pet.size, 't');
    assert.equal(pet.default_confidence, 0.3);
    const option = page.locator(`#model-select option[value="${model}"]`);
    await option.waitFor({ state: 'attached' });
    assert.equal(await option.isDisabled(), false, 'PET-DINO is available in the model catalog');
    assert.match(await option.textContent(), /PET-DINO Swin-T \(experimental\)/);
    await page.locator('#model-select').selectOption(model);
    assert.equal(await page.locator('#confidence').inputValue(), '0.3');
    await page.locator('#object-label').fill('Bolt with red cap');

    const proposalResponse = page.waitForResponse((response) =>
      response.url().endsWith('/api/reference-boxes') && response.request().method() === 'POST',
    { timeout: 240000 });
    await page.locator('#reference-gallery').setInputFiles(reference);
    await page.locator('#reference-editor').waitFor({ state: 'visible' });
    const proposal = await proposalResponse;
    assert.equal(proposal.status(), 200, `Automatic reference boxes: ${await proposal.text()}`);
    const suggested = await proposal.json();
    assert.equal(suggested.model, 'fastsam-s', 'FastSAM-s proposes PET-DINO reference boxes');
    assert.deepEqual([suggested.width, suggested.height], [2252, 4000]);
    save('reference-proposal.json', suggested);
    await page.waitForFunction(() => document.querySelector('#auto-detect-boxes').getAttribute('aria-busy') !== 'true', null, { timeout: 240000 });
    while (await page.locator('.reference-box-remove').count()) {
      await page.locator('.reference-box-remove').first().click();
    }
    assert.equal(await page.locator('.reference-box-marker').count(), 0, 'automatic selections cleared');
    await page.locator('.coordinate-entry summary').click();
    for (const [name, value] of Object.entries({ x1: referenceBox[0], y1: referenceBox[1],
      x2: referenceBox[2], y2: referenceBox[3] })) await page.locator(`#box-${name}`).fill(String(value));
    await page.locator('#add-coordinate-box').click();
    assert.equal(await page.locator('.reference-box-marker').count(), 1);
    assert.deepEqual(JSON.parse(await page.locator('.reference-box-marker').getAttribute('data-box')),
      referenceBox, 'manual reference overlay matches coordinates');
    await page.setViewportSize({ width: 390, height: 844 });
    report.mobileEditor = await noOverflow(page, 'PET-DINO reference editor at 390px');
    await page.screenshot({ path: path.join(output, 'mobile-editor.png'), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });

    await page.locator('#target-gallery').setInputFiles(target);
    await page.waitForFunction(() => document.querySelectorAll('#target-queue .queue-item').length === 1);
    const created = page.waitForResponse((response) => response.url().endsWith('/api/sessions') &&
      response.request().method() === 'POST');
    const firstBatch = page.waitForResponse((response) => /\/api\/sessions\/[^/]+\/batches$/.test(new URL(response.url()).pathname) &&
      response.request().method() === 'POST');
    await page.locator('#start-button').click();
    const creation = await created;
    assert.equal(creation.status(), 201, await creation.text());
    const initial = await creation.json();
    assert.equal((await firstBatch).status(), 202);
    assert.equal(initial.model, model);
    assert.equal(initial.confidence, 0.3);
    assert.equal(initial.label, 'Bolt with red cap');
    assert.deepEqual(initial.references[0].boxes, [referenceBox]);
    assert.deepEqual([initial.references[0].width, initial.references[0].height], [2252, 4000]);
    report.sessionId = initial.id;

    const first = await settled(context, initial.id, 1);
    save('first-session.json', first);
    const original = first.batches[0].results[0];
    assert.equal(first.batches[0].results.length, 1);
    assert.deepEqual([original.width, original.height], [4000, 2252]);
    assert.equal(original.processing.tiles, 24, '4K photo uses 24 overlapping PET-DINO tiles');
    assert.equal(original.processing.tile_size, 800);
    assert.ok(original.processing.duplicates_removed >= 0);
    checkResult(original, 'Original fitting photo');
    assert.equal(first.batches[0].total_count, original.count);
    assert.equal(first.total_count, original.count);
    await cards(page, 1);
    assert.match(await page.locator('.result-body').first().textContent(), /24 tiles/);
    await preview(page, 0, original);
    const annotation = await context.request.get(base + original.annotated_url);
    assert.equal(annotation.status(), 200);
    fs.writeFileSync(path.join(output, 'original-annotated.jpg'), await annotation.body());
    await page.screenshot({ path: path.join(output, 'desktop-first-result.png'), fullPage: true });

    await page.locator('#next-gallery').setInputFiles(nextTarget);
    await page.waitForFunction(() => document.querySelectorAll('#next-queue .queue-item').length === 1);
    const added = page.waitForResponse((response) => /\/api\/sessions\/[^/]+\/batches$/.test(new URL(response.url()).pathname) &&
      response.request().method() === 'POST');
    await page.locator('#upload-button').click();
    assert.equal((await added).status(), 202);
    const accumulated = await settled(context, initial.id, 2);
    const small = accumulated.batches[1].results[0];
    assert.equal(accumulated.batches[1].results.length, 1);
    checkResult(small, 'Second fitting photo');
    assert.equal(accumulated.batches[1].total_count, small.count);
    assert.equal(accumulated.image_count, 2);
    assert.equal(accumulated.total_count, original.count + small.count, 'Session accumulates both batches');
    await cards(page, 2);
    assert.equal(Number((await page.locator('#metric-total').textContent()).replaceAll(',', '')),
      accumulated.total_count);
    assert.equal(Number((await page.locator('#metric-latest').textContent()).replaceAll(',', '')),
      small.count);
    if (report.reviewWarnings.length) assert.equal(await page.locator('#results-state').textContent(), 'Review needed');
    await preview(page, 1, small);

    const finished = page.waitForResponse((response) => response.url().endsWith(`/api/sessions/${initial.id}/finish`) &&
      response.request().method() === 'POST');
    await page.locator('#finish-button').click();
    assert.equal((await finished).status(), 200);
    await page.locator('#new-count-button').waitFor({ state: 'visible' });
    await page.reload();
    await page.locator('#new-count-button').waitFor({ state: 'visible' });
    await cards(page, 2);
    const finalResponse = await context.request.get(`${base}/api/sessions/${initial.id}`);
    assert.equal(finalResponse.status(), 200);
    const final = await finalResponse.json();
    assert.equal(final.status, 'finished');
    assert.equal(final.model, model);
    assert.equal(final.total_count, accumulated.total_count);
    assert.equal(Number((await page.locator('#metric-total').textContent()).replaceAll(',', '')),
      final.total_count);
    assert.equal(await page.evaluate(() => localStorage.getItem('countwell.sessionId')), initial.id);
    await page.setViewportSize({ width: 390, height: 844 });
    report.mobileFinished = await noOverflow(page, 'PET-DINO finished results at 390px');
    await page.screenshot({ path: path.join(output, 'mobile-finished.png'), fullPage: true });
    await preview(page, 0, original);

    const csv = await context.request.get(`${base}/api/sessions/${initial.id}/export.csv`);
    assert.equal(csv.status(), 200);
    const csvText = await csv.text();
    assert.match(csvText, /^session,batch,filename,count,running_total,warning\r?\n/);
    assert.ok(csvText.includes(`${original.filename},${original.count},${original.count},`),
      'CSV includes the original image and first running total');
    assert.ok(csvText.includes(`${small.filename},${small.count},${final.total_count},`),
      'CSV includes the second image and accumulated total');
    fs.writeFileSync(path.join(output, 'export.csv'), csvText);
    assert.deepEqual(report.errors, [], 'No browser or HTTP errors');
    save('finished-session.json', final);
    report.counts = [original.count, small.count];
    report.total = final.total_count;
    report.passed = true;
    console.log(`[pet-dino] PASS: ${report.total} objects across two batches; 24 native 800px tiles`);
  } finally { await browser.close(); }
})().catch((error) => { report.error = error.stack || String(error); console.error(report.error); process.exitCode = 1; })
  .finally(() => { report.finishedAt = new Date().toISOString(); save('browser-report.json', report); });
