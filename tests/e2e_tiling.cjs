/* Real native-resolution upload and persisted result flow for either family. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const root = path.resolve(__dirname, '..');
const model = process.env.TILING_MODEL || 'countgd-plusplus';
const isYOLOE = model.startsWith('yoloe-');
const tileSize = isYOLOE ? 640 : 800;
const expectedTiles = (1 + Math.ceil((4000 - tileSize) / (tileSize * .8))) *
  (1 + Math.ceil((2252 - tileSize) / (tileSize * .8)));
const output = process.env.E2E_OUTPUT_DIR ? path.resolve(process.env.E2E_OUTPUT_DIR) :
  path.join(root, isYOLOE ? `test-results/tiling-${model}` : 'test-results/tiling');
const base = process.env.APP_URL || 'http://localhost:8080';
const report = { base, model, errors: [], referenceWarnings: [], startedAt: new Date().toISOString() };

async function clearSuggestedBoxes(page) {
  while (await page.locator('.reference-box-remove').count()) {
    await page.locator('.reference-box-remove').first().click();
  }
  assert.equal(await page.locator('.reference-box-marker').count(), 0, 'automatic selections cleared');
}

(async () => {
  assert.ok(isYOLOE || model === 'countgd-plusplus', `Unsupported tiling model: ${model}`);
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({
    executablePath: process.env.EDGE_EXE || 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe', headless: true,
  });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    page.on('response', response => {
      if (response.status() >= 400) (response.url().endsWith('/api/reference-boxes') ? report.referenceWarnings : report.errors)
        .push(`${response.status()} ${response.url()}`);
    });
    await page.goto(base);
    await page.locator(`#model-select option[value="${model}"]`).waitFor({ state: 'attached' });
    await page.locator('#model-select').selectOption(model);
    if (isYOLOE) await page.locator('#object-label').fill('Bolt with red cap');
    await page.locator('#confidence').evaluate((input, value) => {
      input.value = value; input.dispatchEvent(new Event('input', { bubbles: true }));
    }, isYOLOE ? '0.25' : '0.5');
    const proposalResponse = page.waitForResponse(response => response.url().endsWith('/api/reference-boxes') &&
      response.request().method() === 'POST', { timeout: 240000 });
    const targetPhoto = path.join(root, 'test-results/fitting-debug/target-original.jpg');
    await page.locator(isYOLOE ? '#reference-gallery' : '#target-gallery').setInputFiles(
      isYOLOE ? path.join(root, 'test-results/fitting-debug/reference-original.jpg') : targetPhoto);
    await page.locator('#reference-editor').waitFor({ state: 'visible' });
    assert.equal((await proposalResponse).status(), 200, 'automatic reference boxes completed');
    await page.waitForFunction(() => document.querySelector('#auto-detect-boxes').getAttribute('aria-busy') !== 'true',
      null, { timeout: 240000 });
    let referenceBoxes;
    if (isYOLOE) {
      await clearSuggestedBoxes(page);
      await page.locator('.coordinate-entry summary').click();
      const referenceBox = { x1: 1000, y1: 1702, x2: 1379, y2: 2256 };
      for (const [name, value] of Object.entries(referenceBox)) {
        await page.locator(`#box-${name}`).fill(String(value));
      }
      await page.locator('#add-coordinate-box').click();
      assert.deepEqual(JSON.parse(await page.locator('.reference-box-marker').getAttribute('data-box')),
        Object.values(referenceBox), 'manual shared reference matches coordinates');
      await page.locator('#target-gallery').setInputFiles(targetPhoto);
    } else {
      referenceBoxes = await page.locator('.reference-box-marker').evaluateAll((markers) =>
        markers.map((marker) => JSON.parse(marker.dataset.box)));
      assert.equal(referenceBoxes.length, 10, 'CountGD++ has ten selected boxes on the target photo');
    }
    const created = page.waitForResponse(response => response.url().endsWith('/api/sessions') && response.request().method() === 'POST');
    await page.locator('#start-button').click();
    const response = await created;
    assert.equal(response.status(), 201);
    const initial = await response.json();
    assert.equal(initial.model, model);
    if (isYOLOE) {
      assert.deepEqual(initial.references[0].boxes, [[1000, 1702, 1379, 2256]]);
    } else {
      assert.equal(initial.reference_mode, 'per_image');
      assert.equal(initial.required_reference_boxes, 10);
      assert.deepEqual(initial.references, []);
    }
    report.sessionId = initial.id;
    console.log(`[tiling] 4K fitting photo submitted: ${initial.id}`);
    let session;
    const deadline = Date.now() + 600000;
    while (Date.now() < deadline) {
      session = await (await context.request.get(`${base}/api/sessions/${initial.id}`)).json();
      const batch = session.batches[0];
      assert.notEqual(batch?.status, 'failed', batch?.error);
      if (batch?.status === 'complete') break;
      await new Promise(resolve => setTimeout(resolve, 1500));
    }
    assert.equal(session.batches[0]?.status, 'complete', 'Tiled inference timed out');
    const result = session.batches[0].results[0];
    if (!isYOLOE) assert.deepEqual(result.reference_boxes, referenceBoxes,
      'CountGD++ uses the ten selected target-photo boxes');
    assert.equal(result.width, 4000);
    assert.equal(result.height, 2252);
    assert.equal(result.processing.tiles, expectedTiles);
    assert.equal(result.processing.tile_size, tileSize);
    assert.ok(result.processing.duplicates_removed >= 0);
    if (!isYOLOE) {
      assert.ok(result.processing.duplicates_removed > 0, 'Overlapping crop repeats were reconciled');
      assert.ok(result.count > 0);
    } else if (!result.count) {
      assert.match(result.warning, /No matches detected/);
    }
    assert.equal(result.count, result.detections.length);
    assert.equal(session.total_count, result.count);
    assert.deepEqual(result.detections.map(d => d.index), Array.from({ length: result.count }, (_, i) => i + 1));
    for (const { box: [x1,y1,x2,y2] } of result.detections) {
      assert.ok(0 <= x1 && x1 < x2 && x2 <= 4000 && 0 <= y1 && y1 < y2 && y2 <= 2252);
    }
    // These separated fittings exposed repeated crop-edge fragments in the
    // first implementation. Check the real output, beyond metadata totals.
    for (let i = 0; i < result.detections.length; i++) {
      const a = result.detections[i].box;
      for (const { box: b } of result.detections.slice(i + 1)) {
        const intersection = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) *
          Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
        const union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection;
        assert.ok(intersection / union <= .55, 'Repeated overlapping detection in separated-parts fixture');
      }
    }
    await page.waitForFunction(() => document.querySelectorAll('.result-card').length === 1);
    assert.ok((await page.locator('.result-body').textContent()).includes(`${expectedTiles} tiles`));
    await page.locator('.result-image img').evaluate(image => image.decode());
    await page.screenshot({ path: path.join(output, 'desktop.png'), fullPage: true });
    await page.locator('#finish-button').click();
    await page.locator('#new-count-button').waitFor({ state: 'visible' });
    await page.reload();
    await page.locator('#new-count-button').waitFor({ state: 'visible' });
    assert.equal((await page.locator('#metric-total').textContent()).trim(), String(result.count));
    assert.ok((await page.locator('.result-body').textContent()).includes(`${expectedTiles} tiles`));
    await page.locator('.result-image img').evaluate(image => image.decode());
    await page.setViewportSize({ width: 390, height: 1000 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
    await page.screenshot({ path: path.join(output, 'mobile.png'), fullPage: true });
    await page.locator('.result-image').click();
    const preview = page.locator('#dialog-image');
    await preview.evaluate(image => image.decode());
    assert.deepEqual(await preview.evaluate(image => [image.naturalWidth, image.naturalHeight]), [4000,2252]);
    const annotation = await context.request.get(base + result.annotated_url);
    assert.equal(annotation.status(), 200);
    fs.writeFileSync(path.join(output, 'annotated.jpg'), await annotation.body());
    const csv = await context.request.get(`${base}/api/sessions/${initial.id}/export.csv`);
    assert.equal(csv.status(), 200);
    fs.writeFileSync(path.join(output, 'export.csv'), await csv.body());
    assert.deepEqual(report.errors, []);
    report.result = result;
    report.finishedAt = new Date().toISOString();
    report.passed = true;
    console.log(`[tiling] PASS: ${result.count} detections, ${result.processing.tiles} tiles, ${result.processing.duplicates_removed} repeated candidates removed`);
  } finally {
    fs.writeFileSync(path.join(output, 'browser-report.json'), JSON.stringify(report, null, 2));
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
