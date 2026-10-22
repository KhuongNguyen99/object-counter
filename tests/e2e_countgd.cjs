/* Real FastSAM + CountGD++ per-photo exact-ten workflow, including a later batch. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const root = path.resolve(__dirname, '..');
const attachments = path.join(root, '.codex-remote-attachments/01a14b6c-1e76-7570-b308-4ac2aa56b41d/e1ec69e0-b412-4636-a935-0b80fa699bcc');
const inputs = [path.join(attachments, '2-6006.jpg'), path.join(root, 'test-results/fitting-debug/target-at-640.jpg')];
const nextInput = path.join(root, 'test-results/ring-debug-input/target-original.jpg');
const base = process.env.APP_URL || 'http://localhost:8080';
const output = path.resolve(process.env.E2E_OUTPUT_DIR || path.join(root, 'test-results/countgd-ten/browser'));
const report = { base, startedAt: new Date().toISOString(), errors: [], proposals: [], requests: [] };
const algorithm = vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(root, 'frontend/public/reference-selection.js'), 'utf8'), algorithm);

async function settled(context, id, batches) {
  const deadline = Date.now() + 600000;
  while (Date.now() < deadline) {
    const response = await context.request.get(`${base}/api/sessions/${id}`);
    assert.equal(response.status(), 200);
    const session = await response.json();
    const failed = session.batches.find(batch => batch.status === 'failed');
    assert.ok(!failed, failed?.error);
    if (session.batches.length === batches && session.batches.every(batch => batch.status === 'complete')) return session;
    await new Promise(resolve => setTimeout(resolve, 1200));
  }
  throw new Error('CountGD++ counting timed out');
}
const selectedBoxes = page => page.locator('.reference-box-marker').evaluateAll(markers => markers.map(marker => JSON.parse(marker.dataset.box)));
async function upload(page, input, files) {
  const before = report.proposals.length;
  await page.locator(input).setInputFiles(files);
  await page.waitForFunction(count => document.querySelectorAll('#reference-thumbs button').length === count &&
    document.querySelector('#auto-detect-boxes').getAttribute('aria-busy') !== 'true' &&
    !document.querySelector('#next-gallery').disabled, files.length, { timeout: 180000 });
  const deadline = Date.now() + 180000;
  while (report.proposals.length < before + files.length && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 25));
  assert.equal(report.proposals.length, before + files.length);
  const chosen = [];
  for (let photo = 0; photo < files.length; photo++) {
    await page.locator('#reference-thumbs button').nth(photo).click();
    const boxes = await selectedBoxes(page);
    const proposal = report.proposals[before + photo];
    const filtered = algorithm.ReferenceSelection.prepare(proposal.boxes, proposal.width, proposal.height, () => .5);
    const valid = new Set([...filtered.boxes, ...filtered.candidates].map(box => JSON.stringify(box)));
    assert.equal(boxes.length, 10, `Photo ${photo + 1} has ten automatic selections`);
    assert.equal(new Set(boxes.map(JSON.stringify)).size, 10);
    assert.ok(boxes.every(box => valid.has(JSON.stringify(box))), 'No size outlier is preselected');
    assert.equal(await page.locator('#candidate-list').count(), 0);
    assert.deepEqual(await page.locator('.reference-box-marker').allTextContents(), Array(10).fill(''));
    if (await page.locator('#reference-inspector').isVisible()) {
      const crop = await page.locator('#reference-crop').getAttribute('data-box');
      assert.ok(!boxes.some(box => JSON.stringify(box) === crop), 'Inspector excludes selected boxes');
    }
    chosen.push(boxes);
  }
  return chosen;
}

(async () => {
  fs.mkdirSync(output, { recursive: true });
  for (const file of [...inputs, nextInput]) assert.ok(fs.existsSync(file), `Missing fixture ${file}`);
  const browser = await chromium.launch({ executablePath: process.env.EDGE_EXE || 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe', headless: true });
  try {
    const context = await browser.newContext({ ignoreHTTPSErrors: true, locale: 'en-US', viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    page.on('response', async response => {
      if (response.url().endsWith('/api/reference-boxes') && response.request().method() === 'POST') {
        if (response.status() !== 200) report.errors.push(`FastSAM ${response.status()}`);
        else report.proposals.push(await response.json());
      }
    });
    page.on('request', request => {
      if (request.method() === 'POST' && /\/api\/sessions(?:\/[^/]+\/batches)?$/.test(new URL(request.url()).pathname)) {
        report.requests.push({ url: request.url(), body: request.postData() });
      }
    });
    await page.goto(base);
    await page.locator('#model-select option[value="countgd-plusplus"]').waitFor({ state: 'attached' });
    assert.equal(await page.locator('#model-select').inputValue(), 'countgd-plusplus');
    assert.equal(await page.locator('#object-description-field').isHidden(), true);
    assert.equal(await page.locator('#reference-upload-actions').isHidden(), true);
    await page.locator('#confidence').evaluate(input => { input.value = '.23'; input.dispatchEvent(new Event('input', { bubbles: true })); });
    const firstBoxes = await upload(page, '#target-gallery', inputs);
    await page.locator('.reference-box-marker').first().click();
    const cropBefore = await page.locator('#reference-crop').getAttribute('data-box');
    await page.keyboard.press('x');
    assert.equal(await page.locator('.reference-box-marker').count(), 9);
    assert.equal(await page.locator('#reference-crop').getAttribute('data-box'), cropBefore, 'X deselects without skipping');
    assert.equal(await page.locator('#reference-done').isDisabled(), true);
    await page.keyboard.press('Space');
    assert.equal(await page.locator('.reference-box-marker').count(), 10);
    firstBoxes[1] = await selectedBoxes(page);
    await page.screenshot({ path: path.join(output, 'references-desktop.png'), fullPage: true });
    await page.locator('#reference-image-viewport').scrollIntoViewIfNeeded();
    const viewport = await page.locator('#reference-image-viewport').boundingBox();
    await page.mouse.move(viewport.x + viewport.width * .5, viewport.y + viewport.height * .4);
    assert.equal(await page.locator('#reference-magnifier').isVisible(), true);
    await page.screenshot({ path: path.join(output, 'hover-magnifier.png'), fullPage: true });
    await page.mouse.wheel(0, -800);
    await page.waitForFunction(() => document.querySelector('#reference-zoom-reset').textContent !== '100%');
    report.zoom = await page.locator('#reference-zoom-reset').textContent();
    await page.locator('#reference-zoom-reset').click();
    await page.setViewportSize({ width: 390, height: 844 });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1));
    await page.screenshot({ path: path.join(output, 'references-mobile.png'), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });
    const creation = page.waitForResponse(response => response.url().endsWith('/api/sessions') && response.request().method() === 'POST');
    await page.locator('#start-button').click();
    const created = await creation;
    assert.equal(created.status(), 201, await created.text());
    const initial = await created.json();
    report.sessionId = initial.id;
    assert.equal(initial.model, 'countgd-plusplus');
    assert.equal(initial.reference_mode, 'per_image');
    assert.equal(initial.required_reference_boxes, 10);
    assert.equal(initial.label, '');
    assert.equal(initial.confidence, .23);
    assert.deepEqual(initial.references, []);
    console.log(`CountGD++ ${initial.id}: two-photo batch submitted`);
    const first = await settled(context, initial.id, 1);
    assert.deepEqual(first.batches[0].results.map(result => result.reference_boxes), firstBoxes);
    for (const result of first.batches[0].results) {
      assert.equal(result.count, result.detections.length);
      assert.ok(result.count > 0);
    }
    console.log(`First batch counts: ${first.batches[0].results.map(result => result.count).join(', ')}`);
    await page.reload();
    await page.waitForFunction(() => !document.querySelector('#session-view').hidden && !document.querySelector('#next-gallery').disabled);
    const laterBoxes = await upload(page, '#next-gallery', [nextInput]);
    await page.locator('#upload-button').click();
    const final = await settled(context, initial.id, 2);
    assert.deepEqual(final.batches[1].results[0].reference_boxes, laterBoxes[0]);
    assert.deepEqual(final.batches[0].results.map(result => result.reference_boxes), firstBoxes);
    assert.equal(final.total_count, final.batches.reduce((sum, batch) => sum + batch.total_count, 0));
    assert.equal(final.image_count, 3);
    report.session = final;
    report.counts = final.batches.flatMap(batch => batch.results.map(result => result.count));
    await page.waitForFunction(() => document.querySelectorAll('.result-card').length === 3);
    await page.screenshot({ path: path.join(output, 'counted-results.png'), fullPage: true });
    await page.locator('#finish-button').click();
    await page.locator('#new-count-button').waitFor({ state: 'visible' });
    await page.reload();
    await page.locator('#new-count-button').waitFor({ state: 'visible' });
    assert.equal((await page.locator('#metric-total').textContent()).trim(), final.total_count.toLocaleString('en-US'));
    assert.equal(report.requests.filter(request => request.url.endsWith('/batches')).length, 2);
    const sessionRequest = report.requests.find(request => request.url.endsWith('/api/sessions'));
    assert.ok(!sessionRequest.body.includes('name="references"'));
    delete report.requests;
    assert.deepEqual(report.errors, []);
    report.passed = true;
    console.log(JSON.stringify({ passed: true, sessionId: initial.id, counts: report.counts }));
  } finally {
    report.finishedAt = new Date().toISOString();
    fs.writeFileSync(path.join(output, 'browser-report.json'), JSON.stringify(report, null, 2));
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
