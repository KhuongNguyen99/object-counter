/* Real text-only, native-resolution browser flow against the packaged app. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const root = path.resolve(__dirname, '..');
const base = process.env.APP_URL || 'http://localhost:8080';
const output = process.env.E2E_OUTPUT_DIR || path.join(root, 'test-results/sam3count-comparison/browser');
const target = path.join(root, 'test-results/ring-debug-input/target-original.jpg');
const nextTarget = path.join(root, '.codex-remote-attachments/01a14b6c-1e76-7570-b308-4ac2aa56b41d/e1ec69e0-b412-4636-a935-0b80fa699bcc/2-6006.jpg');
const report = { base, startedAt: new Date().toISOString(), errors: [] };
fs.mkdirSync(output, { recursive: true });

async function settled(context, id, batches) {
  const deadline = Date.now() + 900000;
  while (Date.now() < deadline) {
    const response = await context.request.get(`${base}/api/sessions/${id}`);
    assert.equal(response.status(), 200);
    const session = await response.json();
    const failed = session.batches.find(batch => batch.status === 'failed');
    assert.ok(!failed, failed?.error);
    if (session.batches.length === batches && session.batches.every(batch => batch.status === 'complete')) return session;
    await new Promise(resolve => setTimeout(resolve, 1500));
  }
  throw new Error('Counting did not finish');
}

(async () => {
  const browser = await chromium.launch({ headless: true,
    executablePath: process.env.EDGE_EXE || 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe' });
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    page.on('response', response => { if (response.status() >= 400) report.errors.push(`${response.status()} ${response.url()}`); });
    await page.goto(base);
    await page.locator('#model-select option[value="sam3count"]').waitFor({ state: 'attached' });
    await page.locator('#model-select').selectOption('sam3count');
    assert.equal(await page.locator('#reference-panel').isVisible(), false);
    assert.equal(await page.locator('#text-prompt-note').isVisible(), true);
    assert.equal(await page.locator('#confidence').inputValue(), '0.35');
    await page.locator('#object-label').fill('white plastic ring');
    await page.locator('#target-gallery').setInputFiles(target);
    await page.waitForFunction(() => document.querySelectorAll('#target-queue .queue-item').length === 1);
    const creation = page.waitForResponse(response => response.url().endsWith('/api/sessions') && response.request().method() === 'POST');
    await page.locator('#start-button').click();
    const created = await creation;
    assert.equal(created.status(), 201, await created.text());
    const initial = await created.json();
    assert.deepEqual(initial.references, []);
    report.sessionId = initial.id;
    const first = await settled(context, initial.id, 1);
    const firstResult = first.batches[0].results[0];
    assert.deepEqual([firstResult.width, firstResult.height], [2252, 4000]);
    assert.ok(firstResult.processing.tiles > 1);
    assert.ok(firstResult.processing.tile_size <= 1008);
    assert.equal(firstResult.count, firstResult.detections.length);
    assert.ok(firstResult.count > 0, 'The rings must produce a real positive detection');
    await page.waitForFunction(() => document.querySelectorAll('.result-card').length === 1);
    await page.locator('.result-image').click();
    await page.locator('#dialog-image').evaluate(image => image.decode());
    assert.deepEqual(await page.locator('#dialog-image').evaluate(image => [image.naturalWidth, image.naturalHeight]), [2252, 4000]);
    await page.locator('#close-dialog').click();
    await page.locator('#next-gallery').setInputFiles(nextTarget);
    await page.waitForFunction(() => document.querySelectorAll('#next-queue .queue-item').length === 1);
    await page.locator('#upload-button').click();
    const second = await settled(context, initial.id, 2);
    assert.equal(second.total_count, second.batches.reduce((sum, batch) => sum + batch.total_count, 0));
    assert.equal(second.image_count, 2);
    await page.waitForFunction(() => document.querySelectorAll('.result-card').length === 2);
    await page.locator('#finish-button').click();
    await page.locator('#new-count-button').waitFor({ state: 'visible' });
    await page.reload();
    await page.locator('#new-count-button').waitFor({ state: 'visible' });
    assert.equal(await page.locator('#metric-total').innerText(), second.total_count.toLocaleString());
    const csv = await context.request.get(`${base}/api/sessions/${initial.id}/export.csv`);
    assert.equal(csv.status(), 200);
    assert.equal((await csv.text()).trim().split('\n').length, 3);
    await page.setViewportSize({ width: 390, height: 844 });
    const bounds = await page.evaluate(() => ({ width: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth }));
    assert.ok(bounds.scroll <= bounds.width + 1, 'Mobile layout must not overflow');
    await page.screenshot({ path: path.join(output, 'mobile-results.png'), fullPage: true });
    report.session = second;
    assert.deepEqual(report.errors, []);
    fs.writeFileSync(path.join(output, 'browser-report.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify({ sessionId: initial.id, counts: second.batches.map(batch => batch.total_count), total: second.total_count }));
  } finally { await browser.close(); }
})().catch(error => { report.errors.push(error.stack); fs.writeFileSync(path.join(output, 'browser-report.json'), JSON.stringify(report, null, 2)); console.error(error); process.exitCode = 1; });
