import { expect, test } from '@playwright/test';

// Optional real-backend regression: start the API on port 8501 with isolated
// workspace/signing keys, then set OMNIFORGE_BROWSER_API_KEY in the test shell.
// For a rapid full-suite run against the packaged SPA, also set
// OMNIFORGE_API__RATE_LIMIT_PER_MINUTE=0 on the test API (as in tests/conftest.py).
// Set PLAYWRIGHT_BASE_URL=http://127.0.0.1:8501 to test the built SPA as served
// by the API; otherwise the default uses Vite's /api proxy on port 5173.
// The default browser suite uses deterministic network fixtures instead.
const apiKey = process.env.OMNIFORGE_BROWSER_API_KEY;
test.skip(!apiKey, 'Set OMNIFORGE_BROWSER_API_KEY and start the isolated workbench API on port 8501.');

test('real charts survive navigation, reload and signed workspace restore, but not changed scope', async ({ page }) => {
  test.setTimeout(120_000);
  await page.addInitScript((key) => localStorage.setItem('omniforge_api_key', key), apiKey!);
  await page.goto('/');

  const rows = ['alpha,beta,target'];
  for (let i = 0; i < 80; i++) rows.push(`${i},${(i * 7) % 31},${(i * 3) + (i % 4)}`);
  await page.locator('input[accept=".csv,.tsv,.xlsx,.xls,.xlsm,.parquet,.json"]').setInputFiles({
    name: 'chart-data.csv', mimeType: 'text/csv', buffer: Buffer.from(rows.join('\n')),
  });
  await expect(page.getByText(/Ingested 'chart-data.csv'/)).toBeVisible({ timeout: 30_000 });

  await page.locator('[data-studio="02"]').click();
  await page.getByRole('button', { name: /Run Clustering Preview/ }).click();
  await expect(page.getByText('Clustering Preview — Winner:', { exact: false })).toBeVisible({ timeout: 60_000 });
  const original = page.locator('.inspect-body canvas');
  await expect(original).toBeVisible();
  await expect.poll(() => original.evaluate((canvas: HTMLCanvasElement) => {
    const pixels = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data;
    return pixels.some((value, index) => index % 4 === 3 && value > 0);
  })).toBe(true);
  await page.getByRole('button', { name: /Zoom Cluster Map/ }).click();
  await expect(page.locator('.inspect-modal-body canvas')).toBeVisible();
  await expect.poll(() => page.locator('.inspect-modal-body canvas').evaluate((canvas: HTMLCanvasElement) => {
    const pixels = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data;
    return pixels.some((value, index) => index % 4 === 3 && value > 0);
  })).toBe(true);
  await page.locator('.inspect-modal-header button[title="Close"]').click();
  await page.getByRole('button', { name: /3\. Data Mining Lab/ }).click();
  await page.getByRole('button', { name: 'Run Advanced Mining (Beam + ARI + dCor)' }).click();
  await expect(page.getByText('Advanced Mining Diagnostics', { exact: false })).toBeVisible({ timeout: 60_000 });
  await page.getByRole('button', { name: /1\. Clustering & Cluster Row Removal/ }).click();

  await page.locator('[data-studio="04"]').click();
  await page.getByRole('button', { name: 'Run Automated EDA' }).click();
  const alphaChart = page.locator('.inspectable-box').filter({ hasText: 'Distribution: alpha' });
  await expect(alphaChart).toBeVisible({ timeout: 30_000 });
  await expect(alphaChart.locator('svg rect').first()).toBeVisible();
  await expect(page.getByText('Numeric Feature Correlation Matrix')).toBeVisible();
  await page.locator('[data-studio="02"]').click();
  await expect(original).toBeVisible();
  await page.reload();
  await page.locator('[data-studio="04"]').click();
  await expect(alphaChart).toBeVisible();

  await page.getByTitle('Save or Restore Portable .omniforge Project').click();
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Download .omniforge File' }).click(),
  ]);
  const archive = await download.path();
  expect(archive).toBeTruthy();

  await page.getByRole('button', { name: 'Exclusions (0)' }).click();
  await page.locator('.modal-panel').getByRole('button', { name: 'alpha', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Exclusions (1)' })).toBeVisible();
  await page.locator('.modal-backdrop').click({ position: { x: 5, y: 5 } });
  await expect(alphaChart).toHaveCount(0);
  await page.locator('[data-studio="02"]').click();
  await expect(page.getByText('Clustering Preview — Winner:', { exact: false })).toHaveCount(0);
  await page.getByRole('button', { name: /3\. Data Mining Lab/ }).click();
  await expect(page.getByText('Advanced Mining Diagnostics', { exact: false })).toHaveCount(0);

  await page.getByTitle('Save or Restore Portable .omniforge Project').click();
  await page.locator('input[accept=".omniforge,.uap"]').setInputFiles(archive!);
  await expect(page.getByText(/Restored OmniForge project/)).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText('Advanced Mining Diagnostics', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: /1\. Clustering & Cluster Row Removal/ }).click();
  await expect(page.getByText('Clustering Preview — Winner:', { exact: false })).toBeVisible();
  await page.locator('[data-studio="04"]').click();
  await expect(alphaChart).toBeVisible();
  await expect(page.getByRole('button', { name: 'Exclusions (0)' })).toBeVisible();
});


test('small Studio 04 results and exclusions survive Studio 01 cleaning', async ({ page }) => {
  await page.addInitScript((key) => localStorage.setItem('omniforge_api_key', key), apiKey!);
  await page.goto('/');
  await page.locator('input[accept=".csv,.tsv,.xlsx,.xls,.xlsm,.parquet,.json"]').setInputFiles({
    name: 'small-study.csv',
    mimeType: 'text/csv',
    buffer: Buffer.from('Skip Me,Reading,Group\nx,1,a\ny,2,b\nz,3,a\nw,4,b\n'),
  });
  await expect(page.getByText(/Ingested 'small-study.csv'/)).toBeVisible();
  await page.getByRole('button', { name: 'Exclusions (0)' }).click();
  await page.getByRole('button', { name: 'Skip Me', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Skip Me', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.locator('.modal-backdrop').click({ position: { x: 5, y: 5 } });

  await page.locator('[data-studio="04"]').click();
  await page.getByRole('button', { name: 'Run Automated EDA' }).click();
  await expect(page.getByText('Analyzed 4 rows in the current study.')).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText('Distribution: Reading')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Exclusions (1)' })).toBeVisible();

  await page.locator('[data-studio="01"]').click();
  await page.getByRole('button', { name: 'Run Deep 6D Quality Profile' }).click();
  const accuracy = page.locator('.kpi-card:visible').filter({ has: page.getByText('accuracy', { exact: true }) });
  await expect(accuracy).toContainText('Unavailable');
  await page.getByRole('button', { name: /Apply & Commit Cleaning/ }).click();
  await expect(page.getByText(/Cleaning \(.*\) committed/)).toBeVisible();
  await expect(page.getByRole('button', { name: 'Exclusions (1)' })).toBeVisible();

  await page.locator('[data-studio="04"]').click();
  await page.getByRole('combobox', { name: 'Statistical Test', exact: true }).selectOption('shapiro');
  const input = page.getByLabel('Column A (Value)');
  await expect(input.locator('option', { hasText: 'Skip Me' })).toHaveCount(0);
  await input.selectOption('Reading');
  await page.getByRole('button', { name: 'Run Hypothesis Test (CUSTOM)' }).click();
  await expect(page.getByText('Not evaluable', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Exclusions (1)' })).toBeVisible();
});

test('native AutoML separates candidate validation, test scores, and unconfirmed lift', async ({ page }) => {
  await page.addInitScript((key) => localStorage.setItem('omniforge_api_key', key), apiKey!);
  await page.goto('/');
  const rows = ['reading,load,target'];
  for (let i = 0; i < 80; i++) rows.push(`${i},${(i * 7) % 31},${i % 2}`);
  await page.locator('input[accept=".csv,.tsv,.xlsx,.xls,.xlsm,.parquet,.json"]').setInputFiles({
    name: 'automl-study.csv', mimeType: 'text/csv', buffer: Buffer.from(rows.join('\n')),
  });
  await expect(page.getByText(/Ingested 'automl-study.csv'/)).toBeVisible();
  await page.getByRole('button', { name: 'Targets (1)' }).click();
  await page.locator('.modal-panel').getByRole('button', { name: 'load', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Targets (2)' })).toBeVisible();
  await page.locator('.modal-backdrop').click({ position: { x: 5, y: 5 } });
  await page.locator('[data-studio="06"]').click();
  const launch = page.getByRole('button', { name: 'Launch AutoML Tournament' });
  await expect(launch).toBeDisabled();
  await page.getByLabel('Target for this run').selectOption('target');
  await page.getByLabel('Time Budget (sec)').fill('10');
  const resultResponse = page.waitForResponse((response) => response.url().endsWith('/api/workbench/s06/automl'));
  await launch.click();
  const result = await (await resultResponse).json();
  expect(result.target).toBe('target');
  expect(result.state.active_input_feature_cols).toEqual(['reading']);
  await expect(page.getByText('Test f1_weighted', { exact: true })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/16 held-out rows/)).toBeVisible();
  await expect(page.getByText(`Selected Candidate Leaderboard · target (${result.backend_used})`, { exact: true })).toBeVisible();
  await expect(page.getByText('Point Lift vs Strong Reference:', { exact: false })).toBeVisible();
  await expect(page.getByText('Paired comparison unavailable', { exact: false }).first()).toBeVisible();
  await expect(page.getByText('Validation f1_weighted (higher is better)').first()).toBeVisible();
});

test('mining selects each outcome explicitly and live charts follow studio scope', async ({ page }) => {
  await page.addInitScript((key) => localStorage.setItem('omniforge_api_key', key), apiKey!);
  await page.goto('/');
  const rows = ['x,z,outcome_a,outcome_b'];
  for (let i = 0; i < 60; i++) rows.push(`${i},${(i * 7) % 31},${i % 3},${i * 3 + i % 4}`);
  await page.locator('input[accept=".csv,.tsv,.xlsx,.xls,.xlsm,.parquet,.json"]').setInputFiles({
    name: 'scoped-mining.csv', mimeType: 'text/csv', buffer: Buffer.from(rows.join('\n')),
  });
  await expect(page.getByText(/Ingested 'scoped-mining.csv'/)).toBeVisible();
  await page.getByRole('button', { name: 'Targets (1)' }).click();
  await page.locator('.modal-panel').getByRole('button', { name: 'outcome_a', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Targets (2)' })).toBeVisible();
  await page.locator('.modal-backdrop').click({ position: { x: 5, y: 5 } });

  await page.locator('[data-studio="02"]').click();
  await page.getByRole('button', { name: /3\. Data Mining Lab/ }).click();
  const run = page.getByRole('button', { name: /Execute Data Mining Suite/ });
  const target = page.getByLabel('Target for this mining run');
  await expect(run).toBeDisabled();
  await target.selectOption('outcome_b');
  await run.click();
  await expect(page.getByText('Mining target: outcome_b', { exact: true })).toBeVisible();
  await target.selectOption('outcome_a');
  await run.click();
  await expect(page.getByText('Mining target: outcome_a', { exact: true })).toBeVisible();

  const picker = page.getByLabel('Select professional chart to generate');
  await page.locator('[data-studio="04"]').click();
  await expect(picker.locator('option[value="normality_qq_plot"]')).toHaveCount(1);
  await expect(picker.locator('option[value="classifier_roc_curve"]')).toHaveCount(0);
  await page.locator('[data-studio="12"]').click();
  await expect(picker.locator('option[value="classifier_roc_curve"]')).toHaveCount(1);
});

test('supervisory quick-train selects an outcome and score-only shows unavailable compliance', async ({ page }) => {
  await page.addInitScript((key) => localStorage.setItem('omniforge_api_key', key), apiKey!);
  await page.goto('/');
  const rows = ['measurement,category,outcome_a,outcome_b'];
  for (let i = 0; i < 60; i++) rows.push(`${i},${i % 2 === 0 ? 'left' : 'right'},${i % 2},${i * 3 + i % 7}`);
  await page.locator('input[accept=".csv,.tsv,.xlsx,.xls,.xlsm,.parquet,.json"]').setInputFiles({
    name: 'supervisory-study.csv', mimeType: 'text/csv', buffer: Buffer.from(rows.join('\n')),
  });
  await expect(page.getByText(/Ingested 'supervisory-study.csv'/)).toBeVisible();
  await page.getByRole('button', { name: 'Targets (1)' }).click();
  await page.locator('.modal-panel').getByRole('button', { name: 'outcome_a', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Targets (2)' })).toBeVisible();
  await page.locator('.modal-backdrop').click({ position: { x: 5, y: 5 } });
  await page.locator('[data-studio="14"]').click();
  const train = page.getByRole('button', { name: /Quick-Train & Select Model/ });
  await expect(train).toBeDisabled();
  await page.getByLabel('Training outcome').selectOption('outcome_b');
  await train.click();
  await expect(page.getByText(/Trained 'Linear Baseline/)).toBeVisible();
  await page.getByRole('button', { name: 'Single Reading Probe' }).click();
  await page.getByLabel('measurement', { exact: true }).fill('35');
  await page.getByLabel('category', { exact: true }).fill('left');
  await expect(page.getByLabel('Measured Actual (outcome_b)')).toBeVisible();
  await page.getByRole('button', { name: 'Evaluate Single Live Reading' }).click();
  await expect(page.getByText('SYSTEM STATUS: UNAVAILABLE', { exact: true })).toBeVisible();
  await expect(page.getByText('Compliance: Unavailable', { exact: true })).toBeVisible();
  await expect(page.getByText('Unevaluable: 1', { exact: true })).toBeVisible();
});

test('real Studio 03 controls run diagnostics and chart the returned VIF map', async ({ page }) => {
  await page.addInitScript((key) => localStorage.setItem('omniforge_api_key', key), apiKey!);
  await page.goto('/');
  const rows = ['feature_a,feature_b'];
  for (let i = 0; i < 120; i++) {
    const x = (i - 60) / 20;
    rows.push(`${x},${0.92 * x + 0.2 * Math.sin(i)}`);
  }
  await page.locator('input[accept=".csv,.tsv,.xlsx,.xls,.xlsm,.parquet,.json"]').setInputFiles({
    name: 'diagnostics-main-audit.csv', mimeType: 'text/csv', buffer: Buffer.from(rows.join('\n')),
  });
  await expect(page.getByText(/Ingested 'diagnostics-main-audit.csv'/)).toBeVisible();
  await page.locator('[data-studio="03"]').click();
  await page.getByRole('button', { name: /custom/i, exact: true }).click();
  const chips = page.locator('.chip-grid .chip.selected');
  await expect(page.getByRole('button', { name: /Multicollinearity Diagnostics/ })).toBeVisible();
  for (const chip of (await chips.all()).reverse()) {
    const label = await chip.textContent();
    if (!/Normality, Distribution Shape|Multicollinearity Diagnostics/.test(label ?? '')) await chip.click();
  }
  await expect(chips).toHaveCount(2);
  await page.getByRole('combobox', { name: 'Multiple Testing Correction' }).selectOption('holm');
  await page.getByRole('combobox', { name: 'Parametric Policy' }).selectOption('non_parametric');
  await page.getByRole('combobox', { name: 'Normality Test' }).selectOption('anderson_darling');
  const response = page.waitForResponse('**/api/workbench/s03/run');
  await page.getByRole('button', { name: 'Execute Universal Diagnostics (CUSTOM)' }).click();
  const result = await response;
  expect(result.status()).toBe(200);
  const { report } = await result.json();
  expect(report.correction_family.method).toBe('holm');
  expect(report.vif_scores.feature_a).toBeGreaterThan(5);
  const chart = page.locator('.inspectable-box').filter({ hasText: /Variance Inflation Factor/ });
  await expect(chart.getByText('feature_a', { exact: true })).toBeVisible();
  await expect(chart.getByText(report.vif_scores.feature_a.toFixed(4), { exact: true }).first()).toBeVisible();
});
