import { expect, test, type Page, type Route } from '@playwright/test';

type Results = Record<string, any>;

const scatter = {
  x: [-2, -1, 0, 1, 2], y: [2, 1, 0, -1, -2], groups: [0, 0, 1, 1, 1],
  total_points: 5, rendered_points: 5, downsampled: false,
};
const eda = (mean: number) => ({
  distributions: [{ column: 'alpha', mean, std: 1, bins: [{ label: 'low', count: 4 }, { label: 'high', count: 8 }] }],
  correlation_matrix: { columns: ['alpha', 'beta'], values: [[1, null], [null, 1]] },
  insights: [],
});
const supervisory = {
  system_status: 'NORMAL', source: 'test batch', kpis: { n_records: 2 },
  trend_series: [{ label: 'a', actual: 4, predicted: 3, baseline: 2 }, { label: 'b', actual: 5, predicted: 4, baseline: 3 }],
  spc_points: [{ label: 'a', residual: 1, status: 'IN_SPEC' }, { label: 'b', residual: -1, status: 'WARNING' }],
  // This is the actual server contract, not the aliases previously assumed by the chart.
  spc_limits: { center: 1.5, ucl: 5, lcl: -3, warn_upper: 3, warn_lower: -2 },
  history: [{ source: 'test batch' }],
};
const initialResults = () => ({
  s02_clustering: { algorithm: 'kmeans', n_clusters: 2, noise_count: 0, metrics: { silhouette: 0.5 }, cluster_choices: [], scatter },
  s02_mining: { manifold: { method: 'SVD', total_explained_pct: 68.5, x_label: 'SVD Component 1', y_label: 'SVD Component 2', scatter } },
  s04_eda: eda(1),
  s05_select: { mode: 'auto', selected_features: ['alpha'], rankings: [{ feature: 'alpha', score: 0.8, selected: true }] },
  s07_explain: { global_importance: { alpha: { importance_mean: 0.4321, importance_std: 0.05 }, beta: { importance_mean: -0.125, importance_std: 0.01 } } },
  s08_forecast: { history: [{ step: -1, actual: 1 }, { step: 0, actual: 2 }], deep_forecast: [{ step: 1, yhat: 3, yhat_lower: 2.5, yhat_upper: 3.5 }] },
  s09_graph: { visualization: { nodes: [{ id: 'a', x: 0, y: 0, degree: 1, community: 0 }, { id: 'b', x: 1, y: 1, degree: 1, community: 0 }], edges: [{ source: 'a', target: 'b' }] }, topological_links: [] },
  s12_report: { title: 'Test report', sections: [], n_charts: 1, chart_previews: [{ title: 'Report visual', image_data_url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aQe4AAAAASUVORK5CYII=' }] },
  s14_supervisory: supervisory,
});

function workspace(results: Results) {
  return {
    session_id: 'browser-test', platform: { name: 'OmniForge', version: '4.0.0' },
    has_dataset: true, dataset_name: 'example.csv', dataset_source: 'test', n_rows: 100, n_cols: 3,
    raw_rows: 100, raw_cols: 3, columns: ['alpha', 'beta', 'target'], numeric_columns: ['alpha', 'beta', 'target'], categorical_columns: [],
    targets: ['target'], primary_target: 'target', excluded_columns: [] as string[], analysis_cols_count: 3,
    can_undo: false, processing_status: {}, cached_result_keys: Object.keys(results), split_lock: null,
    total_deleted_rows: 0, format_audit_issues: 0, active_model_name: 'test-model', active_model_metrics: {},
    active_feature_cols: ['alpha', 'beta'], available_models_count: 1, supervisory_status: 'NORMAL',
    studio_completed: {}, workflow_checklist: [], lineage: [], theme: 'dark', project_name: 'test-project',
  };
}

class MockWorkbench {
  results: Results;
  state: ReturnType<typeof workspace>;
  holdResults = false;
  failResults = false;
  held: Array<() => void> = [];
  resultRequests = 0;
  completedResults = 0;
  restoredResults: Results | null = null;
  edaResponse: Results | null = null;
  explainResponse: Results | null = null;
  commitResponseMode: 'normal' | 'legacy-invalidated' | 'explicit-null' = 'normal';
  relationshipResult: Results = { pairs: [], target_relationships: [] };
  holdRelationships = false;
  heldRelationships: Array<() => void> = [];
  relationshipRequests = 0;
  completedRelationships = 0;
  supervisoryHistory: Results[];
  restoredHistory: Results[] | null = null;
  holdSupervisoryStatus = false;
  heldSupervisoryStatus: Array<() => void> = [];
  completedSupervisoryStatus = 0;

  constructor(results: Results) {
    this.results = results;
    this.state = workspace(results);
    this.supervisoryHistory = results.s14_supervisory?.history ?? [];
  }

  async install(page: Page) {
    await page.route('**/api/workbench/**', async (route: Route) => {
      const path = new URL(route.request().url()).pathname;
      const method = route.request().method();
      let body: unknown = {};
      if (path.endsWith('/state')) body = this.state;
      else if (path.endsWith('/results')) {
        // Snapshot at request arrival so a delayed response really is stale.
        const snapshot = JSON.parse(JSON.stringify({ results: this.results, state: this.state }));
        this.resultRequests += 1;
        if (this.failResults) {
          await route.fulfill({ status: 403, contentType: 'application/json', body: JSON.stringify({ detail: 'Results unavailable' }) });
          this.completedResults += 1;
          return;
        }
        if (this.holdResults) await new Promise<void>((resolve) => this.held.push(resolve));
        body = snapshot;
      } else if (path.endsWith('/s00/data_relationships')) {
        const snapshot = JSON.parse(JSON.stringify(this.relationshipResult));
        this.relationshipRequests += 1;
        if (this.holdRelationships) await new Promise<void>((resolve) => this.heldRelationships.push(resolve));
        body = snapshot;
      } else if (path.endsWith('/s01/catalog')) body = { benchmarks: [], domains: [] };
      else if (path.endsWith('/s01/grid')) body = { columns: [], rows: [], page: 1, total_pages: 1, total_rows: 0 };
      else if (path.endsWith('/s06/backends')) body = { automl_backends: [] };
      else if (path.endsWith('/s09/backends')) body = { nlp: [], vision: { backends: [] } };
      else if (path.endsWith('/s12/catalog')) body = { sections: [], presets: {} };
      else if (path.endsWith('/s14/models')) {
        const snapshot = JSON.parse(JSON.stringify({ models: [], last_evaluation: this.results.s14_supervisory ?? null, history: this.supervisoryHistory }));
        if (this.holdSupervisoryStatus) await new Promise<void>((resolve) => this.heldSupervisoryStatus.push(resolve));
        body = snapshot;
      } else if (path.endsWith('/feature-store/freshness')) body = { as_of: 'now', total_views: 0, online_rows: 0, expired_rows: 0, unknown_timestamp_rows: 0, views: [], alerts: [] };
      else if (path.endsWith('/exclusions') && method === 'POST') {
        const excluded = route.request().postDataJSON().excluded_columns;
        this.results = {};
        this.state = { ...this.state, excluded_columns: excluded, cached_result_keys: [], active_model_name: null as any, available_models_count: 0, supervisory_status: 'IDLE' };
        body = { state: this.state };
      } else if (path.endsWith('/workspace/load') && method === 'POST') {
        this.results = this.restoredResults ?? {};
        this.supervisoryHistory = this.restoredHistory ?? this.results.s14_supervisory?.history ?? [];
        this.state = { ...this.state, cached_result_keys: Object.keys(this.results) };
        body = { state: this.state };
      } else if (path.endsWith('/s04/eda') && method === 'POST') {
        this.results = this.edaResponse ?? {};
        this.state = { ...this.state, cached_result_keys: Object.keys(this.results) };
        body = { ...this.results.s04_eda, state: this.state };
      } else if (path.endsWith('/s02/advanced_mining') && method === 'POST') {
        const result = { target: 'first', cluster_stability_ari: { mean_ari: 0.42 } };
        this.results = { ...this.results, s02_advanced_mining: result };
        this.state = { ...this.state, cached_result_keys: Object.keys(this.results) };
        body = { ...result, state: this.state };
      } else if (path.endsWith('/s02/clustering/commit') && method === 'POST') {
        const result = { ...this.results.s02_clustering, committed_action: 'Attached cluster IDs', provenance: 'COMMITTED' };
        this.results = this.commitResponseMode === 'normal' ? { s02_clustering: result } : {};
        this.state = { ...this.state, cached_result_keys: Object.keys(this.results) };
        const resultField = this.commitResponseMode === 'normal' ? { result } : this.commitResponseMode === 'explicit-null' ? { result: null } : {};
        body = { action: result.committed_action, rows_after: 100, ...resultField, state: this.state };
      } else if (path.endsWith('/s02/anomaly/commit') && method === 'POST') {
        const result = { ...this.results.s02_anomaly, committed_action: 'Removed anomalies', committed_mode: 'remove_anomalies', provenance: 'COMMITTED' };
        this.results = this.commitResponseMode === 'normal' ? { s02_anomaly: result } : {};
        this.state = { ...this.state, cached_result_keys: Object.keys(this.results) };
        const resultField = this.commitResponseMode === 'normal' ? { result } : this.commitResponseMode === 'explicit-null' ? { result: null } : {};
        body = { action: result.committed_action, rows_after: 98, ...resultField, state: this.state };
      } else if (path.endsWith('/s07/explain') && method === 'POST') {
        this.results = { ...this.results, s07_explain: this.explainResponse };
        body = this.explainResponse;
      }
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
      if (path.endsWith('/results')) this.completedResults += 1;
      if (path.endsWith('/s00/data_relationships')) this.completedRelationships += 1;
      if (path.endsWith('/s14/models')) this.completedSupervisoryStatus += 1;
    });
  }

  release() {
    this.holdResults = false;
    for (const resolve of this.held.splice(0)) resolve();
  }

  releaseRelationships() {
    this.holdRelationships = false;
    for (const resolve of this.heldRelationships.splice(0)) resolve();
  }

  releaseSupervisoryStatus() {
    this.holdSupervisoryStatus = false;
    for (const resolve of this.heldSupervisoryStatus.splice(0)) resolve();
  }
}

async function openStudio(page: Page, id: string) {
  await page.locator(`[data-studio="${id}"]`).click();
}

async function paintedPixels(page: Page, selector: string) {
  return page.locator(selector).evaluate((el: HTMLCanvasElement) => {
    const pixels = el.getContext('2d')!.getImageData(0, 0, el.width, el.height).data;
    let painted = 0;
    for (let i = 3; i < pixels.length; i += 4) if (pixels[i]) painted++;
    return painted;
  });
}

test('canvas scatter is drawn in the studio, zoom and pop-out and redraws after navigation and resize', async ({ page }) => {
  const backend = new MockWorkbench(initialResults());
  await backend.install(page);
  await page.goto('/');
  await openStudio(page, '02');
  const original = '.inspect-body canvas';
  await expect(page.locator(original)).toBeVisible();
  await expect.poll(() => paintedPixels(page, original)).toBeGreaterThan(50);
  const beforeWidth = await page.locator(original).evaluate((el: HTMLCanvasElement) => el.width);

  await page.getByRole('button', { name: 'Zoom Cluster Map (kmeans)' }).click();
  const zoom = '.inspect-modal-body canvas';
  await expect(page.locator(zoom)).toBeVisible();
  await expect.poll(() => paintedPixels(page, zoom)).toBeGreaterThan(50);
  await page.locator('.inspect-modal-header button[title="Close"]').click();

  await page.setViewportSize({ width: 1100, height: 800 });
  await expect.poll(() => page.locator(original).evaluate((el: HTMLCanvasElement) => el.width)).not.toBe(beforeWidth);
  await expect.poll(() => paintedPixels(page, original)).toBeGreaterThan(50);
  await openStudio(page, '04');
  await openStudio(page, '02');
  await expect.poll(() => paintedPixels(page, original)).toBeGreaterThan(50);

  const [popup] = await Promise.all([
    page.waitForEvent('popup'),
    page.getByRole('button', { name: 'Pop out Cluster Map (kmeans)' }).click(),
  ]);
  await expect(popup.locator('.stage img')).toHaveCount(1);
  await expect.poll(() => popup.locator('.stage img').evaluate((el: HTMLImageElement) => el.naturalWidth)).toBeGreaterThan(0);
  const expected = await page.locator(original).evaluate((el: HTMLCanvasElement) => el.toDataURL('image/png'));
  await expect(popup.locator('.stage img')).toHaveAttribute('src', expected);
});

test('server chart payloads render correctly across studios and survive navigation and reload', async ({ page }) => {
  const backend = new MockWorkbench(initialResults());
  await backend.install(page);
  await page.goto('/');
  await openStudio(page, '02');
  await page.getByRole('button', { name: /3\. Data Mining Lab/ }).click();
  await expect(page.getByText('Explained Variance: 68.5%')).toBeVisible();
  await expect(page.getByText('2D SVD Manifold Projection')).toBeVisible();
  await expect.poll(() => paintedPixels(page, '.inspect-body canvas')).toBeGreaterThan(50);

  await openStudio(page, '04');
  await expect(page.getByText('Distribution: alpha')).toBeVisible();
  await expect(page.getByText('Numeric Feature Correlation Matrix')).toBeVisible();
  await expect(page.locator('td[title="Correlation unavailable"]')).toHaveCount(2);
  await openStudio(page, '05');
  await expect(page.getByText('Feature Importance / Consensus Ranking')).toBeVisible();
  await openStudio(page, '07');
  await expect(page.getByText('0.4321')).toBeVisible();
  await expect(page.getByText('-0.1250')).toBeVisible();
  await expect(page.getByText('NaN', { exact: true })).toHaveCount(0);
  const importance = page.locator('.inspectable-box').filter({ hasText: 'Global Permutation Importance' });
  const betaBar = importance.locator('[title="beta"]').locator('..').locator('div > div').last();
  await expect(betaBar).toHaveCSS('left', /px/);
  const negativeGeometry = await betaBar.evaluate((el: HTMLElement) => ({ left: el.offsetLeft, width: el.offsetWidth, parentWidth: el.parentElement!.clientWidth }));
  expect(negativeGeometry.left + negativeGeometry.width).toBeLessThanOrEqual(negativeGeometry.parentWidth / 2 + 1);
  await openStudio(page, '08');
  await expect(page.getByText('Forecast & Prediction Intervals')).toBeVisible();
  await openStudio(page, '09');
  await page.getByRole('button', { name: 'Graph Link Prediction & ML' }).click();
  await expect(page.getByText('Graph Topology & Link Prediction')).toBeVisible();
  await openStudio(page, '12');
  await expect(page.getByAltText('Report visual')).toBeVisible();
  await openStudio(page, '14');
  await expect(page.getByText('SYSTEM STATUS: NORMAL')).toBeVisible();
  const spc = page.locator('.inspectable-box').filter({ hasText: 'Shewhart SPC Residual Control Chart' });
  await expect(spc.locator('line[stroke="#fbbf24"]')).toHaveCount(2);
  await expect(spc.getByText('CL 1.5')).toBeVisible();

  await page.reload();
  await openStudio(page, '07');
  await expect(page.getByText('0.4321')).toBeVisible();
  await openStudio(page, '04');
  await expect(page.getByText('Distribution: alpha')).toBeVisible();
});

test('committed cluster preview is labelled historical immediately and after reload', async ({ page }) => {
  const backend = new MockWorkbench({ s02_clustering: initialResults().s02_clustering });
  await backend.install(page);
  await page.goto('/');
  await openStudio(page, '02');
  await expect(page.getByText('Clustering Preview — Winner:', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'Attach cluster_id Column to Dataset' }).click();
  const notice = page.getByText('This chart, its counts and metrics show the original pre-commit preview.', { exact: false });
  await expect(notice).toBeVisible();
  await expect.poll(() => paintedPixels(page, '.inspect-body canvas')).toBeGreaterThan(50);
  await page.reload();
  await openStudio(page, '02');
  await expect(notice).toBeVisible();
  await expect.poll(() => paintedPixels(page, '.inspect-body canvas')).toBeGreaterThan(50);
});

test('committed anomaly chart keeps original counts and historical label after reload', async ({ page }) => {
  const backend = new MockWorkbench({
    s02_anomaly: { method: 'mad', n_anomalies: 2, anomaly_rate_pct: 2, scatter },
  });
  await backend.install(page);
  await page.goto('/');
  await openStudio(page, '02');
  await page.getByRole('button', { name: /2\. Anomaly Detection Screening/ }).click();
  const remove = page.getByRole('button', { name: 'Remove 2 Anomalies' });
  await remove.click();
  const notice = page.getByText('This chart and its anomaly counts show the original pre-commit preview.', { exact: false });
  await expect(notice).toBeVisible();
  await expect(remove).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Attach is_anomaly & Score' })).toBeDisabled();
  await expect(page.getByText('Detected Anomalies: 2 (2%)', { exact: false })).toBeVisible();
  await page.reload();
  await openStudio(page, '02');
  await page.getByRole('button', { name: /2\. Anomaly Detection Screening/ }).click();
  await expect(notice).toBeVisible();
  await expect(remove).toBeDisabled();
  await expect(page.getByText('Detected Anomalies: 2 (2%)', { exact: false })).toBeVisible();
});

for (const mode of ['legacy-invalidated', 'explicit-null'] as const) {
  for (const kind of ['clustering', 'anomaly'] as const) {
    test(`${kind} commit does not resurrect a server-invalidated preview (${mode})`, async ({ page }) => {
      const result = kind === 'clustering'
        ? { s02_clustering: initialResults().s02_clustering }
        : { s02_anomaly: { method: 'mad', n_anomalies: 2, anomaly_rate_pct: 2, scatter } };
      const backend = new MockWorkbench(result);
      backend.commitResponseMode = mode;
      await backend.install(page);
      await page.goto('/');
      await openStudio(page, '02');
      if (kind === 'clustering') {
        await expect(page.getByText('Clustering Preview — Winner:', { exact: false })).toBeVisible();
        await page.getByRole('button', { name: 'Attach cluster_id Column to Dataset' }).click();
        await expect(page.getByText('Committed: Attached cluster IDs.', { exact: false })).toBeVisible();
        await expect(page.getByText('Clustering Preview — Winner:', { exact: false })).toHaveCount(0);
      } else {
        await page.getByRole('button', { name: /2\. Anomaly Detection Screening/ }).click();
        await expect(page.getByText('Detected Anomalies: 2', { exact: false })).toBeVisible();
        await page.getByRole('button', { name: 'Remove 2 Anomalies' }).click();
        await expect(page.getByText('Committed: Removed anomalies.', { exact: false })).toBeVisible();
        await expect(page.getByText('Detected Anomalies: 2', { exact: false })).toHaveCount(0);
      }
      await expect(page.locator('.inspect-body canvas')).toHaveCount(0);
    });
  }
}

test('advanced mining follows server persistence, workspace restore and scope invalidation', async ({ page }) => {
  const backend = new MockWorkbench({});
  backend.restoredResults = { s02_advanced_mining: { target: 'restored', cluster_stability_ari: { mean_ari: 0.77 } } };
  await backend.install(page);
  await page.goto('/');
  await openStudio(page, '02');
  await page.getByRole('button', { name: /3\. Data Mining Lab/ }).click();
  await page.getByRole('button', { name: 'Run Advanced Mining (Beam + ARI + dCor)' }).click();
  const diagnostics = page.locator('pre').filter({ hasText: '"target": "first"' });
  await expect(diagnostics).toBeVisible();
  await openStudio(page, '04');
  await openStudio(page, '02');
  await expect(diagnostics).toBeVisible();
  await page.reload();
  await openStudio(page, '02');
  await page.getByRole('button', { name: /3\. Data Mining Lab/ }).click();
  await expect(diagnostics).toBeVisible();

  await page.getByTitle('Save or Restore Portable .omniforge Project').click();
  await page.locator('input[accept=".omniforge,.uap"]').setInputFiles({ name: 'restored.omniforge', mimeType: 'application/octet-stream', buffer: Buffer.from('test') });
  await expect(page.locator('pre').filter({ hasText: '"target": "restored"' })).toBeVisible();
  await expect(diagnostics).toHaveCount(0);
  await page.getByRole('button', { name: 'Exclusions (0)' }).click();
  await page.locator('.modal-panel').getByRole('button', { name: 'alpha', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Exclusions (1)' })).toBeVisible();
  await page.locator('.modal-backdrop').click({ position: { x: 5, y: 5 } });
  await expect(page.getByText('Advanced Mining Diagnostics', { exact: false })).toHaveCount(0);
});

test('a delayed cache response cannot resurrect charts invalidated by a scope change', async ({ page }) => {
  const backend = new MockWorkbench({ s04_eda: eda(1) });
  backend.holdResults = true;
  await backend.install(page);
  await page.goto('/');
  await expect.poll(() => backend.resultRequests).toBeGreaterThan(0);
  await page.getByRole('button', { name: 'Exclusions (0)' }).click();
  await page.locator('.modal-panel').getByRole('button', { name: 'alpha', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Exclusions (1)' })).toBeVisible();
  await page.locator('.modal-backdrop').click({ position: { x: 5, y: 5 } });
  backend.release();
  await expect.poll(() => backend.completedResults).toBeGreaterThan(0);
  await openStudio(page, '04');
  await expect(page.getByText('Distribution: alpha')).toHaveCount(0);
});

test('relationship charts refresh on workspace replacement and ignore stale responses', async ({ page }) => {
  const backend = new MockWorkbench({});
  const relationships = (pearson: number) => ({
    target: 'target', targets: ['target'], pairs: [],
    target_relationships: [{ feature: 'alpha', target: 'target', pearson, spearman: 0.3, dcor: 0.4, relationship_type: 'linear' }],
  });
  backend.relationshipResult = relationships(0.1234);
  await backend.install(page);
  await page.goto('/');
  await expect(page.getByText('0.1234', { exact: true })).toBeVisible();
  await expect.poll(() => backend.completedRelationships).toBe(backend.relationshipRequests);
  backend.holdRelationships = true;
  await page.getByRole('button', { name: 'Analyze Data Relationships (Chart & Table)' }).click();
  await expect.poll(() => backend.heldRelationships.length).toBe(1);

  // The replacement has identical target and exclusion names but different data.
  backend.relationshipResult = relationships(0.9876);
  backend.holdRelationships = false;
  await page.getByTitle('Save or Restore Portable .omniforge Project').click();
  await page.locator('input[accept=".omniforge,.uap"]').setInputFiles({ name: 'restored.omniforge', mimeType: 'application/octet-stream', buffer: Buffer.from('test') });
  await expect(page.getByText('0.9876', { exact: true })).toBeVisible();
  backend.releaseRelationships();
  await expect.poll(() => backend.completedRelationships).toBe(backend.relationshipRequests);
  await expect(page.getByText('0.9876', { exact: true })).toBeVisible();
  await expect(page.getByText('0.1234', { exact: true })).toHaveCount(0);
});

test('workspace restore replaces same-key cached charts, including the supervisory dashboard', async ({ page }) => {
  const backend = new MockWorkbench({ s04_eda: eda(1), s14_supervisory: supervisory });
  backend.restoredResults = { s04_eda: eda(99), s14_supervisory: { ...supervisory, system_status: 'WARNING', source: 'restored batch' } };
  await backend.install(page);
  await page.goto('/');
  await openStudio(page, '14');
  await expect(page.getByText('SYSTEM STATUS: NORMAL')).toBeVisible();
  await openStudio(page, '04');
  await expect(page.getByText('μ=1 · σ=1')).toBeVisible();
  await page.getByTitle('Save or Restore Portable .omniforge Project').click();
  await page.locator('input[accept=".omniforge,.uap"]').setInputFiles({ name: 'restored.omniforge', mimeType: 'application/octet-stream', buffer: Buffer.from('test') });
  await expect(page.getByText('μ=99 · σ=1')).toBeVisible();
  await openStudio(page, '14');
  await expect(page.getByText('SYSTEM STATUS: WARNING')).toBeVisible();
  await expect(page.getByText('restored batch', { exact: false })).toBeVisible();
});

test('workspace state synchronization fetches server results missing from the client', async ({ page }) => {
  const backend = new MockWorkbench({});
  backend.edaResponse = { s04_eda: eda(2), s05_select: initialResults().s05_select };
  await backend.install(page);
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Exclusions (0)' })).toBeVisible();
  await openStudio(page, '04');
  await page.getByRole('button', { name: 'Run Automated EDA' }).click();
  await expect(page.getByText('μ=2 · σ=1')).toBeVisible();
  await openStudio(page, '05');
  await expect(page.getByText('Feature Importance / Consensus Ranking')).toBeVisible();
});

test('supervisory invalidation hides current charts but retains the historical ledger', async ({ page }) => {
  const backend = new MockWorkbench({ s14_supervisory: supervisory });
  await backend.install(page);
  await page.goto('/');
  await openStudio(page, '14');
  await expect(page.getByText('SYSTEM STATUS: NORMAL')).toBeVisible();
  await page.getByRole('button', { name: 'Exclusions (0)' }).click();
  await page.locator('.modal-panel').getByRole('button', { name: 'alpha', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Exclusions (1)' })).toBeVisible();
  await page.locator('.modal-backdrop').click({ position: { x: 5, y: 5 } });
  await expect(page.getByText('SYSTEM STATUS: NORMAL')).toHaveCount(0);
  await expect(page.getByText('Longitudinal Supervisory Evaluation History (1 runs)')).toBeVisible();
  await expect(page.getByText('Past runs are historical; their model or study scope may differ from the current evaluation.')).toBeVisible();
  await page.reload();
  await openStudio(page, '14');
  await expect(page.getByText('SYSTEM STATUS: NORMAL')).toHaveCount(0);
  await expect(page.getByText('Longitudinal Supervisory Evaluation History (1 runs)')).toBeVisible();
});

test('an old supervisory ledger response cannot overwrite a restored project', async ({ page }) => {
  const backend = new MockWorkbench({ s14_supervisory: supervisory });
  await backend.install(page);
  await page.goto('/');
  await openStudio(page, '14');
  await expect(page.getByText('SYSTEM STATUS: NORMAL')).toBeVisible();
  await expect.poll(() => backend.completedSupervisoryStatus).toBeGreaterThan(0);

  // Delay the catalog response for the old, newly invalidated workspace.
  backend.holdSupervisoryStatus = true;
  await page.getByRole('button', { name: 'Exclusions (0)' }).click();
  await page.locator('.modal-panel').getByRole('button', { name: 'alpha', exact: true }).click();
  await page.locator('.modal-backdrop').click({ position: { x: 5, y: 5 } });
  await expect.poll(() => backend.heldSupervisoryStatus.length).toBe(1);
  await expect(page.getByText('SYSTEM STATUS: NORMAL')).toHaveCount(0);

  backend.restoredHistory = [{ source: 'restored archive batch' }];
  backend.holdSupervisoryStatus = false;
  await page.getByTitle('Save or Restore Portable .omniforge Project').click();
  await page.locator('input[accept=".omniforge,.uap"]').setInputFiles({ name: 'restored.omniforge', mimeType: 'application/octet-stream', buffer: Buffer.from('test') });
  await expect(page.getByText('restored archive batch', { exact: true })).toBeVisible();
  const completedBeforeRelease = backend.completedSupervisoryStatus;
  backend.releaseSupervisoryStatus();
  await expect.poll(() => backend.completedSupervisoryStatus).toBeGreaterThan(completedBeforeRelease);
  await expect(page.getByText('restored archive batch', { exact: true })).toBeVisible();
  await expect(page.getByText('test batch', { exact: true })).toHaveCount(0);
});

test('a delayed cache response cannot overwrite a newer locally computed result', async ({ page }) => {
  const backend = new MockWorkbench({ s07_explain: initialResults().s07_explain });
  backend.explainResponse = { global_importance: { alpha: { importance_mean: 0.8765, importance_std: 0.01 } } };
  backend.holdResults = true;
  await backend.install(page);
  await page.goto('/');
  await expect.poll(() => backend.resultRequests).toBeGreaterThan(0);
  await openStudio(page, '07');
  await page.getByRole('button', { name: 'Run Explainability' }).click();
  await expect(page.getByText('0.8765')).toBeVisible();
  backend.release();
  await expect.poll(() => backend.completedResults).toBeGreaterThan(0);
  await expect(page.getByText('0.8765')).toBeVisible();
  await expect(page.getByText('0.4321')).toHaveCount(0);
});

test('a delayed old-workspace response cannot replace a restored chart with the same key', async ({ page }) => {
  const backend = new MockWorkbench({ s04_eda: eda(1) });
  backend.restoredResults = { s04_eda: eda(99) };
  backend.holdResults = true;
  await backend.install(page);
  await page.goto('/');
  await expect.poll(() => backend.resultRequests).toBeGreaterThan(0);
  backend.holdResults = false; // The new workspace's fetch can complete first.
  await openStudio(page, '04');
  await page.getByTitle('Save or Restore Portable .omniforge Project').click();
  await page.locator('input[accept=".omniforge,.uap"]').setInputFiles({ name: 'restored.omniforge', mimeType: 'application/octet-stream', buffer: Buffer.from('test') });
  await expect(page.getByText('μ=99 · σ=1')).toBeVisible();
  backend.release();
  await expect.poll(() => backend.completedResults).toBeGreaterThan(1);
  await expect(page.getByText('μ=99 · σ=1')).toBeVisible();
  await expect(page.getByText('μ=1 · σ=1')).toHaveCount(0);
});

test('standalone vision results restore even when no dataset is loaded', async ({ page }) => {
  const backend = new MockWorkbench({ s09_vision: { zero_shot: { backend: 'native', predictions: [{ label: 'sky', probability: 0.625 }] } } });
  backend.state = { ...backend.state, has_dataset: false, dataset_name: null as any, n_rows: 0, n_cols: 0, columns: [], numeric_columns: [], targets: [] };
  await backend.install(page);
  await page.goto('/');
  await openStudio(page, '09');
  await page.getByRole('button', { name: 'Vision Foundation Hub' }).click();
  await expect(page.getByText('0.6250')).toBeVisible();
  await page.reload();
  await openStudio(page, '09');
  await page.getByRole('button', { name: 'Vision Foundation Hub' }).click();
  await expect(page.getByText('0.6250')).toBeVisible();
});

test('vision displays heuristic scores and abstention on an ambiguous image', async ({ page }) => {
  const backend = new MockWorkbench({
    s09_vision: { zero_shot: {
      backend: 'native_visual_semantic_fallback', score_kind: 'visual_cue_heuristic', top_label: null,
      predictions: [{ label: 'black', score: 2.4 }, { label: 'dark', score: 2.4 }],
    } },
  });
  await backend.install(page);
  await page.goto('/');
  await openStudio(page, '09');
  await page.getByRole('button', { name: 'Vision Foundation Hub' }).click();
  await expect(page.getByText('No supported top label.')).toBeVisible();
  await expect(page.getByText('Uncalibrated visual scores')).toBeVisible();
  await expect(page.getByText('2.4000', { exact: true })).toHaveCount(2);
});

test('scatter charts handle empty or non-finite data without inventing zero-valued points', async ({ page }) => {
  const results = initialResults();
  results.s02_clustering.scatter = { ...scatter, x: [null, null] as any, y: [null, null] as any, rendered_points: 2, total_points: 2 };
  const backend = new MockWorkbench(results);
  await backend.install(page);
  await page.goto('/');
  await openStudio(page, '02');
  await expect(page.getByText('No finite scatter points available.')).toBeVisible();
  expect(await paintedPixels(page, '.inspect-body canvas')).toBe(0);
});

test('a failed manifold projection explains why no chart is available', async ({ page }) => {
  const backend = new MockWorkbench({ s02_mining: { manifold: { error: 'not enough samples' } } });
  await backend.install(page);
  await page.goto('/');
  await openStudio(page, '02');
  await page.getByRole('button', { name: /3\. Data Mining Lab/ }).click();
  await expect(page.getByText('Manifold projection failed: not enough samples')).toBeVisible();
});

test('forecast gaps and missing interval bounds are not plotted as fabricated zeros or bands', async ({ page }) => {
  const backend = new MockWorkbench({
    s08_forecast: {
      history: [{ step: -2, actual: 1 }, { step: -1, actual: null }, { step: 0, actual: 3 }],
      deep_forecast: [{ step: 1, yhat: 4, yhat_lower: 3, yhat_upper: null }, { step: 2, yhat: 5, yhat_lower: 4, yhat_upper: 6 }],
    },
  });
  await backend.install(page);
  await page.goto('/');
  await openStudio(page, '08');
  const chart = page.locator('.inspectable-box').filter({ hasText: 'Forecast & Prediction Intervals' });
  await expect(chart).toBeVisible();
  await expect(chart.locator('polygon')).toHaveCount(0);
  const paths = await chart.locator('.inspect-body svg path').evaluateAll((els) => els.map((el) => el.getAttribute('d') || ''));
  expect(paths[0].match(/M/g)).toHaveLength(2);
  expect(paths.join(' ')).not.toMatch(/NaN|Infinity/);
});

test('cache restoration failures are visible and can be retried', async ({ page }) => {
  const backend = new MockWorkbench({ s04_eda: eda(1) });
  backend.failResults = true;
  await backend.install(page);
  await page.goto('/');
  await expect(page.getByRole('alert').filter({ hasText: 'Could not restore studio results' })).toBeVisible();
  backend.failResults = false;
  await page.getByRole('button', { name: 'Retry workspace results' }).click();
  await openStudio(page, '04');
  await expect(page.getByText('Distribution: alpha')).toBeVisible();
  await expect(page.getByRole('alert').filter({ hasText: 'Could not restore studio results' })).toHaveCount(0);
});

test('Studio 03 uses catalog IDs and displays the actual tail risk payload', async ({ page }) => {
  const mock = new MockWorkbench(initialResults());
  await mock.install(page);
  await page.route('**/api/workbench/s03/catalog', route => route.fulfill({
    json: { executable_modules: { normality_distribution: 'Catalog normality scan' } },
  }));
  await page.route('**/api/workbench/s03/run', async route => {
    expect(route.request().postDataJSON()).toMatchObject({
      mode: 'custom', selected_modules: ['normality_distribution'],
    });
    await route.fulfill({ json: { state: mock.state, report: {
      total_tests: 3, significant_tests: 1,
      risk_assessment: {
        structural_risk_index: 32.25, risk_band: 'Moderate',
        tail_risk_table: [{ Column: 'alpha', Volatility: 8 }],
      },
    } } });
  });
  await page.goto('/');
  await openStudio(page, '03');
  await page.getByRole('button', { name: /custom/i, exact: true }).click();
  await expect(page.getByRole('button', { name: 'Catalog normality scan' })).toBeVisible();
  await page.getByRole('button', { name: 'Execute Universal Diagnostics (CUSTOM)' }).click();
  await expect(page.getByText('32.3', { exact: true })).toBeVisible();
  await expect(page.getByText('Moderate', { exact: true })).toBeVisible();
  await expect(page.getByText('Column Risk Metrics', { exact: true })).toBeVisible();
});

test('Studio 06 keeps the candidate visible when its parametric reference is unavailable', async ({ page }) => {
  const mock = new MockWorkbench({
    s06_nonparametric: {
      target: 'target', task_type: 'regression', primary_metric: 'R²', elapsed_seconds: 1,
      best_model_name: 'NonParametric_WeightedKNN', best_nonparametric_score: 1,
      parametric_reference_status: 'unavailable', comparison_status: 'unavailable',
      parametric_reference_score: null, lift_vs_parametric: null, nonparametric_outperforms_parametric: null,
      verdict: 'Comparison unavailable: the parametric reference could not be evaluated.',
      leaderboard: [{ model_name: 'NonParametric_WeightedKNN', primary_score: 1, is_nonparametric: true,
        acceptance_status: 'NOT ASSESSED — PARAMETRIC REFERENCE UNAVAILABLE' }],
    },
  });
  await mock.install(page);
  await page.goto('/');
  await page.locator('[data-studio="06"]').click();
  await page.getByRole('button', { name: 'Non-Parametric Analytical Models', exact: true }).click();
  await expect(page.getByText('Top Scoring Non-Parametric Candidate')).toBeVisible();
  await expect(page.locator('.kpi-card').filter({ hasText: 'Best Non-Parametric' })).toContainText('1.0000');
  await expect(page.locator('.kpi-card').filter({ hasText: 'Parametric Reference' })).toContainText('Unavailable');
  await expect(page.locator('.kpi-card').filter({ hasText: 'Lift vs Parametric' })).toContainText('Unavailable');
  await expect(page.getByText('NOT ASSESSED — PARAMETRIC REFERENCE UNAVAILABLE', { exact: true })).toBeVisible();
  await expect(page.getByText('Non-Parametric Candidate Leaderboard', { exact: true })).toBeVisible();
  await expect(page.getByText('Non-Parametric Candidate Scores (R²)', { exact: true })).toBeVisible();
  await expect(page.getByText(/Non-Parametric vs\. Parametric/)).toHaveCount(0);
  await expect(page.getByText(/Champion Non-Parametric|Ready for Studio 14 Live Monitor/)).toHaveCount(0);
});

test('Studio 04 discloses the Shapiro sample in the custom card and per-column battery', async ({ page }) => {
  const note = '5,100 available / 5,000 tested / 100 omitted; cap 5,000 (p-value accuracy)';
  const result = {
    test_name: 'shapiro_wilk', statistic: 0.99, p_value: 0.03, significant: true,
    interpretation: 'Departure from normality.', sample_disclosure: note,
  };
  const backend = new MockWorkbench({ s04_hypothesis: { mode: 'custom', result } });
  await backend.install(page);
  await page.route('**/api/workbench/s04/hypothesis', async (route) => {
    await route.fulfill({ json: {
      mode: 'complete', total_tests: 1, significant_tests: 1,
      battery: [{ scope: 'alpha', test_type: 'shapiro', ...result }],
      state: backend.state,
    } });
  });
  await page.goto('/');
  await openStudio(page, '04');
  await expect(page.getByText(note, { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'Complete (Select All)', exact: true }).click();
  await page.getByRole('button', { name: 'Run Hypothesis Test (COMPLETE)' }).click();
  await expect(page.getByRole('row').filter({ hasText: 'alpha' }).getByText(note, { exact: true })).toBeVisible();
});

test('Studio 03 submits canonical criteria values', async ({ page }) => {
  const backend = new MockWorkbench({});
  await backend.install(page);
  await page.route('**/api/workbench/s03/run', route => route.fulfill({
    json: { state: backend.state, report: {} },
  }));
  await page.goto('/');
  await openStudio(page, '03');
  await page.getByRole('combobox', { name: /Correction/ }).selectOption({ label: 'Holm-Bonferroni' });
  await page.getByRole('combobox', { name: 'Parametric Policy' }).selectOption({ label: 'Non-Parametric' });
  for (const [label, value] of [
    ['Shapiro-Wilk', 'shapiro_wilk'], ["D'Agostino K²", 'dagostino_pearson'],
    ['Anderson-Darling', 'anderson_darling'], ['Kolmogorov-Smirnov', 'kolmogorov_smirnov'],
  ]) {
    await page.getByRole('combobox', { name: 'Normality Test' }).selectOption({ label });
    const request = page.waitForRequest('**/api/workbench/s03/run');
    await page.getByRole('button', { name: 'Execute Universal Diagnostics (COMPLETE)' }).click();
    expect((await request).postDataJSON()).toMatchObject({
      fdr_method: 'holm', parametric_mode: 'non_parametric', normality_method: value,
    });
  }
});

test('Studio 03 charts the VIF map without capping values or using the live form threshold', async ({ page }) => {
  const backend = new MockWorkbench({ s03_diagnostics: { report: {
    criteria: { vif_threshold: 11 }, vif_scores: { x: 65.2236, x2: 150.1234, singular: null },
  } } });
  await backend.install(page);
  await page.goto('/');
  await openStudio(page, '03');
  const chart = page.locator('.inspectable-box').filter({ hasText: /Variance Inflation Factor/ });
  await expect(chart).toContainText('Threshold = 11');
  await expect(chart.getByText('x', { exact: true })).toBeVisible();
  await expect(chart.getByText('65.2236', { exact: true })).toBeVisible();
  await expect(chart.getByText('150.1234', { exact: true })).toBeVisible();
  await expect(chart.getByText('singular', { exact: true })).toHaveCount(0);
  await expect(page.getByText('1 non-finite VIF score(s) not charted.')).toBeVisible();
});

for (const legacy of [false, true]) {
  test(`Studio 06 presents ${legacy ? 'restored legacy' : 'current'} CV results without a significance claim`, async ({ page }) => {
    const verdict = 'Observed CV lift of +0.02100 r2 exceeds the heuristic threshold of 0.015. This is a descriptive model comparison, not a statistical test of nonlinearity.';
    const backend = new MockWorkbench({ s06_nonparametric: {
      target: 'target', task_type: 'regression', primary_metric: 'r2', lift_vs_parametric: 0.021,
      comparison_status: 'available', nonparametric_outperforms_parametric: true,
      ...(legacy
        ? { verdict: 'Observed lift — significant non-linear structure confirmed.' }
        : { verdict, cv_lift_heuristic_threshold: 0.015 }),
    } });
    await backend.install(page);
    await page.goto('/');
    await openStudio(page, '06');
    await page.getByRole('button', { name: 'Non-Parametric Analytical Models', exact: true }).click();
    await expect(page.getByText(/Descriptive CV Comparison/)).toBeVisible();
    await expect(page.getByText(legacy ? /compares observed CV scores/ : /exceeds the heuristic threshold/)).toBeVisible();
    await expect(page.getByText(/significant non-linear structure confirmed/)).toHaveCount(0);
  });
}

test('Studio 06 distinguishes a higher-ranked reference from the selected model', async ({ page }) => {
  const backend = new MockWorkbench({ s06_automl: {
    target: 'target', backend_used: 'native', primary_metric: 'f1_weighted',
    best_model_name: 'random_forest', best_score: 0.9333, lift_over_baseline: -0.0332,
    leaderboard: [
      { model: 'tree_baseline', family: 'baseline', f1_weighted: 1 },
      { model: 'random_forest', family: 'classical_ml', f1_weighted: 0.97 },
    ],
    equation: { model_name: 'random_forest', formula_plain: 'y = x' },
    metadata: { evaluation: { leaderboard_partition: 'validation', best_score_partition: 'test', n_test: 30 } },
  } });
  await backend.install(page);
  await page.goto('/');
  await openStudio(page, '06');
  const baseline = page.getByRole('row', { name: /tree_baseline/ });
  await expect(baseline.getByRole('cell', { name: 'Reference baseline' })).toBeVisible();
  await expect(page.getByRole('row', { name: /random_forest/ }).getByRole('cell', { name: 'Selected model' })).toBeVisible();
  await expect(page.getByText(/A baseline can rank first/)).toBeVisible();
  await expect(page.getByText('Test f1_weighted', { exact: true })).toBeVisible();
  await expect(page.getByText(/Rank #1/)).toHaveCount(0);
});

for (const calibrated of [false, true]) {
  test(`Studio 08 distinguishes ${calibrated ? 'rolling-calibrated' : 'native heuristic'} ML forecast bounds`, async ({ page }) => {
    const points = [{ step: 1, yhat: 4, yhat_lower: 3, yhat_upper: 5 }];
    const backend = new MockWorkbench({ s08_forecast: {
      history: [{ step: 0, actual: 3 }], deep_forecast: points,
      classical_summary: { champion_model: 'ML_RandomForest', forecast: points, metadata: {
        champion_native_interval: calibrated ? null : { method: 'heuristic_refit_one_step_residual_band', coverage_guaranteed: false },
        champion_interval_calibration: { status: calibrated ? 'applied' : 'insufficient_history_or_origins' },
      } },
    } });
    await backend.install(page);
    await page.goto('/');
    await openStudio(page, '08');
    if (calibrated) {
      await expect(page.getByText('ML_RandomForest with calibrated intervals', { exact: true })).toBeVisible();
      await expect(page.getByText(/Native ML bands are heuristic/)).toHaveCount(0);
    } else {
      await expect(page.getByText('ML_RandomForest with heuristic intervals', { exact: true })).toBeVisible();
      await expect(page.getByText(/Coverage is not guaranteed/)).toBeVisible();
      await expect(page.getByText(/retains its native intervals/)).toBeVisible();
    }
  });
}
