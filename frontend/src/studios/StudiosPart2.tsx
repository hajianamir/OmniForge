import React, { useEffect, useState } from 'react';
import { apiGet, apiPost, apiUpload, toErrorMessage } from '../api';
import { BarChartSVG, ForecastFanSVG } from '../components/Charts';
import { SimpleTable } from '../components/DataGrid';
import { StudioHelpButton } from '../components/HelpGuideModal';
import { MathEquationCard } from '../components/MathEquation';
import { ModeBar, type ExecutionMode } from '../components/ModeBar';
import type {
  AutoMLResponse,
  BackendInfo,
  BackendsResponse,
  BatchPredictionResponse,
  CausalResponse,
  DeepLearningResponse,
  ExplainResponse,
  FairnessResponse,
  ForecastResponse,
  NonParametricResponse,
  OptimizeResponse,
  SimulateResponse,
  SinglePredictionResponse,
} from '../studioTypes';
import { ProvenancePill, type StudioProps } from './StudiosPart1';

// ============================================================================
// Studio 06 — Multi-Backend AutoML, Non-Parametric Analytical Models, Deep Learning & Predictions
// ============================================================================
export const Studio06Modeling: React.FC<StudioProps> = ({ ws, onSync, studioCache, setStudioResult, openHelp }) => {
  const [tab, setTab] = useState<'automl' | 'nonparametric' | 'deep' | 'predict'>('automl');
  const [backends, setBackends] = useState<BackendInfo[]>([]);
  const [selectedBackend, setSelectedBackend] = useState('native_stacking');
  const [budget, setBudget] = useState(25);
  const [cvFolds, setCvFolds] = useState(3);
  const automlRes = studioCache.s06_automl ?? null;

  // Non-Parametric Analytical Suite state
  const [npMode, setNpMode] = useState<ExecutionMode>('complete');
  const [npModels, setNpModels] = useState<string[]>(['gam_spline', 'nadaraya_watson', 'kernel_ridge_rbf', 'weighted_knn', 'isotonic']);
  const [npKnots, setNpKnots] = useState(5);
  const [npDegree, setNpDegree] = useState(3);
  const [npBandwidth, setNpBandwidth] = useState<string>('');
  const [npKNeighbors, setNpKNeighbors] = useState(11);
  const [npGamma, setNpGamma] = useState<string>('');
  const npRes = studioCache.s06_nonparametric ?? null;

  const [dlArchs, setDlArchs] = useState<string[]>(['ft_transformer', 'tabnet', 'resnet_tabular', 'mlp', 'gnn_gcn']);
  const [dlEpochs, setDlEpochs] = useState(20);
  const dlRes = studioCache.s06_dl ?? null;

  const [singleRec, setSingleRec] = useState<Record<string, string>>({});
  const inputCols = ws.active_input_feature_cols?.length ? ws.active_input_feature_cols : ws.active_feature_cols;
  const singlePred = studioCache.s06_predict_single ?? null;
  const batchRes = studioCache.s06_predict_batch ?? null;
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    apiGet<BackendsResponse>('/api/workbench/s06/backends')
      .then((res) => setBackends(res.automl_backends || []))
      .catch(() => {});
  }, []);

  const runAutoML = async () => {
    setBusy(true); setErr(null);
    try {
      const res = await apiPost<AutoMLResponse>('/api/workbench/s06/automl', {
        backend: selectedBackend,
        time_budget_seconds: budget,
        cv_folds: cvFolds,
      });
      onSync(res.state);
      setStudioResult('s06_automl', res);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  const toggleNpModel = (m: string) => {
    setNpModels((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]));
  };

  const runNonParametric = async () => {
    setBusy(true); setErr(null);
    try {
      const allModels = ['gam_spline', 'nadaraya_watson', 'kernel_ridge_rbf', 'weighted_knn', 'isotonic'];
      const chosen = npMode === 'custom' ? npModels : npMode === 'auto' ? ['gam_spline', 'nadaraya_watson', 'kernel_ridge_rbf', 'weighted_knn'] : allModels;
      const res = await apiPost<NonParametricResponse>('/api/workbench/s06/nonparametric', {
        models: chosen.length ? chosen : allModels,
        n_knots: npKnots,
        spline_degree: npDegree,
        bandwidth: npBandwidth.trim() ? Number(npBandwidth) : null,
        n_neighbors: npKNeighbors,
        rbf_gamma: npGamma.trim() ? Number(npGamma) : null,
        include_parametric_ref: true,
      });
      onSync(res.state);
      setStudioResult('s06_nonparametric', res);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  const toggleArch = (a: string) => {
    setDlArchs((prev) => (prev.includes(a) ? prev.filter((x) => x !== a) : [...prev, a]));
  };

  const runDeep = async () => {
    setBusy(true); setErr(null);
    try {
      const res = await apiPost<DeepLearningResponse>('/api/workbench/s06/deep-learning', { architectures: dlArchs, epochs: dlEpochs });
      onSync(res.state);
      setStudioResult('s06_dl', res);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  const runSinglePred = async () => {
    setBusy(true); setErr(null);
    try {
      const record: Record<string, string> = {};
      for (const c of inputCols) if (singleRec[c]?.trim()) record[c] = singleRec[c];
      const res = await apiPost<SinglePredictionResponse>('/api/workbench/s06/predict-single', { record });
      setStudioResult('s06_predict_single', res);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  const runBatchPred = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true); setErr(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await apiUpload<BatchPredictionResponse>('/api/workbench/s06/predict-batch', fd);
      setStudioResult('s06_predict_batch', res);
    } catch (er: unknown) { setErr(toErrorMessage(er)); } finally { setBusy(false); e.target.value = ''; }
  };

  return (
    <div>
      <div className="studio-hero">
        <div>
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <h1>06 · Multi-Backend AutoML, Non-Parametric Analytical &amp; Deep Learning</h1>
            <StudioHelpButton section="studio-06" onOpenHelp={openHelp} title="Open Chapter 06: B-Spline GAM, Nadaraya-Watson, RBF Kernel Ridge, AutoML & Deep Learning Formulas" />
          </div>
          <p>4 Native SOTA AutoML Engines + 5 External Adapters, 5 True Non-Parametric Analytical Models (B-Spline GAM, Nadaraya-Watson, RBF Kernel Ridge, Weighted k-NN, Isotonic) with Closed-Form Equations, Deep Learning &amp; Predictions.</p>
        </div>
        <ProvenancePill ws={ws} />
      </div>
      {err && <div className="alert alert-error">{err}</div>}
      <div className="tabs">
        <button className={`tab-btn ${tab === 'automl' ? 'active' : ''}`} onClick={() => setTab('automl')}>Multi-Backend AutoML Hub</button>
        <button className={`tab-btn ${tab === 'nonparametric' ? 'active' : ''}`} onClick={() => setTab('nonparametric')}>Non-Parametric Analytical Models</button>
        <button className={`tab-btn ${tab === 'deep' ? 'active' : ''}`} onClick={() => setTab('deep')}>Paper-Grade Deep Learning Arena</button>
        <button className={`tab-btn ${tab === 'predict' ? 'active' : ''}`} onClick={() => setTab('predict')}>Future Target Predictions</button>
      </div>

      {tab === 'automl' && (
        <>
          <div className="card">
            <div className="card-title">
              <span className="row" style={{ gap: 6, alignItems: 'center' }}>
                Select AutoML Engine (Native &amp; Plug-and-Play External Backends)
                <StudioHelpButton section="model-atlas" onOpenHelp={openHelp} label="Model Selection Atlas" title="Open Master Model & Algorithm Selection Atlas" />
              </span>
            </div>
            <div className="grid-3" style={{ marginBottom: 14 }}>
              {backends.map((b) => (
                <div
                  key={b.id}
                  onClick={() => setSelectedBackend(b.id)}
                  style={{
                    padding: '10px 12px',
                    borderRadius: 10,
                    border: `1px solid ${selectedBackend === b.id ? 'var(--accent)' : 'var(--border)'}`,
                    background: selectedBackend === b.id ? 'var(--accent-bg)' : 'var(--bg)',
                    cursor: 'pointer',
                  }}
                >
                  <div className="row" style={{ justifyContent: 'space-between', marginBottom: 4 }}>
                    <strong style={{ fontSize: 12.5 }}>{b.name}</strong>
                    <span className={`badge ${b.available ? 'badge-green' : 'badge-amber'}`}>
                      {b.available ? (b.category === 'native' ? 'NATIVE' : `v${b.version || 'OK'}`) : 'FALLBACK'}
                    </span>
                  </div>
                  <div style={{ fontSize: 11.5, color: 'var(--text-dim)' }}>{b.description}</div>
                </div>
              ))}
            </div>
            <div className="row">
              <label className="field">Time Budget (sec)
                <input type="number" min={5} max={180} value={budget} onChange={(e) => setBudget(Number(e.target.value))} style={{ width: 100 }} />
              </label>
              <label className="field">CV Folds
                <input type="number" min={2} max={10} value={cvFolds} onChange={(e) => setCvFolds(Number(e.target.value))} style={{ width: 80 }} />
              </label>
              <button className="btn btn-primary" disabled={busy || !ws.has_dataset} onClick={runAutoML}>
                {busy ? <span className="spinner" /> : null} Launch AutoML Tournament
              </button>
            </div>
          </div>

          {automlRes && (
            <div className="card">
              <div className="card-title">
                <span>Champion Model Leaderboard ({automlRes.backend_used ?? 'native'})</span>
                <span className="badge badge-green">Lift vs Baseline: {Number(automlRes.lift_over_baseline ?? 0) >= 0 ? '+' : ''}{Number(automlRes.lift_over_baseline ?? 0).toFixed(4)}</span>
              </div>
              {automlRes.fallback_reason && <div className="alert alert-warn">{automlRes.fallback_reason}</div>}
              <div className="kpi-grid">
                <div className="kpi-card"><div className="kpi-label">Best Model</div><div className="kpi-value" style={{ fontSize: 16 }}>{automlRes.best_model_name}</div></div>
                <div className="kpi-card"><div className="kpi-label">{automlRes.primary_metric}</div><div className="kpi-value">{Number(automlRes.best_score).toFixed(4)}</div></div>
                <div className="kpi-card"><div className="kpi-label">Strong Baseline</div><div className="kpi-value">{Number(automlRes.baseline_score).toFixed(4)}</div></div>
                <div className="kpi-card"><div className="kpi-label">Runtime</div><div className="kpi-value">{automlRes.elapsed_seconds}s</div></div>
              </div>
              <SimpleTable title="Ranked Model Leaderboard & Acceptance Rationale" rows={automlRes.leaderboard || []} />
              {automlRes.equation && (automlRes.equation.formula_plain || automlRes.equation.equation_text) && (
                <div style={{ marginTop: 14 }}>
                  <MathEquationCard
                    defaultOpenCode
                    eq={{
                      ...automlRes.equation,
                      model_name: automlRes.equation.model_name || automlRes.best_model_name || '',
                      formula_plain: automlRes.equation.formula_plain || automlRes.equation.equation_text,
                      rank: 1,
                      primary_metric: automlRes.primary_metric,
                      primary_score: automlRes.best_score,
                      naive_floor_score: automlRes.baseline_score,
                      lift_over_naive: automlRes.lift_over_baseline,
                      selection_criterion: `Max CV ${automlRes.primary_metric ?? ''} + Positive Lift over Strong Baseline (${Number(automlRes.baseline_score).toFixed(4)})`,
                      acceptance_status: Number(automlRes.lift_over_baseline ?? 0) >= 0 ? 'ACCEPTED — AUTOML CHAMPION' : 'ACCEPTED — BEST AVAILABLE',
                      acceptance_reasons: [
                        `Ranked #1 across ${(automlRes.leaderboard || []).length} candidate pipelines in ${automlRes.elapsed_seconds}s using ${automlRes.backend_used}.`,
                        `Achieved ${automlRes.primary_metric ?? ''}=${Number(automlRes.best_score).toFixed(4)} vs strong baseline ${Number(automlRes.baseline_score).toFixed(4)} (lift ${Number(automlRes.lift_over_baseline ?? 0) >= 0 ? '+' : ''}${Number(automlRes.lift_over_baseline ?? 0).toFixed(4)}).`,
                      ],
                    }}
                  />
                </div>
              )}
            </div>
          )}
        </>
      )}

      {tab === 'nonparametric' && (
        <>
          <div className="card">
            <div className="card-title">
              <span>Non-Parametric Analytical Suite (Distribution-Free Smooth &amp; Local Estimators)</span>
              <ModeBar mode={npMode} onChange={setNpMode} />
            </div>
            <p style={{ margin: '0 0 12px', fontSize: 12.5, color: 'var(--text-muted)' }}>
              Benchmarks <strong>5 distribution-free analytical models</strong> — Penalized B-Spline GAM, Nadaraya-Watson Gaussian Kernel Estimator, RBF Kernel Ridge (RKHS Dual), Distance-Weighted k-NN, and Isotonic Monotonic Regression — head-to-head against a Parametric Linear/Logistic reference to detect &amp; quantify non-linear structure with exact mathematical formulas.
            </p>
            {npMode === 'custom' && (
              <div className="chip-grid" style={{ marginBottom: 12 }}>
                {[
                  ['gam_spline', 'Penalized B-Spline GAM'],
                  ['nadaraya_watson', 'Nadaraya-Watson Gaussian Kernel'],
                  ['kernel_ridge_rbf', 'RBF Kernel Ridge (RKHS)'],
                  ['weighted_knn', 'Distance-Weighted k-NN'],
                  ['isotonic', 'Isotonic Monotonic (PAVA)'],
                ].map(([id, label]) => (
                  <button key={id} type="button" className={`chip ${npModels.includes(id) ? 'selected' : ''}`} onClick={() => toggleNpModel(id)}>
                    {label}
                  </button>
                ))}
              </div>
            )}
            <div className="grid-4" style={{ marginBottom: 12 }}>
              <label className="field">B-Spline Knots
                <input type="number" min={3} max={12} value={npKnots} onChange={(e) => setNpKnots(Number(e.target.value))} />
              </label>
              <label className="field">Spline Degree
                <input type="number" min={1} max={3} value={npDegree} onChange={(e) => setNpDegree(Number(e.target.value))} />
              </label>
              <label className="field">Kernel Bandwidth h (blank=Scott)
                <input type="number" step={0.1} min={0.05} placeholder="Auto Scott's Rule" value={npBandwidth} onChange={(e) => setNpBandwidth(e.target.value)} />
              </label>
              <label className="field">k-NN Neighbors / RBF γ
                <div className="row" style={{ gap: 6 }}>
                  <input type="number" min={3} max={51} value={npKNeighbors} onChange={(e) => setNpKNeighbors(Number(e.target.value))} style={{ width: '50%' }} title="k-NN neighbors" />
                  <input type="number" step={0.05} min={0.001} placeholder="Auto γ" value={npGamma} onChange={(e) => setNpGamma(e.target.value)} style={{ width: '50%' }} title="RBF gamma (blank=median heuristic)" />
                </div>
              </label>
            </div>
            <button className="btn btn-primary" disabled={busy || !ws.has_dataset} onClick={runNonParametric}>
              {busy ? <span className="spinner" /> : null} Run Non-Parametric Analytical Benchmark ({npMode.toUpperCase()})
            </button>
          </div>

          {npRes && (
            <>
              <div className={`alert ${npRes.nonparametric_outperforms_parametric ? 'alert-success' : 'alert-info'}`}>
                <strong>Analytical Verdict ({npRes.target} · {npRes.task_type}):</strong> {npRes.verdict}
              </div>
              <div className="kpi-grid">
                <div className="kpi-card">
                  <div className="kpi-label">Champion Non-Parametric</div>
                  <div className="kpi-value" style={{ fontSize: 15 }}>{npRes.best_model_name}</div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Best Non-Parametric ({String(npRes.primary_metric || '').toUpperCase()})</div>
                  <div className="kpi-value">{Number(npRes.best_nonparametric_score ?? 0).toFixed(4)}</div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Parametric Reference</div>
                  <div className="kpi-value">{npRes.parametric_reference_score !== null && npRes.parametric_reference_score !== undefined ? Number(npRes.parametric_reference_score).toFixed(4) : 'N/A'}</div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Lift vs Parametric</div>
                  <div className="kpi-value" style={{ color: Number(npRes.lift_vs_parametric ?? 0) > 0 ? 'var(--green)' : 'var(--text)' }}>
                    {npRes.lift_vs_parametric !== null && npRes.lift_vs_parametric !== undefined ? `${Number(npRes.lift_vs_parametric) >= 0 ? '+' : ''}${Number(npRes.lift_vs_parametric).toFixed(4)}` : 'N/A'}
                  </div>
                </div>
              </div>

              <div className="card">
                <div className="card-title">
                  <span>Non-Parametric vs. Parametric Analytical Leaderboard</span>
                  <span className="badge badge-accent">Runtime: {npRes.elapsed_seconds}s · Ready for Studio 14 Live Monitor</span>
                </div>
                <SimpleTable title="Non-Parametric Analytical Leaderboard" rows={npRes.leaderboard || []} />
              </div>

              {(npRes.spline_components?.length ?? 0) > 0 && (
                <div className="card">
                  <div className="card-title">
                    <span>Additive Penalized B-Spline GAM Component Decomposition</span>
                    <span className="badge badge-green">Per-Feature Smooth fⱼ(xⱼ) Effects</span>
                  </div>
                  <SimpleTable title="GAM Smooth Component Contributions" rows={npRes.spline_components || []} />
                </div>
              )}

              {(npRes.equations?.length ?? 0) > 0 && (
                <div className="card">
                  <div className="card-title">
                    <span>Exact Non-Parametric Analytical Mathematical Equations</span>
                    <span className="badge badge-accent">{(npRes.equations ?? []).length} Models · Closed-Form + LaTeX + Python</span>
                  </div>
                  {(npRes.equations ?? []).map((eq, idx) => (
                    <MathEquationCard key={`np-eq-${eq.model_name || idx}`} eq={{ ...eq, model_name: eq.model_name || `NP Model ${idx + 1}`, target_name: eq.target_name || npRes.target || '' }} />
                  ))}
                </div>
              )}
            </>
          )}
        </>
      )}

      {tab === 'deep' && (
        <div className="card">
          <div className="card-title">Paper-Grade Deep Learning vs. Strong Baselines</div>
          <div className="chip-grid" style={{ marginBottom: 14 }}>
            {[
              ['ft_transformer', 'FT-Transformer ([CLS] GEGLU)'],
              ['tabnet', 'TabNet (Sparsemax Masks)'],
              ['resnet_tabular', 'ResNet-Tabular (Gorishniy 2021)'],
              ['mlp', 'Residual MLP'],
              ['cnn1d', '1D-CNN'],
              ['gnn_gcn', 'GNN (GCN)'],
              ['gnn_gat', 'GNN (Multi-Head GAT)'],
              ['gnn_graphsage', 'GNN (GraphSAGE)'],
            ].map(([id, label]) => (
              <button key={id} className={`chip ${dlArchs.includes(id) ? 'selected' : ''}`} onClick={() => toggleArch(id)}>{label}</button>
            ))}
          </div>
          <div className="row" style={{ marginBottom: 14 }}>
            <label className="field">Epochs
              <input type="number" min={5} max={100} value={dlEpochs} onChange={(e) => setDlEpochs(Number(e.target.value))} style={{ width: 90 }} />
            </label>
            <button className="btn btn-primary" disabled={busy || !ws.has_dataset || !dlArchs.length} onClick={runDeep}>
              {busy ? <span className="spinner" /> : null} Benchmark Deep Architectures
            </button>
          </div>
          {dlRes && (
            <>
              <SimpleTable rows={(dlRes.results || []).map(({ loss_curve: _lossCurve, ...rest }) => rest)} />
              {dlRes.feature_importances && Object.keys(dlRes.feature_importances).length > 0 && (
                <div className="grid-2" style={{ marginTop: 16 }}>
                  {Object.entries(dlRes.feature_importances).map(([mName, list]) => (
                    <div className="card" key={mName} style={{ marginBottom: 0 }}>
                      <div className="card-title">{mName} — Attention / Sparsemax Feature Masks</div>
                      <BarChartSVG horizontal items={(list || []).map((x) => ({ label: x.feature, value: x.importance }))} />
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {tab === 'predict' && (
        <div className="grid-2">
          <div className="card">
            <div className="card-title">Single-Record Interactive Scoring</div>
            {inputCols.length ? (
              <>
                <p>Enter raw values; leave a field blank to use the trained model's missing-value handling.</p>
                <div className="grid-2" style={{ marginBottom: 12 }}>
                  {inputCols.map((c) => (
                    <label className="field" key={c}>{c}
                      <input
                        type={ws.active_input_feature_types?.[c] === 'numeric' ? 'number' : 'text'}
                        step={ws.active_input_feature_types?.[c] === 'numeric' ? 'any' : undefined}
                        placeholder={ws.active_input_feature_types?.[c] === 'datetime' ? 'ISO 8601 datetime' : ''}
                        value={singleRec[c] ?? ''}
                        onChange={(e) => setSingleRec({ ...singleRec, [c]: e.target.value })}
                      />
                    </label>
                  ))}
                </div>
                <button className="btn btn-primary" disabled={busy} onClick={runSinglePred}>Predict Target</button>
                {singlePred && <div className="alert alert-success" style={{ marginTop: 12 }}>Prediction ({singlePred.model}): <strong>{String(singlePred.prediction)}</strong></div>}
              </>
            ) : <div style={{ color: 'var(--text-dim)' }}>Train a model in the AutoML tab first to activate scoring.</div>}
          </div>
          <div className="card">
            <div className="card-title">Batch Target-Free CSV Scoring</div>
            <input type="file" accept=".csv" onChange={runBatchPred} disabled={busy || !ws.active_model_name} />
            {batchRes && (
              <div style={{ marginTop: 12 }}>
                <div className="alert alert-success">Scored {batchRes.n_rows} future rows!</div>
                <SimpleTable rows={batchRes.preview || []} maxRows={20} />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Studio 07 — Responsible AI (Explainability, Fairness & Governance)
// ============================================================================
export const Studio07Responsible: React.FC<StudioProps> = ({ ws, studioCache, setStudioResult, openHelp }) => {
  const expRes = studioCache.s07_explain ?? null;
  const fairRes = studioCache.s07_fairness ?? null;
  const [rowIdx, setRowIdx] = useState(0);
  const [sensCol, setSensCol] = useState(ws.categorical_columns[0] || ws.columns[0] || '');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const runExplain = async () => {
    setBusy(true); setErr(null);
    try {
      const res = await apiPost<ExplainResponse>('/api/workbench/s07/explain', { row_index: rowIdx });
      setStudioResult('s07_explain', res);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  const runFair = async () => {
    setBusy(true); setErr(null);
    try {
      const res = await apiPost<FairnessResponse>('/api/workbench/s07/fairness', { sensitive_col: sensCol || ws.columns[0] });
      setStudioResult('s07_fairness', res);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  return (
    <div>
      <div className="studio-hero">
        <div>
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <h1>07 · Responsible AI — Explainability, Fairness &amp; Governance</h1>
            <StudioHelpButton section="studio-07" onOpenHelp={openHelp} title="Open Chapter 07: SHAP, Permutation Importance, PDP & Fairness 4/5ths Rule" />
          </div>
          <p>Permutation/SHAP global &amp; local attribution, Partial Dependence curves, Demographic Parity / Disparate Impact audits &amp; Model Cards.</p>
        </div>
        <ProvenancePill ws={ws} />
      </div>
      {err && <div className="alert alert-error">{err}</div>}
      <div className="grid-2">
        <div className="card">
          <div className="card-title">Global Permutation Importance &amp; Local Explanation</div>
          <div className="row" style={{ marginBottom: 12 }}>
            <label className="field">Instance Index
              <input type="number" min={0} value={rowIdx} onChange={(e) => setRowIdx(Number(e.target.value))} style={{ width: 90 }} />
            </label>
            <button className="btn btn-primary" disabled={busy || !ws.active_model_name} onClick={runExplain}>Run Explainability</button>
          </div>
          {expRes?.global_importance && (
            <BarChartSVG
              title="Global Permutation Importance"
              horizontal
              items={Object.entries(expRes.global_importance).flatMap(([label, entry]) => {
                const value = typeof entry === 'number' ? entry : entry?.importance_mean;
                return typeof value === 'number' && Number.isFinite(value) ? [{ label, value }] : [];
              })}
            />
          )}
        </div>
        <div className="card">
          <div className="card-title">Fairness Audit &amp; Compliance Model Card</div>
          <div className="row" style={{ marginBottom: 12 }}>
            <select value={sensCol} onChange={(e) => setSensCol(e.target.value)}>
              {ws.columns.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
            <button className="btn btn-primary" disabled={busy || !ws.active_model_name} onClick={runFair}>Run Fairness Audit</button>
          </div>
          {fairRes && (
            <>
              <div className="kpi-grid">
                <div className="kpi-card"><div className="kpi-label">Demographic Parity Diff</div><div className="kpi-value">{Number(fairRes.fairness?.demographic_parity_difference ?? 0).toFixed(4)}</div></div>
                <div className="kpi-card"><div className="kpi-label">Disparate Impact</div><div className="kpi-value">{Number(fairRes.fairness?.disparate_impact_ratio ?? 1).toFixed(3)}</div></div>
              </div>
              <pre className="code-block" style={{ maxHeight: 260 }}>{fairRes.model_card_markdown}</pre>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Studio 08 — Forecasting (N-BEATS / PatchTST / TFT), Causal, Opt & Simulation
// ============================================================================
export const Studio08Decision: React.FC<StudioProps> = ({ ws, onSync, studioCache, setStudioResult, openHelp }) => {
  const [tab, setTab] = useState<'forecast' | 'causal' | 'opt' | 'sim'>('forecast');
  const [targetCol, setTargetCol] = useState(ws.numeric_columns[0] || '');
  const [deepArch, setDeepArch] = useState('tft');
  const [horizon, setHorizon] = useState(14);
  const foreRes = studioCache.s08_forecast ?? null;

  const [treatCol, setTreatCol] = useState(ws.numeric_columns[0] || '');
  const [outCol, setOutCol] = useState(ws.numeric_columns[1] || ws.numeric_columns[0] || '');
  const causalRes = studioCache.s08_causal?.causal ?? null;
  const optRes = studioCache.s08_optimize?.optimization ?? null;
  const simRes = studioCache.s08_simulate?.simulation ?? null;
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const runForecast = async () => {
    setBusy(true); setErr(null);
    try {
      const res = await apiPost<ForecastResponse>('/api/workbench/s08/forecast', { target_col: targetCol || ws.numeric_columns[0], horizon, deep_arch: deepArch, epochs: 25 });
      onSync(res.state);
      setStudioResult('s08_forecast', res);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  return (
    <div>
      <div className="studio-hero">
        <div>
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <h1>08 · Forecasting, Causal Inference, Optimization &amp; Simulation</h1>
            <StudioHelpButton section="studio-08" onOpenHelp={openHelp} title="Open Chapter 08: ARIMA/TFT Forecasting, Doubly Robust AIPW Causal ATE, Markowitz & Monte Carlo" />
          </div>
          <p>Paper-grade N-BEATS, PatchTST &amp; Quantile TFT neural forecasters, AIPW Causal Inference, Markowitz Optimization &amp; Monte Carlo Risk.</p>
        </div>
        <ProvenancePill ws={ws} />
      </div>
      {err && <div className="alert alert-error">{err}</div>}
      <div className="tabs">
        <button className={`tab-btn ${tab === 'forecast' ? 'active' : ''}`} onClick={() => setTab('forecast')}>Deep &amp; Classical Forecasting</button>
        <button className={`tab-btn ${tab === 'causal' ? 'active' : ''}`} onClick={() => setTab('causal')}>Causal Inference (ATE/AIPW)</button>
        <button className={`tab-btn ${tab === 'opt' ? 'active' : ''}`} onClick={() => setTab('opt')}>Portfolio Optimization</button>
        <button className={`tab-btn ${tab === 'sim' ? 'active' : ''}`} onClick={() => setTab('sim')}>Monte Carlo Simulation</button>
      </div>

      {tab === 'forecast' && (
        <div className="card">
          <div className="row" style={{ marginBottom: 14 }}>
            <label className="field">Target Series
              <select value={targetCol} onChange={(e) => setTargetCol(e.target.value)}>
                {ws.numeric_columns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            <label className="field">Deep Forecaster Architecture
              <select value={deepArch} onChange={(e) => setDeepArch(e.target.value)}>
                <option value="tft">TFT — Temporal Fusion Transformer (P10/P50/P90 Quantiles)</option>
                <option value="nbeats">N-BEATS — Doubly Residual Trend + Fourier Stacks</option>
                <option value="patchtst">PatchTST — Patch Token Self-Attention</option>
                <option value="lstm">LSTM Recurrent Sequence</option>
              </select>
            </label>
            <label className="field">Horizon
              <input type="number" min={3} max={60} value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} style={{ width: 85 }} />
            </label>
            <button className="btn btn-primary" disabled={busy || !ws.has_dataset} onClick={runForecast}>
              {busy ? <span className="spinner" /> : null} Forecast Now
            </button>
          </div>
          {foreRes && (
            <>
              <ForecastFanSVG history={foreRes.history || []} forecast={foreRes.deep_forecast || []} />
              <div style={{ marginTop: 12 }}><SimpleTable rows={foreRes.deep_forecast || []} /></div>
            </>
          )}
        </div>
      )}

      {tab === 'causal' && (
        <div className="card">
          <div className="row" style={{ marginBottom: 12 }}>
            <label className="field">Treatment
              <select value={treatCol} onChange={(e) => setTreatCol(e.target.value)}>
                {ws.columns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            <label className="field">Outcome
              <select value={outCol} onChange={(e) => setOutCol(e.target.value)}>
                {ws.numeric_columns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            <button className="btn btn-primary" disabled={busy || !ws.has_dataset} onClick={async () => {
              setBusy(true); setErr(null);
              try {
                const r = await apiPost<CausalResponse>('/api/workbench/s08/causal', { treatment_col: treatCol || ws.columns[0], outcome_col: outCol || ws.numeric_columns[0] });
                if (r.state) onSync(r.state);
                setStudioResult('s08_causal', r);
              } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
            }}>Estimate Causal ATE</button>
          </div>
          {causalRes && <pre className="code-block">{JSON.stringify(causalRes, null, 2)}</pre>}
        </div>
      )}

      {tab === 'opt' && (
        <div className="card">
          <button className="btn btn-primary" disabled={busy || !ws.has_dataset} onClick={async () => {
            setBusy(true); setErr(null);
            try {
              const r = await apiPost<OptimizeResponse>('/api/workbench/s08/optimize', {});
              setStudioResult('s08_optimize', r);
            } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
          }}>Run Markowitz Mean-Variance Portfolio Optimizer</button>
          {optRes && <pre className="code-block" style={{ marginTop: 12 }}>{JSON.stringify(optRes, null, 2)}</pre>}
        </div>
      )}

      {tab === 'sim' && (
        <div className="card">
          <div className="row" style={{ marginBottom: 12 }}>
            <select value={targetCol} onChange={(e) => setTargetCol(e.target.value)}>
              {ws.numeric_columns.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
            <button className="btn btn-primary" disabled={busy || !ws.has_dataset} onClick={async () => {
              setBusy(true); setErr(null);
              try {
                const r = await apiPost<SimulateResponse>('/api/workbench/s08/simulate', { target_col: targetCol || ws.numeric_columns[0], n_simulations: 3000 });
                setStudioResult('s08_simulate', r);
              } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
            }}>Run 3,000-Trial Monte Carlo</button>
          </div>
          {simRes && <pre className="code-block">{JSON.stringify(simRes, null, 2).slice(0, 2600)}</pre>}
        </div>
      )}
    </div>
  );
};
