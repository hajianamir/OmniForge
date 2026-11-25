import React, { useEffect, useState } from 'react';
import { apiDownload, apiGet, apiPost, apiUpload, toErrorMessage } from '../api';
import { BarChartSVG, LineSeriesSVG, NetworkGraphSVG } from '../components/Charts';
import { SimpleTable } from '../components/DataGrid';
import { FeatureFreshnessPanel } from '../components/FeatureFreshnessPanel';
import { StudioHelpButton } from '../components/HelpGuideModal';
import { InspectableBox } from '../components/InspectModal';
import type {
  BackendsProbeResponse,
  ChatResponse,
  ChatTurn,
  CompiledReportResponse,
  DAGResponse,
  DriftResponse,
  GraphResponse,
  MLOpsOverviewResponse,
  NLPResponse,
  RecommendResponse,
  RegisterResponse,
  ReportCatalogResponse,
  ReportSection,
  StateEnvelope,
  SurvivalResponse,
  VisionResponse,
} from '../studioTypes';
import { ProvenancePill, type StudioProps } from './StudiosPart1';

// ============================================================================
// Studio 09 — Specialized AI (Extended NLP, Vision Foundation Hub, Full Graph Link Prediction, RecSys, Survival)
// ============================================================================
export const Studio09Specialized: React.FC<StudioProps> = ({ ws, studioCache, setStudioResult, openHelp }) => {
  const [tab, setTab] = useState<'nlp' | 'vision' | 'graph' | 'recsys' | 'survival'>('nlp');
  const [backends, setBackends] = useState<BackendsProbeResponse | null>(null);

  // NLP
  const [nlpTask, setNlpTask] = useState('overview');
  const [nlpText, setNlpText] = useState(
    'Quarterly revenue grew 18% with strong customer trust and reliable cloud delivery.\nSecurity alert: user john.doe@example.com reported a potential fraud breach from 192.168.1.44.\nCustomer churn increased after terrible service outage and unacceptable billing delays.'
  );
  const [nlpLabels, setNlpLabels] = useState('Finance, Security, Operations, Customer Support');
  const [nlpQuery, setNlpQuery] = useState('What grew by 18%?');
  const nlpRes = studioCache.s09_nlp ?? null;

  // Vision
  const [visLabels, setVisLabels] = useState('sky, forest, building, industrial, document, dark');
  const visRes = studioCache.s09_vision ?? null;

  // Graph
  const [linkMethod, setLinkMethod] = useState('ensemble_rrf');
  const [edgeOp, setEdgeOp] = useState('hadamard');
  const graphRes = studioCache.s09_graph ?? null;

  // Recsys & Survival
  const recRes = studioCache.s09_recommend ?? null;
  const survRes = studioCache.s09_survival ?? null;
  const [durCol, setDurCol] = useState(ws.numeric_columns[0] || '');
  const [evtCol, setEvtCol] = useState(ws.numeric_columns[1] || ws.numeric_columns[0] || '');

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    apiGet<BackendsProbeResponse>('/api/workbench/s09/backends').then(setBackends).catch(() => {});
  }, []);

  const runNLP = async () => {
    setBusy(true); setErr(null);
    try {
      const res = await apiPost<NLPResponse>('/api/workbench/s09/nlp', {
        task: nlpTask,
        custom_texts: nlpText.split('\n').filter((s) => s.trim()),
        candidate_labels: nlpLabels.split(',').map((s) => s.trim()).filter(Boolean),
        question: nlpQuery,
        query: nlpQuery,
      });
      setStudioResult('s09_nlp', res);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  const runVision = async (file?: File) => {
    setBusy(true); setErr(null);
    try {
      const fd = new FormData();
      if (file) fd.append('file', file);
      fd.append('candidate_labels', visLabels);
      fd.append('n_segments', '4');
      setStudioResult('s09_vision', await apiUpload<VisionResponse>('/api/workbench/s09/vision', fd));
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  const runGraph = async () => {
    setBusy(true); setErr(null);
    try {
      setStudioResult('s09_graph', await apiPost<GraphResponse>('/api/workbench/s09/graph', {
        link_method: linkMethod,
        edge_operator: edgeOp,
        run_supervised: true,
        run_neural_gae: true,
        top_k: 12,
      }));
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  return (
    <div>
      <div className="studio-hero">
        <div>
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <h1>09 · Specialized AI — Extended NLP, Vision Foundation Hub &amp; Graph Link Prediction</h1>
            <StudioHelpButton section="studio-09" onOpenHelp={openHelp} title="Open Chapter 09: Kaplan-Meier/CoxPH Survival, Graph Link Prediction, NLP & Vision Guide" />
          </div>
          <p>Zero-Shot, QA, Semantic Search, PII/Toxicity, CLIP/DINOv2/YOLO/Segmentation, 8-Method + Supervised + Neural VGAE Link Prediction.</p>
        </div>
        <ProvenancePill ws={ws} />
      </div>
      {err && <div className="alert alert-error">{err}</div>}
      <div className="tabs">
        <button className={`tab-btn ${tab === 'nlp' ? 'active' : ''}`} onClick={() => setTab('nlp')}>Extended NLP Suite</button>
        <button className={`tab-btn ${tab === 'vision' ? 'active' : ''}`} onClick={() => setTab('vision')}>Vision Foundation Hub</button>
        <button className={`tab-btn ${tab === 'graph' ? 'active' : ''}`} onClick={() => setTab('graph')}>Graph Link Prediction &amp; ML</button>
        <button className={`tab-btn ${tab === 'recsys' ? 'active' : ''}`} onClick={() => setTab('recsys')}>Recommendations</button>
        <button className={`tab-btn ${tab === 'survival' ? 'active' : ''}`} onClick={() => setTab('survival')}>Survival Analysis</button>
      </div>

      {tab === 'nlp' && (
        <div className="card">
          <div className="row" style={{ marginBottom: 10 }}>
            {(backends?.nlp || []).map((b) => (
              <span key={b.id} className={`badge ${b.available ? 'badge-green' : ''}`}>{b.id}: {b.available ? (b.version || 'ON') : 'optional'}</span>
            ))}
          </div>
          <div className="row" style={{ marginBottom: 10 }}>
            <label className="field">NLP Capability
              <select value={nlpTask} onChange={(e) => setNlpTask(e.target.value)}>
                <option value="overview">Full NLP Suite Overview</option>
                <option value="zero_shot">Zero-Shot Text Classification</option>
                <option value="qa">Extractive Question Answering</option>
                <option value="semantic_search">Dense Semantic Search</option>
                <option value="emotions">6-Class Emotions &amp; Aspect Sentiment</option>
                <option value="pii_toxicity">PII Redaction &amp; Toxicity Screening</option>
                <option value="topics">LDA &amp; NMF Topic Discovery</option>
                <option value="cluster">Document Clustering &amp; Near-Duplicates</option>
              </select>
            </label>
            <label className="field" style={{ flex: 1 }}>Candidate Labels / Query
              <input type="text" value={nlpTask === 'qa' || nlpTask === 'semantic_search' ? nlpQuery : nlpLabels} onChange={(e) => (nlpTask === 'qa' || nlpTask === 'semantic_search' ? setNlpQuery(e.target.value) : setNlpLabels(e.target.value))} />
            </label>
          </div>
          <label className="field" style={{ marginBottom: 10 }}>Input Documents (one per line)
            <textarea rows={4} value={nlpText} onChange={(e) => setNlpText(e.target.value)} />
          </label>
          <button className="btn btn-primary" disabled={busy} onClick={runNLP}>{busy ? <span className="spinner" /> : null} Run Advanced NLP</button>
          {nlpRes && (
            <div style={{ marginTop: 12 }}>
              {(() => {
                const zsRows = Array.isArray(nlpRes.zero_shot) ? (nlpRes.zero_shot as Array<Record<string, unknown>>) : [];
                const firstScores = zsRows[0]?.scores;
                const zsItems =
                  typeof firstScores === 'object' && firstScores !== null && !Array.isArray(firstScores)
                    ? Object.entries(firstScores as Record<string, unknown>).flatMap(([lbl, val]) =>
                        typeof val === 'number' && Number.isFinite(val) ? [{ label: lbl, value: val, color: '#38bdf8' }] : [],
                      )
                    : [];
                return zsItems.length > 0 ? (
                  <div style={{ marginBottom: 12 }}>
                    <BarChartSVG title="Zero-Shot Topic / Intent Confidence (Doc #1)" horizontal items={zsItems} />
                  </div>
                ) : null;
              })()}
              <pre className="code-block" style={{ maxHeight: 380 }}>{JSON.stringify(nlpRes, null, 2)}</pre>
            </div>
          )}
        </div>
      )}

      {tab === 'vision' && (
        <div className="card">
          <div className="row" style={{ marginBottom: 10 }}>
            <span className="badge badge-accent">Device: {backends?.vision?.device || 'cpu'}</span>
            {(backends?.vision?.backends || []).map((b) => (
              <span key={b.id} className={`badge ${b.available ? 'badge-green' : ''}`}>{b.id}</span>
            ))}
          </div>
          <div className="row" style={{ marginBottom: 12 }}>
            <label className="field" style={{ flex: 1 }}>Zero-Shot Candidate Labels
              <input type="text" value={visLabels} onChange={(e) => setVisLabels(e.target.value)} />
            </label>
            <input type="file" accept="image/*" onChange={(e) => e.target.files?.[0] && runVision(e.target.files[0])} />
            <button className="btn btn-primary" disabled={busy} onClick={() => runVision()}>Run on Benchmark Scene</button>
          </div>
          {visRes && (
            <div className="grid-2">
              <div>
                <h4>Zero-Shot Classification ({visRes.zero_shot?.backend})</h4>
                {visRes.zero_shot?.top_label === null && <p>No supported top label.</p>}
                {visRes.zero_shot?.note && <p>{visRes.zero_shot.note}</p>}
                <SimpleTable
                  title={visRes.zero_shot?.score_kind === 'cosine_similarity' ? 'Cosine similarity' : 'Uncalibrated visual scores'}
                  rows={(visRes.zero_shot?.predictions || []).map((p) => ({ label: p.label, score: p.score ?? p.probability }))}
                />
              </div>
              <div>
                <h4>Detections ({visRes.detections?.n_detections}) &amp; Regions ({visRes.segmentation?.n_segments})</h4>
                <SimpleTable rows={visRes.segmentation?.regions || []} />
                <div style={{ marginTop: 10 }}><SimpleTable rows={visRes.detections?.detections || []} /></div>
              </div>
            </div>
          )}
        </div>
      )}

      {tab === 'graph' && (
        <div className="card">
          <div className="row" style={{ marginBottom: 14 }}>
            <label className="field">Topological Link Scorer
              <select value={linkMethod} onChange={(e) => setLinkMethod(e.target.value)}>
                <option value="ensemble_rrf">RRF Multi-Method Ensemble</option>
                <option value="adamic_adar">Adamic-Adar Index</option>
                <option value="resource_allocation">Resource Allocation</option>
                <option value="jaccard">Jaccard Coefficient</option>
                <option value="katz">Katz Path Index</option>
                <option value="personalized_pagerank">Personalized PageRank (PPR)</option>
                <option value="lhn">Leicht-Holme-Newman (LHN)</option>
                <option value="preferential_attachment">Preferential Attachment</option>
              </select>
            </label>
            <label className="field">Supervised Edge Operator
              <select value={edgeOp} onChange={(e) => setEdgeOp(e.target.value)}>
                <option value="hadamard">Hadamard (u ⊙ v)</option>
                <option value="l1">Weighted L1 (|u - v|)</option>
                <option value="l2">Weighted L2 ((u - v)²)</option>
                <option value="average">Average ((u + v)/2)</option>
              </select>
            </label>
            <button className="btn btn-primary" disabled={busy} onClick={runGraph}>
              {busy ? <span className="spinner" /> : null} Run Full Link Prediction Suite
            </button>
          </div>

          {graphRes && (
            <>
              <div className="kpi-grid">
                <div className="kpi-card"><div className="kpi-label">Nodes / Edges</div><div className="kpi-value">{graphRes.structure?.n_nodes} / {graphRes.structure?.n_edges}</div></div>
                <div className="kpi-card"><div className="kpi-label">Supervised Link ROC-AUC</div><div className="kpi-value">{Number(graphRes.supervised_link_prediction?.metrics?.roc_auc ?? 0).toFixed(4)}</div></div>
                <div className="kpi-card"><div className="kpi-label">Neural VGAE ROC-AUC</div><div className="kpi-value">{Number(graphRes.neural_gae_link_prediction?.metrics?.roc_auc ?? 0).toFixed(4)}</div></div>
                <div className="kpi-card"><div className="kpi-label">Max K-Core / Bridges</div><div className="kpi-value">{graphRes.structure?.max_k_core} / {graphRes.structure?.n_bridges}</div></div>
              </div>
              <NetworkGraphSVG
                nodes={graphRes.visualization?.nodes || []}
                edges={graphRes.visualization?.edges || []}
                predictedEdges={(graphRes.topological_links || []).map((e) => ({ source: String(e.source), target: String(e.target) }))}
              />
              <div className="grid-2" style={{ marginTop: 14 }}>
                <div>
                  <h4>Topological Predicted Links ({linkMethod})</h4>
                  <SimpleTable rows={graphRes.topological_links || []} />
                </div>
                <div>
                  <h4>Neural VGAE Predicted Links</h4>
                  <SimpleTable rows={graphRes.neural_gae_link_prediction?.top_predictions || []} />
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {tab === 'recsys' && (
        <div className="card">
          <button className="btn btn-primary" disabled={busy} onClick={async () => {
            setBusy(true); setErr(null);
            try { setStudioResult('s09_recommend', await apiPost<RecommendResponse>('/api/workbench/s09/recommend', {})); } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
          }}>Run Hybrid Recommender</button>
          {recRes && (
            <div style={{ marginTop: 12 }}>
              {(recRes.recommendations?.length ?? 0) > 0 && (
                <div style={{ marginBottom: 12 }}>
                  <BarChartSVG
                    title="Hybrid Recommendation Affinity Scores"
                    horizontal
                    items={(recRes.recommendations || []).slice(0, 12).flatMap((r, idx) => {
                      const sc = Number(r.score ?? r.affinity ?? NaN);
                      const lbl = String(r.item ?? r.item_id ?? r.title ?? `Item ${idx + 1}`);
                      return Number.isFinite(sc) ? [{ label: lbl, value: sc, color: '#34d399' }] : [];
                    })}
                  />
                </div>
              )}
              <SimpleTable rows={recRes.recommendations || []} />
            </div>
          )}
        </div>
      )}

      {tab === 'survival' && (
        <div className="card">
          <div className="row" style={{ marginBottom: 12 }}>
            <label className="field">Duration Column
              <select value={durCol} onChange={(e) => setDurCol(e.target.value)}>{(ws.numeric_columns.length ? ws.numeric_columns : ws.columns).map((c) => <option key={c} value={c}>{c}</option>)}</select>
            </label>
            <label className="field">Event Column
              <select value={evtCol} onChange={(e) => setEvtCol(e.target.value)}>{(ws.numeric_columns.length ? ws.numeric_columns : ws.columns).map((c) => <option key={c} value={c}>{c}</option>)}</select>
            </label>
            <button className="btn btn-primary" disabled={busy || !ws.has_dataset} onClick={async () => {
              setBusy(true); setErr(null);
              const cols = ws.numeric_columns.length ? ws.numeric_columns : ws.columns;
              try { setStudioResult('s09_survival', await apiPost<SurvivalResponse>('/api/workbench/s09/survival', { duration_col: durCol || cols[0], event_col: evtCol || cols[1] || cols[0] })); } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
            }}>Fit Kaplan-Meier &amp; CoxPH</button>
          </div>
          {survRes && (
            <>
              <div className="grid-2" style={{ marginBottom: 12 }}>
                {(survRes.km_curve?.length ?? 0) > 0 && (
                  <LineSeriesSVG
                    title="Kaplan-Meier Survival Probability Curve S(t)"
                    subtitle={survRes.median_survival !== null && survRes.median_survival !== undefined ? `Median Survival = ${Number(survRes.median_survival).toFixed(3)}` : 'Stepwise survival function with 95% Greenwood CI'}
                    xLabel="Duration (t)"
                    yLabel="Survival Probability S(t)"
                    series={[
                      {
                        name: 'Kaplan-Meier S(t)',
                        color: '#38bdf8',
                        stepAfter: true,
                        points: (survRes.km_curve || []).map((pt) => ({
                          x: Number(pt.time ?? 0),
                          y: Number(pt.survival_prob ?? 0),
                          lower: typeof pt.ci_lower === 'number' ? pt.ci_lower : null,
                          upper: typeof pt.ci_upper === 'number' ? pt.ci_upper : null,
                        })),
                      },
                    ]}
                  />
                )}
                {survRes.cox && typeof survRes.cox.hazard_ratios === 'object' && survRes.cox.hazard_ratios !== null && (
                  <BarChartSVG
                    title="Cox Proportional Hazards — Covariate Hazard Ratios exp(β)"
                    horizontal
                    items={Object.entries(survRes.cox.hazard_ratios as Record<string, unknown>).flatMap(([feat, hr]) =>
                      typeof hr === 'number' && Number.isFinite(hr)
                        ? [{ label: feat, value: hr, color: hr > 1 ? '#f87171' : '#34d399' }]
                        : [],
                    )}
                  />
                )}
              </div>
              <pre className="code-block">{JSON.stringify({ median_survival: survRes.median_survival, cox: survRes.cox, points: survRes.km_curve?.length }, null, 2)}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Studio 10 — Visual DAG Pipeline Builder
// ============================================================================
export const Studio10DAG: React.FC<StudioProps> = ({ ws, onSync, studioCache, setStudioResult, openHelp }) => {
  const [stages, setStages] = useState<string[]>(['quality', 'clean', 'eda', 'features', 'automl']);
  const dagRes = studioCache.s10_dag ?? null;
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const toggle = (s: string) => setStages((p) => (p.includes(s) ? p.filter((x) => x !== s) : [...p, s]));

  const runDag = async () => {
    setBusy(true); setErr(null);
    try {
      const res = await apiPost<DAGResponse>('/api/workbench/s10/run', { stages });
      onSync(res.state);
      setStudioResult('s10_dag', res);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  return (
    <div>
      <div className="studio-hero">
        <div>
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <h1>10 · Visual DAG Pipeline Builder</h1>
            <StudioHelpButton section="studio-10" onOpenHelp={openHelp} title="Open Chapter 10: Visual DAG Pipeline Builder Reference" />
          </div>
          <p>Compose reproducible end-to-end directed acyclic analytics pipelines with live topological SVG rendering &amp; execution telemetry.</p>
        </div>
        <div className="row">
          <ProvenancePill ws={ws} />
          <button className="btn btn-primary" disabled={busy || !ws.has_dataset || !stages.length} onClick={runDag}>
            {busy ? <span className="spinner" /> : null} Execute Pipeline DAG
          </button>
        </div>
      </div>
      {err && <div className="alert alert-error">{err}</div>}
      <div className="card">
        <div className="chip-grid" style={{ marginBottom: 14 }}>
          {['quality', 'clean', 'eda', 'features', 'automl'].map((s) => (
            <button key={s} className={`chip ${stages.includes(s) ? 'selected' : ''}`} onClick={() => toggle(s)}>{s.toUpperCase()}</button>
          ))}
        </div>
        <InspectableBox title="Topological Pipeline DAG" subtitle={`${stages.length} ordered execution stage(s)`} compact>
          <svg viewBox="0 0 760 110" style={{ width: '100%', height: 110, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
            {stages.map((s, i) => {
              const x = 35 + i * 145;
              return (
                <g key={s}>
                  {i > 0 && <line x1={x - 30} y1={55} x2={x} y2={55} stroke="#38bdf8" strokeWidth={2} />}
                  <rect x={x} y={32} width={115} height={46} rx={10} fill="var(--bg-card)" stroke="#38bdf8" strokeWidth={1.6} />
                  <text x={x + 57} y={59} textAnchor="middle" fill="var(--text)" fontSize={12} fontWeight={700}>{s.toUpperCase()}</text>
                </g>
              );
            })}
          </svg>
        </InspectableBox>
        {dagRes?.execution && (
          <div style={{ marginTop: 12 }}>
            {Array.isArray(dagRes.execution.step_records) && dagRes.execution.step_records.length > 0 && (
              <div style={{ marginBottom: 12 }}>
                <BarChartSVG
                  title="DAG Stage Execution Telemetry (Duration in Seconds)"
                  horizontal
                  items={(dagRes.execution.step_records as Array<Record<string, unknown>>).map((sr) => ({
                    label: String(sr.step_id ?? 'stage'),
                    value: Number(sr.duration_seconds ?? 0),
                    color: sr.status === 'success' ? '#34d399' : '#f87171',
                  }))}
                />
              </div>
            )}
            <pre className="code-block" style={{ maxHeight: 320 }}>{JSON.stringify(dagRes.execution, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// Studio 11 — MLOps (Registry, Experiments & Drift)
// ============================================================================
export const Studio11MLOps: React.FC<StudioProps> = ({ ws, studioCache, setStudioResult, openHelp }) => {
  const [overview, setOverview] = useState<MLOpsOverviewResponse>({ registered_models: [], experiment_runs: [] });
  const [modelName, setModelName] = useState('omniforge_champion');
  const driftRes = studioCache.s11_drift?.drift_report ?? null;
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const refresh = () => apiGet<MLOpsOverviewResponse>('/api/workbench/s11/overview').then(setOverview).catch(() => {});
  useEffect(() => { refresh(); }, []);

  return (
    <div>
      <div className="studio-hero">
        <div>
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <h1>11 · MLOps — Model Registry, Experiments &amp; Drift Monitor</h1>
            <StudioHelpButton section="studio-11" onOpenHelp={openHelp} title="Open Chapter 11: PSI & KS Drift Formulas, Thresholds & Model Registry Governance" />
          </div>
          <p>Register champion models, transition lifecycle stages (Staging → Production) &amp; run PSI / KS / Wasserstein drift monitoring.</p>
        </div>
        <ProvenancePill ws={ws} />
      </div>
      {err && <div className="alert alert-error">{err}</div>}
      <div className="grid-2">
        <div className="card">
          <div className="card-title">Model Registry</div>
          <div className="row" style={{ marginBottom: 12 }}>
            <input type="text" value={modelName} onChange={(e) => setModelName(e.target.value)} />
            <button className="btn btn-primary" disabled={busy || !ws.active_model_name} onClick={async () => {
              setBusy(true); setErr(null);
              try { const r = await apiPost<RegisterResponse>('/api/workbench/s11/register', { model_name: modelName, stage: 'Staging' }); setOverview((o) => ({ ...o, registered_models: r.registered_models ?? o.registered_models })); setStudioResult('s11_registered', r); } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
            }}>Register Active Model</button>
          </div>
          <SimpleTable rows={overview.registered_models || []} />
        </div>
        <div className="card">
          <div className="card-title">Dataset Drift Monitor (PSI / KS-Test)</div>
          <button className="btn btn-primary" disabled={busy || !ws.has_dataset} onClick={async () => {
            setBusy(true); setErr(null);
            try { const r = await apiPost<DriftResponse>('/api/workbench/s11/drift', { split_ratio: 0.5 }); setStudioResult('s11_drift', r); } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
          }}>Run Drift Detection</button>
          {driftRes && (
            <div style={{ marginTop: 12 }}>
              {Array.isArray(driftRes.features) && driftRes.features.length > 0 && (
                <div style={{ marginBottom: 12 }}>
                  <BarChartSVG
                    title="Feature Population Stability Index (PSI) Drift Profile"
                    horizontal
                    items={(driftRes.features as Array<Record<string, unknown>>).slice(0, 14).flatMap((f) => {
                      const col = String(f.column ?? '');
                      const psi = Number(f.psi ?? f.ks_statistic ?? NaN);
                      const status = String(f.status || '').toLowerCase();
                      return col && Number.isFinite(psi)
                        ? [{ label: col, value: psi, color: status === 'critical' ? '#f87171' : status === 'warning' ? '#fbbf24' : '#34d399' }]
                        : [];
                    })}
                  />
                </div>
              )}
              <pre className="code-block" style={{ maxHeight: 300 }}>{JSON.stringify(driftRes, null, 2)}</pre>
            </div>
          )}
        </div>
      </div>
      <FeatureFreshnessPanel />
    </div>
  );
};

// ============================================================================
// Studio 12 — Complete Customizable Executive Reporting & Export Center
// ============================================================================
export const Studio12Reporting: React.FC<StudioProps> = ({ ws, onSync, studioCache, setStudioResult, openHelp }) => {
  const [title, setTitle] = useState('OmniForge Executive Analytics & Supervisory Report');
  const [author, setAuthor] = useState('OmniForge Enterprise Analytics');
  const [catalog, setCatalog] = useState<ReportSection[]>([]);
  const [presets, setPresets] = useState<Record<string, string[]>>({});
  const [selectedSections, setSelectedSections] = useState<string[]>([]);
  const [includeCharts, setIncludeCharts] = useState(true);
  const [includeTables, setIncludeTables] = useState(true);
  const [includeAppendix, setIncludeAppendix] = useState(true);
  const [maxRows, setMaxRows] = useState(100);
  const compiled = studioCache.s12_report ?? null;
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    apiGet<ReportCatalogResponse>('/api/workbench/s12/catalog')
      .then((res) => {
        const secs = res.sections || [];
        setCatalog(secs);
        setPresets(res.presets || {});
        setSelectedSections((prev) => (prev.length ? prev : secs.filter((s) => s.default_on !== false).map((s) => s.key ?? '')));
      })
      .catch(() => {});
  }, [ws.n_rows, ws.active_model_name, ws.supervisory_status]);

  const toggleSection = (key: string) => {
    setSelectedSections((prev) => (prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]));
  };

  const applyPreset = (keys: string[]) => {
    setSelectedSections(keys);
  };

  const compile = async () => {
    setBusy(true); setErr(null);
    try {
      const res = await apiPost<CompiledReportResponse>('/api/workbench/s12/compile', {
        title,
        author,
        selected_sections: selectedSections.length ? selectedSections : null,
        include_charts: includeCharts,
        include_tables: includeTables,
        include_dataset_appendix: includeAppendix,
        max_table_rows: maxRows,
      });
      onSync(res.state);
      setStudioResult('s12_report', res);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  const downloadReport = async (fmt: string, ext: string) => {
    setErr(null);
    try {
      await apiDownload(`/api/workbench/s12/export/${fmt}`, `omniforge_report.${ext}`);
    } catch (e: unknown) {
      setErr(toErrorMessage(e));
    }
  };

  const primaryFormats = [
    ['pdf', 'Export PDF (.pdf)'],
    ['xlsx', 'Export Excel (.xlsx)'],
    ['docx', 'Export Word (.docx)'],
    ['csv', 'Export CSV (.csv)'],
  ];

  const extraFormats = [
    ['csv_bundle', 'CSV Multi-Table Bundle (.zip)'],
    ['pptx', 'PowerPoint Deck (.pptx)'],
    ['html', 'Interactive HTML (.html)'],
    ['md', 'Markdown (.md)'],
    ['json', 'Structured JSON (.json)'],
  ];

  // Group catalog sections by category
  const byCategory: Record<string, ReportSection[]> = {};
  catalog.forEach((sec) => {
    const cat = sec.category ?? 'Core';
    (byCategory[cat] = byCategory[cat] || []).push(sec);
  });

  return (
    <div>
      <div className="studio-hero">
        <div>
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <h1>12 · Complete Customizable Reporting &amp; Export Center</h1>
            <StudioHelpButton section="studio-12" onOpenHelp={openHelp} title="Open Chapter 12: 14-Section Executive Reporting & 8-Format Export Reference" />
          </div>
          <p>Select exact report sections, statistical tests, equations, charts, tables &amp; supervisory alarm audits — preview interactively (with Zoom &amp; Pop-Out) and export to PDF, XLSX, DOCX, CSV &amp; 5 more formats.</p>
        </div>
      </div>

      {err && <div className="alert alert-error">{err}</div>}

      <div className="card">
        <div className="card-title">
          <span>1. Report Metadata &amp; Section Customization ({selectedSections.length} / {catalog.length} sections selected)</span>
          <div className="row" style={{ gap: 6 }}>
            <button className="btn btn-sm" onClick={() => applyPreset(catalog.map((c) => c.key ?? ''))}>Select All</button>
            <button className="btn btn-sm" onClick={() => applyPreset(catalog.filter((c) => c.has_data).map((c) => c.key ?? ''))}>Auto (Ready Only)</button>
            {Object.entries(presets).slice(0, 3).map(([pName, pKeys]) => (
              <button key={pName} className="btn btn-sm" onClick={() => applyPreset(pKeys)}>{pName}</button>
            ))}
            <button className="btn btn-sm" onClick={() => applyPreset([])}>Clear</button>
          </div>
        </div>

        <div className="grid-3" style={{ marginBottom: 14 }}>
          <label className="field">Report Title
            <input type="text" value={title} onChange={(e) => setTitle(e.target.value)} />
          </label>
          <label className="field">Author / Team
            <input type="text" value={author} onChange={(e) => setAuthor(e.target.value)} />
          </label>
          <label className="field">Max Rows per Embedded Table
            <input type="number" min={10} max={500} value={maxRows} onChange={(e) => setMaxRows(Number(e.target.value))} />
          </label>
        </div>

        <div className="row" style={{ gap: 16, marginBottom: 14 }}>
          <label className="row" style={{ gap: 6 }}><input type="checkbox" checked={includeCharts} onChange={(e) => setIncludeCharts(e.target.checked)} /> Embed High-Resolution Charts &amp; Graphs</label>
          <label className="row" style={{ gap: 6 }}><input type="checkbox" checked={includeTables} onChange={(e) => setIncludeTables(e.target.checked)} /> Embed Statistical &amp; Analytical Tables</label>
          <label className="row" style={{ gap: 6 }}><input type="checkbox" checked={includeAppendix} onChange={(e) => setIncludeAppendix(e.target.checked)} /> Include Dataset Summary Appendix</label>
        </div>

        {/* Category-grouped Section Selection Grid */}
        <div className="grid-2" style={{ marginBottom: 14 }}>
          {Object.entries(byCategory).map(([cat, secs]) => (
            <div key={cat} style={{ padding: 12, borderRadius: 10, background: 'var(--bg)', border: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11.5, fontWeight: 800, textTransform: 'uppercase', color: 'var(--accent)', marginBottom: 8 }}>{cat}</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {secs.map((sec) => {
                  const sectionKey = sec.key ?? '';
                  const checked = selectedSections.includes(sectionKey);
                  return (
                    <label
                      key={sectionKey}
                      style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: 8,
                        padding: '6px 8px',
                        borderRadius: 7,
                        background: checked ? 'var(--accent-bg)' : 'transparent',
                        cursor: 'pointer',
                      }}
                    >
                      <input type="checkbox" checked={checked} onChange={() => toggleSection(sectionKey)} style={{ marginTop: 3 }} />
                      <div style={{ flex: 1 }}>
                        <div className="row" style={{ justifyContent: 'space-between', gap: 6 }}>
                          <strong style={{ fontSize: 12.5 }}>{sec.title}</strong>
                          <span className={`badge ${sec.has_data ? 'badge-green' : ''}`} style={{ fontSize: 10 }}>
                            {sec.has_data ? 'DATA READY' : 'OPTIONAL'}
                          </span>
                        </div>
                        <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>{sec.description}</div>
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        <div className="row" style={{ gap: 10, marginBottom: 12 }}>
          <button className="btn btn-primary" disabled={busy || !ws.has_dataset} onClick={compile}>
            {busy ? <span className="spinner" /> : null} Compile Custom Report ({selectedSections.length} Sections)
          </button>
          {primaryFormats.map(([fmt, label]) => (
            <button
              key={fmt}
              className="btn btn-success"
              disabled={!ws.has_dataset}
              onClick={() => void downloadReport(fmt, fmt)}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="row" style={{ gap: 6 }}>
          <span style={{ fontSize: 11.5, color: 'var(--text-dim)' }}>Additional Enterprise Formats:</span>
          {extraFormats.map(([fmt, label]) => (
            <button
              key={fmt}
              className="btn btn-sm"
              disabled={!ws.has_dataset}
              onClick={() => void downloadReport(fmt, fmt === 'csv_bundle' ? 'zip' : fmt)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {compiled && (
        <>
          <div className="card">
            <div className="card-title">
              <span>{compiled.title}</span>
              <span className="badge badge-green">
                {(compiled.sections ?? []).length} sections · {compiled.table_names?.length || 0} tables · {compiled.n_charts || 0} charts
              </span>
            </div>
            <p style={{ color: 'var(--text-muted)', marginTop: 0 }}>{compiled.executive_summary}</p>
            {(compiled.sections || []).map((s, idx) => (
              <div key={`${s.id}-${idx}`} style={{ padding: '10px 0', borderTop: '1px solid var(--border)' }}>
                <strong>{s.title}</strong>
                <p style={{ margin: '4px 0 0', color: 'var(--text-muted)', fontSize: 12.8 }}>{s.narrative}</p>
                {(s.highlights?.length ?? 0) > 0 && (
                  <ul style={{ margin: '6px 0 0', paddingLeft: 18, fontSize: 12, color: 'var(--text-dim)' }}>
                    {(s.highlights ?? []).map((h, i) => <li key={i}>{h}</li>)}
                  </ul>
                )}
              </div>
            ))}
          </div>

          {(compiled.chart_previews?.length ?? 0) > 0 && (
            <div className="card">
              <div className="card-title">Embedded Report Charts &amp; Visuals (Zoom / Pop-Out Supported)</div>
              <div className="report-chart-grid">
                {(compiled.chart_previews ?? []).map((ch, idx) => (
                  <InspectableBox key={idx} title={ch.title} subtitle={ch.caption || ch.section} compact>
                    <img src={ch.image_data_url ?? ''} alt={ch.title ?? 'Report chart'} className="report-chart-img" />
                  </InspectableBox>
                ))}
              </div>
            </div>
          )}

          {compiled.table_previews && Object.keys(compiled.table_previews).length > 0 && (
            <div className="card">
              <div className="card-title">Compiled Analytical Tables ({Object.keys(compiled.table_previews).length})</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                {Object.entries(compiled.table_previews ?? {}).map(([tname, tinfo]) => (
                  <SimpleTable key={tname} title={`${tname} (${tinfo.rows} total rows)`} rows={tinfo.preview || []} maxRows={25} />
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

// ============================================================================
// Studio 13 — Context-Aware AI Analytics Copilot
// ============================================================================
export const Studio13Copilot: React.FC<StudioProps> = ({ ws, studioCache, setStudioResult, openHelp }) => {
  const defaultMsg = {
    role: 'assistant' as const,
    text: 'Welcome to OmniForge AI Analytics Copilot. Ask about your active dataset, statistical diagnostics, AutoML strategy, or next analytical steps.',
  };
  const cachedHistory: ChatTurn[] | undefined = studioCache.s13_chat?.history;
  const suggestedActions: string[] = studioCache.s13_chat?.suggested_actions ?? [];
  const messages = cachedHistory && cachedHistory.length > 0 ? [defaultMsg, ...cachedHistory] : [defaultMsg];
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);

  const sendPrompt = async (rawPrompt: string) => {
    const q = rawPrompt.trim();
    if (!q || busy) return;
    const prevHist = cachedHistory || [];
    setStudioResult('s13_chat', { ...(studioCache.s13_chat || {}), history: [...prevHist, { role: 'user', text: q }] });
    setInput('');
    setBusy(true);
    try {
      const res = await apiPost<ChatResponse>('/api/workbench/s13/chat', { message: q });
      setStudioResult('s13_chat', res);
    } catch (e: unknown) {
      setStudioResult('s13_chat', {
        ...(studioCache.s13_chat || {}),
        history: [...prevHist, { role: 'user', text: q }, { role: 'assistant', text: `Error: ${toErrorMessage(e)}` }],
      });
    } finally { setBusy(false); }
  };

  return (
    <div>
      <div className="studio-hero">
        <div>
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <h1>13 · AI Analytics Copilot</h1>
            <StudioHelpButton section="studio-13" onOpenHelp={openHelp} title="Open Chapter 13: Context-Aware AI Analytics Copilot Guide" />
          </div>
          <p>Dataset-aware conversational assistant connected directly to your live workspace artifacts.</p>
        </div>
        <ProvenancePill ws={ws} />
      </div>
      <div className="card">
        <div className="chat-box">
          {messages.map((m, i) => <div key={i} className={`chat-msg ${m.role}`}>{m.text}</div>)}
        </div>
        {suggestedActions.length > 0 && (
          <div className="chip-grid" style={{ marginBottom: 10 }}>
            {suggestedActions.slice(0, 5).map((act, idx) => (
              <button key={idx} className="chip" disabled={busy} onClick={() => void sendPrompt(act)}>
                {act}
              </button>
            ))}
          </div>
        )}
        <div className="row">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && void sendPrompt(input)}
            placeholder="Ask OmniForge Copilot about your dataset, correlations, anomalies, or AutoML models..."
            style={{ flex: 1 }}
          />
          <button className="btn btn-primary" disabled={busy} onClick={() => void sendPrompt(input)}>{busy ? <span className="spinner" /> : 'Send'}</button>
        </div>
      </div>
    </div>
  );
};
