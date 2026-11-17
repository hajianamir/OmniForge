import React, { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, Database, FileSpreadsheet, Play, ShieldAlert, TrendingDown, TrendingUp, Wrench } from 'lucide-react';
import { apiDownload, apiGet, apiPost, apiUpload } from '../api';
import { BarChartSVG, DivergingDriverBarSVG, SPCControlSVG, SupervisoryTrendSVG } from '../components/Charts';
import { SimpleTable } from '../components/DataGrid';
import { StudioHelpButton } from '../components/HelpGuideModal';
import { MathEquationCard } from '../components/MathEquation';
import { ProvenancePill, type StudioProps } from './StudiosPart1';

/**
 * Studio 14 — Live Prediction, Actual-vs-Baseline Evaluation, Configurable Error Tolerance,
 * Multi-Source Connectors (XLSX, SQL, MS Access, Telemetry Sim, Single Probe) & Engineering Supervisory Dashboard.
 */
export const Studio14Supervisory: React.FC<StudioProps> = ({ ws, onSync, studioCache, setStudioResult, openHelp }) => {
  const [models, setModels] = useState<any[]>([]);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [selectedBaselineId, setSelectedBaselineId] = useState('');
  const [quickMethod, setQuickMethod] = useState('ridge');
  const [sourceTab, setSourceTab] = useState<'file' | 'sql' | 'access' | 'sim' | 'single'>('file');

  // Tolerance & Alarm configuration
  const [actualCol, setActualCol] = useState('');
  const [idCol, setIdCol] = useState('');
  const [maxPctErr, setMaxPctErr] = useState(8.0);
  const [maxAbsErr, setMaxAbsErr] = useState<string>('');
  const [maxSigmaZ, setMaxSigmaZ] = useState(2.5);
  const [ratioMin, setRatioMin] = useState(0.85);
  const [ratioMax, setRatioMax] = useState(1.15);
  const [warnFrac, setWarnFrac] = useState(0.75);
  const [maxBatchBreachPct, setMaxBatchBreachPct] = useState(5.0);
  const [minClfAccPct, setMinClfAccPct] = useState(85.0);

  // Source A: File / Multi-sheet Excel
  const [liveFile, setLiveFile] = useState<File | null>(null);
  const [fileSheets, setFileSheets] = useState<string[]>([]);
  const [selectedSheet, setSelectedSheet] = useState('');

  // Source B: SQL Database
  const [sqlUrl, setSqlUrl] = useState('');
  const [sqlTables, setSqlTables] = useState<any[]>([]);
  const [sqlTable, setSqlTable] = useState('');
  const [sqlQuery, setSqlQuery] = useState('');

  // Source C: MS Access
  const [accessPath, setAccessPath] = useState('');
  const [accessTables, setAccessTables] = useState<any[]>([]);
  const [accessTable, setAccessTable] = useState('');

  // Source D: Operational Telemetry Simulation
  const [simRows, setSimRows] = useState(80);
  const [simFaultPct, setSimFaultPct] = useState(12.0);
  const [simNoise, setSimNoise] = useState(0.04);

  // Source E: Single Live Operational Reading
  const [singleInputs, setSingleInputs] = useState<Record<string, string>>({});
  const [singleActual, setSingleActual] = useState('');

  // Results & History (hydrated immediately from studioCache + /s14/models)
  const [localEvalRes, setLocalEvalRes] = useState<any>(null);
  const evalRes = localEvalRes ?? studioCache.s14_supervisory ?? null;
  const setEvalRes = (val: any) => {
    setLocalEvalRes(val);
    setStudioResult('s14_supervisory', val);
  };
  const [history, setHistory] = useState<any[]>(studioCache.s14_supervisory?.history || []);
  const [alarmFilter, setAlarmFilter] = useState<'ALL' | 'CRITICAL' | 'WARNING'>('ALL');
  const [selectedAlarmIdx, setSelectedAlarmIdx] = useState<number>(0);
  const [previewChampEq, setPreviewChampEq] = useState<any>(null);
  const [previewBaseEq, setPreviewBaseEq] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const loadModels = async () => {
    try {
      const r = await apiGet('/api/workbench/s14/models');
      const list = r.models || [];
      setModels(list);
      if (list.length && !selectedModelId) {
        setSelectedModelId(list[0].model_id);
      }
      if (r.last_evaluation) setEvalRes(r.last_evaluation);
      if (r.history) setHistory(r.history);
    } catch {
      // Ignore if no session yet
    }
  };

  useEffect(() => {
    loadModels();
  }, [ws.active_model_name, ws.available_models_count, ws.dataset_name]);

  useEffect(() => {
    if (!models.length) return;
    const mid = selectedModelId || models[0]?.model_id || '';
    if (!mid) return;
    const q = new URLSearchParams();
    q.set('model_id', mid);
    if (selectedBaselineId) q.set('baseline_model_id', selectedBaselineId);
    apiGet(`/api/workbench/s14/equations?${q.toString()}`)
      .then((r) => {
        if (r?.champion_equation) setPreviewChampEq(r.champion_equation);
        if (r?.baseline_equation) setPreviewBaseEq(r.baseline_equation);
      })
      .catch(() => {});
  }, [selectedModelId, selectedBaselineId, models.length]);

  useEffect(() => {
    if (ws.primary_target && !actualCol) setActualCol(ws.primary_target);
  }, [ws.primary_target, actualCol]);

  const tolPayload = () => ({
    model_id: selectedModelId || models[0]?.model_id || 'active_champion',
    baseline_model_id: selectedBaselineId || null,
    actual_col: actualCol.trim() || null,
    id_col: idCol.trim() || null,
    max_pct_error: maxPctErr,
    max_abs_error: maxAbsErr.trim() ? Number(maxAbsErr) : null,
    max_sigma_z: maxSigmaZ,
    baseline_ratio_min: ratioMin,
    baseline_ratio_max: ratioMax,
    warning_fraction: warnFrac,
    max_batch_breach_rate_pct: maxBatchBreachPct,
    min_classification_accuracy_pct: minClfAccPct,
  });

  const handleQuickTrain = async () => {
    setBusy(true); setErr(null); setMsg(null);
    try {
      const res = await apiPost('/api/workbench/s14/quick-train', { method: quickMethod, epochs: 15 });
      setModels(res.models || []);
      const newest = (res.models || []).find((m: any) => m.label.includes(res.trained_model)) || (res.models || [])[0];
      if (newest) setSelectedModelId(newest.model_id);
      onSync(res.state);
      setMsg(`Trained '${res.trained_model}' (${res.metric}=${res.score}) and registered in Supervisory Monitor.`);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLiveFile(file);
    setFileSheets([]); setSelectedSheet('');
    const lower = file.name.toLowerCase();
    if (lower.endsWith('.xlsx') || lower.endsWith('.xls') || lower.endsWith('.xlsm')) {
      try {
        const fd = new FormData();
        fd.append('file', file);
        const info = await apiUpload('/api/workbench/s01/excel-sheets', fd);
        const sheets: string[] = info.sheets || [];
        setFileSheets(sheets);
        if (sheets.length) setSelectedSheet(sheets[0]);
      } catch {
        // Fallback single-sheet
      }
    }
    e.target.value = '';
  };

  const evaluateFile = async () => {
    if (!liveFile) return;
    setBusy(true); setErr(null); setMsg(null);
    try {
      const fd = new FormData();
      fd.append('file', liveFile);
      if (selectedSheet) fd.append('sheet_name', selectedSheet);
      const t = tolPayload();
      fd.append('model_id', String(t.model_id));
      if (t.baseline_model_id) fd.append('baseline_model_id', t.baseline_model_id);
      if (t.actual_col) fd.append('actual_col', t.actual_col);
      if (t.id_col) fd.append('id_col', t.id_col);
      fd.append('max_pct_error', String(t.max_pct_error));
      if (t.max_abs_error !== null) fd.append('max_abs_error', String(t.max_abs_error));
      fd.append('max_sigma_z', String(t.max_sigma_z));
      fd.append('baseline_ratio_min', String(t.baseline_ratio_min));
      fd.append('baseline_ratio_max', String(t.baseline_ratio_max));
      fd.append('warning_fraction', String(t.warning_fraction));
      fd.append('max_batch_breach_rate_pct', String(t.max_batch_breach_rate_pct));
      fd.append('min_classification_accuracy_pct', String(t.min_classification_accuracy_pct));
      const res = await apiUpload('/api/workbench/s14/evaluate-file', fd);
      onSync(res.state); setEvalRes(res); setHistory(res.history || []);
      setMsg(`Evaluated '${liveFile.name}' — Status: ${res.system_status} (${res.kpis?.breach_count ?? 0} tolerance breaches).`);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  const connectDemoSQL = async () => {
    setBusy(true); setErr(null);
    try {
      const info = await apiPost('/api/workbench/s01/sql/demo', {});
      setSqlUrl(info.connection_url);
      setSqlTables(info.tables || []);
      const liveTbl = (info.tables || []).find((t: any) => t.name.includes('live')) || (info.tables || [])[0];
      if (liveTbl) setSqlTable(liveTbl.name);
      setMsg('Connected to Demo Enterprise SQLite DB — ready to score live_turbine_monitoring_batch!');
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  const inspectSQL = async () => {
    if (!sqlUrl.trim()) return;
    setBusy(true); setErr(null);
    try {
      const info = await apiPost('/api/workbench/s01/sql/inspect', { connection_url: sqlUrl });
      setSqlTables(info.tables || []);
      if (info.tables?.length) setSqlTable(info.tables[0].name);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  const evaluateSQL = async () => {
    setBusy(true); setErr(null); setMsg(null);
    try {
      const res = await apiPost('/api/workbench/s14/evaluate-sql', {
        ...tolPayload(),
        connection_url: sqlUrl,
        table_name: sqlQuery.trim() ? null : (sqlTable || null),
        query: sqlQuery.trim() || null,
      });
      onSync(res.state); setEvalRes(res); setHistory(res.history || []);
      setMsg(`SQL Live Evaluation complete — Status: ${res.system_status} (${res.kpis?.breach_count ?? 0} breaches).`);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  const inspectAccess = async () => {
    if (!accessPath.trim()) return;
    setBusy(true); setErr(null);
    try {
      const info = await apiPost('/api/workbench/s01/access/inspect', { file_path: accessPath });
      setAccessTables(info.tables || []);
      if (info.tables?.length) setAccessTable(info.tables[0].name);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  const evaluateAccess = async () => {
    setBusy(true); setErr(null); setMsg(null);
    try {
      const res = await apiPost('/api/workbench/s14/evaluate-access', {
        ...tolPayload(),
        file_path: accessPath,
        table_name: accessTable,
      });
      onSync(res.state); setEvalRes(res); setHistory(res.history || []);
      setMsg(`MS Access Live Evaluation complete — Status: ${res.system_status}.`);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  const evaluateDemoBatch = async () => {
    setBusy(true); setErr(null); setMsg(null);
    try {
      const res = await apiPost('/api/workbench/s14/evaluate-demo-batch', {
        ...tolPayload(),
        n_rows: simRows,
        inject_fault_pct: simFaultPct,
        noise_scale: simNoise,
      });
      onSync(res.state); setEvalRes(res); setHistory(res.history || []);
      setMsg(`Supervisory evaluation complete — Status: ${res.system_status} (${res.kpis?.breach_count ?? 0} breaches, ${res.kpis?.warning_count ?? 0} warnings).`);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  const evaluateSingle = async () => {
    setBusy(true); setErr(null); setMsg(null);
    try {
      const activeModelObj = models.find((m) => m.model_id === selectedModelId) || models[0];
      const fCols: string[] = activeModelObj?.feature_cols?.length ? activeModelObj.feature_cols : ws.numeric_columns.slice(0, 8);
      const rec: Record<string, any> = {};
      fCols.forEach((c) => {
        const rawV = singleInputs[c];
        rec[c] = rawV !== undefined && rawV !== '' ? (Number.isNaN(Number(rawV)) ? rawV : Number(rawV)) : 0;
      });
      const res = await apiPost('/api/workbench/s14/evaluate-single', {
        ...tolPayload(),
        record: rec,
        actual_value: singleActual.trim() ? (Number.isNaN(Number(singleActual)) ? singleActual : Number(singleActual)) : null,
      });
      onSync(res.state); setEvalRes(res); setHistory(res.history || []);
      setMsg(`Single reading evaluated — Status: ${res.system_status}.`);
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  const activeModelMeta = models.find((m) => m.model_id === selectedModelId) || models[0];
  const singleFeatCols: string[] = activeModelMeta?.feature_cols?.length ? activeModelMeta.feature_cols.slice(0, 12) : ws.numeric_columns.slice(0, 8);

  const statusKey =
    evalRes?.system_status === 'CRITICAL_ALARM'
      ? 'critical'
      : evalRes?.system_status === 'WARNING'
        ? 'warning'
        : 'normal';

  const filteredAlarms = (evalRes?.alarms || []).filter((a: any) =>
    alarmFilter === 'ALL' ? true : String(a.severity || '').toUpperCase().includes(alarmFilter),
  );
  const activeChampEq = evalRes?.champion_equation || previewChampEq;
  const activeBaseEq = evalRes?.baseline_equation || previewBaseEq;
  const rootCause = evalRes?.root_cause_analysis || null;
  const gainVsBase = evalRes?.kpis?.model_vs_baseline_lift_pct ?? evalRes?.kpis?.mae_improvement_vs_baseline_pct;
  const alarmDisplayRows = filteredAlarms.map((a: any) => ({
    row_id: a.row_id,
    severity: a.severity,
    actual: a.actual,
    predicted: a.predicted,
    baseline_predicted: a.baseline_predicted,
    delta_vs_baseline: a.delta_vs_baseline,
    deviation_direction: a.deviation_direction,
    pct_error: a.pct_error,
    top_increase_drivers: a.top_increase_drivers,
    top_decrease_drivers: a.top_decrease_drivers,
    recommended_fix: a.recommended_fix,
    reason: a.reason,
  }));
  const inspectedAlarm = filteredAlarms[Math.min(selectedAlarmIdx, Math.max(0, filteredAlarms.length - 1))] || null;

  return (
    <div>
      <div className="studio-hero">
        <div>
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <h1>14 · Live Prediction, Reality-vs-Baseline &amp; Supervisory Control-Room Monitor</h1>
            <StudioHelpButton section="studio-14" onOpenHelp={openHelp} title="Open Chapter 14: SPC Z-Scores, Error Tolerance Bands, Alarm Matrix & Live Connectors" />
          </div>
          <p>Select any fitted Baseline, Non-Parametric Analytical Model, AutoML Champion, or Neural Network, connect real operational data from Excel (.xlsx), SQL Databases, or MS Access, enforce permissible error tolerances, trigger automated alerts, and monitor real-time engineering KPIs.</p>
        </div>
        <div className="row">
          <ProvenancePill ws={ws} />
          {evalRes && (
            <>
              <button className="btn btn-sm" onClick={() => apiDownload('/api/workbench/s14/export/csv', 'omniforge_supervisory_evaluation.csv')}>Export Scored CSV</button>
              <button className="btn btn-sm" onClick={() => apiDownload('/api/workbench/s14/export/xlsx', 'omniforge_supervisory_evaluation.xlsx')}>Export Scored XLSX</button>
            </>
          )}
        </div>
      </div>

      {msg && <div className="alert alert-success">{msg}</div>}
      {err && <div className="alert alert-error">{err}</div>}

      {/* 1. MODEL & BASELINE SELECTOR + INLINE QUICK-TRAIN */}
      <div className="card">
        <div className="card-title">
          <span>1. Active Prediction Model &amp; Reference Baseline Selection ({models.length} fitted models ready)</span>
          <span className="badge badge-accent">Supports Baselines · Non-Parametric · AutoML · Neural Networks</span>
        </div>

        <div className="grid-2" style={{ marginBottom: 12 }}>
          <label className="field">Primary Evaluation Model (Baseline / AutoML / Deep Learning)
            <select value={selectedModelId} onChange={(e) => setSelectedModelId(e.target.value)}>
              {models.length === 0 && <option value="">No models trained yet — use Quick-Train below or run Studio 03 / 06</option>}
              {models.map((m) => (
                <option key={m.model_id} value={m.model_id}>
                  [{m.family}] {m.label} ({m.n_features} features)
                </option>
              ))}
            </select>
          </label>
          <label className="field">Comparison Reference Baseline (Auto-selected if blank)
            <select value={selectedBaselineId} onChange={(e) => setSelectedBaselineId(e.target.value)}>
              <option value="">Auto-Select Best Fitted Baseline</option>
              {models.map((m) => (
                <option key={m.model_id} value={m.model_id}>[{m.family}] {m.label}</option>
              ))}
            </select>
          </label>
        </div>

        <div className="row" style={{ padding: 10, background: 'var(--bg)', borderRadius: 8, border: '1px solid var(--border)' }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-muted)' }}>Quick-Train Any Model Family Directly:</span>
          <select value={quickMethod} onChange={(e) => setQuickMethod(e.target.value)}>
            <optgroup label="Validated Baselines">
              <option value="ridge">Linear Baseline (Ridge / Logistic)</option>
              <option value="decision_stump">Decision Tree Baseline</option>
            </optgroup>
            <optgroup label="Non-Parametric Analytical Models">
              <option value="gam_spline">Penalized B-Spline GAM (Non-Parametric)</option>
              <option value="nadaraya_watson">Nadaraya-Watson Kernel Estimator</option>
            </optgroup>
            <optgroup label="Tree Ensembles (AutoML)">
              <option value="random_forest">Random Forest Ensemble</option>
              <option value="gradient_boosting">Gradient Boosting Champion</option>
            </optgroup>
            <optgroup label="Neural Networks / Deep Learning">
              <option value="tabnet">TabNet Neural Network</option>
              <option value="resnet_tabular">ResNet-Tabular Neural Network</option>
              <option value="ft_transformer">FT-Transformer Neural Network</option>
              <option value="mlp">Residual MLP Neural Network</option>
            </optgroup>
          </select>
          <button className="btn btn-primary btn-sm" disabled={busy || !ws.has_dataset} onClick={handleQuickTrain}>
            {busy ? <span className="spinner" /> : null} Quick-Train &amp; Select Model
          </button>
        </div>
      </div>

      {/* 1B. EXACT GOVERNING MATHEMATICAL EQUATIONS USED BY MONITORING (CHAMPION & BASELINE) */}
      {(activeChampEq || activeBaseEq) && (
        <div className="card" style={{ borderLeft: '3px solid var(--accent)' }}>
          <div className="card-title">
            <span>Active Governing Mathematical Relationships (Monitoring Champion vs. Reference Baseline)</span>
            <span className="badge badge-accent">Portable External Equations (Python · Excel · SQL · MATLAB)</span>
          </div>
          <p style={{ margin: '0 0 12px', fontSize: 12.5, color: 'var(--text-muted)' }}>
            Exact analytical equations currently driving live predictions, tolerance bands, and Baseline deviation comparisons. Click <strong>Use Outside OmniForge</strong> or <strong>.py</strong> to export directly into external engineering, Excel, SQL, or SCADA/PLC environments.
          </p>
          {activeChampEq && (
            <div style={{ marginBottom: activeBaseEq ? 12 : 0 }}>
              <MathEquationCard
                defaultOpenCode
                eq={{
                  ...activeChampEq,
                  acceptance_status: 'ACTIVE MONITORING CHAMPION',
                }}
              />
            </div>
          )}
          {activeBaseEq && (
            <div>
              <MathEquationCard
                eq={{
                  ...activeBaseEq,
                  acceptance_status: 'REFERENCE BASELINE EQUATION',
                }}
              />
            </div>
          )}
        </div>
      )}

      {/* 2. CONFIGURABLE PERMISSIBLE ERROR TOLERANCE & ALARM RULES */}
      <div className="card">
        <div className="card-title">
          <span>2. Permissible Error Tolerance Limits &amp; Automated Alarm Rules</span>
          <span className="badge badge-amber">Engineering Tolerance Gate</span>
        </div>
        <div className="grid-4" style={{ marginBottom: 10 }}>
          <label className="field">Max Permissible % Error
            <input type="number" step={0.5} min={0.1} max={500} value={maxPctErr} onChange={(e) => setMaxPctErr(Number(e.target.value))} />
          </label>
          <label className="field">Max Absolute Error (optional)
            <input type="number" step={0.5} placeholder="Auto / Disabled" value={maxAbsErr} onChange={(e) => setMaxAbsErr(e.target.value)} />
          </label>
          <label className="field">SPC Control Limit (±σ Z-Score)
            <input type="number" step={0.1} min={1.0} max={6.0} value={maxSigmaZ} onChange={(e) => setMaxSigmaZ(Number(e.target.value))} />
          </label>
          <label className="field">Early Warning Fraction (of limit)
            <input type="number" step={0.05} min={0.3} max={0.99} value={warnFrac} onChange={(e) => setWarnFrac(Number(e.target.value))} />
          </label>
        </div>
        <div className="grid-4">
          <label className="field">Actual / Baseline Min Ratio
            <input type="number" step={0.05} min={0.1} max={1.5} value={ratioMin} onChange={(e) => setRatioMin(Number(e.target.value))} />
          </label>
          <label className="field">Actual / Baseline Max Ratio
            <input type="number" step={0.05} min={0.5} max={5.0} value={ratioMax} onChange={(e) => setRatioMax(Number(e.target.value))} />
          </label>
          <label className="field">Batch Breach Rate Alarm (%)
            <input type="number" step={0.5} min={0} max={100} value={maxBatchBreachPct} onChange={(e) => setMaxBatchBreachPct(Number(e.target.value))} />
          </label>
          <label className="field">Actual Ground-Truth Column
            <input type="text" placeholder={ws.primary_target || 'auto-detect target col'} value={actualCol} onChange={(e) => setActualCol(e.target.value)} />
          </label>
        </div>
        <div className="grid-2" style={{ marginTop: 10 }}>
          <label className="field">Record / Asset ID Column (optional)
            <input type="text" placeholder="e.g., asset_id, turbine_id, timestamp" value={idCol} onChange={(e) => setIdCol(e.target.value)} />
          </label>
          <label className="field">Min Classification Accuracy Alarm (%)
            <input type="number" step={1} min={10} max={100} value={minClfAccPct} onChange={(e) => setMinClfAccPct(Number(e.target.value))} />
          </label>
        </div>
      </div>

      {/* 3. LIVE OPERATIONAL DATA SOURCE CONNECTORS */}
      <div className="card">
        <div className="card-title">3. Connect Live / Real-World Operational Data</div>
        <div className="tabs" style={{ marginBottom: 12 }}>
          <button className={`tab-btn ${sourceTab === 'file' ? 'active' : ''}`} onClick={() => setSourceTab('file')}><FileSpreadsheet size={13} style={{ verticalAlign: -2 }} /> Excel (.xlsx) / CSV</button>
          <button className={`tab-btn ${sourceTab === 'sql' ? 'active' : ''}`} onClick={() => setSourceTab('sql')}><Database size={13} style={{ verticalAlign: -2 }} /> SQL Database</button>
          <button className={`tab-btn ${sourceTab === 'access' ? 'active' : ''}`} onClick={() => setSourceTab('access')}>MS Access (.accdb / .mdb)</button>
          <button className={`tab-btn ${sourceTab === 'sim' ? 'active' : ''}`} onClick={() => setSourceTab('sim')}><Play size={13} style={{ verticalAlign: -2 }} /> Live Telemetry Simulator</button>
          <button className={`tab-btn ${sourceTab === 'single' ? 'active' : ''}`} onClick={() => setSourceTab('single')}>Single Reading Probe</button>
        </div>

        {sourceTab === 'file' && (
          <div>
            <div className="row" style={{ gap: 10, marginBottom: 10 }}>
              <label className="btn" style={{ cursor: 'pointer' }}>
                Select Live .xlsx / .csv / .parquet File
                <input type="file" accept=".xlsx,.xls,.xlsm,.csv,.tsv,.parquet,.json" onChange={handleFileChange} style={{ display: 'none' }} />
              </label>
              {liveFile && <span className="badge badge-accent">{liveFile.name}</span>}
              {fileSheets.length > 0 && (
                <select value={selectedSheet} onChange={(e) => setSelectedSheet(e.target.value)}>
                  {fileSheets.map((s) => <option key={s} value={s}>Sheet: {s}</option>)}
                </select>
              )}
              <button className="btn btn-primary" disabled={busy || !liveFile || !models.length} onClick={evaluateFile}>
                {busy ? <span className="spinner" /> : null} Evaluate File &amp; Check Tolerances
              </button>
            </div>
          </div>
        )}

        {sourceTab === 'sql' && (
          <div>
            <div className="row" style={{ marginBottom: 10 }}>
              <button className="btn btn-success btn-sm" disabled={busy} onClick={connectDemoSQL}>1-Click Connect Demo Enterprise SQLite DB</button>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Score live operational batches directly from PostgreSQL, SQL Server, MySQL, Oracle, or SQLite.</span>
            </div>
            <div className="grid-2" style={{ marginBottom: 10 }}>
              <label className="field">SQLAlchemy Connection URL
                <input type="text" placeholder="sqlite:///... or postgresql://user:pass@host/db" value={sqlUrl} onChange={(e) => setSqlUrl(e.target.value)} />
              </label>
              <div className="row" style={{ alignItems: 'flex-end' }}>
                <button className="btn" disabled={busy || !sqlUrl.trim()} onClick={inspectSQL}>Inspect Tables</button>
                {sqlTables.length > 0 && (
                  <select value={sqlTable} onChange={(e) => setSqlTable(e.target.value)} style={{ flex: 1 }}>
                    {sqlTables.map((t: any) => <option key={t.name} value={t.name}>{t.name} ({t.row_count ?? '?'} rows)</option>)}
                  </select>
                )}
              </div>
            </div>
            <label className="field" style={{ marginBottom: 10 }}>Optional Read-Only SELECT Query
              <input type="text" placeholder="SELECT * FROM live_turbine_monitoring_batch" value={sqlQuery} onChange={(e) => setSqlQuery(e.target.value)} />
            </label>
            <button className="btn btn-primary" disabled={busy || !sqlUrl.trim() || !models.length} onClick={evaluateSQL}>
              {busy ? <span className="spinner" /> : null} Fetch SQL Batch &amp; Evaluate Tolerances
            </button>
          </div>
        )}

        {sourceTab === 'access' && (
          <div>
            <div className="grid-2" style={{ marginBottom: 10 }}>
              <label className="field">MS Access Database Path (.accdb / .mdb)
                <input type="text" placeholder="C:\\Plant\\live_sensors.accdb" value={accessPath} onChange={(e) => setAccessPath(e.target.value)} />
              </label>
              <div className="row" style={{ alignItems: 'flex-end' }}>
                <button className="btn" disabled={busy || !accessPath.trim()} onClick={inspectAccess}>Inspect Tables</button>
                {accessTables.length > 0 && (
                  <select value={accessTable} onChange={(e) => setAccessTable(e.target.value)} style={{ flex: 1 }}>
                    {accessTables.map((t: any) => <option key={t.name} value={t.name}>{t.name}</option>)}
                  </select>
                )}
              </div>
            </div>
            <button className="btn btn-primary" disabled={busy || !accessPath.trim() || !accessTable || !models.length} onClick={evaluateAccess}>
              Fetch MS Access Table &amp; Evaluate Tolerances
            </button>
          </div>
        )}

        {sourceTab === 'sim' && (
          <div>
            <p style={{ marginTop: 0, fontSize: 12.5, color: 'var(--text-muted)' }}>
              Test your tolerance thresholds and alarm pipeline immediately by streaming realistic operational readings derived from your dataset with configurable process-fault excursions.
            </p>
            <div className="grid-4" style={{ marginBottom: 10 }}>
              <label className="field">Batch Size (Rows)
                <input type="number" min={15} max={500} value={simRows} onChange={(e) => setSimRows(Number(e.target.value))} />
              </label>
              <label className="field">Injected Fault Rate (%)
                <input type="number" step={1} min={0} max={50} value={simFaultPct} onChange={(e) => setSimFaultPct(Number(e.target.value))} />
              </label>
              <label className="field">Sensor Noise Scale
                <input type="number" step={0.01} min={0} max={0.5} value={simNoise} onChange={(e) => setSimNoise(Number(e.target.value))} />
              </label>
              <div style={{ display: 'flex', alignItems: 'flex-end' }}>
                <button className="btn btn-primary" disabled={busy || !ws.has_dataset || !models.length} onClick={evaluateDemoBatch}>
                  {busy ? <span className="spinner" /> : null} Run Supervisory Live Batch
                </button>
              </div>
            </div>
          </div>
        )}

        {sourceTab === 'single' && (
          <div>
            <div className="grid-4" style={{ marginBottom: 10 }}>
              {singleFeatCols.map((c) => (
                <label key={c} className="field">{c}
                  <input type="text" placeholder="0" value={singleInputs[c] || ''} onChange={(e) => setSingleInputs((p) => ({ ...p, [c]: e.target.value }))} />
                </label>
              ))}
              <label className="field">Measured Actual ({ws.primary_target || 'Target'})
                <input type="text" placeholder="Optional measured value" value={singleActual} onChange={(e) => setSingleActual(e.target.value)} />
              </label>
            </div>
            <button className="btn btn-primary" disabled={busy || !models.length} onClick={evaluateSingle}>
              Evaluate Single Live Reading
            </button>
          </div>
        )}
      </div>

      {/* 4. ENGINEERING & SUPERVISORY CONTROL-ROOM DASHBOARD */}
      {evalRes && (
        <>
          {/* Live System Alarm Banner */}
          <div className={`supervisory-banner ${statusKey}`}>
            <div className="row" style={{ gap: 12 }}>
              <span className={`alarm-pulse ${statusKey}`} />
              <div>
                <div style={{ fontWeight: 800, fontSize: 15, display: 'flex', alignItems: 'center', gap: 8 }}>
                  {statusKey === 'critical' && <ShieldAlert size={18} color="#ef4444" />}
                  {statusKey === 'warning' && <AlertTriangle size={18} color="#f59e0b" />}
                  {statusKey === 'normal' && <CheckCircle2 size={18} color="#22c55e" />}
                  SYSTEM STATUS: {evalRes.system_status}
                </div>
                <div style={{ fontSize: 12.2, color: 'var(--text-muted)' }}>
                  Model: <strong>{evalRes.kpis?.model_label}</strong> vs Baseline: <strong>{evalRes.kpis?.baseline_label}</strong> · Source: {evalRes.source} · {evalRes.timestamp_utc}
                </div>
              </div>
            </div>
            <div className="row" style={{ gap: 8 }}>
              <span className="badge badge-green">Compliance: {Number(evalRes.kpis?.compliance_rate_pct ?? 100).toFixed(1)}%</span>
              <span className="badge badge-red">Breaches: {evalRes.kpis?.breach_count ?? 0}</span>
              <span className="badge badge-amber">Warnings: {evalRes.kpis?.warning_count ?? 0}</span>
            </div>
          </div>

          {evalRes.system_alerts?.length > 0 && (
            <div className="alert alert-error">
              <strong>Automated Supervisory Alerts Triggered:</strong>
              <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
                {evalRes.system_alerts.map((al: string, i: number) => <li key={i}>{al}</li>)}
              </ul>
            </div>
          )}

          {/* Executive Engineering KPI Grid */}
          <div className="kpi-grid">
            <div className="kpi-card"><div className="kpi-label">Records Evaluated</div><div className="kpi-value">{evalRes.kpis?.n_records ?? 0}</div></div>
            <div className="kpi-card"><div className="kpi-label">Tolerance Compliance</div><div className="kpi-value" style={{ color: (evalRes.kpis?.compliance_rate_pct ?? 100) >= 95 ? 'var(--green)' : 'var(--red)' }}>{Number(evalRes.kpis?.compliance_rate_pct ?? 100).toFixed(1)}%</div></div>
            <div className="kpi-card"><div className="kpi-label">Critical Breaches</div><div className="kpi-value" style={{ color: (evalRes.kpis?.breach_count ?? 0) > 0 ? 'var(--red)' : 'var(--green)' }}>{evalRes.kpis?.breach_count ?? 0}</div></div>
            <div className="kpi-card"><div className="kpi-label">Model MAE / MAPE</div><div className="kpi-value" style={{ fontSize: 16 }}>{evalRes.kpis?.mae !== undefined ? `${Number(evalRes.kpis.mae).toFixed(3)} (${Number(evalRes.kpis.mape_pct ?? 0).toFixed(2)}%)` : `${Number(evalRes.kpis?.accuracy_pct ?? 0).toFixed(1)}% Acc`}</div></div>
            <div className="kpi-card"><div className="kpi-label">Baseline MAE</div><div className="kpi-value" style={{ fontSize: 16 }}>{evalRes.kpis?.baseline_mae !== undefined ? Number(evalRes.kpis.baseline_mae).toFixed(3) : 'N/A'}</div></div>
            <div className="kpi-card"><div className="kpi-label">Gain vs Baseline</div><div className="kpi-value" style={{ color: (gainVsBase ?? 0) >= 0 ? 'var(--green)' : 'var(--amber)' }}>{gainVsBase !== undefined ? `${Number(gainVsBase) >= 0 ? '+' : ''}${Number(gainVsBase).toFixed(1)}%` : 'N/A'}</div></div>
          </div>

          {/* 4B. SIGNED ROOT-CAUSE DEVIATION ATTRIBUTION (WHAT CAUSED INCREASE / DECREASE VS BASELINE & HOW TO FIX IT) */}
          {rootCause && (
            <div className="card" style={{ borderLeft: '3px solid #f59e0b' }} data-supervisory-root-cause="true">
              <div className="card-title">
                <span className="row" style={{ gap: 8 }}>
                  <Wrench size={16} style={{ color: '#f59e0b' }} />
                  Root-Cause Attribution: What Caused the Increase (+) or Decrease (−) Relative to Baseline
                </span>
                <span
                  className={`badge ${
                    rootCause.overall_direction === 'INCREASE_ABOVE_BASELINE'
                      ? 'badge-red'
                      : rootCause.overall_direction === 'DECREASE_BELOW_BASELINE'
                        ? 'badge-accent'
                        : 'badge-green'
                  }`}
                >
                  {String(rootCause.overall_direction || 'ALIGNED').replace(/_/g, ' ')}
                </span>
              </div>

              {/* Executive Diagnosis Narrative */}
              <div
                className="alert alert-info"
                style={{ marginBottom: 12, fontSize: 13, lineHeight: 1.6 }}
              >
                <strong>Engineering Root-Cause Diagnosis ({rootCause.attribution_scope}):</strong> {rootCause.summary_narrative}
              </div>

              {/* Signed Deviation KPIs */}
              <div className="kpi-grid" style={{ marginBottom: 14 }}>
                <div className="kpi-card">
                  <div className="kpi-label">Mean Δ vs Baseline</div>
                  <div className="kpi-value" style={{ fontSize: 17, color: (rootCause.mean_actual_vs_baseline_delta ?? 0) >= 0 ? '#f87171' : '#38bdf8' }}>
                    {(rootCause.mean_actual_vs_baseline_delta ?? 0) >= 0 ? '+' : ''}
                    {Number(rootCause.mean_actual_vs_baseline_delta ?? 0).toFixed(4)}
                  </div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">% Shift vs Baseline</div>
                  <div className="kpi-value" style={{ fontSize: 17, color: (rootCause.mean_actual_vs_baseline_pct ?? 0) >= 0 ? '#f87171' : '#38bdf8' }}>
                    {(rootCause.mean_actual_vs_baseline_pct ?? 0) >= 0 ? '+' : ''}
                    {Number(rootCause.mean_actual_vs_baseline_pct ?? 0).toFixed(2)}%
                  </div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Champion − Baseline Δ</div>
                  <div className="kpi-value" style={{ fontSize: 17 }}>
                    {(rootCause.mean_champion_vs_baseline_delta ?? 0) >= 0 ? '+' : ''}
                    {Number(rootCause.mean_champion_vs_baseline_delta ?? 0).toFixed(4)}
                  </div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label">Unexplained Residual (Act − Model)</div>
                  <div className="kpi-value" style={{ fontSize: 17, color: Math.abs(rootCause.mean_unexplained_residual ?? 0) > Math.abs(rootCause.mean_champion_vs_baseline_delta ?? 0) ? '#fbbf24' : 'var(--green)' }}>
                    {(rootCause.mean_unexplained_residual ?? 0) >= 0 ? '+' : ''}
                    {Number(rootCause.mean_unexplained_residual ?? 0).toFixed(4)}
                  </div>
                </div>
              </div>

              {/* Signed Diverging Waterfall / Tornado Attribution Chart */}
              {rootCause.all_factor_attributions?.length > 0 && (
                <div style={{ marginBottom: 14 }}>
                  <DivergingDriverBarSVG
                    title="Signed Factor Attribution Tornado — Drivers of Increase (+Δ) vs. Decrease (−Δ) Relative to Baseline"
                    subtitle={rootCause.attribution_method}
                    items={rootCause.all_factor_attributions}
                  />
                </div>
              )}

              {/* Split Tables: UPWARD (+ Increase) Drivers vs DOWNWARD (- Decrease) Drivers */}
              <div className="grid-2" style={{ gap: 12, marginBottom: 14 }}>
                <div className="card" style={{ marginBottom: 0, borderTop: '2px solid #f87171' }}>
                  <div className="card-title" style={{ color: '#f87171' }}>
                    <span className="row" style={{ gap: 6 }}>
                      <TrendingUp size={15} /> Factors Driving INCREASE (+Δ Above Baseline)
                    </span>
                    <span className="badge badge-red">{rootCause.increase_drivers?.length ?? 0}</span>
                  </div>
                  {rootCause.increase_drivers?.length > 0 ? (
                    <SimpleTable
                      title="Upward Deviation Drivers (+Δ)"
                      rows={rootCause.increase_drivers.map((d: any) => ({
                        Feature: d.feature,
                        'Live Mean': d.live_mean,
                        'Normal Ref': d.normal_reference,
                        'Shift (σ)': `${d.z_shift >= 0 ? '+' : ''}${d.z_shift}σ`,
                        'Impact (+Δ)': `+${Number(d.signed_impact).toFixed(4)}`,
                        'Share %': `${d.impact_share_pct}%`,
                        'Healthy Band': d.healthy_band,
                      }))}
                      maxRows={10}
                    />
                  ) : (
                    <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>No significant upward (+Δ) drivers detected.</div>
                  )}
                </div>

                <div className="card" style={{ marginBottom: 0, borderTop: '2px solid #38bdf8' }}>
                  <div className="card-title" style={{ color: '#38bdf8' }}>
                    <span className="row" style={{ gap: 6 }}>
                      <TrendingDown size={15} /> Factors Driving DECREASE (−Δ Below Baseline)
                    </span>
                    <span className="badge badge-accent">{rootCause.decrease_drivers?.length ?? 0}</span>
                  </div>
                  {rootCause.decrease_drivers?.length > 0 ? (
                    <SimpleTable
                      title="Downward Deviation Drivers (−Δ)"
                      rows={rootCause.decrease_drivers.map((d: any) => ({
                        Feature: d.feature,
                        'Live Mean': d.live_mean,
                        'Normal Ref': d.normal_reference,
                        'Shift (σ)': `${d.z_shift >= 0 ? '+' : ''}${d.z_shift}σ`,
                        'Impact (−Δ)': Number(d.signed_impact).toFixed(4),
                        'Share %': `${d.impact_share_pct}%`,
                        'Healthy Band': d.healthy_band,
                      }))}
                      maxRows={10}
                    />
                  ) : (
                    <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>No significant downward (−Δ) drivers detected.</div>
                  )}
                </div>
              </div>

              {/* Actionable Engineering Remediation Playbook */}
              {rootCause.remediation_playbook?.length > 0 && (
                <div>
                  <SimpleTable
                    title="Prioritized Corrective Engineering Playbook — How to Restore System to Baseline"
                    rows={rootCause.remediation_playbook}
                    maxRows={10}
                  />
                </div>
              )}
            </div>
          )}

          {/* 4 Engineering Charts (All automatically Zoomable + Pop-Out!) */}
          <div className="grid-2" style={{ marginBottom: 14 }}>
            <div className="card" style={{ marginBottom: 0 }}>
              <SupervisoryTrendSVG points={evalRes.trend_series || []} />
            </div>
            <div className="card" style={{ marginBottom: 0 }}>
              <SPCControlSVG points={evalRes.spc_points || []} limits={evalRes.spc_limits} />
            </div>
          </div>

          {evalRes.error_histogram?.length > 0 && (
            <div className="card">
              <BarChartSVG
                title="Live Prediction Error Distribution (% Error Histogram)"
                items={evalRes.error_histogram.map((b: any) => ({
                  label: b.label || b.bin || '',
                  value: Number(b.count || 0),
                  color: b.breach ? '#ef4444' : '#38bdf8',
                }))}
                height={190}
              />
            </div>
          )}

          {/* Active Alarm & Root-Cause Diagnostics Ledger */}
          <div className="card">
            <div className="card-title">
              <span>Tolerance Breach &amp; Root-Cause Alarm Ledger ({filteredAlarms.length})</span>
              <div className="row" style={{ gap: 6 }}>
                {(['ALL', 'CRITICAL', 'WARNING'] as const).map((f) => (
                  <button key={f} className={`btn btn-sm ${alarmFilter === f ? 'btn-primary' : ''}`} onClick={() => setAlarmFilter(f)}>{f}</button>
                ))}
              </div>
            </div>
            {filteredAlarms.length > 0 ? (
              <>
                <SimpleTable title="Triggered Tolerance Alarms with Signed Baseline Root Causes & Recommended Fixes" rows={alarmDisplayRows} maxRows={100} />

                {/* Interactive Single-Alarm Record Root-Cause Inspector */}
                {inspectedAlarm && inspectedAlarm.factor_breakdown?.length > 0 && (
                  <div style={{ marginTop: 14, padding: 12, borderRadius: 10, background: 'var(--bg-subtle)', border: '1px solid var(--border)' }}>
                    <div className="row" style={{ justifyContent: 'space-between', marginBottom: 10, gap: 10 }}>
                      <div>
                        <strong style={{ fontSize: 13.5, color: 'var(--accent)' }}>
                          Single-Alarm Deep Root-Cause Drilldown
                        </strong>
                        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                          Inspect exact input factors that pushed a specific alarmed reading above or below Baseline:
                        </div>
                      </div>
                      <select
                        value={Math.min(selectedAlarmIdx, filteredAlarms.length - 1)}
                        onChange={(e) => setSelectedAlarmIdx(Number(e.target.value))}
                        style={{ minWidth: 240 }}
                      >
                        {filteredAlarms.slice(0, 40).map((a: any, idx: number) => (
                          <option key={`${a.row_id}-${idx}`} value={idx}>
                            Record {a.row_id} · {a.deviation_direction || a.severity} (Δ={a.delta_vs_baseline ?? 'N/A'})
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="alert alert-warn" style={{ marginBottom: 10, fontSize: 12.5 }}>
                      <strong>Recommended Corrective Action for Record {inspectedAlarm.row_id}:</strong> {inspectedAlarm.recommended_fix}
                    </div>
                    <DivergingDriverBarSVG
                      title={`Record ${inspectedAlarm.row_id} — Signed Input Contribution to Baseline Deviation`}
                      subtitle={`Actual=${inspectedAlarm.actual ?? 'N/A'} | Champion=${inspectedAlarm.predicted} | Baseline=${inspectedAlarm.baseline_predicted} | Δ vs Baseline=${inspectedAlarm.delta_vs_baseline ?? 'N/A'}`}
                      items={inspectedAlarm.factor_breakdown.map((fb: any) => ({
                        feature: fb.feature,
                        signed_impact: fb.signed_impact,
                        z_shift: fb.z_shift,
                        live_mean: fb.live_value,
                        normal_reference: fb.normal_ref,
                      }))}
                    />
                  </div>
                )}
              </>
            ) : (
              <div className="alert alert-success">Zero alarms in this severity filter — all readings operating inside permissible tolerance limits.</div>
            )}
          </div>

          {/* Full Scored Live Telemetry Table */}
          <div className="card">
            <div className="card-title">Full Scored Operational Telemetry (Actual vs. Model vs. Baseline)</div>
            <SimpleTable title="Live Scored Operational Records" rows={evalRes.telemetry_rows || []} maxRows={120} />
          </div>
        </>
      )}

      {/* Longitudinal Supervisory Run History */}
      {history.length > 0 && (
        <div className="card">
          <div className="card-title">Longitudinal Supervisory Evaluation History ({history.length} runs)</div>
          <SimpleTable title="Supervisory Monitoring Run Ledger" rows={history} maxRows={30} />
        </div>
      )}
    </div>
  );
};
