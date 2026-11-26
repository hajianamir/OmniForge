import React, { useEffect, useRef, useState } from 'react';
import { apiDelete, apiDownload, apiGet, apiPost, apiUpload, toErrorMessage } from '../api';
import { BarChartSVG, CanvasScatter, HeatmapGrid } from '../components/Charts';
import { PaginatedDataGrid, SimpleTable } from '../components/DataGrid';
import { StudioHelpButton, type HandbookSectionId } from '../components/HelpGuideModal';
import { LiveChartGenerator } from '../components/LiveChartGenerator';
import { MathEquationCard, type EquationEntry } from '../components/MathEquation';
import { ModeBar, type ExecutionMode } from '../components/ModeBar';
import type {
  AdvancedMiningResponse,
  AnomalyResponse,
  CleanResponse,
  ClusteringCommitResponse,
  AnomalyCommitResponse,
  DeletedLogResponse,
  ExcelSheetsResponse,
  IngestResponse,
  QualityResponse,
  SampleCatalogItem,
  SampleCatalogResponse,
  SQLInspectResponse,
  TableInfo,
  CleaningReport,
  ClusteringResponse,
  ClusterChoice,
  DiagnosticsResponse,
  EDAResponse,
  EngineerResponse,
  HypothesisResponse,
  IntakeAuditPayload,
  MiningResponse,
  SelectionResponse,
  StateEnvelope,
} from '../studioTypes';
import type { StudioCache, StudioPayload, StudioResultKey } from '../studioTypes';
import type { ScatterPayload, WorkspaceState } from '../types';

export interface StudioProps {
  ws: WorkspaceState;
  onSync: (next?: WorkspaceState) => void;
  gridVersion: number;
  bumpGrid: () => void;
  studioCache: StudioCache;
  setStudioResult: (key: StudioResultKey, val: StudioPayload) => void;
  refreshCache: () => void;
  openHelp: (section?: HandbookSectionId) => void;
}

const EMPTY_SCATTER: ScatterPayload = { x: [], y: [], total_points: 0, rendered_points: 0, downsampled: false };

export const ProvenancePill: React.FC<{ ws: WorkspaceState }> = ({ ws }) => {
  if (!ws.has_dataset) return null;
  const processed = Boolean(ws.processing_status?.is_processed);
  const label = ws.processing_status?.detailed_label || (processed ? 'PROCESSED' : 'RAW');
  return (
    <span
      className={`badge ${processed ? 'badge-green' : 'badge-amber'}`}
      title={`Analysis DataFrame: Active ${ws.n_rows}×${ws.n_cols} (${ws.analysis_cols_count} active cols after ${ws.excluded_columns.length} exclusion(s)) vs Immutable Raw ${ws.raw_rows}×${ws.raw_cols}.`}
    >
      Active Input: {label} ({ws.n_rows.toLocaleString()} × {ws.analysis_cols_count})
    </span>
  );
};

// ============================================================================
// Studio 01 — Multi-Source Data Ingestion, Auto Defect Audit & Full Cleaning
// ============================================================================
export const Studio00Ingestion: React.FC<StudioProps> = (props) => {
  const [relData, setRelData] = useState<{
    n_rows: number;
    n_cols: number;
    target?: string | null;
    targets?: string[];
    target_relationships?: Array<{ feature: string; target: string; pearson: number; spearman: number; dcor: number; relationship_type: string }>;
    pairs: Array<{ feature_a: string; feature_b: string; pearson: number; spearman: number; dcor: number; relationship_type: string }>;
    correlation_matrix?: { columns?: string[]; values?: (number | null)[][] };
  } | null>(null);
  const [relBusy, setRelBusy] = useState(false);
  const [relErr, setRelErr] = useState<string | null>(null);
  const relationshipRequest = useRef(0);

  const loadRelationships = async () => {
    const request = ++relationshipRequest.current;
    setRelBusy(true);
    setRelErr(null);
    try {
      const res = await apiGet<typeof relData>('/api/workbench/s00/data_relationships');
      if (request === relationshipRequest.current) setRelData(res);
    } catch (er: unknown) {
      if (request === relationshipRequest.current) {
        setRelData(null);
        setRelErr(toErrorMessage(er));
      }
    } finally {
      if (request === relationshipRequest.current) setRelBusy(false);
    }
  };

  useEffect(() => {
    // Dataset replacements and edits can retain the same target/exclusion
    // names. Clear and recompute the relationship chart on every grid change.
    setRelData(null);
    setRelErr(null);
    if (props.ws.has_dataset) {
      void loadRelationships();
    } else {
      setRelBusy(false);
    }
    return () => { relationshipRequest.current += 1; };
  }, [props.gridVersion, props.ws.has_dataset, props.ws.targets.join(','), props.ws.excluded_columns.join(',')]);

  return (
    <div>
      <Studio01Data {...props} sectionMode="ingestion" />
      {props.ws.has_dataset && (
        <div className="card" style={{ marginTop: 14 }}>
          <div className="card-title">
            <span>Data Relationships &amp; Dependency Types (Studio 00)</span>
            <button className="btn btn-secondary" disabled={relBusy} onClick={loadRelationships}>
              {relBusy ? 'Analyzing…' : 'Analyze Data Relationships (Chart & Table)'}
            </button>
          </div>
          {relErr && <div className="alert alert-error">{relErr}</div>}
          {relData && ((relData.target_relationships?.length ?? 0) > 0 || relData.pairs.length > 0) && (
            <div className="grid-2" style={{ marginBottom: 14 }}>
              {(relData.target_relationships?.length ?? 0) > 0 && (
                <BarChartSVG
                  title={`Target Correlation Profile (${(relData.targets && relData.targets.length > 0 ? relData.targets.join(', ') : relData.target) || 'Target'})`}
                  items={(relData.target_relationships || []).slice(0, 14).map((tr) => ({
                    label: tr.feature,
                    value: Number(tr.pearson) || 0,
                    color: tr.pearson >= 0 ? '#38bdf8' : '#f87171',
                  }))}
                  height={195}
                />
              )}
              {relData.pairs.length > 0 && (
                <BarChartSVG
                  title="Top Pairwise Nonlinear / Linear Dependency (max |r|, dCor)"
                  items={relData.pairs.slice(0, 12).map((p) => ({
                    label: `${p.feature_a}×${p.feature_b}`,
                    value: Math.max(Math.abs(Number(p.pearson) || 0), Number(p.dcor) || 0),
                    color: '#34d399',
                  }))}
                  height={195}
                />
              )}
            </div>
          )}
          {relData?.correlation_matrix && (relData.correlation_matrix.columns?.length ?? 0) > 1 && (
            <div style={{ marginBottom: 14 }}>
              <HeatmapGrid
                title="Studio 00 Variable Association Matrix"
                columns={relData.correlation_matrix.columns || []}
                values={relData.correlation_matrix.values || []}
              />
            </div>
          )}
          {relData && (relData.target_relationships?.length ?? 0) === 0 && relData.pairs.length === 0 && (
            <div style={{ color: 'var(--text-muted)', fontSize: 12.5 }}>
              Need at least 2 active columns and 5 non-empty paired rows to compute variable relationship charts.
            </div>
          )}
          {relData && relData.target_relationships && relData.target_relationships.length > 0 && (
            <div style={{ marginBottom: 14, overflowX: 'auto' }}>
              <div style={{ fontWeight: 700, fontSize: 12.5, marginBottom: 6, color: '#38bdf8' }}>
                Variable Relationships with Target{(relData.targets?.length ?? 0) > 1 ? 's' : ''} ({(relData.targets && relData.targets.length > 0 ? relData.targets.join(', ') : relData.target)}) — {relData.target_relationships.length} pairs
              </div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Target</th>
                    <th>Pearson r</th>
                    <th>Spearman ρ</th>
                    <th>dCor</th>
                    <th>Relationship Type</th>
                  </tr>
                </thead>
                <tbody>
                  {relData.target_relationships.map((tr, idx) => (
                    <tr key={idx}>
                      <td><strong>{tr.feature}</strong></td>
                      <td>{tr.target}</td>
                      <td>{tr.pearson.toFixed(4)}</td>
                      <td>{tr.spearman.toFixed(4)}</td>
                      <td>{tr.dcor.toFixed(4)}</td>
                      <td><span className="badge">{tr.relationship_type}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {relData && relData.pairs.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {relData.pairs.slice(0, 8).map((p, idx) => (
                  <div key={idx} style={{ fontSize: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                      <span><strong>{p.feature_a}</strong> ↔ <strong>{p.feature_b}</strong></span>
                      <span className="badge">{p.relationship_type}</span>
                    </div>
                    <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                      <span style={{ width: 70, color: '#94a3b8' }}>|r|={Math.abs(p.pearson).toFixed(2)}</span>
                      <div style={{ flex: 1, height: 8, background: 'rgba(148,163,184,0.2)', borderRadius: 4, overflow: 'hidden' }}>
                        <div style={{ width: `${Math.min(100, Math.abs(p.pearson) * 100)}%`, height: '100%', background: '#38bdf8' }} />
                      </div>
                      <span style={{ width: 75, color: '#94a3b8' }}>dCor={p.dcor.toFixed(2)}</span>
                      <div style={{ flex: 1, height: 8, background: 'rgba(148,163,184,0.2)', borderRadius: 4, overflow: 'hidden' }}>
                        <div style={{ width: `${Math.min(100, p.dcor * 100)}%`, height: '100%', background: '#34d399' }} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Feature A</th>
                      <th>Feature B</th>
                      <th>Pearson r</th>
                      <th>Spearman ρ</th>
                      <th>Distance Corr (dCor)</th>
                      <th>Relationship Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {relData.pairs.map((p, idx) => (
                      <tr key={idx}>
                        <td>{p.feature_a}</td>
                        <td>{p.feature_b}</td>
                        <td>{p.pearson}</td>
                        <td>{p.spearman}</td>
                        <td>{p.dcor}</td>
                        <td>{p.relationship_type}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export const Studio01Data: React.FC<StudioProps & { sectionMode?: 'ingestion' | 'cleaning' }> = ({ ws, onSync, gridVersion, bumpGrid, studioCache, setStudioResult, refreshCache, openHelp, sectionMode = 'cleaning' }) => {
  const [ingestTab, setIngestTab] = useState<'file' | 'sql' | 'access' | 'samples'>('file');
  const [catalog, setCatalog] = useState<{ benchmarks: SampleCatalogItem[]; domains: SampleCatalogItem[] }>({ benchmarks: [], domains: [] });
  const [viewMode, setViewMode] = useState<'active' | 'raw'>('active');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const quality = studioCache.s01_quality ?? null;
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [referenceKey, setReferenceKey] = useState('');
  const [deletedLog, setDeletedLog] = useState<DeletedLogResponse | null>(null);

  // Excel multi-sheet upload state
  const [pendingExcel, setPendingExcel] = useState<File | null>(null);
  const [excelSheets, setExcelSheets] = useState<string[]>([]);
  const [selectedSheet, setSelectedSheet] = useState<string>('');

  // SQL Connector state
  const [sqlUrl, setSqlUrl] = useState('');
  const [sqlTables, setSqlTables] = useState<TableInfo[]>([]);
  const [sqlTable, setSqlTable] = useState('');
  const [sqlQuery, setSqlQuery] = useState('');

  // MS Access Connector state
  const [accessPath, setAccessPath] = useState('');
  const [accessTables, setAccessTables] = useState<TableInfo[]>([]);
  const [accessTable, setAccessTable] = useState('');

  // Restored & Expanded Data Cleaning state (Auto / Complete / Custom)
  const [cleanMode, setCleanMode] = useState<ExecutionMode>('auto');
  const [imputeNum, setImputeNum] = useState('median');
  const [imputeCat, setImputeCat] = useState('most_frequent');
  const [dropDups, setDropDups] = useState(true);
  const [outlierMethod, setOutlierMethod] = useState('iqr');
  const [outlierAction, setOutlierAction] = useState('flag');
  const [outlierThresh, setOutlierThresh] = useState(1.5);
  const [stdMissingTokens, setStdMissingTokens] = useState(true);
  const [cleanStrings, setCleanStrings] = useState(true);
  const [stripHtml, setStripHtml] = useState(true);
  const [lowercaseStr, setLowercaseStr] = useState(false);
  const [inferTypes, setInferTypes] = useState(true);
  const [dropHighMissing, setDropHighMissing] = useState(true);
  const [missingColThresh, setMissingColThresh] = useState(0.85);
  const [addMissingFlags, setAddMissingFlags] = useState(false);
  const [dropConstCols, setDropConstCols] = useState(true);
  const [stdColNames, setStdColNames] = useState(false);
  const cleanPreview = studioCache.s01_clean ?? null;

  useEffect(() => {
    apiGet<SampleCatalogResponse>('/api/workbench/s01/catalog')
      .then((res) => setCatalog({ benchmarks: res.benchmarks || [], domains: res.domains || [] }))
      .catch(() => {});
  }, []);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setErr(null); setMsg(null);
    const lower = file.name.toLowerCase();
    if (lower.endsWith('.xlsx') || lower.endsWith('.xls') || lower.endsWith('.xlsm')) {
      setBusy(true);
      try {
        const fd = new FormData();
        fd.append('file', file);
        const info = await apiUpload<ExcelSheetsResponse>('/api/workbench/s01/excel-sheets', fd);
        const sheets: string[] = info.sheets || [];
        if (sheets.length > 1) {
          setPendingExcel(file);
          setExcelSheets(sheets);
          setSelectedSheet(sheets[0] ?? '');
          setMsg(`Detected ${sheets.length} Excel worksheets in '${file.name}'. Choose a sheet and click Load Sheet.`);
          setBusy(false);
          e.target.value = '';
          return;
        }
      } catch {
        // Proceed with standard single-sheet upload
      }
      setBusy(false);
    }
    await uploadFileNow(file, undefined);
    e.target.value = '';
  };

  const uploadFileNow = async (file: File, sheetName?: string) => {
    setBusy(true); setErr(null); setMsg(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      if (sheetName) fd.append('sheet_name', sheetName);
      const res = await apiUpload<IngestResponse>('/api/workbench/s01/upload', fd);
      onSync(res.state); bumpGrid();
      setPendingExcel(null); setExcelSheets([]);
      const dc = res.intake_audit?.defect_counts?.total ?? 0;
      setMsg(`Ingested '${file.name}' in ${res.elapsed_seconds}s · Automatic Intake Audit found ${dc} defect(s) (Grade ${res.intake_audit?.quality_grade || 'N/A'}).`);
    } catch (er: unknown) { setErr(toErrorMessage(er)); } finally { setBusy(false); }
  };

  const setupDemoSQL = async () => {
    setBusy(true); setErr(null); setMsg(null);
    try {
      const info = await apiPost<SQLInspectResponse>('/api/workbench/s01/sql/demo', {});
      setSqlUrl(info.connection_url ?? '');
      setSqlTables(info.tables || []);
      if (info.tables?.length) setSqlTable(info.tables[0]?.name ?? '');
      setMsg(`Connected to built-in OmniForge Enterprise Demo SQLite DB (${(info.tables || []).length} tables ready).`);
    } catch (er: unknown) { setErr(toErrorMessage(er)); } finally { setBusy(false); }
  };

  const inspectSQL = async () => {
    if (!sqlUrl.trim()) return;
    setBusy(true); setErr(null);
    try {
      const info = await apiPost<SQLInspectResponse>('/api/workbench/s01/sql/inspect', { connection_url: sqlUrl });
      setSqlTables(info.tables || []);
      if (info.tables?.length) setSqlTable(info.tables[0].name);
      setMsg(`SQL database inspected: ${(info.tables || []).length} table(s) discovered.`);
    } catch (er: unknown) { setErr(toErrorMessage(er)); } finally { setBusy(false); }
  };

  const loadSQL = async () => {
    setBusy(true); setErr(null);
    try {
      const res = await apiPost<IngestResponse>('/api/workbench/s01/sql/load', {
        connection_url: sqlUrl,
        table_name: sqlQuery.trim() ? null : (sqlTable || null),
        query: sqlQuery.trim() || null,
      });
      onSync(res.state); bumpGrid();
      setMsg(`Loaded SQL table in ${res.elapsed_seconds}s · Automatic Intake Audit complete (${res.intake_audit?.defect_counts?.total ?? 0} defects).`);
    } catch (er: unknown) { setErr(toErrorMessage(er)); } finally { setBusy(false); }
  };

  const inspectAccess = async () => {
    if (!accessPath.trim()) return;
    setBusy(true); setErr(null);
    try {
      const info = await apiPost<SQLInspectResponse>('/api/workbench/s01/access/inspect', { file_path: accessPath });
      setAccessTables(info.tables || []);
      if (info.tables?.length) setAccessTable(info.tables[0]?.name ?? '');
      setMsg(`MS Access database inspected: ${(info.tables || []).length} table(s) found.`);
    } catch (er: unknown) { setErr(toErrorMessage(er)); } finally { setBusy(false); }
  };

  const loadAccess = async () => {
    setBusy(true); setErr(null);
    try {
      const res = await apiPost<IngestResponse>('/api/workbench/s01/access/load', { file_path: accessPath, table_name: accessTable });
      onSync(res.state); bumpGrid();
      setMsg(`Loaded MS Access table '${accessTable}' in ${res.elapsed_seconds}s.`);
    } catch (er: unknown) { setErr(toErrorMessage(er)); } finally { setBusy(false); }
  };

  const loadSample = async (sourceType: 'benchmark' | 'domain', name: string) => {
    setBusy(true); setErr(null); setMsg(null);
    try {
      const res = await apiPost<IngestResponse>('/api/workbench/s01/load-sample', { source_type: sourceType, name, n_samples: 800 });
      onSync(res.state); bumpGrid();
      setMsg(`Loaded '${name}' in ${res.elapsed_seconds}s · Auto Intake Audit: Grade ${res.intake_audit?.quality_grade || 'N/A'} (${res.intake_audit?.defect_counts?.total ?? 0} defects).`);
    } catch (er: unknown) { setErr(toErrorMessage(er)); } finally { setBusy(false); }
  };

  const runProfile = async (withReference = false) => {
    setBusy(true); setErr(null);
    try {
      const form = new FormData();
      if (withReference && referenceFile) {
        form.append('file', referenceFile);
        form.append('key_column', referenceKey);
      }
      const res = withReference
        ? await apiUpload<QualityResponse>('/api/workbench/s01/profile-reference', form)
        : await apiPost<QualityResponse>('/api/workbench/s01/profile');
      if (res.state) onSync(res.state);
      setStudioResult('s01_quality', res);
    }
    catch (er: unknown) { setErr(toErrorMessage(er)); } finally { setBusy(false); }
  };

  const runClean = async (previewOnly = false, overrideMode?: ExecutionMode) => {
    setBusy(true); setErr(null); setMsg(null);
    try {
      const useMode = overrideMode || cleanMode;
      const res = await apiPost<CleanResponse>('/api/workbench/s01/clean', {
        mode: useMode,
        impute_numeric: imputeNum,
        impute_categorical: imputeCat,
        drop_duplicates: dropDups,
        handle_outliers: outlierAction,
        outlier_method: outlierMethod,
        outlier_threshold: outlierThresh,
        standardize_missing_tokens: stdMissingTokens,
        clean_strings: cleanStrings,
        lowercase_strings: lowercaseStr,
        strip_html: stripHtml,
        infer_types: inferTypes,
        drop_high_missing_cols: dropHighMissing,
        missing_col_threshold: missingColThresh,
        add_missing_indicators: addMissingFlags,
        drop_constant_columns: dropConstCols,
        standardize_column_names: stdColNames,
        preview_only: previewOnly,
      });
      if (!previewOnly) {
        onSync(res.state);
        setStudioResult('s01_clean', res);
        bumpGrid();
        setMsg(`Cleaning (${useMode.toUpperCase()}) committed: ${res.cleaning_report?.rows_before} → ${res.cleaning_report?.rows_after} rows, missing cells ${res.cleaning_report?.missing_before} → ${res.cleaning_report?.missing_after}.`);
      } else {
        setStudioResult('s01_clean', res);
        setMsg(`Dry-Run Preview (${useMode.toUpperCase()}): would transform ${res.cleaning_report?.rows_before} → ${res.cleaning_report?.rows_after} rows & ${res.cleaning_report?.missing_before} → ${res.cleaning_report?.missing_after} missing cells.`);
      }
    } catch (er: unknown) { setErr(toErrorMessage(er)); } finally { setBusy(false); }
  };

  const ia = ws.intake_audit;

  return (
    <div>
      <div className="studio-hero">
        <div>
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <h1>{sectionMode === 'ingestion' ? '00 · Data Ingestion, Intake Audit & Split Lock' : '01 · Data Preparation & Cleaning'}</h1>
            <StudioHelpButton section="studio-01" onOpenHelp={openHelp} title="Open Chapter 01: Ingestion, Split Lock, Imputation & Outlier Formulas" />
          </div>
          <p>Ingest CSV, Multi-Sheet Excel (.xlsx), Parquet, SQL Databases, or MS Access (.accdb/.mdb) — instant automatic statistical &amp; defect auditing plus full Auto / Complete / Custom data cleaning.</p>
        </div>
        {ws.has_dataset && (
          <div className="row">
            <ProvenancePill ws={ws} />
            <button className="btn btn-sm" onClick={() => apiDownload('/api/workbench/s01/export-clean/csv', `${ws.dataset_name || 'dataset'}_clean.csv`)}>Export Clean CSV</button>
            <button className="btn btn-sm" onClick={() => apiDownload('/api/workbench/s01/export-clean/xlsx', `${ws.dataset_name || 'dataset'}_clean.xlsx`)}>Export Clean XLSX</button>
            <button className="btn btn-sm" disabled={!ws.can_undo} onClick={async () => { const r = await apiPost<StateEnvelope>('/api/workbench/s01/undo'); onSync(r.state); refreshCache(); bumpGrid(); }}>Undo</button>
            <button className="btn btn-sm" onClick={async () => { const r = await apiPost<StateEnvelope>('/api/workbench/s01/revert'); onSync(r.state); refreshCache(); bumpGrid(); }}>Revert to Raw</button>
            <button className="btn btn-sm btn-danger" onClick={async () => { const r = await apiDelete<StateEnvelope>('/api/workbench/dataset'); onSync(r.state); refreshCache(); bumpGrid(); }}>Clear Dataset</button>
          </div>
        )}
      </div>

      {msg && <div className="alert alert-success">{msg}</div>}
      {err && <div className="alert alert-error">{err}</div>}

      <LiveChartGenerator key={sectionMode} studioId={sectionMode === 'ingestion' ? '00' : '01'} ws={ws} studioCache={studioCache} />

      {/* Multi-Source Ingestion Hub */}
      {sectionMode === 'ingestion' && (
      <div className="card">
        <div className="tabs" style={{ marginBottom: 12 }}>
          <button className={`tab-btn ${ingestTab === 'file' ? 'active' : ''}`} onClick={() => setIngestTab('file')}>Upload File (CSV / XLSX / Parquet / JSON)</button>
          <button className={`tab-btn ${ingestTab === 'sql' ? 'active' : ''}`} onClick={() => setIngestTab('sql')}>SQL Database Connector</button>
          <button className={`tab-btn ${ingestTab === 'access' ? 'active' : ''}`} onClick={() => setIngestTab('access')}>MS Access (.accdb / .mdb)</button>
          <button className={`tab-btn ${ingestTab === 'samples' ? 'active' : ''}`} onClick={() => setIngestTab('samples')}>Benchmark &amp; Domain Datasets</button>
        </div>

        {ingestTab === 'file' && (
          <div>
            <div className="row" style={{ marginBottom: 10 }}>
              <label className="btn btn-primary" style={{ cursor: 'pointer' }}>
                {busy ? <span className="spinner" /> : null} Upload CSV / Excel (.xlsx) / Parquet / JSON
                <input type="file" accept=".csv,.tsv,.xlsx,.xls,.xlsm,.parquet,.json" onChange={handleFileSelect} style={{ display: 'none' }} />
              </label>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Multi-sheet Excel workbooks automatically show a worksheet picker; intake defect audit runs automatically upon load.</span>
            </div>
            {pendingExcel && excelSheets.length > 0 && (
              <div className="row" style={{ padding: 10, background: 'var(--bg-subtle)', borderRadius: 8 }}>
                <strong>Excel Workbook: {pendingExcel.name}</strong>
                <select value={selectedSheet} onChange={(e) => setSelectedSheet(e.target.value)}>
                  {excelSheets.map((sh) => <option key={sh} value={sh}>{sh}</option>)}
                </select>
                <button className="btn btn-primary btn-sm" disabled={busy} onClick={() => uploadFileNow(pendingExcel, selectedSheet)}>Load Selected Worksheet</button>
              </div>
            )}
          </div>
        )}

        {ingestTab === 'sql' && (
          <div>
            <div className="row" style={{ marginBottom: 10 }}>
              <button className="btn btn-success btn-sm" disabled={busy} onClick={setupDemoSQL}>1-Click Connect Demo Enterprise SQLite DB</button>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Supports SQLite, PostgreSQL, MySQL, SQL Server, Oracle &amp; DuckDB via SQLAlchemy URLs.</span>
            </div>
            <div className="grid-2" style={{ marginBottom: 10 }}>
              <label className="field">SQLAlchemy Connection URL or Local SQLite Path
                <input type="text" placeholder="sqlite:///path/to/db.sqlite or postgresql://user:pass@host/db" value={sqlUrl} onChange={(e) => setSqlUrl(e.target.value)} />
              </label>
              <div className="row" style={{ alignItems: 'flex-end' }}>
                <button className="btn" disabled={busy || !sqlUrl.trim()} onClick={inspectSQL}>Inspect Tables</button>
                {sqlTables.length > 0 && (
                  <select value={sqlTable} onChange={(e) => setSqlTable(e.target.value)} style={{ flex: 1 }}>
                    {sqlTables.map((t) => <option key={t.name} value={t.name}>{t.name} ({t.row_count ?? '?'} rows)</option>)}
                  </select>
                )}
              </div>
            </div>
            <label className="field" style={{ marginBottom: 10 }}>Optional Read-Only SELECT Query (overrides table selection)
              <input type="text" placeholder="SELECT * FROM industrial_turbine_telemetry WHERE load_pct > 50" value={sqlQuery} onChange={(e) => setSqlQuery(e.target.value)} />
            </label>
            <button className="btn btn-primary" disabled={busy || !sqlUrl.trim() || (!sqlTable && !sqlQuery.trim())} onClick={loadSQL}>
              {busy ? <span className="spinner" /> : null} Load Data from SQL Database
            </button>
          </div>
        )}

        {ingestTab === 'access' && (
          <div>
            <p style={{ marginTop: 0, fontSize: 12.5, color: 'var(--text-muted)' }}>Connect to Microsoft Access <code>.accdb</code> / <code>.mdb</code> databases via Windows ODBC ACE Driver (<code>pyodbc</code>) or Linux/macOS <code>mdbtools</code>.</p>
            <div className="grid-2" style={{ marginBottom: 10 }}>
              <label className="field">MS Access File Path (.accdb / .mdb)
                <input type="text" placeholder="C:\\Data\\plant_operations.accdb or /data/telemetry.mdb" value={accessPath} onChange={(e) => setAccessPath(e.target.value)} />
              </label>
              <div className="row" style={{ alignItems: 'flex-end' }}>
                <button className="btn" disabled={busy || !accessPath.trim()} onClick={inspectAccess}>Inspect Access Tables</button>
                {accessTables.length > 0 && (
                  <select value={accessTable} onChange={(e) => setAccessTable(e.target.value)} style={{ flex: 1 }}>
                    {accessTables.map((t) => <option key={t.name} value={t.name}>{t.name}</option>)}
                  </select>
                )}
              </div>
            </div>
            <button className="btn btn-primary" disabled={busy || !accessPath.trim() || !accessTable} onClick={loadAccess}>
              Load Table from MS Access
            </button>
          </div>
        )}

        {ingestTab === 'samples' && (
          <div className="grid-2">
            <div>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-muted)', marginBottom: 6 }}>Benchmark CSV Datasets</div>
              <div className="row" style={{ gap: 6 }}>
                {(catalog.benchmarks || []).map((b) => (
                  <button key={b.filename} className="btn btn-sm" disabled={busy} onClick={() => loadSample('benchmark', b.filename ?? b.name)}>
                    {(b.filename ?? b.name).replace('.csv', '')} ({b.size_mb} MB)
                  </button>
                ))}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-muted)', marginBottom: 6 }}>Universal Synthetic Generators (Instant)</div>
              <div className="row" style={{ gap: 6 }}>
                {(catalog.domains || []).slice(0, 10).map((d) => (
                  <button key={d.name} className="btn btn-sm" disabled={busy} onClick={() => loadSample('domain', d.name)}>
                    {d.display_name ?? d.name}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
      )}

      {/* AUTOMATIC POST-INGESTION DEFECT & STATISTICAL PRE-SCREEN AUDIT */}
      {sectionMode === 'ingestion' && ws.has_dataset && ia && (
        <div className="card" style={{ borderColor: ia.defect_counts.critical > 0 ? 'var(--red)' : ia.defect_counts.high > 0 ? 'var(--amber)' : 'var(--green)' }}>
          <div className="card-title">
            <span>Automatic Post-Ingestion Defect Audit &amp; Statistical Pre-Screening — {ia.overall_health}</span>
            <div className="row" style={{ gap: 6 }}>
              <span className={`badge ${ia.quality_score >= 85 ? 'badge-green' : ia.quality_score >= 65 ? 'badge-amber' : 'badge-red'}`}>
                Score {ia.quality_score}/100 (Grade {ia.quality_grade})
              </span>
            </div>
          </div>

          <div className="kpi-grid">
            <div className="kpi-card"><div className="kpi-label">Critical Defects</div><div className="kpi-value" style={{ color: ia.defect_counts.critical > 0 ? 'var(--red)' : 'var(--green)' }}>{ia.defect_counts.critical}</div></div>
            <div className="kpi-card"><div className="kpi-label">High Severity</div><div className="kpi-value" style={{ color: ia.defect_counts.high > 0 ? 'var(--amber)' : 'var(--green)' }}>{ia.defect_counts.high}</div></div>
            <div className="kpi-card"><div className="kpi-label">Medium / Low</div><div className="kpi-value">{ia.defect_counts.medium + ia.defect_counts.low}</div></div>
            <div className="kpi-card"><div className="kpi-label">Exact Duplicates</div><div className="kpi-value">{ia.duplicate_rows}</div></div>
            <div className="kpi-card"><div className="kpi-label">Constant Cols</div><div className="kpi-value">{ia.constant_columns.length}</div></div>
            <div className="kpi-card"><div className="kpi-label">Collinear / Leak Alerts</div><div className="kpi-value">{ia.correlation_alerts.length}</div></div>
          </div>

          <ul style={{ margin: '0 0 12px', paddingLeft: 18, fontSize: 12.5, color: 'var(--text-muted)' }}>
            {(ia.summary_bullets || []).map((b, i) => <li key={i}>{b}</li>)}
          </ul>

          {((ia.dimensions && Object.values(ia.dimensions).some((v) => typeof v === 'number' && Number.isFinite(v))) ||
            (ia.outlier_summary?.length ?? 0) > 0) && (
            <div className="grid-2" style={{ marginBottom: 14 }}>
              {ia.dimensions && Object.values(ia.dimensions).some((v) => typeof v === 'number' && Number.isFinite(v)) && (
                <BarChartSVG
                  title="6D Intake Quality & Readiness Profile (%)"
                  horizontal
                  items={Object.entries(ia.dimensions).flatMap(([dim, score]) =>
                    typeof score === 'number' && Number.isFinite(score)
                      ? [{ label: dim, value: score, color: score >= 85 ? '#34d399' : score >= 65 ? '#fbbf24' : '#f87171' }]
                      : [],
                  )}
                />
              )}
              {(ia.outlier_summary?.length ?? 0) > 0 && (
                <BarChartSVG
                  title="Column IQR Outlier Rate (%)"
                  items={(ia.outlier_summary || []).slice(0, 14).map((row) => ({
                    label: String(row.column ?? ''),
                    value: Number(row.iqr_pct ?? 0) || 0,
                    color: Number(row.iqr_pct ?? 0) > 5 ? '#f87171' : '#38bdf8',
                  }))}
                  height={185}
                />
              )}
            </div>
          )}

          {ia.defects?.length > 0 ? (
            <div style={{ marginBottom: 14 }}>
              <SimpleTable title="Detected Dataset Defects & Recommended Remediations" rows={ia.defects} maxRows={100} />
            </div>
          ) : (
            <div className="alert alert-success">Zero structural, format, missingness, duplicate, or statistical defects detected at intake!</div>
          )}

          {ia.statistical_pretests?.length > 0 && (
            <div className="grid-2">
              <SimpleTable title="Automatic Normality & Distribution Pre-Tests" rows={ia.statistical_pretests} maxRows={30} />
              <SimpleTable title="Outlier & Multicollinearity Screening" rows={ia.correlation_alerts.length > 0 ? ia.correlation_alerts : ia.outlier_summary} maxRows={30} />
            </div>
          )}
        </div>
      )}

      {/* RESTORED & EXPANDED DATA CLEANING STUDIO (Auto / Complete / Custom) */}
      {sectionMode === 'cleaning' && ws.has_dataset && (
        <div className="card">
          <div className="card-title">
            <span>Data Cleaning &amp; Remediation Studio (Restored + Expanded)</span>
            <ModeBar mode={cleanMode} onChange={setCleanMode} />
          </div>

          {cleanMode === 'auto' && (
            <p style={{ margin: '0 0 12px', fontSize: 12.5, color: 'var(--text-muted)' }}>
              <strong>Auto Mode</strong> configures every cleaning rule dynamically from your Automatic Intake Defect Audit (token standardization, string/HTML hygiene, type inference, exact duplicates, constant columns, median/mode imputation, and IQR winsorization where heavy outliers exist).
            </p>
          )}
          {cleanMode === 'complete' && (
            <p style={{ margin: '0 0 12px', fontSize: 12.5, color: 'var(--text-muted)' }}>
              <strong>Complete (Select All)</strong> standardizes missing tokens, removes whitespace and HTML, infers types, drops high-missing and constant columns, adds missingness indicators, removes exact duplicates, imputes missing values, and flags IQR outliers.
            </p>
          )}

          {cleanMode === 'custom' && (
            <div style={{ marginBottom: 12 }}>
              <div className="grid-4" style={{ marginBottom: 10 }}>
                <label className="field">Numeric Imputation
                  <select value={imputeNum} onChange={(e) => setImputeNum(e.target.value)}>
                    <option value="median">Median (Robust)</option>
                    <option value="mean">Mean</option>
                    <option value="knn">KNN Imputer</option>
                    <option value="iterative">Iterative MICE</option>
                    <option value="ffill">Forward Fill (Time-Series)</option>
                    <option value="bfill">Backward Fill</option>
                    <option value="interpolate">Linear Interpolation</option>
                    <option value="constant">Constant (0)</option>
                    <option value="none">None</option>
                  </select>
                </label>
                <label className="field">Categorical Imputation
                  <select value={imputeCat} onChange={(e) => setImputeCat(e.target.value)}>
                    <option value="most_frequent">Most Frequent (Mode)</option>
                    <option value="constant">Constant ("MISSING")</option>
                    <option value="none">None</option>
                  </select>
                </label>
                <label className="field">Outlier Method
                  <select value={outlierMethod} onChange={(e) => setOutlierMethod(e.target.value)}>
                    <option value="iqr">IQR Fence</option>
                    <option value="zscore">Z-Score</option>
                    <option value="modified_zscore">Modified Z-Score (MAD)</option>
                    <option value="isolation_forest">Isolation Forest</option>
                    <option value="lof">Local Outlier Factor (LOF)</option>
                    <option value="none">Disabled</option>
                  </select>
                </label>
                <label className="field">Outlier Action
                  <select value={outlierAction} onChange={(e) => setOutlierAction(e.target.value)}>
                    <option value="clip">Winsorize / Clip</option>
                    <option value="remove">Remove Outlier Rows</option>
                    <option value="nan">Replace with NaN</option>
                    <option value="flag">Flag Only</option>
                    <option value="none">Leave Untouched</option>
                  </select>
                </label>
              </div>
              <div className="grid-4" style={{ marginBottom: 10 }}>
                <label className="field">Outlier Threshold
                  <input type="number" step={0.1} min={0.5} max={6} value={outlierThresh} onChange={(e) => setOutlierThresh(Number(e.target.value))} />
                </label>
                <label className="field">Drop Cols Missing &gt; Ratio
                  <input type="number" step={0.05} min={0.3} max={1.0} value={missingColThresh} onChange={(e) => setMissingColThresh(Number(e.target.value))} />
                </label>
                <div className="row" style={{ alignItems: 'center', gap: 10, gridColumn: 'span 2', flexWrap: 'wrap' }}>
                  <label className="row" style={{ gap: 5 }}><input type="checkbox" checked={dropDups} onChange={(e) => setDropDups(e.target.checked)} /> Drop Exact Duplicates</label>
                  <label className="row" style={{ gap: 5 }}><input type="checkbox" checked={stdMissingTokens} onChange={(e) => setStdMissingTokens(e.target.checked)} /> Standardize Missing Tokens</label>
                  <label className="row" style={{ gap: 5 }}><input type="checkbox" checked={cleanStrings} onChange={(e) => setCleanStrings(e.target.checked)} /> Trim/Clean Strings</label>
                  <label className="row" style={{ gap: 5 }}><input type="checkbox" checked={stripHtml} onChange={(e) => setStripHtml(e.target.checked)} /> Strip HTML</label>
                  <label className="row" style={{ gap: 5 }}><input type="checkbox" checked={lowercaseStr} onChange={(e) => setLowercaseStr(e.target.checked)} /> Lowercase Text</label>
                  <label className="row" style={{ gap: 5 }}><input type="checkbox" checked={inferTypes} onChange={(e) => setInferTypes(e.target.checked)} /> Auto Infer Types</label>
                  <label className="row" style={{ gap: 5 }}><input type="checkbox" checked={dropHighMissing} onChange={(e) => setDropHighMissing(e.target.checked)} /> Drop High-Missing Cols</label>
                  <label className="row" style={{ gap: 5 }}><input type="checkbox" checked={dropConstCols} onChange={(e) => setDropConstCols(e.target.checked)} /> Drop Constant Cols</label>
                  <label className="row" style={{ gap: 5 }}><input type="checkbox" checked={addMissingFlags} onChange={(e) => setAddMissingFlags(e.target.checked)} /> Add Missingness Indicators</label>
                  <label className="row" style={{ gap: 5 }}><input type="checkbox" checked={stdColNames} onChange={(e) => setStdColNames(e.target.checked)} /> Standardize Column Names</label>
                </div>
              </div>
            </div>
          )}

          <div className="row">
            <button className="btn" disabled={busy} onClick={() => runClean(true)}>Dry-Run Preview</button>
            <button className="btn btn-primary" disabled={busy} onClick={() => runClean(false)}>
              {busy ? <span className="spinner" /> : null} Apply &amp; Commit Cleaning ({cleanMode.toUpperCase()})
            </button>
            <button className="btn" disabled={busy} onClick={() => runProfile()}>Run Deep 6D Quality Profile</button>
            <button className="btn" onClick={() => apiGet<DeletedLogResponse>('/api/workbench/s01/deleted-log').then(setDeletedLog).catch(() => {})}>
              Deleted Records Audit ({ws.total_deleted_rows})
            </button>
          </div>
          <div className="row" style={{ marginTop: 12 }}>
            <label className="field">Trusted reference CSV
              <input type="file" accept=".csv" onChange={(event) => setReferenceFile(event.target.files?.[0] ?? null)} />
            </label>
            <label className="field">Reference record key
              <select value={referenceKey} onChange={(event) => setReferenceKey(event.target.value)}>
                <option value="">Select a column</option>
                {ws.columns.map((column) => <option key={column} value={column}>{column}</option>)}
              </select>
            </label>
            <button className="btn" disabled={busy || !referenceFile || !ws.columns.includes(referenceKey)} onClick={() => runProfile(true)}>
              Compare with Trusted Reference
            </button>
          </div>

          {cleanPreview?.cleaning_report && (
            <div className="kpi-grid" style={{ marginTop: 12 }}>
              <div className="kpi-card"><div className="kpi-label">Mode</div><div className="kpi-value" style={{ fontSize: 16 }}>{String(cleanPreview.cleaning_report.mode_used || cleanMode).toUpperCase()}</div></div>
              <div className="kpi-card"><div className="kpi-label">Rows</div><div className="kpi-value" style={{ fontSize: 16 }}>{cleanPreview.cleaning_report.rows_before} → {cleanPreview.cleaning_report.rows_after}</div></div>
              <div className="kpi-card"><div className="kpi-label">Columns</div><div className="kpi-value" style={{ fontSize: 16 }}>{cleanPreview.cleaning_report.cols_before} → {cleanPreview.cleaning_report.cols_after}</div></div>
              {Array.isArray((cleanPreview.cleaning_report as Record<string, unknown>).steps_applied) && (
                <div style={{ gridColumn: '1 / -1', marginTop: 8, padding: '10px 12px', borderRadius: 6, background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.3)', fontSize: 12.5 }}>
                  <strong style={{ color: '#34d399' }}>Ordered Cleaning Pipeline Executed (Step-by-Step):</strong>
                  <ol style={{ margin: '6px 0 0 18px', padding: 0 }}>
                    {((cleanPreview.cleaning_report as Record<string, unknown>).steps_applied as string[]).map((step, idx) => (
                      <li key={idx} style={{ marginBottom: 2 }}>{step}</li>
                    ))}
                  </ol>
                </div>
              )}
              {Array.isArray((cleanPreview.cleaning_report as Record<string, unknown>).deleted_records) &&
                ((cleanPreview.cleaning_report as Record<string, unknown>).deleted_records as Array<Record<string, unknown>>).length > 0 && (
                <div style={{ gridColumn: '1 / -1', marginTop: 8, padding: '10px 12px', borderRadius: 6, background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)', fontSize: 12.5 }}>
                  <strong style={{ color: '#f87171' }}>Deleted Rows &amp; Removal Reasons (Why Each Row Was Removed):</strong>
                  <div className="table-wrap" style={{ marginTop: 6, maxHeight: 220, overflowY: 'auto' }}>
                    <table>
                      <thead>
                        <tr>
                          <th>Row Index</th>
                          <th>Stage</th>
                          <th>Reason for Removal</th>
                        </tr>
                      </thead>
                      <tbody>
                        {((cleanPreview.cleaning_report as Record<string, unknown>).deleted_records as Array<Record<string, unknown>>).slice(0, 50).map((rec, idx) => (
                          <tr key={idx}>
                            <td>{String(rec.row_index ?? rec.index ?? idx)}</td>
                            <td>{String(rec.stage ?? 'cleaning')}</td>
                            <td>{String(rec.reason ?? 'Policy threshold exceeded')}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              <div className="kpi-card"><div className="kpi-label">Missing Cells</div><div className="kpi-value" style={{ fontSize: 16 }}>{cleanPreview.cleaning_report.missing_before} → {cleanPreview.cleaning_report.missing_after}</div></div>
              <div style={{ gridColumn: '1 / -1', marginTop: 6 }}>
                <BarChartSVG
                  title="Cleaning Remediation Impact (Before vs. After)"
                  items={[
                    { label: 'Rows Before', value: Number(cleanPreview.cleaning_report.rows_before ?? 0), color: '#64748b' },
                    { label: 'Rows After', value: Number(cleanPreview.cleaning_report.rows_after ?? 0), color: '#34d399' },
                    { label: 'Missing Before', value: Number(cleanPreview.cleaning_report.missing_before ?? 0), color: '#f87171' },
                    { label: 'Missing After', value: Number(cleanPreview.cleaning_report.missing_after ?? 0), color: '#38bdf8' },
                  ]}
                  height={180}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {sectionMode === 'cleaning' && ws.has_dataset && !quality && !cleanPreview?.cleaning_report && ia?.dimensions && (
        <div className="card">
          <div className="card-title">
            <span>Baseline Intake Quality Dimensions (Pre-Cleaning Snapshot)</span>
            <span className="badge badge-accent">Grade {ia.quality_grade} · {ia.quality_score}/100</span>
          </div>
          <BarChartSVG
            title="Pre-Cleaning 6-Dimensional Quality Scores (%)"
            horizontal
            items={Object.entries(ia.dimensions).flatMap(([dim, score]) =>
              typeof score === 'number' && Number.isFinite(score)
                ? [{ label: dim, value: score, color: score >= 85 ? '#34d399' : score >= 65 ? '#fbbf24' : '#f87171' }]
                : [],
            )}
          />
        </div>
      )}

      {quality && (
        <div className="card">
          <div className="card-title">
            <span>6-Dimensional Enterprise Quality &amp; Format Audit</span>
            <span className="badge badge-green">Overall Score: {Number(quality.quality?.overall_score ?? 0).toFixed(1)} / 100</span>
          </div>
          {quality.quality?.reference_matched_rows != null && (
            <p>{quality.quality.reference_matched_rows} matched rows and {quality.quality.reference_comparisons ?? 0} comparable values using {quality.quality.reference_key_columns?.join(', ')}.</p>
          )}
          <div className="kpi-grid">
            {Object.entries(quality.quality?.dimensions || {}).map(([k, v]) => (
              <div key={k} className="kpi-card">
                <div className="kpi-label">{k}</div>
                <div className="kpi-value">{typeof v === 'number' ? `${v.toFixed(1)}%` : 'Unavailable'}</div>
              </div>
            ))}
          </div>
          {quality.quality?.dimensions && Object.keys(quality.quality.dimensions).length > 0 && (
            <div className="grid-2" style={{ marginBottom: 14 }}>
              <BarChartSVG
                title="6-Dimensional Data Quality Breakdown (%)"
                horizontal
                items={Object.entries(quality.quality.dimensions).flatMap(([k, v]) =>
                  typeof v === 'number' && Number.isFinite(v)
                    ? [{ label: k, value: v, color: v >= 85 ? '#34d399' : v >= 65 ? '#fbbf24' : '#f87171' }]
                    : [],
                )}
              />
              {(quality.quality.column_scores?.length ?? 0) > 0 && (
                <BarChartSVG
                  title="Column Completeness / Quality Profile (%)"
                  horizontal
                  items={(quality.quality.column_scores || []).slice(0, 14).flatMap((row) => {
                    const colName = String(row.column ?? row.name ?? '');
                    const rawScore = Number(row.completeness_pct ?? row.quality_score ?? row.completeness ?? NaN);
                    return colName && Number.isFinite(rawScore) ? [{ label: colName, value: rawScore, color: '#38bdf8' }] : [];
                  })}
                />
              )}
            </div>
          )}
          <SimpleTable title="Column-Level Quality Metrics" rows={quality.quality?.column_scores || []} />
        </div>
      )}

      {deletedLog && (
        <div className="card">
          <div className="card-title">Forensic Deleted Records Audit Log</div>
          <SimpleTable title="Deleted Data Audit Trail" rows={deletedLog.log || []} />
        </div>
      )}

      {ws.has_dataset && (
        <div className="card">
          <div className="card-title">
            <span>Server-Side Paginated Data Explorer</span>
            <div className="row">
              <button className={`btn btn-sm ${viewMode === 'active' ? 'btn-primary' : ''}`} onClick={() => setViewMode('active')}>Active ({ws.n_rows.toLocaleString()})</button>
              <button className={`btn btn-sm ${viewMode === 'raw' ? 'btn-primary' : ''}`} onClick={() => setViewMode('raw')}>Original Raw ({ws.raw_rows.toLocaleString()})</button>
            </div>
          </div>
          <PaginatedDataGrid refreshKey={gridVersion} viewMode={viewMode} />
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Studio 02 — Clustering (FIXED Row Deletion + Persistent Green LED),
//             Expanded Anomaly Screening & Restored Data Mining Lab
// ============================================================================
export const Studio02Clustering: React.FC<StudioProps> = ({ ws, onSync, bumpGrid, studioCache, setStudioResult, openHelp }) => {
  const [tab, setTab] = useState<'cluster' | 'anomaly' | 'mining'>('cluster');

  // Clustering state (results hydrated from persistent studioCache)
  const [clMode, setClMode] = useState<ExecutionMode>('auto');
  const [algo, setAlgo] = useState('kmeans');
  const [k, setK] = useState(4);
  const [scaler, setScaler] = useState('standard');
  const clRes = studioCache.s02_clustering ?? null;
  const [selectedDropClusters, setSelectedDropClusters] = useState<number[]>([]);
  const [attachAfterDrop, setAttachAfterDrop] = useState(false);

  // Anomaly state
  const [anMode, setAnMode] = useState<ExecutionMode>('auto');
  const [anomMethod, setAnomMethod] = useState('isolation_forest');
  const [contam, setContam] = useState(0.05);
  const anRes = studioCache.s02_anomaly ?? null;

  // Data Mining Lab state
  const [mineMode, setMineMode] = useState<ExecutionMode>('complete');
  const [selectedMiningTarget, setSelectedMiningTarget] = useState('');
  const miningTarget = ws.targets.length === 1 ? ws.targets[0] : ws.targets.includes(selectedMiningTarget) ? selectedMiningTarget : '';
  const [mineTasks, setMineTasks] = useState<string[]>(['association_rules', 'subgroups', 'manifold']);
  const [manifoldMethod, setManifoldMethod] = useState('pca');
  const [minSupport, setMinSupport] = useState(0.08);
  const [minConf, setMinConf] = useState(0.55);
  const [minLift, setMinLift] = useState(1.1);
  const mineRes = studioCache.s02_mining ?? null;
  const manifold = mineRes?.manifold;
  const explainedPct = typeof manifold?.total_explained_pct === 'number' && Number.isFinite(manifold.total_explained_pct)
    ? manifold.total_explained_pct
    : manifold?.explained_variance_ratio?.every(Number.isFinite)
      ? manifold.explained_variance_ratio.reduce((sum, value) => sum + value, 0) * 100
      : undefined;

  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const runClustering = async () => {
    setBusy(true); setErr(null); setMsg(null);
    try {
      const res = await apiPost<ClusteringResponse>('/api/workbench/s02/clustering', {
        mode: clMode,
        algorithm: algo,
        n_clusters: k,
        scaler,
        run_auto_k: true,
      });
      onSync(res.state);
      setStudioResult('s02_clustering', res);
      setSelectedDropClusters([]);
      setMsg(`Clustering (${clMode.toUpperCase()}) preview ready: winner=${res.algorithm}, found ${res.n_clusters} clusters.`);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  const toggleDropCluster = (cid: number) => {
    setSelectedDropClusters((prev) =>
      prev.includes(cid) ? prev.filter((x) => x !== cid) : [...prev, cid],
    );
  };

  const commitClustering = async (mode: 'attach' | 'remove_clusters') => {
    setBusy(true); setErr(null); setMsg(null);
    try {
      const res = await apiPost<ClusteringCommitResponse>('/api/workbench/s02/clustering/commit', {
        mode,
        remove_cluster_ids: mode === 'remove_clusters' ? selectedDropClusters : [],
        attach_cluster_id: attachAfterDrop,
      });
      onSync(res.state);
      if (res.result) {
        setStudioResult('s02_clustering', { ...res.result, state: res.state });
      } else if (res.result === null) {
        setStudioResult('s02_clustering', null);
      } else if (clRes && (!Array.isArray(res.state.cached_result_keys) || res.state.cached_result_keys.includes('s02_clustering'))) {
        // Compatibility with older servers that only returned the commit summary.
        setStudioResult('s02_clustering', {
          ...clRes,
          cluster_choices: res.remaining_cluster_choices ?? clRes.cluster_choices,
          committed_action: res.action,
        });
      }
      bumpGrid();
      setSelectedDropClusters([]);
      setMsg(`Committed: ${res.action}. Active dataset now has ${Number(res.rows_after ?? 0).toLocaleString()} rows (Studio 02 green status locked ON).`);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  const runAnomaly = async () => {
    setBusy(true); setErr(null); setMsg(null);
    try {
      const res = await apiPost<AnomalyResponse>('/api/workbench/s02/anomaly', {
        mode: anMode,
        method: anomMethod,
        contamination: contam,
      });
      onSync(res.state);
      setStudioResult('s02_anomaly', res);
      setMsg(`Anomaly Screening (${anMode.toUpperCase()}) flagged ${res.n_anomalies} rows (${res.anomaly_rate_pct}%).`);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  const commitAnomaly = async (mode: 'attach' | 'remove_anomalies') => {
    setBusy(true); setErr(null); setMsg(null);
    try {
      const res = await apiPost<AnomalyCommitResponse>('/api/workbench/s02/anomaly/commit', { mode });
      onSync(res.state);
      if (res.result) {
        setStudioResult('s02_anomaly', { ...res.result, state: res.state });
      } else if (res.result === null) {
        setStudioResult('s02_anomaly', null);
      } else if (anRes && (!Array.isArray(res.state.cached_result_keys) || res.state.cached_result_keys.includes('s02_anomaly'))) {
        setStudioResult('s02_anomaly', { ...anRes, committed_action: res.action, committed_mode: mode });
      }
      bumpGrid();
      setMsg(`Committed: ${res.action}. Active dataset now has ${Number(res.rows_after ?? 0).toLocaleString()} rows.`);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  const toggleMineTask = (t: string) => {
    setMineTasks((prev) => (prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]));
  };

  // Like the other dataset-dependent results, this must follow server cache
  // invalidation and workspace restore rather than living in component state.
  const advMineRes = studioCache.s02_advanced_mining ?? null;

  const runAdvancedMining = async () => {
    setBusy(true);
    setErr(null);
    try {
      const url = miningTarget ? `/api/workbench/s02/advanced_mining?target_col=${encodeURIComponent(miningTarget)}` : '/api/workbench/s02/advanced_mining';
      const { state, ...result } = await apiPost<AdvancedMiningResponse>(url, {});
      onSync(state);
      setStudioResult('s02_advanced_mining', result);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : 'Advanced mining failed');
    } finally {
      setBusy(false);
    }
  };

  const runMining = async () => {
    setBusy(true); setErr(null); setMsg(null);
    try {
      const res = await apiPost<MiningResponse>('/api/workbench/s02/mining', {
        mode: mineMode,
        tasks: mineTasks,
        target: miningTarget || null,
        manifold_method: manifoldMethod,
        min_support: minSupport,
        min_confidence: minConf,
        min_lift: minLift,
      });
      onSync(res.state);
      setStudioResult('s02_mining', res);
      setMsg(`Data Mining completed (${(res.tasks_executed || []).join(', ')}).`);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  const selectedDropRows = (clRes?.cluster_choices || [])
    .filter((c) => selectedDropClusters.includes(c.cluster_id))
    .reduce((acc, c) => acc + Number(c.count || 0), 0);

  return (
    <div>
      <div className="studio-hero">
        <div>
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <h1>02 · Clustering, Anomaly Screening &amp; Data Mining Lab</h1>
            <StudioHelpButton section="studio-02" onOpenHelp={openHelp} title="Open Chapter 02: Clustering, Anomaly Detectors, Apriori & Manifold Guide" />
          </div>
          <p>Multi-algorithm clustering benchmark, verified row-level cluster &amp; anomaly deletion with persistent stage tracking, plus Association Rule Mining, Subgroup Discovery &amp; 6-Method Manifold Projection.</p>
        </div>
        <div className="row">
          <ProvenancePill ws={ws} />
          {ws.s02_cluster_committed && ws.s02_cluster_summary && (
            <span className="badge badge-green">✓ Studio 02 Committed ({String(ws.s02_cluster_summary.action ?? '')})</span>
          )}
        </div>
      </div>

      {msg && <div className="alert alert-success">{msg}</div>}
      {err && <div className="alert alert-error">{err}</div>}

      <LiveChartGenerator studioId="02" ws={ws} studioCache={studioCache} />

      <div className="tabs">
        <button className={`tab-btn ${tab === 'cluster' ? 'active' : ''}`} onClick={() => setTab('cluster')}>1. Clustering &amp; Cluster Row Removal</button>
        <button className={`tab-btn ${tab === 'anomaly' ? 'active' : ''}`} onClick={() => setTab('anomaly')}>2. Anomaly Detection Screening</button>
        <button className={`tab-btn ${tab === 'mining' ? 'active' : ''}`} onClick={() => setTab('mining')}>3. Data Mining Lab (Rules · Subgroups · Manifold)</button>
      </div>

      {tab === 'cluster' && (
        <>
          <div className="card">
            <div className="card-title">
              <span>Clustering Engine Configuration</span>
              <ModeBar mode={clMode} onChange={setClMode} />
            </div>
            <div className="row" style={{ gap: 12 }}>
              {clMode === 'custom' && (
                <label className="field">Algorithm
                  <select value={algo} onChange={(e) => setAlgo(e.target.value)}>
                    <option value="kmeans">K-Means</option>
                    <option value="gmm">Gaussian Mixture (GMM)</option>
                    <option value="dbscan">DBSCAN (Density + Noise)</option>
                    <option value="agglomerative">Agglomerative Hierarchical</option>
                    <option value="birch">BIRCH</option>
                    <option value="spectral">Spectral Clustering</option>
                  </select>
                </label>
              )}
              <label className="field">Target Clusters (K)
                <input type="number" min={2} max={15} value={k} onChange={(e) => setK(Number(e.target.value))} style={{ width: 84 }} />
              </label>
              <label className="field">Feature Scaler
                <select value={scaler} onChange={(e) => setScaler(e.target.value)}>
                  <option value="standard">StandardScaler</option>
                  <option value="robust">RobustScaler (Outlier-Safe)</option>
                  <option value="minmax">MinMaxScaler</option>
                </select>
              </label>
              <button className="btn btn-primary" disabled={busy || !ws.has_dataset} onClick={runClustering}>
                {busy ? <span className="spinner" /> : null} Run Clustering Preview ({clMode.toUpperCase()})
              </button>
            </div>
          </div>

          {clRes && (
            <div className="card">
              <div className="card-title">
                <span>Clustering Preview — Winner: {clRes.algorithm} ({clRes.n_clusters} clusters, {clRes.noise_count} noise pts)</span>
                <span className="badge badge-accent">Silhouette = {Number(clRes.metrics?.silhouette ?? 0).toFixed(4)}</span>
              </div>
              {clRes.committed_action && (
                <div className="alert alert-warn" role="status">
                  This chart, its counts and metrics show the original pre-commit preview. Cluster choices below reflect remaining rows.
                </div>
              )}

              {(clRes.algorithm_benchmark?.length ?? 0) > 1 && (
                <div style={{ marginBottom: 14 }}>
                  <SimpleTable title="Multi-Algorithm Clustering Benchmark (Ranked by Silhouette)" rows={clRes.algorithm_benchmark || []} />
                </div>
              )}

              {clRes.scatter_error ? (
                <div className="alert alert-error">{clRes.scatter_error}</div>
              ) : (
                <CanvasScatter title={`Cluster Map (${clRes.algorithm})`} data={clRes.scatter ?? EMPTY_SCATTER} xLabel="PCA 1" yLabel="PCA 2" />
              )}

              {/* FIXED INTERACTIVE CLUSTER SELECTION & REMOVAL PANEL */}
              <div style={{ marginTop: 14, padding: 14, background: 'var(--bg)', borderRadius: 10, border: '1px solid var(--border)' }}>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>Select Corrupt / Noise Clusters to Permanently Remove (or Attach cluster_id)</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>
                  Click one or more cluster tiles below to mark their rows for deletion. All deleted rows are logged into the Forensic Audit Trail and can be undone anytime.
                </div>
                <div className="cluster-choice-grid">
                  {(clRes.cluster_choices || []).map((c) => {
                    const checked = selectedDropClusters.includes(c.cluster_id);
                    return (
                      <button
                        key={c.cluster_id}
                        type="button"
                        data-cluster-id={c.cluster_id}
                        className={`cluster-choice-card ${checked ? 'selected' : ''}`}
                        onClick={() => toggleDropCluster(c.cluster_id)}
                      >
                        <input type="checkbox" readOnly checked={checked} />
                        <div>
                          <div style={{ fontWeight: 700, fontSize: 12.5 }}>{c.label}</div>
                          <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>{Number(c.count).toLocaleString()} rows ({c.pct}%)</div>
                        </div>
                      </button>
                    );
                  })}
                </div>
                <div className="row" style={{ gap: 10 }}>
                  <button
                    className="btn btn-danger"
                    disabled={busy || selectedDropClusters.length === 0}
                    onClick={() => commitClustering('remove_clusters')}
                  >
                    Remove Selected Clusters ({selectedDropRows.toLocaleString()} rows)
                  </button>
                  <button className="btn btn-success" disabled={busy} onClick={() => commitClustering('attach')}>
                    Attach cluster_id Column to Dataset
                  </button>
                  <label className="row" style={{ gap: 5, fontSize: 12 }}>
                    <input type="checkbox" checked={attachAfterDrop} onChange={(e) => setAttachAfterDrop(e.target.checked)} />
                    Keep cluster_id on remaining rows after deletion
                  </label>
                </div>
              </div>

              {(clRes.cluster_profiles?.length ?? 0) > 0 && (
                <div style={{ marginTop: 14 }}>
                  <SimpleTable title="Cluster Statistical Profiles" rows={clRes.cluster_profiles || []} />
                </div>
              )}
            </div>
          )}
        </>
      )}

      {tab === 'anomaly' && (
        <>
          <div className="card">
            <div className="card-title">
              <span>Anomaly Detection Configuration</span>
              <ModeBar mode={anMode} onChange={setAnMode} />
            </div>
            <div className="row" style={{ gap: 12 }}>
              {anMode === 'custom' && (
                <label className="field">Detector
                  <select value={anomMethod} onChange={(e) => setAnomMethod(e.target.value)}>
                    <option value="isolation_forest">Isolation Forest</option>
                    <option value="lof">Local Outlier Factor (LOF)</option>
                    <option value="pca_recon">PCA Reconstruction Error</option>
                    <option value="mad">Robust MAD</option>
                    <option value="zscore">Multivariate Z-Score</option>
                    <option value="ensemble">Multi-Detector Ensemble Consensus</option>
                  </select>
                </label>
              )}
              <label className="field">Contamination Rate
                <input type="number" step={0.01} min={0.01} max={0.35} value={contam} onChange={(e) => setContam(Number(e.target.value))} style={{ width: 95 }} />
              </label>
              <button className="btn btn-primary" disabled={busy || !ws.has_dataset} onClick={runAnomaly}>
                {busy ? <span className="spinner" /> : null} Screen Anomalies ({anMode.toUpperCase()})
              </button>
            </div>
          </div>

          {anRes && (
            <div className="card">
              <div className="card-title">
                <span>Detected Anomalies: {anRes.n_anomalies} ({anRes.anomaly_rate_pct}%) · Method: {anRes.method}</span>
                <div className="row">
                  <button className="btn btn-success btn-sm" disabled={busy || anRes.committed_mode === 'remove_anomalies'} onClick={() => commitAnomaly('attach')}>Attach is_anomaly &amp; Score</button>
                  <button className="btn btn-danger btn-sm" disabled={busy || anRes.committed_mode === 'remove_anomalies'} onClick={() => commitAnomaly('remove_anomalies')}>Remove {anRes.n_anomalies} Anomalies</button>
                </div>
              </div>
              {anRes.committed_action && (
                <div className="alert alert-warn" role="status">
                  This chart and its anomaly counts show the original pre-commit preview. Run screening again for the current dataset.
                </div>
              )}
              {(anRes.detector_comparison?.length ?? 0) > 1 && (
                <div style={{ marginBottom: 12 }}>
                  <SimpleTable title="Multi-Detector Anomaly Comparison" rows={anRes.detector_comparison || []} />
                </div>
              )}
              <CanvasScatter title="Anomaly Screening Scatter" data={anRes.scatter ?? EMPTY_SCATTER} xLabel={anRes.x_axis} yLabel={anRes.y_axis} />
              <div style={{ marginTop: 12 }}>
                <SimpleTable title="Highest-Scoring Anomalous Records" rows={anRes.top_anomalies || []} maxRows={30} />
              </div>
            </div>
          )}
        </>
      )}

      {tab === 'mining' && (
        <>
          <div className="card">
            <div className="card-title">
              <span>Restored Data Mining Lab — Association Rules, Subgroups &amp; Manifold Projection</span>
              <ModeBar mode={mineMode} onChange={setMineMode} />
            </div>
            {ws.targets.length > 1 && (
              <label className="field">Target for this mining run
                <select value={miningTarget} onChange={(event) => setSelectedMiningTarget(event.target.value)} disabled={busy}>
                  <option value="">Choose one target</option>
                  {ws.targets.map((target) => <option value={target} key={target}>{target}</option>)}
                </select>
              </label>
            )}
            {mineMode === 'custom' && (
              <div className="row" style={{ gap: 14, marginBottom: 12 }}>
                <label className="row" style={{ gap: 5 }}><input type="checkbox" checked={mineTasks.includes('association_rules')} onChange={() => toggleMineTask('association_rules')} /> Apriori Association Rules</label>
                <label className="row" style={{ gap: 5 }}><input type="checkbox" checked={mineTasks.includes('subgroups')} onChange={() => toggleMineTask('subgroups')} /> Subgroup Discovery (WRAcc)</label>
                <label className="row" style={{ gap: 5 }}><input type="checkbox" checked={mineTasks.includes('manifold')} onChange={() => toggleMineTask('manifold')} /> Manifold Projection</label>
              </div>
            )}
            <div className="grid-4" style={{ marginBottom: 12 }}>
              <label className="field">Manifold Method
                <select value={manifoldMethod} onChange={(e) => setManifoldMethod(e.target.value)}>
                  <option value="pca">PCA (Linear)</option>
                  <option value="svd">Truncated SVD</option>
                  <option value="tsne">t-SNE (Nonlinear)</option>
                  <option value="isomap">Isomap (Geodesic)</option>
                  <option value="mds">Multidimensional Scaling (MDS)</option>
                  <option value="spectral">Spectral Embedding</option>
                </select>
              </label>
              <label className="field">Min Support
                <input type="number" step={0.02} min={0.01} max={0.8} value={minSupport} onChange={(e) => setMinSupport(Number(e.target.value))} />
              </label>
              <label className="field">Min Confidence
                <input type="number" step={0.05} min={0.1} max={1.0} value={minConf} onChange={(e) => setMinConf(Number(e.target.value))} />
              </label>
              <label className="field">Min Lift
                <input type="number" step={0.1} min={0.5} max={10.0} value={minLift} onChange={(e) => setMinLift(Number(e.target.value))} />
              </label>
            </div>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <button className="btn btn-primary" disabled={busy || !ws.has_dataset || (ws.targets.length > 1 && !miningTarget && (mineMode !== 'custom' || mineTasks.includes('subgroups')))} onClick={runMining}>
                {busy ? <span className="spinner" /> : null} Execute Data Mining Suite ({mineMode.toUpperCase()})
              </button>
              <button className="btn btn-secondary" disabled={busy || !ws.has_dataset || (ws.targets.length > 1 && !miningTarget)} onClick={runAdvancedMining}>
                Run Advanced Mining (Beam + ARI + dCor)
              </button>
            </div>
            {advMineRes && (
              <div className="card" style={{ marginTop: 10 }}>
                <div className="card-title"><span>Advanced Mining Diagnostics (Beam Search, Bootstrap ARI, dCor)</span></div>
                <pre style={{ fontSize: 12, maxHeight: 220, overflow: 'auto', margin: 0 }}>{JSON.stringify(advMineRes, null, 2)}</pre>
              </div>
            )}
          </div>

          {mineRes && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {mineRes.target && <p>Mining target: {mineRes.target}</p>}
              {manifold?.error && <div className="alert alert-error">Manifold projection failed: {manifold.error}</div>}
              {manifold?.scatter && (
                <div className="card">
                  <div className="card-title">
                    <span>Manifold Embedding ({String(manifold.method || 'Projection').toUpperCase()})</span>
                    {explainedPct !== undefined && (
                      <span className="badge badge-accent">Explained Variance: {explainedPct.toFixed(1)}%</span>
                    )}
                  </div>
                  <CanvasScatter title={`2D ${String(manifold.method || 'Projection').toUpperCase()} Manifold Projection`} data={manifold.scatter} xLabel={manifold.x_label || 'Dim 1'} yLabel={manifold.y_label || 'Dim 2'} />
                </div>
              )}
              {mineRes.association_rules && (
                <div className="card">
                  <div className="card-title">Apriori Association Rules ({mineRes.association_rules.rules?.length || 0} rules discovered)</div>
                  <SimpleTable title="Association Rules (Antecedent → Consequent)" rows={mineRes.association_rules.rules || []} />
                </div>
              )}
              {mineRes.subgroups && (
                <div className="card">
                  <div className="card-title">High-Contrast Subgroup Discovery ({mineRes.subgroups.subgroups?.length || 0} subgroups)</div>
                  <SimpleTable title="Discovered Subgroups (Ranked by WRAcc / Lift)" rows={mineRes.subgroups.subgroups || []} />
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
};

// ============================================================================
// Studio 03 — Universal Diagnostics, Ranked Baseline Equations & Risk
// ============================================================================
export const Studio03Diagnostics: React.FC<StudioProps> = ({ ws, onSync, studioCache, setStudioResult, openHelp }) => {
  const [mode, setMode] = useState<ExecutionMode>('complete');
  const [alpha, setAlpha] = useState(0.05);
  const [confLevel, setConfLevel] = useState(0.95);
  const [fdr, setFdr] = useState('fdr_bh');
  const [paramMode, setParamMode] = useState('auto');
  const [normMethod, setNormMethod] = useState('auto');
  const [corrMethod, setCorrMethod] = useState('auto');
  const [vifThresh, setVifThresh] = useState(5.0);
  const [riskAlpha, setRiskAlpha] = useState(0.05);
  const [selectedMods, setSelectedMods] = useState<string[]>([
    'statistical_tests', 'baselines', 'risk_analysis', 'multicollinearity', 'outliers',
  ]);
  const report = studioCache.s03_diagnostics?.report ?? null;
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const ALL_MODS = [
    ['statistical_tests', 'Normality & Hypothesis Tests'],
    ['baselines', 'Strong Baselines & Equations'],
    ['risk_analysis', 'VaR / CVaR / Tail Risk'],
    ['multicollinearity', 'VIF Multicollinearity'],
    ['outliers', 'Multi-Method Outliers'],
    ['feature_screening', 'Target-Feature Screening'],
  ];

  const toggleMod = (m: string) =>
    setSelectedMods((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]));

  const run = async () => {
    setBusy(true); setErr(null);
    try {
      const apiMode = mode === 'complete' ? 'all' : mode;
      const res = await apiPost<DiagnosticsResponse>('/api/workbench/s03/run', {
        mode: apiMode,
        alpha,
        confidence_level: confLevel,
        fdr_method: fdr,
        parametric_mode: paramMode,
        normality_method: normMethod,
        correlation_method: corrMethod,
        vif_threshold: vifThresh,
        risk_tail_alpha: riskAlpha,
        selected_modules: mode === 'custom' ? selectedMods : [],
      });
      onSync(res.state);
      setStudioResult('s03_diagnostics', res);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  return (
    <div>
      <div className="studio-hero">
        <div>
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <h1>03 · Universal Diagnostics, Ranked Baseline Equations &amp; Risk</h1>
            <StudioHelpButton section="studio-03" onOpenHelp={openHelp} title="Open Chapter 03: VIF, VaR/CVaR, Baseline Equations & FDR Corrections" />
          </div>
          <p>Exact statistical criteria control, modern symbolic mathematical baseline equations (Parametric + Non-Parametric Analytical: B-Spline GAM, Nadaraya-Watson Kernel, Weighted k-NN) ranked best-to-worst with explicit acceptance/rejection rationale, and VaR/CVaR tail-risk quantification.</p>
        </div>
        <ProvenancePill ws={ws} />
      </div>
      {err && <div className="alert alert-error">{err}</div>}
      <LiveChartGenerator studioId="03" ws={ws} studioCache={studioCache} />
      <div className="card">
        <div className="card-title">
          <span className="row" style={{ gap: 6, alignItems: 'center' }}>
            Statistical Criteria &amp; Diagnostic Execution Mode
            <StudioHelpButton section="thresholds" onOpenHelp={openHelp} label="Thresholds" title="Open Master Numeric Threshold Compendium" />
          </span>
          <ModeBar mode={mode} onChange={setMode} />
        </div>
        <div className="grid-4" style={{ marginBottom: 12 }}>
          <label className="field">Significance (α)
            <input type="number" step={0.01} min={0.001} max={0.2} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} />
          </label>
          <label className="field">Confidence Level
            <input type="number" step={0.01} min={0.8} max={0.999} value={confLevel} onChange={(e) => setConfLevel(Number(e.target.value))} />
          </label>
          <label className="field">FDR Correction
            <select value={fdr} onChange={(e) => setFdr(e.target.value)}>
              <option value="fdr_bh">Benjamini-Hochberg (FDR)</option>
              <option value="bonferroni">Bonferroni</option>
              <option value="holm">Holm-Bonferroni</option>
              <option value="none">None</option>
            </select>
          </label>
          <label className="field">Tail Risk α (VaR / CVaR)
            <input type="number" step={0.01} min={0.01} max={0.2} value={riskAlpha} onChange={(e) => setRiskAlpha(Number(e.target.value))} />
          </label>
        </div>
        <div className="grid-4" style={{ marginBottom: 12 }}>
          <label className="field">Parametric Policy
            <select value={paramMode} onChange={(e) => setParamMode(e.target.value)}>
              <option value="auto">Auto (Shapiro-Guided)</option>
              <option value="parametric">Parametric</option>
              <option value="nonparametric">Non-Parametric</option>
            </select>
          </label>
          <label className="field">Normality Test
            <select value={normMethod} onChange={(e) => setNormMethod(e.target.value)}>
              <option value="auto">Auto</option>
              <option value="shapiro">Shapiro-Wilk</option>
              <option value="dagostino">D'Agostino K²</option>
              <option value="anderson">Anderson-Darling</option>
              <option value="ks">Kolmogorov-Smirnov</option>
            </select>
          </label>
          <label className="field">Correlation Test
            <select value={corrMethod} onChange={(e) => setCorrMethod(e.target.value)}>
              <option value="auto">Auto</option>
              <option value="pearson">Pearson r</option>
              <option value="spearman">Spearman ρ</option>
              <option value="kendall">Kendall τ</option>
            </select>
          </label>
          <label className="field">VIF Multicollinearity Threshold
            <input type="number" step={0.5} min={2} max={25} value={vifThresh} onChange={(e) => setVifThresh(Number(e.target.value))} />
          </label>
        </div>
        {mode === 'custom' && (
          <div className="chip-grid" style={{ marginBottom: 12 }}>
            {ALL_MODS.map(([id, lbl]) => (
              <button key={id} type="button" className={`chip ${selectedMods.includes(id) ? 'selected' : ''}`} onClick={() => toggleMod(id)}>
                {lbl}
              </button>
            ))}
          </div>
        )}
        <button className="btn btn-primary" disabled={busy || !ws.has_dataset} onClick={run}>
          {busy ? <span className="spinner" /> : null} Execute Universal Diagnostics ({mode.toUpperCase()})
        </button>
      </div>

      {report && (
        <>
          <div className="kpi-grid">
            <div className="kpi-card"><div className="kpi-label">Total Hypothesis Tests</div><div className="kpi-value">{report.total_tests ?? report.hypothesis_ledger?.length ?? 0}</div></div>
            <div className="kpi-card"><div className="kpi-label">Significant (p &lt; α)</div><div className="kpi-value">{report.significant_tests ?? 0}</div></div>
            <div className="kpi-card"><div className="kpi-label">Risk Index</div><div className="kpi-value">{Number(report.risk_assessment?.overall_risk_index ?? 0).toFixed(1)}</div></div>
            <div className="kpi-card"><div className="kpi-label">Risk Band</div><div className="kpi-value">{report.risk_assessment?.overall_risk_band || 'N/A'}</div></div>
          </div>

          {/* Visual Diagnostics Charts: Baseline Comparison, Hypothesis Significance & VIF */}
          <div className="grid-2" style={{ marginBottom: 14 }}>
            {report.baseline_equations && Object.keys(report.baseline_equations).length > 0 && (() => {
              const eqItems = Object.values(report.baseline_equations).flatMap((eqList) =>
                Array.isArray(eqList) ? (eqList as EquationEntry[]) : (Object.values(eqList || {}) as EquationEntry[]),
              );
              const chartItems = eqItems
                .filter((eq) => typeof eq.primary_score === 'number' && Number.isFinite(eq.primary_score))
                .slice(0, 12)
                .map((eq) => ({
                  label: String(eq.model_name || 'Baseline'),
                  value: Number(eq.primary_score),
                  color: eq.rank === 1 ? '#34d399' : '#38bdf8',
                }));
              return chartItems.length > 0 ? (
                <div className="card" style={{ marginBottom: 0 }}>
                  <BarChartSVG title="Ranked Baseline Model Performance" horizontal items={chartItems} />
                </div>
              ) : null;
            })()}

            {(report.hypothesis_ledger?.length ?? 0) > 0 && (
              <div className="card" style={{ marginBottom: 0 }}>
                <BarChartSVG
                  title="Top Hypothesis Significance (−log₁₀ Adjusted p-value)"
                  horizontal
                  items={(report.hypothesis_ledger || [])
                    .slice(0, 12)
                    .flatMap((row) => {
                      const rawP = row.adjusted_p_value;
                      if (typeof rawP !== 'number' || !Number.isFinite(rawP)) return [];
                      const safeP = Math.max(rawP, 1e-16);
                      const negLog = -Math.log10(safeP);
                      return [{
                        label: `${String(row.variable || row.family || 'Test')} (${String(row.test_name || '')})`,
                        value: Number(negLog.toFixed(3)),
                        color: row.decision === 'Reject H0' ? '#f87171' : '#64748b',
                      }];
                    })}
                />
              </div>
            )}
          </div>

          {Array.isArray(report.vif_scores) && (report.vif_scores as Array<Record<string, unknown>>).length > 0 && (
            <div className="card">
              <BarChartSVG
                title={`Variance Inflation Factor (VIF) Multicollinearity Profile (Threshold = ${vifThresh})`}
                horizontal
                items={(report.vif_scores as Array<Record<string, unknown>>).slice(0, 14).flatMap((v) => {
                  const col = String(v.feature ?? v.column ?? '');
                  const vifVal = Number(v.vif ?? NaN);
                  return col && Number.isFinite(vifVal)
                    ? [{ label: col, value: Math.min(vifVal, 100), color: vifVal >= vifThresh ? '#f87171' : '#34d399' }]
                    : [];
                })}
              />
            </div>
          )}

          {/* MODERN RANKED BASELINE MATHEMATICAL EQUATIONS + SELECTION RATIONALE */}
          {report.baseline_equations && Object.keys(report.baseline_equations).length > 0 && (
            <div className="card">
              <div className="card-title">
                <span>Ranked Baseline Mathematical Equations &amp; Acceptance Criteria</span>
                <span className="badge badge-green">Sorted Best → Worst · Ready for Studio 14 Live Monitor</span>
              </div>
              {Object.entries(report.baseline_equations).map(([tgt, eqList]) => {
                const items: EquationEntry[] = Array.isArray(eqList)
                  ? (eqList as EquationEntry[])
                  : Object.values(eqList || {}) as EquationEntry[];
                return (
                  <div key={tgt} style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--accent)', marginBottom: 8 }}>
                      Target Variable: {tgt} ({items.length} ranked baselines evaluated)
                    </div>
                    {items.map((eq, idx) => (
                      <MathEquationCard key={`${tgt}-${eq.model_name || idx}`} eq={{ ...eq, target_name: eq.target_name || tgt }} />
                    ))}
                  </div>
                );
              })}
            </div>
          )}

          <div className="card">
            <div className="card-title">Hypothesis Test Ledger (FDR Adjusted)</div>
            <SimpleTable title="Universal Statistical Diagnostics Ledger" rows={report.hypothesis_ledger || []} maxRows={80} />
          </div>

          {(report.risk_assessment?.column_risks?.length ?? 0) > 0 && (
            <div className="card">
              <div className="card-title">Distribution &amp; Tail-Risk Profile (VaR / CVaR)</div>
              <SimpleTable title="Column Risk Metrics" rows={report.risk_assessment?.column_risks || []} />
            </div>
          )}
        </>
      )}
    </div>
  );
};

// ============================================================================
// Studio 04 — Automated EDA & Hypothesis Lab
// ============================================================================
export const Studio04EDA: React.FC<StudioProps> = ({ ws, onSync, studioCache, setStudioResult, openHelp }) => {
  const eda = studioCache.s04_eda ?? null;
  const [hypMode, setHypMode] = useState<ExecutionMode>('custom');
  const [testType, setTestType] = useState('welch_t');
  const [colA, setColA] = useState('');
  const [colB, setColB] = useState('');
  const [alpha, setAlpha] = useState(0.05);
  const hypRes = studioCache.s04_hypothesis ?? null;
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const columns = ws.columns.filter((column) => !ws.excluded_columns.includes(column) && !ws.audit_only_columns?.includes(column));
  const selectedA = columns.includes(colA) ? colA : ws.numeric_columns.find((column) => columns.includes(column)) ?? columns[0] ?? '';
  const otherColumns = columns.filter((column) => column !== selectedA);
  const selectedB = otherColumns.includes(colB) ? colB : otherColumns[0] ?? '';
  const needsSecondColumn = !['shapiro', 'dagostino', 'anderson', 'ks_normality', 'distribution_fit'].includes(testType);

  const runEDA = async () => {
    setBusy(true); setErr(null);
    try {
      const res = await apiPost<EDAResponse>('/api/workbench/s04/eda');
      onSync(res.state);
      setStudioResult('s04_eda', res);
    }
    catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  const runHyp = async () => {
    setBusy(true); setErr(null);
    try {
      const res = await apiPost<HypothesisResponse>('/api/workbench/s04/hypothesis', {
        mode: hypMode,
        test_type: testType,
        col_a: selectedA || null,
        col_b: needsSecondColumn ? selectedB || null : null,
        alpha,
      });
      if (res.state) onSync(res.state);
      setStudioResult('s04_hypothesis', res);
    } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
  };

  return (
    <div>
      <div className="studio-hero">
        <div>
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <h1>04 · Automated EDA &amp; Hypothesis Lab</h1>
            <StudioHelpButton section="studio-04" onOpenHelp={openHelp} title="Open Chapter 04: Hypothesis Tests, Formulas & Effect Sizes" />
          </div>
          <p>Distribution histograms, correlation heatmaps, automated insights, goodness-of-fit testing &amp; full Auto / Complete / Custom hypothesis batteries.</p>
        </div>
        <div className="row">
          <ProvenancePill ws={ws} />
          <button className="btn btn-primary" disabled={busy || !ws.has_dataset} onClick={runEDA}>
            {busy ? <span className="spinner" /> : null} Run Automated EDA
          </button>
        </div>
      </div>
      <LiveChartGenerator studioId="04" ws={ws} studioCache={studioCache} />
      {err && <div className="alert alert-error" role="alert">{err}</div>}
      {busy && <div role="status">Analyzing the current study data…</div>}

      {!busy && !eda && ws.has_dataset && (
        <div className="card" style={{ borderStyle: 'dashed' }}>
          <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
            Click <strong>Run Automated EDA</strong> above to generate numeric distribution histograms, categorical frequency charts, missingness bars, and the feature correlation heatmap.
          </div>
        </div>
      )}

      {eda && (
        <>
          {eda.n_rows_analyzed !== undefined && (
            <div className="alert alert-info" role="status">
              Analyzed {eda.n_rows_analyzed.toLocaleString()} {eda.sampled ? 'sampled rows' : 'rows'} in the current study.
            </div>
          )}
          {(eda.distributions?.length ?? 0) > 0 && (
            <div className="grid-2">
              {(eda.distributions || []).map((d) => (
                <div key={String(d.column)} className="card">
                  <div className="card-title"><span>{String(d.column)}</span><span className="badge">μ={String(d.mean)} · σ={String(d.std)}</span></div>
                  <BarChartSVG title={`Distribution: ${String(d.column)}`} items={(Array.isArray(d.bins) ? d.bins : []).map((b) => ({ label: String(b.label), value: Number(b.count) || 0 }))} height={175} />
                </div>
              ))}
            </div>
          )}
          {(eda.categorical_distributions?.length ?? 0) > 0 && (
            <div className="grid-2">
              {(eda.categorical_distributions || []).map((cd) => (
                <div key={String(cd.column)} className="card">
                  <div className="card-title">
                    <span>{String(cd.column)}</span>
                    <span className="badge">{Number(cd.unique_count ?? 0)} unique categories</span>
                  </div>
                  <BarChartSVG
                    title={`Category Breakdown: ${String(cd.column)}`}
                    horizontal
                    items={(Array.isArray(cd.top_values) ? cd.top_values : []).map((tv) => ({
                      label: String(tv.label ?? ''),
                      value: Number(tv.count) || 0,
                      color: '#a78bfa',
                    }))}
                  />
                </div>
              ))}
            </div>
          )}
          {(eda.missing_bars?.length ?? 0) > 0 && (
            <div className="card">
              <div className="card-title">Column Missingness Rate (%)</div>
              <BarChartSVG
                title="Missing Values by Column (%)"
                horizontal
                items={(eda.missing_bars || []).map((mb) => ({
                  label: String(mb.column ?? ''),
                  value: Number(mb.missing_pct) || 0,
                  color: Number(mb.missing_pct ?? 0) > 20 ? '#f87171' : '#fbbf24',
                }))}
              />
            </div>
          )}
          {(eda.correlation_matrix?.columns?.length ?? 0) > 1 && (
            <div className="card">
              <div className="card-title">Correlation Matrix Heatmap</div>
              <HeatmapGrid title="Numeric Feature Correlation Matrix" columns={eda.correlation_matrix?.columns || []} values={eda.correlation_matrix?.values || []} />
            </div>
          )}
          <div className="card">
            <div className="card-title">Automated Statistical Insights ({eda.insights?.length || 0})</div>
            <SimpleTable
              title="Automated EDA Insights"
              rows={(eda.insights || []).map((item) => (typeof item === 'string' ? { insight: item } : item))}
            />
          </div>
        </>
      )}

      <div className="card">
        <div className="card-title">
          <span className="row" style={{ gap: 6, alignItems: 'center' }}>
            Hypothesis Lab
            <StudioHelpButton section="stat-atlas" onOpenHelp={openHelp} label="Test Selection Atlas" title="Open Master Statistical Test Decision Atlas" />
          </span>
          <ModeBar mode={hypMode} onChange={setHypMode} />
        </div>
        <div className="row" style={{ marginBottom: 12, gap: 10 }}>
          {hypMode === 'custom' && (
            <>
              <label className="field">Statistical Test
                <select value={testType} onChange={(e) => setTestType(e.target.value)}>
                  <optgroup label="Normality & Goodness-of-Fit">
                    <option value="shapiro">Shapiro-Wilk Normality</option>
                    <option value="dagostino">D'Agostino-Pearson K² Omnibus</option>
                    <option value="anderson">Anderson-Darling Normality</option>
                    <option value="ks_normality">Kolmogorov-Smirnov Normality</option>
                    <option value="distribution_fit">Distribution Goodness-of-Fit (7 Families)</option>
                  </optgroup>
                  <optgroup label="Two-Sample & Paired Comparisons">
                    <option value="welch_t">Welch Two-Sample t-Test</option>
                    <option value="mann_whitney">Mann-Whitney U (Non-Parametric)</option>
                    <option value="paired_t">Paired Student t-Test</option>
                    <option value="wilcoxon">Wilcoxon Signed-Rank Test</option>
                    <option value="ks_2samp">Two-Sample KS Distribution Test</option>
                  </optgroup>
                  <optgroup label="Multi-Group & Variance Homogeneity">
                    <option value="anova">One-Way ANOVA</option>
                    <option value="kruskal">Kruskal-Wallis H-Test</option>
                    <option value="levene">Levene Variance Equality</option>
                    <option value="bartlett">Bartlett Variance Equality</option>
                    <option value="fligner">Fligner-Killeen Robust Variance</option>
                  </optgroup>
                  <optgroup label="Correlation & Categorical Independence">
                    <option value="pearson">Pearson r Correlation</option>
                    <option value="spearman">Spearman ρ Rank Correlation</option>
                    <option value="kendall">Kendall τ Rank Correlation</option>
                    <option value="chi_square">Chi-Square Independence</option>
                    <option value="fisher_exact">Fisher's Exact Test (2×2)</option>
                  </optgroup>
                </select>
              </label>
              <label className="field">Column A (Value)
                <select value={selectedA} onChange={(e) => setColA(e.target.value)}>
                  {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
              {needsSecondColumn && (
                <label className="field">Column B (Group / Pair)
                  <select value={selectedB} onChange={(e) => setColB(e.target.value)}>
                    {otherColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </label>
              )}
            </>
          )}
          <label className="field">α Threshold
            <input type="number" step={0.01} min={0.001} max={0.2} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} style={{ width: 85 }} />
          </label>
          <button className="btn btn-primary" disabled={busy || !ws.has_dataset || (hypMode === 'custom' && (!selectedA || (needsSecondColumn && !selectedB)))} onClick={runHyp}>
            {busy ? <span className="spinner" /> : null} Run Hypothesis Test ({hypMode.toUpperCase()})
          </button>
        </div>

        {hypRes?.battery && (
          <div>
            <div className="kpi-grid">
              <div className="kpi-card"><div className="kpi-label">Tests Executed</div><div className="kpi-value">{hypRes.total_tests}</div></div>
              <div className="kpi-card"><div className="kpi-label">{hypRes.decision_policy ? 'Significant after BH adjustment' : 'Statistically Significant'}</div><div className="kpi-value">{hypRes.significant_tests}</div></div>
            </div>
            {hypRes.decision_policy && <p>{hypRes.decision_policy}. The chart caps −log₁₀ values at 16 for display.</p>}
            {hypRes.battery.length > 0 && (
              <div style={{ marginBottom: 14 }}>
                <BarChartSVG
                  title={hypRes.decision_policy ? 'Hypothesis Battery (−log₁₀ adjusted p-value)' : 'Hypothesis Battery (−log₁₀ p-value)'}
                  horizontal
                  items={hypRes.battery.slice(0, 14).flatMap((row) => {
                    const pVal = hypRes.decision_policy ? row.p_value_fdr : row.p_value;
                    if (typeof pVal !== 'number' || !Number.isFinite(pVal)) return [];
                    const safeP = Math.max(pVal, 1e-16);
                    return [{
                      label: `${String(row.scope ?? row.column ?? row.pair ?? row.variables ?? '')} · ${String(row.test_name ?? row.test ?? '')}`,
                      value: Number((-Math.log10(safeP)).toFixed(3)),
                      color: row.significant ? '#34d399' : '#64748b',
                    }];
                  })}
                />
              </div>
            )}
            {hypRes.battery.length === 0 && <div className="alert alert-info">No tests were evaluable for the selected data.</div>}
            <SimpleTable title={`Hypothesis Test Battery (${(hypRes.mode || '').toUpperCase()})`} rows={hypRes.battery} maxRows={100} />
            {Boolean(hypRes.skipped?.length) && (
              <SimpleTable title="Tests skipped and why" rows={hypRes.skipped ?? []} maxRows={100} />
            )}
          </div>
        )}

        {hypRes?.mode === 'custom' && hypRes.result && (
          <div>
            <div className="alert alert-info">
              {hypRes.result.evaluable === false || hypRes.result.p_value == null ? (
                <><strong>Not evaluable</strong> · {hypRes.result.interpretation}</>
              ) : (
                <><strong>{hypRes.result.test_name}:</strong> statistic={hypRes.result.statistic == null ? 'unavailable' : Number(hypRes.result.statistic).toFixed(4)} · p-value={Number(hypRes.result.p_value).toExponential(3)} · <strong>{hypRes.result.p_value_scope === 'exploratory_selected_family' ? 'EXPLORATORY FITTED-NULL CHECK' : hypRes.result.significant ? 'SIGNIFICANT' : 'NOT SIGNIFICANT'}</strong> · {hypRes.result.interpretation}</>
              )}
            </div>
            {hypRes.result.extra_table && (
              <SimpleTable title="Distribution Fit Ranking by AIC" rows={hypRes.result.extra_table} />
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// Studio 05 — Feature Engineering & Multi-Method Consensus Selection
// ============================================================================
export const Studio05Features: React.FC<StudioProps> = ({ ws, onSync, bumpGrid, studioCache, setStudioResult, openHelp }) => {
  const [engMode, setEngMode] = useState<ExecutionMode>('auto');
  const [selMode, setSelMode] = useState<ExecutionMode>('auto');
  const [interactions, setInteractions] = useState(true);
  const [poly, setPoly] = useState(false);
  const [skew, setSkew] = useState(true);
  const [dtFeat, setDtFeat] = useState(true);
  const [method, setMethod] = useState('mutual_info');
  const [topK, setTopK] = useState(10);
  const engRes = studioCache.s05_engineer ?? null;
  const selRes = studioCache.s05_select ?? null;
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  return (
    <div>
      <div className="studio-hero">
        <div>
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <h1>05 · Feature Engineering &amp; Consensus Feature Selection</h1>
            <StudioHelpButton section="studio-05" onOpenHelp={openHelp} title="Open Chapter 05: Yeo-Johnson, Cyclical Encoding & 4-Method Consensus Selection" />
          </div>
          <p>Create interaction, polynomial, skew-corrected &amp; datetime features and select highest-signal predictors via Auto, Complete (4-method consensus), or Custom selection.</p>
        </div>
        <ProvenancePill ws={ws} />
      </div>
      {err && <div className="alert alert-error">{err}</div>}
      <LiveChartGenerator studioId="05" ws={ws} studioCache={studioCache} />
      <div className="grid-2">
        <div className="card">
          <div className="card-title">
            <span>Feature Engineering</span>
            <ModeBar mode={engMode} onChange={setEngMode} />
          </div>
          {engMode === 'custom' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 14 }}>
              <label className="row"><input type="checkbox" checked={interactions} onChange={(e) => setInteractions(e.target.checked)} /> Pairwise Interaction Terms</label>
              <label className="row"><input type="checkbox" checked={poly} onChange={(e) => setPoly(e.target.checked)} /> Polynomial Features (Degree 2)</label>
              <label className="row"><input type="checkbox" checked={skew} onChange={(e) => setSkew(e.target.checked)} /> Yeo-Johnson Skewness Correction</label>
              <label className="row"><input type="checkbox" checked={dtFeat} onChange={(e) => setDtFeat(e.target.checked)} /> Datetime Cyclical Decomposition</label>
            </div>
          )}
          <button className="btn btn-primary" disabled={busy || !ws.has_dataset} onClick={async () => {
            setBusy(true); setErr(null);
            try {
              const r = await apiPost<EngineerResponse>('/api/workbench/s05/engineer', {
                mode: engMode, add_interactions: interactions, add_polynomials: poly, correct_skew: skew, add_datetime_features: dtFeat, commit: true,
              });
              onSync(r.state);
              setStudioResult('s05_engineer', r);
              bumpGrid();
            } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
          }}>Generate &amp; Commit Features ({engMode.toUpperCase()})</button>
          {engRes && <div className="alert alert-success" style={{ marginTop: 12 }}>Created {engRes.n_created} engineered features.</div>}
        </div>

        <div className="card">
          <div className="card-title">
            <span>Feature Selection</span>
            <ModeBar mode={selMode} onChange={setSelMode} />
          </div>
          <div className="row" style={{ marginBottom: 12 }}>
            {selMode === 'custom' && (
              <label className="field">Method
                <select value={method} onChange={(e) => setMethod(e.target.value)}>
                  <option value="mutual_info">Mutual Information</option>
                  <option value="f_test">Statistical F-Test (ANOVA)</option>
                  <option value="chi2">Chi-Square (Classification)</option>
                  <option value="mrmr">mRMR (Min Redundancy Max Relevance)</option>
                  <option value="importance">Tree Feature Importance</option>
                  <option value="rfe">Recursive Feature Elimination (RFE)</option>
                  <option value="boruta_shadow">Boruta Shadow Permutation</option>
                  <option value="l1">L1 Lasso / Sparse Regularization</option>
                  <option value="variance">Variance Threshold</option>
                  <option value="correlation">Correlation Filter</option>
                </select>
              </label>
            )}
            <label className="field">Top K Features
              <input type="number" min={2} max={50} value={topK} onChange={(e) => setTopK(Number(e.target.value))} style={{ width: 80 }} />
            </label>
          </div>
          <div className="row">
            <button className="btn btn-primary" disabled={busy || !ws.has_dataset} onClick={async () => {
              setBusy(true); setErr(null);
              try {
                const r = await apiPost<SelectionResponse>('/api/workbench/s05/select', { mode: selMode, method, top_k: topK, commit: false });
                onSync(r.state);
                setStudioResult('s05_select', r);
              } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
            }}>Rank Features ({selMode.toUpperCase()})</button>
            <button className="btn btn-success" disabled={busy || !ws.has_dataset} onClick={async () => {
              setBusy(true); setErr(null);
              try {
                const r = await apiPost<SelectionResponse>('/api/workbench/s05/select', { mode: selMode, method, top_k: topK, commit: true });
                onSync(r.state);
                setStudioResult('s05_select', r);
                bumpGrid();
              } catch (e: unknown) { setErr(toErrorMessage(e)); } finally { setBusy(false); }
            }}>Rank &amp; Apply Exclusions</button>
          </div>
        </div>
      </div>

      {selRes?.rankings && (
        <div className="card">
          <div className="card-title">Feature Importance Ranking ({selRes.selected_features?.length || 0} selected · Mode: {String(selRes.mode || selMode).toUpperCase()})</div>
          {selRes.selected_source_columns && (
            <p>{selRes.committed ? 'Applied' : 'Preview'} · {selRes.fit_scope === 'train_only' ? 'Fitted on locked training rows' : 'Fitted on the full study'} · Source predictors {selRes.selected_source_columns.join(', ')}</p>
          )}
          <BarChartSVG
            title="Feature Importance / Consensus Ranking"
            horizontal
            items={(selRes.rankings || []).map((r) => ({ label: r.feature, value: r.score, color: r.selected ? '#34d399' : '#64748b' }))}
          />
        </div>
      )}
    </div>
  );
};
