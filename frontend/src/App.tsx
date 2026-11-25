import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BookOpen,
  Cpu,
  Database,
  FileText,
  FolderOpen,
  Gauge,
  GitBranch,
  HelpCircle,
  Lock,
  MessageSquare,
  Moon,
  Save,
  ScatterChart,
  Server,
  Shield,
  Sliders,
  Sparkles,
  Sun,
  Target,
  TrendingUp,
  Unlock,
  X,
} from 'lucide-react';
import {
  ApiError,
  apiDelete,
  apiDownload,
  apiGet,
  apiPost,
  apiUpload,
  clearStoredCredentials,
  hasStoredCredentials,
  setStoredCredentials,
  toErrorMessage,
} from './api';
import type { HandbookSectionId } from './components/HelpGuideModal';
import type { StudioCache, StudioPayload, StudioResultKey } from './studioTypes';
import { STUDIOS, type StudioId, type WorkspaceState } from './types';

// Code-split studios: each studio chunk downloads only on first visit, then
// stays mounted (keep-alive via display:none) so tabs/forms/results survive.
const Studio00Ingestion = React.lazy(() => import('./studios/StudiosPart1').then((m) => ({ default: m.Studio00Ingestion })));
const Studio01Data = React.lazy(() => import('./studios/StudiosPart1').then((m) => ({ default: m.Studio01Data })));
const Studio02Clustering = React.lazy(() => import('./studios/StudiosPart1').then((m) => ({ default: m.Studio02Clustering })));
const Studio03Diagnostics = React.lazy(() => import('./studios/StudiosPart1').then((m) => ({ default: m.Studio03Diagnostics })));
const Studio04EDA = React.lazy(() => import('./studios/StudiosPart1').then((m) => ({ default: m.Studio04EDA })));
const Studio05Features = React.lazy(() => import('./studios/StudiosPart1').then((m) => ({ default: m.Studio05Features })));
const Studio06Modeling = React.lazy(() => import('./studios/StudiosPart2').then((m) => ({ default: m.Studio06Modeling })));
const Studio07Responsible = React.lazy(() => import('./studios/StudiosPart2').then((m) => ({ default: m.Studio07Responsible })));
const Studio08Decision = React.lazy(() => import('./studios/StudiosPart2').then((m) => ({ default: m.Studio08Decision })));
const Studio09Specialized = React.lazy(() => import('./studios/StudiosPart3').then((m) => ({ default: m.Studio09Specialized })));
const Studio10DAG = React.lazy(() => import('./studios/StudiosPart3').then((m) => ({ default: m.Studio10DAG })));
const Studio11MLOps = React.lazy(() => import('./studios/StudiosPart3').then((m) => ({ default: m.Studio11MLOps })));
const Studio12Reporting = React.lazy(() => import('./studios/StudiosPart3').then((m) => ({ default: m.Studio12Reporting })));
const Studio13Copilot = React.lazy(() => import('./studios/StudiosPart3').then((m) => ({ default: m.Studio13Copilot })));
const Studio14Supervisory = React.lazy(() => import('./studios/Studio14Supervisory').then((m) => ({ default: m.Studio14Supervisory })));
const HelpGuideModal = React.lazy(() => import('./components/HelpGuideModal').then((m) => ({ default: m.HelpGuideModal })));
const StudioHelpButton = React.lazy(() => import('./components/HelpGuideModal').then((m) => ({ default: m.StudioHelpButton })));

const StudioLoadingFallback: React.FC = () => (
  <div style={{ padding: 24, color: 'var(--text-dim)', fontSize: 13 }}>
    <span className="spinner" /> Loading studio chunk…
  </div>
);

const ICONS: Record<StudioId, React.ReactNode> = {
  '00': <Database size={15} />,
  '01': <Database size={15} />,
  '02': <ScatterChart size={15} />,
  '03': <Activity size={15} />,
  '04': <BarChart3 size={15} />,
  '05': <Sliders size={15} />,
  '06': <Cpu size={15} />,
  '07': <Shield size={15} />,
  '08': <TrendingUp size={15} />,
  '09': <Sparkles size={15} />,
  '10': <GitBranch size={15} />,
  '11': <Server size={15} />,
  '12': <FileText size={15} />,
  '13': <MessageSquare size={15} />,
  '14': <Gauge size={15} />,
};

const EMPTY_WS: WorkspaceState = {
  session_id: 'default',
  platform: { name: 'OmniForge', version: '4.0.0', edition: 'Enterprise' },
  has_dataset: false,
  dataset_name: null,
  dataset_source: null,
  n_rows: 0,
  n_cols: 0,
  raw_rows: 0,
  raw_cols: 0,
  columns: [],
  numeric_columns: [],
  categorical_columns: [],
  targets: [],
  primary_target: null,
  excluded_columns: [],
  analysis_cols_count: 0,
  can_undo: false,
  processing_status: {},
  split_lock: null,
  total_deleted_rows: 0,
  format_audit_issues: 0,
  intake_audit: null,
  s02_cluster_committed: false,
  s02_anomaly_committed: false,
  active_model_name: null,
  active_model_metrics: null,
  active_feature_cols: [],
  available_models_count: 0,
  supervisory_status: 'IDLE',
  studio_completed: {},
  workflow_checklist: [],
  lineage: [],
  theme: 'dark',
  project_name: 'omniforge_project',
};

export const App: React.FC = () => {
  const [ws, setWs] = useState<WorkspaceState>(EMPTY_WS);
  const [studio, setStudio] = useState<StudioId>('00');
  const [visitedStudios, setVisitedStudios] = useState<StudioId[]>(['00']);
  const [gridVersion, setGridVersion] = useState(0);
  const [modal, setModal] = useState<'none' | 'targets' | 'exclusions' | 'split' | 'project' | 'auth'>('none');
  const [helpOpen, setHelpOpen] = useState(false);
  const [helpSection, setHelpSection] = useState<HandbookSectionId>('workflow');
  const [toast, setToast] = useState<string | null>(null);
  const [cacheError, setCacheError] = useState<string | null>(null);
  const [authRequired, setAuthRequired] = useState(false);
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [bearerInput, setBearerInput] = useState('');
  const [studioCache, setStudioCache] = useState<StudioCache>({});
  // Keep the current cache synchronously available to event handlers. React
  // state updater functions may run later (or more than once in StrictMode).
  const studioCacheRef = useRef<StudioCache>({});
  const cacheGeneration = useRef(0);
  const cacheRequest = useRef(0);
  const keyRevisions = useRef<Partial<Record<StudioResultKey, number>>>({});

  const openHelp = useCallback((section: HandbookSectionId = 'workflow') => {
    setHelpSection(section);
    setHelpOpen(true);
  }, []);

  const mainPaneRef = useRef<HTMLElement | null>(null);

  const navigateStudio = useCallback((id: StudioId) => {
    setStudio(id);
    setVisitedStudios((prev) => (prev.includes(id) ? prev : [...prev, id]));
    requestAnimationFrame(() => {
      mainPaneRef.current?.scrollTo({ top: 0, behavior: 'auto' });
    });
  }, []);

  // Split Lock form state
  const [splitStrat, setSplitStrat] = useState('auto');
  const [splitTestSize, setSplitTestSize] = useState(0.2);
  const [splitCvFolds, setSplitCvFolds] = useState(5);

  // .omniforge Project save/load
  const [projName, setProjName] = useState('omniforge_project');
  const [projNotes, setProjNotes] = useState('');

  const updateCache = useCallback((next: StudioCache) => {
    studioCacheRef.current = next;
    setStudioCache(next);
  }, []);

  const fetchServerCache = useCallback(() => {
    const generation = cacheGeneration.current;
    const request = ++cacheRequest.current;
    const revisions = { ...keyRevisions.current };
    apiGet<{ results?: Partial<StudioCache> }>('/api/workbench/results')
      .then((res) => {
        // A response from an old workspace/scope, or an older refresh, cannot
        // resurrect charts that have since been invalidated or replaced.
        if (generation !== cacheGeneration.current || request !== cacheRequest.current) return;
        const results = res?.results;
        if (!results || typeof results !== 'object') throw new Error('The server returned no results.');
        setCacheError(null);
        const merged: Record<string, unknown> = { ...studioCacheRef.current };
        for (const [key, value] of Object.entries(results)) {
          const k = key as StudioResultKey;
          // Do not overwrite a result edited locally while this GET was in flight.
          if ((revisions[k] ?? 0) === (keyRevisions.current[k] ?? 0)) merged[key] = value;
        }
        updateCache(merged as StudioCache);
      })
      .catch((err: unknown) => {
        if (generation === cacheGeneration.current && request === cacheRequest.current) {
          setCacheError(`Could not restore studio results: ${toErrorMessage(err)}`);
        }
      });
  }, [updateCache]);

  const setStudioResult = useCallback((key: StudioResultKey, val: StudioPayload) => {
    keyRevisions.current[key] = (keyRevisions.current[key] ?? 0) + 1;
    const next: StudioCache = { ...studioCacheRef.current };
    if (val === undefined || val === null) delete next[key];
    else (next as Record<string, StudioPayload | undefined>)[key] = val;
    updateCache(next);
  }, [updateCache]);

  const syncState = useCallback((next?: WorkspaceState, replaceCache = false) => {
    if (next) {
      cacheGeneration.current += 1;
      setWs(next);
      document.documentElement.setAttribute('data-theme', next.theme === 'light' ? 'light' : 'dark');

      // The server's key list is authoritative even without an active dataset:
      // e.g. standalone NLP and image analyses can still be valid.
      let filtered = replaceCache ? {} : studioCacheRef.current;
      if (Array.isArray(next.cached_result_keys)) {
        const allowed = new Set(next.cached_result_keys);
        filtered = Object.fromEntries(Object.entries(filtered).filter(([key]) => allowed.has(key))) as StudioCache;
      } else if (!next.has_dataset) {
        filtered = {};
      }
      updateCache(filtered);
      if (replaceCache || !Array.isArray(next.cached_result_keys) || next.cached_result_keys.some((key) => !(key in filtered))) {
        fetchServerCache();
      } else {
        setCacheError(null);
      }
      setAuthRequired(false);
      return;
    }
    const generation = cacheGeneration.current;
    apiGet<WorkspaceState>('/api/workbench/state')
      .then((s) => {
        // Initial state can arrive after a user has already replaced the workspace.
        if (generation !== cacheGeneration.current) return;
        syncState(s);
      })
      .catch((err: unknown) => {
        if (generation === cacheGeneration.current) {
          setAuthRequired(err instanceof ApiError && err.status === 401);
          setCacheError(`Could not load workspace: ${toErrorMessage(err)}`);
        }
      });
  }, [fetchServerCache, updateCache]);

  useEffect(() => { syncState(); }, [syncState]);

  const bumpGrid = useCallback(() => setGridVersion((v) => v + 1), []);
  const flash = useCallback((text: string) => {
    setToast(text);
    window.setTimeout(() => setToast((cur) => (cur === text ? null : cur)), 4500);
  }, []);

  const toggleTheme = useCallback(async () => {
    const nextTheme = ws.theme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', nextTheme);
    setWs((prev) => ({ ...prev, theme: nextTheme }));
    try { await apiPost('/api/workbench/theme', { theme: nextTheme }); } catch {}
  }, [ws.theme]);

  const toggleTarget = useCallback(async (col: string) => {
    const next = ws.targets.includes(col) ? ws.targets.filter((c) => c !== col) : [...ws.targets, col];
    if (!next.length) return;
    const r = await apiPost<{ state: WorkspaceState }>('/api/workbench/targets', { targets: next });
    syncState(r.state); bumpGrid();
  }, [ws.targets, syncState, bumpGrid]);

  const toggleExclude = useCallback(async (col: string) => {
    const next = ws.excluded_columns.includes(col)
      ? ws.excluded_columns.filter((c) => c !== col)
      : [...ws.excluded_columns, col];
    const r = await apiPost<{ state: WorkspaceState }>('/api/workbench/exclusions', { excluded_columns: next });
    syncState(r.state); bumpGrid();
  }, [ws.excluded_columns, syncState, bumpGrid]);

  const handleSaveProject = useCallback(async () => {
    try {
      const fd = new FormData();
      fd.append('project_name', projName || ws.project_name);
      fd.append('notes', projNotes);
      await apiDownload('/api/workbench/workspace/save', `${projName || 'omniforge_project'}.omniforge`, 'POST', fd);
      flash('Saved portable .omniforge workspace project.');
      setModal('none');
    } catch (err: unknown) {
      flash(`Save failed: ${toErrorMessage(err)}`);
    }
  }, [projName, projNotes, ws.project_name, flash]);

  const handleLoadProject = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await apiUpload<{ state: WorkspaceState }>('/api/workbench/workspace/load', fd);
      // A restored project can contain different values under the same keys.
      syncState(res.state, true);
      bumpGrid();
      flash(`Restored OmniForge project '${file.name}'.`);
      setModal('none');
    } catch (err: unknown) {
      flash(`Restore failed: ${toErrorMessage(err)}`);
    } finally {
      e.target.value = '';
    }
  }, [syncState, bumpGrid, flash]);

  // Memoized so keep-alive studios (display:none) do not re-render on every
  // App-level state change (toast, modal, theme flash). Studios receive a
  // stable props object unless workspace/cache data actually changes.
  const commonProps = useMemo(
    () => ({
      ws,
      onSync: syncState,
      gridVersion,
      bumpGrid,
      studioCache,
      setStudioResult,
      refreshCache: fetchServerCache,
      openHelp,
    }),
    [ws, syncState, gridVersion, bumpGrid, studioCache, setStudioResult, fetchServerCache, openHelp],
  );

  return (
    <div className="app-shell">
      {/* Persistent Top Header */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">Ω</div>
          <div>
            <span>{ws.platform?.name || 'OmniForge'}</span>
            <span className="badge badge-accent" style={{ marginLeft: 8 }}>v{ws.platform?.version || '4.0.0'} · Enterprise</span>
          </div>
        </div>

        <div className="topbar-meta">
          {ws.supervisory_status && ws.supervisory_status !== 'IDLE' && (
            <button
              className={`btn btn-sm ${ws.supervisory_status === 'CRITICAL_ALARM' ? 'btn-danger' : ws.supervisory_status === 'WARNING' ? '' : 'btn-success'}`}
              onClick={() => navigateStudio('14')}
              title="Open Studio 14 Supervisory Monitor"
            >
              <AlertTriangle size={13} /> Monitor: {ws.supervisory_status}
            </button>
          )}
          {ws.has_dataset ? (
            <>
              <span className="badge badge-green">
                {ws.dataset_name} · {ws.n_rows.toLocaleString()} × {ws.n_cols}
              </span>
              <span
                className={`badge ${ws.processing_status?.is_processed ? 'badge-accent' : 'badge-amber'}`}
                title={`${ws.processing_status?.summary || ''} | Active: ${ws.n_rows}×${ws.n_cols} (Raw: ${ws.raw_rows}×${ws.raw_cols}) — All downstream studios operate on this active dataset.`}
              >
                {ws.processing_status?.detailed_label || (ws.processing_status?.is_processed ? 'PROCESSED' : 'RAW')}
              </span>
              {ws.intake_audit && ws.intake_audit.defect_counts?.total > 0 && (
                <button className="btn btn-sm" onClick={() => navigateStudio('01')} title="Inspect Automatic Intake Defects">
                  Defects: {ws.intake_audit.defect_counts.total} ({ws.intake_audit.quality_grade})
                </button>
              )}
              <button className="btn btn-sm" onClick={() => setModal('targets')}>
                <Target size={13} /> Targets ({ws.targets.length})
              </button>
              <button className="btn btn-sm" onClick={() => setModal('exclusions')}>
                Exclusions ({ws.excluded_columns.length})
              </button>
              <button className={`btn btn-sm ${ws.split_lock?.active ? 'btn-success' : ''}`} onClick={() => setModal('split')}>
                {ws.split_lock?.active ? <Lock size={13} /> : <Unlock size={13} />}
                {ws.split_lock?.active ? `Split Locked (${ws.split_lock.strategy})` : 'Split Lock'}
              </button>
            </>
          ) : (
            <span className="badge">No Dataset Loaded</span>
          )}
        </div>

        <div className="topbar-fixed-actions">
          <button className="btn btn-sm" onClick={() => { setProjName(ws.project_name); setModal('project'); }} title="Save or Restore Portable .omniforge Project">
            <Save size={13} /> Project
          </button>
          <button
            className="btn btn-sm topbar-help-btn"
            data-global-help="true"
            onClick={() => openHelp('workflow')}
            title="Open Complete OmniForge Comprehensive Reference Handbook (Formulas, Thresholds, Tests & All 14 Studios)"
          >
            <HelpCircle size={15} /> Help
          </button>
          <button className="btn btn-sm" onClick={toggleTheme} title="Toggle Dark/Light Mode">
            {ws.theme === 'light' ? <Moon size={14} /> : <Sun size={14} />}
          </button>
        </div>
      </header>

      {/* CRISP-DM Lifecycle Left Sidebar + Persistent Green Completion LEDs */}
      <aside className="sidebar">
        <div className="sidebar-section">CRISP-DM 14-Studio Lifecycle</div>
        {STUDIOS.map((s) => {
          const isDone = Boolean(ws.studio_completed?.[s.id]);
          return (
            <button
              key={s.id}
              data-studio={s.id}
              data-completed={isDone ? 'true' : 'false'}
              className={`nav-item ${studio === s.id ? 'active' : ''}`}
              onClick={() => navigateStudio(s.id)}
              title={`${s.id} · ${s.title}${isDone ? ' (Completed)' : ''}`}
            >
              <span className="nav-code">{s.id}</span>
              {ICONS[s.id]}
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.short}</span>
              <span className={`studio-led ${isDone ? 'done' : ''}`} aria-label={isDone ? 'Completed' : 'Pending'} />
            </button>
          );
        })}

        <div style={{ padding: '10px 8px 4px' }}>
          <button className="btn btn-sm" style={{ width: '100%', justifyContent: 'center' }} onClick={() => openHelp('workflow')}>
            <BookOpen size={13} /> Complete Handbook
          </button>
        </div>

        {ws.workflow_checklist?.length > 0 && (
          <>
            <div className="sidebar-section" style={{ marginTop: 10 }}>Workflow Readiness</div>
            <div style={{ padding: '0 8px', display: 'flex', flexDirection: 'column', gap: 5 }}>
              {ws.workflow_checklist.map((item) => (
                <div key={item.label} style={{ fontSize: 11.5, color: item.done ? 'var(--green)' : 'var(--text-dim)' }}>
                  {item.done ? '✓' : '○'} {item.label}
                </div>
              ))}
            </div>
          </>
        )}
      </aside>

      {/* Main Studio Workspace — Keep visited studios mounted so returning NEVER resets tabs, forms, or results */}
      <main ref={mainPaneRef} className="main-pane">
        {toast && <div className="alert alert-success">{toast}</div>}
        {cacheError && (
          <div className="alert alert-error" role="alert">
            {cacheError}{' '}
            <button className="btn btn-sm" onClick={() => syncState()}>Retry workspace results</button>
            {authRequired && (
              <>
                {' '}
                <button className="btn btn-sm btn-primary" onClick={() => setModal('auth')}>Authenticate Session</button>
              </>
            )}
          </div>
        )}
        <React.Suspense fallback={<StudioLoadingFallback />}>
          {visitedStudios.includes('00') && <div style={{ display: studio === '00' ? 'block' : 'none' }}><Studio00Ingestion {...commonProps} /></div>}
          {visitedStudios.includes('01') && <div style={{ display: studio === '01' ? 'block' : 'none' }}><Studio01Data {...commonProps} /></div>}
          {visitedStudios.includes('02') && <div style={{ display: studio === '02' ? 'block' : 'none' }}><Studio02Clustering {...commonProps} /></div>}
          {visitedStudios.includes('03') && <div style={{ display: studio === '03' ? 'block' : 'none' }}><Studio03Diagnostics {...commonProps} /></div>}
          {visitedStudios.includes('04') && <div style={{ display: studio === '04' ? 'block' : 'none' }}><Studio04EDA {...commonProps} /></div>}
          {visitedStudios.includes('05') && <div style={{ display: studio === '05' ? 'block' : 'none' }}><Studio05Features {...commonProps} /></div>}
          {visitedStudios.includes('06') && <div style={{ display: studio === '06' ? 'block' : 'none' }}><Studio06Modeling {...commonProps} /></div>}
          {visitedStudios.includes('07') && <div style={{ display: studio === '07' ? 'block' : 'none' }}><Studio07Responsible {...commonProps} /></div>}
          {visitedStudios.includes('08') && <div style={{ display: studio === '08' ? 'block' : 'none' }}><Studio08Decision {...commonProps} /></div>}
          {visitedStudios.includes('09') && <div style={{ display: studio === '09' ? 'block' : 'none' }}><Studio09Specialized {...commonProps} /></div>}
          {visitedStudios.includes('10') && <div style={{ display: studio === '10' ? 'block' : 'none' }}><Studio10DAG {...commonProps} /></div>}
          {visitedStudios.includes('11') && <div style={{ display: studio === '11' ? 'block' : 'none' }}><Studio11MLOps {...commonProps} /></div>}
          {visitedStudios.includes('12') && <div style={{ display: studio === '12' ? 'block' : 'none' }}><Studio12Reporting {...commonProps} /></div>}
          {visitedStudios.includes('13') && <div style={{ display: studio === '13' ? 'block' : 'none' }}><Studio13Copilot {...commonProps} /></div>}
          {visitedStudios.includes('14') && <div style={{ display: studio === '14' ? 'block' : 'none' }}><Studio14Supervisory {...commonProps} /></div>}
        </React.Suspense>
      </main>

      <React.Suspense fallback={null}>
        <HelpGuideModal open={helpOpen} initialSection={helpSection} onClose={() => setHelpOpen(false)} onJumpStudio={navigateStudio} />
      </React.Suspense>

      {/* Always-Visible Floating Contextual '?' Quick-Help Launcher for the Current Studio */}
      {!helpOpen && (
        <button
          type="button"
          className="floating-help-fab"
          data-floating-help={`studio-${studio}`}
          onClick={() => openHelp(`studio-${studio}` as HandbookSectionId)}
          title={`Open Comprehensive Reference Chapter for Studio ${studio} (Formulas, Thresholds, Worked Examples & Troubleshooting)`}
          aria-label={`Open Help Chapter for Studio ${studio}`}
        >
          <HelpCircle size={17} />
          <span>? Studio {studio} Guide</span>
        </button>
      )}

      {/* Modals: Multi-Target, Column Exclusions, Split Lock, .omniforge Project */}
      {modal !== 'none' && (
        <div className="modal-backdrop" onClick={() => setModal('none')}>
          <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
            <div className="card-title">
              <span>
                {modal === 'targets' && 'Global Multi-Target Selector'}
                {modal === 'exclusions' && 'Reversible Column Exclusions (All 14 Studios)'}
                {modal === 'split' && 'Anti-Leakage Train/Test Split Lock'}
                {modal === 'project' && 'Portable .omniforge Workspace Save & Restore'}
                {modal === 'auth' && 'Workbench Session Authentication'}
              </span>
              <div className="row" style={{ gap: 6 }}>
                <React.Suspense fallback={null}>
                  <StudioHelpButton
                    section={modal === 'project' ? 'studio-12' : 'studio-01'}
                    onOpenHelp={(sec) => { setModal('none'); openHelp(sec); }}
                    title={
                      modal === 'split'
                        ? 'Read Split Lock vs Data Cleaning Anti-Leakage Guide'
                        : modal === 'project'
                          ? 'Read Portable .omniforge Workspace & Reporting Guide'
                          : 'Read Target Selection, Column Exclusions & Data Provenance Guide'
                    }
                    compact
                  />
                </React.Suspense>
                <button className="btn btn-sm" onClick={() => setModal('none')}><X size={14} /></button>
              </div>
            </div>

            {modal === 'targets' && (
              <div>
                <p style={{ color: 'var(--text-muted)', marginTop: 0 }}>Click columns to toggle target selection across Diagnostics, Feature Selection, AutoML, Supervisory Monitor &amp; Reporting.</p>
                <div className="chip-grid">
                  {ws.columns.map((c) => (
                    <button key={c} className={`chip ${ws.targets.includes(c) ? 'selected' : ''}`} onClick={() => toggleTarget(c)}>{c}</button>
                  ))}
                </div>
              </div>
            )}

            {modal === 'exclusions' && (
              <div>
                <p style={{ color: 'var(--text-muted)', marginTop: 0 }}>Excluded columns are masked reversibly across all 14 studios without mutating raw data.</p>
                <div className="chip-grid" style={{ marginBottom: 14 }}>
                  {ws.columns.map((c) => (
                    <button key={c} className={`chip ${ws.excluded_columns.includes(c) ? 'excluded' : ''}`} onClick={() => toggleExclude(c)}>{c}</button>
                  ))}
                </div>
                {ws.excluded_columns.length > 0 && (
                  <button className="btn btn-sm" onClick={async () => { const r = await apiPost<{ state: WorkspaceState }>('/api/workbench/exclusions', { excluded_columns: [] }); syncState(r.state); bumpGrid(); }}>
                    Restore All Excluded Columns
                  </button>
                )}
              </div>
            )}

            {modal === 'split' && (
              <div>
                <p style={{ color: 'var(--text-muted)', marginTop: 0 }}>Lock a deterministic Train/Test partition BEFORE cleaning, clustering, feature engineering, or modeling to guarantee zero data leakage.</p>
                <div className="grid-3" style={{ marginBottom: 14 }}>
                  <label className="field">Strategy
                    <select value={splitStrat} onChange={(e) => setSplitStrat(e.target.value)}>
                      <option value="auto">Auto (Stratified / Random)</option>
                      <option value="stratified">Stratified</option>
                      <option value="random">Random</option>
                      <option value="time_series">Chronological Time-Series</option>
                    </select>
                  </label>
                  <label className="field">Test Ratio
                    <input type="number" step={0.05} min={0.1} max={0.4} value={splitTestSize} onChange={(e) => setSplitTestSize(Number(e.target.value))} />
                  </label>
                  <label className="field">CV Folds
                    <input type="number" min={2} max={10} value={splitCvFolds} onChange={(e) => setSplitCvFolds(Number(e.target.value))} />
                  </label>
                </div>
                <div className="row">
                  <button className="btn btn-primary" onClick={async () => {
                    const r = await apiPost<{ state: WorkspaceState }>('/api/workbench/split-lock', { strategy: splitStrat, test_size: splitTestSize, cv_folds: splitCvFolds });
                    syncState(r.state); flash('Anti-Leakage Split Lock enabled.'); setModal('none');
                  }}>Lock Train/Test Split</button>
                  {ws.split_lock?.active && (
                    <button className="btn btn-danger" onClick={async () => {
                      const r = await apiDelete<{ state: WorkspaceState }>('/api/workbench/split-lock');
                      syncState(r.state); flash('Split Lock unlocked.'); setModal('none');
                    }}>Unlock Split</button>
                  )}
                </div>
              </div>
            )}

            {modal === 'project' && (
              <div className="grid-2">
                <div>
                  <h4 style={{ marginTop: 0 }}>Save Active Workspace (.omniforge)</h4>
                  <label className="field" style={{ marginBottom: 8 }}>Project Name
                    <input type="text" value={projName} onChange={(e) => setProjName(e.target.value)} />
                  </label>
                  <label className="field" style={{ marginBottom: 10 }}>Notes
                    <textarea rows={2} value={projNotes} onChange={(e) => setProjNotes(e.target.value)} />
                  </label>
                  <button className="btn btn-primary" disabled={!ws.has_dataset} onClick={handleSaveProject}><Save size={14} /> Download .omniforge File</button>
                </div>
                <div>
                  <h4 style={{ marginTop: 0 }}>Restore Portable .omniforge Project</h4>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>Upload any saved OmniForge <code>.omniforge</code> (or legacy <code>.uap</code>) snapshot to restore datasets, exclusions, models &amp; lineage.</p>
                  <label className="btn" style={{ cursor: 'pointer' }}>
                    <FolderOpen size={14} /> Choose .omniforge File
                    <input type="file" accept=".omniforge,.uap" onChange={handleLoadProject} style={{ display: 'none' }} />
                  </label>
                </div>
              </div>
            )}

            {modal === 'auth' && (
              <div>
                <p style={{ color: 'var(--text-muted)', marginTop: 0 }}>
                  Provide an <code>X-API-Key</code> (from <code>OMNIFORGE_SECURITY__API_KEY</code> or{' '}
                  <code>.omniforge_workspace/.secrets/api_key</code>) or a <code>Bearer</code> JWT token for remote / non-loopback servers.
                </p>
                <label className="field" style={{ marginBottom: 8 }}>API Key (X-API-Key)
                  <input
                    type="password"
                    autoComplete="off"
                    value={apiKeyInput}
                    placeholder="Paste server API key..."
                    onChange={(e) => setApiKeyInput(e.target.value)}
                  />
                </label>
                <label className="field" style={{ marginBottom: 12 }}>Bearer JWT Token (optional)
                  <input
                    type="password"
                    autoComplete="off"
                    value={bearerInput}
                    placeholder="Paste Bearer JWT token..."
                    onChange={(e) => setBearerInput(e.target.value)}
                  />
                </label>
                <div className="row" style={{ gap: 8 }}>
                  <button
                    className="btn btn-primary"
                    onClick={() => {
                      setStoredCredentials({ apiKey: apiKeyInput, bearerToken: bearerInput });
                      setApiKeyInput('');
                      setBearerInput('');
                      setModal('none');
                      setCacheError(null);
                      syncState();
                    }}
                  >
                    Save Credentials &amp; Connect
                  </button>
                  {hasStoredCredentials() && (
                    <button
                      className="btn btn-sm"
                      onClick={() => {
                        clearStoredCredentials();
                        setApiKeyInput('');
                        setBearerInput('');
                        setModal('none');
                        syncState();
                      }}
                    >
                      Clear Saved Credentials
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
