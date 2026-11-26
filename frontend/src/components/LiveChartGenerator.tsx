import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { BarChart3, Download, FileSpreadsheet, Image as ImageIcon, RefreshCw, Table } from 'lucide-react';
import { apiDownloadJson, apiGet, apiPost, toErrorMessage } from '../api';
import type { StudioCache } from '../studioTypes';
import type { StudioId, WorkspaceState } from '../types';
import { SimpleTable } from './DataGrid';
import {
  ProfessionalChartDispatcher,
  serializeLiveChartSvg,
  type LiveChartPayload,
  type ProfessionalChartType,
} from './ProfessionalCharts';

export interface LiveChartCatalogItem {
  id: string;
  chart_type: ProfessionalChartType;
  title: string;
  category: string;
  description: string;
  studios: string[];
  recommended_for_studio: boolean;
  ready: boolean;
  missing_reason: string | null;
  supports_feature_x: boolean;
  supports_feature_y: boolean;
  supports_group_by: boolean;
  feature_x_kind: 'numeric' | 'categorical' | 'any' | string;
  feature_y_kind: 'numeric' | 'categorical' | 'any' | string;
}

interface LiveChartCatalogResponse {
  studio_id: string;
  has_dataset: boolean;
  dataset_name: string | null;
  active_model_name: string | null;
  columns: {
    all: string[];
    numeric: string[];
    categorical: string[];
    primary_target: string | null;
    task_type: string;
  };
  charts: LiveChartCatalogItem[];
}

export interface LiveChartGeneratorProps {
  studioId: StudioId | 'all';
  ws: WorkspaceState;
  studioCache?: StudioCache;
  title?: string;
}

const safeSlug = (raw: string): string =>
  raw
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 56) || 'omniforge_chart';

/**
 * On-Demand Professional Chart Studio:
 * Presents a single dropdown selector of 25+ professional statistical, ML, diagnostic,
 * and engineering chart types. ONLY the user-selected chart is computed on demand
 * from real workspace data/models — zero synthetic PRNG data. Supports instant export
 * in SVG, PNG, and multi-sheet XLSX formats.
 */
export const LiveChartGenerator: React.FC<LiveChartGeneratorProps> = ({
  studioId,
  ws,
  studioCache,
  title,
}) => {
  const [catalog, setCatalog] = useState<LiveChartCatalogResponse | null>(null);
  const [selectedChartId, setSelectedChartId] = useState<string>('');
  const [featureX, setFeatureX] = useState<string>('');
  const [featureY, setFeatureY] = useState<string>('');
  const [groupCol, setGroupCol] = useState<string>('');

  const [generated, setGenerated] = useState<LiveChartPayload | null>(null);
  const [busy, setBusy] = useState(false);
  const [exportingFmt, setExportingFmt] = useState<'svg' | 'png' | 'xlsx' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showData, setShowData] = useState(false);

  const stageRef = useRef<HTMLDivElement | null>(null);
  const activeAbortRef = useRef<AbortController | null>(null);

  // Refresh lightweight metadata catalog whenever dataset, model, or cached studio artifacts change
  const cacheFingerprint = useMemo(
    () => (studioCache ? Object.keys(studioCache).filter((k) => Boolean(studioCache[k as keyof StudioCache])).sort().join(',') : ''),
    [studioCache],
  );

  useEffect(() => {
    let alive = true;
    if (!ws.has_dataset) {
      setCatalog(null);
      setGenerated(null);
      setError(null);
      return;
    }
    apiGet<LiveChartCatalogResponse>(`/api/workbench/charts/catalog?studio_id=${encodeURIComponent(studioId)}`)
      .then((res) => {
        if (!alive) return;
        setCatalog(Array.isArray(res?.charts) && Array.isArray(res?.columns?.all) && Array.isArray(res?.columns?.numeric) ? res : null);
      })
      .catch((err: unknown) => {
        if (!alive) return;
        setError(toErrorMessage(err));
        setCatalog(null);
      });
    return () => {
      alive = false;
    };
  }, [
    studioId,
    ws.has_dataset,
    ws.dataset_name,
    ws.n_rows,
    ws.n_cols,
    ws.active_model_name,
    ws.supervisory_status,
    ws.targets.join(','),
    ws.excluded_columns.join(','),
    cacheFingerprint,
  ]);

  const selectedSpec = useMemo(
    () => catalog?.charts.find((c) => c.id === selectedChartId) ?? null,
    [catalog, selectedChartId],
  );

  // Reset invalid column overrides when dataset columns change
  useEffect(() => {
    const all = new Set(catalog?.columns.all || ws.columns);
    if (featureX && !all.has(featureX)) setFeatureX('');
    if (featureY && !all.has(featureY)) setFeatureY('');
    if (groupCol && !all.has(groupCol)) setGroupCol('');
  }, [catalog, ws.columns, featureX, featureY, groupCol]);

  const runSingleChart = useCallback(
    async (
      chartId: string,
      opts?: { fx?: string; fy?: string; grp?: string },
    ) => {
      activeAbortRef.current?.abort();
      if (!chartId) {
        setGenerated(null);
        setError(null);
        return;
      }
      const spec = catalog?.charts.find((c) => c.id === chartId);
      if (spec && !spec.ready) {
        setGenerated(null);
        setError(spec.missing_reason || 'This chart requires an upstream studio step before it can be computed.');
        return;
      }

      const controller = new AbortController();
      activeAbortRef.current = controller;
      setBusy(true);
      setError(null);
      try {
        const res = await apiPost<LiveChartPayload>(
          '/api/workbench/charts/generate',
          {
            chart_id: chartId,
            studio_id: studioId,
            feature_x: (opts?.fx ?? featureX) || null,
            feature_y: (opts?.fy ?? featureY) || null,
            group_col: (opts?.grp ?? groupCol) || null,
          },
          { signal: controller.signal },
        );
        if (!controller.signal.aborted) {
          setGenerated(res);
        }
      } catch (err: unknown) {
        if (!controller.signal.aborted) {
          setGenerated(null);
          setError(toErrorMessage(err));
        }
      } finally {
        if (activeAbortRef.current === controller) {
          setBusy(false);
        }
      }
    },
    [catalog, studioId, featureX, featureY, groupCol],
  );

  // Automatically generate ONLY the selected chart when user picks from dropdown or changes variable selectors
  useEffect(() => {
    if (!ws.has_dataset || !selectedChartId || !catalog) {
      if (!selectedChartId) {
        setGenerated(null);
        setError(null);
      }
      return;
    }
    void runSingleChart(selectedChartId);
    return () => {
      activeAbortRef.current?.abort();
    };
  }, [
    ws.has_dataset,
    ws.n_rows,
    ws.n_cols,
    ws.active_model_name,
    selectedChartId,
    featureX,
    featureY,
    groupCol,
    catalog,
    runSingleChart,
  ]);

  const handleDownload = async (fmt: 'svg' | 'png' | 'xlsx') => {
    if (!generated || !selectedChartId) return;
    setExportingFmt(fmt);
    setError(null);
    const baseName = `${safeSlug(generated.title)}_${generated.chart_type}`;
    const currentTheme: 'light' | 'dark' =
      typeof document !== 'undefined' && document.documentElement.getAttribute('data-theme') === 'light'
        ? 'light'
        : 'dark';

    try {
      if (fmt === 'svg') {
        const serialized = serializeLiveChartSvg(stageRef.current, generated.title, generated.subtitle);
        if (serialized) {
          const blob = new Blob([serialized], { type: 'image/svg+xml;charset=utf-8' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `${baseName}.svg`;
          document.body.appendChild(a);
          a.click();
          a.remove();
          URL.revokeObjectURL(url);
          return;
        }
      }

      await apiDownloadJson(
        '/api/workbench/charts/export',
        `${baseName}.${fmt}`,
        {
          chart_id: selectedChartId,
          format: fmt,
          studio_id: studioId,
          feature_x: featureX || null,
          feature_y: featureY || null,
          group_col: groupCol || null,
          theme: fmt === 'xlsx' ? 'light' : currentTheme,
        },
      );
    } catch (err: unknown) {
      setError(`Export (${fmt.toUpperCase()}) failed: ${toErrorMessage(err)}`);
    } finally {
      setExportingFmt(null);
    }
  };

  const groupedOptions = useMemo(() => {
    const byCat = new Map<string, LiveChartCatalogItem[]>();
    for (const item of catalog?.charts || []) {
      const list = byCat.get(item.category) || [];
      list.push(item);
      byCat.set(item.category, list);
    }
    return Array.from(byCat.entries()).map(([groupLabel, list]) => ({ groupLabel, list }));
  }, [catalog]);

  const xColumns = useMemo(() => {
    const cols = catalog?.columns;
    if (!cols) return ws.columns;
    if (selectedSpec?.feature_x_kind === 'numeric' && cols.numeric.length > 0) return cols.numeric;
    return cols.all;
  }, [catalog, selectedSpec, ws.columns]);

  const yColumns = useMemo(() => {
    const cols = catalog?.columns;
    if (!cols) return ws.columns;
    if (selectedSpec?.feature_y_kind === 'numeric' && cols.numeric.length > 0) return cols.numeric;
    return cols.all;
  }, [catalog, selectedSpec, ws.columns]);

  const tableObjects = useMemo(() => {
    if (!generated || !generated.table_columns.length) return [];
    return generated.table_rows.map((r) => {
      const obj: Record<string, unknown> = {};
      generated.table_columns.forEach((col, i) => {
        obj[col] = r[i] ?? '';
      });
      return obj;
    });
  }, [generated]);

  if (!ws.has_dataset) {
    return null;
  }

  const readyCount = (catalog?.charts || []).filter((c) => c.ready).length;
  const totalCount = (catalog?.charts || []).length;

  return (
    <div className="card" style={{ borderLeft: '3px solid var(--accent)' }}>
      <div className="card-title" style={{ marginBottom: 10, flexWrap: 'wrap' }}>
        <span className="row" style={{ gap: 8, alignItems: 'center' }}>
          <BarChart3 size={16} style={{ color: 'var(--accent)' }} />
          <span>
            {title ||
              (studioId === '12' || studioId === 'all'
                ? 'On-Demand Professional Chart Studio & Multi-Format Exporter (SVG · PNG · XLSX)'
                : `On-Demand Professional Chart Builder · Studio ${studioId}`)}
          </span>
        </span>
        <span className="badge badge-accent">
          {catalog ? `${readyCount}/${totalCount} Live Charts Ready` : error ? 'Chart catalog unavailable' : 'Loading chart catalog'}
        </span>
      </div>

      <div className="row" style={{ gap: 10, alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <label className="field" style={{ flex: '2 1 320px', minWidth: 260 }}>
          Choose Professional Chart Type (Only Selected Chart Is Generated)
          <select
            aria-label="Select professional chart to generate"
            value={selectedChartId}
            onChange={(e) => {
              setSelectedChartId(e.target.value);
              setShowData(false);
            }}
          >
            <option value="">— Select a professional chart to generate on demand —</option>
            {groupedOptions.map((grp) => (
              <optgroup key={grp.groupLabel} label={grp.groupLabel}>
                {grp.list.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.ready ? '✓ ' : '○ '}
                    {item.title} [{item.chart_type}]
                    {!item.ready ? ' — (Requires upstream model/run)' : ''}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </label>

        {selectedSpec?.supports_feature_x && (
          <label className="field" style={{ flex: '1 1 155px' }}>
            Primary Variable (X)
            <select value={featureX} onChange={(e) => setFeatureX(e.target.value)}>
              <option value="">Auto (Recommended)</option>
              {xColumns.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>
        )}

        {selectedSpec?.supports_feature_y && (
          <label className="field" style={{ flex: '1 1 155px' }}>
            Secondary Variable (Y)
            <select value={featureY} onChange={(e) => setFeatureY(e.target.value)}>
              <option value="">Auto (Target / Best Pair)</option>
              {yColumns.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>
        )}

        {selectedSpec?.supports_group_by && (
          <label className="field" style={{ flex: '1 1 155px' }}>
            Stratify / Group By
            <select value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
              <option value="">Auto (Split / Target)</option>
              {(catalog?.columns.all || ws.columns).map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>
        )}

        {selectedChartId && (
          <div className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={busy}
              onClick={() => void runSingleChart(selectedChartId)}
              title="Recompute selected chart from live workspace state"
            >
              <RefreshCw size={13} /> {busy ? 'Computing…' : 'Refresh'}
            </button>
            <button
              type="button"
              className="btn btn-primary"
              disabled={!generated || busy || exportingFmt !== null}
              onClick={() => void handleDownload('svg')}
              title="Download crisp standalone vector SVG"
            >
              <Download size={13} /> {exportingFmt === 'svg' ? 'Exporting…' : 'SVG'}
            </button>
            <button
              type="button"
              className="btn btn-primary"
              disabled={!generated || busy || exportingFmt !== null}
              onClick={() => void handleDownload('png')}
              title="Download high-resolution 200-DPI PNG"
            >
              <ImageIcon size={13} /> {exportingFmt === 'png' ? 'Exporting…' : 'PNG'}
            </button>
            <button
              type="button"
              className="btn btn-success"
              disabled={!generated || busy || exportingFmt !== null}
              onClick={() => void handleDownload('xlsx')}
              title="Download Excel workbook with embedded chart image, native Excel chart & full underlying data sheet"
            >
              <FileSpreadsheet size={13} /> {exportingFmt === 'xlsx' ? 'Exporting…' : 'XLSX'}
            </button>
          </div>
        )}
      </div>

      {selectedSpec && !error && (
        <div style={{ marginTop: 8, fontSize: 11.5, color: 'var(--text-dim)', display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
          <span>{selectedSpec.description}</span>
          {generated && <span className="mono">{generated.provenance}</span>}
        </div>
      )}

      {error && (
        <div className="alert alert-warn" style={{ marginTop: 10, marginBottom: 0 }}>
          {error}
        </div>
      )}

      {busy && !generated && (
        <div style={{ marginTop: 12, color: 'var(--text-muted)', fontSize: 12.5 }}>
          <span className="spinner" /> Computing live statistical chart from active workspace…
        </div>
      )}

      {generated && (
        <div style={{ marginTop: 12 }}>
          {Object.keys(generated.kpis || {}).length > 0 && (
            <div className="row" style={{ gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
              {Object.entries(generated.kpis).map(([k, v]) => (
                <span key={k} className="badge" style={{ padding: '4px 9px', fontSize: 11 }}>
                  <span style={{ color: 'var(--text-dim)', marginRight: 5 }}>{k}:</span>
                  <strong className="mono">{String(v ?? '—')}</strong>
                </span>
              ))}
              <button
                type="button"
                className="btn btn-sm"
                style={{ marginLeft: 'auto' }}
                onClick={() => setShowData((prev) => !prev)}
              >
                <Table size={12} /> {showData ? 'Hide Chart Data' : `Inspect Chart Data (${generated.table_rows.length} rows)`}
              </button>
            </div>
          )}

          <div ref={stageRef}>
            <ProfessionalChartDispatcher payload={generated} />
          </div>

          {showData && tableObjects.length > 0 && (
            <div style={{ marginTop: 10 }}>
              <SimpleTable
                title={`Underlying Data — ${generated.title} (${tableObjects.length} rows)`}
                rows={tableObjects}
                maxRows={50}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};
