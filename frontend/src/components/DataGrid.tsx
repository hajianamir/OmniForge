import React, { useEffect, useRef, useState } from 'react';
import { ApiError, apiGet } from '../api';
import type { GridResponse } from '../types';
import { InspectableBox } from './InspectModal';

/** Debounce delay (ms) before a search keystroke triggers a server round-trip. */
const SEARCH_DEBOUNCE_MS = 300;

export const PaginatedDataGrid: React.FC<{
  refreshKey: number;
  viewMode?: 'active' | 'raw';
  title?: string;
}> = ({ refreshKey, viewMode = 'active', title }) => {
  const [data, setData] = useState<GridResponse | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Debounce the search box: the input stays instant, the server is hit only
  // after the user pauses typing (avoids one request per keystroke).
  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedSearch(search), SEARCH_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [search]);

  useEffect(() => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    const params = new URLSearchParams({
      page: String(page),
      page_size: String(pageSize),
      view_mode: viewMode,
      sort_dir: sortDir,
    });
    if (sortCol) params.set('sort_col', sortCol);
    if (debouncedSearch.trim()) params.set('search', debouncedSearch.trim());
    apiGet<GridResponse>(`/api/workbench/s01/grid?${params.toString()}`, { signal: controller.signal })
      .then((res) => { if (!controller.signal.aborted) setData(res); })
      .catch((err: unknown) => {
        if (err instanceof ApiError && err.code === 'ABORTED') return;
      })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [page, pageSize, sortCol, sortDir, debouncedSearch, viewMode, refreshKey]);

  const toggleSort = (col: string) => {
    if (sortCol === col) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortCol(col);
      setSortDir('asc');
    }
    setPage(1);
  };

  if (!data || !data.columns.length) {
    return <div style={{ color: 'var(--text-dim)', padding: 12 }}>No rows loaded yet.</div>;
  }

  return (
    <InspectableBox
      title={title || (viewMode === 'raw' ? 'Raw Dataset Explorer' : 'Active Dataset Explorer')}
      subtitle={`${data.total_rows.toLocaleString()} rows × ${data.columns.length} columns · Page ${data.page}/${data.total_pages}`}
      compact
    >
      <div>
        <div className="row" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
          <div className="row">
            <input
              type="text"
              placeholder="Filter rows (server-side)..."
              value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(1); }}
              style={{ width: 240 }}
            />
            <select value={pageSize} onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1); }}>
              <option value={25}>25 / page</option>
              <option value={50}>50 / page</option>
              <option value={100}>100 / page</option>
            </select>
            {loading && <span className="spinner" />}
          </div>
          <div className="row" style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            <span>Showing page {data.page} of {data.total_pages} ({data.total_rows.toLocaleString()} rows)</span>
            <button className="btn btn-sm" disabled={data.page <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>Prev</button>
            <button className="btn btn-sm" disabled={data.page >= data.total_pages} onClick={() => setPage((p) => Math.min(data.total_pages, p + 1))}>Next</button>
          </div>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                {data.columns.map((c) => (
                  <th key={c.name} className="sortable" onClick={() => toggleSort(c.name)}>
                    <span>{c.name}</span>
                    {c.is_target && <span className="badge badge-accent" style={{ marginLeft: 5, fontSize: 9.5 }}>TARGET</span>}
                    {c.excluded && <span className="badge badge-red" style={{ marginLeft: 5, fontSize: 9.5 }}>EXCL</span>}
                    {sortCol === c.name && <span style={{ marginLeft: 4 }}>{sortDir === 'asc' ? '▲' : '▼'}</span>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row, idx) => (
                <tr key={idx}>
                  {data.columns.map((c) => (
                    <td key={c.name} style={c.excluded ? { opacity: 0.45 } : undefined}>
                      {row[c.name] === null || row[c.name] === undefined ? <span style={{ color: 'var(--text-dim)' }}>null</span> : String(row[c.name])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </InspectableBox>
  );
}

/**
 * Compact, inspectable table for arbitrary record arrays.
 * Generic over the row type so typed API payloads (interfaces) can be passed
 * directly: interfaces are not assignable to `Record<string, unknown>`, but any
 * object shape renders fine here.
 */
export function SimpleTable<T extends object>({
  rows,
  maxRows = 60,
  title = 'Data Table',
}: {
  rows: T[];
  maxRows?: number;
  title?: string;
}) {
  if (!rows || !rows.length) return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No records available.</div>;
  const cols = Object.keys(rows[0] as Record<string, unknown>);
  return (
    <InspectableBox title={title} subtitle={`${rows.length.toLocaleString()} rows × ${cols.length} cols`} compact>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {rows.slice(0, maxRows).map((r, i) => (
              <tr key={i}>
                {cols.map((c) => {
                  const v = (r as Record<string, unknown>)[c];
                  const text =
                    v === null || v === undefined
                      ? '—'
                      : typeof v === 'number'
                        ? Number.isInteger(v) ? String(v) : v.toFixed(4)
                        : typeof v === 'object'
                          ? JSON.stringify(v)
                          : String(v);
                  return <td key={c} title={text}>{text.length > 90 ? text.slice(0, 88) + '…' : text}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </InspectableBox>
  );
}
