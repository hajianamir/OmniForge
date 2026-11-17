import React, { useEffect, useState } from 'react';
import { apiGet } from '../api';
import type { GridResponse } from '../types';
import { InspectableBox } from './InspectModal';

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
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;
    setLoading(true);
    const params = new URLSearchParams({
      page: String(page),
      page_size: String(pageSize),
      view_mode: viewMode,
      sort_dir: sortDir,
    });
    if (sortCol) params.set('sort_col', sortCol);
    if (search.trim()) params.set('search', search.trim());
    apiGet<GridResponse>(`/api/workbench/s01/grid?${params.toString()}`)
      .then((res) => { if (active) setData(res); })
      .catch(() => {})
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [page, pageSize, sortCol, sortDir, search, viewMode, refreshKey]);

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
};

export const SimpleTable: React.FC<{
  rows: Record<string, unknown>[];
  maxRows?: number;
  title?: string;
}> = ({ rows, maxRows = 60, title = 'Data Table' }) => {
  if (!rows || !rows.length) return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No records available.</div>;
  const cols = Object.keys(rows[0]);
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
                  const v = r[c];
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
};
