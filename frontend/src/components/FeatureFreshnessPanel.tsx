import React, { useCallback, useEffect, useState } from 'react';
import { apiGet, toErrorMessage } from '../api';

interface FreshnessView {
  view_name: string;
  ttl_seconds: number;
  online_rows: number;
  fresh_rows: number;
  expired_rows: number;
  unknown_timestamp_rows: number;
}

interface FreshnessReport {
  as_of: string;
  total_views: number;
  online_rows: number;
  expired_rows: number;
  unknown_timestamp_rows: number;
  views: FreshnessView[];
  alerts: { view_name: string; message: string }[];
}

/** Tenant-scoped snapshot of the feature store's currently materialized vectors. */
export const FeatureFreshnessPanel: React.FC = () => {
  const [report, setReport] = useState<FreshnessReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setReport(await apiGet<FreshnessReport>('/api/workbench/feature-store/freshness'));
      setError(null);
    } catch (e: unknown) {
      setError(toErrorMessage(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => { void refresh(); }, 60_000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  return (
    <section className="card" aria-label="Feature store freshness">
      <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="card-title">Feature Store Freshness</div>
        <button className="btn" onClick={() => { void refresh(); }} disabled={loading}>Refresh</button>
      </div>
      <p>Online feature vectors currently past their TTL. Counts exclude previously purged rows.</p>
      {error && <div className="alert alert-error" role="alert">Freshness unavailable: {error}</div>}
      {report && (
        <>
          <div className="kpi-grid">
            <div className="kpi-card"><div className="kpi-label">Feature Views</div><div className="kpi-value">{report.total_views}</div></div>
            <div className="kpi-card"><div className="kpi-label">Online Vectors</div><div className="kpi-value">{report.online_rows}</div></div>
            <div className="kpi-card"><div className="kpi-label">Expired Vectors</div><div className="kpi-value">{report.expired_rows}</div></div>
          </div>
          {report.alerts.length > 0 && (
            <div className="alert alert-warn" role="alert" style={{ marginTop: 12 }}>
              <strong>Stale features detected</strong>
              <ul>{report.alerts.map((alert) => <li key={alert.view_name}>{alert.message}</li>)}</ul>
            </div>
          )}
          {report.unknown_timestamp_rows > 0 && (
            <div className="alert alert-warn" role="alert">{report.unknown_timestamp_rows} vector(s) have no usable timestamp; freshness is unknown.</div>
          )}
          {report.total_views === 0 ? <p>No feature views are registered for this tenant.</p> : (
            <div style={{ overflowX: 'auto', marginTop: 12 }}>
              <table>
                <thead><tr><th>View</th><th>TTL (seconds)</th><th>Online</th><th>Expired</th><th>Unknown</th></tr></thead>
                <tbody>{report.views.map((view) => (
                  <tr key={view.view_name}>
                    <td>{view.view_name}</td><td>{view.ttl_seconds > 0 ? view.ttl_seconds : 'Disabled'}</td>
                    <td>{view.online_rows}</td><td>{view.expired_rows}</td><td>{view.unknown_timestamp_rows}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          )}
          <small>Snapshot: {report.as_of} · refreshes every minute</small>
        </>
      )}
    </section>
  );
};
