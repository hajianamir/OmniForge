import React, { useCallback, useEffect, useRef } from 'react';
import type { ScatterPayload } from '../types';
import { InspectableBox } from './InspectModal';

const PALETTE = ['#38bdf8', '#a78bfa', '#34d399', '#fbbf24', '#f87171', '#fb7185', '#2dd4bf', '#818cf8', '#f97316', '#c084fc'];

export const CanvasScatter: React.FC<{
  data: ScatterPayload;
  xLabel?: string;
  yLabel?: string;
  height?: number;
  title?: string;
}> = ({ data, xLabel = 'Component 1', yLabel = 'Component 2', height = 340, title = '2D Scatter Projection' }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rafRef = useRef<number>(0);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data.x?.length) return;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 640;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const pad = { l: 48, r: 18, t: 16, b: 36 };
    const pw = width - pad.l - pad.r;
    const ph = height - pad.t - pad.b;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < data.x.length; i++) {
      const xv = Number(data.x[i]);
      const yv = Number(data.y[i]);
      if (Number.isFinite(xv)) {
        if (xv < minX) minX = xv;
        if (xv > maxX) maxX = xv;
      }
      if (Number.isFinite(yv)) {
        if (yv < minY) minY = yv;
        if (yv > maxY) maxY = yv;
      }
    }
    if (!Number.isFinite(minX) || minX === maxX) { minX -= 1; maxX += 1; }
    if (!Number.isFinite(minY) || minY === maxY) { minY -= 1; maxY += 1; }

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.18)';
    ctx.lineWidth = 1;
    for (let k = 0; k <= 4; k++) {
      const gy = pad.t + (ph * k) / 4;
      ctx.beginPath(); ctx.moveTo(pad.l, gy); ctx.lineTo(pad.l + pw, gy); ctx.stroke();
      const gx = pad.l + (pw * k) / 4;
      ctx.beginPath(); ctx.moveTo(gx, pad.t); ctx.lineTo(gx, pad.t + ph); ctx.stroke();
    }

    const uniqGroups = Array.from(new Set((data.groups || []).map(String)));
    const groupColor = new Map<string, string>();
    uniqGroups.forEach((g, idx) => groupColor.set(g, g === '-1' ? '#f87171' : PALETTE[idx % PALETTE.length]));

    for (let i = 0; i < data.x.length; i++) {
      const xv = Number(data.x[i]);
      const yv = Number(data.y[i]);
      if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue;
      const cx = pad.l + ((xv - minX) / (maxX - minX)) * pw;
      const cy = pad.t + ph - ((yv - minY) / (maxY - minY)) * ph;
      const gKey = data.groups ? String(data.groups[i]) : '0';
      ctx.fillStyle = groupColor.get(gKey) || PALETTE[0];
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      ctx.arc(cx, cy, data.x.length > 2500 ? 2.4 : 3.6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px sans-serif';
    ctx.fillText(xLabel, pad.l + pw / 2 - 30, height - 8);
    ctx.save();
    ctx.translate(14, pad.t + ph / 2 + 24);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }, [data, xLabel, yLabel, height]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Redraw crisply when the container resizes (sidebar collapse, window drag,
  // monitor change). Coalesced through requestAnimationFrame so a drag does
  // not queue a redraw per ResizeObserver notification.
  useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(() => draw());
    });
    observer.observe(container);
    return () => {
      observer.disconnect();
      cancelAnimationFrame(rafRef.current);
    };
  }, [draw]);

  const uniqGroups = Array.from(new Set((data.groups || []).map(String))).slice(0, 12);

  return (
    <InspectableBox
      title={title}
      subtitle={`${xLabel} vs ${yLabel} · ${data.rendered_points.toLocaleString()} / ${data.total_points.toLocaleString()} pts`}
      compact
    >
      <div ref={containerRef}>
        <canvas ref={canvasRef} style={{ width: '100%', height, display: 'block', borderRadius: 10, background: 'var(--bg)' }} />
        <div className="row" style={{ marginTop: 8, justifyContent: 'space-between' }}>
          <div className="row" style={{ gap: 6 }}>
            {uniqGroups.map((g, idx) => (
              <span key={g} className="badge" style={{ gap: 5 }}>
                <span style={{ width: 8, height: 8, borderRadius: 99, background: g === '-1' ? '#f87171' : PALETTE[idx % PALETTE.length], display: 'inline-block' }} />
                {g === '-1' ? 'Noise (-1)' : `Group ${g}`}
              </span>
            ))}
          </div>
          <span style={{ fontSize: 11.5, color: 'var(--text-dim)' }}>
            Canvas 60fps · Rendered {data.rendered_points.toLocaleString()} / {data.total_points.toLocaleString()} points
          </span>
        </div>
      </div>
    </InspectableBox>
  );
};

export const BarChartSVG: React.FC<{
  items: { label: string; value: number; color?: string }[];
  height?: number;
  horizontal?: boolean;
  title?: string;
}> = ({ items, height = 220, horizontal = false, title = 'Bar Chart' }) => {
  if (!items.length) return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No data to display.</div>;
  const maxVal = Math.max(...items.map((i) => Math.abs(i.value)), 1e-9);

  if (horizontal) {
    return (
      <InspectableBox title={title} compact>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
          {items.slice(0, 16).map((item, idx) => {
            const pct = Math.min(100, (Math.abs(item.value) / maxVal) * 100);
            return (
              <div key={`${item.label}-${idx}`} style={{ display: 'grid', gridTemplateColumns: '160px 1fr 74px', alignItems: 'center', gap: 10, fontSize: 12 }}>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-muted)' }} title={item.label}>
                  {item.label}
                </span>
                <div style={{ height: 11, background: 'var(--bg-subtle)', borderRadius: 99, overflow: 'hidden' }}>
                  <div style={{ width: `${pct}%`, height: '100%', background: item.color || PALETTE[idx % PALETTE.length], borderRadius: 99 }} />
                </div>
                <span className="mono" style={{ textAlign: 'right', fontSize: 11.5 }}>{Number(item.value).toFixed(4)}</span>
              </div>
            );
          })}
        </div>
      </InspectableBox>
    );
  }

  const w = 520;
  const h = height;
  const pad = { l: 38, r: 10, t: 14, b: 32 };
  const bw = (w - pad.l - pad.r) / Math.max(items.length, 1);

  return (
    <InspectableBox title={title} compact>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height }}>
        {items.map((it, i) => {
          const bh = (Math.max(0, it.value) / maxVal) * (h - pad.t - pad.b);
          const x = pad.l + i * bw + 2;
          const y = h - pad.b - bh;
          return (
            <g key={`${it.label}-${i}`}>
              <rect x={x} y={y} width={Math.max(3, bw - 4)} height={bh} rx={3} fill={it.color || '#38bdf8'} opacity={0.88}>
                <title>{`${it.label}: ${it.value}`}</title>
              </rect>
              {items.length <= 24 && i % Math.max(1, Math.ceil(items.length / 8)) === 0 && (
                <text x={x + bw / 2} y={h - 10} textAnchor="middle" fill="#94a3b8" fontSize={9.5}>
                  {it.label.slice(0, 9)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </InspectableBox>
  );
};

export const ForecastFanSVG: React.FC<{
  history: { step: number; actual: number }[];
  forecast: { step: number; yhat: number; yhat_lower?: number; yhat_upper?: number }[];
  height?: number;
  title?: string;
}> = ({ history, forecast, height = 270, title = 'Forecast & Prediction Intervals' }) => {
  const allVals = [
    ...history.map((h) => h.actual),
    ...forecast.flatMap((f) => [f.yhat, f.yhat_lower ?? f.yhat, f.yhat_upper ?? f.yhat]),
  ].filter(Number.isFinite);
  if (!allVals.length) return null;

  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const spanV = maxV - minV || 1;
  const allSteps = [...history.map((h) => h.step), ...forecast.map((f) => f.step)];
  const minS = Math.min(...allSteps);
  const maxS = Math.max(...allSteps);
  const spanS = maxS - minS || 1;

  const W = 720;
  const H = height;
  const pad = { l: 50, r: 20, t: 18, b: 30 };
  const sx = (s: number) => pad.l + ((s - minS) / spanS) * (W - pad.l - pad.r);
  const sy = (v: number) => pad.t + (H - pad.t - pad.b) - ((v - minV) / spanV) * (H - pad.t - pad.b);

  const histPath = history.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.step).toFixed(1)} ${sy(p.actual).toFixed(1)}`).join(' ');
  const forePath = forecast.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.step).toFixed(1)} ${sy(p.yhat).toFixed(1)}`).join(' ');
  const bandPts = [
    ...forecast.map((p) => `${sx(p.step).toFixed(1)},${sy(p.yhat_upper ?? p.yhat).toFixed(1)}`),
    ...[...forecast].reverse().map((p) => `${sx(p.step).toFixed(1)},${sy(p.yhat_lower ?? p.yhat).toFixed(1)}`),
  ].join(' ');

  return (
    <InspectableBox title={title} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height, background: 'var(--bg)', borderRadius: 10 }}>
        {bandPts && <polygon points={bandPts} fill="rgba(56, 189, 248, 0.22)" />}
        <path d={histPath} fill="none" stroke="#94a3b8" strokeWidth={2} />
        <path d={forePath} fill="none" stroke="#38bdf8" strokeWidth={2.5} />
        <text x={pad.l} y={14} fill="#94a3b8" fontSize={10}>History (gray) · Deep Forecast &amp; Quantile Band (cyan)</text>
        <text x={6} y={pad.t + 6} fill="#64748b" fontSize={10}>{maxV.toFixed(2)}</text>
        <text x={6} y={H - pad.b} fill="#64748b" fontSize={10}>{minV.toFixed(2)}</text>
      </svg>
    </InspectableBox>
  );
};

export const HeatmapGrid: React.FC<{ columns: string[]; values: number[][]; title?: string }> = ({
  columns,
  values,
  title = 'Correlation Matrix Heatmap',
}) => {
  if (!columns.length) return null;
  const colorFor = (v: number) => {
    const clamped = Math.max(-1, Math.min(1, v));
    if (clamped >= 0) return `rgba(56, 189, 248, ${(0.12 + clamped * 0.78).toFixed(2)})`;
    return `rgba(248, 113, 113, ${(0.12 + Math.abs(clamped) * 0.78).toFixed(2)})`;
  };
  return (
    <InspectableBox title={title} subtitle={`${columns.length} × ${columns.length} Pearson matrix`} compact>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Feature</th>
              {columns.map((c) => <th key={c}>{c.slice(0, 12)}</th>)}
            </tr>
          </thead>
          <tbody>
            {columns.map((rowC, r) => (
              <tr key={rowC}>
                <td style={{ fontWeight: 700 }}>{rowC.slice(0, 16)}</td>
                {columns.map((colC, c) => {
                  const v = values[r]?.[c] ?? 0;
                  return (
                    <td key={colC} className="mono" style={{ background: colorFor(v), textAlign: 'center', fontWeight: 600 }}>
                      {v.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </InspectableBox>
  );
};

export const NetworkGraphSVG: React.FC<{
  nodes: { id: string; x: number; y: number; degree: number; community: number }[];
  edges: { source: string; target: string }[];
  predictedEdges?: { source: string; target: string }[];
  height?: number;
  title?: string;
}> = ({ nodes, edges, predictedEdges = [], height = 340, title = 'Graph Topology & Link Prediction' }) => {
  if (!nodes.length) return null;
  const W = 680;
  const H = height;
  const pad = 28;
  const xs = nodes.map((n) => n.x);
  const ys = nodes.map((n) => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = (v: number) => pad + ((v - minX) / (maxX - minX || 1)) * (W - 2 * pad);
  const sy = (v: number) => pad + ((v - minY) / (maxY - minY || 1)) * (H - 2 * pad);
  const map = new Map(nodes.map((n) => [n.id, { x: sx(n.x), y: sy(n.y), deg: n.degree, comm: n.community }]));

  return (
    <InspectableBox title={title} subtitle={`${nodes.length} nodes · ${edges.length} edges`} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height, background: 'var(--bg)', borderRadius: 10 }}>
        {edges.map((e, i) => {
          const a = map.get(e.source);
          const b = map.get(e.target);
          if (!a || !b) return null;
          return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="rgba(148,163,184,0.32)" strokeWidth={1.1} />;
        })}
        {predictedEdges.slice(0, 12).map((e, i) => {
          const a = map.get(String(e.source));
          const b = map.get(String(e.target));
          if (!a || !b) return null;
          return <line key={`p-${i}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#fbbf24" strokeWidth={2} strokeDasharray="4 3" />;
        })}
        {nodes.map((n) => {
          const pt = map.get(n.id)!;
          return (
            <g key={n.id}>
              <circle cx={pt.x} cy={pt.y} r={Math.min(12, 4.5 + Math.sqrt(n.degree) * 1.4)} fill={PALETTE[n.community % PALETTE.length]} stroke="#0f172a" strokeWidth={1.2}>
                <title>{`${n.id} · degree=${n.degree} · community=${n.community}`}</title>
              </circle>
            </g>
          );
        })}
        <text x={10} y={16} fill="#94a3b8" fontSize={10}>Solid: Observed Edges · Dashed Gold: Top Predicted Links</text>
      </svg>
    </InspectableBox>
  );
};

/**
 * Engineering Supervisory Trend Chart:
 * Actual Reality vs. Model Prediction vs. Baseline Expectation + Shaded Tolerance Corridor + Breach Markers.
 */
export const SupervisoryTrendSVG: React.FC<{
  points: {
    index?: number;
    label?: string;
    record_id?: string;
    actual?: number | null;
    predicted?: number | null;
    model_pred?: number;
    baseline?: number | null;
    baseline_pred?: number | null;
    tol_upper?: number | null;
    tol_lower?: number | null;
    severity?: string;
  }[];
  height?: number;
  title?: string;
}> = ({ points, height = 310, title = 'Actual vs. Model & Baseline Predictions (with Tolerance Corridor)' }) => {
  if (!points?.length) return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No trend points available.</div>;

  const vals: number[] = [];
  points.forEach((p) => {
    [p.actual, p.model_pred ?? p.predicted, p.baseline_pred ?? p.baseline, p.tol_upper, p.tol_lower].forEach((v) => {
      if (typeof v === 'number' && Number.isFinite(v)) vals.push(v);
    });
  });
  if (!vals.length) return null;

  const minV = Math.min(...vals);
  const maxV = Math.max(...vals);
  const padV = (maxV - minV) * 0.08 || 1;
  const lo = minV - padV;
  const hi = maxV + padV;

  const W = 780;
  const H = height;
  const pad = { l: 56, r: 20, t: 28, b: 34 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const sx = (i: number) => pad.l + (points.length === 1 ? pw / 2 : (i / (points.length - 1)) * pw);
  const sy = (v: number) => pad.t + ph - ((v - lo) / (hi - lo)) * ph;

  const buildPath = (getter: (p: (typeof points)[number]) => number | null | undefined) => {
    const segs: string[] = [];
    let started = false;
    points.forEach((p, i) => {
      const v = getter(p);
      if (typeof v === 'number' && Number.isFinite(v)) {
        segs.push(`${started ? 'L' : 'M'} ${sx(i).toFixed(1)} ${sy(v).toFixed(1)}`);
        started = true;
      } else {
        started = false;
      }
    });
    return segs.join(' ');
  };

  const upperPts = points
    .map((p, i) => (typeof p.tol_upper === 'number' && Number.isFinite(p.tol_upper) ? `${sx(i).toFixed(1)},${sy(p.tol_upper).toFixed(1)}` : null))
    .filter(Boolean);
  const lowerPts = [...points]
    .reverse()
    .map((p, ri) => {
      const i = points.length - 1 - ri;
      return typeof p.tol_lower === 'number' && Number.isFinite(p.tol_lower) ? `${sx(i).toFixed(1)},${sy(p.tol_lower).toFixed(1)}` : null;
    })
    .filter(Boolean);
  const corridor = upperPts.length === points.length && lowerPts.length === points.length ? [...upperPts, ...lowerPts].join(' ') : '';

  const modelPath = buildPath((p) => p.model_pred ?? p.predicted);
  const basePath = buildPath((p) => p.baseline_pred ?? p.baseline);
  const actualPath = buildPath((p) => p.actual);

  return (
    <InspectableBox title={title} subtitle={`${points.length} operational readings`} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height, background: 'var(--bg)', borderRadius: 10 }}>
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
          const y = pad.t + ph * frac;
          const v = hi - (hi - lo) * frac;
          return (
            <g key={frac}>
              <line x1={pad.l} y1={y} x2={W - pad.r} y2={y} stroke="rgba(148,163,184,0.16)" strokeWidth={1} />
              <text x={pad.l - 6} y={y + 3} textAnchor="end" fill="#94a3b8" fontSize={9.5}>{v.toFixed(1)}</text>
            </g>
          );
        })}

        {corridor && <polygon points={corridor} fill="rgba(52, 211, 153, 0.14)" />}
        {basePath && <path d={basePath} fill="none" stroke="#a78bfa" strokeWidth={1.7} strokeDasharray="5 4" />}
        {modelPath && <path d={modelPath} fill="none" stroke="#38bdf8" strokeWidth={2.2} />}
        {actualPath && <path d={actualPath} fill="none" stroke="#f8fafc" strokeWidth={1.8} opacity={0.92} />}

        {points.map((p, i) => {
          const modelVal = p.model_pred ?? p.predicted;
          const yVal = typeof p.actual === 'number' && Number.isFinite(p.actual) ? p.actual : modelVal;
          if (typeof yVal !== 'number' || !Number.isFinite(yVal)) return null;
          const sev = (p.severity || '').toUpperCase();
          if (sev === 'CRITICAL' || sev === 'BREACH') {
            return (
              <g key={i}>
                <circle cx={sx(i)} cy={sy(yVal)} r={5.2} fill="#ef4444" stroke="#ffffff" strokeWidth={1.4}>
                  <title>{`ALARM [${p.record_id ?? p.label ?? i}]: Actual=${p.actual ?? 'N/A'}, Model=${p.model_pred ?? p.predicted ?? 'N/A'}, Baseline=${p.baseline_pred ?? p.baseline ?? 'N/A'}`}</title>
                </circle>
              </g>
            );
          }
          if (sev === 'WARNING') {
            return (
              <circle key={i} cx={sx(i)} cy={sy(yVal)} r={4.2} fill="#fbbf24" stroke="#0f172a" strokeWidth={1}>
                <title>{`WARNING [${p.record_id ?? p.label ?? i}]: Actual=${p.actual ?? 'N/A'}, Model=${p.model_pred ?? p.predicted ?? 'N/A'}`}</title>
              </circle>
            );
          }
          return null;
        })}

        <g transform={`translate(${pad.l}, 15)`} fontSize={10}>
          <rect x={0} y={-8} width={10} height={8} fill="rgba(52,211,153,0.35)" />
          <text x={14} y={0} fill="#94a3b8">Tolerance Band</text>
          <line x1={102} y1={-4} x2={118} y2={-4} stroke="#f8fafc" strokeWidth={2} />
          <text x={122} y={0} fill="#94a3b8">Actual</text>
          <line x1={164} y1={-4} x2={180} y2={-4} stroke="#38bdf8" strokeWidth={2.2} />
          <text x={184} y={0} fill="#94a3b8">Model Pred</text>
          <line x1={254} y1={-4} x2={270} y2={-4} stroke="#a78bfa" strokeWidth={1.8} strokeDasharray="4 3" />
          <text x={274} y={0} fill="#94a3b8">Baseline</text>
          <circle cx={332} cy={-4} r={4} fill="#ef4444" />
          <text x={340} y={0} fill="#f87171">Tolerance Breach</text>
        </g>
      </svg>
    </InspectableBox>
  );
};

/**
 * Shewhart Statistical Process Control (SPC) Residual / Z-Score Chart with UCL / LCL & Warning Zones.
 */
export const SPCControlSVG: React.FC<{
  points: { index?: number; label?: string; record_id?: string; residual: number; ucl?: number | null; lcl?: number | null; z_score?: number; status?: string; severity?: string }[];
  limits?: { center_line?: number; ucl?: number; lcl?: number; uwl?: number; lwl?: number };
  height?: number;
  title?: string;
}> = ({ points, limits, height = 270, title = 'Shewhart SPC Residual Control Chart (±3σ Limits)' }) => {
  if (!points?.length) return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No SPC residuals available.</div>;

  const vals = points.map((p) => p.residual).filter(Number.isFinite);
  if (limits?.ucl !== undefined) vals.push(limits.ucl);
  if (limits?.lcl !== undefined) vals.push(limits.lcl);
  const minV = Math.min(...vals);
  const maxV = Math.max(...vals);
  const span = (maxV - minV) || 1;
  const lo = minV - span * 0.12;
  const hi = maxV + span * 0.12;

  const W = 780;
  const H = height;
  const pad = { l: 54, r: 64, t: 24, b: 30 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const sx = (i: number) => pad.l + (points.length === 1 ? pw / 2 : (i / (points.length - 1)) * pw);
  const sy = (v: number) => pad.t + ph - ((v - lo) / (hi - lo)) * ph;

  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(i).toFixed(1)} ${sy(p.residual).toFixed(1)}`).join(' ');

  return (
    <InspectableBox title={title} subtitle="Center Line (CL), Warning Limits (±W), Control Limits (UCL/LCL)" compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height, background: 'var(--bg)', borderRadius: 10 }}>
        {limits?.ucl !== undefined && (
          <>
            <line x1={pad.l} y1={sy(limits.ucl)} x2={W - pad.r} y2={sy(limits.ucl)} stroke="#ef4444" strokeWidth={1.6} strokeDasharray="6 4" />
            <text x={W - pad.r + 5} y={sy(limits.ucl) + 3} fill="#ef4444" fontSize={9.5}>UCL {limits.ucl.toFixed(1)}</text>
          </>
        )}
        {limits?.lcl !== undefined && (
          <>
            <line x1={pad.l} y1={sy(limits.lcl)} x2={W - pad.r} y2={sy(limits.lcl)} stroke="#ef4444" strokeWidth={1.6} strokeDasharray="6 4" />
            <text x={W - pad.r + 5} y={sy(limits.lcl) + 3} fill="#ef4444" fontSize={9.5}>LCL {limits.lcl.toFixed(1)}</text>
          </>
        )}
        {limits?.uwl !== undefined && (
          <line x1={pad.l} y1={sy(limits.uwl)} x2={W - pad.r} y2={sy(limits.uwl)} stroke="#fbbf24" strokeWidth={1.1} strokeDasharray="3 3" />
        )}
        {limits?.lwl !== undefined && (
          <line x1={pad.l} y1={sy(limits.lwl)} x2={W - pad.r} y2={sy(limits.lwl)} stroke="#fbbf24" strokeWidth={1.1} strokeDasharray="3 3" />
        )}
        <line x1={pad.l} y1={sy(limits?.center_line ?? 0)} x2={W - pad.r} y2={sy(limits?.center_line ?? 0)} stroke="#34d399" strokeWidth={1.3} />
        <text x={W - pad.r + 5} y={sy(limits?.center_line ?? 0) + 3} fill="#34d399" fontSize={9.5}>CL {(limits?.center_line ?? 0).toFixed(1)}</text>

        <path d={path} fill="none" stroke="#38bdf8" strokeWidth={1.8} />
        {points.map((p, i) => {
          const sev = (p.severity || p.status || '').toUpperCase();
          const col = sev === 'CRITICAL' || sev === 'BREACH' ? '#ef4444' : sev === 'WARNING' ? '#fbbf24' : '#38bdf8';
          const r = sev === 'CRITICAL' || sev === 'BREACH' ? 4.8 : sev === 'WARNING' ? 3.8 : 2.5;
          return (
            <circle key={i} cx={sx(i)} cy={sy(p.residual)} r={r} fill={col}>
              <title>{`[${p.record_id ?? p.label ?? i}] Residual=${Number.isFinite(p.residual) ? p.residual.toFixed(3) : 'N/A'}, Z=${typeof p.z_score === 'number' ? p.z_score.toFixed(2) : 'N/A'} (${sev || 'OK'})`}</title>
            </circle>
          );
        })}
      </svg>
    </InspectableBox>
  );
};

/**
 * Signed Diverging Tornado / Waterfall Attribution Chart for Studio 14 Root-Cause Diagnostics.
 * Positive impacts (+Δ Increase relative to Baseline) extend RIGHT in coral/red;
 * Negative impacts (-Δ Decrease relative to Baseline) extend LEFT in sky/blue.
 */
export const DivergingDriverBarSVG: React.FC<{
  title: string;
  subtitle?: string;
  items: { feature: string; signed_impact: number; z_shift?: number; live_mean?: number; normal_reference?: number }[];
}> = ({ title, subtitle, items }) => {
  const valid = (items || []).filter((it) => typeof it.signed_impact === 'number' && Number.isFinite(it.signed_impact)).slice(0, 12);
  if (!valid.length) return null;

  const maxAbs = Math.max(...valid.map((it) => Math.abs(it.signed_impact)), 1e-6);
  const rowH = 26;
  const W = 760;
  const pad = { l: 175, r: 115, t: 28, b: 18 };
  const H = pad.t + pad.b + valid.length * rowH;
  const plotW = W - pad.l - pad.r;
  const midX = pad.l + plotW / 2;

  return (
    <InspectableBox title={title} subtitle={subtitle || 'Signed feature contribution vs. Reference Baseline'} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10 }}>
        {/* Header labels */}
        <text x={midX - 8} y={16} textAnchor="end" fill="#38bdf8" fontSize={10.5} fontWeight={700}>
          &larr; DECREASE (−Δ below Baseline)
        </text>
        <text x={midX + 8} y={16} textAnchor="start" fill="#f87171" fontSize={10.5} fontWeight={700}>
          INCREASE (+Δ above Baseline) &rarr;
        </text>
        <line x1={midX} y1={22} x2={midX} y2={H - 10} stroke="rgba(148,163,184,0.55)" strokeWidth={1.2} strokeDasharray="3 3" />

        {valid.map((it, idx) => {
          const y = pad.t + idx * rowH;
          const pos = it.signed_impact >= 0;
          const barW = Math.max(3, (Math.abs(it.signed_impact) / maxAbs) * (plotW / 2 - 8));
          const x = pos ? midX : midX - barW;
          const color = pos ? '#f87171' : '#38bdf8';
          const shortFeat = it.feature.length > 22 ? `${it.feature.slice(0, 20)}…` : it.feature;
          const valLabel = `${it.signed_impact >= 0 ? '+' : ''}${it.signed_impact.toFixed(3)}`;
          const zTxt = typeof it.z_shift === 'number' && Number.isFinite(it.z_shift) ? ` (${it.z_shift >= 0 ? '+' : ''}${it.z_shift.toFixed(1)}σ)` : '';
          return (
            <g key={`${it.feature}-${idx}`}>
              <text x={pad.l - 8} y={y + 16} textAnchor="end" fill="#cbd5e1" fontSize={11} fontWeight={600}>
                {shortFeat}
              </text>
              <rect x={x} y={y + 5} width={barW} height={15} rx={4} fill={color} opacity={0.88}>
                <title>{`${it.feature}: Impact = ${valLabel}${zTxt}${it.live_mean !== undefined ? ` | Live=${it.live_mean} vs Ref=${it.normal_reference}` : ''}`}</title>
              </rect>
              <text
                x={W - 8}
                y={y + 16}
                textAnchor="end"
                fill={color}
                fontSize={10.5}
                fontWeight={700}
              >
                {valLabel}{zTxt}
              </text>
            </g>
          );
        })}
      </svg>
    </InspectableBox>
  );
};

