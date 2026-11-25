import React from 'react';
import { InspectableBox } from './InspectModal';

export type ProfessionalChartType =
  | 'gauge'
  | 'radar'
  | 'donut'
  | 'waterfall'
  | 'forest'
  | 'pareto'
  | 'ecdf'
  | 'violin_box'
  | 'qq_plot'
  | 'regression_ci'
  | 'roc_curve'
  | 'pr_curve'
  | 'calibration'
  | 'confusion_matrix'
  | 'gain_lift'
  | 'residual_diag'
  | 'bland_altman'
  | 'cusum'
  | 'lorenz'
  | 'parallel_coords'
  | 'ridgeline'
  | 'sankey'
  | 'funnel'
  | 'heatmap_svg'
  | 'stacked_bar';

export interface LiveChartPayload {
  chart_id: string;
  chart_type: ProfessionalChartType;
  category: string;
  studio_id: string;
  title: string;
  subtitle: string;
  provenance: string;
  kpis: Record<string, string | number | boolean | null>;
  data: Record<string, unknown>;
  table_columns: string[];
  table_rows: (string | number | boolean | null)[][];
}

const PALETTE = [
  '#38bdf8',
  '#a78bfa',
  '#34d399',
  '#fbbf24',
  '#f87171',
  '#fb7185',
  '#2dd4bf',
  '#818cf8',
  '#f97316',
  '#c084fc',
];

const num = (v: unknown, fallback = 0): number =>
  typeof v === 'number' && Number.isFinite(v) ? v : fallback;

const fmtNum = (v: number, digits = 3): string => {
  if (!Number.isFinite(v)) return '—';
  if (Math.abs(v) >= 10000) return v.toLocaleString(undefined, { maximumFractionDigits: 1 });
  if (Math.abs(v) > 0 && Math.abs(v) < 0.001) return v.toExponential(2);
  return Number(v.toFixed(digits)).toString();
};

/* ==========================================================================
   1. GaugeChart (Semi-Circular Radial Dial + Threshold Ticks)
   ========================================================================== */
export const GaugeChart: React.FC<{
  title: string;
  subtitle?: string;
  value: number;
  min?: number;
  max?: number;
  unit?: string;
  statusLabel?: string;
  thresholds?: { value: number; label: string; color: string }[];
}> = ({ title, subtitle, value, min = 0, max = 100, unit = '', statusLabel, thresholds = [] }) => {
  const span = Math.max(1e-6, max - min);
  const clamped = Math.max(min, Math.min(max, value));
  const frac = (clamped - min) / span;
  const W = 520;
  const H = 230;
  const cx = W / 2;
  const cy = H - 42;
  const R = 128;

  const arcPath = (f0: number, f1: number) => {
    const a0 = Math.PI * (1 - f0);
    const a1 = Math.PI * (1 - f1);
    const x0 = cx + R * Math.cos(a0);
    const y0 = cy - R * Math.sin(a0);
    const x1 = cx + R * Math.cos(a1);
    const y1 = cy - R * Math.sin(a1);
    const large = Math.abs(f1 - f0) > 0.5 ? 1 : 0;
    return `M ${x0.toFixed(1)} ${y0.toFixed(1)} A ${R} ${R} 0 ${large} 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`;
  };

  const arcColor = frac >= 0.85 ? '#34d399' : frac >= 0.6 ? '#38bdf8' : '#fbbf24';

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        <path d={arcPath(0, 1)} fill="none" stroke="rgba(148,163,184,0.24)" strokeWidth={22} strokeLinecap="round" />
        {frac > 0.004 && (
          <path d={arcPath(0, Math.max(0.015, frac))} fill="none" stroke={arcColor} strokeWidth={22} strokeLinecap="round">
            <title>{`${title}: ${fmtNum(value, 2)} ${unit}`}</title>
          </path>
        )}
        {thresholds.map((t, i) => {
          const tf = Math.max(0, Math.min(1, (t.value - min) / span));
          const a = Math.PI * (1 - tf);
          const x0 = cx + (R - 16) * Math.cos(a);
          const y0 = cy - (R - 16) * Math.sin(a);
          const x1 = cx + (R + 16) * Math.cos(a);
          const y1 = cy - (R + 16) * Math.sin(a);
          const lx = cx + (R + 30) * Math.cos(a);
          const ly = cy - (R + 26) * Math.sin(a);
          return (
            <g key={i}>
              <line x1={x0} y1={y0} x2={x1} y2={y1} stroke={t.color} strokeWidth={3.2} strokeLinecap="round">
                <title>{t.label}</title>
              </line>
              <text x={lx} y={ly} textAnchor={tf < 0.45 ? 'end' : tf > 0.55 ? 'start' : 'middle'} fill={t.color} fontSize={9.5} fontWeight={700}>
                {t.label}
              </text>
            </g>
          );
        })}
        <text x={cx - R} y={cy + 22} textAnchor="middle" fill="var(--text-dim)" fontSize={10} className="mono">{fmtNum(min, 1)}</text>
        <text x={cx + R} y={cy + 22} textAnchor="middle" fill="var(--text-dim)" fontSize={10} className="mono">{fmtNum(max, 1)}</text>
        <text x={cx} y={cy - 22} textAnchor="middle" fill="var(--text)" fontSize={36} fontWeight={800} className="mono">
          {fmtNum(value, 2)}{unit ? ` ${unit}` : ''}
        </text>
        {statusLabel && (
          <text x={cx} y={cy + 6} textAnchor="middle" fill="var(--text-muted)" fontSize={12} fontWeight={700}>
            {statusLabel}
          </text>
        )}
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   2. RadarChart (Multi-Axis Spider / Polar Polygon Chart)
   ========================================================================== */
export const RadarChart: React.FC<{
  title: string;
  subtitle?: string;
  axes: string[];
  series: { name: string; values: number[]; raw_values?: number[]; color?: string }[];
  maxScale?: number;
}> = ({ title, subtitle, axes, series, maxScale = 100 }) => {
  const n = axes.length;
  if (n < 3) return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>At least 3 axes required for RadarChart.</div>;
  const W = 680;
  const H = 340;
  const cx = W / 2 - 55;
  const cy = H / 2 + 6;
  const R = 118;
  const scale = Math.max(1e-6, maxScale);
  const angles = axes.map((_, i) => (Math.PI * 2 * i) / n - Math.PI / 2);
  const pt = (i: number, radius: number) => ({
    x: cx + Math.cos(angles[i]) * radius,
    y: cy + Math.sin(angles[i]) * radius,
  });

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {[0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <polygon
              points={axes.map((_, i) => {
                const p = pt(i, f * R);
                return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
              }).join(' ')}
              fill="none"
              stroke="rgba(148,163,184,0.22)"
              strokeWidth={1}
            />
            <text x={cx + 4} y={cy - f * R + 9} fill="var(--text-dim)" fontSize={8.5} className="mono">
              {Math.round(f * scale)}
            </text>
          </g>
        ))}
        {axes.map((ax, i) => {
          const p = pt(i, R);
          const lp = pt(i, R + 18);
          const cosA = Math.cos(angles[i]);
          const anchor = Math.abs(cosA) < 0.2 ? 'middle' : cosA > 0 ? 'start' : 'end';
          return (
            <g key={ax}>
              <line x1={cx} y1={cy} x2={p.x} y2={p.y} stroke="rgba(148,163,184,0.24)" strokeWidth={1} />
              <text x={lp.x} y={lp.y + 3} textAnchor={anchor} fill="var(--text-muted)" fontSize={10.5} fontWeight={600}>
                {ax.slice(0, 18)}
              </text>
            </g>
          );
        })}
        {series.map((s, sIdx) => {
          const col = s.color || PALETTE[sIdx % PALETTE.length];
          const polyPts = axes
            .map((_, i) => {
              const frac = Math.max(0, Math.min(1, num(s.values[i]) / scale));
              const p = pt(i, frac * R);
              return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
            })
            .join(' ');
          return (
            <g key={s.name}>
              <polygon
                points={polyPts}
                fill={col}
                fillOpacity={sIdx === 0 ? 0.22 : 0.08}
                stroke={col}
                strokeWidth={2.2}
                strokeDasharray={sIdx > 0 ? '5 3' : undefined}
              />
              {axes.map((ax, i) => {
                const v = num(s.values[i]);
                const rawV = s.raw_values?.[i];
                const p = pt(i, Math.max(0, Math.min(1, v / scale)) * R);
                return (
                  <circle key={i} cx={p.x} cy={p.y} r={3.4} fill={col}>
                    <title>{`${s.name} · ${ax}: ${fmtNum(v, 2)}${rawV !== undefined ? ` (raw: ${fmtNum(rawV, 4)})` : ''}`}</title>
                  </circle>
                );
              })}
            </g>
          );
        })}
        <g transform={`translate(${W - 185}, 34)`}>
          {series.map((s, idx) => {
            const col = s.color || PALETTE[idx % PALETTE.length];
            return (
              <g key={s.name} transform={`translate(0, ${idx * 22})`}>
                <rect x={0} y={-9} width={12} height={10} rx={2.5} fill={col} />
                <text x={18} y={0} fill="var(--text)" fontSize={10.5} fontWeight={600}>
                  {s.name.slice(0, 22)}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   3. DonutChart (Pure-SVG Ring Chart + Integrated Vector Legend)
   ========================================================================== */
export const DonutChart: React.FC<{
  title: string;
  subtitle?: string;
  slices: { label: string; value: number; pct: number; color?: string }[];
  centerValue?: string;
  centerLabel?: string;
}> = ({ title, subtitle, slices, centerValue, centerLabel }) => {
  const valid = slices.filter((s) => num(s.value) > 0);
  if (!valid.length) return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No positive segments to display.</div>;
  const total = valid.reduce((acc, s) => acc + num(s.value), 0) || 1;
  const W = 680;
  const H = Math.max(240, 48 + valid.length * 24);
  const cx = 150;
  const cy = H / 2;
  const rOut = 88;
  const rIn = 52;

  let curAngle = -Math.PI / 2;
  const arcs = valid.map((s, idx) => {
    const rawFrac = num(s.value) / total;
    const sweep = Math.min(Math.PI * 1.9998, rawFrac * Math.PI * 2);
    const a0 = curAngle;
    const a1 = curAngle + sweep;
    curAngle += rawFrac * Math.PI * 2;
    const x0 = cx + rOut * Math.cos(a0);
    const y0 = cy + rOut * Math.sin(a0);
    const x1 = cx + rOut * Math.cos(a1);
    const y1 = cy + rOut * Math.sin(a1);
    const ix1 = cx + rIn * Math.cos(a1);
    const iy1 = cy + rIn * Math.sin(a1);
    const ix0 = cx + rIn * Math.cos(a0);
    const iy0 = cy + rIn * Math.sin(a0);
    const large = sweep > Math.PI ? 1 : 0;
    const d = `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${rOut} ${rOut} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)} L ${ix1.toFixed(2)} ${iy1.toFixed(2)} A ${rIn} ${rIn} 0 ${large} 0 ${ix0.toFixed(2)} ${iy0.toFixed(2)} Z`;
    return { ...s, d, color: s.color || PALETTE[idx % PALETTE.length] };
  });

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {arcs.map((a, idx) => (
          <path key={idx} d={a.d} fill={a.color} stroke="var(--bg)" strokeWidth={1.8}>
            <title>{`${a.label}: ${fmtNum(a.value, 3)} (${a.pct.toFixed(1)}%)`}</title>
          </path>
        ))}
        <text x={cx} y={cy - 2} textAnchor="middle" fill="var(--text)" fontSize={20} fontWeight={800} className="mono">
          {centerValue ?? fmtNum(total, 1)}
        </text>
        {centerLabel && (
          <text x={cx} y={cy + 15} textAnchor="middle" fill="var(--text-dim)" fontSize={10.5}>
            {centerLabel}
          </text>
        )}
        <g transform="translate(295, 28)">
          {arcs.slice(0, 10).map((a, i) => (
            <g key={i} transform={`translate(0, ${i * 22})`}>
              <rect x={0} y={-9} width={11} height={11} rx={3} fill={a.color} />
              <text x={18} y={0} fill="var(--text)" fontSize={11} fontWeight={600}>
                {a.label.slice(0, 28)}
              </text>
              <text x={355} y={0} textAnchor="end" fill="var(--text-muted)" fontSize={11} className="mono">
                {fmtNum(a.value, 2)} ({a.pct.toFixed(1)}%)
              </text>
            </g>
          ))}
        </g>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   4. WaterfallChart (Signed Cumulative Bridge Waterfall)
   ========================================================================== */
export const WaterfallChart: React.FC<{
  title: string;
  subtitle?: string;
  bars: { label: string; delta: number; start: number; end: number; kind: 'start' | 'delta' | 'total' | string }[];
}> = ({ title, subtitle, bars }) => {
  if (!bars.length) return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No waterfall stages available.</div>;
  const W = 740;
  const H = 300;
  const pad = { l: 58, r: 20, t: 28, b: 64 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const allY = [0, ...bars.flatMap((b) => [num(b.start), num(b.end)])];
  const minV = Math.min(...allY);
  const maxV = Math.max(...allY);
  const span = maxV - minV || 1;
  const lo = minV - span * 0.08;
  const hi = maxV + span * 0.14;
  const sy = (v: number) => pad.t + ph - ((v - lo) / (hi - lo)) * ph;
  const bw = pw / bars.length;

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const gy = pad.t + ph * f;
          const gv = hi - (hi - lo) * f;
          return (
            <g key={f}>
              <line x1={pad.l} y1={gy} x2={W - pad.r} y2={gy} stroke="rgba(148,163,184,0.16)" strokeWidth={1} />
              <text x={pad.l - 6} y={gy + 3} textAnchor="end" fill="var(--text-dim)" fontSize={9.5} className="mono">
                {fmtNum(gv, 2)}
              </text>
            </g>
          );
        })}
        <line x1={pad.l} y1={sy(0)} x2={W - pad.r} y2={sy(0)} stroke="rgba(148,163,184,0.45)" strokeWidth={1.1} />
        {bars.map((b, i) => {
          const isTotal = b.kind === 'start' || b.kind === 'total';
          const topVal = isTotal ? Math.max(0, b.end) : Math.max(b.start, b.end);
          const botVal = isTotal ? Math.min(0, b.end) : Math.min(b.start, b.end);
          const yTop = sy(topVal);
          const yBot = sy(botVal);
          const hRect = Math.max(3, Math.abs(yBot - yTop));
          const x = pad.l + i * bw + Math.max(4, bw * 0.14);
          const wRect = Math.max(8, bw * 0.72);
          const col = isTotal ? '#38bdf8' : b.delta >= 0 ? '#34d399' : '#f87171';
          const valTxt = isTotal ? fmtNum(b.end, 3) : `${b.delta >= 0 ? '+' : ''}${fmtNum(b.delta, 3)}`;
          return (
            <g key={`${b.label}-${i}`}>
              {i < bars.length - 1 && (
                <line
                  x1={x + wRect}
                  y1={sy(b.end)}
                  x2={pad.l + (i + 1) * bw + Math.max(4, bw * 0.14)}
                  y2={sy(b.end)}
                  stroke="var(--text-dim)"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                />
              )}
              <rect x={x} y={yTop} width={wRect} height={hRect} rx={3.5} fill={col} opacity={0.9}>
                <title>{`${b.label}: ${valTxt} (Cumulative: ${fmtNum(b.end, 3)})`}</title>
              </rect>
              <text x={x + wRect / 2} y={yTop - 5} textAnchor="middle" fill="var(--text)" fontSize={9.5} fontWeight={700} className="mono">
                {valTxt}
              </text>
              <text
                x={x + wRect / 2}
                y={H - pad.b + 15}
                textAnchor="end"
                transform={`rotate(-24 ${x + wRect / 2} ${H - pad.b + 15})`}
                fill="var(--text-muted)"
                fontSize={10}
              >
                {b.label.slice(0, 16)}
              </text>
            </g>
          );
        })}
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   5. ForestChart (Effect Estimates + 95% Confidence Intervals + Null Line)
   ========================================================================== */
export const ForestChart: React.FC<{
  title: string;
  subtitle?: string;
  rows: { label: string; estimate: number; ci_low: number; ci_high: number; p_value?: number | null }[];
  referenceValue?: number;
  xLabel?: string;
}> = ({ title, subtitle, rows, referenceValue = 0, xLabel = 'Estimate (95% CI)' }) => {
  if (!rows.length) return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No effect estimates available.</div>;
  const rowH = 30;
  const pad = { l: 175, r: 195, t: 28, b: 36 };
  const W = 780;
  const H = pad.t + pad.b + rows.length * rowH;
  const pw = W - pad.l - pad.r;

  const allX = [referenceValue, ...rows.flatMap((r) => [num(r.ci_low), num(r.ci_high), num(r.estimate)])];
  const minX = Math.min(...allX);
  const maxX = Math.max(...allX);
  const span = maxX - minX || 1;
  const lo = minX - span * 0.08;
  const hi = maxX + span * 0.08;
  const sx = (v: number) => pad.l + ((v - lo) / (hi - lo)) * pw;

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const xv = lo + (hi - lo) * f;
          const gx = sx(xv);
          return (
            <g key={f}>
              <line x1={gx} y1={pad.t - 6} x2={gx} y2={H - pad.b} stroke="rgba(148,163,184,0.16)" strokeWidth={1} />
              <text x={gx} y={H - pad.b + 14} textAnchor="middle" fill="var(--text-dim)" fontSize={9.5} className="mono">
                {fmtNum(xv, 2)}
              </text>
            </g>
          );
        })}
        <line
          x1={sx(referenceValue)}
          y1={pad.t - 8}
          x2={sx(referenceValue)}
          y2={H - pad.b}
          stroke="var(--text-muted)"
          strokeDasharray="5 4"
          strokeWidth={1.4}
        />
        <text x={W - 10} y={17} textAnchor="end" fill="var(--text-dim)" fontSize={9.5} fontWeight={700}>
          ESTIMATE [95% CI]
        </text>
        {rows.map((r, i) => {
          const cy = pad.t + i * rowH + rowH / 2;
          const xLo = sx(r.ci_low);
          const xHi = sx(r.ci_high);
          const xEst = sx(r.estimate);
          const excludesNull = r.ci_low > referenceValue || r.ci_high < referenceValue;
          const col = excludesNull ? (r.estimate >= referenceValue ? '#38bdf8' : '#f87171') : '#94a3b8';
          const pTxt = typeof r.p_value === 'number' ? ` p=${r.p_value < 0.001 ? '<.001' : r.p_value.toFixed(3)}` : '';
          return (
            <g key={`${r.label}-${i}`}>
              <text x={pad.l - 10} y={cy + 4} textAnchor="end" fill="var(--text)" fontSize={11} fontWeight={600}>
                {r.label.slice(0, 23)}
              </text>
              <line x1={xLo} y1={cy} x2={xHi} y2={cy} stroke={col} strokeWidth={2.6} strokeLinecap="round" />
              <line x1={xLo} y1={cy - 4.5} x2={xLo} y2={cy + 4.5} stroke={col} strokeWidth={2} />
              <line x1={xHi} y1={cy - 4.5} x2={xHi} y2={cy + 4.5} stroke={col} strokeWidth={2} />
              <circle cx={xEst} cy={cy} r={5} fill={col} stroke="var(--bg)" strokeWidth={1.6}>
                <title>{`${r.label}: ${fmtNum(r.estimate, 4)} [${fmtNum(r.ci_low, 4)}, ${fmtNum(r.ci_high, 4)}]${pTxt}`}</title>
              </circle>
              <text x={W - 10} y={cy + 4} textAnchor="end" fill="var(--text-muted)" fontSize={10} className="mono">
                {`${fmtNum(r.estimate, 3)} [${fmtNum(r.ci_low, 2)}, ${fmtNum(r.ci_high, 2)}]`}
              </text>
            </g>
          );
        })}
        <text x={pad.l + pw / 2} y={H - 5} textAnchor="middle" fill="var(--text-dim)" fontSize={10}>
          {xLabel}
        </text>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   6. ParetoChart (Dual-Axis 80/20 Bars + Cumulative Curve + Cutoff Line)
   ========================================================================== */
export const ParetoChart: React.FC<{
  title: string;
  subtitle?: string;
  items: { label: string; value: number; share_pct: number; cumulative_pct: number }[];
  cutoffPct?: number;
  yLeftLabel?: string;
}> = ({ title, subtitle, items, cutoffPct = 80, yLeftLabel = 'Magnitude' }) => {
  if (!items.length) return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No Pareto items available.</div>;
  const W = 740;
  const H = 310;
  const pad = { l: 56, r: 52, t: 24, b: 64 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const maxVal = Math.max(...items.map((it) => num(it.value)), 1e-6);
  const bw = pw / items.length;
  const syLeft = (v: number) => pad.t + ph - (v / maxVal) * ph;
  const syRight = (pct: number) => pad.t + ph - (Math.min(100, Math.max(0, pct)) / 100) * ph;

  const cumPath = items
    .map((it, idx) => `${idx === 0 ? 'M' : 'L'} ${(pad.l + bw * (idx + 0.5)).toFixed(1)} ${syRight(it.cumulative_pct).toFixed(1)}`)
    .join(' ');

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {[0, 25, 50, 75, 100].map((pct) => {
          const gy = syRight(pct);
          const lv = (maxVal * pct) / 100;
          return (
            <g key={pct}>
              <line x1={pad.l} y1={gy} x2={W - pad.r} y2={gy} stroke="rgba(148,163,184,0.16)" strokeWidth={1} />
              <text x={pad.l - 6} y={gy + 3} textAnchor="end" fill="var(--text-dim)" fontSize={9.5} className="mono">{fmtNum(lv, 2)}</text>
              <text x={W - pad.r + 6} y={gy + 3} textAnchor="start" fill="#fbbf24" fontSize={9.5} className="mono">{pct}%</text>
            </g>
          );
        })}
        <line x1={pad.l} y1={syRight(cutoffPct)} x2={W - pad.r} y2={syRight(cutoffPct)} stroke="#34d399" strokeWidth={1.5} strokeDasharray="5 4" />
        <text x={W - pad.r - 6} y={syRight(cutoffPct) - 5} textAnchor="end" fill="#34d399" fontSize={9.5} fontWeight={700}>
          {cutoffPct}% Vital-Few Threshold
        </text>
        {items.map((it, i) => {
          const x = pad.l + i * bw + Math.max(3, bw * 0.14);
          const wBar = Math.max(6, bw * 0.72);
          const yTop = syLeft(it.value);
          const hBar = Math.max(2, pad.t + ph - yTop);
          return (
            <g key={`${it.label}-${i}`}>
              <rect x={x} y={yTop} width={wBar} height={hBar} rx={3} fill={it.cumulative_pct <= cutoffPct + 1e-6 || i === 0 ? '#38bdf8' : '#64748b'} opacity={0.88}>
                <title>{`${it.label}: ${fmtNum(it.value, 4)} (${it.share_pct.toFixed(1)}%, Cumulative ${it.cumulative_pct.toFixed(1)}%)`}</title>
              </rect>
              <text
                x={pad.l + bw * (i + 0.5)}
                y={H - pad.b + 14}
                textAnchor="end"
                transform={`rotate(-25 ${pad.l + bw * (i + 0.5)} ${H - pad.b + 14})`}
                fill="var(--text-muted)"
                fontSize={9.5}
              >
                {it.label.slice(0, 15)}
              </text>
            </g>
          );
        })}
        <path d={cumPath} fill="none" stroke="#fbbf24" strokeWidth={2.4} />
        {items.map((it, i) => (
          <circle key={i} cx={pad.l + bw * (i + 0.5)} cy={syRight(it.cumulative_pct)} r={3.5} fill="#fbbf24">
            <title>{`Cumulative @ ${it.label}: ${it.cumulative_pct.toFixed(1)}%`}</title>
          </circle>
        ))}
        <text x={14} y={pad.t + ph / 2} textAnchor="middle" fill="var(--text-dim)" fontSize={10} transform={`rotate(-90 14 ${pad.t + ph / 2})`}>
          {yLeftLabel}
        </text>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   7. ECDFChart (Empirical Cumulative Step Distribution + KS Comparison)
   ========================================================================== */
export const ECDFChart: React.FC<{
  title: string;
  subtitle?: string;
  series: { name: string; x: number[]; y: number[]; n: number; median: number; color?: string }[];
  xLabel?: string;
}> = ({ title, subtitle, series, xLabel = 'Value' }) => {
  const valid = series.filter((s) => s.x.length > 0);
  if (!valid.length) return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No ECDF series available.</div>;
  const W = 740;
  const H = 300;
  const pad = { l: 52, r: 20, t: 28, b: 38 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const allX = valid.flatMap((s) => s.x);
  const minX = Math.min(...allX);
  const maxX = Math.max(...allX);
  const spanX = maxX - minX || 1;
  const sx = (v: number) => pad.l + ((v - minX) / spanX) * pw;
  const sy = (p: number) => pad.t + ph - p * ph;

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((p) => (
          <g key={p}>
            <line x1={pad.l} y1={sy(p)} x2={W - pad.r} y2={sy(p)} stroke="rgba(148,163,184,0.18)" strokeDasharray={p === 0.5 ? '4 3' : undefined} />
            <text x={pad.l - 6} y={sy(p) + 3} textAnchor="end" fill="var(--text-dim)" fontSize={9.5} className="mono">{p.toFixed(2)}</text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const xv = minX + spanX * f;
          return (
            <text key={f} x={pad.l + pw * f} y={H - 16} textAnchor="middle" fill="var(--text-dim)" fontSize={9.5} className="mono">
              {fmtNum(xv, 2)}
            </text>
          );
        })}
        {valid.map((s, idx) => {
          const col = s.color || PALETTE[idx % PALETTE.length];
          const cmds: string[] = [];
          s.x.forEach((xv, i) => {
            const px = sx(xv).toFixed(1);
            const py = sy(s.y[i]).toFixed(1);
            if (i === 0) cmds.push(`M ${px} ${sy(0).toFixed(1)} L ${px} ${py}`);
            else cmds.push(`H ${px} V ${py}`);
          });
          return (
            <g key={s.name}>
              <path d={cmds.join(' ')} fill="none" stroke={col} strokeWidth={2.3}>
                <title>{`${s.name} (n=${s.n}, median=${fmtNum(s.median, 3)})`}</title>
              </path>
            </g>
          );
        })}
        <g transform={`translate(${pad.l}, 15)`}>
          {valid.map((s, idx) => (
            <g key={s.name} transform={`translate(${idx * 170}, 0)`}>
              <line x1={0} y1={-4} x2={15} y2={-4} stroke={s.color || PALETTE[idx % PALETTE.length]} strokeWidth={2.5} />
              <text x={20} y={0} fill="var(--text-muted)" fontSize={10}>
                {`${s.name.slice(0, 18)} (n=${s.n})`}
              </text>
            </g>
          ))}
        </g>
        <text x={pad.l + pw / 2} y={H - 3} textAnchor="middle" fill="var(--text-dim)" fontSize={10}>{xLabel}</text>
        <text x={14} y={pad.t + ph / 2} textAnchor="middle" fill="var(--text-dim)" fontSize={10} transform={`rotate(-90 14 ${pad.t + ph / 2})`}>
          Empirical CDF F(x)
        </text>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   8. ViolinBoxChart (Gaussian KDE Violin + Inner Tukey Box-Whisker + Outliers)
   ========================================================================== */
export const ViolinBoxChart: React.FC<{
  title: string;
  subtitle?: string;
  groups: {
    label: string;
    n: number;
    min: number;
    q25: number;
    median: number;
    mean: number;
    q75: number;
    max: number;
    whisker_low: number;
    whisker_high: number;
    outliers: number[];
    kde_y: number[];
    kde_density: number[];
    color?: string;
  }[];
  yLabel?: string;
}> = ({ title, subtitle, groups, yLabel = 'Value' }) => {
  if (!groups.length) return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No distributions available.</div>;
  const W = 740;
  const H = 310;
  const pad = { l: 56, r: 20, t: 20, b: 48 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const allY = groups.flatMap((g) => [g.min, g.max, ...g.kde_y]);
  const lo = Math.min(...allY);
  const hi = Math.max(...allY);
  const span = hi - lo || 1;
  const sy = (v: number) => pad.t + ph - ((v - lo) / span) * ph;
  const gw = pw / groups.length;

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const gv = hi - span * f;
          const gy = pad.t + ph * f;
          return (
            <g key={f}>
              <line x1={pad.l} y1={gy} x2={W - pad.r} y2={gy} stroke="rgba(148,163,184,0.16)" />
              <text x={pad.l - 6} y={gy + 3} textAnchor="end" fill="var(--text-dim)" fontSize={9.5} className="mono">{fmtNum(gv, 2)}</text>
            </g>
          );
        })}
        {groups.map((g, idx) => {
          const cx = pad.l + gw * (idx + 0.5);
          const halfW = Math.min(44, gw * 0.38);
          const col = g.color || PALETTE[idx % PALETTE.length];
          const leftSide = g.kde_y.map((yv, k) => `${(cx - g.kde_density[k] * halfW).toFixed(1)},${sy(yv).toFixed(1)}`);
          const rightSide = [...g.kde_y].reverse().map((yv, rk) => {
            const k = g.kde_y.length - 1 - rk;
            return `${(cx + g.kde_density[k] * halfW).toFixed(1)},${sy(yv).toFixed(1)}`;
          });
          const boxTop = sy(g.q75);
          const boxBot = sy(g.q25);
          return (
            <g key={`${g.label}-${idx}`}>
              <polygon points={[...leftSide, ...rightSide].join(' ')} fill={col} fillOpacity={0.24} stroke={col} strokeWidth={1.5}>
                <title>{`${g.label} (n=${g.n}): Median=${fmtNum(g.median)}, IQR=[${fmtNum(g.q25)}, ${fmtNum(g.q75)}], Mean=${fmtNum(g.mean)}`}</title>
              </polygon>
              <line x1={cx} y1={sy(g.whisker_high)} x2={cx} y2={sy(g.whisker_low)} stroke="var(--text)" strokeWidth={1.3} />
              <rect x={cx - 7} y={boxTop} width={14} height={Math.max(2, boxBot - boxTop)} rx={2} fill={col} fillOpacity={0.78} stroke="var(--bg)" strokeWidth={1} />
              <line x1={cx - 9} y1={sy(g.median)} x2={cx + 9} y2={sy(g.median)} stroke="#fbbf24" strokeWidth={2.5} />
              <circle cx={cx} cy={sy(g.mean)} r={2.8} fill="var(--bg)" stroke="var(--text)" strokeWidth={1.3} />
              {g.outliers.map((o, oi) => (
                <circle key={oi} cx={cx} cy={sy(o)} r={2.3} fill="#f87171" opacity={0.8}>
                  <title>{`Outlier (${g.label}): ${fmtNum(o)}`}</title>
                </circle>
              ))}
              <text x={cx} y={H - 22} textAnchor="middle" fill="var(--text)" fontSize={10.5} fontWeight={600}>
                {g.label.slice(0, 15)}
              </text>
              <text x={cx} y={H - 9} textAnchor="middle" fill="var(--text-dim)" fontSize={9} className="mono">
                n={g.n}
              </text>
            </g>
          );
        })}
        <text x={14} y={pad.t + ph / 2} textAnchor="middle" fill="var(--text-dim)" fontSize={10} transform={`rotate(-90 14 ${pad.t + ph / 2})`}>
          {yLabel}
        </text>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   9. QQPlotChart (Normal Q-Q Plot + 95% Order-Statistic Confidence Envelope)
   ========================================================================== */
export const QQPlotChart: React.FC<{
  title: string;
  subtitle?: string;
  points: { theo: number; emp: number; ref: number; lower: number; upper: number }[];
  variableName?: string;
}> = ({ title, subtitle, points, variableName = 'Sample' }) => {
  if (!points.length) return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No quantile points available.</div>;
  const W = 720;
  const H = 310;
  const pad = { l: 56, r: 22, t: 22, b: 38 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const xs = points.map((p) => p.theo);
  const ys = points.flatMap((p) => [p.emp, p.lower, p.upper]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = (v: number) => pad.l + ((v - minX) / (maxX - minX || 1)) * pw;
  const sy = (v: number) => pad.t + ph - ((v - minY) / (maxY - minY || 1)) * ph;

  const bandPoly = [
    ...points.map((p) => `${sx(p.theo).toFixed(1)},${sy(p.upper).toFixed(1)}`),
    ...[...points].reverse().map((p) => `${sx(p.theo).toFixed(1)},${sy(p.lower).toFixed(1)}`),
  ].join(' ');
  const refPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.theo).toFixed(1)} ${sy(p.ref).toFixed(1)}`).join(' ');

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const gy = pad.t + ph * f;
          const gv = maxY - (maxY - minY) * f;
          return (
            <g key={f}>
              <line x1={pad.l} y1={gy} x2={W - pad.r} y2={gy} stroke="rgba(148,163,184,0.16)" />
              <text x={pad.l - 6} y={gy + 3} textAnchor="end" fill="var(--text-dim)" fontSize={9.5} className="mono">{fmtNum(gv, 2)}</text>
            </g>
          );
        })}
        <polygon points={bandPoly} fill="#38bdf8" fillOpacity={0.16} />
        <path d={refPath} fill="none" stroke="#f87171" strokeWidth={1.8} strokeDasharray="5 4" />
        {points.map((p, i) => (
          <circle key={i} cx={sx(p.theo)} cy={sy(p.emp)} r={2.8} fill="#38bdf8" opacity={0.85}>
            <title>{`Theoretical z=${p.theo.toFixed(3)}, Empirical=${fmtNum(p.emp, 4)}`}</title>
          </circle>
        ))}
        <text x={pad.l + pw / 2} y={H - 6} textAnchor="middle" fill="var(--text-dim)" fontSize={10}>Theoretical Standard Normal Quantiles (z)</text>
        <text x={14} y={pad.t + ph / 2} textAnchor="middle" fill="var(--text-dim)" fontSize={10} transform={`rotate(-90 14 ${pad.t + ph / 2})`}>
          {`Empirical Quantiles (${variableName})`}
        </text>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   10. JointRegressionChart (Scatter + OLS Fit + 95% CI & 95% PI Fan)
   ========================================================================== */
export const JointRegressionChart: React.FC<{
  title: string;
  subtitle?: string;
  points: { x: number; y: number; group: string }[];
  band: { x: number; yhat: number; ci_low: number; ci_high: number; pi_low: number; pi_high: number }[];
  xLabel: string;
  yLabel: string;
}> = ({ title, subtitle, points, band, xLabel, yLabel }) => {
  if (!points.length) return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>No paired observations available.</div>;
  const W = 740;
  const H = 320;
  const pad = { l: 58, r: 22, t: 28, b: 38 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const xs = [...points.map((p) => p.x), ...band.map((b) => b.x)];
  const ys = [...points.map((p) => p.y), ...band.flatMap((b) => [b.pi_low, b.pi_high])];
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = (v: number) => pad.l + ((v - minX) / (maxX - minX || 1)) * pw;
  const sy = (v: number) => pad.t + ph - ((v - minY) / (maxY - minY || 1)) * ph;

  const piPoly = band.length
    ? [...band.map((b) => `${sx(b.x).toFixed(1)},${sy(b.pi_high).toFixed(1)}`), ...[...band].reverse().map((b) => `${sx(b.x).toFixed(1)},${sy(b.pi_low).toFixed(1)}`)].join(' ')
    : '';
  const ciPoly = band.length
    ? [...band.map((b) => `${sx(b.x).toFixed(1)},${sy(b.ci_high).toFixed(1)}`), ...[...band].reverse().map((b) => `${sx(b.x).toFixed(1)},${sy(b.ci_low).toFixed(1)}`)].join(' ')
    : '';
  const linePath = band.map((b, i) => `${i === 0 ? 'M' : 'L'} ${sx(b.x).toFixed(1)} ${sy(b.yhat).toFixed(1)}`).join(' ');

  const groups = Array.from(new Set(points.map((p) => p.group)));
  const gColor = new Map(groups.map((g, i) => [g, PALETTE[i % PALETTE.length]]));

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const gy = pad.t + ph * f;
          const gv = maxY - (maxY - minY) * f;
          return (
            <g key={f}>
              <line x1={pad.l} y1={gy} x2={W - pad.r} y2={gy} stroke="rgba(148,163,184,0.16)" />
              <text x={pad.l - 6} y={gy + 3} textAnchor="end" fill="var(--text-dim)" fontSize={9.5} className="mono">{fmtNum(gv, 2)}</text>
            </g>
          );
        })}
        {piPoly && <polygon points={piPoly} fill="rgba(148,163,184,0.14)" />}
        {ciPoly && <polygon points={ciPoly} fill="rgba(56,189,248,0.24)" />}
        {linePath && <path d={linePath} fill="none" stroke="#f87171" strokeWidth={2.3} />}
        {points.map((p, i) => (
          <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={3.1} fill={gColor.get(p.group) || '#38bdf8'} opacity={0.82}>
            <title>{`${p.group}: (${fmtNum(p.x)}, ${fmtNum(p.y)})`}</title>
          </circle>
        ))}
        <g transform={`translate(${pad.l}, 15)`} fontSize={10}>
          <rect x={0} y={-8} width={10} height={8} fill="rgba(56,189,248,0.35)" />
          <text x={14} y={0} fill="var(--text-muted)">95% Mean CI</text>
          <rect x={92} y={-8} width={10} height={8} fill="rgba(148,163,184,0.28)" />
          <text x={106} y={0} fill="var(--text-muted)">95% Prediction Band</text>
          <line x1={222} y1={-4} x2={238} y2={-4} stroke="#f87171" strokeWidth={2.3} />
          <text x={242} y={0} fill="var(--text-muted)">OLS Fit</text>
        </g>
        <text x={pad.l + pw / 2} y={H - 6} textAnchor="middle" fill="var(--text-dim)" fontSize={10}>{xLabel}</text>
        <text x={14} y={pad.t + ph / 2} textAnchor="middle" fill="var(--text-dim)" fontSize={10} transform={`rotate(-90 14 ${pad.t + ph / 2})`}>{yLabel}</text>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   11. RocCurveChart (ROC + Shaded AUC + Youden Optimal Point)
   ========================================================================== */
export const RocCurveChart: React.FC<{
  title: string;
  subtitle?: string;
  points: { fpr: number; tpr: number; threshold: number }[];
  auc: number;
  optimalPoint?: { fpr: number; tpr: number; threshold: number };
}> = ({ title, subtitle, points, auc, optimalPoint }) => {
  const W = 680;
  const H = 310;
  const pad = { l: 52, r: 24, t: 22, b: 38 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const sx = (v: number) => pad.l + v * pw;
  const sy = (v: number) => pad.t + ph - v * ph;

  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.fpr).toFixed(1)} ${sy(p.tpr).toFixed(1)}`).join(' ');
  const area = `${d} L ${sx(1).toFixed(1)} ${sy(0).toFixed(1)} L ${sx(0).toFixed(1)} ${sy(0).toFixed(1)} Z`;

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line x1={pad.l} y1={sy(f)} x2={W - pad.r} y2={sy(f)} stroke="rgba(148,163,184,0.16)" />
            <text x={pad.l - 6} y={sy(f) + 3} textAnchor="end" fill="var(--text-dim)" fontSize={9.5} className="mono">{f.toFixed(2)}</text>
            <text x={sx(f)} y={H - 18} textAnchor="middle" fill="var(--text-dim)" fontSize={9.5} className="mono">{f.toFixed(2)}</text>
          </g>
        ))}
        <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="var(--text-dim)" strokeDasharray="5 4" strokeWidth={1.3} />
        <path d={area} fill="#38bdf8" fillOpacity={0.15} />
        <path d={d} fill="none" stroke="#38bdf8" strokeWidth={2.6} />
        {optimalPoint && (
          <g>
            <line x1={sx(optimalPoint.fpr)} y1={sy(0)} x2={sx(optimalPoint.fpr)} y2={sy(optimalPoint.tpr)} stroke="#f87171" strokeDasharray="3 3" />
            <circle cx={sx(optimalPoint.fpr)} cy={sy(optimalPoint.tpr)} r={5.5} fill="#f87171" stroke="var(--bg)" strokeWidth={1.8}>
              <title>{`Youden Optimal Threshold = ${optimalPoint.threshold}: TPR=${optimalPoint.tpr}, FPR=${optimalPoint.fpr}`}</title>
            </circle>
          </g>
        )}
        <text x={W - pad.r - 12} y={H - pad.b - 16} textAnchor="end" fill="#38bdf8" fontSize={12.5} fontWeight={800} className="mono">
          ROC-AUC = {auc.toFixed(4)}
        </text>
        <text x={pad.l + pw / 2} y={H - 4} textAnchor="middle" fill="var(--text-dim)" fontSize={10}>False Positive Rate (1 − Specificity)</text>
        <text x={14} y={pad.t + ph / 2} textAnchor="middle" fill="var(--text-dim)" fontSize={10} transform={`rotate(-90 14 ${pad.t + ph / 2})`}>True Positive Rate (Sensitivity)</text>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   12. PRCurveChart (Precision-Recall + Iso-F1 Contours + Optimal F1)
   ========================================================================== */
export const PRCurveChart: React.FC<{
  title: string;
  subtitle?: string;
  points: { recall: number; precision: number }[];
  ap: number;
  prevalence: number;
  optimalPoint?: { recall: number; precision: number; f1: number };
}> = ({ title, subtitle, points, ap, prevalence, optimalPoint }) => {
  const W = 680;
  const H = 310;
  const pad = { l: 52, r: 24, t: 22, b: 38 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const sx = (v: number) => pad.l + v * pw;
  const sy = (v: number) => pad.t + ph - v * ph;
  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.recall).toFixed(1)} ${sy(p.precision).toFixed(1)}`).join(' ');

  const isoF1Paths = [0.2, 0.4, 0.6, 0.8].map((f1) => {
    const segs: string[] = [];
    for (let k = 1; k <= 40; k++) {
      const r = (f1 / 1.98) + ((1 - f1 / 1.98) * k) / 40;
      const p = (f1 * r) / Math.max(1e-6, 2 * r - f1);
      if (p >= 0 && p <= 1) {
        segs.push(`${segs.length === 0 ? 'M' : 'L'} ${sx(r).toFixed(1)} ${sy(p).toFixed(1)}`);
      }
    }
    return { f1, d: segs.join(' ') };
  });

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line x1={pad.l} y1={sy(f)} x2={W - pad.r} y2={sy(f)} stroke="rgba(148,163,184,0.14)" />
            <text x={pad.l - 6} y={sy(f) + 3} textAnchor="end" fill="var(--text-dim)" fontSize={9.5} className="mono">{f.toFixed(2)}</text>
            <text x={sx(f)} y={H - 18} textAnchor="middle" fill="var(--text-dim)" fontSize={9.5} className="mono">{f.toFixed(2)}</text>
          </g>
        ))}
        {isoF1Paths.map((iso) => (
          <path key={iso.f1} d={iso.d} fill="none" stroke="rgba(148,163,184,0.28)" strokeDasharray="2 3" />
        ))}
        <line x1={sx(0)} y1={sy(prevalence)} x2={sx(1)} y2={sy(prevalence)} stroke="var(--text-dim)" strokeDasharray="5 4" />
        <path d={d} fill="none" stroke="#fbbf24" strokeWidth={2.6} />
        {optimalPoint && (
          <circle cx={sx(optimalPoint.recall)} cy={sy(optimalPoint.precision)} r={5.5} fill="#34d399" stroke="var(--bg)" strokeWidth={1.8}>
            <title>{`Peak F1 = ${optimalPoint.f1}: Precision=${optimalPoint.precision}, Recall=${optimalPoint.recall}`}</title>
          </circle>
        )}
        <text x={pad.l + 12} y={H - pad.b - 14} fill="#fbbf24" fontSize={12.5} fontWeight={800} className="mono">
          PR-AUC (AP) = {ap.toFixed(4)}
        </text>
        <text x={pad.l + pw / 2} y={H - 4} textAnchor="middle" fill="var(--text-dim)" fontSize={10}>Recall</text>
        <text x={14} y={pad.t + ph / 2} textAnchor="middle" fill="var(--text-dim)" fontSize={10} transform={`rotate(-90 14 ${pad.t + ph / 2})`}>Precision</text>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   13. CalibrationCurveChart (Reliability Diagram + Wilson CI + Mini Histogram)
   ========================================================================== */
export const CalibrationCurveChart: React.FC<{
  title: string;
  subtitle?: string;
  bins: { bin_center: number; mean_pred: number | null; frac_pos: number | null; ci_low: number | null; ci_high: number | null; count: number }[];
  ece: number;
  brier: number;
}> = ({ title, subtitle, bins, ece, brier }) => {
  const W = 680;
  const H = 330;
  const pad = { l: 52, r: 24, t: 20, b: 36 };
  const pw = W - pad.l - pad.r;
  const topH = 195;
  const histTop = pad.t + topH + 22;
  const histH = H - histTop - pad.b;

  const sx = (v: number) => pad.l + v * pw;
  const sy = (v: number) => pad.t + topH - v * topH;
  const nonEmpty = bins.filter((b) => b.count > 0 && b.mean_pred !== null && b.frac_pos !== null);
  const maxCnt = Math.max(...bins.map((b) => b.count), 1);
  const d = nonEmpty.map((b, i) => `${i === 0 ? 'M' : 'L'} ${sx(b.mean_pred!).toFixed(1)} ${sy(b.frac_pos!).toFixed(1)}`).join(' ');

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={pad.l} y1={sy(f)} x2={W - pad.r} y2={sy(f)} stroke="rgba(148,163,184,0.16)" />
            <text x={pad.l - 6} y={sy(f) + 3} textAnchor="end" fill="var(--text-dim)" fontSize={9.5} className="mono">{f.toFixed(1)}</text>
          </g>
        ))}
        <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="var(--text-dim)" strokeDasharray="5 4" strokeWidth={1.3} />
        <path d={d} fill="none" stroke="#a78bfa" strokeWidth={2.4} />
        {nonEmpty.map((b, i) => (
          <g key={i}>
            {b.ci_low !== null && b.ci_high !== null && (
              <line x1={sx(b.mean_pred!)} y1={sy(b.ci_low)} x2={sx(b.mean_pred!)} y2={sy(b.ci_high)} stroke="#a78bfa" strokeWidth={1.5} opacity={0.75} />
            )}
            <circle cx={sx(b.mean_pred!)} cy={sy(b.frac_pos!)} r={4.2} fill="#a78bfa" stroke="var(--bg)" strokeWidth={1.4}>
              <title>{`Predicted=${b.mean_pred!.toFixed(3)}, Empirical=${b.frac_pos!.toFixed(3)} (n=${b.count})`}</title>
            </circle>
          </g>
        ))}
        <text x={pad.l + 10} y={pad.t + 16} fill="#a78bfa" fontSize={11} fontWeight={700} className="mono">
          ECE = {ece.toFixed(4)} · Brier = {brier.toFixed(4)}
        </text>
        {bins.map((b, i) => {
          const bw = pw / bins.length;
          const bh = (b.count / maxCnt) * histH;
          return (
            <rect key={i} x={pad.l + i * bw + 2} y={histTop + histH - bh} width={Math.max(3, bw - 4)} height={bh} rx={2} fill="#38bdf8" opacity={0.65}>
              <title>{`Bin ${b.bin_center}: n=${b.count}`}</title>
            </rect>
          );
        })}
        <text x={pad.l + pw / 2} y={H - 6} textAnchor="middle" fill="var(--text-dim)" fontSize={10}>Mean Predicted Probability (Bottom: Bin Sample Counts)</text>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   14. ConfusionMatrixChart & 24. HeatmapMatrixSVG (Pure Vector Matrix SVGs)
   ========================================================================== */
export const ConfusionMatrixChart: React.FC<{
  title: string;
  subtitle?: string;
  labels: string[];
  counts: number[][];
  rowPcts: number[][];
}> = ({ title, subtitle, labels, counts, rowPcts }) => {
  const n = labels.length;
  if (!n) return null;
  const cell = Math.max(42, Math.min(72, Math.floor(320 / n)));
  const pad = { l: 110, r: 28, t: 36, b: 64 };
  const W = Math.max(480, pad.l + pad.r + n * cell);
  const H = pad.t + pad.b + n * cell;

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: Math.min(420, H), background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {labels.map((lbl, i) => (
          <g key={lbl}>
            <text x={pad.l - 8} y={pad.t + i * cell + cell / 2 + 4} textAnchor="end" fill="var(--text)" fontSize={10.5} fontWeight={700}>
              {lbl.slice(0, 13)}
            </text>
            <text
              x={pad.l + i * cell + cell / 2}
              y={pad.t + n * cell + 16}
              textAnchor="end"
              transform={`rotate(-25 ${pad.l + i * cell + cell / 2} ${pad.t + n * cell + 16})`}
              fill="var(--text)"
              fontSize={10.5}
              fontWeight={700}
            >
              {lbl.slice(0, 13)}
            </text>
          </g>
        ))}
        {counts.map((row, r) =>
          row.map((cnt, c) => {
            const pct = rowPcts[r]?.[c] ?? 0;
            const alpha = 0.12 + 0.78 * (pct / 100);
            const fill = r === c ? `rgba(56, 189, 248, ${alpha.toFixed(2)})` : `rgba(248, 113, 113, ${(0.08 + 0.68 * (pct / 100)).toFixed(2)})`;
            return (
              <g key={`${r}-${c}`}>
                <rect x={pad.l + c * cell + 2} y={pad.t + r * cell + 2} width={cell - 4} height={cell - 4} rx={5} fill={fill} stroke="var(--border)">
                  <title>{`Actual ${labels[r]} → Predicted ${labels[c]}: ${cnt} (${pct.toFixed(1)}%)`}</title>
                </rect>
                <text x={pad.l + c * cell + cell / 2} y={pad.t + r * cell + cell / 2 - 2} textAnchor="middle" fill="var(--text)" fontSize={11.5} fontWeight={800} className="mono">
                  {cnt}
                </text>
                <text x={pad.l + c * cell + cell / 2} y={pad.t + r * cell + cell / 2 + 12} textAnchor="middle" fill="var(--text-muted)" fontSize={9.5} className="mono">
                  {pct.toFixed(0)}%
                </text>
              </g>
            );
          }),
        )}
        <text x={pad.l + (n * cell) / 2} y={H - 6} textAnchor="middle" fill="var(--text-dim)" fontSize={10}>Predicted Class</text>
        <text x={15} y={pad.t + (n * cell) / 2} textAnchor="middle" fill="var(--text-dim)" fontSize={10} transform={`rotate(-90 15 ${pad.t + (n * cell) / 2})`}>Actual Class</text>
      </svg>
    </InspectableBox>
  );
};

export const HeatmapMatrixSVG: React.FC<{
  title: string;
  subtitle?: string;
  columns: string[];
  values: (number | null)[][];
}> = ({ title, subtitle, columns, values }) => {
  const n = columns.length;
  if (!n) return null;
  const cell = Math.max(34, Math.min(54, Math.floor(380 / n)));
  const pad = { l: 118, r: 24, t: 22, b: 80 };
  const W = Math.max(540, pad.l + pad.r + n * cell);
  const H = pad.t + pad.b + n * cell;
  const cellFill = (v: number | null) => {
    if (v === null || !Number.isFinite(v)) return 'var(--bg-subtle)';
    const c = Math.max(-1, Math.min(1, v));
    return c >= 0
      ? `rgba(56, 189, 248, ${(0.12 + 0.76 * c).toFixed(2)})`
      : `rgba(248, 113, 113, ${(0.12 + 0.76 * Math.abs(c)).toFixed(2)})`;
  };

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: Math.min(460, H), background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {columns.map((col, i) => (
          <g key={col}>
            <text x={pad.l - 6} y={pad.t + i * cell + cell / 2 + 3} textAnchor="end" fill="var(--text)" fontSize={9.5} fontWeight={600}>
              {col.slice(0, 15)}
            </text>
            <text
              x={pad.l + i * cell + cell / 2}
              y={pad.t + n * cell + 14}
              textAnchor="end"
              transform={`rotate(-35 ${pad.l + i * cell + cell / 2} ${pad.t + n * cell + 14})`}
              fill="var(--text)"
              fontSize={9.5}
              fontWeight={600}
            >
              {col.slice(0, 15)}
            </text>
          </g>
        ))}
        {values.map((row, r) =>
          row.map((v, c) => (
            <g key={`${r}-${c}`}>
              <rect x={pad.l + c * cell + 1.5} y={pad.t + r * cell + 1.5} width={cell - 3} height={cell - 3} rx={3.5} fill={cellFill(v)}>
                <title>{`${columns[r]} × ${columns[c]}: r = ${v !== null ? v.toFixed(4) : 'N/A'}`}</title>
              </rect>
              <text x={pad.l + c * cell + cell / 2} y={pad.t + r * cell + cell / 2 + 3.5} textAnchor="middle" fill="var(--text)" fontSize={n > 9 ? 8.5 : 9.5} fontWeight={700} className="mono">
                {v !== null ? v.toFixed(2) : '—'}
              </text>
            </g>
          )),
        )}
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   15. CumulativeGainLiftChart (Gains + Wizard Curve + Random Diagonal)
   ========================================================================== */
export const CumulativeGainLiftChart: React.FC<{
  title: string;
  subtitle?: string;
  points: { pop_pct: number; gain_pct: number; wizard_pct: number; lift: number }[];
}> = ({ title, subtitle, points }) => {
  const W = 680;
  const H = 310;
  const pad = { l: 52, r: 24, t: 24, b: 38 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const sx = (v: number) => pad.l + (v / 100) * pw;
  const sy = (v: number) => pad.t + ph - (v / 100) * ph;

  const gainD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.pop_pct).toFixed(1)} ${sy(p.gain_pct).toFixed(1)}`).join(' ');
  const wizD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.pop_pct).toFixed(1)} ${sy(p.wizard_pct).toFixed(1)}`).join(' ');
  const liftArea = [
    ...points.map((p) => `${sx(p.pop_pct).toFixed(1)},${sy(p.gain_pct).toFixed(1)}`),
    `${sx(100).toFixed(1)},${sy(100).toFixed(1)}`,
    `${sx(0).toFixed(1)},${sy(0).toFixed(1)}`,
  ].join(' ');

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {[0, 20, 40, 60, 80, 100].map((v) => (
          <g key={v}>
            <line x1={pad.l} y1={sy(v)} x2={W - pad.r} y2={sy(v)} stroke="rgba(148,163,184,0.16)" />
            <text x={pad.l - 6} y={sy(v) + 3} textAnchor="end" fill="var(--text-dim)" fontSize={9.5} className="mono">{v}%</text>
            <text x={sx(v)} y={H - 18} textAnchor="middle" fill="var(--text-dim)" fontSize={9.5} className="mono">{v}%</text>
          </g>
        ))}
        <polygon points={liftArea} fill="#38bdf8" fillOpacity={0.14} />
        <line x1={sx(0)} y1={sy(0)} x2={sx(100)} y2={sy(100)} stroke="var(--text-dim)" strokeDasharray="5 4" />
        <path d={wizD} fill="none" stroke="#34d399" strokeDasharray="3 3" strokeWidth={1.8} />
        <path d={gainD} fill="none" stroke="#38bdf8" strokeWidth={2.6} />
        {points.map((p, i) => (
          <circle key={i} cx={sx(p.pop_pct)} cy={sy(p.gain_pct)} r={2.8} fill="#38bdf8">
            <title>{`Top ${p.pop_pct}% targeted → ${p.gain_pct}% positives captured (Lift ${p.lift}×)`}</title>
          </circle>
        ))}
        <text x={pad.l + pw / 2} y={H - 4} textAnchor="middle" fill="var(--text-dim)" fontSize={10}>% of Population Targeted (Ranked by Score)</text>
        <text x={14} y={pad.t + ph / 2} textAnchor="middle" fill="var(--text-dim)" fontSize={10} transform={`rotate(-90 14 ${pad.t + ph / 2})`}>% of Positives Captured</text>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   16. ResidualDiagnosticsChart (Residuals vs Fitted + ±2σ + Local Bias)
   ========================================================================== */
export const ResidualDiagnosticsChart: React.FC<{
  title: string;
  subtitle?: string;
  points: { fitted: number; residual: number; z_resid: number }[];
  trend: { fitted: number; mean_resid: number }[];
  sigma: number;
}> = ({ title, subtitle, points, trend, sigma }) => {
  if (!points.length) return null;
  const W = 740;
  const H = 300;
  const pad = { l: 56, r: 24, t: 24, b: 36 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const xs = points.map((p) => p.fitted);
  const ys = [2 * sigma, -2 * sigma, ...points.map((p) => p.residual)];
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = (v: number) => pad.l + ((v - minX) / (maxX - minX || 1)) * pw;
  const sy = (v: number) => pad.t + ph - ((v - minY) / (maxY - minY || 1)) * ph;
  const trendD = trend.map((t, i) => `${i === 0 ? 'M' : 'L'} ${sx(t.fitted).toFixed(1)} ${sy(t.mean_resid).toFixed(1)}`).join(' ');

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        <line x1={pad.l} y1={sy(0)} x2={W - pad.r} y2={sy(0)} stroke="var(--text-muted)" strokeWidth={1.3} />
        <line x1={pad.l} y1={sy(2 * sigma)} x2={W - pad.r} y2={sy(2 * sigma)} stroke="#f87171" strokeDasharray="5 4" />
        <line x1={pad.l} y1={sy(-2 * sigma)} x2={W - pad.r} y2={sy(-2 * sigma)} stroke="#f87171" strokeDasharray="5 4" />
        {points.map((p, i) => (
          <circle key={i} cx={sx(p.fitted)} cy={sy(p.residual)} r={Math.abs(p.z_resid) > 2 ? 3.8 : 2.8} fill={Math.abs(p.z_resid) > 2 ? '#f87171' : '#38bdf8'} opacity={0.8}>
            <title>{`Fitted=${fmtNum(p.fitted)}, Residual=${fmtNum(p.residual)} (${p.z_resid.toFixed(2)}σ)`}</title>
          </circle>
        ))}
        {trendD && <path d={trendD} fill="none" stroke="#fbbf24" strokeWidth={2.4} />}
        <text x={pad.l + pw / 2} y={H - 5} textAnchor="middle" fill="var(--text-dim)" fontSize={10}>Fitted Value (ŷ)</text>
        <text x={14} y={pad.t + ph / 2} textAnchor="middle" fill="var(--text-dim)" fontSize={10} transform={`rotate(-90 14 ${pad.t + ph / 2})`}>Residual (y − ŷ)</text>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   17. BlandAltmanChart (Mean vs Difference + Bias + ±1.96 SD LoA)
   ========================================================================== */
export const BlandAltmanChart: React.FC<{
  title: string;
  subtitle?: string;
  points: { mean: number; diff: number; outside: boolean }[];
  bias: number;
  loaUpper: number;
  loaLower: number;
}> = ({ title, subtitle, points, bias, loaUpper, loaLower }) => {
  if (!points.length) return null;
  const W = 740;
  const H = 300;
  const pad = { l: 56, r: 78, t: 22, b: 36 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const xs = points.map((p) => p.mean);
  const ys = [loaLower, loaUpper, ...points.map((p) => p.diff)];
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const spanY = Math.max(...ys) - Math.min(...ys) || 1;
  const loY = Math.min(...ys) - spanY * 0.1;
  const hiY = Math.max(...ys) + spanY * 0.1;
  const sx = (v: number) => pad.l + ((v - minX) / (maxX - minX || 1)) * pw;
  const sy = (v: number) => pad.t + ph - ((v - loY) / (hiY - loY)) * ph;

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        <rect x={pad.l} y={sy(loaUpper)} width={pw} height={Math.max(2, sy(loaLower) - sy(loaUpper))} fill="rgba(52,211,153,0.10)" />
        <line x1={pad.l} y1={sy(bias)} x2={W - pad.r} y2={sy(bias)} stroke="#38bdf8" strokeWidth={1.8} />
        <text x={W - pad.r + 4} y={sy(bias) + 3} fill="#38bdf8" fontSize={9.5} className="mono">Bias {fmtNum(bias, 2)}</text>
        <line x1={pad.l} y1={sy(loaUpper)} x2={W - pad.r} y2={sy(loaUpper)} stroke="#f87171" strokeDasharray="5 4" />
        <text x={W - pad.r + 4} y={sy(loaUpper) + 3} fill="#f87171" fontSize={9} className="mono">+1.96SD</text>
        <line x1={pad.l} y1={sy(loaLower)} x2={W - pad.r} y2={sy(loaLower)} stroke="#f87171" strokeDasharray="5 4" />
        <text x={W - pad.r + 4} y={sy(loaLower) + 3} fill="#f87171" fontSize={9} className="mono">-1.96SD</text>
        {points.map((p, i) => (
          <circle key={i} cx={sx(p.mean)} cy={sy(p.diff)} r={p.outside ? 4 : 2.9} fill={p.outside ? '#f87171' : '#34d399'} opacity={0.85}>
            <title>{`Mean=${fmtNum(p.mean)}, Diff=${fmtNum(p.diff)}${p.outside ? ' (Outside 95% LoA)' : ''}`}</title>
          </circle>
        ))}
        <text x={pad.l + pw / 2} y={H - 5} textAnchor="middle" fill="var(--text-dim)" fontSize={10}>Mean of Actual &amp; Predicted ½(y + ŷ)</text>
        <text x={14} y={pad.t + ph / 2} textAnchor="middle" fill="var(--text-dim)" fontSize={10} transform={`rotate(-90 14 ${pad.t + ph / 2})`}>Difference (y − ŷ)</text>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   18. CusumChart (Tabular Upper C+ / Lower C- Sequential Drift Chart)
   ========================================================================== */
export const CusumChart: React.FC<{
  title: string;
  subtitle?: string;
  points: { step: number; c_plus: number; c_minus: number; alarm: boolean }[];
  hLimit: number;
}> = ({ title, subtitle, points, hLimit }) => {
  if (!points.length) return null;
  const W = 740;
  const H = 290;
  const pad = { l: 54, r: 58, t: 24, b: 34 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const xs = points.map((p) => p.step);
  const ys = [hLimit * 1.15, -hLimit * 1.15, ...points.flatMap((p) => [p.c_plus, p.c_minus])];
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = (v: number) => pad.l + ((v - minX) / (maxX - minX || 1)) * pw;
  const sy = (v: number) => pad.t + ph - ((v - minY) / (maxY - minY || 1)) * ph;

  const pPlus = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.step).toFixed(1)} ${sy(p.c_plus).toFixed(1)}`).join(' ');
  const pMinus = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.step).toFixed(1)} ${sy(p.c_minus).toFixed(1)}`).join(' ');

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        <line x1={pad.l} y1={sy(0)} x2={W - pad.r} y2={sy(0)} stroke="rgba(148,163,184,0.4)" />
        <line x1={pad.l} y1={sy(hLimit)} x2={W - pad.r} y2={sy(hLimit)} stroke="#f87171" strokeDasharray="5 4" strokeWidth={1.4} />
        <text x={W - pad.r + 4} y={sy(hLimit) + 3} fill="#f87171" fontSize={9.5} className="mono">+H ({hLimit})</text>
        <line x1={pad.l} y1={sy(-hLimit)} x2={W - pad.r} y2={sy(-hLimit)} stroke="#f87171" strokeDasharray="5 4" strokeWidth={1.4} />
        <text x={W - pad.r + 4} y={sy(-hLimit) + 3} fill="#f87171" fontSize={9.5} className="mono">-H (-{hLimit})</text>
        <path d={pPlus} fill="none" stroke="#38bdf8" strokeWidth={2.1} />
        <path d={pMinus} fill="none" stroke="#a78bfa" strokeWidth={2.1} />
        {points.filter((p) => p.alarm).map((p, i) => (
          <circle key={i} cx={sx(p.step)} cy={sy(p.c_plus > hLimit ? p.c_plus : p.c_minus)} r={4.2} fill="#ef4444">
            <title>{`Step ${p.step} CUSUM Alarm: C+=${p.c_plus}, C-=${p.c_minus}`}</title>
          </circle>
        ))}
        <text x={pad.l} y={15} fill="var(--text-muted)" fontSize={10}>Cyan: Upper CUSUM C⁺ · Purple: Lower CUSUM C⁻ · Red: Drift Alarm (&gt;H)</text>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   19. LorenzGiniChart (Lorenz Curve + Shaded Inequality Area + Gini)
   ========================================================================== */
export const LorenzGiniChart: React.FC<{
  title: string;
  subtitle?: string;
  points: { pop_share: number; outcome_share: number }[];
  gini: number;
}> = ({ title, subtitle, points, gini }) => {
  const W = 640;
  const H = 310;
  const pad = { l: 52, r: 24, t: 22, b: 38 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const sx = (v: number) => pad.l + v * pw;
  const sy = (v: number) => pad.t + ph - v * ph;

  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.pop_share).toFixed(1)} ${sy(p.outcome_share).toFixed(1)}`).join(' ');
  const lens = [
    ...points.map((p) => `${sx(p.pop_share).toFixed(1)},${sy(p.outcome_share).toFixed(1)}`),
    `${sx(0).toFixed(1)},${sy(0).toFixed(1)}`,
  ].join(' ');

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line x1={pad.l} y1={sy(f)} x2={W - pad.r} y2={sy(f)} stroke="rgba(148,163,184,0.16)" />
            <text x={pad.l - 6} y={sy(f) + 3} textAnchor="end" fill="var(--text-dim)" fontSize={9.5} className="mono">{`${Math.round(f * 100)}%`}</text>
            <text x={sx(f)} y={H - 18} textAnchor="middle" fill="var(--text-dim)" fontSize={9.5} className="mono">{`${Math.round(f * 100)}%`}</text>
          </g>
        ))}
        <polygon points={lens} fill="#34d399" fillOpacity={0.18} />
        <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="var(--text-muted)" strokeDasharray="5 4" strokeWidth={1.4} />
        <path d={d} fill="none" stroke="#34d399" strokeWidth={2.6} />
        <text x={pad.l + 14} y={pad.t + 22} fill="#34d399" fontSize={13} fontWeight={800} className="mono">
          Gini = {gini.toFixed(4)}
        </text>
        <text x={pad.l + pw / 2} y={H - 4} textAnchor="middle" fill="var(--text-dim)" fontSize={10}>Cumulative Share of Observations</text>
        <text x={14} y={pad.t + ph / 2} textAnchor="middle" fill="var(--text-dim)" fontSize={10} transform={`rotate(-90 14 ${pad.t + ph / 2})`}>Cumulative Share of Magnitude</text>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   20. ParallelCoordinatesChart (Multivariate Trajectories + Group Medians)
   ========================================================================== */
export const ParallelCoordinatesChart: React.FC<{
  title: string;
  subtitle?: string;
  axes: { name: string; min: number; max: number }[];
  lines: { group: string; color: string; values: number[] }[];
  medians: { group: string; color: string; values: number[] }[];
}> = ({ title, subtitle, axes, lines, medians }) => {
  if (axes.length < 2) return null;
  const W = 760;
  const H = 315;
  const pad = { l: 48, r: 48, t: 36, b: 44 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const sx = (i: number) => pad.l + (i / (axes.length - 1)) * pw;
  const sy = (v: number) => pad.t + ph - Math.max(0, Math.min(1, v)) * ph;

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {lines.map((ln, idx) => (
          <polyline
            key={idx}
            points={ln.values.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(' ')}
            fill="none"
            stroke={ln.color}
            strokeWidth={1.0}
            opacity={0.2}
          />
        ))}
        {medians.map((m) => (
          <g key={m.group}>
            <polyline
              points={m.values.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(' ')}
              fill="none"
              stroke={m.color}
              strokeWidth={3.0}
            >
              <title>{`Median Profile (${m.group})`}</title>
            </polyline>
            {m.values.map((v, i) => (
              <circle key={i} cx={sx(i)} cy={sy(v)} r={3.6} fill={m.color} stroke="var(--bg)" strokeWidth={1.2} />
            ))}
          </g>
        ))}
        {axes.map((ax, i) => (
          <g key={ax.name}>
            <line x1={sx(i)} y1={pad.t} x2={sx(i)} y2={pad.t + ph} stroke="var(--border-strong)" strokeWidth={1.4} />
            <text x={sx(i)} y={pad.t - 8} textAnchor="middle" fill="var(--text-dim)" fontSize={8.5} className="mono">{fmtNum(ax.max, 2)}</text>
            <text x={sx(i)} y={pad.t + ph + 13} textAnchor="middle" fill="var(--text-dim)" fontSize={8.5} className="mono">{fmtNum(ax.min, 2)}</text>
            <text x={sx(i)} y={H - 8} textAnchor="middle" fill="var(--text)" fontSize={10.5} fontWeight={700}>{ax.name.slice(0, 14)}</text>
          </g>
        ))}
        <g transform={`translate(${pad.l}, 14)`}>
          {medians.map((m, idx) => (
            <g key={m.group} transform={`translate(${idx * 145}, 0)`}>
              <line x1={0} y1={-4} x2={14} y2={-4} stroke={m.color} strokeWidth={3} />
              <text x={18} y={0} fill="var(--text-muted)" fontSize={10}>{m.group.slice(0, 16)} (median)</text>
            </g>
          ))}
        </g>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   21. RidgelineKDEChart (Vertically Stacked Joyplot KDE Distributions)
   ========================================================================== */
export const RidgelineKDEChart: React.FC<{
  title: string;
  subtitle?: string;
  xGrid: number[];
  ridges: { label: string; n: number; median: number; density: number[]; color?: string }[];
  xLabel: string;
}> = ({ title, subtitle, xGrid, ridges, xLabel }) => {
  if (!xGrid.length || !ridges.length) return null;
  const rowStep = 38;
  const pad = { l: 145, r: 24, t: 36, b: 36 };
  const W = 740;
  const H = pad.t + pad.b + ridges.length * rowStep;
  const pw = W - pad.l - pad.r;
  const minX = xGrid[0];
  const maxX = xGrid[xGrid.length - 1];
  const sx = (v: number) => pad.l + ((v - minX) / (maxX - minX || 1)) * pw;

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const xv = minX + (maxX - minX) * f;
          const gx = sx(xv);
          return (
            <g key={f}>
              <line x1={gx} y1={pad.t - 10} x2={gx} y2={H - pad.b} stroke="rgba(148,163,184,0.14)" />
              <text x={gx} y={H - 18} textAnchor="middle" fill="var(--text-dim)" fontSize={9.5} className="mono">{fmtNum(xv, 2)}</text>
            </g>
          );
        })}
        {ridges.map((r, idx) => {
          const baseY = pad.t + (idx + 1) * rowStep - 6;
          const amp = 32;
          const col = r.color || PALETTE[idx % PALETTE.length];
          const topPts = xGrid.map((xv, k) => `${sx(xv).toFixed(1)},${(baseY - (r.density[k] ?? 0) * amp).toFixed(1)}`);
          const areaPts = [`${sx(minX).toFixed(1)},${baseY}`, ...topPts, `${sx(maxX).toFixed(1)},${baseY}`].join(' ');
          return (
            <g key={`${r.label}-${idx}`}>
              <text x={pad.l - 10} y={baseY - 4} textAnchor="end" fill="var(--text)" fontSize={10.5} fontWeight={600}>
                {r.label.slice(0, 18)}
              </text>
              <polygon points={areaPts} fill={col} fillOpacity={0.34} />
              <polyline points={topPts.join(' ')} fill="none" stroke={col} strokeWidth={1.8}>
                <title>{`${r.label} (n=${r.n}, median=${fmtNum(r.median)})`}</title>
              </polyline>
              <line x1={sx(r.median)} y1={baseY} x2={sx(r.median)} y2={baseY - 14} stroke="var(--text)" strokeWidth={1.8} />
            </g>
          );
        })}
        <text x={pad.l + pw / 2} y={H - 4} textAnchor="middle" fill="var(--text-dim)" fontSize={10}>{xLabel}</text>
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   22. SankeyFlowChart (Mass-Conserving Alluvial / Sankey Flow Diagram)
   ========================================================================== */
export const SankeyFlowChart: React.FC<{
  title: string;
  subtitle?: string;
  sources: { id: string; label: string; total: number }[];
  targets: { id: string; label: string; total: number }[];
  links: { source: string; target: string; source_label: string; target_label: string; value: number; color?: string }[];
  total: number;
}> = ({ title, subtitle, sources, targets, links, total }) => {
  if (!sources.length || !targets.length) return null;
  const W = 740;
  const H = 310;
  const leftX = 150;
  const rightX = W - 166;
  const barW = 16;
  const gap = 10;
  const tot = Math.max(1, total);

  const availL = Math.max(80, H - 56 - gap * Math.max(0, sources.length - 1));
  let curY = 28;
  const srcMap = new Map<string, { y: number; h: number; offset: number; label: string; total: number }>();
  sources.forEach((s) => {
    const h = Math.max(10, (s.total / tot) * availL);
    srcMap.set(s.id, { y: curY, h, offset: curY, label: s.label, total: s.total });
    curY += h + gap;
  });

  const availR = Math.max(80, H - 56 - gap * Math.max(0, targets.length - 1));
  curY = 28;
  const dstMap = new Map<string, { y: number; h: number; offset: number; label: string; total: number; color: string }>();
  targets.forEach((t, i) => {
    const h = Math.max(10, (t.total / tot) * availR);
    dstMap.set(t.id, { y: curY, h, offset: curY, label: t.label, total: t.total, color: PALETTE[(i + 3) % PALETTE.length] });
    curY += h + gap;
  });

  const midX = (leftX + barW + rightX) / 2;

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {links.map((lnk, i) => {
          const s = srcMap.get(lnk.source);
          const t = dstMap.get(lnk.target);
          if (!s || !t) return null;
          const sh = (lnk.value / Math.max(1, s.total)) * s.h;
          const th = (lnk.value / Math.max(1, t.total)) * t.h;
          const sy0 = s.offset;
          const sy1 = s.offset + sh;
          const ty0 = t.offset;
          const ty1 = t.offset + th;
          s.offset += sh;
          t.offset += th;
          const d = `M ${leftX + barW} ${sy0.toFixed(1)} C ${midX} ${sy0.toFixed(1)}, ${midX} ${ty0.toFixed(1)}, ${rightX} ${ty0.toFixed(1)} L ${rightX} ${ty1.toFixed(1)} C ${midX} ${ty1.toFixed(1)}, ${midX} ${sy1.toFixed(1)}, ${leftX + barW} ${sy1.toFixed(1)} Z`;
          return (
            <path key={i} d={d} fill={lnk.color || '#38bdf8'} fillOpacity={0.34}>
              <title>{`${lnk.source_label} → ${lnk.target_label}: ${lnk.value.toLocaleString()} (${((100 * lnk.value) / tot).toFixed(1)}%)`}</title>
            </path>
          );
        })}
        {sources.map((s, i) => {
          const box = srcMap.get(s.id)!;
          return (
            <g key={s.id}>
              <rect x={leftX} y={box.y} width={barW} height={box.h} rx={3} fill={PALETTE[i % PALETTE.length]} />
              <text x={leftX - 8} y={box.y + box.h / 2 + 4} textAnchor="end" fill="var(--text)" fontSize={10.5} fontWeight={600}>
                {`${s.label.slice(0, 16)} (${s.total})`}
              </text>
            </g>
          );
        })}
        {targets.map((t) => {
          const box = dstMap.get(t.id)!;
          return (
            <g key={t.id}>
              <rect x={rightX} y={box.y} width={barW} height={box.h} rx={3} fill={box.color} />
              <text x={rightX + barW + 8} y={box.y + box.h / 2 + 4} textAnchor="start" fill="var(--text)" fontSize={10.5} fontWeight={600}>
                {`${t.label.slice(0, 16)} (${t.total})`}
              </text>
            </g>
          );
        })}
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   23. FunnelStageChart (Tapered Pipeline Retention Funnel)
   ========================================================================== */
export const FunnelStageChart: React.FC<{
  title: string;
  subtitle?: string;
  stages: { label: string; value: number; pct_of_initial: number; step_conversion_pct: number; color?: string }[];
}> = ({ title, subtitle, stages }) => {
  if (!stages.length) return null;
  const W = 720;
  const rowH = 48;
  const H = 32 + stages.length * rowH;
  const maxVal = Math.max(...stages.map((s) => num(s.value)), 1);
  const maxBarW = 340;
  const centerX = W / 2 + 15;

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        {stages.map((st, idx) => {
          const topFrac = Math.max(0.16, num(st.value) / maxVal);
          const botFrac = idx + 1 < stages.length ? Math.max(0.16, num(stages[idx + 1].value) / maxVal) : topFrac * 0.88;
          const wt = topFrac * maxBarW;
          const wb = botFrac * maxBarW;
          const y0 = 16 + idx * rowH;
          const y1 = y0 + rowH - 6;
          const poly = `${centerX - wt / 2},${y0} ${centerX + wt / 2},${y0} ${centerX + wb / 2},${y1} ${centerX - wb / 2},${y1}`;
          return (
            <g key={st.label}>
              <polygon points={poly} fill={st.color || PALETTE[idx % PALETTE.length]} opacity={0.88}>
                <title>{`${st.label}: ${st.value.toLocaleString()} (${st.pct_of_initial}% of initial, step ${st.step_conversion_pct}%)`}</title>
              </polygon>
              <text x={16} y={y0 + 25} fill="var(--text)" fontSize={11} fontWeight={600}>
                {st.label.slice(0, 28)}
              </text>
              <text x={centerX} y={y0 + 26} textAnchor="middle" fill="#0b1120" fontSize={12.5} fontWeight={800} className="mono">
                {st.value.toLocaleString()}
              </text>
              <text x={W - 14} y={y0 + 25} textAnchor="end" fill="var(--text-muted)" fontSize={10.5} className="mono">
                {st.pct_of_initial.toFixed(1)}%
              </text>
            </g>
          );
        })}
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   25. StackedMarimekkoChart (Pure-SVG 100% Cross-Tabulated Stacked Bars)
   ========================================================================== */
export const StackedMarimekkoChart: React.FC<{
  title: string;
  subtitle?: string;
  segmentNames: string[];
  bars: { label: string; total: number; segments: { name: string; count: number; pct: number; color?: string }[] }[];
}> = ({ title, subtitle, segmentNames, bars }) => {
  if (!bars.length) return null;
  const rowH = 34;
  const pad = { l: 145, r: 64, t: 34, b: 22 };
  const W = 740;
  const H = pad.t + pad.b + bars.length * rowH;
  const pw = W - pad.l - pad.r;

  return (
    <InspectableBox title={title} subtitle={subtitle} compact>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: H, background: 'var(--bg)', borderRadius: 10, display: 'block' }}>
        <g transform={`translate(${pad.l}, 18)`}>
          {segmentNames.slice(0, 6).map((s, idx) => (
            <g key={s} transform={`translate(${idx * 104}, 0)`}>
              <rect x={0} y={-9} width={10} height={10} rx={2.5} fill={PALETTE[idx % PALETTE.length]} />
              <text x={14} y={0} fill="var(--text-muted)" fontSize={10}>{s.slice(0, 12)}</text>
            </g>
          ))}
        </g>
        {bars.map((b, rIdx) => {
          const y = pad.t + rIdx * rowH;
          let curX = pad.l;
          return (
            <g key={b.label}>
              <text x={pad.l - 10} y={y + 18} textAnchor="end" fill="var(--text)" fontSize={10.5} fontWeight={600}>
                {b.label.slice(0, 18)}
              </text>
              {b.segments.map((seg, sIdx) => {
                const wSeg = (Math.max(0, seg.pct) / 100) * pw;
                const xSeg = curX;
                curX += wSeg;
                if (wSeg <= 0.5) return null;
                return (
                  <g key={seg.name}>
                    <rect x={xSeg} y={y + 4} width={wSeg} height={20} fill={seg.color || PALETTE[sIdx % PALETTE.length]} stroke="var(--bg)" strokeWidth={1}>
                      <title>{`${b.label} · ${seg.name}: ${seg.count.toLocaleString()} (${seg.pct.toFixed(1)}%)`}</title>
                    </rect>
                    {wSeg >= 38 && (
                      <text x={xSeg + wSeg / 2} y={y + 18} textAnchor="middle" fill="#0b1120" fontSize={9.5} fontWeight={800} className="mono">
                        {seg.pct.toFixed(0)}%
                      </text>
                    )}
                  </g>
                );
              })}
              <text x={W - 10} y={y + 18} textAnchor="end" fill="var(--text-dim)" fontSize={10} className="mono">
                n={b.total}
              </text>
            </g>
          );
        })}
      </svg>
    </InspectableBox>
  );
};

/* ==========================================================================
   Unified Dispatcher & Standalone SVG Serializer (Resolves CSS Variables)
   ========================================================================== */
export const ProfessionalChartDispatcher: React.FC<{ payload: LiveChartPayload }> = ({ payload }) => {
  const d = payload.data as Record<string, any>;
  switch (payload.chart_type) {
    case 'gauge':
      return (
        <GaugeChart
          title={payload.title}
          subtitle={payload.subtitle}
          value={num(d.value)}
          min={num(d.min, 0)}
          max={num(d.max, 100)}
          unit={String(d.unit ?? '')}
          statusLabel={String(d.status_label ?? '')}
          thresholds={Array.isArray(d.thresholds) ? d.thresholds : []}
        />
      );
    case 'radar':
      return (
        <RadarChart
          title={payload.title}
          subtitle={payload.subtitle}
          axes={Array.isArray(d.axes) ? d.axes : []}
          series={Array.isArray(d.series) ? d.series : []}
          maxScale={num(d.max_scale, 100)}
        />
      );
    case 'donut':
      return (
        <DonutChart
          title={payload.title}
          subtitle={payload.subtitle}
          slices={Array.isArray(d.slices) ? d.slices : []}
          centerValue={typeof d.center_value === 'string' ? d.center_value : undefined}
          centerLabel={typeof d.center_label === 'string' ? d.center_label : undefined}
        />
      );
    case 'waterfall':
      return <WaterfallChart title={payload.title} subtitle={payload.subtitle} bars={Array.isArray(d.bars) ? d.bars : []} />;
    case 'forest':
      return (
        <ForestChart
          title={payload.title}
          subtitle={payload.subtitle}
          rows={Array.isArray(d.rows) ? d.rows : []}
          referenceValue={num(d.reference_value, 0)}
          xLabel={String(d.x_label ?? 'Effect Estimate (95% CI)')}
        />
      );
    case 'pareto':
      return (
        <ParetoChart
          title={payload.title}
          subtitle={payload.subtitle}
          items={Array.isArray(d.items) ? d.items : []}
          cutoffPct={num(d.cutoff_pct, 80)}
          yLeftLabel={String(d.y_left_label ?? 'Magnitude')}
        />
      );
    case 'ecdf':
      return (
        <ECDFChart
          title={payload.title}
          subtitle={payload.subtitle}
          series={Array.isArray(d.series) ? d.series : []}
          xLabel={String(d.x_label ?? 'Value')}
        />
      );
    case 'violin_box':
      return (
        <ViolinBoxChart
          title={payload.title}
          subtitle={payload.subtitle}
          groups={Array.isArray(d.groups) ? d.groups : []}
          yLabel={String(d.y_label ?? 'Value')}
        />
      );
    case 'qq_plot':
      return (
        <QQPlotChart
          title={payload.title}
          subtitle={payload.subtitle}
          points={Array.isArray(d.points) ? d.points : []}
          variableName={String(d.variable_name ?? '')}
        />
      );
    case 'regression_ci':
      return (
        <JointRegressionChart
          title={payload.title}
          subtitle={payload.subtitle}
          points={Array.isArray(d.points) ? d.points : []}
          band={Array.isArray(d.band) ? d.band : []}
          xLabel={String(d.x_label ?? 'X')}
          yLabel={String(d.y_label ?? 'Y')}
        />
      );
    case 'roc_curve':
      return (
        <RocCurveChart
          title={payload.title}
          subtitle={payload.subtitle}
          points={Array.isArray(d.points) ? d.points : []}
          auc={num(d.auc)}
          optimalPoint={d.optimal_point}
        />
      );
    case 'pr_curve':
      return (
        <PRCurveChart
          title={payload.title}
          subtitle={payload.subtitle}
          points={Array.isArray(d.points) ? d.points : []}
          ap={num(d.ap)}
          prevalence={num(d.prevalence, 0.5)}
          optimalPoint={d.optimal_point}
        />
      );
    case 'calibration':
      return (
        <CalibrationCurveChart
          title={payload.title}
          subtitle={payload.subtitle}
          bins={Array.isArray(d.bins) ? d.bins : []}
          ece={num(d.ece)}
          brier={num(d.brier)}
        />
      );
    case 'confusion_matrix':
      return (
        <ConfusionMatrixChart
          title={payload.title}
          subtitle={payload.subtitle}
          labels={Array.isArray(d.labels) ? d.labels : []}
          counts={Array.isArray(d.counts) ? d.counts : []}
          rowPcts={Array.isArray(d.row_pcts) ? d.row_pcts : []}
        />
      );
    case 'gain_lift':
      return <CumulativeGainLiftChart title={payload.title} subtitle={payload.subtitle} points={Array.isArray(d.points) ? d.points : []} />;
    case 'residual_diag':
      return (
        <ResidualDiagnosticsChart
          title={payload.title}
          subtitle={payload.subtitle}
          points={Array.isArray(d.points) ? d.points : []}
          trend={Array.isArray(d.trend) ? d.trend : []}
          sigma={num(d.sigma, 1)}
        />
      );
    case 'bland_altman':
      return (
        <BlandAltmanChart
          title={payload.title}
          subtitle={payload.subtitle}
          points={Array.isArray(d.points) ? d.points : []}
          bias={num(d.bias)}
          loaUpper={num(d.loa_upper)}
          loaLower={num(d.loa_lower)}
        />
      );
    case 'cusum':
      return (
        <CusumChart
          title={payload.title}
          subtitle={payload.subtitle}
          points={Array.isArray(d.points) ? d.points : []}
          hLimit={num(d.h_limit, 4)}
        />
      );
    case 'lorenz':
      return (
        <LorenzGiniChart
          title={payload.title}
          subtitle={payload.subtitle}
          points={Array.isArray(d.points) ? d.points : []}
          gini={num(d.gini)}
        />
      );
    case 'parallel_coords':
      return (
        <ParallelCoordinatesChart
          title={payload.title}
          subtitle={payload.subtitle}
          axes={Array.isArray(d.axes) ? d.axes : []}
          lines={Array.isArray(d.lines) ? d.lines : []}
          medians={Array.isArray(d.medians) ? d.medians : []}
        />
      );
    case 'ridgeline':
      return (
        <RidgelineKDEChart
          title={payload.title}
          subtitle={payload.subtitle}
          xGrid={Array.isArray(d.x_grid) ? d.x_grid : []}
          ridges={Array.isArray(d.ridges) ? d.ridges : []}
          xLabel={String(d.x_label ?? 'Value')}
        />
      );
    case 'sankey':
      return (
        <SankeyFlowChart
          title={payload.title}
          subtitle={payload.subtitle}
          sources={Array.isArray(d.sources) ? d.sources : []}
          targets={Array.isArray(d.targets) ? d.targets : []}
          links={Array.isArray(d.links) ? d.links : []}
          total={num(d.total, 1)}
        />
      );
    case 'funnel':
      return <FunnelStageChart title={payload.title} subtitle={payload.subtitle} stages={Array.isArray(d.stages) ? d.stages : []} />;
    case 'heatmap_svg':
      return (
        <HeatmapMatrixSVG
          title={payload.title}
          subtitle={payload.subtitle}
          columns={Array.isArray(d.columns) ? d.columns : []}
          values={Array.isArray(d.values) ? d.values : []}
        />
      );
    case 'stacked_bar':
      return (
        <StackedMarimekkoChart
          title={payload.title}
          subtitle={payload.subtitle}
          segmentNames={Array.isArray(d.segment_names) ? d.segment_names : []}
          bars={Array.isArray(d.bars) ? d.bars : []}
        />
      );
    default:
      return <div style={{ color: 'var(--text-dim)', fontSize: 12 }}>Unsupported chart type: {payload.chart_type}</div>;
  }
};

/**
 * Serialize a mounted SVG inside ``container`` to standalone XML with all CSS custom
 * properties resolved to concrete theme hex colors so external viewers render identically.
 */
export function serializeLiveChartSvg(container: HTMLElement | null, title: string, subtitle?: string): string | null {
  if (!container || typeof document === 'undefined') return null;
  const svgEl = container.querySelector('svg');
  if (!svgEl) return null;

  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  const tokenMap: Record<string, string> = isLight
    ? {
        'var(--bg)': '#f4f7fb',
        'var(--bg-card)': '#ffffff',
        'var(--bg-subtle)': '#eef3fa',
        'var(--border)': '#d7e1ef',
        'var(--border-strong)': '#b6c6dd',
        'var(--text)': '#0f172a',
        'var(--text-muted)': '#475569',
        'var(--text-dim)': '#64748b',
        'var(--accent)': '#0284c7',
      }
    : {
        'var(--bg)': '#070b14',
        'var(--bg-card)': '#111b2e',
        'var(--bg-subtle)': '#17233b',
        'var(--border)': '#22314f',
        'var(--border-strong)': '#31456b',
        'var(--text)': '#edf2ff',
        'var(--text-muted)': '#94a3b8',
        'var(--text-dim)': '#64748b',
        'var(--accent)': '#38bdf8',
      };

  const clone = svgEl.cloneNode(true) as SVGSVGElement;
  const vb = (clone.getAttribute('viewBox') || '0 0 740 310').split(/\s+/).map(Number);
  const w = vb[2] || 740;
  const h = vb[3] || 310;
  const headerH = 44;

  // Shift plot group down and prepend standalone background + title header
  const innerGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  innerGroup.setAttribute('transform', `translate(0, ${headerH})`);
  while (clone.firstChild) {
    innerGroup.appendChild(clone.firstChild);
  }

  const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  bgRect.setAttribute('x', '0');
  bgRect.setAttribute('y', '0');
  bgRect.setAttribute('width', String(w));
  bgRect.setAttribute('height', String(h + headerH));
  bgRect.setAttribute('rx', '10');
  bgRect.setAttribute('fill', tokenMap['var(--bg)']);

  const titleNode = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  titleNode.setAttribute('x', '16');
  titleNode.setAttribute('y', '21');
  titleNode.setAttribute('fill', tokenMap['var(--text)']);
  titleNode.setAttribute('font-size', '13.5');
  titleNode.setAttribute('font-weight', '700');
  titleNode.setAttribute('font-family', 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif');
  titleNode.textContent = title;

  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  clone.setAttribute('viewBox', `0 0 ${w} ${h + headerH}`);
  clone.setAttribute('width', String(w));
  clone.setAttribute('height', String(h + headerH));
  clone.removeAttribute('style');
  clone.setAttribute('font-family', 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif');
  clone.appendChild(bgRect);
  clone.appendChild(titleNode);

  if (subtitle) {
    const subNode = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    subNode.setAttribute('x', '16');
    subNode.setAttribute('y', '37');
    subNode.setAttribute('fill', tokenMap['var(--text-muted)']);
    subNode.setAttribute('font-size', '10.5');
    subNode.textContent = subtitle.slice(0, 115);
    clone.appendChild(subNode);
  }
  clone.appendChild(innerGroup);

  let rawXml = new XMLSerializer().serializeToString(clone);
  for (const [cssVar, hex] of Object.entries(tokenMap)) {
    rawXml = rawXml.split(cssVar).join(hex);
  }
  return `<?xml version="1.0" encoding="UTF-8"?>\n${rawXml}`;
}
