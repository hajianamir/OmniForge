import React, { useState } from 'react';
import { Award, CheckCircle2, Code2, Copy, Download, Info, XCircle } from 'lucide-react';
import { InspectableBox } from './InspectModal';

export interface StructuredTerm {
  feature: string;
  coefficient: number;
  sign?: string;
  abs_coefficient?: number;
}

export interface EquationEntry {
  model_name: string;
  target_name?: string;
  equation_type?: string;
  model_paradigm?: string;
  formula_plain?: string;
  plain_equation?: string;
  formula_latex?: string;
  latex_equation?: string;
  formula_python?: string;
  formula_excel?: string;
  formula_sql?: string;
  formula_c_matlab?: string;
  external_usage_summary?: string;
  surrogate_plain_equation?: string | null;
  surrogate_latex_equation?: string | null;
  surrogate_fidelity_r2?: number | null;
  surrogate_intercept?: number | null;
  surrogate_coefficients?: Record<string, number>;
  decision_rules_text?: string | null;
  architecture_equations?: string[];
  explanation_notes?: string[];
  intercept?: number | null;
  coefficients?: Record<string, number>;
  coefficients_table?: Record<string, any>[];
  structured_terms?: StructuredTerm[];
  feature_importance?: Record<string, number>;
  metrics?: Record<string, number>;
  rank?: number;
  selection_criterion?: string;
  primary_metric?: string;
  primary_score?: number;
  naive_floor_score?: number;
  lift_over_naive?: number;
  lift_over_naive_pct?: number;
  acceptance_status?: string;
  acceptance_reasons?: string[];
  notes?: string;
}

function prettifyMathText(raw: string): string {
  return raw
    .replace(/\*\*/g, '^')
    .replace(/\*/g, ' · ')
    .replace(/yhat/gi, 'ŷ')
    .replace(/y_hat/gi, 'ŷ')
    .replace(/sigma/gi, 'σ')
    .replace(/beta/gi, 'β')
    .replace(/alpha/gi, 'α')
    .replace(/<=/g, '≤')
    .replace(/>=/g, '≥')
    .replace(/!=/g, '≠')
    .replace(/ - /g, ' − ')
    .replace(/\s+/g, ' ')
    .trim();
}

function statusBadgeStyle(status?: string): { cls: string; icon: React.ReactNode } {
  const s = (status || '').toUpperCase();
  if (s.includes('CHAMPION') || s.includes('ACCEPTED')) {
    return { cls: 'badge-green', icon: <CheckCircle2 size={12} /> };
  }
  if (s.includes('REJECTED')) {
    return { cls: 'badge-red', icon: <XCircle size={12} /> };
  }
  return { cls: 'badge-amber', icon: <Info size={12} /> };
}

/**
 * Modern mathematical equation card with structured term chips, symbolic typography,
 * LaTeX / Python one-click copy, explicit selection criteria, lift over naive floor,
 * acceptance/rejection rationale, and built-in Zoom + Pop-Out inspection.
 */
export const MathEquationCard: React.FC<{
  eq: EquationEntry;
  defaultOpenCode?: boolean;
}> = ({ eq, defaultOpenCode = false }) => {
  const [copied, setCopied] = useState<string | null>(null);
  const [showCode, setShowCode] = useState<boolean>(defaultOpenCode);
  const [codeTab, setCodeTab] = useState<'python' | 'excel' | 'sql' | 'c_matlab' | 'latex'>('python');

  const plainFormula = (eq.formula_plain || eq.plain_equation || eq.surrogate_plain_equation || '').trim();
  const latexFormula = (eq.formula_latex || eq.latex_equation || eq.surrogate_latex_equation || '').trim();
  const pythonFormula = (eq.formula_python || '').trim();
  const excelFormula = (eq.formula_excel || '').trim();
  const sqlFormula = (eq.formula_sql || '').trim();
  const cMatlabFormula = (eq.formula_c_matlab || '').trim();
  const hasExternalKit = Boolean(pythonFormula || excelFormula || sqlFormula || cMatlabFormula || latexFormula);

  const termsFromTable: StructuredTerm[] = Array.isArray(eq.coefficients_table)
    ? eq.coefficients_table
        .filter((r) => r && typeof r === 'object' && typeof (r.coefficient ?? r.weight ?? r['Surrogate Coef (beta)'] ?? r['Raw Coefficient (beta)']) === 'number' && Number.isFinite(r.coefficient ?? r.weight ?? r['Surrogate Coef (beta)'] ?? r['Raw Coefficient (beta)']))
        .map((r) => {
          const val = Number(r.coefficient ?? r.weight ?? r['Surrogate Coef (beta)'] ?? r['Raw Coefficient (beta)']);
          return {
            feature: String(r.feature || r.Feature || r.name || 'feature'),
            coefficient: val,
            sign: val >= 0 ? '+' : '-',
            abs_coefficient: Math.abs(val),
          };
        })
        .sort((a, b) => (b.abs_coefficient ?? 0) - (a.abs_coefficient ?? 0))
    : [];

  const terms: StructuredTerm[] =
    eq.structured_terms && eq.structured_terms.length > 0
      ? eq.structured_terms
      : eq.coefficients && Object.keys(eq.coefficients).length > 0
        ? Object.entries(eq.coefficients)
            .map(([feature, coefficient]) => ({
              feature,
              coefficient: Number(coefficient),
              sign: Number(coefficient) >= 0 ? '+' : '-',
              abs_coefficient: Math.abs(Number(coefficient)),
            }))
            .sort((a, b) => (b.abs_coefficient ?? 0) - (a.abs_coefficient ?? 0))
        : termsFromTable;

  const maxCoef = Math.max(...terms.map((t) => Math.abs(t.coefficient)), 1e-9);
  const badge = statusBadgeStyle(eq.acceptance_status);
  const isClosedFormLinear =
    terms.length > 0 &&
    eq.equation_type !== 'nonparametric_analytical' &&
    eq.equation_type !== 'exact_decision_rules' &&
    eq.equation_type !== 'constant_rule';

  const copyText = (kind: string, txt?: string) => {
    if (!txt) return;
    navigator.clipboard?.writeText(txt).catch(() => {});
    setCopied(kind);
    setTimeout(() => setCopied(null), 1800);
  };

  const downloadStandalonePy = () => {
    const content = [
      '"""',
      'OmniForge Standalone External Deployment Equation',
      `Model: ${eq.model_name}`,
      `Target: ${eq.target_name || 'target'}`,
      eq.external_usage_summary ? `Summary: ${eq.external_usage_summary}` : '',
      plainFormula ? `Primary Relationship: ${plainFormula}` : '',
      eq.surrogate_plain_equation ? `Closed-Form Equation: ${eq.surrogate_plain_equation}` : '',
      excelFormula ? `\nExcel Formula:\n${excelFormula}` : '',
      sqlFormula ? `\nSQL Expression:\n${sqlFormula}` : '',
      cMatlabFormula ? `\nMATLAB / C Formula:\n${cMatlabFormula}` : '',
      latexFormula ? `\nLaTeX:\n${latexFormula}` : '',
      '"""',
      '',
      pythonFormula || `# ${plainFormula}`,
      '',
    ].filter((s) => s !== undefined).join('\n');
    const blob = new Blob([content], { type: 'text/x-python;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const safeName = `${eq.target_name || 'model'}_${eq.model_name}`.replace(/[^a-zA-Z0-9_-]+/g, '_').slice(0, 56);
    a.href = url;
    a.download = `omniforge_${safeName}_external.py`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const activeCodeText =
    codeTab === 'python'
      ? pythonFormula
      : codeTab === 'excel'
        ? excelFormula
        : codeTab === 'sql'
          ? sqlFormula
          : codeTab === 'c_matlab'
            ? cMatlabFormula
            : latexFormula;

  return (
    <InspectableBox
      title={`${eq.rank ? `Rank #${eq.rank} · ` : ''}${eq.model_name}`}
      subtitle={eq.target_name ? `Target: ${eq.target_name} · ${eq.equation_type || 'Analytical Equation'}` : eq.equation_type}
      className="math-eq-shell"
    >
      <div className="math-eq-card">
        {/* Top Rank + Status + Metrics Header */}
        <div className="row" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
          <div className="row" style={{ gap: 8 }}>
            {eq.rank !== undefined && (
              <span className={`badge ${eq.rank === 1 ? 'badge-accent' : ''}`} style={{ fontWeight: 800 }}>
                {eq.rank === 1 && <Award size={12} />} Rank #{eq.rank}
              </span>
            )}
            <strong style={{ fontSize: 14 }}>{eq.model_name}</strong>
            {(eq.model_paradigm || eq.equation_type === 'nonparametric_analytical') && (
              <span
                className={`badge ${
                  (eq.model_paradigm || '').toLowerCase().includes('non') || eq.equation_type === 'nonparametric_analytical'
                    ? 'badge-accent'
                    : ''
                }`}
                style={{ fontSize: 10.5 }}
              >
                {eq.model_paradigm || 'Non-Parametric Analytical'}
              </span>
            )}
            {eq.acceptance_status && (
              <span className={`badge ${badge.cls}`} style={{ gap: 4 }}>
                {badge.icon} {eq.acceptance_status}
              </span>
            )}
          </div>
          <div className="row" style={{ gap: 6 }}>
            {eq.primary_metric && eq.primary_score !== undefined && (
              <span className="badge badge-accent mono">
                {eq.primary_metric.toUpperCase()} = {Number(eq.primary_score).toFixed(4)}
              </span>
            )}
            {eq.lift_over_naive !== undefined && (
              <span className={`badge mono ${eq.lift_over_naive > 0 ? 'badge-green' : eq.lift_over_naive < 0 ? 'badge-red' : ''}`}>
                Lift vs Naive: {eq.lift_over_naive >= 0 ? '+' : ''}{Number(eq.lift_over_naive).toFixed(4)}
              </span>
            )}
            {latexFormula && (
              <button className="btn btn-sm" onClick={() => copyText('latex', latexFormula)}>
                <Copy size={11} /> {copied === 'latex' ? 'Copied LaTeX!' : 'LaTeX'}
              </button>
            )}
            {pythonFormula && (
              <button className="btn btn-sm" onClick={() => copyText('python', pythonFormula)}>
                <Copy size={11} /> {copied === 'python' ? 'Copied Python!' : 'Python'}
              </button>
            )}
            {hasExternalKit && (
              <button className={`btn btn-sm ${showCode ? 'btn-primary' : ''}`} onClick={() => setShowCode((v) => !v)} title="Show Standalone External Formulas (Python, Excel, SQL, MATLAB/C)">
                <Code2 size={12} /> {showCode ? 'Hide External Code' : 'Use Outside OmniForge'}
              </button>
            )}
            {pythonFormula && (
              <button className="btn btn-sm btn-success" onClick={downloadStandalonePy} title="Download self-contained .py scoring script with Excel/SQL/MATLAB formulas">
                <Download size={12} /> .py
              </button>
            )}
          </div>
        </div>

        {/* Rich Mathematical Formula Display */}
        <div className="math-formula-stage">
          {isClosedFormLinear ? (
            <div className="math-symbolic-line">
              <span className="math-lhs">
                <span className="math-var">ŷ</span>
                {eq.target_name && <sub className="math-sub">{eq.target_name}</sub>}
              </span>
              <span className="math-op">=</span>
              {eq.intercept !== undefined && eq.intercept !== null && (
                <span className="math-const">{Number(eq.intercept).toFixed(4)}</span>
              )}
              {terms.slice(0, 12).map((t, idx) => {
                const pos = t.coefficient >= 0;
                const intensity = Math.max(0.25, Math.min(1, Math.abs(t.coefficient) / maxCoef));
                return (
                  <span key={`${t.feature}-${idx}`} className="math-term-group">
                    <span className="math-op">{idx === 0 && (eq.intercept === undefined || eq.intercept === null) ? (pos ? '' : '−') : pos ? '+' : '−'}</span>
                    <span
                      className={`math-term-pill ${pos ? 'pos' : 'neg'}`}
                      style={{ borderColor: pos ? `rgba(52, 211, 153, ${intensity})` : `rgba(248, 113, 113, ${intensity})` }}
                      title={`Coefficient = ${t.coefficient}`}
                    >
                      <span className="math-coef">{Math.abs(t.coefficient).toFixed(4)}</span>
                      <span className="math-dot">·</span>
                      <span className="math-feat">{t.feature}</span>
                    </span>
                  </span>
                );
              })}
              {terms.length > 12 && (
                <span className="badge" style={{ marginLeft: 4 }}>+{terms.length - 12} more terms</span>
              )}
            </div>
          ) : (
            <div className="math-plain-modern" style={{ lineHeight: 1.55 }}>
              {prettifyMathText(plainFormula || latexFormula || `${eq.model_name}: ŷ = f(x)`)}
            </div>
          )}

          {plainFormula && isClosedFormLinear && (
            <div className="math-plain-caption mono">
              {prettifyMathText(plainFormula)}
            </div>
          )}

          {eq.decision_rules_text && (
            <pre
              className="mono"
              style={{
                marginTop: 8,
                padding: '8px 10px',
                borderRadius: 8,
                background: 'rgba(15, 23, 42, 0.55)',
                border: '1px solid var(--border)',
                fontSize: 11.5,
                color: 'var(--text)',
                overflowX: 'auto',
                whiteSpace: 'pre-wrap',
              }}
            >
              {eq.decision_rules_text}
            </pre>
          )}

          {eq.surrogate_plain_equation && eq.surrogate_plain_equation !== plainFormula && (
            <div
              style={{
                marginTop: 8,
                padding: '7px 10px',
                borderRadius: 8,
                background: 'rgba(56, 189, 248, 0.08)',
                border: '1px solid rgba(56, 189, 248, 0.25)',
                fontSize: 11.5,
              }}
            >
              <strong style={{ color: 'var(--accent)' }}>
                Global Linear Surrogate{eq.surrogate_fidelity_r2 !== undefined && eq.surrogate_fidelity_r2 !== null ? ` (Fidelity R² = ${Number(eq.surrogate_fidelity_r2).toFixed(4)})` : ''}:
              </strong>{' '}
              <span className="mono">{prettifyMathText(eq.surrogate_plain_equation)}</span>
            </div>
          )}

          {eq.architecture_equations && eq.architecture_equations.length > 0 && (
            <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px dashed var(--border)', display: 'flex', flexDirection: 'column', gap: 3 }}>
              {eq.architecture_equations.map((line, i) => (
                <div key={i} className="mono" style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                  • {line}
                </div>
              ))}
            </div>
          )}

          {eq.explanation_notes && eq.explanation_notes.length > 0 && (
            <div style={{ marginTop: 8, paddingTop: 6, borderTop: '1px dashed var(--border)', fontSize: 11.5, color: 'var(--text-muted)' }}>
              {eq.explanation_notes.map((n, i) => (
                <div key={i}>• {n}</div>
              ))}
            </div>
          )}
        </div>

        {/* External Deployment & Portable Formula Kit (Python, Excel, SQL, MATLAB/C, LaTeX) */}
        {showCode && hasExternalKit && (
          <div
            className="external-formula-kit"
            style={{
              marginTop: 10,
              padding: '10px 12px',
              borderRadius: 10,
              border: '1px solid rgba(56, 189, 248, 0.35)',
              background: 'var(--bg-subtle)',
            }}
          >
            <div className="row" style={{ justifyContent: 'space-between', marginBottom: 8, gap: 8 }}>
              <div>
                <strong style={{ fontSize: 12.5, color: 'var(--accent)' }}>
                  External Deployment Formula Kit — Use This Relationship Outside OmniForge
                </strong>
                {eq.external_usage_summary && (
                  <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 2 }}>{eq.external_usage_summary}</div>
                )}
              </div>
              <div className="row" style={{ gap: 6 }}>
                <button className="btn btn-sm" onClick={() => copyText(codeTab, activeCodeText)}>
                  <Copy size={11} /> {copied === codeTab ? 'Copied!' : `Copy ${codeTab.toUpperCase()}`}
                </button>
                <button className="btn btn-sm btn-success" onClick={downloadStandalonePy}>
                  <Download size={11} /> Download Standalone .py
                </button>
              </div>
            </div>
            <div className="tabs" style={{ marginBottom: 8 }}>
              {pythonFormula && (
                <button className={`tab-btn ${codeTab === 'python' ? 'active' : ''}`} onClick={() => setCodeTab('python')}>
                  Python (Standalone)
                </button>
              )}
              {excelFormula && (
                <button className={`tab-btn ${codeTab === 'excel' ? 'active' : ''}`} onClick={() => setCodeTab('excel')}>
                  Excel Formula (=...)
                </button>
              )}
              {sqlFormula && (
                <button className={`tab-btn ${codeTab === 'sql' ? 'active' : ''}`} onClick={() => setCodeTab('sql')}>
                  SQL SELECT
                </button>
              )}
              {cMatlabFormula && (
                <button className={`tab-btn ${codeTab === 'c_matlab' ? 'active' : ''}`} onClick={() => setCodeTab('c_matlab')}>
                  MATLAB / C / PLC
                </button>
              )}
              {latexFormula && (
                <button className={`tab-btn ${codeTab === 'latex' ? 'active' : ''}`} onClick={() => setCodeTab('latex')}>
                  LaTeX
                </button>
              )}
            </div>
            <pre
              className="handbook-formula-box"
              style={{
                margin: 0,
                fontSize: 12,
                maxHeight: 280,
                overflowY: 'auto',
              }}
            >
              {activeCodeText || '# Not available for this format.'}
            </pre>
          </div>
        )}

        {/* Selection Criteria & Explicit Acceptance / Rejection Rationale */}
        {(eq.selection_criterion || (eq.acceptance_reasons && eq.acceptance_reasons.length > 0)) && (
          <div className="math-rationale-box">
            {eq.selection_criterion && (
              <div style={{ fontSize: 12, marginBottom: 6 }}>
                <strong style={{ color: 'var(--accent)' }}>Selection Criterion:</strong>{' '}
                <span>{eq.selection_criterion}</span>
                {eq.naive_floor_score !== undefined && (
                  <span style={{ color: 'var(--text-muted)' }}> · Naive Floor = {Number(eq.naive_floor_score).toFixed(4)}</span>
                )}
              </div>
            )}
            {eq.acceptance_reasons && eq.acceptance_reasons.length > 0 && (
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: 'var(--text-muted)' }}>
                {eq.acceptance_reasons.map((reason, i) => (
                  <li key={i} style={{ marginBottom: 2 }}>{reason}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* Full Metrics Strip */}
        {eq.metrics && Object.keys(eq.metrics).length > 0 && (
          <div className="row" style={{ marginTop: 8, gap: 6 }}>
            {Object.entries(eq.metrics).map(([k, v]) => (
              <span key={k} className="badge mono" style={{ fontSize: 11 }}>
                {k}: {typeof v === 'number' ? v.toFixed(4) : String(v)}
              </span>
            ))}
          </div>
        )}
      </div>
    </InspectableBox>
  );
};
