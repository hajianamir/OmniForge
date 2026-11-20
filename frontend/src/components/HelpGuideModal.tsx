import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Compass,
  HelpCircle,
  Layers,
  Printer,
  Search,
  ShieldCheck,
  Sparkles,
  Table2,
  X,
} from 'lucide-react';
import type { StudioId } from '../types';
import {
  ALL_STUDIO_CHAPTERS,
  CHAPTER_BY_STUDIO,
  HANDBOOK_FAQS,
  MASTER_MODEL_DECISION_ATLAS,
  MASTER_STAT_DECISION_ATLAS,
  MASTER_THRESHOLD_COMPENDIUM,
  WORKFLOW_ROADMAP,
  type DecisionMatrix,
  type MethodEntry,
  type StudioChapter,
  type ThresholdTable,
} from '../data/handbookData';

export type HandbookSectionId =
  | 'workflow'
  | 'stat-atlas'
  | 'model-atlas'
  | 'thresholds'
  | 'faq'
  | `studio-${StudioId}`;

const SECTION_ORDER: HandbookSectionId[] = [
  'workflow',
  'stat-atlas',
  'model-atlas',
  'thresholds',
  ...ALL_STUDIO_CHAPTERS.map((c) => `studio-${c.studioId}` as HandbookSectionId),
  'faq',
];

/**
 * Reusable contextual '?' button placed on every studio header & key governance dialog
 * so users can open that exact section's formulas, numeric threshold tables, and test guide in one click.
 */
export const StudioHelpButton: React.FC<{
  section: HandbookSectionId;
  onOpenHelp: (section: HandbookSectionId) => void;
  title?: string;
  label?: string;
  compact?: boolean;
}> = ({ section, onOpenHelp, title, label, compact = false }) => (
  <button
    type="button"
    className="btn btn-sm studio-help-qbtn"
    data-help-trigger={section}
    onClick={() => onOpenHelp(section)}
    title={title || 'Open Complete Reference Guide for this section (Theory, Formulas, Worked Examples, Thresholds & Troubleshooting)'}
    aria-label={title || `Help guide for ${section}`}
    style={{
      padding: compact ? '3px 9px' : '4px 12px',
    }}
  >
    <HelpCircle size={14} />
    <span>{label ? `? ${label}` : compact ? '?' : '? Guide'}</span>
  </button>
);

const ThresholdTableView: React.FC<{ table: ThresholdTable }> = ({ table }) => (
  <div className="card" style={{ marginBottom: 14 }}>
    <div className="card-title" style={{ marginBottom: 6 }}>
      <span className="row" style={{ gap: 6 }}>
        <Table2 size={14} style={{ color: 'var(--accent)' }} />
        {table.title}
      </span>
    </div>
    {table.subtitle && (
      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>{table.subtitle}</div>
    )}
    <div style={{ overflowX: 'auto' }}>
      <table className="data-table" style={{ width: '100%', fontSize: 12.5 }}>
        <thead>
          <tr>
            <th style={{ width: '22%' }}>Metric / Parameter</th>
            <th style={{ width: '24%' }}>Recommended Numeric Range</th>
            <th style={{ width: '27%' }}>Statistical / Engineering Meaning</th>
            <th style={{ width: '27%' }}>Recommended Action</th>
          </tr>
        </thead>
        <tbody>
          {table.rows.map((r, i) => (
            <tr key={`${r.metric}-${i}`}>
              <td><strong>{r.metric}</strong></td>
              <td>
                <span
                  className={`badge mono ${
                    r.level === 'good'
                      ? 'badge-green'
                      : r.level === 'warn'
                        ? 'badge-amber'
                        : r.level === 'bad'
                          ? 'badge-red'
                          : 'badge-accent'
                  }`}
                  style={{ whiteSpace: 'normal', lineHeight: 1.35 }}
                >
                  {r.range}
                </span>
              </td>
              <td style={{ color: 'var(--text-muted)' }}>{r.interpretation}</td>
              <td>{r.action}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

const DecisionMatrixView: React.FC<{ matrix: DecisionMatrix }> = ({ matrix }) => (
  <div className="card" style={{ marginBottom: 14 }}>
    <div className="card-title" style={{ marginBottom: 8 }}>
      <span className="row" style={{ gap: 6 }}>
        <Compass size={14} style={{ color: 'var(--green)' }} />
        {matrix.title}
      </span>
    </div>
    <div style={{ overflowX: 'auto' }}>
      <table className="data-table" style={{ width: '100%', fontSize: 12.5 }}>
        <thead>
          <tr>
            <th style={{ width: '22%' }}>Application / Goal</th>
            <th style={{ width: '19%' }}>Data Condition</th>
            <th style={{ width: '21%' }}>Recommended Method / Test</th>
            <th style={{ width: '18%' }}>Alternative / Fallback</th>
            <th style={{ width: '20%' }}>Scientific Rationale</th>
          </tr>
        </thead>
        <tbody>
          {matrix.rows.map((r, i) => (
            <tr key={`${r.application}-${i}`}>
              <td><strong>{r.application}</strong></td>
              <td style={{ color: 'var(--text-muted)' }}>{r.dataCondition}</td>
              <td><span className="badge badge-green" style={{ whiteSpace: 'normal', lineHeight: 1.35 }}>{r.recommended}</span></td>
              <td style={{ color: 'var(--text-muted)' }}>{r.alternative}</td>
              <td>{r.rationale}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

const MethodCardView: React.FC<{ m: MethodEntry }> = ({ m }) => (
  <div
    className="card"
    style={{
      marginBottom: 12,
      borderLeft: '3px solid var(--accent)',
    }}
  >
    <div className="row" style={{ justifyContent: 'space-between', marginBottom: 6, gap: 8 }}>
      <strong style={{ fontSize: 14.5 }}>{m.name}</strong>
      <span className="badge badge-accent">{m.paradigm}</span>
    </div>

    <div style={{ fontSize: 13, lineHeight: 1.6, marginBottom: 8 }}>
      <strong style={{ color: 'var(--accent)' }}>What It Fundamentally Does:</strong> {m.whatItDoes}
    </div>

    {m.nullHypothesis && (
      <div
        style={{
          fontSize: 12.5,
          padding: '6px 10px',
          borderRadius: 6,
          background: 'rgba(99, 102, 241, 0.10)',
          border: '1px solid rgba(99, 102, 241, 0.28)',
          marginBottom: 8,
        }}
      >
        <strong style={{ color: '#a5b4fc' }}>Null Hypothesis (H₀):</strong> {m.nullHypothesis}
      </div>
    )}

    {m.formula && (
      <div className="handbook-formula-box" style={{ marginBottom: 8 }}>
        <div style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: 0.6, color: 'var(--accent)', marginBottom: 4, fontWeight: 700, fontFamily: 'inherit' }}>
          Mathematical Formula &amp; Relationship
        </div>
        <div className="mono" style={{ fontSize: 13, lineHeight: 1.55, overflowX: 'auto' }}>
          {m.formula.equation}
        </div>
        <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 4, fontFamily: 'inherit' }}>
          <strong>Where:</strong> {m.formula.symbols}
        </div>
      </div>
    )}

    <div className="grid-2" style={{ gap: 8, fontSize: 12.5 }}>
      <div style={{ padding: '6px 8px', borderRadius: 6, background: 'var(--bg-subtle)' }}>
        <div style={{ color: 'var(--green)', fontWeight: 700, marginBottom: 2 }}>
          <CheckCircle2 size={12} style={{ display: 'inline', marginRight: 4, verticalAlign: -1 }} />
          When to Use
        </div>
        <div>{m.whenToUse}</div>
        {m.assumptions && (
          <div style={{ marginTop: 4, fontSize: 11.5, color: 'var(--text-muted)' }}>
            <strong>Assumptions:</strong> {m.assumptions}
          </div>
        )}
      </div>
      <div style={{ padding: '6px 8px', borderRadius: 6, background: 'var(--bg-subtle)' }}>
        <div style={{ color: '#fbbf24', fontWeight: 700, marginBottom: 2 }}>
          <AlertTriangle size={12} style={{ display: 'inline', marginRight: 4, verticalAlign: -1 }} />
          Recommended Parameters &amp; Pitfalls
        </div>
        <div><strong>Settings:</strong> {m.recommendedParams}</div>
        <div style={{ marginTop: 4, fontSize: 11.5, color: 'var(--text-muted)' }}>
          <strong>Watch-out:</strong> {m.pitfalls}
        </div>
      </div>
    </div>
  </div>
);

const StudioChapterView: React.FC<{
  ch: StudioChapter;
  onJumpStudio: (id: StudioId) => void;
  onClose: () => void;
}> = ({ ch, onJumpStudio, onClose }) => {
  const scrollToAnchor = (id: string) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <div>
      {/* Chapter Header Banner */}
      <div
        className="card"
        style={{
          background: 'linear-gradient(135deg, rgba(56, 189, 248, 0.14), rgba(99, 102, 241, 0.10))',
          borderColor: 'rgba(56, 189, 248, 0.40)',
          marginBottom: 12,
        }}
      >
        <div className="row" style={{ justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 }}>
          <div>
            <div className="row" style={{ gap: 8, marginBottom: 4 }}>
              <span className="badge badge-accent" style={{ fontWeight: 800 }}>{ch.chapterNum} · STUDIO {ch.studioId}</span>
              <span className="badge badge-green">Comprehensive Textbook Chapter</span>
            </div>
            <h2 style={{ margin: '4px 0', fontSize: 19.5 }}>{ch.title}</h2>
            <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>{ch.subtitle}</div>
          </div>
          <button
            className="btn btn-primary btn-sm"
            onClick={() => {
              onJumpStudio(ch.studioId);
              onClose();
            }}
          >
            Open Studio {ch.studioId} &rarr;
          </button>
        </div>
        <p style={{ margin: '10px 0 8px', fontSize: 13.5, lineHeight: 1.7 }}>{ch.overview}</p>
        <div style={{ fontSize: 12.2, padding: '7px 11px', borderRadius: 6, background: 'var(--bg-subtle)', border: '1px solid var(--border)' }}>
          <strong style={{ color: 'var(--green)' }}>Data Provenance &amp; Lineage Rule:</strong> {ch.provenanceRule}
        </div>
      </div>

      {/* In-Chapter Quick-Jump Navigation Pills */}
      <div className="handbook-subnav">
        <span style={{ fontSize: 11, fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', alignSelf: 'center', marginRight: 4 }}>
          Jump Inside Chapter:
        </span>
        {ch.deepTheory && ch.deepTheory.length > 0 && (
          <button type="button" className="btn btn-sm" onClick={() => scrollToAnchor(`ch-${ch.studioId}-theory`)}>
            1. Deep Theory ({ch.deepTheory.length})
          </button>
        )}
        <button type="button" className="btn btn-sm" onClick={() => scrollToAnchor(`ch-${ch.studioId}-workflow`)}>
          2. Procedure
        </button>
        <button type="button" className="btn btn-sm" onClick={() => scrollToAnchor(`ch-${ch.studioId}-methods`)}>
          3. Formulas &amp; Tests ({ch.methods.length})
        </button>
        {ch.workedExamples && ch.workedExamples.length > 0 && (
          <button type="button" className="btn btn-sm" onClick={() => scrollToAnchor(`ch-${ch.studioId}-examples`)}>
            4. Worked Examples ({ch.workedExamples.length})
          </button>
        )}
        {ch.uiControlsGuide && ch.uiControlsGuide.length > 0 && (
          <button type="button" className="btn btn-sm" onClick={() => scrollToAnchor(`ch-${ch.studioId}-controls`)}>
            5. UI Controls ({ch.uiControlsGuide.length})
          </button>
        )}
        <button type="button" className="btn btn-sm" onClick={() => scrollToAnchor(`ch-${ch.studioId}-thresholds`)}>
          6. Thresholds
        </button>
        <button type="button" className="btn btn-sm" onClick={() => scrollToAnchor(`ch-${ch.studioId}-matrix`)}>
          7. Decision Matrix
        </button>
        {ch.troubleshootingPlaybook && ch.troubleshootingPlaybook.length > 0 && (
          <button type="button" className="btn btn-sm" onClick={() => scrollToAnchor(`ch-${ch.studioId}-troubleshoot`)}>
            8. Troubleshooting ({ch.troubleshootingPlaybook.length})
          </button>
        )}
      </div>

      {/* 1. Deep Theoretical Foundations & First Principles */}
      {ch.deepTheory && ch.deepTheory.length > 0 && (
        <div id={`ch-${ch.studioId}-theory`} style={{ marginBottom: 14 }}>
          <h3 style={{ margin: '8px 0 10px', fontSize: 15.5, display: 'flex', alignItems: 'center', gap: 6 }}>
            <BookOpen size={15} style={{ color: 'var(--accent)' }} />
            Deep Theoretical Foundations &amp; Engineering First Principles
          </h3>
          {ch.deepTheory.map((sec, idx) => (
            <div key={`th-${idx}`} className="card" style={{ marginBottom: 10, borderLeft: '3px solid #6366f1' }}>
              <div style={{ fontWeight: 800, fontSize: 14, marginBottom: 8, color: 'var(--text)' }}>{sec.heading}</div>
              {sec.paragraphs.map((p, pi) => (
                <p key={pi} style={{ margin: pi === sec.paragraphs.length - 1 ? 0 : '0 0 9px', fontSize: 13.2, lineHeight: 1.72, color: 'var(--text)' }}>
                  {p}
                </p>
              ))}
            </div>
          ))}
        </div>
      )}

      {/* 2. Applications & Step-by-Step */}
      <div id={`ch-${ch.studioId}-workflow`} className="grid-2" style={{ gap: 12, marginBottom: 14 }}>
        <div className="card" style={{ marginBottom: 0 }}>
          <div className="card-title" style={{ marginBottom: 6 }}>Real-World Applications</div>
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12.8, lineHeight: 1.65 }}>
            {ch.realWorldApplications.map((a, i) => (
              <li key={i} style={{ marginBottom: 5 }}>{a}</li>
            ))}
          </ul>
        </div>
        <div className="card" style={{ marginBottom: 0 }}>
          <div className="card-title" style={{ marginBottom: 6 }}>Step-by-Step Procedure Inside Studio {ch.studioId}</div>
          <ol style={{ margin: 0, paddingLeft: 20, fontSize: 12.8, lineHeight: 1.65 }}>
            {ch.stepByStep.map((s, i) => (
              <li key={i} style={{ marginBottom: 5 }}>{s}</li>
            ))}
          </ol>
        </div>
      </div>

      {/* 3. Formulas, Algorithms & Tests */}
      <div id={`ch-${ch.studioId}-methods`} style={{ marginBottom: 14 }}>
        <h3 style={{ margin: '8px 0 10px', fontSize: 15.5, display: 'flex', alignItems: 'center', gap: 6 }}>
          <Sparkles size={15} style={{ color: 'var(--accent)' }} />
          Core Algorithms, Statistical Tests, Mathematical Formulas &amp; Rules ({ch.methods.length})
        </h3>
        {ch.methods.map((m, idx) => (
          <MethodCardView key={`${ch.studioId}-m-${idx}`} m={m} />
        ))}
      </div>

      {/* 4. Step-by-Step Worked Numerical Examples */}
      {ch.workedExamples && ch.workedExamples.length > 0 && (
        <div id={`ch-${ch.studioId}-examples`} style={{ marginBottom: 14 }}>
          <h3 style={{ margin: '8px 0 10px', fontSize: 15.5, display: 'flex', alignItems: 'center', gap: 6 }}>
            <CheckCircle2 size={15} style={{ color: 'var(--green)' }} />
            Step-by-Step Worked Numerical Examples ({ch.workedExamples.length})
          </h3>
          {ch.workedExamples.map((ex, idx) => (
            <div key={`ex-${idx}`} className="card" style={{ marginBottom: 10, borderLeft: '3px solid var(--green)' }}>
              <div style={{ fontWeight: 800, fontSize: 14, color: 'var(--green)', marginBottom: 6 }}>{ex.title}</div>
              <div style={{ fontSize: 12.8, marginBottom: 8, padding: '6px 10px', borderRadius: 6, background: 'var(--bg-subtle)' }}>
                <strong>Scenario:</strong> {ex.scenario}
              </div>
              <div className="handbook-formula-box" style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: 0.6, color: 'var(--accent)', marginBottom: 5, fontWeight: 700, fontFamily: 'inherit' }}>
                  Complete Arithmetic Walkthrough
                </div>
                {ex.stepByStepCalculation.map((line, li) => (
                  <div key={li} style={{ marginBottom: 3 }}>{line}</div>
                ))}
              </div>
              <div className="grid-2" style={{ gap: 8, fontSize: 12.5 }}>
                <div style={{ padding: '7px 10px', borderRadius: 6, background: 'rgba(16, 185, 129, 0.10)', border: '1px solid rgba(16, 185, 129, 0.25)' }}>
                  <strong style={{ color: 'var(--green)' }}>Numerical Conclusion:</strong> {ex.numericalConclusion}
                </div>
                <div style={{ padding: '7px 10px', borderRadius: 6, background: 'rgba(56, 189, 248, 0.10)', border: '1px solid rgba(56, 189, 248, 0.25)' }}>
                  <strong style={{ color: 'var(--accent)' }}>Recommended Engineering Action:</strong> {ex.engineeringAction}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* 5. Complete Studio UI Controls & Parameter Tuning Guide */}
      {ch.uiControlsGuide && ch.uiControlsGuide.length > 0 && (
        <div id={`ch-${ch.studioId}-controls`} className="card" style={{ marginBottom: 14 }}>
          <div className="card-title" style={{ marginBottom: 8 }}>
            <span className="row" style={{ gap: 6 }}>
              <Layers size={14} style={{ color: 'var(--accent)' }} />
              Complete Studio {ch.studioId} UI Controls &amp; Parameter Tuning Reference
            </span>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table" style={{ width: '100%', fontSize: 12.4 }}>
              <thead>
                <tr>
                  <th style={{ width: '19%' }}>UI Control / Setting</th>
                  <th style={{ width: '15%' }}>Default</th>
                  <th style={{ width: '21%' }}>Effect When Increased / Enabled</th>
                  <th style={{ width: '21%' }}>Effect When Decreased / Restricted</th>
                  <th style={{ width: '24%' }}>Expert Guidance</th>
                </tr>
              </thead>
              <tbody>
                {ch.uiControlsGuide.map((row, idx) => (
                  <tr key={`ctrl-${idx}`}>
                    <td><strong>{row.control}</strong></td>
                    <td><span className="badge badge-accent mono" style={{ whiteSpace: 'normal' }}>{row.defaultValue}</span></td>
                    <td style={{ color: 'var(--text-muted)' }}>{row.whenIncreased}</td>
                    <td style={{ color: 'var(--text-muted)' }}>{row.whenDecreased}</td>
                    <td>{row.expertGuidance}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 6. Recommended Numeric Threshold Tables */}
      <div id={`ch-${ch.studioId}-thresholds`}>
        {ch.thresholdTables.map((tbl, idx) => (
          <ThresholdTableView key={`${ch.studioId}-t-${idx}`} table={tbl} />
        ))}
      </div>

      {/* 7. Decision Matrix */}
      <div id={`ch-${ch.studioId}-matrix`}>
        <DecisionMatrixView matrix={ch.decisionMatrix} />
      </div>

      {/* 8. Diagnostic Troubleshooting & Failure-Mode Remediation Playbook */}
      {ch.troubleshootingPlaybook && ch.troubleshootingPlaybook.length > 0 && (
        <div id={`ch-${ch.studioId}-troubleshoot`} className="card" style={{ marginBottom: 14, borderLeft: '3px solid #f59e0b' }}>
          <div className="card-title" style={{ marginBottom: 8 }}>
            <span className="row" style={{ gap: 6, color: '#fbbf24' }}>
              <AlertTriangle size={15} />
              Diagnostic Troubleshooting &amp; Failure-Mode Playbook (Studio {ch.studioId})
            </span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {ch.troubleshootingPlaybook.map((item, idx) => (
              <div key={`tr-${idx}`} style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--bg-subtle)', border: '1px solid var(--border)' }}>
                <div style={{ fontWeight: 800, fontSize: 13.2, color: '#fbbf24', marginBottom: 5 }}>
                  Symptom: {item.symptom}
                </div>
                <div style={{ fontSize: 12.6, marginBottom: 4 }}>
                  <strong>Underlying Mathematical / Engineering Cause:</strong> {item.mathematicalCause}
                </div>
                <div style={{ fontSize: 12.6, color: 'var(--green)' }}>
                  <strong>Exact Remediation Steps:</strong> {item.exactRemediationSteps}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

interface SearchHit {
  section: HandbookSectionId;
  sectionLabel: string;
  title: string;
  snippet: string;
  badge: string;
}

export const HelpGuideModal: React.FC<{
  open: boolean;
  initialSection?: HandbookSectionId;
  onClose: () => void;
  onJumpStudio: (id: StudioId) => void;
}> = ({ open, initialSection = 'workflow', onClose, onJumpStudio }) => {
  const [section, setSection] = useState<HandbookSectionId>(initialSection);
  const [query, setQuery] = useState('');
  const bodyRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (open) {
      setSection(initialSection || 'workflow');
      setQuery('');
    }
  }, [open, initialSection]);

  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = 0;
    }
  }, [section]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  const q = query.trim().toLowerCase();

  const searchHits = useMemo<SearchHit[]>(() => {
    if (!q) return [];
    const hits: SearchHit[] = [];

    for (const w of WORKFLOW_ROADMAP) {
      const hay = `${w.stageName} ${w.objective} ${w.exactActions.join(' ')} ${w.criticalNote || ''}`.toLowerCase();
      if (hay.includes(q)) {
        hits.push({
          section: 'workflow',
          sectionLabel: 'Step-by-Step Workflow',
          title: `Step ${w.step}: ${w.stageName}`,
          snippet: w.objective,
          badge: `Studio ${w.studioId}`,
        });
      }
    }

    for (const ch of ALL_STUDIO_CHAPTERS) {
      const chSec = `studio-${ch.studioId}` as HandbookSectionId;
      if (`${ch.title} ${ch.subtitle} ${ch.overview}`.toLowerCase().includes(q)) {
        hits.push({
          section: chSec,
          sectionLabel: `${ch.chapterNum} · Studio ${ch.studioId}`,
          title: ch.title,
          snippet: ch.overview.slice(0, 180) + '...',
          badge: 'Chapter Overview',
        });
      }
      for (const m of ch.methods) {
        const hay = `${m.name} ${m.paradigm} ${m.whatItDoes} ${m.nullHypothesis || ''} ${m.formula?.equation || ''} ${m.formula?.symbols || ''} ${m.whenToUse} ${m.recommendedParams} ${m.pitfalls}`.toLowerCase();
        if (hay.includes(q)) {
          hits.push({
            section: chSec,
            sectionLabel: `${ch.chapterNum} · Studio ${ch.studioId}`,
            title: m.name,
            snippet: m.formula ? `${m.formula.equation} — ${m.whatItDoes.slice(0, 120)}...` : m.whatItDoes.slice(0, 170) + '...',
            badge: m.paradigm,
          });
        }
      }
      for (const tbl of ch.thresholdTables) {
        for (const r of tbl.rows) {
          if (`${tbl.title} ${r.metric} ${r.range} ${r.interpretation} ${r.action}`.toLowerCase().includes(q)) {
            hits.push({
              section: chSec,
              sectionLabel: `${ch.chapterNum} · Studio ${ch.studioId}`,
              title: `Threshold: ${r.metric} (${r.range})`,
              snippet: `${r.interpretation} — ${r.action}`,
              badge: 'Numeric Table',
            });
          }
        }
      }
      for (const sec of ch.deepTheory || []) {
        const hay = `${sec.heading} ${sec.paragraphs.join(' ')}`.toLowerCase();
        if (hay.includes(q)) {
          hits.push({
            section: chSec,
            sectionLabel: `${ch.chapterNum} · Studio ${ch.studioId}`,
            title: sec.heading,
            snippet: sec.paragraphs[0].slice(0, 185) + '...',
            badge: 'Deep Theory',
          });
        }
      }
      for (const ex of ch.workedExamples || []) {
        const hay = `${ex.title} ${ex.scenario} ${ex.stepByStepCalculation.join(' ')} ${ex.numericalConclusion} ${ex.engineeringAction}`.toLowerCase();
        if (hay.includes(q)) {
          hits.push({
            section: chSec,
            sectionLabel: `${ch.chapterNum} · Studio ${ch.studioId}`,
            title: ex.title,
            snippet: `${ex.numericalConclusion} — ${ex.engineeringAction}`.slice(0, 185),
            badge: 'Worked Example',
          });
        }
      }
      for (const tr of ch.troubleshootingPlaybook || []) {
        const hay = `${tr.symptom} ${tr.mathematicalCause} ${tr.exactRemediationSteps}`.toLowerCase();
        if (hay.includes(q)) {
          hits.push({
            section: chSec,
            sectionLabel: `${ch.chapterNum} · Studio ${ch.studioId}`,
            title: `Troubleshoot: ${tr.symptom}`,
            snippet: `${tr.mathematicalCause} -> ${tr.exactRemediationSteps}`.slice(0, 185),
            badge: 'Troubleshooting',
          });
        }
      }
      for (const ctrl of ch.uiControlsGuide || []) {
        const hay = `${ctrl.control} ${ctrl.defaultValue} ${ctrl.whenIncreased} ${ctrl.whenDecreased} ${ctrl.expertGuidance}`.toLowerCase();
        if (hay.includes(q)) {
          hits.push({
            section: chSec,
            sectionLabel: `${ch.chapterNum} · Studio ${ch.studioId}`,
            title: `Control: ${ctrl.control}`,
            snippet: ctrl.expertGuidance,
            badge: 'UI Control',
          });
        }
      }
    }

    for (const f of HANDBOOK_FAQS) {
      if (`${f.q} ${f.a}`.toLowerCase().includes(q)) {
        hits.push({
          section: 'faq',
          sectionLabel: 'Best Practices & FAQ',
          title: f.q,
          snippet: f.a.slice(0, 180) + '...',
          badge: 'FAQ',
        });
      }
    }

    return hits.slice(0, 40);
  }, [q]);

  if (!open) return null;

  const curIdx = SECTION_ORDER.indexOf(section);
  const prevSec = curIdx > 0 ? SECTION_ORDER[curIdx - 1] : null;
  const nextSec = curIdx >= 0 && curIdx < SECTION_ORDER.length - 1 ? SECTION_ORDER[curIdx + 1] : null;

  const activeStudioChapter: StudioChapter | null = section.startsWith('studio-')
    ? CHAPTER_BY_STUDIO[section.replace('studio-', '') as StudioId] || null
    : null;

  return (
    <div className="modal-backdrop" onClick={onClose} style={{ zIndex: 1300 }}>
      <div
        className="modal-panel handbook-modal handbook-modal-panel"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(1340px, 96vw)',
          maxWidth: '96vw',
          height: '92vh',
          maxHeight: '92vh',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          padding: 0,
        }}
      >
        {/* Top Book Header */}
        <div
          className="row"
          style={{
            justifyContent: 'space-between',
            padding: '12px 18px',
            borderBottom: '1px solid var(--border)',
            background: 'var(--bg-subtle)',
            gap: 12,
            flexWrap: 'wrap',
          }}
        >
          <div className="row" style={{ gap: 10 }}>
            <BookOpen size={20} style={{ color: 'var(--accent)' }} />
            <div>
              <div style={{ fontWeight: 800, fontSize: 16 }}>
                OmniForge Comprehensive Reference Handbook &amp; Analytical Manual
              </div>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                Complete End-to-End CRISP-DM Guide · Mathematical Formulas · Recommended Numeric Thresholds · Application-to-Test Atlases · All 14 Studios
              </div>
            </div>
          </div>

          <div className="row" style={{ gap: 8, flex: '1 1 280px', justifyContent: 'flex-end' }}>
            <div style={{ position: 'relative', minWidth: 240, maxWidth: 340, flex: '1 1 240px' }}>
              <Search size={14} style={{ position: 'absolute', left: 10, top: 9, color: 'var(--text-muted)' }} />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search formulas, Shapiro-Wilk, VIF, PSI, GAM, Split Lock..."
                style={{ width: '100%', paddingLeft: 32, height: 32, fontSize: 12.5 }}
              />
            </div>
            <button className="btn btn-sm" onClick={() => window.print()} title="Print Current Handbook View">
              <Printer size={13} /> Print
            </button>
            <button className="btn btn-sm" onClick={onClose} aria-label="Close Handbook">
              <X size={14} /> Close
            </button>
          </div>
        </div>

        {/* Two-Pane Book Layout: Left TOC + Right Content Reader */}
        <div className="handbook-body-layout" style={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden' }}>
          {/* Left Table of Contents */}
          <nav
            className="handbook-toc handbook-toc-sidebar"
            style={{
              width: 268,
              minWidth: 240,
              borderRight: '1px solid var(--border)',
              overflowY: 'auto',
              padding: '12px 10px',
              background: 'var(--bg-subtle)',
              display: 'flex',
              flexDirection: 'column',
              gap: 3,
            }}
          >
            <div className="sidebar-section" style={{ marginTop: 0 }}>Part I · Master Reference</div>
            <button
              className={`nav-item ${!q && section === 'workflow' ? 'active' : ''}`}
              onClick={() => { setQuery(''); setSection('workflow'); }}
              style={{ fontSize: 12.5 }}
            >
              <Compass size={13} /> Step-by-Step Workflow
            </button>
            <button
              className={`nav-item ${!q && section === 'stat-atlas' ? 'active' : ''}`}
              onClick={() => { setQuery(''); setSection('stat-atlas'); }}
              style={{ fontSize: 12.5 }}
            >
              <Table2 size={13} /> Statistical Test Atlas
            </button>
            <button
              className={`nav-item ${!q && section === 'model-atlas' ? 'active' : ''}`}
              onClick={() => { setQuery(''); setSection('model-atlas'); }}
              style={{ fontSize: 12.5 }}
            >
              <Sparkles size={13} /> Model Selection Atlas
            </button>
            <button
              className={`nav-item ${!q && section === 'thresholds' ? 'active' : ''}`}
              onClick={() => { setQuery(''); setSection('thresholds'); }}
              style={{ fontSize: 12.5 }}
            >
              <ShieldCheck size={13} /> Numeric Thresholds
            </button>
            <button
              className={`nav-item ${!q && section === 'faq' ? 'active' : ''}`}
              onClick={() => { setQuery(''); setSection('faq'); }}
              style={{ fontSize: 12.5 }}
            >
              <HelpCircle size={13} /> Best Practices &amp; FAQ
            </button>

            <div className="sidebar-section" style={{ marginTop: 12 }}>Part II · Studio Chapters (01–14)</div>
            {ALL_STUDIO_CHAPTERS.map((ch) => {
              const secId = `studio-${ch.studioId}` as HandbookSectionId;
              return (
                <button
                  key={ch.studioId}
                  data-toc-studio={ch.studioId}
                  className={`nav-item ${!q && section === secId ? 'active' : ''}`}
                  onClick={() => { setQuery(''); setSection(secId); }}
                  style={{ fontSize: 12, padding: '6px 8px' }}
                  title={`${ch.chapterNum}: ${ch.title}`}
                >
                  <span className="nav-code">{ch.studioId}</span>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {ch.title}
                  </span>
                </button>
              );
            })}
          </nav>

          {/* Right Book Content Area */}
          <div
            ref={bodyRef}
            style={{
              flex: 1,
              overflowY: 'auto',
              padding: '18px 22px',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {/* Live Search Mode */}
            {q ? (
              <div>
                <div className="row" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
                  <h3 style={{ margin: 0 }}>
                    Search Results for &ldquo;{query}&rdquo; ({searchHits.length})
                  </h3>
                  <button className="btn btn-sm" onClick={() => setQuery('')}>Clear Search</button>
                </div>
                {searchHits.length === 0 ? (
                  <div className="alert alert-info">
                    No matching handbook entries found. Try searching for a test name (e.g. Shapiro-Wilk, Welch, Kruskal, Mann-Whitney, Chi-Square), model (GAM, Kernel Ridge, AutoML, FT-Transformer), or metric (VIF, PSI, VaR, Silhouette, Split Lock).
                  </div>
                ) : (
                  searchHits.map((hit, idx) => (
                    <div
                      key={`${hit.section}-${idx}`}
                      className="card"
                      style={{ marginBottom: 10, cursor: 'pointer' }}
                      onClick={() => {
                        setSection(hit.section);
                        setQuery('');
                      }}
                    >
                      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 4 }}>
                        <strong style={{ color: 'var(--accent)', fontSize: 13.5 }}>{hit.title}</strong>
                        <div className="row" style={{ gap: 6 }}>
                          <span className="badge">{hit.sectionLabel}</span>
                          <span className="badge badge-accent">{hit.badge}</span>
                        </div>
                      </div>
                      <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>{hit.snippet}</div>
                    </div>
                  ))
                )}
              </div>
            ) : (
              <>
                {/* SECTION: COMPLETE WORKFLOW */}
                {section === 'workflow' && (
                  <div>
                    <div className="card" style={{ background: 'linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(16, 185, 129, 0.08))', borderColor: 'rgba(56, 189, 248, 0.35)' }}>
                      <div className="row" style={{ gap: 8, marginBottom: 6 }}>
                        <span className="badge badge-accent" style={{ fontWeight: 800 }}>PART I · CHAPTER 0</span>
                        <span className="badge badge-green">Zero-Leakage CRISP-DM Roadmap</span>
                      </div>
                      <h2 style={{ margin: '4px 0 8px', fontSize: 19 }}>Exact Step-by-Step Order to Reach Scientifically Verified Results</h2>
                      <p style={{ margin: '0 0 10px', fontSize: 13.5, lineHeight: 1.65 }}>
                        OmniForge follows a strict, leak-free analytical lifecycle. Follow these 13 ordered stages so every model, statistical test, equation, executive report, and live supervisory alert builds cleanly on audited data.
                      </p>
                      <div
                        style={{
                          padding: '10px 12px',
                          borderRadius: 8,
                          background: 'rgba(245, 158, 11, 0.14)',
                          border: '1px solid rgba(245, 158, 11, 0.42)',
                          fontSize: 12.5,
                          lineHeight: 1.55,
                        }}
                      >
                        <strong style={{ color: '#fbbf24' }}>Critical Anti-Leakage Rule — Split Lock BEFORE Data Cleaning:</strong>{' '}
                        Whenever your workflow includes predictive modeling (Studios 03, 05, 06, 07, 11, or 14), enable <strong>Split Lock</strong> right after loading data and picking Targets — <em>before</em> running Studio 01 Data Cleaning or Studio 02 Clustering. With Split Lock active, OmniForge fits all imputers, outlier fences, scalers, and clusters strictly on Train (<code>fit_transform_split</code>) and applies them to Test with zero data leakage.
                      </div>
                      <div
                        style={{
                          marginTop: 10,
                          padding: '10px 12px',
                          borderRadius: 6,
                          background: 'rgba(56,189,248,0.10)',
                          border: '1px solid rgba(56,189,248,0.35)',
                          fontSize: 12.5,
                        }}
                      >
                        <strong style={{ color: '#38bdf8' }}>Latest Architecture &amp; Capabilities:</strong>{' '}
                        <strong>Studio 00</strong> now handles Data Ingestion, Intake Defect Audit, Split Lock, and the <strong>Data Relationships Chart &amp; Table</strong> (Pearson, Spearman &amp; Distance Correlation). <strong>Studio 01</strong> is dedicated to Data Preparation &amp; Cleaning with a transparent <strong>Step-by-Step Pipeline Log</strong> (<code>steps_applied</code>), unclipped raw view preservation (<code>uap_raw_df</code>), <strong>Benford&apos;s Law</strong>, <strong>MCD Robust Mahalanobis</strong>, and <strong>Cleaning Distortion Audit</strong> (<code>/s01/advanced_audit</code>). <strong>Studio 02</strong> includes <strong>FP-Growth</strong>, <strong>PrefixSpan</strong>, <strong>Subgroup Beam Search</strong>, <strong>Hopkins Tendency</strong>, <strong>Bootstrap ARI</strong>, and <strong>Distance Correlation</strong> (<code>/s02/advanced_mining</code>). Local browser UI sessions automatically bootstrap authentication on localhost.
                      </div>
                    </div>

                    {WORKFLOW_ROADMAP.map((w) => (
                      <div key={w.step} className="card" style={{ marginBottom: 12 }}>
                        <div className="row" style={{ justifyContent: 'space-between', marginBottom: 6, gap: 8 }}>
                          <div className="row" style={{ gap: 8 }}>
                            <span className="badge badge-accent" style={{ fontWeight: 800 }}>STEP {w.step}</span>
                            <strong style={{ fontSize: 14.5 }}>{w.stageName}</strong>
                            <span
                              className={`badge ${
                                w.requirement === 'MANDATORY'
                                  ? 'badge-green'
                                  : w.requirement === 'RECOMMENDED'
                                    ? 'badge-accent'
                                    : 'badge-amber'
                              }`}
                            >
                              {w.requirement}
                            </span>
                          </div>
                          <div className="row" style={{ gap: 6 }}>
                            <button className="btn btn-sm" onClick={() => setSection(`studio-${w.studioId}` as HandbookSectionId)}>
                              <Layers size={12} /> Read Chapter {w.studioId}
                            </button>
                            <button
                              className="btn btn-sm btn-primary"
                              onClick={() => {
                                onJumpStudio(w.studioId);
                                onClose();
                              }}
                            >
                              Go to Studio {w.studioId} &rarr;
                            </button>
                          </div>
                        </div>
                        <p style={{ margin: '4px 0 8px', fontSize: 13 }}>{w.objective}</p>
                        <ol style={{ margin: '0 0 8px', paddingLeft: 20, fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.6 }}>
                          {w.exactActions.map((act, i) => (
                            <li key={i} style={{ marginBottom: 3 }}>{act}</li>
                          ))}
                        </ol>
                        <div style={{ fontSize: 12, padding: '6px 8px', borderRadius: 6, background: 'var(--bg-subtle)' }}>
                          <strong style={{ color: 'var(--green)' }}>Outputs Produced:</strong> {w.expectedOutputs}
                        </div>
                        {w.criticalNote && (
                          <div
                            style={{
                              marginTop: 8,
                              padding: '8px 10px',
                              borderRadius: 6,
                              background: 'rgba(245, 158, 11, 0.12)',
                              border: '1px solid rgba(245, 158, 11, 0.35)',
                              fontSize: 12,
                            }}
                          >
                            <strong style={{ color: '#fbbf24' }}>Rule:</strong> {w.criticalNote}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* SECTION: MASTER STATISTICAL TEST ATLAS */}
                {section === 'stat-atlas' && (
                  <div>
                    <h2 style={{ marginTop: 0, fontSize: 19 }}>Master Statistical &amp; Hypothesis Test Selection Atlas</h2>
                    <p style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: 0 }}>
                      Use this quick-lookup atlas to pick the exact parametric or non-parametric statistical test for your data type, sample size, and distributional shape. Click <strong>Chapter 03</strong> or <strong>Chapter 04</strong> in the left contents for complete formulas and Null Hypotheses (H₀).
                    </p>
                    <DecisionMatrixView matrix={MASTER_STAT_DECISION_ATLAS} />
                    <ThresholdTableView table={MASTER_THRESHOLD_COMPENDIUM} />
                  </div>
                )}

                {/* SECTION: MASTER MODEL SELECTION ATLAS */}
                {section === 'model-atlas' && (
                  <div>
                    <h2 style={{ marginTop: 0, fontSize: 19 }}>Master Predictive Model Selection Atlas</h2>
                    <p style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: 0 }}>
                      Compare Parametric Baselines, True Non-Parametric Analytical Models (closed-form equations &amp; smooth GAM curves), Multi-Backend AutoML, and Paper-Grade Deep Learning architectures side by side. See <strong>Chapter 06</strong> for full formulas.
                    </p>
                    <DecisionMatrixView matrix={MASTER_MODEL_DECISION_ATLAS} />
                  </div>
                )}

                {/* SECTION: NUMERIC THRESHOLD COMPENDIUM */}
                {section === 'thresholds' && (
                  <div>
                    <h2 style={{ marginTop: 0, fontSize: 19 }}>Universal Statistical &amp; Engineering Numeric Threshold Compendium</h2>
                    <p style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: 0 }}>
                      Consolidates recommended numerical cutoffs across data quality, clustering, hypothesis testing, effect sizes, multicollinearity, model evaluation, fairness, drift, and live supervisory SPC alerting.
                    </p>
                    <ThresholdTableView table={MASTER_THRESHOLD_COMPENDIUM} />
                    {ALL_STUDIO_CHAPTERS.flatMap((ch) =>
                      ch.thresholdTables.map((t, idx) => (
                        <ThresholdTableView
                          key={`all-thr-${ch.studioId}-${idx}`}
                          table={{ ...t, title: `Studio ${ch.studioId} — ${t.title}` }}
                        />
                      )),
                    )}
                  </div>
                )}

                {/* SECTION: FAQ */}
                {section === 'faq' && (
                  <div>
                    <h2 style={{ marginTop: 0, fontSize: 19 }}>Operational Best Practices &amp; Frequently Asked Questions</h2>
                    {HANDBOOK_FAQS.map((f, idx) => (
                      <div key={idx} className="card" style={{ marginBottom: 12 }}>
                        <div style={{ fontWeight: 700, fontSize: 14.5, color: 'var(--accent)', marginBottom: 6 }}>
                          Q{idx + 1}. {f.q}
                        </div>
                        <div style={{ fontSize: 13, lineHeight: 1.65 }}>{f.a}</div>
                      </div>
                    ))}
                  </div>
                )}

                {/* SECTION: STUDIO CHAPTER 01..14 */}
                {activeStudioChapter && (
                  <StudioChapterView
                    ch={activeStudioChapter}
                    onJumpStudio={onJumpStudio}
                    onClose={onClose}
                  />
                )}

                {/* Book Prev / Next Chapter Footer */}
                <div
                  className="row"
                  style={{
                    justifyContent: 'space-between',
                    marginTop: 18,
                    paddingTop: 12,
                    borderTop: '1px solid var(--border)',
                  }}
                >
                  <div>
                    {prevSec && (
                      <button className="btn btn-sm" onClick={() => setSection(prevSec)}>
                        <ArrowLeft size={13} /> Previous Section
                      </button>
                    )}
                  </div>
                  <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                    Section {curIdx + 1} of {SECTION_ORDER.length} · OmniForge Reference Manual
                  </div>
                  <div>
                    {nextSec && (
                      <button className="btn btn-sm btn-primary" onClick={() => setSection(nextSec)}>
                        Next Section <ArrowRight size={13} />
                      </button>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
