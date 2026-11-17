import React, { useEffect, useRef, useState } from 'react';
import { ExternalLink, Maximize2, Minimize2, ZoomIn, ZoomOut, RotateCcw, X } from 'lucide-react';

interface InspectableBoxProps {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  compact?: boolean;
}

/**
 * Universal inspection wrapper that equips EVERY chart, graph, image, equation card,
 * and data table across OmniForge with:
 *   1. Full-viewport interactive Zoom Modal (50% - 225% magnification + Escape to close)
 *   2. Dedicated Pop-Out Browser Window (standalone self-contained offline inspection view)
 */
export const InspectableBox: React.FC<InspectableBoxProps> = ({
  title = 'Visual / Table Inspector',
  subtitle,
  children,
  className = '',
  style,
  compact = false,
}) => {
  const [open, setOpen] = useState(false);
  const [zoom, setZoom] = useState(100);
  const contentRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  const handlePopOut = () => {
    const srcEl = contentRef.current;
    if (!srcEl) return;

    // Clone node and convert any live <canvas> elements into crisp data-URL <img> elements
    const clone = srcEl.cloneNode(true) as HTMLElement;
    const origCanvases = srcEl.querySelectorAll('canvas');
    const cloneCanvases = clone.querySelectorAll('canvas');
    origCanvases.forEach((origC, idx) => {
      const targetC = cloneCanvases[idx];
      if (!targetC) return;
      try {
        const img = document.createElement('img');
        img.src = origC.toDataURL('image/png');
        img.style.width = '100%';
        img.style.height = 'auto';
        img.style.borderRadius = '10px';
        img.style.display = 'block';
        targetC.replaceWith(img);
      } catch {
        // Keep canvas fallback if tainted
      }
    });

    const theme = document.documentElement.getAttribute('data-theme') || 'dark';
    const isLight = theme === 'light';
    const bg = isLight ? '#f4f7fb' : '#070b14';
    const card = isLight ? '#ffffff' : '#111b2e';
    const text = isLight ? '#0f172a' : '#edf2ff';
    const muted = isLight ? '#475569' : '#94a3b8';
    const border = isLight ? '#d7e1ef' : '#22314f';

    const popup = window.open('', '_blank', 'width=1280,height=860,resizable=yes,scrollbars=yes');
    if (!popup) {
      // Fallback to in-app fullscreen modal if browser blocks popups
      setOpen(true);
      return;
    }

    const safeTitle = title.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const safeSub = (subtitle || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    popup.document.open();
    popup.document.write(`<!doctype html>
<html data-theme="${theme}">
<head>
<meta charset="utf-8" />
<title>OmniForge Inspector — ${safeTitle}</title>
<style>
  :root { --bg: ${bg}; --bg-card: ${card}; --bg-subtle: ${isLight ? '#eef3fa' : '#17233b'}; --text: ${text}; --text-muted: ${muted}; --text-dim: ${muted}; --border: ${border}; --accent: #38bdf8; --green: #34d399; --amber: #fbbf24; --red: #f87171; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif; font-size: 13.5px; }
  .bar { position: sticky; top: 0; z-index: 10; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 20px; background: var(--bg-card); border-bottom: 1px solid var(--border); }
  .bar h2 { margin: 0; font-size: 15px; }
  .bar small { color: var(--text-muted); font-size: 12px; }
  .ctrl { display: flex; align-items: center; gap: 8px; }
  .ctrl button { padding: 6px 12px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg-subtle); color: var(--text); cursor: pointer; font-weight: 600; }
  .ctrl button:hover { border-color: var(--accent); }
  .stage { padding: 20px; overflow: auto; }
  .zoom-wrap { transform-origin: top left; transition: transform 0.12s ease; background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
  .table-wrap { overflow: auto; max-height: none !important; border: 1px solid var(--border); border-radius: 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }
  th { background: var(--bg-subtle); font-weight: 700; position: sticky; top: 0; }
  .badge { display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 99px; font-size: 11px; font-weight: 600; border: 1px solid var(--border); }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
  svg { max-width: 100%; height: auto !important; }
  img { max-width: 100%; height: auto; }
</style>
</head>
<body>
  <div class="bar">
    <div>
      <h2>Ω OmniForge · ${safeTitle}</h2>
      ${safeSub ? `<small>${safeSub}</small>` : ''}
    </div>
    <div class="ctrl">
      <button onclick="setZ(z - 15)">− Zoom</button>
      <span id="zlabel" class="mono">100%</span>
      <button onclick="setZ(z + 15)">+ Zoom</button>
      <button onclick="setZ(100)">Reset</button>
      <button onclick="window.print()">Print / Save PDF</button>
    </div>
  </div>
  <div class="stage">
    <div id="zw" class="zoom-wrap">${clone.innerHTML}</div>
  </div>
  <script>
    var z = 100;
    function setZ(val) {
      z = Math.max(50, Math.min(250, val));
      document.getElementById('zlabel').textContent = z + '%';
      document.getElementById('zw').style.transform = 'scale(' + (z / 100) + ')';
    }
  </script>
</body>
</html>`);
    popup.document.close();
  };

  return (
    <div className={`inspectable-box ${className}`} style={style}>
      <div className="inspect-toolbar" style={compact ? { marginBottom: 4 } : undefined}>
        <div className="inspect-meta">
          {title && <span className="inspect-title">{title}</span>}
          {subtitle && <span className="inspect-sub">{subtitle}</span>}
        </div>
        <div className="inspect-actions">
          <button
            type="button"
            className="inspect-btn"
            onClick={() => { setZoom(100); setOpen(true); }}
            title="Zoom / Fullscreen View"
            aria-label={`Zoom ${title}`}
          >
            <Maximize2 size={12} /> Zoom
          </button>
          <button
            type="button"
            className="inspect-btn"
            onClick={handlePopOut}
            title="Open in Separate Pop-Out Window"
            aria-label={`Pop out ${title}`}
          >
            <ExternalLink size={12} /> Pop-Out
          </button>
        </div>
      </div>

      <div ref={contentRef} className="inspect-body">
        {children}
      </div>

      {open && (
        <div className="inspect-modal-backdrop" onClick={() => setOpen(false)}>
          <div className="inspect-modal-panel" onClick={(e) => e.stopPropagation()}>
            <div className="inspect-modal-header">
              <div>
                <div style={{ fontWeight: 700, fontSize: 15 }}>{title}</div>
                {subtitle && <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{subtitle}</div>}
              </div>
              <div className="row" style={{ gap: 6 }}>
                <button className="btn btn-sm" onClick={() => setZoom((z) => Math.max(60, z - 20))} title="Zoom Out"><ZoomOut size={13} /></button>
                <span className="badge mono">{zoom}%</span>
                <button className="btn btn-sm" onClick={() => setZoom((z) => Math.min(220, z + 20))} title="Zoom In"><ZoomIn size={13} /></button>
                <button className="btn btn-sm" onClick={() => setZoom(100)} title="Reset Zoom"><RotateCcw size={13} /></button>
                <button className="btn btn-sm" onClick={handlePopOut} title="Open in Separate Window"><ExternalLink size={13} /> Pop-Out</button>
                <button className="btn btn-sm btn-primary" onClick={() => setOpen(false)} title="Close"><Minimize2 size={13} /> <X size={13} /></button>
              </div>
            </div>
            <div className="inspect-modal-body">
              <div style={{ transform: `scale(${zoom / 100})`, transformOrigin: 'top left', width: `${10000 / zoom}%` }}>
                {children}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
