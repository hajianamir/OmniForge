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

const FORBIDDEN_POPOUT_TAGS = new Set([
  'script',
  'iframe',
  'object',
  'embed',
  'applet',
  'base',
  'meta',
  'link',
  'foreignobject',
  'form',
  'input',
  'textarea',
]);

const SAFE_IMAGE_DATA_URI_RE = /^data:image\/(?:png|jpeg|jpg|webp|gif);base64,[A-Za-z0-9+/=\s]+$/i;
const SAFE_HTTP_OR_RELATIVE_RE = /^(?:https?:\/\/|\/|\.\/|#)/i;

function isSafeAttributeUrl(value: string): boolean {
  const trimmed = value.trim();
  if (!trimmed) return true;
  if (SAFE_IMAGE_DATA_URI_RE.test(trimmed)) return true;
  if (SAFE_HTTP_OR_RELATIVE_RE.test(trimmed)) return true;
  return false;
}

/**
 * Recursively sanitize a cloned DOM subtree before inserting it into a pop-out window.
 * Removes executable elements, inline event handlers (on*), and unsafe URI schemes.
 */
function sanitizeClonedSubtree(root: HTMLElement): void {
  const toRemove: Element[] = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
  let node = walker.nextNode();
  while (node) {
    const el = node as Element;
    const tag = el.tagName.toLowerCase();
    if (FORBIDDEN_POPOUT_TAGS.has(tag)) {
      toRemove.push(el);
    } else {
      const attrNames = Array.from(el.attributes).map((a) => a.name);
      for (const attrName of attrNames) {
        const lower = attrName.toLowerCase();
        const val = el.getAttribute(attrName) || '';
        if (lower.startsWith('on') || lower === 'srcdoc' || lower === 'formaction') {
          el.removeAttribute(attrName);
        } else if (
          lower === 'href' ||
          lower === 'xlink:href' ||
          lower === 'src' ||
          lower === 'action'
        ) {
          if (!isSafeAttributeUrl(val)) {
            el.removeAttribute(attrName);
          }
        }
      }
    }
    node = walker.nextNode();
  }
  for (const el of toRemove) {
    el.remove();
  }
}

/**
 * Universal inspection wrapper that equips EVERY chart, graph, image, equation card,
 * and data table across OmniForge with:
 *   1. Full-viewport interactive Zoom Modal (60% - 220% magnification + Escape to close)
 *   2. Dedicated Pop-Out Browser Window built strictly via sanitized DOM APIs and a
 *      script-free Content-Security-Policy (zero document.write / innerHTML sinks).
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

    // Clone node and convert any live <canvas> elements into crisp PNG data-URL <img> elements
    const clone = srcEl.cloneNode(true) as HTMLElement;
    const origCanvases = srcEl.querySelectorAll('canvas');
    const cloneCanvases = clone.querySelectorAll('canvas');
    origCanvases.forEach((origC, idx) => {
      const targetC = cloneCanvases[idx];
      if (!targetC) return;
      try {
        const img = document.createElement('img');
        img.src = origC.toDataURL('image/png');
        img.alt = title;
        img.style.width = '100%';
        img.style.height = 'auto';
        img.style.borderRadius = '10px';
        img.style.display = 'block';
        targetC.replaceWith(img);
      } catch {
        // Keep canvas fallback if tainted
      }
    });
    sanitizeClonedSubtree(clone);

    const rawTheme = document.documentElement.getAttribute('data-theme');
    const theme = rawTheme === 'light' ? 'light' : 'dark';
    const isLight = theme === 'light';
    const bg = isLight ? '#f4f7fb' : '#070b14';
    const card = isLight ? '#ffffff' : '#111b2e';
    const subtle = isLight ? '#eef3fa' : '#17233b';
    const text = isLight ? '#0f172a' : '#edf2ff';
    const muted = isLight ? '#475569' : '#94a3b8';
    const border = isLight ? '#d7e1ef' : '#22314f';

    const popup = window.open('', '_blank', 'width=1280,height=860,resizable=yes,scrollbars=yes');
    if (!popup) {
      // Fallback to in-app fullscreen modal if browser blocks popups
      setOpen(true);
      return;
    }

    const pDoc = popup.document;
    pDoc.documentElement.setAttribute('data-theme', theme);
    pDoc.head.replaceChildren();
    pDoc.body.replaceChildren();

    const metaCharset = pDoc.createElement('meta');
    metaCharset.setAttribute('charset', 'utf-8');
    pDoc.head.appendChild(metaCharset);

    // Enforce strict script-free CSP inside the pop-out window
    const metaCsp = pDoc.createElement('meta');
    metaCsp.setAttribute('http-equiv', 'Content-Security-Policy');
    metaCsp.setAttribute(
      'content',
      "default-src 'none'; img-src data: blob:; style-src 'unsafe-inline'; font-src data:;"
    );
    pDoc.head.appendChild(metaCsp);

    const titleEl = pDoc.createElement('title');
    titleEl.textContent = `OmniForge Inspector — ${title}`;
    pDoc.head.appendChild(titleEl);

    const styleEl = pDoc.createElement('style');
    styleEl.textContent = `
      :root { --bg: ${bg}; --bg-card: ${card}; --bg-subtle: ${subtle}; --text: ${text}; --text-muted: ${muted}; --text-dim: ${muted}; --border: ${border}; --accent: #38bdf8; --green: #34d399; --amber: #fbbf24; --red: #f87171; }
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
    `;
    pDoc.head.appendChild(styleEl);

    const bar = pDoc.createElement('div');
    bar.className = 'bar';

    const metaBox = pDoc.createElement('div');
    const heading = pDoc.createElement('h2');
    heading.textContent = `Ω OmniForge · ${title}`;
    metaBox.appendChild(heading);
    if (subtitle) {
      const subEl = pDoc.createElement('small');
      subEl.textContent = subtitle;
      metaBox.appendChild(subEl);
    }

    const ctrl = pDoc.createElement('div');
    ctrl.className = 'ctrl';

    const btnOut = pDoc.createElement('button');
    btnOut.type = 'button';
    btnOut.textContent = '− Zoom';

    const zLabel = pDoc.createElement('span');
    zLabel.className = 'mono';
    zLabel.textContent = '100%';

    const btnIn = pDoc.createElement('button');
    btnIn.type = 'button';
    btnIn.textContent = '+ Zoom';

    const btnReset = pDoc.createElement('button');
    btnReset.type = 'button';
    btnReset.textContent = 'Reset';

    const btnPrint = pDoc.createElement('button');
    btnPrint.type = 'button';
    btnPrint.textContent = 'Print / Save PDF';

    ctrl.append(btnOut, zLabel, btnIn, btnReset, btnPrint);
    bar.append(metaBox, ctrl);

    const stage = pDoc.createElement('div');
    stage.className = 'stage';
    const zoomWrap = pDoc.createElement('div');
    zoomWrap.className = 'zoom-wrap';
    const importedClone = pDoc.importNode(clone, true);
    zoomWrap.appendChild(importedClone);
    stage.appendChild(zoomWrap);

    pDoc.body.append(bar, stage);

    let popZoom = 100;
    const applyZoom = (nextVal: number) => {
      popZoom = Math.max(50, Math.min(250, nextVal));
      zLabel.textContent = `${popZoom}%`;
      zoomWrap.style.transform = `scale(${popZoom / 100})`;
    };
    btnOut.addEventListener('click', () => applyZoom(popZoom - 15));
    btnIn.addEventListener('click', () => applyZoom(popZoom + 15));
    btnReset.addEventListener('click', () => applyZoom(100));
    btnPrint.addEventListener('click', () => popup.print());
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
