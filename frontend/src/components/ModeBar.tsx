import React from 'react';

export type ExecutionMode = 'auto' | 'complete' | 'custom';

export const ModeBar: React.FC<{
  mode: ExecutionMode;
  onChange: (m: ExecutionMode) => void;
  labels?: { auto?: string; complete?: string; custom?: string };
}> = ({ mode, onChange, labels }) => (
  <div className="mode-bar" role="group" aria-label="Execution Mode">
    <button
      type="button"
      className={`mode-pill ${mode === 'auto' ? 'active' : ''}`}
      onClick={() => onChange('auto')}
    >
      {labels?.auto || 'Auto (Recommended)'}
    </button>
    <button
      type="button"
      className={`mode-pill ${mode === 'complete' ? 'active' : ''}`}
      onClick={() => onChange('complete')}
    >
      {labels?.complete || 'Complete (Select All)'}
    </button>
    <button
      type="button"
      className={`mode-pill ${mode === 'custom' ? 'active' : ''}`}
      onClick={() => onChange('custom')}
    >
      {labels?.custom || 'Custom Select'}
    </button>
  </div>
);
