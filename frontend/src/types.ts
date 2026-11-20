export type StudioId =
  | '00'
  | '01'
  | '02'
  | '03'
  | '04'
  | '05'
  | '06'
  | '07'
  | '08'
  | '09'
  | '10'
  | '11'
  | '12'
  | '13'
  | '14';

export interface StudioDef {
  id: StudioId;
  title: string;
  short: string;
}

export const STUDIOS: StudioDef[] = [
  { id: '00', title: 'Data Ingestion & Split Lock', short: 'Ingestion' },
  { id: '01', title: 'Data Preparation & Cleaning', short: 'Cleaning' },
  { id: '02', title: 'Clustering, Anomaly & Data Mining', short: 'Cluster · Mining' },
  { id: '03', title: 'Universal Diagnostics, Baselines & Risk', short: 'Diagnostics & Risk' },
  { id: '04', title: 'Automated EDA & Hypothesis Lab', short: 'EDA & Hypothesis' },
  { id: '05', title: 'Feature Engineering & Selection', short: 'Feature Studio' },
  { id: '06', title: 'AutoML, Deep Learning & Predictions', short: 'AutoML & Deep AI' },
  { id: '07', title: 'Responsible AI', short: 'Responsible AI' },
  { id: '08', title: 'Forecasting, Causal, Opt & Simulation', short: 'Decision & Forecast' },
  { id: '09', title: 'Specialized AI', short: 'NLP · Vision · Graph' },
  { id: '10', title: 'Visual DAG Pipeline Builder', short: 'DAG Pipelines' },
  { id: '11', title: 'MLOps', short: 'MLOps Registry' },
  { id: '12', title: 'Executive Reporting & Export Center', short: 'Executive Reports' },
  { id: '13', title: 'AI Analytics Copilot', short: 'AI Copilot' },
  { id: '14', title: 'Live Prediction & Supervisory Monitor', short: 'Supervisory Monitor' },
];

export interface SplitLockSummary {
  active: boolean;
  strategy: string;
  n_train: number;
  n_test: number;
  cv_folds: number;
  test_size: number;
  passed: boolean;
}

export interface IntakeDefect {
  severity: 'critical' | 'high' | 'medium' | 'low' | string;
  category: string;
  column: string;
  finding: string;
  metric: string;
  recommended_fix: string;
}

export interface IntakeAuditSummary {
  overall_health: string;
  quality_score: number;
  quality_grade: string;
  dimensions?: Record<string, number>;
  n_rows: number;
  n_cols: number;
  duplicate_rows: number;
  constant_columns: string[];
  high_missing_columns: string[];
  format_health?: string;
  defect_counts: { critical: number; high: number; medium: number; low: number; total: number };
  defects: IntakeDefect[];
  statistical_pretests: Record<string, unknown>[];
  outlier_summary: Record<string, unknown>[];
  correlation_alerts: Record<string, unknown>[];
  auto_clean_preset: Record<string, unknown>;
  summary_bullets: string[];
}

export interface WorkspaceState {
  session_id: string;
  platform: { name: string; version: string; edition?: string };
  has_dataset: boolean;
  dataset_name: string | null;
  dataset_source: string | null;
  n_rows: number;
  n_cols: number;
  raw_rows: number;
  raw_cols: number;
  columns: string[];
  numeric_columns: string[];
  categorical_columns: string[];
  targets: string[];
  primary_target: string | null;
  excluded_columns: string[];
  analysis_cols_count: number;
  can_undo: boolean;
  processing_status: {
    is_processed?: boolean;
    is_cleaned?: boolean;
    cluster_committed?: boolean;
    anomaly_committed?: boolean;
    badge?: string;
    label?: string;
    detailed_label?: string;
    summary?: string;
  };
  cached_result_keys?: string[];
  split_lock: SplitLockSummary | null;
  total_deleted_rows: number;
  format_audit_issues: number;
  intake_audit?: IntakeAuditSummary | null;
  s02_cluster_committed?: boolean;
  s02_cluster_summary?: Record<string, unknown> | null;
  s02_anomaly_committed?: boolean;
  s02_anomaly_summary?: Record<string, unknown> | null;
  active_model_name: string | null;
  active_model_metrics: Record<string, number> | null;
  active_feature_cols: string[];
  available_models_count?: number;
  supervisory_status?: 'IDLE' | 'NORMAL' | 'WARNING' | 'CRITICAL_ALARM' | string;
  studio_completed?: Record<string, boolean>;
  workflow_checklist: { label: string; done: boolean }[];
  lineage: { timestamp?: string; stage: string; action: string; details?: string }[];
  theme: 'dark' | 'light';
  project_name: string;
}

export interface GridColumnMeta {
  name: string;
  dtype: string;
  excluded: boolean;
  is_target: boolean;
}

export interface GridResponse {
  columns: GridColumnMeta[];
  rows: Record<string, unknown>[];
  total_rows: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ScatterPayload {
  x: number[];
  y: number[];
  groups?: (string | number)[];
  total_points: number;
  rendered_points: number;
  downsampled: boolean;
}
