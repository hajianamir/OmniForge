/**
 * Typed contracts for the JSON payloads the workbench API returns to each studio.
 *
 * Every field is optional on purpose: these describe *server responses* (untrusted
 * at the type level) and the studios reuse a payload for both the dry-run preview
 * and the committed result, where only a subset of keys is present. Components
 * therefore stay total without pretending a key always exists.
 *
 * Keep this file in sync with `src/omniforge/api/workbench.py` whenever an endpoint
 * adds/renames a key — a missing key here only degrades a view, but a renamed one
 * silently renders blanks.
 */
import type { ScatterPayload, WorkspaceState } from './types';

/** Any JSON object (rows/tables we only ever hand to `<SimpleTable>`). */
export type JsonRow = Record<string, unknown>;

/** `{ "state": <WorkspaceState> }` — envelope used by most mutating endpoints. */
export interface StateEnvelope {
  state: WorkspaceState;
}

// ---------------------------------------------------------------------------
// Studio 01 — Ingestion, intake audit, cleaning
// ---------------------------------------------------------------------------

export interface TableInfo {
  name: string;
  row_count?: number;
}

export interface SQLInspectResponse {
  connection_url?: string;
  db_path?: string;
  tables?: TableInfo[];
}

export interface ExcelSheetsResponse {
  sheets?: string[];
}

export interface IntakeAuditPayload {
  quality_grade?: string;
  defect_counts?: { total?: number; critical?: number; high?: number; medium?: number; low?: number };
}

export interface IngestResponse extends StateEnvelope {
  elapsed_seconds?: number;
  intake_audit?: IntakeAuditPayload | null;
  format_audit?: JsonRow | null;
}

export interface CleaningReport {
  rows_before?: number;
  rows_after?: number;
  cols_before?: number;
  cols_after?: number;
  missing_before?: number;
  missing_after?: number;
  mode_used?: string;
  [key: string]: unknown;
}

export interface CleanResponse extends StateEnvelope {
  cleaning_report?: CleaningReport;
}

export interface QualityDimensionScores {
  overall_score?: number;
  dimensions?: Record<string, number | null>;
  column_scores?: JsonRow[];
  reference_matched_rows?: number | null;
  reference_comparisons?: number;
  reference_key_columns?: string[];
}

export interface QualityResponse extends StateEnvelope {
  quality?: QualityDimensionScores;
  format_audit?: JsonRow | null;
}

export interface SampleCatalogItem {
  name: string;
  display_name?: string;
  filename?: string;
  size_mb?: number;
  [key: string]: unknown;
}

export interface SampleCatalogResponse {
  benchmarks?: SampleCatalogItem[];
  domains?: SampleCatalogItem[];
}

export interface DeletedLogResponse {
  log?: JsonRow[];
}

// ---------------------------------------------------------------------------
// Studio 02 — Clustering, anomaly, data mining
// ---------------------------------------------------------------------------

export interface ClusterChoice {
  cluster_id: number;
  label: string;
  count: number;
  pct?: number;
  is_noise?: boolean;
}

export interface ClusterAlgorithmScore extends JsonRow {
  algorithm?: string;
}

export interface ClusteringResponse extends StateEnvelope {
  algorithm?: string;
  mode?: string;
  n_clusters?: number;
  noise_count?: number;
  metrics?: Record<string, number>;
  cluster_choices?: ClusterChoice[];
  cluster_profiles?: JsonRow[];
  algorithm_benchmark?: ClusterAlgorithmScore[];
  auto_k_curve?: JsonRow[];
  scatter?: ScatterPayload | null;
  scatter_error?: string | null;
  committed_action?: string;
  provenance?: string | JsonRow;
}

export interface ClusteringCommitResponse extends StateEnvelope {
  action?: string;
  rows_before?: number;
  rows_after?: number;
  rows_removed?: number;
  remaining_cluster_choices?: ClusterChoice[];
  result?: Omit<ClusteringResponse, 'state'> | null;
}

export interface AnomalyResponse extends StateEnvelope {
  method?: string;
  mode?: string;
  n_anomalies?: number;
  anomaly_rate_pct?: number;
  contamination?: number;
  detector_comparison?: JsonRow[];
  top_anomalies?: JsonRow[];
  scatter?: ScatterPayload;
  x_axis?: string;
  y_axis?: string;
  committed_action?: string;
  committed_mode?: string;
  provenance?: string | JsonRow;
}

export interface AnomalyCommitResponse extends StateEnvelope {
  action?: string;
  rows_before?: number;
  rows_after?: number;
  rows_removed?: number;
  result?: Omit<AnomalyResponse, 'state'> | null;
}

export interface ManifoldPayload {
  method?: string;
  scatter?: ScatterPayload;
  total_explained_pct?: number | null;
  explained_variance_pct?: number[];
  x_label?: string;
  y_label?: string;
  error?: string;
  // Older project archives may contain ratios instead of percentages.
  explained_variance_ratio?: number[];
  [key: string]: unknown;
}

export interface MiningResponse extends StateEnvelope {
  target?: string | null;
  tasks_executed?: string[];
  association_rules?: { rules?: JsonRow[] } | null;
  subgroups?: { subgroups?: JsonRow[] } | null;
  manifold?: ManifoldPayload | null;
}

export interface AdvancedMiningResult {
  target?: string | null;
  beam_subgroups?: unknown;
  cluster_stability_ari?: unknown;
  hopkins_clustering_tendency?: unknown;
  distance_correlation?: unknown;
}

export interface AdvancedMiningResponse extends AdvancedMiningResult, StateEnvelope {}

// ---------------------------------------------------------------------------
// Studio 03 — Universal diagnostics & risk
// ---------------------------------------------------------------------------

export interface RiskAssessment {
  structural_risk_index?: number | null;
  risk_band?: string | null;
  tail_risk_table?: JsonRow[];
  [key: string]: unknown;
}

export interface DiagnosticsReport {
  n_scanned?: number;
  sampled_from?: number;
  columns_skipped?: Record<string, string[]>;
  correction_family?: { id: string | null; method: string; n_evaluable: number } | null;
  total_tests?: number;
  significant_tests?: number;
  hypothesis_ledger?: JsonRow[];
  risk_assessment?: RiskAssessment | null;
  module_status?: Record<string, string>;
  baseline_evidence?: Record<string, Record<string, { status: string; role: string; error?: string }>>;
  baseline_equations?: Record<string, unknown[] | Record<string, unknown>>;
  [key: string]: unknown;
}

export interface DiagnosticsResponse extends StateEnvelope {
  report?: DiagnosticsReport;
}

// ---------------------------------------------------------------------------
// Studio 04 — Automated EDA & hypothesis lab
// ---------------------------------------------------------------------------

export interface EDAResponse extends StateEnvelope {
  n_rows_analyzed?: number;
  insights?: (string | JsonRow)[];
  distributions?: { column?: unknown; mean?: unknown; std?: unknown; bins?: { label?: unknown; count?: unknown }[] }[];
  categorical_distributions?: { column?: string; unique_count?: number; top_values?: { label?: string; count?: number }[] }[];
  missing_bars?: { column?: string; missing_pct?: number; [key: string]: unknown }[];
  correlation_matrix?: { columns?: string[]; values?: (number | null)[][] } | null;
  extra_table?: JsonRow[] | null;
  sampled?: boolean;
}

export interface HypothesisSingleResult {
  mode?: string;
  test_name?: string;
  statistic?: number | null;
  p_value?: number | null;
  evaluable?: boolean;
  significant?: boolean;
  alpha?: number;
  interpretation?: string;
  sample_disclosure?: string;
  extra_table?: JsonRow[] | null;
  [key: string]: unknown;
}

export interface HypothesisResponse extends StateEnvelope {
  mode?: string;
  result?: HypothesisSingleResult;
  battery?: JsonRow[];
  skipped?: JsonRow[];
  total_tests?: number;
  significant_tests?: number;
  decision_policy?: string;
}

// ---------------------------------------------------------------------------
// Studio 05 — Feature engineering & selection
// ---------------------------------------------------------------------------

export interface EngineerResponse extends StateEnvelope {
  mode?: string;
  new_features?: string[];
  n_created?: number;
}

export interface FeatureRanking extends JsonRow {
  feature: string;
  score: number;
  selected?: boolean;
}

export interface SelectionResponse extends StateEnvelope {
  selected_source_columns?: string[];
  fit_scope?: 'train_only' | 'full_frame';
  run_scope?: { ignored_columns: string[] } | null;
  run_scope_target?: string | null;
  committed?: boolean;
  mode?: string;
  rankings?: FeatureRanking[];
  selected_features?: string[];
  method_breakdown?: JsonRow[];
}

// ---------------------------------------------------------------------------
// Studio 06 — AutoML, deep learning, predictions
// ---------------------------------------------------------------------------

export interface BackendInfo extends JsonRow {
  id: string;
  name: string;
  description?: string;
  category?: string;
  available?: boolean;
  version?: string;
  notes?: string;
  native_equivalent?: string;
}

export interface BackendsResponse {
  automl_backends?: BackendInfo[];
}

export interface EquationPayload {
  model_name?: string;
  target_name?: string;
  equation_text?: string;
  formula_plain?: string;
  [key: string]: unknown;
}

export interface AutoMLResponse extends StateEnvelope {
  target?: string;
  best_model_name?: string;
  best_score?: number;
  baseline_score?: number | null;
  baseline_name?: string | null;
  lift_over_baseline?: number | null;
  lift_comparison?: { status: string; reason?: string; ci_lower?: number; ci_upper?: number };
  primary_metric?: string;
  backend_used?: string;
  fallback_reason?: string | null;
  elapsed_seconds?: number;
  leaderboard?: JsonRow[];
  equation?: EquationPayload | null;
  metadata?: {
    evaluation?: {
      leaderboard_partition?: string;
      best_score_partition?: string;
      n_test?: number;
    };
  };
}

export interface NonParametricResponse extends StateEnvelope {
  best_model_name?: string;
  best_nonparametric_score?: number | null;
  parametric_reference_score?: number | null;
  parametric_reference_status?: 'available' | 'unavailable';
  comparison_status?: 'available' | 'unavailable';
  lift_vs_parametric?: number | null;
  nonparametric_outperforms_parametric?: boolean | null;
  primary_metric?: string;
  target?: string;
  task_type?: string;
  verdict?: string;
  elapsed_seconds?: number;
  leaderboard?: JsonRow[];
  equations?: EquationPayload[];
  spline_components?: JsonRow[] | null;
}

export interface DeepLearningResponse extends StateEnvelope {
  results?: JsonRow[];
  feature_importances?: Record<string, { feature: string; importance: number }[]>;
  [key: string]: unknown;
}

export interface SinglePredictionResponse {
  model?: string;
  prediction?: number | string | null;
  [key: string]: unknown;
}

export interface BatchPredictionResponse {
  n_rows?: number;
  preview?: JsonRow[];
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Studio 07 — Responsible AI
// ---------------------------------------------------------------------------

export interface ExplainResponse {
  global_importance?: Record<string, number | { importance_mean?: number | null; importance_std?: number | null }>;
  local_explanation?: JsonRow | JsonRow[] | null;
  pdp?: { feature?: string; grid?: number[]; pdp_mean?: number[]; pdp_std?: number[]; [key: string]: unknown } | JsonRow[] | null;
  pdp_feature?: string | null;
  [key: string]: unknown;
}

export interface FairnessReportPayload {
  sensitive_column?: string;
  protected_attribute?: string;
  demographic_parity_diff?: number;
  demographic_parity_difference?: number;
  disparate_impact_ratio?: number;
  equal_opportunity_diff?: number;
  equal_opportunity_difference?: number;
  group_metrics?: Record<string, JsonRow>;
  assessment_status?: 'insufficient_evidence' | 'threshold_failed' | 'thresholds_met' | null;
  cluster_attribute?: string | null;
  n_clusters?: number | null;
  demographic_parity_ci_95?: [number, number] | null;
  violations?: string[];
  groups?: JsonRow[];
  [key: string]: unknown;
}

export interface FairnessResponse {
  fairness?: FairnessReportPayload;
  model_card_markdown?: string;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Studio 08 — Forecast, causal, optimization, simulation
// ---------------------------------------------------------------------------

export interface ForecastPoint {
  step: number | null;
  yhat: number | null;
  yhat_lower?: number | null;
  yhat_upper?: number | null;
  [key: string]: unknown;
}

export interface ForecastHistoryPoint {
  step: number | null;
  actual: number | null;
  [key: string]: unknown;
}

export interface ForecastResponse extends StateEnvelope {
  history?: ForecastHistoryPoint[];
  deep_arch?: string;
  deep_forecast?: ForecastPoint[];
  classical_summary?: {
    champion_model?: string;
    forecast?: ForecastPoint[];
    metadata?: {
      champion_interval_calibration?: {
        status: string;
        nominal_level: number;
        evaluation?: {
          observed_outcomes: number;
          calibrated_coverage: number | null;
          scored_origins?: number;
          requested_origins?: number;
          native_coverage_count?: number;
          native_coverage?: number | null;
          paired_calibrated_coverage?: number | null;
          by_lead?: Array<{
            lead: number;
            observed_outcomes: number;
            calibrated_coverage: number | null;
            native_coverage: number | null;
            native_coverage_count: number;
            paired_calibrated_coverage: number | null;
          }>;
        } | null;
      };
    };
  } | null;
  [key: string]: unknown;
}

export interface CausalPayload {
  [key: string]: unknown;
}

export interface OptimizationPayload {
  [key: string]: unknown;
}

export interface SimulationPayload {
  [key: string]: unknown;
}

export interface CausalResponse extends StateEnvelope {
  causal?: CausalPayload;
  [key: string]: unknown;
}

export interface OptimizeResponse extends StateEnvelope {
  optimization?: OptimizationPayload;
  [key: string]: unknown;
}

export interface SimulateResponse extends StateEnvelope {
  simulation?: SimulationPayload;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Studio 09 — Specialized AI (NLP / Vision / Graph / Recommend / Survival)
// ---------------------------------------------------------------------------

export interface NLPResponse extends StateEnvelope {
  [key: string]: unknown;
}

export interface BackendsProbeResponse {
  nlp?: BackendInfo[];
  text?: BackendInfo[];
  vision?: { device?: string; backends?: BackendInfo[]; [key: string]: unknown };
}

export interface VisionResponse extends StateEnvelope {
  detections?: { n_detections?: number; detections?: JsonRow[]; [key: string]: unknown } | null;
  segmentation?: { n_segments?: number; regions?: JsonRow[]; [key: string]: unknown } | null;
  zero_shot?: {
    backend?: string;
    score_kind?: string;
    top_label?: string | null;
    note?: string;
    predictions?: { label: string; score?: number; probability?: number }[];
  } | null;
  embedding?: { dim?: number; image_shape?: unknown } | null;
  [key: string]: unknown;
}

export interface GraphStructure {
  n_nodes?: number;
  n_edges?: number;
  n_bridges?: number;
  max_k_core?: number;
  [key: string]: unknown;
}

export interface LinkPredictionPayload {
  metrics?: { roc_auc?: number; [key: string]: unknown };
  top_predictions?: JsonRow[];
  [key: string]: unknown;
}

export interface GraphVisualization {
  nodes?: { id: string; x: number; y: number; degree: number; community: number; [key: string]: unknown }[];
  edges?: { source: string; target: string; [key: string]: unknown }[];
  [key: string]: unknown;
}

export interface GraphResponse extends StateEnvelope {
  structure?: GraphStructure | null;
  visualization?: GraphVisualization | null;
  topological_links?: JsonRow[];
  supervised_link_prediction?: LinkPredictionPayload | null;
  neural_gae_link_prediction?: LinkPredictionPayload | null;
  centrality?: JsonRow | null;
  community?: JsonRow | null;
  [key: string]: unknown;
}

export interface RecommendResponse extends StateEnvelope {
  recommendations?: JsonRow[];
  [key: string]: unknown;
}

export interface SurvivalResponse extends StateEnvelope {
  cox?: JsonRow | null;
  km_curve?: JsonRow[];
  median_survival?: number | null;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Studio 10 — DAG pipeline builder
// ---------------------------------------------------------------------------

export interface DAGResponse extends StateEnvelope {
  execution?: JsonRow | null;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Studio 11 — MLOps registry & model monitoring
// ---------------------------------------------------------------------------

export interface RegisteredModel extends JsonRow {
  name?: string;
  stage?: string;
}

export interface DriftReport {
  [key: string]: unknown;
}

export interface MLOpsOverviewResponse {
  registered_models?: RegisteredModel[];
  experiment_runs?: JsonRow[];
}

export interface RegisterResponse extends StateEnvelope, MLOpsOverviewResponse {}

export interface DriftResponse extends StateEnvelope {
  drift_report?: DriftReport;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Studio 12 — Executive reporting center
// ---------------------------------------------------------------------------

export interface ReportSection {
  key?: string;
  section_id?: string;
  title?: string;
  category?: string;
  description?: string;
  default_on?: boolean;
  has_data?: boolean;
  [key: string]: unknown;
}

export interface ReportCatalogResponse {
  sections?: ReportSection[];
  presets?: Record<string, string[]>;
}

export interface ReportSectionOut {
  id?: string;
  key?: string;
  title?: string;
  narrative?: string;
  highlights?: string[];
  category?: string;
  default_on?: boolean;
  [key: string]: unknown;
}

export interface ReportChartPreview {
  title?: string;
  caption?: string;
  section?: string;
  image_data_url?: string;
  [key: string]: unknown;
}

export interface CompiledReportResponse extends StateEnvelope {
  title?: string;
  executive_summary?: string;
  n_charts?: number;
  sections?: ReportSectionOut[];
  table_names?: string[];
  table_previews?: Record<string, { rows?: number; columns?: string[]; preview?: JsonRow[] }>;
  chart_previews?: ReportChartPreview[];
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Studio 13 — AI analytics copilot
// ---------------------------------------------------------------------------

export interface ChatTurn {
  role: 'user' | 'assistant';
  text: string;
}

export interface ChatResponse {
  history?: ChatTurn[];
  reply?: string;
  answer?: string;
  provider_used?: string;
  suggested_actions?: string[];
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Studio 14 — Supervisory monitor
// ---------------------------------------------------------------------------

export interface SupervisoryModelInfo extends JsonRow {
  task_type?: string | null;
  input_columns?: string[];
  input_types?: Record<string, string>;
  model_id: string;
  label: string;
  family?: string;
  target?: string;
  n_features?: number;
  feature_cols?: string[];
  metrics?: Record<string, number>;
}

export interface SupervisoryKPIs extends JsonRow {
  mae?: number;
  rmse?: number;
  baseline_mae?: number;
  mape_pct?: number;
  breach_count?: number;
  warning_count?: number;
  unavailable_count?: number;
  compliance_rate_pct?: number | null;
  model_vs_baseline_lift_pct?: number;
  mae_improvement_vs_baseline_pct?: number;
}

export interface TrendPoint {
  label?: string;
  threshold?: number;
  actual?: number | null;
  predicted?: number | null;
  baseline?: number | null;
  [key: string]: unknown;
}

export interface SPCPoint {
  label?: string;
  residual?: number | null;
  ucl?: number | null;
  lcl?: number | null;
  status?: string;
  [key: string]: unknown;
}

export interface SPCXYPoint {
  index: number;
  record_id?: string;
  actual?: number | null;
  model_pred: number;
  baseline_pred?: number | null;
  tol_upper?: number | null;
  tol_lower?: number | null;
  severity?: string;
  [key: string]: unknown;
}

export interface SPCPointXY {
  index: number;
  record_id?: string;
  residual: number;
  z_score?: number;
  severity?: string;
  [key: string]: unknown;
}

export interface SPCLimits {
  center?: number;
  warn_upper?: number;
  warn_lower?: number;
  // Aliases accepted for older persisted projects.
  center_line?: number;
  ucl?: number;
  lcl?: number;
  uwl?: number;
  lwl?: number;
  [key: string]: unknown;
}

export interface ErrorHistogramBin {
  bin?: string | number;
  label?: string;
  count?: number;
  breach?: boolean;
  [key: string]: unknown;
}

export interface FactorBreakdownItem {
  feature: string;
  signed_impact: number;
  z_shift?: number;
  live_value?: number;
  normal_ref?: number;
  [key: string]: unknown;
}

export interface SupervisoryAlarm {
  row_id: number;
  severity?: string;
  actual?: number;
  predicted?: number;
  baseline_predicted?: number;
  delta_vs_baseline?: number;
  deviation_direction?: string;
  pct_error?: number;
  top_increase_drivers?: string[];
  top_decrease_drivers?: string[];
  reason?: string;
  factor_breakdown?: FactorBreakdownItem[];
  recommended_fix?: string;
  [key: string]: unknown;
}

export interface SupervisoryEvaluation {
  timestamp_utc?: string;
  champion_equation?: EquationPayload | null;
  baseline_equation?: EquationPayload | null;
  root_cause_analysis?: RootCauseReport | null;
  tolerances?: JsonRow;
  system_status?: string;
  source?: string;
  kpis?: SupervisoryKPIs;
  system_alerts?: string[];
  alarms?: SupervisoryAlarm[];
  error_histogram?: ErrorHistogramBin[];
  trend_series?: TrendPoint[];
  spc_points?: SPCPoint[];
  spc_limits?: SPCLimits | null;
  spc_unavailable_reason?: string | null;
  ratio_points?: JsonRow[];
  telemetry_rows?: JsonRow[];
  history?: JsonRow[];
}

export interface SupervisoryModelsResponse {
  models?: SupervisoryModelInfo[];
}

export interface SupervisoryTrainResponse extends SupervisoryModelsResponse, StateEnvelope {
  trained_model?: string;
  metric?: string;
  score?: number;
}

export interface SupervisoryStatusResponse {
  models?: SupervisoryModelInfo[];
  last_evaluation?: SupervisoryEvaluation | null;
  history?: JsonRow[];
  champion_equation?: EquationPayload | null;
  baseline_equation?: EquationPayload | null;
}

export interface SupervisoryEvaluateResponse extends SupervisoryEvaluation, StateEnvelope {
  history?: JsonRow[];
}

export interface RootCauseDriver extends JsonRow {
  feature: string;
  signed_impact: number;
  direction?: string;
  z_shift?: number;
  live_mean?: number;
  normal_reference?: number;
  out_of_band?: boolean;
}

export interface RemediationPlaybookItem extends JsonRow {
  priority: string;
  variable?: string;
  impact_on_target?: string;
  live_vs_normal?: string;
}

export interface RootCauseReport {
  overall_direction?: string;
  attribution_method?: string;
  attribution_scope?: string;
  summary_narrative?: string;
  mean_actual_vs_baseline_delta?: number;
  mean_actual_vs_baseline_pct?: number;
  mean_champion_vs_baseline_delta?: number;
  mean_unexplained_residual?: number;
  increase_drivers?: RootCauseDriver[];
  decrease_drivers?: RootCauseDriver[];
  all_factor_attributions?: RootCauseDriver[];
  remediation_playbook?: RemediationPlaybookItem[];
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Aggregated studio cache (App.tsx keeps one payload per studio result key)
// ---------------------------------------------------------------------------

export interface StudioCache {
  s01_quality?: QualityResponse | null;
  s01_clean?: CleanResponse | null;
  s02_clustering?: ClusteringResponse | null;
  s02_anomaly?: AnomalyResponse | null;
  s02_mining?: MiningResponse | null;
  s02_advanced_mining?: AdvancedMiningResult | null;
  s03_diagnostics?: DiagnosticsResponse | null;
  s04_eda?: EDAResponse | null;
  s04_hypothesis?: HypothesisResponse | null;
  s05_engineer?: EngineerResponse | null;
  s05_select?: SelectionResponse | null;
  s06_automl?: AutoMLResponse | null;
  s06_nonparametric?: NonParametricResponse | null;
  s06_dl?: DeepLearningResponse | null;
  s06_predict_single?: SinglePredictionResponse | null;
  s06_predict_batch?: BatchPredictionResponse | null;
  s07_explain?: ExplainResponse | null;
  s07_fairness?: FairnessResponse | null;
  s08_forecast?: ForecastResponse | null;
  s08_causal?: CausalResponse | null;
  s08_optimize?: OptimizeResponse | null;
  s08_simulate?: SimulateResponse | null;
  s09_nlp?: NLPResponse | null;
  s09_vision?: VisionResponse | null;
  s09_graph?: GraphResponse | null;
  s09_recommend?: RecommendResponse | null;
  s09_survival?: SurvivalResponse | null;
  s10_dag?: DAGResponse | null;
  s11_drift?: DriftResponse | null;
  s11_registered?: RegisterResponse | null;
  s12_report?: CompiledReportResponse | null;
  s13_chat?: ChatResponse | null;
  s14_supervisory?: SupervisoryEvaluation | null;
}

/** Union of every studio result key the app caches. */
export type StudioResultKey = keyof StudioCache;

/** Union of every cached studio payload. */
export type StudioPayload = StudioCache[StudioResultKey];
