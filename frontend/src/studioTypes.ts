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
  dimensions?: Record<string, number>;
  column_scores?: JsonRow[];
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
  scatter?: ScatterPayload;
  committed_action?: string;
  provenance?: JsonRow;
}

export interface ClusteringCommitResponse extends StateEnvelope {
  action?: string;
  rows_before?: number;
  rows_after?: number;
  rows_removed?: number;
  remaining_cluster_choices?: ClusterChoice[];
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
  provenance?: JsonRow;
}

export interface AnomalyCommitResponse extends StateEnvelope {
  action?: string;
  rows_before?: number;
  rows_after?: number;
  rows_removed?: number;
}

export interface ManifoldPayload {
  method?: string;
  scatter?: ScatterPayload;
  explained_variance_ratio?: number[];
  [key: string]: unknown;
}

export interface MiningResponse extends StateEnvelope {
  tasks_executed?: string[];
  association_rules?: { rules?: JsonRow[] } | null;
  subgroups?: { subgroups?: JsonRow[] } | null;
  manifold?: ManifoldPayload | null;
}

// ---------------------------------------------------------------------------
// Studio 03 — Universal diagnostics & risk
// ---------------------------------------------------------------------------

export interface RiskAssessment {
  overall_risk_index?: number;
  overall_risk_band?: string;
  column_risks?: JsonRow[];
  [key: string]: unknown;
}

export interface DiagnosticsReport {
  total_tests?: number;
  significant_tests?: number;
  hypothesis_ledger?: JsonRow[];
  risk_assessment?: RiskAssessment;
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
  insights?: string[];
  distributions?: { column?: unknown; mean?: unknown; std?: unknown; bins?: { label?: unknown; count?: unknown }[] }[];
  missing_bars?: JsonRow[];
  correlation_matrix?: { columns?: string[]; values?: number[][] } | null;
  extra_table?: JsonRow[] | null;
  sampled?: boolean;
}

export interface HypothesisSingleResult {
  mode?: string;
  test_name?: string;
  statistic?: number;
  p_value?: number;
  significant?: boolean;
  alpha?: number;
  interpretation?: string;
  extra_table?: JsonRow[] | null;
  [key: string]: unknown;
}

export interface HypothesisResponse extends StateEnvelope {
  mode?: string;
  result?: HypothesisSingleResult;
  battery?: JsonRow[];
  total_tests?: number;
  significant_tests?: number;
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
  best_model_name?: string;
  best_score?: number;
  baseline_score?: number;
  lift_over_baseline?: number;
  primary_metric?: string;
  backend_used?: string;
  fallback_reason?: string | null;
  elapsed_seconds?: number;
  leaderboard?: JsonRow[];
  equation?: EquationPayload | null;
}

export interface NonParametricResponse extends StateEnvelope {
  best_model_name?: string;
  best_nonparametric_score?: number;
  parametric_reference_score?: number;
  lift_vs_parametric?: number;
  nonparametric_outperforms_parametric?: boolean;
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
  global_importance?: Record<string, number>;
  local_explanation?: JsonRow | JsonRow[] | null;
  pdp?: JsonRow[] | null;
  pdp_feature?: string | null;
  [key: string]: unknown;
}

export interface FairnessReportPayload {
  sensitive_column?: string;
  demographic_parity_difference?: number;
  disparate_impact_ratio?: number;
  equal_opportunity_difference?: number;
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
  step: number;
  yhat: number;
  yhat_lower?: number;
  yhat_upper?: number;
  [key: string]: unknown;
}

export interface ForecastHistoryPoint {
  step: number;
  actual: number;
  [key: string]: unknown;
}

export interface ForecastResponse extends StateEnvelope {
  history?: ForecastHistoryPoint[];
  deep_arch?: string;
  deep_forecast?: ForecastPoint[];
  classical_summary?: JsonRow | null;
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
  zero_shot?: { backend?: string; predictions?: { label: string; probability: number }[]; [key: string]: unknown } | null;
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
  model_vs_baseline_lift_pct?: number;
  mae_improvement_vs_baseline_pct?: number;
}

export interface TrendPoint {
  label?: string;
  actual?: number | null;
  predicted?: number | null;
  baseline?: number | null;
  [key: string]: unknown;
}

export interface SPCPoint {
  label?: string;
  residual?: number;
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
