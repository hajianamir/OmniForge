import type { StudioId } from '../types';

export interface FormulaSpec {
  equation: string;
  symbols: string;
}

export interface MethodEntry {
  name: string;
  paradigm: string;
  whatItDoes: string;
  nullHypothesis?: string;
  formula?: FormulaSpec;
  whenToUse: string;
  assumptions?: string;
  recommendedParams: string;
  pitfalls: string;
}

export interface ThresholdRow {
  metric: string;
  range: string;
  level: 'good' | 'warn' | 'bad' | 'info';
  interpretation: string;
  action: string;
}

export interface ThresholdTable {
  title: string;
  subtitle?: string;
  rows: ThresholdRow[];
}

export interface DecisionRow {
  application: string;
  dataCondition: string;
  recommended: string;
  alternative: string;
  rationale: string;
}

export interface DecisionMatrix {
  title: string;
  rows: DecisionRow[];
}

export interface TheorySection {
  heading: string;
  paragraphs: string[];
}

export interface WorkedExample {
  title: string;
  scenario: string;
  stepByStepCalculation: string[];
  numericalConclusion: string;
  engineeringAction: string;
}

export interface UIControlGuideRow {
  control: string;
  defaultValue: string;
  whenIncreased: string;
  whenDecreased: string;
  expertGuidance: string;
}

export interface TroubleshootingItem {
  symptom: string;
  mathematicalCause: string;
  exactRemediationSteps: string;
}

export interface StudioDeepSupplement {
  deepTheory: TheorySection[];
  workedExamples: WorkedExample[];
  uiControlsGuide: UIControlGuideRow[];
  troubleshootingPlaybook: TroubleshootingItem[];
  extraMethods?: MethodEntry[];
}

export interface StudioChapter {
  studioId: StudioId;
  chapterNum: string;
  title: string;
  subtitle: string;
  overview: string;
  realWorldApplications: string[];
  stepByStep: string[];
  provenanceRule: string;
  methods: MethodEntry[];
  thresholdTables: ThresholdTable[];
  decisionMatrix: DecisionMatrix;
  deepTheory?: TheorySection[];
  workedExamples?: WorkedExample[];
  uiControlsGuide?: UIControlGuideRow[];
  troubleshootingPlaybook?: TroubleshootingItem[];
}

export interface WorkflowRoadmapStep {
  step: number;
  studioId: StudioId;
  stageName: string;
  requirement: 'MANDATORY' | 'RECOMMENDED' | 'CONDITIONAL';
  objective: string;
  exactActions: string[];
  expectedOutputs: string;
  criticalNote?: string;
}

export const WORKFLOW_ROADMAP: WorkflowRoadmapStep[] = [
  {
    step: 1,
    studioId: '01',
    stageName: 'Data Ingestion & Automatic Intake Defect Audit',
    requirement: 'MANDATORY',
    objective: 'Load raw tabular data from any supported source and automatically inspect structural, statistical, and formatting defects before making any changes.',
    exactActions: [
      'Upload CSV, TSV, multi-sheet Excel (.xlsx/.xls/.xlsm), Parquet, JSON, connect SQL (SQLite/PostgreSQL/MySQL/MSSQL), connect MS Access (.accdb/.mdb), or load a benchmark dataset.',
      'Review the automatic InitialIntakeAuditor scorecard (Health Grade A–F, defect counts, severity badges, and recommended remediation rules).',
      'Inspect the paginated Data Explorer in RAW view to confirm column names, data types, and units.',
    ],
    expectedOutputs: 'Immutable Raw Data snapshot + Active Data copy + Intake Defect Audit Scorecard + Format Audit report.',
  },
  {
    step: 2,
    studioId: '01',
    stageName: 'Configure Global Targets & Reversible Column Exclusions',
    requirement: 'MANDATORY',
    objective: 'Define what you want to predict/analyze and prevent non-predictive identifiers or post-outcome leakage columns from corrupting downstream algorithms.',
    exactActions: [
      'Open "Targets" in the topbar and select one or more target variables (y).',
      'Open "Exclusions" in the topbar and exclude row IDs, primary keys, constant metadata codes, free-text notes (if not running NLP), and any leakage variable recorded AFTER the target outcome occurred.',
    ],
    expectedOutputs: 'Clean analysis schema applied automatically across Studios 02–14 without deleting raw columns.',
  },
  {
    step: 3,
    studioId: '01',
    stageName: 'Enable Anti-Leakage Train/Test Split Lock (BEFORE Data Cleaning)',
    requirement: 'RECOMMENDED',
    objective: 'Freeze an auditable Train/Test holdout partition prior to any statistic-dependent preprocessing so validation accuracy remains 100% scientifically honest.',
    exactActions: [
      'Click "Split Lock" in the topbar.',
      'Choose Stratified (for classification), Random (for IID regression), Group (for repeated subjects/machines), or Chronological Time-Series (for ordered temporal data).',
      'Set Test Ratio (standard: 0.20 for N >= 500; 0.25–0.30 for small N) and CV Folds (5 or 10), then click "Lock Train/Test Split".',
    ],
    expectedOutputs: 'Locked Train/Test indices enforced automatically by Studio 01 Cleaning (fit_transform_split) and Studio 02 Clustering (fit_predict_split).',
    criticalNote: 'GOLD-STANDARD RULE: Always enable Split Lock BEFORE Data Cleaning whenever your goal includes predictive modeling (Baselines, AutoML, Non-Parametric Models, Deep Learning, or Supervisory Monitoring). Computing medians, means, outlier fences, or scalers on the combined dataset before splitting leaks test-set statistics into training. Exception: pure structural text cleanup or exporting a clean spreadsheet without modeling does not require Split Lock.',
  },
  {
    step: 4,
    studioId: '01',
    stageName: '6D Quality Audit & Reproducible Data Cleaning Commit',
    requirement: 'MANDATORY',
    objective: 'Remediate missing values, duplicates, corrupt tokens, extreme measurement errors, and constant/high-missing columns with complete audit traceability.',
    exactActions: [
      'Run "Deep 6D Quality Profile" to review Completeness, Uniqueness, Validity, Consistency, Accuracy (requires trusted reference data), and Timeliness scores.',
      'Select Auto (smart safe defaults), Complete (comprehensive sweep), or Custom (explicit imputer, outlier rule, threshold).',
      'Click "Dry-Run Preview" to verify row/column impact first, then click "Apply & Commit Cleaning".',
    ],
    expectedOutputs: 'Active dataset transitions from RAW -> CLEANED; deleted rows/columns recorded permanently in the Forensic Deleted-Records Audit Log (exportable as CSV); Undo & Revert to Raw always available.',
  },
  {
    step: 5,
    studioId: '02',
    stageName: 'Clustering, Multivariate Anomaly Screening & Data Mining',
    requirement: 'RECOMMENDED',
    objective: 'Uncover natural operating regimes, isolate and optionally remove corrupt clusters or multivariate anomalies, and mine association rules or high-risk subgroups.',
    exactActions: [
      'Run Clustering Preview (Auto/Complete benchmarks K-Means, GMM, DBSCAN, Agglomerative, BIRCH, Spectral using Silhouette/CH/DB indices).',
      'Inspect cluster cards and 2D PCA scatter; optionally remove corrupt/noise clusters (e.g., DBSCAN -1) OR attach cluster_id to Active Data.',
      'Switch to Anomaly Screening to evaluate Isolation Forest, LOF, and One-Class SVM; attach anomaly scores or remove confirmed corrupt anomalies.',
      'Optionally run the Data Mining tab (Apriori Association Rules, Subgroup Discovery WRAcc, and Manifold Projections).',
    ],
    expectedOutputs: 'Studio 02 green status LED activates; provenance upgrades to CLEANED + CLUSTER-SCREENED; all downstream studios consume the refined dataset automatically.',
  },
  {
    step: 6,
    studioId: '03',
    stageName: 'Universal Diagnostics, Ranked Baseline Equations & Tail Risk',
    requirement: 'MANDATORY',
    objective: 'Test distributional assumptions, detect multicollinearity (VIF), quantify tail risk (VaR/CVaR), and establish scientifically ranked mathematical baseline equations.',
    exactActions: [
      'Configure significance level alpha (default 0.05), holdout ratio (default 0.20), and multiple-testing correction (Benjamini–Hochberg or Bonferroni).',
      'Run Universal Diagnostics in Complete or Auto mode.',
      'Inspect normality test verdicts (to decide Parametric vs Non-Parametric modeling), VIF > 10 warnings, and every ranked Baseline Equation Card (exact formula, coefficients, Lift vs Naive Floor, LaTeX & Python copy).',
    ],
    expectedOutputs: 'Complete diagnostic table, VIF report, VaR/CVaR risk profile, and ranked Parametric + Non-Parametric baseline models registered for comparison.',
  },
  {
    step: 7,
    studioId: '04',
    stageName: 'Automated EDA & Hypothesis Lab',
    requirement: 'RECOMMENDED',
    objective: 'Visualize distributions and correlations, and choose statistical tests that match the sampling design.',
    exactActions: [
      'Click "Run Automated EDA" to inspect histograms, skewness, correlation heatmaps, missingness bars, and automated statistical insights.',
      'Use Complete for a broader battery, Auto for a smaller battery, or Custom to select a test, input columns, and alpha.',
    ],
    expectedOutputs: 'EDA visualizations, test statistics, raw and adjusted battery p-values, and reasons for skipped tests.',
  },
  {
    step: 8,
    studioId: '05',
    stageName: 'Feature Engineering & 4-Method Consensus Selection',
    requirement: 'CONDITIONAL',
    objective: 'Create nonlinear/skew-corrected/cyclical predictors when needed and prune noisy low-signal features using multi-method consensus.',
    exactActions: [
      'Preview or commit Yeo-Johnson skew corrections, pairwise interactions, polynomial terms, and datetime cyclical encodings (sin/cos).',
      'Preview consensus rankings and review the source predictors. Apply exclusions explicitly to change the predictors used in downstream studios.',
    ],
    expectedOutputs: 'Engineered features committed to Active Data (ENGINEERED provenance) and low-signal columns reversibly excluded.',
  },
  {
    step: 9,
    studioId: '06',
    stageName: 'Predictive Modeling — AutoML, Non-Parametric Analytical & Deep Learning',
    requirement: 'MANDATORY',
    objective: 'Train high-accuracy models strictly benchmarked against strong baselines, obtaining both peak predictive accuracy and explicit closed-form analytical equations.',
    exactActions: [
      'Tab 1 (AutoML Hub): Run Native Stacking, Multi-Layer Stacker, Evolutionary Pipeline, Fast GBDT, or installed External Backends (FLAML/AutoGluon/H2O/TPOT) for maximum tabular accuracy.',
      'Tab 2 (Non-Parametric Analytical): Run Penalized B-Spline GAM, Nadaraya-Watson Kernel, RBF Kernel Ridge, Distance-Weighted k-NN, and Isotonic Regression to get distribution-free models with exact formulas and per-feature GAM curves.',
      'Tab 3 (Deep Learning): Benchmark FT-Transformer, TabNet, ResNet-Tabular, MLP, 1D-CNN, and GNN against baselines.',
      'Tab 4 (Predictions): Score individual records or batch target-free CSV files.',
    ],
    expectedOutputs: 'Champion model promoted to Active Model (with mathematical equation, Lift over Baseline, and leaderboard) across Studios 07, 11, 12, and 14.',
  },
  {
    step: 10,
    studioId: '07',
    stageName: 'Responsible AI — Explainability, Fairness & Governance',
    requirement: 'RECOMMENDED',
    objective: 'Audit how the champion model makes predictions, verify absence of demographic bias, and produce compliance Model Cards before deployment.',
    exactActions: [
      'Run Global Permutation Importance, Local SHAP row attribution, and Partial Dependence Plots (PDP).',
      'Select any sensitive/protected attribute and run the Fairness Audit (Demographic Parity, Equal Opportunity, Disparate Impact >= 0.80) and inspect the generated Model Card.',
    ],
    expectedOutputs: 'Global & local attribution charts, PDP curves, fairness compliance verdict, and audit-ready Model Card.',
  },
  {
    step: 11,
    studioId: '08',
    stageName: 'Decision Intelligence, Specialized AI, DAG Pipelines & MLOps (Studios 08–11)',
    requirement: 'CONDITIONAL',
    objective: 'Extend modeling to forecasting, causal policy estimation, portfolio optimization, Monte Carlo simulation, unstructured modalities (NLP/Vision/Graphs/Survival), DAG automation, and MLOps drift tracking.',
    exactActions: [
      'Studio 08: Forecast future horizons (ARIMA/ETS + TFT/N-BEATS/PatchTST/LSTM), estimate Doubly Robust Causal ATE, optimize portfolios (Markowitz), or run Monte Carlo simulations.',
      'Studio 09: Run NLP, Computer Vision, Graph Link Prediction, Recommender Systems, or Kaplan-Meier/CoxPH Survival Analysis.',
      'Studio 10 & 11: Automate repeatable DAG pipelines, register Staging/Production models, and monitor PSI/KS covariate drift.',
    ],
    expectedOutputs: 'Forecasts, causal effects, optimal allocations, specialized AI insights, versioned registry records, and drift alerts.',
  },
  {
    step: 12,
    studioId: '12',
    stageName: 'Executive Reporting & Multi-Format Export Center',
    requirement: 'MANDATORY',
    objective: 'Assemble an auditable, customizable executive & technical report covering every completed stage and export in publication-ready formats.',
    exactActions: [
      'Select desired chapters from the 14 available sections (or use Complete / Executive Preset).',
      'Click "Compile Report" to preview narrative summaries, tables, and embedded charts.',
      'Download PDF, XLSX workbook, DOCX Word document, HTML, Markdown, CSV, JSON, or LaTeX.',
    ],
    expectedOutputs: 'Publication-ready multi-format reports containing full lineage, equations, metrics, exclusions, and supervisory telemetry.',
  },
  {
    step: 13,
    studioId: '14',
    stageName: 'Live Prediction & Engineering Supervisory Control Room',
    requirement: 'CONDITIONAL',
    objective: 'Deploy your champion and baseline models to monitor real-world incoming telemetry or operational records against strict engineering tolerances with real-time alarms.',
    exactActions: [
      'Select Champion Model and Baseline Reference Model from the live model catalog.',
      'Configure tolerance limits: Max % Error, Max Absolute Error, Max Sigma-Z (SPC), Baseline Ratio envelope, Warning Fraction, and Batch Breach Rate.',
      'Connect live Excel/CSV, SQL Database, MS Access, Live Telemetry Simulator, or Single Sensor Probe and run evaluation.',
      'Monitor HEALTHY / WARNING / CRITICAL_ALARM LEDs, inspect Actual-vs-Predicted SPC plots and prioritized Alarm Logs, and save the workspace via ".omniforge Project".',
    ],
    expectedOutputs: 'Real-time operational alarm triage, SPC telemetry plots, longitudinal run history, and downloadable scored audit files.',
  },
];

export const MASTER_STAT_DECISION_ATLAS: DecisionMatrix = {
  title: 'Master Statistical Test Selection Atlas (Application -> Recommended Test)',
  rows: [
    {
      application: 'Check if a continuous variable is Gaussian / Normal',
      dataCondition: '1 numeric column, n <= 5,000',
      recommended: 'Shapiro-Wilk Test',
      alternative: "D'Agostino K² (n > 5,000) or Anderson-Darling (tail-sensitive)",
      rationale: 'Shapiro-Wilk has the highest statistical power for small-to-medium samples; Anderson-Darling penalizes heavy tails most strongly.',
    },
    {
      application: 'Compare 2 independent groups (e.g., Control vs Treatment, Fault vs Normal)',
      dataCondition: '1 numeric + 1 binary category; approximately normal',
      recommended: "Welch's Two-Sample t-Test",
      alternative: 'Mann-Whitney U Test (if skewed, ordinal, or outliers present)',
      rationale: "Welch's t-test does NOT require equal variances and outperforms Student's t-test whenever group spreads or sizes differ.",
    },
    {
      application: 'Compare before vs. after on the SAME units / sensors / patients',
      dataCondition: '2 paired numeric columns; normal differences',
      recommended: 'Paired t-Test',
      alternative: 'Wilcoxon Signed-Rank Test (if differences are skewed/heavy-tailed)',
      rationale: 'Pairing removes unit-to-unit baseline variance, dramatically increasing statistical power.',
    },
    {
      application: 'Compare 3 or more independent groups (e.g., Shift A/B/C, 4 machine lines)',
      dataCondition: '1 numeric + 1 multi-class category; normal & similar variances',
      recommended: 'One-Way ANOVA (F-Test)',
      alternative: 'Kruskal-Wallis H Test (non-normal / skewed / outliers)',
      rationale: 'Tests whether at least one group central tendency differs without inflating Type-I error via multiple t-tests.',
    },
    {
      application: 'Detect ANY distribution change (mean, variance, skewness, or bimodality)',
      dataCondition: '2 numeric samples or Train vs Live production batch',
      recommended: 'Two-Sample Kolmogorov-Smirnov (KS) Test',
      alternative: 'Population Stability Index (PSI) for binned production drift',
      rationale: 'Distribution-free test on the maximum vertical gap between empirical CDFs.',
    },
    {
      application: 'Test whether groups have equal variance / spread (homoscedasticity)',
      dataCondition: '1 numeric + 1 categorical grouping column',
      recommended: 'Levene Test (median-centered) or Fligner-Killeen',
      alternative: 'Bartlett Test (ONLY if strictly Gaussian — very sensitive to non-normality)',
      rationale: 'Levene and Fligner-Killeen remain reliable under skewness and heavy tails.',
    },
    {
      application: 'Measure relationship between 2 continuous numeric variables',
      dataCondition: '2 numeric columns; linear & roughly Gaussian',
      recommended: 'Pearson Correlation (r)',
      alternative: 'Spearman Rank (ρ) for monotonic/curved; Kendall (τ) for small n / many ties; Mutual Information for U-shaped nonlinear',
      rationale: 'Pearson measures strict linear association; Spearman/Kendall resist outliers and capture monotonic curves.',
    },
    {
      application: 'Test independence between 2 categorical variables (e.g., Region × Failure Type)',
      dataCondition: 'Contingency table with all expected cell counts >= 5',
      recommended: "Pearson's Chi-Square Test (χ²) + Cramér's V",
      alternative: "Fisher's Exact Test (for 2×2 tables with any expected count < 5)",
      rationale: 'Chi-square asymptotic approximation breaks down on sparse cells where Fisher computes the exact hypergeometric p-value.',
    },
    {
      application: 'Control false discoveries when running 10–500+ statistical tests simultaneously',
      dataCondition: 'Exploratory multi-column screening battery',
      recommended: 'Benjamini-Hochberg (FDR, q = 0.05)',
      alternative: 'Holm-Bonferroni or Bonferroni (strict confirmatory Family-Wise Error Rate)',
      rationale: 'At alpha = 0.05 across 100 null tests, ~5 false positives occur by pure chance without correction.',
    },
  ],
};

export const MASTER_MODEL_DECISION_ATLAS: DecisionMatrix = {
  title: 'Master Predictive Model Selection Atlas (Goal & Data Profile -> Best Model Family)',
  rows: [
    {
      application: 'Transparent linear formula, small sample size, or mandatory scientific benchmark',
      dataCondition: 'Linear relationships, low collinearity (VIF < 10), N >= 30',
      recommended: 'Ridge Linear Regression / Logistic Regression (Studio 03)',
      alternative: 'L1 Lasso Selection + Ridge (if many irrelevant predictors)',
      rationale: 'Exact closed-form coefficients in original units; lowest variance on small datasets.',
    },
    {
      application: 'Curved / non-Gaussian relationships WITH transparent analytical equation & smooth per-feature curves',
      dataCondition: 'Non-normal diagnostics, nonlinear marginal effects, N = 100–20,000',
      recommended: 'Penalized B-Spline GAM (Studio 06 Non-Parametric)',
      alternative: 'RBF Kernel Ridge (if strong multi-feature interactions dominate)',
      rationale: 'Additive smooth splines y = β₀ + Σ fⱼ(xⱼ) capture curves without black-box opacity.',
    },
    {
      application: 'Complex smooth engineering/physical response surface with multi-way interactions',
      dataCondition: 'Continuous features, smooth nonlinear physics, N <= 5,000',
      recommended: 'RBF Kernel Ridge (RKHS) (Studio 06 Non-Parametric)',
      alternative: 'Nadaraya-Watson Kernel or Distance-Weighted k-NN (for low p <= 12 local regimes)',
      rationale: 'Infinite-dimensional Gaussian kernel captures arbitrary smooth interaction surfaces analytically.',
    },
    {
      application: 'Physically guaranteed monotonic curve (wear vs time, dose-response, probability calibration)',
      dataCondition: 'Known monotonic trend along a dominant predictor',
      recommended: 'Isotonic Monotonic Regression (PAVA) (Studio 06 Non-Parametric)',
      alternative: 'Penalized B-Spline GAM with smooth regularization',
      rationale: 'Guarantees zero non-physical reversals (strictly non-decreasing or non-increasing step function).',
    },
    {
      application: 'Maximum predictive accuracy on heterogeneous tabular business / industrial datasets',
      dataCondition: 'Mixed numeric & categorical features, interactions, N = 200–1,000,000+',
      recommended: 'Multi-Backend AutoML (Stacking / Multi-Layer / FLAML / AutoGluon)',
      alternative: 'FT-Transformer / ResNet-Tabular (large N > 10,000)',
      rationale: 'Gradient-boosted ensembles + stacked meta-learners consistently dominate general tabular benchmarks.',
    },
    {
      application: 'Medium-to-large nonlinear tabular, ordered spectral vectors, or graph-coupled entities',
      dataCondition: 'Large N, complex representation learning, or relational graph structure',
      recommended: 'FT-Transformer / TabNet / ResNet-Tabular / 1D-CNN / GNN (Studio 06 DL)',
      alternative: 'AutoML Multi-Layer Stacker',
      rationale: 'Attention (FT-Transformer), sparse masks (TabNet), and message passing (GNN) exploit deep structural patterns.',
    },
  ],
};

export const MASTER_THRESHOLD_COMPENDIUM: ThresholdTable = {
  title: 'Universal Statistical & Engineering Numeric Threshold Reference',
  subtitle: 'Standard scientific reference ranges across OmniForge studios — adjust for mission-critical safety or regulatory domains.',
  rows: [
    { metric: 'Significance Level (α)', range: 'α = 0.05 (standard) | 0.01 (strict) | 0.001 (critical)', level: 'info', interpretation: 'Maximum tolerable Type-I false-positive probability.', action: 'Use 0.05 for general analytics, 0.01 for engineering validation, 0.001 for high-risk clinical/safety tests.' },
    { metric: 'Effect Size — Cohen d (t-Tests)', range: '0.20 Small | 0.50 Medium | >= 0.80 Large', level: 'good', interpretation: 'Standardized mean difference independent of sample size n.', action: 'At huge n, tiny d < 0.10 can still give p < 0.001 — require BOTH statistical & practical significance.' },
    { metric: 'Correlation Strength |r| or |ρ|', range: '<0.10 Negligible | 0.10–0.30 Weak | 0.30–0.50 Moderate | 0.50–0.70 Strong | >0.70 Very Strong', level: 'info', interpretation: 'Magnitude of linear (r) or monotonic (ρ) association.', action: 'Pairwise |r| > 0.90 between two predictors flags severe redundancy — check VIF.' },
    { metric: 'Multicollinearity (VIF)', range: '< 5 Safe | 5 – 10 Moderate | > 10 Severe', level: 'warn', interpretation: 'Variance Inflation Factor measuring predictor redundancy.', action: 'If VIF > 10, remove redundant column, apply Studio 05 Selection, or use Ridge/GAM regularization.' },
    { metric: 'Skewness |g₁|', range: '< 0.5 Symmetric | 0.5 – 1.0 Moderate | > 1.0 Highly Skewed', level: 'warn', interpretation: 'Asymmetry of numeric feature distribution.', action: 'If |g₁| > 1.0, use Median/IQR/Mann-Whitney or apply Yeo-Johnson skew correction in Studio 05.' },
    { metric: 'Clustering Silhouette Score', range: '< 0.25 Weak/No Structure | 0.25–0.50 Fair | 0.50–0.70 Strong | > 0.70 Excellent', level: 'good', interpretation: 'Cohesion vs separation across clusters (-1 to +1).', action: 'If < 0.25, try GMM/DBSCAN or verify whether data is genuinely unimodal.' },
    { metric: 'Fairness Disparate Impact (DI)', range: '0.80 <= DI <= 1.25 Compliant | < 0.80 Adverse Impact', level: 'bad', interpretation: 'Four-Fifths (80%) regulatory fairness ratio.', action: 'If DI < 0.80, audit proxy features in Studio 07 and adjust thresholds or reweight training.' },
    { metric: 'Population Stability Index (PSI)', range: '< 0.10 Stable | 0.10 – 0.25 Moderate Drift | >= 0.25 Major Shift', level: 'bad', interpretation: 'Quantifies covariate distribution shift between reference and live data.', action: 'At PSI >= 0.25, trigger model recalibration/retraining and audit upstream sensors.' },
    { metric: 'Supervisory Residual Z-Score (SPC)', range: '|Z| < 2.0 Normal | 2.0 – 3.0 Warning | >= 3.0 Critical Excursion', level: 'bad', interpretation: 'Shewhart Statistical Process Control sigma distance from expected error.', action: 'Exceeding 3σ happens < 0.27% by chance under normal noise — inspect asset immediately.' },
  ],
};

export const HANDBOOK_FAQS = [
  {
    q: 'Must Split Lock be enabled BEFORE Data Cleaning?',
    a: 'YES — whenever your objective includes predictive modeling, baseline comparison, AutoML, Non-Parametric Analytical Models, Deep Learning, or Studio 14 Supervisory deployment. Locking the Train/Test split immediately after loading data and selecting Targets ensures that Studio 01 Cleaning (fit_transform_split) fits medians/means/KNN/MICE imputers, outlier fences, and column filters strictly on Train, and Studio 02 Clustering (fit_predict_split) fits scalers/centroids strictly on Train. Only structural formatting fixes (whitespace/HTML removal) and pure BI spreadsheet export without ML modeling can safely skip Split Lock.',
  },
  {
    q: 'What is the exact difference between Auto, Complete, and Custom execution modes?',
    a: 'AUTO automatically selects the safest, highest-ROI statistical rules and algorithms tailored to your dataset dimensions and types. COMPLETE runs the full exhaustive battery of all available algorithms, tests, or cleaning rules for comprehensive benchmarking. CUSTOM exposes every individual parameter (alpha, outlier method/multiplier, imputer, k, bandwidth, kernels, exact hypothesis test) for expert manual control.',
  },
  {
    q: 'Do downstream studios (02–14) automatically use my Cleaned & Cluster-Screened data?',
    a: 'YES. Every downstream studio reads the latest committed Active Dataset (reflected in the topbar Provenance Badge: RAW -> CLEANED -> CLEANED + CLUSTER-SCREENED -> ENGINEERED). Your Raw Data is never overwritten — you can toggle Active vs Raw in Studio 01, click Undo, Revert to Raw, or export the Forensic Deleted-Records Audit Log at any time.',
  },
  {
    q: 'When should I choose Non-Parametric Analytical Models (Studio 06 Tab 2) instead of Linear models or AutoML?',
    a: 'Choose Studio 06 Non-Parametric Analytical Models (Penalized B-Spline GAM, Nadaraya-Watson, RBF Kernel Ridge, Distance-Weighted k-NN, Isotonic Regression) when Studio 03/04 diagnostics show non-normal distributions or nonlinear curves AND you require explicit mathematical equations, smooth interpretable per-feature curves fⱼ(xⱼ), or monotonic guarantees rather than an opaque black-box ensemble.',
  },
  {
    q: 'How are results preserved when navigating between studios or refreshing the browser?',
    a: 'All visited studios stay mounted in memory (preserving active sub-tabs, parameters, and views instantaneously) AND every completed API payload is cached in the backend session store so even a full browser F5 refresh re-hydrates your work automatically. To archive everything permanently to disk, use ".omniforge Project" -> Download .omniforge.',
  },
  {
    q: 'How do I inspect large tables, charts, or long mathematical equations in full detail?',
    a: 'Every chart, table, and mathematical equation card includes built-in Zoom (fullscreen inspection overlay) and Pop-Out (independent browser window) controls in its top-right corner, plus one-click LaTeX and Python copy buttons on equations.',
  },
];
