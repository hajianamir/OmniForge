import type { StudioChapter } from './handbookTypes';

export const HANDBOOK_CHAPTERS_PART2: StudioChapter[] = [
  // =========================================================================
  // CHAPTER 8 — STUDIO 08
  // =========================================================================
  {
    studioId: '08',
    chapterNum: 'Chapter 08',
    title: 'Forecasting, Causal Inference, Portfolio Optimization & Monte Carlo Simulation',
    subtitle: 'Classical + Deep Time-Series Forecasting (TFT/N-BEATS/PatchTST/LSTM), Doubly Robust Causal ATE (AIPW), Markowitz Optimization, and Probabilistic Simulation.',
    overview:
      'Predictive accuracy tells you what is likely to happen if nothing changes; Decision Intelligence tells you what will happen over future horizons, what happens IF you actively intervene, how to allocate limited resources optimally, and how wide your uncertainty fan is under random shocks. Studio 08 unifies all four quantitative decision-science pillars in one studio.',
    realWorldApplications: [
      'Energy load, retail demand, financial cash-flow, and sensor degradation forecasting across future steps.',
      'Measuring true causal ROI of marketing campaigns, pricing shifts, engineering retrofits, or medical treatments from observational data (AIPW).',
      'Capital, inventory, energy-mix, or portfolio risk-return allocation along the Markowitz Efficient Frontier.',
      'Stress-testing revenue, project schedules, or physical safety margins under 3,000+ stochastic Monte Carlo scenarios.',
    ],
    stepByStep: [
      'Tab 1 (Forecasting): Choose target series, forecast horizon (e.g., 14–30 steps), and Deep Forecaster architecture (TFT, N-BEATS, PatchTST, or LSTM). Click "Run Forecasting Tournament" to benchmark Classical vs Deep trajectories.',
      'Tab 2 (Causal Inference): Select Treatment column (binary/action variable) and Outcome column, then click "Estimate Causal ATE" to compute Doubly Robust AIPW causal lift controlling for confounders.',
      'Tab 3 (Optimization): Run Markowitz Mean-Variance Optimizer over numeric return/asset columns to obtain optimal Sharpe-maximizing weights.',
      'Tab 4 (Simulation): Run 3,000-trial Monte Carlo simulation on your target column to obtain P5 / P50 / P95 confidence intervals and tail risk.',
    ],
    provenanceRule: 'Read-only on Active Data. All four tab results cache independently and flow into Studio 12 Executive Reports.',
    methods: [
      {
        name: 'Classical Time-Series (ARIMA / Exponential Smoothing) vs. Deep Forecasters (N-BEATS, PatchTST, TFT, LSTM)',
        paradigm: 'Time-Series Forecasting (Tab 1)',
        whatItDoes:
          'Classical ARIMA/ETS models decompose a chronological series into autoregressive lags, differenced trends, moving-average errors, and seasonal components. Deep Forecasters learn complex nonlinear multi-horizon temporal patterns: N-BEATS decomposes trend/seasonality basis blocks; PatchTST applies Transformer self-attention over temporal patches; TFT combines variable gating with temporal attention; LSTM maintains recurrent memory cells.',
        formula: {
          equation: 'ARIMA(p,d,q): (1 − Σ φ_i B^i)(1 − B)^d y_t = c + (1 + Σ θ_j B^j) ε_t    |    sMAPE = (200/H) Σ |y_t − ŷ_t| / (|y_t| + |ŷ_t|)',
          symbols: 'B = backshift operator (B y_t = y_{t-1}); H = forecast horizon steps; lower MAE / RMSE / sMAPE = better.',
        },
        whenToUse: 'Classical excels on short (< 200 steps) clean seasonal univariate series. N-BEATS / PatchTST / TFT shine on longer series (>= 250 steps) with complex multi-seasonality or nonlinear dynamics.',
        recommendedParams: 'Always keep Time-Series Split Lock enabled; keep forecast horizon <= 20% of historical series length.',
        pitfalls: 'Never shuffle rows randomly before forecasting — random splits destroy autocorrelation structure and create severe look-ahead leakage.',
      },
      {
        name: 'Doubly Robust Causal Inference — AIPW (Augmented Inverse Probability Weighting)',
        paradigm: 'Causal Effect Estimation (Tab 2)',
        whatItDoes:
          'Answers the causal question "What is the Average Treatment Effect (ATE) of changing Treatment T from 0 to 1 on Outcome Y?" while adjusting for confounding variables X. Combines BOTH a propensity score model e(X) = P(T=1|X) AND outcome regression models μ₁(X), μ₀(X) — remaining statistically consistent if EITHER model is correctly specified ("Doubly Robust").',
        formula: {
          equation: 'τ̂_AIPW = (1/n) Σ_{i=1}^n [ μ̂₁(X_i) − μ̂₀(X_i) + T_i(Y_i − μ̂₁(X_i))/ê(X_i) − (1−T_i)(Y_i − μ̂₀(X_i))/(1−ê(X_i)) ]',
          symbols: 'T_i ∈ {0, 1} treatment indicator; ê(X_i) = propensity score; μ̂₁, μ̂₀ = predicted potential outcomes under treated vs control.',
        },
        whenToUse: 'Observational A/B or policy studies where randomized experiments are infeasible (e.g., maintenance retrofit effect, drug effectiveness, price discount impact).',
        assumptions: 'Ignorability (no major unmeasured confounders) and Positivity (0 < ê(X) < 1 overlap between treated and control units).',
        recommendedParams: 'Verify 95% confidence interval does NOT cross zero before declaring a causal lift.',
        pitfalls: 'Adjusting for post-treatment colliders or extreme propensity scores near 0 or 1 inflates variance — inspect confounder selection carefully.',
      },
      {
        name: 'Markowitz Mean-Variance Portfolio Optimization & Monte Carlo Simulation',
        paradigm: 'Optimization & Stochastic Risk (Tabs 3–4)',
        whatItDoes:
          'Markowitz solves a convex quadratic optimization problem to find asset/resource weights w that minimize total portfolio variance wᵀΣw (or maximize Sharpe Ratio) given expected returns μ and covariance matrix Σ. Monte Carlo generates thousands of stochastic draws from fitted empirical/parametric distributions to quantify P5 downside, P50 median, and P95 upside.',
        formula: {
          equation: 'Markowitz: max_w  (wᵀμ − r_f) / √(wᵀΣw)   s.t.  Σ w_i = 1, w_i ≥ 0    |    Monte Carlo: P_q = Quantile_q({Y^(1), …, Y^(M)})',
          symbols: 'Σ = covariance matrix across assets; r_f = risk-free benchmark; M = simulation trials (default 3,000).',
        },
        whenToUse: 'Markowitz for multi-asset/resource risk-return allocation; Monte Carlo for probabilistic budgeting, reliability margins, and risk stress-testing.',
        recommendedParams: 'M >= 3,000 trials for stable P5 / P95 tail percentiles (standard error shrinks as 1/√M).',
        pitfalls: 'Historical covariance Σ can spike during crises — stress-test allocations with downside scenarios.',
      },
    ],
    thresholdTables: [
      {
        title: 'Forecasting, Causal & Portfolio Quantitative Benchmarks',
        rows: [
          { metric: 'Forecast Horizon Ratio (H / N)', range: '<= 15% Reliable | 15–25% Moderate | > 30% High Uncertainty', level: 'warn', interpretation: 'Forecast error compounds multiplicatively at long horizons.', action: 'Keep multi-step forecasts within <= 20% of observed history length.' },
          { metric: 'Forecast sMAPE / MAPE', range: '< 10% High Accuracy | 10–20% Good | 20–50% Reasonable | > 50% Poor', level: 'good', interpretation: 'Relative forecast error scale (Lewis industry convention).', action: 'Compare directly against seasonal naive benchmark.' },
          { metric: 'Causal Propensity Overlap ê(X)', range: '0.05 – 0.95 Valid | < 0.05 or > 0.95 Positivity Violation', level: 'warn', interpretation: 'Ensures treated and control units are genuinely comparable.', action: 'Trim/clip extreme propensities when overlap is sparse.' },
          { metric: 'Portfolio Sharpe Ratio', range: '< 0.5 Weak | 0.5–1.0 Fair | 1.0–2.0 Strong | > 2.0 Exceptional', level: 'good', interpretation: 'Excess return per unit of volatility risk.', action: 'Prefer diversified portfolios with high Sharpe over concentrated single-asset peaks.' },
        ],
      },
    ],
    decisionMatrix: {
      title: 'Studio 08 Decision Intelligence Selector',
      rows: [
        { application: 'Predict future values of a short seasonal time series', dataCondition: 'Chronological series, N < 200', recommended: 'Classical ARIMA / ETS (Tab 1)', alternative: 'N-BEATS (with small epochs)', rationale: 'Low parameter count prevents neural overfitting on short histories.' },
        { application: 'Forecast complex multi-seasonal telemetry or demand', dataCondition: 'Chronological series, N >= 250', recommended: 'PatchTST / TFT / N-BEATS (Tab 1)', alternative: 'Classical ETS benchmark', rationale: 'Attention and trend/seasonality basis blocks capture long-range dependencies.' },
        { application: 'Measure true intervention ROI without randomized trial', dataCondition: 'Treatment + Outcome + Confounders', recommended: 'Doubly Robust AIPW Causal ATE (Tab 2)', alternative: 'Propensity Matching / Stratification', rationale: 'Unbiased if either propensity OR outcome model is accurate.' },
        { application: 'Quantify P5 worst-case and P95 best-case uncertainty', dataCondition: 'Continuous outcome distribution', recommended: '3,000-Trial Monte Carlo Simulation (Tab 4)', alternative: 'Studio 03 VaR / CVaR', rationale: 'Reveals full probability density of plausible future outcomes.' },
      ],
    },
  },

  // =========================================================================
  // CHAPTER 9 — STUDIO 09
  // =========================================================================
  {
    studioId: '09',
    chapterNum: 'Chapter 09',
    title: 'Specialized AI — NLP, Computer Vision, Graphs, RecSys & Survival Analysis',
    subtitle: 'Dedicated multi-modal analytics for unstructured text, images, relational networks, collaborative recommenders, and right-censored time-to-event data.',
    overview:
      'Not all enterprise intelligence lives in flat numeric tables. Studio 09 extends OmniForge to five specialized scientific modalities — automatically using SOTA pretrained neural adapters (Transformers, CLIP, DINOv2, YOLO, PyG) when installed and seamless native paper-grade engines out-of-the-box with zero internet dependency.',
    realWorldApplications: [
      'NLP: Support ticket routing (Zero-Shot), compliance PII scrubbing, contract/log QA, sentiment, and LDA/NMF topic discovery.',
      'Computer Vision: Automated surface defect tagging, image segmentation, and object detection.',
      'Graph Analytics: Supply-chain bottleneck centrality, fraud-ring community detection, and missing link prediction (RRF / VGAE).',
      'Recommender Systems: Hybrid collaborative + content personalization for products, spare parts, or knowledge articles.',
      'Survival & Reliability Engineering: Equipment Mean Time Between Failures (MTBF), customer lifetime survival, and clinical Kaplan-Meier / CoxPH hazard ratios.',
    ],
    stepByStep: [
      'Tab 1 (NLP): Choose text column (or sample text), enter candidate labels or QA query, and run Zero-Shot / Topic / Sentiment / NER / PII analysis.',
      'Tab 2 (Vision): Upload an image (JPG/PNG), supply candidate scene/defect labels, and run Vision classification, segmentation, and detection.',
      'Tab 3 (Graph): Pick link-prediction method (RRF / Common Neighbors / Adamic-Adar / VGAE) and inspect centrality, communities, and interactive node-edge graph.',
      'Tab 4 (Recommender): Run Hybrid Recommender to generate Top-K personalized recommendations.',
      'Tab 5 (Survival): Select Duration column (time-to-event) and Event indicator column (1 = failure/event occurred, 0 = right-censored), then fit Kaplan-Meier curve & Cox Proportional Hazards.',
    ],
    provenanceRule: 'Read-only on Active Data. All five specialized tabs maintain separate persistent cache entries.',
    methods: [
      {
        name: 'Kaplan-Meier Survival Estimator & Cox Proportional Hazards (CoxPH)',
        paradigm: 'Time-to-Event & Reliability Engineering (Tab 5)',
        whatItDoes:
          'Standard regression fails on time-to-failure data because many machines/patients have NOT failed yet at observation end ("right-censored"). Kaplan-Meier estimates the exact non-parametric survival probability S(t) = P(T > t) properly accounting for censored units. CoxPH estimates how each covariate multiplies instantaneous hazard risk via Hazard Ratios exp(β_j).',
        formula: {
          equation: 'Kaplan-Meier: Ŝ(t) = Π_{t_i ≤ t} (1 − d_i / n_i)    |    CoxPH: h(t | x) = h₀(t) · exp(βᵀx),   HR_j = exp(β_j)',
          symbols: 'd_i = events at time t_i; n_i = units still at risk just before t_i; HR_j > 1 increases failure hazard, HR_j < 1 is protective.',
        },
        whenToUse: 'Mandatory whenever outcome is duration until an event (equipment breakdown, churn tenure, loan default timing, clinical survival) with right-censored active cases.',
        assumptions: 'CoxPH assumes proportional hazards (hazard ratio stays roughly constant over time) and non-informative censoring.',
        recommendedParams: 'Event column must be binary (1 = event observed, 0 = still active/censored); Duration must be strictly positive.',
        pitfalls: 'Dropping censored observations or treating censoring time as failure time severely underestimates true lifespan / MTBF.',
      },
      {
        name: 'Graph Centrality, Community Detection & Link Prediction (Adamic-Adar / RRF / VGAE)',
        paradigm: 'Network & Relational AI (Tab 3)',
        whatItDoes:
          'Measures node influence (PageRank, Degree, Betweenness), partitions tightly interconnected clusters (Communities), and predicts missing or future edges using neighborhood overlap heuristics (Common Neighbors, Adamic-Adar weighting rare shared hubs higher) or Variational Graph Autoencoder (VGAE) embeddings.',
        formula: {
          equation: 'Adamic-Adar: AA(u, v) = Σ_{w ∈ N(u)∩N(v)} 1 / log(|N(w)|)    |    VGAE: P(A_uv = 1 | Z) = σ(z_uᵀ z_v)',
          symbols: 'N(u) = neighbors of node u; z_u = learned latent node embedding vector.',
        },
        whenToUse: 'Fraud rings, supplier dependencies, sensor correlation networks, citation/knowledge graphs, and social interaction analysis.',
        recommendedParams: 'Use Adamic-Adar/RRF for fast interpretable link scoring; VGAE for dense structural latent representations.',
        pitfalls: 'Disconnected tiny graph components yield zero betweenness — inspect connected component count first.',
      },
      {
        name: 'NLP Suite (Zero-Shot, QA, PII Redaction, TF-IDF / NMF / LDA Topics) & Vision Hub',
        paradigm: 'Unstructured Text & Image Intelligence (Tabs 1–2)',
        whatItDoes:
          'Extracts structured labels, named entities, redacted PII tokens, and latent thematic topics from raw text via TF-IDF + Non-Negative Matrix Factorization / LDA (or pretrained Transformer models when installed). Vision Hub extracts semantic tags, region segments, and bounding detections.',
        formula: {
          equation: 'TF-IDF(t, d) = tf(t, d) · log(N / (1 + df(t)))    |    NMF Topic Factorization: V ≈ W · H  (W, H ≥ 0)',
          symbols: 'W = document-topic weights; H = topic-word weights.',
        },
        whenToUse: 'Maintenance logs, customer complaints, medical notes, inspection photos, and unstructured document triage.',
        recommendedParams: 'Provide 3–8 clear, mutually distinct candidate labels for Zero-Shot text or CLIP vision classification.',
        pitfalls: 'Very short texts (< 5 words) carry sparse lexical overlap — combine with metadata features when possible.',
      },
    ],
    thresholdTables: [
      {
        title: 'Survival Analysis & Specialized AI Interpretation Reference',
        rows: [
          { metric: 'CoxPH Hazard Ratio HR = exp(β)', range: 'HR = 1.0 Neutral | HR > 1.25 Risk Factor | HR < 0.80 Protective', level: 'info', interpretation: 'Multiplicative change in instantaneous failure/event rate per 1-unit increase.', action: 'Check p < 0.05 and 95% CI excluding 1.0 before declaring significance.' },
          { metric: 'Concordance Index (C-Index)', range: '0.50 Random | 0.65–0.75 Good | > 0.75 Strong Survival Model', level: 'good', interpretation: 'Survival equivalent of ROC-AUC for ranking event order.', action: 'Compare CoxPH C-index against nonlinear survival features.' },
          { metric: 'Graph Link Prediction AUC', range: '> 0.85 Strong Topology Signal | 0.70–0.85 Useful | < 0.60 Weak', level: 'good', interpretation: 'Ability to distinguish real missing links from random non-edges.', action: 'Try VGAE if local triangle overlap (Common Neighbors) is sparse.' },
        ],
      },
    ],
    decisionMatrix: {
      title: 'Studio 09 Modality Selector',
      rows: [
        { application: 'Equipment failure lifetime / customer churn tenure with active units', dataCondition: 'Duration > 0 + Binary Event (0/1)', recommended: 'Tab 5: Kaplan-Meier & CoxPH Survival', alternative: 'Studio 06 Classification at fixed horizon', rationale: 'Properly incorporates right-censored units still alive at cutoff.' },
        { application: 'Triage maintenance logs, tickets, or redact sensitive PII', dataCondition: 'Text column or pasted text', recommended: 'Tab 1: NLP Suite (Zero-Shot / PII / Topics)', alternative: 'External HuggingFace Transformers backend', rationale: 'Turns raw unstructured text into actionable categories & privacy-safe records.' },
        { application: 'Analyze entity networks, bottlenecks, or hidden links', dataCondition: 'Source/Target edge structure or relational rows', recommended: 'Tab 3: Graph Analytics & Link Prediction', alternative: 'Studio 06 Tab 3 GNN for node-level prediction', rationale: 'Captures relational topology invisible to flat tabular rows.' },
      ],
    },
  },

  // =========================================================================
  // CHAPTER 10 — STUDIO 10
  // =========================================================================
  {
    studioId: '10',
    chapterNum: 'Chapter 10',
    title: 'Visual DAG Pipeline Builder',
    subtitle: 'Reproducible Directed Acyclic Graph (DAG) workflow orchestrator chaining Quality -> Cleaning -> EDA -> Features -> AutoML.',
    overview:
      'Once you have explored your dataset interactively across Studios 01–06, Studio 10 lets you execute a standardized, deterministic, multi-stage Directed Acyclic Graph (DAG) pipeline in one click. Every stage passes verified typed artifacts to the next stage with per-node runtime logging and complete data-lineage recording.',
    realWorldApplications: [
      'Running repeatable weekly or monthly analytics refreshes on new data drops with guaranteed stage ordering.',
      'Benchmarking end-to-end pipeline execution time and verifying zero runtime breakage across stages.',
      'Teaching and auditing clear CRISP-DM execution topology via visual node-edge flow diagrams.',
    ],
    stepByStep: [
      'Toggle which pipeline stages to include (quality, clean, eda, features, automl).',
      'Review the visual DAG dependency graph.',
      'Click "Execute Visual DAG Pipeline" and inspect per-node execution status, timings, and produced artifacts.',
    ],
    provenanceRule: 'Executes selected stages sequentially on your active session workspace and logs a formal Pipeline DAG event in lineage.',
    methods: [
      {
        name: 'Topological DAG Execution Engine',
        paradigm: 'Workflow Orchestration',
        whatItDoes:
          'Validates that the stage graph contains zero cycles (acyclic) and executes nodes in strict topological dependency order, capturing typed intermediate artifacts and full error traces if any upstream node fails.',
        formula: {
          equation: 'DAG G = (V, E),   u → v ∈ E  ⇒  order(u) < order(v)',
          symbols: 'V = analytical stage nodes; E = typed data/artifact handoff edges.',
        },
        whenToUse: 'Whenever you want one-click end-to-end automation after confirming dataset schema and target selection.',
        recommendedParams: 'Standard sequence: quality -> clean -> eda -> features -> automl.',
        pitfalls: 'Enable Split Lock in the topbar BEFORE running a DAG containing clean + features + automl to maintain zero leakage.',
      },
    ],
    thresholdTables: [
      {
        title: 'DAG Pipeline Operational Readiness Checks',
        rows: [
          { metric: 'Topological Integrity', range: '0 Cycles (Valid DAG)', level: 'good', interpretation: 'Guarantees deterministic upstream-to-downstream data flow.', action: 'Always keep Quality & Clean upstream of Features & AutoML.' },
          { metric: 'Stage Failure Gate', range: 'All Nodes Status = COMPLETED', level: 'good', interpretation: 'Confirms every stage succeeded cleanly without silent drops.', action: 'Inspect node logs immediately if any stage reports ERROR.' },
        ],
      },
    ],
    decisionMatrix: {
      title: 'Studio 10 Pipeline Configuration Presets',
      rows: [
        { application: 'Fast data health & cleaning automation', dataCondition: 'Recurring spreadsheet / SQL drops', recommended: 'quality -> clean -> eda', alternative: 'Manual Studio 01 Custom Cleaning', rationale: 'Rapid repeatable hygiene and quality scorecard generation.' },
        { application: 'Full end-to-end automated modeling refresh', dataCondition: 'Locked Split + known Target', recommended: 'quality -> clean -> eda -> features -> automl', alternative: 'Step-by-step interactive Studios 01–06', rationale: 'Reproduces full CRISP-DM training pipeline consistently.' },
      ],
    },
  },

  // =========================================================================
  // CHAPTER 11 — STUDIO 11
  // =========================================================================
  {
    studioId: '11',
    chapterNum: 'Chapter 11',
    title: 'MLOps — Model Registry, Experiment Tracking & Covariate Drift Monitor',
    subtitle: 'Lifecycle version governance (Development / Staging / Production) and statistical data-drift detection via PSI and Kolmogorov-Smirnov tests.',
    overview:
      'Models degrade silently in production when incoming real-world populations shift away from training conditions ("covariate shift"). Studio 11 provides enterprise MLOps governance: register versioned models with stage lifecycle tags (Staging / Production), inspect experiment history, and run mathematical drift audits (PSI & 2-Sample KS) to detect feature shift early BEFORE prediction accuracy collapses.',
    realWorldApplications: [
      'Promoting audited champion models through formal Development -> Staging -> Production release gates.',
      'Detecting seasonal, economic, demographic, or sensor-calibration drift between baseline and current batches.',
      'Triggering disciplined retraining schedules based on quantitative PSI >= 0.25 thresholds rather than guesswork.',
    ],
    stepByStep: [
      'Register Model: Enter a semantic model name (e.g., turbine_eff_champion_v2), verify Active Model is trained, and click "Register Active Model".',
      'Drift Detection: Click "Run Drift Detection" to split reference vs current windows and evaluate per-feature Population Stability Index (PSI) and 2-Sample KS p-values.',
      'Review drifted feature flags and decide whether recalibration or retraining is required.',
    ],
    provenanceRule: 'Read-only on Active Data. Model registrations persist across the session registry and appear in Studio 12 Executive Reports.',
    methods: [
      {
        name: 'Population Stability Index (PSI) & 2-Sample Kolmogorov-Smirnov Drift Test',
        paradigm: 'Production Covariate Shift Detection',
        whatItDoes:
          'PSI bins the reference (expected E_i) and current (actual A_i) distributions into deciles and sums their symmetric Kullback-Leibler divergence to quantify total distribution movement. Two-Sample KS complements PSI with a bin-free exact p-value on maximum empirical CDF distance.',
        formula: {
          equation: 'PSI = Σ_{i=1}^B (A_i − E_i) · ln(A_i / E_i)    |    KS: D = sup_x |F_ref(x) − F_curr(x)|',
          symbols: 'E_i = reference bin fraction; A_i = current bin fraction; B = number of quantile bins (typically 10).',
        },
        whenToUse: 'Run regularly on incoming production data batches, new seasonal windows, or post-maintenance sensor streams.',
        recommendedParams: 'PSI < 0.10 no action; 0.10–0.25 investigate; >= 0.25 retrain/recalibrate.',
        pitfalls: 'KS test becomes hypersensitive at huge N > 50,000 (giving tiny p-values on negligible shifts) — ALWAYS pair KS with PSI magnitude!',
      },
    ],
    thresholdTables: [
      {
        title: 'Industry-Standard MLOps Covariate Drift Thresholds (PSI & KS)',
        rows: [
          { metric: 'Population Stability Index (PSI)', range: '< 0.10', level: 'good', interpretation: 'No significant population shift — model remains well-calibrated.', action: 'Continue routine monitoring.' },
          { metric: 'Population Stability Index (PSI)', range: '0.10 – 0.25', level: 'warn', interpretation: 'Moderate distribution shift detected.', action: 'Inspect drifted feature histograms and verify Studio 14 error trends.' },
          { metric: 'Population Stability Index (PSI)', range: '>= 0.25', level: 'bad', interpretation: 'Major distribution shift — high risk of degraded accuracy.', action: 'Audit data pipeline / sensor calibration and retrain champion model.' },
          { metric: '2-Sample KS Test', range: 'p < 0.05 AND D >= 0.10', level: 'warn', interpretation: 'Statistically AND practically meaningful CDF shift.', action: 'Investigate both location (mean) and spread (variance) changes.' },
        ],
      },
    ],
    decisionMatrix: {
      title: 'Studio 11 Drift Action Decision Matrix',
      rows: [
        { application: 'Routine monthly production health audit', dataCondition: 'Reference batch vs new production batch', recommended: 'PSI + 2-Sample KS Drift Monitor', alternative: 'Studio 14 Live Actual-vs-Predicted Error Monitor', rationale: 'Detects covariate shift even BEFORE ground-truth actual labels arrive.' },
        { application: 'Versioned model release handoff', dataCondition: 'Validated champion after Studio 07 Responsible AI', recommended: 'Model Registry (Staging -> Production)', alternative: 'Export portable .omniforge project archive', rationale: 'Maintains formal audit trail of model version, metrics, and stage.' },
      ],
    },
  },

  // =========================================================================
  // CHAPTER 12 — STUDIO 12
  // =========================================================================
  {
    studioId: '12',
    chapterNum: 'Chapter 12',
    title: 'Executive Reporting & Multi-Format Export Center',
    subtitle: 'Customizable 14-section report compiler with live previews and instant export to PDF, XLSX, DOCX, HTML, Markdown, CSV, JSON, and LaTeX.',
    overview:
      'Analytics only creates impact when communicated clearly to executives, engineers, and auditors. Studio 12 automatically gathers every artifact produced across Studios 01–14 — Intake Defect Audit, Cleaning & Deleted-Records Exclusions, Clustering/Anomalies, Statistical Diagnostics, Ranked Mathematical Equations, Feature Selection, AutoML/Non-Parametric/DL Leaderboards, SHAP/Fairness Model Cards, Forecasting/Causal Decisions, and Studio 14 Live Supervisory Telemetry — into one cohesive, customizable report.',
    realWorldApplications: [
      'Delivering polished executive PDF or Word (DOCX) decision memos to leadership.',
      'Handing auditable multi-sheet Excel (XLSX) workbooks containing every table, metric, and exclusion log to QA/finance teams.',
      'Publishing reproducible LaTeX / Markdown scientific appendices with exact mathematical equations.',
    ],
    stepByStep: [
      'Enter your custom Report Title and Author / Organization name.',
      'Select which of the 14 report sections to include (or click "Select All 14" / "Executive Preset").',
      'Click "Compile Customizable Report" to preview every section’s narrative, key highlights, tables, and embedded charts (with Zoom & Pop-Out).',
      'Download in any of 8 formats: PDF, XLSX, DOCX, HTML, Markdown, CSV, JSON, or LaTeX.',
    ],
    provenanceRule: 'Read-only. Compiles live artifacts from all completed studios; running more studios before compiling enriches more report chapters automatically.',
    methods: [
      {
        name: '14-Section Auditable Synthesis Engine & Lineage Completeness',
        paradigm: 'Multi-Format Publication Compiler',
        whatItDoes:
          'Maps live workspace lineage, statistical tables, ranked mathematical equations, matplotlib/vector figures, and supervisory alarm logs into unified structured report sections exportable across 8 document formats while verifying end-to-end evidence coverage.',
        formula: {
          equation: 'Completeness = |S_populated ∩ S_selected| / max(|S_selected|, 1)   |   Lift_reported = (Score_champ − Score_base) / max(|Score_base|, ε)',
          symbols: 'S_selected = user-selected report sections (1..14); S_populated = sections backed by executed studio artifacts; Score_champ / Score_base = audited validation metrics exported verbatim into tables and executive narratives.',
        },
        whenToUse: 'Final step after completing your desired workflow stages (01 -> 03 -> 06 -> 07 -> 14).',
        recommendedParams: 'Use "Executive Preset" for board summaries (PDF/DOCX) and "Select All 14" for complete engineering/regulatory audit packages (PDF + XLSX).',
        pitfalls: 'Compile the report AFTER running Studio 03, 06, 07, and 14 so those chapters populate with live verified findings rather than empty placeholders.',
      },
    ],
    thresholdTables: [
      {
        title: 'Recommended Export Format by Audience',
        rows: [
          { metric: 'PDF / DOCX / HTML', range: 'Executive & Regulatory Sign-Off', level: 'good', interpretation: 'Formatted narrative + charts + ranked equations ready for presentation.', action: 'Use PDF for immutable archival; DOCX for editable team review.' },
          { metric: 'XLSX / CSV', range: 'Engineering, Finance & Audit Analysis', level: 'good', interpretation: 'Multi-sheet raw tables, coefficients, leaderboards, and deleted-row logs.', action: 'Share XLSX when downstream teams need pivot tables or audits.' },
          { metric: 'LaTeX / Markdown / JSON', range: 'Scientific Papers & Automated Pipelines', level: 'info', interpretation: 'Direct mathematical typesetting (LaTeX) or machine-readable integration (JSON).', action: 'Use LaTeX for academic/technical publications.' },
        ],
      },
    ],
    decisionMatrix: {
      title: 'Studio 12 Export Format Selector',
      rows: [
        { application: 'Boardroom / management final presentation', dataCondition: 'High-level narrative + charts + KPIs', recommended: 'Executive Preset -> PDF or DOCX', alternative: 'Interactive HTML', rationale: 'Concise, visually polished, zero coding knowledge needed to read.' },
        { application: 'Compliance / forensic QA audit package', dataCondition: 'Full lineage, exclusions, equations, fairness, alarms', recommended: 'All 14 Sections -> PDF + XLSX', alternative: 'JSON + Portable .omniforge Archive', rationale: 'Provides both human-readable audit record and raw spreadsheet tables.' },
      ],
    },
  },

  // =========================================================================
  // CHAPTER 13 — STUDIO 13
  // =========================================================================
  {
    studioId: '13',
    chapterNum: 'Chapter 13',
    title: 'Context-Aware AI Analytics Copilot',
    subtitle: 'Workspace-grounded analytical advisor that inspects your live data profile, defects, diagnostics, active model metrics, and alarms.',
    overview:
      'Studio 13 acts as your built-in senior data scientist inside the workspace. Rather than giving generic textbook platitudes, it reads your actual live session state — row/column counts, intake defect grade, Split Lock status, cleaning history, baseline vs AutoML scores, active equations, and Studio 14 alarm counts — to explain results and recommend concrete next actions.',
    realWorldApplications: [
      'Asking "What should I do next based on my current workspace state?"',
      'Interpreting diagnostic warnings (VIF > 10, skewed features, negative baseline lift, or high anomaly rates).',
      'Getting tailored guidance on choosing between Non-Parametric GAM, AutoML, or Deep Learning for your active dataset.',
    ],
    stepByStep: [
      'Open Studio 13 anytime during your workflow (your conversation history persists automatically across studio switches).',
      'Click one of the suggested quick-prompt chips OR type any specific analytical question about your dataset or results.',
      'Apply recommended actions directly in the referenced studio.',
    ],
    provenanceRule: 'Read-only on Active Data. Full conversation transcript persists in s13_chat cache.',
    methods: [
      {
        name: 'Workspace-Grounded Analytical Reasoning & Action Prioritization',
        paradigm: 'Interactive Decision Assistant',
        whatItDoes:
          'Synthesizes live metadata from Intake Audit, Provenance Status, Split Lock, Studio 03 Diagnostics, Studio 06 Models, and Studio 14 Supervisory telemetry to produce specific, factually grounded guidance ranked by urgency and workflow readiness.',
        formula: {
          equation: 'Priority(a) = w_sev · Severity(defect_a) + w_dep · Blocker(stage_a) + w_lift · ExpectedGain(a)',
          symbols: 'Severity ∈ [0,1] from Intake Audit/VIF/Alarm telemetry; Blocker ∈ {0,1} flags missing prerequisites (e.g., Split Lock before Cleaning or Cleaning before AutoML); ExpectedGain scores diagnostic lift potential.',
        },
        whenToUse: 'Whenever you want rapid interpretation of live results or a recommended next step in your CRISP-DM workflow.',
        recommendedParams: 'Ask specific questions referencing your current stage (e.g., "Summarize my active dataset quality and best model").',
        pitfalls: 'Always verify critical scientific/regulatory decisions directly in the underlying tables of Studios 03, 04, 07, and 14.',
      },
    ],
    thresholdTables: [
      {
        title: 'Best Prompts for Maximum Copilot Value',
        rows: [
          { metric: 'Workflow Guidance', range: '"What is my next recommended step?"', level: 'info', interpretation: 'Checks completion LEDs, Split Lock, and provenance.', action: 'Follow prioritized checklist.' },
          { metric: 'Model Diagnosis', range: '"Explain my champion model vs baseline"', level: 'info', interpretation: 'Compares active model metrics & lift vs Studio 03 baseline.', action: 'Verify positive lift before deploying.' },
        ],
      },
    ],
    decisionMatrix: {
      title: 'When to Use Studio 13 Copilot vs. Help Handbook',
      rows: [
        { application: 'Interpret YOUR live dataset metrics, defects & active model', dataCondition: 'Dataset loaded / models trained', recommended: 'Studio 13 AI Analytics Copilot', alternative: 'Inspect Studio 03 / 06 / 12 summaries', rationale: 'Reads your live session numbers directly.' },
        { application: 'Look up exact formulas, numeric thresholds & which statistical test to pick', dataCondition: 'Anytime (even before loading data)', recommended: 'Help & Guide Handbook (? button)', alternative: 'docs/user_guide.md', rationale: 'Complete encyclopedia of equations, decision tables, and numeric cutoffs.' },
      ],
    },
  },

  // =========================================================================
  // CHAPTER 14 — STUDIO 14
  // =========================================================================
  {
    studioId: '14',
    chapterNum: 'Chapter 14',
    title: 'Live Prediction & Engineering Supervisory Control Room',
    subtitle: 'Real-time operational monitoring comparing Champion vs. Baseline predictions against live actuals from Excel, SQL, MS Access, Telemetry Sim, or Single Probe with SPC alerting.',
    overview:
      'Studio 14 bridges analytics and live plant/business operations. Select any trained Champion Model (AutoML, Non-Parametric Analytical GAM/Kernel, or Deep Learning) alongside a Reference Baseline Model, define strict engineering tolerance limits (% Error, Absolute Error, Sigma-Z SPC, Baseline Ratio, Batch Breach Rate), and feed incoming operational records from Excel/CSV, SQL Databases, MS Access, Simulated Fault Streams, or Single Sensor Readings. OmniForge scores every record live, plots Actual vs Predicted vs Baseline with Upper/Lower Control Limits, raises prioritized CRITICAL/WARNING alarms, and maintains a longitudinal audit history.',
    realWorldApplications: [
      'Industrial SCADA / gas turbine / compressor / manufacturing digital-twin anomaly and performance monitoring.',
      'Real-time quality assurance comparing predicted vs measured batch yield/purity/strength against engineering tolerances.',
      'Financial pricing, energy load, or SLA latency operational alerting when live actuals breach expected model corridors.',
      'Live A/B shadow testing of a new Champion model versus an established Baseline model in production.',
    ],
    stepByStep: [
      'Step 1 (Model Selection): Select your Primary Champion Model and optional Reference Baseline Model (or click "Quick-Train Baseline" if starting straight from data).',
      'Step 2 (Tolerance Envelope): Configure Max % Error, Max Absolute Error, Max Sigma-Z (SPC), Baseline Ratio [Min, Max], Warning Fraction (early pre-alarm), and Max Batch Breach Rate %.',
      'Step 3 (Live Data Connector): Choose Excel/CSV upload, SQL Database table/query, MS Access table, Live Telemetry Simulator (with configurable fault injection % to test alarms!), or Single Reading Probe.',
      'Step 4 (Control Room Review): Inspect System Status LED (HEALTHY / WARNING / CRITICAL_ALARM), KPI cards, Actual-vs-Predicted SPC Control Chart, prioritized Alarm Log (filter CRITICAL/WARNING), Full Scored Telemetry table, and download scored CSV/XLSX.',
    ],
    provenanceRule: 'Read-only on training data. Every live evaluation batch is cached in s14_supervisory, appended to Longitudinal Supervisory History, reflected in the topbar Monitor LED, and embedded into Studio 12 Executive Reports.',
    methods: [
      {
        name: 'Multi-Rule Engineering Tolerance & SPC Control Envelope',
        paradigm: 'Real-Time Statistical Process Control (SPC)',
        whatItDoes:
          'Evaluates every incoming operational reading simultaneously against four complementary mathematical alarm rules: scale-free Percentage Error, fixed physical-unit Absolute Error, standardized Shewhart Residual Z-Score (how many training standard deviations σ the actual reading deviates from prediction), and Baseline Ratio drift. A Warning triggers early at warning_fraction × limit so operators intervene BEFORE critical failure.',
        formula: {
          equation: '%Err = 100·|y − ŷ| / max(|y|, ε)   |   AbsErr = |y − ŷ|   |   Z_SPC = |y − ŷ| / σ_train   |   BaseRatio = y / ŷ_base',
          symbols: 'y = live measured actual; ŷ = champion prediction; ŷ_base = baseline prediction; σ_train = training residual standard deviation; UCL/LCL = ŷ ± Z_max · σ_train.',
        },
        whenToUse: 'Use % Error for dimensionless KPIs (efficiency, load, cost); Abs Error for hard physical specs (±2 °C, ±5 bar); Sigma-Z (3σ) for universal Shewhart SPC control charts; Baseline Ratio to catch physical regime shifts.',
        recommendedParams: 'Warning Fraction = 0.75 (fires amber warning at 75% of critical limit); Batch Breach Rate = 5% for batch-level CRITICAL status.',
        pitfalls: '% Error explodes near y ≈ 0 — always pair % Error with Absolute Error or Sigma-Z when measurements can cross zero.',
      },
      {
        name: 'Champion vs. Baseline Live Shadow Benchmarking',
        paradigm: 'Operational Model Verification',
        whatItDoes:
          'Scores every live record with BOTH your Champion Model and a simple transparent Baseline Model simultaneously, measuring whether the Champion still maintains superior accuracy on live unseen operational data.',
        formula: {
          equation: 'Live_Advantage = MAE(y, ŷ_baseline) − MAE(y, ŷ_champion)',
          symbols: 'Positive Live_Advantage confirms Champion outperforms Baseline in live operations; negative values warn of model drift or overfitting.',
        },
        whenToUse: 'Always select a Reference Baseline Model in Studio 14 to continuously verify Champion superiority in the field.',
        recommendedParams: 'Pair your Studio 06 Champion (AutoML or B-Spline GAM) against Studio 03 Ridge/Linear Baseline.',
        pitfalls: 'If Baseline beats Champion on live feeds across multiple batches, check Studio 11 PSI Drift and retrain immediately.',
      },
    ],
    thresholdTables: [
      {
        title: 'Recommended Studio 14 Supervisory Tolerance Presets by Industry',
        subtitle: 'Starting guidelines — always align final limits with your plant SOP, safety factor, or SLA contract.',
        rows: [
          { metric: 'Tight Precision Engineering / Aerospace / Lab', range: 'Max %Err = 2–5% | Z_SPC = 2.5σ | Warning = 0.70 | Breach = 2%', level: 'bad', interpretation: 'Ultra-sensitive early detection for high-cost precision equipment.', action: 'Immediate operator triage on any Warning or single Critical breach.' },
          { metric: 'Standard Industrial / Energy / SCADA (Default)', range: 'Max %Err = 10% | Z_SPC = 3.0σ | BaseRatio = [0.85, 1.15] | Warning = 0.75 | Breach = 5%', level: 'good', interpretation: 'Balanced Shewhart 3σ control envelope (< 0.27% false-alarm rate under normal noise).', action: 'Investigate Warnings proactively; escalate Critical breaches to engineering.' },
          { metric: 'High-Noise Commercial / Demand / Financial', range: 'Max %Err = 15–25% | Z_SPC = 3.5σ | BaseRatio = [0.75, 1.25] | Breach = 10%', level: 'info', interpretation: 'Accommodates high natural volatility without alert fatigue.', action: 'Track longitudinal breach trend across batches for structural shift.' },
          { metric: 'Classification Operational Accuracy Floor', range: 'Min Accuracy >= 85% (Strict >= 92%)', level: 'warn', interpretation: 'Triggers CRITICAL_ALARM if live batch classification accuracy falls below SLA.', action: 'Audit confusion breakdown and Studio 11 covariate drift.' },
        ],
      },
    ],
    decisionMatrix: {
      title: 'Studio 14 Live Source & Tolerance Rule Selector',
      rows: [
        { application: 'Score daily/hourly plant logs or QA laboratory spreadsheets', dataCondition: 'Structured Excel (.xlsx) or CSV drop', recommended: 'Tab: Excel (.xlsx) / CSV + %Err & Z_SPC', alternative: 'SQL Database Connector', rationale: 'Instant drag-and-drop batch telemetry auditing with exportable scored XLSX.' },
        { application: 'Direct automated SCADA / ERP / Historian database polling', dataCondition: 'Live SQLite / PostgreSQL / MySQL / MSSQL / MS Access', recommended: 'Tab: SQL Database or MS Access', alternative: 'Scheduled CSV export ingestion', rationale: 'Queries live operational tables directly with zero manual file exports.' },
        { application: 'Test alarm thresholds and operator workflows before go-live', dataCondition: 'Pre-deployment verification', recommended: 'Tab: Live Telemetry Simulator (inject 5–15% faults)', alternative: 'Historical holdout CSV replay', rationale: 'Safely verifies CRITICAL/WARNING alarm firing and SPC limit visualization.' },
        { application: 'Interactive what-if probe by a control-room engineer', dataCondition: '1 manual reading + optional measured actual', recommended: 'Tab: Single Reading Probe', alternative: 'Studio 06 Tab 4 Single Prediction', rationale: 'Instant residual check against all active tolerance rules.' },
      ],
    },
  },
];
