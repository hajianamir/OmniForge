import type { StudioDeepSupplement } from './handbookTypes';

/**
 * Exhaustive Textbook-Depth Supplements for Studios 08–14.
 * Includes deep theory, worked numerical examples, complete UI controls guides,
 * diagnostic troubleshooting playbooks, and full Studio 14 Root-Cause Attribution mathematics.
 */
export const DEEP_SUPPLEMENTS_PART_2: Record<'08' | '09' | '10' | '11' | '12' | '13' | '14', StudioDeepSupplement> = {
  // =========================================================================
  // STUDIO 08 — DEEP TEXTBOOK SUPPLEMENT
  // =========================================================================
  '08': {
    deepTheory: [
      {
        heading: '8.1 Time-Series Stationarity, Walk-Forward Evaluation & Why MASE Beats MAPE',
        paragraphs: [
          'Standard random K-Fold cross-validation is strictly illegal for time-series forecasting because training on t = 105 to predict t = 102 leaks future macroeconomic shocks and seasonal trends backwards in time. Studio 08 evaluates all forecasting models — Classical (Holt-Winters Exponential Smoothing, ARIMA/SARIMA) and Deep Neural (LSTM, Temporal Convolutional Network, PatchTST, N-BEATS) — exclusively on strictly chronological holdout horizons.',
          'Crucially, never judge intermittent or near-zero demand series by MAPE (100 · |y − ŷ| / |y|): dividing by y = 0 produces infinite error, and MAPE asymmetrically penalizes over-forecasts far more heavily than under-forecasts! Instead, OmniForge reports MASE (Mean Absolute Scaled Error, Hyndman & Koehler 2006) alongside RMSE/MAE: MASE divides your forecast MAE by the in-sample Naive Random-Walk MAE (|y_t − y_{t-1}|). MASE < 1.0 mathematically proves your forecaster beats "predicting yesterday’s value", regardless of units or zero entries.',
        ],
      },
      {
        heading: '8.2 Causal Inference (Potential Outcomes & AIPW Doubly Robust Estimation) vs. Raw Correlation',
        paragraphs: [
          'Predictive models estimate E[Y | X, T = 1] — correlation. Decision makers need the Causal Average Treatment Effect ATE = E[Y(1) − Y(0)]: what happens if we INTERVENE to force T = 1 on everyone vs. T = 0? In observational data, treated and untreated units differ systematically on confounders X (e.g., sicker patients receive aggressive therapy more often, making naive averages falsely imply the therapy hurts survival!).',
          'OmniForge implements Augmented Inverse Probability Weighting (AIPW / Doubly Robust Estimator) alongside Propensity Score Matching (PSM) and Difference-in-Differences (DiD). AIPW combines an outcome regression model μ̂(t, x) with a propensity model ê(x) = P(T=1 | X=x). Its celebrated Double-Robustness theorem guarantees consistency if EITHER the outcome model OR the propensity score model is correctly specified — providing rock-solid causal protection against confounding bias.',
        ],
      },
      {
        heading: '8.3 Constrained Optimization & Stochastic Monte Carlo Risk Propagation',
        paragraphs: [
          'Once a model predicts y = f(x) accurately, engineers ask the inverse prescriptive question: "What controllable input vector x* maximizes yield (or minimizes cost/emissions) subject to physical, budget, and safety bounds g(x) <= 0?" Studio 08 provides Scipy SLSQP/L-BFGS-B for smooth continuous objectives, Linear Programming (HiGHS) for exact global optima on linear systems, and Differential Evolution for rugged non-convex landscapes.',
          'Because real-world parameters fluctuate stochastically around setpoints, Tab 4 runs Monte Carlo Simulation (N = 1,000–50,000 draws) propagating input probability distributions (Normal, LogNormal, Triangular, Uniform) through your engineering/financial equation to deliver exact P5 / P50 / P95 confidence bands and failure-threshold exceedance probabilities.',
        ],
      },
    ],
    workedExamples: [
      {
        title: 'Worked Example 8A: Computing MASE to Verify Real Forecasting Skill',
        scenario: 'Weekly spare-parts demand over the past 5 weeks was [100, 108, 102, 114, 110]. Over a 3-week holdout horizon, your PatchTST model achieves MAE = 4.2 units.',
        stepByStepCalculation: [
          'Step 1 — Compute In-Sample Naive Random-Walk Errors |y_t − y_{t-1}|: |108−100|=8, |102−108|=6, |114−102|=12, |110−114|=4.',
          'Step 2 — In-Sample Naive MAE = (8 + 6 + 12 + 4) / 4 = 30 / 4 = 7.50 units.',
          'Step 3 — Compute MASE = Model_MAE / Naive_MAE = 4.20 / 7.50 = 0.56.',
        ],
        numericalConclusion: 'MASE = 0.56 (< 0.70 Excellent!) proves your forecaster cuts error by 44% relative to the naive persistence baseline.',
        engineeringAction: 'Deploy PatchTST/N-BEATS with 95% conformal prediction intervals and schedule rolling retraining whenever live MASE climbs above 0.85.',
      },
      {
        title: 'Worked Example 8B: Simpson’s Paradox in Observational Treatment Evaluation',
        scenario: 'New preventive maintenance protocol (T=1) was applied mostly to older high-failure machines (60% baseline failure), while healthy new machines stayed on T=0 (10% baseline failure). Naive difference shows higher failures under T=1!',
        stepByStepCalculation: [
          'Naive raw comparison: E[Failure | T=1] − E[Failure | T=0] = +12% (FALSELY implies maintenance causes failure!).',
          'Propensity-adjusted AIPW controlling for machine_age & load_hours compares old treated machines (45% fail) vs old untreated (60% fail -> −15%) and new treated (6% fail) vs new untreated (10% fail -> −4%).',
          'Weighted ATE = −9.5% failure reduction (95% CI: [−14.2%, −4.8%], p < 0.001).',
        ],
        numericalConclusion: 'AIPW reverses the spurious +12% observational correlation to reveal the true −9.5% causal failure reduction.',
        engineeringAction: 'ALWAYS include all known confounders (age, severity, workload) in Studio 08 Causal Inference and inspect propensity overlap (0.05 < e(x) < 0.95) before trusting ATE.',
      },
    ],
    uiControlsGuide: [
      {
        control: 'Forecast Horizon (Steps Ahead) & Architecture',
        defaultValue: 'Horizon = 12, Architecture = Auto / PatchTST / Holt-Winters',
        whenIncreased: 'Longer horizons widen uncertainty intervals exponentially; prefer seasonal/neural architectures with explicit trend damping.',
        whenDecreased: 'Short 1–6 step horizons allow tighter precision and simpler AR/ETS models.',
        expertGuidance: 'Ensure at least 3× the seasonal cycle length of historical points before fitting seasonal or PatchTST models.',
      },
      {
        control: 'Causal Treatment Column & Method (AIPW / PSM / DiD)',
        defaultValue: 'AIPW (Doubly Robust)',
        whenIncreased: 'AIPW handles continuous confounders cleanly with minimum variance.',
        whenDecreased: 'DiD requires explicit pre/post time periods and parallel pre-treatment trends.',
        expertGuidance: 'Verify treatment column is binary (0/1) and verify common support (propensity scores strictly away from 0 and 1).',
      },
      {
        control: 'Monte Carlo Iterations (N)',
        defaultValue: '2,000–5,000',
        whenIncreased: '10,000–50,000 cuts Monte Carlo standard error (σ/√N) for ultra-precise P1/P99 extreme tail probabilities.',
        whenDecreased: '500–1,000 runs instantaneously for rapid sensitivity testing.',
        expertGuidance: 'Use N >= 5,000 whenever computing 95%/99% failure probabilities or CVaR.',
      },
    ],
    troubleshootingPlaybook: [
      {
        symptom: 'Forecast produces a flat horizontal line or MASE > 1.1',
        mathematicalCause: 'Non-stationary trend/seasonality was not captured, input time column wasn’t sorted chronologically, or series length is too short (< 30 steps) for deep architectures.',
        exactRemediationSteps: 'Sort dataset chronologically, switch to Holt-Winters / ARIMA for short series (< 100 points), and inspect Studio 04 autocorrelation / stationarity.',
      },
      {
        symptom: 'AIPW causal estimate shows massive standard error or extreme variance',
        mathematicalCause: 'Positivity violation — some confounder strata have propensity e(x) near 0.00 or 1.00, making inverse weights 1/e(x) explode.',
        exactRemediationSteps: 'Trim extreme propensity scores (keep 0.05 <= e(x) <= 0.95) or exclude variables that perfectly predict treatment assignment (instrumental variables, not confounders).',
      },
    ],
  },

  // =========================================================================
  // STUDIO 09 — DEEP TEXTBOOK SUPPLEMENT
  // =========================================================================
  '09': {
    deepTheory: [
      {
        heading: '9.1 Multi-Modal Analytics: Vision, NLP, Graph Link Prediction & Right-Censored Survival Analysis',
        paragraphs: [
          'Modern enterprise datasets extend far beyond flat numeric spreadsheets. Studio 09 unifies four specialized modalities under one consistent evaluation interface: (1) Computer Vision computes multi-scale visual signatures (color histograms, Sobel edge energy, HOG gradients, Laplacian focus sharpness, plus optional timm/YOLO/OpenCLIP foundation embeddings) for automated defect inspection and visual clustering; (2) Natural Language Processing runs TF-IDF n-grams, SVD topic discovery, lexical/sentiment polarity, named-entity extraction, and optional HuggingFace Transformer embeddings.',
          '(3) Graph Analytics constructs relational networks directly from source-target edge lists or feature similarity graphs, computing PageRank centrality, Betweenness bottleneck hubs, Louvain/greedy modularity communities, and Adamic-Adar / Jaccard / GNN link prediction scores to expose fraud rings, supply-chain choke points, and hidden influence clusters.',
          '(4) Survival & Reliability Analysis solves a critical statistical flaw of ordinary regression: Right-Censoring! When equipment is still running cleanly at inspection time t = 5,000 hrs (event = 0), treating 5,000 as a failure time severely underestimates lifespan, while dropping still-running units throws away your healthiest assets! Kaplan-Meier non-parametric estimators and Cox Proportional Hazards regression rigorously incorporate BOTH failed AND censored observations into exact survival probabilities S(t) = P(T > t) and Hazard Ratios exp(β_j).',
        ],
      },
    ],
    workedExamples: [
      {
        title: 'Worked Example 9A: Kaplan-Meier Survival Curve with Right-Censored Equipment',
        scenario: '10 pumps start at t = 0. Pump A fails at 100 hrs. Pump B is removed cleanly for relocation (censored) at 150 hrs. Pump C fails at 200 hrs.',
        stepByStepCalculation: [
          'At t = 100 hrs: n_at_risk = 10, failures d = 1 -> S(100) = 1.0 × (1 − 1/10) = 0.900 (90.0% survival).',
          'At t = 150 hrs: Pump B is censored (NOT a failure!). Survival stays S(150) = 0.900, and risk set shrinks by 1 for future times.',
          'At t = 200 hrs: n_at_risk = 8 (10 − 1 failed − 1 censored), failures d = 1 -> S(200) = 0.900 × (1 − 1/8) = 0.900 × 0.875 = 0.7875 (78.8% survival).',
        ],
        numericalConclusion: 'True survival at 200 hrs is 78.8% (properly accounting for censoring; naive failure rate 2/10 = 80% overestimates survival by ignoring the lost at-risk unit!).',
        engineeringAction: 'Check Cox Proportional Hazards Harrell’s Concordance Index (C-index >= 0.70) and inspect which covariates have Hazard Ratio exp(β) > 1.5 (accelerated wear drivers).',
      },
    ],
    uiControlsGuide: [
      {
        control: 'Modality Tabs (Vision / NLP / Graph / Survival)',
        defaultValue: 'Auto-detected from dataset columns',
        whenIncreased: 'Switch tabs to analyze image files, text comment columns, source->target entity columns, or duration+event pairs.',
        whenDecreased: 'If no specialized columns exist, Survival can run on any positive duration + binary status pair; Graph can build similarity networks from numeric rows.',
        expertGuidance: 'In Survival Analysis, verify duration > 0 and event is strictly binary (1 = failure/churn/death occurred, 0 = right-censored / still active).',
      },
    ],
    troubleshootingPlaybook: [
      {
        symptom: 'Survival Cox PH reports Concordance Index near 0.50 or convergence warning',
        mathematicalCause: 'Event column is reversed (0/1 swapped), duration contains zeros/negatives, or near-perfect separation exists on a sparse categorical predictor.',
        exactRemediationSteps: 'Verify 1 = event occurred and 0 = censored, filter duration > 0, and exclude ultra-sparse/collinear predictors via Topbar Exclusions.',
      },
    ],
  },

  // =========================================================================
  // STUDIO 10 — DEEP TEXTBOOK SUPPLEMENT
  // =========================================================================
  '10': {
    deepTheory: [
      {
        heading: '10.1 Directed Acyclic Graphs (DAGs), Topological Execution & Reproducible Engineering Pipelines',
        paragraphs: [
          'Ad-hoc notebook experiments fail audit and production handoff because steps get run out of order or silently depend on mutable hidden state. Studio 10 formalizes your entire analytical workflow as a strict Directed Acyclic Graph G = (V, E), where every node v ∈ V is a typed OmniForge operator (Ingest, Clean, Cluster, Diagnostics, Feature Engineering, AutoML, Explainability, Forecasting, Monitoring, Report) and every directed edge (u -> v) encodes an explicit data/artifact dependency.',
          'Before executing a single step, OmniForge validates DAG acyclicity via Kahn’s Topological Sort algorithm O(|V| + |E|). If any accidental feedback loop exists (e.g., Node C feeds back into Node A), the validator rejects execution immediately and highlights the exact cycle. Valid DAGs execute deterministically in topological order, capturing per-node latency, status, row/column shape contracts, and immutable JSON/YAML pipeline blueprints exportable across environments.',
        ],
      },
    ],
    workedExamples: [
      {
        title: 'Worked Example 10A: Designing a Zero-Leakage End-to-End Industrial DAG',
        scenario: 'Automate monthly turbine predictive maintenance refresh from raw CSV ingestion to signed Executive PDF & Supervisory readiness.',
        stepByStepCalculation: [
          'Topological Order: [1: Ingest & Intake Audit] -> [2: Split-Locked Cleaning] -> [3: Anomaly Screening] -> [4: Baseline Diagnostics] -> [5: Consensus Feature Selection] -> [6: AutoML Champion] -> [7: SHAP & Model Card] -> [8: Executive Report Export].',
          'Execution check: 0 cycles (in-degree drops cleanly to 0 at every step); every downstream node consumes strictly the validated artifact of its parent.',
        ],
        numericalConclusion: 'Single-click reproducible execution guarantees identical preprocessing, identical split boundaries, and auditable latency per stage.',
        engineeringAction: 'Export the DAG JSON blueprint and save the .omniforge project archive so colleagues reproduce every metric identically.',
      },
    ],
    uiControlsGuide: [
      {
        control: 'DAG Templates & Node Builder',
        defaultValue: 'Standard CRISP-DM Full Pipeline',
        whenIncreased: 'Add specialized branches (Forecasting, Survival, Responsible AI) in parallel off the Clean/Feature node.',
        whenDecreased: 'Prune optional branches for lightweight rapid retraining runs.',
        expertGuidance: 'Always verify the Topological Order preview before clicking Run Pipeline to ensure Cleaning precedes Feature Engineering and Modeling.',
      },
    ],
    troubleshootingPlaybook: [
      {
        symptom: 'DAG validator rejects graph with "Cycle Detected" error',
        mathematicalCause: 'An edge connects a downstream stage back to an upstream ancestor (violating DAG acyclicity).',
        exactRemediationSteps: 'Remove the backward edge so information flows strictly left-to-right from raw ingestion to modeling and reporting.',
      },
    ],
  },

  // =========================================================================
  // STUDIO 11 — DEEP TEXTBOOK SUPPLEMENT
  // =========================================================================
  '11': {
    deepTheory: [
      {
        heading: '11.1 Population Stability Index (PSI), Covariate Shift vs. Concept Drift & Champion-Challenger Governance',
        paragraphs: [
          'Every deployed machine learning model decays over time because the real world is non-stationary. MLOps distinguishes two fundamentally different failure modes: (1) Covariate / Data Drift — the distribution of input features P(X) shifts (e.g., hotter summer ambient temperatures, new supplier raw-material grades, or a miscalibrated sensor) even though the true physical mapping P(Y | X) remains valid; (2) Concept Drift — the relationship P(Y | X) itself changes (e.g., mechanical wear changes the efficiency curve, or customer behavior permanently shifts after a policy/price change).',
          'Studio 11 quantifies input feature drift using BOTH Population Stability Index (PSI = Σ (A_i − E_i)·ln(A_i / E_i)) and Two-Sample Kolmogorov-Smirnov (KS) distance. Industry standard Basel/MLOps bands classify PSI < 0.10 as Stable (Green), 0.10 <= PSI < 0.25 as Moderate Drift (Investigate / Watch), and PSI >= 0.25 as Significant Drift (Retrain immediately!). Alongside drift monitoring, Studio 11 provides a versioned Model Registry (Staging -> Champion -> Archived), Target-Free Future Batch Scoring (uploading unlabeled future CSV/XLSX files to generate downloadable predictions + uncertainty), and Canary Champion-vs-Challenger deployment gates.',
        ],
      },
    ],
    workedExamples: [
      {
        title: 'Worked Example 11A: Exact Calculation of Population Stability Index (PSI) Across 3 Operating Bins',
        scenario: 'Training reference distribution across Low / Medium / High load bins: E = [0.40, 0.40, 0.20]. New production month shifts sharply toward High load: A = [0.15, 0.35, 0.50].',
        stepByStepCalculation: [
          'Bin 1 (Low): (0.15 − 0.40) × ln(0.15 / 0.40) = (−0.25) × (−0.9808) = +0.2452.',
          'Bin 2 (Medium): (0.35 − 0.40) × ln(0.35 / 0.40) = (−0.05) × (−0.1335) = +0.0067.',
          'Bin 3 (High): (0.50 − 0.20) × ln(0.50 / 0.20) = (+0.30) × (+0.9163) = +0.2749.',
          'Total PSI = 0.2452 + 0.0067 + 0.2749 = 0.5268 >> 0.25 (CRITICAL SHIFT!).',
        ],
        numericalConclusion: 'PSI = 0.527 far exceeds the 0.25 critical threshold — production load has shifted heavily into the high-load tail.',
        engineeringAction: 'Do NOT blindly trust extrapolations; inspect high-load prediction errors in Studio 14 Supervisory Monitoring and schedule Champion retraining incorporating recent high-load data.',
      },
    ],
    uiControlsGuide: [
      {
        control: 'Future Batch Scoring Uploader (Target-Free CSV/XLSX)',
        defaultValue: 'Uses Active Champion Model',
        whenIncreased: 'Upload any unlabeled future file sharing the training predictor columns to append predicted_target + confidence/margin columns.',
        whenDecreased: 'Compare scored future output against training baseline averages to detect silent distribution drift.',
        expertGuidance: 'Verify uploaded future columns match exact training feature names; missing optional columns trigger automatic safe imputation with audit warnings.',
      },
      {
        control: 'Model Registry Lifecycle Promotion',
        defaultValue: 'Registered -> Champion',
        whenIncreased: 'Promote a newer Challenger only after verifying statistically superior validation metrics AND passing Studio 07 Fairness/Explainability checks.',
        whenDecreased: 'Keep prior Champion archived for instant 1-click rollback if live Studio 14 error spikes.',
        expertGuidance: 'Never delete historical model versions — retain full lineage for regulatory audit.',
      },
    ],
    troubleshootingPlaybook: [
      {
        symptom: 'Future Batch Scoring warns of schema mismatch or missing features',
        mathematicalCause: 'Incoming file renamed columns (e.g., "Temp_C" vs "temp_c"), has unexpected text strings inside numeric columns, or omits required predictors.',
        exactRemediationSteps: 'Standardize column headers to match Studio 01 Active Data schema, verify numeric units match training units (e.g., °C vs °F!), and re-upload.',
      },
    ],
  },

  // =========================================================================
  // STUDIO 12 — DEEP TEXTBOOK SUPPLEMENT
  // =========================================================================
  '12': {
    deepTheory: [
      {
        heading: '12.1 Audit-Ready Analytical Governance: Composing Multi-Format 14-Section Executive Deliverables',
        paragraphs: [
          'Analytical insights create zero operational impact until decision makers, engineers, and regulators can inspect, verify, and archive them offline. Studio 12 dynamically harvests live verified results across all 14 studios — Intake Defect Audits, Forensic Deleted-Records Ledgers, Clustering Profiles, Statistical Hypothesis Tables, Ranked Governing Mathematical Equations, AutoML Leaderboards, SHAP Explanations, Fairness Model Cards, Forecasting Benchmarks, Future Predictions, and Studio 14 Supervisory Alarms + Root-Cause Playbooks.',
          'Using the 14 customizable section checkboxes, you can tailor concise 3-section Board Summaries OR full 14-section ISO/regulatory audit dossiers and export simultaneously across 9 enterprise formats: publication-grade PDF (ReportLab vector layout + embedded figures), multi-sheet styled Excel Workbook (.xlsx), editable Microsoft Word (.docx), standalone interactive HTML, Markdown, JSON, YAML, LaTeX (.tex), and raw CSV data bundles.',
        ],
      },
    ],
    workedExamples: [
      {
        title: 'Worked Example 12A: Building Role-Specific Report Packs',
        scenario: 'You need both an Executive Summary for management and a Technical Compliance Dossier for quality auditors from the same completed OmniForge session.',
        stepByStepCalculation: [
          'Pack 1 (Executive Brief): Toggle ON Sections 1 (Executive Overview), 6 (AutoML Champion & Equation), 12 (Future Predictions), 14 (Supervisory Status & Fixes) -> Export as PDF + DOCX.',
          'Pack 2 (Full Technical & Regulatory Audit): Toggle Select All (14/14 Sections including Forensic Deleted-Records Log, Statistical Tests, VIF, SHAP & Fairness Card) -> Export as PDF + multi-sheet XLSX + JSON.',
        ],
        numericalConclusion: 'Both packs share 100% identical traced numbers directly from the active OmniForge session cache — zero manual copy-paste transcription errors.',
        engineeringAction: 'Always save the .omniforge project file alongside your exported PDF/XLSX so any future auditor can re-open and re-verify every chart interactively.',
      },
    ],
    uiControlsGuide: [
      {
        control: '14 Section Checkboxes & Quick Presets',
        defaultValue: 'All Available Completed Sections Checked',
        whenIncreased: 'Include all 14 sections for full end-to-end documentation.',
        whenDecreased: 'Uncheck unused specialized studios to keep executive PDFs crisp and focused.',
        expertGuidance: 'Run your desired studios first so Studio 12 populates live tables, equations, and embedded charts automatically.',
      },
      {
        control: 'Export Format Buttons (PDF / XLSX / DOCX / HTML / MD / JSON / YAML / LaTeX / CSV)',
        defaultValue: 'PDF + XLSX',
        whenIncreased: 'PDF & DOCX are ideal for stakeholder reading; XLSX & CSV for downstream spreadsheet work; JSON/LaTeX for academic or automated archiving.',
        whenDecreased: 'Choose HTML for instant interactive browser sharing with embedded visual tables.',
        expertGuidance: 'Verify the live Report Preview before exporting final executive copies.',
      },
    ],
    troubleshootingPlaybook: [
      {
        symptom: 'A selected report section appears empty or marked "Not yet executed"',
        mathematicalCause: 'That corresponding Studio has not been executed yet in the current session.',
        exactRemediationSteps: 'Jump to the studio in question, click Run, verify its output card populates, then return to Studio 12 (all prior studio results remain cached automatically!) and regenerate.',
      },
    ],
  },

  // =========================================================================
  // STUDIO 13 — DEEP TEXTBOOK SUPPLEMENT
  // =========================================================================
  '13': {
    deepTheory: [
      {
        heading: '13.1 Grounded Workspace-Aware AI Analytics Copilot vs. Hallucinating Generic Chatbots',
        paragraphs: [
          'Generic LLM chat windows have no visibility into your actual data dimensions, missingness defects, Split Lock status, VIF scores, Champion R², or active Supervisory alarms — forcing you to paste summaries manually and inviting fabricated numbers. OmniForge’s Studio 13 Copilot injects a real-time grounded Workspace Context Snapshot directly from the live backend session (row/column counts, target metadata, quality grades, active equations, top features, and live alarm counts) into every response.',
          'Use Studio 13 as a real-time senior statistical consultant: ask it to translate mathematical equations into business language, explain why a hypothesis test rejected H₀, suggest domain-specific feature ratios, interpret conflicting parametric vs. non-parametric results, or draft executive root-cause memos from Studio 14 alarms.',
        ],
      },
    ],
    workedExamples: [
      {
        title: 'Worked Example 13A: Effective High-Precision Copilot Prompts',
        scenario: 'After running Studios 01, 03, 06, and 14, you want an immediate synthesis of what drove today’s supervisory alarm.',
        stepByStepCalculation: [
          'Weak vague prompt: "Is my model good?"',
          'High-precision prompt: "Compare my Studio 06 AutoML Champion against the Studio 03 Baseline, explain the physical meaning of the top 3 equation coefficients, and summarize which features drove the Studio 14 positive deviation above Baseline."',
        ],
        numericalConclusion: 'Specific context-grounded prompts produce concise, quantitative, directly actionable guidance referencing your exact columns and scores.',
        engineeringAction: 'Verify every quantitative suggestion by inspecting the referenced studio’s tables and charts directly.',
      },
    ],
    uiControlsGuide: [
      {
        control: 'Suggested Quick-Prompt Chips & Chat Input',
        defaultValue: 'Workspace-Grounded Analytical Mode',
        whenIncreased: 'Provide exact column names, business costs of False Positives vs False Negatives, or operational constraints for tailored answers.',
        whenDecreased: 'Click any Quick-Prompt chip for instant guided diagnostics on the current dataset.',
        expertGuidance: 'Always treat Copilot as an accelerator alongside — never a substitute for — OmniForge’s exact statistical tests and split-locked validation.',
      },
    ],
    troubleshootingPlaybook: [
      {
        symptom: 'Copilot says no model or diagnostics are available yet',
        mathematicalCause: 'Only data upload has been run; downstream Studios 03/06/14 haven’t produced artifacts yet.',
        exactRemediationSteps: 'Run Studio 01 Intake/Clean -> Studio 03 Baselines -> Studio 06 AutoML so the Copilot snapshot has full quantitative telemetry to reason over.',
      },
    ],
  },

  // =========================================================================
  // STUDIO 14 — DEEP TEXTBOOK SUPPLEMENT (EQUATIONS + SIGNED ROOT-CAUSE ATTRIBUTION!)
  // =========================================================================
  '14': {
    deepTheory: [
      {
        heading: '14.1 Supervisory Engineering Control Room Architecture: Actual vs. Champion vs. Baseline Triad',
        paragraphs: [
          'Industrial SCADA systems, financial risk desks, and clinical monitoring platforms require continuous real-time verification that live reality aligns with validated expectations. Studio 14 monitors EVERY incoming observation (from CSV, Excel .xlsx, SQLite/PostgreSQL/MySQL/SQL Server databases, Microsoft Access .accdb/.mdb, live single-record manual entry, or realistic fault-injection simulation) across an exact three-way triad: (1) Actual Observed Value y_actual (when ground truth arrives), (2) Active Champion Model Prediction ŷ_champion, and (3) Reference Baseline Model Prediction ŷ_baseline.',
          'Simultaneously, Studio 14 displays the EXACT Governing Mathematical Equations for BOTH the Champion and Reference Baseline right inside the Control Room — complete with portable Python, Excel, SQL, and MATLAB/C exports — so control-room engineers see the precise algebraic law generating every prediction and tolerance limit.',
        ],
      },
      {
        heading: '14.2 Signed Root-Cause Attribution: Exactly Identifying What Caused an Increase (+Δ) or Decrease (−Δ) vs. Baseline',
        paragraphs: [
          'Knowing THAT an alarm triggered ("Actual = 118.4, Baseline = 98.2, Error = +20.6%") is only half the battle — operators urgently need to know WHY it increased or decreased and WHAT to adjust right now to fix it! Studio 14 solves this directly using Antithetic Interventional Root-Cause Attribution grounded in your exact training normal operating point x_ref = median(X_train).',
          'For every predictor feature x_j, Studio 14 computes two symmetric counterfactual interventions across the alarmed/deviation slice: (1) Forward Activation Δ⁺_j = f(x_ref with x_j set to live x_j) − f(x_ref), which measures how much moving feature j from its normal reference to its live observed value shifts the response; and (2) Backward Restoration Δ⁻_j = f(x_live) − f(x_live with x_j restored to x_ref), which measures how much restoring feature j back to normal fixes the deviation (capturing full nonlinear interactions!).',
          'Averaging φ_j = 0.5 · (Δ⁺_j + Δ⁻_j) yields an exact signed second-order Shapley-style attribution for every feature — working seamlessly for Linear models, Ridge, Non-Parametric GAMs, Random Forests, Gradient Boosting, AND Stacking Ensembles! Every feature is then strictly partitioned into TWO transparent engineering tables: (1) Factors Driving INCREASE (+Δ Above Baseline, φ_j > 0) and (2) Factors Driving DECREASE (−Δ Below Baseline, φ_j < 0), plotted on a signed Diverging Tornado Chart, AND paired with an exact Residual Check (Actual − Champion): if Unexplained Residual is small, the input drivers explain the entire deviation; if Unexplained Residual is large, an unmeasured external fault, sensor miscalibration, or structural equipment degradation (concept drift) is occurring!',
        ],
      },
      {
        heading: '14.3 Statistical Process Control (SPC) Shewhart Charts & Dual-Limit Tolerance Governance',
        paragraphs: [
          'Studio 14 pairs deterministic engineering tolerance guards (Absolute Error |y − ŷ| > T_abs and Relative Percentage Error 100·|y − ŷ|/|y| > T_pct) with classical Shewhart Statistical Process Control (SPC) limits computed from in-control residual variance: Warning Limits at μ_e ± 2σ_e (≈ 95.4% in-control envelope) and Critical Control Limits (UCL / LCL) at μ_e ± 3σ_e (≈ 99.73% in-control envelope).',
          'Combining Tolerance Bands with SPC Control Limits prevents both blind spots: Tolerance Bands enforce hard customer/safety contractual specifications, while SPC ±3σ limits detect subtle systematic drift or creeping variance instability DAYS before a catastrophic specification breach occurs.',
        ],
      },
    ],
    workedExamples: [
      {
        title: 'Worked Example 14A: Diagnosing a +18.5 kW Power-Consumption Spike Above Baseline & Generating the Exact Fix',
        scenario: 'During live monitoring of a chiller plant, Baseline expected 100.0 kW, Champion predicted 116.0 kW, and Actual metered power spiked to 118.5 kW (+18.5% breach > 10% tolerance!). Training normal medians: Condenser_Approach_Temp = 4.0 °C (live = 9.2 °C, +3.4σ), Cooling_Load = 80% (live = 84%, +0.5σ), Ambient_Temp = 25 °C (live = 22 °C, −0.8σ).',
        stepByStepCalculation: [
          'Step 1 — Total Deviation Decomposition: Actual − Baseline = +18.5 kW. Model-Explained Shift (Champion − Baseline) = +16.0 kW (86.5% explained by measured inputs!). Unexplained Residual (Actual − Champion) = +2.5 kW (minor noise).',
          'Step 2 — Signed Antithetic Attribution φ_j = 0.5·(Δ⁺_j + Δ⁻_j) per feature:',
          '  • Condenser_Approach_Temp (4.0 -> 9.2 °C, +3.4σ): φ = +14.8 kW (UPWARD +Δ INCREASE DRIVER — 76% share!).',
          '  • Cooling_Load (80% -> 84%, +0.5σ): φ = +3.1 kW (UPWARD +Δ INCREASE DRIVER — 16% share).',
          '  • Ambient_Temp (25 -> 22 °C, −0.8σ): φ = −1.9 kW (DOWNWARD −Δ OFFSET — cooler air partially cushioned the spike!).',
          'Step 3 — Net Verification: +14.8 + 3.1 − 1.9 = +16.0 kW (matches Champion − Baseline shift exactly!).',
        ],
        numericalConclusion: 'Condenser_Approach_Temp (+3.4σ above normal) single-handedly caused +14.8 kW (+76% of the excursion), while cooler Ambient_Temp (−1.9 kW) partially offset it.',
        engineeringAction: 'Remediation Playbook Priority P1: Inspect condenser tubes for fouling/scaling and check cooling-water flow rate to bring Condenser_Approach_Temp back inside its healthy training IQR band [3.2, 4.8 °C].',
      },
      {
        title: 'Worked Example 14B: Recognizing Unmeasured Concept Drift / Sensor Failure via Unexplained Residual',
        scenario: 'Baseline predicts 50.0 bar, Champion predicts 50.4 bar (all input features are within ±0.3σ of normal medians!), yet Actual reads 64.0 bar (+28% CRITICAL alarm).',
        stepByStepCalculation: [
          'Total Deviation: Actual − Baseline = +14.0 bar.',
          'Input-Explained Shift (Champion − Baseline) = +0.4 bar (only 3% explained by tracked inputs).',
          'Unexplained Residual (Actual − Champion) = +13.6 bar (97% unexplained!).',
        ],
        numericalConclusion: 'All measured inputs sit inside normal healthy ranges, yet output jumped +13.6 bar -> Classic signature of EITHER an unmeasured external fault (e.g., stuck downstream valve) OR output pressure-sensor calibration drift!',
        engineeringAction: 'Do NOT adjust healthy input setpoints! Dispatch instrumentation technician to verify output sensor zero/span calibration and inspect unmodeled downstream equipment, then retrain model if process baseline permanently shifted.',
      },
    ],
    uiControlsGuide: [
      {
        control: 'Champion Model & Reference Baseline Selectors',
        defaultValue: 'Best AutoML / Active Model vs Strong Studio 03 Baseline',
        whenIncreased: 'Switch Champion or Baseline at any time — Section 1B immediately updates BOTH Governing Mathematical Equations live.',
        whenDecreased: 'If no model exists yet, click Quick-Train & Select Model to train and register both Champion and Baseline in one click.',
        expertGuidance: 'Always pair a high-accuracy Champion (AutoML/GAM) against a transparent linear/baseline reference so lift and deviation remain auditable.',
      },
      {
        control: 'Max Relative Error Tolerance (%) & Max Absolute Error',
        defaultValue: '10% Relative / Auto Absolute',
        whenIncreased: '15–25% reduces nuisance alarms on noisy commercial or behavioral targets.',
        whenDecreased: '2–5% enforces tight aerospace/pharma/precision-manufacturing control limits.',
        expertGuidance: 'Set tolerance slightly above your verified Studio 03/06 validation MAPE/MAE (e.g., 1.5×–2.0× validation error) to catch genuine anomalies without false-alarm fatigue.',
      },
      {
        control: 'Live Data Connectors (CSV / Excel / SQL / MS Access / Single Reading / Fault Simulation)',
        defaultValue: 'Simulation (for instant verification) or File/DB upload',
        whenIncreased: 'Use Live Fault Simulation (Inject Drift/Fault) to stress-test your alarm thresholds and watch Signed Root-Cause Attribution pinpoint the exact perturbed feature.',
        whenDecreased: 'Use Single-Reading Manual Evaluator for instant operator "what-if" diagnostics before applying a setpoint adjustment.',
        expertGuidance: 'Run a test evaluation with Simulated Drift first to calibrate your alert thresholds and verify your operations team understands the Root-Cause Playbook.',
      },
      {
        control: 'Single-Alarm Deep Root-Cause Inspector Dropdown',
        defaultValue: 'Alarm Record #1',
        whenIncreased: 'Select any individual WARNING or CRITICAL record from the Alarm Ledger to view its exact per-record Diverging Driver Chart and tailored corrective fix.',
        whenDecreased: 'Filter Ledger by CRITICAL vs WARNING first to triage highest-severity breaches immediately.',
        expertGuidance: 'Always check BOTH batch-level root causes (systemic drift affecting all rows) AND individual critical alarm breakdowns (localized sudden spikes).',
      },
    ],
    troubleshootingPlaybook: [
      {
        symptom: '100% of incoming live records trigger CRITICAL alarms immediately',
        mathematicalCause: 'Unit mismatch between training and live feed (e.g., training in °C vs live in °F, or training in $k vs live in $), OR tolerance % is set tighter than the model’s inherent validation error.',
        exactRemediationSteps: '1) Check Section 4B Live Mean vs. Normal Reference column — a 10× or 1.8×+32 shift proves unit mismatch immediately! 2) Compare tolerance % against Studio 06 validation MAPE. 3) Verify column names match exactly.',
      },
      {
        symptom: 'Incoming live batch has no ground-truth Target column yet (Score-Only Monitoring)',
        mathematicalCause: 'In real-time prediction before outcomes materialize, y_actual is not yet observed.',
        exactRemediationSteps: 'Studio 14 automatically switches to Score-Only Supervisory Mode: it flags Champion-vs-Baseline divergence (> tolerance), evaluates exact Signed Root-Cause Attribution on Champion − Baseline shifts, and highlights out-of-range input z-shifts (> 2.5σ) proactively!',
      },
    ],
    extraMethods: [
      {
        name: 'Antithetic Interventional Signed Root-Cause Attribution (+Δ Increase vs. −Δ Decrease)',
        paradigm: 'Model-Agnostic Signed Baseline Deviation Diagnostics',
        whatItDoes:
          'Decomposes the exact signed deviation between live system behavior and the training Baseline into per-feature additive contributions φ_j by combining forward single-feature perturbation from normal reference medians (Δ⁺_j) with backward single-feature restoration from the live abnormal state (Δ⁻_j). Strictly partitions causes into Upward Increase Drivers (+Δ) and Downward Decrease Drivers (−Δ) and synthesizes prioritized P1..P6 engineering corrective actions.',
        formula: {
          equation: 'φ_j = ½ [ (f(x_ref[j ← x_live,j]) − f(x_ref)) + (f(x_live) − f(x_live[j ← x_ref,j])) ]   |   Residual = (y_act − ŷ_champ)',
          symbols: 'x_ref = training normal median vector; x_live = alarmed/live input vector; φ_j > 0 drives INCREASE (+Δ) above baseline; φ_j < 0 drives DECREASE (−Δ) below baseline; Residual isolates unmeasured external faults or sensor calibration drift.',
        },
        whenToUse: 'Whenever Studio 14 raises WARNING/CRITICAL alarms or shows systematic shift from Baseline — works across ALL model types (Linear, Ridge, GAM, Random Forest, GBDT, Stacking).',
        assumptions: 'Training normal reference statistics (median, IQR, std) represent healthy in-control operation.',
        recommendedParams: 'Automatic on every Studio 14 batch evaluation (scope = Breached Alarms when present, else full live batch) + per-alarm individual drilldown.',
        pitfalls: 'If Unexplained Residual accounts for > 50% of total deviation while all input z-shifts stay near 0, suspect an unmeasured disturbance or output sensor calibration error rather than input setpoints.',
      },
    ],
  },
};
