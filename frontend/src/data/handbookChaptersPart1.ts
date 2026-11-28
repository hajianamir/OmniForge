import type { StudioChapter } from './handbookTypes';

export const HANDBOOK_CHAPTERS_PART1: StudioChapter[] = [
  // =========================================================================
  // CHAPTER 0 — STUDIO 00
  // =========================================================================
  {
    studioId: '00',
    chapterNum: 'Chapter 00',
    title: 'Data Ingestion, Intake Audit, Split Lock & Data Relationships',
    subtitle: 'Multi-source data loading, immutable raw view preservation, zero-leakage Split Lock, and linear/nonlinear relationship diagnostics.',
    overview:
      'Studio 00 is the entry point of OmniForge. It handles multi-source ingestion (CSV, Excel, Parquet, JSON, SQL, MS Access, and Benchmark datasets), preserves an immutable raw copy (uap_raw_df), executes InitialIntakeAuditor, enforces Split Lock before any cleaning or modeling, and visualizes pairwise variable relationships (Pearson r, Spearman rho, and Distance Correlation dCor) via interactive charts and tables.',
    realWorldApplications: [
      'Onboarding raw enterprise datasets while preserving an unclipped audit copy.',
      'Detecting linear, monotonic, and nonlinear dependencies (dCor) immediately after ingestion.',
      'Locking Train/Test membership by zero-based row positions before Studio 01 Data Cleaning.',
    ],
    stepByStep: [
      '1. Ingest Data Source: Upload CSV/XLSX/Parquet/JSON or connect to SQL/Access or load a benchmark dataset.',
      '2. Inspect Intake Audit & Data Relationships: Review structural defects and click Analyze Data Relationships (Chart & Table) to inspect Pearson, Spearman, and dCor.',
      '3. Enable Split Lock Before Studio 01: Select target/excluded columns and lock Train/Test partitions before proceeding to Studio 01 Data Cleaning.',
    ],
    provenanceRule: 'Studio 00 stores an immutable raw view and positional Split Lock partitions. Realignment requires preserved, unique DataFrame index labels; ambiguous identity invalidates the lock.',
    methods: [],
    thresholdTables: [],
    decisionMatrix: { title: 'Studio 00 Ingestion & Split Lock Matrix', rows: [] },
  },
  // =========================================================================
  // CHAPTER 1 — STUDIO 01
  // =========================================================================
  {
    studioId: '01',
    chapterNum: 'Chapter 01',
    title: 'Data Ingestion, Automatic Defect Audit, Split Lock & Cleaning',
    subtitle: 'Multi-source data loading, 6D quality auditing, zero-leakage Train/Test locking, imputation, and outlier remediation.',
    overview:
      'Studio 01 is the scientific foundation of every OmniForge workflow. Every statistical test, baseline equation, AutoML ensemble, and live supervisory monitor downstream inherits the data quality and anti-leakage guarantees established here. Immediately upon ingestion, OmniForge runs InitialIntakeAuditor to grade structural and statistical health before you alter a single cell, preserves an immutable copy of Raw Data alongside your Active Data, and records every removed record or dropped column in a forensic audit log.',
    realWorldApplications: [
      'Industrial SCADA, turbine, HVAC, and IoT sensor telemetry sanitization (handling sensor dropouts, spikes, and formatting noise).',
      'Enterprise ETL & multi-sheet Excel / SQL / MS Access data onboarding with automated schema validation.',
      'Clinical, financial, and supply-chain dataset preparation with regulatory audit trails of every excluded record.',
      'Zero-leakage Train/Test partitioning prior to machine learning benchmarking.',
    ],
    stepByStep: [
      'Ingest data from File Upload (CSV/TSV/XLSX/Parquet/JSON), SQL Database Connector, MS Access (.accdb/.mdb), or Benchmark Datasets.',
      'Review the automatic Intake Defect Audit scorecard (Grade A–F) and confirm Targets & Exclusions in the topbar.',
      'CRITICAL FOR MODELING: Open "Split Lock" in the topbar and lock your Train/Test partition BEFORE running Data Cleaning.',
      'Click "Run Deep 6D Quality Profile" to inspect Completeness, Uniqueness, Validity, Consistency, Accuracy (requires trusted reference data), and Timeliness.',
      'Select Auto, Complete, or Custom cleaning mode, click "Dry-Run Preview" to verify impact safely, then click "Apply & Commit Cleaning".',
      'Optionally export clean CSV/XLSX, inspect the Forensic Deleted-Records Audit Log, or Undo / Revert to Raw anytime.',
    ],
    provenanceRule:
      'Committing cleaning transitions Active Data from RAW -> CLEANED while keeping Raw Data permanently intact. When Split Lock is active, all statistical imputers and outlier fences fit strictly on Train (fit_transform_split) and transform Test without leakage.',
    methods: [
      {
        name: 'Anti-Leakage Train/Test Split Lock',
        paradigm: 'Validation Governance',
        whatItDoes:
          'Separates the dataset into a frozen Training set and an unseen Holdout Test set BEFORE any data-dependent statistics (medians, means, standard deviations, IQR fences, KNN neighborhoods, scalers, or cluster centroids) are computed. This guarantees that validation metrics measure true out-of-sample generalization rather than memorized test statistics.',
        formula: {
          equation: 'θ_clean = fit(X_train),    X̃_train = transform(X_train; θ_clean),    X̃_test = transform(X_test; θ_clean)',
          symbols: 'θ_clean = learned imputation/outlier/scaling parameters strictly estimated from X_train only.',
        },
        whenToUse: 'Mandatory BEFORE Data Cleaning whenever your goal includes predictive modeling, AutoML, Non-Parametric modeling, Deep Learning, or Studio 14 Supervisory deployment.',
        assumptions: 'Stratified requires discrete target classes; Time-Series requires chronological row order (past -> future).',
        recommendedParams: 'N < 500: test_size = 0.25–0.30, cv_folds = 5 | N = 500–20,000: test_size = 0.20, cv_folds = 5 | N > 20,000: test_size = 0.10–0.20.',
        pitfalls: 'Skipping Split Lock before KNN/MICE/Mean/IQR cleaning inflates test R²/AUC optimistically. Only structural formatting fixes or pure BI exports without ML can skip it.',
      },
      {
        name: 'Numeric Missing-Value Imputation (Median / Mean / KNN / Iterative MICE / Time-Series ffill)',
        paradigm: 'Missing Data Remediation',
        whatItDoes:
          'Replaces missing (NaN/null) cells with statistically principled estimates so downstream algorithms do not drop valuable rows. Median/Mean fill univariate center; KNN averages the k most similar complete observations; Iterative (MICE) models each incomplete column sequentially as a regression on all other features; Forward-Fill (ffill) propagates the last valid past reading along time.',
        formula: {
          equation: 'KNN: x̂_ij = (Σ_{r ∈ N_k(i)} w_ir · x_rj) / (Σ_{r ∈ N_k(i)} w_ir)    |    MICE: x̂_j^(t) = f_j(X_{-j}^(t-1))',
          symbols: 'N_k(i) = k nearest observed rows to row i; w_ir = inverse Euclidean distance; X_{-j} = all columns other than j.',
        },
        whenToUse: 'Median: skewed/outlier-prone numeric columns (safest default). Mean: symmetric bell-shaped columns (<10% missing). KNN/MICE: correlated features with 5–30% missingness (with Split Lock). ffill/Interpolate: chronological time-series.',
        assumptions: 'Mean/Median assume Missing Completely At Random (MCAR); KNN/MICE handle Missing At Random (MAR) using cross-column covariance.',
        recommendedParams: 'Enable "Add Missingness Indicator Flags" when missingness exceeds 5% — missingness itself is often predictive (MNAR).',
        pitfalls: 'Never use backward-fill (bfill) on causal time series — it leaks future values backward (OmniForge automatically enforces causal past-only ffill in causal mode).',
      },
      {
        name: 'IQR Fence vs. Z-Score vs. Modified Z-Score (MAD) Outlier Rules',
        paradigm: 'Univariate Outlier Detection',
        whatItDoes:
          'Identifies numerically extreme values that may represent sensor glitches, data-entry typos, or rare extreme regimes. IQR uses the middle 50% spread (robust to skewness); Z-Score measures standard deviations from the mean (Gaussian only); Modified Z-Score uses Median Absolute Deviation (MAD), resisting "masking" when multiple extreme outliers cluster together.',
        formula: {
          equation: 'IQR: [Q₁ − k·IQR, Q₃ + k·IQR]   |   Z = (x − μ) / σ   |   MAD-Z: M_i = 0.6745 · (x_i − median(x)) / MAD',
          symbols: 'IQR = Q₃ − Q₁; MAD = median(|x_i − median(x)|); 0.6745 scales MAD to match normal σ.',
        },
        whenToUse: 'IQR (k = 1.5 mild, 3.0 extreme) or MAD (|M_i| > 3.5) for almost all real-world skewed data. Z-Score (|Z| > 3.0) ONLY when Studio 03/04 confirms normality.',
        recommendedParams: 'Action = "Clip / Winsorize" preserves row count while bounding extreme spikes; "Flag" preserves values with indicator; "Remove" ONLY for verified measurement corruption.',
        pitfalls: 'Deleting valid extreme values (e.g., high-value customers or real equipment failure spikes) destroys the very tail signal predictive models need to learn.',
      },
      {
        name: '6-Dimensional Enterprise Data Quality Score',
        paradigm: 'Quality Governance',
        whatItDoes:
          'Summarizes six dataset quality dimensions on a 0–100 scale. Accuracy is the percentage of observed values that match a trusted reference over comparable cells. Numeric range conformance and IQR-based plausibility are separate diagnostics.',
        formula: {
          equation: 'Q_overall = Σ(wᵢ · Qᵢ) / Σwᵢ over evaluated dimensions',
          symbols: 'Qᵢ is a dimension score from 0 to 100. The reported effective weights are normalized over the evaluated dimensions.',
        },
        whenToUse: 'Run immediately before and after cleaning to document measurable quality uplift in Studio 12 Executive Reports.',
        recommendedParams: 'Target Overall Score >= 85/100 before modeling.',
        pitfalls: 'A high completeness score can still hide mixed-type strings or disguised nulls ("?", "N/A", "-999") — always enable Standardize Missing Tokens.',
      },
    ],
    thresholdTables: [
      {
        title: 'Recommended Data Cleaning & Missingness Thresholds',
        rows: [
          { metric: 'Column Missingness Rate', range: '< 5%', level: 'good', interpretation: 'Minimal missing data.', action: 'Safe to impute with Median (numeric) / Mode (categorical).' },
          { metric: 'Column Missingness Rate', range: '5% – 30%', level: 'warn', interpretation: 'Moderate missingness — may carry informative signal.', action: 'Use KNN or Iterative MICE (with Split Lock) + enable Missingness Indicator Flags.' },
          { metric: 'Column Missingness Rate', range: '30% – 50%', level: 'warn', interpretation: 'High missingness — heavy imputation bias risk.', action: 'Verify domain relevance; keep only if strongly predictive with missing flag.' },
          { metric: 'Column Missingness Rate', range: '> 50%', level: 'bad', interpretation: 'Severely sparse column.', action: 'Drop automatically via High-Missing Column Threshold (default 0.50).' },
          { metric: 'IQR Multiplier (k)', range: 'k = 1.5 (standard) | k = 3.0 (extreme only)', level: 'info', interpretation: '1.5×IQR flags ~0.7% of normal tails; 3.0×IQR flags extreme spikes only.', action: 'Use 1.5×IQR + Clip for noisy sensors; 3.0×IQR when legitimate heavy tails exist.' },
          { metric: 'Modified Z-Score (MAD)', range: '|M_i| > 3.5', level: 'info', interpretation: 'Iglewicz & Hoaglin robust outlier cutoff.', action: 'Preferred over classical Z-score whenever skewness |g₁| > 0.5.' },
          { metric: '6D Overall Quality Score', range: '>= 85 Excellent | 70–84 Fair | < 70 Poor', level: 'good', interpretation: 'Readiness for modeling & statistical inference.', action: 'Scores < 70 require cleaning & format repair before trusting downstream models.' },
        ],
      },
    ],
    decisionMatrix: {
      title: 'Studio 01 Imputation & Outlier Rule Decision Table',
      rows: [
        { application: 'Skewed industrial / financial numeric column with missing cells', dataCondition: '|Skewness| > 0.5, missing < 20%', recommended: 'Median Imputation + Missing Indicator', alternative: 'Iterative MICE (if correlated features exist)', rationale: 'Median is unaffected by extreme values and preserves central rank.' },
        { application: 'Strongly correlated sensor / lab measurements with missing blocks', dataCondition: 'Multiple numeric sensors, missing 5–30%, Split Locked', recommended: 'KNN or Iterative (MICE) Imputation', alternative: 'Median + Missing Indicator (for very large N > 100k)', rationale: 'Leverages cross-sensor physical correlations instead of flat constant filling.' },
        { application: 'Chronological time-series sensor dropouts', dataCondition: 'Ordered timestamps, continuous readings', recommended: 'Forward-Fill (ffill) or Linear Interpolate', alternative: 'Rolling local median', rationale: 'Preserves temporal continuity without leaking future observations.' },
        { application: 'Spiky measurement glitches in numeric telemetry', dataCondition: 'Non-normal distribution, rare extreme spikes', recommended: 'Modified Z-Score (MAD > 3.5) or IQR (1.5×) -> Clip/Winsorize', alternative: 'Remove rows ONLY if physically impossible (e.g. negative Kelvin)', rationale: 'Winsorizing bounds noise without discarding entire valid rows.' },
      ],
    },
  },

  // =========================================================================
  // CHAPTER 2 — STUDIO 02
  // =========================================================================
  {
    studioId: '02',
    chapterNum: 'Chapter 02',
    title: 'Clustering, Multivariate Anomaly Screening & Data Mining Lab',
    subtitle: 'Unsupervised segmentation, corrupt-regime removal, multivariate outlier detectors, association rules, subgroup discovery, and manifold embeddings.',
    overview:
      'Studio 02 uncovers hidden geometric, density, and co-occurrence structures that univariate cleaning cannot detect. Real-world datasets often contain mixed operating modes (e.g., idle vs peak load, healthy vs failing assets, distinct customer tiers) or multivariate anomalies that look normal in every single column individually yet violate physical relationships jointly. Every operation previews safely before you decide whether to remove corrupt clusters/anomalies or enrich Active Data with unsupervised features.',
    realWorldApplications: [
      'Isolating and removing corrupt sensor regimes, calibration drift batches, or DBSCAN noise points (-1) before predictive modeling.',
      'Segmenting customers, patients, financial accounts, or industrial operating modes into interpretable profiles.',
      'Detecting multivariate fraud, cyber intrusions, or mechanical incipient faults via Isolation Forest / LOF / One-Class SVM.',
      'Mining IF-THEN fault co-occurrence or market-basket rules (Apriori) and high-risk target subgroups (WRAcc).',
    ],
    stepByStep: [
      'Tab 1 (Clustering): Choose Auto (best k via Silhouette sweep), Complete (benchmark all 6 algorithms), or Custom (pick exact algorithm & k/eps), then click "Run Clustering Preview".',
      'Inspect cluster cards, algorithm benchmark table, and 2D PCA scatter. Either select corrupt/noise clusters to Remove OR click "Attach cluster_id" to enrich features.',
      'Tab 2 (Anomaly Screening): Run Isolation Forest, LOF, and/or One-Class SVM to inspect anomaly rate, multi-detector agreement, and Top Anomalous Records; then Attach scores or Remove anomalies.',
      'Tab 3 (Data Mining): Execute Apriori Association Rules, Subgroup Discovery (WRAcc), and/or Manifold Projections (PCA, SVD, t-SNE, Isomap, MDS, Spectral Embedding).',
    ],
    provenanceRule:
      'Runs strictly in PREVIEW_ONLY mode until you click a Commit button. Committing removes selected clusters/anomalies (logged in Deleted-Records Audit) or appends cluster_id / is_anomaly / anomaly_score, lighting up the Studio 02 green status LED (CLEANED + CLUSTER-SCREENED). Helper columns are automatically excluded from re-clustering.',
    methods: [
      {
        name: 'K-Means & BIRCH Clustering',
        paradigm: 'Centroid / Hierarchical Partitioning',
        whatItDoes:
          'Partitions observations into K mutually exclusive spherical clusters by minimizing within-cluster squared Euclidean distance to cluster centroids. BIRCH builds a compact Clustering Feature (CF) tree for ultra-fast scaling on large datasets.',
        formula: {
          equation: 'min_{C₁..C_K}  Σ_{c=1}^K  Σ_{x_i ∈ C_c} ‖x_i − μ_c‖₂²',
          symbols: 'μ_c = centroid mean vector of cluster C_c; K = number of clusters selected via Silhouette optimization.',
        },
        whenToUse: 'Fast, clean baseline segmentation when clusters are roughly convex/spherical and comparable in spread.',
        assumptions: 'Standardized numeric features; sensitive to extreme outliers (clean in Studio 01 first).',
        recommendedParams: 'Start with Auto-k (k = 2..8) maximizing Silhouette Score.',
        pitfalls: 'Cannot separate concentric rings, crescents, or arbitrary density shapes — switch to DBSCAN or Spectral for non-convex geometry.',
      },
      {
        name: 'Gaussian Mixture Model (GMM)',
        paradigm: 'Probabilistic Soft Clustering',
        whatItDoes:
          'Models the dataset as a weighted sum of K multivariate Gaussian distributions with distinct means and covariance ellipses, fitted via Expectation-Maximization (EM). Produces soft membership probabilities rather than rigid boundaries.',
        formula: {
          equation: 'p(x) = Σ_{k=1}^K π_k · 𝒩(x | μ_k, Σ_k),     Σ_{k=1}^K π_k = 1',
          symbols: 'π_k = mixing weight, μ_k = mean vector, Σ_k = full covariance matrix of Gaussian component k.',
        },
        whenToUse: 'Overlapping subpopulations, elongated/elliptical regimes, or when probabilistic cluster confidence is valuable.',
        recommendedParams: 'Select K via Silhouette + BIC/AIC stability; works best when n >> K·p.',
        pitfalls: 'Can overfit singular covariances in very high dimensions without prior PCA/feature selection.',
      },
      {
        name: 'DBSCAN (Density-Based Spatial Clustering of Applications with Noise)',
        paradigm: 'Density-Based Clustering + Noise Isolation',
        whatItDoes:
          'Groups densely packed regions of arbitrary shape and explicitly isolates low-density outlier points as Noise (Cluster -1) without requiring you to guess K in advance.',
        formula: {
          equation: 'Core Point Rule:  |N_ε(x)| = |{y ∈ D : dist(x, y) ≤ ε}| ≥ min_samples',
          symbols: 'ε (eps) = neighborhood radius in standardized space; points not density-reachable from any core point receive label -1.',
        },
        whenToUse: 'Non-convex shapes, unknown number of operating regimes, OR when your primary goal is isolating and deleting corrupt noise points (Cluster -1).',
        recommendedParams: 'Start with min_samples ≈ max(5, 2 × n_features) and tune eps via k-distance elbow (OmniForge Auto estimates eps automatically).',
        pitfalls: 'Struggles when different valid clusters have sharply different internal densities.',
      },
      {
        name: 'Cluster Validity Indices (Silhouette, Calinski-Harabasz, Davies-Bouldin)',
        paradigm: 'Unsupervised Quality Metrics',
        whatItDoes:
          'Mathematically measures how compact each cluster is internally versus how well-separated distinct clusters are from each other.',
        formula: {
          equation: 'Silhouette: s(i) = (b(i) − a(i)) / max(a(i), b(i)) ∈ [−1, +1]    |    DB = (1/K) Σ_i max_{j≠i} ((σ_i + σ_j) / d(μ_i, μ_j))',
          symbols: 'a(i) = mean intra-cluster distance; b(i) = mean nearest-other-cluster distance; higher Silhouette & CH = better; LOWER Davies-Bouldin = better.',
        },
        whenToUse: 'Always compare algorithms and K values objectively using Silhouette + Davies-Bouldin together.',
        recommendedParams: 'Prefer solutions with Silhouette >= 0.35 and low Davies-Bouldin (< 1.2) that also make domain sense.',
        pitfalls: 'Silhouette inherently favors spherical convex clusters (K-Means) over elongated DBSCAN manifolds.',
      },
      {
        name: 'Isolation Forest vs. Local Outlier Factor (LOF) vs. One-Class SVM',
        paradigm: 'Multivariate Anomaly Detection',
        whatItDoes:
          'Detects rows that violate joint multi-feature patterns. Isolation Forest isolates anomalies via fewer random axis splits (shorter path length h(x)); LOF compares a point’s local density against its k neighbors; One-Class SVM wraps a smooth kernel boundary around normal data.',
        formula: {
          equation: 'IForest: s(x, n) = 2^{−E[h(x)] / c(n)}    |    LOF_k(x) = (1/|N_k(x)|) Σ_{o ∈ N_k(x)} lrd_k(o) / lrd_k(x)',
          symbols: 'E[h(x)] = average tree depth to isolate x (s -> 1 indicates strong anomaly); LOF >> 1 indicates much lower density than neighbors.',
        },
        whenToUse: 'Isolation Forest: fast general-purpose default. LOF: anomalies hidden inside clusters of varying density. One-Class SVM: tight normal operating envelope.',
        recommendedParams: 'Contamination = 0.01–0.05 (1%–5%) unless domain failure rate is known; run Complete mode to see consensus across all 3 detectors.',
        pitfalls: 'Always inspect Top Anomalies before deleting — anomalies may represent high-value rare events or critical fault precursors.',
      },
      {
        name: 'Apriori Association Rules & Subgroup Discovery (WRAcc) & Manifold Embedding',
        paradigm: 'Pattern & Structure Mining',
        whatItDoes:
          'Apriori discovers frequent co-occurring attribute rules (IF A -> THEN B). Subgroup Discovery finds interpretable rules whose target rate deviates maximally from the population base rate via WRAcc. Manifold projections (PCA, SVD, t-SNE, Isomap, MDS) compress high-D structure into 2D visuals.',
        formula: {
          equation: 'Lift(A→B) = P(A∩B) / (P(A)·P(B))    |    WRAcc(S) = (n_S / N) · (p_S − p₀)',
          symbols: 'Lift > 1 means positive association beyond chance; n_S/N = subgroup coverage; p_S − p₀ = target rate lift over base rate p₀.',
        },
        whenToUse: 'Apriori for fault co-occurrence & market baskets; WRAcc for actionable high-risk slices; PCA for linear variance axes; t-SNE for local neighborhoods; Isomap/MDS for nonlinear geodesic geometry.',
        recommendedParams: 'Apriori: min_support >= 0.05, min_confidence >= 0.60, focus on Lift >= 1.25.',
        pitfalls: 'High Confidence alone is meaningless if B occurs in 95% of rows — ALWAYS rank rules by Lift. In t-SNE plots, cluster sizes and global inter-cluster distances are non-linear.',
      },
    ],
    thresholdTables: [
      {
        title: 'Clustering, Anomaly & Association Rule Numeric Thresholds',
        rows: [
          { metric: 'Silhouette Score', range: '> 0.70 Strong | 0.50–0.70 Good | 0.25–0.50 Fair | < 0.25 Weak', level: 'good', interpretation: 'Strength of cluster separation.', action: 'If < 0.25 across all algorithms, data may be continuous rather than clustered.' },
          { metric: 'Davies-Bouldin Index (DBI)', range: '< 0.80 Excellent | 0.80–1.50 Acceptable | > 1.50 Overlapping', level: 'good', interpretation: 'Average similarity of each cluster to its closest rival (lower is better).', action: 'Minimize DBI alongside maximizing Silhouette.' },
          { metric: 'Isolation Forest Score s(x)', range: '> 0.65 Strong Anomaly | 0.50–0.65 Borderline | < 0.50 Normal', level: 'warn', interpretation: 'Normalized path-length isolation probability.', action: 'Verify > 0.65 rows in domain audit before deciding Remove vs Attach.' },
          { metric: 'Local Outlier Factor (LOF)', range: '≈ 1.0 Normal | 1.5–2.5 Moderate | > 2.5 Strong Local Outlier', level: 'warn', interpretation: 'Density ratio relative to k nearest neighbors.', action: 'Use LOF > 2.0 when clusters have varying density scales.' },
          { metric: 'Association Rule Lift', range: '< 1.0 Negative | 1.0 Independent | 1.2–2.0 Useful | > 2.0 Strong Rule', level: 'good', interpretation: 'Factor by which A increases probability of B over random chance.', action: 'Prioritize rules with Lift >= 1.25 AND Support >= 0.05.' },
        ],
      },
    ],
    decisionMatrix: {
      title: 'Studio 02 Unsupervised Algorithm Selector',
      rows: [
        { application: 'Clean spherical customer/asset tiers on medium-to-large data', dataCondition: 'Numeric standardized features, roughly convex groups', recommended: 'K-Means (or BIRCH if N > 50,000)', alternative: 'GMM (if groups overlap or stretch elliptically)', rationale: 'Fast, deterministic, interpretable centroids.' },
        { application: 'Isolate & delete arbitrary-shaped noise / faulty sensor clusters', dataCondition: 'Unknown K, noisy outlier cloud around dense regimes', recommended: 'DBSCAN -> Remove Cluster -1', alternative: 'Isolation Forest -> Remove Anomalies', rationale: 'DBSCAN explicitly separates noise (-1) without forcing outliers into valid clusters.' },
        { application: 'Detect subtle multivariate sensor faults inside varying regimes', dataCondition: 'Multi-mode operating data with local density pockets', recommended: 'Local Outlier Factor (LOF)', alternative: 'Isolation Forest + One-Class SVM consensus (Complete mode)', rationale: 'LOF adapts to each operating regime’s local density.' },
        { application: 'Discover interpretable IF-THEN failure/risk rules', dataCondition: 'Mixed categorical/binned numeric attributes', recommended: 'Apriori Association Rules + Subgroup Discovery (WRAcc)', alternative: 'Decision Tree rules in Studio 03/06', rationale: 'Produces directly readable business/engineering rules with Support, Confidence & Lift.' },
      ],
    },
  },

  // =========================================================================
  // CHAPTER 3 — STUDIO 03
  // =========================================================================
  {
    studioId: '03',
    chapterNum: 'Chapter 03',
    title: 'Universal Diagnostics, Ranked Baseline Equations & Tail Risk',
    subtitle: 'Distributional assumption testing, multicollinearity (VIF), financial/operational tail risk (VaR/CVaR), and ranked Parametric + Non-Parametric mathematical baselines.',
    overview:
      'Before running black-box AutoML or Deep Learning, rigorous data science requires answering three questions: (1) Do my features satisfy parametric assumptions (normality, low collinearity), or is my data skewed/nonlinear? (2) What is my downside tail risk? (3) What exact mathematical performance does a transparent baseline achieve over a zero-skill naive floor? Studio 03 answers all three automatically, ranking both Parametric (Ridge, Logistic, Decision Stump) and Non-Parametric Analytical baselines (B-Spline GAM, Nadaraya-Watson, Weighted k-NN) with explicit acceptance/rejection criteria and one-click LaTeX & Python formulas.',
    realWorldApplications: [
      'Scientifically proving whether linear models suffice or nonlinear/non-parametric models are required.',
      'Detecting collinear redundant sensors/financial factors (VIF > 10) that destabilize linear regression coefficients.',
      'Quantifying worst-case operational/financial downside via 95% and 99% Value-at-Risk (VaR) and Expected Shortfall (CVaR).',
      'Establishing mandatory zero-skill Naive Floors and transparent closed-form Baseline Equations that every Studio 06 model must beat.',
    ],
    stepByStep: [
      'Choose execution mode: Complete (full diagnostics + all baselines + risk), Auto (smart suite), or Custom (pick exact tests/criteria).',
      'Set your significance level α (standard 0.05), Holdout Ratio (default 0.20), and multiple-testing correction (Benjamini-Hochberg FDR or Bonferroni).',
      'Click "Execute Universal Diagnostics".',
      'Review the Ranked Baseline Equations section: check Rank #1 Champion Baseline, Lift vs Naive Floor, symbolic coefficient pills, and copy LaTeX or Python formulas.',
      'Check the VIF table for collinearity (> 10) and Tail Risk table (VaR / CVaR).',
    ],
    provenanceRule:
      'Read-only on Active Data. Automatically registers fitted baseline models into the session Baseline Catalog so Studio 06 (Lift vs Baseline), Studio 12 (Executive Reports), and Studio 14 (Supervisory Reference Baseline) can compare against them directly.',
    methods: [
      {
        name: 'Variance Inflation Factor (VIF) Multicollinearity Audit',
        paradigm: 'Linear Dependency Diagnostic',
        whatItDoes:
          'Regresses each numeric predictor x_j against all other predictors to measure how much its estimated regression coefficient variance is inflated by redundancy. High VIF makes individual linear slopes unstable and flips signs unpredictably.',
        formula: {
          equation: 'VIF_j = 1 / (1 − R_j²)',
          symbols: 'R_j² = coefficient of determination when regressing predictor x_j on all other predictors x_{-j}.',
        },
        whenToUse: 'Always check before interpreting linear coefficients or fitting unregularized OLS/Logistic models.',
        recommendedParams: 'VIF < 5 safe; 5–10 moderate caution; > 10 drop/combine redundant predictors or rely on Ridge/GAM regularization.',
        pitfalls: 'High VIF harms coefficient interpretability much more than it harms regularized prediction accuracy (Ridge/GBDT handle collinearity gracefully).',
      },
      {
        name: 'Value-at-Risk (VaR) & Conditional Value-at-Risk (CVaR / Expected Shortfall)',
        paradigm: 'Quantitative Tail Risk Analysis',
        whatItDoes:
          'VaR_α gives the loss/deviation threshold exceeded only (1 − α)% of the time. CVaR_α (Expected Shortfall) computes the AVERAGE severity across all extreme events beyond VaR — capturing catastrophic heavy-tail risk that VaR misses.',
        formula: {
          equation: 'VaR_α = −F⁻¹(1 − α)     |     CVaR_α = −E[X | X ≤ −VaR_α] = (1 / (1 − α)) ∫₀^{1−α} VaR_u du',
          symbols: 'α ∈ {0.95, 0.99} confidence level; F⁻¹ = empirical quantile function.',
        },
        whenToUse: 'Financial portfolio returns, energy peak-load spikes, insurance claims, SLA latency breaches, and engineering stress margins.',
        recommendedParams: 'Report both 95% (routine operational stress) and 99% (extreme crisis tail) VaR & CVaR.',
        pitfalls: 'Parametric Gaussian VaR severely underestimates crash risk on heavy-tailed (high kurtosis) distributions — always prefer Empirical/Historical CVaR.',
      },
      {
        name: 'Ranked Parametric vs. Non-Parametric Baselines & Lift over Naive Floor',
        paradigm: 'Scientific Benchmarking',
        whatItDoes:
          'Trains a zero-skill Naive Floor (Mean/Median for regression; Most-Frequent/Stratified for classification), exact parametric models (Ridge/Logistic), interpretable rule baselines (Decision Stump), and fast distribution-free Non-Parametric baselines (B-Spline GAM, Nadaraya-Watson, Weighted k-NN) on a strict holdout split, then ranks them explicitly with acceptance rationale.',
        formula: {
          equation: 'Lift_vs_Naive = Score_model − Score_naive    |    Ridge: β̂ = (XᵀX + λI)⁻¹ Xᵀy    |    Logistic: P(y=1|x) = 1 / (1 + e^{−(β₀ + βᵀx)})',
          symbols: 'Positive Lift proves genuine predictive signal above trivial constant guessing; λ > 0 stabilizes collinear features.',
        },
        whenToUse: 'Mandatory on every modeling project before invoking complex AutoML or Neural Networks.',
        recommendedParams: 'Default Holdout = 0.20 (or uses Split Lock partition automatically when active).',
        pitfalls: 'If a non-parametric baseline dramatically beats Ridge/Logistic here, your data is nonlinear — immediately inspect Studio 06 Tab 2 (Non-Parametric Analytical) & AutoML!',
      },
      {
        name: 'Multiple-Testing p-Value Corrections (Benjamini-Hochberg FDR vs. Bonferroni / Holm)',
        paradigm: 'False-Discovery Governance',
        whatItDoes:
          'Adjusts raw p-values when running dozens of simultaneous statistical tests so random chance does not produce bogus "significant" findings. Benjamini-Hochberg controls expected False Discovery Rate (FDR); Bonferroni & Holm control strict Family-Wise Error Rate (FWER).',
        formula: {
          equation: 'Benjamini-Hochberg: p_(i) ≤ (i / m)·q    |    Bonferroni: α_adj = α / m    |    Holm: p_(i) ≤ α / (m − i + 1)',
          symbols: 'm = total number of simultaneous tests; p_(1) ≤ .. ≤ p_(m) sorted p-values; q = target FDR (e.g. 0.05).',
        },
        whenToUse: 'Benjamini-Hochberg FDR for exploratory screening batteries; Holm/Bonferroni for strict confirmatory regulatory or clinical claims.',
        recommendedParams: 'Default FDR q = 0.05 for Studio 03 Complete battery.',
        pitfalls: 'Bonferroni becomes overly conservative when m > 50, hiding real signals (high Type-II false-negative error).',
      },
    ],
    thresholdTables: [
      {
        title: 'Studio 03 Diagnostic, Collinearity & Baseline Lift Thresholds',
        rows: [
          { metric: 'VIF (Multicollinearity)', range: '< 5 Safe | 5–10 Moderate | > 10 Severe', level: 'warn', interpretation: 'Variance inflation from redundant predictors.', action: 'Drop/merge columns with VIF > 10 or use Ridge/Lasso/GAM.' },
          { metric: 'Baseline Lift vs Naive Floor', range: '> +0.10 Strong Signal | +0.02–0.10 Modest | <= 0 No Skill', level: 'good', interpretation: 'Gain in R² (regression) or Weighted F1 (classification) over constant prediction.', action: 'If Lift <= 0, inspect target leakage/noise or switch to nonlinear models.' },
          { metric: 'Non-Parametric vs Linear Gap', range: 'Δ > +0.05 favors Non-Parametric', level: 'info', interpretation: 'Quantifies genuine nonlinearity present in the response surface.', action: 'Proceed to Studio 06 Non-Parametric Analytical & AutoML instead of linear-only models.' },
          { metric: 'Excess Kurtosis', range: '< 1 Normal-like | 1–3 Moderate Tails | > 3 Heavy / Fat Tails', level: 'warn', interpretation: 'Frequency of extreme outliers relative to a Gaussian bell curve.', action: 'When > 3, rely on CVaR, MAD, Spearman, and robust/non-parametric models.' },
        ],
      },
    ],
    decisionMatrix: {
      title: 'Studio 03 Diagnostic Verdict -> Next Action Matrix',
      rows: [
        { application: 'Normality p >= 0.05 AND Ridge Baseline beats Non-Parametric', dataCondition: 'Gaussian errors, linear structure, VIF < 5', recommended: 'Linear Ridge / Logistic Equation is optimal & parsimonious', alternative: 'Verify tiny extra lift via Studio 06 AutoML', rationale: 'Simple exact linear formula achieves peak bias-variance tradeoff.' },
        { application: 'Normality p < 0.05 OR Non-Parametric Baseline beats Ridge by > 0.05', dataCondition: 'Skewed/curved response surface', recommended: 'Studio 06 Non-Parametric Analytical (B-Spline GAM / RBF RKHS) + AutoML', alternative: 'Studio 05 Yeo-Johnson + Interaction features', rationale: 'Proves linear functional form underfits true data physics.' },
        { application: 'Severe multicollinearity (VIF > 10) across sensors/factors', dataCondition: 'Redundant correlated predictors', recommended: 'Studio 05 4-Method Consensus Feature Selection or L1 Lasso', alternative: 'PCA / Ridge Regularization', rationale: 'Stabilizes equations and reduces overfitting variance.' },
      ],
    },
  },

  // =========================================================================
  // CHAPTER 4 — STUDIO 04
  // =========================================================================
  {
    studioId: '04',
    chapterNum: 'Chapter 04',
    title: 'Automated EDA & Hypothesis Lab',
    subtitle: 'Exploratory distributions, correlation heatmaps, and statistical hypothesis tests.',
    overview:
      'Studio 04 combines exploratory visualizations with tests matched to the sampling design. Evaluated battery tests report raw and Benjamini–Hochberg adjusted p-values. Tests with insufficient observations or undefined results have a recorded reason.',
    realWorldApplications: [
      'A/B product, marketing, or industrial process change testing (Welch t-test vs Mann-Whitney U).',
      'Before-vs-after calibration, maintenance, or medical intervention validation (Paired t-test vs Wilcoxon Signed-Rank).',
      'Multi-group shift, supplier, regional, or treatment comparison (One-Way ANOVA vs Kruskal-Wallis).',
      'Checking sensor variance stability across lines (Levene / Fligner-Killeen) and categorical association (Chi-Square / Fisher Exact).',
    ],
    stepByStep: [
      'Section 1 (Automated EDA): Click "Run Automated EDA" to generate interactive distributions, correlation matrix heatmap, missingness bars, and automated insights.',
      'Section 2 (Hypothesis Lab): Choose Complete for a broader battery, Auto for a smaller battery, or Custom for a specific method, columns and significance level.',
      'Check the sampling assumptions and read the adjusted p-values for battery decisions. Review effect sizes or group statistics where the method reports them.',
    ],
    provenanceRule: 'Read-only on Active Data. Both Automated EDA and Hypothesis Lab results persist independently in cache and flow into Studio 12 Executive Reports.',
    methods: [
      {
        name: 'Normality Tests: Shapiro-Wilk, D’Agostino K², Anderson-Darling & 1-Sample KS',
        paradigm: 'Goodness-of-Fit (Tests 1–5)',
        whatItDoes:
          'Tests whether a single numeric variable could plausibly come from a Gaussian (normal) bell distribution. Shapiro-Wilk correlates ordered sample quantiles against expected normal order statistics; D’Agostino K² combines standardized skewness and kurtosis; Anderson-Darling places heavy quadratic weight on distribution tails; KS measures maximum CDF deviation.',
        nullHypothesis: 'H₀: The sample is drawn from a Normal distribution 𝒩(μ, σ²). (p < α => Reject normality -> use non-parametric tests).',
        formula: {
          equation: 'Shapiro-Wilk: W = (Σ a_i x_(i))² / Σ (x_i − x̄)²    |    K² = Z₁(skew)² + Z₂(kurt)²    |    A² = −n − (1/n) Σ (2i−1)[ln F(Y_i) + ln(1−F(Y_{n+1−i}))]',
          symbols: 'x_(i) = sorted values; a_i = expected normal order-statistic weights; W close to 1 indicates normality.',
        },
        whenToUse: 'Shapiro-Wilk for n <= 5,000 (highest power); D’Agostino K² for larger samples; Anderson-Darling when tail accuracy (risk/SPC) matters most.',
        recommendedParams: 'α = 0.05; for n > 10,000 always inspect skewness/histogram too because tiny trivial deviations yield p < 0.05.',
        pitfalls: 'Failing to reject H₀ at tiny n < 20 does not prove normality — it often just means low statistical power.',
      },
      {
        name: "Welch's Two-Sample t-Test vs. Mann-Whitney U Test",
        paradigm: '2 Independent Groups (Tests 6–7)',
        whatItDoes:
          "Compares two separate populations (e.g., Machine A vs Machine B). Welch's t-test tests equality of group means WITHOUT assuming equal variances (using Welch-Satterthwaite degrees of freedom). Mann-Whitney U ranks all combined observations and tests whether one group stochastically tends to produce larger values than the other — completely immune to skew and extreme outliers.",
        nullHypothesis: 'H₀ (Welch): μ₁ = μ₂   |   H₀ (Mann-Whitney): P(X₁ > X₂) = 0.50 (identical stochastic dominance).',
        formula: {
          equation: 'Welch: t = (x̄₁ − x̄₂) / √(s₁²/n₁ + s₂²/n₂)    |    Cohen d = (x̄₁ − x̄₂) / s_pooled    |    U = R₁ − n₁(n₁+1)/2',
          symbols: 'x̄₁, x̄₂ = group means; s₁², s₂² = group variances; R₁ = sum of ranks in group 1.',
        },
        whenToUse: "Use Welch's t-test when both groups are roughly normal (or n >= 30 without extreme skew); use Mann-Whitney U when skewed, ordinal, or outlier-prone.",
        assumptions: 'Independent observations across groups (for repeated measurements on the same unit, use Paired t / Wilcoxon instead).',
        recommendedParams: 'Report p-value alongside Cohen d (Welch) or rank-biserial correlation (Mann-Whitney).',
        pitfalls: 'Running an independent test on paired before/after data wastes huge statistical power.',
      },
      {
        name: 'Paired t-Test vs. Wilcoxon Signed-Rank Test',
        paradigm: '2 Paired / Repeated Measures (Tests 8–9)',
        whatItDoes:
          'Compares two aligned numeric measurements taken on the SAME unit (e.g., before vs after maintenance, Sensor 1 vs Sensor 2 on the same timestamp) by analyzing within-row differences d_i = x_i − y_i. Paired t tests mean difference = 0; Wilcoxon ranks |d_i| and compares positive vs negative rank sums.',
        nullHypothesis: 'H₀ (Paired t): μ_d = 0   |   H₀ (Wilcoxon): Median(d_i) = 0 (symmetric difference distribution around zero).',
        formula: {
          equation: 'Paired t = d̄ / (s_d / √n)    |    Wilcoxon: W = min(Σ_{d_i>0} rank(|d_i|),  Σ_{d_i<0} rank(|d_i|))',
          symbols: 'd_i = x_i − y_i paired difference on row i; d̄ = mean difference; s_d = std of differences.',
        },
        whenToUse: 'Paired t when differences d_i are approximately normal; Wilcoxon Signed-Rank when differences are skewed or have outliers.',
        recommendedParams: 'Requires 2 aligned numeric columns with matched rows.',
        pitfalls: 'Requires the difference distribution (not necessarily the raw columns individually) to be roughly normal for Paired t.',
      },
      {
        name: 'One-Way ANOVA (F-Test) vs. Kruskal-Wallis H Test & 2-Sample KS Test',
        paradigm: '3+ Groups & Full-Distribution Shift (Tests 10–12)',
        whatItDoes:
          'ANOVA compares between-group variance against within-group noise to test if 3+ group means are all equal. Kruskal-Wallis is its non-parametric rank counterpart comparing group rank averages. Two-Sample KS detects ANY difference between two distributions (location, spread, skewness, or bimodality).',
        nullHypothesis: 'H₀ (ANOVA): μ₁ = μ₂ = .. = μ_k   |   H₀ (Kruskal): All k distributions identical   |   H₀ (KS): F₁(x) = F₂(x) ∀x.',
        formula: {
          equation: 'ANOVA: F = MS_between / MS_within,  η² = SS_between / SS_total    |    H = [12/(N(N+1))] Σ(R_j²/n_j) − 3(N+1)    |    KS: D = sup_x |F₁(x) − F₂(x)|',
          symbols: 'η² (eta-squared) = proportion of total variance explained by group membership; R_j = rank sum of group j.',
        },
        whenToUse: 'ANOVA for 3+ roughly normal homoscedastic groups; Kruskal-Wallis for skewed/non-normal groups; 2-Sample KS for distribution drift checks.',
        recommendedParams: 'Verify equal variances via Levene test before trusting classical ANOVA; if Levene p < 0.05, prefer Kruskal-Wallis.',
        pitfalls: 'Significant ANOVA/Kruskal tells you AT LEAST ONE group differs — inspect group means/medians to see which.',
      },
      {
        name: 'Variance Homogeneity: Levene, Fligner-Killeen & Bartlett Tests',
        paradigm: 'Spread / Homoscedasticity (Tests 13–15)',
        whatItDoes:
          'Tests whether multiple groups exhibit the same variance/spread (e.g., checking if Machine Line B has noisier tolerances than Line A, or verifying ANOVA assumptions). Levene tests absolute deviations from group medians; Fligner-Killeen ranks deviations (most outlier-robust); Bartlett uses pooled log-variances.',
        nullHypothesis: 'H₀: σ₁² = σ₂² = .. = σ_k² (equal group variances).',
        formula: {
          equation: 'Levene: W = [(N−k)/(k−1)] · [Σ n_i(Z̄_i· − Z̄··)² / Σ Σ (Z_ij − Z̄_i·)²],   Z_ij = |Y_ij − median(Y_i)|',
          symbols: 'k = number of groups; Z_ij = absolute deviation from group i median.',
        },
        whenToUse: 'Use Levene (default) or Fligner-Killeen (heavy tails). Use Bartlett ONLY if normality is strictly confirmed.',
        recommendedParams: 'α = 0.05.',
        pitfalls: 'Bartlett throws false positives aggressively whenever data has mild kurtosis or skewness.',
      },
      {
        name: "Associations: Pearson r, Spearman ρ, Kendall τ, Chi-Square χ² & Fisher's Exact",
        paradigm: 'Bivariate Relationship & Independence (Tests 16–17+)',
        whatItDoes:
          'Quantifies and tests dependence between two variables. Pearson measures straight-line correlation; Spearman correlates ranks (any monotonic curve); Kendall counts concordant vs discordant pairs (robust at small n & ties); Chi-Square & Fisher Exact test whether two categorical variables are independent in a contingency table.',
        nullHypothesis: 'H₀: Variables are linearly/monotonically uncorrelated (r=0, ρ=0, τ=0) or categorically independent.',
        formula: {
          equation: 'Pearson r = Cov(X,Y)/(s_X s_Y)   |   Spearman ρ = 1 − 6Σd_i²/[n(n²−1)]   |   τ = (C−D)/C(n,2)   |   χ² = Σ (O_ij − E_ij)²/E_ij,  V = √(χ² / (N·min(r−1,c−1)))',
          symbols: 'C, D = concordant & discordant pairs; O_ij, E_ij = observed & expected contingency counts; V = Cramér’s V effect size [0, 1].',
        },
        whenToUse: 'Pearson for linear Gaussian; Spearman/Kendall for curved or outlier-prone numeric pairs; Chi-Square for categorical tables (expected >= 5); Fisher Exact for sparse 2×2 tables (expected < 5).',
        recommendedParams: 'Always check Cramér’s V alongside Chi-Square p-value because huge N makes tiny associations statistically significant.',
        pitfalls: 'Correlation NEVER proves causation (use Studio 08 Causal AIPW) and r = 0 does NOT exclude U-shaped nonlinear dependence.',
      },
    ],
    thresholdTables: [
      {
        title: 'Hypothesis Test Effect-Size Benchmarks (Practical Significance)',
        rows: [
          { metric: "Cohen's d (t-Tests)", range: '0.20 Small | 0.50 Medium | >= 0.80 Large', level: 'good', interpretation: 'Mean gap expressed in pooled standard deviations.', action: 'Avoid acting on p < 0.05 if |d| < 0.15 unless cost of intervention is near zero.' },
          { metric: 'Eta-Squared η² (ANOVA)', range: '0.01 Small | 0.06 Medium | >= 0.14 Large', level: 'good', interpretation: 'Fraction of outcome variance explained by group factor.', action: 'η² >= 0.06 signals a practically meaningful grouping factor.' },
          { metric: "Cramér's V (Chi-Square)", range: '0.10 Small | 0.30 Medium | >= 0.50 Large', level: 'good', interpretation: 'Strength of categorical association normalized to [0, 1].', action: 'Focus on V >= 0.20 for business/engineering segmentation.' },
          { metric: 'Pearson r / Spearman ρ / Kendall τ', range: '0.10–0.29 Weak | 0.30–0.49 Moderate | 0.50–0.69 Strong | >= 0.70 Very Strong', level: 'info', interpretation: 'Monotonic/linear relationship magnitude.', action: 'Large gap between |ρ| and |r| reveals strong monotonic curvature.' },
        ],
      },
    ],
    decisionMatrix: {
      title: 'Studio 04 Test Selector',
      rows: [
        { application: 'Test normality of 1 numeric column', dataCondition: 'n <= 5,000 (or > 5,000)', recommended: 'Shapiro-Wilk (or D’Agostino K²)', alternative: 'Anderson-Darling (heavy tails)', rationale: 'Determines whether parametric or non-parametric workflow applies.' },
        { application: 'Compare 2 independent groups', dataCondition: 'Normal vs Skewed/Outliers', recommended: "Welch's t-Test (Normal)", alternative: 'Mann-Whitney U (Skewed)', rationale: 'Robust against unequal variance (Welch) or non-normality (Mann-Whitney).' },
        { application: 'Compare paired before/after readings', dataCondition: '2 aligned numeric columns', recommended: 'Paired t-Test (Normal diffs)', alternative: 'Wilcoxon Signed-Rank (Skewed diffs)', rationale: 'Eliminates subject-level baseline noise.' },
        { application: 'Compare 3+ independent groups', dataCondition: 'Normal + Levene p >= 0.05', recommended: 'One-Way ANOVA', alternative: 'Kruskal-Wallis H Test', rationale: 'Avoids multiple-comparison inflation across group pairs.' },
        { application: 'Test association of 2 categorical columns', dataCondition: 'All expected counts >= 5 vs < 5', recommended: 'Chi-Square + Cramér’s V', alternative: "Fisher's Exact Test (2×2 sparse)", rationale: 'Exact hypergeometric probability prevents sparse-cell distortion.' },
      ],
    },
  },

  // =========================================================================
  // CHAPTER 5 — STUDIO 05
  // =========================================================================
  {
    studioId: '05',
    chapterNum: 'Chapter 05',
    title: 'Feature Engineering & 4-Method Consensus Selection',
    subtitle: 'Skewness power transforms, pairwise interactions, polynomials, cyclical datetime encodings, and bias-resistant multi-method feature selection.',
    overview:
      'Model accuracy often improves more from clean domain-informative features and removal of noisy predictors than from hyperparameter tuning alone. Studio 05 provides two tightly controlled stages: (1) Feature Engineering to linearize skewed variables, capture multiplicative physics (interactions/polynomials), and encode periodic calendar cycles; and (2) Feature Selection using 4 complementary scoring families aggregated via Consensus Ranking to retain high-signal predictors.',
    realWorldApplications: [
      'Stabilizing right-skewed financial amounts, wait times, or vibration amplitudes via Yeo-Johnson transforms.',
      'Creating ratio/product terms (e.g., Pressure × Temperature, Income / Debt) that unlock linear and GAM models.',
      'Encoding hour-of-day, day-of-week, and month-of-year continuously without artificial midnight/December jumps (sin/cos).',
      'Pruning 50–500 raw sensor tags down to the top 10–25 highest-signal variables to cut overfitting and inference latency.',
    ],
    stepByStep: [
      'Section 1 (Feature Engineering): Pick Auto (skew + interactions + datetime), Complete (+ polynomials), or Custom checkboxes. Click "Run Feature Engineering" (toggle Commit to add columns to Active Data).',
      'Section 2 (Feature Selection): Choose Complete (4-Method Consensus: Mutual Information + F-Test + Tree Importance + L1 Lasso), Auto, or Custom single method + Top-K count.',
      'Inspect consensus rankings and optionally commit exclusions (all unselected features are reversibly masked in Topbar -> Exclusions).',
    ],
    provenanceRule:
      'Committing Feature Engineering updates Active Data (ENGINEERED provenance) while preserving upstream S01–S05 caches. Feature Selection updates Excluded Columns reversibly without destroying any raw column.',
    methods: [
      {
        name: 'Yeo-Johnson Skewness Correction & Cyclical Datetime Encoding',
        paradigm: 'Distribution & Temporal Transforms',
        whatItDoes:
          'Yeo-Johnson generalizes Box-Cox power transforms to work seamlessly on zero and negative numbers, finding optimal exponent λ to make skewed features approximately symmetric. Cyclical encoding maps periodic clock/calendar indices onto a unit circle so 23:00 is adjacent to 00:00.',
        formula: {
          equation: 'Yeo-Johnson (x ≥ 0): ψ(λ, x) = [(x+1)^λ − 1]/λ   |   Cyclical: x_sin = sin(2π·t/T),  x_cos = cos(2π·t/T)',
          symbols: 'λ estimated by maximum likelihood; T = cycle period (24h, 7d, 12m, 365d).',
        },
        whenToUse: 'Apply Yeo-Johnson whenever numeric skewness |g₁| > 1.0 before linear/KNN/kernel/neural models; apply cyclical encoding to any timestamp column.',
        recommendedParams: 'Default skew threshold |g₁| > 0.75; keep polynomial degree = 2 (degree 3 explodes dimensionality rapidly).',
        pitfalls: 'Tree models (GBDT/RF) are invariant to monotonic transforms like Yeo-Johnson, though linear/GAM/KNN/NN benefit significantly.',
      },
      {
        name: 'Mutual Information (MI) vs. ANOVA F-Test vs. Tree Importance vs. L1 Lasso Consensus',
        paradigm: '4-Method Feature Selection',
        whatItDoes:
          'Combines four fundamentally different mathematical lenses: F-Test captures linear correlation with target; Mutual Information detects ANY nonlinear/non-monotonic dependency via entropy reduction; Random Forest Importance captures multi-feature tree splits & interactions; L1 Lasso enforces sparse shrinkage driving redundant correlated weights to exact zero.',
        formula: {
          equation: 'MI: I(X; Y) = Σ_{x,y} p(x,y) log[p(x,y) / (p(x)p(y))]    |    Lasso: min_β  ‖y − Xβ‖₂² + λ‖β‖₁',
          symbols: 'I(X; Y) = 0 iff X and Y are completely independent (even nonlinearly); ‖β‖₁ = Σ|β_j| induces exact sparsity.',
        },
        whenToUse: 'Use 4-Method Consensus (Complete mode) as the gold standard so neither linear-only nor tree-only selection biases dominate.',
        recommendedParams: 'Keep top K ≈ min(25, max(5, N / 20)) features to maintain strong samples-per-feature ratio.',
        pitfalls: 'Feature selection cannot detect target leakage on its own — a leaked column will rank #1! Always audit top-ranked features for domain plausibility.',
      },
    ],
    thresholdTables: [
      {
        title: 'Feature Engineering & Selection Recommended Guidelines',
        rows: [
          { metric: 'Samples-to-Features Ratio (N / p)', range: '>= 20 Healthy | 10–19 Caution | < 10 High Overfit Risk', level: 'good', interpretation: 'Statistical degrees of freedom per predictor.', action: 'When N / p < 10, run Studio 05 Selection before AutoML / KNN / Kernel Ridge.' },
          { metric: 'Feature Skewness |g₁|', range: '> 1.0 Apply Yeo-Johnson | <= 0.5 No Transform Needed', level: 'info', interpretation: 'Benefit threshold for power transformation.', action: 'Correct severe skew before Ridge, Logistic, KNN, or Neural Nets.' },
          { metric: 'Near-Zero Variance', range: 'Std ≈ 0 or > 99% single value', level: 'bad', interpretation: 'Carries virtually zero discriminatory information.', action: 'Eliminate via Variance Threshold or Studio 01 Drop Constant Columns.' },
        ],
      },
    ],
    decisionMatrix: {
      title: 'Studio 05 Method Selection Guide',
      rows: [
        { application: 'General-purpose unbiased predictor ranking', dataCondition: 'Mixed linear & nonlinear relationships', recommended: '4-Method Consensus (Complete mode)', alternative: 'Mutual Information + Tree Importance', rationale: 'Averages normalized ranks across linear, information-theoretic, sparse, and interaction criteria.' },
        { application: 'Detecting curved / U-shaped / threshold relationships', dataCondition: 'Suspected strong nonlinearity where Pearson r ≈ 0', recommended: 'Mutual Information (MI)', alternative: 'Random Forest Permutation Importance', rationale: 'MI captures arbitrary statistical dependence beyond linear slopes.' },
        { application: 'Removing redundant collinear copies among 50+ correlated sensors', dataCondition: 'High VIF > 10 across sensor banks', recommended: 'L1 Lasso Selection', alternative: 'Manual VIF pruning + Consensus', rationale: 'L1 penalty zeros out redundant duplicates while preserving the strongest representative.' },
      ],
    },
  },

  // =========================================================================
  // CHAPTER 6 — STUDIO 06
  // =========================================================================
  {
    studioId: '06',
    chapterNum: 'Chapter 06',
    title: 'Multi-Backend AutoML, Non-Parametric Analytical & Deep Learning',
    subtitle: '4 Native SOTA AutoML engines + 5 External adapters, 5 true Non-Parametric Analytical Models with closed-form equations & GAM curves, 8 Paper-Grade Deep Learning architectures, and Predictions.',
    overview:
      'Studio 06 is OmniForge’s unified modeling powerhouse designed around one non-negotiable scientific rule: EVERY complex model is compared head-to-head against strong baselines with explicit Lift over Baseline and mathematical transparency. Whether you need peak tabular accuracy (AutoML), assumption-free smooth mathematical formulas (Non-Parametric Analytical Models), or modern neural representation learning (FT-Transformer, TabNet, ResNet, GNN), all live in one cohesive workbench.',
    realWorldApplications: [
      'High-accuracy predictive maintenance, demand/pricing estimation, credit default, and medical risk scoring (AutoML).',
      'Scientific, engineering, and regulatory modeling requiring exact distribution-free equations and inspectable per-feature curves (Penalized B-Spline GAM, Nadaraya-Watson, RBF Kernel Ridge, Isotonic).',
      'Large-scale tabular attention learning, sparse feature-mask neural modeling, spectral vectors, and graph-coupled telemetry (Deep Learning).',
      'Interactive what-if single-record scoring and target-free batch CSV production scoring.',
    ],
    stepByStep: [
      'Tab 1 (Multi-Backend AutoML Hub): Select Native Stacking, Multi-Layer Stacker, Evolutionary Pipeline, Fast GBDT, or any installed external backend (FLAML/AutoGluon/H2O/TPOT), configure time budget & CV folds, and click "Launch AutoML Tournament". Inspect Leaderboard, Lift vs Baseline, and Champion Equation.',
      'Tab 2 (Non-Parametric Analytical Models): Pick Auto, Complete (all 5 models + parametric Ridge/Logistic reference), or Custom hyperparameters (knots, bandwidth h, k neighbors, RBF γ). Inspect Leaderboard, Additive B-Spline GAM Decomposition table, and Exact Equations with LaTeX & Python copy.',
      'Tab 3 (Deep Learning): Select architectures (FT-Transformer, TabNet, ResNet-Tabular, MLP, 1D-CNN, GCN/GAT/GraphSAGE) and epochs, then benchmark against baselines with live training loss curves.',
      'Tab 4 (Predictions): Score a single custom input record or upload a target-free CSV for batch scoring.',
    ],
    provenanceRule:
      'Read-only on Active Data. Automatically promotes the Champion model to Active Model (used by Studio 07 Explainability, Studio 11 Registry, Studio 12 Reporting) and registers all AutoML, Non-Parametric, and DL models in Studio 14’s Live Supervisory Model Catalog.',
    methods: [
      {
        name: 'Penalized B-Spline GAM (Generalized Additive Model)',
        paradigm: 'Non-Parametric Analytical (Tab 2)',
        whatItDoes:
          'Replaces rigid straight-line slopes β_j·x_j with smooth piecewise cubic B-spline curves f_j(x_j) for every feature, regularized by an integrated second-derivative roughness penalty so curves bend where data demands without wiggly overfitting. Delivers BOTH nonlinear accuracy AND full per-feature analytical transparency.',
        formula: {
          equation: 'ŷ(x) = β₀ + Σ_{j=1}^p f_j(x_j) = β₀ + Σ_{j=1}^p Σ_{k=1}^{K_j} w_{jk} B_{jk}(x_j),     min Σ(y_i − ŷ_i)² + λ Σ_j ∫ [f_j″(t)]² dt',
          symbols: 'B_{jk}(x_j) = cubic B-spline basis function k for feature j; w_{jk} = exact spline weights; λ = smoothness penalty.',
        },
        whenToUse: 'FIRST CHOICE when data is curved/non-normal AND you need transparent mathematical equations and visual per-feature effect curves.',
        recommendedParams: 'n_knots = 5–7 (more knots for complex curves when N > 500; fewer knots = 4 for small N < 200).',
        pitfalls: 'Purely additive across features — if strong multi-variable multiplicative interactions dominate, compare against RBF Kernel Ridge or AutoML.',
      },
      {
        name: 'Nadaraya-Watson Gaussian Kernel Smoother',
        paradigm: 'Non-Parametric Analytical (Tab 2)',
        whatItDoes:
          'Computes the exact locally weighted conditional expectation E[Y | X = x] by placing a smooth Gaussian kernel bell of bandwidth h around query point x, giving exponentially higher weight to closer training observations. Requires ZERO parametric shape assumptions.',
        formula: {
          equation: 'ŷ(x) = [ Σ_{i=1}^n K_h(‖x − x_i‖₂) · y_i ] / [ Σ_{i=1}^n K_h(‖x − x_i‖₂) ],     K_h(u) = exp(−u² / (2h²))',
          symbols: 'h = cross-validated kernel bandwidth (smaller h = sharper local fit; larger h = smoother global average).',
        },
        whenToUse: 'Low-to-moderate dimensions (p <= 12) with smooth local bumps, nonlinear regimes, and zero prior parametric assumptions.',
        recommendedParams: 'Leave bandwidth blank for automatic Cross-Validated Silverman/CV bandwidth selection.',
        pitfalls: 'Suffers from curse of dimensionality when p > 20 — apply Studio 05 Feature Selection first.',
      },
      {
        name: 'RBF Kernel Ridge Regression (RKHS)',
        paradigm: 'Non-Parametric Analytical (Tab 2)',
        whatItDoes:
          'Solves a closed-form linear system inside an infinite-dimensional Reproducing Kernel Hilbert Space (RKHS) induced by the Radial Basis Function (Gaussian) kernel — capturing smooth arbitrary multi-way feature interactions analytically without gradient descent local minima.',
        formula: {
          equation: 'ŷ(x) = Σ_{i=1}^n α_i · exp(−γ ‖x − x_i‖₂²),     α = (K + λI)⁻¹ y',
          symbols: 'K_ij = exp(−γ ‖x_i − x_j‖₂²) Gram kernel matrix; γ = inverse squared length-scale; λ = ridge regularization.',
        },
        whenToUse: 'Gold-standard smooth nonlinear physics/engineering response surfaces with multi-feature interactions on small-to-medium data (N <= 5,000).',
        recommendedParams: 'Leave γ blank for automatic median-heuristic / CV tuning.',
        pitfalls: 'Exact matrix inversion scales O(n³) — OmniForge safely caps/subsamples training support points for large datasets.',
      },
      {
        name: 'Distance-Weighted k-NN & Isotonic Monotonic Regression (PAVA)',
        paradigm: 'Non-Parametric Analytical (Tab 2)',
        whatItDoes:
          'Distance-Weighted k-NN averages the k nearest standardized training neighbors weighted by inverse distance (1 / (d + ε)), adapting freely to multimodal surfaces. Isotonic Regression solves the exact Pool Adjacent Violators Algorithm (PAVA) to produce the optimal strictly monotonic (non-decreasing or non-increasing) step function.',
        formula: {
          equation: 'Weighted k-NN: ŷ(x) = Σ_{i∈N_k(x)} w_i y_i / Σ w_i,  w_i = 1/(d_i + ε)    |    PAVA: min Σ(y_i − ŷ_i)²  s.t.  ŷ₁ ≤ ŷ₂ ≤ .. ≤ ŷ_n',
          symbols: 'N_k(x) = k nearest neighbors; PAVA guarantees zero physically impossible reversals.',
        },
        whenToUse: 'Weighted k-NN for multimodal local regimes (p <= 15). Isotonic (PAVA) whenever physics/domain rules enforce monotonicity (wear vs runtime, dose-response, probability calibration).',
        recommendedParams: 'k ≈ √N (odd integer, e.g., 7–15) for k-NN.',
        pitfalls: 'k-NN degrades with irrelevant noisy columns (always scale & select features first); Isotonic projects onto the single dominant monotonic feature.',
      },
      {
        name: 'Multi-Backend AutoML (Stacking / Multi-Layer / Evolutionary / FLAML / AutoGluon / H2O / TPOT)',
        paradigm: 'Automated Ensemble Learning (Tab 1)',
        whatItDoes:
          'Systematically searches preprocessing pipelines, diverse model families (Gradient Boosting, Random Forests, Extra Trees, Regularized Linear, MLP), and cross-validated stacking meta-learners within your time budget to maximize out-of-fold generalization.',
        formula: {
          equation: 'Stacking: ŷ_meta(x) = g( f̂₁^(OOF)(x), f̂₂^(OOF)(x), …, f̂_M^(OOF)(x) )',
          symbols: 'f̂_m^(OOF) = out-of-fold cross-validated predictions of base model m (prevents meta-learner overfitting).',
        },
        whenToUse: 'Default engine for peak tabular predictive performance on complex real-world heterogeneous datasets.',
        recommendedParams: 'cv_folds = 5 (3 for quick prototypes, 10 for small N < 300); time_budget = 30–120s.',
        pitfalls: 'Always verify positive Lift vs Studio 03 Baselines and audit Studio 07 SHAP attributions before deployment.',
      },
      {
        name: 'Paper-Grade Deep Learning (FT-Transformer, TabNet, ResNet-Tabular, MLP, 1D-CNN, GCN/GAT/GraphSAGE)',
        paradigm: 'Neural Tabular & Graph Architectures (Tab 3)',
        whatItDoes:
          'FT-Transformer tokenizes every feature and applies Multi-Head Self-Attention across features; TabNet learns sequential Sparsemax feature-selection masks at each decision step; ResNet-Tabular uses skip-connected residual blocks; 1D-CNN convolves ordered/spectral feature sequences; GCN/GAT/GraphSAGE aggregate neighbor signals across a k-NN similarity graph.',
        formula: {
          equation: 'Attention: Attn(Q,K,V) = softmax(QKᵀ / √d_k) V   |   ResNet: h_{l+1} = h_l + F(h_l)   |   GNN: h_v^(l+1) = σ(W · AGG({h_u^(l) : u ∈ N(v)}))',
          symbols: 'Q, K, V = learned query/key/value feature token projections; N(v) = graph neighborhood of observation v.',
        },
        whenToUse: 'Medium-to-large nonlinear tabular data (N >= 2,000), spectral/ordered signals (1D-CNN), or coupled relational entities (GNN).',
        recommendedParams: '20–50 epochs; compare directly against Baseline rows shown on the same benchmark leaderboard.',
        pitfalls: 'On small tabular datasets (N < 1,000), GAM / RBF Kernel Ridge / AutoML GBDT frequently generalize better than deep nets.',
      },
    ],
    thresholdTables: [
      {
        title: 'Model Performance & Generalization Evaluation Benchmarks',
        rows: [
          { metric: 'Regression R² (Holdout / CV)', range: '> 0.85 Strong | 0.65–0.85 Good | 0.35–0.65 Moderate | < 0.25 Weak', level: 'good', interpretation: 'Fraction of target variance explained out-of-sample (domain-dependent: physics > 0.90, human behavior ~0.35–0.60).', action: 'Always pair R² with RMSE / MAE in physical units.' },
          { metric: 'Classification Weighted F1 / ROC-AUC', range: 'AUC > 0.90 Excellent | 0.80–0.90 Good | 0.70–0.80 Fair | 0.50 Random', level: 'good', interpretation: 'Discriminatory separation across classes.', action: 'On imbalanced classes (< 10% positive), prioritize F1 / PR-AUC over raw Accuracy.' },
          { metric: 'Train vs Test Score Gap', range: '< 0.05 Healthy | 0.05–0.12 Mild Overfit | > 0.15 Overfitting', level: 'warn', interpretation: 'Generalization gap between training and unseen holdout performance.', action: 'If gap > 0.15, reduce feature count (Studio 05), increase regularization, or add data.' },
          { metric: 'Lift over Studio 03 Baseline', range: '> +0.03 Justifies Complexity | <= 0 Keep Simpler Baseline', level: 'info', interpretation: 'Justifies deploying a complex AutoML/DL model over a transparent linear/GAM baseline.', action: 'Occam’s Razor: if a transparent GAM/Ridge ties a black-box ensemble, deploy the transparent model!' },
        ],
      },
    ],
    decisionMatrix: {
      title: 'Studio 06 Model Family Selection Matrix',
      rows: [
        { application: 'Need highest accuracy on mixed tabular dataset', dataCondition: 'N >= 200, mixed features, complex interactions', recommended: 'Tab 1: Multi-Backend AutoML (Stacking / Multi-Layer / FLAML)', alternative: 'Tab 3: FT-Transformer (if N > 5,000)', rationale: 'Boosted & stacked ensembles dominate general tabular accuracy.' },
        { application: 'Need explicit closed-form equation on nonlinear/non-normal data', dataCondition: 'Curved effects, auditability required', recommended: 'Tab 2: Penalized B-Spline GAM', alternative: 'Tab 2: RBF Kernel Ridge (if multi-feature interactions dominate)', rationale: 'Provides exact analytical equations, LaTeX/Python code, and smooth per-feature curves.' },
        { application: 'Guaranteed monotonic physical/calibration curve', dataCondition: 'Monotonic physics constraint', recommended: 'Tab 2: Isotonic Monotonic Regression (PAVA)', alternative: 'Penalized B-Spline GAM', rationale: 'Zero non-physical slope reversals guaranteed mathematically.' },
        { application: 'Ordered spectral measurements or relational entity graphs', dataCondition: 'Spectral/sequence features or similarity-linked rows', recommended: 'Tab 3: 1D-CNN (spectral) or GCN / GAT / GraphSAGE (relational)', alternative: 'AutoML with engineered lag/graph features', rationale: 'Exploits spatial/topological inductive biases directly.' },
      ],
    },
  },

  // =========================================================================
  // CHAPTER 7 — STUDIO 07
  // =========================================================================
  {
    studioId: '07',
    chapterNum: 'Chapter 07',
    title: 'Responsible AI — Explainability, Fairness & Model Governance',
    subtitle: 'Global Permutation Importance, Local SHAP row attribution, Partial Dependence Plots (PDP), Algorithmic Fairness audits, and compliance Model Cards.',
    overview:
      'High accuracy on a test set is NOT enough for production or regulated decision-making. Studio 07 opens the black box of your Active Champion Model to verify WHICH features drive predictions globally, WHY a single specific decision or alarm occurred locally, HOW the prediction responds as one feature sweeps across its range (PDP), and WHETHER model error or approval rates systematically disadvantage protected demographic or operational groups.',
    realWorldApplications: [
      'Explaining individual loan rejections, clinical diagnoses, or turbine anomaly alerts to domain experts (Local SHAP).',
      'Catching hidden data leakage or spurious proxy features before production deployment (Global Importance + PDP).',
      'Auditing hiring, credit, insurance, and medical triage models against legal 80% Disparate Impact fairness standards.',
      'Generating standardized ISO/NIST Model Cards for enterprise governance sign-off.',
    ],
    stepByStep: [
      'Ensure an Active Model has been trained in Studio 06 (AutoML, Non-Parametric, or DL).',
      'Pick a Row Index to explain locally and click "Compute Global & Local Explanations" to view Permutation Importance, Local SHAP waterfall attributions, and Partial Dependence curve.',
      'Select a Sensitive / Protected Attribute column (e.g., gender, age group, region, facility) and click "Run Fairness Audit & Generate Model Card".',
      'Review Demographic Parity, Equal Opportunity, Disparate Impact ratio, and the auto-generated Markdown Model Card.',
    ],
    provenanceRule: 'Read-only on Active Data & Active Model. Cached explanations, fairness audits, and Model Card embed automatically into Studio 12 Executive Reports.',
    methods: [
      {
        name: 'Permutation Feature Importance',
        paradigm: 'Global Model-Agnostic Explainability',
        whatItDoes:
          'Randomly shuffles one feature column at a time on the unseen test set and measures how much model accuracy drops. A large drop proves the model heavily relies on that feature; zero or negative drop signals an irrelevant or noisy predictor.',
        formula: {
          equation: 'Importance_j = Score(y_test, f̂(X_test)) − (1/R) Σ_{r=1}^R Score(y_test, f̂(X_test^{π_{j,r}}))',
          symbols: 'X_test^{π_{j,r}} = test matrix with column j randomly permuted on repeat r; R = number of shuffles.',
        },
        whenToUse: 'Universal sanity check for ANY trained model family (tree, kernel, GAM, neural, ensemble).',
        recommendedParams: 'n_repeats = 5–10 on holdout test data (NEVER compute only on training data).',
        pitfalls: 'Strongly correlated duplicate features split permutation importance between each other — check Studio 03 VIF too.',
      },
      {
        name: 'Local SHAP (Shapley Additive Explanations) & Partial Dependence Plot (PDP)',
        paradigm: 'Local Attribution & Marginal Effect Curve',
        whatItDoes:
          'SHAP uses cooperative game theory to decompose a single row’s prediction into exact additive contributions φ_j from every feature relative to the average baseline prediction. PDP sweeps one feature across a grid while averaging predictions over all other features to reveal global linear, threshold, or U-shaped response curves.',
        formula: {
          equation: 'SHAP: f̂(x) = φ₀ + Σ_{j=1}^p φ_j(x),   φ_j = Σ_{S⊆F\\{j}} [|S|!(p−|S|−1)!/p!]·[f(S∪{j}) − f(S)]    |    PDP: f̂_S(x_S) = (1/n) Σ_i f̂(x_S, x_C^(i))',
          symbols: 'φ₀ = expected baseline prediction E[f̂(X)]; φ_j > 0 pushes prediction UP, φ_j < 0 pushes prediction DOWN.',
        },
        whenToUse: 'Use Local SHAP whenever stakeholders ask "Why did the model predict X for this specific record?" Use PDP to verify physically monotonic or smooth global behavior.',
        recommendedParams: 'Inspect both extreme high/low predictions and borderline misclassified rows.',
        pitfalls: 'PDP assumes approximate independence between feature S and complementary features C; extreme extrapolation combinations can distort tails.',
      },
      {
        name: 'Algorithmic Fairness (Disparate Impact, Demographic Parity, Equal Opportunity)',
        paradigm: 'Bias & Regulatory Governance',
        whatItDoes:
          'Compares positive prediction rates and true-positive recall across subgroups defined by a protected/sensitive column to verify compliance with the Four-Fifths (80%) rule and equal error-rate parity.',
        formula: {
          equation: 'Disparate Impact (DI) = P(Ŷ=1 | A=unprivileged) / P(Ŷ=1 | A=privileged)    |    DPD = |P(Ŷ=1|A=u) − P(Ŷ=1|A=p)|',
          symbols: 'A = protected group attribute; Equal Opportunity compares True Positive Rate P(Ŷ=1 | Y=1, A) across groups.',
        },
        whenToUse: 'Mandatory for any human-impacting decision system (lending, healthcare, HR, insurance, pricing) or multi-site industrial parity check.',
        recommendedParams: 'Require 0.80 <= DI <= 1.25 and |DPD| <= 0.10.',
        pitfalls: 'Simply dropping the sensitive column does NOT guarantee fairness if correlated proxy variables (e.g., ZIP code) remain.',
      },
    ],
    thresholdTables: [
      {
        title: 'Responsible AI & Fairness Governance Thresholds',
        rows: [
          { metric: 'Disparate Impact Ratio (DI)', range: '0.80 – 1.25 Compliant | < 0.80 or > 1.25 Violation', level: 'bad', interpretation: 'US EEOC / EU AI Act standard Four-Fifths fairness ratio.', action: 'Audit proxy features via SHAP, adjust group thresholds, or rebalance training data.' },
          { metric: 'Demographic Parity Difference (|DPD|)', range: '< 0.05 Ideal | 0.05–0.10 Acceptable | > 0.10 Review Required', level: 'warn', interpretation: 'Absolute gap in positive outcome rate between groups.', action: 'Investigate base-rate differences vs model bias.' },
          { metric: 'Equal Opportunity Difference (|EOD|)', range: '< 0.05 Ideal | 0.05–0.10 Caution | > 0.10 Biased Recall', level: 'warn', interpretation: 'Gap in True Positive Rate among genuinely qualified/positive cases.', action: 'Calibrate decision threshold or add fairness constraints.' },
        ],
      },
    ],
    decisionMatrix: {
      title: 'Studio 07 Governance & Explainability Guide',
      rows: [
        { application: 'Verify top global model drivers & spot data leakage', dataCondition: 'Any trained active model', recommended: 'Global Permutation Importance', alternative: 'GAM spline decomposition (Studio 06)', rationale: 'Single leaked variable immediately spikes far above all physical predictors.' },
        { application: 'Explain a single customer/patient/equipment prediction', dataCondition: 'Specific row index of interest', recommended: 'Local SHAP Attribution', alternative: 'Closed-form Ridge / GAM term substitution', rationale: 'Additive breakdown sums exactly to the individual prediction.' },
        { application: 'Regulatory fairness & Model Card sign-off', dataCondition: 'Sensitive attribute present', recommended: 'Fairness Audit + Model Card Export', alternative: 'Studio 12 Full Governance Report PDF', rationale: 'Documents Disparate Impact, parity metrics, limitations, and lineage.' },
      ],
    },
  },
];
