import type { StudioDeepSupplement } from './handbookTypes';

/**
 * Exhaustive Textbook-Depth Supplements for Studios 01–07.
 * Provides first-principles mathematical theory, concrete numerical worked examples,
 * complete UI control tuning guides, troubleshooting playbooks, and complete method coverage.
 */
export const DEEP_SUPPLEMENTS_PART_1: Record<'01' | '02' | '03' | '04' | '05' | '06' | '07', StudioDeepSupplement> = {
  // =========================================================================
  // STUDIO 01 — DEEP TEXTBOOK SUPPLEMENT
  // =========================================================================
  '01': {
    deepTheory: [
      {
        heading: '1.1 Missing Data Mechanisms: MCAR vs. MAR vs. MNAR — Why Choosing the Wrong Imputer Biases Models',
        paragraphs: [
          'Before filling a single missing cell, a rigorous analyst identifies the statistical mechanism generating the missingness (Rubin’s Taxonomy). Under Missing Completely at Random (MCAR), the probability that value x_ij is missing is independent of both observed and unobserved data: P(M_ij = 1 | X_obs, X_mis) = P(M_ij = 1). Under MCAR, simple listwise deletion remains unbiased (though it wastes sample size), and median/mean imputation preserves location estimates while shrinking variance.',
          'Under Missing at Random (MAR), missingness depends on OTHER observed variables — for example, older industrial turbines experience telemetry drop-outs more often when ambient temperature exceeds 40 °C, or junior customers leave income blank more frequently than senior customers. Here, univariate mean or median imputation distorts conditional distributions and attenuates correlations toward zero, whereas multivariate KNN or Iterative Chained Equations (MICE) reconstructs the missing cell using its correlated neighbors.',
          'Under Missing Not at Random (MNAR), the probability of missingness depends on the unobserved value itself — such as a sensor saturating above 1,200 °C and writing NaN, or ultra-high-income respondents refusing to report salary. No purely statistical imputer can recover MNAR without adding an explicit binary missingness indicator flag (is_missing_col = 1) or using domain bounds. Whenever missingness exceeds 5% on a critical predictor, always inspect whether missingness correlates with your target before choosing an imputer.',
        ],
      },
      {
        heading: '1.2 Mathematical Proof: Why Enabling Split Lock BEFORE Cleaning Eliminates Validation Inflation',
        paragraphs: [
          'Suppose a dataset D has N = 1,000 rows and a feature x with 20% missingness and heavy outliers. If you compute the global median m_all, standard deviation s_all, or IQR outlier fences [Q1 - 1.5·IQR, Q3 + 1.5·IQR] across all 1,000 rows BEFORE splitting into Train (800 rows) and Test (200 rows), then every imputed cell and clipped outlier in the training set contains direct statistical information borrowed from the future test set. In KNN or MICE imputation, test-set rows literally act as nearest neighbors or regression targets for training rows.',
          'When you enable Split Lock in OmniForge BEFORE committing Studio 01 Data Cleaning, the backend invokes DataCleaner.fit_transform_split(train_df, test_df). Every data-dependent parameter — column drop masks, numeric medians/means, KNN distance trees, MICE regression weights, IQR/MAD/Z-score fences, and categorical mode maps — is estimated EXCLUSIVELY on the 800 Train rows, frozen, and then applied identically to the 200 Test rows. The resulting test score mirrors genuine unseen production deployment with zero optimistic bias.',
          'Exception: Purely row-independent structural sanitation (stripping leading/trailing whitespace, decoding HTML entities, converting literal strings "N/A", "null", "--" into true NaN, parsing ISO timestamps) and exact full-row deduplication do not estimate population parameters and are mathematically safe either before or after Split Lock. However, locking Split Lock immediately after loading data and confirming Targets ensures you never accidentally leak statistics during iterative cleaning.',
        ],
      },
      {
        heading: '1.3 Anatomy of the 6-Dimensional Data Quality Score & Forensic Deleted-Records Lineage',
        paragraphs: [
          'OmniForge evaluates every ingested dataset across six orthogonal enterprise dimensions: (1) Completeness (1 − missing cell fraction), (2) Uniqueness (1 − exact duplicate row fraction), (3) Validity (absence of +/-Inf, malformed numeric strings, negative values in strictly positive domains, or constant zero-variance columns), (4) Consistency (uniform casing, whitespace cleanliness, absence of mixed numeric/text columns), (5) Distributional Integrity (skewness and extreme outlier pressure), and (6) Target Readiness (sufficient non-null target variance and healthy class balance).',
          'Crucially, OmniForge enforces Dual-Layer Non-Destructive Provenance: your original uploaded table is frozen immutably as RAW DATA while cleaning commits operate on ACTIVE DATA. Every dropped row or column is recorded with its exact timestamp, reason code (e.g., Duplicate Row, High Missingness >50%, Outlier Trimming), and original row index inside the Forensic Deleted-Records Audit Log, which is exportable to CSV/XLSX and automatically summarized in Studio 12 compliance reports.',
        ],
      },
    ],
    workedExamples: [
      {
        title: 'Worked Example 1A: Choosing Between Classic Z-Score vs. Modified Z-Score (MAD) on Corrupted Sensor Data',
        scenario: 'A pressure transducer logs 9 normal readings [101, 102, 99, 100, 103, 101, 98, 100, 102] bar plus 1 telemetry glitch [450] bar.',
        stepByStepCalculation: [
          'Step 1 — Classic Z-Score (non-robust): Sample mean μ = (906 + 450) / 10 = 135.6 bar. Sample std σ = 110.5 bar.',
          'Compute Classic Z for the 450 bar glitch: Z = (450 − 135.6) / 110.5 = +2.845. Because |Z| = 2.845 < 3.0, Classic Z-Score FAILS TO DETECT the massive 450 bar sensor fault! This phenomenon is called "masking" — the outlier inflates μ and σ so severely that it hides itself.',
          'Step 2 — Modified Z-Score (MAD, robust): Median M = 101.0 bar. Absolute deviations |x_i − 101| = [0, 1, 2, 1, 2, 0, 3, 1, 1, 349]. Sorted: [0, 0, 1, 1, 1, 1, 2, 2, 3, 349]. MAD = median = 1.0 bar.',
          'Compute Modified Z for the 450 bar glitch: M_i = 0.6745 × (450 − 101) / 1.0 = +235.4 >> 3.5! Meanwhile every normal reading has |M_i| <= 0.6745 × 3 / 1 = 2.02 < 3.5.',
        ],
        numericalConclusion: 'Classic Z-Score gave +2.85 (MISSED the 4.5× fault), whereas Modified Z-Score (MAD) gave +235.4 (100% clean detection with zero false alarms on the 9 healthy readings).',
        engineeringAction: 'In Studio 01 Data Cleaning, select Modified Z-Score (MAD, threshold 3.5) or IQR (1.5×) with Winsorize/Clip (or Flag) whenever industrial or financial columns can contain extreme spikes.',
      },
      {
        title: 'Worked Example 1B: Computing Exact IQR Outlier Fences & Winsorization Bounds',
        scenario: 'Monthly customer spend ($) has Q1 (25th percentile) = $40, Median = $75, Q3 (75th percentile) = $120, and one VIP/error row at $980.',
        stepByStepCalculation: [
          'Step 1: IQR = Q3 − Q1 = 120 − 40 = $80.',
          'Step 2: Standard Upper Fence (1.5 × IQR) = Q3 + 1.5 × 80 = 120 + 120 = $240. Lower Fence = max(0, 40 − 120) = -$80 (clipped to $0 for spend).',
          'Step 3: Extreme Upper Fence (3.0 × IQR) = Q3 + 3.0 × 80 = 120 + 240 = $360.',
          'Step 4: Value $980 exceeds both $240 and $360. With Winsorize (Clip), $980 is capped at $240 (preserving the customer record while neutralizing leverage on linear regressions). With Remove, the entire row is dropped and logged in the Forensic Audit Ledger.',
        ],
        numericalConclusion: 'Standard Tukey upper fence = $240; extreme fence = $360; $980 is 10.75 IQRs above Q3.',
        engineeringAction: 'Use Winsorize/Clip when $980 could be a real VIP outlier that would distort linear models; use Remove ONLY when verified as a data-entry glitch (e.g., decimal point error $9.80 entered as $980).',
      },
    ],
    uiControlsGuide: [
      {
        control: 'Execution Mode Bar (Auto / Complete / Custom)',
        defaultValue: 'Auto',
        whenIncreased: 'Complete enables all safe cleaning passes (dedup, whitespace/HTML sanitization, missing imputation, outlier capping, constant-col removal).',
        whenDecreased: 'Custom exposes every individual switch so domain experts control each transformation explicitly.',
        expertGuidance: 'Run Auto or Complete Dry-Run Preview first; switch to Custom if you need KNN/MICE imputation or causal time-series forward-fill.',
      },
      {
        control: 'Numeric Imputation Strategy',
        defaultValue: 'median',
        whenIncreased: 'KNN / Iterative (MICE) captures cross-feature correlations for 5–30% MAR missingness at higher CPU cost.',
        whenDecreased: 'Mean / Median / Zero runs in milliseconds with minimal assumptions.',
        expertGuidance: 'Use Median for skewed tabular data, KNN/MICE for correlated sensor arrays (with Split Lock ON), and Forward-Fill (ffill) for chronological time series.',
      },
      {
        control: 'Outlier Method & Action (Clip / Remove / Flag / None)',
        defaultValue: 'iqr + clip',
        whenIncreased: 'Raising multiplier (e.g., IQR 1.5 -> 3.0 or Z 3.0 -> 4.0) keeps more extreme tail points untouched.',
        whenDecreased: 'Lowering multiplier clips/flags more aggressively (risks truncating genuine extreme operating regimes).',
        expertGuidance: 'Prefer Clip (Winsorize) over Remove for small datasets (<2,000 rows). Never Remove real rare failure spikes if failure detection is your goal!',
      },
      {
        control: 'High-Missing Column Drop Threshold (%)',
        defaultValue: '50%',
        whenIncreased: 'Setting 70–85% retains sparser columns (useful if sparse indicator flags carry signal).',
        whenDecreased: 'Setting 20–30% drops moderately sparse columns aggressively to keep predictors dense.',
        expertGuidance: 'Keep at 45–60% normally; always inspect dropped columns in the Dry-Run Preview before committing.',
      },
      {
        control: 'Causal Time-Series Mode',
        defaultValue: 'Off (Auto for Chronological Split)',
        whenIncreased: 'Enforces strict past-only forward-fill (ffill) even if bfill was requested, blocking look-ahead bias.',
        whenDecreased: 'Allows cross-sectional median/KNN imputation across unordered rows.',
        expertGuidance: 'ALWAYS enable for forecasting, financial returns, and SCADA/IoT time-ordered streams.',
      },
    ],
    troubleshootingPlaybook: [
      {
        symptom: 'Intake Quality Grade is D or F immediately after upload',
        mathematicalCause: 'High missingness (>25% cells), exact duplicate rows, mixed numeric/text columns (e.g., "$1,200" or "N/A" inside numeric columns), or constant zero-variance columns.',
        exactRemediationSteps: '1) Review the Automatic Defect Table in Studio 01. 2) Enable Split Lock in the Topbar. 3) Run Dry-Run Preview in Complete mode to verify string-to-numeric coercion and imputation. 4) Click Apply & Commit Cleaning and inspect the post-clean Quality Grade jump.',
      },
      {
        symptom: 'Validation accuracy drops noticeably after enabling Split Lock before Cleaning',
        mathematicalCause: 'Your previous un-locked score was artificially inflated by train-test leakage (global imputer/scaler/outlier fences peeking at test distribution).',
        exactRemediationSteps: 'Trust the Split-Locked score — it reflects genuine production accuracy. Improve real performance via Studio 02 anomaly screening, Studio 05 feature engineering, and Studio 06 model search rather than removing Split Lock.',
      },
      {
        symptom: 'Important rare anomaly rows disappeared after Data Cleaning',
        mathematicalCause: 'Outlier action was set to "remove" instead of "clip" or "flag", dropping legitimate extreme events.',
        exactRemediationSteps: 'Click "Undo Last Clean" (or "Revert to Raw") in Studio 01, switch Outlier Action to "clip" or "none", and use Studio 02 multivariate Anomaly Screening to inspect extreme points explicitly before deciding their fate.',
      },
    ],
    extraMethods: [
      {
        name: 'Multivariate KNN & Iterative Chained Equations (MICE) Imputation',
        paradigm: 'Multivariate MAR Reconstruction',
        whatItDoes:
          'Instead of replacing missing entries with a single global column median, KNN Imputer finds the k most similar complete rows in standardized feature space and averages their values weighted by inverse distance. Iterative MICE models each incomplete feature as a regularized regression of all other features in round-robin cycles until convergence.',
        formula: {
          equation: 'x̂_ij^(KNN) = [ Σ_{l ∈ N_k(i)} w_il · x_lj ] / [ Σ_{l ∈ N_k(i)} w_il ],   w_il = 1 / (d(x_i, x_l) + ε)',
          symbols: 'N_k(i) = k nearest rows with observed feature j; d(x_i, x_l) = masked Euclidean distance computed over mutually observed coordinates; ε = numerical stability constant.',
        },
        whenToUse: 'Use when missingness is 5–30%, features are moderately-to-strongly correlated (|r| > 0.4), and Split Lock is enabled.',
        assumptions: 'Missing at Random (MAR); requires standardized scaling so large-unit features do not dominate distance calculations.',
        recommendedParams: 'k = 5 neighbors (distance-weighted); MICE max_iter = 10 with BayesianRidge estimator.',
        pitfalls: 'Slow on very wide datasets (>100 columns, >50k rows); MUST be fitted strictly on Train split via Split Lock to prevent severe neighbor leakage.',
      },
      {
        name: 'Causal Time-Series Forward-Fill vs. Look-Ahead Bias Guard',
        paradigm: 'Temporal Data Integrity',
        whatItDoes:
          'Carries forward the last known valid observation x_{t-1} into missing timestamp t without ever reading future timestamps t+1, t+2. OmniForge automatically overrides backward-fill (bfill) into causal forward-fill whenever causal time-series protection is enabled.',
        formula: {
          equation: 'x̂_t = x_{t*},   where t* = max{ s < t : x_s is observed }',
          symbols: 't = current missing timestamp; t* = most recent strictly earlier valid timestamp (zero information from s > t).',
        },
        whenToUse: 'Mandatory for financial price series, industrial SCADA telemetry, medical vitals monitoring, and demand forecasting.',
        assumptions: 'Rows are ordered chronologically along the time index; gaps are short relative to process dynamics.',
        recommendedParams: 'Sort by timestamp first, enable Chronological Split Lock, then choose ffill with causal protection enabled.',
        pitfalls: 'Using bfill or global median on a trending time series leaks future prices/temperatures into the past and creates illusory forecasting accuracy that fails completely in production.',
      },
    ],
  },

  // =========================================================================
  // STUDIO 02 — DEEP TEXTBOOK SUPPLEMENT
  // =========================================================================
  '02': {
    deepTheory: [
      {
        heading: '2.1 Geometric & Probabilistic Mechanics of Unsupervised Structure Discovery',
        paragraphs: [
          'Unsupervised clustering partitions n observations in R^p into coherent subpopulations without using target labels y. However, every clustering algorithm encodes a strong geometric assumption about cluster shape. K-Means minimizes within-cluster squared Euclidean distance (Voronoi tessellation), implicitly assuming spherical, equal-variance, similarly sized clusters. If your real data contains elongated ellipses (e.g., correlated temperature-pressure operating modes) or concentric rings, K-Means artificially slices single physical regimes in half.',
          'Gaussian Mixture Models (GMM) relax K-Means into a soft probabilistic generative model p(x) = Σ_k π_k N(x | μ_k, Σ_k), estimated via Expectation-Maximization (EM). Each cluster acquires its own covariance ellipse Σ_k and soft posterior membership probabilities γ_ik = P(z_i = k | x_i), enabling principled model selection via BIC and uncertainty quantification on boundary points.',
          'DBSCAN abandons centroids entirely and defines clusters as maximally connected regions of high local density (at least min_samples points within radius ε). Points failing to reach any dense core are assigned Cluster -1 (Noise). This makes DBSCAN uniquely powerful in engineering and fraud analytics for isolating arbitrary-shape operating regimes AND cleanly separating corrupt sensor bursts (`Cluster -1`) for inspection or removal.',
        ],
      },
      {
        heading: '2.2 Global vs. Local Multivariate Anomalies: Why Univariate Fences Miss Coupled Failures',
        paragraphs: [
          'Consider a gas turbine where normal exhaust temperature ranges from 420–560 °C and normal load ranges from 20–100 MW. A reading of (Load = 22 MW, Exhaust Temp = 555 °C) looks completely normal in every single column individually — it passes every Studio 01 IQR and MAD fence! Yet physically, running 555 °C at idle 22 MW load is a catastrophic combustion/blade-fouling fault.',
          'Multivariate anomaly detectors in Studio 02 catch exactly these coupled failures. Isolation Forest recursively picks random feature splits: abnormal points sitting in sparse off-manifold regions get isolated in far fewer splits h(x) than dense normal points. Local Outlier Factor (LOF) compares a point’s local density to the local densities of its k nearest neighbors — catching anomalies nestled inside sparse pockets next to dense operating clusters. One-Class SVM learns a smooth kernel decision boundary encapsulating normal operating states.',
        ],
      },
      {
        heading: '2.3 Safe Preview-Before-Commit Lineage: Removing Corrupt Clusters vs. Enriching Features',
        paragraphs: [
          'In OmniForge Studio 02, running Clustering or Anomaly Screening ALWAYS runs in non-destructive PREVIEW mode first. After inspecting the cluster profile table, Silhouette/DBI/CH benchmark, and 2D manifold scatter plot, you have two distinct analytical workflows: (A) Data Sanitation — explicitly delete a verified corrupt cluster (e.g., DBSCAN Cluster -1 or an equipment-off shutdown cluster) or confirmed sensor-fault anomalies, recording every removed row to the Forensic Audit Log; OR (B) Feature Enrichment — attach cluster_id, is_anomaly, and anomaly_score as new engineered columns so downstream Studio 06 models learn distinct slopes/rules per operating regime.',
        ],
      },
    ],
    workedExamples: [
      {
        title: 'Worked Example 2A: Calculating Silhouette Score & Choosing Optimal k',
        scenario: 'Point i belongs to Cluster A. Its average Euclidean distance to other points in Cluster A is a(i) = 1.4. Its average distance to the nearest neighboring Cluster B is b(i) = 4.2.',
        stepByStepCalculation: [
          'Formula: s(i) = (b(i) − a(i)) / max(a(i), b(i)).',
          'Substitute: s(i) = (4.2 − 1.4) / max(1.4, 4.2) = 2.8 / 4.2 = +0.667.',
          'Now compare a poorly placed point j on the overlapping border where a(j) = 3.1 and b(j) = 2.9: s(j) = (2.9 − 3.1) / 3.1 = −0.065 (negative value proves j is closer to the wrong cluster!).',
          'Averaging s(i) across all n points gives the global Silhouette score: k=3 yields 0.61, k=4 yields 0.44, k=5 yields 0.38 -> k=3 is the mathematically cleanest separation.',
        ],
        numericalConclusion: 's(i) = +0.667 confirms tight, well-separated membership; global Silhouette 0.61 (>0.50) selects k=3 as optimal.',
        engineeringAction: 'Accept k=3, inspect the 3 cluster centroids in the Cluster Profile table to name each physical/business segment, and attach cluster_id to Active Data.',
      },
      {
        title: 'Worked Example 2B: Evaluating Apriori Association Rule Lift vs. Misleading Confidence',
        scenario: 'In 1,000 maintenance logs: Event A (High Vibration) occurs 100 times (10%); Event B (Routine Oil Top-Up) occurs 800 times (80%); A and B occur together 75 times (7.5%). Meanwhile Event C (Bearing Failure) occurs 50 times (5%), and A and C occur together 40 times (4%).',
        stepByStepCalculation: [
          'Rule 1 (A => B): Support = 0.075, Confidence = P(B|A) = 75/100 = 0.75 (75%!). Looks high — BUT Lift = 0.75 / 0.80 = 0.9375 < 1.0! High Vibration actually occurs LESS often with Routine Oil Top-Up than random chance.',
          'Rule 2 (A => C): Support = 0.040, Confidence = P(C|A) = 40/100 = 0.40 (40%). Lower confidence than Rule 1 — YET Lift = 0.40 / 0.05 = 8.00! Bearing Failure is 8× MORE LIKELY when High Vibration is present.',
        ],
        numericalConclusion: 'Never sort Apriori rules by Confidence alone! Rule 2 (Lift = 8.0) is a genuine physical failure predictor while Rule 1 (Confidence = 75%, Lift = 0.94) is a statistical illusion.',
        engineeringAction: 'Always filter Association Rules in Studio 02 by Lift >= 1.5 (and Support >= 0.02) before turning rules into operational maintenance alerts.',
      },
    ],
    uiControlsGuide: [
      {
        control: 'Clustering Algorithm & Auto-Benchmark Mode',
        defaultValue: 'Auto (Multi-Algorithm Benchmark)',
        whenIncreased: 'Complete benchmarks K-Means, GMM, DBSCAN, Agglomerative, BIRCH, and Spectral simultaneously and ranks by Silhouette + CH + DBI.',
        whenDecreased: 'Custom lets you lock one exact algorithm and exact cluster count k or DBSCAN eps.',
        expertGuidance: 'Run Auto/Complete first to see whether spherical (K-Means/GMM) or density (DBSCAN) geometry fits your data best.',
      },
      {
        control: 'Number of Clusters (k)',
        defaultValue: 'Auto (Silhouette Sweep k=2..8)',
        whenIncreased: 'Higher k splits data into finer micro-segments (risks overfitting noise and lowering Silhouette).',
        whenDecreased: 'Lower k merges subpopulations into broad macro-regimes.',
        expertGuidance: 'Choose the smallest k near the Silhouette peak where every cluster represents a meaningful, actionable physical or customer segment.',
      },
      {
        control: 'Anomaly Contamination Rate',
        defaultValue: '0.05 (5%)',
        whenIncreased: '0.10–0.15 flags a broader tail of suspicious observations (higher recall, more false alarms).',
        whenDecreased: '0.01–0.02 flags only the most extreme top 1–2% anomalies (high precision).',
        expertGuidance: 'Match contamination to realistic expected fault/fraud prevalence; always inspect flagged rows in the preview table before deleting.',
      },
      {
        control: 'Manifold Projection Method (PCA / SVD / t-SNE / Isomap / MDS)',
        defaultValue: 'PCA',
        whenIncreased: 't-SNE / Isomap reveals curved non-linear local pockets hidden in linear PCA.',
        whenDecreased: 'PCA / SVD preserves global Euclidean distances and runs instantly on large tables.',
        expertGuidance: 'Use PCA for truthful global distances and variance explained; use t-SNE strictly to inspect local neighborhood separation.',
      },
    ],
    troubleshootingPlaybook: [
      {
        symptom: 'All clustering algorithms report Silhouette < 0.20',
        mathematicalCause: 'Either the data forms a single continuous cloud without discrete clusters, OR unscaled high-magnitude features / irrelevant noise columns are drowning out real cluster structure.',
        exactRemediationSteps: '1) Exclude ID/timestamp/constant columns via Topbar Exclusions. 2) Clean extreme skew/outliers in Studio 01 first. 3) If Silhouette stays <0.20 across K-Means, GMM, and DBSCAN, treat data as a continuous manifold and skip cluster deletion.',
      },
      {
        symptom: 'DBSCAN puts >80% of points into Cluster -1 (Noise) or 100% into Cluster 0',
        mathematicalCause: 'ε (eps) is too small (everything becomes noise) or too large (everything merges into one giant blob), common in high dimensions (>15 features).',
        exactRemediationSteps: 'Reduce feature count via Exclusions/PCA, tune eps upward (if too much noise) or downward (if 1 cluster), or switch to GMM / K-Means + Isolation Forest.',
      },
    ],
    extraMethods: [
      {
        name: 'Local Outlier Factor (LOF) & One-Class SVM',
        paradigm: 'Local Density & Kernel Envelope Anomaly Detection',
        whatItDoes:
          'LOF computes each observation’s local reachability density (LRD) relative to its k nearest neighbors and outputs the ratio of neighbors’ average density to the point’s own density. Values near 1.0 are normal; values >> 1.5 indicate local outliers even inside varying-density clusters. One-Class SVM maps data into RBF kernel space and learns the tightest separating hyperplane enclosing normal operating states.',
        formula: {
          equation: 'LOF_k(x) = [ (1/k) Σ_{o ∈ N_k(x)} lrd_k(o) ] / lrd_k(x),   lrd_k(x) = k / Σ_{o ∈ N_k(x)} reach-dist_k(x, o)',
          symbols: 'N_k(x) = k nearest neighbors of x; reach-dist_k(x,o) = max(k-distance(o), d(x,o)); LOF ≈ 1 normal, LOF > 1.5–2.0 local outlier.',
        },
        whenToUse: 'Use LOF when clusters have sharply different densities (e.g., idle vs full-load regimes); use One-Class SVM when training data is clean and you want a smooth novelty envelope.',
        assumptions: 'Continuous standardized feature space; performance degrades in >30 unpruned noise dimensions.',
        recommendedParams: 'LOF n_neighbors = 20 (range 15–35); One-Class SVM nu = 0.05, kernel = rbf.',
        pitfalls: 'LOF requires care when scoring new unseen points (use novelty mode); One-Class SVM scales quadratically for n > 15,000.',
      },
    ],
  },

  // =========================================================================
  // STUDIO 03 — DEEP TEXTBOOK SUPPLEMENT
  // =========================================================================
  '03': {
    deepTheory: [
      {
        heading: '3.1 Why Strong Baselines & Lift over Naive Floor Are Non-Negotiable Scientific Gates',
        paragraphs: [
          'A complex Gradient Boosted Tree or Deep Neural Network achieving R² = 0.72 or Accuracy = 91% sounds impressive in isolation — until you discover that simply predicting the constant majority class ("No Failure") achieves 92% accuracy, or a 3-term Ridge linear equation achieves R² = 0.74! Without an honest baseline benchmark, teams routinely deploy brittle black-box pipelines that perform worse than trivial rules.',
          'Studio 03 establishes the immutable scientific reference floor before any AutoML or Deep Learning runs. It benchmarks BOTH: (1) a Naive Zero-Intelligence Floor (Mean/Median regressor or Stratified/Most-Frequent classifier), (2) Exact Closed-Form Parametric Baselines (Ridge Linear / Regularized Logistic / Decision Stump), AND (3) Distribution-Free Non-Parametric Analytical Baselines (Penalized B-Spline GAM, Nadaraya-Watson Kernel Smoother, Weighted k-NN).',
          'Comparing Parametric vs. Non-Parametric baselines right in Studio 03 yields an immediate structural diagnostic: if B-Spline GAM beats Ridge Linear by > +0.05 R² (or +3% F1), you have mathematical proof that your dataset contains genuine non-linear physics, justifying non-linear models in Studio 06. If Ridge ties or beats GAM, your underlying relationship is predominantly linear and a transparent closed-form linear equation is scientifically optimal (Occam’s Razor).',
        ],
      },
      {
        heading: '3.2 Multicollinearity Geometry (VIF) & Why Correlated Features Destabilize Equations',
        paragraphs: [
          'In Ordinary Least Squares β̂ = (X^T X)^{-1} X^T y, the variance of each estimated slope satisfies Var(β̂_j) = (σ² / SST_j) · VIF_j, where VIF_j = 1 / (1 − R_j²). When feature x_j can be predicted with R_j² = 0.95 from the remaining predictors, VIF_j = 20 — inflating coefficient standard errors by √20 = 4.47×! Signs flip randomly across small samples (e.g., making a physically positive temperature effect appear negative), destroying causal and physical interpretation.',
          'Crucially, multicollinearity harms COEFFICIENT INTERPRETATION far more than raw prediction accuracy. When Studio 03 flags VIF > 10, you have three clean remedies: (1) Drop one redundant column from each collinear pair (keep the more direct physical sensor), (2) Use L2 Ridge regularization (which shrinks correlated coefficients jointly without exploding variance), or (3) Combine collinear sensors via PCA in Studio 05.',
        ],
      },
      {
        heading: '3.3 Coherent Tail Risk Quantification: Why CVaR (Expected Shortfall) Dominates VaR',
        paragraphs: [
          'Value-at-Risk at level α (e.g., 95% confidence, α = 0.05) answers: "What loss threshold will NOT be exceeded 95% of the time?" While intuitive, VaR has two fatal flaws proved by Artzner et al. (1999): (1) it is completely blind to HOW BAD losses become inside the worst 5% tail (a −$10k tail and a −$10M bankruptcy tail can have identical VaR!), and (2) VaR violates sub-additivity, meaning diversifying two portfolios can paradoxically INCREASE VaR.',
          'Conditional Value-at-Risk (CVaR / Expected Shortfall) averages all losses strictly worse than VaR_α: CVaR_α = E[Loss | Loss >= VaR_α]. CVaR is mathematically coherent (satisfies sub-additivity, monotonicity, positive homogeneity, and translation invariance) and is the Basel III international banking & engineering tail-risk standard. Always track the CVaR / VaR ratio: values > 1.5 warn of severe fat-tailed extreme-event risk.',
        ],
      },
    ],
    workedExamples: [
      {
        title: 'Worked Example 3A: Computing VIF & Diagnosing Sign-Flipping Collinearity',
        scenario: 'You regress Turbine Power Output on Fuel Flow (x1) and Compressor Discharge Pressure (x2). Regressing x1 on x2 gives R₁² = 0.94.',
        stepByStepCalculation: [
          'Step 1: Compute VIF₁ = 1 / (1 − R₁²) = 1 / (1 − 0.94) = 1 / 0.06 = 16.67.',
          'Step 2: Compute Standard Error Inflation Factor = √VIF₁ = √16.67 = 4.08× wider confidence intervals than orthogonal features.',
          'Step 3: Compare against threshold table: VIF = 16.67 > 10 (SEVERE COLLINEARITY). Unregularized OLS coefficients for Fuel Flow and Discharge Pressure will oscillate wildly and may even display opposite signs.',
        ],
        numericalConclusion: 'VIF = 16.67 (> 10 critical threshold) proves severe collinearity between x1 and x2.',
        engineeringAction: 'Either exclude the derived/secondary sensor via Topbar Exclusions OR rely on L2-penalized Ridge / B-Spline GAM (Studio 03/06) rather than unregularized OLS.',
      },
      {
        title: 'Worked Example 3B: Computing Empirical 95% VaR & CVaR on 20 Observed Daily P&L Returns',
        scenario: 'Sorted worst 5 daily returns out of 100 days (α = 0.05 -> worst 5 days): [−14.0%, −9.2%, −6.5%, −4.8%, −4.1%, −3.7%, ...].',
        stepByStepCalculation: [
          'Step 1 (VaR at 95% confidence, α = 0.05): The 5th percentile cutoff sits at −4.1%. Thus Historical VaR_95% = 4.1% loss.',
          'Step 2 (CVaR / Expected Shortfall at 95%): Average the 5 observations at or worse than VaR: (−14.0 − 9.2 − 6.5 − 4.8 − 4.1) / 5 = −38.6 / 5 = −7.72%. Thus CVaR_95% = 7.72% expected tail loss.',
          'Step 3 (Tail Fatness Ratio): CVaR / VaR = 7.72 / 4.10 = 1.88 (under a pure Gaussian normal distribution, CVaR_95% / VaR_95% is only 1.25!).',
        ],
        numericalConclusion: 'VaR_95% = 4.1%, CVaR_95% = 7.72%, Ratio = 1.88 >> 1.25 -> Severe fat-tailed downside risk.',
        engineeringAction: 'Budget capital/safety margin to CVaR (7.72%), NEVER just VaR (4.1%), and inspect the −14.0% excursion day in Studio 02 Anomaly Screening.',
      },
    ],
    uiControlsGuide: [
      {
        control: 'Significance Level (α)',
        defaultValue: '0.05',
        whenIncreased: '0.10 increases sensitivity in exploratory pilot studies (more false positives).',
        whenDecreased: '0.01 or 0.001 tightens statistical rigor for safety-critical or high-sample (>50k) datasets.',
        expertGuidance: 'Keep 0.05 for general analytics; drop to 0.01 when n > 10,000 and always check effect size alongside p-values.',
      },
      {
        control: 'Multiple Testing Correction (FDR BH / Bonferroni / Holm / None)',
        defaultValue: 'fdr_bh',
        whenIncreased: 'Bonferroni (α/m) is maximally conservative (strictly controls probability of even 1 false positive).',
        whenDecreased: 'None performs raw unadjusted tests (inflates false-discovery rate across 20+ columns).',
        expertGuidance: 'Use Benjamini-Hochberg (fdr_bh) for exploratory screening across many features; use Holm-Bonferroni for strict confirmatory validation.',
      },
      {
        control: 'Tail Risk α (VaR / CVaR)',
        defaultValue: '0.05 (95% confidence)',
        whenIncreased: '0.10 measures 90% tail (less extreme); 0.01 measures extreme 99% worst-1% catastrophes.',
        whenDecreased: 'Lower α demands larger sample sizes (need n >= 500 for reliable empirical 99% CVaR).',
        expertGuidance: 'Use α = 0.05 for n = 100–1,000 and α = 0.01 when n >= 1,000.',
      },
      {
        control: 'Parametric Policy (Auto / Parametric / Non-Parametric)',
        defaultValue: 'auto',
        whenIncreased: 'Auto runs Shapiro-Wilk normality checks first and automatically routes skewed variables to non-parametric tests/baselines.',
        whenDecreased: 'Forcing Parametric assumes Gaussian distributions; forcing Non-Parametric uses rank/kernel/GAM methods unconditionally.',
        expertGuidance: 'Keep Auto unless regulatory protocol mandates a specific parametric or distribution-free test family.',
      },
    ],
    troubleshootingPlaybook: [
      {
        symptom: 'All baseline models show Lift vs Naive <= 0 (R² <= 0 or Accuracy == Majority Class)',
        mathematicalCause: 'Either features carry no genuine predictive signal for the selected target, target is corrupted/mis-typed, or severe uncleaned outliers / wrong target selection broke learning.',
        exactRemediationSteps: '1) Confirm the right Target column in Topbar Targets. 2) Check Studio 04 correlation heatmap & Mutual Information in Studio 05. 3) Verify you didn’t accidentally exclude the key predictors or leave extreme +/-1e9 glitches uncleaned in Studio 01.',
      },
      {
        symptom: 'Ranked Baseline Equations show 5+ features with VIF > 20',
        mathematicalCause: 'Derived duplicate columns (e.g., total_cost = price * qty, or Celsius + Fahrenheit copies, or dummy-variable trap) exist simultaneously in Active Data.',
        exactRemediationSteps: 'Open Topbar Exclusions, mask one redundant feature per collinear cluster, and re-run Studio 03 to verify VIF drops below 5–10 and coefficients stabilize.',
      },
    ],
  },

  // =========================================================================
  // STUDIO 04 — DEEP TEXTBOOK SUPPLEMENT (COMPLETE ALL-17 HYPOTHESIS TESTS!)
  // =========================================================================
  '04': {
    deepTheory: [
      {
        heading: '4.1 The Classical & Non-Parametric Hypothesis Testing Framework (p-Values, Power & Effect Size)',
        paragraphs: [
          'Every statistical hypothesis test evaluates observed data against a precise mathematical Null Hypothesis H₀ (e.g., "the two group means are equal", "the distribution is Gaussian", or "rows and columns are independent"). The test compresses your data into a single Test Statistic T whose sampling distribution under H₀ is known analytically or asymptotically. The p-value = P(|T_null| >= |T_observed| | H₀ is true) is the exact probability of observing an effect at least as extreme as yours purely by random sampling noise.',
          'CRITICAL STATISTICAL FALLACY TO AVOID: A p-value is NOT the probability that H₀ is true, AND statistical significance (p < 0.05) does NOT imply practical or engineering importance! With n = 100,000 observations, a trivially tiny 0.02% mean shift produces p < 10⁻⁸ even though it is operationally meaningless. Conversely, with n = 12 observations, a massive 20% real improvement may produce p = 0.14 due to low statistical power (1 − β).',
          'Therefore, Studio 04 ALWAYS pairs p-values with (1) distributional assumption checks (normality & variance homogeneity), (2) appropriate non-parametric counterparts (Mann-Whitney U, Wilcoxon Signed-Rank, Kruskal-Wallis H, Spearman ρ, Kendall τ, Fligner-Killeen, Fisher’s Exact), and (3) magnitude / effect size interpretation (Cohen’s d, correlation r/ρ, η², Cramér’s V).',
        ],
      },
      {
        heading: '4.2 Choosing Parametric vs. Non-Parametric Tests: A Complete Decision Logic',
        paragraphs: [
          'Parametric tests (Welch t, Paired t, One-Way ANOVA, Pearson r, Bartlett) estimate parameters of an assumed Gaussian distribution. When the underlying data is genuinely near-normal without severe outliers, parametric tests maximize statistical power (highest chance of detecting real effects at small sample sizes).',
          'Non-parametric tests (Mann-Whitney U, Wilcoxon Signed-Rank, Kruskal-Wallis H, Spearman ρ, Kendall τ, 2-Sample KS, Fligner-Killeen) operate on ranks, signs, or empirical cumulative distribution functions (ECDFs). They achieve ~95.5% of the asymptotic relative efficiency of parametric tests even on perfectly Gaussian data, while remaining 100% immune to heavy skew, outliers, log-normal distributions, and ordinal rating scales! Rule of thumb: when in doubt or when Shapiro-Wilk p < 0.05 with |skew| > 1.0, always report the non-parametric test alongside (or instead of) the parametric test.',
        ],
      },
    ],
    workedExamples: [
      {
        title: 'Worked Example 4A: Welch Two-Sample t-Test & Cohen’s d on Industrial Cooling Efficiency',
        scenario: 'Illustrative summary data for two independent groups. Control mode (n1 = 25, mean1 = 82.0%, s1 = 4.0%) and Eco mode (n2 = 30, mean2 = 85.4%, s2 = 5.5%). Use a two-sided test for a difference at α = 0.05.',
        stepByStepCalculation: [
          'Step 1 — Standard Error of Difference: SE = √(s1²/n1 + s2²/n2) = √(16/25 + 30.25/30) = √(0.64 + 1.0083) = √1.6483 = 1.2839.',
          'Step 2 — Welch t-Statistic: t = (85.4 − 82.0) / 1.2839 = +2.648.',
          'Step 3 — Welch-Satterthwaite Degrees of Freedom: ν ≈ (0.64 + 1.0083)² / [0.64²/24 + 1.0083²/29] = 2.717 / (0.01707 + 0.03506) = 52.1.',
          'Step 4 — Two-sided p-value for t = 2.648 (df = 52): p = 0.0107 < 0.05 -> REJECT H₀.',
          'Step 5 — Effect Size (Cohen’s d): s_pooled ≈ √[(24·16 + 29·30.25)/53] = 4.88. d = (85.4 − 82.0) / 4.88 = +0.697 (Medium-to-Large practical effect!).',
        ],
        numericalConclusion: 't(52.1) = 2.65, p = 0.0107, Cohen’s d = 0.70. The estimated difference is 3.4 percentage points.',
        engineeringAction: 'Check whether the sampling supports independence and compare the estimated gain and its uncertainty with the operating threshold.',
      },
      {
        title: 'Worked Example 4B: Chi-Square Test of Independence & Cramér’s V on Defect Rate by Shift',
        scenario: 'Illustrative counts. Day shift had 18 defects and 182 passes. Night shift had 32 defects and 168 passes. Total N = 400.',
        stepByStepCalculation: [
          'Step 1 — Expected Counts under H₀ (Independence): E(Day,Defect) = 200×50/400 = 25; E(Night,Defect) = 25; E(Day,Pass) = 175; E(Night,Pass) = 175. All E >= 5 -> Chi-Square valid!',
          'Step 2 — Apply the Studio’s 2×2 continuity correction. Each |O−E| = 7 is reduced to 6.5. χ² = 2×6.5²/25 + 2×6.5²/175 = 3.862857 (df = 1).',
          'Step 3 — The p-value for the corrected statistic is 0.04937.',
          'Step 4 — Cramér’s V uses the uncorrected Pearson statistic, 2×7²/25 + 2×7²/175 = 4.48. V = √(4.48 / 400) ≈ 0.10583. The p-value uses the separate corrected test.',
        ],
        numericalConclusion: 'The illustrative defect rates are 16% at night and 9% during the day. The corrected independence test gives p = 0.04937 for this single comparison.',
        engineeringAction: 'Check dependence between observations and possible differences in production mix, ambient conditions and calibration before attributing the association to shift.',
      },
    ],
    uiControlsGuide: [
      {
        control: 'Hypothesis Execution Mode (Auto / Complete / Custom)',
        defaultValue: 'Auto',
        whenIncreased: 'Complete uses a broader set of tests and applies Benjamini–Hochberg adjustment across evaluable results.',
        whenDecreased: 'Custom lets you select one method and its input columns.',
        expertGuidance: 'Run Auto/Complete for global discovery, then switch to Custom to test specific domain hypotheses cleanly.',
      },
      {
        control: 'Statistical Test Selector',
        defaultValue: 'shapiro',
        whenIncreased: 'Includes Normality/GoF (5), Two-Sample & Paired (5), Multi-Group & Variance (5), Correlation & Categorical (5).',
        whenDecreased: 'Distribution Goodness-of-Fit ranks fitted families by AIC and uses a fitted-null KS bootstrap for an exploratory check of the selected family.',
        expertGuidance: 'Match data types strictly: 1 numeric -> Normality/GoF; 1 numeric + 1 grouping column -> Welch t / Mann-Whitney / ANOVA / Kruskal / Levene; 2 numerics -> Pearson / Spearman / Kendall / Paired t / Wilcoxon; 2 categoricals -> Chi-Square / Fisher Exact.',
      },
    ],
    troubleshootingPlaybook: [
      {
        symptom: 'Shapiro-Wilk returns p = 0.0001 on a histogram that looks visually bell-shaped',
        mathematicalCause: 'Sample size n is very large (n > 2,000); Shapiro-Wilk gains extreme power and rejects harmless 1% deviations from perfect mathematical normality.',
        exactRemediationSteps: 'Check Skewness (|γ1| < 0.5) and Excess Kurtosis (|γ2| < 1.0) + visual histogram. Remember Central Limit Theorem: means of large samples are normal even when raw data has tiny departures.',
      },
      {
        symptom: 'Pearson r = 0.08 (not significant) even though scatter plot shows a strong U-shaped or curved relationship',
        mathematicalCause: 'Pearson r ONLY measures straight-line linear correlation; symmetric parabolic or saturating curves cancel positive and negative slopes to near zero.',
        exactRemediationSteps: 'Check Spearman ρ (for monotonic curves) AND run Studio 05 Mutual Information + Studio 06 Non-Parametric B-Spline GAM (which detects arbitrary non-monotonic curves cleanly!).',
      },
    ],
    extraMethods: [
      {
        name: "D'Agostino-Pearson K² Omnibus & Anderson-Darling Normality Tests",
        paradigm: 'Higher-Moment & Tail-Weighted Normality Tests',
        whatItDoes:
          "D'Agostino-Pearson combines standardized skewness Z_s and kurtosis Z_k into an omnibus chi-square statistic K² = Z_s² + Z_k² (works reliably even for very large n where Shapiro-Wilk caps at 5,000). Anderson-Darling places extra quadratic weight 1/[F(x)(1-F(x))] on the extreme left and right distribution tails — making it the sharpest detector of heavy-tailed departures.",
        nullHypothesis: 'H₀: Data is drawn from a Gaussian normal distribution N(μ, σ²).',
        formula: {
          equation: "K² = Z_skew² + Z_kurt² ~ χ²(df=2)   |   A² = −n − (1/n) Σ_{i=1..n} (2i−1) [ ln F(x_(i)) + ln(1 − F(x_(n+1−i))) ]",
          symbols: 'Z_skew, Z_kurt = normal transformations of sample skewness/kurtosis; x_(i) = sorted observations; F = Gaussian CDF evaluated at sample μ, σ.',
        },
        whenToUse: "Use D'Agostino K² for n >= 20 (especially n > 5,000); use Anderson-Darling when accurate tail behavior matters (finance, reliability, SPC limits).",
        assumptions: 'Continuous i.i.d. numeric observations; D’Agostino requires n >= 20.',
        recommendedParams: 'α = 0.05; compare A² against critical value 0.787 (at 5% significance).',
        pitfalls: 'Tied/quantized values inflate A²; always verify visual distribution shape.',
      },
      {
        name: 'Paired Student t-Test & Wilcoxon Signed-Rank Test (Before vs. After)',
        paradigm: 'Matched-Pairs Difference Testing (Parametric vs. Non-Parametric)',
        whatItDoes:
          'When each subject/machine is measured twice (Before vs. After intervention), pairing subtracts out individual baseline differences d_i = x_{i,after} − x_{i,before}, dramatically boosting statistical power. Paired t tests whether mean(d_i) = 0; Wilcoxon Signed-Rank ranks |d_i| and compares the sum of positive vs. negative ranks without requiring Gaussian differences.',
        nullHypothesis: 'H₀: Mean difference μ_d = 0 (Paired t) or median difference = 0 with symmetric differences (Wilcoxon).',
        formula: {
          equation: 't_paired = d̄ / (s_d / √n)   (df = n − 1)   |   W = min( Σ_{d_i > 0} Rank(|d_i|),  Σ_{d_i < 0} Rank(|d_i|) )',
          symbols: 'd_i = matched pair difference; d̄ = mean difference; s_d = std of differences; n = number of non-zero pairs.',
        },
        whenToUse: 'Before-vs-after maintenance, A/B matched twins, dual-sensor calibration checks on the exact same timestamps.',
        assumptions: 'Observations are genuinely paired row-by-row; Paired t assumes normal differences d_i; Wilcoxon assumes symmetric differences.',
        recommendedParams: 'α = 0.05 two-sided; report mean/median Δ alongside p-value.',
        pitfalls: 'Never run an unpaired two-sample t-test on paired before/after rows — you throw away massive statistical power and violate independence.',
      },
      {
        name: 'Two-Sample Kolmogorov-Smirnov (KS) Distribution Test',
        paradigm: 'Non-Parametric Full-Distribution Comparison',
        whatItDoes:
          'Measures the maximum vertical gap D between the empirical cumulative distribution functions (ECDFs) of Sample 1 and Sample 2. Unlike t-tests (mean only) or Levene (variance only), 2-Sample KS detects ANY difference in location, spread, skewness, bimodality, or tail weight simultaneously.',
        nullHypothesis: 'H₀: Sample 1 and Sample 2 come from the exact same continuous probability distribution F₁(x) = F₂(x) for all x.',
        formula: {
          equation: 'D_{n,m} = sup_x | F_{1,n}(x) − F_{2,m}(x) |',
          symbols: 'F_{1,n}(x) = fraction of Sample 1 <= x; F_{2,m}(x) = fraction of Sample 2 <= x; D ∈ [0, 1] (0 = identical ECDFs).',
        },
        whenToUse: 'Detecting distribution shift between Train vs. Production batches, or comparing two sensor populations when shapes differ.',
        assumptions: 'Continuous distributions (conservative with heavy discrete ties).',
        recommendedParams: 'α = 0.05; in MLOps drift monitoring, flag D >= 0.10 (moderate) or D >= 0.20 (major shift).',
        pitfalls: 'Tells you THAT distributions differ, not strictly whether mean or variance drove it — inspect histograms alongside KS.',
      },
      {
        name: 'Levene, Bartlett & Fligner-Killeen Variance Homogeneity Tests',
        paradigm: 'Equality of Spread / Homoscedasticity Tests',
        whatItDoes:
          'Tests whether k groups share equal variance (homoscedasticity — a key assumption of classical ANOVA and OLS). Bartlett is optimal strictly under exact normality but breaks badly under heavy tails; Levene applies ANOVA to |x_ij − median_i| (Brown-Forsythe variant, robust); Fligner-Killeen ranks absolute deviations using normal scores and is the MOST robust variance test against extreme non-normality.',
        nullHypothesis: 'H₀: σ₁² = σ₂² = ... = σ_k² (all groups have identical variance).',
        formula: {
          equation: 'W_Levene = [ (N − k) / (k − 1) ] · [ Σ_i n_i (Z̄_i − Z̄)² ] / [ Σ_i Σ_j (Z_ij − Z̄_i)² ],   Z_ij = |x_ij − median_i|',
          symbols: 'k = number of groups; N = total samples; Z_ij = absolute deviation from group median; W ~ F(k−1, N−k) under H₀.',
        },
        whenToUse: 'Manufacturing process consistency checks (did a new nozzle reduce variance?) and checking ANOVA assumptions.',
        assumptions: 'Independent samples across groups; Bartlett strictly requires Gaussian groups.',
        recommendedParams: 'Prefer Levene (median-centered) or Fligner-Killeen as universal defaults; use Bartlett only after confirming normality.',
        pitfalls: 'If Levene rejects (p < 0.05), use Welch’s t-test / Kruskal-Wallis rather than equal-variance Student t or standard ANOVA.',
      },
      {
        name: "Kendall's Tau (τ) Rank Correlation & Fisher's Exact Test (2×2)",
        paradigm: 'Robust Small-Sample Rank & Exact Contingency Inference',
        whatItDoes:
          "Kendall's τ directly computes the probability difference between concordant pairs (both X and Y move in the same direction) and discordant pairs (move oppositely) — offering tighter small-sample confidence intervals and higher outlier resistance than Spearman ρ. Fisher's Exact Test computes the EXACT hypergeometric probability of a 2×2 table without any large-sample χ² approximation.",
        nullHypothesis: "H₀: τ = 0 (no monotonic ordering concordance) | Fisher H₀: Odds Ratio OR = (a·d)/(b·c) = 1 (row/column independence).",
        formula: {
          equation: 'τ_b = (N_c − N_d) / √[(N_0 − N_1)(N_0 − N_2)]   |   P_Fisher = [ (a+b)!(c+d)!(a+c)!(b+d)! ] / [ N! · a! b! c! d! ]',
          symbols: 'N_c = concordant pairs; N_d = discordant pairs; N_1, N_2 = tie adjustments; a,b,c,d = 2×2 cell counts summing to N.',
        },
        whenToUse: "Use Kendall τ for small samples (n < 30) or ordinal data with many ties; use Fisher's Exact Test whenever ANY cell in a 2×2 table has expected count < 5.",
        assumptions: "Kendall τ assumes ordinal/continuous paired observations; Fisher's Exact assumes fixed table margins.",
        recommendedParams: 'α = 0.05; report Odds Ratio alongside Fisher p-value for clinical/reliability interpretation.',
        pitfalls: 'Note that |τ| is typically ~0.7× smaller numerically than |ρ| for the same strength of association (τ = 0.50 already represents strong monotonic dependence).',
      },
    ],
  },

  // =========================================================================
  // STUDIO 05 — DEEP TEXTBOOK SUPPLEMENT
  // =========================================================================
  '05': {
    deepTheory: [
      {
        heading: '5.1 Feature Representation Geometry: Why Good Features Beat Bigger Models',
        paragraphs: [
          'Every machine learning algorithm searches for decision boundaries inside the coordinate space defined by your features X. If true target physics follows Power = Voltage × Current (multiplicative interaction) or Efficiency peaks smoothly at 2:00 PM daily (periodic 24-hour cycle), a linear model on raw features fails completely, and even a tree ensemble must waste dozens of staircase splits approximating a single smooth curve or diagonal product.',
          'Studio 05 transforms raw columns into mathematically expressive coordinates: (1) Yeo-Johnson Power Transforms find the maximum-likelihood exponent λ mapping skewed positive, zero, OR negative features into near-Gaussian symmetry; (2) Pairwise Interactions (x_i · x_j) and Ratios unlock multiplicative physics directly for linear and GAM models; (3) Cyclical Trigonometric Projections (sin(2πt/P), cos(2πt/P)) preserve circular continuity so Hour 23 and Hour 0 sit adjacent at distance 2·sin(π/24) instead of jumping discontinuously by 23 units!',
        ],
      },
      {
        heading: '5.2 Why 4-Method Consensus Selection Eliminates Single-Metric Blind Spots',
        paragraphs: [
          'Every individual feature selector has a known mathematical blind spot: ANOVA F-test only sees linear correlation and misses U-shaped nonlinear features; L1 Lasso arbitrarily drops one feature out of a correlated pair; Tree Impurity Importance biases toward high-cardinality continuous columns; Mutual Information has higher estimation variance on small samples.',
          'Running Studio 05 in Complete (4-Method Consensus) mode ranks every predictor across ALL FOUR complementary lenses simultaneously — Mutual Information (nonlinear information theory), F-Score (linear SNR), Tree Importance (interaction-aware splits), and L1 Regularized Magnitude (sparse conditional contribution) — and fuses them via rank-averaging. Features scoring in the top ranks across multiple paradigms are genuinely robust predictors, while consistently bottom-ranked noise columns can be safely masked via Reversible Exclusions.',
        ],
      },
    ],
    workedExamples: [
      {
        title: 'Worked Example 5A: Why Cyclical sin/cos Encoding Beats Raw Hour (0–23)',
        scenario: 'Electricity demand at 23:00 (11 PM) and 00:00 (Midnight) is nearly identical (~410 MW), yet raw integer hour encodes them as 23 and 0.',
        stepByStepCalculation: [
          'Raw Euclidean distance between 23:00 and 00:00: |23 − 0| = 23.0 (maximum possible distance — farther apart than Noon and Midnight!).',
          'Cyclical encoding (P = 24): Hour 23 -> (sin(2π·23/24), cos(2π·23/24)) = (−0.259, +0.966). Hour 0 -> (sin(0), cos(0)) = (0.000, +1.000).',
          'Cyclical Euclidean distance between 23:00 and 00:00: √((0 − (−0.259))² + (1.000 − 0.966)²) = 0.261 — IDENTICAL to the 1-hour distance between 12:00 and 13:00 (0.261)!',
        ],
        numericalConclusion: 'Cyclical encoding eliminates the artificial midnight/year-end boundary cliff completely.',
        engineeringAction: 'Enable Datetime Cyclical Features in Studio 05 whenever your table includes timestamps, hour-of-day, day-of-week, or month-of-year.',
      },
    ],
    uiControlsGuide: [
      {
        control: 'Feature Engineering Switches (Interactions / Polynomial / Skew Correction / Datetime)',
        defaultValue: 'Interactions=ON, Skew=ON, Datetime=ON, Poly=OFF',
        whenIncreased: 'Enabling Polynomial degree 2 adds quadratic curves (x²) and cross-products (rapidly expands p -> p(p+3)/2).',
        whenDecreased: 'Disabling Interactions/Poly keeps feature count compact and highly interpretable.',
        expertGuidance: 'On wide datasets (>30 numeric columns), keep Polynomial OFF unless you prune to top 10 predictors first to avoid curse of dimensionality.',
      },
      {
        control: 'Feature Selection Method & Top-K Cutoff',
        defaultValue: 'Auto / 4-Method Consensus (Top K = 10)',
        whenIncreased: 'Larger Top-K retains weaker secondary predictors (safer for tree ensembles).',
        whenDecreased: 'Smaller Top-K produces ultra-lean, low-VIF equations (ideal for Studio 03/06 analytical formulas and small n).',
        expertGuidance: 'Follow the heuristic K <= min(p, n / 15) for linear/kernel models to prevent overfitting.',
      },
    ],
    troubleshootingPlaybook: [
      {
        symptom: 'Feature Engineering created 80+ columns and AutoML CV score dropped',
        mathematicalCause: 'Curse of dimensionality — dozens of weak polynomial/interaction columns diluted tree splits and inflated variance on small sample size n.',
        exactRemediationSteps: 'Immediately run Studio 05 Feature Selection in Complete (4-Method Consensus) mode to keep only the top K = 8–15 highest-signal features (reversibly excluding the rest!).',
      },
    ],
  },

  // =========================================================================
  // STUDIO 06 — DEEP TEXTBOOK SUPPLEMENT (AUTOML + NONPARAMETRIC + DL + EXTERNAL EQUATIONS)
  // =========================================================================
  '06': {
    deepTheory: [
      {
        heading: '6.1 How Multi-Backend AutoML Works Under the Hood & Why Out-of-Fold Stacking Prevents Overfitting',
        paragraphs: [
          'OmniForge Studio 06 combines 4 native paper-grade AutoML engines with 5 plug-and-play external adapters (FLAML, AutoGluon, H2O, TPOT, Auto-Sklearn — with graceful native fallback if an optional package is absent). Rather than trusting a single heuristic algorithm, AutoML conducts stratified/K-fold Cross-Validation across diverse model families (Regularized Linear, Random Forest, Extra Trees, Histogram Gradient Boosting, Multi-Layer Out-of-Fold Stacking, and Evolutionary Pipeline Search), strictly benchmarking every candidate against the Strong Baseline established in Studio 03.',
          'In Multi-Layer Stacking (inspired by AutoGluon-Tabular / Wolpert’s Stacked Generalization), Level-1 base learners (e.g., GBDT, ExtraTrees, RandomForest, Ridge) NEVER feed in-sample training predictions to the Level-2 meta-learner — because overfitted trees would receive 100% weight! Instead, each base learner generates strictly Out-of-Fold (OOF) predictions on held-out folds, and a regularized Level-2 meta-learner (Ridge/Logistic) learns optimal nonnegative/shrunk combination weights minimizing generalization error.',
        ],
      },
      {
        heading: '6.2 Using Your AutoML Champion Relationship OUTSIDE OmniForge (Exact vs. High-Fidelity Portable Equations)',
        paragraphs: [
          'Engineering, finance, and industrial teams frequently need to run the learned mathematical relationship outside Python — inside Microsoft Excel spreadsheets, SQL data warehouses, MATLAB/Simulink simulations, C/C++ firmware, or PLC/SCADA controllers. Immediately after AutoML finishes, OmniForge extracts the Champion’s full ModelEquationReport and opens the External Deployment Formula Kit.',
          'There are two mathematical cases to distinguish carefully: (Case A — Exact Closed-Form Champion): When Linear/Ridge/Lasso/ElasticNet/Logistic wins AutoML (or when you select a Linear/GAM baseline), the exported Python, Excel, SQL, and MATLAB/C equation is 100% mathematically identical to the fitted engine (zero approximation error!).',
          '(Case B — Non-Linear Ensemble / Tree / Neural Champion): Tree ensembles (GBDT, Random Forest, Extra Trees, Stacking) combine hundreds of axis-aligned partition rules T_m(x). OmniForge displays BOTH: (1) the exact structural ensemble formula + representative IF/ELSE decision tree rules, AND (2) a High-Fidelity Closed-Form Analytical Surrogate Equation (ŷ_approx = β₀ + Σ β_j · x_j fitted directly on the Champion’s predictions in raw physical units) accompanied by its exact Surrogate Fidelity R²! Read Fidelity R² carefully: R² >= 0.90 means the portable closed-form Excel/SQL/MATLAB/Python formula reproduces >= 90% of the AutoML Champion’s predictions directly; if R² < 0.80, strong non-linear interactions exist — inspect Studio 06 Non-Parametric B-Spline GAM or download the full model/rules rather than relying solely on a linear approximation.',
        ],
      },
      {
        heading: '6.3 True Non-Parametric Analytical Models vs. Black-Box Trees & Deep Learning',
        paragraphs: [
          'Most tools force a false choice between rigid straight-line regressions and opaque black-box tree ensembles. OmniForge bridges that gap with 5 True Non-Parametric Analytical Models: (1) Penalized B-Spline GAM decomposes predictions cleanly into smooth, inspectable 1D nonlinear curves ŷ = β₀ + f₁(x₁) + f₂(x₂) + ... with exact Effective Degrees of Freedom (EDF); (2) Nadaraya-Watson Kernel Smoother computes smooth local expectations via cross-validated Gaussian bandwidth h; (3) RBF Kernel Ridge solves the closed-form Reproducing Kernel Hilbert Space (RKHS) dual linear system (K + λI)α = y capturing smooth multi-way interactions; (4) Distance-Weighted k-NN adapts locally to multi-modal manifolds; and (5) Isotonic Regression (PAVA) enforces guaranteed physical monotonicity (e.g., wear never decreases with operating hours!).',
          'For large, complex, or specialized datasets, Tab 3 provides 8 Paper-Grade Neural Architectures: FT-Transformer (feature tokenizer + multi-head self-attention across columns), TabNet (sequential sparse attentive feature masks), ResNet-Tabular (skip-connected residual blocks with LayerNorm/Dropout), Residual MLP, 1D-CNN (spectral/waveform sequences), and Graph Neural Networks (GCN, GAT, GraphSAGE) for relational/similarity graphs.',
        ],
      },
    ],
    workedExamples: [
      {
        title: 'Worked Example 6A: Reading & Deploying an AutoML External Formula in Excel and SQL',
        scenario: 'AutoML Champion (HistGradientBoosting, CV R² = 0.932) predicts Turbine Heat Rate (BTU/kWh). OmniForge extracts Surrogate Fidelity R² = 0.948 with equation: HeatRate = 8420.5 + 14.20*(ambient_temp_c) − 18.60*(load_mw) + 42.10*(compressor_fouling_idx).',
        stepByStepCalculation: [
          'Step 1 — Check Fidelity R² = 0.948 (>= 0.90 Very High Fidelity!): The closed-form analytical relationship captures 94.8% of the boosted ensemble’s variance.',
          'Step 2 — Evaluate externally for a new plant state (ambient_temp_c = 30 °C, load_mw = 80 MW, compressor_fouling_idx = 2.5):',
          'HeatRate_hat = 8420.5 + 14.20(30) − 18.60(80) + 42.10(2.5) = 8420.5 + 426.0 − 1488.0 + 105.25 = 7,463.75 BTU/kWh.',
          'Step 3 — Paste directly into Excel: =8420.5 + (14.20*B2) - (18.60*C2) + (42.10*D2), OR click Download Standalone .py to drop predict_heat_rate(row) into any external Python service with zero OmniForge installation!',
        ],
        numericalConclusion: 'Exact external evaluation gives 7,463.75 BTU/kWh with 94.8% fidelity to the full AutoML ensemble.',
        engineeringAction: 'Copy Excel/SQL/Python formula directly from the External Deployment Formula Kit; if fidelity R² were < 0.80, also evaluate B-Spline GAM in Tab 2 to inspect the exact curved terms.',
      },
    ],
    uiControlsGuide: [
      {
        control: 'AutoML Engine Selector (4 Native + 5 External Backends)',
        defaultValue: 'native_stacking',
        whenIncreased: 'native_multilayer_stacker achieves maximum accuracy via multi-layer OOF stacking; native_evolutionary explores wider hyperparameter/pipeline trees.',
        whenDecreased: 'native_fast_gbdt optimizes latency on large datasets (>20k rows).',
        expertGuidance: 'Start with native_stacking (fast + accurate), then test native_multilayer_stacker if you need extra lift over baseline.',
      },
      {
        control: 'Time Budget (sec) & CV Folds',
        defaultValue: '25 sec / 3–5 folds',
        whenIncreased: '60–120 sec & 5 folds produces tighter variance estimates and explores deeper candidates.',
        whenDecreased: '10–15 sec & 2–3 folds gives rapid interactive prototyping.',
        expertGuidance: 'Use 5-fold CV for final production champions on datasets < 10,000 rows.',
      },
      {
        control: 'Non-Parametric Mode (Auto / Complete / Custom)',
        defaultValue: 'Auto (All 5 Models Head-to-Head vs Parametric Reference)',
        whenIncreased: 'Complete benchmarks all 5 analytical non-parametric families simultaneously and ranks lift over Ridge/Logistic.',
        whenDecreased: 'Custom runs only selected non-parametric estimators.',
        expertGuidance: 'ALWAYS inspect the GAM per-feature non-linearity index: values > 0.25 prove a feature acts nonlinearly on the target.',
      },
    ],
    troubleshootingPlaybook: [
      {
        symptom: 'AutoML Surrogate Fidelity R² is low (< 0.65) even though AutoML CV R² is high (0.90)',
        mathematicalCause: 'The true target relationship is strongly non-linear or interaction-heavy (e.g., XOR, sharp thresholds, or quadratic valleys), so a single flat linear plane cannot closely mimic the curved ensemble surface.',
        exactRemediationSteps: '1) Inspect the Decision Tree Rules panel inside the Equation Card to see key non-linear thresholds. 2) Run Tab 2 Non-Parametric B-Spline GAM to inspect exact 1D curves. 3) Engineer interaction/quadratic features in Studio 05 and re-run AutoML so the portable equation captures those curves explicitly (raising Fidelity R² > 0.90!).',
      },
      {
        symptom: 'Deep Learning (FT-Transformer / TabNet) underperforms GBDT / AutoML on my dataset',
        mathematicalCause: 'Tabular neural networks require larger sample sizes (typically n > 3,000–5,000) to overcome inductive-bias advantages of gradient-boosted trees on axis-aligned splits.',
        exactRemediationSteps: 'This is expected scientific behavior (Grinsztajn et al., NeurIPS 2022). Keep the highest-scoring validated AutoML or B-Spline GAM champion rather than forcing deep learning on small tables.',
      },
    ],
    extraMethods: [
      {
        name: 'Paper-Grade Tabular Deep Learning (FT-Transformer, TabNet, ResNet-Tabular, 1D-CNN, GNN)',
        paradigm: 'Differentiable Neural Representation Learning',
        whatItDoes:
          'FT-Transformer projects every numeric/categorical feature into an embedding token and applies Multi-Head Self-Attention to learn arbitrary high-order cross-feature interactions. TabNet applies sequential Sparsemax attention masks selecting distinct subsets of features per decision step (built-in interpretability!). ResNet-Tabular applies residual skip blocks h_{l+1} = h_l + F(LN(h_l)) preventing gradient vanishing.',
        formula: {
          equation: 'Attention(Q,K,V) = softmax(Q K^T / √d_k) V   |   ResNet: h_{l+1} = h_l + Dropout(W_2 · GELU(W_1 · LayerNorm(h_l)))',
          symbols: 'Q, K, V = learned query/key/value projections of feature tokens; d_k = head dimension; LN = Layer Normalization.',
        },
        whenToUse: 'Medium-to-large nonlinear tabular datasets (n >= 2,000), continuous sensor/spectral arrays (1D-CNN), or similarity/network-coupled entities (GCN/GAT/GraphSAGE).',
        assumptions: 'Standardized inputs, clean missing-value handling (Studio 01), sufficient training epochs with early stopping.',
        recommendedParams: 'Epochs = 25–60, weight decay = 1e-4, monitor train vs validation loss curve for divergence.',
        pitfalls: 'Overfits quickly on small datasets (n < 800) without strong weight decay/dropout; always compare directly against Studio 03 Baselines & Studio 06 AutoML.',
      },
    ],
  },

  // =========================================================================
  // STUDIO 07 — DEEP TEXTBOOK SUPPLEMENT
  // =========================================================================
  '07': {
    deepTheory: [
      {
        heading: '7.1 Game-Theoretic Explainability: Exact Axioms of Shapley Values (SHAP) vs. Permutation & PDP',
        paragraphs: [
          'SHAP (SHapley Additive exPlanations, Lundberg & Lee 2017) is the UNIQUE local attribution method satisfying three mathematical axioms simultaneously: (1) Local Accuracy (Additivity): the baseline expectation plus the sum of all feature SHAP values equals the exact model prediction f(x) = E[f(X)] + Σ_j φ_j; (2) Missingness: features absent from the model receive φ_j = 0; and (3) Consistency: if a model changes so a feature’s marginal contribution increases across all coalitions, its attribution never decreases.',
          'Global Permutation Importance answers a complementary question: "How much does out-of-sample validation accuracy/R² collapse if I randomly break feature j’s relationship with the target?" Unlike tree impurity importance (which favors high-cardinality noise columns), Permutation Importance measures true generalization impact on held-out data. Partial Dependence Plots (PDP) sweep one feature across its physical domain while averaging predictions over the empirical distribution of all other features — exposing exact non-linear saturation curves, tipping points, and safe operating windows.',
        ],
      },
      {
        heading: '7.2 Algorithmic Fairness Metrics & Mathematical Trade-Offs (Impossibility Theorem)',
        paragraphs: [
          'Studio 07 evaluates group fairness across sensitive attributes (e.g., gender, age bracket, region, facility) using Demographic Parity Difference (DPD), Disparate Impact Ratio (DI — codified in the US EEOC 4/5ths / 80% rule), and Equal Opportunity Difference (TPR gap among genuinely qualified positives).',
          'Chouldechova (2017) and Kleinberg et al. (2016) proved a fundamental mathematical theorem: whenever true base outcome rates differ across two groups, NO imperfect classifier can simultaneously achieve exact Demographic Parity, Equalized Odds, AND Calibration. Therefore, Responsible AI governance requires explicitly declaring which fairness criterion aligns with your legal/domain mandate and documenting both accuracy and fairness metrics transparently inside the Studio 07 ISO/NIST Model Card.',
        ],
      },
    ],
    workedExamples: [
      {
        title: 'Worked Example 7A: Computing the EEOC 4/5ths Disparate Impact Ratio',
        scenario: 'A credit-approval classifier approves 64% of Group A (privileged reference) and 48% of Group B (unprivileged group).',
        stepByStepCalculation: [
          'Step 1 — Demographic Parity Difference: DPD = |0.64 − 0.48| = 0.16 (16 percentage points > 0.10 review threshold).',
          'Step 2 — Disparate Impact Ratio: DI = P(ŷ=1 | Group B) / P(ŷ=1 | Group A) = 0.48 / 0.64 = 0.750.',
          'Step 3 — Compare against EEOC 4/5ths compliance band [0.80, 1.25]: DI = 0.750 < 0.80 -> ADVERSE IMPACT WARNING!',
        ],
        numericalConclusion: 'DI = 0.750 breaches the 0.80 regulatory four-fifths threshold; DPD = 0.16 confirms substantial selection-rate disparity.',
        engineeringAction: 'Audit proxy features via SHAP (e.g., ZIP code proxying protected status), check Equal Opportunity TPR across groups, and adjust group-aware decision thresholds or remove proxy columns via Topbar Exclusions before production deployment.',
      },
    ],
    uiControlsGuide: [
      {
        control: 'Explainability Feature / PDP Selector',
        defaultValue: 'Top important feature',
        whenIncreased: 'Inspect top 3–5 features sequentially to confirm every learned shape matches domain physics.',
        whenDecreased: 'Compare local SHAP on individual mispredicted rows to catch data leakage or edge-case artifacts.',
        expertGuidance: 'Always verify that the top 3 SHAP/Permutation features make physical/business sense — if an ID or post-outcome column tops SHAP, you discovered target leakage!',
      },
      {
        control: 'Fairness Protected Attribute Selector',
        defaultValue: 'Categorical / Group column',
        whenIncreased: 'Audit every sensitive demographic, geographic, or equipment-vendor grouping.',
        whenDecreased: 'Ensure each subgroup has >= 30 rows so group rates have reliable confidence intervals.',
        expertGuidance: 'Run Fairness Audit before publishing any human-impacting scoring model and archive the generated Model Card in Studio 12.',
      },
    ],
    troubleshootingPlaybook: [
      {
        symptom: 'Permutation Importance shows near-zero drop for two obviously critical physical variables',
        mathematicalCause: 'Severe collinearity (VIF > 10): when you permute Feature A alone, its twin Feature B still supplies the same signal to the model, masking both importances.',
        exactRemediationSteps: 'Inspect Studio 03 VIF & Studio 04 Correlation Heatmap, prune redundant twins via Exclusions, or inspect SHAP values which split credit fairly across correlated predictors.',
      },
    ],
  },
};
