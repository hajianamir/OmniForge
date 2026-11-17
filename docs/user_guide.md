# OmniForge v2.0.0 — Complete Enterprise User Guide & Analytical Reference

## 1. System Requirements

- **Python**: Python **3.12, 3.13, 3.14, or 3.15** (64-bit; recommended **Python 3.15.2** or **Python 3.14.10**).
- **OS**: Windows 10/11 (x64), Linux (x86_64/ARM64), or macOS (Apple Silicon/Intel).
- **Core Stack**: NumPy 2.5+, Pandas 3.1+, SciPy 1.18+, scikit-learn 1.9+, Pydantic 2.14+, FastAPI 0.163+, Plotly 7.1+, DuckDB 2.0+, PyArrow 26+.
- **Zero Runtime Node.js Required**: The React 18 / TypeScript Enterprise Workbench is pre-compiled into `src/omniforge/ui/static_app/` and served natively by FastAPI.

---

## 2. Quick Installation & Launch

### Windows One-Click Installation & Standalone `.exe` Build
1. Double-click **`install.bat`** (creates `.venv`, installs dependencies, runs `omniforge check`, optionally compiles `OmniForge.exe` and desktop shortcut).
2. Double-click **`start.bat`** to open the **OmniForge Enterprise Launch Center** (or pass `--full`, `--ui`, `--api`, `--exe`, `--cli`, `--check`, `--test`).
3. Double-click **`build_exe.bat`** anytime to rebuild standalone executables (`--onedir`, `--onefile`, `--all`, `--with-inno`).

### Linux / macOS / Docker
```bash
pip install --upgrade -r requirements.txt
pip install -e .
omniforge check
uvicorn omniforge.api.main:app --host 0.0.0.0 --port 8501
```
Using Docker:
```bash
docker compose -f docker/docker-compose.yml up -d --build
```
- **Interactive Enterprise Workbench**: `http://localhost:8501`
- **FastAPI OpenAPI / Swagger Docs**: `http://localhost:8501/docs` (Healthcheck: `/api/v1/health`)
- **In-App Book-Grade Reference Handbook**: Click **`Help`** in the topbar or **`Complete Handbook`** in the sidebar to open the full multi-chapter interactive manual (Part I Master Atlases + Part II Chapters 01–14, full-text search, Prev/Next chapter navigation, and Print). Additionally, every individual Studio (`01`–`14`), global modal (**Targets**, **Exclude**, **Split Lock**, **Project**), and analytical selector has a contextual **`?`** button that opens directly to that exact page's dedicated textbook chapter (formulas, symbol glossaries, null hypotheses, recommended numeric threshold tables, and application decision matrices).

---

## 3. Exact Step-by-Step End-to-End Workflow (Ordered Sequence to Reach Verified Results)

Follow this CRISP-DM sequence to guarantee zero data leakage, reproducible data lineage, scientifically validated baselines, and production-ready deliverables:

| Step | Studio / Control | Requirement | Exact Action | Why & What Happens |
|---:|---|---|---|---|
| **1** | **Studio 01 · Ingestion** | Mandatory | Load CSV, TSV, multi-sheet Excel (`.xlsx`), Parquet, JSON, SQL DB, MS Access (`.accdb`/`.mdb`), or a built-in benchmark dataset. | Immediately executes `InitialIntakeAuditor` (health grade A–F, missingness, duplicates, outliers, constant/mixed-type columns, skewness, leakage warnings). |
| **2** | **Topbar · Targets & Exclusions** | Mandatory | Confirm Target Column(s) in **Targets** and mask raw IDs / post-outcome leakage fields in **Exclusions**. | Reversibly scopes all 14 studios without modifying Raw Data. |
| **3** | **Topbar · Split Lock** | **Recommended BEFORE Cleaning** | Click **Split Lock**, select Strategy (`Auto`, `Stratified`, `Random`, or `Chronological Time-Series`), Test Ratio (e.g., 20%), and CV Folds, then click **Lock Train/Test Split**. | **Anti-Leakage Gold Standard**: Forces Studio 01 Cleaning and Studio 02 Clustering to fit statistics (medians, means, outlier fences, scalers, cluster centroids) strictly on Train and transform Test without leakage. |
| **4** | **Studio 01 · Quality & Cleaning** | Mandatory | Run **Quality & Format Audit**, inspect **Dry-Run Preview**, pick `Auto` / `Complete` / `Custom` mode, and click **Apply & Commit Cleaning**. | Updates Active Data (`RAW` -> `CLEANED`), preserves Raw Data separately (with Undo & Revert to Raw), and logs every deleted row/column to the Forensic Deleted-Records Audit Log. |
| **5** | **Studio 02 · Clustering, Anomaly & Mining** | Recommended | Run Clustering, Anomaly Screening, and/or Data Mining in `PREVIEW` mode. Optionally **Remove Selected Corrupt Clusters / Anomalies** or **Attach `cluster_id` / `is_anomaly`**. | Isolates faulty operating regimes/outliers before modeling. Lights up the Studio 02 green status LED (`CLEANED + CLUSTER-SCREENED`). |
| **6** | **Studio 03 · Diagnostics, Baselines & Risk** | Mandatory | Set α (e.g., 0.05) and holdout ratio, then run Diagnostics in `Complete`, `Auto`, or `Custom` mode. | Runs normality/association/VIF/VaR-CVaR diagnostics and ranks **both Parametric and Non-Parametric Analytical Baseline Equations** with explicit Lift over Naive Floor, LaTeX & Python copy. |
| **7** | **Studio 04 · Automated EDA & Hypothesis Lab** | Recommended | Run Automated EDA (distributions, correlations, missingness, insights) and run the 17-Test Hypothesis Lab (`Complete` / `Auto` / `Custom`). | Confirms distributional shape, group differences, and statistical significance before feature engineering and model selection. |
| **8** | **Studio 05 · Feature Engineering & Selection** | Conditional | Optionally generate skew-corrected, interaction, polynomial, or datetime cyclical features and run 4-Method Consensus Selection. | Adds engineered features (`ENGINEERED` provenance) and reversibly excludes low-signal noise columns. |
| **9** | **Studio 06 · AutoML, Non-Parametric & Deep Learning** | Mandatory | Train Multi-Backend AutoML, 5 True Non-Parametric Analytical Models (closed-form formulas & GAM decompositions), and/or Paper-Grade Deep Learning architectures. | Benchmarks every model strictly against strong baselines, promotes the Champion to Active Model, and unlocks Studio 07, 11, 12, and 14. |
| **10** | **Studio 07 · Responsible AI** | Recommended | Run Global Permutation Importance, Local SHAP, Partial Dependence Plots (PDP), Fairness Audit, and Model Card generation. | Verifies explainability, domain plausibility, and fairness compliance (Disparate Impact ≥ 0.80). |
| **11** | **Studios 08–11 · Decision AI, Specialized AI, DAG & MLOps** | Conditional | Run Forecasting / Causal ATE / Markowitz / Monte Carlo (S08), NLP / Vision / Graph / RecSys / Survival (S09), DAG Pipeline (S10), or Model Registry & PSI/KS Drift Monitor (S11). | Extends analysis to forecasting, causal policy impact, unstructured modalities, and production governance. |
| **12** | **Studio 12 · Executive Reporting & Export** | Mandatory | Select desired sections from 14 available chapters, preview narratives/tables/charts, and export **PDF, XLSX, DOCX, HTML, Markdown, CSV, JSON, or LaTeX**. | Compiles full end-to-end audit evidence into publication-ready deliverables. |
| **13** | **Studio 14 · Live Supervisory Control Room** | Operational | Choose Champion + Baseline models, set Tolerance Rules (`% Error`, `Abs Error`, `σ-Z`, `Baseline Ratio`, `Batch Breach %`), connect live CSV/XLSX/SQL/MS Access or Probe input, and evaluate. | Monitors real-time operational accuracy, plots Actual vs Predicted vs Baseline SPC limits, triggers `HEALTHY` / `WARNING` / `CRITICAL_ALARM` LEDs, and exports scored audit logs. |

> **Important Question: Must `Split Lock` be enabled BEFORE `Data Cleaning`?**
> - **YES for predictive modeling (AutoML, Non-Parametric Analytical Models, Deep Learning, Diagnostics Baselines, Supervisory Deployment).** If numeric imputation (mean/median/KNN/MICE), outlier fences (IQR/Z-score/MAD), high-missing column thresholds, or scaling/clustering are computed on the entire dataset *before* splitting, test-set statistics leak into training and inflate validation accuracy. Enabling Split Lock immediately after loading data ensures OmniForge runs `fit_transform_split` in Studio 01 and `fit_predict_split` in Studio 02 with **zero test leakage**.
> - **Structural Exception / Pure BI Cleaning**: Purely row-independent formatting fixes (whitespace/HTML stripping, standardizing `"N/A"` tokens) and exact-duplicate deduplication across identical rows are safe either way. If you only need a cleaned spreadsheet export for external BI tools and will not train models, Split Lock is optional.

---

## 4. Complete Guide to All 14 Studios

### Studio 01 · Data Ingestion, Automatic Defect Audit & Cleaning
- **Purpose**: Ingest multi-source datasets, automatically grade data quality at intake, control Train/Test Split Lock, and apply auditable cleaning rules.
- **Applications**: Spreadsheet QA, industrial SCADA/telemetry preprocessing, SQL/Access ETL validation, forensic record exclusion auditing.
- **Execution Modes**: `Auto` (smart defaults), `Complete` (comprehensive cleaning pass), `Custom` (manual control of every imputer, outlier rule, string sanitizer, and threshold).
- **Provenance**: Updates Active Data while keeping Raw Data intact; supports **Undo**, **Revert to Raw**, **Forensic Deleted-Records Audit**, and clean **CSV/XLSX export**.

### Studio 02 · Clustering, Anomaly Screening & Data Mining Lab
- **Purpose**: Discover subpopulations, flag/remove corrupt regimes or multivariate anomalies, mine co-occurrence rules, discover high-signal subgroups, and visualize high-dimensional manifolds.
- **Applications**: Customer/patient/equipment segmentation, isolating faulty sensor clusters before modeling, fault-code association rules, nonlinear manifold visualization.
- **Safety Model**: Always previews first (`PREVIEW_ONLY`). Nothing modifies Active Data until you explicitly click **Remove Selected Clusters**, **Remove Anomalies**, or **Attach Cluster/Anomaly Features**.

### Studio 03 · Universal Diagnostics, Ranked Baseline Equations & Risk
- **Purpose**: Test parametric assumptions (normality, correlation, multicollinearity VIF), measure financial/operational tail risk (VaR/CVaR), and rank mathematical baseline models against naive floors.
- **Applications**: Determining whether linear or non-parametric models fit your data physics, catching severe feature collinearity (`VIF > 10`), obtaining exact inspectable baseline formulas with LaTeX & Python code.

### Studio 04 · Automated EDA & 17-Test Hypothesis Lab
- **Purpose**: Visual distribution/correlation/missingness profiling plus 17 classical & non-parametric hypothesis tests.
- **Applications**: A/B testing, comparing failure vs normal sensor distributions, validating treatment effects across groups, checking categorical independence.

### Studio 05 · Feature Engineering & 4-Method Consensus Selection
- **Purpose**: Create Yeo-Johnson skew corrections, pairwise interactions, polynomial expansions, and datetime cyclical features; rank predictors via Mutual Information, F-Test, Tree Importance, L1 Lasso, Variance Threshold, or 4-Method Consensus.
- **Applications**: Capturing nonlinearities for interpretable models, reducing feature dimensionality and noise before AutoML or kernel modeling.

### Studio 06 · Multi-Backend AutoML, Non-Parametric Analytical & Deep Learning
- **Purpose**: Unified predictive modeling studio combining:
  1. **Multi-Backend AutoML Hub**: 4 native SOTA engines (`native_stacking`, `native_multilayer_stacker`, `native_evolutionary`, `native_fast_gbdt`) + 5 optional external adapters (`FLAML`, `AutoGluon`, `H2O`, `TPOT`, `Auto-Sklearn`).
  2. **5 True Non-Parametric Analytical Models**: `Penalized B-Spline GAM`, `Nadaraya-Watson Gaussian Kernel`, `RBF Kernel Ridge (RKHS)`, `Distance-Weighted k-NN`, and `Isotonic Monotonic (PAVA)` — each with explicit mathematical equations, LaTeX/Python formulas, GAM per-feature curves, and direct parametric benchmark comparison.
  3. **Paper-Grade Deep Learning**: `FT-Transformer`, `TabNet`, `ResNet-Tabular`, `MLP`, `1D-CNN`, `GCN`, `GAT`, `GraphSAGE` with loss curves & feature importances.
  4. **Single-Row & Batch CSV Scoring**.

### Studio 07 · Responsible AI (Explainability, Fairness & Governance)
- **Purpose**: Global Permutation Importance, Local SHAP explanations, Partial Dependence Plots (PDP), Algorithmic Fairness audits (Demographic Parity, Equal Opportunity, Disparate Impact), and ISO/NIST-style Model Cards.

### Studio 08 · Forecasting, Causal Inference, Optimization & Simulation
- **Purpose**: Classical + Deep Time-Series Forecasting (`TFT`, `N-BEATS`, `PatchTST`, `LSTM`), Doubly Robust Causal ATE (`AIPW`), Markowitz Mean-Variance Optimization, and Monte Carlo Risk Simulation.

### Studio 09 · Specialized AI (NLP, Vision, Graphs, RecSys & Survival)
- **Purpose**: Multi-modal analytics covering NLP (Zero-Shot, QA, PII Redaction, Topic Modeling, NER), Computer Vision (CLIP/DINOv2/YOLO/Segmentation), Graph Analytics & Link Prediction (RRF/VGAE), Hybrid Recommender Systems, and Survival/Reliability Analysis (Kaplan-Meier & Cox Proportional Hazards).

### Studio 10 · Visual DAG Pipeline Builder
- **Purpose**: Composable visual Directed Acyclic Graph orchestrator executing `quality -> clean -> eda -> features -> automl` in one auditable sequence.

### Studio 11 · MLOps (Model Registry, Experiments & Drift Monitor)
- **Purpose**: Register versioned models across lifecycle stages (`Development`, `Staging`, `Production`) and audit covariate shift using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests.

### Studio 12 · Executive Reporting & Export Center
- **Purpose**: Compile up to 14 customizable report sections with narrative insights, embedded tables, and charts, exporting to **PDF, XLSX, DOCX, HTML, Markdown, CSV, JSON, or LaTeX**.

### Studio 13 · Context-Aware AI Analytics Copilot
- **Purpose**: Workspace-grounded AI assistant that inspects live dataset statistics, intake defects, diagnostics, active model performance, and supervisory alerts to answer questions and guide next steps.

### Studio 14 · Live Prediction & Engineering Supervisory Control Room
- **Purpose**: Production monitoring & engineering digital-twin dashboard connecting Excel/CSV, SQL Databases, MS Access, Simulated Fault Streams, or Single Sensor Probes to score Champion vs. Baseline predictions against live actuals with real-time tolerance alerting (`HEALTHY`, `WARNING`, `CRITICAL_ALARM`), SPC control charts, and downloadable scored audit tables.

---

## 5. Complete Algorithm, Statistical Test & Rule Selection Reference ("When to Use What")

### 5.1 Data Cleaning, Imputation & Outlier Rules
| Method / Rule | What It Does | When to Use | Key Caution |
|---|---|---|---|
| **Median Imputation** | Fills missing values with column median | Best general default for skewed numeric features or outlier-prone sensors | Ignores cross-column correlations; fit on Train only |
| **Mean Imputation** | Fills missing values with arithmetic mean | Roughly symmetric Gaussian columns with minimal outliers | Distorted by extreme tails; shrinks variance at >15% missing |
| **KNN / Iterative (MICE)** | Estimates missing entries from nearest rows or chained regressions | Moderate missingness (5–30%) with strongly correlated columns | Slower on large wide datasets; require Split Lock first |
| **Forward-Fill (`ffill`) / Interpolate** | Carries last value forward or linearly interpolates | Chronological time-series / IoT streams | Never use `bfill` for causal forecasting (look-ahead leakage) |
| **IQR Fence (1.5× IQR)** | Flags values outside $[Q_1 - 1.5\cdot\text{IQR},\ Q_3 + 1.5\cdot\text{IQR}]$ | Skewed or non-Gaussian univariate distributions | Prefer Clip/Winsorize or Flag over Remove unless confirmed corrupt |
| **Modified Z-Score (MAD)** | Robust median/MAD deviation score ($>3.5$) | Heavy-tailed data where extreme outliers mask standard mean/SD | Safe default when outliers occur in clusters |
| **Z-Score ($\|z\| > 3$)** | Standard deviations from mean | Approximately Gaussian columns only | Unreliable on skewed distributions |
| **Isolation Forest / LOF** | Multivariate anomaly detection across all features jointly | Anomalies that look normal per column individually but abnormal together | Review flagged rows before deleting genuine rare events |

### 5.2 Clustering, Anomaly & Data Mining Algorithms
| Algorithm | Paradigm | When to Use | Key Caution |
|---|---|---|---|
| **K-Means / BIRCH** | Centroid / Hierarchical tree | Fast, roughly spherical numeric clusters; BIRCH scales well to large $n$ | Requires selecting $k$ (use Silhouette sweep) |
| **Gaussian Mixture (GMM)** | Probabilistic elliptical | Overlapping or elongated clusters with differing covariances | Sensitive when features $\gg$ samples |
| **DBSCAN** | Density-based + Noise (`-1`) | Arbitrary non-convex shapes and isolating corrupt noise points (`Cluster -1`) | Sensitive to sharp density variations |
| **Agglomerative / Spectral** | Hierarchical / Graph Laplacian | Nested hierarchies or non-convex manifold rings ($n < 5,000$) | Quadratic scaling on large tables |
| **Isolation Forest** | Random partition depth | Fast general-purpose global tabular anomaly detector | May miss tiny local density pockets |
| **Local Outlier Factor (LOF)** | Local reachability density | Anomalies hiding inside varying-density clusters | Sensitive to neighborhood size $k$ |
| **One-Class SVM** | Smooth kernel envelope | Tight boundary around clean baseline operating regimes | Slower on large $n$; sensitive to scaling |
| **Apriori Association Rules** | Frequent itemsets | Co-occurring fault codes, baskets, clinical comorbidities (`Lift > 1.2`) | High Confidence alone misleads if consequent is common |
| **Subgroup Discovery (WRAcc)** | Weighted Relative Accuracy | Finding interpretable slices with high target rate / risk concentration | Validate top subgroups on holdout |
| **PCA / SVD / t-SNE / Isomap / MDS** | Dimensionality reduction | PCA/SVD for global linear variance; t-SNE for local neighborhoods; Isomap/MDS for geodesic geometry | Never interpret global t-SNE cluster distances linearly |

### 5.3 All 17 Statistical & Hypothesis Tests + Corrections
| Test | Data Type | When to Use | Non-Parametric / Robust Alternative |
|---|---|---|---|
| **Shapiro-Wilk** | 1 Numeric | Highest-power normality check ($n \le 5,000$) | D'Agostino $K^2$ (for larger $n$) |
| **D'Agostino $K^2$** | 1 Numeric | Normality via combined skewness + kurtosis ($n \ge 20$) | Anderson-Darling |
| **Anderson-Darling** | 1 Numeric | Normality with strongest sensitivity to tail deviations | KS Normality |
| **KS Normality / Distribution Fit** | 1 Numeric | Goodness-of-fit against Gaussian / candidate distributions | Empirical quantiles |
| **Welch's $t$-Test** | Numeric × 2 Groups | Comparing two independent means (no equal-variance assumption) | **Mann-Whitney $U$** (skewed / outliers / ordinal) |
| **Paired $t$-Test** | 2 Paired Numeric | Before-vs-after matched measurements (normal differences) | **Wilcoxon Signed-Rank** (non-normal differences) |
| **2-Sample KS Test** | 2 Numeric / Groups | Detects ANY distribution difference (mean, spread, or shape) | — (already non-parametric) |
| **One-Way ANOVA** | Numeric × 3+ Groups | Comparing means across 3+ roughly normal homoscedastic groups | **Kruskal-Wallis $H$** (non-normal / skewed groups) |
| **Levene / Fligner-Killeen / Bartlett** | Numeric × 2+ Groups | Testing variance equality (Levene/Fligner robust; Bartlett strict Gaussian) | Fligner-Killeen (most robust to heavy tails) |
| **Pearson $r$** | 2 Numeric | Linear association between roughly normal variables | **Spearman $\rho$** or **Kendall $\tau$** (monotonic / outliers) |
| **Chi-Square ($\chi^2$)** | 2 Categorical | Independence test on contingency tables (expected counts $\ge 5$) | **Fisher's Exact Test** (small $2\times 2$ tables) |
| **Benjamini-Hochberg (FDR) / Bonferroni / Holm** | Multiple $p$-values | BH-FDR for exploratory discovery batteries; Bonferroni/Holm for strict confirmatory FWER | Always pair with effect size |
| **VIF & VaR / CVaR** | Numeric Matrix / Risk | `VIF > 10` indicates severe collinearity; `CVaR` measures expected tail loss beyond `VaR` | Drop/regularize collinear predictors |

### 5.4 Predictive Models: Parametric vs. Non-Parametric Analytical vs. AutoML vs. Deep Learning
| Family / Model | Functional Form | Best Applications | Watch-Outs |
|---|---|---|---|
| **Ridge Linear / Logistic** | Exact closed-form $\hat{y} = \beta_0 + \sum \beta_j x_j$ (or $\sigma(\cdot)$) | Mandatory baseline, small $n$, linear physics, strict linear interpretability | Underfits curves & interactions |
| **Penalized B-Spline GAM** | $\hat{y} = \beta_0 + \sum_j f_j(x_j)$ (smooth cubic B-splines) | Curved/non-Gaussian relationships where transparent per-feature analytical curves are required | Additive by default (no high-order interactions) |
| **Nadaraya-Watson Kernel Smoother** | $\hat{y}(x) = \frac{\sum K_h(x-x_i)y_i}{\sum K_h(x-x_i)}$ (CV bandwidth $h$) | Low-dimensional ($p < 15$) distribution-free smooth local expectations | Curse of dimensionality for high $p$ |
| **RBF Kernel Ridge (RKHS)** | $\hat{y}(x) = \sum \alpha_i \exp(-\gamma \|x-x_i\|^2)$ | Smooth nonlinear engineering/scientific response surfaces with multi-way interactions ($n \le 5,000$) | $O(n^3)$ scaling on very large $n$ |
| **Distance-Weighted $k$-NN** | $\hat{y}(x) = \frac{\sum_{i \in N_k(x)} w_i y_i}{\sum w_i}$ | Multimodal local manifolds with zero global shape assumptions | Requires feature scaling & feature selection for high $p$ |
| **Isotonic Monotonic (PAVA)** | Non-decreasing/non-increasing step function | Guaranteed monotonic physics (dose-response, sensor calibration, degradation vs time) | Single dominant monotonic axis |
| **Multi-Backend AutoML** | Ensembles of GBDT, Forests, Linear, Multi-Layer Stacking | Highest general-purpose tabular accuracy on mixed heterogeneous datasets | Audit with Studio 07 SHAP before deployment |
| **FT-Transformer / TabNet / ResNet / MLP / 1D-CNN / GNN** | Neural attention / residual / convolutional / message-passing layers | Medium-to-large nonlinear tabular, spectral (1D-CNN), or graph-coupled (GNN) datasets | On small tables ($n < 1,000$), verify lift over GAM/AutoML |

### 5.5 Live Supervisory Tolerance Rules (Studio 14)
- **Max % Error**: Relative deviation $|y - \hat{y}| / \max(|y|, \varepsilon) \times 100\%$ — ideal for scale-invariant engineering KPIs.
- **Max Absolute Error**: Physical-unit bound $|y - \hat{y}|$ — ideal for temperature, pressure, voltage, or currency caps.
- **Max $\sigma$ Z-Score**: Standardized residual $z_i = (y_i - \hat{y}_i) / \sigma_{\text{train}}$ — Statistical Process Control ($|z| \ge 3$ signals rare abnormal excursion under Gaussian noise, $P(|Z| \ge 3) \approx 0.27\%$).
- **Baseline Ratio Envelope**: Bounds $R_i = y_i / \hat{y}_{\text{baseline}, i}$ within $[\text{min}, \text{max}]$ to detect regime drift even if model error looks small.
- **Warning Fraction**: Triggers `WARNING` alarms early (e.g., at $0.75 \times$ critical limit) so operators intervene before critical trips.

### 5.6 Core Mathematical Formulas & Universal Recommended Numeric Threshold Compendium

#### Core Analytical Formulas (Implemented Across Studios 01–14)
- **Modified Z-Score (MAD, Studio 01)**: $M_i = 0.6745 \cdot \frac{x_i - \text{median}(x)}{\text{MAD}(x)}$, where $\text{MAD} = \text{median}(|x_i - \text{median}(x)|)$.
- **Silhouette Coefficient (Studio 02)**: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]$; Calinski-Harabasz $\text{CH} = \frac{\text{tr}(B_k)/(k-1)}{\text{tr}(W_k)/(n-k)}$; Davies-Bouldin $\text{DB} = \frac{1}{k}\sum_i \max_{j \ne i}\frac{\sigma_i + \sigma_j}{d(c_i, c_j)}$.
- **Isolation Forest Anomaly Score (Studio 02)**: $s(x, n) = 2^{-E[h(x)] / c(n)}$, where $c(n) = 2 H(n-1) - 2(n-1)/n$.
- **Apriori Association Lift (Studio 02)**: $\text{Lift}(A \Rightarrow B) = \frac{P(A \cap B)}{P(A)P(B)}$; **Subgroup WRAcc**: $\text{WRAcc}(S) = \frac{|S|}{N}(p_S - p_0)$.
- **Variance Inflation Factor (Studio 03)**: $\text{VIF}_j = \frac{1}{1 - R_j^2}$; **Historical VaR / CVaR**: $\text{VaR}_\alpha = -Q_\alpha(r)$, $\text{CVaR}_\alpha = -\mathbb{E}[r \mid r \le -\text{VaR}_\alpha]$.
- **Benjamini-Hochberg FDR (Studio 03/04)**: Reject $H_{(1)}, \dots, H_{(k^*)}$ for largest $k$ with $p_{(k)} \le \frac{k}{m}\alpha$; **Welch's $t$**: $t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$; **Cohen's $d$**: $d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$.
- **Yeo-Johnson Transform & Mutual Information (Studio 05)**: $I(X; Y) = \iint p(x,y) \log\frac{p(x,y)}{p(x)p(y)}\,dx\,dy$.
- **Penalized B-Spline GAM (Studio 06)**: $\hat{y}(x) = \beta_0 + \sum_{j=1}^p \sum_{m=1}^{M_j} w_{j,m} B_{j,m}(x_j)$ with $L_2$ roughness penalty $\lambda \|w\|_2^2$.
- **Nadaraya-Watson Kernel Smoother (Studio 06)**: $\hat{y}(x) = \frac{\sum_{i=1}^n \exp(-\|x-x_i\|^2 / (2h^2))\,y_i}{\sum_{i=1}^n \exp(-\|x-x_i\|^2 / (2h^2))}$.
- **RBF Kernel Ridge (RKHS, Studio 06)**: $\hat{y}(x) = \sum_{i=1}^n \alpha_i \exp(-\gamma \|x - x_i\|^2)$, $\alpha = (K + \lambda I)^{-1} y$.
- **SHAP & Disparate Impact (Studio 07)**: $\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}[f(S \cup \{j\}) - f(S)]$; $\text{DI} = \frac{P(\hat{Y}=1 \mid A=\text{unprivileged})}{P(\hat{Y}=1 \mid A=\text{privileged})}$.
- **Doubly Robust Causal AIPW (Studio 08)**: $\hat{\tau}_{\text{AIPW}} = \frac{1}{n}\sum_{i=1}^n \left[\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{T_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1 - \hat{e}(X_i)}\right]$.
- **Kaplan-Meier & Cox PH (Studio 09)**: $\hat{S}(t) = \prod_{t_i \le t}\left(1 - \frac{d_i}{n_i}\right)$; $h(t \mid x) = h_0(t)\exp(\beta^\top x)$.
- **Population Stability Index (Studio 11)**: $\text{PSI} = \sum_{b=1}^B (P_b - Q_b)\ln(P_b / Q_b)$.

#### Universal Recommended Numeric Threshold Compendium
| Metric / Criterion | Green / Healthy | Caution / Review | Critical / Action |
|---|---|---|---|
| **Missing Rate per Column** | `< 5%` | `5% – 30%` (KNN/MICE + missingness indicator) | `> 40–50%` (drop unless domain-mandatory) |
| **Skewness $\|\gamma_1\|$** | `< 0.5` (symmetric) | `0.5 – 1.0` (moderate skew) | `> 1.0` (Yeo-Johnson / non-parametric) |
| **Outlier Fences (IQR / MAD / Z)** | Within bounds | `1.5×IQR` / `MAD > 3.5` / `\|Z\| > 3.0` | `3.0×IQR` / `\|Z\| > 4.0` (extreme) |
| **Silhouette Score (Clustering)** | `>= 0.50` (strong) | `0.25 – 0.50` (moderate) | `< 0.25` (weak / no natural clusters) |
| **Multicollinearity (VIF)** | `< 5` | `5 – 10` (regularize with Ridge/Lasso) | `> 10` (severe collinearity — prune/PCA) |
| **Significance Level ($\alpha$)** | `0.05` (standard) | `0.01` (confirmatory) | `0.001` (safety-critical / large $n$) |
| **Effect Size (Cohen's $d$ / $\|r\|$)** | `d >= 0.8` / `\|r\| >= 0.5` (large) | `d = 0.5` / `\|r\| = 0.3` (medium) | `d <= 0.2` / `\|r\| <= 0.1` (negligible) |
| **Apriori Association Lift** | `>= 1.5` (strong positive) | `1.1 – 1.5` (modest) | `<= 1.0` (independent / negative) |
| **Disparate Impact (Fairness)** | `0.80 – 1.25` (EEOC 4/5ths compliant) | `0.70 – 0.80` (borderline) | `< 0.80` (adverse impact violation) |
| **Covariate Drift (PSI)** | `< 0.10` (stable) | `0.10 – 0.25` (moderate shift — investigate) | `>= 0.25` (major drift — retrain) |
| **SPC Residual Z-Score (Studio 14)** | `\|Z\| < 2.0` (`HEALTHY`) | `2.0 <= \|Z\| < 3.0` (`WARNING`) | `\|Z\| >= 3.0` (`CRITICAL_ALARM`) |
| **Survival C-Index (Studio 09)** | `>= 0.75` (strong discrimination) | `0.65 – 0.75` (useful) | `<= 0.55` (near random) |

### 5.7 External Portable AutoML Equations & Studio 14 Signed Root-Cause Attribution

- **Always-Visible Help & Textbook Reference Handbook**:
  - Topbar **Help** button (`? Help`) is pinned in the non-wrapping top-right action cluster (never clipped on 1280px–1920px displays even with Split Lock & Supervisory Alarms active).
  - Every Studio (`01`–`14`) displays both a high-contrast header **`? Guide`** button AND a persistent bottom-right floating **`? Studio XX Guide`** launcher opening directly to that Studio's exhaustive textbook chapter (deep first-principles theory, complete mathematical formulas, step-by-step worked numerical examples, UI controls tuning guide, numeric threshold tables, decision matrices, and troubleshooting playbooks).
- **Using AutoML Relationships Outside OmniForge (Studio 06 & Studio 14)**:
  - Immediately after AutoML completes (and inside Studio 14's live Control Room), OmniForge extracts exact/high-fidelity governing mathematical relationships (automatically unwrapping scikit-learn `Pipeline`, `Stacking`, `Voting`, linear, tree, GBDT, and non-parametric estimators) and exposes an interactive **External Deployment Formula Kit**:
    - **Standalone Python (`.py`)**: Self-contained zero-dependency scoring function (one-click Copy or `.py` file download via `GET /api/workbench/s06/export-formula`).
    - **Excel Formula (`=...`)**: Copy-paste spreadsheet formula mapped to columns `B2, C2, ...`.
    - **SQL `SELECT` Expression**: Ready to run inside PostgreSQL, SQL Server, DuckDB, Snowflake, or SQLite.
    - **MATLAB / C / PLC / SCADA**: Ready for embedded controllers, Simulink, and engineering scripts.
    - **LaTeX**: Publication-ready mathematical notation + Surrogate Fidelity $R^2$ verification.
- **Studio 14 Signed Root-Cause Baseline Deviation Attribution (`+Δ` Increase vs. `−Δ` Decrease)**:
  - Displays BOTH the **Active Champion Governing Equation** and **Reference Baseline Equation** live (`GET /api/workbench/s14/equations`).
  - On every batch evaluation and per-alarm drilldown, computes Antithetic Interventional Root-Cause Attribution:
    $$\phi_j = \frac{1}{2}\Big[\big(f(x_{\text{ref}}[j \leftarrow x_{\text{live},j}]) - f(x_{\text{ref}})\big) + \big(f(x_{\text{live}}) - f(x_{\text{live}}[j \leftarrow x_{\text{ref},j}])\big)\Big]$$
  - Explicitly separates **Factors Driving INCREASE (`+Δ` Above Baseline)** from **Factors Driving DECREASE (`−Δ` Below Baseline)** on a signed **Diverging Tornado Chart**, quantifies **Unexplained Residual** ($y_{\text{actual}} - \hat{y}_{\text{champion}}$) to distinguish measured input shifts from unmeasured external faults or sensor drift, and generates a prioritized **P1..P6 Corrective Engineering Remediation Playbook** plus an interactive **Single-Alarm Record Inspector**.

---

## 6. End-to-End Python SDK Workflow

```python
import pandas as pd
from omniforge.data import DataCleaner, DataQualityProfiler, CleaningConfig
from omniforge.eda import UniversalDiagnosticsEngine
from omniforge.models import NonParametricAnalyticalSuite
from omniforge.automl import AutoMLPipelineSearch
from omniforge.reporting import ReportGenerator

df = pd.read_csv("examples/datasets/01_ecommerce_churn_messy.csv")

# 1. Profile & Clean
quality = DataQualityProfiler().profile(df)
cleaned, clean_report = DataCleaner(CleaningConfig(numeric_imputation="median")).fit_transform(df)

# 2. Diagnostics & Ranked Baselines
diag = UniversalDiagnosticsEngine().run(cleaned, target_cols=["churn"])

# 3. Non-Parametric Analytical Suite (B-Spline GAM, Nadaraya-Watson, RBF Kernel Ridge, k-NN, Isotonic)
np_res = NonParametricAnalyticalSuite().run(cleaned, target_col="churn")

# 4. AutoML Champion
automl = AutoMLPipelineSearch().fit(cleaned, target_col="churn")

# 5. Executive Report Export
payload = ReportGenerator().build_report(
    df=cleaned,
    quality_report=quality,
    cleaning_report=clean_report,
    automl_result=automl,
)
ReportGenerator().export_pdf(payload, "executive_report.pdf")
```
