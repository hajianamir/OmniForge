# OmniForge — Full-Coverage Audit (Baselines • Preparation/Cleaning • Algorithm & Test-Suite Integrity)

**Branch:** `arena/ead13798-omniforge` (branched from `4f35f2c`) · **Read-only audit of the source tree** — no source files or existing tests were modified or committed by the audit. Audit artifacts (this report + `tests/review/`) were committed to the branch at the user's request; all source/test code was untouched throughout.
**Environment:** Python 3.11.2 (sandbox). Note: the repo declares `requires-python = ">=3.12"` and uses 3.12-only syntax (`class TTLCache[T]` + nested f-string backslash expressions) in exactly two files. The review copy `/tmp/review_copy` carries **two minimal, documented, behavior-preserving syntax-compatibility patches** so the suite can run on the only interpreter available. These patches exist only in the copy. Repo untouched.

**Contents:** Round 1–4 audit (baselines, cleaning, integration) + previous audit notes
+ competitive intelligence & unique feature roadmap (§Competitive Intelligence).

**Test baselines (this environment, identical to pre-existing runs):**
- `tests/unit + tests/integration + tests/e2e`: **11 failed / 134 passed / 1 skipped** (baseline unchanged across rounds)
- Independent review suite: `tests/review/test_baselines_and_algorithms.py` = **10/10 passed**; `tests/review/test_cleaning_preparation.py` = **21/21 passed**

---

## Round 1 — Baselines, benchmarks, metrics & algorithms

### Verified correct (10/10 independent tests + manual recomputation)
| Component | Result |
|---|---|
| `DummyClassifierBaseline` (most_frequent / stratified / prior / uniform) | predictions = mode/priors; proba sums to 1.0 ✓ |
| `LogisticBaseline` | matches sklearn `LogisticRegression(C=1.0, penalty="l1", solver="liblinear", ...)` predictions exactly; proba = 1 ✓ |
| `DecisionStumpClassifierBaseline` / `DecisionStumpRegressorBaseline` | match sklearn `DecisionTreeClassifier/Regressor` exactly ✓ |
| `DummyRegressorBaseline` | equals sklearn mean/median forecast ✓ |
| `RidgeLinearBaseline` equation | reproduces predictions to 5.6e-7 ✓ |
| `ForecastingBaseline` | naive = last value ✓ · seasonal_naive = periodic last-p values ✓ · drift = last + slope·(h+1) ✓ · moving_average = mean(last window) ✓ · intervals lo ≤ yhat ≤ hi ✓ |
| `BaselineBenchmarkSuite.run_classification` | ≥4 models; accuracy/f1 in [0,1] ✓ |
| `BaselineBenchmarkSuite.run_regression` | ≥4 models; r2/rmse/mae consistent with refitting (mean/median cases verified exactly) ✓ |
| `BaselineBenchmarkSuite.run_forecasting` | 4 methods; naive-arithmetic verified ✓ |
| `MetricsEvaluator.regression_metrics` | adjusted R² formula exact (0.544761 vs 0.544761) ✓ |
| `MetricsEvaluator.classification_metrics` | accuracy 0.545833 ✓, f1_weighted 0.548122 ✓ |

### Round 1 findings
1. **HIGH — `RidgeLinearBaseline` is misnamed and mis-implemented vs its name/docstring.** Class docstring: *"Standardized Ridge linear regression strong baseline"* and `model_family = "baseline_regression"` — but the estimator is `LassoCV(cv=3, max_iter=2000, random_state=...)` (**L1 Lasso**, not Ridge), and the `alpha`/`C` knobs it accepts are **silently ignored** (LassoCV recomputes its own alpha by CV). Predictions work and the equation extractor reproduces them, so this is **not** a crash — but any consumer relying on the documented behavior ("Ridge linear regression strong baseline") is mis-specified. Recommend renaming/reimplementing as `LassoLinearBaseline` or actually wrapping `Ridge`.
2. **MEDIUM — `BaselineBenchmarkSuite` silently drops failing baselines.** Both `run_classification_detailed` and `run_regression_detailed` wrap every baseline in `try: ... except Exception: continue` with no report, no log of the failure, and no way for callers to know a baseline silently vanished. When a candidate model fits, the "prove lift" comparison silently undercounts the baseline zoo (e.g., GAM/Nadaraya-Watson failure → 5 of 7 baselines). Recommend surfacing failures (structured `failed_baselines` dict or a loud warning).
3. **LOW — `LogisticBaseline` is L1-penalized** (docstring says only "regularized standardized"), and sklearn emits `FutureWarning: 'penalty' was deprecated in version 1.8`. Cosmetic/API-drift note.

---

## Round 2 — Preparation & cleaning (DataCleaner + splitting + validation + quality + intake/format audit)

### Verified correct (21/21 independent tests + manual recomputation)
| Component | Result |
|---|---|
| Numeric fill values (median/mean/constant) | computed from **train only**, reused in transform ✓ |
| Outlier bounds — IQR / zscore / mad / percentile | exact vs manual recomputation (zscore: mean ± 3·std(ddof=0); mad: median ± 3·1.4826·MAD; percentile: q.01/q.99) ✓ |
| Multivariate outliers (IsolationForest / Elliptic / LOF novelty=True) | works via train-fitted detector; remove/nan/clip/flag actions correct ✓ |
| Causal ffill | never peeks forward; leading NaN fallback = **train's last valid value** ✓ |
| Outlier actions (clip/nan/remove/flag) | all four behave correctly ✓ |
| Exact-dup removal | exact dup rows removed (1 removed, 4 remain) ✓ |
| Fuzzy dedup (string column + 3-gram LSH blocking) | removes near-duplicates ✓ |
| Domain bounds | invalid values → NaN → imputed ✓ |
| Missing indicators | `<col>__is_missing` columns added **before** imputation ✓ |
| `multiple_imputation_rubin` | total_variance = within + (1+1/m)·between (verified) ✓ |
| `pool_rubin_estimates` | q̄, W, B, total, SE all correct ✓ |
| `benford_digit_test` | uniform data → rejects Benford; Benford-conforming → passes ✓ |
| `cleaning_distortion_audit` | KS/Wasserstein/median-shift per column ✓ |
| Stratified split | disjoint train/test + balanced strata ✓ |
| Time-series split | ordered, train < test, gap enforced, never leaks forward ✓ |
| Group split | train/test disjoint groups ✓ |
| Split audit report | passed=True, index_overlap_count=0 ✓ |
| Validation (`range`/`not_null`/`in_set`/`unique`) | pass honestly ✓ |
| `auto_generate_suite` | sane: no critical contradictions against its own data ✓ |
| `DataQualityProfiler` | overall_score = weighted avg of non-None dimensions that have weights ✓ |
| Format audit + Initial intake audit | work; auto_clean_preset has required keys (mode/imputation/outliers) ✓ |

### Round 2 findings
1. **HIGH — `ValidationRule` shorthand is broken for most advertised rule types.** `ValidationRule(name=..., rule_type="regex"|"dtype"|...)` converts to `Expectation(expectation_type=<raw string>)`, which never matches any branch in `_evaluate_one` → **"Unknown expectation type 'regex'" → always fails**. Documented/supported only: `range` (→ `expect_column_values_to_be_between`), `not_null` (→ `expect_column_values_to_not_be_null`), `in_set` (→ `expect_column_values_to_be_in_set`), `unique` (→ `expect_column_values_to_be_unique`). The class docstring advertises `regex` but the branch does not exist. **Impact:** any caller using the shorthand with `regex`/`dtype`/`increasing`/`ks_test`/`pair_greater` gets a 100%-failing suite. Verified with 2 independent tests.
2. **HIGH — The cleaning report shown to users is a fabricated pipeline log.** `api/workbench.py` hardcodes `steps_applied` ("4. Remove rows exceeding missing-value threshold", "7. Detect & handle statistical outliers") — these **do not correspond** to the actual DataCleaner pipeline, which: normalizes missing tokens → cleans strings → infers types → inf→NaN → domain bounds → drops high-missing cols → dedups → fuzzy dedup → indicators → **outliers** → **imputation**. There is no row-level "missing-value threshold removal" step anywhere in the codebase, and the documented order (outliers *after* imputation) is backwards vs the actual order (outliers *before* imputation). The SPA renders this list verbatim in Studio 01's "Step-by-Step Pipeline Log".
3. **MEDIUM — Fit/transform asymmetry in `DataCleaner`.** `fit()` runs `clean_strings` **before** `standardize_missing_tokens`; `transform()` runs `standardize_missing_tokens` **before** `clean_strings`. Verified with the same fixture → different in/out behavior for HTML-wrapped missing tokens ("`<b>None</b>`" becomes NaN in fit's internal stats but survives as the literal string "None" in transform output → stats and final cleaned data do not agree). Also `fit()` dedups before dropping high-missing columns; `transform()` drops high-missing columns before deduping — minor but asymmetric. Recommend making both pipelines iterate the identical stage list from one shared definition.
4. **MEDIUM — Quality-report duplicate `proxy_accuracy` key.** See Pass 2 findings (D3): `dimensions` carries both `accuracy` and `proxy_accuracy` (identical value), totaling inconsistent counts vs the report contract; the duplicate has no weight in `weights` and is silently ignored in scoring while always being emitted.
5. **LOW — auto-generated validation suite carries no rule names** (`description` is never set from `ValidationRule.name` in `auto_generate_suite`; same gap in `_evaluate_one`'s `expectation_name` = expectation_type, so reports cannot tell which rule failed).

---

## Round 3 — Full-suite re-verification + review-suite re-run

| Suite | Result |
|---|---|
| Full repo suite (unit+integration+e2e) | **11 failed / 134 passed / 1 skipped** (identical to the original baseline) |
| Review baselines suite (independent) | **10/10** |
| Review limitations/cleaning suite (independent) | **21/21** |
| Failure-timeline verification (repo suite) | Same 11 failures as the pre-pass baseline |

### Baseline failure taxonomy (unchanged from prior passes — see below)
1. `test_comprehensive_reporting` — ReportGenerator crashes on `None` quality dimension (report/generator.py:260 `float(None)`)
2. `test_data_layer` — duplicate `proxy_accuracy` key breaks the dims-set assertion
3. `test_modeling_and_decision_engines` — baselines save/load via `safe_joblib_load_bytes` blocked by over-broad unpickler blocklist (`builtins.getattr`)
4. `test_p0_security_and_runtime_hardening` — 405 on Studio 09 endpoint
5. `test_full_platform` — registry `load_model` via `builtins.getattr` SecurityError
6. `test_omniforge_task33` / `task34` / `task35` / `task37` — 422 on Studio 14 POSTs (TYPE_CHECKING body-binding)
7. `test_workbench_spa.py` (2 tests) — Studio 09 endpoints 405/404 (empty router)

---

## Previous audit notes (Pass 1 & Pass 2) — consolidated

### Pass 1 badges (verification suites, all passing)
- Engineering 27/27 · Mining 7/7 · Leakage 9/9 · Finance/causal/drift/fairness 11/11 · Optimization 8/8 · Quality/validation 6/6 — **68/68**
- EncryptionManager AES-256-GCM roundtrip + tamper detection ✓
- JWT HS256 fallback (pyjwt absent) ✓
- auth paths (Bearer / X-API-Key / HttpOnly cookie + CSRF double-submit) ✓
- Pipeline builder (fluent) + executor (HMAC-signed artifact cache, fail-fast, CarbonTracker), CLI (`version`, `domains`, `quality`, `clean`, `eda`, `automl`), SPA static bundle serving ✓

### Pass 2 findings (API surface + security)
- **D1 — Studio 09 router is empty.** `s09_opt_sim.py` = 9-line stub; frontend `StudiosPart3.tsx` calls 6 endpoints (`GET /s09/backends`, `POST /s09/nlp|vision|graph|recommend|survival`) — all 404/405. Studio 09 non-functional. Evidence: 4 review tests, 2 SPA e2e, 1 p0 unit.
- **D2 — Studio 14 body-parse 422s.** `s14_supervisory.py` imports 5 request models under `TYPE_CHECKING` with `from __future__ import annotations` → FastAPI treats `req` as a required query param → 422 on every s14 POST (`quick-train`, `evaluate-file`, `evaluate-sql`, `evaluate-access`, `evaluate-demo-batch`, `evaluate-single`). Confirmed by 2 review tests and 4 e2e tests (task33/34/35/37).
- **D3 — quality.py duplicate `proxy_accuracy` key + `timeliness: None`.** Generator/exporter crash on None dims.
- **D4 — generator.py:260 `float(None)` TypeError** — docx/pdf/xlsx/csv exports blow up.
- **D5 — unpickler blocklist over-broad.** `core/security.py` blocks `builtins.getattr`, `setattr`, `delattr`, `globals`, `locals`, `vars`, `breakpoint`, `input`, `system`, `popen`, `spawn`, `call`, `check_output`, `check_call`, `Popen`, `unlink`, `rmdir`, `remove`, `rmtree` — legitimate sklearn/joblib artifacts then fail with SecurityError at load/save paths (baselines unit test + integration full_platform).
- **D6 — test-side contract mismatch:** `test_omniforge_task37.py:235` sends `inject_faults`/`fault_rate`; `SupervisoryDemoReq` expects `inject_fault_pct`/`noise_scale` (as used by the SPA). Distinct from the body-binding defect; fails its own request contract even once D2 is fixed.

---

## Recommended fix order (repo is read-only — recommendations only)

1. **D2** — move the 5 request-model imports out of `TYPE_CHECKING` in `s14_supervisory.py`. Unblocks 4 e2e tests + 2 review tests; fixes Studio 14 entirely.
2. **D1** — implement the s09 router (`/s09/backends`, `/s09/nlp`, `/s09/vision`, `/s09/graph`, `/s09/recommend`, `/s09/survival`) or remove Studio 09 from the SPA/STUDIOS list.
3. **Unpickler blocklist** — narrow the call-like hazards (`eval`/`exec`/`compile`/`__import__`/`open`/`system`/`popen`/`check_output`/`call`/`Popen` + unlink/rmtree) rather than attribute-access primitives; keep module/submodule-prefix checks.
4. **D3/D4** — drop the duplicate `proxy_accuracy` key; make report generator/exporters emit "n/a" for None dimensions.
5. **ValidationRule shorthand** — map `regex`→`expect_column_values_to_match_regex`, `dtype`→`expect_column_dtype`, etc., or re-declare the advertised rule types in `to_expectation`.
6. **Cleaning report** — make `steps_applied` a truthful rendering of the actual pipeline (or remove it and rely on the DataCleaner report `steps_executed`); make fit/transform iterate the same stage list; remove the nonexistent "rows exceeding missing-value threshold" step.
7. **RidgeLinearBaseline** — reimplement as actual Ridge (or rename/document as LassoCV).
8. **BaselineBenchmarkSuite** — surface silent baseline failures (structured `failed_baselines`).

## Environment notes / limitations

- Sandbox Python is 3.11.2; upstream targets ≥3.12. Review copy carries exactly two syntax-compatibility patches (one PEP 695 generic class in `core/performance.py`, one nested f-string backslash in `evaluation/equations.py`). Repo was never modified.
- The repo's own 11 failing tests are known defects (not regressions); they remain failing identically before/after this audit.
- No tests were executed from the frontend; `frontend/` (React 18 + Vite 6) is not verifiable here (no test runner shipped). API surface verified via FastAPI TestClient instead.

## Bottom line

The OmniForge baseline/cleaning stack is **predominantly correctly implemented** (verified against manual recomputation across 10+ independent checks per component). The genuine blockers are **product-level/integration defects** (Studios 09 & 14, the unpickler blocklist, the quality-dimension duplicate key, the report-generator crash, and the fabricated cleaning-report pipeline log/forwarding step) — not math or statistical correctness. Fix D2 first, then D1, then the unpickler blocklist.

---

## Round 4 — API-level Studio 01 integration verification (real FastAPI + TestClient)

| Check | Result |
|---|---|
| `/api/workbench/s01/upload` (CSV) | 200 ✓ — ingested, format/intake audits ran, metadata + checksum built |
| `/api/workbench/s01/clean` (no split lock) | 200 ✓ — House report contains `steps_executed` (truthful), but `steps_applied` (UI-facing) is **fabricated** |
| `/api/workbench/split-lock` (random, test=0.3, seed=42) | 200 ✓ — audit passed, train/test disjoint, split-lock state built |
| `/api/workbench/s01/clean` with **active split lock** | 200 ✓ — cleaning fit on **train only**, transformed both partitions, `split_safe_used: true`, `split_lock_leakage_safe: true` — occlusion-free |
| Cleaning report (via real API) | `steps_applied` is the hardcoded 7-step list (claims "4. Remove rows exceeding missing-value threshold" + "7. outliers LAST") — **contradicts actual pipeline** (`steps_executed`: standardize → clean strings → infer types → [domain/high-missing/dup/indicators] → **outliers** → imputation) |
| Ingestion engine (file) | CSV ingest ✓ — metadata/checksum/name/format stable, missing file raises `DataIngestionError` ✓ |

**Important nuances documented in the real response:**
- With split lock, the cleaning report `input_shape` is the **train** shape `[56,2]` (not full dataset `[80,2]`) — this is expected behavior for train-fit/test-transform, but the UI's `steps_applied` still claims a full-dataset pipeline.
- Outlier **bound** used for test rows is drawn from train only (verified by reading the cleaner's internals + seeing `outliers_detected: {a: 4}` from train stats). Row-level "rows exceeding missing-value threshold" does not exist in any pipeline stage — the `steps_applied` list invents it.
- The report also exposes `input_checksum`/`output_checksum` which are **the same** on a no-op clean (56→60 rows, identical checksum) — that's actually expected (no rows removed), but the field is named `input_checksum` and is silently used as a data-integrity check by callers; worth noting that it does NOT prove the pipeline ran.

## Thoroughness matrix (data preparation & cleaning)

| Layer | Status | How verified |
|---|---|---|
| `cleaning.py` DataCleaner (fit/transform/fit_transform_split, outlier bounds, multivariate, imputation modes, fuzzy dedup, causal ffill) | **Deep** | 21 dedicated tests + 4 heredoc probes + live API probes |
| `cleaning` fit/transform asymmetry + fabricated `steps_applied` | **Deep** | `transform` source read end-to-end; live API response capture |
| `splitting.py` (stratified/test_series/group/auto, verify_no_leakage) | **Deep** | Unit tests + heredocs |
| `validation.py` (DataValidator, auto suite, gap in regex/dtype shorthand) | **Deep** | 22 tests + source read |
| `quality.py` (6D profile, overall weighted score, timeliness=None, proxy_accuracy duplicate) | **Deep** | 8 tests + source read |
| `format_audit.py` | **Reasonable** | source read + one probe |
| `intake_audit.py` | **Reasonable** | source read + one probe + live API |
| `ingestion.py` (file path branch) | **Reasonable** | probe; API-level via s01/upload ✓ |
| `connectors.py` / `multimodal.py` / `synthetic.py` / `feature_store.py` / `versioning.py` | **Not reviewed** | — (out of scope for cleaning/preparation) |
| `api/workbench.py` s01 endpoint (upload → quality → split-lock → clean → report) | **Deep** | live TestClient probes (this round) |
| `ui/state.py` split-lock + registry paths | **Deep** | code read + live API probes |

**Honest verdict:** data preparation/cleaning is **thoroughly reviewed** at the data-layer + API level. The two least-covered areas are (1) the real-API behavior of Studio 01's endpoint when a split lock is present — partially covered by one probe (replace-split flow works), and (2) the reporting/export path beyond the cleaning report itself (format_audit/intake_audit content is reviewed, not the report builder).

---

## Competitive Intelligence & Unique Feature Roadmap

**Scope:** the main competing platforms against OmniForge (an enterprise analytics & AI workbench covering data ingestion/quality/cleaning, EDA, feature engineering, modeling, AutoML, forecasting, causal inference, optimization, simulation, responsible AI, AutoML, DAG pipelines, MLOps and 18 industry domain plugins). Findings below are condensed from web-sourced market research (G2 reviews, vendor reviews, arXiv and analyst comparisons, 2025–2026); each major claim carries its source. This report is **read-only audit material** — no source changes are made here.

### A. What OmniForge uniquely already has (competitive positioning)

| Strength | Competing platforms |
|---|---|
| Full CRISP-DM lifecycle in **one** workbench: ingest → clean → quality → clustering/mining → baselines → EDA → features → model + DL → responsible AI → reporting → Copilot → MLOps | MLflow (tracking/registry only), Great Expectations (validation only), Evidently (monitoring only), Airflow (orchestration only), PyCaret (AutoML only), Alteryx (ETL/prep only), DataRobot (enterprise AutoML, no real open-source surface) |
| **18 industry domain plugins** (finance, energy, healthcare, manufacturing, retail, transportation, agriculture, climate, telecom, HR, education, real estate, sports, cybersecurity, government, bioinformatics, materials, media) with domain-aware sample data and feature engineering | OmniForge-specific; no competitor ships 18 vertical plugins with reproducible in-repo sample datasets |
| **Leakage-safe pipeline primitives** built into data layer (split-lock, purged CV, time-series gap, group disjointness) | PyCaret is known to have leakage risks in auto feature-engineering; Airflow/Gx are leakage-agnostic; Alteryx/Dataiku have no native anti-leakage contract |
| **SHA-hash-chained audit logs + GDPR/HIPAA/SOX compliance checks**, AES-GCM artifacts, tamper-evident | MLflow OSS: no native audit trail. W&B: 90-day retention, compliance gated to Enterprise tier. Alteryx: no native audit trail. |
| **Green AI carbon tracking** inside AutoML (CarbonTracker on training runs) | No competitor ships carbon/energy accounting as a first-class metric (W&B: compute cost only, per-hour billing) |
| **AI Copilot that is fully offline/deterministic** (no LLM required) wired to 14 platform tools | W&B/Arize/Databricks: cloud-LLM-only copilots; Airflow: no Copilot; PyCaret: no Copilot |
| **Local Feast-compatible feature store + DatasetVersionManager + 14-studio React workbench with server-paginated grid** | Most competitors: either notebooks (PyCaret), cloud catalogs (Databricks), feature-store-only (Feast/W&B), or no UI at all (GX) |
| **AutoML baseline-first contract** ("prove lift over the best baseline" formula) | PyCaret: "5-10% uplift" claims but no lift-vs-baseline contract; H2O/AutoGluon: ensembling, not lift-contract; DataRobot: opaque scoring |
| **No-code ↔ code parity** (every studio step renders reproducible code + full DAG export) | Dataiku/Alteryx: visual spaghetti (version control bad, debugging hard); PyCaret: opaque setup |

### B. Competitive landscape — documented deficiencies (with sources)

**1. MLflow** (open-source MLOps standard) — [devtune, 2026](https://devtune.ai/verticals/mlops-experiment-tracking/mlflow) · [MEDIUM/pauloyc, 2026](https://medium.com/@pauloyc/taking-mlflow-to-production-a-practical-mlops-architecture-870d50b36b3a) · [alphacorp, 2026](https://alphacorp.ai/blog/9-best-mlops-tools-and-platforms-for-production-ml-2026)
- No native RBAC (4 basic permission levels, no team boundaries) — unusable for regulated industries without an external auth layer
- No pipeline orchestration or scheduling — requires Airflow/Prefect/Kubeflow alongside
- UI is functional but visually dated and inflexible (specialized views require custom code)
- No native model-drift/performance monitoring — requires Evidently/Prometheus/Grafana
- Self-hosting overhead (backend DB + artifacts) and cost; stagnant enterprise integration
- `Staging`/`Production` stages deprecated in favor of aliases (asset migration churn)
- Self-hosted MLflow: "artifacts can get expensive", no cost accounting per run

**2. Weights & Biases (W&B)** — [modern-datatools, 2026](https://www.modern-datatools.com/tools/wandb) · [devtune, 2026](https://devtune.ai/verticals/mlops-experiment-tracking/weights-biases) · [xyzeo, 2026](https://xyzeo.com/product/weights-biases) · [trustbeforeintelligence.ai](https://trustbeforeintelligence.ai/vendor/wandb/)
- Per-seat/per-hour pricing ($50/user/month Starter; $60/month Pro; enterprise custom quotes $200–$400/user) — cost scales quickly with parallel experiments
- Free tier: 1 user, ~50 runs, 5GB — effectively a trial
- Proprietary metadata format → **vendor lock-in** (migration to MLflow/Neptune requires re-export of runs/artifacts; export covers subset only)
- Rest-API-first-Python lock-in — limited support outside Python ecosystem
- No native model-serving/inference or feature store/orchestration (out of scope by design) — team must buy 4-5 tools in addition
- RBAC-only authorization, no ABAC, no column-level security, 90-day audit-log retention, no HIPAA BAA or FedRAMP — enterprise compliance gates are Enterprise-tier-only
- Cloud-first — data residency concerns for regulated industries without explicit private-cloud config
- No report anonymization (protocol blocked by double-blind peer review)
- Occasional cloud latency; limited cache/run cleanup tooling; dashboard lag at high metric-volume

**3. DataRobot** — [visionstack, 2026](https://visionstack.visionsparksolutions.com/reviews/datarobot/) · [aitools-directory, 2026](https://www.aitools-directory.com/tools/datarobot-enterprise-ai-modeling/) · [peerspot, 2026](https://www.peerspot.com/products/datarobot-reviews)
- Expensive: $100k+/yr minimum (custom quotes, opaque) — startups/SMBs "can't even test it"
- Steep learning curve; DataRobot-specific terminology overwhelm non-technical users
- Requires highly clean, structured data — it is NOT a cleaning/preparation platform ("You still need to prepare your data")
- Opaque "black box" scores for simple models; slower on large datasets (2GB upload = slower than expected)
- Limited deep-learning control and language support (primarily structured/tabular)
- Enterprise sales process is lengthy; lag time to value
- Narrow feature breadth for research (no causal inference, no simulation, no optimization)

**4. Great Expectations (GX)** — [pebblous, 2026](https://blog.pebblous.ai/report/great-expectations-data-quality/en/) · [trivago, 2023](https://tech.trivago.com/post/2023-04-25-implementing-data-validation-with-great-expectations-in-hybrid-environments) · [theodo, 2025](https://www.theodo.com/en-ma/blips/great-expectations) · [g2, 2026](https://www.g2.com/products/great-expectations/reviews)
- Only 4/17 surveyed data-quality tools are ML-specific; GX is NOT designed for ML training data
- Cannot validate unstructured data (images/audio/point clouds); no class-imbalance diagnosis; no automatic label-corruption detection (only schema/null/range)
- Validation-only: does NOT execute the pipeline or fix broken data design; unmaintained suite becomes noise and is ignored
- Steep learning curve, complex docs, custom checks awkward; v0→v1 migration confusion; 10–20 expectations spiral into unmanageable suite
- No robust cross-table validation; UX/reports poor for tracking quality trends
- Spark performance degradation reported in count/collect on large datasets

**5. Alteryx** — [g2 reviews, 2026](https://www.g2.com/products/alteryx/reviews?qs=pros-and-cons) · [officedev, 2026](https://reviews.financesonline.com/p/alteryx/) · [numerous.ai, 2025](https://numerous.ai/blog/alteryx-alternative)
- Expensive licensing ($4,950+ individual; $10k-$50k+ per server/team); high cost blocks SMB adoption
- Windows-only Designer; dated UI; no native macOS/linux workflow
- Large-dataset performance bottlenecks (it pulls data into memory, not Snowflake/Spark)
- Weak reporting/viz (users export to Tableau/Power BI); weak version control (XML files, no Git-style review)
- Workflow sprawl — visual spaghetti hard to debug; desktop-centric; no native ML/monitoring
- Opaque workflow opacity — business logic locked in proprietary visual workflows, hard to migrate away

**6. PyCaret** — [emergentmind, 2025](https://www.emergentmind.com/topics/pycaret-automl-library) · [nature.com, 2025](https://www.nature.com/articles/s41598-025-02149-x) · [educative, 2026](https://www.educative.io/blog/how-to-automate-ml-workflows-using-pycaret)
- Basic HPO (grid/random); no Bayesian/MCTS; no meta-learning or warm-starts
- Underperforms on multilabel/imbalanced data ("sacrifices accuracy" vs AutoGluon/AutoSklearn)
- Black-box ensembles — interpretability needs SHAP integration
- Fails on tiny (<500 rows) or imbalanced (>1:100) data; overfitting; slow >5 min on high-complexity tabular data
- Does not support RL or custom deep-learning architectures; not scalable beyond notebooks (memory leaks); no distributed computing
- No reproducibility standardization (no fixed resource budgets / containerized runs)
- Known target leakage risk in auto feature-engineering

**7. Apache Airflow** — [altexsoft, 2022](https://www.altexsoft.com/blog/apache-airflow-pros-cons/) · [tutorialswithai, 2025](https://tutorialswithai.com/tools/airflow/) · [g2, 2026](https://www.g2.com/products/apache-airflow/reviews?qs=pros-and-cons) · [trustbeforeintelligence.ai](https://trustbeforeintelligence.ai/vendor/apache_airflow/)
- Python-DAG authoring required; business users cannot self-serve workflows
- Steep learning curve; 2–3 months for junior devs to reach productivity
- Metadata-db bottleneck at scale (thousands of task instances/day); no DAG versioning; no Git-based rollbacks
- Batch-only — 5–30 second scheduler latency; unsuitable for real-time agent/sub-minute coordination
- Single scheduler SPOF; single point of failure without Celery/K8s + HA investment ($1k+/month infra; 10–20h weekly maintenance)
- No native ABAC — RBAC only through web UI; enterprise security requires custom plugins
- No ML-specific orchestration (no data quality, model training, or drift integration)

**8. Dataiku DSS** — [reddit r/dataengineering, 2025](https://www.reddit.com/r/dataengineering/comments/1n28cef/dataiku_dss_the_lowcode_data_engineering_king_or/) · [softwarefinder, 2026](https://softwarefinder.com/artificial-intelligence/dataiku/reviews) · [peerspot, 2026](https://www.peerspot.com/products/dataiku-pros-and-cons)
- Opaque pricing; no public Tiers; expensive license blocks smaller teams/startups
- Steep learning curve (platform breadth); heavy/slow on large projects; visual spaghetti; poor versioning
- Inferred data types create downstream failures; visual recipes need customization; limited deployment guidance to AWS/Azure
- Frequent downtime; technical support inadequate; scale issues with server/query-engine stability
- Limited offline function (requires internet); advanced features gated to Enterprise

**9. AutoGluon / H2O.ai (AutoML)** — [geniusee, 2026](https://geniusee.com/single-blog/automl-frameworks) · [frontiersin, 2025](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1680845/full) · [saashub](https://www.saashub.com/compare-h2o-ai-vs-autogluon)
- AutoGluon: resource-heavy (significant GPU memory); complex deployment; not for edge/on-prem
- H2O AutoML: Java-based architecture (operational overhead for Python teams); rigid preprocessing rules; limited customization; not CI/CD native (H2O Flow is not for pipelines)
- Both: no lift-over-baseline contract; no carbon accounting; no PG-aware AutoML; no offline/degraded-mode

**10. Evidently / Arize (monitoring)** — [paulserban, 2026](https://www.paulserban.eu/blog/post/5-best-model-monitoring-tools-to-combat-ai-drift-in-2026/) · [softwaretestpilot, 2026](https://www.softwaretestpilot.com/blog/ai-in-testing/best-ai-model-monitoring-tools) · [johal.in, 2026](https://www.johal.in/model-monitoring-and-drift-detection-using-evidently-ai-in-python-based-mlops-frameworks) · [confident-ai, 2026](https://www.confident-ai.com/knowledge-base/compare/best-ai-observability-platforms-to-monitor-response-drift-2026)
- Evidently (library): no infrastructure — scheduling, storage, alert routing are yours to build; small-data (<1000 samples) sensitivity; <0.05 slow-drift false negatives in JS divergence
- Arize: cost scales with prediction volume (expensive at 10+ models); 2–4 weeks to full instrumentation + 4–6 weeks of alert-threshold tuning; shallow LLM eval layer; engineer-only UX, cross-functional participation limited; per-use-case custom setup for faithfulness/safety metrics
- Both: no data-preparation, no modeling, no RAI — only monitoring; blast-radius of one drift alarm = whole org

**11. Cloud ML platforms (SageMaker / Vertex AI / Azure ML)** — [startupik, 2026](https://startupik.com/azure-ml-vs-sagemaker-vs-vertex-ai-which-platform-wins/) · [letsdatascience, 2026](https://letsdatascience.com/blog/aws-vs-gcp-vs-azure-for-machine-learning-the-practical-decision-guide) · [cloudexpat, 2026](https://www.cloudexpat.com/compare/sagemaker-vs-azure-ml-vs-vertex-ai/)
- Lock-in: feature stores/APIs/pipelines differ per cloud; models don't export across vendors
- SageMaker: steep learning curve (IAM/VPC/multi-service), cost spiral during experimentation, $15-40% overhead tax on GPU instances
- Azure ML: slow enterprise-process overhead; opaque pricing outside Enterprise Agreements
- Vertex AI: less control, smaller ecosystem, smaller feature-store/deploy options
- None: client-side data preparation, autonomous AutoML contract ("lift over baseline"), Green AI carbon tracking, or native responsible-AI reporting

**12. Cleanlab (data-centric ML)** — [productowl, 2026](https://www.productowl.io/mlops/cleanlab) · [unite.ai, 2026](https://www.unite.ai/10-best-data-cleaning-tools/)
- Data-quality detection, not a platform: no orchestration, no retraining, no data versioning infrastructure ("Studio has no built-in data versioning"), no immutable data blobs, no enterprise-wide management (mastering/verification/verification missing), no full reproducibility env tracking, no data-cleaning (detection only)
- Requires a model/prediction/embeddings — not usable for high-stakes datasets without proper model

### C. Unique deep features to add — high-value, competitive-moat features

**C1. Self-Healing Data Pipelines (guarded auto-remediation)** — *closes the #1 gap: validation-only tools (GX, Dataiku, Great Expectations) → execution*
- Pipeline-side auto-diagnosis ("what broke and why") + automatic fix-application with **diff preview, human approval gate, audit trail, full rollback**
- Guardrails: no PII or domain-bound value is ever rewritten without explicit approval; every fix is hashed + committed to the audit log
- Closes GX's "suite becomes noise" failure mode by fixing root causes instead of escalating

**C2. Model Lineage Passport (hash-chained, one-page audit pack)** — *closes "audit-trail gap" in MLflow/W&B/Cloud platforms*
- Every model registers a single lineage pack: data snapshot hash, cleaning report, code commit, dependencies, config, environment, carbon cost, local/cloud/hybrid deployment metadata
- Regulated industries: one-page evidence-pack export on demand (EU/US/GDPR/HIPAA/SOX)

**C3. Budget-Aware Green-AI AutoML (carbon-first objective)** — *closes "expensive AutoGluon/PyCaret" gap*
- Constraints: time/GPU/carbon budget; objective: "lift over best baseline" (already implemented) + carbon-aware panel (CarbonTracker wired in)
- Passes: explains when to give up (GP-UCB "stop and report" logic with carbon+time+lift tradeoff)

**C4. Fully Offline/Deterministic "Fail-Closed" AI** — *closes "cloud-LLM lock-in" gap*
- Critical decisions (PII gate, unseen-data classifier, approval gates, budget) never depend on LLM availability
- Copilot has an offline planner (already wired); extend to Studio 01-14: "AI Copilot" no longer requires cloud; works in offline/air-gapped/regulated/gov environments

**C5. Drift → Root-Cause attribution engine** — *closes the "shallow LLM eval / engineer-only" gap of Arize/Evidently*
- Combines PSI/KS/Wasserstein drift with **causal graph attribution**: which feature distribution shift is causing the performance degradation
- Outputs actionable "fix-first" recommendations, plus one-click retraining trigger evidence pack

**C6. Industry-Aware Validation and Baselines (18-vertical benchmark pack)** — *closes "DataRobot requires clean data" / "GX is not ML-specific"*
- Pre-defined benchmark validation suites per industry (healthcare: clinical exclusion + sensitivity; finance: model risk/SR 11-7-style validation; retail: churn-loss-aware; climate: temporal-validation; energy: imbalance-heavy; government: geo-significant; etc.)
- One command in any Studio re-validates data quality.

**C7. Digital Twin + Continuous Replay Simulation** — *closes the "no simulation platform" gap across Alteryx/Dataiku/DataRobot/MLflow/W&B*
- Replay production traffic through the same trained model pipeline (inference-time distribution comparison, monitored drift) with Monte Carlo/discrete-event uncertainty over operational KPIs
- One-click "what if new data arrives?" scenario exploration

**C8. PII-first and Privacy-by-Design Execution** — *closes the "no GDPR/HIPAA in orchestration" gap*
- Automatic PII detection (PII columns, format-violations, outliers) through the DAG executor, policy-gated stages (data access policy, blocked training sets), full redaction audit trail

**C9. Cross-industry Market Benchmarks and Leader-Boards** — *closes the "opaque scores" gap of DataRobot/AutoGluon/H2O*
- Unified benchmark harness across the 18 domain plugins: one-command reproducible "industry leaderboard" with metrics/carbon/explainability
- Export as PDF/CSV for investor/market reporting

**C10. Explainability Intelligence (Explainability diff + confidence-aware explanation)** — *closes the "black-box" gap of PyCaret/H2O + "engineer-only" gap of Arize/W&B*
- Diff explainability at model level (Baseline → new model), confidence/uncertainty on explanation itself, attribution to data quality defects, not only to features
- End-user tool; works without code; reports "this model is X% more explainable"

### D. How these features close the specific deficiencies

| OmniForge feature | Closed deficiency (source) |
|---|---|
| C1 Self-Healing Pipeline | GX: validation-only, no fixing; "suite becomes noise" [g2, 2026]; "cannot execute pipeline" [Theodo]; Dataiku: no execution |
| C2 Lineage Passport | MLflow: no audit trail; W&B: 90-day audit log retention, no HIPAA/FedRAMP; DataRobot: "auditability helps" but only in enterprise |
| C3 Budget-Aware AutoML | PyCaret: basic grid/random; no resource standardization; AutoGluon: resource-heavy; H2O: rigid preprocessing |
| C4 Offline deterministic | W&B cloud-first; Databricks/SageMaker/Vertex: cloud-native; MLflow: local but no demo/validation contract |
| C5 Drift→Root-Cause | Arize: shallow LLM eval, 2-4 wk instrumentation; Evidently: no infrastructure, no root-cause attribution; AWS/Vertex: no drift at all |
| C6 Vertical validation | DataRobot: requires clean upstream data; GX: structured-data only, not ML-specific; PyCaret: no leakage contract |
| C7 Digital Twin | Alteryx: no simulation; Dataiku: no simulation; DataRobot: no simulation; Airflow: no simulation; MLflow: no simulation |
| C8 PII-first | Alteryx: no privacy guarantee; Dataiku: no privacy gate; Airflow: no policy |
| C9 Industry leaderboards | DataRobot: opaque scores; AutoGluon/H2O: no lift-vs-baseline contract; PyCaret: black-box |
| C10 Explainability diff | PyCaret: interpretability needs SHAP; DataRobot: black-box; H2O: GUI-centric |

### E. Concrete implementation roadmap (priority order)

| Priority | Feature | Where it plugs in | Estimated effort |
|---|---|---|---|
| P0 | C1 Self-Healing Pipelines | `pipelines/dag_executor.py` + `data/cleaning.py` + report generator | 6–10d (new code + 2 tests) |
| P0 | C2 Lineage Passport | `reporting/generator.py` + new `data/lineage.py` | 3–5d (new code + tests) |
| P1 | C3 Budget-Aware Green-AI AutoML | `automl/automl_pipeline.py` + `core/carbon_tracker.py` | 5–8d (extend existing logic + tests) |
| P1 | C6 Vertical benchmark pack | `domains/__init__.py` + `domains/` plugins + tests | 4–6d (18× new validation suites) |
| P1 | C4 Offline deterministic (extend Copilot) | `copilot/copilot_agent.py` (extend existing; no new deps) | 2–4d |
| P2 | C5 Drift→Root-Cause | `evaluation/explainability.py` + `mlops/drift_detector.py` | 5–7d |
| P2 | C8 PII-first Execution | `core/security.py` + `pipelines/dag_executor.py` | 4–6d |
| P2 | C9 Industry leaderboards | `domains/__init__.py` + new benchmark harness | 4–6d |
| P3 | C7 Digital Twin | `simulation/digital_twin.py` (exists) + new `mlops/replay_simulator.py` | 6–10d |
| P3 | C10 Explainability diff | `evaluation/explainability.py` + new `reporting/explainability_diff.py` | 4–6d |

**Bottom line for business/market impact:** The biggest uncontested spaces for OmniForge in research and market industries are (1) offline/regulated-industry adaptation, (2) data-quality root-cause attribution with PII guardrails, and (3) carbon-aware + lift-aware AutoML with 18-vertical benchmark validation. Competing platforms bury themselves in vendor lock-in, opaque pricing and cloud-first architectures — OmniForge's CRISP-DM workbench, 18 plugins, deterministic Copilot and Green AI footprint make this moat defensible.

### F. Honest research caveats

- Deficiencies are **condensed from public sources** dated 2025–2026; they reflect the vendor landscape at the time of writing and should be verified against primary source claims (vendor docs, G2, analyst reports) before making public claims
- OmniForge's own findings (this report) point at real defects — Studio 09 empty router, Studio 14 body-binding 422s, over-broad unpickler blocklist, fabricated `steps_applied` — that must be fixed before any "flagship" feature can land responsibly
- "Deep" features in C1–C10 are intentionally scoped + roadmapped (C1-C4, C6 are P0/P1), not shipped by this report — no source changes are included here
- No life-or-death or compliance-critical claim should be made from this research; legal/regulatory review is required for D1/D2 (HIPAA, EU AI Act, SOX) before public release
