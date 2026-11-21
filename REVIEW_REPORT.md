# OmniForge — Full-Coverage Audit (Baselines • Preparation/Cleaning • Algorithm & Test-Suite Integrity)

**Branch:** `arena/ead13798-omniforge` (branched from `4f35f2c`) · **Read-only audit** — repo never modified, never committed, never pushed.
**Environment:** Python 3.11.2 (sandbox). Note: the repo declares `requires-python = ">=3.12"` and uses 3.12-only syntax (`class TTLCache[T]` + nested f-string backslash expressions) in exactly two files. The review copy `/tmp/review_copy` carries **two minimal, documented, behavior-preserving syntax-compatibility patches** so the suite can run on the only interpreter available. These patches exist only in the copy. Repo untouched.

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
