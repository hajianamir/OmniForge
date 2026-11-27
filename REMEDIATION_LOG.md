# OmniForge audit remediation log

## Executive Summary for Auditor

Audit: 2026-11-27, `53fd5e3`. Branch: `arena/84cfdc36-omniforge`. The original audit is untouched.

| Severity | Closed | Partial | Deferred |
|---|---:|---:|---:|
| S1 | 3 | 0 | 0 |
| S2 | 6 | 3 | 0 |
| S3 | 13 | 0 | 0 |
| S4 | 6 | 0 | 1 |
| **Original findings** | **28** | **3** | **1** |

Phase 1 aligned dispatch and reported evidence. Phase 2 isolated feature scope and guarded
acceptance; optional backend comparability and complete feature evidence remain **partial**
(OF-04/06). Phase 3 unified statistical contracts, but some sibling scan caps remain **partial**
(OF-09). Phase 4 corrected integrity and operator claims; Benford hardening is **deferred**
(OF-31). Ad-hoc OF-13a is **partial** pending dedicated mining-family verification and is not
included in the 32 original findings. The broader run detected a Studio 05 no-op regression;
the follow-up restored legacy no-op behavior while keeping explicit run-only pruning.

Local regression evidence covers all ten §6 strengths at the scope described below, including
locked fitting, estimator/interval checks, state round-trip, security and VIF. The tracked
acceptance and integrity fixtures passed, as did Ruff, CI-extras mypy, latency, Redis and the
SPA build. Official CI parity and a completely verified phase-by-phase full suite are **not**
claimed. See the gate record below and [TRACEABILITY_MATRIX.md](TRACEABILITY_MATRIX.md) for
commit and test-level disposition. No PR was merged and no changelog claims were published.

## Baseline setup

Started at a clean `53fd5e3`. System Python is 3.11, below the declared 3.12 minimum. The standalone Python binary download failed TLS; building CPython from its GitHub source archive instead. Locked npm install and baseline SPA build succeeded with no static asset drift. Playwright's official Chromium CDN failed with ECONNRESET; attempting an npm-sourced Chromium substitute (not the exact CI browser).

## Phase 1 / Iteration 1 / OF-01

Serve an explicit executable catalog distinct from downstream suggestions. Custom chips consume only that catalog. Separate requested modules from modules yielding evidence; report unavailable modules rather than claiming execution. Preserve `all`/`auto` distinctions and model-training authorization.

## Phase 1 / Iteration 1 / OF-22

Reject malformed modes and IDs at the request boundary, including empty custom requests and otherwise ignored selections. Do not alias stale chip IDs to ambiguous analyses. Keep `all`, `complete`, `auto`, and `custom` as explicit values.

## Phase 1 / Iteration 1 / OF-02

Use the existing risk payload vocabulary in React and the narrative, and use `tail_risk_table` as the column evidence rather than adding a second copy. Missing measurement displays as unavailable; no fallback to zero. Leave risk calculations unchanged in this phase.

## Phase 1 / Iteration 1 / OF-03

Preserve numeric baseline maps for SDK consumers and collect a separate per-reference evidence map. Withhold lift unless the intended roster was scored on the same partition; do not silently substitute weaker survivors. Do not expose arbitrary exception text. Keep fitting isolated to development rows and leave preprocessing unchanged.

## Phase 1 / Iteration 1 / OF-11

Use fitted dummies designated by class, require the intended naive roster, and explicitly mark missing floors. No acceptance based on nonexistent comparator. Locked baseline partitions remain authoritative.

Phase 1 browser red (OF-01/OF-02): the new browser scenario reached Custom but could not find the catalog-supplied chip, as expected. An archived copy of `53fd5e3` at `/tmp/of-baseline` preserves the pristine Python baseline while runtime provisioning is still in progress. CPython's first source build lacked OpenSSL, zlib, SQLite and libffi development inputs; an initial `uv --require-hashes` resolved/downloaded the lock, but the editable CI install was blocked by missing zlib. This means the full runtime baseline is not yet established.

Phase 1 baseline update: CPython 3.12.12 was provisioned with local extension builds; the hash-verified `requirements.lock` install and smoke imports succeeded (scikit-learn 1.9.1). The CI extras then installed. Pristine Python suite launched from the archive; Ruff passed. Mypy in the combined developer+Redis environment reports a pre-existing `core/rate_limit.py:36` optional-import assignment error. Browser baseline with local Chromium 158 and a dedicated Vite server: 23 passed. Phase 1 Python fixtures went red on the audit snapshot for missing catalog, malformed mode accepted, unavailable modules attested, missing risk narrative, swallowed baseline failure, and fabricated naive floor. The catalog/risk browser fixture subsequently passed against the change. Baseline/holdout score transport was corrected in the same cluster (OF-29) to avoid reconstructing null or rounded values.

T1 sibling review: AutoML API, CLI, Copilot, and report narrative had assumptions that lift was numeric; Studio 06 also used a zero fallback in React. Patched these consumers with the producer. Remaining sibling sites include deep and non-parametric candidate benches (separate path), introspection/equation extraction, and standalone model heuristics. No exception causes or raw model inputs are rendered.

## Phase 1 — Summary

Seven targeted Python checks passed, including the locked baseline/feature preview integration check, and the catalog/risk browser scenario passed. The full pristine coverage run was started before the baseline fix; it was still running at this checkpoint. Phase 1's code is not an assertion of full §6 or CI verification; see the final roll-up for aggregate results. OF-25's route spelling necessarily changed in the same Studio 03 API documentation diff; OF-29's direct score transport necessarily changed with nullable lift.

## Phase 2 / Iteration 1 / OF-05

Add a modelling-only `ignored_columns` contract; keep statistical scans on the study and keep target and control columns out of predictors. Define constants on the fit partition, not on test. Do not automatically drop IDs or correlations on an advisory heuristic.

## Phase 2 / Iteration 1 / OF-06

Make the fitted L1 regression/classification support explicit, including the possibility of empty support. Keep inclusive strong comparisons; expose feature count and sparse-reference support instead of activating the fold-only selector as a production selector.

## Phase 2 / Iteration 1 / OF-04

Distinguish model selection from statistical superiority. Use same-row paired comparisons only where a defensible independent holdout exists, and withhold validation labels otherwise. Keep validation partitions, preprocessing, and optional-adapter limitations visible.

## Phase 2 / Iteration 1 / OF-24

Compare fitted baselines to fitted dummies on the exact scored cohort; distinguish name/role and point rank from acceptance. Dependent or underpowered holdouts do not support an IID bootstrap.

## Phase 2 / Iteration 1 / OF-20

Select a non-mutating run scope with a locked training cohort. The selection returns explicit request options rather than editing the workspace; studios require an opt-in to consume them. Keep the existing session commit and exclusion/undo behavior.

## Phase 2 — Summary

Ignored-scope, sparse-reference and non-mutating scope tests failed first on the audit snapshot, then eight scope checks passed (including locked splits). The small-holdout fixture independently reproduced a false acceptance on the audit snapshot. Comparison uses paired percentile bootstrap with a Bonferroni family bound (models conditional on trained fits), no IID inference for time/group/unknown design, and visible status for small or unevaluable splits. Existing specialized adapter comparison coverage was updated from macro to weighted F1. Optional backend internal rankings are not uniformly comparable (notably H2O's internal selection): do not give the whole OF-04 row closure credit based on native coverage only. Specialized baseline support is in evaluation metadata; graph/sequence methods are not silently represented as sparse fits. See the final roll-up for gates still in progress.

## Phase 3 / Iteration 1 / OF-09

Publish the actual sampled row count, full frame count, and module cap disclosures, including pair limits when columns alone cannot express omitted comparisons. Preserve quality profiling and baseline cohorts on their own scopes.

## Phase 3 / Iteration 1 / OF-10

Use the Workbench's ten exact integer levels rule, also in the backend hub discovered by search (previously accepted any numeric target up to fifteen values). Preserve explicit task overrides. Cardinality still drives exploratory grouping separately from supervised task selection.

## Phase 3 / Iteration 1 / OF-13

Preserve the existing global Studio 03 family for backward numerical comparability, but name it and disclose family size. Name the Studio 04 battery family separately; single exploratory selection is not a corrected battery.

## Phase 3 / Iteration 1 / OF-14

Centralize interval overlaps and purge/embargo in one kernel. Preserve legacy public list and positional-call conventions with a translating adapter; do not collapse the old symmetric purge into the evaluation API's explicit pre-gap and post-embargo without disclosure.

## Phase 3 / Iteration 1 / OF-12

Use bounded nominal indicators for new numeric-workbench fits instead of ordinal distances. Keep persisted legacy encoding reproducible and preserve sklearn's trained ColumnTransformer in specialist backends.

## Phase 3 / Iteration 1 / OF-23

Retire the duplicated 12-column preparation algorithm in favor of the fitted Workbench implementation. Explicitly document specialist preprocessing and remaining native ordinal handling; do not rewrite persisted model schemas.

## Phase 3 / Iteration 1 / OF-18

Use the uncorrected Pearson statistic for Cramér's V, retaining the existing continuity-corrected or Fisher test for its p-value. Do not alter fairness statistics or VIF.

Phase 2 verification: targeted selection/scope/paired comparison/fitted-schema checks: **13 passed**; mypy found only the same optional-Redis baseline error, and SPA rebuilt successfully. The native strong-failure integration test from Phase 1 is green (and red on pristine). Phase 2 also removed the misleading hard-coded deep baseline framework (OF-28) when exposing its fitted baseline metadata; the dedicated deep benchmark verification follows in Phase 4.

## Phase 4 / Iteration 1 / OF-07

Correct the handbook and migration notes to positional membership and fail-closed index-label realignment. Do not claim an authoritative row-ID writer or modify the existing fail-closed remapping logic.

## Phase 4 / Iteration 1 / OF-08

Make new Split Lock audits strict about exact cross-partition duplicates. Keep the original row membership available for inspection and explicitly mark the warning in the badge, including historical locks. Re-audits after transformations enforce the same duplicate rule.

## Phase 4 / Iteration 1 / OF-15

De-list SDK-only controls in the register and guide without removing or advertising new product pathways. Distinguish forecast interval options already exposed from the separate predictive conformal methods.

## Phase 4 / Iteration 1 / OF-16

Source drift/misattribution: the cited variance maximum is the PSM path; AIPW already uses the sample SD of the influence scores divided by square root n. Document the PSM SATE variance floor and AIPW's small-arm non-cross-fitted fallback. Leave Hájek IPW and both estimators' numerical contracts unchanged.

## Phase 4 / Iteration 1 / OF-17

Treat chi-square MCD as a large-sample screening approximation; require at least max(50, 10p) complete rows. Disclose complete-case denominator, and report fit failures rather than substituting classical covariance and retaining the MCD label.

## Phase 4 / Iteration 1 / OF-19

Relabel the structural index as an uncalibrated screening heuristic and publish its cap/method metadata. Keep its wire keys stable. Do not invent weights or calibration evidence.

## Phase 4 / Iteration 1 / OF-21

Use stratified sampling for capped holdouts when feasible; disclose available and retained class counts and a random fallback. Preserve chronological ordering within the selected indices; no test membership goes into model fitting.

## Phase 4 / Iteration 1 / OF-25

The route and contract were documented with the selector change (Phase 1). Verify the docs point to the catalog and run routes rather than Studio 04's EDA contract.

## Phase 4 / Iteration 1 / OF-26

Remove the dead `penalty=` compatibility paths given the declared scikit-learn minimum. Verify the pinned runtime's active paths; nobody can rely on the deprecated fallback within the supported range.

## Phase 4 / Iteration 1 / OF-27

Document the nominal feature cap includes the infrequent bucket and verify it against a fitted `ColumnTransformer`. No change to its handling of unknown categories.

## Phase 4 / Iteration 1 / OF-28

Use the executed baseline wrapper's framework and implementation rather than a universal sklearn label. Verify alongside deep benchmark evidence; do not label failed fits as running on a particular framework.

## Phase 4 / Iteration 1 / OF-29

Lift can be absent after Phase 1; read the actual reference score and name from evaluation metadata rather than reconstructing it. Add explicit round-trip coverage.

## Phase 4 / Iteration 1 / OF-30

Keep 422 and compatible `detail` while adding a typed, correlated OmniForge error envelope. Validation exceptions may include rejected input or free-form custom exceptions; do not include either in HTTP or logs.

## Phase 4 / Iteration 1 / OF-31

The audit did not reproduce a leading-digit defect. Prefer a regression check for extreme decimal notation and deliberate deferral over an untested rewrite with log10 underflow and exact-power boundary risks.

## Phase 4 / Iteration 1 / OF-32

Document the actual Gaussian random-walk volatility scaling. Do not touch the real split-conformal implementation or imply coverage guarantees for this baseline.

## Phase 3 — Summary

The three additional red fixtures reproduced the high-cardinality ordinal encoding, the thirteenth categorical column loss, and incompatible CPCV interval support. Twelve targeted tests passed after the changes, including existing purge/embargo assertions and numeric inference serialization. SPA rebuild passed. Redis integration: 13 passed against a local Redis 7.4 build (Unix socket instead of CI TCP). Mypy still reports only the known combined-environment Redis typing mismatch. The grep for BH correction also identified three mining candidate families (OF-13a); they now name the adjusted family and its pre-display-truncation size. This sibling has no dedicated red/green result, so it remains separately tracked. The prior native ordinal representation of `BaseOmniForgeModel` remains an explicitly documented path; the distance-baseline Workbench paths use the new nominal representation.

## Phase 3 / Iteration 2 / OF-09, OF-12, OF-23

Sibling review found nested Shapiro and Hodges–Lehmann subsampling inside the reported scan and
native AutoML reference models still receiving `ModelBase`'s ordinal conversion. Mitigation:
carry the nested denominators on ledger rows; fit bounded nominal predictors independently on
validation-training and holdout-training rows. Specialized references receive named features from
the fitted `ColumnTransformer`, not an unnamed matrix. Check that supplied test labels do not
select a model, strong-reference failure still withholds lift, and the Workbench fitted input
schema is unchanged. This is code/data contract work only; no tests were relaxed to accept a
changed holdout. `970b7f3` (+62/−9 across 7 files) also cites OF-18 in its subject; the encoding
changes map to OF-12/23, while the Cramér's V calculation is in `0b6eeb5`.

Red/green: two red failures (missing nested count and two-feature ordinal WeightedKNN), then
seven relevant checks passed. Remaining cap inventory (OF-09 partial): `s04_stats.py` EDA plots,
individual capped tests and hypothesis column slices; `s02_mining.py` first-six distance
correlations and nested sampling in stability/Hopkins. These are not covered by Studio 03
metadata. Do not describe the report scope as exhaustive for those routes.

## Phase 2 / Iteration 2 / OF-05, OF-20

The broader run found `test_noop_selection_does_not_discard_existing_charts` failing: the common
constant pruner had turned a prior Studio 05 no-op commit into a real column exclusion. Restore
legacy selector inputs for preview/commit, retaining train-fitted pruning when `apply_to_run` is
requested. Preserve locked membership, cache invalidation, Undo and full study provenance. Before:
1 failed, 4 passed in E2E fail-fast. After: **4 passed** for no-op, locked commit, locked cohort,
and run-scoped opt-in. `9ccb044`: 2 files, +6/−3. This supersedes the failing result for that fixture,
not for the entire process that had already imported the earlier source.

## Phase 4 / Iteration 2 / OF-04, OF-29

Sibling workflow: Studio 06's prominent green lift badge and “Champion” heading still suggested
acceptance; the equation card also described the strong reference as naive. Use neutral point
lift, identify the selected candidate, show the paired interval or its unavailable reason outside
the optional equation card, and name the actual reference in the card. Do not change the holdout
scores or their selection provenance. A real 16-row test gave a red missing-heading assertion,
then passed against the packaged SPA. The first post-build test expected the display string
`native`; corrected the test to use the API's actual `backend_used` (`omniforge_native`). The
final live run passed **5/5**. This does not certify optional internal search rankings (OF-04 partial).

## Phase 4 — Summary

The first pristine Phase 4 selection was **8 failed, 1 passed**. The mode-invalid envelope case
reached the wrong status (`400`) on pristine, so it is not a clean envelope-only reproduction;
the Phase 3 branch separately reproduced missing `error_code` on a valid validation-failure `422`.
The deep fixture reached the endpoint and failed on missing `holdout_sampling`, not a failed
setup. The route-doc fixture was already satisfied by Phase 1. Direct transport independently
failed on pristine (`0.8` reconstructed instead of `0.8123456789`). After: **12 passed** across
integrity, deep provenance, and exact-score/serialization coverage. The category case confirmed
32 outputs including the infrequent bucket. The exploratory Benford exact-integer scaling check
changed representable values across first-digit boundaries; no tolerance policy is specified.
The committed benign precision test uses interior values at extreme scales and passed. Nonfinite
input and exact-threshold policy remain deferred (OF-31), not a repaired defect.

The MCD guard refuses a small complete-case denominator rather than presenting an asymptotic
percentage as calibrated; the AIPW and PSM estimators were not changed. De-listing paths from
operational controls does not remove their SDK tests. Duplicate audit failures remain visible on
historical and re-audited locks; no implicit repartition is performed. Disclosing deep holdout
sampling changes what users can infer, not the locked membership or train/test fit direction.

Diff: `05ce9ca` (3 files, +43/−3); `5cf9a6c` (35 files, +247/−72 including built SPA and the
OF-04 sibling presentation). Final finding-level statuses, including the earlier phases and
sibling OF-13a, are in `TRACEABILITY_MATRIX.md`. No audit document or changelog edited.

## Executable verification and outstanding gates

- Before production edits the full pristine suite was **started**; it finished later on the
  isolated base snapshot: **664 passed, 3 skipped, 83% coverage** (Redis URL, staged installer,
  and optional PyTorch skips). It was not a completed pre-edit gate for every phase. The phases'
  targeted green runs were 7 + 1 (Phase 1), 13 (Phase 2), 12 (Phase 3), and 12 (Phase 4).
  Overlapping selections are not added together as unique tests. Phase diff stats: `f6ae439`
  28 files +479/−146; `3d16744` 30 files +558/−127; `0b6eeb5` 25 files +419/−159;
  later iterations separately recorded above.
- The exact new red fixtures can be replayed by extracting `git archive 53fd5e3` into a temporary
  directory, copying the corresponding test file into that checkout, and invoking pytest with
  its `src` on `PYTHONPATH`. Run documentation assertions with the archive as the working
  directory. Replay only the relevant fixture on pristine: combined newer suites may replay
  checks fixed in an earlier phase, and `test_sklearn_category_cap_includes_the_infrequent_bucket`
  verifies an existing property rather than requiring a red result.
- Current targeted regression selection: **55 passed**. Re-run the finding fixtures with
  `pytest -q tests/unit/test_standards_audit_*.py` and the named E2E tests in the matrix. Run
  `pytest -q tests --cov=omniforge --cov-report=term-missing --tb=short` for the full coverage gate.
- Hash-locked Python dependency install and import smoke passed. `ruff check src tests`,
  `mypy src/omniforge` (200 files) and `python scripts/latency_budget_benchmark.py` passed on
  local CPython 3.12 with the CI quality extras (without Redis). The separate environment with
  the Redis extra reproduces the baseline optional-module mypy error; this is not claimed fixed.
  Live Redis + shared throttle suite passed **13 tests** against Redis 7.4 Unix socket.
- `npm ci` and `npm run build` passed; served SPA assets regenerated and versioned. Official
  Chromium download failed; npm-sourced Chromium verified **24 deterministic chart/contract
  checks** and **5 live Workbench checks**. The deterministic suite preceded the final neutral
  AutoML copy follow-up; the live suite exercised the rebuilt current SPA. This is not official
  CI-browser-equivalent proof. Docker multi-arch registry/SBOM, Python 3.13, and optional deep
  framework runs were not completed. No GitHub workflow result is claimed.

### §6 strength checks (local scope)

| Strength | Verified evidence or limitation |
|---|---|
| Leakage-first fitting and locked baseline partitions | Splitter and Studio 03 locked cohort tests, feature-scope no-op/commit tests, external supplied-label selection comparison, fitted inference and actual served SPA. No completed full parity before/after each phase. |
| Hájek IPW, HT/ESS/trimming/SMD | `test_p1_12_causal_hajek_abadie_imbens_2sls_and_aipw` in focused run; numerical IPW path untouched. |
| Abadie–Imbens and SATE floor | Same golden estimator fixture and `test_causal_aipw_and_psm_golden_se`; metadata only. |
| Finite-sample and Mondrian conformal | `test_p1_3_split_conformal_exact_order_statistic_and_forecasters` and `test_conformal_aps_mondrian_and_strictly_causal_aci`; doc edits do not change the conformal implementation. |
| Exact/weighted fairness CIs and insufficient evidence | Clustered-vs-IID and missing/single-cluster tests in Phase 4 strength selection; no fairness math changed. |
| Quality effective-weight renormalization | `test_quality_reference_accuracy_and_rule_denominators`. |
| Exclusion/undo/lineage and fail-closed realignment | `test_column_exclusion_truthful_provenance_and_uap_roundtrip`, `test_split_lock_row_id_and_fail_closed`, Studio 05 no-op and run-only integration, signed browser restore. |
| Security/audit baseline | Concurrent hash-chain/logger checks, anchor tamper/headers, signed workspace tamper and live persistence checks, sanitized validation, shared Redis, hash lock. Docker SBOM, official browser and Python 3.13 were not reproduced. |
| ColumnTransformer unknown behavior | Fitted single/batch unseen-category fixture and infrequent-bucket test; transformer still uses `handle_unknown="ignore"`. |
| VIF via lstsq | `test_p1_2_diagnostics_semidev_central_range_and_inf_vif`, `test_normality_vif_nan_and_datetime_nat_handling`; numerical implementation unchanged. |

### Timing clarification

The pristine coverage result is from an isolated archive of the audit commit. Its run began before
the Phase 1 baseline work, not before the earliest selector production edit. Per-phase complete
pre- and post-edit coverage was not achieved. “Before” above means the base source snapshot,
not a completed gate contemporaneous with every edit. Subsequent targeted suite counts do not
stand in for that gate.

### Closeout follow-up: coverage run and Phase 4 / Iteration 3 / OF-21

The earlier full 3.12 coverage process finished at **695 passed, 3 failed, 3 skipped, 83%**.
Its failures were the Studio 05 no-op regression, the first draft (exact-threshold) Benford
fixture, and a strict-equality legacy deep-evaluation assertion that rejected the three new
additive metadata keys. The last test must keep its original 8 evaluation fields and the
train/test membership/encoded-schema assertions, and separately check the new sampling and
class-count values. No production change was made in `dac5073` (1 file, +7/−1). Isolated
reverification of those three reported failures plus the misleading-holdout regression:
**4 passed** in the CI quality environment. The older red full run is not called green.

Fresh chart-fixture browser rerun on the final SPA: **24 passed** with substitute Chromium;
the final real-backend run was **5 passed**. A cross-tree sibling review searched the lift/risk
zero fallbacks (T1), scope/pruning paths from Study to pipelines (T2), factorization/CPCV/FDR
sites (T3), and SDK-only governance methods (T4). Remaining unrelated uses of factorization
(e.g. ingestion and group/simulation codes) were not converted as if they were model distances.
Automated governance's residual recommendation text is guidance, not an available mitigation
button. The remaining cap/adapter/FDR work is explicit in the matrix.

### Final local gate disposition

The second full CPython 3.12 coverage run used the final production source and collected tests
before the test-only commit `dac5073`. It completed with **699 passed, 1 failed, 3 skipped,
83% coverage**; its sole failure was the stale deep evaluation equality assertion described
above. After that run finished, `pytest -q tests --lf --lfnf=none --tb=short` loaded the
updated test and passed (**1 passed**). This verifies the remaining case without weakening
its training-schema or cohort assertions. The skips were the independently tested Redis job,
unstaged Windows installer payload, and unavailable PyTorch. A single uninterrupted green run
on the final test tree and full CI parity (including Python 3.13, official Chromium, and Docker
amd64/arm64 persistence/SBOM) are not claimed. Partial findings remain open for auditor review.
