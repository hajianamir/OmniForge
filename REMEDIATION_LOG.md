# OmniForge audit remediation log

## Executive Summary for Auditor

See the final roll-up appended below and `TRACEABILITY_MATRIX.md` for per-finding outcomes. Audit reference: 2026-11-27, `53fd5e3`; working branch: `arena/84cfdc36-omniforge`. The audit is not edited.

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
