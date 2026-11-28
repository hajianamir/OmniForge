# OmniForge — Round 2 remediation log

## Current summary (supersedes the initial intake below)

**V-01:** implemented `report_unavailable` in `ac7ec24` and corrected Studio 06 wording
in `1fb2d0d`. The [independent auditor closed V-01 at `006038f`](https://github.com/hajianamir/OmniForge/pull/31#issuecomment-7235746160)
under the selected policy after all ten PR checks passed. That verdict is not a release
verdict. The initial empty-scope record below is retained for provenance.

**OF-09, Studio 04 Shapiro only:** separately authorized after V-01. The disclosure fix
is in `dd4901e`; other OF-09 caps and all unrelated Phase 2–4 findings remain partial.

## Initial scope intake (historical — before V-01 selection)

**Disposition: scope gate blocked; no eligible findings established.** The supplied independent verification matrix is explicitly in progress. It uses evidence states, not final finding verdicts. It contains no row assigned `FALSE-CLOSURE`, `REGRESSED-STRENGTH`, or `UNVERIFIABLE`, and the reviewer did not explicitly select any confirmed partial for upgrade. This is not a finding-level closure or release determination.

| Round 2 outcome | Count |
|---|---:|
| Findings reopened | 0 |
| Findings resolved | 0 |
| Reopened findings marked `STILL-PARTIAL` | 0 |
| Fix commits / implementation iterations | 0 / 0 |

## Provenance and preservation

- Audit baseline: `53fd5e339553ac6bf006b1ddb3147f6c09543fbf`.
- Active remediation branch: `arena/84cfdc36-omniforge`. The remote tip matches the auditor's reviewed commit, `fd3647a56e0894ad09f5511d58394d2f8ebe29fc`. The local checkout started at the audit baseline with Round 1 work already present as working-tree changes. Those changes have been left in place, without reset, switch, commit, or push.
- The review names `arena/3ef32075-omniforge`. No matching branch was listed on `origin`, and the supplied excerpts were the only independent verification artifacts available here. No local `VERIFICATION_MATRIX.md`, `VERIFICATION_LOG.md`, or `verification/test_static_residuals.py` was found. Neither the audit nor any verification artifact was edited.

## Scope intake — before iteration 1

Computed the reopen list using the exact verdict filter. No evidence-state label was promoted to a final verdict:

- `Source-supported, runtime pending` is not an assigned `UNVERIFIABLE` verdict. The auditor's Python/browser blockers describe execution limits in that verification environment; they do not, by themselves, reopen any finding.
- The residual and coverage notes for OF-04, OF-06, OF-09, OF-13, OF-13a, OF-14, OF-17, and OF-28, and the policy deferral for OF-31, were not explicitly selected for an upgrade. The DeepBench fallback test gap under OF-21 is also not a qualifying verdict.
- **V-01** is the only additional candidate in the supplied matrix. The auditor reports forced parametric fit failure leaving `0.0` reference evidence and positive acceptance in a controlled method-body harness, plus a Studio 06 API fallback. The matrix calls it an **Open candidate** and expressly withholds a final finding verdict; it does not list it as a new VF defect. The finding was recorded below for handoff, not renamed, assigned a verdict, or treated as authorized production scope.

## Verification and next gate

No auditor adversarial test was run or substituted. No production/test change was made, so the remediation loop did not enter a red/green, full CI-equivalent, or §6 regression run. Neither the auditor's source checks nor prior Round 1 test claims are presented as Round 2 verification.

To enter iteration 1, the record needs an authoritative matrix with qualifying verdicts (or explicit reviewer selection of partials), the associated exact rationale, and the corresponding original auditor tests. If V-01 is confirmed as a newly discovered defect, its original harness is required to reproduce the reference-failure behavior before changing the suite or its API consumer. Any dependency on a frozen finding will be recorded before edits.

[Round 2 traceability matrix](ROUND2_TRACEABILITY_MATRIX.md)

## Updated verification intake — runtime evidence supplied

The auditor now reports an independent CPython 3.12.16 run on `fd3647a`: **700 passed,
3 skipped, 83% coverage**, including **32 standards-audit tests**, plus **5 independent
runtime probes**. The full-suite skips were live Redis, the staged installer payload, and
PyTorch. The frontend build, Ruff, mypy, and latency checks also passed in that environment.
Browser, Python 3.13, and CI/release parity remain outside those results. These are the
updated auditor's results, not a Round 2 replay. The 30 baseline-only failures in the
differential probe are not remediation failures.

The original V-01 runtime harness is now supplied inline, so lack of its source is no
longer the blocker. `test_nonparametric_failure_of_the_linear_reference_still_yields_acceptance`
passes by reproducing the defect candidate: with the linear reference forced to fail, the
suite returns a `0.0` reference, a `1.0` analytical score and lift, and accepts the model.
A pass of this harness is counterexample evidence, not a fixed-behavior assertion. The
probe calls the real suite with stubs; it does not independently exercise the Studio 06
API's zero fallback. Its docstring says `VF-01`; the authoritative matrix identifies the
same probe as **V-01 — Open candidate** and explicitly withholds a final finding verdict.
No separate VF finding was listed and no confirmed partial was selected by the reviewer.

The reopen set therefore remains empty. No test was replayed, no production code was
changed, and no commit or push was made. The remaining scope decision is whether to
explicitly include V-01 or issue qualifying verdicts; the harness no longer needs to be
resupplied for V-01.

## Round 2 / Iteration 1 / V-01 — scope and dependency review

The latest supplied record explicitly selects V-01 and the `report_unavailable` policy. This
supersedes the empty-scope disposition **for V-01 only**. The 667-pass run and its uncommitted
implementation are reported on the auditor branch `arena/3ef32075-omniforge`, not on this
session's `arena/84cfdc36-omniforge`. The latter still has the zero fallback and no copy of
that implementation or its tests. Verified that the worktree's tracked content matched the
published `fd3647a` snapshot and advanced the local branch/index to that ancestor-descendant
tip without rewriting files. The audit and verification artifacts remain read-only.

**Dependency before edits:** Studio 06 API and UI files also carry earlier OF-03/04/05/20/21/29
work. Changes are necessary in their **non-parametric** code only: the suite's null reference,
lift and comparison must reach the API and the UI without numeric fallback, positive status,
or auto-promotion. The packaged SPA must be rebuilt from that UI. Do not change unrelated
AutoML or deep routes, other comparison policies, or any §6 data-partitioning/fit design.
The implementation plan is to replay the auditor's exact quadratic/failed-reference scenario,
then assert the chosen policy against the same inputs; preserve a genuinely measured zero
and retain fitted candidates for inspection without setting a champion.

The shared browser fixture file is extended with one new Studio 06 null-comparison assertion;
no existing assertions for the earlier findings are modified. The authoritative auditor
probe keeps its original counterexample assertions in the replay copy. The new branch
regression uses the same 12-row quadratic/failing-reference fixture and asserts the inverse
policy, with additional route, true-zero and nonfinite-prediction checks.

### Red, implementation, and targeted green

- **Reproduced against `fd3647a`:** the original auditor method probe passed its original
  false-acceptance assertions (one pass means the defect reproduced). Running the corrected
  policy fixtures against the same archived production tree produced four failures: the method
  and API returned `0.0` rather than null; successful zero had no reference status; and
  nonfinite predictions were masked as a zero score. The unchanged auditor probe on the fixed
  source now fails specifically at `assert None == 0.0`, as expected for its inverted oracle.
  The exact auditor API probe body was not supplied; the new route test uses the same real suite,
  stubs, and quadratic relationship and independently checks state and model inspection.
- **Root cause:** missing evidence was represented as a numeric zero at suite initialization and
  in the API. Ranking and equation generation consumed that default unconditionally. Now the
  reference is absent until a fit and finite score are successfully recorded. The suite returns
  null reference/lift/comparison outcome and null champion details when comparison is not
  possible. Successfully fitted candidates retain their scores, rank, equations and inspection
  access. Neither the API nor the Studio 06 display invents a comparison or promotes the
  candidate. Valid zero remains comparable. The logging path records exception type, not raw
  exception text. Existing successful-reference ranking and fit splits were not changed.
- **Scope of fix:** `ac7ec24`, 14 files, +250/−60, including generated SPA chunk renames. No
  audit/verification file, other Studio 06 route, or earlier finding's regression was edited.
- **Green:** the V-01 regression module plus the existing non-parametric e2e selection passed
  **7 tests** (2 deselected). Tests assert null payload/equation comparison, no route promotion,
  continued inspection, a genuinely zero F1 reference, and nonfinite predictions. Code changes
  were developed while the new environment was building; the red and counterexample replays
  were against the unchanged archived `fd3647a`, not reruns against a mutated worktree.

### Verification on this branch

- Local CPython 3.12.12, built from source after the prebuilt release asset redirect was
  unavailable; locked install used `uv pip --require-hashes -r requirements.lock`, followed by
  the CI Python extras. Core imports, Ruff (`src tests`), mypy (200 modules), installed wheel
  resource/typing smoke, and all configured latency/memory limits passed.
- §6 regression selection: **136 passed**. Files: `test_anti_leakage_and_pipeline.py`,
  `test_p1_mathematical_and_statistical_integrity.py`,
  `test_clustered_fairness_and_forecast_calibration.py`,
  `test_phase3_econometrics_conformal_and_governance.py`, `test_audit_metric_provenance.py`,
  `test_format_audit_and_exclusions.py`, `test_audit_anchor_and_browser_headers.py`,
  `test_audit_regressions.py`, `test_remediation_audit_loop.py`, and
  `test_workbench_inference_preprocessing.py`. This covers train/holdout fit invariants;
  Hájek/HT/ESS/trimming/SMD; PSM/SATE; finite-sample/Mondrian conformal; fairness CI and
  insufficient clusters; effective-weight renormalization; exclusion/lineage and fail-closed
  realignment; audit concurrency/tamper and headers; train-fitted unknown-category handling;
  and VIF. Redis integration/shared throttling separately passed **13 tests** using Redis 7.4.9
  over a Unix socket. This does not certify Docker, SBOM or multiversion gates.
- `npm ci` and `npm run build` passed. Substitute Chromium ran **25 deterministic Workbench
  tests** (including V-01 null/reference display) and **5 packaged-backend browser tests**.
  These are executed browser results on this branch, not CI's official Chromium build.

### Final local gate and disposition

`python -m pytest -q tests --cov=omniforge --cov-report=term-missing --tb=short` on the
fixed `arena/84cfdc36-omniforge` tree exited zero: **704 passed, 3 skipped, 83% coverage**
(506.60 s). This is distinct from both the auditor's 700-pass reviewed tree and its 667-pass
other-branch report. Redis skipped in the full run and passed in the separate integration job;
installer payload and PyTorch remain skipped. Python 3.13, official Playwright Chromium,
Docker multi-architecture/SBOM, and the optional framework/release infrastructure were not
verified. Full local coverage and the listed complementary checks are not a CI/release verdict.

**Engineering claim:** V-01 resolved under the selected policy. Before: forced fit failure
could produce zero, positive lift, and champion/active-model claims. After: unavailable
reference and null lift/comparison, no acceptance or auto-promotion, but candidate score and
inspection remain. A real zero is explicitly not missing. No other ID entered Round 2; the
auditor owns severity, independent finding verdicts and release disposition.

## Round 2 / Iteration 2 / V-01 — wording follow-up

The auditor's follow-up identifies two remaining wording mismatches, not a reopened
statistical contract: Studio 06's lineage action and chart/leaderboard headings still
imply a comparison when none exists. Change only the non-parametric wording in the
shared Studio 06 API/UI files and rebuild the checked-in SPA. The available-reference
wording and earlier findings' behavior remain unchanged. Label the initial intake as
historical without rewriting its contents. The prior local test counts remain local
results, not independently reproduced evidence or official CI results.

- **Red/green:** the new API assertion first failed on the comparative lineage action;
  the browser assertion first failed on the missing candidate-only heading. Both passed
  after the wording fix (`1fb2d0d`, 12 files, +24/−12 including packaged SPA assets).
  When comparison is unavailable, lineage now records the number of candidates scored;
  the UI labels the leaderboard and chart as candidate results. Comparison wording is
  unchanged when the comparison is available. Candidate scores and promotion gates are unchanged.
- **Local rerun on `1fb2d0d`:** 7 targeted Python tests and 25 deterministic browser
  tests passed; Ruff, mypy and the SPA build passed. The full Python 3.12 coverage run
  passed with **704 passed, 3 skipped, 83% coverage**; the §6 selection passed **136 tests**.
  The skipped Redis, installer and PyTorch cases match the prior full run. The earlier
  Redis and live-backend browser checks were not repeated for this wording change. These
  results are local engineering evidence, not independent verification or CI parity.

## Round 2 / Iteration 3 / OF-09 — Studio 04 Shapiro disclosure only

The maintainer explicitly approved this item from the auditor's Phase 2–4 worklist.
At `006038f` the auditor reproduced 5,100 rows reported as 5,000 source/scanned;
all ten PR checks passed at that baseline. This authorization does not reopen other
OF-09 caps, other findings, or V-01. OF-09 as a whole stays partial.

**Dependencies logged before implementation:** the Studio 04 router supplies both custom
and per-column battery results; its battery rows are also consumed by Studio 12 preview
and exporters. The shared chart builder has a separate Shapiro cap for the Q–Q plot; the
same payload is reused in the chart UI/gallery and SVG/PNG/XLSX export. The shared Studio
04/05 UI source and compiled SPA are needed for the custom result; existing neighboring
behavior and previous assertions are not to be changed. Preserve the historical test
samples, statistics and correction family. Studio 03's disclosed Shapiro path, intake
pretests, and non-Shapiro caps are outside this selection. No original audit artifacts,
auditor verification files or Round 1 traceability rows will be edited.

### Red/green, disposition and handoff

- Against `006038f`, all seven API cases failed as expected: above the cap the legacy
  `sampled_from` was 5,000, and the report/plot did not expose the sample counts. The
  browser fixture also failed on the missing custom-result disclosure.
- `dd4901e` keeps the Hypothesis Lab's first non-missing rows in study order and the Q–Q
  plot's evenly spaced sorted-rank sample, without changing the Shapiro statistic,
  p-value, or battery membership for evaluable inputs. Available, tested, omitted, cap,
  and selection are structured API fields; original source rows are separate from numeric
  availability. Q–Q point thinning is separate from Shapiro sampling, and a constant Q–Q
  sample reports that no test ran instead of displaying a synthetic result.
- Final focused API tests: **7 passed** using 5,000, 5,001 and 5,100 rows in custom,
  auto and complete modes plus the Q–Q plot; covered incomplete/constant columns,
  Studio 12 preview and JSON/HTML/Markdown/CSV/DOCX/XLSX exports, and Q–Q SVG/XLSX
  export. The statistical values were checked against SciPy on the historical samples.
  Neighboring chart/hypothesis regression run: **34 passed**, including the seven cases
  before the final source-note shortening and added document-format assertions; the
  final focused rerun included those assertions. The §6 selection passed **136 tests**.
- Ruff, mypy (201 modules), and the production SPA build passed. Using substitute
  Chromium, **26 deterministic** packaged Workbench tests (including the new Studio
  04 custom-card and battery assertion) and **5 live-backend** tests passed. Full-suite
  and official PR check outcomes are tracked against the updated PR head in PR #31.
- **Engineering claim:** the approved Studio 04 disclosure residual is addressed;
  this is not a finding-wide OF-09 closure. No other residual was selected. The V-01
  auditor also identified the original pre-fix API probe in the separate audit workspace
  as `verification/test_runtime_adversarial.py::test_studio06_api_exposes_false_acceptance_after_reference_failure`;
  the branch API regression remains distinct and is not labeled a replay of that probe.

## Round 2 / Iteration 4 — authorized Phase 2–4 worklist and remaining OF-09 disclosures

The maintainer authorized the remaining items following the Shapiro-only fix. Baseline
`f0b455a` passed all ten PR checks and 711 local tests. Dependencies logged before edits:
Studio 04 hypothesis payloads and their generic report tables; Studio 02 advanced mining
and diagnostic consumers; the shared Studio 04 chart builder; Studio 06 AutoML adapter
results and display (not the V-01 non-parametric path); and the Studio 01 Benford function
and its cleaning audit. Additional contract and fallback tests are additive only. Do not
change historical audit/verification files or claim new guarantees for MCD, positional
split membership, SDK governance, or uninstalled optional frameworks.

### Selected contracts (not broader guarantees)

- **Phase 2 / OF-04, OF-06:** Compare selected models only on matched held-out
  observations and input scope: weighted F1 for classification, RMSE for regression.
  A backend's native ranking is selection evidence, not a cross-backend ranking;
  AutoGluon/H2O native tables are labeled backend-specific, and are not charted against
  the shared metric. Input count means raw predictors offered to the adapter; a fit count
  means the actual known estimator input dimension; retained count means known selected
  support (e.g. nonzero L1 coefficients), not the raw dimension. Null means unavailable
  or not applicable, not zero. Optional adapter boundary fixtures do not certify installed
  library execution, internal objective equivalence, or feature transformations hidden by
  those libraries. Dependent-split interval calibration is not claimed.
- **Phase 3 / OF-13a:** BH families cover only *generated, eligible* hypotheses. Rules
  must pass support, confidence and lift filters before adjustment, with each direction
  a separate test. Transaction-rule and tabular-rule families are separate; tabular
  item/column/pair exploration is bounded, not exhaustive enumeration. Subgroup candidates
  are the searched bins/categories meeting size constraints against their complement;
  adjustment uses the full eligible set before presentation ranking/truncation. The
  exploratory/data-dependent filtering is not a guarantee of unconditional FDR control.
  The beam-search and standalone FP-growth outputs do not claim adjusted inference.
- **Phase 3 / CPCV:** Preserve distinct public contracts. The canonical API uses 5 folds,
  two test groups, and by default 2 positions before/after; its post-buffer is max of
  explicit and proportional embargo. It streams and omits empty training partitions.
  The legacy API defaults to 6 folds and two test folds, no buffers, and returns a list
  including empty training splits. Its post-buffer adds the symmetric purge gap and
  ceiling-rounded proportional embargo. Both share closed interval overlap purging;
  parity is expected only when inputs and effective buffers/empty behavior match.
- **Phase 4:** Retain MCD as asymptotic screening only, not calibrated inference.
  Positional-only membership and SDK-only governance remain their existing contracts,
  not operational controls. Exercise the real capped/unstratifiable DeepBench API with
  baseline models; PyTorch execution is optional and unverified by this core suite.
  Benford uses positive magnitudes of finite nonzero stored values and their shortest
  decimal spelling. At a leading-digit boundary the stored value decides the digit;
  no epsilon snapping, and no guarantee of invariance across inexact rescaling.

### Verification and disposition

- Archived `f0b455a` plus the new fixtures reproduced nine failed expectations (five
  optional adapter interface cases, the warm-start encoded-dimension contract, and
  the missing Studio 02/04/API and chart disclosures). Five evidence-only cases
  passed on the baseline; those tests do not imply previously broken behavior.
- The focused regression file plus the Shapiro boundary tests passed **21 tests**;
  the added EDA 25,001-row cap test passed separately (**22 covered**). The broader
  focused chart/AutoML/standards run passed **45 tests** before the last additive
  EDA/chart scope fields. The packaged Workbench passed **31 browser tests** using
  substitute Chromium. Ruff, mypy, and the SPA build passed. The full/official gate
  results belong to the updated PR head and will be reported on PR #31.
- The named OF-09 surfaces now distinguish analysis sampling and display limits.
  Remaining uncertainty about optional runtime implementations, calibrated MCD,
  hardware/backend-dependent behavior, or intentionally narrower governance contracts
  is not resolved by the core suite. Finding dispositions and release criteria remain
  the independent reviewer's responsibility.

**Verification follow-up:** the full local run of `8807fd1` exposed a regression in
existing Fisher-z correlation chart wording. The new predictor cap text had replaced
part of the prior inference caveat. Restored the assumption/approximation and exploratory
p-value disclosure alongside the cap, without changing the existing test or calculations.
The original SciPy estimate, confidence interval and wording regression passes again.

## Main audit follow-up — dependency review (before product edits)

The user selected MAIN-01 through MAIN-07 from the independent audit of
`4385647a5ec14806f3c12d6ed3b792069402de64`. Original audit and verification
artifacts stay untouched; the supplied probes are replayed from an isolated archive.

- **MAIN-03 touches the V-01 suite and Studio 06 view.** V-01 is independently
  closed under `report_unavailable`. This change is limited to interpreting
  *available* CV comparisons and naming the display. Preserve null comparison
  evidence, scored candidates, and no acceptance or promotion on reference
  failure. Rerun the reference-failure and zero-reference regressions.
- **MAIN-01/04 touch Studio 03 diagnostics and its view.** Preserve executable
  module dispatch, authorization, and the scope of the evaluable correction
  family. The Holm change adds a supported adjustment, not a new test family.
- **MAIN-05 touches the OF-31 Benford function.** Only the returned p-value
  precision changes; preserve the finite stored-magnitude digit rule.
- **MAIN-02 touches the causal engine.** Preserve valid 0/1 AIPW, continuous
  DML/IV, and the clustered DiD inference contract. Binary estimators must not
  silently reinterpret extra arms or fit when complete cases lose an arm.
- **MAIN-06/07** are formatter and Studio 03 documentation corrections; no
  change to API exception handling, auth, or the historical audit verdicts.

### Main audit replay and remediation evidence

- Extracted only the supplied report metadata and harnesses outside the repository.
  Verified the three original Python harness files against the auditor's SHA-256s.
  Replayed unmodified phase 2 (numerical) and phase 3 (contracts) on a clean
  `git archive` of `4385647`; replayed unmodified phase 3 on the patched worktree.
  The phase 2 multilevel SDK call previously expected a result, so its fixed
  check reused the exact recorded vectors, seed, and estimator options with
  rejection as the expectation; the binary-subset result matched in full.
- Four new Chromium regressions failed as expected against the old packaged
  SPA (short control values, VIF rendering, and current/legacy CV displays).
  They passed after rebuilding. A real-browser regression also submitted fixed
  Studio 03 controls to the API and charted the returned VIF map.
- Focused Python regressions: 55 passed, including V-01 reference failure and
  OF-31 stored-magnitude coverage. Ruff and mypy (201 source files) passed.
  The CI Python extras installed successfully on CPython 3.12.12; numerical
  library versions matched the audit (NumPy 2.5.5, pandas 3.1.1, SciPy 1.18.1,
  scikit-learn 1.9.1 and statsmodels 0.15.0).
- See [the traceability matrix](ROUND2_TRACEABILITY_MATRIX.md#independent-main-audit-follow-up--4385647)
  for the per-finding before/after observations and scope limits.
- Full local Python coverage: **763 passed, 2 skipped, 43 warnings** in 628 s;
  **30,456 / 36,611 statements (83.19%)**. The skips were an unstaged installer
  payload and optional PyTorch. Live Redis passed using a local Redis 7.4.9
  socket. The separate strength and Redis selection passed **149 tests** (136
  §6 tests and 13 shared-rate-limit tests).
- Packaged Workbench Chromium: **36 passed**, including six live API cases.
  A repeat production build reproduced all 12 packaged assets byte-for-byte.
  The hash-pinned dependency install, installed SDK wheel/resource check,
  dependency smoke imports, and strict installed SDK typing check passed.
  No remote CI run or independent post-fix verdict is claimed.

## Post-PR-33 audit follow-up — dependency review before edits

The user selected review and remediation of all seven findings in the latest
uploaded audit. Its exact commit is not stated; reproduction targets the clean
archive of `8992b659ede032407c1c16a6fd792048c9524bf6`. The uploaded report and
harnesses remain unchanged outside the working tree.

- F-01 overlaps **OF-17** (`5cf9a6c`): tighten eligibility and report the observed
  screening fraction and the absence of a calibrated false-positive rate. Keep
  listwise denominator, rank and fit guards. Move the existing gross-outlier
  test fixture into the newly eligible region, recording that dependency here.
- F-02 overlaps existing forecast disclosure and horizon-stratified rolling
  interval evidence. Relabel native recursive ML bands as heuristic; preserve
  their numeric values, point fits, and the separate rolling method.
- F-03 restricts the SDK omnibus decision to its predeclared Iman–Davenport
  approximation; retain the Friedman statistic and p-value as reference evidence.
- F-04 overlaps **OF-14** (`0b6eeb5`): name the legacy adapter explicitly and
  recommend the canonical one without silently changing old imports, defaults,
  empty-train behavior, return type, or buffer semantics.
- F-05 overlaps **OF-10** (`0b6eeb5`): use shared inference for observed chart
  targets while withholding catalog task inference for entirely missing targets.
- F-06 overlaps **OF-18** (`0b6eeb5`): retain the independently confirmed
  uncorrected Pearson Cramér’s V estimate and its separate p-value method.
  Add explicit variant provenance; no bias-correction change.
- F-07 touches presentation of previously verified AutoML partition/reference
  contracts, not ranking, selected model, or lift. Keep disclosed SDK-only
  methods SDK-only; no runtime surface is added.

### Post-PR-33 reproduction and verification

- Replayed unmodified `v01`–`v07` on the archived base and the patched source,
  using explicit `PYTHONPATH` for each run (verified import roots) and one BLAS
  thread. The [post-PR-33 traceability](ROUND2_TRACEABILITY_MATRIX.md#post-pr-33-independent-audit--f-01f-07)
  separates original evidence from supplemental counterexamples and remaining limits.
- New Python regressions are in `tests/unit/test_post_pr33_audit_contracts.py`;
  new Chromium cases are in `frontend/tests/browser/studio-charts.spec.ts`.
  On the archived SPA, the native-band and leaderboard-role checks reproduced as
  failures; the existing rolling-calibrated presentation was already correct.
  The packaged patched Workbench passed **39 Chromium tests**, including six live API
  cases. The original audit, harnesses, and auditor-owned verification files were
  not changed. No independent closure verdict is claimed.
- Local CPython 3.12.12 verification: **175 passed** across the §6 selection,
  new audit regressions, existing standards checks, and shared-rate-limit tests
  with live Redis 7.4.9. Ruff and mypy (201 files) passed with the CI Python
  extras. With the Redis extra present mypy reports the existing optional import
  assignment in `core/rate_limit.py`; CI runs mypy without that extra. No rate-limit
  source was changed.
- The production frontend build reproduced all 12 packaged files byte-for-byte.
  The hash-pinned runtime install, clean wheel and sdist build, installed SDK
  resource smoke (18 domains and `py.typed`), and strict installed SDK typing
  check passed. These are local results, not a remote CI or release verdict.
- The CI latency/memory script passed all six budgets with its unscaled defaults
  (25 timed iterations per path). Docker architecture jobs and the Python 3.13
  matrix leg were not executed in this sandbox.
- Full local Python test and coverage run, with live Redis enabled:
  **779 passed, 2 skipped, 43 warnings** in 629.32 s; **30,472 of 36,617
  statements (83.22%)** covered. The skips were the unstaged installer payload
  and missing optional PyTorch. No local failures remained.
