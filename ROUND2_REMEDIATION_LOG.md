# OmniForge — Round 2 remediation log

## Current summary (supersedes the initial intake below)

**V-01 only:** implemented the selected `report_unavailable` policy on
`arena/84cfdc36-omniforge` in `ac7ec24`. The original counterexample reproduced on the archived
parent. The corrected regression, full local Python coverage suite (704 passed, 3 skipped),
§6 selection, and local browser/Redis checks passed. There are no other reopened or still-partial
Round 2 IDs. This is a remediation claim, not an independent finding verdict or full CI parity.
The initial empty-scope record below is retained for provenance.

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
