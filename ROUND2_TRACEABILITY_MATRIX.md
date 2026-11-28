# OmniForge — Round 2 traceability matrix

Active branch: `arena/84cfdc36-omniforge`. Parent reviewed by the auditor: `fd3647a56e0894ad09f5511d58394d2f8ebe29fc`.

| ID | Prior verdict / scope | Fix commit | Test run | Round 2 claim and reason |
|---|---|---|---|---|
| V-01 | Open candidate against `fd3647a`; explicitly selected for `report_unavailable` | `ac7ec24`, `1fb2d0d` (wording) | Original auditor method probe reproduced on `fd3647a`; original false-acceptance assertion rejected on fixed source. The policy module `tests/unit/test_nonparametric_reference_failure.py` and non-parametric e2e selection passed (7). Lineage and Studio 06 browser headings passed the new wording assertions; the wording follow-up full local Python suite: 704 passed, 3 skipped, 83% coverage; §6 selection: 136 passed. | **CLOSED — remediation claim only.** Failed/nonfinite reference evidence cannot become `0.0`, positive lift, acceptance, or auto-promotion. Successfully scored candidates remain inspectable, and a successfully measured zero remains valid. The [independent auditor closed V-01 at `006038f`](https://github.com/hajianamir/OmniForge/pull/31#issuecomment-7235746160) after 10/10 PR checks passed. Not a release verdict. |

| OF-09 (Studio 04 Shapiro only) | Round 1 partial; explicitly selected narrow disclosure residual | `dd4901e` | Seven API regressions red/green at the 5,000 / 5,001 / 5,100 row boundaries and on reporting, chart and export surfaces; new browser assertion verifies custom result and battery display; §6 selection: 136 passed. | **Disclosure addressed; OF-09 remains PARTIAL.** The route, per-column battery, exported result and Q–Q chart show which observations were tested without changing evaluable test samples. Other caps and any independent verdict remain outside this claim. |

## Follow-up authorization: remaining Phase 2–4 worklist

| Item | Implementation or retained policy | Evidence and limits |
|---|---|---|
| OF-04 / OF-06 | Use prediction-derived weighted F1 / RMSE for selected-model comparison. Treat matched cohorts and feature scope as prerequisites. Show backend-specific rankings separately; distinguish raw input, fitted dimension, and retained support. | Optional-adapter interface fixtures cover all five external IDs with mismatched search scores; fitted encoded counts checked on the warm-start backend. Real optional runtimes and hidden internals remain **unverified**, not equivalent. |
| OF-09 follow-up | Expose Studio 04 EDA, non-Shapiro individual/battery, and plot selection or display limitations; Studio 02 advanced mining reports beam, complete-case, Hopkins and dCor predictor limits. | Boundary tests for 5,100 observations / 15 groups, 8 dCor predictors, and chart display versus fit scope. Shapiro checks remain separate; disclosure is not a claim of exhaustive chart/statistical validity. |
| OF-13a | Separate generated/tested rule and subgroup families; eligible membership precedes display truncation. | Independent small-fixture hypergeometric and BH calculations check membership, ties, filtered/empty cases, and display limits. No unconditional FDR guarantee after data-dependent screening. |
| OF-14 CPCV adapter evidence | Keep the documented distinct adapter defaults, buffer policies, and list/iterator and empty-fold contracts. | Regression exercises contractual differences; existing matched-configuration and label-overlap fixtures retained. No general interchangeability claim. |
| OF-17 / OF-07 / OF-15 | Retain asymptotic screening, positional/explicit-id rules, and SDK-only governance as intentional limits. | No new finite-sample calibration, durable ID writer, or operational governance endpoint is claimed. |
| OF-21 / OF-28 | Cover the locked, capped, unstratifiable holdout with production API and baseline models. | Fallback method and available/retained count checks. Real PyTorch execution is **unverified** in core tests. |
| OF-31 | Benford policy: finite nonzero magnitudes, exact leading digit of the stored value's decimal spelling, no epsilon rounding. | Production cleaning audit fixture with values below/at/above a threshold, zeros, missing and non-finite values; no invariance for values changed by floating-point rounding. |

All claims above are engineering evidence. This follow-up does not reopen historical `VERIFIED-CLOSED` findings; it supplies evidence for the auditor's specific residuals and records the retained contracts.

The earlier in-progress matrices supplied evidence states, not qualifying final verdicts. The separate 667-pass run reported on the auditor's branch is not used as evidence for this implementation.

[Round 2 remediation log and remaining verification limits](ROUND2_REMEDIATION_LOG.md)

## Independent main audit follow-up — `4385647`

Remediation branch: `arena/92c52eb8-omniforge`. These IDs are not historical
finding IDs. The original report and probes were not changed.

| Finding | Reproduced on unchanged `4385647` | Remediation evidence |
|---|---|---|
| MAIN-01 | All six original Studio 03 option probes returned 500. Archived SPA sent the short option values. | Holm now returns 200 and matches statsmodels on the evaluable ledger. Current browser values for the mode and all four normality methods return 200 in the API tests; obsolete short values return structured 422. A live Chromium run exercised Holm, non-parametric policy, and Anderson–Darling. |
| MAIN-02 | Three-arm route returned 200, ATE `8.216801`, CI `[7.202181, 9.231421]`; SDK returned `8.225828`. | Same route fixture returns 400 without ATE/CI; exact SDK fixture raises `CausalInferenceError`. The 0/1 subset's full SDK result is unchanged (ATE `1.973929`, 100 per arm). Regression tests cover all six binary ATE methods, single/filtered arms, nonfinite and invalid levels, and continuous DML. |
| MAIN-03 | Audit linear-null vector reproduced `+0.0210 R²` and the significance claim; digest `64458e3791e50fd33ac361ac02d0751b030017d06be30b6a6283acae07ceed60`. | Same lift and digest, but verdict describes a `0.015` heuristic, not an inferential test. Below-threshold replay also uses descriptive wording. Current and restored legacy views are covered; V-01 reference-failure contracts retained. |
| MAIN-04 | API returned the map `x: 65.2236`, `x2: 65.0584`; archived SPA did not render VIF. | API map retained. Chromium renders real values, shows the recorded threshold, does not cap values at 100, and does not chart nulls as zero. |
| MAIN-05 | API gave `0.0` for the 45-row fixture. | Same fixture gives the unrounded SciPy p-value `3.4571369789354314e-11`; the conclusion and digit-selection rule are unchanged. |
| MAIN-06 | The synthetic exception canary appeared in both original outputs. | Original structlog and stdlib probes no longer emit the canary. JSON/console formatter tests cover chained exceptions and file/stream output. No active application disclosure was established. |
| MAIN-07 | User guide listed `/s03/eda`; original POST returned 405. | User guide and architecture list `GET /s03/catalog` and `POST /s03/run`; an OpenAPI check covers the Studio 03 entries in both guides and the API reference. |

All seven are remediated in the tested cases. This is engineering evidence for
independent review, not an independent closure or release verdict.
