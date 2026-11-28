# OmniForge — Round 2 traceability matrix

Active branch: `arena/84cfdc36-omniforge`. Parent reviewed by the auditor: `fd3647a56e0894ad09f5511d58394d2f8ebe29fc`.

| ID | Prior verdict / scope | Fix commit | Test run | Round 2 claim and reason |
|---|---|---|---|---|
| V-01 | Open candidate against `fd3647a`; explicitly selected for `report_unavailable` | `ac7ec24`, `1fb2d0d` (wording) | Original auditor method probe reproduced on `fd3647a`; original false-acceptance assertion rejected on fixed source. The policy module `tests/unit/test_nonparametric_reference_failure.py` and non-parametric e2e selection passed (7). Lineage and Studio 06 browser headings passed the new wording assertions; the wording follow-up full local Python suite: 704 passed, 3 skipped, 83% coverage; §6 selection: 136 passed. | **CLOSED — remediation claim only.** Failed/nonfinite reference evidence cannot become `0.0`, positive lift, acceptance, or auto-promotion. Successfully scored candidates remain inspectable, and a successfully measured zero remains valid. The [independent auditor closed V-01 at `006038f`](https://github.com/hajianamir/OmniForge/pull/31#issuecomment-7235746160) after 10/10 PR checks passed. Not a release verdict. |

| OF-09 (Studio 04 Shapiro only) | Round 1 partial; explicitly selected narrow disclosure residual | `dd4901e` | Seven API regressions red/green at the 5,000 / 5,001 / 5,100 row boundaries and on reporting, chart and export surfaces; new browser assertion verifies custom result and battery display; §6 selection: 136 passed. | **Disclosure addressed; OF-09 remains PARTIAL.** The route, per-column battery, exported result and Q–Q chart show which observations were tested without changing evaluable test samples. Other caps and any independent verdict remain outside this claim. |

No other rows were reopened or selected for upgrades. The earlier in-progress matrices supplied evidence states, not qualifying final verdicts. The separate 667-pass run reported on the auditor's branch is not used as evidence for this implementation.

[Round 2 remediation log and remaining verification limits](ROUND2_REMEDIATION_LOG.md)
