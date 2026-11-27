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
