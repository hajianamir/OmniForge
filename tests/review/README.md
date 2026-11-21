# OmniForge Review Test Suites — Baselines & Preparation/Cleaning

**Purpose:** independent verification suites for the read-only OmniForge audit
(`REVIEW_REPORT.md` at the repo root). They check the two most safety-critical
layers of the platform:

- `test_baselines_and_algorithms.py` (Round 1) — classifier/regressor/forecasting
  baselines and benchmark suites vs sklearn + manual recomputation.
- `test_cleaning_preparation.py` (Round 2) — `DataCleaner`, splits, validation,
  quality profiling, audio/audit contracts.

## How to run

```bash
# repo requires Python >= 3.12
python -m pytest tests/review -q
```

The tests import the `omniforge` package from the repo root; run from the repo
root or set `PYTHONPATH=src`. They are self-contained (no fixtures from
`tests/conftest.py` or external files).

## Status & caveats

- Round 1 suite: **10/10 passing**. Round 2 suite: **21/21 passing**
  (verified under the audit environment, Python 3.11.2 + two minimal
  behavior-preserving syntax-compatibility patches in the review copy; the
  repo itself requires Python >= 3.12 and ships no such patches).
- These tests document **current** behavior, including two known findings:
  `RidgeLinearBaseline` is implemented as `LassoCV` (mislabeled) and the
  `steps_applied` list shown in the cleaning response is **not** derived from
  the actual `DataCleaner` pipeline.
- The full audit report (baselines, cleaning/preparation, integration and
  security findings) is in `REVIEW_REPORT.md` at the repo root.
- **Not modified during this audit:** no source files, no existing tests — the
  review copy used only for the test environment carried two documented
  syntax-compatibility patches (`core/performance.py` PEP 695 generic,
  `evaluation/equations.py` nested f-string backslash).
