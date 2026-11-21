# Changelog

## [2.0.0] - 2026-11-20
### Added
- Modularized Workbench API into `src/omniforge/api/studios/` (`s00_ingestion.py` through `s14_supervisory.py`).
- Added Studio 00 (`Data Ingestion, Intake Audit, Split Lock & Data Relationships`) and separated Studio 01 (`Data Preparation & Cleaning`) with step-by-step `steps_applied` transparency.
- Added `CombinatorialPurgedKFold` with embargo, `knn_shapley_valuation`, `venn_abers_calibrate`, `nested_cv_evaluate`, `CausalInferenceEngine.export_dowhy_gml`, `probe_foundation_backends`, `fast_read_tabular` (Polars/DuckDB/pandas), `fuzzy_deduplicate_lsh_blocking`, `BackgroundJobQueue`, and `validate_production_promotion_gate`.
- Added `.github/workflows/ci.yml` (Python 3.12–3.13 matrix + coverage) and `docs/MIGRATION_V3.md`.

### Fixed
- Corrected `distance_correlation` denominator formula (`dCor(x, x) == 1.0`).
- Implemented `percentile` and `elliptic` outlier methods in `DataCleaner`.
- Preserved `NaT -> np.nan` and timezone-aware conversion in `_prepare_xy`.
- Returned `NaN` / `None` instead of misleading `1.0` / `100.0` placeholders in `normality_test`, `compute_vif`, and `DataQualityProfiler.timeliness`.

### Extras Compatibility Matrix
- `core`: Python 3.12–3.13
- `api, ui, reporting, data, forecasting`: Python 3.12–3.13
- `experimental` (`catboost`, `autogluon`, `chronos`, `tabpfn`): Python 3.12 recommended
