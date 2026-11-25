"""Phase 1 Critical Modifications verification suite (OmniForge v3.0.0).

Covers:
1. SemVer 3.0.0 metadata consistency across Python SDK, pyproject.toml, and frontend package.json.
2. RidgeLinearBaseline (L2 Hoerl-Kennard / Tikhonov) & LassoLinearBaseline (L1 Tibshirani) mathematical integrity & symbolic equations.
3. ValidationRule.to_expectation() full shorthand coverage, regex/parameter validation, boolean-vs-numeric dtype separation, and datetime pair comparisons (ISO/IEC 25012 & Great Expectations).
4. Honest W3C PROV-DM / NIST AI RMF /s01/clean steps_applied & forensic deleted_records stage/reason provenance (including Split Lock).
5. DataCleaner.fit() / transform() 11-stage canonical symmetry and post-outlier imputation uncontamination (Kapoor & Narayanan 2023).
6. MITRE CWE-22 path containment in LocalFeatureStore & DatasetVersionManager, plus pure (epsilon, 0)-DP Gaussian Copula synthesis (Dwork & Roth 2014; NIST SP 800-226).
"""

from __future__ import annotations

import io
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import Lasso, LassoCV, Ridge

import omniforge
from omniforge.api.auth import get_api_key
from omniforge.api.main import app
from omniforge.core.exceptions import DataError, DataValidationError, FeatureEngineeringError
from omniforge.data.cleaning import CleaningConfig, DataCleaner
from omniforge.data.feature_store import FeatureEntity, LocalFeatureStore
from omniforge.data.synthetic import SyntheticDataGenerator
from omniforge.data.validation import DataValidator, ValidationRule
from omniforge.data.versioning import DatasetVersionManager
from omniforge.evaluation.equations import ModelEquationExtractor
from omniforge.models.baselines import (
    BaselineBenchmarkSuite,
    LassoLinearBaseline,
    RidgeLinearBaseline,
)
from omniforge.models.registry import default_model_catalog
from omniforge.version import get_version_info

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version_3_0_0_consistency() -> None:
    """Verify canonical v4.0.0 semantic version across SDK, pyproject.toml, and SPA manifest."""
    assert omniforge.__version__ == "4.0.0"
    assert get_version_info()["version"] == "4.0.0"

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == "4.0.0"

    pkg_json = json.loads((REPO_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    assert pkg_json["version"] == "4.0.0"

    assert omniforge.RidgeLinearBaseline is RidgeLinearBaseline
    assert omniforge.LassoLinearBaseline is LassoLinearBaseline


def test_ridge_and_lasso_baselines_mathematical_integrity() -> None:
    """Verify RidgeLinearBaseline solves exact L2 normal equations and LassoLinearBaseline enforces L1 sparsity."""
    rng = np.random.default_rng(42)
    n = 120
    z = rng.normal(0.0, 1.0, size=n)
    # Collinear signal features x1, x2 plus pure noise features x3, x4
    X = pd.DataFrame(
        {
            "x1": z + rng.normal(0.0, 0.05, size=n),
            "x2": z + rng.normal(0.0, 0.05, size=n),
            "noise_1": rng.normal(0.0, 1.0, size=n),
            "noise_2": rng.normal(0.0, 1.0, size=n),
        }
    )
    y = 3.0 * z + 1.5 + rng.normal(0.0, 0.1, size=n)

    ridge_low = RidgeLinearBaseline(alpha=0.01, track_carbon=False).fit(X, y)
    ridge_mid = RidgeLinearBaseline(alpha=1.0, track_carbon=False).fit(X, y)
    ridge_high = RidgeLinearBaseline(alpha=100.0, track_carbon=False).fit(X, y)

    assert isinstance(ridge_mid._model, Ridge)
    assert not isinstance(ridge_mid._model, LassoCV)

    # Exact closed-form Hoerl-Kennard / Tikhonov check on standardized predictors:
    # (X_tilde^T X_tilde + alpha * I) w = X_tilde^T (y - mean(y))
    X_tilde = (X.to_numpy() - ridge_mid._scaler.mean_) / ridge_mid._scaler.scale_
    y_centered = y - np.mean(y)
    w_exact = np.linalg.solve(
        X_tilde.T @ X_tilde + 1.0 * np.eye(X.shape[1]),
        X_tilde.T @ y_centered,
    )
    np.testing.assert_allclose(ridge_mid._model.coef_, w_exact, atol=1e-8)

    # Strict L2 norm shrinkage as alpha increases
    norm_low = float(np.linalg.norm(ridge_low._model.coef_, ord=2))
    norm_mid = float(np.linalg.norm(ridge_mid._model.coef_, ord=2))
    norm_high = float(np.linalg.norm(ridge_high._model.coef_, ord=2))
    assert norm_high < norm_mid < norm_low
    # Ridge shares weight across collinear predictors x1 and x2 (both non-zero)
    assert abs(ridge_mid._model.coef_[0]) > 0.5 and abs(ridge_mid._model.coef_[1]) > 0.5

    # Invalid alpha values must be rejected explicitly
    for bad_alpha in (-0.5, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="alpha"):
            RidgeLinearBaseline(alpha=bad_alpha)  # type: ignore[arg-type]

    # LassoLinearBaseline (L1) zeroes out pure noise predictors and works via catalog + suite + equation extractor
    lasso_cv = LassoLinearBaseline(cv=4, track_carbon=False).fit(X, y)
    assert isinstance(lasso_cv._model, LassoCV)
    lasso_fixed = LassoLinearBaseline(alpha=0.25, track_carbon=False).fit(X, y)
    assert isinstance(lasso_fixed._model, Lasso)
    assert abs(lasso_fixed._model.coef_[2]) == 0.0 and abs(lasso_fixed._model.coef_[3]) == 0.0

    # Tiny sample fallback (n=2 < cv=3) must not crash
    tiny_lasso = LassoLinearBaseline(cv=3, track_carbon=False).fit(X.iloc[:2], y[:2])
    assert len(tiny_lasso.predict(X.iloc[:2])) == 2

    assert isinstance(default_model_catalog.create("baseline_lasso", track_carbon=False), LassoLinearBaseline)
    suite_scores, suite_models = BaselineBenchmarkSuite.run_regression_detailed(
        X.iloc[:90], y[:90], X.iloc[90:], y[90:]
    )
    assert "Baseline_RidgeLinear" in suite_scores
    assert "Baseline_LassoLinear" in suite_scores
    assert isinstance(suite_models["Baseline_RidgeLinear"]._model, Ridge)

    # Symbolic equation unscaling exactness for both RidgeLinearBaseline and LassoLinearBaseline
    for bl_model in (ridge_mid, lasso_cv):
        eq = ModelEquationExtractor.extract(bl_model, feature_names=list(X.columns), target_name="y")
        assert eq.intercept is not None
        coef_map = {row["Feature"]: row["Raw Coefficient (beta)"] for row in eq.coefficients_table}
        manual_pred = eq.intercept + sum(coef_map[col] * X[col].to_numpy() for col in X.columns)
        np.testing.assert_allclose(manual_pred, bl_model.predict(X), atol=1e-4)


def test_validation_rule_all_shorthands_and_type_semantics() -> None:
    """Verify every ValidationRule shorthand converts and evaluates accurately."""
    ref_dist = np.linspace(0.0, 10.0, 50)
    df = pd.DataFrame(
        {
            "order_id": ["ORD-0001", "ORD-0002", "ORD-0003", "ORD-0004"],
            "amount": [10.0, 25.0, 40.0, 55.0],
            "status": ["paid", "shipped", "paid", "delivered"],
            "seq": [1, 2, 3, 4],
            "score": [1.5, 3.5, 6.0, 8.5],
            "flag": [True, False, True, True],
            "created_at": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "shipped_at": ["2026-01-02", "2026-01-02", "2026-01-05", "2026-01-06"],
        }
    )

    rules = [
        ValidationRule(name="rows_ok", rule_type="row_count", params={"min": 2, "max": 10}),
        ValidationRule(name="cols_ok", rule_type="columns_match_set", params={"column_set": list(df.columns)}),
        ValidationRule(name="id_exists", rule_type="column_exists", column="order_id"),
        ValidationRule(name="id_not_null", rule_type="not_null", column="order_id"),
        ValidationRule(name="id_null_ratio", rule_type="null_ratio", column="order_id", params={"max_null_ratio": 0.0}),
        ValidationRule(name="id_unique", rule_type="unique", column="order_id"),
        ValidationRule(name="id_regex", rule_type="regex", column="order_id", params={"pattern": r"^ORD-\d{4}$"}),
        ValidationRule(name="amt_range", rule_type="range", column="amount", params={"min": 0.0, "max": 100.0}),
        ValidationRule(name="amt_dtype", rule_type="dtype", column="amount", params={"dtype": "numeric"}),
        ValidationRule(name="flag_bool", rule_type="dtype", column="flag", params={"dtype": "boolean"}),
        ValidationRule(name="status_set", rule_type="in_set", column="status", params={"values": ["paid", "shipped", "delivered"]}),
        ValidationRule(name="seq_inc", rule_type="increasing", column="seq", params={"strict": True}),
        ValidationRule(name="score_ks", rule_type="ks_test", column="score", params={"reference_values": ref_dist, "alpha": 0.01}),
        ValidationRule(name="ship_after_create", rule_type="pair_greater", column="shipped_at", params={"column_b": "created_at", "or_equal": True}),
    ]
    report = DataValidator(rules, suite_name="phase1_full_shorthand").validate(df)
    assert report.passed is True, [r.model_dump() for r in report.results if not r.passed]
    assert report.total_expectations == len(rules)
    assert report.failed_count == 0

    # Boolean column must NOT pass dtype='numeric'
    bool_as_num = DataValidator(
        [ValidationRule(name="flag_not_numeric", rule_type="dtype", column="flag", params={"dtype": "numeric"})]
    ).validate(df)
    assert bool_as_num.passed is False

    # Strict increasing must fail on ties [1, 2, 2, 3], while non-strict passes
    df_ties = pd.DataFrame({"t": [1, 2, 2, 3]})
    assert DataValidator([ValidationRule(name="non_strict", rule_type="increasing", column="t")]).validate(df_ties).passed is True
    assert DataValidator([ValidationRule(name="strict", rule_type="increasing", column="t", params={"strict": True})]).validate(df_ties).passed is False

    # Date pair violation must be caught accurately (not coerced to NaN and ignored!)
    df_bad_dates = pd.DataFrame({"created_at": ["2026-05-10"], "shipped_at": ["2026-05-01"]})
    bad_pair_rep = DataValidator(
        [ValidationRule(name="date_order", rule_type="pair_greater", column="shipped_at", params={"column_b": "created_at"})]
    ).validate(df_bad_dates)
    assert bad_pair_rep.passed is False
    assert bad_pair_rep.results[0].unexpected_count == 1

    # Invalid regex and unknown rule_type must raise DataValidationError at conversion time
    with pytest.raises(DataValidationError, match="invalid regular expression"):
        ValidationRule(name="bad_rx", rule_type="regex", column="order_id", params={"regex": "[unclosed"}).to_expectation()
    with pytest.raises(DataValidationError, match="Unsupported validation rule_type"):
        ValidationRule(name="typo", rule_type="not_a_real_rule", column="order_id").to_expectation()


def test_datacleaner_canonical_stage_symmetry_and_outlier_uncontaminated_imputation() -> None:
    """Verify fit() and transform() run identical stage ordering and fit imputers post-outlier treatment."""
    df = pd.DataFrame(
        {
            "metric": [10.0, 11.0, 12.0, 10.5, 11.5, 11.0, 1_000_000.0, np.nan],
            "sentinel_text": ["<span>N/A</span>", " ok ", " null ", "ok", " ? ", "ok", "ok", "ok"],
            "mostly_empty": [np.nan, np.nan, np.nan, np.nan, np.nan, "diff_only_here", np.nan, np.nan],
        }
    )
    cfg = CleaningConfig(
        clean_strings=True,
        standardize_missing_tokens=True,
        infer_types=False,
        drop_high_missing_cols=True,
        missing_col_threshold=0.80,
        drop_exact_duplicates=True,
        outlier_method="iqr",
        outlier_action="remove",
        outlier_threshold=1.5,
        numeric_imputation="mean",
        categorical_imputation="most_frequent",
    )
    cleaner = DataCleaner(cfg)
    cleaned_ft, rep_ft = cleaner.fit_transform(df)

    cleaner_sep = DataCleaner(cfg).fit(df)
    cleaned_sep, rep_sep = cleaner_sep.transform(df)
    pd.testing.assert_frame_equal(cleaned_ft, cleaned_sep)
    assert rep_ft.steps_executed == rep_sep.steps_executed

    # 1. clean_strings ran before standardize_missing_tokens, turning "<span>N/A</span>", " null ", " ? " into NaN
    assert rep_ft.steps_executed.index("clean_strings") < rep_ft.steps_executed.index("standardize_missing_tokens")
    # 2. mostly_empty (87.5% NaN) was dropped BEFORE exact deduplication, so row 0 (10.0, ok) & row 5 (10.0, diff_only_here -> dropped) deduplicated!
    assert "mostly_empty" not in cleaned_ft.columns
    assert rep_ft.exact_duplicates_removed == 1
    # 3. Extreme outlier 1,000,000.0 was removed BEFORE fitting mean imputer, so NaN was imputed near ~11.0, NOT >100,000!
    assert cleaned_ft["metric"].max() < 20.0
    imputed_val = float(cleaner._state.numeric_fill_values["metric"])
    assert 10.0 <= imputed_val <= 12.0


def test_workbench_s01_clean_honest_steps_and_forensic_deleted_records() -> None:
    """Verify /api/workbench/s01/clean reports only executed steps and true deletion stage/reason."""
    client = TestClient(app, headers={"X-API-Key": get_api_key()})
    hdr = {"X-Session-ID": "phase1-s01-honest-audit"}

    # Build a 20-row dataset with 1 exact duplicate and 1 extreme IQR outlier
    rows = [{"Sensor Val": float(10 + (i % 5)), "Category": "A" if i % 2 == 0 else "B", "Target": float(i % 2)} for i in range(18)]
    rows.append(dict(rows[0]))  # exact duplicate at index 18
    rows.append({"Sensor Val": 99999.0, "Category": "A", "Target": 1.0})  # extreme outlier at index 19
    df = pd.DataFrame(rows)

    csv_bytes = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    r_up = client.post("/api/workbench/s01/upload", headers=hdr, files={"file": ("audit.csv", csv_bytes, "text/csv")})
    assert r_up.status_code == 200, r_up.text
    assert client.post("/api/workbench/targets", headers=hdr, json={"targets": ["Target"]}).status_code == 200

    # Case 1: All cleaning operations disabled -> steps_applied must be empty (zero fabricated claims!)
    r_noop = client.post(
        "/api/workbench/s01/clean",
        headers=hdr,
        json={
            "mode": "custom",
            "clean_strings": False,
            "standardize_missing_tokens": False,
            "infer_types": False,
            "drop_high_missing_cols": False,
            "drop_duplicates": False,
            "handle_outliers": "none",
            "outlier_method": "none",
            "impute_numeric": "none",
            "impute_categorical": "none",
            "drop_constant_columns": False,
            "standardize_column_names": False,
            "preview_only": True,
        },
    )
    assert r_noop.status_code == 200, r_noop.text
    noop_rep = r_noop.json()["cleaning_report"]
    assert noop_rep["steps_applied"] == []
    assert noop_rep["steps_executed"] == []

    # Case 2: Enable Split Lock + Deduplication + Outlier Removal + Column Standardization
    assert client.post("/api/workbench/split-lock", headers=hdr, json={"strategy": "random", "test_size": 0.25, "cv_folds": 3}).status_code == 200
    r_clean = client.post(
        "/api/workbench/s01/clean",
        headers=hdr,
        json={
            "mode": "custom",
            "clean_strings": True,
            "standardize_missing_tokens": True,
            "infer_types": False,
            "drop_high_missing_cols": False,
            "drop_duplicates": True,
            "handle_outliers": "remove",
            "outlier_method": "iqr",
            "outlier_threshold": 1.5,
            "impute_numeric": "median",
            "impute_categorical": "most_frequent",
            "drop_constant_columns": False,
            "standardize_column_names": True,
            "preview_only": False,
        },
    )
    assert r_clean.status_code == 200, r_clean.text
    rep = r_clean.json()["cleaning_report"]
    steps_joined = " | ".join(rep["steps_applied"])
    # Never fabricate row-missingness threshold removal
    assert "Remove rows exceeding missing-value threshold" not in steps_joined
    # Standardize column names happened at the END, not step 1
    assert rep["steps_executed"][-1] == "standardize_column_names"
    assert "Standardize column names" in rep["steps_applied"][-1]

    # Forensic deleted_records must carry real stage & reason (never fallback "Policy threshold exceeded")
    del_recs = rep["deleted_records"]
    assert len(del_recs) >= 2
    stages = {d["stage"] for d in del_recs}
    reasons = " | ".join(d["reason"] for d in del_recs)
    assert "Exact Deduplication" in stages
    assert "Outlier Removal" in stages
    assert "Policy threshold exceeded" not in reasons
    assert "Outlier Removed" in reasons


def test_cwe22_path_traversal_blocked_in_feature_store_and_versioning(tmp_path: Path) -> None:
    """Verify CWE-22 path traversal is blocked on both write and tampered-manifest read paths."""
    fs = LocalFeatureStore(repo_path=tmp_path / "fs_repo")
    df = pd.DataFrame({"entity_id": ["e1", "e2"], "val": [1.0, 2.0]})

    for payload in ("../escape", "../../etc/passwd", "sub/dir", ".hidden", "bad..name"):
        with pytest.raises(FeatureEngineeringError):
            fs.register_feature_view(payload, df, join_keys=["entity_id"])
        with pytest.raises(FeatureEngineeringError):
            fs.register_entity(FeatureEntity(name=payload, join_key="entity_id"))

    spec = fs.register_feature_view("safe_view", df, join_keys=["entity_id"])
    assert Path(spec.offline_path).resolve().is_relative_to((tmp_path / "fs_repo" / "offline").resolve())

    # Tamper with registry.json offline_path to point outside the feature store
    outside_parquet = tmp_path / "secret_outside.parquet"
    df.to_parquet(outside_parquet, index=False)
    reg_path = tmp_path / "fs_repo" / "registry.json"
    reg_data = json.loads(reg_path.read_text(encoding="utf-8"))
    reg_data["feature_views"]["safe_view"]["offline_path"] = str(outside_parquet)
    reg_path.write_text(json.dumps(reg_data), encoding="utf-8")

    with pytest.raises(FeatureEngineeringError, match="CWE-22"):
        fs.materialize_to_online("safe_view")
    with pytest.raises(FeatureEngineeringError, match="CWE-22"):
        fs.get_historical_features(pd.DataFrame({"entity_id": ["e1"]}), ["safe_view:val"])

    # DatasetVersionManager write & tampered manifest read confinement
    dvm = DatasetVersionManager(store_dir=tmp_path / "ver_repo")
    for payload in ("../escape", "../../evil", "nested/ds", "bad..tag"):
        with pytest.raises(DataError):
            dvm.commit(df, payload)
    with pytest.raises(DataError):
        dvm.commit(df, "clean_ds", tag="../bad_tag")

    ver = dvm.commit(df, "clean_ds", tag="v1")
    loaded_df, _ = dvm.checkout("clean_ds", "v1")
    assert len(loaded_df) == 2

    manifest_path = tmp_path / "ver_repo" / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["datasets"]["clean_ds"][-1]["storage_path"] = str(outside_parquet)
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(DataError, match="CWE-22"):
        dvm.checkout("clean_ds", ver.version_id)


def test_differential_privacy_copula_validation_bounds_and_composition() -> None:
    """Verify pure (epsilon, 0)-DP Gaussian Copula rejects invalid epsilon and privatizes all modalities."""
    rng = np.random.default_rng(123)
    n = 200
    x1 = rng.uniform(10.0, 90.0, size=n)
    df = pd.DataFrame(
        {
            "num_cont": x1,
            "num_corr_int": np.rint(2.0 * x1 + rng.normal(0.0, 5.0, size=n)).astype(int),
            "category": ["rare"] * 5 + ["common"] * (n - 5),
            "flag": [True] * 150 + [False] * 50,
            "event_ts": pd.date_range("2025-01-01", periods=n, freq="h"),
        }
    )

    gen = SyntheticDataGenerator(random_seed=99)
    for invalid_eps in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="dp_epsilon"):
            gen.fit_generate_copula(df, n_samples=50, dp_epsilon=invalid_eps)  # type: ignore[arg-type]

    bounds = {
        "num_cont": (0.0, 100.0),
        "num_corr_int": (0.0, 220.0),
        "event_ts": (
            float(pd.Timestamp("2025-01-01").value),
            float(pd.Timestamp("2025-01-15").value),
        ),
    }
    cats = {"category": ["common", "rare", "unobserved_cat"]}

    syn_high_eps = gen.fit_generate_copula(
        df,
        n_samples=300,
        dp_epsilon=25.0,
        dp_bounds=bounds,
        dp_categories=cats,
    )
    assert list(syn_high_eps.columns) == list(df.columns)
    assert len(syn_high_eps) == 300
    assert syn_high_eps["num_cont"].between(0.0, 100.0).all()
    assert syn_high_eps["num_corr_int"].between(0, 220).all()
    assert pd.api.types.is_integer_dtype(syn_high_eps["num_corr_int"])
    assert pd.api.types.is_datetime64_any_dtype(syn_high_eps["event_ts"])
    assert set(syn_high_eps["category"].unique()).issubset({"common", "rare", "unobserved_cat"})
    # High privacy budget preserves strong positive correlation between num_cont and num_corr_int
    assert float(syn_high_eps["num_cont"].corr(syn_high_eps["num_corr_int"])) > 0.65

    # Under a tight budget, fresh noise can introduce unobserved public categories.
    syn_tight_eps = SyntheticDataGenerator(random_seed=7).fit_generate_copula(
        df,
        n_samples=400,
        dp_epsilon=0.05,
        dp_bounds=bounds,
        dp_categories=cats,
    )
    assert set(syn_tight_eps["category"].unique()).issubset(set(cats["category"]))
    assert syn_tight_eps.attrs["privacy_accounting"]["epsilon"] == 0.05
    assert len(gen.fit_generate_copula(df, n_samples=0, dp_epsilon=1.0, dp_bounds=bounds, dp_categories=cats)) == 0
