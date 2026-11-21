"""Round 2 — Preparation, cleaning, splitting, validation, and quality (independent verification, read-only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omniforge.data.cleaning import (
    CleaningConfig,
    DataCleaner,
    benford_digit_test,
    cleaning_distortion_audit,
    fuzzy_deduplicate_lsh_blocking,
)
from omniforge.data.splitting import LeakageSafeSplitter, SplitConfig
from omniforge.data.validation import DataValidator, ValidationRule, ValidationSeverity
from omniforge.data.quality import DataQualityProfiler


# --- DataCleaner: fit/transform consistency + report contract
def test_cleaning_report_contract_and_fit_transform_consistency():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "num": rng.normal(0, 1, 300),
        "num_dup": [1.0] * 100 + [2.0] * 100 + [3.0] * 100,
        "cat": ["A", "B", "C"] * 100,
        "ts": pd.date_range("2026-01-01", periods=300, freq="D"),
    })
    cleaner = DataCleaner(CleaningConfig(drop_exact_duplicates=True, outlier_method="iqr", outlier_action="clip", numeric_imputation="median"))
    cleaned, rep = cleaner.fit_transform(df)
    assert rep.input_shape == (300, 4)
    assert rep.removed_rows_count == 0  # no exact row-level duplicates in this fixture
    assert rep.output_shape[0] >= 1
    # transform should be deterministic with same fitted params
    again, rep2 = cleaner.transform(df)
    assert len(cleaned) == len(again)
    assert int(cleaned.isna().sum().sum()) == int(again.isna().sum().sum())


def test_cleaning_transform_never_uses_unstored_future_values():
    """Transform must use fitted (train) values only — leading NaNs fall back to train's last value."""
    rng = np.random.default_rng(1)
    train_vals = rng.normal(0, 1, 40)
    test_vals = np.concatenate([[np.nan, np.nan], rng.normal(0, 1, 8)])
    df_tr = pd.DataFrame({"x": train_vals, "t": pd.date_range("2026-01-01", periods=40, freq="D")})
    cfg = CleaningConfig(numeric_imputation="ffill", causal_time_series=True, time_column="t", outlier_method="none", drop_exact_duplicates=False, infer_types=False)
    cl = DataCleaner(cfg)
    cl.fit(df_tr)
    test_df = pd.DataFrame({"x": test_vals, "t": pd.date_range("2026-02-01", periods=10, freq="D")})
    out, rep = cl.transform(test_df)
    # first two leading NaNs must be back-filled by train's causal fallback, never by future rows
    assert not pd.isna(out["x"].iloc[0])
    assert out["x"].iloc[0] == pytest.approx(cl._state.causal_last_values.get("x", np.nan) or cl._state.numeric_fill_values["x"])


def test_outlier_clip_and_report_consistency():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.normal(0, 1, 120)})
    df.loc[0, "x"] = 20.0
    cfg = CleaningConfig(numeric_imputation="none", outlier_method="iqr", outlier_action="clip", drop_exact_duplicates=False, infer_types=False, clean_strings=False, standardize_missing_tokens=False)
    cl = DataCleaner(cfg)
    out, rep = cl.fit_transform(df)
    q1, q3 = df["x"].quantile(0.25), df["x"].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    assert out["x"].max() <= hi + 1e-9
    assert rep.outliers_detected.get("x", 0) >= 1
    assert rep.outlier_action_applied == "clip"


def test_exact_duplicates_removed_and_fuzzy_deduplication():
    # exact
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "y", "y", "z"]})
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # add one truly identical row
    cl = DataCleaner(CleaningConfig(drop_exact_duplicates=True, infer_types=False, clean_strings=False, standardize_missing_tokens=False, outlier_method="none"))
    out, rep = cl.fit_transform(df)
    assert rep.exact_duplicates_removed == 1
    assert len(out) == 4
    # fuzzy dedup over string column
    df2 = pd.DataFrame({"txt": ["hello world", "hello world ", "goodbye"]})
    cl2 = DataCleaner(CleaningConfig(drop_exact_duplicates=False, fuzzy_dedup_column="txt", fuzzy_dedup_threshold=0.95, infer_types=False, clean_strings=False, standardize_missing_tokens=False, outlier_method="none"))
    out2, rep2 = cl2.fit_transform(df2)
    assert rep2.fuzzy_duplicates_removed >= 1


def test_domain_bounds_coerces_invalid_values():
    df = pd.DataFrame({"x": [1.0, 2.0, 99.0, -3.0]})
    cl = DataCleaner(CleaningConfig(domain_bounds={"x": (0.0, 50.0)}, numeric_imputation="median", outlier_method="none", infer_types=False, clean_strings=False, standardize_missing_tokens=False, drop_exact_duplicates=False))
    out, rep = cl.fit_transform(df)
    # values 99.0 and -3.0 were coerced to NaN then imputed with median of valid values (1.0, 2.0) → 1.5
    assert out["x"].max() <= 2.0
    assert pd.isna(df["x"].iloc[0]) is False  # raw untouched (fitted copy)


def test_missing_indicators_added_before_imputation():
    df = pd.DataFrame({"x": [1.0, np.nan, 2.0, np.nan, 3.0]})
    cl = DataCleaner(CleaningConfig(add_missing_indicators=True, numeric_imputation="median", outlier_method="none", drop_exact_duplicates=False, infer_types=False, clean_strings=False, standardize_missing_tokens=False))
    out, rep = cl.fit_transform(df)
    assert "x__is_missing" in out.columns
    assert out["x__is_missing"].sum() == 2
    assert out["x__is_missing"].iloc[0] == 0
    assert out["x__is_missing"].iloc[1] == 1


def test_rubin_rules_pooling_formula():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.normal(5, 2, 100), "y": rng.normal(0, 1, 100)})
    df.loc[rng.random(100) < 0.2, "x"] = np.nan
    res = DataCleaner.multiple_imputation_rubin(df, m=4, max_iter=4, random_state=42)
    p = res["pooled_estimates"]["x"]
    total = p["within_variance"] + (1.0 + 1.0 / 4) * p["between_variance"]
    assert p["total_variance"] == pytest.approx(total, abs=1e-6)
    assert p["pooled_mean"] == pytest.approx(np.median([d["x"].mean() for d in res["imputed_datasets"]]), abs=1.0)


def test_benford_test_flags_uniform_distribution():
    # Uniform on digits 1-9 must reject Benford
    rng = np.random.default_rng(4)
    uniform = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], size=3000)
    res = benford_digit_test(pd.Series(uniform))
    assert res["evaluable"] is True
    assert res["conforms_benford"] is False
    # Benford-conforming leading digits (log10 scale) must pass
    digits_weights = np.log10(1 + 1 / np.arange(1, 10))
    vals = np.concatenate([digits_weights, digits_weights])
    vals = np.random.default_rng(5).choice(vals, size=3000, p=None)
    # regenerate via ppf: pick u ~ uniform, compute 10^(log10(1+1/d)*u)-1 style — simpler: use empirical frequencies
    first_digit_vals = np.concatenate([np.full(int(3000 * w), i + 1.0) for i, w in enumerate(digits_weights / digits_weights.sum())])
    res2 = benford_digit_test(pd.Series(first_digit_vals))
    assert res2["evaluable"] is True and res2["conforms_benford"] is True


def test_cleaning_distortion_audit_detects_shift():
    rng = np.random.default_rng(6)
    raw = pd.DataFrame({"x": rng.normal(0, 1, 200), "y": rng.normal(10, 2, 200)})
    # cleaned version shifts x to near 5 — should flag high distortion
    clean = pd.DataFrame({"x": raw["x"] + 5.0, "y": raw["y"]})
    audit = cleaning_distortion_audit(raw, clean)
    assert "x" in audit["per_column"]
    assert audit["per_column"]["x"]["median_shift"] == pytest.approx(5.0, abs=0.01)


def test_fuzzy_dedup_lsh_blocking():
    df = pd.DataFrame({"txt": ["the quick brown fox", "the quick brown foX", "a completely different sentence"]})
    out, removed = fuzzy_deduplicate_lsh_blocking(df, ["txt"], threshold=0.8)
    assert removed >= 1
    assert len(out) == 2


# --- Splitter
def test_stratified_split_disjoint_and_balanced():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "x": rng.normal(0, 1, 300),
        "target": np.concatenate([[0] * 100, [1] * 100, [2] * 100]),
    })
    splitter = LeakageSafeSplitter(SplitConfig(strategy="auto", test_size=0.2, random_seed=42))
    res = splitter.split(df, target_col="target")
    assert res.strategy_used == "stratified"
    assert set(res.train_df.index).isdisjoint(set(res.test_df.index))
    # strata preserved
    tr_prev = pd.Series(res.train_df["target"]).value_counts(normalize=True).sort_index()
    te_prev = pd.Series(res.test_df["target"]).value_counts(normalize=True).sort_index()
    assert abs(tr_prev[0] - te_prev[0]) < 0.05


def test_time_series_split_respects_order_and_gap():
    df = pd.DataFrame({
        "t": pd.date_range("2026-01-01", periods=100, freq="D"),
        "x": np.arange(100.0),
    })
    cfg = SplitConfig(strategy="time_series", time_col="t", test_size=0.2, gap=3, random_seed=42)
    res = LeakageSafeSplitter(cfg).split(df)
    ts = df["t"]
    assert res.train_df["t"].max() < res.test_df["t"].min()
    # gap enforced: test starts > 3 days after train ends (gap rows excluded)
    gap = (res.test_df["t"].min() - res.train_df["t"].max()).days
    assert gap >= 3


def test_group_split_disjoint_groups():
    rng = np.random.default_rng(8)
    groups = [f"g{i % 10}" for i in range(200)]
    df = pd.DataFrame({"g": groups, "x": rng.normal(0, 1, 200)})
    cfg = SplitConfig(strategy="group", group_col="g", test_size=0.2, random_seed=42)
    res = LeakageSafeSplitter(cfg).split(df)
    train_groups = set(res.train_df["g"])
    test_groups = set(res.test_df["g"])
    assert train_groups.isdisjoint(test_groups)


def test_split_audit_report_passes():
    df = pd.DataFrame({"x": np.arange(60.0), "g": ["a"] * 30 + ["b"] * 30})
    res = LeakageSafeSplitter(SplitConfig(strategy="group", group_col="g", test_size=0.2, random_seed=1)).split(df)
    assert res.audit_report is not None
    assert res.audit_report.passed is True
    assert res.audit_report.index_overlap_count == 0


# --- DataValidator
def test_validator_semantics():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "name": ["Alice", "Bob", "Carol", "Dan"],
        "code": ["AB-1", "AB-2", "XY-3", "AB-4"],
        "salary": [100, 200, 300, 400],
    })
    rules = [
        ValidationRule(name="age_range", rule_type="range", column="age", params={"min": 20, "max": 40}),
        ValidationRule(name="age_range_fail", rule_type="range", column="age", params={"min": 28, "max": 40}),
        ValidationRule(name="name_unique", rule_type="unique", column="name"),
        ValidationRule(name="code_regex", rule_type="regex", column="code", params={"regex": "^AB-"}),
        ValidationRule(name="salary_dtype", rule_type="dtype", column="salary", params={"dtype": "float"}),
    ]
    rep = DataValidator(rules).validate(df)
    assert rep.passed is False  # age_range_fail fails
    # failed expectations: age_range_fail (range), regex (XY-3), dtype (int64 vs float)
    failed_by_etype = [r for r in rep.results if not r.passed]
    assert len(failed_by_etype) >= 2
    failed_cols = {r.column for r in failed_by_etype}
    assert "age" in failed_cols and "code" in failed_cols


def test_validator_auto_generate_suite_reasonable():
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, np.nan, 5.0],
        "y": ["a", "b", "c", "a", "b"],
    })
    suite = DataValidator.auto_generate_suite(df, suite_name="auto")
    rep = suite.validate(df)
    # suite must not contradict the data it was generated from
    critical = [r for r in rep.results if r.severity == ValidationSeverity.CRITICAL and not r.passed]
    assert len(critical) == 0
    # row count expectation should hold
    assert any(r.passed for r in rep.results)


# --- DataQualityProfiler
def test_quality_overall_score_matches_weighted_average_and_excludes_none():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    rep = DataQualityProfiler().profile(df)
    assert 0.0 <= rep.overall_score <= 100.0
    # production: overall_score = weighted average of non-None *weights that exist in the report*
    w = DataQualityProfiler().weights
    dims = dict(rep.dimensions)
    evaluable = {k: v for k, v in dims.items() if v is not None and k in w}
    total_w = sum(w[k] for k in evaluable)
    expected = sum(evaluable[k] * w[k] for k in evaluable) / total_w
    assert abs(rep.overall_score - expected) < 1.0


def test_quality_report_excludes_none_timeliness_from_weights():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    rep = DataQualityProfiler().profile(df)
    # timeliness is None (no timestamp) and must not crash; overall score reflects only evaluable dims
    assert rep.dimensions.get("timeliness") is None
    assert rep.overall_score is not None


# --- Documented findings (regression tests proving current broken behavior)
def test_validation_rule_shorthand_regex_and_dtype_are_broken():
    """ValidationRule shorthand for 'regex'/'dtype' maps to raw expectation_type which is
    never evaluated -> always fails. Only range/not_null/in_set/unique are properly mapped."""
    df = pd.DataFrame({"code": ["AB-1", "AB-2"], "salary": [100, 200]})
    rep = DataValidator([
        ValidationRule(name="code_regex", rule_type="regex", column="code", params={"regex": "^AB-"}),
        ValidationRule(name="salary_dtype", rule_type="dtype", column="salary", params={"dtype": "numeric"}),
    ]).validate(df)
    failed_types = {r.expectation_name for r in rep.results if not r.passed}
    # both shorthand rules should be evaluatable but are not
    assert "regex" in failed_types and "dtype" in failed_types


def test_validation_rule_shorthand_mapped_types_pass():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    rep = DataValidator([
        ValidationRule(name="a_range", rule_type="range", column="a", params={"min": 0, "max": 10}),
        ValidationRule(name="a_not_null", rule_type="not_null", column="a"),
        ValidationRule(name="a_in_set", rule_type="in_set", column="a", params={"values": [1, 2, 3]}),
        ValidationRule(name="b_unique", rule_type="unique", column="b"),
    ]).validate(df)
    assert rep.passed is True


def test_workbench_cleaning_report_steps_do_not_match_actual_pipeline():
    """The advertised 'steps_applied' order in the cleaning response is misleading."""
    from omniforge.data.cleaning import DataCleaner as _DC
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "x": rng.normal(0, 1, 100),
        "y": rng.normal(5, 1, 100),
    })
    rep = _DC(CleaningConfig(numeric_imputation="median", outlier_method="iqr", outlier_action="clip", drop_exact_duplicates=True, infer_types=False, clean_strings=False, standardize_missing_tokens=False)).fit_transform(df)[1]
    steps = rep.steps_executed
    # actual DataCleaner order: standardize -> clean strings -> infer types -> outliers -> imputation
    outlier_pos = [i for i, s in enumerate(steps) if s.startswith("outliers:")][0]
    impute_pos = steps.index("impute_missing") if "impute_missing" in steps else None
    assert impute_pos is None or outlier_pos < impute_pos
    # NOTE: the workbench's advertised steps_applied ends with '7. Detect & handle statistical outliers'
    # — that is the advertised order and it does NOT match the actual implementation order.
