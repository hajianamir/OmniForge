"""Additional audit evidence for the explicitly selected Phase 2–4 worklist."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import stats

from omniforge.api.auth import get_api_key
from omniforge.api.main import create_app
from omniforge.data.splitting import CombinatorialPurgedKFold as LegacyCPCV
from omniforge.evaluation.validation import CombinatorialPurgedKFold
from omniforge.models.mining.miner import AssociationRuleMiner, SubgroupDiscoveryMiner


def _data(frame: pd.DataFrame) -> TestClient:
    client = TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": f"residuals-{uuid4().hex}"})
    response = client.post("/api/workbench/s01/upload", files={"file": ("residuals.csv", frame.to_csv(index=False), "text/csv")})
    assert response.status_code == 200, response.text
    return client


def test_non_shapiro_scope_includes_hidden_observations_and_groups() -> None:
    n = 5100
    client = _data(pd.DataFrame({"measurement": np.random.default_rng(23).normal(size=n), "group": np.tile(list("abcdefghijklmno"), 340)}))
    for test in ("ks_normality", "anderson", "dagostino", "levene"):
        response = client.post("/api/workbench/s04/hypothesis", json={"test_type": test, "col_a": "measurement", "col_b": "group", "mode": "custom"})
        assert response.status_code == 200, response.text
        scope = response.json()["result"]["analysis_scope"]
        expected = (15, 12, 3) if test == "levene" else (n, 5000, 100)
        assert (scope["available"], scope["used"], scope["omitted"]) == expected


def test_advanced_mining_discloses_actual_limits() -> None:
    rng = np.random.default_rng(49)
    client = _data(pd.DataFrame(rng.normal(size=(300, 9)), columns=[f"x{i}" for i in range(8)] + ["target"]))
    assert client.post("/api/workbench/targets", json={"targets": ["target"]}).status_code == 200
    response = client.post("/api/workbench/s02/advanced_mining")
    assert response.status_code == 200, response.text
    result = response.json()
    scope = result["analysis_scope"]
    assert scope["clustering"]["complete_case_rows"] == 300
    assert scope["clustering"]["hopkins_sample_cap"] == scope["clustering"]["hopkins_sampled_rows"] == 50
    assert scope["distance_correlation"]["eligible_numeric_predictors"] == 8
    assert scope["distance_correlation"]["evaluated_predictors"] == 6
    assert set(result["distance_correlation"]) == {f"x{i}" for i in range(6)}


def test_locked_deep_holdout_reports_unstratifiable_cap_fallback() -> None:
    frame = pd.DataFrame({"feature": np.arange(300) % 17, "outcome": [i % 2 for i in range(240)] + [0] * 59 + [1]})
    client = _data(frame)
    locked = client.post("/api/workbench/split-lock", json={"strategy": "time_series", "test_size": 0.20})
    assert locked.status_code == 200, locked.text
    result = client.post("/api/workbench/s06/deep-learning", json={"target": "outcome", "architectures": [], "max_rows": 100})
    assert result.status_code == 200, result.text
    evaluation = result.json()["evaluation"]
    assert evaluation["holdout_sampling"] == "random_without_replacement"
    assert evaluation["available_holdout_class_counts"] == {"0": 59, "1": 1}
    assert sum(evaluation["holdout_class_counts"].values()) == 20
    assert all(count <= evaluation["available_holdout_class_counts"][cls] for cls, count in evaluation["holdout_class_counts"].items())


def test_benford_cleaning_audit_uses_finite_stored_magnitudes_at_threshold() -> None:
    frame = pd.DataFrame({"amount": [1.99] * 10 + [-2.0] * 12 + [2.01] * 10 + [0.0, np.inf, -np.inf, np.nan]})
    client = _data(frame)
    result = client.post("/api/workbench/s01/advanced_audit")
    assert result.status_code == 200, result.text
    observed = result.json()["benford_tests"]["amount"]
    assert observed["n"] == 32
    expected = stats.chisquare([10, 22, 0, 0, 0, 0, 0, 0, 0], 32 * np.log10(1 + 1 / np.arange(1, 10)))
    assert observed["chi2"] == round(expected.statistic, 4)


def test_cpcv_legacy_and_canonical_have_distinct_default_and_empty_contracts() -> None:
    frame = pd.DataFrame({"v": range(12)})
    legacy, canonical = LegacyCPCV(), CombinatorialPurgedKFold()
    assert legacy.get_n_splits() == 15
    assert canonical.get_n_splits() == 10
    assert isinstance(legacy.split(frame), list)
    stream = canonical.split(frame)
    assert iter(stream) is stream
    # Every training position is purged under these matched one-sided buffers.
    assert len(LegacyCPCV(n_splits=3, n_test_splits=1, purge_gap=20).split(frame)) == 3
    assert not list(CombinatorialPurgedKFold(n_splits=3, n_test_groups=1, purge_gap=20, embargo_gap=20).split(frame))


def test_mining_families_include_ties_and_filter_before_bh() -> None:
    baskets = [(i, item) for i in range(10) for item in (["a", "b"] if i < 4 else ["a"] if i < 6 else ["c"])]
    frame = pd.DataFrame(baskets, columns=["transaction", "item"])
    mine = AssociationRuleMiner.mine_transactions
    all_rules = mine(frame, "transaction", "item", min_support=0.2, min_lift=1.0, max_rules=10)["rules"]
    assert {(tuple(row["antecedent"]), tuple(row["consequent"])) for row in all_rules} == {(("a",), ("b",)), (("b",), ("a",))}
    p = stats.hypergeom.sf(3, 10, 6, 4)
    for row in all_rules:
        assert row["correction_family_size"] == 2
        assert row["fdr_q_value"] == pytest.approx(p, abs=1e-8)
    assert mine(frame, "transaction", "item", min_support=0.2, max_rules=1)["rules"][0] == all_rules[0]
    assert mine(frame, "transaction", "item", min_support=0.2, min_confidence=0.9)["n_rules"] == 1
    assert mine(frame, "transaction", "item", min_support=0.8)["rules"] == []

    y = np.r_[np.arange(10), np.arange(10) + 4, 20.0]
    cohorts = pd.DataFrame({"group": ["a"] * 10 + ["b"] * 10 + ["filtered"], "target": y})
    rows = SubgroupDiscoveryMiner().discover(cohorts, "target", min_subgroup_size=5, max_subgroups=10)["subgroups"]
    assert {r["subgroup_rule"] for r in rows} == {"group == 'a'", "group == 'b'"}
    p_a, p_b = sorted(round(stats.ttest_ind(y[mask], y[~mask], equal_var=False).pvalue, 6) for mask in (np.arange(21) < 10, (np.arange(21) >= 10) & (np.arange(21) < 20)))
    assert sorted(r["adjusted_p_value"] for r in rows) == [round(min(2 * p_a, p_b), 6), p_b]
    assert all(r["correction_family_size"] == 2 for r in rows)
    assert SubgroupDiscoveryMiner().discover(cohorts, "target", min_subgroup_size=11)["subgroups"] == []


def test_tabular_rule_family_has_explicit_membership_before_display_limit() -> None:
    frame = pd.DataFrame({"a": [1] * 6 + [0] * 4, "b": [1] * 4 + [0] * 6})
    engine = AssociationRuleMiner()
    result = engine.mine(frame, min_support=0.3, min_confidence=0.6, min_lift=1.01, max_rules=10)
    rules = result["rules"]
    # Positive association in both tails of the table; directions are separate hypotheses.
    assert {(r["antecedent"], r["consequent"]) for r in rules} == {
        ("a=1", "b=1"), ("b=1", "a=1"), ("a=0", "b=0"), ("b=0", "a=0"),
    }
    expected = stats.hypergeom.sf(3, 10, 6, 4)
    assert all(row["correction_family_size"] == 4 and row["fdr_q_value"] == pytest.approx(expected, abs=1e-8) for row in rules)
    assert engine.mine(frame, min_support=0.3, min_confidence=0.6, min_lift=1.01, max_rules=1)["rules"] == rules[:1]


@pytest.mark.parametrize("backend", ["h2o", "autogluon", "tpot", "flaml", "autosklearn"])
def test_optional_adapter_contract_does_not_reuse_foreign_search_score_or_guess_dimensions(
    monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    from omniforge.automl.backends import AutoMLBackendHub

    train = pd.DataFrame({"x": np.arange(60, dtype=float) - 30.0})
    train["target"] = (train["x"] >= 0).astype(int)
    holdout = pd.DataFrame({"x": [-9.0, -2.0, 2.0, 9.0]})
    target = pd.Series([0, 0, 1, 1])

    class Perfect:
        def predict(self, features: pd.DataFrame) -> np.ndarray:
            assert list(features.columns) == ["x"]
            return (features["x"] >= 0).to_numpy(dtype=int)

    def fake(*args: object) -> dict[str, object]:
        return {"best_model_name": "selected", "best_model": Perfect(), "best_score": -99.0,
                "elapsed_seconds": 0.0, "leaderboard": [{"model": "selected", "score": -99.0}]}

    monkeypatch.setattr(AutoMLBackendHub, "_try_external_adapter", staticmethod(fake))
    result = AutoMLBackendHub.run(train, "target", backend=backend, holdout=(holdout, target))
    evidence = result.extra["evaluation"]
    assert result.primary_metric == "f1_weighted"
    assert result.best_score == evidence["validation_score"] == 1.0
    assert evidence["comparison_requires_matched_holdout"] is True
    row = result.leaderboard[0]
    assert row["input_features"] == 1 and row["fit_features"] is None and row["retained_features"] is None
    assert evidence["candidate_scores_backend_specific"] == (backend in {"h2o", "autogluon"})


def test_studio04_chart_caps_do_not_change_reported_fitted_cohorts() -> None:
    rng = np.random.default_rng(32)
    # The near-flat eigenvalue spectrum needs more than 12 PCs for 80% variance.
    frame = pd.DataFrame(rng.normal(size=(900, 22)), columns=[f"v{i}" for i in range(22)])
    client = _data(frame)

    def chart(chart_id: str, **extra: str) -> dict:
        response = client.post("/api/workbench/charts/generate", json={"chart_id": chart_id, "studio_id": "04", **extra})
        assert response.status_code == 200, response.text
        return response.json()

    regression = chart("bivariate_regression_ci", feature_x="v0", feature_y="v1")
    assert regression["analysis_scope"] == {"source_rows": 900, "fitted_pairs": 900, "displayed_points": 450, "display_cap": 450}
    pca = chart("pca_eigenvalue_pareto")
    assert pca["analysis_scope"] == {"components_available": 22, "components_displayed": 12, "display_cap": 12}
    assert pca["kpis"]["PCs for 80% Variance"] > 12
    ecdf = chart("empirical_ecdf_ks", feature_x="v0")
    assert ecdf["data"]["series"][0]["n"] == 900
    assert len(ecdf["data"]["series"][0]["x"]) == ecdf["analysis_scope"]["display_cap_per_group"] == 220
    violin = chart("stratified_violin_box", feature_x="v0")
    assert violin["analysis_scope"]["distributions_available"] == 22
    assert violin["analysis_scope"]["distributions_displayed"] == 6


def test_warm_start_adapter_counts_the_fitted_encoded_matrix() -> None:
    from omniforge.automl.backends import AutoMLBackendHub

    rng = np.random.default_rng(44)
    frame = pd.DataFrame({"x": rng.normal(size=60), "category": np.resize(["a", "b", "c"], 60)})
    frame["target"] = (frame["x"] > 0).astype(int)
    result = AutoMLBackendHub.run(frame, "target", backend="meta_bayesian", cv_folds=2)
    assert result.leaderboard
    for row in result.leaderboard:
        assert row["input_features"] == 2
        assert row["fit_features"] == 4
        assert row["retained_features"] is None
    sparse = next(row for row in result.extra["evaluation"]["baseline_feature_evidence"].values() if row.get("reference_kind") == "parsimonious_l1")
    assert sparse["fit_features"] == 4
    assert sparse["retained_features"] == len(sparse["selected_support"])


def test_automated_eda_reports_original_size_and_chart_limits() -> None:
    client = _data(pd.DataFrame({"value": np.arange(25_001)}))
    response = client.post("/api/workbench/s04/eda")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["n_rows_available"] == 25_001
    assert body["n_rows_analyzed"] == body["row_cap"] == 25_000
    assert body["display_limits"]["correlation_columns"] == 12
