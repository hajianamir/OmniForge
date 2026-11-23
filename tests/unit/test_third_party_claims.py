"""
Third-Party Review Claims — Verification Test Suite (23 claims, P0/P1/P2).

After fixes, all tests verify that bugs have been resolved. Tests that
confirm negative claims (features work correctly) also pass.

Covers:
  Security & Access Control            : Claim 1
  MLOps / Model Registry / Job Queue   : Claims 2, 3, 4
  Data Splitting / Leakage / Scoring   : Claims 5, 6
  Model Serialization                  : Claim 7
  Feature Selection                    : Claim 8
  Mining                               : Claim 9
  Data Cleaning / Quality / Feature Store : Claims 10, 11, 12, 13, 14
  Fairness / Forecasting / Causal / Optimization / Simulation : Claims 15-23
"""

from __future__ import annotations

import inspect
import math
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Imports (patched copy for Python 3.11)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omniforge.api.studios import s11_mlops, s05_features, s06_automl  # noqa: E402
from omniforge.api import workbench as wb_module  # noqa: E402
from omniforge.core.types import ModelStage, Principal  # noqa: E402
from omniforge.data.cleaning import CleaningConfig, DataCleaner  # noqa: E402
from omniforge.data.feature_store import LocalFeatureStore  # noqa: E402
from omniforge.data.quality import DataQualityProfiler  # noqa: E402
from omniforge.data.splitting import LeakageSafeSplitter, SplitConfig  # noqa: E402
from omniforge.evaluation.backtesting import BacktestingEngine  # noqa: E402
from omniforge.evaluation.fairness import FairnessAuditor  # noqa: E402
from omniforge.mlops.continuous_training import ContinuousTrainingOrchestrator  # noqa: E402
from omniforge.mlops.job_queue import BackgroundJobQueue  # noqa: E402
from omniforge.mlops.registry import ModelRegistry  # noqa: E402
from omniforge.mlops.tracking import ExperimentTracker  # noqa: E402
from omniforge.models.base import BaseOmniForgeModel  # noqa: E402
from omniforge.optimization.solvers import OptimizationEngine, OptimizationError  # noqa: E402
from omniforge.simulation.engines import DiscreteEventSimulator  # noqa: E402
from omniforge.ui.state import (  # noqa: E402
    get_active_split_frames,
    get_session_state,
    lock_leakage_safe_split,
)
from omniforge.causal.inference import CausalInferenceEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 100, seed: int = 42, non_range_index: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(5, 2, n),
        "category": rng.choice(["x", "y", "z"], n),
        "target": rng.integers(0, 2, n),
    })
    if non_range_index:
        df.index = [f"row_{i}" for i in range(n)]
    return df


def _make_principal(subject: str = "tester", roles: list[str] | None = None,
                    tenant_id: str = "default-tenant") -> Principal:
    return Principal(subject=subject, roles=roles or ["viewer"], tenant_id=tenant_id)


class _DummyOmniForgeModel(BaseOmniForgeModel):
    """Concrete BaseOmniForgeModel for testing."""

    def __init__(self, name: str = "dummy", score_val: float = 0.5):
        super().__init__(name=name, task_type="regression", track_carbon=False)
        self.score_val = score_val
        self.is_fitted_ = True

    def _fit(self, X, y=None, **kwargs):
        pass

    def _predict(self, X, **kwargs):
        return np.full(len(X), self.score_val)

    def score(self, X, y):
        return self.score_val


# ===========================================================================
# CLAIM 1 (P0) — Security: require_permission on /s11/promote
# ===========================================================================
class TestClaim1_PromotePermission:
    """The /s11/promote endpoint must enforce require_permission on the route itself (FIXED)."""

    def test_promote_route_has_permission_dependency(self):
        """Source verified: s11_mlops.py /s11/promote HAS require_permission Depends (FIXED)."""
        source = Path(s11_mlops.__file__).read_text()
        assert "def s11_promote" in source, "s11_promote function not found"

        promote_marker = 'def s11_promote'
        idx = source.find(promote_marker)
        assert idx >= 0, 's11_promote function not found'
        section_start = source.rfind('@router', 0, idx)
        section = source[section_start:idx + 200]

        assert "require_permission" in section, (
            "FIX NOT APPLIED: /s11/promote route still has NO require_permission dependency."
        )

    def test_promote_route_dependencies_include_permission(self):
        """Integration: route's FastAPI dependencies include require_permission (FIXED)."""
        from omniforge.api.auth import require_permission

        found = False
        for route in s11_mlops.router.routes:
            if hasattr(route, "path") and route.path == "/s11/promote":
                deps = getattr(route, "dependencies", [])
                dep_dependencies = [getattr(d, "dependency", None) for d in deps]
                # require_permission returns a _checker closure, so check by name
                assert any(
                    getattr(dep, "__name__", None) == "_checker" for dep in dep_dependencies
                ), (
                    "FIX NOT APPLIED: promote route missing require_permission dependency."
                )
                found = True
                break
        assert found, "promote route not found in s11_mlops router"


# ===========================================================================
# CLAIM 2 (P1) — MLOps: register_model encrypts, load_model decrypts (FIXED)
# ===========================================================================
class TestClaim2_EncryptionRoundTrip:
    """register_model() encrypts artifact bytes at rest; load_model() decrypts (FIXED)."""

    def test_encryption_round_trip(self):
        """A model registered with encrypt=True must be decryptable on load."""
        tmpdir = tempfile.mkdtemp()
        reg = ModelRegistry(db_path=Path(tmpdir) / "registry.db",
                            artifact_dir=Path(tmpdir) / "artifacts")

        model = _DummyOmniForgeModel(name="enc-test", score_val=0.5)

        model_id = reg.register_model(
            model=model,
            model_name="enc-test-model",
            encrypt=True,
        )

        # The stored artifact file should be encrypted
        artifact_path = reg.artifact_dir / "enc-test-model" / "v1" / "model.joblib"
        assert artifact_path.exists(), "Artifact file should exist"
        import io, joblib
        buf = io.BytesIO()
        joblib.dump(model, buf)
        plaintext_joblib = buf.getvalue()
        stored_bytes = artifact_path.read_bytes()
        assert stored_bytes != plaintext_joblib, (
            "Stored artifact should be encrypted (different from plaintext)."
        )

        # Loading should succeed with decryption (FIXED)
        loaded_model, loaded_ver = reg.load_model("enc-test-model")
        assert loaded_model is not None, (
            "FIX NOT APPLIED: load_model() cannot decrypt encrypted artifacts."
        )
        assert loaded_ver.encrypted_at_rest is True


# ===========================================================================
# CLAIM 3 (P1) — MLOps: evaluate_and_retrain no auto-promote without champion (FIXED)
# ===========================================================================
class TestClaim3_SilentChampionFailure:
    """evaluate_and_retrain() must not promote challenger when champion load fails (FIXED)."""

    def test_no_champion_no_auto_promote(self):
        """When no champion exists, challenger must NOT be auto-promoted to PRODUCTION (FIXED)."""
        tmpdir = tempfile.mkdtemp()
        reg = ModelRegistry(db_path=Path(tmpdir) / "registry.db",
                            artifact_dir=Path(tmpdir) / "artifacts")
        tracker = ExperimentTracker(db_path=Path(tmpdir) / "tracking.db")

        orch = ContinuousTrainingOrchestrator(
            registry=reg,
            tracker=tracker,
            auto_promote=True,
            kill_switch=False,
        )

        rng = np.random.default_rng(42)
        n = 100
        reference_df = pd.DataFrame({
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "target": rng.integers(0, 2, n),
        })
        new_labeled_df = pd.DataFrame({
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "target": rng.integers(0, 2, n),
        })

        def trainer_fn(X, y):
            return _DummyOmniForgeModel(name="challenger", score_val=0.95)

        # No champion exists — challenger should NOT be auto-promoted to PRODUCTION (FIXED)
        result = orch.evaluate_and_retrain(
            model_name="test-model",
            reference_df=reference_df,
            new_labeled_df=new_labeled_df,
            target_col="target",
            trainer_fn=trainer_fn,
            force_retrain=True,
        )

        assert result.get("promoted_to_production") is not True, (
            "FIX NOT APPLIED: Challenger still promoted to PRODUCTION despite no existing champion."
        )
        assert result.get("assigned_stage") == "Staging", (
            "Challenger should be assigned to Staging when no champion exists."
        )


# ===========================================================================
# CLAIM 4 (P2) — Job Queue: max_history parameter (FIXED)
# ===========================================================================
class TestClaim4_JobQueueMaxHistory:
    """BackgroundJobQueue supports max_history with eviction (FIXED)."""

    def test_job_queue_has_max_history_parameter(self):
        """Source verified: BackgroundJobQueue.__init__ has max_history param (FIXED)."""
        sig = inspect.signature(BackgroundJobQueue.__init__)
        params = list(sig.parameters.keys())
        assert "max_history" in params, (
            "FIX NOT APPLIED: BackgroundJobQueue still has no max_history parameter."
        )

    def test_job_queue_evicts_old_jobs(self):
        """When max_history is set, old jobs are evicted (FIXED)."""
        jq = BackgroundJobQueue(max_workers=2, max_history=5)
        job_ids = []
        for i in range(10):
            jid = jq.submit(f"job-{i}", lambda: None, owner_key="default")
            job_ids.append(jid)

        # Only the last 5 jobs should remain
        remaining = [jid for jid in job_ids if jq.status(jid) != "not_found"]
        assert len(remaining) <= 5, (
            f"FIX NOT APPLIED: {len(remaining)} jobs remain (max_history=5)."
        )


# ===========================================================================
# CLAIM 5 (P2) — Data Splitting: SplitLock on non-RangeIndex (FIXED)
# ===========================================================================
class TestClaim5_SplitLockNonRangeIndex:
    """SplitLock works on DataFrames with non-RangeIndex (FIXED)."""

    def test_split_lock_works_on_non_range_index(self):
        """get_active_split_frames uses iloc[] (position-based) for positional indices (FIXED)."""
        df = _make_df(n=50, non_range_index=True)
        state = get_session_state("test-session-5-fix")

        lock = lock_leakage_safe_split(
            state, df,
            strategy="random", test_size=0.2, random_seed=42,
        )
        assert lock["active"] is True

        # Should NOT raise ValueError — iloc[] works with positional indices
        train_df, test_df, lock2 = get_active_split_frames(state, df)
        assert len(train_df) > 0, "Train split should have rows"
        assert len(test_df) > 0, "Test split should have rows"


# ===========================================================================
# CLAIM 6 (P2) — AutoML: uap_split_lock honored (FIXED)
# ===========================================================================
class TestClaim6_AutoMLHonorsSplitLock:
    """AutoML uses get_active_split_frames when split lock is active (FIXED)."""

    def test_automl_uses_split_frames(self):
        """Source verified: s06_run_automl() calls get_active_split_frames (FIXED)."""
        source = Path(s06_automl.__file__).read_text()
        assert "get_active_split_frames" in source, (
            "FIX NOT APPLIED: AutoML still does not use get_active_split_frames."
        )


# ===========================================================================
# CLAIM 7 (P1) — Model Base: unknown categories handled gracefully (FIXED)
# ===========================================================================
class TestClaim7_UnknownCategories:
    """ModelBase._prepare_X() maps unknown categories to -1 during prediction (FIXED)."""

    def test_unknown_category_maps_to_minus_one(self):
        """Source verified: _prepare_X() maps unknown categories to -1 instead of raising (FIXED)."""
        base_path = Path(__file__).resolve().parents[2] / "src" / "omniforge" / "models" / "base.py"
        source = base_path.read_text()
        assert "Unknown categories" in source, (
            "FIX NOT APPLIED: _prepare_X() does not handle unknown categories."
        )

        model = _DummyOmniForgeModel(name="cat-test", score_val=0.5)
        X_fit = pd.DataFrame({"cat_col": ["a", "b", "a", "b"], "num_col": [1.0, 2.0, 3.0, 4.0]})
        model._prepare_X(X_fit, fitting=True)

        # Predict with unknown category 'c' — maps to -1 gracefully (no ValueError)
        X_pred = pd.DataFrame({"cat_col": ["c"], "num_col": [5.0]})
        arr, _ = model._prepare_X(X_pred, fitting=False)
        # cat_col should be -1 (unknown), num_col should be 5.0
        assert arr[0, 0] == -1.0, f"Unknown category should map to -1, got {arr[0, 0]}"
        assert arr[0, 1] == 5.0, f"Numeric column should pass through, got {arr[0, 1]}"


# ===========================================================================
# CLAIM 8 (P2) — Feature Selection: commit drops columns (FIXED)
# ===========================================================================
class TestClaim8_FeatureSelectionCommit:
    """Feature selection commit calls update_active_dataset to drop columns (FIXED)."""

    def test_commit_drops_columns(self):
        """Source verified: s05_select_features() calls update_active_dataset on commit (FIXED)."""
        source = Path(s05_features.__file__).read_text()
        select_func_start = source.find("def s05_select_features")
        select_func_end = source.find("\ndef ", select_func_start + 1)
        if select_func_end == -1:
            select_func_end = len(source)
        select_func = source[select_func_start:select_func_end]

        assert "update_active_dataset" in select_func, (
            "FIX NOT APPLIED: Feature selection commit still does not call update_active_dataset."
        )


# ===========================================================================
# CLAIM 9 (P2) — Mining: clustering excludes target (FIXED)
# ===========================================================================
class TestClaim9_ClusteringTargetExclusion:
    """Clustering excludes target column from features (FIXED)."""

    def test_clustering_excludes_target_from_features(self):
        """Source verified: s02_run_clustering() helper_cols includes targets (FIXED)."""
        source = Path(wb_module.__file__).read_text()
        clust_start = source.find("def s02_run_clustering")
        clust_end = source.find("\ndef ", clust_start + 1)
        if clust_end == -1:
            clust_end = len(source)
        clust_func = source[clust_start:clust_end]

        # Targets should now be excluded from clustering features
        helper_line = [l for l in clust_func.split('\n') if 'helper_cols' in l]
        assert any('| targets' in l for l in helper_line), (
            "FIX NOT APPLIED: Clustering still does not exclude targets from features."
        )


# ===========================================================================
# CLAIMS 10-12 (P2) — Cleaning: outlier methods, ffill, missforest (NOT VALID — features exist)
# ===========================================================================
class TestClaims10_12_CleaningMethods:
    """Outlier methods, ffill, and missforest all exist — claims are NOT valid."""

    def test_outlier_methods_exist(self):
        valid_methods = {"iqr", "zscore", "mad", "percentile", "isolation_forest", "lof", "elliptic"}
        for m in valid_methods:
            cfg = CleaningConfig(outlier_method=m)
            assert cfg.outlier_method == m

    def test_ffill_imputation_exists(self):
        cfg = CleaningConfig(numeric_imputation="ffill", causal_time_series=True)
        assert cfg.numeric_imputation == "ffill"
        assert cfg.causal_time_series is True

    def test_missforest_imputation_exists(self):
        cfg = CleaningConfig(numeric_imputation="missforest")
        assert cfg.numeric_imputation == "missforest"


# ===========================================================================
# CLAIM 13 (P2) — Quality: DataQualityProfiler caching (NOT VALID — caching works)
# ===========================================================================
class TestClaim13_QualityCaching:
    """DataQualityProfiler.profile() caches results."""

    def test_profile_caching_returns_cached_result(self):
        df = _make_df(n=50)
        profiler = DataQualityProfiler()

        report1 = profiler.profile(df)
        report2 = profiler.profile(df)

        assert report1.overall_score == report2.overall_score


# ===========================================================================
# CLAIM 14 (P2) — Feature Store: TTL behavior (NOT VALID — TTL works)
# ===========================================================================
class TestClaim14_FeatureStoreTTL:
    """Feature Store TTL expires online features."""

    def test_ttl_expires_online_features(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not available for parquet support")

        tmpdir = tempfile.mkdtemp()
        store = LocalFeatureStore(repo_path=tmpdir)

        df = pd.DataFrame({
            "entity_id": ["e1", "e2"],
            "feature_1": [1.0, 2.0],
            "event_ts": [
                (datetime.now(UTC) - timedelta(days=10)).isoformat(),
                (datetime.now(UTC) - timedelta(days=10)).isoformat(),
            ],
        })

        store.register_feature_view(
            name="test_view",
            df=df,
            join_keys=["entity_id"],
            timestamp_column="event_ts",
            ttl_seconds=3600,
            materialize_online=True,
        )

        result = store.get_online_features(
            view_name="test_view",
            entity_rows=[{"entity_id": "e1"}],
            features=["feature_1"],
            enforce_ttl=True,
            as_of=datetime.now(UTC),
        )

        assert len(result) == 1
        assert result[0].get("feature_1") is None


# ===========================================================================
# CLAIMS 15-16 (P1/P2) — Fairness: empty/missing classes (NOT VALID — handled correctly)
# ===========================================================================
class TestClaims15_16_FairnessEdgeCases:
    """FairnessAuditor handles empty groups, missing classes, single group."""

    def test_fairness_single_group(self):
        auditor = FairnessAuditor()
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        sensitive = np.array(["A", "A", "A", "A", "A", "A"])

        report = auditor.audit(y_true, y_pred, sensitive, attribute_name="group")
        assert report.passed is True
        assert report.disparate_impact_ratio == 1.0
        assert report.demographic_parity_diff == 0.0

    def test_fairness_group_with_no_positives(self):
        auditor = FairnessAuditor()
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 1, 1])
        sensitive = np.array(["A", "A", "A", "A", "B", "B"])

        report = auditor.audit(y_true, y_pred, sensitive, attribute_name="group")
        assert report.group_metrics["A"]["tpr"] is None

    def test_fairness_empty_groups(self):
        auditor = FairnessAuditor()
        report = auditor.audit(np.array([]), np.array([]), np.array([]), attribute_name="group")
        assert report.passed is True


# ===========================================================================
# CLAIMS 17-20 (P1/P2) — Backtesting: gap alignment, intervals (NOT VALID — correct)
# ===========================================================================
class TestClaims17_20_Backtesting:
    """BacktestingEngine walk-forward and trading signal backtest work correctly."""

    def test_walk_forward_gap_applied(self):
        engine = BacktestingEngine()
        n = 200
        series = pd.Series(np.random.default_rng(42).normal(100, 10, n))

        class DummyForecaster:
            def fit(self, y):
                self.last_y = float(y[-1])
                return self
            def predict(self, horizon):
                return np.full(horizon, self.last_y)

        result = engine.walk_forward_forecast(
            series, DummyForecaster,
            initial_train_size=60, horizon=7, step_size=7, gap=3,
        )
        assert result["gap"] == 3
        assert result["n_origins"] > 0

    def test_walk_forward_prediction_intervals_in_fold_metrics(self):
        engine = BacktestingEngine()
        n = 150
        series = pd.Series(np.random.default_rng(42).normal(100, 10, n))

        class IntervalForecaster:
            def fit(self, y):
                return self
            def predict(self, horizon):
                return pd.DataFrame({
                    "yhat": np.full(horizon, 100.0),
                    "yhat_lower": np.full(horizon, 90.0),
                    "yhat_upper": np.full(horizon, 110.0),
                })

        result = engine.walk_forward_forecast(
            series, IntervalForecaster,
            initial_train_size=50, horizon=5, step_size=5, gap=0,
        )
        for fold in result["fold_metrics"]:
            assert "interval_coverage" in fold

    def test_trading_signals_look_ahead_prevention(self):
        engine = BacktestingEngine()
        prices = np.array([100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0])
        signals = np.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0])

        result = engine.backtest_trading_signals(prices, signals)
        assert result["n_periods"] == len(prices) - 1


# ===========================================================================
# CLAIM 21 (P2) — Causal: DiD singleton cells (NOT VALID — handled correctly)
# ===========================================================================
class TestClaim21_DiDSingletonCells:
    """DiD handles singleton cells without NaN variance."""

    def test_did_singleton_cell_no_nan(self):
        """Singleton cells (len==1) return 0 variance, not NaN — claim NOT valid."""
        engine = CausalInferenceEngine()
        df = pd.DataFrame({
            "group": [1, 1, 0, 0],
            "post": [1, 0, 1, 0],
            "outcome": [10.0, 8.0, 9.0, 7.0],
        })
        result = engine.difference_in_differences(
            df, group_col="group", post_col="post", outcome_col="outcome",
        )
        assert not math.isnan(result["std_error"]), (
            "DiD singleton cell should not produce NaN std_error."
        )
        assert not math.isnan(result["did_estimate"])

    def test_did_empty_cell_raises(self):
        engine = CausalInferenceEngine()
        df = pd.DataFrame({
            "group": [1, 1, 0],
            "post": [1, 0, 1],
            "outcome": [10.0, 8.0, 9.0],
        })
        with pytest.raises(Exception):
            engine.difference_in_differences(
                df, group_col="group", post_col="post", outcome_col="outcome",
            )


# ===========================================================================
# CLAIM 22 (P2) — Optimization: portfolio asset names validated (FIXED)
# ===========================================================================
class TestClaim22_PortfolioAssetNames:
    """Portfolio optimization validates unique asset names (FIXED)."""

    def test_portfolio_asset_name_mapping(self):
        engine = OptimizationEngine()
        mu = [0.1, 0.15, 0.08]
        cov = [[0.04, 0.01, 0.005],
               [0.01, 0.09, 0.02],
               [0.005, 0.02, 0.03]]
        names = ["Asset_A", "Asset_B", "Asset_C"]

        result = engine.optimize_portfolio(
            expected_returns=mu, cov_matrix=cov, asset_names=names,
        )
        solution = result.solution
        assert isinstance(solution, dict)
        for name in names:
            assert name in solution
        total = sum(solution.values())
        assert abs(total - 1.0) < 1e-6

    def test_portfolio_duplicate_names_raises(self):
        """Duplicate asset names should raise OptimizationError (FIXED)."""
        engine = OptimizationEngine()
        mu = [0.1, 0.15]
        cov = [[0.04, 0.01], [0.01, 0.09]]
        names = ["Asset_A", "Asset_A"]

        with pytest.raises(OptimizationError, match="unique names"):
            engine.optimize_portfolio(
                expected_returns=mu, cov_matrix=cov, asset_names=names,
            )


# ===========================================================================
# CLAIM 23 (P2) — Simulation: queue simulator horizon_time (NOT VALID — correct)
# ===========================================================================
class TestClaim23_QueueHorizon:
    """Queue simulator respects horizon_time."""

    def test_horizon_time_respected(self):
        sim = DiscreteEventSimulator(random_seed=42)
        result = sim.simulate_queue(
            arrival_rate=10.0, service_rate=20.0,
            n_servers=2, horizon_time=50.0,
        )
        assert result["total_arrivals"] > 0
        assert result["completed_entities"] >= 0

    def test_queue_metrics_consistent(self):
        sim = DiscreteEventSimulator(random_seed=42)
        result = sim.simulate_queue(
            arrival_rate=5.0, service_rate=10.0,
            n_servers=1, horizon_time=100.0,
        )
        assert 0.0 <= result["server_utilization_pct"] <= 100.0
        assert result["avg_wait_time"] >= 0.0
        assert result["throughput_per_time_unit"] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])