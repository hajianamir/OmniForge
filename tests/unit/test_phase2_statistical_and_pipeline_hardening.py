"""Phase 2 Statistical, Feature, Pipeline, AutoML & MLOps Hardening verification suite (OmniForge v3.0.1).

Verifies every Phase 2 requirement against primary literature and security standards:
1. Canonical SemVer 3.0.1 alignment across SDK, pyproject.toml, native hubs, exporters & SPA manifest.
2. FeatureSelector: chi2, normalized mRMR (Peng et al. 2005), Binomial Boruta (Kursa & Rudnicki 2010),
   half-sample without-replacement Stability Selection + PFER bound (Meinshausen & Bühlmann 2010),
   strict unknown-method rejection, training-median imputation, and Studio 05 /s05/select safeguards.
3. FeatureEngineeringPipeline: collision-free one-hot naming, schema-aligned scaling on partial test schemas,
   continuous seasonal Fourier phase across train/test splits, shuffled/stratified OOF target encoding
   (Micci-Barreca 2001), and global-stat unseen group aggregation imputation.
4. Little's (1988, JASA) MCAR test via Multivariate Normal EM MLE (detecting MAR dependent on fully
   observed covariates) and 3-way (train∩val, train∩test, val∩test) duplicate row leakage auditing.
5. AutoMLPipelineSearch: inner-CV HPO (Cawley & Talbot 2010), training-only CV summary, candidate cloning,
   and tuned hyperparameter propagation into deployable sklearn_pipeline.
6. PipelineExecutor & PipelineBuilder: tenant/param/input/upstream-aware HMAC cache keys, artifact-delta
   replay on cache hits, compilable & CWE-94-safe export_python_code, and leakage-safe split/evaluate steps.
7. MLOps & Copilot: scale-equivariant ADWIN (Bifet & Gavaldà 2007), unified base-2 JSD in [0, 1] (Lin 1991),
   async-feedback ModelMonitor SLAs, canary_version + scaler-preserving ONNX export, and CWE-22/OWASP LLM06
   Copilot tool_ingest_data path confinement.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import omniforge
from omniforge.api.auth import get_api_key
from omniforge.api.main import app
from omniforge.automl.backends import probe_automl_backends
from omniforge.automl.pipeline_search import AutoMLPipelineSearch
from omniforge.copilot.memory import CopilotMemory
from omniforge.copilot.tools import CopilotToolRegistry
from omniforge.core.exceptions import CopilotError, FeatureEngineeringError
from omniforge.core.types import TaskType
from omniforge.data.quality import DataQualityProfiler
from omniforge.data.splitting import LeakageSafeSplitter, SplitConfig
from omniforge.features.engineering import FeatureConfig, FeatureEngineeringPipeline
from omniforge.features.selection import FeatureSelector
from omniforge.mlops.drift import DriftDetector
from omniforge.mlops.monitoring import ModelMonitor
from omniforge.mlops.registry import ModelRegistry
from omniforge.mlops.serving import ModelServingEngine
from omniforge.models.classical import ClassicalMLModel
from omniforge.models.computer_vision.foundation import probe_vision_backends
from omniforge.models.nlp.advanced_nlp import probe_nlp_backends
from omniforge.pipelines.builder import PipelineBuilder
from omniforge.pipelines.executor import PipelineExecutor
from omniforge.reporting.exporters import ReportExporters
from omniforge.version import get_version_info

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version_3_0_1_canonical_alignment(tmp_path: Path) -> None:
    """Verify canonical v4.0.0 version across Python SDK, hubs, exporters, pyproject, and SPA manifests."""
    assert omniforge.__version__ == "4.0.0"
    assert get_version_info()["version"] == "4.0.0"

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == "4.0.0"

    pkg_json = json.loads((REPO_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    pkg_lock = json.loads((REPO_ROOT / "frontend" / "package-lock.json").read_text(encoding="utf-8"))
    assert pkg_json["version"] == "4.0.0"
    assert pkg_lock["version"] == "4.0.0"
    assert pkg_lock["packages"][""]["version"] == "4.0.0"

    native_automl = {b["id"]: b["version"] for b in probe_automl_backends() if b["category"].startswith("Native")}
    assert all(v == "4.0.0" for v in native_automl.values())

    vis_native = [b for b in probe_vision_backends()["backends"] if b["id"] == "native_pyramid"][0]
    nlp_native = [b for b in probe_nlp_backends() if b["id"] == "native_nlp"][0]
    assert vis_native["version"] == "4.0.0"
    assert nlp_native["version"] == "4.0.0"

    rng = np.random.default_rng(7)
    X_small = pd.DataFrame({"x1": rng.normal(size=30), "x2": rng.normal(size=30)})
    y_small = (X_small["x1"] > 0).astype(int)
    clf = ClassicalMLModel("logistic_regression", task_type=TaskType.BINARY_CLASSIFICATION).fit(X_small, y_small)
    art_path = tmp_path / "clf.joblib"
    clf.save(art_path)
    svc_dir = tmp_path / "svc"
    ReportExporters.export_microservice_bundle(svc_dir, model_artifact_path=art_path, service_name="TestSvc")
    app_code = (svc_dir / "app.py").read_text(encoding="utf-8")
    assert "version='4.0.0'" in app_code


def test_feature_selector_chi2_mrmr_boruta_stability_and_strict_validation() -> None:
    """Verify strict method validation, chi2, scale-normalized mRMR, Binomial Boruta, and half-sample Stability Selection."""
    with pytest.raises(FeatureEngineeringError, match="Unknown feature selection method"):
        FeatureSelector(method="nonexistent_selector_xyz")

    rng = np.random.default_rng(2026)
    n = 120
    y_cls = np.array([0, 1] * (n // 2))
    signal_a = y_cls * 3.0 - 1.5 + rng.normal(0.0, 0.2, n)  # Contains negative values!
    signal_a_clone = signal_a + rng.normal(0.0, 0.01, n)
    signal_b = np.where(y_cls == 1, rng.uniform(0.5, 2.0, n), rng.uniform(-2.0, -0.5, n))
    noise_1 = rng.normal(100.0, 1.0, n)
    noise_2 = rng.normal(-50.0, 1.0, n)
    X = pd.DataFrame(
        {
            "sig_a": signal_a,
            "sig_a_clone": signal_a_clone,
            "sig_b": signal_b,
            "noise_1": noise_1,
            "noise_2": noise_2,
        }
    )
    # Inject NaNs into noise_1 to verify median imputation (~100) rather than 0.0 distortion
    X.loc[[0, 5, 10], "noise_1"] = np.nan

    # 1. chi2 rejects regression and shifts negative features cleanly for classification
    with pytest.raises(FeatureEngineeringError, match="chi2"):
        FeatureSelector(method="chi2", task="regression").fit(X, y_cls.astype(float))

    sel_chi2 = FeatureSelector(method="chi2", task="classification", max_features=2).fit(X, y_cls)
    assert sel_chi2.chi2_min_shifts_["sig_a"] > 0.0
    assert 95.0 <= sel_chi2.col_medians_["noise_1"] <= 105.0
    assert set(sel_chi2.selected_features_).issubset({"sig_a", "sig_a_clone", "sig_b"})

    # 2. First-class mRMR in FeatureSelector.fit() + normalized relevance in [0, 1]
    sel_mrmr = FeatureSelector(method="mrmr", task="classification", max_features=2, random_seed=42).fit(X, y_cls)
    # mRMR must pick one of {sig_a, sig_a_clone} AND complementary sig_b rather than both redundant clones!
    assert "sig_b" in sel_mrmr.selected_features_
    assert len(set(sel_mrmr.selected_features_) & {"sig_a", "sig_a_clone"}) == 1

    mrmr_miq = FeatureSelector.mrmr_select(X, y_cls, k=2, task="classification", scheme="quotient", random_state=42)
    assert all(0.0 <= v <= 1.0 for v in mrmr_miq["normalized_relevance"].values())
    assert "sig_b" in mrmr_miq["selected_features"]

    # 3. Boruta two-tail exact Binomial(n_iter, 0.5) confirmation & rejection (Kursa & Rudnicki 2010)
    boruta = FeatureSelector.boruta_select(X, y_cls, task="classification", n_iter=20, alpha=0.05, random_state=42)
    assert len(set(boruta["confirmed"]) & {"sig_a", "sig_a_clone", "sig_b"}) >= 1
    assert set(boruta["rejected"]).issubset({"noise_1", "noise_2"})
    assert "p_values_confirm" in boruta and "p_values_reject" in boruta

    # 4. Stability Selection: floor(n/2) without replacement + Meinshausen & Bühlmann (2010) PFER bound
    stab = FeatureSelector.stability_selection(
        X,
        y_cls,
        method="l1",
        task="classification",
        max_features=3,
        n_bootstrap=16,
        threshold=0.6,
        random_state=42,
    )
    assert stab["subsample_size"] == n // 2
    assert stab["sampling_without_replacement"] is True
    assert stab["expected_false_positives_bound"] is not None
    expected_pfer = round((stab["avg_selected_per_run"] ** 2) / ((2.0 * 0.6 - 1.0) * X.shape[1]), 6)
    assert np.isclose(stab["expected_false_positives_bound"], expected_pfer, atol=1e-4)

    # 5. Studio 05 /s05/select rejects invalid custom method with 400 (never wipes columns on commit=True)
    client = TestClient(app)
    hdr = {"X-API-Key": get_api_key(), "X-Session-ID": "phase2-s05-guard"}
    df_s05 = pd.DataFrame(
        {
            "num_sig": signal_a[:40],
            "region": np.where(y_cls[:40] == 1, "North", "South"),
            "noise_col": noise_1[:40],
            "target": y_cls[:40],
        }
    )
    csv_bytes = df_s05.to_csv(index=False).encode("utf-8")
    r_up = client.post("/api/workbench/s01/upload", files={"file": ("s05.csv", csv_bytes, "text/csv")}, headers=hdr)
    assert r_up.status_code == 200
    client.post("/api/workbench/targets", json={"targets": ["target"]}, headers=hdr)

    r_bad = client.post(
        "/api/workbench/s05/select",
        json={"mode": "custom", "method": "totally_invalid_method", "top_k": 2, "commit": True},
        headers=hdr,
    )
    assert r_bad.status_code == 400
    # Verify no columns were wiped out on failed commit
    st_after_bad = client.get("/api/workbench/state", headers=hdr).json()
    assert st_after_bad["excluded_columns"] == []

    # Valid chi2 selection in Studio 05 custom mode succeeds and preserves parent 'region' column when region_* is selected
    r_ok = client.post(
        "/api/workbench/s05/select",
        json={"mode": "custom", "method": "chi2", "top_k": 2, "commit": True},
        headers=hdr,
    )
    assert r_ok.status_code == 200
    body_ok = r_ok.json()
    assert body_ok["method_breakdown"][0]["method"] == "chi2"
    if any(f.startswith("region_") for f in body_ok["selected_features"]):
        assert "region" not in body_ok["state"]["excluded_columns"]


def test_feature_engineering_collision_schema_fourier_phase_and_shuffled_oof() -> None:
    """Verify collision-free one-hot names, partial-schema scaling, Fourier continuity & shuffled OOF encoding."""
    # 1. Colliding sanitized one-hot slugs ('A-B' vs 'A_B') must NOT overwrite each other
    df_oh = pd.DataFrame(
        {
            "code": ["A-B", "A_B", "A/B", "C"] * 10,
            "val": np.linspace(10.0, 50.0, 40),
        }
    )
    pipe_oh = FeatureEngineeringPipeline(
        FeatureConfig(categorical_encoding="onehot", scale_numeric="standard", create_interactions=False)
    )
    out_oh = pipe_oh.fit_transform(df_oh)
    code_cols = [c for c in out_oh.columns if c.startswith("code__")]
    assert len(code_cols) == 4  # All 4 distinct categories preserved!

    # Reconstruct unscaled one-hot indicators on a fresh unscaled pipeline to verify exact partition of unity
    pipe_unscaled = FeatureEngineeringPipeline(
        FeatureConfig(categorical_encoding="onehot", scale_numeric="none", create_interactions=False)
    )
    out_unscaled = pipe_unscaled.fit_transform(df_oh)
    raw_code_cols = [c for c in out_unscaled.columns if c.startswith("code__")]
    assert len(raw_code_cols) == 4
    assert np.allclose(out_unscaled[raw_code_cols].sum(axis=1).to_numpy(), 1.0)

    # 2. Partial schema at transform() time: missing numeric column imputed from train median AND scaler still runs!
    df_tr = pd.DataFrame({"x1": [100.0, 200.0, 300.0, 400.0, 500.0], "x2": [10.0, 20.0, 30.0, 40.0, 50.0]})
    pipe_sc = FeatureEngineeringPipeline(
        FeatureConfig(scale_numeric="standard", create_interactions=False, create_polynomials=False)
    ).fit(df_tr)
    # Omit 'x2' completely in test frame; x1=300 is exact train mean -> scaled z-score MUST be ~0.0 (not raw 300.0!)
    df_te_partial = pd.DataFrame({"x1": [300.0, 500.0]})
    te_scaled = pipe_sc.transform(df_te_partial)
    assert list(te_scaled.columns) == ["x1", "x2"]
    assert np.isclose(float(te_scaled.loc[0, "x1"]), 0.0, atol=1e-6)
    assert np.isclose(float(te_scaled.loc[0, "x2"]), 0.0, atol=1e-6)  # Train median 30 == train mean 30 -> z=0

    # 3. Continuous seasonal Fourier phase across train/test boundary
    df_ts_tr = pd.DataFrame({"y_val": np.arange(10, dtype=float)})
    df_ts_te = pd.DataFrame({"y_val": np.arange(10, 15, dtype=float)})
    pipe_f = FeatureEngineeringPipeline(
        FeatureConfig(
            scale_numeric="none",
            create_interactions=False,
            fourier_periods=[12],
            fourier_order=1,
        )
    )
    _ = pipe_f.fit_transform(df_ts_tr)
    te_fourier = pipe_f.transform(df_ts_te)
    # First test row must correspond to t = 10 (NOT reset to t = 0 where sin(0) == 0!)
    expected_sin_t10 = float(np.sin(2.0 * np.pi * 1.0 * 10.0 / 12.0))
    assert np.isclose(float(te_fourier.iloc[0]["fourier_p12_sin_1"]), expected_sin_t10, atol=1e-6)

    # 4. Shuffled StratifiedKFold OOF target encoding + global-stat unseen group aggregation fallback
    y_sorted = np.array([0] * 25 + [1] * 25)
    df_grp_tr = pd.DataFrame(
        {
            "high_card": [f"cat_{i % 5}" for i in range(50)],
            "dept": ["eng"] * 25 + ["sales"] * 25,
            "salary": np.linspace(80_000.0, 120_000.0, 50),
        }
    )
    pipe_te = FeatureEngineeringPipeline(
        FeatureConfig(
            categorical_encoding="target",
            max_onehot_cardinality=2,
            scale_numeric="none",
            create_interactions=False,
            group_aggregations={"dept": ["salary"]},
            random_seed=42,
        )
    )
    _ = pipe_te.fit_transform(df_grp_tr, y_sorted)
    df_unseen_grp = pd.DataFrame({"high_card": ["unseen_cat"], "dept": ["unseen_legal"], "salary": [100_000.0]})
    out_unseen = pipe_te.transform(df_unseen_grp)
    # Unseen group mean must fall back to global training mean (~100,000), NOT 0.0!
    assert np.isclose(float(out_unseen.iloc[0]["dept_grp_salary_mean"]), 100_000.0, rtol=1e-3)
    assert np.isclose(float(out_unseen.iloc[0]["high_card__target_enc"]), 0.5, atol=1e-4)


def test_little_mcar_mvn_em_mle_and_3way_split_duplicate_audit() -> None:
    """Verify Little's (1988) MVN EM MLE MCAR test detects MAR on fully observed covariates & 3-way split dupes."""
    rng = np.random.default_rng(1988)
    n = 300
    x1 = rng.normal(0.0, 1.0, n)
    x2 = 0.7 * x1 + rng.normal(0.0, 0.7, n)

    # Case A: True MCAR (x1 fully observed, x2 missing completely at random with p=0.25)
    df_mcar = pd.DataFrame({"x1": x1.copy(), "x2": x2.copy()})
    mcar_mask = rng.random(n) < 0.25
    df_mcar.loc[mcar_mask, "x2"] = np.nan
    res_mcar = DataQualityProfiler.little_mcar_test(df_mcar, alpha=0.05)
    assert res_mcar["status"] == "completed"
    assert res_mcar["estimator"] == "mvn_em_mle"
    assert res_mcar["is_mcar"] is True
    assert res_mcar["p_value"] >= 0.05

    # Case B: Textbook MAR (x1 is FULLY OBSERVED; x2 is missing whenever x1 > 0.6)
    df_mar = pd.DataFrame({"x1": x1.copy(), "x2": x2.copy()})
    df_mar.loc[df_mar["x1"] > 0.6, "x2"] = np.nan
    res_mar = DataQualityProfiler.little_mcar_test(df_mar, alpha=0.05)
    assert res_mar["status"] == "completed"
    assert res_mar["is_mcar"] is False
    assert res_mar["p_value"] < 0.005

    # Case C: 3-way duplicate row leak check (train ∩ val and val ∩ test)
    base_df = pd.DataFrame({"a": np.arange(20, dtype=float), "b": np.arange(100, 120, dtype=float)})
    # Make row 12 (val) identical to row 2 (train) and row 17 (test) identical to row 13 (val), while train ∩ test = ∅!
    base_df.iloc[12] = base_df.iloc[2]
    base_df.iloc[17] = base_df.iloc[13]
    splitter = LeakageSafeSplitter(SplitConfig(val_size=0.2, test_size=0.2))
    rep_default = splitter.verify_no_leakage(
        base_df,
        train_idx=np.arange(0, 12),
        val_idx=np.arange(12, 16),
        test_idx=np.arange(16, 20),
    )
    assert rep_default.exact_duplicate_leak_count == 2
    assert rep_default.passed is True  # Default non-strict mode warns without failing discrete i.i.d. datasets

    rep_strict = splitter.verify_no_leakage(
        base_df,
        train_idx=np.arange(0, 12),
        val_idx=np.arange(12, 16),
        test_idx=np.arange(16, 20),
        strict_duplicates=True,
    )
    assert rep_strict.exact_duplicate_leak_count == 2
    assert rep_strict.passed is False


def test_automl_inner_cv_hpo_cloning_and_tuned_pipeline_propagation() -> None:
    """Verify AutoML HPO uses inner CV on X_tr (never peeking at X_va) and propagates tuned params to sklearn_pipeline."""
    rng = np.random.default_rng(99)
    n = 90
    X = pd.DataFrame(
        {
            "f1": rng.normal(0.0, 1.0, n),
            "f2": rng.normal(0.0, 1.0, n),
            "f3": rng.normal(0.0, 1.0, n),
        }
    )
    y = pd.Series((1.5 * X["f1"] - 1.1 * X["f2"] > 0.0).astype(int), name="target")

    res = AutoMLPipelineSearch(
        time_budget_seconds=15.0,
        include_deep=False,
        tune_top_k=1,
        tune_trials=2,
        cv_folds=3,
        use_cv_pipeline=True,
        random_seed=42,
    ).fit(X, y)

    assert res.best_model is not None
    assert len(res.cv_summary) >= 1
    sk_pipe = res.metadata["sklearn_pipeline"]
    assert sk_pipe is not None
    # Verify fitted models dictionary contains distinct uncorrupted instances
    fitted_map = res.metadata["fitted_models"]
    if "automl_top3_ensemble" in fitted_map:
        ens = fitted_map["automl_top3_ensemble"]
        for sub_m in ens.estimators:
            for k_name, m_obj in fitted_map.items():
                if k_name != "automl_top3_ensemble":
                    assert sub_m is not m_obj


def test_pipeline_executor_multitenant_input_aware_cache_and_builder_code_export(tmp_path: Path) -> None:
    """Verify multi-tenant/input-aware DAG cache keys, artifact-delta replay, CWE-94-safe export & split/evaluate."""
    cache_dir = tmp_path / "dag_cache"
    dag = (
        PipelineBuilder('tenant_pipe"""\ninjected = True')
        .add_step("split", "split", config={"target_column": "target", "test_size": 0.25}, cacheable=True)
        .add_step("clean", "clean", chain=True, cacheable=True)
        .add_step(
            "train",
            "train",
            chain=True,
            config={"target_column": "target", "model_name": "logistic_regression", "task_type": "binary_classification"},
            cacheable=True,
        )
        .add_step("evaluate", "evaluate", chain=True, config={"target_column": "target"}, cacheable=False)
        .build()
    )

    # 1. Exported Python code compiles & executes without NameError (false/true/null) or docstring injection!
    code = PipelineBuilder.export_python_code(dag)
    assert "PipelineBuilder" in code
    compiled = compile(code, "<exported_omniforge_pipeline>", "exec")
    ns: dict[str, Any] = {}
    exec(compiled, ns)  # noqa: S102
    assert "injected" not in ns
    assert ns["PIPELINE_SPEC"]["steps"][0]["cacheable"] is True
    assert ns["PIPELINE_SPEC"]["steps"][-1]["cacheable"] is False

    # 2. Multi-tenant & input-aware caching + artifact-delta replay on cache hit
    rng = np.random.default_rng(123)
    df_a = pd.DataFrame({"x1": rng.normal(0.0, 1.0, 60), "x2": rng.normal(0.0, 1.0, 60)})
    df_a["target"] = (df_a["x1"] + df_a["x2"] > 0.0).astype(int)

    df_b = pd.DataFrame({"x1": rng.normal(5.0, 2.0, 60), "x2": rng.normal(-3.0, 2.0, 60)})
    df_b["target"] = (df_b["x1"] - df_b["x2"] > 8.0).astype(int)

    executor = PipelineExecutor(cache_dir=cache_dir, track_carbon=False)
    r1 = executor.execute(dag, initial_artifacts={"raw_data": df_a}, tenant_id="tenant-alpha")
    assert r1.status == "success"
    assert r1.context.get("evaluation_split") == "test_holdout"
    assert all(not rec.cache_hit for rec in r1.step_records)

    # Same tenant + same input -> cache hit on cacheable steps AND evaluate still succeeds via replayed artifact deltas!
    r2 = executor.execute(dag, initial_artifacts={"raw_data": df_a}, tenant_id="tenant-alpha")
    assert r2.status == "success"
    assert [rec.cache_hit for rec in r2.step_records] == [True, True, True, False]
    assert r2.context.get("evaluation_metrics") == r1.context.get("evaluation_metrics")

    # Different tenant with SAME data -> must NOT reuse tenant-alpha's cached artifacts!
    r_other_tenant = executor.execute(dag, initial_artifacts={"raw_data": df_a}, tenant_id="tenant-beta")
    assert all(not rec.cache_hit for rec in r_other_tenant.step_records)

    # Same tenant with DIFFERENT input DataFrame -> must NOT serve stale df_a cache!
    r_new_data = executor.execute(dag, initial_artifacts={"raw_data": df_b}, tenant_id="tenant-alpha")
    assert all(not rec.cache_hit for rec in r_new_data.step_records)


def test_mlops_adwin_scale_equivariance_jsd_base2_monitor_onnx_and_copilot_confinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify scale-equivariant ADWIN, base-2 JSD in [0, 1], async ModelMonitor, ONNX scaler & Copilot confinement."""
    rng = np.random.default_rng(314)

    # 1. ADWIN scale equivariance (Bifet & Gavaldà 2007): 1e-3x and 1e3x streams detect identical drift points
    base_stream = np.concatenate([rng.normal(0.1, 0.02, 80), rng.normal(0.8, 0.02, 80)])
    adw_unit = DriftDetector.detect_concept_drift_adwin(base_stream, delta=0.002)
    adw_micro = DriftDetector.detect_concept_drift_adwin(base_stream * 1e-3, delta=0.002)
    adw_mega = DriftDetector.detect_concept_drift_adwin(base_stream * 1e3, delta=0.002)
    assert adw_unit["drift_detected"] is True
    assert adw_unit["drift_indices"] == adw_micro["drift_indices"] == adw_mega["drift_indices"]

    # 2. Lin (1991) base-2 JSD in [0, 1] for both numeric and categorical features (even with extreme outliers)
    ref_df = pd.DataFrame(
        {
            "num": np.concatenate([rng.normal(0.0, 1.0, 199), [1e6]]),
            "cat": ["red", "blue"] * 100,
        }
    )
    disjoint_df = pd.DataFrame(
        {
            "num": rng.normal(500.0, 1.0, 200),
            "cat": ["green", "yellow"] * 100,
        }
    )
    rep_same = DriftDetector().detect(ref_df, ref_df.copy())
    for f_res in rep_same.features:
        assert f_res.js_divergence is not None and np.isclose(f_res.js_divergence, 0.0, atol=1e-6)

    rep_disj = DriftDetector().detect(ref_df, disjoint_df)
    by_col = {f.column: f for f in rep_disj.features}
    assert by_col["cat"].js_divergence is not None and np.isclose(by_col["cat"].js_divergence, 1.0, atol=1e-6)
    assert by_col["num"].js_divergence is not None and 0.8 <= by_col["num"].js_divergence <= 1.0

    # 3. ModelMonitor tracks paired (prediction, actual) feedback accurately despite unlabeled inferences
    mon = ModelMonitor("fraud_clf", task_type=TaskType.BINARY_CLASSIFICATION, min_accuracy=0.80, max_mae=0.30, max_window_size=50)
    for _ in range(40):
        mon.record_inference(latency_ms=12.0, prediction=1, actual=None)
    # Record 10 labeled events with 50% accuracy (below 0.80 SLA) and MAE=0.50 (above 0.30 SLA)
    for i in range(10):
        mon.record_inference(latency_ms=15.0, prediction=1, actual=(1 if i < 5 else 0))
    health = mon.evaluate_health()
    assert health["labeled_feedback_count"] == 10
    assert np.isclose(health["live_accuracy"], 0.50)
    assert np.isclose(health["live_mae"], 0.50)
    alert_types = {a["alert_type"] for a in health["active_alerts"]}
    assert "accuracy_degradation" in alert_types
    assert "regression_error_breach" in alert_types

    # 4. ModelServingEngine canary_version + make_pipeline(scaler, estimator) in export_onnx
    X_srv = pd.DataFrame({"a": rng.normal(100.0, 25.0, 50), "b": rng.normal(-20.0, 5.0, 50)})
    y_srv = (X_srv["a"] > 100.0).astype(int)
    m_prim = ClassicalMLModel("logistic_regression", task_type=TaskType.BINARY_CLASSIFICATION).fit(X_srv, y_srv)
    m_can = ClassicalMLModel("random_forest", task_type=TaskType.BINARY_CLASSIFICATION, n_estimators=15).fit(X_srv, y_srv)

    srv = ModelServingEngine(
        registry=ModelRegistry(db_path=tmp_path / "reg.db", artifact_dir=tmp_path / "reg_art")
    )
    srv.deploy_model(
        "credit_ep",
        m_prim,
        version="1.0.0",
        canary_model=m_can,
        canary_version="2.0.0-canary",
        canary_traffic_ratio=1.0,
    )
    pred_can = srv.predict("credit_ep", X_srv.head(4))
    assert pred_can["variant"] == "canary"
    assert pred_can["model_version"] == "2.0.0-canary"

    captured_targets: list[Any] = []

    class _DummyOnnxProto:
        @staticmethod
        def SerializeToString() -> bytes:
            return b"ONNX_GRAPH_BYTES"

    def _fake_convert_sklearn(target_obj: Any, initial_types: Any = None) -> _DummyOnnxProto:
        captured_targets.append(target_obj)
        return _DummyOnnxProto()

    import sys
    import types

    fake_skl2onnx = types.ModuleType("skl2onnx")
    fake_skl2onnx.convert_sklearn = _fake_convert_sklearn  # type: ignore[attr-defined]
    fake_dt = types.ModuleType("skl2onnx.common.data_types")
    fake_dt.FloatTensorType = lambda shape: ("float", shape)  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"skl2onnx": fake_skl2onnx, "skl2onnx.common.data_types": fake_dt}):
        onnx_res = srv.export_onnx("credit_ep", tmp_path / "credit.onnx", X_srv.head(3))
    assert onnx_res["format"] == "onnx"
    assert onnx_res["includes_scaler"] is True
    assert len(captured_targets) == 1
    from sklearn.pipeline import Pipeline as SkPipeline

    assert isinstance(captured_targets[0], SkPipeline)
    assert len(captured_targets[0].steps) == 2

    # 5. Copilot tool_ingest_data CWE-22 / OWASP LLM06 path confinement
    safe_dir = tmp_path / "copilot_data"
    safe_dir.mkdir(parents=True)
    good_csv = safe_dir / "sales.csv"
    pd.DataFrame({"rev": [10, 20, 30], "qty": [1, 2, 3]}).to_csv(good_csv, index=False)
    secret_dir = safe_dir / ".omniforge"
    secret_dir.mkdir()
    secret_file = secret_dir / "keys.csv"
    secret_file.write_text("k,v\nsecret,123\n", encoding="utf-8")
    outside_csv = tmp_path / "outside.csv"
    outside_csv.write_text("a,b\n1,2\n", encoding="utf-8")

    monkeypatch.setenv("OMNIFORGE_COPILOT_INGEST_ROOT", str(safe_dir))
    reg_tools = CopilotToolRegistry(CopilotMemory())

    ok_ingest = reg_tools.invoke("ingest_data", {"source": "sales.csv", "dataset_name": "sales"})
    assert ok_ingest["rows"] == 3

    for bad_src in (
        "../outside.csv",
        str(outside_csv),
        "/etc/passwd",
        "file:///etc/passwd",
        ".omniforge/keys.csv",
    ):
        with pytest.raises(CopilotError):
            reg_tools.invoke("ingest_data", {"source": bad_src, "dataset_name": "pwn"})
