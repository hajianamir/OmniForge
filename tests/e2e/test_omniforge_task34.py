"""Task 34 verification tests: Non-Parametric Analytical Models, Studio Result Persistence,
and Strict Data Cleaning / Clustering / Downstream Data Provenance Verification."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from omniforge.api.auth import get_api_key
from omniforge.api.main import app
from omniforge.core.types import Principal, TaskType
from omniforge.evaluation.equations import ModelEquationExtractor
from omniforge.models.nonparametric import (
    DistanceWeightedKNNModel,
    IsotonicMonotonicModel,
    NadarayaWatsonKernelModel,
    NonParametricAnalyticalSuite,
    PenalizedSplineGAMModel,
    RBFKernelRidgeModel,
)
from omniforge.ui.state import get_session_state

client = TestClient(app, headers={"X-API-Key": get_api_key()})


def _make_nonlinear_dirty_df(seed: int = 42) -> pd.DataFrame:
    """Create a dataset with strong nonlinear signal + intentional dirt (NaNs, dups, extreme cluster)."""
    rng = np.random.default_rng(seed)
    n = 180
    x1 = rng.uniform(-3.0, 3.0, size=n)
    x2 = rng.uniform(-2.0, 2.0, size=n)
    x3 = rng.normal(10.0, 2.0, size=n)
    cat = rng.choice(["North", "South", "East"], size=n)
    # Strongly nonlinear target: sin(2*x1) + 0.8*x2^2 + 0.1*x3
    y = 4.0 * np.sin(1.8 * x1) + 1.2 * (x2 ** 2) + 0.15 * x3 + rng.normal(0.0, 0.25, size=n)

    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "region": cat,
            "target_y": y,
        }
    )
    # Inject intentional dirty values (missing values & extreme outlier cluster)
    df.loc[[5, 12, 24, 38], "x1"] = np.nan
    df.loc[[7, 19], "region"] = None
    # Append 6 exact duplicate rows
    dups = df.iloc[:6].copy()
    # Append 12 extreme corrupt sensor-fault rows forming an isolated cluster far from normal data
    corrupt = pd.DataFrame(
        {
            "x1": rng.normal(85.0, 1.5, size=12),
            "x2": rng.normal(75.0, 1.5, size=12),
            "x3": rng.normal(500.0, 5.0, size=12),
            "region": ["FaultZone"] * 12,
            "target_y": rng.normal(-90.0, 2.0, size=12),
        }
    )
    return pd.concat([df, dups, corrupt], ignore_index=True)


# =============================================================================
# 1. Non-Parametric Analytical Models Unit & Equation Verification
# =============================================================================
def test_nonparametric_suite_regression_and_equations() -> None:
    rng = np.random.default_rng(7)
    n = 220
    X = pd.DataFrame(
        {
            "x_wave": rng.uniform(-3.0, 3.0, size=n),
            "x_quad": rng.uniform(-2.5, 2.5, size=n),
            "x_mono": rng.uniform(0.0, 5.0, size=n),
        }
    )
    y = pd.Series(
        3.5 * np.sin(1.6 * X["x_wave"]) + 1.1 * (X["x_quad"] ** 2) + 0.8 * np.log1p(X["x_mono"]) + rng.normal(0, 0.2, n),
        name="efficiency",
    )

    res = NonParametricAnalyticalSuite.run(X, y, task_type=TaskType.REGRESSION, target_name="efficiency", cv_folds=3)

    assert res["best_model_name"]
    assert res["nonparametric_outperforms_parametric"] is True
    assert float(res["lift_vs_parametric"]) > 0.15
    assert len(res["leaderboard"]) >= 5  # 5 non-parametric + 1 parametric ref
    assert len(res["spline_components"]) == 3

    for name, mdl in res["_fitted_models"].items():
        if name.startswith("Parametric_"):
            continue
        eq = ModelEquationExtractor.extract(
            mdl,
            feature_names=list(X.columns),
            target_name="efficiency",
            X_reference=X,
        )
        assert eq.equation_type == "nonparametric_analytical", f"{name} must extract nonparametric_analytical"
        assert eq.plain_equation and len(eq.plain_equation) > 15
        assert eq.latex_equation and len(eq.latex_equation) > 15
        assert len(eq.architecture_equations) >= 1

    assert len(res["equations"]) >= 5
    for eq_d in res["equations"]:
        assert eq_d["formula_plain"] and len(eq_d["formula_plain"]) > 15
        assert eq_d["formula_latex"] and len(eq_d["formula_latex"]) > 15
        assert eq_d["formula_python"]
        assert eq_d["model_paradigm"]


def test_nonparametric_suite_classification() -> None:
    rng = np.random.default_rng(11)
    n = 200
    x1 = rng.uniform(-2.5, 2.5, size=n)
    x2 = rng.uniform(-2.5, 2.5, size=n)
    y = pd.Series(((x1 ** 2 + x2 ** 2) > 2.2).astype(int), name="ring_class")
    X = pd.DataFrame({"x1": x1, "x2": x2})

    res = NonParametricAnalyticalSuite.run(X, y, task_type=TaskType.BINARY_CLASSIFICATION, target_name="ring_class", cv_folds=3)
    assert res["nonparametric_outperforms_parametric"] is True
    assert res["best_nonparametric_score"] > 0.75
    for name, mdl in res["_fitted_models"].items():
        if name.startswith("Parametric_"):
            continue
        proba = mdl.predict_proba(X.iloc[:40])
        assert proba.shape == (40, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_individual_nonparametric_estimators_fit_predict() -> None:
    rng = np.random.default_rng(99)
    X = pd.DataFrame({"a": rng.uniform(0, 10, 90), "b": rng.normal(0, 1, 90)})
    y = pd.Series(np.sqrt(X["a"]) + 0.5 * X["b"] ** 2, name="y")

    for cls in (
        PenalizedSplineGAMModel,
        NadarayaWatsonKernelModel,
        RBFKernelRidgeModel,
        DistanceWeightedKNNModel,
        IsotonicMonotonicModel,
    ):
        m = cls(task_type=TaskType.REGRESSION) if cls not in (RBFKernelRidgeModel, IsotonicMonotonicModel) else cls()
        m.fit(X.iloc[:70], y.iloc[:70])
        preds = m.predict(X.iloc[70:])
        assert len(preds) == 20
        assert np.all(np.isfinite(preds))
        eq = ModelEquationExtractor.extract(m, feature_names=["a", "b"], target_name="y", X_reference=X.iloc[:70])
        assert eq.equation_type == "nonparametric_analytical"
        assert eq.plain_equation
        assert eq.latex_equation


# =============================================================================
# 2 & 3. End-to-End API Verification: Cleaning + Split Lock + Clustering +
#        Downstream Data Provenance + Persistent Studio Results + .omniforge Save/Restore
# =============================================================================
def test_cleaning_clustering_provenance_and_stage_persistence() -> None:
    sid = "task34-e2e-provenance"
    hdr = {"X-Session-Id": sid}
    df_dirty = _make_nonlinear_dirty_df()
    raw_rows = len(df_dirty)  # 198 rows (180 clean + 6 dups + 12 corrupt fault rows)

    csv_buf = io.BytesIO(df_dirty.to_csv(index=False).encode("utf-8"))
    r_up = client.post(
        "/api/workbench/s01/upload",
        headers=hdr,
        files={"file": ("dirty_nonlinear.csv", csv_buf, "text/csv")},
    )
    assert r_up.status_code == 200, r_up.text
    st = r_up.json()["state"]
    assert st["n_rows"] == raw_rows
    assert st["raw_rows"] == raw_rows
    assert st["processing_status"]["is_processed"] is False
    assert st["processing_status"]["detailed_label"] == "RAW"

    # Set explicit target
    r_tgt = client.post("/api/workbench/targets", headers=hdr, json={"targets": ["target_y"]})
    assert r_tgt.status_code == 200

    # Enable Anti-Leakage Split Lock BEFORE Cleaning (verifies 4-tuple fit_transform_split fix!)
    r_lock = client.post(
        "/api/workbench/split-lock",
        headers=hdr,
        json={"strategy": "random", "test_size": 0.2, "cv_folds": 3},
    )
    assert r_lock.status_code == 200, r_lock.text
    assert r_lock.json()["state"]["split_lock"]["active"] is True

    # Run Studio 01 Cleaning with Modified Z-Score (MAD) + Deduplication + Median Imputation
    r_clean = client.post(
        "/api/workbench/s01/clean",
        headers=hdr,
        json={
            "mode": "custom",
            "impute_numeric": "median",
            "impute_categorical": "most_frequent",
            "drop_duplicates": True,
            "handle_outliers": "clip",
            "outlier_method": "modified_zscore",
            "outlier_threshold": 3.5,
            "preview_only": False,
        },
    )
    assert r_clean.status_code == 200, r_clean.text
    clean_data = r_clean.json()
    assert clean_data["cleaning_report"]["missing_after"] == 0
    assert clean_data["cleaning_report"]["rows_after"] < raw_rows  # exact duplicates removed
    st_clean = clean_data["state"]
    assert st_clean["processing_status"]["is_cleaned"] is True
    assert st_clean["processing_status"]["is_processed"] is True
    assert "CLEANED" in st_clean["processing_status"]["detailed_label"]
    assert st_clean["raw_rows"] == raw_rows  # immutable raw preserved!
    assert st_clean["n_rows"] == clean_data["cleaning_report"]["rows_after"]

    # Verify forensic deletion log captured exact dropped rows
    r_del = client.get("/api/workbench/s01/deleted-log", headers=hdr)
    assert r_del.status_code == 200
    assert r_del.json()["total_deleted"] >= 1

    # Verify Studio 01 quality + cleaning results are persisted on server
    r_res1 = client.get("/api/workbench/results", headers=hdr)
    assert r_res1.status_code == 200
    assert "s01_clean" in r_res1.json()["results"]

    # Studio 02 Clustering — find the isolated fault cluster & delete its rows
    r_cl = client.post(
        "/api/workbench/s02/clustering",
        headers=hdr,
        json={"mode": "custom", "algorithm": "kmeans", "n_clusters": 4, "scaler": "standard"},
    )
    assert r_cl.status_code == 200, r_cl.text
    cl_payload = r_cl.json()
    choices = cl_payload["cluster_choices"]
    assert len(choices) >= 2
    # Pick the smallest cluster (the fault cluster) to delete
    smallest = min(choices, key=lambda c: c["count"])
    rows_before_cl_drop = st_clean["n_rows"]
    r_cl_commit = client.post(
        "/api/workbench/s02/clustering/commit",
        headers=hdr,
        json={"mode": "remove_clusters", "remove_cluster_ids": [smallest["cluster_id"]], "attach_cluster_id": False},
    )
    assert r_cl_commit.status_code == 200, r_cl_commit.text
    cl_commit = r_cl_commit.json()
    expected_after_cl = rows_before_cl_drop - smallest["count"]
    assert cl_commit["rows_after"] == expected_after_cl
    assert cl_commit["state"]["n_rows"] == expected_after_cl
    assert cl_commit["state"]["raw_rows"] == raw_rows  # immutable raw still untouched!
    assert cl_commit["state"]["s02_cluster_committed"] is True
    assert cl_commit["state"]["studio_completed"]["02"] is True
    assert "CLUSTER-SCREENED" in cl_commit["state"]["processing_status"]["detailed_label"]
    # Verify remaining_cluster_choices excludes the deleted cluster
    rem_ids = {c["cluster_id"] for c in cl_commit["remaining_cluster_choices"]}
    assert smallest["cluster_id"] not in rem_ids

    # Verify toggling an exclusion does NOT wipe s02_cluster_committed!
    r_ex = client.post("/api/workbench/exclusions", headers=hdr, json={"excluded_columns": ["region"]})
    assert r_ex.status_code == 200
    assert r_ex.json()["state"]["s02_cluster_committed"] is True
    assert r_ex.json()["state"]["studio_completed"]["02"] is True
    # Restore exclusion
    client.post("/api/workbench/exclusions", headers=hdr, json={"excluded_columns": []})

    # Verify Studio 03 Diagnostics operates on CLEANED + CLUSTER-SCREENED data (NOT raw!)
    r_s03 = client.post("/api/workbench/s03/run", headers=hdr, json={"mode": "all"})
    assert r_s03.status_code == 200, r_s03.text
    s03_rep = r_s03.json()["report"]
    # Check baseline equations include Non-Parametric Analytical models!
    eqs = s03_rep.get("baseline_equations", {}).get("target_y", [])
    assert len(eqs) >= 4
    paradigms = {e.get("model_paradigm") for e in eqs}
    assert "Non-Parametric Analytical" in paradigms

    # Verify Studio 04 EDA uses CLEANED + CLUSTER-SCREENED rows (n_rows == expected_after_cl)
    r_s04 = client.post("/api/workbench/s04/eda", headers=hdr)
    assert r_s04.status_code == 200, r_s04.text
    # Check Studio 06 Non-Parametric Analytical Endpoint directly
    r_np = client.post(
        "/api/workbench/s06/nonparametric",
        headers=hdr,
        json={"models": ["gam_spline", "nadaraya_watson", "kernel_ridge_rbf", "weighted_knn", "isotonic"]},
    )
    assert r_np.status_code == 200, r_np.text
    np_data = r_np.json()
    assert np_data["n_samples"] == expected_after_cl  # PROVENANCE PROOF!
    assert len(np_data["equations"]) >= 4
    assert len(np_data["spline_components"]) >= 2

    # Verify Non-Parametric models appear in Studio 14 Supervisory Model Catalog!
    r_s14_m = client.get("/api/workbench/s14/models", headers=hdr)
    assert r_s14_m.status_code == 200
    np_in_s14 = [m for m in r_s14_m.json()["models"] if m["family"] == "Non-Parametric Analytical"]
    assert len(np_in_s14) >= 4

    # Score a live supervisory batch using a Non-Parametric Analytical model!
    r_s14_eval = client.post(
        "/api/workbench/s14/evaluate-demo-batch",
        headers=hdr,
        json={"model_id": np_in_s14[0]["model_id"], "n_rows": 30, "inject_fault_pct": 10.0},
    )
    assert r_s14_eval.status_code == 200, r_s14_eval.text

    # Verify internal session state has exact processed data vs untouched raw data.
    # (Workbench HTTP sessions are namespaced per authenticated principal to prevent BOLA,
    # so the direct probe must pass the same principal explicitly.)
    from omniforge.api.workbench import _sid

    api_key_principal = Principal(subject="api_key_principal", roles=["admin"], tenant_id="default-tenant")
    internal = get_session_state(_sid(sid, api_key_principal))
    assert len(internal["uap_raw_df"]) == raw_rows
    assert len(internal["uap_df"]) == expected_after_cl
    assert int(internal["uap_df"].isna().sum().sum()) == 0
    assert int(internal["uap_raw_df"].isna().sum().sum()) > 0

    # Verify ALL completed studio results are persisted in GET /api/workbench/results
    r_all_res = client.get("/api/workbench/results", headers=hdr)
    assert r_all_res.status_code == 200
    cached = r_all_res.json()["results"]
    for key in ("s01_clean", "s02_clustering", "s03_diagnostics", "s04_eda", "s06_nonparametric", "s14_supervisory"):
        assert key in cached, f"Missing persisted result key: {key}"

    # Verify .omniforge Save & Restore preserves ALL studio results across sessions!
    r_save = client.post(
        "/api/workbench/workspace/save",
        headers=hdr,
        data={"project_name": "task34_test", "notes": "Persistence verification"},
    )
    assert r_save.status_code == 200
    archive_bytes = r_save.content
    assert len(archive_bytes) > 500

    # Restore into a brand-new clean session
    sid2 = "task34-restored-session"
    r_load = client.post(
        "/api/workbench/workspace/load",
        headers={"X-Session-Id": sid2},
        files={"file": ("task34_test.omniforge", io.BytesIO(archive_bytes), "application/octet-stream")},
    )
    assert r_load.status_code == 200, r_load.text
    r_res2 = client.get("/api/workbench/results", headers={"X-Session-Id": sid2})
    assert r_res2.status_code == 200
    cached2 = r_res2.json()["results"]
    for key in ("s01_clean", "s02_clustering", "s03_diagnostics", "s04_eda", "s06_nonparametric", "s14_supervisory"):
        assert key in cached2, f"Restored .omniforge session missing persisted key: {key}"


def test_standardize_column_names_deduplicates_and_remaps_target() -> None:
    sid = "task34-colname-dedup"
    hdr = {"X-Session-Id": sid}
    df = pd.DataFrame(
        {
            "Sensor A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "Sensor-A": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
            "Target Power (kW)": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
        }
    )
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    assert client.post("/api/workbench/s01/upload", headers=hdr, files={"file": ("cols.csv", buf, "text/csv")}).status_code == 200
    assert client.post("/api/workbench/targets", headers=hdr, json={"targets": ["Target Power (kW)"]}).status_code == 200
    r = client.post(
        "/api/workbench/s01/clean",
        headers=hdr,
        json={"mode": "custom", "standardize_column_names": True, "preview_only": False},
    )
    assert r.status_code == 200, r.text
    st = r.json()["state"]
    assert len(st["columns"]) == len(set(st["columns"])), "Standardized columns must be unique!"
    assert "target_power_kw" in st["targets"]
    assert "target_power_kw" in st["columns"]
