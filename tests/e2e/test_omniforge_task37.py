"""End-to-End Verification Suite for Task 37 Upgrades:

1. AutoML & Supervisory Exact/Portable External Mathematical Equations
   (Standalone Python, Excel, SQL, MATLAB/C, LaTeX + downloadable .py script execution).
2. Studio 14 Champion vs. Baseline Live Governing Equations Endpoint (`GET /api/workbench/s14/equations`).
3. Studio 14 Signed Root-Cause Attribution (`+Delta` Increase Drivers vs. `-Delta` Decrease Drivers,
   Unexplained Residual separation, Prioritized Remediation Playbook, and Per-Alarm Drilldown).
4. Frozen launcher path cleanliness (`omniforge` instead of legacy `uap`).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from omniforge.api.main import app
from omniforge.evaluation.equations import ModelEquationExtractor
from omniforge.mlops.supervisory_monitor import SupervisoryPredictionMonitor


def _make_known_physics_df(n: int = 240, seed: int = 42) -> pd.DataFrame:
    """Synthetic engineering dataset with exact known positive & negative physical laws:
    power_kw = 50 + 3.0 * temp_c + 1.5 * pressure_bar - 2.0 * cooling_flow + noise.
    """
    rng = np.random.default_rng(seed)
    temp_c = rng.normal(60.0, 5.0, size=n)
    pressure_bar = rng.normal(20.0, 2.5, size=n)
    cooling_flow = rng.normal(30.0, 4.0, size=n)
    noise = rng.normal(0.0, 0.8, size=n)
    y = 50.0 + 3.0 * temp_c + 1.5 * pressure_bar - 2.0 * cooling_flow + noise
    return pd.DataFrame(
        {
            "temp_c": np.round(temp_c, 3),
            "pressure_bar": np.round(pressure_bar, 3),
            "cooling_flow": np.round(cooling_flow, 3),
            "vibration_aux": np.round(rng.normal(2.0, 0.2, size=n), 3),
            "power_kw": np.round(y, 3),
        }
    )


def test_external_equation_formats_and_executable_python_code() -> None:
    """Verify ModelEquationExtractor unwraps Pipeline/Ensemble models and emits executable standalone Python/Excel/SQL/C formulas."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = _make_known_physics_df(260, seed=7)
    features = ["temp_c", "pressure_bar", "cooling_flow", "vibration_aux"]
    X = df[features]
    y = df["power_kw"].to_numpy()

    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("est", HistGradientBoostingRegressor(max_iter=60, max_depth=4, random_state=42)),
        ]
    )
    pipe.fit(X, y)

    extractor = ModelEquationExtractor()
    report = extractor.extract(
        pipe,
        X_reference=X,
        feature_names=features,
        target_name="power_kw",
    )
    d = report.model_dump()

    # Verify all external deployment kits exist and are rich
    assert d["formula_plain"] and "power_kw" in d["formula_plain"]
    assert d["formula_python"] and "def predict_" in d["formula_python"]
    assert d["formula_excel"] and d["formula_excel"].startswith("=")
    assert d["formula_sql"] and "SELECT" in d["formula_sql"] and "AS predicted_" in d["formula_sql"]
    assert d["formula_c_matlab"] and "y_hat_" in d["formula_c_matlab"]
    assert d["surrogate_fidelity_r2"] is not None and d["surrogate_fidelity_r2"] > 0.85
    assert d["surrogate_intercept"] is not None
    assert len(d["surrogate_coefficients"]) >= 3

    # Dynamically parse AND execute the exported standalone Python code with ZERO OmniForge imports!
    ast.parse(d["formula_python"])
    ns: dict[str, object] = {}
    exec(d["formula_python"], ns)  # noqa: S102 - testing self-generated pure math function
    fn_names = [k for k, v in ns.items() if callable(v) and k.startswith("predict_")]
    assert len(fn_names) == 1
    predict_fn = ns[fn_names[0]]

    # Validate numerical fidelity against the real fitted sklearn pipeline
    sample_rows = X.iloc[:15].to_dict(orient="records")
    ext_preds = np.array([float(predict_fn(r)) for r in sample_rows])  # type: ignore[operator]
    pipe_preds = pipe.predict(X.iloc[:15])
    corr = float(np.corrcoef(ext_preds, pipe_preds)[0, 1])
    assert corr > 0.92, f"Standalone Python formula diverged from AutoML pipeline (corr={corr:.4f})"


def test_supervisory_signed_root_cause_increase_and_decrease_attribution() -> None:
    """Verify Studio 14 engine accurately identifies UPWARD (+Delta) vs DOWNWARD (-Delta) root causes."""
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import Ridge

    train_df = _make_known_physics_df(260, seed=11)
    features = ["temp_c", "pressure_bar", "cooling_flow", "vibration_aux"]
    X_tr = train_df[features]
    y_tr = train_df["power_kw"]

    champ = Ridge(alpha=1.0).fit(X_tr, y_tr)
    base_mean = DummyRegressor(strategy="mean").fit(X_tr, y_tr)

    state = {
        "uap_df": train_df,
        "uap_target": "power_kw",
        "uap_targets": ["power_kw"],
        "uap_active_model": champ,
        "uap_active_model_name": "Ridge_Physics_Champion",
        "uap_active_feature_cols": features,
        "uap_active_model_metrics": {"r2": 0.99},
        "uap_baseline_models": {
            "power_kw": {
                "Normal_Operating_Baseline": {
                    "model": base_mean,
                    "feature_cols": features,
                    "metrics": {"r2": 0.0},
                }
            }
        },
    }

    monitor = SupervisoryPredictionMonitor()
    catalog = monitor.catalog_available_models(state)
    base_id = next(m["model_id"] for m in catalog if m["family"] == "Baseline")

    # Scenario A: Inject HUGE UPWARD fault on temp_c (+25 C -> +75 kW increase)
    # and mild downward cooling_flow boost (+10 -> -20 kW decrease offset)
    live_up = train_df.iloc[:40].copy()
    live_up["temp_c"] = live_up["temp_c"] + 25.0
    live_up["cooling_flow"] = live_up["cooling_flow"] + 10.0
    live_up["power_kw"] = (
        50.0 + 3.0 * live_up["temp_c"] + 1.5 * live_up["pressure_bar"] - 2.0 * live_up["cooling_flow"]
    )

    res_up = monitor.evaluate_incoming(
        state,
        live_up,
        model_id="active_champion",
        baseline_model_id=base_id,
        max_pct_error=5.0,
    )
    assert res_up["champion_equation"] is not None
    assert res_up["champion_equation"]["formula_excel"].startswith("=")
    assert res_up["baseline_equation"] is not None

    rca = res_up["root_cause_analysis"]
    assert rca is not None
    assert rca["overall_direction"] == "INCREASE_ABOVE_BASELINE"
    assert rca["mean_actual_vs_baseline_delta"] > 30.0
    assert len(rca["increase_drivers"]) >= 1
    assert len(rca["decrease_drivers"]) >= 1
    # Top upward driver MUST be temp_c (+3 * +25 = +75 kW)!
    assert rca["increase_drivers"][0]["feature"] == "temp_c"
    assert rca["increase_drivers"][0]["signed_impact"] > 40.0
    # Top downward offset driver MUST be cooling_flow (-2 * +10 = -20 kW)!
    assert rca["decrease_drivers"][0]["feature"] == "cooling_flow"
    assert rca["decrease_drivers"][0]["signed_impact"] < -10.0
    assert len(rca["remediation_playbook"]) >= 1
    assert any("temp_c" in str(r.get("variable", "")) for r in rca["remediation_playbook"])

    # Scenario B: Inject HUGE DOWNWARD fault (temp_c drops -20 C -> -60 kW decrease below baseline)
    live_down = train_df.iloc[:40].copy()
    live_down["temp_c"] = live_down["temp_c"] - 20.0
    live_down["power_kw"] = (
        50.0 + 3.0 * live_down["temp_c"] + 1.5 * live_down["pressure_bar"] - 2.0 * live_down["cooling_flow"]
    )
    res_down = monitor.evaluate_incoming(
        state,
        live_down,
        model_id="active_champion",
        baseline_model_id=base_id,
        max_pct_error=5.0,
    )
    rca_down = res_down["root_cause_analysis"]
    assert rca_down["overall_direction"] == "DECREASE_BELOW_BASELINE"
    assert rca_down["mean_actual_vs_baseline_delta"] < -30.0
    assert rca_down["decrease_drivers"][0]["feature"] == "temp_c"
    assert rca_down["decrease_drivers"][0]["signed_impact"] < -40.0


def test_api_s06_export_formula_and_s14_equations_and_root_cause() -> None:
    """End-to-end FastAPI test covering S01 load -> S06 AutoML + .py export -> S14 equations + root-cause evaluation."""
    from omniforge.api.auth import get_api_key

    client = TestClient(app, headers={"X-API-Key": get_api_key()})

    # 1. Load bundled regression dataset
    r_load = client.post("/api/workbench/s01/load-sample", json={"dataset_key": "02_industrial_predictive_maint"})
    assert r_load.status_code == 200, r_load.text

    # 2. Run Studio 06 AutoML
    r_automl = client.post(
        "/api/workbench/s06/automl",
        json={"backend": "native_fast_gbdt", "time_budget_sec": 15, "cv_folds": 2},
    )
    assert r_automl.status_code == 200, r_automl.text
    am = r_automl.json()
    eq = am.get("equation")
    assert eq is not None
    assert eq.get("formula_python") and "def predict_" in eq["formula_python"]
    assert eq.get("formula_excel") and str(eq["formula_excel"]).startswith("=")
    assert eq.get("formula_sql") and "SELECT" in eq["formula_sql"]
    assert eq.get("formula_c_matlab")

    # 3. Download standalone .py deployment script from Studio 06
    r_dl = client.get("/api/workbench/s06/export-formula")
    assert r_dl.status_code == 200, r_dl.text
    py_code = r_dl.text
    assert "Standalone External Deployment Equation" in py_code
    assert "def predict_" in py_code
    ast.parse(py_code)

    # 4. Studio 14 Live Equations Endpoint (before even evaluating!)
    r_eq14 = client.get("/api/workbench/s14/equations")
    assert r_eq14.status_code == 200, r_eq14.text
    eq14 = r_eq14.json()
    assert eq14.get("champion_equation") is not None
    assert eq14["champion_equation"].get("formula_excel", "").startswith("=")

    # 5. Studio 14 Simulated Drift Evaluation -> Must return Champion/Baseline equations + Signed Root-Cause Attribution + Alarm Drilldown
    r_sim = client.post(
        "/api/workbench/s14/evaluate-demo-batch",
        json={"n_rows": 35, "inject_faults": True, "fault_rate": 0.35, "max_pct_error": 8.0},
    )
    assert r_sim.status_code == 200, r_sim.text
    sim = r_sim.json()
    assert sim.get("champion_equation") is not None
    assert sim.get("baseline_equation") is not None
    rca = sim.get("root_cause_analysis")
    assert rca is not None
    assert "all_factor_attributions" in rca and len(rca["all_factor_attributions"]) >= 1
    assert "remediation_playbook" in rca and len(rca["remediation_playbook"]) >= 1
    assert "summary_narrative" in rca and len(rca["summary_narrative"]) > 30
    if sim.get("alarms"):
        first_alarm = sim["alarms"][0]
        assert "deviation_direction" in first_alarm
        assert "recommended_fix" in first_alarm
        assert "factor_breakdown" in first_alarm


def test_launcher_uses_omniforge_package_path() -> None:
    """Verify PyInstaller launcher points to omniforge rather than legacy uap directory."""
    launcher_src = (Path(__file__).resolve().parents[2] / "src" / "omniforge" / "launcher.py").read_text(
        encoding="utf-8"
    )
    assert 'meipass / "omniforge" / rel' in launcher_src
    assert 'meipass / "uap" / rel' not in launcher_src
