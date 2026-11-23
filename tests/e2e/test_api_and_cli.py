"""End-to-End tests for the FastAPI REST API (with JWT Auth & RBAC) and Typer CLI."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from omniforge.api.auth import get_api_key
from omniforge.api.main import create_app
from omniforge.cli.main import cli


def test_fastapi_e2e_lifecycle() -> None:
    app = create_app()
    client = TestClient(app)

    # Health & Version
    r_health = client.get("/health")
    assert r_health.status_code == 200 and r_health.json()["status"] == "healthy"

    # Unauthenticated calls MUST be rejected (HTTP 401)
    assert client.post("/api/v1/auth/token", json={"username": "anon", "role": "admin"}).status_code == 401
    assert client.get("/api/workbench/state").status_code == 401
    assert client.post("/api/v1/data/quality", json={"records": [{"x": 1}]}).status_code == 401

    # Auth Token (requires valid X-API-Key)
    r_tok = client.post(
        "/api/v1/auth/token",
        json={"username": "lead_ds", "role": "admin", "tenant_id": "ent"},
        headers={"X-API-Key": get_api_key()},
    )
    assert r_tok.status_code == 200
    headers = {"Authorization": f"Bearer {r_tok.json()['access_token']}"}

    # Domains list & sample (authenticated)
    r_dom = client.get("/api/v1/domains", headers=headers)
    assert r_dom.status_code == 200 and len(r_dom.json()["domains"]) == 18
    r_samp = client.get("/api/v1/domains/finance/sample?n_samples=60", headers=headers)
    assert r_samp.status_code == 200
    records = r_samp.json()["preview"]

    # Quality & Clean
    r_q = client.post("/api/v1/data/quality", json={"records": records}, headers=headers)
    assert r_q.status_code == 200 and r_q.json()["overall_score"] > 50

    # Train AutoML + Deploy + Predict
    from omniforge.domains import default_domain_registry

    fin_df = default_domain_registry.get("finance").generate_sample_dataset(n_samples=120)
    r_train = client.post(
        "/api/v1/automl/train",
        json={
            "records": fin_df.to_dict(orient="records"),
            "target_column": "default_flag",
            "time_budget_seconds": 15.0,
            "deploy_endpoint": "finance_live",
        },
        headers=headers,
    )
    assert r_train.status_code == 200
    assert r_train.json()["best_model_name"]

    X_pred = fin_df.drop(columns=["default_flag"]).head(3).to_dict(orient="records")
    r_pred = client.post(
        "/api/v1/serving/finance_live/predict",
        json={"records": X_pred, "return_probabilities": True},
        headers=headers,
    )
    assert r_pred.status_code == 200 and r_pred.json()["n_records"] == 3

    # Copilot REST endpoint
    r_cop = client.post("/api/v1/copilot/chat", json={"message": "list industry domains"}, headers=headers)
    assert r_cop.status_code == 200 and "18" in r_cop.json()["reply"]

    # Portfolio Optimization REST endpoint
    r_opt = client.post(
        "/api/v1/optimization/portfolio",
        json={"expected_returns": [0.08, 0.13], "covariance_matrix": [[0.04, 0.01], [0.01, 0.09]]},
        headers=headers,
    )
    assert r_opt.status_code == 200 and r_opt.json()["status"] == "optimal"


def test_typer_cli_e2e(workspace_tmp: Path) -> None:
    runner = CliRunner()
    res_ver = runner.invoke(cli, ["version"])
    assert res_ver.exit_code == 0 and "4.0.0" in res_ver.stdout and "OmniForge" in res_ver.stdout

    res_dom = runner.invoke(cli, ["domains"])
    assert res_dom.exit_code == 0 and "finance" in res_dom.stdout

    sample_csv = workspace_tmp / "fin.csv"
    res_gen = runner.invoke(cli, ["domains", "--sample", "finance", "--out", str(sample_csv), "--rows", "100"])
    assert res_gen.exit_code == 0 and sample_csv.exists()

    res_q = runner.invoke(cli, ["quality", str(sample_csv)])
    assert res_q.exit_code == 0 and "Overall Quality Score" in res_q.stdout

    res_cop = runner.invoke(cli, ["copilot", "profile data quality", "--data", str(sample_csv)])
    assert res_cop.exit_code == 0 and "Quality Score" in res_cop.stdout
