"""Audit regressions for clustered inference and keyed reference evidence."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from omniforge.api.auth import get_api_key
from omniforge.api.main import create_app
from omniforge.causal.inference import CausalInferenceEngine
from omniforge.core.exceptions import CausalInferenceError


def test_clustered_did_matches_regression_sandwich_without_row_independence() -> None:
    from statsmodels.regression.linear_model import OLS

    records = [
        {"site": site, "group": site % 2, "post": post, "y": 4 * site + obs + post * (site / 3 + 3 * (site % 2))}
        for site in range(8) for post in (0, 1) for obs in range(3)
    ]
    df = pd.DataFrame(records)
    engine = CausalInferenceEngine()
    report = engine.difference_in_differences(df, group_col="group", post_col="post", outcome_col="y", cluster_col="site")
    design = np.column_stack([np.ones(len(df)), df["group"], df["post"], df["group"] * df["post"]])
    oracle = OLS(df["y"], design).fit(cov_type="cluster", cov_kwds={"groups": df["site"], "use_correction": False})
    assert report["did_estimate"] == pytest.approx(oracle.params.iloc[-1])
    assert report["std_error"] == pytest.approx(oracle.bse.iloc[-1] * np.sqrt(8 / 7))
    assert report["degrees_of_freedom"] == 3

    few = df[(df["group"] == 0) | (df["site"] == 1)]
    result = engine.difference_in_differences(few, group_col="group", post_col="post", outcome_col="y", cluster_col="site")
    assert result["inference_status"] == "insufficient_clusters" and result["p_value"] is None
    with pytest.raises(CausalInferenceError, match="one treatment arm"):
        engine.difference_in_differences(df.assign(site=df["site"] // 2), group_col="group", post_col="post",
                                         outcome_col="y", cluster_col="site")

    client = TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": uuid4().hex})
    assert client.post("/api/workbench/s01/upload", files={"file": ("panel.csv", df.to_csv(index=False).encode())}).status_code == 200
    causal = client.post("/api/workbench/s08/causal", json={
        "method": "clustered_did", "treatment_col": "group", "post_col": "post", "outcome_col": "y", "cluster_col": "site",
    })
    assert causal.status_code == 200, causal.text
    assert causal.json()["causal"]["std_error"] == pytest.approx(report["std_error"])


def test_workbench_profiles_trusted_csv_by_key_without_replacing_the_dataset() -> None:
    client = TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": uuid4().hex})
    active = client.post("/api/workbench/s01/upload", files={"file": ("active.csv", b"id,amount\nA,2\nB,3\nC,4\n")})
    assert active.status_code == 200
    report = client.post(
        "/api/workbench/s01/profile-reference", data={"key_column": "id"},
        files={"file": ("trusted.csv", b"id,amount\nC,4\nA,99\nD,1\n")},
    )
    assert report.status_code == 200, report.text
    payload = report.json()["quality"]
    assert payload["reference_key_columns"] == ["id"]
    assert payload["reference_matched_rows"] == payload["reference_comparisons"] == 2
    assert payload["dimensions"]["accuracy"] == 50
    assert client.get("/api/workbench/state").json()["dataset_name"] == active.json()["state"]["dataset_name"]
    duplicate = client.post(
        "/api/workbench/s01/profile-reference", data={"key_column": "id"},
        files={"file": ("trusted.csv", b"id,amount\nA,4\nA,99\n")},
    )
    assert 400 <= duplicate.status_code < 500
