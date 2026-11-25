"""Pass 4 regression tests: CWE-1236 exporter sanitization, CWE-94 microservice bundle safety, Studio 12 narratives, CLI validation, and Docker/SPA invariants."""

from __future__ import annotations

import ast
import csv
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from omniforge.api.auth import get_api_key
from omniforge.api.main import create_app
from omniforge.cli.main import cli
from omniforge.launcher import run_environment_check
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.baselines import LogisticBaseline
from omniforge.reporting.exporters import ReportExporters, sanitize_spreadsheet_value


def test_cwe1236_spreadsheet_formula_injection_sanitized_in_csv_bundle_and_xlsx(tmp_path: Path) -> None:
    """CSV, CSV-bundle ZIP, and XLSX exports must neutralize formula prefixes while preserving signed numbers."""
    assert sanitize_spreadsheet_value("=CMD|'/C calc'!A0") == "'=CMD|'/C calc'!A0"
    assert sanitize_spreadsheet_value("+SUM(A1:A2)") == "'+SUM(A1:A2)"
    assert sanitize_spreadsheet_value("-2+3+cmd") == "'-2+3+cmd"
    assert sanitize_spreadsheet_value("@SUM(1+1)") == "'@SUM(1+1)"
    assert sanitize_spreadsheet_value("\t=1+1") == "'\t=1+1"
    # Legitimate signed floats must remain untouched
    assert sanitize_spreadsheet_value("-42.5") == "-42.5"
    assert sanitize_spreadsheet_value("+3.14e-2") == "+3.14e-2"
    assert sanitize_spreadsheet_value("Normal text") == "Normal text"

    payload = {
        "title": "=HYPERLINK(\"http://evil.invalid\",\"Report\")",
        "domain": "finance",
        "author": "@attacker",
        "generated_at": "2026-11-21T00:00:00Z",
        "kpis": {"=InjectedKPI": "=1+1", "SignedDelta": "-12.34"},
        "sections": [
            {
                "section_id": "executive_summary",
                "heading": "+DangerHeading",
                "body": "=SUM(A1:B2)",
                "bullets": ["-malicious_bullet"],
            }
        ],
        "tables": {
            "Evil_Table": [
                {"col": "=CMD|'/C calc'!A0", "num_str": "-99.5"},
            ]
        },
        "recommendations": ["@Exfiltrate()"],
    }

    csv_path = ReportExporters.to_csv(payload, tmp_path / "out.csv")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    flat_cells = [val for row in rows for val in row.values() if val]
    assert not any(c.startswith(("=CMD", "=HYPERLINK", "=1+1", "=SUM", "+Danger", "@Exfiltrate")) for c in flat_cells)
    assert any(c.startswith("'=CMD") for c in flat_cells)
    assert any(c == "-12.34" for c in flat_cells)

    zip_path = ReportExporters.to_csv_bundle(payload, tmp_path / "bundle.zip")
    with zipfile.ZipFile(zip_path, "r") as zf:
        evil_csv = zf.read("05_evil_table.csv").decode("utf-8-sig")
        assert "'=CMD|'/C calc'!A0" in evil_csv
        assert "-99.5" in evil_csv

    xlsx_path = ReportExporters.to_excel(payload, tmp_path / "out.xlsx")
    wb = openpyxl.load_workbook(xlsx_path, data_only=False)
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            for cell_val in row:
                if isinstance(cell_val, str):
                    assert not cell_val.startswith(("=CMD", "=HYPERLINK", "=1+1", "=SUM", "+Danger", "@Exfiltrate"))


def test_microservice_bundle_escapes_code_injection_and_copies_hmac_sidecar(tmp_path: Path) -> None:
    """export_microservice_bundle must prevent CWE-94 code injection and copy the .meta.json HMAC sidecar."""
    X = pd.DataFrame({"f1": [0.1, 0.2, 0.8, 0.9, 0.15, 0.85], "f2": [1.0, 0.9, 0.1, 0.2, 0.8, 0.3]})
    y = np.array([0, 0, 1, 1, 0, 1])
    model = LogisticBaseline(random_seed=42).fit(X, y)
    art_path = tmp_path / "src_model" / "champ.joblib"
    model.save(art_path)
    assert Path(f"{art_path}.meta.json").is_file()

    injected_name = 'svc"); import os; os.system("echo pwned") #'
    bundle_dir = ReportExporters.export_microservice_bundle(
        tmp_path / "bundle",
        model_artifact_path=art_path,
        service_name=injected_name,
    )
    app_py = (bundle_dir / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(app_py)
    # Ensure no `import os` or `os.system` AST nodes were injected
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name != "os" for alias in node.names)

    # Exported bundle contains both artifact and HMAC sidecar and loads cleanly
    bundled_art = bundle_dir / "champ.joblib"
    assert bundled_art.is_file()
    assert Path(f"{bundled_art}.meta.json").is_file()
    reloaded = BaseOmniForgeModel.load(bundled_art)
    assert np.array_equal(reloaded.predict(X), model.predict(X))


def test_studio12_compile_and_export_preserve_section_titles_and_narratives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Studio 12 /s12/compile and HTML/MD exports must populate non-empty titles and narratives for all sections."""
    monkeypatch.setenv("OMNIFORGE_REGISTRY_DIR", str(tmp_path / "reg"))
    monkeypatch.setenv("OMNIFORGE_TRACKING_DIR", str(tmp_path / "trk"))

    client = TestClient(create_app())
    tok = client.post(
        "/api/v1/auth/token",
        json={"username": "rep-user", "role": "data_scientist", "tenant_id": "t-rep"},
        headers={"X-API-Key": get_api_key()},
    ).json()["access_token"]
    hdrs = {"Authorization": f"Bearer {tok}", "X-Session-Id": "s12-nar"}

    assert (
        client.post(
            "/api/workbench/s01/load-sample",
            json={"source_type": "domain", "name": "finance", "n_samples": 50},
            headers=hdrs,
        ).status_code
        == 200
    )
    r_comp = client.post(
        "/api/workbench/s12/compile",
        json={"selected_sections": ["executive_summary", "intake_audit", "quality_diagnostics", "recommendations"]},
        headers=hdrs,
    )
    assert r_comp.status_code == 200
    secs = r_comp.json()["sections"]
    assert len(secs) >= 3
    for s in secs:
        assert s["title"] and isinstance(s["title"], str)
        assert s["narrative"] and isinstance(s["narrative"], str)

    r_html = client.get("/api/workbench/s12/export/html", headers=hdrs)
    assert r_html.status_code == 200
    assert "Automatic Post-Ingestion Defect Audit" in r_html.text
    assert "<h2></h2>" not in r_html.text


def test_cli_validation_and_docker_spa_invariants(tmp_path: Path) -> None:
    """CLI commands must fail cleanly on invalid arguments, and Docker/SPA environment invariants must hold."""
    runner = CliRunner()
    csv_file = tmp_path / "tiny.csv"
    pd.DataFrame({"a": range(20), "b": range(20)}).to_csv(csv_file, index=False)

    assert runner.invoke(cli, ["automl", str(csv_file), "--target", "missing_col"]).exit_code == 1
    assert runner.invoke(cli, ["automl", str(csv_file), "--target", "b", "--budget", "-5"]).exit_code == 1
    assert runner.invoke(cli, ["eda", str(csv_file), "--target", "missing_col"]).exit_code == 1
    assert runner.invoke(cli, ["domains", "--sample", "nonexistent_vertical"]).exit_code == 1
    assert runner.invoke(cli, ["monitor", "--reference", str(csv_file)]).exit_code == 1

    repo_root = Path(__file__).resolve().parents[2]
    dockerfile = (repo_root / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.12-slim@sha256:" in dockerfile
    assert "--require-hashes --only-binary=:all:" in dockerfile
    assert "RUN --network=none python -m pip wheel" in dockerfile
    lock_generator = (repo_root / "scripts" / "build_docker_locks.sh").read_text(encoding="utf-8")
    for extra in ("api", "ui", "reporting", "data", "forecasting"):
        assert f"--extra {extra}" in lock_generator
    for arch in ("amd64", "arm64"):
        lock = (repo_root / "docker" / f"requirements-{arch}.lock").read_text(encoding="utf-8")
        assert "fastapi==" in lock and "--hash=sha256:" in lock

    dockerignore = (repo_root / ".dockerignore").read_text(encoding="utf-8")
    for pattern in (".git", ".env", ".venv*", "frontend/node_modules/"):
        assert pattern in dockerignore

    env_rep = run_environment_check()
    assert env_rep["resources"]["static_app_exists"] is True
    assert env_rep["required_modules"]["defusedxml"] is True
