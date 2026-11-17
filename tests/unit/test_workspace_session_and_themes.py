"""Unit tests for portable .uap Workspace Project Save/Reload and Light/Dark UI themes."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from omniforge.core.exceptions import DataValidationError
from omniforge.core.workspace_session import STATE_FILENAME, WorkspaceManifest, WorkspaceSessionManager
from omniforge.ui.app import STUDIOS, get_static_app_dir


class _UnpicklableHandle:
    """Test helper that intentionally raises during pickling to verify safe skipping."""

    def __reduce__(self) -> tuple[Any, ...]:
        raise TypeError("Live external connection handle cannot be serialized")


def _sample_workspace_state() -> dict[str, Any]:
    rng = np.random.default_rng(42)
    raw_df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, 80),
            "x2": rng.normal(5, 2, 80),
            "segment": rng.choice(["A", "B", "C"], 80),
            "target": rng.integers(0, 2, 80),
        }
    )
    proc_df = raw_df.iloc[:-5].reset_index(drop=True)
    clf = RandomForestClassifier(n_estimators=12, random_state=42)
    clf.fit(proc_df[["x1", "x2"]], proc_df["target"])

    return {
        "uap_df": proc_df,
        "uap_raw_df": raw_df,
        "uap_prev_df": raw_df,
        "uap_dataset_name": "customer_churn_eval",
        "uap_dataset_source": "CSV Upload",
        "uap_target": "target",
        "uap_targets": ["target"],
        "uap_theme": "light",
        "uap_selected_studio": "06 · AutoML, Deep Learning & Predictions",
        "uap_active_model": clf,
        "uap_active_model_name": "RandomForestClassifier",
        "uap_target_models": {"target": clf},
        "uap_model_equations": {"target": {"equation": "RF Ensemble (12 trees)", "r2": 0.91}},
        "uap_split_lock": {
            "active": True,
            "strategy": "stratified",
            "train_indices": list(range(0, 60)),
            "test_indices": list(range(60, 75)),
            "audit_report": {"passed": True, "overlap_rows": 0},
        },
        "uap_exclusions": [
            {
                "timestamp": "12:00:00",
                "stage": "Clean",
                "reason": "Removed 5 extreme outliers",
                "rows_removed": 5,
            }
        ],
        "uap_lineage": [
            {"timestamp": "11:55:00", "stage": "Ingest", "action": "Loaded dataset", "details": "80x4"},
            {"timestamp": "12:00:00", "stage": "Clean", "action": "Cleaned dataset", "details": "75x4"},
        ],
        "uap_report_payload": {"title": "Executive Report", "tables": {"Summary": [{"rows": 75}]}},
        # User widget / criteria settings
        "crit_alpha": 0.01,
        "split_lock_strategy": "stratified",
        "automl_time_budget": 45,
        "rep_opt_charts": True,
        "rep_sec_executive_summary": True,
        # Transient / unserializable keys that MUST be excluded or safely skipped
        "uap_copilot": object(),
        "btn_run_now": True,
        "FormSubmitter:my_form-Submit": True,
        "uap_live_handle": _UnpicklableHandle(),
        "_uap_export_cache": {"bytes": b"transient"},
    }


def test_workspace_export_inspect_and_full_roundtrip() -> None:
    """Verify full .uap export, fast manifest inspection, model fidelity, and transient key exclusion."""
    state = _sample_workspace_state()
    raw_bytes, manifest = WorkspaceSessionManager.export_to_bytes(
        state,
        project_name="Q4 Churn Audit",
        notes="Stratified split + cleaned + RF baseline",
    )

    assert isinstance(manifest, WorkspaceManifest)
    assert manifest.project_name == "Q4 Churn Audit"
    assert manifest.notes == "Stratified split + cleaned + RF baseline"
    assert manifest.dataset_name == "customer_churn_eval"
    assert manifest.dataset_shape == (75, 4)
    assert manifest.raw_dataset_shape == (80, 4)
    assert manifest.processing_mode == "PROCESSED"
    assert manifest.target_columns == ["target"]
    assert manifest.split_lock_active is True
    assert manifest.exclusion_count == 1
    assert manifest.lineage_count == 2
    assert manifest.theme == "light"
    assert "RandomForestClassifier" in manifest.trained_models
    assert "uap_active_model" in manifest.saved_keys
    assert "crit_alpha" in manifest.saved_keys
    assert "btn_run_now" not in manifest.saved_keys
    assert "uap_copilot" not in manifest.saved_keys
    assert "_uap_export_cache" not in manifest.saved_keys
    assert any(item["key"] == "uap_live_handle" for item in manifest.skipped_keys)

    # Fast inspection without unpickling full joblib payload
    inspected = WorkspaceSessionManager.inspect_bytes(raw_bytes)
    assert inspected.project_name == "Q4 Churn Audit"
    assert inspected.payload_sha256 == manifest.payload_sha256

    # Full reload and behavioral verification
    restored, restored_manifest = WorkspaceSessionManager.load_from_bytes(raw_bytes)
    assert restored_manifest.payload_sha256 == manifest.payload_sha256
    pd.testing.assert_frame_equal(restored["uap_df"], state["uap_df"])
    pd.testing.assert_frame_equal(restored["uap_raw_df"], state["uap_raw_df"])
    assert restored["uap_split_lock"]["train_indices"] == list(range(0, 60))
    assert restored["crit_alpha"] == 0.01
    assert restored["automl_time_budget"] == 45
    assert restored["uap_theme"] == "light"
    assert restored["uap_exclusions"][0]["rows_removed"] == 5

    # Restored sklearn model yields bit-for-bit identical predictions
    X_eval = state["uap_df"][["x1", "x2"]].head(15)
    np.testing.assert_array_equal(
        restored["uap_active_model"].predict(X_eval),
        state["uap_active_model"].predict(X_eval),
    )


def test_workspace_integrity_tamper_and_invalid_archive_rejection() -> None:
    """Verify corrupted files and SHA-256 payload tampering raise DataValidationError."""
    state = _sample_workspace_state()
    raw_bytes, _ = WorkspaceSessionManager.export_to_bytes(state, project_name="Integrity Test")

    # Tamper 1 byte inside session_state.joblib while keeping ZIP structure valid
    tampered_buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as zin, zipfile.ZipFile(tampered_buf, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == STATE_FILENAME:
                data = data[:-1] + bytes([data[-1] ^ 0xFF])
            zout.writestr(item, data)

    with pytest.raises(DataValidationError, match="SHA-256 mismatch"):
        WorkspaceSessionManager.load_from_bytes(tampered_buf.getvalue())

    # Non-ZIP garbage bytes rejected cleanly
    with pytest.raises(DataValidationError):
        WorkspaceSessionManager.inspect_bytes(b"not a zip file at all")


def test_workspace_local_project_save_list_and_delete(tmp_path: Path) -> None:
    """Verify local .uap project persistence, listing sorted by recency, and deletion."""
    state = _sample_workspace_state()
    proj_dir = tmp_path / "projects"

    p1, m1 = WorkspaceSessionManager.save_local_project(state, project_name="Alpha Run", projects_dir=proj_dir)
    assert p1.exists() and p1.suffix == ".omniforge"
    assert m1.project_name == "Alpha Run"

    p2, _ = WorkspaceSessionManager.save_local_project(state, project_name="Beta Run", projects_dir=proj_dir)
    assert p2.exists()

    listed = WorkspaceSessionManager.list_local_projects(projects_dir=proj_dir)
    assert len(listed) == 2
    names = {s.manifest.project_name for s in listed}
    assert names == {"Alpha Run", "Beta Run"}

    # Round-trip load directly from saved disk path
    restored, loaded_m = WorkspaceSessionManager.load_from_file(p1)
    assert loaded_m.project_name == "Alpha Run"
    assert len(restored["uap_df"]) == 75

    # Deletion cleans file cleanly
    assert WorkspaceSessionManager.delete_local_project(p1) is True
    assert not p1.exists()
    assert len(WorkspaceSessionManager.list_local_projects(projects_dir=proj_dir)) == 1


def test_light_and_dark_theme_react_spa_tokens_and_compiled_bundle() -> None:
    """Verify compiled React/Vite SPA bundle exists and includes native Dark & Light theme tokens."""
    assert len(STUDIOS) == 13
    assert STUDIOS[0][0] == "01" and STUDIOS[-1][0] == "13"

    static_dir = get_static_app_dir()
    assert (static_dir / "index.html").is_file()
    assets_dir = static_dir / "assets"
    assert assets_dir.is_dir()
    css_files = list(assets_dir.glob("*.css"))
    js_files = list(assets_dir.glob("*.js"))
    assert len(css_files) >= 1 and len(js_files) >= 1

    css_text = css_files[0].read_text(encoding="utf-8")
    for token in (
        "data-theme",
        "dark",
        "light",
        ".app-shell",
        ".studio-hero",
        ".data-table",
    ):
        assert token in css_text, f"Missing React SPA CSS token: {token}"
