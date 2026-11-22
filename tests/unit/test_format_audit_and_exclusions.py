"""Unit tests for OmniForge v2.0.0 automatic format auditing, column exclusion, deleted-row provenance & truthful pipeline status."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.data.cleaning import CleaningConfig, DataCleaner
from omniforge.data.format_audit import DataFormatAuditor
from omniforge.ui.state import (
    SESSION_STORE,
    export_workspace_project_bytes,
    get_all_deleted_records_df,
    get_analysis_dataframe,
    get_dataset_processing_status,
    get_excluded_columns,
    get_selected_targets,
    get_session_state,
    get_study_df,
    load_new_dataset,
    record_deleted_data,
    record_lineage,
    restore_workspace_project_bytes,
    set_excluded_columns,
    set_selected_targets,
    update_active_dataset,
)
from omniforge.version import __platform_name__, __version__, get_version_info


def test_omniforge_v2_branding_metadata() -> None:
    """Verify OmniForge product name and v4.0.0 semantic version metadata."""
    info = get_version_info()
    assert __platform_name__ == "OmniForge"
    assert __version__ == "4.0.0"
    assert info["name"] == "OmniForge"
    assert info["short_name"] == "OmniForge"
    assert info["version"] == "4.0.0"


def test_format_auditor_detects_comprehensive_format_defects() -> None:
    """Verify DataFormatAuditor detects numeric-as-text, mixed types, inconsistent dates, hidden nulls, whitespace, casing, HTML/mojibake, inf, constant, empty & ID columns."""
    n = 30
    df = pd.DataFrame(
        {
            "Unnamed: 0": list(range(n)),
            "customer_id": [f"CUST-{i:04d}" for i in range(n)],
            " revenue ": [f"${1000 + i * 25:,.2f}" if i != 5 else "N/A" for i in range(n)],
            "euro_price": [f"{10 + i},50" for i in range(n)],
            "mixed_col": [str(i * 2) if i % 2 == 0 else f"text_{i}" for i in range(n)],
            "event_date": [
                f"2026-01-{(i % 28) + 1:02d}" if i < 15 else f"01/{(i % 28) + 1:02d}/2026"
                for i in range(n)
            ],
            "category": [
                "  Active  " if i % 3 == 0 else ("active" if i % 3 == 1 else "ACTIVE")
                for i in range(n)
            ],
            "notes": [f"<b>Item {i}</b>\nline2" if i < 10 else f"cafÃ© {i}" for i in range(n)],
            "score": [float(i) if i != 3 else float("inf") for i in range(n)],
            "empty_col": [np.nan] * n,
            "const_col": ["CONSTANT"] * n,
            "target": [i % 2 for i in range(n)],
        }
    )

    report = DataFormatAuditor().audit(df)
    codes = {iss.issue_code for iss in report.issues}

    assert report.total_issues >= 10
    assert report.high_severity_count >= 4
    assert "UNNAMED_INDEX_COLUMN" in codes
    assert "WHITESPACE_IN_HEADER" in codes
    assert "HIDDEN_MISSING_TOKENS" in codes
    assert "NUMERIC_STORED_AS_TEXT" in codes
    assert "EUROPEAN_DECIMAL_COMMA" in codes
    assert "MIXED_TYPES_IN_COLUMN" in codes
    assert "INCONSISTENT_DATE_FORMATS" in codes
    assert "EXCESS_WHITESPACE" in codes
    assert "INCONSISTENT_CASING" in codes
    assert "CONTROL_OR_HTML_CHARS" in codes
    assert "ENCODING_MOJIBAKE" in codes
    assert "INFINITE_NUMERIC_VALUES" in codes
    assert "ALL_EMPTY_COLUMN" in codes
    assert "CONSTANT_COLUMN" in codes
    assert "SUSPECTED_ID_COLUMN" in codes
    for expected_sug in ("Unnamed: 0", "customer_id", "empty_col", "const_col"):
        assert expected_sug in report.suggested_exclude_columns


def test_format_auditor_clean_dataset_passes_with_zero_issues() -> None:
    """Verify a cleanly formatted numeric + categorical dataset reports zero issues."""
    df = pd.DataFrame(
        {
            "x1": [1.2, 2.4, 3.6, 4.8, 5.1, 6.3, 7.5, 8.7],
            "x2": [10, 20, 30, 40, 50, 60, 70, 80],
            "segment": ["North", "South", "East", "West", "North", "South", "East", "West"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    report = DataFormatAuditor().audit(df)
    assert report.total_issues == 0
    assert report.health_status == "Clean"
    assert report.clean_columns_count == 4


def test_data_cleaner_captures_exact_deleted_rows_with_reasons() -> None:
    """Verify DataCleaner records exact original row index, deletion reason, and values for duplicates & removed outliers."""
    df = pd.DataFrame(
        {
            "x": [10.0, 11.0, 12.0, 10.5, 11.5, 12.5, 10.0, 9999.0],
            "y": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.0, 2.0],
        },
        index=[100, 101, 102, 103, 104, 105, 106, 107],
    )
    cfg = CleaningConfig(
        drop_exact_duplicates=True,
        outlier_method="iqr",
        outlier_action="remove",
        outlier_threshold=1.5,
    )
    cleaner = DataCleaner(cfg)
    cleaned, rep = cleaner.fit_transform(df)

    assert len(cleaned) == 6
    assert rep.exact_duplicates_removed == 1
    assert rep.removed_rows_count == 2
    assert not cleaner.last_removed_rows_.empty
    assert len(cleaner.last_removed_rows_) == 2

    orig_indices = set(cleaner.last_removed_rows_["_original_row_index"].tolist())
    assert orig_indices == {106, 107}
    reasons = cleaner.last_removed_rows_["_deletion_reason"].tolist()
    assert any("Exact Duplicate" in r for r in reasons)
    assert any("Outlier Removed" in r and "x" in r for r in reasons)


def test_column_exclusion_truthful_provenance_and_uap_roundtrip() -> None:
    """Verify non-destructive column exclusion, truthful RAW vs PROCESSED provenance, deleted record logging, and .uap persistence."""
    sid = SESSION_STORE.new_session_id()
    state = get_session_state(sid)

    raw_df = pd.DataFrame(
        {
            "row_id": [f"ID_{i}" for i in range(40)],
            "f1": [float(i) for i in range(40)],
            "f2": [float(i * 2) for i in range(40)],
            "target": [i % 2 for i in range(40)],
        }
    )
    load_new_dataset(state, raw_df, "test_study", "Unit Test", default_target="target")

    # Format audit runs automatically on load
    assert isinstance(state.get("uap_format_audit"), dict)
    assert "row_id" in state["uap_format_audit"]["suggested_exclude_columns"]

    # Initially strictly RAW
    proc0 = get_dataset_processing_status(state)
    assert proc0["is_processed"] is False
    assert proc0["mode_label"] == "RAW"

    # Read-only Clustering Analysis MUST NOT falsely flip banner to PROCESSED
    state["uap_clustering_result"] = {"algorithm": "kmeans"}
    state["uap_s02_cluster_committed"] = False
    record_lineage(state, "Clustering Analysis", "Preview clustering")
    proc_preview = get_dataset_processing_status(state)
    assert proc_preview["is_processed"] is False
    assert proc_preview["mode_label"] == "RAW"
    assert proc_preview["has_pending_clustering"] is True

    # Exclude 'row_id' non-destructively
    applied = set_excluded_columns(state, ["row_id"])
    assert applied == ["row_id"]
    assert "row_id" in state["uap_df"].columns  # Preserved safely in uap_df
    study = get_study_df(state)
    assert study is not None
    assert "row_id" not in study.columns
    assert list(study.columns) == ["f1", "f2", "target"]
    assert list(get_analysis_dataframe(state).columns) == ["f1", "f2", "target"]

    # Selecting an excluded column as target automatically un-excludes it
    set_selected_targets(state, ["row_id"])
    assert "row_id" not in get_excluded_columns(state)
    assert get_selected_targets(state) == ["row_id"]

    # Re-set target to 'target' and exclude 'row_id' again
    set_selected_targets(state, ["target"])
    set_excluded_columns(state, ["row_id"])

    # Simulate Studio 02 Cluster Deletion commit
    dropped = state["uap_df"].iloc[:5].copy()
    kept = state["uap_df"].iloc[5:].copy()
    kept["cluster_id"] = 0
    record_deleted_data(
        state,
        stage="Studio 02 · Clustering Cleaning",
        reason="Cluster Removed (cluster_id=-1)",
        removed_df=dropped,
    )
    update_active_dataset(state, kept, "Clustering Cleaning", "Removed 5 noise rows")
    state["uap_s02_cluster_committed"] = True

    proc_after = get_dataset_processing_status(state)
    assert proc_after["is_processed"] is True
    assert proc_after["is_cluster_applied"] is True
    assert proc_after["has_pending_clustering"] is False
    assert proc_after["deleted_rows_total"] == 5
    all_del = get_all_deleted_records_df(state)
    assert len(all_del) == 5
    assert "_deletion_reason" in all_del.columns

    # Downstream Studios 03+ receive cleaned + clustered + study-scoped data
    downstream_df = get_study_df(state)
    assert downstream_df is not None
    assert len(downstream_df) == 35
    assert "cluster_id" in downstream_df.columns
    assert "row_id" not in downstream_df.columns

    # Roundtrip .uap project persistence preserves excluded columns, format audit & deleted records
    blob, _manifest = export_workspace_project_bytes(state, project_name="omniforge_v2_test")
    sid2 = SESSION_STORE.new_session_id()
    state2 = get_session_state(sid2)
    restore_workspace_project_bytes(state2, blob)
    assert get_excluded_columns(state2) == ["row_id"]
    assert len(get_all_deleted_records_df(state2)) == 5
    assert isinstance(state2.get("uap_format_audit"), dict)
    assert len(get_study_df(state2)) == 35

    SESSION_STORE.delete(sid)
    SESSION_STORE.delete(sid2)
