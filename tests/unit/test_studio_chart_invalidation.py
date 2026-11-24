"""Chart caches must track study scope, split provenance, and standalone results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from omniforge.core.exceptions import DataValidationError
from omniforge.core.performance import compact_scatter_payload
from omniforge.ui.state import (
    clear_split_lock,
    clear_workspace_dataset,
    default_workspace_dict,
    export_workspace_project_bytes,
    get_active_split_frames,
    get_studio_results,
    load_new_dataset,
    lock_leakage_safe_split,
    restore_workspace_project_bytes,
    set_excluded_columns,
    set_selected_targets,
    update_active_dataset,
)


def _state() -> dict[str, Any]:
    state = default_workspace_dict()
    df = pd.DataFrame({"alpha": range(40), "beta": range(40, 80), "target": range(80, 120)})
    state.update(
        {
            "uap_df": df,
            "uap_raw_df": df.copy(),
            "uap_targets": ["target"],
            "uap_target": "target",
            "uap_split_lock": {"active": True, "target_col": "target", "train_indices": list(range(32)), "test_indices": list(range(32, 40))},
            "uap_split_audit_report": {"passed": True},
            "uap_s02_cluster_committed": True,
            "uap_s02_cluster_summary": {"action": "attached"},
            "uap_quality": {"overall_score": 100},
            "uap_clean_report": {"committed": True},
            "uap_clustering_result": {"old": True},
            "uap_clustering_df": df.copy(),
            "uap_eda_report": {"old": True},
            "uap_hypothesis_battery": [{"old": True}],
            "uap_forecast_result": {"old": True},
            "uap_active_model": object(),
            "uap_baseline_models": {"target": {"baseline": {"model": object()}}},
            "uap_supervisory_models": {"quick": {"model": object()}},
            "uap_supervisory_result": {"system_status": "NORMAL"},
            "uap_supervisory_history": [{"source": "historical batch"}],
            "uap_supervisory_scored_df": df.copy(),
            "uap_studio_results": {
                "s01_quality": {"score": 100},
                "s01_clean": {"preview_only": False},
                "s02_clustering": {"scatter": {"x": [1]}},
                "s02_mining": {"manifold": {"method": "pca"}},
                "s03_diagnostics": {"report": {}},
                "s04_eda": {"distributions": [{"column": "alpha"}]},
                "s04_hypothesis": {"battery": []},
                "s05_select": {"rankings": []},
                "s06_automl": {"best_model": "old"},
                "s07_explain": {"global_importance": {}},
                "s08_forecast": {"history": []},
                "s09_graph": {"visualization": {}},
                "s09_nlp": {"text": "standalone"},
                "s09_vision": {"label": "standalone"},
                "s12_report": {"title": "old"},
                "s14_supervisory": {"system_status": "NORMAL"},
            },
        }
    )
    return state


def test_exclusion_invalidates_all_study_dependent_charts_and_models() -> None:
    state = _state()
    lock = state["uap_split_lock"]
    assert set_excluded_columns(state, ["alpha"]) == ["alpha"]
    assert set(get_studio_results(state)) == {"s01_quality", "s01_clean", "s09_nlp", "s09_vision"}
    assert state["uap_split_lock"] is lock  # row split is still valid for the same target
    assert state["uap_split_audit_report"] == {"passed": True}
    assert state["uap_s02_cluster_committed"] is True
    assert state["uap_s02_cluster_summary"] == {"action": "attached"}
    assert state["uap_quality"] == {"overall_score": 100}
    for key in (
        "uap_clustering_result", "uap_clustering_df", "uap_eda_report", "uap_hypothesis_battery",
        "uap_forecast_result", "uap_active_model", "uap_supervisory_result", "uap_supervisory_scored_df",
    ):
        assert state[key] is None, key
    assert state["uap_supervisory_models"] == {}
    # An unchanged request must not erase newly computed results.
    state["uap_studio_results"]["s04_eda"] = {"new": True}
    set_excluded_columns(state, ["alpha"])
    assert state["uap_studio_results"]["s04_eda"] == {"new": True}


def test_target_selection_including_an_excluded_column_invalidates_and_unlocks() -> None:
    state = _state()
    state["uap_excluded_columns"] = ["beta"]
    assert set_selected_targets(state, ["beta"]) == ["beta"]
    assert state["uap_excluded_columns"] == []
    assert state["uap_split_lock"] is None  # the old split was stratified for a different target
    assert state["uap_active_model"] is None
    assert "s04_eda" not in get_studio_results(state)
    assert get_studio_results(state)["s09_nlp"] == {"text": "standalone"}


def test_excluding_current_target_unlocks_old_split() -> None:
    state = _state()
    set_excluded_columns(state, ["target"])
    assert state["uap_target"] == "beta"
    assert state["uap_split_lock"] is None
    assert state["uap_split_audit_report"] is None


def test_lock_and_unlock_invalidate_split_fits_but_keep_full_data_charts() -> None:
    state = _state()
    df = state["uap_df"]
    lock = lock_leakage_safe_split(state, df, strategy="random", test_size=0.2, cv_folds=3)
    assert state["uap_split_lock"] is lock
    keys = set(get_studio_results(state))
    assert {"s02_mining", "s04_eda", "s04_hypothesis", "s08_forecast", "s09_graph", "s09_nlp", "s09_vision"} <= keys
    assert not {"s02_clustering", "s03_diagnostics", "s05_select", "s06_automl", "s07_explain", "s12_report", "s14_supervisory"} & keys
    assert state["uap_clustering_df"] is None
    assert state["uap_baseline_models"] == {}
    assert state["uap_supervisory_models"] == {}
    assert state["uap_supervisory_history"] == [{"source": "historical batch"}]
    assert state["uap_eda_report"] == {"old": True}

    state["uap_studio_results"]["s05_select"] = {"under_lock": True}
    state["uap_studio_results"]["s14_supervisory"] = {"under_lock": True}
    state["uap_supervisory_result"] = {"under_lock": True}
    clear_split_lock(state)
    assert state["uap_split_lock"] is None
    assert "s04_eda" in get_studio_results(state)
    assert "s05_select" not in get_studio_results(state)
    assert "s14_supervisory" not in get_studio_results(state)
    assert state["uap_supervisory_result"] is None


def test_clean_preview_is_invalidated_but_committed_history_and_standalone_results_are_retained() -> None:
    state = _state()
    state["uap_studio_results"]["s01_clean"] = {"preview_only": True}
    set_excluded_columns(state, ["alpha"])
    assert "s01_clean" not in get_studio_results(state)
    assert state["uap_clean_report"] == {"committed": True}

    df = state["uap_df"].copy()
    load_new_dataset(state, df, "next", "test", default_target="target")
    assert set(get_studio_results(state)) == {"s09_nlp", "s09_vision"}
    clear_workspace_dataset(state)
    assert set(get_studio_results(state)) == {"s09_nlp", "s09_vision"}


def test_scatter_drops_nonfinite_pairs_and_keeps_group_alignment() -> None:
    payload = compact_scatter_payload(
        np.array([1.0, np.nan, np.inf, 4.0, 5.0]),
        np.array([2.0, 3.0, 4.0, np.nan, 6.0]),
        groups=np.array([10, 20, 30, 40, 50]),
    )
    assert payload == {
        "x": [1.0, 5.0], "y": [2.0, 6.0], "groups": [10, 50],
        "total_points": 5, "rendered_points": 2, "downsampled": False,
    }
    sampled = compact_scatter_payload(np.arange(10.0), np.arange(10.0), groups=np.arange(10), max_points=3)
    assert sampled["rendered_points"] == 3 and sampled["downsampled"] is True
    assert sampled["x"] == sampled["y"] == sampled["groups"]
    assert sampled == compact_scatter_payload(np.arange(10.0), np.arange(10.0), groups=np.arange(10), max_points=3)


@pytest.mark.parametrize("index_offset", [0, 1000])
def test_dataset_update_realigns_positional_split_after_deletion_and_reordering(index_offset: int) -> None:
    state = _state()
    df = state["uap_df"].copy()
    df.index = pd.Index([index_offset + i * (1 if index_offset == 0 else 3) for i in range(len(df))])
    state["uap_df"] = df
    train_before, test_before, _ = get_active_split_frames(state)
    assert test_before is not None
    # Remove rows from both partitions and reorder the survivors.
    surviving = [i for i in reversed(range(len(df))) if i not in {0, 4, 33, 37}]
    updated = df.iloc[surviving].copy()
    update_active_dataset(state, updated, "Clustering", "Removed rows", row_index_preserved=True)
    train_after, test_after, lock = get_active_split_frames(state)
    assert test_after is not None and lock is not None
    assert set(train_after["alpha"]) == set(train_before["alpha"]) - {0, 4}
    assert set(test_after["alpha"]) == set(test_before["alpha"]) - {33, 37}
    assert set(train_after["alpha"]).isdisjoint(test_after["alpha"])
    assert lock["n_train"] == len(train_after) and lock["n_test"] == len(test_after)
    assert lock["audit_report"]["n_train"] == len(train_after)
    assert lock["audit_report"]["n_test"] == len(test_after)
    assert lock["audit_report"]["index_overlap_count"] == 0


def test_dataset_update_realigns_explicit_row_ids_even_if_index_is_reset() -> None:
    state = _state()
    df = state["uap_df"].copy()
    df["row_id"] = [f"row-{i}" for i in range(len(df))]
    state["uap_df"] = df
    state["uap_split_lock"] = {
        "active": True, "target_col": "target", "row_id_col": "row_id",
        "train_row_ids": df["row_id"].iloc[:32].tolist(),
        "test_row_ids": df["row_id"].iloc[32:].tolist(),
    }
    updated = df.iloc[[i for i in reversed(range(40)) if i not in {2, 35}]].reset_index(drop=True)
    update_active_dataset(state, updated, "Cleaning", "Dropped rows")
    train, test, lock = get_active_split_frames(state)
    assert test is not None and lock is not None
    assert set(train["row_id"]) == {f"row-{i}" for i in range(32)} - {"row-2"}
    assert set(test["row_id"]) == {f"row-{i}" for i in range(32, 40)} - {"row-35"}
    assert set(lock["train_row_ids"]) == set(train["row_id"])
    assert set(lock["test_row_ids"]) == set(test["row_id"])


@pytest.mark.parametrize("invalid_identity", ["duplicate", "unknown", "target_dropped"])
def test_dataset_update_unlocks_if_membership_cannot_be_safely_preserved(invalid_identity: str) -> None:
    state = _state()
    df = state["uap_df"].copy()
    if invalid_identity == "duplicate":
        df.index = [i // 2 for i in range(len(df))]
        state["uap_df"] = df
        updated = df.copy()
    elif invalid_identity == "unknown":
        updated = df.copy()
        updated.index = [*range(39), 999]
    else:
        updated = df.drop(columns=["target"])
    update_active_dataset(state, updated, "Cleaning", "Changed dataset", row_index_preserved=True)
    assert state["uap_split_lock"] is None
    assert state["uap_split_audit_report"] is None
    assert any(entry["action"] == "Unlocked split after dataset update" for entry in state["uap_lineage"])


@pytest.mark.parametrize("change", ["filtered_reset", "reordered_reset", "unchanged_labels"])
def test_dataset_update_without_proven_index_identity_unlocks(change: str) -> None:
    state = _state()
    df = state["uap_df"]
    if change == "filtered_reset":
        # The new RangeIndex is a subset of the old one, but labels now point
        # to different rows. Matching labels alone cannot establish identity.
        updated = df.iloc[5:38].reset_index(drop=True)
    elif change == "reordered_reset":
        updated = df.iloc[::-1].reset_index(drop=True)
    else:
        updated = df.copy()
    update_active_dataset(state, updated, "External transform", "Updated rows")
    assert state["uap_split_lock"] is None
    assert state["uap_split_audit_report"] is None


@pytest.mark.parametrize("constraint", ["group", "time"])
@pytest.mark.parametrize("violation", [False, True])
def test_realign_reaudits_group_and_temporal_constraints(constraint: str, violation: bool) -> None:
    state = _state()
    df = state["uap_df"].copy()
    df["entity"] = ["train"] * 32 + ["test"] * 8
    df["when"] = pd.date_range("2024-01-01", periods=len(df), freq="D")
    state["uap_df"] = df
    lock = state["uap_split_lock"]
    if constraint == "group":
        lock.update(strategy="group", group_col="entity")
    else:
        lock.update(strategy="time_series", time_col="when")
    updated = df.copy()
    if violation:
        if constraint == "group":
            updated.loc[0, "entity"] = "test"
        else:
            updated.loc[0, "when"] = pd.Timestamp("2025-01-01")
    update_active_dataset(state, updated, "Cleaning", "Changed data", row_index_preserved=True)
    if violation:
        assert state["uap_split_lock"] is None
        assert state["uap_split_audit_report"] is None
    else:
        assert state["uap_split_lock"]["active"]
        assert state["uap_split_audit_report"]["passed"]
        assert state["uap_split_audit_report"]["group_overlap_count"] == 0
        assert state["uap_split_audit_report"]["temporal_order_valid"]


def test_split_with_unknown_or_missing_row_ids_is_not_silently_reinterpreted() -> None:
    state = _state()
    df = state["uap_df"].copy()
    df["row_id"] = [f"row-{i}" for i in range(len(df))]
    state["uap_df"] = df
    lock = state["uap_split_lock"]
    lock.update(row_id_col="row_id", train_row_ids=df["row_id"].iloc[:32].tolist(), test_row_ids=df["row_id"].iloc[32:].tolist())
    lock["test_row_ids"][0] = "not-in-dataset"
    with pytest.raises(ValueError, match="Active Split Lock is invalid"):
        get_active_split_frames(state)
    update_active_dataset(state, df, "Cleaning", "Changed dataset", row_index_preserved=True)
    assert state["uap_split_lock"] is None

    # A partially populated row-ID lock must not fall back to positional indices.
    state["uap_split_lock"] = {"active": True, "row_id_col": "row_id", "train_indices": list(range(32)), "test_indices": list(range(32, 40))}
    with pytest.raises(ValueError, match="Active Split Lock is invalid"):
        get_active_split_frames(state)
    update_active_dataset(state, df, "Cleaning", "Changed dataset", row_index_preserved=True)
    assert state["uap_split_lock"] is None


@pytest.mark.parametrize("problem", ["overlap", "duplicate", "out_of_range"])
def test_invalid_split_membership_is_rejected_instead_of_silently_filtered(problem: str) -> None:
    state = _state()
    lock = state["uap_split_lock"]
    # Arrays occur in some older archives; they must not trigger truth-value errors.
    lock["train_indices"] = np.array(lock["train_indices"])
    lock["test_indices"] = np.array(lock["test_indices"])
    if problem == "overlap":
        lock["test_indices"][0] = lock["train_indices"][0]
    elif problem == "duplicate":
        lock["train_indices"][1] = lock["train_indices"][0]
    else:
        lock["test_indices"][0] = 999
    with pytest.raises(ValueError, match="Active Split Lock is invalid"):
        get_active_split_frames(state)


def test_restoring_legacy_workspace_replaces_stale_cache_and_migrates_real_monitor_result() -> None:
    state = _state()
    state["uap_selected_studio"] = "07"
    old_dataset = state["uap_df"]
    old_cache = get_studio_results(state)
    # A rejected or corrupt file must not mutate the workspace.
    with pytest.raises(DataValidationError):
        restore_workspace_project_bytes(state, b"invalid archive")
    assert state["uap_df"] is old_dataset
    assert get_studio_results(state) == old_cache

    legacy = {
        "uap_df": old_dataset.copy(), "uap_raw_df": old_dataset.copy(),
        "uap_target": "target", "uap_selected_studio": "14 · Supervisory",
        "uap_theme_radio": "light",
        "uap_supervisory_result": {"system_status": "WARNING", "source": "legacy batch"},
        "uap_supervisory_history": [{"source": "legacy batch"}],
    }
    blob, manifest = export_workspace_project_bytes(legacy, project_name="legacy")
    assert "uap_studio_results" not in manifest.saved_keys
    restore_workspace_project_bytes(state, blob, restore_studio=False)
    assert state["uap_selected_studio"] == "07"
    assert state["uap_theme"] == "light"
    assert get_studio_results(state) == {
        "s14_supervisory": {
            "system_status": "WARNING", "source": "legacy batch", "history": [{"source": "legacy batch"}],
        }
    }
    assert state["uap_active_model"] is None

    # An explicit newer cache is authoritative: never resurrect a stale raw
    # evaluation which the saved workspace had already invalidated.
    legacy["uap_studio_results"] = {"s04_eda": {"restored": True}}
    blob, _ = export_workspace_project_bytes(legacy, project_name="explicit cache")
    restore_workspace_project_bytes(state, blob)
    assert get_studio_results(state) == {"s04_eda": {"restored": True}}
    assert state["uap_selected_studio"] == "14"
