"""NIST AI RMF MANAGE: kill-switch + auto-promotion gate for the CT orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omniforge.core.types import ModelStage, TaskType
from omniforge.mlops.continuous_training import ContinuousTrainingOrchestrator, _env_bool
from omniforge.mlops.registry import ModelRegistry
from omniforge.mlops.tracking import ExperimentTracker
from omniforge.models.classical import ClassicalMLModel


def _xy(n: int = 100, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.choice(["a", "b"], n)})
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int), name="target")
    return X, y


def _trainer(Xf, yf):
    return ClassicalMLModel("random_forest", task_type=TaskType.BINARY_CLASSIFICATION, track_carbon=False).fit(Xf, yf)


@pytest.fixture()
def ct_args(tmp_path):
    X, y = _xy()
    ref = X.iloc[:70].copy()
    ref["target"] = y.iloc[:70].values
    new = X.iloc[70:].copy()
    new["target"] = y.iloc[70:].values
    reg = ModelRegistry(tmp_path / "reg")
    trk = ExperimentTracker(tmp_path / "trk")
    return reg, trk, ref, new


def test_env_bool_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIFORGE_X_TEST", "true")
    assert _env_bool("OMNIFORGE_X_TEST", False) is True
    monkeypatch.setenv("OMNIFORGE_X_TEST", "off")
    assert _env_bool("OMNIFORGE_X_TEST", True) is False
    monkeypatch.setenv("OMNIFORGE_X_TEST", "garbage")
    assert _env_bool("OMNIFORGE_X_TEST", True) is True  # falls back to default
    monkeypatch.delenv("OMNIFORGE_X_TEST")
    assert _env_bool("OMNIFORGE_X_TEST", True) is True


def test_kill_switch_blocks_everything(ct_args, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OMNIFORGE_AUDIT__ANCHOR_PATH", str(tmp_path / "anchor.jsonl"))
    reg, trk, ref, new = ct_args
    ct = ContinuousTrainingOrchestrator(reg, trk, kill_switch=True)
    res = ct.evaluate_and_retrain(
        model_name="demo_killswitch", reference_df=ref, new_labeled_df=new,
        target_col="target", trainer_fn=_trainer, force_retrain=True,
    )
    assert res["retrained"] is False
    assert res["kill_switch"] is True
    assert res["drift_report"] is None
    assert "kill" in res["reason"].lower()


def test_kill_switch_blocks_even_forced_retrain(ct_args, tmp_path) -> None:
    reg, trk, ref, new = ct_args
    ct = ContinuousTrainingOrchestrator(reg, trk, kill_switch=True)
    res = ct.evaluate_and_retrain(
        model_name="demo_killswitch2", reference_df=ref, new_labeled_df=new,
        target_col="target", trainer_fn=_trainer, force_retrain=True,
    )
    assert res["retrained"] is False


def test_auto_promotion_withheld_registers_staging_only(ct_args, tmp_path) -> None:
    reg, trk, ref, new = ct_args
    ct = ContinuousTrainingOrchestrator(reg, trk, auto_promote=False)
    res = ct.evaluate_and_retrain(
        model_name="demo_no_autopromote", reference_df=ref, new_labeled_df=new,
        target_col="target", trainer_fn=_trainer, force_retrain=True,
    )
    assert res["retrained"] is True
    assert res["kill_switch"] is False
    assert res["auto_promote"] is False
    # No champion exists → nothing to withhold; model registers in Staging
    assert res["promotion_withheld"] is False
    assert res["assigned_stage"] == ModelStage.STAGING.value


def test_auto_promotion_default_promotes_on_empty_registry(ct_args, tmp_path) -> None:
    reg, trk, ref, new = ct_args
    ct = ContinuousTrainingOrchestrator(reg, trk)  # defaults: kill_switch off, auto_promote on
    res = ct.evaluate_and_retrain(
        model_name="demo_promote", reference_df=ref, new_labeled_df=new,
        target_col="target", trainer_fn=_trainer, force_retrain=True,
    )
    assert res["retrained"] is True
    assert res["promotion_withheld"] is False
    assert res["assigned_stage"] == ModelStage.STAGING.value


def test_env_kill_switch(ct_args, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OMNIFORGE_CT_DISABLED", "true")
    reg, trk, ref, new = ct_args
    ct = ContinuousTrainingOrchestrator(reg, trk)  # reads env
    assert ct.kill_switch is True
    res = ct.evaluate_and_retrain(
        model_name="demo_env_kill", reference_df=ref, new_labeled_df=new,
        target_col="target", trainer_fn=_trainer, force_retrain=True,
    )
    assert res["kill_switch"] is True


def test_env_auto_promote_disabled(ct_args, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OMNIFORGE_CT_AUTO_PROMOTE", "false")
    reg, trk, ref, new = ct_args
    ct = ContinuousTrainingOrchestrator(reg, trk)
    assert ct.auto_promote is False
    res = ct.evaluate_and_retrain(
        model_name="demo_env_promote", reference_df=ref, new_labeled_df=new,
        target_col="target", trainer_fn=_trainer, force_retrain=True,
    )
    assert res["assigned_stage"] == ModelStage.STAGING.value
