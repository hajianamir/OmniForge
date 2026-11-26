"""Closed-loop Continuous Training (CT) Orchestrator with drift triggers and leakage-free champion-challenger promotion."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pandas as pd

from omniforge.core.logging import get_logger
from omniforge.core.types import DriftStatus, ModelStage
from omniforge.data.splitting import LeakageSafeSplitter, SplitConfig
from omniforge.mlops.drift import DriftDetector, DriftReport
from omniforge.mlops.registry import ModelRegistry
from omniforge.mlops.tracking import ExperimentTracker
from omniforge.models.base import BaseOmniForgeModel

logger = get_logger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    logger.warning("unparsable_bool_env", var=name, value=raw, fallback=default)
    return default


def _audit_ct_event(action: str, *, model_name: str, status: str, details: dict[str, Any]) -> None:
    """Best-effort audit of CT safety decisions (never raises into the caller)."""
    try:
        from omniforge.config.settings import get_settings
        from omniforge.core.security import AuditLogger

        settings = get_settings()
        AuditLogger(log_path=settings.paths.audit_log_path).log_event(
            action=action,
            resource_type="model",
            resource_id=model_name,
            actor="system",
            tenant_id=str(getattr(settings, "default_tenant_id", "default-tenant")),
            status=status,
            details=details,
        )
    except Exception as exc:  # observability of the block must never defeat the block itself
        logger.error("ct_safety_audit_write_failed", action=action, model=model_name, error=str(exc))


class ContinuousTrainingOrchestrator:
    """Monitors production drift, retrains challenger models, and promotes winners on strictly unseen holdout data."""

    def __init__(
        self,
        registry: ModelRegistry,
        tracker: ExperimentTracker,
        drift_detector: DriftDetector | None = None,
        *,
        kill_switch: bool | None = None,
        auto_promote: bool | None = None,
    ) -> None:
        self.registry = registry
        self.tracker = tracker
        self.drift_detector = drift_detector or DriftDetector()
        # Kill-switch (NIST AI RMF MANAGE — human oversight of the closed loop):
        #   OMNIFORGE_CT_DISABLED=true      -> evaluate_and_retrain() refuses to do ANYTHING
        #                                      (no drift check, no training, no registration).
        #   OMNIFORGE_CT_AUTO_PROMOTE=false -> challengers may train but are NEVER promoted
        #                                      to PRODUCTION automatically (STAGING only);
        #                                      promotion then requires a human decision.
        # ``None`` reads the environment; explicit constructor args override it.
        self.kill_switch = _env_bool("OMNIFORGE_CT_DISABLED", False) if kill_switch is None else bool(kill_switch)
        self.auto_promote = _env_bool("OMNIFORGE_CT_AUTO_PROMOTE", True) if auto_promote is None else bool(auto_promote)

    def evaluate_and_retrain(
        self,
        *,
        model_name: str,
        reference_df: pd.DataFrame,
        new_labeled_df: pd.DataFrame,
        target_col: str,
        trainer_fn: Callable[[pd.DataFrame, pd.Series], BaseOmniForgeModel],
        force_retrain: bool = False,
        min_improvement: float = 0.0,
        holdout_fraction: float = 0.30,
        time_col: str | None = None,
        group_col: str | None = None,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Run drift check and conditionally train, evaluate on strictly unseen holdout, register, and promote a challenger."""
        if self.kill_switch:
            logger.warning("continuous_training_blocked_kill_switch", model=model_name)
            _audit_ct_event(
                "ct:kill_switch_blocked",
                model_name=model_name,
                status="blocked",
                details={"env": "OMNIFORGE_CT_DISABLED", "force_retrain": force_retrain},
            )
            return {
                "retrained": False,
                "kill_switch": True,
                "reason": "Continuous training is disabled (OMNIFORGE_CT_DISABLED / kill_switch=True); no drift check, training, registration, or promotion was performed.",
                "drift_report": None,
            }
        feature_cols = [c for c in reference_df.columns if c != target_col and c in new_labeled_df.columns]
        drift_report: DriftReport = self.drift_detector.detect(
            reference_df[feature_cols], new_labeled_df[feature_cols]
        )

        should_retrain = force_retrain or drift_report.overall_status == DriftStatus.CRITICAL
        if not should_retrain:
            return {
                "retrained": False,
                "reason": f"Drift status is '{drift_report.overall_status.value}'; retraining not required.",
                "drift_report": drift_report.model_dump(),
            }

        # CRITICAL ANTI-LEAKAGE PROTOCOL:
        # Split new_labeled_df into new_train + strictly unseen new_holdout BEFORE training challenger,
        # so champion and challenger are compared apples-to-apples on identical out-of-sample data!
        if len(new_labeled_df) >= 10:
            split_strat = "time_series" if (time_col and time_col in new_labeled_df.columns) else (
                "group" if (group_col and group_col in new_labeled_df.columns) else "auto"
            )
            split_res = LeakageSafeSplitter(
                SplitConfig(
                    strategy=split_strat,  # type: ignore[arg-type]
                    test_size=min(max(holdout_fraction, 0.15), 0.50),
                    time_col=time_col,
                    group_col=group_col,
                    random_seed=random_seed,
                )
            ).split(new_labeled_df, target_col=target_col)
            new_train_df = split_res.train_df
            new_holdout_df = split_res.test_df
            audit_dict = split_res.audit_report.model_dump() if split_res.audit_report else {}
        else:
            # Too small to carve out a holdout: evaluate strictly on new_labeled_df without including it in training
            new_train_df = new_labeled_df.iloc[:0]
            new_holdout_df = new_labeled_df.copy()
            audit_dict = {"passed": True, "note": "Small batch: challenger trained on reference only; evaluated on full new batch."}

        X_hold = new_holdout_df[feature_cols]
        y_hold = new_holdout_df[target_col]

        # Evaluate current production champion on strictly unseen holdout
        champion_score: float | None = None
        champion_version_num: int | None = None
        try:
            champ_model, champ_ver = self.registry.load_model(model_name, stage=ModelStage.PRODUCTION)
            champion_score = float(champ_model.score(X_hold, y_hold))
            champion_version_num = champ_ver.version
        except Exception as exc:
            logger.debug("no_existing_production_champion", model=model_name, error=str(exc))

        # Train challenger strictly on reference + new_train (ZERO overlap with new_holdout!)
        train_parts = [reference_df[[*feature_cols, target_col]]]
        if len(new_train_df) > 0:
            train_parts.append(new_train_df[[*feature_cols, target_col]])
        train_combined = pd.concat(train_parts, ignore_index=True)

        with self.tracker.start_run(experiment_name=f"ct_{model_name}", run_name="challenger_retrain") as run_id:
            challenger = trainer_fn(train_combined[feature_cols], train_combined[target_col])
            challenger_score = float(challenger.score(X_hold, y_hold))
            self.tracker.log_metrics(
                {
                    "challenger_holdout_score": challenger_score,
                    "champion_holdout_score": champion_score if champion_score is not None else 0.0,
                    "drifted_feature_ratio": drift_report.drifted_feature_ratio,
                    "holdout_rows": float(len(new_holdout_df)),
                }
            )
            # When no existing champion exists (champion_score is None), register challenger
            # in STAGING for human review — do NOT auto-promote to PRODUCTION.
            would_have_promoted = (
                champion_score is not None
                and challenger_score >= champion_score + min_improvement
            )
            promoted = would_have_promoted and self.auto_promote and not self.registry.require_fairness_gate
            if would_have_promoted and not promoted:
                logger.warning(
                    "continuous_training_auto_promotion_withheld",
                    model=model_name,
                    challenger_score=round(challenger_score, 4),
                )
                _audit_ct_event(
                    "ct:auto_promotion_withheld",
                    model_name=model_name,
                    status="warning",
                    details={
                        "challenger_score": round(challenger_score, 6),
                        "champion_score": round(champion_score, 6) if champion_score is not None else None,
                        "reason": "fairness_evidence_required" if self.registry.require_fairness_gate else "auto_promote_disabled",
                        "stage": ModelStage.STAGING.value,
                    },
                )
            stage = ModelStage.PRODUCTION if promoted else ModelStage.STAGING
            reg_ver = self.registry.register_model(
                challenger,
                model_name,
                metrics={"holdout_validation_score": round(challenger_score, 6)},
                tags={
                    "ct_run_id": run_id,
                    "trigger": "drift" if not force_retrain else "forced",
                    "leakage_free_holdout": "true",
                },
                stage=stage,
            )

        logger.info(
            "continuous_training_completed",
            model=model_name,
            promoted=promoted,
            new_version=reg_ver.version,
            challenger_score=round(challenger_score, 4),
            champion_score=round(champion_score, 4) if champion_score is not None else None,
            holdout_rows=len(new_holdout_df),
        )
        return {
            "retrained": True,
            "promoted_to_production": promoted,
            "new_version": reg_ver.version,
            "assigned_stage": stage.value,
            "previous_champion_version": champion_version_num,
            "champion_score": round(champion_score, 6) if champion_score is not None else None,
            "challenger_score": round(challenger_score, 6),
            "holdout_rows": int(len(new_holdout_df)),
            "leakage_free_evaluation": True,
            "split_audit": audit_dict,
            "drift_report": drift_report.model_dump(),
            "kill_switch": False,
            "auto_promote": self.auto_promote,
            "promotion_withheld": would_have_promoted and not promoted,
        }

    def monitor_and_rollback(
        self,
        monitor: Any,
        *,
        min_feedback_for_rollback: int = 5,
    ) -> dict[str, Any]:
        """Evaluate live :class:`ModelMonitor` SLAs and automatically roll back PRODUCTION on degradation (RR-038)."""
        if self.kill_switch:
            return {
                "status": "blocked_by_kill_switch",
                "kill_switch": True,
                "rollback_executed": False,
            }
        health: dict[str, Any] = monitor.evaluate_health(
            registry=self.registry,
            auto_rollback=True,
            min_feedback_for_rollback=min_feedback_for_rollback,
        )
        return health
