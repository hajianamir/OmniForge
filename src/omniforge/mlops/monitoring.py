"""Production Model Monitoring, SLA Alerting, and Prometheus Metrics Telemetry."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field

from omniforge.core.logging import get_logger

if TYPE_CHECKING:
    from omniforge.mlops.registry import ModelRegistry

logger = get_logger(__name__)


class AlertEvent(BaseModel):
    """Operational monitoring alert emitted when SLA or quality thresholds are breached."""

    alert_type: str
    severity: str  # warning, critical
    message: str
    observed_value: float
    threshold: float
    triggered_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ModelMonitor:
    """Tracks production inference latency, throughput, prediction distribution, and live error SLAs."""

    def __init__(
        self,
        model_name: str,
        *,
        max_p95_latency_ms: float = 250.0,
        min_accuracy: float | None = 0.75,
        max_mae: float | None = None,
        max_error_rate: float = 0.02,
        max_window_size: int = 10_000,
    ) -> None:
        self.model_name = model_name
        self.max_p95_latency_ms = max_p95_latency_ms
        self.min_accuracy = min_accuracy
        self.max_mae = max_mae
        self.max_error_rate = max_error_rate
        self.max_window_size = max(10, int(max_window_size))
        self._latencies: list[float] = []
        self._predictions: list[float] = []
        self._actuals: list[float] = []
        self._paired_feedback: list[tuple[float, float]] = []
        self._errors: int = 0
        self._total_requests: int = 0
        self.alerts: list[AlertEvent] = []

    def record_inference(
        self,
        *,
        latency_ms: float,
        prediction: float | int | None = None,
        actual: float | int | None = None,
        failed: bool = False,
    ) -> None:
        """Record a single inference telemetry event (with optional ground-truth feedback)."""
        self._total_requests += 1
        self._latencies.append(float(latency_ms))
        if len(self._latencies) > self.max_window_size:
            self._latencies = self._latencies[-self.max_window_size :]
        if failed:
            self._errors += 1
        if prediction is not None:
            self._predictions.append(float(prediction))
            if len(self._predictions) > self.max_window_size:
                self._predictions = self._predictions[-self.max_window_size :]
        if actual is not None and prediction is not None:
            self.record_feedback(prediction=prediction, actual=actual)

    def record_feedback(self, *, prediction: float | int, actual: float | int) -> None:
        """Record a paired (prediction, actual) ground-truth label even when labels arrive asynchronously."""
        p_val = float(prediction)
        a_val = float(actual)
        self._actuals.append(a_val)
        self._paired_feedback.append((p_val, a_val))
        if len(self._actuals) > self.max_window_size:
            self._actuals = self._actuals[-self.max_window_size :]
        if len(self._paired_feedback) > self.max_window_size:
            self._paired_feedback = self._paired_feedback[-self.max_window_size :]

    def evaluate_health(
        self,
        *,
        registry: ModelRegistry | None = None,
        auto_rollback: bool = False,
        min_feedback_for_rollback: int = 5,
    ) -> dict[str, Any]:
        """Compute SLA summary statistics, emit alerts, and optionally trigger automated production rollback."""
        new_alerts: list[AlertEvent] = []
        p50 = float(np.median(self._latencies)) if self._latencies else 0.0
        p95 = float(np.quantile(self._latencies, 0.95)) if self._latencies else 0.0
        p99 = float(np.quantile(self._latencies, 0.99)) if self._latencies else 0.0
        err_rate = self._errors / max(self._total_requests, 1)

        if p95 > self.max_p95_latency_ms:
            new_alerts.append(
                AlertEvent(
                    alert_type="latency_sla_breach",
                    severity="warning",
                    message=f"p95 latency ({p95:.1f} ms) exceeds SLA ({self.max_p95_latency_ms:.1f} ms).",
                    observed_value=round(p95, 3),
                    threshold=self.max_p95_latency_ms,
                )
            )
        if err_rate > self.max_error_rate:
            new_alerts.append(
                AlertEvent(
                    alert_type="error_rate_breach",
                    severity="critical",
                    message=f"Inference error rate ({err_rate*100:.2f}%) exceeds threshold ({self.max_error_rate*100:.2f}%).",
                    observed_value=round(err_rate, 4),
                    threshold=self.max_error_rate,
                )
            )

        live_acc: float | None = None
        live_mae: float | None = None
        live_rmse: float | None = None
        if self._paired_feedback:
            pairs = np.asarray(self._paired_feedback, dtype=float)
            preds = pairs[:, 0]
            acts = pairs[:, 1]
            abs_err = np.abs(preds - acts)
            live_mae = float(np.mean(abs_err))
            live_rmse = float(np.sqrt(np.mean((preds - acts) ** 2)))
            # Classification accuracy when targets are integer/boolean class labels (or min_accuracy SLA is enabled)
            is_discrete_labels = bool(np.all(np.isclose(acts, np.round(acts), atol=1e-9)))
            if is_discrete_labels:
                live_acc = float(np.mean(np.round(preds) == np.round(acts)))
                if self.min_accuracy is not None and live_acc < self.min_accuracy:
                    new_alerts.append(
                        AlertEvent(
                            alert_type="accuracy_degradation",
                            severity="critical",
                            message=f"Live accuracy ({live_acc:.3f}) fell below minimum SLA ({self.min_accuracy:.3f}).",
                            observed_value=round(live_acc, 4),
                            threshold=self.min_accuracy,
                        )
                    )
            if self.max_mae is not None and live_mae > self.max_mae:
                new_alerts.append(
                    AlertEvent(
                        alert_type="regression_error_breach",
                        severity="critical",
                        message=f"Live MAE ({live_mae:.4f}) exceeded maximum SLA ({self.max_mae:.4f}).",
                        observed_value=round(live_mae, 4),
                        threshold=self.max_mae,
                    )
                )

        self.alerts.extend(new_alerts)

        rollback_executed = False
        rolled_back_to_version: int | None = None
        rollback_error: str | None = None
        rollback_disabled = os.getenv("OMNIFORGE_AUTO_ROLLBACK_DISABLED", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        quality_breaches = [
            a
            for a in new_alerts
            if a.alert_type in {"accuracy_degradation", "regression_error_breach", "error_rate_breach"}
        ]
        if (
            auto_rollback
            and not rollback_disabled
            and registry is not None
            and quality_breaches
            and (
                len(self._paired_feedback) >= max(1, int(min_feedback_for_rollback))
                or any(a.alert_type == "error_rate_breach" for a in quality_breaches)
            )
        ):
            reason = "; ".join(a.message for a in quality_breaches)
            try:
                restored = registry.rollback_production(self.model_name, reason=reason)
                rollback_executed = True
                rolled_back_to_version = restored.version
            except Exception as exc:
                rollback_error = str(exc)
                logger.warning("auto_rollback_skipped_or_failed", model=self.model_name, error=rollback_error)

        return {
            "model_name": self.model_name,
            "status": "degraded" if new_alerts else "healthy",
            "total_requests": self._total_requests,
            "labeled_feedback_count": len(self._paired_feedback),
            "error_rate": round(err_rate, 4),
            "p50_latency_ms": round(p50, 3),
            "p95_latency_ms": round(p95, 3),
            "p99_latency_ms": round(p99, 3),
            "live_accuracy": round(live_acc, 4) if live_acc is not None else None,
            "live_mae": round(live_mae, 4) if live_mae is not None else None,
            "live_rmse": round(live_rmse, 4) if live_rmse is not None else None,
            "active_alerts": [a.model_dump() for a in new_alerts],
            "rollback_executed": rollback_executed,
            "rolled_back_to_version": rolled_back_to_version,
            "rollback_error": rollback_error,
        }
