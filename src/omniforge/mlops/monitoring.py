"""Production Model Monitoring, SLA Alerting, and Prometheus Metrics Telemetry."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from omniforge.core.logging import get_logger

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
        min_accuracy: float = 0.75,
        max_error_rate: float = 0.02,
    ) -> None:
        self.model_name = model_name
        self.max_p95_latency_ms = max_p95_latency_ms
        self.min_accuracy = min_accuracy
        self.max_error_rate = max_error_rate
        self._latencies: list[float] = []
        self._predictions: list[float] = []
        self._actuals: list[float] = []
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
        """Record a single inference telemetry event."""
        self._total_requests += 1
        self._latencies.append(float(latency_ms))
        if failed:
            self._errors += 1
        if prediction is not None:
            self._predictions.append(float(prediction))
        if actual is not None and prediction is not None:
            self._actuals.append(float(actual))

    def evaluate_health(self) -> dict[str, Any]:
        """Compute SLA summary statistics and emit alerts for any threshold violations."""
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
        if self._actuals and len(self._actuals) == len(self._predictions):
            preds = np.asarray(self._predictions[-len(self._actuals) :])
            acts = np.asarray(self._actuals)
            live_acc = float(np.mean(np.round(preds) == np.round(acts)))
            if live_acc < self.min_accuracy:
                new_alerts.append(
                    AlertEvent(
                        alert_type="accuracy_degradation",
                        severity="critical",
                        message=f"Live accuracy ({live_acc:.3f}) fell below minimum SLA ({self.min_accuracy:.3f}).",
                        observed_value=round(live_acc, 4),
                        threshold=self.min_accuracy,
                    )
                )

        self.alerts.extend(new_alerts)
        return {
            "model_name": self.model_name,
            "status": "degraded" if new_alerts else "healthy",
            "total_requests": self._total_requests,
            "error_rate": round(err_rate, 4),
            "p50_latency_ms": round(p50, 3),
            "p95_latency_ms": round(p95, 3),
            "p99_latency_ms": round(p99, 3),
            "live_accuracy": round(live_acc, 4) if live_acc is not None else None,
            "active_alerts": [a.model_dump() for a in new_alerts],
        }
