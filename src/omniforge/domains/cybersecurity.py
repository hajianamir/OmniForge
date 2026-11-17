"""Cybersecurity & Threat Intelligence Domain Plugin: Intrusion Detection, UEBA, Anomaly Scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class CybersecurityDomainPlugin(BaseDomainPlugin):
    """Enterprise Cybersecurity & SOC Threat Analytics plugin (NIST-CSF / ISO-27001 / SOC2)."""

    metadata = DomainMetadata(
        name="cybersecurity",
        display_name="Cybersecurity & Threat Detection",
        description="Network intrusion detection, UEBA anomaly scoring, phishing NLP detection, and lateral-movement graphs.",
        compliance_frameworks=["NIST-CSF", "ISO-27001", "SOC2"],
        kpis=["intrusion_rate", "mean_bytes_ratio", "high_entropy_events_pct"],
        recommended_models=["anomaly_detector", "gradient_boosting", "graph_analytics"],
        default_target="is_intrusion",
        config_file="configs/domains/cybersecurity.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"bytes_out", "bytes_in"}.issubset(out.columns):
            out["exfiltration_ratio"] = out["bytes_out"] / np.maximum(out["bytes_in"], 1.0)
        if {"failed_logins", "session_duration_s"}.issubset(out.columns):
            out["failed_login_velocity"] = out["failed_logins"] / np.maximum(out["session_duration_s"] / 60.0, 0.1)
        return out

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if "is_intrusion" in df.columns:
            kpis["intrusion_rate"] = round(float(df["is_intrusion"].mean()), 4)
        if {"bytes_out", "bytes_in"}.issubset(df.columns):
            kpis["mean_bytes_ratio"] = round(float((df["bytes_out"] / np.maximum(df["bytes_in"], 1.0)).mean()), 3)
        return kpis

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        b_in = rng.lognormal(8.0, 1.1, n_samples)
        b_out = rng.lognormal(7.5, 1.3, n_samples)
        failed = rng.poisson(0.8, n_samples)
        dur = rng.uniform(5, 1200, n_samples)
        entropy = rng.uniform(2.5, 7.9, n_samples)
        logit = -4.2 + 0.9 * failed + 0.45 * (entropy - 5.0) + 0.6 * np.log1p(b_out / np.maximum(b_in, 1.0))
        intrusion = (rng.uniform(size=n_samples) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
        return pd.DataFrame(
            {
                "bytes_in": np.round(b_in, 1),
                "bytes_out": np.round(b_out, 1),
                "failed_logins": failed,
                "session_duration_s": np.round(dur, 1),
                "payload_entropy": np.round(entropy, 3),
                "is_intrusion": intrusion,
            }
        )


plugin = CybersecurityDomainPlugin()
