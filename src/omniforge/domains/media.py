"""Media, Entertainment & AdTech Domain Plugin: CTR, Content Recommendations, Audience Engagement."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class MediaDomainPlugin(BaseDomainPlugin):
    """Enterprise Media, Streaming & AdTech Analytics plugin (GDPR / CCPA / COPPA)."""

    metadata = DomainMetadata(
        name="media",
        display_name="Media, Streaming & AdTech",
        description="Click-through rate (CTR) modeling, content recommendations, watch-time retention, and sentiment NLP.",
        compliance_frameworks=["GDPR", "CCPA", "COPPA"],
        kpis=["click_through_rate", "mean_watch_minutes", "completion_rate"],
        recommended_models=["hybrid_recommender", "nlp_engine", "gradient_boosting"],
        default_target="clicked",
        config_file="configs/domains/media.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"watch_minutes", "content_duration_min"}.issubset(out.columns):
            out["completion_ratio"] = np.clip(
                out["watch_minutes"] / np.maximum(out["content_duration_min"], 1.0), 0.0, 1.0
            )
        return out

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if "clicked" in df.columns:
            kpis["click_through_rate"] = round(float(df["clicked"].mean()), 4)
        if "watch_minutes" in df.columns:
            kpis["mean_watch_minutes"] = round(float(df["watch_minutes"].mean()), 2)
        if {"watch_minutes", "content_duration_min"}.issubset(df.columns):
            kpis["completion_rate"] = round(
                float(np.clip(df["watch_minutes"] / np.maximum(df["content_duration_min"], 1.0), 0, 1).mean()), 4
            )
        return kpis

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        duration = rng.uniform(5, 120, n_samples)
        relevance = rng.uniform(0.1, 1.0, n_samples)
        watch = np.clip(duration * relevance * rng.uniform(0.5, 1.15, n_samples), 0.5, duration)
        impressions = rng.integers(1, 25, n_samples)
        logit = -2.6 + 3.2 * relevance + 0.02 * watch - 0.05 * impressions
        clicked = (rng.uniform(size=n_samples) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
        return pd.DataFrame(
            {
                "content_duration_min": np.round(duration, 1),
                "watch_minutes": np.round(watch, 1),
                "relevance_score": np.round(relevance, 3),
                "prior_impressions": impressions,
                "clicked": clicked,
            }
        )


plugin = MediaDomainPlugin()
