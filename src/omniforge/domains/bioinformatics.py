"""Bioinformatics & Life Sciences Domain Plugin: Gene Expression, Biomarkers, Drug-Response Prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class BioinformaticsDomainPlugin(BaseDomainPlugin):
    """Genomics, Proteomics & Life Sciences Analytics plugin (HIPAA / GxP / FAIR-Data)."""

    metadata = DomainMetadata(
        name="bioinformatics",
        display_name="Bioinformatics & Genomics",
        description="High-dimensional gene expression modeling, biomarker selection, molecular response, and survival curves.",
        compliance_frameworks=["HIPAA", "GxP", "FAIR-Data"],
        kpis=["responder_rate", "mean_log2_fold_change", "biomarker_signal_snr"],
        recommended_models=["elastic_net", "random_forest", "cox_ph", "feature_selector"],
        default_target="drug_response",
        config_file="configs/domains/bioinformatics.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        gene_cols = [c for c in out.columns if c.startswith("gene_")]
        if len(gene_cols) >= 2:
            out["gene_signature_mean"] = out[gene_cols].mean(axis=1)
            out["gene_signature_std"] = out[gene_cols].std(axis=1).fillna(0.0)
        return out

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if "drug_response" in df.columns:
            kpis["responder_rate"] = round(float(df["drug_response"].mean()), 4)
        gene_cols = [c for c in df.columns if c.startswith("gene_")]
        if gene_cols:
            kpis["mean_log2_fold_change"] = round(float(df[gene_cols].mean().mean()), 4)
        return kpis

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        data = {f"gene_{i:02d}": np.round(rng.normal(0.0, 1.5, n_samples), 4) for i in range(1, 9)}
        logit = -0.3 + 1.1 * data["gene_01"] - 0.9 * data["gene_02"] + 0.6 * data["gene_03"]
        data["drug_response"] = (rng.uniform(size=n_samples) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
        return pd.DataFrame(data)


plugin = BioinformaticsDomainPlugin()
