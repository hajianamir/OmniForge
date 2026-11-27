"""Uncertainty of score differences for fixed models on a shared holdout."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def paired_score_interval(
    truth: Any,
    candidate: Any,
    reference: Any,
    *,
    metric: str,
    split_strategy: str | None,
    comparisons: int,
    reference_name: str,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired percentile bootstrap, conditional on fits, with Bonferroni family adjustment.

    Positive differences favor the candidate. Random holdouts resample rows; stratified holdouts
    resample within target strata and condition on their observed proportions. This does not
    measure variability from retraining or support clustered or serially dependent holdouts.
    """
    y, pred, ref = (np.asarray(values).ravel() for values in (truth, candidate, reference))
    n = len(y)
    result = {"status": "unavailable", "n_pairs": n, "metric": metric, "reference": reference_name}
    if split_strategy not in ("random", "stratified"):
        return {**result, "reason": "No verified random or stratified holdout design for the row bootstrap."}
    if not 20 <= n <= 10_000 or n != len(pred) or n != len(ref):
        return {**result, "reason": "Requires 20 to 10,000 aligned holdout predictions."}
    strata: list[np.ndarray] = []
    if metric == "f1_weighted":
        labels = pd.Index(pd.unique(np.concatenate([y, pred, ref])))
        y, pred, ref = (labels.get_indexer(values) for values in (y, pred, ref))
        observed_counts = np.bincount(y)
        active_counts = observed_counts[observed_counts > 0]
        if len(active_counts) < 2 or int(active_counts.min()) < 3:
            return {**result, "reason": "Requires at least two represented classes and three observed rows per class."}
        if split_strategy == "stratified":
            strata = [np.flatnonzero(y == value) for value in np.unique(y)]
        equal_candidate, equal_reference = y == pred, y == ref

        def delta(idx: np.ndarray) -> float:
            support = np.bincount(y[idx], minlength=len(labels))

            def f1(p: np.ndarray, hits: np.ndarray) -> float:
                denom = support + np.bincount(p[idx], minlength=len(labels))
                tp = np.bincount(y[idx[hits[idx]]], minlength=len(labels))
                return float(np.sum(support * 2 * tp / np.maximum(denom, 1)) / n)

            return f1(pred, equal_candidate) - f1(ref, equal_reference)

    elif metric in ("rmse", "r2"):
        y, pred, ref = (x.astype(float) for x in (y, pred, ref))
        if not all(np.isfinite(x).all() for x in (y, pred, ref)):
            return {**result, "reason": "Nonfinite outcomes or predictions."}
        candidate_error, reference_error = (y - pred) ** 2, (y - ref) ** 2

        def delta(idx: np.ndarray) -> float:
            a, b = candidate_error[idx].mean(), reference_error[idx].mean()
            if metric == "rmse":
                return float(np.sqrt(b) - np.sqrt(a))
            variance = y[idx].var()
            return float((b - a) / variance) if variance > 0 else float("nan")

    else:
        return {**result, "reason": "Unsupported comparison metric."}

    family_size = max(1, comparisons)
    draws = 2_000
    tail = (1.0 - confidence) / (2 * family_size)
    if tail * draws < 1:
        return {**result, "reason": "Insufficient bootstrap resolution for this family and confidence level."}
    rng = np.random.default_rng(seed)
    differences = np.empty(draws)
    for i in range(draws):
        idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata]) if strata else rng.integers(n, size=n)
        differences[i] = delta(idx)
    if not np.isfinite(differences).all():
        return {**result, "reason": "Score difference was undefined in at least one resample."}
    lower, upper = np.quantile(differences, [tail, 1.0 - tail])
    return {
        **result, "status": "available", "method": "paired_percentile_bootstrap_bonferroni",
        "resampling_unit": "stratified_row" if strata else "row", "family_size": family_size,
        "confidence_level": confidence, "bootstrap_draws": draws,
        "ci_lower": float(lower), "ci_upper": float(upper), "supports_improvement": bool(lower > 0),
    }
