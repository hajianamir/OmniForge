"""Enterprise Data Mining Suite: Association Rule Mining (Apriori), Subgroup Discovery & Manifold Projection."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import MDS, TSNE, Isomap, SpectralEmbedding
from sklearn.preprocessing import StandardScaler

from omniforge.core.logging import get_logger
from omniforge.core.performance import compact_scatter_payload, smart_sample_df
from omniforge.core.utils import to_json_safe

logger = get_logger(__name__)


class AssociationRuleMiner:
    """Frequent Pattern & Association Rule Miner (Apriori-style) for mixed tabular datasets.

    Automatically discretizes numeric columns into readable quantile intervals (Low/Mid/High)
    and mines frequent itemsets & association rules ranked by Lift, Confidence, and Support.
    """

    def mine(
        self,
        df: pd.DataFrame,
        *,
        min_support: float = 0.08,
        min_confidence: float = 0.55,
        min_lift: float = 1.15,
        max_items: int = 24,
        max_rules: int = 50,
    ) -> dict[str, Any]:
        """Mine frequent itemsets and association rules on tabular data."""
        sample_df, _ = smart_sample_df(df, max_rows=10_000)
        n = len(sample_df)
        if n < 10:
            return {"n_transactions": n, "frequent_itemsets": [], "rules": [], "summary": "Insufficient rows for rule mining."}

        # Build binary item matrix from top categoricals + discretized numerics
        item_cols: dict[str, np.ndarray] = {}
        for col in list(sample_df.columns)[:16]:
            if len(item_cols) >= max_items:
                break
            ser = sample_df[col]
            if pd.api.types.is_numeric_dtype(ser):
                clean = pd.to_numeric(ser, errors="coerce").replace([np.inf, -np.inf], np.nan)
                if clean.nunique(dropna=True) <= 2:
                    vals = clean.dropna().unique()
                    for v in vals[:2]:
                        item_cols[f"{col}={v}"] = (clean == v).to_numpy(dtype=bool)
                elif clean.notna().sum() >= 10:
                    try:
                        q = pd.qcut(clean, q=3, labels=["Low", "Mid", "High"], duplicates="drop")
                        for lvl in q.cat.categories:
                            item_cols[f"{col}={lvl}"] = (q == lvl).to_numpy(dtype=bool)
                    except Exception:
                        pass
            else:
                top_cats = ser.dropna().astype(str).value_counts().head(4)
                for cat_val, cnt in top_cats.items():
                    if cnt / n >= min_support:
                        item_cols[f"{col}={str(cat_val)[:24]}"] = (ser.astype(str) == str(cat_val)).to_numpy(dtype=bool)

        if len(item_cols) < 2:
            return {"n_transactions": n, "frequent_itemsets": [], "rules": [], "summary": "Not enough frequent items found."}

        names = list(item_cols.keys())[:max_items]
        mat = np.column_stack([item_cols[k] for k in names]).astype(bool)

        # Level 1 frequent itemsets
        supp1 = mat.mean(axis=0)
        freq1: dict[tuple[int, ...], float] = {
            (i,): float(supp1[i]) for i in range(len(names)) if float(supp1[i]) >= min_support
        }
        idx1 = [k[0] for k in freq1]

        # Level 2 frequent itemsets
        freq2: dict[tuple[int, ...], float] = {}
        for i, j in combinations(idx1, 2):
            # Exclude mutually exclusive levels from the same base feature
            if names[i].split("=")[0] == names[j].split("=")[0]:
                continue
            s = float(np.mean(mat[:, i] & mat[:, j]))
            if s >= min_support:
                freq2[(i, j)] = s

        # Level 3 frequent itemsets (from frequent pairs)
        freq3: dict[tuple[int, ...], float] = {}
        top_pairs = sorted(freq2.items(), key=lambda kv: kv[1], reverse=True)[:120]
        seen3: set[tuple[int, int, int]] = set()
        for (i, j), _ in top_pairs:
            for k in idx1:
                if k <= j:
                    continue
                if names[k].split("=")[0] in {names[i].split("=")[0], names[j].split("=")[0]}:
                    continue
                trip = (i, j, k)
                if trip in seen3:
                    continue
                if (i, k) not in freq2 or (j, k) not in freq2:
                    continue
                seen3.add(trip)
                s = float(np.mean(mat[:, i] & mat[:, j] & mat[:, k]))
                if s >= min_support:
                    freq3[trip] = s

        all_freq = {**freq1, **freq2, **freq3}
        rules: list[dict[str, Any]] = []

        # Generate rules from 2-itemsets and 3-itemsets
        for itemset, supp_ab in {**freq2, **freq3}.items():
            for cons_idx in itemset:
                ant = tuple(x for x in itemset if x != cons_idx)
                supp_a = all_freq.get(ant)
                supp_b = freq1.get((cons_idx,))
                if not supp_a or not supp_b:
                    continue
                conf = supp_ab / max(supp_a, 1e-12)
                lift = conf / max(supp_b, 1e-12)
                if conf < min_confidence or lift < min_lift:
                    continue
                leverage = supp_ab - supp_a * supp_b
                conviction = (1.0 - supp_b) / max(1.0 - conf, 1e-6) if conf < 0.9999 else 999.0
                ant_str = " AND ".join(names[x] for x in ant)
                cons_str = names[cons_idx]
                rules.append(
                    {
                        "antecedent": ant_str,
                        "consequent": cons_str,
                        "rule": f"IF {ant_str}  ⇒  {cons_str}",
                        "support": round(supp_ab, 4),
                        "confidence": round(conf, 4),
                        "lift": round(lift, 4),
                        "leverage": round(leverage, 4),
                        "conviction": round(min(conviction, 999.0), 3),
                        "count": int(round(supp_ab * n)),
                    }
                )

        rules.sort(key=lambda r: (r["lift"], r["confidence"], r["support"]), reverse=True)
        itemsets_out = [
            {"itemset": " + ".join(names[x] for x in k), "size": len(k), "support": round(v, 4), "count": int(round(v * n))}
            for k, v in sorted(all_freq.items(), key=lambda kv: (len(kv[0]), kv[1]), reverse=True)[:40]
        ]

        logger.info("association_rules_mined", transactions=n, rules=len(rules[:max_rules]))
        return to_json_safe(
            {
                "n_transactions": n,
                "n_items_evaluated": len(names),
                "n_frequent_itemsets": len(all_freq),
                "frequent_itemsets": itemsets_out,
                "rules": rules[:max_rules],
                "summary": f"Discovered {len(rules[:max_rules])} strong association rules (min support={min_support:.2f}, confidence={min_confidence:.2f}, lift>={min_lift:.2f}).",
            }
        )


class SubgroupDiscoveryMiner:
    """Statistically validated Subgroup Discovery identifying feature cohorts with highest target lift."""

    def discover(
        self,
        df: pd.DataFrame,
        target_col: str,
        *,
        min_subgroup_size: int = 15,
        max_subgroups: int = 30,
    ) -> dict[str, Any]:
        """Discover high-impact feature subgroups ranked by statistical significance and lift."""
        if target_col not in df.columns:
            return {"target": target_col, "subgroups": [], "summary": f"Target '{target_col}' not found."}
        work = df.dropna(subset=[target_col]).copy()
        if len(work) < 20:
            return {"target": target_col, "subgroups": [], "summary": "Insufficient rows for subgroup discovery."}

        y_raw = work[target_col]
        if pd.api.types.is_numeric_dtype(y_raw):
            y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
            valid_mask = np.isfinite(y)
            work = work.loc[valid_mask]
            y = y[valid_mask]
            target_label = f"Mean({target_col})"
        else:
            pos_class = str(y_raw.value_counts().index[0])
            y = (y_raw.astype(str) == pos_class).to_numpy(dtype=float)
            target_label = f"Rate({target_col}={pos_class})"

        global_mean = float(np.mean(y))
        n_total = len(y)
        subgroups: list[dict[str, Any]] = []

        for col in [c for c in work.columns if c != target_col][:25]:
            ser = work[col]
            bins: list[tuple[str, np.ndarray]] = []
            if pd.api.types.is_numeric_dtype(ser):
                num = pd.to_numeric(ser, errors="coerce")
                try:
                    q = pd.qcut(num, q=4, duplicates="drop")
                    for iv in q.cat.categories:
                        bins.append((f"{col} ∈ {iv}", (q == iv).to_numpy(dtype=bool)))
                except Exception:
                    continue
            else:
                for val, cnt in ser.dropna().astype(str).value_counts().head(8).items():
                    if cnt >= min_subgroup_size:
                        bins.append((f"{col} == '{str(val)[:28]}'", (ser.astype(str) == str(val)).to_numpy(dtype=bool)))

            for cond, mask in bins:
                cnt = int(mask.sum())
                if cnt < min_subgroup_size or cnt > n_total - 5:
                    continue
                in_vals = y[mask]
                out_vals = y[~mask]
                sub_mean = float(np.mean(in_vals))
                diff = sub_mean - global_mean
                lift = sub_mean / global_mean if abs(global_mean) > 1e-9 else 1.0
                try:
                    _, p_val = stats.ttest_ind(in_vals, out_vals, equal_var=False)
                    p_val = float(p_val)
                except Exception:
                    p_val = 1.0
                if not np.isfinite(p_val):
                    p_val = 1.0
                # WRAcc (Weighted Relative Accuracy) quality score
                wracc = (cnt / n_total) * diff
                subgroups.append(
                    {
                        "subgroup_rule": cond,
                        "size": cnt,
                        "coverage_pct": round(cnt / n_total * 100.0, 2),
                        "subgroup_value": round(sub_mean, 4),
                        "global_value": round(global_mean, 4),
                        "delta": round(diff, 4),
                        "lift": round(lift, 3),
                        "wracc": round(wracc, 5),
                        "p_value": round(p_val, 6),
                        "significant": "Yes (p<0.05)" if p_val < 0.05 else "No",
                    }
                )

        subgroups.sort(key=lambda r: (abs(r["wracc"]), -r["p_value"]), reverse=True)
        return to_json_safe(
            {
                "target": target_col,
                "metric_label": target_label,
                "global_baseline": round(global_mean, 4),
                "subgroups": subgroups[:max_subgroups],
                "summary": f"Evaluated feature cohorts against {target_label} (global={global_mean:.4f}); top {min(len(subgroups), max_subgroups)} subgroups ranked by WRAcc & Welch significance.",
            }
        )


class ManifoldProjector:
    """2D Dimensionality Reduction & Manifold Projection (PCA, SVD, t-SNE, Isomap, MDS, Spectral)."""

    SUPPORTED_METHODS: tuple[str, ...] = ("pca", "svd", "tsne", "isomap", "mds", "spectral")

    def project(
        self,
        df: pd.DataFrame,
        *,
        method: str = "pca",
        color_col: str | None = None,
        max_points: int = 2500,
    ) -> dict[str, Any]:
        """Project numeric feature space into 2D coordinates with loadings/variance diagnostics."""
        m = method.lower()
        if m not in self.SUPPORTED_METHODS:
            m = "pca"
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        if len(num_cols) < 2:
            raise ValueError("Manifold projection requires at least 2 numeric columns.")

        # Non-linear manifold embeddings scale O(N^2), cap sample cleanly for interactive speed
        cap = max_points if m in ("pca", "svd") else min(max_points, 700)
        sample_df, was_sampled = smart_sample_df(df, max_rows=cap, stratify_col=color_col if color_col in df.columns else None)
        X_df = sample_df[num_cols].replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(X_df.median(numeric_only=True).fillna(0.0))
        X_s = StandardScaler().fit_transform(X_df.to_numpy(dtype=float))

        explained: list[float] = []
        loadings: list[dict[str, Any]] = []
        if m == "pca":
            model = PCA(n_components=2, random_state=42)
            emb = model.fit_transform(X_s)
            explained = [round(float(v * 100.0), 2) for v in model.explained_variance_ratio_]
            for idx, col in enumerate(num_cols):
                loadings.append(
                    {
                        "feature": col,
                        "PC1_loading": round(float(model.components_[0, idx]), 4),
                        "PC2_loading": round(float(model.components_[1, idx]), 4),
                        "magnitude": round(float(np.hypot(model.components_[0, idx], model.components_[1, idx])), 4),
                    }
                )
            loadings.sort(key=lambda r: r["magnitude"], reverse=True)
        elif m == "svd":
            model = TruncatedSVD(n_components=2, random_state=42)
            emb = model.fit_transform(X_s)
            explained = [round(float(v * 100.0), 2) for v in model.explained_variance_ratio_]
        elif m == "tsne":
            perp = min(30.0, max(5.0, (len(X_s) - 1) / 3.0))
            emb = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto").fit_transform(X_s)
        elif m == "isomap":
            n_nb = min(12, max(3, len(X_s) - 2))
            emb = Isomap(n_components=2, n_neighbors=n_nb).fit_transform(X_s)
        elif m == "mds":
            emb = MDS(n_components=2, random_state=42, normalized_stress="auto", n_init=2, max_iter=150).fit_transform(X_s)
        else:
            n_nb = min(10, max(3, len(X_s) - 2))
            emb = SpectralEmbedding(n_components=2, n_neighbors=n_nb, random_state=42).fit_transform(X_s)

        groups = None
        if color_col and color_col in sample_df.columns:
            ser = sample_df[color_col]
            if ser.nunique(dropna=True) <= 12:
                codes = pd.Categorical(ser.fillna("Missing").astype(str)).codes
                groups = codes
            else:
                try:
                    groups = pd.qcut(pd.to_numeric(ser, errors="coerce"), q=4, labels=False, duplicates="drop").fillna(0).astype(int).to_numpy()
                except Exception:
                    groups = None

        scatter = compact_scatter_payload(emb[:, 0], emb[:, 1], groups=groups, max_points=cap)
        return to_json_safe(
            {
                "method": m.upper(),
                "n_points": len(emb),
                "was_sampled": was_sampled,
                "explained_variance_pct": explained,
                "total_explained_pct": round(sum(explained), 2) if explained else None,
                "loadings": loadings[:20],
                "scatter": scatter,
                "x_label": f"{m.upper()} Component 1" + (f" ({explained[0]}%)" if len(explained) >= 1 else ""),
                "y_label": f"{m.upper()} Component 2" + (f" ({explained[1]}%)" if len(explained) >= 2 else ""),
            }
        )
