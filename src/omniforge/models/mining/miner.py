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
        for column_index, col in enumerate(list(sample_df.columns)[:16]):
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
                        q_raw = pd.qcut(clean, q=3, duplicates="drop")
                        n_bins = len(q_raw.cat.categories)
                        if n_bins >= 2:
                            lvl_labels = ["Low", "High"] if n_bins == 2 else ["Low", "Mid", "High"][:n_bins]
                            q = q_raw.cat.rename_categories(lvl_labels)
                            for lvl in q.cat.categories:
                                item_cols[f"{col}={lvl}"] = (q == lvl).to_numpy(dtype=bool)
                        else:
                            # Heavily tied/zero-inflated numeric column: split around median/mode boundary
                            med_v = float(clean.median())
                            low_mask = (clean <= med_v).to_numpy(dtype=bool)
                            high_mask = (clean > med_v).to_numpy(dtype=bool)
                            if low_mask.any() and high_mask.any():
                                item_cols[f"{col}=Low"] = low_mask
                                item_cols[f"{col}=High"] = high_mask
                    except Exception as exc:
                        # Dataset headers and exception messages may contain private content.
                        logger.warning(
                            "association_binning_failed",
                            column_index=column_index,
                            error_type=type(exc).__name__,
                            fallback="continue_mining",
                        )
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
                cnt_ab = int(round(supp_ab * n))
                cnt_a = max(cnt_ab, int(round(supp_a * n)))
                cnt_b = max(cnt_ab, int(round(supp_b * n)))
                # Webb (2007, Machine Learning) exact one-sided hypergeometric p-value for positive rule association
                p_hyper = float(stats.hypergeom.sf(max(cnt_ab - 1, 0), n, min(cnt_b, n), min(cnt_a, n)))
                if not np.isfinite(p_hyper):
                    p_hyper = 1.0
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
                        "count": cnt_ab,
                        "p_value": round(p_hyper, 8),
                    }
                )

        if rules:
            from omniforge.eda.statistics import StatisticalAnalyzer

            q_vals = StatisticalAnalyzer.adjust_p_values([float(r["p_value"]) for r in rules], method="fdr_bh")
            for r_row, q_v in zip(rules, q_vals, strict=False):
                r_row["fdr_q_value"] = round(float(q_v), 8)
                r_row["significant"] = bool(q_v < 0.05)

        rules.sort(key=lambda r: (r["lift"], r["confidence"], r["support"]), reverse=True)
        itemsets_out = [
            {"itemset": " + ".join(names[x] for x in k), "size": len(k), "support": round(v, 4), "count": int(round(v * n))}
            for k, v in sorted(all_freq.items(), key=lambda kv: (len(kv[0]), kv[1]), reverse=True)[:40]
        ]

        logger.info("association_rules_mined", transactions=n, rules=len(rules[:max_rules]))
        return dict(
            to_json_safe(
                {
                    "n_transactions": n,
                    "n_items_evaluated": len(names),
                    "n_frequent_itemsets": len(all_freq),
                    "frequent_itemsets": itemsets_out,
                    "rules": rules[:max_rules],
                    "summary": f"Discovered {len(rules[:max_rules])} strong association rules (min support={min_support:.2f}, confidence={min_confidence:.2f}, lift>={min_lift:.2f}).",
                }
            )
        )


    @staticmethod
    def mine_transactions(
        df: pd.DataFrame,
        transaction_col: str,
        item_col: str,
        *,
        min_support: float = 0.05,
        min_confidence: float = 0.4,
        min_lift: float = 1.0,
        max_rules: int = 50,
    ) -> dict[str, Any]:
        """Transaction-ID-aware association rule mining with full item keys (no truncation)."""
        from collections import Counter
        from itertools import combinations

        if transaction_col not in df.columns or item_col not in df.columns:
            raise ValueError(f"Columns '{transaction_col}' and '{item_col}' must exist in DataFrame.")
        grouped = (
            df[[transaction_col, item_col]]
            .dropna()
            .groupby(transaction_col)[item_col]
            .apply(lambda s: frozenset(str(v) for v in s))
        )
        n_tx = len(grouped)
        if n_tx == 0:
            return {"n_transactions": 0, "rules": [], "n_rules": 0}

        item_counts: Counter[str] = Counter()
        for basket in grouped:
            for it in basket:
                item_counts[it] += 1

        freq_1 = {it: cnt / n_tx for it, cnt in item_counts.items() if (cnt / n_tx) >= min_support}
        freq_items = sorted(freq_1.keys())
        pair_counts: Counter[tuple[str, str]] = Counter()
        for basket in grouped:
            present = sorted(it for it in basket if it in freq_1)
            for a, b in combinations(present, 2):
                pair_counts[(a, b)] += 1

        rules: list[dict[str, Any]] = []
        for (a, b), cnt in pair_counts.items():
            sup_ab = cnt / n_tx
            if sup_ab < min_support:
                continue
            sup_a, sup_b = freq_1[a], freq_1[b]
            for ant, con, s_ant, s_con in ((a, b, sup_a, sup_b), (b, a, sup_b, sup_a)):
                conf = sup_ab / max(s_ant, 1e-12)
                lift = conf / max(s_con, 1e-12)
                if conf >= min_confidence and lift >= min_lift:
                    cnt_ant = int(item_counts[ant])
                    cnt_con = int(item_counts[con])
                    p_hyper = float(stats.hypergeom.sf(max(int(cnt) - 1, 0), n_tx, min(cnt_con, n_tx), min(cnt_ant, n_tx)))
                    if not np.isfinite(p_hyper):
                        p_hyper = 1.0
                    rules.append(
                        {
                            "antecedent": [ant],
                            "consequent": [con],
                            "support": round(sup_ab, 6),
                            "confidence": round(conf, 6),
                            "lift": round(lift, 6),
                            "count": int(cnt),
                            "p_value": round(p_hyper, 8),
                        }
                    )
        if rules:
            from omniforge.eda.statistics import StatisticalAnalyzer

            q_vals = StatisticalAnalyzer.adjust_p_values([float(r["p_value"]) for r in rules], method="fdr_bh")
            for r_row, q_v in zip(rules, q_vals, strict=False):
                r_row["fdr_q_value"] = round(float(q_v), 8)
                r_row["significant"] = bool(q_v < 0.05)
        rules.sort(key=lambda r: (r["lift"], r["confidence"], r["support"]), reverse=True)
        return {
            "n_transactions": int(n_tx),
            "n_frequent_1_itemsets": int(len(freq_items)),
            "rules": rules[:max_rules],
            "n_rules": int(len(rules)),
        }


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
                lift: float | None = (sub_mean / global_mean) if abs(global_mean) > 1e-9 else None
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
                        "lift": round(lift, 3) if lift is not None else None,
                        "wracc": round(wracc, 5),
                        "p_value": round(p_val, 6),
                        "significant": "Yes (p<0.05)" if p_val < 0.05 else "No",
                    }
                )

        if subgroups:
            from omniforge.eda.statistics import StatisticalAnalyzer

            adj_ps = StatisticalAnalyzer.adjust_p_values([float(r["p_value"]) for r in subgroups], method="fdr_bh")
            for row, adj_p in zip(subgroups, adj_ps, strict=False):
                row["adjusted_p_value"] = round(float(adj_p), 6)
                row["significant"] = "Yes (FDR<0.05)" if adj_p < 0.05 else "No"

        subgroups.sort(key=lambda r: (abs(r["wracc"]), -r["adjusted_p_value"]), reverse=True)
        return dict(
            to_json_safe(
                {
                    "target": target_col,
                    "metric_label": target_label,
                    "global_baseline": round(global_mean, 4),
                    "subgroups": subgroups[:max_subgroups],
                    "summary": f"Evaluated feature cohorts against {target_label} (global={global_mean:.4f}); top {min(len(subgroups), max_subgroups)} subgroups ranked by WRAcc & Welch significance.",
                }
            )
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
        exclude_cols: list[str] | None = None,
        max_points: int = 2500,
    ) -> dict[str, Any]:
        """Project numeric feature space into 2D coordinates with loadings/variance diagnostics."""
        m = method.lower()
        if m not in self.SUPPORTED_METHODS:
            m = "pca"
        excl = set(exclude_cols or []) | ({color_col} if color_col else set())
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excl]
        if len(num_cols) < 2:
            raise ValueError("Manifold projection requires at least 2 numeric feature columns.")

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

        trust: float | None = None
        if len(emb) >= 10:
            try:
                from sklearn.manifold import trustworthiness as _tw

                k_nb = min(5, max(1, len(emb) // 3))
                trust = round(float(_tw(X_s, emb, n_neighbors=k_nb)), 4)
            except Exception:
                trust = None

        scatter = compact_scatter_payload(emb[:, 0], emb[:, 1], groups=groups, max_points=cap)
        return dict(
            to_json_safe(
                {
                    "method": m.upper(),
                    "n_points": len(emb),
                    "was_sampled": was_sampled,
                    "trustworthiness": trust,
                    "explained_variance_pct": explained,
                    "total_explained_pct": round(sum(explained), 2) if explained else None,
                    "loadings": loadings[:20],
                    "scatter": scatter,
                    "x_label": f"{m.upper()} Component 1" + (f" ({explained[0]}%)" if len(explained) >= 1 else ""),
                    "y_label": f"{m.upper()} Component 2" + (f" ({explained[1]}%)" if len(explained) >= 2 else ""),
                }
            )
        )


class _FPNode:
    """Node in an FP-Tree."""

    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item: str | None, count: int, parent: _FPNode | None) -> None:
        self.item = item
        self.count = count
        self.parent = parent
        self.children: dict[str, _FPNode] = {}
        self.link: _FPNode | None = None


def fp_growth_mine(
    transactions: list[list[str]],
    *,
    min_support: float = 0.05,
    min_confidence: float = 0.3,
    min_lift: float = 1.0,
    max_rules: int = 50,
) -> dict[str, Any]:
    """Mine frequent itemsets and association rules using the FP-Growth algorithm."""
    n_tx = len(transactions)
    if n_tx == 0:
        return {"n_transactions": 0, "frequent_itemsets": [], "rules": []}
    min_count = max(1, int(np.ceil(min_support * n_tx)))

    def _build_tree(
        txs: list[tuple[list[str], int]],
    ) -> tuple[_FPNode, dict[str, tuple[int, _FPNode | None]]]:
        counts: dict[str, int] = {}
        for items, mult in txs:
            for it in set(items):
                counts[it] = counts.get(it, 0) + mult
        freq = {k: v for k, v in counts.items() if v >= min_count}
        header: dict[str, tuple[int, _FPNode | None]] = {k: (v, None) for k, v in freq.items()}
        root = _FPNode(None, 0, None)
        for items, mult in txs:
            ordered = sorted([it for it in set(items) if it in freq], key=lambda x: (-freq[x], x))
            curr = root
            for it in ordered:
                if it in curr.children:
                    curr.children[it].count += mult
                else:
                    node = _FPNode(it, mult, curr)
                    curr.children[it] = node
                    cnt_h, head_node = header[it]
                    if head_node is None:
                        header[it] = (cnt_h, node)
                    else:
                        p = head_node
                        while p.link is not None:
                            p = p.link
                        p.link = node
                curr = curr.children[it]
        return root, header

    freq_itemsets: dict[frozenset[str], int] = {}

    def _mine(header: dict[str, tuple[int, _FPNode | None]], prefix: frozenset[str]) -> None:
        sorted_items = sorted(header.items(), key=lambda kv: (kv[1][0], kv[0]))
        for item, (sup, node) in sorted_items:
            new_set = prefix | {item}
            freq_itemsets[new_set] = sup
            cond_txs: list[tuple[list[str], int]] = []
            curr = node
            while curr is not None:
                path: list[str] = []
                parent = curr.parent
                while parent is not None and parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent
                if path:
                    cond_txs.append((path, curr.count))
                curr = curr.link
            if cond_txs:
                _, cond_header = _build_tree(cond_txs)
                if cond_header:
                    _mine(cond_header, new_set)

    _, root_header = _build_tree([(tx, 1) for tx in transactions])
    _mine(root_header, frozenset())

    rules: list[dict[str, Any]] = []
    for iset, sup_ab in freq_itemsets.items():
        if len(iset) < 2:
            continue
        for item in iset:
            ant = iset - {item}
            con = frozenset({item})
            sup_a = freq_itemsets.get(ant, 0)
            sup_b = freq_itemsets.get(con, 0)
            if sup_a == 0 or sup_b == 0:
                continue
            conf = sup_ab / sup_a
            s_ab = sup_ab / n_tx
            s_b = sup_b / n_tx
            lift = conf / max(s_b, 1e-12)
            if conf >= min_confidence and lift >= min_lift:
                rules.append(
                    {
                        "antecedent": sorted(ant),
                        "consequent": [item],
                        "support": round(s_ab, 6),
                        "confidence": round(conf, 6),
                        "lift": round(lift, 6),
                    }
                )
    rules.sort(key=lambda r: (r["lift"], r["confidence"], r["support"]), reverse=True)
    return {
        "n_transactions": n_tx,
        "frequent_itemsets": [
            {"items": sorted(k), "support": round(v / n_tx, 6)}
            for k, v in sorted(freq_itemsets.items(), key=lambda kv: (-kv[1], sorted(kv[0])))
        ],
        "rules": rules[:max_rules],
    }


class SequentialPatternMiner:
    """Sequential pattern mining via PrefixSpan."""

    @staticmethod
    def prefix_span(
        sequences: list[list[str]],
        *,
        min_support: float = 0.2,
        max_len: int = 4,
    ) -> list[dict[str, Any]]:
        """Discover frequent sequential patterns using PrefixSpan projected databases."""
        n_seq = len(sequences)
        if n_seq == 0:
            return []
        min_count = max(1, int(np.ceil(min_support * n_seq)))
        results: list[dict[str, Any]] = []

        def _project(prefix: list[str], db: list[list[str]]) -> None:
            if len(prefix) >= max_len:
                return
            item_counts: dict[str, int] = {}
            for seq in db:
                for it in set(seq):
                    item_counts[it] = item_counts.get(it, 0) + 1
            for it, cnt in sorted(item_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                if cnt < min_count:
                    continue
                new_prefix = [*prefix, it]
                results.append(
                    {
                        "pattern": new_prefix,
                        "count": cnt,
                        "support": round(cnt / n_seq, 6),
                    }
                )
                proj_db: list[list[str]] = []
                for seq in db:
                    try:
                        idx = seq.index(it)
                        if idx + 1 < len(seq):
                            proj_db.append(seq[idx + 1 :])
                    except ValueError:
                        continue
                if proj_db:
                    _project(new_prefix, proj_db)

        _project([], sequences)
        results.sort(key=lambda r: (-r["support"], -len(r["pattern"]), r["pattern"]))
        return results


def subgroup_beam_search(
    df: pd.DataFrame,
    target_col: str,
    *,
    beam_width: int = 5,
    max_depth: int = 2,
    min_support: int = 10,
) -> list[dict[str, Any]]:
    """Multi-condition subgroup discovery using Beam Search."""
    if target_col not in df.columns or len(df) < max(20, min_support):
        return []
    y = pd.to_numeric(df[target_col], errors="coerce")
    valid = y.notna()
    work = df.loc[valid].copy()
    y_vals = y.loc[valid].to_numpy(dtype=float)
    if len(y_vals) < min_support:
        return []
    global_mean = float(np.mean(y_vals))

    # Build atomic selectors
    selectors: list[tuple[str, np.ndarray]] = []
    for col in work.columns:
        if col == target_col:
            continue
        ser = work[col]
        if pd.api.types.is_numeric_dtype(ser):
            med = float(pd.to_numeric(ser, errors="coerce").median())
            m_hi = (pd.to_numeric(ser, errors="coerce") > med).to_numpy()
            if int(m_hi.sum()) >= min_support:
                selectors.append((f"{col} > {med:.3g}", m_hi))
        else:
            for val in ser.astype(str).value_counts().head(4).index:
                m_eq = (ser.astype(str) == str(val)).to_numpy()
                if int(m_eq.sum()) >= min_support:
                    selectors.append((f"{col} == '{val}'", m_eq))

    def _score(mask: np.ndarray) -> tuple[float, float, float | None]:
        sub_y = y_vals[mask]
        sub_mean = float(np.mean(sub_y))
        lift = (sub_mean / global_mean) if abs(global_mean) > 1e-9 else None
        score = abs(sub_mean - global_mean) * float(np.sqrt(len(sub_y)))
        return score, sub_mean, lift

    beam: list[tuple[tuple[str, ...], np.ndarray, float, float, float | None]] = []
    for name, mask in selectors:
        sc, sm, lf = _score(mask)
        beam.append(((name,), mask, sc, sm, lf))
    beam.sort(key=lambda x: x[2], reverse=True)
    beam = beam[:beam_width]

    all_found = list(beam)
    for _depth in range(2, max_depth + 1):
        candidates: list[tuple[tuple[str, ...], np.ndarray, float, float, float | None]] = []
        seen: set[tuple[str, ...]] = set()
        for conds, mask, _, _, _ in beam:
            for name, s_mask in selectors:
                if name in conds:
                    continue
                new_conds = tuple(sorted([*conds, name]))
                if new_conds in seen:
                    continue
                seen.add(new_conds)
                comb = mask & s_mask
                if int(comb.sum()) < min_support:
                    continue
                sc, sm, lf = _score(comb)
                candidates.append((new_conds, comb, sc, sm, lf))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[2], reverse=True)
        beam = candidates[:beam_width]
        all_found.extend(beam)

    all_found.sort(key=lambda x: x[2], reverse=True)
    return [
        {
            "conditions": list(conds),
            "support": int(mask.sum()),
            "subgroup_mean": round(sm, 6),
            "lift": round(lf, 3) if lf is not None else None,
            "score": round(sc, 4),
        }
        for conds, mask, sc, sm, lf in all_found[: beam_width * max_depth]
    ]

