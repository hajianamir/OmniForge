"""Studio 04 (Automated EDA & Hypothesis Lab) router."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from fastapi import APIRouter, Header, HTTPException

from omniforge.api.studios.schemas import HypothesisReq, wb_cache_and_respond
from omniforge.core.performance import smart_sample_df
from omniforge.eda import AutomatedEDA, StatisticalAnalyzer

router = APIRouter(tags=["workbench-s04"])


def _wb() -> Any:
    from omniforge.api import workbench as wb
    return wb


@router.post("/s04/eda")
def s04_run_eda(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    targets = _wb().get_selected_targets(state, df)
    sample_df, was_sampled = smart_sample_df(df, max_rows=25_000, stratify_col=targets[0] if targets else None)
    eda = AutomatedEDA().run(sample_df, target_column=targets[0] if targets else None, generate_figures=False)
    state["uap_eda_report"] = eda

    num_cols = list(sample_df.select_dtypes(include=[np.number]).columns)[:12]
    distributions: list[dict[str, Any]] = []
    for c in num_cols[:6]:
        vals = sample_df[c].dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 5:
            counts, edges = np.histogram(vals, bins=18)
            distributions.append(
                {
                    "column": c,
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                    "bins": [{"label": f"{edges[i]:.2g}", "count": int(counts[i])} for i in range(len(counts))],
                }
            )
    corr_matrix: dict[str, Any] = {"columns": [], "values": []}
    if len(num_cols) >= 2:
        cm = sample_df[num_cols].corr(numeric_only=True).fillna(0.0)
        corr_matrix = {"columns": list(cm.columns), "values": np.round(cm.to_numpy(dtype=float), 3).tolist()}

    missing_bars = [
        {"column": str(c), "missing_pct": round(float(df[c].isna().mean() * 100.0), 2)}
        for c in df.columns
        if df[c].isna().any()
    ][:20]
    insights = [i.model_dump() if hasattr(i, "model_dump") else i for i in getattr(eda, "insights", [])]
    _wb().record_lineage(state, "EDA", "Completed Automated EDA", f"Numeric={len(num_cols)} | Sampled={was_sampled}")
    return wb_cache_and_respond(
        state,
        sid,
        "s04_eda",
        {
            "sampled": was_sampled,
            "n_rows_analyzed": len(sample_df),
            "insights": insights,
            "distributions": distributions,
            "correlation_matrix": corr_matrix,
            "missing_bars": missing_bars,
        },
    )


def _run_single_stat_test(
    df: pd.DataFrame,
    test_type: str,
    col_a: str,
    col_b: str | None,
    alpha: float,
) -> dict[str, Any]:
    from scipy import stats as sp_stats

    t = test_type.lower()
    if t == "shapiro":
        res = StatisticalAnalyzer.normality_test(df[col_a].dropna().iloc[:5000], alpha=alpha)
        out = res.model_dump() if hasattr(res, "model_dump") else dict(res)
        out.setdefault("significant", bool(res.reject_null))
        return out
    if t == "ks_normality":
        res = StatisticalAnalyzer.normality_test(
            df[col_a].dropna().iloc[:5000], alpha=alpha, method="kolmogorov_smirnov"
        )
        return {
            "test_name": "Lilliefors / KS Normality Test",
            "statistic": round(float(res.statistic), 5),
            "p_value": float(res.p_value),
            "alpha": alpha,
            "significant": bool(res.reject_null),
            "interpretation": res.interpretation,
        }
    if t == "anderson":
        vals = pd.to_numeric(df[col_a], errors="coerce").dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5:
            raise HTTPException(status_code=400, detail="Anderson-Darling test requires at least 5 numeric observations.")
        res_ad = StatisticalAnalyzer.anderson_darling_test(vals[:5000], alpha=alpha)
        return {
            "test_name": "Anderson-Darling Normality Test",
            "statistic": round(float(res_ad.statistic), 5),
            "p_value": round(float(res_ad.p_value), 6),
            "alpha": alpha,
            "significant": bool(res_ad.reject_null),
            "interpretation": res_ad.interpretation,
        }
    if t == "dagostino":
        vals = pd.to_numeric(df[col_a], errors="coerce").dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 8:
            raise HTTPException(status_code=400, detail="D'Agostino-Pearson test requires at least 8 numeric observations.")
        stat, pval = sp_stats.normaltest(vals[:5000])
        return {
            "test_name": "D'Agostino-Pearson Omnibus K²",
            "statistic": round(float(stat), 5),
            "p_value": float(pval),
            "alpha": alpha,
            "significant": bool(pval < alpha),
            "interpretation": f"K²={stat:.4f}, p={pval:.4g} — {'significant skew/kurtosis deviation' if pval < alpha else 'passes omnibus normality'}.",
        }
    if t == "distribution_fit":
        vals = pd.to_numeric(df[col_a], errors="coerce").dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)][:4000]
        if len(vals) < 15:
            raise HTTPException(status_code=400, detail="Distribution fitting needs at least 15 values.")
        n_obs = len(vals)
        # Candidate tuple: (display_name, scipy_dist, fit_kwargs, k_free_params)
        candidates: list[tuple[str, Any, dict[str, Any], int]] = [
            ("Normal", sp_stats.norm, {}, 2),
            ("Student-t", sp_stats.t, {}, 3),
            ("Laplace", sp_stats.laplace, {}, 2),
            ("Logistic", sp_stats.logistic, {}, 2),
        ]
        if float(np.min(vals)) > 0.0:
            candidates.extend(
                [
                    ("Log-Normal", sp_stats.lognorm, {"floc": 0.0}, 2),
                    ("Exponential", sp_stats.expon, {"floc": 0.0}, 1),
                    ("Gamma", sp_stats.gamma, {"floc": 0.0}, 2),
                ]
            )
        fits: list[dict[str, Any]] = []
        for dname, dist, fit_kw, k_params in candidates:
            try:
                params = dist.fit(vals, **fit_kw)
                logpdf = np.asarray(dist.logpdf(vals, *params), dtype=float)
                if not np.all(np.isfinite(logpdf)):
                    logpdf = np.clip(np.nan_to_num(logpdf, nan=-700.0, neginf=-700.0, posinf=700.0), -700.0, 700.0)
                ll = float(np.sum(logpdf))
                aic = 2.0 * k_params - 2.0 * ll
                bic = k_params * float(np.log(n_obs)) - 2.0 * ll
                ks_s, ks_p_raw = sp_stats.kstest(vals, dist.cdf, args=params)
                # Use composite-hypothesis Lilliefors p-value for Normal instead of unadjusted KS after MLE
                if dname == "Normal":
                    lf_res = StatisticalAnalyzer.normality_test(
                        pd.Series(vals), alpha=alpha, method="kolmogorov_smirnov"
                    )
                    gof_p = float(lf_res.p_value) if np.isfinite(lf_res.p_value) else float(ks_p_raw)
                else:
                    gof_p = float(ks_p_raw)
                fits.append(
                    {
                        "distribution": dname,
                        "k_params": k_params,
                        "log_likelihood": round(ll, 3),
                        "aic": round(aic, 3),
                        "bic": round(bic, 3),
                        "ks_stat": round(float(ks_s), 5),
                        "p_value": round(gof_p, 5),
                        "good_fit": bool(gof_p >= alpha),
                    }
                )
            except Exception:
                continue
        # Rank by Akaike Information Criterion (Akaike 1974) to penalize parameter complexity fairly
        fits.sort(key=lambda r: (float(r["aic"]), float(r["ks_stat"])))
        best: dict[str, Any] = fits[0] if fits else {"distribution": "Normal", "aic": 0.0, "bic": 0.0, "ks_stat": 0.0, "p_value": 1.0}
        return {
            "test_name": f"Distribution Goodness-of-Fit ({col_a})",
            "statistic": best["ks_stat"],
            "p_value": best["p_value"],
            "alpha": alpha,
            "significant": bool(best["p_value"] < alpha),
            "interpretation": (
                f"Best-fitting family by AIC is {best['distribution']} "
                f"(AIC={best.get('aic', 0.0):.2f}, BIC={best.get('bic', 0.0):.2f}, KS={best['ks_stat']:.4f}, p={best['p_value']:.4g})."
            ),
            "extra_table": fits,
        }
    if t in ("pearson", "spearman", "kendall") and col_b and col_b in df.columns:
        res = StatisticalAnalyzer.correlation_test(df[col_a], df[col_b], method=cast(Any, t), alpha=alpha)
        return res.model_dump() if hasattr(res, "model_dump") else dict(res)
    if t == "chi_square" and col_b and col_b in df.columns:
        res = StatisticalAnalyzer.chi_square_independence(df[col_a], df[col_b], alpha=alpha)
        return res.model_dump() if hasattr(res, "model_dump") else dict(res)
    if t == "fisher_exact" and col_b and col_b in df.columns:
        sub_cat = df[[col_a, col_b]].dropna()
        ct = pd.crosstab(sub_cat[col_a], sub_cat[col_b])
        if ct.shape != (2, 2):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Fisher's Exact Test requires a strict 2x2 contingency table "
                    f"(got {ct.shape[0]}x{ct.shape[1]}); use 'chi_square' for R×C tables."
                ),
            )
        odds, pval = sp_stats.fisher_exact(ct.to_numpy())
        return {
            "test_name": "Fisher's Exact Test (2x2)",
            "statistic": round(float(odds), 5),
            "p_value": float(pval),
            "alpha": alpha,
            "significant": bool(pval < alpha),
            "interpretation": f"Odds Ratio={odds:.4f}, p={pval:.4g} ({'dependent' if pval < alpha else 'independent'}).",
        }
    if t in ("anova", "kruskal") and col_b and col_b in df.columns:
        res = StatisticalAnalyzer.multi_group_test(df, value_col=col_a, group_col=col_b, parametric=(t == "anova"), alpha=alpha)
        return res.model_dump() if hasattr(res, "model_dump") else dict(res)
    if t in ("levene", "bartlett", "fligner") and col_b and col_b in df.columns:
        groups = [pd.to_numeric(g[col_a], errors="coerce").dropna().to_numpy(dtype=float) for _, g in df.groupby(col_b)]
        groups = [g[np.isfinite(g)] for g in groups if len(g) >= 2]
        if len(groups) < 2:
            raise HTTPException(status_code=400, detail=f"Variance test requires >=2 groups in '{col_b}'.")
        fn = {"levene": sp_stats.levene, "bartlett": sp_stats.bartlett, "fligner": sp_stats.fligner}[t]
        stat, pval = fn(*groups[:12])
        return {
            "test_name": f"{t.title()} Homogeneity of Variance Test",
            "statistic": round(float(stat), 5),
            "p_value": float(pval),
            "alpha": alpha,
            "significant": bool(pval < alpha),
            "interpretation": f"Stat={stat:.4f}, p={pval:.4g} — {'heteroscedastic (unequal variances)' if pval < alpha else 'homoscedastic (equal variances)'}.",
        }
    if t in ("paired_t", "wilcoxon", "ks_2samp") and col_b and col_b in df.columns:
        sub = df[[col_a, col_b]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 5:
            raise HTTPException(status_code=400, detail="Need at least 5 paired numeric rows.")
        a_v, b_v = sub[col_a].to_numpy(dtype=float), sub[col_b].to_numpy(dtype=float)
        if t == "paired_t":
            stat, pval = sp_stats.ttest_rel(a_v, b_v)
            label = "Paired Student t-Test"
        elif t == "wilcoxon":
            diff = a_v - b_v
            if np.allclose(diff, 0):
                stat, pval = 0.0, 1.0
            else:
                stat, pval = sp_stats.wilcoxon(diff)
            label = "Wilcoxon Signed-Rank Test"
        else:
            stat, pval = sp_stats.ks_2samp(a_v, b_v)
            label = "Two-Sample Kolmogorov-Smirnov Test"
        return {
            "test_name": label,
            "statistic": round(float(stat), 5),
            "p_value": float(pval),
            "alpha": alpha,
            "significant": bool(pval < alpha),
            "interpretation": f"Stat={stat:.4f}, p={pval:.4g} ({'significant difference' if pval < alpha else 'no significant difference'}).",
        }
    if col_b and col_b in df.columns:
        uniq = df[col_b].dropna().unique()[:2]
        if len(uniq) < 2:
            raise HTTPException(status_code=400, detail=f"Group column '{col_b}' needs at least 2 distinct values.")
        g1 = df.loc[df[col_b] == uniq[0], col_a].dropna()
        g2 = df.loc[df[col_b] == uniq[1], col_a].dropna()
        res = StatisticalAnalyzer.two_sample_test(g1, g2, parametric=(t != "mann_whitney"), alpha=alpha)
        return res.model_dump() if hasattr(res, "model_dump") else dict(res)
    raise HTTPException(status_code=400, detail="Select both Column A and Column B for two-variable tests.")


@router.post("/s04/hypothesis")
def s04_run_hypothesis(req: HypothesisReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    mode = req.mode.lower()

    if mode in ("auto", "complete"):
        num_cols = list(df.select_dtypes(include=[np.number]).columns)[:6]
        cat_cols = [c for c in df.columns if c not in num_cols and 2 <= df[c].nunique(dropna=True) <= 15][:4]
        battery: list[dict[str, Any]] = []
        tests_norm = ["shapiro", "dagostino", "anderson", "distribution_fit"] if mode == "complete" else ["shapiro", "distribution_fit"]
        for nc in num_cols[:4]:
            for tt in tests_norm:
                try:
                    r = _run_single_stat_test(df, tt, nc, None, req.alpha)
                    battery.append({"scope": nc, "test_type": tt, **{k: v for k, v in r.items() if k != "extra_table"}})
                except Exception:
                    continue
        if len(num_cols) >= 2:
            corr_tests = ["pearson", "spearman", "kendall", "ks_2samp"] if mode == "complete" else ["spearman"]
            for i in range(min(3, len(num_cols))):
                for j in range(i + 1, min(4, len(num_cols))):
                    for tt in corr_tests:
                        try:
                            r = _run_single_stat_test(df, tt, num_cols[i], num_cols[j], req.alpha)
                            battery.append({"scope": f"{num_cols[i]} vs {num_cols[j]}", "test_type": tt, **{k: v for k, v in r.items() if k != "extra_table"}})
                        except Exception:
                            continue
        for nc in num_cols[:3]:
            for cc in cat_cols[:2]:
                grp_tests = ["anova", "kruskal", "levene"] if mode == "complete" else ["kruskal"]
                for tt in grp_tests:
                    try:
                        r = _run_single_stat_test(df, tt, nc, cc, req.alpha)
                        battery.append({"scope": f"{nc} by {cc}", "test_type": tt, **{k: v for k, v in r.items() if k != "extra_table"}})
                    except Exception:
                        continue
        if len(cat_cols) >= 2:
            try:
                r = _run_single_stat_test(df, "chi_square", cat_cols[0], cat_cols[1], req.alpha)
                battery.append({"scope": f"{cat_cols[0]} x {cat_cols[1]}", "test_type": "chi_square", **{k: v for k, v in r.items() if k != "extra_table"}})
            except Exception:
                pass
        # Benjamini-Hochberg FDR correction across the entire hypothesis battery
        m_tests = len(battery)
        if m_tests > 0:
            indexed_p = sorted(
                ((i, float(r.get("p_value", 1.0) or 1.0)) for i, r in enumerate(battery)),
                key=lambda x: x[1],
            )
            bh_adj = [1.0] * m_tests
            running_min = 1.0
            for rank_rev in range(m_tests - 1, -1, -1):
                orig_idx, p_val = indexed_p[rank_rev]
                adj = min(running_min, (p_val * m_tests) / (rank_rev + 1))
                running_min = adj
                bh_adj[orig_idx] = min(1.0, max(0.0, adj))
            for i, r in enumerate(battery):
                r["p_value_fdr"] = round(bh_adj[i], 6)
                r["significant_fdr"] = bool(bh_adj[i] < req.alpha)
        state["uap_hypothesis_battery"] = battery
        return wb_cache_and_respond(
            state,
            sid,
            "s04_hypothesis",
            {
                "mode": mode,
                "total_tests": len(battery),
                "significant_tests": sum(1 for r in battery if r.get("significant")),
                "battery": battery,
                "result": battery[0] if battery else None,
            },
        )

    if not req.col_a or req.col_a not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.col_a}' not found.")
    result = _run_single_stat_test(df, req.test_type, req.col_a, req.col_b, req.alpha)
    return wb_cache_and_respond(state, sid, "s04_hypothesis", {"mode": "custom", "result": result})


# ---------------------------------------------------------------------------
# Studio 05 — Feature Engineering & Selection
# ---------------------------------------------------------------------------
