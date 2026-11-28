"""Studio 04 (Automated EDA & Hypothesis Lab) router."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from fastapi import APIRouter, Header, HTTPException

from omniforge.api.studios.schemas import HypothesisReq, wb_cache_and_respond
from omniforge.core.performance import smart_sample_df
from omniforge.eda import AutomatedEDA, StatisticalAnalyzer
from omniforge.eda.shapiro_scope import SHAPIRO_CAP, shapiro_sample_disclosure

router = APIRouter(tags=["workbench-s04"])


def _wb() -> Any:
    from omniforge.api import workbench as wb
    return wb


@router.post("/s04/eda")
def s04_run_eda(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    active_df = _wb()._require_df(state)
    df = _wb().get_analysis_dataframe(state, active_df).replace([np.inf, -np.inf], np.nan)
    targets = _wb().get_selected_targets(state, active_df)
    target = targets[0] if targets else None
    sample_df, was_sampled = smart_sample_df(df, max_rows=25_000, stratify_col=target)
    eda = AutomatedEDA().run(sample_df, target_column=target, generate_figures=False)

    num_cols = list(sample_df.select_dtypes(include=[np.number]).columns)[:12]
    work_num_df = sample_df[num_cols].copy() if num_cols else pd.DataFrame(index=sample_df.index)
    if len(num_cols) < 6:
        for col in sample_df.columns:
            if col in num_cols:
                continue
            ser = sample_df[col]
            if pd.api.types.is_bool_dtype(ser) or pd.api.types.is_datetime64_any_dtype(ser):
                continue
            non_na = int(ser.notna().sum())
            if non_na == 0:
                continue
            coerced = pd.to_numeric(
                ser.astype("string").str.strip().str.replace(",", "", regex=False),
                errors="coerce",
            )
            if float(coerced.notna().sum()) / max(non_na, 1) >= 0.7:
                work_num_df[col] = coerced.astype(float)
                num_cols.append(col)
                if len(num_cols) >= 12:
                    break

    distributions: list[dict[str, Any]] = []
    for c in num_cols[:6]:
        vals = work_num_df[c].dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            lo, hi = float(vals.min()), float(vals.max())
            if lo == hi:
                bins = [{"label": f"{lo:.6g}", "count": len(vals)}]
            else:
                # Bin a scaled copy to keep very large finite ranges representable.
                scale = max(abs(lo), abs(hi))
                counts, edges = np.histogram(vals / scale, bins=min(18, len(vals)))
                bins = [
                    {"label": f"{edges[i] * scale:.6g}", "count": int(count)}
                    for i, count in enumerate(counts)
                ]
            distributions.append(
                {
                    "column": str(c),
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "min": round(lo, 4),
                    "max": round(hi, 4),
                    "bins": bins,
                }
            )

    categorical_distributions: list[dict[str, Any]] = []
    num_set = set(num_cols)
    cat_candidates = [c for c in sample_df.columns if c not in num_set][:6]
    for c in cat_candidates:
        ser = sample_df[c].dropna().astype(str).str.strip()
        ser = ser[ser != ""]
        if ser.empty:
            continue
        vc = ser.value_counts().head(12)
        if len(vc) == 0:
            continue
        categorical_distributions.append(
            {
                "column": str(c),
                "unique_count": int(ser.nunique()),
                "top_values": [{"label": str(lbl)[:32], "count": int(cnt)} for lbl, cnt in vc.items()],
            }
        )

    corr_matrix: dict[str, Any] = {"columns": [], "values": []}
    if len(num_cols) >= 2:
        # Undefined correlations (e.g. constant columns) are not zero.
        # JSON serialization converts the resulting NaNs to null.
        cm = work_num_df[num_cols].corr(numeric_only=True)
        corr_matrix = {"columns": [str(col) for col in cm.columns], "values": np.round(cm.to_numpy(dtype=float), 3).tolist()}

    missing_bars = [
        {"column": str(c), "missing_pct": round(float(df[c].isna().mean() * 100.0), 2)}
        for c in df.columns
        if df[c].isna().any()
    ][:20]
    insights = [i.model_dump() if hasattr(i, "model_dump") else i for i in getattr(eda, "insights", [])]
    state["uap_eda_report"] = eda
    _wb().record_lineage(state, "EDA", "Completed Automated EDA", f"Numeric={len(num_cols)} | Sampled={was_sampled}")
    return wb_cache_and_respond(
        state,
        sid,
        "s04_eda",
        {
            "sampled": was_sampled,
            "n_rows_analyzed": len(sample_df),
            "n_rows_available": len(df),
            "row_cap": 25_000,
            "display_limits": {"numeric_distributions": 6, "categorical_distributions": 6,
                               "categories_per_chart": 12, "correlation_columns": 12, "missing_columns": 20},
            "insights": insights,
            "distributions": distributions,
            "categorical_distributions": categorical_distributions,
            "correlation_matrix": corr_matrix,
            "missing_bars": missing_bars,
        },
    )


def _scope(available: int, used: int, cap: int, *, unit: str, selection: str) -> dict[str, Any]:
    """Expose analysis caps independently of the display/table limit."""
    return {
        "analysis_scope": {"unit": unit, "available": available, "used": used,
                           "omitted": available - used, "cap": cap, "selection": selection},
        "sample_disclosure": f"{available:,} {unit} available / {used:,} used / {available - used:,} omitted; cap {cap:,}",
    }


def _studio_shapiro_disclosure(
    series: pd.Series, details: dict[str, Any] | None = None, *, cap_applied: bool = False
) -> dict[str, Any]:
    available = int(pd.to_numeric(series, errors="coerce").count())
    # The existing route caps non-missing rows before the analyzer coerces numeric strings.
    # Count cap omissions separately from invalid inputs and tests that could not run.
    cap_omissions = (
        int(pd.to_numeric(series.dropna().iloc[SHAPIRO_CAP:], errors="coerce").count()) if cap_applied else 0
    )
    tested = int((details or {}).get("n_scanned", 0))
    scope, note = shapiro_sample_disclosure(
        source_rows=len(series),
        available_rows=available,
        tested_rows=tested,
        cap_omitted_rows=cap_omissions,
        selection="first_non_missing_in_study_order",
    )
    return {
        # Keep the compact description ahead of wide test details in report exports.
        "sample_disclosure": note,
        "shapiro_sample": scope,
        "details": {**(details or {}), "sampled_from": available, "n_scanned": tested},
    }


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
        series = df[col_a]
        res = StatisticalAnalyzer.normality_test(series.dropna().iloc[:SHAPIRO_CAP], alpha=alpha, method="shapiro_wilk")
        out = res.model_dump(exclude={"details"})
        return {**_studio_shapiro_disclosure(series, res.details, cap_applied=True), **out, "significant": bool(res.reject_null)}
    if t == "ks_normality":
        series = df[col_a].dropna()
        sample = series.iloc[:5000]
        res = StatisticalAnalyzer.normality_test(sample, alpha=alpha, method="kolmogorov_smirnov")
        return {
            **_scope(int(pd.to_numeric(series, errors="coerce").count()), int(res.details.get("n_scanned", 0)),
                     5000, unit="observations", selection="first non-missing rows in study order"),
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
            **_scope(len(vals), min(len(vals), 5000), 5000, unit="observations", selection="first finite numeric rows in study order"),
            "test_name": "Anderson-Darling Normality Test",
            "statistic": round(float(res_ad.statistic), 5),
            "p_value": float(res_ad.p_value),
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
            **_scope(len(vals), min(len(vals), 5000), 5000, unit="observations", selection="first finite numeric rows in study order"),
            "test_name": "D'Agostino-Pearson Omnibus K²",
            "statistic": round(float(stat), 5),
            "p_value": float(pval),
            "alpha": alpha,
            "significant": bool(pval < alpha),
            "interpretation": f"K²={stat:.4f}, p={pval:.4g} — {'significant skew/kurtosis deviation' if pval < alpha else 'passes omnibus normality'}.",
        }
    if t == "distribution_fit":
        vals = pd.to_numeric(df[col_a], errors="coerce").dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n_available = len(vals)
        vals = vals[:4000]
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
        fitted_candidates: dict[str, tuple[Any, dict[str, float]]] = {}
        for dname, dist, fit_kw, k_params in candidates:
            try:
                params = dist.fit(vals, **fit_kw)
                logpdf = np.asarray(dist.logpdf(vals, *params), dtype=float)
                if not np.all(np.isfinite(logpdf)):
                    continue
                ll = float(np.sum(logpdf))
                aic = 2.0 * k_params - 2.0 * ll
                bic = k_params * float(np.log(n_obs)) - 2.0 * ll
                ks_s = float(sp_stats.ks_1samp(vals, dist.cdf, args=params).statistic)
                fitted_candidates[dname] = (dist, {"loc": 0.0} if "floc" in fit_kw else {})
                fits.append(
                    {
                        "distribution": dname,
                        "k_params": k_params,
                        "log_likelihood": ll,
                        "aic": aic,
                        "bic": bic,
                        "ks_stat": ks_s,
                    }
                )
            except Exception:
                continue
        # Rank by Akaike Information Criterion (Akaike 1974) to penalize parameter complexity fairly
        fits.sort(key=lambda r: (float(r["aic"]), float(r["ks_stat"])))
        if not fits:
            raise HTTPException(status_code=400, detail="None of the candidate distributions could be fitted.")
        best = fits[0]
        dist, known_params = fitted_candidates[best["distribution"]]
        draws = 399
        gof = sp_stats.goodness_of_fit(
            dist, vals, known_params=known_params or None, statistic="ks",
            n_mc_samples=draws, rng=np.random.default_rng(42),
        )
        gof_p = float(gof.pvalue)
        best.update(p_value=gof_p, bootstrap_ks_stat=float(gof.statistic))
        return {
            **_scope(n_available, n_obs, 4000, unit="observations", selection="first finite numeric rows in study order"),
            "test_name": f"Distribution Goodness-of-Fit ({col_a})",
            "statistic": float(gof.statistic),
            "p_value": gof_p,
            "alpha": alpha,
            "significant": bool(gof_p < alpha),
            "p_value_scope": "exploratory_selected_family",
            "bootstrap_draws": draws,
            "n_observations": n_obs,
            "interpretation": (
                f"Best-fitting family by AIC is {best['distribution']} "
                f"(AIC={best['aic']:.2f}, BIC={best['bic']:.2f}). "
                f"The fitted-null KS bootstrap gives p={gof_p:.4g} using {draws} draws. "
                "Treat this result as exploratory because the same observations selected the family."
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
            **_scope(len(groups), min(len(groups), 12), 12, unit="groups", selection="first eligible groups in group-key order"),
            "test_name": f"{t.title()} Homogeneity of Variance Test",
            "statistic": round(float(stat), 5),
            "p_value": float(pval),
            "alpha": alpha,
            "significant": bool(pval < alpha),
            "interpretation": f"Stat={stat:.4f}, p={pval:.4g} — {'heteroscedastic (unequal variances)' if pval < alpha else 'homoscedastic (equal variances)'}.",
        }
    if t in ("paired_t", "wilcoxon", "ks_2samp") and col_b and col_b in df.columns:
        numeric = df[[col_a, col_b]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if t == "ks_2samp":
            a_v, b_v = (numeric[col].dropna().to_numpy(dtype=float) for col in (col_a, col_b))
            if min(len(a_v), len(b_v)) < 5:
                raise HTTPException(status_code=400, detail="Need at least 5 observed values in each independent sample.")
        else:
            paired = numeric.dropna()
            if len(paired) < 5:
                raise HTTPException(status_code=400, detail="Need at least 5 paired numeric rows.")
            a_v, b_v = (paired[col].to_numpy(dtype=float) for col in (col_a, col_b))
        if t == "paired_t":
            stat, pval = sp_stats.ttest_rel(a_v, b_v)
            label = "Paired Student t-Test"
        elif t == "wilcoxon":
            diff = a_v - b_v
            if np.all(diff == 0):
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
            "sample_sizes": [len(a_v), len(b_v)],
            "interpretation": f"Stat={stat:.4f}, p={pval:.4g} ({'significant difference' if pval < alpha else 'no significant difference'}).",
        }
    if col_b and col_b in df.columns:
        if t not in ("welch_t", "mann_whitney"):
            raise HTTPException(status_code=400, detail="Unknown statistical test.")
        uniq = df[col_b].dropna().unique()
        if len(uniq) != 2:
            raise HTTPException(status_code=400, detail=f"Group column '{col_b}' must have exactly two distinct values.")
        g1 = df.loc[df[col_b] == uniq[0], col_a].dropna()
        g2 = df.loc[df[col_b] == uniq[1], col_a].dropna()
        res = StatisticalAnalyzer.two_sample_test(g1, g2, parametric=(t != "mann_whitney"), alpha=alpha)
        return res.model_dump() if hasattr(res, "model_dump") else dict(res)
    raise HTTPException(status_code=400, detail="Select both Column A and Column B for two-variable tests.")


def _evaluate_stat_test(
    df: pd.DataFrame, test: str, col_a: str, col_b: str | None, alpha: float
) -> dict[str, Any]:
    single_column = test.lower() in {"shapiro", "ks_normality", "anderson", "dagostino", "distribution_fit"}
    if single_column:
        values = pd.to_numeric(df[col_a], errors="coerce").dropna()
        if not values.empty and values.min() == values.max():
            return {
                **(_studio_shapiro_disclosure(df[col_a]) if test.lower() == "shapiro" else {}),
                "test_name": test,
                "statistic": None,
                "p_value": None,
                "alpha": alpha,
                "evaluable": False,
                "significant": False,
                "interpretation": "The selected column has no numeric variation.",
            }
    elif col_b not in df.columns or col_b == col_a:
        raise HTTPException(status_code=400, detail="Select two distinct columns included in the current study.")
    try:
        result = _run_single_stat_test(df, test, col_a, col_b, alpha)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="The selected data do not meet this test's requirements.") from exc
    p_value = result.get("p_value")
    if p_value is None or not np.isfinite(float(p_value)):
        return {
            **result,
            "evaluable": False,
            "significant": False,
            "interpretation": (
                result["interpretation"]
                if result.get("test_name") == "not_evaluable"
                else "This test did not produce a finite p-value for the selected data."
            ),
        }
    result["evaluable"] = True
    result["significant"] = float(p_value) < alpha
    return result


@router.post("/s04/hypothesis")
def s04_run_hypothesis(req: HypothesisReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state)).replace([np.inf, -np.inf], np.nan)
    mode = req.mode.lower()

    if mode in ("auto", "complete"):
        numeric = list(df.select_dtypes(include=[np.number]).columns)
        num_cols = numeric[:6]
        cat_cols = [c for c in df.columns if c not in numeric and 2 <= df[c].nunique(dropna=True) <= 15][:4]
        battery: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        def evaluate(test: str, col_a: str, col_b: str | None, scope: str) -> None:
            entry = {"scope": scope, "test_type": test}
            try:
                result = _evaluate_stat_test(df, test, col_a, col_b, req.alpha)
            except HTTPException as exc:
                if exc.status_code != 400:
                    raise
                skipped.append({**entry, "reason": exc.detail})
                return
            if not result["evaluable"]:
                scope_keys = ("sample_disclosure", "analysis_scope", "shapiro_sample")
                skipped.append({**entry, "reason": result["interpretation"], **{
                    key: result[key] for key in scope_keys if key in result
                }})
                return
            battery.append({**entry, **{k: v for k, v in result.items() if k != "extra_table"}})

        tests_norm = ["shapiro", "dagostino", "anderson"] if mode == "complete" else ["shapiro"]
        for nc in num_cols[:4]:
            for tt in tests_norm:
                evaluate(tt, nc, None, nc)
        corr_tests = ["pearson", "spearman", "kendall"] if mode == "complete" else ["spearman"]
        for i, col_a in enumerate(num_cols[:3]):
            for col_b in num_cols[i + 1 : 4]:
                for tt in corr_tests:
                    evaluate(tt, col_a, col_b, f"{col_a} vs {col_b}")
        grp_tests = ["anova", "kruskal", "levene"] if mode == "complete" else ["kruskal"]
        for nc in num_cols[:3]:
            for cc in cat_cols[:2]:
                for tt in grp_tests:
                    evaluate(tt, nc, cc, f"{nc} by {cc}")
        if len(cat_cols) >= 2:
            evaluate("chi_square", cat_cols[0], cat_cols[1], f"{cat_cols[0]} x {cat_cols[1]}")
        # Report/UI tables derive their columns from the first row. Keep the disclosure
        # visible even if the first evaluable (or skipped) row was not a Shapiro test.
        if any("sample_disclosure" in row for row in battery):
            battery = [{"scope": row["scope"], "test_type": row["test_type"], "sample_disclosure": None, **row} for row in battery]
        if any("sample_disclosure" in row for row in skipped):
            skipped = [{"scope": row["scope"], "test_type": row["test_type"], "sample_disclosure": None, **row} for row in skipped]
        adjusted = StatisticalAnalyzer.adjust_p_values([float(row["p_value"]) for row in battery])
        for row, p_adj in zip(battery, adjusted, strict=True):
            row["correction_family"] = "studio04_evaluable_battery"
            row["significant_raw"] = row["significant"]
            row["p_value_fdr"] = p_adj
            row["significant_fdr"] = p_adj < req.alpha
            row["significant"] = row["significant_fdr"]
        state["uap_hypothesis_battery"] = battery
        return wb_cache_and_respond(
            state,
            sid,
            "s04_hypothesis",
            {
                "mode": mode,
                "total_tests": len(battery),
                "battery_scope": {"numeric_columns_available": len(numeric),
                                  "normality_column_cap": 4, "correlation_column_cap": 4,
                                  "group_value_column_cap": 3, "group_key_column_cap": 2},
                "significant_tests": sum(1 for r in battery if r.get("significant")),
                "raw_significant_tests": sum(1 for r in battery if r.get("significant_raw")),
                "decision_policy": "Benjamini–Hochberg across the evaluable battery",
                "correction_family": {"id": "studio04_evaluable_battery", "method": "fdr_bh", "n_evaluable": len(battery)},
                "battery": battery,
                "skipped": skipped,
                "result": battery[0] if battery else None,
            },
        )

    if not req.col_a or req.col_a not in df.columns:
        raise HTTPException(status_code=400, detail="Select a Column A included in the current study.")
    result = _evaluate_stat_test(df, req.test_type, req.col_a, req.col_b, req.alpha)
    return wb_cache_and_respond(state, sid, "s04_hypothesis", {"mode": "custom", "result": result})


# ---------------------------------------------------------------------------
# Studio 05 — Feature Engineering & Selection
# ---------------------------------------------------------------------------
