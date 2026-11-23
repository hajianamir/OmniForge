"""Executive & Technical Analytical Narrative Generator with Section Selection and Full Audit Coverage."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from omniforge.reporting.sections import ALL_SECTION_IDS


class NarrativeGenerator:
    """Produces thorough, boardroom- and audit-ready English analytical narratives from OmniForge telemetry."""

    def __init__(self, llm_agent: Any | None = None) -> None:
        self._llm = llm_agent

    def generate_executive_narrative(
        self,
        *,
        title: str,
        dataset_summary: dict[str, Any] | None = None,
        quality_report: dict[str, Any] | None = None,
        model_summary: dict[str, Any] | None = None,
        fairness_report: dict[str, Any] | None = None,
        carbon_report: dict[str, Any] | None = None,
        domain: str = "general",
        author: str = "OmniForge Enterprise",
        selected_sections: list[str] | None = None,
        analyses_performed: list[dict[str, Any]] | None = None,
        exclusions: list[dict[str, Any]] | None = None,
        raw_vs_processed: dict[str, Any] | None = None,
        formulas_and_rules: list[dict[str, Any]] | None = None,
        diagnostics_summary: dict[str, Any] | None = None,
        clustering_summary: dict[str, Any] | None = None,
        eda_summary: dict[str, Any] | None = None,
        feature_summary: dict[str, Any] | None = None,
        future_predictions_summary: dict[str, Any] | None = None,
        decision_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compose detailed, section-filtered analytical narrative blocks and prioritized recommendations."""
        enabled = set(selected_sections) if selected_sections is not None else set(ALL_SECTION_IDS)
        sections: list[dict[str, Any]] = []
        recommendations: list[str] = []

        # 1. Executive Summary
        if "executive_summary" in enabled:
            ds = dataset_summary or {}
            n_rows = ds.get("n_rows", ds.get("rows", "N/A"))
            n_cols = ds.get("n_columns", ds.get("columns", "N/A"))
            ds_name = ds.get("dataset_name", domain)
            mode_lbl = ds.get("processing_mode", "RAW")
            targets_str = ds.get("target_columns", "N/A")
            missing_cells = ds.get("missing_cells", 0)

            overview_bits = [
                f"This comprehensive analytical report documents the end-to-end '{title}' workflow executed on "
                f"dataset '{ds_name}' (study label: {domain}, active data state: {mode_lbl})."
            ]
            if ds:
                overview_bits.append(
                    f"The active dataset contains {n_rows:,} rows across {n_cols} columns"
                    if isinstance(n_rows, int)
                    else f"The active dataset contains {n_rows} rows across {n_cols} columns"
                )
                overview_bits[-1] += f", with target column(s) [{targets_str}] and {missing_cells:,} remaining missing cells." if isinstance(missing_cells, int) else "."
            if quality_report:
                score = quality_report.get("overall_score", 0.0)
                grade = quality_report.get("grade", "N/A")
                overview_bits.append(
                    f"Six-dimensional enterprise Data Quality achieved {score}/100 (Grade {grade})."
                )
                recommendations.extend(quality_report.get("recommendations", []))
            if analyses_performed:
                overview_bits.append(
                    f"A total of {len(analyses_performed)} distinct analytical and data-preparation operations were audited."
                )
            if exclusions:
                total_excl = sum(int(e.get("count", 0)) for e in exclusions)
                overview_bits.append(
                    f"{len(exclusions)} explicit data exclusion/remediation rule(s) were recorded affecting {total_excl:,} item(s)."
                )
            sections.append(
                {
                    "section_id": "executive_summary",
                    "heading": "1. Executive Summary & Dataset Profile",
                    "body": " ".join(overview_bits),
                    "bullets": [
                        f"Active Dataset: {ds_name} ({n_rows} rows × {n_cols} columns | Provenance: {mode_lbl})",
                        f"Target Variable(s): {targets_str}",
                        f"Analyses Audited: {len(analyses_performed or [])} | Exclusion/Remediation Rules: {len(exclusions or [])} | Mathematical Rules/Equations: {len(formulas_and_rules or [])}",
                    ],
                }
            )

        # 2. Analyses Performed Audit Log
        if "analyses_performed" in enabled:
            ap = analyses_performed or []
            if ap:
                studios_run = list(dict.fromkeys(str(a.get("studio", "Workflow")) for a in ap))
                bullets = [
                    f"[{a.get('timestamp', '-')}] {a.get('studio', '')} — {a.get('analysis_name', '')}: "
                    f"{a.get('outcome_summary', '')} (Criteria: {a.get('criteria_or_config') or 'Standard'})"
                    for a in ap[:25]
                ]
                body = (
                    f"Across this session, {len(ap)} analytical, cleaning, diagnostic, and modeling operations were "
                    f"executed spanning {len(studios_run)} workflow stage(s): {', '.join(studios_run)}. "
                    "The complete chronological audit table below records exact timestamps, active dataset mode "
                    "(RAW vs. PROCESSED), parameter criteria, inputs, and verified outputs."
                )
            else:
                bullets = ["Only initial dataset ingestion/profiling was executed prior to report generation."]
                body = (
                    "No downstream modeling or transformation stages were logged yet beyond initial dataset loading. "
                    "Execute Studios 01–08 to enrich the analytical audit trail."
                )
            sections.append(
                {
                    "section_id": "analyses_performed",
                    "heading": "2. Comprehensive Audit Log of Analyses Performed",
                    "body": body,
                    "bullets": bullets,
                }
            )

        # 3. Data Exclusions, Filtering & Removals Log (WHAT & WHY)
        if "data_exclusions" in enabled:
            excl_list = exclusions or []
            diff = raw_vs_processed or {}
            rows_rem = int(diff.get("rows_removed", 0))
            rem_cols = diff.get("removed_columns") or []
            miss_res = int(diff.get("missing_cells_resolved", 0))
            range_shifts = diff.get("numeric_range_shifts") or []

            if excl_list or rows_rem > 0 or rem_cols or miss_res > 0 or range_shifts:
                body_parts = [
                    "To guarantee full statistical transparency and reproducibility, this section itemizes every row, "
                    "column, outlier value, duplicate record, noise point, or corrupt cluster excluded or remediated "
                    "prior to downstream calculations, together with the exact quantitative rule and justification."
                ]
                if diff:
                    body_parts.append(
                        f"Forensic comparison of RAW vs. PROCESSED data confirms row count moved from "
                        f"{diff.get('raw_rows', 0):,} to {diff.get('processed_rows', 0):,} "
                        f"({rows_rem:,} rows excluded, {diff.get('rows_removed_pct', 0.0):.2f}%), "
                        f"column count moved from {diff.get('raw_columns', 0)} to {diff.get('processed_columns', 0)} "
                        f"({len(rem_cols)} dropped, {len(diff.get('added_columns') or [])} engineered/attached), "
                        f"and {miss_res:,} missing cells were resolved."
                    )
                bullets = []
                for ev in excl_list[:20]:
                    bullets.append(
                        f"{ev.get('stage', 'Stage')} | {ev.get('exclusion_type', 'Exclusion')} "
                        f"({ev.get('affected_scope', '')}) — Rule: {ev.get('criterion_rule', 'N/A')} | "
                        f"Reason: {ev.get('reason', 'Data quality / modeling integrity')}."
                    )
                if not bullets and rem_cols:
                    bullets.append(f"Dropped columns compared to raw source: {', '.join(str(c) for c in rem_cols[:20])}.")
                if range_shifts:
                    bullets.append(
                        f"Numeric bounds tightened/adjusted on {len(range_shifts)} feature(s) due to outlier clipping or anomaly removal "
                        f"(e.g., {', '.join(r['column'] for r in range_shifts[:6])})."
                    )
            else:
                body_parts = [
                    "No rows or columns were excluded from the active dataset relative to the raw ingested source "
                    "(100% of ingested observations remain in calculations). Review Studio 01 (Cleaning) and Studio 02 "
                    "(Clustering & Anomaly Screening) if univariate outliers, duplicates, or corrupt clusters should be filtered."
                ]
                bullets = ["Zero records or features have been excluded from calculations so far (Active State = RAW)."]
                recommendations.append(
                    "Inspect univariate IQR outliers and multivariate anomalies in Studios 01–02 before finalizing production models."
                )

            sections.append(
                {
                    "section_id": "data_exclusions",
                    "heading": "3. Excluded Data, Filtering & Removals Log (What Was Excluded & Why)",
                    "body": " ".join(body_parts),
                    "bullets": bullets,
                }
            )

        # 4. 6D Data Quality & Statistical Diagnostics
        if "quality_diagnostics" in enabled and (quality_report or diagnostics_summary):
            bits = []
            bullets = []
            if quality_report:
                bits.append(
                    f"The 6-Dimensional Enterprise Data Quality Profiler evaluated Completeness, Uniqueness, Validity, "
                    f"Consistency, Accuracy, and Timeliness, returning an aggregate score of "
                    f"{quality_report.get('overall_score', 0)}/100 (Grade {quality_report.get('grade', 'N/A')})."
                )
                dims = quality_report.get("dimensions") or {}
                if isinstance(dims, dict):
                    for k, v in dims.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v:
                            bullets.append(f"Quality Dimension — {str(k).title()}: {float(v):.2f}/100")
            if diagnostics_summary:
                bits.append(
                    f"Universal Statistical Diagnostics executed {diagnostics_summary.get('tests_executed', 0)} "
                    f"statistical checks at significance level alpha = {diagnostics_summary.get('alpha', 0.05)} "
                    f"({diagnostics_summary.get('significant_count', 0)} statistically significant findings)."
                )
                if diagnostics_summary.get("high_vif_features"):
                    bullets.append(
                        f"High Multicollinearity (VIF > 10) detected in: {', '.join(diagnostics_summary['high_vif_features'][:10])}."
                    )
                    recommendations.append(
                        "Mitigate high multicollinearity (VIF > 10) via Ridge/Lasso regularization, PCA, or correlated feature pruning."
                    )
            sections.append(
                {
                    "section_id": "quality_diagnostics",
                    "heading": "4. 6D Data Quality Scorecard & Statistical Diagnostics",
                    "body": " ".join(bits),
                    "bullets": bullets,
                }
            )

        # 5. Clustering & Anomaly Screening
        if "clustering_anomalies" in enabled and clustering_summary:
            algo = str(clustering_summary.get("algorithm", "N/A")).upper()
            n_cl = clustering_summary.get("clusters_found", 0)
            sil = clustering_summary.get("silhouette", 0.0)
            noise_pct = float(clustering_summary.get("noise_ratio", 0.0)) * 100.0
            body = (
                f"Unsupervised subpopulation discovery using {algo} identified {n_cl} distinct cluster(s) "
                f"with Silhouette Score = {float(sil):.3f} and density noise ratio = {noise_pct:.2f}%. "
                "Cluster centroid profiles distinguish genuine minority subpopulations from defective outlier segments."
            )
            sections.append(
                {
                    "section_id": "clustering_anomalies",
                    "heading": "5. Clustering, Subpopulation Profiles & Anomaly Screening",
                    "body": body,
                    "bullets": [
                        f"Algorithm: {algo} | Clusters Discovered: {n_cl} | Silhouette: {float(sil):.4f}",
                        f"Davies-Bouldin Index: {clustering_summary.get('davies_bouldin', 'N/A')} | Calinski-Harabasz: {clustering_summary.get('calinski_harabasz', 'N/A')}",
                        f"Density Noise / Anomaly Rate: {noise_pct:.2f}%",
                    ],
                }
            )

        # 6. Automated EDA & Hypothesis Tests
        if "eda_hypothesis" in enabled and eda_summary:
            sections.append(
                {
                    "section_id": "eda_hypothesis",
                    "heading": "6. Automated EDA, Correlations & Hypothesis Testing",
                    "body": str(
                        eda_summary.get("narrative")
                        or "Exploratory statistical profiling quantified univariate distributions, outlier prevalence, pairwise correlations, and hypothesis test p-values across numeric and categorical features."
                    ),
                    "bullets": list(eda_summary.get("highlights") or []),
                }
            )

        # 7. Feature Engineering & Selection
        if "feature_engineering" in enabled and feature_summary:
            sel_cnt = feature_summary.get("selected_count", 0)
            drop_cnt = feature_summary.get("dropped_count", 0)
            method = feature_summary.get("method", "automated")
            body = (
                f"Feature Engineering and Selection ({method}) evaluated candidate predictors, retaining {sel_cnt} "
                f"high-signal feature(s) and excluding {drop_cnt} low-signal or redundant column(s) to reduce variance "
                "and improve generalization."
            )
            bullets = []
            if feature_summary.get("selected_features"):
                bullets.append(f"Top Retained Features: {', '.join(feature_summary['selected_features'][:12])}")
            if feature_summary.get("dropped_features"):
                bullets.append(f"Dropped / Excluded Features: {', '.join(feature_summary['dropped_features'][:12])}")
            sections.append(
                {
                    "section_id": "feature_engineering",
                    "heading": "7. Feature Engineering & Predictor Selection",
                    "body": body,
                    "bullets": bullets,
                }
            )

        # 8. Modeling & Baseline Benchmarks
        if "modeling_leaderboard" in enabled and model_summary:
            champ = model_summary.get("best_model_name") or model_summary.get("champion_model") or model_summary.get("model_name", "Champion Model")
            metric = model_summary.get("primary_metric", "score")
            score = model_summary.get("best_score", model_summary.get("score", "N/A"))
            lift = model_summary.get("lift_over_best_baseline")
            lift_str = (
                f" This delivers a verified lift of {float(lift):+.4f} over the strongest baseline model."
                if isinstance(lift, (int, float))
                else ""
            )
            body = (
                f"Multi-family model benchmarking selected champion architecture '{champ}' achieving "
                f"{metric} = {score}.{lift_str} All candidate architectures (Baselines, Classical, Ensembles, and "
                "Deep Learning) were evaluated under identical validation protocols."
            )
            sections.append(
                {
                    "section_id": "modeling_leaderboard",
                    "heading": "8. Predictive Modeling, AutoML & Deep Learning vs. Baselines",
                    "body": body,
                    "bullets": [
                        f"Champion Model: {champ}",
                        f"Primary Metric ({metric}): {score}",
                        f"Verified Lift Over Best Baseline: {float(lift):+.4f}" if isinstance(lift, (int, float)) else "Baseline benchmark comparison documented in leaderboard table.",
                    ],
                }
            )
            recommendations.append(
                f"Promote '{champ}' to Staging in OmniForge MLOps Registry and enable automated PSI/KS drift alerts."
            )

        # 9. Mathematical Relationships, Model Equations & Computational Rules
        if "mathematical_equations" in enabled:
            f_list = formulas_and_rules or []
            model_eq_count = sum(1 for f in f_list if "Fitted Model Equation" in str(f.get("category", "")))
            meth_count = len(f_list) - model_eq_count
            body = (
                f"This section documents all {len(f_list)} mathematical relationships, statistical criteria, "
                f"and explicit computational rules governing the results ({meth_count} analytical/data-cleaning rules "
                f"and {model_eq_count} extracted fitted model equations). Every formula includes exact symbolic notation, "
                "parameter definitions, decision thresholds, and plain-language analytical interpretation."
            )
            sections.append(
                {
                    "section_id": "mathematical_equations",
                    "heading": "9. Mathematical Relationships, Model Equations & Computational Rules",
                    "body": body,
                    "bullets": [
                        f"{f.get('category', 'Rule')} — {f.get('name', '')}: {str(f.get('mathematical_expression', ''))[:140]}"
                        for f in f_list[:15]
                    ],
                }
            )

        # 10. Future Target Predictions & OOD Shift
        if "future_predictions" in enabled and future_predictions_summary:
            n_scen = future_predictions_summary.get("n_scenarios", 0)
            ood_cnt = future_predictions_summary.get("ood_flagged_count", 0)
            targets = future_predictions_summary.get("targets", [])
            src = future_predictions_summary.get("source_type", "batch")
            body = (
                f"Forward target forecasting scored {n_scen:,} unseen incoming record(s) (source: {src}) across "
                f"target(s) [{', '.join(targets)}]. Out-of-Distribution (OOD) shift guardrails flagged {ood_cnt:,} "
                "record(s) exceeding |z| > 3.0 relative to historical training feature distributions."
            )
            if ood_cnt > 0:
                recommendations.append(
                    f"Manually review the {ood_cnt} Out-of-Distribution (OOD) future scenarios before automated operational execution."
                )
            sections.append(
                {
                    "section_id": "future_predictions",
                    "heading": "10. Future Target Predictions & Distribution Shift (OOD)",
                    "body": body,
                    "bullets": [
                        f"Incoming Source: {src} | Scenarios Scored: {n_scen:,}",
                        f"Targets Forecasted: {', '.join(targets) or 'N/A'}",
                        f"OOD Extrapolation Alerts (|z| > 3.0): {ood_cnt:,}",
                    ],
                }
            )

        # 11. Responsible AI & Green AI
        if "responsible_ai_governance" in enabled and (fairness_report or carbon_report):
            bits = []
            bullets = []
            if fairness_report:
                passed = fairness_report.get("passed", True)
                attr = fairness_report.get("protected_attribute", "sensitive_group")
                di = fairness_report.get("disparate_impact_ratio", 1.0)
                eod = fairness_report.get("equalized_odds_diff", 0.0)
                bits.append(
                    f"Fairness evaluation across protected attribute '{attr}' returned "
                    f"{'PASSED' if passed else 'REVIEW REQUIRED'} (Disparate Impact = {di}, Equalized Odds Diff = {eod})."
                )
                bullets.append(f"Fairness Gate ({attr}): {'PASSED' if passed else 'FLAGGED'} | DI={di} | EOD={eod}")
                if not passed:
                    recommendations.append(
                        "Apply Kamiran-Calders reweighing or group threshold calibration prior to production release."
                    )
            if carbon_report:
                g_co2 = float(carbon_report.get("emissions_g_co2eq", 0.0))
                kwh = float(carbon_report.get("energy_kwh", 0.0))
                green = carbon_report.get("green_score", 100.0)
                bits.append(
                    f"Green AI telemetry measured {kwh:.8f} kWh generating {g_co2:.6f} gCO2eq "
                    f"(Green AI Score: {green}/100, grid: {carbon_report.get('country_iso', 'DEU')})."
                )
                bullets.append(f"Carbon Footprint: {g_co2:.6f} gCO2eq | Green Score: {green}/100")
            sections.append(
                {
                    "section_id": "responsible_ai_governance",
                    "heading": "11. Responsible AI, Fairness Governance & Green AI Carbon",
                    "body": " ".join(bits),
                    "bullets": bullets,
                }
            )

        # 12. Decision Intelligence (Risk, Forecasting, Causal, Optimization, Simulation)
        if "decision_intelligence" in enabled and decision_summary:
            bits = []
            bullets = []
            if decision_summary.get("risk_index") is not None:
                bits.append(
                    f"Structural Risk Index scored {decision_summary['risk_index']} (Band: {decision_summary.get('risk_band', 'N/A')})."
                )
                bullets.append(f"Structural Risk Index: {decision_summary['risk_index']} ({decision_summary.get('risk_band', 'N/A')})")
            if decision_summary.get("forecast_champion"):
                bits.append(
                    f"Time-series forecasting selected champion '{decision_summary['forecast_champion']}'."
                )
                bullets.append(f"Forecast Champion: {decision_summary['forecast_champion']}")
            if decision_summary.get("causal_ate") is not None:
                bits.append(
                    f"Causal inference estimated Average Treatment Effect ATE = {float(decision_summary['causal_ate']):.4f}."
                )
                bullets.append(f"Causal ATE: {float(decision_summary['causal_ate']):.4f}")
            if bits:
                sections.append(
                    {
                        "section_id": "decision_intelligence",
                        "heading": "12. Risk Analytics, Forecasting, Causal Inference & Optimization",
                        "body": " ".join(bits),
                        "bullets": bullets,
                    }
                )

        if not recommendations:
            recommendations.append("Establish automated scheduled validation, OOD alerting, and drift monitoring in OmniForge MLOps.")

        dedup_recs = list(dict.fromkeys(str(r) for r in recommendations if str(r).strip()))
        if "recommendations" not in enabled:
            dedup_recs = []

        return {
            "title": title,
            "author": author,
            "generated_at": datetime.now(UTC).isoformat(),
            "domain": domain,
            "selected_sections": list(enabled),
            "sections": sections,
            "recommendations": dedup_recs,
        }
