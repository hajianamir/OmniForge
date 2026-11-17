"""Publication-grade Matplotlib Chart Builder for Multi-Format OmniForge Reports (DOCX, PDF, XLSX, HTML, PPTX)."""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omniforge.reporting.sections import ReportChartArtifact

# Crisp executive palette
_NAVY = "#0F172A"
_BLUE = "#2563EB"
_TEAL = "#0EA5E9"
_EMERALD = "#10B981"
_AMBER = "#F59E0B"
_RED = "#EF4444"
_PURPLE = "#7C3AED"
_SLATE = "#64748B"


def _fig_to_artifact(
    fig: plt.Figure,
    *,
    chart_id: str,
    section_id: str,
    title: str,
    caption: str,
    width_inches: float = 6.3,
    height_inches: float = 3.6,
    dpi: int = 150,
) -> ReportChartArtifact:
    """Serialize a Matplotlib Figure to base64 PNG and close it safely."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return ReportChartArtifact(
        chart_id=chart_id,
        section_id=section_id,
        title=title,
        caption=caption,
        png_base64=encoded,
        width_inches=width_inches,
        height_inches=height_inches,
    )


def _apply_clean_style(ax: plt.Axes) -> None:
    """Apply clean publication styling to a Matplotlib axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#94A3B8")
    ax.set_axisbelow(True)
    ax.tick_params(colors="#1E293B", labelsize=8.5)


class ReportChartBuilder:
    """Generates rich analytical Matplotlib figures from live OmniForge workspace data and artifacts."""

    @classmethod
    def build_all_charts(
        cls,
        *,
        selected_sections: set[str],
        df: pd.DataFrame | None = None,
        raw_df: pd.DataFrame | None = None,
        target_columns: list[str] | None = None,
        quality_report: dict[str, Any] | None = None,
        exclusions: list[dict[str, Any]] | None = None,
        raw_vs_processed: dict[str, Any] | None = None,
        clustering_result: Any | None = None,
        fs_scores: dict[str, float] | None = None,
        fs_selected: list[str] | None = None,
        leaderboard: list[dict[str, Any]] | None = None,
        dl_benchmark_df: pd.DataFrame | None = None,
        model_equations: dict[str, dict[str, dict[str, Any]]] | None = None,
        future_prediction_result: Any | None = None,
        risk_assessment: dict[str, Any] | None = None,
    ) -> list[ReportChartArtifact]:
        """Build all applicable charts matching ``selected_sections``."""
        charts: list[ReportChartArtifact] = []
        targets = target_columns or []

        # 1. Data Exclusions / Raw vs Processed Waterfall
        if "data_exclusions" in selected_sections and (exclusions or raw_vs_processed):
            art = cls._chart_exclusions(exclusions or [], raw_vs_processed or {})
            if art is not None:
                charts.append(art)

        # 2. 6D Quality Scorecard + Missingness by Column
        if "quality_diagnostics" in selected_sections:
            if quality_report:
                art = cls._chart_quality_dimensions(quality_report)
                if art is not None:
                    charts.append(art)
            source_df = raw_df if (raw_df is not None and raw_df.isna().sum().sum() > 0) else df
            if source_df is not None:
                art = cls._chart_missingness_and_dtypes(source_df)
                if art is not None:
                    charts.append(art)

        # 3. EDA: Target Distribution, Numeric Outlier Boxplots & Correlation Heatmap
        if "eda_hypothesis" in selected_sections and df is not None and not df.empty:
            art_dist = cls._chart_target_and_numeric_outliers(df, targets)
            if art_dist is not None:
                charts.append(art_dist)
            art_corr = cls._chart_correlation_heatmap(df)
            if art_corr is not None:
                charts.append(art_corr)

        # 4. Clustering Projection & Subpopulation Sizes
        if "clustering_anomalies" in selected_sections and clustering_result is not None and df is not None:
            art = cls._chart_clustering(df, clustering_result)
            if art is not None:
                charts.append(art)

        # 5. Feature Selection / Importance Ranking
        if "feature_engineering" in selected_sections and fs_scores:
            art = cls._chart_feature_selection(fs_scores, set(fs_selected or []))
            if art is not None:
                charts.append(art)

        # 6. Model Leaderboard & Deep Learning vs Baselines
        if "modeling_leaderboard" in selected_sections:
            if leaderboard:
                art = cls._chart_leaderboard(leaderboard)
                if art is not None:
                    charts.append(art)
            if isinstance(dl_benchmark_df, pd.DataFrame) and not dl_benchmark_df.empty:
                art = cls._chart_dl_vs_baselines(dl_benchmark_df)
                if art is not None:
                    charts.append(art)

        # 7. Mathematical Equation Top Feature Coefficients / Drivers
        if "mathematical_equations" in selected_sections and model_equations:
            art = cls._chart_equation_drivers(model_equations)
            if art is not None:
                charts.append(art)

        # 8. Future Predictions Distribution & OOD Shift
        if "future_predictions" in selected_sections and future_prediction_result is not None:
            art = cls._chart_future_predictions(future_prediction_result)
            if art is not None:
                charts.append(art)

        # 9. Risk Assessment Drivers
        if "decision_intelligence" in selected_sections and risk_assessment:
            art = cls._chart_risk_components(risk_assessment)
            if art is not None:
                charts.append(art)

        return charts

    @staticmethod
    def _chart_exclusions(
        exclusions: list[dict[str, Any]],
        diff: dict[str, Any],
    ) -> ReportChartArtifact | None:
        raw_rows = int(diff.get("raw_rows", 0))
        proc_rows = int(diff.get("processed_rows", 0))
        if raw_rows == 0 and not exclusions:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
        # Left panel: Raw vs Processed Row Retention
        ax0 = axes[0]
        if raw_rows > 0:
            removed_rows = int(diff.get("rows_removed", max(0, raw_rows - proc_rows)))
            labels = ["Raw Ingested\nRows", "Excluded /\nRemoved Rows", "Retained Active\nRows"]
            vals = [raw_rows, removed_rows, proc_rows]
            bar_colors = [_SLATE, _RED if removed_rows > 0 else _EMERALD, _BLUE]
            bars = ax0.bar(labels, vals, color=bar_colors, width=0.55)
            for b, v in zip(bars, vals, strict=False):
                ax0.text(
                    b.get_x() + b.get_width() / 2.0,
                    b.get_height() * 1.02 + 0.5,
                    f"{v:,}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    color=_NAVY,
                )
            ax0.set_title("Row Retention: Raw vs. Processed", fontsize=10, fontweight="bold", color=_NAVY)
            ax0.set_ylabel("Row Count", fontsize=8.5)
            ax0.set_ylim(0, max(vals) * 1.18 + 1)
            ax0.spines["top"].set_visible(False)
            ax0.spines["right"].set_visible(False)
            ax0.grid(axis="y", linestyle="--", alpha=0.35)
        else:
            ax0.text(0.5, 0.5, "No Raw vs. Processed\nrow delta recorded.", ha="center", va="center", color=_SLATE)
            ax0.axis("off")

        # Right panel: Breakdown of Exclusions / Modifications by Type
        ax1 = axes[1]
        if exclusions:
            agg: dict[str, int] = {}
            for ev in exclusions:
                lbl = str(ev.get("exclusion_type", "Exclusion"))[:32]
                agg[lbl] = agg.get(lbl, 0) + int(ev.get("count", 0))
            agg = {k: v for k, v in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:8] if v > 0}
            if agg:
                ys = list(agg.keys())[::-1]
                xs = [agg[k] for k in ys]
                bars = ax1.barh(ys, xs, color=_AMBER, height=0.58)
                for b, v in zip(bars, xs, strict=False):
                    ax1.text(v + max(xs) * 0.02, b.get_y() + b.get_height() / 2.0, f"{v:,}", va="center", fontsize=8, fontweight="bold")
                ax1.set_title("Exclusions & Remediations by Cause", fontsize=10, fontweight="bold", color=_NAVY)
                ax1.set_xlabel("Affected Items (Rows / Values / Columns)", fontsize=8.5)
                ax1.set_xlim(0, max(xs) * 1.22 + 1)
                _apply_clean_style(ax1)
            else:
                ax1.text(0.5, 0.5, "All exclusion events had\n0 excluded records.", ha="center", va="center", color=_SLATE)
                ax1.axis("off")
        else:
            missing_res = int(diff.get("missing_cells_resolved", 0))
            dups_before = int(diff.get("raw_duplicate_rows", 0))
            removed_cols = len(diff.get("removed_columns") or [])
            ys = ["Missing Cells Resolved", "Raw Duplicates Found", "Columns Dropped"]
            xs = [missing_res, dups_before, removed_cols]
            ax1.barh(ys, xs, color=[_TEAL, _RED, _PURPLE], height=0.55)
            for idx, v in enumerate(xs):
                ax1.text(v + max(max(xs), 1) * 0.03, idx, f"{v:,}", va="center", fontsize=8.5, fontweight="bold")
            ax1.set_title("Forensic Data Quality Deltas", fontsize=10, fontweight="bold", color=_NAVY)
            ax1.set_xlim(0, max(max(xs), 1) * 1.25 + 1)
            _apply_clean_style(ax1)

        fig.tight_layout()
        return _fig_to_artifact(
            fig,
            chart_id="data_exclusions_breakdown",
            section_id="data_exclusions",
            title="Data Exclusions, Row Retention & Remediation Breakdown",
            caption=(
                "Left: Comparison of raw ingested rows, excluded rows, and active retained rows used in downstream calculations. "
                "Right: Quantified count of records, cells, or columns excluded/remediated per audit cause."
            ),
            width_inches=6.5,
            height_inches=3.1,
        )

    @staticmethod
    def _chart_quality_dimensions(quality_report: dict[str, Any]) -> ReportChartArtifact | None:
        dims = quality_report.get("dimensions") or {}
        if not isinstance(dims, dict) or not dims:
            # Try standard top-level keys
            cand_keys = ["completeness", "uniqueness", "validity", "consistency", "accuracy", "timeliness"]
            dims = {k.title(): float(quality_report[k]) for k in cand_keys if k in quality_report and isinstance(quality_report[k], (int, float))}
        if not dims:
            return None

        names = [str(k).replace("_", " ").title() for k in dims]
        scores = [float(v) for v in dims.values()]
        bar_colors = [_EMERALD if s >= 90 else (_BLUE if s >= 80 else (_AMBER if s >= 70 else _RED)) for s in scores]

        fig, ax = plt.subplots(figsize=(7.6, 3.6))
        bars = ax.barh(names[::-1], scores[::-1], color=bar_colors[::-1], height=0.58)
        for b, s in zip(bars, scores[::-1], strict=False):
            ax.text(min(s + 1.5, 102), b.get_y() + b.get_height() / 2.0, f"{s:.1f}%", va="center", fontsize=9, fontweight="bold", color=_NAVY)
        overall = quality_report.get("overall_score", 0.0)
        grade = quality_report.get("grade", "-")
        ax.axvline(80, color=_SLATE, linestyle=":", linewidth=1.2, label="Enterprise Threshold (80%)")
        ax.set_xlim(0, 112)
        ax.set_xlabel("Score (0 – 100)", fontsize=9)
        ax.set_title(f"6-Dimensional Enterprise Data Quality Scorecard (Overall: {overall}/100 · Grade {grade})", fontsize=10.5, fontweight="bold", color=_NAVY)
        ax.legend(loc="lower right", fontsize=8)
        _apply_clean_style(ax)
        fig.tight_layout()
        return _fig_to_artifact(
            fig,
            chart_id="quality_six_dimensions",
            section_id="quality_diagnostics",
            title="6-Dimensional Data Quality Scorecard",
            caption=(
                f"Enterprise Data Quality profile scoring {overall}/100 (Grade {grade}). "
                "Bars below the 80% governance line indicate dimensions requiring targeted remediation."
            ),
            width_inches=6.3,
            height_inches=3.2,
        )

    @staticmethod
    def _chart_missingness_and_dtypes(df: pd.DataFrame) -> ReportChartArtifact | None:
        if df.empty:
            return None
        miss_pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
        top_miss = miss_pct[miss_pct > 0].head(12)

        dtype_groups = {
            "Numeric": int(df.select_dtypes(include=[np.number]).shape[1]),
            "Categorical / Text": int(df.select_dtypes(include=["object", "category", "string"]).shape[1]),
            "Datetime": int(df.select_dtypes(include=["datetime", "datetimetz"]).shape[1]),
            "Boolean / Other": int(
                df.shape[1]
                - df.select_dtypes(include=[np.number, "object", "category", "string", "datetime", "datetimetz"]).shape[1]
            ),
        }
        dtype_groups = {k: v for k, v in dtype_groups.items() if v > 0}

        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
        ax0 = axes[0]
        if not top_miss.empty:
            ys = [str(c)[:26] for c in top_miss.index][::-1]
            xs = [float(v) for v in top_miss.values][::-1]
            bars = ax0.barh(ys, xs, color=_RED, alpha=0.85, height=0.6)
            for b, v in zip(bars, xs, strict=False):
                ax0.text(v + max(xs) * 0.03, b.get_y() + b.get_height() / 2.0, f"{v:.1f}%", va="center", fontsize=8)
            ax0.set_xlim(0, max(xs) * 1.25 + 0.5)
            ax0.set_xlabel("Missing Values (%)", fontsize=8.5)
            ax0.set_title("Top Columns with Missing Data", fontsize=10, fontweight="bold", color=_NAVY)
            _apply_clean_style(ax0)
        else:
            ax0.text(0.5, 0.5, "0% Missing Values\n(100% Complete Dataset)", ha="center", va="center", fontsize=10, fontweight="bold", color=_EMERALD)
            ax0.set_title("Missingness Profile", fontsize=10, fontweight="bold", color=_NAVY)
            ax0.axis("off")

        ax1 = axes[1]
        d_labels = list(dtype_groups.keys())
        d_vals = list(dtype_groups.values())
        d_colors = [_BLUE, _TEAL, _PURPLE, _AMBER][: len(d_labels)]
        ax1.pie(
            d_vals,
            labels=[f"{k}\n({v})" for k, v in dtype_groups.items()],
            autopct="%1.0f%%",
            colors=d_colors,
            startangle=120,
            textprops={"fontsize": 8.5, "color": _NAVY},
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        ax1.set_title(f"Column Schema Composition ({df.shape[1]} Cols)", fontsize=10, fontweight="bold", color=_NAVY)

        fig.tight_layout()
        return _fig_to_artifact(
            fig,
            chart_id="missingness_and_schema",
            section_id="quality_diagnostics",
            title="Column Missingness Profile & Schema Composition",
            caption=(
                "Left: Highest missing-value columns before/after cleaning. "
                "Right: Breakdown of dataset columns by statistical data type."
            ),
            width_inches=6.5,
            height_inches=3.1,
        )

    @staticmethod
    def _chart_target_and_numeric_outliers(df: pd.DataFrame, targets: list[str]) -> ReportChartArtifact | None:
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        if not num_cols:
            return None

        sample_df = df.sample(n=min(len(df), 5000), random_state=42) if len(df) > 5000 else df
        target_col = targets[0] if targets and targets[0] in df.columns else num_cols[-1]
        feature_nums = [c for c in num_cols if c != target_col][:5] or num_cols[:5]

        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
        # Left panel: Target distribution
        ax0 = axes[0]
        t_ser = sample_df[target_col].dropna()
        if pd.api.types.is_numeric_dtype(t_ser) and t_ser.nunique() > 10:
            ax0.hist(t_ser, bins=30, color=_BLUE, alpha=0.82, edgecolor="white")
            ax0.axvline(float(t_ser.mean()), color=_RED, linestyle="--", linewidth=1.4, label=f"Mean: {t_ser.mean():.2f}")
            ax0.axvline(float(t_ser.median()), color=_EMERALD, linestyle="-.", linewidth=1.4, label=f"Median: {t_ser.median():.2f}")
            ax0.legend(fontsize=8)
            ax0.set_ylabel("Frequency", fontsize=8.5)
        else:
            vc = t_ser.astype(str).value_counts().head(10)
            ax0.bar([str(k)[:14] for k in vc.index], vc.values, color=_BLUE, alpha=0.85)
            ax0.tick_params(axis="x", rotation=25)
            ax0.set_ylabel("Count", fontsize=8.5)
        ax0.set_title(f"Target Variable Distribution: '{target_col}'", fontsize=10, fontweight="bold", color=_NAVY)
        ax0.spines["top"].set_visible(False)
        ax0.spines["right"].set_visible(False)
        ax0.grid(axis="y", linestyle="--", alpha=0.35)

        # Right panel: Standardized Outlier Spread (IQR outlier % across top numeric features)
        ax1 = axes[1]
        iqr_pcts: list[tuple[str, float]] = []
        for c in num_cols[:20]:
            s = pd.to_numeric(sample_df[c], errors="coerce").dropna()
            if len(s) < 5:
                continue
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q3 - q1
            if iqr <= 1e-12:
                continue
            out_rate = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100.0)
            iqr_pcts.append((str(c)[:24], out_rate))
        iqr_pcts.sort(key=lambda x: x[1], reverse=True)
        top_iqr = iqr_pcts[:8]
        if top_iqr:
            ys = [x[0] for x in top_iqr][::-1]
            xs = [x[1] for x in top_iqr][::-1]
            bars = ax1.barh(ys, xs, color=[_RED if v >= 5 else (_AMBER if v >= 1 else _TEAL) for v in xs], height=0.6)
            for b, v in zip(bars, xs, strict=False):
                ax1.text(v + max(max(xs), 0.5) * 0.03, b.get_y() + b.get_height() / 2.0, f"{v:.1f}%", va="center", fontsize=8)
            ax1.set_xlim(0, max(max(xs), 1.0) * 1.25 + 0.5)
            ax1.set_xlabel("Observations Outside 1.5×IQR Fences (%)", fontsize=8.5)
            ax1.set_title("Univariate IQR Outlier Prevalence (Top Features)", fontsize=10, fontweight="bold", color=_NAVY)
            _apply_clean_style(ax1)
        else:
            # Fallback standardized boxplot
            box_data = [
                ((sample_df[c].dropna() - sample_df[c].mean()) / (sample_df[c].std() + 1e-9)).values
                for c in feature_nums
            ]
            ax1.boxplot(box_data, tick_labels=[c[:12] for c in feature_nums], patch_artist=True)
            ax1.set_title("Standardized Numeric Spread", fontsize=10, fontweight="bold", color=_NAVY)

        fig.tight_layout()
        return _fig_to_artifact(
            fig,
            chart_id="target_and_outliers",
            section_id="eda_hypothesis",
            title="Target Variable Distribution & Univariate IQR Outlier Prevalence",
            caption=(
                f"Left: Empirical distribution of primary target '{target_col}'. "
                "Right: Percentage of observations falling outside Tukey 1.5×IQR fences per numeric column."
            ),
            width_inches=6.5,
            height_inches=3.1,
        )

    @staticmethod
    def _chart_correlation_heatmap(df: pd.DataFrame) -> ReportChartArtifact | None:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return None
        # Keep top 12 highest-variance numeric columns so heatmap remains crisp and legible
        if num_df.shape[1] > 12:
            variances = num_df.var(numeric_only=True).sort_values(ascending=False)
            num_df = num_df[list(variances.index[:12])]
        corr = num_df.corr().fillna(0.0)

        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        labels = [str(c)[:18] for c in corr.columns]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = float(corr.values[i, j])
                txt_col = "white" if abs(val) >= 0.55 else _NAVY
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.2, color=txt_col)
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label("Pearson r", fontsize=8.5)
        ax.set_title("Pearson Correlation Matrix (Top Numeric Variables)", fontsize=10.5, fontweight="bold", color=_NAVY)
        fig.tight_layout()
        return _fig_to_artifact(
            fig,
            chart_id="correlation_heatmap",
            section_id="eda_hypothesis",
            title="Numeric Feature Correlation Heatmap (Pearson r)",
            caption=(
                "Pairwise linear correlation coefficients ranging from -1.0 (strong inverse relationship) "
                "to +1.0 (strong positive relationship). Dark off-diagonal cells highlight collinear pairs."
            ),
            width_inches=6.2,
            height_inches=4.1,
        )

    @staticmethod
    def _chart_clustering(df: pd.DataFrame, clust_res: Any) -> ReportChartArtifact | None:
        labels = getattr(clust_res, "labels", None)
        if labels is None or len(labels) == 0:
            return None
        lbl_arr = np.asarray(labels)
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

        # Left: Cluster size bar chart
        ax0 = axes[0]
        uniq, cnts = np.unique(lbl_arr, return_counts=True)
        names = ["Noise (-1)" if int(u) == -1 else f"Cluster {int(u)}" for u in uniq]
        bar_cols = [_RED if int(u) == -1 else plt.cm.tab10(int(u) % 10) for u in uniq]
        bars = ax0.bar(names, cnts, color=bar_cols, width=0.6)
        for b, v in zip(bars, cnts, strict=False):
            ax0.text(b.get_x() + b.get_width() / 2.0, v * 1.02 + 0.5, f"{int(v):,}", ha="center", fontsize=8, fontweight="bold")
        ax0.tick_params(axis="x", rotation=25)
        ax0.set_ylabel("Records", fontsize=8.5)
        ax0.set_ylim(0, max(cnts) * 1.18 + 1)
        ax0.set_title(f"Cluster Membership ({getattr(clust_res, 'algorithm', 'cluster').upper()})", fontsize=10, fontweight="bold", color=_NAVY)
        ax0.spines["top"].set_visible(False)
        ax0.spines["right"].set_visible(False)
        ax0.grid(axis="y", linestyle="--", alpha=0.35)

        # Right: 2D PCA scatter projection
        ax1 = axes[1]
        pca_2d = getattr(clust_res, "pca_2d", None)
        if pca_2d is not None and len(pca_2d) == len(lbl_arr):
            coords = np.asarray(pca_2d)
            if len(coords) > 3000:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(coords), size=3000, replace=False)
                coords = coords[idx]
                sub_lbl = lbl_arr[idx]
            else:
                sub_lbl = lbl_arr
            for u in uniq:
                mask = sub_lbl == u
                col = _RED if int(u) == -1 else plt.cm.tab10(int(u) % 10)
                mk = "x" if int(u) == -1 else "o"
                ax1.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=16 if int(u) != -1 else 26,
                    alpha=0.72,
                    color=col,
                    marker=mk,
                    label="Noise (-1)" if int(u) == -1 else f"C{int(u)}",
                )
            ax1.set_xlabel("PC1", fontsize=8.5)
            ax1.set_ylabel("PC2", fontsize=8.5)
            ax1.set_title("2D PCA Subpopulation Projection", fontsize=10, fontweight="bold", color=_NAVY)
            ax1.legend(fontsize=7.5, ncol=2, frameon=True)
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
        else:
            ax1.text(0.5, 0.5, "2D PCA projection unavailable.", ha="center", va="center", color=_SLATE)
            ax1.axis("off")

        fig.tight_layout()
        sil = (getattr(clust_res, "metrics", None) or {}).get("silhouette", 0.0)
        return _fig_to_artifact(
            fig,
            chart_id="clustering_overview",
            section_id="clustering_anomalies",
            title="Unsupervised Clustering & Subpopulation Separation",
            caption=(
                f"Algorithm: {getattr(clust_res, 'algorithm', 'N/A').upper()} | "
                f"Clusters found: {getattr(clust_res, 'n_clusters_found', 0)} | "
                f"Silhouette Score: {float(sil):.3f} | Noise points (-1) highlighted in red."
            ),
            width_inches=6.5,
            height_inches=3.1,
        )

    @staticmethod
    def _chart_feature_selection(fs_scores: dict[str, float], selected: set[str]) -> ReportChartArtifact | None:
        if not fs_scores:
            return None
        items = sorted(fs_scores.items(), key=lambda kv: float(kv[1]), reverse=True)[:15]
        names = [str(k)[:26] for k, _ in items][::-1]
        vals = [float(v) for _, v in items][::-1]
        cols = [_EMERALD if k in selected else _SLATE for k, _ in items][::-1]

        fig, ax = plt.subplots(figsize=(7.6, 3.9))
        bars = ax.barh(names, vals, color=cols, height=0.6)
        for b, v in zip(bars, vals, strict=False):
            ax.text(v + max(max(vals), 1e-4) * 0.02, b.get_y() + b.get_height() / 2.0, f"{v:.4f}", va="center", fontsize=8)
        ax.set_xlim(0, max(max(vals), 1e-4) * 1.24)
        ax.set_xlabel("Importance / Selection Score", fontsize=8.5)
        ax.set_title("Feature Selection Ranking (Green = Retained · Slate = Dropped)", fontsize=10, fontweight="bold", color=_NAVY)
        _apply_clean_style(ax)
        fig.tight_layout()
        return _fig_to_artifact(
            fig,
            chart_id="feature_selection_ranking",
            section_id="feature_engineering",
            title="Feature Importance & Selection Retention Ranking",
            caption="Top ranked predictors contrasting retained features (emerald) against pruned/dropped predictors (slate).",
            width_inches=6.3,
            height_inches=3.4,
        )

    @staticmethod
    def _chart_leaderboard(leaderboard: list[dict[str, Any]]) -> ReportChartArtifact | None:
        rows = [r for r in leaderboard if isinstance(r, dict)]
        if not rows:
            return None
        # Detect score key
        score_key = next((k for k in ("score", "cv_score", "primary_score", "f1_macro", "r2", "accuracy", "roc_auc") if k in rows[0]), None)
        name_key = next((k for k in ("model", "model_name", "name") if k in rows[0]), None)
        if score_key is None or name_key is None:
            return None
        valid = []
        for r in rows:
            try:
                valid.append((str(r[name_key]), float(r[score_key]), bool(r.get("is_baseline", "baseline" in str(r[name_key]).lower() or "dummy" in str(r[name_key]).lower()))))
            except (TypeError, ValueError):
                continue
        if not valid:
            return None
        valid.sort(key=lambda x: x[1], reverse=True)
        top = valid[:12][::-1]
        names = [x[0][:30] for x in top]
        vals = [x[1] for x in top]
        cols = [
            _EMERALD if idx == len(top) - 1 else (_AMBER if x[2] else _BLUE)
            for idx, x in enumerate(top)
        ]

        fig, ax = plt.subplots(figsize=(7.8, 3.8))
        bars = ax.barh(names, vals, color=cols, height=0.6)
        span = max(abs(max(vals)), abs(min(vals)), 0.1)
        for b, v in zip(bars, vals, strict=False):
            offset = span * 0.02 if v >= 0 else -span * 0.02
            ha = "left" if v >= 0 else "right"
            ax.text(v + offset, b.get_y() + b.get_height() / 2.0, f"{v:.4f}", va="center", ha=ha, fontsize=8.2, fontweight="bold")
        ax.set_xlabel(f"Primary Metric ({score_key})", fontsize=9)
        ax.set_title("Model Benchmark Leaderboard (Green = Champion · Amber = Baseline)", fontsize=10, fontweight="bold", color=_NAVY)
        _apply_clean_style(ax)
        fig.tight_layout()
        return _fig_to_artifact(
            fig,
            chart_id="model_leaderboard",
            section_id="modeling_leaderboard",
            title="AutoML & Model Leaderboard vs. Baselines",
            caption="Cross-validated predictive comparison showing the selected Champion model (emerald) relative to competing architectures and baseline benchmarks (amber).",
            width_inches=6.4,
            height_inches=3.3,
        )

    @staticmethod
    def _chart_dl_vs_baselines(dl_df: pd.DataFrame) -> ReportChartArtifact | None:
        if dl_df.empty:
            return None
        name_col = next((c for c in ("Model", "model", "Architecture") if c in dl_df.columns), None)
        score_col = next((c for c in ("Score", "Primary_Score", "F1", "R2", "Accuracy") if c in dl_df.columns), None)
        if name_col is None or score_col is None:
            num_cols = list(dl_df.select_dtypes(include=[np.number]).columns)
            if name_col is None:
                name_col = str(dl_df.columns[0])
            if not num_cols:
                return None
            score_col = num_cols[0]

        sub = dl_df[[name_col, score_col]].dropna().copy()
        sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce")
        sub = sub.dropna().sort_values(score_col, ascending=True)
        if sub.empty:
            return None

        fig, ax = plt.subplots(figsize=(7.6, 3.6))
        colors_dl = [
            _AMBER if "baseline" in str(m).lower() or "dummy" in str(m).lower() or "ridge" in str(m).lower() or "logistic" in str(m).lower() else _PURPLE
            for m in sub[name_col]
        ]
        bars = ax.barh([str(m)[:30] for m in sub[name_col]], sub[score_col].values, color=colors_dl, height=0.58)
        for b, v in zip(bars, sub[score_col].values, strict=False):
            ax.text(float(v) + 0.01, b.get_y() + b.get_height() / 2.0, f"{float(v):.4f}", va="center", fontsize=8.5, fontweight="bold")
        ax.set_xlabel(str(score_col), fontsize=9)
        ax.set_title("Deep Learning Architectures vs. Baseline Models", fontsize=10, fontweight="bold", color=_NAVY)
        _apply_clean_style(ax)
        fig.tight_layout()
        return _fig_to_artifact(
            fig,
            chart_id="dl_vs_baselines",
            section_id="modeling_leaderboard",
            title="Deep Learning Architectures vs. Classical Baselines",
            caption="Head-to-head comparison of neural network architectures (purple) against classical baseline benchmarks (amber) on identical splits.",
            width_inches=6.3,
            height_inches=3.2,
        )

    @staticmethod
    def _chart_equation_drivers(model_equations: dict[str, dict[str, dict[str, Any]]]) -> ReportChartArtifact | None:
        # Pick the first model with structured coefficient terms
        chosen_model = None
        chosen_target = None
        chosen_terms: list[dict[str, Any]] = []
        for t_name, eq_map in model_equations.items():
            if not isinstance(eq_map, dict):
                continue
            for m_name, eq_d in eq_map.items():
                terms = (eq_d or {}).get("terms") or []
                if terms and len(terms) > len(chosen_terms):
                    chosen_terms = [t for t in terms if isinstance(t, dict) and "feature" in t and "coefficient" in t]
                    chosen_model = m_name
                    chosen_target = t_name
        if not chosen_terms:
            return None

        top = sorted(chosen_terms, key=lambda t: abs(float(t["coefficient"])), reverse=True)[:12][::-1]
        names = [str(t["feature"])[:26] for t in top]
        coefs = [float(t["coefficient"]) for t in top]
        cols = [_EMERALD if c >= 0 else _RED for c in coefs]

        fig, ax = plt.subplots(figsize=(7.6, 3.8))
        ax.barh(names, coefs, color=cols, height=0.6)
        ax.axvline(0.0, color=_NAVY, linewidth=1.0)
        ax.set_xlabel("Mathematical Coefficient / Sensitivity Weight", fontsize=8.5)
        ax.set_title(f"Top Equation Drivers — {chosen_model} (Target: {chosen_target})", fontsize=10, fontweight="bold", color=_NAVY)
        _apply_clean_style(ax)
        fig.tight_layout()
        return _fig_to_artifact(
            fig,
            chart_id="equation_drivers",
            section_id="mathematical_equations",
            title="Mathematical Equation Sensitivity & Coefficient Drivers",
            caption=(
                f"Magnitude and sign of the largest mathematical weights in {chosen_model} predicting '{chosen_target}'. "
                "Green bars increase the predicted output; red bars decrease it."
            ),
            width_inches=6.3,
            height_inches=3.3,
        )

    @staticmethod
    def _chart_future_predictions(fut_res: Any) -> ReportChartArtifact | None:
        scored_df = getattr(fut_res, "scored_df", None)
        if not isinstance(scored_df, pd.DataFrame) or scored_df.empty:
            return None
        pred_cols = [c for c in scored_df.columns if str(c).startswith("pred_")]
        if not pred_cols:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
        p_col = pred_cols[0]
        s = scored_df[p_col].dropna()
        ax0 = axes[0]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8:
            ax0.hist(s, bins=20, color=_BLUE, alpha=0.85, edgecolor="white")
            ax0.set_ylabel("Scenarios", fontsize=8.5)
        else:
            vc = s.astype(str).value_counts().head(10)
            ax0.bar([str(k)[:14] for k in vc.index], vc.values, color=_BLUE)
            ax0.set_ylabel("Count", fontsize=8.5)
        ax0.set_title(f"Future Predictions: {p_col}", fontsize=10, fontweight="bold", color=_NAVY)
        ax0.spines["top"].set_visible(False)
        ax0.spines["right"].set_visible(False)
        ax0.grid(axis="y", linestyle="--", alpha=0.35)

        ax1 = axes[1]
        if "ood_shift_score" in scored_df.columns:
            ood = pd.to_numeric(scored_df["ood_shift_score"], errors="coerce").dropna()
            ax1.hist(ood, bins=20, color=_AMBER, alpha=0.85, edgecolor="white")
            ax1.axvline(3.0, color=_RED, linestyle="--", linewidth=1.5, label="OOD Threshold (z = 3.0)")
            ax1.legend(fontsize=8)
            ax1.set_xlabel("Max Feature Shift |z|", fontsize=8.5)
            ax1.set_title("Incoming Future Data OOD Shift Score", fontsize=10, fontweight="bold", color=_NAVY)
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.grid(axis="y", linestyle="--", alpha=0.35)
        else:
            ax1.text(0.5, 0.5, "OOD shift score not computed.", ha="center", va="center", color=_SLATE)
            ax1.axis("off")

        fig.tight_layout()
        return _fig_to_artifact(
            fig,
            chart_id="future_predictions_ood",
            section_id="future_predictions",
            title="Future Target Predictions & Out-of-Distribution (OOD) Shift Profile",
            caption=(
                f"Left: Distribution of forward forecasts ({p_col}). "
                "Right: Feature-level OOD z-score distribution relative to the z=3.0 extrapolation threshold."
            ),
            width_inches=6.5,
            height_inches=3.0,
        )

    @staticmethod
    def _chart_risk_components(risk_assessment: dict[str, Any]) -> ReportChartArtifact | None:
        comps = risk_assessment.get("risk_components") or {}
        if not isinstance(comps, dict) or not comps:
            return None
        names = [str(k).replace("_", " ").title() for k in comps]
        vals = [float(v) for v in comps.values()]

        fig, ax = plt.subplots(figsize=(7.4, 3.4))
        cols = [_RED if v >= 70 else (_AMBER if v >= 40 else _EMERALD) for v in vals]
        bars = ax.barh(names[::-1], vals[::-1], color=cols[::-1], height=0.58)
        for b, v in zip(bars, vals[::-1], strict=False):
            ax.text(v + 1.2, b.get_y() + b.get_height() / 2.0, f"{v:.1f}", va="center", fontsize=8.5, fontweight="bold")
        ax.set_xlim(0, max(max(vals), 100) * 1.12)
        ax.set_xlabel("Component Risk Score", fontsize=8.5)
        sri = risk_assessment.get("structural_risk_index", "N/A")
        band = risk_assessment.get("risk_band", "-")
        ax.set_title(f"Structural Risk Index Breakdown (SRI: {sri} · Band: {band})", fontsize=10, fontweight="bold", color=_NAVY)
        _apply_clean_style(ax)
        fig.tight_layout()
        return _fig_to_artifact(
            fig,
            chart_id="risk_components",
            section_id="decision_intelligence",
            title="Structural Risk Index Components",
            caption=f"Decomposition of structural dataset and tail risk factors (Overall Risk Index: {sri}, Band: {band}).",
            width_inches=6.2,
            height_inches=3.0,
        )
