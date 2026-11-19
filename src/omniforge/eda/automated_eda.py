"""One-click Automated EDA orchestrator combining profiling, tests, charts, and narrative."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape as html_escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from omniforge.core.logging import get_logger
from omniforge.eda.insights import AnalyticalInsight, InsightGenerator
from omniforge.eda.profiler import DataProfiler, DatasetProfile
from omniforge.eda.statistics import HypothesisTestResult, StatisticalAnalyzer
from omniforge.eda.visualization import VisualizationEngine

logger = get_logger(__name__)


@dataclass
class EDAReport:
    """Complete Automated EDA deliverable with profile, statistical tests, charts, and narrative."""

    profile: DatasetProfile
    insights: list[AnalyticalInsight]
    narrative: str
    normality_tests: dict[str, HypothesisTestResult] = field(default_factory=dict)
    vif_scores: dict[str, float] = field(default_factory=dict)
    figures: dict[str, go.Figure] = field(default_factory=dict)

    def to_html(self, output_path: str | Path | None = None) -> str:
        """Render the complete EDA report into a standalone HTML document with escaped data fields."""
        insight_items = "".join(
            f"<li><strong>[{html_escape(str(ins.severity).upper(), quote=True)}] "
            f"{html_escape(str(ins.title), quote=True)}</strong>: "
            f"{html_escape(str(ins.description), quote=True)} "
            f"<em>({html_escape(str(ins.recommendation), quote=True)})</em></li>"
            for ins in self.insights
        )
        col_rows = "".join(
            f"<tr><td>{html_escape(str(c.name), quote=True)}</td>"
            f"<td>{html_escape(str(c.dtype), quote=True)}</td>"
            f"<td>{html_escape(str(c.semantic_type), quote=True)}</td>"
            f"<td>{html_escape(str(c.missing_pct), quote=True)}%</td>"
            f"<td>{html_escape(str(c.unique_count), quote=True)}</td>"
            f"<td>{html_escape(str(c.mean), quote=True) if c.mean is not None else '-'}</td>"
            f"<td>{html_escape(str(c.std), quote=True) if c.std is not None else '-'}</td></tr>"
            for c in self.profile.columns
        )
        chart_sections = ""
        for idx, (fig_name, fig) in enumerate(self.figures.items()):
            inc_js = "cdn" if idx == 0 else False
            safe_fig_title = html_escape(str(fig_name).replace("_", " ").title(), quote=True)
            chart_sections += (
                f"<section class='card'><h3>{safe_fig_title}</h3>"
                f"{VisualizationEngine.figure_to_html(fig, include_plotlyjs=inc_js)}</section>"
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>OmniForge Automated EDA Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
  h1, h2, h3 {{ color: #38bdf8; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .kpi {{ background: #1e293b; border-radius: 10px; padding: 16px; border: 1px solid #334155; }}
  .kpi .val {{ font-size: 1.6rem; font-weight: 700; color: #f8fafc; }}
  .card {{ background: #1e293b; border-radius: 10px; padding: 20px; margin-bottom: 20px; border: 1px solid #334155; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #334155; text-align: left; }}
  th {{ color: #94a3b8; }}
</style>
</head>
<body>
<h1>OmniForge — Automated EDA Report</h1>
<div class="grid">
  <div class="kpi"><div>Rows</div><div class="val">{self.profile.n_rows:,}</div></div>
  <div class="kpi"><div>Columns</div><div class="val">{self.profile.n_columns}</div></div>
  <div class="kpi"><div>Missing Cells</div><div class="val">{self.profile.missing_pct:.2f}%</div></div>
  <div class="kpi"><div>Duplicates</div><div class="val">{self.profile.duplicate_rows:,}</div></div>
  <div class="kpi"><div>Memory</div><div class="val">{self.profile.memory_mb:.2f} MB</div></div>
</div>
<div class="card"><h2>Automated Insights</h2><ul>{insight_items}</ul></div>
<div class="card"><h2>Column Profiles</h2>
<table><thead><tr><th>Column</th><th>Dtype</th><th>Semantic Type</th><th>Missing %</th><th>Unique</th><th>Mean</th><th>Std</th></tr></thead>
<tbody>{col_rows}</tbody></table></div>
{chart_sections}
</body>
</html>"""
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
        return html

    def export_html(self, output_path: str | Path) -> Path:
        """Write standalone HTML EDA report to ``output_path`` and return its :class:`Path`."""
        dest = Path(output_path)
        self.to_html(dest)
        return dest


class AutomatedEDA:
    """High-level one-click Automated EDA orchestrator."""

    def __init__(self, theme: str = "dark", max_distribution_plots: int = 8) -> None:
        self.profiler = DataProfiler()
        self.stats = StatisticalAnalyzer()
        self.viz = VisualizationEngine(theme=theme)
        self.insight_gen = InsightGenerator()
        self.max_distribution_plots = max_distribution_plots

    def run(
        self,
        df: pd.DataFrame,
        *,
        target_column: str | None = None,
        generate_figures: bool = True,
    ) -> EDAReport:
        """Execute full Automated EDA workflow on ``df``."""
        profile = self.profiler.profile(df, target_column=target_column)
        insights, narrative = self.insight_gen.generate(df, profile, target_column=target_column)

        normality_tests: dict[str, HypothesisTestResult] = {}
        for col in profile.numeric_columns[:15]:
            normality_tests[col] = self.stats.normality_test(df[col])

        vif_scores = self.stats.compute_vif(df[profile.numeric_columns[:20]]) if len(profile.numeric_columns) >= 2 else {}

        figures: dict[str, go.Figure] = {}
        if generate_figures and len(df) > 0:
            figures["missingness"] = self.viz.missingness_bar(df)
            if len(profile.numeric_columns) >= 2:
                figures["correlation_heatmap"] = self.viz.correlation_heatmap(df)
            for col in list(df.columns)[: self.max_distribution_plots]:
                figures[f"dist_{col}"] = self.viz.distribution_plot(df, str(col))
            if target_column and target_column in df.columns:
                top_feats = list(profile.target_correlations.keys())[:3] or [
                    c for c in df.columns if c != target_column
                ][:3]
                for feat in top_feats:
                    figures[f"target_vs_{feat}"] = self.viz.target_relationship_plot(
                        df, str(feat), target_column
                    )

        logger.info(
            "automated_eda_completed",
            insights=len(insights),
            figures=len(figures),
            vif_computed=len(vif_scores),
        )
        return EDAReport(
            profile=profile,
            insights=insights,
            narrative=narrative,
            normality_tests=normality_tests,
            vif_scores=vif_scores,
            figures=figures,
        )
