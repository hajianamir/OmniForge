"""Interactive Plotly visualization builder with enterprise dark and light themes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class VisualizationEngine:
    """Generates interactive, self-contained Plotly figures for EDA, modeling, and reporting."""

    def __init__(self, theme: str = "dark") -> None:
        self.theme = theme
        self.template = "plotly_dark" if theme == "dark" else "plotly_white"
        self.accent_color = "#38BDF8" if theme == "dark" else "#0284C7"

    def distribution_plot(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Create a histogram + marginal box plot for numeric columns or bar chart for categoricals."""
        series = df[column].dropna()
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            fig = px.histogram(
                df,
                x=column,
                marginal="box",
                nbins=min(40, max(10, int(np.sqrt(max(len(series), 1))))),
                title=f"Distribution of {column}",
                template=self.template,
                color_discrete_sequence=[self.accent_color],
            )
        else:
            vc = series.astype(str).value_counts().head(20).reset_index()
            vc.columns = [column, "count"]
            fig = px.bar(
                vc,
                x=column,
                y="count",
                title=f"Top Categories in {column}",
                template=self.template,
                color_discrete_sequence=[self.accent_color],
            )
        fig.update_layout(margin=dict(l=40, r=20, t=50, b=40))
        return fig

    def correlation_heatmap(self, df: pd.DataFrame, method: str = "pearson") -> go.Figure:
        """Generate an annotated interactive correlation matrix heatmap."""
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            fig = go.Figure()
            fig.update_layout(
                title="Correlation Heatmap (Requires >=2 numeric columns)", template=self.template
            )
            return fig
        corr = num_df.corr(method=method).round(3)
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            title=f"{method.title()} Correlation Matrix",
            template=self.template,
        )
        fig.update_layout(margin=dict(l=50, r=20, t=50, b=50))
        return fig

    def missingness_bar(self, df: pd.DataFrame) -> go.Figure:
        """Visualize missing value percentage per column."""
        miss_pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
        miss_df = miss_pct.reset_index()
        miss_df.columns = ["column", "missing_pct"]
        fig = px.bar(
            miss_df,
            x="column",
            y="missing_pct",
            title="Missing Value Percentage by Column",
            template=self.template,
            color="missing_pct",
            color_continuous_scale="YlOrRd",
        )
        fig.update_layout(yaxis_title="Missing (%)", margin=dict(l=40, r=20, t=50, b=60))
        return fig

    def target_relationship_plot(
        self, df: pd.DataFrame, feature_col: str, target_col: str
    ) -> go.Figure:
        """Plot bivariate relationship between a feature and target column."""
        if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 15:
            if pd.api.types.is_numeric_dtype(df[feature_col]):
                fig = px.scatter(
                    df.head(5000),
                    x=feature_col,
                    y=target_col,
                    title=f"{feature_col} vs {target_col}",
                    template=self.template,
                    opacity=0.7,
                )
            else:
                fig = px.box(
                    df,
                    x=feature_col,
                    y=target_col,
                    title=f"{target_col} by {feature_col}",
                    template=self.template,
                )
        else:
            if pd.api.types.is_numeric_dtype(df[feature_col]):
                fig = px.box(
                    df,
                    x=target_col,
                    y=feature_col,
                    color=target_col,
                    title=f"{feature_col} Distribution by {target_col}",
                    template=self.template,
                )
            else:
                ct = pd.crosstab(df[feature_col].astype(str), df[target_col].astype(str)).reset_index()
                melted = ct.melt(id_vars=feature_col, var_name=target_col, value_name="count")
                fig = px.bar(
                    melted,
                    x=feature_col,
                    y="count",
                    color=target_col,
                    barmode="group",
                    title=f"{feature_col} vs {target_col}",
                    template=self.template,
                )
        fig.update_layout(margin=dict(l=40, r=20, t=50, b=40))
        return fig

    def time_series_plot(
        self,
        df: pd.DataFrame,
        time_col: str,
        value_col: str,
        forecast_df: pd.DataFrame | None = None,
    ) -> go.Figure:
        """Plot historical series with optional forecast and confidence intervals."""
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[value_col],
                mode="lines",
                name="Historical",
                line=dict(color=self.accent_color, width=2),
            )
        )
        if forecast_df is not None and len(forecast_df) > 0:
            x_fc = forecast_df[time_col] if time_col in forecast_df.columns else forecast_df.index
            y_fc = forecast_df["yhat"] if "yhat" in forecast_df.columns else forecast_df.iloc[:, 0]
            fig.add_trace(
                go.Scatter(
                    x=x_fc,
                    y=y_fc,
                    mode="lines",
                    name="Forecast",
                    line=dict(color="#F97316", width=2.5, dash="dash"),
                )
            )
            if "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=list(x_fc) + list(x_fc)[::-1],
                        y=list(forecast_df["yhat_upper"]) + list(forecast_df["yhat_lower"])[::-1],
                        fill="toself",
                        fillcolor="rgba(249, 115, 22, 0.2)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="Prediction Interval",
                    )
                )
        fig.update_layout(
            title=f"Time Series: {value_col}",
            xaxis_title=time_col,
            yaxis_title=value_col,
            template=self.template,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig

    def feature_importance_plot(
        self, importances: dict[str, float], top_n: int = 20, title: str = "Feature Importance"
    ) -> go.Figure:
        """Horizontal bar chart of top feature importances or SHAP attributions."""
        items = sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
        items.reverse()
        fig = go.Figure(
            go.Bar(
                x=[v for _, v in items],
                y=[k for k, _ in items],
                orientation="h",
                marker_color=self.accent_color,
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Importance / Attribution",
            template=self.template,
            margin=dict(l=140, r=20, t=50, b=40),
        )
        return fig

    def quality_radar_plot(self, dimensions: dict[str, float]) -> go.Figure:
        """Spider/radar chart for the six data quality dimensions."""
        labels = [k.title() for k in dimensions.keys()]
        values = list(dimensions.values())
        if labels:
            labels.append(labels[0])
            values.append(values[0])
        fig = go.Figure(
            data=go.Scatterpolar(
                r=values, theta=labels, fill="toself", line_color=self.accent_color
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Six-Dimensional Data Quality Radar",
            template=self.template,
        )
        return fig

    @staticmethod
    def figure_to_html(fig: go.Figure, include_plotlyjs: str | bool = "cdn") -> str:
        """Serialize a Plotly figure to an embeddable HTML div string."""
        return str(fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs))

    @staticmethod
    def figure_to_dict(fig: go.Figure) -> dict[str, Any]:
        """Serialize a Plotly figure into a JSON-compatible dictionary."""
        return dict(fig.to_dict())
