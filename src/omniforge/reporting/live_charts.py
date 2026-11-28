"""Live On-Demand Professional Chart Computation Engine (Zero Synthetic Data).

Computes publication-grade statistical, exploratory, modeling, responsible-AI,
and supervisory engineering charts strictly from the caller's live workspace
DataFrame, locked train/test splits, and trained studio artifacts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)

from omniforge.core.types import TaskType
from omniforge.core.utils import to_json_object
from omniforge.data.quality import DataQualityProfiler
from omniforge.eda.shapiro_scope import SHAPIRO_CAP, shapiro_sample_disclosure
from omniforge.mlops.supervisory_monitor import SupervisoryPredictionMonitor
from omniforge.ui.state import (
    get_active_split_frames,
    get_analysis_dataframe,
    get_dataset_processing_status,
    get_excluded_columns,
    get_selected_targets,
)

PALETTE: tuple[str, ...] = (
    "#38bdf8",
    "#a78bfa",
    "#34d399",
    "#fbbf24",
    "#f87171",
    "#fb7185",
    "#2dd4bf",
    "#818cf8",
    "#f97316",
    "#c084fc",
)


class LiveChartUnavailable(ValueError):
    """A user-facing chart prerequisite or supported validation failure."""


@dataclass(frozen=True)
class LiveChartSpec:
    """Declarative specification for an on-demand live chart."""

    id: str
    chart_type: str
    title: str
    category: str
    description: str
    studios: tuple[str, ...]
    requirement: str = "dataset"  # "dataset" | "model_any" | "model_clf" | "opt" | "supervisory"
    supports_feature_x: bool = False
    supports_feature_y: bool = False
    supports_group_by: bool = False
    feature_x_kind: str = "any"  # "numeric" | "categorical" | "any"
    feature_y_kind: str = "any"


LIVE_CHART_SPECS: tuple[LiveChartSpec, ...] = (
    # 1. Gauge Charts
    LiveChartSpec(
        id="dataset_readiness_gauge",
        chart_type="gauge",
        title="Data Quality & Readiness Score Gauge",
        category="Quality, Composition & Pipeline",
        description="6-dimensional dataset readiness dial with enterprise warning (60) and pass (85) thresholds.",
        studios=("00", "01", "03", "12"),
    ),
    LiveChartSpec(
        id="model_champion_gauge",
        chart_type="gauge",
        title="Champion Model Primary Score Gauge",
        category="Model Evaluation & Diagnostics",
        description="Radial score dial for the selected trained model on the displayed evaluation cohort.",
        studios=("03", "06", "07", "11", "12", "14"),
        requirement="model_any",
    ),
    LiveChartSpec(
        id="supervisory_compliance_gauge",
        chart_type="gauge",
        title="Supervisory Tolerance Compliance Gauge",
        category="Supervisory, Drift & SPC",
        description="Operational compliance rate (%) within configured error tolerance bounds in Studio 14.",
        studios=("11", "12", "14"),
        requirement="supervisory",
    ),
    # 2. Radar Charts
    LiveChartSpec(
        id="quality_6d_radar",
        chart_type="radar",
        title="6D Data Quality & Integrity Radar",
        category="Quality, Composition & Pipeline",
        description="Polar profile across Completeness, Uniqueness, Validity, Consistency, Outlier Health & Schema.",
        studios=("00", "01", "03", "12"),
    ),
    LiveChartSpec(
        id="group_centroid_radar",
        chart_type="radar",
        title="Multi-Feature Group / Cluster Centroid Radar",
        category="Relationships, Flow & Multivariate",
        description="Normalized multi-axis feature profiles across clusters, target classes, or categorical strata.",
        studios=("02", "04", "05", "07", "12"),
        supports_group_by=True,
    ),
    LiveChartSpec(
        id="model_multimetric_radar",
        chart_type="radar",
        title="Model Multi-Metric Evaluation Radar",
        category="Model Evaluation & Diagnostics",
        description="Polar comparison of the active trained model vs. reference baseline across 5 core metrics.",
        studios=("03", "06", "07", "11", "12", "14"),
        requirement="model_any",
    ),
    # 3. Donut Charts
    LiveChartSpec(
        id="categorical_share_donut",
        chart_type="donut",
        title="Category / Class Share Donut Chart",
        category="Quality, Composition & Pipeline",
        description="Proportional ring breakdown of real category or target class frequencies.",
        studios=("00", "01", "02", "04", "07", "09", "12"),
        supports_feature_x=True,
        feature_x_kind="any",
    ),
    LiveChartSpec(
        id="schema_split_donut",
        chart_type="donut",
        title="Schema & Split Partition Composition Donut",
        category="Quality, Composition & Pipeline",
        description="Proportional breakdown of Numeric, Categorical, Target, and Excluded study columns.",
        studios=("00", "01", "05", "10", "12"),
    ),
    LiveChartSpec(
        id="portfolio_allocation_donut",
        chart_type="donut",
        title="Optimal Decision / Portfolio Allocation Donut",
        category="Quality, Composition & Pipeline",
        description="Optimal capital/resource allocation weights solved in Studio 08 Optimization.",
        studios=("08", "12"),
        requirement="opt",
    ),
    # 4. Waterfall Charts
    LiveChartSpec(
        id="pipeline_row_waterfall",
        chart_type="waterfall",
        title="Data Pipeline Row Retention Waterfall",
        category="Quality, Composition & Pipeline",
        description="Signed cumulative bridge from Raw Ingested Rows through cleaning/filtering to Active/Split rows.",
        studios=("00", "01", "02", "10", "12"),
    ),
    LiveChartSpec(
        id="model_attribution_waterfall",
        chart_type="waterfall",
        title="Prediction Driver Attribution Waterfall",
        category="Attribution, Effect Size & Pareto",
        description="Signed additive feature attribution bridge from Baseline E[f(X)] to individual prediction f(x).",
        studios=("06", "07", "08", "12", "14"),
        requirement="model_any",
    ),
    # 5. Forest Plots
    LiveChartSpec(
        id="feature_correlation_forest",
        chart_type="forest",
        title="Feature-Target Correlation Forest Plot (95% Fisher-z CI)",
        category="Attribution, Effect Size & Pareto",
        description="Exact Pearson correlations with analytical Fisher z-transform 95% confidence intervals and p-values.",
        studios=("00", "03", "04", "05", "08", "12"),
        supports_feature_x=True,
        feature_x_kind="numeric",
    ),
    LiveChartSpec(
        id="standardized_effect_forest",
        chart_type="forest",
        title="Standardized Multivariate Effect Forest Plot (95% CI)",
        category="Attribution, Effect Size & Pareto",
        description="Standardized multivariate regression coefficients (beta ± 1.96 SE) estimating controlled feature effects.",
        studios=("03", "04", "05", "06", "07", "08", "09", "12"),
        supports_feature_x=True,
        feature_x_kind="numeric",
    ),
    # 6. Pareto Charts
    LiveChartSpec(
        id="missing_variance_pareto",
        chart_type="pareto",
        title="Data Defect / Dispersion 80-20 Pareto Chart",
        category="Attribution, Effect Size & Pareto",
        description="Dual-axis 80/20 Pareto ranking of column missingness (or dispersion when 100% complete).",
        studios=("00", "01", "03", "04", "12"),
    ),
    LiveChartSpec(
        id="pca_eigenvalue_pareto",
        chart_type="pareto",
        title="PCA Eigenvalue Scree & Cumulative Variance Pareto",
        category="Attribution, Effect Size & Pareto",
        description="Principal component explained-variance bars with cumulative % curve and 80% subspace cutoff.",
        studios=("02", "04", "05", "06", "12"),
    ),
    LiveChartSpec(
        id="feature_importance_pareto",
        chart_type="pareto",
        title="Predictive Feature Importance 80-20 Pareto",
        category="Attribution, Effect Size & Pareto",
        description="Ranks features by live predictive importance / mutual information to isolate the 80% vital few.",
        studios=("05", "06", "07", "11", "12", "14"),
    ),
    # 7. ECDF Charts
    LiveChartSpec(
        id="empirical_ecdf_ks",
        chart_type="ecdf",
        title="Empirical CDF (ECDF) & Kolmogorov-Smirnov Comparison",
        category="Distributions, Normality & KDE",
        description="Exact step-function ECDF with quartile guides and two-sample KS statistic across strata/splits.",
        studios=("00", "03", "04", "07", "08", "09", "11", "12"),
        supports_feature_x=True,
        supports_group_by=True,
        feature_x_kind="numeric",
    ),
    # 8. Violin + Box-Whisker Hybrid
    LiveChartSpec(
        id="stratified_violin_box",
        chart_type="violin_box",
        title="Hybrid KDE Violin & Tukey Box-Whisker Plot",
        category="Distributions, Normality & KDE",
        description="Gaussian kernel density envelope + inner IQR box, median, mean diamond, Tukey whiskers & ANOVA test.",
        studios=("02", "03", "04", "05", "07", "08", "12"),
        supports_feature_x=True,
        supports_group_by=True,
        feature_x_kind="numeric",
    ),
    # 9. Normal Q-Q Plot
    LiveChartSpec(
        id="normality_qq_plot",
        chart_type="qq_plot",
        title="Normal Q-Q Probability Plot & 95% Confidence Envelope",
        category="Distributions, Normality & KDE",
        description="Theoretical vs. empirical quantiles with order-statistic 95% envelope and Shapiro-Wilk normality test.",
        studios=("03", "04", "05", "06", "08", "12", "14"),
        supports_feature_x=True,
        feature_x_kind="numeric",
    ),
    # 10. Bivariate OLS Regression + 95% CI/PI
    LiveChartSpec(
        id="bivariate_regression_ci",
        chart_type="regression_ci",
        title="Bivariate Scatter with OLS Fit, 95% CI & Prediction Band",
        category="Relationships, Flow & Multivariate",
        description="Observed scatter with fitted linear regression, Student-t 95% mean confidence band & 95% prediction band.",
        studios=("00", "03", "04", "05", "06", "08", "12"),
        supports_feature_x=True,
        supports_feature_y=True,
        supports_group_by=True,
        feature_x_kind="numeric",
        feature_y_kind="numeric",
    ),
    # 11. ROC Curve
    LiveChartSpec(
        id="classifier_roc_curve",
        chart_type="roc_curve",
        title="ROC Curve, AUC & Youden Optimal Operating Threshold",
        category="Model Evaluation & Diagnostics",
        description="Held-out Receiver Operating Characteristic curve with Gini index and optimal Youden-J threshold.",
        studios=("03", "06", "07", "09", "11", "12", "14"),
        requirement="model_clf",
    ),
    # 12. Precision-Recall Curve
    LiveChartSpec(
        id="classifier_pr_curve",
        chart_type="pr_curve",
        title="Precision-Recall Curve, PR-AUC & Iso-F1 Contours",
        category="Model Evaluation & Diagnostics",
        description="Precision vs. Recall trajectory with class-prevalence baseline, iso-F1 contours, and optimal F1 point.",
        studios=("03", "06", "07", "09", "11", "12", "14"),
        requirement="model_clf",
    ),
    # 13. Calibration / Reliability Diagram
    LiveChartSpec(
        id="classifier_calibration",
        chart_type="calibration",
        title="Probability Calibration (Reliability) & ECE Diagram",
        category="Model Evaluation & Diagnostics",
        description="Predicted probability vs. empirical positive rate with Wilson 95% CIs, Brier Score, and ECE.",
        studios=("03", "06", "07", "11", "12", "14"),
        requirement="model_clf",
    ),
    # 14. Confusion Matrix Heatmap
    LiveChartSpec(
        id="classifier_confusion_matrix",
        chart_type="confusion_matrix",
        title="Annotated Confusion Matrix, Recall % & MCC Heatmap",
        category="Model Evaluation & Diagnostics",
        description="Exact class-by-class confusion matrix with row-normalized recall percentages, Macro-F1, and MCC.",
        studios=("03", "06", "07", "09", "12", "14"),
        requirement="model_clf",
    ),
    # 15. Cumulative Gains & Lift
    LiveChartSpec(
        id="classifier_gain_lift",
        chart_type="gain_lift",
        title="Cumulative Gains & Decile Lift Curve",
        category="Model Evaluation & Diagnostics",
        description="Business targeting curve plotting % population contacted vs. % positive outcomes captured + Wizard curve.",
        studios=("06", "07", "08", "11", "12", "14"),
        requirement="model_clf",
    ),
    # 16. Residual Diagnostics
    LiveChartSpec(
        id="model_residual_diagnostics",
        chart_type="residual_diag",
        title="Residuals vs. Fitted & Heteroscedasticity Diagnostic Plot",
        category="Model Evaluation & Diagnostics",
        description="Evaluates prediction errors (y - yhat) vs. fitted values with ±2σ bands, local bias curve & Durbin-Watson.",
        studios=("03", "06", "07", "08", "12", "14"),
        requirement="model_any",
    ),
    # 17. Bland-Altman Agreement
    LiveChartSpec(
        id="bland_altman_agreement",
        chart_type="bland_altman",
        title="Bland-Altman Method Agreement & 95% Limits of Agreement",
        category="Supervisory, Drift & SPC",
        description="Plots ½(Actual + Predicted) vs. (Actual - Predicted) with Mean Bias and ±1.96 SD Limits of Agreement.",
        studios=("06", "08", "12", "14"),
        requirement="model_any",
    ),
    # 18. Tabular CUSUM Control Chart
    LiveChartSpec(
        id="cusum_sequential_control",
        chart_type="cusum",
        title="Tabular CUSUM (C+ / C-) Sequential Change-Point Chart",
        category="Supervisory, Drift & SPC",
        description="Montgomery upper/lower cumulative sum drift detector (k=0.5σ, H=4.0σ) on sequential observations or residuals.",
        studios=("03", "08", "11", "12", "14"),
        supports_feature_x=True,
        feature_x_kind="numeric",
    ),
    # 19. Lorenz & Gini Concentration Curve
    LiveChartSpec(
        id="lorenz_gini_curve",
        chart_type="lorenz",
        title="Lorenz Concentration Curve & Gini Coefficient",
        category="Distributions, Normality & KDE",
        description="Cumulative population share vs. cumulative outcome concentration with exact Gini and Top-20% share.",
        studios=("00", "02", "04", "07", "08", "12"),
        supports_feature_x=True,
        feature_x_kind="numeric",
    ),
    # 20. Multivariate Parallel Coordinates
    LiveChartSpec(
        id="multivariate_parallel_coords",
        chart_type="parallel_coords",
        title="Multivariate Parallel Coordinates Profile Plot",
        category="Relationships, Flow & Multivariate",
        description="Simultaneous normalized trajectories across top numeric variables colored by class/cluster + group medians.",
        studios=("02", "04", "05", "06", "12"),
        supports_group_by=True,
    ),
    # 21. Ridgeline KDE Joyplot
    LiveChartSpec(
        id="ridgeline_kde_joyplot",
        chart_type="ridgeline",
        title="Ridgeline Multi-Distribution KDE Plot",
        category="Distributions, Normality & KDE",
        description="Vertically stacked smooth Gaussian kernel density curves with median markers across strata or top features.",
        studios=("02", "04", "05", "08", "12"),
        supports_feature_x=True,
        supports_group_by=True,
        feature_x_kind="numeric",
    ),
    # 22. Alluvial / Sankey Flow
    LiveChartSpec(
        id="alluvial_sankey_flow",
        chart_type="sankey",
        title="Alluvial / Sankey Proportional Flow Diagram",
        category="Relationships, Flow & Multivariate",
        description="Mass-conserving flow ribbons connecting source strata/categories to target outcomes or split partitions.",
        studios=("00", "01", "02", "04", "07", "10", "12"),
        supports_feature_x=True,
        supports_feature_y=True,
        feature_x_kind="any",
        feature_y_kind="any",
    ),
    # 23. Lifecycle Funnel
    LiveChartSpec(
        id="data_lifecycle_funnel",
        chart_type="funnel",
        title="CRISP-DM Data & Split Retention Funnel",
        category="Quality, Composition & Pipeline",
        description="Proportional funnel tracking record survival across Raw Intake, Complete Cases, Cleaned Active & Split Lock.",
        studios=("00", "01", "02", "05", "10", "12"),
    ),
    # 24. Vector Correlation Heatmap SVG
    LiveChartSpec(
        id="correlation_heatmap_svg",
        chart_type="heatmap_svg",
        title="Vector Correlation & Association Matrix Heatmap",
        category="Relationships, Flow & Multivariate",
        description="Publication vector correlation matrix (Pearson r + encoded categoricals) with diverging scale & cell annotations.",
        studios=("00", "03", "04", "05", "12"),
    ),
    # 25. Cross-Tabulated Stacked Composition Bar
    LiveChartSpec(
        id="stacked_composition_bar",
        chart_type="stacked_bar",
        title="Cross-Tabulated 100% Stacked Composition & Cramér's V",
        category="Relationships, Flow & Multivariate",
        description="Proportional stacked outcome composition across groups/quartiles with contingency chi-square & Cramér's V.",
        studios=("01", "02", "04", "07", "09", "12"),
        supports_feature_x=True,
        supports_group_by=True,
        feature_x_kind="any",
    ),
)

_SPEC_BY_ID: dict[str, LiveChartSpec] = {s.id: s for s in LIVE_CHART_SPECS}


# ---------------------------------------------------------------------------
# Internal Helpers: Safe Numeric / Categorical / Model Resolution
# ---------------------------------------------------------------------------


def _finite_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _infer_task_type(series: pd.Series) -> TaskType:
    s = series.dropna()
    if s.empty:
        return TaskType.REGRESSION
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return TaskType.BINARY_CLASSIFICATION if s.nunique() <= 2 else TaskType.MULTICLASS_CLASSIFICATION
    arr = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) > 0 and len(np.unique(arr)) <= 10 and np.all(np.isclose(arr, np.round(arr))):
        return TaskType.BINARY_CLASSIFICATION if len(np.unique(arr)) <= 2 else TaskType.MULTICLASS_CLASSIFICATION
    return TaskType.REGRESSION


def _numeric_working_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return a finite numeric DataFrame, encoding categorical columns via frequency/codes if needed."""
    label_notes: dict[str, str] = {}
    out: dict[str, pd.Series] = {}
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_bool_dtype(ser):
            out[str(col)] = ser.astype(float)
        elif pd.api.types.is_numeric_dtype(ser):
            s_num = pd.to_numeric(ser, errors="coerce").replace([np.inf, -np.inf], np.nan)
            if s_num.notna().sum() >= 2:
                out[str(col)] = s_num
        elif pd.api.types.is_datetime64_any_dtype(ser):
            s_dt = pd.to_datetime(ser, errors="coerce")
            vals = s_dt.astype("int64").astype(float) / 1e9
            vals[s_dt.isna()] = np.nan
            if vals.notna().sum() >= 2:
                out[str(col)] = vals
                label_notes[str(col)] = f"{col} (epoch s)"
    if len(out) < 2:
        for col in df.columns:
            c_name = str(col)
            if c_name in out:
                continue
            ser = df[col].dropna().astype(str)
            if ser.nunique() < 2:
                continue
            # Deterministic frequency-rank encoding for purely categorical frames
            freqs = ser.value_counts(normalize=True)
            order = {cat: idx + float(freqs[cat]) for idx, cat in enumerate(freqs.index)}
            mapped = df[col].map(lambda v, m=order: m.get(str(v), np.nan) if pd.notna(v) else np.nan)
            out[c_name] = pd.to_numeric(mapped, errors="coerce")
            label_notes[c_name] = f"{c_name} (cat rank)"
    return pd.DataFrame(out, index=df.index), label_notes


def _discretize_series(ser: pd.Series, max_bins: int = 6) -> pd.Series:
    """Convert a high-cardinality or numeric series into clean human-readable strata."""
    clean = ser.dropna()
    if clean.empty:
        return pd.Series(["Missing"] * len(ser), index=ser.index)
    if pd.api.types.is_numeric_dtype(ser) and not pd.api.types.is_bool_dtype(ser) and clean.nunique() > max_bins:
        try:
            binned = pd.qcut(pd.to_numeric(ser, errors="coerce"), q=min(4, clean.nunique()), duplicates="drop")
            return binned.astype(str).where(ser.notna(), "Missing")
        except Exception:
            pass
    str_ser = ser.where(ser.notna(), "Missing").astype(str)
    top = list(str_ser.value_counts().head(max_bins).index)
    return str_ser.map(lambda v: v if v in top else "Other")


@dataclass
class _ResolvedModelEval:
    model_name: str
    estimator: Any
    target_col: str
    task_type: TaskType
    X_eval: pd.DataFrame
    y_true: pd.Series
    y_pred: np.ndarray
    y_proba: np.ndarray | None
    classes: list[Any]
    feature_cols: list[str]
    eval_scope: str
    decision_scores: np.ndarray | None = None
    probability_error: str | None = None


def _resolve_model_eval(
    state: dict[str, Any], df: pd.DataFrame, *, unavailable: list[str] | None = None,
) -> _ResolvedModelEval | None:
    """Evaluate with the model's fitted schema and the recorded outcome."""

    def explain(reason: str) -> None:
        if unavailable is not None:
            unavailable.append(reason)

    catalog = SupervisoryPredictionMonitor.catalog_available_models(state)
    if not catalog:
        return None
    entry = catalog[0]
    est = entry["_estimator"]
    model_name = str(entry.get("label") or entry.get("model_id") or "Trained Model")
    target_col = str(entry.get("target") or "")
    if target_col not in df.columns:
        explain("The trained model's outcome is absent from the active study.")
        return None

    recorded = SupervisoryPredictionMonitor._evaluation_holdout(state, entry)
    if recorded is not None:
        X_eval, y_eval = recorded
        eval_scope = f"Held-Out Evaluation Set (n={len(X_eval):,})"
    else:
        if entry["model_id"] == "active_champion" and state.get("uap_active_holdout_fingerprint"):
            explain("The saved holdout no longer matches the recorded model inputs and outcome.")
            return None
        try:
            _train_df, test_df, lock = get_active_split_frames(state, df)
        except ValueError:
            explain("The current split cannot be verified for chart scoring.")
            return None
        if test_df is not None and lock is not None:
            eval_df = test_df.dropna(subset=[target_col])
            eval_scope = f"Current Locked Partition (n={len(eval_df):,})"
        else:
            clean_df = df.dropna(subset=[target_col])
            cut = max(2, int(len(clean_df) * 0.75))
            eval_df = clean_df.iloc[cut:] if len(clean_df) - cut >= 8 else clean_df
            eval_scope = f"Evaluation Slice (n={len(eval_df):,}; holdout status unverified)"
        if eval_df.empty:
            explain("The evaluation rows contain no observed values of the model's outcome.")
            return None
        y_eval = eval_df[target_col]
        raw_X = eval_df.drop(columns=[target_col])
        features = getattr(est, "feature_names_in_", None)
        legacy_X = state.get("uap_active_X_test") if entry["model_id"] == "active_champion" else None
        if features is None and isinstance(legacy_X, pd.DataFrame):
            features = legacy_X.columns
        try:
            X_eval = SupervisoryPredictionMonitor._model_input(
                raw_X, entry, None, list(features) if features is not None else None, strict=True,
            )
        except (ValueError, TypeError):
            explain("The fitted model input schema could not be applied to these evaluation rows.")
            return None

    try:
        y_pred = np.asarray(est.predict(X_eval))
    except Exception:
        explain("The fitted model could not score these evaluation rows.")
        return None

    task = SupervisoryPredictionMonitor._model_task(est) or _infer_task_type(y_eval)
    valid = y_pred.shape == (len(X_eval),) and pd.notna(y_pred).all()
    if valid and task == TaskType.REGRESSION:
        valid = np.isfinite(pd.to_numeric(y_pred, errors="coerce")).all()
    if not valid:
        explain("The model emitted missing or invalid predictions for these evaluation rows.")
        return None
    raw_classes = getattr(est, "classes_", None)
    classes: list[Any] = list(raw_classes) if raw_classes is not None else []
    y_proba: np.ndarray | None = None
    decision_scores: np.ndarray | None = None
    probability_error: str | None = None
    if task in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        probability_error = "The estimator does not provide valid class probabilities."
        if hasattr(est, "predict_proba"):
            try:
                raw_p = np.asarray(est.predict_proba(X_eval), dtype=float)
                if (
                    len(classes) >= 2
                    and raw_p.shape == (len(X_eval), len(classes))
                    and np.isfinite(raw_p).all()
                    and (raw_p >= 0.0).all()
                    and (raw_p <= 1.0).all()
                    and np.allclose(raw_p.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
                ):
                    y_proba = raw_p
                    probability_error = None
            except Exception:
                pass
        if y_proba is None and len(classes) == 2 and hasattr(est, "decision_function"):
            try:
                scores = np.asarray(est.decision_function(X_eval), dtype=float)
                if scores.shape == (len(X_eval),) and np.isfinite(scores).all():
                    decision_scores = scores
            except Exception:
                pass

    return _ResolvedModelEval(
        model_name=model_name,
        estimator=est,
        target_col=target_col,
        task_type=task,
        X_eval=X_eval,
        y_true=y_eval.reset_index(drop=True),
        y_pred=y_pred,
        y_proba=y_proba,
        classes=classes,
        feature_cols=list(X_eval.columns),
        eval_scope=eval_scope,
        decision_scores=decision_scores,
        probability_error=probability_error,
    )


def _provenance_banner(state: dict[str, Any], df: pd.DataFrame) -> str:
    status = get_dataset_processing_status(state)
    badge = str(status.get("detailed_label") or ("PROCESSED" if status.get("is_processed") else "RAW"))
    ds_name = str(state.get("uap_dataset_name") or "Active Dataset")
    lock = state.get("uap_split_lock")
    split_txt = (
        f" · Split Locked ({lock.get('n_train', 0):,} train / {lock.get('n_test', 0):,} test)"
        if isinstance(lock, dict) and lock.get("active")
        else ""
    )
    return f"{ds_name} · {badge} ({len(df):,} × {df.shape[1]}){split_txt}"


def _spec_readiness(
    spec: LiveChartSpec,
    state: dict[str, Any],
    has_df: bool,
    model_eval: _ResolvedModelEval | None,
    model_error: str | None = None,
) -> tuple[bool, str | None]:
    if not has_df:
        return False, "Ingest or load a dataset in Studio 00 / 01 first."
    if spec.requirement == "dataset":
        return True, None
    if spec.requirement == "opt":
        opt = state.get("uap_opt_result")
        if isinstance(opt, dict) and (opt.get("solution") or opt.get("weights")):
            return True, None
        return False, "Run Portfolio / Resource Optimization in Studio 08 first."
    if spec.requirement == "supervisory":
        sup = state.get("uap_supervisory_result")
        if isinstance(sup, dict) and sup.get("kpis"):
            if spec.id == "supervisory_compliance_gauge" and not math.isfinite(
                _finite_float(sup["kpis"].get("compliance_rate_pct"), math.nan)
            ):
                return False, "Tolerance compliance is unavailable without evaluated observations."
            return True, None
        return False, "Run a Supervisory Evaluation batch or simulation in Studio 14 first."
    if spec.requirement == "model_any":
        if model_eval is not None or (
            spec.id != "model_champion_gauge" and isinstance(state.get("uap_supervisory_result"), dict)
        ):
            return True, None
        return False, model_error or "Train a model in Studio 03 (Diagnostics), Studio 06 (AutoML / DL), or Studio 14 first."
    if spec.requirement == "model_clf":
        if model_eval is None:
            return False, model_error or "Train a classification model in Studio 03, Studio 06, or Studio 14 first."
        if model_eval.task_type not in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            return (
                False,
                f"Active model '{model_eval.model_name}' targets regression column '{model_eval.target_col}'; classification target required.",
            )
        return True, None
    return True, None


# ---------------------------------------------------------------------------
# Public Catalog API
# ---------------------------------------------------------------------------


def _chart_studio(studio_id: str) -> str:
    studio = (studio_id or "all").strip().lower().removeprefix("s")
    if studio not in {"all", *(f"{index:02d}" for index in range(15))}:
        raise LiveChartUnavailable("Choose a studio from 00 through 14 or the all-studio chart gallery.")
    return studio


def get_live_chart_catalog(state: dict[str, Any], studio_id: str = "all") -> dict[str, Any]:
    """Return available live charts and column selectors for ``studio_id`` (or ``'all'``)."""
    raw_active = state.get("uap_df")
    has_df = isinstance(raw_active, pd.DataFrame) and not raw_active.empty
    df = get_analysis_dataframe(state, raw_active) if has_df else pd.DataFrame()
    has_df = not df.empty

    num_cols = list(df.select_dtypes(include=[np.number]).columns) if has_df else []
    all_cols = [str(c) for c in df.columns] if has_df else []
    cat_cols = [c for c in all_cols if c not in num_cols]
    targets = get_selected_targets(state, df) if has_df else []
    primary_target = targets[0] if targets else None
    task_str = "unsupervised"
    if has_df and primary_target and primary_target in df.columns:
        t_enum = _infer_task_type(df[primary_target])
        task_str = "regression" if t_enum == TaskType.REGRESSION else "classification"

    unavailable: list[str] = []
    model_eval = _resolve_model_eval(state, df, unavailable=unavailable) if has_df else None

    studio_norm = _chart_studio(studio_id)
    charts_out: list[dict[str, Any]] = []
    for spec in LIVE_CHART_SPECS:
        in_studio = studio_norm == "all" or studio_norm in spec.studios
        if not in_studio and studio_norm != "12":
            continue
        ready, reason = _spec_readiness(spec, state, has_df, model_eval, next(iter(unavailable), None))
        charts_out.append(
            {
                "id": spec.id,
                "chart_type": spec.chart_type,
                "title": spec.title,
                "category": spec.category,
                "description": spec.description,
                "studios": list(spec.studios),
                "recommended_for_studio": in_studio,
                "ready": ready,
                "missing_reason": reason,
                "supports_feature_x": spec.supports_feature_x,
                "supports_feature_y": spec.supports_feature_y,
                "supports_group_by": spec.supports_group_by,
                "feature_x_kind": spec.feature_x_kind,
                "feature_y_kind": spec.feature_y_kind,
            }
        )

    # Sort so studio-recommended & ready charts surface first within dropdown groups
    charts_out.sort(key=lambda item: (not item["recommended_for_studio"], not item["ready"], item["category"], item["title"]))

    return to_json_object(
        {
            "studio_id": studio_norm,
            "has_dataset": has_df,
            "dataset_name": state.get("uap_dataset_name"),
            "active_model_name": model_eval.model_name if model_eval else state.get("uap_active_model_name"),
            "columns": {
                "all": all_cols,
                "numeric": [str(c) for c in num_cols],
                "categorical": cat_cols,
                "primary_target": primary_target,
                "task_type": task_str,
            },
            "charts": charts_out,
        }
    )


# ---------------------------------------------------------------------------
# Public Single-Chart Generator API
# ---------------------------------------------------------------------------


def generate_live_chart(
    state: dict[str, Any],
    *,
    chart_id: str,
    studio_id: str = "all",
    feature_x: str | None = None,
    feature_y: str | None = None,
    group_col: str | None = None,
) -> dict[str, Any]:
    """Compute ONLY the single requested ``chart_id`` from live workspace data & model state."""
    spec = _SPEC_BY_ID.get(chart_id)
    if spec is None:
        raise LiveChartUnavailable(f"Unknown chart_id {chart_id!r}. Valid IDs: {sorted(_SPEC_BY_ID)}")
    studio_id = _chart_studio(studio_id)
    if studio_id not in ("all", "12") and studio_id not in spec.studios:
        raise LiveChartUnavailable("The requested chart is outside this studio's chart catalog.")

    raw_active = state.get("uap_df")
    if not isinstance(raw_active, pd.DataFrame) or raw_active.empty:
        raise LiveChartUnavailable("No active dataset loaded. Ingest a dataset in Studio 00 / 01 first.")

    df = get_analysis_dataframe(state, raw_active)
    if df.empty:
        raise LiveChartUnavailable("Active analysis DataFrame is empty after exclusions.")

    unavailable: list[str] = []
    model_eval = _resolve_model_eval(state, df, unavailable=unavailable) if spec.requirement in ("model_any", "model_clf") else None
    ready, reason = _spec_readiness(spec, state, True, model_eval, next(iter(unavailable), None))
    if not ready:
        raise LiveChartUnavailable(reason or f"Chart '{spec.title}' is not ready in the current workspace state.")

    prov = _provenance_banner(state, df)
    targets = get_selected_targets(state, df)
    primary_target = targets[0] if targets else None
    num_df, label_notes = _numeric_working_frame(df)
    num_cols: list[str] = [str(c) for c in num_df.columns]

    def _pick_num_x(prefer_non_target: bool = True) -> str:
        if feature_x and feature_x in num_df.columns:
            return feature_x
        if prefer_non_target:
            non_t = [c for c in num_cols if c != primary_target]
            if non_t:
                return non_t[0]
        if primary_target and primary_target in num_df.columns:
            return primary_target
        if num_cols:
            return num_cols[0]
        raise LiveChartUnavailable("At least one numeric or encodable column is required for this chart.")

    def _pick_num_y(exclude_col: str | None = None) -> str:
        if feature_y and feature_y in num_df.columns and feature_y != exclude_col:
            return feature_y
        if primary_target and primary_target in num_df.columns and primary_target != exclude_col:
            return primary_target
        for c in num_cols:
            if c != exclude_col:
                return c
        return exclude_col if exclude_col is not None else _pick_num_x(False)

    def _disp(col: str) -> str:
        return label_notes.get(col, col)

    # Dispatch strictly to the single requested chart builder
    if chart_id == "dataset_readiness_gauge":
        payload = _build_dataset_readiness_gauge(state, df)
    elif chart_id == "model_champion_gauge":
        payload = _build_model_champion_gauge(model_eval)
    elif chart_id == "supervisory_compliance_gauge":
        payload = _build_supervisory_compliance_gauge(state)
    elif chart_id == "quality_6d_radar":
        payload = _build_quality_6d_radar(state, df)
    elif chart_id == "group_centroid_radar":
        payload = _build_group_centroid_radar(state, df, num_df, group_col, primary_target)
    elif chart_id == "model_multimetric_radar":
        payload = _build_model_multimetric_radar(model_eval)
    elif chart_id == "categorical_share_donut":
        payload = _build_categorical_share_donut(df, feature_x, primary_target)
    elif chart_id == "schema_split_donut":
        payload = _build_schema_split_donut(state, raw_active)
    elif chart_id == "portfolio_allocation_donut":
        payload = _build_portfolio_allocation_donut(state)
    elif chart_id == "pipeline_row_waterfall":
        payload = _build_pipeline_row_waterfall(state, raw_active, df)
    elif chart_id == "model_attribution_waterfall":
        payload = _build_model_attribution_waterfall(state, model_eval)
    elif chart_id == "feature_correlation_forest":
        ref_col = (
            feature_x
            if (feature_x and feature_x in num_df.columns)
            else (primary_target if (primary_target is not None and primary_target in num_df.columns) else _pick_num_x(False))
        )
        payload = _build_feature_correlation_forest(num_df, ref_col, _disp)
    elif chart_id == "standardized_effect_forest":
        ref_col = (
            feature_x
            if (feature_x and feature_x in num_df.columns)
            else (primary_target if (primary_target is not None and primary_target in num_df.columns) else _pick_num_x(False))
        )
        payload = _build_standardized_effect_forest(num_df, ref_col, _disp)
    elif chart_id == "missing_variance_pareto":
        payload = _build_missing_variance_pareto(df, num_df)
    elif chart_id == "pca_eigenvalue_pareto":
        payload = _build_pca_eigenvalue_pareto(num_df)
    elif chart_id == "feature_importance_pareto":
        payload = _build_feature_importance_pareto(state, num_df, primary_target)
    elif chart_id == "empirical_ecdf_ks":
        col_x = _pick_num_x(prefer_non_target=False)
        payload = _build_empirical_ecdf_ks(state, df, num_df, col_x, _disp(col_x), group_col, primary_target)
    elif chart_id == "stratified_violin_box":
        col_x = _pick_num_x(prefer_non_target=True)
        payload = _build_stratified_violin_box(df, num_df, col_x, _disp(col_x), group_col, primary_target)
    elif chart_id == "normality_qq_plot":
        col_x = _pick_num_x(prefer_non_target=False)
        payload = _build_normality_qq_plot(num_df[col_x], _disp(col_x))
    elif chart_id == "bivariate_regression_ci":
        col_x = _pick_num_x(prefer_non_target=True)
        col_y = _pick_num_y(exclude_col=col_x)
        payload = _build_bivariate_regression_ci(df, num_df, col_x, col_y, _disp(col_x), _disp(col_y), group_col, primary_target)
    elif chart_id == "classifier_roc_curve":
        assert model_eval is not None
        payload = _build_classifier_roc_curve(model_eval)
    elif chart_id == "classifier_pr_curve":
        assert model_eval is not None
        payload = _build_classifier_pr_curve(model_eval)
    elif chart_id == "classifier_calibration":
        assert model_eval is not None
        payload = _build_classifier_calibration(model_eval)
    elif chart_id == "classifier_confusion_matrix":
        assert model_eval is not None
        payload = _build_classifier_confusion_matrix(model_eval)
    elif chart_id == "classifier_gain_lift":
        assert model_eval is not None
        payload = _build_classifier_gain_lift(model_eval)
    elif chart_id == "model_residual_diagnostics":
        payload = _build_model_residual_diagnostics(state, model_eval)
    elif chart_id == "bland_altman_agreement":
        payload = _build_bland_altman_agreement(state, model_eval)
    elif chart_id == "cusum_sequential_control":
        col_x = _pick_num_x(prefer_non_target=False) if studio_id != "14" else ""
        payload = _build_cusum_sequential_control(state, num_df, col_x, _disp(col_x), studio_id)
    elif chart_id == "lorenz_gini_curve":
        col_x = _pick_num_x(prefer_non_target=False)
        payload = _build_lorenz_gini_curve(num_df[col_x], _disp(col_x))
    elif chart_id == "multivariate_parallel_coords":
        payload = _build_multivariate_parallel_coords(df, num_df, group_col, primary_target)
    elif chart_id == "ridgeline_kde_joyplot":
        col_x = _pick_num_x(prefer_non_target=True)
        payload = _build_ridgeline_kde_joyplot(df, num_df, col_x, _disp(col_x), group_col, primary_target)
    elif chart_id == "alluvial_sankey_flow":
        payload = _build_alluvial_sankey_flow(df, feature_x, feature_y, primary_target)
    elif chart_id == "data_lifecycle_funnel":
        payload = _build_data_lifecycle_funnel(state, raw_active, df)
    elif chart_id == "correlation_heatmap_svg":
        payload = _build_correlation_heatmap_svg(num_df)
    elif chart_id == "stacked_composition_bar":
        payload = _build_stacked_composition_bar(df, feature_x, group_col, primary_target)
    else:
        raise LiveChartUnavailable(f"Unhandled chart_id {chart_id!r}")

    return to_json_object(
        {
            "chart_id": spec.id,
            "chart_type": spec.chart_type,
            "category": spec.category,
            "studio_id": studio_id,
            "provenance": prov,
            **payload,
        }
    )


# ---------------------------------------------------------------------------
# Individual Mathematical & Statistical Chart Builders (All 25 Chart Types)
# ---------------------------------------------------------------------------


def _quality_profile(state: dict[str, Any], df: pd.DataFrame) -> tuple[float, str, dict[str, float]]:
    stored = state.get("uap_quality")
    if stored is not None and hasattr(stored, "model_dump"):
        stored = stored.model_dump()
    if not isinstance(stored, dict) or not isinstance(stored.get("dimensions"), dict):
        stored = DataQualityProfiler().profile(df).model_dump()
    dimensions = {
        name.title(): float(value)
        for name, value in stored["dimensions"].items()
        if isinstance(value, (int, float)) and np.isfinite(value)
    }
    return float(stored["overall_score"]), str(stored.get("grade") or ""), dimensions


def _build_dataset_readiness_gauge(state: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    score, grade, _ = _quality_profile(state, df)
    missing_pct = round(100.0 * float(df.isna().sum().sum()) / max(1, df.size), 2)
    dup_rows = int(df.duplicated().sum())
    return {
        "title": "Data Quality & Readiness Score Gauge",
        "subtitle": f"Grade {grade or 'Unavailable'} · Missing cells {missing_pct:.2f}% · Duplicate rows {dup_rows:,}",
        "kpis": {
            "Readiness Score": round(score, 2),
            "Quality Grade": grade or "Unavailable",
            "Missing Cell %": missing_pct,
            "Duplicate Rows": dup_rows,
        },
        "data": {
            "value": round(score, 2),
            "min": 0.0,
            "max": 100.0,
            "unit": "/ 100",
            "status_label": f"Grade {grade}" if grade else ("Enterprise Ready" if score >= 85 else "Needs Review"),
            "thresholds": [
                {"value": 60.0, "label": "Min Viable (60)", "color": "#fbbf24"},
                {"value": 85.0, "label": "Target (85)", "color": "#34d399"},
            ],
        },
        "table_columns": ["Metric", "Value", "Scale_Min", "Scale_Max", "Warn_Threshold", "Target_Threshold"],
        "table_rows": [["Data Readiness Score", round(score, 2), 0.0, 100.0, 60.0, 85.0]],
    }


def _build_model_champion_gauge(model_eval: _ResolvedModelEval | None) -> dict[str, Any]:
    if model_eval is None:
        raise LiveChartUnavailable("A model and evaluable outcome cohort are required for the score gauge.")
    if model_eval.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        y_true = model_eval.y_true
        y_pred = model_eval.y_pred
        acc = float(accuracy_score(y_true, y_pred))
        score = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metric_name = "Macro-F1 (%)"
        extra_kpi = {"Accuracy (%)": round(acc * 100.0, 2)}
    else:
        y_true = pd.to_numeric(model_eval.y_true, errors="coerce").to_numpy(dtype=float)
        y_pred = np.asarray(model_eval.y_pred, dtype=float)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        if finite.sum() < 2 or np.ptp(y_true[finite]) == 0:
            raise LiveChartUnavailable("R² requires at least two finite observations and a nonconstant outcome.")
        score = float(r2_score(y_true[finite], y_pred[finite], force_finite=False))
        metric_name = "R² (%)"
        extra_kpi = {"RMSE": round(float(root_mean_squared_error(y_true[finite], y_pred[finite])), 4)}
    if not math.isfinite(score):
        raise LiveChartUnavailable("The model score is undefined on the evaluation cohort.")
    val = round(score * 100.0, 2)
    label = model_eval.model_name
    return {
        "title": f"Champion Model Score Gauge — {label}",
        "subtitle": f"{metric_name} = {val:.2f}% · {model_eval.eval_scope}",
        "kpis": {metric_name: val, **extra_kpi},
        "data": {
            "value": val,
            "min": min(0.0, val),
            "max": 100.0,
            "unit": "%",
            "status_label": label[:28],
            "thresholds": [],
        },
        "table_columns": ["Model", "Metric", "Score_Pct", "Evaluation_Scope"],
        "table_rows": [[label, metric_name, val, model_eval.eval_scope]],
    }


def _build_supervisory_compliance_gauge(state: dict[str, Any]) -> dict[str, Any]:
    sup = state.get("uap_supervisory_result") or {}
    kpis = sup.get("kpis") or {}
    breaches = int(kpis.get("breach_count", 0) or 0)
    warnings = int(kpis.get("warning_count", 0) or 0)
    n_rec = int(kpis.get("n_records", len(sup.get("trend_series") or [])) or 1)
    comp = _finite_float(kpis.get("compliance_rate_pct"), math.nan)
    if not math.isfinite(comp):
        raise LiveChartUnavailable("Tolerance compliance is unavailable without evaluated observations.")
    comp = round(max(0.0, min(100.0, comp)), 2)
    status = str(sup.get("system_status") or "NORMAL")
    return {
        "title": "Supervisory Tolerance Compliance Gauge",
        "subtitle": f"Status: {status} · {breaches} critical breaches & {warnings} warnings across {n_rec:,} live records",
        "kpis": {
            "Compliance (%)": comp,
            "System Status": status,
            "Breaches": breaches,
            "Warnings": warnings,
            "Records Scored": n_rec,
        },
        "data": {
            "value": comp,
            "min": 0.0,
            "max": 100.0,
            "unit": "%",
            "status_label": status,
            "thresholds": [
                {"value": 90.0, "label": "Warning (90%)", "color": "#fbbf24"},
                {"value": 95.0, "label": "SLA Target (95%)", "color": "#34d399"},
            ],
        },
        "table_columns": ["System_Status", "Compliance_Pct", "Breach_Count", "Warning_Count", "Records_Evaluated"],
        "table_rows": [[status, comp, breaches, warnings, n_rec]],
    }


def _build_quality_6d_radar(state: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    overall_act, _, dims_act = _quality_profile(state, df)
    axes = list(dims_act)
    raw_df = state.get("uap_raw_df")
    dims_raw: dict[str, float] = {}
    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty and not raw_df.equals(df):
        _, _, raw_scores = _quality_profile({}, raw_df)
        if all(axis in raw_scores for axis in axes):
            dims_raw = raw_scores
    if not axes:
        raise LiveChartUnavailable("No quality dimensions could be evaluated for this dataset.")
    series: list[dict[str, Any]] = [
        {"name": "Active Dataset", "values": [dims_act[a] for a in axes], "color": "#38bdf8"},
    ]
    if dims_raw:
        series.append({"name": "Raw Intake", "values": [dims_raw[a] for a in axes], "color": "#a78bfa"})

    rows = [[ax, *(round(float(s["values"][idx]), 2) for s in series)] for idx, ax in enumerate(axes)]
    return {
        "title": "6D Data Quality & Integrity Radar",
        "subtitle": f"Weighted Data Quality Index = {overall_act:.2f}/100 across {len(df):,} rows × {df.shape[1]} columns",
        "kpis": {"Data Quality Index": overall_act, **{k: v for k, v in dims_act.items()}},
        "data": {"axes": axes, "series": series, "max_scale": 100.0},
        "table_columns": ["Dimension", *(s["name"] for s in series)],
        "table_rows": rows,
    }


def _build_group_centroid_radar(
    state: dict[str, Any],
    df: pd.DataFrame,
    num_df: pd.DataFrame,
    group_col: str | None,
    primary_target: str | None,
) -> dict[str, Any]:
    # Choose grouping variable: user group_col -> committed cluster -> clustering result -> target strata -> quartiles
    grp_name = "Stratum"
    if group_col and group_col in df.columns:
        grp_ser = _discretize_series(df[group_col], max_bins=4)
        grp_name = group_col
    elif "cluster_id" in df.columns:
        grp_ser = df["cluster_id"].astype(str).map(lambda v: f"Cluster {v}")
        grp_name = "cluster_id"
    elif primary_target and primary_target in df.columns:
        grp_ser = _discretize_series(df[primary_target], max_bins=4)
        grp_name = primary_target
    else:
        first_col = num_df.columns[0]
        grp_ser = _discretize_series(num_df[first_col], max_bins=3)
        grp_name = f"{first_col} Terciles"

    feat_candidates = [c for c in num_df.columns if c != grp_name][:7]
    if len(feat_candidates) < 3:
        feat_candidates = list(num_df.columns[:7])
    if len(feat_candidates) < 3:
        raise LiveChartUnavailable("Radar profile requires at least 3 numeric/encodable features.")

    # Normalize each feature to [0, 100] across the dataset so group centroids compare cleanly
    sub = num_df[feat_candidates].copy()
    mins = sub.min(axis=0)
    spans = (sub.max(axis=0) - mins).replace(0.0, 1.0)
    norm_sub = ((sub - mins) / spans) * 100.0
    norm_sub["_grp"] = grp_ser.to_numpy()
    raw_sub = sub.copy()
    raw_sub["_grp"] = grp_ser.to_numpy()

    top_groups = list(grp_ser.value_counts().head(4).index)
    series: list[dict[str, Any]] = []
    for idx, g in enumerate(top_groups):
        g_norm = norm_sub.loc[norm_sub["_grp"] == g, feat_candidates].mean(axis=0)
        g_raw = raw_sub.loc[raw_sub["_grp"] == g, feat_candidates].mean(axis=0)
        series.append(
            {
                "name": str(g)[:22],
                "values": [round(_finite_float(g_norm[c]), 2) for c in feat_candidates],
                "raw_values": [round(_finite_float(g_raw[c]), 4) for c in feat_candidates],
                "color": PALETTE[idx % len(PALETTE)],
            }
        )

    rows = [
        [c, *(s["values"][i] for s in series), *(s["raw_values"][i] for s in series)]
        for i, c in enumerate(feat_candidates)
    ]
    return {
        "title": f"Multi-Feature Centroid Radar by {grp_name}",
        "subtitle": f"Min-max normalized [0–100] group centroids across {len(feat_candidates)} features ({len(top_groups)} strata)",
        "kpis": {"Grouping Variable": grp_name, "Strata Compared": len(top_groups), "Axes": len(feat_candidates)},
        "data": {"axes": feat_candidates, "series": series, "max_scale": 100.0},
        "table_columns": [
            "Feature",
            *(f"{s['name']}_Norm_0_100" for s in series),
            *(f"{s['name']}_Raw_Mean" for s in series),
        ],
        "table_rows": rows,
    }


def _build_model_multimetric_radar(model_eval: _ResolvedModelEval | None) -> dict[str, Any]:
    if model_eval is None:
        raise LiveChartUnavailable("Train a model first to generate the Model Multi-Metric Evaluation Radar.")

    if model_eval.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        y_t = model_eval.y_true.astype(str)
        y_p = pd.Series(model_eval.y_pred).astype(str)
        acc = float(accuracy_score(y_t, y_p)) * 100.0
        f1_m = float(f1_score(y_t, y_p, average="macro", zero_division=0)) * 100.0
        f1_w = float(f1_score(y_t, y_p, average="weighted", zero_division=0)) * 100.0
        mcc_norm = (float(matthews_corrcoef(y_t, y_p)) + 1.0) * 50.0
        axes = ["Accuracy", "Macro-F1", "Weighted-F1", "Scaled MCC"]
        m_vals = [round(x, 2) for x in (acc, f1_m, f1_w, mcc_norm)]
        # Majority-class baseline on exact same y_true
        maj = y_t.value_counts().index[0]
        y_base = pd.Series([maj] * len(y_t))
        b_acc = float(accuracy_score(y_t, y_base)) * 100.0
        b_f1m = float(f1_score(y_t, y_base, average="macro", zero_division=0)) * 100.0
        b_f1w = float(f1_score(y_t, y_base, average="weighted", zero_division=0)) * 100.0
        b_mcc = (float(matthews_corrcoef(y_t, y_base)) + 1.0) * 50.0
        b_vals = [round(x, 2) for x in (b_acc, b_f1m, b_f1w, b_mcc)]
        if len(model_eval.classes) == 2 and (model_eval.y_proba is not None or model_eval.decision_scores is not None):
            y_bin, scores, _ = _extract_binary_clf_scores(model_eval)
            if len(np.unique(y_bin)) == 2:
                axes.append("ROC AUC")
                m_vals.append(round(float(roc_auc_score(y_bin, scores)) * 100.0, 2))
                b_vals.append(round(float(roc_auc_score(y_bin, np.full(len(y_bin), 1.0))) * 100.0, 2))
    else:
        y_t_arr = pd.to_numeric(model_eval.y_true, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y_p_arr = np.asarray(model_eval.y_pred, dtype=float)
        std_y = float(np.std(y_t_arr)) or 1.0
        r2 = max(0.0, min(1.0, float(r2_score(y_t_arr, y_p_arr)))) * 100.0
        nrmse_score = max(0.0, 1.0 - float(root_mean_squared_error(y_t_arr, y_p_arr)) / (2.0 * std_y)) * 100.0
        nmae_score = max(0.0, 1.0 - float(mean_absolute_error(y_t_arr, y_p_arr)) / (2.0 * std_y)) * 100.0
        corr = abs(_finite_float(stats.pearsonr(y_t_arr, y_p_arr)[0] if len(y_t_arr) >= 3 and np.std(y_p_arr) > 0 else 0.0)) * 100.0
        within_1sd = float(np.mean(np.abs(y_t_arr - y_p_arr) <= std_y)) * 100.0
        axes = ["R² Score", "1 - Norm RMSE", "1 - Norm MAE", "|Pearson r|", "Within ±1σ (%)"]
        m_vals = [round(x, 2) for x in (r2, nrmse_score, nmae_score, corr, within_1sd)]
        mean_pred = np.full_like(y_t_arr, float(np.mean(y_t_arr)))
        b_nrmse = max(0.0, 1.0 - float(root_mean_squared_error(y_t_arr, mean_pred)) / (2.0 * std_y)) * 100.0
        b_nmae = max(0.0, 1.0 - float(mean_absolute_error(y_t_arr, mean_pred)) / (2.0 * std_y)) * 100.0
        b_w1 = float(np.mean(np.abs(y_t_arr - mean_pred) <= std_y)) * 100.0
        b_vals = [0.0, round(b_nrmse, 2), round(b_nmae, 2), 0.0, round(b_w1, 2)]

    series = [
        {"name": model_eval.model_name[:26], "values": m_vals, "color": "#38bdf8"},
        {"name": "Naive Baseline", "values": b_vals, "color": "#a78bfa"},
    ]
    return {
        "title": f"Model Multi-Metric Radar — {model_eval.model_name}",
        "subtitle": f"{model_eval.eval_scope} · Target: {model_eval.target_col}",
        "kpis": {axes[i]: m_vals[i] for i in range(len(axes))},
        "data": {"axes": axes, "series": series, "max_scale": 100.0},
        "table_columns": ["Metric_0_100", model_eval.model_name, "Naive_Baseline"],
        "table_rows": [[axes[i], m_vals[i], b_vals[i]] for i in range(len(axes))],
    }


def _build_categorical_share_donut(
    df: pd.DataFrame,
    feature_x: str | None,
    primary_target: str | None,
) -> dict[str, Any]:
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    col = (
        feature_x
        if (feature_x and feature_x in df.columns)
        else (cat_cols[0] if cat_cols else (primary_target if primary_target in df.columns else str(df.columns[0])))
    )
    ser = _discretize_series(df[col], max_bins=8)
    vc = ser.value_counts()
    total = int(vc.sum()) or 1
    slices: list[dict[str, Any]] = [
        {
            "label": str(lbl),
            "value": int(cnt),
            "pct": round(100.0 * int(cnt) / total, 2),
            "color": PALETTE[idx % len(PALETTE)],
        }
        for idx, (lbl, cnt) in enumerate(vc.items())
    ]
    probs = np.asarray([float(s["value"]) / total for s in slices], dtype=float)
    entropy = float(-np.sum(probs * np.log2(np.maximum(probs, 1e-12))))
    return {
        "title": f"Category / Stratum Share Donut — {col}",
        "subtitle": f"{len(slices)} active segments across {total:,} records · Shannon Entropy H = {entropy:.2f} bits",
        "kpis": {
            "Column": str(col),
            "Total Records": total,
            "Distinct Segments": len(slices),
            "Dominant Share (%)": slices[0]["pct"] if slices else 0.0,
            "Entropy (bits)": round(entropy, 3),
        },
        "data": {
            "slices": slices,
            "center_value": f"{total:,}",
            "center_label": str(col)[:16],
        },
        "table_columns": ["Segment", "Count", "Share_Pct"],
        "table_rows": [[s["label"], s["value"], s["pct"]] for s in slices],
    }


def _build_schema_split_donut(state: dict[str, Any], raw_active: pd.DataFrame) -> dict[str, Any]:
    excluded = set(get_excluded_columns(state, raw_active))
    targets = set(get_selected_targets(state, raw_active))
    num_c = 0
    cat_c = 0
    dt_c = 0
    for c in raw_active.columns:
        if c in excluded or c in targets:
            continue
        if pd.api.types.is_datetime64_any_dtype(raw_active[c]):
            dt_c += 1
        elif pd.api.types.is_numeric_dtype(raw_active[c]):
            num_c += 1
        else:
            cat_c += 1
    raw_items = [
        ("Numeric Predictors", num_c, "#38bdf8"),
        ("Categorical Predictors", cat_c, "#a78bfa"),
        ("Target Column(s)", len(targets), "#34d399"),
        ("Datetime Predictors", dt_c, "#fbbf24"),
        ("Excluded Columns", len(excluded), "#f87171"),
    ]
    non_zero = [it for it in raw_items if it[1] > 0]
    total = sum(it[1] for it in non_zero) or 1
    slices = [
        {"label": lbl, "value": cnt, "pct": round(100.0 * cnt / total, 2), "color": col}
        for lbl, cnt, col in non_zero
    ]
    return {
        "title": "Dataset Schema & Study Role Composition Donut",
        "subtitle": f"{raw_active.shape[1]} total columns ({raw_active.shape[1] - len(excluded)} active in study)",
        "kpis": {lbl: cnt for lbl, cnt, _ in non_zero},
        "data": {"slices": slices, "center_value": str(raw_active.shape[1]), "center_label": "Columns"},
        "table_columns": ["Schema_Role", "Column_Count", "Share_Pct"],
        "table_rows": [[s["label"], s["value"], s["pct"]] for s in slices],
    }


def _build_portfolio_allocation_donut(state: dict[str, Any]) -> dict[str, Any]:
    opt = state.get("uap_opt_result") or {}
    sol = opt.get("solution") or opt.get("weights") or {}
    if isinstance(sol, list):
        pairs = [(str(r.get("asset") or r.get("variable") or f"Var_{i}"), _finite_float(r.get("weight", r.get("value", 0.0)))) for i, r in enumerate(sol) if isinstance(r, dict)]
    elif isinstance(sol, dict):
        pairs = [(str(k), _finite_float(v)) for k, v in sol.items()]
    else:
        pairs = []
    pos_pairs = [(k, abs(v)) for k, v in pairs if abs(v) > 1e-6]
    if not pos_pairs:
        raise LiveChartUnavailable("Optimization result contains no non-zero allocation weights.")
    pos_pairs.sort(key=lambda kv: kv[1], reverse=True)
    total = sum(v for _, v in pos_pairs) or 1.0
    slices = [
        {
            "label": k,
            "value": round(v, 4),
            "pct": round(100.0 * v / total, 2),
            "color": PALETTE[idx % len(PALETTE)],
        }
        for idx, (k, v) in enumerate(pos_pairs[:10])
    ]
    obj_val = _finite_float(opt.get("objective_value"), total)
    return {
        "title": "Optimal Resource / Portfolio Allocation Donut",
        "subtitle": f"Studio 08 Solver · {len(slices)} active allocations · Objective = {obj_val:.4f}",
        "kpis": {"Active Assets": len(slices), "Objective Value": round(obj_val, 4), "Top Weight (%)": slices[0]["pct"]},
        "data": {"slices": slices, "center_value": f"{len(slices)}", "center_label": "Assets"},
        "table_columns": ["Asset_Variable", "Allocation_Weight", "Share_Pct"],
        "table_rows": [[s["label"], s["value"], s["pct"]] for s in slices],
    }


def _build_pipeline_row_waterfall(state: dict[str, Any], raw_active: pd.DataFrame, df: pd.DataFrame) -> dict[str, Any]:
    raw_df = state.get("uap_raw_df")
    start_rows = int(len(raw_df)) if isinstance(raw_df, pd.DataFrame) and not raw_df.empty else int(len(raw_active))
    bars: list[dict[str, Any]] = [
        {"label": "Raw Intake", "delta": start_rows, "start": 0, "end": start_rows, "kind": "start"}
    ]
    running = start_rows
    del_log = state.get("uap_deleted_records_log") or []
    if del_log:
        by_stage: dict[str, int] = {}
        for entry in del_log:
            if isinstance(entry, dict):
                st = str(entry.get("stage") or "Cleaning")[:18]
                rm = int(entry.get("rows_removed", entry.get("rows_deleted", 0)) or 0)
                if rm > 0:
                    by_stage[st] = by_stage.get(st, 0) + rm
        for st, rm in by_stage.items():
            nxt = max(0, running - rm)
            bars.append({"label": st, "delta": -(running - nxt), "start": running, "end": nxt, "kind": "delta"})
            running = nxt

    active_rows = int(len(df))
    if running != active_rows:
        diff = active_rows - running
        bars.append({"label": "Net Transform", "delta": diff, "start": running, "end": active_rows, "kind": "delta"})
        running = active_rows

    lock = state.get("uap_split_lock")
    if isinstance(lock, dict) and lock.get("active"):
        n_test = int(lock.get("n_test") or 0)
        n_train = int(lock.get("n_train") or max(0, running - n_test))
        bars.append({"label": "Holdout Test", "delta": -n_test, "start": running, "end": n_train, "kind": "delta"})
        bars.append({"label": "Train Split", "delta": n_train, "start": 0, "end": n_train, "kind": "total"})
        end_rows = n_train
    else:
        if len(bars) == 1:
            incomp = int(df.isna().any(axis=1).sum())
            comp = active_rows - incomp
            bars.append({"label": "Partial Missing", "delta": -incomp, "start": active_rows, "end": comp, "kind": "delta"})
            bars.append({"label": "Complete Rows", "delta": comp, "start": 0, "end": comp, "kind": "total"})
            end_rows = comp
        else:
            bars.append({"label": "Active Study", "delta": active_rows, "start": 0, "end": active_rows, "kind": "total"})
            end_rows = active_rows

    return {
        "title": "Data Pipeline Row Retention Waterfall",
        "subtitle": f"Bridge from {start_rows:,} initial records to {end_rows:,} modeling records (Δ = {end_rows - start_rows:+,})",
        "kpis": {
            "Initial Rows": start_rows,
            "Active Rows": active_rows,
            "Terminal Stage Rows": end_rows,
            "Retention (%)": round(100.0 * end_rows / max(1, start_rows), 2),
        },
        "data": {"bars": bars, "net_change": end_rows - start_rows},
        "table_columns": ["Pipeline_Stage", "Kind", "Start_Rows", "Signed_Delta", "End_Rows"],
        "table_rows": [[b["label"], b["kind"], b["start"], b["delta"], b["end"]] for b in bars],
    }


def _build_model_attribution_waterfall(state: dict[str, Any], model_eval: _ResolvedModelEval | None) -> dict[str, Any]:
    # Prefer Studio 07 SHAP local explanation or Studio 14 root-cause drivers if available, else compute exact perturbation local attributions on model_eval[0]
    shap_res = state.get("uap_shap_result")
    sup_res = state.get("uap_supervisory_result")

    contributions: list[tuple[str, float]] = []
    base_val = 0.0
    scope_desc = ""

    local_shap = (shap_res.get("local_explanation") or shap_res.get("local")) if isinstance(shap_res, dict) else None
    if local_shap:
        loc = local_shap
        if isinstance(loc, dict):
            base_val = _finite_float(loc.get("base_value", loc.get("expected_value", 0.0)))
            contribs_raw = loc.get("contributions") or loc.get("shap_values") or {}
            if isinstance(contribs_raw, dict):
                contributions = [(str(k), _finite_float(v)) for k, v in contribs_raw.items()]
            elif isinstance(contribs_raw, list):
                contributions = [
                    (str(it.get("feature", f"f{i}")), _finite_float(it.get("shap_value", it.get("contribution", it.get("delta", 0.0)))))
                    for i, it in enumerate(contribs_raw)
                    if isinstance(it, dict)
                ]
            scope_desc = "Studio 07 Local SHAP Attribution (Observation #0)"

    if not contributions and model_eval is not None and len(model_eval.X_eval) >= 2:
        X_ev = model_eval.X_eval
        x0 = X_ev.iloc[[0]].copy()
        if model_eval.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION) and model_eval.y_proba is not None:
            pos_idx = min(1, model_eval.y_proba.shape[1] - 1)
            base_val = float(np.mean(model_eval.y_proba[:, pos_idx]))
            pred0 = float(model_eval.y_proba[0, pos_idx])

            def _score_row(df_row: pd.DataFrame) -> float:
                p = np.asarray(model_eval.estimator.predict_proba(df_row), dtype=float)
                return float(p[0, pos_idx]) if p.ndim == 2 else float(p[0])
        else:
            preds_all = np.asarray(model_eval.y_pred, dtype=float)
            base_val = float(np.mean(preds_all))
            pred0 = float(preds_all[0])

            def _score_row(df_row: pd.DataFrame) -> float:
                return float(np.asarray(model_eval.estimator.predict(df_row), dtype=float)[0])

        # Exact single-feature marginal ablation vs background median/mode, rescaled to sum to (pred0 - base_val)
        ref_row = X_ev.iloc[[0]].copy()
        for c in X_ev.columns:
            if pd.api.types.is_numeric_dtype(X_ev[c]):
                ref_row[c] = float(X_ev[c].median())
            else:
                modes = X_ev[c].mode()
                ref_row[c] = modes.iloc[0] if not modes.empty else X_ev[c].iloc[0]

        raw_deltas: list[tuple[str, float]] = []
        for c in list(X_ev.columns)[:16]:
            perturbed = x0.copy()
            perturbed[c] = ref_row[c].iloc[0]
            try:
                d_c = pred0 - _score_row(perturbed)
            except Exception:
                d_c = 0.0
            raw_deltas.append((str(c), _finite_float(d_c)))

        total_gap = pred0 - base_val
        raw_sum = sum(v for _, v in raw_deltas)
        if abs(raw_sum) > 1e-9:
            scale = total_gap / raw_sum
            contributions = [(k, v * scale) for k, v in raw_deltas]
        else:
            contributions = raw_deltas
        scope_desc = f"{model_eval.model_name} · Test Observation #0"

    if not contributions and isinstance(sup_res, dict):
        rc = sup_res.get("root_cause_analysis") or {}
        drivers = rc.get("all_factor_attributions") or []
        base_val = _finite_float(rc.get("mean_champion_vs_baseline_delta"), 0.0)
        contributions = [(str(d.get("feature", "")), _finite_float(d.get("signed_impact"))) for d in drivers if isinstance(d, dict)]
        scope_desc = "Studio 14 Supervisory Root-Cause Attribution"

    if not contributions:
        raise LiveChartUnavailable("Unable to extract local feature attributions; train a model in Studio 06 first.")

    contributions.sort(key=lambda kv: abs(kv[1]), reverse=True)
    top = contributions[:8]
    tail = sum(v for _, v in contributions[8:])
    if abs(tail) > 1e-6:
        top.append(("Other Features", tail))

    bars: list[dict[str, Any]] = [
        {"label": "Baseline E[f(X)]", "delta": round(base_val, 4), "start": 0.0, "end": round(base_val, 4), "kind": "start"}
    ]
    run = base_val
    for feat, delta in top:
        nxt = run + delta
        bars.append({"label": feat[:18], "delta": round(delta, 4), "start": round(run, 4), "end": round(nxt, 4), "kind": "delta"})
        run = nxt
    bars.append({"label": "Prediction f(x)", "delta": round(run, 4), "start": 0.0, "end": round(run, 4), "kind": "total"})

    return {
        "title": "Prediction Driver Attribution Waterfall",
        "subtitle": f"{scope_desc} · E[f(X)] = {base_val:.4f} → f(x) = {run:.4f}",
        "kpis": {
            "Baseline E[f(X)]": round(base_val, 4),
            "Final Prediction f(x)": round(run, 4),
            "Net Attribution Δ": round(run - base_val, 4),
            "Top Driver": top[0][0] if top else "N/A",
        },
        "data": {"bars": bars, "net_change": round(run - base_val, 4)},
        "table_columns": ["Feature_Stage", "Kind", "Cumulative_Start", "Signed_Contribution", "Cumulative_End"],
        "table_rows": [[b["label"], b["kind"], b["start"], b["delta"], b["end"]] for b in bars],
    }


def _build_feature_correlation_forest(
    num_df: pd.DataFrame,
    ref_col: str,
    disp_fn: Any,
) -> dict[str, Any]:
    others = [c for c in num_df.columns if c != ref_col]
    if not others:
        raise LiveChartUnavailable("Correlation Forest Plot requires at least 2 numeric/encodable columns.")

    rows_out: list[dict[str, Any]] = []
    y_ser = num_df[ref_col]
    for c in others[:16]:
        pair = pd.concat([num_df[c], y_ser], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        n = len(pair)
        if n < 5:
            continue
        x_arr = pair.iloc[:, 0].to_numpy(dtype=float)
        y_arr = pair.iloc[:, 1].to_numpy(dtype=float)
        if np.all(x_arr == x_arr[0]) or np.all(y_arr == y_arr[0]):
            continue
        result = stats.pearsonr(x_arr, y_arr)
        interval = result.confidence_interval(confidence_level=0.95)
        if not np.isfinite([result.statistic, result.pvalue, interval.low, interval.high]).all():
            continue
        rows_out.append(
            {
                "label": disp_fn(c),
                "estimate": float(result.statistic),
                "ci_low": float(interval.low),
                "ci_high": float(interval.high),
                "p_value": float(result.pvalue),
                "n": n,
            }
        )

    if not rows_out:
        raise LiveChartUnavailable(f"No non-constant paired variables available to correlate against '{ref_col}'.")

    rows_out.sort(key=lambda r: abs(r["estimate"]), reverse=True)
    sig_cnt = sum(1 for r in rows_out if r["p_value"] < 0.05)
    return {
        "title": f"Feature Correlation Forest Plot vs. {disp_fn(ref_col)}",
        "subtitle": "Approximate Fisher-z 95% intervals assume independent pairs and bivariate normality. P-values are unadjusted and exploratory.",
        "kpis": {
            "Reference Variable": disp_fn(ref_col),
            "Features Tested": len(rows_out),
            "Unadjusted p<0.05": sig_cnt,
            "Strongest |r|": round(abs(rows_out[0]["estimate"]), 4),
        },
        "data": {"rows": rows_out, "reference_value": 0.0, "x_label": f"Pearson Correlation r with {disp_fn(ref_col)}"},
        "table_columns": ["Feature", "Pearson_r", "CI95_Low", "CI95_High", "P_Value", "Paired_N"],
        "table_rows": [[r["label"], r["estimate"], r["ci_low"], r["ci_high"], r["p_value"], r["n"]] for r in rows_out],
    }


def _build_standardized_effect_forest(
    num_df: pd.DataFrame,
    ref_col: str,
    disp_fn: Any,
) -> dict[str, Any]:
    predictors = [c for c in num_df.columns if c != ref_col][:14]
    if not predictors:
        raise LiveChartUnavailable("Standardized Multivariate Effect Forest Plot requires at least 2 numeric/encodable columns.")

    work = num_df[[*predictors, ref_col]].dropna()
    if len(work) < 8:
        work = num_df[[*predictors, ref_col]].copy()
        for c in work.columns:
            work[c] = work[c].fillna(work[c].median())
        work = work.dropna()
    if len(work) < 6:
        raise LiveChartUnavailable("At least 6 observations are required to estimate multivariate 95% confidence intervals.")

    X_raw = work[predictors].to_numpy(dtype=float)
    y_raw = work[ref_col].to_numpy(dtype=float)
    x_std = np.std(X_raw, axis=0)
    keep_mask = x_std > 1e-12
    if not np.any(keep_mask):
        raise LiveChartUnavailable("All predictor columns have zero variance.")

    kept_cols = [predictors[i] for i in range(len(predictors)) if keep_mask[i]]
    X = (X_raw[:, keep_mask] - np.mean(X_raw[:, keep_mask], axis=0)) / x_std[keep_mask]
    y_sd = float(np.std(y_raw)) or 1.0
    y = (y_raw - float(np.mean(y_raw))) / y_sd

    n, p = X.shape
    lam = 1e-2
    reg = Ridge(alpha=lam, fit_intercept=False)
    reg.fit(X, y)
    beta = np.asarray(reg.coef_, dtype=float)
    resid = y - (X @ beta)
    dof = max(1, n - p - 1)
    sigma2 = float(np.sum(resid**2)) / dof
    xtx_inv = np.linalg.pinv(X.T @ X + lam * np.eye(p))
    cov_beta = sigma2 * (xtx_inv @ (X.T @ X) @ xtx_inv)
    se = np.sqrt(np.maximum(np.diag(cov_beta), 1e-12))
    t_crit = float(stats.t.ppf(0.975, df=dof))
    r2_mult = float(r2_score(y, X @ beta))

    rows_out: list[dict[str, Any]] = []
    for idx, col_name in enumerate(kept_cols):
        b = float(beta[idx])
        s = float(se[idx])
        t_stat = b / s if s > 0 else 0.0
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=dof)))
        rows_out.append(
            {
                "label": disp_fn(col_name),
                "estimate": round(b, 4),
                "ci_low": round(b - t_crit * s, 4),
                "ci_high": round(b + t_crit * s, 4),
                "p_value": round(p_val, 6),
            }
        )
    rows_out.sort(key=lambda r: abs(r["estimate"]), reverse=True)
    return {
        "title": f"Standardized Multivariate Effect Forest Plot → {disp_fn(ref_col)}",
        "subtitle": f"Standardized Ridge-OLS β coefficients ± 95% Student-t CIs (n={n:,}, R²={r2_mult:.3f})",
        "kpis": {
            "Outcome": disp_fn(ref_col),
            "Sample Size n": n,
            "Model R²": round(r2_mult, 4),
            "Top Effect |β|": round(abs(rows_out[0]["estimate"]), 4),
        },
        "data": {"rows": rows_out, "reference_value": 0.0, "x_label": "Standardized Effect Size (SD change in Y per 1 SD in X)"},
        "table_columns": ["Predictor", "Std_Beta", "CI95_Low", "CI95_High", "P_Value"],
        "table_rows": [[r["label"], r["estimate"], r["ci_low"], r["ci_high"], r["p_value"]] for r in rows_out],
    }


def _build_missing_variance_pareto(df: pd.DataFrame, num_df: pd.DataFrame) -> dict[str, Any]:
    miss_counts = df.isna().sum()
    non_zero_miss = miss_counts[miss_counts > 0].sort_values(ascending=False)
    if len(non_zero_miss) >= 2:
        raw_pairs = [(str(col), float(cnt)) for col, cnt in non_zero_miss.head(14).items()]
        metric_lbl = "Missing Cells"
        chart_title = "Column Missingness Concentration 80-20 Pareto"
    else:
        # Dataset has 0 or 1 missing column: rank real robust dispersion (IQR / entropy) across columns
        disp_pairs: list[tuple[str, float]] = []
        for c in num_df.columns:
            s = num_df[c].dropna()
            if len(s) >= 2:
                iqr = float(np.percentile(s, 75) - np.percentile(s, 25))
                sd = float(np.std(s))
                disp_pairs.append((str(c), max(iqr, sd)))
        disp_pairs.sort(key=lambda kv: kv[1], reverse=True)
        raw_pairs = [(k, v) for k, v in disp_pairs[:14] if v > 0]
        if not raw_pairs:
            raw_pairs = [(str(c), float(df[c].nunique(dropna=True))) for c in df.columns[:14]]
        metric_lbl = "Dispersion (max IQR, SD)"
        chart_title = "Feature Dispersion Concentration 80-20 Pareto (0% Missing Dataset)"

    total = sum(v for _, v in raw_pairs) or 1.0
    cum = 0.0
    items: list[dict[str, Any]] = []
    vital_few = 0
    crossed = False
    for lbl, val in raw_pairs:
        share = 100.0 * val / total
        cum += share
        if not crossed:
            vital_few += 1
            if cum >= 80.0:
                crossed = True
        items.append(
            {
                "label": lbl,
                "value": round(val, 4),
                "share_pct": round(share, 2),
                "cumulative_pct": round(min(100.0, cum), 2),
            }
        )

    return {
        "title": chart_title,
        "subtitle": f"Top {vital_few} of {len(items)} columns account for ≥80% of cumulative {metric_lbl.lower()}",
        "kpis": {
            "Metric": metric_lbl,
            "Vital Few (≥80%)": vital_few,
            "Columns Ranked": len(items),
            "Top Column Share (%)": items[0]["share_pct"] if items else 0.0,
        },
        "data": {"items": items, "cutoff_pct": 80.0, "vital_few_count": vital_few, "y_left_label": metric_lbl},
        "table_columns": ["Column", metric_lbl, "Share_Pct", "Cumulative_Pct"],
        "table_rows": [[it["label"], it["value"], it["share_pct"], it["cumulative_pct"]] for it in items],
    }


def _build_pca_eigenvalue_pareto(num_df: pd.DataFrame) -> dict[str, Any]:
    if num_df.shape[1] < 2 or len(num_df) < 4:
        raise LiveChartUnavailable("PCA Eigenvalue Pareto requires at least 2 numeric/encodable columns and 4 rows.")
    work = num_df.copy()
    for c in work.columns:
        med = work[c].median()
        work[c] = work[c].fillna(med if math.isfinite(_finite_float(med)) else 0.0)
    X = work.to_numpy(dtype=float)
    stds = np.std(X, axis=0)
    keep = stds > 1e-12
    if int(np.sum(keep)) < 2:
        raise LiveChartUnavailable("At least 2 non-constant columns are required for PCA decomposition.")
    X_std = (X[:, keep] - np.mean(X[:, keep], axis=0)) / stds[keep]
    _u, s_vals, _vt = np.linalg.svd(X_std, full_matrices=False)
    eig = (s_vals**2) / max(1, len(X_std) - 1)
    total_var = float(np.sum(eig)) or 1.0

    items: list[dict[str, Any]] = []
    cum = 0.0
    vital_few = 0
    crossed = False
    for idx, ev in enumerate(eig[:12], start=1):
        pct = 100.0 * float(ev) / total_var
        cum += pct
        if not crossed:
            vital_few += 1
            if cum >= 80.0:
                crossed = True
        items.append(
            {
                "label": f"PC{idx}",
                "value": round(float(ev), 4),
                "share_pct": round(pct, 2),
                "cumulative_pct": round(min(100.0, cum), 2),
            }
        )

    kaiser_k = int(np.sum(eig >= 1.0))
    return {
        "title": "PCA Eigenvalue Scree & Cumulative Explained Variance Pareto",
        "subtitle": f"{vital_few} principal components capture ≥80% total variance · {kaiser_k} PCs satisfy Kaiser criterion (λ ≥ 1)",
        "kpis": {
            "PCs for 80% Variance": vital_few,
            "Kaiser PCs (λ≥1)": kaiser_k,
            "PC1 Variance (%)": items[0]["share_pct"],
            "PC1+PC2 Variance (%)": round(items[0]["share_pct"] + (items[1]["share_pct"] if len(items) > 1 else 0.0), 2),
        },
        "data": {"items": items, "cutoff_pct": 80.0, "vital_few_count": vital_few, "y_left_label": "Eigenvalue (λ)"},
        "table_columns": ["Principal_Component", "Eigenvalue", "Explained_Variance_Pct", "Cumulative_Variance_Pct"],
        "table_rows": [[it["label"], it["value"], it["share_pct"], it["cumulative_pct"]] for it in items],
    }


def _build_feature_importance_pareto(
    state: dict[str, Any],
    num_df: pd.DataFrame,
    primary_target: str | None,
) -> dict[str, Any]:
    fs_scores = state.get("uap_fs_scores")
    shap_res = state.get("uap_shap_result")
    source_label = ""
    pairs: list[tuple[str, float]] = []

    global_shap = (shap_res.get("global_importance") or shap_res.get("global")) if isinstance(shap_res, dict) else None
    if isinstance(global_shap, dict):
        gi = global_shap
        for k, v in gi.items():
            if isinstance(v, dict):
                pairs.append((str(k), abs(_finite_float(v.get("importance_mean")))))
            else:
                pairs.append((str(k), abs(_finite_float(v))))
        source_label = "Studio 07 Global Permutation / SHAP Importance"
    elif isinstance(fs_scores, dict) and fs_scores:
        pairs = [(str(k), abs(_finite_float(v))) for k, v in fs_scores.items()]
        source_label = "Studio 05 Feature Selection Scores"
    else:
        est = state.get("uap_active_model")
        f_cols = list(state.get("uap_active_feature_cols") or [])
        if est is not None and hasattr(est, "feature_importances_") and f_cols:
            imp = np.asarray(est.feature_importances_, dtype=float)
            if len(imp) == len(f_cols):
                pairs = [(str(f_cols[i]), abs(float(imp[i]))) for i in range(len(f_cols))]
                source_label = f"{state.get('uap_active_model_name', 'Active Model')} Feature Importances"

    if not pairs:
        tgt = primary_target if (primary_target and primary_target in num_df.columns) else (num_df.columns[-1] if len(num_df.columns) else None)
        if tgt is None or num_df.shape[1] < 2:
            raise LiveChartUnavailable("At least 2 columns are required to compute feature importance.")
        y_s = num_df[tgt]
        for c in num_df.columns:
            if c == tgt:
                continue
            valid = pd.concat([num_df[c], y_s], axis=1).dropna()
            if len(valid) >= 4 and valid.iloc[:, 0].std() > 0 and valid.iloc[:, 1].std() > 0:
                r_s = abs(float(stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])[0]))
                pairs.append((str(c), _finite_float(r_s)))
        source_label = f"|Spearman ρ| Predictive Association with '{tgt}'"

    pairs = [(k, v) for k, v in pairs if v > 0]
    if not pairs:
        raise LiveChartUnavailable("No positive feature importance scores available.")
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    top = pairs[:14]
    total = sum(v for _, v in top) or 1.0
    cum = 0.0
    vital = 0
    crossed = False
    items: list[dict[str, Any]] = []
    for k, v in top:
        sh = 100.0 * v / total
        cum += sh
        if not crossed:
            vital += 1
            if cum >= 80.0:
                crossed = True
        items.append({"label": k, "value": round(v, 5), "share_pct": round(sh, 2), "cumulative_pct": round(min(100.0, cum), 2)})

    return {
        "title": "Predictive Feature Importance 80-20 Pareto",
        "subtitle": f"{source_label} · Top {vital} features drive ≥80% of total importance",
        "kpis": {
            "Source": source_label[:34],
            "Vital Few (≥80%)": vital,
            "Top Feature": items[0]["label"],
            "Top Feature Share (%)": items[0]["share_pct"],
        },
        "data": {"items": items, "cutoff_pct": 80.0, "vital_few_count": vital, "y_left_label": "Importance Score"},
        "table_columns": ["Feature", "Importance_Score", "Share_Pct", "Cumulative_Pct"],
        "table_rows": [[it["label"], it["value"], it["share_pct"], it["cumulative_pct"]] for it in items],
    }


def _ecdf_points(arr: np.ndarray, max_pts: int = 220) -> tuple[list[float], list[float]]:
    s = np.sort(arr)
    n = len(s)
    if n <= max_pts:
        xs = [round(float(v), 5) for v in s]
        ys = [round(float((i + 1) / n), 5) for i in range(n)]
        return xs, ys
    idx = np.unique(np.linspace(0, n - 1, max_pts).round().astype(int))
    xs = [round(float(s[i]), 5) for i in idx]
    ys = [round(float((i + 1) / n), 5) for i in idx]
    return xs, ys


def _build_empirical_ecdf_ks(
    state: dict[str, Any],
    df: pd.DataFrame,
    num_df: pd.DataFrame,
    col_x: str,
    disp_x: str,
    group_col: str | None,
    primary_target: str | None,
) -> dict[str, Any]:
    ser = num_df[col_x]
    series_out: list[dict[str, Any]] = []
    ks_stat: float | None = None
    ks_p: float | None = None
    group_desc = "Full Active Sample"

    # Stratify if user selected group_col, or if Split Lock is active, or if binary/small categorical target exists
    raw_groups: list[tuple[str, np.ndarray]] = []
    if group_col and group_col in df.columns and group_col != col_x:
        g_ser = _discretize_series(df[group_col], max_bins=4)
        for g_val in g_ser.value_counts().head(4).index:
            arr = ser.loc[g_ser == g_val].dropna().to_numpy(dtype=float)
            if len(arr) >= 3:
                raw_groups.append((f"{group_col}={g_val}", arr))
        group_desc = f"Stratified by {group_col}"
    else:
        tr_df, te_df, lock = get_active_split_frames(state, df)
        if tr_df is not None and te_df is not None and lock is not None and col_x in tr_df.columns and col_x in te_df.columns:
            tr_a = pd.to_numeric(tr_df[col_x], errors="coerce").dropna().to_numpy(dtype=float)
            te_a = pd.to_numeric(te_df[col_x], errors="coerce").dropna().to_numpy(dtype=float)
            if len(tr_a) >= 3 and len(te_a) >= 3:
                raw_groups = [("Train Split", tr_a), ("Test Split", te_a)]
                group_desc = "Train vs. Locked Test Split"
        elif primary_target and primary_target in df.columns and primary_target != col_x and df[primary_target].nunique(dropna=True) <= 4:
            g_ser = df[primary_target].dropna().astype(str)
            for g_val in g_ser.value_counts().head(4).index:
                arr = ser.loc[df[primary_target].astype(str) == g_val].dropna().to_numpy(dtype=float)
                if len(arr) >= 3:
                    raw_groups.append((f"{primary_target}={g_val}", arr))
            if len(raw_groups) >= 2:
                group_desc = f"Stratified by {primary_target}"

    if not raw_groups:
        full_arr = ser.dropna().to_numpy(dtype=float)
        if len(full_arr) < 2:
            raise LiveChartUnavailable(f"Column '{col_x}' has fewer than 2 finite values.")
        raw_groups = [(disp_x, full_arr)]

    if len(raw_groups) >= 2:
        ks_res = stats.ks_2samp(raw_groups[0][1], raw_groups[1][1])
        ks_stat = round(float(ks_res.statistic), 4)
        ks_p = round(float(ks_res.pvalue), 6)

    table_rows: list[list[Any]] = []
    for idx, (g_name, arr) in enumerate(raw_groups):
        xs, ys = _ecdf_points(arr)
        q25, med, q75 = (round(float(v), 4) for v in np.percentile(arr, [25, 50, 75]))
        series_out.append(
            {
                "name": g_name[:26],
                "x": xs,
                "y": ys,
                "n": int(len(arr)),
                "q25": q25,
                "median": med,
                "q75": q75,
                "color": PALETTE[idx % len(PALETTE)],
            }
        )
        for xv, yv in zip(xs, ys, strict=True):
            table_rows.append([g_name, xv, yv])

    sub_ks = f" · 2-Sample KS D = {ks_stat:.4f} (p = {ks_p:.4g})" if ks_stat is not None and ks_p is not None else ""
    kpis: dict[str, Any] = {
        "Variable": disp_x,
        "Median (P50)": series_out[0]["median"],
        "IQR (P75-P25)": round(series_out[0]["q75"] - series_out[0]["q25"], 4),
    }
    if ks_stat is not None:
        kpis["KS Statistic D"] = ks_stat
        kpis["KS p-value"] = ks_p

    return {
        "title": f"Empirical Cumulative Distribution (ECDF) — {disp_x}",
        "subtitle": f"{group_desc}{sub_ks}",
        "kpis": kpis,
        "data": {"series": series_out, "x_label": disp_x, "ks_stat": ks_stat, "ks_pvalue": ks_p},
        "table_columns": ["Stratum", col_x, "Cumulative_Probability_Fn"],
        "table_rows": table_rows[:600],
    }


def _kde_curve(arr: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(arr) < 2 or float(np.std(arr)) <= 1e-12:
        dens = np.exp(-0.5 * ((grid - float(np.mean(arr))) / 0.1) ** 2)
        return dens / max(float(np.max(dens)), 1e-12)
    try:
        kde = stats.gaussian_kde(arr)
        dens = kde(grid)
        mx = float(np.max(dens))
        return dens / mx if mx > 0 else np.zeros_like(grid)
    except Exception:
        hist, edges = np.histogram(arr, bins=min(20, max(5, len(arr) // 4)), range=(float(grid[0]), float(grid[-1])), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        interp = np.interp(grid, centers, hist)
        mx = float(np.max(interp))
        return interp / mx if mx > 0 else np.zeros_like(grid)


def _build_stratified_violin_box(
    df: pd.DataFrame,
    num_df: pd.DataFrame,
    col_x: str,
    disp_x: str,
    group_col: str | None,
    primary_target: str | None,
) -> dict[str, Any]:
    chosen_grp = (
        group_col
        if (group_col and group_col in df.columns and group_col != col_x)
        else (primary_target if (primary_target and primary_target in df.columns and primary_target != col_x and df[primary_target].nunique(dropna=True) <= 8) else None)
    )

    samples: list[tuple[str, np.ndarray]] = []
    if chosen_grp:
        g_ser = _discretize_series(df[chosen_grp], max_bins=6)
        for g_val in g_ser.value_counts().head(6).index:
            arr = num_df.loc[g_ser == g_val, col_x].dropna().to_numpy(dtype=float)
            if len(arr) >= 3:
                samples.append((str(g_val)[:20], arr))
        y_label = disp_x
        title_str = f"Violin & Tukey Box-Whisker — {disp_x} by {chosen_grp}"
    else:
        # Compare standardized z-distributions across top numeric columns
        for c in list(num_df.columns)[:6]:
            s = num_df[c].dropna().to_numpy(dtype=float)
            if len(s) >= 3:
                sd = float(np.std(s))
                z = (s - float(np.mean(s))) / sd if sd > 1e-12 else s - float(np.mean(s))
                samples.append((str(c)[:18], z))
        y_label = "Standardized Z-Score ((x - μ) / σ)" if len(samples) > 1 else disp_x
        title_str = "Multi-Feature Standardized Violin & Box-Whisker Comparison"

    if not samples:
        arr = num_df[col_x].dropna().to_numpy(dtype=float)
        if len(arr) < 3:
            raise LiveChartUnavailable("Violin & Boxplot requires at least 3 finite observations.")
        samples = [(disp_x, arr)]
        y_label = disp_x
        title_str = f"Violin & Tukey Box-Whisker — {disp_x}"

    all_vals = np.concatenate([a for _, a in samples])
    g_min, g_max = float(np.min(all_vals)), float(np.max(all_vals))
    if g_max <= g_min:
        g_min -= 1.0
        g_max += 1.0

    groups_out: list[dict[str, Any]] = []
    table_rows: list[list[Any]] = []
    for idx, (lbl, arr) in enumerate(samples):
        lo_a, q25, med, q75, hi_a = (float(v) for v in np.percentile(arr, [0, 25, 50, 75, 100]))
        mean_v = float(np.mean(arr))
        iqr = q75 - q25
        w_lo = float(np.min(arr[arr >= q25 - 1.5 * iqr])) if np.any(arr >= q25 - 1.5 * iqr) else lo_a
        w_hi = float(np.max(arr[arr <= q75 + 1.5 * iqr])) if np.any(arr <= q75 + 1.5 * iqr) else hi_a
        outs = arr[(arr < w_lo) | (arr > w_hi)]
        if len(outs) > 24:
            outs = np.quantile(outs, np.linspace(0, 1, 24))
        grid = np.linspace(lo_a, hi_a, 36) if hi_a > lo_a else np.linspace(lo_a - 0.5, hi_a + 0.5, 12)
        dens = _kde_curve(arr, grid)
        groups_out.append(
            {
                "label": lbl,
                "n": int(len(arr)),
                "min": round(lo_a, 4),
                "q25": round(q25, 4),
                "median": round(med, 4),
                "mean": round(mean_v, 4),
                "q75": round(q75, 4),
                "max": round(hi_a, 4),
                "whisker_low": round(w_lo, 4),
                "whisker_high": round(w_hi, 4),
                "outliers": [round(float(o), 4) for o in outs],
                "kde_y": [round(float(v), 4) for v in grid],
                "kde_density": [round(float(d), 4) for d in dens],
                "color": PALETTE[idx % len(PALETTE)],
            }
        )
        table_rows.append([lbl, int(len(arr)), round(mean_v, 4), round(lo_a, 4), round(q25, 4), round(med, 4), round(q75, 4), round(hi_a, 4), int(len(outs))])

    anova_p: float | None = None
    if chosen_grp and len(samples) >= 2:
        try:
            _f_stat, p_v = stats.f_oneway(*(a for _, a in samples))
            if math.isfinite(float(p_v)):
                anova_p = round(float(p_v), 6)
        except Exception:
            anova_p = None

    sub = f"{len(groups_out)} distributions · Total n = {len(all_vals):,}" + (f" · One-way ANOVA p = {anova_p:.4g}" if anova_p is not None else "")
    kpis: dict[str, Any] = {"Distributions": len(groups_out), "Total Observations": int(len(all_vals))}
    if anova_p is not None:
        kpis["ANOVA p-value"] = anova_p

    return {
        "title": title_str,
        "subtitle": sub,
        "kpis": kpis,
        "data": {"groups": groups_out, "y_label": y_label, "global_min": round(g_min, 4), "global_max": round(g_max, 4)},
        "table_columns": ["Group", "N", "Mean", "Min", "Q25", "Median", "Q75", "Max", "Outlier_Count"],
        "table_rows": table_rows,
    }


def _build_normality_qq_plot(ser: pd.Series, disp_x: str) -> dict[str, Any]:
    arr = pd.to_numeric(ser, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(arr) < 5:
        raise LiveChartUnavailable(f"Q-Q Plot requires at least 5 finite observations in '{disp_x}'.")

    arr_sorted = np.sort(arr)
    n = len(arr_sorted)
    # Blom's plotting positions: (i - 0.375) / (n + 0.25)
    probs = (np.arange(1, n + 1, dtype=float) - 0.375) / (n + 0.25)
    theo_z = stats.norm.ppf(probs)

    q25_e, q75_e = float(np.percentile(arr_sorted, 25)), float(np.percentile(arr_sorted, 75))
    z25, z75 = float(stats.norm.ppf(0.25)), float(stats.norm.ppf(0.75))
    slope = (q75_e - q25_e) / (z75 - z25) if q75_e > q25_e else (float(np.std(arr_sorted)) or 1.0)
    intercept = float(np.median(arr_sorted)) - slope * 0.0

    # Pointwise 95% order-statistic confidence band around reference line
    pdf_z = np.maximum(stats.norm.pdf(theo_z), 1e-6)
    se_q = (slope / pdf_z) * np.sqrt(probs * (1.0 - probs) / n)

    if n > 260:
        keep_idx = np.unique(np.concatenate([np.arange(0, min(15, n)), np.linspace(0, n - 1, 230).round().astype(int), np.arange(max(0, n - 15), n)]))
    else:
        keep_idx = np.arange(n)

    pts = [
        {
            "theo": round(float(theo_z[i]), 4),
            "emp": round(float(arr_sorted[i]), 4),
            "ref": round(float(intercept + slope * theo_z[i]), 4),
            "lower": round(float(intercept + slope * theo_z[i] - 1.96 * se_q[i]), 4),
            "upper": round(float(intercept + slope * theo_z[i] + 1.96 * se_q[i]), 4),
        }
        for i in keep_idx
    ]

    shapiro_sample = arr_sorted if n <= SHAPIRO_CAP else arr_sorted[np.linspace(0, n - 1, SHAPIRO_CAP).round().astype(int)]
    evaluated = float(np.std(shapiro_sample)) > 1e-12
    if evaluated:
        sw_stat, sw_p = stats.shapiro(shapiro_sample)
        sw_label = f"W = {float(sw_stat):.4f} (p = {float(sw_p):.4g})"
        sw_w, sw_p_value = round(float(sw_stat), 4), round(float(sw_p), 6)
        skew_v = float(stats.skew(arr_sorted))
        kurt_v = float(stats.kurtosis(arr_sorted))
    else:
        sw_label = "not run (no variation)"
        sw_w, sw_p_value = None, None
        skew_v, kurt_v = 0.0, 0.0
    scope, note = shapiro_sample_disclosure(
        source_rows=len(ser),
        available_rows=n,
        tested_rows=len(shapiro_sample) if evaluated else 0,
        cap_omitted_rows=n - len(shapiro_sample),
        selection="evenly_spaced_sorted_ranks",
    )
    return {
        "title": f"Normal Q-Q Probability Plot — {disp_x}",
        # The chart image exporter truncates long subtitles; lead with the Shapiro sample.
        "subtitle": f"Shapiro-Wilk: {note} · {sw_label} · Skew = {skew_v:.3f}, Excess Kurtosis = {kurt_v:.3f}",
        "shapiro_sample": scope,
        "kpis": {
            "Shapiro-Wilk W": sw_w,
            "Shapiro p-value": sw_p_value,
            "Skewness": round(skew_v, 4),
            "Excess Kurtosis": round(kurt_v, 4),
            "Sample Size n": n,
        },
        "data": {
            "points": pts,
            "ref_slope": round(slope, 5),
            "ref_intercept": round(intercept, 5),
            "variable_name": disp_x,
        },
        "table_columns": ["Theoretical_Normal_Quantile", "Empirical_Quantile", "Reference_Line", "CI95_Lower", "CI95_Upper"],
        "table_rows": [[p["theo"], p["emp"], p["ref"], p["lower"], p["upper"]] for p in pts],
    }


def _build_bivariate_regression_ci(
    df: pd.DataFrame,
    num_df: pd.DataFrame,
    col_x: str,
    col_y: str,
    disp_x: str,
    disp_y: str,
    group_col: str | None,
    primary_target: str | None,
) -> dict[str, Any]:
    g_col = group_col if (group_col and group_col in df.columns) else (primary_target if (primary_target and primary_target in df.columns and primary_target not in (col_x, col_y) and df[primary_target].nunique(dropna=True) <= 6) else None)
    pair = pd.DataFrame({"x": num_df[col_x], "y": num_df[col_y]})
    if g_col:
        pair["group"] = _discretize_series(df[g_col], max_bins=6)
    else:
        pair["group"] = "Observation"
    pair = pair.dropna(subset=["x", "y"])
    n = len(pair)
    if n < 4:
        raise LiveChartUnavailable(f"At least 4 paired finite rows are required between '{col_x}' and '{col_y}'.")

    x_arr = pair["x"].to_numpy(dtype=float)
    y_arr = pair["y"].to_numpy(dtype=float)
    x_mean = float(np.mean(x_arr))
    y_mean = float(np.mean(y_arr))
    s_xx = float(np.sum((x_arr - x_mean) ** 2))

    if s_xx > 1e-12 and float(np.std(y_arr)) > 1e-12:
        res_lin = stats.linregress(x_arr, y_arr)
        slope = float(res_lin.slope)
        intercept = float(res_lin.intercept)
        r_val = float(res_lin.rvalue)
        p_val = float(res_lin.pvalue)
        rho_val = _finite_float(stats.spearmanr(x_arr, y_arr)[0])
    else:
        slope, intercept, r_val, p_val, rho_val = 0.0, y_mean, 0.0, 1.0, 0.0

    r2 = r_val**2
    y_hat = intercept + slope * x_arr
    dof = max(1, n - 2)
    s_e = math.sqrt(float(np.sum((y_arr - y_hat) ** 2)) / dof)
    t_crit = float(stats.t.ppf(0.975, df=dof))

    x_grid = np.linspace(float(np.min(x_arr)), float(np.max(x_arr)), 50)
    band = []
    for xg in x_grid:
        yg = intercept + slope * float(xg)
        lev = (1.0 / n) + (((float(xg) - x_mean) ** 2) / s_xx if s_xx > 1e-12 else 0.0)
        ci_half = t_crit * s_e * math.sqrt(lev)
        pi_half = t_crit * s_e * math.sqrt(1.0 + lev)
        band.append(
            {
                "x": round(float(xg), 5),
                "yhat": round(yg, 5),
                "ci_low": round(yg - ci_half, 5),
                "ci_high": round(yg + ci_half, 5),
                "pi_low": round(yg - pi_half, 5),
                "pi_high": round(yg + pi_half, 5),
            }
        )

    # Deterministic stride downsampling to at most 450 scatter points for crisp vector SVG
    if n > 450:
        idx = np.linspace(0, n - 1, 450).round().astype(int)
        sub_p = pair.iloc[idx]
    else:
        sub_p = pair

    scatter_pts: list[dict[str, Any]] = [
        {"x": round(float(r.x), 5), "y": round(float(r.y), 5), "group": str(r.group)}
        for r in sub_p.itertuples(index=False)
    ]

    sign_str = "+" if intercept >= 0 else "-"
    eq_str = f"ŷ = {slope:.4g}·x {sign_str} {abs(intercept):.4g}"
    return {
        "title": f"OLS Regression & 95% Confidence/Prediction Fan — {disp_y} vs. {disp_x}",
        "subtitle": f"{eq_str} · R² = {r2:.4f}, Pearson r = {r_val:.4f} (p = {p_val:.4g}), Spearman ρ = {rho_val:.4f}",
        "kpis": {
            "R²": round(r2, 4),
            "Pearson r": round(r_val, 4),
            "Spearman ρ": round(rho_val, 4),
            "Slope β₁": round(slope, 5),
            "Slope p-value": round(p_val, 6),
            "Paired n": n,
        },
        "data": {
            "points": scatter_pts,
            "band": band,
            "x_label": disp_x,
            "y_label": disp_y,
            "equation": eq_str,
        },
        "table_columns": [col_x, col_y, "Stratum", "OLS_Fitted"],
        "table_rows": [
            [p["x"], p["y"], p["group"], round(intercept + slope * float(p["x"]), 5)]
            for p in scatter_pts
        ],
    }


def _extract_binary_clf_scores(
    model_eval: _ResolvedModelEval, *, require_probabilities: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return one-vs-rest labels and validated probabilities or binary decision scores."""
    classes = model_eval.classes
    if model_eval.y_proba is not None and len(classes) >= 2:
        if len(classes) == 2:
            pos_idx = 1
        else:
            counts = model_eval.y_true.value_counts()
            observed = [(int(counts.get(label, 0)), pos) for pos, label in enumerate(classes)]
            pos_idx = min((entry for entry in observed if entry[0] > 0), default=(0, 0))[1]
        label = classes[pos_idx]
        scores = model_eval.y_proba[:, pos_idx]
    elif not require_probabilities and model_eval.decision_scores is not None and len(classes) == 2:
        label = classes[1]
        scores = model_eval.decision_scores
    else:
        raise LiveChartUnavailable(model_eval.probability_error or "The estimator does not provide valid scores for this chart.")
    y_bin = (model_eval.y_true == label).to_numpy(dtype=int)
    return y_bin, np.asarray(scores, dtype=float), str(label)


def _build_classifier_roc_curve(model_eval: _ResolvedModelEval) -> dict[str, Any]:
    y_bin, scores, pos_cls = _extract_binary_clf_scores(model_eval)
    if len(np.unique(y_bin)) < 2:
        raise LiveChartUnavailable("Evaluation split contains only a single target class; ROC curve requires both positive and negative examples.")

    fpr, tpr, thresholds = roc_curve(y_bin, scores)
    auc_val = float(roc_auc_score(y_bin, scores))
    gini = 2.0 * auc_val - 1.0
    j_scores = tpr - fpr
    best_i = int(np.argmax(j_scores))
    opt_fpr = float(fpr[best_i])
    opt_tpr = float(tpr[best_i])
    opt_thr = _finite_float(thresholds[best_i], 0.5)

    step = max(1, len(fpr) // 200)
    idx = sorted(set(range(0, len(fpr), step)) | {best_i, len(fpr) - 1})
    pts = [
        {"fpr": round(float(fpr[i]), 5), "tpr": round(float(tpr[i]), 5), "threshold": round(_finite_float(thresholds[i], 0.0), 5)}
        for i in idx
    ]
    return {
        "title": f"ROC Curve & Youden Threshold — {model_eval.model_name}",
        "subtitle": f"Positive Class '{pos_cls}' ({model_eval.eval_scope}, {'probabilities' if model_eval.y_proba is not None else 'decision scores'}) · ROC-AUC = {auc_val:.4f}, Gini = {gini:.4f}",
        "kpis": {
            "ROC-AUC": round(auc_val, 4),
            "Gini (2·AUC-1)": round(gini, 4),
            "Youden J*": round(float(j_scores[best_i]), 4),
            "Optimal Threshold": round(opt_thr, 4),
            "TPR at Optimal": round(opt_tpr, 4),
            "FPR at Optimal": round(opt_fpr, 4),
        },
        "data": {
            "points": pts,
            "auc": round(auc_val, 4),
            "gini": round(gini, 4),
            "optimal_point": {"fpr": round(opt_fpr, 4), "tpr": round(opt_tpr, 4), "threshold": round(opt_thr, 4)},
            "positive_class": pos_cls,
        },
        "table_columns": ["False_Positive_Rate", "True_Positive_Rate", "Score_Threshold"],
        "table_rows": [[p["fpr"], p["tpr"], p["threshold"]] for p in pts],
    }


def _build_classifier_pr_curve(model_eval: _ResolvedModelEval) -> dict[str, Any]:
    y_bin, scores, pos_cls = _extract_binary_clf_scores(model_eval)
    if len(np.unique(y_bin)) < 2:
        raise LiveChartUnavailable("Evaluation split contains only a single target class; PR curve requires both classes.")

    precision, recall, thresholds = precision_recall_curve(y_bin, scores)
    ap_val = float(average_precision_score(y_bin, scores))
    prevalence = float(np.mean(y_bin))
    f1_arr = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-12)
    best_i = int(np.argmax(f1_arr))
    opt_rec = float(recall[best_i])
    opt_prec = float(precision[best_i])
    opt_f1 = float(f1_arr[best_i])
    opt_thr = _finite_float(thresholds[min(best_i, max(0, len(thresholds) - 1))], 0.5) if len(thresholds) else 0.5

    # Sort by recall ascending for clean curve rendering
    order = np.argsort(recall)
    rec_s = recall[order]
    prec_s = precision[order]
    step = max(1, len(rec_s) // 200)
    idx = sorted(set(range(0, len(rec_s), step)) | {len(rec_s) - 1})
    pts = [{"recall": round(float(rec_s[i]), 5), "precision": round(float(prec_s[i]), 5)} for i in idx]

    return {
        "title": f"Precision-Recall Curve & Iso-F1 Contours — {model_eval.model_name}",
        "subtitle": f"Class '{pos_cls}' (Prevalence = {prevalence:.3f}) · PR-AUC (AP) = {ap_val:.4f} · Peak F1 = {opt_f1:.4f}",
        "kpis": {
            "PR-AUC (Avg Precision)": round(ap_val, 4),
            "Peak F1": round(opt_f1, 4),
            "Class Prevalence": round(prevalence, 4),
            "Lift vs Random": round(ap_val / max(prevalence, 1e-6), 2),
            "Optimal Threshold": round(opt_thr, 4),
        },
        "data": {
            "points": pts,
            "ap": round(ap_val, 4),
            "prevalence": round(prevalence, 4),
            "optimal_point": {"recall": round(opt_rec, 4), "precision": round(opt_prec, 4), "f1": round(opt_f1, 4)},
            "positive_class": pos_cls,
        },
        "table_columns": ["Recall", "Precision"],
        "table_rows": [[p["recall"], p["precision"]] for p in pts],
    }


def _build_classifier_calibration(model_eval: _ResolvedModelEval) -> dict[str, Any]:
    y_bin, scores, pos_cls = _extract_binary_clf_scores(model_eval, require_probabilities=True)
    if len(np.unique(y_bin)) < 2:
        raise LiveChartUnavailable("Calibration reliability diagram requires both positive and negative outcomes.")

    scores_c = np.clip(scores, 0.0, 1.0)
    brier = float(brier_score_loss(y_bin, scores_c))
    n_bins = min(10, max(4, len(y_bin) // 8))
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins_out: list[dict[str, Any]] = []
    ece = 0.0
    n_tot = len(y_bin)
    for b in range(n_bins):
        lo_e, hi_e = float(edges[b]), float(edges[b + 1])
        mask = (scores_c >= lo_e) & (scores_c <= hi_e if b == n_bins - 1 else scores_c < hi_e)
        cnt = int(np.sum(mask))
        if cnt == 0:
            bins_out.append(
                {
                    "bin_center": round(0.5 * (lo_e + hi_e), 3),
                    "mean_pred": None,
                    "frac_pos": None,
                    "ci_low": None,
                    "ci_high": None,
                    "count": 0,
                }
            )
            continue
        mp = float(np.mean(scores_c[mask]))
        fp = float(np.mean(y_bin[mask]))
        ece += (cnt / n_tot) * abs(fp - mp)
        # Wilson score 95% binomial CI
        z = 1.96
        denom = 1.0 + (z**2) / cnt
        center = (fp + (z**2) / (2.0 * cnt)) / denom
        half = (z * math.sqrt((fp * (1.0 - fp) + (z**2) / (4.0 * cnt)) / cnt)) / denom
        bins_out.append(
            {
                "bin_center": round(0.5 * (lo_e + hi_e), 3),
                "mean_pred": round(mp, 4),
                "frac_pos": round(fp, 4),
                "ci_low": round(max(0.0, center - half), 4),
                "ci_high": round(min(1.0, center + half), 4),
                "count": cnt,
            }
        )

    non_empty = [b for b in bins_out if b["count"] > 0]
    return {
        "title": f"Probability Calibration (Reliability) Diagram — {model_eval.model_name}",
        "subtitle": f"Class '{pos_cls}' · Brier Score = {brier:.4f} · Expected Calibration Error (ECE) = {ece:.4f}",
        "kpis": {
            "Brier Score": round(brier, 4),
            "ECE": round(ece, 4),
            "Non-Empty Bins": len(non_empty),
            "Eval Samples": n_tot,
        },
        "data": {"bins": bins_out, "brier": round(brier, 4), "ece": round(ece, 4), "positive_class": pos_cls},
        "table_columns": ["Bin_Midpoint", "Sample_Count", "Mean_Predicted_Prob", "Empirical_Positive_Rate", "Wilson_CI95_Low", "Wilson_CI95_High"],
        "table_rows": [[b["bin_center"], b["count"], b["mean_pred"], b["frac_pos"], b["ci_low"], b["ci_high"]] for b in bins_out],
    }


def _build_classifier_confusion_matrix(model_eval: _ResolvedModelEval) -> dict[str, Any]:
    y_t = model_eval.y_true.astype(str).to_numpy()
    y_p = pd.Series(model_eval.y_pred).astype(str).to_numpy()
    labels = sorted(set(y_t) | set(y_p))[:10]
    cm = confusion_matrix(y_t, y_p, labels=labels)
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_pct = (cm / row_sums) * 100.0

    acc = float(accuracy_score(y_t, y_p))
    f1_m = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(y_t, y_p))

    table_rows: list[list[Any]] = []
    for r_i, lbl_r in enumerate(labels):
        for c_i, lbl_c in enumerate(labels):
            table_rows.append([lbl_r, lbl_c, int(cm[r_i, c_i]), round(float(cm_pct[r_i, c_i]), 2)])

    return {
        "title": f"Confusion Matrix & Class Recall Heatmap — {model_eval.model_name}",
        "subtitle": f"{model_eval.eval_scope} · Accuracy = {acc * 100:.2f}%, Macro-F1 = {f1_m:.4f}, MCC = {mcc:.4f}",
        "kpis": {
            "Accuracy (%)": round(acc * 100.0, 2),
            "Macro-F1": round(f1_m, 4),
            "Matthews Corr (MCC)": round(mcc, 4),
            "Classes": len(labels),
        },
        "data": {
            "labels": labels,
            "counts": cm.tolist(),
            "row_pcts": np.round(cm_pct, 2).tolist(),
        },
        "table_columns": ["Actual_Class", "Predicted_Class", "Count", "Row_Recall_Pct"],
        "table_rows": table_rows,
    }


def _build_classifier_gain_lift(model_eval: _ResolvedModelEval) -> dict[str, Any]:
    y_bin, scores, pos_cls = _extract_binary_clf_scores(model_eval)
    total_pos = int(np.sum(y_bin))
    n = len(y_bin)
    if total_pos <= 0 or n < 4:
        raise LiveChartUnavailable("Cumulative Gains & Lift curve requires at least 1 positive instance in the evaluation set.")

    order = np.argsort(-scores)
    y_sorted = y_bin[order]
    cum_pos = np.cumsum(y_sorted) / total_pos

    grid_pcts = np.linspace(0.0, 1.0, 21)
    curve_pts: list[dict[str, float]] = []
    prevalence = total_pos / n
    for gp in grid_pcts:
        if gp <= 0.0:
            curve_pts.append({"pop_pct": 0.0, "gain_pct": 0.0, "wizard_pct": 0.0, "lift": 1.0})
        else:
            k_idx = min(n - 1, max(0, int(round(gp * n)) - 1))
            gain = float(cum_pos[k_idx])
            wiz = min(1.0, gp / max(prevalence, 1e-9))
            lift = gain / gp
            curve_pts.append(
                {
                    "pop_pct": round(gp * 100.0, 1),
                    "gain_pct": round(gain * 100.0, 2),
                    "wizard_pct": round(wiz * 100.0, 2),
                    "lift": round(lift, 3),
                }
            )

    lift_10 = next((p["lift"] for p in curve_pts if abs(p["pop_pct"] - 10.0) < 0.1), curve_pts[1]["lift"])
    lift_20 = next((p["lift"] for p in curve_pts if abs(p["pop_pct"] - 20.0) < 0.1), curve_pts[2]["lift"])
    gain_20 = next((p["gain_pct"] for p in curve_pts if abs(p["pop_pct"] - 20.0) < 0.1), 20.0)

    return {
        "title": f"Cumulative Gains & Decile Lift Curve — {model_eval.model_name}",
        "subtitle": f"Top 20% highest-scored records capture {gain_20:.1f}% of positives (Top-10% Lift = {lift_10:.2f}×, Top-20% Lift = {lift_20:.2f}×)",
        "kpis": {
            "Top-10% Lift": round(lift_10, 3),
            "Top-20% Lift": round(lift_20, 3),
            "Gain at Top-20% (%)": round(gain_20, 2),
            "Positive Rate (%)": round(prevalence * 100.0, 2),
        },
        "data": {"points": curve_pts, "positive_class": pos_cls, "prevalence_pct": round(prevalence * 100.0, 2)},
        "table_columns": ["Population_Targeted_Pct", "Cumulative_Gain_Pct", "Wizard_Optimal_Pct", "Cumulative_Lift"],
        "table_rows": [[p["pop_pct"], p["gain_pct"], p["wizard_pct"], p["lift"]] for p in curve_pts],
    }


def _extract_actual_and_pred_numeric(state: dict[str, Any], model_eval: _ResolvedModelEval | None) -> tuple[np.ndarray, np.ndarray, str]:
    """Extract numeric ``(actual, predicted, label)`` from supervisory evaluation or active model."""
    sup = state.get("uap_supervisory_result")
    if (
        isinstance(sup, dict) and sup.get("trend_series")
        and (sup.get("kpis") or {}).get("task_mode") == "regression"
    ):
        acts: list[float] = []
        preds: list[float] = []
        for pt in sup["trend_series"]:
            if isinstance(pt, dict):
                a = pt.get("actual")
                p = pt.get("model_pred", pt.get("predicted"))
                if a is not None and p is not None and math.isfinite(_finite_float(a, math.nan)) and math.isfinite(_finite_float(p, math.nan)):
                    acts.append(float(a))
                    preds.append(float(p))
        if len(acts) >= 4:
            lbl = str((sup.get("kpis") or {}).get("model_label") or "Supervisory Model")
            return np.asarray(acts, dtype=float), np.asarray(preds, dtype=float), lbl

    if model_eval is None:
        raise LiveChartUnavailable("Train a model or run Studio 14 Supervisory Evaluation first.")

    if model_eval.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        y_bin, scores, pos_cls = _extract_binary_clf_scores(model_eval, require_probabilities=True)
        return y_bin.astype(float), scores.astype(float), f"{model_eval.model_name} (P[{pos_cls}])"

    y_a = pd.to_numeric(model_eval.y_true, errors="coerce").to_numpy(dtype=float)
    y_p = np.asarray(model_eval.y_pred, dtype=float)
    mask = np.isfinite(y_a) & np.isfinite(y_p)
    if int(np.sum(mask)) < 4:
        raise LiveChartUnavailable("At least 4 finite actual/predicted pairs are required.")
    return y_a[mask], y_p[mask], model_eval.model_name


def _build_model_residual_diagnostics(state: dict[str, Any], model_eval: _ResolvedModelEval | None) -> dict[str, Any]:
    y_act, y_pred, lbl = _extract_actual_and_pred_numeric(state, model_eval)
    resid = y_act - y_pred
    sd_r = float(np.std(resid)) or 1e-6
    rmse = float(root_mean_squared_error(y_act, y_pred))
    mae = float(mean_absolute_error(y_act, y_pred))
    r2 = float(r2_score(y_act, y_pred)) if float(np.std(y_act)) > 1e-12 else 0.0
    # Durbin-Watson autocorrelation statistic on residuals
    dw = float(np.sum(np.diff(resid) ** 2) / max(float(np.sum(resid**2)), 1e-12)) if len(resid) >= 3 else 2.0

    n = len(resid)
    idx = np.linspace(0, n - 1, 420).round().astype(int) if n > 420 else np.arange(n)
    pts = [
        {
            "fitted": round(float(y_pred[i]), 5),
            "residual": round(float(resid[i]), 5),
            "z_resid": round(float(resid[i] / sd_r), 3),
        }
        for i in idx
    ]

    # Binned local mean bias curve across fitted range
    order = np.argsort(y_pred)
    n_bins = min(12, max(4, n // 8))
    chunks = np.array_split(order, n_bins)
    trend = [
        {"fitted": round(float(np.mean(y_pred[ch])), 5), "mean_resid": round(float(np.mean(resid[ch])), 5)}
        for ch in chunks
        if len(ch) > 0
    ]
    out_2sd_pct = round(100.0 * float(np.mean(np.abs(resid) > 2.0 * sd_r)), 2)

    return {
        "title": f"Residuals vs. Fitted Diagnostic Plot — {lbl}",
        "subtitle": f"RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f} · Durbin-Watson = {dw:.3f} · >2σ = {out_2sd_pct:.1f}%",
        "kpis": {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R²": round(r2, 4),
            "Residual SD (σ)": round(sd_r, 4),
            "Durbin-Watson": round(dw, 3),
            "Outside ±2σ (%)": out_2sd_pct,
        },
        "data": {"points": pts, "trend": trend, "sigma": round(sd_r, 5)},
        "table_columns": ["Fitted_Prediction", "Residual_Actual_Minus_Pred", "Standardized_Z_Residual"],
        "table_rows": [[p["fitted"], p["residual"], p["z_resid"]] for p in pts],
    }


def _build_bland_altman_agreement(state: dict[str, Any], model_eval: _ResolvedModelEval | None) -> dict[str, Any]:
    y_act, y_pred, lbl = _extract_actual_and_pred_numeric(state, model_eval)
    means = 0.5 * (y_act + y_pred)
    diffs = y_act - y_pred
    bias = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    loa_upper = bias + 1.96 * sd_diff
    loa_lower = bias - 1.96 * sd_diff
    outside_pct = round(100.0 * float(np.mean((diffs > loa_upper) | (diffs < loa_lower))), 2)

    n = len(diffs)
    idx = np.linspace(0, n - 1, 420).round().astype(int) if n > 420 else np.arange(n)
    pts = [
        {
            "mean": round(float(means[i]), 5),
            "diff": round(float(diffs[i]), 5),
            "outside": bool(diffs[i] > loa_upper or diffs[i] < loa_lower),
        }
        for i in idx
    ]

    return {
        "title": f"Bland-Altman Method Agreement Plot — {lbl}",
        "subtitle": f"Mean Bias = {bias:+.4f} · 95% Limits of Agreement [{loa_lower:.4f}, {loa_upper:.4f}] ({outside_pct:.1f}% outside LoA)",
        "kpis": {
            "Mean Bias (d̄)": round(bias, 4),
            "Upper 95% LoA": round(loa_upper, 4),
            "Lower 95% LoA": round(loa_lower, 4),
            "Outside LoA (%)": outside_pct,
            "Sample Pairs": n,
        },
        "data": {
            "points": pts,
            "bias": round(bias, 5),
            "loa_upper": round(loa_upper, 5),
            "loa_lower": round(loa_lower, 5),
        },
        "table_columns": ["Pair_Mean", "Difference_Actual_Minus_Pred", "Outside_95_LoA"],
        "table_rows": [[p["mean"], p["diff"], p["outside"]] for p in pts],
    }


def _build_cusum_sequential_control(
    state: dict[str, Any],
    num_df: pd.DataFrame,
    col_x: str,
    disp_x: str,
    studio_id: str,
) -> dict[str, Any]:
    if studio_id == "14":
        sup = state.get("uap_supervisory_result") or {}
        limits = sup.get("spc_limits") or {}
        mu0 = _finite_float(limits.get("center"), math.nan)
        sigma0 = _finite_float(limits.get("sigma"), math.nan)
        if not math.isfinite(mu0) or not math.isfinite(sigma0) or sigma0 <= 0:
            raise LiveChartUnavailable("Supervisory CUSUM requires a verified held-out residual reference.")
        residual_sequence = state.get("uap_supervisory_residual_sequence")
        if isinstance(residual_sequence, np.ndarray) and residual_sequence.ndim == 1:
            arr = residual_sequence
        else:
            points = sup.get("spc_points") or []
            arr = np.asarray([_finite_float(p.get("residual"), math.nan) for p in points], dtype=float)
        if len(arr) != (sup.get("kpis") or {}).get("n_records"):
            raise LiveChartUnavailable("Supervisory CUSUM requires the full, ordered residual sequence.")
        if not np.isfinite(arr).all():
            raise LiveChartUnavailable("Supervisory CUSUM requires a complete sequential residual series.")
        src_name = "Studio 14 residuals · held-out reference"
    else:
        arr = num_df[col_x].dropna().to_numpy(dtype=float)
        src_name = f"{disp_x} · retrospective reference"
        if len(arr) < 6:
            raise LiveChartUnavailable("Retrospective CUSUM needs at least five reference rows and one subsequent row.")
        ref_n = max(5, len(arr) // 2)
        mu0 = float(np.mean(arr[:ref_n]))
        sigma0 = float(np.std(arr[:ref_n], ddof=1))
        arr = arr[ref_n:]
        if not math.isfinite(sigma0) or sigma0 <= 0:
            raise LiveChartUnavailable("The retrospective reference has no estimable variation.")

    if not len(arr):
        raise LiveChartUnavailable("No sequential observations are available for CUSUM.")

    z = (arr - mu0) / sigma0
    k_slack = 0.5
    h_limit = 4.0
    c_plus = np.zeros(len(z), dtype=float)
    c_minus = np.zeros(len(z), dtype=float)
    for i in range(len(z)):
        prev_p = c_plus[i - 1] if i > 0 else 0.0
        prev_m = c_minus[i - 1] if i > 0 else 0.0
        c_plus[i] = max(0.0, prev_p + z[i] - k_slack)
        c_minus[i] = min(0.0, prev_m + z[i] + k_slack)

    alarm_mask = (c_plus > h_limit) | (np.abs(c_minus) > h_limit)
    n_alarms = int(np.sum(alarm_mask))
    first_alarm = int(np.argmax(alarm_mask)) + 1 if n_alarms > 0 else None

    step = max(1, len(z) // 320)
    idx = sorted(set(range(0, len(z), step)) | set(np.where(alarm_mask)[0].tolist()) | {len(z) - 1})[:420]
    pts = [
        {
            "step": int(i + 1),
            "c_plus": round(float(c_plus[i]), 4),
            "c_minus": round(float(c_minus[i]), 4),
            "alarm": bool(alarm_mask[i]),
        }
        for i in idx
    ]

    return {
        "title": f"Tabular CUSUM (C+ / C-) Drift Control Chart — {src_name}",
        "subtitle": f"Slack k = {k_slack}σ, Decision Interval H = ±{h_limit}σ · {n_alarms} alarm points"
        + (f" (First threshold crossing at t = {first_alarm})" if first_alarm else " (No threshold crossings)"),
        "kpis": {
            "Baseline Mean μ₀": round(mu0, 4),
            "Baseline SD σ₀": round(sigma0, 4),
            "Decision Limit H (σ)": h_limit,
            "Alarm Points": n_alarms,
            "First Change-Point": first_alarm if first_alarm is not None else "None",
        },
        "data": {"points": pts, "h_limit": h_limit, "source_name": src_name},
        "table_columns": ["Observation_Index", "CUSUM_Upper_C_Plus", "CUSUM_Lower_C_Minus", "Exceeds_H_Limit"],
        "table_rows": [[p["step"], p["c_plus"], p["c_minus"], p["alarm"]] for p in pts],
    }


def _build_lorenz_gini_curve(ser: pd.Series, disp_x: str) -> dict[str, Any]:
    arr = pd.to_numeric(ser, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(arr) < 3:
        raise LiveChartUnavailable(f"Lorenz Curve requires at least 3 finite observations in '{disp_x}'.")

    min_v = float(np.min(arr))
    shifted = arr - min_v if min_v < 0 else arr.copy()
    note = f" (shifted by {-min_v:+.4g} for non-negativity)" if min_v < 0 else ""
    s = np.sort(shifted)
    total = float(np.sum(s))
    n = len(s)
    if total <= 1e-12:
        pop = np.linspace(0.0, 1.0, 21)
        lor = pop.copy()
        gini = 0.0
        top20 = 20.0
    else:
        cum_w = np.concatenate([[0.0], np.cumsum(s) / total])
        cum_p = np.linspace(0.0, 1.0, n + 1)
        area = float(np.trapezoid(cum_w, cum_p))
        gini = max(0.0, min(1.0, 1.0 - 2.0 * area))
        cut80 = int(round(0.8 * n))
        top20 = 100.0 * float(np.sum(s[cut80:])) / total
        idx = np.unique(np.linspace(0, n, min(120, n + 1)).round().astype(int))
        pop = cum_p[idx]
        lor = cum_w[idx]

    pts = [{"pop_share": round(float(p), 5), "outcome_share": round(float(w), 5)} for p, w in zip(pop, lor, strict=True)]
    return {
        "title": f"Lorenz Concentration Curve & Gini Index — {disp_x}",
        "subtitle": f"Gini Coefficient = {gini:.4f} · Top 20% holds {top20:.1f}% of cumulative magnitude{note}",
        "kpis": {
            "Gini Coefficient": round(gini, 4),
            "Top-20% Share (%)": round(top20, 2),
            "Sample Size n": n,
        },
        "data": {"points": pts, "gini": round(gini, 4), "top20_pct": round(top20, 2), "variable_name": disp_x},
        "table_columns": ["Cumulative_Population_Share", "Cumulative_Outcome_Share", "Equality_Diagonal"],
        "table_rows": [[p["pop_share"], p["outcome_share"], p["pop_share"]] for p in pts],
    }


def _build_multivariate_parallel_coords(
    df: pd.DataFrame,
    num_df: pd.DataFrame,
    group_col: str | None,
    primary_target: str | None,
) -> dict[str, Any]:
    cols = list(num_df.columns[:7])
    if len(cols) < 3:
        raise LiveChartUnavailable("Parallel Coordinates requires at least 3 numeric/encodable columns.")

    g_col = (
        group_col
        if (group_col and group_col in df.columns)
        else ("cluster_id" if "cluster_id" in df.columns else primary_target if (primary_target and primary_target in df.columns) else None)
    )
    work = num_df[cols].copy()
    work["_grp"] = _discretize_series(df[g_col], max_bins=4).to_numpy() if g_col else "All"
    work = work.dropna(subset=cols)
    if len(work) < 3:
        work = num_df[cols].fillna(num_df[cols].median())
        work["_grp"] = _discretize_series(df[g_col], max_bins=4).to_numpy() if g_col else "All"

    mins = work[cols].min(axis=0)
    maxs = work[cols].max(axis=0)
    spans = (maxs - mins).replace(0.0, 1.0)
    norm_df = (work[cols] - mins) / spans
    norm_df["_grp"] = work["_grp"].to_numpy()

    axes_meta = [
        {"name": str(c), "min": round(float(mins[c]), 4), "max": round(float(maxs[c]), 4)}
        for c in cols
    ]
    top_groups = list(pd.Series(norm_df["_grp"]).value_counts().head(4).index)
    color_map = {str(g): PALETTE[i % len(PALETTE)] for i, g in enumerate(top_groups)}

    # Group median profiles
    medians = [
        {
            "group": str(g),
            "color": color_map[str(g)],
            "values": [round(float(norm_df.loc[norm_df["_grp"] == g, c].median()), 4) for c in cols],
        }
        for g in top_groups
    ]

    # Sample up to 110 real individual trajectories
    n = len(norm_df)
    idx = np.linspace(0, n - 1, min(110, n)).round().astype(int)
    sub_norm = norm_df.iloc[idx]
    sub_raw = work.iloc[idx]
    lines = [
        {
            "group": str(sub_norm["_grp"].iloc[k]),
            "color": color_map.get(str(sub_norm["_grp"].iloc[k]), PALETTE[0]),
            "values": [round(float(sub_norm[c].iloc[k]), 4) for c in cols],
        }
        for k in range(len(sub_norm))
    ]
    table_rows = [
        [str(sub_raw["_grp"].iloc[k]), *(round(float(sub_raw[c].iloc[k]), 4) for c in cols)]
        for k in range(len(sub_raw))
    ]

    return {
        "title": "Multivariate Parallel Coordinates Profile Plot",
        "subtitle": f"{len(cols)} normalized variables · {len(lines)} real trajectories + {len(medians)} bold group-median profiles",
        "kpis": {"Axes": len(cols), "Trajectories": len(lines), "Strata": len(medians), "Colored By": g_col or "Uniform"},
        "data": {"axes": axes_meta, "lines": lines, "medians": medians},
        "table_columns": ["Stratum", *(str(c) for c in cols)],
        "table_rows": table_rows,
    }


def _build_ridgeline_kde_joyplot(
    df: pd.DataFrame,
    num_df: pd.DataFrame,
    col_x: str,
    disp_x: str,
    group_col: str | None,
    primary_target: str | None,
) -> dict[str, Any]:
    g_col = (
        group_col
        if (group_col and group_col in df.columns and group_col != col_x)
        else (primary_target if (primary_target and primary_target in df.columns and primary_target != col_x and df[primary_target].nunique(dropna=True) <= 7) else None)
    )

    raw_ridges: list[tuple[str, np.ndarray]] = []
    if g_col:
        g_ser = _discretize_series(df[g_col], max_bins=6)
        for g_val in g_ser.value_counts().head(6).index:
            arr = num_df.loc[g_ser == g_val, col_x].dropna().to_numpy(dtype=float)
            if len(arr) >= 3:
                raw_ridges.append((f"{g_val}", arr))
        x_lbl = disp_x
        title = f"Ridgeline KDE Density Plot — {disp_x} across {g_col}"
    else:
        for c in list(num_df.columns)[:7]:
            arr = num_df[c].dropna().to_numpy(dtype=float)
            if len(arr) >= 3:
                sd = float(np.std(arr))
                z = (arr - float(np.mean(arr))) / sd if sd > 1e-12 else arr - float(np.mean(arr))
                raw_ridges.append((str(c), z))
        x_lbl = "Standardized Z-Score ((x - μ) / σ)"
        title = "Multi-Feature Standardized Ridgeline KDE Plot"

    if not raw_ridges:
        raise LiveChartUnavailable("Ridgeline KDE requires at least 3 finite observations.")

    all_v = np.concatenate([a for _, a in raw_ridges])
    lo, hi = float(np.percentile(all_v, 1)), float(np.percentile(all_v, 99))
    if hi <= lo:
        lo, hi = float(np.min(all_v)) - 1.0, float(np.max(all_v)) + 1.0
    pad = 0.08 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 64)

    ridges_out: list[dict[str, Any]] = []
    table_rows: list[list[Any]] = []
    for idx, (lbl, arr) in enumerate(raw_ridges):
        dens = _kde_curve(arr, grid)
        med = round(float(np.median(arr)), 4)
        iqr = round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 4)
        ridges_out.append(
            {
                "label": lbl[:22],
                "n": int(len(arr)),
                "median": med,
                "iqr": iqr,
                "density": [round(float(d), 4) for d in dens],
                "color": PALETTE[idx % len(PALETTE)],
            }
        )
        table_rows.append([lbl, int(len(arr)), med, iqr, round(float(np.mean(arr)), 4), round(float(np.std(arr)), 4)])

    return {
        "title": title,
        "subtitle": f"Gaussian Kernel Density Estimation across {len(ridges_out)} distributions (shared domain [{grid[0]:.2f}, {grid[-1]:.2f}])",
        "kpis": {"Ridges": len(ridges_out), "Grid Points": len(grid), "Total Observations": int(len(all_v))},
        "data": {"x_grid": [round(float(v), 4) for v in grid], "ridges": ridges_out, "x_label": x_lbl},
        "table_columns": ["Distribution", "Sample_N", "Median", "IQR", "Mean", "Std_Dev"],
        "table_rows": table_rows,
    }


def _build_alluvial_sankey_flow(
    df: pd.DataFrame,
    feature_x: str | None,
    feature_y: str | None,
    primary_target: str | None,
) -> dict[str, Any]:
    cols = [str(c) for c in df.columns]
    if len(cols) < 2:
        raise LiveChartUnavailable("Alluvial / Sankey Flow requires at least 2 columns.")
    cat_cols = [str(c) for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    col_src = feature_x if (feature_x and feature_x in df.columns) else (cat_cols[0] if cat_cols else cols[0])
    col_dst = (
        feature_y
        if (feature_y and feature_y in df.columns and feature_y != col_src)
        else (primary_target if (primary_target and primary_target in df.columns and primary_target != col_src) else next((c for c in cols if c != col_src), cols[0]))
    )

    src_ser = _discretize_series(df[col_src], max_bins=5)
    dst_ser = _discretize_series(df[col_dst], max_bins=5)
    ctab = pd.crosstab(src_ser, dst_ser)

    src_nodes = [{"id": f"src::{s}", "label": str(s), "total": int(ctab.loc[s].sum())} for s in ctab.index]
    dst_nodes = [{"id": f"dst::{d}", "label": str(d), "total": int(ctab[d].sum())} for d in ctab.columns]
    links: list[dict[str, Any]] = []
    for s_idx, s in enumerate(ctab.index):
        for d in ctab.columns:
            cnt = int(ctab.loc[s, d])
            if cnt > 0:
                links.append(
                    {
                        "source": f"src::{s}",
                        "source_label": str(s),
                        "target": f"dst::{d}",
                        "target_label": str(d),
                        "value": cnt,
                        "color": PALETTE[s_idx % len(PALETTE)],
                    }
                )

    total_flow = int(ctab.to_numpy().sum()) or 1
    return {
        "title": f"Alluvial / Sankey Flow — {col_src} → {col_dst}",
        "subtitle": f"Mass-conserving flow of {total_flow:,} records across {len(src_nodes)} source strata × {len(dst_nodes)} destination strata",
        "kpis": {
            "Source Column": col_src,
            "Target Column": col_dst,
            "Total Flow (n)": total_flow,
            "Active Flow Ribbons": len(links),
        },
        "data": {
            "source_title": col_src,
            "target_title": col_dst,
            "sources": src_nodes,
            "targets": dst_nodes,
            "links": links,
            "total": total_flow,
        },
        "table_columns": [f"Source_{col_src}", f"Target_{col_dst}", "Record_Count", "Flow_Share_Pct"],
        "table_rows": [[lnk["source_label"], lnk["target_label"], lnk["value"], round(100.0 * lnk["value"] / total_flow, 2)] for lnk in links],
    }


def _build_data_lifecycle_funnel(state: dict[str, Any], raw_active: pd.DataFrame, df: pd.DataFrame) -> dict[str, Any]:
    raw_df = state.get("uap_raw_df")
    n_raw = int(len(raw_df)) if isinstance(raw_df, pd.DataFrame) and not raw_df.empty else int(len(raw_active))
    n_non_dup = int(len(raw_df.drop_duplicates())) if isinstance(raw_df, pd.DataFrame) and not raw_df.empty else int(len(raw_active.drop_duplicates()))
    n_active = int(len(df))
    n_complete = int(df.dropna().shape[0])
    lock = state.get("uap_split_lock")
    n_train = int(lock.get("n_train") or round(0.8 * n_active)) if isinstance(lock, dict) and lock.get("active") else int(round(0.8 * n_active))
    train_lbl = "Locked Train Split" if (isinstance(lock, dict) and lock.get("active")) else "80% Modeling Partition"

    raw_stages = [
        ("1. Raw Ingested Records", n_raw),
        ("2. Deduplicated Unique Rows", n_non_dup),
        ("3. Active Post-Cleaning Rows", n_active),
        ("4. Complete Zero-Missing Rows", n_complete),
        (f"5. {train_lbl}", n_train),
    ]
    top_n = max(1, n_raw)
    stages_out: list[dict[str, Any]] = []
    prev_v = top_n
    for idx, (lbl, cnt) in enumerate(raw_stages):
        pct_of_top = round(100.0 * cnt / top_n, 2)
        step_pct = round(100.0 * cnt / max(1, prev_v), 2) if idx > 0 else 100.0
        stages_out.append(
            {
                "label": lbl,
                "value": cnt,
                "pct_of_initial": pct_of_top,
                "step_conversion_pct": step_pct,
                "color": PALETTE[idx % len(PALETTE)],
            }
        )
        prev_v = max(1, cnt)

    return {
        "title": "CRISP-DM Data & Split Retention Funnel",
        "subtitle": f"{n_raw:,} initial rows → {n_active:,} active rows → {n_train:,} training rows ({stages_out[-1]['pct_of_initial']:.1f}% net yield)",
        "kpis": {
            "Raw Ingested": n_raw,
            "Active Study Rows": n_active,
            "Complete Rows": n_complete,
            "Training Rows": n_train,
        },
        "data": {"stages": stages_out},
        "table_columns": ["Funnel_Stage", "Row_Count", "Pct_Of_Initial", "Step_Retention_Pct"],
        "table_rows": [[s["label"], s["value"], s["pct_of_initial"], s["step_conversion_pct"]] for s in stages_out],
    }


def _build_correlation_heatmap_svg(num_df: pd.DataFrame) -> dict[str, Any]:
    cols = [c for c in num_df.columns if num_df[c].nunique(dropna=True) > 1][:12]
    if len(cols) < 2:
        raise LiveChartUnavailable("Correlation Heatmap requires at least 2 non-constant columns.")
    corr = num_df[cols].corr(method="pearson")
    mat_vals: list[list[float | None]] = []
    table_rows: list[list[Any]] = []
    max_off = 0.0
    max_pair = ""
    for r_i, rc in enumerate(cols):
        row_out: list[float | None] = []
        for c_i, cc in enumerate(cols):
            v = corr.iloc[r_i, c_i]
            fv = round(float(v), 4) if pd.notna(v) and math.isfinite(float(v)) else None
            row_out.append(fv)
            if r_i < c_i and fv is not None:
                table_rows.append([str(rc), str(cc), fv])
                if abs(fv) > abs(max_off):
                    max_off = fv
                    max_pair = f"{rc} × {cc}"
        mat_vals.append(row_out)

    table_rows.sort(key=lambda r: abs(r[2]), reverse=True)
    return {
        "title": "Vector Correlation & Association Matrix Heatmap",
        "subtitle": f"{len(cols)} × {len(cols)} pairwise Pearson correlation matrix · Peak off-diagonal r = {max_off:+.3f} ({max_pair or 'N/A'})",
        "kpis": {"Matrix Dimension": f"{len(cols)}×{len(cols)}", "Peak Pairwise r": round(max_off, 4), "Peak Pair": max_pair or "N/A"},
        "data": {"columns": [str(c) for c in cols], "values": mat_vals},
        "table_columns": ["Feature_A", "Feature_B", "Pearson_r"],
        "table_rows": table_rows,
    }


def _build_stacked_composition_bar(
    df: pd.DataFrame,
    feature_x: str | None,
    group_col: str | None,
    primary_target: str | None,
) -> dict[str, Any]:
    cols = [str(c) for c in df.columns]
    if len(cols) < 2:
        raise LiveChartUnavailable("Stacked Composition Bar Chart requires at least 2 columns.")
    cat_cols = [str(c) for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    col_x = feature_x if (feature_x and feature_x in df.columns) else (cat_cols[0] if cat_cols else cols[0])
    col_g = (
        group_col
        if (group_col and group_col in df.columns and group_col != col_x)
        else (primary_target if (primary_target and primary_target in df.columns and primary_target != col_x) else next((c for c in cols if c != col_x), cols[0]))
    )

    x_ser = _discretize_series(df[col_x], max_bins=7)
    g_ser = _discretize_series(df[col_g], max_bins=5)
    ctab = pd.crosstab(x_ser, g_ser)

    # Chi-square & Cramér's V association
    cramers_v = 0.0
    chi2_p = 1.0
    if ctab.shape[0] >= 2 and ctab.shape[1] >= 2:
        try:
            chi2, p_v, _dof, _exp = stats.chi2_contingency(ctab.to_numpy())
            n_tot = int(ctab.to_numpy().sum())
            min_dim = min(ctab.shape[0] - 1, ctab.shape[1] - 1)
            cramers_v = math.sqrt(float(chi2) / max(1, n_tot * min_dim))
            chi2_p = float(p_v)
        except Exception:
            pass

    seg_names = [str(c) for c in ctab.columns]
    bars_out: list[dict[str, Any]] = []
    table_rows: list[list[Any]] = []
    for r_lbl in ctab.index:
        row_tot = int(ctab.loc[r_lbl].sum()) or 1
        segs = []
        for s_idx, s_name in enumerate(seg_names):
            cnt = int(ctab.loc[r_lbl, s_name])
            pct = round(100.0 * cnt / row_tot, 2)
            segs.append({"name": s_name, "count": cnt, "pct": pct, "color": PALETTE[s_idx % len(PALETTE)]})
            table_rows.append([str(r_lbl), s_name, cnt, pct])
        bars_out.append({"label": str(r_lbl), "total": row_tot, "segments": segs})

    return {
        "title": f"100% Stacked Outcome Composition — {col_x} × {col_g}",
        "subtitle": f"Cramér's V = {cramers_v:.4f} (χ² p = {chi2_p:.4g}) across {len(bars_out)} strata × {len(seg_names)} outcome segments",
        "kpis": {
            "Primary Axis": col_x,
            "Segment By": col_g,
            "Cramér's V": round(cramers_v, 4),
            "Chi-Square p-value": round(chi2_p, 6),
        },
        "data": {
            "x_label": col_x,
            "group_label": col_g,
            "segment_names": seg_names,
            "bars": bars_out,
        },
        "table_columns": [col_x, col_g, "Count", "Row_Share_Pct"],
        "table_rows": table_rows,
    }
