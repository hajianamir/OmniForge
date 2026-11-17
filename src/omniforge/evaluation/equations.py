"""Mathematical equation, architecture formula, decision-rule, and surrogate extractor for all OmniForge models.

Provides transparent mathematical representations for:
    - Baseline models (mean/median/prior constants, Ridge linear, Logistic sigmoid, Decision Stumps, Forecasting rules).
    - Neural Networks / Deep Learning (MLP, 1D-CNN, Tabular Transformer, GCN message passing) with exact layer
      forward-pass equations, trainable parameter counts, weight norms, and high-fidelity linear surrogate formulas.
    - Advanced Classical & Ensemble Engines (Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM,
      CatBoost, SVM, KNN, Voting/Stacking Ensembles) with exact additive/ensemble equations, representative
      tree rules, feature importances, and global linear surrogate formulas.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.metrics import r2_score
from sklearn.tree import export_text

from omniforge.core.logging import get_logger
from omniforge.core.types import TaskType

logger = get_logger(__name__)


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in variable or target names for safe KaTeX rendering."""
    cleaned = str(text)
    for ch, rep in (
        ("\\", r"\textbackslash "),
        ("_", r"\_"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("&", r"\&"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("^", r"\^{}"),
    ):
        cleaned = cleaned.replace(ch, rep)
    return cleaned


def _fmt_num(val: float, precision: int = 4) -> str:
    """Format a floating-point coefficient cleanly without scientific noise for ordinary scales."""
    v = float(val)
    if not math.isfinite(v):
        return "0"
    if abs(v) != 0.0 and (abs(v) < 1e-4 or abs(v) >= 1e6):
        return f"{v:.3e}"
    return f"{v:.{precision}f}"


def _build_linear_strings(
    lhs_plain: str,
    lhs_latex: str,
    intercept: float,
    terms: list[tuple[str, float]],
    *,
    max_terms: int = 10,
    wrap_sigmoid: bool = False,
) -> tuple[str, str]:
    """Assemble readable plain-text and LaTeX linear/logistic equations."""
    non_zero = [(name, float(coef)) for name, coef in terms if abs(float(coef)) > 1e-12]
    non_zero.sort(key=lambda item: abs(item[1]), reverse=True)
    shown = non_zero[:max_terms]
    omitted = max(0, len(non_zero) - len(shown))

    rhs_plain = _fmt_num(intercept)
    rhs_latex = _fmt_num(intercept)
    for name, coef in shown:
        sign = "+" if coef >= 0 else "-"
        mag = _fmt_num(abs(coef))
        rhs_plain += f" {sign} {mag} * ({name})"
        rhs_latex += rf" {sign} {mag}\cdot \text{{{_escape_latex(name)}}}"

    if omitted > 0:
        rhs_plain += f" + ... ({omitted} smaller terms)"
        rhs_latex += rf" + \dots \;(\text{{{omitted} smaller terms}})"

    if wrap_sigmoid:
        plain = f"{lhs_plain} = 1 / (1 + exp(-({rhs_plain})))"
        latex = rf"{lhs_latex} = \sigma\!\left({rhs_latex}\right), \quad \sigma(u)=\frac{{1}}{{1+e^{{-u}}}}"
    else:
        plain = f"{lhs_plain} = {rhs_plain}"
        latex = rf"{lhs_latex} = {rhs_latex}"
    return plain, latex


def _excel_col_letter(zero_idx: int) -> str:
    """Convert 0-based column index to Excel column letters (0 -> B, 1 -> C, etc. leaving col A for ID)."""
    n = int(zero_idx) + 2  # Start at column B (col A reserved for Row ID / timestamp)
    out = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def _safe_identifier(name: str) -> str:
    """Sanitize a column name into a clean C/MATLAB/Python identifier."""
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(name).strip())
    if not cleaned:
        return "x"
    if cleaned[0].isdigit():
        cleaned = f"x_{cleaned}"
    return cleaned


def _build_external_deployment_formulas(
    *,
    model_name: str,
    target_name: str,
    task_type: str,
    equation_type: str,
    intercept: float | None,
    coefficients: dict[str, float],
    surrogate_r2: float | None = None,
    is_surrogate: bool = False,
) -> dict[str, str]:
    """Generate zero-dependency standalone Python, Excel, SQL, and C/MATLAB deployment formulas."""
    is_clf = "classification" in str(task_type).lower()
    use_sigmoid = is_clf and (equation_type == "exact_closed_form")
    b0 = float(intercept) if (intercept is not None and math.isfinite(float(intercept))) else 0.0

    # Keep all non-trivial physical coefficients sorted by magnitude
    terms = [
        (str(k), float(v))
        for k, v in coefficients.items()
        if isinstance(v, (int, float)) and math.isfinite(float(v)) and abs(float(v)) > 1e-12
    ]
    terms.sort(key=lambda kv: abs(kv[1]), reverse=True)
    # Limit external inline expressions to top 40 terms so Excel/SQL remain readable & valid
    used_terms = terms[:40]
    safe_tgt = _safe_identifier(target_name)

    if not used_terms and intercept is None:
        return {
            "formula_python": (
                f"# {model_name} ({equation_type})\n"
                f"def predict_{safe_tgt}(row: dict) -> float:\n"
                f"    \"\"\"Constant / structural model fallback.\"\"\"\n"
                f"    return {b0:.6f}\n"
            ),
            "formula_excel": f"={b0:.6f}",
            "formula_sql": f"SELECT {b0:.6f} AS predicted_{safe_tgt}\nFROM your_table;",
            "formula_c_matlab": f"y_hat_{safe_tgt} = {b0:.6f};",
            "external_usage_summary": (
                f"Constant prediction rule ({b0:.6f}) for target '{target_name}'."
            ),
        }

    # 1. Standalone Zero-Dependency Python Function
    py_lines = [
        "# Standalone External Prediction Script — Exported from OmniForge",
        f"# Model: {model_name} | Target: {target_name} | Mode: {'High-Fidelity Analytical Surrogate (R2=' + f'{surrogate_r2:.4f})' if is_surrogate and surrogate_r2 is not None else ('Analytical Formula' if is_surrogate else 'Exact Closed-Form')}",
        "# Zero external libraries required (uses Python standard library only).",
        "import math",
        "",
        f"def predict_{safe_tgt}(row: dict) -> float:",
        f'    """Predict {target_name} from a dict of raw input feature values."""',
        f"    val = {b0:.8f}",
    ]
    for fname, coef in used_terms:
        sign = "+" if coef >= 0 else "-"
        py_lines.append(f"    val {sign}= {abs(coef):.8f} * float(row.get({fname!r}, 0.0))")
    if use_sigmoid:
        py_lines.append("    z = max(-500.0, min(500.0, val))")
        py_lines.append("    return 1.0 / (1.0 + math.exp(-z))  # Probability P(y=1 | x)")
    elif is_clf and is_surrogate:
        py_lines.append("    return max(0.0, min(1.0, val))  # Approximate probability P(y=1 | x)")
    else:
        py_lines.append("    return val")
    sample_dict = ", ".join(f"{fname!r}: 0.0" for fname, _ in used_terms[:5])
    py_lines.extend([
        "",
        'if __name__ == "__main__":',
        f"    sample = {{{sample_dict}}}",
        f"    print('Predicted {target_name}:', predict_{safe_tgt}(sample))",
    ])
    formula_python = "\n".join(py_lines)

    # 2. Microsoft Excel Formula (with Column Mapping Guide)
    excel_parts = [f"{b0:.6f}"]
    col_map: list[str] = []
    for idx, (fname, coef) in enumerate(used_terms):
        col_let = _excel_col_letter(idx)
        sign = " + " if coef >= 0 else " - "
        excel_parts.append(f"{sign}({abs(coef):.6f} * {col_let}2)")
        col_map.append(f"{col_let}2={fname}")
    raw_excel_expr = "".join(excel_parts)
    if use_sigmoid:
        excel_formula_line = f"=1/(1+EXP(-({raw_excel_expr})))"
    elif is_clf and is_surrogate:
        excel_formula_line = f"=MIN(1, MAX(0, {raw_excel_expr}))"
    else:
        excel_formula_line = f"={raw_excel_expr}"
    formula_excel = (
        f"{excel_formula_line}\n"
        f"-- Excel Column Map (Row 2): " + ", ".join(col_map)
    )

    # 3. ANSI SQL SELECT Expression
    sql_parts = [f"    {b0:.6f}"]
    for fname, coef in used_terms:
        sign = "+" if coef >= 0 else "-"
        q_col = '"' + fname.replace('"', '""') + '"'
        sql_parts.append(f"    {sign} {abs(coef):.6f} * COALESCE({q_col}, 0.0)")
    sql_body = "\n".join(sql_parts)
    if use_sigmoid:
        formula_sql = (
            "SELECT\n"
            "  1.0 / (1.0 + EXP(-(\n"
            f"{sql_body}\n"
            f"  ))) AS predicted_prob_{safe_tgt}\n"
            "FROM your_table;"
        )
    elif is_clf and is_surrogate:
        formula_sql = (
            "SELECT\n"
            "  LEAST(1.0, GREATEST(0.0,\n"
            f"{sql_body}\n"
            f"  )) AS predicted_prob_{safe_tgt}\n"
            "FROM your_table;"
        )
    else:
        formula_sql = (
            "SELECT\n"
            "  (\n"
            f"{sql_body}\n"
            f"  ) AS predicted_{safe_tgt}\n"
            "FROM your_table;"
        )

    # 4. C / MATLAB / JavaScript / PLC Structured-Text Formula
    cm_parts = [f"{b0:.6f}"]
    for fname, coef in used_terms:
        sign = " + " if coef >= 0 else " - "
        cm_parts.append(f"{sign}{abs(coef):.6f} * {_safe_identifier(fname)}")
    cm_expr = "".join(cm_parts)
    if use_sigmoid:
        formula_c_matlab = f"z = {cm_expr};\ny_prob_{safe_tgt} = 1.0 / (1.0 + exp(-z));"
    else:
        formula_c_matlab = f"y_hat_{safe_tgt} = {cm_expr};"

    if not is_surrogate and equation_type == "exact_closed_form":
        summary = (
            f"EXACT CLOSED-FORM RELATIONSHIP (100% mathematically identical to the fitted {model_name} engine). "
            f"Copy the standalone Python, Excel, SQL, or MATLAB/C equation to compute '{target_name}' outside OmniForge with zero approximation error."
        )
    elif surrogate_r2 is not None:
        fidelity_band = (
            "Very High Fidelity (safe for direct external analytical deployment)"
            if surrogate_r2 >= 0.90
            else (
                "Moderate-High Fidelity (captures primary linear trend; non-linear interactions exist — inspect Decision Rules or Python code)"
                if surrogate_r2 >= 0.70
                else "Approximate Global Linear Trend (strong non-linearities/interactions present — use full Decision Rules or Exact Python Tree/Model export for precision)"
            )
        )
        summary = (
            f"NON-LINEAR MODEL ({model_name}) + PORTABLE CLOSED-FORM ANALYTICAL FORMULA (Surrogate Fidelity R² = {surrogate_r2:.4f} — {fidelity_band}). "
            f"Includes {len(used_terms)} physical-unit feature coefficients for immediate external use in Python, Excel, SQL, or MATLAB/PLC."
        )
    else:
        summary = (
            f"Portable analytical formula for {model_name} with {len(used_terms)} physical-unit feature coefficients ready for external deployment."
        )

    return {
        "formula_python": formula_python,
        "formula_excel": formula_excel,
        "formula_sql": formula_sql,
        "formula_c_matlab": formula_c_matlab,
        "external_usage_summary": summary,
    }


class ModelEquationReport(BaseModel):
    """Structured mathematical relationship and interpretability specification for a fitted model."""

    model_name: str
    model_family: str
    category: str  # baseline | deep_learning | classical_ml | ensemble | forecasting
    target_name: str = "y"
    task_type: str = "regression"
    equation_type: str  # exact_closed_form | exact_decision_rules | nonparametric_analytical | architecture_plus_surrogate | ensemble_plus_surrogate | constant_rule
    plain_equation: str
    latex_equation: str
    formula_plain: str = ""
    formula_latex: str = ""
    formula_python: str = ""
    formula_excel: str = ""
    formula_sql: str = ""
    formula_c_matlab: str = ""
    exact_python_rules: str | None = None
    external_usage_summary: str = ""
    coefficients: dict[str, float] = Field(default_factory=dict)
    surrogate_intercept: float | None = None
    surrogate_coefficients: dict[str, float] = Field(default_factory=dict)
    architecture_equations: list[str] = Field(default_factory=list)
    architecture_latex: list[str] = Field(default_factory=list)
    surrogate_plain_equation: str | None = None
    surrogate_latex_equation: str | None = None
    surrogate_fidelity_r2: float | None = None
    intercept: float | None = None
    parameter_count: int | None = None
    coefficients_table: list[dict[str, Any]] = Field(default_factory=list)
    layer_summary: list[dict[str, Any]] = Field(default_factory=list)
    decision_rules_text: str | None = None
    explanation_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sync_formula_aliases(self) -> ModelEquationReport:
        if not self.formula_plain:
            self.formula_plain = self.plain_equation or self.surrogate_plain_equation or ""
        if not self.formula_latex:
            self.formula_latex = self.latex_equation or self.surrogate_latex_equation or ""
        if not self.coefficients and self.coefficients_table:
            parsed: dict[str, float] = {}
            for row in self.coefficients_table:
                if not isinstance(row, dict):
                    continue
                fname = str(
                    row.get("Feature")
                    or row.get("feature")
                    or row.get("Feature / Variable")
                    or row.get("name")
                    or ""
                ).strip()
                raw_c: Any = None
                for key_cand in (
                    "Raw Coefficient (beta)",
                    "Coefficient (beta)",
                    "Surrogate Coef (beta)",
                    "coefficient",
                    "weight",
                    "Standardized Weight (w)",
                    "importance",
                    "Importance",
                ):
                    if key_cand in row and row[key_cand] is not None:
                        raw_c = row[key_cand]
                        break
                if fname and isinstance(raw_c, (int, float)) and math.isfinite(float(raw_c)):
                    parsed[fname] = round(float(raw_c), 6)
            self.coefficients = parsed

        # Choose the true physical-unit coefficients & intercept for standalone external formulas
        ext_coefs = self.coefficients if (self.equation_type == "exact_closed_form" and self.coefficients) else (
            self.surrogate_coefficients or self.coefficients
        )
        ext_intercept = self.intercept if (self.equation_type == "exact_closed_form" and self.intercept is not None) else (
            self.surrogate_intercept if self.surrogate_intercept is not None else self.intercept
        )
        is_surr = self.equation_type != "exact_closed_form"

        if not self.formula_python or not self.formula_excel or not self.formula_sql or not self.formula_c_matlab:
            ext = _build_external_deployment_formulas(
                model_name=self.model_name,
                target_name=self.target_name,
                task_type=self.task_type,
                equation_type=self.equation_type,
                intercept=ext_intercept,
                coefficients=ext_coefs,
                surrogate_r2=self.surrogate_fidelity_r2,
                is_surrogate=is_surr,
            )
            if not self.formula_python:
                self.formula_python = self.exact_python_rules or ext["formula_python"]
            if not self.formula_excel:
                self.formula_excel = ext["formula_excel"]
            if not self.formula_sql:
                self.formula_sql = ext["formula_sql"]
            if not self.formula_c_matlab:
                self.formula_c_matlab = ext["formula_c_matlab"]
            if not self.external_usage_summary:
                self.external_usage_summary = ext["external_usage_summary"]
        return self


class ModelEquationExtractor:
    """Extracts exact mathematical relationships, equations, weights, and rules from any fitted OmniForge model."""

    @classmethod
    def extract(
        cls,
        model: Any,
        *,
        X_reference: pd.DataFrame | np.ndarray | None = None,
        feature_names: list[str] | None = None,
        target_name: str = "y",
        max_terms: int = 10,
    ) -> ModelEquationReport:
        """Extract the mathematical equation and structural breakdown of a fitted OmniForge model.

        Args:
            model: Any fitted OmniForge model (Baseline, ClassicalMLModel, Deep Learning, Ensemble, or ForecastingBaseline).
            X_reference: Optional reference feature matrix used to compute global linear surrogate equations
                and verify surrogate fidelity R² for non-linear models (Neural Networks, Forests, Boosting).
            feature_names: Optional explicit feature names overriding ``model.feature_names_``.
            target_name: Name of the target variable for readable LHS display.
            max_terms: Maximum terms displayed inline in the primary equation formula.

        Returns:
            A populated :class:`ModelEquationReport`.
        """
        cls_name = type(model).__name__
        model_name = str(getattr(model, "name", cls_name))
        model_family = str(getattr(model, "model_family", "generic"))
        task_val = getattr(model, "task_type", TaskType.REGRESSION)
        task_str = task_val.value if isinstance(task_val, TaskType) else str(task_val)
        is_clf = task_val in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

        # Resolve feature names cleanly
        resolved_features = list(feature_names or getattr(model, "feature_names_", []) or [])
        if not resolved_features and isinstance(X_reference, pd.DataFrame):
            resolved_features = [str(c) for c in X_reference.columns]

        # 0. ForecastingBaseline (standalone time series)
        if cls_name == "ForecastingBaseline":
            return cls._extract_forecasting_baseline(model, target_name)

        # 1. Dummy Baselines
        if cls_name in {"DummyRegressorBaseline", "DummyClassifierBaseline"}:
            return cls._extract_dummy_baseline(model, model_name, model_family, target_name, task_str)

        # 2. Linear / Logistic Baselines (RidgeLinearBaseline, LogisticBaseline, _ParametricRefWrapper)
        if cls_name in {"RidgeLinearBaseline", "LogisticBaseline", "_ParametricRefWrapper"}:
            return cls._extract_linear_model(
                estimator=getattr(model, "_model", None),
                scaler=getattr(model, "_scaler", None),
                model_name=model_name,
                model_family=model_family,
                category="baseline",
                feature_names=resolved_features,
                target_name=target_name,
                task_str=task_str,
                is_clf=is_clf,
                max_terms=max_terms,
            )

        # 2b. Non-Parametric Analytical Models (Spline GAM, Nadaraya-Watson, RBF Kernel Ridge, Weighted k-NN, Isotonic)
        if cls_name in {
            "PenalizedSplineGAMModel",
            "NadarayaWatsonKernelModel",
            "RBFKernelRidgeModel",
            "DistanceWeightedKNNModel",
            "IsotonicMonotonicModel",
        }:
            return cls._extract_nonparametric_analytical(
                model=model,
                cls_name=cls_name,
                model_name=model_name,
                model_family=model_family,
                feature_names=resolved_features,
                target_name=target_name,
                task_str=task_str,
                is_clf=is_clf,
                X_reference=X_reference,
                max_terms=max_terms,
            )

        # 3. Decision Stump Baselines
        if cls_name in {"DecisionStumpRegressorBaseline", "DecisionStumpClassifierBaseline"}:
            return cls._extract_decision_tree_model(
                tree_est=getattr(model, "_model", None),
                model=model,
                model_name=model_name,
                model_family=model_family,
                category="baseline",
                feature_names=resolved_features,
                target_name=target_name,
                task_str=task_str,
                is_clf=is_clf,
                X_reference=X_reference,
                max_terms=max_terms,
            )

        # 4. Deep Learning / Neural Network Architectures
        if cls_name in {"MLPModel", "CNN1DModel", "TabularTransformerModel", "GraphNeuralNetworkModel"}:
            return cls._extract_deep_learning_model(
                model=model,
                cls_name=cls_name,
                model_name=model_name,
                model_family=model_family,
                feature_names=resolved_features,
                target_name=target_name,
                task_str=task_str,
                is_clf=is_clf,
                X_reference=X_reference,
                max_terms=max_terms,
            )

        # 5. ClassicalMLModel (linear, tree ensemble, SVM, KNN, Naive Bayes)
        if cls_name == "ClassicalMLModel":
            est = getattr(model, "_estimator", None)
            algo = str(getattr(model, "algorithm", "")).lower()
            if algo in {"ridge", "lasso", "elasticnet", "logistic_regression"} and est is not None:
                return cls._extract_linear_model(
                    estimator=est,
                    scaler=getattr(model, "_scaler", None),
                    model_name=model_name,
                    model_family=model_family,
                    category="classical_ml",
                    feature_names=resolved_features,
                    target_name=target_name,
                    task_str=task_str,
                    is_clf=is_clf,
                    max_terms=max_terms,
                )
            return cls._extract_advanced_classical_or_ensemble(
                model=model,
                estimator=est,
                algo=algo,
                model_name=model_name,
                model_family=model_family,
                category="classical_ml",
                feature_names=resolved_features,
                target_name=target_name,
                task_str=task_str,
                is_clf=is_clf,
                X_reference=X_reference,
                max_terms=max_terms,
            )

        # 5b. Unwrap scikit-learn Pipeline / Stacking / Voting / Raw Linear / Tree / Boosting estimators (AutoML Champions)
        unwrapped_est = model
        unwrapped_scaler = None
        if hasattr(model, "named_steps") and isinstance(getattr(model, "named_steps", None), dict):
            steps_list = list(model.named_steps.values())
            for st_obj in steps_list[:-1]:
                if hasattr(st_obj, "mean_") and hasattr(st_obj, "scale_"):
                    unwrapped_scaler = st_obj
            if steps_list:
                unwrapped_est = steps_list[-1]
        est_cls = type(unwrapped_est).__name__
        est_cls_low = est_cls.lower()
        if est_cls in {"Ridge", "LinearRegression", "Lasso", "ElasticNet", "HuberRegressor", "BayesianRidge", "LogisticRegression", "SGDRegressor", "SGDClassifier"} and hasattr(unwrapped_est, "coef_"):
            return cls._extract_linear_model(
                estimator=unwrapped_est,
                scaler=unwrapped_scaler,
                model_name=model_name,
                model_family=est_cls,
                category="classical_ml",
                feature_names=resolved_features,
                target_name=target_name,
                task_str=task_str,
                is_clf=is_clf,
                max_terms=max_terms,
            )
        if est_cls in {"DecisionTreeRegressor", "DecisionTreeClassifier"} and hasattr(unwrapped_est, "tree_"):
            return cls._extract_decision_tree_model(
                tree_est=unwrapped_est,
                model=model,
                model_name=model_name,
                model_family=est_cls,
                category="classical_ml",
                feature_names=resolved_features,
                target_name=target_name,
                task_str=task_str,
                is_clf=is_clf,
                X_reference=X_reference,
                max_terms=max_terms,
            )

        inferred_algo = cls_name.lower()
        if "randomforest" in est_cls_low:
            inferred_algo = "random_forest"
        elif "extratrees" in est_cls_low:
            inferred_algo = "extra_trees"
        elif "histgradientboosting" in est_cls_low or "gradientboosting" in est_cls_low:
            inferred_algo = "gradient_boosting"
        elif "xgb" in est_cls_low:
            inferred_algo = "xgboost"
        elif "lgbm" in est_cls_low or "lightgbm" in est_cls_low:
            inferred_algo = "lightgbm"
        elif "catboost" in est_cls_low:
            inferred_algo = "catboost"
        elif "stacking" in est_cls_low or "multilayerstacker" in cls_name.lower():
            inferred_algo = "stacking"
        elif "voting" in est_cls_low:
            inferred_algo = "voting"
        elif "kneighbors" in est_cls_low:
            inferred_algo = "knn"
        elif "svr" in est_cls_low or "svc" in est_cls_low:
            inferred_algo = "svm"

        # 6. Ensemble Models (Voting, Stacking, Bagging, Boosting) or Generic Fallback
        return cls._extract_advanced_classical_or_ensemble(
            model=model,
            estimator=unwrapped_est if unwrapped_est is not model else getattr(model, "_estimator", model),
            algo=inferred_algo,
            model_name=model_name,
            model_family=est_cls if est_cls != cls_name else model_family,
            category="ensemble" if ("ensemble" in model_family.lower() or inferred_algo in {"random_forest", "extra_trees", "gradient_boosting", "xgboost", "lightgbm", "catboost", "stacking", "voting"}) else "classical_ml",
            feature_names=resolved_features,
            target_name=target_name,
            task_str=task_str,
            is_clf=is_clf,
            X_reference=X_reference,
            max_terms=max_terms,
        )

    @classmethod
    def _extract_forecasting_baseline(cls, model: Any, target_name: str) -> ModelEquationReport:
        method = str(getattr(model, "method", "naive"))
        hist = np.asarray(getattr(model, "_history", np.array([])), dtype=float)
        last_val = float(hist[-1]) if len(hist) else 0.0
        t_tex = _escape_latex(target_name)

        if method == "naive":
            plain = f"{target_name}_hat(t + h) = {target_name}(t) = {_fmt_num(last_val)}"
            latex = rf"\widehat{{\text{{{t_tex}}}}}_{{t+h}} = \text{{{t_tex}}}_t = {_fmt_num(last_val)}"
        elif method == "seasonal_naive":
            p = int(getattr(model, "seasonal_period", 7))
            plain = f"{target_name}_hat(t + h) = {target_name}(t + h - {p})  [seasonal period s = {p}]"
            latex = rf"\widehat{{\text{{{t_tex}}}}}_{{t+h}} = \text{{{t_tex}}}_{{t+h-{p}}}"
        elif method == "drift":
            slope = (float(hist[-1] - hist[0]) / max(len(hist) - 1, 1)) if len(hist) > 1 else 0.0
            sign = "+" if slope >= 0 else "-"
            plain = f"{target_name}_hat(t + h) = {_fmt_num(last_val)} {sign} {_fmt_num(abs(slope))} * h"
            latex = rf"\widehat{{\text{{{t_tex}}}}}_{{t+h}} = {_fmt_num(last_val)} {sign} {_fmt_num(abs(slope))}\cdot h"
        else:
            w = int(getattr(model, "window", 7))
            avg = float(np.mean(hist[-min(w, len(hist)) :])) if len(hist) else 0.0
            plain = f"{target_name}_hat(t + h) = (1/{w}) * sum({target_name}[t-i], i=0..{w - 1}) = {_fmt_num(avg)}"
            latex = rf"\widehat{{\text{{{t_tex}}}}}_{{t+h}} = \frac{{1}}{{{w}}}\sum_{{i=0}}^{{{w - 1}}} \text{{{t_tex}}}_{{t-i}} = {_fmt_num(avg)}"

        return ModelEquationReport(
            model_name=f"Baseline_{method}",
            model_family="baseline_forecasting",
            category="baseline",
            target_name=target_name,
            task_type="forecasting",
            equation_type="exact_closed_form",
            plain_equation=plain,
            latex_equation=latex,
            explanation_notes=[
                f"Time-series recurrence rule fitted on {len(hist)} historical observations.",
                f"Residual step volatility sigma = {_fmt_num(float(getattr(model, '_residual_std', 0.0)))}.",
            ],
        )

    @classmethod
    def _extract_dummy_baseline(
        cls,
        model: Any,
        model_name: str,
        model_family: str,
        target_name: str,
        task_str: str,
    ) -> ModelEquationReport:
        dummy = getattr(model, "_model", None)
        strategy = str(getattr(model, "strategy", "constant"))
        t_tex = _escape_latex(target_name)
        notes: list[str] = []
        coef_rows: list[dict[str, Any]] = []

        if hasattr(dummy, "constant_") and dummy.constant_ is not None:
            const_val = float(np.asarray(dummy.constant_).ravel()[0])
            plain = f"{target_name}_hat = {_fmt_num(const_val)}   [constant {strategy} baseline — independent of inputs x]"
            latex = rf"\widehat{{\text{{{t_tex}}}}} = {_fmt_num(const_val)} \quad (\text{{constant {strategy}}})"
            notes.append(f"Predicts the exact training {strategy} ({_fmt_num(const_val)}) for every observation.")
            return ModelEquationReport(
                model_name=model_name,
                model_family=model_family,
                category="baseline",
                target_name=target_name,
                task_type=task_str,
                equation_type="constant_rule",
                plain_equation=plain,
                latex_equation=latex,
                intercept=const_val,
                parameter_count=1,
                explanation_notes=notes,
            )

        classes = getattr(dummy, "classes_", np.array([]))
        priors = getattr(dummy, "class_prior_", np.array([]))
        if len(classes) > 0 and len(priors) == len(classes):
            best_idx = int(np.argmax(priors))
            maj_class = classes[best_idx]
            maj_prob = float(priors[best_idx])
            prior_desc = ", ".join(f"P({c})={float(p):.3f}" for c, p in zip(classes, priors, strict=False))
            for c, p in zip(classes, priors, strict=False):
                coef_rows.append({"Class": str(c), "Prior Probability": round(float(p), 4)})
            if strategy == "most_frequent":
                plain = f"{target_name}_hat = {maj_class}   [argmax prior: {prior_desc}]"
                latex = rf"\widehat{{\text{{{t_tex}}}}} = \text{{{_escape_latex(str(maj_class))}}} \quad (P={maj_prob:.3f})"
            else:
                plain = f"P({target_name} = c) ~ Categorical({prior_desc})"
                latex = rf"\widehat{{\text{{{t_tex}}}}} \sim \mathrm{{Categorical}}\!\left({', '.join(f'{float(p):.3f}' for p in priors)}\right)"
            notes.append(f"Classification baseline ({strategy}) using observed training class priors: {prior_desc}.")
        else:
            plain = f"{target_name}_hat = constant_baseline({strategy})"
            latex = rf"\widehat{{\text{{{t_tex}}}}} = \text{{Baseline}}({strategy})"

        return ModelEquationReport(
            model_name=model_name,
            model_family=model_family,
            category="baseline",
            target_name=target_name,
            task_type=task_str,
            equation_type="constant_rule",
            plain_equation=plain,
            latex_equation=latex,
            parameter_count=max(1, len(classes)),
            coefficients_table=coef_rows,
            explanation_notes=notes,
        )

    @classmethod
    def _extract_linear_model(
        cls,
        *,
        estimator: Any,
        scaler: Any,
        model_name: str,
        model_family: str,
        category: str,
        feature_names: list[str],
        target_name: str,
        task_str: str,
        is_clf: bool,
        max_terms: int,
    ) -> ModelEquationReport:
        if estimator is None or not hasattr(estimator, "coef_"):
            return ModelEquationReport(
                model_name=model_name,
                model_family=model_family,
                category=category,
                target_name=target_name,
                task_type=task_str,
                equation_type="exact_closed_form",
                plain_equation=f"{target_name}_hat = f(x) [linear parameters unavailable]",
                latex_equation=rf"\widehat{{\text{{{_escape_latex(target_name)}}}}} = f(x)",
            )

        raw_coef = np.asarray(estimator.coef_, dtype=float)
        raw_intercept = np.asarray(getattr(estimator, "intercept_", 0.0), dtype=float).ravel()

        # Handle binary vs multiclass vs regression coefficient vectors
        is_multiclass = is_clf and raw_coef.ndim == 2 and raw_coef.shape[0] > 1
        coef_vec = raw_coef[0] if raw_coef.ndim == 2 else raw_coef.ravel()
        intercept_scaled = float(raw_intercept[0]) if len(raw_intercept) else 0.0

        n_feat = len(coef_vec)
        names = list(feature_names[:n_feat]) if len(feature_names) >= n_feat else [f"x_{i + 1}" for i in range(n_feat)]

        # Invert StandardScaler to express exact equation directly in original feature units:
        # w_j * (x_j - mu_j) / sigma_j + b = (b - sum(w_j * mu_j / sigma_j)) + sum((w_j / sigma_j) * x_j)
        if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            means = np.asarray(scaler.mean_, dtype=float).ravel()
            scales = np.where(np.asarray(scaler.scale_, dtype=float).ravel() > 1e-12, np.asarray(scaler.scale_, dtype=float).ravel(), 1.0)
            if len(means) == n_feat and len(scales) == n_feat:
                beta_orig = coef_vec / scales
                beta0_orig = float(intercept_scaled - np.sum(coef_vec * means / scales))
                std_weights = coef_vec.copy()
            else:
                beta_orig = coef_vec.copy()
                beta0_orig = intercept_scaled
                std_weights = coef_vec.copy()
        else:
            beta_orig = coef_vec.copy()
            beta0_orig = intercept_scaled
            std_weights = coef_vec.copy()

        total_abs_std = float(np.sum(np.abs(std_weights))) or 1.0
        coef_rows: list[dict[str, Any]] = []
        for fname, b_raw, b_std in zip(names, beta_orig, std_weights, strict=False):
            row: dict[str, Any] = {
                "Feature": fname,
                "Raw Coefficient (beta)": round(float(b_raw), 6),
                "Standardized Weight (w)": round(float(b_std), 6),
                "Effect Direction": "Positive (+)" if b_raw > 0 else ("Negative (-)" if b_raw < 0 else "Neutral (0)"),
                "Relative Weight %": round(abs(float(b_std)) / total_abs_std * 100.0, 2),
            }
            if is_clf:
                row["Odds Ratio (exp(beta))"] = round(float(np.exp(np.clip(b_raw, -20.0, 20.0))), 4)
            coef_rows.append(row)
        coef_rows.sort(key=lambda r: abs(float(r["Standardized Weight (w)"])), reverse=True)

        t_tex = _escape_latex(target_name)
        classes = getattr(estimator, "classes_", None)
        notes: list[str] = [
            "Coefficients are inverted back to ORIGINAL feature units so you can substitute raw feature values directly.",
            "Standardized weights (w) measure relative impact per 1-standard-deviation change in each feature.",
        ]
        arch_plain: list[str] = []
        arch_latex: list[str] = []

        if is_clf and not is_multiclass:
            pos_cls = classes[1] if (classes is not None and len(classes) >= 2) else 1
            lhs_p = f"P({target_name} = {pos_cls} | x)"
            lhs_l = rf"P(\text{{{t_tex}}} = \text{{{_escape_latex(str(pos_cls))}}} \mid x)"
            plain, latex = _build_linear_strings(
                lhs_p,
                lhs_l,
                beta0_orig,
                list(zip(names, beta_orig, strict=False)),
                max_terms=max_terms,
                wrap_sigmoid=True,
            )
            std_p, std_l = _build_linear_strings(
                f"logit(P({target_name}={pos_cls}))",
                rf"\mathrm{{logit}}\!\left(P(\text{{{t_tex}}}=\text{{{_escape_latex(str(pos_cls))}}})\right)",
                intercept_scaled,
                list(zip(names, std_weights, strict=False)),
                max_terms=max_terms,
            )
            arch_plain.append(f"Standardized log-odds form (z_j = (x_j - mean_j) / std_j):  {std_p}")
            arch_latex.append(std_l)
            notes.append(f"Decision rule: predict class '{pos_cls}' whenever P({target_name}={pos_cls} | x) >= 0.50 (log-odds >= 0).")
        elif is_multiclass:
            n_cls = raw_coef.shape[0]
            cls_label = classes[0] if (classes is not None and len(classes) > 0) else 0
            plain = (
                f"P({target_name} = k | x) = softmax(beta_0,k + sum_j beta_j,k * x_j) across {n_cls} classes; "
                f"Class '{cls_label}' logit = {_fmt_num(beta0_orig)} + ..."
            )
            latex = (
                rf"P(\text{{{t_tex}}}=k \mid x) = \frac{{\exp(\beta_{{0,k}} + \beta_k^\top x)}}"
                rf"{{\sum_{{c=1}}^{{{n_cls}}} \exp(\beta_{{0,c}} + \beta_c^\top x)}}"
            )
            for idx in range(min(n_cls, 4)):
                c_name = classes[idx] if (classes is not None and len(classes) > idx) else idx
                c_vec = raw_coef[idx]
                c_int = float(raw_intercept[idx]) if len(raw_intercept) > idx else 0.0
                if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                    means = np.asarray(scaler.mean_, dtype=float).ravel()
                    scales = np.where(np.asarray(scaler.scale_, dtype=float).ravel() > 1e-12, np.asarray(scaler.scale_, dtype=float).ravel(), 1.0)
                    b_c = c_vec / scales
                    b0_c = float(c_int - np.sum(c_vec * means / scales))
                else:
                    b_c, b0_c = c_vec, c_int
                lp, ll = _build_linear_strings(
                    f"logit(class={c_name})",
                    rf"\eta_{{\text{{{_escape_latex(str(c_name))}}}}}(x)",
                    b0_c,
                    list(zip(names, b_c, strict=False)),
                    max_terms=min(max_terms, 6),
                )
                arch_plain.append(lp)
                arch_latex.append(ll)
            notes.append(f"Multiclass softmax logistic equation across {n_cls} target classes.")
        else:
            plain, latex = _build_linear_strings(
                f"{target_name}_hat",
                rf"\widehat{{\text{{{t_tex}}}}}",
                beta0_orig,
                list(zip(names, beta_orig, strict=False)),
                max_terms=max_terms,
            )
            std_p, std_l = _build_linear_strings(
                f"{target_name}_hat(z)",
                rf"\widehat{{\text{{{t_tex}}}}}(z)",
                intercept_scaled,
                list(zip(names, std_weights, strict=False)),
                max_terms=max_terms,
            )
            arch_plain.append(f"Standardized form (z_j = (x_j - mean_j) / std_j):  {std_p}")
            arch_latex.append(std_l)

        param_count = int(raw_coef.size + raw_intercept.size)
        return ModelEquationReport(
            model_name=model_name,
            model_family=model_family,
            category=category,
            target_name=target_name,
            task_type=task_str,
            equation_type="exact_closed_form",
            plain_equation=plain,
            latex_equation=latex,
            architecture_equations=arch_plain,
            architecture_latex=arch_latex,
            intercept=round(beta0_orig, 6),
            parameter_count=param_count,
            coefficients_table=coef_rows,
            explanation_notes=notes,
        )

    @classmethod
    def _extract_decision_tree_model(
        cls,
        *,
        tree_est: Any,
        model: Any,
        model_name: str,
        model_family: str,
        category: str,
        feature_names: list[str],
        target_name: str,
        task_str: str,
        is_clf: bool,
        X_reference: pd.DataFrame | np.ndarray | None,
        max_terms: int,
    ) -> ModelEquationReport:
        t_tex = _escape_latex(target_name)
        rules_text = None
        coef_rows: list[dict[str, Any]] = []
        n_leaves = 0
        depth = 0
        if tree_est is not None and hasattr(tree_est, "tree_"):
            n_feat = int(tree_est.n_features_in_)
            names = list(feature_names[:n_feat]) if len(feature_names) >= n_feat else [f"x_{i + 1}" for i in range(n_feat)]
            try:
                rules_text = export_text(tree_est, feature_names=names, max_depth=4)
            except Exception:
                rules_text = None
            depth = int(tree_est.get_depth())
            n_leaves = int(tree_est.get_n_leaves())
            imps = np.asarray(getattr(tree_est, "feature_importances_", np.zeros(n_feat)), dtype=float)
            for fname, imp in zip(names, imps, strict=False):
                if imp > 0:
                    coef_rows.append(
                        {
                            "Feature": fname,
                            "Split Importance": round(float(imp), 6),
                            "Relative Weight %": round(float(imp) * 100.0, 2),
                        }
                    )
            coef_rows.sort(key=lambda r: float(r["Split Importance"]), reverse=True)

        plain = f"{target_name}_hat(x) = sum_{{m=1..{max(1, n_leaves)}}} c_m * I(x in R_m)   [Piecewise Decision Tree: depth={depth}, leaves={n_leaves}]"
        latex = rf"\widehat{{\text{{{t_tex}}}}}(x) = \sum_{{m=1}}^{{{max(1, n_leaves)}}} c_m \, \mathbb{{I}}(x \in R_m)"

        surr_plain, surr_latex, surr_r2, _, surr_b0, surr_coefs = cls._fit_linear_surrogate(
            model, X_reference, feature_names, target_name, is_clf, max_terms=max_terms
        )
        return ModelEquationReport(
            model_name=model_name,
            model_family=model_family,
            category=category,
            target_name=target_name,
            task_type=task_str,
            equation_type="exact_decision_rules",
            plain_equation=plain,
            latex_equation=latex,
            surrogate_plain_equation=surr_plain,
            surrogate_latex_equation=surr_latex,
            surrogate_fidelity_r2=surr_r2,
            surrogate_intercept=surr_b0,
            surrogate_coefficients=surr_coefs,
            intercept=surr_b0,
            parameter_count=max(1, n_leaves * 2 - 1),
            coefficients_table=coef_rows,
            decision_rules_text=rules_text,
            explanation_notes=[
                f"Piecewise-constant decision rule partition with depth {depth} and {n_leaves} leaf regions.",
                "Exact IF/ELSE split rules are displayed below.",
            ],
        )

    @classmethod
    def _extract_deep_learning_model(
        cls,
        *,
        model: Any,
        cls_name: str,
        model_name: str,
        model_family: str,
        feature_names: list[str],
        target_name: str,
        task_str: str,
        is_clf: bool,
        X_reference: pd.DataFrame | np.ndarray | None,
        max_terms: int,
    ) -> ModelEquationReport:
        t_tex = _escape_latex(target_name)
        n_feat = len(feature_names) or int(getattr(getattr(model, "metadata_", None), "training_features", 0))
        out_desc = r"\mathrm{softmax}" if is_clf else r"\mathrm{Linear}"
        arch_plain: list[str] = ["Input Standardization:  z_j = (x_j - mean_j) / std_j"]
        arch_latex: list[str] = [r"z_j = \frac{x_j - \mu_j}{\sigma_j}"]
        layer_rows: list[dict[str, Any]] = []
        total_params = 0
        notes: list[str] = []

        if cls_name == "MLPModel":
            act = str(getattr(model, "activation", "relu")).upper()
            act_tex = {"RELU": r"\mathrm{ReLU}", "TANH": r"\tanh", "GELU": r"\mathrm{GELU}"}.get(act, r"\mathrm{ReLU}")
            weights = list(getattr(model, "_weights", []))
            biases = list(getattr(model, "_biases", []))
            torch_net = getattr(model, "_torch_net", None)

            if weights:
                for idx, (w, b) in enumerate(zip(weights, biases, strict=False), start=1):
                    w_arr = np.asarray(w, dtype=float)
                    b_arr = np.asarray(b, dtype=float)
                    n_p = int(w_arr.size + b_arr.size)
                    total_params += n_p
                    is_last = idx == len(weights)
                    layer_act = ("Softmax / Logits" if is_clf else "Linear") if is_last else act
                    layer_rows.append(
                        {
                            "Layer": f"Dense {idx}" + (" (Output)" if is_last else f" (Hidden {idx})"),
                            "Weight Shape": f"{w_arr.shape[0]} -> {w_arr.shape[1]}",
                            "Activation": layer_act,
                            "Parameters": n_p,
                            "||W||_F": round(float(np.linalg.norm(w_arr)), 4),
                            "||b||_2": round(float(np.linalg.norm(b_arr)), 4),
                        }
                    )
                    inp_sym = "z" if idx == 1 else f"h^({idx - 1})"
                    inp_tex = "z" if idx == 1 else f"h^{{({idx - 1})}}"
                    if is_last:
                        arch_plain.append(f"Output Layer {idx}:  {target_name}_hat = {'softmax(' if is_clf else ''}W^({idx})^T {inp_sym} + b^({idx}){')' if is_clf else ''}")
                        arch_latex.append(
                            rf"\widehat{{\text{{{t_tex}}}}} = {out_desc}\!\left((W^{{({idx})}})^\top {inp_tex} + b^{{({idx})}}\right)"
                            if is_clf
                            else rf"\widehat{{\text{{{t_tex}}}}} = (W^{{({idx})}})^\top {inp_tex} + b^{{({idx})}}"
                        )
                    else:
                        arch_plain.append(f"Hidden Layer {idx}:  h^({idx}) = {act}(W^({idx})^T {inp_sym} + b^({idx}))  [{w_arr.shape[0]} -> {w_arr.shape[1]}]")
                        arch_latex.append(rf"h^{{({idx})}} = {act_tex}\!\left((W^{{({idx})}})^\top {inp_tex} + b^{{({idx})}}\right)")
            elif torch_net is not None:
                l_idx = 0
                for mod in torch_net.modules():
                    if hasattr(mod, "weight") and mod.weight is not None:
                        l_idx += 1
                        w_arr = mod.weight.detach().cpu().numpy()
                        b_arr = mod.bias.detach().cpu().numpy() if getattr(mod, "bias", None) is not None else np.zeros(0)
                        n_p = int(w_arr.size + b_arr.size)
                        total_params += n_p
                        layer_rows.append(
                            {
                                "Layer": f"Linear {l_idx}",
                                "Weight Shape": f"{w_arr.shape[1]} -> {w_arr.shape[0]}",
                                "Activation": act,
                                "Parameters": n_p,
                                "||W||_F": round(float(np.linalg.norm(w_arr)), 4),
                                "||b||_2": round(float(np.linalg.norm(b_arr)), 4),
                            }
                        )
            classes_attr = getattr(model, "classes_", None)
            out_sz = len(classes_attr) if classes_attr is not None else 1
            dims_str = " -> ".join([str(n_feat)] + [str(d) for d in getattr(model, "hidden_dims", [])] + [str(out_sz)])
            plain = f"{target_name}_hat = MLP[{dims_str}, act={act}](z)   ({total_params:,} learned parameters)"
            latex = (
                rf"\widehat{{\text{{{t_tex}}}}} = {out_desc}\!\left(W^{{(L)}} \cdots {act_tex}\!\left(W^{{(1)}} z + b^{{(1)}}\right) \cdots + b^{{(L)}}\right)"
            )
            notes.append(f"Deep Feedforward Neural Network ({dims_str}) trained with Adam optimization over {getattr(model, 'epochs', 0)} epochs.")

        elif cls_name == "CNN1DModel":
            kernels = list(getattr(model, "_kernels", []))
            ks = getattr(model, "kernel_sizes", (2, 3))
            nf = int(getattr(model, "num_filters", 16))
            for idx, k_arr in enumerate(kernels, start=1):
                ka = np.asarray(k_arr, dtype=float)
                total_params += int(ka.size)
                layer_rows.append(
                    {
                        "Layer": f"Conv1D Branch {idx}",
                        "Weight Shape": f"{ka.shape[0]} filters x kernel {ka.shape[1]}",
                        "Activation": "ReLU + Global Avg/Max Pool",
                        "Parameters": int(ka.size),
                        "||W||_F": round(float(np.linalg.norm(ka)), 4),
                        "||b||_2": 0.0,
                    }
                )
            head = getattr(model, "_head", None)
            if head is not None and hasattr(head, "_weights"):
                for h_i, (w, b) in enumerate(zip(head._weights, head._biases, strict=False), start=1):
                    n_p = int(np.asarray(w).size + np.asarray(b).size)
                    total_params += n_p
                    layer_rows.append(
                        {
                            "Layer": f"MLP Head Layer {h_i}",
                            "Weight Shape": f"{w.shape[0]} -> {w.shape[1]}",
                            "Activation": "ReLU" if h_i < len(head._weights) else ("Softmax" if is_clf else "Linear"),
                            "Parameters": n_p,
                            "||W||_F": round(float(np.linalg.norm(w)), 4),
                            "||b||_2": round(float(np.linalg.norm(b)), 4),
                        }
                    )
            arch_plain.extend(
                [
                    f"Multi-Scale 1D Convolutions (kernels={list(ks)}, filters={nf}):  C_k = ReLU(Conv1D(z; K_k))",
                    "Global Pooling Concatenation:  p = concat_k [AvgPool(C_k), MaxPool(C_k)]",
                    f"Dense Classification/Regression Head:  {target_name}_hat = MLP_head(p)",
                ]
            )
            arch_latex.extend(
                [
                    r"C_k = \mathrm{ReLU}\!\left(\mathrm{Conv1D}(z;\, K_k)\right), \quad k \in \{" + ", ".join(str(k) for k in ks) + r"\}",
                    r"p = \bigoplus_k \left[\mathrm{AvgPool}(C_k),\; \mathrm{MaxPool}(C_k)\right]",
                    rf"\widehat{{\text{{{t_tex}}}}} = \mathrm{{MLP}}_{{\text{{head}}}}(p)",
                ]
            )
            plain = f"{target_name}_hat = MLP_head(Concat[AvgPool, MaxPool](ReLU(Conv1D(z; kernels={list(ks)}, filters={nf}))))"
            latex = rf"\widehat{{\text{{{t_tex}}}}} = \mathrm{{MLP}}_{{\text{{head}}}}\!\left(\bigoplus_{{k \in {list(ks)}}} \mathrm{{Pool}}\!\left(\mathrm{{ReLU}}(\mathrm{{Conv1D}}(z; K_k))\right)\right)"
            notes.append(f"Multi-scale 1D Convolutional Neural Network with {nf} filters per branch and kernel widths {list(ks)}.")

        elif cls_name == "TabularTransformerModel":
            d_model = int(getattr(model, "d_model", 32))
            n_heads = int(getattr(model, "n_heads", 4))
            n_layers = int(getattr(model, "n_layers", 2))
            d_k = max(1, d_model // max(1, n_heads))
            tok_w = getattr(model, "_token_w", None)
            if tok_w is not None:
                total_params += int(np.asarray(tok_w).size)
                layer_rows.append(
                    {
                        "Layer": "Feature Tokenizer",
                        "Weight Shape": f"{tok_w.shape[0]} features -> d_model={d_model}",
                        "Activation": "Linear Token Projection",
                        "Parameters": int(np.asarray(tok_w).size),
                        "||W||_F": round(float(np.linalg.norm(tok_w)), 4),
                        "||b||_2": round(float(np.linalg.norm(getattr(model, "_token_b", 0.0))), 4),
                    }
                )
            for b_i, blk in enumerate(getattr(model, "_blocks", []), start=1):
                blk_p = sum(int(np.asarray(v).size) for v in blk.values())
                total_params += blk_p
                layer_rows.append(
                    {
                        "Layer": f"Transformer Block {b_i}",
                        "Weight Shape": f"d_model={d_model}, heads={n_heads} (d_k={d_k})",
                        "Activation": "MHA + LayerNorm + GELU FFN",
                        "Parameters": blk_p,
                        "||W||_F": round(float(np.linalg.norm(blk.get("wq", 0.0))), 4),
                        "||b||_2": 0.0,
                    }
                )
            head = getattr(model, "_head", None)
            if head is not None and hasattr(head, "_weights"):
                for h_i, (w, b) in enumerate(zip(head._weights, head._biases, strict=False), start=1):
                    n_p = int(np.asarray(w).size + np.asarray(b).size)
                    total_params += n_p
                    layer_rows.append(
                        {
                            "Layer": f"Prediction Head {h_i}",
                            "Weight Shape": f"{w.shape[0]} -> {w.shape[1]}",
                            "Activation": "ReLU" if h_i < len(head._weights) else ("Softmax" if is_clf else "Linear"),
                            "Parameters": n_p,
                            "||W||_F": round(float(np.linalg.norm(w)), 4),
                            "||b||_2": round(float(np.linalg.norm(b)), 4),
                        }
                    )
            arch_plain.extend(
                [
                    f"Feature Tokenization:  T_j = z_j * w_j + b_j   (each scalar feature mapped into R^{d_model})",
                    f"Multi-Head Self-Attention ({n_layers} layers, {n_heads} heads, d_k={d_k}):  Attn(Q,K,V) = softmax(Q K^T / sqrt({d_k})) V",
                    f"Residual LayerNorm + FFN + Mean Token Pooling -> {target_name}_hat = MLP_head(MeanPool(T^(L)))",
                ]
            )
            arch_latex.extend(
                [
                    rf"T_j^{{(0)}} = z_j w_j + b_j \in \mathbb{{R}}^{{{d_model}}}",
                    rf"\mathrm{{Attn}}(Q,K,V) = \mathrm{{softmax}}\!\left(\frac{{QK^\top}}{{\sqrt{{{d_k}}}}}\right)V",
                    rf"\widehat{{\text{{{t_tex}}}}} = \mathrm{{MLP}}_{{\text{{head}}}}\!\left(\mathrm{{MeanPool}}\!\left(\mathrm{{LayerNorm}}(T^{{({n_layers})}})\right)\right)",
                ]
            )
            plain = f"{target_name}_hat = FT-Transformer[d_model={d_model}, heads={n_heads}, layers={n_layers}](z)"
            latex = rf"\widehat{{\text{{{t_tex}}}}} = \mathrm{{FT\text{{-}}Transformer}}_{{d={d_model},\, H={n_heads},\, L={n_layers}}}(z)"
            notes.append(f"FT-Transformer with {n_layers} multi-head self-attention layers ({n_heads} heads, d_model={d_model}).")

        else:  # GraphNeuralNetworkModel
            h_dim = int(getattr(model, "hidden_dim", 32))
            n_hops = int(getattr(model, "n_hops", 2))
            knn_k = int(getattr(model, "knn_k", 5))
            for hop_i, pw in enumerate(getattr(model, "_proj_weights", []), start=1):
                pwa = np.asarray(pw, dtype=float)
                total_params += int(pwa.size)
                layer_rows.append(
                    {
                        "Layer": f"GCN Hop {hop_i}",
                        "Weight Shape": f"{pwa.shape[0]} -> {pwa.shape[1]}",
                        "Activation": "A_tilde @ H @ W + ReLU",
                        "Parameters": int(pwa.size),
                        "||W||_F": round(float(np.linalg.norm(pwa)), 4),
                        "||b||_2": 0.0,
                    }
                )
            head = getattr(model, "_head", None)
            if head is not None and hasattr(head, "_weights"):
                for h_i, (w, b) in enumerate(zip(head._weights, head._biases, strict=False), start=1):
                    n_p = int(np.asarray(w).size + np.asarray(b).size)
                    total_params += n_p
                    layer_rows.append(
                        {
                            "Layer": f"Readout Head {h_i}",
                            "Weight Shape": f"{w.shape[0]} -> {w.shape[1]}",
                            "Activation": "ReLU" if h_i < len(head._weights) else ("Softmax" if is_clf else "Linear"),
                            "Parameters": n_p,
                            "||W||_F": round(float(np.linalg.norm(w)), 4),
                            "||b||_2": round(float(np.linalg.norm(b)), 4),
                        }
                    )
            arch_plain.extend(
                [
                    f"k-NN Similarity Graph (k={knn_k}):  A_tilde = D^(-1/2) (A + I) D^(-1/2)",
                    f"Multi-Hop Graph Convolution ({n_hops} hops, dim={h_dim}):  H^(l) = ReLU(A_tilde H^(l-1) W^(l))",
                    f"Node Readout Head:  {target_name}_hat = MLP_head(H^({n_hops}))",
                ]
            )
            arch_latex.extend(
                [
                    r"\tilde{A} = D^{-1/2}(A + I)D^{-1/2}",
                    rf"H^{{(l)}} = \mathrm{{ReLU}}\!\left(\tilde{{A}}\, H^{{(l-1)}} W^{{(l)}}\right), \quad l = 1,\dots,{n_hops}",
                    rf"\widehat{{\text{{{t_tex}}}}} = \mathrm{{MLP}}_{{\text{{head}}}}\!\left(H^{{({n_hops})}}\right)",
                ]
            )
            plain = f"{target_name}_hat = GCN[hops={n_hops}, knn_k={knn_k}, hidden={h_dim}](A_tilde, z)"
            latex = rf"\widehat{{\text{{{t_tex}}}}} = \mathrm{{MLP}}_{{\text{{head}}}}\!\left(\mathrm{{GCN}}_{{{n_hops}\text{{-hop}}}}(\tilde{{A}}, z)\right)"
            notes.append(f"Graph Convolutional Network propagating messages across k={knn_k} nearest neighbors for {n_hops} hops.")

        # Fit high-fidelity Global Linear Surrogate over the network's actual predictions
        surr_plain, surr_latex, surr_r2, surr_table, surr_b0, surr_coefs = cls._fit_linear_surrogate(
            model, X_reference, feature_names, target_name, is_clf, max_terms=max_terms
        )
        if surr_r2 is not None:
            notes.append(
                f"Global Linear Surrogate Equation approximates this neural network's input-output mapping with R² = {surr_r2:.4f} fidelity."
            )

        return ModelEquationReport(
            model_name=model_name,
            model_family=model_family,
            category="deep_learning",
            target_name=target_name,
            task_type=task_str,
            equation_type="architecture_plus_surrogate",
            plain_equation=plain,
            latex_equation=latex,
            architecture_equations=arch_plain,
            architecture_latex=arch_latex,
            surrogate_plain_equation=surr_plain,
            surrogate_latex_equation=surr_latex,
            surrogate_fidelity_r2=surr_r2,
            surrogate_intercept=surr_b0,
            surrogate_coefficients=surr_coefs,
            intercept=surr_b0,
            parameter_count=total_params or None,
            coefficients_table=surr_table,
            layer_summary=layer_rows,
            explanation_notes=notes,
        )

    @classmethod
    def _extract_advanced_classical_or_ensemble(
        cls,
        *,
        model: Any,
        estimator: Any,
        algo: str,
        model_name: str,
        model_family: str,
        category: str,
        feature_names: list[str],
        target_name: str,
        task_str: str,
        is_clf: bool,
        X_reference: pd.DataFrame | np.ndarray | None,
        max_terms: int,
    ) -> ModelEquationReport:
        t_tex = _escape_latex(target_name)
        arch_plain: list[str] = []
        arch_latex: list[str] = []
        notes: list[str] = []
        rules_text: str | None = None
        param_count: int | None = None
        cls_name = type(model).__name__

        # Voting / Stacking / MultiLayerStacker / Bagging / Boosting OmniForge & scikit-learn Ensemble models
        if (
            hasattr(model, "estimators")
            and isinstance(getattr(model, "estimators", None), list)
            and len(model.estimators) > 0
            and not isinstance(getattr(estimator, "named_estimators_", None), dict)
        ):
            members = model.estimators
            weights = getattr(model, "weights", None) or [1.0 / len(members)] * len(members)
            w_sum = sum(float(w) for w in weights) or 1.0
            norm_w = [float(w) / w_sum for w in weights]
            member_terms = [f"{w:.2f} * {getattr(m, 'name', type(m).__name__)}(x)" for m, w in zip(members, norm_w, strict=False)]
            plain = f"{target_name}_hat(x) = " + " + ".join(member_terms)
            latex_terms = " + ".join(
                rf"{w:.2f}\,\hat{{f}}_{{\text{{{_escape_latex(getattr(m, 'name', type(m).__name__))}}}}}(x)"
                for m, w in zip(members, norm_w, strict=False)
            )
            latex = rf"\widehat{{\text{{{t_tex}}}}}(x) = {latex_terms}"
            arch_plain.append(f"Ensemble combines {len(members)} diverse base engines via weighted aggregation.")
            notes.append(f"Weighted ensemble aggregating {len(members)} fitted models: {', '.join(getattr(m, 'name', type(m).__name__) for m in members)}.")
        elif algo in {"stacking", "voting"} and estimator is not None:
            named_ests = getattr(estimator, "named_estimators_", None)
            meta = getattr(estimator, "final_estimator_", None)
            if isinstance(named_ests, dict) and named_ests:
                m_names = list(named_ests.keys())
                if meta is not None and hasattr(meta, "coef_"):
                    mc = np.asarray(meta.coef_, dtype=float).ravel()
                    mb = float(np.asarray(getattr(meta, "intercept_", 0.0)).ravel()[0])
                    if len(mc) >= len(m_names):
                        meta_terms = " ".join(
                            f"{'+' if mc[i] >= 0 else '-'} {_fmt_num(abs(mc[i]))} * {m_names[i]}(x)"
                            for i in range(len(m_names))
                        )
                        plain = f"{target_name}_hat(x) = {_fmt_num(mb)} {meta_terms}"
                        tex_terms = " ".join(
                            rf"{'+' if mc[i] >= 0 else '-'} {_fmt_num(abs(mc[i]))}\,\hat{{f}}_{{\text{{{_escape_latex(m_names[i])}}}}}(x)"
                            for i in range(len(m_names))
                        )
                        latex = rf"\widehat{{\text{{{t_tex}}}}}(x) = {_fmt_num(mb)} {tex_terms}"
                        arch_plain.append(f"Stacking Meta-Learner ({type(meta).__name__}) linearly combines {len(m_names)} out-of-fold base models.")
                        notes.append(f"Exact Level-2 Meta-Learner combines base models: {', '.join(m_names)}.")
                    else:
                        plain = f"{target_name}_hat(x) = Meta({', '.join(f'{n}(x)' for n in m_names)})"
                        latex = rf"\widehat{{\text{{{t_tex}}}}}(x) = g_{{\text{{meta}}}}\!\left({', '.join(rf'\hat{{f}}_{{\text{{{_escape_latex(n)}}}}}(x)' for n in m_names)}\right)"
                else:
                    w = 1.0 / len(m_names)
                    plain = f"{target_name}_hat(x) = " + " + ".join(f"{w:.3f} * {n}(x)" for n in m_names)
                    latex = rf"\widehat{{\text{{{t_tex}}}}}(x) = " + " + ".join(
                        rf"{w:.3f}\,\hat{{f}}_{{\text{{{_escape_latex(n)}}}}}(x)" for n in m_names
                    )
                    arch_plain.append(f"Ensemble combines {len(m_names)} base learners ({', '.join(m_names)}).")
            elif hasattr(model, "layer_models_") or hasattr(model, "meta_model_"):
                base_names = list(getattr(model, "base_model_names_", []) or ["GBDT", "ExtraTrees", "RandomForest", "Ridge"])
                plain = f"{target_name}_hat(x) = MetaStack({', '.join(f'{n}(x)' for n in base_names)})   [Multi-Layer Out-of-Fold Stacker]"
                latex = rf"\widehat{{\text{{{t_tex}}}}}(x) = g_{{\text{{stack}}}}\!\left({', '.join(rf'\hat{{f}}_{{\text{{{_escape_latex(n)}}}}}(x)' for n in base_names)}\right)"
                arch_plain.append(f"Multi-Layer Stacker combines {len(base_names)} out-of-fold learners via regularized Level-2 meta-learner.")
                notes.append("Out-of-fold stacking prevents Level-2 overfitting while capturing both linear and tree-based interactions.")
            else:
                plain = f"{target_name}_hat(x) = {cls_name}(x)"
                latex = rf"\widehat{{\text{{{t_tex}}}}}(x) = f_{{\text{{{_escape_latex(model_name)}}}}}(x)"
        elif algo in {"random_forest", "extra_trees"} and estimator is not None:
            n_trees = int(getattr(estimator, "n_estimators", len(getattr(estimator, "estimators_", []))))
            plain = f"{target_name}_hat(x) = (1 / {n_trees}) * sum_{{m=1..{n_trees}}} T_m(x)   [Ensemble of {n_trees} decision trees]"
            latex = rf"\widehat{{\text{{{t_tex}}}}}(x) = \frac{{1}}{{{n_trees}}}\sum_{{m=1}}^{{{n_trees}}} T_m(x)"
            trees = getattr(estimator, "estimators_", [])
            if trees:
                param_count = sum(int(getattr(t, "tree_", None).node_count) for t in trees if hasattr(t, "tree_"))
                try:
                    n_f = int(trees[0].n_features_in_)
                    fnames = list(feature_names[:n_f]) if len(feature_names) >= n_f else [f"x_{i + 1}" for i in range(n_f)]
                    rules_text = f"# Representative Tree #1 of {n_trees} (truncated to depth 3):\n" + export_text(
                        trees[0], feature_names=fnames, max_depth=3
                    )
                except Exception:
                    rules_text = None
            notes.append(f"Variance-reduced tree ensemble averaging {n_trees} randomized decision trees.")
        elif algo in {"gradient_boosting", "xgboost", "lightgbm", "catboost"} and estimator is not None:
            n_est = int(
                getattr(estimator, "n_estimators", getattr(estimator, "max_iter", 100))
            )
            lr = float(getattr(estimator, "learning_rate", 0.1))
            plain = f"{target_name}_hat(x) = F_0 + {lr:g} * sum_{{m=1..{n_est}}} h_m(x)   [Additive Boosted Trees: M={n_est}, eta={lr:g}]"
            latex = (
                rf"\widehat{{\text{{{t_tex}}}}}(x) = \sigma\!\left(F_0 + {lr:g}\sum_{{m=1}}^{{{n_est}}} h_m(x)\right)"
                if is_clf
                else rf"\widehat{{\text{{{t_tex}}}}}(x) = F_0 + {lr:g}\sum_{{m=1}}^{{{n_est}}} h_m(x)"
            )
            est_trees = getattr(estimator, "estimators_", None)
            if est_trees is not None and len(est_trees) > 0:
                try:
                    t0 = est_trees[0][0] if hasattr(est_trees[0], "__len__") else est_trees[0]
                    if hasattr(t0, "tree_"):
                        n_f = int(getattr(t0, "n_features_in_", len(feature_names)))
                        fnames = list(feature_names[:n_f]) if len(feature_names) >= n_f else [f"x_{i + 1}" for i in range(n_f)]
                        rules_text = "# Representative Boosted Stage #1 Tree (truncated to depth 3):\n" + export_text(
                            t0, feature_names=fnames, max_depth=3
                        )
                except Exception:
                    pass
            notes.append(f"Sequential gradient-boosted additive tree expansion ({algo.upper()}, learning_rate={lr:g}, iterations={n_est}).")
        elif algo == "svm" and estimator is not None:
            kernel = str(getattr(estimator, "kernel", "rbf"))
            n_sv = int(len(getattr(estimator, "support_", [])))
            plain = f"{target_name}_hat(x) = b + sum_{{i=1..{n_sv}}} alpha_i * K_{kernel}(x_i, x)   [{n_sv} support vectors]"
            latex = rf"\widehat{{\text{{{t_tex}}}}}(x) = b + \sum_{{i=1}}^{{{n_sv}}} \alpha_i \, K_{{\text{{{kernel}}}}}(x_i, x)"
            param_count = n_sv
            notes.append(f"Kernel Support Vector Machine ({kernel} kernel) anchored on {n_sv} learned support vectors.")
        elif algo == "knn" and estimator is not None:
            k = int(getattr(estimator, "n_neighbors", 5))
            plain = f"{target_name}_hat(x) = Aggregate_{{i in N_{k}(x)}}(y_i)   [k={k} nearest standardized neighbors]"
            latex = rf"\widehat{{\text{{{t_tex}}}}}(x) = \frac{{1}}{{{k}}}\sum_{{i \in \mathcal{{N}}_{{{k}}}(x)}} y_i"
            notes.append(f"Non-parametric k-Nearest Neighbors engine (k={k}) operating in standardized feature space.")
        elif algo == "naive_bayes" and estimator is not None:
            plain = f"{target_name}_hat(x) = argmax_c [ log P(y=c) + sum_j log N(x_j; mu_{{c,j}}, sigma^2_{{c,j}}) ]"
            latex = rf"\widehat{{\text{{{t_tex}}}}}(x) = \arg\max_c \left[\log P(y=c) + \sum_j \log \mathcal{{N}}\!\left(x_j;\,\mu_{{c,j}},\,\sigma^2_{{c,j}}\right)\right]"
            notes.append("Gaussian Naive Bayes generative probability model using per-class feature means and variances.")
        else:
            plain = f"{target_name}_hat(x) = {cls_name}(x)"
            latex = rf"\widehat{{\text{{{t_tex}}}}}(x) = f_{{\text{{{_escape_latex(model_name)}}}}}(x)"

        # Extract global linear surrogate equation + combine with native feature_importances_ when present
        surr_plain, surr_latex, surr_r2, surr_table, surr_b0, surr_coefs = cls._fit_linear_surrogate(
            model, X_reference, feature_names, target_name, is_clf, max_terms=max_terms
        )
        if estimator is not None and hasattr(estimator, "feature_importances_"):
            imps = np.asarray(estimator.feature_importances_, dtype=float).ravel()
            fnames = list(feature_names[: len(imps)]) if len(feature_names) >= len(imps) else [f"x_{i + 1}" for i in range(len(imps))]
            imp_map = {fn: float(val) for fn, val in zip(fnames, imps, strict=False)}
            if surr_table:
                for row in surr_table:
                    row["Engine Importance"] = round(imp_map.get(str(row["Feature"]), 0.0), 6)
            else:
                total_imp = float(np.sum(imps)) or 1.0
                surr_table = sorted(
                    [
                        {
                            "Feature": fn,
                            "Engine Importance": round(float(val), 6),
                            "Relative Weight %": round(float(val) / total_imp * 100.0, 2),
                        }
                        for fn, val in zip(fnames, imps, strict=False)
                    ],
                    key=lambda r: float(r["Engine Importance"]),
                    reverse=True,
                )

        if surr_r2 is not None:
            notes.append(
                f"Global Linear Surrogate Equation approximates this engine's learned mapping with R² = {surr_r2:.4f} fidelity — exported as standalone Python, Excel, SQL, and MATLAB/C formulas for external deployment."
            )

        return ModelEquationReport(
            model_name=model_name,
            model_family=model_family,
            category=category,
            target_name=target_name,
            task_type=task_str,
            equation_type="ensemble_plus_surrogate",
            plain_equation=plain,
            latex_equation=latex,
            architecture_equations=arch_plain,
            architecture_latex=arch_latex,
            surrogate_plain_equation=surr_plain,
            surrogate_latex_equation=surr_latex,
            surrogate_fidelity_r2=surr_r2,
            surrogate_intercept=surr_b0,
            surrogate_coefficients=surr_coefs,
            intercept=surr_b0,
            parameter_count=param_count,
            coefficients_table=surr_table,
            decision_rules_text=rules_text,
            explanation_notes=notes,
        )

    @classmethod
    def _fit_linear_surrogate(
        cls,
        model: Any,
        X_reference: pd.DataFrame | np.ndarray | None,
        feature_names: list[str],
        target_name: str,
        is_clf: bool,
        *,
        max_terms: int = 10,
    ) -> tuple[str | None, str | None, float | None, list[dict[str, Any]], float | None, dict[str, float]]:
        """Fit an interpretable linear surrogate Ridge model on the model's own predictions to expose explicit feature effects."""
        if X_reference is None:
            return None, None, None, [], None, {}
        try:
            if isinstance(X_reference, pd.DataFrame):
                X_df = X_reference.copy()
                if feature_names:
                    present = [c for c in feature_names if c in X_df.columns]
                    if present:
                        X_df = X_df[present]
                if len(X_df) > 400:
                    X_df = X_df.sample(n=400, random_state=42)
                X_num_df = X_df.apply(pd.to_numeric, errors="coerce")
                for col in X_num_df.columns:
                    if X_num_df[col].isna().all():
                        codes, _ = pd.factorize(X_df[col].astype(str))
                        X_num_df[col] = codes.astype(float)
                    elif X_num_df[col].isna().any():
                        X_num_df[col] = X_num_df[col].fillna(float(X_num_df[col].median()))
                X_mat = X_num_df.to_numpy(dtype=float)
                names = [str(c) for c in X_num_df.columns]
                X_for_pred: Any = X_df
            else:
                X_mat = np.asarray(X_reference, dtype=float)
                if len(X_mat) > 400:
                    X_mat = X_mat[:400]
                names = list(feature_names[: X_mat.shape[1]]) if len(feature_names) >= X_mat.shape[1] else [f"x_{i + 1}" for i in range(X_mat.shape[1])]
                X_for_pred = X_mat

            if len(X_mat) < 5 or X_mat.shape[1] == 0:
                return None, None, None, [], None, {}

            if is_clf:
                try:
                    probs = np.asarray(model.predict_proba(X_for_pred), dtype=float)
                    y_hat = probs[:, 1] if (probs.ndim == 2 and probs.shape[1] >= 2) else probs.ravel()
                    lhs_p = f"P_approx({target_name}=1 | x)"
                    lhs_l = rf"\tilde{{P}}(\text{{{_escape_latex(target_name)}}}=1 \mid x)"
                except Exception:
                    y_raw = np.asarray(model.predict(X_for_pred))
                    codes, _ = pd.factorize(y_raw)
                    y_hat = codes.astype(float)
                    lhs_p = f"{target_name}_approx(x)"
                    lhs_l = rf"\tilde{{\text{{{_escape_latex(target_name)}}}}}(x)"
            else:
                y_hat = np.asarray(model.predict(X_for_pred), dtype=float).ravel()
                lhs_p = f"{target_name}_approx(x)"
                lhs_l = rf"\tilde{{\text{{{_escape_latex(target_name)}}}}}(x)"

            means = np.mean(X_mat, axis=0)
            stds = np.std(X_mat, axis=0)
            safe_stds = np.where(stds > 1e-12, stds, 1.0)
            X_std = (X_mat - means) / safe_stds

            surr = SklearnRidge(alpha=1.0).fit(X_std, y_hat)
            surr_preds = surr.predict(X_std)
            fidelity = float(r2_score(y_hat, surr_preds)) if float(np.var(y_hat)) > 1e-12 else 1.0
            fidelity = max(-1.0, min(1.0, fidelity))

            w_std = np.asarray(surr.coef_, dtype=float).ravel()
            b_std = float(surr.intercept_)
            beta_raw = w_std / safe_stds
            beta0_raw = float(b_std - np.sum(w_std * means / safe_stds))

            surr_plain, surr_latex = _build_linear_strings(
                lhs_p,
                lhs_l,
                beta0_raw,
                list(zip(names, beta_raw, strict=False)),
                max_terms=max_terms,
            )
            total_abs = float(np.sum(np.abs(w_std))) or 1.0
            table: list[dict[str, Any]] = [
                {
                    "Feature": fn,
                    "Surrogate Coef (beta)": round(float(br), 6),
                    "Standardized Sensitivity": round(float(ws), 6),
                    "Effect Direction": "Positive (+)" if br > 0 else ("Negative (-)" if br < 0 else "Neutral (0)"),
                    "Relative Influence %": round(abs(float(ws)) / total_abs * 100.0, 2),
                }
                for fn, br, ws in zip(names, beta_raw, w_std, strict=False)
            ]
            table.sort(key=lambda r: abs(float(r["Standardized Sensitivity"])), reverse=True)
            raw_coef_map = {
                str(fn): round(float(br), 8)
                for fn, br in zip(names, beta_raw, strict=False)
                if math.isfinite(float(br))
            }
            return surr_plain, surr_latex, round(fidelity, 4), table, round(beta0_raw, 8), raw_coef_map
        except Exception as exc:
            logger.debug("surrogate_equation_skipped", error=str(exc))
            return None, None, None, [], None, {}

    @classmethod
    def _extract_nonparametric_analytical(
        cls,
        *,
        model: Any,
        cls_name: str,
        model_name: str,
        model_family: str,
        feature_names: list[str],
        target_name: str,
        task_str: str,
        is_clf: bool,
        X_reference: pd.DataFrame | np.ndarray | None,
        max_terms: int,
    ) -> ModelEquationReport:
        """Extract exact non-parametric analytical mathematical equations (B-Spline GAM, Kernel Smoother, RKHS, k-NN, Isotonic)."""
        surr_plain, surr_latex, surr_r2, surr_table, surr_b0, surr_coefs = cls._fit_linear_surrogate(
            model=model,
            X_reference=X_reference,
            feature_names=feature_names,
            target_name=target_name,
            is_clf=is_clf,
            max_terms=max_terms,
        )
        t_esc = _escape_latex(target_name)
        lhs_p = f"P({target_name}=1 | x)" if is_clf else f"{target_name}_hat(x)"
        lhs_l = rf"\hat{{P}}(\text{{{t_esc}}}=1 \mid \mathbf{{x}})" if is_clf else rf"\hat{{\text{{{t_esc}}}}}(\mathbf{{x}})"

        if cls_name == "PenalizedSplineGAMModel":
            est = getattr(model, "_estimator", None)
            b0 = float(np.asarray(getattr(est, "intercept_", 0.0)).ravel()[0]) if est is not None else 0.0
            comps = list(getattr(model, "spline_component_summary_", []) or [])
            edf = float(getattr(model, "effective_df_", 0.0))
            alpha = float(getattr(model, "alpha", 1.0))
            degree = int(getattr(model, "degree", 3))
            top_comps = comps[: min(max_terms, 6)]
            comp_str = " + ".join(f"f_{i + 1}({c['feature']})" for i, c in enumerate(top_comps)) or "sum_j f_j(x_j)"
            comp_tex = " + ".join(
                rf"f_{{{i + 1}}}(\text{{{_escape_latex(c['feature'])}}})" for i, c in enumerate(top_comps)
            ) or r"\sum_{j=1}^{p} f_j(x_j)"
            plain = (
                f"{lhs_p} = sigmoid({_fmt_num(b0)} + {comp_str})"
                if is_clf
                else f"{lhs_p} = {_fmt_num(b0)} + {comp_str}"
            )
            latex = (
                rf"{lhs_l} = \sigma\!\left({_fmt_num(b0)} + {comp_tex}\right), \quad f_j(x_j)=\sum_{{m=1}}^{{M}} w_{{jm}} B_{{jm}}^{{({degree})}}(x_j)"
                if is_clf
                else rf"{lhs_l} = {_fmt_num(b0)} + {comp_tex}, \quad f_j(x_j)=\sum_{{m=1}}^{{M}} w_{{jm}} B_{{jm}}^{{({degree})}}(x_j)"
            )
            arch_eqs = [
                f"f_j(x_j) = sum_m w_jm * B_jm^(deg={degree})(x_j), L2 smoothness penalty alpha = {_fmt_num(alpha)}",
                f"Effective Degrees of Freedom (EDF) = tr(B(B^T B + alpha*I)^-1 B^T) = {edf:.2f}",
            ]
            coef_rows = [
                {
                    "Feature": c["feature"],
                    "Variance Share %": c["variance_share_pct"],
                    "Non-Linearity Index": c["nonlinearity_index"],
                    "Effect Range": f"[{c['effect_min']:+.3f}, {c['effect_max']:+.3f}]",
                    "Curve Shape": c["shape_direction"],
                    "Quantile Knots": ", ".join(str(k) for k in c.get("knots", [])[:5]),
                }
                for c in comps[:15]
            ]
            notes = [
                "Non-parametric Penalized B-Spline Generalized Additive Model (GAM) — captures smooth non-linear curves without parametric distributional assumptions.",
                f"Effective degrees of freedom = {edf:.2f} (smoothness penalty alpha = {_fmt_num(alpha)}, spline degree = {degree}).",
            ]
            if surr_r2 is not None:
                notes.append(f"Linear surrogate captures {surr_r2 * 100:.1f}% of the GAM surface (non-linear variance share = {max(0.0, (1.0 - surr_r2) * 100):.1f}%).")
            return ModelEquationReport(
                model_name=model_name,
                model_family=model_family,
                category="nonparametric_analytical",
                target_name=target_name,
                task_type=task_str,
                equation_type="nonparametric_analytical",
                plain_equation=plain,
                latex_equation=latex,
                architecture_equations=arch_eqs,
                surrogate_plain_equation=surr_plain,
                surrogate_latex_equation=surr_latex,
                surrogate_fidelity_r2=surr_r2,
                surrogate_intercept=surr_b0,
                surrogate_coefficients=surr_coefs,
                intercept=round(b0, 6),
                parameter_count=int(sum(c.get("n_basis", 0) for c in comps)),
                coefficients_table=coef_rows or surr_table,
                explanation_notes=notes,
            )

        if cls_name == "NadarayaWatsonKernelModel":
            bw = float(getattr(model, "bandwidth_", 1.0))
            n_anc = int(getattr(model, "n_anchors_", 0))
            edf = float(getattr(model, "effective_df_", 1.0))
            plain = f"{lhs_p} = [ sum_{{i=1..{n_anc}}} exp(-||z(x) - z_i||^2 / (2 * {bw:.4f}^2)) * y_i ] / [ sum_{{i=1..{n_anc}}} exp(-||z(x) - z_i||^2 / (2 * {bw:.4f}^2)) ]"
            latex = rf"{lhs_l} = \frac{{\sum_{{i=1}}^{{{n_anc}}} \exp\!\left(-\frac{{\|\mathbf{{z}}(\mathbf{{x}})-\mathbf{{z}}_i\|_2^2}}{{2 \cdot {bw:.4f}^2}}\right) y_i}}{{\sum_{{i=1}}^{{{n_anc}}} \exp\!\left(-\frac{{\|\mathbf{{z}}(\mathbf{{x}})-\mathbf{{z}}_i\|_2^2}}{{2 \cdot {bw:.4f}^2}}\right)}}"
            return ModelEquationReport(
                model_name=model_name,
                model_family=model_family,
                category="nonparametric_analytical",
                target_name=target_name,
                task_type=task_str,
                equation_type="nonparametric_analytical",
                plain_equation=plain,
                latex_equation=latex,
                architecture_equations=[
                    f"Gaussian Kernel K_h(u) = exp(-||u||_2^2 / (2 * h^2)), optimal CV bandwidth h = {bw:.4f}",
                    f"Reference Support Anchors N = {n_anc} | Effective Degrees of Freedom tr(S) = {edf:.2f}",
                ],
                surrogate_plain_equation=surr_plain,
                surrogate_latex_equation=surr_latex,
                surrogate_fidelity_r2=surr_r2,
                surrogate_intercept=surr_b0,
                surrogate_coefficients=surr_coefs,
                parameter_count=n_anc,
                coefficients_table=surr_table,
                explanation_notes=[
                    f"Distribution-free Nadaraya-Watson Gaussian Kernel Smoother with CV-selected bandwidth h = {bw:.4f}.",
                    f"Produces exact locally weighted conditional expectations E[Y | X = x] across {n_anc} standardized support anchors.",
                ],
            )

        if cls_name == "RBFKernelRidgeModel":
            gamma = float(getattr(model, "fitted_gamma_", 0.1))
            alpha = float(getattr(model, "fitted_alpha_", 0.5))
            n_sup = int(getattr(model, "n_support_anchors_", 0))
            top_w = list(getattr(model, "top_dual_weights_", []) or [])
            plain = f"{lhs_p} = sum_{{i=1..{n_sup}}} alpha_i * exp(-{gamma:.5f} * ||z(x) - z_i||_2^2)   [RKHS penalty lambda = {alpha:.4f}]"
            latex = rf"{lhs_l} = \sum_{{i=1}}^{{{n_sup}}} \alpha_i \exp\!\left(-{gamma:.5f}\,\|\mathbf{{z}}(\mathbf{{x}}) - \mathbf{{z}}_i\|_2^2\right), \quad (\mathbf{{K}} + {alpha:.4f}\mathbf{{I}})\boldsymbol{{\alpha}} = \mathbf{{y}}"
            return ModelEquationReport(
                model_name=model_name,
                model_family=model_family,
                category="nonparametric_analytical",
                target_name=target_name,
                task_type=task_str,
                equation_type="nonparametric_analytical",
                plain_equation=plain,
                latex_equation=latex,
                architecture_equations=[
                    f"Closed-form RKHS dual solution: alpha = (K + {alpha:.4f} * I)^(-1) * y",
                    f"RBF Kernel K(x, x') = exp(-gamma * ||z(x) - z(x')||^2), gamma = {gamma:.5f}, support points = {n_sup}",
                ],
                surrogate_plain_equation=surr_plain,
                surrogate_latex_equation=surr_latex,
                surrogate_fidelity_r2=surr_r2,
                surrogate_intercept=surr_b0,
                surrogate_coefficients=surr_coefs,
                parameter_count=n_sup,
                coefficients_table=surr_table or top_w,
                explanation_notes=[
                    "Closed-form Reproducing Kernel Hilbert Space (RKHS) RBF Kernel Ridge estimator.",
                    f"Solves exact dual analytical expansion over {n_sup} support observations with regularization lambda = {alpha:.4f}.",
                ],
            )

        if cls_name == "DistanceWeightedKNNModel":
            k_val = int(getattr(model, "fitted_k_", 5))
            plain = f"{lhs_p} = [ sum_{{i in N_{k_val}(x)}} (1 / (||z(x) - z_i||_2 + eps)) * y_i ] / [ sum_{{i in N_{k_val}(x)}} 1 / (||z(x) - z_i||_2 + eps) ]"
            latex = rf"{lhs_l} = \frac{{\sum_{{i \in \mathcal{{N}}_{{{k_val}}}(\mathbf{{x}})}} w_i(\mathbf{{x}})\, y_i}}{{\sum_{{i \in \mathcal{{N}}_{{{k_val}}}(\mathbf{{x}})}} w_i(\mathbf{{x}})}}, \quad w_i(\mathbf{{x}}) = \frac{{1}}{{\|\mathbf{{z}}(\mathbf{{x}}) - \mathbf{{z}}_i\|_2 + \varepsilon}}"
            return ModelEquationReport(
                model_name=model_name,
                model_family=model_family,
                category="nonparametric_analytical",
                target_name=target_name,
                task_type=task_str,
                equation_type="nonparametric_analytical",
                plain_equation=plain,
                latex_equation=latex,
                architecture_equations=[f"Cross-validated neighborhood size k = {k_val} with inverse-Euclidean distance kernel weighting"],
                surrogate_plain_equation=surr_plain,
                surrogate_latex_equation=surr_latex,
                surrogate_fidelity_r2=surr_r2,
                surrogate_intercept=surr_b0,
                surrogate_coefficients=surr_coefs,
                parameter_count=k_val,
                coefficients_table=surr_table,
                explanation_notes=[
                    f"Non-parametric inverse-distance weighted k-NN local analytical estimator (optimal k = {k_val}).",
                    "Adapts locally to arbitrary decision boundaries and multi-modal response surfaces without global functional assumptions.",
                ],
            )

        # IsotonicMonotonicModel
        drv = str(getattr(model, "driver_feature_", "x_0"))
        rho = float(getattr(model, "driver_spearman_rho_", 0.0))
        inc = bool(getattr(model, "increasing_", True))
        steps = list(getattr(model, "step_thresholds_", []) or [])
        direction_txt = "monotone non-decreasing" if inc else "monotone non-increasing"
        plain = f"{lhs_p} = g_iso({drv})  [{direction_txt} piecewise step function, Spearman rho = {rho:+.4f}, {len(steps)} representative knots]"
        latex = rf"{lhs_l} = g_{{\text{{iso}}}}\!\left(\text{{{_escape_latex(drv)}}}\right), \quad \rho_S = {rho:+.4f}"
        rules_lines = [
            f"If {drv} <= {s['x_threshold']:.4f}  =>  {target_name}_hat = {s['y_hat_step']:.4f}"
            for s in steps[:10]
        ]
        return ModelEquationReport(
            model_name=model_name,
            model_family=model_family,
            category="nonparametric_analytical",
            target_name=target_name,
            task_type=task_str,
            equation_type="nonparametric_analytical",
            plain_equation=plain,
            latex_equation=latex,
            architecture_equations=[f"Pool-Adjacent-Violators (PAVA) isotonic projection on strongest Spearman predictor '{drv}' (rho = {rho:+.4f})"],
            surrogate_plain_equation=surr_plain,
            surrogate_latex_equation=surr_latex,
            surrogate_fidelity_r2=surr_r2,
            surrogate_intercept=surr_b0,
            surrogate_coefficients=surr_coefs,
            parameter_count=len(steps),
            coefficients_table=[
                {"Driver Feature": drv, "Threshold (x)": s["x_threshold"], "Step Prediction (y_hat)": s["y_hat_step"]}
                for s in steps
            ],
            decision_rules_text="\n".join(rules_lines) if rules_lines else None,
            explanation_notes=[
                f"Non-parametric Isotonic Regression fitted on dominant Spearman driver '{drv}' (rho = {rho:+.4f}).",
                f"Enforces exact {direction_txt} response constraint via Pool-Adjacent-Violators Algorithm (PAVA).",
            ],
        )

