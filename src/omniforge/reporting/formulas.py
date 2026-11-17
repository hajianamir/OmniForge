"""Comprehensive Catalog of Mathematical Relationships, Model Equations, Computational Rules & Explanations."""

from __future__ import annotations

from typing import Any

from omniforge.reporting.sections import FormulaRuleRecord


class FormulaRulesCatalog:
    """Compiles analytical formulas, computational rules, and fitted model equations with detailed explanations."""

    @staticmethod
    def build_catalog(
        *,
        cleaning_report: dict[str, Any] | None = None,
        quality_report: dict[str, Any] | None = None,
        diagnostics_report: Any | None = None,
        clustering_result: Any | None = None,
        anomaly_meta: dict[str, Any] | None = None,
        eda_report: Any | None = None,
        feature_selection_method: str | None = None,
        model_equations: dict[str, dict[str, dict[str, Any]]] | None = None,
        automl_result: Any | None = None,
        future_prediction_result: Any | None = None,
        fairness_report: dict[str, Any] | None = None,
        causal_result: Any | None = None,
        forecast_report: Any | None = None,
        carbon_report: dict[str, Any] | None = None,
        include_core_methodology: bool = True,
    ) -> list[FormulaRuleRecord]:
        """Collect all applicable methodological formulas and fitted model equations with explanations."""
        records: list[FormulaRuleRecord] = []

        # 1. Data Quality & Cleaning Computational Rules
        if quality_report is not None or include_core_methodology:
            records.append(
                FormulaRuleRecord(
                    category="Data Quality & Profiling",
                    name="6-Dimensional Enterprise Data Quality Index (DQI)",
                    mathematical_expression=(
                        "DQI = 0.25*Completeness + 0.20*Uniqueness + 0.20*Validity "
                        "+ 0.15*Consistency + 0.10*Accuracy + 0.10*Timeliness"
                    ),
                    variables_and_parameters=(
                        "Completeness = 100*(1 - missing_cells / total_cells); "
                        "Uniqueness = 100*(1 - duplicate_rows / n_rows); "
                        "Validity, Consistency, Accuracy, Timeliness in [0, 100]."
                    ),
                    computational_rule=(
                        "Grades: A+ (>=95), A (>=90), B (>=80), C (>=70), D (>=60), F (<60). "
                        "Datasets scoring below 80 require cleaning and anomaly screening prior to modeling."
                    ),
                    detailed_explanation=(
                        "Provides a single auditable governance score summarizing structural reliability across six "
                        "orthogonal dimensions so teams can quantify data-health improvements before and after cleaning."
                    ),
                    scope="Studio 01 · Data Quality",
                )
            )

        out_method = str((cleaning_report or {}).get("outlier_method", "")).lower()
        out_action = str((cleaning_report or {}).get("outlier_action", "clip")).lower()
        if cleaning_report is not None or include_core_methodology:
            if out_method in ("", "iqr", "none"):
                records.append(
                    FormulaRuleRecord(
                        category="Data Cleaning & Outlier Exclusion Rule",
                        name="Tukey Interquartile Range (IQR) Outlier Fence",
                        mathematical_expression=(
                            "IQR = Q3 - Q1;  Lower_Fence = Q1 - k*IQR;  Upper_Fence = Q3 + k*IQR  (default k = 1.5)"
                        ),
                        variables_and_parameters=(
                            "Q1 = 25th percentile, Q3 = 75th percentile, k = 1.5 (standard outlier) or 3.0 (extreme outlier)."
                        ),
                        computational_rule=(
                            f"Flag observation x if x < Lower_Fence OR x > Upper_Fence. "
                            f"Configured action: '{out_action}' (clip winsorizes values to fence bounds; "
                            "drop excludes the full record; flag preserves data and adds an indicator)."
                        ),
                        detailed_explanation=(
                            "Unlike mean/standard-deviation rules, Tukey IQR fences rely on rank quantiles and therefore "
                            "resist distortion by extreme sensor glitches or fat-tailed outliers. Target columns are "
                            "optionally protected from clipping to preserve genuine extreme events."
                        ),
                        scope="Studio 01 · Data Cleaning",
                    )
                )
            if out_method == "zscore" or include_core_methodology:
                records.append(
                    FormulaRuleRecord(
                        category="Data Cleaning & Outlier Exclusion Rule",
                        name="Standardized Z-Score & Modified MAD Z-Score",
                        mathematical_expression=(
                            "z_i = (x_i - mu) / sigma;   Robust M_i = 0.6745 * (x_i - median(x)) / MAD"
                        ),
                        variables_and_parameters=(
                            "mu = sample mean, sigma = sample standard deviation, "
                            "MAD = median(|x_i - median(x)|), threshold tau = 3.0 (standard) or 3.5 (robust)."
                        ),
                        computational_rule=(
                            "An observation is flagged as an anomaly whenever |z_i| > tau (or |M_i| > 3.5 for skewed data)."
                        ),
                        detailed_explanation=(
                            "Measures how many standard deviations (or scaled median absolute deviations) a value lies "
                            "away from the center. Under Gaussian normality, |z| > 3.0 corresponds to roughly 0.27% tail probability."
                        ),
                        scope="Studio 01 / Studio 02 · Outlier Screening",
                    )
                )
            if cleaning_report is not None:
                imp_num = str(cleaning_report.get("numeric_imputation", "median"))
                imp_cat = str(cleaning_report.get("categorical_imputation", "mode"))
                records.append(
                    FormulaRuleRecord(
                        category="Data Cleaning & Imputation Rule",
                        name="Missing-Value Imputation & Deduplication Rule",
                        mathematical_expression=(
                            f"x_missing_num <- {imp_num}(x_observed);   x_missing_cat <- {imp_cat}(x_observed);   "
                            "Duplicate(r_i, r_j) iff all columns match exactly"
                        ),
                        variables_and_parameters=(
                            f"Numeric strategy = '{imp_num}', Categorical strategy = '{imp_cat}', "
                            "Exact duplicates removed keeping the first chronological occurrence."
                        ),
                        computational_rule=(
                            "Imputation parameters are fit on observed non-null cells and applied deterministically. "
                            "Exact duplicate rows are dropped to prevent data leakage and inflated sample significance."
                        ),
                        detailed_explanation=(
                            "Median imputation preserves the central tendency under skewness and extreme outliers without "
                            "shrinking valid row count, while deduplication prevents identical records from appearing in "
                            "both train and validation folds."
                        ),
                        scope="Studio 01 · Data Cleaning",
                    )
                )

        # 2. Clustering & Anomaly Screening Formulas
        if clustering_result is not None or anomaly_meta is not None:
            records.append(
                FormulaRuleRecord(
                    category="Clustering & Subpopulation Discovery",
                    name="Cluster Cohesion & Separation Metrics (Silhouette, DBI, Calinski-Harabasz)",
                    mathematical_expression=(
                        "s(i) = (b(i) - a(i)) / max(a(i), b(i));   "
                        "DBI = (1/K) * sum_i max_{j!=i} ((sigma_i + sigma_j) / d(c_i, c_j))"
                    ),
                    variables_and_parameters=(
                        "a(i) = mean intra-cluster distance for point i; b(i) = mean distance to nearest neighboring cluster; "
                        "sigma_i = average distance of points in cluster i to centroid c_i; K = number of clusters."
                    ),
                    computational_rule=(
                        "Higher Silhouette (closer to +1.0) and higher Calinski-Harabasz indicate tighter, well-separated "
                        "clusters; lower Davies-Bouldin Index (closer to 0) indicates less cluster overlap. "
                        "Density algorithms (DBSCAN/HDBSCAN) assign sparse points to noise label -1."
                    ),
                    detailed_explanation=(
                        "Validates whether discovered subpopulations reflect genuine structural regimes vs. random noise. "
                        "When noise points (-1) or confirmed corrupt sensor clusters are removed, downstream predictive models "
                        "learn cleaner decision boundaries without overfitting to defective observations."
                    ),
                    scope="Studio 02 · Clustering & Anomaly Screening",
                )
            )
            records.append(
                FormulaRuleRecord(
                    category="Multivariate Anomaly Screening",
                    name="Isolation Forest Anomaly Score & LOF Density Ratio",
                    mathematical_expression=(
                        "s(x, n) = 2^(-E[h(x)] / c(n));   "
                        "LOF_k(x) = (1/|N_k(x)|) * sum_{o in N_k(x)} (lrd_k(o) / lrd_k(x))"
                    ),
                    variables_and_parameters=(
                        "E[h(x)] = average path length to isolate point x across random partition trees; "
                        "c(n) = expected path length of unsuccessful BST search in n samples; "
                        "lrd_k(x) = local reachability density across k nearest neighbors."
                    ),
                    computational_rule=(
                        "Points isolated in very few random splits have s(x, n) -> 1.0 (strong global anomaly). "
                        "Points with LOF_k(x) >> 1.0 occupy a substantially sparser neighborhood than their peers "
                        "and are flagged as local multivariate anomalies."
                    ),
                    detailed_explanation=(
                        "Captures multi-column anomalies that look normal in every individual column (passing univariate IQR) "
                        "yet violate joint physical or behavioral relationships across features simultaneously."
                    ),
                    scope="Studio 02 · Anomaly Screening",
                )
            )

        # 3. Statistical Diagnostics & Hypothesis Rules
        if diagnostics_report is not None or eda_report is not None or include_core_methodology:
            alpha = 0.05
            if diagnostics_report is not None:
                crit = getattr(diagnostics_report, "criteria", None)
                if crit is not None:
                    alpha = float(getattr(crit, "alpha", 0.05))
            records.append(
                FormulaRuleRecord(
                    category="Statistical Inference & Diagnostics",
                    name="Hypothesis Testing Decision Rule, p-Value & Effect Size",
                    mathematical_expression=(
                        "Reject H0 iff p_value < alpha (alpha = "
                        f"{alpha:.4f});   "
                        "Cohen_d = (mu_1 - mu_2) / s_pooled;   "
                        "r_xy = cov(X, Y) / (s_X * s_Y)"
                    ),
                    variables_and_parameters=(
                        f"alpha = Type-I error significance threshold ({alpha}); "
                        "s_pooled = pooled standard deviation; r_xy in [-1, +1] = Pearson linear correlation; "
                        "Spearman rho = Pearson r computed on rank-transformed observations."
                    ),
                    computational_rule=(
                        f"A statistical finding is declared significant when p < {alpha:.4f} (adjusted via Benjamini-Hochberg "
                        "FDR or Bonferroni when multiple tests are executed). Practical significance requires evaluating "
                        "effect size (|d| >= 0.2 small, >= 0.5 medium, >= 0.8 large) alongside p-values."
                    ),
                    detailed_explanation=(
                        "Separates reproducible statistical signals from sampling noise. Combining p-values with effect size "
                        "prevents over-interpreting tiny, practically irrelevant differences on large datasets."
                    ),
                    scope="Studio 03 & 04 · Diagnostics & Hypothesis Lab",
                )
            )
            records.append(
                FormulaRuleRecord(
                    category="Multicollinearity & Distribution Diagnostics",
                    name="Variance Inflation Factor (VIF) & Higher-Order Moments",
                    mathematical_expression=(
                        "VIF_j = 1 / (1 - R_j^2);   "
                        "Skewness = E[((X - mu)/sigma)^3];   "
                        "Excess_Kurtosis = E[((X - mu)/sigma)^4] - 3"
                    ),
                    variables_and_parameters=(
                        "R_j^2 = coefficient of determination obtained by regressing predictor X_j on all other predictors."
                    ),
                    computational_rule=(
                        "VIF > 5 signals moderate multicollinearity; VIF > 10 signals severe collinearity that inflates "
                        "linear coefficient variance and destabilizes OLS equations. |Skewness| > 1 flags strong asymmetry; "
                        "Excess Kurtosis > 3 indicates heavy fat tails."
                    ),
                    detailed_explanation=(
                        "Guides whether regularized models (Ridge/Lasso), PCA, or tree/neural architectures should replace "
                        "unregularized linear models, and whether log/Yeo-Johnson transformations are warranted."
                    ),
                    scope="Studio 03 & 04 · Diagnostics & EDA",
                )
            )

        # 4. Feature Selection Rule
        if feature_selection_method:
            records.append(
                FormulaRuleRecord(
                    category="Feature Selection & Pruning Rule",
                    name=f"Feature Ranking & Retention Criterion ({feature_selection_method})",
                    mathematical_expression=(
                        "I(X_j; Y) = sum_{x,y} p(x,y) * log(p(x,y) / (p(x)*p(y)))   [Mutual Information / Importance Rank]"
                    ),
                    variables_and_parameters=(
                        "X_j = candidate predictor feature j; Y = target variable; max_features = user retention budget."
                    ),
                    computational_rule=(
                        f"Features are scored via '{feature_selection_method}', sorted in descending order of predictive "
                        "signal, and only the top-ranked predictors (plus protected target columns) are retained; "
                        "low-scoring redundant or noise predictors are excluded."
                    ),
                    detailed_explanation=(
                        "Eliminating zero-information and highly redundant predictors reduces overfitting, lowers variance, "
                        "accelerates training, and simplifies final model equations."
                    ),
                    scope="Studio 05 · Feature Engineering & Selection",
                )
            )

        # 5. Extracted Fitted Model Equations (EXACT equations from ModelEquationExtractor!)
        if model_equations:
            for target_name, eq_map in model_equations.items():
                if not isinstance(eq_map, dict):
                    continue
                for model_name, eq_dict in eq_map.items():
                    if not isinstance(eq_dict, dict):
                        continue
                    cat = str(eq_dict.get("category", "Trained Model"))
                    eq_type = str(eq_dict.get("equation_type", "exact"))
                    plain_eq = str(eq_dict.get("plain_equation") or "").strip()
                    surr_eq = str(eq_dict.get("surrogate_plain_equation") or "").strip()
                    surr_r2 = eq_dict.get("surrogate_fidelity_r2")
                    param_cnt = eq_dict.get("parameter_count")
                    notes = str(eq_dict.get("notes") or "").strip()
                    arch = str(eq_dict.get("architecture_summary") or "").strip()
                    terms = eq_dict.get("terms") or []
                    rules = eq_dict.get("decision_rules") or []

                    expr_parts: list[str] = []
                    if plain_eq:
                        expr_parts.append(plain_eq)
                    if surr_eq:
                        fid = f" (Surrogate Fidelity R^2 = {float(surr_r2):.4f})" if isinstance(surr_r2, (int, float)) else ""
                        expr_parts.append(f"Linear Sensitivity Approximation{fid}: {surr_eq}")
                    if rules:
                        expr_parts.append("Top Decision Rules: " + " || ".join(str(r) for r in rules[:3]))

                    # Build clear variable / driver summary from extracted terms
                    top_terms = [t for t in terms if isinstance(t, dict)][:8]
                    if top_terms:
                        var_desc = "; ".join(
                            f"{t.get('feature', '?')} (coef={float(t.get('coefficient', 0.0)):+.4f}, "
                            f"{'increases' if float(t.get('coefficient', 0.0)) >= 0 else 'decreases'} prediction)"
                            for t in top_terms
                        )
                    else:
                        var_desc = (
                            f"Target = '{target_name}'; Parameters = {param_cnt if param_cnt is not None else 'N/A'}"
                            + (f"; Architecture = {arch}" if arch else "")
                        )

                    # Build detailed plain-English interpretation
                    expl_bits = [
                        f"Fitted {cat} model '{model_name}' predicting target '{target_name}' "
                        f"(representation type: {eq_type}, parameters: {param_cnt if param_cnt is not None else 'N/A'})."
                    ]
                    if top_terms:
                        pos = [t["feature"] for t in top_terms if float(t.get("coefficient", 0.0)) > 0][:3]
                        neg = [t["feature"] for t in top_terms if float(t.get("coefficient", 0.0)) < 0][:3]
                        if pos:
                            expl_bits.append(f"Primary positive drivers (higher value raises predicted {target_name}): {', '.join(pos)}.")
                        if neg:
                            expl_bits.append(f"Primary negative drivers (higher value lowers predicted {target_name}): {', '.join(neg)}.")
                    if isinstance(surr_r2, (int, float)):
                        expl_bits.append(
                            f"The global linear sensitivity surrogate explains {float(surr_r2) * 100:.1f}% of this model's "
                            "prediction variance, revealing how nonlinear interactions compare to additive linear effects."
                        )
                    if notes:
                        expl_bits.append(notes)

                    rule_str = (
                        "Evaluates the exact closed-form equation on preprocessed numeric features."
                        if "exact" in eq_type.lower() and "neural" not in eq_type.lower()
                        else (
                            "Propagates standardized inputs through affine weight layers and nonlinear activations "
                            "h^(l) = sigma(W^(l)*h^(l-1) + b^(l)) to produce the final prediction."
                            if "neural" in eq_type.lower() or "deep" in cat.lower()
                            else "Aggregates conditional IF-THEN partition splits across base trees (bagging average or boosted additive residual sum)."
                        )
                    )

                    records.append(
                        FormulaRuleRecord(
                            category=f"Fitted Model Equation ({cat})",
                            name=f"{model_name} [Target: {target_name}]",
                            mathematical_expression="\n".join(expr_parts) or "Model equation extracted via empirical surrogate.",
                            variables_and_parameters=var_desc,
                            computational_rule=rule_str,
                            detailed_explanation=" ".join(expl_bits),
                            scope=f"Target: {target_name}",
                        )
                    )

        # 6. Model Evaluation & Baseline Lift Formulas
        if automl_result is not None or model_equations or include_core_methodology:
            records.append(
                FormulaRuleRecord(
                    category="Model Evaluation & Baseline Benchmark Rules",
                    name="Predictive Performance & Lift Over Strongest Baseline",
                    mathematical_expression=(
                        "R^2 = 1 - sum((y_i - y_hat_i)^2) / sum((y_i - y_bar)^2);   "
                        "RMSE = sqrt((1/n) * sum((y_i - y_hat_i)^2));   "
                        "F1 = 2 * (Precision * Recall) / (Precision + Recall);   "
                        "Lift = Score(Champion) - Score(Best_Baseline)"
                    ),
                    variables_and_parameters=(
                        "y_i = observed ground truth, y_hat_i = predicted target, y_bar = mean target; "
                        "Precision = TP/(TP+FP), Recall = TP/(TP+FN)."
                    ),
                    computational_rule=(
                        "Every advanced model (Classical, Ensemble, AutoML, Deep Learning) is benchmarked on identical "
                        "cross-validation / holdout splits against naive and linear baselines. Positive Lift verifies that "
                        "added model complexity genuinely improves generalization."
                    ),
                    detailed_explanation=(
                        "Prevents deploying complex black-box models when simpler interpretable baselines match or beat "
                        "them. R^2 measures proportion of variance explained (regression); F1 and ROC-AUC measure "
                        "discrimination quality under class imbalance (classification)."
                    ),
                    scope="Studio 03 & 06 · Baselines & Modeling",
                )
            )

        # 7. Future Target Prediction & OOD Shift Rule
        if future_prediction_result is not None:
            records.append(
                FormulaRuleRecord(
                    category="Future Scoring & Out-of-Distribution (OOD) Guardrail",
                    name="Feature-Level Standardized Shift (OOD) Rule",
                    mathematical_expression=(
                        "z_OOD(x_j) = |x_j - mu_train,j| / sigma_train,j;   "
                        "OOD_Score(x) = max_j z_OOD(x_j)"
                    ),
                    variables_and_parameters=(
                        "mu_train,j and sigma_train,j = mean and standard deviation of feature j measured on the training set."
                    ),
                    computational_rule=(
                        "Incoming future records with OOD_Score > 3.0 are flagged as Out-of-Distribution (Extrapolation Risk), "
                        "warning decision-makers that predictions lie outside the historical training support."
                    ),
                    detailed_explanation=(
                        "Machine learning models can produce overconfident extrapolations on unseen extreme inputs. "
                        "Flagging OOD records alongside nearest-cluster assignment protects operational decisions."
                    ),
                    scope="Studio 06 · Future Target Prediction",
                )
            )

        # 8. Risk Assessment (VaR / CVaR)
        if diagnostics_report is not None and getattr(diagnostics_report, "risk_assessment", None):
            records.append(
                FormulaRuleRecord(
                    category="Risk Analytics & Tail Stress",
                    name="Value-at-Risk (VaR), Conditional VaR (CVaR / Expected Shortfall) & Structural Risk Index",
                    mathematical_expression=(
                        "VaR_alpha = Quantile_alpha(L);   "
                        "CVaR_alpha = E[L | L >= VaR_alpha]"
                    ),
                    variables_and_parameters=(
                        "L = adverse tail distribution loss/metric; alpha = tail confidence level (e.g., 95% or 99%)."
                    ),
                    computational_rule=(
                        "VaR bounds worst-case outcomes at confidence alpha; CVaR averages severity strictly beyond VaR "
                        "to capture fat-tail catastrophe exposure."
                    ),
                    detailed_explanation=(
                        "Essential for heavy-tailed engineering, clinical, and financial series where Gaussian variance "
                        "severely underestimates extreme failure or loss events."
                    ),
                    scope="Studio 03 · Risk Analytics",
                )
            )

        # 9. Responsible AI Fairness & Green AI
        if fairness_report is not None:
            records.append(
                FormulaRuleRecord(
                    category="Responsible AI & Fairness Governance",
                    name="Disparate Impact Ratio (4/5ths Rule) & Equalized Odds Difference",
                    mathematical_expression=(
                        "DI = P(Y_hat = 1 | G = unprivileged) / P(Y_hat = 1 | G = privileged);   "
                        "EOD = max(|TPR_u - TPR_p|, |FPR_u - FPR_p|)"
                    ),
                    variables_and_parameters=(
                        "G = protected group attribute; TPR = True Positive Rate; FPR = False Positive Rate."
                    ),
                    computational_rule=(
                        "Fairness gate passes when DI in [0.80, 1.25] (regulatory 4/5ths rule) and EOD <= 0.10."
                    ),
                    detailed_explanation=(
                        "Quantifies whether automated decisions systematically disadvantage protected subpopulations in "
                        "either selection rates (DI) or error rates (EOD)."
                    ),
                    scope="Studio 07 · Responsible AI",
                )
            )

        if causal_result is not None:
            records.append(
                FormulaRuleRecord(
                    category="Causal Inference",
                    name="Average Treatment Effect (ATE) via Potential Outcomes",
                    mathematical_expression="ATE = E[Y(1) - Y(0)] = E_X[ E[Y | T=1, X] - E[Y | T=0, X] ]",
                    variables_and_parameters="T in {0, 1} = treatment assignment; Y(1), Y(0) = potential outcomes; X = confounders.",
                    computational_rule=(
                        "Adjusts for observed confounders X (via IPW, Doubly Robust AIPW, or Outcome Regression) "
                        "to isolate causal intervention impact rather than raw correlation."
                    ),
                    detailed_explanation=(
                        "Estimates what would change if an intervention were actively applied across the population, "
                        "controlling for selection bias."
                    ),
                    scope="Studio 08 · Causal Inference",
                )
            )

        if forecast_report is not None:
            records.append(
                FormulaRuleRecord(
                    category="Time-Series Forecasting",
                    name="Multi-Horizon Forecast Error & Confidence Bands",
                    mathematical_expression=(
                        "MAPE = (100/h) * sum(|(y_t - y_hat_t) / y_t|);   "
                        "PI_95(h) = y_hat_{T+h} +/- 1.96 * sigma_e * sqrt(h)"
                    ),
                    variables_and_parameters="h = forecast step ahead; sigma_e = one-step residual standard deviation.",
                    computational_rule="Models compete on chronological holdout splits without shuffling to prevent temporal leakage.",
                    detailed_explanation="Uncertainty intervals widen with forecast horizon h as compounding errors accumulate.",
                    scope="Studio 08 · Forecasting",
                )
            )

        if carbon_report is not None:
            records.append(
                FormulaRuleRecord(
                    category="Green AI & Sustainability",
                    name="Compute Carbon Footprint & Green AI Score",
                    mathematical_expression="Emissions_gCO2eq = Energy_kWh * Grid_Carbon_Intensity(gCO2eq / kWh)",
                    variables_and_parameters="Energy_kWh = measured CPU/RAM/GPU energy draw; Grid_Carbon_Intensity = regional electricity mix.",
                    computational_rule="Tracks energy consumed during model training and penalizes high-emission runs with minimal accuracy lift.",
                    detailed_explanation="Enables transparent accuracy-versus-sustainability trade-off governance.",
                    scope="Studio 06 & 07 · Green AI",
                )
            )

        return records
