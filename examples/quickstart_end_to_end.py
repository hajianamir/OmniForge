"""End-to-End Quickstart Example demonstrating the full OmniForge lifecycle."""

from __future__ import annotations

from pathlib import Path

import omniforge


def main() -> None:
    uap.set_global_seed(42)
    out_dir = Path("./.omniforge_workspace/quickstart_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic dataset with missingness & protected group attribute
    df = uap.SyntheticDataGenerator(random_state=42).generate_classification_dataset(
        n_samples=320, n_features=8, missing_rate=0.05
    )

    # 2. Data Quality & Cleaning
    quality = uap.DataQualityProfiler().profile(df)
    print(f"[Data Quality] Score: {quality.overall_score}/100 (Grade {quality.grade})")
    df_clean, clean_report = uap.DataCleaner().fit_transform(df)
    print(f"[Cleaning] Missing values: {clean_report.missing_values_before} -> {clean_report.missing_values_after}")

    # 3. Automated EDA
    eda = uap.AutomatedEDA().run(df_clean, target_column="target", generate_figures=True)
    eda.export_html(out_dir / "eda_report.html")

    # 4. AutoML with mandatory strong baseline comparison & carbon tracking
    X = df_clean.drop(columns=["target"])
    y = df_clean["target"]
    automl = uap.AutoMLPipelineSearch(time_budget_seconds=25.0, tune_trials=4).fit(X, y)
    print(
        f"[AutoML] Champion: {automl.best_model_name} | "
        f"{automl.primary_metric}={automl.best_score:.4f} | "
        f"Lift over best baseline={automl.lift_over_best_baseline:+.4f}"
    )

    # 5. Responsible AI: SHAP + Fairness + Model Card
    shap_local = uap.InterpretabilityEngine().explain_shap_local(automl.best_model, X, X.iloc[0], n_samples=25)
    print(f"[SHAP] Top local drivers: {list(shap_local['attributions'].items())[:3]}")
    fairness = uap.FairnessAuditor().audit(y, automl.best_model.predict(X), df_clean["group_attr"], attribute_name="group_attr")
    print(f"[Fairness] Passed={fairness.passed} | Disparate Impact={fairness.disparate_impact_ratio}")

    # 6. MLOps Registration & Serving
    registry = uap.ModelRegistry(db_path=out_dir / "registry.db", artifact_dir=out_dir / "reg_artifacts")
    ver = registry.register_model(automl.best_model, "quickstart_classifier", stage=uap.core.ModelStage.PRODUCTION)
    serving = uap.ModelServingEngine(registry=registry)
    serving.deploy_from_registry("scoring_v1", "quickstart_classifier")
    pred_out = serving.predict("scoring_v1", X.head(3), return_probabilities=True)
    print(f"[Serving] Endpoint predictions: {pred_out['predictions']} (latency={pred_out['latency_ms']} ms)")

    # 7. Multi-format Executive Reporting
    reporter = uap.ReportGenerator()
    payload = reporter.build_report(
        title="OmniForge Quickstart Executive Analytics Report",
        dataset_summary={"n_rows": len(df_clean), "n_columns": len(df_clean.columns)},
        quality_report=quality.model_dump(),
        model_summary={
            "best_model_name": automl.best_model_name,
            "primary_metric": automl.primary_metric,
            "best_score": automl.best_score,
            "lift_over_best_baseline": automl.lift_over_best_baseline,
        },
        fairness_report=fairness.model_dump(),
        carbon_report=automl.carbon_report.model_dump() if automl.carbon_report else None,
        tables={"Leaderboard": automl.leaderboard},
    )
    for ext in ("html", "md", "json", "xlsx", "docx", "pptx", "pdf"):
        reporter.export(payload, out_dir / f"executive_report.{ext}")
    print(f"[Reporting] Exported 7 report formats to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
