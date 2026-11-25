"""End-to-End Quickstart Example demonstrating the full OmniForge lifecycle."""

from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import train_test_split

import omniforge
from omniforge.mlops.fairness_evidence import FairnessEvidence


def main() -> None:
    omniforge.set_global_seed(42)
    out_dir = Path("./.omniforge_workspace/quickstart_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic dataset with missingness & protected group attribute
    df = omniforge.SyntheticDataGenerator(random_state=42).generate_classification_dataset(
        n_samples=320, n_features=8, missing_rate=0.05
    )

    # 2. Data Quality & Cleaning
    quality = omniforge.DataQualityProfiler().profile(df)
    print(f"[Data Quality] Score: {quality.overall_score}/100 (Grade {quality.grade})")
    development, holdout = train_test_split(df, test_size=0.20, stratify=df["target"], random_state=42)
    df_clean, holdout_clean, clean_report, _ = omniforge.DataCleaner().fit_transform_split(development, holdout)
    print(f"[Cleaning] Missing values: {clean_report.missing_values_before} -> {clean_report.missing_values_after}")

    # 3. Automated EDA
    eda = omniforge.AutomatedEDA().run(df_clean, target_column="target", generate_figures=True)
    eda.export_html(out_dir / "eda_report.html")

    # 4. AutoML with mandatory strong baseline comparison & carbon tracking
    X = df_clean.drop(columns=["target", "group_attr"])
    y = df_clean["target"]
    X_test = holdout_clean.drop(columns=["target", "group_attr"])
    y_test = holdout.loc[X_test.index, "target"]
    groups_test = holdout.loc[X_test.index, "group_attr"]
    automl = omniforge.AutoMLPipelineSearch(time_budget_seconds=25.0, tune_trials=4).fit(X, y, holdout=(X_test, y_test))
    print(
        f"[AutoML] Champion: {automl.best_model_name} | "
        f"{automl.primary_metric}={automl.best_score:.4f} | "
        f"Lift over best baseline={automl.lift_over_best_baseline:+.4f}"
    )

    # 5. Responsible AI: SHAP + Fairness + Model Card
    shap_local = omniforge.InterpretabilityEngine().explain_shap_local(automl.best_model, X, X.iloc[0], n_samples=25)
    print(f"[SHAP] Top local drivers: {list(shap_local['attributions'].items())[:3]}")
    fairness = omniforge.FairnessAuditor().audit(
        y_test, automl.best_model.predict(X_test), groups_test, attribute_name="group_attr"
    )
    print(f"[Fairness] Passed={fairness.passed} | Disparate Impact={fairness.disparate_impact_ratio}")

    # 6. MLOps Registration & Serving
    registry = omniforge.ModelRegistry(db_path=out_dir / "registry.db", artifact_dir=out_dir / "reg_artifacts")
    evidence = FairnessEvidence.capture(automl.best_model, X_test, y_test, groups_test, fairness)
    version = registry.register_model(
        automl.best_model, "quickstart_classifier", fairness_evidence=evidence, stage=omniforge.core.ModelStage.STAGING
    )
    if fairness.passed:
        version = registry.transition_stage(
            version.model_name, version.version, omniforge.core.ModelStage.PRODUCTION, require_fairness_gate=True
        )
    serving = omniforge.ModelServingEngine(registry=registry)
    serving.deploy_from_registry("scoring_v1", version.model_name, stage=version.stage)
    pred_out = serving.predict("scoring_v1", X_test.head(3), return_probabilities=True)
    print(f"[Serving {version.stage.value}] Predictions: {pred_out['predictions']} (latency={pred_out['latency_ms']} ms)")

    # 7. Multi-format Executive Reporting
    reporter = omniforge.ReportGenerator()
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
