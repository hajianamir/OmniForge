"""Integration tests for Evaluation, Responsible AI, AutoML, DAG Pipelines, MLOps, Copilot, Reporting & 18 Domains."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from omniforge.automl.nas import NeuralArchitectureSearcher
from omniforge.automl.optimizer import HyperparameterOptimizer
from omniforge.automl.pipeline_search import AutoMLPipelineSearch
from omniforge.copilot.agent import OmniForgeCopilotAgent
from omniforge.core.types import ModelStage, TaskType
from omniforge.data.cleaning import DataCleaner
from omniforge.domains import default_domain_registry
from omniforge.evaluation.backtesting import BacktestingEngine
from omniforge.evaluation.fairness import FairnessAuditor
from omniforge.evaluation.interpretability import InterpretabilityEngine
from omniforge.evaluation.metrics import MetricsEvaluator
from omniforge.evaluation.model_cards import ModelCardGenerator
from omniforge.evaluation.robustness import ConformalPredictor, RobustnessTester
from omniforge.evaluation.validation import CombinatorialPurgedKFold, CrossValidator
from omniforge.mlops.continuous_training import ContinuousTrainingOrchestrator
from omniforge.mlops.drift import DriftDetector
from omniforge.mlops.monitoring import ModelMonitor
from omniforge.mlops.registry import ModelRegistry
from omniforge.mlops.serving import ModelServingEngine
from omniforge.mlops.tracking import ExperimentTracker
from omniforge.models.classical import ClassicalMLModel
from omniforge.pipelines.builder import PipelineBuilder
from omniforge.pipelines.executor import PipelineExecutor
from omniforge.reporting.exporters import ReportExporters
from omniforge.reporting.generator import ReportGenerator


def test_evaluation_and_responsible_ai(classification_df: pd.DataFrame) -> None:
    df, _ = DataCleaner().fit_transform(classification_df)
    X, y = df.drop(columns=["target"]), df["target"]
    model = ClassicalMLModel(algorithm="random_forest", task_type=TaskType.BINARY_CLASSIFICATION, n_estimators=25).fit(X, y)

    metrics = MetricsEvaluator.classification_metrics(y, model.predict(X), y_proba=model.predict_proba(X))
    assert "auc_roc" in metrics and metrics["accuracy"] > 0.7

    cv = CrossValidator(strategy="stratified_kfold", n_splits=3).cross_validate(
        lambda: ClassicalMLModel(algorithm="random_forest", task_type=TaskType.BINARY_CLASSIFICATION, n_estimators=15),
        X,
        y,
    )
    assert len(cv["folds"]) == 3

    cpcv = list(CombinatorialPurgedKFold(n_splits=4, n_test_groups=2, purge_gap=2).split(X))
    assert len(cpcv) == 6

    interp = InterpretabilityEngine()
    perm = interp.permutation_importance(model, X.head(80), y.head(80), n_repeats=2)
    assert len(perm) == X.shape[1]
    shap_res = interp.explain_shap_local(model, X.head(60), X.iloc[0], n_samples=16)
    assert len(shap_res["attributions"]) == X.shape[1]
    lime_res = interp.explain_lime_local(model, X.head(60), X.iloc[0], n_perturbations=40)
    assert len(lime_res["feature_weights"]) >= 1

    fair = FairnessAuditor().audit(y, model.predict(X), df["group_attr"], attribute_name="group_attr")
    assert 0.0 <= fair.disparate_impact_ratio <= 1.0
    weights = FairnessAuditor.compute_reweighing_weights(y, df["group_attr"])
    assert len(weights) == len(y)

    conf = ConformalPredictor(model, alpha=0.1).calibrate(X.iloc[:120], y.iloc[:120])
    sets = conf.predict_sets(X.iloc[120:140])
    assert len(sets) == 20

    rob = RobustnessTester().evaluate(model, X.iloc[:80], y.iloc[:80])
    assert "clean_score" in rob

    rets = np.random.default_rng(0).normal(0.001, 0.01, size=100)
    signals = np.ones(100)
    bt = BacktestingEngine.backtest_trading_signals(rets, signals)
    assert "sharpe_ratio" in bt

    card = ModelCardGenerator.build_model_card(model.metadata_, evaluation_metrics=metrics, fairness_report=fair)
    assert "# Model Card:" in card.to_markdown()


def test_automl_hpo_nas_and_dag_pipeline(classification_df: pd.DataFrame) -> None:
    df, _ = DataCleaner().fit_transform(classification_df)
    X, y = df.drop(columns=["target"]), df["target"]

    hpo = HyperparameterOptimizer(strategy="bayesian", max_trials=8).optimize(
        {"x": {"type": "float", "low": -3.0, "high": 3.0}},
        lambda p: -((p["x"] - 1.2) ** 2),
    )
    assert hpo.best_score > -1.0

    res = AutoMLPipelineSearch(time_budget_seconds=20.0, tune_trials=3, include_deep=True).fit(X, y)
    assert res.best_model is not None
    assert len(res.leaderboard) >= 5

    nas = NeuralArchitectureSearcher(
        task_type=TaskType.BINARY_CLASSIFICATION, population_size=3, generations=2, epochs_per_eval=3
    ).search(X.select_dtypes(include=[np.number]).to_numpy(), y.to_numpy())
    assert "hidden_dims" in nas["best_architecture"]

    dag = (
        PipelineBuilder("integration_pipe")
        .add_step("quality", "quality")
        .add_step("split", "split", chain=True, config={"target_column": "target"})
        .add_step("clean", "clean", chain=True)
        .add_step("automl", "automl", chain=True, config={"target_column": "target", "time_budget_seconds": 12.0})
        .add_step("evaluate", "evaluate", chain=True, config={"target_column": "target"})
        .build()
    )
    assert "flowchart LR" in dag.to_mermaid()
    assert "PipelineBuilder" in PipelineBuilder.export_python_code(dag)
    exec_res = PipelineExecutor().execute(dag, initial_artifacts={"data": classification_df})
    assert exec_res.status == "success"


def test_mlops_copilot_reporting_and_18_domains(
    classification_df: pd.DataFrame, workspace_tmp: Path
) -> None:
    df, _ = DataCleaner().fit_transform(classification_df)
    X, y = df.drop(columns=["target"]), df["target"]
    model = ClassicalMLModel(algorithm="random_forest", task_type=TaskType.BINARY_CLASSIFICATION, n_estimators=20).fit(X, y)

    tracker = ExperimentTracker(db_path=workspace_tmp / "track.db", artifact_root=workspace_tmp / "art")
    with tracker.start_run("integration_exp", "run_1") as rid:
        tracker.log_params({"n_estimators": 20})
        tracker.log_metrics({"f1": 0.91})
    assert tracker.get_run(rid).status == "FINISHED"

    reg = ModelRegistry(db_path=workspace_tmp / "reg.db", artifact_dir=workspace_tmp / "reg_art")
    v1 = reg.register_model(model, "credit_clf", metrics={"f1": 0.91}, stage=ModelStage.PRODUCTION)
    assert v1.version == 1

    srv = ModelServingEngine(registry=reg)
    srv.deploy_from_registry("credit_ep", "credit_clf", stage=ModelStage.PRODUCTION)
    pred = srv.predict("credit_ep", X.head(4), return_probabilities=True)
    assert pred["n_records"] == 4

    shifted = df.copy()
    shifted["feature_0"] = shifted["feature_0"] * 2.0 + 10.0
    drift = DriftDetector().detect(df, shifted)
    assert drift.drifted_feature_count >= 1
    ph = DriftDetector.detect_concept_drift_page_hinkley([0.05] * 25 + [0.45] * 25, threshold=2.0)
    assert ph["drift_detected"] is True

    mon = ModelMonitor("credit_clf", max_p95_latency_ms=100.0)
    mon.record_inference(latency_ms=15.0, prediction=1, actual=1)
    assert mon.evaluate_health()["status"] == "healthy"

    ct = ContinuousTrainingOrchestrator(registry=reg, tracker=tracker)
    ct_res = ct.evaluate_and_retrain(
        model_name="credit_clf",
        reference_df=df,
        new_labeled_df=shifted,
        target_col="target",
        trainer_fn=lambda Xf, yf: ClassicalMLModel(
            algorithm="gradient_boosting", task_type=TaskType.BINARY_CLASSIFICATION, n_estimators=20
        ).fit(Xf, yf),
    )
    assert ct_res["retrained"] is True

    agent = OmniForgeCopilotAgent(provider="local")
    agent.load_dataframe(df, name="test_df")
    r1 = agent.chat("profile data quality")
    assert "Quality Score" in r1.reply
    r2 = agent.chat("list all industry domains")
    assert "18" in r2.reply

    plugins = default_domain_registry.list_plugins()
    assert len(plugins) == 18
    for p in plugins:
        sdf = p.engineer_domain_features(p.generate_sample_dataset(n_samples=40))
        kpis = p.compute_domain_kpis(sdf)
        assert len(sdf) == 40
        assert isinstance(kpis, dict)

    reporter = ReportGenerator()
    payload = reporter.build_report(
        title="Integration Verification Report",
        domain="finance",
        dataset_summary={"n_rows": len(df), "n_columns": len(df.columns)},
        tables={"Sample": [{"a": 1, "b": 2}]},
    )
    for ext in ("html", "md", "json", "xlsx", "docx", "pptx", "pdf"):
        p_out = reporter.export(payload, workspace_tmp / f"rep.{ext}")
        assert p_out.exists() and p_out.stat().st_size > 0
    nb_p = ReportExporters.to_jupyter_notebook(workspace_tmp / "analysis.ipynb")
    assert nb_p.exists()
    ms_dir = ReportExporters.export_microservice_bundle(
        workspace_tmp / "microservice", model_artifact_path=v1.artifact_path
    )
    assert (ms_dir / "Dockerfile").exists() and (ms_dir / "app.py").exists()
