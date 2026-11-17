"""Full OmniForge Copilot Tool Registry exposing every major platform subsystem as callable agent tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from omniforge.copilot.memory import CopilotMemory
from omniforge.core.exceptions import CopilotError
from omniforge.core.utils import to_json_safe


@dataclass
class CopilotTool:
    """Metadata and callable binding for a single Copilot tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]


class CopilotToolRegistry:
    """Registers and dispatches typed OmniForge platform tools against shared :class:`CopilotMemory`."""

    def __init__(self, memory: CopilotMemory) -> None:
        self.memory = memory
        self._tools: dict[str, CopilotTool] = {}
        self._register_platform_tools()

    def register(self, tool: CopilotTool) -> None:
        """Register a new Copilot tool."""
        self._tools[tool.name] = tool

    def list_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI/Anthropic-compatible tool specifications."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def invoke(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke a registered tool by name and return JSON-safe structured output."""
        if tool_name not in self._tools:
            raise CopilotError(
                f"Unknown Copilot tool '{tool_name}'.",
                suggestions=[f"Available tools: {sorted(self._tools.keys())}"],
            )
        try:
            res = self._tools[tool_name].handler(**(arguments or {}))
            return to_json_safe(res)
        except Exception as exc:
            raise CopilotError(
                f"Copilot tool '{tool_name}' execution failed: {exc}",
                context={"tool": tool_name, "arguments": arguments or {}},
                cause=exc,
            ) from exc

    def _register_platform_tools(self) -> None:
        def tool_ingest_data(source: str, dataset_name: str = "dataset") -> dict[str, Any]:
            from omniforge.data.ingestion import DataIngestionEngine

            res = DataIngestionEngine().ingest(source, name=dataset_name)
            self.memory.set_dataset(dataset_name, res.data)
            self.memory.working_state["reports"][f"meta_{dataset_name}"] = res.metadata
            return {
                "dataset_name": dataset_name,
                "rows": res.metadata.n_rows,
                "columns": res.metadata.n_columns,
                "column_names": list(res.data.columns),
                "checksum": res.metadata.checksum_sha256[:16],
            }

        def tool_generate_demo_data(
            task: str = "classification", n_samples: int = 300, dataset_name: str = "demo_data"
        ) -> dict[str, Any]:
            from omniforge.data.synthetic import SyntheticDataGenerator

            gen = SyntheticDataGenerator()
            if task == "regression":
                df = gen.generate_regression_dataset(n_samples=n_samples)
            elif task == "forecasting":
                df = gen.generate_time_series_dataset(n_periods=max(90, n_samples))
            elif task == "causal":
                df = gen.generate_causal_dataset(n_samples=n_samples)
            else:
                df = gen.generate_classification_dataset(n_samples=n_samples, missing_rate=0.05)
            self.memory.set_dataset(dataset_name, df)
            return {"dataset_name": dataset_name, "task": task, "rows": len(df), "columns": list(df.columns)}

        def tool_profile_quality(dataset_name: str | None = None) -> dict[str, Any]:
            from omniforge.data.quality import DataQualityProfiler

            name, df = self.memory.get_dataset(dataset_name)
            rep = DataQualityProfiler().profile(df)
            self.memory.working_state["reports"][f"quality_{name}"] = rep
            return rep.model_dump()

        def tool_clean_data(
            dataset_name: str | None = None,
            numeric_imputation: str = "median",
            outlier_method: str = "iqr",
            outlier_action: str = "clip",
        ) -> dict[str, Any]:
            from omniforge.data.cleaning import CleaningConfig, DataCleaner

            name, df = self.memory.get_dataset(dataset_name)
            cfg = CleaningConfig(
                numeric_imputation=numeric_imputation,
                outlier_method=outlier_method,
                outlier_action=outlier_action,
            )
            cleaned, rep = DataCleaner(cfg).fit_transform(df)
            out_name = f"{name}_clean"
            self.memory.set_dataset(out_name, cleaned)
            self.memory.working_state["reports"][f"clean_{out_name}"] = rep
            return {"output_dataset": out_name, **rep.model_dump()}

        def tool_run_eda(
            dataset_name: str | None = None, target_column: str | None = None
        ) -> dict[str, Any]:
            from omniforge.eda.automated_eda import AutomatedEDA

            name, df = self.memory.get_dataset(dataset_name)
            rep = AutomatedEDA().run(df, target_column=target_column, generate_figures=False)
            self.memory.working_state["reports"][f"eda_{name}"] = rep
            return {
                "dataset": name,
                "rows": rep.profile.n_rows,
                "columns": rep.profile.n_columns,
                "insights": [i.model_dump() for i in rep.insights],
                "narrative": rep.narrative,
            }

        def tool_train_automl(
            target_column: str = "target",
            dataset_name: str | None = None,
            time_budget_seconds: float = 30.0,
            model_name: str = "automl_champion",
        ) -> dict[str, Any]:
            from omniforge.automl.pipeline_search import AutoMLPipelineSearch

            _, df = self.memory.get_dataset(dataset_name)
            if target_column not in df.columns:
                target_column = str(df.columns[-1])
            X = df.drop(columns=[target_column])
            y = df[target_column]
            res = AutoMLPipelineSearch(
                time_budget_seconds=time_budget_seconds, tune_trials=4, include_deep=True
            ).fit(X, y)
            self.memory.set_model(model_name, res.best_model)
            self.memory.working_state["reports"][f"automl_{model_name}"] = res
            return {
                "model_name": model_name,
                "best_algorithm": res.best_model_name,
                "task_type": res.task_type.value,
                "primary_metric": res.primary_metric,
                "best_score": res.best_score,
                "lift_over_best_baseline": res.lift_over_best_baseline,
                "leaderboard_top5": res.leaderboard[:5],
                "carbon_g_co2eq": res.carbon_report.emissions_g_co2eq if res.carbon_report else 0.0,
            }

        def tool_run_forecast(
            target_column: str = "y",
            time_column: str | None = "timestamp",
            horizon: int = 14,
            dataset_name: str | None = None,
        ) -> dict[str, Any]:
            from omniforge.forecasting.engine import ForecastingEngine

            _, df = self.memory.get_dataset(dataset_name)
            t_col = time_column if (time_column and time_column in df.columns) else None
            rep = ForecastingEngine(include_deep=False).forecast(
                df, target_col=target_column, time_col=t_col, horizon=horizon
            )
            self.memory.working_state["reports"]["forecast_report"] = rep
            return {
                "champion_model": rep.champion_model,
                "horizon": rep.horizon,
                "leaderboard": rep.leaderboard[:5],
                "forecast_preview": rep.forecast.head(5).to_dict(orient="records"),
            }

        def tool_explain_model(
            model_name: str | None = None, dataset_name: str | None = None, target_column: str = "target"
        ) -> dict[str, Any]:
            from omniforge.evaluation.interpretability import InterpretabilityEngine

            m_name, model = self.memory.get_model(model_name)
            _, df = self.memory.get_dataset(dataset_name)
            X = df.drop(columns=[target_column], errors="ignore")
            interp = InterpretabilityEngine()
            shap_local = interp.explain_shap_local(model, X, X.iloc[0], n_samples=24)
            global_imp = model.get_feature_importance()
            return {
                "model": m_name,
                "global_importance": dict(list(global_imp.items())[:10]),
                "local_shap_sample_0": shap_local,
            }

        def tool_audit_fairness(
            sensitive_column: str = "group_attr",
            target_column: str = "target",
            model_name: str | None = None,
            dataset_name: str | None = None,
        ) -> dict[str, Any]:
            from omniforge.evaluation.fairness import FairnessAuditor

            _, model = self.memory.get_model(model_name)
            _, df = self.memory.get_dataset(dataset_name)
            X = df.drop(columns=[target_column], errors="ignore")
            preds = model.predict(X)
            rep = FairnessAuditor().audit(df[target_column], preds, df[sensitive_column], attribute_name=sensitive_column)
            return rep.model_dump()

        def tool_check_drift(
            reference_dataset: str, current_dataset: str
        ) -> dict[str, Any]:
            from omniforge.mlops.drift import DriftDetector

            _, ref_df = self.memory.get_dataset(reference_dataset)
            _, cur_df = self.memory.get_dataset(current_dataset)
            return DriftDetector().detect(ref_df, cur_df).model_dump()

        def tool_run_causal(
            treatment_col: str = "treatment",
            outcome_col: str = "outcome",
            method: str = "aipw",
            dataset_name: str | None = None,
        ) -> dict[str, Any]:
            from omniforge.causal.inference import CausalInferenceEngine

            _, df = self.memory.get_dataset(dataset_name)
            confounders = [
                c for c in df.select_dtypes(include=["number"]).columns if c not in (treatment_col, outcome_col)
            ]
            res = CausalInferenceEngine(bootstrap_rounds=20).estimate_ate(
                df, treatment_col=treatment_col, outcome_col=outcome_col, confounders=confounders, method=method
            )
            return res.model_dump()

        def tool_generate_pipeline_code(target_column: str = "target", pipeline_name: str = "copilot_pipeline") -> dict[str, Any]:
            from omniforge.pipelines.builder import PipelineBuilder

            dag = (
                PipelineBuilder(name=pipeline_name)
                .add_step("ingest", "ingest")
                .add_step("clean", "clean", chain=True)
                .add_step("quality", "quality", chain=True)
                .add_step("automl", "automl", chain=True, config={"target_column": target_column})
                .add_step("evaluate", "evaluate", chain=True, config={"target_column": target_column})
                .build()
            )
            return {
                "pipeline_name": pipeline_name,
                "mermaid_dag": dag.to_mermaid(),
                "python_code": PipelineBuilder.export_python_code(dag),
            }

        def tool_list_domains() -> dict[str, Any]:
            from omniforge.domains.plugin_sdk import default_domain_registry

            return {
                "domains": [
                    {"name": p.metadata.name, "display_name": p.metadata.display_name, "compliance": p.metadata.compliance_frameworks}
                    for p in default_domain_registry.list_plugins()
                ]
            }

        def tool_workspace_status() -> dict[str, Any]:
            return {
                "loaded_datasets": {
                    k: {"rows": len(v), "cols": len(v.columns)}
                    for k, v in self.memory.working_state["datasets"].items()
                    if isinstance(v, pd.DataFrame)
                },
                "trained_models": list(self.memory.working_state["models"].keys()),
                "active_dataset": self.memory.working_state.get("active_dataset"),
                "active_model": self.memory.working_state.get("active_model"),
            }

        specs = [
            CopilotTool("ingest_data", "Ingest dataset from file path, database URI, or API.", {"source": "str", "dataset_name": "str"}, tool_ingest_data),
            CopilotTool("generate_demo_data", "Generate a synthetic dataset for classification, regression, forecasting, or causal tasks.", {"task": "str", "n_samples": "int"}, tool_generate_demo_data),
            CopilotTool("profile_quality", "Profile 6-dimensional enterprise data quality score.", {"dataset_name": "str?"}, tool_profile_quality),
            CopilotTool("clean_data", "Clean missing values, duplicates, strings, and outliers with audit report.", {"dataset_name": "str?", "numeric_imputation": "str"}, tool_clean_data),
            CopilotTool("run_eda", "Run Automated EDA with statistical profile and executive insights.", {"dataset_name": "str?", "target_column": "str?"}, tool_run_eda),
            CopilotTool("train_automl", "Train AutoML models with strong baseline benchmarking and carbon tracking.", {"target_column": "str", "time_budget_seconds": "float"}, tool_train_automl),
            CopilotTool("run_forecast", "Run statistical, ML, and foundation time-series forecasting tournament.", {"target_column": "str", "horizon": "int"}, tool_run_forecast),
            CopilotTool("explain_model", "Compute global feature importance and local SHAP explanations.", {"model_name": "str?", "target_column": "str"}, tool_explain_model),
            CopilotTool("audit_fairness", "Audit subgroup fairness (Disparate Impact, Equalized Odds, Demographic Parity).", {"sensitive_column": "str", "target_column": "str"}, tool_audit_fairness),
            CopilotTool("check_drift", "Run PSI/KS/Wasserstein drift analysis between reference and current datasets.", {"reference_dataset": "str", "current_dataset": "str"}, tool_check_drift),
            CopilotTool("run_causal", "Estimate causal ATE and run DoWhy-style refutation checks.", {"treatment_col": "str", "outcome_col": "str", "method": "str"}, tool_run_causal),
            CopilotTool("generate_pipeline_code", "Generate a validated DAG Mermaid graph and standalone Python pipeline code.", {"target_column": "str", "pipeline_name": "str"}, tool_generate_pipeline_code),
            CopilotTool("list_domains", "List all installed industry domain plugins and compliance frameworks.", {}, tool_list_domains),
            CopilotTool("workspace_status", "Inspect currently loaded datasets and trained models in Copilot memory.", {}, tool_workspace_status),
        ]
        for tool in specs:
            self.register(tool)
