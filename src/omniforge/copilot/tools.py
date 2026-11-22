"""Full OmniForge Copilot Tool Registry exposing every major platform subsystem as callable agent tools."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from omniforge.copilot.memory import CopilotMemory
from omniforge.core.exceptions import CopilotError
from omniforge.core.utils import to_json_object

_FORBIDDEN_SYSTEM_PREFIXES: tuple[str, ...] = ("/etc", "/proc", "/sys", "/root", "/dev", "/var/run")


def _validate_copilot_ingest_source(source: str) -> str | Path:
    """Enforce OWASP LLM06 / MITRE CWE-22 path confinement on Copilot file ingestion."""
    raw = str(source).strip()
    if not raw or "\x00" in raw:
        raise CopilotError("Invalid or empty data source provided to Copilot tool_ingest_data.")

    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https"):
        return raw
    if parsed.scheme and parsed.scheme not in ("file", ""):
        raise CopilotError(
            f"Unsupported URI scheme {parsed.scheme!r} in Copilot tool_ingest_data; only confined local files or http(s) URLs are permitted."
        )

    local_str = parsed.path if parsed.scheme == "file" else raw
    path_parts = Path(local_str).parts
    if ".." in path_parts:
        raise CopilotError(f"Path traversal ('..') is forbidden in Copilot data ingestion: {source!r}")

    allowed_root = Path(os.environ.get("OMNIFORGE_COPILOT_INGEST_ROOT") or Path.cwd()).resolve()
    cand = Path(local_str)
    resolved = (cand if cand.is_absolute() else (allowed_root / cand)).resolve()

    resolved_posix = resolved.as_posix()
    for prefix in _FORBIDDEN_SYSTEM_PREFIXES:
        if resolved_posix == prefix or resolved_posix.startswith(prefix + "/"):
            raise CopilotError(f"Access to sensitive system path is forbidden: {source!r}")

    if not resolved.is_relative_to(allowed_root):
        raise CopilotError(
            f"Data source {source!r} resolves outside allowed Copilot ingest root ({allowed_root})."
        )

    rel_parts = resolved.relative_to(allowed_root).parts
    if any(part.startswith(".") for part in rel_parts):
        raise CopilotError(f"Access to hidden dotfiles or secret directories is forbidden: {source!r}")

    return resolved


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
            return to_json_object(res)
        except Exception as exc:
            raise CopilotError(
                f"Copilot tool '{tool_name}' execution failed: {exc}",
                context={"tool": tool_name, "arguments": arguments or {}},
                cause=exc,
            ) from exc

    def _register_platform_tools(self) -> None:
        def tool_ingest_data(source: str, dataset_name: str = "dataset") -> dict[str, Any]:
            from omniforge.data.ingestion import DataIngestionEngine

            safe_source = _validate_copilot_ingest_source(source)
            res = DataIngestionEngine().ingest(safe_source, name=dataset_name)
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

        # ------------------------------------------------------------------ Phase 6 C4 tools
        def tool_root_cause_diagnosis(
            reference_dataset: str = "reference_dataset",
            current_dataset: str = "current_dataset",
            target_col: str | None = None,
            time_col: str | None = None,
        ) -> dict[str, Any]:
            """C5 causal drift root-cause: drift -> Delta-attribution -> DAG ancestors."""
            from omniforge.mlops.root_cause import DriftRootCauseAnalyzer

            _, ref_df = self.memory.get_dataset(reference_dataset)
            _, cur_df = self.memory.get_dataset(current_dataset)
            model = self.memory.working_state["models"].get("automl_champion")
            feat_cols = [c for c in ref_df.columns if c not in (target_col, time_col)]
            res = DriftRootCauseAnalyzer().analyze(
                ref_df, cur_df, model=model, feature_cols=feat_cols,
                target_col=target_col, time_col=time_col,
            )
            self.memory.working_state["reports"]["root_cause"] = res
            return {
                "overall_status": str(res.overall_status.value),
                "n_reference": res.n_reference,
                "n_current": res.n_current,
                "root_causes": res.root_causes,
                "symptoms": res.symptoms,
                "drifted_features": [
                    {
                        "column": f.column,
                        "status": str(f.status.value),
                        "psi": f.psi,
                        "delta_prediction": f.delta_prediction,
                        "is_root_cause": f.is_root_cause,
                        "is_symptom": f.is_symptom,
                        "upstream_ancestors": f.upstream_ancestors,
                        "remediation": f.remediation,
                    }
                    for f in res.drifted_features
                ],
                "remediation_prescriptions": res.remediation_prescriptions,
                "total_attribution": res.total_attribution,
            }

        def tool_twin_control(
            n_x: int = 2,
            n_u: int = 1,
            horizon: int = 10,
            n_steps: int = 8,
            dt: float = 1.0,
            setpoint: list[float] | None = None,
            measurements: list[list[float]] | None = None,
        ) -> dict[str, Any]:
            """C7 closed-loop Digital Twin MPC control."""
            from omniforge.simulation import ClosedLoopMPCController, build_default_twin

            twin = build_default_twin(n_x=n_x, n_u=n_u, dt=dt)
            controller = ClosedLoopMPCController(twin, horizon=horizon, random_seed=42)
            if measurements is None:
                import numpy as np
                rng = np.random.default_rng(1234)
                measurements = [rng.normal(0.0, 0.5, size=n_x) for _ in range(n_steps)]
            meas = [np.asarray(m, dtype=float) for m in measurements][:n_steps]
            sp = None
            if setpoint:
                sp = np.asarray(setpoint, dtype=float).ravel()[:n_x]
            report = controller.run_closed_loop(meas, setpoint=sp)
            self.memory.working_state["reports"]["twin_control"] = report
            return {
                "n_steps_applied": report.n_steps_applied,
                "all_constraints_satisfied": report.all_constraints_satisfied,
                "constraint_violations": report.constraint_violations,
                "total_cost": report.total_cost,
                "state_trajectory": report.state_trajectory,
                "control_history": report.control_history,
                "setpoint": report.setpoint,
                "stage_costs": report.stage_costs,
            }

        def tool_split_lock(
            strategy: str = "auto",
            test_size: float = 0.2,
            cv_folds: int = 5,
            time_col: str | None = None,
            group_col: str | None = None,
            target_col: str | None = None,
            gap: int = 0,
            dataset_name: str | None = None,
        ) -> dict[str, Any]:
            """Enforce leakage-safe Split Lock (train/test integrity)."""
            from omniforge.core.contracts import LeakageGuard

            _, df = self.memory.get_dataset(dataset_name)
            # Deterministic holdout split
            n = len(df)
            cut = max(1, int(n * (1.0 - float(test_size))))
            train_df = df.iloc[:cut].copy()
            test_df = df.iloc[cut:].copy()
            report = LeakageGuard.validate_split(
                train_df, test_df, group_col=group_col, time_col=time_col,
            )
            self.memory.working_state["reports"]["split_lock"] = report
            return report.model_dump()

        def tool_pii_audit(
            dataset_name: str | None = None,
            pii_policy: str = "pseudonymize",
        ) -> dict[str, Any]:
            """C8 zero-trust PII audit: detect + pseudonymize or block."""
            from omniforge.core.security import DataFramePIIGuard

            _, df = self.memory.get_dataset(dataset_name)
            guard = DataFramePIIGuard(tenant_id="copilot", pii_policy=pii_policy)
            redacted, report = guard.transform(df)
            out_name = f"{self.memory.working_state.get('active_dataset', 'dataset')}_pii_guarded"
            self.memory.set_dataset(out_name, redacted)
            self.memory.working_state["reports"]["pii_audit"] = report
            return {
                "pii_policy": pii_policy,
                "pii_detected": report["pii_detected"],
                "pii_columns": report["pii_columns"],
                "pii_cells_total": report["pii_cells_total"],
                "redacted_columns": report.get("redacted_columns", {}),
                "output_dataset": out_name,
            }

        def tool_pareto_automl(
            target_column: str = "target",
            dataset_name: str | None = None,
            time_budget_seconds: float = 30.0,
            max_carbon_grams_co2e: float | None = None,
            max_inference_latency_ms: float | None = None,
        ) -> dict[str, Any]:
            """C3 Green-AI Pareto AutoML with carbon / latency budgets."""
            from omniforge.automl.pipeline_search import AutoMLPipelineSearch

            _, df = self.memory.get_dataset(dataset_name)
            if target_column not in df.columns:
                target_column = str(df.columns[-1])
            X = df.drop(columns=[target_column])
            y = df[target_column]
            res = AutoMLPipelineSearch(
                time_budget_seconds=time_budget_seconds,
                tune_trials=4,
                include_deep=True,
                max_carbon_grams_co2e=max_carbon_grams_co2e,
                max_inference_latency_ms=max_inference_latency_ms,
            ).fit(X, y)
            self.memory.set_model("pareto_champion", res.best_model)
            self.memory.working_state["reports"]["pareto_automl"] = res
            return {
                "best_algorithm": res.best_model_name,
                "task_type": res.task_type.value,
                "primary_metric": res.primary_metric,
                "best_score": res.best_score,
                "lift_over_best_baseline": res.lift_over_best_baseline,
                "carbon_gco2e": res.carbon_report.emissions_g_co2eq if res.carbon_report else 0.0,
                "pareto_frontier": res.pareto_frontier,
                "max_carbon_grams_co2e": res.max_carbon_grams_co2e,
                "max_inference_latency_ms": res.max_inference_latency_ms,
                "leaderboard_top5": res.leaderboard[:5],
            }

        def tool_domain_validate(
            domain: str,
            dataset_name: str | None = None,
        ) -> dict[str, Any]:
            """Validate active dataset against an industry domain plugin (18 domains)."""
            from omniforge.domains.plugin_sdk import default_domain_registry

            _, df = self.memory.get_dataset(dataset_name)
            plugin = default_domain_registry.get(domain)
            if plugin is None:
                available = ", ".join(p.metadata.name for p in default_domain_registry.list_plugins())
                raise ValueError(f"Unknown domain '{domain}'. Available: {available}")
            checks = {}
            if hasattr(plugin, "validate_dataset"):
                checks = plugin.validate_dataset(df)
            elif hasattr(plugin, "engineer_domain_features"):
                engineered = plugin.engineer_domain_features(df)
                checks = {
                    "engineered_columns": [c for c in engineered.columns if c not in df.columns],
                    "rows": int(len(engineered)),
                }
            else:
                checks = {"note": f"Domain '{domain}' loaded; no validate_dataset method."}
            self.memory.working_state["reports"][f"domain_{domain}"] = checks
            return {"domain": domain, "compliance": getattr(plugin.metadata, "compliance_frameworks", []), **checks}

        def tool_register_model(
            model_name: str = "omniforge_champion",
            stage: str = "Staging",
        ) -> dict[str, Any]:
            """Register active model to the MLOps Model Registry (Studio 11)."""
            from omniforge.api import workbench as wb
            from omniforge.core.types import ModelStage

            state = self.memory.working_state
            model = state["models"].get("pareto_champion") or state["models"].get("automl_champion")
            if model is None:
                raise ValueError("No trained model in Copilot memory. Run train_automl or pareto_automl first.")
            try:
                ms = ModelStage(stage)
            except ValueError:
                return {
                    "error": f"Invalid model stage '{stage}'.",
                    "available_stages": [s.value for s in ModelStage],
                }
            reg = wb.get_tenant_registry()
            ver = reg.register_model(
                model=model, model_name=model_name, owner="copilot", intended_use="",
                stage=ms,
            )
            return {"registered": ver.model_dump() if hasattr(ver, "model_dump") else dict(ver)}

        def tool_promote_model(
            model_name: str,
            version: int,
            stage: str = "Production",
        ) -> dict[str, Any]:
            """Promote a registered model to a new stage (Studio 11)."""
            from omniforge.api import workbench as wb
            from omniforge.core.types import ModelStage

            try:
                ms = ModelStage(stage)
            except ValueError:
                return {
                    "error": f"Invalid model stage '{stage}'.",
                    "available_stages": [s.value for s in ModelStage],
                }
            reg = wb.get_tenant_registry()
            updated = reg.transition_stage(
                model_name, version, ms, fairness_report=None,
            )
            return {"updated": updated.model_dump() if hasattr(updated, "model_dump") else dict(updated)}

        def tool_list_studios() -> dict[str, Any]:
            """List all 15 OmniForge Workbench studios."""
            studios = [
                {"id": "s00", "name": "Ingestion", "route": "/api/workbench/s00/ingest"},
                {"id": "s01", "name": "Cleaning", "route": "/api/workbench/s01/clean"},
                {"id": "s02", "name": "Mining", "route": "/api/workbench/s02/mine"},
                {"id": "s03", "name": "EDA", "route": "/api/workbench/s03/eda"},
                {"id": "s04", "name": "Diagnostics", "route": "/api/workbench/s04/diagnostics"},
                {"id": "s05", "name": "Feature Engineering", "route": "/api/workbench/s05/features"},
                {"id": "s06", "name": "AutoML", "route": "/api/workbench/s06/run"},
                {"id": "s07", "name": "Advanced AI", "route": "/api/workbench/s07/advanced"},
                {"id": "s08", "name": "Causal", "route": "/api/workbench/s08/causal"},
                {"id": "s09", "name": "Specialized AI", "route": "/api/workbench/s09/nlp"},
                {"id": "s10", "name": "RAI", "route": "/api/workbench/s10/rai"},
                {"id": "s11", "name": "MLOps", "route": "/api/workbench/s11/overview"},
                {"id": "s12", "name": "Reports", "route": "/api/workbench/s12/report"},
                {"id": "s13", "name": "Copilot", "route": "/api/workbench/s13/chat"},
                {"id": "s14", "name": "Supervisory", "route": "/api/workbench/s14/models"},
            ]
            return {"studios": studios, "count": len(studios)}

        def tool_studio_overview(studio_id: str) -> dict[str, Any]:
            """Get overview payload for a specific studio."""
            from omniforge.api import workbench as wb

            state = self.memory.working_state
            if studio_id == "s11":
                reg = wb.get_tenant_registry()
                return {"registered_models": [v.model_dump() for v in reg.list_versions()]}
            if studio_id == "s13":
                return {"history": state.get("uap_chat_history", [])[-10:]}
            return {"studio": studio_id, "note": "overview not implemented for this studio"}

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
            # --- Phase 6 C4 air-gapped deterministic tools ---
            CopilotTool("root_cause_diagnosis", "C5 causal drift root-cause: drift -> Delta-attribution -> DAG ancestors -> remediation.", {"reference_dataset": "str", "current_dataset": "str", "target_col": "str?", "time_col": "str?"}, tool_root_cause_diagnosis),
            CopilotTool("twin_control", "C7 closed-loop Digital Twin MPC control (constrained receding horizon).", {"n_x": "int", "n_u": "int", "horizon": "int", "n_steps": "int", "dt": "float", "setpoint": "list[float]?", "measurements": "list[list[float]]?"}, tool_twin_control),
            CopilotTool("split_lock", "Enforce leakage-safe Split Lock train/test integrity (LeakageGuard).", {"strategy": "str?", "test_size": "float?", "cv_folds": "int?", "time_col": "str?", "group_col": "str?", "target_col": "str?", "gap": "int?", "dataset_name": "str?"}, tool_split_lock),
            CopilotTool("pii_audit", "C8 zero-trust PII audit: detect + pseudonymize or block (fail-closed).", {"dataset_name": "str?", "pii_policy": "str?"}, tool_pii_audit),
            CopilotTool("pareto_automl", "C3 Green-AI Pareto AutoML with carbon / inference-latency budgets.", {"target_column": "str", "time_budget_seconds": "float?", "max_carbon_grams_co2e": "float?", "max_inference_latency_ms": "float?"}, tool_pareto_automl),
            CopilotTool("domain_validate", "Validate active dataset against an industry domain plugin (18 domains).", {"domain": "str", "dataset_name": "str?"}, tool_domain_validate),
            CopilotTool("register_model", "Register active model to the MLOps Model Registry (Studio 11).", {"model_name": "str?", "stage": "str?"}, tool_register_model),
            CopilotTool("promote_model", "Promote a registered model to a new stage (Studio 11).", {"model_name": "str", "version": "int", "stage": "str?"}, tool_promote_model),
            CopilotTool("list_studios", "List all 15 OmniForge Workbench studios and routes.", {}, tool_list_studios),
            CopilotTool("studio_overview", "Get overview payload for a specific studio (s11, s13, ...).", {"studio_id": "str"}, tool_studio_overview),
        ]
        for tool in specs:
            self.register(tool)
