"""Fluent programmatic & declarative YAML/JSON PipelineBuilder with full Python pro-code export."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from omniforge.core.exceptions import PipelineError
from omniforge.core.types import TaskType
from omniforge.pipelines.base import FunctionStep, PipelineContext, PipelineStep
from omniforge.pipelines.dag import PipelineDAG


def _builtin_step_handlers() -> dict[str, Callable[..., Any]]:
    """Return built-in declarative step implementations connecting every OmniForge subsystem."""

    def _resolve_df(ctx: PipelineContext, primary_key: str) -> Any:
        for k in (primary_key, "features_data", "clean_data", "raw_data", "data"):
            val = ctx.get(k)
            if val is not None:
                return val
        raise PipelineError(
            f"No DataFrame found in pipeline context (checked '{primary_key}', 'features_data', 'clean_data', 'raw_data', 'data')."
        )

    def step_ingest(ctx: PipelineContext, source: Any = None, output_key: str = "raw_data", **kw: Any) -> Any:
        from omniforge.data.ingestion import DataIngestionEngine

        src = source if source is not None else ctx.get("source")
        res = DataIngestionEngine().ingest(src, **kw)
        ctx.set(output_key, res.data)
        ctx.set(f"{output_key}_metadata", res.metadata)
        return res.data

    def step_clean(ctx: PipelineContext, input_key: str = "raw_data", output_key: str = "clean_data", **kw: Any) -> Any:
        from omniforge.data.cleaning import CleaningConfig, DataCleaner

        df = _resolve_df(ctx, input_key)
        cleaner = DataCleaner(CleaningConfig(**kw))
        cleaned, report = cleaner.fit_transform(df)
        ctx.set(output_key, cleaned)
        ctx.set("cleaning_report", report)
        return cleaned

    def step_quality(ctx: PipelineContext, input_key: str = "clean_data", **kw: Any) -> Any:
        from omniforge.data.quality import DataQualityProfiler

        df = _resolve_df(ctx, input_key)
        report = DataQualityProfiler().profile(df, **kw)
        ctx.set("quality_report", report)
        return report

    def step_validate(ctx: PipelineContext, input_key: str = "clean_data", **kw: Any) -> Any:
        from omniforge.data.validation import DataValidator

        df = _resolve_df(ctx, input_key)
        validator = DataValidator.auto_generate_suite(df)
        report = validator.validate(df)
        ctx.set("validation_report", report)
        return report

    def step_eda(ctx: PipelineContext, input_key: str = "clean_data", target_column: str | None = None, **kw: Any) -> Any:
        from omniforge.eda.automated_eda import AutomatedEDA

        df = _resolve_df(ctx, input_key)
        report = AutomatedEDA().run(df, target_column=target_column, generate_figures=kw.get("generate_figures", False))
        ctx.set("eda_report", report)
        return report

    def step_features(
        ctx: PipelineContext,
        input_key: str = "clean_data",
        output_key: str = "features_data",
        target_column: str | None = None,
        **kw: Any,
    ) -> Any:
        from omniforge.features.engineering import FeatureConfig, FeatureEngineeringPipeline

        df = _resolve_df(ctx, input_key)
        y = df[target_column] if (target_column and target_column in df.columns) else None
        X = df.drop(columns=[target_column]) if (target_column and target_column in df.columns) else df
        fe = FeatureEngineeringPipeline(FeatureConfig(**kw))
        X_eng = fe.fit_transform(X, y)
        if y is not None and target_column:
            X_eng[target_column] = y.values
        ctx.set(output_key, X_eng)
        ctx.set("feature_pipeline", fe)
        return X_eng

    def step_train(
        ctx: PipelineContext,
        input_key: str = "features_data",
        target_column: str = "target",
        model_name: str = "random_forest",
        task_type: str = "binary_classification",
        **kw: Any,
    ) -> Any:
        from omniforge.models.registry import create_model

        df = _resolve_df(ctx, input_key)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        model = create_model(model_name, task_type=TaskType(task_type), **kw)
        model.fit(X, y)
        ctx.set("trained_model", model)
        return model

    def step_automl(
        ctx: PipelineContext,
        input_key: str = "clean_data",
        target_column: str = "target",
        **kw: Any,
    ) -> Any:
        from omniforge.automl.pipeline_search import AutoMLPipelineSearch

        df = _resolve_df(ctx, input_key)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        res = AutoMLPipelineSearch(**kw).fit(X, y)
        ctx.set("automl_result", res)
        ctx.set("trained_model", res.best_model)
        return res

    def step_evaluate(
        ctx: PipelineContext,
        input_key: str = "clean_data",
        target_column: str = "target",
        **kw: Any,
    ) -> Any:
        from omniforge.evaluation.metrics import MetricsEvaluator

        model = ctx.get("trained_model")
        df = ctx.get(input_key)
        X = df.drop(columns=[target_column])
        y = df[target_column].to_numpy()
        preds = model.predict(X)
        if model.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            metrics = MetricsEvaluator.classification_metrics(y, preds, model.predict_proba(X))
        else:
            metrics = MetricsEvaluator.regression_metrics(y, preds, n_features=X.shape[1])
        ctx.set("evaluation_metrics", metrics)
        return metrics

    return {
        "ingest": step_ingest,
        "clean": step_clean,
        "quality": step_quality,
        "validate": step_validate,
        "eda": step_eda,
        "features": step_features,
        "train": step_train,
        "automl": step_automl,
        "evaluate": step_evaluate,
    }


class PipelineBuilder:
    """Fluent & declarative pipeline builder supporting YAML/JSON specs and Python code export."""

    def __init__(self, name: str = "uap_pipeline") -> None:
        self._dag = PipelineDAG(name=name)
        self._last_step_id: str | None = None
        self._handlers = _builtin_step_handlers()

    def add_step(
        self,
        step_id: str,
        step_or_type: PipelineStep | str | Callable[..., Any],
        *,
        depends_on: list[str] | None = None,
        chain: bool = False,
        config: dict[str, Any] | None = None,
        cacheable: bool = False,
        retries: int = 0,
    ) -> PipelineBuilder:
        """Add a built-in declarative step, custom :class:`PipelineStep`, or Python callable."""
        deps = list(depends_on or [])
        if chain and not deps and self._last_step_id is not None:
            deps = [self._last_step_id]

        if isinstance(step_or_type, PipelineStep):
            step = step_or_type
            if deps:
                step.depends_on = deps
        elif isinstance(step_or_type, str):
            key = step_or_type.lower()
            if key not in self._handlers:
                raise PipelineError(
                    f"Unknown built-in pipeline step type '{step_or_type}'.",
                    suggestions=[f"Built-in types: {sorted(self._handlers.keys())}"],
                )
            step = FunctionStep(
                step_id=step_id,
                fn=self._handlers[key],
                depends_on=deps,
                config=config,
                cacheable=cacheable,
                retries=retries,
                step_type=key,
            )
        elif callable(step_or_type):
            step = FunctionStep(
                step_id=step_id,
                fn=step_or_type,
                depends_on=deps,
                config=config,
                cacheable=cacheable,
                retries=retries,
                step_type="custom_callable",
            )
        else:
            raise PipelineError("step_or_type must be a PipelineStep, built-in type string, or callable.")

        self._dag.add_step(step)
        self._last_step_id = step.step_id
        return self

    def build(self) -> PipelineDAG:
        """Validate and return the compiled :class:`PipelineDAG`."""
        self._dag.validate()
        return self._dag

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> PipelineDAG:
        """Build a validated :class:`PipelineDAG` from a JSON/YAML dictionary specification."""
        builder = cls(name=str(spec.get("name", "declarative_pipeline")))
        for item in spec.get("steps", []):
            builder.add_step(
                step_id=str(item["step_id"]),
                step_or_type=str(item["step_type"]),
                depends_on=list(item.get("depends_on", [])),
                config=dict(item.get("config", {})),
                cacheable=bool(item.get("cacheable", False)),
                retries=int(item.get("retries", 0)),
            )
        return builder.build()

    @classmethod
    def from_yaml(cls, path_or_text: str | Path) -> PipelineDAG:
        """Load a declarative pipeline from a YAML file path or YAML string."""
        p = Path(path_or_text)
        raw = p.read_text(encoding="utf-8") if p.exists() else str(path_or_text)
        spec = yaml.safe_load(raw)
        if not isinstance(spec, dict):
            raise PipelineError("Pipeline YAML specification must be a mapping.")
        return cls.from_dict(spec)

    @staticmethod
    def export_python_code(dag: PipelineDAG) -> str:
        """Export any visual/declarative :class:`PipelineDAG` into standalone runnable Python code."""
        spec = dag.to_dict()
        spec_json = json.dumps(spec, indent=4)
        return f'''"""Auto-exported OmniForge Production Pipeline: {dag.name}."""

from __future__ import annotations

from omniforge.pipelines import PipelineBuilder, PipelineExecutor

PIPELINE_SPEC = {spec_json}


def run_pipeline(initial_artifacts: dict | None = None):
    """Build and execute the '{dag.name}' DAG with full auditing and reproducibility."""
    dag = PipelineBuilder.from_dict(PIPELINE_SPEC)
    executor = PipelineExecutor()
    return executor.execute(dag, initial_artifacts=initial_artifacts or {{}})


if __name__ == "__main__":
    result = run_pipeline()
    print("Pipeline status:", result.status)
    for rec in result.step_records:
        print(f"  - {{rec.step_id}} ({{rec.step_type}}): {{rec.status}} in {{rec.duration_seconds:.3f}}s")
'''
