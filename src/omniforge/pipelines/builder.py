"""Fluent programmatic & declarative YAML/JSON PipelineBuilder with full Python pro-code export."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
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

    def step_split(
        ctx: PipelineContext,
        input_key: str = "raw_data",
        target_column: str | None = None,
        strategy: str = "auto",
        test_size: float = 0.20,
        val_size: float = 0.0,
        group_col: str | None = None,
        time_col: str | None = None,
        gap: int = 0,
        random_seed: int = 42,
        **kw: Any,
    ) -> Any:
        from omniforge.data.splitting import LeakageSafeSplitter, SplitConfig

        if ctx.get("preprocessing_fit_scope") == "full_frame":
            raise PipelineError("Split raw data before fitting cleaning or feature transforms.")
        df = _resolve_df(ctx, input_key)
        if target_column is not None and target_column not in df.columns:
            raise PipelineError("The configured split target does not exist in the input frame.")
        cfg = SplitConfig(
            strategy=strategy,  # type: ignore[arg-type]
            test_size=test_size,
            val_size=val_size,
            group_col=group_col,
            time_col=time_col,
            gap=gap,
            random_seed=random_seed,
        )
        res = LeakageSafeSplitter(cfg).split(
            df,
            target_col=target_column if (target_column and target_column in df.columns) else None,
        )
        ctx.set("train_data", res.train_df)
        ctx.set("test_data", res.test_df)
        ctx.set("val_data", res.val_df)
        ctx.set("split_target", target_column)
        ctx.set("split_config", cfg)
        ctx.set("split_result", res)
        return res.train_df

    def step_clean(ctx: PipelineContext, input_key: str = "raw_data", output_key: str = "clean_data", **kw: Any) -> Any:
        from omniforge.data.cleaning import CleaningConfig, DataCleaner

        tr_df = ctx.get("train_data")
        te_df = ctx.get("test_data")
        split_cfg = ctx.get("split_config")
        target = ctx.get("split_target")
        if target is not None:
            kw["exclude_columns"] = list(dict.fromkeys([*(kw.get("exclude_columns") or []), target]))
        if split_cfg is not None and split_cfg.time_col and ctx.get("split_result").strategy_used == "time_series":
            kw["time_column"] = split_cfg.time_col
            kw["causal_time_series"] = True
            kw["exclude_columns"] = list(dict.fromkeys([*(kw.get("exclude_columns") or []), split_cfg.time_col]))
        cleaner = DataCleaner(CleaningConfig(**kw))
        if isinstance(tr_df, pd.DataFrame) and isinstance(te_df, pd.DataFrame):
            tr_clean, report = cleaner.fit_transform(tr_df)
            te_clean, te_report = cleaner.transform(te_df)
            va_df = ctx.get("val_data")
            if isinstance(va_df, pd.DataFrame):
                va_clean, va_report = cleaner.transform(va_df)
                ctx.set("val_data", va_clean)
                ctx.set("validation_cleaning_report", va_report)
            ctx.set("train_data", tr_clean)
            ctx.set("test_data", te_clean)
            ctx.set(output_key, tr_clean)
            ctx.set("cleaning_report", report)
            ctx.set("test_cleaning_report", te_report)
            ctx.set("preprocessing_fit_scope", "train_only")
            return tr_clean

        df = _resolve_df(ctx, input_key)
        cleaned, report = cleaner.fit_transform(df)
        ctx.set(output_key, cleaned)
        ctx.set("cleaning_report", report)
        ctx.set("preprocessing_fit_scope", "full_frame")
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

        fe = FeatureEngineeringPipeline(FeatureConfig(**kw))
        tr_df = ctx.get("train_data")
        te_df = ctx.get("test_data")
        split_target = ctx.get("split_target")
        if target_column is not None and split_target is not None and target_column != split_target:
            raise PipelineError("The feature target must match the declared split target.")
        target_column = target_column or split_target
        if isinstance(tr_df, pd.DataFrame) and isinstance(te_df, pd.DataFrame):
            has_t = bool(target_column and target_column in tr_df.columns)
            y_tr = tr_df[target_column] if has_t else None
            X_tr = tr_df.drop(columns=[target_column]) if has_t else tr_df
            X_te = te_df.drop(columns=[target_column]) if (target_column and target_column in te_df.columns) else te_df
            tr_eng = fe.fit_transform(X_tr, y_tr)
            te_eng = fe.transform(X_te)
            if has_t and target_column:
                tr_eng[target_column] = y_tr.to_numpy()  # type: ignore[union-attr]
            if target_column and target_column in te_df.columns:
                te_eng[target_column] = te_df[target_column].to_numpy()
            va_df = ctx.get("val_data")
            if isinstance(va_df, pd.DataFrame):
                va_features = va_df.drop(columns=[target_column]) if has_t else va_df
                va_eng = fe.transform(va_features)
                if has_t and target_column:
                    va_eng[target_column] = va_df[target_column].to_numpy()
                ctx.set("val_data", va_eng)
            ctx.set("train_data", tr_eng)
            ctx.set("test_data", te_eng)
            ctx.set(output_key, tr_eng)
            ctx.set("feature_pipeline", fe)
            ctx.set("preprocessing_fit_scope", "train_only")
            return tr_eng

        df = _resolve_df(ctx, input_key)
        y = df[target_column] if (target_column and target_column in df.columns) else None
        X = df.drop(columns=[target_column]) if (target_column and target_column in df.columns) else df
        X_eng = fe.fit_transform(X, y)
        if y is not None and target_column:
            X_eng[target_column] = y.values
        ctx.set(output_key, X_eng)
        ctx.set("feature_pipeline", fe)
        ctx.set("preprocessing_fit_scope", "full_frame")
        return X_eng

    def _prepare_train_and_holdout(
        ctx: PipelineContext,
        input_key: str,
        target_column: str,
        *,
        fit_full_data: bool = False,
    ) -> pd.DataFrame:
        preprocessing_scope = ctx.get("preprocessing_fit_scope")
        if preprocessing_scope == "full_frame":
            raise PipelineError("Predictive pipelines must split raw data before learned preprocessing.")
        if preprocessing_scope == "train_only" and ctx.get("split_target") != target_column:
            raise PipelineError("Declare the model target in the split step before fitting preprocessing.")
        tr_df = ctx.get("train_data")
        if isinstance(tr_df, pd.DataFrame) and target_column in tr_df.columns:
            return tr_df
        df = _resolve_df(ctx, input_key)
        if not fit_full_data and len(df) >= 10 and target_column in df.columns:
            from omniforge.data.splitting import LeakageSafeSplitter, SplitConfig

            split_res = LeakageSafeSplitter(SplitConfig(test_size=0.20, random_seed=42)).split(
                df, target_col=target_column
            )
            ctx.set("_eval_holdout_data", split_res.test_df)
            return split_res.train_df
        return df

    def step_train(
        ctx: PipelineContext,
        input_key: str = "features_data",
        target_column: str = "target",
        model_name: str = "random_forest",
        task_type: str = "binary_classification",
        **kw: Any,
    ) -> Any:
        from omniforge.models.registry import create_model

        fit_full = bool(kw.pop("fit_full_data", False))
        df = _prepare_train_and_holdout(ctx, input_key, target_column, fit_full_data=fit_full).dropna(subset=[target_column])
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

        fit_full = bool(kw.pop("fit_full_data", False))
        df = _prepare_train_and_holdout(ctx, input_key, target_column, fit_full_data=fit_full).dropna(subset=[target_column])
        X = df.drop(columns=[target_column])
        y = df[target_column]
        te_df = ctx.get("test_data", ctx.get("_eval_holdout_data"))
        holdout = None
        if isinstance(te_df, pd.DataFrame) and target_column in te_df:
            observed = te_df.dropna(subset=[target_column])
            if observed.empty:
                raise PipelineError("The independent holdout has no observed target values.")
            holdout = (observed.drop(columns=[target_column]), observed[target_column])
        res = AutoMLPipelineSearch(**kw).fit(X, y, holdout=holdout)
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
        if model is None:
            raise PipelineError("step_evaluate requires a fitted 'trained_model' in PipelineContext.")
        te_df = ctx.get("test_data")
        auto_holdout = ctx.get("_eval_holdout_data")
        if isinstance(te_df, pd.DataFrame) and target_column in te_df.columns:
            df = te_df
            ctx.set("evaluation_split", "test_holdout")
        elif isinstance(auto_holdout, pd.DataFrame) and target_column in auto_holdout.columns:
            df = auto_holdout
            ctx.set("evaluation_split", "auto_holdout")
        else:
            df = _resolve_df(ctx, input_key)
            ctx.set("evaluation_split", "in_sample")
        df = df.dropna(subset=[target_column])
        if df.empty:
            raise PipelineError("Evaluation needs observed target values.")
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
        "split": step_split,
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

    def __init__(self, name: str = "omniforge_pipeline") -> None:
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
        implementation_version: str | None = None,
    ) -> PipelineBuilder:
        """Add a built-in declarative step, custom :class:`PipelineStep`, or Python callable."""
        deps = list(depends_on or [])
        if chain and not deps and self._last_step_id is not None:
            deps = [self._last_step_id]

        if isinstance(step_or_type, PipelineStep):
            step = step_or_type
            if deps:
                step.depends_on = deps
            if implementation_version is not None:
                step.implementation_version = implementation_version
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
                implementation_version=implementation_version or "builtin",
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
                implementation_version=implementation_version,
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
                implementation_version=item.get("implementation_version"),
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
        spec_json_literal = repr(json.dumps(spec, sort_keys=True))
        safe_title = (
            str(dag.name)
            .replace("\\", "\\\\")
            .replace('"""', '\\"\\"\\"')
            .replace("'''", "\\'\\'\\'")
            .replace("\n", " ")
            .replace("\r", " ")
        )
        return (
            f'"""Auto-exported OmniForge Production Pipeline: {safe_title}."""\n\n'
            "from __future__ import annotations\n\n"
            "import json\n\n"
            "from omniforge.pipelines import PipelineBuilder, PipelineExecutor\n\n"
            f"_PIPELINE_SPEC_JSON = {spec_json_literal}\n"
            "PIPELINE_SPEC = json.loads(_PIPELINE_SPEC_JSON)\n\n\n"
            "def run_pipeline(initial_artifacts: dict | None = None):\n"
            f'    """Build and execute the {safe_title!r} DAG with full auditing and reproducibility."""\n'
            "    dag = PipelineBuilder.from_dict(PIPELINE_SPEC)\n"
            "    executor = PipelineExecutor()\n"
            "    return executor.execute(dag, initial_artifacts=initial_artifacts or {})\n\n\n"
            'if __name__ == "__main__":\n'
            "    result = run_pipeline()\n"
            '    print("Pipeline status:", result.status)\n'
            "    for rec in result.step_records:\n"
            '        print(f"  - {rec.step_id} ({rec.step_type}): {rec.status} in {rec.duration_seconds:.3f}s")\n'
        )
