"""Enterprise FastAPI REST Backend exposing all OmniForge Analytics & AI capabilities."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from omniforge.api.auth import jwt_manager, require_permission
from omniforge.api.schemas import (
    AutoMLRequest,
    CausalRequest,
    CopilotChatRequest,
    DatasetPayload,
    DriftRequest,
    ForecastRequest,
    PipelineExecuteRequest,
    PortfolioOptimizationRequest,
    PredictRequest,
    StageTransitionRequest,
    TokenRequest,
    TokenResponse,
)
from omniforge.automl.pipeline_search import AutoMLPipelineSearch
from omniforge.causal.inference import CausalInferenceEngine
from omniforge.copilot.agent import OmniForgeCopilotAgent
from omniforge.core.constants import RBAC_ROLE_PERMISSIONS
from omniforge.core.exceptions import OmniForgeError
from omniforge.core.logging import get_logger
from omniforge.core.types import ModelStage, Principal
from omniforge.core.utils import to_json_safe
from omniforge.data.cleaning import DataCleaner
from omniforge.data.quality import DataQualityProfiler
from omniforge.domains import default_domain_registry
from omniforge.eda.automated_eda import AutomatedEDA
from omniforge.forecasting.engine import ForecastingEngine
from omniforge.mlops.drift import DriftDetector
from omniforge.mlops.registry import ModelRegistry
from omniforge.mlops.serving import ModelServingEngine
from omniforge.optimization.solvers import OptimizationEngine
from omniforge.pipelines.builder import PipelineBuilder
from omniforge.pipelines.executor import PipelineExecutor
from omniforge.version import get_version_info

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the enterprise OmniForge FastAPI application."""
    app = FastAPI(
        title="OmniForge Enterprise Analytics & AI API",
        description="Production REST API for multi-source data, automatic format auditing, EDA, AutoML, Forecasting, Causal, Optimization, Responsible AI, MLOps, and AI Copilot.",
        version=get_version_info()["version"],
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=500)

    from omniforge.api.workbench import router as workbench_router

    app.include_router(workbench_router)

    registry = ModelRegistry()
    serving = ModelServingEngine(registry=registry)
    copilot = OmniForgeCopilotAgent(provider="local")

    @app.exception_handler(OmniForgeError)
    async def uap_error_handler(request: Request, exc: OmniForgeError) -> JSONResponse:
        return JSONResponse(status_code=400, content=to_json_safe(exc.to_dict()))

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled_api_error",
            path=str(request.url.path),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content=to_json_safe(
                {
                    "error_type": type(exc).__name__,
                    "detail": f"{type(exc).__name__}: {exc}",
                    "message": str(exc),
                    "path": str(request.url.path),
                }
            ),
        )

    @app.get("/health", tags=["System"])
    @app.get("/api/v1/health", tags=["System"])
    def health() -> dict[str, Any]:
        return {"status": "healthy", **get_version_info()}

    @app.get("/api/v1/version", tags=["System"])
    def version() -> dict[str, Any]:
        return get_version_info()

    @app.post("/api/v1/auth/token", response_model=TokenResponse, tags=["Security"])
    def issue_token(req: TokenRequest) -> TokenResponse:
        if req.role not in RBAC_ROLE_PERMISSIONS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown role '{req.role}'. Valid roles: {sorted(RBAC_ROLE_PERMISSIONS.keys())}",
            )
        principal = Principal(subject=req.username, roles=[req.role], tenant_id=req.tenant_id)
        token = jwt_manager.create_token(principal)
        return TokenResponse(access_token=token, role=req.role, tenant_id=req.tenant_id)

    @app.get("/api/v1/domains", tags=["Domains"])
    def list_domains() -> dict[str, Any]:
        return {"domains": [p.summary() for p in default_domain_registry.list_plugins()]}

    @app.get("/api/v1/domains/{domain_name}/sample", tags=["Domains"])
    def domain_sample(domain_name: str, n_samples: int = 50) -> dict[str, Any]:
        plugin = default_domain_registry.get(domain_name)
        df = plugin.generate_sample_dataset(n_samples=max(10, min(n_samples, 2000)))
        df_feat = plugin.engineer_domain_features(df)
        kpis = plugin.compute_domain_kpis(df_feat)
        return to_json_safe(
            {
                "domain": plugin.metadata.name,
                "kpis": kpis,
                "n_rows": len(df_feat),
                "columns": list(df_feat.columns),
                "preview": df_feat.head(10).to_dict(orient="records"),
            }
        )

    @app.post("/api/v1/data/upload", tags=["Data"])
    async def upload_dataset(
        file: UploadFile = File(...),
        _principal: Principal = Depends(require_permission("data:write")),
    ) -> dict[str, Any]:
        raw = await file.read()
        fname = (file.filename or "upload.csv").lower()
        if fname.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(raw))
        elif fname.endswith(".json"):
            df = pd.read_json(io.BytesIO(raw))
        else:
            df = pd.read_csv(io.BytesIO(raw))
        copilot.load_dataframe(df, name=fname)
        quality = DataQualityProfiler().profile(df)
        return to_json_safe(
            {
                "filename": file.filename,
                "rows": len(df),
                "columns": list(df.columns),
                "quality": quality.model_dump(),
            }
        )

    @app.post("/api/v1/data/quality", tags=["Data"])
    def data_quality(
        payload: DatasetPayload,
        _principal: Principal = Depends(require_permission("data:read")),
    ) -> dict[str, Any]:
        df = pd.DataFrame(payload.records)
        return to_json_safe(DataQualityProfiler().profile(df).model_dump())

    @app.post("/api/v1/data/clean", tags=["Data"])
    def clean_data(
        payload: DatasetPayload,
        _principal: Principal = Depends(require_permission("data:write")),
    ) -> dict[str, Any]:
        df = pd.DataFrame(payload.records)
        cleaned, report = DataCleaner().fit_transform(df)
        return to_json_safe(
            {
                "report": report.model_dump(),
                "cleaned_records": cleaned.to_dict(orient="records"),
            }
        )

    @app.post("/api/v1/eda/run", tags=["EDA"])
    def run_eda(
        payload: DatasetPayload,
        target_column: str | None = None,
        _principal: Principal = Depends(require_permission("data:read")),
    ) -> dict[str, Any]:
        df = pd.DataFrame(payload.records)
        rep = AutomatedEDA().run(df, target_column=target_column, generate_figures=False)
        return to_json_safe(
            {
                "profile": rep.profile.model_dump(),
                "insights": [i.model_dump() for i in rep.insights],
                "narrative": rep.narrative,
            }
        )

    @app.post("/api/v1/automl/train", tags=["AutoML"])
    def train_automl(
        req: AutoMLRequest,
        _principal: Principal = Depends(require_permission("model:train")),
    ) -> dict[str, Any]:
        df = pd.DataFrame(req.records)
        if req.target_column not in df.columns:
            raise HTTPException(status_code=422, detail=f"Target column '{req.target_column}' missing.")
        X = df.drop(columns=[req.target_column])
        y = df[req.target_column]
        res = AutoMLPipelineSearch(
            time_budget_seconds=req.time_budget_seconds, tune_trials=4, include_deep=True
        ).fit(X, y)

        reg_info = None
        if req.register_as:
            ver = registry.register_model(
                res.best_model,
                req.register_as,
                metrics={res.primary_metric: res.best_score},
                stage=ModelStage.STAGING,
            )
            reg_info = ver.model_dump()
        if req.deploy_endpoint:
            serving.deploy_model(req.deploy_endpoint, res.best_model, version="v1")

        return to_json_safe(
            {
                "best_model_name": res.best_model_name,
                "task_type": res.task_type.value,
                "primary_metric": res.primary_metric,
                "best_score": res.best_score,
                "lift_over_best_baseline": res.lift_over_best_baseline,
                "leaderboard": res.leaderboard,
                "carbon_report": res.carbon_report.model_dump() if res.carbon_report else None,
                "registered_version": reg_info,
                "deployed_endpoint": req.deploy_endpoint,
            }
        )

    @app.post("/api/v1/forecast/run", tags=["Forecasting"])
    def run_forecast(
        req: ForecastRequest,
        _principal: Principal = Depends(require_permission("model:train")),
    ) -> dict[str, Any]:
        df = pd.DataFrame(req.records)
        rep = ForecastingEngine(include_deep=False).forecast(
            df,
            target_col=req.target_column,
            time_col=req.time_column if (req.time_column and req.time_column in df.columns) else None,
            horizon=req.horizon,
        )
        return to_json_safe(
            {
                "champion_model": rep.champion_model,
                "leaderboard": rep.leaderboard,
                "forecast": rep.forecast.to_dict(orient="records"),
            }
        )

    @app.post("/api/v1/causal/estimate", tags=["Causal"])
    def estimate_causal(
        req: CausalRequest,
        _principal: Principal = Depends(require_permission("model:train")),
    ) -> dict[str, Any]:
        df = pd.DataFrame(req.records)
        res = CausalInferenceEngine(bootstrap_rounds=25).estimate_ate(
            df,
            treatment_col=req.treatment_col,
            outcome_col=req.outcome_col,
            confounders=req.confounders,
            method=req.method,
        )
        return to_json_safe(res.model_dump())

    @app.post("/api/v1/optimization/portfolio", tags=["Optimization"])
    def optimize_portfolio(req: PortfolioOptimizationRequest) -> dict[str, Any]:
        res = OptimizationEngine().optimize_portfolio(
            req.expected_returns,
            req.covariance_matrix,
            risk_aversion=req.risk_aversion,
            max_weight=req.max_weight,
        )
        return to_json_safe(res.model_dump())

    @app.get("/api/v1/serving/endpoints", tags=["MLOps"])
    def list_endpoints() -> dict[str, Any]:
        return {"endpoints": serving.list_endpoints()}

    @app.post("/api/v1/serving/{endpoint}/predict", tags=["MLOps"])
    def serve_predict(endpoint: str, req: PredictRequest) -> dict[str, Any]:
        return to_json_safe(
            serving.predict(endpoint, req.records, return_probabilities=req.return_probabilities)
        )

    @app.get("/api/v1/registry/models", tags=["MLOps"])
    def list_registered_models(model_name: str | None = None) -> dict[str, Any]:
        return {"versions": [v.model_dump() for v in registry.list_versions(model_name)]}

    @app.post("/api/v1/registry/{model_name}/v{version}/stage", tags=["MLOps"])
    def transition_model_stage(
        model_name: str,
        version: int,
        req: StageTransitionRequest,
        _principal: Principal = Depends(require_permission("model:deploy")),
    ) -> dict[str, Any]:
        ver = registry.transition_stage(model_name, version, ModelStage(req.target_stage))
        return to_json_safe(ver.model_dump())

    @app.post("/api/v1/mlops/drift", tags=["MLOps"])
    def check_drift(req: DriftRequest) -> dict[str, Any]:
        ref_df = pd.DataFrame(req.reference_records)
        cur_df = pd.DataFrame(req.current_records)
        return to_json_safe(DriftDetector().detect(ref_df, cur_df).model_dump())

    @app.post("/api/v1/copilot/chat", tags=["Copilot"])
    def copilot_chat(req: CopilotChatRequest) -> dict[str, Any]:
        if req.records:
            copilot.load_dataframe(pd.DataFrame(req.records), name="api_chat_dataset")
        copilot.provider = req.provider
        resp = copilot.chat(req.message)
        return to_json_safe(resp.model_dump())

    @app.post("/api/v1/pipelines/execute", tags=["Pipelines"])
    def execute_pipeline(
        req: PipelineExecuteRequest,
        _principal: Principal = Depends(require_permission("pipeline:execute")),
    ) -> dict[str, Any]:
        dag = PipelineBuilder.from_dict(req.pipeline_spec)
        initial: dict[str, Any] = {}
        if req.records:
            initial["data"] = pd.DataFrame(req.records)
        res = PipelineExecutor().execute(dag, initial_artifacts=initial)
        return to_json_safe(res.model_dump())

    # Serve compiled React/Vite TypeScript SPA (OmniForge Enterprise Workbench)
    static_candidates = [
        Path(__file__).resolve().parent.parent / "ui" / "static_app",
    ]
    try:
        from omniforge.launcher import resolve_resource_path

        static_candidates.insert(0, resolve_resource_path("omniforge/ui/static_app"))
    except Exception:
        pass
    static_dir = next((p for p in static_candidates if p.is_dir() and (p / "index.html").is_file()), None)
    if static_dir is not None:
        assets_dir = static_dir / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="spa_assets")

        @app.get("/", include_in_schema=False)
        @app.get("/{full_path:path}", include_in_schema=False)
        def serve_spa(full_path: str = "") -> Any:
            if full_path.startswith(("api/", "docs", "redoc", "openapi.json", "health")):
                raise HTTPException(status_code=404, detail="Not found")
            candidate = (static_dir / full_path).resolve()
            if full_path and candidate.is_file() and str(candidate).startswith(str(static_dir.resolve())):
                return FileResponse(str(candidate))
            return FileResponse(
                str(static_dir / "index.html"),
                headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
            )
    else:

        @app.get("/", include_in_schema=False)
        def spa_missing() -> HTMLResponse:
            return HTMLResponse(
                "<html><body style='font-family:sans-serif;background:#0b1120;color:#e2e8f0;padding:40px'>"
                "<h1>OmniForge FastAPI Backend Active</h1>"
                "<p>Compiled React SPA not found at <code>src/omniforge/ui/static_app</code>. "
                "Visit <a href='/docs' style='color:#38bdf8'>/docs</a> for interactive API docs.</p>"
                "</body></html>"
            )

    return app


app = create_app()
