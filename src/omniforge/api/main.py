"""Enterprise FastAPI REST Backend exposing all OmniForge Analytics & AI capabilities."""

from __future__ import annotations

import hashlib
import io
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from omniforge.api.auth import (
    WORKBENCH_AUTH_COOKIE,
    access_controller,
    clear_local_workbench_cookies,
    has_valid_workbench_cookie,
    issue_local_workbench_cookies,
    jwt_manager,
    require_permission,
    should_auto_bootstrap_local_ui,
    verify_api_key,
)
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
from omniforge.config.settings import get_settings
from omniforge.copilot.agent import OmniForgeCopilotAgent
from omniforge.core.constants import RBAC_ROLE_PERMISSIONS
from omniforge.core.exceptions import AuthorizationError, OmniForgeError, SecurityError
from omniforge.core.logging import get_logger
from omniforge.core.security import RequestBodySizeLimitMiddleware, SlidingWindowRateLimitMiddleware
from omniforge.core.types import ModelStage, Principal
from omniforge.core.utils import to_json_object, to_json_safe
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

    settings = get_settings()
    allowed_origins = [o.strip() for o in settings.api.cors_origins if o and o.strip()]
    allow_credentials = bool(settings.api.cors_allow_credentials) and ("*" not in allowed_origins)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Workbench-Token",
            "X-Session-ID",
            "X-CSRF-Token",
        ],
    )
    app.add_middleware(GZipMiddleware, minimum_size=500)
    # Hard request-body cap (413) — prevents memory-exhaustion via huge uploads.
    app.add_middleware(RequestBodySizeLimitMiddleware, max_body_bytes=settings.api.max_request_body_bytes)
    # Enforce the declared `api.rate_limit_per_minute` policy (429 + Retry-After).
    # 0 disables (used by the test-suite to avoid cross-test throttling).
    from omniforge.core.rate_limit import build_rate_limiter

    app.add_middleware(
        SlidingWindowRateLimitMiddleware,
        max_requests_per_minute=settings.api.rate_limit_per_minute,
        limiter=build_rate_limiter(settings.api, max_requests_per_minute=settings.api.rate_limit_per_minute),
    )

    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        if (
            response.status_code == 401
            and request.cookies.get(WORKBENCH_AUTH_COOKIE)
            and not has_valid_workbench_cookie(request)
            and "set-cookie" not in response.headers
        ):
            clear_local_workbench_cookies(response, secure=(request.url.scheme == "https"))
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        # Browser-hardening headers (audit P1-11). CSP ships in Report-Only mode first:
        # violations are surfaced to the browser console/report endpoint without blocking
        # the self-hosted SPA, so it can be enforced later once report traffic is clean.
        if settings.api.csp_report_only_enabled and settings.api.csp_report_only_policy.strip():
            response.headers.setdefault(
                "Content-Security-Policy-Report-Only",
                settings.api.csp_report_only_policy.strip(),
            )
        if settings.api.hsts_enabled and settings.api.hsts_max_age > 0:
            hsts = f"max-age={int(settings.api.hsts_max_age)}"
            if settings.api.hsts_include_subdomains:
                hsts += "; includeSubDomains"
            response.headers.setdefault("Strict-Transport-Security", hsts)
        return response

    from omniforge.api.workbench import parse_model_stage, read_upload_bounded
    from omniforge.api.workbench import router as workbench_router

    app.include_router(workbench_router)

    default_registry = ModelRegistry()
    default_serving = ModelServingEngine(registry=default_registry)
    tenant_lock = threading.RLock()
    tenant_mlops: OrderedDict[str, tuple[ModelRegistry, ModelServingEngine]] = OrderedDict()
    copilot_sessions: OrderedDict[str, tuple[threading.RLock, OmniForgeCopilotAgent]] = OrderedDict()

    def _get_api_mlops(principal: Principal) -> tuple[ModelRegistry, ModelServingEngine]:
        tenant = str(getattr(principal, "tenant_id", "") or "default-tenant")
        if tenant == "default-tenant":
            return default_registry, default_serving
        with tenant_lock:
            if tenant in tenant_mlops:
                tenant_mlops.move_to_end(tenant)
                return tenant_mlops[tenant]
            slug = hashlib.sha256(tenant.encode("utf-8", errors="replace")).hexdigest()[:16]
            base = Path("./.omniforge_workspace/tenants") / slug / "api_v1"
            reg = ModelRegistry(
                db_path=base / "model_registry.db",
                artifact_dir=base / "artifacts" / "registry",
            )
            srv = ModelServingEngine(registry=reg)
            tenant_mlops[tenant] = (reg, srv)
            while len(tenant_mlops) > 128:
                tenant_mlops.popitem(last=False)
            return reg, srv

    def _get_api_copilot(principal: Principal) -> tuple[threading.RLock, OmniForgeCopilotAgent]:
        tenant = str(getattr(principal, "tenant_id", "") or "default-tenant")
        subj = str(getattr(principal, "subject", "") or "anonymous")
        key = f"{tenant}\x00{subj}"
        with tenant_lock:
            if key in copilot_sessions:
                copilot_sessions.move_to_end(key)
                return copilot_sessions[key]
            pair = (threading.RLock(), OmniForgeCopilotAgent(provider="local"))
            copilot_sessions[key] = pair
            while len(copilot_sessions) > 128:
                copilot_sessions.popitem(last=False)
            return pair

    @app.exception_handler(OmniForgeError)
    async def uap_error_handler(request: Request, exc: OmniForgeError) -> JSONResponse:
        status_code = 403 if isinstance(exc, SecurityError) else 400
        logger.warning(
            "omniforge_api_error",
            path=str(request.url.path),
            error_code=exc.error_code,
            trace_id=exc.trace_id,
            cause=f"{type(exc.cause).__name__}: {exc.cause}" if exc.cause is not None else None,
        )
        return JSONResponse(
            status_code=status_code,
            content=to_json_safe(exc.to_dict(include_internal=False)),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        error_id = str(uuid.uuid4())
        logger.error(
            "unhandled_api_error",
            error_id=error_id,
            path=str(request.url.path),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        payload: dict[str, Any] = {
            "error_type": "InternalServerError",
            "detail": f"An internal server error occurred. Reference ID: {error_id}",
            "error_id": error_id,
            "path": str(request.url.path),
        }
        return JSONResponse(status_code=500, content=to_json_safe(payload))

    def _resolve_static_dir() -> Path | None:
        local_dir = Path(__file__).resolve().parent.parent / "ui" / "static_app"
        candidates: list[Path] = [local_dir]
        try:
            from omniforge.launcher import is_frozen, resolve_resource_path

            frozen_candidate = resolve_resource_path("omniforge/ui/static_app")
            if is_frozen():
                candidates.insert(0, frozen_candidate)
            else:
                candidates.append(frozen_candidate)
        except Exception:
            pass
        return next((p for p in candidates if p.is_dir() and (p / "index.html").is_file()), None)

    static_dir = _resolve_static_dir()
    spa_bundle_name: str | None = None
    if static_dir is not None and (static_dir / "assets").is_dir():
        js_files = sorted(p.name for p in (static_dir / "assets").glob("index-*.js"))
        spa_bundle_name = js_files[-1] if js_files else None

    @app.get("/health", tags=["System"])
    @app.get("/api/v1/health", tags=["System"])
    def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            **get_version_info(),
            "spa_ready": static_dir is not None,
            "spa_bundle": spa_bundle_name,
        }

    @app.get("/api/v1/version", tags=["System"])
    def version() -> dict[str, Any]:
        return get_version_info()

    @app.post("/api/v1/auth/token", response_model=TokenResponse, tags=["Security"])
    def issue_token(
        req: TokenRequest,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        authorization: str | None = Header(default=None),
    ) -> TokenResponse:
        # Require valid API key (header or body) or existing admin Bearer JWT before minting any token.
        authenticated = False
        auth_via_api_key = False
        caller: Principal | None = None
        if verify_api_key(x_api_key) or verify_api_key(req.api_key):
            authenticated = True
            auth_via_api_key = True
        elif authorization and authorization.lower().startswith("bearer "):
            try:
                candidate_caller = jwt_manager.verify_token(authorization.split(None, 1)[1])
                if {"admin", "platform_admin"} & set(candidate_caller.roles):
                    authenticated = True
                    caller = candidate_caller
            except Exception:
                authenticated = False

        if not authenticated:
            raise HTTPException(
                status_code=401,
                detail="Valid X-API-Key header, body api_key, or admin Bearer token is required to mint a JWT.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if req.role not in RBAC_ROLE_PERMISSIONS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown role '{req.role}'. Valid roles: {sorted(RBAC_ROLE_PERMISSIONS.keys())}",
            )
        # Enforce P1-12 tenant isolation & prevent CWE-269 privilege escalation for tenant-scoped admins
        if not auth_via_api_key and caller is not None and "platform_admin" not in set(caller.roles):
            if req.role == "platform_admin":
                raise HTTPException(
                    status_code=403,
                    detail="Tenant-scoped admin cannot mint platform_admin tokens.",
                )
            if req.tenant_id != caller.tenant_id:
                raise HTTPException(
                    status_code=403,
                    detail=f"Tenant-scoped admin for '{caller.tenant_id}' cannot mint tokens for tenant '{req.tenant_id}'.",
                )
        principal = Principal(subject=req.username, roles=[req.role], tenant_id=req.tenant_id)
        token = jwt_manager.create_token(principal)
        return TokenResponse(access_token=token, role=req.role, tenant_id=req.tenant_id)

    @app.get("/api/system/capabilities", tags=["System"])
    @app.get("/api/v1/system/capabilities", tags=["System"])
    def system_capabilities(
        _principal: Principal = Depends(require_permission("data:read")),
    ) -> dict[str, Any]:
        from omniforge.capabilities import probe_runtime_capabilities

        return to_json_object(probe_runtime_capabilities())

    @app.get("/api/v1/domains", tags=["Domains"])
    def list_domains(
        _principal: Principal = Depends(require_permission("data:read")),
    ) -> dict[str, Any]:
        return {"domains": [p.summary() for p in default_domain_registry.list_plugins()]}

    @app.get("/api/v1/domains/{domain_name}/sample", tags=["Domains"])
    def domain_sample(
        domain_name: str,
        n_samples: int = 50,
        _principal: Principal = Depends(require_permission("data:read")),
    ) -> dict[str, Any]:
        plugin = default_domain_registry.get(domain_name)
        df = plugin.generate_sample_dataset(n_samples=max(10, min(n_samples, 2000)))
        df_feat = plugin.engineer_domain_features(df)
        kpis = plugin.compute_domain_kpis(df_feat)
        return to_json_object(
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
        raw = await read_upload_bounded(file)
        fname = (file.filename or "upload.csv").lower()
        if fname.endswith((".pkl", ".pickle", ".joblib", ".dill")):
            raise HTTPException(
                status_code=400,
                detail="Pickle/joblib uploads are blocked for security (CWE-502). Use CSV, Parquet, or JSON.",
            )
        if fname.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(raw))
        elif fname.endswith(".json"):
            df = pd.read_json(io.BytesIO(raw))
        else:
            df = pd.read_csv(io.BytesIO(raw))
        cop_lock, cop_agent = _get_api_copilot(_principal)
        with cop_lock:
            cop_agent.load_dataframe(df, name=fname)
        quality = DataQualityProfiler().profile(df)
        return to_json_object(
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
        return to_json_object(DataQualityProfiler().profile(df).model_dump())

    @app.post("/api/v1/data/clean", tags=["Data"])
    def clean_data(
        payload: DatasetPayload,
        _principal: Principal = Depends(require_permission("data:write")),
    ) -> dict[str, Any]:
        df = pd.DataFrame(payload.records)
        cleaned, report = DataCleaner().fit_transform(df)
        return to_json_object(
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
        return to_json_object(
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

        reg, srv = _get_api_mlops(_principal)
        reg_info = None
        if req.register_as:
            ver = reg.register_model(
                res.best_model,
                req.register_as,
                metrics={res.primary_metric: res.best_score},
                stage=ModelStage.STAGING,
            )
            reg_info = ver.model_dump()
        if req.deploy_endpoint:
            srv.deploy_model(req.deploy_endpoint, res.best_model, version="v1")

        return to_json_object(
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
        return to_json_object(
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
        return to_json_object(res.model_dump())

    @app.post("/api/v1/optimization/portfolio", tags=["Optimization"])
    def optimize_portfolio(
        req: PortfolioOptimizationRequest,
        _principal: Principal = Depends(require_permission("model:train")),
    ) -> dict[str, Any]:
        res = OptimizationEngine().optimize_portfolio(
            req.expected_returns,
            req.covariance_matrix,
            risk_aversion=req.risk_aversion,
            max_weight=req.max_weight,
        )
        return to_json_object(res.model_dump())

    @app.get("/api/v1/serving/endpoints", tags=["MLOps"])
    def list_endpoints(
        _principal: Principal = Depends(require_permission("data:read")),
    ) -> dict[str, Any]:
        _, srv = _get_api_mlops(_principal)
        return {"endpoints": srv.list_endpoints()}

    @app.post("/api/v1/serving/{endpoint}/predict", tags=["MLOps"])
    def serve_predict(
        endpoint: str,
        req: PredictRequest,
        _principal: Principal = Depends(require_permission("data:read")),
    ) -> dict[str, Any]:
        _, srv = _get_api_mlops(_principal)
        return to_json_object(
            srv.predict(endpoint, req.records, return_probabilities=req.return_probabilities)
        )

    @app.get("/api/v1/registry/models", tags=["MLOps"])
    def list_registered_models(
        model_name: str | None = None,
        _principal: Principal = Depends(require_permission("data:read")),
    ) -> dict[str, Any]:
        reg, _ = _get_api_mlops(_principal)
        return {"versions": [v.model_dump() for v in reg.list_versions(model_name)]}

    @app.post("/api/v1/registry/{model_name}/v{version}/stage", tags=["MLOps"])
    def transition_model_stage(
        model_name: str,
        version: int,
        req: StageTransitionRequest,
        _principal: Principal = Depends(require_permission("model:deploy")),
    ) -> dict[str, Any]:
        stage = parse_model_stage(req.target_stage, error_status=422)
        if stage == ModelStage.PRODUCTION:
            try:
                access_controller.authorize(_principal, "model:promote")
            except AuthorizationError as exc:
                raise HTTPException(status_code=403, detail=exc.message) from exc
        reg, _ = _get_api_mlops(_principal)
        ver = reg.transition_stage(model_name, version, stage)
        return to_json_object(ver.model_dump())

    @app.post("/api/v1/mlops/drift", tags=["MLOps"])
    def check_drift(
        req: DriftRequest,
        _principal: Principal = Depends(require_permission("data:read")),
    ) -> dict[str, Any]:
        ref_df = pd.DataFrame(req.reference_records)
        cur_df = pd.DataFrame(req.current_records)
        return to_json_object(DriftDetector().detect(ref_df, cur_df).model_dump())

    @app.post("/api/v1/copilot/chat", tags=["Copilot"])
    def copilot_chat(
        req: CopilotChatRequest,
        _principal: Principal = Depends(require_permission("data:read")),
    ) -> dict[str, Any]:
        cop_lock, cop_agent = _get_api_copilot(_principal)
        with cop_lock:
            if req.records:
                cop_agent.load_dataframe(pd.DataFrame(req.records), name="api_chat_dataset")
            cop_agent.provider = str(req.provider or "local").lower()
            resp = cop_agent.chat(req.message)
        return to_json_object(resp.model_dump())

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
        return to_json_object(res.to_dict())

    # Serve compiled React/Vite TypeScript SPA (OmniForge Enterprise Workbench)
    if static_dir is not None:
        assets_dir = static_dir / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="spa_assets")

        @app.get("/", include_in_schema=False)
        @app.get("/{full_path:path}", include_in_schema=False)
        def serve_spa(request: Request, full_path: str = "") -> Any:
            if full_path.startswith(("api/", "docs", "redoc", "openapi.json", "health")):
                raise HTTPException(status_code=404, detail="Not found")
            static_root = static_dir.resolve()
            candidate = (static_root / full_path).resolve()
            if full_path and candidate.is_file() and candidate.is_relative_to(static_root):
                return FileResponse(str(candidate))
            resp = FileResponse(
                str(static_root / "index.html"),
                headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
            )
            secure_cookie = request.url.scheme == "https"
            if should_auto_bootstrap_local_ui(request):
                issue_local_workbench_cookies(resp, secure=secure_cookie)
            elif request.cookies.get(WORKBENCH_AUTH_COOKIE) and not has_valid_workbench_cookie(request):
                clear_local_workbench_cookies(resp, secure=secure_cookie)
            return resp
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
