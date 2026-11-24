# OmniForge API, State & Frontend Reference

> Verified against commit `b7dc7ae` (2026-11-24); see `../SKILL.md` for provenance and authority policy.

## FastAPI app — `src/omniforge/api/main.py` (`create_app()`, `app = create_app()`)

Middleware (order matters): `CORSMiddleware` → `GZipMiddleware(500)` →
`RequestBodySizeLimitMiddleware` → `SlidingWindowRateLimitMiddleware` →
`security_headers_middleware` (`nosniff`, `SAMEORIGIN`, `same-origin`
referrer, `Content-Security-Policy-Report-Only`, HSTS). Handlers:
`OmniForgeError` → 403/400 with `to_json_safe(exc.to_dict())`; unhandled →
500 with `error_id` (no stack leak). Tenant-scoped MLOps
(`_get_api_mlops`, LRU 128, per-tenant SQLite under
`.omniforge_workspace/tenants/<sha16>/api_v1/`) and copilot sessions
(`_get_api_copilot`, key `tenant\x00subject`, LRU 128). SPA: `/assets`
static mount + catch-all `/{full_path:path}` with traversal guard
(`is_relative_to`), `no-store` headers, optional cookie bootstrap; fallback
HTML when the bundle is missing.

**Port semantics (clarified):** `create_app()` is one app serving REST **and**
the SPA. `omniforge serve` runs it on :8000 (REST-focused, no browser);
`omniforge ui` runs the same app on :8501 and opens a browser;
`omniforge desktop` / `launch_full` runs **two** instances (API :8000 in a
background thread + workbench :8501 foreground). `launch_api` /
`launch_ui` / `launch_full` are in `launcher.py`; `ui/app.py::
run_workbench_server` is the UI entry. Docker `entrypoint.sh` modes
`api`/`ui`/`worker`/`cli` map to the same.

**`/api/v1/*` routes:** `GET /health`, `GET /api/v1/health`,
`GET /api/v1/version`, `POST /api/v1/auth/token` [Security],
`GET /api/v1/domains`, `GET /api/v1/domains/{domain_name}/sample` [Domains],
`POST /api/v1/data/upload|quality|clean` [Data], `POST /api/v1/eda/run` [EDA],
`POST /api/v1/automl/train` [AutoML], `POST /api/v1/forecast/run`
[Forecasting], `POST /api/v1/causal/estimate` [Causal],
`POST /api/v1/optimization/portfolio` [Optimization],
`GET /api/v1/serving/endpoints`, `POST /api/v1/serving/{endpoint}/predict`,
`GET /api/v1/registry/models`,
`POST /api/v1/registry/{model_name}/v{version}/stage`,
`POST /api/v1/mlops/drift` [MLOps], `POST /api/v1/copilot/chat` [Copilot],
`POST /api/v1/pipelines/execute` [Pipelines]. Schemas: `api/schemas.py`.

## Auth — `api/auth.py`

`jwt_manager`, eagerly-built `access_controller` (mandatory audit, refuses
to boot unaudited). `get_api_key` (settings → env → 0600 keyfile, min 24);
`verify_api_key` (constant-time). Cookies `omniforge_wb_auth`
(`{nonce}.{ts}.{mac}`, 8h, `HttpOnly; SameSite=Strict`) + `omniforge_csrf`;
`issue_local_workbench_cookies`, `should_auto_bootstrap_local_ui`.
`get_current_principal` (Bearer → API key → cookie) sets
`_request_principal_ctx` ContextVar so downstream `_sid` binds tenant+subject
without threading the principal through every call. `require_workbench_auth`,
`require_permission(...)` as `Depends`. Unauthenticated workbench requests
are rejected.

## Jobs — `api/jobs.py`

`BackgroundJobManager(max_workers=4, max_history=128)` (ThreadPoolExecutor);
`submit(name, fn, *args, owner_key, **kwargs)` → `JobRecord`
(`queued/running/completed/failed`); `get(job_id, owner_key)` returns `None`
on owner mismatch (BOLA guard); `list_recent`. Singleton `job_manager`.
Exposed as `GET /api/workbench/jobs`, `GET /jobs/{id}`. Distinct from
`mlops/job_queue.py::BackgroundJobQueue` (async queue + MLflow config).

## Workbench — `api/workbench.py` (1775 lines)

`router = APIRouter(prefix="/api/workbench")`. Helpers: `_current_principal`,
`_tenant_slug`, `get_tenant_registry/tracker`, `parse_model_stage`,
`current_job_owner_key`, `read_upload_bounded`, `_display_session_id`
(strips `ws:<owner_tag>:`), `_sid` (tenant+subject-bound key), `_require_df`,
`_benchmark_datasets_dir`, `_discover_benchmark_csvs`, `_state_summary`,
`_cache_and_respond`, `_infer_task`, `_prepare_xy`. Core endpoints:
`GET /state`, `GET /results`, `GET /benchmarks`,
`POST /theme|targets|exclusions|split-lock`, `DELETE /split-lock`,
`POST /workspace/clear|save|load`, `GET /jobs[/{id}]`, `GET /s01/catalog`,
`DELETE /dataset`, `POST /s01/excel-sheets|upload|sql/demo|sql/inspect|
sql/load|access/inspect|access/load|load-sample`, `GET /s01/grid`,
`POST /s01/quality|profile|clean`, `GET /s01/export-clean/{fmt}`,
`POST /s01/undo|reset-raw|revert`, `GET /s01/deleted-audit|deleted-log|
deleted-audit/csv`, `POST /s02/clustering|clustering/commit|anomaly|
anomaly/commit|mining`. Then includes `s00`–`s14` routers.

## Studio routers — `api/studios/` (all under `/api/workbench`)

Shared schemas: `studios/schemas.py` (47 request classes +
`wb_cache_and_respond`). Sizes verified at `b7dc7ae`: `s09_opt_sim.py` is
873 lines, the largest studio router; next largest is `s14_supervisory.py`
at 521, then `s06_automl.py` at 494.

| File | Routes |
|---|---|
| `s00_ingestion.py` (149) | `GET /s00/data_relationships` |
| `s01_cleaning.py` (50) | `POST /s01/advanced_audit` |
| `s02_mining.py` (60) | `POST /s02/advanced_mining` |
| `s03_eda.py` (80) | `GET /s03/catalog`, `POST /s03/run` |
| `s04_stats.py` (365) | `POST /s04/eda`, `POST /s04/hypothesis` |
| `s05_features.py` (215) | `POST /s05/engineer`, `POST /s05/select` |
| `s06_automl.py` (494) | `GET /s06/backends`, `POST /s06/automl`, `GET /s06/export-formula`, `POST /s06/nonparametric`, `POST /s06/deep-learning`, `POST /s06/predict-single`, `POST /s06/predict-batch` |
| `s07_advanced.py` (108) | `POST /s07/explain`, `POST /s07/fairness` |
| `s08_causal.py` (137) | `POST /s08/forecast`, `POST /s08/causal`, `POST /s08/optimize`, `POST /s08/simulate` |
| `s09_opt_sim.py` (873) | `GET /s09/backends`, `POST /s09/nlp|vision|graph|recommend|survival|twin-control` |
| `s10_rai.py` (188) | `POST /s10/run` (DAG execution) |
| `s11_mlops.py` (148) | `GET /s11/overview`, `POST /s11/register|promote|rollback|drift|root-cause` (`promote` requires `model:promote`) |
| `s12_reports.py` (341) | `GET /s12/catalog`, `POST /s12/compile`, `GET /s12/export/{fmt}` |
| `s13_copilot.py` (54) | `POST /s13/chat` |
| `s14_supervisory.py` (521) | `GET /s14/models`, `GET /s14/equations`, `POST /s14/quick-train|evaluate-file|evaluate-sql|evaluate-access|evaluate-demo-batch|evaluate-single|root-cause`, `GET /s14/export/{fmt}` |

## Workspace state lifecycle — `ui/state.py` + `core/workspace_session.py`

`ServerSessionRegistry(ttl=86400, max_sessions=128)` maps browser
`session_id` → per-session dict from `default_workspace_dict()` (~90 `uap_*`
keys: datasets, targets/exclusions, split-lock, deleted-row log,
`uap_studio_results`, models/equations, supervisory history, DAG, report,
lineage, theme, project). **The lifecycle of a state key (read this before
adding one):**

1. **Declare** the key in `default_workspace_dict()` with a safe default
   (`None` / `{}` / `[]`).
2. **Write** via the owning function (`load_new_dataset`,
   `lock_leakage_safe_split`, `save_studio_result`, …) — never mutate from a
   router directly without recording lineage (`record_lineage`) and, where
   relevant, exclusions (`record_exclusion`).
3. **Invalidate downstream:** dataset/target/split changes call
   `_reset_downstream_state` (clears `_DOWNSTREAM_KEYS`: models, equations,
   reports, metrics — dicts reset to `{}`, lists to `[]`, else `None`) and
   `_prune_studio_results_for_stage` (keeps upstream `s01_/s02_/…` prefixes
   per `_STUDIO_RESULT_KEEP_PREFIXES`). If your key is downstream of
   dataset/target/split, add it to `_DOWNSTREAM_KEYS`.
4. **Persist:** `.uap` export includes a key only if
   `workspace_session._should_persist_key` returns true — key must start with
   a persisted prefix (`_PERSISTED_KEY_PREFIXES`), not be in
   `_EXCLUDED_EXACT_KEYS` / `_EXCLUDED_KEY_PATTERNS`, not start with `_`,
   and not be callable or an uploaded-file object. If your key must survive
   save/load, verify it passes this filter; if it must *not* survive (live
   engine handles, file streams), ensure it is excluded.
5. **Surface:** `workflow_completion_status` / `studio_completion_map` /
   `_detect_completed_stages` derive checklist progress from key presence —
   add your key there if it represents a completed stage.
6. **Frontend:** mirror the key in `WorkspaceState` (`frontend/src/types.ts`)
   and the payload type in `studioTypes.ts`.

`ui/app.py` — `create_workbench_app()` → `create_app()`;
`run_workbench_server(host, port, debug, open_browser)` (uvicorn + browser
thread); legacy aliases `create_dash_app` / `run_dash_server`.

## Frontend — `frontend/`

React 18.3, `lucide-react`, Vite 6, TypeScript 5.7 (`strict`, `noEmit`,
`jsx: react-jsx`). Scripts: `dev` (:5173, proxies `/api` + `/health` →
`127.0.0.1:8501`), `build` (`tsc && vite build`), `preview`.
`vite.config.ts`: `outDir = ../src/omniforge/ui/static_app`,
`emptyOutDir`, `target es2020`, `manualChunks` → `react-vendor`,
`icons-vendor`, `vendor`. Committed bundle: `index.html` +
`assets/{index-*.js/css, react-vendor-*.js, icons-vendor-*.js, vendor-*.js,
StudiosPart1/2/3-*.js, Studio14Supervisory-*.js, HelpGuideModal-*.js}`.

| File | Role |
|---|---|
| `main.tsx` | React root. |
| `App.tsx` (568) | Shell: sidebar (15 studios), header (project, handbook, theme, Studio 14, intake inspector), toast/modal, **keep-alive mounting** (`visitedStudios` + `display:none`), `React.lazy` per part, server-cache hydration (`GET /results`). |
| `api.ts` (384) | `apiGet/Post/Delete/Upload/Download`; `ApiError` (`retryable` on timeout/network/429/5xx); `DEFAULT_TIMEOUTS` (GET 30s, POST 300s, up/down 120s); `getSessionId()` (`of_`+UUID, localStorage); CSRF from `omniforge_csrf`; API-key/Bearer from localStorage; `AbortSignal.timeout/any`. |
| `types.ts` (153) | `StudioId` `'00'`–`'14'`, `STUDIOS` (15), `WorkspaceState`, `SplitLockSummary`, `IntakeAuditSummary`, `GridResponse`, `ScatterPayload`. |
| `studioTypes.ts` (797) | Per-studio payload types. **Contract:** keep in sync with `workbench.py` — a renamed key silently renders blanks. All fields optional (untrusted server data). |
| `studios/StudiosPart1.tsx` (1663) | Studios 00–05 + `StudioProps`, `ProvenancePill`. |
| `studios/StudiosPart2.tsx` (637) | Studios 06–08. |
| `studios/StudiosPart3.tsx` (710) | Studios 09–13. |
| `studios/Studio14Supervisory.tsx` (899) | Studio 14 (separate chunk). |
| `components/Charts.tsx` (611) | `CanvasScatter`, `BarChartSVG`, `ForecastFanSVG`, `HeatmapGrid`, `NetworkGraphSVG`, `SupervisoryTrendSVG`, `SPCControlSVG`, `DivergingDriverBarSVG` (SVG/canvas, no chart lib). |
| `components/DataGrid.tsx` (174) | `PaginatedDataGrid`, `SimpleTable`. |
| `components/HelpGuideModal.tsx` (1040) | `HelpGuideModal`, `StudioHelpButton`, `StudioChapterView`. |
| `components/InspectModal.tsx` (316) | `InspectableBox` (safe URLs, sanitized clone). |
| `components/MathEquation.tsx` (462) | `MathEquationCard`. |
| `components/ModeBar.tsx` (33) | `ExecutionMode` (`auto/complete/custom`). |
| `data/handbook*.ts` (~2900) | `handbookTypes` (`FormulaSpec`, `MethodEntry`, `ThresholdTable`, `DecisionMatrix`, `TheorySection`, `WorkedExample`), `handbookData` (`ALL_STUDIO_CHAPTERS`, `CHAPTER_BY_STUDIO`), Parts 1–2 + Deep 1–2. |
| `styles.css` | Dark/light themes. |

Rules: relative `/api/...` paths only (never `localhost` in browser code);
stable `commonProps` to avoid remounts; rebuild + commit `static_app` after
any change (see `packaging.md` for why the bundle is committed).
