# OmniForge API, state and frontend reference

> The baseline revision reviewed is pinned in [SKILL.md](../SKILL.md).
> The workspace lifecycle and Split Lock notes below also include partial
> working-tree updates; the full skill has not been re-reviewed. Source paths
> are relative to the repository root. Verify actual handlers before use.

## Application and route map

`src/omniforge/api/main.py::create_app()` builds the FastAPI app with REST,
`/docs`, `/redoc`, static assets and SPA catch-all. `omniforge serve` runs
that app on port 8000 by default; `omniforge ui` runs the same API and SPA on
8501, normally opening a browser. `omniforge desktop` starts the API listener
in a background thread and the UI listener in the foreground, within one
process. Defaults bind to loopback. `src/omniforge/ui/app.py` provides the
workbench server and retains legacy Dash-named aliases.

- Public system routes include `GET /health`, `GET /api/v1/health` and
  `GET /api/v1/version`. `POST /api/v1/auth/token` requires an API key or
  authorized admin token to mint a token. Other `/api/v1/*` handlers declare
  permissions individually in `src/omniforge/api/main.py` and schemas in
  `src/omniforge/api/schemas.py`.
  Route groups include domains, data, EDA, AutoML, forecasting, causal,
  optimization, serving, registry, drift, Copilot and pipelines.
- `src/omniforge/api/workbench.py::router` includes `require_workbench_auth` for all
  `/api/workbench/*` routes and then includes the studio routers. That is an
  authentication requirement, **not** a blanket action-specific permission
  check. For example, Studio 11 promotion explicitly requires
  `model:promote`, while adjacent registration and rollback handlers do not
  declare that dependency (`src/omniforge/api/studios/s11_mlops.py`). Inspect each handler.
- Common workbench endpoints include `GET /state`, `GET /results`,
  `POST /targets`, `POST /exclusions`, `POST`/`DELETE /split-lock`,
  `POST /workspace/{save,load,clear}`, `GET /jobs`, `GET /jobs/{job_id}`,
  `POST /jobs/{job_id}/cancel`, and `GET /feature-store/freshness`, all under
  `/api/workbench`. Dataset upload/cleaning and several `s02` routes are in
  `src/omniforge/api/workbench.py`; the other studio handlers live in
  `src/omniforge/api/studios/`.
- `src/omniforge/capabilities.py` mentions `/api/system/capabilities` and
  frontend install buttons in comments, but no such implemented API route or
  wired install button was found at this revision. Do not build an integration
  on that comment without implementing and testing it.

Router basenames in this table are under `src/omniforge/api/studios/`.

| Router | Representative paths under `/api/workbench` |
|---|---|
| `s00_ingestion.py` | `GET /s00/data_relationships` |
| `s01_cleaning.py`, `s02_mining.py` | `POST /s01/advanced_audit`, `/s02/advanced_mining` |
| `s03_eda.py`, `s04_stats.py` | `GET /s03/catalog`; `POST /s03/run`, `/s04/eda`, `/s04/hypothesis` |
| `s05_features.py` | `POST /s05/engineer`, `/s05/select` |
| `s06_automl.py` | `GET /s06/backends`, `/s06/export-formula`; `POST /s06/automl`, `/s06/nonparametric`, `/s06/deep-learning`, `/s06/predict-single`, `/s06/predict-batch` |
| `s07_advanced.py`, `s08_causal.py` | explain/fairness and forecast/causal/optimize/simulate |
| `s09_opt_sim.py`, `s10_rai.py` | specialized NLP/vision/graph/recommendation/survival/twin control and `/s10/run` DAG |
| `s11_mlops.py` | overview, register, promote, rollback, drift and root-cause |
| `s12_reports.py`, `s13_copilot.py` | report catalog/compile/export and Copilot chat |
| `s14_supervisory.py` | models/equations, quick-train, evaluate and export |

## Ownership and workspace lifecycle

`src/omniforge/api/auth.py::get_current_principal` authenticates Bearer JWT, API key or a
local cookie and stores the principal in a request context. `_sid` in
`src/omniforge/api/workbench.py` namespaces the client-provided `X-Session-ID` by a hash of
both tenant and subject. If called internally without an authenticated
context, `_sid` retains legacy unscoped behavior; route protection matters.
`src/omniforge/ui/state.py::ServerSessionRegistry` stores workspaces in process memory
(default TTL 86,400 seconds, maximum 128 sessions). A restart/eviction loses
unsaved state; multiple API processes do not automatically share it.

State keys and caches are not governed by one universal invalidation rule:

1. `default_workspace_dict()` supplies the initial `uap_*` keys.
   `load_new_dataset()` and `clear_workspace_dataset()` prune dataset-dependent
   results while retaining standalone NLP and vision results. Dataset updates
   reset/prune downstream state using `_STUDIO_RESULT_KEEP_PREFIXES`; committed
   clustering/anomaly charts are labelled as historical previews.
2. `set_selected_targets()`, `set_excluded_columns()`,
   `lock_leakage_safe_split()` and `clear_split_lock()` invalidate affected
   caches and models. Full-data analyses and historical provenance may be
   retained when still applicable. Dataset updates re-align locked positions
   using explicit unique row IDs or row labels from a transformation whose
   caller attests it preserves those labels. Ambiguous identity, invalid
   membership or a failed re-audit unlocks the split; initial lock creation can
   still report an audit warning. Verify each producer's dependencies rather
   than assuming every cached result is valid after every mutation.
3. `save_studio_result()` and `wb_cache_and_respond()` support frontend
   hydration. Routers also directly mutate state; inspect their write paths,
   lineage and exclusions rather than assuming all writes use a single helper.
4. `src/omniforge/core/workspace_session.py::_should_persist_key` filters prefixes,
   transient patterns, callables and upload objects. Export also probes
   serializability and can skip keys. When adding a key, test defaults,
   ownership, invalidation, stage preservation, serialization/restore,
   completion indicators and any frontend representation independently.

Exports contain sensitive workspace data and models. New downloads use
`.omniforge`; `.uap` is recognized as a legacy extension. Default loading
requires a valid payload HMAC and restricted deserialization, and simply
inspecting the manifest is not signature verification. See
[security](security.md) and [packaging](packaging.md).

## Split Lock, training and inference

`get_active_split_frames()` returns locked train/test partitions when an
active lock is valid, or the full dataframe as train with no lock when it is
inactive. An invalid active lock can raise rather than fall back. Behavior
differs by route:

- Studio 01 cleaning uses `DataCleaner.fit_transform_split` when a locked
  train/test pair is available; inspect the chosen cleaning mode and row
  mutations (`src/omniforge/api/workbench.py`). Studio 05 engineering fits the
  locked train frame when available; without a lock it fits the full dataset
  and reports a leakage warning. Feature selection also has a separate
  full-data preparation path; do not infer that its entire evaluation pipeline
  is leakage-free from the lock alone.
- Studio 06 AutoML uses locked train as its input when a valid lock exists,
  otherwise the full analysis dataframe
  (`src/omniforge/api/studios/s06_automl.py::s06_run_automl`).
  This is a holdout boundary, not proof that every backend's internal CV is
  fold-safe. `uap_active_X_test` and `uap_active_y_test` are assigned a tail
  of the AutoML input, not the locked test partition; do not treat those keys
  as an independent test set.
- Studio 06 **nonparametric** fits `_prepare_xy_fitted()` on the full analysis
  dataframe before `NonParametricAnalyticalSuite` creates CV folds. Studio 14
  **quick-train** fits it on the full analysis dataframe before its 80/20
  positional split. Neither handler calls `get_active_split_frames`. Numeric
  medians and category vocabularies can therefore include evaluation rows;
  Split Lock does not make these scores leakage-free. Fit within each training
  fold/partition and add adversarial held-out-category and median tests before
  making a stronger claim.

`src/omniforge/features/inference.py::WorkbenchInputPreprocessor` remembers input column
order and types. In numeric mode it retains fitted numeric/datetime medians
and categorical encodings; unseen one-hot categories produce zeros and unseen
high-cardinality categories use `-1`. In raw mode it normalizes the schema
and leaves encoding/imputation to the trained backend. Studio 06
`_prediction_frame()` uses the stored preprocessor for **both** single and
batch scoring; malformed values become 422 errors. Old workspaces without a
preprocessor use historic numeric alignment and zero-fill until retrained.
The batch path reads CSV as strings before applying the fitted schema. See
`tests/unit/test_workbench_inference_preprocessing.py`.

## Background jobs and feature freshness

`src/omniforge/api/jobs.py::BackgroundJobManager` runs callables in a process-local thread
pool, with bounded local/shared queue capacity. The configured manager uses
`settings.jobs.db_path` or `paths.workspace_dir/workbench_jobs.db`; a directly
constructed manager without a path uses in-memory SQLite. SQLite stores
owner-scoped status, cancellation and JSON results, **not** callables or
workspaces. The workbench owner key hashes tenant and subject;
`platform_admin` queries use the separate unscoped path in
`current_job_owner_key(for_query=True)`, including job cancellation.

Queued work cancels immediately. Running opted-in work cancels cooperatively
at checkpoints; after publication begins, cancellation is too late. Studio 06
AutoML supports this when submitted with `async_job`; a full queue returns
429. A lost/expired worker causes its unfinished jobs to be marked failed,
not resumed. Shared status across workers requires the same database and
reliable SQLite locking; workspace operations also require sticky routing or
an implemented shared workspace store. `src/omniforge/mlops/job_queue.py` is separate and
in-memory. Test cross-process visibility and failure as well as single-process
behavior (`tests/unit/test_persistent_workbench_{jobs,processes}.py`).

`LocalFeatureStore.freshness_report()` reports current registered views and
online vectors without exposing entity keys or feature values. For TTL-enabled
views it counts expired and unknown timestamps separately. It is not a historical
expiration counter: purged rows are absent, and TTL-disabled views count all
current rows as fresh without checking their timestamps. The
tenant-scoped endpoint backs `frontend/src/components/FeatureFreshnessPanel.tsx`
in Studios 11 and 14; each mounted panel refreshes every 60 seconds and has a
manual refresh button. See `tests/unit/test_feature_freshness_monitoring.py`.

## React/Vite frontend

- `frontend/src/App.tsx` uses lazy studio chunks and keeps visited studios
  mounted while hiding inactive ones; `frontend/src/types.ts` defines navigation
  and `WorkspaceState`, and `frontend/src/studioTypes.ts` describes payloads.
- `frontend/src/api.ts` provides GET/POST/DELETE/upload/download helpers,
  supplies a localStorage session ID, and attaches configured API-key/Bearer
  credentials and the CSRF cookie value. Review browser storage exposure and
  avoid logging those values.
- `frontend/vite.config.ts` binds the dev server to `0.0.0.0:5173` and
  proxies `/api` and `/health` to the server-side `127.0.0.1:8501`. Browser
  requests should use relative paths; the browser cannot reach a sandbox's
  localhost directly. For proxied previews, also verify Vite accepts the
  preview host/origin; binding alone is not sufficient.
- `npm run build` type-checks and writes the committed
  `src/omniforge/ui/static_app/` bundle. Rebuild after frontend changes and
  inspect the diff. CI verifies the committed bundle against the Vite build.
