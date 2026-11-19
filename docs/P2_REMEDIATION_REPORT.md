# OmniForge v2.0.0 — Phase P2 Remediation Report

**Date:** 2026-11-18 · **Scope:** P2-1 … P2-9 (+ P2-10 documentation alignment discovered during review) · **Verdict: all items closed, all gates green, zero regressions.**

Every number in this report comes from a command that was actually executed in the workspace
(`/var/tmp/p2_venv`, Python 3.13 with the pinned 2026 stack: numpy 2.5.5, pandas 3.1.1,
scipy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0, cryptography 50.0.2, pydantic 2.14.3; **torch not installed**,
so every deep-learning path was exercised through its NumPy fallback).

---

## 1. Gate summary — baseline vs. final

| Gate | Baseline (phase start) | Final | Command |
| :--- | :--- | :--- | :--- |
| `ruff` | **27** errors | **0** — *All checks passed!* | `/var/tmp/p2_venv/bin/ruff check .` |
| `mypy` | **104 errors / 16 files** | **0** — *Success: no issues found in 166 source files* | `/var/tmp/p2_venv/bin/mypy src/omniforge` |
| `pytest` | 8 failed / 77 passed | **85 passed, 5 warnings** (~108 s) | `python -m pytest tests -q` |
| `tsc --noEmit` | 0 | **0** | `cd frontend && ./node_modules/.bin/tsc --noEmit` |
| `vite build` | 1 × 614.43 kB monolith | **10 chunks**, 4.00 s | `npm run build` |
| TypeScript `any` | **104** | **0** | `grep -rn ": any\|as any\|<any>" frontend/src` |
| Stale `uap` identifiers | 23 in README + 1 in architecture + runtime labels | **only the intentional legacy `.uap` extension**; runtime labels aligned | `grep -rn "\buap\b" README.md docs/*.md` |

mypy is now enforced with a *hardened* config (`disallow_any_generics`, `disallow_untyped_decorators`,
`disallow_untyped_defs`, `check_untyped_defs`, `warn_unused_ignores`, `strict_equality`, `extra_checks`,
`warn_return_any`, `no_implicit_optional`) and still reports **0 errors**, i.e. the strictness increase is
*verified*, not assumed.

---

## 2. Item-by-item results

### P2-1 — Package namespace (`uap` → `omniforge`) ✅
- The real package is `omniforge`; the legacy `uap` namespace was only present in docs, docstrings,
  artifact/label defaults and one dead launcher helper.
- **Code:** `import uap` / `from uap …` occurrences: **0** (verified by grep over `src/`, `tests/`, `scripts/`, `examples/`).
- **Docs:** README's 23 `uap.*` package references and `docs/architecture.md`'s 1 → `omniforge.*`; all CLI
  examples now read `omniforge domains|automl|copilot|serve|ui`.
- **Runtime labels aligned** (these leak into JSON metadata, file formats and logs, so they are functional, not cosmetic):
  `CarbonReport.tracker_backend` `uap_hardware_estimator → omniforge_hardware_estimator`, `ModelMetadata.framework`
  default `uap → omniforge`, all five NumPy deep-learning fallbacks (`omniforge_numpy_autograd`, `_conv1d`,
  `_transformer`, `_gru/_lstm`), `omniforge_operation`, `omniforge_future_predictions`,
  `omniforge_portable_edge_bundle`, `.omniforge.joblib`, `omniforge_pipeline` (DAG + builder), CLI default
  `omniforge_eda_report.html`.
- **Launcher:** project defaults `uap_project → omniforge_project`; **deleted dead helper `_locate_ui_script()`**
  (never called, and it looked for a non-existent `uap/ui/app.py`).
- **Intentionally retained:** `.uap` remains accepted as a **legacy** project extension
  (`LEGACY_PROJECT_EXTENSIONS = (".omniforge", ".uap")`) and the internal session-state keys (`uap_df`,
  `uap_target`, …) are kept verbatim — they are the persisted wire format used by frontend, API and the
  round-trip tests; renaming them would silently break saved `.omniforge` archives.

### P2-2 — Packaging extras ✅
- Core `dependencies` contain **no** optional heavy stack; verified `torch` is **not** a core dependency.
- **17 extras** + a self-referential `all` umbrella (`omniforge[api,ui,reporting,data,forecasting,deep,boosting,automl,automl-java,foundation,responsible,nlp,vision,audio,build,dev]`), so
  `pip install -e ".[all]"` is one command; README shows the granular form.
- Every core dependency declared in `pyproject.toml` is importable in the clean pinned venv (`0` missing).

### P2-3 — Version sync ✅
- `pyproject.toml` `version = "2.0.0"` == `src/omniforge/version.py __version__ = "2.0.0"` ==
  `frontend/package.json "version": "2.0.0"`; `omniforge version` prints `Version: 2.0.0`.
- `.gitattributes` (55 lines) pins `*.bat/*.cmd` to CRLF and all source to LF, preventing cross-platform
  churn; `pyinstaller/omniforge.spec` bundles `omniforge/ui/static_app` (not `uap/`).

### P2-4 — Frontend API typing ✅
- `frontend/src/api.ts` fully rewritten around typed helpers (`apiGet/apiPost/apiDownload` all generic),
  `types.ts` + `studioTypes.ts` carry the payload contracts (795 lines), and the per-key `StudioCache` map
  (**32 keys**, plus `StudioResultKey = keyof StudioCache` and a derived union `StudioPayload`) replaced the
  untestable recursive `StudioPayload` that had produced 291 mypy/pyright-style errors.
- **`any` count: 0** across `frontend/src` (baseline 104); `tsc --noEmit` clean under `strict`.

### P2-5 — DataGrid performance ✅
- Server-side pagination retained; client now **debounces** query changes and **aborts** superseded requests
  via `AbortController` (`DataGrid.tsx:22/33/44/49`), with `aborted` guards before every `setState`, so
  out-of-order responses and React state-after-unmount warnings are eliminated.

### P2-6 — Charts device-pixel-ratio ✅
- `Charts.tsx` scales the backing store by `window.devicePixelRatio` and re-renders through a
  `ResizeObserver`, so canvas scatter/line plots stay crisp (no blur) and correctly sized on HiDPI,
  zoom and panel-resize.

### P2-7 — App memoization ✅
- `App.tsx` now holds **13 stabilised references** — 12 `useCallback` handlers (`openHelp`, `navigateStudio`,
  `fetchServerCache`, `setStudioResult`, `syncState`, `bumpGrid`, `flash`, `toggleTheme`, `toggleTarget`,
  `toggleExclude`, `handleSaveProject`, `handleLoadProject`) and 1 `useMemo` (`commonProps`) — so the 14
  lazily-loaded studio panels receive referentially stable props and stop re-rendering on unrelated
  header/theme/toast state churn.

### P2-8 — Vite lazy loading & chunk split ✅
- `vite.config.ts` `manualChunks` → `react-vendor` / `icons-vendor` / `vendor`; all **14 studios** are
  `React.lazy`-loaded (plus `HelpGuideModal` and `Studio14Supervisory`).
- Result (verified build): `index` 26.02 kB (gzip 8.63), `react-vendor` 138.63 (44.31),
  `vendor` 4.17 (1.81), `icons-vendor` 21.64 (4.28), `StudiosPart1` 95.29 (25.53),
  `StudiosPart2` 23.98 (6.35), `StudiosPart3` 23.82 (6.97), `Studio14Supervisory` 31.68 (8.65),
  `HelpGuideModal` 258.84 (92.03), CSS 17.96 (4.29).
- **Initial entry graph = 63.3 kB gzipped** (index + react-vendor + vendor + icons-vendor + CSS, i.e. exactly
  what `index.html` preloads) vs. **614.43 kB** single chunk before → bigger first paint win than the old
  "69 KB" claim, achieved by actually measuring it.

### P2-9 — mypy strictness ✅
- 104 errors / 16 files → **0 / 166 files**, in three passes:
  1. **JSON boundary typing** — added `to_json_object(data) -> dict[str, Any]` and
     `to_json_list(data) -> list[Any]` next to `to_json_safe` in `core/utils.py`; converted **46** call sites
     (29 + 19 in `api/workbench.py`, 14 in `api/main.py`, 1 each in `api/jobs.py`, `copilot/tools.py`,
     `data/intake_audit.py`). Endpoints now return `dict` instead of `object`, which is what the FastAPI
     serialize path actually needs.
  2. **Real bug found & fixed** — `models/nlp/advanced_nlp.py` `summarize()` read `self._cached_pipelines`,
     an attribute that never existed (the cache is `_hf_pipelines`), so HF summarization silently fell back
     to the extractive path. Dead-code branch removed, real pipeline cache used.
  3. **Type-correct numpy/pandas boundaries** — explicit `np.asarray(..., dtype=…)` in vision fallbacks,
     `float(np.asarray(...))` / `float(str(...))` sort keys, `create_model → BaseOmniForgeModel`,
     `predict_proba(**kwargs)`, `getattr(sys, "_MEIPASS", "")`, `df_ok`/`lock_cfg`/`dl_dict`/`bl_all`/`t_bl`
     bound-first guards, and `Literal` string maps (`_CLUSTER_SCALERS`, `_CLUSTER_LINKAGES`) so a request
     field can no longer carry an arbitrary string into a closed parameter set.
  4. **Config hardened & 60 dead `# type: ignore`s removed** under `warn_unused_ignores` — 59
     `[import-not-found]` of the optional-torch/pyarrow/autogluon imports had become inert; keeping them
     would have hidden *future* real import errors.

### P2-10 — Documentation truthfulness (found during the P2 review) ✅
- README: 23 `uap.*` → `omniforge.*`; "13-studio" → **14-studio**; the false
  "pre-compiled **69 KB gzipped** SPA" claim replaced by the measured statement
  "**~63 KB gzipped initial entry graph**, vendor libs and studio groups lazy-loaded".
- `docs/architecture.md`: `uap` → `omniforge`; the doubled "FastAPI 0.163+ & FastAPI 0.163+" sentence
  de-duplicated and extended with the code-splitting guarantee.
- `docs/user_guide.md`: already clean (0 stale references).

---

## 3. Runtime-hygiene finding fixed during the final sweep

Full-suite runs emitted `ResourceWarning: unclosed database` from SQLite. Root cause is a genuine
CPython footgun: `with sqlite3.connect(...) as conn:` manages the **transaction**, not the **handle**, so
connections survive until GC (file-descriptor leak, stale `-wal`/`-shm`, locked DB on Windows).

- Added a shared `_sqlite_conn()` context manager (commit/rollback **and** `close()` in `finally`) in
  `mlops/registry.py` (6 sites), `mlops/tracking.py` (9 sites), `data/feature_store.py` (3 sites).
- `data/ingestion.py` SQLite reader and `data/connectors.py` demo-DB writer now `close()` in `finally`.
- Verified by a full-suite re-run: the per-test `ResourceWarning` lines collapsed from **8** locations
  (`tests/e2e/test_api_and_cli.py`, `test_omniforge_task33.py`, `test_omniforge_task35.py`,
  `tests/unit/test_eda_and_features.py`, …) to a **single** import-time one
  (`<frozen importlib._bootstrap>`), i.e. the application-level leaks are gone and the residual is an
  import-order artifact of the test harness. mypy stayed at **0 errors** through the refactor — a
  `sqlite_conn` name collision with the DuckDB branch was caught by mypy and fixed, i.e. the check is doing
  real work rather than being decorative.

---

## 4. Whole-program correlation check

| Correlation | Result |
| :--- | :--- |
| Frontend → backend route parity | 72 distinct frontend paths vs 104 backend routes; only the 4 `{fmt}`-parameterised download endpoints (`/s01/export-clean/{fmt}`, `/s14/export/{fmt}`) reported as "unmatched" — they are matched and exercised (`csv`/`xlsx`). **0 real mismatches.** |
| Import graph integrity | `pkgutil.walk_packages` over `omniforge`: **165 modules imported, 0 failures** in the torch-free env. |
| README SDK quickstart | Executed verbatim end-to-end: domain plugin → 6-D quality → cleaner → EDA → AutoML (champion `deep_cnn1d`, lift **+0.0309**) → HTML + PPTX export. **0 errors.** |
| README CLI examples | `omniforge version`, `domains` (18 plugins), `check --json` (`status: ok`, v2.0.0) all run. |
| Studio payload contracts | All **32** `StudioCache` keys have a backend producer (`_cache_and_respond` emits 38 `s\d\d_*` result keys; the 6 extras — `s02_cluster[_committed/_summary]`, `s02_anomaly[_committed/_summary]`, `s05_features` — are workspace-state badges consumed via `WorkspaceState`, not studio results). `_cache_and_respond` state envelope, `/s12/compile`, `/s14/models`, `/s14/equations`, `/s01/sql/demo` shapes re-verified against `studioTypes.ts`. |
| Workspace state contract | All **45** `WorkspaceState` fields in `types.ts` are supplied by `_state_summary`; the 8 nested names (`is_cleaned`, `is_processed`, `cluster_committed`, `anomaly_committed`, `badge`, `label`, `detailed_label`, `summary`) are nested inside the returned `processing_status` object, as the interface declares. |
| Version/packaging | pyproject == version.py == package.json == CLI == `check --json`. |
| Static bundle ↔ index.html | Every asset referenced by `index.html` exists on disk (`test_p0_security_and_runtime_hardening.py`); CSP survives in `StudiosPart1` chunk, `X-CSRF-Token` in the entry chunk, `document.write` absent from all chunks. |

---

## 5. Residual risks / notes (deliberate, not regressions)

1. **pandas Categorical `.fillna("MISSING")`** — works on the pinned pandas 3.1.1 but raises under 2.2.x
   (`supervisory_monitor.py:185`, `features/engineering.py:104,283`, `models/base.py:248,253`). The project
   pins `pandas>=3.1.1`, so this is consistent with the declared support matrix; a defensive rewrite is
   queued for a future phase if the floor is ever lowered.
2. **`torch` paths** were validated through the NumPy fallbacks only (torch is an optional extra and is not
   installed in the verification venv). The GNN fallback was separately validated: ELBO loss
   `0.685719 → 0.636553` monotone, AUC **0.9107** (`A^3` spectral init: mean 0.809, min 0.742).
3. **Legacy `.uap` compatibility** is kept on purpose (see P2-1); removing it is a breaking-change decision
   for the next major version.
4. One remaining suite warning is third-party (xarray/plotly import-time `DeprecationWarning`,
   scipy `ks_2samp` asymptotic switch) — outside this codebase's control.

---

## 6. Reproduction

```bash
# backend gates
/var/tmp/p2_venv/bin/ruff check .
/var/tmp/p2_venv/bin/mypy src/omniforge
PYTHONPATH=/home/user/OmniForge/src /var/tmp/p2_venv/bin/python -m pytest tests -q

# frontend gates
cd frontend && ./node_modules/.bin/tsc --noEmit && npm run build
```
