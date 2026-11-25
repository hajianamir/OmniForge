# OmniForge packaging, upgrade and troubleshooting reference

> Reverified at the code revision in [SKILL.md](../SKILL.md). Inspect the
> actual environment and preserve user data before installation or repair.

## Packs, tiers and Python extras

`src/omniforge/capabilities.py` is the stdlib-only installer catalogue. It
lists five core and seven optional packs; pip extras in `pyproject.toml` are a
separate interface and do not match these pack sets exactly.

| Pack | Catalogue purpose |
|---|---|
| `core-runtime` | Scientific stack, settings, CLI and basic analytics |
| `api-server` | FastAPI, Uvicorn, authentication and the workbench server |
| `statistics` | statsmodels-based statistics and forecasting |
| `data-io` | Arrow, DuckDB, SQLAlchemy and related data I/O |
| `reporting` | Report and spreadsheet dependencies |
| `boosting`, `deep` | Optional native boosted-tree and PyTorch backends |
| `vision`, `nlp`, `foundation` | Optional vision, NLP and foundation-model dependencies |
| `automl`, `responsible` | Optional AutoML, explainability and carbon backends |

`TIERS` selects `simple` = core-runtime + statistics + data-io + reporting;
`advanced` adds api-server; `everything` adds all optional packs. These are
preselected dependencies, not studio access controls or verified capability
promises. In particular, **Simple does not select FastAPI/Uvicorn**, although
the React workbench requires the server. `resolve_selection()` unions the
requested pack IDs but does not enforce an explicit dependency graph; for
example, the catalogue describes foundation as requiring deep. Test clean
installs and actual optional backend behavior for the chosen Python/platform.
The catalogue's comments about a capabilities API and in-app install buttons
are not evidence of implemented UI/API integration.

## Two lock systems and the selection mismatch

- `requirements.in` is floating runtime intent; `requirements.txt` includes
  `requirements.lock`, a hash-pinned universal lock generated for Python 3.12.
  `requirements-build.in`, `requirements-build.lock` and
  `requirements-build.txt` cover build tools separately. CI installs the
  runtime lock with `pip install --require-hashes -r requirements.lock`.
- `scripts/build_locks.py` compiles `locks/{simple,advanced,everything}.lock`,
  `pack-*.lock`, and `dev.lock` for Windows/Python 3.13. Each optional pack
  lock is built from **Advanced plus that one pack**; `dev.lock` is the
  developer-tool set, not a full tier. Regenerate with
  `python scripts/build_locks.py` or compare with `--check` when `uv` and the
  required package sources are available. Pin the tool version used by the
  relevant release workflow and review generated diffs.

**The current provisioner does not select an arbitrary union lock.**
`installer/provision.py::_lock_for` uses the list explicitly passed in
`--packs`: exactly one optional request prefers `pack-<id>.lock`, otherwise it
prefers `<tier>.lock`, with `everything.lock` as a last resort. Examples
with the committed locks present:

| Requested tier / extra packs | Lock actually selected | Consequence |
|---|---|---|
| `advanced` + `boosting,vision` | `advanced.lock` | Neither optional pack is guaranteed to be installed. |
| `everything` + `vision` | `pack-vision.lock` | The selected lock is Advanced + vision, not Everything. |
| `simple` + `vision` | `pack-vision.lock` | The lock includes Advanced, not just Simple + vision. |

With a lock present, `install_packs()` calls `uv pip sync --require-hashes`:
it syncs to that lock and can remove packages already in the environment.
The planned requirements/pack list and `--check-only` do not prove the lock
matches the requested union. After installation `verify()` import-probes the
selected imports and may fail/rollback, but passing probes or having a receipt
is not proof of exact versions, model weights, or end-to-end functionality.
A missing lock triggers a non-release direct-resolution fallback. Fix lock
selection and add real clean-install/upgrade tests before promising arbitrary
pack combinations.

The Inno wizard's `GetTierArgument` returns Simple or Advanced based on the
API component and passes selected optional checkboxes as `--packs`; it does not
pass `everything` as the tier. Its wheelhouse option passes `--wheels`, not
`--offline`. Supplying a wheelhouse alone allows network access for missing
packages. For a genuinely offline provisioner run, pass **`--offline`** as
well, provide all required compatible wheels, and test in a network-isolated
clean environment. Do not infer offline behavior from installer comments.

## Build and deployment map

| Area | Source | Practical scope |
|---|---|---|
| SPA | `frontend/`, `src/omniforge/ui/static_app/`, `pyproject.toml` | Build with `cd frontend && npm ci && npm run build`; commit the static bundle so packaged runtimes need not build it with Node. CI checks it for drift. |
| PyInstaller | `packaging/build_exe.py`, `packaging/pyinstaller/*.spec`, `packaging/entry_{desktop,cli}.py` | Frozen executable/launcher build; see also `packaging/inno_setup/omniforge_installer.iss`. |
| Tiered installer | `installer/omniforge_installer_tiers.iss`, `installer/provision.py` | Wizard, private venv, plans, import verification and receipt; see lock caveat above. |
| Bootstrap/ISS | `scripts/build_bootstrap.py`, `scripts/make_launcher_shim.py`, `scripts/check_iss.py` | Stage runtime/shim and statically validate the installer; a full Windows build needs the staged payload and Inno Setup. |
| SBOM | `src/omniforge/core/sbom.py`, `scripts/generate_sbom.py`, `omniforge sbom` | Inspect generated provenance; `--verify` checks hash pinning, not runtime security. |
| Containers | `docker/Dockerfile`, `docker/docker-compose.yml`, `docker/entrypoint.sh` | Non-root Python 3.12 image; API, UI, monitor worker and CLI modes; API defaults to one worker. |

The Docker build pins the Python base image and installs hash-verified
Linux amd64 and arm64 dependency locks. The project wheel builds offline.
Regenerate the locks with `bash scripts/build_docker_locks.sh`.

The Compose file starts separate API and UI processes on ports 8000 and 8501
and mounts a shared workspace volume at `/var/lib/omniforge`. Registry and
tracking defaults derive from `OMNIFORGE_PATHS__WORKSPACE_DIR`, and tenant
stores live under its `tenants` directory. Explicit path overrides keep their
configured location. Workbench sessions remain in memory per process.
`scripts/smoke_docker_registry.py` verifies registry records and signed artifacts
after container replacement. Back up the database, artifacts, and signing keys
together before moving an existing workspace.

`.github/workflows/windows-installer.yml` is a separate manual/tag workflow,
not an ordinary PR check. It builds the installer and exercises Simple and
Advanced provisioning; it does not prove all checkbox combinations or offline
upgrades. See [testing](testing.md) for the PR CI jobs.

## Safe upgrade and troubleshooting

1. Inventory workspaces, per-tenant DBs, feature stores, artifacts, audit logs,
   project archives, signing/encryption keys and pseudonym salts. Securely
   back them up and test recovery before replacing images or environments.
   Never assume `.omniforge_workspace/` is disposable or delete/rotate keys
   merely to bypass a signature failure.
2. Review version/schema/API changes, package constraints, lock selection and
   frontend bundle; update code and compatibility tests together. Recompile
   the relevant lockfiles and test clean provisioning on the intended platform.
3. Test project and model save/load with the original keys, the supported
   dependencies and the restricted deserializer. A signature failure may mean
   corruption, a wrong key, or an untrusted artifact; investigate rather than
   switching off verification or blindly adding unpickler allowlist entries.
4. For multi-worker/container deployments, explicitly plan workspace affinity
   or shared state, a reliable shared jobs DB, storage mounts and network
   security. A shared job-status database alone does not share live workspaces.
5. Diagnose 413/429 using the configured body limit and rate limiter; account
   for proxy IP aggregation and Redis fallback before changing limits. If a
   model or cached result seems inconsistent after a target/split change,
   invalidate affected outputs and retrain while preserving needed user data.
   Review `src/omniforge/ui/state.py`—those mutations do not currently
   invalidate all downstream state.
6. If registry promotion is blocked, inspect the fairness result and policy.
   Failed fairness blocks Production without an override; missing fairness
   blocks only when the gate is required. Overrides should follow the user's
   governance process, not be inserted as an automatic workaround.
